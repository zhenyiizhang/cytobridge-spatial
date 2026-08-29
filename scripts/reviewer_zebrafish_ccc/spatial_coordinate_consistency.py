#!/usr/bin/env python3
"""Visualize and test spatial-coordinate consistency for selected LR axes.

The workflow compares CytoBridge LR-compatible edge scores with reconstructed
COMMOT cell-cell flow on one frozen aligned coordinate system.  It deliberately
does not compare raw score units.  Instead, it selects positive top-fraction
edges within each method, turns their edge midpoints into unit-mass spatial
fields on a common grid, and measures field overlap, high-density-region (HDR)
overlap, and one-to-one midpoint matching.

The score-permutation null keeps each method's positive edge support fixed and
uses an audited adaptive hierarchy of type/distance/LR-activity strata. Consequently,
the null preserves tissue geometry, edge counts, local-distance structure, and
most LR-expression geography while breaking the association between score rank
and a particular edge location.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from scipy.spatial import cKDTree
from scipy.stats import spearmanr


CB_SCORE = "cytobridge_attention_lr_flow"
COMMOT_SCORE = "commot_flow"
PRIMARY_COMPONENTS = (
    ("lr_only", "LR activity only", "mean_scaled_lr_activity"),
    ("attention_only", "Attention only\n(on LR-positive edges)", "mean_attention_abs"),
    ("attention_lr", "Attention × LR", "cytobridge_attention_lr_flow"),
    ("exact_message_lr", "Exact message × LR", "cytobridge_exact_message_lr_flow"),
)


@dataclass(frozen=True)
class Grid:
    x_edges: np.ndarray
    y_edges: np.ndarray
    x_centers: np.ndarray
    y_centers: np.ndarray
    xx: np.ndarray
    yy: np.ndarray
    tissue_mask: np.ndarray
    dx: float
    dy: float


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--bundle-dir", required=True, type=Path)
    result.add_argument("--coordinates-csv", required=True, type=Path)
    result.add_argument("--output-dir", required=True, type=Path)
    result.add_argument("--graph-cutoff", type=float)
    result.add_argument("--primary-top-fraction", type=float, default=0.20)
    result.add_argument("--primary-scale", type=float, default=0.50)
    result.add_argument(
        "--top-fractions", default="0.05,0.10,0.20,0.30",
        help="Comma-separated positive-edge fractions for sensitivity analysis.",
    )
    result.add_argument(
        "--scale-factors", default="0.25,0.50,1.00",
        help="Comma-separated bandwidth/radius multiples of the graph cutoff.",
    )
    result.add_argument("--permutations", type=int, default=200)
    result.add_argument("--seed", type=int, default=20260722)
    result.add_argument("--grid-step-factor", type=float, default=0.125)
    result.add_argument("--tissue-mask-radius-factor", type=float, default=0.50)
    result.add_argument(
        "--permutation-bins",
        type=int,
        default=5,
        help="Quantile bins used for distance and LR activity in the adaptive null.",
    )
    result.add_argument("--min-permutation-stratum", type=int, default=10)
    result.add_argument(
        "--max-global-fallback-fraction",
        type=float,
        default=0.05,
        help="Fail if any null assignment silently falls back to one global pool above this fraction.",
    )
    result.add_argument("--overwrite", action="store_true")
    return result


def _parse_floats(value: str) -> tuple[float, ...]:
    result = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not result or any(not np.isfinite(item) or item <= 0 for item in result):
        raise ValueError(f"Expected positive comma-separated numbers, got {value!r}")
    return result


def _require(frame: pd.DataFrame, columns: Iterable[str], source: Path | str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{source} lacks required columns: {missing}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record(path: Path, *, relative_to: Path | None = None) -> dict[str, object]:
    display = path.relative_to(relative_to) if relative_to is not None else path
    return {
        "path": str(display),
        "bytes": int(path.stat().st_size),
        "sha256": _sha256(path),
    }


def _prepare_output(path: Path, overwrite: bool) -> Path:
    path = path.expanduser().resolve()
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory is not empty: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _read_cutoff(bundle: Path, explicit: float | None) -> float:
    if explicit is not None:
        cutoff = float(explicit)
    else:
        path = bundle / "manifests" / "biological_consistency_manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        cutoff = float(
            manifest["display"]["spatial_midpoint_colocalization"] [
                "frozen_spatial_graph_cutoff"
            ]
        )
    if not np.isfinite(cutoff) or cutoff <= 0:
        raise ValueError(f"Invalid graph cutoff: {cutoff}")
    return cutoff


def load_inputs(
    bundle: Path, coordinates_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tables = bundle / "tables"
    selection_path = tables / "biological_example_selection.csv"
    cb_path = tables / "selected_cytobridge_cell_edges.csv.gz"
    commot_path = tables / "selected_commot_cell_flows.csv.gz"
    for path in (selection_path, cb_path, commot_path, coordinates_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    selection = pd.read_csv(selection_path)
    cb = pd.read_csv(cb_path)
    commot = pd.read_csv(commot_path)
    coordinates = pd.read_csv(coordinates_path)
    _require(
        selection,
        ["example_id", "stage", "stage_label", "ligand", "receptor", "selection_family"],
        selection_path,
    )
    _require(
        cb,
        [
            "example_id", "stage", "source_index", "target_index", "sender_type",
            "receiver_type", "mean_scaled_lr_activity", "mean_attention_abs",
            "mean_exact_message", "mean_spatial_distance", CB_SCORE,
            "cytobridge_exact_message_lr_flow",
        ],
        cb_path,
    )
    _require(
        commot,
        [
            "example_id", "stage", "source_cell_id", "target_cell_id", "sender_type",
            "receiver_type", COMMOT_SCORE,
        ],
        commot_path,
    )
    _require(
        coordinates,
        ["cell_index_global", "cell_id", "stage", "stage_label", "cell_type", "x", "y"],
        coordinates_path,
    )
    if coordinates["cell_index_global"].duplicated().any():
        raise ValueError("coordinates cell_index_global values are not unique")
    if coordinates["cell_id"].astype(str).duplicated().any():
        raise ValueError("coordinates cell_id values are not unique")
    if selection["example_id"].duplicated().any():
        raise ValueError("selection contains duplicate example_id rows")
    selected = set(selection["example_id"].astype(str))
    if not selected <= set(cb["example_id"].astype(str)):
        raise ValueError("Not every selected example appears in the CytoBridge edge table")
    if not selected <= set(commot["example_id"].astype(str)):
        raise ValueError("Not every selected example appears in the COMMOT flow table")
    return selection.reset_index(drop=True), cb, commot, coordinates


def attach_coordinates(
    cb: pd.DataFrame, commot: pd.DataFrame, coordinates: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_index = coordinates.set_index("cell_index_global")
    by_id = coordinates.assign(cell_id=coordinates["cell_id"].astype(str)).set_index("cell_id")
    result_cb = cb.copy()
    for prefix, column in (("source", "source_index"), ("target", "target_index")):
        indices = pd.to_numeric(result_cb[column], errors="raise").astype(int)
        if not set(indices) <= set(by_index.index):
            raise ValueError(f"CytoBridge {column} values do not all map to coordinates")
        mapped = by_index.loc[indices, ["x", "y", "stage"]].reset_index(drop=True)
        result_cb[f"{prefix}_x"] = mapped["x"].to_numpy(float)
        result_cb[f"{prefix}_y"] = mapped["y"].to_numpy(float)
        if not np.allclose(mapped["stage"].to_numpy(float), result_cb["stage"].to_numpy(float)):
            raise ValueError(f"CytoBridge {prefix} coordinate stage mismatch")

    result_commot = commot.copy()
    for prefix, column in (("source", "source_cell_id"), ("target", "target_cell_id")):
        ids = result_commot[column].astype(str)
        if not set(ids) <= set(by_id.index):
            raise ValueError(f"COMMOT {column} values do not all map to coordinates")
        mapped = by_id.loc[ids, ["x", "y", "stage"]].reset_index(drop=True)
        result_commot[f"{prefix}_x"] = mapped["x"].to_numpy(float)
        result_commot[f"{prefix}_y"] = mapped["y"].to_numpy(float)
        if not np.allclose(
            mapped["stage"].to_numpy(float), result_commot["stage"].to_numpy(float)
        ):
            raise ValueError(f"COMMOT {prefix} coordinate stage mismatch")
    return result_cb, result_commot


def positive_nonself(frame: pd.DataFrame, score: str) -> pd.DataFrame:
    values = pd.to_numeric(frame[score], errors="coerce")
    if {"source_index", "target_index"} <= set(frame.columns):
        nonself = frame["source_index"].ne(frame["target_index"])
    else:
        nonself = frame["source_cell_id"].astype(str).ne(frame["target_cell_id"].astype(str))
    return frame.loc[np.isfinite(values) & values.gt(0) & nonself].copy()


def edge_midpoints(frame: pd.DataFrame) -> np.ndarray:
    return np.column_stack(
        [
            (frame["source_x"].to_numpy(float) + frame["target_x"].to_numpy(float)) / 2,
            (frame["source_y"].to_numpy(float) + frame["target_y"].to_numpy(float)) / 2,
        ]
    )


def select_top(frame: pd.DataFrame, score: str, fraction: float) -> pd.DataFrame:
    if not 0 < fraction <= 1:
        raise ValueError("top fraction must be in (0, 1]")
    supported = positive_nonself(frame, score)
    if supported.empty:
        return supported
    n = max(1, int(math.ceil(fraction * len(supported))))
    return supported.sort_values(
        [score, "source_x", "source_y", "target_x", "target_y"],
        ascending=[False, True, True, True, True],
        kind="mergesort",
    ).head(n)


def make_grid(
    stage_coordinates: pd.DataFrame,
    *,
    cutoff: float,
    step_factor: float,
    mask_radius_factor: float,
) -> Grid:
    if stage_coordinates.empty:
        raise ValueError("Cannot create a spatial grid without stage cells")
    step = cutoff * float(step_factor)
    if step <= 0:
        raise ValueError("grid step must be positive")
    padding = cutoff
    x = stage_coordinates["x"].to_numpy(float)
    y = stage_coordinates["y"].to_numpy(float)
    x_edges = np.arange(x.min() - padding, x.max() + padding + step, step)
    y_edges = np.arange(y.min() - padding, y.max() + padding + step, step)
    if len(x_edges) < 3 or len(y_edges) < 3:
        raise ValueError("Spatial grid has fewer than two cells on an axis")
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    xx, yy = np.meshgrid(x_centers, y_centers)
    points = np.column_stack([xx.ravel(), yy.ravel()])
    nearest = cKDTree(np.column_stack([x, y])).query(points, k=1)[0]
    tissue_mask = (nearest <= cutoff * float(mask_radius_factor)).reshape(xx.shape)
    return Grid(
        x_edges=x_edges,
        y_edges=y_edges,
        x_centers=x_centers,
        y_centers=y_centers,
        xx=xx,
        yy=yy,
        tissue_mask=tissue_mask,
        dx=float(np.diff(x_edges).mean()),
        dy=float(np.diff(y_edges).mean()),
    )


def spatial_field(points: np.ndarray, grid: Grid, bandwidth: float) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) == 0:
        raise ValueError("spatial_field requires at least one two-dimensional point")
    histogram, _, _ = np.histogram2d(
        points[:, 1], points[:, 0], bins=[grid.y_edges, grid.x_edges]
    )
    smoothed = gaussian_filter(
        histogram,
        sigma=(float(bandwidth) / grid.dy, float(bandwidth) / grid.dx),
        mode="constant",
        truncate=4.0,
    )
    smoothed[~grid.tissue_mask] = 0.0
    total = float(smoothed.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Spatial field has zero mass inside the tissue mask")
    return smoothed / total


def hdr_mask(field: np.ndarray, mass: float) -> tuple[np.ndarray, float]:
    if not 0 < mass <= 1:
        raise ValueError("HDR mass must be in (0, 1]")
    values = np.asarray(field, dtype=float)
    positive = values[np.isfinite(values) & (values > 0)]
    if positive.size == 0:
        return np.zeros_like(values, dtype=bool), float("nan")
    ordered = np.sort(positive)[::-1]
    cumulative = np.cumsum(ordered)
    threshold = float(ordered[min(np.searchsorted(cumulative, mass), len(ordered) - 1)])
    return values >= threshold, threshold


def field_metrics(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.shape != right.shape:
        raise ValueError("Spatial fields must share a grid")
    left80, _ = hdr_mask(left, 0.80)
    right80, _ = hdr_mask(right, 0.80)
    denominator = int(left80.sum() + right80.sum())
    return {
        "field_overlap_ovl": float(np.minimum(left, right).sum()),
        "hdr80_dice": (
            float(2 * np.logical_and(left80, right80).sum() / denominator)
            if denominator
            else float("nan")
        ),
        "hdr80_shared_grid_cells": int(np.logical_and(left80, right80).sum()),
        "hdr80_cytobridge_grid_cells": int(left80.sum()),
        "hdr80_commot_grid_cells": int(right80.sum()),
    }


def one_to_one_match_f1(left: np.ndarray, right: np.ndarray, radius: float) -> dict[str, float | int]:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if len(left) == 0 or len(right) == 0:
        return {"one_to_one_matches": 0, "spatial_match_f1": float("nan")}
    neighbors = cKDTree(right).query_ball_point(left, r=float(radius))
    rows: list[int] = []
    columns: list[int] = []
    for row, local in enumerate(neighbors):
        rows.extend([row] * len(local))
        columns.extend(local)
    if not rows:
        matches = 0
    else:
        graph = csr_matrix(
            (np.ones(len(rows), dtype=np.int8), (rows, columns)),
            shape=(len(left), len(right)),
        )
        assignment = maximum_bipartite_matching(graph, perm_type="column")
        matches = int(np.sum(assignment >= 0))
    return {
        "one_to_one_matches": matches,
        "spatial_match_f1": float(2 * matches / (len(left) + len(right))),
    }


def _safe_qcut(values: pd.Series, q: int = 10) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.nunique(dropna=True) < 2:
        return pd.Series("0", index=values.index, dtype=str)
    ranked = numeric.rank(method="average")
    bins = pd.qcut(ranked, q=min(q, int(ranked.nunique())), labels=False, duplicates="drop")
    return bins.fillna(-1).astype(int).astype(str)


def permutation_strata_assignment(
    frame: pd.DataFrame, *, method: str, min_size: int, bins: int = 4
) -> pd.DataFrame:
    """Assign auditable strata with hierarchical, covariate-preserving coarsening.

    Exact sender→receiver type is retained when that fine stratum is large
    enough. Sparse type strata fall back to distance×LR bins for CytoBridge or
    distance bins for COMMOT; only still-sparse bins are coarsened further.
    """
    if min_size < 2:
        raise ValueError("min_size must be at least 2")
    if bins < 2:
        raise ValueError("bins must be at least 2")
    distance = np.hypot(
        frame["source_x"].to_numpy(float) - frame["target_x"].to_numpy(float),
        frame["source_y"].to_numpy(float) - frame["target_y"].to_numpy(float),
    )
    distance_bin = _safe_qcut(pd.Series(distance, index=frame.index), q=bins)
    pair = frame["sender_type"].astype(str) + "→" + frame["receiver_type"].astype(str)
    if method == "cytobridge":
        lr_bin = _safe_qcut(frame["mean_scaled_lr_activity"], q=bins)
        covariate = "d" + distance_bin + "|lr" + lr_bin
    elif method in {"cytobridge_no_lr", "commot"}:
        lr_bin = pd.Series("NA", index=frame.index, dtype=str)
        covariate = "d" + distance_bin
    else:
        raise ValueError(f"Unknown method: {method}")
    fine = pair + "|" + covariate
    fine_ok = fine.map(fine.value_counts()).ge(int(min_size))
    distance_only = "d" + distance_bin
    remaining_after_fine = ~fine_ok
    covariate_counts = covariate.loc[remaining_after_fine].value_counts()
    covariate_ok = covariate.map(covariate_counts).fillna(0).ge(int(min_size))
    remaining_after_covariate = remaining_after_fine & ~covariate_ok
    distance_counts = distance_only.loc[remaining_after_covariate].value_counts()
    distance_ok = distance_only.map(distance_counts).fillna(0).ge(int(min_size))

    labels = pd.Series(index=frame.index, dtype=str)
    levels = pd.Series(index=frame.index, dtype=str)
    labels.loc[fine_ok] = "fine_type_covariate|" + fine.loc[fine_ok]
    levels.loc[fine_ok] = "fine_type_covariate"
    use_covariate = remaining_after_fine & covariate_ok
    labels.loc[use_covariate] = "pooled_covariate|" + covariate.loc[use_covariate]
    levels.loc[use_covariate] = "pooled_covariate"
    use_distance = remaining_after_covariate & distance_ok
    labels.loc[use_distance] = "pooled_distance|" + distance_only.loc[use_distance]
    levels.loc[use_distance] = "pooled_distance"
    use_global = labels.isna()
    labels.loc[use_global] = "global"
    levels.loc[use_global] = "global"
    return pd.DataFrame(
        {
            "permutation_stratum": labels.astype(str),
            "permutation_level": levels.astype(str),
            "distance_bin": distance_bin.astype(str),
            "lr_activity_bin": lr_bin.astype(str),
            "edge_distance": distance,
            "lr_activity": (
                pd.to_numeric(frame["mean_scaled_lr_activity"], errors="coerce")
                if method.startswith("cytobridge")
                else np.nan
            ),
        },
        index=frame.index,
    )


def permutation_strata(
    frame: pd.DataFrame, *, method: str, min_size: int, bins: int = 4
) -> pd.Series:
    return permutation_strata_assignment(
        frame, method=method, min_size=min_size, bins=bins
    )["permutation_stratum"]


def permutation_strata_audit(
    selection: pd.DataFrame,
    cb: pd.DataFrame,
    commot: pd.DataFrame,
    *,
    min_size: int,
    bins: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for selected in selection.itertuples(index=False):
        example_id = str(selected.example_id)
        cb_primary, commot_primary = example_supports(example_id, cb, commot)
        cb_component = positive_nonself(
            cb.loc[cb["example_id"].astype(str).eq(example_id)].copy(),
            "mean_scaled_lr_activity",
        ).reset_index(drop=True)
        for analysis_id, method, frame in (
            ("primary_score_null", "cytobridge", cb_primary),
            ("primary_score_null", "commot", commot_primary),
            ("component_modifier_null", "cytobridge", cb_component),
            ("component_score_null", "cytobridge_no_lr", cb_component),
        ):
            assignment = permutation_strata_assignment(
                frame, method=method, min_size=min_size, bins=bins
            )
            stratum_sizes = assignment["permutation_stratum"].value_counts()
            assignment_hash = hashlib.sha256(
                "\n".join(assignment["permutation_stratum"].astype(str)).encode("utf-8")
            ).hexdigest()
            per_stratum = assignment.groupby("permutation_stratum", sort=False).agg(
                stratum_size=("permutation_stratum", "size"),
                distance_span=("edge_distance", lambda values: float(values.max() - values.min())),
                lr_activity_span=("lr_activity", lambda values: float(values.max() - values.min())),
            )
            for level, local in assignment.groupby("permutation_level", sort=False):
                local_sizes = local["permutation_stratum"].map(stratum_sizes)
                local_strata = per_stratum.loc[local["permutation_stratum"].unique()]
                rows.append(
                    {
                        "example_id": example_id,
                        "stage": float(selected.stage),
                        "stage_label": str(selected.stage_label),
                        "ligand": str(selected.ligand),
                        "receptor": str(selected.receptor),
                        "analysis": analysis_id,
                        "method": method,
                        "coarsening_level": str(level),
                        "n_edges": int(len(local)),
                        "fraction_edges": float(len(local) / len(frame)),
                        "n_strata": int(local["permutation_stratum"].nunique()),
                        "min_realized_stratum_size": int(local_sizes.min()),
                        "median_realized_stratum_size": float(local_sizes.median()),
                        "max_realized_stratum_size": int(local_sizes.max()),
                        "movable_edge_fraction_overall": float(
                            assignment["permutation_stratum"].map(stratum_sizes).ge(2).mean()
                        ),
                        "median_within_stratum_distance_span": float(
                            local_strata["distance_span"].median()
                        ),
                        "median_within_stratum_lr_activity_span": (
                            float(local_strata["lr_activity_span"].median())
                            if method.startswith("cytobridge")
                            else float("nan")
                        ),
                        "requested_min_stratum_size": int(min_size),
                        "quantile_bins": int(bins),
                        "assignment_sha256": assignment_hash,
                    }
                )
    return pd.DataFrame(rows)


def permute_within_strata(
    values: np.ndarray, strata: Sequence[str], rng: np.random.Generator
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    labels = np.asarray(strata, dtype=str)
    result = values.copy()
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        result[indices] = values[rng.permutation(indices)]
    return result


def _top_positions(scores: np.ndarray, fraction: float) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    finite = np.flatnonzero(np.isfinite(scores) & (scores > 0))
    if finite.size == 0:
        return finite
    n = max(1, int(math.ceil(fraction * len(finite))))
    order = np.argsort(-scores[finite], kind="mergesort")
    return finite[order[:n]]


def example_supports(
    example_id: str, cb: pd.DataFrame, commot: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    local_cb = positive_nonself(cb.loc[cb["example_id"].astype(str).eq(example_id)], CB_SCORE)
    local_commot = positive_nonself(
        commot.loc[commot["example_id"].astype(str).eq(example_id)], COMMOT_SCORE
    )
    if local_cb.empty or local_commot.empty:
        raise ValueError(f"Example {example_id} lacks positive non-self support")
    return local_cb.reset_index(drop=True), local_commot.reset_index(drop=True)


def evaluate_selection(
    cb_points: np.ndarray,
    commot_points: np.ndarray,
    *,
    grid: Grid,
    bandwidth: float,
    match_radius: float,
) -> dict[str, float | int | np.ndarray]:
    cb_field = spatial_field(cb_points, grid, bandwidth)
    commot_field = spatial_field(commot_points, grid, bandwidth)
    metrics: dict[str, float | int | np.ndarray] = {
        **field_metrics(cb_field, commot_field),
        **one_to_one_match_f1(cb_points, commot_points, match_radius),
        "cytobridge_field": cb_field,
        "commot_field": commot_field,
    }
    return metrics


def sensitivity_and_null(
    selection: pd.DataFrame,
    cb: pd.DataFrame,
    commot: pd.DataFrame,
    coordinates: pd.DataFrame,
    *,
    cutoff: float,
    top_fractions: Sequence[float],
    scale_factors: Sequence[float],
    permutations: int,
    seed: int,
    grid_step_factor: float,
    tissue_mask_radius_factor: float,
    min_stratum: int,
    permutation_bins: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(int(seed))
    rows: list[dict[str, object]] = []
    for selected in selection.itertuples(index=False):
        example_id = str(selected.example_id)
        local_cb, local_commot = example_supports(example_id, cb, commot)
        stage_cells = coordinates.loc[np.isclose(coordinates["stage"], float(selected.stage))]
        grid = make_grid(
            stage_cells,
            cutoff=cutoff,
            step_factor=grid_step_factor,
            mask_radius_factor=tissue_mask_radius_factor,
        )
        cb_all_points = edge_midpoints(local_cb)
        commot_all_points = edge_midpoints(local_commot)
        cb_scores = local_cb[CB_SCORE].to_numpy(float)
        commot_scores = local_commot[COMMOT_SCORE].to_numpy(float)
        cb_strata = permutation_strata(
            local_cb, method="cytobridge", min_size=min_stratum,
            bins=permutation_bins,
        ).to_numpy(str)
        commot_strata = permutation_strata(
            local_commot, method="commot", min_size=min_stratum,
            bins=permutation_bins,
        ).to_numpy(str)
        for fraction in top_fractions:
            cb_observed = _top_positions(cb_scores, fraction)
            commot_observed = _top_positions(commot_scores, fraction)
            permuted_indices = []
            for _ in range(int(permutations)):
                permuted_indices.append(
                    (
                        _top_positions(
                            permute_within_strata(cb_scores, cb_strata, rng), fraction
                        ),
                        _top_positions(
                            permute_within_strata(commot_scores, commot_strata, rng), fraction
                        ),
                    )
                )
            cb_unique_top_sets = len(
                {tuple(np.sort(indices).tolist()) for indices, _ in permuted_indices}
            )
            commot_unique_top_sets = len(
                {tuple(np.sort(indices).tolist()) for _, indices in permuted_indices}
            )
            cb_mean_fraction_changed = float(
                np.mean(
                    [
                        1.0
                        - len(np.intersect1d(cb_observed, indices, assume_unique=False))
                        / max(1, len(cb_observed))
                        for indices, _ in permuted_indices
                    ]
                )
            )
            commot_mean_fraction_changed = float(
                np.mean(
                    [
                        1.0
                        - len(np.intersect1d(commot_observed, indices, assume_unique=False))
                        / max(1, len(commot_observed))
                        for _, indices in permuted_indices
                    ]
                )
            )
            for scale in scale_factors:
                bandwidth = cutoff * float(scale)
                observed = evaluate_selection(
                    cb_all_points[cb_observed],
                    commot_all_points[commot_observed],
                    grid=grid,
                    bandwidth=bandwidth,
                    match_radius=bandwidth,
                )
                null_values: dict[str, list[float]] = {
                    "field_overlap_ovl": [],
                    "hdr80_dice": [],
                    "spatial_match_f1": [],
                }
                for cb_indices, commot_indices in permuted_indices:
                    null = evaluate_selection(
                        cb_all_points[cb_indices],
                        commot_all_points[commot_indices],
                        grid=grid,
                        bandwidth=bandwidth,
                        match_radius=bandwidth,
                    )
                    for metric in null_values:
                        null_values[metric].append(float(null[metric]))
                for metric, values in null_values.items():
                    null_array = np.asarray(values, dtype=float)
                    observed_value = float(observed[metric])
                    finite = null_array[np.isfinite(null_array)]
                    rows.append(
                        {
                            "example_id": example_id,
                            "stage": float(selected.stage),
                            "stage_label": str(selected.stage_label),
                            "ligand": str(selected.ligand),
                            "receptor": str(selected.receptor),
                            "selection_family": str(selected.selection_family),
                            "top_fraction": float(fraction),
                            "scale_factor": float(scale),
                            "bandwidth_and_match_radius": bandwidth,
                            "metric": metric,
                            "observed": observed_value,
                            "null_mean": float(np.mean(finite)),
                            "null_ci_low": float(np.quantile(finite, 0.025)),
                            "null_ci_high": float(np.quantile(finite, 0.975)),
                            "observed_minus_null_mean": observed_value - float(np.mean(finite)),
                            "observed_over_null_mean": (
                                observed_value / float(np.mean(finite))
                                if float(np.mean(finite)) > 0
                                else float("nan")
                            ),
                            "empirical_p_greater_equal": float(
                                (1 + np.sum(finite >= observed_value)) / (1 + len(finite))
                            ),
                            "n_permutations": int(len(finite)),
                            "n_cytobridge_positive_edges": int(len(local_cb)),
                            "n_commot_positive_edges": int(len(local_commot)),
                            "n_cytobridge_top_edges": int(len(cb_observed)),
                            "n_commot_top_edges": int(len(commot_observed)),
                            "n_unique_cytobridge_top_sets": int(cb_unique_top_sets),
                            "n_unique_commot_top_sets": int(commot_unique_top_sets),
                            "cytobridge_mean_top_membership_fraction_changed": cb_mean_fraction_changed,
                            "commot_mean_top_membership_fraction_changed": commot_mean_fraction_changed,
                            "null_rule": (
                                "independent score permutation on fixed positive non-self support; "
                                "adaptive hierarchy retains sender→receiver type when sufficiently "
                                "supported, otherwise coarsens to distance×LR-activity quantile bins "
                                "for CytoBridge or distance bins for COMMOT"
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def primary_metrics_and_fields(
    selection: pd.DataFrame,
    cb: pd.DataFrame,
    commot: pd.DataFrame,
    coordinates: pd.DataFrame,
    *,
    cutoff: float,
    top_fraction: float,
    scale_factor: float,
    grid_step_factor: float,
    tissue_mask_radius_factor: float,
) -> tuple[pd.DataFrame, dict[str, dict[str, object]]]:
    rows: list[dict[str, object]] = []
    payload: dict[str, dict[str, object]] = {}
    for selected in selection.itertuples(index=False):
        example_id = str(selected.example_id)
        local_cb, local_commot = example_supports(example_id, cb, commot)
        cb_top = select_top(local_cb, CB_SCORE, top_fraction)
        commot_top = select_top(local_commot, COMMOT_SCORE, top_fraction)
        stage_cells = coordinates.loc[np.isclose(coordinates["stage"], float(selected.stage))]
        grid = make_grid(
            stage_cells,
            cutoff=cutoff,
            step_factor=grid_step_factor,
            mask_radius_factor=tissue_mask_radius_factor,
        )
        evaluated = evaluate_selection(
            edge_midpoints(cb_top),
            edge_midpoints(commot_top),
            grid=grid,
            bandwidth=cutoff * scale_factor,
            match_radius=cutoff * scale_factor,
        )
        rows.append(
            {
                "example_id": example_id,
                "stage": float(selected.stage),
                "stage_label": str(selected.stage_label),
                "ligand": str(selected.ligand),
                "receptor": str(selected.receptor),
                "selection_family": str(selected.selection_family),
                "top_fraction": top_fraction,
                "scale_factor": scale_factor,
                "bandwidth_and_match_radius": cutoff * scale_factor,
                "n_cytobridge_top_edges": int(len(cb_top)),
                "n_commot_top_edges": int(len(commot_top)),
                **{
                    key: value
                    for key, value in evaluated.items()
                    if not isinstance(value, np.ndarray)
                },
            }
        )
        payload[example_id] = {
            "selection": selected,
            "stage_cells": stage_cells,
            "grid": grid,
            "cytobridge_top": cb_top,
            "commot_top": commot_top,
            "cytobridge_field": evaluated["cytobridge_field"],
            "commot_field": evaluated["commot_field"],
        }
    return pd.DataFrame(rows), payload


def _draw_tissue(ax: plt.Axes, cells: pd.DataFrame) -> None:
    ax.scatter(
        cells["x"], cells["y"], s=2.0, color="#D7DADF", alpha=0.52,
        linewidths=0, rasterized=True, zorder=0,
    )


def _style_spatial_axis(ax: plt.Axes) -> None:
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_primary_spatial_fields(
    metrics: pd.DataFrame,
    payload: Mapping[str, Mapping[str, object]],
    out: Path,
) -> None:
    fig, axes = plt.subplots(len(metrics), 4, figsize=(16.2, 4.4 * len(metrics)))
    axes = np.asarray(axes).reshape(len(metrics), 4)
    for row_index, row in enumerate(metrics.itertuples(index=False)):
        item = payload[str(row.example_id)]
        cells = item["stage_cells"]
        grid: Grid = item["grid"]  # type: ignore[assignment]
        cb_field = np.asarray(item["cytobridge_field"], dtype=float)
        commot_field = np.asarray(item["commot_field"], dtype=float)
        extent = [grid.x_edges[0], grid.x_edges[-1], grid.y_edges[0], grid.y_edges[-1]]
        positive = np.concatenate([cb_field[cb_field > 0], commot_field[commot_field > 0]])
        vmax = float(np.quantile(positive, 0.995)) if positive.size else 1.0
        label = f"{str(row.ligand).upper()}→{str(row.receptor).upper()} | {row.stage_label}"
        for column in range(4):
            _draw_tissue(axes[row_index, column], cells)  # type: ignore[arg-type]
            _style_spatial_axis(axes[row_index, column])
        # Keep the true cell cloud visible; very low Gaussian tails otherwise
        # turn the whole tissue silhouette nearly black and hide hotspot shape.
        masked_cb = np.ma.masked_where(
            (~grid.tissue_mask) | (cb_field < 0.02 * vmax), cb_field
        )
        masked_commot = np.ma.masked_where(
            (~grid.tissue_mask) | (commot_field < 0.02 * vmax), commot_field
        )
        axes[row_index, 0].imshow(
            masked_cb, origin="lower", extent=extent, cmap="magma", vmin=0, vmax=vmax,
            interpolation="bilinear", alpha=0.90, zorder=1,
        )
        axes[row_index, 1].imshow(
            masked_commot, origin="lower", extent=extent, cmap="magma", vmin=0, vmax=vmax,
            interpolation="bilinear", alpha=0.90, zorder=1,
        )
        cb80, cb80_threshold = hdr_mask(cb_field, 0.80)
        co80, co80_threshold = hdr_mask(commot_field, 0.80)
        _, cb50_threshold = hdr_mask(cb_field, 0.50)
        _, co50_threshold = hdr_mask(commot_field, 0.50)
        for threshold, linestyle, linewidth in (
            (cb80_threshold, "--", 1.4), (cb50_threshold, "-", 2.0)
        ):
            axes[row_index, 2].contour(
                grid.xx, grid.yy, cb_field, levels=[threshold], colors=["#D56A00"],
                linestyles=[linestyle], linewidths=[linewidth], zorder=3,
            )
        for threshold, linestyle, linewidth in (
            (co80_threshold, "--", 1.4), (co50_threshold, "-", 2.0)
        ):
            axes[row_index, 2].contour(
                grid.xx, grid.yy, commot_field, levels=[threshold], colors=["#118C7E"],
                linestyles=[linestyle], linewidths=[linewidth], zorder=3,
            )
        categories = np.zeros_like(cb_field, dtype=int)
        categories[cb80 & ~co80] = 1
        categories[~cb80 & co80] = 2
        categories[cb80 & co80] = 3
        categories[~grid.tissue_mask] = 0
        agreement_cmap = ListedColormap(
            [(1, 1, 1, 0), "#E38B37", "#39A89B", "#7756B3"]
        )
        axes[row_index, 3].imshow(
            np.ma.masked_where(categories == 0, categories), origin="lower", extent=extent,
            cmap=agreement_cmap, vmin=0, vmax=3, interpolation="nearest", alpha=0.80,
            zorder=1,
        )
        axes[row_index, 0].set_ylabel(label, fontsize=11, weight="bold")
        axes[row_index, 3].text(
            0.02, 0.02,
            f"OVL={row.field_overlap_ovl:.2f}\nDice80={row.hdr80_dice:.2f}\nMatchF1={row.spatial_match_f1:.2f}",
            transform=axes[row_index, 3].transAxes, ha="left", va="bottom", fontsize=9,
            bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "#D5D8DC", "alpha": 0.9},
        )
    titles = (
        "CytoBridge top 20% positive edges\nequal-edge midpoint density",
        "COMMOT top 20% positive edges\nequal-edge midpoint density",
        "Same coordinates: HDR contours\nsolid=50% mass; dashed=80% mass",
        "80%-HDR regions\npurple=shared",
    )
    for ax, title in zip(axes[0], titles):
        ax.set_title(title, fontsize=11.5, weight="bold")
    axes[0, 2].legend(
        handles=[
            Line2D([0], [0], color="#D56A00", lw=2, label="CytoBridge"),
            Line2D([0], [0], color="#118C7E", lw=2, label="COMMOT"),
        ],
        loc="lower right", frameon=True, fontsize=8,
    )
    axes[0, 3].legend(
        handles=[
            Line2D([0], [0], color="#7756B3", lw=6, label="shared"),
            Line2D([0], [0], color="#E38B37", lw=6, label="CytoBridge only"),
            Line2D([0], [0], color="#39A89B", lw=6, label="COMMOT only"),
        ],
        loc="lower right", frameon=True, fontsize=8,
    )
    fig.suptitle(
        "Spatial-coordinate consistency of selected ligand–receptor axes",
        fontsize=16, weight="bold", y=0.995,
    )
    fig.text(
        0.5, 0.006,
        "Brighter = more selected midpoints after unit-mass smoothing (raw score selects edges but does not weight brightness). "
        "OVL = field overlap; Dice80 = 80%-region overlap; MatchF1 = one-to-one midpoint matching. Not ground truth.",
        ha="center", fontsize=8.9, color="#4A4F55",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.975))
    for suffix in ("png", "pdf"):
        fig.savefig(out.with_suffix(f".{suffix}"), dpi=260, bbox_inches="tight")
    plt.close(fig)


def plot_sensitivity(summary: pd.DataFrame, out: Path) -> None:
    examples = summary[["example_id", "ligand", "receptor", "stage_label"]].drop_duplicates()
    fig, axes = plt.subplots(len(examples), 2, figsize=(13.8, 3.9 * len(examples)), sharex=True)
    axes = np.asarray(axes).reshape(len(examples), 2)
    colors = {0.25: "#7B6BA8", 0.5: "#D56A00", 1.0: "#118C7E"}
    for row_index, example in enumerate(examples.itertuples(index=False)):
        local = summary.loc[summary["example_id"].eq(example.example_id)]
        for column, metric in enumerate(("field_overlap_ovl", "spatial_match_f1")):
            ax = axes[row_index, column]
            subset = local.loc[local["metric"].eq(metric)]
            for scale, group in subset.groupby("scale_factor", sort=True):
                group = group.sort_values("top_fraction")
                x = 100 * group["top_fraction"].to_numpy(float)
                color = colors.get(float(scale), None)
                ax.fill_between(
                    x, group["null_ci_low"], group["null_ci_high"], color=color,
                    alpha=0.12, linewidth=0,
                )
                ax.plot(x, group["null_mean"], color=color, ls="--", lw=1.2, alpha=0.8)
                ax.plot(
                    x, group["observed"], color=color, marker="o", lw=2.0,
                    label=f"{scale:g}× cutoff",
                )
                if np.isclose(float(scale), 0.5) and np.any(np.isclose(x, 20.0)):
                    primary_row = group.loc[np.isclose(x, 20.0)].iloc[0]
                    ax.scatter(
                        [20.0], [primary_row["observed"]], marker="*", s=95,
                        color="#202428", edgecolor="white", linewidth=0.6, zorder=6,
                    )
            ax.axvline(20.0, color="#7A7F85", ls=":", lw=0.9, alpha=0.75)
            ax.set_ylim(0, 1.02)
            ax.grid(axis="y", color="#E7E9EC", lw=0.7)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_xlabel("Top positive edges (%)")
            if column == 0:
                ax.set_ylabel(
                    f"{str(example.ligand).upper()}→{str(example.receptor).upper()}\n{example.stage_label}",
                    fontsize=10.5, weight="bold",
                )
        axes[row_index, 0].set_title("Normalized field overlap (OVL)" if row_index == 0 else "")
        axes[row_index, 1].set_title("One-to-one spatial MatchF1" if row_index == 0 else "")
    handles, labels = axes[0, 1].get_legend_handles_labels()
    handles.append(
        Line2D([0], [0], marker="*", color="none", markerfacecolor="#202428",
               markersize=9, label="primary: top 20%, 0.5× cutoff")
    )
    axes[0, 1].legend(handles=handles, loc="upper left", frameon=True, fontsize=8)
    primary = summary.loc[
        summary["metric"].eq("field_overlap_ovl")
        & np.isclose(summary["top_fraction"], 0.20)
        & np.isclose(summary["scale_factor"], 0.50)
    ]
    n_above = int((primary["observed"] > primary["null_ci_high"]).sum())
    fig.suptitle(
        "Observed spatial agreement versus adaptive fixed-support score-permutation null",
        fontsize=15, weight="bold", y=1.005,
    )
    fig.text(
        0.5, 0.004,
        f"★ primary setting (top 20%, 0.5× cutoff): {n_above}/{len(primary)} OVL axes above the null interval. "
        "Colors are bandwidth multiples for OVL and match-radius multiples for MatchF1; solid=observed, dashed/ribbon=null.",
        ha="center", fontsize=8.9, color="#4A4F55",
    )
    fig.tight_layout(rect=(0, 0.025, 1, 0.98))
    for suffix in ("png", "pdf"):
        fig.savefig(out.with_suffix(f".{suffix}"), dpi=260, bbox_inches="tight")
    plt.close(fig)


def component_controls(
    selection: pd.DataFrame,
    cb: pd.DataFrame,
    commot: pd.DataFrame,
    coordinates: pd.DataFrame,
    *,
    cutoff: float,
    top_fraction: float,
    scale_factor: float,
    permutations: int,
    seed: int,
    grid_step_factor: float,
    tissue_mask_radius_factor: float,
    min_stratum: int,
    permutation_bins: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(int(seed) + 991)
    rows: list[dict[str, object]] = []
    for selected in selection.itertuples(index=False):
        example_id = str(selected.example_id)
        local_cb = cb.loc[cb["example_id"].astype(str).eq(example_id)].copy()
        # Hold the LR-positive, non-self CytoBridge support fixed for every component.
        local_cb = positive_nonself(local_cb, "mean_scaled_lr_activity").reset_index(drop=True)
        _, local_commot = example_supports(example_id, cb, commot)
        stage_cells = coordinates.loc[np.isclose(coordinates["stage"], float(selected.stage))]
        grid = make_grid(
            stage_cells, cutoff=cutoff, step_factor=grid_step_factor,
            mask_radius_factor=tissue_mask_radius_factor,
        )
        commot_top = select_top(local_commot, COMMOT_SCORE, top_fraction)
        commot_field = spatial_field(
            edge_midpoints(commot_top), grid, cutoff * scale_factor
        )
        cb_points = edge_midpoints(local_cb)
        for component_id, component_label, score_column in PRIMARY_COMPONENTS:
            scores = pd.to_numeric(local_cb[score_column], errors="coerce").to_numpy(float)
            observed_indices = _top_positions(scores, top_fraction)
            observed_field = spatial_field(
                cb_points[observed_indices], grid, cutoff * scale_factor
            )
            observed = field_metrics(observed_field, commot_field)
            null: list[float] = []
            null_top_sets: list[tuple[int, ...]] = []
            conditional_on_lr = component_id in {"attention_lr", "exact_message_lr"}
            strata = permutation_strata(
                local_cb,
                method=("cytobridge" if conditional_on_lr else "cytobridge_no_lr"),
                min_size=min_stratum,
                bins=permutation_bins,
            ).to_numpy(str)
            for _ in range(int(permutations)):
                if conditional_on_lr:
                    modifier_column = (
                        "mean_attention_abs"
                        if component_id == "attention_lr"
                        else "mean_exact_message"
                    )
                    modifier = local_cb[modifier_column].to_numpy(float)
                    permuted_modifier = permute_within_strata(modifier, strata, rng)
                    permuted_scores = (
                        local_cb["mean_scaled_lr_activity"].to_numpy(float)
                        * permuted_modifier
                    )
                    null_rule = (
                        "keep LR activity and COMMOT fixed; permute the attention/message modifier "
                        "within audited adaptive type/distance/LR strata"
                    )
                else:
                    permuted_scores = permute_within_strata(scores, strata, rng)
                    null_rule = (
                        "keep COMMOT fixed; permute the component score within audited adaptive "
                        "type/distance strata"
                    )
                indices = _top_positions(permuted_scores, top_fraction)
                null_top_sets.append(tuple(np.sort(indices).tolist()))
                field = spatial_field(cb_points[indices], grid, cutoff * scale_factor)
                null.append(float(field_metrics(field, commot_field)["field_overlap_ovl"]))
            null_array = np.asarray(null, dtype=float)
            rows.append(
                {
                    "example_id": example_id,
                    "stage": float(selected.stage),
                    "stage_label": str(selected.stage_label),
                    "ligand": str(selected.ligand),
                    "receptor": str(selected.receptor),
                    "component": component_id,
                    "component_label": component_label.replace("\n", " "),
                    "score_column": score_column,
                    "top_fraction": top_fraction,
                    "scale_factor": scale_factor,
                    "field_overlap_ovl": float(observed["field_overlap_ovl"]),
                    "hdr80_dice": float(observed["hdr80_dice"]),
                    "null_mean": float(np.mean(null_array)),
                    "null_ci_low": float(np.quantile(null_array, 0.025)),
                    "null_ci_high": float(np.quantile(null_array, 0.975)),
                    "observed_minus_null_mean": float(observed["field_overlap_ovl"])
                    - float(np.mean(null_array)),
                    "empirical_p_greater_equal": float(
                        (1 + np.sum(null_array >= float(observed["field_overlap_ovl"])))
                        / (1 + len(null_array))
                    ),
                    "n_permutations": len(null_array),
                    "n_unique_permuted_top_sets": int(len(set(null_top_sets))),
                    "mean_top_membership_fraction_changed": float(
                        np.mean(
                            [
                                1.0
                                - len(np.intersect1d(observed_indices, np.asarray(item), assume_unique=False))
                                / max(1, len(observed_indices))
                                for item in null_top_sets
                            ]
                        )
                    ),
                    "null_rule": null_rule,
                }
            )
    result = pd.DataFrame(rows)
    baseline = result.loc[result["component"].eq("lr_only"), ["example_id", "field_overlap_ovl"]]
    baseline = baseline.rename(columns={"field_overlap_ovl": "lr_only_field_overlap_ovl"})
    result = result.merge(baseline, on="example_id", validate="many_to_one")
    result["delta_vs_lr_only"] = (
        result["field_overlap_ovl"] - result["lr_only_field_overlap_ovl"]
    )
    return result


def plot_component_controls(controls: pd.DataFrame, out: Path) -> None:
    examples = controls[["example_id", "ligand", "receptor", "stage_label"]].drop_duplicates()
    fig, axes = plt.subplots(1, len(examples), figsize=(5.1 * len(examples), 4.9), sharey=True)
    axes = np.atleast_1d(axes)
    colors = ["#9A9FA5", "#B7A2D6", "#D56A00", "#397BA6"]
    order = [item[0] for item in PRIMARY_COMPONENTS]
    labels = [item[1] for item in PRIMARY_COMPONENTS]
    for ax, example in zip(axes, examples.itertuples(index=False)):
        local = controls.loc[controls["example_id"].eq(example.example_id)].set_index("component").loc[order]
        x = np.arange(len(order))
        observed = local["field_overlap_ovl"].to_numpy(float)
        ax.bar(x, observed, color=colors, width=0.70, alpha=0.90)
        null_mean = local["null_mean"].to_numpy(float)
        errors = np.vstack(
            [null_mean - local["null_ci_low"].to_numpy(float), local["null_ci_high"].to_numpy(float) - null_mean]
        )
        ax.errorbar(
            x, null_mean, yerr=errors, fmt="o", color="#202428", capsize=3,
            markersize=4, lw=1.1, label="permutation null mean ±95% interval",
        )
        for index, value in enumerate(observed):
            ax.text(
                index, max(value - 0.035, 0.015), f"{value:.2f}", ha="center",
                va="top", fontsize=9, color="white", weight="bold",
            )
        attention_delta = float(local.loc["attention_lr", "delta_vs_lr_only"])
        ax.text(
            2, observed[2] + 0.035, f"Δ vs LR-only\n{attention_delta:+.3f}",
            ha="center", va="bottom", fontsize=8.2, color="#8A3F00", weight="bold",
        )
        exact_row = local.loc["exact_message_lr"]
        if (
            float(exact_row["observed_minus_null_mean"]) > 0
            and float(exact_row["empirical_p_greater_equal"]) < 0.05
        ):
            ax.text(
                3, observed[3] + 0.035,
                f"above null\nunadj. P={float(exact_row['empirical_p_greater_equal']):.3g}",
                ha="center", va="bottom", fontsize=8.1, color="#245D7C", weight="bold",
            )
        ax.set_xticks(x, labels, fontsize=8.5)
        ax.set_title(
            f"{str(example.ligand).upper()}→{str(example.receptor).upper()} | {example.stage_label}",
            fontsize=10.5, weight="bold",
        )
        ax.set_ylim(0, 1.08)
        ax.grid(axis="y", color="#E7E9EC", lw=0.7)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Spatial field overlap with COMMOT (OVL)")
    axes[-1].legend(loc="upper right", frameon=True, fontsize=8)
    fig.suptitle(
        "Does attention add spatial information beyond ligand–receptor expression geography?",
        fontsize=14.5, weight="bold", y=1.01,
    )
    fig.text(
        0.5, 0.005,
        "Bars = observed OVL; black dot/line = permutation-null mean/95% interval. Orange labels give attention×LR − LR-only. "
        "Only a positive increment plus an above-null test supports added attention information.",
        ha="center", fontsize=8.9, color="#4A4F55",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.98))
    for suffix in ("png", "pdf"):
        fig.savefig(out.with_suffix(f".{suffix}"), dpi=260, bbox_inches="tight")
    plt.close(fig)


def _cell_percentiles(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    result = np.zeros(len(values), dtype=float)
    positive = values > 0
    if positive.any():
        result[positive] = pd.Series(values[positive]).rank(method="average", pct=True).to_numpy()
    return result


def _array_spearman(left: np.ndarray, right: np.ndarray, mask: np.ndarray) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    if mask.sum() < 3 or np.unique(left[mask]).size < 2 or np.unique(right[mask]).size < 2:
        return float("nan")
    return float(spearmanr(left[mask], right[mask]).statistic)


def _positive_top_cell_set(values: np.ndarray, fraction: float = 0.20) -> set[int]:
    values = np.asarray(values, dtype=float)
    positive = np.flatnonzero(values > 0)
    if positive.size == 0:
        return set()
    n = max(1, int(math.ceil(fraction * len(positive))))
    order = np.argsort(-values[positive], kind="mergesort")
    return set(positive[order[:n]].tolist())


def direction_cell_fields(
    selection: pd.DataFrame,
    cb: pd.DataFrame,
    commot: pd.DataFrame,
    coordinates: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, dict[str, np.ndarray | pd.DataFrame | object]]]:
    rows: list[dict[str, object]] = []
    payload: dict[str, dict[str, np.ndarray | pd.DataFrame | object]] = {}
    for selected in selection.itertuples(index=False):
        example_id = str(selected.example_id)
        local_cb, local_commot = example_supports(example_id, cb, commot)
        cells = coordinates.loc[np.isclose(coordinates["stage"], float(selected.stage))].copy()
        cell_ids = cells["cell_id"].astype(str).tolist()
        global_indices = cells["cell_index_global"].astype(int).tolist()
        id_position = {value: index for index, value in enumerate(cell_ids)}
        global_position = {value: index for index, value in enumerate(global_indices)}
        arrays: dict[str, np.ndarray] = {}
        for direction, cb_key, co_key in (
            ("outgoing", "source_index", "source_cell_id"),
            ("incoming", "target_index", "target_cell_id"),
        ):
            cb_values = np.zeros(len(cells), dtype=float)
            co_values = np.zeros(len(cells), dtype=float)
            for index, value in zip(local_cb[cb_key].astype(int), local_cb[CB_SCORE].to_numpy(float)):
                cb_values[global_position[int(index)]] += float(value)
            for cell_id, value in zip(
                local_commot[co_key].astype(str), local_commot[COMMOT_SCORE].to_numpy(float)
            ):
                co_values[id_position[cell_id]] += float(value)
            arrays[f"cytobridge_{direction}"] = cb_values
            arrays[f"commot_{direction}"] = co_values
            cb_mass = cb_values / cb_values.sum() if cb_values.sum() > 0 else cb_values
            co_mass = co_values / co_values.sum() if co_values.sum() > 0 else co_values
            union_active = (cb_values > 0) | (co_values > 0)
            correlation = _array_spearman(
                cb_values, co_values, np.ones(len(cb_values), dtype=bool)
            )
            active_correlation = _array_spearman(cb_values, co_values, union_active)
            cb_positive = set(np.flatnonzero(cb_values > 0).tolist())
            co_positive = set(np.flatnonzero(co_values > 0).tolist())
            support_union = cb_positive | co_positive
            cb_top = _positive_top_cell_set(cb_values)
            co_top = _positive_top_cell_set(co_values)
            top_union = cb_top | co_top
            rows.append(
                {
                    "example_id": example_id,
                    "stage": float(selected.stage),
                    "stage_label": str(selected.stage_label),
                    "ligand": str(selected.ligand),
                    "receptor": str(selected.receptor),
                    "direction": direction,
                    "cell_mass_overlap_ovl": float(np.minimum(cb_mass, co_mass).sum()),
                    "spearman_all_stage_cells_including_zeros": correlation,
                    "spearman_active_union_cells": active_correlation,
                    "positive_cell_support_jaccard": (
                        float(len(cb_positive & co_positive) / len(support_union))
                        if support_union
                        else float("nan")
                    ),
                    "top20_positive_cell_jaccard": (
                        float(len(cb_top & co_top) / len(top_union))
                        if top_union
                        else float("nan")
                    ),
                    "n_stage_cells": int(len(cells)),
                    "n_cytobridge_positive_cells": int(np.sum(cb_values > 0)),
                    "n_commot_positive_cells": int(np.sum(co_values > 0)),
                }
            )
        payload[example_id] = {"selection": selected, "cells": cells, **arrays}
    return pd.DataFrame(rows), payload


def plot_direction_cell_fields(
    metrics: pd.DataFrame,
    payload: Mapping[str, Mapping[str, np.ndarray | pd.DataFrame | object]],
    out: Path,
) -> None:
    examples = metrics[["example_id", "ligand", "receptor", "stage_label"]].drop_duplicates()
    fig, axes = plt.subplots(len(examples), 4, figsize=(15.2, 4.0 * len(examples)))
    axes = np.asarray(axes).reshape(len(examples), 4)
    columns = (
        ("cytobridge_outgoing", "CytoBridge outgoing\n(sender activity)"),
        ("commot_outgoing", "COMMOT outgoing\n(sender activity)"),
        ("cytobridge_incoming", "CytoBridge incoming\n(receiver activity)"),
        ("commot_incoming", "COMMOT incoming\n(receiver activity)"),
    )
    last = None
    for row_index, example in enumerate(examples.itertuples(index=False)):
        item = payload[str(example.example_id)]
        cells: pd.DataFrame = item["cells"]  # type: ignore[assignment]
        for column_index, (key, title) in enumerate(columns):
            ax = axes[row_index, column_index]
            percentile = _cell_percentiles(np.asarray(item[key], dtype=float))
            last = ax.scatter(
                cells["x"], cells["y"], c=percentile, s=5.0, cmap="viridis",
                vmin=0, vmax=1, linewidths=0, rasterized=True,
            )
            _style_spatial_axis(ax)
            if row_index == 0:
                ax.set_title(title, fontsize=11, weight="bold")
        outgoing = metrics.loc[
            metrics["example_id"].eq(example.example_id) & metrics["direction"].eq("outgoing")
        ].iloc[0]
        incoming = metrics.loc[
            metrics["example_id"].eq(example.example_id) & metrics["direction"].eq("incoming")
        ].iloc[0]
        axes[row_index, 0].set_ylabel(
            f"{str(example.ligand).upper()}→{str(example.receptor).upper()}\n{example.stage_label}",
            fontsize=10, weight="bold",
        )
        axes[row_index, 0].text(
            0.98, 0.02,
            f"CB ↔ COMMOT\nactive-cell ρ={outgoing['spearman_active_union_cells']:.2f}\nmass OVL={outgoing['cell_mass_overlap_ovl']:.2f}",
            transform=axes[row_index, 0].transAxes, ha="right", va="bottom", fontsize=7.8,
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#D5D8DC", "alpha": 0.88},
            zorder=8,
            clip_on=False,
        )
        axes[row_index, 2].text(
            0.98, 0.02,
            f"CB ↔ COMMOT\nactive-cell ρ={incoming['spearman_active_union_cells']:.2f}\nmass OVL={incoming['cell_mass_overlap_ovl']:.2f}",
            transform=axes[row_index, 2].transAxes, ha="right", va="bottom", fontsize=7.8,
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#D5D8DC", "alpha": 0.88},
            zorder=8,
            clip_on=False,
        )
    if last is not None:
        colorbar_axis = fig.add_axes([0.93, 0.36, 0.012, 0.28])
        colorbar = fig.colorbar(last, cax=colorbar_axis)
        colorbar.set_label("Within-method cell activity percentile (zeros = 0)")
    fig.suptitle(
        "Direction-aware spatial consistency: where are sender and receiver activities?",
        fontsize=15, weight="bold", y=0.995,
    )
    fig.text(
        0.5, 0.006,
        "Colors are within-method percentiles, not comparable raw intensities. mass OVL asks whether broad activity regions overlap; "
        "active-cell ρ asks whether the same individual cells rank highly.",
        ha="center", fontsize=8.9, color="#4A4F55",
    )
    fig.subplots_adjust(left=0.075, right=0.91, bottom=0.055, top=0.90, wspace=0.16, hspace=0.12)
    for suffix in ("png", "pdf"):
        fig.savefig(out.with_suffix(f".{suffix}"), dpi=260, bbox_inches="tight")
    plt.close(fig)


def _fmt(value: float) -> str:
    return "NA" if not np.isfinite(value) else f"{value:.3f}"


def write_readme_cn(
    path: Path,
    primary: pd.DataFrame,
    sensitivity: pd.DataFrame,
    controls: pd.DataFrame,
    direction: pd.DataFrame,
    strata_audit: pd.DataFrame,
    *,
    cutoff: float,
    permutations: int,
    permutation_bins: int,
    min_permutation_stratum: int,
) -> None:
    primary_rows = "\n".join(
        f"| {row.ligand.upper()}→{row.receptor.upper()} ({row.stage_label}) | "
        f"{row.field_overlap_ovl:.3f} | {row.hdr80_dice:.3f} | "
        f"{row.spatial_match_f1:.3f} | {int(row.n_cytobridge_top_edges)} / {int(row.n_commot_top_edges)} |"
        for row in primary.itertuples(index=False)
    )
    control_rows = []
    for example_id, group in controls.groupby("example_id", sort=False):
        first = group.iloc[0]
        lookup = group.set_index("component")
        control_rows.append(
            f"| {str(first.ligand).upper()}→{str(first.receptor).upper()} ({first.stage_label}) | "
            f"{lookup.loc['lr_only', 'field_overlap_ovl']:.3f} | "
            f"{lookup.loc['attention_only', 'field_overlap_ovl']:.3f} | "
            f"{lookup.loc['attention_lr', 'field_overlap_ovl']:.3f} | "
            f"{lookup.loc['attention_lr', 'delta_vs_lr_only']:+.3f} | "
            f"{lookup.loc['exact_message_lr', 'field_overlap_ovl']:.3f} |"
        )
    direction_rows = []
    for example_id, group in direction.groupby("example_id", sort=False):
        first = group.iloc[0]
        out = group.loc[group["direction"].eq("outgoing")].iloc[0]
        incoming = group.loc[group["direction"].eq("incoming")].iloc[0]
        direction_rows.append(
            f"| {str(first.ligand).upper()}→{str(first.receptor).upper()} ({first.stage_label}) | "
            f"{_fmt(float(out.spearman_active_union_cells))} | {out.cell_mass_overlap_ovl:.3f} | "
            f"{out.top20_positive_cell_jaccard:.3f} | "
            f"{_fmt(float(incoming.spearman_active_union_cells))} | {incoming.cell_mass_overlap_ovl:.3f} | "
            f"{incoming.top20_positive_cell_jaccard:.3f} |"
        )
    primary_null = sensitivity.loc[
        np.isclose(sensitivity["top_fraction"], primary["top_fraction"].iloc[0])
        & np.isclose(sensitivity["scale_factor"], primary["scale_factor"].iloc[0])
        & sensitivity["metric"].eq("field_overlap_ovl")
    ]
    null_rows = "\n".join(
        f"| {row.ligand.upper()}→{row.receptor.upper()} ({row.stage_label}) | "
        f"{row.observed:.3f} | {row.null_mean:.3f} [{row.null_ci_low:.3f}, {row.null_ci_high:.3f}] | "
        f"{row.observed_minus_null_mean:+.3f} | {row.empirical_p_greater_equal:.4f} |"
        for row in primary_null.itertuples(index=False)
    )
    null_interpretation = []
    for row in primary_null.itertuples(index=False):
        if row.observed > row.null_ci_high:
            verdict = "高于 null 95% 区间"
        elif row.observed < row.null_ci_low:
            verdict = "低于 null 95% 区间"
        else:
            verdict = "落在 null 95% 区间内"
        null_interpretation.append(
            f"- {row.ligand.upper()}→{row.receptor.upper()}：observed OVL={row.observed:.3f}，"
            f"null={row.null_mean:.3f} [{row.null_ci_low:.3f}, {row.null_ci_high:.3f}]，{verdict}。"
        )
    attention_increment = controls.loc[controls["component"].eq("attention_lr")]
    positive_increment = int((attention_increment["delta_vs_lr_only"] > 0).sum())
    exact_positive = controls.loc[
        controls["component"].eq("exact_message_lr")
        & controls["observed_minus_null_mean"].gt(0)
        & controls["empirical_p_greater_equal"].lt(0.05)
    ]
    if len(exact_positive) == 1:
        exact = exact_positive.iloc[0]
        exact_note = (
            f"- 探索性例外：{str(exact.ligand).upper()}→{str(exact.receptor).upper()} 的 "
            f"exact-message×LR 高于 modifier-permutation null（OVL={exact.field_overlap_ovl:.3f}，"
            f"null mean={exact.null_mean:.3f}，未校正单侧 P={exact.empirical_p_greater_equal:.4f}）。"
            "这是 exact-message 分量的轴特异结果，不能改写成 attention 获得了空间验证。"
        )
    else:
        exact_note = (
            f"- 有 {len(exact_positive)} 条 exact-message×LR 轴超过对应 null；"
            "这些是探索性分量结果，不替代 attention 增量检验。"
        )
    strata_rows: list[str] = []
    primary_strata = strata_audit.loc[strata_audit["analysis"].eq("primary_score_null")]
    for (example_id, method), group in primary_strata.groupby(
        ["example_id", "method"], sort=False
    ):
        first = group.iloc[0]
        fractions = group.set_index("coarsening_level")["fraction_edges"].to_dict()
        strata_rows.append(
            f"| {str(first.ligand).upper()}→{str(first.receptor).upper()} | {method} | "
            f"{fractions.get('fine_type_covariate', 0.0):.1%} | "
            f"{fractions.get('pooled_covariate', 0.0):.1%} | "
            f"{fractions.get('pooled_distance', 0.0):.1%} | "
            f"{fractions.get('global', 0.0):.1%} | "
            f"{group['movable_edge_fraction_overall'].min():.1%} |"
        )
    text = f"""# 斑马鱼空间坐标一致性：怎么画、怎么看

## 先说这次改了什么

旧箭头图一次叠加很多细胞和箭头，肉眼很难判断一致性，而且 COMMOT 点更多时，“附近有点”的覆盖率会天然升高。本目录改用四类更直接的空间图：

1. 两种方法在同一坐标上的单位质量 hotspot field；
2. 50%/80% high-density region（HDR）轮廓与共有区域；
3. 对 top fraction 和空间尺度的敏感性，以及固定 edge support 的 score-permutation null；
4. sender/outgoing 与 receiver/incoming 分开画，避免 midpoint 掩盖方向差异。

所有图都使用同一个 `spatial_aligned` 坐标。固定 graph cutoff 为 `{cutoff:.6f}`；主图平滑带宽和一对一匹配半径均为半个 cutoff。每种方法先在自身正分、非 `i→i` edges 内取 top 20%，再各自归一化为空间总质量 1，因此没有直接比较两个软件的 raw score 单位。Raw score 只决定哪些 edge 入选；入选后每条 edge 在 hotspot histogram 中等权计数，不再按 raw score 加权。

## 图 1：`spatial_hotspot_consistency`

每行是一条预先选定的 LR 轴：

- 第 1、2 列：CytoBridge 与 COMMOT 的 top-edge midpoint hotspot；
- 第 3 列：橙色是 CytoBridge HDR，绿色是 COMMOT HDR；实线包住 50% 空间质量，虚线包住 80%；
- 第 4 列：80%-HDR 的共有区域为紫色，橙色/绿色分别是单方法区域。

三个数字：

- `OVL=Σ min(f_CB,f_COMMOT)`：两个单位质量空间场重叠多少，0=完全分开，1=完全相同；
- `Dice80`：双方 80%-HDR 的面积 Dice；
- `MatchF1`：规定半径内的最大一对一 midpoint matching。一个密集 COMMOT 点不能重复匹配很多 CytoBridge 点。

| LR 轴 | OVL | Dice80 | one-to-one MatchF1 | top edges CB / COMMOT |
|---|---:|---:|---:|---:|
{primary_rows}

## 图 2：`spatial_null_sensitivity`

实线是观察值，虚线和淡色带是 score-permutation null 的均值与 95% 区间。Null 固定每种方法已有的正分 edge support、空间坐标和 edge 数量。它先尝试保留 sender→receiver type 与 `{permutation_bins}` 档距离/LR activity；不足 `{min_permutation_stratum}` 条边的 type 层逐级合并到 covariate-only、distance-only，最后才允许极少量 global fallback。共 `{permutations}` 次置换。

实际分层不是一句“已分层”带过，而是写入 `permutation_strata_diagnostics.csv`：

| LR 轴 | method | exact type+covariate | pooled covariate | pooled distance | global fallback | movable edges |
|---|---|---:|---:|---:|---:|---:|
{chr(10).join(strata_rows)}

`global fallback` 应接近 0；`movable edges` 应接近 100%。表中还记录每层的 realized stratum size、距离/LR span 和 assignment hash，避免分层静默坍缩。

主设置（top 20%、0.5×cutoff）的 OVL 为：

| LR 轴 | observed OVL | null mean [95% interval] | observed−null | empirical P |
|---|---:|---:|---:|---:|
{null_rows}

只有实线在多种 top fraction/scale 下持续高于 null band，才能叫 robust spatial ranking consistency。若观察值与 null 接近，说明肉眼热点相似主要来自共同 tissue/edge-support geography，而不是双方对 edge 强弱的相同排序。

当前主设置的直接判断：

{chr(10).join(null_interpretation)}

## 图 3：`spatial_component_control`

这是最关键的 reviewer control。四个 CytoBridge 分数全部限制在同一批 LR-positive edges，再分别与 COMMOT 空间场比较：

- LR-only；
- attention-only；
- attention×LR；
- exact-message×LR。

| LR 轴 | LR-only OVL | attention-only OVL | attention×LR OVL | attention×LR − LR-only | exact-message×LR OVL |
|---|---:|---:|---:|---:|---:|
{chr(10).join(control_rows)}

如果 `attention×LR − LR-only > 0` 且 modifier-permutation 的观察值超过 null，才能说 attention 在共同 LR expression geography 之外增加了空间一致性。该检验固定 LR activity 与观察到的 COMMOT 场，只在自适应 type/distance/LR strata 内置换 attention；LR-only 和 attention-only 的 score null 不用 LR 自身分层。

## 图 4：`spatial_sender_receiver_consistency`

midpoint 不区分“谁发送、谁接收”。这张图把每个细胞的 outgoing 和 incoming edge score 分别求和，再在相同真实坐标上画 percentile。

| LR 轴 | outgoing active-union rho | outgoing mass OVL | outgoing top-cell Jaccard | incoming active-union rho | incoming mass OVL | incoming top-cell Jaccard |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(direction_rows)}

主表中的 Spearman 只在双方 active-cell union 内计算，避免几千个共同零值人为抬高相关；机器表仍保留“全部 stage cells、缺失记 0”的审计列。sender/receiver 一致性低时，不能用 midpoint overlap 声称方向一致。

## 当前最重要的结论

三条轴的 raw top-hotspot OVL 可以是中等正值，因此坐标图肉眼会看到部分共有区域；但 fixed-support null 才回答“高分排序是否比共同组织/edge support 本身更一致”。若上面的 observed OVL 没有超过 null band，就不能把 raw hotspot overlap 写成额外的独立验证。

在 component control 中，`attention×LR` 相对 `LR-only` 为正的轴数是 **{positive_increment}/{len(attention_increment)}**。因此只有在该增量为正且超过 modifier-permutation null 时，才能把空间一致性归因于 attention；否则更合理的解释是共同 LR expression geography 提供了主要空间结构。

{exact_note}

## 能说什么，不能说什么

可以说：这些预先选择的 LR 例子在真实空间坐标上呈现多少 hotspot consistency，并且这种一致性是否超过固定-support 的分数置换基线。

不能说：

- 空间热点相似就是 exact cell-edge accuracy；
- midpoint 相似证明 sender→receiver 方向相同；
- COMMOT 是实验 ground truth；
- 三条经过规则筛选的例子代表全部 LR axes；
- attention×LR 是模型原生 CCC probability。

输入表中的 CytoBridge cell edge 是在其实际出现的 grouping seeds 上取均值，未出现的 seed 没有补零；因此 seed coverage 不平衡仍是一个限制。外部方法还与 CytoBridge 共用表达矩阵、细胞注释、空间坐标及部分 LR 数据库，这是一致性分析，不是完全独立验证。
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parser().parse_args()
    bundle = args.bundle_dir.expanduser().resolve()
    coordinates_path = args.coordinates_csv.expanduser().resolve()
    output = _prepare_output(args.output_dir, bool(args.overwrite))
    cutoff = _read_cutoff(bundle, args.graph_cutoff)
    top_fractions = _parse_floats(args.top_fractions)
    scale_factors = _parse_floats(args.scale_factors)
    if args.permutations < 20:
        raise ValueError("At least 20 permutations are required")
    if not 0 < args.primary_top_fraction <= 1:
        raise ValueError("primary top fraction must be in (0, 1]")
    if args.primary_scale <= 0:
        raise ValueError("primary scale must be positive")
    if args.permutation_bins < 2:
        raise ValueError("permutation bins must be at least 2")
    if args.min_permutation_stratum < 2:
        raise ValueError("minimum permutation stratum must be at least 2")
    if not 0 <= args.max_global_fallback_fraction <= 1:
        raise ValueError("max global fallback fraction must be in [0, 1]")
    selection, cb, commot, coordinates = load_inputs(bundle, coordinates_path)
    cb, commot = attach_coordinates(cb, commot, coordinates)

    strata_audit = permutation_strata_audit(
        selection,
        cb,
        commot,
        min_size=int(args.min_permutation_stratum),
        bins=int(args.permutation_bins),
    )
    global_rows = strata_audit.loc[strata_audit["coarsening_level"].eq("global")]
    if (
        not global_rows.empty
        and global_rows["fraction_edges"].max()
        > float(args.max_global_fallback_fraction)
    ):
        worst = global_rows.sort_values("fraction_edges", ascending=False).iloc[0]
        raise ValueError(
            "Adaptive permutation strata exceed global-fallback limit: "
            f"{worst.example_id}/{worst.analysis}/{worst.method}="
            f"{worst.fraction_edges:.3%} > {args.max_global_fallback_fraction:.3%}"
        )
    if strata_audit["movable_edge_fraction_overall"].min() < 0.95:
        raise ValueError("Fewer than 95% of edges are movable in a permutation assignment")
    strata_path = output / "permutation_strata_diagnostics.csv"
    strata_audit.to_csv(strata_path, index=False)

    primary, payload = primary_metrics_and_fields(
        selection, cb, commot, coordinates, cutoff=cutoff,
        top_fraction=float(args.primary_top_fraction),
        scale_factor=float(args.primary_scale),
        grid_step_factor=float(args.grid_step_factor),
        tissue_mask_radius_factor=float(args.tissue_mask_radius_factor),
    )
    primary_path = output / "spatial_primary_metrics.csv"
    primary.to_csv(primary_path, index=False)
    plot_primary_spatial_fields(primary, payload, output / "spatial_hotspot_consistency")

    sensitivity = sensitivity_and_null(
        selection, cb, commot, coordinates, cutoff=cutoff,
        top_fractions=top_fractions, scale_factors=scale_factors,
        permutations=int(args.permutations), seed=int(args.seed),
        grid_step_factor=float(args.grid_step_factor),
        tissue_mask_radius_factor=float(args.tissue_mask_radius_factor),
        min_stratum=int(args.min_permutation_stratum),
        permutation_bins=int(args.permutation_bins),
    )
    sensitivity_path = output / "spatial_null_sensitivity.csv.gz"
    sensitivity.to_csv(sensitivity_path, index=False, compression="gzip")
    plot_sensitivity(sensitivity, output / "spatial_null_sensitivity")

    controls = component_controls(
        selection, cb, commot, coordinates, cutoff=cutoff,
        top_fraction=float(args.primary_top_fraction), scale_factor=float(args.primary_scale),
        permutations=int(args.permutations), seed=int(args.seed),
        grid_step_factor=float(args.grid_step_factor),
        tissue_mask_radius_factor=float(args.tissue_mask_radius_factor),
        min_stratum=int(args.min_permutation_stratum),
        permutation_bins=int(args.permutation_bins),
    )
    controls_path = output / "spatial_component_control_metrics.csv"
    controls.to_csv(controls_path, index=False)
    plot_component_controls(controls, output / "spatial_component_control")

    direction, direction_payload = direction_cell_fields(
        selection, cb, commot, coordinates
    )
    direction_path = output / "spatial_sender_receiver_metrics.csv"
    direction.to_csv(direction_path, index=False)
    plot_direction_cell_fields(
        direction, direction_payload, output / "spatial_sender_receiver_consistency"
    )

    readme_path = output / "README_CN.md"
    write_readme_cn(
        readme_path, primary, sensitivity, controls, direction, strata_audit,
        cutoff=cutoff, permutations=int(args.permutations),
        permutation_bins=int(args.permutation_bins),
        min_permutation_stratum=int(args.min_permutation_stratum),
    )

    artifact_paths = sorted(
        path for path in output.iterdir() if path.is_file() and path.name != "manifest.json"
    )
    manifest = {
        "schema_version": 1,
        "workflow": "zebrafish_spatial_coordinate_consistency",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "bundle_manifest": _record(bundle / "bundle_manifest.json"),
            "coordinates_csv": _record(coordinates_path),
        },
        "parameters": {
            "graph_cutoff": cutoff,
            "primary_top_fraction": float(args.primary_top_fraction),
            "primary_scale": float(args.primary_scale),
            "top_fractions": list(top_fractions),
            "scale_factors": list(scale_factors),
            "permutations": int(args.permutations),
            "seed": int(args.seed),
            "grid_step_factor": float(args.grid_step_factor),
            "tissue_mask_radius_factor": float(args.tissue_mask_radius_factor),
            "min_permutation_stratum": int(args.min_permutation_stratum),
            "permutation_bins": int(args.permutation_bins),
            "max_global_fallback_fraction": float(args.max_global_fallback_fraction),
        },
        "claims": {
            "spatial_consistency_not_ground_truth": True,
            "midpoint_overlap_not_direction_accuracy": True,
            "component_control_required_for_attention_increment": True,
            "selected_examples_not_all_lr_axes": True,
        },
        "artifacts": [_record(path, relative_to=output) for path in artifact_paths],
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"output_dir": str(output), "n_artifacts": len(artifact_paths)},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
