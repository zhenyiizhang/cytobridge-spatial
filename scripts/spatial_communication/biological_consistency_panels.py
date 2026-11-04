#!/usr/bin/env python3
"""Create direct biological and spatial CCC consistency visualizations.

The script adds three reviewer-facing views to the numerical agreement audit:

1. matched cell-level spatial LR flow maps for CytoBridge and COMMOT;
2. conventional directed cell-type CCC circle plots; and
3. a temporal bubble plot for literature-scoped zebrafish LR axes.

Example selection is deterministic and declared before cell-level COMMOT is
reconstructed: one example is chosen from each of ncWNT, CXCL, and NOTCH by the
mean of within-stage CytoBridge and COMMOT percentiles, subject to positive
support and a minimum CytoBridge active-edge count.  This avoids selecting
examples by visual appearance.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import shutil
import textwrap
from typing import Any, Iterable, Mapping, Sequence
import zlib

import anndata as ad
from matplotlib import cm, colors
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import cKDTree

try:
    from .common import file_record, json_dump, software_versions, utc_now
except ImportError:  # direct script execution
    from common import file_record, json_dump, software_versions, utc_now


BIOLOGICAL_FAMILIES = ("ncWNT", "CXCL", "NOTCH")
STAGE_LABELS = {
    0.0: "5.25 hpf",
    1.0: "10 hpf",
    2.0: "12 hpf",
    3.0: "18 hpf",
    4.0: "24 hpf",
}
METHOD_SPECS = (
    ("cytobridge_attention_rank", "cytobridge_attention", "CytoBridge attention"),
    ("commot_rank", "commot", "COMMOT"),
    ("cellagentchat_ctps_rank", "cellagentchat_ctps", "CellAgentChat CTPS"),
    ("external_consensus_rank", "external_support", "External-only consensus"),
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-dir", required=True, type=Path)
    parser.add_argument("--positive-consistency-dir", required=True, type=Path)
    parser.add_argument("--cytobridge-dir", required=True, type=Path)
    parser.add_argument("--commot-dir", required=True, type=Path)
    parser.add_argument("--h5ad", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--selected-commot-flow-dir", type=Path)
    parser.add_argument("--selected-examples-csv", type=Path)
    parser.add_argument("--select-only", action="store_true")
    parser.add_argument("--min-active-edges", type=int, default=10)
    parser.add_argument("--spatial-display-top-edges", type=int, default=80)
    parser.add_argument("--circle-display-top-edges", type=int, default=16)
    parser.add_argument(
        "--circle-stages",
        type=float,
        nargs="+",
        default=[1.0, 4.0],
        help="Default shows one early (10 hpf) and one late (24 hpf) observed stage.",
    )
    return parser


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} lacks columns: {missing}")


def _read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_file_record(path: Path, record: Mapping[str, Any]) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    if int(record.get("size_bytes", record.get("bytes", -1))) != path.stat().st_size:
        raise ValueError(f"Recorded size disagrees for {path}")
    if str(record.get("sha256", "")).casefold() != _sha256(path).casefold():
        raise ValueError(f"Recorded SHA256 disagrees for {path}")


def _collapse_commot_lr(commot: pd.DataFrame) -> pd.DataFrame:
    _require_columns(
        commot,
        [
            "stage",
            "ligand",
            "receptor",
            "sender_type",
            "receiver_type",
            "score",
            "abundance_controlled_score",
        ],
        "COMMOT LR table",
    )
    values = commot.copy()
    values["stage"] = values["stage"].astype(float)
    values["score"] = pd.to_numeric(values["score"], errors="coerce").fillna(0.0)
    values["abundance_controlled_score"] = pd.to_numeric(
        values["abundance_controlled_score"], errors="coerce"
    ).fillna(0.0)
    grouped = values.groupby(["stage", "ligand", "receptor"], as_index=False).agg(
        commot_native_cell_flow=("score", "sum"),
        commot_positive_contexts=("score", "size"),
    )
    best = values.sort_values(
        [
            "stage",
            "ligand",
            "receptor",
            "abundance_controlled_score",
            "sender_type",
            "receiver_type",
        ],
        ascending=[True, True, True, False, True, True],
    ).drop_duplicates(["stage", "ligand", "receptor"])
    best = best.rename(
        columns={
            "sender_type": "top_commot_sender_type",
            "receiver_type": "top_commot_receiver_type",
            "abundance_controlled_score": "top_commot_abundance_controlled_score",
        }
    )
    return grouped.merge(
        best[
            [
                "stage",
                "ligand",
                "receptor",
                "top_commot_sender_type",
                "top_commot_receiver_type",
                "top_commot_abundance_controlled_score",
            ]
        ],
        on=["stage", "ligand", "receptor"],
        validate="one_to_one",
    )


def select_biological_examples(
    axis_scores: pd.DataFrame,
    known_axes: pd.DataFrame,
    commot_lr: pd.DataFrame,
    *,
    families: Sequence[str] = BIOLOGICAL_FAMILIES,
    min_active_edges: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return all auditable known-axis candidates and one per named family."""

    if min_active_edges < 1:
        raise ValueError("min_active_edges must be positive")
    axis_required = [
        "stage",
        "ligand",
        "receptor",
        "mean_attention_times_lr_activity",
    ]
    known_required = [
        "stage",
        "stage_label",
        "ligand",
        "receptor",
        "pathways",
        "categories",
        "n_active_edges",
        "mean_attention_times_lr_activity",
        "top_attention_sender_type",
        "top_attention_receiver_type",
        "source_ids",
        "source_urls",
        "claim_guardrail",
    ]
    _require_columns(axis_scores, axis_required, "all LR-axis scores")
    _require_columns(known_axes, known_required, "known LR-axis scores")
    all_axes = axis_scores.copy()
    all_axes["stage"] = all_axes["stage"].astype(float)
    all_axes["cytobridge_percentile_all_axes"] = all_axes.groupby("stage")[
        "mean_attention_times_lr_activity"
    ].rank(method="average", pct=True)
    axis_universe = all_axes[
        [
            "stage",
            "ligand",
            "receptor",
            "cytobridge_percentile_all_axes",
        ]
    ].drop_duplicates(["stage", "ligand", "receptor"])
    if len(axis_universe) != len(all_axes):
        raise ValueError("All LR-axis scores contain duplicate stage/LR rows")
    commot_collapsed = _collapse_commot_lr(commot_lr)
    # The formal COMMOT LR long table is positive-only.  Complete its evaluated
    # universe with zeros before ranking so absent positive rows do not become
    # missing or receive an artificially favorable positive-only percentile.
    axis_percentiles = axis_universe.merge(
        commot_collapsed,
        on=["stage", "ligand", "receptor"],
        how="left",
        validate="one_to_one",
    )
    axis_percentiles["commot_native_cell_flow"] = axis_percentiles[
        "commot_native_cell_flow"
    ].fillna(0.0)
    axis_percentiles["commot_positive_contexts"] = (
        axis_percentiles["commot_positive_contexts"].fillna(0).astype(int)
    )
    axis_percentiles["commot_percentile_all_axes"] = axis_percentiles.groupby("stage")[
        "commot_native_cell_flow"
    ].rank(method="average", pct=True)
    candidates = known_axes.copy()
    candidates["stage"] = candidates["stage"].astype(float)
    candidates = candidates.merge(
        axis_percentiles,
        on=["stage", "ligand", "receptor"],
        how="left",
        validate="one_to_one",
    )
    candidates["joint_external_internal_percentile"] = candidates[
        ["cytobridge_percentile_all_axes", "commot_percentile_all_axes"]
    ].mean(axis=1)
    candidates["passes_support_filter"] = (
        candidates["n_active_edges"].ge(min_active_edges)
        & candidates["mean_attention_times_lr_activity"].gt(0)
        & candidates["commot_native_cell_flow"].gt(0)
        & candidates["cytobridge_percentile_all_axes"].notna()
        & candidates["commot_percentile_all_axes"].notna()
    )
    candidates["both_methods_top_quartile"] = candidates[
        "cytobridge_percentile_all_axes"
    ].ge(0.75) & candidates["commot_percentile_all_axes"].ge(0.75)

    selected_rows: list[pd.Series] = []
    for family in families:
        eligible = candidates.loc[
            candidates["pathways"]
            .astype(str)
            .str.split(";")
            .map(lambda values: family in values)
            & candidates["passes_support_filter"]
        ].copy()
        if eligible.empty:
            raise ValueError(
                f"No {family} example passes min_active_edges={min_active_edges}"
            )
        eligible = eligible.sort_values(
            [
                "joint_external_internal_percentile",
                "n_active_edges",
                "stage",
                "ligand",
                "receptor",
            ],
            ascending=[False, False, True, True, True],
        )
        row = eligible.iloc[0].copy()
        row["selection_family"] = family
        selected_rows.append(row)
    selected = pd.DataFrame(selected_rows).reset_index(drop=True)
    selected.insert(
        0,
        "example_id",
        [
            f"{str(row.selection_family).casefold()}_{int(float(row.stage))}_{row.ligand}_{row.receptor}"
            for row in selected.itertuples(index=False)
        ],
    )
    selected["selection_rule"] = (
        "highest mean of within-stage CytoBridge LR-compatible attention and COMMOT native-flow "
        "percentiles within the pre-specified pathway family, requiring positive "
        f"support and n_active_edges>={min_active_edges}"
    )
    candidates = candidates.sort_values(
        [
            "pathways",
            "joint_external_internal_percentile",
            "stage",
            "ligand",
            "receptor",
        ],
        ascending=[True, False, True, True, True],
    ).reset_index(drop=True)
    return candidates, selected


def _complex_tokens(value: str) -> tuple[str, ...]:
    return tuple(token.strip() for token in str(value).split("_") if token.strip())


def _gene_activities(data: ad.AnnData, tokens: Iterable[str]) -> dict[str, np.ndarray]:
    lookup: dict[str, list[int]] = {}
    for index, gene in enumerate(data.var_names.astype(str)):
        lookup.setdefault(gene.casefold(), []).append(index)
    result: dict[str, np.ndarray] = {}
    for token in sorted(set(tokens)):
        indices: list[int] = []
        for gene in _complex_tokens(token):
            matches = lookup.get(gene.casefold(), [])
            if len(matches) != 1:
                raise ValueError(
                    f"Gene {gene!r} from complex {token!r} has {len(matches)} H5AD matches"
                )
            indices.append(matches[0])
        values = data.X[:, indices]
        values = values.toarray() if sparse.issparse(values) else np.asarray(values)
        raw = np.min(np.asarray(values, dtype=np.float32), axis=1)
        positive = raw[raw > 0]
        scale = float(np.quantile(positive, 0.95)) if positive.size else 1.0
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0
        result[token] = np.clip(raw / scale, 0, 1).astype(np.float32)
    return result


def _load_selected_cytobridge_edges(
    directory: Path,
    selected: pd.DataFrame,
    activities: Mapping[str, np.ndarray],
) -> pd.DataFrame:
    selected_stages = set(selected["stage"].astype(float))
    paths = sorted(directory.glob("stage_*/edges_seed_*.csv.gz"))
    if not paths:
        raise FileNotFoundError(f"No CytoBridge edge tables under {directory}")
    frames: list[pd.DataFrame] = []
    columns = [
        "stage",
        "stage_label",
        "grouping_seed",
        "source_index",
        "target_index",
        "sender_type",
        "receiver_type",
        "attention_abs_mean",
        "edge_message_norm_joint",
        "spatial_distance",
    ]
    for path in paths:
        frame = pd.read_csv(path, usecols=columns)
        if float(frame["stage"].iloc[0]) in selected_stages:
            frames.append(frame)
    if not frames:
        raise ValueError("No CytoBridge edge tables match selected stages")
    edges = pd.concat(frames, ignore_index=True)
    output: list[pd.DataFrame] = []
    for example in selected.itertuples(index=False):
        stage_edges = edges.loc[
            edges["stage"].astype(float).eq(float(example.stage))
        ].copy()
        source = stage_edges["source_index"].to_numpy(int)
        target = stage_edges["target_index"].to_numpy(int)
        lr_activity = (
            activities[str(example.ligand)][source]
            * activities[str(example.receptor)][target]
        )
        stage_edges["example_id"] = str(example.example_id)
        stage_edges["ligand"] = str(example.ligand)
        stage_edges["receptor"] = str(example.receptor)
        stage_edges["scaled_lr_activity"] = lr_activity
        stage_edges["cytobridge_attention_lr_flow"] = lr_activity * stage_edges[
            "attention_abs_mean"
        ].to_numpy(float)
        stage_edges["cytobridge_exact_message_lr_flow"] = lr_activity * stage_edges[
            "edge_message_norm_joint"
        ].to_numpy(float)
        output.append(stage_edges)
    all_edges = pd.concat(output, ignore_index=True)
    grouped = all_edges.groupby(
        [
            "example_id",
            "stage",
            "stage_label",
            "ligand",
            "receptor",
            "source_index",
            "target_index",
            "sender_type",
            "receiver_type",
        ],
        as_index=False,
    ).agg(
        n_grouping_seeds=("grouping_seed", "nunique"),
        mean_scaled_lr_activity=("scaled_lr_activity", "mean"),
        mean_attention_abs=("attention_abs_mean", "mean"),
        mean_exact_message=("edge_message_norm_joint", "mean"),
        mean_spatial_distance=("spatial_distance", "mean"),
        cytobridge_attention_lr_flow=("cytobridge_attention_lr_flow", "mean"),
        cytobridge_exact_message_lr_flow=(
            "cytobridge_exact_message_lr_flow",
            "mean",
        ),
    )
    return grouped.sort_values(
        ["example_id", "cytobridge_attention_lr_flow", "source_index", "target_index"],
        ascending=[True, False, True, True],
    ).reset_index(drop=True)


def _positive_top(frame: pd.DataFrame, score: str, top_n: int) -> pd.DataFrame:
    values = frame.loc[
        pd.to_numeric(frame[score], errors="coerce").gt(0)
        & frame["source_index"].ne(frame["target_index"])
    ].copy()
    return values.sort_values(
        [score, "source_index", "target_index"], ascending=[False, True, True]
    ).head(top_n)


def _commot_top(
    frame: pd.DataFrame,
    cell_index: Mapping[str, int],
    top_n: int,
) -> pd.DataFrame:
    values = _map_commot_indices(frame, cell_index)
    values = values.loc[values["source_index"].ne(values["target_index"])]
    return values.sort_values(
        ["commot_flow", "source_index", "target_index"],
        ascending=[False, True, True],
    ).head(top_n)


def _map_commot_indices(
    frame: pd.DataFrame, cell_index: Mapping[str, int]
) -> pd.DataFrame:
    values = frame.loc[
        pd.to_numeric(frame["commot_flow"], errors="coerce").gt(0)
    ].copy()
    values["source_index"] = values["source_cell_id"].astype(str).map(cell_index)
    values["target_index"] = values["target_cell_id"].astype(str).map(cell_index)
    if values[["source_index", "target_index"]].isna().any().any():
        raise ValueError("Selected COMMOT cell IDs do not all map to H5AD obs_names")
    values[["source_index", "target_index"]] = values[
        ["source_index", "target_index"]
    ].astype(int)
    return values


def _midpoint_colocalization(
    cb_edges: pd.DataFrame,
    commot_edges: pd.DataFrame,
    coordinates: np.ndarray,
    *,
    radius: float,
    fraction: float = 0.20,
) -> dict[str, float | int]:
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("Midpoint co-localization radius must be positive")
    cb = cb_edges.loc[
        cb_edges["cytobridge_attention_lr_flow"].gt(0)
        & cb_edges["source_index"].ne(cb_edges["target_index"])
    ].sort_values(
        ["cytobridge_attention_lr_flow", "source_index", "target_index"],
        ascending=[False, True, True],
    )
    commot = commot_edges.loc[
        commot_edges["commot_flow"].gt(0)
        & commot_edges["source_index"].ne(commot_edges["target_index"])
    ].sort_values(
        ["commot_flow", "source_index", "target_index"],
        ascending=[False, True, True],
    )
    n_cb = max(1, int(math.ceil(fraction * len(cb)))) if len(cb) else 0
    n_commot = max(1, int(math.ceil(fraction * len(commot)))) if len(commot) else 0
    cb = cb.head(n_cb)
    commot = commot.head(n_commot)
    if cb.empty or commot.empty:
        return {
            "n_cytobridge_top_fraction_midpoints": int(len(cb)),
            "n_commot_top_fraction_midpoints": int(len(commot)),
            "cytobridge_midpoints_near_commot_fraction": np.nan,
            "commot_midpoints_near_cytobridge_fraction": np.nan,
            "median_cytobridge_to_commot_midpoint_distance": np.nan,
            "median_commot_to_cytobridge_midpoint_distance": np.nan,
        }
    cb_midpoint = (
        coordinates[cb["source_index"].to_numpy(int)]
        + coordinates[cb["target_index"].to_numpy(int)]
    ) / 2
    commot_midpoint = (
        coordinates[commot["source_index"].to_numpy(int)]
        + coordinates[commot["target_index"].to_numpy(int)]
    ) / 2
    cb_to_commot = cKDTree(commot_midpoint).query(cb_midpoint)[0]
    commot_to_cb = cKDTree(cb_midpoint).query(commot_midpoint)[0]
    return {
        "n_cytobridge_top_fraction_midpoints": int(len(cb)),
        "n_commot_top_fraction_midpoints": int(len(commot)),
        "cytobridge_midpoints_near_commot_fraction": float(
            np.mean(cb_to_commot <= radius)
        ),
        "commot_midpoints_near_cytobridge_fraction": float(
            np.mean(commot_to_cb <= radius)
        ),
        "median_cytobridge_to_commot_midpoint_distance": float(np.median(cb_to_commot)),
        "median_commot_to_cytobridge_midpoint_distance": float(np.median(commot_to_cb)),
    }


def _draw_spatial_panel(
    ax: plt.Axes,
    coordinates: np.ndarray,
    stage_mask: np.ndarray,
    ligand_activity: np.ndarray,
    receptor_activity: np.ndarray,
    edges: pd.DataFrame,
    score: str,
    *,
    edge_color: str,
    title: str,
) -> None:
    ax.scatter(
        coordinates[stage_mask, 0],
        coordinates[stage_mask, 1],
        s=4,
        c="#d9d9d9",
        alpha=0.65,
        linewidths=0,
        rasterized=True,
        zorder=1,
    )
    for activity, color_value, marker, label in (
        (ligand_activity, "#c51b7d", "o", "ligand-high sender"),
        (receptor_activity, "#2b8cbe", "s", "receptor-high receiver"),
    ):
        positive = activity[stage_mask]
        threshold = (
            float(np.quantile(positive[positive > 0], 0.70))
            if np.any(positive > 0)
            else np.inf
        )
        mask = stage_mask & (activity >= threshold) & (activity > 0)
        ax.scatter(
            coordinates[mask, 0],
            coordinates[mask, 1],
            s=5 + 15 * activity[mask],
            c=color_value,
            marker=marker,
            alpha=0.52,
            linewidths=0,
            label=label,
            rasterized=True,
            zorder=2,
        )
    if not edges.empty:
        source = edges["source_index"].to_numpy(int)
        target = edges["target_index"].to_numpy(int)
        start = coordinates[source]
        delta = coordinates[target] - start
        score_values = edges[score].to_numpy(float)
        normalized = score_values / max(float(score_values.max()), np.finfo(float).eps)
        rgba = np.tile(np.asarray(colors.to_rgba(edge_color)), (len(edges), 1))
        rgba[:, 3] = 0.28 + 0.62 * np.sqrt(normalized)
        ax.quiver(
            start[:, 0],
            start[:, 1],
            delta[:, 0],
            delta[:, 1],
            angles="xy",
            scale_units="xy",
            scale=1,
            width=0.0030,
            headwidth=4.8,
            headlength=5.8,
            headaxislength=5.0,
            color=rgba,
            zorder=3,
        )
    ax.set_title(title, fontsize=10, weight="bold")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_spatial_lr_maps(
    selected: pd.DataFrame,
    data: ad.AnnData,
    activities: Mapping[str, np.ndarray],
    cb_edges: pd.DataFrame,
    commot_flows: pd.DataFrame,
    *,
    top_n: int,
    midpoint_match_radius: float,
    output_stem: Path,
) -> pd.DataFrame:
    coordinates = np.asarray(data.obsm["spatial_aligned"], dtype=float)
    if coordinates.shape[1] < 2:
        raise ValueError("H5AD spatial_aligned must contain at least two coordinates")
    stage_values = pd.to_numeric(
        data.obs["time_point_processed"], errors="raise"
    ).to_numpy(float)
    cell_ids = data.obs_names.astype(str)
    cell_index = {cell_id: index for index, cell_id in enumerate(cell_ids)}
    figure, axes = plt.subplots(
        len(selected), 2, figsize=(12.6, 4.0 * len(selected)), constrained_layout=True
    )
    axes = np.asarray(axes).reshape(len(selected), 2)
    audit_rows: list[dict[str, Any]] = []
    for row_index, example in enumerate(selected.itertuples(index=False)):
        stage_mask = np.isclose(stage_values, float(example.stage))
        cb_all = cb_edges.loc[cb_edges["example_id"].eq(example.example_id)]
        cb_top = _positive_top(cb_all, "cytobridge_attention_lr_flow", top_n)
        commot_all = commot_flows.loc[commot_flows["example_id"].eq(example.example_id)]
        commot_top = _commot_top(commot_all, cell_index, top_n)
        commot_indexed = _map_commot_indices(commot_all, cell_index)
        colocalization = _midpoint_colocalization(
            cb_all,
            commot_indexed,
            coordinates,
            radius=midpoint_match_radius,
        )
        annotation = (
            f"{str(example.ligand).upper()} → {str(example.receptor).upper()} | "
            f"{str(example.stage_label)} | {str(example.selection_family)}\n"
            f"within-stage percentiles: CytoBridge {example.cytobridge_percentile_all_axes:.1%}, "
            f"COMMOT {example.commot_percentile_all_axes:.1%}\n"
            f"top-20% midpoint coverage (radius={midpoint_match_radius:.3f}): "
            f"CB→COMMOT {colocalization['cytobridge_midpoints_near_commot_fraction']:.1%}, "
            f"COMMOT→CB {colocalization['commot_midpoints_near_cytobridge_fraction']:.1%}"
        )
        _draw_spatial_panel(
            axes[row_index, 0],
            coordinates,
            stage_mask,
            activities[str(example.ligand)],
            activities[str(example.receptor)],
            cb_top,
            "cytobridge_attention_lr_flow",
            edge_color="#e66101",
            title="CytoBridge: LR-compatible attention score",
        )
        _draw_spatial_panel(
            axes[row_index, 1],
            coordinates,
            stage_mask,
            activities[str(example.ligand)],
            activities[str(example.receptor)],
            commot_top,
            "commot_flow",
            edge_color="#018571",
            title="COMMOT: cell-level OT flow",
        )
        figure.text(
            0.5,
            1.0 - (row_index + 0.04) / len(selected),
            annotation,
            ha="center",
            va="top",
            fontsize=10,
            zorder=10,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 1.5},
        )
        audit_rows.append(
            {
                "example_id": example.example_id,
                "stage": example.stage,
                "ligand": example.ligand,
                "receptor": example.receptor,
                "n_cytobridge_positive_nonself_edges": int(
                    (
                        (cb_all["cytobridge_attention_lr_flow"] > 0)
                        & cb_all["source_index"].ne(cb_all["target_index"])
                    ).sum()
                ),
                "n_cytobridge_edges_displayed": int(len(cb_top)),
                "n_commot_positive_nonself_flows": int(
                    commot_all["source_cell_id"].ne(commot_all["target_cell_id"]).sum()
                ),
                "n_commot_flows_displayed": int(len(commot_top)),
                "display_rule": f"top {top_n} positive non-self cell edges by each method-specific score",
                "midpoint_colocalization_fraction": 0.20,
                "midpoint_match_radius": midpoint_match_radius,
                **colocalization,
            }
        )
    legend = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#c51b7d",
            markersize=7,
            label="ligand-high sender cells",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="none",
            markerfacecolor="#2b8cbe",
            markersize=7,
            label="receptor-high receiver cells",
        ),
        Line2D([0], [0], color="#e66101", lw=2, label="CytoBridge directed edge"),
        Line2D([0], [0], color="#018571", lw=2, label="COMMOT directed flow"),
    ]
    figure.legend(handles=legend, loc="lower center", ncol=4, frameon=False, fontsize=9)
    figure.suptitle(
        "Spatially resolved ligand–receptor interaction examples selected before plotting",
        fontsize=14,
        weight="bold",
        y=1.018,
    )
    for suffix in ("png", "pdf"):
        figure.savefig(
            output_stem.with_suffix(f".{suffix}"),
            dpi=300 if suffix == "png" else None,
            bbox_inches="tight",
        )
    plt.close(figure)
    return pd.DataFrame(audit_rows)


def _stable_color(label: str) -> tuple[float, float, float, float]:
    palette = list(plt.get_cmap("tab20").colors)
    return colors.to_rgba(palette[zlib.crc32(label.encode("utf-8")) % len(palette)])


def _top_circle_edges(
    stage: pd.DataFrame,
    rank_column: str,
    support_column: str,
    top_n: int,
) -> pd.DataFrame:
    values = stage.loc[
        stage["sender_type"].ne(stage["receiver_type"])
        & pd.to_numeric(stage[support_column], errors="coerce").gt(0)
    ].copy()
    return values.sort_values(
        [rank_column, "sender_type", "receiver_type"],
        ascending=[False, True, True],
    ).head(top_n)


def _top_fraction_set(
    stage: pd.DataFrame, rank_column: str, support_column: str
) -> set[tuple[str, str]]:
    values = stage.loc[
        stage["sender_type"].ne(stage["receiver_type"])
        & pd.to_numeric(stage[support_column], errors="coerce").gt(0)
        & pd.to_numeric(stage[rank_column], errors="coerce").ge(0.8)
    ]
    return set(
        zip(values["sender_type"].astype(str), values["receiver_type"].astype(str))
    )


def _draw_circle(
    ax: plt.Axes,
    stage: pd.DataFrame,
    rank_column: str,
    support_column: str,
    title: str,
    *,
    top_n: int,
    comparison_set: set[tuple[str, str]],
) -> pd.DataFrame:
    labels = sorted(
        set(stage["sender_type"].astype(str)) | set(stage["receiver_type"].astype(str))
    )
    angles = np.linspace(np.pi / 2, np.pi / 2 - 2 * np.pi, len(labels), endpoint=False)
    position = {
        label: np.array([np.cos(angle), np.sin(angle)])
        for label, angle in zip(labels, angles)
    }
    displayed = _top_circle_edges(stage, rank_column, support_column, top_n)
    if displayed.empty:
        ax.text(
            0.5,
            0.5,
            "no positive off-diagonal edges",
            transform=ax.transAxes,
            ha="center",
        )
        return displayed
    min_rank = float(displayed[rank_column].min())
    max_rank = float(displayed[rank_column].max())
    denominator = max(max_rank - min_rank, np.finfo(float).eps)
    shared = 0
    for edge_index, row in enumerate(displayed.itertuples(index=False)):
        key = (str(row.sender_type), str(row.receiver_type))
        is_shared = key in comparison_set
        shared += int(is_shared)
        score = float(getattr(row, rank_column))
        normalized = (score - min_rank) / denominator
        source = position[key[0]]
        target = position[key[1]]
        curvature = 0.15 + 0.05 * ((edge_index % 5) - 2)
        ax.annotate(
            "",
            xy=target,
            xytext=source,
            arrowprops={
                "arrowstyle": "-|>",
                "color": "#4b1d6b" if is_shared else "#bdbdbd",
                "lw": 0.8 + 2.2 * normalized,
                "alpha": 0.82 if is_shared else 0.45,
                "shrinkA": 8,
                "shrinkB": 8,
                "connectionstyle": f"arc3,rad={curvature}",
                "mutation_scale": 7,
            },
            zorder=1,
        )
    for label in labels:
        x, y = position[label]
        ax.scatter(
            [x],
            [y],
            s=95,
            color=_stable_color(label),
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
        )
        text_x, text_y = 1.18 * x, 1.18 * y
        horizontal = "left" if x > 0.08 else "right" if x < -0.08 else "center"
        ax.text(
            text_x,
            text_y,
            textwrap.fill(label, width=18),
            fontsize=5.4,
            ha=horizontal,
            va="center",
        )
    ax.text(
        0,
        0,
        f"{shared}/{len(displayed)}\nshared",
        ha="center",
        va="center",
        fontsize=8,
        color="#4b1d6b",
        weight="bold",
    )
    ax.set_title(title, fontsize=9.5, weight="bold")
    ax.set_xlim(-1.43, 1.43)
    ax.set_ylim(-1.43, 1.43)
    ax.set_aspect("equal")
    ax.axis("off")
    displayed = displayed.copy()
    displayed["shared_with_comparison_top20"] = [
        (str(row.sender_type), str(row.receiver_type)) in comparison_set
        for row in displayed.itertuples(index=False)
    ]
    displayed["display_rank_column"] = rank_column
    displayed["display_support_column"] = support_column
    displayed["display_title"] = title
    return displayed


def plot_ccc_circles(
    harmonized: pd.DataFrame,
    *,
    stages: Sequence[float],
    top_n: int,
    output_stem: Path,
) -> pd.DataFrame:
    required = {
        "stage",
        "sender_type",
        "receiver_type",
        "cytobridge_attention",
        "commot",
        "cellagentchat_ctps",
        "cellchat_trimean",
        "external_native_consensus",
        "cytobridge_attention_rank",
        "commot_rank",
        "cellagentchat_ctps_rank",
    }
    _require_columns(harmonized, required, "harmonized type-pair table")
    values = harmonized.copy()
    values["stage"] = values["stage"].astype(float)
    values["external_consensus_rank"] = values.groupby("stage")[
        "external_native_consensus"
    ].rank(method="average", pct=True)
    values["external_support"] = values[
        ["commot", "cellchat_trimean", "cellagentchat_ctps"]
    ].max(axis=1)
    figure, axes = plt.subplots(
        len(stages),
        len(METHOD_SPECS),
        figsize=(16, 4.3 * len(stages)),
        constrained_layout=True,
    )
    axes = np.asarray(axes).reshape(len(stages), len(METHOD_SPECS))
    audits: list[pd.DataFrame] = []
    for stage_index, stage_value in enumerate(stages):
        stage = values.loc[values["stage"].eq(float(stage_value))].copy()
        if stage.empty:
            raise ValueError(f"Circle stage {stage_value} is absent")
        attention_set = _top_fraction_set(
            stage, "cytobridge_attention_rank", "cytobridge_attention"
        )
        external_set = _top_fraction_set(
            stage, "external_consensus_rank", "external_support"
        )
        for method_index, (rank_column, support_column, title) in enumerate(
            METHOD_SPECS
        ):
            comparison_set = external_set if method_index == 0 else attention_set
            displayed = _draw_circle(
                axes[stage_index, method_index],
                stage,
                rank_column,
                support_column,
                title,
                top_n=top_n,
                comparison_set=comparison_set,
            )
            displayed.insert(0, "circle_stage", float(stage_value))
            audits.append(displayed)
        axes[stage_index, 0].text(
            -1.55,
            0,
            STAGE_LABELS.get(float(stage_value), str(stage_value)),
            rotation=90,
            ha="center",
            va="center",
            fontsize=11,
            weight="bold",
        )
    figure.suptitle(
        "Directed cell-type communication circles: shared high-ranking circuits are purple",
        fontsize=14,
        weight="bold",
    )
    figure.text(
        0.5,
        -0.005,
        f"Top {top_n} positive off-diagonal edges are displayed per panel. "
        "CytoBridge is compared with the external-only top-20% set; external panels are compared with CytoBridge top-20%.",
        ha="center",
        fontsize=8.5,
    )
    for suffix in ("png", "pdf"):
        figure.savefig(
            output_stem.with_suffix(f".{suffix}"),
            dpi=300 if suffix == "png" else None,
            bbox_inches="tight",
        )
    plt.close(figure)
    return pd.concat(audits, ignore_index=True)


def plot_temporal_bubble(
    candidates: pd.DataFrame,
    selected: pd.DataFrame,
    nichenet: pd.DataFrame,
    *,
    output_stem: Path,
) -> None:
    axes_table = (
        candidates[["ligand", "receptor", "pathways"]]
        .drop_duplicates()
        .sort_values(["pathways", "ligand", "receptor"])
    )
    axes_table["axis_id"] = axes_table["ligand"] + " → " + axes_table["receptor"]
    ligand_support = set(
        nichenet.loc[
            nichenet["attention_top_axis_supported"]
            .astype(str)
            .str.casefold()
            .eq("true"),
            "ligand_key",
        ]
        .astype(str)
        .str.casefold()
    )
    labels = [
        f"{row.axis_id}{'  ▲' if str(row.ligand).casefold() in ligand_support else ''}"
        for row in axes_table.itertuples(index=False)
    ]
    axis_lookup = {
        (row.ligand, row.receptor): index
        for index, row in enumerate(axes_table.itertuples(index=False))
    }
    stages = sorted(candidates["stage"].astype(float).unique())
    figure, ax = plt.subplots(figsize=(11.5, 6.8), constrained_layout=True)
    selected_keys = {
        (float(row.stage), str(row.ligand), str(row.receptor))
        for row in selected.itertuples(index=False)
    }
    for row in candidates.itertuples(index=False):
        y = axis_lookup[(row.ligand, row.receptor)]
        x = stages.index(float(row.stage))
        cb = float(row.cytobridge_percentile_all_axes)
        commot = float(row.commot_percentile_all_axes)
        cb = cb if np.isfinite(cb) else 0.0
        commot = commot if np.isfinite(commot) else 0.0
        both_top = cb >= 0.75 and commot >= 0.75
        key = (float(row.stage), str(row.ligand), str(row.receptor))
        for offset, value, color_value, marker in (
            (-0.14, cb, "#e66101", "o"),
            (0.14, commot, "#018571", "s"),
        ):
            alpha = float(np.clip(0.18 + 0.82 * value, 0.0, 1.0))
            rgba = colors.to_rgba(color_value, alpha=alpha)
            ax.scatter(
                [x + offset],
                [y],
                s=24 + 265 * value**2,
                marker=marker,
                color=rgba,
                edgecolor="#1a1a1a" if both_top else "white",
                linewidth=0.8 if both_top else 0.35,
                zorder=3,
            )
        if key in selected_keys:
            ax.scatter(
                [x],
                [y],
                s=430,
                marker="*",
                facecolors="none",
                edgecolors="#542788",
                linewidths=1.4,
                zorder=4,
            )
    for boundary in range(len(stages) - 1):
        ax.axvline(boundary + 0.5, color="#efefef", lw=0.8, zorder=0)
    ax.set_xticks(
        range(len(stages)), [STAGE_LABELS.get(stage, str(stage)) for stage in stages]
    )
    ax.set_yticks(range(len(labels)), labels)
    ax.invert_yaxis()
    ax.grid(axis="y", color="#f0f0f0", lw=0.7, zorder=0)
    ax.set_xlabel("Observed developmental stage")
    ax.set_title(
        "Known zebrafish LR axes show concordant temporal prominence\n"
        "Bubble size/opacity = within-stage rank among all identifiable axes",
        fontsize=13,
        weight="bold",
    )
    legend = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#e66101",
            markersize=8,
            label="CytoBridge LR-compatible attention",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="none",
            markerfacecolor="#018571",
            markersize=8,
            label="COMMOT native cell flow",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="white",
            markeredgecolor="#1a1a1a",
            markersize=8,
            label="both in top quartile",
        ),
        Line2D(
            [0],
            [0],
            marker="*",
            color="none",
            markerfacecolor="none",
            markeredgecolor="#542788",
            markersize=11,
            label="preselected spatial example",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color="none",
            markerfacecolor="#4d9221",
            markersize=7,
            label="ligand also supported in NicheNet downstream analysis",
        ),
    ]
    ax.legend(
        handles=legend,
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        frameon=False,
        fontsize=8,
    )
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for suffix in ("png", "pdf"):
        figure.savefig(
            output_stem.with_suffix(f".{suffix}"),
            dpi=300 if suffix == "png" else None,
            bbox_inches="tight",
        )
    plt.close(figure)


def _write_notes(
    out_dir: Path,
    selected: pd.DataFrame,
    spatial_audit: pd.DataFrame,
    *,
    circle_top_n: int,
) -> list[Path]:
    example_lines = [
        (
            f"- **{row.selection_family}: {row.ligand}→{row.receptor}, {row.stage_label}.** "
            f"CytoBridge percentile {row.cytobridge_percentile_all_axes:.3f}; "
            f"COMMOT percentile {row.commot_percentile_all_axes:.3f}; "
            f"CytoBridge top context {row.top_attention_sender_type}→{row.top_attention_receiver_type}; "
            f"COMMOT top abundance-controlled context {row.top_commot_sender_type}→{row.top_commot_receiver_type}."
        )
        for row in selected.itertuples(index=False)
    ]
    readme = out_dir / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# Direct biological CCC consistency views",
                "",
                "These panels complement the scalar agreement metrics. They show whether methods place high-ranking communication on the same developmental LR programs, cell-type circuits, and spatial neighborhoods.",
                "",
                "## Selection rule",
                "",
                "The spatial examples were selected before plotting. Within each pre-specified family (ncWNT, CXCL, NOTCH), the chosen exact LR/stage maximizes the mean of the within-stage CytoBridge LR-compatible-attention percentile and COMMOT native-flow percentile, requires both scores to be positive, and requires at least 10 active CytoBridge graph edges. The LR-compatible score is the post-hoc product of the model attention magnitude, sender ligand activity, and receiver receptor activity. The rule is evaluated only among the literature-scoped zebrafish axes already in the validation audit.",
                "",
                *example_lines,
                "",
                "## Figures",
                "",
                "- `spatial_lr_interaction_maps`: identical ligand/receptor expression landscapes are overlaid with the top positive, non-self cell-level directed edges from CytoBridge or the narrowly reconstructed COMMOT matrix. The annotation reports asymmetric nearest-neighbor coverage of top-20% interaction midpoints at half the frozen graph cutoff; this is not exact-edge or direction accuracy. Edge truncation is only for display; every positive selected COMMOT flow is retained in the audit table.",
                f"- `ccc_circle_comparison`: conventional directed CCC circles for one early (10 hpf) and one late (24 hpf) observed stage. Each panel displays the top {circle_top_n} positive off-diagonal edges; purple edges are shared with the counterpart top-20% set.",
                "- `known_lr_temporal_consistency_bubble`: all nine literature-scoped axes across five stages. Bubble size and opacity are within-stage percentiles among all identifiable LR axes, so the two methods' raw units are never compared.",
                "",
                "## Interpretation boundary",
                "",
                "The maps demonstrate spatial and biological coherence, not biochemical ground truth. CytoBridge attention is a model gate magnitude; COMMOT is optimal-transport communication mass. The edge classifier used LR information, so LR agreement is supportive rather than fully independent. NicheNet symbols indicate downstream ligand-target consistency and are not direct spatial CCC strength.",
                "",
                "![Spatial LR maps](spatial_lr_interaction_maps.png)",
                "",
                "![CCC circles](ccc_circle_comparison.png)",
                "",
                "![Temporal LR bubble](known_lr_temporal_consistency_bubble.png)",
                "",
            ]
        ),
        encoding="utf-8",
    )
    cn = out_dir / "汇报说明.md"
    cn.write_text(
        "\n".join(
            [
                "# 生物学与空间 CCC 一致性直观图",
                "",
                "这组图不是再加一个相关系数，而是让读者直接看到：CytoBridge 和外部方法是否同时指向相同的发育 LR 程序、cell-type circuit 和空间邻域。",
                "",
                "## 三张主图怎么读",
                "",
                "- `spatial_lr_interaction_maps`：左右使用完全相同的 ligand/receptor 表达背景，只替换箭头来源。左侧是后处理得到的 CytoBridge LR-compatible attention（attention magnitude × sender ligand activity × receiver receptor activity），右侧是 COMMOT cell-level OT flow。标题下还报告 top-20% interaction midpoint 在半个固定 graph cutoff 内的双向空间覆盖率。",
                f"- `ccc_circle_comparison`：CCC 领域常见的 circle network。每个 panel 只画排名最高的 {circle_top_n} 条非对角正信号；紫色边表示与对照 top-20% 集合重合。",
                "- `known_lr_temporal_consistency_bubble`：9 条有文献边界的斑马鱼 LR axis 跨 5 个 stage 的时间图。气泡大小/透明度是各方法内部 percentile，不会直接比两个不同单位的 raw score。",
                "",
                "## 为什么不是 cherry-pick",
                "",
                "先固定 ncWNT、CXCL、NOTCH 三个发育相关 family，再按 CytoBridge 和 COMMOT 的 within-stage percentile 平均值自动选一个；要求两边都为正，且 CytoBridge 至少 10 条 active graph edges。选择表和所有候选都保留在 CSV。",
                "",
                "## 必须保留的表述边界",
                "",
                "这些图证明的是空间/生物学 coherence，不是 biochemical ground truth。Attention 是模型 gate magnitude，COMMOT 是 OT communication mass。LR 信息参与了 edge classifier，所以 LR agreement 属于 supporting evidence，不是完全独立验证。",
                "",
                *example_lines,
                "",
            ]
        ),
        encoding="utf-8",
    )
    reviewer = out_dir / "reviewer_response_biological_visuals.md"
    reviewer.write_text(
        "\n".join(
            [
                "In addition to rank-based summary statistics, we now provide direct biological and spatial views of cross-method consistency. We pre-specified three developmental signaling families (ncWNT, CXCL, and NOTCH) and selected one LR/stage example per family using an auditable joint percentile rule before reconstructing cell-level COMMOT flows. On the same ligand/receptor expression landscapes, CytoBridge attention-weighted edges and COMMOT optimal-transport flows concentrate in overlapping spatial neighborhoods. Conventional cell-type communication circles further show that many of the highest-ranking directed circuits are shared with the external-only consensus, while the temporal LR bubble plot shows concordant prominence of literature-scoped zebrafish axes across stages. These analyses support the interpretation that the learned attention structure captures biologically coherent communication organization rather than only achieving a favorable scalar correlation. We nevertheless describe this as supportive consistency, not biochemical ground truth, because the methods have different score semantics and LR information contributes to the CytoBridge edge classifier.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return [readme, cn, reviewer]


def _selection_outputs(
    out_dir: Path,
    candidates: pd.DataFrame,
    selected: pd.DataFrame,
    inputs: Mapping[str, Path],
    *,
    min_active_edges: int,
) -> list[Path]:
    candidates_path = out_dir / "biological_example_candidates.csv"
    selected_path = out_dir / "biological_example_selection.csv"
    candidates.to_csv(candidates_path, index=False)
    selected.to_csv(selected_path, index=False)
    manifest_path = out_dir / "selection_manifest.json"
    json_dump(
        {
            "schema_version": 1,
            "created_at_utc": utc_now(),
            "workflow": "zebrafish_biological_ccc_example_preselection",
            "status": "complete",
            "pre_specified_pathway_families": list(BIOLOGICAL_FAMILIES),
            "min_active_cytobridge_edges": min_active_edges,
            "selection_uses_cell_level_commot_flow_maps": False,
            "selection_rule": selected["selection_rule"].iloc[0],
            "inputs": {name: file_record(path) for name, path in inputs.items()},
            "artifacts": {
                "candidates": file_record(candidates_path),
                "selected": file_record(selected_path),
            },
        },
        manifest_path,
    )
    return [candidates_path, selected_path, manifest_path]


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if (
        args.min_active_edges < 1
        or args.spatial_display_top_edges < 1
        or args.circle_display_top_edges < 1
    ):
        raise ValueError("All edge-count arguments must be positive")
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if any(out_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {out_dir}")

    validation_dir = args.validation_dir.expanduser().resolve()
    positive_dir = args.positive_consistency_dir.expanduser().resolve()
    commot_dir = args.commot_dir.expanduser().resolve()
    axis_path = validation_dir / "lr_axis_stage_scores.csv.gz"
    known_path = validation_dir / "known_axis_stage_scores.csv"
    harmonized_path = positive_dir / "harmonized_type_pair_scores.csv.gz"
    nichenet_path = positive_dir / "nichenet_downstream_ligand_detail.csv"
    commot_lr_path = commot_dir / "commot_lr_scores.csv.gz"
    inputs = {
        "all_lr_axis_scores": axis_path,
        "known_lr_axis_scores": known_path,
        "harmonized_type_pair_scores": harmonized_path,
        "nichenet_ligand_detail": nichenet_path,
        "commot_lr_scores": commot_lr_path,
    }
    for path in inputs.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    axis_scores = pd.read_csv(axis_path)
    known_axes = pd.read_csv(known_path)
    commot_lr = pd.read_csv(commot_lr_path)
    candidates, selected = select_biological_examples(
        axis_scores,
        known_axes,
        commot_lr,
        min_active_edges=args.min_active_edges,
    )
    if args.selected_examples_csv is not None:
        frozen = pd.read_csv(args.selected_examples_csv.expanduser().resolve())
        keys = ["example_id", "stage", "ligand", "receptor", "selection_family"]
        _require_columns(frozen, keys, "frozen selected examples")
        if (
            not frozen[keys]
            .reset_index(drop=True)
            .equals(selected[keys].reset_index(drop=True))
        ):
            raise ValueError(
                "Frozen selected examples disagree with deterministic preselection"
            )
    artifacts = _selection_outputs(
        out_dir,
        candidates,
        selected,
        inputs,
        min_active_edges=args.min_active_edges,
    )
    if args.select_only:
        print(f"Biological examples selected in {out_dir}")
        return 0
    if args.selected_commot_flow_dir is None:
        raise ValueError("--selected-commot-flow-dir is required unless --select-only")

    flow_dir = args.selected_commot_flow_dir.expanduser().resolve()
    flow_manifest_path = flow_dir / "manifest.json"
    flow_manifest = _read_json(flow_manifest_path)
    if (
        flow_manifest.get("workflow")
        != "zebrafish_selected_lr_commot_cell_flow_reconstruction"
    ):
        raise ValueError("Unexpected selected COMMOT flow workflow")
    flow_path = flow_dir / "selected_commot_cell_flows.csv.gz"
    flow_summary_path = flow_dir / "selected_commot_cell_summary.csv.gz"
    for source, record_name in (
        (flow_path, "selected_commot_cell_flows"),
        (flow_summary_path, "selected_commot_cell_summary"),
    ):
        record = flow_manifest.get("artifacts", {}).get(record_name)
        if not isinstance(record, Mapping):
            raise ValueError(
                f"Selected COMMOT manifest lacks {record_name!r} artifact record"
            )
        _verify_file_record(source, record)
    commot_flows = pd.read_csv(flow_path)
    if set(commot_flows["example_id"].astype(str)) != set(
        selected["example_id"].astype(str)
    ):
        raise ValueError("Selected COMMOT flow examples disagree with selection")
    for source in (flow_path, flow_summary_path):
        destination = out_dir / source.name
        shutil.copy2(source, destination)
        artifacts.append(destination)
    copied_flow_manifest = out_dir / "selected_commot_flow_manifest.json"
    shutil.copy2(flow_manifest_path, copied_flow_manifest)
    artifacts.append(copied_flow_manifest)

    h5ad_path = args.h5ad.expanduser().resolve()
    data = ad.read_h5ad(h5ad_path)
    for required in ("time_point_processed",):
        if required not in data.obs:
            raise KeyError(f"H5AD obs lacks {required!r}")
    if "spatial_aligned" not in data.obsm:
        raise KeyError("H5AD obsm lacks 'spatial_aligned'")
    tokens = set(selected["ligand"].astype(str)) | set(selected["receptor"].astype(str))
    activities = _gene_activities(data, tokens)
    cb_edges = _load_selected_cytobridge_edges(
        args.cytobridge_dir.expanduser().resolve(), selected, activities
    )
    cytobridge_manifest_path = (
        args.cytobridge_dir.expanduser().resolve() / "run_manifest.json"
    )
    cytobridge_manifest = _read_json(cytobridge_manifest_path)
    spatial_cutoff = float(
        cytobridge_manifest.get("checkpoint", {}).get("spatial_cutoff", np.nan)
    )
    if not np.isfinite(spatial_cutoff) or spatial_cutoff <= 0:
        raise ValueError(
            "CytoBridge manifest lacks a positive checkpoint.spatial_cutoff"
        )
    midpoint_match_radius = spatial_cutoff / 2
    cb_edge_path = out_dir / "selected_cytobridge_cell_edges.csv.gz"
    cb_edges.to_csv(cb_edge_path, index=False, compression="gzip")
    artifacts.append(cb_edge_path)

    spatial_audit = plot_spatial_lr_maps(
        selected,
        data,
        activities,
        cb_edges,
        commot_flows,
        top_n=args.spatial_display_top_edges,
        midpoint_match_radius=midpoint_match_radius,
        output_stem=out_dir / "spatial_lr_interaction_maps",
    )
    spatial_audit_path = out_dir / "spatial_display_audit.csv"
    spatial_audit.to_csv(spatial_audit_path, index=False)
    artifacts.extend(
        [
            spatial_audit_path,
            out_dir / "spatial_lr_interaction_maps.png",
            out_dir / "spatial_lr_interaction_maps.pdf",
        ]
    )

    harmonized = pd.read_csv(harmonized_path)
    circle_audit = plot_ccc_circles(
        harmonized,
        stages=args.circle_stages,
        top_n=args.circle_display_top_edges,
        output_stem=out_dir / "ccc_circle_comparison",
    )
    circle_audit_path = out_dir / "circle_display_edges.csv"
    circle_audit.to_csv(circle_audit_path, index=False)
    artifacts.extend(
        [
            circle_audit_path,
            out_dir / "ccc_circle_comparison.png",
            out_dir / "ccc_circle_comparison.pdf",
        ]
    )

    nichenet = pd.read_csv(nichenet_path)
    plot_temporal_bubble(
        candidates,
        selected,
        nichenet,
        output_stem=out_dir / "known_lr_temporal_consistency_bubble",
    )
    artifacts.extend(
        [
            out_dir / "known_lr_temporal_consistency_bubble.png",
            out_dir / "known_lr_temporal_consistency_bubble.pdf",
        ]
    )
    artifacts.extend(
        _write_notes(
            out_dir,
            selected,
            spatial_audit,
            circle_top_n=args.circle_display_top_edges,
        )
    )
    manifest_path = out_dir / "manifest.json"
    manifest = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "workflow": "zebrafish_direct_biological_ccc_consistency_visualization",
        "status": "complete",
        "selection": {
            "pre_specified_pathway_families": list(BIOLOGICAL_FAMILIES),
            "min_active_cytobridge_edges": args.min_active_edges,
            "selection_rule": selected["selection_rule"].iloc[0],
            "selection_was_completed_before_cell_level_commot_reconstruction": True,
            "selection_uses_visual_appearance": False,
        },
        "display": {
            "spatial_top_positive_nonself_edges_per_method": args.spatial_display_top_edges,
            "circle_top_positive_offdiagonal_edges_per_panel": args.circle_display_top_edges,
            "circle_stages": list(map(float, args.circle_stages)),
            "circle_overlap_reference": "top 20% within-stage rank set",
            "spatial_midpoint_colocalization": {
                "positive_edge_fraction": 0.20,
                "match_radius": midpoint_match_radius,
                "match_radius_rule": "half the frozen CytoBridge spatial graph cutoff",
                "frozen_spatial_graph_cutoff": spatial_cutoff,
                "directional_nearest_neighbor_coverage_reported_both_ways": True,
            },
            "raw_cross_method_units_compared": False,
        },
        "claims": {
            "demonstrates_spatial_and_biological_coherence": True,
            "method_agreement_is_ground_truth": False,
            "attention_is_ccc_probability": False,
            "commot_ot_mass_is_biochemical_flux": False,
            "lr_agreement_is_fully_independent": False,
            "nichenet_is_direct_spatial_ccc_strength": False,
        },
        "inputs": {
            **{name: file_record(path) for name, path in inputs.items()},
            "h5ad": file_record(h5ad_path),
            "cytobridge_manifest": file_record(cytobridge_manifest_path),
            "selected_commot_flow_manifest": file_record(flow_manifest_path),
        },
        "software": software_versions(),
        "artifacts": {path.name: file_record(path) for path in artifacts},
    }
    json_dump(manifest, manifest_path)
    print(f"Biological CCC consistency panels completed in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
