"""Shared contracts for five-dataset spatial communication consistency.

The analysis compares directed sender-cell-type to receiver-cell-type rankings.
Native method scores are deliberately never pooled because CytoBridge exact
messages, attention gates, CellChat probabilities, COMMOT transport mass,
CellAgentChat CTPS, and NicheNet ligand activity do not share a unit.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse, stats


ANALYSIS_SEED = 20260816
TERMINAL_SAMPLE_N = 3000
TOP_FRACTION = 0.20

# Frozen before the five-dataset result matrix was produced.  These thresholds
# decide main-figure inclusion only; every attempted method remains in the
# complete audit tables regardless of outcome.
MAIN_FIGURE_GATE = {
    "minimum_valid_datasets": 4,
    "minimum_positive_datasets": 4,
    "minimum_median_spearman_rho": 0.20,
    "minimum_median_top_fraction_jaccard": 0.15,
    "primary_cytobridge_view": "CytoBridge exact message",
}

FORMAL_DATASET_CONTRACTS: dict[str, dict[str, object]] = {
    "zebrafish": {
        "display_name": "Zebrafish",
        "cell_type_key": "Annotation",
        "time_key": "time_point_processed",
        "counts_layer": "counts",
        "normalization_target_sum": 1105.0,
        "terminal_time": 4.0,
        "previous_time": 3.0,
        "cellchat_species": "zebrafish",
        "database_file": "CellChatDB.ligrec.zebrafish.csv",
        "database_scope": "species-matched zebrafish CellChatDB",
    },
    "mosta": {
        "display_name": "MOSTA",
        "cell_type_key": "Annotation",
        "time_key": "time_point_processed",
        "counts_layer": "count",
        "normalization_target_sum": 10000.0,
        "terminal_time": 3.0,
        "previous_time": 2.0,
        "cellchat_species": "mouse",
        "database_file": "CellChatDB.ligrec.mouse.csv",
        "database_scope": "species-matched mouse CellChatDB",
    },
    "arista": {
        "display_name": "ARISTA",
        "cell_type_key": "Annotation",
        "time_key": "time_point_processed",
        "counts_layer": "counts",
        "normalization_target_sum": 2841.0,
        "terminal_time": 4.0,
        "previous_time": 3.0,
        "cellchat_species": "human",
        "database_file": "CellChatDB.ligrec.human.csv",
        "database_scope": "species-matched human CellChatDB",
    },
    "admouse": {
        "display_name": "AdMouse",
        "cell_type_key": "major_annotation",
        "time_key": "time_point_processed",
        "counts_layer": "counts",
        "normalization_target_sum": 10000.0,
        "terminal_time": 2.0,
        "previous_time": 1.0,
        "cellchat_species": "mouse",
        "database_file": "CellChatDB.ligrec.mouse.csv",
        "database_scope": "species-matched mouse CellChatDB; seven complete LR pairs in the 347-gene panel",
    },
    "chicken_heart": {
        "display_name": "Chicken heart",
        "cell_type_key": "celltype_prediction",
        "time_key": "time_point_processed",
        "counts_layer": "counts",
        "normalization_target_sum": 9743.5,
        "terminal_time": 3.0,
        "previous_time": 2.0,
        "cellchat_species": "human",
        "database_file": "CellChatDB.ligrec.human.csv",
        "database_scope": "human conserved-symbol proxy; not a species-complete Gallus gallus screen",
    },
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _matrix_values(matrix: object) -> np.ndarray:
    return np.asarray(matrix.data if sparse.issparse(matrix) else matrix).ravel()


def stratified_sample_indices(
    labels: Iterable[object], *, total: int, seed: int
) -> np.ndarray:
    """Select a deterministic near-proportional sample retaining every type."""

    values = np.asarray([str(value) for value in labels], dtype=object)
    if total <= 0:
        raise ValueError("total must be positive")
    if len(values) <= total:
        return np.arange(len(values), dtype=np.int64)
    groups, counts = np.unique(values, return_counts=True)
    if total < len(groups):
        raise ValueError("sample size is smaller than the terminal cell-type universe")
    ideal = total * counts.astype(float) / len(values)
    allocation = np.minimum(counts, np.maximum(1, np.floor(ideal).astype(int)))
    while int(allocation.sum()) < total:
        candidates = np.flatnonzero(allocation < counts)
        chosen = candidates[np.argmax(ideal[candidates] - allocation[candidates])]
        allocation[chosen] += 1
    while int(allocation.sum()) > total:
        candidates = np.flatnonzero(allocation > 1)
        chosen = candidates[np.argmax(allocation[candidates] - ideal[candidates])]
        allocation[chosen] -= 1
    rng = np.random.default_rng(int(seed))
    selected: list[np.ndarray] = []
    for group, amount in zip(groups, allocation, strict=True):
        candidates = np.flatnonzero(values == group)
        selected.append(np.sort(rng.choice(candidates, int(amount), replace=False)))
    return np.sort(np.concatenate(selected)).astype(np.int64, copy=False)


def _verify_reconstructed_x(
    data: ad.AnnData,
    rows: np.ndarray,
    *,
    counts_layer: str,
    target_sum: float,
    tolerance: float,
) -> float:
    counts = data.layers[counts_layer][rows]
    counts_csr = sparse.csr_matrix(counts, dtype=np.float64)
    raw_values = _matrix_values(counts_csr)
    if raw_values.size and (
        not np.isfinite(raw_values).all()
        or float(raw_values.min()) < 0
        or float(np.max(np.abs(raw_values - np.rint(raw_values)))) > 1e-5
    ):
        raise ValueError(
            "declared count layer is not finite nonnegative integer-like data"
        )
    library = np.asarray(counts_csr.sum(axis=1)).ravel()
    if np.any(~np.isfinite(library)) or np.any(library <= 0):
        raise ValueError("sample contains a nonpositive or nonfinite raw library size")
    reconstructed = counts_csr.multiply((float(target_sum) / library)[:, None]).tocsr()
    reconstructed.data = np.log1p(reconstructed.data)
    observed = sparse.csr_matrix(data.X[rows], dtype=np.float64)
    residual = (reconstructed - observed).tocsr()
    maximum = float(np.max(np.abs(residual.data))) if residual.nnz else 0.0
    if maximum > float(tolerance):
        raise ValueError(
            "accepted X does not match counts-derived normalize_total+log1p: "
            f"max residual={maximum:.6g}, tolerance={tolerance:.6g}"
        )
    return maximum


def prepare_shared_samples(
    input_h5ad: str | Path,
    output_dir: str | Path,
    *,
    dataset: str,
    expected_h5ad_sha256: str,
    sample_n: int = TERMINAL_SAMPLE_N,
    seed: int = ANALYSIS_SEED,
    source_x_tolerance: float = 1e-5,
) -> dict[str, object]:
    """Freeze terminal and preceding-stage cells shared by all methods.

    The terminal roster is identical in both emitted H5ADs.  The two-stage
    H5AD adds a separately stratified preceding-stage roster for NicheNet's
    receiver-response calculation; terminal pair-score methods consume only
    ``terminal_sample.h5ad``.
    """

    if dataset not in FORMAL_DATASET_CONTRACTS:
        raise KeyError(f"unknown formal spatial dataset: {dataset}")
    contract = FORMAL_DATASET_CONTRACTS[dataset]
    source = Path(input_h5ad).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"output directory exists: {output}")
    if sha256_file(source) != str(expected_h5ad_sha256).lower():
        raise ValueError("input H5AD SHA256 differs from the accepted binding")
    data = ad.read_h5ad(source)
    label_key = str(contract["cell_type_key"])
    time_key = str(contract["time_key"])
    counts_layer = str(contract["counts_layer"])
    for key in (label_key, time_key):
        if key not in data.obs:
            raise KeyError(f"accepted H5AD lacks obs[{key!r}]")
    if counts_layer not in data.layers:
        raise KeyError(f"accepted H5AD lacks layers[{counts_layer!r}]")
    for key in ("spatial_aligned", "X_latent"):
        if key not in data.obsm:
            raise KeyError(f"accepted H5AD lacks obsm[{key!r}]")
    times = pd.to_numeric(data.obs[time_key], errors="raise").to_numpy(float)
    terminal_time = float(contract["terminal_time"])
    previous_time = float(contract["previous_time"])
    labels = data.obs[label_key].astype(str).to_numpy()
    terminal_all = np.flatnonzero(np.isclose(times, terminal_time, rtol=0, atol=1e-8))
    previous_all = np.flatnonzero(np.isclose(times, previous_time, rtol=0, atol=1e-8))
    if not len(terminal_all) or not len(previous_all):
        raise ValueError("accepted H5AD lacks the declared terminal or previous stage")
    terminal_local = stratified_sample_indices(
        labels[terminal_all], total=int(sample_n), seed=int(seed)
    )
    previous_local = stratified_sample_indices(
        labels[previous_all], total=int(sample_n), seed=int(seed) + 1
    )
    terminal_rows = terminal_all[terminal_local]
    previous_rows = previous_all[previous_local]
    if len(set(data.obs_names[terminal_rows].astype(str))) != len(terminal_rows):
        raise ValueError("terminal sample has duplicate observation names")
    checked_rows = np.unique(np.concatenate((terminal_rows, previous_rows)))
    max_residual = _verify_reconstructed_x(
        data,
        checked_rows,
        counts_layer=counts_layer,
        target_sum=float(contract["normalization_target_sum"]),
        tolerance=float(source_x_tolerance),
    )
    output.mkdir(parents=True)

    terminal = data[terminal_rows].copy()
    terminal.obs["ccc_cell_type"] = terminal.obs[label_key].astype(str)
    terminal.obs["ccc_stage"] = terminal.obs[time_key].astype(float)
    terminal.obs["ccc_stage_label"] = f"terminal_{terminal_time:g}"
    terminal.obs["time_label"] = terminal.obs["ccc_stage_label"]
    terminal_path = output / "terminal_sample.h5ad"
    terminal.write_h5ad(terminal_path, compression="gzip")

    two_stage_rows = np.concatenate((previous_rows, terminal_rows))
    two_stage = data[two_stage_rows].copy()
    two_stage.obs["ccc_cell_type"] = two_stage.obs[label_key].astype(str)
    two_stage.obs["ccc_stage"] = two_stage.obs[time_key].astype(float)
    two_stage.obs["ccc_stage_label"] = np.where(
        np.isclose(two_stage.obs["ccc_stage"].astype(float), terminal_time),
        f"terminal_{terminal_time:g}",
        f"previous_{previous_time:g}",
    )
    two_stage.obs["time_label"] = two_stage.obs["ccc_stage_label"]
    two_stage_path = output / "terminal_previous_sample.h5ad"
    two_stage.write_h5ad(two_stage_path, compression="gzip")

    roster = pd.DataFrame(
        {
            "dataset": dataset,
            "stage_role": ["terminal"] * len(terminal_rows)
            + ["previous"] * len(previous_rows),
            "source_row": np.concatenate((terminal_rows, previous_rows)),
            "obs_name": np.concatenate(
                (
                    data.obs_names[terminal_rows].astype(str),
                    data.obs_names[previous_rows].astype(str),
                )
            ),
            "cell_type": np.concatenate((labels[terminal_rows], labels[previous_rows])),
            "stage": np.concatenate((times[terminal_rows], times[previous_rows])),
        }
    )
    roster_path = output / "sample_roster.csv"
    roster.to_csv(roster_path, index=False)
    manifest: dict[str, object] = {
        "schema_version": 1,
        "workflow": "five_dataset_spatial_communication_shared_sample",
        "dataset": dataset,
        "contract": contract,
        "selection": {
            "seed": int(seed),
            "maximum_cells_per_stage": int(sample_n),
            "rule": "deterministic near-proportional cell-type-stratified sample retaining every observed type",
            "terminal_cells_available": int(len(terminal_all)),
            "terminal_cells_selected": int(len(terminal_rows)),
            "previous_cells_available": int(len(previous_all)),
            "previous_cells_selected": int(len(previous_rows)),
            "terminal_cell_types": sorted(set(labels[terminal_rows])),
        },
        "expression": {
            "source": f"layers[{counts_layer!r}]",
            "transform": "normalize_total over all genes, then log1p exactly once",
            "target_sum": float(contract["normalization_target_sum"]),
            "accepted_x_reconstruction_max_abs_residual": max_residual,
            "accepted_x_reconstruction_tolerance": float(source_x_tolerance),
        },
        "source_h5ad": {
            "path": str(source),
            "sha256": str(expected_h5ad_sha256).lower(),
            "size_bytes": int(source.stat().st_size),
        },
        "artifacts": {},
    }
    for path in (terminal_path, two_stage_path, roster_path):
        manifest["artifacts"][path.name] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "size_bytes": int(path.stat().st_size),
        }
    _write_json(output / "manifest.json", manifest)
    return manifest


def rank_percentile(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="raise")
    if numeric.isna().any() or not np.isfinite(numeric.to_numpy(float)).all():
        raise ValueError("rank input contains missing or nonfinite scores")
    if len(numeric) <= 1:
        return pd.Series(np.ones(len(numeric)), index=numeric.index, dtype=float)
    return numeric.rank(method="average", pct=True)


def _positive_top_keys(
    frame: pd.DataFrame, *, score_column: str, top_fraction: float
) -> set[tuple[str, str]]:
    positive = frame.loc[pd.to_numeric(frame[score_column], errors="raise") > 0].copy()
    if positive.empty:
        return set()
    requested = max(1, int(math.ceil(len(frame) * float(top_fraction))))
    k = min(requested, len(positive))
    threshold = positive[score_column].nlargest(k).min()
    selected = positive.loc[positive[score_column] >= threshold]
    return set(
        zip(selected.sender_type.astype(str), selected.receiver_type.astype(str))
    )


def pairwise_cytobridge_metrics(
    long_scores: pd.DataFrame,
    *,
    cytobridge_views: Sequence[str] = (
        "CytoBridge exact message",
        "CytoBridge attention",
    ),
    top_fraction: float = TOP_FRACTION,
) -> pd.DataFrame:
    """Compare each external method to each CytoBridge view on shared keys."""

    required = {
        "dataset",
        "sender_type",
        "receiver_type",
        "method",
        "score",
        "available",
    }
    if not required.issubset(long_scores.columns):
        raise ValueError(
            f"score table lacks {sorted(required.difference(long_scores.columns))}"
        )
    external = sorted(set(long_scores.method.astype(str)).difference(cytobridge_views))
    rows: list[dict[str, object]] = []
    keys = ["sender_type", "receiver_type"]
    for dataset, dataset_table in long_scores.groupby("dataset", sort=True):
        for view in cytobridge_views:
            left = dataset_table.loc[
                dataset_table.method.astype(str).eq(view)
                & dataset_table.available.astype(bool),
                keys + ["score"],
            ].rename(columns={"score": "left_score"})
            if left.duplicated(keys).any():
                raise ValueError(f"duplicate {view} directed keys for {dataset}")
            for method in external:
                right = dataset_table.loc[
                    dataset_table.method.astype(str).eq(method)
                    & dataset_table.available.astype(bool),
                    keys + ["score"],
                ].rename(columns={"score": "right_score"})
                if right.duplicated(keys).any():
                    raise ValueError(f"duplicate {method} directed keys for {dataset}")
                merged = left.merge(right, on=keys, how="inner", validate="one_to_one")
                valid = (
                    len(merged) >= 4
                    and merged.left_score.nunique() > 1
                    and merged.right_score.nunique() > 1
                )
                rho = (
                    float(
                        stats.spearmanr(merged.left_score, merged.right_score).statistic
                    )
                    if valid
                    else np.nan
                )
                left_top = _positive_top_keys(
                    merged.rename(columns={"left_score": "score"}),
                    score_column="score",
                    top_fraction=top_fraction,
                )
                right_top = _positive_top_keys(
                    merged.rename(columns={"right_score": "score"}),
                    score_column="score",
                    top_fraction=top_fraction,
                )
                union = left_top | right_top
                jaccard = len(left_top & right_top) / len(union) if union else np.nan
                rows.append(
                    {
                        "dataset": str(dataset),
                        "cytobridge_view": view,
                        "external_method": method,
                        "n_shared_directed_pairs": int(len(merged)),
                        "spearman_rho": rho,
                        "top_fraction": float(top_fraction),
                        "top_left_n": int(len(left_top)),
                        "top_right_n": int(len(right_top)),
                        "top_intersection_n": int(len(left_top & right_top)),
                        "top_jaccard": float(jaccard)
                        if np.isfinite(jaccard)
                        else np.nan,
                        "metric_available": bool(valid),
                    }
                )
    return pd.DataFrame(rows)


def evaluate_main_figure_gate(
    metrics: pd.DataFrame,
    *,
    gate: Mapping[str, object] = MAIN_FIGURE_GATE,
) -> pd.DataFrame:
    """Apply the frozen cross-dataset inclusion gate without hiding failures."""

    primary = str(gate["primary_cytobridge_view"])
    selected = metrics.loc[metrics.cytobridge_view.astype(str).eq(primary)].copy()
    rows: list[dict[str, object]] = []
    for method, table in selected.groupby("external_method", sort=True):
        valid = table.loc[table.metric_available.astype(bool)].copy()
        rho = pd.to_numeric(valid.spearman_rho, errors="coerce").dropna()
        jaccard = pd.to_numeric(valid.top_jaccard, errors="coerce").dropna()
        n_valid = int(len(rho))
        n_positive = int((rho > 0).sum())
        median_rho = float(rho.median()) if len(rho) else np.nan
        median_jaccard = float(jaccard.median()) if len(jaccard) else np.nan
        checks = {
            "valid_dataset_count": n_valid >= int(gate["minimum_valid_datasets"]),
            "positive_dataset_count": n_positive
            >= int(gate["minimum_positive_datasets"]),
            "median_spearman": np.isfinite(median_rho)
            and median_rho >= float(gate["minimum_median_spearman_rho"]),
            "median_top_jaccard": np.isfinite(median_jaccard)
            and median_jaccard >= float(gate["minimum_median_top_fraction_jaccard"]),
        }
        rows.append(
            {
                "external_method": method,
                "n_valid_datasets": n_valid,
                "n_positive_spearman_datasets": n_positive,
                "median_spearman_rho": median_rho,
                "median_top_jaccard": median_jaccard,
                **{f"passes_{key}": bool(value) for key, value in checks.items()},
                "include_in_main_figure": bool(all(checks.values())),
                "decision_rule_frozen_before_results": True,
            }
        )
    return pd.DataFrame(rows)
