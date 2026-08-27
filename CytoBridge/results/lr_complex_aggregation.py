"""Processed results for ligand-receptor complex aggregation sensitivity."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ._io import read_manifest, require_files, resolve_results_dir


DATASET_ORDER = ("zebrafish", "mosta", "arista", "chicken_heart")
DATASET_LABELS = {
    "zebrafish": "Zebrafish",
    "mosta": "MOSTA",
    "arista": "ARISTA",
    "chicken_heart": "Chicken Heart",
}
DATASET_LABEL_ORDER = tuple(DATASET_LABELS[name] for name in DATASET_ORDER)
SCOPE_ORDER = ("all_scored_pairs", "multisubunit_pairs")

_REQUIRED_COLUMNS = {
    "time",
    "pair",
    "score_min",
    "score_geometric_mean",
    "is_multisubunit",
}


@dataclass(frozen=True)
class LRComplexAggregationResults:
    """Paired LR scores and the tables calculated for the figure."""

    source_dir: Path
    manifest: dict[str, Any]
    paired_scores: pd.DataFrame
    per_time_summary: pd.DataFrame
    dataset_summary: pd.DataFrame


def _read_boolean(series: pd.Series, source: Path) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    values = normalized.map({"true": True, "false": False})
    if values.isna().any():
        invalid = sorted(normalized.loc[values.isna()].unique())[:5]
        raise ValueError(f"{source} has invalid is_multisubunit values: {invalid}")
    return values.astype(bool)


def _validate_table(table: pd.DataFrame, *, dataset: str, source: Path) -> pd.DataFrame:
    missing = sorted(_REQUIRED_COLUMNS.difference(table.columns))
    if missing:
        raise ValueError(f"{source} is missing columns: {missing}")

    result = table.copy()
    result["time"] = pd.to_numeric(result["time"], errors="coerce")
    for column in ("score_min", "score_geometric_mean"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    numeric = result[["time", "score_min", "score_geometric_mean"]].to_numpy(
        dtype=float
    )
    if not np.isfinite(numeric).all():
        raise ValueError(f"{source} contains non-finite time or score values")
    if (result[["score_min", "score_geometric_mean"]] < 0).any().any():
        raise ValueError(f"{source} contains negative LR scores")

    result["pair"] = result["pair"].astype(str).str.strip()
    if result["pair"].eq("").any():
        raise ValueError(f"{source} contains an empty LR pair name")
    result["is_multisubunit"] = _read_boolean(result["is_multisubunit"], source)
    result.insert(0, "dataset", dataset)

    if result.duplicated(["dataset", "time", "pair"]).any():
        raise ValueError(f"{source} contains duplicate dataset-time-pair rows")
    inconsistent = (
        result.groupby(["dataset", "pair"], sort=False)["is_multisubunit"]
        .nunique(dropna=False)
        .loc[lambda values: values > 1]
    )
    if not inconsistent.empty:
        examples = [pair for _, pair in inconsistent.index[:5]]
        raise ValueError(
            f"{source} changes is_multisubunit within an LR pair: {examples}"
        )
    if result["time"].nunique() < 2:
        raise ValueError(f"{source} must contain at least two time points")
    return result


def _safe_spearman(first: np.ndarray, second: np.ndarray) -> float:
    if first.size < 2 or np.all(first == first[0]) or np.all(second == second[0]):
        return float("nan")
    return float(spearmanr(first, second).statistic)


def _top_set(table: pd.DataFrame, column: str, top_n: int) -> set[str]:
    ordered = table.sort_values(
        [column, "pair"],
        ascending=[False, True],
        kind="mergesort",
    )
    return set(ordered.head(top_n)["pair"].astype(str))


def _top_overlap(table: pd.DataFrame) -> tuple[int, int, float]:
    n_pairs = int(len(table))
    if n_pairs == 0:
        return 0, 0, float("nan")
    top_n = min(n_pairs, max(1, min(10, math.ceil(0.2 * n_pairs))))
    minimum = _top_set(table, "score_min", top_n)
    geometric = _top_set(table, "score_geometric_mean", top_n)
    overlap = minimum.intersection(geometric)
    union = minimum.union(geometric)
    return top_n, len(overlap), float(len(overlap) / len(union))


def summarize_lr_complex_aggregation(
    paired_scores: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate per-time and per-dataset LR aggregation summaries."""

    per_time_rows: list[dict[str, object]] = []
    dataset_rows: list[dict[str, object]] = []
    for dataset in DATASET_ORDER:
        dataset_table = paired_scores.loc[paired_scores["dataset"].eq(dataset)]
        if dataset_table.empty:
            raise ValueError(f"Paired LR scores are missing dataset {dataset!r}")
        times = np.sort(dataset_table["time"].unique().astype(float))
        time_min = float(times.min())
        time_range = float(times.max() - time_min)
        if time_range <= 0:
            raise ValueError(f"Dataset {dataset!r} needs more than one distinct time")

        for time_value in times:
            current = dataset_table.loc[dataset_table["time"].eq(time_value)]
            scopes = (
                ("all_scored_pairs", current),
                (
                    "multisubunit_pairs",
                    current.loc[current["is_multisubunit"]],
                ),
            )
            for scope, subset in scopes:
                first = subset["score_min"].to_numpy(dtype=float)
                second = subset["score_geometric_mean"].to_numpy(dtype=float)
                top_n, overlap_n, jaccard = _top_overlap(subset)
                per_time_rows.append(
                    {
                        "dataset": DATASET_LABELS[dataset],
                        "time": float(time_value),
                        "normalized_time": float((time_value - time_min) / time_range),
                        "scope": scope,
                        "n_pairs": int(len(subset)),
                        "spearman": _safe_spearman(first, second),
                        "top_n": top_n,
                        "top_overlap_n": overlap_n,
                        "top_jaccard": jaccard,
                    }
                )

        pooled_minimum = dataset_table["score_min"].to_numpy(dtype=float)
        pooled_geometric = dataset_table["score_geometric_mean"].to_numpy(dtype=float)
        dataset_rows.append(
            {
                "dataset": DATASET_LABELS[dataset],
                "n_scored_pairs": int(dataset_table["pair"].nunique()),
                "n_multisubunit_pairs": int(
                    dataset_table.loc[
                        dataset_table["is_multisubunit"], "pair"
                    ].nunique()
                ),
                "pooled_spearman": _safe_spearman(
                    pooled_minimum,
                    pooled_geometric,
                ),
            }
        )

    per_time = pd.DataFrame(per_time_rows)
    dataset_summary = pd.DataFrame(dataset_rows)
    all_scope = per_time.loc[per_time["scope"].eq("all_scored_pairs")]
    minimum_rank = all_scope.groupby("dataset", sort=False)["spearman"].min()
    minimum_jaccard = all_scope.groupby("dataset", sort=False)["top_jaccard"].min()
    dataset_summary["min_per_time_spearman"] = dataset_summary["dataset"].map(
        minimum_rank
    )
    dataset_summary["min_top10_jaccard"] = dataset_summary["dataset"].map(
        minimum_jaccard
    )
    return per_time, dataset_summary


def load_lr_complex_aggregation_results(
    results_dir: str | Path | None = None,
) -> LRComplexAggregationResults:
    """Load paired LR scores and calculate the figure tables.

    Parameters
    ----------
    results_dir
        Directory containing one ``paired_scores.csv`` file in each dataset
        subdirectory. Packaged data are used when this argument is omitted.
    """

    source_dir = resolve_results_dir(results_dir, slug="lr_complex_aggregation")
    relative_names = tuple(f"{dataset}/paired_scores.csv" for dataset in DATASET_ORDER)
    paths = require_files(source_dir, relative_names)
    tables = []
    for dataset in DATASET_ORDER:
        relative_name = f"{dataset}/paired_scores.csv"
        table = pd.read_csv(paths[relative_name], float_precision="round_trip")
        tables.append(
            _validate_table(table, dataset=dataset, source=paths[relative_name])
        )
    paired_scores = pd.concat(tables, ignore_index=True)
    if paired_scores.duplicated(["dataset", "time", "pair"]).any():
        raise ValueError("Paired LR scores contain duplicate dataset-time-pair rows")
    per_time, dataset_summary = summarize_lr_complex_aggregation(paired_scores)
    return LRComplexAggregationResults(
        source_dir=source_dir,
        manifest=read_manifest(source_dir),
        paired_scores=paired_scores,
        per_time_summary=per_time,
        dataset_summary=dataset_summary,
    )


def _write_csv(table: pd.DataFrame, path: Path) -> None:
    table.to_csv(
        path,
        index=False,
        float_format="%.17g",
        na_rep="",
        lineterminator="\n",
    )


def write_lr_complex_aggregation_tables(
    results: LRComplexAggregationResults,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write the calculated per-time and per-dataset tables."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "per_time": output / "lr_complex_aggregation_per_time.csv",
        "dataset_summary": output / "lr_complex_aggregation_dataset_summary.csv",
    }
    _write_csv(results.per_time_summary, paths["per_time"])
    _write_csv(results.dataset_summary, paths["dataset_summary"])
    return paths


def plot_lr_complex_aggregation(
    results: LRComplexAggregationResults,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """Render the LR-complex aggregation figure as PDF and PNG."""

    from ._lr_complex_aggregation_plot import plot_lr_complex_aggregation as _plot

    return _plot(results, output_dir)
