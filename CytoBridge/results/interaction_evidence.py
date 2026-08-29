"""Results for the LR-prior ablation and stVCR comparison."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ._io import read_manifest, require_files, resolve_results_dir


DATASET_ORDER = ("zebrafish", "mosta", "arista", "admouse", "chicken_heart")
DATASET_LABELS = {
    "zebrafish": "Zebrafish",
    "mosta": "MOSTA",
    "arista": "ARISTA",
    "admouse": "AD mouse",
    "chicken_heart": "Chicken\nheart",
}
SPACE_ORDER = ("joint", "spatial", "state")
SPACE_LABELS = {"joint": "Joint", "spatial": "Spatial", "state": "Gene state"}
NO_LR_TARGETS = {
    "zebrafish": {1, 2, 3, 4},
    "mosta": {1, 2, 3},
    "arista": {1, 2, 3, 4},
    "admouse": {1, 2},
    "chicken_heart": {1, 2, 3},
}
LOTO_TARGETS = {
    "zebrafish": {1, 2, 3},
    "mosta": {1, 2},
    "arista": {1, 2, 3},
    "admouse": {1},
    "chicken_heart": {1, 2},
}
CYTOBRIDGE_COLUMN = "CytoBridge-0.015"
NO_LR_COMPARISON = "No LR prior minus full model"
EXTERNAL_COMPARISON = "stVCR minus CytoBridge"


@dataclass(frozen=True)
class LRPriorStVCRResults:
    """Paired errors and summaries used by the S39 comparison."""

    source_dir: Path
    manifest: dict[str, Any]
    no_lr: pd.DataFrame
    stvcr: pd.DataFrame
    panel_summary: pd.DataFrame


def _require_columns(frame: pd.DataFrame, columns: set[str], source: Path) -> None:
    missing = sorted(columns.difference(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing columns: {missing}")


def _sem(values: pd.Series | np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    if array.size <= 1:
        return float("nan")
    return float(np.std(array, ddof=1) / math.sqrt(array.size))


def _sort_table(table: pd.DataFrame) -> pd.DataFrame:
    result = table.copy()
    result["__dataset"] = result["dataset"].map(
        {name: index for index, name in enumerate(DATASET_ORDER)}
    )
    result["__space"] = result["space"].map(
        {name: index for index, name in enumerate(SPACE_ORDER)}
    )
    return (
        result.sort_values(["__dataset", "target", "__space"], kind="mergesort")
        .drop(columns=["__dataset", "__space"])
        .reset_index(drop=True)
    )


def _validate_categories(table: pd.DataFrame, source: Path) -> None:
    unknown_datasets = sorted(
        set(table["dataset"].astype(str)).difference(DATASET_ORDER)
    )
    if unknown_datasets:
        raise ValueError(f"{source} contains unknown datasets: {unknown_datasets}")
    unknown_spaces = sorted(set(table["space"].astype(str)).difference(SPACE_ORDER))
    if unknown_spaces:
        raise ValueError(
            f"{source} contains unknown evaluation spaces: {unknown_spaces}"
        )


def _validate_keys(
    table: pd.DataFrame,
    *,
    targets: dict[str, set[int]],
    source: Path,
) -> None:
    keys = ["dataset", "target", "space"]
    if table.duplicated(keys).any():
        raise ValueError(f"{source} contains duplicate dataset-target-space rows")
    observed = {
        (str(row.dataset), int(row.target), str(row.space))
        for row in table.itertuples(index=False)
    }
    expected = {
        (dataset, target, space)
        for dataset, dataset_targets in targets.items()
        for target in dataset_targets
        for space in SPACE_ORDER
    }
    if observed != expected:
        missing = sorted(expected.difference(observed))[:5]
        extra = sorted(observed.difference(expected))[:5]
        raise ValueError(
            f"{source} has incomplete row keys; missing={missing}, extra={extra}"
        )


def _numeric_values(
    table: pd.DataFrame, columns: list[str], source: Path
) -> np.ndarray:
    values = table[columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError(f"{source} contains non-finite values in {columns}")
    return values


def _validate_no_lr(table: pd.DataFrame, source: Path) -> pd.DataFrame:
    columns = [
        "dataset",
        "target",
        "space",
        "metric",
        "full",
        "no_lr_prior",
        "no_lr_prior_minus_full",
        "no_lr_prior_relative_to_full",
    ]
    _require_columns(table, set(columns), source)
    result = table[columns].copy()
    _validate_categories(result, source)
    _validate_keys(result, targets=NO_LR_TARGETS, source=source)
    if set(result["metric"].astype(str)) != {"sliced_w2"}:
        raise ValueError(f"{source} must contain only the sliced_w2 metric")
    numeric = _numeric_values(result, columns[4:], source)
    full, no_lr = numeric[:, 0], numeric[:, 1]
    if (full <= 0).any() or (no_lr <= 0).any():
        raise ValueError(f"{source} contains non-positive arm means")
    if not np.allclose(no_lr - full, numeric[:, 2], rtol=1e-12, atol=1e-15):
        raise ValueError(f"{source} contains inconsistent Full/No-LR differences")
    if not np.allclose((no_lr - full) / full, numeric[:, 3], rtol=1e-12, atol=1e-15):
        raise ValueError(f"{source} contains inconsistent Full/No-LR ratios")
    return _sort_table(result)


def _validate_stvcr(table: pd.DataFrame, source: Path) -> pd.DataFrame:
    columns = [
        "dataset",
        "target",
        "space",
        CYTOBRIDGE_COLUMN,
        "stVCR",
        "stvcr_minus_cytobridge",
        "stvcr_relative_to_cytobridge",
    ]
    _require_columns(table, set(columns), source)
    result = table[columns].copy()
    _validate_categories(result, source)
    _validate_keys(result, targets=LOTO_TARGETS, source=source)
    numeric = _numeric_values(result, columns[3:7], source)
    cytobridge, stvcr = numeric[:, 0], numeric[:, 1]
    if (cytobridge <= 0).any() or (stvcr <= 0).any():
        raise ValueError(f"{source} contains non-positive method means")
    if not np.allclose(stvcr - cytobridge, numeric[:, 2], rtol=1e-12, atol=1e-15):
        raise ValueError(f"{source} contains inconsistent method differences")
    if not np.allclose(
        (stvcr - cytobridge) / cytobridge,
        numeric[:, 3],
        rtol=1e-12,
        atol=1e-15,
    ):
        raise ValueError(f"{source} contains inconsistent method ratios")
    return _sort_table(result)


def build_lr_prior_stvcr_panel_summary(
    no_lr: pd.DataFrame,
    stvcr: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate the dataset-space means and target-stage standard errors."""

    def summarize(
        table: pd.DataFrame,
        *,
        value_column: str,
        comparison: str,
    ) -> pd.DataFrame:
        summary = (
            table.groupby(["dataset", "space"], sort=False)[value_column]
            .agg(n_targets="size", mean_relative_difference="mean", sem=_sem)
            .reset_index()
        )
        dataset_means = table.groupby("dataset", sort=False)[value_column].mean()
        summary["dataset_mean_relative_difference"] = summary["dataset"].map(
            dataset_means
        )
        summary.insert(0, "comparison", comparison)
        return summary

    summary = pd.concat(
        [
            summarize(
                no_lr,
                value_column="no_lr_prior_relative_to_full",
                comparison=NO_LR_COMPARISON,
            ),
            summarize(
                stvcr,
                value_column="stvcr_relative_to_cytobridge",
                comparison=EXTERNAL_COMPARISON,
            ),
        ],
        ignore_index=True,
    )
    if (
        len(summary) != 30
        or summary.duplicated(["comparison", "dataset", "space"]).any()
    ):
        raise ValueError("Interaction panel summary must contain 30 unique rows")
    return summary


def load_lr_prior_stvcr_results(
    results_dir: str | Path | None = None,
) -> LRPriorStVCRResults:
    """Load paired reconstruction errors and calculate the panel summary.

    Parameters
    ----------
    results_dir
        Directory containing the two compact CSV files. Packaged example data
        are used when this argument is omitted.
    """

    source_dir = resolve_results_dir(results_dir, slug="interaction_evidence")
    paths = require_files(
        source_dir,
        ("no_lr_paired_target_deltas.csv", "stvcr_paired_target_deltas.csv"),
    )
    no_lr = _validate_no_lr(
        pd.read_csv(
            paths["no_lr_paired_target_deltas.csv"], float_precision="round_trip"
        ),
        paths["no_lr_paired_target_deltas.csv"],
    )
    stvcr = _validate_stvcr(
        pd.read_csv(
            paths["stvcr_paired_target_deltas.csv"], float_precision="round_trip"
        ),
        paths["stvcr_paired_target_deltas.csv"],
    )
    return LRPriorStVCRResults(
        source_dir=source_dir,
        manifest=read_manifest(source_dir),
        no_lr=no_lr,
        stvcr=stvcr,
        panel_summary=build_lr_prior_stvcr_panel_summary(no_lr, stvcr),
    )


def lr_prior_stvcr_statistics(
    results: LRPriorStVCRResults,
) -> dict[str, float | int]:
    """Return selected values from the calculated panel data."""

    no_lr_means = results.no_lr.groupby("dataset", sort=False)[
        "no_lr_prior_relative_to_full"
    ].mean()
    return {
        "matched_cells": int(len(results.no_lr)),
        "external_comparison_cells": int(len(results.stvcr)),
        "panel_summary_rows": int(len(results.panel_summary)),
        "no_lr_mean_relative_change_arista": float(no_lr_means["arista"]),
        "no_lr_mean_relative_change_chicken_heart": float(no_lr_means["chicken_heart"]),
    }


def _write_csv(table: pd.DataFrame, path: Path) -> None:
    table.to_csv(
        path,
        index=False,
        float_format="%.17g",
        na_rep="",
        lineterminator="\n",
    )


def write_lr_prior_stvcr_tables(
    results: LRPriorStVCRResults,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write the paired inputs and calculated panel summary."""

    output = Path(output_dir)
    paths = {
        "no_lr": output / "no_lr_paired_target_deltas.csv",
        "external_comparison": output / "stvcr_paired_target_deltas.csv",
        "panel_summary": output / "panel_summary.csv",
    }
    _write_csv(results.no_lr, paths["no_lr"])
    _write_csv(results.stvcr, paths["external_comparison"])
    _write_csv(results.panel_summary, paths["panel_summary"])
    return paths


def plot_lr_prior_stvcr(
    results: LRPriorStVCRResults,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """Draw the S39 comparison as PDF and PNG."""

    from ._interaction_evidence_plot import plot_lr_prior_stvcr as _plot

    return _plot(results, output_dir)


# Compatibility names retained for code written before CytoBridge 1.5.
InteractionEvidenceResults = LRPriorStVCRResults
build_interaction_evidence_panel_summary = build_lr_prior_stvcr_panel_summary
load_interaction_evidence_results = load_lr_prior_stvcr_results
interaction_evidence_statistics = lr_prior_stvcr_statistics
write_interaction_evidence_tables = write_lr_prior_stvcr_tables
plot_interaction_evidence = plot_lr_prior_stvcr
