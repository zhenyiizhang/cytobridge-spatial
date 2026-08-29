"""Full-model training time and peak-memory measurements."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ._io import read_json, require_files, resolve_results_dir


DATASET_ORDER = (
    "admouse",
    "arista",
    "chicken_heart",
    "mosta",
    "zebrafish",
)
DATASET_LABELS = {
    "admouse": "AD mouse",
    "arista": "ARISTA",
    "chicken_heart": "Chicken heart",
    "mosta": "MOSTA",
    "zebrafish": "Zebrafish",
}
RAW_COLUMNS = (
    "dataset",
    "time_points_used_for_training",
    "training_time_point_labels",
    "observed_cells_or_spots",
    "training_time_seconds",
    "peak_host_memory_mib",
    "peak_gpu_allocation_mib",
)
DISPLAY_COLUMNS = (
    "Dataset",
    "Time points used for training",
    "Observed cells/spots",
    "Training time (min)",
    "Peak host memory (GiB)",
    "Peak GPU allocation (GiB)",
)

_MEASUREMENT_CONTRACT = {
    "row_definition": "one measured full-model run per dataset",
    "training_stages": 6,
    "hardware": {
        "gpu_count": 1,
        "gpu_model": "NVIDIA GeForce RTX 4090 D",
    },
    "training_time": {
        "field": "training_time_seconds",
        "unit": "s",
        "scope": "TrainingPipeline.train",
        "includes": [
            "stage preparation",
            "optimizer setup",
            "epochs",
            "checkpoint selection",
            "checkpoint writing",
        ],
        "excludes": [
            "preprocessing",
            "prediction",
            "evaluation",
            "downstream analysis",
            "AnnData serialization",
        ],
    },
    "host_memory": {
        "field": "peak_host_memory_mib",
        "unit": "MiB",
        "scope": "process-lifetime maximum resident set size sampled after training",
    },
    "gpu_memory": {
        "field": "peak_gpu_allocation_mib",
        "unit": "MiB",
        "scope": "maximum PyTorch allocation across training stages",
    },
    "run_summary": "single measured runs; no averaging across runs",
}
_DISPLAY_CONTRACT = {
    "training_time": {"unit": "min", "divisor": 60, "decimal_places": 1},
    "memory": {"unit": "GiB", "divisor": 1024, "decimal_places": 2},
}


@dataclass(frozen=True)
class FullModelComputeCostResults:
    """Raw measurements and metadata for the full-model compute-cost table."""

    source_dir: Path
    manifest: dict[str, Any]
    measurements: pd.DataFrame


def _validate_manifest(manifest: dict[str, Any], source: Path) -> None:
    if manifest.get("schema_version") != 1:
        raise ValueError(f"{source} has an unsupported schema version")
    if manifest.get("analysis") != "full_model_compute_cost":
        raise ValueError(f"{source} does not describe full-model compute cost")
    if manifest.get("manuscript_table") != "Supplementary Table 2":
        raise ValueError(f"{source} has an unexpected manuscript table")
    if manifest.get("measurement") != _MEASUREMENT_CONTRACT:
        raise ValueError(f"{source} has an unexpected measurement contract")
    if manifest.get("display") != _DISPLAY_CONTRACT:
        raise ValueError(f"{source} has an unexpected display contract")


def _integer_column(table: pd.DataFrame, name: str, source: Path) -> pd.Series:
    values = pd.to_numeric(table[name], errors="coerce")
    array = values.to_numpy(dtype=float)
    if not np.isfinite(array).all() or not np.equal(array, np.floor(array)).all():
        raise ValueError(f"{source} contains invalid integer values in {name}")
    if not values.gt(0).all():
        raise ValueError(f"{source} contains non-positive values in {name}")
    return values.astype(int)


def _positive_float_column(
    table: pd.DataFrame,
    name: str,
    source: Path,
) -> pd.Series:
    values = pd.to_numeric(table[name], errors="coerce")
    array = values.to_numpy(dtype=float)
    if not np.isfinite(array).all():
        raise ValueError(f"{source} contains non-finite values in {name}")
    if not values.gt(0).all():
        raise ValueError(f"{source} contains non-positive values in {name}")
    return values.astype(float)


def _validate_measurements(table: pd.DataFrame, source: Path) -> pd.DataFrame:
    missing = sorted(set(RAW_COLUMNS).difference(table.columns))
    if missing:
        raise ValueError(f"{source} is missing columns: {missing}")

    result = table.loc[:, RAW_COLUMNS].copy()
    if result["dataset"].isna().any():
        raise ValueError(f"{source} contains missing dataset names")
    result["dataset"] = result["dataset"].astype(str).str.strip()
    if result["dataset"].duplicated().any():
        raise ValueError(f"{source} contains duplicate datasets")
    if set(result["dataset"]) != set(DATASET_ORDER):
        raise ValueError(f"{source} does not match the five-dataset roster")

    result["time_points_used_for_training"] = _integer_column(
        result,
        "time_points_used_for_training",
        source,
    )
    result["observed_cells_or_spots"] = _integer_column(
        result,
        "observed_cells_or_spots",
        source,
    )
    for name in (
        "training_time_seconds",
        "peak_host_memory_mib",
        "peak_gpu_allocation_mib",
    ):
        result[name] = _positive_float_column(result, name, source)

    labels = result["training_time_point_labels"]
    if labels.isna().any():
        raise ValueError(f"{source} contains missing training time-point labels")
    result["training_time_point_labels"] = labels.astype(str).str.strip()
    if result["training_time_point_labels"].eq("").any():
        raise ValueError(f"{source} contains empty training time-point labels")
    label_counts = result["training_time_point_labels"].map(
        lambda value: len([part for part in value.split(",") if part.strip()])
    )
    if not label_counts.equals(result["time_points_used_for_training"]):
        raise ValueError(
            f"{source} has time-point counts that do not match the label lists"
        )

    order = {dataset: index for index, dataset in enumerate(DATASET_ORDER)}
    return (
        result.assign(_order=result["dataset"].map(order))
        .sort_values("_order", kind="mergesort")
        .drop(columns="_order")
        .reset_index(drop=True)
    )


def load_full_model_compute_cost(
    results_dir: str | Path | None = None,
) -> FullModelComputeCostResults:
    """Load the raw full-model compute-cost measurements.

    Parameters
    ----------
    results_dir
        Directory containing ``full_model_compute_cost.csv`` and
        ``manifest.json``. Packaged data are used when omitted.
    """

    source_dir = resolve_results_dir(results_dir, slug="full_model_compute_cost")
    paths = require_files(
        source_dir,
        ("full_model_compute_cost.csv", "manifest.json"),
    )
    manifest = read_json(paths["manifest.json"])
    _validate_manifest(manifest, paths["manifest.json"])
    measurements = _validate_measurements(
        pd.read_csv(
            paths["full_model_compute_cost.csv"],
            float_precision="round_trip",
        ),
        paths["full_model_compute_cost.csv"],
    )
    return FullModelComputeCostResults(
        source_dir=source_dir,
        manifest=manifest,
        measurements=measurements,
    )


def format_full_model_compute_cost(
    results: FullModelComputeCostResults,
) -> pd.DataFrame:
    """Return the six display columns used in Supplementary Table 2."""

    raw = results.measurements
    return pd.DataFrame(
        {
            "Dataset": raw["dataset"].map(DATASET_LABELS),
            "Time points used for training": [
                f"{count}: {labels}"
                for count, labels in zip(
                    raw["time_points_used_for_training"],
                    raw["training_time_point_labels"],
                )
            ],
            "Observed cells/spots": raw["observed_cells_or_spots"].map(
                lambda value: f"{int(value):,}"
            ),
            "Training time (min)": raw["training_time_seconds"].map(
                lambda value: f"{float(value) / 60:.1f}"
            ),
            "Peak host memory (GiB)": raw["peak_host_memory_mib"].map(
                lambda value: f"{float(value) / 1024:.2f}"
            ),
            "Peak GPU allocation (GiB)": raw["peak_gpu_allocation_mib"].map(
                lambda value: f"{float(value) / 1024:.2f}"
            ),
        },
        columns=DISPLAY_COLUMNS,
    )


def _markdown_table(table: pd.DataFrame) -> str:
    headers = [str(column) for column in table.columns]
    alignments = [":---", ":---", "---:", "---:", "---:", "---:"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(alignments) + " |",
    ]
    for row in table.itertuples(index=False, name=None):
        values = [str(value).replace("|", "\\|") for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def write_full_model_compute_cost_tables(
    results: FullModelComputeCostResults,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write the raw, display, and Markdown forms of the table."""

    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    raw_path = output / "full_model_compute_cost.csv"
    display_path = output / "full_model_compute_cost_table.csv"
    markdown_path = output / "full_model_compute_cost_table.md"

    results.measurements.to_csv(raw_path, index=False)
    display = format_full_model_compute_cost(results)
    display.to_csv(display_path, index=False)
    markdown_path.write_text(_markdown_table(display), encoding="utf-8")
    return {
        "raw": raw_path,
        "display": display_path,
        "markdown": markdown_path,
    }
