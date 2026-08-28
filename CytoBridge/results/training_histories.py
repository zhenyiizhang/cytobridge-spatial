"""Load and plot the six-stage training histories used in the paper."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ._io import (
    prepare_output_dir,
    read_manifest,
    require_files,
    resolve_results_dir,
)


@dataclass(frozen=True)
class TrainingStage:
    """Display and smoothing settings for one training stage."""

    stage_index: int
    stage: str
    label: str
    mode: str
    objective: str
    y_label: str
    configured_epochs: int
    smoothing_window: int
    color: str


STAGES = (
    TrainingStage(
        0,
        "Pretrain",
        "Pretrain",
        "neural_ode",
        "composite_training_loss",
        "Training loss",
        100,
        11,
        "#355C7D",
    ),
    TrainingStage(
        1,
        "Refine",
        "Refine",
        "neural_ode",
        "composite_training_loss",
        "Training loss",
        100,
        11,
        "#07838B",
    ),
    TrainingStage(
        2,
        "Init_interaction",
        "Init interaction",
        "neural_ode",
        "composite_training_loss",
        "Training loss",
        50,
        7,
        "#7B6BA8",
    ),
    TrainingStage(
        3,
        "Train_Score",
        "Train Score",
        "score_matching",
        "score_matching_loss",
        "Score loss",
        2001,
        101,
        "#C95F72",
    ),
    TrainingStage(
        4,
        "Finetune",
        "Finetune",
        "neural_ode",
        "composite_training_loss",
        "Training loss",
        1000,
        51,
        "#D58B32",
    ),
    TrainingStage(
        5,
        "Score_Refine",
        "Score Refine",
        "score_matching",
        "score_matching_loss",
        "Score loss",
        2001,
        101,
        "#8E6C4A",
    ),
)

DATASET_ORDER = (
    "Zebrafish",
    "MOSTA",
    "ARISTA",
    "AD mouse",
    "Chicken heart",
)

HISTORY_COLUMNS = ("stage_index", "stage", "epoch", "loss")
CHECKPOINT_COLUMNS = (
    "dataset",
    "stage",
    "first_checkpoint_metric",
    "selected_checkpoint_metric",
    "selected_epoch",
    "configured_epochs",
    "percent_reduction",
)
_FILES = ("arista_training_history.csv", "panel_metrics.csv", "manifest.json")


@dataclass(frozen=True)
class TrainingHistoryResults:
    """Compact tables and settings for the training-history figure."""

    source_dir: Path
    manifest: dict[str, Any]
    history: pd.DataFrame
    checkpoint_summary: pd.DataFrame


def _require_columns(
    frame: pd.DataFrame, columns: tuple[str, ...], source: Path
) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing columns: {missing}")


def _integer_column(frame: pd.DataFrame, column: str, source: Path) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="raise")
    array = values.to_numpy(dtype=float)
    if not np.isfinite(array).all() or not np.equal(array, np.floor(array)).all():
        raise ValueError(f"{source} must contain integer values in {column!r}")
    return values.astype(int)


def _validate_manifest(manifest: dict[str, Any], source: Path) -> None:
    if manifest.get("analysis") != "training_histories":
        raise ValueError(f"{source} must describe training_histories")
    if manifest.get("displayed_dataset") != "arista":
        raise ValueError(f"{source} must select the ARISTA history")
    if manifest.get("dataset_order") != list(DATASET_ORDER):
        raise ValueError(f"{source} contains an unexpected dataset order")

    history = manifest.get("history", {})
    if history.get("file") != "arista_training_history.csv":
        raise ValueError(f"{source} contains an unexpected history file")
    if history.get("columns") != list(HISTORY_COLUMNS) or history.get("rows") != 5252:
        raise ValueError(f"{source} contains an unexpected history schema")

    summary = manifest.get("checkpoint_summary", {})
    if summary.get("file") != "panel_metrics.csv":
        raise ValueError(f"{source} contains an unexpected checkpoint-summary file")
    if summary.get("columns") != list(CHECKPOINT_COLUMNS) or summary.get("rows") != 30:
        raise ValueError(f"{source} contains an unexpected checkpoint-summary schema")

    if manifest.get("stages") != [asdict(stage) for stage in STAGES]:
        raise ValueError(f"{source} contains unexpected stage settings")


def _validate_history(frame: pd.DataFrame, source: Path) -> pd.DataFrame:
    _require_columns(frame, HISTORY_COLUMNS, source)
    history = frame.loc[:, HISTORY_COLUMNS].copy()
    history["stage_index"] = _integer_column(history, "stage_index", source)
    history["epoch"] = _integer_column(history, "epoch", source)
    history["loss"] = pd.to_numeric(history["loss"], errors="raise")
    if not np.isfinite(history["loss"].to_numpy(dtype=float)).all():
        raise ValueError(f"{source} contains non-finite loss values")
    if history.duplicated(["stage_index", "stage", "epoch"]).any():
        raise ValueError(f"{source} contains duplicate stage-epoch rows")

    observed_stages = list(
        history[["stage_index", "stage"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    expected_stages = [(stage.stage_index, stage.stage) for stage in STAGES]
    if observed_stages != expected_stages:
        raise ValueError(f"{source} contains an unexpected stage order")

    ordered = history.sort_values(["stage_index", "epoch"], kind="stable")
    if not ordered.index.equals(history.index):
        raise ValueError(f"{source} must be ordered by stage and epoch")

    for stage in STAGES:
        current = history.loc[
            history["stage_index"].eq(stage.stage_index)
            & history["stage"].eq(stage.stage)
        ]
        epochs = current["epoch"].tolist()
        if epochs != list(range(1, stage.configured_epochs + 1)):
            raise ValueError(
                f"{source} contains an incomplete epoch sequence for {stage.stage}"
            )
    if len(history) != 5252:
        raise ValueError(f"{source} must contain 5,252 rows")
    return history.reset_index(drop=True)


def _expected_configured_epochs(dataset: str, stage: TrainingStage) -> int:
    if dataset == "AD mouse" and stage.stage in {"Train_Score", "Score_Refine"}:
        return 3001
    return stage.configured_epochs


def _validate_checkpoint_summary(frame: pd.DataFrame, source: Path) -> pd.DataFrame:
    _require_columns(frame, CHECKPOINT_COLUMNS, source)
    summary = frame.loc[:, CHECKPOINT_COLUMNS].copy()
    for column in (
        "first_checkpoint_metric",
        "selected_checkpoint_metric",
        "percent_reduction",
    ):
        summary[column] = pd.to_numeric(summary[column], errors="raise")
    summary["selected_epoch"] = _integer_column(summary, "selected_epoch", source)
    summary["configured_epochs"] = _integer_column(summary, "configured_epochs", source)
    numeric = summary[
        [
            "first_checkpoint_metric",
            "selected_checkpoint_metric",
            "selected_epoch",
            "configured_epochs",
            "percent_reduction",
        ]
    ].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError(f"{source} contains non-finite values")
    if summary.duplicated(["dataset", "stage"]).any():
        raise ValueError(f"{source} contains duplicate dataset-stage rows")

    expected_keys = [
        (dataset, stage.stage) for dataset in DATASET_ORDER for stage in STAGES
    ]
    observed_keys = list(
        summary[["dataset", "stage"]].itertuples(index=False, name=None)
    )
    if observed_keys != expected_keys:
        raise ValueError(f"{source} must contain the complete dataset-stage grid")

    stage_lookup = {stage.stage: stage for stage in STAGES}
    for row in summary.itertuples(index=False):
        stage = stage_lookup[str(row.stage)]
        configured = _expected_configured_epochs(str(row.dataset), stage)
        if int(row.configured_epochs) != configured:
            raise ValueError(
                f"{source} contains an unexpected epoch count for "
                f"{row.dataset}/{row.stage}"
            )
        if not 1 <= int(row.selected_epoch) <= configured:
            raise ValueError(
                f"{source} contains an invalid selected epoch for "
                f"{row.dataset}/{row.stage}"
            )

    first = summary["first_checkpoint_metric"].to_numpy(dtype=float)
    selected = summary["selected_checkpoint_metric"].to_numpy(dtype=float)
    if np.any(first == 0.0):
        raise ValueError(f"{source} cannot calculate a reduction from a zero baseline")
    calculated = 100.0 * (1.0 - selected / first)
    reported = summary["percent_reduction"].to_numpy(dtype=float)
    if not np.allclose(calculated, reported, rtol=0.0, atol=1.0e-8):
        raise ValueError(f"{source} contains inconsistent percent reductions")
    return summary.reset_index(drop=True)


def load_training_history_results(
    results_dir: str | Path | None = None,
) -> TrainingHistoryResults:
    """Load the saved training history and retained-checkpoint table.

    Parameters
    ----------
    results_dir
        Directory containing the three compact result files. Packaged data are
        used when this argument is omitted.
    """

    source_dir = resolve_results_dir(results_dir, slug="training_histories")
    paths = require_files(source_dir, _FILES)
    manifest = read_manifest(source_dir)
    _validate_manifest(manifest, paths["manifest.json"])
    history = _validate_history(
        pd.read_csv(
            paths["arista_training_history.csv"],
            float_precision="round_trip",
        ),
        paths["arista_training_history.csv"],
    )
    checkpoint_summary = _validate_checkpoint_summary(
        pd.read_csv(paths["panel_metrics.csv"], float_precision="round_trip"),
        paths["panel_metrics.csv"],
    )
    return TrainingHistoryResults(
        source_dir=source_dir,
        manifest=manifest,
        history=history,
        checkpoint_summary=checkpoint_summary,
    )


def centered_moving_mean(values: np.ndarray | pd.Series, window: int) -> np.ndarray:
    """Return the edge-padded centered moving mean used in the figure."""

    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("values must be a non-empty one-dimensional array")
    if not np.isfinite(array).all():
        raise ValueError("values must be finite")
    width = max(3, min(int(window), len(array)))
    if width % 2 == 0:
        width += 1
    if width > len(array):
        width = len(array) if len(array) % 2 == 1 else len(array) - 1
    left = width // 2
    padded = np.pad(array, (left, left), mode="edge")
    return np.convolve(padded, np.ones(width) / width, mode="valid")


def calculate_smoothed_training_history(
    results: TrainingHistoryResults,
) -> pd.DataFrame:
    """Add the stage-specific centered moving mean to the history table."""

    history = results.history.copy()
    history["smoothed_loss"] = np.nan
    for stage in STAGES:
        mask = history["stage_index"].eq(stage.stage_index) & history["stage"].eq(
            stage.stage
        )
        history.loc[mask, "smoothed_loss"] = centered_moving_mean(
            history.loc[mask, "loss"].to_numpy(dtype=float),
            stage.smoothing_window,
        )
    return history


def write_training_history_tables(
    results: TrainingHistoryResults,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write the smoothed history and retained-checkpoint table."""

    output = prepare_output_dir(output_dir)
    paths = {
        "smoothed_history": output / "arista_training_history_smoothed.csv",
        "checkpoint_summary": output / "panel_metrics.csv",
    }
    calculate_smoothed_training_history(results).to_csv(
        paths["smoothed_history"], index=False
    )
    results.checkpoint_summary.to_csv(paths["checkpoint_summary"], index=False)
    return paths


def plot_training_histories(
    results: TrainingHistoryResults,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """Render the six-stage training-history figure as PDF and PNG."""

    from ._training_histories_plot import plot_training_histories as _plot

    return _plot(results, output_dir)
