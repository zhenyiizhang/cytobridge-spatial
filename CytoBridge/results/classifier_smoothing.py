"""Processed results for classifier spatial-smoothing sensitivity."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ._io import read_json, read_manifest, require_files, resolve_results_dir


DATASET_ORDER = ("zebrafish", "mosta", "arista", "admouse", "chicken_heart")
DATASET_LABELS = {
    "zebrafish": "Zebrafish",
    "mosta": "MOSTA",
    "arista": "ARISTA",
    "admouse": "AD mouse",
    "chicken_heart": "Chicken heart",
}
K_VALUES = (1, 5, 10, 20, 50)
FORMAL_K = {
    "zebrafish": 10,
    "mosta": 10,
    "arista": 10,
    "admouse": 1,
    "chicken_heart": 1,
}

_FILES = (
    "five_dataset_k_metrics.csv",
    "formal_k_policy.csv",
    "frame_sensitivity.csv",
    "transition_by_interval.csv",
    "arista_selection.json",
    "heart_selection.json",
)


@dataclass(frozen=True)
class ClassifierSmoothingResults:
    """Tables used to calculate and plot classifier-smoothing sensitivity."""

    source_dir: Path
    manifest: dict[str, Any]
    metrics: pd.DataFrame
    policy: pd.DataFrame
    frames: pd.DataFrame
    intervals: pd.DataFrame
    composition: pd.DataFrame
    transition: pd.DataFrame
    selections: dict[str, dict[str, Any]]


def _require_columns(frame: pd.DataFrame, columns: set[str], source: Path) -> None:
    missing = sorted(columns.difference(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing columns: {missing}")


def _require_finite(frame: pd.DataFrame, columns: list[str], source: Path) -> None:
    values = frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError(f"{source} contains non-finite values in {columns}")


def _sem(values: pd.Series | np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    if array.size <= 1:
        return 0.0
    return float(np.std(array, ddof=1) / math.sqrt(array.size))


def _validate_dataset_values(
    frame: pd.DataFrame,
    *,
    allowed: set[str],
    source: Path,
) -> None:
    observed = set(frame["dataset"].astype(str))
    unknown = sorted(observed.difference(allowed))
    if unknown:
        raise ValueError(f"{source} contains unknown datasets: {unknown}")


def _validate_selection(path: Path, *, dataset: str) -> dict[str, Any]:
    selection = read_json(path)
    if selection.get("dataset") != dataset:
        raise ValueError(f"{path} must describe dataset {dataset!r}")
    if int(selection.get("selected_k", -1)) != 1:
        raise ValueError(f"{path} must select k=1")
    candidates = tuple(int(value) for value in selection.get("candidate_k", []))
    if candidates != K_VALUES:
        raise ValueError(f"{path} must contain candidate k values {K_VALUES}")
    return selection


def load_classifier_smoothing_results(
    results_dir: str | Path | None = None,
) -> ClassifierSmoothingResults:
    """Load classifier-smoothing inputs and calculate the panel summaries.

    Parameters
    ----------
    results_dir
        Directory containing the compact processed files. Packaged example data
        are used when this argument is omitted.
    """

    source_dir = resolve_results_dir(results_dir, slug="classifier_smoothing")
    paths = require_files(source_dir, _FILES)
    metrics = pd.read_csv(paths["five_dataset_k_metrics.csv"])
    policy = pd.read_csv(paths["formal_k_policy.csv"])
    frames = pd.read_csv(paths["frame_sensitivity.csv"])
    intervals = pd.read_csv(paths["transition_by_interval.csv"])

    _require_columns(
        metrics,
        {"dataset", "k", "n_heldout", "balanced_accuracy", "macro_f1", "accuracy"},
        paths["five_dataset_k_metrics.csv"],
    )
    _require_columns(
        policy,
        {"dataset", "accuracy_best_k", "formal_analysis_k"},
        paths["formal_k_policy.csv"],
    )
    _require_columns(
        frames,
        {"dataset", "time", "requested_k", "composition_tv"},
        paths["frame_sensitivity.csv"],
    )
    _require_columns(
        intervals,
        {"time_from", "time_to", "k", "particle_count", "transition_fraction"},
        paths["transition_by_interval.csv"],
    )

    allowed_datasets = set(DATASET_ORDER)
    _validate_dataset_values(
        metrics, allowed=allowed_datasets, source=paths["five_dataset_k_metrics.csv"]
    )
    _validate_dataset_values(
        policy, allowed=allowed_datasets, source=paths["formal_k_policy.csv"]
    )
    _validate_dataset_values(
        frames, allowed={"zebrafish"}, source=paths["frame_sensitivity.csv"]
    )

    if metrics.duplicated(["dataset", "k"]).any():
        raise ValueError("Classifier metrics contain duplicate dataset-k rows")
    expected_metric_keys = {(dataset, k) for dataset in DATASET_ORDER for k in K_VALUES}
    observed_metric_keys = {
        (str(row.dataset), int(row.k)) for row in metrics.itertuples(index=False)
    }
    if observed_metric_keys != expected_metric_keys:
        raise ValueError("Classifier metrics must contain the complete dataset-k grid")
    _require_finite(
        metrics,
        ["n_heldout", "balanced_accuracy", "macro_f1", "accuracy"],
        paths["five_dataset_k_metrics.csv"],
    )

    if policy.duplicated(["dataset"]).any():
        raise ValueError("Smoothing policy contains duplicate dataset rows")
    policy_map = {
        str(row.dataset): int(row.formal_analysis_k)
        for row in policy.itertuples(index=False)
    }
    if policy_map != FORMAL_K:
        raise ValueError(
            f"Formal k policy differs from the figure contract: {policy_map}"
        )
    best_map = {
        str(row.dataset): int(row.accuracy_best_k)
        for row in policy.itertuples(index=False)
    }
    if best_map != {dataset: 1 for dataset in DATASET_ORDER}:
        raise ValueError(
            f"Held-out accuracy selections differ from the figure contract: {best_map}"
        )

    if frames.duplicated(["dataset", "time", "requested_k"]).any():
        raise ValueError("Frame sensitivity contains duplicate dataset-time-k rows")
    _require_finite(
        frames,
        ["time", "requested_k", "composition_tv"],
        paths["frame_sensitivity.csv"],
    )
    frame_counts = frames.groupby("requested_k", sort=True).size().to_dict()
    if frame_counts != {k: 9 for k in K_VALUES}:
        raise ValueError(f"Expected nine generated frames per k, found {frame_counts}")
    composition = (
        frames.groupby("requested_k", sort=True)["composition_tv"]
        .agg(["mean", _sem])
        .reset_index()
        .rename(columns={"requested_k": "k", "_sem": "sem"})
    )
    composition[["mean", "sem"]] *= 100.0

    if intervals.duplicated(["time_from", "time_to", "k"]).any():
        raise ValueError("Transition sensitivity contains duplicate interval-k rows")
    _require_finite(
        intervals,
        ["time_from", "time_to", "k", "particle_count", "transition_fraction"],
        paths["transition_by_interval.csv"],
    )
    if set(intervals["k"].astype(int)) != set(K_VALUES):
        raise ValueError("Transition sensitivity must contain the complete k grid")
    if len(intervals[["time_from", "time_to"]].drop_duplicates()) != 4:
        raise ValueError("Transition sensitivity must contain four time intervals")
    interval_counts = intervals.groupby("k", sort=True).size().to_dict()
    if interval_counts != {k: 4 for k in K_VALUES}:
        raise ValueError(
            f"Expected four transition intervals per k, found {interval_counts}"
        )
    if set(intervals["particle_count"].astype(int)) != {563}:
        raise ValueError(
            "Transition sensitivity must use the fixed 563-particle cohort"
        )
    transition = (
        intervals.groupby("k", sort=True)["transition_fraction"]
        .agg(["mean", _sem])
        .reset_index()
        .rename(columns={"_sem": "sem"})
    )
    transition[["mean", "sem"]] *= 100.0

    selections = {
        "arista": _validate_selection(paths["arista_selection.json"], dataset="arista"),
        "chicken_heart": _validate_selection(
            paths["heart_selection.json"], dataset="chicken_heart"
        ),
    }
    return ClassifierSmoothingResults(
        source_dir=source_dir,
        manifest=read_manifest(source_dir),
        metrics=metrics.sort_values(["dataset", "k"]).reset_index(drop=True),
        policy=policy.reset_index(drop=True),
        frames=frames.reset_index(drop=True),
        intervals=intervals.reset_index(drop=True),
        composition=composition,
        transition=transition,
        selections=selections,
    )


def classifier_smoothing_statistics(
    results: ClassifierSmoothingResults,
) -> dict[str, float | int]:
    """Return selected values from the calculated panel data."""

    def balanced_accuracy(dataset: str, k: int) -> float:
        value = results.metrics.loc[
            results.metrics["dataset"].eq(dataset)
            & results.metrics["k"].astype(int).eq(k),
            "balanced_accuracy",
        ]
        return float(value.iloc[0])

    transition_k1 = float(
        results.transition.loc[results.transition["k"].eq(1), "mean"].iloc[0]
    )
    transition_k10 = float(
        results.transition.loc[results.transition["k"].eq(10), "mean"].iloc[0]
    )
    return {
        "arista_balanced_accuracy_k1": balanced_accuracy("arista", 1),
        "arista_balanced_accuracy_k10": balanced_accuracy("arista", 10),
        "chicken_heart_balanced_accuracy_k1": balanced_accuracy("chicken_heart", 1),
        "chicken_heart_balanced_accuracy_k10": balanced_accuracy("chicken_heart", 10),
        "zebrafish_composition_tv_percent_k10": float(
            results.composition.loc[results.composition["k"].eq(10), "mean"].iloc[0]
        ),
        "zebrafish_transition_fraction_percent_k1": transition_k1,
        "zebrafish_transition_fraction_percent_k10": transition_k10,
        "zebrafish_transition_percentage_point_delta_k10_vs_k1": (
            transition_k10 - transition_k1
        ),
    }


def write_classifier_smoothing_tables(
    results: ClassifierSmoothingResults,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write the panel-data tables produced by the analysis."""

    output = Path(output_dir)
    paths = {
        "metrics": output / "five_dataset_k_metrics.csv",
        "policy": output / "formal_k_policy.csv",
        "composition": output / "composition_summary.csv",
        "transition": output / "transition_summary.csv",
    }
    results.metrics.to_csv(paths["metrics"], index=False)
    results.policy.to_csv(paths["policy"], index=False)
    results.composition.to_csv(paths["composition"], index=False)
    results.transition.to_csv(paths["transition"], index=False)
    return paths


def plot_classifier_smoothing(
    results: ClassifierSmoothingResults,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """Render the classifier-smoothing figure as PDF and PNG."""

    from ._classifier_smoothing_plot import plot_classifier_smoothing as _plot

    return _plot(results, output_dir)
