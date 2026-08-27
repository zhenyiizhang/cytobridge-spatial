"""Processed inputs and calculations for AGIST supplementary figures."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ._io import read_json, require_files, resolve_results_dir


VELOCITY_SUMMARY_COLUMNS = (
    "n",
    "mean",
    "std",
    "se",
    "ci95_low",
    "ci95_high",
    "median",
    "q25",
    "q75",
)
INFERENCE_SEEDS = (1, 4, 8, 32, 256)
_FILES = (
    "manifest.json",
    "full_recompute_inputs.csv",
    "s2_velocity_cosine_per_cell.csv.gz",
    "s2_velocity_cosine_by_time.csv",
    "s2_velocity_cosine_by_state_cluster.csv",
    "s2_velocity_cosine_overall.csv",
    "s2_velocity_cosine_by_time_and_state_cluster.csv",
    "s2_cluster_selection_diagnostics.csv",
    "s3_observed_snapshots.npz",
    "s3_display_trajectories.npz",
    "s3_growth_mass_metrics.csv",
    "s3_interaction_radial_curve.csv",
    "s3_interaction_ablation_metrics.csv",
)


@dataclass(frozen=True)
class AgistFigureData:
    """Packaged processed inputs for Supplementary Figures S2 and S3."""

    source_dir: Path
    manifest: dict[str, Any]
    velocity_per_cell: pd.DataFrame
    source_velocity_by_time: pd.DataFrame
    source_velocity_by_cluster: pd.DataFrame
    source_velocity_overall: pd.DataFrame
    source_velocity_by_time_cluster: pd.DataFrame
    cluster_diagnostics: pd.DataFrame
    observed_time: np.ndarray
    observed_spatial: np.ndarray
    observed_gene: np.ndarray
    trajectory_time: np.ndarray
    trajectory_ground_truth: np.ndarray
    trajectory_predicted: np.ndarray
    trajectory_indices: np.ndarray
    growth_metrics: pd.DataFrame
    radial_curve: pd.DataFrame
    ablation_metrics: pd.DataFrame
    full_recompute_inputs: pd.DataFrame


@dataclass(frozen=True)
class AgistFigurePanels:
    """Calculated panel tables used by both renderers."""

    velocity_by_time: pd.DataFrame
    velocity_by_cluster: pd.DataFrame
    velocity_overall: pd.DataFrame
    velocity_by_time_cluster: pd.DataFrame
    growth_summary: pd.DataFrame
    potential_curve: pd.DataFrame
    ablation_summary: pd.DataFrame


def _require_columns(
    frame: pd.DataFrame,
    columns: set[str] | tuple[str, ...],
    *,
    source: Path,
) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing columns: {missing}")


def _require_finite(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
    *,
    source: Path,
) -> None:
    values = frame.loc[:, columns].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError(f"{source} contains non-finite values")


def _validate_manifest(manifest: dict[str, Any], source: Path) -> None:
    if manifest.get("analysis") != "agist_figures":
        raise ValueError(f"{source} must describe agist_figures")
    expected_files = set(_FILES).difference({"manifest.json"})
    if not isinstance(manifest.get("files"), dict) or set(manifest["files"]) != expected_files:
        raise ValueError(f"{source} contains an unexpected file roster")
    full_rerun = manifest.get("full_rerun")
    if not isinstance(full_rerun, dict) or full_rerun.get("included") is not False:
        raise ValueError(f"{source} must distinguish processed plotting from a full rerun")
    if full_rerun.get("registry") != "full_recompute_inputs.csv":
        raise ValueError(f"{source} contains an unexpected external-input registry")


def _summarize_series(values: pd.Series) -> pd.Series:
    finite = values[np.isfinite(values.to_numpy(dtype=float))].to_numpy(dtype=float)
    if not len(finite):
        raise ValueError("Velocity summary group contains no finite values")
    standard_deviation = float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0
    standard_error = standard_deviation / np.sqrt(len(finite))
    mean = float(np.mean(finite))
    return pd.Series(
        {
            "n": len(finite),
            "mean": mean,
            "std": standard_deviation,
            "se": standard_error,
            "ci95_low": mean - 1.96 * standard_error,
            "ci95_high": mean + 1.96 * standard_error,
            "median": float(np.median(finite)),
            "q25": float(np.quantile(finite, 0.25)),
            "q75": float(np.quantile(finite, 0.75)),
        }
    )


def summarize_agist_velocity(
    per_cell: pd.DataFrame,
    group_columns: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Calculate velocity-cosine summaries for the requested partitions."""

    long = per_cell.melt(
        id_vars=["row_index", "time", "state_cluster"],
        value_vars=["physical_cosine", "gene_cosine"],
        var_name="velocity_space",
        value_name="cosine",
    )
    long["velocity_space"] = long["velocity_space"].map(
        {"physical_cosine": "physical", "gene_cosine": "gene"}
    )
    result = (
        long.groupby([*group_columns, "velocity_space"], observed=True)["cosine"]
        .apply(_summarize_series)
        .unstack()
        .reset_index()
    )
    result["n"] = result["n"].astype(int)
    return result


def _sort_velocity(frame: pd.DataFrame, keys: tuple[str, ...]) -> pd.DataFrame:
    return frame.sort_values(["velocity_space", *keys]).reset_index(drop=True)


def _compare_velocity_summary(
    calculated: pd.DataFrame,
    source_table: pd.DataFrame,
    *,
    keys: tuple[str, ...],
    source: Path,
) -> None:
    ordered_calculated = _sort_velocity(calculated, keys)
    ordered_source = _sort_velocity(source_table, keys)
    key_columns = ["velocity_space", *keys]
    if not ordered_calculated[key_columns].equals(ordered_source[key_columns]):
        raise ValueError(f"{source} does not match the packaged partition keys")
    if not np.allclose(
        ordered_calculated.loc[:, VELOCITY_SUMMARY_COLUMNS],
        ordered_source.loc[:, VELOCITY_SUMMARY_COLUMNS],
        rtol=1e-11,
        atol=1e-13,
    ):
        raise ValueError(f"{source} does not match the cell-level calculations")


def _validate_velocity(
    per_cell: pd.DataFrame,
    source_tables: dict[str, tuple[pd.DataFrame, tuple[str, ...], Path]],
    diagnostics: pd.DataFrame,
    *,
    per_cell_source: Path,
    diagnostics_source: Path,
) -> None:
    required = {
        "row_index",
        "time",
        "state_cluster",
        "physical_cosine",
        "gene_cosine",
    }
    _require_columns(per_cell, required, source=per_cell_source)
    _require_finite(
        per_cell,
        ("row_index", "time", "physical_cosine", "gene_cosine"),
        source=per_cell_source,
    )
    if len(per_cell) != 31_816:
        raise ValueError(f"{per_cell_source} must contain 31,816 rows")
    if not np.array_equal(per_cell["row_index"].to_numpy(dtype=int), np.arange(len(per_cell))):
        raise ValueError(f"{per_cell_source} contains an unexpected row order")
    if set(per_cell["time"].astype(float)) != {0.0, 1.0, 2.0, 3.0}:
        raise ValueError(f"{per_cell_source} contains an unexpected time grid")
    if set(per_cell["state_cluster"].astype(str)) != {"C1", "C2", "C3"}:
        raise ValueError(f"{per_cell_source} contains an unexpected state partition")
    cosines = per_cell[["physical_cosine", "gene_cosine"]].to_numpy(dtype=float)
    if np.any(cosines < -1.000001) or np.any(cosines > 1.000001):
        raise ValueError(f"{per_cell_source} contains values outside cosine bounds")

    calculations = {
        "time": summarize_agist_velocity(per_cell, ("time",)),
        "cluster": summarize_agist_velocity(per_cell, ("state_cluster",)),
        "overall": summarize_agist_velocity(per_cell),
        "time_cluster": summarize_agist_velocity(
            per_cell, ("time", "state_cluster")
        ),
    }
    for name, (source_table, keys, source) in source_tables.items():
        _require_columns(
            source_table,
            {"velocity_space", *keys, *VELOCITY_SUMMARY_COLUMNS},
            source=source,
        )
        _compare_velocity_summary(
            calculations[name], source_table, keys=keys, source=source
        )

    _require_columns(
        diagnostics,
        {"n_clusters", "silhouette", "minimum_cluster_size", "maximum_cluster_size"},
        source=diagnostics_source,
    )
    _require_finite(
        diagnostics,
        ("n_clusters", "silhouette", "minimum_cluster_size", "maximum_cluster_size"),
        source=diagnostics_source,
    )
    selected = diagnostics.sort_values(
        ["silhouette", "n_clusters"], ascending=[False, True]
    ).iloc[0]
    if int(selected["n_clusters"]) != 3:
        raise ValueError(f"{diagnostics_source} must select three state partitions")


def _load_npz(path: Path, names: tuple[str, ...]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        missing = set(names).difference(archive.files)
        if missing:
            raise ValueError(f"{path} is missing arrays: {sorted(missing)}")
        return {name: np.asarray(archive[name]) for name in names}


def _validate_attraction_arrays(
    observed: dict[str, np.ndarray],
    trajectories: dict[str, np.ndarray],
    *,
    observed_source: Path,
    trajectory_source: Path,
) -> None:
    time = observed["time"].astype(float)
    spatial = observed["spatial"].astype(float)
    gene = observed["gene"].astype(float)
    if spatial.shape != gene.shape or spatial.shape != (2243, 2) or time.shape != (2243,):
        raise ValueError(f"{observed_source} contains unexpected snapshot arrays")
    if not all(np.isfinite(value).all() for value in (time, spatial, gene)):
        raise ValueError(f"{observed_source} contains non-finite values")
    counts = {float(value): int(np.sum(np.isclose(time, value))) for value in np.unique(time)}
    if counts != {0.0: 400, 1.0: 423, 2.0: 447, 3.0: 473, 4.0: 500}:
        raise ValueError(f"{observed_source} contains unexpected snapshot counts")

    time_points = trajectories["time_points"].astype(float)
    ground_truth = trajectories["ground_truth"].astype(float)
    predicted = trajectories["predicted"].astype(float)
    selected = trajectories["selected_indices"].astype(int)
    if time_points.shape != (201,) or ground_truth.shape != (201, 60, 4):
        raise ValueError(f"{trajectory_source} contains unexpected trajectory arrays")
    if predicted.shape != ground_truth.shape or selected.shape != (60,):
        raise ValueError(f"{trajectory_source} contains inconsistent trajectory arrays")
    expected = np.sort(np.random.default_rng(11).choice(400, size=60, replace=False))
    if not np.array_equal(selected, expected):
        raise ValueError(f"{trajectory_source} contains an unexpected display subset")
    if not all(
        np.isfinite(value).all()
        for value in (time_points, ground_truth, predicted)
    ):
        raise ValueError(f"{trajectory_source} contains non-finite values")
    if not np.isclose(time_points[0], 0.0) or not np.isclose(time_points[-1], 4.0):
        raise ValueError(f"{trajectory_source} contains an unexpected time range")


def _validate_attraction_tables(
    growth: pd.DataFrame,
    radial: pd.DataFrame,
    ablation: pd.DataFrame,
    registry: pd.DataFrame,
    *,
    paths: dict[str, Path],
) -> None:
    _require_columns(
        growth,
        {"seed", "time", "observed_relative_mass", "predicted_relative_mass", "absolute_tmv"},
        source=paths["s3_growth_mass_metrics.csv"],
    )
    _require_finite(
        growth,
        ("seed", "time", "observed_relative_mass", "predicted_relative_mass", "absolute_tmv"),
        source=paths["s3_growth_mass_metrics.csv"],
    )
    expected_growth = {
        (seed, float(time)) for seed in INFERENCE_SEEDS for time in range(5)
    }
    if set(growth[["seed", "time"]].itertuples(index=False, name=None)) != expected_growth:
        raise ValueError("Growth metrics must contain the complete seed-time grid")

    _require_columns(
        radial,
        {"distance", "true_coefficient", "learned_coefficient"},
        source=paths["s3_interaction_radial_curve.csv"],
    )
    _require_finite(
        radial,
        ("distance", "true_coefficient", "learned_coefficient"),
        source=paths["s3_interaction_radial_curve.csv"],
    )
    if len(radial) != 164 or not radial["distance"].is_monotonic_increasing:
        raise ValueError("Radial interaction table must contain 164 ordered rows")

    _require_columns(
        ablation,
        {"seed", "condition", "time", "space", "w1", "w2"},
        source=paths["s3_interaction_ablation_metrics.csv"],
    )
    _require_finite(
        ablation,
        ("seed", "time", "w1", "w2"),
        source=paths["s3_interaction_ablation_metrics.csv"],
    )
    expected_ablation = {
        (seed, condition, float(time), space)
        for seed in INFERENCE_SEEDS
        for condition in ("interaction_on", "interaction_off")
        for time in (1, 2, 3, 4)
        for space in ("joint", "spatial", "gene")
    }
    actual_ablation = set(
        ablation[["seed", "condition", "time", "space"]].itertuples(
            index=False, name=None
        )
    )
    if actual_ablation != expected_ablation:
        raise ValueError("Ablation metrics must contain the complete paired grid")

    _require_columns(
        registry,
        {"figure", "role", "relative_identifier", "availability"},
        source=paths["full_recompute_inputs.csv"],
    )
    if registry["relative_identifier"].astype(str).str.startswith("/").any():
        raise ValueError("External-input identifiers must be relative")


def load_agist_figures(results_dir: str | Path | None = None) -> AgistFigureData:
    """Load and validate the processed inputs for AGIST figures S2 and S3."""

    source_dir = resolve_results_dir(results_dir, slug="agist_figures")
    paths = require_files(source_dir, _FILES)
    manifest = read_json(paths["manifest.json"])
    _validate_manifest(manifest, paths["manifest.json"])

    per_cell = pd.read_csv(paths["s2_velocity_cosine_per_cell.csv.gz"])
    source_by_time = pd.read_csv(paths["s2_velocity_cosine_by_time.csv"])
    source_by_cluster = pd.read_csv(paths["s2_velocity_cosine_by_state_cluster.csv"])
    source_overall = pd.read_csv(paths["s2_velocity_cosine_overall.csv"])
    source_time_cluster = pd.read_csv(
        paths["s2_velocity_cosine_by_time_and_state_cluster.csv"]
    )
    diagnostics = pd.read_csv(paths["s2_cluster_selection_diagnostics.csv"])
    _validate_velocity(
        per_cell,
        {
            "time": (source_by_time, ("time",), paths["s2_velocity_cosine_by_time.csv"]),
            "cluster": (
                source_by_cluster,
                ("state_cluster",),
                paths["s2_velocity_cosine_by_state_cluster.csv"],
            ),
            "overall": (source_overall, (), paths["s2_velocity_cosine_overall.csv"]),
            "time_cluster": (
                source_time_cluster,
                ("time", "state_cluster"),
                paths["s2_velocity_cosine_by_time_and_state_cluster.csv"],
            ),
        },
        diagnostics,
        per_cell_source=paths["s2_velocity_cosine_per_cell.csv.gz"],
        diagnostics_source=paths["s2_cluster_selection_diagnostics.csv"],
    )

    observed = _load_npz(
        paths["s3_observed_snapshots.npz"], ("time", "spatial", "gene")
    )
    trajectories = _load_npz(
        paths["s3_display_trajectories.npz"],
        ("time_points", "selected_indices", "ground_truth", "predicted"),
    )
    _validate_attraction_arrays(
        observed,
        trajectories,
        observed_source=paths["s3_observed_snapshots.npz"],
        trajectory_source=paths["s3_display_trajectories.npz"],
    )
    growth = pd.read_csv(paths["s3_growth_mass_metrics.csv"])
    radial = pd.read_csv(paths["s3_interaction_radial_curve.csv"])
    ablation = pd.read_csv(paths["s3_interaction_ablation_metrics.csv"])
    registry = pd.read_csv(paths["full_recompute_inputs.csv"])
    _validate_attraction_tables(growth, radial, ablation, registry, paths=paths)

    return AgistFigureData(
        source_dir=source_dir,
        manifest=manifest,
        velocity_per_cell=per_cell,
        source_velocity_by_time=source_by_time,
        source_velocity_by_cluster=source_by_cluster,
        source_velocity_overall=source_overall,
        source_velocity_by_time_cluster=source_time_cluster,
        cluster_diagnostics=diagnostics,
        observed_time=observed["time"].astype(float),
        observed_spatial=observed["spatial"].astype(float),
        observed_gene=observed["gene"].astype(float),
        trajectory_time=trajectories["time_points"].astype(float),
        trajectory_ground_truth=trajectories["ground_truth"].astype(float),
        trajectory_predicted=trajectories["predicted"].astype(float),
        trajectory_indices=trajectories["selected_indices"].astype(int),
        growth_metrics=growth,
        radial_curve=radial,
        ablation_metrics=ablation,
        full_recompute_inputs=registry,
    )


def attraction_potential(
    distance: np.ndarray,
    coefficient: np.ndarray,
    *,
    cutoff: float = 0.30,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate a radial coefficient with zero potential at the cutoff."""

    mask = np.asarray(distance, dtype=float) <= cutoff
    radius = np.asarray(distance, dtype=float)[mask]
    values = np.asarray(coefficient, dtype=float)[mask].copy()
    order = np.argsort(radius)
    radius = radius[order]
    values = values[order]
    at_cutoff = np.isclose(radius, cutoff, atol=1e-12, rtol=0.0)
    interior = np.flatnonzero(radius < cutoff)
    if np.any(at_cutoff) and interior.size:
        values[at_cutoff] = values[interior[-1]]
    segments = 0.5 * (values[:-1] + values[1:]) * np.diff(radius)
    integral_to_cutoff = np.zeros_like(radius)
    integral_to_cutoff[:-1] = np.cumsum(segments[::-1])[::-1]
    potential = -integral_to_cutoff
    potential[at_cutoff] = 0.0
    return radius, potential


def calculate_agist_figure_panels(data: AgistFigureData) -> AgistFigurePanels:
    """Recalculate the summary tables plotted in S2 and S3."""

    by_time = _sort_velocity(
        summarize_agist_velocity(data.velocity_per_cell, ("time",)), ("time",)
    )
    by_cluster = _sort_velocity(
        summarize_agist_velocity(data.velocity_per_cell, ("state_cluster",)),
        ("state_cluster",),
    )
    overall = _sort_velocity(summarize_agist_velocity(data.velocity_per_cell), ())
    time_cluster = _sort_velocity(
        summarize_agist_velocity(data.velocity_per_cell, ("time", "state_cluster")),
        ("time", "state_cluster"),
    )

    growth = (
        data.growth_metrics.groupby("time", sort=True)
        .agg(
            observed_relative_mass=("observed_relative_mass", "first"),
            predicted_mean=("predicted_relative_mass", "mean"),
            predicted_sd=("predicted_relative_mass", "std"),
            n=("predicted_relative_mass", "size"),
        )
        .reset_index()
    )
    radius, true_potential = attraction_potential(
        data.radial_curve["distance"].to_numpy(dtype=float),
        data.radial_curve["true_coefficient"].to_numpy(dtype=float),
    )
    _, learned_potential = attraction_potential(
        data.radial_curve["distance"].to_numpy(dtype=float),
        data.radial_curve["learned_coefficient"].to_numpy(dtype=float),
    )
    potential = pd.DataFrame(
        {
            "distance": radius,
            "true_potential": true_potential,
            "learned_potential": learned_potential,
        }
    )

    ablation = (
        data.ablation_metrics.groupby(["space", "condition", "time"], sort=True)["w1"]
        .agg(mean="mean", std="std", n="size")
        .reset_index()
    )
    ablation["sem"] = ablation["std"] / np.sqrt(ablation["n"])
    return AgistFigurePanels(
        velocity_by_time=by_time,
        velocity_by_cluster=by_cluster,
        velocity_overall=overall,
        velocity_by_time_cluster=time_cluster,
        growth_summary=growth,
        potential_curve=potential,
        ablation_summary=ablation,
    )


def write_agist_figure_tables(
    panels: AgistFigurePanels,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write the recalculated panel tables."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    tables = {
        "s2_by_time": (panels.velocity_by_time, output / "s2_velocity_by_time.csv"),
        "s2_by_cluster": (
            panels.velocity_by_cluster,
            output / "s2_velocity_by_state_cluster.csv",
        ),
        "s2_overall": (panels.velocity_overall, output / "s2_velocity_overall.csv"),
        "s2_time_cluster": (
            panels.velocity_by_time_cluster,
            output / "s2_velocity_by_time_and_state_cluster.csv",
        ),
        "s3_growth": (panels.growth_summary, output / "s3_growth_summary.csv"),
        "s3_potential": (panels.potential_curve, output / "s3_potential_curve.csv"),
        "s3_ablation": (panels.ablation_summary, output / "s3_ablation_summary.csv"),
    }
    for frame, path in tables.values():
        frame.to_csv(path, index=False)
    return {name: path for name, (_, path) in tables.items()}


def plot_agist_figures(
    data: AgistFigureData,
    panels: AgistFigurePanels,
    output_dir: str | Path,
) -> dict[str, tuple[Path, Path]]:
    """Render Supplementary Figures S2 and S3 as PDF and PNG."""

    from ._agist_figures_plot import plot_agist_figures as _plot

    return _plot(data, panels, output_dir)
