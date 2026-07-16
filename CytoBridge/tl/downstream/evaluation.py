"""Distribution-level evaluation for observed and generated trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .downstream_data import infer_time_key, parse_time_value
from .simulation import simulate_sde_points

__all__ = [
    "DistributionMetricComparison",
    "DistributionEvaluationResult",
    "compare_distribution_metric_tables",
    "compute_distribution_metrics",
    "compute_local_structure_metrics",
    "evaluate_model_distributions",
    "plot_generated_vs_observed",
    "save_distribution_metric_comparison",
    "save_distribution_evaluation",
]


@dataclass(frozen=True)
class DistributionEvaluationResult:
    """Generated trajectory, matched observations, and distribution metrics."""

    time_points: tuple[float, ...]
    spatial_dim: int
    predicted_points: Mapping[float, np.ndarray]
    predicted_weights: Mapping[float, np.ndarray]
    observed_points: Mapping[float, np.ndarray]
    metrics: pd.DataFrame
    settings: Mapping[str, object]


@dataclass(frozen=True)
class DistributionMetricComparison:
    """Long-form metrics, summaries, and paired deltas for fitted models."""

    metrics: pd.DataFrame
    summary: pd.DataFrame
    paired_deltas: pd.DataFrame
    baseline: str


def _normalized_weights(weights: Optional[np.ndarray], n: int) -> np.ndarray:
    if n <= 0:
        raise ValueError("A distribution must contain at least one point.")
    if weights is None:
        return np.full(n, 1.0 / n, dtype=np.float64)
    values = np.asarray(weights, dtype=np.float64).reshape(-1)
    if values.shape[0] != n:
        raise ValueError(f"weights has {values.shape[0]} rows, expected {n}.")
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("weights must be finite and non-negative.")
    total = float(values.sum())
    if total <= 0:
        raise ValueError("weights must have a positive sum.")
    return values / total


def _prepare_ot_samples(
    predicted: np.ndarray,
    observed: np.ndarray,
    predicted_weights: Optional[np.ndarray],
    *,
    max_ot_points: Optional[int],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    predicted = np.asarray(predicted, dtype=np.float64)
    observed = np.asarray(observed, dtype=np.float64)
    if predicted.ndim != 2 or observed.ndim != 2:
        raise ValueError("predicted and observed must both be 2D arrays.")
    if predicted.shape[1] != observed.shape[1]:
        raise ValueError(
            f"Feature mismatch: predicted={predicted.shape[1]}, observed={observed.shape[1]}."
        )
    pred_weights = _normalized_weights(predicted_weights, predicted.shape[0])

    if max_ot_points is None:
        cap = max(predicted.shape[0], observed.shape[0])
    else:
        cap = int(max_ot_points)
        if cap <= 0:
            raise ValueError("max_ot_points must be positive or None.")

    if predicted.shape[0] > cap:
        # Weighted resampling preserves the weighted empirical measure while
        # keeping the exact EMD problem bounded for large spatial datasets.
        idx_pred = rng.choice(
            predicted.shape[0], size=cap, replace=True, p=pred_weights
        )
        predicted = predicted[idx_pred]
        pred_weights = np.full(cap, 1.0 / cap, dtype=np.float64)
    if observed.shape[0] > cap:
        idx_obs = rng.choice(observed.shape[0], size=cap, replace=False)
        observed = observed[idx_obs]
    obs_weights = np.full(observed.shape[0], 1.0 / observed.shape[0], dtype=np.float64)
    return predicted, observed, pred_weights, obs_weights


def compute_distribution_metrics(
    predicted: np.ndarray,
    observed: np.ndarray,
    *,
    predicted_weights: Optional[np.ndarray] = None,
    max_ot_points: Optional[int] = 1024,
    random_seed: int = 42,
) -> dict[str, float | int]:
    """Compute weighted Wasserstein-1 and Wasserstein-2 distances."""
    import ot
    from scipy.spatial.distance import cdist

    pred, obs, pred_w, obs_w = _prepare_ot_samples(
        predicted,
        observed,
        predicted_weights,
        max_ot_points=max_ot_points,
        rng=np.random.default_rng(int(random_seed)),
    )
    distances = cdist(pred, obs, metric="euclidean")
    w1 = float(ot.emd2(pred_w, obs_w, distances, numItermax=int(1e7)))
    w2_sq = float(ot.emd2(pred_w, obs_w, distances**2, numItermax=int(1e7)))
    return {
        "w1": w1,
        "w2": float(np.sqrt(max(w2_sq, 0.0))),
        "ot_predicted_points": int(pred.shape[0]),
        "ot_observed_points": int(obs.shape[0]),
    }


def compute_local_structure_metrics(
    predicted: np.ndarray,
    observed: np.ndarray,
    *,
    max_points: Optional[int] = 5000,
    random_seed: int = 42,
) -> dict[str, float | int]:
    """Measure local dispersion and support coverage without OT resampling.

    W1/W2 can remain small when a generated cloud covers the correct global
    region but collapses into dense particle clumps. These diagnostics compare
    within-cloud nearest-neighbor scales and use the observed cloud's 95th
    percentile self-neighbor distance as a support radius. Subsampling is
    always without replacement so it cannot manufacture duplicate particles.
    """
    from scipy.spatial import cKDTree

    pred = np.asarray(predicted, dtype=np.float64)
    obs = np.asarray(observed, dtype=np.float64)
    if pred.ndim != 2 or obs.ndim != 2:
        raise ValueError("predicted and observed must both be 2D arrays.")
    if pred.shape[1] != obs.shape[1]:
        raise ValueError(
            f"Feature mismatch: predicted={pred.shape[1]}, observed={obs.shape[1]}."
        )
    if pred.shape[0] < 2 or obs.shape[0] < 2:
        raise ValueError("Local-structure metrics require at least two points per cloud.")
    if not np.isfinite(pred).all() or not np.isfinite(obs).all():
        raise ValueError("predicted and observed must be finite.")

    if max_points is not None:
        cap = int(max_points)
        if cap < 2:
            raise ValueError("max_points must be at least 2 or None.")
        rng = np.random.default_rng(int(random_seed))
        if pred.shape[0] > cap:
            pred = pred[rng.choice(pred.shape[0], size=cap, replace=False)]
        if obs.shape[0] > cap:
            obs = obs[rng.choice(obs.shape[0], size=cap, replace=False)]

    pred_tree = cKDTree(pred)
    obs_tree = cKDTree(obs)
    pred_nn = np.asarray(pred_tree.query(pred, k=2, workers=1)[0])[:, 1]
    obs_nn = np.asarray(obs_tree.query(obs, k=2, workers=1)[0])[:, 1]
    pred_nn_median = float(np.median(pred_nn))
    obs_nn_median = float(np.median(obs_nn))
    eps = np.finfo(np.float64).eps
    observed_radius = max(float(np.quantile(obs_nn, 0.95)), eps)
    observed_scale = max(obs_nn_median, eps)
    observed_to_pred = np.asarray(pred_tree.query(obs, k=1, workers=1)[0])
    predicted_to_obs = np.asarray(obs_tree.query(pred, k=1, workers=1)[0])

    return {
        "predicted_nn_median": pred_nn_median,
        "observed_nn_median": obs_nn_median,
        "nn_dispersion_ratio": float(pred_nn_median / observed_scale),
        "support_recall_at_observed_q95": float(
            np.mean(observed_to_pred <= observed_radius)
        ),
        "support_precision_at_observed_q95": float(
            np.mean(predicted_to_obs <= observed_radius)
        ),
        "clump_fraction_at_0_1_observed_nn": float(
            np.mean(pred_nn <= 0.1 * observed_scale)
        ),
        "structure_predicted_points": int(pred.shape[0]),
        "structure_observed_points": int(obs.shape[0]),
    }


def _feature_spaces(spatial_dim: int, dim: int) -> dict[str, slice]:
    spaces = {"joint": slice(0, dim)}
    if spatial_dim > 0:
        spaces["spatial"] = slice(0, spatial_dim)
    if dim > spatial_dim:
        spaces["pca"] = slice(spatial_dim, dim)
    return spaces


def evaluate_model_distributions(
    adata,
    model,
    *,
    time_points: Optional[Sequence[float]] = None,
    n_samples: int = 5000,
    dt: float = 0.01,
    sigma: float = 0.03,
    include_score: bool = True,
    interaction_m: int = 1024,
    max_ot_points: Optional[int] = 1024,
    structure_max_points: Optional[int] = 5000,
    device: str = "cuda",
    time_key: Optional[str] = "time_point_processed",
    obsm_key: str = "X_latent",
    spatial_key: str = "spatial_aligned",
    concat_spatial: Optional[bool] = True,
    random_seed: int = 42,
    include_initial_time: bool = False,
    verbose: bool = True,
) -> DistributionEvaluationResult:
    """Simulate a fitted model and evaluate W1, W2, and TMV at observed times.

    W1/W2 are reported in the complete joint state, physical space, and PCA
    space. TMV follows the historical CytoBridge evaluation notebook:
    ``abs(predicted_relative_mass - observed_relative_mass) /
    observed_relative_mass``. Predicted particle weights are kept unnormalized
    for TMV and normalized only for optimal transport.
    """
    import torch

    resolved_time_key = infer_time_key(adata.obs, preferred=time_key)
    observed_times = np.asarray(
        [parse_time_value(value) for value in adata.obs[resolved_time_key]],
        dtype=np.float64,
    )
    if time_points is None:
        time_points = sorted(float(value) for value in np.unique(observed_times))
    else:
        time_points = [float(value) for value in time_points]
    if not time_points:
        raise ValueError("time_points must be non-empty.")

    latent = np.asarray(adata.obsm[obsm_key], dtype=np.float32)
    use_spatial = (
        bool(concat_spatial)
        if concat_spatial is not None
        else spatial_key in adata.obsm
    )
    if use_spatial:
        spatial = np.asarray(adata.obsm[spatial_key], dtype=np.float32)
        features = np.hstack((spatial, latent)).astype(np.float32)
        spatial_dim = int(spatial.shape[1])
    else:
        features = latent
        spatial_dim = 0

    np.random.seed(int(random_seed))
    torch.manual_seed(int(random_seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(random_seed))
    if hasattr(model, "to"):
        model.to(device)
    if hasattr(model, "eval"):
        model.eval()

    points, weights = simulate_sde_points(
        adata=adata,
        model=model,
        dim=int(features.shape[1]),
        time_index=0,
        n_samples=int(n_samples),
        ts_points=time_points,
        dt=float(dt),
        sigma=float(sigma),
        include_score=bool(include_score),
        interaction_m=int(interaction_m),
        device=device,
        time_key=resolved_time_key,
        obsm_key=obsm_key,
        spatial_key=spatial_key,
        concat_spatial=concat_spatial,
        verbose=verbose,
    )

    predicted_points: dict[float, np.ndarray] = {}
    predicted_weights_by_time: dict[float, np.ndarray] = {}
    observed_points: dict[float, np.ndarray] = {}
    rows: list[dict[str, float | int | str]] = []
    initial_count = int(np.isclose(observed_times, time_points[0]).sum())
    spaces = _feature_spaces(spatial_dim, int(features.shape[1]))

    for time_idx, time_value in enumerate(time_points):
        pred = np.asarray(points[time_idx], dtype=np.float32)
        pred_w = np.asarray(weights[time_idx], dtype=np.float64).reshape(-1)
        observed = features[np.isclose(observed_times, float(time_value))]
        if observed.shape[0] == 0:
            raise ValueError(f"No observed cells found at time {time_value}.")
        predicted_points[time_value] = pred
        predicted_weights_by_time[time_value] = pred_w
        observed_points[time_value] = observed

        predicted_mass = float(pred_w.sum())
        observed_mass_relative = float(observed.shape[0] / initial_count)
        tmv_absolute = float(abs(predicted_mass - observed_mass_relative))
        tmv = float(tmv_absolute / observed_mass_relative)
        if time_idx == 0 and not include_initial_time:
            continue

        for space_idx, (space_name, columns) in enumerate(spaces.items()):
            metric = compute_distribution_metrics(
                pred[:, columns],
                observed[:, columns],
                predicted_weights=pred_w,
                max_ot_points=max_ot_points,
                random_seed=int(random_seed) + time_idx * 101 + space_idx,
            )
            structure = compute_local_structure_metrics(
                pred[:, columns],
                observed[:, columns],
                max_points=structure_max_points,
                random_seed=int(random_seed) + time_idx * 101 + space_idx,
            )
            rows.append(
                {
                    "time": float(time_value),
                    "space": space_name,
                    "n_predicted": int(pred.shape[0]),
                    "n_observed": int(observed.shape[0]),
                    "predicted_mass": predicted_mass,
                    "observed_mass_relative": observed_mass_relative,
                    "tmv_absolute": tmv_absolute,
                    "tmv": tmv,
                    **metric,
                    **structure,
                }
            )

    settings = {
        "n_samples": int(n_samples),
        "dt": float(dt),
        "sigma": float(sigma),
        "include_score": bool(include_score),
        "interaction_m": int(interaction_m),
        "max_ot_points": None if max_ot_points is None else int(max_ot_points),
        "structure_max_points": (
            None if structure_max_points is None else int(structure_max_points)
        ),
        "random_seed": int(random_seed),
        "time_key": resolved_time_key,
        "obsm_key": obsm_key,
        "spatial_key": spatial_key,
    }
    return DistributionEvaluationResult(
        time_points=tuple(time_points),
        spatial_dim=spatial_dim,
        predicted_points=predicted_points,
        predicted_weights=predicted_weights_by_time,
        observed_points=observed_points,
        metrics=pd.DataFrame(rows),
        settings=settings,
    )


def plot_generated_vs_observed(
    result: DistributionEvaluationResult,
    *,
    space: str,
    out_path: str | Path,
    max_points: int = 5000,
    random_seed: int = 42,
) -> Path:
    """Save paired observed/generated scatter maps for spatial or PCA space."""
    import matplotlib.pyplot as plt

    if space not in {"spatial", "pca"}:
        raise ValueError("space must be 'spatial' or 'pca'.")
    if space == "spatial" and result.spatial_dim < 2:
        raise ValueError("The result does not contain 2D spatial coordinates.")
    start = 0 if space == "spatial" else result.spatial_dim
    rng = np.random.default_rng(int(random_seed))
    n_rows = len(result.time_points)
    fig, axes = plt.subplots(
        n_rows, 2, figsize=(8, max(3.0, n_rows * 3.0)), squeeze=False
    )
    fig.patch.set_facecolor("white")

    for row, time_value in enumerate(result.time_points):
        observed = np.asarray(result.observed_points[time_value])[:, start : start + 2]
        predicted = np.asarray(result.predicted_points[time_value])[
            :, start : start + 2
        ]
        if observed.shape[0] > max_points:
            observed = observed[
                rng.choice(observed.shape[0], size=int(max_points), replace=False)
            ]
        if predicted.shape[0] > max_points:
            predicted = predicted[
                rng.choice(predicted.shape[0], size=int(max_points), replace=False)
            ]
        joined = np.vstack((observed, predicted))
        x_pad = max(float(np.ptp(joined[:, 0])) * 0.03, 1e-6)
        y_pad = max(float(np.ptp(joined[:, 1])) * 0.03, 1e-6)
        xlim = (
            float(joined[:, 0].min() - x_pad),
            float(joined[:, 0].max() + x_pad),
        )
        ylim = (
            float(joined[:, 1].min() - y_pad),
            float(joined[:, 1].max() + y_pad),
        )
        panel_data = (
            (observed, "Observed", "#4C4C4C"),
            (predicted, "Generated", "#A33A3A"),
        )
        for column, (values, title, color) in enumerate(panel_data):
            ax = axes[row, column]
            ax.set_facecolor("white")
            ax.scatter(
                values[:, 0], values[:, 1], s=2, alpha=0.4, c=color, linewidths=0
            )
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(f"{title}, t={time_value:g}", fontsize=9, color="black")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color("black")
    fig.suptitle(
        f"Generated versus observed distributions ({space})",
        fontsize=12,
        color="black",
    )
    fig.tight_layout()
    out_path = Path(out_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def save_distribution_evaluation(
    result: DistributionEvaluationResult,
    output_dir: str | Path,
) -> dict[str, str]:
    """Save metrics plus spatial and PCA observed/generated figures."""
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "distribution_metrics.csv"
    result.metrics.to_csv(metrics_path, index=False)
    samples_path = output_dir / "distribution_samples.npz"
    sample_payload: dict[str, np.ndarray] = {
        "time_points": np.asarray(result.time_points, dtype=np.float64),
        "spatial_dim": np.asarray([result.spatial_dim], dtype=np.int64),
    }
    for index, time_value in enumerate(result.time_points):
        sample_payload[f"predicted_{index}"] = np.asarray(
            result.predicted_points[time_value], dtype=np.float32
        )
        sample_payload[f"predicted_weights_{index}"] = np.asarray(
            result.predicted_weights[time_value], dtype=np.float32
        )
        sample_payload[f"observed_{index}"] = np.asarray(
            result.observed_points[time_value], dtype=np.float32
        )
    np.savez_compressed(samples_path, **sample_payload)
    paths = {"metrics": str(metrics_path), "samples": str(samples_path)}
    if result.spatial_dim >= 2:
        paths["spatial_figure"] = str(
            plot_generated_vs_observed(
                result,
                space="spatial",
                out_path=output_dir / "generated_vs_observed_spatial.svg",
            )
        )
    first_points = next(iter(result.observed_points.values()))
    if first_points.shape[1] - result.spatial_dim >= 2:
        paths["pca_figure"] = str(
            plot_generated_vs_observed(
                result,
                space="pca",
                out_path=output_dir / "generated_vs_observed_pca.svg",
            )
        )
    return paths


def compare_distribution_metric_tables(
    metrics_by_model: Mapping[str, pd.DataFrame],
    *,
    baseline: str,
) -> DistributionMetricComparison:
    """Compare model metric tables produced by :func:`evaluate_model_distributions`.

    Every table must contain one row per ``(time, space)`` and the same paired
    time/space grid. Deltas are ``candidate - baseline``; negative W1/W2/TMV
    deltas therefore indicate improvement relative to the baseline model.
    """
    if baseline not in metrics_by_model:
        raise KeyError(f"baseline '{baseline}' is not present in metrics_by_model.")
    if len(metrics_by_model) < 2:
        raise ValueError("At least two model metric tables are required.")

    required = {"time", "space", "w1", "w2", "tmv"}
    frames = []
    grids: dict[str, set[tuple[float, str]]] = {}
    for raw_name, raw_table in metrics_by_model.items():
        name = str(raw_name)
        table = pd.DataFrame(raw_table).copy()
        missing = sorted(required.difference(table.columns))
        if missing:
            raise KeyError(f"Metric table '{name}' is missing columns: {missing}")
        if table.duplicated(["time", "space"]).any():
            raise ValueError(
                f"Metric table '{name}' contains duplicate (time, space) rows."
            )
        table.insert(0, "model", name)
        frames.append(table)
        grids[name] = set(
            zip(table["time"].astype(float), table["space"].astype(str))
        )

    baseline_grid = grids[baseline]
    mismatched = {
        name: sorted(grid.symmetric_difference(baseline_grid))
        for name, grid in grids.items()
        if grid != baseline_grid
    }
    if mismatched:
        raise ValueError(
            "All metric tables must use the same (time, space) grid; "
            f"mismatches={mismatched}"
        )

    metrics = pd.concat(frames, ignore_index=True)
    value_columns = [
        name
        for name in (
            "w1",
            "w2",
            "tmv",
            "tmv_absolute",
            "nn_dispersion_ratio",
            "support_recall_at_observed_q95",
            "support_precision_at_observed_q95",
            "clump_fraction_at_0_1_observed_nn",
        )
        if name in metrics.columns
    ]
    summary = (
        metrics.groupby(["model", "space"], sort=True)[value_columns]
        .mean()
        .reset_index()
    )

    baseline_table = metrics.loc[
        metrics["model"] == baseline,
        ["time", "space", *value_columns],
    ].copy()
    paired = []
    for name in metrics["model"].drop_duplicates():
        if name == baseline:
            continue
        candidate = metrics.loc[
            metrics["model"] == name,
            ["time", "space", *value_columns],
        ].copy()
        merged = candidate.merge(
            baseline_table,
            on=["time", "space"],
            suffixes=("_candidate", "_baseline"),
            validate="1:1",
        )
        merged.insert(0, "candidate", name)
        merged.insert(1, "baseline", baseline)
        for metric in value_columns:
            candidate_col = f"{metric}_candidate"
            baseline_col = f"{metric}_baseline"
            delta = merged[candidate_col] - merged[baseline_col]
            merged[f"{metric}_delta"] = delta
            denominator = merged[baseline_col].replace(0.0, np.nan)
            merged[f"{metric}_relative_delta"] = delta / denominator
        paired.append(merged)

    paired_deltas = pd.concat(paired, ignore_index=True)
    return DistributionMetricComparison(
        metrics=metrics,
        summary=summary,
        paired_deltas=paired_deltas,
        baseline=str(baseline),
    )


def save_distribution_metric_comparison(
    comparison: DistributionMetricComparison,
    output_dir: str | Path,
) -> dict[str, str]:
    """Save paired model metrics and a time-resolved comparison figure."""
    import textwrap

    import matplotlib.pyplot as plt

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "model_metrics_long.csv"
    summary_path = output_dir / "model_metrics_mean_by_space.csv"
    deltas_path = output_dir / "model_metrics_paired_deltas.csv"
    comparison.metrics.to_csv(metrics_path, index=False)
    comparison.summary.to_csv(summary_path, index=False)
    comparison.paired_deltas.to_csv(deltas_path, index=False)

    spaces = list(dict.fromkeys(comparison.metrics["space"].astype(str)))
    metric_names = ["w1", "w2", "tmv"]
    models = list(dict.fromkeys(comparison.metrics["model"].astype(str)))
    markers = ("o", "s", "^", "D", "v", "P", "X")
    line_styles = ("-", "--", "-.", ":")
    fig, axes = plt.subplots(
        len(spaces),
        len(metric_names),
        figsize=(4.2 * len(metric_names), 3.6 * len(spaces)),
        squeeze=False,
        sharex=True,
    )
    fig.patch.set_facecolor("white")
    for row, space in enumerate(spaces):
        for col, metric in enumerate(metric_names):
            ax = axes[row, col]
            ax.set_facecolor("white")
            for model_index, model_name in enumerate(models):
                subset = comparison.metrics.loc[
                    (comparison.metrics["space"].astype(str) == space)
                    & (comparison.metrics["model"].astype(str) == model_name)
                ].sort_values("time")
                ax.plot(
                    subset["time"],
                    subset[metric],
                    marker=markers[model_index % len(markers)],
                    linestyle=line_styles[model_index % len(line_styles)],
                    linewidth=1.8,
                    label=textwrap.fill(model_name.replace("_", " "), width=34),
                )
            ax.set_title(f"{space}: {metric.upper()}")
            ax.set_xlabel("Time")
            ax.set_ylabel(metric.upper())
            ax.grid(alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        legend_columns = min(2, max(1, len(labels)))
        fig.legend(handles, labels, loc="upper center", ncol=legend_columns)
        fig.subplots_adjust(
            top=0.81 if len(labels) > 2 else 0.88,
            bottom=0.07,
            hspace=0.52,
            wspace=0.25,
        )
    fig_path = output_dir / "model_metric_comparison.svg"
    fig.savefig(fig_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    paths = {
        "metrics": str(metrics_path),
        "summary": str(summary_path),
        "paired_deltas": str(deltas_path),
        "figure": str(fig_path),
    }

    local_metrics = [
        name
        for name in (
            "nn_dispersion_ratio",
            "support_recall_at_observed_q95",
            "support_precision_at_observed_q95",
            "clump_fraction_at_0_1_observed_nn",
        )
        if name in comparison.metrics.columns
    ]
    if local_metrics:
        local_fig, local_axes = plt.subplots(
            len(spaces),
            len(local_metrics),
            figsize=(4.2 * len(local_metrics), 3.6 * len(spaces)),
            squeeze=False,
            sharex=True,
        )
        local_fig.patch.set_facecolor("white")
        ideal_values = {
            "nn_dispersion_ratio": 1.0,
            "support_recall_at_observed_q95": 1.0,
            "support_precision_at_observed_q95": 1.0,
            "clump_fraction_at_0_1_observed_nn": 0.0,
        }
        titles = {
            "nn_dispersion_ratio": "NN dispersion ratio",
            "support_recall_at_observed_q95": "Support recall",
            "support_precision_at_observed_q95": "Support precision",
            "clump_fraction_at_0_1_observed_nn": "Clump fraction",
        }
        for row, space in enumerate(spaces):
            for col, metric in enumerate(local_metrics):
                ax = local_axes[row, col]
                ax.set_facecolor("white")
                for model_index, model_name in enumerate(models):
                    subset = comparison.metrics.loc[
                        (comparison.metrics["space"].astype(str) == space)
                        & (comparison.metrics["model"].astype(str) == model_name)
                    ].sort_values("time")
                    ax.plot(
                        subset["time"],
                        subset[metric],
                        marker=markers[model_index % len(markers)],
                        linestyle=line_styles[model_index % len(line_styles)],
                        linewidth=1.8,
                        label=textwrap.fill(model_name.replace("_", " "), width=34),
                    )
                ax.axhline(
                    ideal_values[metric],
                    color="#777777",
                    linewidth=0.9,
                    linestyle=":",
                    alpha=0.8,
                )
                ax.set_title(f"{space}: {titles[metric]}")
                ax.set_xlabel("Time")
                ax.set_ylabel(titles[metric])
                ax.grid(alpha=0.25)
        handles, labels = local_axes[0, 0].get_legend_handles_labels()
        if handles:
            legend_columns = min(2, max(1, len(labels)))
            local_fig.legend(
                handles,
                labels,
                loc="upper center",
                ncol=legend_columns,
            )
            local_fig.subplots_adjust(
                top=0.81 if len(labels) > 2 else 0.88,
                bottom=0.07,
                hspace=0.52,
                wspace=0.3,
            )
        local_fig_path = output_dir / "model_local_structure_comparison.svg"
        local_fig.savefig(
            local_fig_path,
            bbox_inches="tight",
            facecolor="white",
        )
        plt.close(local_fig)
        paths["local_structure_figure"] = str(local_fig_path)

    return paths
