from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Callable, Mapping, Optional, Sequence, TYPE_CHECKING

import numpy as np
import pandas as pd
from PIL import Image, ImageChops

from .downstream_data import (
    adata_to_aligned_dataframe,
    infer_feature_columns,
    infer_time_key,
    parse_time_value,
)
from .evaluation import compute_distribution_metrics
from .pipeline_utils import set_global_random_seed
from .runtime import build_dynamical_runtime
from .simulation import simulate_sde_points_split_from_x0

if TYPE_CHECKING:
    import anndata as ad


__all__ = [
    "AblationGifResult",
    "AblationPanelSeriesResult",
    "VirtualAblationResult",
    "compute_virtual_ablation_metrics",
    "crop_ablation_panel",
    "export_ablation_gifs",
    "export_ablation_panel_series",
    "run_virtual_cell_type_ablation",
    "trim_white",
]


@dataclass(frozen=True)
class AblationPanelSeriesResult:
    output_dir: Path
    files: list[Path]


@dataclass(frozen=True)
class AblationGifResult:
    output_dir: Path
    gifs: list[Path]


@dataclass(frozen=True)
class VirtualAblationResult:
    """Outputs from a baseline-versus-cell-type-removal experiment.

    ``baseline_points`` and every value in ``ablation_points`` are split-SDE
    trajectories aligned to ``time_points``.  The optional label trajectories
    come from the caller-supplied ``trajectory_labeler``; the dynamical
    simulation itself never requires a classifier.
    """

    start_time: float
    time_points: tuple[float, ...]
    initial_obs_names: tuple[str, ...]
    baseline_points: np.ndarray
    ablation_points: Mapping[str, np.ndarray]
    baseline_labels: Optional[tuple[np.ndarray, ...]]
    ablation_labels: Mapping[str, tuple[np.ndarray, ...]]
    metrics: pd.DataFrame
    label_composition: pd.DataFrame
    output_dir: Optional[Path]
    files: tuple[Path, ...]
    settings: Mapping[str, object]


def _as_trajectory(points, *, name: str, n_times: int) -> np.ndarray:
    frames = [np.asarray(frame, dtype=np.float32) for frame in points]
    if len(frames) != int(n_times):
        raise ValueError(
            f"{name} returned {len(frames)} frames, expected {n_times}."
        )
    feature_dims = {frame.shape[1] for frame in frames if frame.ndim == 2}
    if any(frame.ndim != 2 for frame in frames):
        bad = [frame.shape for frame in frames if frame.ndim != 2]
        raise ValueError(f"{name} contains non-matrix frames: {bad}.")
    if len(feature_dims) != 1:
        raise ValueError(
            f"{name} has inconsistent feature dimensions: {sorted(feature_dims)}."
        )
    trajectory = np.empty(len(frames), dtype=object)
    trajectory[:] = frames
    return trajectory


def _as_label_trajectory(
    labels,
    *,
    points: np.ndarray,
    name: str,
) -> tuple[np.ndarray, ...]:
    arrays = tuple(np.asarray(values).astype(str).reshape(-1) for values in labels)
    if len(arrays) != len(points):
        raise ValueError(
            f"{name} returned {len(arrays)} label frames, expected {len(points)}."
        )
    for idx, (values, frame) in enumerate(zip(arrays, points)):
        if values.shape[0] != np.asarray(frame).shape[0]:
            raise ValueError(
                f"{name} frame {idx} has {values.shape[0]} labels for "
                f"{np.asarray(frame).shape[0]} points."
            )
    return arrays


def _rms_radius(points: np.ndarray) -> float:
    if points.shape[0] == 0:
        return float("nan")
    centered = points - np.mean(points, axis=0, keepdims=True)
    return float(np.sqrt(np.mean(np.sum(centered * centered, axis=1))))


def compute_virtual_ablation_metrics(
    baseline_points,
    ablation_points: Mapping[str, object],
    time_points: Sequence[float],
    *,
    spatial_dim: int = 2,
    max_ot_points: Optional[int] = 1024,
    random_seed: int = 42,
    paired_ot_support: bool = False,
) -> pd.DataFrame:
    """Compare ablated split-SDE distributions with their baseline.

    Metrics are deliberately model- and dataset-agnostic.  For every variant,
    time, and available feature space (joint, spatial, latent), the table
    reports Wasserstein-1/Wasserstein-2 distance to the matched baseline,
    particle counts, centroid displacement, and RMS cloud radius.  W1/W2 use
    the same empirical optimal-transport implementation as the generic
    distribution evaluator.  The OT solve is exact on retained support; when
    either uniform cloud exceeds ``max_ot_points``, it is deterministically
    subsampled without replacement and the retained counts are recorded.
    ``paired_ot_support=True`` is reserved for identity-aligned clouds: it
    requires equal row counts and applies one shared set of row indices to the
    baseline and ablation clouds before OT.  This prevents a point cap from
    manufacturing non-zero distance between identical aligned clouds.

    Particle count is listed on every space row to keep the table tidy and
    directly groupable by ``variant/time/space``.  Ablation particles are the
    ``predicted`` distribution and baseline particles are the ``observed``
    distribution for the OT call; both receive uniform empirical weights.
    """

    times = tuple(float(value) for value in time_points)
    if not times:
        raise ValueError("time_points must be non-empty.")
    baseline = _as_trajectory(
        baseline_points, name="baseline_points", n_times=len(times)
    )
    if len(baseline) == 0:
        raise ValueError("baseline_points must be non-empty.")
    feature_dim = int(np.asarray(baseline[0]).shape[1])
    spatial_dim = int(spatial_dim)
    if spatial_dim < 0 or spatial_dim > feature_dim:
        raise ValueError(
            f"spatial_dim must be in [0, {feature_dim}], got {spatial_dim}."
        )
    if max_ot_points is not None and int(max_ot_points) <= 0:
        raise ValueError("max_ot_points must be positive or None.")

    spaces: dict[str, slice] = {"joint": slice(0, feature_dim)}
    if spatial_dim > 0:
        spaces["spatial"] = slice(0, spatial_dim)
    if spatial_dim < feature_dim:
        spaces["latent"] = slice(spatial_dim, feature_dim)

    rows: list[dict[str, object]] = []
    for variant_index, (variant, raw_points) in enumerate(ablation_points.items()):
        variant_points = _as_trajectory(
            raw_points,
            name=f"ablation_points[{variant!r}]",
            n_times=len(times),
        )
        for time_index, (time_value, baseline_frame, variant_frame) in enumerate(
            zip(times, baseline, variant_points)
        ):
            base = np.asarray(baseline_frame, dtype=np.float64)
            ablated = np.asarray(variant_frame, dtype=np.float64)
            if base.shape[1] != feature_dim or ablated.shape[1] != feature_dim:
                raise ValueError(
                    f"Feature mismatch at time {time_value}: baseline={base.shape}, "
                    f"variant={ablated.shape}."
                )
            n_baseline = int(base.shape[0])
            n_ablation = int(ablated.shape[0])
            count_ratio = (
                float(n_ablation / n_baseline)
                if n_baseline > 0
                else float("nan")
            )
            paired_indices: Optional[np.ndarray] = None
            paired_seed = (
                int(random_seed)
                + 100_000 * int(variant_index)
                + 100 * int(time_index)
            )
            if bool(paired_ot_support):
                if n_baseline != n_ablation:
                    raise ValueError(
                        "paired_ot_support=True requires equal baseline and "
                        "ablation row counts at every time point."
                    )
                retained = n_baseline
                if max_ot_points is not None:
                    retained = min(retained, int(max_ot_points))
                if retained < n_baseline:
                    paired_indices = np.sort(
                        np.random.default_rng(paired_seed).choice(
                            n_baseline,
                            size=retained,
                            replace=False,
                        )
                    )
                else:
                    paired_indices = np.arange(n_baseline, dtype=int)
            for space_index, (space, columns) in enumerate(spaces.items()):
                base_space = base[:, columns]
                ablated_space = ablated[:, columns]
                if n_baseline > 0 and n_ablation > 0:
                    centroid_shift = float(
                        np.linalg.norm(
                            np.mean(ablated_space, axis=0)
                            - np.mean(base_space, axis=0)
                        )
                    )
                    ot_seed = (
                        paired_seed
                        if bool(paired_ot_support)
                        else (
                            int(random_seed)
                            + 100_000 * int(variant_index)
                            + 100 * int(time_index)
                            + int(space_index)
                        )
                    )
                    if bool(paired_ot_support):
                        if paired_indices is None:
                            raise RuntimeError(
                                "Internal error: paired OT indices were not initialized."
                            )
                        distribution_metrics = compute_distribution_metrics(
                            ablated_space[paired_indices],
                            base_space[paired_indices],
                            max_ot_points=None,
                            random_seed=ot_seed,
                        )
                    else:
                        distribution_metrics = compute_distribution_metrics(
                            ablated_space,
                            base_space,
                            max_ot_points=max_ot_points,
                            random_seed=ot_seed,
                        )
                else:
                    centroid_shift = float("nan")
                    ot_seed = (
                        int(random_seed)
                        + 100_000 * int(variant_index)
                        + 100 * int(time_index)
                        + int(space_index)
                    )
                    distribution_metrics = {
                        "w1": float("nan"),
                        "w2": float("nan"),
                        "ot_predicted_points": 0,
                        "ot_observed_points": 0,
                    }
                baseline_radius = _rms_radius(base_space)
                ablation_radius = _rms_radius(ablated_space)
                rows.append(
                    {
                        "variant": str(variant),
                        "time_index": int(time_index),
                        "time": float(time_value),
                        "space": space,
                        "n_baseline": n_baseline,
                        "n_ablation": n_ablation,
                        "count_delta": int(n_ablation - n_baseline),
                        "count_ratio": count_ratio,
                        "w1": float(distribution_metrics["w1"]),
                        "w2": float(distribution_metrics["w2"]),
                        "ot_ablation_points": int(
                            distribution_metrics["ot_predicted_points"]
                        ),
                        "ot_baseline_points": int(
                            distribution_metrics["ot_observed_points"]
                        ),
                        "ot_random_seed": int(ot_seed),
                        "ot_support_is_identity_paired": bool(paired_ot_support),
                        "ot_sampling_policy": (
                            "identity_paired_shared_indices"
                            if bool(paired_ot_support)
                            else "independent_empirical_support"
                        ),
                        "ot_support_index_sha256": (
                            sha256(
                                np.asarray(
                                    paired_indices,
                                    dtype="<i8",
                                ).tobytes()
                            ).hexdigest()
                            if paired_indices is not None
                            else None
                        ),
                        "centroid_shift": centroid_shift,
                        "baseline_rms_radius": baseline_radius,
                        "ablation_rms_radius": ablation_radius,
                        "rms_radius_delta": float(
                            ablation_radius - baseline_radius
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _compute_label_composition(
    *,
    baseline_labels: tuple[np.ndarray, ...],
    ablation_labels: Mapping[str, tuple[np.ndarray, ...]],
    time_points: Sequence[float],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for variant, variant_labels in ablation_labels.items():
        for time_index, (time_value, baseline, ablated) in enumerate(
            zip(time_points, baseline_labels, variant_labels)
        ):
            baseline_values, baseline_counts = np.unique(
                np.asarray(baseline).astype(str), return_counts=True
            )
            ablation_values, ablation_counts = np.unique(
                np.asarray(ablated).astype(str), return_counts=True
            )
            baseline_map = dict(zip(baseline_values.tolist(), baseline_counts.tolist()))
            ablation_map = dict(zip(ablation_values.tolist(), ablation_counts.tolist()))
            total_baseline = int(np.sum(baseline_counts))
            total_ablation = int(np.sum(ablation_counts))
            for label in sorted(set(baseline_map) | set(ablation_map)):
                baseline_count = int(baseline_map.get(label, 0))
                ablation_count = int(ablation_map.get(label, 0))
                baseline_fraction = (
                    float(baseline_count / total_baseline)
                    if total_baseline > 0
                    else float("nan")
                )
                ablation_fraction = (
                    float(ablation_count / total_ablation)
                    if total_ablation > 0
                    else float("nan")
                )
                rows.append(
                    {
                        "variant": str(variant),
                        "time_index": int(time_index),
                        "time": float(time_value),
                        "label": str(label),
                        "baseline_count": baseline_count,
                        "ablation_count": ablation_count,
                        "count_delta": int(ablation_count - baseline_count),
                        "baseline_fraction": baseline_fraction,
                        "ablation_fraction": ablation_fraction,
                        "fraction_delta": float(
                            ablation_fraction - baseline_fraction
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(text)).strip("_.-")
    return slug or "ablation"


def _select_snapshot_indices(
    time_points: Sequence[float], snapshot_times: Optional[Sequence[float]]
) -> list[int]:
    times = np.asarray(time_points, dtype=float)
    if snapshot_times is None:
        count = min(5, len(times))
        return np.unique(np.linspace(0, len(times) - 1, count).round().astype(int)).tolist()
    indices: list[int] = []
    for requested in snapshot_times:
        matches = np.flatnonzero(
            np.isclose(times, float(requested), rtol=0.0, atol=1e-8)
        )
        if matches.size == 0:
            raise ValueError(
                f"snapshot time {requested} is not present in time_points={times.tolist()}."
            )
        indices.append(int(matches[0]))
    return list(dict.fromkeys(indices))


def _trajectory_axis_limits(
    trajectories: Sequence[np.ndarray], plot_dims: tuple[int, int]
) -> tuple[float, float, float, float]:
    nonempty = [
        np.asarray(frame, dtype=np.float64)[:, plot_dims]
        for trajectory in trajectories
        for frame in trajectory
        if np.asarray(frame).shape[0] > 0
    ]
    if not nonempty:
        return (-1.0, 1.0, -1.0, 1.0)
    values = np.concatenate(nonempty, axis=0)
    x_min, y_min = np.nanmin(values, axis=0)
    x_max, y_max = np.nanmax(values, axis=0)
    x_pad = max(1e-6, float(x_max - x_min) * 0.05)
    y_pad = max(1e-6, float(y_max - y_min) * 0.05)
    return (
        float(x_min - x_pad),
        float(x_max + x_pad),
        float(y_min - y_pad),
        float(y_max + y_pad),
    )


def _render_virtual_ablation_snapshots(
    *,
    baseline_points: np.ndarray,
    ablation_points: Mapping[str, np.ndarray],
    time_points: Sequence[float],
    baseline_labels: Optional[tuple[np.ndarray, ...]],
    ablation_labels: Mapping[str, tuple[np.ndarray, ...]],
    label_to_color: Optional[Mapping[str, str]],
    snapshot_times: Optional[Sequence[float]],
    plot_dims: tuple[int, int],
    point_size: float,
    point_alpha: float,
    formats: Sequence[str],
    out_dir: Path,
) -> list[Path]:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    format_values = tuple(str(value).lower().lstrip(".") for value in formats)
    unsupported = sorted(set(format_values) - {"png", "pdf", "svg"})
    if unsupported:
        raise ValueError(f"Unsupported snapshot formats: {unsupported}.")
    indices = _select_snapshot_indices(time_points, snapshot_times)
    axis_limits = _trajectory_axis_limits(
        [baseline_points, *ablation_points.values()], plot_dims
    )

    colors = dict(label_to_color or {})
    if baseline_labels is not None and not colors:
        all_labels = sorted(
            {
                str(label)
                for arrays in [baseline_labels, *ablation_labels.values()]
                for values in arrays
                for label in values
            }
        )
        cmap = plt.get_cmap("tab20")
        for index, label in enumerate(all_labels):
            rgb = cmap(index % cmap.N)[:3]
            colors[label] = "#{:02x}{:02x}{:02x}".format(
                int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
            )

    files: list[Path] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for variant, variant_points in ablation_points.items():
        variant_dir = out_dir / _slugify(variant)
        variant_dir.mkdir(parents=True, exist_ok=True)
        for time_index in indices:
            baseline = np.asarray(baseline_points[time_index], dtype=np.float32)
            ablated = np.asarray(variant_points[time_index], dtype=np.float32)
            fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.4), dpi=180)
            for side, (ax, points, title) in enumerate(
                zip(
                    axes,
                    (baseline, ablated),
                    ("Baseline", f"Ablation: {variant}"),
                )
            ):
                if baseline_labels is None:
                    point_colors = "#4c78a8" if side == 0 else "#e45756"
                else:
                    labels = (
                        baseline_labels[time_index]
                        if side == 0
                        else ablation_labels[variant][time_index]
                    )
                    point_colors = [colors.get(str(value), "#888888") for value in labels]
                if points.shape[0] > 0:
                    ax.scatter(
                        points[:, plot_dims[0]],
                        points[:, plot_dims[1]],
                        s=float(point_size),
                        c=point_colors,
                        alpha=float(point_alpha),
                        linewidths=0,
                        rasterized=points.shape[0] > 30000,
                    )
                ax.set_xlim(axis_limits[0], axis_limits[1])
                ax.set_ylim(axis_limits[2], axis_limits[3])
                ax.set_aspect("equal")
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                ax.set_title(f"{title}\n(n={points.shape[0]})", fontsize=10)
            fig.suptitle(f"t = {float(time_points[time_index]):g}")
            fig.tight_layout()
            stem = variant_dir / (
                f"frame_{time_index:03d}_t_{float(time_points[time_index]):g}"
            )
            for extension in format_values:
                path = stem.with_suffix(f".{extension}")
                fig.savefig(path, bbox_inches="tight")
                files.append(path)
            plt.close(fig)

    if colors:
        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor=color,
                markeredgecolor="none",
                label=label,
                markersize=6,
            )
            for label, color in colors.items()
        ]
        fig_height = max(2.0, min(12.0, 0.28 * len(handles) + 0.5))
        fig, ax = plt.subplots(figsize=(4.5, fig_height), dpi=180)
        ax.legend(handles=handles, loc="center left", frameon=False)
        ax.axis("off")
        for extension in format_values:
            path = out_dir / f"label_legend.{extension}"
            fig.savefig(path, bbox_inches="tight")
            files.append(path)
        plt.close(fig)
    return files


def run_virtual_cell_type_ablation(
    adata: "ad.AnnData",
    model,
    *,
    ablations: Mapping[str, Sequence[str]],
    time_points: Sequence[float],
    output_dir: Optional[str | Path] = None,
    time_index: int = 0,
    n_samples: Optional[int] = None,
    dt: float = 0.05,
    resample_dt: Optional[float] = None,
    sigma: float = 0.03,
    sigma_by_dim: Optional[Sequence[float]] = None,
    growth_alpha: float = 1.0,
    interaction_m: int = 1024,
    max_particles: Optional[int] = None,
    device: str = "cuda",
    time_key: Optional[str] = "time_point_processed",
    annotation_key: str = "Annotation",
    obsm_key: str = "X_latent",
    spatial_key: str = "spatial_aligned",
    concat_spatial: Optional[bool] = True,
    spatial_dim: int = 2,
    random_seed: int = 42,
    common_random_seed: bool = True,
    max_ot_points: Optional[int] = 1024,
    trajectory_labeler: Optional[
        Callable[[np.ndarray, Sequence[float]], Sequence[Sequence[object]]]
    ] = None,
    save_data: bool = True,
    save_snapshots: bool = True,
    snapshot_times: Optional[Sequence[float]] = None,
    snapshot_plot_dims: tuple[int, int] = (0, 1),
    snapshot_point_size: float = 4.0,
    snapshot_alpha: float = 0.85,
    snapshot_formats: Sequence[str] = ("png", "pdf"),
    label_to_color: Optional[Mapping[str, str]] = None,
    verbose: bool = True,
) -> VirtualAblationResult:
    """Run reproducible split-SDE baseline and virtual cell-type removals.

    The initial cohort is selected once from ``time_index``.  Every ablation
    branch is an exact subset of that cohort after removing one or more labels;
    removed cells are never replaced.  The workflow does not re-anchor or warp
    trajectories.  This makes the API suitable for virtual sensitivity
    diagnostics across datasets while leaving label names and classifier choice
    to callers; it is not, by itself, a causal intervention estimate.

    Parameters
    ----------
    ablations
        Mapping from a caller-defined variant name to one or more exact labels
        in ``adata.obs[annotation_key]``.  For example, two independent
        experiments can be expressed as ``{"remove_A": ["A"],
        "remove_B": ["B"]}``; a joint removal uses ``["A", "B"]``.
    trajectory_labeler
        Optional callable invoked as ``labeler(points, time_points)`` for the
        baseline and every variant.  It may wrap
        :func:`predict_labels_for_trajectories`; its returned label arrays are
        validated, exported, and used for composition metrics/snapshots.
    common_random_seed
        If true, every branch starts from the same SDE seed.  If false, branch
        seeds are ``random_seed + branch_index``.  The exact selected starting
        cohort is shared under either setting.  Because removals change tensor
        shape and row order, equal branch seeds do not provide cell-ID-matched
        Brownian increments; use multiple seeds for inferential uncertainty.
    growth_alpha
        Multiplier applied to the learned growth rate during split events.
        It is forwarded unchanged to
        :func:`simulate_sde_points_split_from_x0` and recorded in the manifest.
    resample_dt
        Fixed population split/extinction event interval.  When provided,
        requested output frames no longer determine the biological resampling
        schedule.  Use the integration ``dt`` for a frame-grid-invariant run.
    max_particles
        Optional fail-fast ceiling checked before split-event allocation.  This
        prevents an unexpectedly large learned growth rate from exhausting
        memory; it does not downsample or otherwise alter a valid trajectory.
    max_ot_points
        Maximum number of ablation and baseline particles used by each exact
        W1/W2 calculation.  Larger clouds are deterministically subsampled;
        pass ``None`` to use every particle.
    """

    if not (hasattr(adata, "obs") and hasattr(adata, "obsm")):
        raise TypeError(f"adata must be AnnData-like, got {type(adata)}.")
    if annotation_key not in adata.obs.columns:
        raise KeyError(f"adata.obs['{annotation_key}'] is missing.")
    if not ablations:
        raise ValueError("ablations must contain at least one variant.")
    times = tuple(float(value) for value in time_points)
    if not times:
        raise ValueError("time_points must be non-empty.")
    if any(b <= a for a, b in zip(times[:-1], times[1:])):
        raise ValueError("time_points must be strictly increasing.")
    if not np.isfinite(float(dt)) or float(dt) <= 0:
        raise ValueError("dt must be finite and > 0.")
    if resample_dt is not None and (
        not np.isfinite(float(resample_dt)) or float(resample_dt) <= 0
    ):
        raise ValueError("resample_dt must be finite and > 0 when provided.")
    if not np.isfinite(float(sigma)) or float(sigma) < 0:
        raise ValueError("sigma must be finite and >= 0.")
    if not np.isfinite(float(growth_alpha)):
        raise ValueError("growth_alpha must be finite.")
    if n_samples is not None and int(n_samples) <= 0:
        raise ValueError("n_samples must be positive or None.")
    if max_particles is not None and int(max_particles) <= 0:
        raise ValueError("max_particles must be positive or None.")
    if max_ot_points is not None and int(max_ot_points) <= 0:
        raise ValueError("max_ot_points must be positive or None.")

    normalized_ablations: dict[str, tuple[str, ...]] = {}
    file_stems: set[str] = set()
    for raw_name, raw_labels in ablations.items():
        name = str(raw_name).strip()
        if not name:
            raise ValueError("Ablation names must be non-empty.")
        raw_label_values = (raw_labels,) if isinstance(raw_labels, str) else raw_labels
        labels = tuple(
            dict.fromkeys(
                str(label)
                for label in raw_label_values
                if str(label).strip() != ""
            )
        )
        if not labels:
            raise ValueError(f"Ablation {name!r} does not specify any labels.")
        stem = _slugify(name)
        if stem in file_stems:
            raise ValueError(
                f"Ablation names produce a duplicate output stem: {stem!r}."
            )
        file_stems.add(stem)
        normalized_ablations[name] = labels

    resolved_time_key = infer_time_key(adata.obs, preferred=time_key)
    observed_times = np.asarray(
        [parse_time_value(value) for value in adata.obs[resolved_time_key]],
        dtype=np.float64,
    )
    unique_times = sorted(float(value) for value in np.unique(observed_times))
    time_index = int(time_index)
    if time_index < 0 or time_index >= len(unique_times):
        raise ValueError(
            f"time_index={time_index} out of range [0, {len(unique_times) - 1}]."
        )
    start_time = float(unique_times[time_index])
    if not np.isclose(times[0], start_time, rtol=0.0, atol=1e-8):
        raise ValueError(
            f"time_points must start at selected observed time {start_time}, "
            f"got {times[0]}."
        )

    initial_positions = np.flatnonzero(
        np.isclose(observed_times, start_time, rtol=0.0, atol=1e-9)
    )
    if initial_positions.size == 0:
        raise ValueError(f"No cells found at selected start time {start_time}.")
    rng = np.random.default_rng(int(random_seed))
    if n_samples is not None and initial_positions.size > int(n_samples):
        initial_positions = np.sort(
            rng.choice(initial_positions, size=int(n_samples), replace=False)
        )

    initial_labels = (
        adata.obs.iloc[initial_positions][annotation_key].astype(str).to_numpy()
    )
    available_labels = set(initial_labels.tolist())
    requested_labels = {
        label for labels in normalized_ablations.values() for label in labels
    }
    missing_labels = sorted(requested_labels - available_labels)
    if missing_labels:
        raise ValueError(
            f"Ablation labels are absent at start time {start_time}: {missing_labels}. "
            f"Available labels: {sorted(available_labels)}."
        )

    branch_positions: dict[str, np.ndarray] = {}
    for name, labels in normalized_ablations.items():
        keep = ~np.isin(initial_labels, np.asarray(labels, dtype=str))
        if not np.any(keep):
            raise ValueError(
                f"Ablation {name!r} removes every cell in the selected initial cohort."
            )
        branch_positions[name] = initial_positions[keep]

    aligned_frame, aligned_time_key = adata_to_aligned_dataframe(
        adata,
        time_key=resolved_time_key,
        obsm_key=obsm_key,
        spatial_key=spatial_key,
        concat_spatial=concat_spatial,
        annotation_key=annotation_key,
    )
    if aligned_time_key != resolved_time_key:
        raise RuntimeError(
            f"Aligned-data time key changed unexpectedly: {resolved_time_key!r} "
            f"-> {aligned_time_key!r}."
        )
    feature_columns = list(
        infer_feature_columns(aligned_frame, annotation_column=annotation_key)
    )
    feature_matrix = aligned_frame[feature_columns].to_numpy(dtype=np.float32)
    if feature_matrix.shape[0] != int(adata.n_obs):
        raise RuntimeError(
            f"Aligned feature matrix has {feature_matrix.shape[0]} rows for "
            f"AnnData with {adata.n_obs} observations."
        )

    if hasattr(model, "f_net") and hasattr(model, "score_net"):
        runtime = model
    else:
        runtime = build_dynamical_runtime(model)
    runtime_model = getattr(runtime, "model", model)
    if hasattr(runtime_model, "to"):
        runtime_model.to(device)
    if hasattr(runtime_model, "eval"):
        runtime_model.eval()

    def simulate_positions(positions: np.ndarray, *, seed: int, name: str) -> np.ndarray:
        set_global_random_seed(int(seed))
        points = simulate_sde_points_split_from_x0(
            x0=feature_matrix[positions],
            f_net=runtime.f_net,
            score_net=runtime.score_net,
            ts_points=times,
            dt=float(dt),
            sigma=float(sigma),
            sigma_by_dim=sigma_by_dim,
            growth_alpha=float(growth_alpha),
            interaction_m=int(interaction_m),
            device=device,
            verbose=bool(verbose),
            resample_dt=resample_dt,
            max_particles=max_particles,
        )
        return _as_trajectory(points, name=name, n_times=len(times))

    baseline_points = simulate_positions(
        initial_positions, seed=int(random_seed), name="baseline"
    )
    variant_points: dict[str, np.ndarray] = {}
    simulation_seeds = {"baseline": int(random_seed)}
    for branch_index, (name, positions) in enumerate(branch_positions.items(), start=1):
        seed = int(random_seed) if common_random_seed else int(random_seed) + branch_index
        simulation_seeds[name] = seed
        variant_points[name] = simulate_positions(
            positions, seed=seed, name=f"ablation[{name!r}]"
        )

    baseline_labels: Optional[tuple[np.ndarray, ...]] = None
    variant_labels: dict[str, tuple[np.ndarray, ...]] = {}
    if trajectory_labeler is not None:
        baseline_labels = _as_label_trajectory(
            trajectory_labeler(baseline_points, times),
            points=baseline_points,
            name="trajectory_labeler(baseline)",
        )
        for name, points in variant_points.items():
            variant_labels[name] = _as_label_trajectory(
                trajectory_labeler(points, times),
                points=points,
                name=f"trajectory_labeler({name!r})",
            )

    metrics = compute_virtual_ablation_metrics(
        baseline_points,
        variant_points,
        times,
        spatial_dim=int(spatial_dim),
        max_ot_points=max_ot_points,
        random_seed=int(random_seed),
    )
    label_composition = (
        _compute_label_composition(
            baseline_labels=baseline_labels,
            ablation_labels=variant_labels,
            time_points=times,
        )
        if baseline_labels is not None
        else pd.DataFrame(
            columns=[
                "variant",
                "time_index",
                "time",
                "label",
                "baseline_count",
                "ablation_count",
                "count_delta",
                "baseline_fraction",
                "ablation_fraction",
                "fraction_delta",
            ]
        )
    )

    settings: dict[str, object] = {
        "start_time": start_time,
        "time_index": time_index,
        "time_key": resolved_time_key,
        "annotation_key": annotation_key,
        "obsm_key": obsm_key,
        "spatial_key": spatial_key,
        "concat_spatial": concat_spatial,
        "spatial_dim": int(spatial_dim),
        "n_initial": int(initial_positions.size),
        "n_samples_cap": None if n_samples is None else int(n_samples),
        "dt": float(dt),
        "resample_dt": None if resample_dt is None else float(resample_dt),
        "sigma": float(sigma),
        "sigma_by_dim": (
            None
            if sigma_by_dim is None
            else [float(value) for value in sigma_by_dim]
        ),
        "growth_alpha": float(growth_alpha),
        "interaction_m": int(interaction_m),
        "max_particles": (
            None if max_particles is None else int(max_particles)
        ),
        "device": str(device),
        "random_seed": int(random_seed),
        "common_random_seed": bool(common_random_seed),
        "max_ot_points": (
            None if max_ot_points is None else int(max_ot_points)
        ),
        "distribution_metrics": (
            "uniform empirical W1/W2 from ablation branch to matched baseline; "
            "deterministic OT subsampling when max_ot_points is exceeded"
        ),
        "random_stream_coupling": (
            "same branch-level seed; not cell-ID-matched after cohort removal"
            if common_random_seed
            else "independent deterministic branch seeds"
        ),
        "simulation_seeds": simulation_seeds,
        "ablations": {
            name: list(labels) for name, labels in normalized_ablations.items()
        },
        "initial_counts": {
            str(label): int(np.sum(initial_labels == label))
            for label in sorted(available_labels)
        },
        "variant_initial_counts": {
            name: int(len(positions)) for name, positions in branch_positions.items()
        },
        "simulation": "continuous split-SDE; no re-anchoring; no spatial warp; no replacement",
    }

    out_path = Path(output_dir) if output_dir is not None else None
    files: list[Path] = []
    if out_path is not None:
        out_path.mkdir(parents=True, exist_ok=True)
        if save_data:
            trajectories_dir = out_path / "trajectories"
            trajectories_dir.mkdir(parents=True, exist_ok=True)
            baseline_path = trajectories_dir / "baseline_points.npy"
            np.save(baseline_path, baseline_points, allow_pickle=True)
            files.append(baseline_path)
            if baseline_labels is not None:
                baseline_label_path = trajectories_dir / "baseline_labels.npy"
                np.save(
                    baseline_label_path,
                    np.asarray(baseline_labels, dtype=object),
                    allow_pickle=True,
                )
                files.append(baseline_label_path)
            for name, points in variant_points.items():
                stem = _slugify(name)
                point_path = trajectories_dir / f"{stem}_points.npy"
                np.save(point_path, points, allow_pickle=True)
                files.append(point_path)
                if name in variant_labels:
                    label_path = trajectories_dir / f"{stem}_labels.npy"
                    np.save(
                        label_path,
                        np.asarray(variant_labels[name], dtype=object),
                        allow_pickle=True,
                    )
                    files.append(label_path)

            metrics_path = out_path / "ablation_metrics.csv"
            metrics.to_csv(metrics_path, index=False)
            files.append(metrics_path)
            if baseline_labels is not None:
                composition_path = out_path / "label_composition.csv"
                label_composition.to_csv(composition_path, index=False)
                files.append(composition_path)

            cohort = pd.DataFrame(
                {
                    "obs_name": np.asarray(adata.obs_names.astype(str))[initial_positions],
                    "initial_label": initial_labels,
                }
            )
            for name, labels in normalized_ablations.items():
                cohort[f"kept__{_slugify(name)}"] = ~np.isin(
                    initial_labels, np.asarray(labels, dtype=str)
                )
            cohort_path = out_path / "initial_cohort.csv"
            cohort.to_csv(cohort_path, index=False)
            files.append(cohort_path)

            manifest_path = out_path / "manifest.json"
            manifest = {
                "schema_version": 1,
                "time_points": list(times),
                "settings": settings,
                "trajectory_shapes": {
                    "baseline": [list(np.asarray(frame).shape) for frame in baseline_points],
                    **{
                        name: [list(np.asarray(frame).shape) for frame in points]
                        for name, points in variant_points.items()
                    },
                },
            }
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
            )
            files.append(manifest_path)

        if save_snapshots:
            if len(snapshot_plot_dims) != 2 or min(snapshot_plot_dims) < 0:
                raise ValueError("snapshot_plot_dims must contain two non-negative dimensions.")
            feature_dim = int(np.asarray(baseline_points[0]).shape[1])
            if max(snapshot_plot_dims) >= feature_dim:
                raise ValueError(
                    f"snapshot_plot_dims={snapshot_plot_dims} exceeds feature dimension {feature_dim}."
                )
            files.extend(
                _render_virtual_ablation_snapshots(
                    baseline_points=baseline_points,
                    ablation_points=variant_points,
                    time_points=times,
                    baseline_labels=baseline_labels,
                    ablation_labels=variant_labels,
                    label_to_color=label_to_color,
                    snapshot_times=snapshot_times,
                    plot_dims=tuple(int(value) for value in snapshot_plot_dims),
                    point_size=float(snapshot_point_size),
                    point_alpha=float(snapshot_alpha),
                    formats=snapshot_formats,
                    out_dir=out_path / "snapshots",
                )
            )

    return VirtualAblationResult(
        start_time=start_time,
        time_points=times,
        initial_obs_names=tuple(
            np.asarray(adata.obs_names.astype(str))[initial_positions].tolist()
        ),
        baseline_points=baseline_points,
        ablation_points=variant_points,
        baseline_labels=baseline_labels,
        ablation_labels=variant_labels,
        metrics=metrics,
        label_composition=label_composition,
        output_dir=out_path,
        files=tuple(files),
        settings=settings,
    )


def trim_white(img: Image.Image) -> Image.Image:
    bg = Image.new("RGB", img.size, "white")
    diff = ImageChops.difference(img.convert("RGB"), bg)
    bbox = diff.getbbox()
    if bbox is None:
        return img
    x0, y0, x1, y1 = bbox
    return img.crop((max(0, x0 - 2), max(0, y0 - 2), min(img.width, x1 + 2), min(img.height, y1 + 2)))


def crop_ablation_panel(frame_path: str | Path, side: str) -> Image.Image:
    frame_path = Path(frame_path)
    img = Image.open(frame_path).convert("RGB")
    w, h = img.size
    if side == "baseline":
        box = (int(0.065 * w), int(0.19 * h), int(0.455 * w), int(0.945 * h))
    elif side == "ablation":
        box = (int(0.545 * w), int(0.19 * h), int(0.935 * w), int(0.945 * h))
    else:
        raise ValueError(f"Unsupported side '{side}'")
    return trim_white(img.crop(box))


def export_ablation_panel_series(
    *,
    baseline_frames_dir: str | Path,
    comparison_frame_dirs: Mapping[str, str | Path],
    frame_ids: Mapping[str, str],
    out_dir: str | Path,
) -> AblationPanelSeriesResult:
    baseline_frames_dir = Path(baseline_frames_dir)
    comparison_frame_dirs = {k: Path(v) for k, v in comparison_frame_dirs.items()}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files: list[Path] = []
    for label, fid in frame_ids.items():
        baseline_png = out_dir / f"baseline__{label}.png"
        crop_ablation_panel(baseline_frames_dir / f"frame_{fid}.png", "baseline").save(baseline_png)
        files.append(baseline_png)
        for variant_name, variant_dir in comparison_frame_dirs.items():
            variant_png = out_dir / f"{variant_name}__{label}.png"
            crop_ablation_panel(variant_dir / f"frame_{fid}.png", "ablation").save(variant_png)
            files.append(variant_png)
    return AblationPanelSeriesResult(output_dir=out_dir, files=files)


def export_ablation_gifs(
    *,
    frame_dirs: Mapping[str, str | Path],
    out_dir: str | Path,
    frame_step: int = 2,
    resize_factor: float = 0.5,
    duration_ms: int = 120,
) -> AblationGifResult:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gif_paths: list[Path] = []

    for label, frame_dir in frame_dirs.items():
        frame_dir = Path(frame_dir)
        frame_paths = sorted(frame_dir.glob("frame_*.png"))[:: max(1, frame_step)]
        if not frame_paths:
            continue
        frames: list[Image.Image] = []
        for path in frame_paths:
            img = Image.open(path).convert("P", palette=Image.ADAPTIVE)
            if resize_factor != 1.0:
                new_size = (
                    max(1, int(img.width * resize_factor)),
                    max(1, int(img.height * resize_factor)),
                )
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            frames.append(img)
        out_path = out_dir / f"{label}.gif"
        frames[0].save(
            out_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration_ms,
            loop=0,
            optimize=False,
            disposal=2,
        )
        gif_paths.append(out_path)

    return AblationGifResult(output_dir=out_dir, gifs=gif_paths)
