#!/usr/bin/env python3
"""Generate, train and evaluate the minimal spatial attraction benchmark."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
from typing import Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import yaml

from CytoBridge.benchmarks.spatial_attraction import (
    SpatialAttractionSpec,
    SUPPORTED_VERSIONS,
    V5_VERSION,
    V6_VERSION,
    V7_VERSION,
    V8_VERSION,
    attraction_coefficient,
    generate_spatial_attraction_benchmark,
    probe_learned_gene_force_projection,
    probe_learned_spatial_attraction,
)


DEFAULT_BACKGROUND_CONFIG = (
    REPO_ROOT / "CytoBridge/configs/spatial_synthetic_background.yaml"
)
DEFAULT_TARGET_CONFIG = (
    REPO_ROOT / "CytoBridge/configs/spatial_synthetic_attraction.yaml"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a YAML mapping: {path}")
    return payload


def _replace_stage_epochs(config: dict, stage: str, epochs: int | None) -> None:
    if epochs is None:
        return
    if int(epochs) <= 0:
        raise ValueError("Epoch overrides must be positive.")
    for item in config["training"]["plan"]:
        if str(item["name"]) == str(stage):
            item["epochs"] = int(epochs)
            return
    raise KeyError(f"Stage not found in config: {stage}")


def _resolve_n_particles(version: str, requested: int | None) -> int:
    if requested is not None:
        return int(requested)
    return 400 if str(version) in {V5_VERSION, V6_VERSION, V7_VERSION, V8_VERSION} else 256


def _resolve_interaction_strength(version: str, requested: float | None) -> float:
    if requested is not None:
        return float(requested)
    if str(version) in {V6_VERSION, V7_VERSION, V8_VERSION}:
        return 0.25
    if str(version) == V5_VERSION:
        return 0.50
    return 1.20


def _resolve_gene_interaction_gain(
    version: str, requested: float | None
) -> float:
    if requested is not None:
        return float(requested)
    return 3.0 if str(version) == V8_VERSION else 1.0


def _stage_checkpoint_path(model_dir: Path, config: Mapping[str, object], stage: str) -> Path:
    plan = config.get("training", {}).get("plan", [])
    for item in plan:
        if str(item.get("name")) == str(stage):
            strategy = str(item.get("save_strategy", "best")).lower()
            if strategy not in {"best", "last"}:
                raise ValueError(
                    f"Unsupported save_strategy {strategy!r} for stage {stage!r}."
                )
            return model_dir / stage / f"{strategy}_model.pth"
    raise KeyError(f"Stage not found in config: {stage}")


def generate(args: argparse.Namespace) -> dict:
    version = str(args.version)
    n_particles = _resolve_n_particles(version, args.n_particles)
    interaction_strength = _resolve_interaction_strength(
        version, args.interaction_strength
    )
    measurement_spatial_sigma = (
        0.005
        if args.measurement_spatial_sigma is None
        and version in {V5_VERSION, V6_VERSION, V7_VERSION, V8_VERSION}
        else 0.004
        if args.measurement_spatial_sigma is None
        else float(args.measurement_spatial_sigma)
    )
    measurement_gene_sigma = (
        0.010
        if args.measurement_gene_sigma is None
        and version in {V5_VERSION, V6_VERSION, V7_VERSION, V8_VERSION}
        else 0.008
        if args.measurement_gene_sigma is None
        else float(args.measurement_gene_sigma)
    )
    spec = SpatialAttractionSpec(
        version=version,
        n_particles=n_particles,
        dt=float(args.dt),
        interaction_cutoff=float(args.cutoff),
        interaction_strength=interaction_strength,
        sigma=float(args.sigma),
        measurement_spatial_sigma=measurement_spatial_sigma,
        measurement_gene_sigma=measurement_gene_sigma,
        target_final_mass_ratio=float(args.target_final_mass_ratio),
        gene_interaction_gain=_resolve_gene_interaction_gain(
            version, args.gene_interaction_gain
        ),
    )
    return dict(generate_spatial_attraction_benchmark(args.data_dir, spec=spec))


def train(args: argparse.Namespace) -> dict:
    from CytoBridge.tl.train import fit

    data_dir = Path(args.data_dir).expanduser().resolve()
    model_root = Path(args.model_root).expanduser().resolve()
    if model_root.exists() and any(model_root.iterdir()):
        raise FileExistsError(f"model_root must be empty: {model_root}")
    model_root.mkdir(parents=True, exist_ok=True)
    background_dir = model_root / "background"
    target_dir = model_root / "attractive"

    background_config_path = Path(args.background_config).expanduser().resolve()
    target_config_path = Path(args.target_config).expanduser().resolve()
    background_config = _load_yaml(background_config_path)
    target_config = _load_yaml(target_config_path)
    background_config["ckpt_dir"] = str(background_dir)
    target_config["ckpt_dir"] = str(target_dir)
    _replace_stage_epochs(
        background_config, "Pretrain_vg", args.background_epochs
    )
    _replace_stage_epochs(
        target_config, "Calibrate_interaction", args.interaction_epochs
    )
    _replace_stage_epochs(target_config, "Train_score", args.score_epochs)

    control_h5ad = data_dir / "no_interaction_observed.h5ad"
    attractive_h5ad = data_dir / "attractive_observed.h5ad"
    for path in (control_h5ad, attractive_h5ad):
        if not path.is_file():
            raise FileNotFoundError(path)

    fit(
        str(control_h5ad),
        config=background_config,
        device=args.device,
        time_key="time_point_processed",
        obsm_key="X_latent",
        is_spatial=True,
        spatial_key="spatial_aligned",
        evaluate_after_training=False,
    )
    source_checkpoint = background_dir / "Pretrain_vg/best_model.pth"
    if not source_checkpoint.is_file():
        raise FileNotFoundError(source_checkpoint)
    target_config["initialization"] = {
        "component_checkpoint": str(source_checkpoint),
        "components": ["velocity", "growth", "score", "interaction"],
    }
    fit(
        str(attractive_h5ad),
        config=target_config,
        device=args.device,
        time_key="time_point_processed",
        obsm_key="X_latent",
        is_spatial=True,
        spatial_key="spatial_aligned",
        evaluate_after_training=False,
    )

    interaction_checkpoint = _stage_checkpoint_path(
        target_dir, target_config, "Calibrate_interaction"
    )
    score_checkpoint = target_dir / "Train_score/score_model.pth"
    for path in (interaction_checkpoint, score_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest = {
        "schema_version": "cytobridge_spatial_attraction_training/1",
        "data_manifest": str(data_dir / "manifest.json"),
        "data_manifest_sha256": _sha256(data_dir / "manifest.json"),
        "device": str(args.device),
        "background": {
            "config_source": str(background_config_path),
            "config_source_sha256": _sha256(background_config_path),
            "resolved_config": str(background_dir / "config.yaml"),
            "checkpoint": str(source_checkpoint),
            "checkpoint_sha256": _sha256(source_checkpoint),
        },
        "attractive": {
            "config_source": str(target_config_path),
            "config_source_sha256": _sha256(target_config_path),
            "resolved_config": str(target_dir / "config.yaml"),
            "interaction_checkpoint": str(interaction_checkpoint),
            "interaction_checkpoint_sha256": _sha256(interaction_checkpoint),
            "score_checkpoint": str(score_checkpoint),
            "score_checkpoint_sha256": _sha256(score_checkpoint),
        },
        "background_is_matched_no_interaction_control": True,
        "force_supervision_used": False,
    }
    manifest_path = model_root / "training_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def _feature_spaces(values: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "joint": values,
        "spatial": values[:, :2],
        "gene": values[:, 2:],
    }


def _plot_snapshot_grid(
    *,
    time_points: Sequence[float],
    ground_truth: np.ndarray,
    predicted: np.ndarray,
    columns: slice,
    axis_names: tuple[str, str],
    title: str,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    times = np.asarray(time_points, dtype=float)
    combined = np.concatenate(
        (ground_truth[:, :, columns], predicted[:, :, columns]), axis=1
    )
    limits = []
    for dimension in range(2):
        low = float(np.min(combined[:, :, dimension]))
        high = float(np.max(combined[:, :, dimension]))
        padding = max(0.04 * (high - low), 1e-3)
        limits.append((low - padding, high + padding))
    fig, axes = plt.subplots(2, len(times), figsize=(2.55 * len(times), 5.2))
    for column, time_value in enumerate(times):
        for row, (values, label, color) in enumerate(
            (
                (ground_truth, "GT", "#555555"),
                (predicted, "CytoBridge", "#007C83"),
            )
        ):
            ax = axes[row, column]
            panel = values[column, :, columns]
            ax.scatter(panel[:, 0], panel[:, 1], s=5, alpha=0.48, color=color)
            ax.set_xlim(*limits[0])
            ax.set_ylim(*limits[1])
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(f"{label}, t={time_value:g}", fontsize=9)
            ax.tick_params(labelsize=7)
            if column == 0:
                ax.set_ylabel(axis_names[1], fontsize=8)
            if row == 1:
                ax.set_xlabel(axis_names[0], fontsize=8)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_w1(metrics: pd.DataFrame, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    summary = (
        metrics.groupby(["space", "condition", "time"], as_index=False)
        .agg(mean=("w1", "mean"), sd=("w1", "std"))
        .fillna({"sd": 0.0})
    )
    spaces = ["joint", "spatial", "gene"]
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.5), sharex=True)
    for axis, space in zip(axes, spaces):
        for condition, color in (("interaction_on", "#007C83"), ("interaction_off", "#CC6677")):
            selected = summary.loc[
                (summary["space"] == space) & (summary["condition"] == condition)
            ].sort_values("time")
            axis.errorbar(
                selected["time"], selected["mean"], yerr=selected["sd"],
                marker="o", linewidth=2, capsize=2, color=color,
                label=condition.replace("_", " "),
            )
        axis.set_title(space)
        axis.set_xlabel("time")
        axis.set_ylabel("W1")
        axis.grid(alpha=0.2)
    axes[0].legend(frameon=False)
    fig.suptitle("Same-checkpoint interaction ablation", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_gene_ablation_delta(
    paired: pd.DataFrame,
    *,
    assessment_start_time: float,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    selected = paired.loc[paired["space"] == "gene"].copy()
    summary = selected.groupby("time", as_index=False).agg(
        mean=("off_minus_on", "mean"),
        sd=("off_minus_on", "std"),
        fraction_positive=(
            "off_minus_on",
            lambda values: float(np.mean(np.asarray(values) > 0)),
        ),
    ).fillna({"sd": 0.0})
    fig, axis = plt.subplots(figsize=(6.2, 3.8))
    for _, seed_values in selected.groupby("seed"):
        seed_values = seed_values.sort_values("time")
        axis.plot(
            seed_values["time"],
            seed_values["off_minus_on"],
            color="#9AA5B1",
            linewidth=0.9,
            alpha=0.6,
        )
    axis.errorbar(
        summary["time"],
        summary["mean"],
        yerr=summary["sd"],
        marker="o",
        linewidth=2.4,
        capsize=3,
        color="#007C83",
        label="mean +/- SD (5 seeds)",
    )
    axis.axhline(0.0, color="#555555", linewidth=1.0)
    axis.axvspan(
        float(assessment_start_time),
        float(summary["time"].max()),
        color="#E6F4F1",
        alpha=0.7,
        label="formal assessment window",
    )
    axis.set_xlabel("time")
    axis.set_ylabel("Gene W1 off - on")
    axis.set_title("Interaction contribution to gene-state accuracy")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _assess_gene_ablation(
    paired_summary: pd.DataFrame,
    *,
    assessment_start_time: float,
    required: bool,
    strict_final_effect_size: float = 0.005,
) -> dict[str, object]:
    """Assess whether the learned interaction consistently helps gene rollout.

    The formal gate encodes the accepted benchmark criterion: improvement must
    be positive at every assessed time and at least 80% of paired inference
    seeds must agree.  The historical absolute effect-size threshold remains a
    separately reported, non-blocking diagnostic.
    """
    selected = paired_summary.loc[
        (paired_summary["space"] == "gene")
        & (paired_summary["time"] >= float(assessment_start_time))
    ].copy()
    final_row = selected.loc[selected["time"] == selected["time"].max()]
    final_delta = (
        float(final_row["off_minus_on"].iloc[0]) if not final_row.empty else None
    )
    strict_passed = bool(
        final_delta is not None and final_delta >= float(strict_final_effect_size)
    )
    passed = bool(
        not required
        or (
            not selected.empty
            and (selected["off_minus_on"] > 0.0).all()
            and (selected["fraction_positive"] >= 0.8).all()
        )
    )
    return {
        "required": bool(required),
        "assessment_start_time": float(assessment_start_time),
        "minimum_final_off_minus_on": float(strict_final_effect_size),
        "minimum_is_blocking": False,
        "strict_effect_size_passed": strict_passed,
        "final_w1_on": (
            float(final_row["w1_on"].iloc[0]) if not final_row.empty else None
        ),
        "final_w1_off": (
            float(final_row["w1_off"].iloc[0]) if not final_row.empty else None
        ),
        "final_off_minus_on": final_delta,
        "passed": passed,
    }


def _plot_pattern(pattern: pd.DataFrame, scale: float, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0), sharex=True)
    axes[0].plot(
        pattern["distance"], pattern["true_coefficient"],
        linewidth=2.4, color="#CC6677", label="GT attraction",
    )
    axes[0].plot(
        pattern["distance"], pattern["learned_coefficient"],
        linewidth=2.0, color="#007C83", label="learned (raw)",
    )
    axes[0].set_title("Raw force magnitude")
    axes[0].legend(frameon=False)
    axes[1].plot(
        pattern["distance"], pattern["true_coefficient"],
        linewidth=2.4, color="#CC6677", label="GT attraction",
    )
    axes[1].plot(
        pattern["distance"],
        pattern["learned_coefficient"] * float(scale),
        linewidth=2.0, color="#007C83", linestyle="--",
        label=f"learned ×{scale:.2g}",
    )
    axes[1].set_title("Shape-only diagnostic")
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.axhline(0.0, color="#777777", linewidth=0.8)
        axis.set_xlabel("spatial pair distance")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("target-to-source attraction coefficient")
    fig.suptitle("Learned spatial interaction pattern", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_mass(mass_table: pd.DataFrame, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    summary = mass_table.groupby("time", as_index=False).agg(
        observed=("observed_relative_mass", "first"),
        predicted=("predicted_relative_mass", "mean"),
        predicted_sd=("predicted_relative_mass", "std"),
    ).fillna({"predicted_sd": 0.0})
    fig, axis = plt.subplots(figsize=(5.4, 3.8))
    axis.plot(summary["time"], summary["observed"], marker="o", linewidth=2.2, color="#555555", label="observed count ratio")
    axis.errorbar(summary["time"], summary["predicted"], yerr=summary["predicted_sd"], marker="o", linewidth=2.2, capsize=2, color="#007C83", label="learned growth")
    axis.set_xlabel("time")
    axis.set_ylabel("relative total mass")
    axis.set_title("Growth / total-mass recovery")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_dense_trajectories(
    *,
    dense_time: np.ndarray,
    ground_truth: np.ndarray,
    predicted: np.ndarray,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize

    selected = np.linspace(0, ground_truth.shape[1] - 1, 80, dtype=int)
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.3))
    norm = Normalize(vmin=float(dense_time[0]), vmax=float(dense_time[-1]))
    for axis, values, title in (
        (axes[0], ground_truth, "GT dense trajectories"),
        (axes[1], predicted, "CytoBridge dense trajectories"),
    ):
        for particle in selected:
            xy = values[:, particle, :2]
            segments = np.stack((xy[:-1], xy[1:]), axis=1)
            collection = LineCollection(segments, cmap="viridis", norm=norm, linewidths=0.55, alpha=0.55)
            collection.set_array(dense_time[:-1])
            axis.add_collection(collection)
        axis.autoscale()
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("spatial_x")
        axis.set_ylabel("spatial_y")
        axis.set_title(title)
        axis.grid(alpha=0.15)
    scalar = plt.cm.ScalarMappable(norm=norm, cmap="viridis")
    fig.suptitle("Identity-preserving fixed-population rollout", fontsize=13)
    fig.subplots_adjust(
        top=0.84, bottom=0.14, left=0.08, right=0.86, wspace=0.24
    )
    color_axis = fig.add_axes((0.89, 0.19, 0.022, 0.58))
    fig.colorbar(scalar, cax=color_axis, label="time")
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_gene_force_projection(
    learned: np.ndarray,
    target: np.ndarray,
    output_path: Path,
    *,
    deployed_title: str = "Learned gene-force map",
) -> None:
    import matplotlib.pyplot as plt

    learned_values = np.asarray(learned, dtype=float)
    target_values = np.asarray(target, dtype=float)
    limit = max(
        1.0,
        float(np.max(np.abs(learned_values))),
        float(np.max(np.abs(target_values))),
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0))
    for axis, values, title in (
        (axes[0], target_values, "GT gene-force map"),
        (axes[1], learned_values, deployed_title),
    ):
        image = axis.imshow(values, cmap="coolwarm", vmin=-limit, vmax=limit)
        for row in range(values.shape[0]):
            for column in range(values.shape[1]):
                axis.text(
                    column,
                    row,
                    f"{values[row, column]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=10,
                )
        axis.set_xticks([0, 1], ["force x", "force y"])
        axis.set_yticks([0, 1], ["gene1", "gene2"])
        axis.set_title(title)
    fig.suptitle("Spatial-force to gene-force projection")
    fig.subplots_adjust(top=0.80, bottom=0.18, left=0.09, right=0.83, wspace=0.48)
    color_axis = fig.add_axes((0.87, 0.22, 0.022, 0.54))
    fig.colorbar(image, cax=color_axis, label="linear coefficient")
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _assess_gene_force_projection(
    learned: np.ndarray,
    target: np.ndarray,
) -> dict[str, object]:
    """Assess the axis pattern separately from non-identifiable raw amplitude."""

    learned_projection = np.asarray(learned, dtype=float)
    target_projection = np.asarray(target, dtype=float)
    if learned_projection.shape != (2, 2) or target_projection.shape != (2, 2):
        raise ValueError("gene-force projection assessment requires two 2x2 matrices")
    target_norm = max(
        float(np.linalg.norm(target_projection)), np.finfo(float).eps
    )
    relative_frobenius_error = float(
        np.linalg.norm(learned_projection - target_projection) / target_norm
    )
    learned_norm_squared = float(np.sum(learned_projection**2))
    positive_shape_scale = float(
        np.sum(learned_projection * target_projection)
        / max(learned_norm_squared, np.finfo(float).eps)
    )
    shape_aligned_relative_error = float(
        np.linalg.norm(
            positive_shape_scale * learned_projection - target_projection
        )
        / target_norm
    )
    diagonal = np.diag(learned_projection)
    off_diagonal = learned_projection.copy()
    np.fill_diagonal(off_diagonal, 0.0)
    gain = float(np.mean(np.diag(target_projection)))
    mean_diagonal = float(np.mean(diagonal))
    structural_checks = {
        "positive_diagonal": bool(np.all(diagonal > 0.0)),
        "positive_shape_alignment": bool(positive_shape_scale > 0.0),
        "balanced_diagonal": bool(
            np.min(diagonal) / max(np.max(diagonal), np.finfo(float).eps)
            >= 0.40
        ),
        "limited_cross_axis_leakage": bool(
            np.max(np.abs(off_diagonal))
            <= 0.50 * max(abs(mean_diagonal), np.finfo(float).eps)
        ),
        "shape_aligned_matrix_error": bool(shape_aligned_relative_error <= 0.50),
    }
    amplitude_checks = {
        "diagonal_scale": bool(
            np.all(diagonal >= 0.25 * gain)
            and np.all(diagonal <= 1.75 * gain)
        ),
        "raw_matrix_error": bool(relative_frobenius_error <= 0.75),
    }
    return {
        "required": True,
        "target": target_projection.tolist(),
        "learned": learned_projection.tolist(),
        "relative_frobenius_error": relative_frobenius_error,
        "positive_shape_scale": positive_shape_scale,
        "shape_aligned_relative_error": shape_aligned_relative_error,
        "checks": structural_checks,
        "amplitude_diagnostics": amplitude_checks,
        "strict_amplitude_passed": bool(all(amplitude_checks.values())),
        "passed": bool(all(structural_checks.values())),
    }


def _plot_gt_interaction_signal(table: pd.DataFrame, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(5.2, 3.5))
    colors_by_space = {"spatial": "#00858A", "gene": "#CC6677", "joint": "#394B59"}
    for space in ("spatial", "gene", "joint"):
        subset = table.loc[table["space"] == space].sort_values("time")
        axis.plot(
            subset["time"],
            subset["w1"],
            marker="o",
            linewidth=2.0,
            label=space,
            color=colors_by_space[space],
        )
    axis.set_xlabel("time")
    axis.set_ylabel("GT W1: interaction vs no interaction")
    axis.set_title("Intrinsic interaction signal in the frozen GT")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _two_band_statistics(points: np.ndarray) -> dict[str, np.ndarray | float]:
    spatial = np.asarray(points, dtype=float)[:, :2]
    if spatial.ndim != 2 or spatial.shape[0] < 4 or spatial.shape[1] != 2:
        raise ValueError("two-band morphology requires at least four 2D points")
    split = float(np.median(spatial[:, 1]))
    groups = (spatial[spatial[:, 1] <= split], spatial[spatial[:, 1] > split])
    if min(group.shape[0] for group in groups) < 2:
        raise ValueError("median split produced an empty two-band group")
    centers = np.stack([group.mean(axis=0) for group in groups])
    within_std = np.stack([group.std(axis=0) for group in groups]).mean(axis=0)
    separation = float(np.linalg.norm(centers[1] - centers[0]))
    return {
        "centers": centers,
        "within_std": within_std,
        "separation": separation,
    }


def _assess_final_two_band_morphology(
    ground_truth: np.ndarray,
    predictions_by_seed: Mapping[int, np.ndarray],
) -> tuple[pd.DataFrame, dict[str, object]]:
    gt = _two_band_statistics(np.asarray(ground_truth)[-1])
    gt_std = np.maximum(
        np.asarray(gt["within_std"], dtype=float), np.finfo(float).eps
    )
    gt_centers = np.asarray(gt["centers"], dtype=float)
    gt_separation = max(float(gt["separation"]), np.finfo(float).eps)
    rows: list[dict[str, float | int]] = []
    for seed, values in predictions_by_seed.items():
        predicted = _two_band_statistics(np.asarray(values)[-1])
        predicted_std = np.asarray(predicted["within_std"], dtype=float)
        predicted_centers = np.asarray(predicted["centers"], dtype=float)
        rows.append(
            {
                "seed": int(seed),
                "gt_within_std_x": float(gt_std[0]),
                "gt_within_std_y": float(gt_std[1]),
                "predicted_within_std_x": float(predicted_std[0]),
                "predicted_within_std_y": float(predicted_std[1]),
                "within_std_ratio_x": float(predicted_std[0] / gt_std[0]),
                "within_std_ratio_y": float(predicted_std[1] / gt_std[1]),
                "gt_band_separation": float(gt_separation),
                "predicted_band_separation": float(predicted["separation"]),
                "band_separation_relative_error": float(
                    abs(float(predicted["separation"]) - gt_separation)
                    / gt_separation
                ),
                "band_centroid_rmse": float(
                    np.sqrt(np.mean((predicted_centers - gt_centers) ** 2))
                ),
            }
        )
    table = pd.DataFrame(rows)
    summary = {
        "gt_within_std_x": float(gt_std[0]),
        "gt_within_std_y": float(gt_std[1]),
        "mean_predicted_within_std_x": float(table["predicted_within_std_x"].mean()),
        "mean_predicted_within_std_y": float(table["predicted_within_std_y"].mean()),
        "mean_within_std_ratio_x": float(table["within_std_ratio_x"].mean()),
        "mean_within_std_ratio_y": float(table["within_std_ratio_y"].mean()),
        "mean_band_separation_relative_error": float(
            table["band_separation_relative_error"].mean()
        ),
        "mean_band_centroid_rmse": float(table["band_centroid_rmse"].mean()),
        "checks": {
            "within_width_x": bool(
                2.0 / 3.0 <= table["within_std_ratio_x"].mean() <= 1.50
            ),
            "within_width_y": bool(
                2.0 / 3.0 <= table["within_std_ratio_y"].mean() <= 1.50
            ),
            "band_separation": bool(
                table["band_separation_relative_error"].mean() <= 0.15
            ),
            "band_centroids": bool(table["band_centroid_rmse"].mean() <= 0.05),
        },
    }
    summary["passed"] = bool(all(summary["checks"].values()))
    return table, summary


def evaluate(args: argparse.Namespace) -> dict:
    import anndata as ad
    from scipy.stats import spearmanr

    from CytoBridge.tl.downstream.checkpoint import load_dynamical_model_from_dir
    from CytoBridge.tl.downstream.evaluation import compute_distribution_metrics
    from CytoBridge.tl.downstream.simulation import simulate_sde_from_x0

    data_dir = Path(args.data_dir).expanduser().resolve()
    direct_model_dir = getattr(args, "model_dir", None)
    if direct_model_dir is not None:
        model_dir = Path(direct_model_dir).expanduser().resolve()
    else:
        model_root = Path(args.model_root).expanduser().resolve()
        model_dir = model_root / "attractive"
    output = Path(args.evaluation_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"evaluation_dir must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    data_manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    spec_dict = data_manifest["spec"]
    spec = SpatialAttractionSpec(**{
        key: tuple(value) if key == "time_points" else value
        for key, value in spec_dict.items()
    })
    reference = np.load(data_dir / "attractive_fixed_reference.npz")
    control_reference = np.load(data_dir / "no_interaction_fixed_reference.npz")
    gt_time = np.asarray(reference["time_points"], dtype=float)
    gt_points = np.asarray(reference["snapshot_state"], dtype=np.float32)
    control_gt_points = np.asarray(
        control_reference["snapshot_state"], dtype=np.float32
    )
    gt_signal_rows: list[dict[str, float | str]] = []
    for time_index, time_value in enumerate(gt_time):
        attractive_spaces = _feature_spaces(gt_points[time_index])
        control_spaces = _feature_spaces(control_gt_points[time_index])
        for space in ("spatial", "gene", "joint"):
            metric = compute_distribution_metrics(
                attractive_spaces[space],
                control_spaces[space],
                max_ot_points=int(spec.n_particles),
                random_seed=0,
            )
            gt_signal_rows.append(
                {
                    "time": float(time_value),
                    "space": space,
                    "w1": float(metric["w1"]),
                    "w2": float(metric["w2"]),
                }
            )
    gt_signal_table = pd.DataFrame(gt_signal_rows)
    gt_signal_table.to_csv(output / "ground_truth_interaction_signal.csv", index=False)
    _plot_gt_interaction_signal(
        gt_signal_table, output / "ground_truth_interaction_signal.png"
    )
    x0 = gt_points[0]
    loaded = load_dynamical_model_from_dir(
        model_dir,
        dim=4,
        device=args.device,
        stage=args.stage,
    )
    model = loaded.model
    interaction_scale = float(getattr(args, "interaction_scale", 1.0))
    if not np.isfinite(interaction_scale) or interaction_scale < 0.0:
        raise ValueError("interaction_scale must be finite and non-negative.")
    if interaction_scale != 1.0:
        model.interaction_net.register_forward_hook(
            lambda _module, _inputs, output: output * interaction_scale
        )
    seeds = [int(token) for token in str(args.seeds).split(",") if token.strip()]
    if len(seeds) < 3 or len(set(seeds)) != len(seeds):
        raise ValueError("Evaluation requires at least three unique seeds.")

    rows: list[dict[str, object]] = []
    on_points_by_seed: dict[int, np.ndarray] = {}
    for seed in seeds:
        for include_interaction, condition in (
            (True, "interaction_on"),
            (False, "interaction_off"),
        ):
            points, _, _ = simulate_sde_from_x0(
                x0=x0,
                model=model,
                ts_points=gt_time,
                dt=float(args.eval_dt),
                sigma=float(spec.sigma),
                include_score=not args.no_score,
                include_interaction=include_interaction,
                interaction_m=int(spec.n_particles),
                device=args.device,
                noise_seed=seed,
                growth_mode=str(args.fixed_cell_dynamics_growth_mode),
                verbose=False,
            )
            if include_interaction:
                on_points_by_seed[seed] = points
            for time_index, time_value in enumerate(gt_time[1:], start=1):
                predicted_spaces = _feature_spaces(points[time_index])
                observed_spaces = _feature_spaces(gt_points[time_index])
                for space in ("joint", "spatial", "gene"):
                    metric = compute_distribution_metrics(
                        predicted_spaces[space],
                        observed_spaces[space],
                        max_ot_points=int(spec.n_particles),
                        random_seed=seed,
                    )
                    rows.append({
                        "seed": seed,
                        "condition": condition,
                        "time": float(time_value),
                        "space": space,
                        "w1": metric["w1"],
                        "w2": metric["w2"],
                    })
    metrics = pd.DataFrame(rows)
    metrics.to_csv(output / "interaction_ablation_metrics.csv", index=False)
    np.savez_compressed(
        output / "fixed_population_rollouts.npz",
        time_points=gt_time,
        ground_truth=gt_points,
        **{
            f"interaction_on_seed_{seed}": values
            for seed, values in on_points_by_seed.items()
        },
    )

    paired = metrics.pivot_table(
        index=["seed", "time", "space"], columns="condition", values="w1"
    ).reset_index()
    paired["off_minus_on"] = paired["interaction_off"] - paired["interaction_on"]
    paired.to_csv(output / "interaction_ablation_paired_deltas.csv", index=False)
    paired_summary = paired.groupby(["time", "space"], as_index=False).agg(
        n_seeds=("seed", "nunique"),
        w1_on=("interaction_on", "mean"),
        w1_off=("interaction_off", "mean"),
        off_minus_on=("off_minus_on", "mean"),
        fraction_positive=("off_minus_on", lambda values: float(np.mean(np.asarray(values) > 0))),
    )
    paired_summary.to_csv(output / "interaction_ablation_summary.csv", index=False)

    morphology_table, morphology_assessment = _assess_final_two_band_morphology(
        gt_points, on_points_by_seed
    )
    morphology_table.to_csv(output / "final_spatial_morphology_metrics.csv", index=False)
    (output / "final_spatial_morphology_assessment.json").write_text(
        json.dumps(_json_safe(morphology_assessment), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    observed = ad.read_h5ad(data_dir / "attractive_observed.h5ad")
    observed_times = np.asarray(observed.obs["time_point_processed"], dtype=float)
    counts = pd.Series(observed_times).value_counts().sort_index()
    observed_mass = counts / float(counts.iloc[0])
    mass_rows: list[dict[str, float | int]] = []
    for seed in seeds:
        _, weights, _ = simulate_sde_from_x0(
            x0=x0,
            model=model,
            ts_points=gt_time,
            dt=float(args.eval_dt),
            sigma=float(spec.sigma),
            include_score=not args.no_score,
            include_interaction=True,
            interaction_m=int(spec.n_particles),
            device=args.device,
            noise_seed=seed,
            growth_mode="learned",
            verbose=False,
        )
        predicted_mass = weights[:, :, 0].sum(axis=1)
        for time_index, time_value in enumerate(gt_time):
            observed_value = float(observed_mass.loc[float(time_value)])
            mass_rows.append({
                "seed": seed,
                "time": float(time_value),
                "observed_relative_mass": observed_value,
                "predicted_relative_mass": float(predicted_mass[time_index]),
                "absolute_tmv": abs(float(predicted_mass[time_index]) - observed_value),
            })
    mass_table = pd.DataFrame(mass_rows)
    mass_table.to_csv(output / "growth_mass_metrics.csv", index=False)

    cutoff = float(spec.interaction_cutoff)
    interior_distances = np.linspace(0.006, cutoff - 0.006, 160)
    distances = np.unique(
        np.concatenate(
            (
                np.asarray([1e-4]),
                interior_distances,
                np.asarray([cutoff - 1e-4, cutoff, cutoff + 1e-4]),
            )
        )
    )
    true_coefficient = attraction_coefficient(
        distances,
        cutoff=cutoff,
        strength=float(spec.interaction_strength),
    )
    learned_coefficient = probe_learned_spatial_attraction(
        model, distances, time=2.0, device=args.device
    )
    interior = (distances >= interior_distances[0]) & (
        distances <= interior_distances[-1]
    )
    true_interior = true_coefficient[interior]
    learned_interior = learned_coefficient[interior]
    denominator = float(np.dot(learned_interior, learned_interior))
    shape_scale = (
        float(np.dot(learned_interior, true_interior) / denominator)
        if denominator > np.finfo(float).eps else 0.0
    )
    aligned = learned_coefficient * shape_scale
    true_rms = max(
        float(np.sqrt(np.mean(true_interior**2))), np.finfo(float).eps
    )
    sign_accuracy = float(np.mean(learned_interior > 0.0))
    spearman = float(spearmanr(true_interior, learned_interior).statistic)
    pearson = float(np.corrcoef(true_interior, learned_interior)[0, 1])
    shape_nrmse = float(
        np.sqrt(np.mean((aligned[interior] - true_interior) ** 2)) / true_rms
    )
    raw_nrmse = float(
        np.sqrt(np.mean((learned_interior - true_interior) ** 2)) / true_rms
    )
    amplitude_ratio = float(
        np.sqrt(np.mean(learned_interior**2)) / true_rms
    )
    peak_error = float(
        abs(
            interior_distances[int(np.argmax(learned_interior))]
            - interior_distances[int(np.argmax(true_interior))]
        )
    )
    pattern = pd.DataFrame({
        "distance": distances,
        "true_coefficient": true_coefficient,
        "learned_coefficient": learned_coefficient,
        "shape_aligned_learned": aligned,
        "is_interior_assessment_point": interior,
    })
    pattern.to_csv(output / "interaction_radial_curve.csv", index=False)
    pattern_assessment = {
        "sign_accuracy": sign_accuracy,
        "spearman": spearman,
        "pearson": pearson,
        "positive_shape_scale": shape_scale,
        "raw_nrmse": raw_nrmse,
        "raw_rms_amplitude_ratio": amplitude_ratio,
        "shape_aligned_nrmse": shape_nrmse,
        "peak_distance_error": peak_error,
        "learned_near_origin": float(learned_coefficient[0]),
        "learned_cutoff_left": float(
            learned_coefficient[np.flatnonzero(distances < cutoff)[-1]]
        ),
        "learned_at_cutoff": float(
            learned_coefficient[np.flatnonzero(distances == cutoff)[0]]
        ),
        "learned_outside_cutoff": float(learned_coefficient[-1]),
        "checks": {
            "predominantly_attractive": sign_accuracy >= 0.80,
            "positive_alignment": shape_scale > 0.0,
            "shape_correlation": pearson >= 0.50,
            "shape_error": shape_nrmse <= 0.75,
            "raw_magnitude_error": raw_nrmse <= 0.65,
            "raw_amplitude": 0.50 <= amplitude_ratio <= 1.50,
            "peak_location": peak_error <= 0.08,
            "hard_cutoff": bool(
                abs(float(learned_coefficient[-2])) <= 1e-8
                and abs(float(learned_coefficient[-1])) <= 1e-8
            ),
        },
    }
    structural_pattern_keys = (
        "predominantly_attractive",
        "positive_alignment",
        "shape_correlation",
        "shape_error",
        "peak_location",
        "hard_cutoff",
    )
    pattern_assessment["shape_passed"] = bool(
        all(pattern_assessment["checks"][key] for key in structural_pattern_keys)
    )
    pattern_assessment["strict_amplitude_passed"] = bool(
        pattern_assessment["checks"]["raw_magnitude_error"]
        and pattern_assessment["checks"]["raw_amplitude"]
    )
    pattern_assessment["passed"] = bool(all(pattern_assessment["checks"].values()))
    (output / "interaction_pattern_assessment.json").write_text(
        json.dumps(_json_safe(pattern_assessment), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    gene_projection_required = str(spec.version) in {V7_VERSION, V8_VERSION}
    gene_projection_assessment: dict[str, object] = {
        "required": gene_projection_required,
        "passed": True,
    }
    if gene_projection_required:
        interaction_network = getattr(model, "interaction_net", None)
        projection_mode = str(
            getattr(
                interaction_network,
                "radial_gene_projection_mode",
                "learned",
            )
        )
        projection_parameter_count = int(
            sum(
                parameter.numel()
                for name, parameter in interaction_network.named_parameters()
                if "gene_force_projection" in name
            )
        )
        learned_projection = probe_learned_gene_force_projection(
            model,
            distance=0.5 * float(spec.interaction_cutoff),
            time=2.0,
            device=args.device,
        )
        target_projection = float(spec.gene_interaction_gain) * np.eye(2)
        gene_projection_assessment = _assess_gene_force_projection(
            learned_projection, target_projection
        )
        gene_projection_assessment.update(
            {
                "map_mode": projection_mode,
                "trainable_projection_parameter_count": projection_parameter_count,
                "deployed": learned_projection.tolist(),
            }
        )
        pd.DataFrame(
            [
                {
                    "output_gene": f"gene{row + 1}",
                    "input_spatial_force": f"force_{'xy'[column]}",
                    "target": float(target_projection[row, column]),
                    "learned": float(learned_projection[row, column]),
                    "deployed": float(learned_projection[row, column]),
                    "map_mode": projection_mode,
                }
                for row in range(2)
                for column in range(2)
            ]
        ).to_csv(output / "gene_force_projection.csv", index=False)
        _plot_gene_force_projection(
            learned_projection,
            target_projection,
            output / "gene_force_projection.png",
            deployed_title=(
                "Fixed identity gene-force map"
                if projection_mode == "identity"
                else (
                    "Fixed GT gene-force map"
                    if projection_mode == "fixed"
                    else "Learned gene-force map"
                )
            ),
        )
    (output / "gene_force_projection_assessment.json").write_text(
        json.dumps(_json_safe(gene_projection_assessment), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    _plot_snapshot_grid(
        time_points=gt_time,
        ground_truth=gt_points,
        predicted=on_points_by_seed[seeds[0]],
        columns=slice(0, 2),
        axis_names=("spatial_x", "spatial_y"),
        title="Fixed-population spatial snapshots",
        output_path=output / "gt_vs_generated_spatial.png",
    )
    _plot_snapshot_grid(
        time_points=gt_time,
        ground_truth=gt_points,
        predicted=on_points_by_seed[seeds[0]],
        columns=slice(2, 4),
        axis_names=("gene1", "gene2"),
        title="Fixed-population 2D gene-state snapshots",
        output_path=output / "gt_vs_generated_gene.png",
    )
    _plot_w1(metrics, output / "interaction_on_vs_off_w1.png")
    _plot_gene_ablation_delta(
        paired,
        assessment_start_time=float(gt_time[0])
        + 0.5 * float(gt_time[-1] - gt_time[0]),
        output_path=output / "gene_interaction_ablation_delta.png",
    )
    _plot_pattern(pattern, shape_scale, output / "interaction_pattern.png")
    _plot_mass(mass_table, output / "growth_mass_curve.png")

    dense_time = np.asarray(reference["dense_time"], dtype=float)
    gt_dense = np.asarray(reference["dense_state"], dtype=np.float32)
    predicted_dense, _, _ = simulate_sde_from_x0(
        x0=x0,
        model=model,
        ts_points=dense_time,
        dt=float(args.eval_dt),
        sigma=float(spec.sigma),
        include_score=not args.no_score,
        include_interaction=True,
        interaction_m=int(spec.n_particles),
        device=args.device,
        noise_seed=seeds[0],
        growth_mode=str(args.fixed_cell_dynamics_growth_mode),
        verbose=False,
    )
    _plot_dense_trajectories(
        dense_time=dense_time,
        ground_truth=gt_dense,
        predicted=predicted_dense,
        output_path=output / "dense_spatial_trajectories.png",
    )
    np.savez_compressed(
        output / f"dense_rollout_seed_{seeds[0]}.npz",
        time_points=dense_time,
        ground_truth=gt_dense,
        predicted=predicted_dense,
        seed=int(seeds[0]),
    )

    required_summary = paired_summary.loc[
        paired_summary["space"].isin(["joint", "spatial"])
    ]
    ablation_passed = bool(
        (required_summary["off_minus_on"] > 0.0).all()
        and (required_summary["fraction_positive"] >= 0.8).all()
    )
    gene_ablation_required = str(spec.version) in {V7_VERSION, V8_VERSION}
    gene_ablation_start_time = float(gt_time[0]) + 0.5 * float(
        gt_time[-1] - gt_time[0]
    )
    gene_ablation = _assess_gene_ablation(
        paired_summary,
        assessment_start_time=gene_ablation_start_time,
        required=gene_ablation_required,
    )
    gene_ablation_passed = bool(gene_ablation["passed"])
    strict_gene_effect_size_passed = bool(
        gene_ablation["strict_effect_size_passed"]
    )
    mass_noninitial = mass_table.loc[mass_table["time"] > float(gt_time[0])]
    mass_mae = float(mass_noninitial["absolute_tmv"].mean())
    mass_max = float(mass_noninitial["absolute_tmv"].max())
    growth_passed = bool(mass_mae <= 0.10 and mass_max <= 0.20)
    on_summary = (
        metrics.loc[metrics["condition"] == "interaction_on"]
        .groupby(["time", "space"], as_index=False)
        .agg(mean_w1=("w1", "mean"), sd_w1=("w1", "std"))
    )
    # Coordinates are normalized benchmark units.  These fixed limits require
    # every evaluated time point to remain close to GT; a large late-rollout
    # error can no longer pass merely because interaction-off is worse.
    absolute_limits = {"joint": 0.10, "spatial": 0.08, "gene": 0.05}
    absolute_distribution: dict[str, dict[str, object]] = {}
    for space, limit in absolute_limits.items():
        values = on_summary.loc[on_summary["space"] == space, "mean_w1"]
        maximum = float(values.max())
        absolute_distribution[space] = {
            "maximum_timepoint_mean_w1": maximum,
            "threshold": float(limit),
            "passed": bool(maximum <= limit),
        }
    absolute_passed = bool(
        all(item["passed"] for item in absolute_distribution.values())
    )
    accepted_pattern = bool(
        pattern_assessment["shape_passed"]
        if str(spec.version) in {V6_VERSION, V7_VERSION, V8_VERSION}
        else pattern_assessment["passed"]
    )
    morphology_required = str(spec.version) in {
        V5_VERSION, V6_VERSION, V7_VERSION, V8_VERSION
    }
    morphology_passed = bool(
        morphology_assessment["passed"] if morphology_required else True
    )
    gene_projection_passed = bool(gene_projection_assessment["passed"])
    acceptance = {
        "schema_version": "cytobridge_spatial_attraction_acceptance/6",
        "overall_passed": bool(
            ablation_passed
            and gene_ablation_passed
            and gene_projection_passed
            and absolute_passed
            and accepted_pattern
            and growth_passed
            and morphology_passed
        ),
        "checks": {
            "interaction_on_beats_off_joint_and_spatial": ablation_passed,
            "interaction_on_beats_off_gene": gene_ablation_passed,
            "gene_interaction_strict_effect_size": strict_gene_effect_size_passed,
            "gene_force_projection": gene_projection_passed,
            "absolute_distribution_accuracy": absolute_passed,
            "interaction_pattern": accepted_pattern,
            "interaction_pattern_raw_amplitude": pattern_assessment[
                "strict_amplitude_passed"
            ],
            "final_spatial_morphology": morphology_passed,
            "growth_total_mass": growth_passed,
            "noise_enabled": float(spec.sigma) > 0.0,
        },
        "absolute_distribution": absolute_distribution,
        "gene_interaction_ablation": gene_ablation,
        "gene_force_projection": gene_projection_assessment,
        "growth": {"mean_absolute_tmv": mass_mae, "max_absolute_tmv": mass_max},
        "final_spatial_morphology": morphology_assessment,
        "interaction_pattern": pattern_assessment,
        "protocol": {
            "state_order": ["spatial_x", "spatial_y", "gene1", "gene2"],
            "inference_seeds": seeds,
            "sigma": float(spec.sigma),
            "dt": float(args.eval_dt),
            "same_checkpoint_on_off": True,
            "score_included": not args.no_score,
            "fixed_cell_dynamics_growth_mode": str(
                args.fixed_cell_dynamics_growth_mode
            ),
            "distribution_metric_particle_weights": "uniform",
            "growing_mass_growth_mode": "learned",
            "interaction_scale": interaction_scale,
            "final_spatial_morphology_required": morphology_required,
            "gene_interaction_ablation_required": gene_ablation_required,
            "gene_force_projection_required": gene_projection_required,
        },
        "checkpoint": {
            "weight_stage": loaded.weight_stage,
            "weight_path": str(loaded.weight_path),
            "weight_sha256": _sha256(loaded.weight_path),
            "score_stage": loaded.score_stage,
            "score_path": str(loaded.score_path) if loaded.score_path else None,
            "score_sha256": _sha256(loaded.score_path) if loaded.score_path else None,
        },
    }
    acceptance_path = output / "acceptance.json"
    acceptance_path.write_text(
        json.dumps(_json_safe(acceptance), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return acceptance


def _add_shared_data_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-dir", required=True, type=Path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser("generate", help="Generate benchmark data")
    _add_shared_data_arguments(generate_parser)
    generate_parser.add_argument(
        "--n-particles",
        type=int,
        default=None,
        help="Number of persistent particles (default: 400 for v5-v8, 256 for historical versions).",
    )
    generate_parser.add_argument(
        "--version", choices=SUPPORTED_VERSIONS, default=SUPPORTED_VERSIONS[0]
    )
    generate_parser.add_argument("--dt", type=float, default=0.02)
    generate_parser.add_argument("--cutoff", type=float, default=0.30)
    generate_parser.add_argument("--interaction-strength", type=float, default=None)
    generate_parser.add_argument("--sigma", type=float, default=0.015)
    generate_parser.add_argument("--measurement-spatial-sigma", type=float, default=None)
    generate_parser.add_argument("--measurement-gene-sigma", type=float, default=None)
    generate_parser.add_argument("--target-final-mass-ratio", type=float, default=1.25)
    generate_parser.add_argument("--gene-interaction-gain", type=float, default=None)

    train_parser = subparsers.add_parser("train", help="Train background then attraction")
    _add_shared_data_arguments(train_parser)
    train_parser.add_argument("--model-root", required=True, type=Path)
    train_parser.add_argument("--device", default="cuda")
    train_parser.add_argument("--background-config", type=Path, default=DEFAULT_BACKGROUND_CONFIG)
    train_parser.add_argument("--target-config", type=Path, default=DEFAULT_TARGET_CONFIG)
    train_parser.add_argument("--background-epochs", type=int, default=None)
    train_parser.add_argument("--interaction-epochs", type=int, default=None)
    train_parser.add_argument("--score-epochs", type=int, default=None)

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate fitted attraction model")
    _add_shared_data_arguments(eval_parser)
    eval_model_group = eval_parser.add_mutually_exclusive_group(required=True)
    eval_model_group.add_argument(
        "--model-root",
        type=Path,
        help="Training root containing the attraction model in attractive/.",
    )
    eval_model_group.add_argument(
        "--model-dir",
        type=Path,
        help="Direct path to a fitted attraction model directory.",
    )
    eval_parser.add_argument("--evaluation-dir", required=True, type=Path)
    eval_parser.add_argument("--device", default="cuda")
    eval_parser.add_argument("--stage", default="Calibrate_interaction")
    eval_parser.add_argument("--seeds", default="1,4,8,32,256")
    eval_parser.add_argument("--eval-dt", type=float, default=0.02)
    eval_parser.add_argument(
        "--interaction-scale",
        type=float,
        default=1.0,
        help=(
            "Diagnostic global multiplier applied to the learned interaction "
            "output during rollout; 1.0 is the unmodified checkpoint."
        ),
    )
    eval_parser.add_argument("--no-score", action="store_true")
    eval_parser.add_argument(
        "--fixed-cell-dynamics-growth-mode",
        choices=("frozen_uniform", "learned"),
        default="frozen_uniform",
        help=(
            "Internal particle-mass dynamics used by the fixed-cell coordinate "
            "rollout; distribution metrics remain uniformly weighted."
        ),
    )

    all_parser = subparsers.add_parser("all", help="Generate, train and evaluate")
    _add_shared_data_arguments(all_parser)
    all_parser.add_argument("--model-root", required=True, type=Path)
    all_parser.add_argument("--evaluation-dir", required=True, type=Path)
    all_parser.add_argument("--device", default="cuda")
    all_parser.add_argument("--background-config", type=Path, default=DEFAULT_BACKGROUND_CONFIG)
    all_parser.add_argument("--target-config", type=Path, default=DEFAULT_TARGET_CONFIG)
    all_parser.add_argument("--background-epochs", type=int, default=None)
    all_parser.add_argument("--interaction-epochs", type=int, default=None)
    all_parser.add_argument("--score-epochs", type=int, default=None)
    all_parser.add_argument(
        "--n-particles",
        type=int,
        default=None,
        help="Number of persistent particles (default: 400 for v5-v8, 256 for historical versions).",
    )
    all_parser.add_argument(
        "--version", choices=SUPPORTED_VERSIONS, default=SUPPORTED_VERSIONS[0]
    )
    all_parser.add_argument("--dt", type=float, default=0.02)
    all_parser.add_argument("--cutoff", type=float, default=0.30)
    all_parser.add_argument("--interaction-strength", type=float, default=None)
    all_parser.add_argument("--sigma", type=float, default=0.015)
    all_parser.add_argument("--measurement-spatial-sigma", type=float, default=None)
    all_parser.add_argument("--measurement-gene-sigma", type=float, default=None)
    all_parser.add_argument("--target-final-mass-ratio", type=float, default=1.25)
    all_parser.add_argument("--gene-interaction-gain", type=float, default=None)
    all_parser.add_argument("--stage", default="Calibrate_interaction")
    all_parser.add_argument("--seeds", default="1,4,8,32,256")
    all_parser.add_argument("--eval-dt", type=float, default=0.02)
    all_parser.add_argument(
        "--interaction-scale",
        type=float,
        default=1.0,
        help=(
            "Diagnostic global multiplier applied to the learned interaction "
            "output during rollout; 1.0 is the unmodified checkpoint."
        ),
    )
    all_parser.add_argument("--no-score", action="store_true")
    all_parser.add_argument(
        "--fixed-cell-dynamics-growth-mode",
        choices=("frozen_uniform", "learned"),
        default="frozen_uniform",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "generate":
        result = generate(args)
    elif args.command == "train":
        result = train(args)
    elif args.command == "evaluate":
        result = evaluate(args)
    elif args.command == "all":
        generate(args)
        train(args)
        result = evaluate(args)
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(json.dumps(_json_safe(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
