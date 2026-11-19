#!/usr/bin/env python3
"""Run and report fixed-checkpoint interaction on/off sensitivity.

This entry point evaluates the five accepted full models without retraining.
For each dataset, both branches begin from the exact same earliest observed
cells and use the same stochastic seed.  The only intervention is replacing
the learned interaction force with the package zero-force adapter during one
continuous split-SDE propagation.  Growth-dependent resampling is disabled so
cell identity and row order remain paired across branches.

The result is a model-sensitivity analysis.  It is not a matched retraining
ablation, a causal knockout, or an uncertainty estimate.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import CytoBridge as cb  # noqa: E402
import CytoBridge.workflow as workflow  # noqa: E402


SCHEMA_VERSION = 1
DATASETS = ("zebrafish", "mosta", "arista", "admouse", "chicken_heart")
DISPLAY_NAMES = {
    "zebrafish": "Zebrafish",
    "mosta": "MOSTA",
    "arista": "ARISTA",
    "admouse": "AdMouse",
    "chicken_heart": "Chicken heart",
}
RANDOM_SEED = 42
INTERACTION_M = 1024
SPATIAL_DIM = 2
MAX_OT_POINTS = 1024


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "path": str(resolved),
        "sha256": _sha256(resolved),
        "size": int(resolved.stat().st_size),
    }


def _require_sha256(path: Path, expected: str, *, label: str) -> None:
    expected = str(expected).strip().lower()
    if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
        raise ValueError(f"{label} expected SHA-256 is invalid: {expected!r}.")
    observed = _sha256(path)
    if observed != expected:
        raise ValueError(
            f"{label} SHA-256 mismatch: expected {expected}, observed {observed}."
        )


def _require_empty(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.exists() and any(resolved.iterdir()):
        raise FileExistsError(f"Output directory must be new or empty: {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _string_sequence_sha256(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def _time_grid(config: Mapping[str, Any]) -> tuple[float, ...]:
    downstream = config.get("downstream", {})
    observed = [float(value) for value in downstream.get("observed", [])]
    interpolated = [float(value) for value in downstream.get("interpolated", [])]
    times = tuple(sorted(set(observed + interpolated)))
    if len(times) < 2 or any(b <= a for a, b in zip(times[:-1], times[1:])):
        raise ValueError(
            "Workflow config must define at least two ordered time points."
        )
    return times


def paired_displacement_metrics(
    interaction_on: Sequence[np.ndarray],
    interaction_off: Sequence[np.ndarray],
    time_points: Sequence[float],
    *,
    spatial_dim: int = SPATIAL_DIM,
) -> pd.DataFrame:
    """Compute row-paired displacements for fixed-population trajectories."""

    rows: list[dict[str, Any]] = []
    spaces = {
        "joint": slice(None),
        "spatial": slice(0, int(spatial_dim)),
        "expression": slice(int(spatial_dim), None),
    }
    if not (len(interaction_on) == len(interaction_off) == len(time_points)):
        raise ValueError("Both trajectories must match the declared time grid.")
    for time_index, (time_value, on_frame, off_frame) in enumerate(
        zip(time_points, interaction_on, interaction_off)
    ):
        on = np.asarray(on_frame, dtype=np.float64)
        off = np.asarray(off_frame, dtype=np.float64)
        if on.shape != off.shape or on.ndim != 2:
            raise ValueError(
                f"Paired trajectory shape mismatch at t={time_value}: "
                f"on={on.shape}, off={off.shape}."
            )
        if on.shape[1] <= int(spatial_dim):
            raise ValueError("Trajectory has no expression-state dimensions.")
        for space, columns in spaces.items():
            distances = np.linalg.norm(off[:, columns] - on[:, columns], axis=1)
            rows.append(
                {
                    "time_index": int(time_index),
                    "time": float(time_value),
                    "space": space,
                    "n_paired": int(len(distances)),
                    "mean_displacement": float(np.mean(distances)),
                    "rms_displacement": float(np.sqrt(np.mean(distances**2))),
                    "median_displacement": float(np.median(distances)),
                    "p95_displacement": float(np.quantile(distances, 0.95)),
                    "max_displacement": float(np.max(distances)),
                }
            )
    return pd.DataFrame(rows)


def coupled_distribution_metrics(
    interaction_on: Sequence[np.ndarray],
    interaction_off: Sequence[np.ndarray],
    time_points: Sequence[float],
    *,
    spatial_dim: int = SPATIAL_DIM,
    max_ot_points: int = MAX_OT_POINTS,
    random_seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Compute exact OT on one shared deterministic row subset per pair.

    The generic evaluator independently subsamples its two input clouds. That
    is appropriate for unrelated empirical samples, but it creates a positive
    sampling floor here even at t0, where the paired on/off rows are identical.
    This fixed-population experiment instead retains the same row indices from
    both branches before solving exact OT.
    """

    if not (len(interaction_on) == len(interaction_off) == len(time_points)):
        raise ValueError("Both trajectories must match the declared time grid.")
    spaces = {
        "joint": slice(None),
        "spatial": slice(0, int(spatial_dim)),
        "latent": slice(int(spatial_dim), None),
    }
    rows: list[dict[str, Any]] = []
    for time_index, (time_value, on_frame, off_frame) in enumerate(
        zip(time_points, interaction_on, interaction_off)
    ):
        on = np.asarray(on_frame, dtype=np.float64)
        off = np.asarray(off_frame, dtype=np.float64)
        if on.shape != off.shape or on.ndim != 2:
            raise ValueError(
                f"Paired trajectory shape mismatch at t={time_value}: "
                f"on={on.shape}, off={off.shape}."
            )
        n_cells = int(on.shape[0])
        for space_index, (space, columns) in enumerate(spaces.items()):
            on_space = on[:, columns]
            off_space = off[:, columns]
            seed = int(random_seed) + 100 * int(time_index) + int(space_index)
            if n_cells > int(max_ot_points):
                indices = np.sort(
                    np.random.default_rng(seed).choice(
                        n_cells, size=int(max_ot_points), replace=False
                    )
                )
            else:
                indices = np.arange(n_cells, dtype=int)
            retained_on = on_space[indices]
            retained_off = off_space[indices]
            distances = cb.tl.compute_distribution_metrics(
                retained_off,
                retained_on,
                max_ot_points=None,
                random_seed=seed,
            )
            centroid_shift = float(
                np.linalg.norm(np.mean(off_space, axis=0) - np.mean(on_space, axis=0))
            )
            on_centered = on_space - np.mean(on_space, axis=0, keepdims=True)
            off_centered = off_space - np.mean(off_space, axis=0, keepdims=True)
            on_radius = float(
                np.sqrt(np.mean(np.sum(on_centered * on_centered, axis=1)))
            )
            off_radius = float(
                np.sqrt(np.mean(np.sum(off_centered * off_centered, axis=1)))
            )
            rows.append(
                {
                    "variant": "interaction_off",
                    "time_index": int(time_index),
                    "time": float(time_value),
                    "space": space,
                    "n_baseline": n_cells,
                    "n_ablation": n_cells,
                    "count_delta": 0,
                    "count_ratio": 1.0,
                    "w1": float(distances["w1"]),
                    "w2": float(distances["w2"]),
                    "ot_ablation_points": int(len(indices)),
                    "ot_baseline_points": int(len(indices)),
                    "ot_random_seed": seed,
                    "ot_sampling": "shared_paired_row_indices_without_replacement",
                    "centroid_shift": centroid_shift,
                    "baseline_rms_radius": on_radius,
                    "ablation_rms_radius": off_radius,
                    "rms_radius_delta": float(off_radius - on_radius),
                }
            )
    return pd.DataFrame(rows)


def _workflow_config_artifact(config_name: str, config_source: str) -> dict[str, Any]:
    if str(config_source).startswith("packaged preset:"):
        path = (
            Path(cb.__file__).resolve().parent
            / "workflow_configs"
            / f"{config_name}.json"
        )
    else:
        path = Path(config_source)
    return _artifact(path)


def run_dataset(args: argparse.Namespace) -> Path:
    import anndata as ad

    dataset_name = str(args.dataset)
    if dataset_name not in DATASETS:
        raise ValueError(f"Unsupported dataset {dataset_name!r}; expected {DATASETS}.")
    aligned_h5ad = args.aligned_h5ad.expanduser().resolve()
    training_summary = (
        args.model_dir.expanduser().resolve() / "training_run_summary.json"
    )
    if not aligned_h5ad.is_file():
        raise FileNotFoundError(aligned_h5ad)
    if not training_summary.is_file():
        raise FileNotFoundError(training_summary)
    _require_sha256(aligned_h5ad, args.expected_aligned_sha256, label="aligned H5AD")
    _require_sha256(
        training_summary,
        args.expected_training_summary_sha256,
        label="training run summary",
    )
    output_dir = _require_empty(args.output_dir)

    config, config_source = workflow.load_workflow_config(dataset_name)
    dataset = config["dataset"]
    if str(dataset.get("name")) != dataset_name:
        raise RuntimeError("Workflow preset dataset identity mismatch.")
    time_points = _time_grid(config)
    downstream = config["downstream"]
    dt = float(downstream.get("split_sde_dt", 0.01))
    sigma = float(downstream.get("split_sigma", 0.03))

    adata = ad.read_h5ad(aligned_h5ad)
    annotation_key = str(dataset.get("annotation_key", "Annotation"))
    dataframe, resolved_time_key = cb.tl.adata_to_aligned_dataframe(
        adata,
        time_key=dataset.get("time_key"),
        obsm_key=str(dataset.get("obsm_key", "X_latent")),
        spatial_key=str(dataset.get("spatial_key", "spatial_aligned")),
        concat_spatial=dataset.get("concat_spatial", True),
        annotation_key=annotation_key,
    )
    feature_columns = cb.tl.infer_feature_columns(
        dataframe, annotation_column=annotation_key
    )
    if len(feature_columns) != 52:
        raise RuntimeError(
            f"Formal interaction sensitivity requires 2 spatial + 50 expression "
            f"dimensions; observed {len(feature_columns)}."
        )
    loaded = cb.tl.load_dynamical_model_from_dir(
        args.model_dir.expanduser().resolve(),
        dim=len(feature_columns),
        device=args.device,
    )
    components = {
        str(component).strip().lower()
        for component in getattr(loaded.model, "components", [])
    }
    if (
        "interaction" not in components
        or getattr(loaded.model, "interaction_net", None) is None
    ):
        raise RuntimeError(
            "The accepted full model has no learned interaction component."
        )

    times = dataframe["samples"].to_numpy(dtype=float)
    source_time = float(np.min(times))
    source_mask = np.isclose(times, source_time, rtol=0.0, atol=1e-9)
    x0 = dataframe.loc[source_mask, feature_columns].to_numpy(dtype=np.float32)
    source_obs_names = tuple(np.asarray(adata.obs_names.astype(str))[source_mask])
    if x0.shape[0] < 2:
        raise RuntimeError("Earliest observed slice contains fewer than two cells.")

    result = cb.tl.run_virtual_interaction_ablation(
        x0,
        loaded.model,
        time_points=time_points,
        output_dir=output_dir / "ablation",
        variant_name="interaction_off",
        dt=dt,
        resample_dt=dt,
        sigma=sigma,
        growth_alpha=0.0,
        interaction_m=INTERACTION_M,
        max_particles=max(100_000, int(x0.shape[0])),
        spatial_dim=SPATIAL_DIM,
        device=args.device,
        random_seed=RANDOM_SEED,
        save_data=True,
        save_snapshots=False,
        verbose=True,
    )
    counts_on = [int(np.asarray(frame).shape[0]) for frame in result.baseline_points]
    counts_off = [int(np.asarray(frame).shape[0]) for frame in result.ablated_points]
    expected_count = int(x0.shape[0])
    if counts_on != [expected_count] * len(time_points) or counts_off != [
        expected_count
    ] * len(time_points):
        raise RuntimeError(
            "Fixed-population interaction sensitivity changed particle counts: "
            f"on={counts_on}, off={counts_off}."
        )

    paired = paired_displacement_metrics(
        result.baseline_points,
        result.ablated_points,
        time_points,
        spatial_dim=SPATIAL_DIM,
    )
    paired_path = output_dir / "paired_displacement_metrics.csv"
    paired.to_csv(paired_path, index=False)
    distribution_path = output_dir / "ablation" / "interaction_ablation_metrics.csv"
    distribution = coupled_distribution_metrics(
        result.baseline_points,
        result.ablated_points,
        time_points,
        spatial_dim=SPATIAL_DIM,
        max_ot_points=MAX_OT_POINTS,
        random_seed=RANDOM_SEED,
    )
    distribution.to_csv(distribution_path, index=False)
    if set(distribution["space"]) != {"joint", "spatial", "latent"}:
        raise RuntimeError("Interaction-ablation distribution spaces are incomplete.")

    output_paths = [
        output_dir / "ablation" / "manifest.json",
        distribution_path,
        output_dir / "ablation" / "trajectories" / "initial_x0.npy",
        output_dir / "ablation" / "trajectories" / "interaction_on_points.npy",
        output_dir / "ablation" / "trajectories" / "interaction_off_points.npy",
        paired_path,
    ]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "created_at": _utc_now(),
        "dataset": dataset_name,
        "display_name": DISPLAY_NAMES[dataset_name],
        "claim_scope": (
            "fixed-checkpoint interaction-force sensitivity; no retraining and no "
            "causal interpretation"
        ),
        "input_h5ad": _artifact(aligned_h5ad),
        "workflow_config": _workflow_config_artifact(dataset_name, config_source),
        "training_run_summary": _artifact(training_summary),
        "checkpoint": {
            "weight_stage": str(loaded.weight_stage),
            "weight": _artifact(loaded.weight_path),
            "score_stage": loaded.score_stage,
            "score": None
            if loaded.score_path is None
            else _artifact(loaded.score_path),
        },
        "source_population": {
            "resolved_time_key": str(resolved_time_key),
            "source_time": source_time,
            "n_cells": expected_count,
            "obs_names_sha256": _string_sequence_sha256(source_obs_names),
            "state_sha256": _array_sha256(x0),
            "state_shape": list(x0.shape),
            "sampling": "all earliest-observed cells in aligned H5AD order",
        },
        "protocol": {
            "interaction_on_scale": 1.0,
            "interaction_off_scale": 0.0,
            "same_trained_checkpoint": True,
            "training_performed": False,
            "same_initial_cells": True,
            "same_branch_seed": True,
            "random_seed": RANDOM_SEED,
            "continuous_from_source": True,
            "observed_slice_reanchoring": False,
            "spatial_warp": False,
            "growth_alpha": 0.0,
            "fixed_population": True,
            "sigma": sigma,
            "dt": dt,
            "time_points": list(time_points),
            "interaction_group_size": INTERACTION_M,
            "spatial_dimensions": SPATIAL_DIM,
            "expression_dimensions": 50,
            "distribution_ot_max_points": MAX_OT_POINTS,
            "distribution_ot_sampling": (
                "exact OT on shared deterministic paired-row support"
            ),
            "interpretation": (
                "single-seed model sensitivity; not a matched retraining ablation, "
                "causal knockout, or uncertainty estimate"
            ),
        },
        "package": {
            "version": str(getattr(cb, "__version__", "unknown")),
            "runner": _artifact(Path(__file__)),
            "ablation_module": _artifact(
                Path(
                    sys.modules[
                        cb.tl.run_virtual_interaction_ablation.__module__
                    ].__file__
                )
            ),
        },
        "outputs": [_artifact(path) for path in output_paths],
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (output_dir / "manifest.json.sha256").write_text(
        f"{_sha256(manifest_path)}  {manifest_path.name}\n", encoding="utf-8"
    )
    print(manifest_path)
    return manifest_path


def _load_trajectory(path: Path) -> tuple[np.ndarray, ...]:
    values = np.load(path, allow_pickle=True)
    return tuple(np.asarray(frame, dtype=np.float32) for frame in values)


def _plot_indices(n: int, *, dataset_index: int, maximum: int = 12_000) -> np.ndarray:
    if n <= maximum:
        return np.arange(n, dtype=int)
    rng = np.random.default_rng(RANDOM_SEED + 1000 * int(dataset_index))
    return np.sort(rng.choice(n, size=maximum, replace=False))


def _figure_rc() -> dict[str, Any]:
    return {
        "font.family": "Arial",
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }


def _plot_endpoint_snapshots(
    records: Mapping[str, Mapping[str, Any]], output_dir: Path
) -> tuple[Path, Path]:
    pdf_path = output_dir / "interaction_on_off_endpoint_snapshots.pdf"
    png_path = output_dir / "interaction_on_off_endpoint_snapshots.png"
    with mpl.rc_context(_figure_rc()):
        fig, axes = plt.subplots(5, 2, figsize=(8.27, 11.69), squeeze=False)
        for row, dataset in enumerate(DATASETS):
            record = records[dataset]
            on = record["on"][-1]
            off = record["off"][-1]
            indices = _plot_indices(len(on), dataset_index=row)
            combined = np.vstack((on[:, :2], off[:, :2]))
            low = np.nanmin(combined, axis=0)
            high = np.nanmax(combined, axis=0)
            pad = np.maximum((high - low) * 0.04, 1e-6)
            for column, (points, condition, color) in enumerate(
                (
                    (on, "With interaction", "#07838B"),
                    (off, "Without interaction", "#CC6677"),
                )
            ):
                ax = axes[row, column]
                ax.scatter(
                    points[indices, 0],
                    points[indices, 1],
                    s=2.0,
                    c=color,
                    alpha=0.55,
                    linewidths=0,
                    rasterized=len(indices) > 6000,
                )
                ax.set_xlim(low[0] - pad[0], high[0] + pad[0])
                ax.set_ylim(low[1] - pad[1], high[1] + pad[1])
                ax.set_aspect("equal")
                ax.set_xticks([])
                ax.set_yticks([])
                ax.spines[:].set_visible(False)
                if row == 0:
                    ax.set_title(condition, fontweight="bold", pad=4)
                if column == 0:
                    ax.set_ylabel(
                        DISPLAY_NAMES[dataset],
                        rotation=90,
                        fontweight="bold",
                        labelpad=8,
                    )
                if column == 1:
                    ax.text(
                        0.98,
                        0.02,
                        f"t = {record['times'][-1]:g}\nn = {len(points):,}",
                        ha="right",
                        va="bottom",
                        transform=ax.transAxes,
                        fontsize=8,
                        color="#24313A",
                    )
        fig.text(0.035, 0.982, "a", fontsize=14, fontweight="bold", va="top")
        fig.text(
            0.075,
            0.982,
            "Endpoint spatial distributions",
            fontsize=12,
            fontweight="bold",
            va="top",
        )
        fig.subplots_adjust(
            left=0.12, right=0.98, top=0.945, bottom=0.035, hspace=0.13, wspace=0.08
        )
        fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.08)
        fig.savefig(png_path, dpi=320, bbox_inches="tight", pad_inches=0.08)
        plt.close(fig)
    return pdf_path, png_path


def _plot_effect_metrics(
    final_table: pd.DataFrame, output_dir: Path
) -> tuple[Path, Path]:
    pdf_path = output_dir / "interaction_on_off_effect_metrics.pdf"
    png_path = output_dir / "interaction_on_off_effect_metrics.png"
    order = list(DATASETS)
    labels = [DISPLAY_NAMES[value] for value in order]
    x = np.arange(len(order))
    panels = (
        ("spatial_w2", "Spatial W2"),
        ("expression_w2", "Expression W2"),
        ("spatial_rms_displacement", "Spatial paired RMS displacement"),
        ("expression_rms_displacement", "Expression paired RMS displacement"),
    )
    with mpl.rc_context(_figure_rc()):
        fig, axes = plt.subplots(2, 2, figsize=(8.27, 6.4), squeeze=False)
        for panel_index, (column, title) in enumerate(panels):
            ax = axes.flat[panel_index]
            values = [
                float(
                    final_table.loc[final_table["dataset"] == dataset, column].iloc[0]
                )
                for dataset in order
            ]
            ax.bar(x, values, width=0.68, color="#07838B")
            ax.set_title(title, fontweight="bold")
            ax.set_xticks(x, labels, rotation=24, ha="right")
            ax.set_ylabel("Distance")
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", color="#D7DDE2", linewidth=0.6)
        fig.text(0.035, 0.98, "b", fontsize=14, fontweight="bold", va="top")
        fig.text(
            0.075,
            0.98,
            "Fixed-checkpoint interaction sensitivity",
            fontsize=12,
            fontweight="bold",
            va="top",
        )
        fig.subplots_adjust(
            left=0.10, right=0.98, top=0.90, bottom=0.14, hspace=0.42, wspace=0.30
        )
        fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.08)
        fig.savefig(png_path, dpi=320, bbox_inches="tight", pad_inches=0.08)
        plt.close(fig)
    return pdf_path, png_path


def report(args: argparse.Namespace) -> Path:
    run_root = args.run_root.expanduser().resolve()
    output_dir = _require_empty(args.output_dir)
    current_runner_sha = _sha256(Path(__file__))
    records: dict[str, dict[str, Any]] = {}
    metric_tables: list[pd.DataFrame] = []
    paired_tables: list[pd.DataFrame] = []
    manifest_artifacts: list[dict[str, Any]] = []
    for dataset in DATASETS:
        dataset_root = run_root / dataset
        manifest_path = dataset_root / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "complete" or manifest.get("dataset") != dataset:
            raise RuntimeError(
                f"Incomplete or mismatched dataset manifest: {manifest_path}"
            )
        if manifest["package"]["runner"]["sha256"] != current_runner_sha:
            raise RuntimeError(
                f"Dataset {dataset} was produced by different runner bytes."
            )
        protocol = manifest["protocol"]
        required = {
            "same_trained_checkpoint": True,
            "training_performed": False,
            "same_initial_cells": True,
            "same_branch_seed": True,
            "continuous_from_source": True,
            "observed_slice_reanchoring": False,
            "growth_alpha": 0.0,
            "fixed_population": True,
            "expression_dimensions": 50,
        }
        if any(protocol.get(key) != value for key, value in required.items()):
            raise RuntimeError(
                f"Dataset {dataset} protocol is not the formal contract."
            )
        trajectories = dataset_root / "ablation" / "trajectories"
        on = _load_trajectory(trajectories / "interaction_on_points.npy")
        off = _load_trajectory(trajectories / "interaction_off_points.npy")
        times = tuple(float(value) for value in protocol["time_points"])
        records[dataset] = {"on": on, "off": off, "times": times}
        metrics = pd.read_csv(
            dataset_root / "ablation" / "interaction_ablation_metrics.csv"
        )
        metrics.insert(0, "dataset", dataset)
        metrics["space"] = metrics["space"].replace({"latent": "expression"})
        metric_tables.append(metrics)
        paired = pd.read_csv(dataset_root / "paired_displacement_metrics.csv")
        paired.insert(0, "dataset", dataset)
        paired_tables.append(paired)
        manifest_artifacts.append(_artifact(manifest_path))

    metrics_all = pd.concat(metric_tables, ignore_index=True)
    paired_all = pd.concat(paired_tables, ignore_index=True)
    metrics_path = output_dir / "distribution_metrics.csv"
    paired_path = output_dir / "paired_displacement_metrics.csv"
    metrics_all.to_csv(metrics_path, index=False)
    paired_all.to_csv(paired_path, index=False)

    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        final_time = float(records[dataset]["times"][-1])
        distribution = metrics_all.loc[
            (metrics_all["dataset"] == dataset)
            & np.isclose(metrics_all["time"], final_time)
        ]
        paired = paired_all.loc[
            (paired_all["dataset"] == dataset)
            & np.isclose(paired_all["time"], final_time)
        ]
        rows.append(
            {
                "dataset": dataset,
                "display_name": DISPLAY_NAMES[dataset],
                "final_time": final_time,
                "n_cells": int(
                    distribution.loc[
                        distribution["space"] == "joint", "n_baseline"
                    ].iloc[0]
                ),
                "joint_w2": float(
                    distribution.loc[distribution["space"] == "joint", "w2"].iloc[0]
                ),
                "spatial_w2": float(
                    distribution.loc[distribution["space"] == "spatial", "w2"].iloc[0]
                ),
                "expression_w2": float(
                    distribution.loc[distribution["space"] == "expression", "w2"].iloc[
                        0
                    ]
                ),
                "joint_rms_displacement": float(
                    paired.loc[paired["space"] == "joint", "rms_displacement"].iloc[0]
                ),
                "spatial_rms_displacement": float(
                    paired.loc[paired["space"] == "spatial", "rms_displacement"].iloc[0]
                ),
                "expression_rms_displacement": float(
                    paired.loc[
                        paired["space"] == "expression", "rms_displacement"
                    ].iloc[0]
                ),
            }
        )
    final_table = pd.DataFrame(rows)
    final_path = output_dir / "final_effect_summary.csv"
    final_table.to_csv(final_path, index=False)

    snapshot_pdf, snapshot_png = _plot_endpoint_snapshots(records, output_dir)
    metrics_pdf, metrics_png = _plot_effect_metrics(final_table, output_dir)
    caption_path = output_dir / "CAPTION.md"
    caption_lines = [
        "# Fixed-checkpoint interaction on/off sensitivity",
        "",
        "(a) Endpoint spatial distributions propagated continuously from the same ",
        "earliest observed cells with the trained interaction force retained or set ",
        "to zero. (b) Endpoint distribution W2 and row-paired RMS displacement in ",
        "spatial and 50-dimensional expression state. Both branches use the same ",
        "trained full-model checkpoints, source cells, diffusion seed, and numerical ",
        "settings. Growth-dependent resampling is disabled. This single-seed analysis ",
        "measures fixed-model sensitivity and is not a retraining ablation, causal ",
        "knockout, or uncertainty estimate.",
        "",
        "Final-time values:",
    ]
    for row in final_table.itertuples(index=False):
        caption_lines.append(
            f"- {row.display_name}: spatial W2={row.spatial_w2:.4g}, "
            f"expression W2={row.expression_w2:.4g}, spatial paired RMS="
            f"{row.spatial_rms_displacement:.4g}, expression paired RMS="
            f"{row.expression_rms_displacement:.4g}."
        )
    caption_lines.append("")
    caption_path.write_text("\n".join(caption_lines), encoding="utf-8")

    provenance_path = output_dir / "PROVENANCE.md"
    provenance_path.write_text(
        "\n".join(
            (
                "# Figure provenance",
                "",
                f"- Created: {_utc_now()}",
                f"- Source run root: `{run_root}`",
                f"- Plotting script: `{Path(__file__).resolve()}`",
                f"- Plotting script SHA-256: `{current_runner_sha}`",
                "- Experiment: five accepted full-model checkpoints, virtual "
                "interaction force on/off at inference only.",
                "- Rebuild: run the `report` subcommand with the same run root and a "
                "new empty output directory.",
                "- Interpretation: fixed-model single-seed sensitivity; no retraining "
                "and no causal claim.",
                "",
            )
        ),
        encoding="utf-8",
    )
    outputs = [
        metrics_path,
        paired_path,
        final_path,
        snapshot_pdf,
        snapshot_png,
        metrics_pdf,
        metrics_png,
        caption_path,
        provenance_path,
    ]
    report_manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "created_at": _utc_now(),
        "claim_scope": "fixed-checkpoint interaction-force sensitivity",
        "dataset_manifests": manifest_artifacts,
        "runner": _artifact(Path(__file__)),
        "outputs": [_artifact(path) for path in outputs],
    }
    manifest_path = output_dir / "report_manifest.json"
    manifest_path.write_text(
        json.dumps(report_manifest, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "report_manifest.json.sha256").write_text(
        f"{_sha256(manifest_path)}  {manifest_path.name}\n", encoding="utf-8"
    )
    print(manifest_path)
    return manifest_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="Run one dataset condition pair.")
    run_parser.add_argument("--dataset", choices=DATASETS, required=True)
    run_parser.add_argument("--aligned-h5ad", type=Path, required=True)
    run_parser.add_argument("--expected-aligned-sha256", required=True)
    run_parser.add_argument("--model-dir", type=Path, required=True)
    run_parser.add_argument("--expected-training-summary-sha256", required=True)
    run_parser.add_argument("--output-dir", type=Path, required=True)
    run_parser.add_argument("--device", default="cuda:0")

    report_parser = subparsers.add_parser("report", help="Aggregate five runs.")
    report_parser.add_argument("--run-root", type=Path, required=True)
    report_parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "run":
        run_dataset(args)
    else:
        report(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
