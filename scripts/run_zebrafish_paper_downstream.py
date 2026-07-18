#!/usr/bin/env python3
"""Reproduce the zebrafish manuscript downstream analyses from a native model.

This runner consumes the clean-counts aligned AnnData and a *current* CytoBridge
six-stage training directory.  It intentionally does not read historical
interpolated H5AD files, cached communication pickles, or copied manuscript
PDFs.  Every biological result is regenerated through public CytoBridge APIs.

The workflow keeps visualization-only spatial warping isolated from model-state
analyses:

* S22 uses one global continuous split-SDE on a fixed dense grid and applies
  the historical piecewise warp only to displayed coordinates (k=8).  Mosaic
  frames are selected from that same trajectory; integer frames are observed
  cells and half-time frames are generated cells.
* S24 virtual ablation and S25/communication use unwarped trajectories.
* Communication uses a hybrid no-warp series: observed cells at integer times
  and generated cells at intermediate times.

Stages are resumable.  A completed stage is skipped when its input/settings
signature and every recorded output still match; pass ``--force`` to rerun it.

Examples
--------
Run all manuscript analyses::

    python scripts/run_zebrafish_paper_downstream.py \
      --aligned-h5ad RUN/preprocess/zebrafish_aligned.h5ad \
      --model-dir RUN/conditions/alpha_express_0015/training \
      --lr-database RUN/assets/CellChatDB.ligrec.zebrafish.csv \
      --output-dir RUN/conditions/alpha_express_0015/paper_downstream \
      --stage all --profile full --device cuda

Resume only S25 and communication::

    python scripts/run_zebrafish_paper_downstream.py ... \
      --stage s25,communication

The default shared classifier cache is placed beside the preprocessing folder,
so two alpha-expression conditions trained from the same aligned H5AD reuse the
same label classifiers.  Use ``--shared-cache-dir`` to choose another location.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Callable, Mapping, Optional, Sequence

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import CytoBridge as cb  # noqa: E402


ALL_STAGES = (
    "classifier",
    "velocity",
    "s22",
    "growth",
    "ablation",
    "s25",
    "communication",
)
OBSERVED_TIMES = (0.0, 1.0, 2.0, 3.0, 4.0)
HALF_TIMES = tuple(float(value) for value in np.arange(0.0, 4.0 + 0.5, 0.5))
MAIN_CLASSIFIER_CACHE_TAG = "zebrafish-paper-main-spatial2-latent10"
S22_MOSAIC_COLUMNS = 3
S25_HEATMAP_COLUMNS = 2
IMPLEMENTATION_SOURCE_RELATIVE_PATHS = tuple(
    sorted(
        path.relative_to(REPO_ROOT).as_posix()
        for path in (REPO_ROOT / "CytoBridge").rglob("*.py")
        if path.is_file()
    )
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_ready(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_json_ready(item) for item in value.tolist()]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(item) for item in value]
    return value


def _stable_hash(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        _json_ready(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _git_revision() -> dict[str, object]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return {"commit": revision, "dirty": dirty}
    except Exception:
        return {"commit": None, "dirty": None}


def _require_file(path: str | Path, description: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Missing {description}: {resolved}")
    return resolved


def _require_dir(path: str | Path, description: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"Missing {description}: {resolved}")
    return resolved


def _time_grid(start: float, end: float, step: float) -> list[float]:
    if not np.isfinite(step) or float(step) <= 0:
        raise ValueError("time-grid step must be finite and > 0")
    span = float(end) - float(start)
    n_steps = int(round(span / float(step)))
    if not np.isclose(n_steps * float(step), span, rtol=0.0, atol=1e-8):
        raise ValueError(
            f"step={step} does not exactly tile interval [{start}, {end}]"
        )
    return [float(round(float(start) + i * float(step), 10)) for i in range(n_steps + 1)]


def _parse_stages(raw: str) -> list[str]:
    values = [value.strip().lower() for value in str(raw).split(",") if value.strip()]
    if not values:
        raise ValueError("--stage must not be empty")
    if "all" in values:
        if len(values) != 1:
            raise ValueError("Use --stage all by itself, or provide a comma-separated subset")
        return list(ALL_STAGES)
    unknown = sorted(set(values).difference(ALL_STAGES))
    if unknown:
        raise ValueError(f"Unknown stages {unknown}; valid stages: {list(ALL_STAGES)}")
    selected = set(values)
    # Always execute in dependency order, even if a caller writes
    # ``--stage communication,s25``.
    return [stage for stage in ALL_STAGES if stage in selected]


def _parse_csv_strings(raw: str) -> list[str]:
    values = [value.strip().lower() for value in str(raw).split(",") if value.strip()]
    if not values:
        raise ValueError("Expected at least one comma-separated value")
    return list(dict.fromkeys(values))


def _joint_features(
    adata: ad.AnnData,
    *,
    latent_key: str,
    spatial_key: str,
) -> np.ndarray:
    if latent_key not in adata.obsm:
        raise KeyError(f"adata.obsm['{latent_key}'] is required")
    if spatial_key not in adata.obsm:
        raise KeyError(f"adata.obsm['{spatial_key}'] is required")
    spatial = np.asarray(adata.obsm[spatial_key], dtype=np.float32)
    latent = np.asarray(adata.obsm[latent_key], dtype=np.float32)
    if spatial.ndim != 2 or spatial.shape[1] != 2:
        raise ValueError(
            f"Zebrafish spatial coordinates must be N x 2, got {spatial.shape}"
        )
    if spatial.shape[0] != latent.shape[0]:
        raise ValueError("Spatial and latent row counts do not match")
    return np.hstack((spatial, latent)).astype(np.float32, copy=False)


def _label_colors(
    adata: ad.AnnData,
    *,
    annotation_key: str,
    color_key: Optional[str],
) -> dict[str, str]:
    labels = adata.obs[annotation_key].astype(str).to_numpy()
    if color_key and color_key in adata.obs:
        table = pd.DataFrame(
            {
                "label": labels,
                "color": adata.obs[color_key].astype(str).to_numpy(),
            }
        )
        mapping: dict[str, str] = {}
        for label, subset in table.groupby("label", sort=False):
            colors = subset["color"].dropna().astype(str)
            colors = colors.loc[colors.str.strip().ne("") & colors.str.lower().ne("nan")]
            if not colors.empty:
                mapping[str(label)] = str(colors.mode().iloc[0])
        if set(np.unique(labels)).issubset(mapping):
            return mapping
    return cb.tl.load_label_to_color(labels)


def _minimal_state_adata(
    points: np.ndarray,
    labels: Sequence[object],
    *,
    annotation_key: str,
) -> ad.AnnData:
    points = np.asarray(points, dtype=np.float32)
    labels = np.asarray(labels).astype(str)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"State points must be N x D (D>=3), got {points.shape}")
    if labels.shape[0] != points.shape[0]:
        raise ValueError("State point and label counts do not match")
    result = ad.AnnData(X=points)
    result.obs[annotation_key] = labels
    result.obsm["spatial"] = points[:, :2].copy()
    return result


def _write_state_bundle(
    adata_dict: Mapping[str, ad.AnnData],
    time_points: Sequence[float],
    output_dir: str | Path,
    *,
    annotation_key: str,
    source_by_time: Optional[Mapping[float, str]] = None,
) -> list[Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    files: list[Path] = []
    for index, time_value in enumerate(time_points):
        key = str(float(time_value))
        if key not in adata_dict:
            raise KeyError(f"State dictionary is missing time key {key!r}")
        state = adata_dict[key]
        points = np.asarray(state.X, dtype=np.float32)
        # Store fixed-width Unicode, not an object array, so bundles remain
        # loadable with NumPy's safe ``allow_pickle=False`` default.
        labels = np.asarray(
            state.obs[annotation_key].astype(str).tolist(), dtype=str
        )
        filename = f"frame_{index:03d}.npz"
        path = output / filename
        np.savez_compressed(path, points=points, labels=labels)
        files.append(path)
        records.append(
            {
                "index": int(index),
                "time": float(time_value),
                "key": key,
                "file": filename,
                "sha256": _sha256(path),
                "n_cells": int(points.shape[0]),
                "feature_dim": int(points.shape[1]),
                "source": (
                    None
                    if source_by_time is None
                    else str(source_by_time[float(time_value)])
                ),
            }
        )
    index_path = output / "index.json"
    index_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "annotation_key": annotation_key,
                "frames": records,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    files.append(index_path)
    return files


def _read_state_bundle(
    bundle_dir: str | Path,
    *,
    annotation_key: str,
) -> tuple[dict[str, ad.AnnData], list[float], dict[float, str]]:
    bundle = Path(bundle_dir)
    index_path = _require_file(bundle / "index.json", "trajectory-state index")
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    states: dict[str, ad.AnnData] = {}
    times: list[float] = []
    sources: dict[float, str] = {}
    for record in payload["frames"]:
        time_value = float(record["time"])
        frame_path = _require_file(bundle / record["file"], "trajectory frame")
        expected_sha256 = record.get("sha256")
        if expected_sha256 is not None and _sha256(frame_path) != str(expected_sha256):
            raise RuntimeError(f"Trajectory frame checksum mismatch: {frame_path}")
        frame = np.load(frame_path)
        points = np.asarray(frame["points"], dtype=np.float32)
        labels = np.asarray(frame["labels"]).astype(str)
        states[str(time_value)] = _minimal_state_adata(
            points, labels, annotation_key=annotation_key
        )
        times.append(time_value)
        if record.get("source") is not None:
            sources[time_value] = str(record["source"])
    return states, times, sources


@dataclass
class RunContext:
    args: argparse.Namespace
    adata: ad.AnnData
    df: pd.DataFrame
    loaded: object
    runtime: object
    dim: int
    spatial_dim: int
    output_dir: Path
    shared_cache_dir: Path
    label_to_color: dict[str, str]
    common_signature: dict[str, object]


def _stage_manifest_path(ctx: RunContext, stage: str) -> Path:
    return ctx.output_dir / stage / "stage_manifest.json"


def _recorded_outputs_exist(manifest: Mapping[str, object]) -> bool:
    outputs = manifest.get("outputs", [])
    if not outputs:
        return False
    artifacts = manifest.get("output_artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != len(outputs):
        # Backward-compatible handling for manifests written before checksums
        # were recorded.  Empty files are never considered resumable outputs.
        return all(
            Path(str(path)).is_file() and Path(str(path)).stat().st_size > 0
            for path in outputs
        )
    for record in artifacts:
        if not isinstance(record, Mapping):
            return False
        path = Path(str(record.get("path", "")))
        if not path.is_file() or path.stat().st_size <= 0:
            return False
        if int(record.get("size_bytes", -1)) != int(path.stat().st_size):
            return False
        if str(record.get("sha256", "")) != _sha256(path):
            return False
    return True


def _require_current_stage_manifest(ctx: RunContext, stage: str) -> dict[str, object]:
    """Load an upstream stage only when it matches the current run contract."""
    manifest_path = _stage_manifest_path(ctx, stage)
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Current {stage!r} stage manifest is required: {manifest_path}."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_signature = _stable_hash(
        {
            "stage": stage,
            "common": ctx.common_signature,
            "settings": manifest.get("settings", {}),
        }
    )
    if manifest.get("status") != "complete":
        raise RuntimeError(f"Upstream stage {stage!r} is not complete: {manifest_path}")
    if manifest.get("signature") != expected_signature:
        raise RuntimeError(
            f"Upstream stage {stage!r} was produced by different data/model/code "
            f"settings. Rerun that stage before consuming it: {manifest_path}"
        )
    if not _recorded_outputs_exist(manifest):
        raise RuntimeError(
            f"Upstream stage {stage!r} has missing or modified outputs: {manifest_path}"
        )
    return manifest


def _execute_stage(
    ctx: RunContext,
    stage: str,
    settings: Mapping[str, object],
    action: Callable[[Path], tuple[Sequence[str | Path], Mapping[str, object]]],
) -> dict[str, object]:
    stage_dir = ctx.output_dir / stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    signature = _stable_hash(
        {
            "stage": stage,
            "common": ctx.common_signature,
            "settings": settings,
        }
    )
    manifest_path = _stage_manifest_path(ctx, stage)
    if manifest_path.exists() and not ctx.args.force:
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            prior.get("status") == "complete"
            and prior.get("signature") == signature
            and _recorded_outputs_exist(prior)
        ):
            print(f"[resume] {stage}: signature matches; keeping existing outputs")
            return prior
        print(f"[resume] {stage}: inputs/settings changed or outputs missing; rerunning")

    started = time.time()
    print(f"[stage] {stage}: start")
    outputs, details = action(stage_dir)
    output_paths = [str(Path(path).expanduser().resolve()) for path in outputs]
    missing = [path for path in output_paths if not Path(path).is_file()]
    if missing:
        raise RuntimeError(f"Stage {stage!r} reported missing outputs: {missing}")
    empty = [path for path in output_paths if Path(path).stat().st_size <= 0]
    if empty:
        raise RuntimeError(f"Stage {stage!r} reported empty outputs: {empty}")
    output_artifacts = [
        {
            "path": path,
            "size_bytes": int(Path(path).stat().st_size),
            "sha256": _sha256(Path(path)),
        }
        for path in output_paths
    ]
    manifest = {
        "schema_version": 1,
        "stage": stage,
        "status": "complete",
        "signature": signature,
        "started_at": datetime.fromtimestamp(started, tz=timezone.utc).isoformat(),
        "completed_at": _utc_now(),
        "elapsed_seconds": float(time.time() - started),
        "settings": _json_ready(settings),
        "details": _json_ready(details),
        "outputs": output_paths,
        "output_artifacts": output_artifacts,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"[stage] {stage}: done in {manifest['elapsed_seconds']:.1f}s")
    return manifest


def _main_classifier_settings(ctx: RunContext) -> dict[str, object]:
    epochs = 2 if ctx.args.profile == "smoke" else int(ctx.args.classifier_epochs)
    return {
        "contract": "time + aligned spatial(2) + leading latent PCs(10)",
        "n_joint_features": 12,
        "hidden_size": 128,
        "epochs": epochs,
        "learning_rate": 1e-3,
        "test_size": 0.1,
        "best_epoch_metric": "accuracy",
        "train_on_full_data": False,
        "knn_neighbors": 10,
        "seed": int(ctx.args.random_seed),
        "cache_dir": str(ctx.shared_cache_dir / "trajectory_classifier"),
        "cache_tag": MAIN_CLASSIFIER_CACHE_TAG,
    }


def _train_main_classifier(ctx: RunContext):
    settings = _main_classifier_settings(ctx)
    cached, cache_path = cb.tl.train_cached_mlp_classifier_from_adata(
        ctx.adata,
        cache_dir=settings["cache_dir"],
        cache_tag=MAIN_CLASSIFIER_CACHE_TAG,
        label_col=ctx.args.annotation_key,
        time_key=ctx.args.time_key,
        obsm_key=ctx.args.latent_key,
        spatial_key=ctx.args.spatial_key,
        concat_spatial=True,
        hidden_size=int(settings["hidden_size"]),
        epochs=int(settings["epochs"]),
        lr=float(settings["learning_rate"]),
        test_size=float(settings["test_size"]),
        seed=int(settings["seed"]),
        device=ctx.args.device,
        include_time_feature=True,
        n_features=int(settings["n_joint_features"]),
        best_epoch_metric=str(settings["best_epoch_metric"]),
        train_on_full_data=bool(settings["train_on_full_data"]),
    )
    return cached, Path(cache_path)


def _stage_classifier(ctx: RunContext) -> dict[str, object]:
    settings = _main_classifier_settings(ctx)

    def action(stage_dir: Path):
        cached, cache_path = _train_main_classifier(ctx)
        summary_path = stage_dir / "classifier_summary.json"
        summary = {
            "cache_path": str(cache_path),
            "accuracy": cached.accuracy,
            "balanced_accuracy": cached.balanced_accuracy,
            "metadata": cached.metadata,
            "evaluation": cached.evaluation,
        }
        summary_path.write_text(
            json.dumps(_json_ready(summary), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return [cache_path, summary_path], summary

    return _execute_stage(ctx, "classifier", settings, action)


def _stage_velocity(ctx: RunContext) -> dict[str, object]:
    interaction_cfg = ctx.loaded.config["model"]["interaction_net"]
    settings = {
        "time_points": [0.0, 2.0, 4.0],
        "components": {
            "drift": "intrinsic",
            "interaction": "interaction",
            "score": "score",
            "full": "full",
        },
        "projection_modes": {
            "spatial_direct": "component[:, :2], feature_matrix=None",
            "latent_to_spatial": (
                "component[:, 2:] with feature_matrix=joint_features[:, 2:], "
                "projected onto aligned spatial coordinates"
            ),
        },
        "interaction_m": int(ctx.args.interaction_m),
        "interaction_cutoff": float(interaction_cfg["cutoff"]),
        "n_neighbors": int(ctx.args.velocity_neighbors),
    }

    def action(stage_dir: Path):
        import matplotlib.pyplot as plt

        components = cb.tl.compute_velocity_components_from_adata(
            ctx.adata,
            ctx.loaded.model,
            dim=ctx.dim,
            interaction_m=int(ctx.args.interaction_m),
            interaction_threshold=float(interaction_cfg["cutoff"]),
            device=ctx.args.device,
            time_key=ctx.args.time_key,
            obsm_key=ctx.args.latent_key,
            spatial_key=ctx.args.spatial_key,
            concat_spatial=True,
            write_to_adata=False,
            reuse_if_present=False,
        )
        data_path = stage_dir / "velocity_components.npz"
        np.savez_compressed(
            data_path,
            times=components["times"],
            features=components["features"],
            drift=components["drift"],
            interaction=components["interaction"],
            score=components["score"],
            full=components["full"],
        )
        outputs: list[Path] = [data_path]
        plot_fallbacks: list[dict[str, str]] = []
        labels_all = ctx.adata.obs[ctx.args.annotation_key].astype(str).to_numpy()
        for time_value in settings["time_points"]:
            mask = np.isclose(
                components["times"], float(time_value), rtol=0.0, atol=1e-9
            )
            if not np.any(mask):
                raise ValueError(f"No cells at requested velocity time {time_value}")
            coords = components["features"][mask, :2]
            features = components["features"][mask]
            labels = labels_all[mask]
            for name, panel_label in settings["components"].items():
                direct_path = (
                    stage_dir
                    / f"spatial_direct_{panel_label}_t{time_value:g}.pdf"
                )
                direct_plot = cb.pl.plot_velocity_component(
                    coords=coords,
                    velocity=components[name][mask, :2],
                    feature_matrix=None,
                    labels=labels,
                    label_to_color=ctx.label_to_color,
                    title=f"{panel_label} velocity (spatial direct), t={time_value:g}",
                    out_path=str(direct_path),
                    density=2.0,
                    basis="spatial",
                    show_legend=False,
                    n_neighbors=int(ctx.args.velocity_neighbors),
                )
                direct_reason = getattr(direct_plot, "uns", {}).get(
                    "velocity_plot_fallback"
                )
                if direct_reason:
                    plot_fallbacks.append(
                        {"file": str(direct_path), "reason": str(direct_reason)}
                    )
                plt.close("all")
                outputs.append(direct_path)

                latent_path = (
                    stage_dir
                    / f"latent_to_spatial_{panel_label}_t{time_value:g}.pdf"
                )
                latent_plot = cb.pl.plot_velocity_component(
                    coords=coords,
                    velocity=components[name][mask, 2:],
                    feature_matrix=features[:, 2:],
                    labels=labels,
                    label_to_color=ctx.label_to_color,
                    title=(
                        f"{panel_label} velocity (latent to spatial), "
                        f"t={time_value:g}"
                    ),
                    out_path=str(latent_path),
                    density=2.0,
                    basis="spatial",
                    show_legend=False,
                    n_neighbors=int(ctx.args.velocity_neighbors),
                )
                latent_reason = getattr(latent_plot, "uns", {}).get(
                    "velocity_plot_fallback"
                )
                if latent_reason:
                    plot_fallbacks.append(
                        {"file": str(latent_path), "reason": str(latent_reason)}
                    )
                plt.close("all")
                outputs.append(latent_path)
        identity_error = float(
            np.max(
                np.abs(
                    components["full"]
                    - components["drift"]
                    - components["interaction"]
                    - components["score"]
                )
            )
        )
        return outputs, {
            "n_cells": int(ctx.adata.n_obs),
            "full_identity_max_error": identity_error,
            "all_finite": bool(
                all(
                    np.isfinite(components[name]).all()
                    for name in settings["components"]
                )
            ),
            "plot_fallbacks": plot_fallbacks,
        }

    return _execute_stage(ctx, "velocity", settings, action)


def _run_interpolation(
    ctx: RunContext,
    *,
    output_dir: Path,
    time_points: Sequence[float],
    use_real_for_observed: bool,
    display_piecewise_warp: bool,
):
    interp = [
        float(value)
        for value in time_points
        if not any(np.isclose(value, obs, rtol=0.0, atol=1e-9) for obs in OBSERVED_TIMES)
    ]
    classifier_settings = _main_classifier_settings(ctx)
    return cb.tl.run_interpolation_workflow(
        df=ctx.df,
        dim=ctx.dim,
        annotation_key=ctx.args.annotation_key,
        runtime=ctx.runtime,
        device=ctx.args.device,
        output_dir=str(output_dir),
        requested_plot_points=[float(value) for value in time_points],
        interp_time_points=interp,
        no_interp=False,
        use_real_for_observed=bool(use_real_for_observed),
        classifier_cache_dir=str(ctx.shared_cache_dir / "trajectory_classifier"),
        classifier_cache_tag=MAIN_CLASSIFIER_CACHE_TAG,
        classifier_adata=ctx.adata,
        classifier_time_key=ctx.args.time_key,
        classifier_obsm_key=ctx.args.latent_key,
        classifier_spatial_key=ctx.args.spatial_key,
        classifier_concat_spatial=True,
        classifier_epochs=int(classifier_settings["epochs"]),
        classifier_hidden_size=int(classifier_settings["hidden_size"]),
        classifier_lr=float(classifier_settings["learning_rate"]),
        classifier_test_size=float(classifier_settings["test_size"]),
        classifier_train_on_full_data=False,
        classifier_best_metric="accuracy",
        classifier_n_pcs=12,
        classifier_knn_neighbors=10,
        sde_n_samples=(
            int(ctx.args.smoke_n_samples)
            if ctx.args.profile == "smoke"
            else ctx.args.sde_n_samples
        ),
        skip_nonsplit_sde=True,
        split_sde_dt=float(ctx.args.sde_dt),
        split_sigma_scalar=float(ctx.args.sde_sigma),
        split_growth_alpha=float(ctx.args.growth_alpha),
        split_interaction_m=int(ctx.args.interaction_m),
        split_resample_dt=float(ctx.args.sde_dt),
        split_max_particles=int(ctx.args.sde_max_particles),
        spatial_warp_to_observed_piecewise=bool(display_piecewise_warp),
        spatial_warp_visualization_only=bool(display_piecewise_warp),
        spatial_warp_k=8,
        spatial_warp_eps=1e-6,
        random_seed=int(ctx.args.random_seed),
    )


def _stage_s22(ctx: RunContext) -> dict[str, object]:
    video_step = 1.0 if ctx.args.profile == "smoke" else float(ctx.args.video_step)
    video_times = _time_grid(0.0, 4.0, video_step)
    if ctx.args.profile == "smoke":
        # Retain at least one generated frame in every observed interval.
        video_times = list(HALF_TIMES)
        simulation_times = list(HALF_TIMES)
    else:
        simulation_times = _time_grid(
            0.0, 4.0, float(ctx.args.s22_simulation_step)
        )
        missing_render_times = [
            value
            for value in [*HALF_TIMES, *video_times]
            if not any(
                np.isclose(value, simulated, rtol=0.0, atol=1e-9)
                for simulated in simulation_times
            )
        ]
        if missing_render_times:
            raise ValueError(
                "S22 mosaic/video times must be a subset of the fixed simulation "
                f"grid; missing={sorted(set(missing_render_times))}. Choose "
                "--video-step as an integer multiple of --s22-simulation-step."
            )
    formats = _parse_csv_strings(ctx.args.video_formats)
    unsupported = sorted(set(formats).difference({"gif", "mp4"}))
    if unsupported:
        raise ValueError(f"Unsupported --video-formats values: {unsupported}")
    settings = {
        "mosaic_times": list(HALF_TIMES),
        "video_times": video_times,
        "observed_integer_frames": True,
        "generated_noninteger_frames": True,
        "simulation": (
            "one global split SDE on the dense grid; x and log-mass are continuous "
            "across observed intervals"
        ),
        "simulation_grid": list(simulation_times),
        "simulation_step": (
            None
            if ctx.args.profile == "smoke"
            else float(ctx.args.s22_simulation_step)
        ),
        "mosaic_is_subsample_of_dense_trajectory": True,
        "mosaic_layout": {
            "columns": S22_MOSAIC_COLUMNS,
            "show_axes": False,
            "show_legend": True,
        },
        "canonical_prewarp_trajectory_consumers": ["s25", "communication"],
        "dt": float(ctx.args.sde_dt),
        "split_resample_dt": float(ctx.args.sde_dt),
        "sigma": float(ctx.args.sde_sigma),
        "growth_alpha": float(ctx.args.growth_alpha),
        "interaction_m": int(ctx.args.interaction_m),
        "sde_n_samples": (
            int(ctx.args.smoke_n_samples)
            if ctx.args.profile == "smoke"
            else ctx.args.sde_n_samples
        ),
        "max_particles": int(ctx.args.sde_max_particles),
        "display_warp": {
            "piecewise": True,
            "visualization_only": True,
            "k": 8,
            "eps": 1e-6,
        },
        "classifier": _main_classifier_settings(ctx),
        "video_fps": int(ctx.args.video_fps),
        "video_formats": formats,
        "point_size": float(ctx.args.point_size),
    }

    def action(stage_dir: Path):
        import matplotlib.pyplot as plt

        dense_result = _run_interpolation(
            ctx,
            output_dir=stage_dir / "workflow_shared_dense",
            time_points=simulation_times,
            use_real_for_observed=True,
            display_piecewise_warp=True,
        )
        source_by_time = {
            float(value): (
                "observed"
                if any(np.isclose(value, obs, rtol=0.0, atol=1e-9) for obs in OBSERVED_TIMES)
                else "generated_display_warp"
            )
            for value in HALF_TIMES
        }
        outputs = _write_state_bundle(
            dense_result.adata_dict,
            HALF_TIMES,
            stage_dir / "mosaic_states",
            annotation_key=ctx.args.annotation_key,
            source_by_time=source_by_time,
        )
        if (
            dense_result.sde_points_split_prewarp is None
            or dense_result.predicted_labels_split_prewarp is None
        ):
            raise RuntimeError(
                "S22 display-only warp did not preserve the canonical pre-warp "
                "trajectory and labels."
            )
        canonical_states: dict[str, ad.AnnData] = {}
        for time_value in HALF_TIMES:
            matches = [
                index
                for index, simulated in enumerate(dense_result.ts_points)
                if np.isclose(time_value, simulated, rtol=0.0, atol=1e-9)
            ]
            if len(matches) != 1:
                raise RuntimeError(
                    f"Canonical S22 trajectory has {len(matches)} frames for "
                    f"t={time_value}; expected exactly one."
                )
            index = matches[0]
            canonical_states[str(float(time_value))] = _minimal_state_adata(
                np.asarray(
                    dense_result.sde_points_split_prewarp[index], dtype=np.float32
                ),
                np.asarray(
                    dense_result.predicted_labels_split_prewarp[index]
                ).astype(str),
                annotation_key=ctx.args.annotation_key,
            )
        prewarp_source_by_time = {
            float(value): (
                "observed_seed_predicted_labels"
                if np.isclose(value, 0.0, rtol=0.0, atol=1e-9)
                else "generated_prewarp"
            )
            for value in HALF_TIMES
        }
        prewarp_outputs = _write_state_bundle(
            canonical_states,
            HALF_TIMES,
            stage_dir / "canonical_prewarp_states",
            annotation_key=ctx.args.annotation_key,
            source_by_time=prewarp_source_by_time,
        )
        outputs.extend(prewarp_outputs)
        source_table = pd.DataFrame(
            {
                "time": list(HALF_TIMES),
                "display_source": [
                    source_by_time[float(value)] for value in HALF_TIMES
                ],
                "canonical_prewarp_source": [
                    prewarp_source_by_time[float(value)] for value in HALF_TIMES
                ],
                "s25_analysis_source": [
                    (
                        "observed_actual_annotation"
                        if float(value) in OBSERVED_TIMES
                        else "generated_prewarp_direct_classifier"
                    )
                    for value in HALF_TIMES
                ],
                "communication_source": [
                    (
                        "observed_actual_annotation"
                        if float(value) in OBSERVED_TIMES
                        else "generated_prewarp"
                    )
                    for value in HALF_TIMES
                ],
            }
        )
        source_path = stage_dir / "frame_sources.csv"
        source_table.to_csv(source_path, index=False)
        outputs.append(source_path)

        snapshot_dir = stage_dir / "mosaic_snapshots"
        cb.tl.save_timepoint_snapshots(
            adata_dict=dense_result.adata_dict,
            time_keys=[str(float(value)) for value in HALF_TIMES],
            annotation_key=ctx.args.annotation_key,
            label_to_color=ctx.label_to_color,
            snapshot_dir=str(snapshot_dir),
            background_color="white",
            font_color="black",
            snapshot_point_size=float(ctx.args.point_size),
            snapshot_alpha=0.9,
            mosaic_cols=S22_MOSAIC_COLUMNS,
            mosaic_cell_size=3.0,
            mosaic_show_title=True,
            save_pdf=True,
        )
        outputs.extend(sorted(snapshot_dir.glob("*")))

        mosaic_points = np.empty(len(HALF_TIMES), dtype=object)
        mosaic_labels: list[np.ndarray] = []
        for index, time_value in enumerate(HALF_TIMES):
            state = dense_result.adata_dict[str(float(time_value))]
            mosaic_points[index] = np.asarray(state.X, dtype=np.float32)
            mosaic_labels.append(
                state.obs[ctx.args.annotation_key].astype(str).to_numpy()
            )
        mosaic_pdf = stage_dir / "S22_piecewise_display_warp_mosaic.pdf"
        fig = cb.pl.plot_trajectory_grid(
            sde_points=mosaic_points,
            time_values=HALF_TIMES,
            dim_pairs=((0, 1),),
            labels_list=mosaic_labels,
            label_to_color=ctx.label_to_color,
            out_path=str(mosaic_pdf),
            figsize_per_panel=(2.6, 2.6),
            point_size=float(ctx.args.point_size),
            alpha=0.9,
            title="Zebrafish trajectory (observed integers; generated half-times)",
            n_cols=S22_MOSAIC_COLUMNS,
            show_axes=False,
            show_legend=True,
            equal_aspect=True,
            legend_title="Cell type",
            legend_fontsize=6.0,
        )
        mosaic_png = stage_dir / "S22_piecewise_display_warp_mosaic.png"
        fig.savefig(mosaic_png, dpi=240, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        outputs.extend([mosaic_pdf, mosaic_png])

        dense_points = np.empty(len(video_times), dtype=object)
        dense_labels: list[np.ndarray] = []
        for index, time_value in enumerate(video_times):
            state = dense_result.adata_dict[str(float(time_value))]
            dense_points[index] = np.asarray(state.X, dtype=np.float32)
            dense_labels.append(
                state.obs[ctx.args.annotation_key].astype(str).to_numpy()
            )
        animation_errors: dict[str, str] = {}
        for extension in formats:
            animation_path = stage_dir / f"S22_piecewise_display_warp_dense.{extension}"
            if extension == "mp4" and shutil.which("ffmpeg") is None:
                animation_errors[extension] = "ffmpeg is not installed"
                continue
            try:
                cb.pl.plot_trajectory_gif(
                    sde_points=dense_points,
                    time_values=video_times,
                    labels_list=dense_labels,
                    label_to_color=ctx.label_to_color,
                    out_path=str(animation_path),
                    dim_pair=(0, 1),
                    point_size=max(1.0, float(ctx.args.point_size)),
                    alpha=0.9,
                    fps=int(ctx.args.video_fps),
                )
                outputs.append(animation_path)
            except Exception as exc:
                if extension != "mp4":
                    raise
                animation_errors[extension] = f"{type(exc).__name__}: {exc}"

        return outputs, {
            "classifier_cache_path": dense_result.classifier_cache_path,
            "classifier_accuracy": dense_result.classifier_accuracy,
            "classifier_balanced_accuracy": dense_result.classifier_balanced_accuracy,
            "mosaic_cell_counts": {
                str(float(t)): int(dense_result.adata_dict[str(float(t))].n_obs)
                for t in HALF_TIMES
            },
            "video_frame_count": int(len(video_times)),
            "simulation_frame_count": int(len(simulation_times)),
            "shared_simulation_for_mosaic_and_video": True,
            "animation_errors": animation_errors,
            "prewarp_states_are_used_for_labels": True,
            "canonical_prewarp_state_index": str(
                (stage_dir / "canonical_prewarp_states" / "index.json").resolve()
            ),
            "canonical_prewarp_state_index_sha256": _sha256(
                stage_dir / "canonical_prewarp_states" / "index.json"
            ),
            "simulation_seeds": dense_result.simulation_seeds,
        }

    return _execute_stage(ctx, "s22", settings, action)


def _stage_growth(ctx: RunContext) -> dict[str, object]:
    settings = {
        "observed_times": list(OBSERVED_TIMES),
        "normalization": "independent per-time 5th-95th percentile scaling",
        "raw_growth_exported": True,
        "composite_layout": {
            "columns": 2,
            "shared_display_scale": [0.0, 1.0],
        },
        "point_size": float(ctx.args.point_size),
    }

    def action(stage_dir: Path):
        observed_float_keys = _observed_state_dict(ctx)
        observed_states = {
            str(float(time_value)): state
            for time_value, state in observed_float_keys.items()
        }
        raw = cb.tl.evaluate_growth_by_timepoint(
            observed_states,
            model=ctx.loaded.model,
            time_points=OBSERVED_TIMES,
            annotation_key=ctx.args.annotation_key,
            spatial_key="spatial",
            value_key="growth_rate",
            device=ctx.args.device,
        )
        raw_path = stage_dir / "growth_per_cell.csv"
        raw.to_csv(raw_path, index=False)
        outputs: list[Path] = [raw_path]

        common_plot_options = {
            "adata_dict": observed_states,
            "spatial_key": "spatial",
            "value_key": "growth_rate",
            "cmap": "RdYlBu_r",
            "point_size": float(ctx.args.point_size),
            "lower_quantile": 0.05,
            "upper_quantile": 0.95,
            "scale_mode": "per_time_0_1",
            "colorbar_label": "g (scaled 5-95%)",
        }
        for time_index, time_value in enumerate(OBSERVED_TIMES):
            path = stage_dir / f"growth_t{time_index}.pdf"
            cb.pl.plot_growth_timepoint_grid(
                **common_plot_options,
                time_points=[time_value],
                time_keys=[str(float(time_value))],
                out_path=str(path),
                n_cols=1,
                shared_colorbar=False,
            )
            outputs.append(path)

        for extension in ("pdf", "png"):
            path = stage_dir / f"S23_growth_observed_grid.{extension}"
            cb.pl.plot_growth_timepoint_grid(
                **common_plot_options,
                time_points=OBSERVED_TIMES,
                out_path=str(path),
                n_cols=2,
                shared_colorbar=True,
                title="Growth-rate maps across observed zebrafish stages",
            )
            outputs.append(path)
        return outputs, {
            "n_panels": len(OBSERVED_TIMES),
            "n_growth_values": int(len(raw)),
            "composite_outputs": [
                str(stage_dir / "S23_growth_observed_grid.pdf"),
                str(stage_dir / "S23_growth_observed_grid.png"),
            ],
        }

    return _execute_stage(ctx, "growth", settings, action)


def _ablation_classifier(ctx: RunContext, stage_dir: Path):
    from sklearn.decomposition import PCA

    latent = np.asarray(ctx.adata.obsm[ctx.args.latent_key], dtype=np.float32)
    if latent.shape[1] < 10:
        raise ValueError(
            f"Ablation classifier requires at least 10 latent PCs, got {latent.shape[1]}"
        )
    pca = PCA(n_components=10, random_state=int(ctx.args.random_seed))
    latent_pca10 = pca.fit_transform(latent).astype(np.float32)
    classifier_features = np.hstack(
        (
            np.asarray(ctx.adata.obsm[ctx.args.spatial_key], dtype=np.float32),
            latent_pca10,
        )
    ).astype(np.float32)
    classifier_adata = ad.AnnData(
        X=np.zeros((ctx.adata.n_obs, 0), dtype=np.float32),
        obs=ctx.adata.obs[[ctx.args.annotation_key, ctx.args.time_key]].copy(),
    )
    classifier_adata.obsm["X_ablation_classifier"] = classifier_features
    epochs = 2 if ctx.args.profile == "smoke" else int(ctx.args.ablation_classifier_epochs)
    cached, cache_path = cb.tl.train_cached_mlp_classifier_from_adata(
        classifier_adata,
        cache_dir=ctx.shared_cache_dir / "ablation_classifier",
        cache_tag="zebrafish-paper-ablation-spatial2-pca10",
        label_col=ctx.args.annotation_key,
        time_key=ctx.args.time_key,
        obsm_key="X_ablation_classifier",
        concat_spatial=False,
        hidden_size=256,
        epochs=epochs,
        lr=1e-3,
        test_size=0.1,
        seed=int(ctx.args.random_seed),
        device=ctx.args.device,
        include_time_feature=True,
        n_features=12,
        best_epoch_metric="bacc",
        train_on_full_data=True,
    )
    pca_path = stage_dir / "ablation_classifier_pca10.npz"
    np.savez_compressed(
        pca_path,
        components=np.asarray(pca.components_, dtype=np.float32),
        mean=np.asarray(pca.mean_, dtype=np.float32),
        explained_variance=np.asarray(pca.explained_variance_, dtype=np.float32),
        explained_variance_ratio=np.asarray(
            pca.explained_variance_ratio_, dtype=np.float32
        ),
        singular_values=np.asarray(pca.singular_values_, dtype=np.float32),
        n_samples_seen=np.asarray([latent.shape[0]], dtype=np.int64),
    )

    def labeler(points, time_points):
        transformed = np.empty(len(points), dtype=object)
        for index, frame in enumerate(points):
            frame = np.asarray(frame, dtype=np.float32)
            transformed[index] = np.hstack(
                (frame[:, :2], pca.transform(frame[:, 2:]))
            ).astype(np.float32)
        return cb.tl.predict_labels_for_trajectories(
            sde_points=transformed,
            ts_points=time_points,
            model=cached.model,
            label_encoder=cached.label_encoder,
            feature_dim=12,
            device=ctx.args.device,
            knn_neighbors=10,
            include_time_feature=True,
        )

    return cached, Path(cache_path), pca_path, labeler


def _stage_ablation(ctx: RunContext) -> dict[str, object]:
    step = 1.0 if ctx.args.profile == "smoke" else float(ctx.args.ablation_step)
    time_points = _time_grid(0.0, 4.0, step)
    settings = {
        "time_points": time_points,
        "simulation": "continuous split SDE from observed t0; no warp, reanchor, replacement",
        "dt": float(ctx.args.sde_dt),
        "split_resample_dt": float(ctx.args.sde_dt),
        "sigma": float(ctx.args.sde_sigma),
        "growth_alpha": float(ctx.args.growth_alpha),
        "interaction_m": int(ctx.args.interaction_m),
        "n_samples": (
            int(ctx.args.smoke_n_samples)
            if ctx.args.profile == "smoke"
            else None
        ),
        "snapshot_point_size": float(ctx.args.point_size),
        "composite_layout": {
            "rows": "observed times",
            "columns": ["baseline", "remove_YSL", "remove_EVL"],
            "shared_axis_limits": True,
            "semantics": "virtual sensitivity analysis; not a causal knockout estimate",
        },
        "animation_fps": int(ctx.args.video_fps),
        "max_particles": int(ctx.args.sde_max_particles),
        "ablations": {
            "remove_YSL": [ctx.args.ysl_label],
            "remove_EVL": [ctx.args.evl_label],
        },
        "classifier": {
            "contract": "time + spatial2 + fresh PCA10(original latent50)",
            "hidden_size": 256,
            "epochs": (
                2
                if ctx.args.profile == "smoke"
                else int(ctx.args.ablation_classifier_epochs)
            ),
            "best_epoch_metric": "bacc",
            "train_on_full_data": True,
            "knn_neighbors": 10,
        },
        "random_stream_coupling": (
            "same branch-level seed; not cell-ID-matched after cohort removal; "
            "single-seed manuscript parity"
        ),
    }

    def action(stage_dir: Path):
        import matplotlib.pyplot as plt

        if ctx.args.profile == "full":
            t0_mask = np.isclose(
                ctx.df["samples"].to_numpy(dtype=float),
                0.0,
                rtol=0.0,
                atol=1e-9,
            )
            t0_labels = ctx.df.loc[t0_mask, ctx.args.annotation_key].astype(str)
            t0_counts = t0_labels.value_counts().to_dict()
            expected = {
                "total": 563,
                str(ctx.args.ysl_label): 29,
                str(ctx.args.evl_label): 272,
            }
            actual = {
                "total": int(t0_mask.sum()),
                str(ctx.args.ysl_label): int(
                    t0_counts.get(str(ctx.args.ysl_label), 0)
                ),
                str(ctx.args.evl_label): int(
                    t0_counts.get(str(ctx.args.evl_label), 0)
                ),
            }
            if actual != expected:
                raise ValueError(
                    "The clean-counts t0 cohort does not match the frozen S24 "
                    f"contract: expected {expected}, got {actual}."
                )
        cached, cache_path, pca_path, labeler = _ablation_classifier(ctx, stage_dir)
        result = cb.tl.run_virtual_cell_type_ablation(
            ctx.adata,
            ctx.runtime,
            ablations=settings["ablations"],
            time_points=time_points,
            output_dir=stage_dir / "experiment",
            time_index=0,
            n_samples=(
                int(ctx.args.smoke_n_samples)
                if ctx.args.profile == "smoke"
                else None
            ),
            dt=float(ctx.args.sde_dt),
            resample_dt=float(ctx.args.sde_dt),
            sigma=float(ctx.args.sde_sigma),
            growth_alpha=float(ctx.args.growth_alpha),
            interaction_m=int(ctx.args.interaction_m),
            max_particles=int(ctx.args.sde_max_particles),
            device=ctx.args.device,
            time_key=ctx.args.time_key,
            annotation_key=ctx.args.annotation_key,
            obsm_key=ctx.args.latent_key,
            spatial_key=ctx.args.spatial_key,
            concat_spatial=True,
            spatial_dim=2,
            random_seed=int(ctx.args.random_seed),
            common_random_seed=True,
            trajectory_labeler=labeler,
            save_data=True,
            save_snapshots=True,
            snapshot_times=OBSERVED_TIMES,
            snapshot_plot_dims=(0, 1),
            snapshot_point_size=float(ctx.args.point_size),
            snapshot_alpha=0.9,
            snapshot_formats=("png", "pdf"),
            label_to_color=ctx.label_to_color,
            verbose=True,
        )
        outputs = [cache_path, pca_path, *result.files]
        if ctx.args.profile == "full":
            expected_variant_counts = {"remove_YSL": 534, "remove_EVL": 291}
            actual_variant_counts = {
                str(key): int(value)
                for key, value in result.settings["variant_initial_counts"].items()
            }
            if actual_variant_counts != expected_variant_counts:
                raise RuntimeError(
                    "Unexpected S24 ablation cohort sizes: expected "
                    f"{expected_variant_counts}, got {actual_variant_counts}."
                )
        comparison_trajectories = {
            "baseline": result.baseline_points,
            **result.ablation_points,
        }
        comparison_labels = None
        if result.baseline_labels is not None:
            comparison_labels = {
                "baseline": result.baseline_labels,
                **result.ablation_labels,
            }
        condition_titles = {
            "baseline": "Baseline",
            "remove_YSL": "Virtual YSL removal",
            "remove_EVL": "Virtual EVL removal",
        }
        for extension in ("pdf", "png"):
            path = stage_dir / f"S24_virtual_ablation_grid.{extension}"
            figure = cb.pl.plot_trajectory_comparison_grid(
                trajectories=comparison_trajectories,
                time_values=time_points,
                labels_by_condition=comparison_labels,
                label_to_color=ctx.label_to_color,
                selected_times=OBSERVED_TIMES,
                condition_titles=condition_titles,
                dim_pair=(0, 1),
                point_size=float(ctx.args.point_size),
                alpha=0.9,
                shared_axis_limits=True,
                show_counts=False,
                show_legend=False,
                title="Virtual cell-type-removal sensitivity across zebrafish stages",
                out_path=str(path),
            )
            plt.close(figure)
            outputs.append(path)
        animation_dir = stage_dir / "animations"
        animation_dir.mkdir(parents=True, exist_ok=True)
        animation_points = {"baseline": result.baseline_points, **result.ablation_points}
        animation_labels = {
            "baseline": result.baseline_labels,
            **result.ablation_labels,
        }
        for name, points in animation_points.items():
            path = animation_dir / f"{name}.gif"
            cb.pl.plot_trajectory_gif(
                sde_points=points,
                time_values=time_points,
                labels_list=animation_labels.get(name),
                label_to_color=ctx.label_to_color,
                out_path=str(path),
                dim_pair=(0, 1),
                point_size=max(1.0, float(ctx.args.point_size)),
                alpha=0.9,
                fps=int(ctx.args.video_fps),
            )
            outputs.append(path)
        return outputs, {
            "classifier_cache_path": str(cache_path),
            "classifier_accuracy": cached.accuracy,
            "classifier_balanced_accuracy": cached.balanced_accuracy,
            "n_initial": int(len(result.initial_obs_names)),
            "variant_initial_counts": result.settings["variant_initial_counts"],
            "simulation_seeds": result.settings["simulation_seeds"],
        }

    return _execute_stage(ctx, "ablation", settings, action)


def _stage_s25(ctx: RunContext) -> dict[str, object]:
    top_n = 25 if ctx.args.profile == "smoke" else int(ctx.args.s25_top_genes)
    canonical_index = (
        ctx.output_dir / "s22" / "canonical_prewarp_states" / "index.json"
    )
    upstream_s22_manifest = (
        _require_current_stage_manifest(ctx, "s22")
        if canonical_index.exists()
        else None
    )
    settings = {
        "time_points": list(HALF_TIMES),
        "trajectory": (
            "observed integer states with actual annotations + canonical generated "
            "pre-warp half-time states from one global continuous split SDE"
        ),
        "dt": float(ctx.args.sde_dt),
        "split_resample_dt": float(ctx.args.sde_dt),
        "sigma": float(ctx.args.sde_sigma),
        "growth_alpha": float(ctx.args.growth_alpha),
        "interaction_m": int(ctx.args.interaction_m),
        "sde_n_samples": (
            int(ctx.args.smoke_n_samples)
            if ctx.args.profile == "smoke"
            else ctx.args.sde_n_samples
        ),
        "max_particles": int(ctx.args.sde_max_particles),
        "cell_type_filter": ctx.args.ysl_label,
        "target_classifier_knn_neighbors": int(
            ctx.args.s25_classifier_knn_neighbors
        ),
        "target_classifier_policy": (
            "direct classifier labels at generated half-times; spatial kNN smoothing "
            "is configurable separately from the S22 display-label policy"
        ),
        "top_variable_genes": top_n,
        "heatmap_panel_columns": S25_HEATMAP_COLUMNS,
        "preferred_species_tag": ctx.args.preferred_species_tag,
        "normalization": "gene-wise zscore",
        "linkage": "average/euclidean",
        "cluster_order": "dendrogram",
        "n_clusters": int(ctx.args.s25_n_clusters),
        "classifier": _main_classifier_settings(ctx),
        "canonical_trajectory": (
            {
                "path": str(canonical_index.resolve()),
                "sha256": _sha256(canonical_index),
                "source_stage": "s22",
                "source_stage_signature": upstream_s22_manifest.get("signature"),
            }
            if upstream_s22_manifest is not None
            else None
        ),
        "missing_target_policy": (
            "smoke_only_first_8_cells_with_manifest_warning"
            if ctx.args.profile == "smoke"
            else "strict_error"
        ),
    }

    def action(stage_dir: Path):
        if canonical_index.exists():
            states, canonical_times, canonical_sources = _read_state_bundle(
                canonical_index.parent,
                annotation_key=ctx.args.annotation_key,
            )
            if canonical_times != list(HALF_TIMES):
                raise RuntimeError(
                    "S22 canonical pre-warp trajectory does not match the S25 grid."
                )
            invalid_sources = {
                time_value: source
                for time_value, source in canonical_sources.items()
                if source
                not in {"observed_seed_predicted_labels", "generated_prewarp"}
            }
            if invalid_sources:
                raise RuntimeError(
                    f"Unexpected canonical trajectory sources: {invalid_sources}"
                )
            assert upstream_s22_manifest is not None
            s22_details = upstream_s22_manifest.get("details", {})
            classifier_cache_path = s22_details.get("classifier_cache_path")
            classifier_accuracy = s22_details.get("classifier_accuracy")
            classifier_balanced_accuracy = s22_details.get(
                "classifier_balanced_accuracy"
            )
            simulation_seeds = s22_details.get("simulation_seeds", {})
            trajectory_source = "s22_canonical_prewarp"
        else:
            result = _run_interpolation(
                ctx,
                output_dir=stage_dir / "workflow",
                time_points=HALF_TIMES,
                use_real_for_observed=False,
                display_piecewise_warp=False,
            )
            states = result.adata_dict
            classifier_cache_path = result.classifier_cache_path
            classifier_accuracy = result.classifier_accuracy
            classifier_balanced_accuracy = result.classifier_balanced_accuracy
            simulation_seeds = result.simulation_seeds
            trajectory_source = "stage_local_global_simulation"
        outputs = _write_state_bundle(
            states,
            HALF_TIMES,
            stage_dir / "generated_states",
            annotation_key=ctx.args.annotation_key,
            source_by_time={
                float(value): (
                    "observed_seed_predicted_labels"
                    if np.isclose(value, 0.0, rtol=0.0, atol=1e-9)
                    else "generated_prewarp"
                )
                for value in HALF_TIMES
            },
        )
        observed_states = _observed_state_dict(ctx)
        cached_classifier = cb.tl.load_cached_mlp_classifier(
            str(classifier_cache_path), device=ctx.args.device
        )
        analysis_states: dict[str, ad.AnnData] = {}
        analysis_source_by_time: dict[float, str] = {}
        for time_value in HALF_TIMES:
            key = str(float(time_value))
            if float(time_value) in observed_states:
                analysis_states[key] = observed_states[float(time_value)]
                analysis_source_by_time[float(time_value)] = (
                    "observed_actual_annotation"
                )
                continue
            points = np.asarray(states[key].X, dtype=np.float32)
            labels = cb.tl.predict_labels_for_points(
                points=points,
                time_value=float(time_value),
                model=cached_classifier.model,
                label_encoder=cached_classifier.label_encoder,
                feature_dim=int(_main_classifier_settings(ctx)["n_joint_features"]),
                device=ctx.args.device,
                knn_neighbors=int(ctx.args.s25_classifier_knn_neighbors),
                include_time_feature=cached_classifier.include_time_feature,
            )
            analysis_states[key] = _minimal_state_adata(
                points, labels, annotation_key=ctx.args.annotation_key
            )
            analysis_source_by_time[float(time_value)] = (
                "generated_prewarp_classifier_knn_"
                f"{int(ctx.args.s25_classifier_knn_neighbors)}"
            )
        outputs.extend(
            _write_state_bundle(
                analysis_states,
                HALF_TIMES,
                stage_dir / "hybrid_analysis_states",
                annotation_key=ctx.args.annotation_key,
                source_by_time=analysis_source_by_time,
            )
        )
        ysl_states: dict[str, ad.AnnData] = {}
        ysl_counts: dict[str, int] = {}
        smoke_fallbacks: list[dict[str, object]] = []
        for time_value in HALF_TIMES:
            key = str(float(time_value))
            state = analysis_states[key]
            mask = (
                state.obs[ctx.args.annotation_key].astype(str).to_numpy()
                == str(ctx.args.ysl_label)
            )
            if not np.any(mask):
                if ctx.args.profile != "smoke":
                    raise ValueError(
                        f"No predicted {ctx.args.ysl_label!r} cells at t={time_value}; "
                        "cannot reproduce the S25 cell-type-specific heatmap"
                    )
                # A two-epoch smoke classifier is deliberately too small to
                # guarantee every rare label.  Select a deterministic handful
                # only so CI/smoke runs can exercise inverse PCA, clustering,
                # and plotting.  Full scientific runs remain strict above.
                fallback_n = min(8, int(state.n_obs))
                if fallback_n == 0:
                    raise ValueError(
                        f"Generated state at t={time_value} is empty; "
                        "the S25 smoke mechanics cannot be exercised"
                    )
                mask = np.zeros(state.n_obs, dtype=bool)
                mask[:fallback_n] = True
                smoke_fallbacks.append(
                    {
                        "time": float(time_value),
                        "selected_first_n_cells": fallback_n,
                        "reason": f"no predicted {ctx.args.ysl_label!r} cells",
                    }
                )
            ysl_states[key] = state[mask].copy()
            ysl_counts[key] = int(np.sum(mask))

        temporal = cb.tl.summarize_temporal_gene_patterns(
            ysl_states,
            ctx.adata,
            time_points=HALF_TIMES,
            spatial_dim=2,
            loadings_key="PCs",
            reference_layer=None,
            n_top_genes=top_n,
            n_cluster_genes=top_n,
            n_clusters=int(ctx.args.s25_n_clusters),
            preferred_species_tag=ctx.args.preferred_species_tag,
            profile_normalization="zscore",
            profile_linkage_method="average",
            profile_cluster_order="dendrogram",
        )
        tables = {
            "mean_expression.csv": temporal.expression,
            "top_variable_genes.csv": temporal.top_variable_genes,
            "normalized_profiles.csv": temporal.clustering.normalized_profiles,
            "cluster_assignments.csv": temporal.clustering.assignments,
            "cluster_prototypes.csv": temporal.clustering.prototypes,
            "cluster_diagnostics.csv": temporal.clustering.diagnostics,
            "gene_name_map.csv": temporal.gene_name_map,
        }
        for filename, table in tables.items():
            path = stage_dir / filename
            index = filename in {"mean_expression.csv", "normalized_profiles.csv"}
            table.to_csv(path, index=index)
            outputs.append(path)
        heatmap_pdf = stage_dir / "S25_YSL_top250_temporal_variance_heatmap.pdf"
        heatmap_png = stage_dir / "S25_YSL_top250_temporal_variance_heatmap.png"
        cb.pl.plot_temporal_gene_heatmap(
            temporal.expression,
            temporal.top_variable_genes,
            out_path=heatmap_pdf,
            top_n=top_n,
            title=f"{ctx.args.ysl_label}: top {top_n} temporal-variance genes",
            panel_columns=S25_HEATMAP_COLUMNS,
        )
        cb.pl.plot_temporal_gene_heatmap(
            temporal.expression,
            temporal.top_variable_genes,
            out_path=heatmap_png,
            top_n=top_n,
            title=f"{ctx.args.ysl_label}: top {top_n} temporal-variance genes",
            panel_columns=S25_HEATMAP_COLUMNS,
        )
        outputs.extend([heatmap_pdf, heatmap_png])
        settings_path = stage_dir / "temporal_settings.json"
        settings_path.write_text(
            json.dumps(_json_ready(temporal.settings), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        outputs.append(settings_path)
        return outputs, {
            "classifier_cache_path": classifier_cache_path,
            "classifier_accuracy": classifier_accuracy,
            "classifier_balanced_accuracy": classifier_balanced_accuracy,
            "trajectory_source": trajectory_source,
            "generated_cell_counts": {
                str(float(t)): int(states[str(float(t))].n_obs)
                for t in HALF_TIMES
            },
            "analysis_cell_counts": {
                str(float(t)): int(analysis_states[str(float(t))].n_obs)
                for t in HALF_TIMES
            },
            "analysis_state_sources": analysis_source_by_time,
            "ysl_cell_counts": ysl_counts,
            "simulation_seeds": simulation_seeds,
            "smoke_only_target_fallbacks": smoke_fallbacks,
            "scientific_use_warning": (
                "Smoke-only fallback cells are not YSL and these outputs must not be "
                "used scientifically."
                if smoke_fallbacks
                else None
            ),
        }

    return _execute_stage(ctx, "s25", settings, action)


def _observed_state_dict(ctx: RunContext) -> dict[float, ad.AnnData]:
    joint = _joint_features(
        ctx.adata, latent_key=ctx.args.latent_key, spatial_key=ctx.args.spatial_key
    )
    raw_times = np.asarray(
        [cb.tl.parse_time_value(value) for value in ctx.adata.obs[ctx.args.time_key]],
        dtype=np.float64,
    )
    labels = ctx.adata.obs[ctx.args.annotation_key].astype(str).to_numpy()
    result: dict[float, ad.AnnData] = {}
    for time_value in OBSERVED_TIMES:
        mask = np.isclose(raw_times, time_value, rtol=0.0, atol=1e-9)
        if not np.any(mask):
            raise ValueError(f"No observed cells at canonical time {time_value}")
        result[time_value] = _minimal_state_adata(
            joint[mask], labels[mask], annotation_key=ctx.args.annotation_key
        )
    return result


def _plot_communication_heatmaps(
    communications: Mapping[str, Mapping[str, object]],
    output_dir: Path,
    *,
    selected_times: Sequence[float],
    display_min: float,
) -> list[Path]:
    import matplotlib.pyplot as plt
    import seaborn as sns

    outputs = []
    for time_value in selected_times:
        record = communications[str(float(time_value))]
        values = np.asarray(record["M_per_source"], dtype=float).copy()
        if float(display_min) > 0:
            values[values < float(display_min)] = 0.0
        labels = [str(value) for value in np.asarray(record["types"]).tolist()]
        figure, axis = plt.subplots(figsize=(8.2, 7.2), facecolor="white")
        sns.heatmap(
            values,
            xticklabels=labels,
            yticklabels=labels,
            cmap="PuBu",
            square=True,
            ax=axis,
            cbar_kws={"label": "Attention per source cell"},
        )
        axis.set_title(f"Cell-type communication, t={time_value:g}")
        axis.set_xlabel("Receiver")
        axis.set_ylabel("Sender")
        figure.tight_layout()
        path = output_dir / f"communication_t{time_value:g}.pdf"
        figure.savefig(path, bbox_inches="tight", facecolor="white")
        plt.close(figure)
        outputs.append(path)
    return outputs


def _select_lr_pair(table: pd.DataFrame, requested: str) -> str:
    pairs = table["pair"].astype(str)
    matches = pairs.loc[pairs.str.casefold() == str(requested).casefold()].unique()
    if len(matches) == 1:
        return str(matches[0])
    ligand, _, receptor = str(requested).partition("_")
    nearby = sorted(
        {
            pair
            for pair in pairs.unique()
            if ligand.casefold() in pair.casefold()
            or receptor.casefold() in pair.casefold()
        }
    )[:30]
    raise KeyError(
        f"LR pair {requested!r} was not scored. Related scored pairs: {nearby}"
    )


def _plot_target_lr(
    pair_timecourse: pd.DataFrame,
    celltype_timecourse: pd.DataFrame,
    *,
    pair: str,
    output_dir: Path,
) -> list[Path]:
    import matplotlib.pyplot as plt
    import seaborn as sns

    pair_table = pair_timecourse.loc[pair_timecourse["pair"] == pair].copy()
    cell_table = celltype_timecourse.loc[celltype_timecourse["pair"] == pair].copy()
    pair_path = output_dir / f"{pair}_timecourse.csv"
    cell_path = output_dir / f"{pair}_celltype_timecourse.csv"
    pair_table.to_csv(pair_path, index=False)
    cell_table.to_csv(cell_path, index=False)

    figure, axes = plt.subplots(1, 2, figsize=(13.0, 4.8), facecolor="white")
    ordered = pair_table.sort_values("time")
    axes[0].plot(
        ordered["time"], ordered["score"], marker="o", linewidth=2.2, color="#2b6cb0"
    )
    axes[0].set_title(f"{pair}: total LR communication")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("LR score")
    axes[0].grid(axis="y", alpha=0.2)

    incoming = cell_table.pivot_table(
        index="cell_type", columns="time", values="incoming", aggfunc="sum", fill_value=0.0
    )
    incoming = incoming.loc[incoming.max(axis=1).sort_values(ascending=False).index]
    sns.heatmap(
        incoming,
        cmap="magma",
        ax=axes[1],
        cbar_kws={"label": "Incoming LR score"},
    )
    axes[1].set_title(f"{pair}: receiving cell types")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Cell type")
    figure.tight_layout()
    figure_path = output_dir / f"{pair}_communication_panel.pdf"
    figure.savefig(figure_path, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return [pair_path, cell_path, figure_path]


def _stage_communication(ctx: RunContext) -> dict[str, object]:
    if ctx.args.lr_database is None:
        raise ValueError("--lr-database is required for the communication stage")
    lr_database = _require_file(ctx.args.lr_database, "zebrafish LR database")
    max_cells = (
        int(ctx.args.smoke_n_samples)
        if ctx.args.profile == "smoke"
        else ctx.args.communication_max_cells
    )
    s25_state_index = _require_file(
        ctx.output_dir / "s25" / "generated_states" / "index.json",
        "S25 no-warp generated-state index",
    )
    upstream_s25_manifest = _require_current_stage_manifest(ctx, "s25")
    settings = {
        "time_points": list(HALF_TIMES),
        "states": "observed integer frames + generated no-warp half-time frames",
        "attention_matrix": "M_per_source",
        "remove_self_loop": False,
        "winsor_quantile": float(ctx.args.communication_winsor_quantile),
        "max_cells_per_timepoint": max_cells,
        "display_min_only": float(ctx.args.communication_display_min),
        "lr_database": str(lr_database),
        "lr_database_sha256": _sha256(lr_database),
        "target_pair": ctx.args.lr_pair,
        "s25_state_index": str(s25_state_index),
        "s25_state_index_sha256": _sha256(s25_state_index),
        "s25_stage_signature": upstream_s25_manifest.get("signature"),
        "expression_space": (
            "log-space pseudobulk: expm1(cell-type mean log1p normalized "
            "expression); not mean raw counts"
        ),
        "observed_expression": (
            "real aligned_adata.X (log1p) at integer times; generated inverse-PCA "
            "log1p state at half times"
        ),
        "complex_mode": "min",
        "preferred_species_tag": ctx.args.preferred_species_tag,
        "lr_n_clusters": int(ctx.args.lr_n_clusters),
        "random_seed": int(ctx.args.random_seed),
    }

    def action(stage_dir: Path):
        generated, generated_times, _ = _read_state_bundle(
            ctx.output_dir / "s25" / "generated_states",
            annotation_key=ctx.args.annotation_key,
        )
        if generated_times != list(HALF_TIMES):
            raise RuntimeError(
                "S25 generated-state time grid does not match the communication grid; "
                "rerun --stage s25,communication"
            )
        observed = _observed_state_dict(ctx)
        hybrid: dict[str, ad.AnnData] = {}
        source_by_time: dict[float, str] = {}
        for time_value in HALF_TIMES:
            if time_value in observed:
                hybrid[str(float(time_value))] = observed[time_value]
                source_by_time[float(time_value)] = "observed"
            else:
                hybrid[str(float(time_value))] = generated[str(float(time_value))]
                source_by_time[float(time_value)] = "generated_prewarp"
        outputs = _write_state_bundle(
            hybrid,
            HALF_TIMES,
            stage_dir / "hybrid_states",
            annotation_key=ctx.args.annotation_key,
            source_by_time=source_by_time,
        )
        attention_dir = stage_dir / "attention"
        communication_pickle = stage_dir / "communications.pkl"
        communications = cb.tl.compute_timepoint_communications(
            adata_dict=hybrid,
            time_points=HALF_TIMES,
            annotation_key=ctx.args.annotation_key,
            f_net=ctx.runtime.f_net,
            device=ctx.args.device,
            out_dir=str(attention_dir),
            save_dense_attention_matrix=False,
            remove_self_loop=False,
            winsor_quantile=float(ctx.args.communication_winsor_quantile),
            save_pickle_path=str(communication_pickle),
            max_cells_per_timepoint=max_cells,
            random_seed=int(ctx.args.random_seed),
        )
        outputs.extend([communication_pickle, *sorted(attention_dir.glob("*"))])
        outputs.extend(
            _plot_communication_heatmaps(
                communications,
                stage_dir,
                selected_times=(0.0, 2.0, 4.0),
                display_min=float(ctx.args.communication_display_min),
            )
        )

        lr = cb.tl.project_communication_to_lr_timecourses(
            hybrid,
            ctx.adata,
            communications,
            lr_database,
            time_points=HALF_TIMES,
            annotation_key=ctx.args.annotation_key,
            matrix_key="M_per_source",
            spatial_dim=2,
            loadings_key="PCs",
            reference_layer=None,
            expression_space="count",
            complex_mode="min",
            require_all_subunits=False,
            duplicate_policy="first",
            preferred_species_tag=ctx.args.preferred_species_tag,
            n_clusters=int(ctx.args.lr_n_clusters),
            profile_linkage_method="average",
            profile_cluster_order="dendrogram",
            observed_adata=ctx.adata,
            observed_time_key=ctx.args.time_key,
            observed_time_points=OBSERVED_TIMES,
            observed_annotation_key=ctx.args.annotation_key,
            observed_layer=None,
            observed_expression_space="log1p",
            observed_missing_time_policy="error",
        )
        tables = {
            "lr_pair_timecourse.csv": lr.pair_timecourse,
            "lr_celltype_timecourse.csv": lr.celltype_timecourse,
            "lr_pattern_summary.csv": lr.pattern_summary,
            "lr_coverage.csv": lr.coverage,
            "lr_normalized_profiles.csv": lr.clustering.normalized_profiles,
            "lr_cluster_assignments.csv": lr.clustering.assignments,
            "lr_cluster_prototypes.csv": lr.clustering.prototypes,
            "lr_cluster_diagnostics.csv": lr.clustering.diagnostics,
        }
        for filename, table in tables.items():
            path = stage_dir / filename
            table.to_csv(
                path,
                index=filename in {"lr_normalized_profiles.csv"},
            )
            outputs.append(path)
        target_pair = _select_lr_pair(lr.pair_timecourse, ctx.args.lr_pair)
        outputs.extend(
            _plot_target_lr(
                lr.pair_timecourse,
                lr.celltype_timecourse,
                pair=target_pair,
                output_dir=stage_dir,
            )
        )
        settings_path = stage_dir / "lr_settings.json"
        settings_path.write_text(
            json.dumps(_json_ready(lr.settings), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        outputs.append(settings_path)
        return outputs, {
            "target_pair_resolved": target_pair,
            "n_lr_pairs_scored": int(lr.pair_timecourse["pair"].nunique()),
            "state_sources": source_by_time,
            "coverage": lr.coverage.to_dict(orient="records"),
        }

    return _execute_stage(ctx, "communication", settings, action)


def _build_context(args: argparse.Namespace) -> RunContext:
    args.aligned_h5ad = _require_file(args.aligned_h5ad, "aligned zebrafish H5AD")
    args.model_dir = _require_dir(args.model_dir, "native six-stage model directory")
    args.output_dir = Path(args.output_dir).expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    adata = sc.read_h5ad(args.aligned_h5ad)
    if args.annotation_key not in adata.obs:
        if "bin_annotation" in adata.obs:
            adata.obs[args.annotation_key] = adata.obs["bin_annotation"].astype(str)
        else:
            raise KeyError(
                f"Aligned H5AD lacks obs['{args.annotation_key}'] and 'bin_annotation'"
            )
    if args.time_key not in adata.obs:
        raise KeyError(f"Aligned H5AD lacks obs['{args.time_key}']")
    joint = _joint_features(
        adata, latent_key=args.latent_key, spatial_key=args.spatial_key
    )
    df, resolved_time_key = cb.tl.adata_to_aligned_dataframe(
        adata,
        time_key=args.time_key,
        obsm_key=args.latent_key,
        spatial_key=args.spatial_key,
        concat_spatial=True,
        annotation_key=args.annotation_key,
    )
    if resolved_time_key != args.time_key:
        raise RuntimeError(
            f"Requested time key {args.time_key!r} resolved as {resolved_time_key!r}"
        )
    observed = sorted(float(value) for value in df["samples"].unique())
    if observed != list(OBSERVED_TIMES):
        raise ValueError(
            f"Expected clean zebrafish model times {list(OBSERVED_TIMES)}, got {observed}"
        )
    loaded = cb.tl.load_dynamical_model_from_dir(
        args.model_dir, dim=int(joint.shape[1]), device=args.device
    )
    loaded.model.eval()
    runtime = cb.tl.build_dynamical_runtime(loaded)
    model_config_path = _require_file(
        args.model_dir / "config.yaml", "resolved model config"
    )
    resolved_interaction_config = (
        loaded.config.get("model", {}).get("interaction_net", {})
    )
    raw_edge_predictor = resolved_interaction_config.get("edge_predictor_path")
    edge_predictor_path = None
    if raw_edge_predictor:
        candidate = Path(str(raw_edge_predictor)).expanduser()
        raw_edge_root = resolved_interaction_config.get("edge_predictor_root")
        if not candidate.is_absolute() and raw_edge_root is not None:
            candidate = Path(str(raw_edge_root)).expanduser() / candidate
        edge_predictor_path = (
            candidate.resolve()
            if candidate.is_absolute()
            else (Path.cwd() / candidate).resolve()
        )
    if args.shared_cache_dir is None:
        # .../RUN/preprocess/zebrafish_aligned.h5ad -> .../RUN/shared_downstream_cache
        shared_cache_dir = args.aligned_h5ad.parent.parent / "shared_downstream_cache"
    else:
        shared_cache_dir = Path(args.shared_cache_dir).expanduser().resolve()
    shared_cache_dir.mkdir(parents=True, exist_ok=True)
    label_to_color = _label_colors(
        adata,
        annotation_key=args.annotation_key,
        color_key=args.color_key,
    )
    signature_sources = (
        Path(__file__).resolve(),
        *(REPO_ROOT / path for path in IMPLEMENTATION_SOURCE_RELATIVE_PATHS),
    )
    common_signature = {
        "aligned_h5ad": str(args.aligned_h5ad),
        "aligned_h5ad_sha256": _sha256(args.aligned_h5ad),
        "model_dir": str(args.model_dir),
        "weight_stage": loaded.weight_stage,
        "weight_path": str(loaded.weight_path),
        "weight_sha256": _sha256(loaded.weight_path),
        "score_stage": loaded.score_stage,
        "score_path": None if loaded.score_path is None else str(loaded.score_path),
        "score_sha256": (
            None if loaded.score_path is None else _sha256(loaded.score_path)
        ),
        "dim": int(joint.shape[1]),
        "spatial_dim": 2,
        "random_seed": int(args.random_seed),
        "device": str(args.device),
        "profile": str(args.profile),
        "shared_cache_dir": str(shared_cache_dir),
        "data_contract": {
            "annotation_key": str(args.annotation_key),
            "color_key": None if args.color_key is None else str(args.color_key),
            "time_key": str(args.time_key),
            "latent_key": str(args.latent_key),
            "spatial_key": str(args.spatial_key),
        },
        "model_config_path": str(model_config_path),
        "model_config_sha256": _sha256(model_config_path),
        "resolved_model_config_sha256": _stable_hash(loaded.config),
        "edge_predictor_path": (
            None if edge_predictor_path is None else str(edge_predictor_path)
        ),
        "edge_predictor_sha256": (
            None
            if edge_predictor_path is None
            else _sha256(_require_file(edge_predictor_path, "edge predictor"))
        ),
        "git": _git_revision(),
        "runner_path": str(Path(__file__).resolve()),
        "runner_sha256": _sha256(Path(__file__).resolve()),
        "implementation_sha256": {
            str(path.relative_to(REPO_ROOT)): _sha256(path)
            for path in signature_sources
        },
        "implementation_file_count": int(len(signature_sources)),
    }
    return RunContext(
        args=args,
        adata=adata,
        df=df,
        loaded=loaded,
        runtime=runtime,
        dim=int(joint.shape[1]),
        spatial_dim=2,
        output_dir=args.output_dir,
        shared_cache_dir=shared_cache_dir,
        label_to_color=label_to_color,
        common_signature=common_signature,
    )


def _write_root_manifest(
    ctx: RunContext,
    selected_stages: Sequence[str],
    stage_manifests: Mapping[str, Mapping[str, object]],
) -> Path:
    training_defaults = ctx.loaded.config.get("training", {}).get("defaults", {})
    interaction_cfg = ctx.loaded.config.get("model", {}).get("interaction_net", {})
    complete_manifests: dict[str, Mapping[str, object]] = {}
    for name in ALL_STAGES:
        try:
            manifest = _require_current_stage_manifest(ctx, name)
        except (OSError, ValueError, RuntimeError, json.JSONDecodeError):
            continue
        complete_manifests[name] = manifest
    complete_manifests.update(stage_manifests)
    payload = {
        "schema_version": 1,
        "workflow": "zebrafish_native_paper_downstream",
        "completed_at": _utc_now(),
        "selected_stages_this_invocation": list(selected_stages),
        "completed_stages": list(complete_manifests),
        "common": ctx.common_signature,
        "profile": ctx.args.profile,
        "shared_cache_dir": str(ctx.shared_cache_dir),
        "gpu_assignment": {
            "requested_device": ctx.args.device,
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "CYTOBRIDGE_ASSIGNED_GPU": os.environ.get("CYTOBRIDGE_ASSIGNED_GPU"),
        },
        "model": {
            "alpha_spatial": training_defaults.get("alpha_spatial"),
            "alpha_express": training_defaults.get("alpha_express"),
            "sigma": training_defaults.get("sigma"),
            "interaction_cutoff": interaction_cfg.get("cutoff"),
            "edge_predictor_path": interaction_cfg.get("edge_predictor_path"),
            "edge_predictor_threshold": interaction_cfg.get("edge_predictor_thre"),
        },
        "preprocess": {
            "n_obs": int(ctx.adata.n_obs),
            "n_vars": int(ctx.adata.n_vars),
            "time_counts": {
                str(key): int(value)
                for key, value in ctx.adata.obs[ctx.args.time_key]
                .value_counts()
                .sort_index()
                .items()
            },
            "preprocess_info": ctx.adata.uns.get("preprocess_info", {}),
            "pca_center_info": ctx.adata.uns.get("pca_center_info", {}),
        },
        "stage_manifests": {
            name: str(_stage_manifest_path(ctx, name)) for name in complete_manifests
        },
        "stage_signatures": {
            name: manifest.get("signature")
            for name, manifest in complete_manifests.items()
        },
    }
    path = ctx.output_dir / "run_manifest.json"
    path.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True), encoding="utf-8"
    )
    return path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--aligned-h5ad", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument(
        "--lr-database",
        type=Path,
        default=None,
        help="Required only for the communication stage.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--stage",
        default="all",
        help=(
            "'all' or a comma-separated subset of: " + ", ".join(ALL_STAGES)
        ),
    )
    parser.add_argument("--profile", choices=("full", "smoke"), default="full")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--shared-cache-dir", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--annotation-key", default="Annotation")
    parser.add_argument("--color-key", default="Color")
    parser.add_argument("--time-key", default="time_point_processed")
    parser.add_argument("--latent-key", default="X_latent")
    parser.add_argument("--spatial-key", default="spatial_aligned")

    parser.add_argument("--interaction-m", type=int, default=1024)
    parser.add_argument("--sde-dt", type=float, default=0.05)
    parser.add_argument("--sde-sigma", type=float, default=0.03)
    parser.add_argument(
        "--growth-alpha",
        type=float,
        default=1.0,
        help=(
            "Split-SDE growth multiplier; 1.0 is the recovered historical "
            "zebrafish workflow value."
        ),
    )
    parser.add_argument(
        "--sde-n-samples",
        type=int,
        default=None,
        help="Optional full-run t0 cap; default uses all t0 cells.",
    )
    parser.add_argument(
        "--sde-max-particles",
        type=int,
        default=100000,
        help=(
            "Fail-fast ceiling checked before split-event allocation. This is a "
            "safety guard, not a downsampling target."
        ),
    )
    parser.add_argument("--smoke-n-samples", type=int, default=64)
    parser.add_argument("--point-size", type=float, default=2.0)

    parser.add_argument("--classifier-epochs", type=int, default=500)
    parser.add_argument(
        "--s22-simulation-step",
        type=float,
        default=0.1,
        help=(
            "Output grid for the single full S22 simulation. Mosaic and video "
            "frames are selected from this trajectory; split/resampling events "
            "remain fixed by --sde-dt."
        ),
    )
    parser.add_argument("--video-step", type=float, default=0.1)
    parser.add_argument("--video-fps", type=int, default=10)
    parser.add_argument("--video-formats", default="gif,mp4")
    parser.add_argument("--velocity-neighbors", type=int, default=30)

    parser.add_argument("--ablation-step", type=float, default=0.05)
    parser.add_argument("--ablation-classifier-epochs", type=int, default=800)
    parser.add_argument("--ysl-label", default="Yolk Syncytial Layer")
    parser.add_argument("--evl-label", default="EVL")

    parser.add_argument("--s25-top-genes", type=int, default=250)
    parser.add_argument("--s25-n-clusters", type=int, default=4)
    parser.add_argument(
        "--s25-classifier-knn-neighbors",
        type=int,
        default=1,
        help=(
            "Spatial label-smoothing k used only to select the rare target cell "
            "type at generated S25 half-times. Default 1 keeps direct classifier "
            "labels; S22 display labels retain their separate k=10 policy."
        ),
    )
    parser.add_argument("--preferred-species-tag", default=None)

    parser.add_argument("--communication-max-cells", type=int, default=None)
    parser.add_argument("--communication-winsor-quantile", type=float, default=0.995)
    parser.add_argument(
        "--communication-display-min",
        type=float,
        default=0.0,
        help="Display-only cutoff for communication heatmaps; raw matrices are unchanged.",
    )
    parser.add_argument("--lr-pair", default="cxcl12a_cxcr4a")
    parser.add_argument("--lr-n-clusters", type=int, default=4)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    if int(args.sde_max_particles) <= 0:
        raise ValueError("--sde-max-particles must be > 0")
    if int(args.s25_classifier_knn_neighbors) <= 0:
        raise ValueError("--s25-classifier-knn-neighbors must be > 0")
    selected_stages = _parse_stages(args.stage)
    if "communication" in selected_stages and "s25" not in selected_stages:
        state_index = Path(args.output_dir).expanduser().resolve() / "s25" / "generated_states" / "index.json"
        if not state_index.exists():
            raise FileNotFoundError(
                "The communication stage requires S25 no-warp generated states. "
                "Run --stage s25,communication (or --stage all) first."
            )
    ctx = _build_context(args)
    runners = {
        "classifier": _stage_classifier,
        "velocity": _stage_velocity,
        "s22": _stage_s22,
        "growth": _stage_growth,
        "ablation": _stage_ablation,
        "s25": _stage_s25,
        "communication": _stage_communication,
    }
    manifests: dict[str, dict[str, object]] = {}
    for stage in selected_stages:
        manifests[stage] = runners[stage](ctx)
    root_manifest = _write_root_manifest(ctx, selected_stages, manifests)
    print(
        json.dumps(
            {
                "workflow": "zebrafish_native_paper_downstream",
                "output_dir": str(ctx.output_dir),
                "root_manifest": str(root_manifest),
                "stages": {
                    stage: str(_stage_manifest_path(ctx, stage))
                    for stage in selected_stages
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
