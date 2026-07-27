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
* Communication uses a hybrid no-warp state population: observed cells at
  integer times and generated cells at intermediate times. This state-source
  choice is separate from the LR expression measurement policy.

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
        raise ValueError(f"step={step} does not exactly tile interval [{start}, {end}]")
    return [
        float(round(float(start) + i * float(step), 10)) for i in range(n_steps + 1)
    ]


def _parse_stages(raw: str) -> list[str]:
    values = [value.strip().lower() for value in str(raw).split(",") if value.strip()]
    if not values:
        raise ValueError("--stage must not be empty")
    if "all" in values:
        if len(values) != 1:
            raise ValueError(
                "Use --stage all by itself, or provide a comma-separated subset"
            )
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
            colors = colors.loc[
                colors.str.strip().ne("") & colors.str.lower().ne("nan")
            ]
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
        labels = np.asarray(state.obs[annotation_key].astype(str).tolist(), dtype=str)
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
        print(
            f"[resume] {stage}: inputs/settings changed or outputs missing; rerunning"
        )

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
                    stage_dir / f"spatial_direct_{panel_label}_t{time_value:g}.pdf"
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
                    stage_dir / f"latent_to_spatial_{panel_label}_t{time_value:g}.pdf"
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
        if not any(
            np.isclose(value, obs, rtol=0.0, atol=1e-9) for obs in OBSERVED_TIMES
        )
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
        simulation_times = _time_grid(0.0, 4.0, float(ctx.args.s22_simulation_step))
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
            None if ctx.args.profile == "smoke" else float(ctx.args.s22_simulation_step)
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
        "generated_display_label_knn_neighbors": 10,
        "generated_display_label_policy": (
            "legacy spatial kNN smoothing used by S22 display outputs only"
        ),
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
                if any(
                    np.isclose(value, obs, rtol=0.0, atol=1e-9)
                    for obs in OBSERVED_TIMES
                )
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
                np.asarray(dense_result.predicted_labels_split_prewarp[index]).astype(
                    str
                ),
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
                        else "generated_prewarp_reclassified_by_communication_stage"
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
        "state_source": "observed_anchor_joint_state(spatial_aligned+X_latent)",
        "growth_semantics": "frozen_model_growth_head_per_cell_output",
        "normalization": "independent per-time 5th-95th percentile scaling",
        "cross_stage_display_amplitude_comparable": False,
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
            "colorbar_label": "model g (within-stage p5-p95 scaled)",
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
                title=(
                    "Model-predicted growth output at observed zebrafish stages\n"
                    "(each stage independently p5-p95 scaled)"
                ),
            )
            outputs.append(path)
        return outputs, {
            "n_panels": len(OBSERVED_TIMES),
            "n_growth_values": int(len(raw)),
            "state_source": "observed_anchor_joint_state(spatial_aligned+X_latent)",
            "cross_stage_display_amplitude_comparable": False,
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
    epochs = (
        2 if ctx.args.profile == "smoke" else int(ctx.args.ablation_classifier_epochs)
    )
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
            int(ctx.args.smoke_n_samples) if ctx.args.profile == "smoke" else None
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
                str(ctx.args.ysl_label): int(t0_counts.get(str(ctx.args.ysl_label), 0)),
                str(ctx.args.evl_label): int(t0_counts.get(str(ctx.args.evl_label), 0)),
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
                int(ctx.args.smoke_n_samples) if ctx.args.profile == "smoke" else None
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
        animation_points = {
            "baseline": result.baseline_points,
            **result.ablation_points,
        }
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
    external_bundle_arg = getattr(ctx.args, "s25_canonical_state_bundle", None)
    external_bundle = (
        None
        if external_bundle_arg is None
        else _require_dir(
            external_bundle_arg,
            "external S25 canonical pre-warp state bundle",
        )
    )
    canonical_index = (
        external_bundle / "index.json"
        if external_bundle is not None
        else ctx.output_dir / "s22" / "canonical_prewarp_states" / "index.json"
    )
    if external_bundle is not None:
        canonical_index = _require_file(
            canonical_index, "external S25 canonical state index"
        )
        upstream_s22_manifest = None
        external_manifest_path = external_bundle.parent / "stage_manifest.json"
        external_s22_manifest = (
            json.loads(external_manifest_path.read_text(encoding="utf-8"))
            if external_manifest_path.is_file()
            else None
        )
        if external_s22_manifest is not None:
            if external_s22_manifest.get("status") != "complete":
                raise RuntimeError(
                    "External S22 state bundle belongs to an incomplete stage: "
                    f"{external_manifest_path}"
                )
            if not _recorded_outputs_exist(external_s22_manifest):
                raise RuntimeError(
                    "External S22 outputs no longer match its stage manifest: "
                    f"{external_manifest_path}"
                )
            recorded_outputs = {
                str(Path(path).expanduser().resolve())
                for path in external_s22_manifest.get("outputs", [])
            }
            if str(canonical_index.resolve()) not in recorded_outputs:
                raise RuntimeError(
                    "External canonical state index is not recorded by its adjacent "
                    f"S22 manifest: {canonical_index}"
                )
    else:
        upstream_s22_manifest = (
            _require_current_stage_manifest(ctx, "s22")
            if canonical_index.exists()
            else None
        )
        external_s22_manifest = None
        external_manifest_path = None
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
        "target_classifier_knn_neighbors": int(ctx.args.s25_classifier_knn_neighbors),
        "target_classifier_policy": (
            "direct classifier labels at generated half-times; spatial kNN smoothing "
            "is configurable separately from the S22 display-label policy"
        ),
        "top_variable_genes": top_n,
        "gene_expression_space": (
            "per-cell rank-retained inverse-PCA processed log1p, clipped at zero "
            "before the mean for both observed and generated states"
        ),
        "signed_gene_diagnostic": (
            "unclipped inverse-PCA means and per-time negative-value diagnostics"
        ),
        "observed_gene_validation": (
            "exact observed integer-time YSL log1p means are exported separately; "
            "they are not mixed into the rank-50 temporal trajectory"
        ),
        "pca_center": "persisted fit-time adata.var['pca_center']",
        "pca_feature_policy": (
            "active retained-component loadings only; center-only genes excluded"
        ),
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
                "source_stage": (
                    "external_validated_s22_bundle"
                    if external_bundle is not None
                    else "s22"
                ),
                "source_stage_signature": (
                    external_s22_manifest.get("signature")
                    if external_s22_manifest is not None
                    else (
                        upstream_s22_manifest.get("signature")
                        if upstream_s22_manifest is not None
                        else None
                    )
                ),
                "source_stage_manifest": (
                    str(external_manifest_path.resolve())
                    if external_manifest_path is not None
                    and external_manifest_path.is_file()
                    else None
                ),
                "source_stage_manifest_sha256": (
                    _sha256(external_manifest_path)
                    if external_manifest_path is not None
                    and external_manifest_path.is_file()
                    else None
                ),
            }
            if canonical_index.exists()
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
                if source not in {"observed_seed_predicted_labels", "generated_prewarp"}
            }
            if invalid_sources:
                raise RuntimeError(
                    f"Unexpected canonical trajectory sources: {invalid_sources}"
                )
            source_manifest = (
                external_s22_manifest
                if external_s22_manifest is not None
                else upstream_s22_manifest
            )
            s22_details = (
                {} if source_manifest is None else source_manifest.get("details", {})
            )
            classifier_cache_path = s22_details.get("classifier_cache_path")
            classifier_accuracy = s22_details.get("classifier_accuracy")
            classifier_balanced_accuracy = s22_details.get(
                "classifier_balanced_accuracy"
            )
            simulation_seeds = s22_details.get("simulation_seeds", {})
            trajectory_source = (
                "external_validated_s22_canonical_prewarp"
                if external_bundle is not None
                else "s22_canonical_prewarp"
            )
            if classifier_cache_path is None:
                cached, cache_path = _train_main_classifier(ctx)
                classifier_cache_path = str(cache_path)
                classifier_accuracy = cached.accuracy
                classifier_balanced_accuracy = cached.balanced_accuracy
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
                analysis_source_by_time[
                    float(time_value)
                ] = "observed_actual_annotation"
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
            mask = state.obs[ctx.args.annotation_key].astype(str).to_numpy() == str(
                ctx.args.ysl_label
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
            active_features_only=True,
            clip_min=0.0,
        )
        reference_var_names = pd.Index(ctx.adata.var_names.astype(str))
        active_var_names = pd.Index(temporal.gene_name_map["var_name"].astype(str))
        if not reference_var_names.is_unique or not active_var_names.is_unique:
            raise ValueError(
                "S25 exact-observed validation requires unique reference feature names."
            )
        active_indexer = reference_var_names.get_indexer(active_var_names)
        if np.any(active_indexer < 0):
            missing = active_var_names[active_indexer < 0][:5].tolist()
            raise ValueError(
                "S25 active inverse-PCA features are missing from aligned expression; "
                f"examples={missing}."
            )
        observed_times = np.asarray(
            [
                cb.tl.parse_time_value(value)
                for value in ctx.adata.obs[ctx.args.time_key]
            ],
            dtype=np.float64,
        )
        observed_labels = ctx.adata.obs[ctx.args.annotation_key].astype(str).to_numpy()
        exact_columns: dict[float, np.ndarray] = {}
        validation_rows: list[dict[str, object]] = []
        top_gene_names = set(temporal.top_variable_genes["gene"].astype(str).tolist())
        top_mask = temporal.expression.index.astype(str).isin(top_gene_names)
        from scipy import sparse

        for time_value in OBSERVED_TIMES:
            observed_mask = np.isclose(
                observed_times, float(time_value), rtol=0.0, atol=1e-9
            ) & (observed_labels == str(ctx.args.ysl_label))
            if not observed_mask.any():
                raise ValueError(
                    f"No exact observed {ctx.args.ysl_label!r} cells at "
                    f"t={time_value} for S25 validation."
                )
            observed_matrix = ctx.adata.X[observed_mask][:, active_indexer]
            if sparse.issparse(observed_matrix):
                exact_mean = np.asarray(observed_matrix.mean(axis=0)).reshape(-1)
            else:
                exact_mean = np.asarray(observed_matrix, dtype=np.float64).mean(axis=0)
            exact_columns[float(time_value)] = exact_mean
            for decoder_name, decoded in (
                (
                    "signed_inverse_pca",
                    temporal.signed_expression[float(time_value)].to_numpy(
                        dtype=np.float64
                    ),
                ),
                (
                    "clipped_inverse_pca",
                    temporal.expression[float(time_value)].to_numpy(dtype=np.float64),
                ),
            ):
                for scope, scope_mask in (
                    ("all_active_features", np.ones(len(exact_mean), dtype=bool)),
                    ("top_temporal_variance", np.asarray(top_mask, dtype=bool)),
                ):
                    exact_values = exact_mean[scope_mask]
                    decoded_values = decoded[scope_mask]
                    delta = decoded_values - exact_values
                    correlation = (
                        float(np.corrcoef(exact_values, decoded_values)[0, 1])
                        if exact_values.size > 1
                        and np.std(exact_values) > 0
                        and np.std(decoded_values) > 0
                        else np.nan
                    )
                    validation_rows.append(
                        {
                            "time": float(time_value),
                            "decoder": decoder_name,
                            "scope": scope,
                            "n_cells": int(observed_mask.sum()),
                            "n_features": int(exact_values.size),
                            "rmse": float(np.sqrt(np.mean(delta**2))),
                            "mae": float(np.mean(np.abs(delta))),
                            "mean_bias": float(np.mean(delta)),
                            "pearson_r": correlation,
                        }
                    )
        exact_observed_expression = pd.DataFrame(
            exact_columns,
            index=temporal.expression.index.copy(),
        )
        observed_validation = pd.DataFrame(validation_rows)
        tables = {
            "mean_expression.csv": temporal.expression,
            "mean_inverse_pca_log1p_clipped.csv": temporal.expression,
            "mean_inverse_pca_log1p_signed.csv": temporal.signed_expression,
            "inverse_pca_reconstruction_diagnostics.csv": (
                temporal.reconstruction_diagnostics
            ),
            "observed_exact_log1p_anchors.csv": exact_observed_expression,
            "observed_vs_inverse_pca_metrics.csv": observed_validation,
            "top_variable_genes.csv": temporal.top_variable_genes,
            "normalized_profiles.csv": temporal.clustering.normalized_profiles,
            "cluster_assignments.csv": temporal.clustering.assignments,
            "cluster_prototypes.csv": temporal.clustering.prototypes,
            "cluster_diagnostics.csv": temporal.clustering.diagnostics,
            "gene_name_map.csv": temporal.gene_name_map,
        }
        for filename, table in tables.items():
            path = stage_dir / filename
            index = filename in {
                "mean_expression.csv",
                "mean_inverse_pca_log1p_clipped.csv",
                "mean_inverse_pca_log1p_signed.csv",
                "observed_exact_log1p_anchors.csv",
                "normalized_profiles.csv",
            }
            table.to_csv(path, index=index)
            outputs.append(path)
        import matplotlib.pyplot as plt

        figure, axes = plt.subplots(2, 3, figsize=(12.0, 7.6), facecolor="white")
        for axis, time_value in zip(axes.ravel(), OBSERVED_TIMES):
            exact_values = exact_observed_expression[float(time_value)].to_numpy(
                dtype=np.float64
            )
            decoded_values = temporal.expression[float(time_value)].to_numpy(
                dtype=np.float64
            )
            axis.scatter(
                exact_values,
                decoded_values,
                s=5,
                alpha=0.35,
                color="#4C78A8",
                linewidths=0,
            )
            axis.scatter(
                exact_values[np.asarray(top_mask, dtype=bool)],
                decoded_values[np.asarray(top_mask, dtype=bool)],
                s=7,
                alpha=0.55,
                color="#E45756",
                linewidths=0,
                label=f"top {top_n}",
            )
            lo = float(min(exact_values.min(), decoded_values.min()))
            hi = float(max(exact_values.max(), decoded_values.max()))
            axis.plot([lo, hi], [lo, hi], color="black", linewidth=0.8, alpha=0.7)
            metric_row = observed_validation.loc[
                (observed_validation["time"] == float(time_value))
                & (observed_validation["decoder"] == "clipped_inverse_pca")
                & (observed_validation["scope"] == "all_active_features")
            ].iloc[0]
            axis.set_title(
                f"t={time_value:g}: r={metric_row['pearson_r']:.3f}, "
                f"RMSE={metric_row['rmse']:.3f}"
            )
            axis.set_xlabel("exact observed mean log1p")
            axis.set_ylabel("clipped inverse-PCA mean")
        axes.ravel()[-1].axis("off")
        axes.ravel()[0].legend(frameon=False, loc="lower right")
        figure.suptitle(
            f"{ctx.args.ysl_label}: observed anchors vs rank-50 reconstruction"
        )
        figure.tight_layout()
        for suffix in ("pdf", "png"):
            validation_path = (
                stage_dir / f"S25_observed_vs_inverse_pca_validation.{suffix}"
            )
            figure.savefig(validation_path, dpi=300, bbox_inches="tight")
            outputs.append(validation_path)
        plt.close(figure)
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
                str(float(t)): int(states[str(float(t))].n_obs) for t in HALF_TIMES
            },
            "analysis_cell_counts": {
                str(float(t)): int(analysis_states[str(float(t))].n_obs)
                for t in HALF_TIMES
            },
            "analysis_state_sources": analysis_source_by_time,
            "ysl_cell_counts": ysl_counts,
            "gene_expression_contract": temporal.settings,
            "observed_inverse_pca_validation": validation_rows,
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


def _build_explicitly_labeled_hybrid_states(
    generated_states: Mapping[str, ad.AnnData],
    observed_states: Mapping[float, ad.AnnData],
    *,
    time_points: Sequence[float],
    annotation_key: str,
    cached_classifier: object,
    classifier_feature_dim: int,
    device: str,
    knn_neighbors: int,
) -> tuple[
    dict[str, ad.AnnData],
    dict[float, str],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Build an observed/generated series with an explicit label policy.

    Generated-state annotations stored in an upstream trajectory bundle are
    provenance, not an analysis contract: they may have been produced for a
    display-only workflow with a different spatial-smoothing value.  This
    helper therefore re-predicts every generated frame with the classifier and
    ``knn_neighbors`` requested by the consuming analysis.  Observed frames
    retain their experimental annotations.
    """

    knn_neighbors = int(knn_neighbors)
    if knn_neighbors <= 0:
        raise ValueError("knn_neighbors must be > 0")
    hybrid: dict[str, ad.AnnData] = {}
    source_by_time: dict[float, str] = {}
    assignment_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    count_rows: list[dict[str, object]] = []
    for raw_time in time_points:
        time_value = float(raw_time)
        key = str(time_value)
        if time_value in observed_states:
            state = observed_states[time_value]
            labels = state.obs[annotation_key].astype(str).to_numpy()
            hybrid[key] = state
            source_by_time[time_value] = "observed_actual_annotation"
            summary_rows.append(
                {
                    "time": time_value,
                    "state_source": "observed",
                    "n_cells": int(state.n_obs),
                    "n_labels_changed": 0,
                    "fraction_labels_changed": 0.0,
                    "knn_neighbors": np.nan,
                    "points_preserved_exactly": True,
                }
            )
            for label, count in pd.Series(labels).value_counts().sort_index().items():
                count_rows.append(
                    {
                        "time": time_value,
                        "state_source": "observed",
                        "label_policy": "actual_annotation",
                        "label": str(label),
                        "n_cells": int(count),
                    }
                )
            continue
        if key not in generated_states:
            raise KeyError(f"Generated state dictionary is missing time key {key!r}")
        inherited_state = generated_states[key]
        if annotation_key not in inherited_state.obs:
            raise KeyError(
                f"Generated state {key!r} is missing annotation {annotation_key!r}"
            )
        points = np.asarray(inherited_state.X, dtype=np.float32)
        inherited_labels = inherited_state.obs[annotation_key].astype(str).to_numpy()
        analysis_labels = np.asarray(
            cb.tl.predict_labels_for_points(
                points=points,
                time_value=time_value,
                model=cached_classifier.model,
                label_encoder=cached_classifier.label_encoder,
                feature_dim=int(classifier_feature_dim),
                device=device,
                knn_neighbors=knn_neighbors,
                include_time_feature=cached_classifier.include_time_feature,
            )
        ).astype(str)
        if analysis_labels.shape != inherited_labels.shape:
            raise RuntimeError(
                "Classifier returned a different number of generated labels: "
                f"time={time_value}, expected={len(inherited_labels)}, "
                f"actual={len(analysis_labels)}"
            )
        changed = analysis_labels != inherited_labels
        hybrid[key] = _minimal_state_adata(
            points, analysis_labels, annotation_key=annotation_key
        )
        if not np.array_equal(np.asarray(hybrid[key].X), points):
            raise RuntimeError(
                f"Generated points changed while relabeling time {time_value}."
            )
        source_by_time[time_value] = f"generated_prewarp_classifier_knn_{knn_neighbors}"
        summary_rows.append(
            {
                "time": time_value,
                "state_source": "generated_prewarp",
                "n_cells": int(len(analysis_labels)),
                "n_labels_changed": int(changed.sum()),
                "fraction_labels_changed": float(changed.mean()),
                "knn_neighbors": knn_neighbors,
                "points_preserved_exactly": True,
            }
        )
        assignment_rows.extend(
            {
                "time": time_value,
                "row_index": int(row_index),
                "inherited_label": str(inherited),
                "analysis_label": str(analysis),
                "changed": bool(is_changed),
                "knn_neighbors": knn_neighbors,
            }
            for row_index, (inherited, analysis, is_changed) in enumerate(
                zip(inherited_labels, analysis_labels, changed)
            )
        )
        for policy, labels in (
            ("inherited_bundle", inherited_labels),
            (f"classifier_knn_{knn_neighbors}", analysis_labels),
        ):
            for label, count in pd.Series(labels).value_counts().sort_index().items():
                count_rows.append(
                    {
                        "time": time_value,
                        "state_source": "generated_prewarp",
                        "label_policy": policy,
                        "label": str(label),
                        "n_cells": int(count),
                    }
                )
    return (
        hybrid,
        source_by_time,
        pd.DataFrame(assignment_rows),
        pd.DataFrame(summary_rows),
        pd.DataFrame(count_rows),
    )


def _compare_lr_measurement_contracts(
    hybrid_result: object,
    all_inverse_result: object,
    *,
    observed_times: Sequence[float],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame,]:
    """Compare exact-observed and all-inverse-PCA LR measurements.

    The legacy hybrid trajectory uses exact expression at observed integer
    times and inverse-PCA expression at generated half-times. The corrected
    default uses inverse-PCA expression throughout. This diagnostic holds
    states, labels, and communication matrices fixed and changes only that
    expression measurement operator.
    """

    observed = {float(value) for value in observed_times}
    hybrid_settings = getattr(hybrid_result, "settings", None)
    inverse_settings = getattr(all_inverse_result, "settings", None)
    if isinstance(hybrid_settings, Mapping) and isinstance(inverse_settings, Mapping):
        for contract_key in (
            "feature_universe",
            "pca_feature_coverage",
            "global_lr_coverage",
            "pca_center_source",
        ):
            if hybrid_settings.get(contract_key) != inverse_settings.get(contract_key):
                raise RuntimeError(
                    "Hybrid and all-inverse LR projections used different "
                    f"{contract_key!r} contracts."
                )
    pair = hybrid_result.pair_timecourse.merge(
        all_inverse_result.pair_timecourse,
        on=["pair", "time"],
        how="outer",
        suffixes=("_hybrid", "_all_inverse_pca"),
        indicator="pair_presence",
        validate="one_to_one",
    )
    pair["measurement_source_hybrid"] = np.where(
        pair["time"].astype(float).isin(observed),
        "exact_observed_expression",
        "inverse_pca_expression",
    )
    pair["score_delta_all_inverse_minus_hybrid"] = (
        pair["score_all_inverse_pca"] - pair["score_hybrid"]
    )
    pair["score_abs_delta"] = pair["score_delta_all_inverse_minus_hybrid"].abs()
    pair["score_ratio_all_inverse_over_hybrid"] = np.where(
        pair["score_hybrid"] > 0,
        pair["score_all_inverse_pca"] / pair["score_hybrid"],
        np.nan,
    )
    pair["score_log2_ratio_all_inverse_over_hybrid"] = np.where(
        (pair["score_hybrid"] > 0) & (pair["score_all_inverse_pca"] > 0),
        np.log2(pair["score_all_inverse_pca"] / pair["score_hybrid"]),
        np.nan,
    )
    symmetric_denominator = (
        pair["score_hybrid"].abs() + pair["score_all_inverse_pca"].abs()
    )
    pair["score_symmetric_relative_error"] = np.where(
        symmetric_denominator > 0,
        2.0 * pair["score_abs_delta"] / symmetric_denominator,
        0.0,
    )
    pair["zero_status"] = np.select(
        (
            pair["pair_presence"].eq("left_only"),
            pair["pair_presence"].eq("right_only"),
            pair["score_hybrid"].eq(0) & pair["score_all_inverse_pca"].eq(0),
            pair["score_hybrid"].eq(0) & pair["score_all_inverse_pca"].gt(0),
            pair["score_hybrid"].gt(0) & pair["score_all_inverse_pca"].eq(0),
        ),
        (
            "all_inverse_missing",
            "hybrid_missing",
            "both_zero",
            "hybrid_only_zero",
            "all_inverse_only_zero",
        ),
        default="both_nonzero",
    )

    celltype = hybrid_result.celltype_timecourse.merge(
        all_inverse_result.celltype_timecourse,
        on=["pair", "time", "cell_type"],
        how="outer",
        suffixes=("_hybrid", "_all_inverse_pca"),
        indicator="celltype_presence",
        validate="one_to_one",
    )
    celltype["measurement_source_hybrid"] = np.where(
        celltype["time"].astype(float).isin(observed),
        "exact_observed_expression",
        "inverse_pca_expression",
    )
    for column in ("incoming", "outgoing", "total"):
        celltype[f"{column}_delta_all_inverse_minus_hybrid"] = (
            celltype[f"{column}_all_inverse_pca"] - celltype[f"{column}_hybrid"]
        )
    common_celltypes = celltype["celltype_presence"].eq("both")
    if not np.array_equal(
        celltype.loc[common_celltypes, "n_cells_hybrid"].to_numpy(),
        celltype.loc[common_celltypes, "n_cells_all_inverse_pca"].to_numpy(),
    ):
        raise RuntimeError(
            "LR measurement comparison changed cell counts even though states and "
            "labels were held fixed."
        )

    metric_rows: list[dict[str, object]] = []
    for time_value, subset in pair.groupby("time", sort=True):
        common = subset["pair_presence"].eq("both")
        hybrid_values = subset.loc[common, "score_hybrid"].to_numpy(dtype=float)
        inverse_values = subset.loc[common, "score_all_inverse_pca"].to_numpy(
            dtype=float
        )
        delta = inverse_values - hybrid_values
        hybrid_sum = float(np.sum(hybrid_values))
        inverse_sum = float(np.sum(inverse_values))
        rmse = float(np.sqrt(np.mean(delta**2))) if len(delta) else np.nan
        hybrid_rms = (
            float(np.sqrt(np.mean(hybrid_values**2))) if len(delta) else np.nan
        )
        if len(delta) > 1 and np.std(hybrid_values) > 0 and np.std(inverse_values) > 0:
            pearson_r = float(np.corrcoef(hybrid_values, inverse_values)[0, 1])
            spearman_r = float(
                pd.Series(hybrid_values).corr(
                    pd.Series(inverse_values), method="spearman"
                )
            )
        else:
            pearson_r = np.nan
            spearman_r = np.nan
        metric_rows.append(
            {
                "time": float(time_value),
                "measurement_source_hybrid": (
                    "exact_observed_expression"
                    if float(time_value) in observed
                    else "inverse_pca_expression"
                ),
                "n_pairs_common": int(common.sum()),
                "n_pairs_hybrid_only": int(
                    subset["pair_presence"].eq("left_only").sum()
                ),
                "n_pairs_all_inverse_only": int(
                    subset["pair_presence"].eq("right_only").sum()
                ),
                "rmse": (rmse),
                "relative_rmse": (
                    float(rmse / hybrid_rms)
                    if len(delta) and hybrid_rms > 0
                    else np.nan
                ),
                "mae": float(np.mean(np.abs(delta))) if len(delta) else np.nan,
                "mean_bias_all_inverse_minus_hybrid": (
                    float(np.mean(delta)) if len(delta) else np.nan
                ),
                "max_abs_delta": (
                    float(np.max(np.abs(delta))) if len(delta) else np.nan
                ),
                "relative_l1": (
                    float(np.sum(np.abs(delta)) / np.sum(np.abs(hybrid_values)))
                    if len(delta) and np.sum(np.abs(hybrid_values)) > 0
                    else np.nan
                ),
                "hybrid_score_sum": hybrid_sum,
                "all_inverse_pca_score_sum": inverse_sum,
                "score_sum_ratio_all_inverse_over_hybrid": (
                    float(inverse_sum / hybrid_sum) if hybrid_sum > 0 else np.nan
                ),
                "n_zero_mismatches": int(
                    (np.equal(hybrid_values, 0) != np.equal(inverse_values, 0)).sum()
                ),
                "median_abs_log2_ratio": (
                    float(
                        np.median(
                            np.abs(
                                np.log2(
                                    inverse_values[
                                        (hybrid_values > 0) & (inverse_values > 0)
                                    ]
                                    / hybrid_values[
                                        (hybrid_values > 0) & (inverse_values > 0)
                                    ]
                                )
                            )
                        )
                    )
                    if np.any((hybrid_values > 0) & (inverse_values > 0))
                    else np.nan
                ),
                "top10_pair_overlap": (
                    int(
                        len(
                            set(
                                subset.loc[common].nlargest(
                                    min(10, int(common.sum())), "score_hybrid"
                                )["pair"]
                            )
                            & set(
                                subset.loc[common].nlargest(
                                    min(10, int(common.sum())),
                                    "score_all_inverse_pca",
                                )["pair"]
                            )
                        )
                    )
                    if common.any()
                    else 0
                ),
                "pearson_r": pearson_r,
                "spearman_r": spearman_r,
            }
        )
    metrics = pd.DataFrame(metric_rows)
    generated_metrics = metrics.loc[
        metrics["measurement_source_hybrid"].eq("inverse_pca_expression")
    ]
    if (generated_metrics["max_abs_delta"].fillna(0.0) > 1e-12).any():
        raise RuntimeError(
            "Common hybrid/all-inverse LR pairs disagree at generated times, where "
            "their expression contracts must be identical."
        )

    continuity_rows: list[dict[str, object]] = []
    available_times = sorted(pair["time"].dropna().astype(float).unique())
    generated_times = [value for value in available_times if value not in observed]
    for anchor_time in sorted(observed):
        left_candidates = [value for value in generated_times if value < anchor_time]
        right_candidates = [value for value in generated_times if value > anchor_time]
        left_time = max(left_candidates) if left_candidates else None
        right_time = min(right_candidates) if right_candidates else None
        if left_time is not None and right_time is not None:
            neighbor_mode = "two_sided_linear"
        elif left_time is not None:
            neighbor_mode = "one_sided_left"
        elif right_time is not None:
            neighbor_mode = "one_sided_right"
        else:
            continue
        anchor_rows = pair.loc[
            np.isclose(pair["time"].astype(float), anchor_time)
            & pair["pair_presence"].eq("both")
        ]
        for anchor in anchor_rows.itertuples(index=False):
            neighbor_scores: dict[str, float] = {}
            for side, neighbor_time in (("left", left_time), ("right", right_time)):
                if neighbor_time is None:
                    continue
                match = pair.loc[
                    pair["pair"].eq(anchor.pair)
                    & np.isclose(pair["time"].astype(float), neighbor_time)
                    & pair["pair_presence"].eq("both")
                ]
                if len(match) == 1:
                    neighbor_scores[side] = float(match.iloc[0]["score_hybrid"])
            if left_time is not None and right_time is not None:
                if set(neighbor_scores) != {"left", "right"}:
                    continue
                right_weight = (anchor_time - left_time) / (right_time - left_time)
                reference_score = (1.0 - right_weight) * neighbor_scores[
                    "left"
                ] + right_weight * neighbor_scores["right"]
            elif "left" in neighbor_scores:
                reference_score = neighbor_scores["left"]
            elif "right" in neighbor_scores:
                reference_score = neighbor_scores["right"]
            else:
                continue
            exact_score = float(anchor.score_hybrid)
            decoded_score = float(anchor.score_all_inverse_pca)
            exact_residual = exact_score - reference_score
            decoded_residual = decoded_score - reference_score
            exact_slope_jump = np.nan
            decoded_slope_jump = np.nan
            if set(neighbor_scores) == {"left", "right"}:
                exact_slope_jump = (neighbor_scores["right"] - exact_score) / (
                    right_time - anchor_time
                ) - (exact_score - neighbor_scores["left"]) / (anchor_time - left_time)
                decoded_slope_jump = (neighbor_scores["right"] - decoded_score) / (
                    right_time - anchor_time
                ) - (decoded_score - neighbor_scores["left"]) / (
                    anchor_time - left_time
                )
            continuity_rows.append(
                {
                    "pair": str(anchor.pair),
                    "anchor_time": anchor_time,
                    "neighbor_mode": neighbor_mode,
                    "left_generated_time": left_time,
                    "right_generated_time": right_time,
                    "neighbor_reference_score": reference_score,
                    "exact_observed_anchor_score": exact_score,
                    "inverse_pca_anchor_score": decoded_score,
                    "exact_anchor_residual": exact_residual,
                    "inverse_pca_anchor_residual": decoded_residual,
                    "abs_residual_exact_minus_inverse": (
                        abs(exact_residual) - abs(decoded_residual)
                    ),
                    "exact_anchor_slope_jump": exact_slope_jump,
                    "inverse_pca_anchor_slope_jump": decoded_slope_jump,
                }
            )
    continuity = pd.DataFrame(continuity_rows)
    continuity_metric_rows: list[dict[str, object]] = []
    if not continuity.empty:
        for anchor_time, subset in continuity.groupby("anchor_time", sort=True):
            continuity_metric_rows.append(
                {
                    "anchor_time": float(anchor_time),
                    "neighbor_mode": str(subset["neighbor_mode"].iloc[0]),
                    "n_pairs": int(len(subset)),
                    "mean_abs_residual_exact": float(
                        subset["exact_anchor_residual"].abs().mean()
                    ),
                    "mean_abs_residual_inverse_pca": float(
                        subset["inverse_pca_anchor_residual"].abs().mean()
                    ),
                    "median_abs_residual_exact": float(
                        subset["exact_anchor_residual"].abs().median()
                    ),
                    "median_abs_residual_inverse_pca": float(
                        subset["inverse_pca_anchor_residual"].abs().median()
                    ),
                    "fraction_inverse_pca_closer_to_generated_neighbors": float(
                        (subset["abs_residual_exact_minus_inverse"] > 0).mean()
                    ),
                    "mean_abs_slope_jump_exact": float(
                        subset["exact_anchor_slope_jump"].abs().mean()
                    ),
                    "mean_abs_slope_jump_inverse_pca": float(
                        subset["inverse_pca_anchor_slope_jump"].abs().mean()
                    ),
                }
            )
    continuity_metrics = pd.DataFrame(continuity_metric_rows)
    return pair, celltype, metrics, continuity, continuity_metrics


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
        index="cell_type",
        columns="time",
        values="incoming",
        aggfunc="sum",
        fill_value=0.0,
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
    classifier_cache_value = upstream_s25_manifest.get("details", {}).get(
        "classifier_cache_path"
    )
    if classifier_cache_value is None:
        raise RuntimeError(
            "The current S25 manifest does not record classifier_cache_path; "
            "rerun --stage s25,communication with the current runner."
        )
    classifier_cache = _require_file(
        classifier_cache_value, "S25 trajectory classifier cache"
    )
    communication_knn = int(ctx.args.communication_classifier_knn_neighbors)
    lr_expression_policy = str(ctx.args.lr_expression_time_policy)
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
        "generated_label_classifier": {
            "policy": (
                "explicitly re-predict every generated pre-warp frame; never "
                "inherit display labels from the state bundle"
            ),
            "knn_neighbors": communication_knn,
            "default_semantics": (
                "k=1 is direct MLP prediction; k=10 is available only as an "
                "explicit legacy manuscript-parity sensitivity"
            ),
            "cache_path": str(classifier_cache),
            "cache_sha256": _sha256(classifier_cache),
            "observed_label_policy": "actual annotation",
        },
        "expression_space": (
            "arithmetic cell-type mean after per-cell conversion to normalized "
            "count-like abundance; not raw counts"
        ),
        "primary_lr_expression_time_policy": lr_expression_policy,
        "primary_lr_expression_contract": (
            "all times use per-cell clipped expm1(inverse-PCA log1p), including "
            "observed integer-state PCA coordinates"
            if lr_expression_policy == "all_inverse_pca"
            else "legacy hybrid: exact observed expression at integer times and "
            "inverse-PCA expression at generated half-times"
        ),
        "lr_expression_validation": (
            "both all-inverse-PCA and hybrid exact-observed projections are always "
            "exported; their difference isolates the measurement operator"
        ),
        "lr_feature_policy": (
            "one active-loading PCA feature universe across all observed/generated "
            "times; center-only subunits excluded globally"
        ),
        "complex_mode": str(ctx.args.lr_complex_mode),
        "complex_require_all_subunits": True,
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
        cached_classifier = cb.tl.load_cached_mlp_classifier(
            str(classifier_cache), device=ctx.args.device
        )
        expected_feature_dim = int(_main_classifier_settings(ctx)["n_joint_features"])
        if int(cached_classifier.feature_dim) != expected_feature_dim:
            raise RuntimeError(
                "Classifier feature contract differs from the zebrafish joint-state "
                f"contract: cache={cached_classifier.feature_dim}, "
                f"expected={expected_feature_dim}."
            )
        if not bool(cached_classifier.include_time_feature):
            raise RuntimeError("Communication classifier cache must include time.")
        if str(cached_classifier.label_col) != str(ctx.args.annotation_key):
            raise RuntimeError(
                "Communication classifier annotation mismatch: "
                f"cache={cached_classifier.label_col!r}, "
                f"requested={ctx.args.annotation_key!r}."
            )
        (
            hybrid,
            source_by_time,
            label_assignments,
            label_summary,
            label_counts,
        ) = _build_explicitly_labeled_hybrid_states(
            generated,
            observed,
            time_points=HALF_TIMES,
            annotation_key=ctx.args.annotation_key,
            cached_classifier=cached_classifier,
            classifier_feature_dim=int(cached_classifier.feature_dim),
            device=ctx.args.device,
            knn_neighbors=communication_knn,
        )
        outputs = _write_state_bundle(
            hybrid,
            HALF_TIMES,
            stage_dir / "hybrid_states",
            annotation_key=ctx.args.annotation_key,
            source_by_time=source_by_time,
        )
        for filename, table in (
            ("generated_label_assignments.csv", label_assignments),
            ("generated_label_reclassification_summary.csv", label_summary),
            ("communication_label_counts.csv", label_counts),
        ):
            path = stage_dir / filename
            table.to_csv(path, index=False)
            outputs.append(path)
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

        lr_common = {
            "time_points": HALF_TIMES,
            "annotation_key": ctx.args.annotation_key,
            "matrix_key": "M_per_source",
            "spatial_dim": 2,
            "loadings_key": "PCs",
            "reference_layer": None,
            "expression_space": "count",
            "complex_mode": str(ctx.args.lr_complex_mode),
            "require_all_subunits": True,
            "duplicate_policy": "first",
            "preferred_species_tag": ctx.args.preferred_species_tag,
            "n_clusters": int(ctx.args.lr_n_clusters),
            "profile_linkage_method": "average",
            "profile_cluster_order": "dendrogram",
        }
        hybrid_lr = cb.tl.project_communication_to_lr_timecourses(
            hybrid,
            ctx.adata,
            communications,
            lr_database,
            **lr_common,
            observed_adata=ctx.adata,
            observed_time_key=ctx.args.time_key,
            observed_time_points=OBSERVED_TIMES,
            observed_annotation_key=ctx.args.annotation_key,
            observed_layer=None,
            observed_expression_space="log1p",
            observed_missing_time_policy="error",
        )
        all_inverse_lr = cb.tl.project_communication_to_lr_timecourses(
            hybrid,
            ctx.adata,
            communications,
            lr_database,
            **lr_common,
        )
        primary_lr = (
            all_inverse_lr if lr_expression_policy == "all_inverse_pca" else hybrid_lr
        )
        (
            lr_measurement_pair,
            lr_measurement_celltype,
            lr_measurement_metrics,
            lr_anchor_continuity,
            lr_anchor_continuity_metrics,
        ) = _compare_lr_measurement_contracts(
            hybrid_lr,
            all_inverse_lr,
            observed_times=OBSERVED_TIMES,
        )
        tables = {
            "lr_pair_timecourse.csv": primary_lr.pair_timecourse,
            "lr_celltype_timecourse.csv": primary_lr.celltype_timecourse,
            "lr_pattern_summary.csv": primary_lr.pattern_summary,
            "lr_coverage.csv": primary_lr.coverage,
            "lr_normalized_profiles.csv": primary_lr.clustering.normalized_profiles,
            "lr_cluster_assignments.csv": primary_lr.clustering.assignments,
            "lr_cluster_prototypes.csv": primary_lr.clustering.prototypes,
            "lr_cluster_diagnostics.csv": primary_lr.clustering.diagnostics,
            "lr_all_inverse_pca_pair_timecourse.csv": (all_inverse_lr.pair_timecourse),
            "lr_all_inverse_pca_celltype_timecourse.csv": (
                all_inverse_lr.celltype_timecourse
            ),
            "lr_all_inverse_pca_pattern_summary.csv": all_inverse_lr.pattern_summary,
            "lr_all_inverse_pca_cluster_assignments.csv": (
                all_inverse_lr.clustering.assignments
            ),
            "lr_hybrid_exact_observed_pair_timecourse.csv": (hybrid_lr.pair_timecourse),
            "lr_hybrid_exact_observed_celltype_timecourse.csv": (
                hybrid_lr.celltype_timecourse
            ),
            "lr_hybrid_exact_observed_pattern_summary.csv": hybrid_lr.pattern_summary,
            "lr_hybrid_exact_observed_cluster_assignments.csv": (
                hybrid_lr.clustering.assignments
            ),
            "lr_hybrid_vs_all_inverse_pair_scores.csv": lr_measurement_pair,
            "lr_hybrid_vs_all_inverse_celltype_scores.csv": (lr_measurement_celltype),
            "lr_observed_vs_inverse_pca_metrics.csv": lr_measurement_metrics,
            "lr_anchor_source_switch_diagnostics.csv": lr_anchor_continuity,
            "lr_anchor_source_switch_metrics.csv": lr_anchor_continuity_metrics,
        }
        for filename, table in tables.items():
            path = stage_dir / filename
            table.to_csv(
                path,
                index=filename in {"lr_normalized_profiles.csv"},
            )
            outputs.append(path)
        target_pair = _select_lr_pair(primary_lr.pair_timecourse, ctx.args.lr_pair)
        outputs.extend(
            _plot_target_lr(
                primary_lr.pair_timecourse,
                primary_lr.celltype_timecourse,
                pair=target_pair,
                output_dir=stage_dir,
            )
        )
        settings_path = stage_dir / "lr_settings.json"
        settings_path.write_text(
            json.dumps(_json_ready(primary_lr.settings), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        outputs.append(settings_path)
        all_inverse_settings_path = stage_dir / "lr_all_inverse_pca_settings.json"
        all_inverse_settings_path.write_text(
            json.dumps(_json_ready(all_inverse_lr.settings), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        outputs.append(all_inverse_settings_path)
        hybrid_settings_path = stage_dir / "lr_hybrid_exact_observed_settings.json"
        hybrid_settings_path.write_text(
            json.dumps(_json_ready(hybrid_lr.settings), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        outputs.append(hybrid_settings_path)
        return outputs, {
            "target_pair_resolved": target_pair,
            "primary_lr_expression_time_policy": lr_expression_policy,
            "n_lr_pairs_scored": int(primary_lr.pair_timecourse["pair"].nunique()),
            "state_sources": source_by_time,
            "coverage": primary_lr.coverage.to_dict(orient="records"),
            "generated_label_classifier": {
                "knn_neighbors": communication_knn,
                "cache_path": str(classifier_cache),
                "cache_sha256": _sha256(classifier_cache),
                "feature_dim": int(cached_classifier.feature_dim),
                "feature_cols": list(cached_classifier.feature_cols),
                "label_col": str(cached_classifier.label_col),
                "include_time_feature": bool(cached_classifier.include_time_feature),
                "source_fingerprint": cached_classifier.metadata.get("source", {}).get(
                    "fingerprint"
                ),
            },
            "generated_label_reclassification": label_summary.to_dict(orient="records"),
            "communication_label_counts": label_counts.to_dict(orient="records"),
            "lr_measurement_contract_validation": (
                lr_measurement_metrics.to_dict(orient="records")
            ),
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
    resolved_interaction_config = loaded.config.get("model", {}).get(
        "interaction_net", {}
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
        help=("'all' or a comma-separated subset of: " + ", ".join(ALL_STAGES)),
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
        "--s25-canonical-state-bundle",
        type=Path,
        default=None,
        help=(
            "Optional existing canonical_prewarp_states directory. The index and "
            "frame hashes are validated, and an adjacent complete S22 manifest is "
            "validated when present. Use this for downstream-only recomputation "
            "without silently simulating a different trajectory."
        ),
    )
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
    parser.add_argument(
        "--communication-classifier-knn-neighbors",
        type=int,
        default=1,
        help=(
            "Generated-frame label policy for communication/LR. Default 1 is "
            "direct MLP prediction; pass 10 explicitly for legacy manuscript "
            "label-smoothing sensitivity. Observed annotations are unchanged."
        ),
    )
    parser.add_argument("--communication-winsor-quantile", type=float, default=0.995)
    parser.add_argument(
        "--communication-display-min",
        type=float,
        default=0.0,
        help="Display-only cutoff for communication heatmaps; raw matrices are unchanged.",
    )
    parser.add_argument("--lr-pair", default="cxcl12a_cxcr4a")
    parser.add_argument("--lr-n-clusters", type=int, default=4)
    parser.add_argument(
        "--lr-complex-mode",
        choices=("min", "geometric_mean", "mean", "product"),
        default="min",
        help=(
            "Aggregation across subunits of a heteromeric ligand or receptor. "
            "Use geometric_mean for the reviewer sensitivity run; min preserves "
            "the manuscript AND-gate definition."
        ),
    )
    parser.add_argument(
        "--lr-expression-time-policy",
        choices=("all_inverse_pca", "hybrid_exact_observed"),
        default="all_inverse_pca",
        help=(
            "Expression measurement used by the primary LR time course. The "
            "default all_inverse_pca applies one comparable decoder at every time; "
            "hybrid_exact_observed is retained only for legacy manuscript parity. "
            "Both projections are exported regardless of this selection."
        ),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    if int(args.sde_max_particles) <= 0:
        raise ValueError("--sde-max-particles must be > 0")
    if int(args.s25_classifier_knn_neighbors) <= 0:
        raise ValueError("--s25-classifier-knn-neighbors must be > 0")
    if int(args.communication_classifier_knn_neighbors) <= 0:
        raise ValueError("--communication-classifier-knn-neighbors must be > 0")
    selected_stages = _parse_stages(args.stage)
    if "communication" in selected_stages and "s25" not in selected_stages:
        state_index = (
            Path(args.output_dir).expanduser().resolve()
            / "s25"
            / "generated_states"
            / "index.json"
        )
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
