#!/usr/bin/env python3
"""Focused immutable MOSTA Fig. 4a global-t0 40k/50k sensitivity run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import torch

import CytoBridge as cb


TIMES = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
GENERATED = [0.5, 1.5, 2.5]
OBSERVED = [0.0, 1.0, 2.0, 3.0]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_time(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--aligned-h5ad", required=True)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--classifier-cache-path", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--package-commit", required=True)
    p.add_argument("--expected-input-sha256", required=True)
    p.add_argument("--n-samples", type=int, required=True, choices=(40000, 50000))
    p.add_argument("--seed", type=int, default=42)
    return p


def main() -> int:
    args = parser().parse_args()
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "slice_data").mkdir()
    (out / "provenance").mkdir()
    started = time.time()

    if int(args.n_samples) not in (40000, 50000):
        raise ValueError("This focused sensitivity is locked to n_samples=40000 or 50000")
    aligned = Path(args.aligned_h5ad).resolve()
    model_dir = Path(args.model_dir).resolve()
    classifier_cache = Path(args.classifier_cache_path).resolve()
    for path in (aligned, model_dir, classifier_cache):
        if not path.exists():
            raise FileNotFoundError(path)

    adata = ad.read_h5ad(aligned)
    df, resolved_time_key = cb.tl.adata_to_aligned_dataframe(
        adata,
        time_key="time_point_processed",
        obsm_key="X_latent",
        spatial_key="spatial_aligned",
        concat_spatial=True,
        annotation_key="Annotation",
    )
    feature_cols = cb.tl.infer_feature_columns(df, annotation_column="Annotation")
    if len(feature_cols) != 52:
        raise ValueError(f"Expected 52 model features, found {len(feature_cols)}")
    observed_from_data = sorted(map(float, df["samples"].unique()))
    if observed_from_data != OBSERVED:
        raise ValueError(f"Observed time mismatch: {observed_from_data}")

    loaded = cb.tl.load_dynamical_model_from_dir(
        model_dir,
        dim=len(feature_cols),
        device=args.device,
    )
    runtime = cb.tl.build_dynamical_runtime(loaded)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    result = cb.tl.run_interpolation_workflow(
        df=df,
        dim=len(feature_cols),
        annotation_key="Annotation",
        runtime=runtime,
        device=args.device,
        output_dir=str(out / "workflow"),
        requested_plot_points=TIMES,
        interp_time_points=GENERATED,
        max_observed_timepoints=4,
        use_real_for_observed=True,
        classifier_cache_path=str(classifier_cache),
        classifier_cache_dir=str(out / "classifier_cache_unused"),
        classifier_adata=adata,
        classifier_time_key=resolved_time_key,
        classifier_obsm_key="X_latent",
        classifier_spatial_key="spatial_aligned",
        classifier_concat_spatial=True,
        classifier_epochs=500,
        classifier_hidden_size=128,
        classifier_lr=0.001,
        classifier_best_metric="bacc",
        classifier_strict_stratification=True,
        classifier_knn_neighbors=10,
        sde_n_samples=int(args.n_samples),
        skip_nonsplit_sde=True,
        sde_dt=0.05,
        split_sde_dt=0.05,
        split_sigma_scalar=0.03,
        split_daughter_noise_std=0.0,
        split_growth_alpha=1.0,
        split_interaction_m=1024,
        split_resample_dt=None,
        split_max_particles=None,
        split_sde_piecewise=False,
        split_sde_piecewise_include_end=False,
        piecewise_observed_sample_mode="t0_fixed",
        spatial_warp_to_observed=False,
        spatial_warp_to_observed_piecewise=False,
        spatial_warp_visualization_only=False,
        slice_max_cells_per_timepoint=None,
        random_seed=int(args.seed),
    )

    if list(map(float, result.ts_points)) != TIMES:
        raise ValueError(f"Result times mismatch: {result.ts_points}")
    snapshot_rows = []
    generated_counts = {}
    for time_value in TIMES:
        a = result.adata_dict[str(float(time_value))]
        expected_origin = "observed_real" if time_value in OBSERVED else "generated_global_t0"
        expected_anchor = time_value if time_value in OBSERVED else 0.0
        if str(a.uns.get("slice_origin")) != expected_origin:
            raise ValueError(f"Origin mismatch at t={time_value}: {a.uns.get('slice_origin')}")
        if float(a.uns.get("source_anchor_time")) != expected_anchor:
            raise ValueError(f"Anchor mismatch at t={time_value}: {a.uns.get('source_anchor_time')}")
        state = np.asarray(a.X)
        spatial = np.asarray(a.obsm["spatial"])
        if not np.isfinite(state).all() or not np.array_equal(state[:, :2], spatial):
            raise ValueError(f"State/spatial integrity failure at t={time_value}")
        path = out / "slice_data" / f"time_{safe_time(time_value)}.h5ad"
        a.write_h5ad(path)
        row = {
            "time": time_value,
            "origin": expected_origin,
            "source_anchor_time": expected_anchor,
            "n": int(a.n_obs),
            "n_labels": int(a.obs["Annotation"].astype(str).nunique()),
            "path": str(path),
            "sha256": sha256(path),
        }
        snapshot_rows.append(row)
        if expected_origin == "generated_global_t0":
            generated_counts[str(time_value)] = int(a.n_obs)

    expected_loaded_contract = {
        "weight_stage": "Finetune",
        "score_stage": "Score_Refine",
    }
    actual_loaded_contract = {
        "weight_stage": getattr(loaded, "weight_stage", None),
        "score_stage": getattr(loaded, "score_stage", None),
    }
    if actual_loaded_contract != expected_loaded_contract:
        raise ValueError(f"Loaded checkpoint stage mismatch: {actual_loaded_contract}")

    summary = {
        "schema_version": 1,
        "status": "complete",
        "purpose": "MOSTA manuscript Fig. 4a initial-particle-cap sensitivity",
        "dataset": "mosta",
        "trajectory_mode": "global_t0_extrapolation",
        "restart_from_preceding_observed_stage": False,
        "spatial_warp": False,
        "observed_times": OBSERVED,
        "generated_times": GENERATED,
        "n_samples": int(args.n_samples),
        "seed": int(args.seed),
        "split_population_seed": int(args.seed) + 1,
        "parameters": {
            "split_sde_dt": 0.05,
            "split_sigma": 0.03,
            "daughter_noise_std": 0.0,
            "growth_alpha": 1.0,
            "interaction_group_size": 1024,
            "classifier_k": 10,
        },
        "classifier": {
            "cache_path": str(classifier_cache),
            "cache_sha256": sha256(classifier_cache),
            "accuracy": result.classifier_accuracy,
            "balanced_accuracy": result.classifier_balanced_accuracy,
        },
        "model": {
            "directory": str(model_dir),
            **actual_loaded_contract,
        },
        "aligned_h5ad": {
            "path": str(aligned),
            "expected_sha256": args.expected_input_sha256,
            "shape": [int(adata.n_obs), int(adata.n_vars)],
        },
        "package": {
            "commit": args.package_commit,
            "module_path": str(Path(cb.__file__).resolve()),
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": args.device,
            "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "pid": os.getpid(),
            "cuda_peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None,
            "cuda_peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else None,
        },
        "simulation_seeds": result.simulation_seeds,
        "generated_counts": generated_counts,
        "snapshots": snapshot_rows,
        "wall_seconds": time.time() - started,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "COMPLETE").write_text("complete\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
