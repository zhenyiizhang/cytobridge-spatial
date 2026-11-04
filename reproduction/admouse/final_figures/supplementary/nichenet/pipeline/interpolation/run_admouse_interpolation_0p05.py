#!/usr/bin/env python3
"""Run the formal AdMouse r2 baseline rollout on a 0.05 output-time grid.

The model, observed anchors, classifier, seed, split-SDE dt, sigma, growth,
interaction grouping, and observed-slice replacement follow the original
AdMouse workflow.  The intended scientific change is the output grid only:
0.0--2.5 inclusive at 0.05 spacing (51 nodes instead of 26).
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch

import CytoBridge as cb


FINAL_ROOT = Path(__file__).resolve().parents[4]
DATA_ROOT = FINAL_ROOT.parent
ROOT = DATA_ROOT / "nichenet" / "new_interpolation"
RESULT = ROOT / "results" / "interpolation"
SLICE_DIR = RESULT / "slice_data"
ALIGNED_H5AD = DATA_ROOT / "admouse_0815/refs/training/adata.h5ad"
MODEL_DIR = DATA_ROOT / "admouse_0815/refs/training"
CLASSIFIER = (
    DATA_ROOT
    / "admouse_0815/refs/accepted_downstream/classifier_cache"
    / "classifier_resmlp_46ee959d0b1f14db.pt"
)

OBSERVED_TIMES = [0.0, 1.0, 2.0]
TIME_GRID = np.round(np.arange(0.0, 2.5 + 0.025, 0.05), 2).tolist()
INTERPOLATED_TIMES = [t for t in TIME_GRID if t not in OBSERVED_TIMES]

TIME_KEY = "time_point_processed"
ANNOTATION_KEY = "major_annotation"
LATENT_KEY = "X_latent"
SPATIAL_KEY = "spatial_aligned"
CONCAT_SPATIAL = True
SEED = 42
K_NEIGHBORS = 1
SDE_DT = 0.05
SPLIT_SDE_DT = 0.01
SPLIT_SIGMA = 0.03
SPLIT_GROWTH_ALPHA = 1.0
INTERACTION_M = 1024
DEVICE = "cuda:0"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def token(value: float) -> str:
    return f"{float(value):.2f}".replace(".", "p")


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the formal AdMouse rollout.")
    RESULT.mkdir(parents=True, exist_ok=True)
    SLICE_DIR.mkdir(parents=True, exist_ok=True)

    adata = ad.read_h5ad(ALIGNED_H5AD)
    observed = sorted(
        pd.to_numeric(adata.obs[TIME_KEY], errors="raise").unique().astype(float)
    )
    if observed != OBSERVED_TIMES:
        raise RuntimeError(f"Observed time mismatch: {observed} != {OBSERVED_TIMES}")

    aligned_table, _ = cb.tl.adata_to_aligned_dataframe(
        adata,
        time_key=TIME_KEY,
        obsm_key=LATENT_KEY,
        spatial_key=SPATIAL_KEY,
        concat_spatial=CONCAT_SPATIAL,
        annotation_key=ANNOTATION_KEY,
    )
    model_dim = int(adata.obsm[SPATIAL_KEY].shape[1]) + int(
        adata.obsm[LATENT_KEY].shape[1]
    )
    loaded = cb.tl.load_dynamical_model_from_dir(
        MODEL_DIR, dim=model_dim, device=DEVICE
    )
    runtime = cb.tl.build_dynamical_runtime(loaded)

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    started = time.perf_counter()
    trajectory = cb.tl.run_interpolation_workflow(
        df=aligned_table,
        dim=model_dim,
        annotation_key=ANNOTATION_KEY,
        runtime=runtime,
        device=DEVICE,
        output_dir=str(RESULT),
        requested_plot_points=TIME_GRID,
        interp_time_points=INTERPOLATED_TIMES,
        max_observed_timepoints=len(OBSERVED_TIMES),
        use_real_for_observed=True,
        classifier_cache_path=str(CLASSIFIER),
        classifier_adata=adata,
        classifier_time_key=TIME_KEY,
        classifier_obsm_key=LATENT_KEY,
        classifier_spatial_key=SPATIAL_KEY,
        classifier_concat_spatial=CONCAT_SPATIAL,
        classifier_epochs=500,
        classifier_hidden_size=128,
        classifier_lr=0.001,
        classifier_test_size=0.1,
        classifier_best_metric="bacc",
        classifier_strict_stratification=True,
        classifier_knn_neighbors=K_NEIGHBORS,
        sde_n_samples=None,
        skip_nonsplit_sde=True,
        sde_dt=SDE_DT,
        split_sde_dt=SPLIT_SDE_DT,
        split_sigma_scalar=SPLIT_SIGMA,
        split_daughter_noise_std=0.0,
        split_growth_alpha=SPLIT_GROWTH_ALPHA,
        split_interaction_m=INTERACTION_M,
        split_resample_dt=None,
        split_max_particles=None,
        split_sde_piecewise=False,
        spatial_warp_to_observed=False,
        random_seed=SEED,
    )
    elapsed = time.perf_counter() - started

    rows = []
    for t, key in zip(trajectory.ts_points, trajectory.time_keys):
        slice_adata = trajectory.adata_dict[key]
        path = SLICE_DIR / f"time_{token(t)}.h5ad"
        slice_adata.write_h5ad(path)
        labels = slice_adata.obs[ANNOTATION_KEY].astype(str).to_numpy()
        rows.append(
            {
                "time": float(t),
                "time_token": token(t),
                "n_particles": int(slice_adata.n_obs),
                "n_microglia": int(np.sum(labels == "Microglia")),
                "slice_origin": str(slice_adata.uns["slice_origin"]),
                "source_anchor_time": float(slice_adata.uns["source_anchor_time"]),
                "slice_file": str(path),
            }
        )
    pd.DataFrame(rows).to_csv(RESULT / "timepoint_manifest.csv", index=False)

    protocol = {
        "status": "complete",
        "time_grid_change_only": {
            "original_interval": 0.1,
            "new_interval": 0.05,
            "original_nodes": 26,
            "new_nodes": len(TIME_GRID),
        },
        "time_grid": TIME_GRID,
        "observed_times": OBSERVED_TIMES,
        "trajectory_mode": "global_t0_extrapolation_with_real_observed_slice_replacement",
        "split_event_note": (
            "With split_resample_dt=None, the package applies a split event at each "
            "requested output node; this is a new numerical protocol, not post-hoc plotting interpolation."
        ),
        "model_dir": str(MODEL_DIR),
        "model_checkpoint": str(loaded.weight_path),
        "model_checkpoint_sha256": sha256(Path(loaded.weight_path)),
        "score_checkpoint": str(loaded.score_path),
        "score_checkpoint_sha256": sha256(Path(loaded.score_path)),
        "classifier": str(CLASSIFIER),
        "classifier_sha256": sha256(CLASSIFIER),
        "classifier_accuracy": trajectory.classifier_accuracy,
        "classifier_balanced_accuracy": trajectory.classifier_balanced_accuracy,
        "classifier_k": K_NEIGHBORS,
        "split_sde_dt": SPLIT_SDE_DT,
        "split_sigma": SPLIT_SIGMA,
        "split_growth_alpha": SPLIT_GROWTH_ALPHA,
        "split_daughter_noise_std": 0.0,
        "interaction_group_size": INTERACTION_M,
        "seed": SEED,
        "elapsed_seconds": elapsed,
    }
    (RESULT / "run_protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )
    print(json.dumps(protocol, indent=2))


if __name__ == "__main__":
    main()
