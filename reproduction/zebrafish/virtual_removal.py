"""Recalculate the time-zero YSL and EVL removals used in S33 and S34."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import CytoBridge as cb
from reproduction.zebrafish.daughter_noise import ObservedSupportLinkPredictor, TIMES


def run(data_dir, output_dir, seed=42, device="cuda:0"):
    import anndata as ad

    data, output = Path(data_dir), Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    adata = ad.read_h5ad(data / "aligned.h5ad")
    latent = np.asarray(adata.obsm["X_latent"], dtype=np.float32)
    loaded = cb.tl.load_dynamical_model_from_dir(data / "model", dim=latent.shape[1]+2, device=device)
    loaded.model.eval()
    runtime = cb.tl.build_dynamical_runtime(loaded)
    interaction = runtime.f_net.interaction_net
    interaction.link_predictor = ObservedSupportLinkPredictor(interaction.link_predictor, latent).to(device)
    cb.tl.set_global_random_seed(seed)
    result = cb.tl.run_virtual_cell_type_ablation(
        adata, runtime, ablations={"remove_YSL": ("Yolk Syncytial Layer",), "remove_EVL": ("EVL",)},
        time_points=TIMES, output_dir=output / "experiment", time_index=0, n_samples=None,
        dt=.005, resample_dt=.05, sigma=.03, growth_alpha=1., interaction_m=1024,
        max_particles=100000, device=device, time_key="time_point_processed", annotation_key="Annotation",
        obsm_key="X_latent", spatial_key="spatial_aligned", concat_spatial=True, spatial_dim=2,
        random_seed=seed, interaction_seed=seed+10001, common_random_seed=True, max_ot_points=1024,
        mass_control=False, trajectory_labeler=None, save_data=True, save_snapshots=False, verbose=True)
    (output / "run_summary.json").write_text(json.dumps(dict(
        seed=seed, model=str(loaded.weight_path), data=str((data/"aligned.h5ad").resolve()),
        time_points=TIMES.tolist(), dt=.005, resample_dt=.05, sigma=.03, growth_alpha=1.,
        interaction_m=1024, interaction_seed=seed+10001, max_ot_points=1024,
        initial_counts={"baseline": len(result.baseline_points[0]),
                        **{name: len(points[0]) for name, points in result.ablation_points.items()}},
    ), indent=2) + "\n")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/zebrafish"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    print(run(args.data_dir, args.output_dir, args.seed, args.device))
