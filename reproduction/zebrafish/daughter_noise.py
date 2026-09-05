"""Simulate the global-time-zero daughter-noise comparison used in S37."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import CytoBridge as cb

TIMES = np.linspace(0, 4, 81)
DISPLAY = np.arange(0, 81, 10)
NOISES = (0., .01, .03, .06)


class ObservedSupportLinkPredictor(torch.nn.Module):
    """Use learned edges whose two expression states lie in observed support."""

    def __init__(self, predictor, observed):
        super().__init__()
        self.base = predictor
        self.register_buffer("lower", torch.as_tensor(observed.min(axis=0)))
        self.register_buffer("upper", torch.as_tensor(observed.max(axis=0)))
        self.max_norm = float(np.linalg.norm(observed, axis=1).max())

    def forward(self, pairs):
        logits = self.base(pairs)
        dim = pairs.shape[1] // 2
        accepted = torch.ones(len(pairs), dtype=torch.bool, device=pairs.device)
        for states in (pairs[:, 2:dim], pairs[:, dim+2:]):
            accepted &= ((states >= self.lower) & (states <= self.upper)).all(dim=1)
            accepted &= torch.linalg.vector_norm(states, dim=1) <= self.max_norm
        return torch.where(accepted.reshape_as(logits), logits, torch.full_like(logits, -1e9))


def object_array(values):
    result = np.empty(len(values), dtype=object)
    for index, value in enumerate(values):
        result[index] = np.asarray(value)
    return result


def simulate(data_dir, output_dir, seed=42, device="cuda:0"):
    data, output = Path(data_dir), Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    import anndata as ad

    observed = ad.read_h5ad(data / "aligned.h5ad", backed="r")
    try:
        latent = np.asarray(observed.obsm["X_latent"], dtype=np.float32)
        spatial = np.asarray(observed.obsm["spatial_aligned"], dtype=np.float32)
        obs = observed.obs.copy()
    finally:
        observed.file.close()
    time = obs.time_point_processed.to_numpy(float)
    initial = np.isclose(time, 0)
    x0 = np.column_stack((spatial[initial], latent[initial])).astype(np.float32)
    initial_labels = obs.loc[initial, "Annotation"].astype(str).to_numpy()
    model = cb.tl.load_dynamical_model_from_dir(data / "model", dim=x0.shape[1], device=device)
    runtime = cb.tl.build_dynamical_runtime(model)
    interaction = runtime.f_net.interaction_net
    interaction.link_predictor = ObservedSupportLinkPredictor(
        interaction.link_predictor, latent).to(device)
    model.model.eval()
    classifier = cb.tl.load_cached_mlp_classifier(
        str(data / "classifier_cache/classifier_resmlp_25f65c49dc60ea4c.pt"), device=device)
    composition, lineage, particles = [], [], []
    for noise in NOISES:
        print(f"Seed {seed}, daughter noise {noise:g}", flush=True)
        cb.tl.set_global_random_seed(seed)
        points, ids = cb.tl.simulate_sde_points_split_from_x0(
            x0=x0, f_net=runtime.f_net, score_net=runtime.score_net, ts_points=TIMES,
            dt=.005, sigma=.03, sigma_by_dim=None, growth_alpha=1.,
            interaction_m=1024, device=device, verbose=False,
            resample_dt=.05, max_particles=100000, daughter_noise_std=noise,
            initial_lineage_ids=np.arange(len(x0)), return_lineage_ids=True,
            interaction_seed=seed+10001,
        )
        labels = []
        for i in DISPLAY:
            frame, ancestors = np.asarray(points[i]), np.asarray(ids[i], dtype=int)
            if not np.isfinite(frame).all() or len(frame) == 0:
                raise ValueError(f"Invalid population at time {TIMES[i]}")
            assigned = initial_labels.copy() if i == 0 else np.asarray(
                cb.tl.predict_labels_for_points(
                    points=frame, time_value=float(TIMES[i]), model=classifier.model,
                    label_encoder=classifier.label_encoder, feature_dim=52,
                    device=device, knn_neighbors=10, include_time_feature=True)).astype(str)
            labels.append(assigned)
            counts = pd.Series(assigned).value_counts()
            for celltype, count in counts.items():
                composition.append(dict(daughter_noise_std=noise, seed=seed, time=TIMES[i],
                                        celltype=celltype, count=count, fraction=count/len(frame),
                                        n_particles=len(frame)))
            transitions = pd.DataFrame({"source_celltype": initial_labels[ancestors],
                                        "target_celltype": assigned}).value_counts().rename("count").reset_index()
            descendants = transitions.groupby("source_celltype")["count"].transform("sum")
            transitions["fraction_within_source"] = transitions["count"] / descendants
            transitions["n_source_descendants"] = descendants
            transitions["n_source_initial"] = transitions.source_celltype.map(pd.Series(initial_labels).value_counts())
            lineage.append(transitions.assign(daughter_noise_std=noise, seed=seed, time=TIMES[i]))
            particles.append(dict(daughter_noise_std=noise, seed=seed, time=TIMES[i], n_particles=len(frame)))
        np.savez_compressed(output / f"daughter_noise_{str(noise).replace('.', 'p')}_trajectory.npz",
                            dense_times=TIMES, points=object_array(points), lineage_ids=object_array(ids),
                            display_times=TIMES[DISPLAY], display_labels=object_array(labels),
                            display_lineage_ids=object_array([ids[i] for i in DISPLAY]))
    pd.DataFrame(composition).to_csv(output / "composition_long.csv", index=False)
    pd.concat(lineage).to_csv(output / "lineage_transition_long.csv", index=False)
    pd.DataFrame(particles).to_csv(output / "particle_counts.csv", index=False)
    observed = obs.groupby(["time_point_processed", "Annotation"], observed=True).size().rename("count").reset_index()
    observed.columns = ["time", "celltype", "count"]
    observed["n_cells"] = observed.groupby("time")["count"].transform("sum")
    observed["fraction"] = observed["count"] / observed["n_cells"]
    observed.to_csv(output / "observed_composition.csv", index=False)
    (output / "run_summary.json").write_text(json.dumps(dict(
        status="complete", seed=seed, data_dir=str(data.resolve()), model=str(model.weight_path),
        initial_time=0, initial_count=len(x0), final_time=4, daughter_noise_std=NOISES,
        dt=.005, resample_dt=.05, sigma=.03, growth_alpha=1., interaction_seed=seed+10001,
    ), indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/zebrafish"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    simulate(args.data_dir, args.output_dir, args.seed, args.device)
