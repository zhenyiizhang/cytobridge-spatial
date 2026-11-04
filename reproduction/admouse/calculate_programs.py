"""Generate AD populations, microglial gene profiles and LR time courses."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

import CytoBridge as cb
from reproduction.admouse.simulation import simulate_from_x0

TIMES = np.round(np.arange(0.0, 2.51, 0.1), 1)


def gene_profiles(reference, populations, output):
    """Inverse PCA, average reconstructed counts within microglia, then standardize each gene."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    loadings = np.asarray(reference.varm["PCs"], dtype=np.float32)
    if "pca_center" in reference.var:
        center = reference.var["pca_center"].to_numpy(dtype=np.float32)
    else:
        center = np.asarray(reference.X.mean(axis=0)).reshape(-1).astype(np.float32)
    max_loading = np.max(np.abs(loadings), axis=1)
    active = max_loading > max(1e-12, 1e-7 * float(max_loading.max()))
    names = reference.var_names[active].astype(str)
    means, sizes = [], []
    for time in TIMES:
        population = populations[float(time)]
        chosen = population.obs["major_annotation"].astype(str).eq("Microglia").to_numpy()
        if not chosen.any():
            raise ValueError(f"No predicted microglia at model time {time:g}")
        latent = np.asarray(population.X[chosen, 2:], dtype=np.float32)
        expression = np.maximum(latent @ loadings[active].T + center[active][None, :], 0.0)
        means.append(np.expm1(expression).mean(axis=0))
        sizes.append({"time": float(time), "n_microglia": int(chosen.sum())})
    values = np.asarray(means, dtype=float).T
    normalized = (values - values.mean(axis=1, keepdims=True)) / np.maximum(values.std(axis=1, keepdims=True), 1e-12)
    profiles = pd.DataFrame(normalized, index=names, columns=TIMES)
    profiles.rename_axis("gene").reset_index().to_csv(output / "gene_zscore_profiles.csv", index=False)
    pd.DataFrame({"gene": names, "temporal_variance": values.var(axis=1),
                  "peak_time": TIMES[normalized.argmax(axis=1)]}).to_csv(output / "gene_temporal_metadata.csv", index=False)
    pd.DataFrame(values, index=names, columns=TIMES).rename_axis("gene").to_csv(output / "microglia_mean_reconstructed_count.csv")
    pd.DataFrame(sizes).to_csv(output / "microglia_cell_counts.csv", index=False)
    return output


def generate(data_dir, output_dir, *, model_dir=None, classifier_path=None, device="cuda:0", lr=True):
    data, output = Path(data_dir).resolve(), Path(output_dir).resolve()
    if data == output or data in output.parents:
        raise ValueError("Choose an output directory outside the downloaded inputs")
    output.mkdir(parents=True, exist_ok=False)
    model_dir = Path(model_dir) if model_dir is not None else data / "model"
    classifier_path = Path(classifier_path) if classifier_path is not None else data / "classifier_cache/classifier_resmlp_46ee959d0b1f14db.pt"
    reference = ad.read_h5ad(data / "aligned.h5ad")
    initial = np.isclose(reference.obs["time_point_processed"].to_numpy(float), 0)
    x0 = np.column_stack((reference.obsm["spatial_aligned"][initial], reference.obsm["X_latent"][initial])).astype(np.float32)
    loaded = cb.tl.load_dynamical_model_from_dir(model_dir, dim=52, device=device,
        edge_predictor_path=data / "edge_classifier/admouse_edge_model.pt")
    loaded.model.eval()
    cb.tl.set_global_random_seed(42)
    states = simulate_from_x0(x0, loaded.model, device=device, times=TIMES)
    classifier = cb.tl.load_cached_mlp_classifier(str(classifier_path), device=device)
    labels = cb.tl.predict_labels_for_trajectories(sde_points=states, ts_points=TIMES,
        model=classifier.model, label_encoder=classifier.label_encoder,
        feature_dim=classifier.feature_dim, device=device, knn_neighbors=1,
        include_time_feature=classifier.include_time_feature)
    state_dir = output / "generated_states"
    state_dir.mkdir()
    populations = {}
    for time, values, celltypes in zip(TIMES, states, labels):
        population = ad.AnnData(X=np.asarray(values, dtype=np.float32))
        population.obs["major_annotation"] = np.asarray(celltypes).astype(str)
        population.obsm["spatial"] = population.X[:, :2].copy()
        population.uns["time"] = float(time)
        population.write_h5ad(state_dir / f"time_{time:g}.h5ad", compression="gzip")
        populations[float(time)] = population
    gene_profiles(reference, populations, output / "gene_programs")
    if lr:
        slices = {str(float(t)): populations[float(t)] for t in TIMES}
        communications = cb.tl.compute_timepoint_communications(
            adata_dict=slices, time_points=TIMES.tolist(), annotation_key="major_annotation",
            f_net=loaded.model, device=device, out_dir=str(output / "communication"),
            save_dense_attention_matrix=False, remove_self_loop=True,
            save_pickle_path=str(output / "communications.pkl"), max_cells_per_timepoint=None, random_seed=42)
        database = Path(cb.__file__).parent / "workflow_databases/CellChatDB.ligrec.mouse.csv"
        lr_result = cb.tl.project_communication_to_lr_timecourses(
            slices, reference, communications, database, time_points=TIMES.tolist(),
            annotation_key="major_annotation", spatial_dim=2, reference_layer=None,
            expression_space="log1p", complex_mode="min", require_all_subunits=True, n_clusters=4)
        lr_result.pair_timecourse.to_csv(output / "lr_pair_timecourse.csv", index=False)
    (output / "calculation.json").write_text(json.dumps({
        "data": str(data / "aligned.h5ad"), "model": str(loaded.weight_path),
        "classifier": str(classifier_path), "times": TIMES.tolist(),
        "seed": 42, "dt": 0.01, "sigma": 0.03, "interaction_m": 1024,
        "populations": {str(t): p.n_obs for t, p in populations.items()},
        "gene_program_inputs": "gene_programs", "lr_calculated": lr,
    }, indent=2) + "\n")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/admouse"))
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--classifier-cache", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--without-lr", action="store_true")
    args = parser.parse_args()
    print(generate(args.data_dir, args.output_dir, model_dir=args.model_dir,
                   classifier_path=args.classifier_cache, device=args.device, lr=not args.without_lr))
