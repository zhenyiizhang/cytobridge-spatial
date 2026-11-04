"""Calculate the Trem2 and Spp1 perturbations in Figure 6 and S30."""
from __future__ import annotations
import argparse
import gc
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

import CytoBridge as cb
from reproduction.admouse.calculate_programs import TIMES
from reproduction.admouse.gene_sets import GENE_SETS
from reproduction.admouse.simulation import simulate_from_x0


def module_contrasts(reference, baseline, perturbed, gene, direction, time):
    """Average per-gene z-score changes using equally weighted population moments."""
    from CytoBridge.tl.downstream.temporal import inverse_pca_states, simplify_gene_names
    # The published Spp1 analysis clipped reconstructed log-expression at zero.
    clip = 0.0 if gene == "Spp1" else None
    base_values = inverse_pca_states(reference, np.asarray(baseline, dtype=np.float32), clip_min=clip)
    values = inverse_pca_states(reference, np.asarray(perturbed, dtype=np.float32), clip_min=clip)
    base_mean, mean = base_values.mean(axis=0).astype(float), values.mean(axis=0).astype(float)
    base_var, var = base_values.var(axis=0).astype(float), values.var(axis=0).astype(float)
    center = (base_mean + mean) / 2
    sd = np.sqrt(np.maximum(0.5*(base_var+(base_mean-center)**2)+0.5*(var+(mean-center)**2), 1e-24))
    z_base, z_pert = (base_mean-center)/sd, (mean-center)/sd
    symbols = simplify_gene_names(reference.var_names)
    loadings = np.asarray(reference.varm["PCs"], dtype=np.float32)
    magnitude = np.abs(loadings).max(axis=1)
    active = magnitude > max(1e-12, 1e-7 * float(magnitude.max()))
    if gene == "Spp1":
        # Keep the original per-cell calculation and floating-point order.
        pooled_mean = 0.5 * (base_values.mean(axis=0) + values.mean(axis=0))
        pooled_second = 0.5 * ((base_values**2).mean(axis=0) + (values**2).mean(axis=0))
        pooled_sd = np.sqrt(np.maximum(pooled_second - pooled_mean**2, 1e-12))
        base_scores = (base_values - pooled_mean) / pooled_sd
        perturbed_scores = (values - pooled_mean) / pooled_sd
    rows = []
    for module, genes in GENE_SETS.items():
        indices = symbols.index[symbols.gene_symbol.isin(genes) & active].to_numpy(dtype=int)
        if not len(indices):
            continue
        score_base = float(base_scores[:, indices].mean()) if gene == "Spp1" else float(z_base[indices].mean())
        score_perturbed = float(perturbed_scores[:, indices].mean()) if gene == "Spp1" else float(z_pert[indices].mean())
        rows.append(dict(perturbed_gene=gene, direction=direction, time=time,
            population="all_particles", module=module, n_reference_particles=len(baseline),
            n_perturbed_particles=len(perturbed), n_genes=len(indices),
            mean_reference=score_base, mean_perturbed=score_perturbed,
            delta=score_perturbed-score_base,
            normalization="equal-weight pooled particle moments"))
    return rows


def save_case(directory, state, labels, model, device):
    directory = Path(directory)
    (directory / "states").mkdir(parents=True, exist_ok=True)
    (directory / "labels_k1").mkdir(exist_ok=True)
    np.save(directory / "states/generated_t2.5.npy", state)
    np.save(directory / "labels_k1/labels_t2.5.npy", np.asarray(labels).astype(str))
    population = ad.AnnData(X=np.empty((len(state), 0), dtype=np.float32))
    population.obsm["spatial_aligned"] = state[:, :2]
    population.obsm["X_latent"] = state[:, 2:]
    cb.tl.save_interpolated_attention(population, 2.5, model=model, device=device,
        out_dir=str(directory / "communication/attention"), save_dense_matrix=False)


def run(data_dir, baseline_dir, output_dir, *, gene="Trem2", device="cuda:0", model_dir=None,
        classifier_path=None):
    if gene not in {"Trem2", "Spp1"}:
        raise ValueError("Choose Trem2 or Spp1")
    data, baseline_dir, output = (Path(p).resolve() for p in (data_dir, baseline_dir, output_dir))
    if any(output == p or p in output.parents for p in (data, baseline_dir)):
        raise ValueError("Choose an output directory outside the input and baseline directories")
    record = json.loads((baseline_dir / "calculation.json").read_text())
    if Path(record["data"]).resolve() != (data / "aligned.h5ad").resolve():
        raise ValueError("Use the same aligned data for baseline and perturbed populations")
    reference = ad.read_h5ad(data / "aligned.h5ad")
    initial = np.isclose(reference.obs["time_point_processed"].to_numpy(float), 0)
    x0 = np.column_stack((reference.obsm["spatial_aligned"][initial], reference.obsm["X_latent"][initial])).astype(np.float32)
    loaded = cb.tl.load_dynamical_model_from_dir(model_dir or data / "model", dim=52, device=device,
        edge_predictor_path=data / "edge_classifier/admouse_edge_model.pt")
    loaded.model.eval()
    if Path(record["model"]).resolve() != Path(loaded.weight_path).resolve():
        raise ValueError("Generate the baseline with the same model before running its perturbations")
    classifier_path = Path(classifier_path or record["classifier"]).resolve()
    if classifier_path != Path(record["classifier"]).resolve():
        raise ValueError("Use the same classifier for baseline and perturbed populations")
    output.mkdir(parents=True, exist_ok=False)
    classifier = cb.tl.load_cached_mlp_classifier(str(classifier_path), device=device)
    # Trem2 uses the scale-1 run. Spp1 uses the scale-2.5 run.
    scale = 1.0 if gene == "Trem2" else 2.5
    loading = np.asarray(reference.varm["PCs"])[reference.var_names.get_loc(gene)]
    loading = loading.astype(np.float64 if gene == "Trem2" else np.float32)
    pc_std = np.asarray(reference.obsm["X_latent"]).std(axis=0, ddof=0)
    delta = (scale * pc_std * (loading / np.linalg.norm(loading))).astype(np.float32)
    baseline = {t: ad.read_h5ad(baseline_dir / "generated_states" / f"time_{t:g}.h5ad") for t in (2.4, 2.5)}
    rows = []
    for direction, sign in (("low", -1), ("high", 1)):
        print(f"Simulating {gene}: {direction}", flush=True)
        perturbed_x0 = x0.copy()
        perturbed_x0[:, 2:] += sign * delta
        cb.tl.set_global_random_seed(42)
        states = simulate_from_x0(perturbed_x0, loaded.model, device=device, times=TIMES)
        for time in (2.4, 2.5):
            state = states[int(round(time*10))]
            saved = output / "states" / direction
            saved.mkdir(parents=True, exist_ok=True)
            np.save(saved / f"time_{time:g}.npy", state)
            rows.extend(module_contrasts(reference, baseline[time].X, state, gene, direction, time))
        if gene == "Trem2":
            labels = cb.tl.predict_labels_for_trajectories(sde_points=[states[-1]], ts_points=[2.5],
                model=classifier.model, label_encoder=classifier.label_encoder, feature_dim=classifier.feature_dim,
                device=device, knn_neighbors=1, include_time_feature=classifier.include_time_feature)[0]
            save_case(output / "figure_source/whole_tissue/perturbations/Trem2" / direction,
                      states[-1], labels, loaded.model, device)
        del states
        gc.collect()
    scores = pd.DataFrame(rows)
    scores.to_csv(output / f"{gene.lower()}_module_scores.csv", index=False)
    if gene == "Trem2":
        # Store the same newly generated baseline alongside the two perturbations.
        source = output / "figure_source"
        population = baseline[2.5]
        save_case(source / "baseline", population.X, population.obs.major_annotation.to_numpy(), loaded.model, device)
        import shutil
        (source / "compat_base/01_interpolation").mkdir(parents=True)
        shutil.copy2(source / "baseline/states/generated_t2.5.npy", source / "compat_base/01_interpolation/generated_t2.5.npy")
        shutil.copytree(source / "baseline/labels_k1", source / "whole_tissue/baseline_labels_k1")
        shutil.copytree(source / "baseline/communication/attention", source / "whole_tissue/baseline_communication/attention/edges")
    (output / "calculation.json").write_text(json.dumps(dict(gene=gene, scale=scale, seed=42,
        model=str(loaded.weight_path), baseline=str(baseline_dir.resolve()), classifier=str(classifier_path)), indent=2)+"\n")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/admouse"))
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--classifier-cache", type=Path)
    parser.add_argument("--gene", choices=["Trem2", "Spp1"], required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    print(run(args.data_dir, args.baseline_dir, args.output_dir, gene=args.gene,
              device=args.device, model_dir=args.model_dir, classifier_path=args.classifier_cache))
