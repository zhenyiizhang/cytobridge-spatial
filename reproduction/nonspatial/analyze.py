"""Post-training analyses for the distributed S4/S5 state-space checkpoints.

Uses the original field, simulation, clone-fate and LR-attribution functions.
Model directories are explicit because the paper Full and No-interaction
models predate the public matched-training workflow's run manifests.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

from reproduction.nonspatial.model import load_state_model


def _output(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def _models(full, no, device):
    return {
        "full": load_state_model(full, device=device, stage="Finetune"),
        "no_interaction": load_state_model(no, device=device, stage="Finetune_no_interaction"),
    }


def _analysis_prior(edge_prior_manifest, output):
    """Resolve relocated prior assets by hash without changing downloaded files."""
    import CytoBridge
    from CytoBridge.nonspatial.interaction_attribution import _sha256
    original = Path(edge_prior_manifest).resolve()
    manifest = json.loads(original.read_text())
    for name in ("lr_pair_metadata.csv", "link_predictor.pt", "cell_splits.npz", "pair_samples.npz"):
        entry = manifest["artifacts"][name]
        declared = Path(entry["path"])
        candidates = [original.parent / name, declared if declared.is_absolute() else original.parent / declared]
        path = next((p for p in candidates if p.is_file() and _sha256(p) == entry["sha256"]), None)
        if path is None:
            raise FileNotFoundError(f"Missing digest-matched edge-prior artifact {name}")
        entry["path"] = str(path.resolve())
    database = manifest["inputs"]["lr_database"]
    bundled = Path(CytoBridge.__file__).parent / "workflow_databases/CellChatDB.ligrec.mouse.csv"
    if _sha256(bundled) != database["sha256"]:
        raise ValueError("Bundled mouse LR database differs from the fitted prior")
    database["path"] = str(bundled.resolve())
    destination = Path(output) / "analysis_edge_prior.json"
    destination.write_text(json.dumps(manifest, indent=2) + "\n")
    return destination


def distribution(prepared_h5ad, full_model_dir, no_model_dir, output_dir, *, device="cuda", seed=42, n_samples=2048):
    """Simulate both selected models and calculate weighted W1/W2/TMV."""
    import anndata as ad
    from CytoBridge.tl.downstream import evaluate_model_distributions, save_distribution_evaluation
    output = _output(output_dir)
    data = ad.read_h5ad(prepared_h5ad)
    rows = []
    for condition, loaded in _models(full_model_dir, no_model_dir, device).items():
        result = evaluate_model_distributions(
            data, loaded.model, n_samples=n_samples, dt=.05, sigma=.1,
            include_score=True, interaction_m=16, max_ot_points=1024,
            structure_max_points=2048, device=device, concat_spatial=False,
            random_seed=seed, verbose=False,
        )
        save_distribution_evaluation(result, output / condition, save_figures=False)
        rows.append(result.metrics.assign(condition=condition, inference_seed=seed))
        loaded.model.to("cpu")
    path = output / "paired_distribution_metrics.csv"
    pd.concat(rows, ignore_index=True).to_csv(path, index=False)
    return path


def direction(source_h5ad, prepared_h5ad, pca_artifacts_npz, full_model_dir, no_model_dir, output_dir, *, device="cuda", grouping_seeds=(101, 202, 303, 404, 505)):
    """Calculate the original one-shot new-RNA reference and model alignment."""
    import torch
    from CytoBridge.nonspatial.scnt_direction import build_direction_reference, model_field, alignment_tables
    output = _output(output_dir)
    data, times, reference, audit = build_direction_reference(
        Path(source_h5ad), Path(prepared_h5ad), Path(pca_artifacts_npz),
        target_sum=1e4, labeling_time=2.,
    )
    rows = []
    for condition, loaded in _models(full_model_dir, no_model_dir, device).items():
        field = model_field(loaded.model, data.obsm["X_latent"], times,
            include_interaction=condition == "full", grouping_seeds=grouping_seeds,
            batch_size=512, device=torch.device(device))
        cells, summary = alignment_tables(seed=42,
            condition="full_interaction_noise" if condition == "full" else "no_interaction_noise",
            obs_names=data.obs_names.astype(str), times=times, reference=reference, field=field)
        cells.to_csv(output / f"{condition}_cellwise.csv.gz", index=False)
        rows.append(summary)
        loaded.model.to("cpu")
    path = output / "timewise_scnt_direction_alignment.csv"
    pd.concat(rows, ignore_index=True).to_csv(path, index=False)
    (output / "reference_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    return path


def trajectory(prepared_h5ad, full_model_dir, output_dir, *, device="cuda", seed=42):
    """Generate 41 identity-preserving 50-PC snapshots for the S5b field."""
    import anndata as ad
    import torch
    from CytoBridge.nonspatial.weinreb_simulation import simulate_sde_from_x0
    output = _output(output_dir)
    data = ad.read_h5ad(prepared_h5ad)
    latent = np.asarray(data.obsm["X_latent"], np.float32)
    times = data.obs["time_point_processed"].to_numpy(float)
    source = np.flatnonzero(np.isclose(times, 0))
    indices = np.random.default_rng(seed).choice(source, size=2048, replace=False)
    loaded = load_state_model(full_model_dir, device=device, stage="Finetune")
    torch.manual_seed(seed)
    grid = np.linspace(0., 2., 41)
    points, weights, _ = simulate_sde_from_x0(x0=latent[indices], model=loaded.model,
        ts_points=grid, dt=.05, sigma=.1, include_score=True, include_interaction=True,
        interaction_m=16, device=device, noise_seed=seed, verbose=False)
    path = output / "full_dense_trajectory.npz"
    np.savez_compressed(path, time_points=grid, points=points, weights=weights)
    (output / "simulation_settings.json").write_text(json.dumps({"seed": seed,
        "dt": .05, "sigma": .1, "interaction_group_size": 16,
        "initial_cell_ids": data.obs_names[indices].tolist()}, indent=2) + "\n")
    return path


def clone_fate(prepared_h5ad, full_model_dir, no_model_dir, output_dir, *, device="cuda", seeds=tuple(range(10))):
    """Propagate all source cells and compare weighted fate distributions by clone.

    The paper classifier is uniform-weighted k=20 fitted on all terminal cells.
    All source cells provide interaction context; only clone-positive cells
    with a terminal lineage are scored, as in the July 18 paper evaluator.
    """
    import anndata as ad
    import torch
    from sklearn.neighbors import KNeighborsClassifier
    from CytoBridge.nonspatial.weinreb_simulation import simulate_sde_from_x0
    from CytoBridge.nonspatial.clone_fate import evaluate_clone_fate_agreement
    from CytoBridge.nonspatial.weinreb_fate import _save_evaluation
    output = _output(output_dir)
    data = ad.read_h5ad(prepared_h5ad)
    latent = np.asarray(data.obsm["X_latent"], np.float32)
    times = data.obs["time_point_processed"].to_numpy(float)
    lineages = data.obs["lineage_id"].astype(str).to_numpy()
    clone_ids = data.obs["clone"].to_numpy(int)
    labels = data.obs["Cell type annotation"].astype(str).to_numpy()
    source = np.flatnonzero(np.isclose(times, 0))
    terminal = np.flatnonzero(np.isclose(times, 2))
    target = terminal[clone_ids[terminal] > 0]
    classifier = KNeighborsClassifier(n_neighbors=20, weights="uniform", metric="euclidean", n_jobs=1).fit(latent[terminal], labels[terminal])
    evaluable = (clone_ids[source] > 0) & np.isin(lineages[source], lineages[target])
    summaries = []
    for condition, loaded in _models(full_model_dir, no_model_dir, device).items():
        endpoint_labels, endpoint_weights = [], []
        for seed in seeds:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            points, weights, _ = simulate_sde_from_x0(x0=latent[source], model=loaded.model,
                ts_points=[0., 2.], dt=.1, sigma=.1, include_score=True,
                include_interaction=condition == "full", interaction_m=16,
                device=device, noise_seed=seed, verbose=False)
            endpoint_labels.append(classifier.predict(points[-1][evaluable]))
            endpoint_weights.append(weights[-1, :, 0][evaluable])
        result = evaluate_clone_fate_agreement(
            np.tile(lineages[source][evaluable], len(seeds)), np.concatenate(endpoint_labels),
            lineages[target], labels[target], generated_endpoint_weights=np.concatenate(endpoint_weights),
            categories=tuple(dict.fromkeys([*labels[target], *classifier.classes_])), min_source=1, min_target=1)
        _save_evaluation(result, output / condition)
        summaries.append({"condition": condition, "training_seed": 42, **result.summary})
        loaded.model.to("cpu")
    path = output / "clone_fate_summary.csv"
    pd.DataFrame(summaries).to_csv(path, index=False)
    (output / "simulation_settings.json").write_text(json.dumps({"seeds": list(seeds), "dt": .1, "sigma": .1, "source_time": 0, "target_time": 2, "classifier": "uniform k=20, all terminal cells", "n_source_cells_simulated": int(len(source)), "n_source_cells_scored": int(evaluable.sum()), "n_target_clone_positive_cells": int(len(target))}, indent=2) + "\n")
    return path


def attribution(expression_h5ad, prepared_h5ad, edge_prior_manifest, model_dirs, output_dir, *, cell_type_key, device="cuda", training_seeds=(42,), grouping_seeds=(101, 202, 303, 404, 505)):
    """Recalculate exact sender messages and expression-weighted LR scores.

    Uses the original one-layer decomposition and its hierarchical averaging:
    grouping replicates within a model, then equal weight across fitted models.
    CellChat is an independent expression-only reference, not an output here.
    """
    import anndata as ad
    from CytoBridge.nonspatial.interaction_attribution import _sha256
    from CytoBridge.tl.downstream.lr_drift_attribution import (
        load_edge_prior_manifest, scaled_lr_activities_from_manifest,
        compute_type_lr_scores, analyze_exact_groupings, summarize_drift_across_seeds,
    )
    if len(model_dirs) != len(training_seeds):
        raise ValueError("Supply one training seed per model directory")
    output = _output(output_dir)
    manifest, manifest_path = load_edge_prior_manifest(_analysis_prior(edge_prior_manifest, output))
    for key, path in (("expression_h5ad", expression_h5ad), ("latent_h5ad", prepared_h5ad)):
        if _sha256(Path(path)) != manifest["inputs"][key]["sha256"]:
            raise ValueError(f"{key} differs from the fitted prior's input")
    expression, data = ad.read_h5ad(expression_h5ad), ad.read_h5ad(prepared_h5ad)
    if not np.array_equal(expression.obs_names, data.obs_names):
        raise ValueError("Expression and latent cells differ or have different row order")
    latent = np.asarray(data.obsm["X_latent"], np.float32)
    times = data.obs["time_point_processed"].to_numpy(float)
    observed_times = expression.obs[manifest["configuration"]["time_key"]].to_numpy(float)
    labels = data.obs[cell_type_key].astype(str).to_numpy()
    ligand, receptor, metadata, normalization = scaled_lr_activities_from_manifest(expression, manifest, manifest_path)
    _, pathway_q, total_q = compute_type_lr_scores(ligand, receptor, metadata, times=observed_times, cell_types=labels)
    del expression, ligand, receptor
    rows, checks = [], []
    for model_dir, seed in zip(model_dirs, training_seeds):
        loaded = load_state_model(model_dir, device=device, stage="Finetune")
        drift, diagnostics = analyze_exact_groupings(loaded.model.interaction_net, latent,
            observed_times=observed_times, model_times=times, cell_types=labels,
            grouping_seeds=grouping_seeds, training_seed=seed, model_label=f"seed{seed}",
            group_size=16, device=device)
        rows.append(drift)
        checks.append(diagnostics)
        loaded.model.to("cpu")
    by_seed, summary = summarize_drift_across_seeds(pd.concat(rows, ignore_index=True))
    by_seed.to_csv(output / "exact_message_by_training_seed.csv", index=False)
    summary.to_csv(output / "exact_message_summary.csv", index=False)
    pd.concat(checks, ignore_index=True).to_csv(output / "reconstruction_diagnostics.csv", index=False)
    keys = ["time", "sender_type", "receiver_type"]
    d_columns = keys + ["D_AB_mean", "D_AB_training_seed_std", "connected_edge_count_mean", "A_AB_receiver_share_all_mean", "mean_cosine_to_total_interaction_mean"]
    network = total_q.merge(summary[d_columns], on=keys, validate="one_to_one")
    network["S_AB_total"] = network.D_AB_mean * network.Q_AB_total
    network.to_csv(output / "cell_type_interaction_network.csv", index=False)
    pathways = pathway_q.merge(summary[d_columns], on=keys, validate="many_to_one")
    pathways["S_AB_pathway"] = pathways.D_AB_mean * pathways.Q_AB_pathway
    pathways["pathway_cellchat_harmonized"] = pathways.pathway.replace({"SEMATOSTATIN": "SOMATOSTATIN"})
    pathways.to_csv(output / "cell_type_pathway_scores.csv.gz", index=False)
    (output / "analysis_settings.json").write_text(json.dumps({"model_dirs": [str(p) for p in model_dirs], "training_seeds": list(training_seeds), "grouping_seeds": list(grouping_seeds), "normalization": normalization}, indent=2) + "\n")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["distribution", "clone-fate", "direction", "attribution", "trajectory"])
    parser.add_argument("--prepared-h5ad", required=True, type=Path)
    parser.add_argument("--full-model-dir", type=Path)
    parser.add_argument("--no-interaction-model-dir", type=Path)
    parser.add_argument("--model-dir", type=Path, action="append")
    parser.add_argument("--training-seed", type=int, action="append")
    parser.add_argument("--grouping-seed", type=int, action="append")
    parser.add_argument("--expression-h5ad", type=Path)
    parser.add_argument("--edge-prior-manifest", type=Path)
    parser.add_argument("--cell-type-key")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--source-h5ad", type=Path)
    parser.add_argument("--pca-artifacts-npz", type=Path)
    parser.add_argument("--seed", action="append", type=int)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.operation == "trajectory":
        if not args.full_model_dir:
            parser.error("trajectory requires --full-model-dir")
        print(trajectory(args.prepared_h5ad, args.full_model_dir, args.output_dir,
            device=args.device, seed=(args.seed or [42])[0]))
        return
    if args.operation == "attribution":
        if not all((args.model_dir, args.training_seed, args.expression_h5ad, args.edge_prior_manifest, args.cell_type_key)):
            parser.error("attribution requires --model-dir, --training-seed, --expression-h5ad, --edge-prior-manifest and --cell-type-key")
        print(attribution(args.expression_h5ad, args.prepared_h5ad, args.edge_prior_manifest,
            args.model_dir, args.output_dir, cell_type_key=args.cell_type_key, device=args.device,
            training_seeds=tuple(args.training_seed), grouping_seeds=tuple(args.grouping_seed or (101, 202, 303, 404, 505))))
        return
    if not args.full_model_dir or not args.no_interaction_model_dir:
        parser.error("This operation requires both model directories")
    shared = dict(prepared_h5ad=args.prepared_h5ad, full_model_dir=args.full_model_dir,
        no_model_dir=args.no_interaction_model_dir, output_dir=args.output_dir, device=args.device)
    if args.operation == "distribution":
        result = distribution(**shared, seed=(args.seed or [42])[0])
    elif args.operation == "clone-fate":
        result = clone_fate(**shared, seeds=tuple(args.seed or range(10)))
    else:
        if not args.source_h5ad or not args.pca_artifacts_npz:
            parser.error("direction requires --source-h5ad and --pca-artifacts-npz")
        result = direction(**shared, source_h5ad=args.source_h5ad, pca_artifacts_npz=args.pca_artifacts_npz)
    print(result)


if __name__ == "__main__":
    main()
