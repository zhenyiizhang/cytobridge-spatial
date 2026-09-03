#!/usr/bin/env python3
"""Evaluate one Zebrafish model on common cells and deterministic interaction groups."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import torch

from CytoBridge.tl import (
    evaluate_model_distributions,
    load_dynamical_model_from_dir,
    save_distribution_evaluation,
)
from CytoBridge.tl.core.interaction import cal_interaction
from CytoBridge.tl.downstream.spatial_interaction_attribution import (
    analyze_spatial_gnn_by_celltype,
)


GROUPING_SEED = 20260903
INTERACTION_GROUP_SIZE = 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--aligned-h5ad", type=Path, required=True)
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--edge-predictor", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def interaction_readout_baseline(interaction_net, feature_dim: int, device: str) -> torch.Tensor:
    """Return the neighbor-independent GNN readout bias in joint state space."""
    linear_layers = [
        module
        for module in interaction_net.gene_readout.modules()
        if isinstance(module, torch.nn.Linear)
    ]
    if len(linear_layers) != 1 or linear_layers[0].bias is None:
        raise TypeError("Expected one biased Linear layer in interaction_net.gene_readout")
    gene_bias = linear_layers[0].bias.detach().to(device=device, dtype=torch.float32)
    if int(gene_bias.numel()) != feature_dim - 2:
        raise ValueError("Interaction readout bias does not match the joint feature dimension")
    return torch.cat((torch.zeros(2, device=device), gene_bias), dim=0)


def component_arrays(adata, model, device: str) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    spatial = np.asarray(adata.obsm["spatial_aligned"], dtype=np.float32)
    latent = np.asarray(adata.obsm["X_latent"], dtype=np.float32)
    features = np.hstack((spatial, latent)).astype(np.float32)
    times = np.asarray(adata.obs["time_point_processed"], dtype=np.float64)
    output = {
        "intrinsic": np.zeros_like(features),
        "interaction": np.zeros_like(features),
        "score": np.zeros_like(features),
        "total": np.zeros_like(features),
        "growth": np.zeros(features.shape[0], dtype=np.float32),
        "velocity_network_output": np.zeros_like(features),
        "interaction_network_output": np.zeros_like(features),
        "interaction_readout_baseline": np.zeros_like(features),
    }
    component_rows = []
    model.eval()
    readout_baseline = interaction_readout_baseline(
        model.interaction_net, features.shape[1], device
    )
    for time_index, time_value in enumerate(np.unique(times)):
        positions = np.flatnonzero(np.isclose(times, time_value))
        x = torch.as_tensor(features[positions], dtype=torch.float32, device=device)
        t = torch.full((positions.size, 1), float(time_value), dtype=torch.float32, device=device)
        lnw = torch.full(
            (positions.size, 1),
            -float(np.log(positions.size)),
            dtype=torch.float32,
            device=device,
        )
        with torch.no_grad():
            intrinsic = model.predict_velocity(t=t, x=x)
            growth = model.predict_growth(t=t, x=x).reshape(-1)
            generator = torch.Generator(device=device)
            generator.manual_seed(GROUPING_SEED + time_index)
            interaction = cal_interaction(
                z=x,
                lnw=lnw,
                interaction_potential=model.interaction_net,
                m=INTERACTION_GROUP_SIZE,
                cutoff=1000.0,
                use_mass=bool(getattr(model, "use_growth_in_ode_inter", True)),
                t=torch.tensor([float(time_value)], dtype=torch.float32, device=device),
                generator=generator,
            )
        with torch.enable_grad():
            x_for_score = x.detach().requires_grad_(True)
            _, score = model.compute_score(t=t, x=x_for_score, create_graph=False)
        baseline = readout_baseline[None, :].expand(positions.size, -1)
        arrays = {
            "intrinsic": (intrinsic + baseline).detach().cpu().numpy(),
            "interaction": (interaction - baseline).detach().cpu().numpy(),
            "score": score.detach().cpu().numpy(),
            "growth": growth.detach().cpu().numpy(),
            "velocity_network_output": intrinsic.detach().cpu().numpy(),
            "interaction_network_output": interaction.detach().cpu().numpy(),
            "interaction_readout_baseline": baseline.detach().cpu().numpy(),
        }
        arrays["total"] = arrays["intrinsic"] + arrays["interaction"] + arrays["score"]
        for name, values in arrays.items():
            output[name][positions] = values
        intrinsic_norm = np.linalg.norm(arrays["intrinsic"], axis=1)
        interaction_norm = np.linalg.norm(arrays["interaction"], axis=1)
        total_norm = np.linalg.norm(arrays["total"], axis=1)
        denominator = intrinsic_norm + interaction_norm
        fraction = np.divide(
            interaction_norm,
            denominator,
            out=np.zeros_like(interaction_norm),
            where=denominator > 1e-12,
        )
        component_rows.append(
            {
                "time": float(time_value),
                "n_cells": int(positions.size),
                "intrinsic_norm_median": float(np.median(intrinsic_norm)),
                "interaction_norm_median": float(np.median(interaction_norm)),
                "total_norm_median": float(np.median(total_norm)),
                "interaction_fraction_median": float(np.median(fraction)),
                "growth_median": float(np.median(arrays["growth"])),
                "growth_abs_median": float(np.median(np.abs(arrays["growth"]))),
            }
        )
    output["features"] = features
    output["times"] = times
    return output, pd.DataFrame(component_rows)


def type_pair_tables(adata, model, device: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    spatial = np.asarray(adata.obsm["spatial_aligned"], dtype=np.float32)
    latent = np.asarray(adata.obsm["X_latent"], dtype=np.float32)
    features = np.hstack((spatial, latent)).astype(np.float32)
    times = np.asarray(adata.obs["time_point_processed"], dtype=np.float64)
    labels = np.asarray(adata.obs["Annotation"].astype(str), dtype=object)
    pair_tables = []
    reconstruction_tables = []
    for time_index, time_value in enumerate(np.unique(times)):
        positions = np.flatnonzero(np.isclose(times, time_value))
        x = torch.as_tensor(features[positions], dtype=torch.float32, device=device)
        lnw = torch.full(
            (positions.size, 1),
            -float(np.log(positions.size)),
            dtype=torch.float32,
            device=device,
        )
        result = analyze_spatial_gnn_by_celltype(
            model.interaction_net,
            x,
            lnw,
            torch.tensor([float(time_value)], dtype=torch.float32, device=device),
            labels[positions],
            group_size=INTERACTION_GROUP_SIZE,
            grouping_seed=GROUPING_SEED + time_index,
            spatial_dim=2,
        )
        pair_table = result.type_pair_table.copy()
        pair_table.insert(0, "time", float(time_value))
        pair_tables.append(pair_table)
        reconstruction = result.reconstruction_table.copy()
        reconstruction.insert(0, "time", float(time_value))
        reconstruction_tables.append(reconstruction)
        del result, x, lnw
        torch.cuda.empty_cache()
    return (
        pd.concat(pair_tables, ignore_index=True),
        pd.concat(reconstruction_tables, ignore_index=True),
    )


def main() -> int:
    args = arguments()
    aligned_h5ad = args.aligned_h5ad.expanduser().resolve(strict=True)
    training_dir = args.training_dir.expanduser().resolve(strict=True)
    edge_predictor = args.edge_predictor.expanduser().resolve(strict=True)
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Evaluation directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    adata = sc.read_h5ad(aligned_h5ad)
    dimension = int(adata.obsm["spatial_aligned"].shape[1] + adata.obsm["X_latent"].shape[1])
    loaded = load_dynamical_model_from_dir(
        training_dir,
        dim=dimension,
        device=args.device,
        edge_predictor_path=edge_predictor,
    )
    model = loaded.model.to(args.device).eval()

    arrays, component_summary = component_arrays(adata, model, args.device)
    np.savez_compressed(
        output_dir / "observed_cell_components.npz",
        **arrays,
        obs_names=np.asarray(adata.obs_names.astype(str), dtype=object),
    )
    component_summary.to_csv(output_dir / "component_summary_by_time.csv", index=False)
    pair_table, reconstruction_table = type_pair_tables(adata, model, args.device)
    pair_table.to_csv(output_dir / "directed_celltype_pair_attribution.csv", index=False)
    reconstruction_table.to_csv(output_dir / "edge_reconstruction_check.csv", index=False)

    distribution = evaluate_model_distributions(
        adata,
        model,
        n_samples=5000,
        dt=0.01,
        sigma=float(loaded.config["training"]["defaults"].get("sigma", 0.03)),
        include_score=True,
        interaction_m=INTERACTION_GROUP_SIZE,
        max_ot_points=1024,
        device=args.device,
        random_seed=42,
        include_initial_time=False,
    )
    distribution_paths = save_distribution_evaluation(
        distribution, output_dir / "distribution_evaluation"
    )
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "condition": args.condition,
        "aligned_h5ad": str(aligned_h5ad),
        "training_dir": str(training_dir),
        "training_config": str(training_dir / "config.yaml"),
        "training_config_sha256": sha256(training_dir / "config.yaml"),
        "weight_checkpoint": str(loaded.weight_path),
        "weight_checkpoint_sha256": sha256(loaded.weight_path),
        "score_checkpoint": str(loaded.score_path),
        "score_checkpoint_sha256": sha256(loaded.score_path),
        "weight_stage": loaded.weight_stage,
        "score_stage": loaded.score_stage,
        "fixed_evaluation_settings": {
            "observed_cells": int(adata.n_obs),
            "time_key": "time_point_processed",
            "cell_type_key": "Annotation",
            "interaction_group_size": INTERACTION_GROUP_SIZE,
            "interaction_grouping_seed_base": GROUPING_SEED,
            "distribution_rollout_seed": 42,
            "distribution_n_samples": 5000,
            "distribution_max_ot_points": 1024,
            "decomposition_convention": (
                "The neighbor-independent gene-readout bias of the interaction GNN "
                "is assigned to intrinsic-context. The interaction-associated field "
                "contains only exact directed-edge messages. Their sum with score is "
                "identical to the package total drift."
            ),
        },
        "files": {
            "cell_components": "observed_cell_components.npz",
            "component_summary": "component_summary_by_time.csv",
            "directed_pair_attribution": "directed_celltype_pair_attribution.csv",
            "edge_reconstruction_check": "edge_reconstruction_check.csv",
            "distribution_evaluation": {
                key: str(value) for key, value in distribution_paths.items()
            },
        },
    }
    (output_dir / "evaluation_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
