#!/usr/bin/env python3
"""Summarize an LR-informed non-spatial GNN by original scNT cell type.

The cell-type labels are read only after model fitting.  ``D_AB`` is the norm
of the exact learned sender-to-receiver GNN message, ``Q_AB`` is the frozen
CellChatDB expression compatibility, and ``S_AB = D_AB * Q_AB`` is an
LR-annotated interaction-drift score.  None is a causal communication
probability.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import anndata as ad
import numpy as np
import pandas as pd
import torch

from CytoBridge.tl.downstream import load_dynamical_model_from_dir
from CytoBridge.tl.downstream.lr_drift_attribution import (
    analyze_exact_groupings,
    compute_type_lr_scores,
    load_edge_prior_manifest,
    scaled_lr_activities_from_manifest,
    summarize_drift_across_seeds,
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expression-h5ad", required=True, type=Path)
    parser.add_argument("--latent-h5ad", required=True, type=Path)
    parser.add_argument("--edge-prior-manifest", required=True, type=Path)
    parser.add_argument("--training-run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--cell-type-key", default="cell_type")
    parser.add_argument("--time-key", default="time_point_processed")
    parser.add_argument("--training-seed", type=int, default=42)
    parser.add_argument(
        "--grouping-seeds", nargs="+", type=int, default=[101, 202, 303, 404, 505]
    )
    parser.add_argument("--group-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_file(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    expression_path = _require_file(args.expression_h5ad)
    latent_path = _require_file(args.latent_h5ad)
    training_dir = args.training_run_dir.expanduser().resolve()
    model_dir = training_dir / "model"
    run_manifest_path = _require_file(training_dir / "run_manifest.json")
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Refusing non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    edge_manifest, edge_manifest_path = load_edge_prior_manifest(
        args.edge_prior_manifest
    )
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    if str(run_manifest.get("condition")) != "full":
        raise ValueError("Interaction attribution requires a full-condition model")
    if int(run_manifest.get("seed", -1)) != int(args.training_seed):
        raise ValueError("Requested training seed does not match run_manifest.json")
    if _sha256(latent_path) != str(run_manifest.get("prepared_sha256")):
        raise ValueError("Latent H5AD does not match the trained model manifest")
    bound_prior = run_manifest.get("edge_prior")
    if not isinstance(bound_prior, dict):
        raise ValueError("Training manifest is missing a bound LR edge prior")
    if str(bound_prior.get("manifest_sha256")) != _sha256(edge_manifest_path):
        raise ValueError("Training and analysis LR-prior manifests differ")

    for key, path in (
        ("expression_h5ad", expression_path),
        ("latent_h5ad", latent_path),
    ):
        expected = str(edge_manifest["inputs"][key]["sha256"])
        if _sha256(path) != expected:
            raise ValueError(f"Frozen input hash mismatch for {key}")

    expression = ad.read_h5ad(expression_path)
    latent_adata = ad.read_h5ad(latent_path)
    if not np.array_equal(
        expression.obs_names.astype(str), latent_adata.obs_names.astype(str)
    ):
        raise ValueError("Expression and latent cell identities differ")
    if args.cell_type_key not in latent_adata.obs:
        raise KeyError(f"Missing cell-type column {args.cell_type_key!r}")
    if args.time_key not in latent_adata.obs:
        raise KeyError(f"Missing time column {args.time_key!r}")

    latent = np.asarray(latent_adata.obsm["X_latent"], dtype=np.float32)
    model_times = latent_adata.obs[args.time_key].to_numpy(dtype=float)
    expression_time_key = str(edge_manifest["configuration"]["time_key"])
    expression_times = expression.obs[expression_time_key].to_numpy(dtype=float)
    time_mapping = {}
    for observed_time in sorted(np.unique(expression_times)):
        mapped = np.unique(model_times[np.isclose(expression_times, observed_time)])
        if len(mapped) != 1:
            raise ValueError(
                f"Observed time {observed_time!r} does not map to one model time."
            )
        time_mapping[str(float(observed_time))] = float(mapped[0])
    if len(set(time_mapping.values())) != len(time_mapping):
        raise ValueError("Distinct observed times must map to distinct model times.")
    labels = latent_adata.obs[args.cell_type_key].astype(str).to_numpy()
    if np.any(pd.isna(labels)) or np.any(labels == ""):
        raise ValueError("Cell-type labels must be complete")
    if np.unique(labels).size < 2:
        raise ValueError("At least two cell types are required")

    counts = (
        pd.DataFrame({"time": model_times, "cell_type": labels})
        .groupby(["time", "cell_type"], as_index=False)
        .size()
        .rename(columns={"size": "n_cells"})
    )
    counts.to_csv(output_dir / "cell_type_counts_by_time.csv", index=False)

    ligand, receptor, lr_metadata, normalization = scaled_lr_activities_from_manifest(
        expression,
        edge_manifest,
        edge_manifest_path,
    )
    pair_q, pathway_q, total_q = compute_type_lr_scores(
        ligand,
        receptor,
        lr_metadata,
        times=expression_times,
        cell_types=labels,
    )
    del ligand, receptor

    device = torch.device(args.device)
    loaded = load_dynamical_model_from_dir(
        model_dir,
        dim=int(latent.shape[1]),
        device=device,
        stage="Finetune",
    )
    drift, diagnostics = analyze_exact_groupings(
        loaded.model.interaction_net,
        latent,
        observed_times=expression_times,
        model_times=model_times,
        cell_types=labels,
        grouping_seeds=tuple(int(value) for value in args.grouping_seeds),
        training_seed=int(args.training_seed),
        model_label=f"seed{int(args.training_seed)}",
        group_size=int(args.group_size),
        device=device,
    )
    weight_hash = _sha256(loaded.weight_path)
    score_hash = _sha256(loaded.score_path)
    loaded.model.to("cpu")
    del loaded
    if device.type == "cuda":
        torch.cuda.empty_cache()

    by_seed, drift_summary = summarize_drift_across_seeds(drift)
    drift.to_csv(
        output_dir / "exact_message_by_grouping.csv.gz",
        index=False,
        compression="gzip",
    )
    by_seed.to_csv(output_dir / "exact_message_by_training_seed.csv", index=False)
    drift_summary.to_csv(output_dir / "exact_message_summary.csv", index=False)
    diagnostics.to_csv(output_dir / "reconstruction_diagnostics.csv", index=False)

    d_columns = [
        "time",
        "sender_type",
        "receiver_type",
        "D_AB_mean",
        "D_AB_training_seed_std",
        "connected_edge_count_mean",
        "A_AB_receiver_share_all_mean",
        "mean_cosine_to_total_interaction_mean",
    ]
    d_table = drift_summary[
        [column for column in d_columns if column in drift_summary]
    ].copy()
    network = total_q.merge(
        d_table,
        on=["time", "sender_type", "receiver_type"],
        how="inner",
        validate="one_to_one",
    )
    network["S_AB_total"] = network["D_AB_mean"] * network["Q_AB_total"]
    network.to_csv(output_dir / "cell_type_interaction_network.csv", index=False)

    pathways = pathway_q.merge(
        d_table,
        on=["time", "sender_type", "receiver_type"],
        how="inner",
        validate="many_to_one",
    )
    pathways["S_AB_pathway"] = pathways["D_AB_mean"] * pathways["Q_AB_pathway"]
    pathway_aliases = {"SEMATOSTATIN": "SOMATOSTATIN"}
    pathways["pathway_cellchat_harmonized"] = pathways["pathway"].replace(
        pathway_aliases
    )
    pathways.to_csv(
        output_dir / "cell_type_pathway_scores.csv.gz",
        index=False,
        compression="gzip",
    )
    pair_q.to_csv(
        output_dir / "cell_type_lr_pair_compatibility.csv.gz",
        index=False,
        compression="gzip",
    )
    (
        pathways.groupby(["time", "pathway"], as_index=False)
        .agg(
            S_total=("S_AB_pathway", "sum"),
            Q_total=("Q_AB_pathway", "sum"),
            n_cell_type_pairs=("S_AB_pathway", "size"),
        )
        .to_csv(output_dir / "pathway_summary.csv", index=False)
    )

    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis": "post-training exact GNN interaction-message and LR attribution by original scNT cell type",
        "interpretation": {
            "D_AB": "mean norm of exact learned GNN message from sender type A to receiver type B",
            "Q_AB": "frozen pathway-balanced observed LR compatibility",
            "S_AB": "D_AB * Q_AB; LR-annotated interaction-drift score, not a causal communication probability",
        },
        "commot_to_cellchat_pathway_aliases": pathway_aliases,
        "label_usage": {
            "cell_type_key": str(args.cell_type_key),
            "used_in_preprocessing": False,
            "used_in_training": False,
            "used_post_training_for_grouped_interpretation_only": True,
        },
        "training_seed": int(args.training_seed),
        "grouping_seeds": [int(value) for value in args.grouping_seeds],
        "group_size": int(args.group_size),
        "observed_to_model_time_mapping": time_mapping,
        "n_cells": int(latent_adata.n_obs),
        "n_cell_types": int(np.unique(labels).size),
        "model_weight": {
            "path": str(model_dir),
            "finetune_weight_sha256": weight_hash,
            "score_weight_sha256": score_hash,
        },
        "inputs": {
            "expression_h5ad": str(expression_path),
            "expression_sha256": _sha256(expression_path),
            "latent_h5ad": str(latent_path),
            "latent_sha256": _sha256(latent_path),
            "training_manifest": str(run_manifest_path),
            "training_manifest_sha256": _sha256(run_manifest_path),
            "edge_prior_manifest": str(edge_manifest_path),
            "edge_prior_manifest_sha256": _sha256(edge_manifest_path),
        },
        "normalization": normalization,
        "row_counts": {
            "interaction_network": int(len(network)),
            "pathway_scores": int(len(pathways)),
            "lr_pair_compatibility": int(len(pair_q)),
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "n_cell_types": int(np.unique(labels).size),
                "interaction_rows": int(len(network)),
                "pathway_rows": int(len(pathways)),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
