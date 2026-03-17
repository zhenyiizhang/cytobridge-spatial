#!/usr/bin/env python3
"""Project-level preprocessing pipeline.

This script intentionally lives under `scripts/` (not package API) because it
combines multiple stages:
1) gene preprocessing + spatial alignment
2) interaction graph generation
3) edge predictor training
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd
import scanpy as sc

# Ensure local package import works when running as a script.
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from CytoBridge.pp.edge_prediction import train_edge_predictor
from CytoBridge.pp.interaction_graph import (
    estimate_neighborhood_threshold_from_aligned_spatial,
    generate_interaction_graph,
    sanitize_interaction_graph_uns,
)
from CytoBridge.pp.preprocess import preprocess
from CytoBridge.pp.spatial_align import AlignConfig, align_spatial


def run_preprocessing_pipeline(
    data_name: str,
    h5ad_path: str,
    time_key: str,
    output_dir: str = "results/",
    align_config: Optional[AlignConfig] = None,
    database_path: str = "database/CellNEST_database.csv",
    split: int = 0,
    edge_epochs: int = 100,
    batch_indices: Optional[Sequence[int]] = None,
    neighborhood_threshold: Optional[float] = None,
    spatial_key: str = "spatial_aligned",
    device: str = "cuda",
) -> dict:
    """Run preprocess+alignment -> graph generation -> edge predictor training."""
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    aligned_csv_path = output_root / f"{data_name}_aligned.csv"
    aligned_h5ad_path = output_root / f"{data_name}_aligned.h5ad"
    graph_input_dir = output_root / "input_graph"
    metadata_dir = output_root / "metadata"
    edge_model_path = output_root / "edge_classifier" / f"{data_name}_edge_model.pt"

    if align_config is None:
        align_config = AlignConfig()

    print(f"=== Starting preprocessing pipeline: {data_name} ===")
    print(f"[1/3] Preprocessing genes + aligning spatial coordinates -> {aligned_h5ad_path}")
    adata_raw = sc.read_h5ad(h5ad_path)
    adata_preprocessed = preprocess(
        adata=adata_raw,
        time_key=time_key,
        n_top_genes=align_config.n_top_genes,
        dim_reduction="pca",
        n_pcs=align_config.n_pcs,
    )
    adata_aligned = align_spatial(
        adata_or_h5ad=adata_preprocessed,
        time_key=time_key,
        cfg=align_config,
        batch_indices=batch_indices,
        device=device,
        output_csv=str(aligned_csv_path),
        output_h5ad=str(aligned_h5ad_path),
    )

    # Estimate one global neighborhood threshold from aligned spatial coordinates.
    if neighborhood_threshold is None or float(neighborhood_threshold) <= 0:
        neighborhood_threshold, recommended_spot_diameter, nn_stats_df, spatial_key_used = (
            estimate_neighborhood_threshold_from_aligned_spatial(
                adata_aligned,
                time_key=time_key,
                spatial_key=spatial_key,
                recommended_spot_scale=1.2,
                neighborhood_factor=4.0,
                store_nn1_in_obs=True,
                store_in_uns=True,
            )
        )
        print(
            "[2/3] Auto neighborhood threshold estimated from aligned NN stats: "
            f"{neighborhood_threshold:.6f} "
            f"(recommended_spot_diameter_mean={recommended_spot_diameter:.6f}, spatial_key={spatial_key_used})"
        )
        nn_stats_df.to_csv(output_root / f"{data_name}_nn1_stats.csv", index=False)
        sanitize_interaction_graph_uns(adata_aligned)
        adata_aligned.write_h5ad(aligned_h5ad_path)
    else:
        neighborhood_threshold = float(neighborhood_threshold)
        spatial_key_used = spatial_key
        recommended_spot_diameter = neighborhood_threshold / 4.0
        adata_aligned.uns.setdefault("interaction_graph", {})
        adata_aligned.uns["interaction_graph"]["neighborhood_threshold"] = neighborhood_threshold
        adata_aligned.uns["interaction_graph"]["recommended_spot_diameter"] = recommended_spot_diameter
        adata_aligned.uns["interaction_graph"]["spatial_key"] = spatial_key_used
        sanitize_interaction_graph_uns(adata_aligned)
        adata_aligned.write_h5ad(aligned_h5ad_path)
        print(
            "[2/3] Using user-provided neighborhood threshold: "
            f"{neighborhood_threshold:.6f} (spatial_key={spatial_key_used})"
        )

    print("[2/3] Building interaction graphs for each aligned time point")
    processed_time_key = "time_point_processed"
    if processed_time_key not in adata_aligned.obs:
        raise KeyError(
            f"Expected '{processed_time_key}' in adata.obs after preprocess, "
            "but it is missing."
        )
    processed_times = sorted(pd.to_numeric(adata_aligned.obs[processed_time_key], errors="raise").unique())

    for slice_idx, t_processed in enumerate(processed_times):
        slice_name = f"{data_name}_t{slice_idx}"
        print(f"  - {slice_name} (processed_time={float(t_processed)})")
        generate_interaction_graph(
            data_name=slice_name,
            data_from=adata_aligned,
            data_to=str(graph_input_dir),
            metadata_to=str(metadata_dir),
            database_path=database_path,
            split=split,
            time_key=processed_time_key,
            time_value=float(t_processed),
            neighborhood_threshold=neighborhood_threshold,
            spot_diameter=recommended_spot_diameter,
            spatial_key=spatial_key_used,
            expression_layer="counts",
            auto_neighborhood_threshold=False,
        )

    print(f"[3/3] Training edge predictor -> {edge_model_path}")
    edge_result = train_edge_predictor(
        data_name=data_name,
        adata_or_h5ad=adata_aligned,
        graph_input_dir=str(graph_input_dir),
        output_model_path=str(edge_model_path),
        epochs=edge_epochs,
        spatial_dim=align_config.spatial_dim,
        device=device,
    )
    sanitize_interaction_graph_uns(adata_aligned)
    adata_aligned.write_h5ad(aligned_h5ad_path)
    print(
        "[3/3] Edge predictor threshold selected from validation: "
        f"{edge_result['edge_predictor_threshold']:.2f}"
    )

    print(f"=== Done: outputs under {output_root} ===")
    return {
        "aligned_csv": str(aligned_csv_path),
        "aligned_h5ad": str(aligned_h5ad_path),
        "graph_input_dir": str(graph_input_dir),
        "metadata_dir": str(metadata_dir),
        "edge_model_path": str(edge_model_path),
        "edge_meta_path": str(edge_result["meta_path"]),
        "edge_predictor_threshold": float(edge_result["edge_predictor_threshold"]),
        "neighborhood_threshold": float(neighborhood_threshold),
        "recommended_spot_diameter": float(recommended_spot_diameter),
        "spatial_key": spatial_key_used,
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run project-level preprocessing pipeline.")
    p.add_argument("--data-name", required=True)
    p.add_argument("--h5ad-path", required=True)
    p.add_argument("--time-key", required=True)
    p.add_argument("--output-dir", default="results/")
    p.add_argument("--database-path", default="database/CellNEST_database.csv")
    p.add_argument("--split", type=int, default=0)
    p.add_argument("--edge-epochs", type=int, default=100)
    p.add_argument("--batch-indices", default=None, help="Comma-separated batch indices (optional).")
    p.add_argument(
        "--neighborhood-threshold",
        type=float,
        default=0.0,
        help="Global neighborhood threshold. <=0 means auto estimate from aligned spatial NN stats.",
    )
    p.add_argument("--spatial-key", default="spatial_aligned")
    p.add_argument("--device", default="cuda")

    # alignment knobs (keep simple and optional)
    p.add_argument("--spatial-dim", type=int, default=2)
    p.add_argument("--n-top-genes", type=int, default=2000)
    p.add_argument("--n-pcs", type=int, default=50)
    p.add_argument("--phase1-epochs", type=int, default=10000)
    p.add_argument("--phase2-epochs", type=int, default=500)
    return p


def main() -> None:
    args = _build_parser().parse_args()
    batch_indices = None
    if args.batch_indices:
        batch_indices = [int(x) for x in str(args.batch_indices).split(",") if x.strip()]

    cfg = AlignConfig(
        spatial_dim=int(args.spatial_dim),
        n_top_genes=int(args.n_top_genes),
        n_pcs=int(args.n_pcs),
        phase1_epochs=int(args.phase1_epochs),
        phase2_epochs=int(args.phase2_epochs),
    )

    run_preprocessing_pipeline(
        data_name=args.data_name,
        h5ad_path=args.h5ad_path,
        time_key=args.time_key,
        output_dir=args.output_dir,
        align_config=cfg,
        database_path=args.database_path,
        split=int(args.split),
        edge_epochs=int(args.edge_epochs),
        batch_indices=batch_indices,
        neighborhood_threshold=float(args.neighborhood_threshold),
        spatial_key=str(args.spatial_key),
        device=args.device,
    )


if __name__ == "__main__":
    main()
