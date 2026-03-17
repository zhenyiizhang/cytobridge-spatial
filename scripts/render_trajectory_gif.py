#!/usr/bin/env python3
"""Render trajectory GIF from trained CytoBridge model and AnnData."""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import torch

from CytoBridge.pl import plot_trajectory_gif
from CytoBridge.tl import predict_labels_for_trajectories, simulate_sde_points_split, train_mlp_classifier
from CytoBridge.tl.downstream.checkpoint import load_dynamical_model_from_dir
from CytoBridge.tl.downstream.downstream_data import build_time_grid


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Render trajectory GIF with predicted labels.")
    p.add_argument("--adata-h5ad", required=True, help="Input AnnData path.")
    p.add_argument("--model-dir", required=True, help="Model directory with checkpoints.")
    p.add_argument("--out-dir", required=True, help="Output directory.")
    p.add_argument("--label-key", default="bin_annotation", help="Label column in adata.obs for coloring.")
    p.add_argument("--time-key", default="time_point_processed", help="Time key in adata.obs.")
    p.add_argument("--obsm-key", default="X_latent", help="Feature embedding key in adata.obsm.")
    p.add_argument("--spatial-key", default="spatial_aligned", help="Spatial key in adata.obsm.")
    p.add_argument("--subdivisions", type=int, default=8, help="Subdivisions between observed time points.")
    p.add_argument("--fps", type=int, default=8, help="GIF frame rate.")
    p.add_argument("--n-samples", type=int, default=5000, help="Number of particles to simulate.")
    p.add_argument("--sigma", type=float, default=0.03, help="SDE sigma.")
    p.add_argument("--dt", type=float, default=0.01, help="SDE dt.")
    p.add_argument("--classifier-epochs", type=int, default=120, help="Classifier training epochs.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--device", default=None, help="cpu/cuda. Default: auto.")
    p.add_argument(
        "--gif-name",
        default="trajectory_predicted_bin_annotation_dense.gif",
        help="Output GIF filename.",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    adata_path = Path(args.adata_h5ad)
    model_dir = Path(args.model_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/6] Loading AnnData...")
    adata = ad.read_h5ad(adata_path)
    if args.label_key not in adata.obs.columns:
        raise KeyError(f"adata.obs['{args.label_key}'] not found.")
    if args.obsm_key not in adata.obsm:
        raise KeyError(f"adata.obsm['{args.obsm_key}'] not found.")
    if args.spatial_key not in adata.obsm:
        raise KeyError(f"adata.obsm['{args.spatial_key}'] not found.")

    latent = np.asarray(adata.obsm[args.obsm_key], dtype=np.float32)
    spatial = np.asarray(adata.obsm[args.spatial_key], dtype=np.float32)
    dim = int(latent.shape[1] + spatial.shape[1])
    print(f"[info] n_obs={adata.n_obs}, dim={dim}, device={device}")

    print("[2/6] Loading model...")
    loaded = load_dynamical_model_from_dir(model_dir=str(model_dir), dim=dim, device=device)
    model = loaded.model

    print("[3/6] Building dense time grid...")
    observed_times, ts_points = build_time_grid(
        adata=adata,
        time_key=args.time_key,
        subdivisions=int(args.subdivisions),
    )
    print(f"[info] observed={len(observed_times)} points, simulated frames={len(ts_points)}")

    print("[4/6] Running split-SDE simulation...")
    sde_points = simulate_sde_points_split(
        adata=adata,
        model=model,
        dim=dim,
        time_index=0,
        n_samples=int(args.n_samples),
        ts_points=ts_points,
        dt=float(args.dt),
        sigma=float(args.sigma),
        interaction_m=1024,
        device=device,
        time_key=args.time_key,
        obsm_key=args.obsm_key,
        spatial_key=args.spatial_key,
        concat_spatial=True,
    )
    sde_path = out_dir / f"sde_points_split_sub{int(args.subdivisions)}.npy"
    np.save(sde_path, sde_points, allow_pickle=True)

    print("[5/6] Training classifier and predicting labels...")
    clf, le, acc = train_mlp_classifier(
        adata,
        label_col=args.label_key,
        time_key=args.time_key,
        obsm_key=args.obsm_key,
        spatial_key=args.spatial_key,
        concat_spatial=True,
        epochs=int(args.classifier_epochs),
        hidden_size=128,
        seed=int(args.seed),
        device=device,
        include_time_feature=True,
    )
    print(f"[info] classifier_acc={acc:.4f}")

    labels_list = predict_labels_for_trajectories(
        sde_points=sde_points,
        ts_points=ts_points,
        model=clf,
        label_encoder=le,
        feature_dim=dim,
        device=device,
        knn_neighbors=50,
        include_time_feature=True,
    )

    print("[6/6] Rendering GIF...")
    gif_path = out_dir / args.gif_name
    plot_trajectory_gif(
        sde_points=sde_points,
        time_values=ts_points,
        labels_list=labels_list,
        label_to_color=None,
        out_path=str(gif_path),
        dim_pair=(0, 1),
        fps=int(args.fps),
    )
    print(f"[done] gif={gif_path}")
    print(f"[done] sde={sde_path}")


if __name__ == "__main__":
    main()
