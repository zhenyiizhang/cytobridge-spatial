#!/usr/bin/env python3
"""Render side-by-side baseline vs start-time cell-type knockout trajectory GIFs."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Sequence

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation, PillowWriter

from CytoBridge.tl import predict_labels_for_trajectories, simulate_sde_points_split, train_mlp_classifier
from CytoBridge.tl.downstream.checkpoint import load_dynamical_model_from_dir
from CytoBridge.tl.downstream.downstream_data import build_time_grid


def _slugify(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", str(text)).strip("_")


def _build_label_to_color(labels: Sequence[str]) -> dict[str, str]:
    unique_labels = list(dict.fromkeys([str(x) for x in labels]))
    cmap = plt.get_cmap("tab20")
    out: dict[str, str] = {}
    for i, lab in enumerate(unique_labels):
        rgb = cmap(i % cmap.N)[:3]
        out[str(lab)] = "#{:02x}{:02x}{:02x}".format(
            int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
        )
    return out


def _labels_to_colors(labels: Sequence[str], label_to_color: dict[str, str]) -> list[str]:
    return [label_to_color.get(str(x), "#888888") for x in labels]


def _render_comparison_gif(
    baseline_points: np.ndarray,
    knockout_points: np.ndarray,
    ts_points: Sequence[float],
    baseline_labels: Sequence[Sequence[str]],
    knockout_labels: Sequence[Sequence[str]],
    *,
    knocked_label: str,
    out_path: Path,
    label_to_color: dict[str, str],
    fps: int,
) -> None:
    d1, d2 = 0, 1
    base_frames = [np.asarray(baseline_points[i], dtype=float) for i in range(len(ts_points))]
    ko_frames = [np.asarray(knockout_points[i], dtype=float) for i in range(len(ts_points))]

    all_x = []
    all_y = []
    for pts in base_frames + ko_frames:
        all_x.append(pts[:, d1])
        all_y.append(pts[:, d2])
    x_all = np.concatenate(all_x)
    y_all = np.concatenate(all_y)
    x_min, x_max = float(np.min(x_all)), float(np.max(x_all))
    y_min, y_max = float(np.min(y_all)), float(np.max(y_all))
    x_pad = max(1e-6, (x_max - x_min) * 0.05)
    y_pad = max(1e-6, (y_max - y_min) * 0.05)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=150)
    ax_l, ax_r = axes
    ax_l.set_title("Baseline")
    ax_r.set_title(f"Knockout: {knocked_label}")
    for ax in axes:
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_aspect("equal")

    scat_l = ax_l.scatter([], [], s=4, alpha=0.85, linewidths=0)
    scat_r = ax_r.scatter([], [], s=4, alpha=0.85, linewidths=0)
    suptitle = fig.suptitle("")

    def _update(frame_idx: int):
        pts_l = base_frames[frame_idx]
        pts_r = ko_frames[frame_idx]
        lbl_l = baseline_labels[frame_idx]
        lbl_r = knockout_labels[frame_idx]

        n_l = min(len(pts_l), len(lbl_l))
        n_r = min(len(pts_r), len(lbl_r))
        pts_l = pts_l[:n_l]
        pts_r = pts_r[:n_r]
        lbl_l = lbl_l[:n_l]
        lbl_r = lbl_r[:n_r]

        scat_l.set_offsets(np.column_stack([pts_l[:, d1], pts_l[:, d2]]))
        scat_r.set_offsets(np.column_stack([pts_r[:, d1], pts_r[:, d2]]))
        scat_l.set_color(_labels_to_colors(lbl_l, label_to_color))
        scat_r.set_color(_labels_to_colors(lbl_r, label_to_color))
        suptitle.set_text(f"t = {ts_points[frame_idx]}")
        return scat_l, scat_r, suptitle

    anim = FuncAnimation(fig, _update, frames=len(ts_points), interval=max(1, int(1000 / max(1, fps))), blit=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(out_path), writer=PillowWriter(fps=max(1, int(fps))))
    plt.close(fig)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Render baseline-vs-knockout GIFs for each start-time cell type.")
    p.add_argument("--adata-h5ad", required=True, help="Input AnnData path.")
    p.add_argument("--model-dir", required=True, help="Model dir with ckpt/config.")
    p.add_argument("--out-dir", required=True, help="Output dir for knockout comparison GIFs.")
    p.add_argument("--label-key", default="bin_annotation", help="obs label key for classifier/knockout.")
    p.add_argument("--time-key", default="time_point_processed", help="obs time key.")
    p.add_argument("--obsm-key", default="X_latent", help="obsm latent key.")
    p.add_argument("--spatial-key", default="spatial_aligned", help="obsm spatial key.")
    p.add_argument("--subdivisions", type=int, default=8, help="Frames between adjacent observed time points.")
    p.add_argument("--fps", type=int, default=4, help="GIF fps.")
    p.add_argument("--n-samples", type=int, default=5000, help="Simulation sample size.")
    p.add_argument("--sigma", type=float, default=0.03, help="SDE sigma.")
    p.add_argument("--dt", type=float, default=0.01, help="SDE dt.")
    p.add_argument("--classifier-epochs", type=int, default=500, help="Classifier training epochs.")
    p.add_argument("--classifier-lr", type=float, default=1e-3, help="Classifier learning rate.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--device", default=None, help="cpu/cuda. Default auto.")
    p.add_argument(
        "--start-time-index",
        type=int,
        default=0,
        help="Start from observed time index (0-based in sorted observed times). Default: 0.",
    )
    p.add_argument(
        "--start-time-value",
        type=float,
        default=None,
        help="Optional explicit start time value. If set, overrides --start-time-index.",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    adata = ad.read_h5ad(args.adata_h5ad)
    if args.label_key not in adata.obs.columns:
        raise KeyError(f"adata.obs['{args.label_key}'] not found.")
    if args.obsm_key not in adata.obsm:
        raise KeyError(f"adata.obsm['{args.obsm_key}'] not found.")
    if args.spatial_key not in adata.obsm:
        raise KeyError(f"adata.obsm['{args.spatial_key}'] not found.")

    labels_all = adata.obs[args.label_key].astype(str).values
    label_to_color = _build_label_to_color(labels_all)

    time_vals = np.asarray(adata.obs[args.time_key].values, dtype=float)
    observed_times = sorted(np.unique(time_vals).tolist())
    if args.start_time_value is not None:
        start_t = float(args.start_time_value)
        candidates = [t for t in observed_times if np.isclose(t, start_t, rtol=0.0, atol=1e-9)]
        if not candidates:
            raise ValueError(f"--start-time-value={start_t} not found in observed times: {observed_times}")
        start_t = float(candidates[0])
        time_index = int(observed_times.index(start_t))
    else:
        time_index = int(args.start_time_index)
        if time_index < 0 or time_index >= len(observed_times):
            raise ValueError(f"--start-time-index={time_index} out of range [0, {len(observed_times)-1}]")
        start_t = float(observed_times[time_index])

    mask_start = np.isclose(time_vals, start_t, rtol=0.0, atol=1e-9)
    start_types = sorted(adata.obs.loc[mask_start, args.label_key].astype(str).unique().tolist())
    if len(start_types) == 0:
        raise ValueError(f"No cell types found at start time t={start_t}.")

    latent = np.asarray(adata.obsm[args.obsm_key], dtype=np.float32)
    spatial = np.asarray(adata.obsm[args.spatial_key], dtype=np.float32)
    dim = int(latent.shape[1] + spatial.shape[1])

    print(
        f"[info] device={device}, dim={dim}, start_time={start_t}, "
        f"start_index={time_index}, start_types={len(start_types)}"
    )
    print(f"[info] start-time cell types: {start_types}")

    loaded = load_dynamical_model_from_dir(model_dir=args.model_dir, dim=dim, device=device)
    model = loaded.model
    print(f"[info] loaded stages: weight={loaded.weight_stage}, score={loaded.score_stage}")

    _, ts_points_all = build_time_grid(adata=adata, time_key=args.time_key, subdivisions=int(args.subdivisions))
    ts_points = [float(t) for t in ts_points_all if (t > start_t or np.isclose(t, start_t, rtol=0.0, atol=1e-9))]
    if not ts_points:
        raise ValueError(f"No simulation grid points found at/after start time t={start_t}.")
    print(f"[info] simulated frames (from t={start_t}): {len(ts_points)}")

    print("[1/3] Train classifier once...")
    clf, le, acc = train_mlp_classifier(
        adata,
        label_col=args.label_key,
        time_key=args.time_key,
        obsm_key=args.obsm_key,
        spatial_key=args.spatial_key,
        concat_spatial=True,
        epochs=int(args.classifier_epochs),
        hidden_size=128,
        lr=float(args.classifier_lr),
        seed=int(args.seed),
        device=device,
        include_time_feature=True,
    )
    print(f"[info] classifier_acc={acc:.4f}")

    print("[2/3] Simulate baseline...")
    baseline_points = simulate_sde_points_split(
        adata=adata,
        model=model,
        dim=dim,
        time_index=time_index,
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
    baseline_labels = predict_labels_for_trajectories(
        sde_points=baseline_points,
        ts_points=ts_points,
        model=clf,
        label_encoder=le,
        feature_dim=dim,
        device=device,
        knn_neighbors=10,
        include_time_feature=True,
    )

    print("[3/3] Run knockouts and render comparison GIFs...")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "baseline_sde_points.npy", baseline_points, allow_pickle=True)

    for idx, cell_type in enumerate(start_types, start=1):
        print(f"[{idx}/{len(start_types)}] knockout: {cell_type}")
        knockout_points = simulate_sde_points_split(
            adata=adata,
            model=model,
            dim=dim,
            time_index=time_index,
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
            exclude_cell_types=[cell_type],
            annotation_key=args.label_key,
        )
        knockout_labels = predict_labels_for_trajectories(
            sde_points=knockout_points,
            ts_points=ts_points,
            model=clf,
            label_encoder=le,
            feature_dim=dim,
            device=device,
            knn_neighbors=10,
            include_time_feature=True,
        )

        safe_name = _slugify(cell_type)
        np.save(out_dir / f"knockout_t{start_t:g}_{safe_name}_sde_points.npy", knockout_points, allow_pickle=True)
        gif_path = out_dir / f"compare_baseline_vs_knockout_{safe_name}.gif"
        _render_comparison_gif(
            baseline_points=baseline_points,
            knockout_points=knockout_points,
            ts_points=ts_points,
            baseline_labels=baseline_labels,
            knockout_labels=knockout_labels,
            knocked_label=cell_type,
            out_path=gif_path,
            label_to_color=label_to_color,
            fps=int(args.fps),
        )
        print(f"[done] {gif_path}")

    print(f"[all done] outputs under: {out_dir}")


if __name__ == "__main__":
    main()
