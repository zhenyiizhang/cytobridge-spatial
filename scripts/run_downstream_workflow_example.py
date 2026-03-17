#!/usr/bin/env python3
"""Run a complete downstream analysis workflow using the CytoBridge package.

This script is a fixed-order, end-to-end example that mirrors the ST-1104
downstream notebooks:
1) Interpolation via split-SDE (observed + interpolated sub-steps)
2) Classifier training + trajectory label prediction
3) Velocity decomposition + scVelo stream plots + intrinsic-vs-interaction correlation
4) Growth maps + gene-velocity embeddings
5) Attention-based communication + Sankey + 3D spatiotemporal plot
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd


def _find_first(path: Path, patterns: Sequence[str]) -> Optional[Path]:
    for pat in patterns:
        hits = sorted(path.glob(pat))
        if hits:
            return hits[0]
    return None


def _load_label_to_color(labels: Sequence[str]) -> Dict[str, str]:
    import matplotlib.pyplot as plt

    unique_labels = list(dict.fromkeys([str(x) for x in labels]))
    cmap = plt.get_cmap("tab20")
    label_to_color = {}
    for idx, lab in enumerate(unique_labels):
        rgb = cmap(idx % cmap.N)[:3]
        label_to_color[str(lab)] = "#{:02x}{:02x}{:02x}".format(
            int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
        )
    return label_to_color


def _coerce_feature_matrix_from_adata(
    adata,
    *,
    obsm_key: str,
    spatial_key: str,
    concat_spatial: Optional[bool],
) -> tuple[np.ndarray, int]:
    if obsm_key in adata.obsm:
        latent = np.asarray(adata.obsm[obsm_key], dtype=np.float32)
    else:
        latent = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
        latent = np.asarray(latent, dtype=np.float32)

    use_spatial = bool(concat_spatial) if concat_spatial is not None else (spatial_key in adata.obsm)
    if not use_spatial:
        return latent, 0

    if spatial_key not in adata.obsm:
        raise KeyError(f"concat_spatial=True but adata.obsm['{spatial_key}'] is missing.")
    spatial = np.asarray(adata.obsm[spatial_key], dtype=np.float32)
    if spatial.shape[0] != latent.shape[0]:
        raise ValueError(
            f"Row mismatch between '{spatial_key}' ({spatial.shape[0]}) and "
            f"'{obsm_key}' ({latent.shape[0]})."
        )
    return np.hstack((spatial, latent)).astype(np.float32), int(spatial.shape[1])


def _merge_annotation_into_adata(
    adata,
    annotation_csv: str,
    *,
    annotation_key: str,
    cell_id_column: str = "cell_id",
) -> None:
    anno_df = pd.read_csv(annotation_csv)
    if annotation_key not in anno_df.columns:
        raise KeyError(f"annotation csv missing required column '{annotation_key}'")

    if cell_id_column in anno_df.columns:
        label_map = anno_df.set_index(cell_id_column)[annotation_key].astype(str)
        mapped = adata.obs_names.to_series().map(label_map)
        if mapped.isna().any():
            raise ValueError("Annotation merge by cell_id produced missing labels.")
        adata.obs[annotation_key] = mapped.values
        return

    if len(anno_df) != adata.n_obs:
        raise ValueError(
            "Annotation row count mismatch. Provide a `cell_id` column in annotation CSV "
            "or identical row order/length."
        )
    adata.obs[annotation_key] = anno_df[annotation_key].astype(str).values


def _parse_comma_separated_tokens(raw: Optional[str]) -> list[str]:
    if raw is None:
        return []
    return [x.strip() for x in str(raw).split(",") if x.strip()]


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CytoBridge downstream workflow example (fixed pipeline).")
    p.add_argument("--model-dir", required=True, help="Results directory containing config.yaml and stage checkpoints.")
    p.add_argument("--out-dir", default=None, help="Output directory (default: <model-dir>/downstream_workflow).")
    p.add_argument(
        "--aligned-h5ad",
        default=None,
        help="Aligned h5ad (recommended): use adata.obs[time_key] + adata.obsm[obsm_key] as aligned features.",
    )
    p.add_argument(
        "--time-key",
        default=None,
        help="Time column in aligned h5ad adata.obs (e.g. 'samples', 'time', 'Batch'). If omitted, auto-detect.",
    )
    p.add_argument(
        "--obsm-key",
        default="X_latent",
        help="Feature matrix key in aligned h5ad adata.obsm (default: X_latent; fallback to adata.X).",
    )
    p.add_argument(
        "--spatial-key",
        default="spatial_aligned",
        help="Spatial key in aligned h5ad adata.obsm (default: spatial_aligned).",
    )
    p.add_argument(
        "--concat-spatial",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Whether to concatenate spatial_key with obsm_key for h5ad input. "
            "Default:auto (enabled when spatial_key exists)."
        ),
    )
    p.add_argument("--annotation-csv", default=None, help="Optional CSV providing an Annotation column aligned to rows.")
    p.add_argument("--annotation-key", default="Annotation", help="Annotation column name (default: Annotation).")
    p.add_argument("--device", default=None, help="Torch device override (cpu/cuda).")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")

    # Interpolation / SDE
    p.add_argument("--sde-samples", type=int, default=5000, help="Max particles for SDE simulation.")
    p.add_argument("--split-dt", type=float, default=0.01, help="Split-SDE dt (default: 0.01).")
    p.add_argument("--split-sigma", type=float, default=0.03, help="Split-SDE sigma (default: 0.03).")
    p.add_argument(
        "--time-subdivisions",
        type=int,
        default=2,
        help=(
            "Sub-interval count between adjacent observed times for trajectory simulation "
            "(2=midpoints, larger means denser/continuous trajectories)."
        ),
    )
    p.add_argument(
        "--trajectory-gif",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to render trajectory animation GIF after classifier prediction.",
    )
    p.add_argument("--trajectory-gif-fps", type=int, default=4, help="Trajectory GIF fps (default: 4).")
    p.add_argument(
        "--exclude-cell-indices",
        default=None,
        help=(
            "Optional perturbation: comma-separated cell indices (0-based) or obs_names to remove "
            "before simulation starts, e.g. '1,5,12' or 'cellA,cellB'."
        ),
    )
    p.add_argument(
        "--exclude-cell-types",
        default=None,
        help=(
            "Optional perturbation: comma-separated cell types to remove before simulation starts, "
            "using --annotation-key labels."
        ),
    )

    # Classifier
    p.add_argument("--classifier-epochs", type=int, default=200, help="Classifier epochs.")
    p.add_argument("--classifier-hidden", type=int, default=128, help="Classifier hidden size (kept for API parity).")
    p.add_argument("--knn-neighbors", type=int, default=50, help="KNN refinement neighbors.")

    # 3D
    p.add_argument("--skip-3d", action="store_true", help="Skip 3D spatiotemporal plot.")
    p.add_argument(
        "--use-simulated-for-observed",
        dest="use_real_for_observed",
        action="store_false",
        help="Use simulated SDE points even for observed timepoints (default uses real data).",
    )
    p.set_defaults(use_real_for_observed=True)
    p.add_argument("--comm-focus-label", default=None, help="Focus communication edges on a cell type.")
    p.add_argument("--edge-top-k", type=int, default=6, help="Top-K communication edges per timepoint.")
    p.add_argument("--fate-focus-label", default=None, help="Focus fate ribbons on a cell type.")
    p.add_argument("--fate-min-flow", type=float, default=10.0, help="Minimum fate ribbon flow count.")
    p.add_argument("--focus-anchor-label", default=None, help="Enable focus-anchor mode for 3D with this label.")
    return p


def main() -> None:
    args = _build_argparser().parse_args()

    model_dir = Path(args.model_dir)
    out_dir = Path(args.out_dir) if args.out_dir else (model_dir / "downstream_workflow")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- device ----
    import torch

    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    from CytoBridge.tl.downstream.downstream_data import build_time_grid, infer_time_key, parse_time_value

    # ---- load aligned data (AnnData-first) ----
    aligned_h5ad = Path(args.aligned_h5ad) if args.aligned_h5ad else None
    if aligned_h5ad is None:
        aligned_h5ad = _find_first(model_dir, ["*aligned*.h5ad", "*.h5ad"])

    if aligned_h5ad is None or (not aligned_h5ad.exists()):
        raise FileNotFoundError(
            "Aligned h5ad not found. Please provide --aligned-h5ad, "
            "or place '*aligned*.h5ad' under model-dir."
        )

    adata_obj = None
    import anndata as ad

    adata_obj = ad.read_h5ad(aligned_h5ad)
    if args.annotation_key not in adata_obj.obs.columns and args.annotation_csv:
        _merge_annotation_into_adata(
            adata_obj,
            args.annotation_csv,
            annotation_key=args.annotation_key,
            cell_id_column="cell_id",
        )

    feature_matrix, _ = _coerce_feature_matrix_from_adata(
        adata_obj,
        obsm_key=args.obsm_key,
        spatial_key=args.spatial_key,
        concat_spatial=args.concat_spatial,
    )
    dim = int(feature_matrix.shape[1])
    print(f"[input] rows={adata_obj.n_obs}, feature_dim={dim}, aligned_h5ad=True")

    # ---- load model ----
    from CytoBridge.tl.downstream.checkpoint import load_dynamical_model_from_dir

    try:
        loaded = load_dynamical_model_from_dir(model_dir=model_dir, dim=dim, device=device)
    except RuntimeError as exc:
        if "size mismatch" in str(exc):
            raise RuntimeError(
                "Checkpoint input dim does not match downstream features. "
                "For spatial training, use concatenated spatial+latent input "
                "(try --concat-spatial --spatial-key spatial_aligned). "
                f"Current inferred dim={dim}."
            ) from exc
        raise
    model = loaded.model

    # ---- time axis ----
    observed_times, ts_points = build_time_grid(
        adata=adata_obj,
        time_key=args.time_key,
        subdivisions=int(args.time_subdivisions),
    )
    print(
        f"[time] observed_points={len(observed_times)}, simulated_points={len(ts_points)}, "
        f"subdivisions={int(args.time_subdivisions)}"
    )

    # ------------------------------------------------------------------
    # 01) Interpolation (split-SDE)
    # ------------------------------------------------------------------
    from CytoBridge.tl import simulate_sde_points_split
    from CytoBridge.pl import plot_sde_vs_real_from_adata, plot_trajectory_gif

    interp_dir = out_dir / "01_interpolation"
    interp_dir.mkdir(exist_ok=True)

    sde_points = simulate_sde_points_split(
        adata=adata_obj,
        model=model,
        dim=dim,
        time_index=0,
        n_samples=int(args.sde_samples),
        ts_points=ts_points,
        dt=float(args.split_dt),
        sigma=float(args.split_sigma),
        interaction_m=1024,
        device=device,
        time_key=args.time_key,
        obsm_key=args.obsm_key,
        spatial_key=args.spatial_key,
        concat_spatial=args.concat_spatial,
        exclude_indices=_parse_comma_separated_tokens(args.exclude_cell_indices),
        exclude_cell_types=_parse_comma_separated_tokens(args.exclude_cell_types),
        annotation_key=args.annotation_key,
    )
    np.save(interp_dir / "sde_points_split.npy", sde_points, allow_pickle=True)

    plot_sde_vs_real_from_adata(
        adata=adata_obj,
        sde_points=sde_points,
        time_values=ts_points,
        dim_pairs=((0, 1),),
        out_prefix=str(interp_dir / "sde_vs_real"),
        time_key=args.time_key,
        obsm_key=args.obsm_key,
        spatial_key=args.spatial_key,
        concat_spatial=args.concat_spatial,
    )

    # ------------------------------------------------------------------
    # 02) Classifier + predicted labels
    # ------------------------------------------------------------------
    predicted_labels_list = None
    label_to_color = None
    clf_dir = out_dir / "02_classifier"
    clf_dir.mkdir(exist_ok=True)

    if args.annotation_key in adata_obj.obs.columns:
        from CytoBridge.tl import predict_labels_for_trajectories, train_mlp_classifier

        clf, label_enc, acc = train_mlp_classifier(
            adata_obj,
            label_col=args.annotation_key,
            time_key=args.time_key,
            obsm_key=args.obsm_key,
            spatial_key=args.spatial_key,
            concat_spatial=args.concat_spatial,
            hidden_size=int(args.classifier_hidden),
            epochs=int(args.classifier_epochs),
            seed=int(args.seed),
            device=device,
        )
        with (clf_dir / "classifier_metrics.json").open("w", encoding="utf-8") as f:
            json.dump({"acc": acc}, f, indent=2)

        predicted_labels_list = predict_labels_for_trajectories(
            sde_points=sde_points,
            ts_points=ts_points,
            model=clf,
            label_encoder=label_enc,
            feature_dim=dim,
            device=device,
            knn_neighbors=int(args.knn_neighbors),
        )
        with (clf_dir / "predicted_labels.json").open("w", encoding="utf-8") as f:
            json.dump([list(map(str, x)) for x in predicted_labels_list], f)

        label_to_color = _load_label_to_color(adata_obj.obs[args.annotation_key].astype(str).values)
        with (clf_dir / "label_to_color.json").open("w", encoding="utf-8") as f:
            json.dump(label_to_color, f, indent=2)

        if args.trajectory_gif:
            gif_path = interp_dir / "trajectory_predicted_labels.gif"
            plot_trajectory_gif(
                sde_points=sde_points,
                time_values=ts_points,
                labels_list=predicted_labels_list,
                label_to_color=label_to_color,
                out_path=str(gif_path),
                dim_pair=(0, 1),
                fps=int(args.trajectory_gif_fps),
            )

    # ------------------------------------------------------------------
    # 03) Velocity plots
    # ------------------------------------------------------------------
    from CytoBridge.tl import compute_velocity_components_from_adata
    from CytoBridge.pl import (
        plot_intrinsic_interaction_direction_correlation_from_adata,
        plot_velocity_component,
    )

    vel_dir = out_dir / "03_velocity"
    vel_dir.mkdir(exist_ok=True)

    vel_comp = compute_velocity_components_from_adata(
        adata=adata_obj,
        model=model,
        dim=dim,
        interaction_m=1024,
        device=device,
        time_key=args.time_key,
        obsm_key=args.obsm_key,
        spatial_key=args.spatial_key,
        concat_spatial=args.concat_spatial,
        write_to_adata=True,
        reuse_if_present=True,
    )

    resolved_time_key = infer_time_key(adata_obj.obs, preferred=args.time_key)
    obs_times = np.asarray([parse_time_value(v) for v in adata_obj.obs[resolved_time_key].values], dtype=np.float64)
    if args.spatial_key in adata_obj.obsm:
        all_coords = np.asarray(adata_obj.obsm[args.spatial_key], dtype=np.float32)
    else:
        all_coords = np.asarray(vel_comp["features"][:, :2], dtype=np.float32)

    all_labels = (
        adata_obj.obs[args.annotation_key].astype(str).values
        if args.annotation_key in adata_obj.obs.columns
        else None
    )
    vel_drift = np.asarray(adata_obj.obsm["velocity_model"], dtype=np.float32)
    vel_interaction = (
        np.asarray(adata_obj.obsm["interaction_model"], dtype=np.float32)
        if "interaction_model" in adata_obj.obsm
        else np.zeros_like(vel_drift)
    )
    vel_full = np.asarray(adata_obj.obsm["full_drift_model"], dtype=np.float32)
    latent_matrix = np.asarray(
        adata_obj.obsm[args.obsm_key]
        if args.obsm_key in adata_obj.obsm
        else (adata_obj.X.toarray() if hasattr(adata_obj.X, "toarray") else np.asarray(adata_obj.X)),
        dtype=np.float32,
    )
    vel_drift_latent = (
        np.asarray(adata_obj.obsm["velocity_latent"], dtype=np.float32)
        if "velocity_latent" in adata_obj.obsm
        else vel_drift
    )
    vel_interaction_latent = (
        np.asarray(adata_obj.obsm["interaction_latent"], dtype=np.float32)
        if "interaction_latent" in adata_obj.obsm
        else vel_interaction
    )
    vel_full_latent = (
        np.asarray(adata_obj.obsm["full_drift_latent"], dtype=np.float32)
        if "full_drift_latent" in adata_obj.obsm
        else vel_full
    )

    for idx, t in enumerate(observed_times):
        mask = np.isclose(obs_times, float(t), rtol=0.0, atol=1e-9)
        if not np.any(mask):
            continue
        coords = all_coords[mask, :2]
        labels_t = all_labels[mask] if all_labels is not None else None

        plot_velocity_component(
            coords=coords,
            velocity=vel_drift[mask, :2],
            labels=labels_t,
            label_to_color=label_to_color,
            title=f"Spatial intrinsic (t={t})",
            out_path=str(vel_dir / f"velocity_scvelo_spatial_intrinsic_t{idx}.svg"),
            basis="spatial",
            show_legend=True,
        )
        plot_velocity_component(
            coords=coords,
            velocity=vel_interaction[mask, :2],
            labels=labels_t,
            label_to_color=label_to_color,
            title=f"Spatial interaction (t={t})",
            out_path=str(vel_dir / f"velocity_scvelo_spatial_interaction_t{idx}.svg"),
            basis="spatial",
            show_legend=True,
        )
        plot_velocity_component(
            coords=coords,
            velocity=vel_full[mask, :2],
            labels=labels_t,
            label_to_color=label_to_color,
            title=f"Spatial full (t={t})",
            out_path=str(vel_dir / f"velocity_scvelo_spatial_full_t{idx}.svg"),
            basis="spatial",
            show_legend=True,
        )

        # Gene velocity: use full latent vectors, projected on spatial coordinates.
        if latent_matrix.shape[1] >= 2:
            plot_velocity_component(
                coords=coords,
                velocity=vel_drift_latent[mask],
                feature_matrix=latent_matrix[mask],
                labels=labels_t,
                label_to_color=label_to_color,
                title=f"Gene intrinsic (t={t})",
                out_path=str(vel_dir / f"velocity_scvelo_gene_intrinsic_t{idx}.svg"),
                basis="spatial",
                show_legend=True,
            )
            plot_velocity_component(
                coords=coords,
                velocity=vel_interaction_latent[mask],
                feature_matrix=latent_matrix[mask],
                labels=labels_t,
                label_to_color=label_to_color,
                title=f"Gene interaction (t={t})",
                out_path=str(vel_dir / f"velocity_scvelo_gene_interaction_t{idx}.svg"),
                basis="spatial",
                show_legend=True,
            )
            plot_velocity_component(
                coords=coords,
                velocity=vel_full_latent[mask],
                feature_matrix=latent_matrix[mask],
                labels=labels_t,
                label_to_color=label_to_color,
                title=f"Gene full (t={t})",
                out_path=str(vel_dir / f"velocity_scvelo_gene_full_t{idx}.svg"),
                basis="spatial",
                show_legend=True,
            )

    corr_spatial_key = args.spatial_key
    if corr_spatial_key not in adata_obj.obsm:
        corr_spatial_key = "spatial_for_velocity_plot"
        adata_obj.obsm[corr_spatial_key] = all_coords

    plot_intrinsic_interaction_direction_correlation_from_adata(
        adata=adata_obj,
        out_path=str(vel_dir / "velocity_spatial_direction_correlation_spatial.svg"),
        time_key=args.time_key,
        spatial_key=corr_spatial_key,
        drift_key="velocity_model",
        interaction_key="interaction_model",
    )

    # ------------------------------------------------------------------
    # 04) Growth + gene velocity embeddings
    # ------------------------------------------------------------------
    from CytoBridge.pl import gene_velocity_embeddings_from_adata, plot_growth_per_time_from_adata

    gg_dir = out_dir / "04_growth_gene"
    gg_dir.mkdir(exist_ok=True)
    plot_growth_per_time_from_adata(
        adata=adata_obj,
        dim=dim,
        model=model,
        out_dir=str(gg_dir / "growth"),
        device=device,
        time_key=args.time_key,
        obsm_key=args.obsm_key,
        spatial_key=args.spatial_key,
        concat_spatial=args.concat_spatial,
    )
    gene_velocity_embeddings_from_adata(
        adata=adata_obj,
        dim=dim,
        model=model,
        out_dir=str(gg_dir / "gene_velocity"),
        label_to_color=label_to_color,
        device=device,
        time_key=args.time_key,
        obsm_key=args.obsm_key,
        spatial_key=args.spatial_key,
        concat_spatial=args.concat_spatial,
        annotation_column=args.annotation_key,
    )

    # ------------------------------------------------------------------
    # 05) Communication + sankey + 3D
    # ------------------------------------------------------------------
    comm_dir = out_dir / "05_communication"
    comm_dir.mkdir(exist_ok=True)
    attn_dir = comm_dir / "attention"
    attn_dir.mkdir(exist_ok=True)

    from CytoBridge.tl import analyze_attention_by_celltype, save_interpolated_attention
    from CytoBridge.pl import plot_3d_spatial_sankey_style, plot_sankey

    # Build per-time AnnData (real for observed, simulated for unseen).
    import anndata as ad

    observed_set = set([float(x) for x in observed_times])
    adata_dict = {}
    feature_matrix_obs, _ = _coerce_feature_matrix_from_adata(
        adata_obj,
        obsm_key=args.obsm_key,
        spatial_key=args.spatial_key,
        concat_spatial=args.concat_spatial,
    )
    resolved_time_key = infer_time_key(adata_obj.obs, preferred=args.time_key)
    obs_times = np.asarray([parse_time_value(v) for v in adata_obj.obs[resolved_time_key].values], dtype=np.float64)
    obs_labels = (
        adata_obj.obs[args.annotation_key].astype(str).values
        if args.annotation_key in adata_obj.obs.columns
        else np.array(["Unknown"] * adata_obj.n_obs, dtype=object)
    )
    for t_idx, t in enumerate(ts_points):
        t_key = str(t)
        if float(t) in observed_set and args.use_real_for_observed:
            mask_t = np.isclose(obs_times, float(t), rtol=0.0, atol=1e-9)
            X = feature_matrix_obs[mask_t].astype(np.float32)
            labels_t = obs_labels[mask_t].astype(str)
        else:
            X = np.asarray(sde_points[t_idx], dtype=np.float32)
            if predicted_labels_list is not None:
                labels_t = np.asarray(predicted_labels_list[t_idx]).astype(str)
                if len(labels_t) != len(X):
                    # Best-effort alignment if sizes differ
                    min_len = min(len(labels_t), len(X))
                    X = X[:min_len]
                    labels_t = labels_t[:min_len]
            else:
                labels_t = np.array(["Unknown"] * len(X), dtype=object)

        adata_t = ad.AnnData(X=X)
        adata_t.obsm["spatial"] = X[:, :2]
        adata_t.obs[args.annotation_key] = labels_t
        adata_dict[t_key] = adata_t

    all_time_communications = {}
    for t in ts_points:
        key = str(t)
        adata_t = adata_dict[key]
        attn_out = save_interpolated_attention(
            adata=adata_t,
            time_value=float(t),
            model=model,
            device=device,
            out_dir=str(attn_dir),
        )
        comm = analyze_attention_by_celltype(
            edge_index=attn_out["edge_index"],
            attn=attn_out["attn_mean"],
            labels=adata_t.obs[args.annotation_key].values,
            spatial_coord=adata_t.obsm["spatial"],
            time_title=key,
            remove_self_loop=True,
            winsor_quantile=0.995,
            distance_bins=None,
            n_permutations=0,
            show_plots=False,
        )
        all_time_communications[key] = comm

    with (comm_dir / "all_time_communications.pkl").open("wb") as f:
        pickle.dump(all_time_communications, f)

    if predicted_labels_list is not None:
        sankey_path = comm_dir / "lineage_sankey.html"
        plot_sankey(
            predicted_labels_list=predicted_labels_list,
            out_html=str(sankey_path),
            time_keys=[str(x) for x in ts_points],
            show_time_axis=True,
            title="Cell lineage Sankey (predicted)",
            label_to_color=label_to_color,
        )

    if not args.skip_3d and predicted_labels_list is not None and label_to_color is not None:
        html_path = comm_dir / "spatiotemporal_3d.html"
        plot_3d_spatial_sankey_style(
            adata_dict=adata_dict,
            all_time_communications=all_time_communications,
            time_keys=[str(x) for x in ts_points],
            label_to_color=label_to_color,
            predicted_labels_list=predicted_labels_list,
            spatial_key="spatial",
            annotation_key=args.annotation_key,
            z_spacing=3.8,
            intra_threshold=0.0,
            edge_focus_celltype=args.comm_focus_label,
            edge_top_k=int(args.edge_top_k) if args.edge_top_k else None,
            ribbon_min_count=float(args.fate_min_flow) if args.fate_min_flow is not None else None,
            ribbon_focus_celltype=args.fate_focus_label,
            focus_anchor_label=args.focus_anchor_label,
            background_color="white",
            font_color="#1a1a1a",
            show_slice_border=True,
            slice_border_width=5,
            observed_time_points=[float(x) for x in observed_times],
            generated_time_points=[float(x) for x in ts_points if float(x) not in observed_set],
            out_html=str(html_path),
            show_time_axis=False,
            show_legend=False,
            show_title=False,
            width=1200,
            height=900,
        )

    # ---- manifest ----
    manifest = {
        "model_dir": str(model_dir),
        "aligned_h5ad": str(aligned_h5ad) if aligned_h5ad is not None else None,
        "dim": dim,
        "device": device,
        "weight_stage": loaded.weight_stage,
        "score_stage": loaded.score_stage,
        "out_dir": str(out_dir),
        "ts_points": [float(x) for x in ts_points],
        "perturbation": {
            "exclude_cell_indices": _parse_comma_separated_tokens(args.exclude_cell_indices),
            "exclude_cell_types": _parse_comma_separated_tokens(args.exclude_cell_types),
            "annotation_key": args.annotation_key,
        },
    }
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[done] outputs -> {out_dir}")


if __name__ == "__main__":
    main()
