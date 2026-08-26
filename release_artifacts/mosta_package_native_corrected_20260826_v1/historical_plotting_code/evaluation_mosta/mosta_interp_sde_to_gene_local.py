#!/usr/bin/env python3
"""Generate interpolated MOSTA timepoint in gene space via split-SDE + PCA backprojection.

Pipeline:
1) split-SDE in latent/PCA space (x1..x52; x1/x2 are spatial, x3..x52 are PC1..PC50),
2) classifier prediction for interpolated labels,
3) backproject PCs to gene space using `mosta_pca_components_with_gene_names.csv`,
4) save interpolated AnnData with gene-space `.X` (default: direct reconstructed log1p) + spatial coordinates.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from evaluation.arista_code.arista_helpers import (
    load_config,
    load_models,
    predict_labels_for_trajectories,
    simulate_sde_points_split,
    train_mlp_classifier,
)


DEFAULT_CONFIG = "config/mosta_config.yaml"
DEFAULT_DATA_CSV = "evaluation/mosta/data/mosta_four_time_with_celltype_refined.csv"
DEFAULT_PCA_COMPONENTS = "mosta_pca_components_with_gene_names.csv"
DEFAULT_OUTPUT_DIR = "results/mosta_interp_0p5"


def _parse_csv_floats(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _sample_observed_x0(
    df: pd.DataFrame,
    *,
    time_value: float,
    feature_cols: Sequence[str],
    label_col: Optional[str] = None,
    n_samples_cap: Optional[int],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    subset = df[df["samples"] == float(time_value)]
    X = subset[list(feature_cols)].values.astype(np.float32)
    if label_col and label_col in subset.columns:
        labels = subset[label_col].astype(str).values
    else:
        labels = np.array([], dtype=str)
    if n_samples_cap is None:
        return X, labels
    cap = int(n_samples_cap)
    if cap <= 0:
        raise ValueError("--n-samples must be > 0")
    if X.shape[0] <= cap:
        return X, labels
    idx = rng.choice(X.shape[0], size=cap, replace=False)
    if labels.shape[0] == 0:
        return X[idx], labels
    return X[idx], labels[idx]


def _simulate_sde_points_split_from_x0(
    *,
    x0: np.ndarray,
    f_net,
    score_net,
    ts_points: Sequence[float],
    dt: float,
    sigma: float,
    growth_alpha: float,
    interaction_m: int,
    device: str,
    verbose: bool = True,
) -> np.ndarray:
    from DeepRUOT.interaction import cal_interaction, euler_sdeint_split

    if len(ts_points) < 1:
        raise ValueError("ts_points must be non-empty")

    x0_t = torch.tensor(np.asarray(x0, dtype=np.float32), device=device)
    lnw0 = torch.log(torch.ones(x0_t.shape[0], 1, device=device) / x0_t.shape[0])
    initial_state = (x0_t, lnw0)

    class SDE(torch.nn.Module):
        noise_type = "diagonal"
        sde_type = "ito"

        def __init__(self, ode_drift, g, score, interaction, sigma):
            super().__init__()
            self.drift = ode_drift
            self.score = score
            self.sigma = sigma
            self.interaction = interaction
            self.g_net = g

        def f(self, t, y):
            z, lnw = y
            with torch.no_grad():
                drift = self.drift(t, z)
                dlnw = self.g_net(t, z) * growth_alpha
                net_forces = cal_interaction(z, lnw, self.interaction, t, m=interaction_m)
            t_expand = t.expand(z.shape[0], 1)
            score_grad = self.score.compute_gradient(t_expand, z)
            return (drift + score_grad + net_forces, dlnw)

        def g(self, t, y):
            return torch.ones_like(y) * self.sigma

    if verbose:
        t_min = float(min(ts_points))
        t_max = float(max(ts_points))
        est_steps = int(round((t_max - t_min) / dt)) if dt > 0 else None
        print(
            "[piecewise split-SDE] start | "
            f"n_init={x0_t.shape[0]}, ts_points={len(ts_points)}, "
            f"dt={dt}, sigma={sigma}, growth_alpha={growth_alpha}, "
            f"t_range=({t_min},{t_max}), est_steps={est_steps}"
        )

    sde = SDE(f_net.v_net, f_net.g_net, score_net, f_net.interaction_net, sigma=sigma)
    ts_tensor = torch.tensor(list(ts_points), dtype=torch.float32, device=device)
    sde_points, _ = euler_sdeint_split(sde, initial_state, dt=dt, ts=ts_tensor, noise_std=0.0)
    out = [p.detach().cpu().numpy() for p in sde_points]
    return np.array(out, dtype=object)


def _compute_spatial_warp_displacements(
    query_xy: np.ndarray,
    anchor_source_xy: np.ndarray,
    anchor_target_xy: np.ndarray,
    *,
    k: int,
    eps: float,
) -> np.ndarray:
    from sklearn.neighbors import NearestNeighbors

    query_xy = np.asarray(query_xy, dtype=np.float32)
    anchor_source_xy = np.asarray(anchor_source_xy, dtype=np.float32)
    anchor_target_xy = np.asarray(anchor_target_xy, dtype=np.float32)
    if anchor_source_xy.shape[0] == 0 or anchor_target_xy.shape[0] == 0:
        return np.zeros((query_xy.shape[0], 2), dtype=np.float32)

    target_nn = NearestNeighbors(n_neighbors=1)
    target_nn.fit(anchor_target_xy)
    _, target_idx = target_nn.kneighbors(anchor_source_xy)
    anchor_disp = anchor_target_xy[target_idx[:, 0]] - anchor_source_xy

    k_eff = max(1, min(int(k), anchor_source_xy.shape[0]))
    source_nn = NearestNeighbors(n_neighbors=k_eff)
    source_nn.fit(anchor_source_xy)
    dists, src_idx = source_nn.kneighbors(query_xy[:, :2])

    weights = 1.0 / np.maximum(dists, float(eps))
    weights /= weights.sum(axis=1, keepdims=True)
    disp = (anchor_disp[src_idx] * weights[..., None]).sum(axis=1)
    return disp.astype(np.float32, copy=False)


def _simulate_piecewise_spatially_warped_split(
    *,
    x0: np.ndarray,
    f_net,
    score_net,
    observed_time_points: Sequence[float],
    ts_points: Sequence[float],
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    label_col: str,
    dt: float,
    sigma: float,
    growth_alpha: float,
    interaction_m: int,
    device: str,
    rng: np.random.Generator,
    k: int,
    eps: float,
) -> np.ndarray:
    ts_sorted = [float(t) for t in ts_points]
    observed_sorted = [float(t) for t in observed_time_points if ts_sorted[0] <= float(t) <= ts_sorted[-1]]
    if len(observed_sorted) < 2:
        return _simulate_sde_points_split_from_x0(
            x0=x0,
            f_net=f_net,
            score_net=score_net,
            ts_points=ts_sorted,
            dt=dt,
            sigma=sigma,
            growth_alpha=growth_alpha,
            interaction_m=interaction_m,
            device=device,
            verbose=True,
        )

    current_x0 = np.asarray(x0, dtype=np.float32)
    points_by_time: Dict[float, np.ndarray] = {}
    feature_cols = list(feature_cols)

    def _contains_time(times: Sequence[float], t: float) -> bool:
        return any(np.isclose(float(v), float(t)) for v in times)

    for t_start, t_end in zip(observed_sorted[:-1], observed_sorted[1:]):
        seg_requested = [float(t) for t in ts_sorted if float(t_start) <= float(t) <= float(t_end)]
        seg_ts = list(seg_requested)
        if not _contains_time(seg_ts, t_start):
            seg_ts.append(float(t_start))
        if not _contains_time(seg_ts, t_end):
            seg_ts.append(float(t_end))
        seg_ts = sorted(seg_ts)
        if len(seg_ts) <= 1:
            current_x0 = np.asarray(current_x0, dtype=np.float32).copy()
            if _contains_time(seg_requested, t_start):
                points_by_time[float(t_start)] = current_x0.copy()
            continue

        print(f"[spatial-warp piecewise] segment {t_start}->{t_end} | targets={seg_ts}")
        seg_points = _simulate_sde_points_split_from_x0(
            x0=current_x0,
            f_net=f_net,
            score_net=score_net,
            ts_points=seg_ts,
            dt=dt,
            sigma=sigma,
            growth_alpha=growth_alpha,
            interaction_m=interaction_m,
            device=device,
            verbose=True,
        )

        source_endpoint_xy = np.asarray(seg_points[-1], dtype=np.float32)[:, :2]
        X_target, _ = _sample_observed_x0(
            df,
            time_value=float(t_end),
            feature_cols=feature_cols,
            label_col=label_col,
            n_samples_cap=min(int(source_endpoint_xy.shape[0]), int((df["samples"] == float(t_end)).sum())),
            rng=rng,
        )
        target_endpoint_xy = np.asarray(X_target, dtype=np.float32)[:, :2]

        for t_val, pts_raw in zip(seg_ts, seg_points):
            pts = np.asarray(pts_raw, dtype=np.float32).copy()
            alpha = (float(t_val) - float(t_start)) / max(float(t_end - t_start), float(eps))
            if alpha > 0.0:
                disp = _compute_spatial_warp_displacements(
                    pts[:, :2],
                    source_endpoint_xy,
                    target_endpoint_xy,
                    k=k,
                    eps=eps,
                )
                pts[:, :2] = pts[:, :2] + float(alpha) * disp
            if _contains_time(seg_requested, t_val):
                points_by_time[float(t_val)] = pts

        warped_endpoint = None
        for t_val, pts_raw in zip(seg_ts, seg_points):
            if np.isclose(float(t_val), float(t_end)):
                warped_endpoint = np.asarray(pts_raw, dtype=np.float32).copy()
                break
        if warped_endpoint is None:
            raise RuntimeError(f"Internal error: missing segment endpoint at t={t_end}")
        if not np.isclose(float(t_end), float(t_start)):
            disp_end = _compute_spatial_warp_displacements(
                warped_endpoint[:, :2],
                source_endpoint_xy,
                target_endpoint_xy,
                k=k,
                eps=eps,
            )
            warped_endpoint[:, :2] = warped_endpoint[:, :2] + disp_end
        current_x0 = warped_endpoint

    missing = [
        float(t)
        for t in ts_sorted
        if not any(np.isclose(float(t), float(k)) for k in points_by_time.keys())
    ]
    if missing:
        raise ValueError(f"Piecewise spatial-warp split-SDE missing timepoints: {missing}")
    out_points: List[np.ndarray] = []
    for t in ts_sorted:
        key = next(k for k in points_by_time.keys() if np.isclose(float(k), float(t)))
        out_points.append(points_by_time[key])
    return np.array(out_points, dtype=object)


def _load_optional_pca_mean(path: Optional[str], genes: np.ndarray) -> Tuple[Optional[np.ndarray], str]:
    if not path:
        return None, "none"
    if not os.path.exists(path):
        raise FileNotFoundError(f"PCA mean file not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        v = np.load(path)
        v = np.asarray(v, dtype=np.float32).ravel()
        if v.shape[0] != genes.shape[0]:
            raise ValueError(f"PCA mean length mismatch: {v.shape[0]} vs n_genes={genes.shape[0]}")
        return v, "npy"

    # CSV/TSV with either:
    # 1) two columns: gene_short_name, mean
    # 2) one numeric column in same order as genes
    sep = "\t" if ext in (".tsv", ".txt") else ","
    df = pd.read_csv(path, sep=sep)
    if df.shape[1] == 1:
        v = df.iloc[:, 0].to_numpy(dtype=np.float32).ravel()
        if v.shape[0] != genes.shape[0]:
            raise ValueError(f"PCA mean length mismatch: {v.shape[0]} vs n_genes={genes.shape[0]}")
        return v, "single_col"

    cols = [c.lower() for c in df.columns]
    gene_col = None
    mean_col = None
    for c in ("gene_short_name", "gene", "gene_name"):
        if c in cols:
            gene_col = df.columns[cols.index(c)]
            break
    for c in ("mean", "pca_mean", "center", "mu"):
        if c in cols:
            mean_col = df.columns[cols.index(c)]
            break
    if gene_col is None or mean_col is None:
        raise ValueError(
            f"Could not infer gene/mean columns from {path}. "
            "Need either 1 numeric col or {gene_short_name, mean}."
        )

    m = pd.Series(df[mean_col].to_numpy(dtype=np.float32), index=df[gene_col].astype(str))
    if not np.all(pd.Index(genes).isin(m.index)):
        miss = pd.Index(genes)[~pd.Index(genes).isin(m.index)]
        raise ValueError(f"PCA mean CSV missing genes: {list(miss[:10])} ... total_missing={miss.shape[0]}")
    v = m.loc[pd.Index(genes)].to_numpy(dtype=np.float32)
    return v, "gene_keyed"


def _load_pca_components(path: str) -> tuple[np.ndarray, np.ndarray]:
    comp_df = pd.read_csv(path)
    if "gene_short_name" not in comp_df.columns:
        raise KeyError(f"'gene_short_name' missing in {path}")
    pc_cols = [f"PC{i}" for i in range(1, 51)]
    miss_pc = [c for c in pc_cols if c not in comp_df.columns]
    if miss_pc:
        raise KeyError(f"PC columns missing in {path}: {miss_pc}")
    genes = comp_df["gene_short_name"].astype(str).to_numpy()
    loadings = comp_df[pc_cols].to_numpy(dtype=np.float32)
    return genes, loadings


def _build_gene_space_adata(
    *,
    x_t: np.ndarray,
    labels_t: np.ndarray,
    time_value: float,
    annotation_col: str,
    genes: np.ndarray,
    loadings: np.ndarray,
    pca_mean: Optional[np.ndarray],
    pca_mean_mode: str,
    output_x_space: str,
    count_transform: str,
    start_time: float,
    ts_points: Sequence[float],
    n_samples: int,
    spatial_warp_to_observed_piecewise: bool,
    spatial_warp_k: int,
    spatial_warp_eps: float,
) -> ad.AnnData:
    pc_scores = x_t[:, 2:52].astype(np.float32)
    if pc_scores.shape[1] != loadings.shape[1]:
        raise ValueError(f"PC mismatch: scores={pc_scores.shape}, loadings={loadings.shape}")

    gene_centered = pc_scores @ loadings.T
    if pca_mean is None:
        gene_recon = gene_centered
    else:
        gene_recon = gene_centered + pca_mean[None, :]

    gene_log1p = gene_recon.astype(np.float32)
    if count_transform == "clip":
        gene_counts = np.clip(np.expm1(gene_log1p), 0.0, None)
    else:
        gene_counts = np.log1p(np.exp(gene_log1p))

    X_out = gene_log1p if output_x_space == "log1p" else gene_counts
    adata_t = ad.AnnData(X=X_out.astype(np.float32))
    adata_t.var_names = genes
    adata_t.obs[annotation_col] = labels_t.astype(str)
    adata_t.obs["timepoint"] = str(time_value)
    adata_t.obs["samples"] = float(time_value)
    adata_t.obsm["spatial"] = x_t[:, :2].astype(np.float32)
    adata_t.obsm["latent_x"] = x_t
    adata_t.layers["recon_log1p"] = gene_log1p.astype(np.float32)
    adata_t.layers["recon_count"] = gene_counts.astype(np.float32)
    adata_t.layers["centered_recon"] = gene_centered.astype(np.float32)
    adata_t.layers["recon_raw"] = gene_recon.astype(np.float32)
    adata_t.uns["reconstruction_info"] = {
        "note": (
            "Reconstructed from PCA scores/loadings. Default X uses direct reconstructed log1p space. "
            "It is not guaranteed to equal original UMI counts."
        ),
        "output_x_space": output_x_space,
        "count_transform": count_transform,
        "pca_mean_file": "",
        "pca_mean_mode": pca_mean_mode,
        "start_time": float(start_time),
        "target_time": float(time_value),
        "ts_points": [float(t) for t in ts_points],
        "n_samples_request": int(n_samples),
        "n_cells_generated": int(adata_t.n_obs),
        "spatial_warp_to_observed_piecewise": bool(spatial_warp_to_observed_piecewise),
        "spatial_warp_k": int(spatial_warp_k),
        "spatial_warp_eps": float(spatial_warp_eps),
    }
    return adata_t


def _time_token(t: float) -> str:
    return f"{float(t):.3f}".replace(".", "p").replace("-", "n")


def _write_gene_space_output(
    *,
    adata_t: ad.AnnData,
    output_dir: str,
    time_value: float,
    annotation_col: str,
    pca_mean_file: Optional[str],
    pca_mean_mode: str,
    spatial_warp_to_observed_piecewise: bool,
    spatial_warp_k: int,
    spatial_warp_eps: float,
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    t_str = _time_token(time_value)
    out_h5ad = os.path.join(output_dir, f"adata_t{t_str}_with_genes.h5ad")
    adata_t.uns["reconstruction_info"]["pca_mean_file"] = pca_mean_file or ""
    adata_t.write_h5ad(out_h5ad)
    print("Saved interpolated gene-space h5ad:", out_h5ad)

    manifest_path = os.path.join(output_dir, "manifest.txt")
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write(f"target_time={time_value}\n")
        f.write(f"start_time={adata_t.uns['reconstruction_info']['start_time']}\n")
        f.write(f"ts_points={adata_t.uns['reconstruction_info']['ts_points']}\n")
        f.write(f"n_cells={adata_t.n_obs}\n")
        f.write(f"n_genes={adata_t.n_vars}\n")
        f.write(f"h5ad={out_h5ad}\n")
        f.write(f"annotation_col={annotation_col}\n")
        f.write(f"output_x_space={adata_t.uns['reconstruction_info']['output_x_space']}\n")
        f.write(f"count_transform={adata_t.uns['reconstruction_info']['count_transform']}\n")
        f.write(f"pca_mean_file={pca_mean_file if pca_mean_file else ''}\n")
        f.write(f"pca_mean_mode={pca_mean_mode}\n")
        f.write(f"spatial_warp_to_observed_piecewise={bool(spatial_warp_to_observed_piecewise)}\n")
        f.write(f"spatial_warp_k={int(spatial_warp_k)}\n")
        f.write(f"spatial_warp_eps={float(spatial_warp_eps)}\n")
    print("Saved manifest:", manifest_path)
    return out_h5ad


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate interpolated MOSTA h5ad in gene space (split-SDE).")
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--data-csv", default=DEFAULT_DATA_CSV)
    p.add_argument(
        "--classifier-data-csv",
        default=None,
        help="Optional CSV used only for classifier training when --data-csv has no annotation column.",
    )
    p.add_argument("--annotation-col", default="Annotation")
    p.add_argument("--pca-components-csv", default=DEFAULT_PCA_COMPONENTS)
    p.add_argument("--pca-mean-file", default=None, help="Optional PCA mean vector file (.npy/.csv/.tsv).")
    p.add_argument("--target-time", type=float, default=0.5)
    p.add_argument("--ts-points", default="0.0,0.5,1.0")
    p.add_argument("--start-time", type=float, default=None, help="SDE initialization time (default=min observed).")
    p.add_argument("--n-samples", type=int, default=12000)
    p.add_argument("--split-dt", type=float, default=0.05)
    p.add_argument("--split-sigma", type=float, default=0.03)
    p.add_argument("--split-growth-alpha", type=float, default=1.0)
    p.add_argument("--interaction-m", type=int, default=1024)
    p.add_argument(
        "--spatial-warp-to-observed-piecewise",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Run split-SDE piecewise across observed intervals and spatially warp each segment endpoint "
            "toward observed coordinates, then continue from warped endpoint."
        ),
    )
    p.add_argument("--spatial-warp-k", type=int, default=8, help="kNN size for piecewise spatial warp.")
    p.add_argument("--spatial-warp-eps", type=float, default=1e-6, help="Numerical epsilon for IDW weights.")
    p.add_argument("--classifier-n-pcs", type=int, default=10)
    p.add_argument("--classifier-epochs", type=int, default=500)
    p.add_argument("--classifier-hidden", type=int, default=128)
    p.add_argument(
        "--classifier-best-metric",
        choices=["accuracy", "bacc"],
        default="accuracy",
        help="Metric used to keep the best classifier epoch (must match cache metadata to reuse cache).",
    )
    p.add_argument(
        "--classifier-train-on-full-data",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Train classifier on full data (must match cache metadata to reuse cache).",
    )
    p.add_argument("--classifier-cache-dir", default="results/mosta_classifier_cache")
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--output-h5ad", default=None, help="Default: <output-dir>/adata_t0p500_with_genes.h5ad")
    p.add_argument(
        "--export-all-times",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Export one gene-space h5ad for every time in --ts-points from a single simulation run. "
            "Files are written to <output-dir>/t{time_token}/adata_t{time_token}_with_genes.h5ad."
        ),
    )
    p.add_argument(
        "--output-x-space",
        default="count",
        choices=["count", "log1p"],
        help="Which space to put into adata.X. Default is pseudo-count for downstream LR usage.",
    )
    p.add_argument(
        "--count-transform",
        default="clip",
        choices=["clip", "softplus"],
        help=(
            "Only used when --output-x-space=count. "
            "'clip': expm1(log1p_recon) then clip to >=0. "
            "'softplus': legacy smooth non-negative mapping."
        ),
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    cfg = load_config(args.config)
    dim = int(cfg["data"]["dim"])
    if dim < 52:
        raise ValueError(f"Expected dim>=52 for MOSTA latent+PC space, got {dim}")

    df_raw = pd.read_csv(args.data_csv, low_memory=False)
    has_ann_in_data_csv = args.annotation_col in df_raw.columns
    df_sim = df_raw.copy()
    req_cols = ["samples"] + [f"x{i}" for i in range(1, dim + 1)]
    for c in req_cols:
        if c not in df_sim.columns:
            raise KeyError(f"Required column '{c}' missing in {args.data_csv}")
    df_sim = df_sim[req_cols].copy()
    df_sim["samples"] = df_sim["samples"].astype(float)
    df_sim = df_sim.sort_values("samples").reset_index(drop=True)

    if has_ann_in_data_csv:
        df_clf = df_raw[req_cols + [args.annotation_col]].copy()
    else:
        clf_csv = args.classifier_data_csv
        if not clf_csv:
            raise KeyError(
                f"Annotation column '{args.annotation_col}' not found in {args.data_csv}. "
                "Pass --classifier-data-csv with the label column."
            )
        df_clf = pd.read_csv(clf_csv, low_memory=False)
        if args.annotation_col not in df_clf.columns:
            raise KeyError(f"Annotation column '{args.annotation_col}' not found in {clf_csv}")
        for c in req_cols:
            if c not in df_clf.columns:
                raise KeyError(f"Required column '{c}' missing in classifier csv: {clf_csv}")
        df_clf = df_clf[req_cols + [args.annotation_col]].copy()
        df_clf["samples"] = df_clf["samples"].astype(float)
        df_clf = df_clf.sort_values("samples").reset_index(drop=True)

    ts_points = _parse_csv_floats(args.ts_points)
    if args.target_time not in ts_points:
        raise ValueError(f"--target-time {args.target_time} must be included in --ts-points {ts_points}")
    observed_times = sorted(df_sim["samples"].unique().tolist())
    start_time = float(min(observed_times)) if args.start_time is None else float(args.start_time)
    if start_time not in observed_times:
        raise ValueError(f"--start-time {start_time} not in observed samples: {observed_times}")
    start_index = observed_times.index(start_time)

    f_net, score_net, exp_dir, device = load_models(
        cfg,
        exp_name=cfg["exp"]["name"],
        model_tag="model_final",
        score_tag="score_model",
    )
    print("Loaded models from:", exp_dir, "| device:", device)

    # Train / load classifier (same style as communication script)
    feature_dim = max(1, min(int(args.classifier_n_pcs), dim))
    feature_cols = ["samples"] + [f"x{i}" for i in range(1, feature_dim + 1)]
    clf_model, label_encoder, acc = train_mlp_classifier(
        df=df_clf,
        feature_cols=feature_cols,
        label_col=args.annotation_col,
        hidden_size=args.classifier_hidden,
        epochs=args.classifier_epochs,
        cache_dir=args.classifier_cache_dir,
        cache_tag=None,
        df_source_path=args.data_csv,
        reuse_if_possible=True,
        progress=True,
        device=device,
        best_epoch_metric=str(args.classifier_best_metric),
        train_on_full_data=bool(args.classifier_train_on_full_data),
    )
    print(f"Classifier ready | acc={acc:.4f}")

    # Split-SDE simulation
    if args.spatial_warp_to_observed_piecewise:
        if min(ts_points) < start_time:
            raise ValueError(
                "--spatial-warp-to-observed-piecewise requires min(--ts-points) >= --start-time "
                f"(got min(ts)={min(ts_points)} < start_time={start_time})."
            )
        latent_feature_cols = [f"x{i}" for i in range(1, dim + 1)]
        x0_init, _ = _sample_observed_x0(
            df=df_sim,
            time_value=start_time,
            feature_cols=latent_feature_cols,
            label_col=None,
            n_samples_cap=args.n_samples,
            rng=rng,
        )
        print(
            "[spatial-warp piecewise] enabled | "
            f"k={args.spatial_warp_k}, eps={args.spatial_warp_eps}, "
            f"start_time={start_time}, n_init={x0_init.shape[0]}"
        )
        sde_points = _simulate_piecewise_spatially_warped_split(
            x0=x0_init,
            f_net=f_net,
            score_net=score_net,
            observed_time_points=[t for t in observed_times if float(t) >= float(start_time)],
            ts_points=ts_points,
            df=df_sim,
            feature_cols=latent_feature_cols,
            label_col=None,
            dt=args.split_dt,
            sigma=args.split_sigma,
            growth_alpha=args.split_growth_alpha,
            interaction_m=args.interaction_m,
            device=device,
            rng=rng,
            k=args.spatial_warp_k,
            eps=args.spatial_warp_eps,
        )
    else:
        sde_points = simulate_sde_points_split(
            df=df_sim,
            dim=dim,
            f_net=f_net,
            score_net=score_net,
            time_index=start_index,
            n_samples=args.n_samples,
            ts_points=ts_points,
            dt=args.split_dt,
            sigma=args.split_sigma,
            growth_alpha=args.split_growth_alpha,
            interaction_m=args.interaction_m,
            device=device,
            verbose=True,
        )

    pred_labels = predict_labels_for_trajectories(
        sde_points=sde_points,
        ts_points=ts_points,
        model=clf_model,
        label_encoder=label_encoder,
        feature_dim=feature_dim,
        device=device,
        knn_neighbors=10,
    )

    genes, loadings = _load_pca_components(args.pca_components_csv)
    pca_mean, pca_mean_mode = _load_optional_pca_mean(args.pca_mean_file, genes=genes)

    if args.export_all_times:
        export_rows = []
        for t_val, pts_raw, labels_raw in zip(ts_points, sde_points, pred_labels):
            x_t = np.asarray(pts_raw, dtype=np.float32)
            labels_t = np.asarray(labels_raw).astype(str)
            print(f"Interpolated time {t_val} | cells={x_t.shape[0]} | latent_dim={x_t.shape[1]}")
            adata_t = _build_gene_space_adata(
                x_t=x_t,
                labels_t=labels_t,
                time_value=float(t_val),
                annotation_col=args.annotation_col,
                genes=genes,
                loadings=loadings,
                pca_mean=pca_mean,
                pca_mean_mode=pca_mean_mode,
                output_x_space=args.output_x_space,
                count_transform=args.count_transform,
                start_time=float(start_time),
                ts_points=ts_points,
                n_samples=args.n_samples,
                spatial_warp_to_observed_piecewise=bool(args.spatial_warp_to_observed_piecewise),
                spatial_warp_k=int(args.spatial_warp_k),
                spatial_warp_eps=float(args.spatial_warp_eps),
            )
            subdir = os.path.join(args.output_dir, f"t{_time_token(float(t_val))}")
            out_h5ad = _write_gene_space_output(
                adata_t=adata_t,
                output_dir=subdir,
                time_value=float(t_val),
                annotation_col=args.annotation_col,
                pca_mean_file=args.pca_mean_file,
                pca_mean_mode=pca_mean_mode,
                spatial_warp_to_observed_piecewise=bool(args.spatial_warp_to_observed_piecewise),
                spatial_warp_k=int(args.spatial_warp_k),
                spatial_warp_eps=float(args.spatial_warp_eps),
            )
            export_rows.append({"time_key": str(float(t_val)), "h5ad": out_h5ad})

        export_map_path = os.path.join(args.output_dir, "interp_map.json")
        with open(export_map_path, "w", encoding="utf-8") as f:
            pd.Series({row["time_key"]: row["h5ad"] for row in export_rows}).to_json(f, indent=2)
        print("Saved export map:", export_map_path)
    else:
        idx = ts_points.index(args.target_time)
        x_t = np.asarray(sde_points[idx], dtype=np.float32)
        labels_t = np.asarray(pred_labels[idx]).astype(str)
        print(f"Interpolated time {args.target_time} | cells={x_t.shape[0]} | latent_dim={x_t.shape[1]}")
        adata_t = _build_gene_space_adata(
            x_t=x_t,
            labels_t=labels_t,
            time_value=float(args.target_time),
            annotation_col=args.annotation_col,
            genes=genes,
            loadings=loadings,
            pca_mean=pca_mean,
            pca_mean_mode=pca_mean_mode,
            output_x_space=args.output_x_space,
            count_transform=args.count_transform,
            start_time=float(start_time),
            ts_points=ts_points,
            n_samples=args.n_samples,
            spatial_warp_to_observed_piecewise=bool(args.spatial_warp_to_observed_piecewise),
            spatial_warp_k=int(args.spatial_warp_k),
            spatial_warp_eps=float(args.spatial_warp_eps),
        )
        if args.output_h5ad:
            out_dir = os.path.dirname(args.output_h5ad) or args.output_dir
            os.makedirs(out_dir, exist_ok=True)
            out_h5ad = args.output_h5ad
            adata_t.uns["reconstruction_info"]["pca_mean_file"] = args.pca_mean_file or ""
            adata_t.write_h5ad(out_h5ad)
            print("Saved interpolated gene-space h5ad:", out_h5ad)
        else:
            out_h5ad = _write_gene_space_output(
                adata_t=adata_t,
                output_dir=args.output_dir,
                time_value=float(args.target_time),
                annotation_col=args.annotation_col,
                pca_mean_file=args.pca_mean_file,
                pca_mean_mode=pca_mean_mode,
                spatial_warp_to_observed_piecewise=bool(args.spatial_warp_to_observed_piecewise),
                spatial_warp_k=int(args.spatial_warp_k),
                spatial_warp_eps=float(args.spatial_warp_eps),
            )
        if args.output_h5ad:
            manifest_path = os.path.join(args.output_dir, "manifest.txt")
            with open(manifest_path, "w", encoding="utf-8") as f:
                f.write(f"target_time={args.target_time}\n")
                f.write(f"start_time={start_time}\n")
                f.write(f"ts_points={ts_points}\n")
                f.write(f"n_cells={adata_t.n_obs}\n")
                f.write(f"n_genes={adata_t.n_vars}\n")
                f.write(f"h5ad={out_h5ad}\n")
                f.write(f"annotation_col={args.annotation_col}\n")
                f.write(f"output_x_space={args.output_x_space}\n")
                f.write(f"count_transform={args.count_transform}\n")
                f.write(f"pca_mean_file={args.pca_mean_file if args.pca_mean_file else ''}\n")
                f.write(f"pca_mean_mode={pca_mean_mode}\n")
                f.write(f"spatial_warp_to_observed_piecewise={bool(args.spatial_warp_to_observed_piecewise)}\n")
                f.write(f"spatial_warp_k={int(args.spatial_warp_k)}\n")
                f.write(f"spatial_warp_eps={float(args.spatial_warp_eps)}\n")
            print("Saved manifest:", manifest_path)


if __name__ == "__main__":
    main()
