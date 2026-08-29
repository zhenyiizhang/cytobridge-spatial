#!/usr/bin/env python3
"""
MOSTA multilayer communication (DeepRUOTv2) + 3D spatiotemporal plot (focus-anchor).

This script is a corrected/local runnable replacement for
`evaluation/mosta/code/mosta_multilayer_communication.ipynb`, based on the
working pipeline in:
  `evaluation/arista_code/3d_plot_5_slices_focus_anchor_local.py`

Key fixes vs the notebook:
- Use correct imports under `evaluation/arista_code/` (the notebook used non-existent `evaluation.arista_helpers`).
- Remove Lustre-only hard-coded paths; use project-root-relative defaults.
- Fix the split-SDE block (the notebook cell has a syntax/indentation error).

Requirements:
- Run inside the DeepRUOTv2 conda env (torch, torch_geometric, scanpy/scvelo, plotly+kaleido).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
import torch

os.environ.setdefault("MPLBACKEND", "Agg")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from evaluation.arista_code.arista_helpers import (  # noqa: E402
    analyze_attention_by_celltype,
    load_config,
    load_models,
    predict_labels_for_trajectories,
    save_interpolated_attention,
    simulate_sde_points,
    simulate_sde_points_split,
    train_mlp_classifier,
)
from evaluation.arista_code.arista_helpers_focus_anchor import (  # noqa: E402
    plot_3d_spatial_sankey_style_focus_anchor,
)


# =========================
# User-editable parameters
# =========================

# You can either edit these defaults, or override a subset via CLI flags.

# Data / model
config_path = "config/mosta_config.yaml"
data_csv = "evaluation/mosta/data/mosta_four_time_with_celltype_refined.csv"
annotation_key = "Annotation"

# Output
output_dir = "results/mosta_communication_5_slices_focus_anchor_local"

# Optional: reuse annotation colors from h5ad (loaded backed='r' to avoid 38GB RAM usage)
color_h5ad = "spatial_data/Mouse_embryo_all_stage.h5ad"
label_color_json = None  # optional JSON mapping label->color

# Random seed (set None to disable)
random_seed = 42

# Observed/interpolated timepoints
# - If `interp_time_points` is empty, no SDE simulation is needed (only real data used).
# - If non-empty, we simulate trajectories and classify simulated cells for lineage ribbons.
interp_time_points = [0.5, 1.5, 2.5]
target_total_slices = None  # set e.g. 5 to downselect observed slices evenly
use_real_for_observed = True

# 3D plot: which timepoints to render as slices (communication 3D).
# Set to None to render all `ts_points`. Example: [0.0, 0.5, 1.0]
plot_3d_time_points = [0.0, 0.5, 1.0]

# SDE settings (only used when interpolation requested)
sde_dt = 0.05
split_sde_dt = 0.05
split_sigma = 0.03
split_sigma_spatial = None
split_sigma_gene = None
split_sigma_by_dim = None
split_growth_alpha = 1
piecewise_observed_sample_mode = "t0_fixed"  # "t0_fixed" | "per_timepoint"
spatial_warp_to_observed = False
spatial_warp_to_observed_piecewise = False
spatial_warp_k = 8
spatial_warp_eps = 1e-6

# Classifier settings (only used when interpolation requested)
classifier_epochs = 500
classifier_hidden = 128
classifier_n_pcs = 12  # None uses x1..x52; or set int for speed (e.g. 10)
classifier_knn_neighbors = 10  # set 1 to disable KNN refinement
classifier_best_metric = "accuracy"  # "accuracy" | "bacc"
classifier_train_on_full_data = False
classifier_save_test_models = False
classifier_last_k_epochs = 5

# Communication aggregation settings
remove_self_loop = False
winsor_quantile = 0.995

# 3D plot controls
reverse_time_order = True
comm_focus_label = "Brain"  # focus communication edges on one label; set None to disable
comm_edge_threshold = 0.0  # draw if weight > threshold
comm_edge_top_k = 5  # per-timepoint top-K edges (set None to disable)
comm_edge_top_k_focus_label = comm_focus_label

fate_focus_label = "Brain"  # set None to disable ribbon filtering
fate_focus_mode = "source"  # 'either' | 'source' | 'target'
fate_min_flow = None  # minimum count threshold for ribbon rendering (set 0 to disable)
fate_keep_source_cumfrac = 0.8  # e.g. 0.8 keeps top outgoing links per source until >=80% coverage (None disables)

# Focus-anchor: for edges/ribbons involving focus_anchor_label, use local centroids near focus.
focus_anchor_label = fate_focus_label
focus_anchor_frac = 0.2
focus_anchor_k = None
focus_anchor_radius = None
focus_anchor_min_count = None

# Styling
background_color = None
font_color = "#1a1a1a"
comm_edge_color = "rgba(25,25,25,0.75)"
z_spacing = 1.0

# Slice styling for observed vs generated (subtle cool vs warm)
slice_border_color_observed = "#5f6a72"  # cool gray
slice_border_color_generated = "#8c6d5a"  # warm taupe
slice_fill_color_observed = "#e6f0f6"  # light blue-gray
slice_fill_color_generated = "#f6eee5"  # light sand
slice_fill_opacity = 0.5
slice_border_width = 5

# Export settings (Plotly requires kaleido for static export)
export_svg = True
export_pdf = True
export_png = True
png_scale = 2
vector_scale = 3

# Attention export: dense N x N matrix is memory-heavy for large slices.
save_dense_attention_matrix = False

# Lineage Sankey (cell fate flow) from non-split SDE labels
plot_lineage_sankey = True
sankey_min_flow = None  # None means no filtering
sankey_normalize_mode = None  # None | 'source' | 'global'
sankey_keep_source_cumfrac = 0.8  # e.g. 0.8 keeps top outgoing links per source until >=80% coverage (None disables)
sankey_style = "nature-methods"  # 'default' | 'nature-methods'


# =========================
# Helpers
# =========================


def _select_evenly_spaced(values: Sequence[float], n_keep: int) -> list[float]:
    if n_keep >= len(values):
        return list(values)
    idx = np.linspace(0, len(values) - 1, num=n_keep)
    idx = [int(round(i)) for i in idx]
    seen = set()
    idx_unique = []
    for i in idx:
        if i not in seen:
            idx_unique.append(i)
            seen.add(i)
    if len(idx_unique) < n_keep:
        for i in range(len(values)):
            if i in seen:
                continue
            idx_unique.append(i)
            if len(idx_unique) == n_keep:
                break
    idx_unique = sorted(idx_unique)
    return [values[i] for i in idx_unique]


def load_label_to_color(
    labels: np.ndarray,
    label_color_json: Optional[str] = None,
    color_h5ad: Optional[str] = None,
    annotation_key: str = "Annotation",
) -> Dict[str, str]:
    if label_color_json and os.path.exists(label_color_json):
        with open(label_color_json, "r", encoding="utf-8") as f:
            return json.load(f)

    if color_h5ad and os.path.exists(color_h5ad):
        try:
            import anndata as ad

            adata = ad.read_h5ad(color_h5ad, backed="r")
            try:
                key = annotation_key if annotation_key in adata.obs else None
                if key is None and annotation_key.lower() in adata.obs:
                    key = annotation_key.lower()
                if key:
                    colors_key = f"{key}_colors"
                    colors = adata.uns.get(colors_key)
                    if colors is not None:
                        categories = (
                            adata.obs[key].cat.categories
                            if hasattr(adata.obs[key], "cat")
                            else sorted(adata.obs[key].unique())
                        )
                        return {str(c): str(col) for c, col in zip(categories, colors)}
            finally:
                try:
                    adata.file.close()
                except Exception:
                    pass
        except Exception as exc:
            print(f"Color map load failed from {color_h5ad}: {exc}")

    import matplotlib.pyplot as plt

    unique_labels = list(dict.fromkeys([str(x) for x in labels]))
    cmap = plt.get_cmap("tab20")
    out = {}
    for idx, lab in enumerate(unique_labels):
        rgb = cmap(idx % cmap.N)[:3]
        out[str(lab)] = "#{:02x}{:02x}{:02x}".format(
            int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
        )
    return out


def _require_columns(df: pd.DataFrame, cols: Sequence[str], ctx: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {ctx}: {missing}")


def _parse_csv_floats(value: str) -> list[float]:
    if value is None:
        return []
    parts = [p.strip() for p in str(value).split(",") if p.strip()]
    return [float(p) for p in parts]


def _parse_csv_floats_or_all(value: Optional[str]) -> Optional[list[float]]:
    if value is None:
        return None
    s = str(value).strip().lower()
    if s in ("", "all", "none"):
        return None
    return _parse_csv_floats(value)


def _resolve_split_sigma(dim: int, args: argparse.Namespace) -> tuple[float, Optional[list[float]]]:
    sigma_by_dim_text = getattr(args, "split_sigma_by_dim", None)
    sigma_spatial = getattr(args, "split_sigma_spatial", None)
    sigma_gene = getattr(args, "split_sigma_gene", None)
    sigma_scalar = float(args.split_sigma)

    if sigma_by_dim_text not in (None, ""):
        sigma_by_dim = _parse_csv_floats(sigma_by_dim_text)
        if len(sigma_by_dim) != dim:
            raise ValueError(
                f"--split-sigma-by-dim must contain exactly {dim} comma-separated values; "
                f"got {len(sigma_by_dim)}"
            )
        return sigma_scalar, [float(x) for x in sigma_by_dim]

    if sigma_spatial is None and sigma_gene is None:
        return sigma_scalar, None

    sigma_spatial = sigma_scalar if sigma_spatial is None else float(sigma_spatial)
    sigma_gene = sigma_scalar if sigma_gene is None else float(sigma_gene)
    sigma_by_dim = [sigma_spatial] * min(2, dim)
    if dim > 2:
        sigma_by_dim.extend([sigma_gene] * (dim - 2))
    return sigma_scalar, sigma_by_dim


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

    if query_xy.ndim != 2 or query_xy.shape[1] < 2:
        raise ValueError("query_xy must be a 2D array with at least 2 columns")
    if anchor_source_xy.ndim != 2 or anchor_source_xy.shape[1] != 2:
        raise ValueError("anchor_source_xy must have shape (n, 2)")
    if anchor_target_xy.ndim != 2 or anchor_target_xy.shape[1] != 2:
        raise ValueError("anchor_target_xy must have shape (m, 2)")
    if anchor_source_xy.shape[0] == 0 or anchor_target_xy.shape[0] == 0:
        return np.zeros((query_xy.shape[0], 2), dtype=np.float32)

    # Build anchor-wise displacement by mapping each simulated endpoint point
    # to its nearest observed endpoint point in spatial coordinates.
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


def _apply_spatial_warp_to_segments(
    *,
    sde_points_split: np.ndarray,
    ts_points: Sequence[float],
    observed_time_points: Sequence[float],
    df: pd.DataFrame,
    feature_cols_full: Sequence[str],
    label_col: str,
    rng: np.random.Generator,
    piecewise: bool,
    piecewise_include_end: bool,
    piecewise_endpoint_by_observed: Optional[Dict[float, np.ndarray]],
    use_real_for_observed: bool,
    k: int,
    eps: float,
) -> np.ndarray:
    if len(ts_points) == 0 or len(observed_time_points) < 2:
        return sde_points_split

    sde_points_out = np.array(
        [np.asarray(p, dtype=np.float32).copy() for p in sde_points_split],
        dtype=object,
    )
    ts_index = {float(t): i for i, t in enumerate(ts_points)}
    observed_set = {float(t) for t in observed_time_points}

    for t_start, t_end in zip(observed_time_points[:-1], observed_time_points[1:]):
        t_start = float(t_start)
        t_end = float(t_end)
        interior_ts = sorted([float(t) for t in ts_points if t_start < float(t) < t_end])

        if piecewise:
            if not piecewise_include_end or not piecewise_endpoint_by_observed:
                print(
                    f"[spatial-warp] skip segment {t_start}->{t_end}: "
                    "--split-sde-piecewise requires --split-sde-piecewise-include-end for warp anchors"
                )
                continue
            source_endpoint = piecewise_endpoint_by_observed.get(t_end)
            if source_endpoint is None:
                print(f"[spatial-warp] skip segment {t_start}->{t_end}: missing simulated endpoint cache")
                continue
            source_endpoint_xy = np.asarray(source_endpoint, dtype=np.float32)[:, :2]
        else:
            idx_end = ts_index.get(t_end)
            if idx_end is None:
                continue
            source_endpoint_xy = np.asarray(sde_points_out[idx_end], dtype=np.float32)[:, :2]

        if source_endpoint_xy.shape[0] == 0:
            continue

        X_target, _ = _sample_observed_x0(
            df,
            time_value=t_end,
            feature_cols=feature_cols_full,
            label_col=label_col,
            n_samples_cap=int(source_endpoint_xy.shape[0]),
            rng=rng,
        )
        target_endpoint_xy = np.asarray(X_target, dtype=np.float32)[:, :2]
        if target_endpoint_xy.shape[0] == 0:
            continue

        segment_apply_ts = list(interior_ts)
        if (not use_real_for_observed) and (t_end in ts_index):
            segment_apply_ts.append(t_end)
        if len(segment_apply_ts) == 0:
            continue

        for t_val in segment_apply_ts:
            idx = ts_index.get(float(t_val))
            if idx is None:
                continue
            alpha = (float(t_val) - t_start) / max(t_end - t_start, float(eps))
            pts = np.asarray(sde_points_out[idx], dtype=np.float32)
            if pts.shape[0] == 0:
                continue
            disp = _compute_spatial_warp_displacements(
                pts[:, :2],
                source_endpoint_xy,
                target_endpoint_xy,
                k=k,
                eps=eps,
            )
            pts[:, :2] = pts[:, :2] + float(alpha) * disp
            sde_points_out[idx] = pts

        print(
            f"[spatial-warp] segment {t_start}->{t_end} | "
            f"anchors_sim={source_endpoint_xy.shape[0]} anchors_real={target_endpoint_xy.shape[0]} "
            f"targets={segment_apply_ts}"
        )

    return sde_points_out


def _simulate_piecewise_spatially_warped_split(
    *,
    x0: np.ndarray,
    f_net,
    score_net,
    observed_time_points: Sequence[float],
    ts_points: Sequence[float],
    df: pd.DataFrame,
    feature_cols_full: Sequence[str],
    label_col: str,
    dt: float,
    sigma: float,
    sigma_by_dim: Optional[Sequence[float]],
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
            sigma_by_dim=sigma_by_dim,
            growth_alpha=growth_alpha,
            interaction_m=interaction_m,
            device=device,
            verbose=True,
        )

    current_x0 = np.asarray(x0, dtype=np.float32)
    points_by_time: Dict[float, np.ndarray] = {}

    for t_start, t_end in zip(observed_sorted[:-1], observed_sorted[1:]):
        seg_ts = [float(t) for t in ts_sorted if float(t_start) <= float(t) <= float(t_end)]
        if len(seg_ts) == 0:
            continue
        if len(seg_ts) == 1:
            points_by_time[float(seg_ts[0])] = np.asarray(current_x0, dtype=np.float32).copy()
            continue

        print(f"[spatial-warp piecewise] segment {t_start}->{t_end} | targets={seg_ts}")
        seg_points = _simulate_sde_points_split_from_x0(
            x0=current_x0,
            f_net=f_net,
            score_net=score_net,
            ts_points=seg_ts,
            dt=dt,
            sigma=sigma,
            sigma_by_dim=sigma_by_dim,
            growth_alpha=growth_alpha,
            interaction_m=interaction_m,
            device=device,
            verbose=True,
        )

        source_endpoint_xy = np.asarray(seg_points[-1], dtype=np.float32)[:, :2]
        X_target, _ = _sample_observed_x0(
            df,
            time_value=float(t_end),
            feature_cols=feature_cols_full,
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
            points_by_time[float(t_val)] = pts

        current_x0 = np.asarray(points_by_time[float(t_end)], dtype=np.float32).copy()

    missing = [float(t) for t in ts_sorted if float(t) not in points_by_time]
    if missing:
        raise ValueError(f"Piecewise spatial-warp split-SDE missing timepoints: {missing}")

    return np.array([points_by_time[float(t)] for t in ts_sorted], dtype=object)


def _downsample_xy(
    X: np.ndarray,
    y: np.ndarray,
    max_n: Optional[int],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if max_n is None or max_n <= 0:
        return X, y
    n = int(X.shape[0])
    if n <= max_n:
        return X, y
    idx = rng.choice(n, size=int(max_n), replace=False)
    return X[idx], y[idx]


def _sample_observed_x0(
    df: pd.DataFrame,
    *,
    time_value: float,
    feature_cols: Sequence[str],
    label_col: str,
    n_samples_cap: Optional[int],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    subset = df[df["samples"] == float(time_value)]
    X = subset[list(feature_cols)].values.astype(np.float32)
    labels = subset[label_col].astype(str).values
    if n_samples_cap is None:
        return X, labels
    cap = int(n_samples_cap)
    if cap <= 0:
        raise ValueError("--sde-n-samples must be > 0 when provided")
    if X.shape[0] <= cap:
        return X, labels
    idx = rng.choice(X.shape[0], size=cap, replace=False)
    return X[idx], labels[idx]


def _simulate_sde_points_split_from_x0(
    *,
    x0: np.ndarray,
    f_net,
    score_net,
    ts_points: Sequence[float],
    dt: float,
    sigma: float,
    sigma_by_dim: Optional[Sequence[float]],
    growth_alpha: float,
    interaction_m: int,
    device: str,
    verbose: bool = True,
) -> np.ndarray:
    """
    Split-SDE simulation starting from a provided initial state x0 (no internal resampling).

    This mirrors `evaluation.arista_code.arista_helpers.simulate_sde_points_split`, but accepts x0
    directly so we can restart from each observed timepoint distribution.
    """
    import torch
    from DeepRUOT.interaction import cal_interaction, euler_sdeint_split

    if len(ts_points) < 1:
        raise ValueError("ts_points must be non-empty")

    x0_t = torch.tensor(np.asarray(x0, dtype=np.float32), device=device)
    lnw0 = torch.log(torch.ones(x0_t.shape[0], 1, device=device) / x0_t.shape[0])
    initial_state = (x0_t, lnw0)

    class SDE(torch.nn.Module):
        noise_type = "diagonal"
        sde_type = "ito"

        def __init__(self, ode_drift, g, score, interaction, sigma, sigma_by_dim):
            super().__init__()
            self.drift = ode_drift
            self.score = score
            self.interaction = interaction
            self.g_net = g
            if sigma_by_dim is None:
                self.register_buffer("sigma_vec", None)
                self.sigma = float(sigma)
            else:
                sigma_arr = np.asarray(list(sigma_by_dim), dtype=np.float32).reshape(-1)
                if sigma_arr.shape[0] != x0_t.shape[1]:
                    raise ValueError(
                        f"sigma_by_dim must have length {x0_t.shape[1]}, got {sigma_arr.shape[0]}"
                    )
                self.register_buffer("sigma_vec", torch.tensor(sigma_arr, dtype=torch.float32))
                self.sigma = None

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
            if self.sigma_vec is None:
                return torch.ones_like(y) * self.sigma
            return self.sigma_vec.to(device=y.device, dtype=y.dtype).unsqueeze(0).expand_as(y)

    if verbose:
        try:
            t_min = float(min(ts_points))
            t_max = float(max(ts_points))
            est_steps = int(round((t_max - t_min) / dt)) if dt > 0 else None
        except Exception:
            t_min, t_max, est_steps = None, None, None
        print(
            "[piecewise split-SDE] start | "
            f"n_init={x0_t.shape[0]}, ts_points={len(ts_points)}, "
            f"dt={dt}, sigma={'vector' if sigma_by_dim is not None else sigma}, "
            f"growth_alpha={growth_alpha}, "
            f"t_range=({t_min},{t_max}), est_steps={est_steps}"
        )

    sde = SDE(
        f_net.v_net,
        f_net.g_net,
        score_net,
        f_net.interaction_net,
        sigma=sigma,
        sigma_by_dim=sigma_by_dim,
    )
    ts_tensor = torch.tensor(list(ts_points), dtype=torch.float32, device=device)
    sde_points, _ = euler_sdeint_split(sde, initial_state, dt=dt, ts=ts_tensor, noise_std=0.0)
    sde_point_np = [p.detach().cpu().numpy() for p in sde_points]

    if verbose:
        print(
            "[piecewise split-SDE] done | "
            f"timepoints={len(sde_point_np)}, "
            f"shape0={sde_point_np[0].shape if sde_point_np else None}"
        )
    return np.array(sde_point_np, dtype=object)


def _predict_labels_for_points(
    *,
    points: np.ndarray,
    time_value: float,
    model,
    label_encoder,
    feature_dim: int,
    device: str,
    knn_neighbors: int = 50,
) -> np.ndarray:
    import numpy as np
    import torch
    from sklearn.neighbors import KNeighborsClassifier

    pts = np.asarray(points, dtype=np.float32)
    n = int(pts.shape[0])
    if n == 0:
        return np.asarray([], dtype=str)

    model.eval()
    model.to(device)

    traj_t_tensor = torch.tensor(pts, dtype=torch.float32)
    samples_t = torch.full((n, 1), fill_value=float(time_value))
    input_t = torch.cat((samples_t, traj_t_tensor[:, : int(feature_dim)]), dim=1)

    with torch.no_grad():
        outputs = model(input_t.float().to(device))
        _, predicted = torch.max(outputs, 1)
        predicted_labels = label_encoder.inverse_transform(predicted.detach().cpu().numpy())

    coords = input_t[:, 1:3].cpu().numpy()
    k = min(int(knn_neighbors), int(coords.shape[0]))
    if k <= 1:
        return np.asarray(predicted_labels).astype(str)
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(coords, predicted_labels)
    refined_labels = knn.predict(coords)
    return np.asarray(refined_labels).astype(str)


def save_timepoint_snapshots(
    *,
    adata_dict,
    time_keys: Sequence[str],
    annotation_key: str,
    label_to_color: Dict[str, str],
    observed_time_points: Optional[Sequence[float]] = None,
    observed_variants: Optional[Dict[float, Dict[str, tuple[np.ndarray, np.ndarray]]]] = None,
    snapshot_dir: str,
    background_color: Optional[str],
    font_color: str,
    snapshot_point_size: float,
    snapshot_alpha: float,
    mosaic_cols: int,
    mosaic_cell_size: float,
    mosaic_show_title: bool,
    save_pdf: bool = True,
) -> None:
    """
    Save 2D per-timepoint scatter snapshots (SVG + optional PDF), a mosaic panel, and a label legend.

    Logic matches `evaluation/mosta/code/mosta_multilayer_communication.ipynb` (cells 19/21/27),
    but uses this script's `adata_dict` (including interpolated slices).
    """
    import math

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    bg = background_color or "white"
    os.makedirs(snapshot_dir, exist_ok=True)

    panels: list[dict] = []
    for tk in time_keys:
        t_val = float(tk)
        if observed_variants is not None and (t_val in observed_variants):
            variants = observed_variants[t_val]
            for suffix, (coords, labels) in variants.items():
                panels.append(
                    {
                        "tk": tk,
                        "suffix": suffix,
                        "title": f"t = {tk} ({suffix})",
                        "coords": np.asarray(coords),
                        "labels": np.asarray(labels).astype(str),
                    }
                )
        else:
            ad = adata_dict[tk]
            coords = np.asarray(ad.obsm["spatial"])
            labels = ad.obs[annotation_key].astype(str).values
            panels.append(
                {
                    "tk": tk,
                    "suffix": None,
                    "title": f"t = {tk}",
                    "coords": coords,
                    "labels": np.asarray(labels).astype(str),
                }
            )

    for panel in panels:
        coords = np.asarray(panel["coords"])
        labels = np.asarray(panel["labels"]).astype(str)
        colors = [label_to_color.get(str(l), "#888888") for l in labels]
        rasterized = coords.shape[0] > 30000

        fig, ax = plt.subplots(figsize=(4.2, 4.2), dpi=300)
        fig.patch.set_facecolor(bg)
        ax.set_facecolor(bg)

        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            s=float(snapshot_point_size),
            c=colors,
            linewidths=0,
            alpha=float(snapshot_alpha),
            rasterized=rasterized,
        )
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(panel["title"], color=font_color, fontsize=12, pad=6)

        tk = panel["tk"]
        suffix = panel["suffix"]
        stem = f"time_{tk}" if suffix is None else f"time_{tk}__{suffix}"
        out_path = os.path.join(snapshot_dir, f"{stem}.svg")
        fig.savefig(out_path, format="svg", facecolor=bg, bbox_inches="tight")
        if save_pdf:
            out_path = os.path.join(snapshot_dir, f"{stem}.pdf")
            fig.savefig(out_path, format="pdf", facecolor=bg, bbox_inches="tight")
        plt.close(fig)

    # Mosaic panel
    n_panels = len(panels)
    cols = max(1, int(mosaic_cols))
    rows = math.ceil(n_panels / cols)
    fig_w = cols * float(mosaic_cell_size)
    fig_h = rows * float(mosaic_cell_size)

    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h), dpi=300)
    fig.patch.set_facecolor(bg)
    axes = axes if isinstance(axes, np.ndarray) else np.array([[axes]])
    axes = axes.reshape(rows, cols)

    for idx, panel in enumerate(panels):
        r, c = divmod(idx, cols)
        ax = axes[r, c]
        coords = np.asarray(panel["coords"])
        labels = np.asarray(panel["labels"]).astype(str)
        colors = [label_to_color.get(str(l), "#888888") for l in labels]
        rasterized = coords.shape[0] > 30000

        ax.set_facecolor(bg)
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            s=float(snapshot_point_size),
            c=colors,
            linewidths=0,
            alpha=float(snapshot_alpha),
            rasterized=rasterized,
        )
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        if mosaic_show_title:
            title = panel["tk"] if panel.get("suffix") is None else f"{panel['tk']} ({panel['suffix']})"
            ax.set_title(title, color=font_color, fontsize=8, pad=3)

    for idx in range(n_panels, rows * cols):
        r, c = divmod(idx, cols)
        axes[r, c].axis("off")
        axes[r, c].set_facecolor(bg)

    mosaic_path = os.path.join(snapshot_dir, "timepoint_mosaic.svg")
    fig.savefig(mosaic_path, format="svg", facecolor=bg, bbox_inches="tight")
    plt.close(fig)

    # Label legend
    legend_path = os.path.join(snapshot_dir, "label_legend.svg")
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=label_to_color[k], markersize=6, label=k)
        for k in label_to_color.keys()
    ]

    fig, ax = plt.subplots(figsize=(4, 6), facecolor=bg)
    ax.set_facecolor(bg)
    ax.legend(handles=handles, loc="center left", frameon=False, labelcolor=font_color)
    ax.axis("off")
    fig.savefig(legend_path, format="svg", facecolor=bg, bbox_inches="tight")
    plt.close(fig)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="MOSTA multilayer communication + 3D focus-anchor plot")
    parser.add_argument("--config", default=config_path)
    parser.add_argument("--data-csv", default=data_csv)
    parser.add_argument("--annotation-key", default=annotation_key)
    parser.add_argument("--output-dir", default=output_dir)
    parser.add_argument("--color-h5ad", default=color_h5ad)
    parser.add_argument("--label-color-json", default=label_color_json)
    parser.add_argument("--interp-time-points", default=",".join(str(x) for x in interp_time_points))
    parser.add_argument("--no-interp", action="store_true", help="Disable interpolation (no SDE/classifier)")
    parser.add_argument(
        "--plot-3d-time-points",
        default=",".join(str(x) for x in plot_3d_time_points) if plot_3d_time_points is not None else "all",
        help="Comma-separated timepoints to render in the 3D communication plot (use 'all' to render all).",
    )
    parser.add_argument("--target-total-slices", type=int, default=target_total_slices)
    parser.add_argument(
        "--max-observed-timepoints",
        type=int,
        default=None,
        help=(
            "Optional hard cap on number of observed timepoints kept from the input CSV. "
            "If --plot-3d-time-points specifies observed times, those are always kept."
        ),
    )
    parser.add_argument(
        "--use-real-for-observed",
        action=argparse.BooleanOptionalAction,
        default=use_real_for_observed,
        help="Use real cells for observed timepoints (disable to use split-SDE generated points instead).",
    )
    parser.add_argument("--classifier-epochs", type=int, default=classifier_epochs)
    parser.add_argument("--classifier-hidden", type=int, default=classifier_hidden)
    parser.add_argument("--classifier-n-pcs", type=int, default=classifier_n_pcs)
    parser.add_argument(
        "--classifier-knn-neighbors",
        type=int,
        default=classifier_knn_neighbors,
        help=(
            "KNN neighbors used to spatially refine classifier labels after MLP prediction. "
            "Set to 1 to disable refinement."
        ),
    )
    parser.add_argument(
        "--classifier-best-metric",
        choices=["accuracy", "bacc"],
        default=classifier_best_metric,
        help="Metric used to keep the best classifier epoch: accuracy or balanced accuracy (bacc).",
    )
    parser.add_argument(
        "--classifier-train-on-full-data",
        action=argparse.BooleanOptionalAction,
        default=classifier_train_on_full_data,
        help="Train classifier using all rows without train/val split.",
    )
    parser.add_argument(
        "--classifier-save-test-models",
        action=argparse.BooleanOptionalAction,
        default=classifier_save_test_models,
        help="Save classifier checkpoints for offline testing: best_acc.pt, best_bacc.pt, and last-K epochs.",
    )
    parser.add_argument(
        "--classifier-last-k-epochs",
        type=int,
        default=classifier_last_k_epochs,
        help="How many final epoch checkpoints to save when --classifier-save-test-models is enabled.",
    )
    parser.add_argument(
        "--classifier-checkpoint-dir",
        default=None,
        help="Directory for classifier checkpoints (default: <output_dir>/classifier_checkpoints).",
    )
    parser.add_argument(
        "--classifier-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse/save classifier weights in a cache dir to avoid retraining every run.",
    )
    parser.add_argument(
        "--classifier-cache-dir",
        default=None,
        help="Directory for classifier cache files (default: <output_dir>/classifier_cache).",
    )
    parser.add_argument(
        "--classifier-cache-tag",
        default=None,
        help="Optional tag mixed into the cache key (useful if you want separate caches).",
    )
    parser.add_argument(
        "--split-sde-piecewise",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Run split-SDE for interpolation in piecewise segments, restarting from each observed timepoint's "
            "real cell distribution (e.g. 0->1, then 1->2, etc.)."
        ),
    )
    parser.add_argument(
        "--split-sde-piecewise-include-end",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "In piecewise split-SDE mode, also integrate to the next observed endpoint even if not needed for "
            "requested interpolation times (slower, but matches 'run to 1, then restart at 1' literally)."
        ),
    )
    parser.add_argument(
        "--piecewise-observed-sample-mode",
        choices=["t0_fixed", "per_timepoint"],
        default=piecewise_observed_sample_mode,
        help=(
            "Observed-start sampling policy for --split-sde-piecewise: "
            "t0_fixed uses one global cap from earliest observed timepoint; "
            "per_timepoint uses min(--sde-n-samples, cells_at_that_timepoint) independently per observed time."
        ),
    )
    parser.add_argument(
        "--split-sigma",
        type=float,
        default=split_sigma,
        help="Scalar split-SDE noise used when no per-dimension override is set.",
    )
    parser.add_argument(
        "--split-sigma-spatial",
        type=float,
        default=split_sigma_spatial,
        help="Override split-SDE sigma for spatial dimensions x1,x2.",
    )
    parser.add_argument(
        "--split-sigma-gene",
        type=float,
        default=split_sigma_gene,
        help="Override split-SDE sigma for non-spatial dimensions x3..xN.",
    )
    parser.add_argument(
        "--split-sigma-by-dim",
        default=split_sigma_by_dim,
        help="Optional comma-separated split-SDE sigma for every dimension x1..xN.",
    )
    parser.add_argument(
        "--spatial-warp-to-observed",
        action=argparse.BooleanOptionalAction,
        default=spatial_warp_to_observed,
        help=(
            "After split-SDE, warp only spatial dims x1,x2 toward each observed endpoint using a "
            "segment-wise nearest-neighbor displacement field. Gene dims are unchanged."
        ),
    )
    parser.add_argument(
        "--spatial-warp-to-observed-piecewise",
        action=argparse.BooleanOptionalAction,
        default=spatial_warp_to_observed_piecewise,
        help=(
            "Run split-SDE in observed-time segments; after each segment, warp only spatial dims x1,x2 "
            "toward the observed endpoint shape and use the warped endpoint as the next segment start."
        ),
    )
    parser.add_argument(
        "--spatial-warp-k",
        type=int,
        default=spatial_warp_k,
        help="Number of simulated endpoint anchors used to interpolate spatial warp displacements.",
    )
    parser.add_argument(
        "--spatial-warp-eps",
        type=float,
        default=spatial_warp_eps,
        help="Small positive constant for inverse-distance spatial warp weights.",
    )
    parser.add_argument(
        "--skip-nonsplit-sde",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Skip non-split SDE simulation and only run split/piecewise interpolation.",
    )
    parser.add_argument(
        "--plot-lineage-sankey",
        action=argparse.BooleanOptionalAction,
        default=plot_lineage_sankey,
        help="Plot a lineage Sankey (cell fate flow) over all timepoints using non-split SDE labels.",
    )
    parser.add_argument("--sankey-min-flow", type=float, default=sankey_min_flow)
    parser.add_argument(
        "--sankey-keep-source-cumfrac",
        type=float,
        default=sankey_keep_source_cumfrac,
        help=(
            "Optional proportion filter for the lineage Sankey: for each source label within each time slice, keep the "
            "strongest outgoing transitions until cumulative flow coverage reaches this fraction (e.g. 0.8). Range: (0, 1]."
        ),
    )
    parser.add_argument(
        "--sankey-style",
        choices=["default", "nature-methods"],
        default=sankey_style,
        help="Styling preset for the lineage Sankey (recommended: nature-methods).",
    )
    parser.add_argument(
        "--sankey-normalize-mode",
        choices=["none", "source", "global"],
        default="none" if sankey_normalize_mode is None else sankey_normalize_mode,
        help="Normalize Sankey link values: none|source|global.",
    )
    parser.add_argument(
        "--sde-n-samples",
        type=int,
        default=None,
        help="Number of unique initial cells sampled at the earliest observed timepoint for SDE simulation (no replacement).",
    )
    parser.add_argument(
        "--slice-max-cells-per-timepoint",
        "--max-cells-per-timepoint",
        dest="slice_max_cells_per_timepoint",
        type=int,
        default=None,
        help="Optional downsample per slice for attention/3D (leave unset to use all cells).",
    )
    parser.add_argument("--skip-snapshots", action="store_true", help="Skip 2D per-timepoint SVG/PDF snapshots")
    parser.add_argument("--snapshot-point-size", type=float, default=2.5)
    parser.add_argument("--snapshot-alpha", type=float, default=0.9)
    parser.add_argument("--mosaic-cols", type=int, default=4)
    parser.add_argument("--mosaic-cell-size", type=float, default=2.2)
    parser.add_argument("--mosaic-no-title", action="store_true", help="Hide titles in the timepoint mosaic panel")
    parser.add_argument(
        "--fate-min-flow",
        type=float,
        default=fate_min_flow,
        help="Minimum count threshold for rendering fate-flow ribbons (set 0 to disable).",
    )
    parser.add_argument(
        "--fate-keep-source-cumfrac",
        type=float,
        default=fate_keep_source_cumfrac,
        help=(
            "Optional proportion filter for fate-flow ribbons: for each source label, keep the strongest outgoing "
            "transitions until cumulative count coverage reaches this fraction (e.g. 0.8). Range: (0, 1]."
        ),
    )
    parser.add_argument("--random-seed", type=int, default=random_seed)
    parser.add_argument("--skip-export", action="store_true", help="Skip static svg/pdf/png export")
    parser.add_argument(
        "--save-dense-attention-matrix",
        action=argparse.BooleanOptionalAction,
        default=save_dense_attention_matrix,
        help=(
            "Save dense attention matrix per slice (N x N). "
            "Disable for large real timepoints to avoid O(N^2) memory usage."
        ),
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "run_args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    if args.random_seed is not None:
        random.seed(args.random_seed)
        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    config = load_config(args.config)
    if int(args.classifier_knn_neighbors) <= 0:
        raise ValueError("--classifier-knn-neighbors must be > 0")
    dim = int(config["data"]["dim"])
    split_sigma_scalar, split_sigma_vector = _resolve_split_sigma(dim, args)

    df = pd.read_csv(args.data_csv, low_memory=False)
    _require_columns(df, ["samples"] + [f"x{i}" for i in range(1, dim + 1)], ctx=args.data_csv)
    if args.annotation_key not in df.columns:
        raise ValueError(f"Expected '{args.annotation_key}' column in {args.data_csv}.")
    df = df.copy()
    df["samples"] = df["samples"].astype(float)
    df[args.annotation_key] = df[args.annotation_key].astype(str)
    # Ensure df['samples'].unique() (used inside simulate_sde_points*) is in time order.
    df = df.sort_values("samples").reset_index(drop=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    f_net, score_net, exp_dir, device = load_models(
        config,
        exp_name=config["exp"]["name"],
        device=device,
        model_tag="model_final",
        score_tag="score_model",
    )

    print("Project root:", PROJECT_ROOT)
    print("Device:", device)
    print("Experiment dir:", exp_dir)
    print("Data:", args.data_csv, "| rows:", len(df))
    if split_sigma_vector is None:
        print("Split-SDE sigma:", split_sigma_scalar)
    else:
        print(
            "Split-SDE sigma vector | "
            f"spatial={split_sigma_vector[: min(2, len(split_sigma_vector))]} "
            f"gene_head={split_sigma_vector[2:5]} "
            f"dim={len(split_sigma_vector)}"
        )
    print(
        "Sankey settings | keep_source_cumfrac=",
        args.sankey_keep_source_cumfrac,
        "| min_flow=",
        args.sankey_min_flow,
        "| normalize_mode=",
        args.sankey_normalize_mode,
    )
    if args.spatial_warp_to_observed and args.spatial_warp_to_observed_piecewise:
        raise ValueError(
            "Use only one of --spatial-warp-to-observed or --spatial-warp-to-observed-piecewise."
        )
    if args.spatial_warp_to_observed_piecewise and args.split_sde_piecewise:
        raise ValueError(
            "--spatial-warp-to-observed-piecewise conflicts with --split-sde-piecewise; "
            "both are segment-wise restart modes."
        )
    if args.spatial_warp_to_observed and args.split_sde_piecewise and (not args.split_sde_piecewise_include_end):
        args.split_sde_piecewise_include_end = True
        print(
            "[info] enabling --split-sde-piecewise-include-end because "
            "--spatial-warp-to-observed with --split-sde-piecewise needs simulated segment endpoints as warp anchors."
        )
    if args.spatial_warp_to_observed:
        print(
            "Spatial warp: enabled | "
            f"k={args.spatial_warp_k} eps={args.spatial_warp_eps} "
            f"piecewise={args.split_sde_piecewise}"
        )
    if args.spatial_warp_to_observed_piecewise:
        print(
            "Spatial warp piecewise: enabled | "
            f"k={args.spatial_warp_k} eps={args.spatial_warp_eps}"
        )

    observed_time_points = sorted(df["samples"].unique().tolist())
    requested_plot_points = _parse_csv_floats_or_all(args.plot_3d_time_points)
    required_obs_points = set(requested_plot_points) if requested_plot_points is not None else set()

    if args.max_observed_timepoints is not None:
        max_obs = int(args.max_observed_timepoints)
        if max_obs <= 0:
            raise ValueError("--max-observed-timepoints must be > 0")
        if max_obs < len(observed_time_points):
            required_obs = [t for t in observed_time_points if t in required_obs_points]
            if len(required_obs) > max_obs:
                raise ValueError(
                    f"--max-observed-timepoints={max_obs} is smaller than required observed plot points "
                    f"{sorted(required_obs)} from --plot-3d-time-points={args.plot_3d_time_points}"
                )
            remaining = [t for t in observed_time_points if t not in set(required_obs)]
            keep_extra = max_obs - len(required_obs)
            extra = _select_evenly_spaced(remaining, keep_extra) if keep_extra > 0 else []
            observed_time_points = sorted(set(required_obs + extra))
            print("Capped observed timepoints:", observed_time_points)
    interp_points = [] if args.no_interp else _parse_csv_floats(args.interp_time_points)
    interp_points = [float(t) for t in interp_points if float(t) not in observed_time_points]
    if args.split_sde_piecewise and len(interp_points) == 0:
        print("[warn] --split-sde-piecewise has no effect without interpolation points; disabling it.")
        args.split_sde_piecewise = False
    if args.spatial_warp_to_observed_piecewise and len(interp_points) == 0:
        print("[warn] --spatial-warp-to-observed-piecewise has no effect without interpolation points; disabling it.")
        args.spatial_warp_to_observed_piecewise = False
    if args.target_total_slices is not None and args.split_sde_piecewise:
        print("[warn] --target-total-slices is ignored when --split-sde-piecewise is enabled (needs full observed segments).")
    if args.target_total_slices is not None and args.spatial_warp_to_observed_piecewise:
        print(
            "[warn] --target-total-slices is ignored when --spatial-warp-to-observed-piecewise is enabled "
            "(needs full observed segments)."
        )
    if args.target_total_slices is not None and (not args.split_sde_piecewise) and (not args.spatial_warp_to_observed_piecewise):
        keep_observed = max(1, int(args.target_total_slices) - len(interp_points))
        if keep_observed < len(observed_time_points):
            observed_time_points = _select_evenly_spaced(observed_time_points, keep_observed)
            print("Selected observed timepoints:", observed_time_points)

    ts_points = sorted(set(observed_time_points + interp_points))
    time_keys = [str(t) for t in ts_points]

    plot_3d_points = requested_plot_points
    if plot_3d_points is None:
        plot_3d_ts_points = list(ts_points)
    else:
        ts_set = {float(t) for t in ts_points}
        missing = [float(t) for t in plot_3d_points if float(t) not in ts_set]
        if missing:
            raise ValueError(f"--plot-3d-time-points contains values not in ts_points: {missing} (ts_points={ts_points})")
        plot_3d_ts_points = [float(t) for t in plot_3d_points]
    plot_3d_time_keys = [str(t) for t in plot_3d_ts_points]

    need_interp = len(interp_points) > 0
    piecewise_x0_by_observed: Optional[Dict[float, np.ndarray]] = None
    piecewise_labels_by_observed: Optional[Dict[float, np.ndarray]] = None
    piecewise_endpoint_by_observed: Optional[Dict[float, np.ndarray]] = None
    if need_interp:
        print(
            "Interpolation enabled | interp_points=",
            interp_points,
            "| observed_time_points=",
            observed_time_points,
        )
        feature_cols_full = [f"x{i}" for i in range(1, dim + 1)]
        if args.classifier_n_pcs is None:
            classifier_feature_dim = dim
        else:
            classifier_feature_dim = max(1, min(int(args.classifier_n_pcs), dim))
        clf_cols = ["samples"] + [f"x{i}" for i in range(1, classifier_feature_dim + 1)]
        t_train0 = time.perf_counter()
        print(
            "Training classifier | rows=",
            len(df),
            "| features=",
            len(clf_cols),
            "| epochs=",
            args.classifier_epochs,
            "| hidden=",
            args.classifier_hidden,
            "| best_metric=",
            args.classifier_best_metric,
            "| full_data=",
            args.classifier_train_on_full_data,
            "| save_test_models=",
            args.classifier_save_test_models,
        )
        cache_dir = None
        if args.classifier_cache:
            cache_dir = args.classifier_cache_dir or os.path.join(args.output_dir, "classifier_cache")
        checkpoint_dir = None
        if args.classifier_save_test_models:
            if int(args.classifier_last_k_epochs) < 0:
                raise ValueError("--classifier-last-k-epochs must be >= 0")
            checkpoint_dir = args.classifier_checkpoint_dir or os.path.join(args.output_dir, "classifier_checkpoints")

        reuse_cached = bool(args.classifier_cache)
        if args.classifier_save_test_models and reuse_cached:
            reuse_cached = False
            print("[info] --classifier-save-test-models enabled; forcing retrain (cache reuse disabled).")

        model, label_encoder, acc = train_mlp_classifier(
            df,
            feature_cols=clf_cols,
            label_col=args.annotation_key,
            hidden_size=args.classifier_hidden,
            epochs=args.classifier_epochs,
            cache_dir=cache_dir,
            cache_tag=args.classifier_cache_tag,
            df_source_path=args.data_csv,
            reuse_if_possible=reuse_cached,
            progress=True,
            device=device,
            best_epoch_metric=args.classifier_best_metric,
            train_on_full_data=bool(args.classifier_train_on_full_data),
            checkpoint_dir=checkpoint_dir,
            save_best_acc=bool(args.classifier_save_test_models),
            save_best_bacc=bool(args.classifier_save_test_models),
            save_last_k_epochs=int(args.classifier_last_k_epochs),
        )
        eval_name = "train_full_acc" if args.classifier_train_on_full_data else "val_acc"
        print(f"Classifier trained in {time.perf_counter() - t_train0:.1f}s | {eval_name}={acc:.4f}")
        if checkpoint_dir is not None:
            print("Classifier checkpoints dir:", checkpoint_dir)

        # Use the number of cells at the earliest observed time as a stable sample size.
        t0 = float(min(observed_time_points))
        n_samples = int((df["samples"] == t0).sum())
        if args.sde_n_samples is not None:
            if args.sde_n_samples <= 0:
                raise ValueError("--sde-n-samples must be > 0")
            n_samples = min(n_samples, int(args.sde_n_samples))
        print("SDE n_samples (from t0):", n_samples)
        if args.split_sde_piecewise:
            print("Piecewise observed-start sampling mode:", args.piecewise_observed_sample_mode)

        # Non-split SDE (for trajectory labels)
        sde_points = None
        if args.skip_nonsplit_sde:
            print("Skipping non-split SDE (--skip-nonsplit-sde).")
        else:
            t_sde0 = time.perf_counter()
            print("Simulating non-split SDE...")
            sde_points, _ = simulate_sde_points(
                df=df,
                dim=dim,
                f_net=f_net,
                score_net=score_net,
                time_index=0,
                n_samples=n_samples,
                ts_points=ts_points,
                dt=sde_dt,
                sigma=0.0,
                include_score=False,
                device=device,
            )
            print(f"Non-split SDE done in {time.perf_counter() - t_sde0:.1f}s")

        # Split SDE (for interpolated slices in 3D)
        t_sde_split0 = time.perf_counter()
        print("Simulating split SDE...")
        sde_points_split_prewarp = None
        if args.spatial_warp_to_observed_piecewise:
            if args.spatial_warp_k <= 0:
                raise ValueError("--spatial-warp-k must be > 0")
            if args.spatial_warp_eps <= 0:
                raise ValueError("--spatial-warp-eps must be > 0")
            rng_warp_piecewise = np.random.default_rng(
                1 if args.random_seed is None else int(args.random_seed) + 1
            )
            x0_warp, _ = _sample_observed_x0(
                df,
                time_value=t0,
                feature_cols=feature_cols_full,
                label_col=args.annotation_key,
                n_samples_cap=n_samples,
                rng=rng_warp_piecewise,
            )
            sde_points_split = _simulate_piecewise_spatially_warped_split(
                x0=x0_warp,
                f_net=f_net,
                score_net=score_net,
                observed_time_points=observed_time_points,
                ts_points=ts_points,
                df=df,
                feature_cols_full=feature_cols_full,
                label_col=args.annotation_key,
                dt=split_sde_dt,
                sigma=split_sigma_scalar,
                sigma_by_dim=split_sigma_vector,
                growth_alpha=split_growth_alpha,
                interaction_m=1024,
                device=device,
                rng=rng_warp_piecewise,
                k=int(args.spatial_warp_k),
                eps=float(args.spatial_warp_eps),
            )
        elif args.split_sde_piecewise:
            feature_cols_full = [f"x{i}" for i in range(1, dim + 1)]
            rng_piecewise = np.random.default_rng(0 if args.random_seed is None else int(args.random_seed))
            x0_by_observed: Dict[float, np.ndarray] = {}
            labels0_by_observed: Dict[float, np.ndarray] = {}
            if args.piecewise_observed_sample_mode == "per_timepoint":
                piecewise_cap = None if args.sde_n_samples is None else int(args.sde_n_samples)
            else:
                piecewise_cap = int(n_samples)
            for t_obs in observed_time_points:
                x0_t, labels_t = _sample_observed_x0(
                    df,
                    time_value=float(t_obs),
                    feature_cols=feature_cols_full,
                    label_col=args.annotation_key,
                    # Per observed-start sampling policy (global t0 cap vs per-timepoint cap).
                    n_samples_cap=piecewise_cap,
                    rng=rng_piecewise,
                )
                x0_by_observed[float(t_obs)] = x0_t
                labels0_by_observed[float(t_obs)] = labels_t
            piecewise_x0_by_observed = x0_by_observed
            piecewise_labels_by_observed = labels0_by_observed
            piecewise_endpoint_by_observed = {}

            points_by_time: Dict[float, np.ndarray] = {}
            # Always define observed points as their (optionally capped) real distributions,
            # so each segment can restart from observed.
            for t_obs in observed_time_points:
                points_by_time[float(t_obs)] = x0_by_observed[float(t_obs)]

            observed_set = {float(t) for t in observed_time_points}
            for t_start, t_end in zip(observed_time_points[:-1], observed_time_points[1:]):
                mids = sorted([t for t in interp_points if float(t_start) < float(t) < float(t_end)])
                if (not mids) and (not args.split_sde_piecewise_include_end):
                    continue
                seg_ts: list[float] = [float(t_start)] + [float(t) for t in mids]
                if args.split_sde_piecewise_include_end:
                    seg_ts.append(float(t_end))
                print(f"[piecewise split-SDE] segment {t_start}->{t_end} | targets={seg_ts}")
                seg_points = _simulate_sde_points_split_from_x0(
                    x0=x0_by_observed[float(t_start)],
                    f_net=f_net,
                    score_net=score_net,
                    ts_points=seg_ts,
                    dt=split_sde_dt,
                    sigma=split_sigma_scalar,
                    sigma_by_dim=split_sigma_vector,
                    growth_alpha=split_growth_alpha,
                    interaction_m=1024,
                    device=device,
                    verbose=True,
                )
                for t_val, pts in zip(seg_ts, seg_points):
                    # Keep observed slices anchored to observed distributions (restart points).
                    if float(t_val) in observed_set:
                        if args.split_sde_piecewise_include_end and float(t_val) == float(t_end):
                            piecewise_endpoint_by_observed[float(t_end)] = np.asarray(pts, dtype=np.float32)
                        continue
                    points_by_time[float(t_val)] = np.asarray(pts, dtype=np.float32)

            missing = [float(t) for t in ts_points if float(t) not in points_by_time]
            if missing:
                raise ValueError(f"Piecewise split-SDE missing timepoints: {missing}")
            sde_points_split = np.array([points_by_time[float(t)] for t in ts_points], dtype=object)
        else:
            sde_points_split = simulate_sde_points_split(
                df=df,
                dim=dim,
                f_net=f_net,
                score_net=score_net,
                time_index=0,
                n_samples=n_samples,
                ts_points=ts_points,
                dt=split_sde_dt,
                sigma=split_sigma_scalar,
                sigma_by_dim=split_sigma_vector,
                growth_alpha=split_growth_alpha,
                device=device,
            )
        if args.spatial_warp_to_observed:
            sde_points_split_prewarp = np.array(
                [np.asarray(p, dtype=np.float32).copy() for p in sde_points_split],
                dtype=object,
            )
            if args.spatial_warp_k <= 0:
                raise ValueError("--spatial-warp-k must be > 0")
            if args.spatial_warp_eps <= 0:
                raise ValueError("--spatial-warp-eps must be > 0")
            rng_warp = np.random.default_rng(
                1 if args.random_seed is None else int(args.random_seed) + 1
            )
            sde_points_split = _apply_spatial_warp_to_segments(
                sde_points_split=sde_points_split,
                ts_points=ts_points,
                observed_time_points=observed_time_points,
                df=df,
                feature_cols_full=feature_cols_full,
                label_col=args.annotation_key,
                rng=rng_warp,
                piecewise=bool(args.split_sde_piecewise),
                piecewise_include_end=bool(args.split_sde_piecewise_include_end),
                piecewise_endpoint_by_observed=piecewise_endpoint_by_observed,
                use_real_for_observed=bool(args.use_real_for_observed),
                k=int(args.spatial_warp_k),
                eps=float(args.spatial_warp_eps),
            )
        print(f"Split SDE done in {time.perf_counter() - t_sde_split0:.1f}s")

        predicted_labels_list = None
        if sde_points is not None:
            predicted_labels_list = predict_labels_for_trajectories(
                sde_points=sde_points,
                ts_points=ts_points,
                model=model,
                label_encoder=label_encoder,
                feature_dim=classifier_feature_dim,
                device=device,
                knn_neighbors=int(args.classifier_knn_neighbors),
            )

        predicted_labels_split = predict_labels_for_trajectories(
            sde_points=sde_points_split,
            ts_points=ts_points,
            model=model,
            label_encoder=label_encoder,
            feature_dim=classifier_feature_dim,
            device=device,
            knn_neighbors=int(args.classifier_knn_neighbors),
        )
        predicted_labels_split_prewarp = None
        if sde_points_split_prewarp is not None:
            predicted_labels_split_prewarp = predict_labels_for_trajectories(
                sde_points=sde_points_split_prewarp,
                ts_points=ts_points,
                model=model,
                label_encoder=label_encoder,
                feature_dim=classifier_feature_dim,
                device=device,
                knn_neighbors=int(args.classifier_knn_neighbors),
            )
    else:
        predicted_labels_list = None
        sde_points_split = None
        predicted_labels_split = None
        predicted_labels_split_prewarp = None

    # Build adata_dict for 3D plot + attention
    import anndata as ad

    feature_cols_full = [f"x{i}" for i in range(1, dim + 1)]
    adata_dict = {}
    rng = np.random.default_rng(0 if args.random_seed is None else int(args.random_seed))
    for t in ts_points:
        key = str(t)
        if args.use_real_for_observed and (t in observed_time_points):
            subset = df[df["samples"] == float(t)]
            X = subset[feature_cols_full].values.astype(np.float32)
            labels = subset[args.annotation_key].astype(str).values
        elif (not args.use_real_for_observed) and args.split_sde_piecewise and (t in observed_time_points):
            if sde_points_split is None or predicted_labels_split is None:
                raise ValueError("Piecewise split-SDE observed slice requested but split outputs are missing.")
            if args.spatial_warp_to_observed and float(t) != float(min(observed_time_points)):
                idx = ts_points.index(t)
                X = np.array(sde_points_split[idx], dtype=np.float32)
                labels = np.asarray(predicted_labels_split[idx]).astype(str)
            else:
                if piecewise_x0_by_observed is None or piecewise_labels_by_observed is None:
                    raise ValueError("Piecewise split-SDE enabled but observed x0/labels cache is missing.")
                X = np.asarray(piecewise_x0_by_observed[float(t)], dtype=np.float32)
                # In piecewise mode, the first observed slice still uses the observed restart distribution.
                labels = np.asarray(piecewise_labels_by_observed[float(t)]).astype(str)
        else:
            if sde_points_split is None or predicted_labels_split is None:
                raise ValueError("Interpolation requested but split SDE outputs are missing.")
            idx = ts_points.index(t)
            X = np.array(sde_points_split[idx], dtype=np.float32)
            labels = np.asarray(predicted_labels_split[idx]).astype(str)

        X, labels = _downsample_xy(X, labels, args.slice_max_cells_per_timepoint, rng)
        adata_t = ad.AnnData(X=X)
        adata_t.obs[args.annotation_key] = labels
        adata_t.obsm["spatial"] = X[:, :2]
        adata_dict[key] = adata_t

    # Colors
    label_to_color = load_label_to_color(
        df[args.annotation_key].astype(str).values,
        label_color_json=args.label_color_json,
        color_h5ad=args.color_h5ad,
        annotation_key=args.annotation_key,
    )
    with open(os.path.join(args.output_dir, "label_to_color.json"), "w", encoding="utf-8") as f:
        json.dump(label_to_color, f, indent=2)

    # Lineage Sankey (cell fate flow) over all timepoints (0, 0.5, 1, 1.5, 2, 2.5, 3 by default)
    fig_sankey = None
    if args.plot_lineage_sankey:
        if predicted_labels_list is None:
            print("[warn] --plot-lineage-sankey requested but interpolation/classifier was not run; skipping.")
        else:
            from evaluation.arista_code.arista_helpers import plot_sankey

            sankey_path = os.path.join(args.output_dir, "lineage_sankey.html")
            normalize_mode = None if args.sankey_normalize_mode == "none" else args.sankey_normalize_mode
            fig_sankey = plot_sankey(
                predicted_labels_list=predicted_labels_list,
                out_html=sankey_path,
                time_keys=time_keys,
                show_time_axis=True,
                min_flow=args.sankey_min_flow,
                keep_source_cumfrac=args.sankey_keep_source_cumfrac,
                normalize_mode=normalize_mode,
                label_to_color=label_to_color,
                style=args.sankey_style,
                title="Cell Fate Transitions",
            )
            print("Saved:", sankey_path)

    if not args.skip_snapshots:
        snapshot_dir = os.path.join(args.output_dir, "timepoint_svg")
        observed_variants = None
        if len(observed_time_points) > 0:
            feature_cols_snapshot = [f"x{i}" for i in range(1, dim + 1)]
            rng_snapshot = np.random.default_rng(
                100 if args.random_seed is None else int(args.random_seed) + 100
            )
            ts_index = {float(t): i for i, t in enumerate(ts_points)}
            compare_variants = {}
            has_generated_compare = False

            for t_obs in observed_time_points:
                t_obs_f = float(t_obs)
                subset_obs = df[df["samples"] == t_obs_f]
                if subset_obs.empty:
                    continue

                X_obs = subset_obs[feature_cols_snapshot].values.astype(np.float32)
                labels_obs = subset_obs[args.annotation_key].astype(str).values
                X_obs, labels_obs = _downsample_xy(
                    X_obs,
                    labels_obs,
                    args.slice_max_cells_per_timepoint,
                    rng_snapshot,
                )
                compare_variants.setdefault(t_obs_f, {})["observed"] = (
                    np.asarray(X_obs, dtype=np.float32)[:, :2],
                    np.asarray(labels_obs).astype(str),
                )

                X_gen = None
                labels_gen = None
                idx = ts_index.get(t_obs_f)
                use_final_generated = (
                    args.spatial_warp_to_observed_piecewise
                    or (not args.split_sde_piecewise)
                    or (
                        args.split_sde_piecewise
                        and args.spatial_warp_to_observed
                        and (not args.use_real_for_observed)
                    )
                )
                if (
                    use_final_generated
                    and sde_points_split is not None
                    and predicted_labels_split is not None
                    and idx is not None
                ):
                    X_gen = np.asarray(sde_points_split[idx], dtype=np.float32)
                    labels_gen = np.asarray(predicted_labels_split[idx]).astype(str)
                elif (
                    piecewise_endpoint_by_observed is not None
                    and t_obs_f in piecewise_endpoint_by_observed
                ):
                    X_gen = np.asarray(piecewise_endpoint_by_observed[t_obs_f], dtype=np.float32)
                    labels_gen = _predict_labels_for_points(
                        points=X_gen,
                        time_value=t_obs_f,
                        model=model,
                        label_encoder=label_encoder,
                        feature_dim=classifier_feature_dim,
                        device=device,
                        knn_neighbors=int(args.classifier_knn_neighbors),
                    )

                if X_gen is not None and labels_gen is not None:
                    X_gen, labels_gen = _downsample_xy(
                        X_gen,
                        labels_gen,
                        args.slice_max_cells_per_timepoint,
                        rng_snapshot,
                    )
                    compare_variants.setdefault(t_obs_f, {})["generated"] = (
                        np.asarray(X_gen, dtype=np.float32)[:, :2],
                        np.asarray(labels_gen).astype(str),
                    )
                    has_generated_compare = True

            if has_generated_compare:
                observed_variants = compare_variants

            # Preserve pre/post-warp inspection panels for interior interpolation slices.
            if (
                args.split_sde_piecewise
                and args.split_sde_piecewise_include_end
                and args.spatial_warp_to_observed
                and sde_points_split_prewarp is not None
                and predicted_labels_split_prewarp is not None
                and sde_points_split is not None
                and predicted_labels_split is not None
            ):
                if observed_variants is None:
                    observed_variants = {}
                observed_set = {float(t) for t in observed_time_points}
                for t_val in ts_points:
                    t_float = float(t_val)
                    if t_float in observed_set:
                        continue
                    idx = ts_index.get(t_float)
                    if idx is None:
                        continue
                    observed_variants.setdefault(t_float, {})["prewarp"] = (
                        np.asarray(sde_points_split_prewarp[idx], dtype=np.float32)[:, :2],
                        np.asarray(predicted_labels_split_prewarp[idx]).astype(str),
                    )
                    observed_variants.setdefault(t_float, {})["postwarp"] = (
                        np.asarray(sde_points_split[idx], dtype=np.float32)[:, :2],
                        np.asarray(predicted_labels_split[idx]).astype(str),
                    )
        save_timepoint_snapshots(
            adata_dict=adata_dict,
            time_keys=time_keys,
            annotation_key=args.annotation_key,
            label_to_color=label_to_color,
            observed_time_points=observed_time_points,
            observed_variants=observed_variants,
            snapshot_dir=snapshot_dir,
            background_color=background_color,
            font_color=font_color,
            snapshot_point_size=args.snapshot_point_size,
            snapshot_alpha=args.snapshot_alpha,
            mosaic_cols=args.mosaic_cols,
            mosaic_cell_size=args.mosaic_cell_size,
            mosaic_show_title=(not args.mosaic_no_title),
            save_pdf=bool(export_pdf),
        )
        print("Saved snapshots to:", snapshot_dir)

    # Compute communication matrices (for the 3D communication plot).
    # By default we only compute for the timepoints that will be rendered in 3D
    # (e.g. 0, 0.5, 1). This is much faster than computing all `ts_points`.
    attn_dir = os.path.join(args.output_dir, "attention")
    os.makedirs(attn_dir, exist_ok=True)

    all_time_communications = {}
    for t in plot_3d_ts_points:
        key = str(t)
        adata_t = adata_dict[key]
        print("Time", key, "cells", adata_t.n_obs)

        attn_out = save_interpolated_attention(
            adata_t,
            time_value=float(t),
            f_net=f_net,
            device=device,
            out_dir=attn_dir,
            save_dense_matrix=bool(args.save_dense_attention_matrix),
        )

        comm = analyze_attention_by_celltype(
            edge_index=attn_out["edge_index"],
            attn=attn_out["attn_mean"],
            labels=adata_t.obs[args.annotation_key].values,
            spatial_coord=adata_t.obsm["spatial"],
            time_title=key,
            remove_self_loop=remove_self_loop,
            winsor_quantile=winsor_quantile,
            distance_bins=None,
            n_permutations=0,
            plot=False,
        )
        all_time_communications[key] = comm

    import pickle

    comm_path = os.path.join(args.output_dir, "mosta_all_time_communications.pkl")
    with open(comm_path, "wb") as f:
        pickle.dump(all_time_communications, f)
    print("Saved:", comm_path)

    # 3D plot
    focus_source_only = fate_focus_mode == "source"
    focus_target_only = fate_focus_mode == "target"

    spatiotemporal_path = os.path.join(args.output_dir, "spatiotemporal_3d.html")

    adata_dict_3d = {k: adata_dict[k] for k in plot_3d_time_keys}
    comm_3d = {k: all_time_communications[k] for k in plot_3d_time_keys}
    predicted_labels_3d = None
    if predicted_labels_list is not None:
        idxs = [ts_points.index(float(t)) for t in plot_3d_ts_points]
        predicted_labels_3d = [predicted_labels_list[i] for i in idxs]
    else:
        # no-interp path: use observed labels directly so 3D plotting does not
        # assume classifier outputs exist.
        predicted_labels_3d = [
            np.asarray(adata_dict_3d[k].obs[args.annotation_key]).astype(str)
            for k in plot_3d_time_keys
        ]
    observed_time_points_3d = [float(t) for t in observed_time_points if float(t) in set(plot_3d_ts_points)]
    interp_points_3d = [float(t) for t in interp_points if float(t) in set(plot_3d_ts_points)]

    fig_3d = plot_3d_spatial_sankey_style_focus_anchor(
        adata_dict=adata_dict_3d,
        all_time_communications=comm_3d,
        time_keys=plot_3d_time_keys,
        label_to_color=label_to_color,
        predicted_labels_list=predicted_labels_3d,
        spatial_key="spatial",
        z_spacing=z_spacing,
        reverse_time_order=reverse_time_order,
        intra_threshold=comm_edge_threshold,
        edge_focus_celltype=comm_focus_label,
        edge_top_k=comm_edge_top_k,
        edge_top_k_focus_label=comm_edge_top_k_focus_label,
        ribbon_min_count=float(args.fate_min_flow) if args.fate_min_flow is not None else None,
        ribbon_keep_source_cumfrac=args.fate_keep_source_cumfrac,
        ribbon_focus_celltype=fate_focus_label,
        ribbon_focus_source_only=focus_source_only if fate_focus_label else False,
        ribbon_focus_target_only=focus_target_only if fate_focus_label else False,
        background_color=background_color,
        font_color=font_color,
        anchor_mode="centroid",
        anchor_subsample=1000,
        highlight_endpoints=True,
        endpoint_size=6,
        endpoint_opacity=0.9,
        edge_color=comm_edge_color,
        edge_line_width_base=5,
        edge_line_width_scale=0.7,
        bidirectional_offset=0.2,
        bidirectional_curve=True,
        bidirectional_curve_points=18,
        ribbon_line_width_base=6,
        ribbon_line_width_scale=1.0,
        ribbon_line_alpha=0.55,
        ribbon_line_curve=0.12,
        ribbon_line_points=18,
        point_size=1.0,
        observed_point_subsample=None,
        generated_point_subsample=None,
        observed_point_alpha=0.7,
        generated_point_alpha=0.7,
        slices_only=False,
        show_time_axis=False,
        show_legend=False,
        show_title=False,
        show_slice_border=True,
        slice_border_width=slice_border_width,
        slice_border_color_observed=slice_border_color_observed,
        slice_border_color_generated=slice_border_color_generated,
        slice_fill_color_observed=slice_fill_color_observed,
        slice_fill_color_generated=slice_fill_color_generated,
        slice_fill_opacity=slice_fill_opacity,
        observed_time_points=observed_time_points_3d,
        generated_time_points=interp_points_3d,
        width=1400,
        height=1000,
        out_html=spatiotemporal_path,
        focus_anchor_label=focus_anchor_label,
        focus_anchor_k=focus_anchor_k,
        focus_anchor_frac=focus_anchor_frac,
        focus_anchor_radius=focus_anchor_radius,
        focus_anchor_min_count=focus_anchor_min_count,
        annotation_key=args.annotation_key,
    )

    print("Saved:", spatiotemporal_path)
    try:
        fig_3d.update_layout(
            scene_camera=dict(
                eye=dict(x=1.7, y=1.0, z=0.9),
                projection=dict(type="orthographic"),
            ),
            margin=dict(l=10, r=10, t=10, b=10),
            scene=dict(
                domain=dict(x=[0.0, 1.0], y=[0.0, 1.0]),
                aspectratio=dict(x=1.2, y=1.0, z=1.6),
            ),
            font=dict(family="Helvetica", size=16, color=font_color),
        )
    except Exception:
        pass

    try:
        if args.skip_export:
            return
        import plotly.io as pio

        if fig_sankey is not None:
            if export_svg:
                pio.write_image(fig_sankey, os.path.join(args.output_dir, "lineage_sankey.svg"), scale=vector_scale)
            if export_pdf:
                pio.write_image(fig_sankey, os.path.join(args.output_dir, "lineage_sankey.pdf"), scale=vector_scale)
        if export_svg:
            pio.write_image(fig_3d, os.path.join(args.output_dir, "spatiotemporal_3d.svg"), scale=vector_scale)
        if export_pdf:
            pio.write_image(fig_3d, os.path.join(args.output_dir, "spatiotemporal_3d.pdf"), scale=vector_scale)
        if export_png:
            pio.write_image(fig_3d, os.path.join(args.output_dir, "spatiotemporal_3d.png"), scale=png_scale)
        print("Exported vector/bitmap files.")
        print("Note: Plotly 3D exports are rasterized inside SVG/PDF.")
    except Exception as exc:
        print("Export failed (likely missing kaleido):", exc)


if __name__ == "__main__":
    main()
