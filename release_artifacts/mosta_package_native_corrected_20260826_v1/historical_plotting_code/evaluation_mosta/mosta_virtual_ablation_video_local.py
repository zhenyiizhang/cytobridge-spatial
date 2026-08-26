#!/usr/bin/env python3
"""
MOSTA virtual tissue ablation video (baseline vs counterfactual).

This script builds two forward branches on the MOSTA 4-timepoint data:
- baseline: start from observed t0 pool
- ablation: remove a target Annotation at t0, then forward simulate

Outputs:
- side-by-side 3D PNG frames for each continuous timepoint
- GIF assembled from frames
- summary.json with run metadata and sanity checks

Default setup follows the agreed project spec:
- target label: Cartilage primordium
- hard ablation at t=0.0
- strict counterfactual (no re-anchoring to observed t=1,2,3)
- time horizon: 0.0 -> 3.0 with step 0.1
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import sys
import time
from typing import Dict, Optional, Sequence

import imageio.v2 as imageio
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
    save_interpolated_attention,
    train_mlp_classifier,
)
from evaluation.common.nature_animation_style import (  # noqa: E402
    apply_nature_methods_mpl_style,
    get_nature_scatter_defaults,
)

DEFAULT_SHARED_CLASSIFIER_CACHE_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "mosta_interp_0_3_0208_n_pc_12",
    "classifier_cache",
)
NATURE_SCATTER_DEFAULTS = get_nature_scatter_defaults()
apply_nature_methods_mpl_style()


def _require_columns(df: pd.DataFrame, cols: Sequence[str], ctx: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {ctx}: {missing}")


def _float_close(a: float, b: float, eps: float = 1e-8) -> bool:
    return abs(float(a) - float(b)) <= eps


def _find_matching_time(time_value: float, observed: Sequence[float], eps: float = 1e-8) -> float:
    for t in observed:
        if _float_close(float(time_value), float(t), eps=eps):
            return float(t)
    raise ValueError(f"Timepoint {time_value} not found in observed times: {sorted(float(x) for x in observed)}")


def _build_ts_points(time_start: float, time_end: float, time_step: float) -> list[float]:
    if time_step <= 0:
        raise ValueError("--time-step must be > 0")
    if time_end < time_start:
        raise ValueError("--time-end must be >= --time-start")

    n_steps = int(round((time_end - time_start) / time_step))
    if n_steps < 0:
        raise ValueError("Invalid time range.")

    points = [round(float(time_start + i * time_step), 10) for i in range(n_steps + 1)]
    if not _float_close(points[-1], time_end):
        points.append(round(float(time_end), 10))

    # De-duplicate and sort stably.
    out: list[float] = []
    for x in points:
        if len(out) == 0 or not _float_close(x, out[-1]):
            out.append(x)
    return out


def _tag_float(v: float) -> str:
    s = f"{float(v):.6f}".rstrip("0").rstrip(".")
    s = s.replace("-", "m").replace(".", "p")
    return s


def _default_color_map(labels: Sequence[str]) -> Dict[str, str]:
    import matplotlib.pyplot as plt

    uniq = list(dict.fromkeys(str(x) for x in labels))
    cmap = plt.get_cmap("tab20")
    out: Dict[str, str] = {}
    for idx, lab in enumerate(uniq):
        rgb = cmap(idx % cmap.N)[:3]
        out[str(lab)] = "#{:02x}{:02x}{:02x}".format(
            int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
        )
    return out


def load_label_to_color(
    labels: np.ndarray,
    label_color_json: Optional[str] = None,
    color_h5ad: Optional[str] = None,
    annotation_key: str = "Annotation",
) -> Dict[str, str]:
    if label_color_json and os.path.exists(label_color_json):
        with open(label_color_json, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict) and len(obj) > 0:
            return {str(k): str(v) for k, v in obj.items()}

    if color_h5ad and os.path.exists(color_h5ad):
        try:
            import anndata as ad

            adata = ad.read_h5ad(color_h5ad, backed="r")
            try:
                key = annotation_key if annotation_key in adata.obs else None
                if key is None and annotation_key.lower() in adata.obs:
                    key = annotation_key.lower()
                if key is not None:
                    colors_key = f"{key}_colors"
                    colors = adata.uns.get(colors_key)
                    if colors is not None:
                        cats = (
                            adata.obs[key].cat.categories
                            if hasattr(adata.obs[key], "cat")
                            else sorted(adata.obs[key].astype(str).unique())
                        )
                        return {str(c): str(col) for c, col in zip(cats, colors)}
            finally:
                try:
                    adata.file.close()
                except Exception:
                    pass
        except Exception as exc:
            print(f"[warn] Failed to load colors from h5ad: {exc}")

    return _default_color_map(labels)


def _sample_rows_no_replace(
    X: np.ndarray,
    y: np.ndarray,
    n: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if n <= 0:
        raise ValueError("Sample size must be > 0")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y size mismatch")
    if n > X.shape[0]:
        raise ValueError(f"Requested n={n} > pool size {X.shape[0]}")
    idx = rng.choice(X.shape[0], size=n, replace=False)
    return X[idx], y[idx]


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


def _apply_spatial_warp_to_observed_segments(
    *,
    sde_points: np.ndarray,
    ts_points: Sequence[float],
    observed_times: Sequence[float],
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    annotation_key: str,
    rng: np.random.Generator,
    k: int,
    eps: float,
) -> np.ndarray:
    if len(ts_points) == 0 or len(observed_times) < 2:
        return sde_points

    out = np.array([np.asarray(p, dtype=np.float32).copy() for p in sde_points], dtype=object)
    ts_index = {float(t): i for i, t in enumerate(ts_points)}

    for t_start, t_end in zip(observed_times[:-1], observed_times[1:]):
        t_start = float(t_start)
        t_end = float(t_end)
        idx_end = ts_index.get(t_end)
        if idx_end is None:
            continue

        source_endpoint_xy = np.asarray(out[idx_end], dtype=np.float32)[:, :2]
        if source_endpoint_xy.shape[0] == 0:
            continue

        df_target = df[df["samples"] == t_end]
        X_target_pool = df_target[list(feature_cols)].values.astype(np.float32)
        y_target_pool = df_target[annotation_key].astype(str).values
        if X_target_pool.shape[0] == 0:
            continue
        X_target, _ = _sample_rows_no_replace(
            X_target_pool,
            y_target_pool,
            n=min(int(source_endpoint_xy.shape[0]), int(X_target_pool.shape[0])),
            rng=rng,
        )
        target_endpoint_xy = np.asarray(X_target, dtype=np.float32)[:, :2]

        segment_apply_ts = sorted([float(t) for t in ts_points if t_start < float(t) <= t_end])
        for t_val in segment_apply_ts:
            idx = ts_index.get(float(t_val))
            if idx is None:
                continue
            pts = np.asarray(out[idx], dtype=np.float32)
            if pts.shape[0] == 0:
                continue
            alpha = (float(t_val) - t_start) / max(t_end - t_start, float(eps))
            disp = _compute_spatial_warp_displacements(
                pts[:, :2],
                source_endpoint_xy,
                target_endpoint_xy,
                k=k,
                eps=eps,
            )
            pts[:, :2] = pts[:, :2] + float(alpha) * disp
            out[idx] = pts

        print(
            f"[spatial-warp] baseline segment {t_start}->{t_end} | "
            f"anchors_sim={source_endpoint_xy.shape[0]} anchors_real={target_endpoint_xy.shape[0]} "
            f"targets={len(segment_apply_ts)}"
        )

    return out


def _simulate_piecewise_spatially_warped_baseline(
    *,
    x0: np.ndarray,
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    annotation_key: str,
    observed_times: Sequence[float],
    ts_points: Sequence[float],
    f_net,
    score_net,
    dt: float,
    sigma: float,
    growth_alpha: float,
    interaction_m: int,
    interaction_scale: float,
    device: str,
    rng: np.random.Generator,
    warp_k: int,
    warp_eps: float,
) -> np.ndarray:
    ts_sorted = [float(t) for t in ts_points]
    obs_sorted = [float(t) for t in observed_times if float(ts_sorted[0]) <= float(t) <= float(ts_sorted[-1])]
    if len(obs_sorted) < 2:
        return _simulate_sde_points_split_from_x0(
            x0=x0,
            f_net=f_net,
            score_net=score_net,
            ts_points=ts_sorted,
            dt=dt,
            sigma=sigma,
            growth_alpha=growth_alpha,
            interaction_m=interaction_m,
            interaction_scale=interaction_scale,
            device=device,
            verbose=True,
        )

    current_x0 = np.asarray(x0, dtype=np.float32)
    points_by_time: Dict[float, np.ndarray] = {}

    for t_start, t_end in zip(obs_sorted[:-1], obs_sorted[1:]):
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
            growth_alpha=growth_alpha,
            interaction_m=interaction_m,
            interaction_scale=interaction_scale,
            device=device,
            verbose=True,
        )

        source_endpoint_xy = np.asarray(seg_points[-1], dtype=np.float32)[:, :2]
        df_target = df[df["samples"] == float(t_end)]
        X_target_pool = df_target[list(feature_cols)].values.astype(np.float32)
        y_target_pool = df_target[annotation_key].astype(str).values
        if X_target_pool.shape[0] == 0:
            raise ValueError(f"No observed rows available at endpoint time {t_end}")
        X_target, _ = _sample_rows_no_replace(
            X_target_pool,
            y_target_pool,
            n=min(int(source_endpoint_xy.shape[0]), int(X_target_pool.shape[0])),
            rng=rng,
        )
        target_endpoint_xy = np.asarray(X_target, dtype=np.float32)[:, :2]

        for t_val, pts_raw in zip(seg_ts, seg_points):
            pts = np.asarray(pts_raw, dtype=np.float32).copy()
            alpha = (float(t_val) - float(t_start)) / max(float(t_end - t_start), float(warp_eps))
            if alpha > 0.0:
                disp = _compute_spatial_warp_displacements(
                    pts[:, :2],
                    source_endpoint_xy,
                    target_endpoint_xy,
                    k=int(warp_k),
                    eps=float(warp_eps),
                )
                pts[:, :2] = pts[:, :2] + float(alpha) * disp
            points_by_time[float(t_val)] = pts

        current_x0 = np.asarray(points_by_time[float(t_end)], dtype=np.float32).copy()

    missing = [float(t) for t in ts_sorted if float(t) not in points_by_time]
    if missing:
        # Simulate any trailing times beyond the last observed anchor without extra warp.
        last_known_t = max(points_by_time.keys())
        trailing_ts = [float(t) for t in ts_sorted if float(t) > float(last_known_t)]
        if trailing_ts:
            seg_ts = [float(last_known_t)] + trailing_ts
            trailing_points = _simulate_sde_points_split_from_x0(
                x0=current_x0,
                f_net=f_net,
                score_net=score_net,
                ts_points=seg_ts,
                dt=dt,
                sigma=sigma,
                growth_alpha=growth_alpha,
                interaction_m=interaction_m,
                interaction_scale=interaction_scale,
                device=device,
                verbose=True,
            )
            for t_val, pts_raw in zip(seg_ts[1:], trailing_points[1:]):
                points_by_time[float(t_val)] = np.asarray(pts_raw, dtype=np.float32)

    missing = [float(t) for t in ts_sorted if float(t) not in points_by_time]
    if missing:
        raise ValueError(f"Piecewise spatial-warp baseline missing timepoints: {missing}")

    return np.array([points_by_time[float(t)] for t in ts_sorted], dtype=object)


def _parse_target_labels(target_labels_csv: str, target_label_fallback: str) -> list[str]:
    raw = str(target_labels_csv or "").strip()
    if raw:
        parts = [x.strip() for x in raw.split(",") if x.strip()]
    else:
        parts = [str(target_label_fallback).strip()] if str(target_label_fallback).strip() else []
    # de-duplicate with stable order
    dedup = list(dict.fromkeys(parts))
    if len(dedup) == 0:
        raise ValueError("No valid target labels resolved. Provide --target-label or --target-labels.")
    return dedup


def _format_target_labels(target_labels: Sequence[str]) -> str:
    labels = [str(x) for x in target_labels]
    if len(labels) == 1:
        return labels[0]
    return ",".join(labels)


def _apply_label_ablation(
    X_pool: np.ndarray,
    labels_pool: np.ndarray,
    target_labels: Sequence[str],
    remove_frac: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, int, int, Dict[str, int], Dict[str, int]]:
    if not (0.0 <= float(remove_frac) <= 1.0):
        raise ValueError("--ablation-remove-frac must be in [0, 1]")

    labels_pool = np.asarray(labels_pool).astype(str)
    target_labels = [str(x) for x in target_labels]
    target_labels = list(dict.fromkeys([x for x in target_labels if x]))
    if len(target_labels) == 0:
        raise ValueError("target_labels must be non-empty")

    available_set = set(labels_pool.tolist())
    missing_targets = [x for x in target_labels if x not in available_set]
    if len(missing_targets) > 0:
        raise ValueError(
            f"Target label(s) not found in ablation start pool: {missing_targets}. "
            f"Available labels include: {sorted(available_set)[:20]}"
        )

    target_idx = np.where(np.isin(labels_pool, target_labels))[0]
    n_target = int(target_idx.shape[0])

    if n_target == 0:
        raise ValueError(
            f"No target labels found in ablation start pool: {target_labels}. "
            f"Available labels include: {sorted(available_set)[:20]}"
        )

    n_target_per_label: Dict[str, int] = {}
    n_remove_per_label: Dict[str, int] = {}
    remove_idx_list: list[np.ndarray] = []

    for label in target_labels:
        label_idx = np.where(labels_pool == label)[0]
        n_t = int(label_idx.shape[0])
        n_target_per_label[label] = n_t
        n_r = int(round(n_t * float(remove_frac)))
        n_r = max(0, min(n_t, n_r))
        if remove_frac > 0 and n_r == 0 and n_t > 0:
            n_r = 1
        n_remove_per_label[label] = n_r
        if n_r > 0:
            remove_idx_list.append(rng.choice(label_idx, size=n_r, replace=False))

    if len(remove_idx_list) > 0:
        remove_idx = np.concatenate(remove_idx_list, axis=0)
    else:
        remove_idx = np.zeros((0,), dtype=np.int64)
    n_remove = int(remove_idx.shape[0])

    keep_mask = np.ones(labels_pool.shape[0], dtype=bool)
    if n_remove > 0:
        keep_mask[remove_idx] = False

    X_kept = np.asarray(X_pool)[keep_mask]
    y_kept = labels_pool[keep_mask]
    return X_kept, y_kept, n_target, n_remove, n_target_per_label, n_remove_per_label


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
    interaction_scale: float,
    device: str,
    verbose: bool = True,
) -> np.ndarray:
    """
    Split-SDE simulation starting from user-provided x0.

    Mirrors arista helper behavior but avoids internal resampling from dataframe.
    """
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
                if abs(float(interaction_scale)) <= 1e-12:
                    net_forces = torch.zeros_like(z)
                else:
                    net_forces = cal_interaction(z, lnw, self.interaction, t, m=interaction_m) * float(interaction_scale)
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
            "[split-sde from x0] start | "
            f"n_init={x0_t.shape[0]}, n_times={len(ts_points)}, "
            f"dt={dt}, sigma={sigma}, growth_alpha={growth_alpha}, "
            f"interaction_scale={float(interaction_scale)}, "
            f"t_range=({t_min},{t_max}), est_steps={est_steps}"
        )

    sde = SDE(f_net.v_net, f_net.g_net, score_net, f_net.interaction_net, sigma=sigma)
    ts_tensor = torch.tensor(list(ts_points), dtype=torch.float32, device=device)
    sde_points, _ = euler_sdeint_split(sde, initial_state, dt=dt, ts=ts_tensor, noise_std=0.0)
    out = [p.detach().cpu().numpy() for p in sde_points]

    if verbose:
        print(
            "[split-sde from x0] done | "
            f"timepoints={len(out)}, shape0={out[0].shape if out else None}"
        )

    return np.array(out, dtype=object)


def _build_classifier_feature_spec(
    *,
    dim: int,
    start_idx_1based: int,
    n_pcs: int,
) -> tuple[list[str], list[int]]:
    if int(start_idx_1based) < 1:
        raise ValueError("--classifier-feature-start must be >= 1")
    if int(start_idx_1based) > int(dim):
        raise ValueError(f"--classifier-feature-start={start_idx_1based} exceeds dim={dim}")
    if int(n_pcs) <= 0:
        raise ValueError("--classifier-n-pcs must be > 0")

    end_idx = int(start_idx_1based) + int(n_pcs) - 1
    if end_idx > int(dim):
        raise ValueError(
            f"Requested classifier feature range x{start_idx_1based}..x{end_idx} exceeds x1..x{dim}"
        )

    cols = [f"x{i}" for i in range(int(start_idx_1based), end_idx + 1)]
    # Convert x1..xN to 0-based indices over traj arrays.
    idxs = [i - 1 for i in range(int(start_idx_1based), end_idx + 1)]
    return cols, idxs


def _predict_labels_for_trajectories_with_indices(
    *,
    sde_points: np.ndarray,
    ts_points: Sequence[float],
    model,
    label_encoder,
    feature_indices: Sequence[int],
    device: str,
    knn_neighbors: int = 10,
) -> list[np.ndarray]:
    from sklearn.neighbors import KNeighborsClassifier

    feat_idx = np.asarray(list(feature_indices), dtype=np.int64)
    if feat_idx.size == 0:
        raise ValueError("feature_indices must be non-empty")

    model.eval()
    model.to(device)

    predicted_labels_list: list[np.ndarray] = []
    for i, t in enumerate(ts_points):
        traj_t = np.asarray(sde_points[i], dtype=np.float32)
        n_samples = int(traj_t.shape[0])
        if n_samples == 0:
            predicted_labels_list.append(np.asarray([], dtype=str))
            continue

        feats = traj_t[:, feat_idx]
        traj_feat_t = torch.tensor(feats, dtype=torch.float32)
        samples_t = torch.full((n_samples, 1), fill_value=float(t), dtype=torch.float32)
        input_t = torch.cat((samples_t, traj_feat_t), dim=1)

        with torch.no_grad():
            outputs = model(input_t.float().to(device))
            _, predicted = torch.max(outputs, 1)
            predicted_labels = label_encoder.inverse_transform(predicted.detach().cpu().numpy())

        # Keep spatial KNN refinement in physical x1/x2 space, independent of classifier feature slice.
        coords = traj_t[:, :2]
        k = min(int(knn_neighbors), int(coords.shape[0]))
        if k <= 1:
            refined_labels = np.asarray(predicted_labels).astype(str)
        else:
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(coords, predicted_labels)
            refined_labels = np.asarray(knn.predict(coords)).astype(str)

        predicted_labels_list.append(refined_labels)

    return predicted_labels_list


def _downsample_for_render(
    X: np.ndarray,
    labels: np.ndarray,
    max_n: Optional[int],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if max_n is None or max_n <= 0:
        return X, labels
    n = int(X.shape[0])
    if n <= int(max_n):
        return X, labels
    idx = rng.choice(n, size=int(max_n), replace=False)
    return X[idx], labels[idx]


def _compute_xy_limits(
    baseline_points: Sequence[np.ndarray],
    ablation_points: Sequence[np.ndarray],
    pad_frac: float = 0.03,
) -> tuple[tuple[float, float], tuple[float, float]]:
    x_min, x_max = np.inf, -np.inf
    y_min, y_max = np.inf, -np.inf

    for arr in list(baseline_points) + list(ablation_points):
        a = np.asarray(arr, dtype=np.float32)
        if a.size == 0:
            continue
        x = a[:, 0]
        y = a[:, 1]
        x_min = min(x_min, float(np.min(x)))
        x_max = max(x_max, float(np.max(x)))
        y_min = min(y_min, float(np.min(y)))
        y_max = max(y_max, float(np.max(y)))

    if not np.isfinite(x_min) or not np.isfinite(y_min):
        raise ValueError("Failed to compute axis limits from simulated points.")

    x_span = max(1e-8, x_max - x_min)
    y_span = max(1e-8, y_max - y_min)
    x_pad = x_span * float(pad_frac)
    y_pad = y_span * float(pad_frac)

    return (x_min - x_pad, x_max + x_pad), (y_min - y_pad, y_max + y_pad)


def _compute_morph_delta_metrics(
    *,
    baseline_xy: np.ndarray,
    ablation_xy: np.ndarray,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    grid_size: int = 64,
) -> Dict[str, float]:
    b = np.asarray(baseline_xy, dtype=np.float32)
    a = np.asarray(ablation_xy, dtype=np.float32)
    if b.ndim != 2 or a.ndim != 2 or b.shape[1] < 2 or a.shape[1] < 2:
        raise ValueError("baseline_xy and ablation_xy must be 2D arrays with >=2 columns")

    x_span = max(1e-8, float(xlim[1] - xlim[0]))
    y_span = max(1e-8, float(ylim[1] - ylim[0]))
    diag = float(np.hypot(x_span, y_span))
    eps = 1e-8

    c_b = np.mean(b[:, :2], axis=0)
    c_a = np.mean(a[:, :2], axis=0)
    centroid_shift_norm = float(np.linalg.norm(c_b - c_a) / max(diag, eps))

    spread_b = float(np.sqrt(np.mean(np.sum((b[:, :2] - c_b) ** 2, axis=1))))
    spread_a = float(np.sqrt(np.mean(np.sum((a[:, :2] - c_a) ** 2, axis=1))))
    spread_ratio_delta = float(abs(spread_a - spread_b) / max(spread_b, eps))

    g = int(max(8, grid_size))

    def _occupancy(arr_xy: np.ndarray) -> np.ndarray:
        xx = arr_xy[:, 0]
        yy = arr_xy[:, 1]
        xi = ((xx - float(xlim[0])) / x_span * g).astype(np.int64)
        yi = ((yy - float(ylim[0])) / y_span * g).astype(np.int64)
        xi = np.clip(xi, 0, g - 1)
        yi = np.clip(yi, 0, g - 1)
        occ = np.zeros((g, g), dtype=bool)
        occ[xi, yi] = True
        return occ

    occ_b = _occupancy(b[:, :2])
    occ_a = _occupancy(a[:, :2])
    inter = int(np.logical_and(occ_b, occ_a).sum())
    union = int(np.logical_or(occ_b, occ_a).sum())
    occupancy_iou = float(inter / union) if union > 0 else 1.0

    morph_delta_t = float(0.6 * centroid_shift_norm + 0.4 * (1.0 - occupancy_iou))
    return {
        "centroid_shift_norm": centroid_shift_norm,
        "spread_ratio_delta": spread_ratio_delta,
        "occupancy_iou": occupancy_iou,
        "morph_delta_t": morph_delta_t,
    }


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    s = str(hex_color).strip().lstrip("#")
    # Support both #RRGGBB and #RRGGBBAA formats (drop alpha if provided).
    if len(s) == 8:
        s = s[:6]
    if len(s) != 6:
        return (136, 136, 136)
    try:
        return tuple(int(s[i : i + 2], 16) for i in (0, 2, 4))
    except Exception:
        return (136, 136, 136)


def _rgb_to_hex(rgb: tuple[float, float, float]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*[max(0, min(255, int(v))) for v in rgb])


def _blend_hex(c1: str, c2: str, t: float) -> str:
    r1, g1, b1 = _hex_to_rgb(c1)
    r2, g2, b2 = _hex_to_rgb(c2)
    return _rgb_to_hex((r1 * (1 - t) + r2 * t, g1 * (1 - t) + g2 * t, b1 * (1 - t) + b2 * t))


def _desaturate_hex(color: str, amount: float = 0.55) -> str:
    a = float(np.clip(amount, 0.0, 1.0))
    return _blend_hex(color, "#c0c0c0", a)


def _edge_contrast_color(color: str, darken: float = 0.35) -> str:
    """Darken edge color for better contrast against dense same-color points."""
    rgb = np.asarray(_hex_to_rgb(color), dtype=np.float32) / 255.0
    d = float(np.clip(darken, 0.0, 0.85))
    rgb = np.clip(rgb * (1.0 - d), 0.0, 1.0)
    return _rgb_to_hex((rgb[0] * 255.0, rgb[1] * 255.0, rgb[2] * 255.0))


def _label_centroids(coords: np.ndarray, labels: np.ndarray) -> Dict[str, np.ndarray]:
    cent = {}
    labels_np = np.asarray(labels).astype(str)
    for lab in np.unique(labels_np):
        mask = labels_np == lab
        if int(np.sum(mask)) > 0:
            cent[str(lab)] = np.asarray(coords)[mask].mean(axis=0)
    return cent


def _focus_neighbor_centroids(
    coords: np.ndarray,
    labels: np.ndarray,
    focus_type: str,
    neighbor_pct: float = 0.1,
) -> Dict[str, np.ndarray]:
    labels_np = np.asarray(labels).astype(str)
    coords_np = np.asarray(coords)
    if focus_type not in set(labels_np.tolist()):
        return _label_centroids(coords_np, labels_np)

    focus_mask = labels_np == str(focus_type)
    focus_centroid = coords_np[focus_mask].mean(axis=0)
    cent = {}
    for lab in np.unique(labels_np):
        mask = labels_np == lab
        if int(np.sum(mask)) == 0:
            continue
        if str(lab) == str(focus_type):
            cent[str(lab)] = coords_np[mask].mean(axis=0)
            continue
        coords_l = coords_np[mask]
        d = np.linalg.norm(coords_l - focus_centroid, axis=1)
        pct = float(neighbor_pct)
        if pct > 1.0:
            pct = pct / 100.0
        pct = float(np.clip(pct, 0.0, 1.0))
        k = max(1, int(np.ceil(pct * len(coords_l))))
        idx = np.argsort(d)[:k]
        cent[str(lab)] = coords_l[idx].mean(axis=0)
    return cent


def _get_focus_edges(
    comm_result: Optional[dict],
    focus_type: str,
    mode: str = "both",
    top_k: int = 4,
    min_weight: float = 0.0,
    quantile: Optional[float] = None,
) -> list[tuple[float, str, str]]:
    if not comm_result:
        return []
    M = comm_result.get("M_per_source")
    types = comm_result.get("types")
    if M is None or types is None:
        return []
    types_list = [str(x) for x in list(types)]
    if str(focus_type) not in types_list:
        return []

    idx = types_list.index(str(focus_type))
    flows: list[tuple[float, str, str]] = []
    for j, tname in enumerate(types_list):
        if j == idx:
            continue
        if mode in ("outgoing", "both"):
            flows.append((float(M[idx, j]), str(focus_type), str(tname)))
        if mode in ("incoming", "both"):
            flows.append((float(M[j, idx]), str(tname), str(focus_type)))
    flows = [(w, s, d) for (w, s, d) in flows if float(w) > float(min_weight)]
    if not flows:
        return []
    if quantile is not None:
        weights = np.array([f[0] for f in flows], dtype=float)
        q = float(np.quantile(weights, float(quantile)))
        flows = [f for f in flows if float(f[0]) >= q]
    flows.sort(key=lambda x: float(x[0]), reverse=True)
    return flows[: max(1, int(top_k))]


def _draw_edge(ax, p1, p2, color: str, lw: float, alpha: float, rad: float = 0.15) -> None:
    from matplotlib import patches
    import matplotlib.patheffects as pe

    edge_color = _edge_contrast_color(color)

    arrow = patches.FancyArrowPatch(
        p1,
        p2,
        arrowstyle="-|>",
        mutation_scale=10 + float(lw) * 1.5,
        lw=float(lw),
        color=str(edge_color),
        alpha=float(alpha),
        connectionstyle=f"arc3,rad={float(rad)}",
        shrinkA=4,
        shrinkB=4,
    )
    # White halo keeps arrows legible even when point colors are similar.
    arrow.set_path_effects(
        [
            pe.Stroke(linewidth=max(1.0, float(lw) + 1.5), foreground="white", alpha=min(1.0, 0.95 * float(alpha))),
            pe.Normal(),
        ]
    )
    ax.add_patch(arrow)


def _render_frame_side_by_side_3d(
    *,
    baseline_xy: np.ndarray,
    baseline_labels: np.ndarray,
    ablation_xy: np.ndarray,
    ablation_labels: np.ndarray,
    time_value: float,
    label_to_color: Dict[str, str],
    out_png: str,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    zlim: tuple[float, float],
    elev: float,
    azim: float,
    point_size: float,
    alpha: float,
    dpi: int,
    baseline_title: str,
    ablation_title: str,
) -> None:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12.5, 6.0), dpi=int(dpi))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")

    def _draw(ax, xy: np.ndarray, labels: np.ndarray, title: str) -> None:
        labels = np.asarray(labels).astype(str)
        colors = [label_to_color.get(str(l), "#888888") for l in labels]
        z = np.full(xy.shape[0], fill_value=float(time_value), dtype=np.float32)

        ax.scatter(
            xy[:, 0],
            xy[:, 1],
            z,
            c=colors,
            s=float(point_size),
            linewidths=0.0,
            alpha=float(alpha),
            depthshade=False,
        )
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_zlim(*zlim)
        ax.set_title(f"{title}\nt={time_value:.1f}", fontsize=10)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_zlabel("time")
        ax.view_init(elev=float(elev), azim=float(azim))
        try:
            ax.set_proj_type("ortho")
        except Exception:
            pass

    _draw(ax1, baseline_xy, baseline_labels, baseline_title)
    _draw(ax2, ablation_xy, ablation_labels, ablation_title)
    fig.suptitle("MOSTA Virtual Tissue Ablation: Baseline vs Counterfactual", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def _render_frame_side_by_side_2d(
    *,
    baseline_xy: np.ndarray,
    baseline_labels: np.ndarray,
    ablation_xy: np.ndarray,
    ablation_labels: np.ndarray,
    time_value: float,
    label_to_color: Dict[str, str],
    out_png: str,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    point_size: float,
    alpha: float,
    dpi: int,
    panel_size: float,
    baseline_title: str,
    ablation_title: str,
) -> None:
    import matplotlib.pyplot as plt

    panel_size = float(max(2.0, panel_size))
    fig, axes = plt.subplots(1, 2, figsize=(panel_size * 2.0, panel_size), dpi=int(dpi))
    fig.patch.set_facecolor("white")

    def _draw(ax, xy: np.ndarray, labels: np.ndarray, title: str) -> None:
        labels = np.asarray(labels).astype(str)
        colors = [label_to_color.get(str(l), "#888888") for l in labels]
        rasterized = xy.shape[0] > 30000

        ax.set_facecolor("white")
        ax.scatter(
            xy[:, 0],
            xy[:, 1],
            s=float(point_size),
            c=colors,
            linewidths=0,
            alpha=float(alpha),
            rasterized=rasterized,
        )
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(f"{title}\nt={time_value:.1f}", fontsize=10)

    _draw(axes[0], baseline_xy, baseline_labels, baseline_title)
    _draw(axes[1], ablation_xy, ablation_labels, ablation_title)
    fig.suptitle("MOSTA Virtual Tissue Ablation: Baseline vs Counterfactual", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def _render_frame_single_3d(
    *,
    baseline_xy: np.ndarray,
    baseline_labels: np.ndarray,
    time_value: float,
    label_to_color: Dict[str, str],
    out_png: str,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    zlim: tuple[float, float],
    elev: float,
    azim: float,
    point_size: float,
    alpha: float,
    dpi: int,
    baseline_title: str,
    show_titles: bool = True,
) -> None:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(6.2, 5.8), dpi=int(dpi))
    ax = fig.add_subplot(1, 1, 1, projection="3d")

    labels = np.asarray(baseline_labels).astype(str)
    colors = [label_to_color.get(str(l), "#888888") for l in labels]
    z = np.full(baseline_xy.shape[0], fill_value=float(time_value), dtype=np.float32)
    ax.scatter(
        baseline_xy[:, 0],
        baseline_xy[:, 1],
        z,
        c=colors,
        s=float(point_size),
        linewidths=0.0,
        alpha=float(alpha),
        depthshade=False,
    )
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)
    if bool(show_titles):
        ax.set_title(f"{baseline_title}\nt={time_value:.1f}", fontsize=float(NATURE_SCATTER_DEFAULTS["title_fontsize"]))
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("time")
    ax.view_init(elev=float(elev), azim=float(azim))
    try:
        ax.set_proj_type("ortho")
    except Exception:
        pass

    if bool(show_titles):
        fig.suptitle("MOSTA Baseline Trajectory", fontsize=float(NATURE_SCATTER_DEFAULTS["suptitle_fontsize"]))
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def _render_frame_single_2d(
    *,
    baseline_xy: np.ndarray,
    baseline_labels: np.ndarray,
    time_value: float,
    label_to_color: Dict[str, str],
    out_png: str,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    point_size: float,
    alpha: float,
    dpi: int,
    panel_size: float,
    baseline_title: str,
    comm_result: Optional[dict] = None,
    draw_focus_interactions: bool = False,
    focus_celltype: str = "Brain",
    focus_mode: str = "both",
    focus_top_k: int = 2,
    focus_min_weight: float = 0.0,
    focus_edge_quantile: Optional[float] = None,
    focus_edge_width: tuple[float, float] = (1.0, 2.6),
    focus_edge_alpha: tuple[float, float] = (0.25, 0.85),
    focus_edge_curve: float = 0.18,
    focus_edge_curve_bi: float = 0.35,
    focus_neighbor_pct: float = 0.1,
    focus_other_desaturate: float = 0.55,
    focus_show_endpoints: bool = True,
    focus_endpoint_size: float = 26.0,
    focus_endpoint_alpha: float = 0.95,
    focus_endpoint_edgecolor: str = "#ffffff",
    focus_endpoint_linewidth: float = 0.8,
    show_titles: bool = True,
) -> None:
    import matplotlib.pyplot as plt

    panel_size = float(max(2.0, panel_size))
    fig, ax = plt.subplots(1, 1, figsize=(panel_size, panel_size), dpi=int(dpi))
    fig.patch.set_facecolor("white")

    labels = np.asarray(baseline_labels).astype(str)
    base_colors = [label_to_color.get(str(l), "#888888") for l in labels]
    if bool(draw_focus_interactions):
        colors = [
            (label_to_color.get(str(l), "#888888") if str(l) == str(focus_celltype) else _desaturate_hex(label_to_color.get(str(l), "#888888"), focus_other_desaturate))
            for l in labels
        ]
    else:
        colors = base_colors
    rasterized = baseline_xy.shape[0] > 30000

    ax.set_facecolor("white")
    ax.scatter(
        baseline_xy[:, 0],
        baseline_xy[:, 1],
        s=float(point_size),
        c=colors,
        linewidths=0,
        alpha=float(alpha),
        rasterized=rasterized,
    )
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if bool(show_titles):
        ax.set_title(f"{baseline_title}\nt={time_value:.1f}", fontsize=float(NATURE_SCATTER_DEFAULTS["title_fontsize"]))

    if bool(draw_focus_interactions):
        centroids = _focus_neighbor_centroids(
            baseline_xy,
            labels,
            focus_type=str(focus_celltype),
            neighbor_pct=float(focus_neighbor_pct),
        )
        flows = _get_focus_edges(
            comm_result=comm_result,
            focus_type=str(focus_celltype),
            mode=str(focus_mode),
            top_k=int(focus_top_k),
            min_weight=float(focus_min_weight),
            quantile=focus_edge_quantile,
        )
        if flows:
            directed_pairs = {(s, d) for _, s, d in flows}
            bidirectional_pairs = {tuple(sorted((s, d))) for s, d in directed_pairs if (d, s) in directed_pairs}
            max_w = max([float(w) for w, _, _ in flows])
            highlight_nodes: list[str] = []
            for w, src, dst in flows:
                if src not in centroids or dst not in centroids:
                    continue
                t = float(w) / max_w if max_w > 0 else 0.0
                lw = float(focus_edge_width[0]) + t * (float(focus_edge_width[1]) - float(focus_edge_width[0]))
                a = float(focus_edge_alpha[0]) + t * (float(focus_edge_alpha[1]) - float(focus_edge_alpha[0]))
                edge_color = label_to_color.get(str(dst), label_to_color.get(str(src), "#111111"))
                pair = tuple(sorted((src, dst)))
                if pair in bidirectional_pairs:
                    rad = float(focus_edge_curve_bi)
                else:
                    rad = float(focus_edge_curve) if str(src) < str(dst) else -float(focus_edge_curve)
                _draw_edge(ax, centroids[src], centroids[dst], edge_color, lw=lw, alpha=a, rad=rad)
                highlight_nodes.extend([str(src), str(dst)])

            if bool(focus_show_endpoints):
                unique_nodes = [x for x in dict.fromkeys(highlight_nodes) if x in centroids]
                if unique_nodes:
                    node_xy = np.asarray([centroids[x] for x in unique_nodes], dtype=np.float32)
                    node_colors = [label_to_color.get(x, "#111111") for x in unique_nodes]
                    ax.scatter(
                        node_xy[:, 0],
                        node_xy[:, 1],
                        s=float(max(1.0, focus_endpoint_size)),
                        c=node_colors,
                        linewidths=float(max(0.0, focus_endpoint_linewidth)),
                        edgecolors=str(focus_endpoint_edgecolor),
                        alpha=float(np.clip(focus_endpoint_alpha, 0.0, 1.0)),
                        zorder=6,
                    )

        ax.text(
            0.02,
            0.98,
            f"Focus: {focus_celltype} ({focus_mode})",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=float(NATURE_SCATTER_DEFAULTS["legend_fontsize"]),
            color="#111111",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.8),
        )

    if bool(show_titles):
        fig.suptitle("MOSTA Baseline Trajectory", fontsize=float(NATURE_SCATTER_DEFAULTS["suptitle_fontsize"]))
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def _ensure_color_map_has_labels(
    label_to_color: Dict[str, str],
    labels: Sequence[str],
) -> Dict[str, str]:
    labels = [str(x) for x in labels]
    missing = [x for x in labels if x not in label_to_color]
    if len(missing) == 0:
        return label_to_color

    extra = _default_color_map(missing)
    merged = dict(label_to_color)
    merged.update(extra)
    return merged


def _apply_observed_anchor_if_needed(
    *,
    branch_name: str,
    ts_points: Sequence[float],
    sde_points: np.ndarray,
    pred_labels: list[np.ndarray],
    df: pd.DataFrame,
    annotation_key: str,
    feature_cols: Sequence[str],
    observed_times: Sequence[float],
    start_time: float,
    target_labels: Sequence[str],
    remove_frac: float,
    n_samples_branch: int,
    rng: np.random.Generator,
) -> None:
    """Optional non-strict mode: re-anchor branch states at observed times."""
    for t in observed_times:
        if _float_close(float(t), float(start_time)):
            continue
        # only apply for simulated times that are present in ts_points
        idx = None
        for i, tt in enumerate(ts_points):
            if _float_close(float(tt), float(t)):
                idx = i
                break
        if idx is None:
            continue

        df_t = df[df["samples"] == float(t)]
        X_pool = df_t[list(feature_cols)].values.astype(np.float32)
        y_pool = df_t[annotation_key].astype(str).values

        if branch_name == "ablation":
            X_pool, y_pool, _, _, _, _ = _apply_label_ablation(
                X_pool=X_pool,
                labels_pool=y_pool,
                target_labels=target_labels,
                remove_frac=remove_frac,
                rng=rng,
            )

        if X_pool.shape[0] < n_samples_branch:
            raise ValueError(
                f"Observed anchor at t={t} for branch '{branch_name}' has only {X_pool.shape[0]} rows, "
                f"but branch requires {n_samples_branch}."
            )

        X_anchor, y_anchor = _sample_rows_no_replace(X_pool, y_pool, n=n_samples_branch, rng=rng)
        sde_points[idx] = X_anchor
        pred_labels[idx] = np.asarray(y_anchor).astype(str)


def _run_baseline_only_pipeline(
    *,
    args,
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    observed_times: Sequence[float],
    t_start_obs: float,
    ts_points: Sequence[float],
    f_net,
    score_net,
    model,
    label_encoder,
    clf_feature_indices: Sequence[int],
    device: str,
    exp_dir: str,
    rng: np.random.Generator,
) -> None:
    frames_dir = os.path.join(args.output_dir, "frames")
    baseline_dir = os.path.join(args.output_dir, "branch_baseline")
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(baseline_dir, exist_ok=True)

    df_t0 = df[df["samples"] == float(t_start_obs)]
    X_base_pool = df_t0[list(feature_cols)].values.astype(np.float32)
    y_base_pool = df_t0[args.annotation_key].astype(str).values

    if args.sde_n_samples <= 0:
        raise ValueError("--sde-n-samples must be > 0")
    n_init_baseline = int(min(args.sde_n_samples, X_base_pool.shape[0]))
    if n_init_baseline <= 0:
        raise ValueError(f"Invalid initial sample size baseline={n_init_baseline}. Pool size={X_base_pool.shape[0]}")

    X0_baseline, y0_baseline = _sample_rows_no_replace(X_base_pool, y_base_pool, n_init_baseline, rng)
    np.savez_compressed(
        os.path.join(baseline_dir, "init_pool.npz"),
        X0=X0_baseline,
        labels=np.asarray(y0_baseline).astype(str),
    )

    baseline_interaction_scale = 0.0 if bool(args.baseline_zero_interaction) else 1.0
    t_sim0 = time.perf_counter()
    if bool(args.spatial_warp_to_observed):
        if int(args.spatial_warp_k) <= 0:
            raise ValueError("--spatial-warp-k must be > 0")
        if float(args.spatial_warp_eps) <= 0:
            raise ValueError("--spatial-warp-eps must be > 0")
        rng_warp = np.random.default_rng(1 if args.random_seed is None else int(args.random_seed) + 1)
        sde_baseline = _simulate_piecewise_spatially_warped_baseline(
            x0=X0_baseline,
            df=df,
            feature_cols=feature_cols,
            annotation_key=args.annotation_key,
            observed_times=observed_times,
            ts_points=ts_points,
            f_net=f_net,
            score_net=score_net,
            dt=float(args.split_sde_dt),
            sigma=float(args.split_sigma),
            growth_alpha=float(args.split_growth_alpha),
            interaction_m=int(args.interaction_m),
            interaction_scale=float(baseline_interaction_scale),
            device=device,
            rng=rng_warp,
            warp_k=int(args.spatial_warp_k),
            warp_eps=float(args.spatial_warp_eps),
        )
    else:
        sde_baseline = _simulate_sde_points_split_from_x0(
            x0=X0_baseline,
            f_net=f_net,
            score_net=score_net,
            ts_points=ts_points,
            dt=float(args.split_sde_dt),
            sigma=float(args.split_sigma),
            growth_alpha=float(args.split_growth_alpha),
            interaction_m=int(args.interaction_m),
            interaction_scale=float(baseline_interaction_scale),
            device=device,
            verbose=True,
        )
    print(f"Baseline simulation finished in {time.perf_counter() - t_sim0:.1f}s")

    pred_baseline = _predict_labels_for_trajectories_with_indices(
        sde_points=sde_baseline,
        ts_points=ts_points,
        model=model,
        label_encoder=label_encoder,
        feature_indices=clf_feature_indices,
        device=device,
    )

    if not args.strict_counterfactual:
        print("[warn] strict-counterfactual disabled: re-anchoring baseline states at observed times")
        _apply_observed_anchor_if_needed(
            branch_name="baseline",
            ts_points=ts_points,
            sde_points=sde_baseline,
            pred_labels=pred_baseline,
            df=df,
            annotation_key=args.annotation_key,
            feature_cols=feature_cols,
            observed_times=observed_times,
            start_time=t_start_obs,
            target_labels=[],
            remove_frac=0.0,
            n_samples_branch=n_init_baseline,
            rng=rng,
        )

    np.savez_compressed(
        os.path.join(baseline_dir, "trajectory_pred.npz"),
        ts=np.asarray(ts_points, dtype=np.float32),
        labels=np.asarray(pred_baseline, dtype=object),
    )

    observed_label_values = df[args.annotation_key].astype(str).values
    label_to_color = load_label_to_color(
        labels=observed_label_values,
        label_color_json=args.label_color_json,
        color_h5ad=args.color_h5ad,
        annotation_key=args.annotation_key,
    )
    all_pred_labels: list[str] = []
    for arr in pred_baseline:
        all_pred_labels.extend(np.asarray(arr).astype(str).tolist())
    label_to_color = _ensure_color_map_has_labels(label_to_color, all_pred_labels)
    with open(os.path.join(args.output_dir, "label_to_color.json"), "w", encoding="utf-8") as f:
        json.dump(label_to_color, f, indent=2, ensure_ascii=False)

    all_time_communications: Dict[str, dict] = {}
    comm_path: Optional[str] = None
    comm_sampling_csv_path: Optional[str] = None
    comm_source_by_time: Dict[str, str] = {}
    comm_cells_by_time: Dict[str, int] = {}
    if bool(args.draw_focus_interactions):
        attn_dir = os.path.join(args.output_dir, "attention")
        os.makedirs(attn_dir, exist_ok=True)
        try:
            import anndata as ad
        except Exception as exc:
            raise RuntimeError(
                "draw-focus-interactions requires anndata. Install anndata or disable --draw-focus-interactions."
            ) from exc

        max_cells = int(args.attention_max_cells)
        if max_cells <= 0:
            raise ValueError("--attention-max-cells must be > 0")

        for idx, t in enumerate(ts_points):
            if int(args.attention_stride) > 1 and (idx % int(args.attention_stride) != 0):
                continue
            key = str(float(t))
            source = "simulated"
            t_obs_match = None
            if bool(args.attention_use_real_observed):
                for t_obs in observed_times:
                    if _float_close(float(t), float(t_obs)):
                        t_obs_match = float(t_obs)
                        break

            if t_obs_match is not None:
                df_t_obs = df[df["samples"] == float(t_obs_match)]
                X_t = df_t_obs[list(feature_cols)].values.astype(np.float32)
                labels_t = df_t_obs[args.annotation_key].astype(str).values
                source = "observed"
            else:
                X_t = np.asarray(sde_baseline[idx], dtype=np.float32)
                labels_t = np.asarray(pred_baseline[idx]).astype(str)

            n_before = int(X_t.shape[0])
            if n_before > max_cells:
                take = rng.choice(n_before, size=max_cells, replace=False)
                X_t = X_t[take]
                labels_t = labels_t[take]
            n_after = int(X_t.shape[0])

            comm_source_by_time[key] = str(source)
            comm_cells_by_time[key] = int(n_after)

            adata_t = ad.AnnData(X=X_t)
            adata_t.obs["Annotation"] = labels_t
            adata_t.obsm["spatial"] = X_t[:, :2]

            print(f"[focus-comm] time={key} source={source} cells={n_after}/{n_before}")
            attn_out = save_interpolated_attention(
                adata_t,
                time_value=float(t),
                f_net=f_net,
                device=device,
                out_dir=attn_dir,
            )
            comm = analyze_attention_by_celltype(
                edge_index=attn_out["edge_index"],
                attn=attn_out["attn_mean"],
                labels=adata_t.obs["Annotation"].values,
                spatial_coord=adata_t.obsm["spatial"],
                time_title=key,
                remove_self_loop=False,
                winsor_quantile=0.995,
                distance_bins=None,
                n_permutations=0,
                plot=False,
            )
            all_time_communications[key] = comm

        if all_time_communications:
            computed_keys = sorted(all_time_communications.keys(), key=lambda x: float(x))

            def _nearest_key(target: str) -> str:
                return min(computed_keys, key=lambda k: abs(float(k) - float(target)))

            for t in ts_points:
                k = str(float(t))
                if k not in all_time_communications:
                    nearest = _nearest_key(k)
                    all_time_communications[k] = all_time_communications[nearest]
                    comm_source_by_time[k] = f"nearest:{comm_source_by_time.get(nearest, 'unknown')}"
                    comm_cells_by_time[k] = int(comm_cells_by_time.get(nearest, 0))

            comm_path = os.path.join(args.output_dir, "all_time_communications.pkl")
            with open(comm_path, "wb") as f:
                pickle.dump(all_time_communications, f)
            print("Saved focus communication cache:", comm_path)

            comm_rows = []
            for k in sorted(all_time_communications.keys(), key=lambda x: float(x)):
                comm_rows.append(
                    {
                        "time": float(k),
                        "source": str(comm_source_by_time.get(k, "unknown")),
                        "n_cells_used": int(comm_cells_by_time.get(k, 0)),
                    }
                )
            comm_sampling_csv_path = os.path.join(args.output_dir, "communication_sampling_by_time.csv")
            pd.DataFrame(comm_rows).to_csv(comm_sampling_csv_path, index=False)
            print("Saved communication sampling stats:", comm_sampling_csv_path)
        else:
            print("[focus-comm] no attention computed; disable overlay or check --attention-stride")

    xlim_global, ylim_global = _compute_xy_limits(
        sde_baseline,
        sde_baseline,
        pad_frac=float(args.axis_pad_frac),
    )
    zlim = (float(args.time_start), float(args.time_end)) if args.video_style == "fixed_3d" else None
    frame_paths: list[str] = []
    render_rng = np.random.default_rng((0 if args.random_seed is None else int(args.random_seed)) + 1024)

    baseline_counts_rows: list[dict] = []
    t_render0 = time.perf_counter()
    warned_3d_overlay = False
    for i, t in enumerate(ts_points):
        Xb = np.asarray(sde_baseline[i], dtype=np.float32)
        yb = np.asarray(pred_baseline[i]).astype(str)

        if args.axis_limit_mode == "per_timepoint":
            xlim_frame, ylim_frame = _compute_xy_limits([Xb[:, :2]], [Xb[:, :2]], pad_frac=float(args.axis_pad_frac))
        else:
            xlim_frame, ylim_frame = xlim_global, ylim_global

        Xb_vis, yb_vis = _downsample_for_render(Xb, yb, args.video_point_subsample, render_rng)
        frame_path = os.path.join(frames_dir, f"frame_{i:03d}.png")
        if args.video_style == "fixed_2d":
            comm_t = all_time_communications.get(str(float(t))) if bool(args.draw_focus_interactions) else None
            _render_frame_single_2d(
                baseline_xy=Xb_vis[:, :2],
                baseline_labels=yb_vis,
                time_value=float(t),
                label_to_color=label_to_color,
                out_png=frame_path,
                xlim=xlim_frame,
                ylim=ylim_frame,
                point_size=float(args.point_size),
                alpha=float(args.point_alpha),
                dpi=int(args.frame_dpi),
                panel_size=float(args.panel_size),
                baseline_title="Baseline",
                comm_result=comm_t,
                draw_focus_interactions=bool(args.draw_focus_interactions),
                focus_celltype=str(args.focus_celltype),
                focus_mode=str(args.focus_mode),
                focus_top_k=int(args.focus_top_k),
                focus_min_weight=float(args.focus_min_weight),
                focus_edge_quantile=args.focus_edge_quantile,
                focus_edge_width=(float(args.focus_edge_width_min), float(args.focus_edge_width_max)),
                focus_edge_alpha=(float(args.focus_edge_alpha_min), float(args.focus_edge_alpha_max)),
                focus_edge_curve=float(args.focus_edge_curve),
                focus_edge_curve_bi=float(args.focus_edge_curve_bi),
                focus_neighbor_pct=float(args.focus_neighbor_pct),
                focus_other_desaturate=float(args.focus_other_desaturate),
                focus_show_endpoints=bool(args.focus_show_endpoints),
                focus_endpoint_size=float(args.focus_endpoint_size),
                focus_endpoint_alpha=float(args.focus_endpoint_alpha),
                focus_endpoint_edgecolor=str(args.focus_endpoint_edgecolor),
                focus_endpoint_linewidth=float(args.focus_endpoint_linewidth),
                show_titles=bool(args.show_titles),
            )
        else:
            if bool(args.draw_focus_interactions) and not warned_3d_overlay:
                print("[warn] Focus communication overlay is implemented for --video-style fixed_2d only.")
                warned_3d_overlay = True
            assert zlim is not None
            _render_frame_single_3d(
                baseline_xy=Xb_vis[:, :2],
                baseline_labels=yb_vis,
                time_value=float(t),
                label_to_color=label_to_color,
                out_png=frame_path,
                xlim=xlim_frame,
                ylim=ylim_frame,
                zlim=zlim,
                elev=float(args.camera_elev),
                azim=float(args.camera_azim),
                point_size=float(args.point_size),
                alpha=float(args.point_alpha),
                dpi=int(args.frame_dpi),
                baseline_title="Baseline",
                show_titles=bool(args.show_titles),
            )
        frame_paths.append(frame_path)
        baseline_counts_rows.append({"time": float(t), "n_baseline": int(Xb.shape[0])})
        print(f"Rendered baseline frame {i + 1}/{len(ts_points)} | t={t:.3f}")

    print(f"Baseline frame rendering finished in {time.perf_counter() - t_render0:.1f}s")

    counts_csv_path = os.path.join(args.output_dir, "baseline_counts_by_time.csv")
    pd.DataFrame(baseline_counts_rows).sort_values("time").to_csv(counts_csv_path, index=False)

    gif_name = f"baseline_{_tag_float(args.time_start)}_to_{_tag_float(args.time_end)}_step{_tag_float(args.time_step)}.gif"
    gif_path = os.path.join(args.output_dir, gif_name)
    duration = 1.0 / max(1, int(args.gif_fps))
    images = [imageio.imread(p) for p in frame_paths]
    imageio.mimsave(gif_path, images, duration=duration, loop=0)
    print("Saved GIF:", gif_path)

    summary = {
        "status": "ok",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "render_mode": "baseline_only",
        "data_csv": args.data_csv,
        "annotation_key": args.annotation_key,
        "time_start": float(args.time_start),
        "time_end": float(args.time_end),
        "time_step": float(args.time_step),
        "timepoints": [float(x) for x in ts_points],
        "n_frames": int(len(ts_points)),
        "gif_path": gif_path,
        "frames_dir": frames_dir,
        "frame_count": int(len(frame_paths)),
        "baseline_counts_by_time_csv": counts_csv_path,
        "observed_times": [float(x) for x in observed_times],
        "n_observed_rows": int(df.shape[0]),
        "n_t0_pool_baseline": int(X_base_pool.shape[0]),
        "n_init_baseline": int(n_init_baseline),
        "classifier": {
            "n_pcs": int(len(clf_feature_indices)),
            "feature_start_x": int(args.classifier_feature_start),
            "feature_end_x": int(args.classifier_feature_start + len(clf_feature_indices) - 1),
            "epochs": int(args.classifier_epochs),
            "hidden": int(args.classifier_hidden),
            "best_metric": str(args.classifier_best_metric),
            "train_on_full_data": bool(args.classifier_train_on_full_data),
        },
        "simulation": {
            "split_sde_dt": float(args.split_sde_dt),
            "split_sigma": float(args.split_sigma),
            "split_growth_alpha": float(args.split_growth_alpha),
            "spatial_warp_to_observed": bool(args.spatial_warp_to_observed),
            "spatial_warp_k": int(args.spatial_warp_k),
            "spatial_warp_eps": float(args.spatial_warp_eps),
            "spatial_warp_mode": "piecewise_rerun" if bool(args.spatial_warp_to_observed) else "off",
            "interaction_m": int(args.interaction_m),
            "baseline_zero_interaction": bool(args.baseline_zero_interaction),
            "baseline_interaction_scale": float(baseline_interaction_scale),
            "sde_n_samples_requested": int(args.sde_n_samples),
            "model_exp_dir": exp_dir,
            "device": device,
        },
        "camera": {
            "style": args.video_style,
            "layout": "single_panel",
            "elev": float(args.camera_elev),
            "azim": float(args.camera_azim),
            "axis_limit_mode": str(args.axis_limit_mode),
            "axis_pad_frac": float(args.axis_pad_frac),
            "xlim_global": [float(xlim_global[0]), float(xlim_global[1])],
            "ylim_global": [float(ylim_global[0]), float(ylim_global[1])],
            "zlim": [float(zlim[0]), float(zlim[1])] if zlim is not None else None,
            "point_size": float(args.point_size),
            "point_alpha": float(args.point_alpha),
            "panel_size": float(args.panel_size),
            "video_point_subsample": int(args.video_point_subsample),
            "gif_fps": int(args.gif_fps),
            "frame_dpi": int(args.frame_dpi),
            "show_titles": bool(args.show_titles),
        },
        "communication": {
            "draw_focus_interactions": bool(args.draw_focus_interactions),
            "focus_celltype": str(args.focus_celltype),
            "focus_mode": str(args.focus_mode),
            "focus_top_k": int(args.focus_top_k),
            "focus_min_weight": float(args.focus_min_weight),
            "focus_edge_quantile": None if args.focus_edge_quantile is None else float(args.focus_edge_quantile),
            "focus_edge_width": [float(args.focus_edge_width_min), float(args.focus_edge_width_max)],
            "focus_edge_alpha": [float(args.focus_edge_alpha_min), float(args.focus_edge_alpha_max)],
            "focus_edge_curve": float(args.focus_edge_curve),
            "focus_edge_curve_bi": float(args.focus_edge_curve_bi),
            "focus_neighbor_pct": float(args.focus_neighbor_pct),
            "focus_other_desaturate": float(args.focus_other_desaturate),
            "focus_show_endpoints": bool(args.focus_show_endpoints),
            "focus_endpoint_size": float(args.focus_endpoint_size),
            "focus_endpoint_alpha": float(args.focus_endpoint_alpha),
            "focus_endpoint_edgecolor": str(args.focus_endpoint_edgecolor),
            "focus_endpoint_linewidth": float(args.focus_endpoint_linewidth),
            "attention_stride": int(args.attention_stride),
            "attention_max_cells": int(args.attention_max_cells),
            "attention_use_real_observed": bool(args.attention_use_real_observed),
            "all_time_communications_pkl": comm_path,
            "communication_sampling_by_time_csv": comm_sampling_csv_path,
        },
    }

    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("Saved summary:", summary_path)
    print("Done.")


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="MOSTA virtual tissue ablation video (baseline vs counterfactual)")

    # Core paths
    parser.add_argument("--config", default="config/mosta_config.yaml")
    parser.add_argument(
        "--data-csv",
        default="evaluation/mosta/data/mosta_four_time_with_celltype_refined.csv",
    )
    parser.add_argument("--annotation-key", default="Annotation")
    parser.add_argument("--output-dir", default="results/mosta_virtual_ablation_cartilage_full")

    # Ablation controls
    parser.add_argument("--target-label", default="Cartilage primordium")
    parser.add_argument(
        "--target-labels",
        default="",
        help="Optional comma-separated multi-label ablation targets. Overrides --target-label when provided.",
    )
    parser.add_argument("--ablation-start-time", type=float, default=0.0)
    parser.add_argument("--ablation-remove-frac", type=float, default=1.0)
    parser.add_argument(
        "--mass-control",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep baseline/ablation initial sample counts identical.",
    )
    parser.add_argument(
        "--strict-counterfactual",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, never re-anchor to observed times after t0.",
    )

    # Time controls
    parser.add_argument("--time-start", type=float, default=0.0)
    parser.add_argument("--time-end", type=float, default=3.0)
    parser.add_argument("--time-step", type=float, default=0.05)

    # Sampling / simulation controls
    parser.add_argument("--sde-n-samples", type=int, default=50000)
    parser.add_argument("--split-sde-dt", type=float, default=0.05)
    parser.add_argument("--split-sigma", type=float, default=0.03)
    parser.add_argument("--split-growth-alpha", type=float, default=1.0)
    parser.add_argument(
        "--spatial-warp-to-observed",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "After baseline simulation, warp only spatial dims x1,x2 toward observed endpoint shapes "
            "segment-by-segment. Gene dims are unchanged."
        ),
    )
    parser.add_argument(
        "--spatial-warp-k",
        type=int,
        default=8,
        help="Number of simulated endpoint anchors used to interpolate spatial warp displacements.",
    )
    parser.add_argument(
        "--spatial-warp-eps",
        type=float,
        default=1e-6,
        help="Small positive constant for inverse-distance spatial warp weights.",
    )
    parser.add_argument("--interaction-m", type=int, default=1024)
    parser.add_argument(
        "--baseline-zero-interaction",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If true, set interaction term to zero only for baseline branch during SDE inference.",
    )

    # Classifier controls (shared across branches)
    parser.add_argument("--classifier-epochs", type=int, default=500)
    parser.add_argument("--classifier-hidden", type=int, default=128)
    parser.add_argument("--classifier-n-pcs", type=int, default=12)
    parser.add_argument(
        "--classifier-feature-start",
        type=int,
        default=1,
        help="1-based start feature index for classifier over x1..x52 (e.g. 3 means start from x3).",
    )
    parser.add_argument("--classifier-best-metric", choices=["accuracy", "bacc"], default="bacc")
    parser.add_argument(
        "--classifier-train-on-full-data",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--classifier-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse/save classifier weights in cache dir.",
    )
    parser.add_argument(
        "--classifier-cache-dir",
        default=DEFAULT_SHARED_CLASSIFIER_CACHE_DIR,
        help=(
            "Classifier cache directory. Default points to the shared MOSTA interp cache: "
            "results/mosta_interp_0_3_0208_n_pc_12/classifier_cache"
        ),
    )
    parser.add_argument(
        "--classifier-cache-path",
        default=None,
        help="Optional explicit classifier cache .pt path; takes precedence over --classifier-cache-dir.",
    )
    parser.add_argument(
        "--classifier-cache-tag",
        default=None,
        help="Optional cache tag mixed into key. Keep unset to match shared cache naming.",
    )

    # Video controls
    parser.add_argument(
        "--render-mode",
        choices=["baseline_only", "baseline_vs_ablation"],
        default="baseline_only",
        help="Render only baseline branch or legacy baseline-vs-ablation side-by-side panels.",
    )
    parser.add_argument("--video-layout", choices=["side_by_side"], default="side_by_side")
    parser.add_argument("--video-style", choices=["fixed_2d", "fixed_3d"], default="fixed_2d")
    parser.add_argument("--video-point-subsample", type=int, default=50000)
    parser.add_argument("--gif-fps", type=int, default=4)
    parser.add_argument("--frame-dpi", type=int, default=int(NATURE_SCATTER_DEFAULTS["frame_dpi"]))
    parser.add_argument("--point-size", type=float, default=float(NATURE_SCATTER_DEFAULTS["point_size"]))
    parser.add_argument("--point-alpha", type=float, default=float(NATURE_SCATTER_DEFAULTS["point_alpha"]))
    parser.add_argument(
        "--show-titles",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Show panel title and figure title on rendered frames.",
    )
    parser.add_argument(
        "--draw-focus-interactions",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Overlay focus cell-type communication arrows on baseline 2D frames.",
    )
    parser.add_argument("--focus-celltype", default="Brain")
    parser.add_argument("--focus-mode", choices=["incoming", "outgoing", "both"], default="both")
    parser.add_argument("--focus-top-k", type=int, default=4)
    parser.add_argument("--focus-min-weight", type=float, default=0.0)
    parser.add_argument(
        "--focus-edge-quantile",
        type=float,
        default=None,
        help="Optional quantile filter for focus edges per frame, e.g. 0.8.",
    )
    parser.add_argument("--focus-edge-width-min", type=float, default=1.0)
    parser.add_argument("--focus-edge-width-max", type=float, default=2.6)
    parser.add_argument("--focus-edge-alpha-min", type=float, default=0.25)
    parser.add_argument("--focus-edge-alpha-max", type=float, default=0.85)
    parser.add_argument("--focus-edge-curve", type=float, default=0.18)
    parser.add_argument("--focus-edge-curve-bi", type=float, default=0.35)
    parser.add_argument("--focus-neighbor-pct", type=float, default=0.2)
    parser.add_argument("--focus-other-desaturate", type=float, default=0)
    parser.add_argument(
        "--focus-show-endpoints",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Draw highlighted dots at communication edge endpoints.",
    )
    parser.add_argument("--focus-endpoint-size", type=float, default=26.0)
    parser.add_argument("--focus-endpoint-alpha", type=float, default=0.95)
    parser.add_argument("--focus-endpoint-edgecolor", default="#ffffff")
    parser.add_argument("--focus-endpoint-linewidth", type=float, default=0.8)
    parser.add_argument("--attention-stride", type=int, default=2)
    parser.add_argument(
        "--attention-max-cells",
        type=int,
        default=30000,
        help="Max cells per timepoint used for communication attention computation.",
    )
    parser.add_argument(
        "--attention-use-real-observed",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use real observed cells (instead of simulated baseline cells) at observed times for communication computation.",
    )
    parser.add_argument(
        "--axis-limit-mode",
        choices=["global", "per_timepoint"],
        default="per_timepoint",
        help="2D/3D view limits: fixed global limits or recomputed per timepoint.",
    )
    parser.add_argument("--axis-pad-frac", type=float, default=0.03)
    parser.add_argument(
        "--panel-size",
        type=float,
        default=4.2,
        help="2D panel size in inches, matching snapshot style when set to 4.2.",
    )
    parser.add_argument("--morph-grid-size", type=int, default=64, help="Grid size for occupancy-based morphology delta.")
    parser.add_argument("--camera-elev", type=float, default=28.0)
    parser.add_argument("--camera-azim", type=float, default=-54.0)

    # Color controls
    parser.add_argument("--color-h5ad", default="spatial_data/Mouse_embryo_all_stage.h5ad")
    parser.add_argument("--label-color-json", default=None)

    # Runtime
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])

    args = parser.parse_args(list(argv) if argv is not None else None)
    target_labels = _parse_target_labels(args.target_labels, args.target_label)
    target_labels_display = _format_target_labels(target_labels)

    os.makedirs(args.output_dir, exist_ok=True)
    frames_dir = os.path.join(args.output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    baseline_dir = os.path.join(args.output_dir, "branch_baseline")
    os.makedirs(baseline_dir, exist_ok=True)
    ablation_dir = os.path.join(args.output_dir, "branch_ablation")
    if args.render_mode == "baseline_vs_ablation":
        os.makedirs(ablation_dir, exist_ok=True)

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

    rng = np.random.default_rng(0 if args.random_seed is None else int(args.random_seed))

    config = load_config(args.config)
    dim = int(config["data"]["dim"])
    feature_cols = [f"x{i}" for i in range(1, dim + 1)]

    df = pd.read_csv(args.data_csv, low_memory=False)
    _require_columns(df, ["samples"] + feature_cols + [args.annotation_key], args.data_csv)
    df = df.copy()
    df["samples"] = df["samples"].astype(float)
    df[args.annotation_key] = df[args.annotation_key].astype(str)
    df = df.sort_values("samples").reset_index(drop=True)

    observed_times = sorted(float(x) for x in df["samples"].unique().tolist())
    t_start_obs = _find_matching_time(args.ablation_start_time, observed_times)

    if not _float_close(float(args.time_start), float(args.ablation_start_time)):
        raise ValueError(
            "This script currently expects --time-start == --ablation-start-time. "
            "Set both to the same value (default 0.0)."
        )

    ts_points = _build_ts_points(args.time_start, args.time_end, args.time_step)

    if ts_points[0] < min(observed_times) - 1e-8:
        raise ValueError(f"time-start {ts_points[0]} is before earliest observed time {min(observed_times)}")

    # Runtime device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested, but CUDA is not available")

    # Load dynamics model
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
    print("Data rows:", len(df))
    print("Observed times:", observed_times)
    print("Simulated times:", ts_points[0], "->", ts_points[-1], "| n=", len(ts_points))
    if bool(args.spatial_warp_to_observed):
        print(
            "Spatial warp: enabled | "
            f"k={int(args.spatial_warp_k)} eps={float(args.spatial_warp_eps)} "
            "baseline_only=true counterfactual=false"
        )

    # Shared classifier training (single model for both branches)
    clf_feature_cols, clf_feature_indices = _build_classifier_feature_spec(
        dim=dim,
        start_idx_1based=int(args.classifier_feature_start),
        n_pcs=int(args.classifier_n_pcs),
    )
    classifier_feature_dim = int(len(clf_feature_cols))
    clf_cols = ["samples"] + list(clf_feature_cols)
    print(
        "Classifier feature slice:",
        f"x{args.classifier_feature_start}..x{args.classifier_feature_start + classifier_feature_dim - 1}",
        f"(n={classifier_feature_dim})",
    )

    cache_dir = None
    cache_path = None
    if args.classifier_cache:
        if args.classifier_cache_path:
            cache_path = str(args.classifier_cache_path)
            print("Classifier cache path (explicit):", cache_path)
        else:
            preferred_cache_dir = str(args.classifier_cache_dir) if args.classifier_cache_dir else ""
            if preferred_cache_dir and os.path.isdir(preferred_cache_dir):
                cache_dir = preferred_cache_dir
                print("Classifier cache dir (preferred):", cache_dir)
            else:
                fallback_dir = os.path.join(args.output_dir, "classifier_cache")
                cache_dir = fallback_dir
                if preferred_cache_dir:
                    print(
                        "[warn] Preferred classifier cache dir not found:",
                        preferred_cache_dir,
                        "| fallback to:",
                        fallback_dir,
                    )
                else:
                    print("Classifier cache dir (fallback):", fallback_dir)

    t_clf0 = time.perf_counter()
    model, label_encoder, clf_acc = train_mlp_classifier(
        df,
        feature_cols=clf_cols,
        label_col=args.annotation_key,
        hidden_size=int(args.classifier_hidden),
        epochs=int(args.classifier_epochs),
        cache_path=cache_path,
        cache_dir=cache_dir,
        cache_tag=args.classifier_cache_tag,
        df_source_path=args.data_csv,
        reuse_if_possible=bool(args.classifier_cache),
        progress=True,
        device=device,
        best_epoch_metric=args.classifier_best_metric,
        train_on_full_data=bool(args.classifier_train_on_full_data),
    )
    print(f"Classifier ready in {time.perf_counter() - t_clf0:.1f}s | metric={clf_acc:.4f}")

    # Build t0 pools
    df_t0 = df[df["samples"] == float(t_start_obs)]
    X_base_pool = df_t0[feature_cols].values.astype(np.float32)
    y_base_pool = df_t0[args.annotation_key].astype(str).values

    if args.render_mode == "baseline_only":
        _run_baseline_only_pipeline(
            args=args,
            df=df,
            feature_cols=feature_cols,
            observed_times=observed_times,
            t_start_obs=t_start_obs,
            ts_points=ts_points,
            f_net=f_net,
            score_net=score_net,
            model=model,
            label_encoder=label_encoder,
            clf_feature_indices=clf_feature_indices,
            device=device,
            exp_dir=exp_dir,
            rng=rng,
        )
        return

    X_ab_pool, y_ab_pool, n_target_t0, n_removed_t0, n_target_t0_per_label, n_removed_t0_per_label = _apply_label_ablation(
        X_pool=X_base_pool,
        labels_pool=y_base_pool,
        target_labels=target_labels,
        remove_frac=args.ablation_remove_frac,
        rng=rng,
    )

    if args.sde_n_samples <= 0:
        raise ValueError("--sde-n-samples must be > 0")

    if args.mass_control:
        n_init = int(min(args.sde_n_samples, X_base_pool.shape[0], X_ab_pool.shape[0]))
        n_init_baseline = n_init
        n_init_ablation = n_init
    else:
        n_init_baseline = int(min(args.sde_n_samples, X_base_pool.shape[0]))
        n_init_ablation = int(min(args.sde_n_samples, X_ab_pool.shape[0]))

    if n_init_baseline <= 0 or n_init_ablation <= 0:
        raise ValueError(
            f"Invalid initial sample sizes baseline={n_init_baseline}, ablation={n_init_ablation}. "
            f"Pool sizes baseline={X_base_pool.shape[0]}, ablation={X_ab_pool.shape[0]}"
        )

    X0_baseline, y0_baseline = _sample_rows_no_replace(X_base_pool, y_base_pool, n_init_baseline, rng)
    X0_ablation, y0_ablation = _sample_rows_no_replace(X_ab_pool, y_ab_pool, n_init_ablation, rng)

    # Sanity checks at t0
    y0_baseline_arr = np.asarray(y0_baseline).astype(str)
    y0_ablation_arr = np.asarray(y0_ablation).astype(str)
    target_mask_baseline = np.isin(y0_baseline_arr, target_labels)
    target_mask_ablation = np.isin(y0_ablation_arr, target_labels)
    t0_baseline_target_count = int(np.sum(target_mask_baseline))
    t0_ablation_target_count = int(np.sum(target_mask_ablation))
    t0_baseline_target_count_per_label = {k: int(np.sum(y0_baseline_arr == k)) for k in target_labels}
    t0_ablation_target_count_per_label = {k: int(np.sum(y0_ablation_arr == k)) for k in target_labels}

    if args.ablation_remove_frac >= 0.999 and t0_ablation_target_count != 0:
        raise RuntimeError(
            "Ablation sanity check failed: target label still exists in ablation t0 sample "
            f"({t0_ablation_target_count} cells)."
        )

    # Save branch initialization artifacts
    np.savez_compressed(
        os.path.join(baseline_dir, "init_pool.npz"),
        X0=X0_baseline,
        labels=np.asarray(y0_baseline).astype(str),
    )
    np.savez_compressed(
        os.path.join(ablation_dir, "init_pool.npz"),
        X0=X0_ablation,
        labels=np.asarray(y0_ablation).astype(str),
    )

    # Simulate both branches
    t_sim0 = time.perf_counter()
    baseline_interaction_scale = 0.0 if bool(args.baseline_zero_interaction) else 1.0
    ablation_interaction_scale = 1.0
    if bool(args.spatial_warp_to_observed):
        if int(args.spatial_warp_k) <= 0:
            raise ValueError("--spatial-warp-k must be > 0")
        if float(args.spatial_warp_eps) <= 0:
            raise ValueError("--spatial-warp-eps must be > 0")
        rng_warp = np.random.default_rng(1 if args.random_seed is None else int(args.random_seed) + 1)
        sde_baseline = _simulate_piecewise_spatially_warped_baseline(
            x0=X0_baseline,
            df=df,
            feature_cols=feature_cols,
            annotation_key=args.annotation_key,
            observed_times=observed_times,
            ts_points=ts_points,
            f_net=f_net,
            score_net=score_net,
            dt=float(args.split_sde_dt),
            sigma=float(args.split_sigma),
            growth_alpha=float(args.split_growth_alpha),
            interaction_m=int(args.interaction_m),
            interaction_scale=float(baseline_interaction_scale),
            device=device,
            rng=rng_warp,
            warp_k=int(args.spatial_warp_k),
            warp_eps=float(args.spatial_warp_eps),
        )
    else:
        sde_baseline = _simulate_sde_points_split_from_x0(
            x0=X0_baseline,
            f_net=f_net,
            score_net=score_net,
            ts_points=ts_points,
            dt=float(args.split_sde_dt),
            sigma=float(args.split_sigma),
            growth_alpha=float(args.split_growth_alpha),
            interaction_m=int(args.interaction_m),
            interaction_scale=float(baseline_interaction_scale),
            device=device,
            verbose=True,
        )
    sde_ablation = _simulate_sde_points_split_from_x0(
        x0=X0_ablation,
        f_net=f_net,
        score_net=score_net,
        ts_points=ts_points,
        dt=float(args.split_sde_dt),
        sigma=float(args.split_sigma),
        growth_alpha=float(args.split_growth_alpha),
        interaction_m=int(args.interaction_m),
        interaction_scale=float(ablation_interaction_scale),
        device=device,
        verbose=True,
    )
    print(f"Both branch simulations finished in {time.perf_counter() - t_sim0:.1f}s")

    # Shared classifier predictions
    pred_baseline = _predict_labels_for_trajectories_with_indices(
        sde_points=sde_baseline,
        ts_points=ts_points,
        model=model,
        label_encoder=label_encoder,
        feature_indices=clf_feature_indices,
        device=device,
    )
    pred_ablation = _predict_labels_for_trajectories_with_indices(
        sde_points=sde_ablation,
        ts_points=ts_points,
        model=model,
        label_encoder=label_encoder,
        feature_indices=clf_feature_indices,
        device=device,
    )

    if not args.strict_counterfactual:
        print("[warn] strict-counterfactual disabled: re-anchoring branch states at observed times")
        _apply_observed_anchor_if_needed(
            branch_name="baseline",
            ts_points=ts_points,
            sde_points=sde_baseline,
            pred_labels=pred_baseline,
            df=df,
            annotation_key=args.annotation_key,
            feature_cols=feature_cols,
            observed_times=observed_times,
            start_time=t_start_obs,
            target_labels=target_labels,
            remove_frac=args.ablation_remove_frac,
            n_samples_branch=n_init_baseline,
            rng=rng,
        )
        _apply_observed_anchor_if_needed(
            branch_name="ablation",
            ts_points=ts_points,
            sde_points=sde_ablation,
            pred_labels=pred_ablation,
            df=df,
            annotation_key=args.annotation_key,
            feature_cols=feature_cols,
            observed_times=observed_times,
            start_time=t_start_obs,
            target_labels=target_labels,
            remove_frac=args.ablation_remove_frac,
            n_samples_branch=n_init_ablation,
            rng=rng,
        )

    # Save branch trajectories (compact, object array)
    np.savez_compressed(
        os.path.join(baseline_dir, "trajectory_pred.npz"),
        ts=np.asarray(ts_points, dtype=np.float32),
        labels=np.asarray(pred_baseline, dtype=object),
    )
    np.savez_compressed(
        os.path.join(ablation_dir, "trajectory_pred.npz"),
        ts=np.asarray(ts_points, dtype=np.float32),
        labels=np.asarray(pred_ablation, dtype=object),
    )

    # Build color map from observed + predicted labels
    observed_label_values = df[args.annotation_key].astype(str).values
    label_to_color = load_label_to_color(
        labels=observed_label_values,
        label_color_json=args.label_color_json,
        color_h5ad=args.color_h5ad,
        annotation_key=args.annotation_key,
    )
    all_pred_labels = []
    for arr in pred_baseline:
        all_pred_labels.extend(np.asarray(arr).astype(str).tolist())
    for arr in pred_ablation:
        all_pred_labels.extend(np.asarray(arr).astype(str).tolist())
    label_to_color = _ensure_color_map_has_labels(label_to_color, all_pred_labels)
    with open(os.path.join(args.output_dir, "label_to_color.json"), "w", encoding="utf-8") as f:
        json.dump(label_to_color, f, indent=2, ensure_ascii=False)

    # Render frames
    xlim_global, ylim_global = _compute_xy_limits(
        sde_baseline,
        sde_ablation,
        pad_frac=float(args.axis_pad_frac),
    )
    zlim = (float(args.time_start), float(args.time_end)) if args.video_style == "fixed_3d" else None

    frame_paths: list[str] = []
    morph_rows: list[dict] = []
    render_rng = np.random.default_rng((0 if args.random_seed is None else int(args.random_seed)) + 1024)

    t_render0 = time.perf_counter()
    for i, t in enumerate(ts_points):
        Xb = np.asarray(sde_baseline[i], dtype=np.float32)
        yb = np.asarray(pred_baseline[i]).astype(str)
        Xa = np.asarray(sde_ablation[i], dtype=np.float32)
        ya = np.asarray(pred_ablation[i]).astype(str)

        if args.axis_limit_mode == "per_timepoint":
            xlim_frame, ylim_frame = _compute_xy_limits(
                [Xb[:, :2]],
                [Xa[:, :2]],
                pad_frac=float(args.axis_pad_frac),
            )
        else:
            xlim_frame, ylim_frame = xlim_global, ylim_global

        morph = _compute_morph_delta_metrics(
            baseline_xy=Xb[:, :2],
            ablation_xy=Xa[:, :2],
            xlim=xlim_global,
            ylim=ylim_global,
            grid_size=int(args.morph_grid_size),
        )
        morph_rows.append(
            {
                "time": float(t),
                "n_baseline": int(Xb.shape[0]),
                "n_ablation": int(Xa.shape[0]),
                **morph,
            }
        )

        Xb_vis, yb_vis = _downsample_for_render(Xb, yb, args.video_point_subsample, render_rng)
        Xa_vis, ya_vis = _downsample_for_render(Xa, ya, args.video_point_subsample, render_rng)

        frame_path = os.path.join(frames_dir, f"frame_{i:03d}.png")
        if args.video_style == "fixed_2d":
            _render_frame_side_by_side_2d(
                baseline_xy=Xb_vis[:, :2],
                baseline_labels=yb_vis,
                ablation_xy=Xa_vis[:, :2],
                ablation_labels=ya_vis,
                time_value=float(t),
                label_to_color=label_to_color,
                out_png=frame_path,
                xlim=xlim_frame,
                ylim=ylim_frame,
                point_size=float(args.point_size),
                alpha=float(args.point_alpha),
                dpi=int(args.frame_dpi),
                panel_size=float(args.panel_size),
                baseline_title="Baseline",
                ablation_title=f"Ablation: -{target_labels_display}",
            )
        else:
            _render_frame_side_by_side_3d(
                baseline_xy=Xb_vis[:, :2],
                baseline_labels=yb_vis,
                ablation_xy=Xa_vis[:, :2],
                ablation_labels=ya_vis,
                time_value=float(t),
                label_to_color=label_to_color,
                out_png=frame_path,
                xlim=xlim_frame,
                ylim=ylim_frame,
                zlim=zlim,
                elev=float(args.camera_elev),
                azim=float(args.camera_azim),
                point_size=float(args.point_size),
                alpha=float(args.point_alpha),
                dpi=int(args.frame_dpi),
                baseline_title="Baseline",
                ablation_title=f"Ablation: -{target_labels_display}",
            )
        frame_paths.append(frame_path)
        print(f"Rendered frame {i + 1}/{len(ts_points)} | t={t:.3f}")

    print(f"Frame rendering finished in {time.perf_counter() - t_render0:.1f}s")

    morph_df = pd.DataFrame(morph_rows).sort_values("time").reset_index(drop=True)
    morph_csv_path = os.path.join(args.output_dir, "morphology_delta_by_time.csv")
    morph_df.to_csv(morph_csv_path, index=False)

    t_vals = morph_df["time"].to_numpy(dtype=float)
    delta_vals = morph_df["morph_delta_t"].to_numpy(dtype=float)
    auc_morph = float(np.trapz(delta_vals, t_vals)) if len(delta_vals) > 1 else 0.0
    morph_summary = {
        "status": "ok",
        "target_label": target_labels_display,
        "target_labels": list(target_labels),
        "metrics": {
            "mean_morph_delta": float(np.mean(delta_vals)) if len(delta_vals) > 0 else float("nan"),
            "max_morph_delta": float(np.max(delta_vals)) if len(delta_vals) > 0 else float("nan"),
            "auc_morph_delta": auc_morph,
            "mean_centroid_shift_norm": float(np.mean(morph_df["centroid_shift_norm"])) if len(morph_df) > 0 else float("nan"),
            "mean_occupancy_iou": float(np.mean(morph_df["occupancy_iou"])) if len(morph_df) > 0 else float("nan"),
            "mean_spread_ratio_delta": float(np.mean(morph_df["spread_ratio_delta"])) if len(morph_df) > 0 else float("nan"),
        },
        "paths": {
            "morphology_delta_by_time_csv": morph_csv_path,
        },
    }
    morph_summary_path = os.path.join(args.output_dir, "morphology_delta_summary.json")
    with open(morph_summary_path, "w", encoding="utf-8") as f:
        json.dump(morph_summary, f, indent=2, ensure_ascii=False)
    print("Saved:", morph_csv_path)
    print("Saved:", morph_summary_path)

    # Assemble GIF
    gif_name = f"baseline_vs_ablation_{_tag_float(args.time_start)}_to_{_tag_float(args.time_end)}_step{_tag_float(args.time_step)}.gif"
    gif_path = os.path.join(args.output_dir, gif_name)

    duration = 1.0 / max(1, int(args.gif_fps))
    images = [imageio.imread(p) for p in frame_paths]
    imageio.mimsave(gif_path, images, duration=duration, loop=0)
    print("Saved GIF:", gif_path)

    # Summary
    summary = {
        "status": "ok",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "render_mode": "baseline_vs_ablation",
        "data_csv": args.data_csv,
        "annotation_key": args.annotation_key,
        "target_label": target_labels_display,
        "target_labels": list(target_labels),
        "ablation_start_time": float(args.ablation_start_time),
        "ablation_remove_frac": float(args.ablation_remove_frac),
        "mass_control": bool(args.mass_control),
        "strict_counterfactual": bool(args.strict_counterfactual),
        "time_start": float(args.time_start),
        "time_end": float(args.time_end),
        "time_step": float(args.time_step),
        "timepoints": [float(x) for x in ts_points],
        "n_frames": int(len(ts_points)),
        "gif_path": gif_path,
        "frames_dir": frames_dir,
        "frame_count": int(len(frame_paths)),
        "frame_files_head": [os.path.basename(x) for x in frame_paths[:5]],
        "frame_files_tail": [os.path.basename(x) for x in frame_paths[-5:]],
        "morphology_delta_by_time_csv": morph_csv_path,
        "morphology_delta_summary_json": morph_summary_path,
        "observed_times": [float(x) for x in observed_times],
        "n_observed_rows": int(df.shape[0]),
        "n_t0_pool_baseline": int(X_base_pool.shape[0]),
        "n_t0_pool_ablation": int(X_ab_pool.shape[0]),
        "n_target_t0_before_ablation": int(n_target_t0),
        "n_target_t0_before_ablation_per_label": n_target_t0_per_label,
        "n_target_t0_removed": int(n_removed_t0),
        "n_target_t0_removed_per_label": n_removed_t0_per_label,
        "n_init_baseline": int(n_init_baseline),
        "n_init_ablation": int(n_init_ablation),
        "n_target_in_baseline_init_sample": int(t0_baseline_target_count),
        "n_target_in_baseline_init_sample_per_label": t0_baseline_target_count_per_label,
        "n_target_in_ablation_init_sample": int(t0_ablation_target_count),
        "n_target_in_ablation_init_sample_per_label": t0_ablation_target_count_per_label,
        "classifier": {
            "n_pcs": int(classifier_feature_dim),
            "feature_start_x": int(args.classifier_feature_start),
            "feature_end_x": int(args.classifier_feature_start + classifier_feature_dim - 1),
            "feature_cols": list(clf_feature_cols),
            "epochs": int(args.classifier_epochs),
            "hidden": int(args.classifier_hidden),
            "best_metric": str(args.classifier_best_metric),
            "train_on_full_data": bool(args.classifier_train_on_full_data),
            "cache_enabled": bool(args.classifier_cache),
            "cache_path": cache_path,
            "cache_dir": cache_dir,
            "cache_tag": args.classifier_cache_tag,
            "reported_metric": float(clf_acc),
        },
        "simulation": {
            "split_sde_dt": float(args.split_sde_dt),
            "split_sigma": float(args.split_sigma),
            "split_growth_alpha": float(args.split_growth_alpha),
            "spatial_warp_to_observed": bool(args.spatial_warp_to_observed),
            "spatial_warp_k": int(args.spatial_warp_k),
            "spatial_warp_eps": float(args.spatial_warp_eps),
            "spatial_warp_mode": "piecewise_rerun_baseline_only" if bool(args.spatial_warp_to_observed) else "off",
            "interaction_m": int(args.interaction_m),
            "baseline_zero_interaction": bool(args.baseline_zero_interaction),
            "baseline_interaction_scale": float(baseline_interaction_scale),
            "ablation_interaction_scale": float(ablation_interaction_scale),
            "sde_n_samples_requested": int(args.sde_n_samples),
            "model_exp_dir": exp_dir,
            "device": device,
        },
        "camera": {
            "style": args.video_style,
            "layout": args.video_layout,
            "elev": float(args.camera_elev),
            "azim": float(args.camera_azim),
            "axis_limit_mode": str(args.axis_limit_mode),
            "axis_pad_frac": float(args.axis_pad_frac),
            "xlim_global": [float(xlim_global[0]), float(xlim_global[1])],
            "ylim_global": [float(ylim_global[0]), float(ylim_global[1])],
            "zlim": [float(zlim[0]), float(zlim[1])] if zlim is not None else None,
            "point_size": float(args.point_size),
            "point_alpha": float(args.point_alpha),
            "panel_size": float(args.panel_size),
            "video_point_subsample": int(args.video_point_subsample),
            "gif_fps": int(args.gif_fps),
            "frame_dpi": int(args.frame_dpi),
        },
        "notes": {
            "biological_viability_mode": "weak_constraint",
            "phase1_outputs": "morphology_video_plus_morph_delta_metrics",
            "no_attention_comm_or_lineage_metrics": True,
            "time_horizon_extrapolation": False,
        },
    }

    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("Saved summary:", os.path.join(args.output_dir, "summary.json"))
    print("Done.")


if __name__ == "__main__":
    main()
