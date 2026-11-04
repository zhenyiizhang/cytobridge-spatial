#!/usr/bin/env python3
"""Render a publication-style 3-panel hotspot figure for MOSTA (t=0,0.5,1).

This script is intentionally focused on one job: make the three hotspot
maps visually consistent and "Nature-like" for figure assembly.
"""

from __future__ import annotations

import argparse
import os
import pickle
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap


DEFAULT_SCORES_T0 = "results/mosta_wnt_0_05_1/rerun_0_05_1/proj_real01/lr_scores_0.0.pkl"
DEFAULT_SCORES_T05 = "results/mosta_wnt_0_05_1/rerun_0_05_1/proj_interp05/lr_scores_0.5.pkl"
DEFAULT_SCORES_T1 = "results/mosta_wnt_0_05_1/rerun_0_05_1/proj_real01/lr_scores_1.0.pkl"
DEFAULT_REAL_H5AD = "spatial_data/Mouse_embryo_all_stage.h5ad"
DEFAULT_INTERP_H5AD = "results/mosta_lr_batch/interp_t0p5_n50000/adata_t0p500_with_genes.h5ad"
DEFAULT_OUT_PREFIX = "results/mosta_wnt_0_05_1/incoming_triptych_nature/Wnt3a_Fzd7_Lrp6_incoming_triptych"


@dataclass
class PanelData:
    label: str
    coords: np.ndarray
    values_raw: np.ndarray
    invert_y: bool


def _resolve_column(columns: Sequence[str], requested: str, fallbacks: Sequence[str]) -> str:
    cols = list(columns)
    if requested in cols:
        return requested
    low_to_col = {c.lower(): c for c in cols}
    if requested.lower() in low_to_col:
        return low_to_col[requested.lower()]
    for fb in fallbacks:
        if fb in cols:
            return fb
        if fb.lower() in low_to_col:
            return low_to_col[fb.lower()]
    raise KeyError(f"Could not resolve column '{requested}' from {cols}")


def _hotspot_by_cell_type(scores_pkl: str, lr_pair: str, mode: str) -> Dict[str, float]:
    with open(scores_pkl, "rb") as f:
        data = pickle.load(f)
    lr_scores = data["lr_scores"]
    if lr_pair not in lr_scores:
        sample = sorted(lr_scores.keys())[:20]
        raise KeyError(f"LR pair '{lr_pair}' not found in {scores_pkl}. Sample keys: {sample}")
    lr_mat = np.asarray(lr_scores[lr_pair], dtype=float)
    cell_types = [str(x) for x in np.asarray(data["cell_types"]).tolist()]
    if mode == "incoming":
        vals = lr_mat.sum(axis=0)  # receiver side
    elif mode == "outgoing":
        vals = lr_mat.sum(axis=1)  # sender side
    elif mode == "total":
        vals = lr_mat.sum(axis=0) + lr_mat.sum(axis=1)
    else:
        raise ValueError(f"Unsupported hotspot mode: {mode}")
    return {ct: float(v) for ct, v in zip(cell_types, vals)}


def _load_panel(
    *,
    label: str,
    h5ad_path: str,
    scores_pkl: str,
    lr_pair: str,
    annotation_col: str,
    time_key: str | None,
    timepoint_col: str,
    spatial_key: str,
    invert_y: bool,
    no_filter_time: bool,
    hotspot_mode: str,
) -> PanelData:
    ct_to_hot = _hotspot_by_cell_type(scores_pkl=scores_pkl, lr_pair=lr_pair, mode=hotspot_mode)

    adata = ad.read_h5ad(h5ad_path, backed="r")
    ann_col = _resolve_column(adata.obs.columns, annotation_col, ["annotation", "Annotation", "cell_type"])

    if no_filter_time:
        adata_t = adata
    else:
        if time_key is None:
            raise ValueError("time_key is required when no_filter_time=False")
        tp_col = _resolve_column(adata.obs.columns, timepoint_col, ["timepoint", "samples", "time", "batch"])
        mask = adata.obs[tp_col].astype(str).to_numpy() == str(time_key)
        if not mask.any():
            avail = sorted(pd.Series(adata.obs[tp_col].astype(str)).unique().tolist())
            raise KeyError(f"time_key '{time_key}' not in {tp_col}. Available: {avail}")
        adata_t = adata[np.flatnonzero(mask), :]

    adata_t = adata_t.to_memory() if hasattr(adata_t, "to_memory") else adata_t.copy()
    try:
        if hasattr(adata, "file") and adata.file is not None:
            adata.file.close()
    except Exception:
        pass

    if spatial_key not in adata_t.obsm:
        raise KeyError(f"spatial key '{spatial_key}' not in adata.obsm")
    coords = np.asarray(adata_t.obsm[spatial_key])[:, :2]
    ann = adata_t.obs[ann_col].astype(str).to_numpy()
    values_raw = np.array([ct_to_hot.get(a, 0.0) for a in ann], dtype=float)
    return PanelData(label=label, coords=coords, values_raw=values_raw, invert_y=invert_y)


def _clip_norm(values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    v = np.clip(values, lo, hi)
    if hi <= lo:
        return np.zeros_like(v, dtype=float)
    return (v - lo) / (hi - lo)


def _robust_bounds(values: np.ndarray, q_low: float, q_high: float) -> Tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0, 1.0
    lo = float(np.percentile(finite, q_low))
    hi = float(np.percentile(finite, q_high))
    if hi <= lo:
        lo = float(np.min(finite))
        hi = float(np.max(finite))
    if hi <= lo:
        hi = lo + 1e-12
    return lo, hi


def _build_cmap(name: str):
    if name.lower() == "incoming_pastel":
        return LinearSegmentedColormap.from_list(
            "incoming_pastel",
            ["#f8fbfd", "#eaf5f4", "#d4ece9", "#b7dedb", "#93c9cb", "#66abb4"],
        )
    return plt.get_cmap(name)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Optimized 3-panel hotspot figure.")
    p.add_argument("--lr-pair", default="Wnt3a_Fzd7_Lrp6")
    p.add_argument("--hotspot-mode", choices=["incoming", "outgoing", "total"], default="incoming")
    p.add_argument("--scores-t0", default=DEFAULT_SCORES_T0)
    p.add_argument("--scores-t05", default=DEFAULT_SCORES_T05)
    p.add_argument("--scores-t1", default=DEFAULT_SCORES_T1)
    p.add_argument("--real-h5ad", default=DEFAULT_REAL_H5AD)
    p.add_argument("--interp-h5ad", default=DEFAULT_INTERP_H5AD)
    p.add_argument("--real-time-0", default="E12.5")
    p.add_argument("--real-time-1", default="E13.5")
    p.add_argument("--annotation-col-real", default="annotation")
    p.add_argument("--annotation-col-interp", default="Annotation")
    p.add_argument("--timepoint-col", default="timepoint")
    p.add_argument("--spatial-key", default="spatial")
    p.add_argument(
        "--interp-time-key",
        default=None,
        help="Optional time key for middle panel; used when --filter-interp-time is set.",
    )
    p.add_argument(
        "--filter-interp-time",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Filter middle panel by --interp-time-key in --interp-h5ad instead of using all cells.",
    )
    p.add_argument("--label-mid", default="t=0.5", help="Display label for middle panel title.")
    p.add_argument("--invert-y-real", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--invert-y-interp", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--norm-mode", choices=["global", "per_panel"], default="per_panel")
    p.add_argument("--q-low", type=float, default=1.0)
    p.add_argument("--q-high", type=float, default=99.0)
    p.add_argument("--cmap", default="incoming_pastel", help="Use 'incoming_pastel' or any matplotlib cmap name.")
    p.add_argument("--point-size", type=float, default=3.0)
    p.add_argument("--alpha", type=float, default=0.95)
    p.add_argument("--fig-width", type=float, default=11.0)
    p.add_argument("--fig-height", type=float, default=3.8)
    p.add_argument("--dpi", type=int, default=500)
    p.add_argument("--out-prefix", default=DEFAULT_OUT_PREFIX)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    out_dir = os.path.dirname(args.out_prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    panels: List[PanelData] = [
        _load_panel(
            label=f"{args.real_time_0}",
            h5ad_path=args.real_h5ad,
            scores_pkl=args.scores_t0,
            lr_pair=args.lr_pair,
            annotation_col=args.annotation_col_real,
            time_key=args.real_time_0,
            timepoint_col=args.timepoint_col,
            spatial_key=args.spatial_key,
            invert_y=bool(args.invert_y_real),
            no_filter_time=False,
            hotspot_mode=args.hotspot_mode,
        ),
        _load_panel(
            label=args.label_mid,
            h5ad_path=args.interp_h5ad,
            scores_pkl=args.scores_t05,
            lr_pair=args.lr_pair,
            annotation_col=args.annotation_col_interp,
            time_key=args.interp_time_key,
            timepoint_col=args.timepoint_col,
            spatial_key=args.spatial_key,
            invert_y=bool(args.invert_y_interp),
            no_filter_time=(not bool(args.filter_interp_time)),
            hotspot_mode=args.hotspot_mode,
        ),
        _load_panel(
            label=f"{args.real_time_1}",
            h5ad_path=args.real_h5ad,
            scores_pkl=args.scores_t1,
            lr_pair=args.lr_pair,
            annotation_col=args.annotation_col_real,
            time_key=args.real_time_1,
            timepoint_col=args.timepoint_col,
            spatial_key=args.spatial_key,
            invert_y=bool(args.invert_y_real),
            no_filter_time=False,
            hotspot_mode=args.hotspot_mode,
        ),
    ]

    if args.norm_mode == "global":
        all_vals = np.concatenate([p.values_raw for p in panels], axis=0)
        lo, hi = _robust_bounds(all_vals, q_low=args.q_low, q_high=args.q_high)
        vals_norm = [_clip_norm(p.values_raw, lo, hi) for p in panels]
    else:
        vals_norm = []
        for p in panels:
            lo, hi = _robust_bounds(p.values_raw, q_low=args.q_low, q_high=args.q_high)
            vals_norm.append(_clip_norm(p.values_raw, lo, hi))

    cmap = _build_cmap(args.cmap)
    fig, axes = plt.subplots(1, 3, figsize=(args.fig_width, args.fig_height), dpi=args.dpi, facecolor="white")

    for i, (ax, panel, val) in enumerate(zip(axes, panels, vals_norm)):
        coords = panel.coords
        # light tissue silhouette for smoother look
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c="#edf2f4",
            s=max(0.5, args.point_size * 0.9),
            alpha=0.35,
            edgecolors="none",
            rasterized=True,
        )
        sc = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=val,
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
            s=args.point_size,
            alpha=args.alpha,
            edgecolors="none",
            rasterized=True,
        )
        ax.set_aspect("equal")
        ax.set_axis_off()
        if panel.invert_y:
            ax.invert_yaxis()
        ax.set_title(panel.label, fontsize=11, pad=7)
        ax.text(0.0, 1.02, chr(ord("a") + i), transform=ax.transAxes, fontsize=11, fontweight="bold")

    cbar = fig.colorbar(sc, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
    cbar.set_label(f"{args.hotspot_mode.capitalize()} hotspot (normalized)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    cbar.outline.set_visible(False)

    fig.suptitle(f"{args.lr_pair} {args.hotspot_mode} hotspot", y=0.99, fontsize=12)

    pdf_path = f"{args.out_prefix}.pdf"
    png_path = f"{args.out_prefix}.png"
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    fig.savefig(png_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    stats = []
    for p, v in zip(panels, vals_norm):
        stats.append(
            {
                "panel": p.label,
                "n_cells": int(p.coords.shape[0]),
                "raw_min": float(np.min(p.values_raw)),
                "raw_max": float(np.max(p.values_raw)),
                "norm_min": float(np.min(v)),
                "norm_max": float(np.max(v)),
            }
        )
    stats_df = pd.DataFrame(stats)
    stats_path = f"{args.out_prefix}_stats.csv"
    stats_df.to_csv(stats_path, index=False)

    print("Saved:", pdf_path)
    print("Saved:", png_path)
    print("Saved:", stats_path)


if __name__ == "__main__":
    main()
