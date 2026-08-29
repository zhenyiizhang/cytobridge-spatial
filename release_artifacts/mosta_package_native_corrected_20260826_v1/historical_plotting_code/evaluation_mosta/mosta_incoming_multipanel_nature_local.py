#!/usr/bin/env python3
"""Render a publication-style hotspot multi-panel figure for MOSTA."""

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


@dataclass
class PanelSpec:
    label: str
    scores_pkl: str
    h5ad_path: str
    annotation_col: str
    time_key: str | None
    no_filter_time: bool
    invert_y: bool


@dataclass
class PanelData:
    label: str
    coords: np.ndarray
    values_raw: np.ndarray
    invert_y: bool


def _split_csv_keep_empty(s: str) -> List[str]:
    return [x.strip() for x in str(s).split(",")]


def _parse_bool_csv(s: str) -> List[bool]:
    out: List[bool] = []
    for tok in _split_csv_keep_empty(s):
        low = tok.lower()
        if low in ("1", "true", "t", "yes", "y"):
            out.append(True)
        elif low in ("0", "false", "f", "no", "n"):
            out.append(False)
        else:
            raise ValueError(f"Invalid bool token '{tok}' in CSV: {s}")
    return out


def _normalize_time_token(tok: str) -> str | None:
    if tok is None:
        return None
    t = str(tok).strip()
    if t == "" or t.lower() in ("na", "none", "null"):
        return None
    return t


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
        vals = lr_mat.sum(axis=0)
    elif mode == "outgoing":
        vals = lr_mat.sum(axis=1)
    elif mode == "total":
        vals = lr_mat.sum(axis=0) + lr_mat.sum(axis=1)
    else:
        raise ValueError(f"Unsupported hotspot mode: {mode}")
    return {ct: float(v) for ct, v in zip(cell_types, vals)}


def _load_panel(
    *,
    spec: PanelSpec,
    lr_pair: str,
    hotspot_mode: str,
    timepoint_col: str,
    spatial_key: str,
) -> PanelData:
    ct_to_in = _hotspot_by_cell_type(scores_pkl=spec.scores_pkl, lr_pair=lr_pair, mode=hotspot_mode)

    adata = ad.read_h5ad(spec.h5ad_path, backed="r")
    ann_col = _resolve_column(adata.obs.columns, spec.annotation_col, ["annotation", "Annotation", "cell_type"])

    if spec.no_filter_time:
        adata_t = adata
    else:
        if spec.time_key is None:
            raise ValueError("time_key is required when no_filter_time=False")
        tp_col = _resolve_column(adata.obs.columns, timepoint_col, ["timepoint", "samples", "time", "batch"])
        mask = adata.obs[tp_col].astype(str).to_numpy() == str(spec.time_key)
        if not mask.any():
            avail = sorted(pd.Series(adata.obs[tp_col].astype(str)).unique().tolist())
            raise KeyError(f"time_key '{spec.time_key}' not in {tp_col}. Available: {avail}")
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
    values_raw = np.array([ct_to_in.get(a, 0.0) for a in ann], dtype=float)
    return PanelData(label=spec.label, coords=coords, values_raw=values_raw, invert_y=spec.invert_y)


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
    p = argparse.ArgumentParser(description="Optimized hotspot multi-panel figure.")
    p.add_argument("--lr-pair", default="Wnt3a_Fzd7_Lrp6")
    p.add_argument("--hotspot-mode", choices=["incoming", "outgoing", "total"], default="incoming")
    p.add_argument("--labels", required=True, help="CSV labels, e.g. E12.5,0.5,E13.5")
    p.add_argument("--scores-pkls", required=True, help="CSV pickle paths (same length as labels)")
    p.add_argument("--h5ads", required=True, help="CSV h5ad paths (same length as labels)")
    p.add_argument("--annotation-cols", required=True, help="CSV annotation columns (same length as labels)")
    p.add_argument(
        "--time-keys",
        required=True,
        help="CSV time keys (same length). Use NA for not used when corresponding no-filter flag is 1.",
    )
    p.add_argument(
        "--no-filter-flags",
        required=True,
        help="CSV bools 0/1 for time filtering (same length). 1 means use all cells in that h5ad.",
    )
    p.add_argument(
        "--invert-y-flags",
        required=True,
        help="CSV bools 0/1 for Y inversion per panel (same length).",
    )
    p.add_argument("--timepoint-col", default="timepoint")
    p.add_argument("--spatial-key", default="spatial")
    p.add_argument("--norm-mode", choices=["global", "per_panel"], default="per_panel")
    p.add_argument("--q-low", type=float, default=1.0)
    p.add_argument("--q-high", type=float, default=99.0)
    p.add_argument("--cmap", default="incoming_pastel", help="Use 'incoming_pastel' or any matplotlib cmap name.")
    p.add_argument("--point-size", type=float, default=3.0)
    p.add_argument("--alpha", type=float, default=0.95)
    p.add_argument("--fig-width", type=float, default=20.0)
    p.add_argument("--fig-height", type=float, default=3.8)
    p.add_argument("--dpi", type=int, default=500)
    p.add_argument("--out-prefix", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    labels = _split_csv_keep_empty(args.labels)
    scores_pkls = _split_csv_keep_empty(args.scores_pkls)
    h5ads = _split_csv_keep_empty(args.h5ads)
    annotation_cols = _split_csv_keep_empty(args.annotation_cols)
    time_keys = [_normalize_time_token(x) for x in _split_csv_keep_empty(args.time_keys)]
    no_filter_flags = _parse_bool_csv(args.no_filter_flags)
    invert_y_flags = _parse_bool_csv(args.invert_y_flags)

    n = len(labels)
    lengths = {
        "scores-pkls": len(scores_pkls),
        "h5ads": len(h5ads),
        "annotation-cols": len(annotation_cols),
        "time-keys": len(time_keys),
        "no-filter-flags": len(no_filter_flags),
        "invert-y-flags": len(invert_y_flags),
    }
    bad = {k: v for k, v in lengths.items() if v != n}
    if bad:
        raise ValueError(f"All CSV args must have same length as labels={n}; mismatched={bad}")

    specs: List[PanelSpec] = []
    for i in range(n):
        specs.append(
            PanelSpec(
                label=labels[i],
                scores_pkl=scores_pkls[i],
                h5ad_path=h5ads[i],
                annotation_col=annotation_cols[i],
                time_key=time_keys[i],
                no_filter_time=no_filter_flags[i],
                invert_y=invert_y_flags[i],
            )
        )

    out_dir = os.path.dirname(args.out_prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    panels = [
        _load_panel(
            spec=s,
            lr_pair=args.lr_pair,
            hotspot_mode=args.hotspot_mode,
            timepoint_col=args.timepoint_col,
            spatial_key=args.spatial_key,
        )
        for s in specs
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
    fig, axes = plt.subplots(1, n, figsize=(args.fig_width, args.fig_height), dpi=args.dpi, facecolor="white")
    if n == 1:
        axes = [axes]

    for i, (ax, panel, val) in enumerate(zip(axes, panels, vals_norm)):
        coords = panel.coords
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

    cbar = fig.colorbar(sc, ax=axes, fraction=0.02, pad=0.01)
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
