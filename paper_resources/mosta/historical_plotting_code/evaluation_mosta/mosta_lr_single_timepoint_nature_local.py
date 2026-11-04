#!/usr/bin/env python3
"""Single-timepoint LR spatial plots in a Nature Methods-like style.

Outputs:
- ligand expression spatial map
- receptor expression spatial map
- ligand*receptor interaction-potential map
- incoming LR communication hotspot
- outgoing LR communication hotspot
- total LR communication hotspot (incoming + outgoing)
- combined panel (all six maps)
"""

from __future__ import annotations

import argparse
import os
import pickle
import shutil
from typing import Dict, List, Optional, Sequence, Tuple

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse


DEFAULT_SCORES_PKL = "results/mosta_lr_projection_local_debug_deepruotv2/lr_scores_E12.5.pkl"
DEFAULT_ADATA_H5AD = "spatial_data/Mouse_embryo_all_stage.h5ad"
DEFAULT_LR_DB = "database/CellChatDB.ligrec.mouse.csv"
DEFAULT_OUT_DIR = "results/mosta_lr_single_timepoint"


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


def _safe_norm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    lo = np.nanmin(x)
    hi = np.nanmax(x)
    if (not np.isfinite(lo)) or (not np.isfinite(hi)) or (hi <= lo):
        return np.zeros_like(x, dtype=float)
    return (x - lo) / (hi - lo)


def _robust_range(values: np.ndarray, lo_q: float = 1.0, hi_q: float = 99.0) -> Tuple[float, float]:
    v = np.asarray(values, dtype=float)
    finite = v[np.isfinite(v)]
    if finite.size == 0:
        return 0.0, 1.0
    lo = float(np.percentile(finite, lo_q))
    hi = float(np.percentile(finite, hi_q))
    if hi <= lo:
        lo = float(np.min(finite))
        hi = float(np.max(finite))
    if hi <= lo:
        hi = lo + 1e-9
    return lo, hi


def _clip_norm01(values: np.ndarray, lo_q: float, hi_q: float) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    lo, hi = _robust_range(v, lo_q=lo_q, hi_q=hi_q)
    v = np.clip(v, lo, hi)
    if hi <= lo:
        return np.zeros_like(v, dtype=float)
    return (v - lo) / (hi - lo)


def _load_lr_db(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"LR database is empty: {path}")

    cols = list(df.columns)
    low_to_col = {c.lower(): c for c in cols}
    lig_col = None
    rec_col = None

    for c in ("ligand", "0"):
        if c in low_to_col:
            lig_col = low_to_col[c]
            break
    for c in ("receptor", "1"):
        if c in low_to_col:
            rec_col = low_to_col[c]
            break

    if lig_col is None or rec_col is None:
        non_unnamed = [c for c in cols if not c.lower().startswith("unnamed")]
        if len(non_unnamed) >= 2:
            lig_col = non_unnamed[0]
            rec_col = non_unnamed[1]

    if lig_col is None or rec_col is None:
        raise ValueError(f"Could not identify ligand/receptor columns from {path}, columns={cols}")

    out = df[[lig_col, rec_col]].rename(columns={lig_col: "ligand", rec_col: "receptor"}).copy()
    out["ligand"] = out["ligand"].astype(str).str.strip()
    out["receptor"] = out["receptor"].astype(str).str.strip()
    out["lr_pair"] = out["ligand"] + "_" + out["receptor"]
    return out.drop_duplicates(subset=["lr_pair"]).reset_index(drop=True)


def _parse_lr_tokens(lr_pair: str, lr_db: Optional[pd.DataFrame]) -> Tuple[str, str]:
    if lr_db is not None:
        hit = lr_db[lr_db["lr_pair"] == lr_pair]
        if len(hit) == 1:
            row = hit.iloc[0]
            return str(row["ligand"]), str(row["receptor"])
    # fallback
    if "_" not in lr_pair:
        raise ValueError(f"Invalid lr_pair '{lr_pair}', expected 'Ligand_Receptor'")
    ligand = lr_pair.split("_", 1)[0]
    receptor = lr_pair.split("_", 1)[1]
    return ligand, receptor


def _split_complex(token: str) -> List[str]:
    return [x.strip() for x in str(token).split("_") if x.strip()]


def _extract_gene_vector(adata_t, gene: str, x_space: str) -> Optional[np.ndarray]:
    var_names = pd.Index(adata_t.var_names.astype(str))
    if gene not in var_names:
        return None
    vec = adata_t[:, [gene]].X
    if sparse.issparse(vec):
        vec = vec.toarray().ravel()
    else:
        vec = np.asarray(vec).ravel()
    vec = vec.astype(float)
    if x_space == "log1p":
        vec = np.expm1(vec)
    return vec


def _combine_subunit_vectors(vectors: List[np.ndarray], mode: str) -> np.ndarray:
    arr = np.stack(vectors, axis=0)
    if mode == "min":
        return arr.min(axis=0)
    if mode == "mean":
        return arr.mean(axis=0)
    if mode == "product":
        return arr.prod(axis=0)
    raise ValueError(f"Unsupported complex mode: {mode}")


def _token_expression_vector(
    adata_t,
    token: str,
    x_space: str,
    complex_mode: str,
) -> Tuple[np.ndarray, List[str]]:
    present: List[np.ndarray] = []
    missing: List[str] = []
    for g in _split_complex(token):
        v = _extract_gene_vector(adata_t, g, x_space=x_space)
        if v is None:
            missing.append(g)
        else:
            present.append(v)
    if not present:
        return np.zeros(adata_t.n_obs, dtype=float), missing
    return _combine_subunit_vectors(present, complex_mode), missing


def _plot_single(
    coords: np.ndarray,
    values: np.ndarray,
    title: str,
    out_path: str,
    cmap: str,
    point_size: float,
    alpha: float,
    invert_y: bool,
    force_01: bool = False,
) -> None:
    if force_01:
        vmin, vmax = 0.0, 1.0
    else:
        vmin, vmax = _robust_range(values, lo_q=1.0, hi_q=99.0)
    fig, ax = plt.subplots(figsize=(6, 7), dpi=300, facecolor="white")
    sc = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=values,
        cmap=cmap,
        s=point_size,
        alpha=alpha,
        edgecolors="none",
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(title, fontsize=12, pad=8)
    ax.set_axis_off()
    if invert_y:
        ax.invert_yaxis()
    cb = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_panel(
    coords: np.ndarray,
    panel_items: List[Tuple[str, np.ndarray, str]],
    out_path: str,
    point_size: float,
    alpha: float,
    invert_y: bool,
    force_01: bool = False,
) -> None:
    n = len(panel_items)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 5.8), dpi=300, facecolor="white")
    if n == 1:
        axes = [axes]

    for ax, (title, values, cmap) in zip(axes, panel_items):
        if force_01:
            vmin, vmax = 0.0, 1.0
        else:
            vmin, vmax = _robust_range(values, lo_q=1.0, hi_q=99.0)
        sc = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=values,
            cmap=cmap,
            s=point_size,
            alpha=alpha,
            edgecolors="none",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(title, fontsize=11, pad=6)
        ax.set_axis_off()
        if invert_y:
            ax.invert_yaxis()
        cb = fig.colorbar(sc, ax=ax, fraction=0.042, pad=0.01)
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=7)

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _rotate_coords(coords: np.ndarray, deg: float) -> np.ndarray:
    if abs(float(deg)) < 1e-12:
        return coords
    rad = np.deg2rad(float(deg))
    c = np.cos(rad)
    s = np.sin(rad)
    rot = np.array([[c, -s], [s, c]], dtype=float)
    ctr = coords.mean(axis=0, keepdims=True)
    centered = coords - ctr
    return centered @ rot.T + ctr


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot LR spatial maps for one timepoint.")
    p.add_argument("--scores-pkl", default=DEFAULT_SCORES_PKL)
    p.add_argument("--adata-h5ad", default=DEFAULT_ADATA_H5AD, help="Real or interpolated h5ad.")
    p.add_argument("--lr-db", default=DEFAULT_LR_DB)
    p.add_argument("--lr-pair", required=True, help="e.g. Ptprm_Ptprm")
    p.add_argument("--time-key", default=None, help="Override time key. For interpolation single-slice h5ad, can omit.")
    p.add_argument("--timepoint-col", default="timepoint")
    p.add_argument("--annotation-col", default="annotation")
    p.add_argument("--spatial-key", default="spatial")
    p.add_argument("--output-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--point-size", type=float, default=5.0)
    p.add_argument("--alpha", type=float, default=0.9)
    p.add_argument("--invert-y", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--rotate-deg", type=float, default=0.0, help="Rotate spatial coordinates by degrees around centroid.")
    p.add_argument("--incoming-cmap", default="viridis", help="Colormap for incoming hotspot.")
    p.add_argument("--outgoing-cmap", default="plasma", help="Colormap for outgoing hotspot.")
    p.add_argument("--total-cmap", default="cividis", help="Colormap for total hotspot (incoming + outgoing).")
    p.add_argument(
        "--expr-normalize-01",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For ligand/receptor/product maps: clip by quantiles then normalize to 0-1.",
    )
    p.add_argument("--expr-q-low", type=float, default=1.0, help="Lower quantile for expression clipping.")
    p.add_argument("--expr-q-high", type=float, default=99.0, help="Upper quantile for expression clipping.")
    p.add_argument("--x-space", default="log1p", choices=["log1p", "count"])
    p.add_argument("--complex-mode", default="min", choices=["min", "mean", "product"])
    p.add_argument("--no-filter-time", action="store_true", help="Use all cells in --adata-h5ad without time filtering.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.scores_pkl, "rb") as f:
        data = pickle.load(f)

    lr_scores: Dict[str, np.ndarray] = data["lr_scores"]
    if args.lr_pair not in lr_scores:
        sample = sorted(lr_scores.keys())[:15]
        raise KeyError(f"LR pair '{args.lr_pair}' not found in {args.scores_pkl}. Sample keys: {sample}")

    lr_mat = np.asarray(lr_scores[args.lr_pair], dtype=float)
    cell_types = [str(x) for x in np.asarray(data["cell_types"]).tolist()]
    if lr_mat.shape != (len(cell_types), len(cell_types)):
        raise ValueError(f"LR matrix shape mismatch: {lr_mat.shape} vs ({len(cell_types)}, {len(cell_types)})")

    time_key = str(args.time_key) if args.time_key is not None else str(data.get("time_key"))
    if not time_key or time_key == "None":
        if not args.no_filter_time:
            raise ValueError("Could not infer time key; pass --time-key or use --no-filter-time.")

    # LR hotspot (from projected LR matrix, mapped back by cell type)
    incoming_by_type = lr_mat.sum(axis=0)  # receiver total
    outgoing_by_type = lr_mat.sum(axis=1)  # sender total
    total_by_type = incoming_by_type + outgoing_by_type
    incoming_norm = _safe_norm(incoming_by_type)
    outgoing_norm = _safe_norm(outgoing_by_type)
    total_norm = _safe_norm(total_by_type)
    ct_to_in = {ct: float(v) for ct, v in zip(cell_types, incoming_norm)}
    ct_to_out = {ct: float(v) for ct, v in zip(cell_types, outgoing_norm)}
    ct_to_total = {ct: float(v) for ct, v in zip(cell_types, total_norm)}

    # Load adata (real or interpolated)
    adata = ad.read_h5ad(args.adata_h5ad, backed="r")
    ann_col = _resolve_column(adata.obs.columns, args.annotation_col, ["annotation", "Annotation", "cell_type"])

    if args.no_filter_time:
        adata_t = adata
        used_time_key = time_key if time_key and time_key != "None" else "allcells"
    else:
        tp_col = _resolve_column(adata.obs.columns, args.timepoint_col, ["timepoint", "samples", "time", "batch"])
        mask = adata.obs[tp_col].astype(str).to_numpy() == time_key
        if not mask.any():
            available = sorted(pd.Series(adata.obs[tp_col].astype(str)).unique().tolist())
            raise KeyError(f"time_key '{time_key}' not in {tp_col}. Available: {available}")
        adata_t = adata[np.flatnonzero(mask), :]
        used_time_key = time_key

    adata_t = adata_t.to_memory() if hasattr(adata_t, "to_memory") else adata_t.copy()
    try:
        if hasattr(adata, "file") and adata.file is not None:
            adata.file.close()
    except Exception:
        pass

    if args.spatial_key not in adata_t.obsm:
        raise KeyError(f"spatial key '{args.spatial_key}' not in adata.obsm")
    coords = np.asarray(adata_t.obsm[args.spatial_key])[:, :2]
    coords = _rotate_coords(coords, deg=args.rotate_deg)
    ann = adata_t.obs[ann_col].astype(str).to_numpy()

    incoming_cell = np.array([ct_to_in.get(a, 0.0) for a in ann], dtype=float)
    outgoing_cell = np.array([ct_to_out.get(a, 0.0) for a in ann], dtype=float)
    total_cell = np.array([ct_to_total.get(a, 0.0) for a in ann], dtype=float)

    # Ligand / receptor / product maps (cell-level expression)
    lr_db = _load_lr_db(args.lr_db) if args.lr_db else None
    ligand_token, receptor_token = _parse_lr_tokens(args.lr_pair, lr_db)
    ligand_expr, ligand_missing = _token_expression_vector(
        adata_t, token=ligand_token, x_space=args.x_space, complex_mode=args.complex_mode
    )
    receptor_expr, receptor_missing = _token_expression_vector(
        adata_t, token=receptor_token, x_space=args.x_space, complex_mode=args.complex_mode
    )
    lr_product = ligand_expr * receptor_expr
    if args.expr_normalize_01:
        ligand_plot = _clip_norm01(ligand_expr, lo_q=args.expr_q_low, hi_q=args.expr_q_high)
        receptor_plot = _clip_norm01(receptor_expr, lo_q=args.expr_q_low, hi_q=args.expr_q_high)
        product_plot = _clip_norm01(lr_product, lo_q=args.expr_q_low, hi_q=args.expr_q_high)
    else:
        ligand_plot = ligand_expr
        receptor_plot = receptor_expr
        product_plot = lr_product

    pair_safe = args.lr_pair.replace("/", "_")
    prefix = f"{pair_safe}_t{used_time_key}"

    paths = {
        "ligand": os.path.join(args.output_dir, f"{prefix}_ligand.pdf"),
        "receptor": os.path.join(args.output_dir, f"{prefix}_receptor.pdf"),
        "product": os.path.join(args.output_dir, f"{prefix}_ligand_x_receptor.pdf"),
        "incoming": os.path.join(args.output_dir, f"{prefix}_incoming.pdf"),
        "outgoing": os.path.join(args.output_dir, f"{prefix}_outgoing.pdf"),
        "total": os.path.join(args.output_dir, f"{prefix}_total.pdf"),
        "panel": os.path.join(args.output_dir, f"{prefix}_panel6.pdf"),
        "panel_legacy": os.path.join(args.output_dir, f"{prefix}_panel5.pdf"),
    }

    _plot_single(
        coords,
        ligand_plot,
        f"Ligand: {ligand_token} ({used_time_key})",
        paths["ligand"],
        "Reds",
        args.point_size,
        args.alpha,
        args.invert_y,
        args.expr_normalize_01,
    )
    _plot_single(
        coords,
        receptor_plot,
        f"Receptor: {receptor_token} ({used_time_key})",
        paths["receptor"],
        "Blues",
        args.point_size,
        args.alpha,
        args.invert_y,
        args.expr_normalize_01,
    )
    _plot_single(
        coords,
        product_plot,
        f"Ligand x Receptor ({used_time_key})",
        paths["product"],
        "magma",
        args.point_size,
        args.alpha,
        args.invert_y,
        args.expr_normalize_01,
    )
    _plot_single(
        coords,
        incoming_cell,
        f"{args.lr_pair} incoming ({used_time_key})",
        paths["incoming"],
        args.incoming_cmap,
        args.point_size,
        args.alpha,
        args.invert_y,
        True,
    )
    _plot_single(
        coords,
        outgoing_cell,
        f"{args.lr_pair} outgoing ({used_time_key})",
        paths["outgoing"],
        args.outgoing_cmap,
        args.point_size,
        args.alpha,
        args.invert_y,
        True,
    )
    _plot_single(
        coords,
        total_cell,
        f"{args.lr_pair} total ({used_time_key})",
        paths["total"],
        args.total_cmap,
        args.point_size,
        args.alpha,
        args.invert_y,
        True,
    )

    panel_items = [
        (f"Ligand\n{ligand_token}", ligand_plot, "Reds"),
        (f"Receptor\n{receptor_token}", receptor_plot, "Blues"),
        ("Ligand x Receptor", product_plot, "magma"),
        ("Incoming hotspot", incoming_cell, args.incoming_cmap),
        ("Outgoing hotspot", outgoing_cell, args.outgoing_cmap),
        ("Total hotspot", total_cell, args.total_cmap),
    ]
    _plot_panel(
        coords=coords,
        panel_items=panel_items,
        out_path=paths["panel"],
        point_size=args.point_size,
        alpha=args.alpha,
        invert_y=args.invert_y,
        force_01=args.expr_normalize_01,
    )
    shutil.copyfile(paths["panel"], paths["panel_legacy"])

    summary = pd.DataFrame(
        {
            "cell_type": cell_types,
            "incoming_raw": incoming_by_type,
            "outgoing_raw": outgoing_by_type,
            "total_raw": total_by_type,
            "incoming_norm": incoming_norm,
            "outgoing_norm": outgoing_norm,
            "total_norm": total_norm,
        }
    ).sort_values("total_raw", ascending=False)
    summary["lr_pair"] = args.lr_pair
    summary["time_key"] = used_time_key
    summary["ligand_token"] = ligand_token
    summary["receptor_token"] = receptor_token
    summary["ligand_missing_subunits"] = ",".join(ligand_missing) if ligand_missing else ""
    summary["receptor_missing_subunits"] = ",".join(receptor_missing) if receptor_missing else ""
    summary_path = os.path.join(args.output_dir, f"{prefix}_type_scores.csv")
    summary.to_csv(summary_path, index=False)

    print("Saved ligand:", paths["ligand"])
    print("Saved receptor:", paths["receptor"])
    print("Saved product:", paths["product"])
    print("Saved incoming:", paths["incoming"])
    print("Saved outgoing:", paths["outgoing"])
    print("Saved total:", paths["total"])
    print("Saved panel:", paths["panel"])
    print("Saved panel (legacy name):", paths["panel_legacy"])
    print("Saved summary:", summary_path)
    if ligand_missing or receptor_missing:
        print("Missing subunits | ligand:", ligand_missing, "| receptor:", receptor_missing)


if __name__ == "__main__":
    main()
