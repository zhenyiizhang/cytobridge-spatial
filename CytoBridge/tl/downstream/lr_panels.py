from __future__ import annotations

import os
import pickle
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from scipy import sparse


@dataclass(frozen=True)
class LRMultipanelSpec:
    label: str
    scores_pkl: str | Path
    h5ad_path: str | Path
    annotation_col: str
    time_key: str | None = None
    no_filter_time: bool = True
    invert_y: bool = False


@dataclass(frozen=True)
class LRPanelResult:
    ligand_pdf: Path
    receptor_pdf: Path
    top_receivers_png: Path
    incoming_multipanel_pdf: Path


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


def _robust_range(values: np.ndarray, lo_q: float = 1.0, hi_q: float = 99.0) -> tuple[float, float]:
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


def _load_lr_db(path: str | Path) -> pd.DataFrame:
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


def _parse_lr_tokens(lr_pair: str, lr_db: pd.DataFrame | None) -> tuple[str, str]:
    if lr_db is not None:
        hit = lr_db[lr_db["lr_pair"] == lr_pair]
        if len(hit) == 1:
            row = hit.iloc[0]
            return str(row["ligand"]), str(row["receptor"])
    if "_" not in lr_pair:
        raise ValueError(f"Invalid lr_pair '{lr_pair}', expected 'Ligand_Receptor'")
    ligand = lr_pair.split("_", 1)[0]
    receptor = lr_pair.split("_", 1)[1]
    return ligand, receptor


def _split_complex(token: str) -> list[str]:
    return [x.strip() for x in str(token).split("_") if x.strip()]


def _extract_gene_vector(adata_t: ad.AnnData, gene: str, x_space: str) -> np.ndarray | None:
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


def _combine_subunit_vectors(vectors: list[np.ndarray], mode: str) -> np.ndarray:
    arr = np.stack(vectors, axis=0)
    if mode == "min":
        return arr.min(axis=0)
    if mode == "mean":
        return arr.mean(axis=0)
    if mode == "product":
        return arr.prod(axis=0)
    raise ValueError(f"Unsupported complex mode: {mode}")


def _token_expression_vector(
    adata_t: ad.AnnData,
    token: str,
    x_space: str,
    complex_mode: str,
) -> tuple[np.ndarray, list[str]]:
    present: list[np.ndarray] = []
    missing: list[str] = []
    for gene in _split_complex(token):
        v = _extract_gene_vector(adata_t, gene, x_space=x_space)
        if v is None:
            missing.append(gene)
        else:
            present.append(v)
    if not present:
        return np.zeros(adata_t.n_obs, dtype=float), missing
    return _combine_subunit_vectors(present, complex_mode), missing


def _hotspot_by_cell_type(scores_pkl: str | Path, lr_pair: str, mode: str) -> dict[str, float]:
    with open(scores_pkl, "rb") as handle:
        data = pickle.load(handle)
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


def _extract_coords(adata_t: ad.AnnData, spatial_key: str) -> np.ndarray:
    if spatial_key in adata_t.obsm:
        return np.asarray(adata_t.obsm[spatial_key])[:, :2]
    obs_cols = list(adata_t.obs.columns)
    low_to_col = {c.lower(): c for c in obs_cols}
    for x_key, y_key in [("spatial_x", "spatial_y"), ("x", "y"), ("x1", "x2")]:
        if x_key in low_to_col and y_key in low_to_col:
            return np.column_stack(
                [adata_t.obs[low_to_col[x_key]].to_numpy(), adata_t.obs[low_to_col[y_key]].to_numpy()]
            ).astype(float)
    raise KeyError(f"spatial key '{spatial_key}' not found in obsm and no fallback obs columns found.")


def render_lr_expression_panels(
    *,
    scores_pkl: str | Path,
    h5ad_path: str | Path,
    lr_pair: str,
    lr_db_path: str | Path,
    out_dir: str | Path,
    x_space: str = "log1p",
    complex_mode: str = "min",
    point_size_bg: float = 4.4,
    point_size_fg: float = 5.4,
    alpha_bg: float = 0.30,
    alpha_fg: float = 1.0,
) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    adata = ad.read_h5ad(h5ad_path)
    coords = np.asarray(adata.obsm["spatial"], dtype=float)
    lr_df = _load_lr_db(lr_db_path)
    ligand_token, receptor_token = _parse_lr_tokens(lr_pair, lr_df)
    ligand_raw, _ = _token_expression_vector(adata, ligand_token, x_space=x_space, complex_mode=complex_mode)
    receptor_raw, _ = _token_expression_vector(adata, receptor_token, x_space=x_space, complex_mode=complex_mode)
    ligand = _clip_norm01(ligand_raw, 10.0, 95.0)
    receptor = _clip_norm01(receptor_raw, 10.0, 95.0)

    soft_reds = LinearSegmentedColormap.from_list(
        "soft_reds_nonwhite", ["#efc0b8", "#e99a8e", "#df7568", "#cf5548", "#b7382f", "#8d1e19"]
    )
    soft_blues = LinearSegmentedColormap.from_list(
        "soft_blues_nonwhite", ["#c6dbef", "#9ecae1", "#6baed6", "#4292c6", "#2171b5", "#084594"]
    )

    outputs: dict[str, Path] = {}
    for name, values, cmap in [("ligand", ligand, soft_reds), ("receptor", receptor, soft_blues)]:
        fig, ax = plt.subplots(figsize=(5.8, 6.6), dpi=400, facecolor="white")
        ax.scatter(coords[:, 0], coords[:, 1], c="#e6ecef", s=point_size_bg, alpha=alpha_bg, edgecolors="none", rasterized=True)
        sc = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=values,
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
            s=point_size_fg,
            alpha=alpha_fg,
            edgecolors="none",
            rasterized=True,
        )
        ax.set_aspect("equal")
        ax.set_axis_off()
        cbar = fig.colorbar(sc, ax=ax, orientation="vertical", fraction=0.038, pad=0.02)
        cbar.outline.set_visible(False)
        cbar.ax.tick_params(labelsize=8)
        pdf = out_dir / f"{lr_pair}_t1.5_{name}.pdf"
        png = out_dir / f"{lr_pair}_t1.5_{name}.png"
        fig.savefig(pdf, bbox_inches="tight", facecolor="white")
        fig.savefig(png, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        outputs[name] = pdf
    return outputs["ligand"], outputs["receptor"]


def render_top_receivers_barplot(
    *,
    type_scores_csv: str | Path,
    out_prefix: str | Path,
    top_n: int = 6,
    score_col: str = "incoming_norm",
    cell_type_col: str = "cell_type",
    color: str = "#6aaeb3",
) -> Path:
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(type_scores_csv).sort_values(score_col, ascending=False).head(top_n)
    sns.set_theme(style="whitegrid", context="paper")
    fig, ax = plt.subplots(figsize=(4.8, 3.2), dpi=260)
    sns.barplot(data=df, y=cell_type_col, x=score_col, color=color, ax=ax)
    ax.set_xlabel("Normalized incoming")
    ax.set_ylabel("")
    fig.tight_layout()
    png = out_prefix.with_suffix(".png")
    pdf = out_prefix.with_suffix(".pdf")
    svg = out_prefix.with_suffix(".svg")
    fig.savefig(png, bbox_inches="tight", facecolor="white", dpi=260)
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    fig.savefig(svg, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png


def _load_multipanel_data(
    *,
    spec: LRMultipanelSpec,
    lr_pair: str,
    hotspot_mode: str,
    timepoint_col: str,
    spatial_key: str,
) -> tuple[np.ndarray, np.ndarray]:
    ct_to_score = _hotspot_by_cell_type(scores_pkl=spec.scores_pkl, lr_pair=lr_pair, mode=hotspot_mode)
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
    coords = _extract_coords(adata_t, spatial_key)
    ann = adata_t.obs[ann_col].astype(str).to_numpy()
    values_raw = np.array([ct_to_score.get(label, 0.0) for label in ann], dtype=float)
    return coords, values_raw


def render_lr_incoming_multipanel(
    *,
    specs: Sequence[LRMultipanelSpec],
    lr_pair: str,
    out_prefix: str | Path,
    hotspot_mode: str = "incoming",
    timepoint_col: str = "time",
    spatial_key: str = "spatial",
    q_low: float = 10.0,
    q_high: float = 95.0,
    point_size_bg: float = 2.4,
    point_size_fg: float = 3.0,
    alpha_bg: float = 0.28,
    alpha_fg: float = 1.0,
) -> Path:
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    panels = [
        (spec.label, *_load_multipanel_data(spec=spec, lr_pair=lr_pair, hotspot_mode=hotspot_mode, timepoint_col=timepoint_col, spatial_key=spatial_key), spec.invert_y)
        for spec in specs
    ]
    vals_norm = []
    for _, _, values_raw, _ in panels:
        vals_norm.append(_clip_norm01(values_raw, q_low, q_high))
    cmap = LinearSegmentedColormap.from_list(
        "incoming_pastel_nonwhite",
        ["#badbdb", "#9fcccc", "#80bbbb", "#5ea7ab", "#3d8f9d", "#236c82"],
    )
    fig, axes = plt.subplots(3, 3, figsize=(13.5, 11.8), dpi=500, facecolor="white")
    axes = axes.flatten()
    sc = None
    for ax, (label, coords, _, invert_y), values in zip(axes, panels, vals_norm):
        ax.scatter(coords[:, 0], coords[:, 1], c="#e3eaed", s=point_size_bg, alpha=alpha_bg, edgecolors="none", rasterized=True)
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=values, cmap=cmap, vmin=0.0, vmax=1.0, s=point_size_fg, alpha=alpha_fg, edgecolors="none", rasterized=True)
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.set_title(label, fontsize=10, pad=5)
        if invert_y:
            ax.invert_yaxis()
    cbar = fig.colorbar(sc, ax=axes, fraction=0.018, pad=0.01)
    cbar.ax.tick_params(labelsize=8)
    cbar.outline.set_visible(False)
    png = out_prefix.with_suffix(".png")
    pdf = out_prefix.with_suffix(".pdf")
    fig.savefig(png, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return pdf


def copy_if_needed(*, source: str | Path, target: str | Path, overwrite: bool = True) -> Path:
    source = Path(source)
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if overwrite or not target.exists():
        shutil.copy2(source, target)
    return target
