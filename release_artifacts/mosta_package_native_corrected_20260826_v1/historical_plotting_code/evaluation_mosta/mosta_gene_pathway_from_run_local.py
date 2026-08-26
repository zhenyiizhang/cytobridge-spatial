#!/usr/bin/env python3
"""MOSTA downstream gene/pathway analysis from an existing focus-anchor run.

This script rebuilds fully generated latent slices from a saved run directory,
backprojects them to gene space, and performs cell-type-specific temporal gene
analyses.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", os.path.join("/tmp", "mosta_mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join("/tmp", "mosta_xdg_cache"))

import anndata as ad
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from scipy.ndimage import gaussian_filter1d

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from evaluation.arista_code.arista_helpers import (  # noqa: E402
    load_config,
    load_models,
    predict_labels_for_trajectories,
    simulate_sde_points_split,
    train_mlp_classifier,
)


DEFAULT_RUN_DIR = "results/mosta_interp_0_3_0208_n_pc_12"
DEFAULT_PCA_COMPONENTS = "mosta_pca_components_with_gene_names.csv"
DEFAULT_CONFIG = "config/mosta_config.yaml"
DEFAULT_GSEA_DB_DIR = "gsea_databases"
DEFAULT_CELL_TYPE = "Brain"
DEFAULT_MARKER_GENES = [
    "Fabp7",
    "Mki67",
    "Sox11",
    "Dcx",
    "Tubb3",
    "Gap43",
    "Stmn2",
    "Map2",
    "Nr2f2",
    "Meis2",
]


@dataclass
class EnrichmentStatus:
    enabled: bool
    reason: str
    db_name: str
    db_path: Optional[str]
    gseapy_available: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MOSTA fully-generated gene trajectory + pathway analysis from an existing run dir."
    )
    parser.add_argument("--run-dir", default=DEFAULT_RUN_DIR)
    parser.add_argument("--cell-type", default=DEFAULT_CELL_TYPE)
    parser.add_argument("--annotation-key", default=None)
    parser.add_argument("--pca-components-csv", default=DEFAULT_PCA_COMPONENTS)
    parser.add_argument("--pca-mean-file", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--gene-h5ad-mode",
        choices=["minimal", "full", "none"],
        default="minimal",
        help=(
            "How much gene-space AnnData to persist. "
            "'minimal' saves only X/obs/var/spatial, "
            "'full' also saves latent_x and reconstruction layers, "
            "'none' skips writing gene h5ad files."
        ),
    )
    parser.add_argument("--top-n-variable", type=int, default=250)
    parser.add_argument("--smooth-sigma", type=float, default=0.5)
    parser.add_argument(
        "--n-clusters",
        default="2",
        help="Pattern cluster count, or 'auto' to choose by silhouette score over a small K range.",
    )
    parser.add_argument("--min-genes-per-cluster", type=int, default=5)
    parser.add_argument(
        "--split-dt-scale",
        type=float,
        default=1.0,
        help="Scale factor applied to the source run split-SDE dt. Use 0.5 to halve the step size.",
    )
    parser.add_argument(
        "--time-grid-scale",
        type=float,
        default=1.0,
        help=(
            "Scale factor applied to the source run time spacing. "
            "Use 0.5 to make the plotted/generated time grid twice as dense "
            "(e.g. 0,0.5,1 -> 0,0.25,0.5,0.75,1)."
        ),
    )
    parser.add_argument("--enrichment-db", default="KEGG")
    parser.add_argument("--marker-gene-file", default=None)
    parser.add_argument(
        "--allow-real-observed-run",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow consuming a run whose run_args used real observed slices.",
    )
    parser.add_argument("--gsea-db-dir", default=DEFAULT_GSEA_DB_DIR)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def _parse_csv_floats(value: Optional[str]) -> List[float]:
    if value is None:
        return []
    return [float(x.strip()) for x in str(value).split(",") if x.strip()]


def _format_time_token(time_value: float) -> str:
    return f"{float(time_value):.3f}".replace(".", "p").replace("-", "n")


def _slugify(text: str) -> str:
    out = []
    for ch in str(text).strip().lower():
        if ch.isalnum():
            out.append(ch)
        else:
            out.append("_")
    slug = "".join(out)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_") or "celltype"


def _build_dense_time_grid(time_points: Sequence[float], scale: float) -> List[float]:
    pts = sorted(float(t) for t in time_points)
    if len(pts) < 2:
        return pts
    if scale <= 0:
        raise ValueError(f"time grid scale must be > 0, got {scale}")
    diffs = np.diff(np.asarray(pts, dtype=np.float64))
    positive = diffs[diffs > 0]
    if positive.size == 0:
        return pts
    base_step = float(np.min(positive))
    new_step = base_step * float(scale)
    if new_step <= 0:
        raise ValueError(f"effective time step must be > 0, got {new_step}")
    start = pts[0]
    end = pts[-1]
    dense = np.arange(start, end + (new_step * 0.5), new_step, dtype=np.float64)
    dense = np.round(dense, 6)
    if abs(dense[-1] - end) > 1e-6:
        dense = np.append(dense, end)
    return [float(x) for x in dense]


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
    idx = pd.Index(genes)
    if not np.all(idx.isin(m.index)):
        miss = idx[~idx.isin(m.index)]
        raise ValueError(f"PCA mean CSV missing genes: {list(miss[:10])} ... total_missing={miss.shape[0]}")
    return m.loc[idx].to_numpy(dtype=np.float32), "gene_keyed"


def _sample_observed_x0(
    df: pd.DataFrame,
    *,
    time_value: float,
    feature_cols: Sequence[str],
    label_col: str,
    n_samples_cap: Optional[int],
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    subset = df[df["samples"] == float(time_value)]
    X = subset[list(feature_cols)].values.astype(np.float32)
    labels = subset[label_col].astype(str).values
    if n_samples_cap is None or X.shape[0] <= int(n_samples_cap):
        return X, labels
    idx = rng.choice(X.shape[0], size=int(n_samples_cap), replace=False)
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
) -> np.ndarray:
    import torch
    from DeepRUOT.interaction import cal_interaction, euler_sdeint_split

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

    sde = SDE(f_net.v_net, f_net.g_net, score_net, f_net.interaction_net, sigma=sigma)
    ts_tensor = torch.tensor(list(ts_points), dtype=torch.float32, device=device)
    sde_points, _ = euler_sdeint_split(sde, initial_state, dt=dt, ts=ts_tensor, noise_std=0.0)
    return np.array([p.detach().cpu().numpy() for p in sde_points], dtype=object)


def _resolve_enrichment_db(db_name: str, gsea_db_dir: str) -> EnrichmentStatus:
    try:
        from gseapy import enrichr  # noqa: F401

        gseapy_available = True
    except Exception:
        gseapy_available = False

    db_dir_abs = gsea_db_dir
    if not os.path.isabs(db_dir_abs):
        db_dir_abs = os.path.join(PROJECT_ROOT, gsea_db_dir)
    db_map = {
        "KEGG": os.path.join(db_dir_abs, "KEGG_2019_Mouse.gmt"),
        "GO": os.path.join(db_dir_abs, "GO_Biological_Process_2021.gmt"),
        "Reactome": os.path.join(db_dir_abs, "Reactome_2016.gmt"),
    }
    db_path = db_map.get(db_name, db_name)
    if db_name in db_map and not os.path.exists(db_path):
        return EnrichmentStatus(
            enabled=False,
            reason=f"database file missing: {db_path}",
            db_name=db_name,
            db_path=db_path,
            gseapy_available=gseapy_available,
        )
    if not gseapy_available:
        return EnrichmentStatus(
            enabled=False,
            reason="gseapy not available",
            db_name=db_name,
            db_path=db_path,
            gseapy_available=False,
        )
    return EnrichmentStatus(
        enabled=True,
        reason="ok",
        db_name=db_name,
        db_path=db_path,
        gseapy_available=True,
    )


def _zscore_rows(matrix: np.ndarray) -> np.ndarray:
    out = np.zeros_like(matrix, dtype=np.float32)
    means = matrix.mean(axis=1, keepdims=True)
    stds = matrix.std(axis=1, keepdims=True)
    mask = stds.squeeze(axis=1) > 0
    out[mask] = ((matrix[mask] - means[mask]) / stds[mask]).astype(np.float32)
    out[~mask] = matrix[~mask].astype(np.float32)
    return out


def _cluster_row_order(matrix: np.ndarray) -> np.ndarray:
    if matrix.shape[0] <= 1:
        return np.arange(matrix.shape[0])
    link = linkage(matrix, method="average", metric="euclidean")
    return leaves_list(link)


def _choose_cluster_labels(
    zscore_full: np.ndarray,
    n_clusters_arg: str,
) -> Tuple[np.ndarray, int, pd.DataFrame]:
    if zscore_full.shape[0] == 1:
        metrics = pd.DataFrame([{"k": 1, "silhouette": np.nan, "n_clusters_found": 1}])
        return np.array([1], dtype=int), 1, metrics

    link = linkage(zscore_full, method="average", metric="euclidean")
    raw_arg = str(n_clusters_arg).strip().lower()
    if raw_arg != "auto":
        chosen = int(raw_arg)
        labels = fcluster(link, chosen, criterion="maxclust")
        metrics = pd.DataFrame([{"k": chosen, "silhouette": np.nan, "n_clusters_found": int(len(np.unique(labels)))}])
        return labels, chosen, metrics

    from sklearn.metrics import silhouette_score

    n_genes = int(zscore_full.shape[0])
    max_k = min(8, max(2, n_genes - 1))
    rows: List[Dict[str, object]] = []
    best_k = None
    best_score = float("-inf")
    best_labels = None
    for k in range(2, max_k + 1):
        labels = fcluster(link, k, criterion="maxclust")
        n_found = int(len(np.unique(labels)))
        score = np.nan
        if n_found >= 2 and n_found < n_genes:
            try:
                score = float(silhouette_score(zscore_full, labels, metric="euclidean"))
            except Exception:
                score = np.nan
        rows.append({"k": k, "silhouette": score, "n_clusters_found": n_found})
        if not np.isnan(score) and score > best_score:
            best_score = score
            best_k = k
            best_labels = labels

    if best_labels is None or best_k is None:
        best_k = min(2, n_genes)
        best_labels = fcluster(link, best_k, criterion="maxclust")
    return np.asarray(best_labels, dtype=int), int(best_k), pd.DataFrame(rows)


def _save_heatmap(
    matrix: np.ndarray,
    row_labels: Sequence[str],
    time_points: Sequence[float],
    title: str,
    color_label: str,
    save_path: str,
    *,
    cmap: str,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    fig_h = max(6.0, 0.12 * len(row_labels))
    fig, ax = plt.subplots(figsize=(10, fig_h), dpi=300)
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, interpolation="nearest", vmin=vmin, vmax=vmax)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=6)
    ax.set_xticks(np.arange(len(time_points)))
    ax.set_xticklabels([f"{t:.1f}" for t in time_points], rotation=45, ha="right")
    ax.set_xlabel("Time")
    ax.set_ylabel("Genes")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, shrink=0.9)
    cbar.set_label(color_label)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_all_gene_trajectories(
    expr_matrix: np.ndarray,
    time_points: Sequence[float],
    save_path: str,
    title: str,
    ylabel: str,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 7), dpi=300)
    for row in expr_matrix:
        ax.plot(time_points, row, color="#3b6ea8", alpha=0.08, linewidth=0.6)
    mean_curve = expr_matrix.mean(axis=0)
    ax.plot(time_points, mean_curve, color="#a32136", linewidth=2.0, label="Mean")
    ax.set_xlabel("Time")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.2, linestyle="--")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _load_marker_genes(path: Optional[str]) -> List[str]:
    if not path:
        return list(DEFAULT_MARKER_GENES)
    marker_df = pd.read_csv(path, sep=None, engine="python")
    if marker_df.shape[1] == 0:
        return list(DEFAULT_MARKER_GENES)
    series = marker_df.iloc[:, 0].astype(str).str.strip()
    return [g for g in series.tolist() if g]


def _extract_expression_matrix(
    gene_slices: Dict[str, ad.AnnData],
    *,
    cell_type: str,
    annotation_key: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int]]:
    time_points: List[float] = []
    expr_data: List[np.ndarray] = []
    cell_counts: List[int] = []
    gene_names: Optional[np.ndarray] = None
    for time_key, adata_t in sorted(gene_slices.items(), key=lambda kv: float(kv[0])):
        labels = adata_t.obs[annotation_key].astype(str).values
        mask = labels == str(cell_type)
        n_cells = int(mask.sum())
        if n_cells == 0:
            continue
        expr = adata_t[mask].X.mean(axis=0)
        if hasattr(expr, "A1"):
            expr = expr.A1
        expr = np.asarray(expr, dtype=np.float32).ravel()
        expr_data.append(expr)
        time_points.append(float(time_key))
        cell_counts.append(n_cells)
        if gene_names is None:
            gene_names = adata_t.var_names.astype(str).to_numpy()
    if not expr_data or gene_names is None:
        raise ValueError(f"No cells found for cell_type={cell_type!r}")
    expr_matrix = np.vstack(expr_data).T
    return expr_matrix, np.asarray(time_points, dtype=np.float32), gene_names, cell_counts


def _run_pathway_enrichment(
    pattern_genes: Dict[int, List[str]],
    *,
    status: EnrichmentStatus,
    top_n: int = 5,
) -> Tuple[Dict[int, pd.DataFrame], List[str]]:
    results: Dict[int, pd.DataFrame] = {}
    notes: List[str] = []
    if not status.enabled:
        notes.append(status.reason)
        for cluster_id in pattern_genes:
            results[cluster_id] = pd.DataFrame(
                columns=["Term", "Adjusted P-value", "P-value", "Genes", "Cluster"]
            )
        return results, notes

    from gseapy import enrichr

    for cluster_id, genes in pattern_genes.items():
        if len(genes) < 2:
            notes.append(f"cluster {cluster_id}: skipped enrichment due to too few genes")
            results[cluster_id] = pd.DataFrame(
                columns=["Term", "Adjusted P-value", "P-value", "Genes", "Cluster"]
            )
            continue
        try:
            enr = enrichr(
                gene_list=genes,
                gene_sets=status.db_path,
                organism="mouse",
                outdir=None,
                cutoff=0.05,
            )
            if enr.results is None or enr.results.empty:
                df_enr = pd.DataFrame(columns=["Term", "Adjusted P-value", "P-value", "Genes"])
            else:
                df_enr = enr.results.copy()
                df_enr = df_enr.sort_values("Adjusted P-value", kind="stable").head(top_n).reset_index(drop=True)
            df_enr["Cluster"] = int(cluster_id)
            results[cluster_id] = df_enr
        except Exception as exc:
            notes.append(f"cluster {cluster_id}: enrichment failed: {exc}")
            results[cluster_id] = pd.DataFrame(
                columns=["Term", "Adjusted P-value", "P-value", "Genes", "Cluster"]
            )
    return results, notes


def _plot_pattern_pathway_summary(
    zscore_matrix_sorted: np.ndarray,
    sorted_gene_labels: Sequence[str],
    sorted_cluster_labels: Sequence[int],
    time_points: Sequence[float],
    pathway_results: Dict[int, pd.DataFrame],
    *,
    cell_type: str,
    save_path: str,
) -> None:
    fig = plt.figure(figsize=(14, max(8, 0.12 * len(sorted_gene_labels))), dpi=300)
    gs = fig.add_gridspec(1, 3, width_ratios=[6, 0.2, 3], wspace=0.2)
    ax_heat = fig.add_subplot(gs[0, 0])
    ax_cbar = fig.add_subplot(gs[0, 1])
    ax_text = fig.add_subplot(gs[0, 2])

    im = ax_heat.imshow(zscore_matrix_sorted, aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2)
    ax_heat.set_xticks(np.arange(len(time_points)))
    ax_heat.set_xticklabels([f"{t:.1f}" for t in time_points], rotation=45, ha="right")
    ax_heat.set_yticks([])
    ax_heat.set_xlabel("Time")
    ax_heat.set_ylabel("Genes")
    ax_heat.set_title(f"{cell_type}: temporal gene patterns")

    if len(sorted_cluster_labels) > 0:
        start = 0
        labels_arr = np.asarray(sorted_cluster_labels)
        for cluster_id in np.unique(labels_arr):
            count = int(np.sum(labels_arr == cluster_id))
            center = start + count / 2.0
            ax_heat.axhline(start - 0.5, color="white", linewidth=1.2)
            ax_heat.text(
                -0.02,
                center,
                f"P{int(cluster_id)}",
                transform=ax_heat.get_yaxis_transform(),
                ha="right",
                va="center",
                fontsize=9,
                fontweight="bold",
            )
            start += count
        ax_heat.axhline(start - 0.5, color="white", linewidth=1.2)

    cbar = fig.colorbar(im, cax=ax_cbar)
    cbar.set_label("Gene-wise z-score")

    ax_text.axis("off")
    y = 0.98
    for cluster_id in sorted(pathway_results):
        df_pw = pathway_results[cluster_id]
        ax_text.text(0.0, y, f"Pattern {cluster_id}", fontsize=10, fontweight="bold", va="top")
        y -= 0.04
        if df_pw.empty:
            ax_text.text(0.02, y, "No enrichment result", fontsize=8, va="top")
            y -= 0.06
            continue
        for _, row in df_pw.head(5).iterrows():
            term = str(row.get("Term", "NA"))
            if len(term) > 42:
                term = term[:39] + "..."
            pval = row.get("Adjusted P-value", np.nan)
            ptxt = "NA" if pd.isna(pval) else f"{float(pval):.2e}"
            ax_text.text(0.02, y, f"- {term} ({ptxt})", fontsize=8, va="top")
            y -= 0.035
        y -= 0.03

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _set_random_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir
    if not os.path.isabs(run_dir):
        run_dir = os.path.join(PROJECT_ROOT, run_dir)
    run_args_path = os.path.join(run_dir, "run_args.json")
    if not os.path.exists(run_args_path):
        raise FileNotFoundError(f"run_args.json not found: {run_args_path}")

    with open(run_args_path, "r", encoding="utf-8") as f:
        run_args = json.load(f)

    if bool(run_args.get("use_real_for_observed", True)) and not args.allow_real_observed_run:
        raise ValueError(
            "This script defaults to fully generated runs only. "
            "Pass --allow-real-observed-run if you really want to consume a real-observed run."
        )

    seed = args.seed if args.seed is not None else run_args.get("random_seed", None)
    _set_random_seed(seed)

    config_path = run_args.get("config", DEFAULT_CONFIG)
    data_csv = run_args.get("data_csv")
    annotation_key = args.annotation_key or run_args.get("annotation_key", "Annotation")
    if data_csv is None:
        raise ValueError("run_args.json missing data_csv")
    if not os.path.isabs(config_path):
        config_path = os.path.join(PROJECT_ROOT, config_path)
    if not os.path.isabs(data_csv):
        data_csv = os.path.join(PROJECT_ROOT, data_csv)
    pca_components_csv = args.pca_components_csv
    if not os.path.isabs(pca_components_csv):
        pca_components_csv = os.path.join(PROJECT_ROOT, pca_components_csv)

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.join(run_dir, f"gene_pathway_{_slugify(args.cell_type)}")
    elif not os.path.isabs(output_dir):
        output_dir = os.path.join(PROJECT_ROOT, output_dir)
    gene_h5ad_dir = os.path.join(output_dir, "gene_h5ad")
    figures_dir = os.path.join(output_dir, "figures")
    tables_dir = os.path.join(output_dir, "tables")
    os.makedirs(gene_h5ad_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    cfg = load_config(config_path)
    dim = int(cfg["data"]["dim"])

    df_raw = pd.read_csv(data_csv, low_memory=False)
    req_cols = ["samples"] + [f"x{i}" for i in range(1, dim + 1)]
    missing = [c for c in req_cols if c not in df_raw.columns]
    if missing:
        raise ValueError(f"Missing required columns in data_csv: {missing}")
    if annotation_key not in df_raw.columns:
        raise ValueError(f"Missing annotation column {annotation_key!r} in {data_csv}")
    df_raw = df_raw.copy()
    df_raw["samples"] = df_raw["samples"].astype(float)
    df_raw[annotation_key] = df_raw[annotation_key].astype(str)
    df_raw = df_raw.sort_values("samples").reset_index(drop=True)

    observed_times = sorted(df_raw["samples"].unique().tolist())
    interp_times = [] if bool(run_args.get("no_interp", False)) else _parse_csv_floats(run_args.get("interp_time_points"))
    interp_times = [float(t) for t in interp_times if float(t) not in observed_times]
    source_ts_points = sorted(set(observed_times + interp_times))
    ts_points = _build_dense_time_grid(source_ts_points, float(args.time_grid_scale))
    if len(ts_points) < 2:
        raise ValueError(f"Need at least two time points, got {ts_points}")

    f_net, score_net, exp_dir, device = load_models(
        cfg,
        exp_name=cfg["exp"]["name"],
        device=None,
        model_tag="model_final",
        score_tag="score_model",
    )

    classifier_feature_dim = max(1, min(int(run_args.get("classifier_n_pcs") or dim), dim))
    feature_cols = ["samples"] + [f"x{i}" for i in range(1, classifier_feature_dim + 1)]
    classifier_cache_dir = run_args.get("classifier_cache_dir")
    if not classifier_cache_dir and bool(run_args.get("classifier_cache", True)):
        classifier_cache_dir = os.path.join(run_dir, "classifier_cache")
    elif classifier_cache_dir and not os.path.isabs(classifier_cache_dir):
        classifier_cache_dir = os.path.join(PROJECT_ROOT, classifier_cache_dir)

    clf_model, label_encoder, classifier_acc = train_mlp_classifier(
        df=df_raw,
        feature_cols=feature_cols,
        label_col=annotation_key,
        hidden_size=int(run_args.get("classifier_hidden", 128)),
        epochs=int(run_args.get("classifier_epochs", 500)),
        cache_dir=classifier_cache_dir,
        cache_tag=run_args.get("classifier_cache_tag"),
        df_source_path=data_csv,
        reuse_if_possible=bool(run_args.get("classifier_cache", True)),
        progress=True,
        device=device,
        best_epoch_metric=str(run_args.get("classifier_best_metric", "accuracy")),
        train_on_full_data=bool(run_args.get("classifier_train_on_full_data", False)),
    )

    t0 = float(min(observed_times))
    start_index = observed_times.index(t0)
    n_samples = int((df_raw["samples"] == t0).sum())
    if run_args.get("sde_n_samples") is not None:
        n_samples = min(n_samples, int(run_args["sde_n_samples"]))

    split_sde_piecewise = bool(run_args.get("split_sde_piecewise", False))
    piecewise_include_end = bool(run_args.get("split_sde_piecewise_include_end", False))
    base_split_dt = float(run_args.get("split_sde_dt", 0.05))
    split_dt = base_split_dt * float(args.split_dt_scale)
    if split_dt <= 0:
        raise ValueError(f"split dt must be > 0, got {split_dt}")
    feature_cols_full = [f"x{i}" for i in range(1, dim + 1)]
    piecewise_real_labels: Dict[float, np.ndarray] = {}

    if split_sde_piecewise:
        rng = np.random.default_rng(0 if seed is None else int(seed))
        x0_by_observed: Dict[float, np.ndarray] = {}
        for t_obs in observed_times:
            x0_t, labels_t = _sample_observed_x0(
                df_raw,
                time_value=float(t_obs),
                feature_cols=feature_cols_full,
                label_col=annotation_key,
                n_samples_cap=n_samples,
                rng=rng,
            )
            x0_by_observed[float(t_obs)] = x0_t
            piecewise_real_labels[float(t_obs)] = labels_t.astype(str)

        points_by_time: Dict[float, np.ndarray] = {float(t): x0_by_observed[float(t)] for t in observed_times}
        for t_start, t_end in zip(observed_times[:-1], observed_times[1:]):
            mids = sorted([t for t in interp_times if float(t_start) < float(t) < float(t_end)])
            if (not mids) and (not piecewise_include_end):
                continue
            seg_ts: List[float] = [float(t_start)] + [float(t) for t in mids]
            if piecewise_include_end:
                seg_ts.append(float(t_end))
            seg_points = _simulate_sde_points_split_from_x0(
                x0=x0_by_observed[float(t_start)],
                f_net=f_net,
                score_net=score_net,
                ts_points=seg_ts,
                dt=split_dt,
                sigma=float(run_args.get("split_sigma", 0.03)),
                growth_alpha=float(run_args.get("split_growth_alpha", 1.0)),
                interaction_m=1024,
                device=device,
            )
            for t_val, pts in zip(seg_ts, seg_points):
                if float(t_val) in set(float(t) for t in observed_times):
                    continue
                points_by_time[float(t_val)] = np.asarray(pts, dtype=np.float32)

        missing_piecewise = [float(t) for t in ts_points if float(t) not in points_by_time]
        if missing_piecewise:
            raise ValueError(f"Piecewise split-SDE missing timepoints: {missing_piecewise}")
        sde_points = np.array([points_by_time[float(t)] for t in ts_points], dtype=object)
    else:
        sde_points = simulate_sde_points_split(
            df=df_raw[req_cols],
            dim=dim,
            f_net=f_net,
            score_net=score_net,
            time_index=start_index,
            n_samples=n_samples,
            ts_points=ts_points,
            dt=split_dt,
            sigma=float(run_args.get("split_sigma", 0.03)),
            growth_alpha=float(run_args.get("split_growth_alpha", 1.0)),
            interaction_m=1024,
            device=device,
            verbose=True,
        )

    predicted_labels_list = predict_labels_for_trajectories(
        sde_points=sde_points,
        ts_points=ts_points,
        model=clf_model,
        label_encoder=label_encoder,
        feature_dim=classifier_feature_dim,
        device=device,
        knn_neighbors=10,
    )

    latent_slices: Dict[str, Dict[str, np.ndarray]] = {}
    for idx, t in enumerate(ts_points):
        labels_t = np.asarray(predicted_labels_list[idx]).astype(str)
        if split_sde_piecewise and float(t) in piecewise_real_labels:
            labels_t = piecewise_real_labels[float(t)]
        X_t = np.asarray(sde_points[idx], dtype=np.float32)
        latent_slices[str(float(t))] = {
            "X_latent": X_t,
            "labels": labels_t,
            "spatial": X_t[:, :2].astype(np.float32),
        }

    comp_df = pd.read_csv(pca_components_csv)
    if "gene_short_name" not in comp_df.columns:
        raise ValueError(f"'gene_short_name' missing in {pca_components_csv}")
    pc_cols = [f"PC{i}" for i in range(1, 51)]
    miss_pc = [c for c in pc_cols if c not in comp_df.columns]
    if miss_pc:
        raise ValueError(f"PC columns missing in {pca_components_csv}: {miss_pc}")
    genes = comp_df["gene_short_name"].astype(str).to_numpy()
    loadings = comp_df[pc_cols].to_numpy(dtype=np.float32)
    pca_mean, pca_mean_mode = _load_optional_pca_mean(args.pca_mean_file, genes)

    gene_slices: Dict[str, ad.AnnData] = {}
    for time_key, latent in sorted(latent_slices.items(), key=lambda kv: float(kv[0])):
        x_t = np.asarray(latent["X_latent"], dtype=np.float32)
        pc_scores = x_t[:, 2:52].astype(np.float32)
        if pc_scores.shape[1] != loadings.shape[1]:
            raise ValueError(f"PC mismatch for time {time_key}: {pc_scores.shape} vs {loadings.shape}")
        gene_centered = pc_scores @ loadings.T
        gene_recon = gene_centered if pca_mean is None else gene_centered + pca_mean[None, :]
        gene_log1p = gene_recon.astype(np.float32)
        gene_count = np.clip(np.expm1(gene_log1p), 0.0, None).astype(np.float32)

        adata_t = ad.AnnData(X=gene_log1p.astype(np.float32))
        adata_t.var_names = genes
        adata_t.obs[annotation_key] = np.asarray(latent["labels"]).astype(str)
        adata_t.obs["samples"] = float(time_key)
        adata_t.obs["timepoint"] = str(time_key)
        adata_t.obsm["spatial"] = np.asarray(latent["spatial"], dtype=np.float32)
        adata_t.uns["reconstruction_info"] = {
            "source_run_dir": run_dir,
            "time_key": float(time_key),
            "output_x_space": "log1p",
            "pca_mean_file": args.pca_mean_file or "",
            "pca_mean_mode": pca_mean_mode,
        }
        if args.gene_h5ad_mode == "full":
            adata_t.obsm["latent_x"] = x_t.astype(np.float32)
            adata_t.layers["recon_log1p"] = gene_log1p
            adata_t.layers["recon_count"] = gene_count
            adata_t.layers["centered_recon"] = gene_centered.astype(np.float32)
            adata_t.layers["recon_raw"] = gene_recon.astype(np.float32)
        if args.gene_h5ad_mode != "none":
            out_h5ad = os.path.join(gene_h5ad_dir, f"adata_t{_format_time_token(float(time_key))}_with_genes.h5ad")
            adata_t.write_h5ad(out_h5ad, compression="gzip")
        gene_slices[str(float(time_key))] = adata_t

    expr_matrix, time_points, gene_names, cell_counts = _extract_expression_matrix(
        gene_slices,
        cell_type=args.cell_type,
        annotation_key=annotation_key,
    )
    if expr_matrix.shape[1] < 3:
        raise ValueError(
            f"{args.cell_type!r} appears in fewer than 3 timepoints ({expr_matrix.shape[1]}), "
            "cannot do temporal downstream analysis."
        )

    cell_slug = _slugify(args.cell_type)
    expr_df = pd.DataFrame(expr_matrix, index=gene_names, columns=[f"{t:.3f}" for t in time_points])
    expr_df.index.name = "gene"
    expr_path = os.path.join(tables_dir, f"{cell_slug}_expression_matrix.csv")
    expr_df.to_csv(expr_path)

    _plot_all_gene_trajectories(
        expr_matrix,
        time_points,
        os.path.join(figures_dir, f"{cell_slug}_all_genes_mean_trajectory.pdf"),
        title=f"{args.cell_type}: all-gene mean trajectories",
        ylabel="Mean reconstructed log1p expression",
    )
    zscore_expr = _zscore_rows(expr_matrix)
    _plot_all_gene_trajectories(
        zscore_expr,
        time_points,
        os.path.join(figures_dir, f"{cell_slug}_all_genes_zscore_trajectory.pdf"),
        title=f"{args.cell_type}: all-gene z-score trajectories",
        ylabel="Gene-wise z-score",
    )

    top_n = min(int(args.top_n_variable), expr_matrix.shape[0])
    gene_vars = np.var(expr_matrix, axis=1)
    top_idx = np.argsort(gene_vars)[-top_n:]
    top_matrix = expr_matrix[top_idx]
    top_genes = gene_names[top_idx]
    top_z = _zscore_rows(top_matrix)
    top_order = _cluster_row_order(top_z)
    top_mean_sorted = top_matrix[top_order]
    top_z_sorted = top_z[top_order]
    top_genes_sorted = top_genes[top_order]
    pd.DataFrame(
        {
            "gene": top_genes,
            "variance": gene_vars[top_idx],
        }
    ).sort_values("variance", ascending=False).to_csv(
        os.path.join(tables_dir, f"{cell_slug}_top{top_n}_variable_genes.csv"),
        index=False,
    )
    _save_heatmap(
        top_mean_sorted,
        top_genes_sorted,
        time_points,
        title=f"{args.cell_type}: top {top_n} variable genes (mean)",
        color_label="Mean reconstructed log1p",
        save_path=os.path.join(figures_dir, f"{cell_slug}_top{top_n}_variable_heatmap_mean.pdf"),
        cmap="OrRd",
        vmin=np.percentile(top_mean_sorted, 5),
        vmax=np.percentile(top_mean_sorted, 95),
    )
    _save_heatmap(
        top_z_sorted,
        top_genes_sorted,
        time_points,
        title=f"{args.cell_type}: top {top_n} variable genes (z-score)",
        color_label="Gene-wise z-score",
        save_path=os.path.join(figures_dir, f"{cell_slug}_top{top_n}_variable_heatmap_zscore.pdf"),
        cmap="RdBu_r",
        vmin=-2,
        vmax=2,
    )

    marker_genes = _load_marker_genes(args.marker_gene_file)
    marker_present = [g for g in marker_genes if g in set(gene_names)]
    marker_missing = [g for g in marker_genes if g not in set(gene_names)]
    if not marker_present:
        raise ValueError("No marker genes were found in reconstructed gene space.")
    marker_idx = np.array([np.where(gene_names == g)[0][0] for g in marker_present], dtype=int)
    marker_mean = expr_matrix[marker_idx]
    marker_z = _zscore_rows(marker_mean)
    pd.DataFrame({"gene": marker_present}).to_csv(
        os.path.join(tables_dir, f"{cell_slug}_marker_genes_present.csv"),
        index=False,
    )
    _save_heatmap(
        marker_mean,
        marker_present,
        time_points,
        title=f"{args.cell_type}: marker panel (mean)",
        color_label="Mean reconstructed log1p",
        save_path=os.path.join(figures_dir, f"{cell_slug}_marker_panel_heatmap_mean.pdf"),
        cmap="OrRd",
        vmin=np.percentile(marker_mean, 5),
        vmax=np.percentile(marker_mean, 95),
    )
    _save_heatmap(
        marker_z,
        marker_present,
        time_points,
        title=f"{args.cell_type}: marker panel (z-score)",
        color_label="Gene-wise z-score",
        save_path=os.path.join(figures_dir, f"{cell_slug}_marker_panel_heatmap_zscore.pdf"),
        cmap="RdBu_r",
        vmin=-2,
        vmax=2,
    )
    fig, ax = plt.subplots(figsize=(12, 7), dpi=300)
    colors = plt.cm.tab10(np.linspace(0, 1, len(marker_present)))
    for color, gene_name, row in zip(colors, marker_present, marker_mean):
        ax.plot(time_points, row, color=color, linewidth=2.0, alpha=0.9, label=gene_name)
    ax.set_xlabel("Time")
    ax.set_ylabel("Mean reconstructed log1p")
    ax.set_title(f"{args.cell_type}: marker trajectories")
    ax.grid(True, alpha=0.2, linestyle="--")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, f"{cell_slug}_marker_panel_lines.pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)

    zscore_full = _zscore_rows(expr_matrix)
    cluster_labels, chosen_n_clusters, cluster_metrics_df = _choose_cluster_labels(zscore_full, args.n_clusters)
    row_order = _cluster_row_order(zscore_full)
    assign_df = pd.DataFrame({"gene": gene_names, "cluster": cluster_labels})
    assign_df.to_csv(os.path.join(tables_dir, f"{cell_slug}_pattern_cluster_assignments.csv"), index=False)
    cluster_metrics_df.to_csv(os.path.join(tables_dir, f"{cell_slug}_cluster_selection_metrics.csv"), index=False)

    pattern_genes: Dict[int, List[str]] = {}
    pattern_curves_records: List[Dict[str, object]] = []
    for cluster_id in sorted(np.unique(cluster_labels)):
        genes_cluster = assign_df.loc[assign_df["cluster"] == cluster_id, "gene"].tolist()
        if len(genes_cluster) >= int(args.min_genes_per_cluster):
            pattern_genes[int(cluster_id)] = genes_cluster
        pd.DataFrame({"gene": genes_cluster}).to_csv(
            os.path.join(tables_dir, f"{cell_slug}_pattern_cluster_{int(cluster_id)}_genes.csv"),
            index=False,
        )
        idx_cluster = np.where(cluster_labels == cluster_id)[0]
        cluster_z = zscore_full[idx_cluster]
        mean_curve = cluster_z.mean(axis=0)
        std_curve = cluster_z.std(axis=0)
        for t, mean_value, std_value in zip(time_points, mean_curve, std_curve):
            pattern_curves_records.append(
                {
                    "cluster": int(cluster_id),
                    "time": float(t),
                    "mean": float(mean_value),
                    "std": float(std_value),
                    "n_genes": int(len(idx_cluster)),
                }
            )
    pd.DataFrame(pattern_curves_records).to_csv(
        os.path.join(tables_dir, f"{cell_slug}_pattern_curves.csv"),
        index=False,
    )

    enrichment_status = _resolve_enrichment_db(args.enrichment_db, args.gsea_db_dir)
    pathway_results, enrichment_notes = _run_pathway_enrichment(
        pattern_genes,
        status=enrichment_status,
        top_n=5,
    )
    for cluster_id in sorted(np.unique(cluster_labels)):
        df_pw = pathway_results.get(int(cluster_id))
        if df_pw is None:
            df_pw = pd.DataFrame(columns=["Term", "Adjusted P-value", "P-value", "Genes", "Cluster"])
        df_pw.to_csv(
            os.path.join(tables_dir, f"{cell_slug}_pathway_cluster_{int(cluster_id)}.csv"),
            index=False,
        )

    sorted_zscore = zscore_full[row_order]
    sorted_gene_labels = gene_names[row_order]
    sorted_cluster_labels = cluster_labels[row_order]
    _plot_pattern_pathway_summary(
        sorted_zscore,
        sorted_gene_labels,
        sorted_cluster_labels,
        time_points,
        pathway_results,
        cell_type=args.cell_type,
        save_path=os.path.join(figures_dir, f"{cell_slug}_pattern_pathway_summary.pdf"),
    )

    manifest = {
        "run_dir": run_dir,
        "run_args_path": run_args_path,
        "config_path": config_path,
        "data_csv": data_csv,
        "annotation_key": annotation_key,
        "cell_type": args.cell_type,
        "seed": seed,
        "device": device,
        "exp_dir": exp_dir,
        "classifier_acc": float(classifier_acc),
        "time_points": [float(t) for t in ts_points],
        "source_time_points": [float(t) for t in source_ts_points],
        "time_points_used_for_expression": [float(t) for t in time_points],
        "n_samples": int(n_samples),
        "n_genes_total": int(len(gene_names)),
        "n_timepoints_expression": int(expr_matrix.shape[1]),
        "cell_counts": [int(x) for x in cell_counts],
        "top_n_variable": int(top_n),
        "n_clusters_requested": args.n_clusters,
        "n_clusters": int(chosen_n_clusters),
        "min_genes_per_cluster": int(args.min_genes_per_cluster),
        "pca_components_csv": pca_components_csv,
        "pca_mean_file": args.pca_mean_file or "",
        "pca_mean_mode": pca_mean_mode,
        "split_sde_piecewise": split_sde_piecewise,
        "split_sde_piecewise_include_end": piecewise_include_end,
        "split_dt_base": base_split_dt,
        "split_dt_scale": float(args.split_dt_scale),
        "split_dt_effective": split_dt,
        "time_grid_scale": float(args.time_grid_scale),
        "use_real_for_observed": bool(run_args.get("use_real_for_observed", True)),
        "enrichment_status": asdict(enrichment_status),
        "enrichment_notes": enrichment_notes,
        "marker_genes_present": marker_present,
        "marker_genes_missing": marker_missing,
        "output_dir": output_dir,
        "gene_h5ad_mode": args.gene_h5ad_mode,
        "gene_h5ad_dir": gene_h5ad_dir,
        "figures_dir": figures_dir,
        "tables_dir": tables_dir,
    }
    manifest_path = os.path.join(output_dir, "analysis_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("Saved gene-space h5ad to:", gene_h5ad_dir)
    print("Saved figures to:", figures_dir)
    print("Saved tables to:", tables_dir)
    print("Saved manifest to:", manifest_path)


if __name__ == "__main__":
    main()
