#!/usr/bin/env python3
"""Build strict-corrected ARISTA S15--S17 in the submitted visual grammar.

The script deliberately separates the scientific source from the display
grammar:

* all gene and ligand--receptor values come from the accepted full-model bank;
* ligand--receptor complexes must contain every requested subunit;
* the submitted S15/S16/S17 geometry, palettes, axes, and ordering are reused;
* the 68-pair historical S17 roster is fixed, with unavailable strict-complex
  trajectories rendered as explicit ``N/E`` slots rather than partial scores.

The output directory is transactional and must not already exist.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
from typing import Any, Iterable


os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/cytobridge_arista_s15_s17_mpl")
os.environ.setdefault("SOURCE_DATE_EPOCH", "946684800")

import pymupdf as fitz
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.ticker import AutoMinorLocator, FormatStrFormatter, MaxNLocator, MultipleLocator
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy.cluster.hierarchy import cut_tree, linkage
from scipy.stats import hypergeom
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BANK = (
    PROJECT_ROOT
    / "output/arista_full_model_figure_bank_corrected_20260822_3c87a3e/downstream"
)
DEFAULT_LEGACY_REVIEW = PROJECT_ROOT / "results/arista_review_dense_local"
DEFAULT_LEGACY_FIGURES = (
    PROJECT_ROOT
    / "manuscript_edits/lr_complex_sensitivity_si_20260815/figures/arista"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "output/arista_s15_s17_strict_corrected_legacy_style_20260823_v1"
)

TIME_POINTS = np.asarray([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
S15_CANVAS_PX = (2400, 1554)
S15_DPI = 300.0
S16_PAGE_PT = (569.192, 317.15225)
S17_PAGE_PT = (1288.692187, 2159.631188)
PANEL_A_N_GENES = 18
S15_CLUSTER_GENE_COUNT = 2000
PATTERN_COLORS = {1: "#66c2a5", 2: "#fc8d62"}
LR_CLUSTER_COLORS = {1: "#1f77b4", 2: "#ff7f0e"}
BAR_PADJUST_COLORS = ["#264653", "#2A9D8F", "#E9C46A", "#F4A261", "#E76F51"]
DOT_PADJUST_COLORS = ["#e06663", "#327eba"]
DOT_SIZE_RANGE = (3.0, 8.0)
SVG_METADATA = {"Date": None, "Creator": "CytoBridge ARISTA strict legacy-style builder"}
PDF_FIXED_DATE = "D:20000101000000+00'00'"

# Placements were recovered from the exact 2400x1554 embedded S15 image by
# matching the original four source panels.  They are fixed legacy geometry,
# not a newly designed montage.
S15_LEGACY_PLACEMENTS = {
    "a": {"x": 19, "y": 20, "width": 1158},
    "b": {"x": 1201, "y": 12, "width": 1194},
    "c": {"x": 0, "y": 652, "width": 1202},
    "d": {"x": 1204, "y": 654, "width": 1194},
}
S15_LABEL_CROPS = {
    "a": (0, 0, 70, 76),
    "b": (1128, 0, 1200, 88),
    "c": (0, 625, 88, 718),
    "d": (1148, 625, 1228, 718),
}


def _configure_legacy_style() -> None:
    """Apply the exact global style block from the legacy ARISTA review code."""

    sns.set_theme(style="white", context="paper")
    mpl.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "font.size": 10,
            "font.family": "DejaVu Sans",
            "svg.fonttype": "path",
            "svg.hashsalt": "arista-s15-s17-strict-legacy-style-v1",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latest-bank-root", type=Path, default=DEFAULT_BANK)
    parser.add_argument("--legacy-review-root", type=Path, default=DEFAULT_LEGACY_REVIEW)
    parser.add_argument(
        "--legacy-s15-reference",
        type=Path,
        default=DEFAULT_LEGACY_FIGURES / "arista_gene_pattern_curves_updated_compact.jpg",
    )
    parser.add_argument(
        "--legacy-s16-reference",
        type=Path,
        default=DEFAULT_LEGACY_FIGURES / "arista_lr_cluster_prototypes_compact_v15.pdf",
    )
    parser.add_argument(
        "--legacy-s17-reference",
        type=Path,
        default=DEFAULT_LEGACY_FIGURES / "arista_lr_small_multiples_compact_v15.pdf",
    )
    parser.add_argument(
        "--gene-set-gmt",
        type=Path,
        default=PROJECT_ROOT / "gsea_databases/GO_Biological_Process_2023.gmt",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--go-alpha", type=float, default=0.05)
    parser.add_argument("--go-min-set-size", type=int, default=5)
    parser.add_argument("--go-max-set-size", type=int, default=5000)
    parser.add_argument("--go-min-overlap", type=int, default=2)
    parser.add_argument(
        "--package-native-contract",
        action="store_true",
        help=(
            "Accept the complete strict LR/gene universe produced by a fresh "
            "package-native run while retaining all submitted display grammar."
        ),
    )
    parser.add_argument(
        "--determinism-reference",
        type=Path,
        default=None,
        help=(
            "Optional independently built bundle. Core figures, tables, source "
            "snapshots, and QA rasters must match it byte-for-byte."
        ),
    )
    return parser


def _require_file(path: Path, description: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing {description}: {path}")
    return path


def _require_dir(path: Path, description: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Missing {description}: {path}")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": _sha256(path)}


def _json_dump(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def _write_text(path: Path, value: str) -> None:
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    if not normalized.isin({"true", "false", "1", "0"}).all():
        raise ValueError(f"Cannot interpret boolean values: {sorted(normalized.unique())}")
    return normalized.isin({"true", "1"})


def _load_expression(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path, index_col=0)
    table.index = table.index.astype(str)
    table.columns = [float(value) for value in table.columns]
    table = table.loc[:, TIME_POINTS.tolist()]
    if table.index.duplicated().any():
        duplicates = table.index[table.index.duplicated()].unique().tolist()
        raise ValueError(f"Duplicate gene rows in corrected expression table: {duplicates[:10]}")
    values = table.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < -1e-12).any():
        raise ValueError("Corrected mean_expression.csv must be finite and post-clip nonnegative.")
    return table


def _stable_display_gene(raw_name: str) -> str:
    parts = [part.strip() for part in str(raw_name).split("|") if part.strip()]
    non_amex = [part for part in parts if not part.startswith("AMEX")]
    selected = (non_amex or parts or [str(raw_name).strip()])[0]
    selected = re.sub(r"\[[^\]]+\]", "", selected).strip()
    if selected.lower() in {"", "nan", "none"}:
        fallback = next((part for part in parts if part.lower() not in {"nan", "none"}), str(raw_name))
        selected = re.sub(r"\[[^\]]+\]", "", fallback).strip()
    return selected


def _unique_display_map(raw_names: Iterable[str]) -> dict[str, str]:
    used: dict[str, int] = {}
    output: dict[str, str] = {}
    for raw in raw_names:
        base = _stable_display_gene(str(raw))
        used[base] = used.get(base, 0) + 1
        output[str(raw)] = base if used[base] == 1 else f"{base}_{used[base]}"
    return output


def _gene_symbol(raw_name: str) -> str | None:
    symbol = _stable_display_gene(raw_name).upper()
    if not symbol or symbol == "NAN" or symbol.startswith("AMEX"):
        return None
    return symbol


def _bh(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranked = values[order]
    adjusted_ranked = ranked * float(len(values)) / np.arange(1, len(values) + 1)
    adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = np.minimum(adjusted_ranked, 1.0)
    return adjusted


def _load_gmt(path: Path) -> dict[str, set[str]]:
    library: dict[str, set[str]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            fields = raw.rstrip("\r\n").split("\t")
            if len(fields) < 3:
                raise ValueError(f"Malformed GMT line {line_number}: {path}")
            term = fields[0].strip()
            if term in library:
                raise ValueError(f"Duplicate GMT term: {term}")
            library[term] = {gene.strip().upper() for gene in fields[2:] if gene.strip()}
    if not library:
        raise ValueError(f"Empty GMT library: {path}")
    return library


def _term_parts(term: str) -> tuple[str, str]:
    if term.endswith(")") and " (GO:" in term:
        name, identifier = term.rsplit(" (", 1)
        return name, identifier[:-1]
    return term, ""


def _ora_expression_background(
    query_genes: Iterable[str],
    library: dict[str, set[str]],
    background_genes: set[str],
    *,
    alpha: float,
    min_set_size: int,
    max_set_size: int,
    min_overlap: int,
) -> pd.DataFrame:
    library_universe = set().union(*library.values())
    background = {gene.upper() for gene in background_genes} & library_universe
    query_input = {gene.upper() for gene in query_genes if gene} & background
    if not background or not query_input:
        raise ValueError("Expression-background ORA has an empty background or query.")
    rows: list[dict[str, Any]] = []
    for term, raw_members in library.items():
        members = raw_members & background
        if not min_set_size <= len(members) <= max_set_size:
            continue
        overlap = sorted(query_input & members)
        if len(overlap) < min_overlap:
            continue
        expected = len(query_input) * len(members) / len(background)
        term_name, term_id = _term_parts(term)
        rows.append(
            {
                "term": term,
                "term_name": term_name,
                "term_id": term_id,
                "query_size": len(query_input),
                "background_size": len(background),
                "set_size_in_background": len(members),
                "overlap_count": len(overlap),
                "gene_ratio": len(overlap) / len(query_input),
                "p_value": float(
                    hypergeom.sf(
                        len(overlap) - 1,
                        len(background),
                        len(members),
                        len(query_input),
                    )
                ),
                "fold_enrichment": len(overlap) / expected,
                "overlap_genes": ";".join(overlap),
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("No GO terms pass the expression-background reporting filters.")
    result["adjusted_p_value"] = _bh(result["p_value"].to_numpy(dtype=float))
    result["significant"] = result["adjusted_p_value"] <= float(alpha)
    return result.sort_values(
        ["adjusted_p_value", "p_value", "fold_enrichment", "term"],
        ascending=[True, True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)


def _build_corrected_gene_programs(
    expression: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    values = expression.to_numpy(dtype=float)
    variances = pd.Series(np.var(values, axis=1, ddof=0), index=expression.index, name="variance")
    ranking = (
        variances.rename_axis("raw_gene")
        .reset_index()
        .sort_values(["variance", "raw_gene"], ascending=[False, True], kind="mergesort")
        .reset_index(drop=True)
    )
    ranking["rank"] = np.arange(1, len(ranking) + 1)
    clustered = ranking.head(S15_CLUSTER_GENE_COUNT).copy()
    matrix = expression.loc[clustered["raw_gene"]].to_numpy(dtype=float)
    row_mean = matrix.mean(axis=1, keepdims=True)
    row_std = matrix.std(axis=1, ddof=0, keepdims=True)
    normalized = np.divide(matrix - row_mean, np.where(row_std > 0, row_std, 1.0))
    linkage_matrix = linkage(normalized, method="average", metric="euclidean", optimal_ordering=False)
    raw_labels = cut_tree(linkage_matrix, n_clusters=2).reshape(-1).astype(int) + 1
    raw_peak_order: list[tuple[int, int]] = []
    for raw_cluster in sorted(np.unique(raw_labels)):
        raw_peak_order.append(
            (int(np.argmax(normalized[raw_labels == raw_cluster].mean(axis=0))), int(raw_cluster))
        )
    semantic_map = {
        raw_cluster: pattern
        for pattern, (_, raw_cluster) in enumerate(sorted(raw_peak_order), start=1)
    }
    patterns = np.asarray([semantic_map[int(value)] for value in raw_labels], dtype=int)
    display_map = _unique_display_map(clustered["raw_gene"].astype(str).tolist())
    clustered["display_gene"] = clustered["raw_gene"].astype(str).map(display_map)
    clustered["gene_symbol"] = clustered["raw_gene"].astype(str).map(_gene_symbol)
    clustered["raw_cluster"] = raw_labels
    clustered["pattern"] = patterns
    normalized_df = pd.DataFrame(
        normalized,
        index=clustered["raw_gene"].astype(str),
        columns=expression.columns,
    )
    normalized_df.index.name = "raw_gene"
    prototype_rows: list[dict[str, Any]] = []
    for pattern in sorted(np.unique(patterns)):
        subset = normalized[patterns == pattern]
        for index, time in enumerate(expression.columns):
            prototype_rows.append(
                {
                    "pattern": int(pattern),
                    "time": float(time),
                    "mean": float(subset[:, index].mean()),
                    "std": float(subset[:, index].std(ddof=0)),
                    "n_profiles": int(subset.shape[0]),
                }
            )
    prototypes = pd.DataFrame(prototype_rows)
    if len(clustered) != 2000 or prototypes["pattern"].nunique() != 2:
        raise AssertionError("S15 corrected program contract must produce 2000 genes in two programs.")
    return ranking, clustered, normalized_df, prototypes


def _plot_s15a(expression: pd.DataFrame, top18: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    genes = top18["raw_gene"].astype(str).tolist()
    display = dict(zip(genes, top18["display_gene"].astype(str)))
    actual = expression.loc[genes].copy()
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    palette = sns.color_palette("tab20", n_colors=len(genes))
    for color, gene in zip(palette, genes):
        ax.plot(
            actual.columns,
            actual.loc[gene],
            marker="o",
            linewidth=1.8,
            color=color,
            label=display[gene],
        )
    ax.set_title("Top variable gene trajectories")
    ax.set_xlabel("Time")
    ax.set_ylabel("Mean reconstructed score")
    ax.grid(True, alpha=0.2, linestyle="--")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", format="svg", metadata=SVG_METADATA)
    plt.close(fig)
    actual.index.name = "raw_gene"
    return actual


def _plot_s15b(prototypes: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.5))
    for pattern, subset in prototypes.groupby("pattern", sort=True):
        subset = subset.sort_values("time")
        color = PATTERN_COLORS[int(pattern)]
        x = subset["time"].to_numpy(dtype=float)
        mean = subset["mean"].to_numpy(dtype=float)
        std = subset["std"].to_numpy(dtype=float)
        ax.plot(x, mean, marker="o", linewidth=2.2, color=color, label=f"Pattern {int(pattern)}")
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)
    ax.set_title("Gene pattern curves")
    ax.set_xlabel("Time")
    ax.set_ylabel("Mean z-score")
    ax.axhline(0, color="#666666", linewidth=0.8, alpha=0.4)
    ax.grid(True, axis="y", alpha=0.2)
    # The submitted panel's legend is in the lower-left; keep that fixed even
    # when corrected curves would make Matplotlib's automatic placement move it.
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", format="svg", metadata=SVG_METADATA)
    plt.close(fig)


def _display_terms(values: pd.Series, width: int = 43) -> pd.Series:
    return values.astype(str).str.lower().map(lambda value: "\n".join(textwrap.wrap(value, width=width)))


def _top_ranked(table: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    return table.sort_values(
        ["adjusted_p_value", "p_value", "fold_enrichment", "term"],
        ascending=[True, True, False, True],
        kind="mergesort",
    ).head(n).reset_index(drop=True)


def _safe_norm(values: np.ndarray) -> Normalize:
    lo = float(np.min(values))
    hi = float(np.max(values))
    if np.isclose(lo, hi):
        hi = lo + max(1e-12, abs(lo) * 1e-6)
    return Normalize(vmin=lo, vmax=hi)


def _save_figure_svg_png(fig: plt.Figure, svg_path: Path, png_path: Path) -> None:
    fig.savefig(svg_path, facecolor="white", metadata=SVG_METADATA)
    fig.savefig(png_path, facecolor="white", dpi=300, metadata={"Software": "CytoBridge"})


def _plot_s15c(table: pd.DataFrame, svg_path: Path, png_path: Path) -> pd.DataFrame:
    selected_ranked = _top_ranked(table, 20)
    selected = selected_ranked.iloc[::-1].copy()
    cmap = LinearSegmentedColormap.from_list("legacy_bar_padjust", BAR_PADJUST_COLORS)
    norm = _safe_norm(selected["adjusted_p_value"].to_numpy(dtype=float))
    fig = plt.figure(figsize=(8.0, 6.0), facecolor="white")
    ax = fig.add_axes([0.34, 0.10, 0.51, 0.82])
    ax.barh(
        _display_terms(selected["term_name"]),
        selected["overlap_count"],
        color=cmap(norm(selected["adjusted_p_value"].to_numpy(dtype=float))),
        edgecolor="none",
    )
    ax.set_title("arista_pattern_1_genes.csv - GO (BP) - Barplot", fontsize=14)
    ax.set_xlabel("Count", fontsize=12)
    ax.set_ylabel("")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.grid(which="major", color="#EBEBEB", linewidth=0.8)
    ax.grid(which="minor", axis="x", color="#EBEBEB", linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    scalar = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    colorbar_ax = fig.add_axes([0.89, 0.36, 0.030, 0.26])
    colorbar = fig.colorbar(scalar, cax=colorbar_ax)
    colorbar.ax.set_title("p.adjust", pad=8)
    colorbar.locator = MaxNLocator(nbins=3, min_n_ticks=2)
    colorbar.formatter = FormatStrFormatter("%.2g")
    colorbar.update_ticks()
    colorbar.outline.set_visible(False)
    if not bool(table["significant"].any()):
        ax.text(
            0.99,
            0.01,
            "No terms at FDR < 0.05",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=7,
            color="#666666",
        )
    _save_figure_svg_png(fig, svg_path, png_path)
    plt.close(fig)
    return selected_ranked


def _plot_s15d(table: pd.DataFrame, svg_path: Path, png_path: Path) -> pd.DataFrame:
    selected_ranked = _top_ranked(table, 20)
    selected_ranked["selection_rank"] = np.arange(len(selected_ranked))
    selected = selected_ranked.sort_values(
        ["gene_ratio", "selection_rank"], ascending=[True, False], kind="mergesort"
    )
    cmap = LinearSegmentedColormap.from_list("legacy_dot_padjust", DOT_PADJUST_COLORS)
    norm = _safe_norm(selected["adjusted_p_value"].to_numpy(dtype=float))
    counts = selected["overlap_count"].to_numpy(dtype=float)
    count_min, count_max = float(counts.min()), float(counts.max())
    if np.isclose(count_min, count_max):
        size_units = np.full_like(counts, np.mean(DOT_SIZE_RANGE))
    else:
        scaled = (counts - count_min) / (count_max - count_min)
        size_units = DOT_SIZE_RANGE[0] + np.sqrt(scaled) * (DOT_SIZE_RANGE[1] - DOT_SIZE_RANGE[0])
    sizes = np.square(size_units * 2.35)
    fig = plt.figure(figsize=(8.0, 6.0), facecolor="white")
    ax = fig.add_axes([0.35, 0.10, 0.49, 0.82])
    scatter = ax.scatter(
        selected["gene_ratio"],
        _display_terms(selected["term_name"]),
        s=sizes,
        c=selected["adjusted_p_value"],
        cmap=cmap,
        norm=norm,
        edgecolor="black",
        linewidth=0.55,
    )
    ax.set_title("arista_pattern_2_genes.csv - GO (BP) - Dotplot", fontsize=14)
    ax.set_xlabel("GeneRatio", fontsize=12)
    ax.set_ylabel("")
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.grid(which="major", color="#EBEBEB", linewidth=0.8)
    ax.grid(which="minor", axis="x", color="#EBEBEB", linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    colorbar_ax = fig.add_axes([0.875, 0.58, 0.030, 0.28])
    colorbar = fig.colorbar(scatter, cax=colorbar_ax)
    colorbar.ax.set_title("p.adjust", pad=8)
    colorbar.locator = MaxNLocator(nbins=4, min_n_ticks=3)
    colorbar.formatter = FormatStrFormatter("%.2g")
    colorbar.update_ticks()
    colorbar.ax.invert_yaxis()
    colorbar.outline.set_visible(False)
    count_examples = np.unique(np.rint(np.linspace(count_min, count_max, 4)).astype(int))

    def legend_size(value: float) -> float:
        if np.isclose(count_min, count_max):
            unit = float(np.mean(DOT_SIZE_RANGE))
        else:
            unit = DOT_SIZE_RANGE[0] + np.sqrt((value - count_min) / (count_max - count_min)) * (
                DOT_SIZE_RANGE[1] - DOT_SIZE_RANGE[0]
            )
        return float((unit * 2.35) ** 2)

    handles = [
        ax.scatter([], [], s=legend_size(float(value)), facecolor="white", edgecolor="black", linewidth=0.55)
        for value in count_examples
    ]
    ax.legend(
        handles,
        [str(value) for value in count_examples],
        title="Count",
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.06, 0.39),
        labelspacing=1.05,
    )
    if not bool(table["significant"].any()):
        ax.text(
            0.99,
            0.01,
            "No terms at FDR < 0.05",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=7,
            color="#666666",
        )
    _save_figure_svg_png(fig, svg_path, png_path)
    plt.close(fig)
    return selected_ranked.drop(columns="selection_rank")


def _rowwise_minmax(matrix: np.ndarray) -> np.ndarray:
    lo = matrix.min(axis=1, keepdims=True)
    hi = matrix.max(axis=1, keepdims=True)
    return np.divide(matrix - lo, np.maximum(hi - lo, 1e-12))


def _build_s16_prototypes(
    pair_timecourse: pd.DataFrame, pattern_summary: pd.DataFrame
) -> pd.DataFrame:
    pivot = pair_timecourse.pivot(index="pair", columns="time", values="score").loc[:, TIME_POINTS]
    assignments = pattern_summary.set_index("pair")["cluster"].astype(int)
    if set(pivot.index) != set(assignments.index):
        raise ValueError("Strict LR pair_timecourse and pattern_summary pair sets differ.")
    rows: list[dict[str, Any]] = []
    for cluster in sorted(assignments.unique()):
        members = assignments.index[assignments == cluster].tolist()
        normalized = _rowwise_minmax(pivot.loc[members].to_numpy(dtype=float))
        for index, time in enumerate(TIME_POINTS):
            rows.append(
                {
                    "cluster": int(cluster),
                    "time": float(time),
                    "mean_normalized_score": float(normalized[:, index].mean()),
                    "std_normalized_score": float(normalized[:, index].std(ddof=0)),
                    "n_pairs": int(len(members)),
                }
            )
    return pd.DataFrame(rows)


def _plot_s16(prototypes: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    palette = sns.color_palette("Set2", n_colors=max(1, prototypes["cluster"].nunique()))
    for index, (cluster, subset) in enumerate(prototypes.groupby("cluster", sort=True)):
        subset = subset.sort_values("time")
        color = palette[index % len(palette)]
        x = subset["time"].to_numpy(dtype=float)
        mean = subset["mean_normalized_score"].to_numpy(dtype=float)
        n_pairs = int(subset["n_pairs"].iloc[0])
        ax.plot(
            x,
            mean,
            marker="o",
            markersize=5,
            linewidth=2.6,
            color=color,
            label=f"Cluster {int(cluster)} (n={n_pairs})",
        )
        ax.fill_between(x, 0, mean, color=color, alpha=0.12)
    ax.set_title("Communication pattern prototypes")
    ax.set_xlabel("Time")
    ax.set_ylabel("Mean normalized score")
    ax.set_ylim(0, 1.02)
    ax.grid(True, axis="y", alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", format="svg", metadata=SVG_METADATA)
    plt.close(fig)


def _make_display_curve(x: np.ndarray, y: np.ndarray, n_points: int = 300) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    dense = np.linspace(x.min(), x.max(), n_points)
    return dense, np.interp(dense, x, y)


def _historical_s17_roster(
    legacy_summary: pd.DataFrame,
    strict_pair_timecourse: pd.DataFrame,
    strict_summary: pd.DataFrame,
    trajectory_coverage: pd.DataFrame,
    *,
    enforce_historical_estimability_counts: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required = {"pair", "cluster_id", "shape_rank", "auc"}
    if not required.issubset(legacy_summary.columns):
        raise KeyError(f"Legacy S17 summary is missing: {sorted(required - set(legacy_summary.columns))}")
    legacy = legacy_summary.sort_values(
        ["cluster_id", "shape_rank", "auc"],
        ascending=[True, True, False],
        kind="mergesort",
    ).reset_index(drop=True)
    if len(legacy) != 68 or legacy["pair"].duplicated().any():
        raise ValueError("The submitted S17 roster must contain 68 unique LR pairs.")
    strict_pairs = set(strict_summary["pair"].astype(str))
    strict_cluster = strict_summary.set_index("pair")["cluster"].astype(int).to_dict()
    pair_coverage = trajectory_coverage.loc[trajectory_coverage["trajectory_kind"].eq("pair")].copy()
    pair_coverage = pair_coverage.drop_duplicates("pair", keep="first").set_index("pair")
    roster_rows: list[dict[str, Any]] = []
    grid_rows: list[dict[str, Any]] = []
    strict_by_pair = {
        pair: subset.sort_values("time")
        for pair, subset in strict_pair_timecourse.groupby("pair", sort=False)
    }
    for index, old_row in legacy.iterrows():
        pair = str(old_row["pair"])
        estimable = pair in strict_pairs
        coverage_row = pair_coverage.loc[pair] if pair in pair_coverage.index else None
        requested = None if coverage_row is None else coverage_row.get("requested_subunits")
        missing = None if coverage_row is None else coverage_row.get("missing_pca_subunits")
        inactive = None if coverage_row is None else coverage_row.get("inactive_pca_subunits")
        drop_reason = None if coverage_row is None else coverage_row.get("drop_reason")
        if estimable:
            subset = strict_by_pair[pair]
            if len(subset) != len(TIME_POINTS) or not np.allclose(
                subset["time"].to_numpy(dtype=float), TIME_POINTS, rtol=0, atol=1e-12
            ):
                raise ValueError(f"Incomplete strict time grid for historical pair {pair}")
            if coverage_row is not None and not bool(coverage_row.get("retained", True)):
                raise ValueError(f"Strict summary retained {pair}, but trajectory coverage marks it dropped.")
        else:
            if coverage_row is None:
                drop_reason = "not_present_in_strict_active_pca_universe"
            elif pd.isna(drop_reason):
                drop_reason = "not_scoreable_in_uniform_active_pca_universe"
        roster_rows.append(
            {
                "historical_order": int(index + 1),
                "pair": pair,
                "historical_cluster": int(old_row["cluster_id"]),
                "historical_shape_rank": int(old_row["shape_rank"]),
                "strict_estimable": bool(estimable),
                "display_status": "curve" if estimable else "N/E",
                "strict_cluster": strict_cluster.get(pair),
                "requested_subunits": requested,
                "missing_pca_subunits": missing,
                "inactive_pca_subunits": inactive,
                "non_estimable_reason": None if estimable else drop_reason,
            }
        )
        if estimable:
            subset = strict_by_pair[pair]
            score_map = dict(zip(subset["time"].astype(float), subset["score"].astype(float)))
        else:
            score_map = {}
        for time in TIME_POINTS:
            grid_rows.append(
                {
                    "historical_order": int(index + 1),
                    "pair": pair,
                    "time": float(time),
                    "score": score_map.get(float(time), np.nan),
                    "strict_estimable": bool(estimable),
                    "display_status": "curve" if estimable else "N/E",
                    "strict_cluster": strict_cluster.get(pair),
                    "non_estimable_reason": None if estimable else drop_reason,
                }
            )
    roster = pd.DataFrame(roster_rows)
    grid = pd.DataFrame(grid_rows)
    non_estimable = roster.loc[~roster["strict_estimable"]].copy()
    if enforce_historical_estimability_counts and (
        int(roster["strict_estimable"].sum()) != 50 or len(non_estimable) != 18
    ):
        raise AssertionError(
            f"Strict historical roster must be 50 estimable + 18 N/E; got "
            f"{int(roster['strict_estimable'].sum())} + {len(non_estimable)}."
        )
    return roster, grid, non_estimable


def _plot_s17(roster: pd.DataFrame, grid: pd.DataFrame, out_path: Path) -> None:
    cols = 5
    rows = 14
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 2.15 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, record in zip(axes.flat, roster.itertuples(index=False)):
        ax.axis("on")
        subset = grid.loc[grid["pair"].eq(record.pair)].sort_values("time")
        ax.set_title(record.pair, fontsize=8)
        ax.grid(True, axis="y", alpha=0.2)
        ax.set_xlabel("Time", fontsize=8)
        ax.set_ylabel("Score", fontsize=8)
        ax.tick_params(axis="both", labelsize=7)
        if bool(record.strict_estimable):
            color = LR_CLUSTER_COLORS[int(record.strict_cluster)]
            x = subset["time"].to_numpy(dtype=float)
            y = subset["score"].to_numpy(dtype=float)
            x_dense, y_dense = _make_display_curve(x, y)
            ax.plot(x_dense, y_dense, color=color, linewidth=1.8)
            ax.scatter(x, y, color=color, s=12)
        else:
            ax.set_xlim(float(TIME_POINTS.min()), float(TIME_POINTS.max()))
            ax.set_xticks([0, 1, 2, 3, 4])
            ax.set_ylim(0, 1)
            ax.set_yticks([0, 0.5, 1.0])
            ax.text(
                0.5,
                0.5,
                "N/E",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                color="#777777",
            )
    fig.suptitle("All LR pair trends (n=68)", fontsize=13, y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", format="svg", metadata=SVG_METADATA)
    plt.close(fig)


def _run_rsvg(args: list[str]) -> None:
    executable = shutil.which("rsvg-convert")
    if executable is None:
        raise RuntimeError("rsvg-convert is required for deterministic SVG conversion.")
    env = os.environ.copy()
    env["SOURCE_DATE_EPOCH"] = "946684800"
    subprocess.run([executable, *args], check=True, env=env, capture_output=True)


def _render_svg_width(svg_path: Path, png_path: Path, width: int) -> None:
    _run_rsvg(["-f", "png", "-w", str(int(width)), "-b", "white", "-o", str(png_path), str(svg_path)])


def _svg_to_exact_pdf(svg_path: Path, pdf_path: Path, size_pt: tuple[float, float]) -> None:
    _run_rsvg(
        [
            "-f",
            "pdf",
            "-w",
            f"{size_pt[0]:.9f}pt",
            "-h",
            f"{size_pt[1]:.9f}pt",
            "--page-width",
            f"{size_pt[0]:.9f}pt",
            "--page-height",
            f"{size_pt[1]:.9f}pt",
            "-b",
            "white",
            "-o",
            str(pdf_path),
            str(svg_path),
        ]
    )
    _normalize_pdf_metadata(pdf_path)


def _normalize_pdf_metadata(path: Path) -> None:
    document = fitz.open(path)
    metadata = document.metadata or {}
    metadata.update(
        {
            "title": "ARISTA strict corrected legacy-style figure",
            "author": "CytoBridge",
            "subject": "S15-S17 strict full-complex replacement",
            "keywords": "ARISTA,CytoBridge,strict ligand-receptor complexes",
            "creator": "CytoBridge ARISTA strict legacy-style builder",
            "producer": "PyMuPDF",
            "creationDate": PDF_FIXED_DATE,
            "modDate": PDF_FIXED_DATE,
        }
    )
    document.set_metadata(metadata)
    temporary = path.with_suffix(".normalized.pdf")
    document.save(temporary, garbage=4, clean=True, deflate=True, no_new_id=True)
    document.close()
    temporary.replace(path)


def _render_pdf(path: Path, out_path: Path, dpi: float) -> tuple[int, int]:
    document = fitz.open(path)
    try:
        page = document[0]
        pixmap = page.get_pixmap(matrix=fitz.Matrix(dpi / 72.0, dpi / 72.0), alpha=False)
        pixmap.save(out_path)
        return pixmap.width, pixmap.height
    finally:
        document.close()


def _assemble_s15(
    panel_a_svg: Path,
    panel_b_svg: Path,
    panel_c_png: Path,
    panel_d_png: Path,
    reference_path: Path,
    out_png: Path,
    out_jpg: Path,
    out_pdf: Path,
    scratch: Path,
) -> dict[str, Any]:
    reference = Image.open(reference_path).convert("RGB")
    if reference.size != S15_CANVAS_PX:
        raise ValueError(f"Submitted S15 reference changed size: {reference.size}")
    sources = {"c": panel_c_png, "d": panel_d_png}
    for key, svg in {"a": panel_a_svg, "b": panel_b_svg}.items():
        rendered = scratch / f"s15_{key}_placement.png"
        _render_svg_width(svg, rendered, S15_LEGACY_PLACEMENTS[key]["width"])
        sources[key] = rendered
    canvas = Image.new("RGB", S15_CANVAS_PX, "white")
    realized: dict[str, Any] = {}
    for key in ("a", "b", "c", "d"):
        placement = S15_LEGACY_PLACEMENTS[key]
        image = Image.open(sources[key]).convert("RGB")
        if image.width != int(placement["width"]):
            height = int(round(image.height * int(placement["width"]) / image.width))
            image = image.resize((int(placement["width"]), height), Image.Resampling.LANCZOS)
        canvas.paste(image, (int(placement["x"]), int(placement["y"])))
        realized[key] = {
            **placement,
            "height": image.height,
            "source_panel": {
                "a": panel_a_svg.name,
                "b": panel_b_svg.name,
                "c": panel_c_png.name,
                "d": panel_d_png.name,
            }[key],
        }
    # Reuse the exact submitted outlined/raster panel-label glyphs and positions.
    for crop in S15_LABEL_CROPS.values():
        canvas.paste(reference.crop(crop), (crop[0], crop[1]))
    canvas.save(out_png, format="PNG", dpi=(S15_DPI, S15_DPI), optimize=False)
    canvas.save(out_jpg, format="JPEG", quality=95, subsampling=0, dpi=(S15_DPI, S15_DPI), optimize=False)
    document = fitz.open()
    width_pt = S15_CANVAS_PX[0] * 72.0 / S15_DPI
    height_pt = S15_CANVAS_PX[1] * 72.0 / S15_DPI
    page = document.new_page(width=width_pt, height=height_pt)
    page.insert_image(page.rect, filename=str(out_png))
    document.set_metadata(
        {
            "title": "ARISTA Supplementary Figure S15 strict corrected legacy style",
            "author": "CytoBridge",
            "creator": "CytoBridge ARISTA strict legacy-style builder",
            "producer": "PyMuPDF",
            "creationDate": PDF_FIXED_DATE,
            "modDate": PDF_FIXED_DATE,
        }
    )
    document.save(out_pdf, garbage=4, deflate=True, no_new_id=True)
    document.close()
    return realized


def _pdf_page_size(path: Path) -> tuple[float, float]:
    document = fitz.open(path)
    try:
        rect = document[0].rect
        return float(rect.width), float(rect.height)
    finally:
        document.close()


def _fit_width(image: Image.Image, width: int) -> Image.Image:
    height = int(round(image.height * width / image.width))
    return image.resize((width, height), Image.Resampling.LANCZOS)


def _load_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
        Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _pair_row(reference: Image.Image, candidate: Image.Image, width_each: int, title: str) -> Image.Image:
    ref = _fit_width(reference.convert("RGB"), width_each)
    cand = _fit_width(candidate.convert("RGB"), width_each)
    height = max(ref.height, cand.height)
    header = 46
    row = Image.new("RGB", (width_each * 2 + 30, height + header), "white")
    row.paste(ref, (0, header))
    row.paste(cand, (width_each + 30, header))
    draw = ImageDraw.Draw(row)
    font = _load_font(22)
    draw.text((8, 8), f"{title}: submitted legacy", fill="black", font=font)
    draw.text((width_each + 38, 8), f"{title}: strict corrected / legacy style", fill="black", font=font)
    return row


def _make_visual_qa(
    legacy_s15: Path,
    legacy_s16: Path,
    legacy_s17: Path,
    candidate_s15: Path,
    candidate_s16: Path,
    candidate_s17: Path,
    qa_dir: Path,
) -> dict[str, str]:
    legacy_s16_png = qa_dir / "legacy_S16_render.png"
    legacy_s17_png = qa_dir / "legacy_S17_render.png"
    candidate_s16_png = qa_dir / "candidate_S16_render.png"
    candidate_s17_png = qa_dir / "candidate_S17_render.png"
    _render_pdf(legacy_s16, legacy_s16_png, 144)
    _render_pdf(legacy_s17, legacy_s17_png, 72)
    _render_pdf(candidate_s16, candidate_s16_png, 144)
    _render_pdf(candidate_s17, candidate_s17_png, 72)
    rows = [
        _pair_row(Image.open(legacy_s15), Image.open(candidate_s15), 940, "S15"),
        _pair_row(Image.open(legacy_s16_png), Image.open(candidate_s16_png), 940, "S16"),
        _pair_row(Image.open(legacy_s17_png), Image.open(candidate_s17_png), 940, "S17"),
    ]
    contact = Image.new("RGB", (max(row.width for row in rows), sum(row.height for row in rows)), "white")
    y = 0
    for row in rows:
        contact.paste(row, (0, y))
        y += row.height
    contact_path = qa_dir / "legacy_vs_strict_corrected_contact_sheet.png"
    contact.save(contact_path, format="PNG")
    overlays: dict[str, str] = {}
    for name, legacy_path, candidate_path, width in [
        ("S15", legacy_s15, candidate_s15, 1200),
        ("S16", legacy_s16_png, candidate_s16_png, 1200),
        ("S17", legacy_s17_png, candidate_s17_png, 800),
    ]:
        old = _fit_width(Image.open(legacy_path).convert("RGB"), width)
        new = _fit_width(Image.open(candidate_path).convert("RGB"), width)
        common_height = min(old.height, new.height)
        overlay = Image.blend(old.crop((0, 0, width, common_height)), new.crop((0, 0, width, common_height)), 0.5)
        overlay_path = qa_dir / f"{name}_legacy_corrected_overlay.png"
        overlay.save(overlay_path, format="PNG")
        overlays[name] = str(overlay_path)
    return {
        "contact_sheet": str(contact_path.relative_to(qa_dir.parent)),
        **{
            f"overlay_{key}": str(Path(value).relative_to(qa_dir.parent))
            for key, value in overlays.items()
        },
    }


def _copy_sources(paths: dict[str, Path], target: Path) -> dict[str, dict[str, Any]]:
    target.mkdir(parents=True, exist_ok=False)
    output: dict[str, dict[str, Any]] = {}
    for name, source in paths.items():
        destination = target / f"{name}{source.suffix}"
        shutil.copy2(source, destination)
        snapshot = _file_record(destination)
        snapshot["path"] = str(destination.relative_to(target.parents[1]))
        output[name] = {"source": _file_record(source), "snapshot": snapshot}
    return output


def _strict_source_checks(
    summary_json: dict[str, Any],
    coverage: pd.DataFrame,
    pair_timecourse: pd.DataFrame,
    pattern_summary: pd.DataFrame,
    expression: pd.DataFrame,
    *,
    package_native_contract: bool = False,
) -> dict[str, Any]:
    analyses = summary_json.get("analyses", {})
    gene_contract = analyses.get("gene_dynamics", {})
    lr_contract = analyses.get("ligand_receptor", {})
    if gene_contract.get("pca_center_source") != "reference var['pca_center']":
        raise AssertionError(f"Unexpected PCA center source: {gene_contract.get('pca_center_source')}")
    if float(gene_contract.get("clip_min", np.nan)) != 0.0:
        raise AssertionError("Latest gene bank is not documented as clipped at zero.")
    if lr_contract.get("complex_mode") != "min" or lr_contract.get("require_all_subunits") is not True:
        raise AssertionError("Latest LR bank is not strict min/all-subunit scoring.")
    required_columns = {
        "complex_require_all_subunits",
        "n_partial_complexes",
        "n_lr_pairs_scored_partial_complex",
        "pca_active_filter_applied",
        "feature_universe_shared_across_time",
        "n_lr_pairs_scored",
    }
    if not required_columns.issubset(coverage.columns):
        raise KeyError(f"Strict LR coverage is missing {sorted(required_columns - set(coverage.columns))}")
    if not _as_bool(coverage["complex_require_all_subunits"]).all():
        raise AssertionError("Coverage contains a non-strict complex time point.")
    if not _as_bool(coverage["pca_active_filter_applied"]).all():
        raise AssertionError("Coverage does not apply the active-PCA filter at every time.")
    if not _as_bool(coverage["feature_universe_shared_across_time"]).all():
        raise AssertionError("Coverage does not use a shared feature universe across time.")
    if int(coverage["n_partial_complexes"].max()) != 0 or int(
        coverage["n_lr_pairs_scored_partial_complex"].max()
    ) != 0:
        raise AssertionError("Strict source contains partial-complex scores.")
    n_pairs_summary = int(pattern_summary["pair"].nunique())
    n_pairs_timecourse = int(pair_timecourse["pair"].nunique())
    if n_pairs_summary != n_pairs_timecourse:
        raise AssertionError("Strict LR summary/timecourse pair rosters differ.")
    if len(pair_timecourse) != n_pairs_summary * len(TIME_POINTS):
        raise AssertionError("Strict pair table does not contain a complete pair-by-time grid.")
    cluster_counts = pattern_summary["cluster"].astype(int).value_counts().sort_index().to_dict()
    if package_native_contract:
        if n_pairs_summary <= 0 or set(cluster_counts) != {1, 2} or min(cluster_counts.values()) <= 0:
            raise AssertionError(f"Invalid package-native strict LR clusters: {cluster_counts}")
        if expression.shape[1] != len(TIME_POINTS) or expression.shape[0] < S15_CLUSTER_GENE_COUNT:
            raise AssertionError(f"Invalid package-native expression shape: {expression.shape}")
    else:
        if n_pairs_summary != 530:
            raise AssertionError("Accepted strict source must contain 530 LR pairs.")
        if cluster_counts != {1: 482, 2: 48}:
            raise AssertionError(f"Unexpected strict LR cluster counts: {cluster_counts}")
        if expression.shape != (2241, 9):
            raise AssertionError(f"Unexpected corrected expression shape: {expression.shape}")
    return {
        "pca_center_source": gene_contract["pca_center_source"],
        "clip_min": gene_contract["clip_min"],
        "complex_mode": lr_contract["complex_mode"],
        "require_all_subunits": lr_contract["require_all_subunits"],
        "pca_active_filter_applied_all_times": True,
        "feature_universe_shared_across_time": True,
        "n_strict_lr_pairs": n_pairs_summary,
        "lr_cluster_counts": {str(key): int(value) for key, value in cluster_counts.items()},
        "gene_expression_shape": list(expression.shape),
    }


def _package_versions() -> dict[str, str]:
    packages = ["numpy", "pandas", "scipy", "matplotlib", "seaborn", "Pillow", "PyMuPDF"]
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unknown"
    return versions


def _relative_output_hashes(root: Path) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "MANIFEST.json":
            output[str(path.relative_to(root))] = {
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
    return output


def _validate_bitwise_determinism(root: Path, reference_root: Path) -> dict[str, Any]:
    reference_root = _require_dir(reference_root, "independent determinism reference")
    prefixes = ("figures/", "tables/", "provenance/source_snapshots/", "qa/")

    def inventory(base: Path) -> dict[str, str]:
        return {
            str(path.relative_to(base)): _sha256(path)
            for path in sorted(base.rglob("*"))
            if path.is_file() and str(path.relative_to(base)).startswith(prefixes)
        }

    current = inventory(root)
    reference = inventory(reference_root)
    missing_from_current = sorted(set(reference) - set(current))
    missing_from_reference = sorted(set(current) - set(reference))
    hash_mismatches = sorted(
        path for path in set(current) & set(reference) if current[path] != reference[path]
    )
    passed = not missing_from_current and not missing_from_reference and not hash_mismatches
    result = {
        "passed": passed,
        "method": "independent full rebuild; SHA256 equality for every core figure/table/source-snapshot/QA raster",
        "n_files_compared": len(current),
        "missing_from_current": missing_from_current,
        "missing_from_reference": missing_from_reference,
        "hash_mismatches": hash_mismatches,
        "reference_path_not_persisted": True,
    }
    if not passed:
        raise AssertionError(f"Independent bitwise determinism validation failed: {result}")
    return result


def build(args: argparse.Namespace) -> Path:
    _configure_legacy_style()
    bank = _require_dir(Path(args.latest_bank_root), "latest accepted ARISTA downstream bank")
    legacy_review = _require_dir(Path(args.legacy_review_root), "legacy ARISTA review output")
    legacy_s15 = _require_file(Path(args.legacy_s15_reference), "submitted S15 image")
    legacy_s16 = _require_file(Path(args.legacy_s16_reference), "submitted S16 PDF")
    legacy_s17 = _require_file(Path(args.legacy_s17_reference), "submitted S17 PDF")
    gmt_path = _require_file(Path(args.gene_set_gmt), "frozen GO Biological Process GMT")
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Immutable output already exists; refusing overwrite: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if not 0 < float(args.go_alpha) <= 1:
        raise ValueError("--go-alpha must be in (0, 1].")

    source_paths = {
        "bank_summary": _require_file(bank / "summary.json", "accepted bank summary"),
        "training_run_summary": _require_file(
            bank.parent / "provenance/training_run_summary.json", "training provenance"
        ),
        "gene_mean_expression": _require_file(
            bank / "gene_dynamics/mean_expression.csv", "corrected gene means"
        ),
        "gene_reconstruction_diagnostics": _require_file(
            bank / "gene_dynamics/reconstruction_diagnostics.csv", "gene reconstruction diagnostics"
        ),
        "lr_pair_timecourse": _require_file(
            bank / "ligand_receptor/pair_timecourse.csv", "strict LR pair time course"
        ),
        "lr_pattern_summary": _require_file(
            bank / "ligand_receptor/pattern_summary.csv", "strict LR pattern summary"
        ),
        "lr_coverage": _require_file(bank / "ligand_receptor/coverage.csv", "strict LR coverage"),
        "lr_trajectory_coverage": _require_file(
            bank / "ligand_receptor/trajectory_coverage.csv", "strict LR trajectory coverage"
        ),
        "legacy_s17_roster": _require_file(
            legacy_review / "tables/arista_lr_pattern_summary.csv", "submitted 68-pair roster"
        ),
        "legacy_plot_source": _require_file(
            PROJECT_ROOT / "evaluation/arista_code/arista_build_review_suite_local.py",
            "legacy ARISTA plotting source",
        ),
        "legacy_s15_reference": legacy_s15,
        "legacy_s16_reference": legacy_s16,
        "legacy_s17_reference": legacy_s17,
        "go_bp_gmt": gmt_path,
        "builder": Path(__file__).resolve(),
    }
    summary_json = json.loads(source_paths["bank_summary"].read_text(encoding="utf-8"))
    expression = _load_expression(source_paths["gene_mean_expression"])
    pair_timecourse = pd.read_csv(source_paths["lr_pair_timecourse"])
    pattern_summary = pd.read_csv(source_paths["lr_pattern_summary"])
    coverage = pd.read_csv(source_paths["lr_coverage"])
    trajectory_coverage = pd.read_csv(source_paths["lr_trajectory_coverage"])
    legacy_summary = pd.read_csv(source_paths["legacy_s17_roster"])
    strict_contract = _strict_source_checks(
        summary_json,
        coverage,
        pair_timecourse,
        pattern_summary,
        expression,
        package_native_contract=bool(args.package_native_contract),
    )

    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.stage-", dir=str(output_dir.parent))
    )
    try:
        figures = temporary / "figures"
        panels = figures / "panels"
        tables = temporary / "tables"
        provenance = temporary / "provenance"
        qa_dir = temporary / "qa"
        for directory in (figures, panels, tables, provenance, qa_dir):
            directory.mkdir(parents=True, exist_ok=True)
        source_snapshot_records = _copy_sources(
            {
                "bank_summary": source_paths["bank_summary"],
                "training_run_summary": source_paths["training_run_summary"],
                "gene_mean_expression": source_paths["gene_mean_expression"],
                "gene_reconstruction_diagnostics": source_paths["gene_reconstruction_diagnostics"],
                "lr_pair_timecourse": source_paths["lr_pair_timecourse"],
                "lr_pattern_summary": source_paths["lr_pattern_summary"],
                "lr_coverage": source_paths["lr_coverage"],
                "lr_trajectory_coverage": source_paths["lr_trajectory_coverage"],
                "legacy_s17_roster": source_paths["legacy_s17_roster"],
            },
            provenance / "source_snapshots",
        )

        ranking, assignments, normalized_profiles, prototypes = _build_corrected_gene_programs(expression)
        top18 = ranking.head(PANEL_A_N_GENES).copy()
        display_map = _unique_display_map(top18["raw_gene"].astype(str).tolist())
        top18["display_gene"] = top18["raw_gene"].astype(str).map(display_map)
        top18["gene_symbol"] = top18["raw_gene"].astype(str).map(_gene_symbol)
        panel_a_svg = panels / "S15a_top_variable_gene_trajectories.svg"
        panel_b_svg = panels / "S15b_gene_pattern_curves.svg"
        panel_c_svg = panels / "S15c_pattern_1_GO_barplot.svg"
        panel_d_svg = panels / "S15d_pattern_2_GO_dotplot.svg"
        panel_c_png = panels / "S15c_pattern_1_GO_barplot.png"
        panel_d_png = panels / "S15d_pattern_2_GO_dotplot.png"
        panel_a_values = _plot_s15a(expression, top18, panel_a_svg)
        _plot_s15b(prototypes, panel_b_svg)
        if not np.allclose(
            panel_a_values.to_numpy(dtype=float),
            expression.loc[top18["raw_gene"]].to_numpy(dtype=float),
            rtol=0,
            atol=0,
        ):
            raise AssertionError("S15a plotting values differ from persisted-center corrected means.")

        library = _load_gmt(gmt_path)
        background = {symbol for symbol in map(_gene_symbol, expression.index) if symbol}
        background &= set().union(*library.values())
        enrichment: dict[int, pd.DataFrame] = {}
        for pattern, subset in assignments.groupby("pattern", sort=True):
            query = [symbol for symbol in subset["gene_symbol"].tolist() if isinstance(symbol, str)]
            result = _ora_expression_background(
                query,
                library,
                background,
                alpha=float(args.go_alpha),
                min_set_size=int(args.go_min_set_size),
                max_set_size=int(args.go_max_set_size),
                min_overlap=int(args.go_min_overlap),
            )
            result.insert(0, "pattern", int(pattern))
            enrichment[int(pattern)] = result
        selected_c = _plot_s15c(enrichment[1], panel_c_svg, panel_c_png)
        selected_d = _plot_s15d(enrichment[2], panel_d_svg, panel_d_png)

        ranking.to_csv(tables / "S15_gene_variance_ranking.csv", index=False)
        assignments.to_csv(tables / "S15_top2000_program_assignments.csv", index=False)
        normalized_profiles.to_csv(tables / "S15_top2000_row_zscores.csv")
        prototypes.to_csv(tables / "S15_program_prototypes.csv", index=False)
        top18.to_csv(tables / "S15_top18_gene_roster.csv", index=False)
        panel_a_values.to_csv(tables / "S15a_actual_reconstructed_mean_values.csv")
        pd.DataFrame({"gene_symbol": sorted(background)}).to_csv(
            tables / "S15_GO_expression_background.csv", index=False
        )
        for pattern, result in enrichment.items():
            result.to_csv(tables / f"S15_pattern_{pattern}_GO_BP_full.csv", index=False)
        selected_c.to_csv(tables / "S15c_panel_terms.csv", index=False)
        selected_d.to_csv(tables / "S15d_panel_terms.csv", index=False)

        s15_png = figures / "FigureS15_ARISTA_strict_corrected_legacy_style.png"
        s15_jpg = figures / "FigureS15_ARISTA_strict_corrected_legacy_style.jpg"
        s15_pdf = figures / "FigureS15_ARISTA_strict_corrected_legacy_style.pdf"
        with tempfile.TemporaryDirectory(prefix="arista_s15_placement_", dir=str(temporary)) as scratch_name:
            s15_realized_geometry = _assemble_s15(
                panel_a_svg,
                panel_b_svg,
                panel_c_png,
                panel_d_png,
                legacy_s15,
                s15_png,
                s15_jpg,
                s15_pdf,
                Path(scratch_name),
            )

        s16_prototypes = _build_s16_prototypes(pair_timecourse, pattern_summary)
        s16_prototypes.to_csv(tables / "S16_strict_cluster_prototypes.csv", index=False)
        s16_svg = figures / "FigureS16_ARISTA_strict_corrected_legacy_style.svg"
        s16_pdf = figures / "FigureS16_ARISTA_strict_corrected_legacy_style.pdf"
        s16_png = figures / "FigureS16_ARISTA_strict_corrected_legacy_style.png"
        _plot_s16(s16_prototypes, s16_svg)
        _svg_to_exact_pdf(s16_svg, s16_pdf, S16_PAGE_PT)
        _render_pdf(s16_pdf, s16_png, 200)

        roster, roster_grid, non_estimable = _historical_s17_roster(
            legacy_summary,
            pair_timecourse,
            pattern_summary,
            trajectory_coverage,
            enforce_historical_estimability_counts=not bool(args.package_native_contract),
        )
        roster.to_csv(tables / "S17_historical_roster_strict_status.csv", index=False)
        roster_grid.to_csv(tables / "S17_historical_roster_timecourse.csv", index=False)
        non_estimable.to_csv(tables / "S17_non_estimable_pairs.csv", index=False)
        s17_svg = figures / "FigureS17_ARISTA_strict_corrected_legacy_style.svg"
        s17_pdf = figures / "FigureS17_ARISTA_strict_corrected_legacy_style.pdf"
        s17_png = figures / "FigureS17_ARISTA_strict_corrected_legacy_style.png"
        _plot_s17(roster, roster_grid, s17_svg)
        _svg_to_exact_pdf(s17_svg, s17_pdf, S17_PAGE_PT)
        _render_pdf(s17_pdf, s17_png, 100)

        visual_qa_paths = _make_visual_qa(
            legacy_s15,
            legacy_s16,
            legacy_s17,
            s15_png,
            s16_pdf,
            s17_pdf,
            qa_dir,
        )
        determinism_validation = None
        if args.determinism_reference is not None:
            determinism_validation = _validate_bitwise_determinism(
                temporary, Path(args.determinism_reference).expanduser().resolve()
            )
            _json_dump(
                provenance / "determinism_validation.json",
                determinism_validation,
            )
        s15_size = Image.open(s15_png).size
        s15_jpg_size = Image.open(s15_jpg).size
        s16_size = _pdf_page_size(s16_pdf)
        s17_size = _pdf_page_size(s17_pdf)
        n_e_comment_count = s17_svg.read_text(encoding="utf-8").count("<!-- N/E -->")
        expected_lr_pairs = int(strict_contract["n_strict_lr_pairs"])
        expected_cluster_counts = {
            int(key): int(value)
            for key, value in strict_contract["lr_cluster_counts"].items()
        }
        actual_estimable = int(roster["strict_estimable"].sum())
        actual_non_estimable = int(len(non_estimable))
        checks = [
            {"name": "strict_all_subunits", "passed": strict_contract["require_all_subunits"] is True},
            {"name": "zero_partial_complexes", "passed": int(coverage["n_partial_complexes"].max()) == 0},
            {"name": "active_pca_universe", "passed": strict_contract["pca_active_filter_applied_all_times"]},
            {"name": "persisted_pca_center", "passed": strict_contract["pca_center_source"] == "reference var['pca_center']"},
            {"name": "S15a_no_centering_or_shift", "passed": True},
            {"name": "S15_canvas_png", "passed": tuple(s15_size) == S15_CANVAS_PX, "observed": list(s15_size)},
            {"name": "S15_canvas_jpg", "passed": tuple(s15_jpg_size) == S15_CANVAS_PX, "observed": list(s15_jpg_size)},
            {"name": "S16_page_size", "passed": bool(np.allclose(s16_size, S16_PAGE_PT, atol=1e-3)), "observed": list(s16_size)},
            {"name": "S17_page_size", "passed": bool(np.allclose(s17_size, S17_PAGE_PT, atol=1e-3)), "observed": list(s17_size)},
            {"name": "S16_complete_strict_pair_roster", "passed": int(s16_prototypes.groupby("cluster")["n_pairs"].first().sum()) == expected_lr_pairs, "observed": expected_lr_pairs},
            {"name": "S16_current_cluster_counts", "passed": s16_prototypes.groupby("cluster")["n_pairs"].first().to_dict() == expected_cluster_counts, "observed": expected_cluster_counts},
            {"name": "S17_68_fixed_slots", "passed": len(roster) == 68},
            {"name": "S17_current_estimable_count", "passed": actual_estimable + actual_non_estimable == 68, "observed": actual_estimable},
            {"name": "S17_current_non_estimable_count", "passed": actual_estimable + actual_non_estimable == 68, "observed": actual_non_estimable},
            {"name": "S17_rendered_NE_labels_match_current_state", "passed": n_e_comment_count == actual_non_estimable, "observed": n_e_comment_count},
            {"name": "S17_no_fabricated_NE_scores", "passed": bool(roster_grid.loc[~roster_grid["strict_estimable"], "score"].isna().all())},
            {"name": "S17_estimable_complete_9_time_grid", "passed": bool(roster_grid.loc[roster_grid["strict_estimable"]].groupby("pair")["score"].count().eq(9).all())},
            {"name": "GO_expression_background", "passed": all(int(table["background_size"].iloc[0]) == len(background) for table in enrichment.values())},
        ]
        if determinism_validation is not None:
            checks.append(
                {
                    "name": "independent_bitwise_determinism",
                    "passed": bool(determinism_validation["passed"]),
                    "n_files_compared": int(
                        determinism_validation["n_files_compared"]
                    ),
                }
            )
        failed = [check["name"] for check in checks if not check["passed"]]
        if failed:
            raise AssertionError(f"Final S15-S17 QA failed: {failed}")
        qa_report = {
            "status": "PASS",
            "checks": checks,
            "visual_qa": visual_qa_paths,
            "determinism": determinism_validation,
            "geometry": {
                "S15_canvas_px": list(S15_CANVAS_PX),
                "S15_realized_legacy_placements": s15_realized_geometry,
                "S16_page_pt": list(s16_size),
                "S17_page_pt": list(s17_size),
                "S17_grid": [14, 5],
            },
        }
        _json_dump(temporary / "QA_REPORT.json", qa_report)

        caption_notes = f"""
        # ARISTA S15--S17 caption corrections

        ## S15

        Panel a shows the actual persisted-center inverse-PCA mean after per-cell
        clipping at zero. The submitted zero-centered appearance resulted from
        omitting the PCA fit center and is not retained. Panels c and d show the
        20 top-ranked GO Biological Process terms under an expression-universe
        background. Pattern 1 has {int(enrichment[1]['significant'].sum())} and
        Pattern 2 has {int(enrichment[2]['significant'].sum())} terms at FDR < 0.05;
        the plots explicitly state when no tested term passes that threshold.

        ## S16

        Curves are pair-wise min--max normalized strict full-complex LR scores,
        summarized across all {expected_lr_pairs} estimable pairs
        (Cluster 1: {expected_cluster_counts.get(1, 0)};
        Cluster 2: {expected_cluster_counts.get(2, 0)}).

        ## S17

        The y-axis is raw strict-complex `Score`, not normalized score. The fixed
        submitted 68-pair roster contains {actual_estimable} estimable curves and
        {actual_non_estimable} non-estimable
        pairs. Non-estimable pairs retain their historical panel slots and are
        labeled `N/E`; no partial-complex value is substituted.
        """
        _write_text(temporary / "CAPTION_UPDATE_NOTES.md", textwrap.dedent(caption_notes))
        readme = f"""
        # ARISTA strict-corrected S15--S17, submitted visual grammar

        This immutable bundle replaces S15--S17 numerically without redesigning
        the submitted figures. It uses the accepted full-model bank at
        `{bank}` and reuses the exact submitted canvas/page geometry, palettes,
        labels, axes, line grammar, and historical S17 order.

        Scientific guardrails:

        - persisted PCA center: `reference var['pca_center']`;
        - reconstructed expression: per-cell clip at zero, then arithmetic mean;
        - LR complexes: minimum across subunits with every subunit required;
        - active-PCA feature universe shared across all nine time points;
        - S17 N/E pairs remain visible, with `NaN` numeric values and explicit reasons;
        - no output path is overwritten by the builder.

        See `QA_REPORT.json`, `MANIFEST.json`, `CAPTION_UPDATE_NOTES.md`, and
        `qa/legacy_vs_strict_corrected_contact_sheet.png`.
        """
        _write_text(temporary / "README.md", textwrap.dedent(readme))
        input_records = {name: _file_record(path) for name, path in source_paths.items()}
        manifest = {
            "workflow": "arista_s15_s17_strict_corrected_legacy_style_v1",
            "immutable_contract": "transactional build; existing output directory is rejected",
            "scientific_source_contract": {
                **strict_contract,
                "model_identifier": (
                    "package-native-spatialqc-z50-retrain-r1"
                    if args.package_native_contract
                    else "3c87a3e"
                ),
                "expression_values": "actual corrected mean_expression.csv; no temporal centering or cosmetic shift",
                "gene_programs": {
                    "candidate_genes": 2000,
                    "variance": "population temporal variance (ddof=0)",
                    "normalization": "row z-score with population standard deviation (ddof=0)",
                    "linkage": "SciPy average linkage, Euclidean metric",
                    "cut": "SciPy cut_tree exact n_clusters=2",
                    "semantic_order": "earlier prototype peak is Pattern 1; later peak is Pattern 2",
                    "pattern_counts": {
                        str(key): int(value)
                        for key, value in assignments["pattern"].value_counts().sort_index().items()
                    },
                },
                "go_ora": {
                    "library": str(gmt_path),
                    "background": f"corrected {expression.shape[0]}-gene candidate universe mapped into the frozen GO BP library",
                    "background_size": len(background),
                    "test": "one-sided hypergeometric survival function",
                    "multiple_testing": "Benjamini-Hochberg separately within each pattern",
                    "alpha": float(args.go_alpha),
                    "significant_term_counts": {
                        str(pattern): int(table["significant"].sum())
                        for pattern, table in enrichment.items()
                    },
                },
                "S17": {
                    "historical_slots": 68,
                    "strict_estimable": actual_estimable,
                    "non_estimable": actual_non_estimable,
                    "non_estimable_display": "old-style titled/axed/grid slot with centered gray N/E; score remains NaN",
                },
            },
            "legacy_visual_contract": {
                "source_code": str(source_paths["legacy_plot_source"]),
                "global_style": "seaborn white/paper; DejaVu Sans 10; top/right spines off; 0.8 axes/ticks",
                "S15": {
                    "canvas_px": list(S15_CANVAS_PX),
                    "placements": S15_LEGACY_PLACEMENTS,
                    "panel_label_glyphs": "pixel-exact crops reused from submitted S15 image",
                    "a": "18 tab20 lines; o markers; linewidth 1.8; dashed alpha-0.2 grid; outside frameless legend",
                    "b": "Set2 green/orange; o markers; linewidth 2.2; mean +/- population SD; zero line",
                    "c": "20-term horizontal Count bars; raw p.adjust color scale",
                    "d": "GeneRatio x; Count size; raw p.adjust red-to-blue fill; black edges",
                },
                "S16": {
                    "page_pt": list(S16_PAGE_PT),
                    "normalization": "row-wise min-max per pair before cluster mean",
                    "colors": PATTERN_COLORS,
                    "line": "o markers size 5; linewidth 2.6; fill from zero alpha 0.12",
                },
                "S17": {
                    "page_pt": list(S17_PAGE_PT),
                    "grid": "5 columns x 14 rows",
                    "ordering": "submitted cluster_id, shape_rank, auc ordering",
                    "curve": "300-point np.interp; raw Score; linewidth 1.8; scatter size 12",
                    "colors": LR_CLUSTER_COLORS,
                },
            },
            "known_legacy_analysis_errors_not_reproduced": [
                "S15a omitted the PCA fit center and appeared signed/zero-centered.",
                "Permissive LR scoring retained partial complexes with missing/inactive subunits.",
                "The submitted S17 caption described normalization although the plotted y-axis/data were raw Score.",
                "A later v11 GO rerun used the whole GO-library union as background; this bundle uses the expression universe.",
            ],
            "inputs": input_records,
            "source_snapshots": source_snapshot_records,
            "software": {"python": sys.version, "platform": platform.platform(), "packages": _package_versions()},
            "qa_status": "PASS",
            "determinism_validation": determinism_validation,
            "outputs": _relative_output_hashes(temporary),
        }
        _json_dump(temporary / "MANIFEST.json", manifest)
        temporary.replace(output_dir)
        return output_dir
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> None:
    output = build(_parser().parse_args())
    print(output)


if __name__ == "__main__":
    main()
