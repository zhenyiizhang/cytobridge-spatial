"""ARISTA gene-program plots from the original paper plotting functions."""
from pathlib import Path
import textwrap
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.ticker import AutoMinorLocator, FormatStrFormatter, MaxNLocator
import numpy as np
import pandas as pd
import seaborn as sns


def save(fig, path, **kwargs):
    """Save the requested format and a vector PDF from the same live figure."""
    fig.savefig(path, **kwargs)
    if Path(path).suffix == '.svg':
        pdf_kwargs = {k: v for k, v in kwargs.items() if k not in ('metadata', 'format')}
        fig.savefig(Path(path).with_suffix('.pdf'), **pdf_kwargs)


PATTERN_COLORS = {1: "#66c2a5", 2: "#fc8d62"}

BAR_PADJUST_COLORS = ["#264653", "#2A9D8F", "#E9C46A", "#F4A261", "#E76F51"]

DOT_PADJUST_COLORS = ["#e06663", "#327eba"]

DOT_SIZE_RANGE = (3.0, 8.0)

SVG_METADATA = {"Date": None, "Creator": "CytoBridge ARISTA strict legacy-style builder"}

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
            "font.family": "Arial",
            "svg.fonttype": "path",
            "svg.hashsalt": "arista-s15-s17-strict-legacy-style-v1",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

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
    save(fig, out_path, bbox_inches="tight", format="svg", metadata=SVG_METADATA)
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
    save(fig, out_path, bbox_inches="tight", format="svg", metadata=SVG_METADATA)
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
    save(fig, svg_path, facecolor="white", metadata=SVG_METADATA)
    save(fig, png_path, facecolor="white", dpi=300, metadata={"Software": "CytoBridge"})

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
