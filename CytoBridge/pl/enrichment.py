"""Dataset-agnostic plots for gene-set over-representation results."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["plot_enrichment_bar", "plot_enrichment_dot"]


def _output_path(value: str | Path) -> Path:
    path = Path(value).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _select_terms(results: pd.DataFrame, top_n: int) -> pd.DataFrame:
    required = {
        "term_name",
        "overlap_count",
        "gene_ratio",
        "adjusted_p_value",
        "p_value",
        "fold_enrichment",
    }
    missing = sorted(required.difference(results.columns))
    if missing:
        raise KeyError(f"results is missing columns: {missing}")
    if int(top_n) <= 0:
        raise ValueError("top_n must be positive.")
    table = results.copy()
    table = table[np.isfinite(table["adjusted_p_value"].astype(float))]
    table = table.sort_values(
        ["adjusted_p_value", "p_value", "fold_enrichment"],
        ascending=[True, True, False],
        kind="mergesort",
    ).head(int(top_n))
    if table.empty:
        raise ValueError("No finite enrichment terms are available to plot.")
    return table.iloc[::-1].reset_index(drop=True)


def _color_values(table: pd.DataFrame) -> np.ndarray:
    adjusted = np.maximum(
        table["adjusted_p_value"].to_numpy(dtype=float),
        np.finfo(float).tiny,
    )
    return -np.log10(adjusted)


def plot_enrichment_bar(
    results: pd.DataFrame,
    *,
    out_path: str | Path,
    top_n: int = 20,
    title: str = "Gene-set enrichment",
    x_label: str = "Overlap count",
) -> Path:
    """Plot top terms as horizontal bars colored by adjusted p-value."""
    import matplotlib.pyplot as plt

    table = _select_terms(results, top_n)
    color_values = _color_values(table)
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(vmin=float(color_values.min()), vmax=float(color_values.max()))
    if np.isclose(norm.vmin, norm.vmax):
        norm = plt.Normalize(vmin=0.0, vmax=max(1.0, float(norm.vmax)))
    height = max(4.5, 0.31 * len(table) + 1.7)
    fig, ax = plt.subplots(figsize=(8.2, height), facecolor="white")
    ax.barh(
        table["term_name"].astype(str),
        table["overlap_count"].to_numpy(dtype=float),
        color=cmap(norm(color_values)),
        edgecolor="none",
    )
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.18)
    scalar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    colorbar = fig.colorbar(scalar, ax=ax, pad=0.02)
    colorbar.set_label("-log10 adjusted p-value")
    fig.tight_layout()
    path = _output_path(out_path)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_enrichment_dot(
    results: pd.DataFrame,
    *,
    out_path: str | Path,
    top_n: int = 20,
    title: str = "Gene-set enrichment",
    x_label: str = "Gene ratio",
) -> Path:
    """Plot top terms with gene ratio, overlap count, and adjusted p-value."""
    import matplotlib.pyplot as plt

    table = _select_terms(results, top_n)
    color_values = _color_values(table)
    counts = table["overlap_count"].to_numpy(dtype=float)
    sizes = 30.0 + 170.0 * counts / max(float(counts.max()), 1.0)
    height = max(4.5, 0.31 * len(table) + 1.7)
    fig, ax = plt.subplots(figsize=(8.2, height), facecolor="white")
    scatter = ax.scatter(
        table["gene_ratio"].to_numpy(dtype=float),
        table["term_name"].astype(str),
        s=sizes,
        c=color_values,
        cmap="viridis",
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.18)
    colorbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    colorbar.set_label("-log10 adjusted p-value")
    handles, labels = scatter.legend_elements(
        prop="sizes",
        alpha=0.7,
        num=3,
        func=lambda size: np.maximum((size - 30.0) / 170.0 * counts.max(), 1.0),
        fmt="{x:.0f}",
    )
    ax.legend(handles, labels, title="Overlap", frameon=False, loc="lower right")
    fig.tight_layout()
    path = _output_path(out_path)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path
