#!/usr/bin/env python3
"""Draw ARISTA S22 GO panels from clusterProfiler results. Adapted from the original figure source."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.ticker import AutoMinorLocator, FormatStrFormatter, MaxNLocator
import numpy as np
import pandas as pd

from . import gene_programs as legacy


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gene_ratio(value: str) -> float:
    numerator, denominator = str(value).split("/", 1)
    return float(numerator) / float(denominator)


def save_panel(fig: plt.Figure, svg_path: Path, png_path: Path) -> None:
    legacy.save(fig, svg_path, facecolor="white", metadata=legacy.SVG_METADATA)
    fig.savefig(png_path, facecolor="white", dpi=300, metadata={"Software": "CytoBridge"})
    plt.close(fig)


def plot_pattern1(table: pd.DataFrame, svg_path: Path, png_path: Path) -> pd.DataFrame:
    ranked = table.sort_values(
        ["pvalue", "Count", "Description"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    if ranked.empty:
        raise ValueError("Pattern 1 unexpectedly has no clusterProfiler terms")
    selected_ranked = ranked.head(20).copy()
    selected = selected_ranked.iloc[::-1].copy()
    cmap = LinearSegmentedColormap.from_list("legacy_bar_padjust", legacy.BAR_PADJUST_COLORS)
    values = selected["pvalue"].to_numpy(dtype=float)
    norm = legacy._safe_norm(values)
    fig = plt.figure(figsize=(8.0, 6.0), facecolor="white")
    ax = fig.add_axes([0.34, 0.10, 0.51, 0.82])
    ax.barh(
        legacy._display_terms(selected["Description"], width=42),
        selected["Count"],
        color=cmap(norm(values)),
        edgecolor="none",
    )
    ax.set_title("GO biological processes: pattern 1", fontsize=14)
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
    cax = fig.add_axes([0.89, 0.36, 0.030, 0.26])
    colorbar = fig.colorbar(scalar, cax=cax)
    colorbar.ax.set_title("P value", pad=8)
    colorbar.locator = MaxNLocator(nbins=3, min_n_ticks=2)
    colorbar.formatter = FormatStrFormatter("%.2g")
    colorbar.update_ticks()
    colorbar.outline.set_visible(False)
    save_panel(fig, svg_path, png_path)
    return selected_ranked


def plot_pattern2(table: pd.DataFrame, summary: pd.DataFrame, svg_path: Path, png_path: Path) -> pd.DataFrame:
    row = summary.loc[summary["pattern"] == 2].iloc[0]
    ranked = table.sort_values(
        ["pvalue", "Count", "Description"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    if ranked.empty:
        raise ValueError("Pattern 2 unexpectedly has no clusterProfiler terms")
    selected_ranked = ranked.head(20).copy()
    selected_ranked["gene_ratio_numeric"] = selected_ranked["GeneRatio"].map(gene_ratio)
    selected_ranked["selection_rank"] = np.arange(len(selected_ranked))
    selected = selected_ranked.sort_values(
        ["gene_ratio_numeric", "selection_rank"], ascending=[True, False], kind="mergesort"
    )
    cmap = LinearSegmentedColormap.from_list("legacy_dot_nominal_p", legacy.DOT_PADJUST_COLORS)
    norm = legacy._safe_norm(selected["pvalue"].to_numpy(dtype=float))
    counts = selected["Count"].to_numpy(dtype=float)
    count_min, count_max = float(counts.min()), float(counts.max())
    if np.isclose(count_min, count_max):
        size_units = np.full_like(counts, np.mean(legacy.DOT_SIZE_RANGE))
    else:
        scaled = (counts - count_min) / (count_max - count_min)
        size_units = legacy.DOT_SIZE_RANGE[0] + np.sqrt(scaled) * (
            legacy.DOT_SIZE_RANGE[1] - legacy.DOT_SIZE_RANGE[0]
        )
    sizes = np.square(size_units * 2.35)
    fig = plt.figure(figsize=(8.0, 6.0), facecolor="white")
    ax = fig.add_axes([0.39, 0.10, 0.45, 0.82])
    scatter = ax.scatter(
        selected["gene_ratio_numeric"],
        legacy._display_terms(selected["Description"], width=38),
        s=sizes,
        c=selected["pvalue"],
        cmap=cmap,
        norm=norm,
        edgecolor="black",
        linewidth=0.55,
    )
    ax.set_title("GO biological processes: pattern 2", fontsize=14)
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
    colorbar.ax.set_title("P value", pad=8)
    colorbar.locator = MaxNLocator(nbins=4, min_n_ticks=3)
    colorbar.formatter = FormatStrFormatter("%.2g")
    colorbar.update_ticks()
    colorbar.ax.invert_yaxis()
    colorbar.outline.set_visible(False)
    count_examples = np.unique(np.rint(np.linspace(count_min, count_max, 4)).astype(int))

    def legend_size(value: float) -> float:
        if np.isclose(count_min, count_max):
            unit = float(np.mean(legacy.DOT_SIZE_RANGE))
        else:
            unit = legacy.DOT_SIZE_RANGE[0] + np.sqrt(
                (value - count_min) / (count_max - count_min)
            ) * (legacy.DOT_SIZE_RANGE[1] - legacy.DOT_SIZE_RANGE[0])
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
    save_panel(fig, svg_path, png_path)
    return selected_ranked.drop(columns="selection_rank")


