#!/usr/bin/env python3
"""Render corrected MOSTA SI Figure S8 in the submitted visual grammar.

Numerical truth comes only from the audited latest-package S8 tables.  The
layout and styling reproduce the figure embedded on page 34 of the submitted
SI and its historical notebook renderer.  No numerical coordinate is rotated,
stretched, warped, or otherwise transformed to imitate the old result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
import seaborn as sns


FIGSIZE_IN = (8.43, 11.0)  # 1686 x 2200 px at 200 dpi: submitted aspect ratio.
PROGRAM_COLORS = {1: "#1f77b4", 2: "#ff7f0e"}
PANEL_BG = "#ffffff"
GRID_COLOR = "#d9d4cb"
TEXT_DARK = "#2f2b28"
TEXT_MID = "#625b56"
HEATMAP_CMAP = "viridis"
HEATMAP_CLIP = 1.8


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(path: Path) -> dict[str, object]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256(resolved),
    }


def sparse_time_ticks(ax: mpl.axes.Axes, times: np.ndarray) -> None:
    positions = np.unique(np.linspace(0, len(times) - 1, 6, dtype=int))
    ax.set_xticks(positions + 0.5)
    ax.set_xticklabels(
        [f"{times[index]:.2f}" for index in positions],
        rotation=35,
        ha="right",
        fontsize=8,
    )


def load_inputs(input_dir: Path) -> dict[str, pd.DataFrame]:
    paths = {
        "mean": input_dir / "brain_hvg_mean_log1p_by_time.csv",
        "zscore": input_dir / "brain_hvg_gene_wise_zscore.csv",
        "top20": input_dir / "brain_top20_temporal_variable_genes.csv",
        "variance_rank": input_dir / "brain_hvg_temporal_variance_rank.csv",
        "assignments": input_dir / "brain_hvg_ward_k2_assignments.csv",
        "prototypes": input_dir / "brain_hvg_ward_k2_prototypes.csv",
        "representatives": input_dir / "brain_program_representative_genes_top5.csv",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing S8 input files: {missing}")

    mean = pd.read_csv(paths["mean"], index_col=0)
    zscore = pd.read_csv(paths["zscore"], index_col=0)
    mean.columns = [float(value) for value in mean.columns]
    zscore.columns = [float(value) for value in zscore.columns]
    top20 = pd.read_csv(paths["top20"])
    variance_rank = pd.read_csv(paths["variance_rank"])
    assignments = pd.read_csv(paths["assignments"])
    prototypes = pd.read_csv(paths["prototypes"])
    representatives = pd.read_csv(paths["representatives"])

    if mean.shape != (2000, 13) or zscore.shape != (2000, 13):
        raise ValueError(f"Expected 2000 x 13 matrices, got {mean.shape}, {zscore.shape}")
    if not np.allclose(mean.columns.to_numpy(float), zscore.columns.to_numpy(float)):
        raise ValueError("Mean-log1p and z-score time grids differ")
    if float(np.nanmin(mean.to_numpy(float))) < -1e-12:
        raise ValueError("Corrected clipped mean-log1p table contains negative values")
    if top20.shape[0] != 20 or top20["gene"].duplicated().any():
        raise ValueError("Top-variable table must contain 20 unique genes")
    if not set(top20["gene"]).issubset(mean.index):
        raise ValueError("Top-variable genes are absent from mean-log1p matrix")
    if variance_rank.shape[0] != 2000 or variance_rank["gene"].duplicated().any():
        raise ValueError("Variance ranking must contain all 2,000 unique HVGs")
    if variance_rank.head(20)["gene"].tolist() != top20["gene"].tolist():
        raise ValueError("Saved top-20 table does not equal the variance-rank prefix")
    if sorted(assignments["cluster"].unique().tolist()) != [1, 2]:
        raise ValueError("Expected exact Ward k=2 assignments")
    if assignments.shape[0] != 2000:
        raise ValueError("Expected assignments for all 2,000 original HVGs")
    cluster_counts = assignments.groupby("cluster").size().to_dict()
    if cluster_counts != {1: 883, 2: 1117}:
        raise ValueError(f"Unexpected corrected program sizes: {cluster_counts}")
    if representatives.groupby("program").size().to_dict() != {1: 5, 2: 5}:
        raise ValueError("Expected five representative genes per program")

    return {
        "mean": mean,
        "zscore": zscore,
        "top20": top20,
        "variance_rank": variance_rank,
        "assignments": assignments,
        "prototypes": prototypes,
        "representatives": representatives,
        "paths": paths,
    }


def render(inputs: dict[str, object], output_dir: Path) -> dict[str, Path]:
    mean: pd.DataFrame = inputs["mean"]  # type: ignore[assignment]
    zscore: pd.DataFrame = inputs["zscore"]  # type: ignore[assignment]
    top20: pd.DataFrame = inputs["top20"]  # type: ignore[assignment]
    variance_rank: pd.DataFrame = inputs["variance_rank"]  # type: ignore[assignment]
    prototypes: pd.DataFrame = inputs["prototypes"]  # type: ignore[assignment]
    representatives: pd.DataFrame = inputs["representatives"]  # type: ignore[assignment]
    times = mean.columns.to_numpy(dtype=float)

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig = plt.figure(figsize=FIGSIZE_IN, facecolor=PANEL_BG)

    # Panel a: exact historical single-axis/legend grammar, corrected values.
    # The submitted notebook renders the top 25 (the compact top-20 table is a
    # downstream audit convenience, not the panel-a display count).
    display_genes = variance_rank.head(25)["gene"].astype(str).tolist()
    ax_a = fig.add_axes([0.066, 0.623, 0.750, 0.304])
    palette = sns.color_palette("tab10", n_colors=25)
    for color, gene in zip(palette, display_genes):
        ax_a.plot(
            times,
            mean.loc[gene].to_numpy(float),
            marker="o",
            markersize=4.2,
            linewidth=1.8,
            color=color,
            label=gene,
        )
    ax_a.set_title("Top variable gene trajectories", pad=7)
    ax_a.set_xlabel("Time")
    ax_a.set_ylabel("Mean reconstructed log1p")
    ax_a.set_xticks(np.arange(0.0, 3.01, 0.5))
    ax_a.grid(True, alpha=0.2, linestyle="--")
    ax_a.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=8.7,
        handlelength=2.0,
        handletextpad=0.55,
        labelspacing=0.55,
        borderaxespad=0.0,
    )
    fig.text(0.014, 0.974, "a", fontsize=18, fontweight="bold", ha="left", va="top")

    # Panel b header and exact two-program geometry.
    fig.text(0.022, 0.552, "b", fontsize=18, fontweight="bold", ha="left", va="top")
    fig.text(
        0.0735,
        0.552,
        "Brain temporal programs (k=2)",
        fontsize=14,
        ha="left",
        va="top",
        color=TEXT_DARK,
    )

    curve_boxes = {
        1: [0.160, 0.361, 0.293, 0.118],
        2: [0.524, 0.361, 0.293, 0.118],
    }
    heat_boxes = {
        1: [0.160, 0.057, 0.283, 0.254],
        2: [0.512, 0.057, 0.283, 0.254],
    }
    heat_axes: list[mpl.axes.Axes] = []
    for program in (1, 2):
        color = PROGRAM_COLORS[program]
        curve = prototypes.loc[prototypes["cluster"] == program].sort_values("time")
        ax_curve = fig.add_axes(curve_boxes[program])
        curve_times = curve["time"].to_numpy(float)
        curve_mean = curve["mean"].to_numpy(float)
        curve_std = curve["std"].fillna(0.0).to_numpy(float)
        ax_curve.plot(curve_times, curve_mean, color=color, linewidth=2.4, solid_capstyle="round")
        ax_curve.fill_between(
            curve_times,
            curve_mean - curve_std,
            curve_mean + curve_std,
            color=color,
            alpha=0.14,
        )
        ax_curve.scatter(
            curve_times,
            curve_mean,
            color=color,
            s=18,
            zorder=3,
            edgecolor=PANEL_BG,
            linewidth=0.45,
        )
        ax_curve.axhline(0, color=TEXT_MID, linewidth=0.8, alpha=0.32)
        ax_curve.set_title(f"Pattern {program}", loc="left", fontsize=12, color=TEXT_DARK, pad=5)
        ax_curve.set_xlim(times.min() - 0.15, times.max() + 0.15)
        ax_curve.set_ylim(-2.65, 2.15)
        ax_curve.set_ylabel("Program z" if program == 1 else "")
        if program == 2:
            ax_curve.set_yticklabels([])
        ax_curve.tick_params(axis="x", labelbottom=False)
        ax_curve.tick_params(axis="y", labelsize=8, colors=TEXT_MID)
        ax_curve.grid(True, axis="y", alpha=0.22, linestyle="-", color=GRID_COLOR)
        peak_time = float(curve.loc[curve["mean"].idxmax(), "time"])
        n_genes = int(curve["n_profiles"].iloc[0])
        ax_curve.text(
            0.01,
            0.05,
            f"peak {peak_time:.2f}\n{n_genes} genes",
            transform=ax_curve.transAxes,
            ha="left",
            va="bottom",
            fontsize=8,
            color=TEXT_MID,
        )

        genes = representatives.loc[representatives["program"] == program, "gene"].astype(str).tolist()
        heat = zscore.loc[genes].copy()
        heat = (
            heat.assign(_peak=heat.idxmax(axis=1).astype(float))
            .sort_values("_peak", kind="stable")
            .drop(columns="_peak")
        )
        ax_heat = fig.add_axes(heat_boxes[program])
        sns.heatmap(
            heat,
            ax=ax_heat,
            cmap=HEATMAP_CMAP,
            center=0,
            vmin=-HEATMAP_CLIP,
            vmax=HEATMAP_CLIP,
            cbar=False,
            linewidths=0.5,
            linecolor="#f1ece4",
        )
        ax_heat.set_title(
            ", ".join(heat.index[:3]),
            fontsize=8.5,
            loc="left",
            pad=8,
            color=TEXT_MID,
        )
        ax_heat.set_xlabel("Time")
        ax_heat.set_ylabel("")
        ax_heat.set_yticklabels(
            ax_heat.get_yticklabels(), rotation=0, fontsize=8, fontstyle="italic"
        )
        ax_heat.tick_params(axis="y", length=0, pad=8)
        sparse_time_ticks(ax_heat, times)
        heat_axes.append(ax_heat)

    cax = fig.add_axes([0.807, 0.105, 0.012, 0.195])
    # Use a PDF-native Gouraud mesh instead of Matplotlib's Colorbar, which
    # otherwise converts continuous color strips into an embedded bitmap.
    color_values = np.linspace(-HEATMAP_CLIP, HEATMAP_CLIP, 256)
    grid_x, grid_y = np.meshgrid(np.asarray([0.0, 1.0]), color_values)
    color_grid = np.repeat(color_values[:, None], 2, axis=1)
    cax.pcolormesh(
        grid_x,
        grid_y,
        color_grid,
        cmap=HEATMAP_CMAP,
        norm=Normalize(-HEATMAP_CLIP, HEATMAP_CLIP),
        shading="gouraud",
        rasterized=False,
    )
    cax.set_xlim(0.0, 1.0)
    cax.set_ylim(-HEATMAP_CLIP, HEATMAP_CLIP)
    cax.set_xticks([])
    cax.set_yticks(np.arange(-1.5, 1.51, 0.5))
    cax.yaxis.tick_right()
    cax.yaxis.set_label_position("right")
    cax.set_ylabel("Gene-wise z-score", fontsize=9)
    cax.tick_params(axis="y", labelsize=8)
    for spine in cax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color("black")

    output_dir.mkdir(parents=True, exist_ok=False)
    stem = "Figure_S8_MOSTA_latest_package_brain_gene_programs_exact_submitted_style"
    outputs = {
        "pdf": output_dir / f"{stem}.pdf",
        "svg": output_dir / f"{stem}.svg",
        "png": output_dir / f"{stem}.png",
    }
    fig.savefig(outputs["pdf"], format="pdf", facecolor=PANEL_BG, bbox_inches=None)
    fig.savefig(outputs["svg"], format="svg", facecolor=PANEL_BG, bbox_inches=None)
    fig.savefig(outputs["png"], format="png", dpi=200, facecolor=PANEL_BG, bbox_inches=None)
    plt.close(fig)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--style-reference", type=Path, required=True)
    args = parser.parse_args()

    inputs = load_inputs(args.input_dir.resolve())
    outputs = render(inputs, args.output_dir.resolve())
    manifest = {
        "figure": "SI Figure S8a-b",
        "status": "rendered_from_independently_audited_latest_package_tables",
        "numerical_contract": {
            "cells": "Brain cells from 13 fully generated global-t0 states",
            "trajectory": "50,000 fixed persistent particles from global t0; no observed-data restart",
            "genes": "2,000 original statistical HVGs; 747 LR-required additions excluded",
            "reconstruction": "persisted reference pca_center; per-cell inverse PCA; clip at zero before arithmetic mean",
            "standardization": "gene-wise population z-score (ddof=0)",
            "clustering": "Ward Euclidean exact cut_tree k=2; program IDs ordered by prototype peak time",
            "program_sizes": {"1": 883, "2": 1117},
        },
        "style_contract": {
            "primary": identity(args.style_reference.resolve()),
            "grammar": "submitted SI page-34 embedded JPEG plus historical notebook renderer",
            "panel_a_display_selection": "first 25 genes of audited temporal-variance ranking, matching the submitted notebook",
            "top_palette": "seaborn tab10 requested with n_colors=25",
            "program_colors": {str(key): value for key, value in PROGRAM_COLORS.items()},
            "heatmap": {"cmap": HEATMAP_CMAP, "vmin": -HEATMAP_CLIP, "vmax": HEATMAP_CLIP},
            "canvas_inches": list(FIGSIZE_IN),
            "raster_preview_pixels": [1686, 2200],
            "coordinate_warp": False,
        },
        "inputs": {key: identity(path) for key, path in inputs["paths"].items()},
        "outputs": {key: identity(path) for key, path in outputs.items()},
        "software": {
            "matplotlib": mpl.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "seaborn": sns.__version__,
        },
    }
    manifest_path = args.output_dir / "render_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"outputs": {key: str(path) for key, path in outputs.items()}, "manifest": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
