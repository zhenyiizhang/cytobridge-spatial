#!/usr/bin/env python3
"""Build a one-page A4 figure for a finite-range attraction benchmark run.

The figure is deliberately rebuilt from the archived H5AD, rollout NPZ files,
and metric CSV files.  It does not raster-combine earlier diagnostic panels.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import anndata as ad
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = (
    ROOT
    / "results/spatial_synthetic_attraction_v11c_realdata_epoch_allocation_20260810"
)
EVAL_ROOT = RUN_ROOT / "evaluation_fixed400_five_seed"
FIGURE_ROOT = RUN_ROOT / "manuscript_figure"
PDF_OUTPUT = ROOT / "output/pdf/finite_range_cell_cell_attraction_benchmark.pdf"

GT_COLOR = "#4D4D4D"
LEARNED_COLOR = "#007C91"
OFF_COLOR = "#D55E00"
GRID_COLOR = "#E3E6E8"
TEXT_COLOR = "#222222"
HEADING_COLOR = "#222222"

# Avoid the fluorescent yellow endpoint of full-range viridis on a white page.
_VIRIDIS = mpl.colormaps["viridis"]
TIME_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "viridis_manuscript",
    _VIRIDIS(np.linspace(0.08, 0.85, 256)),
)

SPATIAL_LIMITS = ((0.10, 0.92), (0.12, 0.90))
GENE_LIMITS = ((-0.68, 0.68), (-0.42, 0.42))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8.5,
            "axes.titlesize": 8.7,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 8.2,
            "ytick.labelsize": 8.2,
            "legend.fontsize": 8.3,
            "axes.linewidth": 0.60,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "text.color": TEXT_COLOR,
            "axes.labelcolor": TEXT_COLOR,
            "axes.titlecolor": TEXT_COLOR,
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
        }
    )


def clean_axis(ax: mpl.axes.Axes, *, grid: bool = False) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#4A4A4A")
    ax.spines["bottom"].set_color("#4A4A4A")
    if grid:
        ax.grid(axis="y", color=GRID_COLOR, linewidth=0.42, alpha=0.65, zorder=0)
    else:
        ax.grid(False)
    ax.set_axisbelow(True)


def group_heading(ax: mpl.axes.Axes, label: str, title: str) -> None:
    ax.axis("off")
    ax.text(
        0.0,
        0.55,
        label,
        fontsize=11.5,
        fontweight="bold",
        va="center",
        ha="left",
        color=HEADING_COLOR,
    )
    ax.text(
        0.038,
        0.55,
        title,
        fontsize=10.3,
        fontweight="semibold",
        va="center",
        ha="left",
        color=HEADING_COLOR,
    )


def as_dense(array) -> np.ndarray:
    if hasattr(array, "toarray"):
        return np.asarray(array.toarray())
    return np.asarray(array)


def draw_observed_snapshots(fig: plt.Figure, slot) -> None:
    panel = slot.subgridspec(3, 5, height_ratios=[0.25, 1.0, 1.0], hspace=0.20, wspace=0.17)
    heading = fig.add_subplot(panel[0, :])
    group_heading(heading, "a", "Spatiotemporal snapshots")

    adata = ad.read_h5ad(RUN_ROOT / "training/model/adata.h5ad")
    times = np.asarray(adata.obs["samples"], dtype=float)
    spatial = np.asarray(adata.obsm["spatial_aligned"], dtype=float)
    gene = as_dense(adata.X).astype(float)
    unique_times = np.sort(np.unique(times))

    for col, time in enumerate(unique_times):
        mask = np.isclose(times, time)
        color = TIME_CMAP(time / unique_times.max())
        for row, values, limits in (
            (1, spatial, SPATIAL_LIMITS),
            (2, gene, GENE_LIMITS),
        ):
            ax = fig.add_subplot(panel[row, col])
            ax.scatter(
                values[mask, 0],
                values[mask, 1],
                s=2.45,
                color=color,
                alpha=0.72,
                edgecolors="none",
            )
            ax.set_xlim(*limits[0])
            ax.set_ylim(*limits[1])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal", adjustable="box")
            for spine in ax.spines.values():
                spine.set_color("#C8CDD1")
                spine.set_linewidth(0.45)
            if row == 1:
                ax.set_title(
                    f"t = {time:g}\nn = {mask.sum()}",
                    pad=1.5,
                    fontsize=8.4,
                    linespacing=0.95,
                )
            if col == 0:
                ax.set_ylabel("Spatial" if row == 1 else "Gene", labelpad=3.2, fontweight="semibold")


def trajectory_axis(
    ax: mpl.axes.Axes,
    states: np.ndarray,
    time_points: np.ndarray,
    dims: tuple[int, int],
    limits: tuple[tuple[float, float], tuple[float, float]],
    *,
    title: str,
    selected_cells: np.ndarray,
) -> None:
    norm = mpl.colors.Normalize(vmin=float(time_points.min()), vmax=float(time_points.max()))
    values = states[:, selected_cells][:, :, list(dims)]
    for cell in range(values.shape[1]):
        xy = values[:, cell, :]
        segments = np.stack([xy[:-1], xy[1:]], axis=1)
        collection = LineCollection(
            segments,
            cmap=TIME_CMAP,
            norm=norm,
            linewidths=0.36,
            alpha=0.27,
            zorder=1,
        )
        collection.set_array(time_points[:-1])
        ax.add_collection(collection)
    ax.scatter(
        values[0, :, 0],
        values[0, :, 1],
        s=4.0,
        color=TIME_CMAP(0.0),
        alpha=0.78,
        edgecolors="none",
        zorder=3,
    )
    ax.scatter(
        values[-1, :, 0],
        values[-1, :, 1],
        s=4.5,
        color=TIME_CMAP(1.0),
        alpha=0.84,
        edgecolors="#4D4D4D",
        linewidths=0.14,
        zorder=4,
    )
    ax.set_xlim(*limits[0])
    ax.set_ylim(*limits[1])
    ax.set_title(title, pad=2.0)
    clean_axis(ax)


def draw_trajectories(fig: plt.Figure, slot) -> None:
    panel = slot.subgridspec(3, 2, height_ratios=[0.11, 1.0, 1.0], hspace=0.27, wspace=0.18)
    heading = fig.add_subplot(panel[0, :])
    group_heading(
        heading,
        "b",
        "Cell-state trajectories",
    )

    dense = np.load(EVAL_ROOT / "dense_rollout_seed_1.npz")
    time_points = dense["time_points"]
    cell_count = dense["ground_truth"].shape[1]
    rng = np.random.default_rng(11)
    selected = np.sort(rng.choice(cell_count, size=60, replace=False))
    spaces = [
        ("Spatial", (0, 1), SPATIAL_LIMITS, ("spatial x", "spatial y")),
        ("Gene", (2, 3), GENE_LIMITS, ("gene 1", "gene 2")),
    ]
    models = [("Ground truth", dense["ground_truth"]), ("CytoBridge", dense["predicted"])]
    for row, (space_name, dims, limits, labels) in enumerate(spaces, start=1):
        for col, (model_name, states) in enumerate(models):
            ax = fig.add_subplot(panel[row, col])
            trajectory_axis(
                ax,
                states,
                time_points,
                dims,
                limits,
                title=model_name if row == 1 else "",
                selected_cells=selected,
            )
            if col == 0:
                ax.set_ylabel(labels[1])
            else:
                ax.set_yticklabels([])
            if row == 2:
                ax.set_xlabel(labels[0])
            else:
                ax.set_xticklabels([])

    legend = [
        Line2D([], [], marker="o", linestyle="none", markersize=3.4, color=TIME_CMAP(0.0), label="t = 0"),
        Line2D([], [], marker="o", linestyle="none", markersize=3.4, color=TIME_CMAP(1.0), label="t = 4"),
    ]
    heading.legend(
        handles=legend,
        loc="center right",
        frameon=False,
        ncol=2,
        handletextpad=0.45,
        columnspacing=1.2,
        bbox_to_anchor=(1.0, 0.55),
    )


def attraction_potential(distance: np.ndarray, coefficient: np.ndarray, cutoff: float = 0.30) -> tuple[np.ndarray, np.ndarray]:
    mask = distance <= cutoff
    radius = np.asarray(distance[mask], dtype=float)
    values = np.asarray(coefficient[mask], dtype=float).copy()
    order = np.argsort(radius)
    radius = radius[order]
    values = values[order]
    at_cutoff = np.isclose(radius, cutoff, atol=1e-12, rtol=0.0)
    interior = np.flatnonzero(radius < cutoff)
    if np.any(at_cutoff) and interior.size:
        values[at_cutoff] = values[interior[-1]]
    segments = 0.5 * (values[:-1] + values[1:]) * np.diff(radius)
    integral_to_cutoff = np.zeros_like(radius)
    integral_to_cutoff[:-1] = np.cumsum(segments[::-1])[::-1]
    potential = -integral_to_cutoff
    potential[at_cutoff] = 0.0
    return radius, potential


def draw_growth_and_interaction(fig: plt.Figure, slot) -> None:
    panel = slot.subgridspec(2, 3, height_ratios=[0.10, 1.0], hspace=0.04, wspace=0.39)

    headings = (
        ("C", "Population growth"),
        ("D", "Radial cell-cell attraction"),
        ("E", "Interaction potential"),
    )
    for col, (label, title) in enumerate(headings):
        heading = fig.add_subplot(panel[0, col])
        heading.axis("off")
        heading.text(0.0, 0.48, label.lower(), fontsize=11.5, fontweight="bold", color=HEADING_COLOR, va="center")
        heading.text(0.11, 0.48, title, fontsize=10.3, fontweight="semibold", color=HEADING_COLOR, va="center")

    growth_rows = read_csv(EVAL_ROOT / "growth_mass_metrics.csv")
    growth_times = sorted({float(row["time"]) for row in growth_rows})
    observed = []
    learned_mean = []
    learned_std = []
    for time in growth_times:
        rows = [row for row in growth_rows if float(row["time"]) == time]
        observed.append(float(rows[0]["observed_relative_mass"]))
        learned_values = np.asarray([float(row["predicted_relative_mass"]) for row in rows])
        learned_mean.append(float(learned_values.mean()))
        learned_std.append(float(learned_values.std(ddof=1)))
    ax = fig.add_subplot(panel[1, 0])
    ax.plot(growth_times, observed, color=GT_COLOR, marker="o", markersize=3.4, linewidth=1.5, label="Ground truth")
    ax.errorbar(
        growth_times,
        learned_mean,
        yerr=learned_std,
        color=LEARNED_COLOR,
        marker="o",
        markersize=3.4,
        linewidth=1.5,
        capsize=1.8,
        label="CytoBridge",
    )
    ax.set_xlabel("time")
    ax.set_ylabel("relative total mass")
    ax.set_xticks(growth_times)
    ax.set_ylim(1.0, 1.50)
    ax.set_yticks([1.0, 1.1, 1.2, 1.3, 1.4, 1.5])
    ax.legend(frameon=False, loc="upper left", handlelength=1.8, handletextpad=0.5)
    clean_axis(ax, grid=True)

    curve_rows = read_csv(EVAL_ROOT / "interaction_radial_curve.csv")
    distance = np.asarray([float(row["distance"]) for row in curve_rows])
    true_coefficient = np.asarray([float(row["true_coefficient"]) for row in curve_rows])
    learned_coefficient = np.asarray([float(row["learned_coefficient"]) for row in curve_rows])
    ax = fig.add_subplot(panel[1, 1])
    ax.plot(distance, true_coefficient, color=GT_COLOR, linewidth=1.65, label="Ground truth")
    learned_interior = distance < 0.30
    ax.plot(
        distance[learned_interior],
        learned_coefficient[learned_interior],
        color=LEARNED_COLOR,
        linewidth=1.65,
        label="CytoBridge",
    )
    ax.set_xlim(0, 0.30)
    ax.set_ylim(-0.16, 0.55)
    ax.set_xlabel("pair distance r")
    ax.set_ylabel("radial coefficient c(r)")
    clean_axis(ax, grid=False)

    radius, gt_potential = attraction_potential(distance, true_coefficient)
    _, learned_potential = attraction_potential(distance, learned_coefficient)
    ax = fig.add_subplot(panel[1, 2])
    ax.plot(
        radius,
        gt_potential,
        color=GT_COLOR,
        linewidth=1.65,
        solid_capstyle="butt",
        clip_on=True,
        label="Ground truth",
    )
    ax.plot(
        radius,
        learned_potential,
        color=LEARNED_COLOR,
        linewidth=1.65,
        solid_capstyle="butt",
        clip_on=True,
        label="CytoBridge",
    )
    ax.set_xlim(0, 0.30)
    ax.set_ylim(-0.10, 0.02)
    ax.set_xlabel("pair distance r")
    ax.set_ylabel("U(r), U(cutoff) = 0")
    clean_axis(ax, grid=False)


def draw_ablation(fig: plt.Figure, slot) -> None:
    panel = slot.subgridspec(2, 3, height_ratios=[0.15, 1.0], hspace=0.10, wspace=0.36)
    heading = fig.add_subplot(panel[0, :])
    group_heading(
        heading,
        "f",
        "Effect of cell-cell interactions",
    )
    heading.legend(
        handles=[
            Line2D([], [], color=LEARNED_COLOR, marker="o", markersize=3.2, linewidth=1.4, label="With interaction"),
            Line2D([], [], color=OFF_COLOR, marker="o", markersize=3.2, linewidth=1.4, label="Without interaction"),
        ],
        loc="center right",
        frameon=False,
        ncol=2,
        handlelength=1.5,
        handletextpad=0.45,
        columnspacing=1.0,
        bbox_to_anchor=(1.0, 0.55),
    )
    rows = read_csv(EVAL_ROOT / "interaction_ablation_metrics.csv")
    spaces = [("joint", "Joint state"), ("spatial", "Spatial"), ("gene", "Gene expression")]
    for col, (space, label) in enumerate(spaces):
        ax = fig.add_subplot(panel[1, col])
        for condition, color, legend_label in (
            ("interaction_on", LEARNED_COLOR, "With interaction"),
            ("interaction_off", OFF_COLOR, "Without interaction"),
        ):
            means = []
            standard_errors = []
            times = [1.0, 2.0, 3.0, 4.0]
            for time in times:
                values = np.asarray(
                    [
                        float(row["w1"])
                        for row in rows
                        if row["space"] == space
                        and row["condition"] == condition
                        and float(row["time"]) == time
                    ]
                )
                means.append(float(values.mean()))
                standard_errors.append(float(values.std(ddof=1) / np.sqrt(len(values))))
            ax.errorbar(
                times,
                means,
                yerr=standard_errors,
                color=color,
                marker="o",
                markersize=3.0,
                linewidth=1.4,
                capsize=1.4,
                label=legend_label,
            )
        ax.set_title(label, fontsize=8.7, pad=3.0)
        ax.set_xlabel("time")
        ax.set_ylabel("W1" if col == 0 else "")
        ax.set_xticks([1, 2, 3, 4])
        clean_axis(ax, grid=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=RUN_ROOT,
        help="Run directory containing training/model and evaluation_fixed400_five_seed.",
    )
    parser.add_argument(
        "--pdf-output",
        type=Path,
        default=PDF_OUTPUT,
        help="Additional PDF copy written outside the run directory.",
    )
    parser.add_argument(
        "--output-stem",
        default="finite_range_cell_cell_attraction_benchmark",
        help="Filename stem for the PDF and PNG written under manuscript_figure.",
    )
    return parser


def main() -> None:
    global RUN_ROOT, EVAL_ROOT, FIGURE_ROOT, PDF_OUTPUT
    args = build_parser().parse_args()
    RUN_ROOT = args.run_root.expanduser().resolve()
    EVAL_ROOT = RUN_ROOT / "evaluation_fixed400_five_seed"
    FIGURE_ROOT = RUN_ROOT / "manuscript_figure"
    PDF_OUTPUT = args.pdf_output.expanduser().resolve()

    configure_style()
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    PDF_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
    outer = fig.add_gridspec(
        4,
        1,
        height_ratios=[2.38, 3.18, 2.02, 1.82],
        left=0.070,
        right=0.980,
        top=0.975,
        bottom=0.055,
        hspace=0.14,
    )

    draw_observed_snapshots(fig, outer[0, 0])
    draw_trajectories(fig, outer[1, 0])
    draw_growth_and_interaction(fig, outer[2, 0])
    draw_ablation(fig, outer[3, 0])

    png_path = FIGURE_ROOT / f"{args.output_stem}.png"
    pdf_path = FIGURE_ROOT / f"{args.output_stem}.pdf"
    fig.savefig(png_path, dpi=320, facecolor="white")
    fig.savefig(pdf_path, facecolor="white")
    fig.savefig(PDF_OUTPUT, facecolor="white")
    plt.close(fig)
    print(png_path)
    print(pdf_path)
    print(PDF_OUTPUT)


if __name__ == "__main__":
    main()
