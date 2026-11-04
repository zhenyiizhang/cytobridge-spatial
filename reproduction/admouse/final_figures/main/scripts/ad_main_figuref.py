#!/usr/bin/env python3
"""Draw Figure 6f from saved Trem2 perturbation states and cell-type labels."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.collections import PathCollection
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
DATA = ROOT / "data" / "ad_main_figuref"
# In the displayed, counterclockwise-rotated coordinate system, this moves
# the fixed ROI slightly left and upward while preserving its size.
DISPLAY_ROI_SHIFT = 0.2
FOCUS_CELLTYPES = {
    "Excitatory neurons",
    "Inhibitory neurons",
    "Microglia",
}
BACKGROUND_CELLTYPE_COLOR = "#d9d9d9"
SOURCE_SCRIPT = Path(__file__).with_name("admouse_trem2_plot.py")


def load_source_module():
    spec = importlib.util.spec_from_file_location("trem2_scale1_full_figure", SOURCE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def rotate_cases_counterclockwise(cases, spatial_window, source):
    """Return display-only spatial cases rotated 90 degrees counterclockwise.

    The underlying saved states are never changed.  Rotating the coordinates,
    shared window, and fixed ROI together keeps cells, sparse edges, and the
    ROI outline aligned in each condition.
    """
    rotated_cases = {}
    for condition, case in cases.items():
        rotated = dict(case)
        xy = case["xy"]
        rotated["xy"] = np.column_stack((-xy[:, 1], xy[:, 0]))
        rotated_cases[condition] = rotated

    x_min, x_max, y_min, y_max = spatial_window
    rotated_window = (-y_max, -y_min, x_min, x_max)
    roi_x, roi_y = source.ROI_CENTER
    # (x, y) -> (-y, x): increasing both original coordinates moves the
    # displayed ROI left and up after the counterclockwise rotation.
    roi_x += DISPLAY_ROI_SHIFT
    roi_y += DISPLAY_ROI_SHIFT
    source.ROI_CENTER = (-roi_y, roi_x)
    return rotated_cases, rotated_window


def draw_panel_a(
    source,
    cases,
    composition,
    spatial_window,
    attention_vmax,
    output_stem="ad_main_figuref",
    focus_other_alpha: float | None = None,
    composition_colors: dict[str, str] | None = None,
) -> None:
    if composition_colors is None:
        composition_colors = source.CELLTYPE_COLORS
    fig = plt.figure(figsize=(8.27, 6.45))
    grid = fig.add_gridspec(
        4,
        6,
        height_ratios=[0.25, 1.55, 1.55, 0.62],
        left=0.08,
        right=0.98,
        bottom=0.07,
        top=0.98,
        hspace=0.42,
        wspace=0.55,
    )
    source.panel_heading(
        fig.add_subplot(grid[0, :]),
        "f",
        "Perturbation of cell–cell interactions",
    )
    cases, spatial_window = rotate_cases_counterclockwise(
        cases, spatial_window, source
    )
    for column, condition in enumerate(source.SPATIAL_CONDITIONS):
        full_axis = fig.add_subplot(grid[1, 2 * column : 2 * column + 2])
        source.plot_spatial_case(
            full_axis,
            cases[condition],
            source.SPATIAL_TITLES[condition],
            spatial_window,
            attention_vmax,
            roi=False,
        )
        if focus_other_alpha is not None:
            for celltype, collection in zip(source.CELLTYPE_ORDER, full_axis.collections):
                if isinstance(collection, PathCollection):
                    collection.set_alpha(1.0 if celltype == "Microglia" else focus_other_alpha)
        for text in full_axis.texts[:]:
            if text.get_text().startswith("No saved Microglia"):
                text.remove()
        # One condition-specific composition bar beside each full spatial view.
        # Heights are the endpoint fractions used in panel b; no values are
        # altered or recalculated for visualization.
        composition_axis = full_axis.inset_axes([1.035, 0.10, 0.040, 0.80])
        subset = composition.loc[
            composition["condition"].eq(condition)
        ].set_index("celltype")
        bottom = 0.0
        for celltype in source.CELLTYPE_ORDER:
            percentage = 100.0 * float(subset.loc[celltype, "fraction"])
            composition_axis.bar(
                0,
                percentage,
                bottom=bottom,
                width=1.0,
                color=composition_colors[celltype],
                edgecolor="none",
                linewidth=0,
            )
            bottom += percentage
        composition_axis.set_xlim(-0.5, 0.5)
        composition_axis.set_ylim(0, 100)
        composition_axis.set_xticks([])
        composition_axis.set_yticks([])
        for spine in composition_axis.spines.values():
            spine.set_color("black")
            spine.set_linewidth(0.65)
        composition_axis.yaxis.set_label_position("right")
        composition_axis.set_ylabel("Cell proportion (%)", fontsize=7.2, labelpad=2)
        roi_axis = fig.add_subplot(grid[2, 2 * column : 2 * column + 2])
        source.plot_spatial_case(
            roi_axis,
            cases[condition],
            f"{source.SPATIAL_TITLES[condition]} | fixed ROI",
            spatial_window,
            attention_vmax,
            roi=True,
        )
        if focus_other_alpha is not None:
            for celltype, collection in zip(source.CELLTYPE_ORDER, roi_axis.collections):
                if isinstance(collection, PathCollection):
                    collection.set_alpha(1.0 if celltype == "Microglia" else focus_other_alpha)
        # The ROI panels retain the points and ROI-specific view but omit the
        # repeated no-edge notice used for the full-tissue diagnostic panels.
        for text in roi_axis.texts[:]:
            if text.get_text().startswith("No saved Microglia"):
                text.remove()
        for axis in (full_axis, roi_axis):
            for collection in axis.collections:
                if isinstance(collection, PathCollection):
                    collection.set_rasterized(True)

    legend_axis = fig.add_subplot(grid[3, :])
    legend_axis.axis("off")
    legend_axis.legend(
        handles=[
            Patch(facecolor=composition_colors[celltype], label=celltype)
            for celltype in source.CELLTYPE_ORDER
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=4,
        frameon=False,
        handlelength=1.0,
        handletextpad=0.35,
        columnspacing=0.9,
    )
    stem = FIGURES / output_stem
    fig.savefig(stem.with_suffix(".png"), dpi=900)
    fig.savefig(stem.with_suffix(".pdf"), dpi=900)
    plt.close(fig)


def draw_panel_b(source, composition) -> None:
    fig = plt.figure(figsize=(6.8, 5.2))
    grid = fig.add_gridspec(
        3,
        3,
        height_ratios=[0.26, 1.55, 0.58],
        left=0.12,
        right=0.97,
        bottom=0.07,
        top=0.97,
        hspace=0.36,
        wspace=0.55,
    )
    source.panel_heading(
        fig.add_subplot(grid[0, :]), "b", "Endpoint cell-type composition | t=2.5"
    )
    condition_order = ["baseline", "low", "high"]
    condition_titles = {
        "baseline": "Baseline",
        "low": "Trem2 low",
        "high": "Trem2 high",
    }
    for column, condition in enumerate(condition_order):
        axis = fig.add_subplot(grid[1, column])
        subset = composition.loc[composition["condition"] == condition].set_index("celltype")
        bottom = 0.0
        for celltype in source.CELLTYPE_ORDER:
            percentage = 100.0 * float(subset.loc[celltype, "fraction"])
            axis.bar(
                0,
                percentage,
                bottom=bottom,
                width=1.0,
                color=source.CELLTYPE_COLORS[celltype],
                edgecolor="none",
                linewidth=0,
            )
            bottom += percentage
        axis.set_xlim(-0.5, 0.5)
        axis.set_ylim(0, 100)
        axis.set_xticks([])
        axis.set_yticks([0, 20, 40, 60, 80, 100])
        axis.set_title(condition_titles[condition], fontsize=13, pad=8)
        axis.set_ylabel("Cell proportion (%)")
        axis.tick_params(axis="y", direction="out", width=0.9, length=3.5, pad=3)
        for spine in axis.spines.values():
            spine.set_color("black")
            spine.set_linewidth(0.9)

    legend_axis = fig.add_subplot(grid[2, :])
    legend_axis.axis("off")
    legend_axis.legend(
        handles=[
            Patch(facecolor=source.CELLTYPE_COLORS[celltype], label=celltype)
            for celltype in source.CELLTYPE_ORDER
        ],
        loc="upper center",
        ncol=2,
        frameon=False,
        handlelength=1.0,
        handletextpad=0.35,
        columnspacing=0.9,
    )
    stem = FIGURES / "admouse_trem2_whole_tissue_scale1_panel_b_composition"
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)


def draw_figure(source, output_dir: Path) -> None:
    """Draw the accepted neuron/microglia view and tabulate all cell types."""
    global FIGURES
    FIGURES = Path(output_dir)
    FIGURES.mkdir(parents=True, exist_ok=True)
    source.apply_style()
    composition_colors = dict(source.CELLTYPE_COLORS)
    source.CELLTYPE_COLORS = {
        celltype: color if celltype in FOCUS_CELLTYPES else BACKGROUND_CELLTYPE_COLOR
        for celltype, color in source.CELLTYPE_COLORS.items()
    }
    source.CELLTYPE_ORDER = [
        celltype for celltype in source.CELLTYPE_ORDER
        if celltype not in FOCUS_CELLTYPES
    ] + ["Excitatory neurons", "Inhibitory neurons", "Microglia"]
    cases = {
        condition: source.load_case("whole_tissue", condition)
        for condition in source.SPATIAL_CONDITIONS
    }
    rows = []
    for condition, case in cases.items():
        labels = case["labels"]
        counts = pd.Series(labels).value_counts()
        for celltype in source.CELLTYPE_ORDER:
            count = int(counts.get(celltype, 0))
            rows.append(dict(condition=condition,
                condition_label=source.SPATIAL_TITLES[condition],
                celltype=celltype, count=count, fraction=count / len(labels),
                total=len(labels)))
    composition = pd.DataFrame(rows)
    tables = FIGURES / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    composition.to_csv(tables / "whole_tissue_composition_endpoint.csv", index=False)
    spatial_window = source.spatial_limits({"whole_tissue": cases})
    selected = [source.microglia_neuron_edges(case)[2] for case in cases.values()]
    nonempty = [values for values in selected if len(values)]
    if not nonempty:
        nonempty = [case["attention"] for case in cases.values() if len(case["attention"])]
    vmax = float(np.quantile(np.concatenate(nonempty), .995)) if nonempty else 1.0
    draw_panel_a(source, cases, composition, spatial_window, vmax,
                 focus_other_alpha=0.25, composition_colors=composition_colors)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/admouse"),
                        help="AD data directory containing trem2/figure_source.")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    data = args.data_dir.resolve()
    output = args.output_dir.resolve()
    if output == data or data in output.parents or output == ROOT or ROOT in output.parents:
        parser.error("Choose an output directory outside the input data and source archive.")
    source = load_source_module()
    source.SOURCE_DIR = data / "trem2/figure_source"
    draw_figure(source, output)


if __name__ == "__main__":
    main()
