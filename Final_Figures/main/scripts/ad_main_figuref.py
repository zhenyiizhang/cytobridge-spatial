#!/usr/bin/env python3
"""Render standalone panels a and b from the Trem2 whole-tissue scale-1.0 run."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.collections import PathCollection
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT.parent
FIGURES = ROOT / "figures"
DATA = ROOT / "data" / "ad_main_figuref"
SOURCE_SCRIPT = (
    DATA_ROOT
    / "admouse_0815/trem2_whole_tissue_scale1_20260820/scripts"
    / "make_admouse_scale1_whole_tissue_figure.py"
)


def load_source_module():
    spec = importlib.util.spec_from_file_location("trem2_scale1_full_figure", SOURCE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def draw_panel_a(source, cases, composition, spatial_window, attention_vmax) -> None:
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
        "a",
        "Whole-tissue Trem2 sensitivity | scale 1.0 | formal r2 model",
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
                color=source.CELLTYPE_COLORS[celltype],
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
        for axis in (full_axis, roi_axis):
            for collection in axis.collections:
                if isinstance(collection, PathCollection):
                    collection.set_rasterized(True)

    legend_axis = fig.add_subplot(grid[3, :])
    legend_axis.axis("off")
    legend_axis.legend(
        handles=[
            Patch(facecolor=source.CELLTYPE_COLORS[celltype], label=celltype)
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
    colorbar_axis = legend_axis.inset_axes([0.35, 0.02, 0.30, 0.15])
    colorbar = mpl.colorbar.ColorbarBase(
        colorbar_axis,
        cmap=plt.get_cmap("Greys"),
        norm=Normalize(0, attention_vmax),
        orientation="horizontal",
    )
    colorbar.set_label("Saved GNN edge attention (shared scale)", fontsize=8)
    colorbar.ax.tick_params(labelsize=7.5, length=2, pad=1)

    stem = FIGURES / "admouse_trem2_whole_tissue_scale1_panel_a_spatial"
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


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    source = load_source_module()
    source.apply_style()
    cases = {
        condition: source.load_case("whole_tissue", condition)
        for condition in source.SPATIAL_CONDITIONS
    }
    tables = source.prepare_scope_tables("whole_tissue")
    spatial_window = source.spatial_limits({"whole_tissue": cases})
    limits = source.shared_plot_limits(
        {"whole_tissue": cases}, {"whole_tissue": tables}
    )
    tables["composition"].to_csv(DATA / "composition_endpoint_t2p5.csv", index=False)
    draw_panel_a(
        source, cases, tables["composition"], spatial_window,
        limits["spatial_attention_vmax"]
    )
    draw_panel_b(source, tables["composition"])


if __name__ == "__main__":
    main()
