"""Matplotlib renderer for ARISTA local interaction domains."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib as mpl
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
from scipy.spatial import ConvexHull

from ._io import prepare_output_dir
from .arista_local_domains import DOMAIN_ORDER

if TYPE_CHECKING:
    import pandas as pd

    from .arista_local_domains import AristaLocalDomainData, AristaLocalDomainPanels


TEXT = "#111111"
GRID = "#D9E8F0"
REFERENCE = "#214761"
NULL = "#7FA9C4"
DOMAIN_COLORS = {
    "N1_sfrpEGC_VLMC": "#CC6677",
    "N2_reaEGC_wntEGC": "#07838B",
}
DOMAIN_SHORT = {
    "N1_sfrpEGC_VLMC": "sfrpEGC–VLMC",
    "N2_reaEGC_wntEGC": "reaEGC–wntEGC",
}
DOMAIN_ROLES = {
    "N1_sfrpEGC_VLMC": "Matrix/trophic program",
    "N2_reaEGC_wntEGC": "Reactive/adhesion program",
}
ARISTA_RC = {
    "font.family": "Arial",
    "font.size": 9.0,
    "axes.titlesize": 9.0,
    "axes.labelsize": 9.0,
    "axes.linewidth": 0.65,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.axisbelow": True,
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "xtick.labelsize": 9.0,
    "ytick.labelsize": 9.0,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "legend.fontsize": 9.0,
    "legend.frameon": False,
    "lines.linewidth": 1.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "text.color": TEXT,
    "axes.labelcolor": TEXT,
    "axes.titlecolor": TEXT,
    "xtick.color": TEXT,
    "ytick.color": TEXT,
    "savefig.facecolor": "white",
    "savefig.dpi": 320,
}


def _heading(
    figure: plt.Figure,
    label: str,
    title: str,
    x: float,
    y: float,
) -> None:
    figure.text(
        x,
        y,
        label,
        fontsize=14,
        fontweight="bold",
        ha="left",
        va="center",
        color=TEXT,
    )
    figure.text(
        x + 0.037,
        y,
        title,
        fontsize=12,
        fontweight="bold",
        ha="left",
        va="center",
        color=TEXT,
    )


def _clean_axis(ax: plt.Axes, *, grid_axis: str | None = None) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    if grid_axis is None:
        ax.grid(False)
    else:
        ax.grid(
            axis=grid_axis,
            color=GRID,
            linewidth=0.5,
            alpha=0.7,
            zorder=0,
        )
    ax.set_axisbelow(True)


def _outline(ax: plt.Axes, points: np.ndarray, color: str) -> None:
    if points.shape[0] < 3:
        return
    hull = ConvexHull(points)
    polygon = np.vstack((points[hull.vertices], points[hull.vertices[0]]))
    ax.plot(polygon[:, 0], polygon[:, 1], color="white", linewidth=3.1, zorder=3)
    ax.plot(polygon[:, 0], polygon[:, 1], color=color, linewidth=1.8, zorder=4)


def _plot_spatial_map(ax: plt.Axes, cells: "pd.DataFrame") -> None:
    scatter = ax.scatter(
        cells["paper_x"],
        cells["paper_y"],
        c=cells["cosine_full_vs_interaction"],
        cmap="plasma",
        norm=TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0),
        s=10,
        linewidths=0,
        alpha=0.96,
        zorder=1,
    )
    for index, domain in enumerate(DOMAIN_ORDER, start=1):
        subset = cells.loc[cells["two_niche_region"].fillna("").eq(domain)]
        points = subset[["paper_x", "paper_y"]].to_numpy(float)
        _outline(ax, points, DOMAIN_COLORS[domain])
        centroid = np.median(points, axis=0)
        ax.text(
            centroid[0],
            centroid[1],
            f"N{index}",
            fontsize=9,
            fontweight="bold",
            color=DOMAIN_COLORS[domain],
            ha="center",
            va="center",
            zorder=5,
            path_effects=[path_effects.withStroke(linewidth=2.5, foreground="white")],
        )
    ax.set_aspect("equal")
    ax.set_axis_off()
    colorbar_axis = inset_axes(
        ax,
        width="54%",
        height="3.6%",
        loc="lower left",
        borderpad=0.7,
    )
    colorbar = plt.colorbar(scatter, cax=colorbar_axis, orientation="horizontal")
    colorbar.set_ticks([-1, 0, 1])
    colorbar.ax.tick_params(labelsize=8, length=2, pad=1)
    colorbar.set_label(
        "Full–interaction spatial-velocity cosine",
        fontsize=8,
        labelpad=1,
    )


def _plot_attention(ax: plt.Axes, summary: "pd.DataFrame") -> None:
    y_positions = np.array([1.0, 0.0])
    for y, domain in zip(y_positions, DOMAIN_ORDER):
        row = summary.loc[summary["niche"].eq(domain)].iloc[0]
        observed = float(row["observed_attention_per_cell"])
        null_mean = float(row["null_mean"])
        null_sd = float(row["null_sd"])
        fold = float(row["fold_over_null"])
        ax.errorbar(
            null_mean,
            y,
            xerr=null_sd,
            fmt="o",
            markersize=5.5,
            markerfacecolor=NULL,
            markeredgecolor=REFERENCE,
            markeredgewidth=0.6,
            ecolor=NULL,
            elinewidth=1.2,
            capsize=3,
            zorder=2,
        )
        ax.scatter(
            observed,
            y,
            marker="D",
            s=42,
            color=DOMAIN_COLORS[domain],
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
        )
        ax.text(
            observed + 0.10,
            y,
            f"{fold:.2f}×",
            fontsize=8.5,
            va="center",
            color=DOMAIN_COLORS[domain],
        )
    ax.set_yticks(y_positions, ["N1", "N2"])
    ax.set_xlim(0.55, 4.15)
    ax.set_ylim(-0.55, 1.55)
    ax.set_xlabel("Selected attention per domain cell")
    ax.set_title("Above composition-matched null", loc="left", pad=7)
    ax.tick_params(axis="y", length=0)
    _clean_axis(ax, grid_axis="x")
    ax.scatter([], [], marker="D", s=35, color=REFERENCE, label="Observed")
    ax.errorbar([], [], xerr=[], fmt="o", color=NULL, label="Matched null mean ± s.d.")
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        ["Observed domain", "Random same-composition cells"],
        loc="lower right",
        frameon=False,
        fontsize=8,
        handletextpad=0.5,
    )


def _plot_edge_structure(ax: plt.Axes, summary: "pd.DataFrame") -> None:
    ordered = summary.reset_index(drop=True)
    y = np.arange(len(ordered))[::-1]
    colors = [DOMAIN_COLORS[domain] for domain in ordered["niche"]]
    ax.barh(
        y,
        ordered["attention_percent"],
        height=0.58,
        color=colors,
        alpha=0.88,
    )
    for y_value, value in zip(y, ordered["attention_percent"].to_numpy(float)):
        ax.text(value + 1.2, y_value, f"{value:.0f}%", va="center", fontsize=8)
    ax.set_yticks(y, ordered["edge_class"].tolist())
    ax.set_xlim(0, 68)
    ax.set_xlabel("Share of selected attention (%)")
    ax.set_title("Dominant sender–receiver structure", loc="left", pad=7)
    ax.tick_params(axis="y", length=0, labelsize=8)
    _clean_axis(ax, grid_axis="x")
    ax.axhline(2.5, color=GRID, linewidth=0.8)
    ax.text(
        0.98,
        0.97,
        "N1",
        transform=ax.transAxes,
        ha="right",
        va="top",
        color=DOMAIN_COLORS[DOMAIN_ORDER[0]],
        fontweight="bold",
    )
    ax.text(
        0.98,
        0.40,
        "N2",
        transform=ax.transAxes,
        ha="right",
        va="top",
        color=DOMAIN_COLORS[DOMAIN_ORDER[1]],
        fontweight="bold",
    )


def _plot_pathways(
    ax: plt.Axes,
    pathways: "pd.DataFrame",
    domain: str,
) -> None:
    table = pathways.loc[pathways["module"].eq(domain)]
    y = np.arange(len(table))[::-1]
    color = DOMAIN_COLORS[domain]
    for y_value, value in zip(
        y,
        table["log2_fold_over_null"].to_numpy(float),
    ):
        ax.plot([0, value], [y_value, y_value], color=color, linewidth=1.6, alpha=0.78)
        ax.scatter(
            value,
            y_value,
            s=38,
            color=color,
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )
    ax.axvline(0, color=REFERENCE, linewidth=0.8)
    ax.set_yticks(y, table["pathway"].astype(str))
    ax.set_xlim(-0.08, 2.75)
    ax.set_xlabel("log2 enrichment over same-composition null")
    ax.set_title(
        f"{DOMAIN_SHORT[domain]}  ·  {DOMAIN_ROLES[domain]}",
        loc="left",
        color=color,
        pad=7,
    )
    ax.tick_params(axis="y", length=0)
    _clean_axis(ax, grid_axis="x")


def _plot_lr_axes(
    ax: plt.Axes,
    lr_axes: "pd.DataFrame",
    domain: str,
) -> None:
    table = lr_axes.loc[lr_axes["niche"].eq(domain)]
    y = np.arange(len(table))[::-1]
    values = table["log2_fold_over_null"].to_numpy(float)
    color = DOMAIN_COLORS[domain]
    ax.barh(y, values, height=0.55, color=color, alpha=0.88)
    labels = []
    for row in table.itertuples(index=False):
        receptor = str(row.receptor).replace("_", "/")
        labels.append(
            f"{row.ligand}–{receptor}\n{row.dominant_sender}→{row.dominant_receiver}"
        )
    for y_value, value, fold in zip(
        y,
        values,
        table["fold_over_null_mean"].to_numpy(float),
    ):
        ax.text(
            value + 0.08,
            y_value,
            f"{fold:.1f}×",
            va="center",
            fontsize=8,
            color=color,
        )
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 5.65)
    ax.set_xlabel("log2 pair enrichment over same-composition null")
    ax.set_title(
        f"{DOMAIN_SHORT[domain]}-associated domain",
        loc="left",
        color=color,
        pad=7,
    )
    ax.tick_params(axis="y", length=0, labelsize=8)
    _clean_axis(ax, grid_axis="x")


def _make_figure(
    data: "AristaLocalDomainData",
    panels: "AristaLocalDomainPanels",
) -> plt.Figure:
    figure = plt.figure(figsize=(8.27, 11.69), facecolor="white")
    _heading(figure, "a", "Spatial interaction domains", 0.065, 0.956)
    _heading(figure, "b", "Organized cell-state interactions", 0.625, 0.956)
    _heading(
        figure,
        "c",
        "Domain-specific candidate repair programs",
        0.065,
        0.535,
    )
    _heading(figure, "d", "Candidate ligand–receptor axes", 0.065, 0.276)

    spatial_axis = figure.add_axes([0.070, 0.595, 0.500, 0.315])
    attention_axis = figure.add_axes([0.655, 0.765, 0.305, 0.125])
    edge_axis = figure.add_axes([0.655, 0.560, 0.305, 0.135])
    pathway_axes = (
        figure.add_axes([0.105, 0.330, 0.365, 0.165]),
        figure.add_axes([0.585, 0.330, 0.365, 0.165]),
    )
    lr_axes = (
        figure.add_axes([0.165, 0.080, 0.300, 0.155]),
        figure.add_axes([0.665, 0.080, 0.285, 0.155]),
    )

    _plot_spatial_map(spatial_axis, data.roi_assignments)
    _plot_attention(attention_axis, panels.attention)
    _plot_edge_structure(edge_axis, panels.edge_structure)
    for axis, domain in zip(pathway_axes, DOMAIN_ORDER):
        _plot_pathways(axis, panels.pathways, domain)
    for axis, domain in zip(lr_axes, DOMAIN_ORDER):
        _plot_lr_axes(axis, panels.lr_axes, domain)
    return figure


def render_arista_local_domains(
    data: "AristaLocalDomainData",
    panels: "AristaLocalDomainPanels",
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """Render the local-domain figure as PDF and PNG."""

    output = prepare_output_dir(output_dir)
    stem = "FigureS_ARISTA_Figure5c_local_interaction_niches_clean"
    pdf = output / f"{stem}.pdf"
    png = output / f"{stem}.png"
    with mpl.rc_context(ARISTA_RC):
        figure = _make_figure(data, panels)
        figure.savefig(
            pdf,
            facecolor="white",
            metadata={"Creator": "CytoBridge"},
        )
        figure.savefig(
            png,
            dpi=320,
            facecolor="white",
            metadata={"Software": "CytoBridge"},
        )
        plt.close(figure)
    return pdf, png
