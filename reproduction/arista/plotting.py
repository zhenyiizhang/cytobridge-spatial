"""Figure 5 plotting functions from the original ARISTA figure source.

Only numerical plotting routines are included here. Input selection is in
main_figure.py. The labels follow the current manuscript.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm, colors
from matplotlib.colorbar import ColorbarBase
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch, Ellipse, FancyArrowPatch, Polygon, Rectangle


def configure_style() -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial"],
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )

def save_figure(fig: plt.Figure, stem: Path) -> list[Path]:
    paths = [stem.with_suffix(suffix) for suffix in (".svg", ".pdf", ".png")]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        kwargs: dict[str, Any] = {
            "bbox_inches": "tight",
            "pad_inches": 0.04,
            "facecolor": "white",
        }
        if path.suffix == ".png":
            kwargs["dpi"] = 320
        fig.savefig(path, **kwargs)
    return paths

def scatter_by_celltype(
    ax: plt.Axes,
    frame: pd.DataFrame,
    palette: dict[str, str],
    *,
    size: float,
    alpha: float,
) -> None:
    for label, color in palette.items():
        sub = frame[frame["celltype"] == label]
        if sub.empty:
            continue
        ax.scatter(
            sub["x"],
            sub["y"],
            s=size,
            color=color,
            alpha=alpha,
            linewidths=0,
        )

def add_spatial_glyph(ax: plt.Axes, color: str = "#211917") -> None:
    ax.annotate(
        "",
        xy=(0.18, 0.10),
        xytext=(0.04, 0.10),
        xycoords="axes fraction",
        arrowprops=dict(arrowstyle="-|>", lw=1.0, color=color),
    )
    ax.annotate(
        "",
        xy=(0.04, 0.25),
        xytext=(0.04, 0.10),
        xycoords="axes fraction",
        arrowprops=dict(arrowstyle="-|>", lw=1.0, color=color),
    )
    ax.text(0.14, 0.13, "r2", transform=ax.transAxes, fontsize=8, color=color)
    ax.text(0.065, 0.22, "r1", transform=ax.transAxes, fontsize=8, color=color)

def plot_figure5b(frame: pd.DataFrame, palette: dict[str, str], stem: Path) -> list[Path]:
    shown = frame[frame["displayed_point_glyph"]]
    fig, ax = plt.subplots(figsize=(4.3, 4.25))
    scatter_by_celltype(ax, shown, palette, size=2.2, alpha=0.9)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Generated samples", fontsize=12, fontweight="bold", pad=8)
    add_spatial_glyph(ax)
    ax.text(0.5, -0.015, "t=3.5DPI", transform=ax.transAxes, ha="center", fontsize=10)
    paths = save_figure(fig, stem)
    plt.close(fig)
    return paths

def add_velocity_palette(adata, palette: dict[str, str]) -> list[str]:
    labels = adata.obs["Annotation"].astype(str).to_numpy()
    categories = [label for label in palette if label in set(labels)]
    adata.obs["Annotation"] = pd.Categorical(labels, categories=categories, ordered=True)
    color_list = [palette[label] for label in categories]
    adata.uns["Annotation_colors"] = color_list
    return color_list

def draw_velocity_left(
    ax: plt.Axes,
    velocity_adata,
    palette: dict[str, str],
    roi_bounds: tuple[float, float, float, float],
) -> None:
    import scvelo as scv
    from scvelo.plotting.velocity_embedding_grid import compute_velocity_on_grid

    coords = np.asarray(velocity_adata.obsm["X_spatial"], dtype=np.float32)
    embedded = np.asarray(velocity_adata.obsm["velocity_spatial"], dtype=np.float32)
    colors_for_types = add_velocity_palette(velocity_adata, palette)
    # Use the archived scVelo renderer, not the later 68-by-68 IDW grid.
    grid, velocity = compute_velocity_on_grid(
        X_emb=coords, V_emb=embedded, density=1, smooth=None, min_mass=None,
        n_neighbors=None, autoscale=False, adjust_for_stream=True, cutoff_perc=None)
    speed = np.sqrt(np.sum(velocity ** 2, axis=0))
    width = np.nan_to_num(2 * speed / np.nanmax(speed), nan=0.)
    scv.pl.velocity_embedding_stream(
        velocity_adata, basis='spatial', vkey='velocity', color='Annotation',
        palette=colors_for_types, density=2., ax=ax, show=False,
        legend_loc='none', X_grid=grid, V_grid=velocity, linewidth=width,
        alpha=.3, title='Spatial velocity')
    ax.set_title("Spatial velocity", fontsize=12, fontweight="bold", pad=7)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    x0, x1, y0, y1 = roi_bounds
    ax.add_patch(
        Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            fill=False,
            edgecolor="#68737a",
            linewidth=2.1,
            zorder=40,
        )
    )
    add_spatial_glyph(ax)
    ax.text(
        0.50,
        -0.045,
        "Brain regeneration",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10,
        color="black",
    )

def draw_roi(
    ax: plt.Axes,
    vector_table: pd.DataFrame,
    roi_bounds: tuple[float, float, float, float],
    focus_bounds: tuple[float, float, float, float],
) -> Any:
    roi = vector_table[vector_table["in_roi"]]
    norm = colors.TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
    scatter = ax.scatter(
        roi["paper_x"],
        roi["paper_y"],
        c=roi["cosine_full_vs_interaction"],
        cmap="plasma",
        norm=norm,
        s=4.2,
        linewidths=0,
        alpha=0.95,
    )
    x0, x1, y0, y1 = roi_bounds
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#68737a")
        spine.set_linewidth(1.4)
    # The final Figure 5c retains the grey ROI but has no nested red box.
    ax.set_title(
        "Spatial velocity cosine similarity\n(interaction vs full spatial velocity)",
        fontsize=11,
        fontweight="bold",
        pad=7,
    )
    return scatter

def add_low_high_colorbar(fig: plt.Figure, ax: plt.Axes) -> None:
    fig.canvas.draw()
    box = ax.get_position()
    cax = fig.add_axes([box.x0, box.y0 - 0.085, box.width, 0.028])
    ColorbarBase(
        cax,
        cmap=cm.get_cmap("plasma"),
        norm=colors.TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0),
        orientation="horizontal",
    )
    cax.set_xticks([])
    cax.set_yticks([])
    for spine in cax.spines.values():
        spine.set_visible(False)
    cax.text(0.0, -0.9, "Low", transform=cax.transAxes, ha="left", va="top", fontsize=9)
    cax.text(1.0, -0.9, "High", transform=cax.transAxes, ha="right", va="top", fontsize=9)

def plot_figure5c(
    velocity_adata,
    vector_table: pd.DataFrame,
    roi_bounds: tuple[float, float, float, float],
    focus_bounds: tuple[float, float, float, float],
    palette: dict[str, str],
    output_dir: Path,
) -> dict[str, list[Path]]:
    outputs: dict[str, list[Path]] = {}

    fig_left, ax_left = plt.subplots(figsize=(5.0, 4.7))
    draw_velocity_left(ax_left, velocity_adata.copy(), palette, roi_bounds)
    outputs["Figure5c_spatial_migration_velocity"] = save_figure(
        fig_left, output_dir / "Figure5c_spatial_migration_velocity"
    )
    plt.close(fig_left)

    fig_right, ax_right = plt.subplots(figsize=(4.8, 4.7))
    draw_roi(ax_right, vector_table, roi_bounds, focus_bounds)
    fig_right.subplots_adjust(left=0.08, right=0.97, bottom=0.22, top=0.84)
    add_low_high_colorbar(fig_right, ax_right)
    outputs["Figure5c_roi_cosine_similarity"] = save_figure(
        fig_right, output_dir / "Figure5c_roi_cosine_similarity"
    )
    plt.close(fig_right)

    # The paper inset is roughly three quarters of the left tissue width and
    # sits slightly higher, with short connectors.  Explicit axes positions
    # preserve that geometry despite both panels enforcing equal data aspect.
    fig = plt.figure(figsize=(9.3, 4.75))
    ax_l = fig.add_axes([0.02, 0.18, 0.44, 0.68])
    ax_r = fig.add_axes([0.49, 0.27, 0.30, 0.52])
    draw_velocity_left(ax_l, velocity_adata.copy(), palette, roi_bounds)
    draw_roi(ax_r, vector_table, roi_bounds, focus_bounds)
    add_low_high_colorbar(fig, ax_r)
    x0, x1, y0, y1 = roi_bounds
    fig.add_artist(
        ConnectionPatch(
            xyA=(x1, y1),
            coordsA=ax_l.transData,
            xyB=(0.0, 1.0),
            coordsB=ax_r.transAxes,
            color="#68737a",
            linewidth=1.1,
        )
    )
    fig.add_artist(
        ConnectionPatch(
            xyA=(x1, y0),
            coordsA=ax_l.transData,
            xyB=(0.0, 0.0),
            coordsB=ax_r.transAxes,
            color="#68737a",
            linewidth=1.1,
        )
    )
    outputs["Figure5c_spatial_and_roi"] = save_figure(
        fig, output_dir / "Figure5c_spatial_and_roi"
    )
    plt.close(fig)
    return outputs

def plot_figure5e(grouped: pd.DataFrame, stem: Path) -> list[Path]:
    from .growth_plot import plot_growth_interaction

    return plot_growth_interaction(grouped, stem)
