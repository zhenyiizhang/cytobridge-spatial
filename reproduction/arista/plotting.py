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
    from scipy.spatial import cKDTree

    coords = np.asarray(velocity_adata.obsm["X_spatial"], dtype=float)
    embedded = np.asarray(velocity_adata.obsm["velocity_spatial"], dtype=float)
    labels = velocity_adata.obs["Annotation"].astype(str).to_numpy()
    background = pd.DataFrame(
        {"x": coords[:, 0], "y": coords[:, 1], "celltype": labels}
    )
    scatter_by_celltype(ax, background, palette, size=4.0, alpha=0.56)

    # The corrected scVelo graph projection above is the numerical velocity
    # calculation.  For the paper-style black stream glyph, interpolate those
    # already embedded finite vectors onto a regular spatial grid.  A finite
    # k-nearest-neighbor display grid avoids scVelo's occasional NaN path at
    # the tissue boundary while leaving the graph projection unchanged.
    x_lo, x_hi = np.quantile(coords[:, 0], [0.002, 0.998])
    y_lo, y_hi = np.quantile(coords[:, 1], [0.002, 0.998])
    x_grid = np.linspace(x_lo, x_hi, 68)
    y_grid = np.linspace(y_lo, y_hi, 68)
    mesh_x, mesh_y = np.meshgrid(x_grid, y_grid)
    query = np.c_[mesh_x.ravel(), mesh_y.ravel()]
    tree = cKDTree(coords)
    distances, indices = tree.query(query, k=40, workers=1)
    weights = 1.0 / np.maximum(distances, 1e-7) ** 2
    weights /= weights.sum(axis=1, keepdims=True)
    grid_velocity = np.einsum("ij,ijk->ik", weights, embedded[indices])
    cell_neighbor_distances, _ = tree.query(coords, k=2, workers=1)
    support_radius = 3.0 * float(np.quantile(cell_neighbor_distances[:, 1], 0.95))
    supported = distances[:, 0] <= support_radius
    speed = np.linalg.norm(grid_velocity, axis=1)
    supported &= np.isfinite(speed) & (speed > np.quantile(speed[np.isfinite(speed)], 0.05))
    u = np.ma.masked_where(~supported.reshape(mesh_x.shape), grid_velocity[:, 0].reshape(mesh_x.shape))
    v = np.ma.masked_where(~supported.reshape(mesh_y.shape), grid_velocity[:, 1].reshape(mesh_y.shape))
    ax.streamplot(
        x_grid,
        y_grid,
        u,
        v,
        color="#111111",
        density=1.8,
        linewidth=0.65,
        arrowsize=0.75,
        minlength=0.08,
        maxlength=3.5,
        zorder=12,
    )
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
    fx0, fx1, fy0, fy1 = focus_bounds
    ax.add_patch(
        Rectangle(
            (fx0, fy0),
            fx1 - fx0,
            fy1 - fy0,
            fill=False,
            edgecolor="#ff5a52",
            linewidth=1.7,
            zorder=20,
        )
    )
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
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    sc = ax.scatter(
        grouped["interaction_mean"],
        grouped["growth_mean"],
        s=np.clip(grouped["n"].to_numpy(dtype=float), 20, 400),
        c=grouped["time_idx"],
        cmap="plasma",
        alpha=0.82,
        edgecolors="white",
        linewidths=0.35,
    )
    ax.set_xlabel("Mean interaction magnitude", fontsize=10)
    ax.set_ylabel("Mean growth", fontsize=10)
    ax.yaxis.set_label_coords(-0.15, 0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for name in ("left", "bottom"):
        ax.spines[name].set_color("#c6c6c6")
        ax.spines[name].set_linewidth(1.2)
    ax.text(
        0.60,
        0.91,
        "Dot=one (time, celltype)\nSize=cells in group",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
    )
    purple = "#3b00b5"
    late = grouped[grouped["time"] == grouped["time"].max()]
    target = late.iloc[int(np.argmin(np.abs(late["growth_mean"] - late["growth_mean"].median())))]
    ax.annotate(
        "Time increasing",
        xy=(target["interaction_mean"], target["growth_mean"]),
        xycoords="data",
        xytext=(0.58, 0.03),
        textcoords="axes fraction",
        color=purple,
        fontsize=10,
        arrowprops=dict(arrowstyle="->", color=purple, lw=1.8),
    )
    high_target = late.iloc[int(np.argmax(late["growth_mean"].to_numpy()))]
    ax.annotate(
        "",
        xy=(high_target["interaction_mean"], high_target["growth_mean"]),
        xycoords="data",
        xytext=(0.42, 0.43),
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", color=purple, lw=1.8),
    )
    ax.annotate(
        "",
        xy=(1.035, -0.005),
        xytext=(-0.075, -0.005),
        xycoords="axes fraction",
        arrowprops=dict(arrowstyle="->", color="#211917", lw=1.0),
    )
    ax.annotate(
        "",
        xy=(-0.075, 1.02),
        xytext=(-0.075, -0.005),
        xycoords="axes fraction",
        arrowprops=dict(arrowstyle="->", color="#211917", lw=1.0),
    )
    cbar = fig.colorbar(sc, ax=ax, pad=0.04, fraction=0.045)
    cbar.set_ticks([])
    cbar.set_label("Time index", fontsize=10)
    cbar.outline.set_edgecolor("#c6c6c6")
    cbar.outline.set_linewidth(1.0)
    cbar.ax.text(1.35, 1.0, "High", transform=cbar.ax.transAxes, ha="left", va="center", fontsize=9)
    cbar.ax.text(1.35, 0.0, "Low", transform=cbar.ax.transAxes, ha="left", va="center", fontsize=9)
    fig.subplots_adjust(left=0.20, right=0.87, bottom=0.18, top=0.96)
    paths = save_figure(fig, stem)
    plt.close(fig)
    return paths
