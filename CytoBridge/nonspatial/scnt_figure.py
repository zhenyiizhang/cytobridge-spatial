"""Build the accepted scNT non-spatial interaction A4 figure.

The layout follows the Weinreb non-spatial interaction A4 figure, while the
contents are adapted to the scNT cortical KCl time course. No rendered panel is
reused. Bundle selection and SHA-256 validation are owned by
``CytoBridge.nonspatial.figures``.

Panel b uses saved paired CytoBridge trajectories for the full-drift fields.
Its final map is reconstructed from the exact sender-specific GNN interaction
drift vectors, summed by receiver type and smoothly displayed in PC1--PC2.
It is neither RNA velocity nor a subtraction between independently trained
models.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

from .scnt_figure_style import (
    A4_PORTRAIT,
    ABLATION_COLOR,
    CYTOBRIDGE_COLOR,
    GRID_COLOR,
    GT_COLOR,
    HEADING_COLOR,
    TEXT_COLOR,
    apply_style,
    clean_axis,
    panel_heading,
    save_figure,
)


HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent
PANEL_DATA = BUNDLE / "panel_data"
METRICS = BUNDLE / "metrics"
FIGURE_STEM = "scnt_nonspatial_interaction_a4"

TEAL = CYTOBRIDGE_COLOR
CORAL = ABLATION_COLOR
ROSE = "#B64E6C"
MID_GRAY = "#8A939A"
DARK_GRAY = "#111111"

CELL_COLORS = {
    "Ex": "#4C78A8",
    "EX-NP1": "#72A7D8",
    "EX-NP2": "#9AC5E8",
    "RG": "#59A14F",
    "Inh-NP": "#F28E2B",
    "Inh1": "#E15759",
    "Inh2": "#B55D92",
    "Inh3": "#8E6C8A",
    "Inh4": "#D99BC6",
}
CELL_ORDER = ["Ex", "EX-NP1", "EX-NP2", "RG", "Inh-NP", "Inh1", "Inh2", "Inh3", "Inh4"]
TIME_ORDER = [0.0, 0.25, 0.5, 1.0, 2.0]
TIME_LABELS = {
    0.0: "0 min",
    0.25: "15 min",
    0.5: "30 min",
    1.0: "60 min",
    2.0: "120 min",
}

COPIED_NAMES = {
    "trajectory": "full_paired_dense_trajectory.npz",
    "full_distribution": "full_distribution_metrics.csv",
    "no_interaction_distribution": "no_interaction_distribution_metrics.csv",
    "direction": "timewise_scnt_direction_alignment.csv",
    "direction_manifest": "scnt_direction_evaluation_manifest.json",
    "interaction_network": "cell_type_interaction_network.csv",
    "interaction_summary": "exact_message_summary.csv",
    "pathways": "cell_type_pathway_scores.csv.gz",
    "cellchat": "cellchat_edge_summary.csv",
    "counts": "cell_type_counts_by_time.csv",
}


def require_panel_data() -> None:
    expected = ["observed_cells.csv.gz", "source_manifest.json", *COPIED_NAMES.values()]
    missing = [name for name in expected if not (PANEL_DATA / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "The validated archived panel-data bundle is incomplete. Missing: "
            + ", ".join(missing)
        )


def panel_container(
    fig: plt.Figure, spec, label: str, title: str, *, title_x: float = 0.07
):
    inner = spec.subgridspec(2, 1, height_ratios=[0.15, 1.0], hspace=0.025)
    heading = fig.add_subplot(inner[0])
    panel_heading(heading, label, title, title_x=title_x)
    return inner[1], heading


def pc_limits(cells: pd.DataFrame):
    xlo, xhi = cells["pc1"].quantile([0.002, 0.998])
    ylo, yhi = cells["pc2"].quantile([0.002, 0.998])
    xpad = 0.035 * (xhi - xlo)
    ypad = 0.035 * (yhi - ylo)
    return (float(xlo - xpad), float(xhi + xpad)), (
        float(ylo - ypad),
        float(yhi + ypad),
    )


def draw_observed_panel(fig: plt.Figure, spec, cells: pd.DataFrame) -> None:
    inner = spec.subgridspec(4, 1, height_ratios=[0.29, 1.0, 0.10, 0.22], hspace=0.035)
    head = fig.add_subplot(inner[0])
    panel_heading(
        head,
        "a",
        "Observed scNT cortical cell states across KCl stimulation",
        title_x=0.075,
        y=0.74,
    )
    maps = inner[1].subgridspec(1, 5, wspace=0.035)
    xlim, ylim = pc_limits(cells)
    rng = np.random.default_rng(20260811)
    for column, time in enumerate(TIME_ORDER):
        ax = fig.add_subplot(maps[0, column])
        subset = cells[np.isclose(cells["time"], time)]
        counts = subset["cell_type"].value_counts()
        for cell_type in counts.sort_values(ascending=False).index:
            points = subset[subset["cell_type"] == cell_type]
            order = np.arange(len(points))
            rng.shuffle(order)
            points = points.iloc[order]
            ax.scatter(
                points["pc1"],
                points["pc2"],
                s=1.55,
                color=CELL_COLORS[cell_type],
                alpha=0.66,
                linewidths=0,
            )
        ax.set_title(
            f"{TIME_LABELS[time]}\nn = {len(subset):,}",
            fontsize=8.0,
            pad=0,
            linespacing=0.92,
        )
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        if column == 0:
            ax.set_ylabel("PC2", fontsize=7.8, labelpad=2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for side in ["bottom", "left"]:
            ax.spines[side].set_color("#5A6268")
            ax.spines[side].set_linewidth(0.65)
    coord = fig.add_subplot(inner[2])
    coord.axis("off")
    coord.text(
        0.5, 0.52, "PC1", ha="center", va="center", fontsize=7.8, color=HEADING_COLOR
    )
    legend_ax = fig.add_subplot(inner[3])
    legend_ax.axis("off")
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markersize=4.1,
            color=CELL_COLORS[name],
            label=name,
        )
        for name in CELL_ORDER
    ]
    legend_ax.legend(
        handles=handles,
        ncol=9,
        loc="center",
        frameon=False,
        fontsize=7.2,
        columnspacing=0.72,
        handletextpad=0.28,
        borderaxespad=0,
    )


def smoothed_particle_field(
    points: np.ndarray, velocity: np.ndarray, xlim, ylim, *, n_grid: int = 42
):
    x = np.linspace(*xlim, n_grid)
    y = np.linspace(*ylim, n_grid)
    xx, yy = np.meshgrid(x, y)
    query = np.column_stack([xx.ravel(), yy.ravel()])
    tree = cKDTree(points)
    k = min(96, len(points))
    distances, indices = tree.query(query, k=k)
    local_scale = np.maximum(distances[:, min(31, k - 1)], np.finfo(float).eps)
    weights = np.exp(-0.5 * (distances / local_scale[:, None]) ** 2)
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), np.finfo(float).eps)
    smoothed = np.einsum("nk,nkd->nd", weights, velocity[indices])
    sample_dist, _ = tree.query(points, k=min(24, len(points)))
    support_cutoff = float(np.quantile(sample_dist[:, -1], 0.92))
    support = distances[:, min(23, k - 1)] <= support_cutoff
    return (
        x,
        y,
        smoothed[:, 0].reshape(xx.shape),
        smoothed[:, 1].reshape(xx.shape),
        support.reshape(xx.shape),
    )


def trajectory_field(trajectory, target_time: float, xlim, ylim):
    times = trajectory["time_points"].astype(float)
    points = trajectory["points"].astype(float)
    index = int(np.argmin(np.abs(times - target_time)))
    if index == 0:
        left, right = 0, 1
    elif index == len(times) - 1:
        left, right = index - 1, index
    else:
        left, right = index - 1, index + 1
    dt = float(times[right] - times[left])
    velocity = (points[right, :, :2] - points[left, :, :2]) / dt
    state = points[index, :, :2]
    return smoothed_particle_field(state, velocity, xlim, ylim)


def interaction_field(cells: pd.DataFrame, summary: pd.DataFrame, xlim, ylim):
    time = 2.0
    observed = cells[np.isclose(cells["time"], time)]
    centroids = observed.groupby("cell_type")[["pc1", "pc2"]].mean().reindex(CELL_ORDER)
    block = summary[np.isclose(summary["time"], time)].copy()
    totals = (
        block.groupby("receiver_type")[["drift_pc_1_mean", "drift_pc_2_mean"]]
        .sum()
        .reindex(CELL_ORDER)
        .fillna(0)
    )
    base_points = centroids[["pc1", "pc2"]].to_numpy(dtype=float)
    base_velocity = totals[["drift_pc_1_mean", "drift_pc_2_mean"]].to_numpy(dtype=float)
    x = np.linspace(*xlim, 42)
    y = np.linspace(*ylim, 42)
    xx, yy = np.meshgrid(x, y)
    query = np.column_stack([xx.ravel(), yy.ravel()])
    pairwise = np.linalg.norm(base_points[:, None, :] - base_points[None, :, :], axis=2)
    nonzero = pairwise[pairwise > 0]
    bandwidth = max(float(np.median(nonzero)) * 0.58, np.finfo(float).eps)
    distances = np.linalg.norm(query[:, None, :] - base_points[None, :, :], axis=2)
    weights = np.exp(-0.5 * (distances / bandwidth) ** 2)
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), np.finfo(float).eps)
    field = weights @ base_velocity
    observed_tree = cKDTree(observed[["pc1", "pc2"]].to_numpy(dtype=float))
    d, _ = observed_tree.query(query, k=min(24, len(observed)))
    sample_d, _ = observed_tree.query(
        observed[["pc1", "pc2"]].to_numpy(dtype=float), k=min(24, len(observed))
    )
    support = d[:, -1] <= float(np.quantile(sample_d[:, -1], 0.92))
    return (
        x,
        y,
        field[:, 0].reshape(xx.shape),
        field[:, 1].reshape(xx.shape),
        support.reshape(xx.shape),
        totals.reset_index(),
    )


def deterministic_background(subset: pd.DataFrame, n: int = 2600) -> pd.DataFrame:
    return subset if len(subset) <= n else subset.sample(n=n, random_state=20260811)


def draw_stream_field(ax, cells, grid, *, time: float, color: str, speed_limit: float):
    subset = deterministic_background(cells[np.isclose(cells["time"], time)])
    for cell_type in (
        subset["cell_type"].value_counts().sort_values(ascending=False).index
    ):
        points = subset[subset["cell_type"] == cell_type]
        ax.scatter(
            points["pc1"],
            points["pc2"],
            s=1.4,
            color=CELL_COLORS[cell_type],
            alpha=0.23,
            linewidths=0,
            zorder=0,
        )
    x, y, u_raw, v_raw, mask = grid[:5]
    u = np.ma.masked_where(~mask, u_raw)
    v = np.ma.masked_where(~mask, v_raw)
    speed = np.ma.masked_where(~mask, np.sqrt(u_raw**2 + v_raw**2))
    linewidth = 0.32 + 0.82 * np.clip(
        speed.filled(0) / max(speed_limit, np.finfo(float).eps), 0, 1
    )
    ax.streamplot(
        x,
        y,
        u,
        v,
        color=color,
        density=0.72,
        linewidth=linewidth,
        arrowsize=0.55,
        minlength=0.08,
        maxlength=2.7,
        zorder=2,
    )
    ax.set_xlim(float(x.min()), float(x.max()))
    ax.set_ylim(float(y.min()), float(y.max()))
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(labelsize=7, pad=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)


def draw_model_field_panel(fig, spec, cells, trajectory, summary):
    body, _ = panel_container(
        fig,
        spec,
        "b",
        "Interaction-informed dynamics in PCA state space",
        title_x=0.065,
    )
    axes_spec = body.subgridspec(1, 4, wspace=0.16)
    xlim, ylim = pc_limits(cells)
    full_times = [0.0, 0.5, 1.0]
    full_grids = [trajectory_field(trajectory, time, xlim, ylim) for time in full_times]
    inter_grid = interaction_field(cells, summary, xlim, ylim)
    full_speeds = [np.sqrt(grid[2] ** 2 + grid[3] ** 2)[grid[4]] for grid in full_grids]
    full_limit = float(np.quantile(np.concatenate(full_speeds), 0.92))
    inter_speed = np.sqrt(inter_grid[2] ** 2 + inter_grid[3] ** 2)[inter_grid[4]]
    inter_limit = float(np.quantile(inter_speed, 0.92))
    specifications = [
        (0.0, full_grids[0], TEAL, "0 min · full drift"),
        (0.5, full_grids[1], TEAL, "30 min · full drift"),
        (1.0, full_grids[2], TEAL, "60 min · full drift"),
        (2.0, inter_grid, ROSE, "120 min · interaction component"),
    ]
    for index, (time, grid, color, title) in enumerate(specifications):
        ax = fig.add_subplot(axes_spec[0, index])
        draw_stream_field(
            ax,
            cells,
            grid,
            time=time,
            color=color,
            speed_limit=full_limit if index < 3 else inter_limit,
        )
        ax.set_title(title, fontsize=8.3, pad=2)
        ax.set_xlabel("PC1", fontsize=8)
        if index == 0:
            ax.set_ylabel("PC2", fontsize=8)
        else:
            ax.set_yticklabels([])
    return inter_grid[-1]


def condition_legend_handles():
    return [
        Line2D(
            [], [], marker="o", linestyle="", color=TEAL, markersize=4.6, label="Full"
        ),
        Line2D(
            [],
            [],
            marker="s",
            linestyle="",
            markerfacecolor="white",
            markeredgecolor=CORAL,
            markersize=4.4,
            label="No interaction",
        ),
    ]


def draw_dumbbell_axis(
    ax, full: pd.Series, no_interaction: pd.Series, *, xlabel: str, xlim=None
):
    times = [0.25, 0.5, 1.0, 2.0]
    y = np.arange(len(times))[::-1]
    for yy, time in zip(y, times):
        f = float(full.loc[time])
        n = float(no_interaction.loc[time])
        ax.add_patch(
            FancyArrowPatch(
                (n, yy),
                (f, yy),
                arrowstyle="-|>",
                mutation_scale=8.5,
                linewidth=1.25,
                color=TEAL,
                zorder=3,
            )
        )
        ax.scatter(
            n,
            yy,
            s=28,
            marker="s",
            facecolor="white",
            edgecolor=CORAL,
            linewidth=0.8,
            zorder=4,
        )
        ax.scatter(
            f,
            yy,
            s=32,
            marker="o",
            color=TEAL,
            edgecolor="white",
            linewidth=0.5,
            zorder=5,
        )
    ax.set_yticks(y, labels=[TIME_LABELS[t] for t in times])
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel, labelpad=2, fontweight="bold")
    clean_axis(ax, grid=True)
    ax.grid(axis="y", visible=False)
    ax.tick_params(labelsize=7.1)


def draw_distribution_panel(fig, spec, full_df, no_df):
    body, heading = panel_container(fig, spec, "c", "Distribution error", title_x=0.09)
    heading.legend(
        handles=condition_legend_handles(),
        loc="center right",
        ncol=2,
        frameon=False,
        fontsize=6.5,
        handletextpad=0.22,
        columnspacing=0.55,
        borderaxespad=0,
    )
    axes = body.subgridspec(1, 2, wspace=0.28)
    full = full_df.set_index("time")
    no = no_df.set_index("time")
    for column, metric in enumerate(["w1", "w2"]):
        ax = fig.add_subplot(axes[0, column])
        combined = np.r_[full[metric].to_numpy(), no[metric].to_numpy()]
        pad = 0.08 * float(np.ptp(combined))
        draw_dumbbell_axis(
            ax,
            full[metric],
            no[metric],
            xlabel=f"{metric.upper()}  ↓",
            xlim=(combined.min() - pad, combined.max() + pad),
        )
        if column > 0:
            ax.set_yticklabels([])


def draw_direction_panel(fig, spec, direction):
    body, heading = panel_container(
        fig, spec, "d", "scNT new-RNA direction", title_x=0.082
    )
    heading.legend(
        handles=condition_legend_handles(),
        loc="center right",
        ncol=2,
        frameon=False,
        fontsize=6.5,
        handletextpad=0.22,
        columnspacing=0.55,
        borderaxespad=0,
    )
    condition_names = {"full_interaction_noise": "full", "no_interaction_noise": "no"}
    subset = direction[direction["condition"].isin(condition_names)].copy()
    subset["short"] = subset["condition"].map(condition_names)
    averaged = subset.groupby("short")[
        ["cell_cosine_mean", "cell_cosine_median"]
    ].mean()
    ax = fig.add_subplot(body)
    specifications = [
        ("Mean", "cell_cosine_mean", 1.0),
        ("Median", "cell_cosine_median", 0.0),
    ]
    for _, metric, y in specifications:
        full = float(averaged.loc["full", metric])
        no_interaction = float(averaged.loc["no", metric])
        ax.add_patch(
            FancyArrowPatch(
                (no_interaction, y),
                (full, y),
                arrowstyle="-|>",
                mutation_scale=10,
                linewidth=1.8,
                color=TEAL,
                zorder=3,
            )
        )
        ax.scatter(
            no_interaction,
            y,
            s=48,
            marker="s",
            facecolor="white",
            edgecolor=CORAL,
            linewidth=0.9,
            zorder=4,
        )
        ax.scatter(
            full,
            y,
            s=54,
            marker="o",
            color=TEAL,
            edgecolor="white",
            linewidth=0.6,
            zorder=5,
        )
    ax.set_yticks([1.0, 0.0], labels=["Mean", "Median"])
    ax.set_xlim(0.0, 0.014)
    ax.set_ylim(-0.55, 1.55)
    ax.set_xlabel("Cellwise cosine agreement  ↑", labelpad=2, fontweight="bold")
    clean_axis(ax, grid=True)
    ax.grid(axis="y", visible=False)
    ax.tick_params(labelsize=7.2)


def circle_positions(types):
    angles = np.linspace(np.pi / 2, np.pi / 2 - 2 * np.pi, len(types), endpoint=False)
    return {
        name: 0.82 * np.array([np.cos(angle), np.sin(angle)])
        for name, angle in zip(types, angles)
    }


def select_network_edges(network, cellchat, *, time=2.0, top_n=24):
    learned = network[np.isclose(network["time"], time)].copy()
    chat = cellchat[np.isclose(cellchat["time"], time)].copy()
    learned = learned[learned["sender_type"] != learned["receiver_type"]]
    chat = chat[chat["sender_type"] != chat["receiver_type"]]
    learned = learned[learned["D_AB_mean"] > 0].nlargest(top_n, "D_AB_mean").copy()
    chat = (
        chat[chat["cellchat_native_significant"] > 0]
        .nlargest(top_n, "cellchat_native_significant")
        .copy()
    )
    rows = []
    for method, block, score in [
        ("CytoBridge D", learned, "D_AB_mean"),
        ("CellChat", chat, "cellchat_native_significant"),
    ]:
        block = block.copy()
        block["method"] = method
        block["display_score"] = block[score]
        block["display_rank"] = np.arange(1, len(block) + 1)
        rows.append(
            block[
                [
                    "method",
                    "sender_type",
                    "receiver_type",
                    "display_score",
                    "display_rank",
                ]
            ]
        )
    return pd.concat(rows, ignore_index=True)


def network_concordance(network, cellchat, *, time=2.0):
    a = network[np.isclose(network["time"], time)][
        ["sender_type", "receiver_type", "D_AB_mean"]
    ]
    b = cellchat[np.isclose(cellchat["time"], time)][
        ["sender_type", "receiver_type", "cellchat_native_significant"]
    ]
    joined = a.merge(b, on=["sender_type", "receiver_type"], validate="one_to_one")
    rho = float(
        spearmanr(joined["D_AB_mean"], joined["cellchat_native_significant"]).statistic
    )
    return rho, len(joined), joined


def draw_circle_network(ax, edges, counts, *, method, title):
    positions = circle_positions(CELL_ORDER)
    selected = edges[edges["method"] == method].sort_values(
        "display_rank", ascending=False
    )
    max_rank = max(float(selected["display_rank"].max()), 1.0)
    for row in selected.itertuples(index=False):
        source = positions[row.sender_type]
        target = positions[row.receiver_type]
        vector = target - source
        length = float(np.linalg.norm(vector))
        if length <= 0:
            continue
        unit = vector / length
        start = source + 0.075 * unit
        end = target - 0.075 * unit
        strength = 1.0 - (float(row.display_rank) - 1.0) / max(max_rank - 1.0, 1.0)
        curvature = (
            0.075
            if CELL_ORDER.index(row.sender_type) < CELL_ORDER.index(row.receiver_type)
            else -0.075
        )
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=5.5 + 2.8 * strength,
                connectionstyle=f"arc3,rad={curvature}",
                linewidth=0.35 + 1.55 * strength,
                color=CELL_COLORS[row.sender_type],
                alpha=0.20 + 0.58 * strength,
                zorder=1,
            )
        )
    count_series = (
        counts[np.isclose(counts["time"], 2.0)]
        .set_index("cell_type")["n_cells"]
        .reindex(CELL_ORDER)
    )
    values = np.log1p(count_series.to_numpy(dtype=float))
    scaled = (values - values.min()) / max(float(np.ptp(values)), np.finfo(float).eps)
    for name, size in zip(CELL_ORDER, 86 + 118 * scaled):
        point = positions[name]
        ax.scatter(
            point[0],
            point[1],
            s=float(size),
            color=CELL_COLORS[name],
            edgecolor="white",
            linewidth=0.9,
            zorder=4,
        )
        label_point = 1.16 * point
        ha = (
            "left"
            if label_point[0] > 0.08
            else "right"
            if label_point[0] < -0.08
            else "center"
        )
        va = (
            "bottom"
            if label_point[1] > 0.15
            else "top"
            if label_point[1] < -0.15
            else "center"
        )
        ax.text(
            label_point[0],
            label_point[1],
            name,
            ha=ha,
            va=va,
            fontsize=6.5,
            color=TEXT_COLOR,
        )
    ax.set_xlim(-1.22, 1.22)
    ax.set_ylim(-1.22, 1.22)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=8.8, pad=1.5, color=HEADING_COLOR, fontweight="bold")


def draw_network_panel(fig, spec, network, cellchat, counts):
    edges = select_network_edges(network, cellchat)
    rho, n_contexts, joined = network_concordance(network, cellchat)
    body, heading = panel_container(
        fig, spec, "e", "120 min directed interactions", title_x=0.058
    )
    heading.text(
        0.995,
        0.50,
        f"all {n_contexts} pairs · Spearman ρ = {rho:.3f}",
        ha="right",
        va="center",
        fontsize=7.5,
        color=HEADING_COLOR,
        fontweight="bold",
    )
    axes = body.subgridspec(1, 2, wspace=0.12)
    ax1 = fig.add_subplot(axes[0, 0])
    draw_circle_network(
        ax1, edges, counts, method="CytoBridge D", title="CytoBridge learned message D"
    )
    ax2 = fig.add_subplot(axes[0, 1])
    draw_circle_network(
        ax2, edges, counts, method="CellChat", title="CellChat communication"
    )
    return edges, rho, joined


def select_pathways(pathways, *, time=2.0, top_n=8):
    block = pathways[np.isclose(pathways["time"], time)].copy()
    block = block[
        (block["sender_type"] != block["receiver_type"]) & (block["S_AB_pathway"] > 0)
    ]
    total = float(block["S_AB_pathway"].sum())
    selected = block.groupby("pathway", as_index=False).agg(
        summed_S=("S_AB_pathway", "sum"), n_positive_pairs=("S_AB_pathway", "size")
    )
    selected = (
        selected.sort_values(["summed_S", "pathway"], ascending=[False, True])
        .head(top_n)
        .copy()
    )
    selected["share_pct"] = 100 * selected["summed_S"] / total
    selected["display_rank"] = np.arange(1, len(selected) + 1)
    return selected


def draw_pathway_panel(fig, spec, pathways):
    selected = select_pathways(pathways)
    body, _ = panel_container(fig, spec, "f", "LR-supported pathways", title_x=0.13)
    ax = fig.add_subplot(body)
    plotted = selected.sort_values("share_pct", ascending=True)
    y = np.arange(len(plotted))
    values = plotted["share_pct"].to_numpy(dtype=float)
    labels = {
        "PTN": "PTN (pleiotrophin)",
        "MK": "MK (midkine)",
        "GRN": "GRN (granulin)",
        "MIF": "MIF",
        "NRG": "Neuregulin",
        "EGF": "EGF",
    }
    ax.barh(y, values, height=0.62, color=TEAL, alpha=0.88)
    ax.set_yticks(
        y, labels=[labels.get(str(name), str(name)) for name in plotted["pathway"]]
    )
    ax.set_xlim(0, max(values) * 1.22)
    ax.set_xlabel("Share of total learned LR-supported\nsignal (%)", labelpad=3)
    for yy, value in zip(y, values):
        ax.text(
            value + max(values) * 0.025,
            yy,
            f"{value:.1f}",
            ha="left",
            va="center",
            fontsize=7.0,
            color=HEADING_COLOR,
        )
    clean_axis(ax, grid=True)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="both", labelsize=7.0)
    ax.text(
        0.02,
        -0.29,
        "D: learned GNN message strength   ·   Q: LR-expression compatibility",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.7,
        color=DARK_GRAY,
    )
    return selected


def write_metrics(
    full_dist,
    no_dist,
    direction,
    interaction_totals,
    network_edges,
    rho,
    joined,
    pathways,
):
    METRICS.mkdir(parents=True, exist_ok=True)
    distribution = full_dist[["time", "w1", "w2", "tmv_absolute"]].merge(
        no_dist[["time", "w1", "w2", "tmv_absolute"]],
        on="time",
        suffixes=("_full", "_no_interaction"),
    )
    distribution.to_csv(
        METRICS / "distribution_full_vs_no_interaction.csv", index=False
    )
    direction.to_csv(
        METRICS / "scnt_new_rna_direction_full_vs_no_interaction.csv", index=False
    )
    interaction_totals.to_csv(
        METRICS / "interaction_component_receiver_type_vectors_120min.csv", index=False
    )
    network_edges.to_csv(METRICS / "network_display_edges_120min.csv", index=False)
    joined.to_csv(
        METRICS / "cytobridge_D_vs_cellchat_all_pairs_120min.csv", index=False
    )
    pathways.to_csv(METRICS / "top_lr_supported_pathways_120min.csv", index=False)
    direction_primary = direction[
        direction["condition"].isin(["full_interaction_noise", "no_interaction_noise"])
    ].copy()
    direction_pivot = direction_primary.pivot(
        index="time_hours",
        columns="condition",
        values=["cell_cosine_mean", "cell_cosine_median"],
    )
    mean_full = float(
        direction_pivot[("cell_cosine_mean", "full_interaction_noise")].mean()
    )
    mean_no = float(
        direction_pivot[("cell_cosine_mean", "no_interaction_noise")].mean()
    )
    median_full = float(
        direction_pivot[("cell_cosine_median", "full_interaction_noise")].mean()
    )
    median_no = float(
        direction_pivot[("cell_cosine_median", "no_interaction_noise")].mean()
    )
    mean_wins = int(
        (
            direction_pivot[("cell_cosine_mean", "full_interaction_noise")]
            > direction_pivot[("cell_cosine_mean", "no_interaction_noise")]
        ).sum()
    )
    median_wins = int(
        (
            direction_pivot[("cell_cosine_median", "full_interaction_noise")]
            > direction_pivot[("cell_cosine_median", "no_interaction_noise")]
        ).sum()
    )
    summary = {
        "dataset": {
            "system": "primary mouse cortical culture",
            "perturbation": "KCl stimulation",
            "cells": 20547,
            "times_minutes": [0, 15, 30, 60, 120],
            "cell_types": CELL_ORDER,
        },
        "conditions": {
            "full": "interaction + score/noise",
            "no_interaction": "independently trained without interaction, score/noise retained",
        },
        "model_field": {
            "full_drift": "finite-difference conditional mean from saved paired full-model trajectory in PC1-PC2",
            "interaction_component": "sum of exact sender-specific GNN interaction drift by receiver type, smoothly displayed in PC1-PC2",
            "is_rna_velocity": False,
            "is_model_subtraction": False,
        },
        "direction_evaluation": {
            "reference": "sealed post-training one-shot scNT new-RNA direction projected through the exact HVG2000 PCA loadings",
            "metric": ["cell_cosine_mean", "cell_cosine_median"],
            "training_seed": 42,
            "higher_is_better": True,
            "new_rna_used_for_training": False,
            "equal_endpoint_average": {
                "mean_cellwise_cosine_full": mean_full,
                "mean_cellwise_cosine_no_interaction": mean_no,
                "median_cellwise_cosine_full": median_full,
                "median_cellwise_cosine_no_interaction": median_no,
            },
            "endpoint_wins_full": {
                "mean": mean_wins,
                "median": median_wins,
                "out_of": int(len(direction_pivot)),
            },
            "inferential_scope": "one paired computational training seed; descriptive, not a significance test",
        },
        "network": {
            "time_minutes": 120,
            "score": "D_AB_mean",
            "cellchat_score": "cellchat_native_significant",
            "spearman_rho_all_81_pairs": rho,
            "display": "independent top-24 positive heterotypic directed edges within each method; widths are within-method ranks",
        },
        "pathway": {
            "time_minutes": 120,
            "score": "S_AB_pathway = D_AB_mean * Q_AB_pathway",
            "display": "top eight pathways by summed positive heterotypic S; bar is share of total positive heterotypic S",
        },
    }
    (METRICS / "figure_summary_metrics.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )


def build_figure(dpi: int) -> None:
    apply_style()
    mpl.rcParams.update(
        {
            "axes.titleweight": "normal",
            "legend.handlelength": 1.25,
            "figure.dpi": 120,
            "savefig.bbox": None,
        }
    )
    cells = pd.read_csv(PANEL_DATA / "observed_cells.csv.gz")
    trajectory = np.load(PANEL_DATA / "full_paired_dense_trajectory.npz")
    full_dist = pd.read_csv(PANEL_DATA / "full_distribution_metrics.csv")
    no_dist = pd.read_csv(PANEL_DATA / "no_interaction_distribution_metrics.csv")
    direction = pd.read_csv(PANEL_DATA / "timewise_scnt_direction_alignment.csv")
    network = pd.read_csv(PANEL_DATA / "cell_type_interaction_network.csv")
    summary = pd.read_csv(PANEL_DATA / "exact_message_summary.csv")
    pathways = pd.read_csv(PANEL_DATA / "cell_type_pathway_scores.csv.gz")
    cellchat = pd.read_csv(PANEL_DATA / "cellchat_edge_summary.csv")
    counts = pd.read_csv(PANEL_DATA / "cell_type_counts_by_time.csv")

    fig = plt.figure(figsize=A4_PORTRAIT)
    outer = fig.add_gridspec(
        4,
        1,
        left=0.075,
        right=0.975,
        top=0.985,
        bottom=0.055,
        height_ratios=[1.85, 1.65, 1.60, 2.18],
        hspace=0.20,
    )
    draw_observed_panel(fig, outer[0], cells)
    interaction_totals = draw_model_field_panel(
        fig, outer[1], cells, trajectory, summary
    )
    middle = outer[2].subgridspec(1, 2, wspace=0.20)
    draw_distribution_panel(fig, middle[0], full_dist, no_dist)
    draw_direction_panel(fig, middle[1], direction)
    bottom = outer[3].subgridspec(1, 2, width_ratios=[2.08, 0.92], wspace=0.18)
    network_edges, rho, joined = draw_network_panel(
        fig, bottom[0], network, cellchat, counts
    )
    selected_pathways = draw_pathway_panel(fig, bottom[1], pathways)

    save_figure(
        fig, BUNDLE / f"{FIGURE_STEM}.pdf", BUNDLE / f"{FIGURE_STEM}.png", dpi=dpi
    )
    plt.close(fig)
    write_metrics(
        full_dist,
        no_dist,
        direction,
        interaction_totals,
        network_edges,
        rho,
        joined,
        selected_pathways,
    )
