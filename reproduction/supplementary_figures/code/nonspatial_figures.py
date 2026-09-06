"""Shared Matplotlib renderer for grouped non-spatial Figures S4--S5."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Mapping

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .nonspatial_figures import NonspatialDatasetResults, NonspatialFigureResults, NonspatialPanels


A4_PAGE = (595.2 / 72.0, 841.2 / 72.0)
TEAL = "#07838B"
CORAL = "#CC6677"
ROSE = "#B64E6C"
GRID = "#D7DDE2"
HEADING = "#000000"
WEINREB_TEXT = "#111111"
SCNT_TEXT = "#24313A"
STEMS = {
    "s4": "supplementary_figure_s4_weinreb_nonspatial",
    "s5": "supplementary_figure_s5_scnt_nonspatial",
}
SCNT_TIME_LABELS = {
    0.0: "0 min",
    0.25: "15 min",
    0.5: "30 min",
    1.0: "60 min",
    2.0: "120 min",
}
WEINREB_NETWORK_ORDER = (
    "Undifferentiated",
    "Neutrophil",
    "Monocyte",
    "Meg",
    "Mast",
    "Lymphoid",
    "Erythroid",
    "Eos",
    "Ccr7_DC",
    "Baso",
    "pDC",
)


def _rc(text_color: str) -> dict[str, object]:
    return {
        "font.family": "Arial",
        "font.size": 9.0,
        "axes.titlesize": 9.0,
        "axes.labelsize": 9.0,
        "xtick.labelsize": 9.0,
        "ytick.labelsize": 9.0,
        "legend.fontsize": 9.0,
        "axes.linewidth": 0.65,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "text.color": text_color,
        "axes.labelcolor": text_color,
        "axes.titlecolor": HEADING,
        "xtick.color": text_color,
        "ytick.color": text_color,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.titleweight": "normal",
        "legend.handlelength": 1.25,
        "figure.dpi": 120,
        "savefig.bbox": None,
    }


def _save(figure: plt.Figure, output: Path, figure_id: str) -> tuple[Path, Path]:
    stem = output / STEMS[figure_id]
    pdf = stem.with_suffix(".pdf")
    png = stem.with_suffix(".png")
    figure.savefig(pdf, facecolor="white")
    figure.savefig(png, dpi=320, facecolor="white")
    plt.close(figure)
    return pdf, png


def _panel_heading(
    axis: plt.Axes,
    label: str,
    title: str,
    *,
    title_x: float = 0.07,
    y: float = 0.55,
) -> None:
    axis.axis("off")
    axis.text(
        0,
        y,
        label.lower(),
        fontsize=14,
        fontweight="bold",
        va="center",
        ha="left",
        color=HEADING,
    )
    axis.text(
        title_x,
        y,
        title,
        fontsize=12,
        fontweight="bold",
        va="center",
        ha="left",
        color=HEADING,
    )


def _panel_container(
    figure: plt.Figure,
    spec,
    label: str,
    title: str,
    *,
    title_x: float | None = None,
):
    inner = spec.subgridspec(2, 1, height_ratios=[0.15, 1.0], hspace=0.025)
    heading = figure.add_subplot(inner[0])
    position = title_x if title_x is not None else (0.062 if len(title) > 25 else 0.07)
    _panel_heading(heading, label, title, title_x=position)
    return inner[1], heading


def _clean_axis(axis: plt.Axes, *, grid: bool = True) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    if grid:
        axis.grid(color=GRID, linewidth=0.45, alpha=0.55, zorder=0)
    else:
        axis.grid(False)
    axis.set_axisbelow(True)


def _cell_frame(dataset: "NonspatialDatasetResults", *, spring: bool = False) -> pd.DataFrame:
    values: dict[str, object] = {
        "time": dataset.cells.times,
        "cell_type": dataset.cells.labels.astype(str),
        "pc1": dataset.cells.pc_xy[:, 0],
        "pc2": dataset.cells.pc_xy[:, 1],
    }
    if spring:
        if dataset.cells.spring_xy is None:
            raise ValueError("SPRING coordinates are unavailable")
        values["spring_x"] = dataset.cells.spring_xy[:, 0]
        values["spring_y"] = dataset.cells.spring_xy[:, 1]
    return pd.DataFrame(values)


def _condition_handles(*, hollow: bool = False) -> list[Line2D]:
    return [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            color=TEAL,
            markersize=4.6,
            label="With interaction",
        ),
        Line2D(
            [],
            [],
            marker="s",
            linestyle="",
            markerfacecolor="white" if hollow else CORAL,
            markeredgecolor=CORAL,
            color=CORAL,
            markersize=4.4,
            label="Without interaction",
        ),
    ]


def _circle_positions(types: tuple[str, ...]) -> dict[str, np.ndarray]:
    angles = np.linspace(np.pi / 2, np.pi / 2 - 2 * np.pi, len(types), endpoint=False)
    return {
        name: 0.82 * np.asarray([np.cos(angle), np.sin(angle)])
        for name, angle in zip(types, angles, strict=True)
    }


def _draw_circle_network(
    axis: plt.Axes,
    edges: pd.DataFrame,
    nodes: pd.DataFrame,
    *,
    method: str,
    title: str,
    order: tuple[str, ...],
    colors: Mapping[str, str],
    text_color: str,
    display_names: Mapping[str, str] | None = None,
    extent: float = 1.2,
    node_base: float = 85.0,
    node_span: float = 120.0,
) -> None:
    positions = _circle_positions(order)
    selected = edges.loc[edges["method"].eq(method)].sort_values("display_rank", ascending=False)
    max_rank = max(float(selected["display_rank"].max()), 1.0)
    for row in selected.itertuples(index=False):
        source = positions[str(row.sender_type)]
        target = positions[str(row.receiver_type)]
        vector = target - source
        length = float(np.linalg.norm(vector))
        if length <= 0:
            continue
        unit = vector / length
        start = source + 0.075 * unit
        end = target - 0.075 * unit
        strength = 1.0 - (float(row.display_rank) - 1.0) / max(max_rank - 1.0, 1.0)
        curvature = 0.075 if order.index(str(row.sender_type)) < order.index(str(row.receiver_type)) else -0.075
        axis.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=5.5 + 2.8 * strength,
                connectionstyle=f"arc3,rad={curvature}",
                linewidth=0.35 + 1.55 * strength,
                color=colors[str(row.sender_type)],
                alpha=0.20 + 0.58 * strength,
                zorder=1,
            )
        )
    counts = nodes.set_index("cell_type")["n_cells"].reindex(order).to_numpy(float)
    logged = np.log1p(counts)
    scaled = (logged - logged.min()) / max(float(np.ptp(logged)), np.finfo(float).eps)
    for name, size in zip(order, node_base + node_span * scaled, strict=True):
        point = positions[name]
        axis.scatter(
            point[0],
            point[1],
            s=float(size),
            color=colors[name],
            edgecolor="white",
            linewidth=0.9,
            zorder=4,
        )
        label_point = 1.16 * point
        horizontal = "left" if label_point[0] > 0.08 else "right" if label_point[0] < -0.08 else "center"
        vertical = "bottom" if label_point[1] > 0.15 else "top" if label_point[1] < -0.15 else "center"
        axis.text(
            label_point[0],
            label_point[1],
            (display_names or {}).get(name, name),
            ha=horizontal,
            va=vertical,
            fontsize=6.5,
            color=text_color,
        )
    axis.set_xlim(-extent, extent)
    axis.set_ylim(-extent, extent)
    axis.set_aspect("equal")
    axis.axis("off")
    axis.set_title(title, fontsize=8.8, pad=1.5, color=HEADING, fontweight="bold")


def _weinreb_observed(
    figure: plt.Figure,
    spec,
    dataset: "NonspatialDatasetResults",
    cells: pd.DataFrame,
) -> None:
    inner = spec.subgridspec(4, 1, height_ratios=[0.29, 1.0, 0.10, 0.22], hspace=0.035)
    heading = figure.add_subplot(inner[0])
    _panel_heading(heading, "a", "Observed cell states across time", title_x=0.078, y=0.74)
    maps = inner[1].subgridspec(1, 3, wspace=0.035)
    xlim = dataset.fields["spring_xlim"]
    ylim = dataset.fields["spring_ylim"]
    rng = np.random.default_rng(20260810)
    for column, day in enumerate((2, 4, 6)):
        axis = figure.add_subplot(maps[0, column])
        subset = cells.loc[np.isclose(cells["time"], day)]
        counts = subset["cell_type"].value_counts()
        for cell_type in counts.sort_values(ascending=False).index:
            points = subset.loc[subset["cell_type"].eq(cell_type)]
            order = np.arange(len(points))
            rng.shuffle(order)
            points = points.iloc[order]
            axis.scatter(
                points["spring_x"],
                points["spring_y"],
                s=2.0,
                c=dataset.colors[str(cell_type)],
                alpha=0.70,
                linewidths=0,
                rasterized=False,
            )
        axis.set_title(f"Day {day}\nn = {len(subset):,}", fontsize=8.5, pad=0.0, linespacing=0.92)
        axis.set_xlim(*xlim)
        axis.set_ylim(*ylim)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xticks([])
        axis.set_yticks([])
        if column == 0:
            axis.set_ylabel("SPRING 2", fontsize=7.8, labelpad=2.0)
        axis.spines[["top", "right"]].set_visible(False)
        for side in ("bottom", "left"):
            axis.spines[side].set_color("#5A6268")
            axis.spines[side].set_linewidth(0.65)
    coordinate = figure.add_subplot(inner[2])
    coordinate.axis("off")
    coordinate.text(0.5, 0.52, "SPRING 1", ha="center", va="center", fontsize=7.8, color=HEADING)
    legend_axis = figure.add_subplot(inner[3])
    legend_axis.axis("off")
    legend_axis.legend(
        handles=[
            Line2D([], [], marker="o", linestyle="", markersize=4.2, color=dataset.colors[name], label=name)
            for name in dataset.cells.label_names.astype(str)
        ],
        ncol=6,
        loc="center",
        frameon=False,
        fontsize=7.6,
        columnspacing=0.9,
        handletextpad=0.35,
        borderaxespad=0,
    )


def _weinreb_stream(
    axis: plt.Axes,
    dataset: "NonspatialDatasetResults",
    cells: pd.DataFrame,
    *,
    day: int,
    field: str,
    color: str,
    speed_limit: float,
) -> None:
    subset = cells.loc[np.isclose(cells["time"], day)]
    if len(subset) > 2600:
        subset = subset.sample(n=2600, random_state=20260810)
    for cell_type in subset["cell_type"].value_counts().sort_values(ascending=False).index:
        points = subset.loc[subset["cell_type"].eq(cell_type)]
        axis.scatter(
            points["pc1"],
            points["pc2"],
            s=1.45,
            color=dataset.colors[str(cell_type)],
            alpha=0.24,
            linewidths=0,
            rasterized=False,
            zorder=0,
        )
    prefix = f"day{day}"
    x = dataset.fields[f"{prefix}_x_axis"]
    y = dataset.fields[f"{prefix}_y_axis"]
    mask = dataset.fields[f"{prefix}_support_mask"].astype(bool)
    u = np.ma.masked_where(~mask, dataset.fields[f"{prefix}_{field}_u"])
    v = np.ma.masked_where(~mask, dataset.fields[f"{prefix}_{field}_v"])
    speed = np.ma.masked_where(~mask, dataset.fields[f"{prefix}_{field}_speed"])
    linewidth = 0.32 + 0.80 * np.clip(speed.filled(0) / speed_limit, 0, 1)
    axis.streamplot(
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
    axis.set_xlim(float(x.min()), float(x.max()))
    axis.set_ylim(float(y.min()), float(y.max()))
    axis.set_aspect("equal", adjustable="box")
    axis.tick_params(labelsize=7, pad=1)
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(False)


def _weinreb_model(
    figure: plt.Figure,
    spec,
    dataset: "NonspatialDatasetResults",
    cells: pd.DataFrame,
) -> None:
    body, _ = _panel_container(figure, spec, "b", "Interaction-informed dynamics in PCA state space")
    axes = body.subgridspec(1, 4, wspace=0.16)
    full_speeds = []
    for day in (2, 4, 6):
        mask = dataset.fields[f"day{day}_support_mask"].astype(bool)
        full_speeds.append(dataset.fields[f"day{day}_lr_full_drift_speed"][mask])
    full_limit = float(np.quantile(np.concatenate(full_speeds), 0.92))
    interaction_mask = dataset.fields["day6_support_mask"].astype(bool)
    interaction_limit = float(np.quantile(dataset.fields["day6_lr_interaction_speed"][interaction_mask], 0.92))
    specifications = (
        (2, "lr_full_drift", TEAL, "Day 2 · full drift"),
        (4, "lr_full_drift", TEAL, "Day 4 · full drift"),
        (6, "lr_full_drift", TEAL, "Day 6 · full drift"),
        (6, "lr_interaction", ROSE, "Day 6 · interaction component"),
    )
    for index, (day, field, color, title) in enumerate(specifications):
        axis = figure.add_subplot(axes[0, index])
        _weinreb_stream(
            axis,
            dataset,
            cells,
            day=day,
            field=field,
            color=color,
            speed_limit=full_limit if field == "lr_full_drift" else interaction_limit,
        )
        axis.set_title(title, fontsize=8.5, pad=2)
        axis.set_xlabel("PC1", fontsize=8)
        if index == 0:
            axis.set_ylabel("PC2", fontsize=8)
        else:
            axis.set_yticklabels([])


def _weinreb_distribution(axis: plt.Axes, distribution: pd.DataFrame) -> None:
    endpoints = (
        (1.0, "w1", "D4 · W1"),
        (1.0, "w2", "D4 · W2"),
        (2.0, "w1", "D6 · W1"),
        (2.0, "w2", "D6 · W2"),
    )
    y = np.arange(len(endpoints))[::-1]
    for yy, (time, metric, _) in zip(y, endpoints, strict=True):
        row = distribution.loc[np.isclose(distribution["time"], time)].iloc[0]
        full = float(row[f"{metric}_full"])
        no_interaction = float(row[f"{metric}_no_interaction"])
        axis.add_patch(FancyArrowPatch((no_interaction, yy), (full, yy), arrowstyle="-|>", mutation_scale=10, linewidth=1.9, color=TEAL, zorder=3))
        axis.scatter(no_interaction, yy, s=48, marker="s", color=CORAL, edgecolor="white", linewidth=0.6, zorder=4)
        axis.scatter(full, yy, s=54, marker="o", color=TEAL, edgecolor="white", linewidth=0.6, zorder=5)
    values = np.r_[
        distribution[["w1_full", "w2_full"]].to_numpy().ravel(),
        distribution[["w1_no_interaction", "w2_no_interaction"]].to_numpy().ravel(),
    ]
    pad = 0.08 * float(np.ptp(values))
    axis.set_xlim(float(values.min() - pad), float(values.max() + pad))
    axis.set_yticks(y, [item[2] for item in endpoints])
    axis.set_xlabel("Weighted PCA distance  ↓", labelpad=3, fontweight="bold")
    _clean_axis(axis, grid=True)
    axis.grid(axis="y", visible=False)


def _weinreb_clone(figure: plt.Figure, spec, clone: pd.DataFrame) -> None:
    outer = spec.subgridspec(2, 1, height_ratios=[0.14, 1.0], hspace=0.01)
    legend = figure.add_subplot(outer[0])
    legend.axis("off")
    legend.legend(handles=_condition_handles(), loc="center", ncol=2, frameon=False, fontsize=6.5, handletextpad=0.25, columnspacing=0.7, borderaxespad=0)
    indexed = clone.set_index("metric")
    axes = outer[1].subgridspec(1, 2, wspace=0.24)
    for column, (label, metric) in enumerate((("TV agreement", "tv_agreement"), ("JS similarity", "js_similarity"))):
        row = indexed.loc[metric]
        full = float(row["full"])
        no_interaction = float(row["no_interaction"])
        pad = max(0.008, 0.55 * abs(full - no_interaction))
        axis = figure.add_subplot(axes[0, column])
        axis.add_patch(FancyArrowPatch((no_interaction, 0.5), (full, 0.5), arrowstyle="-|>", mutation_scale=10, linewidth=1.9, color=TEAL, zorder=3))
        axis.scatter(no_interaction, 0.5, s=48, marker="s", color=CORAL, edgecolor="white", linewidth=0.6, zorder=4)
        axis.scatter(full, 0.5, s=54, marker="o", color=TEAL, edgecolor="white", linewidth=0.6, zorder=5)
        axis.set_xlim(min(full, no_interaction) - pad, max(full, no_interaction) + pad)
        axis.set_ylim(0.0, 1.0)
        axis.set_yticks([0.5])
        axis.set_yticklabels(["Seed 42"] if column == 0 else [])
        axis.set_xlabel(f"{label}  ↑", labelpad=2, fontweight="bold")
        _clean_axis(axis, grid=True)
        axis.grid(axis="y", visible=False)
        axis.tick_params(axis="x", labelsize=7.2)


def _weinreb_pathways(axis: plt.Axes, pathways: pd.DataFrame) -> None:
    plotted = pathways.sort_values("share_of_day6_score_pct", ascending=True)
    y = np.arange(len(plotted))
    values = plotted["share_of_day6_score_pct"].to_numpy(float)
    labels = {"GRN": "GRN (granulin)", "GALECTIN": "Galectin", "CCL": "CCL chemokines", "ANNEXIN": "Annexin"}
    axis.barh(y, values, height=0.62, color=TEAL, alpha=0.88)
    axis.set_yticks(y, labels=[labels.get(str(name), str(name)) for name in plotted["pathway"]])
    axis.set_xlim(0, max(values) * 1.18)
    axis.set_xlabel("Relative interaction signal (%)")
    for yy, value in zip(y, values, strict=True):
        axis.text(value + max(values) * 0.025, yy, f"{value:.1f}", ha="left", va="center", fontsize=7.0, color=HEADING)
    _clean_axis(axis, grid=True)
    axis.grid(axis="y", visible=False)
    axis.tick_params(axis="both", labelsize=7.0)


def _render_weinreb(results: "NonspatialFigureResults", panels: "NonspatialPanels", output: Path) -> tuple[Path, Path]:
    dataset = results.weinreb
    cells = _cell_frame(dataset, spring=True)
    with mpl.rc_context(_rc(WEINREB_TEXT)):
        figure = plt.figure(figsize=A4_PAGE)
        outer = figure.add_gridspec(4, 1, left=0.075, right=0.975, top=0.985, bottom=0.045, height_ratios=[1.85, 2.20, 1.55, 2.15], hspace=0.25)
        _weinreb_observed(figure, outer[0], dataset, cells)
        _weinreb_model(figure, outer[1], dataset, cells)
        middle = outer[2].subgridspec(1, 2, wspace=0.20)
        distribution_body, _ = _panel_container(figure, middle[0], "c", "Distribution error")
        distribution_axis = figure.add_subplot(distribution_body)
        _weinreb_distribution(distribution_axis, panels.weinreb_distribution)
        distribution_axis.legend(handles=_condition_handles(), loc="upper right", ncol=1, frameon=False, fontsize=6.5, handletextpad=0.25, borderaxespad=0.25)
        clone_body, _ = _panel_container(figure, middle[1], "d", "Clone-fate agreement")
        _weinreb_clone(figure, clone_body, panels.weinreb_clone_fate)
        bottom = outer[3].subgridspec(1, 2, width_ratios=[2.08, 0.92], wspace=0.18)
        network_body, network_heading = _panel_container(figure, bottom[0], "e", "Day 6 directed cell–cell interactions")
        rho = float(panels.weinreb_concordance.loc[np.isclose(panels.weinreb_concordance["day"], 6.0), "spearman_rho"].iloc[0])
        network_heading.text(0.995, 0.50, f"Spearman ρ = {rho:.3f}", ha="right", va="center", fontsize=7.6, color=HEADING, fontweight="bold")
        network_axes = network_body.subgridspec(1, 2, wspace=0.12)
        order = WEINREB_NETWORK_ORDER
        _draw_circle_network(
            figure.add_subplot(network_axes[0, 0]),
            dataset.network_edges,
            dataset.network_nodes,
            method="CytoBridge D",
            title="CytoBridge interaction network",
            order=order,
            colors=dataset.colors,
            text_color=WEINREB_TEXT,
            display_names={"Undifferentiated": "Undiff."},
        )
        _draw_circle_network(
            figure.add_subplot(network_axes[0, 1]),
            dataset.network_edges,
            dataset.network_nodes,
            method="CellChat",
            title="CellChat communication network",
            order=order,
            colors=dataset.colors,
            text_color=WEINREB_TEXT,
            display_names={"Undifferentiated": "Undiff."},
        )
        pathway_body, _ = _panel_container(figure, bottom[1], "f", "Day 6 signaling pathways")
        _weinreb_pathways(figure.add_subplot(pathway_body), panels.weinreb_pathways)
        return _save(figure, output, "s4")


def _scnt_observed(figure: plt.Figure, spec, dataset: "NonspatialDatasetResults", cells: pd.DataFrame) -> None:
    inner = spec.subgridspec(4, 1, height_ratios=[0.29, 1.0, 0.10, 0.22], hspace=0.035)
    heading = figure.add_subplot(inner[0])
    _panel_heading(heading, "a", "Observed scNT cortical cell states across KCl stimulation", title_x=0.075, y=0.74)
    maps = inner[1].subgridspec(1, 5, wspace=0.035)
    xlim = dataset.fields["pc_xlim"]
    ylim = dataset.fields["pc_ylim"]
    rng = np.random.default_rng(20260811)
    for column, time in enumerate((0.0, 0.25, 0.5, 1.0, 2.0)):
        axis = figure.add_subplot(maps[0, column])
        subset = cells.loc[np.isclose(cells["time"], time)]
        counts = subset["cell_type"].value_counts()
        for cell_type in counts.sort_values(ascending=False).index:
            points = subset.loc[subset["cell_type"].eq(cell_type)]
            order = np.arange(len(points))
            rng.shuffle(order)
            points = points.iloc[order]
            axis.scatter(points["pc1"], points["pc2"], s=1.55, color=dataset.colors[str(cell_type)], alpha=0.66, linewidths=0)
        axis.set_title(f"{SCNT_TIME_LABELS[time]}\nn = {len(subset):,}", fontsize=8.0, pad=0, linespacing=0.92)
        axis.set_xlim(*xlim)
        axis.set_ylim(*ylim)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xticks([])
        axis.set_yticks([])
        if column == 0:
            axis.set_ylabel("PC2", fontsize=7.8, labelpad=2)
        axis.spines[["top", "right"]].set_visible(False)
        for side in ("bottom", "left"):
            axis.spines[side].set_color("#5A6268")
            axis.spines[side].set_linewidth(0.65)
    coordinate = figure.add_subplot(inner[2])
    coordinate.axis("off")
    coordinate.text(0.5, 0.52, "PC1", ha="center", va="center", fontsize=7.8, color=HEADING)
    legend_axis = figure.add_subplot(inner[3])
    legend_axis.axis("off")
    legend_axis.legend(
        handles=[Line2D([], [], marker="o", linestyle="", markersize=4.1, color=dataset.colors[name], label=name) for name in dataset.cells.label_names.astype(str)],
        ncol=9,
        loc="center",
        frameon=False,
        fontsize=7.2,
        columnspacing=0.72,
        handletextpad=0.28,
        borderaxespad=0,
    )


def _scnt_grid(dataset: "NonspatialDatasetResults", prefix: str) -> tuple[np.ndarray, ...]:
    return tuple(dataset.fields[f"{prefix}_{name}"] for name in ("x", "y", "u", "v", "mask"))


def _scnt_stream(
    axis: plt.Axes,
    dataset: "NonspatialDatasetResults",
    cells: pd.DataFrame,
    grid: tuple[np.ndarray, ...],
    *,
    time: float,
    color: str,
    speed_limit: float,
) -> None:
    subset = cells.loc[np.isclose(cells["time"], time)]
    if len(subset) > 2600:
        subset = subset.sample(n=2600, random_state=20260811)
    for cell_type in subset["cell_type"].value_counts().sort_values(ascending=False).index:
        points = subset.loc[subset["cell_type"].eq(cell_type)]
        axis.scatter(points["pc1"], points["pc2"], s=1.4, color=dataset.colors[str(cell_type)], alpha=0.23, linewidths=0, zorder=0)
    x, y, u_raw, v_raw, mask = grid
    u = np.ma.masked_where(~mask, u_raw)
    v = np.ma.masked_where(~mask, v_raw)
    speed = np.ma.masked_where(~mask, np.sqrt(u_raw**2 + v_raw**2))
    linewidth = 0.32 + 0.82 * np.clip(speed.filled(0) / max(speed_limit, np.finfo(float).eps), 0, 1)
    axis.streamplot(x, y, u, v, color=color, density=0.72, linewidth=linewidth, arrowsize=0.55, minlength=0.08, maxlength=2.7, zorder=2)
    axis.set_xlim(float(x.min()), float(x.max()))
    axis.set_ylim(float(y.min()), float(y.max()))
    axis.set_aspect("equal", adjustable="box")
    axis.tick_params(labelsize=7, pad=1)
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(False)


def _scnt_model(figure: plt.Figure, spec, dataset: "NonspatialDatasetResults", cells: pd.DataFrame) -> None:
    body, _ = _panel_container(figure, spec, "b", "Interaction-informed dynamics in PCA state space", title_x=0.065)
    axes = body.subgridspec(1, 4, wspace=0.16)
    full_grids = [_scnt_grid(dataset, f"full{index}") for index in range(3)]
    interaction = _scnt_grid(dataset, "interaction")
    full_speeds = [np.sqrt(grid[2] ** 2 + grid[3] ** 2)[grid[4]] for grid in full_grids]
    full_limit = float(np.quantile(np.concatenate(full_speeds), 0.92))
    interaction_limit = float(np.quantile(np.sqrt(interaction[2] ** 2 + interaction[3] ** 2)[interaction[4]], 0.92))
    specifications = (
        (0.0, full_grids[0], TEAL, "0 min · full drift"),
        (0.5, full_grids[1], TEAL, "30 min · full drift"),
        (1.0, full_grids[2], TEAL, "60 min · full drift"),
        (2.0, interaction, ROSE, "120 min · interaction component"),
    )
    for index, (time, grid, color, title) in enumerate(specifications):
        axis = figure.add_subplot(axes[0, index])
        _scnt_stream(axis, dataset, cells, grid, time=time, color=color, speed_limit=full_limit if index < 3 else interaction_limit)
        axis.set_title(title, fontsize=8.3, pad=2)
        axis.set_xlabel("PC1", fontsize=8)
        if index == 0:
            axis.set_ylabel("PC2", fontsize=8)
        else:
            axis.set_yticklabels([])


def _scnt_dumbbells(axis: plt.Axes, full: pd.Series, no_interaction: pd.Series, *, xlabel: str) -> None:
    times = (0.25, 0.5, 1.0, 2.0)
    y = np.arange(len(times))[::-1]
    for yy, time in zip(y, times, strict=True):
        full_value = float(full.loc[time])
        no_value = float(no_interaction.loc[time])
        axis.add_patch(FancyArrowPatch((no_value, yy), (full_value, yy), arrowstyle="-|>", mutation_scale=8.5, linewidth=1.25, color=TEAL, zorder=3))
        axis.scatter(no_value, yy, s=28, marker="s", facecolor="white", edgecolor=CORAL, linewidth=0.8, zorder=4)
        axis.scatter(full_value, yy, s=32, marker="o", color=TEAL, edgecolor="white", linewidth=0.5, zorder=5)
    combined = np.r_[full.to_numpy(float), no_interaction.to_numpy(float)]
    pad = 0.08 * float(np.ptp(combined))
    axis.set_xlim(float(combined.min() - pad), float(combined.max() + pad))
    axis.set_yticks(y, labels=[SCNT_TIME_LABELS[time] for time in times])
    axis.set_xlabel(xlabel, labelpad=2, fontweight="bold")
    _clean_axis(axis, grid=True)
    axis.grid(axis="y", visible=False)
    axis.tick_params(labelsize=7.1)


def _scnt_distribution(figure: plt.Figure, spec, distribution: pd.DataFrame) -> None:
    body, _ = _panel_container(figure, spec, "c", "Distribution error", title_x=0.09)
    outer = body.subgridspec(2, 1, height_ratios=[0.13, 1.0], hspace=0.01)
    legend = figure.add_subplot(outer[0])
    legend.axis("off")
    legend.legend(handles=_condition_handles(hollow=True), loc="center", ncol=2, frameon=False, fontsize=6.5, handletextpad=0.22, columnspacing=0.65, borderaxespad=0)
    axes = outer[1].subgridspec(1, 2, wspace=0.28)
    indexed = distribution.set_index("time")
    for column, metric in enumerate(("w1", "w2")):
        axis = figure.add_subplot(axes[0, column])
        _scnt_dumbbells(axis, indexed[f"{metric}_full"], indexed[f"{metric}_no_interaction"], xlabel=f"{metric.upper()}  ↓")
        if column > 0:
            axis.set_yticklabels([])


def _scnt_direction(figure: plt.Figure, spec, direction: pd.DataFrame) -> None:
    body, _ = _panel_container(figure, spec, "d", "scNT new-RNA direction", title_x=0.082)
    outer = body.subgridspec(2, 1, height_ratios=[0.13, 1.0], hspace=0.01)
    legend = figure.add_subplot(outer[0])
    legend.axis("off")
    legend.legend(handles=_condition_handles(hollow=True), loc="center", ncol=2, frameon=False, fontsize=6.5, handletextpad=0.22, columnspacing=0.65, borderaxespad=0)
    indexed = direction.set_index("condition")
    axis = figure.add_subplot(outer[1])
    for label, metric, y in (("Mean", "cell_cosine_mean", 1.0), ("Median", "cell_cosine_median", 0.0)):
        full = float(indexed.loc["full_interaction_noise", metric])
        no_interaction = float(indexed.loc["no_interaction_noise", metric])
        axis.add_patch(FancyArrowPatch((no_interaction, y), (full, y), arrowstyle="-|>", mutation_scale=10, linewidth=1.8, color=TEAL, zorder=3))
        axis.scatter(no_interaction, y, s=48, marker="s", facecolor="white", edgecolor=CORAL, linewidth=0.9, zorder=4)
        axis.scatter(full, y, s=54, marker="o", color=TEAL, edgecolor="white", linewidth=0.6, zorder=5)
    axis.set_yticks([1.0, 0.0], labels=["Mean", "Median"])
    axis.set_xlim(0.0, 0.014)
    axis.set_ylim(-0.55, 1.55)
    axis.set_xlabel("Cellwise cosine agreement  ↑", labelpad=2, fontweight="bold")
    _clean_axis(axis, grid=True)
    axis.grid(axis="y", visible=False)
    axis.tick_params(labelsize=7.2)


def _scnt_pathways(axis: plt.Axes, pathways: pd.DataFrame) -> None:
    plotted = pathways.sort_values("share_pct", ascending=True)
    y = np.arange(len(plotted))
    values = plotted["share_pct"].to_numpy(float)
    labels = {"PTN": "PTN (pleiotrophin)", "MK": "MK (midkine)", "GRN": "GRN (granulin)", "NRG": "Neuregulin", "EGF": "EGF"}
    axis.barh(y, values, height=0.62, color=TEAL, alpha=0.88)
    axis.set_yticks(y, labels=[labels.get(str(name), str(name)) for name in plotted["pathway"]])
    axis.set_xlim(0, max(values) * 1.22)
    axis.set_xlabel("Share of total learned LR-supported\nsignal (%)", labelpad=3)
    for yy, value in zip(y, values, strict=True):
        axis.text(value + max(values) * 0.025, yy, f"{value:.1f}", ha="left", va="center", fontsize=7.0, color=HEADING)
    _clean_axis(axis, grid=True)
    axis.grid(axis="y", visible=False)
    axis.tick_params(axis="both", labelsize=7.0)
    axis.text(0.02, -0.29, "D: learned GNN message strength   ·   Q: LR-expression compatibility", transform=axis.transAxes, ha="left", va="top", fontsize=6.7, color=SCNT_TEXT)


def _render_scnt(results: "NonspatialFigureResults", panels: "NonspatialPanels", output: Path) -> tuple[Path, Path]:
    dataset = results.scnt
    cells = _cell_frame(dataset)
    with mpl.rc_context(_rc(SCNT_TEXT)):
        figure = plt.figure(figsize=A4_PAGE)
        outer = figure.add_gridspec(4, 1, left=0.075, right=0.975, top=0.985, bottom=0.055, height_ratios=[1.85, 1.65, 1.60, 2.18], hspace=0.20)
        _scnt_observed(figure, outer[0], dataset, cells)
        _scnt_model(figure, outer[1], dataset, cells)
        middle = outer[2].subgridspec(1, 2, wspace=0.20)
        _scnt_distribution(figure, middle[0], panels.scnt_distribution)
        _scnt_direction(figure, middle[1], panels.scnt_direction)
        bottom = outer[3].subgridspec(1, 2, width_ratios=[2.08, 0.92], wspace=0.18)
        network_body, network_heading = _panel_container(figure, bottom[0], "e", "120 min directed interactions", title_x=0.058)
        rho = float(results.scnt.metrics["network"]["spearman_rho_all_81_pairs"])
        network_heading.text(0.995, 0.50, f"all 81 pairs · Spearman ρ = {rho:.3f}", ha="right", va="center", fontsize=7.5, color=HEADING, fontweight="bold")
        axes = network_body.subgridspec(1, 2, wspace=0.12)
        order = tuple(dataset.cells.label_names.astype(str))
        _draw_circle_network(
            figure.add_subplot(axes[0, 0]),
            dataset.network_edges,
            dataset.network_nodes,
            method="CytoBridge D",
            title="CytoBridge learned message D",
            order=order,
            colors=dataset.colors,
            text_color=SCNT_TEXT,
            extent=1.22,
            node_base=86,
            node_span=118,
        )
        _draw_circle_network(
            figure.add_subplot(axes[0, 1]),
            dataset.network_edges,
            dataset.network_nodes,
            method="CellChat",
            title="CellChat communication",
            order=order,
            colors=dataset.colors,
            text_color=SCNT_TEXT,
            extent=1.22,
            node_base=86,
            node_span=118,
        )
        pathway_body, _ = _panel_container(figure, bottom[1], "f", "LR-supported pathways", title_x=0.13)
        _scnt_pathways(figure.add_subplot(pathway_body), panels.scnt_pathways)
        return _save(figure, output, "s5")


def render_nonspatial_figures(
    results: "NonspatialFigureResults",
    panels: "NonspatialPanels",
    output: Path,
    figures: tuple[str, ...],
) -> dict[str, tuple[Path, Path]]:
    """Render requested figures in canonical display order."""

    rendered: dict[str, tuple[Path, Path]] = {}
    for figure_id in figures:
        rendered[figure_id] = (
            _render_weinreb(results, panels, output)
            if figure_id == "s4"
            else _render_scnt(results, panels, output)
        )
    return rendered
