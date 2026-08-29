"""Render the compact AGIST supplementary-figure bundle."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from ._io import prepare_output_dir
from .agist_figures import AgistFigureData, AgistFigurePanels


TEAL = "#07838B"
REFERENCE = "#59616A"
IQR_COLOR = "#C7CDD1"
S2_GRID = "#D7DDE2"
S2_TEXT = "#24313A"
S2_HEADING = "#102A43"

GROUND_TRUTH = "#4D4D4D"
LEARNED = "#007C91"
INTERACTION_OFF = "#D55E00"
S3_GRID = "#E3E6E8"
S3_TEXT = "#222222"

SPATIAL_LIMITS = ((0.10, 0.92), (0.12, 0.90))
GENE_LIMITS = ((-0.68, 0.68), (-0.42, 0.42))


def _time_colormap() -> mpl.colors.Colormap:
    base = mpl.colormaps["viridis"]
    return mpl.colors.LinearSegmentedColormap.from_list(
        "viridis_manuscript", base(np.linspace(0.08, 0.85, 256))
    )


def _s2_axis_style(axis: mpl.axes.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="y", color=S2_GRID, linewidth=0.45, alpha=0.65, zorder=0)
    axis.set_axisbelow(True)
    axis.axhline(
        1.0,
        color="#A7ADB2",
        linewidth=0.75,
        linestyle=(0, (3, 2)),
        zorder=1,
    )
    axis.tick_params(length=2.5, width=0.6)


def _draw_velocity_summary(
    axis: mpl.axes.Axes,
    table: pd.DataFrame,
    category_column: str,
    categories: list[object],
    labels: list[str],
) -> None:
    subset = table.set_index(category_column).loc[categories].reset_index()
    x = np.arange(len(subset), dtype=float)
    mean = subset["mean"].to_numpy(dtype=float)
    median = subset["median"].to_numpy(dtype=float)
    ci_low = subset["ci95_low"].to_numpy(dtype=float)
    ci_high = subset["ci95_high"].to_numpy(dtype=float)
    q25 = subset["q25"].to_numpy(dtype=float)
    q75 = subset["q75"].to_numpy(dtype=float)

    axis.vlines(
        x, q25, q75, color=IQR_COLOR, linewidth=6.5, capstyle="round", zorder=2
    )
    axis.scatter(
        x,
        median,
        s=22,
        marker="D",
        facecolor="white",
        edgecolor=REFERENCE,
        linewidth=1.0,
        zorder=4,
    )
    axis.plot(x, mean, color=TEAL, linewidth=1.4, zorder=3)
    axis.errorbar(
        x,
        mean,
        yerr=np.vstack([mean - ci_low, ci_high - mean]),
        fmt="o",
        color=TEAL,
        markerfacecolor=TEAL,
        markeredgecolor=TEAL,
        markersize=4.5,
        capsize=2.6,
        elinewidth=1.0,
        capthick=1.0,
        zorder=5,
    )
    axis.set_xticks(x, labels)
    axis.set_xlim(-0.38, len(subset) - 0.62)
    axis.set_ylim(0.75, 1.012)
    axis.set_yticks(np.arange(0.75, 1.001, 0.05))
    _s2_axis_style(axis)


def _s2_heading(
    figure: plt.Figure, axis: mpl.axes.Axes, label: str, title: str
) -> None:
    bounds = axis.get_position()
    y = bounds.y1 + 0.035
    figure.text(
        bounds.x0,
        y,
        label,
        fontsize=14,
        fontweight="bold",
        color=S2_HEADING,
        ha="left",
        va="center",
    )
    figure.text(
        bounds.x0 + 0.038,
        y,
        title,
        fontsize=12,
        fontweight="bold",
        color=S2_HEADING,
        ha="left",
        va="center",
    )


def _plot_s2(
    panels: AgistFigurePanels, pdf_path: Path, png_path: Path
) -> None:
    style = {
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
        "savefig.facecolor": "white",
        "text.color": S2_TEXT,
        "axes.labelcolor": S2_TEXT,
        "axes.titlecolor": S2_TEXT,
        "xtick.color": S2_TEXT,
        "ytick.color": S2_TEXT,
    }
    with mpl.rc_context(style):
        figure, axes = plt.subplots(2, 2, figsize=(8.27, 7.25), sharey=True)
        figure.subplots_adjust(
            left=0.105,
            right=0.975,
            bottom=0.105,
            top=0.835,
            hspace=0.56,
            wspace=0.25,
        )

        by_time = panels.velocity_by_time
        by_cluster = panels.velocity_by_cluster
        time_categories = sorted(by_time["time"].unique().tolist())
        time_counts = (
            by_time[by_time["velocity_space"] == "physical"]
            .set_index("time")["n"]
            .astype(int)
        )
        time_labels = [
            f"t = {time:g}\nn = {time_counts.loc[time]:,}"
            for time in time_categories
        ]

        cluster_categories = sorted(by_cluster["state_cluster"].unique().tolist())
        cluster_counts = (
            by_cluster[by_cluster["velocity_space"] == "physical"]
            .set_index("state_cluster")["n"]
            .astype(int)
        )
        cluster_labels = [
            f"{cluster}\nn = {cluster_counts.loc[cluster]:,}"
            for cluster in cluster_categories
        ]

        panel_specs = (
            (
                axes[0, 0],
                by_time[by_time["velocity_space"] == "physical"],
                "time",
                time_categories,
                time_labels,
            ),
            (
                axes[0, 1],
                by_time[by_time["velocity_space"] == "gene"],
                "time",
                time_categories,
                time_labels,
            ),
            (
                axes[1, 0],
                by_cluster[by_cluster["velocity_space"] == "physical"],
                "state_cluster",
                cluster_categories,
                cluster_labels,
            ),
            (
                axes[1, 1],
                by_cluster[by_cluster["velocity_space"] == "gene"],
                "state_cluster",
                cluster_categories,
                cluster_labels,
            ),
        )
        for axis, table, category, values, labels in panel_specs:
            _draw_velocity_summary(axis, table, category, values, labels)

        axes[0, 0].set_ylabel("Velocity cosine similarity")
        axes[1, 0].set_ylabel("Velocity cosine similarity")
        axes[0, 0].set_xlabel("Simulated time point", labelpad=5)
        axes[0, 1].set_xlabel("Simulated time point", labelpad=5)
        axes[1, 0].set_xlabel("State-space cluster", labelpad=5)
        axes[1, 1].set_xlabel("State-space cluster", labelpad=5)

        headings = (
            ("a", "Physical dynamics across time"),
            ("b", "Gene dynamics across time"),
            ("c", "Physical dynamics across state partitions"),
            ("d", "Gene dynamics across state partitions"),
        )
        for axis, (label, title) in zip(axes.flat, headings):
            _s2_heading(figure, axis, label, title)

        figure.legend(
            handles=[
                Line2D(
                    [0],
                    [0],
                    color=TEAL,
                    marker="o",
                    markersize=4.5,
                    linewidth=1.4,
                    label="Mean and 95% CI",
                ),
                Line2D(
                    [0],
                    [0],
                    color=IQR_COLOR,
                    marker="D",
                    markerfacecolor="white",
                    markeredgecolor=REFERENCE,
                    markersize=4.5,
                    linewidth=5.5,
                    label="Median and IQR",
                ),
                Line2D(
                    [0],
                    [0],
                    color="#A7ADB2",
                    linewidth=0.8,
                    linestyle=(0, (3, 2)),
                    label="Perfect directional agreement",
                ),
            ],
            loc="upper center",
            bbox_to_anchor=(0.54, 0.962),
            ncol=3,
            frameon=False,
            handlelength=2.3,
            columnspacing=1.8,
            handletextpad=0.6,
        )
        figure.savefig(pdf_path, facecolor="white")
        figure.savefig(png_path, dpi=320, facecolor="white")
        plt.close(figure)


def _s3_clean_axis(axis: mpl.axes.Axes, *, grid: bool = False) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#4A4A4A")
    axis.spines["bottom"].set_color("#4A4A4A")
    if grid:
        axis.grid(axis="y", color=S3_GRID, linewidth=0.42, alpha=0.65, zorder=0)
    else:
        axis.grid(False)
    axis.set_axisbelow(True)


def _s3_group_heading(
    axis: mpl.axes.Axes, label: str, title: str
) -> None:
    axis.axis("off")
    axis.text(
        0.0,
        0.55,
        label,
        fontsize=11.5,
        fontweight="bold",
        va="center",
        ha="left",
        color=S3_TEXT,
    )
    axis.text(
        0.038,
        0.55,
        title,
        fontsize=10.3,
        fontweight="semibold",
        va="center",
        ha="left",
        color=S3_TEXT,
    )


def _draw_observed_snapshots(
    figure: plt.Figure,
    slot: mpl.gridspec.SubplotSpec,
    data: AgistFigureData,
    time_cmap: mpl.colors.Colormap,
) -> None:
    panel = slot.subgridspec(
        3, 5, height_ratios=[0.25, 1.0, 1.0], hspace=0.20, wspace=0.17
    )
    heading = figure.add_subplot(panel[0, :])
    _s3_group_heading(heading, "a", "Spatiotemporal snapshots")
    unique_times = np.sort(np.unique(data.observed_time))
    for column, time in enumerate(unique_times):
        mask = np.isclose(data.observed_time, time)
        color = time_cmap(time / unique_times.max())
        for row, values, limits in (
            (1, data.observed_spatial, SPATIAL_LIMITS),
            (2, data.observed_gene, GENE_LIMITS),
        ):
            axis = figure.add_subplot(panel[row, column])
            axis.scatter(
                values[mask, 0],
                values[mask, 1],
                s=2.45,
                color=color,
                alpha=0.72,
                edgecolors="none",
            )
            axis.set_xlim(*limits[0])
            axis.set_ylim(*limits[1])
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_aspect("equal", adjustable="box")
            for spine in axis.spines.values():
                spine.set_color("#C8CDD1")
                spine.set_linewidth(0.45)
            if row == 1:
                axis.set_title(
                    f"t = {time:g}\nn = {mask.sum()}",
                    pad=1.5,
                    fontsize=8.4,
                    linespacing=0.95,
                )
            if column == 0:
                axis.set_ylabel(
                    "Spatial" if row == 1 else "Gene",
                    labelpad=3.2,
                    fontweight="semibold",
                )


def _trajectory_axis(
    axis: mpl.axes.Axes,
    states: np.ndarray,
    time_points: np.ndarray,
    dims: tuple[int, int],
    limits: tuple[tuple[float, float], tuple[float, float]],
    *,
    title: str,
    time_cmap: mpl.colors.Colormap,
) -> None:
    norm = mpl.colors.Normalize(
        vmin=float(time_points.min()), vmax=float(time_points.max())
    )
    values = states[:, :, list(dims)]
    for cell in range(values.shape[1]):
        xy = values[:, cell, :]
        segments = np.stack([xy[:-1], xy[1:]], axis=1)
        collection = LineCollection(
            segments,
            cmap=time_cmap,
            norm=norm,
            linewidths=0.36,
            alpha=0.27,
            zorder=1,
        )
        collection.set_array(time_points[:-1])
        axis.add_collection(collection)
    axis.scatter(
        values[0, :, 0],
        values[0, :, 1],
        s=4.0,
        color=time_cmap(0.0),
        alpha=0.78,
        edgecolors="none",
        zorder=3,
    )
    axis.scatter(
        values[-1, :, 0],
        values[-1, :, 1],
        s=4.5,
        color=time_cmap(1.0),
        alpha=0.84,
        edgecolors="#4D4D4D",
        linewidths=0.14,
        zorder=4,
    )
    axis.set_xlim(*limits[0])
    axis.set_ylim(*limits[1])
    axis.set_title(title, pad=2.0)
    _s3_clean_axis(axis)


def _draw_trajectories(
    figure: plt.Figure,
    slot: mpl.gridspec.SubplotSpec,
    data: AgistFigureData,
    time_cmap: mpl.colors.Colormap,
) -> None:
    panel = slot.subgridspec(
        3, 2, height_ratios=[0.11, 1.0, 1.0], hspace=0.27, wspace=0.18
    )
    heading = figure.add_subplot(panel[0, :])
    _s3_group_heading(heading, "b", "Cell-state trajectories")
    spaces = (
        ((0, 1), SPATIAL_LIMITS, ("spatial x", "spatial y")),
        ((2, 3), GENE_LIMITS, ("gene 1", "gene 2")),
    )
    models = (
        ("Ground truth", data.trajectory_ground_truth),
        ("CytoBridge", data.trajectory_predicted),
    )
    for row, (dims, limits, labels) in enumerate(spaces, start=1):
        for column, (model_name, states) in enumerate(models):
            axis = figure.add_subplot(panel[row, column])
            _trajectory_axis(
                axis,
                states,
                data.trajectory_time,
                dims,
                limits,
                title=model_name if row == 1 else "",
                time_cmap=time_cmap,
            )
            if column == 0:
                axis.set_ylabel(labels[1])
            else:
                axis.set_yticklabels([])
            if row == 2:
                axis.set_xlabel(labels[0])
            else:
                axis.set_xticklabels([])
    heading.legend(
        handles=[
            Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                markersize=3.4,
                color=time_cmap(0.0),
                label="t = 0",
            ),
            Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                markersize=3.4,
                color=time_cmap(1.0),
                label="t = 4",
            ),
        ],
        loc="center right",
        frameon=False,
        ncol=2,
        handletextpad=0.45,
        columnspacing=1.2,
        bbox_to_anchor=(1.0, 0.55),
    )


def _draw_growth_and_interaction(
    figure: plt.Figure,
    slot: mpl.gridspec.SubplotSpec,
    data: AgistFigureData,
    panels: AgistFigurePanels,
) -> None:
    panel = slot.subgridspec(
        2, 3, height_ratios=[0.10, 1.0], hspace=0.04, wspace=0.39
    )
    for column, (label, title) in enumerate(
        (
            ("c", "Population growth"),
            ("d", "Radial cell-cell attraction"),
            ("e", "Interaction potential"),
        )
    ):
        heading = figure.add_subplot(panel[0, column])
        heading.axis("off")
        heading.text(
            0.0,
            0.48,
            label,
            fontsize=11.5,
            fontweight="bold",
            color=S3_TEXT,
            va="center",
        )
        heading.text(
            0.11,
            0.48,
            title,
            fontsize=10.3,
            fontweight="semibold",
            color=S3_TEXT,
            va="center",
        )

    growth = panels.growth_summary
    axis = figure.add_subplot(panel[1, 0])
    axis.plot(
        growth["time"],
        growth["observed_relative_mass"],
        color=GROUND_TRUTH,
        marker="o",
        markersize=3.4,
        linewidth=1.5,
        label="Ground truth",
    )
    axis.errorbar(
        growth["time"],
        growth["predicted_mean"],
        yerr=growth["predicted_sd"],
        color=LEARNED,
        marker="o",
        markersize=3.4,
        linewidth=1.5,
        capsize=1.8,
        label="CytoBridge",
    )
    axis.set_xlabel("time")
    axis.set_ylabel("relative total mass")
    axis.set_xticks(growth["time"])
    axis.set_ylim(1.0, 1.50)
    axis.set_yticks([1.0, 1.1, 1.2, 1.3, 1.4, 1.5])
    axis.legend(
        frameon=False,
        loc="upper left",
        handlelength=1.8,
        handletextpad=0.5,
    )
    _s3_clean_axis(axis, grid=True)

    radial = data.radial_curve
    distance = radial["distance"].to_numpy(dtype=float)
    true_coefficient = radial["true_coefficient"].to_numpy(dtype=float)
    learned_coefficient = radial["learned_coefficient"].to_numpy(dtype=float)
    axis = figure.add_subplot(panel[1, 1])
    axis.plot(
        distance,
        true_coefficient,
        color=GROUND_TRUTH,
        linewidth=1.65,
        label="Ground truth",
    )
    learned_interior = distance < 0.30
    axis.plot(
        distance[learned_interior],
        learned_coefficient[learned_interior],
        color=LEARNED,
        linewidth=1.65,
        label="CytoBridge",
    )
    axis.set_xlim(0, 0.30)
    axis.set_ylim(-0.16, 0.55)
    axis.set_xlabel("pair distance r")
    axis.set_ylabel("radial coefficient c(r)")
    _s3_clean_axis(axis)

    potential = panels.potential_curve
    axis = figure.add_subplot(panel[1, 2])
    axis.plot(
        potential["distance"],
        potential["true_potential"],
        color=GROUND_TRUTH,
        linewidth=1.65,
        solid_capstyle="butt",
        clip_on=True,
        label="Ground truth",
    )
    axis.plot(
        potential["distance"],
        potential["learned_potential"],
        color=LEARNED,
        linewidth=1.65,
        solid_capstyle="butt",
        clip_on=True,
        label="CytoBridge",
    )
    axis.set_xlim(0, 0.30)
    axis.set_ylim(-0.10, 0.02)
    axis.set_xlabel("pair distance r")
    axis.set_ylabel("U(r), U(cutoff) = 0")
    _s3_clean_axis(axis)


def _draw_ablation(
    figure: plt.Figure,
    slot: mpl.gridspec.SubplotSpec,
    panels: AgistFigurePanels,
) -> None:
    panel = slot.subgridspec(
        2, 3, height_ratios=[0.15, 1.0], hspace=0.10, wspace=0.36
    )
    heading = figure.add_subplot(panel[0, :])
    _s3_group_heading(heading, "f", "Effect of cell-cell interactions")
    heading.legend(
        handles=[
            Line2D(
                [],
                [],
                color=LEARNED,
                marker="o",
                markersize=3.2,
                linewidth=1.4,
                label="With interaction",
            ),
            Line2D(
                [],
                [],
                color=INTERACTION_OFF,
                marker="o",
                markersize=3.2,
                linewidth=1.4,
                label="Without interaction",
            ),
        ],
        loc="center right",
        frameon=False,
        ncol=2,
        handlelength=1.5,
        handletextpad=0.45,
        columnspacing=1.0,
        bbox_to_anchor=(1.0, 0.55),
    )
    table = panels.ablation_summary
    for column, (space, label) in enumerate(
        (("joint", "Joint state"), ("spatial", "Spatial"), ("gene", "Gene expression"))
    ):
        axis = figure.add_subplot(panel[1, column])
        for condition, color in (
            ("interaction_on", LEARNED),
            ("interaction_off", INTERACTION_OFF),
        ):
            subset = table[
                (table["space"] == space) & (table["condition"] == condition)
            ].sort_values("time")
            axis.errorbar(
                subset["time"],
                subset["mean"],
                yerr=subset["sem"],
                color=color,
                marker="o",
                markersize=3.0,
                linewidth=1.4,
                capsize=1.4,
            )
        axis.set_title(label, fontsize=8.7, pad=3.0)
        axis.set_xlabel("time")
        axis.set_ylabel("W1" if column == 0 else "")
        axis.set_xticks([1, 2, 3, 4])
        _s3_clean_axis(axis, grid=True)


def _plot_s3(
    data: AgistFigureData,
    panels: AgistFigurePanels,
    pdf_path: Path,
    png_path: Path,
) -> None:
    style = {
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
        "text.color": S3_TEXT,
        "axes.labelcolor": S3_TEXT,
        "axes.titlecolor": S3_TEXT,
        "xtick.color": S3_TEXT,
        "ytick.color": S3_TEXT,
    }
    with mpl.rc_context(style):
        figure = plt.figure(figsize=(8.27, 11.69), facecolor="white")
        outer = figure.add_gridspec(
            4,
            1,
            height_ratios=[2.38, 3.18, 2.02, 1.82],
            left=0.070,
            right=0.980,
            top=0.975,
            bottom=0.055,
            hspace=0.14,
        )
        time_cmap = _time_colormap()
        _draw_observed_snapshots(figure, outer[0, 0], data, time_cmap)
        _draw_trajectories(figure, outer[1, 0], data, time_cmap)
        _draw_growth_and_interaction(figure, outer[2, 0], data, panels)
        _draw_ablation(figure, outer[3, 0], panels)
        figure.savefig(png_path, dpi=320, facecolor="white")
        figure.savefig(pdf_path, facecolor="white")
        plt.close(figure)


def plot_agist_figures(
    data: AgistFigureData,
    panels: AgistFigurePanels,
    output_dir: str | Path,
) -> dict[str, tuple[Path, Path]]:
    """Render Supplementary Figures S2 and S3 as PDF and PNG."""

    output = prepare_output_dir(output_dir)
    s2_pdf = output / "supplementary_figure_s2.pdf"
    s2_png = output / "supplementary_figure_s2.png"
    s3_pdf = output / "supplementary_figure_s3.pdf"
    s3_png = output / "supplementary_figure_s3.png"
    _plot_s2(panels, s2_pdf, s2_png)
    _plot_s3(data, panels, s3_pdf, s3_png)
    return {"s2": (s2_pdf, s2_png), "s3": (s3_pdf, s3_png)}
