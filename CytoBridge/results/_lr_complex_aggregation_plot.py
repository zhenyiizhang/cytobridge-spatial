"""Figure renderer for LR-complex aggregation sensitivity."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from ._style import (
    GRID_COLOR,
    LR_COMPLEX_COLORS,
    LR_COMPLEX_RC,
    TEXT_COLOR,
)
from .lr_complex_aggregation import (
    DATASET_LABEL_ORDER,
    LRComplexAggregationResults,
)


def _clean_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(TEXT_COLOR)
    ax.spines["bottom"].set_color(TEXT_COLOR)
    ax.grid(color=GRID_COLOR, linewidth=0.45, alpha=0.65)
    ax.set_axisbelow(True)


def _panel_heading(ax: plt.Axes, label: str, title: str) -> None:
    ax.set_axis_off()
    ax.text(
        0.0,
        0.5,
        label,
        fontsize=14,
        fontweight="bold",
        ha="left",
        va="center",
        color=TEXT_COLOR,
    )
    ax.text(
        0.105,
        0.5,
        title,
        fontsize=12,
        fontweight="bold",
        ha="left",
        va="center",
        color=TEXT_COLOR,
    )


def _coverage(ax: plt.Axes, results: LRComplexAggregationResults) -> None:
    summary = results.dataset_summary.set_index("dataset")
    multi = [
        int(summary.loc[name, "n_multisubunit_pairs"]) for name in DATASET_LABEL_ORDER
    ]
    total = [int(summary.loc[name, "n_scored_pairs"]) for name in DATASET_LABEL_ORDER]
    y = np.arange(len(DATASET_LABEL_ORDER))
    ax.barh(
        y,
        multi,
        color=[LR_COMPLEX_COLORS[name] for name in DATASET_LABEL_ORDER],
        height=0.62,
    )
    ax.set_yticks(y, DATASET_LABEL_ORDER)
    ax.invert_yaxis()
    ax.set_xlabel("Scored multi-subunit LR pairs")
    xmax = max(multi) * 1.25
    ax.set_xlim(0, xmax)
    for index, (count, all_count) in enumerate(zip(multi, total)):
        ax.text(
            count + xmax * 0.018,
            index,
            f"{count:,} of {all_count:,}",
            va="center",
            ha="left",
            fontsize=9,
        )
    _clean_axis(ax)


def _pooled_rank(ax: plt.Axes, results: LRComplexAggregationResults) -> None:
    per_time = results.per_time_summary
    summary = results.dataset_summary.set_index("dataset")
    y = np.arange(len(DATASET_LABEL_ORDER))
    for index, name in enumerate(DATASET_LABEL_ORDER):
        values = per_time.loc[
            per_time["dataset"].eq(name) & per_time["scope"].eq("all_scored_pairs"),
            "spearman",
        ].dropna()
        offsets = (
            np.linspace(-0.13, 0.13, len(values))
            if len(values) > 1
            else np.zeros(len(values))
        )
        ax.scatter(
            values,
            index + offsets,
            color=LR_COMPLEX_COLORS[name],
            s=26,
            edgecolor="white",
            linewidth=0.45,
            zorder=2,
        )
        ax.scatter(
            [float(summary.loc[name, "pooled_spearman"])],
            [index],
            color=LR_COMPLEX_COLORS[name],
            marker="D",
            s=64,
            zorder=3,
            edgecolor=TEXT_COLOR,
            linewidth=0.8,
        )
    ax.set_yticks(y, DATASET_LABEL_ORDER)
    ax.invert_yaxis()
    ax.set_xlim(0.90, 1.005)
    ax.set_xlabel("Spearman rank correlation")
    ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=TEXT_COLOR,
                marker="o",
                linestyle="none",
                markersize=5,
                label="Single time point",
            ),
            Line2D(
                [0],
                [0],
                color=TEXT_COLOR,
                marker="D",
                linestyle="none",
                markersize=6,
                label="All times combined",
            ),
        ],
        frameon=False,
        ncol=2,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.02),
        borderaxespad=0,
        columnspacing=1.0,
        handletextpad=0.6,
    )
    _clean_axis(ax)


def _top_overlap(ax: plt.Axes, results: LRComplexAggregationResults) -> None:
    boundaries = np.array([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0001])
    color_map = mpl.colors.ListedColormap(
        ["#B85C6B", "#D78993", "#E8B8B8", "#BFDAD7", "#66AFA9", "#087F8C"]
    )
    color_norm = mpl.colors.BoundaryNorm(boundaries, color_map.N)
    scatter = None
    for row, name in enumerate(DATASET_LABEL_ORDER):
        table = results.per_time_summary.loc[
            results.per_time_summary["dataset"].eq(name)
            & results.per_time_summary["scope"].eq("all_scored_pairs")
        ].sort_values("time")
        scatter = ax.scatter(
            table["normalized_time"],
            np.full(len(table), row),
            c=table["top_jaccard"],
            cmap=color_map,
            norm=color_norm,
            marker="s",
            s=150,
            edgecolor="white",
            linewidth=0.7,
        )
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.55, len(DATASET_LABEL_ORDER) - 0.45)
    ax.set_yticks(np.arange(len(DATASET_LABEL_ORDER)), DATASET_LABEL_ORDER)
    ax.invert_yaxis()
    ax.set_xlabel("Normalized developmental time")
    _clean_axis(ax)
    if scatter is None:
        raise ValueError("Top-pair overlap panel has no data")
    colorbar = ax.figure.colorbar(
        scatter,
        ax=ax,
        boundaries=boundaries,
        ticks=np.arange(0.4, 1.01, 0.1),
        pad=0.025,
        fraction=0.035,
    )
    colorbar.set_label("Top-10 Jaccard")
    colorbar.outline.set_edgecolor(TEXT_COLOR)
    colorbar.ax.tick_params(colors=TEXT_COLOR)


def plot_lr_complex_aggregation(
    results: LRComplexAggregationResults,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """Render the A4 panel layout."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    pdf = output / "lr_complex_aggregation.pdf"
    png = output / "lr_complex_aggregation.png"

    with mpl.rc_context(LR_COMPLEX_RC):
        figure = plt.figure(figsize=(8.27, 11.69))
        grid = figure.add_gridspec(
            4,
            2,
            height_ratios=[0.13, 1.0, 0.13, 1.28],
            hspace=0.2,
            wspace=0.42,
            left=0.13,
            right=0.91,
            top=0.91,
            bottom=0.08,
        )
        figure.suptitle(
            "Ligand–receptor complex aggregation sensitivity",
            fontsize=13,
            fontweight="bold",
            y=0.965,
        )
        _panel_heading(figure.add_subplot(grid[0, 0]), "a", "Multi-subunit coverage")
        _panel_heading(figure.add_subplot(grid[0, 1]), "b", "LR rank agreement")
        _coverage(figure.add_subplot(grid[1, 0]), results)
        _pooled_rank(figure.add_subplot(grid[1, 1]), results)
        _panel_heading(figure.add_subplot(grid[2, :]), "c", "Top-10 LR overlap")
        _top_overlap(figure.add_subplot(grid[3, :]), results)
        figure.savefig(pdf)
        figure.savefig(png, dpi=320)
        plt.close(figure)
    return pdf, png
