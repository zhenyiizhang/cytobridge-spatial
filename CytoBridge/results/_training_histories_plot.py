"""Matplotlib renderer for the six-stage training-history figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

from ._io import prepare_output_dir
from .training_histories import (
    STAGES,
    TrainingHistoryResults,
    calculate_smoothed_training_history,
)


GRID_COLOR = "#D7DDE2"
TRAINING_HISTORY_RC = {
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
    "text.color": "#000000",
    "axes.labelcolor": "#000000",
    "axes.titlecolor": "#000000",
    "xtick.color": "#000000",
    "ytick.color": "#000000",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
}


def _panel_heading(axis: mpl.axes.Axes, label: str, title: str) -> None:
    axis.axis("off")
    axis.text(
        0.0,
        0.55,
        label,
        fontsize=14.0,
        fontweight="bold",
        va="center",
        ha="left",
        color="#000000",
    )
    axis.text(
        0.13,
        0.55,
        title,
        fontsize=12.0,
        fontweight="bold",
        va="center",
        ha="left",
        color="#000000",
    )


def _clean_axis(axis: mpl.axes.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(color=GRID_COLOR, linewidth=0.45, alpha=0.55)
    axis.set_axisbelow(True)


def plot_training_histories(
    results: TrainingHistoryResults,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """Render the six-stage training-history figure as PDF and PNG."""

    output = prepare_output_dir(output_dir)
    pdf = output / "representative_training_curves.pdf"
    png = output / "representative_training_curves.png"
    history = calculate_smoothed_training_history(results)

    with mpl.rc_context(TRAINING_HISTORY_RC):
        figure = plt.figure(figsize=(10.8, 6.4))
        grid = figure.add_gridspec(
            nrows=4,
            ncols=3,
            height_ratios=[0.14, 1.0, 0.14, 1.0],
            left=0.07,
            right=0.985,
            top=0.975,
            bottom=0.09,
            hspace=0.42,
            wspace=0.37,
        )

        for index, stage in enumerate(STAGES):
            block = index // 3
            column = index % 3
            heading = figure.add_subplot(grid[block * 2, column])
            _panel_heading(
                heading,
                "abcdef"[index],
                f"ARISTA — {stage.label}",
            )

            axis = figure.add_subplot(grid[block * 2 + 1, column])
            current = history.loc[
                history["stage_index"].eq(stage.stage_index)
                & history["stage"].eq(stage.stage)
            ]
            axis.plot(
                current["epoch"].to_numpy(dtype=int),
                current["loss"].to_numpy(dtype=float),
                color=stage.color,
                alpha=0.11,
                linewidth=0.55,
            )
            axis.plot(
                current["epoch"].to_numpy(dtype=int),
                current["smoothed_loss"].to_numpy(dtype=float),
                color=stage.color,
                linewidth=1.9,
            )
            axis.set_xlabel("Epoch")
            axis.set_ylabel(stage.y_label)
            axis.margins(x=0)
            _clean_axis(axis)

        figure.savefig(pdf, facecolor="white")
        figure.savefig(png, dpi=420, facecolor="white")
        plt.close(figure)
    return pdf, png
