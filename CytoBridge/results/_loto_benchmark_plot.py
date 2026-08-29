"""Figure renderer for the five-dataset LOTO benchmark."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from ._style import (
    COMPARISON_COLOR,
    CYTOBRIDGE_COLOR,
    GRID_COLOR,
    LOTO_BENCHMARK_RC,
    save_figure,
)
from .loto_benchmark import DATASET_LABELS, LotoBenchmarkData


def _style_axis(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="y", color=GRID_COLOR, linewidth=0.5, alpha=0.75, zorder=0)
    axis.tick_params(labelsize=8.5)
    axis.set_axisbelow(True)


def _panel_heading(axis: plt.Axes, label: str, title: str) -> None:
    axis.axis("off")
    axis.text(
        0.0,
        0.54,
        label,
        ha="left",
        va="center",
        fontsize=14,
        fontweight="bold",
        color="#000000",
    )
    axis.text(
        0.075,
        0.54,
        title,
        ha="left",
        va="center",
        fontsize=12,
        fontweight="bold",
        color="#000000",
    )


def plot_loto_benchmark(
    data: LotoBenchmarkData,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """Render the A4 panel layout."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    pdf = output / "five_dataset_loto_benchmark.pdf"
    png = output / "five_dataset_loto_benchmark.png"
    datasets = list(data.protocol["datasets"])
    methods = list(data.protocol["comparison_method_order"])
    summary = data.dataset_summary
    maximum_ratio = float(summary["relative_sliced_w2"].max())
    y_max = max(3.75, math.ceil((maximum_ratio + 0.15) * 4.0) / 4.0)
    y_ticks = np.arange(0.0, math.floor(y_max) + 1.0, 1.0)

    with mpl.rc_context(LOTO_BENCHMARK_RC):
        figure = plt.figure(figsize=(8.27, 11.69))
        outer = figure.add_gridspec(
            4,
            2,
            left=0.09,
            right=0.975,
            top=0.93,
            bottom=0.065,
            wspace=0.22,
            hspace=0.29,
        )
        for index, (spec, method) in enumerate(zip(outer, methods)):
            inner = spec.subgridspec(2, 1, height_ratios=(0.15, 0.85), hspace=0.04)
            heading = figure.add_subplot(inner[0, 0])
            axis = figure.add_subplot(inner[1, 0])
            _panel_heading(
                heading,
                chr(ord("a") + index),
                data.protocol["display_names"][method],
            )
            panel = summary.loc[summary["method"].eq(method)].set_index("dataset")
            if set(panel.index) != set(datasets):
                raise ValueError(f"Dataset summary is incomplete for {method}")
            x = np.arange(len(datasets), dtype=float)
            width = 0.34
            ratios = np.asarray(
                [
                    float(panel.loc[dataset, "relative_sliced_w2"])
                    for dataset in datasets
                ]
            )
            axis.bar(
                x - width / 2,
                np.ones(len(datasets)),
                width=width,
                color=CYTOBRIDGE_COLOR,
                edgecolor="white",
                linewidth=0.55,
                zorder=2,
            )
            axis.bar(
                x + width / 2,
                ratios,
                width=width,
                color=COMPARISON_COLOR,
                edgecolor="white",
                linewidth=0.55,
                zorder=2,
            )
            axis.axhline(1.0, color="#000000", linewidth=0.65, zorder=1)
            axis.set_xticks(x)
            axis.set_xticklabels([DATASET_LABELS[item] for item in datasets])
            axis.set_ylim(0.0, y_max)
            axis.set_yticks(y_ticks)
            axis.margins(x=0.035)
            _style_axis(axis)
            if index % 2 == 0:
                axis.set_ylabel("Relative Sliced-W2\n(CytoBridge = 1)")

        figure.legend(
            handles=[
                mpatches.Patch(color=CYTOBRIDGE_COLOR, label="CytoBridge"),
                mpatches.Patch(color=COMPARISON_COLOR, label="Comparison method"),
            ],
            loc="upper center",
            bbox_to_anchor=(0.54, 0.982),
            ncol=2,
            frameon=False,
            fontsize=9,
            handlelength=1.2,
            columnspacing=1.1,
        )
        save_figure(
            figure,
            pdf,
            png,
            dpi=320,
            pdf_metadata={
                "CreationDate": None,
                "ModDate": None,
                "Creator": "CytoBridge",
                "Producer": "Matplotlib",
            },
            png_metadata={"Software": "CytoBridge"},
        )
        plt.close(figure)
    return pdf, png
