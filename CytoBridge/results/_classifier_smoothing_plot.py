"""Matplotlib renderer for classifier spatial-smoothing sensitivity."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

from ._io import prepare_output_dir
from ._style import (
    CLASSIFIER_COLORS,
    CLASSIFIER_RC,
    TEXT_COLOR,
    clean_axis,
    panel_heading,
    save_figure,
)
from .classifier_smoothing import (
    DATASET_LABELS,
    DATASET_ORDER,
    FORMAL_K,
    K_VALUES,
    ClassifierSmoothingResults,
)


def plot_classifier_smoothing(
    results: ClassifierSmoothingResults,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """Render the classifier-smoothing figure as PDF and PNG."""

    output = prepare_output_dir(output_dir)
    pdf = output / "classifier_spatial_smoothing_sensitivity.pdf"
    png = output / "classifier_spatial_smoothing_sensitivity.png"

    with mpl.rc_context(CLASSIFIER_RC):
        figure = plt.figure(figsize=(11.69, 8.27))
        grid = figure.add_gridspec(
            4,
            2,
            height_ratios=(0.20, 2.05, 0.20, 2.20),
            left=0.07,
            right=0.98,
            top=0.95,
            bottom=0.10,
            hspace=0.50,
            wspace=0.30,
        )

        head_a = figure.add_subplot(grid[0, :])
        panel_heading(head_a, "a", "Held-out classification")
        ax_a = figure.add_subplot(grid[1, :])

        dataset_x = np.arange(len(DATASET_ORDER), dtype=float)
        bar_width = 0.15
        offsets = (np.arange(len(K_VALUES)) - 2.0) * bar_width
        for offset, k in zip(offsets, K_VALUES):
            values = []
            for dataset in DATASET_ORDER:
                row = results.metrics.loc[
                    results.metrics["dataset"].eq(dataset)
                    & results.metrics["k"].astype(int).eq(k)
                ].iloc[0]
                values.append(float(row["balanced_accuracy"]))
            ax_a.bar(
                dataset_x + offset,
                values,
                width=bar_width,
                color=CLASSIFIER_COLORS[k],
                edgecolor="white",
                linewidth=0.35,
                zorder=3,
            )
        ax_a.set_xticks(
            dataset_x,
            [
                f"{DATASET_LABELS[dataset]}\nformal k = {FORMAL_K[dataset]}"
                for dataset in DATASET_ORDER
            ],
        )
        ax_a.set_ylabel("Held-out balanced accuracy")
        ax_a.set_ylim(0.0, 1.02)
        clean_axis(ax_a)
        head_a.legend(
            handles=[
                Patch(
                    facecolor=CLASSIFIER_COLORS[k], edgecolor="none", label=f"k = {k}"
                )
                for k in K_VALUES
            ],
            ncol=5,
            loc="center right",
            bbox_to_anchor=(1.0, 0.52),
            frameon=False,
            handlelength=1.2,
            columnspacing=1.1,
        )

        head_b = figure.add_subplot(grid[2, 0])
        panel_heading(head_b, "b", "Zebrafish composition sensitivity")
        head_c = figure.add_subplot(grid[2, 1])
        panel_heading(head_c, "c", "Zebrafish transition fraction")
        ax_b = figure.add_subplot(grid[3, 0])
        ax_c = figure.add_subplot(grid[3, 1])

        k_x = np.arange(len(K_VALUES), dtype=float)
        ax_b.bar(
            k_x,
            results.composition["mean"],
            yerr=results.composition["sem"],
            color=[CLASSIFIER_COLORS[k] for k in K_VALUES],
            edgecolor="white",
            linewidth=0.35,
            capsize=2.5,
            error_kw={"elinewidth": 0.8, "ecolor": TEXT_COLOR},
            zorder=3,
        )
        ax_b.set_xticks(k_x, [str(k) for k in K_VALUES])
        ax_b.set_xlabel("Spatial neighborhood, k")
        ax_b.set_ylabel("Composition TV from k = 1 (%)")
        ax_b.set_ylim(bottom=0)
        clean_axis(ax_b)

        comparison_k = (1, 10)
        comparison = results.transition.loc[
            results.transition["k"].astype(int).isin(comparison_k)
        ].copy()
        comparison = comparison.set_index("k").loc[list(comparison_k)].reset_index()
        comparison_x = np.arange(len(comparison_k), dtype=float)
        interval_pairs = sorted(
            {
                (float(row.time_from), float(row.time_to))
                for row in results.intervals.itertuples(index=False)
            }
        )
        interval_colors = mpl.colormaps["viridis"](
            np.linspace(0.10, 0.82, len(interval_pairs))
        )
        for color, (time_from, time_to) in zip(interval_colors, interval_pairs):
            interval_rows = results.intervals.loc[
                results.intervals["time_from"].astype(float).eq(time_from)
                & results.intervals["time_to"].astype(float).eq(time_to)
                & results.intervals["k"].astype(int).isin(comparison_k)
            ].copy()
            interval_values = 100.0 * interval_rows.set_index(
                interval_rows["k"].astype(int)
            ).loc[list(comparison_k), "transition_fraction"].to_numpy(dtype=float)
            ax_c.plot(
                comparison_x,
                interval_values,
                color=color,
                marker="o",
                markersize=5.0,
                linewidth=1.2,
                label=f"{time_from:g}→{time_to:g}",
                zorder=3,
            )
        ax_c.errorbar(
            comparison_x,
            comparison["mean"],
            yerr=comparison["sem"],
            fmt="D",
            markersize=5.0,
            markerfacecolor="white",
            markeredgecolor=TEXT_COLOR,
            color=TEXT_COLOR,
            linewidth=1.0,
            capsize=3.0,
            label="Mean ± s.e.m.",
            zorder=4,
        )
        ax_c.set_xticks(comparison_x, ("No smoothing\nk = 1", "Formal\nk = 10"))
        ax_c.set_ylabel("Particles changing label (%)")
        ax_c.set_ylim(0, 102)
        ax_c.set_xlim(-0.18, 1.18)
        clean_axis(ax_c)
        ax_c.legend(
            title="Time interval",
            loc="lower center",
            bbox_to_anchor=(0.5, 0.04),
            ncol=3,
            frameon=False,
            handlelength=1.7,
            columnspacing=1.2,
        )

        save_figure(figure, pdf, png, dpi=320)
        plt.close(figure)
    return pdf, png
