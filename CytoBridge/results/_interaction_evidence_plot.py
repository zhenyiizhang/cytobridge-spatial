"""Matplotlib renderer for the LR-prior and stVCR comparison."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ._io import prepare_output_dir
from ._style import (
    INTERACTION_A4_PORTRAIT,
    INTERACTION_EXTERNAL_COLOR,
    INTERACTION_FULL_COLOR,
    INTERACTION_NO_LR_COLOR,
    INTERACTION_MARK_COLOR,
    INTERACTION_RC,
    INTERACTION_SPACE_COLORS,
    INTERACTION_TEXT_COLOR,
    interaction_panel_heading,
    save_figure,
)
from .interaction_evidence import (
    DATASET_LABELS,
    DATASET_ORDER,
    EXTERNAL_COMPARISON,
    NO_LR_COMPARISON,
    SPACE_LABELS,
    SPACE_ORDER,
    LRPriorStVCRResults,
)


def _style_axis(ax: mpl.axes.Axes, *, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    ax.grid(
        axis=grid_axis,
        color="#D7DDE2",
        linewidth=0.5,
        alpha=0.75,
        zorder=0,
    )
    ax.tick_params(colors=INTERACTION_TEXT_COLOR)
    ax.xaxis.label.set_color(INTERACTION_TEXT_COLOR)
    ax.yaxis.label.set_color(INTERACTION_TEXT_COLOR)


def _summary_for(
    panel_summary: pd.DataFrame,
    *,
    comparison: str,
) -> pd.DataFrame:
    summary = panel_summary.loc[panel_summary["comparison"].eq(comparison)].copy()
    if len(summary) != len(DATASET_ORDER) * len(SPACE_ORDER):
        raise ValueError(f"Panel summary is incomplete for {comparison}")
    return summary


def _plot_normalized_aggregate(
    ax: mpl.axes.Axes,
    panel_summary: pd.DataFrame,
    *,
    comparison: str,
    baseline_label: str,
    comparison_label: str,
    baseline_color: str,
    comparison_color: str,
) -> None:
    summary = _summary_for(panel_summary, comparison=comparison)
    x_base = np.arange(len(DATASET_ORDER), dtype=float)
    width = 0.34
    relative = np.array(
        [
            float(
                summary.loc[
                    summary["dataset"].eq(dataset),
                    "dataset_mean_relative_difference",
                ].iloc[0]
            )
            for dataset in DATASET_ORDER
        ]
    )
    comparison_values = 1.0 + relative
    if not np.isfinite(comparison_values).all() or (comparison_values <= 0).any():
        raise ValueError("Normalized aggregate contains an invalid ratio")
    ax.bar(
        x_base - width / 2,
        np.ones(len(DATASET_ORDER)),
        width=width,
        color=baseline_color,
        edgecolor="white",
        linewidth=0.55,
        label=baseline_label,
        zorder=2,
    )
    ax.bar(
        x_base + width / 2,
        comparison_values,
        width=width,
        color=comparison_color,
        edgecolor="white",
        linewidth=0.55,
        label=comparison_label,
        zorder=2,
    )
    ax.set_xticks(x_base)
    ax.set_xticklabels([DATASET_LABELS[item] for item in DATASET_ORDER])
    ax.set_ylabel("Relative sliced-W2\n(reference = 1)")
    ax.set_ylim(0, float(max(1.0, comparison_values.max()) * 1.12))
    ax.margins(x=0.04)
    _style_axis(ax)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.0, 1.015),
        frameon=False,
        ncol=2,
        handlelength=1.2,
        columnspacing=1.1,
        borderaxespad=0,
    )


def _plot_relative_effects(
    ax: mpl.axes.Axes,
    raw_table: pd.DataFrame,
    panel_summary: pd.DataFrame,
    *,
    comparison: str,
    value_column: str,
    reference_label: str,
) -> None:
    summary = _summary_for(panel_summary, comparison=comparison)
    x_base = np.arange(len(DATASET_ORDER), dtype=float)
    offsets = {"joint": -0.23, "spatial": 0.0, "state": 0.23}
    bar_width = 0.19
    extrema = [0.0]
    for space in SPACE_ORDER:
        space_summary = summary.loc[summary["space"].eq(space)].set_index("dataset")
        means = [
            100.0 * float(space_summary.loc[dataset, "mean_relative_difference"])
            for dataset in DATASET_ORDER
        ]
        errors = [
            100.0 * float(space_summary.loc[dataset, "sem"])
            for dataset in DATASET_ORDER
        ]
        raw_by_dataset = [
            100.0
            * raw_table.loc[
                raw_table["dataset"].eq(dataset) & raw_table["space"].eq(space),
                value_column,
            ].to_numpy(dtype=float)
            for dataset in DATASET_ORDER
        ]
        positions = x_base + offsets[space]
        ax.bar(
            positions,
            means,
            width=bar_width,
            color=INTERACTION_SPACE_COLORS[space],
            alpha=0.88,
            edgecolor="white",
            linewidth=0.5,
            zorder=2,
            label=SPACE_LABELS[space],
        )
        ax.errorbar(
            positions,
            means,
            yerr=errors,
            fmt="none",
            ecolor=INTERACTION_MARK_COLOR,
            elinewidth=0.7,
            capsize=2.0,
            capthick=0.7,
            zorder=4,
        )
        for mean_value, error_value in zip(means, errors):
            if np.isfinite(error_value):
                extrema.extend([mean_value - error_value, mean_value + error_value])
        for dataset_index, values in enumerate(raw_by_dataset):
            extrema.extend(values.tolist())
            jitter = (
                np.array([0.0])
                if len(values) == 1
                else np.linspace(-0.035, 0.035, len(values))
            )
            ax.scatter(
                positions[dataset_index] + jitter,
                values,
                s=13,
                facecolor="white",
                edgecolor=INTERACTION_MARK_COLOR,
                linewidth=0.55,
                zorder=5,
            )
    low, high = float(min(extrema)), float(max(extrema))
    span = max(high - low, 1.0)
    padding = span * 0.12
    ax.set_ylim(low - padding, high + padding)
    ax.axhline(0, color=INTERACTION_MARK_COLOR, linewidth=0.7, zorder=1)
    ax.set_xticks(x_base)
    ax.set_xticklabels([DATASET_LABELS[item] for item in DATASET_ORDER])
    ax.set_ylabel(f"Relative sliced-W2 change\nvs {reference_label} (%)")
    ax.margins(x=0.03)
    _style_axis(ax)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.0, 1.015),
        frameon=False,
        ncol=3,
        handlelength=1.2,
        columnspacing=0.9,
        borderaxespad=0,
    )


def _make_figure(results: LRPriorStVCRResults) -> plt.Figure:
    figure = plt.figure(figsize=INTERACTION_A4_PORTRAIT)
    outer = figure.add_gridspec(
        nrows=2,
        ncols=2,
        left=0.09,
        right=0.97,
        bottom=0.07,
        top=0.975,
        wspace=0.24,
        hspace=0.25,
    )
    panels = []
    for spec in outer:
        inner = spec.subgridspec(
            nrows=2,
            ncols=1,
            height_ratios=[0.12, 0.88],
            hspace=0.03,
        )
        panels.append((figure.add_subplot(inner[0]), figure.add_subplot(inner[1])))

    heading, axis = panels[0]
    interaction_panel_heading(heading, "a", "LR-prior ablation")
    _plot_normalized_aggregate(
        axis,
        results.panel_summary,
        comparison=NO_LR_COMPARISON,
        baseline_label="Full model",
        comparison_label="No LR prior",
        baseline_color=INTERACTION_FULL_COLOR,
        comparison_color=INTERACTION_NO_LR_COLOR,
    )
    heading, axis = panels[1]
    interaction_panel_heading(heading, "b", "Matched effect by evaluation space")
    _plot_relative_effects(
        axis,
        results.no_lr,
        results.panel_summary,
        comparison=NO_LR_COMPARISON,
        value_column="no_lr_prior_relative_to_full",
        reference_label="Full model",
    )
    heading, axis = panels[2]
    interaction_panel_heading(heading, "c", "Held-out method comparison")
    _plot_normalized_aggregate(
        axis,
        results.panel_summary,
        comparison=EXTERNAL_COMPARISON,
        baseline_label="CytoBridge",
        comparison_label="stVCR",
        baseline_color=INTERACTION_FULL_COLOR,
        comparison_color=INTERACTION_EXTERNAL_COLOR,
    )
    heading, axis = panels[3]
    interaction_panel_heading(heading, "d", "Method comparison by evaluation space")
    _plot_relative_effects(
        axis,
        results.stvcr,
        results.panel_summary,
        comparison=EXTERNAL_COMPARISON,
        value_column="stvcr_relative_to_cytobridge",
        reference_label="CytoBridge",
    )
    return figure


def plot_lr_prior_stvcr(
    results: LRPriorStVCRResults,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """Draw the S39 comparison as PDF and PNG."""

    output = prepare_output_dir(output_dir)
    pdf = output / "lr_prior_stvcr_comparison.pdf"
    png = output / "lr_prior_stvcr_comparison.png"
    with mpl.rc_context(INTERACTION_RC):
        figure = _make_figure(results)
        save_figure(
            figure,
            pdf,
            png,
            dpi=320,
            pdf_metadata={"Creator": "CytoBridge"},
            png_metadata={"Software": "CytoBridge"},
        )
        plt.close(figure)
    return pdf, png


plot_interaction_evidence = plot_lr_prior_stvcr
