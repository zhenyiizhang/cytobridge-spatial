#!/usr/bin/env python3
"""Reproduce Supplementary Figure S46 from its archived source tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


DATASETS = ("zebrafish", "mosta", "arista", "chicken_heart")
DISPLAY_NAMES = {
    "zebrafish": "Zebrafish",
    "mosta": "MOSTA",
    "arista": "ARISTA",
    "chicken_heart": "Chicken heart",
}
RECEIVER_DISPLAY = {
    "zebrafish": "Musculature / YSL",
    "mosta": "Connective tissue",
    "arista": "tlNBL",
    "chicken_heart": "Valve cells",
}
A4_PORTRAIT = (8.27, 11.69)
ACCENT = "#277F7A"
SECONDARY_GREY = "#9AA1A6"
GRID_COLOR = "#D9DEE2"


def apply_style() -> None:
    mpl.rcParams.update(
        {
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
            "text.color": "black",
            "axes.labelcolor": "black",
            "axes.titlecolor": "black",
            "axes.edgecolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def read_panel_tables(panel_data_dir: Path) -> dict[str, pd.DataFrame]:
    filenames = {
        "metrics": "global_pair_metrics.csv",
        "support": "model_linked_external_support.csv",
        "molecular": "model_biology_molecular_panel.csv",
        "chains": "model_first_nichenet_chains.csv",
    }
    tables = {
        name: pd.read_csv(panel_data_dir / filename)
        for name, filename in filenames.items()
    }
    metrics = tables["metrics"]
    required_metrics = {
        "dataset",
        "cytobridge_view",
        "external_method",
        "spearman_rho",
        "top_jaccard",
        "metric_available",
    }
    if not required_metrics.issubset(metrics.columns):
        raise ValueError("global_pair_metrics.csv does not have the required columns")
    for name in ("support", "molecular"):
        found = set(tables[name]["dataset"].astype(str))
        if not set(DATASETS).issubset(found):
            raise ValueError(f"{filenames[name]} does not cover all displayed datasets")
    found_chains = set(tables["chains"]["dataset"].astype(str))
    if not set(DATASETS).issubset(found_chains):
        raise ValueError("model_first_nichenet_chains.csv is incomplete")
    return tables


def panel_heading(axis: plt.Axes, label: str, title: str) -> None:
    axis.set_axis_off()
    axis.text(0.0, 0.52, label, fontsize=14, fontweight="bold", va="center")
    axis.text(0.065, 0.52, title, fontsize=12, fontweight="bold", va="center")


def draw_metric_axis(
    axis: plt.Axes,
    method_tables: dict[str, pd.DataFrame],
    *,
    metric: str,
    title: str,
    x_label: str,
    show_y: bool,
) -> None:
    methods = ("COMMOT", "CellAgentChat")
    offsets = (-0.10, 0.10)
    markers = {"COMMOT": "s", "CellAgentChat": "D"}
    colors = {"COMMOT": ACCENT, "CellAgentChat": SECONDARY_GREY}
    for method, offset in zip(methods, offsets, strict=True):
        table = method_tables[method]
        x_values = []
        y_values = []
        for row_index, dataset in enumerate(DATASETS):
            row = table.loc[dataset]
            available = str(row.metric_available).casefold() in {"true", "1"}
            value = float(row[metric]) if available else np.nan
            if np.isfinite(value):
                x_values.append(value)
                y_values.append(row_index + offset)
        axis.scatter(
            x_values,
            y_values,
            s=42,
            marker=markers[method],
            color=colors[method],
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )
    axis.set_xlim(-0.03, 1.05)
    axis.set_ylim(len(DATASETS) - 0.5, -0.5)
    axis.set_yticks(
        np.arange(len(DATASETS), dtype=float),
        [DISPLAY_NAMES[name] for name in DATASETS] if show_y else [],
    )
    axis.set_title(title, fontsize=9.2, pad=7)
    axis.set_xlabel(x_label, fontsize=8.5)
    axis.grid(axis="x", color=GRID_COLOR, linewidth=0.6)
    axis.axvline(0.0, color="#9AA3AA", linewidth=0.8, zorder=1)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(labelsize=7.5)


def draw_selection_table(
    axis: plt.Axes,
    support: pd.DataFrame,
    molecular: pd.DataFrame,
) -> None:
    axis.set_axis_off()
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(-0.55, 4.62)
    axis.text(0.44, 4.38, "CytoBridge selection", fontsize=7.8, fontweight="bold", ha="center")
    axis.text(0.885, 4.38, "COMMOT comparison", fontsize=7.8, fontweight="bold", ha="center")
    axis.plot([0.13, 0.745], [4.18, 4.18], color="black", linewidth=0.55)
    axis.plot([0.77, 1.0], [4.18, 4.18], color="black", linewidth=0.55)
    axis.text(0.00, 3.91, "Dataset", fontsize=7.1, fontweight="bold")
    axis.text(0.13, 3.91, "Sender", fontsize=7.1, fontweight="bold")
    axis.text(0.29, 3.91, "Receiver", fontsize=7.1, fontweight="bold")
    axis.text(0.565, 3.99, "Selected\nLR axis", fontsize=6.8, fontweight="bold", ha="center", va="top", linespacing=1.1)
    axis.text(0.705, 3.99, "Database\npathway", fontsize=6.8, fontweight="bold", ha="center", va="top", linespacing=1.1)
    axis.text(0.835, 3.99, "Same-pair\nLR rank", fontsize=6.8, fontweight="bold", ha="center", va="top", linespacing=1.1)
    axis.text(0.95, 3.99, "Pair\npercentile", fontsize=6.8, fontweight="bold", ha="center", va="top", linespacing=1.1)
    axis.plot([0.0, 1.0], [3.50, 3.50], color="black", linewidth=0.75)
    for row_index, dataset in enumerate(DATASETS):
        row = support.loc[dataset]
        molecular_row = molecular.loc[dataset]
        y = float(len(DATASETS) - 1 - row_index)
        if row_index:
            axis.plot([0.0, 1.0], [y + 0.50, y + 0.50], color=GRID_COLOR, linewidth=0.55)
        axis.text(0.00, y, DISPLAY_NAMES[dataset], fontsize=7.5, fontweight="bold", va="center")
        axis.text(0.13, y, str(row.sender_type), fontsize=7.2, va="center")
        axis.text(0.29, y, RECEIVER_DISPLAY[dataset], fontsize=7.2, va="center")
        axis.text(0.51, y, f"{str(row.ligand).upper()}–{str(row.receptor).upper()}", fontsize=7.5, fontweight="bold", va="center")
        axis.text(0.66, y, str(molecular_row.cytobridge_pathway), fontsize=7.2, va="center")
        axis.text(0.835, y, f"{int(molecular_row.commot_within_pair_rank)} / {int(molecular_row.within_pair_lr_count):,}", fontsize=7.2, ha="center", va="center")
        axis.text(0.95, y, f"{100 * float(row.commot_pair_percentile):.1f}", fontsize=7.2, ha="center", va="center")
    axis.plot([0.0, 1.0], [-0.50, -0.50], color="black", linewidth=0.75)


def draw_target_table(axis: plt.Axes, chains: pd.DataFrame) -> None:
    axis.set_axis_off()
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(-0.62, 4.38)
    axis.text(0.40, 4.18, "CytoBridge selection", fontsize=7.8, fontweight="bold", ha="center")
    axis.text(0.865, 4.18, "COMMOT and NicheNet results", fontsize=7.8, fontweight="bold", ha="center")
    axis.plot([0.14, 0.64], [3.99, 3.99], color="black", linewidth=0.55)
    axis.plot([0.66, 1.0], [3.99, 3.99], color="black", linewidth=0.55)
    axis.text(0.00, 3.72, "Dataset", fontsize=7.0, fontweight="bold")
    axis.text(0.14, 3.79, "Directed pair\n(sender; receiver)", fontsize=6.8, fontweight="bold", va="top", linespacing=1.1)
    axis.text(0.40, 3.79, "LR axis\nDatabase pathway", fontsize=6.8, fontweight="bold", va="top", linespacing=1.1)
    axis.text(0.70, 3.86, "COMMOT\nsame-axis\npercentile", fontsize=6.5, fontweight="bold", ha="center", va="top", linespacing=1.0)
    axis.text(0.875, 3.86, "NicheNet-predicted\nreceiver\ntargets", fontsize=6.5, fontweight="bold", ha="center", va="top", linespacing=1.0)
    axis.plot([0.0, 1.0], [3.34, 3.34], color="black", linewidth=0.75)
    for row_index, dataset in enumerate(DATASETS):
        group = chains.loc[chains["dataset"].astype(str).eq(dataset)].sort_values("receiver_target_rank")
        first = group.iloc[0]
        targets = ", ".join(group["receiver_target"].astype(str).tolist())
        y = float(2.95 - row_index)
        if row_index:
            axis.plot([0.0, 1.0], [y + 0.50, y + 0.50], color=GRID_COLOR, linewidth=0.55)
        axis.text(0.00, y, DISPLAY_NAMES[dataset], fontsize=7.5, fontweight="bold", va="center")
        axis.text(0.14, y + 0.12, f"Sender: {first.sender_type}", fontsize=7.2, va="center")
        axis.text(0.14, y - 0.14, f"Receiver: {first.receiver_type}", fontsize=7.2, va="center")
        axis.text(0.40, y + 0.12, f"{str(first.ligand).upper()}–{str(first.receptor).upper()}", fontsize=7.5, fontweight="bold", va="center")
        axis.text(0.40, y - 0.14, str(first.pathways), fontsize=7.2, va="center")
        axis.text(0.70, y, f"{100 * float(first.commot_percentile):.1f}", fontsize=7.5, ha="center", va="center")
        axis.text(0.80, y, targets, fontsize=7.4, va="center")
    axis.plot([0.0, 1.0], [-0.50, -0.50], color="black", linewidth=0.75)


def plot_figure(panel_data_dir: Path, output_dir: Path) -> tuple[Path, Path]:
    tables = read_panel_tables(panel_data_dir)
    metrics = tables["metrics"]
    metrics = metrics.loc[
        metrics["cytobridge_view"].eq("CytoBridge exact message")
        & metrics["external_method"].isin(["COMMOT", "CellAgentChat"])
    ]
    method_tables = {
        method: metrics.loc[metrics["external_method"].eq(method)].set_index("dataset")
        for method in ("COMMOT", "CellAgentChat")
    }
    support = tables["support"].set_index("dataset")
    molecular = tables["molecular"].set_index("dataset")
    chains = tables["chains"]

    apply_style()
    fig = plt.figure(figsize=A4_PORTRAIT)
    outer = fig.add_gridspec(
        6,
        1,
        height_ratios=(0.065, 0.225, 0.065, 0.255, 0.065, 0.305),
        left=0.12,
        right=0.965,
        top=0.97,
        bottom=0.07,
        hspace=0.28,
    )
    panel_heading(fig.add_subplot(outer[0]), "a", "Directed-pair consistency")
    panel_a = outer[1].subgridspec(1, 2, wspace=0.27)
    rho_axis = fig.add_subplot(panel_a[0])
    jaccard_axis = fig.add_subplot(panel_a[1])
    draw_metric_axis(rho_axis, method_tables, metric="spearman_rho", title="Rank agreement", x_label="Spearman rank correlation (ρ)", show_y=True)
    draw_metric_axis(jaccard_axis, method_tables, metric="top_jaccard", title="Top-ranked pair overlap", x_label="Top-20% directed-pair Jaccard", show_y=False)
    rho_axis.legend(
        handles=[
            Line2D([0], [0], marker="s", color="none", markerfacecolor=ACCENT, markeredgecolor="white", markersize=7, label="COMMOT"),
            Line2D([0], [0], marker="D", color="none", markerfacecolor=SECONDARY_GREY, markeredgecolor="white", markersize=6.5, label="CellAgentChat"),
        ],
        frameon=False,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.11),
        ncol=2,
        fontsize=7.2,
        handletextpad=0.45,
        columnspacing=1.2,
    )
    panel_heading(fig.add_subplot(outer[2]), "b", "CytoBridge-selected LR axes")
    draw_selection_table(fig.add_subplot(outer[3]), support, molecular)
    panel_heading(fig.add_subplot(outer[4]), "c", "NicheNet receiver targets for CytoBridge-selected axes")
    draw_target_table(fig.add_subplot(outer[5]), chains)

    output_dir.mkdir(parents=True, exist_ok=True)
    pdf = output_dir / "spatial_communication_comparison_s46.pdf"
    png = output_dir / "spatial_communication_comparison_s46.png"
    fig.savefig(pdf, facecolor="white")
    fig.savefig(png, dpi=320, facecolor="white")
    plt.close(fig)
    return pdf, png


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    plot_figure(args.panel_data_dir.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    main()
