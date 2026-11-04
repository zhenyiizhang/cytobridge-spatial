#!/usr/bin/env python3
"""Draw the A4 Zebrafish decomposition-stability figure from archived tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


A4_PORTRAIT = (8.27, 11.69)
BLACK = "#111111"
GREY = "#8C959D"
GRID = "#DCE1E5"
NAVY = "#164B73"
TEAL = "#27857B"
ORANGE = "#E58B2A"
PURPLE = "#7A5AA6"
def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "text.color": BLACK,
            "axes.labelcolor": BLACK,
            "axes.titlecolor": BLACK,
            "axes.edgecolor": BLACK,
            "xtick.color": BLACK,
            "ytick.color": BLACK,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def heading(axis: plt.Axes, label: str, title: str, *, title_x: float = 0.065) -> None:
    axis.set_axis_off()
    axis.text(0.0, 0.52, label, fontsize=14, fontweight="bold", va="center")
    axis.text(title_x, 0.52, title, fontsize=12, fontweight="bold", va="center")


def clean_axis(axis: plt.Axes, *, grid_axis: str = "y") -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis=grid_axis, color=GRID, linewidth=0.55, zorder=0)
    axis.set_axisbelow(True)


def box_and_points(axis: plt.Axes, data: list[np.ndarray], labels: list[str], colors: list[str]) -> None:
    positions = np.arange(len(data), dtype=float)
    box = axis.boxplot(
        data,
        positions=positions,
        widths=0.48,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": BLACK, "linewidth": 1.0},
        whiskerprops={"color": GREY, "linewidth": 0.8},
        capprops={"color": GREY, "linewidth": 0.8},
        boxprops={"edgecolor": BLACK, "linewidth": 0.7},
    )
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.24)
    for index, (values, color) in enumerate(zip(data, colors)):
        jitter = np.linspace(-0.13, 0.13, len(values)) if len(values) > 1 else np.array([0.0])
        axis.scatter(
            np.full(len(values), index) + jitter,
            values,
            s=22,
            color=color,
            edgecolor="white",
            linewidth=0.45,
            zorder=3,
        )
    axis.set_xticks(positions, labels)


def panel_a(axis: plt.Axes, panel_data: Path) -> None:
    seed_components = pd.read_csv(panel_data / "training_seed_component_agreement.csv")
    data = []
    labels = ["Total\ndynamics", "Intrinsic-\ncontext", "Growth", "Directed-pair\nranking"]
    colors = [NAVY, TEAL, PURPLE, ORANGE]
    for name in ("total", "intrinsic", "growth"):
        space = "scalar" if name == "growth" else "state"
        values = seed_components.loc[
            seed_components["component"].eq(name) & seed_components["space"].eq(space),
            "cosine_median",
        ].to_numpy(float)
        data.append(values)
    pairs = pd.read_csv(panel_data / "training_seed_directed_pair_agreement.csv")
    data.append(pairs["directed_pair_spearman"].to_numpy(float))
    box_and_points(axis, data, labels, colors)
    axis.set_ylim(0.80, 1.005)
    axis.set_ylabel("Agreement")
    axis.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.2f"))
    clean_axis(axis)


def panel_b(component_axis: plt.Axes, ranking_axis: plt.Axes, panel_data: Path) -> None:
    setting_order = [
        "Neighborhood 0.8×",
        "Neighborhood 1.2×",
        "Expression loss weight 0.05",
        "Transport:mass weight 10:1",
    ]
    display = [
        "Neighborhood 0.8×",
        "Neighborhood 1.2×",
        "Expression weight 0.05",
        "OT:mass 10:1",
    ]
    components = pd.read_csv(panel_data / "model_setting_component_agreement.csv")
    interaction = components.loc[
        components["component"].eq("interaction") & components["space"].eq("state")
    ]
    for row_index, setting in enumerate(setting_order):
        values = interaction.loc[
            interaction["setting"].eq(setting), "cosine_median"
        ].to_numpy(float)
        jitter = np.linspace(-0.10, 0.10, len(values)) if len(values) > 1 else np.array([0.0])
        component_axis.scatter(
            values,
            np.full(len(values), row_index) + jitter,
            s=25,
            color=TEAL,
            edgecolor="none",
            zorder=3,
        )
    component_axis.set_yticks(np.arange(len(display)), display)
    component_axis.set_ylim(len(display) - 0.5, -0.5)
    component_axis.set_xlim(0.80, 1.005)
    component_axis.set_xlabel("Interaction-state cosine similarity")
    component_axis.set_title("Interaction component", pad=6)
    clean_axis(component_axis, grid_axis="x")

    pairs = pd.read_csv(panel_data / "model_setting_directed_pair_agreement.csv")
    metric_styles = (
        ("directed_pair_spearman", TEAL),
        ("top20_weighted_jaccard", ORANGE),
    )
    metric_offsets = (-0.11, 0.11)
    for metric_index, (metric, color) in enumerate(metric_styles):
        for row_index, setting in enumerate(setting_order):
            values = pairs.loc[pairs["setting"].eq(setting), metric].to_numpy(float)
            jitter = np.linspace(-0.045, 0.045, len(values)) if len(values) > 1 else np.array([0.0])
            ranking_axis.scatter(
                values,
                row_index + metric_offsets[metric_index] + jitter,
                s=25,
                color=color,
                marker="o",
                edgecolor="none",
                zorder=3,
            )
    ranking_axis.set_yticks(np.arange(len(display)), [])
    ranking_axis.set_ylim(len(display) - 0.5, -0.5)
    ranking_axis.set_xlim(0.78, 1.005)
    ranking_axis.set_xlabel("Directed-pair agreement")
    ranking_axis.set_title("Cell-type interaction ranking", pad=6)
    clean_axis(ranking_axis, grid_axis="x")


def panel_c(axis: plt.Axes, panel_data: Path) -> None:
    w1 = pd.read_csv(panel_data / "distribution_w1_summary.csv")
    matching = {}
    for seed in (42, 43, 44):
        matching[f"formal_seed{seed}_cutoff0p8"] = f"formal_seed{seed}_cutoff1p0"
        matching[f"formal_seed{seed}_cutoff1p2"] = f"formal_seed{seed}_cutoff1p0"
    matching["alpha_expr_005_seed42_cutoff1p0"] = "formal_seed42_cutoff1p0"
    matching["ot_mass_10_to_1_seed42_cutoff1p0"] = "formal_seed42_cutoff1p0"
    family_order = ["Neighborhood 0.8×", "Neighborhood 1.2×", "Expression weight 0.05", "OT:mass 10:1"]
    records = []
    for row in w1.itertuples(index=False):
        if row.space not in {"spatial", "state"} or row.condition not in matching:
            continue
        reference = matching[row.condition]
        denominator = float(w1.loc[w1["condition"].eq(reference) & w1["space"].eq(row.space), "w1_mean"].iloc[0])
        if "cutoff0p8" in row.condition:
            family = "Neighborhood 0.8×"
        elif "cutoff1p2" in row.condition:
            family = "Neighborhood 1.2×"
        elif row.condition.startswith("alpha_expr"):
            family = "Expression weight 0.05"
        else:
            family = "OT:mass 10:1"
        ratio = float(row.w1_mean) / denominator
        records.append({"family": family, "space": row.space, "change": 100.0 * (ratio - 1.0)})
    ratios = pd.DataFrame(records)
    offsets = {"spatial": -0.10, "state": 0.10}
    colors = {"spatial": ORANGE, "state": NAVY}
    for space in ("spatial", "state"):
        for family_index, family in enumerate(family_order):
            values = ratios.loc[ratios["family"].eq(family) & ratios["space"].eq(space), "change"].to_numpy(float)
            jitter = np.linspace(-0.045, 0.045, len(values)) if len(values) > 1 else np.array([0.0])
            axis.scatter(
                values,
                family_index + offsets[space] + jitter,
                s=24,
                color=colors[space],
                marker="o",
                edgecolor="none",
                zorder=3,
            )
    axis.axvline(0.0, color=GREY, linewidth=0.8, linestyle="--")
    axis.set_yticks(np.arange(len(family_order)), family_order)
    axis.set_ylim(len(family_order) - 0.5, -0.5)
    axis.set_xlim(-1.5, 4.1)
    axis.set_xticks([-1, 0, 1, 2, 3, 4])
    axis.set_xlabel("Change in W1 distance (%)")
    clean_axis(axis, grid_axis="x")


def main() -> int:
    args = arguments()
    panel_data = args.panel_data.expanduser().resolve(strict=True)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    apply_style()
    figure = plt.figure(figsize=A4_PORTRAIT)
    outer = figure.add_gridspec(
        6,
        1,
        height_ratios=(0.05, 0.18, 0.05, 0.22, 0.05, 0.19),
        left=0.20,
        right=0.965,
        top=0.975,
        bottom=0.065,
        hspace=0.34,
    )
    heading(figure.add_subplot(outer[0]), "a", "Agreement across training seeds")
    panel_a(figure.add_subplot(outer[1]), panel_data)
    b_heading = figure.add_subplot(outer[2])
    heading(b_heading, "b", "Sensitivity to model settings")
    b_heading.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor=TEAL, markeredgecolor="none", label="Rank agreement"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=ORANGE, markeredgecolor="none", label="Top-pair overlap"),
        ],
        frameon=False,
        ncol=2,
        loc="center right",
        bbox_to_anchor=(1.0, 0.52),
    )
    b_panels = outer[3].subgridspec(1, 2, width_ratios=(1.1, 0.9), wspace=0.25)
    panel_b(figure.add_subplot(b_panels[0]), figure.add_subplot(b_panels[1]), panel_data)
    c_heading = figure.add_subplot(outer[4])
    heading(c_heading, "c", "Reconstruction accuracy")
    c_heading.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor=ORANGE, markeredgecolor="none", label="Spatial"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=NAVY, markeredgecolor="none", label="State"),
        ],
        frameon=False,
        ncol=2,
        loc="center right",
        bbox_to_anchor=(1.0, 0.52),
    )
    panel_c(figure.add_subplot(outer[5]), panel_data)
    pdf_path = output_dir / "zebrafish_decomposition_stability_a4.pdf"
    png_path = output_dir / "zebrafish_decomposition_stability_a4.png"
    figure.savefig(pdf_path)
    figure.savefig(png_path, dpi=320)
    plt.close(figure)
    print(pdf_path)
    print(png_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
