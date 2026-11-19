#!/usr/bin/env python3
"""Draw the whole-tissue scale-1.0 Trem2 module response in panel-g style."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT.parent
SOURCE = (
    DATA_ROOT
    / "admouse_0815/trem2_whole_tissue_scale1_20260820/data"
    / "trem2_module_scores.csv"
)
DATA = ROOT / "data" / "ad_main_figureg" / "trem2_whole_tissue_scale1_module_contrasts.csv"
OUTPUT = ROOT / "figures" / "ad_main_figureg"
DISPLAY_TIME = 2.4

MODULES = [
    "DAM_microglia",
    "DAM_Lipid_Metabolism",
    "Lysosome_Phagosome",
    "AB_Clearance_Endolysosomal",
    "Inflammation_Complement",
    "Astrocyte_Reactive",
    "SPP1_CD44_axis",
]
LABELS = [
    "DAM",
    "DAM Lipid\nMetabolism",
    "Phagolysosome",
    "Aβ Clearance",
    "Complement\ninflammation",
    "Reactive\nAstrocytes",
    "SPP1 CD44\naxis",
]

ACTIVATION_COLOR = "#B5667A"
KNOCKDOWN_COLOR = "#9EBED0"


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 12,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 13,
            "axes.linewidth": 1.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def read_values() -> tuple[np.ndarray, np.ndarray]:
    table = pd.read_csv(SOURCE)
    selected = table[
        table["perturbed_gene"].eq("Trem2")
        & np.isclose(table["time"], DISPLAY_TIME)
        & table["population"].eq("all_particles")
        & table["module"].isin(MODULES)
    ].copy()
    selected.to_csv(DATA, index=False)
    activation = (
        selected[selected["direction"].eq("high")]
        .set_index("module")
        .reindex(MODULES)["delta"]
        .to_numpy(dtype=float)
    )
    knockdown = (
        selected[selected["direction"].eq("low")]
        .set_index("module")
        .reindex(MODULES)["delta"]
        .to_numpy(dtype=float)
    )
    if np.isnan(activation).any() or np.isnan(knockdown).any():
        raise ValueError("Missing requested scale-1.0 whole-tissue module values")
    return activation, knockdown


def main() -> None:
    configure_style()
    activation, knockdown = read_values()
    x = np.arange(len(MODULES), dtype=float)
    width = 0.36
    fig, ax = plt.subplots(figsize=(11.08, 5.48), dpi=180)

    ax.bar(
        x - width / 2,
        activation,
        width,
        color=ACTIVATION_COLOR,
        edgecolor="white",
        linewidth=0.7,
        label="In silico Trem2 Activation",
        zorder=3,
    )
    ax.bar(
        x + width / 2,
        knockdown,
        width,
        color=KNOCKDOWN_COLOR,
        edgecolor="white",
        linewidth=0.7,
        label="In silico Trem2 Knockdown",
        zorder=3,
    )

    ax.axhline(0, color="black", linestyle="--", linewidth=1.5, zorder=4)
    for value in (-0.5, 0.5):
        ax.axhline(value, color="#D9D9D9", linewidth=2.0, zorder=1)

    maximum = max(float(np.max(np.abs(activation))), float(np.max(np.abs(knockdown))))
    axis_limit = max(0.82, np.ceil((maximum + 0.08) * 10) / 10)
    ax.set_xlim(-0.5, len(MODULES) - 0.5)
    ax.set_ylim(-axis_limit, axis_limit)
    ax.set_yticks([-0.5, 0.5], ["−0.5", "0.5"])
    ax.set_ylabel("Module Score Change")
    ax.set_xticks(x, LABELS, rotation=48, ha="right", rotation_mode="anchor")
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", direction="out", width=1.3, length=5)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.5)

    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.005),
        ncol=2,
        frameon=False,
        handlelength=1.5,
        handletextpad=0.35,
        columnspacing=3.0,
        borderaxespad=0,
    )
    ax.text(
        -0.095,
        1.17,
        "g",
        transform=ax.transAxes,
        fontsize=22,
        fontweight="bold",
        ha="left",
        va="top",
    )
    ax.text(
        0.99,
        1.17,
        "Whole tissue | scale 1.0 | t=2.4",
        transform=ax.transAxes,
        fontsize=11,
        ha="right",
        va="top",
    )

    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.34, top=0.79)
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
