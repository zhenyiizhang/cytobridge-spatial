#!/usr/bin/env python3
"""Draw formal AdMouse LR dynamics with color-only encoding.

The formal r2 input, pair ordering, within-pair z-score colors, axes, and
layout are retained. All dots have one fixed area; raw-score size encoding and
its annotation are intentionally omitted.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "lr_pair_timecourse.csv"
FIGURE_DIR = ROOT / "figures"
OUTPUT_STEM = "admouse_lr_interaction_dynamics_color_only"
DOT_SIZE = 55.0
DISPLAY_MAX_TIME = 2.4


def style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial"],
            "font.size": 8.5,
            "axes.titlesize": 9,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "axes.linewidth": 0.65,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def panel_label(ax: mpl.axes.Axes, label: str) -> None:
    ax.text(
        -0.08,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def clean_axis(ax: mpl.axes.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(False)


def lr_matrix() -> tuple[pd.DataFrame, pd.DataFrame]:
    lr = pd.read_csv(DATA_FILE)
    full_wide = lr.pivot(index="pair", columns="time", values="score").fillna(0.0)
    order = full_wide.max(axis=1).sort_values(ascending=False).index.tolist()
    full_wide = full_wide.loc[order]
    z = full_wide.sub(full_wide.mean(axis=1), axis=0)
    sd = full_wide.std(axis=1).replace(0, np.nan)
    z = z.div(sd, axis=0).fillna(0.0)
    # Keep the formal full-series ordering and z-score normalization; only
    # remove t=2.5 from the displayed matrix.
    display_columns = [
        column for column in full_wide.columns if float(column) <= DISPLAY_MAX_TIME
    ]
    wide = full_wide.loc[:, display_columns]
    z = z.loc[:, display_columns]
    return wide, z


def draw_lr_dotplot(ax: mpl.axes.Axes) -> mpl.collections.PathCollection:
    wide, z = lr_matrix()
    times = wide.columns.astype(float).to_numpy()
    xx, yy = np.meshgrid(np.arange(len(times)), np.arange(len(wide)))
    dots = ax.scatter(
        xx.ravel(),
        yy.ravel(),
        s=DOT_SIZE,
        c=z.to_numpy(float).ravel(),
        cmap="RdBu_r",
        vmin=-2.3,
        vmax=2.3,
        edgecolor="#444444",
        linewidth=0.25,
    )
    ax.set_yticks(np.arange(len(wide)), [x.replace("_", "–") for x in wide.index])
    tick_idx = np.arange(0, len(times), 5)
    if tick_idx[-1] != len(times) - 1:
        tick_idx = np.append(tick_idx, len(times) - 1)
    ax.set_xticks(tick_idx, [f"{times[i]:.1f}" for i in tick_idx])
    ax.invert_yaxis()
    ax.set_xlabel("Model time")
    ax.set_ylabel("Ligand–receptor pair")
    ax.set_title("Model-derived LR interaction dynamics", loc="left", pad=5)
    clean_axis(ax)
    return dots


def main() -> None:
    style()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.27, 4.85))
    fig.subplots_adjust(left=0.20, right=0.89, bottom=0.16, top=0.88)
    dots = draw_lr_dotplot(ax)
    panel_label(ax, "e")
    colorbar = fig.colorbar(dots, ax=ax, fraction=0.035, pad=0.025)
    colorbar.set_label("Within-pair z-score")
    fig.suptitle(
        "AdMouse ligand–receptor dynamics | current formal continuous-t0 run",
        y=0.97,
    )
    fig.savefig(FIGURE_DIR / f"{OUTPUT_STEM}.pdf", bbox_inches="tight")
    fig.savefig(FIGURE_DIR / f"{OUTPUT_STEM}.png", dpi=320, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
