#!/usr/bin/env python3
"""Plot selected AdMouse spatial snapshots from the r2 baseline rollout.

This script does not run inference or modify source data. It reads only the
saved baseline states and k=1 labels from the formal continuous-t0 r2 run.
Requested physical
ages are converted to normalized model time by piecewise-linear interpolation
between the formal anchors 2.5 months -> 0, 5.7 months -> 1, and
17.9 months -> 2. The nearest saved 0.1-time snapshot is then read.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


FINAL_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = FINAL_ROOT.parent
R2_ROOT = (
    DATA_ROOT.parent
    / "cytobridge/projects/CytoBridge-ST-1104/runs"
    / "admouse-lr-perturbation-20260814-3c87a3e-r2"
)
STATE_DIR = R2_ROOT / "compat_base" / "01_interpolation"
LABEL_DIR = R2_ROOT / "whole_tissue" / "baseline_labels_k1"
OUTPUT_DIR = FINAL_ROOT / "figures"

AGE_ANCHORS = np.array([2.5, 5.7, 17.9], dtype=float)
MODEL_TIME_ANCHORS = np.array([0.0, 1.0, 2.0], dtype=float)
REQUESTED_AGES = [3.8, 5.7, 8.1]

CELLTYPE_ORDER = [
    "Astrocytes",
    "Excitatory neurons",
    "Fibroblast",
    "Inhibitory neurons",
    "Microglia",
    "OPC",
    "Oligodendrocytes",
    "Pericytes/Endothelial",
]

# Original AdMouse manuscript palette.
CELLTYPE_COLORS = {
    "Astrocytes": "#1f77b4",
    "Excitatory neurons": "#ff7f0e",
    "Fibroblast": "#2ca02c",
    "Inhibitory neurons": "#d62728",
    "Microglia": "#9467bd",
    "OPC": "#8c564b",
    "Oligodendrocytes": "#e377c2",
    "Pericytes/Endothelial": "#7f7f7f",
}


def age_to_model_time(age_months: float) -> float:
    """Invert the formal piecewise-linear model-time-to-age mapping."""

    return float(np.interp(age_months, AGE_ANCHORS, MODEL_TIME_ANCHORS))


def saved_time_for_age(age_months: float) -> float:
    """Return the nearest available 0.1 model-time snapshot."""

    return round(age_to_model_time(age_months) * 10.0) / 10.0


def time_token(model_time: float) -> str:
    return f"{model_time:g}"


def load_snapshot(age_months: float) -> dict:
    model_time_exact = age_to_model_time(age_months)
    model_time_saved = saved_time_for_age(age_months)
    token = time_token(model_time_saved)
    state_path = STATE_DIR / f"generated_t{token}.npy"
    label_path = LABEL_DIR / f"labels_t{token}.npy"
    state = np.load(state_path)
    labels = np.load(label_path).astype(str)
    if len(state) != len(labels):
        raise ValueError(
            f"State/label length mismatch at t={model_time_saved}: "
            f"{len(state)} versus {len(labels)}"
        )
    result = {
        "age": age_months,
        "model_time_exact": model_time_exact,
        "model_time_saved": model_time_saved,
        "state_path": state_path,
        "label_path": label_path,
        "coords": np.asarray(state[:, :2], dtype=float),
        "labels": labels,
        "slice_origin": "r2_baseline_continuous_t0",
    }
    return result


def style_axis(ax: mpl.axes.Axes) -> None:
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_snapshot(ax: mpl.axes.Axes, snapshot: dict) -> None:
    labels = snapshot["labels"]
    # Counter-clockwise 90-degree rotation applied consistently to every
    # standalone and three-panel spatial snapshot: (x, y) -> (-y, x).
    raw_coords = snapshot["coords"]
    coords = np.column_stack((-raw_coords[:, 1], raw_coords[:, 0]))
    colors = [CELLTYPE_COLORS.get(label, "#888888") for label in labels]
    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        s=1.4,
        c=colors,
        linewidths=0,
        alpha=0.90,
        rasterized=True,
    )
    style_axis(ax)
    age = snapshot["age"]
    saved = snapshot["model_time_saved"]
    ax.set_title(
        f"{age:.1f} months\nmodel t = {saved:.1f} (r2 baseline)",
        fontsize=11,
        pad=7,
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    snapshots = [load_snapshot(age) for age in REQUESTED_AGES]

    # One standalone PNG/PDF per requested age.
    for snapshot in snapshots:
        age_token = str(snapshot["age"]).replace(".", "p")
        fig, ax = plt.subplots(figsize=(4.2, 4.2), dpi=300)
        plot_snapshot(ax, snapshot)
        fig.tight_layout()
        stem = OUTPUT_DIR / f"admouse_spatial_{age_token}_months"
        fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
        fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)

    # Chronological three-panel summary.
    snapshots = sorted(snapshots, key=lambda item: item["age"])
    fig, axes = plt.subplots(1, 3, figsize=(11.1, 3.8), dpi=300)
    for ax, snapshot in zip(axes, snapshots):
        plot_snapshot(ax, snapshot)

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=CELLTYPE_COLORS[celltype],
            markeredgecolor="none",
            markersize=5,
            label=celltype,
        )
        for celltype in CELLTYPE_ORDER
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=4,
        frameon=False,
        fontsize=8,
        columnspacing=1.2,
        handletextpad=0.35,
    )
    fig.suptitle(
        "AdMouse r2 baseline spatial distributions at selected ages",
        fontsize=13,
        y=0.995,
    )
    fig.subplots_adjust(left=0.02, right=0.98, top=0.84, bottom=0.22, wspace=0.07)
    stem = OUTPUT_DIR / "admouse_spatial_selected_ages_3p8_5p7_8p1_months"
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
