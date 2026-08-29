#!/usr/bin/env python3
"""Render corrected MOSTA Supplementary Figure S6 in submitted visual grammar.

The numerical table is the accepted latest-package, fully generated global-t0
50k rollout.  Plot construction is transcribed from cells 6 and 8 of the
submitted MOSTA composition notebook; the two standalone notebook panels are
stacked as in SI page 32.  No smoothing, category cherry-picking, rotation,
stretching, coordinate warp, or package-default restyling is applied.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[3]
BUNDLE = Path(__file__).resolve().parents[1]
SHARED = (
    REPO
    / "output/mosta_si_shared_compute_20260825_v1/server_download"
    / "si-s4-s10-global-t0-n50000-dense13-2b3c79e-seed42-20260825-v1"
)
INPUT_TABLE = SHARED / "s6_composition/celltype_composition_fully_generated.csv"
PALETTE_PATH = BUNDLE / "input/label_to_color.json"
NOTEBOOK_PATH = BUNDLE / "source/mosta-brain-cell-composition-review.ipynb"
AUDIT_PATH = BUNDLE / "tables/s6_numerical_audit.json"
PANEL_A_STYLE_REFERENCE = BUNDLE / "source/submitted_s6_panel_a_style_reference.svg"
PANEL_B_STYLE_REFERENCE = BUNDLE / "source/submitted_s6_panel_b_style_reference.svg"
FIGURE_STEM = "Figure_S6_MOSTA_latest_package_global_t0_50k_exact_submitted_style"
SUBMITTED_DISPLAY_CELLTYPES = (
    "Brain",
    "Connective tissue",
    "Cavity",
    "Epidermis",
    "Muscle",
    "Jaw and tooth",
    "Meninges",
    "Liver",
    "Cartilage primordium",
    "Spinal cord",
    "Heart",
    "GI tract",
    "Dorsal root ganglion",
    "Cartilage",
    "Adipose tissue",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def configure_submitted_style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "font.size": 10,
            "font.family": "DejaVu Sans",
            "svg.fonttype": "none",
            "svg.hashsalt": "mosta-s6-20260825-v1",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def build_tables() -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    composition = pd.read_csv(INPUT_TABLE)
    count_table = composition.pivot(
        index="time", columns="celltype", values="count"
    ).fillna(0).astype(int)
    fraction_table = composition.pivot(
        index="time", columns="celltype", values="fraction"
    ).fillna(0.0)

    # The old notebook selected 15 labels from the old numerical table.  The
    # resulting legend/stack order is frozen in the submitted output and is a
    # style/content-selection authority here: keep it fixed and replace values
    # only.  Re-ranking on corrected values would hide Cartilage in Other and
    # would no longer be an equivalent replacement of the submitted panel.
    selected = list(SUBMITTED_DISPLAY_CELLTYPES)
    missing = [label for label in selected if label not in fraction_table]
    if missing:
        raise RuntimeError(f"submitted S6 display labels missing: {missing}")
    other_labels = [
        str(label) for label in fraction_table.columns if str(label) not in selected
    ]

    fraction_plot = fraction_table[selected].copy()
    fraction_plot["Other"] = fraction_table[other_labels].sum(axis=1)
    count_plot = count_table[selected].copy()
    count_plot["Other"] = count_table[other_labels].sum(axis=1)
    fraction_plot = fraction_plot.sort_index()
    count_plot = count_plot.sort_index()

    if not np.allclose(fraction_plot.sum(axis=1), 1.0, atol=1e-12, rtol=0):
        raise RuntimeError("top15+Other fractions do not sum to one")
    if not np.array_equal(
        count_plot.sum(axis=1).to_numpy(dtype=int),
        composition.groupby("time", sort=True)["count"].sum().to_numpy(dtype=int),
    ):
        raise RuntimeError("top15+Other counts do not preserve totals")
    return count_plot, fraction_plot * 100.0, selected, other_labels


def colors_for(columns: list[str], palette: dict[str, str]) -> list[str]:
    return [
        "#c9c3b8" if celltype == "Other" else palette.get(celltype, "#808080")
        for celltype in columns
    ]


def plot_count_panel(
    ax: mpl.axes.Axes,
    count_plot: pd.DataFrame,
    colors: list[str],
) -> None:
    x = count_plot.index.to_numpy(dtype=float)
    y = [count_plot[column].to_numpy(dtype=float) for column in count_plot.columns]
    ax.stackplot(
        x,
        y,
        colors=colors,
        alpha=0.95,
        linewidth=0.5,
        edgecolor="white",
    )
    ax.set_xlim(float(x.min()), float(x.max()))
    ax.set_xlabel("Time")
    ax.set_ylabel("Number of cells")
    ax.set_title(
        "Brain cell counts across observed and interpolated time points",
        loc="left",
        fontsize=12.5,
    )
    ax.grid(axis="y", color="#e9e3d8", linewidth=0.8)
    ax.set_axisbelow(True)
    handles = [
        mpl.patches.Patch(facecolor=color, edgecolor="none", label=celltype)
        for celltype, color in zip(count_plot.columns, colors)
    ]
    ax.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
        title="Cell type",
        ncol=1,
        fontsize=8,
        title_fontsize=8.5,
    )


def plot_fraction_panel(
    ax: mpl.axes.Axes,
    fraction_pct: pd.DataFrame,
    colors: list[str],
) -> None:
    bottom = np.zeros(fraction_pct.shape[0], dtype=float)
    x = np.arange(fraction_pct.shape[0], dtype=float)
    for celltype, color in zip(fraction_pct.columns, colors):
        values = fraction_pct[celltype].to_numpy(dtype=float)
        ax.bar(
            x,
            values,
            bottom=bottom,
            width=0.76,
            color=color,
            edgecolor="white",
            linewidth=0.6,
            label=celltype,
        )
        bottom += values
    ax.set_xlim(-0.55, len(x) - 0.45)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Cell proportion (%)")
    ax.set_xlabel("Time")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{time:.2f}" for time in fraction_pct.index], rotation=0)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.grid(axis="y", color="#e9e3d8", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(
        "Brain cell composition across observed and interpolated time points",
        loc="left",
        fontsize=12.5,
    )
    legend = ax.legend(
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
        title="Cell type",
        ncol=1,
        fontsize=8,
        title_fontsize=8.5,
    )
    legend_handles = getattr(legend, "legendHandles", None)
    if legend_handles is None:
        legend_handles = getattr(legend, "legend_handles", [])
    for handle in legend_handles:
        if hasattr(handle, "set_linewidth"):
            handle.set_linewidth(0)


def save_figure(fig: mpl.figure.Figure, path: Path) -> None:
    metadata = {
        "Title": "Supplementary Figure S6 — MOSTA counts and composition",
        "Creator": "CytoBridge S6 exact submitted-style renderer",
        "CreationDate": None,
        "ModDate": None,
    }
    suffix = path.suffix.lower()
    kwargs: dict[str, object] = {
        "bbox_inches": "tight",
        "facecolor": "white",
    }
    if suffix == ".pdf":
        kwargs["metadata"] = metadata
    elif suffix == ".png":
        kwargs["dpi"] = 300
    fig.savefig(path, **kwargs)


def main() -> None:
    audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    if audit.get("status") != "PASS":
        raise RuntimeError("S6 numerical audit has not passed")
    configure_submitted_style()
    palette = json.loads(PALETTE_PATH.read_text(encoding="utf-8"))
    count_plot, fraction_pct, selected, other_labels = build_tables()
    colors = colors_for(list(count_plot.columns), palette)

    figures = BUNDLE / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    # Submitted notebook panels are 11x4.2 and 11x4.6 inches.  The combined
    # SI assembly keeps that relative geometry and the original right legends.
    fig = plt.figure(figsize=(11.0, 8.8), facecolor="white")
    grid = fig.add_gridspec(2, 1, height_ratios=(4.2, 4.6), hspace=0.29)
    ax_count = fig.add_subplot(grid[0, 0])
    ax_fraction = fig.add_subplot(grid[1, 0])
    plot_count_panel(ax_count, count_plot, colors)
    plot_fraction_panel(ax_fraction, fraction_pct, colors)
    ax_count.text(
        -0.085,
        1.045,
        "a",
        transform=ax_count.transAxes,
        fontsize=14.5,
        fontweight="bold",
        va="bottom",
        ha="left",
    )
    ax_fraction.text(
        -0.085,
        1.045,
        "b",
        transform=ax_fraction.transAxes,
        fontsize=14.5,
        fontweight="bold",
        va="bottom",
        ha="left",
    )
    fig.subplots_adjust(left=0.09, right=0.78, top=0.965, bottom=0.075)

    outputs: dict[str, dict[str, object]] = {}
    for extension in ("pdf", "svg", "png"):
        path = figures / f"{FIGURE_STEM}.{extension}"
        save_figure(fig, path)
        outputs[extension] = {
            "path": str(path.resolve()),
            "sha256": sha256(path),
            "size_bytes": int(path.stat().st_size),
        }
    plt.close(fig)

    count_path = figures / f"{FIGURE_STEM}_panel_a_counts.csv"
    fraction_path = figures / f"{FIGURE_STEM}_panel_b_percent.csv"
    order_path = figures / f"{FIGURE_STEM}_stack_order.csv"
    count_plot.rename_axis(index="time").reset_index().to_csv(count_path, index=False)
    fraction_pct.rename_axis(index="time").reset_index().to_csv(fraction_path, index=False)
    pd.DataFrame(
        {
            "stack_index": np.arange(len(count_plot.columns), dtype=int),
            "celltype": list(count_plot.columns),
            "color": colors,
            "is_other": [label == "Other" for label in count_plot.columns],
        }
    ).to_csv(order_path, index=False)

    manifest = {
        "schema_version": 1,
        "panel": "Supplementary Figure S6a-b",
        "dataset": "MOSTA",
        "status": "candidate_pending_pdf_visual_qa",
        "arista_assets_used": False,
        "forbidden_transforms": {"rotation": False, "stretch": False, "warp": False},
        "numerical_truth": {
            "table": {"path": str(INPUT_TABLE.resolve()), "sha256": sha256(INPUT_TABLE)},
            "audit": {"path": str(AUDIT_PATH.resolve()), "sha256": sha256(AUDIT_PATH)},
            "package_commit": audit["release"]["package_commit"],
            "package_archive_sha256": audit["release"]["package_archive_sha256"],
            "aligned_h5ad_sha256": audit["release"]["aligned_h5ad_sha256"],
            "finetune_sha256": audit["release"]["finetune_sha256"],
            "score_sha256": audit["release"]["score_sha256"],
            "classifier_sha256": audit["release"]["classifier_sha256"],
            "classifier_k": audit["release"]["classifier_k"],
            "state": "fully generated global-t0 50k split-SDE at all 13 times",
            "times": list(map(float, count_plot.index)),
            "totals": list(map(int, count_plot.sum(axis=1))),
            "rejected_old_mixed_table_used": False,
        },
        "style_truth": {
            "notebook": {
                "path": str(NOTEBOOK_PATH.resolve()),
                "sha256": sha256(NOTEBOOK_PATH),
                "cells": [6, 8],
            },
            "saved_panel_style_references": {
                "panel_a": {
                    "path": str(PANEL_A_STYLE_REFERENCE.resolve()),
                    "sha256": sha256(PANEL_A_STYLE_REFERENCE),
                },
                "panel_b": {
                    "path": str(PANEL_B_STYLE_REFERENCE.resolve()),
                    "sha256": sha256(PANEL_B_STYLE_REFERENCE),
                },
            },
            "submitted_si": {
                "path": "/Users/zhenyizhang/Desktop/202511/nbme预投稿/投稿/投稿修改/si.pdf",
                "page": 32,
            },
            "palette": {"path": str(PALETTE_PATH.resolve()), "sha256": sha256(PALETTE_PATH)},
            "font": "DejaVu Sans",
            "top_n_celltypes": 15,
            "selection_rule": "freeze the 15 labels and order in the submitted S6 notebook output/SI; replace corrected values only",
            "submitted_display_celltypes": selected,
            "collapsed_to_other": other_labels,
            "stack_order": list(count_plot.columns),
            "colors": dict(zip(count_plot.columns, colors)),
            "panel_a": {
                "geometry": "stacked area",
                "alpha": 0.95,
                "linewidth": 0.5,
                "edgecolor": "white",
                "x_axis": "continuous model time 0-3",
            },
            "panel_b": {
                "geometry": "stacked bars",
                "bar_width": 0.76,
                "linewidth": 0.6,
                "edgecolor": "white",
                "y_limits": [0, 100],
            },
            "legend": "right side, one column, frame off, title Cell type",
            "grid": {"axis": "y", "color": "#e9e3d8", "linewidth": 0.8},
        },
        "outputs": outputs
        | {
            "panel_a_counts": {"path": str(count_path.resolve()), "sha256": sha256(count_path)},
            "panel_b_percent": {"path": str(fraction_path.resolve()), "sha256": sha256(fraction_path)},
            "stack_order": {"path": str(order_path.resolve()), "sha256": sha256(order_path)},
        },
    }
    manifest_path = figures / f"{FIGURE_STEM}_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
