#!/usr/bin/env python3
"""Render corrected MOSTA S5 values with the submitted notebook style."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TIMES = tuple(float(value) for value in np.arange(0.0, 3.0001, 0.25))
DISPLAY_TIMES = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0)
ANCHOR_STAGES = {0.0, 1.0, 2.0, 3.0}
STYLE_NOTEBOOK_SHA256 = "3ae4d01d807d00819b2c93f13822f61eda12a9947d732e749bac517509643efc"
STYLE_REFERENCE_SVG_SHA256 = "3def60307d13da9efaf4909f6c0fdd5426ec13340d1b1a8a2efec284dff33c42"
COLORMAP_COLORS = ["#17324d", "#245b78", "#1f8a8a", "#7bc8a4", "#e8f6ef"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest(root: Path) -> None:
    if not (root / "COMPLETE").is_file():
        raise RuntimeError("Numerical root is not complete.")
    for raw in (root / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        expected, relative = raw.split(maxsplit=1)
        path = root / relative.lstrip("*")
        if sha256(path) != expected:
            raise RuntimeError(f"Numerical SHA256 mismatch: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--numerical-root", required=True)
    parser.add_argument("--numerical-audit", required=True)
    parser.add_argument("--style-notebook", required=True)
    parser.add_argument("--style-reference-svg", required=True)
    parser.add_argument("--rejected-mixed-table", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.numerical_root).resolve()
    audit_path = Path(args.numerical_audit).resolve()
    notebook_path = Path(args.style_notebook).resolve()
    style_reference_path = Path(args.style_reference_svg).resolve()
    rejected_mixed_path = Path(args.rejected_mixed_table).resolve()
    output_dir = Path(args.output_dir).resolve()
    verify_manifest(root)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "pass":
        raise RuntimeError("S5 numerical audit has not passed.")
    if sha256(notebook_path) != STYLE_NOTEBOOK_SHA256:
        raise RuntimeError("Submitted S5 style notebook hash mismatch.")
    if sha256(style_reference_path) != STYLE_REFERENCE_SVG_SHA256:
        raise RuntimeError("Historical S5 style reference hash mismatch.")

    growth_path = root / "s5_growth" / "growth_by_cell_fully_generated.csv"
    contract_path = root / "s5_growth" / "growth_contract.json"
    growth = pd.read_csv(growth_path)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if tuple(float(value) for value in contract["time_grid"]) != TIMES:
        raise RuntimeError("S5 numerical contract has the wrong calculation grid.")
    if tuple(float(value) for value in contract["display_times"]) != DISPLAY_TIMES:
        raise RuntimeError("S5 numerical contract has the wrong display grid.")
    if contract["state_source"] != "generated_global_t0" or bool(contract["spatial_warp"]):
        raise RuntimeError("S5 numerical contract has the wrong origin/warp setting.")
    brain = growth.loc[growth["celltype"].astype(str) == "Brain"].copy()
    if tuple(sorted(brain["time"].astype(float).unique())) != TIMES:
        raise RuntimeError("S5 Brain growth does not cover all calculation times.")
    if not np.isfinite(brain[["time", "x", "y", "growth"]].to_numpy(dtype=float)).all():
        raise RuntimeError("S5 Brain plot values contain non-finite values.")

    # Exact submitted source: scale and frame are computed on all 13 Brain
    # slices; t=1.50 is removed from display only after those calculations.
    vmin = float(contract["vmin"])
    vmax = float(contract["vmax"])
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    growth_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "nature_growth",
        COLORMAP_COLORS,
    )
    x_min, x_max = float(brain["x"].min()), float(brain["x"].max())
    y_min, y_max = float(brain["y"].min()), float(brain["y"].max())
    x_pad = 0.04 * (x_max - x_min)
    y_pad = 0.04 * (y_max - y_min)

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
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    n_rows = 3
    n_cols = 4
    figure = plt.figure(
        figsize=(3.02 * n_cols + 0.55, 2.38 * n_rows + 0.45),
        facecolor="white",
    )
    grid = figure.add_gridspec(
        n_rows,
        n_cols + 1,
        width_ratios=([1.0] * n_cols) + [0.07],
        wspace=0.10,
        hspace=0.16,
    )
    axes = [figure.add_subplot(grid[row, column]) for row in range(n_rows) for column in range(n_cols)]
    colorbar_axis = figure.add_subplot(grid[:, -1])
    plotted_rows: list[pd.DataFrame] = []
    panel_audit_rows: list[dict[str, object]] = []
    for axis, time_value in zip(axes, DISPLAY_TIMES):
        subset = brain.loc[np.isclose(brain["time"].astype(float), time_value)].copy()
        if subset.empty:
            raise RuntimeError(f"S5 display time {time_value:g} has no Brain cells.")
        clipped = np.clip(subset["growth"].to_numpy(dtype=float), vmin, vmax)
        axis.scatter(
            subset["x"],
            subset["y"],
            c=clipped,
            cmap=growth_cmap,
            norm=norm,
            s=2.2,
            alpha=0.92,
            linewidths=0,
            rasterized=True,
        )
        axis.set_xlim(x_min - x_pad, x_max + x_pad)
        axis.set_ylim(y_min - y_pad, y_max + y_pad)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_aspect("equal")
        axis.set_facecolor("#f7f9fb")
        anchor_stage = float(time_value) in ANCHOR_STAGES
        axis.set_title(
            f"t={time_value:.2f}",
            loc="left",
            fontsize=9.4,
            color="#243746" if anchor_stage else "#4f6272",
            fontweight="bold" if anchor_stage else "normal",
            pad=3,
        )
        for spine in axis.spines.values():
            spine.set_linewidth(1.0 if anchor_stage else 0.7)
            spine.set_color("#8aa2b2" if anchor_stage else "#dbe4ea")
        display_values = subset[["time", "time_key", "cell_index", "x", "y", "growth", "celltype"]].copy()
        display_values["growth_clipped_for_colormap"] = clipped
        plotted_rows.append(display_values)
        panel_audit_rows.append(
            {
                "display_order": len(panel_audit_rows) + 1,
                "time": float(time_value),
                "n_brain_cells": int(len(subset)),
                "anchor_stage_title_style": anchor_stage,
                "growth_mean": float(subset["growth"].mean()),
                "growth_median": float(subset["growth"].median()),
                "fraction_below_vmin": float(np.mean(subset["growth"] < vmin)),
                "fraction_above_vmax": float(np.mean(subset["growth"] > vmax)),
                "xlim_min": x_min - x_pad,
                "xlim_max": x_max + x_pad,
                "ylim_min": y_min - y_pad,
                "ylim_max": y_max + y_pad,
            }
        )

    scalar = mpl.cm.ScalarMappable(norm=norm, cmap=growth_cmap)
    scalar.set_array([])
    colorbar = figure.colorbar(scalar, cax=colorbar_axis)
    colorbar.set_label("Predicted growth rate", fontsize=9, color="#33424d")
    colorbar.ax.tick_params(labelsize=8, width=0.7, length=3, colors="#51606b")
    colorbar.outline.set_linewidth(0.7)
    colorbar.outline.set_edgecolor("#c9d4db")
    figure.suptitle(
        "Brain growth-rate maps across observed and interpolated time points",
        x=0.02,
        y=0.985,
        ha="left",
        fontsize=13,
        color="#1f2d38",
    )
    figure.text(
        0.02,
        0.952,
        "Panels share a fixed latent spatial frame and a common growth-rate scale for direct comparison.",
        ha="left",
        va="top",
        fontsize=8.5,
        color="#647480",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = "Figure_S5_MOSTA_latest_package_global_t0_growth_exact_submitted_style"
    pdf = output_dir / f"{stem}.pdf"
    svg = output_dir / f"{stem}.svg"
    png = output_dir / f"{stem}.png"
    plotted_path = output_dir / f"{stem}_plotted_brain_growth.csv.gz"
    panel_audit_path = output_dir / f"{stem}_panel_values.csv"
    figure.savefig(pdf, bbox_inches="tight", facecolor="white", dpi=300)
    figure.savefig(svg, bbox_inches="tight", facecolor="white", dpi=300)
    figure.savefig(png, bbox_inches="tight", facecolor="white", dpi=300)
    plt.close(figure)
    pd.concat(plotted_rows, ignore_index=True).to_csv(
        plotted_path,
        index=False,
        compression={"method": "gzip", "compresslevel": 9, "mtime": 0},
    )
    pd.DataFrame(panel_audit_rows).to_csv(panel_audit_path, index=False)

    manifest = {
        "schema_version": 1,
        "status": "candidate_pending_pdf_visual_qa",
        "dataset": "MOSTA",
        "panel": "Supplementary Figure S5",
        "numerical_truth": {
            "root": str(root),
            "manifest_sha256": sha256(root / "SHA256SUMS.txt"),
            "growth_table": {"path": str(growth_path), "sha256": sha256(growth_path)},
            "growth_contract": {"path": str(contract_path), "sha256": sha256(contract_path)},
            "numerical_audit": {"path": str(audit_path), "sha256": sha256(audit_path)},
            "state_source": "fully_generated_global_t0_all_13_times",
            "calculation_times": list(TIMES),
            "display_times": list(DISPLAY_TIMES),
            "display_only_omission": 1.5,
            "growth_api": "CytoBridge.tl.evaluate_growth_by_timepoint",
            "common_scale": {"percentiles": [5.0, 95.0], "vmin": vmin, "vmax": vmax},
            "spatial_warp": False,
            "growth_smoothing": False,
            "rejected_mixed_table_not_used": {
                "path": str(rejected_mixed_path),
                "sha256": sha256(rejected_mixed_path),
                "reason": "integer anchors observed_real while intermediate times generated_global_t0",
            },
        },
        "style_truth": {
            "notebook": {"path": str(notebook_path), "sha256": STYLE_NOTEBOOK_SHA256, "cell": 10},
            "historical_reference_svg": {"path": str(style_reference_path), "sha256": STYLE_REFERENCE_SVG_SHA256},
            "layout": "3x4 plus one shared vertical colorbar",
            "figure_inches_before_tight_bbox": [12.63, 7.59],
            "fixed_frame": [x_min - x_pad, x_max + x_pad, y_min - y_pad, y_max + y_pad],
            "colormap_colors": COLORMAP_COLORS,
            "point_size": 2.2,
            "point_alpha": 0.92,
            "point_linewidth": 0,
            "dense_point_layer_rasterized_as_submitted": True,
            "text_axes_colorbar_remain_vector": True,
            "font": "DejaVu Sans",
        },
        "outputs": {
            "pdf": {"path": str(pdf), "sha256": sha256(pdf)},
            "svg": {"path": str(svg), "sha256": sha256(svg)},
            "png": {"path": str(png), "sha256": sha256(png)},
            "plotted_values": {"path": str(plotted_path), "sha256": sha256(plotted_path)},
            "panel_values": {"path": str(panel_audit_path), "sha256": sha256(panel_audit_path)},
        },
        "forbidden_transforms": {"rotation": False, "stretch": False, "warp": False},
        "arista_assets_used": False,
    }
    manifest_path = output_dir / f"{stem}_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
