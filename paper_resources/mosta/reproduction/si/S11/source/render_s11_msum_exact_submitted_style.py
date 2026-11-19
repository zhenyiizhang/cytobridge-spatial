#!/usr/bin/env python3
"""Render corrected package-native M_sum MOSTA S11 in submitted visual grammar."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator


STYLE_NOTEBOOK_SHA256 = "255f96c8c572898f460cca33a1f7b6ea7bf385a5d2ae77b440d285f871c0e4e0"
STYLE_ORACLE_SVG_SHA256 = "ab461c87c6b15353e7c71a1c582bb8a489b96b339d4cf21bf8494071e31ff010"
PALETTE = {1: "#D97757", 2: "#2A7F9E", 3: "#6A994E"}
EXPECTED_TIMES = np.asarray([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
EXPECTED_QUOTAS = {1: 12, 2: 11, 3: 8}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-component-root", required=True)
    parser.add_argument("--seed-stability-root", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--selection-manifest", required=True)
    parser.add_argument("--style-notebook", required=True)
    parser.add_argument("--style-oracle-svg", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(root: Path) -> dict[str, Any]:
    if not (root / "COMPLETE").is_file() or not (root / "SHA256SUMS.txt").is_file():
        raise RuntimeError(f"Input is not sealed: {root}")
    checked = 0
    for raw in (root / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        expected, relative = raw.split(maxsplit=1)
        path = root / relative.lstrip("*")
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"Checksum mismatch: {path}")
        checked += 1
    return {"root": str(root), "manifest_sha256": sha256(root / "SHA256SUMS.txt"), "files_verified": checked}


def normalized_by_pair(path: Path, assignments: pd.DataFrame) -> pd.DataFrame:
    table = pd.read_csv(path, index_col=0)
    table.columns = [float(value) for value in table.columns]
    table.index = table.index.astype(str)
    mapping = assignments.set_index("profile")["pair_id"].astype(str)
    if set(table.index) == set(mapping.index):
        table.index = [mapping.loc[value] for value in table.index]
    elif set(table.index) != set(assignments["pair_id"].astype(str)):
        raise RuntimeError("Normalized profile identity mismatch")
    return table.loc[:, EXPECTED_TIMES.tolist()]


def display_curve(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x_dense = np.linspace(float(x.min()), float(x.max()), 300)
    return x_dense, PchipInterpolator(x, y)(x_dense)


def main() -> None:
    args = parse_args()
    baseline_root = Path(args.baseline_component_root).resolve()
    stability_root = Path(args.seed_stability_root).resolve()
    selection_path = Path(args.selection).resolve()
    selection_manifest_path = Path(args.selection_manifest).resolve()
    notebook_path = Path(args.style_notebook).resolve()
    oracle_path = Path(args.style_oracle_svg).resolve()
    output_dir = Path(args.output_dir).resolve()
    baseline_contract = verify(baseline_root)
    stability_contract = verify(stability_root)
    if sha256(notebook_path) != STYLE_NOTEBOOK_SHA256:
        raise RuntimeError("Historical S11 style notebook hash mismatch")
    if sha256(oracle_path) != STYLE_ORACLE_SVG_SHA256:
        raise RuntimeError("Historical S11 SVG style oracle hash mismatch")
    selection_manifest = json.loads(selection_manifest_path.read_text(encoding="utf-8"))
    if selection_manifest["output"]["sha256"] != sha256(selection_path):
        raise RuntimeError("Selection manifest hash mismatch")

    base = baseline_root / "tables" / "M_sum"
    pair_path = base / "lr_pair_timecourse.csv"
    assignments_path = base / "lr_pattern_assignments.csv"
    normalized_path = base / "lr_normalized_profiles.csv"
    prototypes_path = base / "lr_pattern_prototypes.csv"
    pair = pd.read_csv(pair_path)
    assignments = pd.read_csv(assignments_path)
    normalized = normalized_by_pair(normalized_path, assignments)
    selection = pd.read_csv(selection_path).sort_values("display_order")
    if len(selection) != 31 or selection["display_order"].tolist() != list(range(1, 32)):
        raise RuntimeError("Submitted geometry requires exactly 31 ordered profiles")
    quotas = selection["cluster"].astype(int).value_counts().sort_index().to_dict()
    if quotas != EXPECTED_QUOTAS:
        raise RuntimeError(f"Corrected submitted-geometry quotas failed: {quotas}")
    if not (
        (selection["cluster_seed42"] == selection["cluster_seed43"])
        & (selection["cluster_seed42"] == selection["cluster_seed44"])
    ).all():
        raise RuntimeError("A displayed profile is not cluster-stable across seeds")
    if set(selection["pair_id"].astype(str)) - set(pair["pair_id"].astype(str)):
        raise RuntimeError("A displayed profile is absent from corrected M_sum numerical truth")

    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    ncols = 4
    nrows = math.ceil(len(selection) / ncols)
    figure, axes_grid = plt.subplots(
        nrows, ncols, figsize=(13.6, 2.9 * nrows), dpi=240,
        squeeze=False, facecolor="white",
    )
    axes = axes_grid.ravel()
    plotted_rows = []
    maximum_normalized_error = 0.0
    for axis, selected in zip(axes, selection.itertuples(index=False)):
        subset = pair.loc[pair["pair_id"].astype(str) == str(selected.pair_id)].sort_values("time")
        x = subset["time"].to_numpy(dtype=float)
        y = subset["score"].to_numpy(dtype=float)
        if x.shape != EXPECTED_TIMES.shape or not np.array_equal(x, EXPECTED_TIMES):
            raise RuntimeError(f"Profile lacks seven exact half-step values: {selected.pair_id}")
        denominator = max(float(y.max() - y.min()), 1e-12)
        y_plot = (y - y.min()) / denominator
        package_values = normalized.loc[str(selected.pair_id)].to_numpy(dtype=float)
        error = float(np.max(np.abs(y_plot - package_values), initial=0.0))
        maximum_normalized_error = max(maximum_normalized_error, error)
        if error > 1e-12:
            raise RuntimeError(f"Renderer normalization diverges from package output: {error}")
        x_dense, y_dense = display_curve(x, y_plot)
        cluster = int(selected.cluster)
        color = PALETTE[cluster]

        # Exact submitted visual grammar. PCHIP is display-only; every computed
        # half-step remains visible as a raw marker and is persisted below.
        axis.plot(x_dense, y_dense, color=color, linewidth=2.2)
        axis.scatter(x, y_plot, s=18, color=color, edgecolor="white", linewidth=0.6, zorder=3)
        axis.set_title(f"Pattern {cluster}  {selected.pair}", loc="left", fontsize=9, pad=4, color=color)
        axis.set_xlim(float(x.min()), float(x.max()))
        axis.set_ylim(-0.03, 1.03)
        axis.grid(axis="y", color="#D8DEE3", linewidth=0.6)
        axis.tick_params(labelsize=8)
        axis.set_xlabel("time", fontsize=8)
        axis.set_ylabel("norm score", fontsize=8)
        axis.set_facecolor("#FBFAF8")
        for side in ("left", "bottom"):
            axis.spines[side].set_color(color)
            axis.spines[side].set_linewidth(1.2)
        for time_value, raw_score, normalized_score in zip(x, y, y_plot):
            plotted_rows.append(
                {
                    "display_order": int(selected.display_order),
                    "cluster": cluster,
                    "pair_id": str(selected.pair_id),
                    "pair": str(selected.pair),
                    "time": float(time_value),
                    "raw_score": float(raw_score),
                    "normalized_score": float(normalized_score),
                }
            )
    for axis in axes[len(selection):]:
        axis.axis("off")
    figure.suptitle("Shape-clustered LR curve panel", x=0.01, ha="left", y=1.01, fontsize=13)
    figure.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = "Figure_S11_MOSTA_corrected_package_Msum_average_k3_exact_submitted_style_representative31"
    svg = output_dir / f"{stem}.svg"
    pdf = output_dir / f"{stem}.pdf"
    png = output_dir / f"{stem}.png"
    plotted = output_dir / f"{stem}_plotted_values.csv"
    figure.savefig(svg, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf, bbox_inches="tight", facecolor="white")
    figure.savefig(png, bbox_inches="tight", facecolor="white", dpi=300)
    plt.close(figure)
    pd.DataFrame(plotted_rows).to_csv(plotted, index=False)

    stability_summary = json.loads((stability_root / "summary.json").read_text(encoding="utf-8"))
    manifest = {
        "schema_version": 1,
        "status": "candidate_pending_pdf_visual_qa",
        "dataset": "MOSTA",
        "panel": "Supplementary Figure S11",
        "numerical_truth": {
            "baseline_seed42_component": baseline_contract,
            "matrix_key": "M_sum",
            "matrix_interpretation": "package-native total attention by sender/receiver cell-type pair; matches submitted panel estimand without legacy per-time max normalization",
            "pair_timecourse": {"path": str(pair_path), "sha256": sha256(pair_path)},
            "assignments": {"path": str(assignments_path), "sha256": sha256(assignments_path)},
            "normalized_profiles": {"path": str(normalized_path), "sha256": sha256(normalized_path)},
            "prototypes": {"path": str(prototypes_path), "sha256": sha256(prototypes_path)},
            "state_source": "fully_generated_global_t0_50k_start_unwarped",
            "expression_source": "inverse_pca_all_seven_times",
            "communication_cells_per_time": 12000,
            "clustering": "package minmax + average linkage exact k=3 + peak-time order",
            "legacy_per_time_max_normalization": False,
            "maximum_renderer_vs_package_normalized_error": maximum_normalized_error,
        },
        "sampling_stability": {
            "contract": stability_contract,
            "comparisons": stability_summary["comparisons"],
            "selected_profiles_same_cluster_seeds_42_43_44": True,
        },
        "content_selection": {
            "path": str(selection_path), "sha256": sha256(selection_path),
            "manifest": str(selection_manifest_path), "manifest_sha256": sha256(selection_manifest_path),
            "cluster_counts": {str(key): int(value) for key, value in quotas.items()},
        },
        "style_truth": {
            "notebook": str(notebook_path), "notebook_sha256": STYLE_NOTEBOOK_SHA256,
            "oracle_svg": str(oracle_path), "oracle_svg_sha256": STYLE_ORACLE_SVG_SHA256,
            "columns": 4, "rows": 8, "profiles": 31, "raw_markers_per_profile": 7,
            "curve": "PCHIP display only; raw half-step markers visible",
            "palette": {str(key): value for key, value in PALETTE.items()},
            "figure_inches_before_tight_bbox": [13.6, 23.2],
        },
        "outputs": {
            "svg": {"path": str(svg), "sha256": sha256(svg)},
            "pdf": {"path": str(pdf), "sha256": sha256(pdf)},
            "png": {"path": str(png), "sha256": sha256(png)},
            "plotted_values": {"path": str(plotted), "sha256": sha256(plotted)},
        },
        "forbidden_transforms": {"rotation": False, "stretch": False, "warp": False},
        "arista_assets_used": False,
    }
    manifest_path = output_dir / f"{stem}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
