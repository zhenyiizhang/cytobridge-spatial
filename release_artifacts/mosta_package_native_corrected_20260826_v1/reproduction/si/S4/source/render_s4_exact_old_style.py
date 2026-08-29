#!/usr/bin/env python3
"""Render corrected MOSTA S4 using the submitted interpolation-mosaic grammar."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import anndata as ad
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


PANELS = (
    ("0.0 (observed)", "s4/observed_t0.h5ad"),
    ("0.0 (generated)", "generated_states/time_0.h5ad"),
    ("0.5", "generated_states/time_0p5.h5ad"),
    ("1.0 (generated)", "generated_states/time_1.h5ad"),
    ("1.5", "generated_states/time_1p5.h5ad"),
    ("2.0 (generated)", "generated_states/time_2.h5ad"),
    ("2.5", "generated_states/time_2p5.h5ad"),
    ("3.0 (generated)", "generated_states/time_3.h5ad"),
)
EXPECTED_PALETTE_SHA256 = "7e95e868e0a6ecd4a2ed13b57e6a8223e77e2302a0f9634ca30f41390c040b71"
STYLE_NOTEBOOK_SHA256 = "3f941d2b3d589161e0a01db7f2cee419685c7ed6dcd6de6160f04d3f5719968a"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shared-root", required=True)
    parser.add_argument("--palette", required=True)
    parser.add_argument("--style-notebook", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--vector-points", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.shared_root).resolve()
    palette_path = Path(args.palette).resolve()
    notebook_path = Path(args.style_notebook).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if sha256(palette_path) != EXPECTED_PALETTE_SHA256:
        raise RuntimeError("MOSTA submitted palette hash mismatch")
    if sha256(notebook_path) != STYLE_NOTEBOOK_SHA256:
        raise RuntimeError("MOSTA interpolation notebook hash mismatch")
    palette = json.loads(palette_path.read_text(encoding="utf-8"))

    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "DejaVu Sans",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    panel_data = []
    for title, relative in PANELS:
        path = root / relative
        data = ad.read_h5ad(path, backed="r")
        coords = np.asarray(data.obsm["spatial"], dtype=np.float32)
        labels = data.obs["Annotation"].astype(str).to_numpy()
        unknown = sorted(set(labels).difference(palette))
        if unknown:
            raise RuntimeError(f"Palette missing labels in {title}: {unknown}")
        panel_data.append((title, path, coords, labels))
        data.file.close()

    cols = 4
    rows = math.ceil(len(panel_data) / cols)
    figure, axes = plt.subplots(
        rows,
        cols,
        figsize=(cols * 2.2, rows * 2.2),
        dpi=300,
        squeeze=False,
    )
    figure.patch.set_facecolor("white")
    for axis, (title, _, coords, labels) in zip(axes.flat, panel_data):
        colors = [palette[str(label)] for label in labels]
        axis.set_facecolor("white")
        axis.scatter(
            coords[:, 0],
            coords[:, 1],
            s=2.5,
            c=colors,
            linewidths=0,
            alpha=0.9,
            rasterized=not bool(args.vector_points),
        )
        axis.set_aspect("equal")
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)
        axis.set_title(title, color="#1a1a1a", fontsize=8, pad=3)
    figure.tight_layout()

    suffix = "vector_points" if args.vector_points else "exact_old_rasterized_points"
    stem = f"Figure_S4_MOSTA_latest_package_global_t0_50k_{suffix}"
    pdf = output / f"{stem}.pdf"
    svg = output / f"{stem}.svg"
    png = output / f"{stem}.png"
    figure.savefig(pdf, format="pdf", facecolor="white", bbox_inches="tight")
    figure.savefig(svg, format="svg", facecolor="white", bbox_inches="tight")
    figure.savefig(png, format="png", facecolor="white", bbox_inches="tight", dpi=300)
    plt.close(figure)

    manifest = {
        "schema_version": 1,
        "status": "candidate_pending_visual_QA",
        "dataset": "MOSTA",
        "panel": "Supplementary Figure S4",
        "numerical_truth": {
            "shared_root": str(root),
            "shared_complete": (root / "COMPLETE").is_file(),
            "panels": [
                {
                    "title": title,
                    "path": str(path),
                    "sha256": sha256(path),
                    "n_cells": int(coords.shape[0]),
                }
                for title, path, coords, _ in panel_data
            ],
            "global_t0": True,
            "classifier_k": 10,
            "spatial_warp": False,
        },
        "style_truth": {
            "notebook": str(notebook_path),
            "notebook_sha256": STYLE_NOTEBOOK_SHA256,
            "palette": str(palette_path),
            "palette_sha256": EXPECTED_PALETTE_SHA256,
            "layout": "2x4",
            "mosaic_cell_size_inches": 2.2,
            "point_size": 2.5,
            "alpha": 0.9,
            "title_font_size": 8,
            "title_pad": 3,
            "axes": "off",
            "aspect": "equal",
            "independent_native_coordinate_limits": True,
            "points_rasterized_as_historical_renderer": not bool(args.vector_points),
        },
        "outputs": {
            "pdf": {"path": str(pdf), "sha256": sha256(pdf)},
            "svg": {"path": str(svg), "sha256": sha256(svg)},
            "png": {"path": str(png), "sha256": sha256(png)},
        },
        "forbidden_transforms": {"rotation": False, "stretch": False, "warp": False},
        "arista_assets_used": False,
    }
    manifest_path = output / f"{stem}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
