#!/usr/bin/env python3
"""Export package-native ARISTA Figure 5c state for the legacy renderer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np

import build_main5abce_s13_paper_contract as legacy


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANUSCRIPT_COORDINATES = (
    PROJECT_ROOT / "repositories/cb_reproducibility/data/arista/arista_1108_with_annotation.csv"
)
DEFAULT_MANUSCRIPT_ROI = (
    PROJECT_ROOT
    / "repositories/cb_reproducibility/results/"
    "arista_velocity_spatial_direction_correlation_roi_t1_scvelo_only_notebook/"
    "velocity_spatial_direction_correlation_roi_t1_scvelo_only.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--velocity-npz", required=True, type=Path)
    parser.add_argument("--time1-slice", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--manuscript-coordinates", type=Path, default=DEFAULT_MANUSCRIPT_COORDINATES)
    parser.add_argument("--manuscript-roi", type=Path, default=DEFAULT_MANUSCRIPT_ROI)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite immutable output: {output_dir}")
    output_dir.mkdir(parents=True)
    state = ad.read_h5ad(args.time1_slice.expanduser().resolve())
    record = {
        "coords": np.asarray(state.obsm["spatial"], dtype=np.float32),
        "labels": state.obs["Annotation"].astype(str).to_numpy(),
    }
    _, table, roi_bounds, focus_bounds, alignment = legacy.prepare_figure5c(
        args.velocity_npz.expanduser().resolve(),
        record,
        args.manuscript_coordinates.expanduser().resolve(),
        args.manuscript_roi.expanduser().resolve(),
        output_dir,
    )
    manifest = {
        "workflow": "ARISTA package-native Figure 5c numerical state",
        "n_cells": int(len(table)),
        "roi_n": int(table["in_roi"].sum()),
        "roi_bounds": list(roi_bounds),
        "nested_display_bounds": list(focus_bounds),
        "display_similarity_alignment": alignment,
        "numerical_contract": {
            "left": "full package velocity, direct spatial component, 30-NN scVelo projection",
            "right": "full and interaction first-two-dimensional spatial components projected independently before cellwise cosine",
            "display": "historical spatial basis only; no model quantity is recomputed after display mapping",
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(output_dir), "n_cells": len(table)}, indent=2))


if __name__ == "__main__":
    main()
