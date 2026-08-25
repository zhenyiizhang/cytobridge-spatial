#!/usr/bin/env python3
"""Render corrected package-native ARISTA Figure 5a with the old plot grammar.

Numerical communication matrices and fixed-particle lineage labels come from
the new package run.  Point geometry comes from its k=1 visualization-only
spatial-warp snapshots.  Every visual parameter is copied from the historical
``3d_plot_5_slices_focus_anchor_local.py`` Figure 5a renderer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree as ET

import anndata as ad
import numpy as np
import plotly.io as pio

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.arista_code.arista_helpers_focus_anchor import (
    plot_3d_spatial_sankey_style_focus_anchor,
)


TIMES = (0.0, 0.5, 1.0, 1.5, 2.0)
OBSERVED_TIMES = (0.0, 1.0, 2.0)
GENERATED_TIMES = (0.5, 1.5)
FILL_RE = re.compile(r"(?:^|;)\s*fill:\s*(#[0-9a-fA-F]{6})")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recolored-input-dir", required=True, type=Path)
    parser.add_argument("--legacy-palette", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def snapshot_path(snapshot_dir: Path, time: float) -> Path:
    if time in OBSERVED_TIMES:
        return snapshot_dir / f"time_{time:.1f}__Observed.svg"
    return snapshot_dir / f"time_{time:.1f}.svg"


def parse_snapshot(path: Path, palette: dict[str, str]) -> tuple[np.ndarray, np.ndarray]:
    root = ET.parse(path).getroot()
    namespace = "{http://www.w3.org/2000/svg}"
    collections = [
        node
        for node in root.iter(namespace + "g")
        if node.get("id", "").startswith("PathCollection_")
    ]
    if not collections:
        raise ValueError(f"No PathCollection found in {path}")
    uses = list(collections[0].iter(namespace + "use"))
    if not uses:
        raise ValueError(f"No cell markers found in {path}")
    color_to_label = {str(color).lower(): str(label) for label, color in palette.items()}
    coordinates: list[tuple[float, float]] = []
    labels: list[str] = []
    for node in uses:
        match = FILL_RE.search(node.get("style", ""))
        if match is None:
            raise ValueError(f"Marker without fill in {path}")
        color = match.group(1).lower()
        if color not in color_to_label:
            raise KeyError(f"Marker color {color} is absent from submitted palette")
        coordinates.append((float(node.get("x")), -float(node.get("y"))))
        labels.append(color_to_label[color])
    return np.asarray(coordinates, dtype=np.float32), np.asarray(labels, dtype=str)


def main() -> None:
    args = parse_args()
    input_dir = args.recolored_input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    palette_path = args.legacy_palette.expanduser().resolve()
    palette = json.loads(palette_path.read_text(encoding="utf-8"))
    snapshot_dir = input_dir / "snapshots"
    comm_path = input_dir / "all_time_communications.pkl"
    lineage_path = input_dir / "fixed_particle_lineage_labels.npz"
    output_dir.mkdir(parents=True, exist_ok=False)

    adata_dict: dict[str, ad.AnnData] = {}
    snapshot_inventory: list[dict[str, object]] = []
    for time in TIMES:
        source = snapshot_path(snapshot_dir, time)
        coordinates, labels = parse_snapshot(source, palette)
        state = ad.AnnData(X=np.zeros((len(labels), 1), dtype=np.float32))
        state.obs["Annotation"] = labels
        state.obsm["spatial"] = coordinates
        key = str(time)
        adata_dict[key] = state
        snapshot_inventory.append(
            {
                "time": time,
                "source": "Observed" if time in OBSERVED_TIMES else "Generated",
                "path": str(source),
                "sha256": sha256(source),
                "n_cells": int(len(labels)),
                "n_cell_types": int(len(np.unique(labels))),
            }
        )

    with comm_path.open("rb") as handle:
        all_communications = pickle.load(handle)
    all_communications = {str(time): all_communications[str(time)] for time in TIMES}
    lineage = np.load(lineage_path, allow_pickle=False)
    lineage_times = np.asarray(lineage["time_points"], dtype=float)
    predicted_labels: list[np.ndarray] = []
    for time in TIMES:
        matches = np.flatnonzero(np.isclose(lineage_times, time))
        if len(matches) != 1:
            raise ValueError(f"Fixed-particle lineage lacks unique time {time}")
        predicted_labels.append(np.asarray(lineage[f"labels_{int(matches[0])}"]).astype(str))

    time_keys = [str(time) for time in TIMES]
    fig = plot_3d_spatial_sankey_style_focus_anchor(
        adata_dict=adata_dict,
        all_time_communications=all_communications,
        time_keys=time_keys,
        label_to_color=palette,
        predicted_labels_list=predicted_labels,
        spatial_key="spatial",
        z_spacing=3.8,
        reverse_time_order=False,
        intra_threshold=0.0,
        edge_focus_celltype="reaEGC",
        edge_top_k=6,
        edge_top_k_focus_label="reaEGC",
        ribbon_min_count=10,
        ribbon_keep_source_cumfrac=0.85,
        ribbon_focus_celltype=["reaEGC"],
        ribbon_focus_source_only=True,
        ribbon_focus_target_only=False,
        background_color=None,
        font_color="#1a1a1a",
        anchor_mode="centroid",
        anchor_subsample=1000,
        highlight_endpoints=True,
        endpoint_size=6,
        endpoint_opacity=0.9,
        edge_color="rgba(25,25,25,0.75)",
        edge_show_arrows=True,
        edge_arrow_position=0.7,
        edge_arrow_in_slice_plane=True,
        edge_arrow_length_scale=0.14,
        edge_arrow_width_scale=0.65,
        edge_line_width_base=5,
        edge_line_width_scale=0.7,
        edge_center_highlight=False,
        edge_center_highlight_width_scale=0.45,
        edge_center_highlight_alpha=0.9,
        bidirectional_offset=0.2,
        bidirectional_curve=True,
        bidirectional_curve_points=18,
        ribbon_line_width_base=6,
        ribbon_line_width_scale=1.0,
        ribbon_line_alpha=0.55,
        ribbon_line_curve=0.12,
        ribbon_line_points=18,
        ribbon_center_highlight=False,
        ribbon_center_highlight_width_scale=0.5,
        ribbon_center_highlight_alpha=0.9,
        point_size=1.0,
        observed_point_subsample=None,
        generated_point_subsample=None,
        observed_point_alpha=0.7,
        generated_point_alpha=0.7,
        observed_point_line_width=0.0,
        generated_point_line_width=0.0,
        generated_point_line_color=None,
        slices_only=False,
        show_time_axis=False,
        show_legend=False,
        show_title=False,
        show_slice_border=True,
        slice_border_width=5,
        slice_border_color_observed="#5f6a72",
        slice_border_color_generated="#8c6d5a",
        slice_fill_color_observed="#e6f0f6",
        slice_fill_color_generated="#f6eee5",
        slice_fill_opacity=0.5,
        observed_time_points=list(OBSERVED_TIMES),
        generated_time_points=list(GENERATED_TIMES),
        focus_anchor_label="reaEGC",
        focus_anchor_k=None,
        focus_anchor_frac=0.2,
        focus_anchor_radius=None,
        focus_anchor_min_count=None,
        width=int(11.69 * 300),
        height=int(8.27 * 300),
        out_html=None,
    )
    fig.update_layout(
        scene_camera=dict(
            eye=dict(x=1.7, y=1.0, z=0.9),
            projection=dict(type="orthographic"),
        ),
        margin=dict(l=10, r=10, t=10, b=10),
        scene=dict(
            domain=dict(x=[0.0, 1.0], y=[0.0, 1.0]),
            aspectratio=dict(x=1.2, y=1.0, z=1.6),
        ),
        font=dict(family="Helvetica", size=16, color="#1a1a1a"),
    )

    html = output_dir / "spatiotemporal_3d.html"
    svg = output_dir / "spatiotemporal_3d.svg"
    pdf = output_dir / "spatiotemporal_3d.pdf"
    png = output_dir / "spatiotemporal_3d.png"
    fig.write_html(html)
    width, height = int(11.69 * 300), int(8.27 * 300)
    pio.write_image(fig, svg, width=width, height=height, scale=3)
    pio.write_image(fig, pdf, width=width, height=height, scale=3)
    pio.write_image(fig, png, width=width, height=height, scale=2)

    trace_types: dict[str, int] = {}
    for trace in fig.data:
        trace_types[trace.type] = trace_types.get(trace.type, 0) + 1
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_contract": {
            "communications": "fresh package-native matrices",
            "lineage": "fresh fixed-particle package-native labels",
            "spatial": "fresh k=1 visualization-only warped snapshots",
        },
        "style_contract": {
            "source": "evaluation/arista_code/3d_plot_5_slices_focus_anchor_local.py",
            "palette_path": str(palette_path),
            "palette_sha256": sha256(palette_path),
            "parameters": "literal historical Figure 5a values",
            "time_slices": list(TIMES),
        },
        "inputs": {
            "communications": {"path": str(comm_path), "sha256": sha256(comm_path)},
            "lineage": {"path": str(lineage_path), "sha256": sha256(lineage_path)},
            "snapshots": snapshot_inventory,
        },
        "qa": {
            "trace_types": trace_types,
            "reaEGC_color": palette.get("reaEGC"),
            "expected_reaEGC_color": "#BA0900",
            "output_exists": {path.name: path.is_file() and path.stat().st_size > 0 for path in [html, svg, pdf, png]},
        },
        "outputs": {
            path.name: {"sha256": sha256(path), "size_bytes": path.stat().st_size}
            for path in [html, svg, pdf, png]
        },
    }
    if manifest["qa"]["reaEGC_color"].lower() != "#ba0900":
        raise AssertionError("Submitted reaEGC color is not locked")
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(output_dir), "qa": manifest["qa"]}, indent=2))


if __name__ == "__main__":
    main()
