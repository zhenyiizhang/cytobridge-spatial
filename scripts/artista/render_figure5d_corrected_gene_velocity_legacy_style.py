#!/usr/bin/env python3
"""Render corrected ARISTA Figure 5d with the frozen legacy scVelo grammar.

This script is intentionally a renderer, not an analysis pipeline.  It accepts
the immutable numerical state produced by
``build_figure5d_corrected_gene_velocity_state.py`` and draws the raw corrected
PCA coordinates and graph-projected gene velocity without rotating, scaling,
or otherwise aligning them to the manuscript coordinates.

The historical display contract is reproduced literally:

* scVelo 0.3.3, Scanpy 1.11.2, and Matplotlib 3.10.3;
* ``velocity_embedding_stream`` with density 2, a 6 x 6 inch canvas, black
  dynamically weighted streamlines, ``-|>`` arrows, arrow size 1, maximum
  length 4, and bidirectional integration;
* a raster-friendly categorical ``.`` scatter at alpha 0.3 using the frozen
  16-cell-type palette plus gray ``Other``;
* all 46,209 cells contribute to the stream grid, while the frozen 20-cell
  display mask affects only scatter visibility (46,189 visible glyphs).

The full SVG is retained as a direct legacy-call oracle with the historical
title and right-side legend.  The stream-only and scatter-only SVGs share its
exact canvas/viewBox and contain no title or legend, making them safe source
layers for object-level placement into the final Illustrator artwork.

Example (state hashes must come from the completed immutable state bundle)::

    NUMBA_CACHE_DIR=/tmp/cb_fig5d_numba_cache \
    MPLCONFIGDIR=/tmp/cb_fig5d_mpl_cache \
    python scripts/arista_paper_equivalent/\
render_figure5d_corrected_gene_velocity_legacy_style.py \
      --state-npz /path/to/figure5d_corrected_gene_velocity_state.npz \
      --state-manifest /path/to/state_manifest.json \
      --expected-state-sha256 <sha256> \
      --expected-state-manifest-sha256 <sha256> \
      --palette-json results/arista_mosta_growth_gene/label_to_color.json \
      --oracle-svg results/arista_mosta_growth_gene/gene_velocity/\
velocity_gene_full_pca_celltype.svg \
      --final-ai /path/to/final/Arista.ai \
      --output-dir output/arista_figure5d_corrected_legacy_renderer_v1

The output directory is immutable: an existing path is never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault(
    "NUMBA_CACHE_DIR",
    str(Path(tempfile.gettempdir()) / "cytobridge_arista_fig5d_numba_cache"),
)
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "cytobridge_arista_fig5d_mpl_cache"),
)
Path(os.environ["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import anndata as ad
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scvelo as scv
from scvelo.plotting.velocity_embedding_grid import compute_velocity_on_grid


WORKFLOW = "arista_figure5d_corrected_raw_pca_legacy_renderer_v1"

EXPECTED_VERSIONS = {
    "matplotlib": "3.10.3",
    "scanpy": "1.11.2",
    "scvelo": "0.3.3",
    "anndata": "0.11.4",
}
EXPECTED_PALETTE_SHA256 = (
    "983b941fc93efe155511994d1d4b16cba5e11982cd81fb298d9a4a78907fbdd7"
)
EXPECTED_ORACLE_SHA256 = (
    "3896065330c26b9e1290b6110c16ecb2c6f477073f54c74bb0ef3a7311a482fa"
)
EXPECTED_FINAL_AI_SHA256 = (
    "673dc81f4856833c30c943ad5f2f4af9e69f771cea4cb63f2484ffbd18907694"
)
EXPECTED_CORRECTED_PCA_RAW_SHA256 = (
    "7434617ebe5ba4cbf88ade8d21b86756a817f144fbb522bbfc0fde4957fa1552"
)
EXPECTED_EMBEDDED_VELOCITY_RAW_SHA256 = (
    "8f06f0b1c7e65453da1e663ad382515dc200a0bb054c276be9a5061b83993eda"
)

EXPECTED_STATE_SCHEMA = "cytobridge.arista.fig5d.corrected-gene-velocity-state.v1"
EXPECTED_N_COMPUTE = 46_209
EXPECTED_N_VISIBLE = 46_189
EXPECTED_N_HIDDEN_SCATTER = 20
EXPECTED_TIME_COUNTS = {0.0: 7_668, 1.0: 8_106, 2.0: 9_440, 3.0: 9_676, 4.0: 11_319}

COLORED_CELL_TYPES = (
    "cckIN",
    "dpEX",
    "mpEX",
    "mpIN",
    "nptxEX",
    "npyIN",
    "ntng1IN",
    "rIPC1",
    "rIPC2",
    "rIPC4",
    "reaEGC",
    "ribEGC",
    "scgnIN",
    "sfrpEGC",
    "sstIN",
    "wntEGC",
)
DISPLAY_CATEGORIES = (*COLORED_CELL_TYPES, "Other")
OTHER_COLOR = "#D0D0D0"
HISTORICAL_FINAL_AI_LABELS = (
    "mpEX",
    "dpEX",
    "nptxEX",
    "rIPC2",
    "rIPC1",
    "rIPC4",
    "wntEGC",
    "reaEGC",
    "ribEGC",
    "sfrpEGC",
    "Other",
    "scgnIN",
)

STATE_KEYS = {
    "corrected_raw_pca",
    "embedded_gene_velocity_pca",
    "labels",
    "times",
    "source_obs_ids",
    "composite_keys",
    "display_mask",
}

SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"
ET.register_namespace("", SVG_NS)
ET.register_namespace("xlink", XLINK_NS)

FULL_SVG_NAME = "velocity_gene_full_pca_celltype_corrected_legacy_full.svg"
STREAM_SVG_NAME = "velocity_gene_full_pca_celltype_corrected_legacy_stream_only.svg"
SCATTER_SVG_NAME = "velocity_gene_full_pca_celltype_corrected_legacy_scatter_only.svg"
GRID_NPZ_NAME = "figure5d_corrected_stream_grid_full46209.npz"
GRID_CSV_NAME = "figure5d_corrected_stream_grid_directions.csv"
ONDATA_ALL17_SVG_NAME = (
    "velocity_gene_full_pca_celltype_corrected_ondata_labels_all17_only.svg"
)
ONDATA_PAPER12_SVG_NAME = (
    "velocity_gene_full_pca_celltype_corrected_ondata_labels_paper12_only.svg"
)
ONDATA_POSITIONS_CSV_NAME = "figure5d_corrected_ondata_label_positions.csv"
MANUAL_ARROW_EVIDENCE_CSV_NAME = "figure5d_corrected_manual_white_arrow_evidence.csv"

# Object-level measurements from the locked final Arista.ai.  Fractions use
# the Figure 5d Matplotlib axes clip, with x increasing rightward and top-y
# increasing downward.  They are style/annotation anchors only.
FINAL_AI_FIG5D_AXES_CLIP_TOP_PT = (
    22.5005634,
    635.6650723,
    229.8165655,
    837.6652960,
)
FINAL_AI_WHITE_ARROW_AXES_FRACTIONS = {
    "curve_start": (0.753649, 0.543460),
    "curve_end": (0.284245, 0.703658),
    "head_anchor": (0.239521, 0.656190),
    "curve_bbox_center": (
        (0.263677 + 0.753649) / 2.0,
        (0.331124 + 0.703658) / 2.0,
    ),
}
FINAL_AI_WHITE_ARROW_OBJECT_SHA256 = (
    "905d7925ee246a077139637cfc1094d11249f1e9240c5f5e2e998d8dbbec10a8"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _require_sha256(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(f"{label} SHA256 changed: {actual} != {expected}")
    return actual


def _sha256_arg(value: str) -> str:
    normalized = value.strip().lower()
    if re.fullmatch(r"[0-9a-f]{64}", normalized) is None:
        raise argparse.ArgumentTypeError("Expected a 64-character lowercase SHA-256 value")
    return normalized


def _resolved_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Missing {label}: {resolved}")
    return resolved


def _svg_geometry(path: Path) -> dict[str, str]:
    root = ET.parse(path).getroot()
    return {
        "width": root.attrib["width"],
        "height": root.attrib["height"],
        "viewBox": root.attrib["viewBox"],
    }


def _element_id(element: ET.Element) -> str:
    return element.attrib.get("id", "")


def _find_id(root: ET.Element, element_id: str) -> ET.Element:
    for element in root.iter():
        if _element_id(element) == element_id:
            return element
    raise ValueError(f"SVG element #{element_id} is missing")


def _stream_widths(line_collection: ET.Element) -> np.ndarray:
    values: list[float] = []
    for path in line_collection.iter(f"{{{SVG_NS}}}path"):
        style = path.attrib.get("style", "")
        match = re.search(r"(?:^|;)\s*stroke-width:\s*([0-9.eE+-]+)", style)
        if match:
            values.append(float(match.group(1)))
    return np.asarray(values, dtype=float)


def _svg_layer_stats(path: Path) -> dict[str, Any]:
    root = ET.parse(path).getroot()
    axes = _find_id(root, "axes_1")
    images = list(axes.iter(f"{{{SVG_NS}}}image"))
    line_groups = [
        element
        for element in axes.iter()
        if _element_id(element).startswith("LineCollection_")
    ]
    arrow_groups = [
        element
        for element in axes.iter()
        if re.fullmatch(r"patch_\d+", _element_id(element)) is not None
    ]
    widths = (
        _stream_widths(line_groups[0])
        if len(line_groups) == 1
        else np.asarray([], dtype=float)
    )
    return {
        "geometry": _svg_geometry(path),
        "embedded_image_count": len(images),
        "embedded_image_geometry": (
            {
                key: images[0].attrib.get(key)
                for key in ("x", "y", "width", "height", "transform")
            }
            if len(images) == 1
            else None
        ),
        "line_collection_count": len(line_groups),
        "line_path_count": (
            len(list(line_groups[0].iter(f"{{{SVG_NS}}}path")))
            if len(line_groups) == 1
            else 0
        ),
        "arrow_group_count": len(arrow_groups),
        "stream_width_count": int(widths.size),
        "stream_width_unique_rounded_6": int(np.unique(np.round(widths, 6)).size),
        "stream_width_min": float(np.nanmin(widths)) if widths.size else None,
        "stream_width_max": float(np.nanmax(widths)) if widths.size else None,
    }


def _svg_category_comments(path: Path) -> list[str]:
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    root = ET.parse(path, parser=parser).getroot()
    axes = _find_id(root, "axes_1")
    return [
        comment
        for child in axes
        if (comment := _group_comment_text(child)) is not None
        and comment in set(DISPLAY_CATEGORIES)
    ]


def _write_layer_sidecar(source_svg: Path, output_svg: Path, role: str) -> None:
    """Extract a source layer without changing the source SVG coordinate frame."""
    if role not in {"stream", "scatter"}:
        raise ValueError(f"Unknown SVG sidecar role: {role}")
    tree = ET.parse(source_svg)
    root = tree.getroot()
    figure = _find_id(root, "figure_1")
    axes = _find_id(root, "axes_1")

    # Component sidecars intentionally contain only the data axes.  Keeping
    # the root viewBox unchanged makes the two layers exactly co-registered.
    for child in list(figure):
        if child is not axes:
            figure.remove(child)

    if role == "stream":
        keep = lambda child_id, tag: (
            child_id.startswith("LineCollection_")
            or re.fullmatch(r"patch_\d+", child_id) is not None
        )
    else:
        keep = lambda child_id, tag: tag == f"{{{SVG_NS}}}image"

    for child in list(axes):
        if not keep(_element_id(child), child.tag):
            axes.remove(child)
    tree.write(output_svg, encoding="utf-8", xml_declaration=True)


def _group_comment_text(group: ET.Element) -> str | None:
    for child in group:
        if child.tag is ET.Comment:
            return (child.text or "").strip()
    return None


def _write_ondata_labels_sidecar(
    *,
    source_svg: Path,
    output_svg: Path,
    full_geometry: dict[str, str],
    keep_labels: tuple[str, ...],
) -> int:
    """Retain selected scVelo on-data text groups on the full legacy canvas."""
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    tree = ET.parse(source_svg, parser=parser)
    root = tree.getroot()
    figure = _find_id(root, "figure_1")
    axes = _find_id(root, "axes_1")
    keep_set = set(keep_labels)

    for child in list(figure):
        if child is not axes:
            figure.remove(child)
    # scVelo adds the labels in categorical order.  Because its white-stroke
    # path effect expands each label to raw SVG paths, Matplotlib does not
    # retain the usual text comment for these 17 groups.  The title is the
    # only ``text_*`` group with a comment and is deliberately excluded.
    label_groups = [
        child
        for child in axes
        if _element_id(child).startswith("text_")
        and _group_comment_text(child) is None
    ]
    if len(label_groups) != len(DISPLAY_CATEGORIES):
        raise ValueError(
            f"scVelo on-data SVG has {len(label_groups)} label path groups, expected 17"
        )
    group_to_label = {
        group: label for group, label in zip(label_groups, DISPLAY_CATEGORIES)
    }
    kept: list[str] = []
    for child in list(axes):
        label = group_to_label.get(child)
        if label in keep_set:
            child.insert(0, ET.Comment(f" {label} "))
            kept.append(label)
        else:
            axes.remove(child)
    if set(kept) != keep_set or len(kept) != len(keep_set):
        raise ValueError(
            f"On-data label extraction changed: kept={sorted(kept)}, "
            f"expected={sorted(keep_set)}"
        )
    root.attrib.update(full_geometry)
    tree.write(output_svg, encoding="utf-8", xml_declaration=True)
    return len(kept)


def _load_state_manifest(path: Path, state_path: Path, state_sha256: str) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != EXPECTED_STATE_SCHEMA:
        raise ValueError(
            f"Corrected state schema changed: {manifest.get('schema')} != {EXPECTED_STATE_SCHEMA}"
        )
    outputs = manifest.get("outputs", {})
    state_record = outputs.get(state_path.name)
    if not isinstance(state_record, dict) or state_record.get("sha256") != state_sha256:
        raise ValueError("State manifest does not lock the supplied numerical NPZ")

    scientific = manifest.get("scientific_state", {})
    expected_counts = {
        "n_cells_compute": EXPECTED_N_COMPUTE,
        "n_cells_visible": EXPECTED_N_VISIBLE,
        "n_graph_only_hidden_scatter": EXPECTED_N_HIDDEN_SCATTER,
        "corrected_raw_pca_sha256": EXPECTED_CORRECTED_PCA_RAW_SHA256,
        "embedded_velocity_raw_sha256": EXPECTED_EMBEDDED_VELOCITY_RAW_SHA256,
    }
    for key, expected in expected_counts.items():
        if scientific.get(key) != expected:
            raise ValueError(
                f"State-manifest scientific contract changed for {key}: "
                f"{scientific.get(key)} != {expected}"
            )

    alignment = manifest.get("display_alignment", {})
    if alignment.get("mapping") != "none":
        raise ValueError("Figure 5d renderer refuses a mapped/aligned state")
    if alignment.get("coordinates") != "raw corrected fresh PCA":
        raise ValueError("Figure 5d renderer requires raw corrected fresh PCA coordinates")
    if alignment.get("manuscript_coordinate_reuse") is not False:
        raise ValueError("Figure 5d renderer refuses manuscript-coordinate reuse")

    algorithm = manifest.get("algorithm_contract", {})
    if algorithm.get("display_orientation") != (
        "none; raw corrected fresh PCA coordinates are used directly"
    ):
        raise ValueError("State manifest no longer locks the raw corrected PCA orientation")
    return manifest


def _load_palette(path: Path) -> dict[str, str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    palette = {str(key): str(value) for key, value in raw.items()}
    missing = sorted(set(COLORED_CELL_TYPES) - set(palette))
    if missing:
        raise KeyError(f"Frozen legacy palette is missing labels: {missing}")
    display = {label: palette[label] for label in COLORED_CELL_TYPES}
    display["Other"] = OTHER_COLOR
    return display


def _load_state(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(STATE_KEYS - set(archive.files))
        if missing:
            raise KeyError(f"Corrected Figure 5d state is missing arrays: {missing}")
        state = {key: np.asarray(archive[key]) for key in STATE_KEYS}

    coordinates = state["corrected_raw_pca"]
    velocity = state["embedded_gene_velocity_pca"]
    labels = state["labels"].astype(str)
    times = np.asarray(state["times"], dtype=np.float64)
    source_ids = state["source_obs_ids"].astype(str)
    composite_keys = state["composite_keys"].astype(str)
    display_mask = np.asarray(state["display_mask"], dtype=bool)

    vector_shapes = {
        "labels": labels,
        "times": times,
        "source_obs_ids": source_ids,
        "composite_keys": composite_keys,
        "display_mask": display_mask,
    }
    if coordinates.shape != (EXPECTED_N_COMPUTE, 2):
        raise ValueError(f"Corrected raw PCA shape changed: {coordinates.shape}")
    if velocity.shape != (EXPECTED_N_COMPUTE, 2):
        raise ValueError(f"Corrected embedded velocity shape changed: {velocity.shape}")
    for name, array in vector_shapes.items():
        if array.shape != (EXPECTED_N_COMPUTE,):
            raise ValueError(f"Corrected state {name} shape changed: {array.shape}")
    if not np.isfinite(coordinates).all() or not np.isfinite(velocity).all():
        raise ValueError("Corrected Figure 5d coordinates/velocity contain non-finite values")
    if not np.isfinite(times).all():
        raise ValueError("Corrected Figure 5d time values contain non-finite values")
    if int(display_mask.sum()) != EXPECTED_N_VISIBLE:
        raise ValueError(f"Visible-cell roster changed: {display_mask.sum()} != {EXPECTED_N_VISIBLE}")
    if int((~display_mask).sum()) != EXPECTED_N_HIDDEN_SCATTER:
        raise ValueError("The official 20-cell scatter-only exclusion mask changed")
    if np.unique(source_ids).size != EXPECTED_N_COMPUTE:
        raise ValueError("Corrected Figure 5d source_obs_ids are not unique")
    if np.unique(composite_keys).size != EXPECTED_N_COMPUTE:
        raise ValueError("Corrected Figure 5d composite keys are not unique")
    actual_time_counts = {
        float(time): int(count)
        for time, count in zip(*np.unique(times, return_counts=True))
    }
    if actual_time_counts != EXPECTED_TIME_COUNTS:
        raise ValueError(f"Observed time roster changed: {actual_time_counts}")

    pca_hash = _array_sha256(coordinates)
    velocity_hash = _array_sha256(velocity)
    if pca_hash != EXPECTED_CORRECTED_PCA_RAW_SHA256:
        raise ValueError(f"Corrected raw PCA bytes changed: {pca_hash}")
    if velocity_hash != EXPECTED_EMBEDDED_VELOCITY_RAW_SHA256:
        raise ValueError(f"Corrected embedded velocity bytes changed: {velocity_hash}")

    return {
        "coordinates": coordinates,
        "velocity": velocity,
        "labels": labels,
        "times": times,
        "source_obs_ids": source_ids,
        "composite_keys": composite_keys,
        "display_mask": display_mask,
    }


def _build_visible_adata(
    *,
    state: dict[str, np.ndarray],
    palette: dict[str, str],
) -> ad.AnnData:
    display_mask = state["display_mask"]
    labels = state["labels"]
    display_labels = np.where(
        np.isin(labels, np.asarray(COLORED_CELL_TYPES, dtype=object)),
        labels,
        "Other",
    )
    visible_labels = display_labels[display_mask]
    missing_categories = [
        category for category in DISPLAY_CATEGORIES if category not in set(visible_labels)
    ]
    if missing_categories:
        raise ValueError(f"Legacy Figure 5d display categories disappeared: {missing_categories}")

    plot = ad.AnnData(X=np.zeros((EXPECTED_N_VISIBLE, 1), dtype=np.float32))
    plot.obsm["X_pca"] = np.asarray(state["coordinates"][display_mask]).copy()
    plot.obsm["velocity_pca"] = np.asarray(state["velocity"][display_mask]).copy()
    plot.obs["paper_cell_type"] = pd.Categorical(
        visible_labels,
        categories=list(DISPLAY_CATEGORIES),
        ordered=False,
    )
    plot.obs["time"] = state["times"][display_mask]
    plot.uns["paper_cell_type_colors"] = [palette[label] for label in DISPLAY_CATEGORIES]
    # This is not needed when velocity_pca exists, but explicitly documents
    # that no renderer-side velocity projection is allowed.
    plot.uns["velocity_params"] = {"embeddings": ["pca"]}
    return plot


def _compute_full_stream_grid(
    coordinates: np.ndarray,
    velocity: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mirror scVelo 0.3.3's stream-grid branch on all corrected cells."""
    x_grid, v_grid = compute_velocity_on_grid(
        X_emb=np.asarray(coordinates),
        V_emb=np.asarray(velocity),
        density=1,
        smooth=None,
        min_mass=None,
        n_neighbors=None,
        autoscale=False,
        adjust_for_stream=True,
        cutoff_perc=None,
    )
    lengths = np.sqrt((v_grid**2).sum(0))
    finite = np.isfinite(lengths)
    if not finite.any() or float(np.nanmax(lengths)) <= 0:
        raise ValueError("Corrected Figure 5d stream grid has no finite nonzero velocities")
    linewidth = 2.0 * lengths / lengths[finite].max()
    if x_grid.shape != (2, 50) or v_grid.shape != (2, 50, 50):
        raise ValueError(
            f"Legacy scVelo stream-grid shape changed: X={x_grid.shape}, V={v_grid.shape}"
        )
    if np.unique(np.round(linewidth[np.isfinite(linewidth)], 8)).size <= 1:
        raise ValueError("Legacy dynamic stream linewidth unexpectedly became constant")
    return x_grid, v_grid, linewidth


def _set_frozen_legacy_figure_params() -> None:
    scv.settings.set_figure_params("scvelo")
    # The locked SVG oracle resolves the historical sans-serif request to
    # DejaVu Sans (its glyph IDs are ``DejaVuSans-*``), not to the Arial now
    # installed on this workstation.  Force the oracle-resolved family so a
    # newer font cache cannot silently change title/legend metrics or bbox.
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["font.sans-serif"] = [
        "DejaVu Sans",
        "Arial",
        "Helvetica",
        "sans-serif",
    ]
    matplotlib.rcParams["svg.hashsalt"] = WORKFLOW


def _render_legacy_full(
    *,
    plot: ad.AnnData,
    palette: dict[str, str],
    x_grid: np.ndarray,
    v_grid: np.ndarray,
    linewidth: np.ndarray,
    output_svg: Path,
    title: str = "Gene velocity full (PCA, cell type)",
) -> None:
    # This call deliberately follows growth_gene.py and does not import the
    # newer CytoBridge paper mplstyle.  The manuscript oracle was generated by
    # scVelo's own style state.
    _set_frozen_legacy_figure_params()
    ax = scv.pl.velocity_embedding_stream(
        plot,
        basis="pca",
        vkey="velocity",
        color="paper_cell_type",
        density=2,
        smooth=None,
        min_mass=None,
        cutoff_perc=None,
        arrow_color="black",
        arrow_size=1,
        arrow_style="-|>",
        max_length=4,
        integration_direction="both",
        linewidth=linewidth,
        n_neighbors=None,
        recompute=False,
        palette=[palette[label] for label in DISPLAY_CATEGORIES],
        size=None,
        alpha=0.3,
        X_grid=x_grid,
        V_grid=v_grid,
        sort_order=True,
        legend_loc="right",
        title=title,
        figsize=(6, 6),
        frameon=None,
        marker=".",
        show=False,
    )
    fig = ax.figure
    try:
        fig.savefig(output_svg, format="svg", dpi=300, bbox_inches="tight")
    finally:
        plt.close(fig)


def _render_ondata_work_svg(
    *,
    plot: ad.AnnData,
    palette: dict[str, str],
    x_grid: np.ndarray,
    v_grid: np.ndarray,
    linewidth: np.ndarray,
    output_svg: Path,
    title: str = "Gene velocity full (PCA, cell type)",
) -> list[dict[str, Any]]:
    """Run scVelo's exact on-data label placement and record its medians."""
    _set_frozen_legacy_figure_params()
    ax = scv.pl.velocity_embedding_stream(
        plot,
        basis="pca",
        vkey="velocity",
        color="paper_cell_type",
        density=2,
        smooth=None,
        min_mass=None,
        cutoff_perc=None,
        arrow_color="black",
        arrow_size=1,
        arrow_style="-|>",
        max_length=4,
        integration_direction="both",
        linewidth=linewidth,
        n_neighbors=None,
        recompute=False,
        palette=[palette[label] for label in DISPLAY_CATEGORIES],
        size=None,
        alpha=0.3,
        X_grid=x_grid,
        V_grid=v_grid,
        sort_order=True,
        legend_loc="on data",
        title=title,
        figsize=(6, 6),
        frameon=None,
        marker=".",
        show=False,
    )
    fig = ax.figure
    try:
        fig.canvas.draw()
        rows: list[dict[str, Any]] = []
        coordinates = np.asarray(plot.obsm["X_pca"])
        labels = plot.obs["paper_cell_type"].astype(str).to_numpy()
        text_by_label = {text.get_text(): text for text in ax.texts}
        if set(text_by_label) != set(DISPLAY_CATEGORIES):
            raise ValueError(
                "scVelo on-data label roster changed: "
                f"{sorted(text_by_label)} != {sorted(DISPLAY_CATEGORIES)}"
            )
        for category in DISPLAY_CATEGORIES:
            text = text_by_label[category]
            x_data, y_data = (float(value) for value in text.get_position())
            expected_median = np.nanmedian(coordinates[labels == category], axis=0)
            if not np.allclose(
                [x_data, y_data], expected_median, rtol=0.0, atol=1e-12
            ):
                raise ValueError(
                    f"scVelo on-data position for {category} is no longer its median"
                )
            x_axes, y_axes = ax.transAxes.inverted().transform(
                ax.transData.transform([x_data, y_data])
            )
            rows.append(
                {
                    "cell_type": category,
                    "n_visible_cells": int(np.sum(labels == category)),
                    "raw_corrected_pca_median_x": x_data,
                    "raw_corrected_pca_median_y": y_data,
                    "axes_fraction_x": float(x_axes),
                    "axes_fraction_y": float(y_axes),
                    "in_historical_final_ai_12": category
                    in HISTORICAL_FINAL_AI_LABELS,
                    "placement_grammar": "scVelo legend_loc='on data'; coordinate-wise median",
                }
            )
        fig.savefig(output_svg, format="svg", dpi=300, bbox_inches="tight")
    finally:
        plt.close(fig)
    return rows


def _write_stream_direction_csv(
    *,
    x_grid: np.ndarray,
    v_grid: np.ndarray,
    linewidth: np.ndarray,
    output_csv: Path,
) -> int:
    rows: list[dict[str, Any]] = []
    for iy, y_value in enumerate(x_grid[1]):
        for ix, x_value in enumerate(x_grid[0]):
            vx = float(v_grid[0, iy, ix])
            vy = float(v_grid[1, iy, ix])
            valid = bool(np.isfinite(vx) and np.isfinite(vy))
            speed = float(np.hypot(vx, vy)) if valid else np.nan
            rows.append(
                {
                    "grid_x_index": ix,
                    "grid_y_index": iy,
                    "raw_corrected_pca_x": float(x_value),
                    "raw_corrected_pca_y": float(y_value),
                    "velocity_x": vx,
                    "velocity_y": vy,
                    "speed": speed,
                    "unit_direction_x": vx / speed if valid and speed > 0 else np.nan,
                    "unit_direction_y": vy / speed if valid and speed > 0 else np.nan,
                    "direction_degrees_ccw_from_positive_pc1": (
                        float(np.degrees(np.arctan2(vy, vx))) if valid else np.nan
                    ),
                    "dynamic_linewidth": float(linewidth[iy, ix]),
                    "stream_grid_unmasked": valid,
                }
            )
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    return len(rows)


def _sample_stream_grid_at_top_fraction(
    *,
    x_grid: np.ndarray,
    v_grid: np.ndarray,
    x_fraction: float,
    top_y_fraction: float,
) -> dict[str, Any]:
    nx, ny = len(x_grid[0]), len(x_grid[1])
    ix_float = float(x_fraction * (nx - 1))
    iy_float = float((1.0 - top_y_fraction) * (ny - 1))
    ix0, iy0 = int(np.floor(ix_float)), int(np.floor(iy_float))
    ix1, iy1 = min(ix0 + 1, nx - 1), min(iy0 + 1, ny - 1)
    tx, ty = ix_float - ix0, iy_float - iy0
    vectors = np.asarray(
        [
            v_grid[:, iy0, ix0],
            v_grid[:, iy0, ix1],
            v_grid[:, iy1, ix0],
            v_grid[:, iy1, ix1],
        ]
    )
    weights = np.asarray(
        [(1 - tx) * (1 - ty), tx * (1 - ty), (1 - tx) * ty, tx * ty]
    )
    if np.isfinite(vectors).all():
        vector = np.sum(vectors * weights[:, None], axis=0)
        method = "bilinear_four_finite_grid_nodes"
        nearest_ix, nearest_iy = int(round(ix_float)), int(round(iy_float))
    else:
        valid_y, valid_x = np.where(np.isfinite(v_grid).all(axis=0))
        nearest = int(
            np.argmin((valid_x - ix_float) ** 2 + (valid_y - iy_float) ** 2)
        )
        nearest_ix, nearest_iy = int(valid_x[nearest]), int(valid_y[nearest])
        vector = np.asarray(v_grid[:, nearest_iy, nearest_ix])
        method = "nearest_finite_grid_node"
    speed = float(np.linalg.norm(vector))
    if not np.isfinite(speed) or speed <= 0:
        raise ValueError("Manual-arrow evidence sampled a nonfinite/zero corrected vector")
    unit = vector / speed
    raw_x = float(
        x_grid[0, 0] + x_fraction * (x_grid[0, -1] - x_grid[0, 0])
    )
    raw_y = float(
        x_grid[1, -1]
        - top_y_fraction * (x_grid[1, -1] - x_grid[1, 0])
    )
    return {
        "axes_fraction_x": float(x_fraction),
        "axes_top_fraction_y": float(top_y_fraction),
        "raw_corrected_pca_x": raw_x,
        "raw_corrected_pca_y": raw_y,
        "grid_x_float": ix_float,
        "grid_y_float_bottom_origin": iy_float,
        "nearest_grid_x_index": nearest_ix,
        "nearest_grid_y_index": nearest_iy,
        "corrected_velocity_x": float(vector[0]),
        "corrected_velocity_y": float(vector[1]),
        "corrected_speed": speed,
        "corrected_unit_direction_x": float(unit[0]),
        "corrected_unit_direction_y": float(unit[1]),
        "corrected_direction_degrees_ccw_from_positive_pc1": float(
            np.degrees(np.arctan2(unit[1], unit[0]))
        ),
        "interpolation_method": method,
    }


def _write_manual_arrow_evidence(
    *,
    x_grid: np.ndarray,
    v_grid: np.ndarray,
    output_csv: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sampled: dict[str, dict[str, Any]] = {
        name: _sample_stream_grid_at_top_fraction(
            x_grid=x_grid,
            v_grid=v_grid,
            x_fraction=fractions[0],
            top_y_fraction=fractions[1],
        )
        for name, fractions in FINAL_AI_WHITE_ARROW_AXES_FRACTIONS.items()
    }
    start = np.asarray(
        [
            sampled["curve_start"]["raw_corrected_pca_x"],
            sampled["curve_start"]["raw_corrected_pca_y"],
        ]
    )
    head = np.asarray(
        [
            sampled["head_anchor"]["raw_corrected_pca_x"],
            sampled["head_anchor"]["raw_corrected_pca_y"],
        ]
    )
    historical_chord = head - start
    historical_chord /= np.linalg.norm(historical_chord)
    historical_angle = float(
        np.degrees(np.arctan2(historical_chord[1], historical_chord[0]))
    )

    rows: list[dict[str, Any]] = []
    for name, row in sampled.items():
        corrected_unit = np.asarray(
            [row["corrected_unit_direction_x"], row["corrected_unit_direction_y"]]
        )
        cosine = float(np.dot(corrected_unit, historical_chord))
        angle_delta = float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
        rows.append(
            {
                "anchor": name,
                **row,
                "historical_white_arrow_chord_unit_x": float(historical_chord[0]),
                "historical_white_arrow_chord_unit_y": float(historical_chord[1]),
                "historical_white_arrow_chord_angle_degrees": historical_angle,
                "cosine_corrected_vs_historical_chord": cosine,
                "absolute_angle_delta_degrees": angle_delta,
            }
        )
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    head_row = next(row for row in rows if row["anchor"] == "head_anchor")
    end_row = next(row for row in rows if row["anchor"] == "curve_end")
    summary = {
        "historical_chord_unit": historical_chord.astype(float).tolist(),
        "historical_chord_angle_degrees": historical_angle,
        "head_corrected_direction_degrees": head_row[
            "corrected_direction_degrees_ccw_from_positive_pc1"
        ],
        "head_cosine_corrected_vs_historical_chord": head_row[
            "cosine_corrected_vs_historical_chord"
        ],
        "head_absolute_angle_delta_degrees": head_row[
            "absolute_angle_delta_degrees"
        ],
        "curve_end_cosine_corrected_vs_historical_chord": end_row[
            "cosine_corrected_vs_historical_chord"
        ],
        "decision_evidence": "do_not_reuse_historical_white_arrow_direction_unmodified",
    }
    return rows, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-npz", required=True, type=Path)
    parser.add_argument("--state-manifest", required=True, type=Path)
    parser.add_argument("--expected-state-sha256", required=True, type=_sha256_arg)
    parser.add_argument(
        "--expected-state-manifest-sha256", required=True, type=_sha256_arg
    )
    parser.add_argument("--palette-json", required=True, type=Path)
    parser.add_argument("--oracle-svg", required=True, type=Path)
    parser.add_argument("--final-ai", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--dynamic-state-contract",
        action="store_true",
        help=(
            "Read cell counts, time counts, and numerical array hashes from the "
            "supplied immutable state manifest. This is required for a newly "
            "trained package-native ARISTA model; the historical plotting "
            "grammar and locked visual oracles remain unchanged."
        ),
    )
    parser.add_argument(
        "--expected-palette-sha256",
        type=_sha256_arg,
        default=EXPECTED_PALETTE_SHA256,
    )
    parser.add_argument(
        "--expected-oracle-sha256",
        type=_sha256_arg,
        default=EXPECTED_ORACLE_SHA256,
    )
    parser.add_argument(
        "--expected-final-ai-sha256",
        type=_sha256_arg,
        default=EXPECTED_FINAL_AI_SHA256,
    )
    return parser.parse_args()


def main() -> int:
    global EXPECTED_N_COMPUTE
    global EXPECTED_N_VISIBLE
    global EXPECTED_N_HIDDEN_SCATTER
    global EXPECTED_TIME_COUNTS
    global EXPECTED_CORRECTED_PCA_RAW_SHA256
    global EXPECTED_EMBEDDED_VELOCITY_RAW_SHA256

    args = parse_args()
    state_path = _resolved_file(args.state_npz, "corrected Figure 5d state NPZ")
    state_manifest_path = _resolved_file(
        args.state_manifest, "corrected Figure 5d state manifest"
    )
    palette_path = _resolved_file(args.palette_json, "frozen ARISTA legacy palette")
    oracle_path = _resolved_file(args.oracle_svg, "historical Figure 5d SVG oracle")
    final_ai_path = _resolved_file(args.final_ai, "final Arista.ai artwork")

    versions = {
        "python": sys.version.split()[0],
        "matplotlib": matplotlib.__version__,
        "scanpy": sc.__version__,
        "scvelo": scv.__version__,
        "anndata": ad.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    for package, expected in EXPECTED_VERSIONS.items():
        if versions[package] != expected:
            raise RuntimeError(
                f"Historical Figure 5d renderer requires {package}={expected}; "
                f"got {versions[package]}"
            )

    input_hashes = {
        "state_npz": _require_sha256(
            state_path, args.expected_state_sha256, "corrected state NPZ"
        ),
        "state_manifest": _require_sha256(
            state_manifest_path,
            args.expected_state_manifest_sha256,
            "corrected state manifest",
        ),
        "palette_json": _require_sha256(
            palette_path, args.expected_palette_sha256, "frozen legacy palette"
        ),
        "oracle_svg": _require_sha256(
            oracle_path, args.expected_oracle_sha256, "historical Figure 5d SVG"
        ),
        "final_ai": _require_sha256(
            final_ai_path, args.expected_final_ai_sha256, "final Arista.ai artwork"
        ),
    }
    if args.dynamic_state_contract:
        dynamic_manifest = json.loads(state_manifest_path.read_text(encoding="utf-8"))
        scientific = dynamic_manifest.get("scientific_state", {})
        time_counts = scientific.get("observed_counts_by_time")
        required_dynamic = {
            "n_cells_compute": scientific.get("n_cells_compute"),
            "n_cells_visible": scientific.get("n_cells_visible"),
            "n_graph_only_hidden_scatter": scientific.get(
                "n_graph_only_hidden_scatter"
            ),
            "corrected_raw_pca_sha256": scientific.get(
                "corrected_raw_pca_sha256"
            ),
            "embedded_velocity_raw_sha256": scientific.get(
                "embedded_velocity_raw_sha256"
            ),
            "observed_counts_by_time": time_counts,
        }
        missing_dynamic = [
            key for key, value in required_dynamic.items() if value is None
        ]
        if missing_dynamic:
            raise ValueError(
                "Dynamic Figure 5d state manifest is incomplete: "
                f"{missing_dynamic}"
            )
        EXPECTED_N_COMPUTE = int(scientific["n_cells_compute"])
        EXPECTED_N_VISIBLE = int(scientific["n_cells_visible"])
        EXPECTED_N_HIDDEN_SCATTER = int(
            scientific["n_graph_only_hidden_scatter"]
        )
        EXPECTED_TIME_COUNTS = {
            float(time): int(count) for time, count in dict(time_counts).items()
        }
        EXPECTED_CORRECTED_PCA_RAW_SHA256 = str(
            scientific["corrected_raw_pca_sha256"]
        )
        EXPECTED_EMBEDDED_VELOCITY_RAW_SHA256 = str(
            scientific["embedded_velocity_raw_sha256"]
        )
        if EXPECTED_N_COMPUTE <= 0 or EXPECTED_N_VISIBLE <= 0:
            raise ValueError("Dynamic Figure 5d state has an empty cell roster")
        if EXPECTED_N_VISIBLE + EXPECTED_N_HIDDEN_SCATTER != EXPECTED_N_COMPUTE:
            raise ValueError(
                "Dynamic Figure 5d visible/hidden counts do not sum to compute N"
            )
    state_manifest = _load_state_manifest(
        state_manifest_path, state_path, input_hashes["state_npz"]
    )
    velocity_component = str(
        state_manifest.get("algorithm_contract", {}).get(
            "velocity_component", "full"
        )
    )
    if velocity_component not in {"full", "drift"}:
        raise ValueError(
            f"Unsupported Figure 5d velocity component: {velocity_component}"
        )
    sidecar_title = (
        "Gene velocity full (PCA, cell type)"
        if velocity_component == "full"
        else "Gene velocity intrinsic (PCA, cell type)"
    )
    state = _load_state(state_path)
    palette = _load_palette(palette_path)

    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite immutable output: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.render-", dir=output_dir.parent)
    )
    try:
        plot = _build_visible_adata(state=state, palette=palette)
        x_grid, v_grid, linewidth = _compute_full_stream_grid(
            state["coordinates"], state["velocity"]
        )

        grid_path = stage / GRID_NPZ_NAME
        grid_csv_path = stage / GRID_CSV_NAME
        np.savez_compressed(
            grid_path,
            X_grid=x_grid,
            V_grid=v_grid,
            dynamic_linewidth=linewidth,
        )
        grid_direction_rows = _write_stream_direction_csv(
            x_grid=x_grid,
            v_grid=v_grid,
            linewidth=linewidth,
            output_csv=grid_csv_path,
        )
        if grid_direction_rows != 2_500:
            raise RuntimeError(f"Corrected stream direction grid changed: {grid_direction_rows}")
        manual_arrow_evidence_csv = stage / MANUAL_ARROW_EVIDENCE_CSV_NAME
        manual_arrow_rows, manual_arrow_summary = _write_manual_arrow_evidence(
            x_grid=x_grid,
            v_grid=v_grid,
            output_csv=manual_arrow_evidence_csv,
        )
        if len(manual_arrow_rows) != len(FINAL_AI_WHITE_ARROW_AXES_FRACTIONS):
            raise RuntimeError("Manual white-arrow corrected direction evidence changed")

        full_svg = stage / FULL_SVG_NAME
        stream_svg = stage / STREAM_SVG_NAME
        scatter_svg = stage / SCATTER_SVG_NAME
        ondata_work_svg = stage / ".velocity_gene_ondata_work.svg"
        ondata_all17_svg = stage / ONDATA_ALL17_SVG_NAME
        ondata_paper12_svg = stage / ONDATA_PAPER12_SVG_NAME
        ondata_positions_csv = stage / ONDATA_POSITIONS_CSV_NAME
        _render_legacy_full(
            plot=plot,
            palette=palette,
            x_grid=x_grid,
            v_grid=v_grid,
            linewidth=linewidth,
            output_svg=full_svg,
            title=sidecar_title,
        )

        oracle_geometry = _svg_geometry(oracle_path)
        full_geometry = _svg_geometry(full_svg)
        if full_geometry != oracle_geometry:
            raise RuntimeError(
                "Legacy renderer canvas/layout drifted from the historical SVG: "
                f"{full_geometry} != {oracle_geometry}"
            )

        _write_layer_sidecar(full_svg, stream_svg, "stream")
        _write_layer_sidecar(full_svg, scatter_svg, "scatter")
        full_stats = _svg_layer_stats(full_svg)
        ondata_rows = _render_ondata_work_svg(
            plot=plot,
            palette=palette,
            x_grid=x_grid,
            v_grid=v_grid,
            linewidth=linewidth,
            output_svg=ondata_work_svg,
            title=sidecar_title,
        )
        pd.DataFrame(ondata_rows).to_csv(ondata_positions_csv, index=False)
        ondata_work_stats = _svg_layer_stats(ondata_work_svg)
        if (
            ondata_work_stats["embedded_image_geometry"]
            != full_stats["embedded_image_geometry"]
        ):
            raise RuntimeError(
                "On-data label call no longer shares the legacy data-axis placement"
            )
        ondata_all17_count = _write_ondata_labels_sidecar(
            source_svg=ondata_work_svg,
            output_svg=ondata_all17_svg,
            full_geometry=full_geometry,
            keep_labels=DISPLAY_CATEGORIES,
        )
        ondata_paper12_count = _write_ondata_labels_sidecar(
            source_svg=ondata_work_svg,
            output_svg=ondata_paper12_svg,
            full_geometry=full_geometry,
            keep_labels=HISTORICAL_FINAL_AI_LABELS,
        )
        ondata_work_svg.unlink()
        stream_stats = _svg_layer_stats(stream_svg)
        scatter_stats = _svg_layer_stats(scatter_svg)
        ondata_all17_stats = _svg_layer_stats(ondata_all17_svg)
        ondata_paper12_stats = _svg_layer_stats(ondata_paper12_svg)
        oracle_stats = _svg_layer_stats(oracle_path)

        if full_stats["embedded_image_count"] != 1:
            raise RuntimeError("Legacy full SVG must contain exactly one raster scatter image")
        if full_stats["line_collection_count"] != 1:
            raise RuntimeError("Legacy full SVG must contain one vector stream LineCollection")
        if full_stats["line_path_count"] <= 0 or full_stats["arrow_group_count"] <= 0:
            raise RuntimeError("Legacy full SVG lost vector streamlines or arrows")
        if full_stats["stream_width_unique_rounded_6"] <= 1:
            raise RuntimeError("Legacy full SVG lost dynamic stream widths")
        if full_stats["embedded_image_geometry"] != oracle_stats["embedded_image_geometry"]:
            raise RuntimeError(
                "Legacy scatter raster placement drifted from the historical SVG: "
                f"{full_stats['embedded_image_geometry']} != "
                f"{oracle_stats['embedded_image_geometry']}"
            )
        if stream_stats["embedded_image_count"] != 0:
            raise RuntimeError("Stream-only SVG unexpectedly contains a raster image")
        if stream_stats["line_collection_count"] != 1:
            raise RuntimeError("Stream-only SVG lost its vector LineCollection")
        if stream_stats["arrow_group_count"] != full_stats["arrow_group_count"]:
            raise RuntimeError("Stream-only SVG lost vector arrow patches")
        if scatter_stats["embedded_image_count"] != 1:
            raise RuntimeError("Scatter-only SVG lost its raster-friendly scatter layer")
        if scatter_stats["line_collection_count"] != 0:
            raise RuntimeError("Scatter-only SVG still contains vector streamlines")
        if scatter_stats["arrow_group_count"] != 0:
            raise RuntimeError("Scatter-only SVG still contains vector arrows")
        if ondata_all17_count != 17 or ondata_paper12_count != 12:
            raise RuntimeError("Corrected on-data label sidecar counts changed")
        if set(_svg_category_comments(ondata_all17_svg)) != set(DISPLAY_CATEGORIES):
            raise RuntimeError("All-17 on-data label SVG roster changed")
        if set(_svg_category_comments(ondata_paper12_svg)) != set(
            HISTORICAL_FINAL_AI_LABELS
        ):
            raise RuntimeError("Paper-12 on-data label SVG roster changed")
        for label_stats in (ondata_all17_stats, ondata_paper12_stats):
            if (
                label_stats["embedded_image_count"] != 0
                or label_stats["line_collection_count"] != 0
                or label_stats["arrow_group_count"] != 0
            ):
                raise RuntimeError("On-data labels-only SVG contains a non-text data layer")
        if not (
            full_stats["geometry"]
            == stream_stats["geometry"]
            == scatter_stats["geometry"]
            == ondata_all17_stats["geometry"]
            == ondata_paper12_stats["geometry"]
            == oracle_geometry
        ):
            raise RuntimeError("Figure 5d layer sidecars are not exactly co-registered")

        manifest: dict[str, Any] = {
            "schema": "cytobridge.arista.fig5d.corrected-legacy-renderer.v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "workflow": WORKFLOW,
            "runtime": versions,
            "scientific_contract": {
                "coordinates": "raw corrected fresh PCA; no display transform",
                "velocity": (
                    "corrected graph-projected full 50D gene velocity in raw corrected PCA"
                    if velocity_component == "full"
                    else "corrected graph-projected intrinsic drift 50D gene velocity in raw corrected PCA"
                ),
                "velocity_component": velocity_component,
                "n_cells_stream_compute": EXPECTED_N_COMPUTE,
                "n_cells_scatter_visible": EXPECTED_N_VISIBLE,
                "n_cells_scatter_hidden_only": EXPECTED_N_HIDDEN_SCATTER,
                "stream_grid_uses_hidden_cells": True,
                "scatter_mask_changes_stream_grid": False,
                "corrected_raw_pca_sha256": _array_sha256(state["coordinates"]),
                "embedded_velocity_raw_sha256": _array_sha256(state["velocity"]),
                "state_display_alignment": state_manifest["display_alignment"],
            },
            "legacy_style_contract": {
                "renderer": "scv.pl.velocity_embedding_stream",
                "basis": "pca",
                "figsize_inches": [6.0, 6.0],
                "density": 2.0,
                "grid_density_internal": 1.0,
                "grid_shape": [50, 50],
                "grid_default_n_neighbors": int(EXPECTED_N_COMPUTE / 50),
                "smooth_argument": None,
                "smooth_effective": 0.5,
                "min_mass_argument": None,
                "cutoff_perc_argument": None,
                "arrow_color": "black",
                "arrow_size": 1.0,
                "arrow_style": "-|>",
                "max_length": 4.0,
                "integration_direction": "both",
                "linewidth": "scVelo 0.3.3 dynamic 2*speed/max(speed)",
                "scatter_marker": ".",
                "scatter_alpha": 0.3,
                "scatter_size": (
                    "8 * scvelo.plotting.utils.default_size"
                    f"(n={EXPECTED_N_VISIBLE})"
                ),
                "scatter_vector_friendly": True,
                "font_family_resolved_from_oracle": "DejaVu Sans",
                "oracle_title_glyph_prefix": "DejaVuSans-",
                "cell_type_order": list(DISPLAY_CATEGORIES),
                "cell_type_colors": [palette[label] for label in DISPLAY_CATEGORIES],
                "full_sidecar_title": sidecar_title,
                "full_sidecar_legend_loc": "right",
                "component_sidecars_have_title_or_legend": False,
                "final_ai_assembly_uses_legacy_legend": False,
                "savefig": {"format": "svg", "dpi": 300, "bbox_inches": "tight"},
                "oracle_geometry": oracle_geometry,
            },
            "corrected_ondata_label_contract": {
                "renderer_grammar": "scVelo 0.3.3 legend_loc='on data'",
                "placement_statistic": "coordinate-wise median in raw corrected PCA",
                "all17_sidecar": ONDATA_ALL17_SVG_NAME,
                "historical_final_ai_12_sidecar": ONDATA_PAPER12_SVG_NAME,
                "historical_final_ai_12_roster": list(HISTORICAL_FINAL_AI_LABELS),
                "position_table": ONDATA_POSITIONS_CSV_NAME,
                "positions": ondata_rows,
                "warning": (
                    "The old fixed Illustrator label coordinates are not reused; "
                    "raw corrected PCA materially differs from the paper PCA."
                ),
            },
            "manual_arrow_direction_evidence": {
                "grid_npz": GRID_NPZ_NAME,
                "grid_csv": GRID_CSV_NAME,
                "n_grid_rows": grid_direction_rows,
                "coordinate_system": "raw corrected PCA",
                "direction_columns": [
                    "unit_direction_x",
                    "unit_direction_y",
                    "direction_degrees_ccw_from_positive_pc1",
                ],
                "usage": (
                    "Map the historical white-arrow anchor into raw corrected PCA, "
                    "then inspect/interpolate this grid; do not preserve its stale direction."
                ),
                "final_ai_axes_clip_top_pt": list(FINAL_AI_FIG5D_AXES_CLIP_TOP_PT),
                "historical_arrow_axes_top_fractions": {
                    name: list(fractions)
                    for name, fractions in FINAL_AI_WHITE_ARROW_AXES_FRACTIONS.items()
                },
                "historical_white_arrow_object_sha256": (
                    FINAL_AI_WHITE_ARROW_OBJECT_SHA256
                ),
                "localized_evidence_csv": MANUAL_ARROW_EVIDENCE_CSV_NAME,
                "localized_samples": manual_arrow_rows,
                "localized_summary": manual_arrow_summary,
            },
            "inputs": {
                "state_npz": {
                    "path": str(state_path),
                    "sha256": input_hashes["state_npz"],
                },
                "state_manifest": {
                    "path": str(state_manifest_path),
                    "sha256": input_hashes["state_manifest"],
                },
                "palette_json": {
                    "path": str(palette_path),
                    "sha256": input_hashes["palette_json"],
                },
                "oracle_svg": {
                    "path": str(oracle_path),
                    "sha256": input_hashes["oracle_svg"],
                    "role": "style/layout oracle only; no coordinates or velocities are read",
                },
                "final_ai": {
                    "path": str(final_ai_path),
                    "sha256": input_hashes["final_ai"],
                    "role": (
                        "object-level axes/white-arrow annotation geometry oracle only; "
                        "no scientific coordinates or velocities are read"
                    ),
                },
            },
            "object_qa": {
                "historical_oracle": oracle_stats,
                "corrected_full": full_stats,
                "corrected_stream_only": stream_stats,
                "corrected_scatter_only": scatter_stats,
                "corrected_ondata_labels_all17_only": ondata_all17_stats,
                "corrected_ondata_labels_paper12_only": ondata_paper12_stats,
            },
            "outputs": {},
        }
        for path in (
            full_svg,
            stream_svg,
            scatter_svg,
            ondata_all17_svg,
            ondata_paper12_svg,
            ondata_positions_csv,
            grid_path,
            grid_csv_path,
            manual_arrow_evidence_csv,
        ):
            manifest["outputs"][path.name] = {
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
        manifest_path = stage / "render_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        stage.rename(output_dir)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise

    print(f"Output: {output_dir}")
    print(json.dumps(manifest["scientific_contract"], indent=2))
    print(json.dumps(manifest["object_qa"]["corrected_full"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
