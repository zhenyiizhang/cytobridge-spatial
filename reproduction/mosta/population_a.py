"""Figure 4a population rendering in the paper layout."""
from __future__ import annotations

import hashlib

import argparse

import json

import sys

from pathlib import Path

import anndata as ad

import fitz

import matplotlib as mpl

import matplotlib.pyplot as plt

import numpy as np

from PIL import Image, ImageDraw

CROP = fitz.Rect(0.0, 0.0, 595.2760009765625, 192.0)

POINT_SIZE_PT2 = 0.35

LEGACY_AI_ORDER = [
    "Blood vessel", "Brain", "Cartilage primordium", "Cavity", "Choroid plexus",
    "Connective tissue", "Dorsal root ganglion", "Epidermis", "GI tract", "Heart",
    "Jaw and tooth", "Kidney", "Liver", "Lung", "Lung primordium", "Meninges",
    "Mesentery", "Mesothelium", "Mucosal epithelium", "Muscle", "Ovary", "Pancreas",
    "Spinal cord", "Sympathetic nerve", "Urogenital ridge",
]

LEGACY_AI_PALETTE = {
    "Blood vessel": "#b3a726ff",
    "Brain": "#ef833aff",
    "Cartilage primordium": "#3cb44bff",
    "Cavity": "#dfdce0ff",
    "Choroid plexus": "#bd3addff",
    "Connective tissue": "#0bd3b1ff",
    "Dorsal root ganglion": "#b74c11ff",
    "Epidermis": "#036df4ff",
    "GI tract": "#5c5ca6ff",
    "Heart": "#d3245aff",
    "Jaw and tooth": "#f062f9ff",
    "Kidney": "#62cfe8ff",
    "Liver": "#c923b1ff",
    "Lung": "#7dc243ff",
    "Lung primordium": "#7ec136ff",
    "Meninges": "#dfca43ff",
    "Mesentery": "#e71d36ff",
    "Mesothelium": "#ff8383ff",
    "Mucosal epithelium": "#2f7dd1ff",
    "Muscle": "#af1041ff",
    "Ovary": "#55afd9ff",
    "Pancreas": "#739b1eff",
    "Spinal cord": "#f9d5baff",
    "Sympathetic nerve": "#cc5a0dff",
    "Urogenital ridge": "#887ab8ff",
}

PANELS = [
    {"time": 0.0, "file": "time_0.h5ad", "xref": 39, "width": 281, "height": 375, "bbox": [30.7846012, 64.1207886, 98.2245941, 154.1207886], "origin": "observed_real", "anchor": 0.0},
    {"time": 0.5, "file": "time_0p5.h5ad", "xref": 169, "width": 609, "height": 890, "bbox": [83.6306992, 64.1207886, 149.0263367, 159.6907654], "origin": "generated_global_t0", "anchor": 0.0},
    {"time": 1.0, "file": "time_1.h5ad", "xref": 40, "width": 314, "height": 397, "bbox": [144.2082825, 70.2987366, 219.5682831, 165.5787354], "origin": "observed_real", "anchor": 1.0},
    {"time": 1.5, "file": "time_1p5.h5ad", "xref": 172, "width": 645, "height": 890, "bbox": [195.4359283, 68.3892822, 261.9849243, 160.2165985], "origin": "generated_global_t0", "anchor": 0.0},
    {"time": 2.0, "file": "time_2.h5ad", "xref": 41, "width": 284, "height": 386, "bbox": [259.8041382, 71.6187897, 327.9641418, 164.2587891], "origin": "observed_real", "anchor": 2.0},
    {"time": 2.5, "file": "time_2p5.h5ad", "xref": 175, "width": 595, "height": 890, "bbox": [316.9075928, 66.5596924, 379.2486877, 159.8093872], "origin": "generated_global_t0", "anchor": 0.0},
    {"time": 3.0, "file": "time_3.h5ad", "xref": 42, "width": 273, "height": 405, "bbox": [380.5755310, 68.3892975, 446.0955200, 165.5892944], "origin": "observed_real", "anchor": 3.0},
]

def rgba_to_rgb01(value: str) -> tuple[float, float, float]:
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) / 255.0 for i in (0, 2, 4))

def prepare_render_panels(input_dir: Path) -> list[dict]:
    """Keep original placements except an isotropic E15 occupancy correction.

    The original E15 raster box has aspect 0.669 while the corrected raw E15
    coordinate extent is about 0.724.  Fitting that wider state into the old
    box made it look shorter than E14/E14.5.  We retain the original height,
    expand the point-cloud object symmetrically in x, and render on a matching
    canvas.  This changes only object scale/placement inside the dashed panel;
    it does not change, crop, rotate, or anisotropically stretch coordinates.
    """
    prepared: list[dict] = []
    for original in PANELS:
        panel = dict(original)
        panel["render_width"] = int(original["width"])
        panel["render_height"] = int(original["height"])
        panel["placement_bbox"] = list(original["bbox"])
        panel["placement_mode"] = "exact_original_ai_bbox"
        if float(panel["time"]) == 2.5:
            a = ad.read_h5ad(input_dir / panel["file"], backed="r")
            xy = np.asarray(a.obsm["spatial"])
            a.file.close()
            extent = np.ptp(xy, axis=0)
            raw_aspect = float(extent[0] / extent[1])
            x0, y0, x1, y1 = map(float, original["bbox"])
            height_pt = y1 - y0
            width_pt = height_pt * raw_aspect
            center_x = 0.5 * (x0 + x1)
            panel["placement_bbox"] = [center_x - 0.5 * width_pt, y0, center_x + 0.5 * width_pt, y1]
            panel["render_height"] = int(original["height"])
            panel["render_width"] = int(round(panel["render_height"] * raw_aspect))
            panel["placement_mode"] = "isotropic_expand_x_to_corrected_raw_aspect"
            panel["raw_coordinate_aspect"] = raw_aspect
        prepared.append(panel)
    return prepared

def update_original_form_placement(
    doc: fitz.Document,
    page: fitz.Page,
    form_xref: int,
    bbox: list[float],
) -> None:
    """Update the existing Illustrator Form XObject matrix in place.

    This preserves the original content-stream order, unlike inserting a new
    overlay image, and therefore keeps E15 behind the E15.5 slice.
    """
    x0, y0, x1, y1 = map(float, bbox)
    width = x1 - x0
    height = y1 - y0
    pdf_y_top = float(page.rect.height) - y0
    pdf_y_bottom = float(page.rect.height) - y1
    stream = (
        "q\n"
        "/GS0 gs\n"
        "/Perceptual ri\n"
        f"{width:.7f} 0 0 {-height:.7f} {x0:.7f} {pdf_y_top:.7f} cm\n"
        "/Im0 Do\n"
        "Q\n"
    ).encode("ascii")
    doc.update_stream(form_xref, stream)
    doc.xref_set_key(
        form_xref,
        "BBox",
        f"[{x0:.7f} {pdf_y_top:.7f} {x1:.7f} {pdf_y_bottom:.7f}]",
    )

def render_transparent_points(
    input_dir: Path,
    panel: dict,
    palette: dict[str, str],
    ordered_labels: list[str],
    out: Path,
) -> None:
    a = ad.read_h5ad(input_dir / panel["file"])
    if str(a.uns.get("slice_origin")) != panel["origin"]:
        raise RuntimeError(f"Origin mismatch for {panel['file']}")
    if float(a.uns.get("source_anchor_time")) != float(panel["anchor"]):
        raise RuntimeError(f"Anchor mismatch for {panel['file']}")
    xy = np.asarray(a.obsm["spatial"])
    if not np.array_equal(np.asarray(a.X)[:, :2], xy):
        raise RuntimeError(f"Spatial basis mismatch for {panel['file']}")
    labels = a.obs["Annotation"].astype(str).to_numpy()

    bbox = panel["placement_bbox"]
    bbox_width_pt = float(bbox[2] - bbox[0])
    effective_dpi = float(panel["render_width"]) * 72.0 / bbox_width_pt
    mpl.rcParams.update({"figure.facecolor": "none", "savefig.facecolor": "none"})
    fig = plt.figure(
        figsize=(float(panel["render_width"]) / effective_dpi, float(panel["render_height"]) / effective_dpi),
        dpi=effective_dpi,
        facecolor="none",
    )
    ax = fig.add_axes([0, 0, 1, 1], facecolor="none")
    for label in ordered_labels:
        mask = labels == label
        if np.any(mask):
            ax.scatter(
                xy[mask, 0],
                xy[mask, 1],
                s=POINT_SIZE_PT2,
                marker="o",
                c=palette[label],
                alpha=0.9,
                linewidths=0,
                edgecolors="none",
                rasterized=False,
            )
    x_min, y_min = xy.min(axis=0)
    x_max, y_max = xy.max(axis=0)
    x_pad = max(0.008 * (x_max - x_min), 1e-6)
    y_pad = max(0.008 * (y_max - y_min), 1e-6)
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    fig.savefig(out, dpi=effective_dpi, transparent=True, bbox_inches=None, pad_inches=0)
    plt.close(fig)
    with Image.open(out) as image:
        if image.size != (int(panel["render_width"]), int(panel["render_height"])):
            raise RuntimeError(f"Replacement size mismatch for {panel['file']}: {image.size}")

def replace_legend(page: fitz.Page, palette: dict[str, str], ordered_labels: list[str]) -> None:
    # The Illustrator legend starts at x=498.58 with 3.3-pt circular markers
    # and a single text column at x=510.4.  Only vertical density changes to
    # accommodate the complete corrected label roster.
    page.draw_rect(fitz.Rect(492.0, 4.5, CROP.x1, CROP.y1), color=None, fill=(1, 1, 1), overlay=True)
    fontname = "ArialMOSTA"
    page.insert_font(fontname=fontname, fontfile=str(ARIAL))
    ys = np.linspace(12.95, 187.0, len(ordered_labels))
    for y, label in zip(ys, ordered_labels):
        color = rgba_to_rgb01(palette[label])
        page.draw_circle(fitz.Point(500.23, float(y)), 1.65, color=color, fill=color, width=0.1, overlay=True)
        page.insert_text(
            fitz.Point(510.4, float(y) + 1.55),
            label,
            fontname=fontname,
            fontsize=4.35,
            color=(0, 0, 0),
            overlay=True,
        )

from matplotlib.font_manager import findfont
ARIAL = Path(findfont("Arial", fallback_to_default=False))
ARIAL_FONT = str(ARIAL)
