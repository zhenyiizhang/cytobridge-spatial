"""Figure 4b population rendering in the paper layout."""
from __future__ import annotations

import hashlib

import json

from pathlib import Path

import sys

import fitz

import matplotlib as mpl

import matplotlib.pyplot as plt

from matplotlib.colors import LinearSegmentedColormap

import numpy as np

import pandas as pd

from PIL import Image, ImageDraw

RENDER_VERSION = "accepted_compute_display50k_per_panel_q1_q99_ai_xrefs_annotated_v6"

NORM_COLUMN = "total_norm_per_panel_q1_q99"

CROP = fitz.Rect(0.0, 201.0, 326.6, 459.5)

POINT_SIZE_PT2 = 3.0

BASE_POINT_SIZE_PT2 = 2.7

ALPHA = 0.95

BASE_ALPHA = 0.35

CMAP_COLORS = (
    "#f8fbfd",
    "#eaf5f4",
    "#d4ece9",
    "#b7dedb",
    "#93c9cb",
    "#66abb4",
)

PANELS = (
    {
        "time": 0.0,
        "stage": "E12.5",
        "xref": 37,
        "width": 907,
        "height": 1343,
        "bbox": [2.1388890743, 252.8696136475, 99.2780151367, 396.7040710449],
        "origin": "observed_real",
        "anchor": 0.0,
    },
    {
        "time": 0.5,
        "stage": "E13.0",
        "xref": 35,
        "width": 925,
        "height": 1343,
        "bbox": [100.6464157104, 249.8218994141, 204.3396301270, 400.3732604980],
        "origin": "generated_global_t0",
        "anchor": 0.0,
    },
    {
        "time": 1.0,
        "stage": "E13.5",
        "xref": 36,
        "width": 962,
        "height": 1343,
        "bbox": [205.9485015869, 249.8938598633, 313.2412109375, 399.6798400879],
        "origin": "observed_real",
        "anchor": 1.0,
    },
)

ANNOTATION_COLOR = (0, 0, 0)

def add_generated_panel_annotations(page: fitz.Page) -> None:
    """Add requested E13.0 CP identity and keep the original Meninges text readable.

    Coordinates are in the original Illustrator page coordinate system.  The
    two dashed ellipses enclose the two Choroid plexus components identified
    directly from the corrected t=0.5 cell-type mapping (237 and 60 cells).
    """
    if not Path(ARIAL_FONT).is_file():
        raise RuntimeError(f"Arial style font is missing: {ARIAL_FONT}")

    # Match the original 9-pt Arial / #66abb4 annotation grammar.  The label is
    # positioned above the two corrected Choroid plexus components so it does
    # not collide with the legacy paired Meninges arrows.
    page.insert_text(
        fitz.Point(126.0, 267.0),
        "Choroid Plexus",
        fontsize=9.0,
        fontname="ArialFig4bOverlay",
        fontfile=ARIAL_FONT,
        color=ANNOTATION_COLOR,
        overlay=True,
    )
    page.draw_oval(
        fitz.Rect(111.5, 267.0, 129.5, 300.5),
        color=ANNOTATION_COLOR,
        dashes="[4 4] 0",
        width=1.0,
        overlay=True,
    )
    page.draw_oval(
        fitz.Rect(170.5, 276.0, 186.0, 294.5),
        color=ANNOTATION_COLOR,
        dashes="[4 4] 0",
        width=1.0,
        overlay=True,
    )

    # The corrected Jaw-and-tooth hotspot underneath the legacy E13.0 label is
    # darker than in the old raster.  Redraw the same text at the exact legacy
    # baseline with a narrow white stroke, preserving its cyan fill and position.
    meninges_origin = fitz.Point(128.42039489746094, 328.43780517578125)
    for dx, dy in (
        (-0.30, 0.0),
        (0.30, 0.0),
        (0.0, -0.30),
        (0.0, 0.30),
        (-0.22, -0.22),
        (-0.22, 0.22),
        (0.22, -0.22),
        (0.22, 0.22),
    ):
        page.insert_text(
            meninges_origin + (dx, dy),
            "Meninges",
            fontsize=9.0,
            fontname="ArialFig4bOverlay",
            fontfile=ARIAL_FONT,
            color=(1.0, 1.0, 1.0),
            overlay=True,
        )
    page.insert_text(
        meninges_origin,
        "Meninges",
        fontsize=9.0,
        fontname="ArialFig4bOverlay",
        fontfile=ARIAL_FONT,
        color=ANNOTATION_COLOR,
        overlay=True,
    )

def render_panel(table: pd.DataFrame, panel: dict[str, object], out: Path) -> dict[str, object]:
    values = table[NORM_COLUMN].to_numpy(dtype=np.float64)
    available = table["cell_type_score_available"].to_numpy(dtype=bool)
    xy = table[["x", "y"]].to_numpy(dtype=np.float64)
    if len(table) == 0 or not np.isfinite(xy).all():
        raise RuntimeError(f"Invalid render table for {panel['stage']}.")
    if not np.isfinite(values[available]).all() or not np.isnan(values[~available]).all():
        raise RuntimeError(f"Available/unavailable score contract failed for {panel['stage']}.")
    if np.any(values[available] < 0.0) or np.any(values[available] > 1.0):
        raise RuntimeError(f"Normalized hotspot values outside [0,1] for {panel['stage']}.")
    cmap = LinearSegmentedColormap.from_list("incoming_pastel", CMAP_COLORS)
    width = int(panel["width"])
    height = int(panel["height"])
    dpi = 500.0
    mpl.rcParams.update({"figure.facecolor": "white", "savefig.facecolor": "white"})
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi, facecolor="white")
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], facecolor="white")
    ax.scatter(
        xy[:, 0],
        xy[:, 1],
        c="#edf2f4",
        s=BASE_POINT_SIZE_PT2,
        alpha=BASE_ALPHA,
        edgecolors="none",
        linewidths=0,
        rasterized=False,
    )
    ax.scatter(
        xy[available, 0],
        xy[available, 1],
        c=values[available],
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        s=POINT_SIZE_PT2,
        alpha=ALPHA,
        edgecolors="none",
        linewidths=0,
        rasterized=False,
    )
    extent = np.ptp(xy, axis=0)
    x_pad = max(float(extent[0]) * 0.008, 1e-9)
    y_pad = max(float(extent[1]) * 0.008, 1e-9)
    ax.set_xlim(float(xy[:, 0].min()) - x_pad, float(xy[:, 0].max()) + x_pad)
    ax.set_ylim(float(xy[:, 1].min()) - y_pad, float(xy[:, 1].max()) + y_pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    fig.savefig(out, dpi=dpi, bbox_inches=None, pad_inches=0, facecolor="white")
    plt.close(fig)
    return {"time": float(panel["time"]), "cells": len(table)}

from matplotlib.font_manager import findfont
ARIAL = Path(findfont("Arial", fallback_to_default=False))
ARIAL_FONT = str(ARIAL)
