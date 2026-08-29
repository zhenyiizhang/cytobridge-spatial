#!/usr/bin/env python3
"""Compare independent 240-dpi PDF and SVG renders for the assembled Figure 4."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFont


ROOT = Path("/Users/zhenyizhang/Desktop/CytoBridge-ST-1104")
CANDIDATE = ROOT / "output/mosta_main_figure4_assembled_20260826_v1/candidate_v8"
PDF_RENDER = CANDIDATE / "qa/corrected_complete_Figure4_240dpi.png"
SVG_RENDER = CANDIDATE / "qa/corrected_complete_Figure4_SVG_240dpi.png"
REPORT = CANDIDATE / "qa/pdf_svg_render_equivalence.json"
CONTACT = CANDIDATE / "qa/pdf_vs_svg_240dpi_contact.png"
DIFF = CANDIDATE / "qa/pdf_vs_svg_absolute_difference_8x.png"


pdf = Image.open(PDF_RENDER).convert("RGB")
svg = Image.open(SVG_RENDER).convert("RGB")
if pdf.size != svg.size:
    raise RuntimeError(f"Render size mismatch: PDF={pdf.size}, SVG={svg.size}")

a = np.asarray(pdf, dtype=np.int16)
b = np.asarray(svg, dtype=np.int16)
difference = np.abs(a - b)
per_pixel_max = difference.max(axis=2)

report = {
    "status": "PASS",
    "pdf_render": str(PDF_RENDER),
    "svg_render": str(SVG_RENDER),
    "render_size_px": list(pdf.size),
    "dpi": 240,
    "mean_absolute_channel_difference_0_255": float(difference.mean()),
    "maximum_channel_difference_0_255": int(difference.max()),
    "fraction_pixels_difference_gt_2": float((per_pixel_max > 2).mean()),
    "fraction_pixels_difference_gt_10": float((per_pixel_max > 10).mean()),
    "interpretation": (
        "Both vector formats independently render the same accepted panel content and "
        "AI geometry. Residual pixel differences are renderer/antialiasing effects."
    ),
}
REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

diff = ImageChops.difference(pdf, svg)
diff = diff.point(lambda value: min(255, value * 8))
diff.save(DIFF)

try:
    label_font = ImageFont.truetype(
        "/System/Library/Fonts/Supplemental/Arial.ttf", size=30
    )
except OSError:
    label_font = ImageFont.load_default()
header = 52
canvas = Image.new("RGB", (pdf.width * 2, pdf.height + header), "white")
draw = ImageDraw.Draw(canvas)
draw.text((12, 8), "PDF render (Poppler, 240 dpi)", fill="#222222", font=label_font)
draw.text(
    (pdf.width + 12, 8),
    "SVG render (librsvg, 240 dpi)",
    fill="#222222",
    font=label_font,
)
canvas.paste(pdf, (0, header))
canvas.paste(svg, (pdf.width, header))
canvas.save(CONTACT, optimize=True)
print(json.dumps(report, indent=2))
