#!/usr/bin/env python3
"""Assemble corrected ARISTA Figure 5c in the frozen final-Illustrator layout.

Only two scientific layers are changed:

* the historical inline scVelo stream paths are replaced by paths rendered
  from the accepted corrected velocity state; and
* the 1,454 historical ROI marker Form XObjects keep their exact geometry but
  receive the corrected full-spatial-versus-interaction-spatial cosine colors.

The tissue raster, panel titles, coordinate glyph, ROI boxes and connectors,
colorbar, typography, and every object outside panel c are retained from the
locked ``Arista.ai`` oracle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import tempfile
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pymupdf as fitz
import numpy as np
from PIL import Image


DEFAULT_VECTOR_TEMPLATE = Path("/Users/zhenyizhang/Desktop/202511/CytoBRIDGE/Arista.ai")
DEFAULT_SPATIAL_REFERENCE = Path(
    "results/arista_mosta_velocity/scvelo_streams/velocity_scvelo_spatial_full_t1.svg"
)
DEFAULT_ROI_REFERENCE = Path(
    "results/arista_mosta_velocity/velocity_spatial_direction_correlation_roi_t1_scvelo_only.svg"
)

EXPECTED_TEMPLATE_SHA256 = "673dc81f4856833c30c943ad5f2f4af9e69f771cea4cb63f2484ffbd18907694"
EXPECTED_SPATIAL_REFERENCE_SHA256 = "54b9634794cbf413304b5bbe4246f1b733c421b78b4ef53307e081c2d39a0e7c"
EXPECTED_ROI_REFERENCE_SHA256 = "c144a6b4ca626d8d10fcce20e6e30f62afc681a337974284d7f2e99d3ec87238"
EXPECTED_STREAMS_SHA256 = "efe8bc1788c8c27f68bd4222ee1e81e8d7253c7648bc3f9fbabdea699f7b2df8"
EXPECTED_CORRECTED_ROI_SHA256 = "547ecada2f4d90a9cada1c4ae7f81aa7f2fd6b3435de4a2c934aeea17f007501"
EXPECTED_RENDER_MANIFEST_SHA256 = "c8a1429cf32c2c38433acdd622cc700305ee8888bb2023cffff7956be2966a73"
EXPECTED_STREAM_LINE_COUNT = 5693
EXPECTED_STREAM_ARROW_COUNT = 149

PANEL_CROP = fitz.Rect(205.5, 437.0, 595.276, 632.0)

# Measured from the final Illustrator artwork.  Coordinates are expressed in
# the SVG/top-left convention.  Stroke widths were not scaled by Illustrator.
OLD_SVG_TO_AI_X_SCALE = 0.540765126605
OLD_SVG_TO_AI_Y_SCALE = 0.524257992628
OLD_SVG_TO_AI_X_OFFSET_TOP = 225.818321258
OLD_SVG_TO_AI_Y_OFFSET_TOP = 450.153864023

OLD_ROI_FIRST_FORM = 332
OLD_ROI_LAST_FORM = 1785
ROI_MARKER_COUNT = OLD_ROI_LAST_FORM - OLD_ROI_FIRST_FORM + 1
HISTORICAL_ROI_ALPHA = 0.949997

SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"
FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
LINE_PATH_RE = re.compile(
    rf"^\s*M\s+({FLOAT})\s+({FLOAT})\s+L\s+({FLOAT})\s+({FLOAT})\s*$"
)
QUAD_PATH_RE = re.compile(
    rf"^\s*M\s+({FLOAT})\s+({FLOAT})\s+Q\s+({FLOAT})\s+({FLOAT})\s+({FLOAT})\s+({FLOAT})\s*$"
)
HEAD_PATH_RE = re.compile(
    rf"^\s*M\s+({FLOAT})\s+({FLOAT})\s+L\s+({FLOAT})\s+({FLOAT})\s+L\s+({FLOAT})\s+({FLOAT})\s+z\s*$",
    re.IGNORECASE,
)

FM327_CALL_RE = re.compile(
    rb"q\n0 841\.89 595\.276 -841\.89 re\nW n\nq\n"
    rb"/Perceptual ri\n/GS0 gs\n0 TL/Fm327 Do\nQ\nQ\n"
)
FM328_CALL_RE = re.compile(
    rb"q\n0 841\.89 595\.276 -841\.89 re\nW n\nq\n"
    rb"0 g\n0 G\n1 w 10 M 0 j 0 J \n/GS2 gs\n0 TL/Fm328 Do\nQ\nQ\n"
)
INLINE_BEGIN = b"% CYTOBRIDGE_FIG5C_CORRECTED_SCVELO_BEGIN\n"
INLINE_END = b"% CYTOBRIDGE_FIG5C_CORRECTED_SCVELO_END\n"
POST_STREAM_GRAPHICS_STATE = (
    "/CS0 CS 0 0 0 SCN\n"
    "/CS0 cs 0 0 0 scn\n"
    "0.514 w 4 M 1 j 1 J [] 0 d\n"
    "/Perceptual ri\n"
    "/GS0 gs\n"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_hash(path: Path, expected: str, label: str) -> str:
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(f"{label} SHA256 changed: {actual} != {expected}")
    return actual


def _find_id(root: ET.Element, element_id: str) -> ET.Element:
    for element in root.iter():
        if element.attrib.get("id") == element_id:
            return element
    raise ValueError(f"SVG element #{element_id} is missing")


def _svg_geometry(path: Path) -> dict[str, str]:
    root = ET.parse(path).getroot()
    return {
        "width": root.attrib["width"],
        "height": root.attrib["height"],
        "viewBox": root.attrib["viewBox"],
    }


def _clip_rect(path: Path) -> tuple[float, float, float, float]:
    root = ET.parse(path).getroot()
    clips = list(root.iter(f"{{{SVG_NS}}}clipPath"))
    if len(clips) != 1 or len(list(clips[0])) != 1:
        raise ValueError(f"Expected one rectangular SVG clip in {path}")
    rectangle = list(clips[0])[0]
    if rectangle.tag != f"{{{SVG_NS}}}rect":
        raise ValueError(f"Expected a rectangular SVG clip in {path}")
    return tuple(float(rectangle.attrib[key]) for key in ("x", "y", "width", "height"))


def _style_width(style: str) -> float:
    match = re.search(r"(?:^|;)\s*stroke-width:\s*(%s)" % FLOAT, style)
    if not match:
        raise ValueError(f"Stream path is missing a stroke width: {style}")
    if "stroke: #000000" not in style:
        raise ValueError(f"Stream path is no longer black: {style}")
    return float(match.group(1))


def _fmt(value: float, digits: int = 4) -> str:
    if abs(value) < 0.5 * 10 ** (-digits):
        value = 0.0
    rendered = f"{value:.{digits}f}".rstrip("0").rstrip(".")
    return "0" if rendered in {"", "-0"} else rendered


def _main_content_stream(page: fitz.Page) -> int:
    candidates = [(len(page.parent.xref_stream(xref)), xref) for xref in page.get_contents()]
    if not candidates:
        raise RuntimeError("Illustrator page has no content stream")
    return max(candidates)[1]


def _xobject_map(document: fitz.Document, page: fitz.Page) -> dict[int, int]:
    kind, value = document.xref_get_key(page.xref, "Resources/XObject")
    if kind != "dict":
        raise ValueError("Page XObject resource dictionary is missing")
    rows = re.findall(r"/Fm(\d+)\s+(\d+)\s+(\d+)\s+R", value)
    mapping = {int(name): int(xref) for name, xref, _generation in rows}
    if len(mapping) != len(rows):
        raise ValueError("Duplicate Illustrator Form resource names detected")
    return mapping


def _extract_old_stream_block(stream: bytes) -> tuple[int, int, bytes]:
    left = list(FM327_CALL_RE.finditer(stream))
    if len(left) != 1:
        raise ValueError(f"Expected one final-AI Fm327 call; found {len(left)}")
    start = left[0].end()
    right = FM328_CALL_RE.search(stream, start)
    if right is None:
        raise ValueError("Final-AI Fm328 sentinel is missing")
    block = stream[start : right.start()]
    if len(block) < 400_000 or block.count(b"\nS\n") < 7_000:
        raise ValueError("Historical Figure 5c inline stream block changed")
    return start, right.start(), block


def _transform_factory(
    *,
    new_clip: tuple[float, float, float, float],
    old_clip: tuple[float, float, float, float],
    page_height: float,
):
    nx, ny, nw, nh = new_clip
    ox, oy, ow, oh = old_clip
    if not np.allclose([nw, nh], [ow, oh], rtol=0.0, atol=1e-9):
        raise ValueError(f"Historical/current scVelo axes sizes differ: {new_clip} vs {old_clip}")
    x_shift = ox - nx
    y_shift = oy - ny

    def transform(x: float, y: float) -> tuple[float, float]:
        old_x = float(x) + x_shift
        old_y = float(y) + y_shift
        ai_x = OLD_SVG_TO_AI_X_SCALE * old_x + OLD_SVG_TO_AI_X_OFFSET_TOP
        ai_y_top = OLD_SVG_TO_AI_Y_SCALE * old_y + OLD_SVG_TO_AI_Y_OFFSET_TOP
        return ai_x, float(page_height) - ai_y_top

    return transform, x_shift, y_shift


def _build_corrected_inline_streams(
    *,
    streams_svg: Path,
    spatial_reference_svg: Path,
    page_height: float,
) -> tuple[bytes, dict[str, Any]]:
    root = ET.parse(streams_svg).getroot()
    axes = _find_id(root, "axes_1")
    collection = _find_id(root, "LineCollection_1")
    line_paths = list(collection.iter(f"{{{SVG_NS}}}path"))
    arrow_groups = [
        child for child in axes if child.attrib.get("id", "").startswith("patch_")
    ]
    if (
        len(line_paths) != EXPECTED_STREAM_LINE_COUNT
        or len(arrow_groups) != EXPECTED_STREAM_ARROW_COUNT
    ):
        raise ValueError(
            f"Corrected scVelo topology changed: {len(line_paths)} lines, "
            f"{len(arrow_groups)} arrows"
        )

    new_clip = _clip_rect(streams_svg)
    old_clip = _clip_rect(spatial_reference_svg)
    transform, x_shift, y_shift = _transform_factory(
        new_clip=new_clip,
        old_clip=old_clip,
        page_height=page_height,
    )

    old_x, old_y, old_w, old_h = old_clip
    clip_top_x0 = OLD_SVG_TO_AI_X_SCALE * old_x + OLD_SVG_TO_AI_X_OFFSET_TOP
    clip_top_x1 = OLD_SVG_TO_AI_X_SCALE * (old_x + old_w) + OLD_SVG_TO_AI_X_OFFSET_TOP
    clip_top_y0 = OLD_SVG_TO_AI_Y_SCALE * old_y + OLD_SVG_TO_AI_Y_OFFSET_TOP
    clip_top_y1 = OLD_SVG_TO_AI_Y_SCALE * (old_y + old_h) + OLD_SVG_TO_AI_Y_OFFSET_TOP
    # The original final panel ends at y=632 pt.  Corrected paths can touch the
    # full Matplotlib clip at ~634 pt, so intersect with the immutable panel
    # boundary to guarantee that no neighboring panel is modified.
    clip_top_x0 = max(clip_top_x0, PANEL_CROP.x0)
    clip_top_x1 = min(clip_top_x1, PANEL_CROP.x1)
    clip_top_y0 = max(clip_top_y0, PANEL_CROP.y0)
    clip_top_y1 = min(clip_top_y1, PANEL_CROP.y1)
    clip_pdf_x = clip_top_x0
    clip_pdf_y = page_height - clip_top_y1
    clip_pdf_width = clip_top_x1 - clip_top_x0
    clip_pdf_height = clip_top_y1 - clip_top_y0

    chunks: list[str] = [
        INLINE_BEGIN.decode("ascii"),
        "q\n",
        f"{_fmt(clip_pdf_x)} {_fmt(clip_pdf_y)} {_fmt(clip_pdf_width)} {_fmt(clip_pdf_height)} re\n",
        "W n\n",
        "/CS0 CS 0 0 0 SCN\n",
        "/CS0 cs 0 0 0 scn\n",
        "/Perceptual ri\n",
        "/GS0 gs\n",
    ]

    widths: list[float] = []
    for path in line_paths:
        match = LINE_PATH_RE.fullmatch(path.attrib["d"])
        if match is None:
            raise ValueError(f"Unexpected LineCollection path grammar: {path.attrib['d']}")
        x0, y0, x1, y1 = (float(value) for value in match.groups())
        p0 = transform(x0, y0)
        p1 = transform(x1, y1)
        width = _style_width(path.attrib.get("style", ""))
        widths.append(width)
        chunks.extend(
            [
                f"{_fmt(width, 3)} w 4 M 1 j 0 J [] 0 d\n",
                f"{_fmt(p0[0])} {_fmt(p0[1])} m\n",
                f"{_fmt(p1[0])} {_fmt(p1[1])} l\n",
                "S\n",
            ]
        )

    arrow_path_count = 0
    for group in arrow_groups:
        paths = list(group.iter(f"{{{SVG_NS}}}path"))
        if len(paths) != 2:
            raise ValueError(f"Arrow group {group.attrib.get('id')} has {len(paths)} paths")
        shaft, head = paths
        shaft_match = QUAD_PATH_RE.fullmatch(shaft.attrib["d"])
        head_match = HEAD_PATH_RE.fullmatch(head.attrib["d"])
        if shaft_match is None or head_match is None:
            raise ValueError(f"Unexpected arrow grammar in {group.attrib.get('id')}")
        shaft_width = _style_width(shaft.attrib.get("style", ""))
        head_width = _style_width(head.attrib.get("style", ""))
        if abs(shaft_width - head_width) > 1e-12:
            raise ValueError(f"Arrow shaft/head widths differ in {group.attrib.get('id')}")
        if "stroke-linecap: round" not in shaft.attrib.get("style", ""):
            raise ValueError("Corrected arrow shaft lost round linecaps")
        if "stroke-linecap: round" not in head.attrib.get("style", ""):
            raise ValueError("Corrected arrow head lost round linecaps")
        widths.append(shaft_width)

        sx0, sy0, sqx, sqy, sx2, sy2 = (float(value) for value in shaft_match.groups())
        raw_p0 = np.asarray([sx0, sy0], dtype=float)
        raw_q = np.asarray([sqx, sqy], dtype=float)
        raw_p2 = np.asarray([sx2, sy2], dtype=float)
        raw_c1 = raw_p0 + (2.0 / 3.0) * (raw_q - raw_p0)
        raw_c2 = raw_p2 + (2.0 / 3.0) * (raw_q - raw_p2)
        p0 = transform(*raw_p0)
        c1 = transform(*raw_c1)
        c2 = transform(*raw_c2)
        p2 = transform(*raw_p2)
        chunks.extend(
            [
                f"{_fmt(shaft_width, 3)} w 4 M 1 j 1 J [] 0 d\n",
                f"{_fmt(p0[0])} {_fmt(p0[1])} m\n",
                f"{_fmt(c1[0])} {_fmt(c1[1])} {_fmt(c2[0])} {_fmt(c2[1])} "
                f"{_fmt(p2[0])} {_fmt(p2[1])} c\n",
                "S\n",
            ]
        )

        hx0, hy0, hx1, hy1, hx2, hy2 = (float(value) for value in head_match.groups())
        hp0 = transform(hx0, hy0)
        hp1 = transform(hx1, hy1)
        hp2 = transform(hx2, hy2)
        head_geometry = (
            f"{_fmt(hp0[0])} {_fmt(hp0[1])} m\n"
            f"{_fmt(hp1[0])} {_fmt(hp1[1])} l\n"
            f"{_fmt(hp2[0])} {_fmt(hp2[1])} l\n"
            "h\n"
        )
        chunks.extend(
            [
                "q\n",
                head_geometry,
                "f\n",
                "Q\n",
                f"{_fmt(head_width, 3)} w 4 M 1 j 1 J [] 0 d\n",
                head_geometry,
                "S\n",
            ]
        )
        arrow_path_count += 2

    # Illustrator's historical inline stream block deliberately leaves this
    # graphics state active.  Later page objects inherit it, so a scoped
    # replacement alone would change objects outside panel c.  Restore the
    # exact historical tail after closing our local clip scope.
    chunks.extend(
        ["Q\n", POST_STREAM_GRAPHICS_STATE, INLINE_END.decode("ascii")]
    )
    payload = "".join(chunks).encode("ascii")
    return payload, {
        "line_path_count": len(line_paths),
        "arrow_group_count": len(arrow_groups),
        "arrow_svg_path_count": arrow_path_count,
        "stroke_width_source_min_pt": float(min(widths)),
        "stroke_width_source_max_pt": float(max(widths)),
        "stroke_width_pdf_rounding_decimals": 3,
        "new_svg_clip": list(new_clip),
        "old_svg_clip": list(old_clip),
        "new_to_old_svg_translation": [float(x_shift), float(y_shift)],
        "old_svg_to_final_ai_affine_top_coordinates": [
            [OLD_SVG_TO_AI_X_SCALE, 0.0, OLD_SVG_TO_AI_X_OFFSET_TOP],
            [0.0, OLD_SVG_TO_AI_Y_SCALE, OLD_SVG_TO_AI_Y_OFFSET_TOP],
            [0.0, 0.0, 1.0],
        ],
        "final_pdf_clip_rect_pt": [
            float(clip_pdf_x),
            float(clip_pdf_y),
            float(clip_pdf_width),
            float(clip_pdf_height),
        ],
        "inline_stream_sha256": _sha256_bytes(payload),
    }


def _roi_marker_colors(path: Path) -> list[str]:
    root = ET.parse(path).getroot()
    collection = _find_id(root, "PathCollection_1")
    rows: list[str] = []
    for marker in collection.iter(f"{{{SVG_NS}}}use"):
        if "x" not in marker.attrib or "y" not in marker.attrib:
            continue
        style = marker.attrib.get("style", "")
        fill = re.search(r"(?:^|;)\s*fill:\s*(#[0-9a-fA-F]{6})", style)
        alpha = re.search(r"(?:^|;)\s*fill-opacity:\s*(%s)" % FLOAT, style)
        if fill is None or alpha is None:
            raise ValueError(f"Corrected ROI marker style changed: {style}")
        if abs(float(alpha.group(1)) - 0.95) > 1e-12:
            raise ValueError(f"Corrected ROI marker alpha changed: {alpha.group(1)}")
        rows.append(fill.group(1).lower())
    if len(rows) != ROI_MARKER_COUNT:
        raise ValueError(f"Corrected ROI has {len(rows)} markers, expected {ROI_MARKER_COUNT}")
    return rows


def _hex_to_ai_rgb(color: str) -> tuple[float, float, float]:
    return tuple(int(color[index : index + 2], 16) / 255.0 for index in (1, 3, 5))


def _ai_color_line(color: str) -> bytes:
    rgb = _hex_to_ai_rgb(color)
    components = " ".join(_fmt(value, 3) for value in rgb)
    return f"/CS0 cs {components}  scn\n".encode("ascii")


def _replace_roi_form_colors(
    *, document: fitz.Document, page: fitz.Page, corrected_colors: list[str]
) -> dict[str, Any]:
    mapping = _xobject_map(document, page)
    targets = list(range(OLD_ROI_FIRST_FORM, OLD_ROI_LAST_FORM + 1))
    missing = [form_id for form_id in targets if form_id not in mapping]
    if missing:
        raise ValueError(f"Final-AI ROI marker Forms are missing: {missing[:5]}")
    target_xrefs = [mapping[form_id] for form_id in targets]
    if len(set(target_xrefs)) != ROI_MARKER_COUNT:
        raise ValueError("Final-AI ROI marker Forms do not have unique XObjects")

    color_re = re.compile(rb"^/CS0 cs ([0-9.]+) ([0-9.]+) ([0-9.]+)  scn\n")
    before_colors: list[tuple[float, float, float]] = []
    changed = 0
    geometry_tail_hashes_before: list[str] = []
    geometry_tail_hashes_after: list[str] = []
    for xref, corrected_hex in zip(target_xrefs, corrected_colors):
        stream = document.xref_stream(xref)
        match = color_re.match(stream)
        if match is None:
            raise ValueError(f"ROI marker XObject {xref} color grammar changed")
        before_rgb = tuple(float(value) for value in match.groups())
        before_colors.append(before_rgb)
        replacement = _ai_color_line(corrected_hex)
        updated = replacement + stream[match.end() :]
        changed += int(updated != stream)
        geometry_tail_hashes_before.append(_sha256_bytes(stream[match.end() :]))
        geometry_tail_hashes_after.append(_sha256_bytes(updated[len(replacement) :]))
        document.update_stream(xref, updated)
    if geometry_tail_hashes_before != geometry_tail_hashes_after:
        raise AssertionError("ROI marker geometry changed while replacing colors")
    return {
        "form_first": OLD_ROI_FIRST_FORM,
        "form_last": OLD_ROI_LAST_FORM,
        "form_count": ROI_MARKER_COUNT,
        "unique_xref_count": len(set(target_xrefs)),
        "forms_with_changed_color_stream": changed,
        "geometry_tail_hashes_equal": True,
        "historical_first_rgb": list(before_colors[0]),
        "corrected_first_hex": corrected_colors[0],
        "corrected_last_hex": corrected_colors[-1],
    }


def _assemble_fullpage(
    *,
    vector_template: Path,
    streams_svg: Path,
    spatial_reference_svg: Path,
    corrected_roi_svg: Path,
    output_pdf: Path,
) -> dict[str, Any]:
    document = fitz.open(vector_template)
    try:
        if document.page_count != 1:
            raise ValueError("Arista.ai must contain exactly one page")
        page = document[0]
        main_xref = _main_content_stream(page)
        original_stream = document.xref_stream(main_xref)
        start, end, old_block = _extract_old_stream_block(original_stream)
        corrected_inline, stream_stats = _build_corrected_inline_streams(
            streams_svg=streams_svg,
            spatial_reference_svg=spatial_reference_svg,
            page_height=float(page.rect.height),
        )
        updated_stream = original_stream[:start] + corrected_inline + original_stream[end:]
        document.update_stream(main_xref, updated_stream)

        corrected_colors = _roi_marker_colors(corrected_roi_svg)
        roi_stats = _replace_roi_form_colors(
            document=document,
            page=page,
            corrected_colors=corrected_colors,
        )
        document.save(output_pdf, garbage=4, deflate=True)
    finally:
        document.close()
    return {
        "main_content_xref_before_save": main_xref,
        "historical_inline_byte_start": start,
        "historical_inline_byte_end": end,
        "historical_inline_size_bytes": len(old_block),
        "historical_inline_sha256": _sha256_bytes(old_block),
        "corrected_inline_size_bytes": len(corrected_inline),
        **stream_stats,
        "roi_recolor": roi_stats,
    }


def _annotation_spans(page: fitz.Page) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                rows.append(
                    {
                        "text": span["text"],
                        "font": span["font"],
                        "size": round(float(span["size"]), 6),
                        "color": int(span["color"]),
                        "origin": [round(float(value), 6) for value in span["origin"]],
                        "bbox": [round(float(value), 6) for value in span["bbox"]],
                    }
                )
    return rows


def _image_inventory(page: fitz.Page) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for image in page.get_image_info(hashes=True, xrefs=True):
        digest = image.get("digest", b"")
        rows.append(
            {
                "width": int(image["width"]),
                "height": int(image["height"]),
                "bbox": [round(float(value), 6) for value in image["bbox"]],
                "digest": digest.hex() if isinstance(digest, bytes) else str(digest),
            }
        )
    return rows


def _render_page_array(path: Path, scale: float) -> np.ndarray:
    document = fitz.open(path)
    try:
        pixmap = document[0].get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
        return np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
            pixmap.height, pixmap.width, pixmap.n
        )[:, :, :3]
    finally:
        document.close()


def _roi_drawings(
    page: fitz.Page, *, region: fitz.Rect = fitz.Rect(444.0, 483.0, 586.0, 593.0)
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for drawing in page.get_drawings():
        rect = drawing["rect"]
        if drawing.get("fill") is None:
            continue
        if not (1.78 <= rect.width <= 1.82 and 1.78 <= rect.height <= 1.82):
            continue
        if not region.intersects(rect):
            continue
        rows.append(drawing)
    return rows


def _candidate_inline_block(page: fitz.Page) -> bytes:
    stream = page.parent.xref_stream(_main_content_stream(page))
    start = stream.find(INLINE_BEGIN)
    end = stream.find(INLINE_END)
    if start < 0 or end < 0 or end < start:
        raise ValueError("Corrected inline scVelo sentinels are missing after save")
    return stream[start : end + len(INLINE_END)]


def _candidate_roi_colors(document: fitz.Document, page: fitz.Page) -> list[tuple[float, float, float]]:
    mapping = _xobject_map(document, page)
    color_re = re.compile(rb"^/CS0 cs ([0-9.]+) ([0-9.]+) ([0-9.]+)  scn\n")
    rows: list[tuple[float, float, float]] = []
    for form_id in range(OLD_ROI_FIRST_FORM, OLD_ROI_LAST_FORM + 1):
        stream = document.xref_stream(mapping[form_id])
        match = color_re.match(stream)
        if match is None:
            raise ValueError(f"Candidate ROI marker Form Fm{form_id} changed")
        rows.append(tuple(float(value) for value in match.groups()))
    return rows


def _roi_page_alpha(document: fitz.Document, page: fitz.Page) -> tuple[int, float]:
    stream = document.xref_stream(_main_content_stream(page))
    invocation_re = re.compile(
        rb"q\n444\.002 249\.539 141\.138 109\.006 re\nW n\nq\n"
        rb"0 g\n0 G\n/GS3 gs\n0 TL/Fm(\d+) Do\nQ\nQ\n"
    )
    form_ids = [int(match.group(1)) for match in invocation_re.finditer(stream)]
    expected = list(range(OLD_ROI_FIRST_FORM, OLD_ROI_LAST_FORM + 1))
    if form_ids != expected:
        raise ValueError(
            f"Final-AI ROI invocation order changed: {len(form_ids)} calls"
        )
    kind, resources = document.xref_get_key(page.xref, "Resources/ExtGState")
    if kind != "dict":
        raise ValueError("Page ExtGState resources are missing")
    match = re.search(r"/GS3\s+(\d+)\s+(\d+)\s+R", resources)
    if match is None:
        raise ValueError("Page GS3 resource is missing")
    state = document.xref_object(int(match.group(1)), compressed=False)
    alpha_match = re.search(r"/ca\s+([0-9.]+)", state)
    if alpha_match is None:
        raise ValueError("Page GS3 fill alpha is missing")
    return len(form_ids), float(alpha_match.group(1))


def _qa_fullpage(
    *, original_ai: Path, candidate_pdf: Path, corrected_roi_svg: Path, replacement: dict[str, Any]
) -> dict[str, Any]:
    original = fitz.open(original_ai)
    candidate = fitz.open(candidate_pdf)
    try:
        original_page = original[0]
        candidate_page = candidate[0]
        page_size_equal = np.allclose(original_page.rect, candidate_page.rect, atol=1e-6)
        annotations_equal = _annotation_spans(original_page) == _annotation_spans(candidate_page)
        images_equal = _image_inventory(original_page) == _image_inventory(candidate_page)

        inline = _candidate_inline_block(candidate_page)
        inline_sha_equal = _sha256_bytes(inline) == replacement["inline_stream_sha256"]
        inline_line_count = inline.count(b" l\nS\n")
        inline_cubic_count = inline.count(b" c\nS\n")
        inline_head_fill_count = inline.count(b"h\nf\n")
        inline_head_stroke_count = inline.count(b"h\nS\n")

        expected_hex = _roi_marker_colors(corrected_roi_svg)
        expected_rgb = np.asarray(
            [tuple(round(value, 3) for value in _hex_to_ai_rgb(color)) for color in expected_hex],
            dtype=float,
        )
        actual_rgb = np.asarray(_candidate_roi_colors(candidate, candidate_page), dtype=float)
        roi_form_color_max_error = float(np.max(np.abs(actual_rgb - expected_rgb)))
        roi_form_color_mismatch = int(
            np.sum(np.max(np.abs(actual_rgb - expected_rgb), axis=1) > 1e-12)
        )

        marker_drawings = _roi_drawings(candidate_page)
        roi_alpha_call_count, roi_page_alpha = _roi_page_alpha(candidate, candidate_page)
        # PyMuPDF's get_drawings() reports Form-local opacity and does not
        # multiply the calling page's GS3.  The actual rendered alpha is locked
        # by the unchanged 1,454 GS3 invocations checked above.
        marker_local_alphas = sorted(
            {round(float(row.get("fill_opacity", -1.0)), 6) for row in marker_drawings}
        )
        marker_diameters = np.asarray(
            [[row["rect"].width, row["rect"].height] for row in marker_drawings], dtype=float
        )
        marker_no_stroke = all(row.get("color") is None for row in marker_drawings)
    finally:
        original.close()
        candidate.close()

    outside_stats: dict[str, Any] = {}
    outside_pass = True
    for scale in (1.0, 2.0):
        old_pixels = _render_page_array(original_ai, scale=scale)
        new_pixels = _render_page_array(candidate_pdf, scale=scale)
        if old_pixels.shape != new_pixels.shape:
            raise AssertionError("Full-page render dimensions changed")
        outside = np.ones(old_pixels.shape[:2], dtype=bool)
        x0 = int(math.floor(PANEL_CROP.x0 * scale))
        y0 = int(math.floor(PANEL_CROP.y0 * scale))
        x1 = int(math.ceil(PANEL_CROP.x1 * scale))
        y1 = int(math.ceil(PANEL_CROP.y1 * scale))
        outside[y0:y1, x0:x1] = False
        delta = np.abs(new_pixels.astype(np.int16) - old_pixels.astype(np.int16))
        maximum = int(delta[outside].max())
        mae = float(delta[outside].mean())
        outside_stats[f"scale_{scale:g}"] = {"max_abs_error_0_255": maximum, "mae": mae}
        outside_pass &= maximum == 0

    marker_count = len(marker_drawings)
    marker_size_min = marker_diameters.min(axis=0).tolist() if marker_count else []
    marker_size_max = marker_diameters.max(axis=0).tolist() if marker_count else []
    passed = (
        bool(page_size_equal)
        and annotations_equal
        and images_equal
        and inline_sha_equal
        and inline_line_count == replacement["line_path_count"]
        and inline_cubic_count == replacement["arrow_group_count"]
        and inline_head_fill_count == replacement["arrow_group_count"]
        and inline_head_stroke_count == replacement["arrow_group_count"]
        and roi_form_color_mismatch == 0
        and marker_count == ROI_MARKER_COUNT
        and roi_alpha_call_count == ROI_MARKER_COUNT
        and abs(roi_page_alpha - HISTORICAL_ROI_ALPHA) <= 1e-12
        and marker_no_stroke
        and outside_pass
    )
    return {
        "passed": bool(passed),
        "page_size_equal": bool(page_size_equal),
        "all_text_spans_equal": annotations_equal,
        "all_existing_raster_images_equal": images_equal,
        "corrected_inline_sha_equal": inline_sha_equal,
        "corrected_inline_line_count": inline_line_count,
        "corrected_inline_cubic_arrow_count": inline_cubic_count,
        "corrected_inline_arrow_head_fill_count": inline_head_fill_count,
        "corrected_inline_arrow_head_stroke_count": inline_head_stroke_count,
        "roi_form_color_mismatch_count": roi_form_color_mismatch,
        "roi_form_color_max_abs_error": roi_form_color_max_error,
        "roi_vector_marker_count": marker_count,
        "roi_marker_page_alpha_call_count": roi_alpha_call_count,
        "roi_marker_page_fill_opacity": roi_page_alpha,
        "roi_marker_form_local_opacity_values_get_drawings": marker_local_alphas,
        "roi_marker_no_stroke": marker_no_stroke,
        "roi_marker_diameter_min_pt": marker_size_min,
        "roi_marker_diameter_max_pt": marker_size_max,
        "outside_panel": outside_stats,
    }


def _make_physical_panel_pdf(fullpage_pdf: Path, panel_pdf: Path) -> None:
    source = fitz.open(fullpage_pdf)
    try:
        page = source[0]
        page.set_cropbox(page.mediabox)
        full = fitz.Rect(page.rect)
        redactions = [
            fitz.Rect(0.0, 0.0, full.width, PANEL_CROP.y0),
            fitz.Rect(0.0, PANEL_CROP.y1, full.width, full.height),
            fitz.Rect(0.0, PANEL_CROP.y0, PANEL_CROP.x0, PANEL_CROP.y1),
        ]
        for rectangle in redactions:
            page.add_redact_annot(rectangle, fill=None, cross_out=False)
        page.apply_redactions(images=1, graphics=1, text=0)
        page.clean_contents(sanitize=True)

        cropped = fitz.open()
        try:
            output_page = cropped.new_page(width=PANEL_CROP.width, height=PANEL_CROP.height)
            output_page.show_pdf_page(
                output_page.rect,
                source,
                0,
                clip=PANEL_CROP,
                keep_proportion=False,
            )
            cropped.set_metadata(source.metadata)
            cropped.save(panel_pdf, garbage=4, clean=True, deflate=True)
        finally:
            cropped.close()
    finally:
        source.close()


def _render_panel(pdf_path: Path, png_path: Path, dpi: float) -> tuple[int, int, list[float]]:
    document = fitz.open(pdf_path)
    try:
        page = document[0]
        pixmap = page.get_pixmap(matrix=fitz.Matrix(dpi / 72.0, dpi / 72.0), alpha=False)
        image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
        image.save(png_path, dpi=(dpi, dpi), optimize=True)
    finally:
        document.close()
    with Image.open(png_path) as image:
        declared_dpi = [float(value) for value in image.info.get("dpi", (0.0, 0.0))]
    return pixmap.width, pixmap.height, declared_dpi


def _write_svg(pdf_path: Path, svg_path: Path) -> None:
    document = fitz.open(pdf_path)
    try:
        svg = document[0].get_svg_image(text_as_path=True)
    finally:
        document.close()
    svg_path.write_text(svg, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--streams-svg", required=True, type=Path)
    parser.add_argument("--corrected-roi-svg", required=True, type=Path)
    parser.add_argument("--render-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--vector-template", type=Path, default=DEFAULT_VECTOR_TEMPLATE)
    parser.add_argument("--spatial-reference-svg", type=Path, default=DEFAULT_SPATIAL_REFERENCE)
    parser.add_argument("--roi-reference-svg", type=Path, default=DEFAULT_ROI_REFERENCE)
    parser.add_argument("--stem", default="Figure5c_ARISTA_corrected_finalAI_style")
    parser.add_argument("--dpi", type=float, default=300.0)
    parser.add_argument(
        "--dynamic-scientific-payload",
        action="store_true",
        help=(
            "Accept new corrected stream/ROI payloads and their manifest while "
            "keeping the Illustrator template and both historical style "
            "reference SVGs hash-locked."
        ),
    )
    return parser.parse_args()


def main() -> int:
    global EXPECTED_STREAM_LINE_COUNT
    global EXPECTED_STREAM_ARROW_COUNT
    args = parse_args()
    inputs = {
        "vector_template": (args.vector_template, EXPECTED_TEMPLATE_SHA256),
        "spatial_reference_svg": (
            args.spatial_reference_svg,
            EXPECTED_SPATIAL_REFERENCE_SHA256,
        ),
        "roi_reference_svg": (args.roi_reference_svg, EXPECTED_ROI_REFERENCE_SHA256),
        "corrected_streams_svg": (args.streams_svg, EXPECTED_STREAMS_SHA256),
        "corrected_roi_svg": (args.corrected_roi_svg, EXPECTED_CORRECTED_ROI_SHA256),
        "render_manifest": (args.render_manifest, EXPECTED_RENDER_MANIFEST_SHA256),
    }
    locked_hashes: dict[str, str] = {}
    for label, (path, expected) in inputs.items():
        if not path.exists():
            raise FileNotFoundError(path)
        if args.dynamic_scientific_payload and label in {
            "corrected_streams_svg",
            "corrected_roi_svg",
            "render_manifest",
        }:
            locked_hashes[label] = _sha256(path)
        else:
            locked_hashes[label] = _require_hash(path, expected, label)

    render_manifest = json.loads(args.render_manifest.read_text(encoding="utf-8"))
    if render_manifest["scientific_state"]["n_cells"] != 8106:
        raise ValueError("Accepted corrected Figure 5c must contain 8,106 t1 cells")
    if render_manifest["scientific_state"]["roi_n"] != ROI_MARKER_COUNT:
        raise ValueError("Accepted corrected Figure 5c must contain the frozen 1,454-cell ROI")
    object_counts = render_manifest["object_counts"]
    if args.dynamic_scientific_payload:
        if (
            int(object_counts.get("line_path_count", 0)) <= 0
            or int(object_counts.get("arrow_group_count", 0)) <= 0
            or int(object_counts.get("marker_count", 0)) != ROI_MARKER_COUNT
        ):
            raise ValueError(
                f"New corrected Figure 5c payload is incomplete: {object_counts}"
            )
        EXPECTED_STREAM_LINE_COUNT = int(object_counts["line_path_count"])
        EXPECTED_STREAM_ARROW_COUNT = int(object_counts["arrow_group_count"])
    elif object_counts != {
        "line_path_count": 5693,
        "arrow_group_count": 149,
        "marker_count": ROI_MARKER_COUNT,
    }:
        raise ValueError(f"Corrected Figure 5c object counts changed: {object_counts}")

    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite immutable output: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.assemble-", dir=output_dir.parent))
    try:
        fullpage_pdf = stage / f"{args.stem}_fullpage.pdf"
        panel_pdf = stage / f"{args.stem}.pdf"
        panel_svg = stage / f"{args.stem}.svg"
        panel_png = stage / f"{args.stem}.png"

        replacement = _assemble_fullpage(
            vector_template=args.vector_template,
            streams_svg=args.streams_svg,
            spatial_reference_svg=args.spatial_reference_svg,
            corrected_roi_svg=args.corrected_roi_svg,
            output_pdf=fullpage_pdf,
        )
        qa = _qa_fullpage(
            original_ai=args.vector_template,
            candidate_pdf=fullpage_pdf,
            corrected_roi_svg=args.corrected_roi_svg,
            replacement=replacement,
        )
        if not qa["passed"]:
            raise RuntimeError(f"Figure 5c full-page QA failed: {qa}")

        _make_physical_panel_pdf(fullpage_pdf, panel_pdf)
        png_width, png_height, png_dpi = _render_panel(panel_pdf, panel_png, args.dpi)
        _write_svg(panel_pdf, panel_svg)

        panel_document = fitz.open(panel_pdf)
        try:
            panel_page = panel_document[0]
            panel_size = [float(panel_page.rect.width), float(panel_page.rect.height)]
            panel_images = _image_inventory(panel_page)
            panel_markers = _roi_drawings(
                panel_page,
                region=fitz.Rect(
                    444.0 - PANEL_CROP.x0,
                    483.0 - PANEL_CROP.y0,
                    586.0 - PANEL_CROP.x0,
                    593.0 - PANEL_CROP.y0,
                ),
            )
            panel_marker_local_alphas = sorted(
                {round(float(row.get("fill_opacity", -1.0)), 6) for row in panel_markers}
            )
        finally:
            panel_document.close()
        if not np.allclose(panel_size, [PANEL_CROP.width, PANEL_CROP.height], atol=1e-3):
            raise RuntimeError(f"Figure 5c physical panel size changed: {panel_size}")
        if len(panel_markers) != ROI_MARKER_COUNT:
            raise RuntimeError(
                f"Figure 5c ROI markers changed after physical crop: {len(panel_markers)}"
            )
        if not all(abs(value - args.dpi) <= 0.1 for value in png_dpi):
            raise RuntimeError(f"Figure 5c PNG DPI metadata changed: {png_dpi}")

        manifest = {
            "schema": "cytobridge.arista.fig5c.original-layout.v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "figure": "ARISTA Figure 5c",
            "assembly_contract": (
                "Corrected 30-NN/scVelo spatial migration stream and corrected frozen-ROI "
                "full-spatial-vs-interaction-spatial cosine; exact final Illustrator display grammar"
            ),
            "inputs": {
                label: {"path": str(path.resolve()), "sha256": locked_hashes[label]}
                for label, (path, _expected) in inputs.items()
            },
            "scientific_state": render_manifest["scientific_state"],
            "style_contract": {
                **render_manifest["style_contract"],
                "display_coordinate_policy": (
                    "frozen manuscript x1/x2 positions; corrected velocity vectors mapped by the "
                    "single full-slice similarity transform archived in the accepted sidecar"
                ),
                "left_tissue_raster": "unchanged final-AI Fm327/Im0",
                "left_stream_stroke_width_policy": "historical unscaled SVG widths, rounded to 3 decimals as in final AI",
                "right_marker_geometry": "unchanged final-AI Fm332-Fm1785",
                "right_marker_alpha": HISTORICAL_ROI_ALPHA,
                "right_colorbar": "unchanged final-AI Fm331 fixed plasma Low-to-High bar",
                "preserved_editorial_objects": [
                    "panel label c",
                    "Spatial migration velocity",
                    "Spatial velocity cosine simlarity / (interaction VS migration)",
                    "Brain regeneration",
                    "r1/r2 coordinate glyph",
                    "gray ROI box and connectors",
                    "red nested display box",
                    "Low/High colorbar labels",
                ],
            },
            "replacement": replacement,
            "qa": qa,
            "outputs": {
                "fullpage_pdf": {"path": fullpage_pdf.name, "sha256": _sha256(fullpage_pdf)},
                "panel_pdf": {
                    "path": panel_pdf.name,
                    "sha256": _sha256(panel_pdf),
                    "page_size_pt": panel_size,
                    "raster_image_count": len(panel_images),
                    "raster_images": panel_images,
                    "roi_vector_marker_count": len(panel_markers),
                    "roi_marker_page_fill_opacity": HISTORICAL_ROI_ALPHA,
                    "roi_marker_form_local_opacity_values_get_drawings": panel_marker_local_alphas,
                },
                "panel_svg": {"path": panel_svg.name, "sha256": _sha256(panel_svg)},
                "panel_png": {
                    "path": panel_png.name,
                    "sha256": _sha256(panel_png),
                    "size_px": [png_width, png_height],
                    "dpi": png_dpi,
                },
            },
        }
        (stage / f"{args.stem}_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        stage.rename(output_dir)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise

    print(f"Output: {output_dir}")
    print(json.dumps(qa, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
