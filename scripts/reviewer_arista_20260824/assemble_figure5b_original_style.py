#!/usr/bin/env python3
"""Replace only the Figure 5b marker block in the final ARISTA Illustrator page.

The final AI artwork contains 7,739 vector marker Form invocations copied from
the historical ``time_0.5.svg``.  This assembler removes precisely that
contiguous marker block, inserts the corrected vector PathCollection using the
measured historical affine, and leaves all editorial text, coordinate glyphs,
and neighboring panels untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import fitz
import numpy as np
from PIL import Image


DEFAULT_VECTOR_TEMPLATE = Path("/Users/zhenyizhang/Desktop/202511/CytoBRIDGE/Arista.ai")
DEFAULT_LAYOUT_PNG = Path("/Users/zhenyizhang/Desktop/202511/CytoBRIDGE/Arista.png")
EXPECTED_TEMPLATE_SHA256 = "673dc81f4856833c30c943ad5f2f4af9e69f771cea4cb63f2484ffbd18907694"
EXPECTED_REFERENCE_SHA256 = "ce4da7d34c9ebe35c803785c013ffd1b2eb561487e04aa94c3d81009c71fd7e6"
EXPECTED_PALETTE_SHA256 = "983b941fc93efe155511994d1d4b16cba5e11982cd81fb298d9a4a78907fbdd7"

# Exact vector registration from the final historical snapshot SVG to Arista.ai.
SVG_TO_AI_X_SCALE = 0.9232106506196
SVG_TO_AI_Y_SCALE = 0.9232105200406
SVG_TO_AI_X_OFFSET = -9.382843881239
SVG_TO_AI_Y_OFFSET = 430.8126505540

OLD_MARKER_FIRST = 1787
OLD_MARKER_LAST = 9525
OLD_MARKER_COUNT = OLD_MARKER_LAST - OLD_MARKER_FIRST + 1
OLD_MARKER_CLIP = fitz.Rect(0.0, 451.4170, 213.6280, 629.1230)
PANEL_CROP = fitz.Rect(0.0, 437.0, 205.5, 632.0)
HISTORICAL_MARKER_ALPHA = 0.899994

MARKER_BLOCK_RE = re.compile(
    rb"q\n0 212\.767 213\.628 177\.706 re\nW n\nq\n0 g\n0 G\n"
    rb"/GS4 gs\n0 TL/Fm(\d+) Do\nQ\nQ\n"
)
PLACEHOLDER = b"\n%%CYTOBRIDGE_FIG5B_CORRECTED_MARKERS%%\n"
PDF_ID_RE = re.compile(rb"/ID\[<[0-9A-Fa-f]{32}><[0-9A-Fa-f]{32}>\]")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_hash(path: Path, expected: str, label: str) -> str:
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(f"{label} SHA256 changed: {actual} != {expected}")
    return actual


def _normalize_pdf_trailer_id(path: Path) -> str:
    """Replace PyMuPDF's random trailer ID with a content-derived stable ID."""

    payload = path.read_bytes()
    matches = list(PDF_ID_RE.finditer(payload))
    if len(matches) != 1:
        raise ValueError(f"Expected one PDF trailer ID in {path}; found {len(matches)}")
    zeroed = PDF_ID_RE.sub(
        b"/ID[<00000000000000000000000000000000><00000000000000000000000000000000>]",
        payload,
        count=1,
    )
    digest = hashlib.sha256(zeroed).hexdigest().upper().encode("ascii")
    stable = b"/ID[<" + digest[:32] + b"><" + digest[32:] + b">]"
    normalized = PDF_ID_RE.sub(stable, payload, count=1)
    if len(normalized) != len(payload):
        raise AssertionError("Stable PDF trailer-ID replacement changed file length")
    path.write_bytes(normalized)
    return stable.decode("ascii")


def _svg_size_pt(path: Path) -> tuple[float, float]:
    head = path.read_text(encoding="utf-8")[:1000]
    width_match = re.search(r'\bwidth="([0-9.]+)pt"', head)
    height_match = re.search(r'\bheight="([0-9.]+)pt"', head)
    if not width_match or not height_match:
        raise ValueError(f"Could not read SVG point dimensions from {path}")
    return float(width_match.group(1)), float(height_match.group(1))


def _svg_to_pdf_document(path: Path, expected_marker_count: int) -> fitz.Document:
    # MuPDF ignores ``fill-opacity`` when it is attached to individual SVG
    # ``use`` nodes.  The equivalent PathCollection-level opacity is preserved
    # as a PDF ExtGState, so normalize only this generated marker collection
    # before import and verify the resulting vector semantics explicitly.
    source = path.read_text(encoding="utf-8")
    per_marker_alpha = "; fill-opacity: 0.9"
    if source.count(per_marker_alpha) != expected_marker_count:
        raise ValueError(
            "Point SVG alpha grammar changed: "
            f"found {source.count(per_marker_alpha)}, expected {expected_marker_count}"
        )
    collection_open = '<g id="PathCollection_1">'
    if source.count(collection_open) != 1:
        raise ValueError("Point SVG must contain exactly one PathCollection_1 group")
    source = source.replace(
        collection_open,
        f'<g id="PathCollection_1" opacity="{HISTORICAL_MARKER_ALPHA}">',
        1,
    ).replace(per_marker_alpha, "")

    svg = fitz.open(stream=source.encode("utf-8"), filetype="svg")
    try:
        pdf_bytes = svg.convert_to_pdf()
    finally:
        svg.close()
    document = fitz.open("pdf", pdf_bytes)
    drawings = document[0].get_drawings()
    if len(drawings) != expected_marker_count:
        document.close()
        raise ValueError(
            f"Imported point count changed: {len(drawings)} != {expected_marker_count}"
        )
    if not all(
        abs(float(row.get("fill_opacity", -1.0)) - HISTORICAL_MARKER_ALPHA) <= 1e-6
        for row in drawings
    ):
        document.close()
        raise ValueError("PDF import did not preserve the historical marker alpha=0.9")
    return document


def _main_content_stream(page: fitz.Page) -> int:
    candidates = [(len(page.parent.xref_stream(xref)), xref) for xref in page.get_contents()]
    if not candidates:
        raise RuntimeError("Illustrator page has no content stream")
    return max(candidates)[1]


def _remove_historical_marker_block(stream: bytes) -> tuple[bytes, dict[str, Any]]:
    matches = list(MARKER_BLOCK_RE.finditer(stream))
    form_ids = [int(match.group(1)) for match in matches]
    expected = list(range(OLD_MARKER_FIRST, OLD_MARKER_LAST + 1))
    if form_ids != expected:
        raise ValueError(
            f"Historical Figure 5b marker forms changed: found {len(form_ids)} "
            f"from {form_ids[:1]} to {form_ids[-1:]}."
        )
    if not all(left.end() == right.start() for left, right in zip(matches, matches[1:])):
        raise ValueError("Historical Figure 5b marker invocations are no longer contiguous")
    start, end = matches[0].start(), matches[-1].end()
    replacement = stream[:start] + PLACEHOLDER + stream[end:]
    return replacement, {
        "old_form_first": OLD_MARKER_FIRST,
        "old_form_last": OLD_MARKER_LAST,
        "old_form_count": len(matches),
        "content_byte_start": start,
        "content_byte_end": end,
    }


def _drop_old_marker_resources(document: fitz.Document, page: fitz.Page) -> int:
    kind, value = document.xref_get_key(page.xref, "Resources/XObject")
    if kind != "dict":
        raise ValueError("Page XObject resource dictionary is missing")
    reference_re = re.compile(r"/Fm(\d+)\s+(\d+)\s+(\d+)\s+R")
    removed = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal removed
        form_id = int(match.group(1))
        if OLD_MARKER_FIRST <= form_id <= OLD_MARKER_LAST:
            removed += 1
            return ""
        return match.group(0)

    cleaned = reference_re.sub(replace, value)
    if removed != OLD_MARKER_COUNT:
        raise ValueError(f"Removed {removed} old marker resources, expected {OLD_MARKER_COUNT}")
    document.xref_set_key(page.xref, "Resources/XObject", cleaned)
    return removed


def _replace_markers(
    *,
    vector_template: Path,
    points_svg: Path,
    output_pdf: Path,
    expected_marker_count: int,
) -> dict[str, Any]:
    document = fitz.open(vector_template)
    points_document = _svg_to_pdf_document(points_svg, expected_marker_count)
    try:
        if document.page_count != 1 or points_document.page_count != 1:
            raise ValueError("Template and point collection must each contain exactly one page")
        page = document[0]
        original_page_size = [float(page.rect.width), float(page.rect.height)]
        main_xref = _main_content_stream(page)
        original_stream = document.xref_stream(main_xref)
        stream_with_placeholder, removal = _remove_historical_marker_block(original_stream)

        svg_width, svg_height = _svg_size_pt(points_svg)
        target = fitz.Rect(
            SVG_TO_AI_X_OFFSET,
            SVG_TO_AI_Y_OFFSET,
            SVG_TO_AI_X_OFFSET + svg_width * SVG_TO_AI_X_SCALE,
            SVG_TO_AI_Y_OFFSET + svg_height * SVG_TO_AI_Y_SCALE,
        )
        contents_before = set(page.get_contents())
        page.show_pdf_page(
            target,
            points_document,
            0,
            keep_proportion=False,
            overlay=False,
        )
        added_contents = [xref for xref in page.get_contents() if xref not in contents_before]
        if len(added_contents) != 1:
            raise RuntimeError(f"Expected one imported Form invocation stream; got {added_contents}")
        invocation = document.xref_stream(added_contents[0]).strip()
        if b" Do" not in invocation:
            raise RuntimeError(f"Imported point Form invocation is malformed: {invocation!r}")
        if stream_with_placeholder.count(PLACEHOLDER) != 1:
            raise RuntimeError("Corrected marker insertion placeholder is not unique")
        updated_stream = stream_with_placeholder.replace(PLACEHOLDER, b"\n" + invocation + b"\n")
        document.update_stream(main_xref, updated_stream)
        document.update_stream(added_contents[0], b" ")
        removed_resources = _drop_old_marker_resources(document, page)
        document.save(output_pdf, garbage=4, clean=True, deflate=True)
    finally:
        points_document.close()
        document.close()

    stable_pdf_id = _normalize_pdf_trailer_id(output_pdf)

    return {
        **removal,
        "removed_old_marker_resources": removed_resources,
        "original_page_size_pt": original_page_size,
        "source_svg_size_pt": [svg_width, svg_height],
        "source_svg_to_ai_affine": [
            [SVG_TO_AI_X_SCALE, 0.0, SVG_TO_AI_X_OFFSET],
            [0.0, SVG_TO_AI_Y_SCALE, SVG_TO_AI_Y_OFFSET],
            [0.0, 0.0, 1.0],
        ],
        "insert_target_rect_pt": [float(value) for value in target],
        "pdf_import_alpha_strategy": "PathCollection group opacity ExtGState",
        "imported_marker_fill_opacity": HISTORICAL_MARKER_ALPHA,
        "stable_pdf_trailer_id": stable_pdf_id,
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
            fitz.Rect(PANEL_CROP.x1, PANEL_CROP.y0, full.width, PANEL_CROP.y1),
        ]
        for rect in redactions:
            page.add_redact_annot(rect, fill=None, cross_out=False)
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
    _normalize_pdf_trailer_id(panel_pdf)


def _render_panel(pdf_path: Path, png_path: Path, dpi: float) -> tuple[int, int, list[float]]:
    document = fitz.open(pdf_path)
    try:
        page = document[0]
        scale = dpi / 72.0
        pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
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


def _render_page_array(path: Path, scale: float) -> np.ndarray:
    document = fitz.open(path)
    try:
        pixmap = document[0].get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
        return np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
            pixmap.height, pixmap.width, pixmap.n
        )[:, :, :3]
    finally:
        document.close()


def _annotation_spans(page: fitz.Page) -> list[dict[str, Any]]:
    wanted = {"b", "Generated samples", "r1", "r2", "t=3.5DPI"}
    rows: list[dict[str, Any]] = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                if span["text"] not in wanted:
                    continue
                rows.append(
                    {
                        "text": span["text"],
                        "font": span["font"],
                        "size": float(span["size"]),
                        "color": int(span["color"]),
                        "bbox": [float(value) for value in span["bbox"]],
                    }
                )
    return sorted(rows, key=lambda row: (row["text"], row["bbox"]))


def _qa_fullpage(
    *,
    original_ai: Path,
    fullpage_pdf: Path,
    palette: dict[str, str],
    expected_marker_count: int,
) -> dict[str, Any]:
    original = fitz.open(original_ai)
    candidate = fitz.open(fullpage_pdf)
    try:
        original_page = original[0]
        candidate_page = candidate[0]
        annotations_equal = _annotation_spans(original_page) == _annotation_spans(candidate_page)
        page_size_equal = np.allclose(original_page.rect, candidate_page.rect, atol=1e-6)
        image_geometry_original = sorted((row[2], row[3]) for row in original_page.get_images(full=True))
        image_geometry_candidate = sorted((row[2], row[3]) for row in candidate_page.get_images(full=True))
        raster_images_unchanged = image_geometry_original == image_geometry_candidate

        palette_rgb = {
            tuple(int(color[index : index + 2], 16) / 255.0 for index in (1, 3, 5))
            for color in palette.values()
        }
        marker_drawings = []
        for drawing in candidate_page.get_drawings():
            rect = drawing["rect"]
            fill = drawing.get("fill")
            if fill is None or not OLD_MARKER_CLIP.intersects(rect):
                continue
            if abs(rect.width - 1.4597) > 0.01 or abs(rect.height - 1.4597) > 0.01:
                continue
            if not any(np.allclose(fill, color, atol=2e-3) for color in palette_rgb):
                continue
            marker_drawings.append(drawing)
        marker_opacities = sorted(
            {round(float(drawing.get("fill_opacity", -1.0)), 6) for drawing in marker_drawings}
        )
        marker_alpha_equal = (
            len(marker_drawings) == expected_marker_count
            and marker_opacities == [HISTORICAL_MARKER_ALPHA]
        )

        streams = b"\n".join(
            candidate.xref_stream(xref) for xref in candidate_page.get_contents()
        )
        old_marker_calls_remaining = sum(
            streams.count(f"/Fm{form_id} Do".encode("ascii"))
            for form_id in range(OLD_MARKER_FIRST, OLD_MARKER_LAST + 1)
        )
    finally:
        original.close()
        candidate.close()

    original_pixels = _render_page_array(original_ai, scale=2.0)
    candidate_pixels = _render_page_array(fullpage_pdf, scale=2.0)
    if original_pixels.shape != candidate_pixels.shape:
        raise AssertionError("Full-page render dimensions changed")
    outside = np.ones(original_pixels.shape[:2], dtype=bool)
    x0 = int(math.floor(PANEL_CROP.x0 * 2.0))
    y0 = int(math.floor(PANEL_CROP.y0 * 2.0))
    x1 = int(math.ceil(PANEL_CROP.x1 * 2.0))
    y1 = int(math.ceil(PANEL_CROP.y1 * 2.0))
    outside[y0:y1, x0:x1] = False
    delta = np.abs(candidate_pixels.astype(np.int16) - original_pixels.astype(np.int16))
    outside_max = int(delta[outside].max())
    outside_mae = float(delta[outside].mean())

    passed = (
        annotations_equal
        and page_size_equal
        and raster_images_unchanged
        and len(marker_drawings) == expected_marker_count
        and marker_alpha_equal
        and old_marker_calls_remaining == 0
        and outside_max == 0
    )
    return {
        "passed": passed,
        "annotations_equal": annotations_equal,
        "page_size_equal": bool(page_size_equal),
        "existing_raster_image_geometry_unchanged": raster_images_unchanged,
        "corrected_vector_marker_count": len(marker_drawings),
        "expected_corrected_marker_count": int(expected_marker_count),
        "corrected_marker_fill_opacity_values": marker_opacities,
        "corrected_marker_alpha_equal": marker_alpha_equal,
        "old_marker_calls_remaining": old_marker_calls_remaining,
        "outside_panel_max_abs_error_0_255": outside_max,
        "outside_panel_mae_0_255": outside_mae,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points-svg", required=True, type=Path)
    parser.add_argument("--render-manifest", required=True, type=Path)
    parser.add_argument("--canonical-reference-svg", required=True, type=Path)
    parser.add_argument("--palette-json", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--vector-template", type=Path, default=DEFAULT_VECTOR_TEMPLATE)
    parser.add_argument("--layout-png", type=Path, default=DEFAULT_LAYOUT_PNG)
    parser.add_argument("--stem", default="Figure5b_ARISTA_corrected_finalAI_style")
    parser.add_argument("--dpi", type=float, default=300.0)
    parser.add_argument(
        "--expected-marker-count",
        type=int,
        default=7767,
        help=(
            "Hash-locked displayed marker count. Use 7766 only with the audited "
            "generated t=0.5 row-3291 display-filter payload."
        ),
    )
    parser.add_argument(
        "--dynamic-scientific-payload",
        action="store_true",
        help=(
            "Accept the positive marker count declared by a fresh corrected run. "
            "The Illustrator template, palette, marker geometry, alpha, and affine remain locked."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = [
        args.points_svg,
        args.render_manifest,
        args.canonical_reference_svg,
        args.palette_json,
        args.vector_template,
        args.layout_png,
    ]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(missing)
    template_sha = _require_hash(args.vector_template, EXPECTED_TEMPLATE_SHA256, "Arista.ai")
    reference_sha = _require_hash(
        args.canonical_reference_svg, EXPECTED_REFERENCE_SHA256, "Figure 5b reference SVG"
    )
    palette_sha = _require_hash(args.palette_json, EXPECTED_PALETTE_SHA256, "ARISTA palette")
    render_manifest = json.loads(args.render_manifest.read_text(encoding="utf-8"))
    expected_marker_count = int(render_manifest["scientific_state"]["n_displayed"])
    if expected_marker_count != int(args.expected_marker_count):
        raise ValueError(
            "Figure 5b display marker contract changed: "
            f"manifest={expected_marker_count}, CLI={args.expected_marker_count}"
        )
    if not args.dynamic_scientific_payload and expected_marker_count not in {7766, 7767}:
        raise ValueError(
            "Figure 5b accepts only the full 7,767-row display or the audited "
            f"single-glyph filter (7,766); got {expected_marker_count}"
        )
    if args.dynamic_scientific_payload and expected_marker_count <= 0:
        raise ValueError("Fresh Figure 5b payload must contain a positive marker count")
    palette = json.loads(args.palette_json.read_text(encoding="utf-8"))

    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.assemble-", dir=output_dir.parent))
    try:
        fullpage_pdf = stage / f"{args.stem}_fullpage.pdf"
        panel_pdf = stage / f"{args.stem}.pdf"
        panel_svg = stage / f"{args.stem}.svg"
        panel_png = stage / f"{args.stem}.png"
        replacement = _replace_markers(
            vector_template=args.vector_template,
            points_svg=args.points_svg,
            output_pdf=fullpage_pdf,
            expected_marker_count=expected_marker_count,
        )
        qa = _qa_fullpage(
            original_ai=args.vector_template,
            fullpage_pdf=fullpage_pdf,
            palette=palette,
            expected_marker_count=expected_marker_count,
        )
        if not qa["passed"]:
            raise RuntimeError(f"Figure 5b full-page QA failed: {qa}")
        _make_physical_panel_pdf(fullpage_pdf, panel_pdf)
        png_width, png_height, png_dpi = _render_panel(panel_pdf, panel_png, args.dpi)
        _write_svg(panel_pdf, panel_svg)

        panel_document = fitz.open(panel_pdf)
        try:
            panel_page = panel_document[0]
            panel_page_size = [float(panel_page.rect.width), float(panel_page.rect.height)]
            panel_image_count = len(panel_page.get_images(full=True))
            panel_marker_drawings = [
                drawing
                for drawing in panel_page.get_drawings()
                if drawing.get("fill") is not None
                and abs(drawing["rect"].width - 1.4597) <= 0.01
                and abs(drawing["rect"].height - 1.4597) <= 0.01
            ]
            panel_marker_opacities = sorted(
                {
                    round(float(drawing.get("fill_opacity", -1.0)), 6)
                    for drawing in panel_marker_drawings
                }
            )
        finally:
            panel_document.close()
        if panel_image_count != 0:
            raise RuntimeError("Figure 5b panel must remain fully vector; raster image found")
        if len(panel_marker_drawings) != expected_marker_count:
            raise RuntimeError(
                "Figure 5b panel marker count changed after physical crop: "
                f"{len(panel_marker_drawings)} != {expected_marker_count}"
            )
        if panel_marker_opacities != [HISTORICAL_MARKER_ALPHA]:
            raise RuntimeError(
                "Figure 5b panel marker alpha changed after physical crop: "
                f"{panel_marker_opacities}"
            )
        if not np.allclose(panel_page_size, [PANEL_CROP.width, PANEL_CROP.height], atol=1e-3):
            raise RuntimeError(f"Figure 5b physical panel size changed: {panel_page_size}")
        if not all(abs(value - args.dpi) <= 0.1 for value in png_dpi):
            raise RuntimeError(f"Figure 5b PNG DPI metadata changed: {png_dpi}")

        manifest = {
            "schema": "cytobridge.arista.fig5b.original-layout.v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "figure": "ARISTA Figure 5b",
            "assembly_contract": (
                "Corrected generated t=0.5 state; exact final Illustrator marker affine and untouched annotations"
            ),
            "scientific_payload_mode": (
                "dynamic fresh corrected run" if args.dynamic_scientific_payload else "historical locked count"
            ),
            "inputs": {
                "points_svg": {"path": str(args.points_svg.resolve()), "sha256": _sha256(args.points_svg)},
                "render_manifest": {
                    "path": str(args.render_manifest.resolve()),
                    "sha256": _sha256(args.render_manifest),
                },
                "canonical_reference_svg": {
                    "path": str(args.canonical_reference_svg.resolve()),
                    "sha256": reference_sha,
                },
                "palette_json": {"path": str(args.palette_json.resolve()), "sha256": palette_sha},
                "vector_template": {"path": str(args.vector_template.resolve()), "sha256": template_sha},
                "layout_png": {"path": str(args.layout_png.resolve()), "sha256": _sha256(args.layout_png)},
            },
            "scientific_state": render_manifest["scientific_state"],
            "style_contract": {
                **render_manifest["style_contract"],
                "historical_marker_count": OLD_MARKER_COUNT,
                "corrected_marker_count": expected_marker_count,
                "historical_svg_to_ai_affine": replacement["source_svg_to_ai_affine"],
                "marker_diameter_final_pt": 2.0 * math.sqrt(2.5) / 2.0 * SVG_TO_AI_X_SCALE,
                "marker_alpha": 0.9,
                "marker_alpha_final_ai_pdf": HISTORICAL_MARKER_ALPHA,
                "marker_stroke": "none",
                "original_annotations_preserved": [
                    "panel label b",
                    "Generated samples title",
                    "r1/r2 coordinate glyph",
                    "t=3.5DPI",
                ],
            },
            "replacement": replacement,
            "qa": qa,
            "outputs": {
                "fullpage_pdf": {"path": fullpage_pdf.name, "sha256": _sha256(fullpage_pdf)},
                "panel_pdf": {
                    "path": panel_pdf.name,
                    "sha256": _sha256(panel_pdf),
                    "page_size_pt": panel_page_size,
                    "raster_image_count": panel_image_count,
                    "vector_marker_count": len(panel_marker_drawings),
                    "marker_fill_opacity_values": panel_marker_opacities,
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
