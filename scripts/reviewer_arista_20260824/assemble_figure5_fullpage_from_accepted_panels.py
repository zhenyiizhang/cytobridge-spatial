#!/usr/bin/env python3
"""Assemble one ARISTA Figure 5 page from the accepted corrected panel bundles.

This is an assembly-only program.  It does not recompute any scientific state
and it does not redraw the Illustrator layout.  Starting from the hash-locked
PDF-compatible ``Arista.ai`` page, it transplants the already-QA'd scientific
replacement payloads for panels a--e into their mutually disjoint original
content spans / image objects.  The unchanged Illustrator objects remain the
objects from the locked source document.

The output is deliberately immutable: the target directory must not already
contain files.  Rebuild into a new directory when a second copy is required.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable

import pymupdf as fitz
import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
VECTOR_TEMPLATE = Path("/Users/zhenyizhang/Desktop/202511/CytoBRIDGE/Arista.ai")

PANEL_A_DIR = (
    REPO_ROOT
    / "output/arista_figure5a_corrected_original_layout_20260823_v6_foregroundz004_finalqa"
)
PANEL_B_DIR = (
    REPO_ROOT
    / "output/arista_figure5b_corrected_original_layout_20260823_v4_finalqa_deterministic"
)
PANEL_C_DIR = REPO_ROOT / "output/arista_figure5c_corrected_original_layout_20260822_v2_finalqa"
PANEL_D_DIR = REPO_ROOT / "output/arista_figure5d_corrected_original_layout_20260823_v5_field_arrow"
PANEL_E_DIR = REPO_ROOT / "output/arista_figure5e_corrected_original_layout_20260823_v2_finalqa"

PANEL_A_PDF = PANEL_A_DIR / "Figure5a_ARISTA_corrected_finalAI_style.pdf"
PANEL_A_MANIFEST = PANEL_A_DIR / "Figure5a_ARISTA_corrected_finalAI_style_manifest.json"
PANEL_B_PDF = PANEL_B_DIR / "Figure5b_ARISTA_corrected_finalAI_style_fullpage.pdf"
PANEL_B_MANIFEST = PANEL_B_DIR / "Figure5b_ARISTA_corrected_finalAI_style_manifest.json"
PANEL_B_POINTS = (
    REPO_ROOT
    / "output/arista_figure5b_corrected_oldstyle_20260823_v3_displayfilter3291"
    / "time_0.5_points_only.svg"
)
PANEL_C_PDF = PANEL_C_DIR / "Figure5c_ARISTA_corrected_finalAI_style_fullpage.pdf"
PANEL_C_MANIFEST = PANEL_C_DIR / "Figure5c_ARISTA_corrected_finalAI_style_manifest.json"
PANEL_D_PDF = PANEL_D_DIR / "Figure5d_ARISTA_corrected_finalAI_style_fullpage.pdf"
PANEL_D_MANIFEST = PANEL_D_DIR / "Figure5d_ARISTA_corrected_finalAI_style_manifest.json"
PANEL_E_PDF = PANEL_E_DIR / "Figure5e_ARISTA_corrected_finalAI_style_fullpage.pdf"
PANEL_E_MANIFEST = PANEL_E_DIR / "Figure5e_ARISTA_corrected_finalAI_style_manifest.json"

EXPECTED_FILE_HASHES = {
    VECTOR_TEMPLATE: "673dc81f4856833c30c943ad5f2f4af9e69f771cea4cb63f2484ffbd18907694",
    PANEL_A_PDF: "bc72a77c2cc3daaec48e5cde6a021eafcea2fe291212c2648752ae341fd55163",
    PANEL_A_MANIFEST: "fb58c8760029fa56bfd6b7f761f4bb11b644d8cf1db63c9959e0af3b647c26c0",
    PANEL_B_PDF: "d18e53c763e0eaf07d8226e139b5b38d9dd84e16962cb79807e371c5c31f6e79",
    PANEL_B_MANIFEST: "deb361e1895cbcdcf778498f1651337e15d10a7bb57e1598ce1ee48b604a1e2e",
    PANEL_B_POINTS: "2bb7b44233356daf8d572b6a850731d37746eb4847d45959eca3a3ec9aff30fd",
    PANEL_C_PDF: "01887d2fc5715398ff71792fd382848704ac1c509e49c236201d844ebb2b4d22",
    PANEL_C_MANIFEST: "cf13558f357dfb8d5302d72e074410256cbcf2875275faedb8460dd549485753",
    PANEL_D_PDF: "d342872f78dcc1b189c688887d3beb9fd66badaa86de59d06be081567b082e19",
    PANEL_D_MANIFEST: "1e0a1f85aa262abb5564899ad0c94e5606bf71463ad254259d0a97ad86d35e18",
    PANEL_E_PDF: "ba764823739e14c9d2d887fa3332a0f1fad069cfdd1bab1f4085b6c353a42185",
    PANEL_E_MANIFEST: "848f292a4e0168879908408ca7adf55434d37303fec701218947f36224cc008b",
}

PAGE_XREF = 7
MAIN_CONTENT_XREF = 8
ORIGINAL_XREF_LENGTH = 28856
ORIGINAL_MAIN_SHA256 = "454fded539ece9063010eddaa0a62d174c428fe73667d927f1426384f7b7c499"

# All offsets refer to the decoded, hash-locked original main content stream.
SPAN_E = (89, 20030)
SPAN_C = (20107, 486615)
SPAN_D_STREAM = (559095, 870414)
SPAN_D_LABELS = (870415, 972341)
SPAN_B = (1087709, 1644917)
SPAN_D_WHITE_ARROW = (1644951, 1645489)

EXPECTED_OLD_SPAN_HASHES = {
    "5e bubbles": "dbf19addd92a63629726f090a75b3560c0ea076853035f3c1a4c5a362183ae7d",
    "5c streams": "f071b63af75a979f6b45fbf6a5f8c729d2d3e0b4305cd6204b2bc12b212efd0e",
    "5d streams": "3b9c028f2fe46f15cbab91444d031f57f5d70351bcc021974a1fd79858287db9",
    "5d labels": "c373b2128fb21982b06beb017e3cd355df11fbf9ba0f7dfcda5e290d31ab9c93",
    "5b markers": "276b8fd28acfb89ad1a548a1a1a85fe23c6f4f45839573e0873e450ddb352423",
    "5d white arrow": "905d7925ee246a077139637cfc1094d11249f1e9240c5f5e2e998d8dbbec10a8",
}

C_BEGIN = b"% CYTOBRIDGE_FIG5C_CORRECTED_SCVELO_BEGIN\n"
C_END = b"% CYTOBRIDGE_FIG5C_CORRECTED_SCVELO_END\n"
D_BEGIN = b"% CYTOBRIDGE_FIG5D_CORRECTED_STREAM_BEGIN\n"
D_END = b"% CYTOBRIDGE_FIG5D_CORRECTED_STREAM_END\n"
E_BEGIN = b"% CYTOBRIDGE_FIG5E_CORRECTED_BUBBLES_BEGIN\n"
E_END = b"% CYTOBRIDGE_FIG5E_CORRECTED_BUBBLES_END\n"
D_ARROW_BEGIN = b"% CYTOBRIDGE_FIG5D_CORRECTED_FIELD_ARROW_BEGIN\n"
D_ARROW_END = b"% CYTOBRIDGE_FIG5D_CORRECTED_FIELD_ARROW_END"
EXPECTED_NEW_BLOCK_HASHES = {
    "5c streams": "38f6db5bc7b089b6f36e773c9622979df992922b86adbaa18f1989b2c0184865",
    "5d streams": "b2dbd93dfcedf9a40a1cb5e8776d5abb3e99b62b9cfa6f7a9e168d49492ca308",
    "5d labels": "29f9ba989d586b81dd595bbcc433446b83035bbfbb28f1a6d5fadb3b44018908",
    "5e bubbles": "c413a774a7f66e8543d50b621665453b72c6d144ebd929e5440e8e73d2f10a5d",
    "5d corrected field arrow payload": "0e9b35599b3e2819dd498c3874f23de44b2f98efa437cbd995da7a94f198420e",
}
EXPECTED_D_V4_BASE_CONTENT_SHA256 = "89414729a7f3e188156dba8d0101dd99eb959a85d7a3d26bad45de9a21c796ea"
EXPECTED_HISTORICAL_ARROW_OBJECT_SHA256 = (
    "905d7925ee246a077139637cfc1094d11249f1e9240c5f5e2e998d8dbbec10a8"
)

A_IMAGE_XREF = 9766
D_IMAGE_XREF = 9545
D_SMASK_XREF = 28820
C_FORM_FIRST = 332
C_FORM_LAST = 1785
B_FORM_FIRST = 1787
B_FORM_LAST = 9525
B_MARKER_COUNT = 7766
B_MARKER_ALPHA = 0.899994
B_MARKER_CLIP = fitz.Rect(0.0, 451.4170, 213.6280, 629.1230)
B_SVG_TO_AI = (0.9232106506196, 0.9232105200406, -9.382843881239, 430.8126505540)

EXPECTED_A_RAW_RGB_SHA256 = "4c606c1e379f817624801f0c02507c719517eebe194b4e367b02c3a144ad0306"
EXPECTED_D_RAW_RGB_SHA256 = "9a1c20ac5113ec8a498d944dfc63ef89a033070615094b8418d524159a5d4d27"
EXPECTED_D_RAW_ALPHA_SHA256 = "db51aaa5c0cb0cbfbe08441243e12616b62c3ebbbe328adc999f0144f616fba1"
DYNAMIC_PANEL_PAYLOADS = False


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_inputs() -> dict[str, str]:
    if DYNAMIC_PANEL_PAYLOADS:
        paths = [
            VECTOR_TEMPLATE,
            PANEL_A_PDF,
            PANEL_A_MANIFEST,
            PANEL_B_PDF,
            PANEL_B_MANIFEST,
            PANEL_B_POINTS,
            PANEL_C_PDF,
            PANEL_C_MANIFEST,
            PANEL_D_PDF,
            PANEL_D_MANIFEST,
            PANEL_E_PDF,
            PANEL_E_MANIFEST,
        ]
        rows = {}
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(path)
            rows[str(path.resolve())] = _sha256(path)
        if rows[str(VECTOR_TEMPLATE.resolve())] != EXPECTED_FILE_HASHES[VECTOR_TEMPLATE]:
            raise ValueError("Locked Illustrator template changed")
        return rows
    rows: dict[str, str] = {}
    for path, expected in EXPECTED_FILE_HASHES.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(f"Input SHA256 changed for {path}: {actual} != {expected}")
        rows[str(path.resolve())] = actual
    return rows


def _main_stream(document: fitz.Document) -> bytes:
    page = document[0]
    if page.xref != PAGE_XREF:
        raise ValueError(f"Locked page xref changed: {page.xref}")
    if page.get_contents() != [MAIN_CONTENT_XREF]:
        raise ValueError(f"Locked page content structure changed: {page.get_contents()}")
    return document.xref_stream(MAIN_CONTENT_XREF)


def _candidate_stream(path: Path) -> bytes:
    document = fitz.open(path)
    try:
        page = document[0]
        candidates = [(len(document.xref_stream(xref)), xref) for xref in page.get_contents()]
        if not candidates:
            raise ValueError(f"No content stream in {path}")
        return document.xref_stream(max(candidates)[1])
    finally:
        document.close()


def _sentinel_block(
    stream: bytes, begin: bytes, end: bytes, expected_hash: str | None
) -> bytes:
    if stream.count(begin) != 1 or stream.count(end) != 1:
        raise ValueError("Accepted replacement sentinel count changed")
    start = stream.index(begin)
    stop = stream.index(end, start) + len(end)
    block = stream[start:stop]
    actual = _sha256_bytes(block)
    if expected_hash is not None and actual != expected_hash:
        raise ValueError(f"Accepted replacement block changed: {actual} != {expected_hash}")
    return block


def _replacement_payloads(original_stream: bytes) -> dict[str, bytes]:
    spans = [
        ("5e bubbles", SPAN_E),
        ("5c streams", SPAN_C),
        ("5d streams", SPAN_D_STREAM),
        ("5d labels", SPAN_D_LABELS),
        ("5b markers", SPAN_B),
        ("5d white arrow", SPAN_D_WHITE_ARROW),
    ]
    previous_end = -1
    for name, (start, end) in spans:
        if start < previous_end:
            raise ValueError(f"Replacement span overlap at {name}")
        previous_end = end
        actual = _sha256_bytes(original_stream[start:end])
        if actual != EXPECTED_OLD_SPAN_HASHES[name]:
            raise ValueError(f"Locked original span changed for {name}: {actual}")

    c_stream = _candidate_stream(PANEL_C_PDF)
    d_stream = _candidate_stream(PANEL_D_PDF)
    e_stream = _candidate_stream(PANEL_E_PDF)
    c_block = _sentinel_block(
        c_stream,
        C_BEGIN,
        C_END,
        None if DYNAMIC_PANEL_PAYLOADS else EXPECTED_NEW_BLOCK_HASHES["5c streams"],
    )
    d_block = _sentinel_block(
        d_stream,
        D_BEGIN,
        D_END,
        None if DYNAMIC_PANEL_PAYLOADS else EXPECTED_NEW_BLOCK_HASHES["5d streams"],
    )
    e_block = _sentinel_block(
        e_stream,
        E_BEGIN,
        E_END,
        None if DYNAMIC_PANEL_PAYLOADS else EXPECTED_NEW_BLOCK_HASHES["5e bubbles"],
    )

    d_block_end = d_stream.index(D_END) + len(D_END)
    label_start = d_block_end + 1
    after_label_anchor = original_stream[SPAN_D_LABELS[1] : SPAN_D_LABELS[1] + 128]
    if d_stream.count(after_label_anchor) != 1:
        raise ValueError("Could not uniquely locate the accepted Figure 5d label-block end")
    label_end = d_stream.index(after_label_anchor)
    d_labels = d_stream[label_start:label_end]
    if (
        not DYNAMIC_PANEL_PAYLOADS
        and _sha256_bytes(d_labels) != EXPECTED_NEW_BLOCK_HASHES["5d labels"]
    ):
        raise ValueError("Accepted Figure 5d label block changed")

    historical_arrow = original_stream[SPAN_D_WHITE_ARROW[0] : SPAN_D_WHITE_ARROW[1]]
    if _sha256_bytes(historical_arrow) != EXPECTED_HISTORICAL_ARROW_OBJECT_SHA256:
        raise ValueError("Locked historical white-arrow object changed")
    if DYNAMIC_PANEL_PAYLOADS:
        if d_stream.count(D_ARROW_BEGIN) or d_stream.count(D_ARROW_END):
            raise ValueError("Dynamic Figure 5d unexpectedly contains a manual field-arrow wrapper")
        if historical_arrow in d_stream:
            raise ValueError("Dynamic Figure 5d still contains the scientifically stale white arrow")
        arrow_payload = b""
    else:
        if d_stream.count(D_ARROW_BEGIN) != 1 or d_stream.count(D_ARROW_END) != 1:
            raise ValueError("Accepted Figure 5d corrected-field arrow marker count changed")
        arrow_begin = d_stream.index(D_ARROW_BEGIN)
        # The accepted v5 file is the byte-exact v4 content followed by one leading
        # newline, the complete marked wrapper, and one trailing newline.  Preserve
        # that entire accepted append payload when restoring the original arrow
        # layer role in the aggregate stream.
        arrow_payload_start = arrow_begin - 1
        arrow_payload = d_stream[arrow_payload_start:]
        if _sha256_bytes(d_stream[:arrow_payload_start]) != EXPECTED_D_V4_BASE_CONTENT_SHA256:
            raise ValueError("Figure 5d v5 no longer has the accepted v4 byte-exact prefix")
        if _sha256_bytes(arrow_payload) != EXPECTED_NEW_BLOCK_HASHES["5d corrected field arrow payload"]:
            raise ValueError("Accepted Figure 5d corrected-field arrow payload changed")
        if arrow_payload.count(historical_arrow) != 1:
            raise ValueError("Corrected-field arrow wrapper must contain the locked arrow object once")

    return {
        "5e bubbles": e_block,
        "5c streams": c_block,
        "5d streams": d_block,
        "5d labels": d_labels,
        "5d corrected field arrow": arrow_payload,
    }


def _svg_size_pt(path: Path) -> tuple[float, float]:
    head = path.read_text(encoding="utf-8")[:1000]
    width = re.search(r'\bwidth="([0-9.]+)pt"', head)
    height = re.search(r'\bheight="([0-9.]+)pt"', head)
    if width is None or height is None:
        raise ValueError(f"Could not read point dimensions from {path}")
    return float(width.group(1)), float(height.group(1))


def _b_points_document() -> fitz.Document:
    source = PANEL_B_POINTS.read_text(encoding="utf-8")
    per_marker_alpha = "; fill-opacity: 0.9"
    if source.count(per_marker_alpha) != B_MARKER_COUNT:
        raise ValueError("Accepted Figure 5b marker SVG alpha grammar changed")
    collection_open = '<g id="PathCollection_1">'
    if source.count(collection_open) != 1:
        raise ValueError("Accepted Figure 5b PathCollection_1 grammar changed")
    source = source.replace(
        collection_open,
        f'<g id="PathCollection_1" opacity="{B_MARKER_ALPHA}">',
        1,
    ).replace(per_marker_alpha, "")
    svg = fitz.open(stream=source.encode("utf-8"), filetype="svg")
    try:
        pdf_bytes = svg.convert_to_pdf()
    finally:
        svg.close()
    result = fitz.open("pdf", pdf_bytes)
    drawings = result[0].get_drawings()
    if len(drawings) != B_MARKER_COUNT:
        result.close()
        raise ValueError("Accepted Figure 5b vector marker count changed during import")
    if not all(
        abs(float(row.get("fill_opacity", -1.0)) - B_MARKER_ALPHA) <= 1e-6
        for row in drawings
    ):
        result.close()
        raise ValueError("Accepted Figure 5b marker alpha changed during import")
    return result


def _insert_b_form(document: fitz.Document, page: fitz.Page) -> tuple[bytes, dict[str, Any]]:
    points = _b_points_document()
    try:
        width, height = _svg_size_pt(PANEL_B_POINTS)
        sx, sy, ox, oy = B_SVG_TO_AI
        target = fitz.Rect(ox, oy, ox + width * sx, oy + height * sy)
        before = set(page.get_contents())
        page.show_pdf_page(target, points, 0, keep_proportion=False, overlay=False)
        added = [xref for xref in page.get_contents() if xref not in before]
        if len(added) != 1:
            raise ValueError(f"Expected one Figure 5b import invocation stream, found {added}")
        invocation = document.xref_stream(added[0]).strip()
        if invocation.count(b" Do") != 1:
            raise ValueError(f"Malformed Figure 5b imported invocation: {invocation!r}")
        match = re.search(rb"/(fzFrm\d+)\s+Do", invocation)
        if match is None:
            raise ValueError(f"Figure 5b wrapper Form name changed: {invocation!r}")
        form_name = match.group(1).decode("ascii")
        # The invocation is transplanted into xref 8 below.  Restore the exact
        # original one-stream page structure while retaining the imported Form.
        document.xref_set_key(page.xref, "Contents", f"{MAIN_CONTENT_XREF} 0 R")
        document.update_stream(added[0], b" ", compress=True)
    finally:
        points.close()

    kind, resources = document.xref_get_key(page.xref, "Resources/XObject")
    if kind != "dict":
        raise ValueError("Page XObject dictionary is missing after Figure 5b import")
    if re.search(rf"/{re.escape(form_name)}\s+\d+\s+0\s+R", resources) is None:
        raise ValueError("Imported Figure 5b wrapper Form is not reachable")
    removed = 0

    def drop(match: re.Match[str]) -> str:
        nonlocal removed
        form_id = int(match.group(1))
        if B_FORM_FIRST <= form_id <= B_FORM_LAST:
            removed += 1
            return ""
        return match.group(0)

    cleaned = re.sub(r"/Fm(\d+)\s+\d+\s+\d+\s+R", drop, resources)
    if removed != B_FORM_LAST - B_FORM_FIRST + 1:
        raise ValueError(f"Removed {removed} historical Figure 5b Form resources")
    document.xref_set_key(page.xref, "Resources/XObject", cleaned)
    return invocation, {
        "wrapper_form_name": form_name,
        "invocation_sha256": _sha256_bytes(invocation),
        "removed_historical_resources": removed,
        "target_rect_pt": [float(value) for value in target],
    }


def _xobject_map(document: fitz.Document, page: fitz.Page) -> dict[str, int]:
    kind, resources = document.xref_get_key(page.xref, "Resources/XObject")
    if kind != "dict":
        raise ValueError("Page XObject resources are missing")
    return {
        name: int(xref)
        for name, xref in re.findall(r"/(\S+)\s+(\d+)\s+\d+\s+R", resources)
    }


def _copy_c_roi_streams(document: fitz.Document, page: fitz.Page) -> dict[str, Any]:
    accepted = fitz.open(PANEL_C_PDF)
    try:
        source_map = _xobject_map(accepted, accepted[0])
        target_map = _xobject_map(document, page)
        changed = 0
        copied_hashes: list[str] = []
        for form_id in range(C_FORM_FIRST, C_FORM_LAST + 1):
            name = f"Fm{form_id}"
            if name not in source_map or name not in target_map:
                raise ValueError(f"Figure 5c ROI resource {name} is missing")
            source_stream = accepted.xref_stream(source_map[name])
            old_stream = document.xref_stream(target_map[name])
            if source_stream != old_stream:
                changed += 1
            document.update_stream(target_map[name], source_stream, compress=True)
            copied_hashes.append(_sha256_bytes(source_stream))
    finally:
        accepted.close()
    return {
        "form_first": C_FORM_FIRST,
        "form_last": C_FORM_LAST,
        "form_count": len(copied_hashes),
        "forms_changed_from_original": changed,
        "ordered_stream_hashes_sha256": _sha256_bytes("\n".join(copied_hashes).encode("ascii")),
    }


def _copy_d_scatter(document: fitz.Document, page: fitz.Page) -> dict[str, Any]:
    accepted = fitz.open(PANEL_D_PDF)
    try:
        source_map = _xobject_map(accepted, accepted[0])
        target_map = _xobject_map(document, page)
        if target_map.get("Im0") != D_IMAGE_XREF or "Im0" not in source_map:
            raise ValueError("Figure 5d /Im0 mapping changed")
        source_image = source_map["Im0"]
        source_kind, source_smask_value = accepted.xref_get_key(source_image, "SMask")
        target_kind, target_smask_value = document.xref_get_key(D_IMAGE_XREF, "SMask")
        source_match = re.fullmatch(r"(\d+)\s+0\s+R", source_smask_value.strip())
        target_match = re.fullmatch(r"(\d+)\s+0\s+R", target_smask_value.strip())
        if source_kind != "xref" or source_match is None:
            raise ValueError("Accepted Figure 5d scatter SMask is missing")
        if target_kind != "xref" or target_match is None or int(target_match.group(1)) != D_SMASK_XREF:
            raise ValueError("Locked Figure 5d scatter SMask changed")
        rgb = accepted.xref_stream(source_image)
        alpha = accepted.xref_stream(int(source_match.group(1)))
        rgb_sha = _sha256_bytes(rgb)
        alpha_sha = _sha256_bytes(alpha)
        if not DYNAMIC_PANEL_PAYLOADS and rgb_sha != EXPECTED_D_RAW_RGB_SHA256:
            raise ValueError("Accepted Figure 5d RGB payload changed")
        if not DYNAMIC_PANEL_PAYLOADS and alpha_sha != EXPECTED_D_RAW_ALPHA_SHA256:
            raise ValueError("Accepted Figure 5d alpha payload changed")
        document.update_stream(D_IMAGE_XREF, rgb, compress=True)
        document.update_stream(D_SMASK_XREF, alpha, compress=True)
    finally:
        accepted.close()
    return {
        "image_xref": D_IMAGE_XREF,
        "smask_xref": D_SMASK_XREF,
        "raw_rgb_sha256": rgb_sha,
        "raw_alpha_sha256": alpha_sha,
    }


def _copy_a_core(document: fitz.Document) -> dict[str, Any]:
    accepted = fitz.open(PANEL_A_PDF)
    try:
        images = accepted[0].get_images(full=True)
        candidates = [row for row in images if int(row[2]) == 5723 and int(row[3]) == 5761]
        if len(candidates) != 1:
            raise ValueError(f"Accepted Figure 5a core inventory changed: {candidates}")
        source_image = int(candidates[0][0])
        rgb = accepted.xref_stream(source_image)
        rgb_sha = _sha256_bytes(rgb)
        if not DYNAMIC_PANEL_PAYLOADS and rgb_sha != EXPECTED_A_RAW_RGB_SHA256:
            raise ValueError("Accepted Figure 5a RGB payload changed")
        kind, color_value = accepted.xref_get_key(source_image, "ColorSpace")
        if kind != "xref":
            raise ValueError("Accepted Figure 5a ICCBased ColorSpace changed")
        source_color_xref = int(color_value.split()[0])
        color_object = accepted.xref_object(source_color_xref, compressed=False)
        match = re.search(r"/ICCBased\s+(\d+)\s+0\s+R", color_object)
        if match is None:
            raise ValueError("Accepted Figure 5a ICC profile reference changed")
        source_icc_xref = int(match.group(1))

        new_icc_xref = document.get_new_xref()
        document.update_object(new_icc_xref, accepted.xref_object(source_icc_xref, compressed=False))
        document.update_stream(new_icc_xref, accepted.xref_stream(source_icc_xref), compress=True)
        new_color_xref = document.get_new_xref()
        new_color_object = re.sub(
            r"/ICCBased\s+\d+\s+0\s+R",
            f"/ICCBased {new_icc_xref} 0 R",
            color_object,
        )
        document.update_object(new_color_xref, new_color_object)

        image_object = accepted.xref_object(source_image, compressed=False)
        image_object = re.sub(
            r"/ColorSpace\s+\d+\s+0\s+R",
            f"/ColorSpace {new_color_xref} 0 R",
            image_object,
        )
        document.update_object(A_IMAGE_XREF, image_object)
        document.update_stream(A_IMAGE_XREF, rgb, compress=True)
    finally:
        accepted.close()
    return {
        "target_image_xref": A_IMAGE_XREF,
        "raw_rgb_sha256": rgb_sha,
        "new_colorspace_xref": new_color_xref,
        "new_icc_profile_xref": new_icc_xref,
    }


def _apply_replacements(
    original: bytes,
    replacements: list[tuple[int, int, bytes, str]],
) -> tuple[bytes, list[dict[str, Any]]]:
    previous_start = len(original) + 1
    updated = original
    ledger: list[dict[str, Any]] = []
    for start, end, replacement, name in sorted(replacements, key=lambda row: row[0], reverse=True):
        if end > previous_start:
            raise ValueError(f"Overlapping aggregate replacement at {name}")
        old = original[start:end]
        updated = updated[:start] + replacement + updated[end:]
        ledger.append(
            {
                "panel_payload": name,
                "old_span": [start, end],
                "old_size_bytes": len(old),
                "old_sha256": _sha256_bytes(old),
                "new_size_bytes": len(replacement),
                "new_sha256": _sha256_bytes(replacement),
            }
        )
        previous_start = start
    return updated, sorted(ledger, key=lambda row: row["old_span"][0])


def _assemble(output_pdf: Path) -> dict[str, Any]:
    document = fitz.open(VECTOR_TEMPLATE)
    try:
        if document.xref_length() != ORIGINAL_XREF_LENGTH:
            raise ValueError("Locked Illustrator xref inventory changed")
        original_stream = _main_stream(document)
        if _sha256_bytes(original_stream) != ORIGINAL_MAIN_SHA256:
            raise ValueError("Locked Illustrator main content stream changed")
        payloads = _replacement_payloads(original_stream)
        page = document[0]
        b_invocation, b_stats = _insert_b_form(document, page)
        b_block = b"\n" + b_invocation + b"\n"
        replacements = [
            (*SPAN_E, payloads["5e bubbles"], "5e bubbles"),
            (*SPAN_C, payloads["5c streams"], "5c streams"),
            (*SPAN_D_STREAM, payloads["5d streams"], "5d streams"),
            (*SPAN_D_LABELS, payloads["5d labels"], "5d labels"),
            (*SPAN_B, b_block, "5b markers"),
            (
                *SPAN_D_WHITE_ARROW,
                payloads["5d corrected field arrow"],
                "5d corrected field arrow",
            ),
        ]
        updated_stream, ledger = _apply_replacements(original_stream, replacements)
        document.update_stream(MAIN_CONTENT_XREF, updated_stream, compress=True)
        c_stats = _copy_c_roi_streams(document, page)
        d_stats = _copy_d_scatter(document, page)
        a_stats = _copy_a_core(document)
        if page.get_contents() != [MAIN_CONTENT_XREF]:
            raise AssertionError(f"Aggregate page content structure changed: {page.get_contents()}")
        document.save(
            output_pdf,
            garbage=0,
            clean=False,
            deflate=True,
            no_new_id=True,
            preserve_metadata=True,
        )
    finally:
        document.close()
    return {
        "original_main_content_sha256": ORIGINAL_MAIN_SHA256,
        "updated_main_content_sha256": _sha256_bytes(updated_stream),
        "updated_main_content_size_bytes": len(updated_stream),
        "replacement_ledger": ledger,
        "panel_a": a_stats,
        "panel_b": b_stats,
        "panel_c": c_stats,
        "panel_d": d_stats,
    }


def _render_mupdf(path: Path, scale: float) -> np.ndarray:
    document = fitz.open(path)
    try:
        pixmap = document[0].get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
        return np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
            pixmap.height, pixmap.width, pixmap.n
        )[:, :, :3].copy()
    finally:
        document.close()


def _render_poppler(path: Path, scale: float, scratch: Path, tag: str) -> np.ndarray:
    executable = shutil.which("pdftoppm")
    if executable is None:
        raise RuntimeError("pdftoppm is required for independent Poppler QA")
    prefix = scratch / f"{tag}_{scale:g}x"
    subprocess.run(
        [
            executable,
            "-f",
            "1",
            "-l",
            "1",
            "-singlefile",
            "-r",
            f"{72.0 * scale:.8f}",
            "-png",
            str(path),
            str(prefix),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    png = prefix.with_suffix(".png")
    with Image.open(png) as image:
        return np.asarray(image.convert("RGB")).copy()


def _make_a_only(path: Path) -> None:
    document = fitz.open(VECTOR_TEMPLATE)
    try:
        _copy_a_core(document)
        document.save(path, garbage=0, clean=False, deflate=True, no_new_id=True)
    finally:
        document.close()


def _pixel_composition_qa(
    output_pdf: Path,
    a_only_pdf: Path,
    scratch: Path,
) -> dict[str, Any]:
    singles = {
        "a": a_only_pdf,
        "b": PANEL_B_PDF,
        "c": PANEL_C_PDF,
        "d": PANEL_D_PDF,
        "e": PANEL_E_PDF,
    }
    renderers = {
        "mupdf": lambda path, scale, tag: _render_mupdf(path, scale),
        "poppler": lambda path, scale, tag: _render_poppler(path, scale, scratch, tag),
    }
    report: dict[str, Any] = {}
    all_passed = True
    for renderer_name, renderer in renderers.items():
        scale_rows: dict[str, Any] = {}
        for scale in (1.0, 2.0, 4.0):
            base = renderer(VECTOR_TEMPLATE, scale, f"{renderer_name}_base")
            final = renderer(output_pdf, scale, f"{renderer_name}_final")
            if final.shape != base.shape:
                raise ValueError("Aggregate page dimensions changed")
            rendered = {
                name: renderer(path, scale, f"{renderer_name}_{name}")
                for name, path in singles.items()
            }
            if any(array.shape != base.shape for array in rendered.values()):
                raise ValueError("Accepted single-panel fullpage render dimensions changed")
            masks = {name: np.any(array != base, axis=2) for name, array in rendered.items()}
            names = list(masks)
            pair_rows = []
            conflicting_overlap = 0
            for index, left in enumerate(names):
                for right in names[index + 1 :]:
                    overlap = masks[left] & masks[right]
                    count = int(overlap.sum())
                    conflict = int(
                        np.any(rendered[left] != rendered[right], axis=2)[overlap].sum()
                    ) if count else 0
                    conflicting_overlap += conflict
                    pair_rows.append(
                        {
                            "panels": [left, right],
                            "overlap_pixels": count,
                            "conflicting_overlap_pixels": conflict,
                        }
                    )
            expected = base.copy()
            for name in names:
                expected[masks[name]] = rendered[name][masks[name]]
            delta = np.abs(final.astype(np.int16) - expected.astype(np.int16))
            differing = int(np.any(delta != 0, axis=2).sum())
            passed = conflicting_overlap == 0 and differing == 0
            all_passed &= passed
            scale_rows[f"scale_{scale:g}"] = {
                "passed": passed,
                "shape_hwc": list(base.shape),
                "panel_changed_pixel_counts": {
                    name: int(mask.sum()) for name, mask in masks.items()
                },
                "pairwise_overlap": pair_rows,
                "conflicting_overlap_pixels": conflicting_overlap,
                "aggregate_vs_composed_oracle_differing_pixels": differing,
                "aggregate_vs_composed_oracle_max_abs_error_0_255": int(delta.max()),
                "aggregate_vs_composed_oracle_mae_0_255": float(delta.mean()),
            }
        report[renderer_name] = scale_rows
    return {"passed": all_passed, **report}


def _text_spans(page: fitz.Page) -> list[dict[str, Any]]:
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
    rows = []
    for item in page.get_image_info(hashes=True, xrefs=True):
        digest = item.get("digest", b"")
        rows.append(
            {
                "number": int(item["number"]),
                "width": int(item["width"]),
                "height": int(item["height"]),
                "bbox": [round(float(value), 6) for value in item["bbox"]],
                "transform": [round(float(value), 6) for value in item["transform"]],
                "digest": digest.hex() if isinstance(digest, bytes) else str(digest),
            }
        )
    return rows


def _semantic_object(document: fitz.Document, xref: int) -> dict[str, Any]:
    """Return an order- and compression-insensitive PDF object description."""

    keys = list(document.xref_get_keys(xref))
    if not keys:
        # Array / scalar indirect objects do not expose dictionary keys.
        return {
            "indirect_value": re.sub(
                r"\s+", " ", document.xref_object(xref, compressed=False)
            ).strip()
        }
    ignored = {"Length", "Filter"} if document.xref_is_stream(xref) else set()
    values: dict[str, Any] = {}
    for key in sorted(set(keys) - ignored):
        kind, value = document.xref_get_key(xref, key)
        if kind in {"int", "float"}:
            values[key] = ["number", float(value)]
        else:
            values[key] = [kind, value]
    return values


def _image_geometry_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["width"],
        row["height"],
        *tuple(round(float(value), 6) for value in row["bbox"]),
        *tuple(round(float(value), 6) for value in row["transform"]),
    )


def _group_image_digests(rows: list[dict[str, Any]]) -> dict[tuple[Any, ...], list[str]]:
    result: dict[tuple[Any, ...], list[str]] = {}
    for row in rows:
        result.setdefault(_image_geometry_key(row), []).append(row["digest"])
    return {key: sorted(values) for key, values in result.items()}


def _static_object_qa(output_pdf: Path, assembly: dict[str, Any]) -> dict[str, Any]:
    original = fitz.open(VECTOR_TEMPLATE)
    candidate = fitz.open(output_pdf)
    c_targets: set[int] = set()
    try:
        original_page = original[0]
        candidate_page = candidate[0]
        original_map = _xobject_map(original, original_page)
        for form_id in range(C_FORM_FIRST, C_FORM_LAST + 1):
            c_targets.add(original_map[f"Fm{form_id}"])
        excluded = {PAGE_XREF, MAIN_CONTENT_XREF, A_IMAGE_XREF, D_IMAGE_XREF, D_SMASK_XREF} | c_targets
        object_mismatches: list[int] = []
        stream_mismatches: list[int] = []
        checked_objects = 0
        checked_streams = 0
        for xref in range(1, ORIGINAL_XREF_LENGTH):
            if xref in excluded:
                continue
            checked_objects += 1
            if _semantic_object(original, xref) != _semantic_object(candidate, xref):
                object_mismatches.append(xref)
            if original.xref_is_stream(xref):
                checked_streams += 1
                if original.xref_stream(xref) != candidate.xref_stream(xref):
                    stream_mismatches.append(xref)

        candidate_stream = candidate.xref_stream(MAIN_CONTENT_XREF)
        content_exact = _sha256_bytes(candidate_stream) == assembly["updated_main_content_sha256"]
        text_equal = _text_spans(original_page) == _text_spans(candidate_page)
        page_size_equal = np.allclose(original_page.rect, candidate_page.rect, atol=1e-6)
        old_images = _image_inventory(original_page)
        new_images = _image_inventory(candidate_page)
        accepted_a = fitz.open(PANEL_A_PDF)
        accepted_d = fitz.open(PANEL_D_PDF)
        try:
            accepted_a_images = _image_inventory(accepted_a[0])
            accepted_d_images = _image_inventory(accepted_d[0])
        finally:
            accepted_a.close()
            accepted_d.close()
        old_groups = _group_image_digests(old_images)
        new_groups = _group_image_digests(new_images)
        accepted_a_groups = _group_image_digests(accepted_a_images)
        accepted_d_groups = _group_image_digests(accepted_d_images)
        a_key = next(
            key for key in new_groups if key[0:2] == (5723, 5761) and np.allclose(key[2:6], [23.78199, 14.293762, 441.495361, 434.780701], atol=1e-5)
        )
        d_key = next(
            key for key in new_groups if key[0:2] == (1397, 1362) and np.allclose(key[2:6], [22.943546, 635.961548, 229.225922, 837.075806], atol=1e-5)
        )
        a_accepted_key = next(key for key in accepted_a_groups if key[0:2] == (5723, 5761))
        d_accepted_key = next(
            key for key in accepted_d_groups if key[0:2] == (1397, 1362) and np.allclose(key[2:6], [22.943546, 635.961548, 229.225922, 837.075806], atol=1e-5)
        )
        static_image_keys = set(old_groups) - {a_key, d_key}
        static_image_groups_equal = all(old_groups[key] == new_groups.get(key) for key in static_image_keys)
        a_visible_payload_equal = new_groups[a_key] == accepted_a_groups[a_accepted_key]
        d_visible_payload_equal = new_groups[d_key] == accepted_d_groups[d_accepted_key]

        old_resource_images = {row[0]: row for row in original_page.get_images(full=True)}
        new_resource_images = {row[0]: row for row in candidate_page.get_images(full=True)}
        static_resource_images_equal = all(
            old_resource_images[xref] == new_resource_images.get(xref)
            for xref in old_resource_images
            if xref not in {A_IMAGE_XREF, D_IMAGE_XREF}
        )
        a_resource = new_resource_images.get(A_IMAGE_XREF)
        d_resource = new_resource_images.get(D_IMAGE_XREF)
        targeted_resource_geometry_equal = (
            a_resource is not None
            and a_resource[1] == 0
            and a_resource[2:4] == (5723, 5761)
            and d_resource is not None
            and d_resource[1] == D_SMASK_XREF
            and d_resource[2:4] == (1397, 1362)
        )

        accepted_c = fitz.open(PANEL_C_PDF)
        try:
            accepted_c_map = _xobject_map(accepted_c, accepted_c[0])
            candidate_map = _xobject_map(candidate, candidate_page)
            c_mismatches = [
                f"Fm{form_id}"
                for form_id in range(C_FORM_FIRST, C_FORM_LAST + 1)
                if candidate.xref_stream(candidate_map[f"Fm{form_id}"])
                != accepted_c.xref_stream(accepted_c_map[f"Fm{form_id}"])
            ]
        finally:
            accepted_c.close()

        a_raw_equal = (
            _sha256_bytes(candidate.xref_stream(A_IMAGE_XREF))
            == assembly["panel_a"]["raw_rgb_sha256"]
        )
        d_raw_equal = (
            _sha256_bytes(candidate.xref_stream(D_IMAGE_XREF))
            == assembly["panel_d"]["raw_rgb_sha256"]
        )
        d_alpha_equal = (
            _sha256_bytes(candidate.xref_stream(D_SMASK_XREF))
            == assembly["panel_d"]["raw_alpha_sha256"]
        )

        palette = json.loads(PANEL_A_MANIFEST.read_text(encoding="utf-8"))["inputs"]["palette_json"]["label_to_color"]
        palette_rgb = {
            tuple(int(color[index : index + 2], 16) / 255.0 for index in (1, 3, 5))
            for color in palette.values()
        }
        b_markers = []
        for drawing in candidate_page.get_drawings():
            rect = drawing["rect"]
            fill = drawing.get("fill")
            if fill is None or not B_MARKER_CLIP.intersects(rect):
                continue
            if abs(rect.width - 1.4597) > 0.01 or abs(rect.height - 1.4597) > 0.01:
                continue
            if any(np.allclose(fill, color, atol=2e-3) for color in palette_rgb):
                b_markers.append(drawing)
        b_opacities = sorted(
            {round(float(drawing.get("fill_opacity", -1.0)), 6) for drawing in b_markers}
        )
        old_dynamic_call_count = sum(
            candidate_stream.count(f"/Fm{form_id} Do".encode("ascii"))
            for form_id in range(1, 327)
        ) + sum(
            candidate_stream.count(f"/Fm{form_id} Do".encode("ascii"))
            for form_id in range(B_FORM_FIRST, B_FORM_LAST + 1)
        )
        sentinel_counts = {
            "5c_begin": candidate_stream.count(C_BEGIN),
            "5c_end": candidate_stream.count(C_END),
            "5d_begin": candidate_stream.count(D_BEGIN),
            "5d_end": candidate_stream.count(D_END),
            "5e_begin": candidate_stream.count(E_BEGIN),
            "5e_end": candidate_stream.count(E_END),
            "5d_corrected_field_arrow_begin": candidate_stream.count(D_ARROW_BEGIN),
            "5d_corrected_field_arrow_end": candidate_stream.count(D_ARROW_END),
        }
        original_arrow_object = original.xref_stream(MAIN_CONTENT_XREF)[
            SPAN_D_WHITE_ARROW[0] : SPAN_D_WHITE_ARROW[1]
        ]
        corrected_arrow_object_count = candidate_stream.count(original_arrow_object)
        if DYNAMIC_PANEL_PAYLOADS:
            corrected_arrow_payload_exact = (
                candidate_stream.count(D_ARROW_BEGIN) == 0
                and candidate_stream.count(D_ARROW_END) == 0
            )
            arrow_sentinel_contract = all(
                sentinel_counts[key] == 1
                for key in ("5c_begin", "5c_end", "5d_begin", "5d_end", "5e_begin", "5e_end")
            ) and sentinel_counts["5d_corrected_field_arrow_begin"] == 0 and sentinel_counts[
                "5d_corrected_field_arrow_end"
            ] == 0
            expected_arrow_object_count = 0
        else:
            corrected_arrow_payload_start = candidate_stream.index(D_ARROW_BEGIN) - 1
            corrected_arrow_payload_end = candidate_stream.index(D_ARROW_END) + len(D_ARROW_END) + 1
            corrected_arrow_payload = candidate_stream[
                corrected_arrow_payload_start:corrected_arrow_payload_end
            ]
            corrected_arrow_payload_exact = (
                _sha256_bytes(corrected_arrow_payload)
                == EXPECTED_NEW_BLOCK_HASHES["5d corrected field arrow payload"]
            )
            arrow_sentinel_contract = all(value == 1 for value in sentinel_counts.values())
            expected_arrow_object_count = 1
    finally:
        original.close()
        candidate.close()

    passed = (
        content_exact
        and text_equal
        and page_size_equal
        and static_image_groups_equal
        and a_visible_payload_equal
        and d_visible_payload_equal
        and static_resource_images_equal
        and targeted_resource_geometry_equal
        and not object_mismatches
        and not stream_mismatches
        and not c_mismatches
        and a_raw_equal
        and d_raw_equal
        and d_alpha_equal
        and len(b_markers) == B_MARKER_COUNT
        and b_opacities == [B_MARKER_ALPHA]
        and old_dynamic_call_count == 0
        and arrow_sentinel_contract
        and corrected_arrow_object_count == expected_arrow_object_count
        and corrected_arrow_payload_exact
    )
    return {
        "passed": passed,
        "page_size_equal": page_size_equal,
        "all_extractable_text_spans_equal": text_equal,
        "main_content_stream_exact": content_exact,
        "original_xrefs_checked_as_static": checked_objects,
        "original_static_streams_checked": checked_streams,
        "static_object_mismatch_count": len(object_mismatches),
        "static_object_mismatch_xrefs": object_mismatches[:50],
        "static_stream_mismatch_count": len(stream_mismatches),
        "static_stream_mismatch_xrefs": stream_mismatches[:50],
        "static_visible_image_groups_equal": static_image_groups_equal,
        "figure5a_visible_image_group_equal_to_accepted_panel": a_visible_payload_equal,
        "figure5d_visible_image_group_equal_to_accepted_panel": d_visible_payload_equal,
        "static_resource_image_entries_equal": static_resource_images_equal,
        "targeted_resource_image_geometry_equal": targeted_resource_geometry_equal,
        "figure5a_raw_payload_exact": a_raw_equal,
        "figure5c_roi_form_stream_mismatches": c_mismatches,
        "figure5d_rgb_payload_exact": d_raw_equal,
        "figure5d_alpha_payload_exact": d_alpha_equal,
        "figure5b_vector_marker_count": len(b_markers),
        "figure5b_marker_fill_opacities": b_opacities,
        "historical_5b_and_5e_form_call_count": old_dynamic_call_count,
        "replacement_sentinel_counts": sentinel_counts,
        "corrected_field_arrow_payload_exact": corrected_arrow_payload_exact,
        "locked_historical_arrow_object_count_inside_corrected_wrapper": corrected_arrow_object_count,
        "locked_historical_arrow_object_sha256": EXPECTED_HISTORICAL_ARROW_OBJECT_SHA256,
    }


def _save_preview(pdf: Path, png: Path, dpi: float = 300.0) -> dict[str, Any]:
    array = _render_mupdf(pdf, dpi / 72.0)
    Image.fromarray(array, mode="RGB").save(png, dpi=(dpi, dpi), optimize=True)
    with Image.open(png) as image:
        declared = [float(value) for value in image.info.get("dpi", (0.0, 0.0))]
    return {"pixel_size": [array.shape[1], array.shape[0]], "declared_dpi": declared}


def _write_provenance(path: Path) -> None:
    if DYNAMIC_PANEL_PAYLOADS:
        path.write_text(
            """# Provenance: package-native ARISTA Figure 5 full page

This page is assembled from the hash-locked PDF-compatible `Arista.ai` and
the fresh package-native Figure 5a--e panel bundles listed in `manifest.json`.
The model, communication, spatial velocity, gene velocity, growth, and
interaction values come from the spatial-QC z=50 package-native retraining
run. The submitted Illustrator layout, cell-type palette, typography,
annotations, marker geometry, colormaps, and panel placements are retained.

Figure 5a uses the submitted five-slice focus-anchor renderer and canonical
27-cell-type palette, followed only by the audited +0.04 foreground-z lift.
Figure 5b retains every fresh generated t=0.5 cell and the submitted vector
marker grammar. Figure 5d removes the scientifically stale historical manual
white arrow and does not invent a replacement; its corrected scVelo field is
shown directly. No panel-level display deletion is applied.
""",
            encoding="utf-8",
        )
        return
    path.write_text(
        """# Provenance: corrected ARISTA Figure 5 full page

This page is assembled from the hash-locked PDF-compatible `Arista.ai` and the
accepted corrected-original-style Figure 5a, 5b, 5c, 5d, and 5e bundles listed
in `manifest.json`. No scientific state was recomputed and no visual element
was redesigned. The accepted scientific payloads were transplanted into their
non-overlapping original content spans or image XObjects. All other original
Illustrator objects were retained and audited object-by-object.

Figure 5d is the accepted v5 field-arrow bundle. Its corrected scatter,
velocity stream, and labels retain the accepted v4 payloads, while its complete
marked corrected-field arrow wrapper is restored at the historical foreground
arrow span. The wrapper contains the exact locked Illustrator arrow geometry
under the accepted deterministic rigid transform.

Panels 5a and 5b apply the same audited display-only exclusion for corrected
generated t=0.5 row 3291. All 7,767 corrected rows remain in archived source
state and tables; only this one raster/vector glyph is hidden. Figure 5a also
applies a renderer-only +0.04 z offset to biological foreground traces so cell
points, communication lines/arrows, lineage paths, and endpoint markers remain
visibly above the five unchanged slice planes. Scientific values, x/y
coordinates, trace styling, slice borders, and the Illustrator layout are
unchanged.

The panel-specific bundles remain authoritative for scientific provenance and
standalone panel exports. This bundle is the deterministic one-page assembly.

## Source paths

- Locked Illustrator page: `/Users/zhenyizhang/Desktop/202511/CytoBRIDGE/Arista.ai`
- Figure 5a bundle: `output/arista_figure5a_corrected_original_layout_20260823_v6_foregroundz004_finalqa`
- Figure 5b bundle: `output/arista_figure5b_corrected_original_layout_20260823_v4_finalqa_deterministic`
- Figure 5c bundle: `output/arista_figure5c_corrected_original_layout_20260822_v2_finalqa`
- Figure 5d bundle: `output/arista_figure5d_corrected_original_layout_20260823_v5_field_arrow`
- Figure 5e bundle: `output/arista_figure5e_corrected_original_layout_20260823_v2_finalqa`

The embedded raster layers are intentional and inherited from the accepted
old-style panels: the Figure 5a scientific core, the Figure 5c tissue layer and
color bars, and the Figure 5d corrected scatter. All other plotted geometry is
retained as PDF vector content according to the accepted panel contracts.

## SHA-256

Exact input hashes are recorded in `Figure5_ARISTA_corrected_finalAI_style_manifest.json`.
Exact output hashes are recorded in `SHA256SUMS.txt`.

Rebuild from the repository root with:

`python scripts/arista_paper_equivalent/assemble_figure5_fullpage_from_accepted_panels.py --output-dir <NEW_EMPTY_OUTPUT_DIR>`
""",
        encoding="utf-8",
    )


def _write_checksums(directory: Path, names: Iterable[str]) -> None:
    rows = [f"{_sha256(directory / name)}  {name}" for name in sorted(names)]
    (directory / "SHA256SUMS.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dynamic-panel-payloads", action="store_true")
    parser.add_argument("--panel-a-pdf", type=Path, default=PANEL_A_PDF)
    parser.add_argument("--panel-a-manifest", type=Path, default=PANEL_A_MANIFEST)
    parser.add_argument("--panel-b-pdf", type=Path, default=PANEL_B_PDF)
    parser.add_argument("--panel-b-manifest", type=Path, default=PANEL_B_MANIFEST)
    parser.add_argument("--panel-b-points", type=Path, default=PANEL_B_POINTS)
    parser.add_argument("--panel-c-pdf", type=Path, default=PANEL_C_PDF)
    parser.add_argument("--panel-c-manifest", type=Path, default=PANEL_C_MANIFEST)
    parser.add_argument("--panel-d-pdf", type=Path, default=PANEL_D_PDF)
    parser.add_argument("--panel-d-manifest", type=Path, default=PANEL_D_MANIFEST)
    parser.add_argument("--panel-e-pdf", type=Path, default=PANEL_E_PDF)
    parser.add_argument("--panel-e-manifest", type=Path, default=PANEL_E_MANIFEST)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    global DYNAMIC_PANEL_PAYLOADS
    global PANEL_A_PDF, PANEL_A_MANIFEST, PANEL_A_DIR
    global PANEL_B_PDF, PANEL_B_MANIFEST, PANEL_B_POINTS, PANEL_B_DIR, B_MARKER_COUNT
    global PANEL_C_PDF, PANEL_C_MANIFEST, PANEL_C_DIR
    global PANEL_D_PDF, PANEL_D_MANIFEST, PANEL_D_DIR
    global PANEL_E_PDF, PANEL_E_MANIFEST, PANEL_E_DIR
    DYNAMIC_PANEL_PAYLOADS = bool(args.dynamic_panel_payloads)
    PANEL_A_PDF = args.panel_a_pdf.expanduser().resolve()
    PANEL_A_MANIFEST = args.panel_a_manifest.expanduser().resolve()
    PANEL_A_DIR = PANEL_A_PDF.parent
    PANEL_B_PDF = args.panel_b_pdf.expanduser().resolve()
    PANEL_B_MANIFEST = args.panel_b_manifest.expanduser().resolve()
    PANEL_B_POINTS = args.panel_b_points.expanduser().resolve()
    PANEL_B_DIR = PANEL_B_PDF.parent
    PANEL_C_PDF = args.panel_c_pdf.expanduser().resolve()
    PANEL_C_MANIFEST = args.panel_c_manifest.expanduser().resolve()
    PANEL_C_DIR = PANEL_C_PDF.parent
    PANEL_D_PDF = args.panel_d_pdf.expanduser().resolve()
    PANEL_D_MANIFEST = args.panel_d_manifest.expanduser().resolve()
    PANEL_D_DIR = PANEL_D_PDF.parent
    PANEL_E_PDF = args.panel_e_pdf.expanduser().resolve()
    PANEL_E_MANIFEST = args.panel_e_manifest.expanduser().resolve()
    PANEL_E_DIR = PANEL_E_PDF.parent
    if DYNAMIC_PANEL_PAYLOADS:
        b_manifest = json.loads(PANEL_B_MANIFEST.read_text(encoding="utf-8"))
        B_MARKER_COUNT = int(b_manifest["scientific_state"]["n_displayed"])
        if B_MARKER_COUNT <= 0:
            raise ValueError("Dynamic Figure 5b marker count must be positive")
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Immutable output directory is not empty: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    input_hashes = _require_inputs()

    with tempfile.TemporaryDirectory(prefix="arista_figure5_fullpage_", dir=output_dir.parent) as temp:
        stage = Path(temp)
        prefix = (
            "Figure5_ARISTA_package_native_finalAI_style"
            if DYNAMIC_PANEL_PAYLOADS
            else "Figure5_ARISTA_corrected_finalAI_style"
        )
        pdf_name = f"{prefix}.pdf"
        png_name = f"{prefix}.png"
        qa_name = f"{prefix}_QA.json"
        manifest_name = f"{prefix}_manifest.json"
        script_name = Path(__file__).name
        provenance_name = "PROVENANCE.md"
        pdf = stage / pdf_name
        png = stage / png_name

        assembly = _assemble(pdf)
        with tempfile.TemporaryDirectory(prefix="arista_figure5_oracle_") as oracle_temp:
            oracle_root = Path(oracle_temp)
            a_only = oracle_root / "Figure5a_only_fullpage.pdf"
            _make_a_only(a_only)
            pixel_qa = _pixel_composition_qa(pdf, a_only, oracle_root)
        object_qa = _static_object_qa(pdf, assembly)
        if not pixel_qa["passed"] or not object_qa["passed"]:
            raise RuntimeError(
                "Aggregate Figure 5 QA failed: "
                + json.dumps({"pixel": pixel_qa, "object": object_qa}, indent=2)
            )
        preview = _save_preview(pdf, png)
        with tempfile.TemporaryDirectory(prefix="arista_figure5_repeat_") as repeat_temp:
            repeat_root = Path(repeat_temp)
            repeat_pdf = repeat_root / pdf_name
            repeat_png = repeat_root / png_name
            repeat_assembly = _assemble(repeat_pdf)
            repeat_preview = _save_preview(repeat_pdf, repeat_png)
            determinism = {
                "passed": (
                    pdf.read_bytes() == repeat_pdf.read_bytes()
                    and png.read_bytes() == repeat_png.read_bytes()
                    and assembly["updated_main_content_sha256"]
                    == repeat_assembly["updated_main_content_sha256"]
                    and preview == repeat_preview
                ),
                "pdf_byte_identical_on_independent_rebuild": pdf.read_bytes()
                == repeat_pdf.read_bytes(),
                "pdf_sha256_primary": _sha256(pdf),
                "pdf_sha256_repeat": _sha256(repeat_pdf),
                "png_byte_identical_on_independent_rebuild": png.read_bytes()
                == repeat_png.read_bytes(),
                "png_sha256_primary": _sha256(png),
                "png_sha256_repeat": _sha256(repeat_png),
                "main_content_sha256_primary": assembly["updated_main_content_sha256"],
                "main_content_sha256_repeat": repeat_assembly["updated_main_content_sha256"],
                "preview_contract_equal": preview == repeat_preview,
            }
        if not determinism["passed"]:
            raise RuntimeError(f"Aggregate Figure 5 determinism QA failed: {determinism}")
        qa = {
            "schema": "cytobridge.arista.figure5.fullpage-assembly-qa.v2",
            "passed": True,
            "source_compatibility": {
                "locked_template_sha256": EXPECTED_FILE_HASHES[VECTOR_TEMPLATE],
                "same_locked_template_for_all_panels": True,
                "replacement_spans_pairwise_non_overlapping": True,
            },
            "assembly": assembly,
            "static_object_qa": object_qa,
            "pixel_composition_qa": pixel_qa,
            "determinism_qa": determinism,
            "preview": preview,
        }
        (stage / qa_name).write_text(json.dumps(qa, indent=2, ensure_ascii=False), encoding="utf-8")
        shutil.copy2(Path(__file__).resolve(), stage / script_name)
        _write_provenance(stage / provenance_name)

        outputs = {
            pdf_name: {"sha256": _sha256(pdf), "size_bytes": pdf.stat().st_size},
            png_name: {"sha256": _sha256(png), "size_bytes": png.stat().st_size},
            qa_name: {"sha256": _sha256(stage / qa_name), "size_bytes": (stage / qa_name).stat().st_size},
            script_name: {
                "sha256": _sha256(stage / script_name),
                "size_bytes": (stage / script_name).stat().st_size,
            },
            provenance_name: {
                "sha256": _sha256(stage / provenance_name),
                "size_bytes": (stage / provenance_name).stat().st_size,
            },
        }
        manifest = {
            "schema": "cytobridge.arista.figure5.fullpage-assembly.v2",
            "figure": "ARISTA Figure 5, panels a-e",
            "assembly_contract": (
                (
                    "Deterministic one-page assembly of fresh package-native Figure 5a-e "
                    "scientific payloads in the locked original Illustrator page; submitted "
                    "palette/style retained; Figure 5a has the renderer-only +0.04 foreground-z "
                    "lift; all fresh Figure 5b markers retained; stale historical Figure 5d "
                    "manual white arrow removed without replacement"
                )
                if DYNAMIC_PANEL_PAYLOADS
                else (
                    "Deterministic one-page assembly of accepted corrected panel payloads in the "
                    "locked original Illustrator page, including the accepted Figure 5d v5 "
                    "corrected-field arrow and the audited Figure 5a/5b generated t=0.5 "
                    "single-glyph display filter, with the renderer-only Figure 5a +0.04 "
                    "biological-foreground z lift; no scientific recomputation or redesign"
                )
            ),
            "inputs": input_hashes,
            "source_panel_bundles": {
                "5a": str(PANEL_A_DIR.relative_to(REPO_ROOT)),
                "5b": str(PANEL_B_DIR.relative_to(REPO_ROOT)),
                "5c": str(PANEL_C_DIR.relative_to(REPO_ROOT)),
                "5d": str(PANEL_D_DIR.relative_to(REPO_ROOT)),
                "5e": str(PANEL_E_DIR.relative_to(REPO_ROOT)),
            },
            "replacement_ledger": assembly["replacement_ledger"],
            "qa": {
                "passed": True,
                "object_qa_passed": object_qa["passed"],
                "pixel_composition_qa_passed": pixel_qa["passed"],
                "determinism_qa_passed": determinism["passed"],
                "qa_file": qa_name,
                "qa_sha256": outputs[qa_name]["sha256"],
            },
            "outputs": outputs,
            "rebuild_command": (
                "python scripts/arista_paper_equivalent/"
                "assemble_figure5_fullpage_from_accepted_panels.py "
                "--output-dir <NEW_EMPTY_OUTPUT_DIR>"
            ),
        }
        (stage / manifest_name).write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        checksum_names = [*outputs, manifest_name]
        _write_checksums(stage, checksum_names)

        if output_dir.exists():
            output_dir.rmdir()
        stage.rename(output_dir)

    print(f"PDF: {output_dir / pdf_name}")
    print(f"PNG: {output_dir / png_name}")
    print(f"Manifest: {output_dir / manifest_name}")
    print(f"QA: {output_dir / qa_name}")
    print(f"SHA256SUMS: {output_dir / 'SHA256SUMS.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
