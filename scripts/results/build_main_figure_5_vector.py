#!/usr/bin/env python3
"""Apply the Main Figure 5 scientific-label v5 correction to a formal base PDF."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pymupdf as fitz
import numpy as np


PAGE_POINTS = (595.276, 841.89)
OUTPUT_STEM = "Figure5_ARISTA_package_native_scientific_labels"
BASE_SCHEMA = "cytobridge.arista.figure5.fullpage-assembly.v2"
BASE_FIGURE = "ARISTA Figure 5, panels a-e"
OLD_LABELS = (
    "Spatial migration velocity",
    "Spatial velocity cosine simlarity",
    "     (interaction VS migration)",
)
NEW_LABELS = (
    "Spatial velocity",
    "Spatial velocity cosine similarity",
    "(interaction vs full spatial velocity)",
)
ALLOWED_RECTS = (
    (232.0, 435.0, 412.0, 469.0),
    (405.0, 431.0, PAGE_POINTS[0], 475.0),
)


def _require_file(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def _prepare_output_dir(path: Path) -> Path:
    output = path.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    return output


def _load_base_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Base manifest must contain a JSON object")
    if value.get("schema") != BASE_SCHEMA or value.get("figure") != BASE_FIGURE:
        raise ValueError("Base manifest does not describe the formal Figure 5 page")
    return value


def _find_unique_span(page: fitz.Page, text: str) -> dict[str, Any]:
    matches: list[dict[str, Any]] = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            for span in line["spans"]:
                if span["text"] == text:
                    matches.append(span)
    if len(matches) != 1:
        raise ValueError(f"Expected one label {text!r}, found {len(matches)}")
    return matches[0]


def _rgb_from_int(value: int) -> tuple[float, float, float]:
    return (
        ((value >> 16) & 255) / 255,
        ((value >> 8) & 255) / 255,
        (value & 255) / 255,
    )


def _inside_allowed_rect(rect: fitz.Rect) -> bool:
    return any(fitz.Rect(values).contains(rect) for values in ALLOWED_RECTS)


def _insert_centered_text(
    page: fitz.Page,
    *,
    center_x: float,
    baseline_y: float,
    text: str,
    font_name: str,
    font_path: Path,
    fontsize: float,
    color: tuple[float, float, float],
) -> tuple[float, float, float, float]:
    font = fitz.Font(fontfile=str(font_path))
    width = font.text_length(text, fontsize=fontsize)
    origin = fitz.Point(center_x - width / 2, baseline_y)
    page.insert_text(
        origin,
        text,
        fontname=font_name,
        fontsize=fontsize,
        color=color,
        overlay=True,
    )
    return (
        float(origin.x),
        float(baseline_y - fontsize * 1.05),
        float(origin.x + width),
        float(baseline_y + fontsize * 0.38),
    )


def _render_array(path: Path, dpi: int) -> np.ndarray:
    with fitz.open(path) as document:
        pixmap = document[0].get_pixmap(
            matrix=fitz.Matrix(dpi / 72.0, dpi / 72.0),
            alpha=False,
        )
    return np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
        pixmap.height,
        pixmap.width,
        pixmap.n,
    )[:, :, :3]


def _outside_label_equivalence(
    base_pdf: Path,
    result_pdf: Path,
    *,
    dpi: int = 144,
) -> dict[str, object]:
    before = _render_array(base_pdf, dpi)
    after = _render_array(result_pdf, dpi)
    if before.shape != after.shape:
        raise ValueError("Vector rebuild changed the page dimensions")
    mask = np.zeros(before.shape[:2], dtype=bool)
    scale = dpi / 72.0
    for x0, y0, x1, y1 in ALLOWED_RECTS:
        px0 = max(0, math.floor(x0 * scale))
        py0 = max(0, math.floor(y0 * scale))
        px1 = min(mask.shape[1], math.ceil(x1 * scale))
        py1 = min(mask.shape[0], math.ceil(y1 * scale))
        mask[py0:py1, px0:px1] = True
    difference = np.any(before != after, axis=2) & ~mask
    return {
        "comparison_dpi": dpi,
        "outside_label_changed_pixels": int(difference.sum()),
        "passed": bool(not difference.any()),
    }


def _render_png(pdf_path: Path, png_path: Path, dpi: int) -> None:
    with fitz.open(pdf_path) as document:
        if document.page_count != 1:
            raise ValueError("Figure 5 output must contain one page")
        pixmap = document[0].get_pixmap(
            matrix=fitz.Matrix(dpi / 72.0, dpi / 72.0),
            alpha=False,
        )
        pixmap.save(png_path)


def build_vector_figure(
    base_pdf: Path,
    base_manifest: Path,
    font_file: Path,
    output_dir: Path,
    *,
    dpi: int = 300,
) -> dict[str, object]:
    """Create the vector-label v5 PDF, PNG preview, and sanitized manifest."""

    base_pdf = _require_file(base_pdf)
    base_manifest = _require_file(base_manifest)
    font_file = _require_file(font_file)
    _load_base_manifest(base_manifest)
    if dpi <= 0:
        raise ValueError("DPI must be positive")
    output = _prepare_output_dir(output_dir)
    pdf_path = output / f"{OUTPUT_STEM}.pdf"
    png_path = output / f"{OUTPUT_STEM}.png"

    document = fitz.open(base_pdf)
    try:
        if document.page_count != 1:
            raise ValueError("Formal Figure 5 base must contain one page")
        page = document[0]
        if not (
            math.isclose(page.rect.width, PAGE_POINTS[0], abs_tol=0.02)
            and math.isclose(page.rect.height, PAGE_POINTS[1], abs_tol=0.02)
        ):
            raise ValueError(f"Unexpected Figure 5 page size: {page.rect}")
        spans = [_find_unique_span(page, text) for text in OLD_LABELS]
        old_rects = [fitz.Rect(span["bbox"]) for span in spans]
        if not all(_inside_allowed_rect(rect) for rect in old_rects):
            raise ValueError("Figure 5 base labels moved outside the v5 correction areas")
        for rect in old_rects:
            page.add_redact_annot(rect, fill=(1, 1, 1))
        page.apply_redactions(images=0, graphics=0, text=0)

        font_name = "ArialBoldFigure5Correction"
        page.insert_font(fontname=font_name, fontfile=str(font_file))
        color = _rgb_from_int(int(spans[0]["color"]))
        new_rects = [
            _insert_centered_text(
                page,
                center_x=(old_rects[0].x0 + old_rects[0].x1) / 2,
                baseline_y=float(spans[0]["origin"][1]),
                text=NEW_LABELS[0],
                font_name=font_name,
                font_path=font_file,
                fontsize=float(spans[0]["size"]),
                color=color,
            ),
            _insert_centered_text(
                page,
                center_x=(old_rects[1].x0 + old_rects[1].x1) / 2,
                baseline_y=float(spans[1]["origin"][1]),
                text=NEW_LABELS[1],
                font_name=font_name,
                font_path=font_file,
                fontsize=float(spans[1]["size"]),
                color=color,
            ),
            _insert_centered_text(
                page,
                center_x=502.666,
                baseline_y=float(spans[2]["origin"][1]),
                text=NEW_LABELS[2],
                font_name=font_name,
                font_path=font_file,
                fontsize=11.0,
                color=color,
            ),
        ]
        if not all(_inside_allowed_rect(fitz.Rect(rect)) for rect in new_rects):
            raise ValueError("Corrected Figure 5 labels do not fit their panel areas")
        document.save(pdf_path, garbage=4, deflate=True, no_new_id=True)
    finally:
        document.close()

    with fitz.open(pdf_path) as result:
        text = result[0].get_text("text").replace("\u00a0", " ")
    if not all(label in text for label in NEW_LABELS) or any(
        label.strip() in text for label in OLD_LABELS
    ):
        raise ValueError("Figure 5 v5 text replacement did not apply cleanly")
    equivalence = _outside_label_equivalence(base_pdf, pdf_path)
    if not equivalence["passed"]:
        raise ValueError("Figure 5 changed outside the three label areas")
    _render_png(pdf_path, png_path, dpi)

    summary: dict[str, object] = {
        "analysis": "main_figure_5_vector_labels",
        "scientific_label_release": "v5",
        "base_contract": {"schema": BASE_SCHEMA, "figure": BASE_FIGURE},
        "label_mapping": dict(zip((item.strip() for item in OLD_LABELS), NEW_LABELS)),
        "page_points": list(PAGE_POINTS),
        "preview_dpi": dpi,
        "files": {"pdf": pdf_path.name, "png": png_path.name},
        "outside_label_equivalence": equivalence,
    }
    manifest_path = output / "vector_rebuild_manifest.json"
    manifest_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-pdf", type=Path, required=True)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument(
        "--font-file",
        type=Path,
        required=True,
        help=(
            "bold font file for the three corrected labels; Arial Bold matches "
            "the current scientific-label page"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        json.dumps(
            build_vector_figure(
                args.base_pdf,
                args.base_manifest,
                args.font_file,
                args.output_dir,
                dpi=args.dpi,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
