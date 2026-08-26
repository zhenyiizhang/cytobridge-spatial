#!/usr/bin/env python3
"""Place exact-old-code Fig. 4e fields into the original Illustrator layout."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import fitz
import numpy as np


EXPECTED_AI_SHA256 = "340a5ed88dc911d6923bc6b21cf1ceb39fdbef16edf2e325822a5b422045cbc2"
EXPECTED_SOURCES_AUDIT_SHA256 = ""
EXPECTED_ANNOTATION_AUDIT_SHA256 = "ad4fad73c72c311f5efa4fab4bbdebbf56f0975830c5fbe497e5fb39288e7cd5"

AI_CROP = fitz.Rect(286.0, 462.0, 595.2760009765625, 841.8900146484375)
OUTER_BORDER = fitz.Rect(286.384, 497.302, 590.156, 832.979)
PANEL_RECTS = {
    "gene_full": fitz.Rect(293.2336730957031, 518.6785888671875, 424.50286865234375, 618.0916748046875),
    "gene_interaction": fitz.Rect(450.05767822265625, 513.8720703125, 586.3214111328125, 617.067626953125),
    "physical_full": fitz.Rect(289.7364807128906, 652.4078369140625, 429.0517272949219, 757.9143676757812),
    "physical_interaction": fitz.Rect(451.6594543457031, 652.6712036132812, 583.97119140625, 752.8738403320312),
}
PANEL_FILES = {
    "gene_full": "Fig4e_gene_full_latest52D_m1024_exact_old_notebook__axes_only.pdf",
    "gene_interaction": "Fig4e_gene_interaction_latest52D_m1024_exact_old_notebook__axes_only.pdf",
    "physical_full": "Fig4e_physical_full_latest52D_m1024_exact_old_notebook__axes_only.pdf",
    "physical_interaction": "Fig4e_physical_interaction_latest52D_m1024_exact_old_notebook__axes_only.pdf",
}
AI_STROKE_PARITY_PANEL_FILES = {
    panel: name.replace("__axes_only.pdf", "__AI_ready_stroke_parity.pdf")
    for panel, name in PANEL_FILES.items()
}

# Replace the complete historical numerical/annotation field in one operation.
# This prevents old arrows and coloured callout rules from leaking around the
# replacement xrefs.  The title above and the seven-class legend below remain
# the original Illustrator objects.
NUMERICAL_FIELD_KNOCKOUT = fitz.Rect(286.5, 497.4, 589.5, 773.3)

# Illustrator text bboxes recovered from Figure_mouse1.ai.  These six labels
# are reconstituted after the complete-field knockout with the original Arial
# family, sizes, wording, and horizontal alignment.
AI_LABELS = (
    ("Gene space", fitz.Rect(406.1606, 497.5239, 473.5286, 514.1559), 12.0, True),
    ("Full", fitz.Rect(340.9166, 618.0519, 360.2486, 634.0239), 12.0, False),
    ("Interaction", fitz.Rect(495.7286, 618.0519, 551.7566, 634.0239), 12.0, False),
    ("Physical space", fitz.Rect(396.2487, 634.0719, 482.3006, 650.7039), 12.0, True),
    ("Full", fitz.Rect(340.8087, 756.6039, 360.1407, 772.5759), 12.0, False),
    ("Interaction", fitz.Rect(484.1487, 756.6039, 540.1766, 772.5759), 12.0, False),
)
ARIAL_REGULAR = Path("/System/Library/Fonts/Supplemental/Arial.ttf")
ARIAL_BOLD = Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf")

BLUE_CALLOUT = (0.58, 0.808, 0.898)
SALMON_CALLOUT = (0.914, 0.635, 0.639)
CALLOUT_TEXT = (34 / 255, 30 / 255, 31 / 255)
CALLOUTS = (
    (
        "Developmental gradient",
        fitz.Rect(319.342, 606.454, 426.595, 619.563),
        fitz.Point(320.3418, 615.1752),
        9.14134,
        BLUE_CALLOUT,
    ),
    (
        "Interaction drive",
        fitz.Rect(512.302, 604.824, 586.536, 617.933),
        fitz.Point(513.3018, 613.5453),
        9.14134,
        SALMON_CALLOUT,
    ),
    (
        "Tissue expansion",
        fitz.Rect(351.366, 745.646, 431.114, 758.755),
        fitz.Point(352.3667, 754.3675),
        9.14134,
        BLUE_CALLOUT,
    ),
    (
        "Cortical plate consolidation",
        fitz.Rect(459.291, 744.476, 583.930, 758.020),
        fitz.Point(460.2905, 753.4994),
        9.35260,
        SALMON_CALLOUT,
    ),
)
TISSUE_LABELS = (
    ("EX Neurons", fitz.Point(329.9223, 516.9856)),
    ("RG/IP", fitz.Point(327.0063, 587.1976)),
    ("EX Neurons", fitz.Point(425.4778, 531.6042)),
    ("RG/IP", fitz.Point(488.9458, 575.5482)),
)

# Exact Illustrator arrow paths and heads.  The annotation audit supplies the
# final tail/tip pair; a rigid 2-D transform maps each original glyph to that
# pair, preserving its path curvature, stroke width, head shape, and length.
ARROW_GLYPHS = {
    "gf_1": {
        "panel": "gene_full", "color": BLUE_CALLOUT,
        "tail": (328.507, 575.549), "tip": (361.556, 549.086),
        "path": ("line", ((355.9544, 553.5715), (328.5074, 575.5485))),
        "head": ((351.6989, 550.5910), (361.5559, 549.0860), (357.9319, 558.3760)),
    },
    "gf_2": {
        "panel": "gene_full", "color": BLUE_CALLOUT,
        "tail": (386.598, 543.003), "tip": (424.281, 532.650),
        "path": ("line", ((417.3607, 534.5507), (386.5977, 543.0027))),
        "head": ((414.6328, 530.1290), (424.2808, 532.6500), (417.2748, 539.7451)),
    },
    "gf_3": {
        "panel": "gene_full", "color": BLUE_CALLOUT,
        "tail": (361.717, 583.344), "tip": (404.897, 580.995),
        "path": ("line", ((397.7318, 581.3849), (361.7168, 583.3439))),
        "head": ((396.0041, 576.4852), (404.8971, 580.9952), (396.5461, 586.4432)),
    },
    "gi_1": {
        "panel": "gene_interaction", "color": SALMON_CALLOUT,
        "tail": (530.302, 517.612), "tip": (497.563, 525.576),
        "path": ("bezier", ((503.2120, 521.1465), (509.0990, 519.0545), (519.0280, 517.6575), (530.3020, 517.6125))),
        "head": ((507.4953, 524.6876), (497.5633, 525.5756), (501.7593, 516.5296)),
    },
    "gi_2": {
        "panel": "gene_interaction", "color": SALMON_CALLOUT,
        "tail": (484.891, 554.346), "tip": (452.152, 562.309),
        "path": ("bezier", ((457.8008, 557.8801), (463.6878, 555.7881), (473.6168, 554.3911), (484.8908, 554.3461))),
        "head": ((462.0841, 561.4212), (452.1521, 562.3092), (456.3481, 553.2632)),
    },
    "gi_3": {
        "panel": "gene_interaction", "color": SALMON_CALLOUT,
        "tail": (483.079, 591.080), "tip": (450.340, 599.043),
        "path": ("bezier", ((455.9890, 594.6138), (461.8760, 592.5219), (471.8050, 591.1248), (483.0790, 591.0798))),
        "head": ((460.2723, 598.1548), (450.3403, 599.0428), (454.5363, 589.9968)),
    },
    "pf_1": {
        "panel": "physical_full", "color": BLUE_CALLOUT,
        "tail": (323.396, 692.993), "tip": (327.007, 671.314),
        "path": ("line", ((325.8275, 678.3924), (323.3955, 692.9934))),
        "head": ((320.6692, 679.0121), (327.0072, 671.3141), (330.5062, 680.6511)),
    },
    "pf_2": {
        "panel": "physical_full", "color": BLUE_CALLOUT,
        "tail": (398.287, 741.346), "tip": (393.496, 683.638),
        "path": ("line", ((394.0894, 690.7894), (398.2874, 741.3464))),
        "head": ((389.2409, 692.6561), (393.4959, 683.6381), (399.1789, 691.8311)),
    },
    "pi_1": {
        "panel": "physical_interaction", "color": SALMON_CALLOUT,
        "tail": (476.123, 663.131), "tip": (494.587, 672.737),
        "path": ("line", ((488.2205, 669.4249), (476.1235, 663.1309))),
        "head": ((489.2276, 664.3280), (494.5866, 672.7370), (484.6246, 673.1750)),
    },
    "pi_2": {
        "panel": "physical_interaction", "color": SALMON_CALLOUT,
        "tail": (523.301, 686.542), "tip": (503.995, 678.767),
        "path": ("line", ((510.6524, 681.4479), (523.3014, 686.5419))),
        "head": ((510.1432, 686.6183), (503.9952, 678.7673), (513.8682, 677.3673)),
    },
    "pi_3": {
        "panel": "physical_interaction", "color": SALMON_CALLOUT,
        "tail": (438.370, 702.528), "tip": (456.833, 712.134),
        "path": ("line", ((450.4673, 708.8217), (438.3703, 702.5277))),
        "head": ((451.4745, 703.7248), (456.8335, 712.1338), (446.8715, 712.5718)),
    },
    "pi_4": {
        "panel": "physical_interaction", "color": SALMON_CALLOUT,
        "tail": (485.548, 725.939), "tip": (466.242, 718.164),
        "path": ("line", ((472.8992, 720.8447), (485.5482, 725.9387))),
        "head": ((472.3901, 726.0151), (466.2421, 718.1641), (476.1151, 716.7642)),
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def freeze_tree(root: Path) -> None:
    for path in root.rglob("*"):
        path.chmod(0o444 if path.is_file() else 0o555)
    root.chmod(0o555)


def fit_inside(target: fitz.Rect, source: fitz.Rect) -> fitz.Rect:
    scale = min(target.width / source.width, target.height / source.height)
    width = source.width * scale
    height = source.height * scale
    x0 = target.x0 + (target.width - width) / 2.0
    y0 = target.y0 + (target.height - height) / 2.0
    return fitz.Rect(x0, y0, x0 + width, y0 + height)


def rigid_arrow_transform(glyph: dict, final_tail: np.ndarray, final_tip: np.ndarray):
    original_tail = np.asarray(glyph["tail"], dtype=float)
    original_tip = np.asarray(glyph["tip"], dtype=float)
    source = original_tip - original_tail
    target = np.asarray(final_tip, dtype=float) - np.asarray(final_tail, dtype=float)
    if not np.isclose(np.linalg.norm(source), np.linalg.norm(target), rtol=0, atol=1e-6):
        raise RuntimeError("Annotation audit changed an Illustrator arrow length.")
    source_angle = np.arctan2(source[1], source[0])
    target_angle = np.arctan2(target[1], target[0])
    angle = target_angle - source_angle
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
        dtype=float,
    )

    def transform(point):
        value = np.asarray(point, dtype=float)
        mapped = np.asarray(final_tail, dtype=float) + rotation @ (value - original_tail)
        return fitz.Point(float(mapped[0]), float(mapped[1]))

    return transform, float(np.degrees(angle))


def draw_annotation_arrow(page: fitz.Page, glyph: dict, final_tail, final_tip) -> float:
    transform, angle_degrees = rigid_arrow_transform(
        glyph, np.asarray(final_tail, dtype=float), np.asarray(final_tip, dtype=float)
    )
    path_kind, path_points = glyph["path"]
    shape = page.new_shape()
    mapped_path = [transform(point) for point in path_points]
    if path_kind == "line":
        shape.draw_line(mapped_path[0], mapped_path[1])
    elif path_kind == "bezier":
        shape.draw_bezier(*mapped_path)
    else:
        raise RuntimeError(f"Unsupported Illustrator arrow path: {path_kind}")
    shape.finish(color=glyph["color"], width=5.0, lineCap=0, lineJoin=0)
    shape.commit(overlay=True)

    head = page.new_shape()
    mapped_head = [transform(point) for point in glyph["head"]]
    head.draw_polyline(mapped_head + [mapped_head[0]])
    head.finish(color=glyph["color"], fill=glyph["color"], width=0)
    head.commit(overlay=True)
    return angle_degrees


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-ai", type=Path, required=True)
    parser.add_argument("--sources-dir", type=Path, required=True)
    parser.add_argument("--annotation-audit", type=Path, required=True)
    parser.add_argument(
        "--source-mode",
        choices=("axes_scaled", "AI_stroke_parity"),
        default="axes_scaled",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()

    original_ai = args.original_ai.expanduser().resolve()
    sources_dir = args.sources_dir.expanduser().resolve()
    source_audit = sources_dir / "render_audit.json"
    if args.source_mode == "AI_stroke_parity":
        axes_dir = sources_dir / "AI_ready_stroke_parity_sources"
        selected_panel_files = AI_STROKE_PARITY_PANEL_FILES
    else:
        axes_dir = sources_dir / "axes_only_for_AI_assembly"
        selected_panel_files = PANEL_FILES
    annotation_audit_path = args.annotation_audit.expanduser().resolve()
    if sha256(original_ai) != EXPECTED_AI_SHA256:
        raise RuntimeError("Original Illustrator identity failed.")
    source_audit_data = json.loads(source_audit.read_text(encoding="utf-8"))
    if (
        source_audit_data.get("status") != "PASS"
        or source_audit_data.get("style", {}).get("no_manual_reimplementation") is not True
        or source_audit_data.get("calculation", {}).get("interaction_m") != 1024
    ):
        raise RuntimeError("Exact-old-code source audit is not passing.")
    if (
        args.source_mode == "AI_stroke_parity"
        and "AI_composition_behavior" not in source_audit_data.get("style", {})
    ):
        raise RuntimeError("Source audit lacks Illustrator stroke-parity evidence.")
    if sha256(annotation_audit_path) != EXPECTED_ANNOTATION_AUDIT_SHA256:
        raise RuntimeError("Fig. 4e annotation audit identity failed.")
    annotation_audit = json.loads(annotation_audit_path.read_text(encoding="utf-8"))
    if (
        annotation_audit.get("status") != "PASS"
        or annotation_audit.get("calculation_contract", {}).get("interaction_m") != 1024
        or any(
            value.get("status") != "PASS"
            for value in annotation_audit.get("manuscript_messages", {}).values()
        )
    ):
        raise RuntimeError("Fig. 4e biological annotation audit is not passing.")
    panel_paths = {key: axes_dir / name for key, name in selected_panel_files.items()}
    for path in panel_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    out_dir = args.output_dir.expanduser().resolve()
    if args.freeze and out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite immutable output: {out_dir}")
    provenance_dir = out_dir / "provenance"
    out_dir.mkdir(parents=True)
    provenance_dir.mkdir()

    original = fitz.open(original_ai)
    if len(original) != 1:
        raise RuntimeError("Expected a one-page PDF-compatible Illustrator file.")
    full = fitz.open()
    full_page = full.new_page(width=original[0].rect.width, height=original[0].rect.height)
    full_page.show_pdf_page(full_page.rect, original, 0)
    full_page.draw_rect(
        NUMERICAL_FIELD_KNOCKOUT,
        color=(1, 1, 1),
        fill=(1, 1, 1),
        width=0,
        overlay=True,
    )

    placement_audit = {}
    opened_panels = []
    for panel, target in PANEL_RECTS.items():
        panel_doc = fitz.open(panel_paths[panel])
        opened_panels.append(panel_doc)
        source_rect = panel_doc[0].rect
        placed = fit_inside(target, source_rect)
        if args.source_mode == "AI_stroke_parity" and not np.isclose(
            placed.width / source_rect.width, 1.0, rtol=0, atol=2e-6
        ):
            raise RuntimeError(
                f"AI stroke-parity source {panel} would be rescaled: "
                f"{placed.width / source_rect.width}"
            )
        full_page.show_pdf_page(placed, panel_doc, 0, keep_proportion=True, overlay=True)
        placement_audit[panel] = {
            "source_pdf": str(panel_paths[panel]),
            "source_sha256": sha256(panel_paths[panel]),
            "source_page_points": [source_rect.width, source_rect.height],
            "target_original_AI_points": [target.x0, target.y0, target.x1, target.y1],
            "placed_original_AI_points": [placed.x0, placed.y0, placed.x1, placed.y1],
            "uniform_scale": placed.width / source_rect.width,
            "rotation": False,
            "anisotropic_stretch": False,
            "warp": False,
            "source_mode": args.source_mode,
        }

    for font_path in (ARIAL_REGULAR, ARIAL_BOLD):
        if not font_path.is_file():
            raise FileNotFoundError(font_path)
    full_page.insert_font(fontname="ArialFigure", fontfile=str(ARIAL_REGULAR))
    full_page.insert_font(fontname="ArialFigureBold", fontfile=str(ARIAL_BOLD))
    label_audit = []
    for label, source_bbox, fontsize, bold in AI_LABELS:
        # Slightly expand the recovered Illustrator bbox for PyMuPDF's line
        # metrics while preserving the exact centre and nominal font size.
        text_rect = fitz.Rect(
            source_bbox.x0 - 3.0,
            source_bbox.y0 - 2.0,
            source_bbox.x1 + 3.0,
            source_bbox.y1 + 3.0,
        )
        result = full_page.insert_textbox(
            text_rect,
            label,
            fontname="ArialFigureBold" if bold else "ArialFigure",
            fontsize=fontsize,
            color=(0, 0, 0),
            align=fitz.TEXT_ALIGN_CENTER,
            overlay=True,
        )
        if result < 0:
            raise RuntimeError(f"Could not place Illustrator label {label!r}: {result}")
        label_audit.append(
            {
                "text": label,
                "original_AI_bbox": [
                    source_bbox.x0,
                    source_bbox.y0,
                    source_bbox.x1,
                    source_bbox.y1,
                ],
                "font": "Arial Bold" if bold else "Arial",
                "fontsize_points": fontsize,
                "alignment": "center",
            }
        )

    arrow_style_audit = []
    final_arrow_records = {
        arrow["id"]: arrow
        for arrows in annotation_audit["final_arrow_geometry"].values()
        for arrow in arrows
    }
    if set(final_arrow_records) != set(ARROW_GLYPHS):
        raise RuntimeError("Annotation audit and Illustrator arrow glyph identities differ.")
    for arrow_id, glyph in ARROW_GLYPHS.items():
        final = final_arrow_records[arrow_id]
        rotation_degrees = draw_annotation_arrow(
            full_page, glyph, final["tail"], final["tip"]
        )
        arrow_style_audit.append(
            {
                "id": arrow_id,
                "panel": glyph["panel"],
                "decision": final["decision"],
                "stroke_width_points": 5.0,
                "original_AI_color": list(glyph["color"]),
                "rigid_annotation_rotation_degrees": rotation_degrees,
                "data_field_rotated": False,
                "data_field_stretched": False,
                "data_field_warped": False,
            }
        )

    for label, point in TISSUE_LABELS:
        full_page.insert_text(
            point,
            label,
            fontname="ArialFigure",
            fontsize=12.0,
            color=(0, 0, 0),
            overlay=True,
        )
    for label, rect, origin, fontsize, color in CALLOUTS:
        full_page.draw_rect(rect, color=color, fill=color, width=0, overlay=True)
        full_page.insert_text(
            origin,
            label,
            fontname="ArialFigureBold",
            fontsize=fontsize,
            color=CALLOUT_TEXT,
            overlay=True,
        )
    # Restore the exact original two-point outer rule after the white knockouts.
    full_page.draw_rect(OUTER_BORDER, color=(35 / 255, 24 / 255, 21 / 255), width=2.0, overlay=True)

    full_path = out_dir / "Fig4e_MOSTA_latest52D_m1024_exact_old_code_full_AI_page.pdf"
    full.save(full_path, garbage=4, deflate=True)

    cropped = fitz.open()
    crop_page = cropped.new_page(width=AI_CROP.width, height=AI_CROP.height)
    crop_page.show_pdf_page(crop_page.rect, full, 0, clip=AI_CROP, keep_proportion=True)
    pdf_path = out_dir / "Fig4e_MOSTA_latest52D_m1024_exact_old_code_AI_layout.pdf"
    cropped.save(pdf_path, garbage=4, deflate=True)
    png_path = out_dir / "Fig4e_MOSTA_latest52D_m1024_exact_old_code_AI_layout_600dpi.png"
    pix = crop_page.get_pixmap(matrix=fitz.Matrix(600 / 72, 600 / 72), alpha=False)
    pix.save(png_path)
    svg_path = out_dir / "Fig4e_MOSTA_latest52D_m1024_exact_old_code_AI_layout.svg"
    svg_path.write_text(crop_page.get_svg_image(text_as_path=False), encoding="utf-8")

    for panel_doc in opened_panels:
        panel_doc.close()
    original.close()
    full.close()
    cropped.close()

    for source, destination in (
        (original_ai, provenance_dir / "style_authority__Figure_mouse1.ai"),
        (source_audit, provenance_dir / "source_render_audit.json"),
        (annotation_audit_path, provenance_dir / "annotation_semantics_audit.json"),
        (Path(__file__).resolve(), provenance_dir / Path(__file__).name),
    ):
        shutil.copy2(source, destination)

    audit = {
        "schema_version": 1,
        "status": "PASS",
        "dataset": "MOSTA",
        "panel": "Fig4e",
        "calculation_source": {
            "exact_old_code_sources_audit": str(source_audit),
            "exact_old_code_sources_audit_sha256": sha256(source_audit),
            "interaction_m": 1024,
            "accepted_latest_model": True,
            "annotation_semantics_audit": str(annotation_audit_path),
            "annotation_semantics_audit_sha256": EXPECTED_ANNOTATION_AUDIT_SHA256,
        },
        "style_source": {
            "original_AI": str(original_ai),
            "original_AI_sha256": EXPECTED_AI_SHA256,
            "numerical_panel_source_mode": args.source_mode,
            "AI_stroke_parity_behavior": (
                "scatter markers receive exact uniform raster geometry scale; vector stream linewidth retains notebook point width; stream arrow glyph geometry receives the same uniform scale"
                if args.source_mode == "AI_stroke_parity"
                else "entire axes-only PDF receives uniform AI scale"
            ),
            "retained": [
                "page geometry",
                "panel hierarchy",
                "seven-class two-column legend",
                "outer border",
            ],
            "reconstituted_from_original_AI_geometry": [
                "Gene space heading",
                "Full/Interaction top column labels",
                "Physical space heading",
                "Full/Interaction bottom column labels",
            ],
            "labels": label_audit,
            "numerical_field_knockout_original_AI_points": [
                NUMERICAL_FIELD_KNOCKOUT.x0,
                NUMERICAL_FIELD_KNOCKOUT.y0,
                NUMERICAL_FIELD_KNOCKOUT.x1,
                NUMERICAL_FIELD_KNOCKOUT.y1,
            ],
            "narrative_annotations": {
                "status": "restored after biological interpretation PASS",
                "callout_boxes_text_and_tissue_labels": "exact original Illustrator geometry, Arial family, size, wording, and colors",
                "arrows": arrow_style_audit,
                "arrow_policy": "retain exact original glyph when direction-supported; otherwise rigidly reorient the annotation glyph to the corrected local field",
            },
        },
        "placements": placement_audit,
        "forbidden_transform_gate": {
            "rotation": False,
            "anisotropic_stretch": False,
            "warp": False,
        },
        "outputs": {
            path.name: {"sha256": sha256(path), "size_bytes": path.stat().st_size}
            for path in (full_path, pdf_path, svg_path, png_path)
        },
    }
    audit_path = out_dir / "assembly_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    checksum_paths = sorted(
        path for path in out_dir.rglob("*")
        if path.is_file() and path.name not in {"SHA256SUMS.json", "COMPLETE"}
    )
    checks = {str(path.relative_to(out_dir)): sha256(path) for path in checksum_paths}
    (out_dir / "SHA256SUMS.json").write_text(
        json.dumps(checks, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (out_dir / "COMPLETE").write_text("COMPLETE\n", encoding="utf-8")
    if args.freeze:
        freeze_tree(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
