#!/usr/bin/env python3
"""Render 40k/50k MOSTA Fig. 4a candidates in the original Illustrator style.

The copied Illustrator PDF is the style oracle.  Its title, E-stage labels,
oblique frame geometry, solid/dashed strokes, generated labels, typography and
artboard placement remain untouched.  Seven transparent point-cloud images are
replaced with accepted package-native global-t0 states.  The legend roster is
updated to all labels actually present while keeping the original one-column
circle-marker grammar.
"""

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


RUN = Path(__file__).resolve().parents[1]
PALETTE_FILE = RUN / "provenance" / "label_to_color.json"
PROV = RUN / "provenance"
QC = RUN / "qc"
AI = PROV / "Figure_mouse1_style_oracle.ai"

AI_SHA256 = "340a5ed88dc911d6923bc6b21cf1ceb39fdbef16edf2e325822a5b422045cbc2"
ARIAL = Path("/System/Library/Fonts/Supplemental/Arial.ttf")
CROP = fitz.Rect(0.0, 0.0, 595.2760009765625, 192.0)
# Fine-point grammar from the manuscript-style source.  Density now comes from
# the actual package simulation, not from enlarging markers or duplicating
# coordinates.
POINT_SIZE_PT2 = 0.35

# Extracted directly from the colored circle objects in Figure_mouse1.ai.
# These values, including Illustrator color conversion, are the style truth.
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


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--n-initial", type=int, required=True, choices=(40000, 50000))
    return p


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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


def render_crop(page: fitz.Page, out: Path, dpi: float = 220.0) -> None:
    pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72.0, dpi / 72.0), clip=CROP, alpha=False)
    pix.save(out)


def make_contact(original: Path, corrected: Path, out: Path) -> None:
    a = Image.open(original).convert("RGB")
    b = Image.open(corrected).convert("RGB")
    canvas = Image.new("RGB", (a.width + b.width, max(a.height, b.height) + 38), "white")
    canvas.paste(a, (0, 38))
    canvas.paste(b, (a.width, 38))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 10), "Original Illustrator style oracle", fill="black")
    draw.text((a.width + 8, 10), "Corrected global-t0 points in original Illustrator composition", fill="black")
    canvas.save(out)


def main() -> int:
    args = parser().parse_args()
    n_initial = int(args.n_initial)
    input_dir = RUN / "runs" / f"n{n_initial}" / "slice_data"
    summary_file = RUN / "runs" / f"n{n_initial}" / "summary.json"
    out = RUN / "figures" / f"n{n_initial}"
    qa = RUN / "qa" / f"n{n_initial}"
    render_version = f"n{n_initial}_fine_points_isotropic_E15_original_zorder_v3"
    if json.loads((QC / "particle_sensitivity_gate.json").read_text()).get("status") != "PASS":
        raise RuntimeError("Refusing to render: numerical gate is not PASS")
    if sha256(AI) != AI_SHA256:
        raise RuntimeError("Illustrator style oracle hash mismatch")
    if not ARIAL.exists():
        raise RuntimeError(f"Arial font not found at {ARIAL}")

    out.mkdir(parents=True, exist_ok=True)
    qa.mkdir(parents=True, exist_ok=True)
    replacement_dir = qa / f"replacement_pointclouds_rgba_{render_version}"
    replacement_dir.mkdir(parents=True, exist_ok=True)
    canonical_palette = json.loads(PALETTE_FILE.read_text())
    palette = dict(canonical_palette)
    palette.update(LEGACY_AI_PALETTE)

    labels_present: set[str] = set()
    render_panels = prepare_render_panels(input_dir)
    for panel in render_panels:
        a = ad.read_h5ad(input_dir / panel["file"], backed="r")
        labels_present.update(a.obs["Annotation"].astype(str).unique())
        a.file.close()
    legacy_present = [label for label in LEGACY_AI_ORDER if label in labels_present]
    additional_present = [
        label for label in canonical_palette
        if label in labels_present and label not in LEGACY_AI_PALETTE
    ]
    ordered_labels = legacy_present + additional_present
    if set(ordered_labels) != labels_present:
        raise RuntimeError(f"Missing palette labels: {sorted(labels_present - set(ordered_labels))}")

    doc = fitz.open(AI)
    page = doc[0]
    image_lookup = {item[0]: item for item in page.get_images(full=True)}
    placement_lookup = {int(item["xref"]): item for item in page.get_image_info(xrefs=True)}
    replacements = []
    for panel in render_panels:
        xref = int(panel["xref"])
        if xref not in image_lookup:
            raise RuntimeError(f"Expected Illustrator image xref {xref} is missing")
        image_info = image_lookup[xref]
        if int(image_info[2]) != int(panel["width"]) or int(image_info[3]) != int(panel["height"]):
            raise RuntimeError(f"Illustrator image dimensions changed for xref {xref}")
        rects = page.get_image_rects(xref)
        if len(rects) != 1 or not np.allclose(np.asarray(rects[0]), np.asarray(panel["bbox"]), atol=1e-4):
            raise RuntimeError(f"Illustrator placement changed for xref {xref}: {rects}")
        png = replacement_dir / f"time_{str(panel['time']).replace('.', 'p')}_xref{xref}.png"
        render_transparent_points(input_dir, panel, palette, ordered_labels, png)
        placement_transform = placement_lookup[xref]["transform"]
        uses_original_placement = panel["placement_mode"] == "exact_original_ai_bbox"
        preflipped_for_ai_matrix = bool(float(placement_transform[3]) < 0.0)
        if preflipped_for_ai_matrix:
            with Image.open(png) as replacement_image:
                replacement_image.transpose(Image.Transpose.FLIP_TOP_BOTTOM).save(png)

        original_base = fitz.Pixmap(doc, xref)
        original_smask_xref = int(image_info[1])
        original_rgba = (
            fitz.Pixmap(original_base, fitz.Pixmap(doc, original_smask_xref))
            if original_smask_xref > 0
            else original_base
        )
        original_samples = np.frombuffer(original_rgba.samples, dtype=np.uint8).reshape(
            original_rgba.height, original_rgba.width, original_rgba.n
        )
        original_alpha = original_samples[:, :, -1] if original_rgba.alpha else np.full(
            (original_rgba.height, original_rgba.width), 255, dtype=np.uint8
        )
        replacement_alpha = np.asarray(Image.open(png).convert("RGBA").getchannel("A"))
        if uses_original_placement:
            page.replace_image(xref, filename=str(png))
        else:
            form_xref = int(image_info[9])
            if form_xref <= 0:
                raise RuntimeError(f"Expected Illustrator Form XObject parent for xref {xref}")
            page.replace_image(xref, filename=str(png))
            update_original_form_placement(doc, page, form_xref, panel["placement_bbox"])
        visible = replacement_alpha > 10
        if not np.any(visible):
            raise RuntimeError(f"Empty replacement alpha for {panel['file']}")
        yy, xx = np.where(visible)
        placement_width = float(panel["placement_bbox"][2] - panel["placement_bbox"][0])
        placement_height = float(panel["placement_bbox"][3] - panel["placement_bbox"][1])
        content_width_pt = placement_width * float(xx.max() + 1 - xx.min()) / float(replacement_alpha.shape[1])
        content_height_pt = placement_height * float(yy.max() + 1 - yy.min()) / float(replacement_alpha.shape[0])
        replacements.append({
            "time": panel["time"],
            "origin": panel["origin"],
            "source_anchor_time": panel["anchor"],
            "xref": xref,
            "original_pixel_size": [panel["width"], panel["height"]],
            "replacement_pixel_size": [panel["render_width"], panel["render_height"]],
            "original_illustrator_bbox_points": panel["bbox"],
            "replacement_illustrator_bbox_points": panel["placement_bbox"],
            "placement_mode": panel["placement_mode"],
            "illustrator_form_xref": int(image_info[9]),
            "z_order_contract": "original Illustrator Form XObject content-stream order retained",
            "illustrator_transform": [float(value) for value in placement_transform],
            "preflipped_for_negative_ai_y_matrix": preflipped_for_ai_matrix,
            "visible_content_points": {"width": content_width_pt, "height": content_height_pt},
            "alpha_density_calibration": {
                "original_alpha_mean": float(original_alpha.mean() / 255.0),
                "replacement_alpha_mean": float(replacement_alpha.mean() / 255.0),
                "original_coverage_alpha_gt_128": float(np.mean(original_alpha > 128)),
                "replacement_coverage_alpha_gt_128": float(np.mean(replacement_alpha > 128)),
            },
            "replacement": str(png.relative_to(RUN)),
            "replacement_sha256": sha256(png),
        })

    replace_legend(page, palette, ordered_labels)
    fullpage_qa_pdf = qa / f"Figure_mouse1_fullpage_Fig4a_replaced_{render_version}_QA_ONLY.pdf"
    doc.save(fullpage_qa_pdf, garbage=4, deflate=True)
    doc.close()

    corrected_full = fitz.open(fullpage_qa_pdf)
    original_full = fitz.open(AI)
    original_png = qa / f"original_ai_fig4a_crop_{render_version}.png"
    corrected_png = qa / f"corrected_ai_fig4a_crop_{render_version}.png"
    render_crop(original_full[0], original_png)
    render_crop(corrected_full[0], corrected_png)

    crop_doc = fitz.open()
    crop_page = crop_doc.new_page(width=CROP.width, height=CROP.height)
    crop_page.show_pdf_page(crop_page.rect, corrected_full, 0, clip=CROP)
    pdf = out / f"Fig4a_MOSTA_global_t0_original_AI_equivalent_{render_version}.pdf"
    crop_doc.save(pdf, garbage=4, deflate=True)
    svg = out / f"Fig4a_MOSTA_global_t0_original_AI_equivalent_{render_version}.svg"
    svg.write_text(crop_doc[0].get_svg_image(text_as_path=False))
    png = out / f"Fig4a_MOSTA_global_t0_original_AI_equivalent_{render_version}.png"
    pix = crop_doc[0].get_pixmap(matrix=fitz.Matrix(320.0 / 72.0, 320.0 / 72.0), alpha=False)
    pix.save(png)

    a = np.asarray(Image.open(original_png).convert("RGB"), dtype=np.int16)
    b = np.asarray(Image.open(corrected_png).convert("RGB"), dtype=np.int16)
    mask = np.zeros(a.shape[:2], dtype=bool)
    scale = 220.0 / 72.0
    for panel in render_panels:
        x0, y0, x1, y1 = panel["placement_bbox"]
        ix0, iy0 = max(0, int((x0 - 2.5) * scale)), max(0, int((y0 - 2.5) * scale))
        ix1, iy1 = min(mask.shape[1], int((x1 + 2.5) * scale) + 1), min(mask.shape[0], int((y1 + 2.5) * scale) + 1)
        mask[iy0:iy1, ix0:ix1] = True
    mask[:, int(492.0 * scale) :] = True
    diff = np.abs(a - b).max(axis=2)
    outside = diff[~mask]
    height_by_time = {
        float(item["time"]): float(item["visible_content_points"]["height"])
        for item in replacements
    }
    e15_height_ratios = {
        "E15_over_E14": height_by_time[2.5] / height_by_time[1.5],
        "E15_over_E14p5": height_by_time[2.5] / height_by_time[2.0],
    }
    no_false_contraction = min(e15_height_ratios.values()) >= 0.98
    unchanged_style = float(np.mean(outside > 1)) < 1e-4
    style_qa = {
        "schema_version": 1,
        "status": "PASS" if unchanged_style and no_false_contraction else "FAIL",
        "comparison": "Original Illustrator crop vs corrected crop outside the seven replaced point-cloud rectangles and updated legend area",
        "outside_replacement_mean_abs_maxchannel": float(outside.mean()),
        "outside_replacement_fraction_gt_1": float(np.mean(outside > 1)),
        "outside_replacement_max": int(outside.max()),
        "unaltered_style_elements": [
            "artboard and crop geometry",
            "panel title and label a",
            "seven oblique plane frames",
            "solid/dashed frame strokes",
            "E-stage labels",
            "tilted Generated labels",
            "all relative panel placements",
        ],
        "intentional_changes": [
            "seven point-cloud image objects replaced with accepted package-native states",
            "legend roster expanded from the legacy subset to all 33 labels present; original one-column marker grammar retained",
            "E15 point-cloud object expanded symmetrically in x at fixed height so its corrected wider aspect does not create a false contraction",
            "E15 placement matrix updated inside its original Illustrator Form XObject so the original E15/E15.5 z-order is retained",
        ],
        "geometry_contract": {
            "rotation": False,
            "anisotropic_stretch": False,
            "spatial_warp": False,
            "cropped_cells": False,
            "replacement_placement_matrices": "exact original Illustrator image bounding boxes except E15 isotropic x expansion within its dashed frame",
            "z_order": "original Illustrator Form XObject content-stream order retained; no overlay insertion",
            "within_image_axis": "equal aspect; isotropic fit",
        },
        "biological_message_gate": {
            "status": "PASS" if no_false_contraction else "FAIL",
            "criterion": "E15 visible point-cloud height must be at least 98% of both E14 and E14.5 after isotropic placement",
            "visible_content_height_points": height_by_time,
            "ratios": e15_height_ratios,
            "numerical_source": "qc/E15_no_contraction_gate.csv proves corrected raw E15 q99/covariance area grows versus E14 and is comparable to E14.5",
        },
    }
    style_qa["render_version"] = render_version
    style_qa["point_size_pt2"] = POINT_SIZE_PT2
    (qa / f"ai_style_equivalence_qa_{render_version}.json").write_text(json.dumps(style_qa, indent=2) + "\n")
    make_contact(original_png, corrected_png, qa / f"original_vs_corrected_ai_contact_{render_version}.png")

    manifest = {
        "schema_version": 1,
        "panel": "Fig4a",
        "status": "complete" if style_qa["status"] == "PASS" else "failed_style_qa",
        "calculation_gate": "qc/particle_sensitivity_gate.json",
        "style_oracle": str(AI.relative_to(RUN)),
        "style_oracle_sha256": AI_SHA256,
        "style_oracle_creator": "Adobe Illustrator 30.2 (Macintosh)",
        "trajectory_mode": "global_t0_extrapolation",
        "n_initial_particles": n_initial,
        "server_run": f"/data/cytobridge/projects/CytoBridge-ST-1104/runs/mosta-paper-figures/fig4a-global-t0-n{n_initial}-2b3c79e-seed42-20260825-v1",
        "server_summary_sha256": sha256(summary_file),
        "package_commit": "2b3c79eff3face7c4dd33de24d45384b9dbd8a84",
        "render_version": render_version,
        "point_size_pt2": POINT_SIZE_PT2,
        "palette_contract": {
            "legacy_labels": "exact RGB extracted from Figure_mouse1.ai circle objects",
            "legacy_order": LEGACY_AI_ORDER,
            "additional_labels": additional_present,
            "additional_label_source": "canonical MOSTA label_to_color.json because these labels are absent from the Fig. 4a Illustrator legend",
            "canonical_vs_ai_differences": {
                label: {"canonical": canonical_palette[label], "illustrator": LEGACY_AI_PALETTE[label]}
                for label in LEGACY_AI_ORDER
                if canonical_palette.get(label, "").lower() != LEGACY_AI_PALETTE[label].lower()
            },
        },
        "restart_from_preceding_observed_stage": False,
        "spatial_warp": False,
        "false_contraction_visual_gate": style_qa["biological_message_gate"],
        "replacements": replacements,
        "legend_labels": ordered_labels,
        "legend_label_count": len(ordered_labels),
        "outputs": [str(path.relative_to(RUN)) for path in (pdf, svg, png)],
        "fullpage_qa_only": str(fullpage_qa_pdf.relative_to(RUN)),
        "style_qa": str((qa / f"ai_style_equivalence_qa_{render_version}.json").relative_to(RUN)),
        "note": "The full-page QA file still contains legacy b-e values and is not a manuscript deliverable. Only the standalone Fig. 4a crop is panel-ready.",
    }
    (out / f"manifest_{render_version}.json").write_text(json.dumps(manifest, indent=2) + "\n")
    crop_doc.close()
    original_full.close()
    corrected_full.close()
    print(json.dumps({"status": manifest["status"], "outputs": manifest["outputs"], "style_qa": style_qa}, indent=2))
    return 0 if manifest["status"] == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())
