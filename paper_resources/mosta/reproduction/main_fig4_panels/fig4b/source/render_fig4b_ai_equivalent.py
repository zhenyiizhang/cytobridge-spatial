#!/usr/bin/env python3
"""Replace only the three Fig. 4b embryo rasters in the original AI layout."""

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


RUN = Path(__file__).resolve().parents[1]
SERVER = RUN / "server_run_v4"
AI = RUN / "provenance" / "Figure_mouse1_style_oracle.ai"
QA = RUN / "qa"
FIGURES = RUN / "figures"
AI_SHA256 = "340a5ed88dc911d6923bc6b21cf1ceb39fdbef16edf2e325822a5b422045cbc2"
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

ANNOTATION_COLOR = (0.4000000059604645, 0.6710000038146973, 0.7059999704360962)
ARIAL_FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"


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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
    with Image.open(out) as image:
        if image.size != (width, height):
            raise RuntimeError(f"Raster size mismatch for {panel['stage']}: {image.size}")
        rgb = np.asarray(image.convert("RGB"))
    nonwhite = np.any(rgb < 250, axis=2)
    if not nonwhite.any():
        raise RuntimeError(f"Empty replacement raster for {panel['stage']}.")
    yy, xx = np.where(nonwhite)
    bbox = list(map(float, panel["bbox"]))
    placement_width = bbox[2] - bbox[0]
    placement_height = bbox[3] - bbox[1]
    return {
        "time": float(panel["time"]),
        "stage": str(panel["stage"]),
        "n_cells": int(len(table)),
        "n_score_available_cells": int(available.sum()),
        "n_score_unavailable_cells": int((~available).sum()),
        "unavailable_display_policy": "base grey only; no zero imputation",
        "n_cell_types": int(table["cell_type"].nunique()),
        "coordinate_extent": [float(extent[0]), float(extent[1])],
        "coordinate_aspect": float(extent[0] / extent[1]),
        "canvas_pixel_size": [width, height],
        "canvas_aspect": float(width / height),
        "illustrator_bbox_points": bbox,
        "visible_content_points": {
            "width": placement_width * float(xx.max() + 1 - xx.min()) / width,
            "height": placement_height * float(yy.max() + 1 - yy.min()) / height,
        },
        "replacement": str(out.relative_to(RUN)),
        "replacement_sha256": sha256(out),
    }


def render_crop(page: fitz.Page, out: Path, dpi: float) -> None:
    pix = page.get_pixmap(
        matrix=fitz.Matrix(dpi / 72.0, dpi / 72.0),
        clip=CROP,
        alpha=False,
    )
    pix.save(out)


def make_contact(original: Path, corrected: Path, out: Path) -> None:
    left = Image.open(original).convert("RGB")
    right = Image.open(corrected).convert("RGB")
    canvas = Image.new("RGB", (left.width + right.width, max(left.height, right.height) + 36), "white")
    canvas.paste(left, (0, 36))
    canvas.paste(right, (left.width, 36))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 10), "Original Illustrator panel", fill="black")
    draw.text((left.width + 8, 10), "Corrected package-native values in original style", fill="black")
    canvas.save(out)


def main() -> int:
    if sha256(AI) != AI_SHA256:
        raise RuntimeError("Illustrator style-oracle hash mismatch.")
    gate = json.loads((SERVER / "calculation_gate.json").read_text(encoding="utf-8"))
    if gate.get("status") != "PASS":
        raise RuntimeError("Refusing render because calculation gate is not PASS.")
    mapping = pd.read_csv(SERVER / "cell_mapping.csv.gz")
    expected_counts = {0.0: 51365, 0.5: 63533, 1.0: 77369}
    actual_counts = mapping.groupby("time").size().to_dict()
    if actual_counts != expected_counts:
        raise RuntimeError(f"Display cell-count mismatch: {actual_counts}")
    mapping[NORM_COLUMN] = np.nan
    per_panel_bounds: list[dict[str, float | int | str]] = []
    for panel in PANELS:
        time_value = float(panel["time"])
        time_mask = np.isclose(mapping["time"], time_value)
        available_mask = time_mask & mapping["cell_type_score_available"].astype(bool)
        finite = mapping.loc[available_mask, "total_raw"].to_numpy(dtype=np.float64)
        if not len(finite) or not np.isfinite(finite).all():
            raise RuntimeError(f"No finite values for per-panel normalization at t={time_value:g}.")
        low = float(np.percentile(finite, 1.0))
        high = float(np.percentile(finite, 99.0))
        if not high > low:
            raise RuntimeError(f"Degenerate per-panel normalization at t={time_value:g}.")
        mapping.loc[available_mask, NORM_COLUMN] = np.clip(
            (mapping.loc[available_mask, "total_raw"] - low) / (high - low),
            0.0,
            1.0,
        )
        per_panel_bounds.append(
            {
                "time": time_value,
                "stage": str(panel["stage"]),
                "n_finite_cells": int(len(finite)),
                "q1_low": low,
                "q99_high": high,
            }
        )

    replacements_dir = QA / f"replacement_rasters_{RENDER_VERSION}"
    replacements_dir.mkdir(exist_ok=False)
    FIGURES.mkdir(exist_ok=True)
    doc = fitz.open(AI)
    page = doc[0]
    original_text = page.get_text("text")
    image_lookup = {item[0]: item for item in page.get_images(full=True)}
    image_info = {int(item["xref"]): item for item in page.get_image_info(xrefs=True)}
    replacements: list[dict[str, object]] = []
    for panel in PANELS:
        time_value = float(panel["time"])
        subset = mapping.loc[np.isclose(mapping["time"], time_value)].copy()
        png = replacements_dir / f"{panel['stage'].replace('.', 'p')}_xref{panel['xref']}.png"
        record = render_panel(subset, panel, png)
        xref = int(panel["xref"])
        if xref not in image_lookup or xref not in image_info:
            raise RuntimeError(f"Illustrator image xref {xref} is missing.")
        item = image_lookup[xref]
        if (int(item[2]), int(item[3])) != (int(panel["width"]), int(panel["height"])):
            raise RuntimeError(f"Illustrator pixel dimensions changed for xref {xref}.")
        rects = page.get_image_rects(xref)
        if len(rects) != 1 or not np.allclose(
            np.asarray(rects[0]), np.asarray(panel["bbox"]), atol=1e-4
        ):
            raise RuntimeError(f"Illustrator placement changed for xref {xref}: {rects}")
        transform = [float(value) for value in image_info[xref]["transform"]]
        if abs(transform[1]) > 1e-9 or abs(transform[2]) > 1e-9 or transform[0] <= 0 or transform[3] <= 0:
            raise RuntimeError(f"Unexpected rotated/reflected AI transform for xref {xref}: {transform}")
        page.replace_image(xref, filename=str(png))
        record.update(
            {
                "xref": xref,
                "illustrator_transform": transform,
                "slice_origin": str(panel["origin"]),
                "source_anchor_time": float(panel["anchor"]),
                "placement_mode": "exact original Illustrator image bbox and z-order",
                "rotation": False,
                "reflection": False,
                "anisotropic_data_transform": False,
                "spatial_warp": False,
            }
        )
        replacements.append(record)

    if page.get_text("text") != original_text:
        raise RuntimeError("AI text layer changed during raster replacement.")
    add_generated_panel_annotations(page)
    fullpage = QA / f"Figure_mouse1_fullpage_Fig4b_replaced_{RENDER_VERSION}_QA_ONLY.pdf"
    doc.save(fullpage, garbage=4, deflate=True)
    doc.close()

    corrected = fitz.open(fullpage)
    original = fitz.open(AI)
    original_png = QA / f"original_ai_fig4b_crop_{RENDER_VERSION}.png"
    corrected_png = QA / f"corrected_ai_fig4b_crop_{RENDER_VERSION}.png"
    render_crop(original[0], original_png, 320.0)
    render_crop(corrected[0], corrected_png, 320.0)

    crop_doc = fitz.open()
    crop_page = crop_doc.new_page(width=CROP.width, height=CROP.height)
    crop_page.show_pdf_page(crop_page.rect, corrected, 0, clip=CROP)
    # Fig. 4c's large raster begins at page x=316.086 pt and geometrically
    # overlaps the rectangular Fig. 4b crop below its title.  Mask only that
    # adjacent-panel spill in the standalone export; every Fig. 4b object ends
    # left of x=313.25 pt in this y interval.  The full-page QA remains intact.
    crop_page.draw_rect(
        fitz.Rect(315.8 - CROP.x0, 231.0 - CROP.y0, CROP.width, 416.0 - CROP.y0),
        color=None,
        fill=(1.0, 1.0, 1.0),
        overlay=True,
    )
    stem = f"Fig4b_MOSTA_Wnt3a_Fzd7_Lrp6_total_{RENDER_VERSION}"
    pdf = FIGURES / f"{stem}.pdf"
    svg = FIGURES / f"{stem}.svg"
    png = FIGURES / f"{stem}.png"
    crop_doc.save(pdf, garbage=4, deflate=True)
    svg.write_text(crop_doc[0].get_svg_image(text_as_path=False), encoding="utf-8")
    pix = crop_doc[0].get_pixmap(matrix=fitz.Matrix(320.0 / 72.0, 320.0 / 72.0), alpha=False)
    pix.save(png)

    a = np.asarray(Image.open(original_png).convert("RGB"), dtype=np.int16)
    b = np.asarray(Image.open(corrected_png).convert("RGB"), dtype=np.int16)
    if a.shape != b.shape:
        raise RuntimeError("Original/corrected QA renders have different geometry.")
    mask = np.zeros(a.shape[:2], dtype=bool)
    scale = 320.0 / 72.0
    for panel in PANELS:
        x0, y0, x1, y1 = map(float, panel["bbox"])
        x0 -= CROP.x0
        x1 -= CROP.x0
        y0 -= CROP.y0
        y1 -= CROP.y0
        ix0, iy0 = max(0, int(x0 * scale) - 2), max(0, int(y0 * scale) - 2)
        ix1 = min(mask.shape[1], int(x1 * scale) + 3)
        iy1 = min(mask.shape[0], int(y1 * scale) + 3)
        mask[iy0:iy1, ix0:ix1] = True
    diff = np.abs(a - b).max(axis=2)
    outside = diff[~mask]
    style_status = "PASS" if float(np.mean(outside > 1)) < 1e-4 else "FAIL"
    style_qa = {
        "schema_version": 1,
        "status": style_status,
        "comparison": "original AI crop vs corrected crop outside the three replaced embryo image rectangles",
        "outside_replacement_mean_abs_maxchannel": float(outside.mean()),
        "outside_replacement_fraction_gt_1": float(np.mean(outside > 1)),
        "outside_replacement_max": int(outside.max()),
        "unchanged_objects": [
            "panel label b and title",
            "Wnt3a-Fzd7/Lrp6 label",
            "Observed/Generated labels",
            "all original Choroid Plexus, Meninges, and Epidermis annotations and arrows",
            "E12.5/E13.0/E13.5 labels",
            "developmental time arrow",
            "Low-High colorbar and Interaction Score label",
            "all original relative placements and z-order",
        ],
        "intentional_changes": [
            "three embryo raster objects replaced with corrected package-native total hotspot values",
            "the original per-panel q1-q99 robust normalization is retained so each slice shows its internal hotspot localization",
            "E13.0 uses accepted corrected communication/expression scores mapped onto the selected 50k global-t0 display cohort",
            "40 display-only Pancreas/Spinal cord cells absent from the accepted communication type universe are retained as base grey with unavailable NaN scores",
            "two E13.0 Choroid Plexus components are explicitly circled and labeled in the original 9-pt Arial/#66abb4 annotation grammar",
            "the legacy E13.0 Meninges text is redrawn at its exact baseline with a sub-point white halo because the corrected Jaw-and-tooth hotspot beneath it is darker",
        ],
        "geometry_contract": {
            "rotation": False,
            "reflection": False,
            "anisotropic_stretch": False,
            "spatial_warp": False,
            "crop_or_cell_deletion": False,
            "within_panel_axis": "equal aspect; symmetric whitespace when raw aspect differs from the AI canvas",
        },
        "replacement_records": replacements,
    }
    (QA / f"ai_style_equivalence_qa_{RENDER_VERSION}.json").write_text(
        json.dumps(style_qa, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    contact = QA / f"original_vs_corrected_ai_contact_{RENDER_VERSION}.png"
    make_contact(original_png, corrected_png, contact)

    manifest = {
        "schema_version": 1,
        "status": "complete" if style_status == "PASS" else "failed_style_qa",
        "dataset": "mosta",
        "panel": "Fig4b",
        "style_oracle": str(AI.relative_to(RUN)),
        "style_oracle_sha256": AI_SHA256,
        "style_oracle_creator": "Adobe Illustrator 30.2 (Macintosh)",
        "calculation_gate": "server_run_v4/calculation_gate.json",
        "server_manifest_sha256": sha256(SERVER / "run_manifest.json"),
        "package_commit": gate["package_commit"],
        "trajectory_mode": "global_t0_extrapolation",
        "restart_from_preceding_observed_stage": False,
        "spatial_warp": False,
        "initial_particles": 50000,
        "compute_particle_contract": "accepted workflow configured initial cap 12000; t=0.5 realized 15144 cells",
        "display_particle_contract": "50k initial-particle global-t0 run; t=0.5 realized 63533 cells",
        "ligand": "Wnt3a",
        "receptor_complex": "Fzd7_Lrp6",
        "complex_mode": "min",
        "require_all_subunits": True,
        "hotspot": "total (incoming + outgoing)",
        "normalization": {
            "mode": "per_panel_q1_q99",
            "bounds": per_panel_bounds,
            "style_source": "historical MOSTA manifest and original AI panel",
            "interpretation": "colors compare relative spatial hotspot intensity within each slice only; they do not encode cross-time absolute magnitude",
            "raw_cross_time_pair_scores": {
                "E12.5": 0.7908472467859028,
                "E13.0": 0.1499012076559161,
                "E13.5": 0.2283976323146597
            }
        },
        "display_mapping": gate["display_mapping"],
        "particle_density_sensitivity": gate["particle_density_sensitivity"],
        "generated_annotation_identity": {
            "Choroid plexus": {
                "raw_total_score": 0.0367532946730778,
                "per_panel_normalized_score": 1.0,
                "component_cell_counts": [237, 60],
                "component_centers_xy": [[-0.862, 3.024], [0.572, 2.996]],
                "annotation": "two dashed ellipses plus Choroid Plexus label",
            },
            "other_saturated_region": {
                "cell_type": "Jaw and tooth",
                "n_display_cells": 3607,
                "raw_total_score": 0.0326394759079971,
                "per_panel_normalized_score": 1.0,
                "center_xy": [-0.617210, 2.149729],
                "reason_for_saturation": "raw score is the E13.0 q99 normalization bound",
            },
            "Meninges": {
                "n_display_cells": 4902,
                "raw_total_score": 0.0175583383217383,
                "per_panel_normalized_score": 0.4758080815859703,
                "text_policy": "legacy position preserved; sub-point white halo added for legibility",
            },
        },
        "colormap": list(CMAP_COLORS),
        "point_style": {
            "colored_size_pt2": POINT_SIZE_PT2,
            "colored_alpha": ALPHA,
            "base_color": "#edf2f4",
            "base_size_pt2": BASE_POINT_SIZE_PT2,
            "base_alpha": BASE_ALPHA,
            "edgecolors": "none",
            "source": "historical MOSTA notebook and original AI rasters",
        },
        "replacements": replacements,
        "outputs": {
            "pdf": str(pdf.relative_to(RUN)),
            "svg": str(svg.relative_to(RUN)),
            "png": str(png.relative_to(RUN)),
            "fullpage_qa_only": str(fullpage.relative_to(RUN)),
            "contact": str(contact.relative_to(RUN)),
            "style_qa": str((QA / f"ai_style_equivalence_qa_{RENDER_VERSION}.json").relative_to(RUN)),
        },
        "hybrid_vector_note": "Original AI uses raster embryo maps; typography, arrows, annotations, colorbar, and composition remain vector in PDF/SVG.",
        "standalone_crop_cleanup": "Masked only the adjacent Fig. 4c raster spill at page x>=315.8 pt, y=231-416 pt; no Fig. 4b object occupies that region.",
        "fullpage_warning": "The full-page QA PDF retains legacy values in panels a,c,d,e and is not a manuscript deliverable.",
    }
    manifest_path = FIGURES / f"manifest_{RENDER_VERSION}.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    crop_doc.close()
    original.close()
    corrected.close()
    print(json.dumps({"status": manifest["status"], "outputs": manifest["outputs"]}, indent=2))
    return 0 if manifest["status"] == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())
