#!/usr/bin/env python3
"""Assemble accepted MOSTA Fig. 4a-e panels in the exact original AI layout."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

import fitz
from PIL import Image, ImageDraw, ImageFont
from pypdf import PdfReader, PdfWriter
from pypdf._page import PageObject
from pypdf.generic import DecodedStreamObject, NameObject


WORKSPACE = Path("/Users/zhenyizhang/Desktop/CytoBridge-ST-1104")
RUN_ROOT = WORKSPACE / "output/mosta_main_figure4_assembled_20260826_v1"
OUTPUT = RUN_ROOT / "candidate_v8"
SOURCE_SCRIPT = RUN_ROOT / "source/assemble_complete_figure4.py"
PANEL_ARCHIVE = WORKSPACE / "output/mosta_main_fig4_completion_20260825_v1/archive_v1"
STYLE_AI = PANEL_ARCHIVE / "style_authority/Figure_mouse1.ai"
MANUSCRIPT = Path(
    "/Users/zhenyizhang/Desktop/202511/nbme预投稿/投稿/投稿修改/"
    "cytobridge_manuscript_latest_clean.pdf"
)

PAGE_WIDTH = 595.2760009765625
PAGE_HEIGHT = 841.8900146484375

# PyMuPDF/Illustrator crop coordinates use a top-left origin.  These exact
# rectangles were frozen when each accepted panel was independently audited.
PANELS = [
    {
        "panel": "a",
        "pdf": "panels/fig4a/figures/fig4a.pdf",
        "svg": "panels/fig4a/figures/fig4a.svg",
        "sha256": "46714b8a416c634ff132f2706cfbf2d5f5d1106ac36eeba1216bdecf2cc589f7",
        "crop_top_left": [0.0, 0.0, 595.2760009765625, 192.0],
    },
    {
        "panel": "b",
        "pdf": "panels/fig4b/figures/fig4b.pdf",
        "svg": "panels/fig4b/figures/fig4b.svg",
        "sha256": "b6f565b16b1c603f5f51df65ad2fb3edca9b17c38ce17cf6c002a927fbba5350",
        "crop_top_left": [0.0, 183.5, 326.6, 442.0],
    },
    {
        "panel": "c",
        "pdf": "panels/fig4c/figures/fig4c.pdf",
        "svg": "panels/fig4c/figures/fig4c.svg",
        "sha256": "bc387a82c0409c976cf977f44d91f7b6d0fda54fa2a6b01d31429aea70dc1737",
        "crop_top_left": [316.0863952636719, 202.0, 595.0, 440.0],
    },
    {
        "panel": "d",
        "pdf": "panels/fig4d/figures/fig4d.pdf",
        "svg": "panels/fig4d/figures/fig4d.svg",
        "sha256": "52c84b237a1944fea2f1cbf0352b64b5b82a2307317752337161b168105c8992",
        "crop_top_left": [0.0, 463.8900146484375, 290.0, 841.8900146484375],
    },
    {
        "panel": "e",
        "pdf": "panels/fig4e/figures/fig4e.pdf",
        "svg": "panels/fig4e/figures/fig4e.svg",
        "sha256": "de1ae3596088fb560f2bf0f817c4cca2b3984dcd7d191793c198d097d78083a5",
        "crop_top_left": [286.0, 462.0, 595.2760009765625, 841.8900146484375],
    },
]

# These two cross-panel zoom connectors are independent Illustrator objects,
# rather than children of either accepted standalone panel.  Their page-space
# coordinates and graphics state were extracted directly from STYLE_AI content
# stream operations 119949 and 119955 (PDF origin: lower left).  This restores
# the original visual grammar without estimating or redrawing the geometry.
CONNECTORS = [
    {
        "source_operation": 119949,
        "p0_pdf": [105.4854, 315.5952],
        "p1_pdf": [285.8274, 344.5882],
        "width_pt": 2.0,
        "stroke_rgb": [0.137, 0.09, 0.082],
        "line_cap": 0,
        "line_join": 0,
    },
    {
        "source_operation": 119955,
        "p0_pdf": [105.4854, 248.9149],
        "p1_pdf": [285.8274, 8.9109],
        "width_pt": 2.0,
        "stroke_rgb": [0.137, 0.09, 0.082],
        "line_cap": 0,
        "line_join": 0,
    },
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def render_pdf(pdf: Path, png: Path, dpi: int) -> None:
    prefix = png.with_suffix("")
    subprocess.run(
        ["pdftoppm", "-png", "-r", str(dpi), "-singlefile", str(pdf), str(prefix)],
        check=True,
    )
    produced = prefix.with_suffix(".png")
    if produced != png:
        produced.replace(png)


def font(size: int) -> ImageFont.ImageFont:
    for path in (
        Path("/Library/Fonts/Arial.ttf"),
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
    ):
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def make_contact(original: Path, corrected: Path, output: Path) -> None:
    old = Image.open(original).convert("RGB")
    new = Image.open(corrected).convert("RGB")
    width = max(old.width, new.width)

    def fit(image: Image.Image) -> Image.Image:
        if image.width == width:
            return image
        ratio = width / image.width
        return image.resize((width, round(image.height * ratio)), Image.Resampling.LANCZOS)

    old = fit(old)
    new = fit(new)
    header = 62
    gap = 20
    canvas = Image.new("RGB", (width, header * 2 + gap + old.height + new.height), "white")
    draw = ImageDraw.Draw(canvas)
    label_font = font(32)
    draw.text((18, 10), "Original Illustrator layout", fill="#222222", font=label_font)
    canvas.paste(old, (0, header))
    y = header + old.height + gap
    draw.text((18, y + 10), "Corrected package-native Figure 4", fill="#222222", font=label_font)
    canvas.paste(new, (0, y + header))
    canvas.save(output, dpi=(240, 240), optimize=True)


def add_pdf_connectors(master: PageObject) -> None:
    commands = ["q"]
    for item in CONNECTORS:
        r, g, b = item["stroke_rgb"]
        x0, y0 = item["p0_pdf"]
        x1, y1 = item["p1_pdf"]
        commands.extend(
            [
                f'{item["width_pt"]:.6g} w',
                f'{item["line_cap"]} J',
                f'{item["line_join"]} j',
                f"{r:.6g} {g:.6g} {b:.6g} RG",
                f"{x0:.6f} {y0:.6f} m",
                f"{x1:.6f} {y1:.6f} l",
                "S",
            ]
        )
    commands.append("Q")
    overlay = PageObject.create_blank_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
    stream = DecodedStreamObject()
    stream.set_data(("\n".join(commands) + "\n").encode("ascii"))
    overlay[NameObject("/Contents")] = stream
    master.merge_page(overlay, over=True)


def make_master_svg(path: Path, placements: list[dict]) -> None:
    images = []
    for item in placements:
        # Derive each child SVG from the exact accepted panel PDF used by the
        # master PDF.  This avoids backend drift in independently saved SVGs
        # (notably font and panel-label differences) while remaining vector.
        document = fitz.open(item["source_pdf"])
        source = document[0].get_svg_image(text_as_path=True)
        document.close()
        source = re.sub(r"^\s*<\?xml.*?\?>\s*", "", source, count=1, flags=re.DOTALL)
        source = re.sub(r"^\s*<!DOCTYPE.*?>\s*", "", source, count=1, flags=re.DOTALL)
        prefix = f'fig4{item["panel"]}_'
        # A standalone master SVG must place all five child documents in one
        # DOM.  Prefix every child id/reference to prevent collisions between
        # repeated Illustrator ids such as clip_1, mask_3, and image_4.
        ids = sorted(set(re.findall(r'\bid="([^"]+)"', source)), key=len, reverse=True)
        for old_id in ids:
            new_id = prefix + old_id
            source = source.replace(f'id="{old_id}"', f'id="{new_id}"')
            source = source.replace(f'#{old_id}', f'#{new_id}')
        x = item["target_top_left"][0]
        y = item["target_top_left"][1]
        w = item["source_width_pt"]
        h = item["source_height_pt"]
        # Normalize child viewport dimensions to unitless master-SVG user
        # coordinates.  Some accepted SVGs use `pt`; leaving that unit in a
        # nested document would cause librsvg/browser CSS 96-dpi conversion.
        root_end = source.find(">")
        if root_end < 0:
            raise RuntimeError(f'Invalid SVG root for panel {item["panel"]}')
        root = source[: root_end + 1]
        root = re.sub(r'\bwidth="[^"]+"', f'width="{w:.12g}"', root, count=1)
        root = re.sub(r'\bheight="[^"]+"', f'height="{h:.12g}"', root, count=1)
        source = root + source[root_end + 1 :]
        if item["panel"] == "e":
            # PyMuPDF 1.27 omits this isolated one-character text object from
            # SVG extraction although it is present in the accepted PDF.  Its
            # exact PDF text audit is: Arial-BoldMT, 14 pt, origin
            # (43.479248, 21.255005), black. Restore that object explicitly.
            source = source.replace(
                "</svg>",
                (
                    '<text x="43.479248" y="21.255005" font-family="Arial" '
                    'font-size="14" font-weight="bold" fill="#000000">e</text>\n'
                    "</svg>"
                ),
                1,
            )
        images.append(
            f'  <g transform="translate({x:.12g} {y:.12g})">\n{source}\n  </g>'
        )
    connector_lines = []
    for item in CONNECTORS:
        r, g, b = item["stroke_rgb"]
        x0, y0_pdf = item["p0_pdf"]
        x1, y1_pdf = item["p1_pdf"]
        y0 = PAGE_HEIGHT - y0_pdf
        y1 = PAGE_HEIGHT - y1_pdf
        connector_lines.append(
            "  <line x1=\"{x0:.6f}\" y1=\"{y0:.6f}\" x2=\"{x1:.6f}\" "
            "y2=\"{y1:.6f}\" stroke=\"rgb({r:.3%},{g:.3%},{b:.3%})\" "
            "stroke-width=\"{width:.6g}\" stroke-linecap=\"butt\" "
            "stroke-linejoin=\"miter\"/>".format(
                x0=x0,
                y0=y0,
                x1=x1,
                y1=y1,
                r=r,
                g=g,
                b=b,
                width=item["width_pt"],
            )
        )
    content = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{PAGE_WIDTH}pt" '
        f'height="{PAGE_HEIGHT}pt" viewBox="0 0 {PAGE_WIDTH} {PAGE_HEIGHT}">\n'
        + f'  <rect width="{PAGE_WIDTH}" height="{PAGE_HEIGHT}" fill="#ffffff"/>\n'
        + "\n".join(images + connector_lines)
        + "\n</svg>\n"
    )
    path.write_text(content, encoding="utf-8")


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite: {OUTPUT}")
    if sha256(STYLE_AI) != "340a5ed88dc911d6923bc6b21cf1ceb39fdbef16edf2e325822a5b422045cbc2":
        raise RuntimeError("Illustrator style authority changed")
    if sha256(MANUSCRIPT) != "94c26a14500b16706ab9647ce26c628b9b7f642a58faf79421dd17577cae4337":
        raise RuntimeError("Manuscript style authority changed")

    (OUTPUT / "figures").mkdir(parents=True)
    (OUTPUT / "qa").mkdir(parents=True)
    (OUTPUT / "source").mkdir(parents=True)

    master = PageObject.create_blank_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
    placements = []
    for spec in PANELS:
        source_pdf = PANEL_ARCHIVE / spec["pdf"]
        source_svg = PANEL_ARCHIVE / spec["svg"]
        if sha256(source_pdf) != spec["sha256"]:
            raise RuntimeError(f"Accepted panel changed: {spec['panel']}")
        reader = PdfReader(source_pdf)
        if len(reader.pages) != 1:
            raise RuntimeError(f"Expected one page: {source_pdf}")
        page = reader.pages[0]
        source_width = float(page.mediabox.width)
        source_height = float(page.mediabox.height)
        x0, y0, x1, y1 = spec["crop_top_left"]
        target_width = x1 - x0
        target_height = y1 - y0
        width_error = source_width - target_width
        height_error = source_height - target_height
        if abs(width_error) > 0.001 or abs(height_error) > 0.001:
            raise RuntimeError(
                f"Panel {spec['panel']} size mismatch: "
                f"source=({source_width},{source_height}), target=({target_width},{target_height})"
            )
        translate_x = x0
        translate_y = PAGE_HEIGHT - y0 - source_height
        master.merge_translated_page(page, translate_x, translate_y, expand=False, over=True)
        placements.append(
            {
                "panel": spec["panel"],
                "source_pdf": str(source_pdf),
                "source_pdf_sha256": sha256(source_pdf),
                "source_svg": str(source_svg),
                "source_svg_sha256": sha256(source_svg),
                "source_width_pt": source_width,
                "source_height_pt": source_height,
                "target_top_left": spec["crop_top_left"],
                "pdf_translation_bottom_left": [translate_x, translate_y],
                "scale_x": 1.0,
                "scale_y": 1.0,
                "rotation_degrees": 0.0,
                "width_error_pt": width_error,
                "height_error_pt": height_error,
            }
        )

    add_pdf_connectors(master)

    pdf = OUTPUT / "figures/Figure_4_MOSTA_corrected_complete_exact_AI_layout.pdf"
    svg = OUTPUT / "figures/Figure_4_MOSTA_corrected_complete_exact_AI_layout.svg"
    png = OUTPUT / "figures/Figure_4_MOSTA_corrected_complete_exact_AI_layout_300dpi.png"
    writer = PdfWriter()
    writer.add_page(master)
    writer.add_metadata(
        {
            "/Title": "Figure 4 - corrected package-native MOSTA results",
            "/Subject": "Exact original Illustrator panel layout with accepted corrected panels a-e",
        }
    )
    with pdf.open("wb") as handle:
        writer.write(handle)
    make_master_svg(svg, placements)
    render_pdf(pdf, png, 300)

    original_png = OUTPUT / "qa/original_AI_240dpi.png"
    corrected_png = OUTPUT / "qa/corrected_complete_Figure4_240dpi.png"
    render_pdf(STYLE_AI, original_png, 240)
    render_pdf(pdf, corrected_png, 240)
    make_contact(
        original_png,
        corrected_png,
        OUTPUT / "qa/original_vs_corrected_complete_Figure4_contact.png",
    )

    shutil.copy2(SOURCE_SCRIPT, OUTPUT / "source/assemble_complete_figure4.py")
    audit = {
        "status": "CANDIDATE_READY_FOR_VISUAL_QA",
        "canvas_pt": [PAGE_WIDTH, PAGE_HEIGHT],
        "layout_truth": str(STYLE_AI),
        "layout_truth_sha256": sha256(STYLE_AI),
        "numerical_truth": str(PANEL_ARCHIVE),
        "source_panel_checksum_manifest_sha256": sha256(PANEL_ARCHIVE / "SHA256SUMS.txt"),
        "placements": placements,
        "cross_panel_connectors": CONNECTORS,
        "master_svg_method": (
            "PyMuPDF get_svg_image(text_as_path=True) from each exact accepted "
            "panel PDF, with collision-safe id prefixing and unitless nested viewports; "
            "isolated Fig4e label restored from its audited PDF text object"
        ),
        "connector_style_truth_method": (
            "Direct extraction from Illustrator-compatible PDF content stream; "
            "no coordinate or style estimation"
        ),
        "guards": {
            "panel_scaling_used": False,
            "panel_rotation_used": False,
            "panel_warp_used": False,
            "recomposition_design_changes": False,
            "only_exact_AI_coordinate_translation_used": True,
            "only_exact_AI_connector_objects_restored": True,
            "arista_data_labels_palette_model_or_analysis_used": False,
        },
        "outputs": {
            "pdf": {"path": str(pdf), "sha256": sha256(pdf)},
            "svg": {"path": str(svg), "sha256": sha256(svg)},
            "png": {"path": str(png), "sha256": sha256(png)},
        },
    }
    (OUTPUT / "assembly_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
