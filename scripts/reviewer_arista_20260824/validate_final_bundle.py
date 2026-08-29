#!/usr/bin/env python3
"""Validate the final corrected-equivalent ARISTA figure bundle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


EXPECTED_FIGURES = (
    "Figure5",
    "FigureS12",
    "FigureS13",
    "FigureS14",
    "FigureS15",
    "FigureS16",
    "FigureS17",
)
EXPECTED_PALETTE_SHA256 = "983b941fc93efe155511994d1d4b16cba5e11982cd81fb298d9a4a78907fbdd7"
FORBIDDEN_PALETTE_SHA256 = "163bf139"
FORBIDDEN_FONT_NAMES = ("DejaVu", "Liberation", "Helvetica")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def find_pdffonts() -> str | None:
    direct = shutil.which("pdffonts")
    if direct:
        return direct
    candidates = sorted(
        Path.home().glob(
            ".cache/codex-runtimes/*/dependencies/native/poppler/poppler/bin/pdffonts"
        )
    )
    return str(candidates[-1]) if candidates else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", required=True)
    args = parser.parse_args()
    bundle = Path(args.bundle_root).expanduser().resolve()
    checks: list[dict] = []

    def check(name: str, condition: bool, details: object = None) -> None:
        checks.append({"name": name, "passed": bool(condition), "details": details})

    assembly_path = bundle / "ASSEMBLY_MANIFEST.json"
    check("assembly manifest exists", assembly_path.is_file(), str(assembly_path))
    if not assembly_path.is_file():
        raise SystemExit(1)
    assembly = load_json(assembly_path)

    check("figure roster", tuple(assembly.get("panel_order", [])) == EXPECTED_FIGURES, assembly.get("panel_order"))
    check(
        "canonical palette contract",
        assembly.get("scientific_contract", {}).get("palette_sha256") == EXPECTED_PALETTE_SHA256,
        assembly.get("scientific_contract", {}).get("palette_sha256"),
    )
    check(
        "corrected algorithm policy",
        assembly.get("scientific_contract", {}).get("algorithm_policy")
        == "current corrected downstream algorithms; no legacy numerical results",
        assembly.get("scientific_contract", {}).get("algorithm_policy"),
    )

    output_hashes: dict[str, dict[str, str]] = {}
    source_hashes: dict[str, str] = {}
    for figure in EXPECTED_FIGURES:
        record = assembly.get("outputs", {}).get(figure, {})
        output_hashes[figure] = {}
        for kind in ("svg", "pdf", "png"):
            item = record.get(kind, {})
            path = bundle / item.get("path", "")
            exists = path.is_file() and path.stat().st_size > 0
            check(f"{figure} {kind} exists and nonempty", exists, str(path))
            if exists:
                actual = sha256(path)
                output_hashes[figure][kind] = actual
                check(f"{figure} {kind} hash replay", actual == item.get("sha256"), actual)

    for panel, item in assembly.get("sources", {}).items():
        path = bundle / item.get("path", "")
        exists = path.is_file() and path.stat().st_size > 0
        check(f"source {panel} exists", exists, str(path))
        if exists:
            actual = sha256(path)
            source_hashes[panel] = actual
            check(f"source {panel} hash replay", actual == item.get("sha256"), actual)

    external_image_refs: dict[str, list[str]] = {}
    for figure in EXPECTED_FIGURES:
        svg_path = bundle / assembly["outputs"][figure]["svg"]["path"]
        try:
            root = ET.parse(svg_path).getroot()
            check(f"{figure} SVG parses", True, root.get("viewBox"))
            refs: list[str] = []
            for element in root.iter():
                if element.tag.rsplit("}", 1)[-1] != "image":
                    continue
                href = element.get("href") or element.get("{http://www.w3.org/1999/xlink}href") or ""
                if href and not href.startswith("data:"):
                    refs.append(href)
            external_image_refs[figure] = refs
            check(f"{figure} has no external image dependencies", not refs, refs)
            text = svg_path.read_text(encoding="utf-8", errors="replace")
            forbidden_fonts = [font for font in FORBIDDEN_FONT_NAMES if font in text]
            check(f"{figure} SVG font normalization", not forbidden_fonts, forbidden_fonts)
        except Exception as exc:  # pragma: no cover - diagnostic path
            check(f"{figure} SVG parses", False, repr(exc))

    pdfinfo = shutil.which("pdfinfo")
    pdffonts = find_pdffonts()
    check("pdfinfo available", pdfinfo is not None, pdfinfo)
    check("pdffonts available", pdffonts is not None, pdffonts)
    pdf_font_audit: dict[str, str] = {}
    for figure in EXPECTED_FIGURES:
        pdf_path = bundle / assembly["outputs"][figure]["pdf"]["path"]
        if pdfinfo:
            info = subprocess.run([pdfinfo, str(pdf_path)], check=True, capture_output=True, text=True).stdout
            match = re.search(r"^Pages:\s+(\d+)$", info, flags=re.MULTILINE)
            pages = int(match.group(1)) if match else None
            check(f"{figure} PDF is one page", pages == 1, pages)
        if pdffonts:
            fonts = subprocess.run([pdffonts, str(pdf_path)], check=True, capture_output=True, text=True).stdout
            pdf_font_audit[figure] = fonts
            forbidden = [font for font in FORBIDDEN_FONT_NAMES if font in fonts]
            check(f"{figure} PDF has no fallback font", not forbidden, forbidden)
            check(f"{figure} PDF embeds Arial", "Arial" in fonts, fonts.splitlines()[2:6])

    results = bundle / "server_results"
    main_manifest = load_json(results / "main5abce_s13_paper_contract_full46209_canonical983b_v3/manifest.json")
    fig5d_manifest = load_json(results / "figure5d_full46209_paper_aligned_v7/evidence/Figure5d_manifest.json")
    display_manifest = load_json(results / "display_contract_v4_latest_full46209_canonical983b/manifest.json")
    s15_manifest = load_json(results / "s15_paper_contract_full46209_v11/manifest.json")
    lr_manifest = load_json(results / "temporal_s15_s17_manuscript68_full46209_display_v3/manifest.json")
    temporal_manifest = load_json(results / "temporal_s15_s17_full46209_3c87a3e/run_manifest.json")

    check("accepted model commit", main_manifest["accepted_model"]["package_commit"] == "3c87a3e", main_manifest["accepted_model"])
    check("main observed compute cohort", main_manifest["computation_contract"]["observed_compute_n"] == 46209, main_manifest["computation_contract"]["observed_compute_n"])
    check("Figure 5c corrected t1 cohort", main_manifest["computation_contract"]["figure5c"]["n_cells"] == 8106, main_manifest["computation_contract"]["figure5c"]["n_cells"])
    check("Figure 5c frozen manuscript ROI", main_manifest["computation_contract"]["figure5c"]["roi_n"] == 1454, main_manifest["computation_contract"]["figure5c"]["roi_n"])
    check("Figure 5c cosine scale", main_manifest["computation_contract"]["figure5c"]["roi_cosine_norm"] == [-1.0, 0.0, 1.0], main_manifest["computation_contract"]["figure5c"]["roi_cosine_norm"])
    check("Figure 5c nested box is display-only", main_manifest["computation_contract"]["figure5c"]["nested_red_display_annotation_scope"].startswith("visual callout only"), main_manifest["computation_contract"]["figure5c"]["nested_red_display_annotation_scope"])
    check("Figure 5e full corrected cohort", main_manifest["computation_contract"]["figure5e"]["n_raw_cells"] == 46209, main_manifest["computation_contract"]["figure5e"]["n_raw_cells"])
    check("S13 dense rows", main_manifest["computation_contract"]["s13"]["n_dense_grid_rows"] == 82329, main_manifest["computation_contract"]["s13"]["n_dense_grid_rows"])

    exclusion_path = bundle / "evidence/observed_display_exclusions_20.csv"
    with exclusion_path.open(newline="", encoding="utf-8") as handle:
        exclusions = list(csv.DictReader(handle))
    exclusion_counts = Counter(row["time_point_processed"] for row in exclusions)
    check("20-cell display-only roster size", len(exclusions) == 20, len(exclusions))
    check("20-cell display-only roster time counts", exclusion_counts == Counter({"2.0": 4, "3.0": 13, "4.0": 3}), dict(exclusion_counts))
    check("main exclusion scope", main_manifest["display_contract"]["exclusion_scope"].startswith("observed point-cloud glyphs only"), main_manifest["display_contract"]["exclusion_scope"])
    check("S12 exclusion scope", display_manifest["display_contract"]["observed_display_exclusions"]["policy"].endswith("no input mutation and no statistical exclusion"), display_manifest["display_contract"]["observed_display_exclusions"])

    check("Figure 5d full graph cohort", fig5d_manifest["roster_and_display_coordinate_validation"]["compute_n"] == 46209, fig5d_manifest["roster_and_display_coordinate_validation"]["compute_n"])
    check("Figure 5d old coordinates anchor-only", fig5d_manifest["display_projection"]["old_coordinates_are_anchor_only"] is True, fig5d_manifest["display_projection"])
    check("Figure 5d canonical palette", fig5d_manifest["inputs"]["palette_json_sha256"] == EXPECTED_PALETTE_SHA256, fig5d_manifest["inputs"]["palette_json_sha256"])

    s14_counts_path = results / "display_contract_v4_latest_full46209_canonical983b/tables/s14b_celltype_counts_full.csv"
    with s14_counts_path.open(newline="", encoding="utf-8") as handle:
        s14_rows = list(csv.DictReader(handle))
    s14_totals = {row["time"]: sum(int(value) for key, value in row.items() if key != "time") for row in s14_rows}
    check("S14 has nine time points", len(s14_rows) == 9, list(s14_totals))
    check("S14 fixed-particle denominator", all(total == 7668 for total in s14_totals.values()), s14_totals)

    s15_summary = s15_manifest["summary"]
    check("S15 corrected pattern counts", s15_summary["pattern_counts"] == {"1": 448, "2": 1552}, s15_summary["pattern_counts"])
    check("S15 corrected peak times", s15_summary["pattern_peak_times"] == {"1": 0.5, "2": 3.0}, s15_summary["pattern_peak_times"])
    check("S15 current particle contract", s15_manifest["analysis_contract"]["population_contract"]["interpolated_state_particles"] == 3072, s15_manifest["analysis_contract"]["population_contract"])

    check("temporal n_samples", temporal_manifest["config"]["n_samples"] == 3072, temporal_manifest["config"]["n_samples"])
    check("temporal communication cap", temporal_manifest["settings"]["communication_max_cells_per_timepoint"] == 3072, temporal_manifest["settings"]["communication_max_cells_per_timepoint"])
    check("temporal seed", temporal_manifest["settings"]["communication_random_seed"] == 42, temporal_manifest["settings"]["communication_random_seed"])
    check("LR corrected display counts", lr_manifest["counts"] == {"display_roster": 68, "scored": 67, "not_scored": 1, "display_cluster_1": 41, "display_cluster_2": 26}, lr_manifest["counts"])
    check("LR not-scored pair explicit", lr_manifest["not_scored"] == ["NEGR1_NEGR1"], lr_manifest["not_scored"])
    check("LR raw-score display", "raw-score" in lr_manifest["display_contract"]["S17"], lr_manifest["display_contract"]["S17"])

    selected_manifest_text = "\n".join(
        json.dumps(item, sort_keys=True)
        for item in (assembly, main_manifest, fig5d_manifest, display_manifest, s15_manifest, lr_manifest, temporal_manifest)
    )
    check("forbidden generic palette absent", FORBIDDEN_PALETTE_SHA256 not in selected_manifest_text, FORBIDDEN_PALETTE_SHA256)

    failed = [item for item in checks if not item["passed"]]
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "bundle_root": str(bundle),
        "status": "PASS" if not failed else "FAIL",
        "summary": {"checks": len(checks), "passed": len(checks) - len(failed), "failed": len(failed)},
        "checks": checks,
        "output_sha256": output_hashes,
        "source_sha256": source_hashes,
        "external_image_references": external_image_refs,
        "caption_notes": "CAPTION_UPDATE_NOTES.md",
    }
    report_path = bundle / "FINAL_QA_REPORT.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(report_path), **report["summary"], "status": report["status"]}, indent=2))
    if failed:
        for item in failed:
            print(f"FAIL: {item['name']}: {item['details']}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
