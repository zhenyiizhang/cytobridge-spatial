#!/usr/bin/env python3
"""Seal the accepted complete MOSTA Figure 4 bundle as an immutable archive."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
from pathlib import Path


ROOT = Path("/Users/zhenyizhang/Desktop/CytoBridge-ST-1104")
RUN = ROOT / "output/mosta_main_figure4_assembled_20260826_v1"
SOURCE = RUN / "candidate_v8"
ARCHIVE = RUN / "archive_v1"
UPSTREAM = ROOT / "output/mosta_main_fig4_completion_20260825_v1/archive_v1"
ASSEMBLY_SOURCE = RUN / "source"
MANUSCRIPT = Path(
    "/Users/zhenyizhang/Desktop/202511/nbme预投稿/投稿/投稿修改/"
    "cytobridge_manuscript_latest_clean.pdf"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


if ARCHIVE.exists():
    raise FileExistsError(f"Refusing to overwrite immutable archive: {ARCHIVE}")

required_hashes = {
    "figures/Figure_4_MOSTA_corrected_complete_exact_AI_layout.pdf":
        "45beb12c6314052c4e33ce73255dcd8511a2e9e81e0a765ad858b0961cf80b40",
    "figures/Figure_4_MOSTA_corrected_complete_exact_AI_layout.svg":
        "e4c993d0b73456b83ce933196c5e0e70468a78bf59c93dc1b24665b891c5d73e",
    "figures/Figure_4_MOSTA_corrected_complete_exact_AI_layout_300dpi.png":
        "d3a33f830ac66f343382b54d1b0fed383ee1a6205bdf02eb437138a2759b559a",
}
for relative, expected in required_hashes.items():
    actual = sha256(SOURCE / relative)
    if actual != expected:
        raise RuntimeError(f"Candidate changed: {relative}: {actual} != {expected}")

ARCHIVE.mkdir(parents=True)
for directory in ("figures", "qa", "source"):
    shutil.copytree(SOURCE / directory, ARCHIVE / directory)
copy_file(SOURCE / "assembly_audit.json", ARCHIVE / "assembly_audit.json")
copy_file(SOURCE / "provenance.md", ARCHIVE / "provenance.md")
copy_file(ASSEMBLY_SOURCE / "seal_figure4_archive.py", ARCHIVE / "source/seal_figure4_archive.py")

copy_file(
    UPSTREAM / "main_fig4_completion_manifest.json",
    ARCHIVE / "evidence/upstream_main_fig4_completion_manifest.json",
)
copy_file(
    UPSTREAM / "main_fig4_completion_matrix.md",
    ARCHIVE / "evidence/upstream_main_fig4_completion_matrix.md",
)
copy_file(
    UPSTREAM / "SHA256SUMS.txt",
    ARCHIVE / "evidence/upstream_panel_archive_SHA256SUMS.txt",
)
copy_file(
    UPSTREAM / "style_authority/Figure_mouse1.ai",
    ARCHIVE / "style_authority/Figure_mouse1.ai",
)

style_contract = {
    "layout_authority": str(UPSTREAM / "style_authority/Figure_mouse1.ai"),
    "layout_authority_sha256": sha256(UPSTREAM / "style_authority/Figure_mouse1.ai"),
    "submitted_manuscript": str(MANUSCRIPT),
    "submitted_manuscript_sha256": sha256(MANUSCRIPT),
    "submitted_manuscript_figure_page": 38,
    "submitted_manuscript_caption_page": 39,
    "panel_transform": "translation only; scale_x=scale_y=1; rotation=0; no warp",
    "cross_panel_connector_source": "direct Illustrator-compatible PDF content stream extraction",
}
(ARCHIVE / "style_authority/style_contract.json").write_text(
    json.dumps(style_contract, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
)

manifest = {
    "schema_version": 1,
    "status": "PASS_IMMUTABLE",
    "scope": "Complete main-text Figure 4, MOSTA panels a-e",
    "archived_on": "2026-08-26",
    "candidate_source": str(SOURCE),
    "accepted_panel_archive": str(UPSTREAM),
    "package_commit": "2b3c79eff3face7c4dd33de24d45384b9dbd8a84",
    "reference_h5ad_sha256": "8b9ca0ad3475040235036548d54b96272bf6c49f057f6c2a643152c11350ce25",
    "canvas_pt": [595.2760009765625, 841.8900146484375],
    "panels": ["a", "b", "c", "d", "e"],
    "primary_outputs": {
        relative: {"sha256": sha256(ARCHIVE / relative)}
        for relative in required_hashes
    },
    "qa": {
        "validator_errors": 0,
        "validator_warning": "67 embedded image objects are intentional dense data layers",
        "pdf_poppler_render": "qa/corrected_complete_Figure4_240dpi.png",
        "svg_librsvg_render": "qa/corrected_complete_Figure4_SVG_240dpi.png",
        "pdf_svg_contact": "qa/pdf_vs_svg_240dpi_contact.png",
        "original_corrected_contact": "qa/original_vs_corrected_complete_Figure4_contact.png",
    },
    "guards": {
        "panel_scaling": False,
        "panel_rotation": False,
        "panel_warp": False,
        "free_redesign": False,
        "arista_material_used": False,
        "corrected_package_native_inputs": True,
        "original_AI_visual_grammar": True,
    },
}
(ARCHIVE / "figure4_assembly_manifest.json").write_text(
    json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
)

checksum_lines = []
for path in sorted(ARCHIVE.rglob("*")):
    if not path.is_file() or path.name in {"SHA256SUMS.txt", "COMPLETE"}:
        continue
    checksum_lines.append(f"{sha256(path)}  {path.relative_to(ARCHIVE)}")
(ARCHIVE / "SHA256SUMS.txt").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
(ARCHIVE / "COMPLETE").write_text("PASS_IMMUTABLE\n", encoding="utf-8")

for path in sorted(ARCHIVE.rglob("*"), reverse=True):
    if path.is_file():
        path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
for path in sorted((p for p in ARCHIVE.rglob("*") if p.is_dir()), reverse=True):
    path.chmod(stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
ARCHIVE.chmod(stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)

print(json.dumps(manifest, indent=2, ensure_ascii=False))
