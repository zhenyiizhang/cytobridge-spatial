#!/usr/bin/env python3
"""Build the immutable master delivery for all accepted MOSTA manuscript figures."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import stat
from pathlib import Path


WORKSPACE = Path("/Users/zhenyizhang/Desktop/CytoBridge-ST-1104")
BUILD_ROOT = WORKSPACE / "output/mosta_all_panels_delivery_20260826_v1"
ARCHIVE = BUILD_ROOT / "archive_v1"
SCRIPT = BUILD_ROOT / "source/build_mosta_master_delivery.py"

MAIN_PDF = Path(
    "/Users/zhenyizhang/Desktop/202511/nbme预投稿/投稿/投稿修改/"
    "cytobridge_manuscript_latest_clean.pdf"
)
SI_PDF = Path(
    "/Users/zhenyizhang/Desktop/202511/nbme预投稿/投稿/投稿修改/si.pdf"
)

MODEL_RUN = (
    "/data/cytobridge/projects/CytoBridge-ST-1104/runs/"
    "corrected-matched-ablation-20260813-3c87a3e-r1/mosta/training"
)
PACKAGE_RELEASE = (
    "/data/cytobridge/projects/CytoBridge-ST-1104/software/"
    "cytobridge-release-2b3c79e-runtime"
)
SHARED_50K_RUN = (
    "/data/cytobridge/projects/CytoBridge-ST-1104/runs/"
    "si-s4-s10-global-t0-n50000-dense13-2b3c79e-seed42-20260825-v1"
)
CLUSTERPROFILER_RUN = (
    "/data/cytobridge/projects/CytoBridge-ST-1104/runs/"
    "si-s9-s10-clusterprofiler-mouse-all-pool-2b3c79e-20260826-v1"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_checksum_line(line: str) -> tuple[str, str]:
    checksum, filename = line.rstrip("\n").split(maxsplit=1)
    if filename.startswith("*"):
        filename = filename[1:]
    return checksum, filename


def verify_checksum_manifest(root: Path, manifest_name: str) -> dict:
    manifest = root / manifest_name
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    checked = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, rel = parse_checksum_line(line)
        candidate = root / rel
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        actual = sha256(candidate)
        if actual != expected:
            raise RuntimeError(
                f"Checksum mismatch in {manifest}: {rel}: {actual} != {expected}"
            )
        checked.append({"path": rel, "sha256": actual})
    return {
        "archive_root": str(root),
        "checksum_manifest": manifest_name,
        "checksum_manifest_sha256": sha256(manifest),
        "verified_file_count": len(checked),
        "status": "PASS",
    }


MAIN_ARCHIVE = WORKSPACE / "output/mosta_main_fig4_completion_20260825_v1/archive_v1"
S4_ARCHIVE = WORKSPACE / "output/mosta_si_s4_20260825_v1/archive_v1"
S5_ARCHIVE = WORKSPACE / "output/mosta_si_s5_growth_20260825_v1/archive_v1"
S6_ARCHIVE = WORKSPACE / "output/mosta_si_s6_composition_20260825_v1/archive_v1"
S7_ARCHIVE = WORKSPACE / "output/mosta_si_s7_lineage_20260826_v2"
S8_ARCHIVE = WORKSPACE / "output/mosta_si_s8_gene_programs_20260826_v1/archive_v1"
S9_S10_ARCHIVE = WORKSPACE / "output/mosta_si_s9_s10_clusterprofiler_20260826_v1/archive_v1"
S11_ARCHIVE = WORKSPACE / "output/mosta_si_s11_msum_corrected_20260826_v1/archive_v1"


RECORDS = [
    {
        "panel": "Fig. 4a",
        "archive": MAIN_ARCHIVE,
        "checksum_manifest": "SHA256SUMS.txt",
        "source_stem": "panels/fig4a/figures/fig4a",
        "delivery_stem": "figures/main/Fig4a",
        "numerical_truth": "Latest accepted package-native corrected MOSTA model output; global t0, 50,000 particles.",
        "style_truth": "Submitted Fig. 4a / original code and AI layout.",
        "qa": "Calculation gate PASS; exact-layout style gate PASS; Poppler PDF QA PASS.",
    },
    {
        "panel": "Fig. 4b",
        "archive": MAIN_ARCHIVE,
        "checksum_manifest": "SHA256SUMS.txt",
        "source_stem": "panels/fig4b/figures/fig4b",
        "delivery_stem": "figures/main/Fig4b",
        "numerical_truth": "Latest accepted package-native corrected MOSTA spatial annotation result.",
        "style_truth": "Submitted Fig. 4b / original code and AI layout.",
        "qa": "Calculation, biological-interpretation, style, and Poppler PDF gates PASS.",
    },
    {
        "panel": "Fig. 4c",
        "archive": MAIN_ARCHIVE,
        "checksum_manifest": "SHA256SUMS.txt",
        "source_stem": "panels/fig4c/figures/fig4c",
        "delivery_stem": "figures/main/Fig4c",
        "numerical_truth": "Latest package classifier, k=10; accepted E15.0 to E15.5 cartilage-lineage interval.",
        "style_truth": "Submitted Fig. 4c / historical lineage-transition plotting grammar.",
        "qa": "Classifier/transition audit PASS; exact-old-style and Poppler PDF gates PASS.",
    },
    {
        "panel": "Fig. 4d",
        "archive": MAIN_ARCHIVE,
        "checksum_manifest": "SHA256SUMS.txt",
        "source_stem": "panels/fig4d/figures/fig4d",
        "delivery_stem": "figures/main/Fig4d",
        "numerical_truth": "Latest accepted package-native corrected MOSTA downstream result.",
        "style_truth": "Submitted Fig. 4d and historical notebook layout.",
        "qa": "Numerical, style, and Poppler PDF gates PASS.",
    },
    {
        "panel": "Fig. 4e",
        "archive": MAIN_ARCHIVE,
        "checksum_manifest": "SHA256SUMS.txt",
        "source_stem": "panels/fig4e/figures/fig4e",
        "delivery_stem": "figures/main/Fig4e",
        "numerical_truth": "Latest accepted package-native corrected brain velocity/interaction-velocity result.",
        "style_truth": "Original MOSTA brain velocity notebooks and submitted AI geometry.",
        "qa": "Velocity component/basis/annotation audit PASS; exact-source render and Poppler PDF gates PASS.",
    },
    {
        "panel": "Fig. S4",
        "archive": S4_ARCHIVE,
        "checksum_manifest": "CHECKSUMS.sha256",
        "source_stem": "figures/Figure_S4_MOSTA_latest_package_global_t0_50k_exact_old_rasterized_points",
        "delivery_stem": "figures/si/Figure_S4",
        "numerical_truth": "Shared latest package global-t0 trajectory, 50,000 particles, 13 dense time points.",
        "style_truth": "Exact submitted S4 spatial montage; dense point layers intentionally rasterized inside vector PDF/SVG.",
        "qa": "Native geometry/composition audit PASS; 240-dpi Poppler visual QA PASS.",
    },
    {
        "panel": "Fig. S5",
        "archive": S5_ARCHIVE,
        "checksum_manifest": "CHECKSUMS.sha256",
        "source_stem": "figures/Figure_S5_MOSTA_latest_package_global_t0_growth_exact_submitted_style",
        "delivery_stem": "figures/si/Figure_S5",
        "numerical_truth": "Shared latest package global-t0 result with corrected package-native growth analysis.",
        "style_truth": "Exact submitted growth-panel grammar; dense spatial point layer intentionally rasterized.",
        "qa": "Growth/state-alignment/residual audits PASS; 240-dpi Poppler visual QA PASS.",
    },
    {
        "panel": "Fig. S6",
        "archive": S6_ARCHIVE,
        "checksum_manifest": "CHECKSUMS.sha256",
        "source_stem": "figures/Figure_S6_MOSTA_latest_package_global_t0_50k_exact_submitted_style",
        "delivery_stem": "figures/si/Figure_S6",
        "numerical_truth": "Shared latest package global-t0 50,000-particle cell-composition result.",
        "style_truth": "Exact submitted counts/percentage composition grammar and fixed submitted label order.",
        "qa": "Composition/growth cross-check PASS; fully vector PDF; 240-dpi Poppler visual QA PASS.",
    },
    {
        "panel": "Fig. S7",
        "archive": S7_ARCHIVE,
        "checksum_manifest": "SHA256SUMS.txt",
        "source_stem": "figures/Figure_S7_MOSTA_latest_package_global_t0_fixed_particle_k10_exact_old_plotly_style_Arial",
        "delivery_stem": "figures/si/Figure_S7",
        "numerical_truth": "Latest package global-t0 fixed-particle lineage; latest classifier k=10.",
        "style_truth": "Exact historical Plotly Sankey grammar and Arial typography.",
        "qa": "Particle identity/interval/filter audits PASS; 180-dpi Poppler visual QA PASS.",
    },
    {
        "panel": "Fig. S8",
        "archive": S8_ARCHIVE,
        "checksum_manifest": "CHECKSUMS.sha256",
        "source_stem": "figures/Figure_S8_MOSTA_latest_package_brain_gene_programs_exact_submitted_style",
        "delivery_stem": "figures/si/Figure_S8",
        "numerical_truth": "Latest package brain gene-program downstream analysis.",
        "style_truth": "Exact submitted brain gene-program heatmap/trajectory grammar.",
        "qa": "Gene-program numerical/interpretability audit PASS; 200-dpi Poppler visual QA PASS.",
    },
    {
        "panel": "Fig. S9",
        "archive": S9_S10_ARCHIVE,
        "checksum_manifest": "CHECKSUMS.sha256",
        "source_stem": "figures/s9/Figure_S9_MOSTA_latest_package_clusterProfiler_GO_exact_submitted_style",
        "delivery_stem": "figures/si/Figure_S9",
        "numerical_truth": "Latest package brain patterns; server clusterProfiler 4.10.0, org.Mm.eg.db 3.18.0, GO ALL, BH per query.",
        "style_truth": "Exact submitted S9 GO-panel grammar.",
        "qa": "Independent hypergeometric/BH cross-check PASS; PDF validation and 200/300-dpi Poppler QA PASS.",
    },
    {
        "panel": "Fig. S10",
        "archive": S9_S10_ARCHIVE,
        "checksum_manifest": "CHECKSUMS.sha256",
        "source_stem": "figures/s10/Figure_S10_MOSTA_latest_package_DP3_clusterProfiler_GO_exact_submitted_style",
        "delivery_stem": "figures/si/Figure_S10",
        "numerical_truth": "Latest package DP3 developmental waves; server clusterProfiler 4.10.0, org.Mm.eg.db 3.18.0, GO ALL, BH per query.",
        "style_truth": "Exact submitted S10 wave-map and GO-panel grammar.",
        "qa": "DP3 objective/phase and independent GO cross-check PASS; 200-dpi Poppler visual QA PASS.",
    },
    {
        "panel": "Fig. S11",
        "archive": S11_ARCHIVE,
        "checksum_manifest": "CHECKSUMS.sha256",
        "source_stem": "figures/Figure_S11_MOSTA_corrected_package_Msum_average_k3_exact_submitted_style_representative31",
        "delivery_stem": "figures/si/Figure_S11",
        "numerical_truth": "Correct package M_sum cell-type-aggregated total communication; average linkage k=3; seed-stable representative selection.",
        "style_truth": "Exact submitted 4x8 small-multiples geometry, palette, axes, line/marker and title grammar.",
        "qa": "Component decomposition and seeds 42/43/44 stability PASS; no artificial synchronized pulse; 180-dpi Poppler visual QA PASS.",
    },
]


def ensure_source_assets() -> None:
    expected = {
        MAIN_PDF: "94c26a14500b16706ab9647ce26c628b9b7f642a58faf79421dd17577cae4337",
        SI_PDF: "150deefb96083732a7aa7ac89bda1556c3ee4900699ed84946ecf7de48f9c93d",
    }
    for path, checksum in expected.items():
        if not path.is_file() or sha256(path) != checksum:
            raise RuntimeError(f"Manuscript source mismatch: {path}")


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def seal_read_only(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_file():
            path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        elif path.is_dir():
            path.chmod(
                stat.S_IRUSR
                | stat.S_IXUSR
                | stat.S_IRGRP
                | stat.S_IXGRP
                | stat.S_IROTH
                | stat.S_IXOTH
            )
    root.chmod(
        stat.S_IRUSR
        | stat.S_IXUSR
        | stat.S_IRGRP
        | stat.S_IXGRP
        | stat.S_IROTH
        | stat.S_IXOTH
    )


def main() -> None:
    if ARCHIVE.exists():
        raise FileExistsError(f"Immutable output already exists: {ARCHIVE}")
    ensure_source_assets()

    unique_archives = []
    seen = set()
    for record in RECORDS:
        key = (record["archive"], record["checksum_manifest"])
        if key not in seen:
            seen.add(key)
            unique_archives.append(key)
    source_verification = [
        verify_checksum_manifest(root, checksum_manifest)
        for root, checksum_manifest in unique_archives
    ]

    (ARCHIVE / "figures/main").mkdir(parents=True)
    (ARCHIVE / "figures/si").mkdir(parents=True)
    (ARCHIVE / "inventory").mkdir(parents=True)
    (ARCHIVE / "caption").mkdir(parents=True)
    (ARCHIVE / "source").mkdir(parents=True)

    verification_by_root = {
        item["archive_root"]: item for item in source_verification
    }
    contract_rows = []
    delivered = []
    for record in RECORDS:
        source_pdf = record["archive"] / f"{record['source_stem']}.pdf"
        source_svg = record["archive"] / f"{record['source_stem']}.svg"
        target_pdf = ARCHIVE / f"{record['delivery_stem']}.pdf"
        target_svg = ARCHIVE / f"{record['delivery_stem']}.svg"
        for source in (source_pdf, source_svg):
            if not source.is_file():
                raise FileNotFoundError(source)
        shutil.copy2(source_pdf, target_pdf)
        shutil.copy2(source_svg, target_svg)
        if sha256(source_pdf) != sha256(target_pdf) or sha256(source_svg) != sha256(target_svg):
            raise RuntimeError(f"Copy verification failed: {record['panel']}")

        archive_verification = verification_by_root[str(record["archive"])]
        contract_rows.append(
            {
                "panel": record["panel"],
                "source_archive": str(record["archive"]),
                "source_checksum_manifest": record["checksum_manifest"],
                "source_checksum_manifest_sha256": archive_verification[
                    "checksum_manifest_sha256"
                ],
                "source_pdf": str(source_pdf.relative_to(WORKSPACE)),
                "source_pdf_sha256": sha256(source_pdf),
                "delivered_pdf": str(target_pdf.relative_to(ARCHIVE)),
                "delivered_pdf_sha256": sha256(target_pdf),
                "delivered_svg": str(target_svg.relative_to(ARCHIVE)),
                "delivered_svg_sha256": sha256(target_svg),
                "numerical_truth": record["numerical_truth"],
                "style_truth": record["style_truth"],
                "qa_status": record["qa"],
                "status": "ACCEPTED_PASS",
            }
        )
        delivered.append(
            {
                "panel": record["panel"],
                "pdf": str(target_pdf.relative_to(ARCHIVE)),
                "pdf_sha256": sha256(target_pdf),
                "svg": str(target_svg.relative_to(ARCHIVE)),
                "svg_sha256": sha256(target_svg),
            }
        )

    write_csv(ARCHIVE / "ARCHIVE_CONTRACTS.csv", contract_rows)
    (ARCHIVE / "SOURCE_ARCHIVE_VERIFICATION.json").write_text(
        json.dumps(source_verification, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    inventory_root = WORKSPACE / "output/mosta_si_inventory_20260825_v3"
    for rel in (
        "MOSTA_SI_PANEL_INVENTORY.md",
        "mosta_si_panel_inventory.json",
        "inventory_bundle_manifest.json",
    ):
        shutil.copy2(inventory_root / rel, ARCHIVE / "inventory" / rel)
    shutil.copy2(
        S11_ARCHIVE / "caption/S11_caption_replacement.md",
        ARCHIVE / "caption/S11_caption_replacement.md",
    )
    shutil.copy2(
        MAIN_ARCHIVE / "main_fig4_completion_matrix.md",
        ARCHIVE / "inventory/main_fig4_completion_matrix.md",
    )
    shutil.copy2(SCRIPT, ARCHIVE / "source/build_mosta_master_delivery.py")

    index_lines = [
        "# MOSTA main-text and supplementary-figure master delivery",
        "",
        "Status: **COMPLETE / ACCEPTED / IMMUTABLE**",
        "",
        "This bundle is a convenient, checksum-locked index of every accepted MOSTA figure in the submitted main text and Supplementary Information. It copies the final PDF/SVG artifacts byte-for-byte; complete numerical inputs, audit tables, plotting sources, QA renders, and provenance remain in the referenced source archives listed in `ARCHIVE_CONTRACTS.csv`.",
        "",
        "## Scope",
        "",
        "- Main text: Fig. 4a-4e.",
        "- Supplementary Information: Fig. S4-S11.",
        "- Fig. S12 onward starts ARISTA and is deliberately out of MOSTA scope.",
        "",
        "## Locked interpretation",
        "",
        f"- Numerical truth: corrected package-native MOSTA model `{MODEL_RUN}` under release commit `2b3c79eff3face7c4dd33de24d45384b9dbd8a84`.",
        f"- Shared dense trajectory truth for S4-S10 where applicable: `{SHARED_50K_RUN}`; 50,000 particles, 13 times from t0=0 to t=3, global-t0 propagation, no observed-state restart.",
        f"- GO truth for S9/S10: server run `{CLUSTERPROFILER_RUN}` using clusterProfiler 4.10.0 and org.Mm.eg.db 3.18.0.",
        "- Style truth: submitted manuscript/SI panels, original MOSTA notebooks/scripts, and the original Illustrator layout where applicable.",
        "- No ARISTA data, labels, palette, model result, or analysis logic was used. A historical helper imported for plotting syntax in one archive is style authority only, as declared there.",
        "- No rotation, stretch, shear, projection warp, or other geometry manipulation was used to imitate the submitted appearance.",
        "",
        "## Delivered figures",
        "",
        "| Panel | PDF | SVG | Numerical/style/QA status |",
        "|---|---|---|---|",
    ]
    for row in contract_rows:
        index_lines.append(
            f"| {row['panel']} | `{row['delivered_pdf']}` | `{row['delivered_svg']}` | ACCEPTED PASS |"
        )
    index_lines.extend(
        [
            "",
            "## Important corrected analyses",
            "",
            "- Fig. 4c and S7 use the unchanged latest classifier with k=10. The accepted Fig. 4c interval is E15.0 to E15.5; the result was not relabelled to manufacture the old three-category message.",
            "- Fig. S9/S10 use genuine server clusterProfiler enrichment. S9 pattern 2 displays 11 terms because exactly 11 pass the specified significance gate; no filler terms were introduced.",
            "- Fig. S11 uses package `M_sum`, matching the submitted cell-type-aggregated total-attention estimand. The rejected `M_per_source` version created normalization-driven synchronized pulses. Seeds 42/43/44 give median pairwise-profile correlations of 0.9993-0.9996 and adjusted Rand indices of 0.859-0.932; the final 31 representatives are stability/effect-size selected rather than visually cherry-picked.",
            "",
            "## Manuscript sources",
            "",
            f"- Main PDF: `{MAIN_PDF}`; SHA-256 `{sha256(MAIN_PDF)}`.",
            f"- SI PDF: `{SI_PDF}`; SHA-256 `{sha256(SI_PDF)}`.",
            "",
            "Every source archive checksum manifest was reverified before copying. `CHECKSUMS.sha256` verifies this master delivery; `SOURCE_ARCHIVE_VERIFICATION.json` records the upstream archive checks.",
        ]
    )
    (ARCHIVE / "DELIVERY_INDEX.md").write_text(
        "\n".join(index_lines) + "\n", encoding="utf-8"
    )

    manifest = {
        "status": "COMPLETE_ACCEPTED_MOSTA_MAIN_AND_SI",
        "created_date": "2026-08-26",
        "scope": {
            "main": ["Fig. 4a", "Fig. 4b", "Fig. 4c", "Fig. 4d", "Fig. 4e"],
            "si": [
                "Fig. S4",
                "Fig. S5",
                "Fig. S6",
                "Fig. S7",
                "Fig. S8",
                "Fig. S9",
                "Fig. S10",
                "Fig. S11",
            ],
            "out_of_scope": "Fig. S12 onward is ARISTA, not MOSTA.",
        },
        "package": {
            "release_path": PACKAGE_RELEASE,
            "commit": "2b3c79eff3face7c4dd33de24d45384b9dbd8a84",
            "archive_sha256": "06852992db0d8ebedd8f0baa19a3f539bcdd271dfc0c6fae9774dd553c8fe55e",
        },
        "accepted_model_run": MODEL_RUN,
        "finetune_sha256": "d08d21b86fdcd92c748bb54ca81f998fbb157ca5a7acd9548bbfe16c573bfaa5",
        "score_sha256": "d7d06657f8548618db1bc85409e73305fbf59feb9446793550b2c9761639e52a",
        "aligned_h5ad_sha256": "8b9ca0ad3475040235036548d54b96272bf6c49f057f6c2a643152c11350ce25",
        "classifier_k10_sha256": "f938575c145baa9002de695c39a8637f57d6cb3a06ccfaf4d18b707ca962a7e0",
        "shared_global_t0_50k_run": SHARED_50K_RUN,
        "clusterprofiler": {
            "run": CLUSTERPROFILER_RUN,
            "clusterProfiler_version": "4.10.0",
            "org_Mm_eg_db_version": "3.18.0",
            "ontology": "ALL pooled",
            "adjustment": "Benjamini-Hochberg per query",
        },
        "manuscript_sources": {
            "main_pdf": str(MAIN_PDF),
            "main_pdf_sha256": sha256(MAIN_PDF),
            "si_pdf": str(SI_PDF),
            "si_pdf_sha256": sha256(SI_PDF),
        },
        "guardrails": {
            "arista_dataset_assets_used": False,
            "arista_labels_or_palette_used": False,
            "arista_model_or_analysis_result_used": False,
            "rotation_stretch_shear_or_projection_warp_used": False,
            "computed_values_and_style_authority_separated": True,
        },
        "delivered": delivered,
        "upstream_archive_verification": source_verification,
        "master_pdf_count": len(delivered),
        "master_svg_count": len(delivered),
    }
    (ARCHIVE / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (ARCHIVE / "COMPLETE").write_text(
        "COMPLETE_ACCEPTED_MOSTA_MAIN_AND_SI\n", encoding="utf-8"
    )

    checksums = []
    for path in sorted(p for p in ARCHIVE.rglob("*") if p.is_file()):
        if path.name == "CHECKSUMS.sha256":
            continue
        checksums.append(f"{sha256(path)}  {path.relative_to(ARCHIVE)}")
    (ARCHIVE / "CHECKSUMS.sha256").write_text(
        "\n".join(checksums) + "\n", encoding="utf-8"
    )
    seal_read_only(ARCHIVE)
    print(f"Created immutable master delivery: {ARCHIVE}")
    print(f"Panels: {len(RECORDS)}; PDFs: {len(delivered)}; SVGs: {len(delivered)}")
    print(f"Master manifest SHA256: {sha256(ARCHIVE / 'MANIFEST.json')}")
    print(f"Master checksums SHA256: {sha256(ARCHIVE / 'CHECKSUMS.sha256')}")


if __name__ == "__main__":
    main()
