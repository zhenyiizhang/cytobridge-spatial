#!/usr/bin/env python3
"""Create an immutable, self-contained corrected MOSTA S11 figure bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import stat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--working-root", required=True)
    parser.add_argument("--baseline-component-root", required=True)
    parser.add_argument("--seed-stability-root", required=True)
    parser.add_argument("--style-notebook", required=True)
    parser.add_argument("--style-oracle-svg", required=True)
    parser.add_argument("--archive-root", required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_file(source: Path, target: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def copy_tree(source: Path, target: Path) -> None:
    if not source.is_dir():
        raise NotADirectoryError(source)
    shutil.copytree(
        source,
        target,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )


def main() -> None:
    args = parse_args()
    working = Path(args.working_root).resolve()
    baseline = Path(args.baseline_component_root).resolve()
    stability = Path(args.seed_stability_root).resolve()
    notebook = Path(args.style_notebook).resolve()
    oracle = Path(args.style_oracle_svg).resolve()
    archive = Path(args.archive_root).resolve()
    if archive.exists():
        raise FileExistsError(f"Immutable archive target exists: {archive}")
    archive.mkdir(parents=True, exist_ok=False)

    for directory in ("figures", "tables", "source", "qa", "caption"):
        copy_tree(working / directory, archive / directory)
    copy_file(working / "PROVENANCE.md", archive / "PROVENANCE.md")

    copy_tree(baseline / "tables" / "M_sum", archive / "numerical_truth" / "seed42_M_sum")
    copy_file(baseline / "summary.json", archive / "numerical_truth" / "component_audit_summary.json")
    copy_file(
        baseline / "tables" / "communication_component_exact_audit.csv",
        archive / "numerical_truth" / "communication_component_exact_audit.csv",
    )
    copy_file(baseline / "SHA256SUMS.txt", archive / "numerical_truth" / "source_component_SHA256SUMS.txt")
    copy_tree(baseline / "provenance", archive / "numerical_truth" / "component_provenance")

    copy_file(stability / "summary.json", archive / "sampling_stability" / "summary.json")
    copy_file(stability / "SHA256SUMS.txt", archive / "sampling_stability" / "source_stability_SHA256SUMS.txt")
    copy_tree(stability / "tables", archive / "sampling_stability" / "tables")
    copy_tree(stability / "provenance", archive / "sampling_stability" / "provenance")

    copy_file(notebook, archive / "style_truth" / notebook.name)
    copy_file(oracle, archive / "style_truth" / oracle.name)

    figure_stem = "Figure_S11_MOSTA_corrected_package_Msum_average_k3_exact_submitted_style_representative31"
    pdf = archive / "figures" / f"{figure_stem}.pdf"
    svg = archive / "figures" / f"{figure_stem}.svg"
    selection = archive / "tables" / "s11_msum_stable_representative31.csv"
    qa = json.loads((archive / "qa" / "figure_bundle_validation.json").read_text(encoding="utf-8"))
    if qa.get("status") != "PASS" or qa.get("errors"):
        raise RuntimeError("Figure QA has not passed")

    manifest = {
        "schema_version": 1,
        "status": "ACCEPTED_IMMUTABLE_S11_PANEL",
        "dataset": "MOSTA",
        "panel": "Supplementary Figure S11",
        "decision": {
            "numerical_truth": "accepted corrected MOSTA release + global-t0 fully generated states + package-native M_sum",
            "style_truth": "submitted MOSTA S11 notebook and SVG oracle",
            "rejected_candidate": "M_per_source k=3 because temporal global-magnitude decline plus rowwise min-max created synchronized half-step pulses",
            "legacy_normalization_used": False,
            "manual_smoothing_used": False,
            "rotation_stretch_or_warp": False,
            "arista_assets_used": False,
        },
        "package": {
            "commit": "2b3c79eff3face7c4dd33de24d45384b9dbd8a84",
            "archive_sha256": "06852992db0d8ebedd8f0baa19a3f539bcdd271dfc0c6fae9774dd553c8fe55e",
            "reference_h5ad_sha256": "8b9ca0ad3475040235036548d54b96272bf6c49f057f6c2a643152c11350ce25",
            "finetune_sha256": "d08d21b86fdcd92c748bb54ca81f998fbb157ca5a7acd9548bbfe16c573bfaa5",
            "score_sha256": "d7d06657f8548618db1bc85409e73305fbf59feb9446793550b2c9761639e52a",
        },
        "computation": {
            "matrix_key": "M_sum",
            "times": [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
            "communication_sample_size": 12000,
            "expression": "inverse-PCA count space at every time",
            "complex_mode": "min",
            "require_all_subunits": True,
            "n_retained_pairs": 1757,
            "clustering": "minmax + average linkage exact k=3 + peak-time order",
            "seed42_cluster_sizes": {"1": 1632, "2": 108, "3": 17},
            "display_cluster_counts": {"1": 12, "2": 11, "3": 8},
            "display_selection": "stable clusters at seeds 42/43/44; top-half within-cluster effect-size gate; then nearest to seed-specific prototypes",
        },
        "input_contracts": {
            "baseline_component_manifest_sha256": sha256(baseline / "SHA256SUMS.txt"),
            "seed_stability_manifest_sha256": sha256(stability / "SHA256SUMS.txt"),
            "style_notebook_sha256": sha256(notebook),
            "style_oracle_svg_sha256": sha256(oracle),
        },
        "outputs": {
            "pdf": {"path": str(pdf.relative_to(archive)), "sha256": sha256(pdf)},
            "svg": {"path": str(svg.relative_to(archive)), "sha256": sha256(svg)},
            "selection": {"path": str(selection.relative_to(archive)), "sha256": sha256(selection)},
            "qa_status": qa["status"],
            "pdf_vector_embedded_images": qa["details"]["embedded_images"],
            "pdf_page_size_pt": [qa["details"]["page_width_pt"], qa["details"]["page_height_pt"]],
        },
    }
    (archive / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (archive / "COMPLETE").write_text("complete\n", encoding="utf-8")
    files = sorted(path for path in archive.rglob("*") if path.is_file() and path.name != "CHECKSUMS.sha256")
    (archive / "CHECKSUMS.sha256").write_text(
        "\n".join(f"{sha256(path)}  {path.relative_to(archive)}" for path in files) + "\n",
        encoding="utf-8",
    )
    for path in archive.rglob("*"):
        if path.is_file():
            path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    for directory in sorted((p for p in archive.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
        directory.chmod(0o555)
    archive.chmod(0o555)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
