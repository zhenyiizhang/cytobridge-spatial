#!/usr/bin/env python3
"""Build the immutable MOSTA S9/S10 archive manifest from archived files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(path: Path, root: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True, type=Path)
    args = parser.parse_args()
    root = args.archive.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)

    audit = load_json(root / "audit" / "numerical_audit.json")
    if audit.get("status") != "PASS" or audit.get("errors"):
        raise RuntimeError("Numerical audit must be PASS with no errors")
    cluster = load_json(root / "numerical_inputs" / "clusterprofiler_server_run" / "manifest.json")
    shared = load_json(root / "numerical_inputs" / "shared_summary.json")
    render = load_json(root / "render_manifest.json")
    validation = {
        "s9": load_json(root / "qa" / "s9_validation_report.json"),
        "s10": load_json(root / "qa" / "s10_validation_report.json"),
    }
    for panel, report in validation.items():
        if report.get("errors") or report.get("warnings"):
            raise RuntimeError(f"{panel} validation is not clean")

    excluded = {"MANIFEST.json", "CHECKSUMS.sha256"}
    files = {
        path.relative_to(root).as_posix(): identity(path, root)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name not in excluded
    }
    manifest = {
        "schema_version": 1,
        "dataset": "MOSTA",
        "panels": ["S9a-b", "S10a-d"],
        "status": "PASS_READY_TO_SEAL",
        "principle": "corrected package-native numerical truth with submitted MOSTA style truth",
        "package": {
            "commit": shared["package_commit"],
            "release": shared["package_release"],
        },
        "model": shared["model"],
        "clusterprofiler_contract": cluster["calculation_contract"],
        "clusterprofiler_software": cluster["software"],
        "clusterprofiler_queries": cluster["queries"],
        "numerical_audit": audit,
        "render_contract": render["scientific_contract"],
        "style_authority": render["style_authority"],
        "validation": validation,
        "final_outputs": {
            "s9_pdf": files["figures/s9/Figure_S9_MOSTA_latest_package_clusterProfiler_GO_exact_submitted_style.pdf"],
            "s9_svg": files["figures/s9/Figure_S9_MOSTA_latest_package_clusterProfiler_GO_exact_submitted_style.svg"],
            "s10_pdf": files["figures/s10/Figure_S10_MOSTA_latest_package_DP3_clusterProfiler_GO_exact_submitted_style.pdf"],
            "s10_svg": files["figures/s10/Figure_S10_MOSTA_latest_package_DP3_clusterProfiler_GO_exact_submitted_style.svg"],
        },
        "coordinate_warp": False,
        "arista_content_used": False,
        "files": files,
    }
    output = root / "MANIFEST.json"
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    checksum_path = root / "CHECKSUMS.sha256"
    checksum_lines = [
        f"{sha256(path)}  {path.relative_to(root).as_posix()}"
        for path in sorted(root.rglob("*"))
        if path.is_file() and path != checksum_path
    ]
    checksum_path.write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    print(output)
    print(checksum_path)


if __name__ == "__main__":
    main()
