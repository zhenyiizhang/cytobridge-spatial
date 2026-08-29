#!/usr/bin/env python3
"""Finalize, checksum, and verify the reader-facing MOSTA release bundle."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


DEST = Path(
    "/private/tmp/cytobridge_arista_release_20260825/"
    "release_artifacts/mosta_package_native_corrected_20260826_v1"
)

EXPECTED_MODELS = {
    "model/checkpoints/Finetune/best_model.pth":
        "d08d21b86fdcd92c748bb54ca81f998fbb157ca5a7acd9548bbfe16c573bfaa5",
    "model/checkpoints/Score_Refine/score_model.pth":
        "d7d06657f8548618db1bc85409e73305fbf59feb9446793550b2c9761639e52a",
    "model/classifier_cache/classifier_resmlp_6d2d7acf7d0ed92d.pt":
        "f938575c145baa9002de695c39a8637f57d6cb3a06ccfaf4d18b707ca962a7e0",
}

EXPECTED_COMPLETE_FIG4 = {
    "figures/main/Figure_4_complete.pdf":
        "45beb12c6314052c4e33ce73255dcd8511a2e9e81e0a765ad858b0961cf80b40",
    "figures/main/Figure_4_complete.svg":
        "e4c993d0b73456b83ce933196c5e0e70468a78bf59c93dc1b24665b891c5d73e",
    "figures/main/Figure_4_complete_300dpi.png":
        "d3a33f830ac66f343382b54d1b0fed383ee1a6205bdf02eb437138a2759b559a",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


for relative, expected in {**EXPECTED_MODELS, **EXPECTED_COMPLETE_FIG4}.items():
    path = DEST / relative
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f"Identity mismatch: {relative}: {actual} != {expected}")

master_delivery = json.loads(
    (DEST / "provenance/master_delivery/MANIFEST.json").read_text(encoding="utf-8")
)
figure_records = []
for path in sorted((DEST / "figures").rglob("*")):
    if path.is_file():
        figure_records.append(
            {
                "path": str(path.relative_to(DEST)),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
        )

manifest = {
    "schema_version": 1,
    "status": "READER_RELEASE_COMPLETE",
    "created_date": "2026-08-26",
    "scope": {
        "main": ["Figure 4a", "Figure 4b", "Figure 4c", "Figure 4d", "Figure 4e", "complete Figure 4"],
        "supplementary": ["Figure S4", "Figure S5", "Figure S6", "Figure S7", "Figure S8", "Figure S9", "Figure S10", "Figure S11"],
        "out_of_scope": "Figure S12 onward is ARISTA",
    },
    "software": {
        "repository": "https://github.com/zhenyiizhang/cytobridge-spatial",
        "branch": "release/cytobridge-reproducible-20260812",
        "package_commit_used_for_all_numerical_results": "2b3c79eff3face7c4dd33de24d45384b9dbd8a84",
    },
    "model": {
        "accepted_server_run": "/data/cytobridge/projects/CytoBridge-ST-1104/runs/corrected-matched-ablation-20260813-3c87a3e-r1/mosta",
        "files": {
            relative: {"sha256": expected, "bytes": (DEST / relative).stat().st_size}
            for relative, expected in EXPECTED_MODELS.items()
        },
        "config": {
            "path": "model/checkpoints/config.yaml",
            "sha256": sha256(DEST / "model/checkpoints/config.yaml"),
        },
    },
    "data": {
        "aligned_h5ad_sha256": "8b9ca0ad3475040235036548d54b96272bf6c49f057f6c2a643152c11350ce25",
        "aligned_h5ad_size": "15 GB",
        "git_policy": "not stored in Git; rebuild from public MOSTA data with packaged preprocessing",
        "public_source_documentation": "../../docs/data_checkpoints.md",
    },
    "numerical_contract": {
        "seed": 42,
        "alpha_spatial": 10,
        "alpha_express": 0.015,
        "classifier_k": 10,
        "global_t0_for_generated_intermediate_stages": True,
        "shared_dense_trajectory_initial_particles": 50000,
        "arista_data_labels_palette_model_or_analysis_used": False,
    },
    "figures": figure_records,
    "accepted_panel_manifest": master_delivery,
    "visual_contract": {
        "style_authority": "submitted manuscript/SI, historical MOSTA code, and Figure_mouse1.ai",
        "complete_figure4_transform": "translation only; scale 1; rotation 0; no warp",
        "vector_first": True,
        "intentional_raster_layers": "dense scatter/spatial data layers only",
    },
}
(DEST / "MANIFEST.json").write_text(
    json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
)

verify = '''#!/usr/bin/env python3
"""Verify every file in the MOSTA reader release against CHECKSUMS.sha256."""
from __future__ import annotations
import hashlib
from pathlib import Path

root = Path(__file__).resolve().parent
errors = []
for line in (root / "CHECKSUMS.sha256").read_text().splitlines():
    if not line.strip():
        continue
    expected, relative = line.split("  ", 1)
    path = root / relative
    if not path.is_file():
        errors.append(f"missing: {relative}")
        continue
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected:
        errors.append(f"mismatch: {relative}: {digest} != {expected}")
if errors:
    raise SystemExit("\\n".join(errors))
print("PASS: all MOSTA reader-release files match CHECKSUMS.sha256")
'''
(DEST / "verify_release.py").write_text(verify, encoding="utf-8")

paths = [
    path for path in sorted(DEST.rglob("*"))
    if path.is_file() and path.name != "CHECKSUMS.sha256"
]
lines = [f"{sha256(path)}  {path.relative_to(DEST)}" for path in paths]
(DEST / "CHECKSUMS.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")

print(
    json.dumps(
        {
            "status": "PASS",
            "files": len(paths),
            "bytes": sum(path.stat().st_size for path in paths),
            "manifest_sha256": sha256(DEST / "MANIFEST.json"),
            "checksums_sha256": sha256(DEST / "CHECKSUMS.sha256"),
        },
        indent=2,
    )
)
