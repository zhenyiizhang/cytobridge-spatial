#!/usr/bin/env python3
"""Render package-native ARISTA S12 with the submitted S12 plotting grammar.

Observed panels are read from the newly preprocessed package cohort. Generated
panels, including generated states at integer observed times, are read from the
visualization-only piecewise-warp run.  No panel-level display deletion is
applied: the clearly detached observed cells were already removed upstream by
the label-blind package spatial-QC rule.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import build_s12_s14_legacy_style_corrected as legacy


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PALETTE = (
    PROJECT_ROOT / "repositories/cb_reproducibility/assets/arista/label_to_color.json"
)
DEFAULT_LEGACY_EXTRA = (
    PROJECT_ROOT
    / "output/arista_paper_equivalent_corrected_20260822_3c87a3e/evidence/observed_display_exclusions_20.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-bank", required=True, type=Path)
    parser.add_argument("--formal-warp-dir", required=True, type=Path)
    parser.add_argument("--identity-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--palette", type=Path, default=DEFAULT_PALETTE)
    parser.add_argument("--legacy-extra-evidence", type=Path, default=DEFAULT_LEGACY_EXTRA)
    parser.add_argument("--audit-z-threshold", type=float, default=50.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    full_bank = args.full_bank.expanduser().resolve()
    formal_dir = args.formal_warp_dir.expanduser().resolve()
    identity_csv = args.identity_csv.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    palette_path = args.palette.expanduser().resolve()
    legacy_extra = args.legacy_extra_evidence.expanduser().resolve()
    directories = legacy.initialize_output(output_dir)
    palette = legacy.load_palette(palette_path)
    panels = legacy.load_s12_panels(
        full_bank,
        formal_dir,
        palette,
        identity_csv=identity_csv,
    )

    audit, _objective_flag_keys, audit_summary = legacy.build_outlier_audit(
        panels,
        palette,
        legacy_extra,
        directories["tables"],
        float(args.audit_z_threshold),
    )
    # The audit is diagnostic only. Upstream observed QC is the sole deletion
    # rule; generated states remain complete and are never used for training.
    name = "FigureS12_ARISTA_package_native_warpk1_oldstyle_FINAL"
    paths, inventory = legacy.plot_s12(
        panels,
        palette,
        set(),
        False,
        name,
        directories,
    )
    inventory.to_csv(
        directories["tables"] / "s12_panel_inventory_complete_display.csv",
        index=False,
    )
    flagged = audit[audit["objective_display_isolation_flag"]].copy()
    audit_summary["computation_policy"] = (
        "all package-QC observed rows and all generated rows are retained in S12"
    )
    audit_summary["legacy_20_policy"] = (
        "not applied; the package cohort was filtered upstream with the declared spatial-QC rule"
    )
    flagged.to_csv(
        directories["tables"] / "s12_generated_and_observed_isolation_diagnostic.csv",
        index=False,
    )

    script_snapshot = output_dir / "scripts" / Path(__file__).name
    script_snapshot.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__).resolve(), script_snapshot)
    formal_manifest = formal_dir / "manifest.json"
    input_paths = [palette_path, legacy_extra, formal_manifest, identity_csv]
    input_paths.extend(
        full_bank / "slice_data" / f"time_{legacy.time_token(time)}.h5ad"
        for time in legacy.OBSERVED_TIMES
    )
    input_paths.extend(
        formal_dir
        / "snapshots"
        / (
            f"time_{time:.1f}__Generated.svg"
            if time in legacy.OBSERVED_TIMES
            else f"time_{time:.1f}.svg"
        )
        for time in legacy.DENSE_TIMES
    )
    missing = [str(path) for path in input_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing S12 inputs:\n" + "\n".join(missing))

    output_files = sorted(path for path in output_dir.rglob("*") if path.is_file())
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "workflow": "ARISTA package-native S12 with visualization-only k=1 spatial warp and submitted plotting grammar",
        "scientific_contract": {
            "observed_cells_removed_upstream_only": True,
            "secondary_display_filter": False,
            "generated_integer_times_included": True,
            "generated_midpoints_included": True,
            "spatial_warp_visualization_only": True,
            "spatial_warp_k": 1,
            "communication_and_classification_use_prewarp_state": True,
            "total_display_rows": int(len(audit)),
        },
        "diagnostic_only_spatial_isolation_audit": {
            **audit_summary,
            "flagged_rows": int(len(flagged)),
            "affects_display": False,
        },
        "display_contract": {
            "layout": [
                None if item is None else {"time": item[0], "source": item[1]}
                for item in legacy.S12_LAYOUT
            ],
            "point_area_pt2": 2.5,
            "alpha": 0.9,
            "canvas_pt": list(legacy.S12_SUBMITTED_CANVAS_PT),
            "palette_sha256": legacy.sha256(palette_path),
        },
        "inputs": {str(path): legacy.sha256(path) for path in input_paths},
        "outputs": {
            str(path.relative_to(output_dir)): {
                "sha256": legacy.sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for path in output_files
        },
        "primary_outputs": {key: str(value) for key, value in paths.items()},
    }
    (output_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(output_dir), "primary": manifest["primary_outputs"]}, indent=2))


if __name__ == "__main__":
    main()
