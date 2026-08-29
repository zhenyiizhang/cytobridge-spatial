#!/usr/bin/env python3
"""Build package-native ARISTA S13--S14 in the submitted SI style."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import build_s12_s14_legacy_style_corrected as legacy


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PALETTE_CANDIDATES = (
    PROJECT_ROOT / "repositories/cb_reproducibility/assets/arista/label_to_color.json",
    PROJECT_ROOT
    / "release_artifacts/arista_package_native_spatialqc_z50_retrain_20260824_r1"
    / "main_run/downstream/label_to_color.json",
)
DEFAULT_PALETTE = next(
    (path for path in PALETTE_CANDIDATES if path.is_file()), PALETTE_CANDIDATES[0]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--downstream-dir", required=True, type=Path)
    parser.add_argument("--formal-warp-dir", required=True, type=Path)
    parser.add_argument(
        "--lineage-dir",
        type=Path,
        default=None,
        help="Optional corrected Sankey directory; defaults to --formal-warp-dir.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--palette", type=Path, default=DEFAULT_PALETTE)
    parser.add_argument("--font-family", default="DejaVu Sans")
    parser.add_argument(
        "--annotate-injury-reference",
        action="store_true",
        help=(
            "Annotate the right-hemisphere lower injury region on the t=0 "
            "reference panel without changing growth values or normalization."
        ),
    )
    parser.add_argument(
        "--annotate-injury-all-panels",
        action="store_true",
        help=(
            "Repeat the same schematic lower-right-hemisphere injury ROI on "
            "all nine S13 panels. This implies --annotate-injury-reference."
        ),
    )
    return parser.parse_args()


def legacy_seed42_sample(growth: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(legacy.DISPLAY_SAMPLE_SEED)
    sampled = []
    for time, sub in growth.groupby("time", sort=True):
        sub = sub.sort_values("cell_index").reset_index(drop=True)
        n_display = min(legacy.DISPLAY_SAMPLE_CAP, len(sub))
        chosen = np.sort(rng.choice(len(sub), size=n_display, replace=False))
        selected = sub.iloc[chosen].copy()
        selected["source"] = (
            "observed" if float(time) in legacy.OBSERVED_TIMES else "simulated"
        )
        selected["composite_id"] = ""
        selected["n_compute_panel"] = int(len(sub))
        selected["display_sample_rank"] = np.arange(len(selected), dtype=int)
        selected["objective_isolation_flag"] = False
        selected["s12_source_row_id"] = [
            f"generated:{float(time):.1f}:{int(index)}"
            if float(time) not in legacy.OBSERVED_TIMES
            else "package-QC-observed"
            for index in selected["cell_index"]
        ]
        sampled.append(selected)
    result = pd.concat(sampled, ignore_index=True)
    expected = legacy.DISPLAY_SAMPLE_CAP * len(legacy.DENSE_TIMES)
    if len(result) != expected:
        raise AssertionError(f"S13 legacy sample changed: {len(result)} != {expected}")
    return result


def main() -> None:
    args = parse_args()
    downstream = args.downstream_dir.expanduser().resolve()
    formal = args.formal_warp_dir.expanduser().resolve()
    lineage_dir = (
        args.lineage_dir.expanduser().resolve() if args.lineage_dir is not None else formal
    )
    output_dir = args.output_dir.expanduser().resolve()
    palette_path = args.palette.expanduser().resolve()
    directories = legacy.initialize_output(output_dir)
    palette = legacy.load_palette(palette_path)

    growth_path = downstream / "growth/growth_by_cell.csv"
    composition_path = formal / "fixed_particle_composition.csv"
    required = [
        growth_path,
        composition_path,
        lineage_dir / "lineage_sankey.svg",
        lineage_dir / "lineage_sankey.pdf",
        lineage_dir / "lineage_sankey.png",
        formal / "manifest.json",
        palette_path,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing S13/S14 inputs:\n" + "\n".join(missing))

    growth = pd.read_csv(growth_path)
    if tuple(sorted(growth["time"].astype(float).unique())) != legacy.DENSE_TIMES:
        raise ValueError("S13 growth table does not contain the nine-time dense grid")
    sample = legacy_seed42_sample(growth)
    sample.to_csv(directories["tables"] / "s13_seed42_display_sample.csv", index=False)
    s13_name = "FigureS13_ARISTA_package_native_oldstyle_FINAL"
    s13_paths, s13_scales = legacy.plot_s13(
        sample,
        False,
        s13_name,
        directories,
        directories["tables"],
        fit_package_native_canvas=True,
        annotate_injury_reference=(
            args.annotate_injury_reference or args.annotate_injury_all_panels
        ),
        annotate_injury_all_panels=args.annotate_injury_all_panels,
        font_family=args.font_family,
    )

    s14a_name = "PanelS14a_ARISTA_package_native_fixed_particle_lineage"
    s14b_name = "PanelS14b_ARISTA_package_native_fixed_particle_composition"
    s14_name = "FigureS14_ARISTA_package_native_oldstyle_FINAL"
    s14a_paths = legacy.copy_s14a_sources(lineage_dir, s14a_name, directories)
    count_table, fraction_table = legacy.calculate_corrected_composition(composition_path)
    count_table.to_csv(
        directories["tables"] / "s14_fixed_particle_counts.csv", index_label="time"
    )
    fraction_table.to_csv(
        directories["tables"] / "s14_fixed_particle_fractions.csv", index_label="time"
    )
    s14b_paths, _display_pct = legacy.plot_s14b(
        fraction_table,
        palette,
        s14b_name,
        directories,
        directories["tables"],
    )
    s14_paths = legacy.render_s14_composite(
        s14a_paths["svg"], s14b_paths["svg"], s14_name, directories
    )

    script_snapshot = output_dir / "scripts" / Path(__file__).name
    script_snapshot.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__).resolve(), script_snapshot)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "workflow": "ARISTA package-native S13-S14 with submitted plotting grammar",
        "scientific_contract": {
            "s13_growth_source": str(growth_path),
            "s13_compute_rows": int(len(growth)),
            "s13_display_sample_seed": legacy.DISPLAY_SAMPLE_SEED,
            "s13_display_cap_per_time": legacy.DISPLAY_SAMPLE_CAP,
            "s13_secondary_display_filter": False,
            "s13_injury_annotation": (
                "the same schematic anatomical locator is repeated on all nine "
                "panels in the right-hemisphere lower region; not inferred from "
                "growth values"
                if args.annotate_injury_all_panels
                else (
                    "schematic anatomical locator on t=0 right-hemisphere lower region; "
                    "not inferred from growth values"
                    if args.annotate_injury_reference
                    else None
                )
            ),
            "s14_identity_source": "non_split_fixed_particles",
            "s14_particle_counts_by_time": {
                str(float(time)): int(value)
                for time, value in count_table.sum(axis=1).items()
            },
        },
        "display_contract": {
            "S13": (
                "submitted 3x3 viridis seed-42 sample/cap with per-panel q05-q95; "
                "optional fixed anatomical injury locator"
            ),
            "S14a": "nature-methods Plotly Sankey in submitted compact crop",
            "S14b": "submitted 11x4.8-inch top-15-plus-Other stacked-bar grammar",
            "palette_sha256": legacy.sha256(palette_path),
            "font_family": str(args.font_family),
        },
        "inputs": {str(path): legacy.sha256(path) for path in required},
        "primary_outputs": {
            "S13": {key: str(path) for key, path in s13_paths.items()},
            "S14": {key: str(path) for key, path in s14_paths.items()},
        },
        "s13_scale_table": s13_scales.to_dict(orient="records"),
    }
    (output_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(output_dir), "primary": manifest["primary_outputs"]}, indent=2))


if __name__ == "__main__":
    main()
