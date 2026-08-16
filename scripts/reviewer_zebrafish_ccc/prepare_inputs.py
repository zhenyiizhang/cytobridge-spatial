#!/usr/bin/env python3
"""Prepare one shared, counts-derived input bundle for spatial CCC methods."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .common import json_dump, prepare_inputs, write_input_bundle
except ImportError:  # direct ``python path/to/prepare_inputs.py`` execution
    from common import json_dump, prepare_inputs, write_input_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5ad", type=Path, required=True)
    parser.add_argument("--lr-database", type=Path, required=True)
    parser.add_argument(
        "--preprocess-audit",
        type=Path,
        required=False,
        help="Passed formal audit that freezes the primary normalization target and input hashes.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--counts-layer", default="counts")
    parser.add_argument("--label-col", default="Annotation")
    parser.add_argument("--stage-col", default="time_point_processed")
    parser.add_argument("--time-col", default="time")
    parser.add_argument("--spatial-key", default="spatial_aligned")
    parser.add_argument(
        "--target-sum",
        default="audit",
        help=(
            "Use 'audit' with --preprocess-audit, or supply the exact numeric "
            "normalization target recorded by the accepted H5AD."
        ),
    )
    parser.add_argument("--integer-tolerance", type=float, default=1e-5)
    parser.add_argument("--source-x-tolerance", type=float, default=1e-10)
    parser.add_argument("--max-cells-per-stage", type=int, default=0)
    parser.add_argument("--subsample-seed", type=int, default=20260722)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.target_sum.lower() == "audit":
        if args.preprocess_audit is None:
            raise ValueError("--preprocess-audit is required with --target-sum audit")
        target_sum: float | str = "audit"
    else:
        target_sum = float(args.target_sum)
        if target_sum <= 0:
            raise ValueError("--target-sum must be positive or 'audit'")
    if args.max_cells_per_stage < 0:
        raise ValueError("--max-cells-per-stage must be zero or positive")
    prepared = prepare_inputs(
        args.h5ad,
        args.lr_database,
        preprocess_audit_path=args.preprocess_audit,
        counts_layer=args.counts_layer,
        label_col=args.label_col,
        stage_col=args.stage_col,
        time_col=None if args.time_col.lower() == "none" else args.time_col,
        spatial_key=args.spatial_key,
        target_sum=target_sum,
        integer_tolerance=args.integer_tolerance,
        source_x_tolerance=args.source_x_tolerance,
    )
    manifest = write_input_bundle(
        prepared,
        args.out_dir,
        source_h5ad=args.h5ad,
        source_lr_database=args.lr_database,
        source_preprocess_audit=args.preprocess_audit,
        max_cells_per_stage=args.max_cells_per_stage,
        subsample_seed=args.subsample_seed,
    )
    json_dump(
        {
            "out_dir": str(args.out_dir.resolve()),
            "n_stages": len(manifest["stages"]),
            "n_lr_rows": manifest["database"]["rows"],
        },
        args.out_dir / "prepare_summary.json",
    )
    print(f"Prepared {len(manifest['stages'])} stages in {args.out_dir}")


if __name__ == "__main__":
    main()
