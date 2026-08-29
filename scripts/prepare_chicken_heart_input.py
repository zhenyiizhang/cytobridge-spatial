#!/usr/bin/env python3
"""Prepare the reference-aligned chicken-heart input."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from CytoBridge.pp.chicken_heart import (
    ChickenHeartContractError as ContractError,
    apply_chicken_heart_coordinate_contract as _apply_anatomical_coordinate_contract,
    chicken_heart_anatomical_orientation_qc as _anatomical_orientation_qc,
)
from CytoBridge.pp.chicken_heart_input import (
    EXPECTED_COUNTS,
    INPUT_SCHEMA_VERSION,
    RAW_FILENAMES,
    REQUIRED_METADATA,
    TIME_MAPPING,
    TIMEPOINTS,
    _source_alignment_qc as _orientation_qc,
    _validate_reference_input as _validate_reference,
    assemble_chicken_heart_reference_counts,
    prepare_chicken_heart_input,
)


SCHEMA_VERSION = INPUT_SCHEMA_VERSION
assemble_reviewed_counts = assemble_chicken_heart_reference_counts
assemble_reviewed_chicken_heart_counts = assemble_chicken_heart_reference_counts
prepare = prepare_chicken_heart_input


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--metadata-h5ad", type=Path, required=True)
    parser.add_argument("--aligned-reference-h5ad", type=Path, required=True)
    parser.add_argument("--output-h5ad", type=Path, required=True)
    parser.add_argument("--output-table", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--graph-database", type=Path)
    parser.add_argument(
        "--repair-legacy-d7-left-right",
        action="store_true",
        help=(
            "Reflect only D7 around its stage mean x when the input fails solely "
            "because of the known D7 RV/LV left-right mirror."
        ),
    )
    return parser


def main() -> None:
    options = _parser().parse_args()
    manifest = prepare_chicken_heart_input(
        raw_dir=options.raw_dir,
        metadata_h5ad=options.metadata_h5ad,
        aligned_reference_h5ad=options.aligned_reference_h5ad,
        output_h5ad=options.output_h5ad,
        output_table=options.output_table,
        manifest_path=options.manifest,
        graph_database=options.graph_database,
        repair_legacy_d7_left_right=options.repair_legacy_d7_left_right,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
