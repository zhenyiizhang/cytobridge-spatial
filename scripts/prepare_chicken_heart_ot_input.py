#!/usr/bin/env python3
"""Prepare the raw-coordinate chicken-heart input for package OT alignment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from CytoBridge.pp.chicken_heart_input import (
    EXPECTED_COUNTS,
    OT_INPUT_SCHEMA_VERSION,
    TIMEPOINTS,
    prepare_chicken_heart_ot_adata,
    prepare_chicken_heart_ot_input,
)


SCHEMA_VERSION = OT_INPUT_SCHEMA_VERSION
prepare_ot_input = prepare_chicken_heart_ot_adata


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-h5ad", type=Path, required=True)
    parser.add_argument("--output-h5ad", type=Path, required=True)
    parser.add_argument("--output-table", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    manifest = prepare_chicken_heart_ot_input(
        input_h5ad=args.input_h5ad,
        output_h5ad=args.output_h5ad,
        output_table=args.output_table,
        manifest_path=args.manifest,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
