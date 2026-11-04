#!/usr/bin/env python3
"""Reproduce the LR-prior and external-method comparison figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from CytoBridge.results._cli import new_output_dir, write_run_summary
from CytoBridge.results.interaction_evidence import (
    load_interaction_evidence_results,
    plot_interaction_evidence,
    write_interaction_evidence_tables,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Directory containing the compact result files. Defaults to packaged data.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="New or empty output directory.",
    )
    return parser.parse_args()


def run(results_dir: Path | None, output_dir: Path) -> dict[str, object]:
    output = new_output_dir(output_dir)
    results = load_interaction_evidence_results(results_dir)
    tables = write_interaction_evidence_tables(results, output)
    pdf, png = plot_interaction_evidence(results, output)
    summary: dict[str, object] = {
        "analysis": "interaction_evidence",
        "input_directory": str(results.source_dir),
        "pdf": str(pdf),
        "png": str(png),
        "tables": {name: str(path) for name, path in tables.items()},
    }
    write_run_summary(output, summary)
    return summary


def main() -> None:
    args = parse_args()
    print(json.dumps(run(args.results_dir, args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
