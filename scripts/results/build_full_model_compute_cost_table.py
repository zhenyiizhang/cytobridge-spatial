#!/usr/bin/env python3
"""Build the full-model compute-cost table from packaged measurements."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from CytoBridge.results._cli import new_output_dir, write_run_summary
from CytoBridge.results.compute_cost import (
    load_full_model_compute_cost,
    write_full_model_compute_cost_tables,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Directory containing the raw table and manifest. Defaults to packaged data.",
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
    results = load_full_model_compute_cost(results_dir)
    paths = write_full_model_compute_cost_tables(results, output)
    summary: dict[str, object] = {
        "analysis": "full_model_compute_cost",
        "rows": len(results.measurements),
        "outputs": {name: path.name for name, path in paths.items()},
    }
    write_run_summary(output, summary)
    return summary


def main() -> None:
    args = parse_args()
    print(json.dumps(run(args.results_dir, args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
