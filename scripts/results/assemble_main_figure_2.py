#!/usr/bin/env python3
"""Assemble Main Figure 2 from frozen panels a--d and panel-e result tables."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from CytoBridge.results._cli import new_output_dir, write_run_summary
from CytoBridge.results.main_figure_2 import (
    assemble_main_figure_2,
    load_main_figure_2,
    write_main_figure_2_tables,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Directory containing the compact inputs. Defaults to packaged data.",
    )
    parser.add_argument(
        "--output-dir", required=True, type=Path, help="New or empty output directory."
    )
    parser.add_argument(
        "--dpi", type=int, default=300, help="Raster output resolution."
    )
    return parser.parse_args()


def run(
    results_dir: Path | None, output_dir: Path, *, dpi: int = 300
) -> dict[str, object]:
    output = new_output_dir(output_dir)
    data = load_main_figure_2(results_dir)
    tables = write_main_figure_2_tables(data, output)
    pdf, png = assemble_main_figure_2(data, output, dpi=dpi)
    summary: dict[str, object] = {
        "analysis": "main_figure_2",
        "figure_action": "result-summary-redraw + external-assembly",
        "frozen_panels": ["a", "b", "c", "d"],
        "redrawn_panels": ["e"],
        "input_directory": str(data.source_dir),
        "pdf": str(pdf),
        "png": str(png),
        "tables": {name: str(path) for name, path in tables.items()},
    }
    write_run_summary(output, summary)
    return summary


def main() -> None:
    args = parse_args()
    print(
        json.dumps(
            run(args.results_dir, args.output_dir, dpi=args.dpi),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
