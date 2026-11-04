#!/usr/bin/env python3
"""Reproduce the zebrafish attention-validation figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from CytoBridge.results._cli import new_output_dir, write_run_summary
from CytoBridge.results.zebrafish_attention import (
    load_zebrafish_attention_results,
    plot_zebrafish_attention,
    write_zebrafish_attention_tables,
    zebrafish_attention_statistics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Directory containing the ten processed panel-data files. Defaults to packaged data.",
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
    results = load_zebrafish_attention_results(results_dir)
    tables = write_zebrafish_attention_tables(results, output)
    pdf, png = plot_zebrafish_attention(results, output)
    summary: dict[str, object] = {
        "analysis": "zebrafish_attention",
        "input": "packaged data" if results_dir is None else "results directory",
        "figure": {"pdf": pdf.name, "png": png.name},
        "tables": {name: path.name for name, path in tables.items()},
        "statistics": zebrafish_attention_statistics(results),
    }
    write_run_summary(output, summary)
    return summary


def main() -> None:
    args = parse_args()
    print(json.dumps(run(args.results_dir, args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
