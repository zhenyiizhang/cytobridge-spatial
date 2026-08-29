#!/usr/bin/env python3
"""Reproduce AGIST Supplementary Figures S2 and S3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from CytoBridge.results._cli import new_output_dir, write_run_summary
from CytoBridge.results.agist_figures import (
    calculate_agist_figure_panels,
    load_agist_figures,
    plot_agist_figures,
    write_agist_figure_tables,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Directory containing compact inputs. Defaults to packaged data.",
    )
    parser.add_argument(
        "--output-dir", required=True, type=Path, help="New or empty output directory."
    )
    return parser.parse_args()


def run(results_dir: Path | None, output_dir: Path) -> dict[str, object]:
    output = new_output_dir(output_dir)
    data = load_agist_figures(results_dir)
    panels = calculate_agist_figure_panels(data)
    tables = write_agist_figure_tables(panels, output)
    figures = plot_agist_figures(data, panels, output)
    summary: dict[str, object] = {
        "analysis": "agist_figures",
        "input_directory": str(data.source_dir),
        "figures": {
            figure: {"pdf": str(paths[0]), "png": str(paths[1])}
            for figure, paths in figures.items()
        },
        "tables": {name: str(path) for name, path in tables.items()},
    }
    write_run_summary(output, summary)
    return summary


def main() -> None:
    args = parse_args()
    print(json.dumps(run(args.results_dir, args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
