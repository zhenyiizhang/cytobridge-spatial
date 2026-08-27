#!/usr/bin/env python3
"""Reproduce zebrafish Supplementary Figures S31--S38."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from CytoBridge.results._cli import new_output_dir, write_run_summary
from CytoBridge.results.zebrafish_si import (
    FIGURE_IDS,
    calculate_zebrafish_si_panels,
    load_zebrafish_si_results,
    plot_zebrafish_si,
    write_zebrafish_si_tables,
    zebrafish_si_statistics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Directory containing the compact result bundle. Defaults to packaged data.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New or empty output directory.",
    )
    parser.add_argument(
        "--figures",
        nargs="+",
        choices=FIGURE_IDS,
        default=list(FIGURE_IDS),
        help="Figure identifiers to render in display order.",
    )
    return parser.parse_args()


def run(
    results_dir: Path | None,
    output_dir: Path,
    figures: tuple[str, ...] | list[str] = FIGURE_IDS,
) -> dict[str, object]:
    output = new_output_dir(output_dir)
    results = load_zebrafish_si_results(results_dir)
    panels = calculate_zebrafish_si_panels(results)
    tables = write_zebrafish_si_tables(panels, output)
    rendered = plot_zebrafish_si(results, output, panels, figures)
    summary: dict[str, object] = {
        "analysis": "zebrafish_si_s31_s38",
        "source": "packaged" if results_dir is None else "results directory",
        "figures": {
            figure_id: {"pdf": paths[0].name, "png": paths[1].name}
            for figure_id, paths in rendered.items()
        },
        "tables": {name: path.name for name, path in tables.items()},
        "statistics": zebrafish_si_statistics(results, panels),
    }
    write_run_summary(output, summary)
    return summary


def main() -> None:
    args = parse_args()
    print(
        json.dumps(
            run(args.results_dir, args.output_dir, args.figures),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
