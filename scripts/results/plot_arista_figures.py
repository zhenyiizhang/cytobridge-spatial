#!/usr/bin/env python3
"""Reproduce ARISTA Supplementary Figures S17--S22 from compact package data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from CytoBridge.results._cli import new_output_dir, write_run_summary
from CytoBridge.results.arista_supplementary_figures import (
    FIGURE_ORDER,
    calculate_arista_supplementary_pages,
    load_arista_supplementary_figures,
    plot_arista_supplementary_figures,
    select_arista_supplementary_pages,
    write_arista_supplementary_tables,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Directory containing compact ARISTA files. Defaults to packaged data.",
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
        choices=FIGURE_ORDER,
        default=list(FIGURE_ORDER),
        help="Current supplementary figure identifiers to export.",
    )
    return parser.parse_args()


def run(
    results_dir: Path | None,
    output_dir: Path,
    figures: tuple[str, ...],
) -> dict[str, object]:
    output = new_output_dir(output_dir)
    data = load_arista_supplementary_figures(results_dir)
    pages = calculate_arista_supplementary_pages(data)
    selected = select_arista_supplementary_pages(pages, figures)
    tables = write_arista_supplementary_tables(data, selected, output)
    rendered = plot_arista_supplementary_figures(
        data,
        output,
        selected,
        figures,
    )
    summary: dict[str, object] = {
        "analysis": "arista_supplementary_figures",
        "source": "packaged" if results_dir is None else data.source_dir.name,
        "figures": [page.figure for page in selected],
        "files": {
            figure: {"pdf": paths[0].name, "png": paths[1].name}
            for figure, paths in rendered.items()
        },
        "table_count": len(tables),
        "pdf_equivalence": "same page appearance and geometry; raster-only PDFs",
    }
    write_run_summary(output, summary)
    return summary


def main() -> None:
    args = parse_args()
    print(
        json.dumps(
            run(args.results_dir, args.output_dir, tuple(args.figures)),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
