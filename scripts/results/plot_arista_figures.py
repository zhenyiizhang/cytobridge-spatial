#!/usr/bin/env python3
"""Export released ARISTA S17--S22 pages as reference assets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from CytoBridge.results._cli import new_output_dir, write_run_summary
from CytoBridge.results.arista_supplementary_figures import (
    FIGURE_ORDER,
    calculate_arista_supplementary_pages,
    load_arista_figure_release,
    load_arista_supplementary_figures,
    plot_arista_supplementary_figures,
    resolve_arista_release_dir,
    select_arista_supplementary_pages,
    write_arista_source_index,
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
        "--release-dir",
        type=Path,
        default=None,
        help=(
            "Repository ARISTA release directory. When unavailable, the compact "
            "package export remains usable."
        ),
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
    release_dir: Path | None,
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
    try:
        formal_root = resolve_arista_release_dir(release_dir)
    except FileNotFoundError:
        if release_dir is not None:
            raise
        formal_release = None
    else:
        formal_release = load_arista_figure_release(formal_root)
    formal_index = (
        write_arista_source_index(formal_release, output)
        if formal_release is not None
        else None
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
        "export_type": "released reference pages; no analysis is recalculated",
        "pdf_equivalence": "same page appearance and geometry; raster-only PDFs",
        "formal_source_index": None if formal_index is None else formal_index.name,
    }
    write_run_summary(output, summary)
    return summary


def main() -> None:
    args = parse_args()
    print(
        json.dumps(
            run(
                args.results_dir,
                args.release_dir,
                args.output_dir,
                tuple(args.figures),
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
