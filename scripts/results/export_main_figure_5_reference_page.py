#!/usr/bin/env python3
"""Export the packaged ARISTA Main Figure 5 reference page."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from CytoBridge.results._cli import new_output_dir, write_run_summary
from CytoBridge.results.main_figure_5 import (
    export_main_figure_5_reference_page,
    load_main_figure_5,
    validate_main_figure_5_reference_page,
    write_main_figure_5_tables,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Directory containing compact Figure 5 files. Defaults to packaged data.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New or empty output directory.",
    )
    return parser.parse_args()


def run(results_dir: Path | None, output_dir: Path) -> dict[str, object]:
    output = new_output_dir(output_dir)
    data = load_main_figure_5(results_dir)
    page = validate_main_figure_5_reference_page(data)
    tables = write_main_figure_5_tables(data, page, output)
    pdf, png = export_main_figure_5_reference_page(data, output, page)
    summary: dict[str, object] = {
        "analysis": "main_figure_5",
        "figure_action": "reference-export",
        "numerical_recalculation": False,
        "scientific_label_release": data.manifest["scientific_label_release"],
        "source": "packaged" if results_dir is None else data.source_dir.name,
        "canvas_pixels": [page.width_pixels, page.height_pixels],
        "page_points": [page.width_points, page.height_points],
        "panel_count": page.panel_count,
        "files": {"pdf": pdf.name, "png": png.name},
        "tables": {name: path.name for name, path in tables.items()},
        "pdf_equivalence": "same page appearance and labels; raster-only PDF",
    }
    write_run_summary(output, summary)
    return summary


def main() -> None:
    args = parse_args()
    print(json.dumps(run(args.results_dir, args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
