#!/usr/bin/env python3
"""Assemble MOSTA Main Figure 4 and export Supplementary Figures S9--S16."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from CytoBridge.results._cli import new_output_dir, write_run_summary
from CytoBridge.results.mosta_figures import (
    assemble_main_figure_4,
    export_mosta_supplementary_figures,
    load_mosta_figure_release,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--release-dir",
        type=Path,
        default=None,
        help=(
            "MOSTA reader-release directory. Defaults to the repository "
            "release artifact."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--preview-dpi", type=int, default=160)
    return parser.parse_args()


def run(
    release_dir: Path | None,
    output_dir: Path,
    *,
    dpi: int,
    preview_dpi: int,
) -> dict[str, object]:
    output = new_output_dir(output_dir)
    release = load_mosta_figure_release(release_dir)
    main_pdf, main_png = assemble_main_figure_4(
        release,
        output / "main_figure_4",
        dpi=dpi,
    )
    supplementary = export_mosta_supplementary_figures(
        release,
        output / "supplementary_figures",
        preview_dpi=preview_dpi,
    )
    summary: dict[str, object] = {
        "analysis": "mosta_manuscript_figures",
        "figure_action": "external-assembly + reference-export",
        "figure_index": (
            "supplementary_figures/mosta_figure_index.csv"
        ),
        "main_figure": {
            "paper_location": "Main Figure 4",
            "pdf": str(main_pdf.relative_to(output)),
            "png": str(main_png.relative_to(output)),
            "pdf_content": "five released vector panels plus two page connectors",
        },
        "supplementary_figures": {
            figure_id: {
                name: str(path.relative_to(output)) for name, path in files.items()
            }
            for figure_id, files in supplementary.items()
        },
        "current_supplementary_numbers": [f"S{number}" for number in range(9, 17)],
        "supplementary_content": "released vector PDF and SVG pages",
    }
    write_run_summary(output, summary)
    return summary


def main() -> None:
    args = parse_args()
    print(
        json.dumps(
            run(
                args.release_dir,
                args.output_dir,
                dpi=args.dpi,
                preview_dpi=args.preview_dpi,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
