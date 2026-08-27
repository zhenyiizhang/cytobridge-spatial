#!/usr/bin/env python3
"""Reproduce the ARISTA local interaction-domain figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from CytoBridge.results._cli import new_output_dir, write_run_summary
from CytoBridge.results.arista_local_domains import (
    calculate_arista_local_domain_panels,
    load_arista_local_domains,
    plot_arista_local_domains,
    write_arista_local_domain_tables,
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
    data = load_arista_local_domains(results_dir)
    panels = calculate_arista_local_domain_panels(data)
    tables = write_arista_local_domain_tables(panels, output)
    pdf, png = plot_arista_local_domains(data, output, panels)
    summary: dict[str, object] = {
        "analysis": "arista_local_domains",
        "source": "packaged" if results_dir is None else data.source_dir.name,
        "rows": {
            "roi_assignments": len(data.roi_assignments),
            "domain_metadata": len(data.domain_metadata),
            "celltype_edges": len(data.celltype_edges),
            "attention_null": len(data.attention_null),
            "pathway_null": len(data.pathway_null),
            "lr_pair_null": len(data.lr_pair_null),
        },
        "files": {"pdf": pdf.name, "png": png.name},
        "tables": {name: path.name for name, path in tables.items()},
    }
    write_run_summary(output, summary)
    return summary


def main() -> None:
    args = parse_args()
    print(json.dumps(run(args.results_dir, args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
