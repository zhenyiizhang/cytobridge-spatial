#!/usr/bin/env python3
"""Reproduce corrected ARISTA Supplementary Figures S21 and S22 from CSV tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from CytoBridge.results._cli import new_output_dir, write_run_summary
from CytoBridge.results.arista_supplementary_figures import (
    calculate_arista_ligand_receptor_panels,
    load_arista_supplementary_figures,
    plot_arista_ligand_receptor_figures,
    write_arista_ligand_receptor_tables,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Directory containing the released ARISTA tables. Defaults to package data.",
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
    data = load_arista_supplementary_figures(results_dir)
    panels = calculate_arista_ligand_receptor_panels(data)
    figures = plot_arista_ligand_receptor_figures(data, output, panels)
    tables = write_arista_ligand_receptor_tables(data, output, panels)
    counts = (
        panels.prototypes
        .groupby("cluster")["n_pairs"]
        .first()
        .astype(int)
        .to_dict()
    )
    roster = panels.display_roster
    selected_k = int(
        panels.k_selection.sort_values(
            ["silhouette", "k"], ascending=[False, True]
        ).iloc[0]["k"]
    )
    summary: dict[str, object] = {
        "analysis": "arista_ligand_receptor_figures",
        "calculation": "deterministic clustering of all released LR time courses",
        "input_profiles": int(len(panels.assignments)),
        "selected_k": selected_k,
        "figures": {
            figure: {"pdf": paths[0].name, "png": paths[1].name}
            for figure, paths in figures.items()
        },
        "cluster_counts": {str(key): value for key, value in counts.items()},
        "displayed_pairs": int(len(roster)),
        "displayed_per_cluster": {
            str(key): int(value)
            for key, value in roster.groupby("cluster").size().items()
        },
        "tables": {name: path.name for name, path in tables.items()},
    }
    write_run_summary(output, summary)
    return summary


def main() -> None:
    args = parse_args()
    print(json.dumps(run(args.results_dir, args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
