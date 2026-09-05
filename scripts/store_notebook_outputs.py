"""Copy executed outputs into a notebook only when its code is unchanged."""
from __future__ import annotations

import argparse
from pathlib import Path

import nbformat


def store_outputs(source: Path, executed: Path) -> None:
    original = nbformat.read(source, as_version=4)
    run = nbformat.read(executed, as_version=4)
    source_cells = [c for c in original.cells if c.cell_type == "code"]
    run_cells = [c for c in run.cells if c.cell_type == "code"]
    if [c.source for c in source_cells] != [c.source for c in run_cells]:
        raise ValueError("Executed code differs from the published notebook.")
    for source_cell, run_cell in zip(source_cells, run_cells):
        if run_cell.execution_count is None:
            raise ValueError("Every code cell must have been executed.")
        if any(o.output_type == "error" for o in run_cell.outputs):
            raise ValueError("The executed notebook contains an error.")
        # Keep tables and calculated plots; omit library progress logs.
        source_cell.outputs = [o for o in run_cell.outputs if o.output_type != "stream"]
        source_cell.execution_count = run_cell.execution_count
    nbformat.write(original, source)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("executed", type=Path)
    args = parser.parse_args()
    store_outputs(args.source, args.executed)
