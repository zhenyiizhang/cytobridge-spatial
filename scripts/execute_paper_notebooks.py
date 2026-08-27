#!/usr/bin/env python3
"""Execute the compact paper-figure notebooks in temporary directories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import nbformat
from nbclient import NotebookClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = PROJECT_ROOT / "docs" / "tutorials" / "paper_figures"


def execute_notebook(path: Path, run_dir: Path, *, timeout: int) -> dict[str, object]:
    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(run_dir)}},
    )
    client.execute()
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    return {
        "notebook": path.name,
        "code_cells": len(code_cells),
        "executed_cells": sum(cell.execution_count is not None for cell in code_cells),
    }


def run(output_dir: Path, *, timeout: int) -> list[dict[str, object]]:
    notebooks = sorted(NOTEBOOK_DIR.glob("*.ipynb"))
    if not notebooks:
        raise FileNotFoundError(f"No notebooks found in {NOTEBOOK_DIR}")
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for path in notebooks:
        run_dir = output_dir / path.stem
        run_dir.mkdir()
        summaries.append(execute_notebook(path, run_dir, timeout=timeout))
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()

    if args.output_dir is not None:
        summaries = run(args.output_dir.expanduser().resolve(), timeout=args.timeout)
        print(json.dumps(summaries, indent=2))
        return

    with tempfile.TemporaryDirectory(prefix="cytobridge-paper-notebooks-") as tmp:
        summaries = run(Path(tmp), timeout=args.timeout)
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
