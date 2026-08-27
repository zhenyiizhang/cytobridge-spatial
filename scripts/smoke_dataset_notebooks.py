#!/usr/bin/env python3
"""Run the public dataset tutorials from top to bottom."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "docs" / "tutorials" / "dataset_workflows"
NOTEBOOKS = (
    ROOT / "docs" / "tutorials" / "your_data.ipynb",
    ROOT / "docs" / "tutorials" / "data_preparation" / "synthetic_preprocessing.ipynb",
    NOTEBOOK_DIR / "zebrafish.ipynb",
    NOTEBOOK_DIR / "mosta.ipynb",
    NOTEBOOK_DIR / "arista.ipynb",
    NOTEBOOK_DIR / "admouse.ipynb",
    NOTEBOOK_DIR / "chicken_heart.ipynb",
)


def execute_notebook(
    path: Path,
    *,
    timeout: int,
    save_outputs: bool,
) -> dict[str, object]:
    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
    )
    client.execute()

    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    executed_cells = sum(cell.execution_count is not None for cell in code_cells)
    if executed_cells != len(code_cells):
        raise RuntimeError(
            f"{path.name}: executed {executed_cells} of {len(code_cells)} code cells"
        )
    if save_outputs:
        nbformat.write(notebook, path)

    return {
        "notebook": path.name,
        "code_cells": len(code_cells),
        "executed_cells": executed_cells,
        "outputs": sum(len(cell.get("outputs", ())) for cell in code_cells),
    }


def run(*, timeout: int, save_outputs: bool) -> list[dict[str, object]]:
    return [
        execute_notebook(
            path,
            timeout=timeout,
            save_outputs=save_outputs,
        )
        for path in NOTEBOOKS
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument(
        "--save-outputs",
        action="store_true",
        help="Store the executed cells in the published notebooks.",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            run(timeout=args.timeout, save_outputs=args.save_outputs),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
