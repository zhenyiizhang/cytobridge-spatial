#!/usr/bin/env python3
"""Execute the small examples, or explicitly select a study-data tutorial."""

from __future__ import annotations

import argparse
import json
import os
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

_pythonpath = os.environ.get("PYTHONPATH")
os.environ["PYTHONPATH"] = (
    str(ROOT) if not _pythonpath else os.pathsep.join((str(ROOT), _pythonpath))
)
os.environ.setdefault("PYDEVD_DISABLE_FILE_VALIDATION", "1")


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


def run(*, timeout: int, save_outputs: bool, dataset: str | None = None) -> list[dict[str, object]]:
    # Dataset notebooks train real models. Documentation checks only execute
    # the examples that do not require study data or a GPU.
    selected = [p for p in NOTEBOOKS if p.stem == dataset] if dataset else NOTEBOOKS[:2]
    if not selected:
        raise ValueError(f"Unknown dataset: {dataset}")
    return [
        execute_notebook(path, timeout=timeout, save_outputs=save_outputs)
        for path in selected
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument(
        "--dataset", choices=[p.stem for p in NOTEBOOKS[2:]],
        help="Run a study-data notebook, including GPU training. Add its data first.",
    )
    parser.add_argument(
        "--save-outputs",
        action="store_true",
        help="Store the executed cells in the published notebooks.",
    )
    args = parser.parse_args()
    print(json.dumps(run(timeout=args.timeout, save_outputs=args.save_outputs, dataset=args.dataset), indent=2))


if __name__ == "__main__":
    main()
