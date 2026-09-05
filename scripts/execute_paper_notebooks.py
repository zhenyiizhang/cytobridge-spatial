#!/usr/bin/env python3
"""Execute the paper-figure notebooks and optionally save their outputs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

import nbformat
from nbclient import NotebookClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = PROJECT_ROOT / "docs" / "tutorials" / "paper_figures"
OPERATIONS = {
    "main_figure_2": "redraw panel e and assemble existing panels a-d",
    "main_figure_4": "assemble existing vector panels",
    "main_figure_5": "copy an existing figure page",
    "mosta_figures": "export existing figure pages",
    "arista_figures": "redraw S23-S24; display existing S19-S22 pages",
    "compute_cost": "format recorded measurements as a table",
}

# The kernels run in temporary output directories. Keep the source checkout on
# their import path when this script is used before installing a wheel.
_pythonpath = os.environ.get("PYTHONPATH")
os.environ["PYTHONPATH"] = (
    str(PROJECT_ROOT)
    if not _pythonpath
    else os.pathsep.join((str(PROJECT_ROOT), _pythonpath))
)
os.environ.setdefault("PYDEVD_DISABLE_FILE_VALIDATION", "1")


def _portable_output_paths(notebook, run_dir: Path) -> None:
    """Replace temporary run-directory prefixes in displayed path values."""

    prefixes = {str(run_dir), str(run_dir.resolve())}

    def replace(value):
        if isinstance(value, str):
            for prefix in prefixes:
                value = value.replace(f"{prefix}/", "")
            return value
        if isinstance(value, list):
            for index, item in enumerate(value):
                value[index] = replace(item)
            return value
        if isinstance(value, dict):
            for key, item in value.items():
                value[key] = replace(item)
            return value
        return value

    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = replace(cell.get("outputs", []))


def execute_notebook(
    path: Path,
    run_dir: Path,
    *,
    timeout: int,
    save_outputs: bool,
) -> dict[str, object]:
    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(run_dir)}},
    )
    client.execute()
    _portable_output_paths(notebook, run_dir)
    if save_outputs:
        nbformat.write(notebook, path)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    return {
        "notebook": path.name,
        "operation": OPERATIONS.get(path.stem, "draw figures from included numerical results"),
        "training_performed": False,
        "code_cells": len(code_cells),
        "executed_cells": sum(cell.execution_count is not None for cell in code_cells),
        "outputs": sum(len(cell.get("outputs", ())) for cell in code_cells),
    }


def run(
    output_dir: Path,
    *,
    timeout: int,
    save_outputs: bool = False,
    notebook_names: tuple[str, ...] = (),
) -> list[dict[str, object]]:
    if notebook_names:
        notebooks = []
        for name in notebook_names:
            filename = name if name.endswith(".ipynb") else f"{name}.ipynb"
            path = NOTEBOOK_DIR / filename
            if not path.is_file():
                raise FileNotFoundError(f"Unknown paper-figure notebook: {name}")
            notebooks.append(path)
        if len(notebooks) != len(set(notebooks)):
            raise ValueError("A notebook was selected more than once")
    else:
        notebooks = sorted(NOTEBOOK_DIR.glob("*.ipynb"))
    if not notebooks:
        raise FileNotFoundError(f"No notebooks found in {NOTEBOOK_DIR}")
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for path in notebooks:
        run_dir = output_dir / path.stem
        run_dir.mkdir()
        summaries.append(
            execute_notebook(
                path,
                run_dir,
                timeout=timeout,
                save_outputs=save_outputs,
            )
        )
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--report", type=Path, help="Save the execution summary as JSON.")
    parser.add_argument(
        "--notebook",
        action="append",
        default=[],
        help=(
            "Notebook filename or stem to execute. Repeat this option to run "
            "several notebooks; omit it to run the full paper-figure set."
        ),
    )
    parser.add_argument(
        "--save-outputs",
        action="store_true",
        help="Store the executed cells in the published notebooks.",
    )
    args = parser.parse_args()

    if args.output_dir is not None:
        summaries = run(
            args.output_dir.expanduser().resolve(),
            timeout=args.timeout,
            save_outputs=args.save_outputs,
            notebook_names=tuple(args.notebook),
        )
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(summaries, indent=2) + "\n")
        print(json.dumps(summaries, indent=2))
        return

    with tempfile.TemporaryDirectory(prefix="cytobridge-paper-notebooks-") as tmp:
        summaries = run(
            Path(tmp),
            timeout=args.timeout,
            save_outputs=args.save_outputs,
            notebook_names=tuple(args.notebook),
        )
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(summaries, indent=2) + "\n")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
