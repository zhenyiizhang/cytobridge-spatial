from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
DATASET_NOTEBOOK_DIR = ROOT / "docs" / "tutorials" / "dataset_workflows"
OWN_DATA_NOTEBOOK = ROOT / "docs" / "tutorials" / "your_data.ipynb"
NOTEBOOKS = {
    "zebrafish.ipynb": "zebrafish",
    "mosta.ipynb": "mosta",
    "arista.ipynb": "arista",
    "admouse.ipynb": "admouse",
    "chicken_heart.ipynb": "chicken_heart",
}
SMOKE_RUNNER = ROOT / "scripts" / "smoke_dataset_notebooks.py"


def _notebook_text(notebook: dict) -> tuple[str, str]:
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    markdown = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "markdown"
    )
    return code, markdown


@pytest.mark.parametrize(("filename", "preset"), NOTEBOOKS.items())
def test_dataset_notebook_is_executed_and_portable(
    filename: str,
    preset: str,
) -> None:
    path = DATASET_NOTEBOOK_DIR / filename
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["kernelspec"] == {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }

    cell_ids = [cell["id"] for cell in notebook["cells"]]
    assert len(cell_ids) == len(set(cell_ids))

    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert code_cells
    assert all(cell["execution_count"] is not None for cell in code_cells)
    assert sum(len(cell.get("outputs", ())) for cell in code_cells) >= 4
    for cell in code_cells:
        ast.parse("".join(cell.get("source", [])))

    code, markdown = _notebook_text(notebook)
    public_text = f"{code}\n{markdown}".casefold()
    serialized = json.dumps(notebook, ensure_ascii=False)

    forbidden_public_phrases = {
        "learning goals",
        "scientific question",
        "scientific contract",
        "notes and interpretation",
        "exercise",
        "reviewer",
        "hash-verified",
        "sha256",
    }
    assert not {phrase for phrase in forbidden_public_phrases if phrase in public_text}
    for machine_path in ("/Users/", "/home/", "/tmp/", "/private/tmp/"):
        assert machine_path not in serialized

    headings = (
        "## Setup",
        "## Data preparation",
        "## Training",
        "## Downstream analysis",
        "## Paper figures",
        "## Saved files",
    )
    positions = [markdown.index(heading) for heading in headings]
    assert positions == sorted(positions)

    assert f"PRESET = '{preset}'" in code
    assert "load_workflow_config(PRESET)" in code
    assert "WorkflowOptions" in code
    assert "build_workflow_plan" in code
    assert "render_workflow_plan" in code
    assert "run_workflow" in code
    assert "RUN_PREPARATION = False" in code
    assert "RUN_PREPROCESS_AND_TRAIN = False" in code
    assert "RUN_DOWNSTREAM = False" in code
    assert 'scientific["classifier_k"]' in code
    assert 'steps=("preprocess", "train")' in code
    assert 'steps=("downstream",)' in code
    assert "train=True" in code
    assert "sys.path" not in code
    assert "Path(__file__)" not in code

    if preset == "chicken_heart":
        assert "prepare_chicken_heart_input" in code
        assert "prepare_chicken_heart_ot_input" in code
        assert "RUN_RAW_DATA_ASSEMBLY = False" in code


def test_notebook_runner_executes_the_published_sources() -> None:
    source = SMOKE_RUNNER.read_text(encoding="utf-8")
    ast.parse(source)
    assert all(filename in source for filename in NOTEBOOKS)
    assert "synthetic_preprocessing.ipynb" in source
    assert "your_data.ipynb" in source
    assert "NotebookClient" in source
    assert "--save-outputs" in source
    assert "client.execute()" in source
    assert "executed_cells != len(code_cells)" in source
    assert "replace_source" not in source
    assert "synthetic API wiring" not in source


def test_dataset_notebook_generator_matches_public_notebooks() -> None:
    source = (ROOT / "scripts" / "build_dataset_tutorials.py").read_text(
        encoding="utf-8"
    )
    ast.parse(source)
    assert all(preset in source for preset in NOTEBOOKS.values())
    assert "Learning goals" not in source
    assert "Notes and interpretation" not in source
    assert "run_workflow" in source
    assert "build_own_data_notebook" in source


def test_own_data_notebook_is_executed_and_has_complete_commands() -> None:
    notebook = json.loads(OWN_DATA_NOTEBOOK.read_text(encoding="utf-8"))
    code, markdown = _notebook_text(notebook)
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert all(cell["execution_count"] is not None for cell in code_cells)
    assert "--export-config" in code
    assert "--dry-run" in code
    assert "--train" in code
    assert "--step downstream" in code
    assert "## Expected output locations" in markdown


def test_tutorial_navigation_has_one_dataset_section() -> None:
    tutorial_index = (ROOT / "docs" / "tutorials" / "index.md").read_text(
        encoding="utf-8"
    )
    dataset_index = (DATASET_NOTEBOOK_DIR / "index.md").read_text(encoding="utf-8")
    paper_index = (
        ROOT / "docs" / "tutorials" / "paper_figures" / "index.md"
    ).read_text(encoding="utf-8")

    assert tutorial_index.count("## Reuse a paper dataset workflow") == 1
    assert "Dataset notebooks" not in tutorial_index
    assert tutorial_index.count("dataset_workflows/index") == 2
    assert tutorial_index.count("your_data") == 2
    assert "paper_figures/index" in tutorial_index
    for preset in NOTEBOOKS.values():
        assert preset in dataset_index
    for figure in ("main_figure_2", "main_figure_4", "main_figure_5"):
        assert figure in paper_index


def test_readme_lists_the_notebook_locations() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    notebook_readme = (ROOT / "notebooks" / "README.md").read_text(encoding="utf-8")

    for filename in NOTEBOOKS:
        assert filename in notebook_readme
    assert "docs/tutorials/data_preparation" in notebook_readme
    assert "docs/tutorials/dataset_workflows" in notebook_readme
    assert "docs/tutorials/paper_figures" in notebook_readme
    assert "## Tutorials and paper figures" in readme
    assert "docs/tutorials/" in readme
