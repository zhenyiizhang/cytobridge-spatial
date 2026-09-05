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
PAPER_STEP_COUNTS = {
    "zebrafish.ipynb": 4,
    "mosta.ipynb": 2,
    "arista.ipynb": 4,
    "admouse.ipynb": 4,
    "chicken_heart.ipynb": 4,
}
NOTEBOOK_RUNNER = ROOT / "scripts" / "execute_dataset_notebooks.py"


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


@pytest.mark.parametrize(("filename", "dataset"), NOTEBOOKS.items())
def test_dataset_notebook_is_executable_and_portable(filename: str, dataset: str) -> None:
    notebook = json.loads((DATASET_NOTEBOOK_DIR / filename).read_text())
    assert notebook["metadata"]["cytobridge"]["runs_training"] is False
    code, markdown = _notebook_text(notebook)
    assert notebook['metadata']['cytobridge']['dataset'] == dataset
    required = {
        'chicken_heart': ('run_interpolation_workflow', 'evaluate_growth_by_timepoint'),
        'mosta': ('run_interpolation_workflow', 'evaluate_growth_by_timepoint', 'summarize_label_composition'),
        'arista': ('run_interpolation_workflow', 'evaluate_growth_by_timepoint'),
        'admouse': ('simulate_sde_points_split', 'predict_labels_for_trajectories', 'summarize_label_composition'),
        'zebrafish': ('model_state_adata', 'evaluate_growth_by_timepoint'),
    }
    for function in ('load_dynamical_model_from_dir', *required[dataset]):
        assert f"cb.tl.{function}(" in code
    assert "run_workflow(" not in code
    assert "RUN_TRAINING" not in code
    assert "build_workflow_plan" not in code
    assert '.to_csv(' in code
    assert 'Image(filename=' not in code
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert all(cell["execution_count"] is not None for cell in code_cells)
    assert sum("image/png" in out.get("data", {})
               for cell in code_cells for out in cell["outputs"]) >= {
                   'chicken_heart': 2, 'mosta': 3, 'arista': 1, 'admouse': 1, 'zebrafish': 1,
               }[dataset]
    assert not any(out.get('output_type') == 'error'
                   for cell in code_cells for out in cell['outputs'])
    # Real study-data tutorials are not executed by a documentation build.
    # A successful branch that skips training is not a reproduced result.
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
    # Each dataset teaches its own paper analysis, not an identical generic recipe.
    assert '## Load' in markdown
    assert '## Draw' in markdown or '## Spatial' in markdown
    assert "data_checkpoints.md" in markdown
    assert "training.md" in markdown
    assert "PROJECT_DIR" in markdown and "PROJECT_DIR" in code
    text = (code + markdown).lower()
    for phrase in ("learning goals", "exercise", "notes and interpretation",
                   "handoff", "hash-verified", "smoke", "dry run"):
        assert phrase not in text
    for path in ("/Users/", "/home/", "/tmp/"):
        assert path not in json.dumps(notebook)


def test_notebook_runner_executes_the_published_sources() -> None:
    source = NOTEBOOK_RUNNER.read_text(encoding="utf-8")
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
    assert "handoff" not in source.casefold()
    assert "build_analysis_notebook" in source
    assert "build_own_data_notebook" in source
    assert "build_synthetic_preprocessing_notebook" in source


def test_own_data_notebook_is_executed_and_has_complete_commands() -> None:
    notebook = json.loads(OWN_DATA_NOTEBOOK.read_text())
    code, markdown = _notebook_text(notebook)
    assert all(
        cell["execution_count"] is not None
        for cell in notebook["cells"] if cell["cell_type"] == "code"
    )
    assert "--train" in markdown
    assert "train=True" not in code  # This page does not start training during a docs check.
    assert "config[\"preprocess\"][\"annotation_source\"]" in code
    assert "align.pop(\"spatial_obs_keys\", None)" in code
    assert 'config["preprocess"]["batch_indices"] = None' in code
    assert "time_mapping" in code
    assert "reuse_model.md" in markdown
    assert "## Open the results" in markdown


def test_tutorial_navigation_has_one_dataset_section() -> None:
    tutorial_index = (ROOT / "docs" / "tutorials" / "index.md").read_text(
        encoding="utf-8"
    )
    dataset_index = (DATASET_NOTEBOOK_DIR / "index.md").read_text(encoding="utf-8")
    paper_index = (
        ROOT / "docs" / "tutorials" / "paper_figures" / "index.md"
    ).read_text(encoding="utf-8")

    assert tutorial_index.count("## Paper datasets") == 1
    assert "Dataset notebooks" not in tutorial_index
    assert tutorial_index.count("dataset_workflows/") >= 5
    assert "your_data" in tutorial_index
    home = (ROOT / "docs/index.md").read_text()
    assert home.count("tutorials/dataset_workflows/index") == 2  # Card and toctree.
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
