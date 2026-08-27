from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
DATASET_NOTEBOOK_DIR = ROOT / "docs" / "tutorials" / "dataset_workflows"
NOTEBOOKS = {
    "zebrafish.ipynb": ("zebrafish", 10),
    "mosta.ipynb": ("mosta", 10),
    "arista.ipynb": ("arista", 10),
    "admouse.ipynb": ("admouse", 1),
}
SMOKE_RUNNER = ROOT / "scripts" / "smoke_dataset_notebooks.py"
CHICKEN_NOTEBOOK = DATASET_NOTEBOOK_DIR / "chicken_heart.ipynb"


@pytest.mark.parametrize(("filename", "expected"), NOTEBOOKS.items())
def test_dataset_notebook_is_clean_runnable_source(
    filename: str,
    expected: tuple[str, int],
) -> None:
    dataset, classifier_k = expected
    notebook = json.loads((DATASET_NOTEBOOK_DIR / filename).read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["kernelspec"] == {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    language_version = tuple(
        int(part)
        for part in notebook["metadata"]["language_info"]["version"].split(".")[:2]
    )
    assert language_version == (3, 11)
    assert language_version < (3, 12)

    cell_ids = [cell["id"] for cell in notebook["cells"]]
    assert len(cell_ids) == len(set(cell_ids))

    text_parts: list[str] = []
    markdown_parts: list[str] = []
    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        text_parts.append(source)
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None
            assert cell["outputs"] == []
            ast.parse(source)
        else:
            markdown_parts.append(source)
    text = "\n".join(text_parts)
    markdown = "\n".join(markdown_parts)
    public_text = text.casefold()

    forbidden_public_phrases = {
        "learning goals",
        "## outline",
        "reading the results",
        "exercise",
        "scientific question",
        "scientific contract",
        "notes and interpretation",
        "reviewer",
        "manuscript",
        "claim",
        "hash-verified",
    }
    assert not {phrase for phrase in forbidden_public_phrases if phrase in public_text}
    assert "sys.path" not in text
    assert "repo_root" not in public_text
    assert "Path(__file__)" not in text
    assert "/Users/" not in text
    assert "/home/" not in text

    section_order = [
        "## 1. Set input and output paths",
        "## 2. Load the packaged preset",
        "## 3. Optional training",
        "## 4. Load the model checkpoint",
        "## 5. Calculate interpolated states and cell labels",
        "## 6. Calculate cell-type composition",
        "## 7. Calculate velocity and growth",
        "## 8. Calculate sparse cell-type communication",
        "## 9. Calculate ligand–receptor trajectories",
        "## 10. Calculate distribution metrics",
        "## Outputs",
    ]
    section_positions = [markdown.index(heading) for heading in section_order]
    assert section_positions == sorted(section_positions)

    assert "import CytoBridge as cb" in text
    assert f"DATASET_PRESET = '{dataset}'" in text
    assert "LR_DATABASE_OVERRIDE: Path | None = None" in text
    assert "if LR_DATABASE_OVERRIDE is None:" in text
    assert "LR_DATABASE = cb.pp.bundled_graph_database_path(DATASET_PRESET)" in text
    assert "required_external_inputs = {" in text
    assert 'Path("inputs/CellChatDB.ligrec.' not in text
    assert "a ligand-receptor table" not in text
    assert "Provide the external files for the wheel-packaged" not in text
    assert '"lr_database": LR_DATABASE,\n    **(' not in text
    assert "RUN_TRAINING = False" in text
    assert "RUN_FULL_SCOPE = False" in text
    assert 'downstream_preset = workflow_preset["downstream"]' in text
    assert "load_workflow_config(DATASET_PRESET)" in text
    assert "reuse_if_present=False" in text
    assert "compute_timepoint_communications" in text
    assert 'record["edge_selection"]' in text
    assert "save_dense_attention_matrix=False" in text
    assert 'complex_mode="min"' in text
    assert "require_all_subunits=True" in text
    assert "spatial_warp_to_observed=False" in text
    assert "trajectory.communication_adata_dict" in text
    assert f'"classifier_k": {classifier_k}' not in text
    assert 'scientific_preset["classifier_k"]' in text
    assert 'downstream_preset["classifier_epochs"]' in text
    assert 'downstream_preset["classifier_hidden_size"]' in text
    assert 'downstream_preset["classifier_lr"]' in text
    assert 'downstream_preset["classifier_best_metric"]' in text
    assert 'downstream_preset["classifier_strict_stratification"]' in text

    assert 'downstream_preset["observed"]' in text
    assert 'downstream_preset["interpolated"]' in text
    assert 'downstream_preset["sde_n_samples"]' in text
    assert 'downstream_preset["sde_dt"]' in text
    assert 'downstream_preset["split_sde_dt"]' in text
    assert 'downstream_preset["split_sigma"]' in text
    assert 'downstream_preset["split_growth_alpha"]' in text
    assert 'downstream_preset.get("lineage_enabled", False)' in text
    assert "if RUN_FULL_SCOPE:" in text
    assert "interpolation_times = list(PRESET_INTERPOLATED_TIMES)" in text
    assert "analysis_particles = COMPACT_PARTICLES" in text

    assert "classifier_epochs=CLASSIFIER_EPOCHS" in text
    assert "classifier_hidden_size=CLASSIFIER_HIDDEN_SIZE" in text
    assert "classifier_lr=CLASSIFIER_LR" in text
    assert "classifier_best_metric=CLASSIFIER_BEST_METRIC" in text
    assert "classifier_strict_stratification=CLASSIFIER_STRICT_STRATIFICATION" in text
    assert "sde_n_samples=analysis_particles" in text
    assert "sde_dt=SDE_DT" in text
    assert "split_sde_dt=SPLIT_SDE_DT" in text
    assert "split_sigma_scalar=SPLIT_SIGMA" in text
    assert "split_growth_alpha=SPLIT_GROWTH_ALPHA" in text
    assert "skip_nonsplit_sde=not LINEAGE_ENABLED" in text
    assert "n_samples=evaluation_particles" in text
    assert "dt=SDE_DT" in text
    assert "sigma=SPLIT_SIGMA" in text

    forbidden_hardcoding = {
        "TUTORIAL_PARTICLES",
        "split_sde_dt=0.01",
        "split_sde_dt=0.05",
        "split_sigma_scalar=0.03",
        "classifier_epochs=500",
        "classifier_hidden_size=128",
        "classifier_lr=1e-3",
        "dt=0.01",
        "sigma=0.03",
    }
    assert not {fragment for fragment in forbidden_hardcoding if fragment in text}


def test_notebook_smoke_runner_is_explicitly_limited_and_parseable() -> None:
    source = SMOKE_RUNNER.read_text(encoding="utf-8")
    ast.parse(source)
    assert all(filename in source for filename in NOTEBOOKS)
    assert CHICKEN_NOTEBOOK.name in source
    assert "NotebookClient" in source
    assert "estimate_neighborhood_threshold_from_aligned_spatial" in source
    assert "summarize_label_composition" in source
    assert "plot_celltype_composition" in source
    assert "does not train or load a model" in source
    assert "not a full dataset execution" in source
    assert '"training_or_checkpoint_load": False' in source


def test_chicken_heart_notebook_uses_the_installed_package_workflow() -> None:
    notebook = json.loads(CHICKEN_NOTEBOOK.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    text_parts = []
    markdown_parts = []
    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        text_parts.append(source)
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None
            assert cell["outputs"] == []
            ast.parse(source)
        else:
            markdown_parts.append(source)
    text = "\n".join(text_parts)
    markdown = "\n".join(markdown_parts)
    lowered = text.casefold()

    section_order = [
        "## 1. Inputs",
        "## 2. Prepare or validate data",
        "## 3. Load the preset and build a plan",
        "## 4. Train a new model or load a checkpoint",
        "## 5. Run downstream analysis and save outputs",
        "## Outputs",
    ]
    positions = [markdown.index(heading) for heading in section_order]
    assert positions == sorted(positions)

    assert "import CytoBridge as cb" in text
    assert 'DATASET_PRESET = "chicken_heart"' in text
    assert "load_workflow_config(DATASET_PRESET)" in text
    assert "build_workflow_plan" in text
    assert "render_workflow_plan" in text
    assert "run_workflow" in text
    assert "cb.pp.prepare_chicken_heart_input" in text
    assert "cb.pp.prepare_chicken_heart_ot_input" in text
    assert "validate_prepared_chicken_heart_input" in text
    assert "validate_chicken_heart_ot_input" in text
    assert "RUN_PREPARATION = False" in text
    assert "REPAIR_LEGACY_D7_LEFT_RIGHT = False" in text
    assert "RUN_TRAINING = False" in text
    assert "LOAD_MODEL = False" in text
    assert "RUN_DOWNSTREAM = False" in text
    assert "MODEL_DIR = Path(" in text
    assert "TRAIN_OUTPUT_DIR = Path(" in text
    assert "load_dynamical_model_from_dir" in text
    assert '"cuda" if torch.cuda.is_available() else "cpu"' in text
    assert "cuda:0" not in lowered
    assert "subprocess" not in text
    assert "sys.path" not in text
    assert "repo_root" not in lowered
    assert "spatial_data" not in lowered
    assert "scripts/prepare_chicken_heart" not in lowered
    assert not {
        phrase
        for phrase in (
            "learning goals",
            "scientific contract",
            "pitfalls",
            "notes and interpretation",
            "exercise",
            "reviewer",
            "signed",
            "immutable",
            "claim",
            "sha256",
            "sha-256",
        )
        if phrase in lowered
    }


def test_admouse_notebook_uses_packaged_edge_settings_and_panel_projection() -> None:
    notebook = json.loads(
        (DATASET_NOTEBOOK_DIR / "admouse.ipynb").read_text(encoding="utf-8")
    )
    text = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert 'interaction_config["edge_predictor_thre"]' in text
    assert "EDGE_PREDICTOR_PATH" in text
    assert "EDGE_PREDICTOR_THRESHOLD" in text
    assert "0.9956824779510498" in text
    assert "seven complete ligand–receptor" in text
    assert 'complex_mode="min"' in text
    assert "require_all_subunits=True" in text
    assert "save_dense_attention_matrix=False" in text
    assert "AGE_ANCHORS_MONTHS = {0.0: 2.5, 1.0: 5.7, 2.0: 17.9}" in text


def test_readme_and_docs_publish_all_dataset_notebooks() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    tutorial_index = (ROOT / "docs" / "tutorials" / "index.md").read_text(
        encoding="utf-8"
    )
    notebook_readme = (ROOT / "notebooks" / "README.md").read_text(encoding="utf-8")

    for filename in NOTEBOOKS:
        assert filename in notebook_readme
    assert CHICKEN_NOTEBOOK.name in notebook_readme
    assert "docs/tutorials/data_preparation" in notebook_readme
    assert "docs/tutorials/dataset_workflows" in notebook_readme
    assert "docs/tutorials/paper_figures" in notebook_readme
    for dataset in ("zebrafish", "mosta", "arista", "admouse", "chicken_heart"):
        assert dataset in tutorial_index
    assert "## Tutorials and paper figures" in readme
    assert "docs/tutorials/" in readme
    assert "Dataset workflows show the package calls" in readme
