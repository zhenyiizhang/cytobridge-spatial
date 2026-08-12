from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = {
    "01_zebrafish.ipynb": ("zebrafish", 10),
    "02_mosta.ipynb": ("mosta", 10),
    "03_arista.ipynb": ("arista", 10),
    "04_admouse.ipynb": ("admouse", 1),
}
SMOKE_RUNNER = ROOT / "scripts" / "smoke_dataset_notebooks.py"


@pytest.mark.parametrize(("filename", "expected"), NOTEBOOKS.items())
def test_dataset_notebook_is_clean_runnable_source(
    filename: str,
    expected: tuple[str, int],
) -> None:
    dataset, classifier_k = expected
    notebook = json.loads((ROOT / "notebooks" / filename).read_text(encoding="utf-8"))
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
    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        text_parts.append(source)
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None
            assert cell["outputs"] == []
            ast.parse(source)
    text = "\n".join(text_parts)

    assert f"DATASET_PRESET = '{dataset}'" in text
    assert "RUN_TRAINING = False" in text
    assert "RUN_FORMAL_SCOPE = False" in text
    assert 'downstream_preset = workflow_preset["downstream"]' in text
    assert "load_workflow_config(DATASET_PRESET)" in text
    assert "reuse_if_present=False" in text
    assert "compute_timepoint_communications" in text
    assert 'complex_mode="min"' in text
    assert "require_all_subunits=True" in text
    assert "keep quantitative analyses on unwarped states" in text.lower()
    assert f'"classifier_k": {classifier_k}' not in text
    assert "scientific_preset[\"classifier_k\"]" in text
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
    assert "if RUN_FORMAL_SCOPE:" in text
    assert "interpolation_times = list(FORMAL_INTERPOLATED_TIMES)" in text
    assert "analysis_particles = COMPACT_PARTICLES" in text
    assert 'scope_label = "formal preset"' in text
    assert 'scope_label = "compact walkthrough"' in text

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
    assert "NotebookClient" in source
    assert "estimate_neighborhood_threshold_from_aligned_spatial" in source
    assert "summarize_label_composition" in source
    assert "plot_celltype_composition" in source
    assert "does not train or load a model" in source
    assert "not a formal dataset execution" in source
    assert '"training_or_checkpoint_load": False' in source


def test_readme_and_docs_publish_all_dataset_notebooks() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    tutorial_index = (ROOT / "docs" / "tutorials" / "index.md").read_text(
        encoding="utf-8"
    )
    notebook_readme = (ROOT / "notebooks" / "README.md").read_text(
        encoding="utf-8"
    )

    for filename in NOTEBOOKS:
        assert filename in notebook_readme
    for dataset in ("zebrafish", "mosta", "arista", "admouse"):
        assert dataset in tutorial_index
    assert "four dataset notebooks" in readme
    assert "docs/" in readme
