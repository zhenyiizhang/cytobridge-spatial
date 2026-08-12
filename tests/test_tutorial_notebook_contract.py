from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "01_synthetic_preprocessing_contract.ipynb"
README = ROOT / "README.md"


def _load_notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def test_synthetic_preprocessing_notebook_is_clean_and_stable() -> None:
    notebook = _load_notebook()
    cells = notebook["cells"]
    expected_ids = [
        "intro",
        "outline",
        "setup",
        "raw-contract",
        "make-data",
        "preprocess-step",
        "run-preprocess",
        "provenance-step",
        "inspect-provenance",
        "pitfall",
        "guard-demo",
        "exercise",
        "exercise-answer",
        "next-steps",
    ]

    assert notebook["nbformat"] == 4
    assert notebook["nbformat_minor"] >= 5
    assert [cell["id"] for cell in cells] == expected_ids
    assert len(set(expected_ids)) == len(expected_ids)

    for cell in cells:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None
            assert cell["outputs"] == []


def test_synthetic_preprocessing_notebook_has_explicit_scientific_contracts() -> None:
    notebook = _load_notebook()
    text = "".join(
        line
        for cell in notebook["cells"]
        for line in cell.get("source", [])
    )

    required_fragments = {
        "rng.poisson",
        'expression_layer="counts"',
        'raw_count_validation="strict"',
        'time_mapping={"E0": 0.0, "E1": 1.0, "E2": 2.0}',
        'required_latent_features=["Gene000", "Gene039"]',
        'counts[:, 39] = 0',
        "Path(CytoBridge.__file__).resolve()",
        'processed.uns["preprocess_info"]',
        '"double-transform"',
        "array_sha256",
        "python -m pip install -e '.[preprocess]'",
        "synthetic data only",
    }
    missing = {fragment for fragment in required_fragments if fragment not in text}
    assert not missing

    forbidden_fragments = {
        "read_h5ad(",
        "read_csv(",
        "/data/",
        "/home/",
        "/lustre/",
        "/Users/",
        "scp ",
        "ssh ",
        "pip install 'CytoBridge[preprocess]'",
    }
    forbidden_present = {fragment for fragment in forbidden_fragments if fragment in text}
    assert not forbidden_present


def test_readme_scopes_tutorial_to_a_source_checkout() -> None:
    readme = README.read_text(encoding="utf-8")
    assert "Run the source-checkout preprocessing tutorial" in readme
    tutorial = readme.split(
        "### Run the source-checkout preprocessing tutorial", 1
    )[1].split("\n### ", 1)[0]
    assert "From a source checkout" in tutorial
    assert "python -m pip install -e '.[preprocess]'" in tutorial
    assert "pip install 'CytoBridge[preprocess]'" not in tutorial
    assert "notebooks/01_synthetic_preprocessing_contract.ipynb" in tutorial
