from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = (
    ROOT / "docs" / "tutorials" / "data_preparation" / "synthetic_preprocessing.ipynb"
)
README = ROOT / "README.md"


def _load_notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def test_synthetic_preprocessing_notebook_is_clean_and_stable() -> None:
    notebook = _load_notebook()
    cells = notebook["cells"]
    expected_ids = [
        "intro",
        "setup",
        "raw-input",
        "make-data",
        "preprocessing",
        "run-preprocess",
        "metadata",
        "inspect-metadata",
        "input-validation",
        "double-transform-check",
        "outputs",
    ]

    assert notebook["nbformat"] == 4
    assert notebook["nbformat_minor"] >= 5
    assert [cell["id"] for cell in cells] == expected_ids
    assert len(set(expected_ids)) == len(expected_ids)

    for cell in cells:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None
            assert cell["outputs"] == []


def test_synthetic_preprocessing_notebook_uses_public_api_and_checks_outputs() -> None:
    notebook = _load_notebook()
    text = "".join(
        line for cell in notebook["cells"] for line in cell.get("source", [])
    )

    required_fragments = {
        "rng.poisson",
        'expression_layer="counts"',
        'raw_count_validation="strict"',
        'time_mapping={"E0": 0.0, "E1": 1.0, "E2": 2.0}',
        'required_latent_features=["Gene000", "Gene039"]',
        "counts[:, 39] = 0",
        'version("CytoBridge")',
        'processed.uns["preprocess_info"]',
        '"double-transform"',
        "preprocess_summary",
        '"latent_shape"',
        '"latent_all_finite"',
        '"mapped_times"',
        '"required_features_in_pca"',
        'np.isfinite(processed.obsm["X_latent"]).all()',
        'processed.obsm["X_latent"].mean(axis=0)',
        "The example uses synthetic data",
    }
    missing = {fragment for fragment in required_fragments if fragment not in text}
    assert not missing
    assert "sha256" not in text.casefold()
    assert "sha-256" not in text.casefold()
    assert "hash-verified" not in text.casefold()

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
    }
    assert not {
        phrase for phrase in forbidden_public_phrases if phrase in text.casefold()
    }
    assert "sys.path" not in text
    assert "repo_root" not in text.casefold()
    assert "import CytoBridge" in text

    section_order = [
        "## 1. Create the input AnnData object",
        "## 2. Run preprocessing",
        "## 3. Check preprocessing metadata and arrays",
        "## 4. Validate reuse of a processed object",
        "## Outputs",
    ]
    section_positions = [text.index(heading) for heading in section_order]
    assert section_positions == sorted(section_positions)

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
    forbidden_present = {
        fragment for fragment in forbidden_fragments if fragment in text
    }
    assert not forbidden_present


def test_readme_links_the_tutorial_and_input_guides() -> None:
    readme = README.read_text(encoding="utf-8")
    assert "## Tutorials and paper figures" in readme
    tutorial = readme.split("## Tutorials and paper figures", 1)[1].split("\n## ", 1)[0]
    assert "tutorial index" in tutorial
    assert "docs/tutorials/" in tutorial
    assert "external AnnData files and checkpoints" in tutorial
    assert "data and checkpoint guide" in readme
