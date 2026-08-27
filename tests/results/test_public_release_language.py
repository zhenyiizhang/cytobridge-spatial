from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

PUBLIC_GUIDE_NAMES = (
    "installation.md",
    "quickstart.md",
    "data_checkpoints.md",
    "downstream.md",
    "nonspatial_workflows.md",
    "benchmarks.md",
    "training_compute.md",
    "paper_reproduction.md",
    "limitations.md",
    "contributing.md",
    "release_notes.md",
)

PUBLIC_DOCS = (
    REPOSITORY_ROOT / "docs" / "index.md",
    *(REPOSITORY_ROOT / "docs" / name for name in PUBLIC_GUIDE_NAMES),
    *sorted((REPOSITORY_ROOT / "docs" / "tutorials").rglob("*.md")),
    *sorted((REPOSITORY_ROOT / "docs" / "api").rglob("*.rst")),
)

PUBLIC_TEXT_FILES = (
    REPOSITORY_ROOT / "README.md",
    *PUBLIC_DOCS,
    *sorted((REPOSITORY_ROOT / "CytoBridge" / "results").rglob("*.py")),
    *sorted((REPOSITORY_ROOT / "CytoBridge" / "results" / "data").rglob("*.json")),
    *sorted((REPOSITORY_ROOT / "CytoBridge" / "results" / "data").rglob("*.csv")),
    *sorted((REPOSITORY_ROOT / "CytoBridge" / "results" / "data").rglob("*.csv.gz")),
    REPOSITORY_ROOT / "docs" / "data" / "paper_reproduction_registry.csv",
    REPOSITORY_ROOT / "CytoBridge" / "pp" / "chicken_heart_input.py",
    REPOSITORY_ROOT / "scripts" / "execute_paper_notebooks.py",
    *sorted((REPOSITORY_ROOT / "scripts" / "results").rglob("*.py")),
)

LOCAL_OR_AUDIT_MARKERS = (
    "/users/",
    "/home/",
    "/lustre/",
    "/opt/",
    "/data/",
    "sha256",
    "sha-256",
    "hash",
    "reviewer",
    "accepted",
    "immutable",
    "reviewed",
    "claim_guardrail",
    "literature_direction_context",
    "descriptive_technical",
)

PUBLIC_PROSE_MARKERS = (
    "learning goals",
    "scientific question",
    "scientific contract",
    "formal scope",
    "notes and interpretation",
    "exercise",
    "reviewer",
    "accepted",
    "immutable",
    "win count",
    "outperform",
)

NOTEBOOK_PORTABILITY_MARKERS = (
    "sys.path",
    "repo_root",
    "path.cwd(",
)

COMPLETED_NOTEBOOK_OUTPUTS = {
    "agist_figures.ipynb": "agist_figures",
    "arista_figures.ipynb": "arista_supplementary_figures",
    "main_figure_2.ipynb": "main_figure_2",
    "main_figure_4.ipynb": "main_figure_4",
    "classifier_smoothing.ipynb": "classifier_smoothing",
    "lr_complex_aggregation.ipynb": "lr_complex_aggregation",
    "lr_prior_ablation_stvcr.ipynb": "interaction_evidence",
    "loto_benchmark.ipynb": "loto_benchmark",
    "main_figure_5.ipynb": "main_figure_5",
    "mosta_figures.ipynb": "mosta_figures",
    "training_histories.ipynb": "training_histories_notebook",
    "compute_cost.ipynb": "full_model_compute_cost_notebook",
    "arista_local_domains.ipynb": "arista_local_domains",
    "zebrafish_attention.ipynb": "zebrafish_attention",
    "zebrafish_si_s27_s34.ipynb": "zebrafish_si_s27_s34",
    "nonspatial_figures.ipynb": "nonspatial_figures",
}

PUBLIC_NOTEBOOKS = (
    *sorted((REPOSITORY_ROOT / "notebooks").glob("*.ipynb")),
    *sorted((REPOSITORY_ROOT / "docs" / "tutorials").rglob("*.ipynb")),
)


def _notebook_source(path: Path) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell.get("source", ())) for cell in notebook.get("cells", ())
    )


def _public_text(path: Path) -> str:
    if path.name.endswith(".gz"):
        with gzip.open(path, mode="rt", encoding="utf-8") as handle:
            return handle.read()
    return path.read_text(encoding="utf-8")


@pytest.mark.parametrize("path", PUBLIC_TEXT_FILES, ids=lambda path: str(path.name))
def test_public_files_do_not_expose_local_or_audit_markers(path: Path) -> None:
    text = _public_text(path).lower()
    found = [marker for marker in LOCAL_OR_AUDIT_MARKERS if marker in text]
    assert not found, f"{path} contains public-release markers: {found}"


@pytest.mark.parametrize(
    "path",
    PUBLIC_NOTEBOOKS,
    ids=lambda path: str(path.name),
)
def test_notebooks_use_reproduction_only_prose(path: Path) -> None:
    text = _notebook_source(path).lower()
    found = [
        marker
        for marker in (
            *LOCAL_OR_AUDIT_MARKERS,
            *PUBLIC_PROSE_MARKERS,
            *NOTEBOOK_PORTABILITY_MARKERS,
        )
        if marker in text
    ]
    assert not found, f"{path} contains notebook prose markers: {found}"


@pytest.mark.parametrize(
    ("notebook_name", "output_slug"),
    COMPLETED_NOTEBOOK_OUTPUTS.items(),
)
def test_completed_notebooks_use_installed_package(
    notebook_name: str,
    output_slug: str,
) -> None:
    path = REPOSITORY_ROOT / "docs" / "tutorials" / "paper_figures" / notebook_name
    source = _notebook_source(path)
    lowered = source.lower()
    assert "from cytobridge.results" in lowered
    assert not any(marker in lowered for marker in NOTEBOOK_PORTABILITY_MARKERS)
    assert f'output_dir = Path("outputs") / "{output_slug}"' in source


@pytest.mark.parametrize(
    "path",
    (REPOSITORY_ROOT / "README.md", *PUBLIC_DOCS),
    ids=lambda path: str(path.name),
)
def test_public_docs_avoid_internal_revision_language(path: Path) -> None:
    text = path.read_text(encoding="utf-8").lower()
    found = [marker for marker in PUBLIC_PROSE_MARKERS if marker in text]
    assert not found, f"{path} contains internal revision language: {found}"
