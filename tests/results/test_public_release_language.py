from __future__ import annotations

import gzip
import json
import re
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
    "contributing.md",
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
    "sha256",
    "sha-256",
    "hash",
    "accepted",
    "immutable",
    "reviewed",
    "claim_guardrail",
    "literature_direction_context",
    "descriptive_technical",
)

# Relative paths such as outputs/admouse/data/ are valid reader inputs.
PRIVATE_DATA_ROOT = re.compile(r"(?<![\w./-])/data/")
NUMERICAL_DATA_NOTEBOOKS = {
    "arista_figures.ipynb": "reproduction.arista.supplementary",
    "main_figure_4.ipynb": "reproduction.mosta.main_figure",
    "main_figure_5.ipynb": "reproduction.arista.main_figure",
    "mosta_figures.ipynb": "reproduction.mosta.figures",
}

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
    "lr_prior_ablation_stvcr.ipynb": "lr_prior_stvcr",
    "loto_benchmark.ipynb": "loto_benchmark",
    "main_figure_5.ipynb": "main_figure_5",
    "mosta_figures.ipynb": "mosta_figures",
    "training_histories.ipynb": "training_histories_notebook",
    "compute_cost.ipynb": "full_model_compute_cost_notebook",
    "arista_local_domains.ipynb": "arista_local_domains",
    "zebrafish_attention.ipynb": "zebrafish_attention",
    "zebrafish_si_s31_s38.ipynb": "zebrafish_si_s31_s38",
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


def _notebook_markdown(path: Path) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell.get("source", ()))
        for cell in notebook.get("cells", ())
        if cell.get("cell_type") == "markdown"
    )


def _public_text(path: Path) -> str:
    if path.name.endswith(".gz"):
        with gzip.open(path, mode="rt", encoding="utf-8") as handle:
            return handle.read()
    return path.read_text(encoding="utf-8")


@pytest.mark.parametrize("path", PUBLIC_TEXT_FILES, ids=lambda path: str(path.name))
def test_public_files_do_not_expose_local_or_audit_markers(path: Path) -> None:
    text = _public_text(path).lower()
    markers = LOCAL_OR_AUDIT_MARKERS
    if path == REPOSITORY_ROOT / "CytoBridge/results/data/downloads/manifest.json":
        # This machine-readable field verifies downloaded archives. It is not
        # tutorial prose, and the downloader requires its original field name.
        markers = tuple(marker for marker in markers if marker != "sha256")
    found = [marker for marker in markers if marker in text]
    if PRIVATE_DATA_ROOT.search(text):
        found.append("absolute /data/ path")
    assert not found, f"{path} contains public-release markers: {found}"


@pytest.mark.parametrize(
    "path",
    PUBLIC_NOTEBOOKS,
    ids=lambda path: str(path.name),
)
def test_notebooks_use_reproduction_only_prose(path: Path) -> None:
    text = _notebook_markdown(path).lower()
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
    if notebook_name in NUMERICAL_DATA_NOTEBOOKS:
        assert f"from {NUMERICAL_DATA_NOTEBOOKS[notebook_name]} import draw_" in source
        assert "cb.datasets.download(" in source
        assert 'project / "data/' in source
        assert "source checkout" in lowered
        assert "dataset_workflows/" in source
        assert not any(token in source for token in ("export_mosta", "export_arista", "assemble_main_figure_4"))
        assert not any(marker in lowered for marker in NOTEBOOK_PORTABILITY_MARKERS)
        return
    assert "from cytobridge.results" in lowered
    assert "## run the notebook" in lowered
    assert "## where the inputs come from" not in lowered
    assert "[analysis and input guide](../../reference/figure_sources/" in lowered
    # The plot notebook and the history of its numerical inputs are separate
    # reading routes. Check that the linked guide exists and names its inputs.
    guide_target = source.split("[analysis and input guide](", 1)[1].split(")", 1)[0]
    guide = (path.parent / guide_target).resolve()
    assert guide.is_file()
    guide_text = guide.read_text(encoding="utf-8").lower()
    assert "## calculation programs" in guide_text
    assert "start with:" in guide_text
    assert "writes:" in guide_text
    assert "..." not in source
    assert not any(marker in lowered for marker in NOTEBOOK_PORTABILITY_MARKERS)
    assert f'output_dir = Path("outputs") / "{output_slug}"' in source


def test_s39_notebook_uses_the_reader_facing_api() -> None:
    path = (
        REPOSITORY_ROOT
        / "docs"
        / "tutorials"
        / "paper_figures"
        / "lr_prior_ablation_stvcr.ipynb"
    )
    source = _notebook_source(path)
    assert "load_lr_prior_stvcr_results" in source
    assert "plot_lr_prior_stvcr" in source
    assert "interaction_evidence" not in source


@pytest.mark.parametrize(
    "notebook_name",
    tuple(name for name in COMPLETED_NOTEBOOK_OUTPUTS if name != "compute_cost.ipynb"),
)
def test_figure_notebooks_show_outputs_created_by_their_plotting_cells(
    notebook_name: str,
) -> None:
    path = REPOSITORY_ROOT / "docs" / "tutorials" / "paper_figures" / notebook_name
    source = _notebook_source(path)
    if notebook_name in NUMERICAL_DATA_NOTEBOOKS:
        assert "display(Image(filename=str(path)" in source
        assert "if Path(path).suffix ==" in source
        assert source.index("figures = draw_") < source.rindex("show(")
        return
    assert "display(Image(filename" in source
    preview_position = source.index("display(Image(filename")
    producer_positions = [
        source.find(token)
        for token in ("plot_", "assemble_", "export_")
        if source.find(token) >= 0
    ]
    assert producer_positions
    assert min(producer_positions) < preview_position


@pytest.mark.parametrize(
    "path",
    (REPOSITORY_ROOT / "README.md", *PUBLIC_DOCS),
    ids=lambda path: str(path.name),
)
def test_public_docs_avoid_internal_revision_language(path: Path) -> None:
    text = path.read_text(encoding="utf-8").lower()
    found = [marker for marker in PUBLIC_PROSE_MARKERS if marker in text]
    assert not found, f"{path} contains internal revision language: {found}"


@pytest.mark.parametrize(
    "path",
    PUBLIC_DOCS,
    ids=lambda path: str(path.name),
)
def test_public_docs_avoid_maintenance_jargon(path: Path) -> None:
    text = path.read_text(encoding="utf-8").lower()
    markers = (
        "public smoke",
        "smoke command",
        "dry run",
        "availability:",
        "artifact chain",
        "reproduction route",
        "provenance break",
        "manuscript result bundle",
        "package-native",
        "handoff",
    )
    found = [marker for marker in markers if marker in text]
    assert not found, f"{path} contains maintenance jargon: {found}"


def test_home_page_uses_reader_cards_and_hides_project_records() -> None:
    index = (REPOSITORY_ROOT / "docs" / "index.md").read_text(encoding="utf-8")
    config = (REPOSITORY_ROOT / "docs" / "conf.py").read_text(encoding="utf-8")

    assert index.count("{grid-item-card}") == 6
    assert "cytobridge-home-cards" in index
    assert "limitations" not in index.casefold()
    assert '"limitations.md"' in config
    assert '"release_notes.md"' in config
