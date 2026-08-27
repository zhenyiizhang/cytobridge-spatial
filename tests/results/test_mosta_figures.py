from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pymupdf as fitz
import numpy as np
from pypdf import PdfReader
import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from CytoBridge.results.mosta_figures import (  # noqa: E402
    RELEASE_DIRECTORY,
    RELEASE_ENVIRONMENT_VARIABLE,
    assemble_main_figure_4,
    export_mosta_supplementary_figures,
    load_mosta_figure_release,
    rebuild_main_figure_4,
    resolve_mosta_release_dir,
    write_mosta_figure_index,
)


CURRENT_SUPPLEMENTARY = tuple(f"S{number}" for number in range(9, 17))
RELEASE_SUPPLEMENTARY = tuple(f"S{number}" for number in range(4, 12))
PAGE_POINTS = (595.2760009765625, 841.8900146484375)
PANEL_SPECS = (
    ("a", "Fig4a.pdf", 595.2760009765625, 192.0),
    ("b", "Fig4b.pdf", 326.6, 258.5),
    ("c", "Fig4c.pdf", 278.9136047363281, 238.0),
    ("d", "Fig4d.pdf", 290.0, 378.0),
    ("e", "Fig4e.pdf", 309.2760009765625, 379.8900146484375),
)
SOURCE_FILES = (
    "reproduction/main_fig4_panels/fig4a/source/"
    "run_mosta_fig4a_global_t0_particle_sensitivity.py",
    "reproduction/main_fig4_panels/fig4b/source/"
    "server_compute_fig4b_package_compute_display50k.py",
    "reproduction/main_fig4_panels/fig4c/source/legacy_mosta_cartilage_lineage.py",
    "reproduction/main_fig4_panels/fig4d/source/"
    "render_fig4d_original_ai_equivalent.py",
    "reproduction/main_fig4_panels/fig4e/source/"
    "render_fig4e_exact_notebook_sources.py",
    "reproduction/main_figure4_complete/source/assemble_complete_figure4.py",
    "reproduction/shared_global_t0_50k/source/"
    "server_compute_mosta_si_shared.py",
    "reproduction/si/S4/source/audit_s4_compute.py",
    "reproduction/si/S4/source/render_s4_exact_old_style.py",
    "reproduction/si/S5/source/audit_s5_latest_package_compute.py",
    "reproduction/si/S5/source/render_s5_corrected_exact_submitted_style.py",
    "reproduction/si/S6/source/audit_s6_latest_package_composition.py",
    "reproduction/si/S6/source/render_s6_corrected_exact_submitted_style.py",
    "reproduction/si/S7/source/audit_s7_latest_package_fixed_particle_lineage.py",
    "reproduction/si/S7/source/server_render_s7_exact_old_plotly_style_arial.py",
    "reproduction/si/S8/source/audit_s8_latest_package_gene_programs.py",
    "reproduction/si/S8/source/render_s8_corrected_exact_submitted_style.py",
    "reproduction/si/S9_S10/source/run_mosta_s9_s10_clusterprofiler.R",
    "reproduction/si/S9_S10/source/"
    "audit_mosta_s9_s10_clusterprofiler_and_dp3.py",
    "reproduction/si/S9_S10/source/"
    "render_mosta_s9_s10_clusterprofiler_exact_submitted_style.py",
    "reproduction/si/S11/source/select_s11_msum_stable_representative31.py",
    "reproduction/si/S11/source/render_s11_msum_exact_submitted_style.py",
)
MODEL_FILES = (
    "model/checkpoints/config.yaml",
    "model/checkpoints/Finetune/best_model.pth",
    "model/checkpoints/Score_Refine/score_model.pth",
    "model/classifier_cache/classifier_resmlp_6d2d7acf7d0ed92d.pt",
)


def _write_pdf(path: Path, width: float, height: float, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    document = fitz.open()
    page = document.new_page(width=width, height=height)
    page.draw_rect(
        fitz.Rect(1, 1, max(2, width - 1), max(2, height - 1)),
        color=(0.1, 0.3, 0.5),
        width=0.5,
    )
    page.insert_text((8, min(20, height - 2)), label, fontsize=8)
    document.save(path)
    document.close()


def _minimal_release(tmp_path: Path) -> Path:
    root = tmp_path / RELEASE_DIRECTORY
    root.mkdir()
    manifest = {
        "schema_version": 1,
        "status": "READER_RELEASE_COMPLETE",
        "software": {
            "package_commit_used_for_all_numerical_results": "package-version",
        },
    }
    (root / "MANIFEST.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    for label, filename, width, height in PANEL_SPECS:
        _write_pdf(root / "figures/main/panels" / filename, width, height, label)
    _write_pdf(
        root / "figures/main/Figure_4_complete.pdf",
        PAGE_POINTS[0],
        PAGE_POINTS[1],
        "complete",
    )
    main_svg = root / "figures/main/Figure_4_complete.svg"
    main_svg.parent.mkdir(parents=True, exist_ok=True)
    main_svg.write_text("<svg xmlns='http://www.w3.org/2000/svg'/>", encoding="utf-8")
    for release_id in RELEASE_SUPPLEMENTARY:
        _write_pdf(
            root / f"figures/si/Figure_{release_id}.pdf",
            300,
            420,
            release_id,
        )
        svg = root / f"figures/si/Figure_{release_id}.svg"
        svg.write_text(
            f"<svg xmlns='http://www.w3.org/2000/svg'><text>{release_id}</text></svg>",
            encoding="utf-8",
        )
    for relative_path in (*SOURCE_FILES, *MODEL_FILES):
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("reader fixture\n", encoding="utf-8")
    return root


def _available_full_release() -> Path | None:
    repository_release = REPOSITORY_ROOT / "release_artifacts" / RELEASE_DIRECTORY
    if repository_release.is_dir():
        return repository_release
    environment_value = os.environ.get(RELEASE_ENVIRONMENT_VARIABLE)
    if environment_value and Path(environment_value).is_dir():
        return Path(environment_value)
    return None


def _render_pdf(path: Path, dpi: int) -> np.ndarray:
    with fitz.open(path) as document:
        pixmap = document[0].get_pixmap(
            matrix=fitz.Matrix(dpi / 72.0, dpi / 72.0),
            alpha=False,
        )
    return np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
        pixmap.height,
        pixmap.width,
        pixmap.n,
    )[:, :, :3]


def test_mosta_release_index_uses_current_paper_numbers(tmp_path: Path) -> None:
    release = load_mosta_figure_release(_minimal_release(tmp_path))
    assert release.figure_index["paper_location"].tolist() == [
        "Main Figure 4",
        *(f"Supplementary Figure {figure}" for figure in CURRENT_SUPPLEMENTARY),
    ]
    assert release.figure_index["release_location"].tolist() == [
        "Main Figure 4",
        *(f"Supplementary Figure {figure}" for figure in RELEASE_SUPPLEMENTARY),
    ]
    for column in (
        "vector_pdf",
        "vector_svg",
        "calculation_scripts",
        "renderer",
        "model_assets",
    ):
        for value in release.figure_index[column]:
            for relative_path in str(value).split(";"):
                assert not Path(relative_path).is_absolute()
                assert (release.root / relative_path).is_file()


def test_default_release_resolution_uses_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _minimal_release(tmp_path)
    monkeypatch.setenv(RELEASE_ENVIRONMENT_VARIABLE, str(root))
    assert resolve_mosta_release_dir() == root.resolve()
    assert load_mosta_figure_release().root == root.resolve()


def test_main_figure_4_assembly_uses_panels_and_connectors(tmp_path: Path) -> None:
    release = load_mosta_figure_release(_minimal_release(tmp_path))
    output = tmp_path / "main-output"
    pdf, png = assemble_main_figure_4(release, output, dpi=72)
    assert pdf.is_file()
    assert png.is_file()
    with fitz.open(pdf) as document:
        assert document.page_count == 1
        assert document[0].rect.width == pytest.approx(PAGE_POINTS[0], abs=0.01)
        assert document[0].rect.height == pytest.approx(PAGE_POINTS[1], abs=0.01)
        text = document[0].get_text("text")
    assert all(label in text for label in "abcde")
    content = PdfReader(pdf).pages[0].get_contents().get_data()
    assert b"105.4854 315.5952 m" in content
    assert b"285.8274 8.9109 l" in content
    assert pdf.read_bytes() != (
        release.root / "figures/main/Figure_4_complete.pdf"
    ).read_bytes()


def test_supplementary_export_preserves_vector_pages(tmp_path: Path) -> None:
    release = load_mosta_figure_release(_minimal_release(tmp_path))
    output = tmp_path / "si-output"
    exported = export_mosta_supplementary_figures(
        release,
        output,
        preview_dpi=72,
    )
    assert tuple(exported) == CURRENT_SUPPLEMENTARY
    index = release.figure_index.set_index("paper_location")
    for figure_id, paths in exported.items():
        row = index.loc[f"Supplementary Figure {figure_id}"]
        assert paths["pdf"].read_bytes() == (
            release.root / row["vector_pdf"]
        ).read_bytes()
        assert paths["svg"].read_bytes() == (
            release.root / row["vector_svg"]
        ).read_bytes()
        assert paths["png"].is_file()
    written_index = output / "mosta_figure_index.csv"
    assert written_index.is_file()


def test_mosta_figure_index_writer(tmp_path: Path) -> None:
    release = load_mosta_figure_release(_minimal_release(tmp_path))
    path = write_mosta_figure_index(release, tmp_path / "index")
    assert path.name == "mosta_figure_index.csv"
    assert "Supplementary Figure S16" in path.read_text(encoding="utf-8")


def test_mosta_cli_uses_current_numbers(tmp_path: Path) -> None:
    release_dir = _minimal_release(tmp_path)
    output = tmp_path / "cli-output"
    completed = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "scripts/results/export_mosta_figures.py"),
            "--release-dir",
            str(release_dir),
            "--output-dir",
            str(output),
            "--dpi",
            "72",
            "--preview-dpi",
            "72",
        ],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(REPOSITORY_ROOT)},
    )
    summary = json.loads(completed.stdout)
    assert summary["analysis"] == "mosta_manuscript_figures"
    assert summary["figure_action"] == "external-assembly + reference-export"
    assert summary["current_supplementary_numbers"] == list(CURRENT_SUPPLEMENTARY)
    assert set(summary["supplementary_figures"]) == set(CURRENT_SUPPLEMENTARY)
    assert summary["main_figure"]["pdf"].endswith("main_figure_4.pdf")
    assert str(tmp_path) not in completed.stdout
    assert json.loads((output / "run_summary.json").read_text()) == summary


def test_incomplete_mosta_release_is_rejected(tmp_path: Path) -> None:
    root = _minimal_release(tmp_path)
    manifest_path = root / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = "INCOMPLETE"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="not complete"):
        load_mosta_figure_release(root)


def test_main_figure_4_matches_full_release_render(tmp_path: Path) -> None:
    root = _available_full_release()
    if root is None:
        pytest.skip("repository MOSTA release is not present yet")
    release = load_mosta_figure_release(root)
    rebuilt_pdf, _ = assemble_main_figure_4(
        release,
        tmp_path / "full-release-rebuild",
        dpi=300,
    )
    source_pdf = release.root / "figures/main/Figure_4_complete.pdf"
    source_render = _render_pdf(source_pdf, 300)
    rebuilt_render = _render_pdf(rebuilt_pdf, 300)
    assert source_render.shape == rebuilt_render.shape == (3508, 2481, 3)
    assert np.array_equal(source_render, rebuilt_render)


def test_mosta_module_keeps_vector_dependencies_lazy() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import CytoBridge.results.mosta_figures; "
                "assert 'fitz' not in sys.modules; "
                "assert 'pypdf' not in sys.modules"
            ),
        ],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(REPOSITORY_ROOT)},
    )
    assert completed.stdout == ""


def test_main_figure_4_legacy_name_remains_available() -> None:
    assert callable(rebuild_main_figure_4)
    assert "Compatibility alias" in (rebuild_main_figure_4.__doc__ or "")


@pytest.mark.parametrize(
    ("notebook_name", "output_slug", "title"),
    (
        ("main_figure_4.ipynb", "main_figure_4", "Main Figure 4: MOSTA"),
        ("mosta_figures.ipynb", "mosta_figures", "Supplementary Figures S9–S16"),
    ),
)
def test_mosta_notebooks_use_public_reader_api(
    notebook_name: str,
    output_slug: str,
    title: str,
) -> None:
    path = REPOSITORY_ROOT / "docs/tutorials/paper_figures" / notebook_name
    notebook = json.loads(path.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", ())) for cell in notebook["cells"]
    )
    assert title in source
    assert "from CytoBridge.results import" in source
    assert "load_mosta_figure_release()" in source
    assert f'output_dir = Path("outputs") / "{output_slug}"' in source
    if notebook_name == "main_figure_4.ipynb":
        assert "assemble_main_figure_4" in source
    else:
        assert "export_mosta_supplementary_figures" in source
    assert any(
        cell.get("outputs")
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
