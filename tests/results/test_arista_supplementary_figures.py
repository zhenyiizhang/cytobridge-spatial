from __future__ import annotations

import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys

import fitz
import matplotlib as mpl
from matplotlib import font_manager
import pandas as pd
from PIL import Image, ImageChops
import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE_ROOT))

from CytoBridge.results.arista_supplementary_figures import (  # noqa: E402
    FIGURE_ORDER,
    calculate_arista_supplementary_pages,
    load_arista_supplementary_figures,
    plot_arista_supplementary_figures,
    select_arista_supplementary_pages,
    write_arista_supplementary_tables,
)


def _fixture_copy(tmp_path: Path) -> Path:
    source = load_arista_supplementary_figures().source_dir
    target = tmp_path / "arista_supplementary_figures"
    shutil.copytree(source, target)
    return target


def test_packaged_arista_supplementary_contract() -> None:
    data = load_arista_supplementary_figures()
    pages = calculate_arista_supplementary_pages(data)
    assert tuple(page.figure for page in pages) == FIGURE_ORDER
    assert [page.raster_crc32 for page in pages] == [
        "573fd645",
        "b3d4f41c",
        "d48a9fbf",
        "1893c30e",
        "bd12e84e",
        "6e978c28",
    ]
    assert [(page.width_pixels, page.height_pixels) for page in pages] == [
        (2106, 2093),
        (3751, 3606),
        (2333, 2400),
        (2400, 1554),
        (1582, 881),
        (1790, 3000),
    ]
    assert set(data.tables) == set(data.manifest["tables"])
    assert len(data.tables) == 13


def test_formal_release_tables_retain_display_values() -> None:
    tables = load_arista_supplementary_figures().tables
    assert len(tables["interpolation_panel_inventory"]) == 14
    assert tables["growth_panel_scale"]["n_display"].eq(2500).all()
    counts = tables["lineage_composition_counts"].drop(columns="time")
    assert counts.sum(axis=1).eq(7668).all()
    prototypes = tables["gene_program_prototypes"].groupby("pattern", sort=True)
    assert prototypes.size().to_dict() == {1: 9, 2: 9}
    assert prototypes["n_profiles"].first().to_dict() == {1: 484, 2: 1516}
    clusters = tables["ligand_receptor_cluster_prototypes"].groupby(
        "cluster",
        sort=True,
    )
    assert clusters.size().to_dict() == {1: 9, 2: 9}
    assert clusters["n_pairs"].first().to_dict() == {1: 1, 2: 530}
    roster = tables["ligand_receptor_display_roster"]
    assert len(roster) == 68
    assert roster["estimable"].value_counts().to_dict() == {True: 51, False: 17}
    assert len(tables["ligand_receptor_pair_timecourse"]) == 612


def test_full_recompute_registry_uses_relative_paths() -> None:
    registry = load_arista_supplementary_figures().full_recompute_inputs
    assert set(registry["input_id"]) >= {
        "raw_h5ad",
        "aligned_h5ad",
        "model_dir",
        "growth_by_cell",
        "fixed_particle_composition",
        "gene_mean_expression",
        "lr_pair_timecourse",
    }
    for value in registry["relative_path"]:
        path = PurePosixPath(value)
        assert not path.is_absolute()
        assert ".." not in path.parts
        assert not str(value).startswith(("/Users/", "/home/", "/data/"))


def test_arista_supplementary_tables_are_written(tmp_path: Path) -> None:
    data = load_arista_supplementary_figures()
    pages = calculate_arista_supplementary_pages(data)
    written = write_arista_supplementary_tables(data, pages, tmp_path)
    assert len(written) == 16
    summary = pd.read_csv(written["page_summary"])
    assert summary["figure"].tolist() == list(FIGURE_ORDER)
    for table_id, table in data.tables.items():
        copied = pd.read_csv(written[table_id], keep_default_na=False)
        pd.testing.assert_frame_equal(copied, table)


def test_arista_supplementary_pages_render_with_release_geometry(
    tmp_path: Path,
) -> None:
    mpl.use("Agg", force=True)
    data = load_arista_supplementary_figures()
    pages = calculate_arista_supplementary_pages(data)
    written = plot_arista_supplementary_figures(data, tmp_path, pages)
    assert tuple(written) == FIGURE_ORDER
    for page in pages:
        pdf_path, png_path = written[page.figure]
        assert png_path.read_bytes() == data.raster_paths[page.figure].read_bytes()
        with fitz.open(pdf_path) as document:
            assert document.page_count == 1
            assert document[0].rect.width == pytest.approx(page.width_points, abs=0.01)
            assert document[0].rect.height == pytest.approx(page.height_points, abs=0.01)
            pixmap = document[0].get_pixmap(
                matrix=fitz.Matrix(
                    page.reference_dpi / 72.0,
                    page.reference_dpi / 72.0,
                ),
                alpha=False,
            )
            rendered = Image.frombytes(
                "RGB",
                (pixmap.width, pixmap.height),
                pixmap.samples,
            )
        with Image.open(png_path) as source:
            source_rgb = source.convert("RGB")
        if page.figure == "S19":
            assert rendered.size == (source_rgb.width + 1, source_rgb.height)
        else:
            assert rendered.size == source_rgb.size
            assert ImageChops.difference(source_rgb, rendered).getbbox() is None


def test_arista_supplementary_selection_is_current_and_unique() -> None:
    pages = calculate_arista_supplementary_pages(
        load_arista_supplementary_figures()
    )
    assert [page.figure for page in select_arista_supplementary_pages(
        pages,
        ("S22", "S17"),
    )] == ["S17", "S22"]
    with pytest.raises(ValueError, match="duplicates"):
        select_arista_supplementary_pages(pages, ("S17", "S17"))
    with pytest.raises(ValueError, match="Unknown"):
        select_arista_supplementary_pages(pages, ("S16",))


def test_arista_supplementary_module_import_is_matplotlib_lazy() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import CytoBridge.results.arista_supplementary_figures; "
                "assert 'matplotlib' not in sys.modules"
            ),
        ],
        cwd=PACKAGE_ROOT,
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(PACKAGE_ROOT)},
    )
    assert completed.stdout == ""


def test_arista_supplementary_cli_exports_selected_pages(tmp_path: Path) -> None:
    output = tmp_path / "cli"
    completed = subprocess.run(
        [
            sys.executable,
            str(PACKAGE_ROOT / "scripts/results/plot_arista_figures.py"),
            "--output-dir",
            str(output),
            "--figures",
            "S17",
            "S22",
        ],
        cwd=PACKAGE_ROOT,
        check=True,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "MPLBACKEND": "Agg",
            "MPLCONFIGDIR": str(tmp_path / "mpl"),
            "PYTHONPATH": str(PACKAGE_ROOT),
        },
    )
    summary = json.loads(completed.stdout)
    assert summary["analysis"] == "arista_supplementary_figures"
    assert summary["source"] == "packaged"
    assert summary["figures"] == ["S17", "S22"]
    assert set(summary["files"]) == {"S17", "S22"}
    assert str(tmp_path) not in completed.stdout


def test_changed_arista_raster_is_rejected(tmp_path: Path) -> None:
    results_dir = _fixture_copy(tmp_path)
    path = results_dir / "FigureS17_ARISTA_interpolation.png"
    payload = bytearray(path.read_bytes())
    payload[-20] ^= 1
    path.write_bytes(payload)
    with pytest.raises(ValueError, match="compact page contract"):
        load_arista_supplementary_figures(results_dir)


@pytest.mark.parametrize(
    "absolute_path",
    ["/data/arista/Regeneration.h5ad", "C:\\arista\\Regeneration.h5ad"],
)
def test_absolute_arista_registry_path_is_rejected(
    tmp_path: Path,
    absolute_path: str,
) -> None:
    results_dir = _fixture_copy(tmp_path)
    path = results_dir / "full_recompute_inputs.csv"
    table = pd.read_csv(path, keep_default_na=False)
    table.loc[0, "relative_path"] = absolute_path
    table.to_csv(path, index=False)
    with pytest.raises(ValueError, match="non-relative input path"):
        load_arista_supplementary_figures(results_dir)


def test_main_figure_5_vector_script_is_portable(tmp_path: Path) -> None:
    base_pdf = tmp_path / "base.pdf"
    base_manifest = tmp_path / "base_manifest.json"
    output = tmp_path / "vector"
    document = fitz.open()
    page = document.new_page(width=595.276, height=841.89)
    page.draw_rect(fitz.Rect(20, 20, 100, 100), color=(0.2, 0.3, 0.4))
    page.insert_text((250, 450), "Spatial migration velocity", fontsize=12)
    page.insert_text((420, 450), "Spatial velocity cosine simlarity", fontsize=10)
    page.insert_text((410, 466), "     (interaction VS migration)", fontsize=11)
    document.save(base_pdf)
    document.close()
    base_manifest.write_text(
        json.dumps(
            {
                "schema": "cytobridge.arista.figure5.fullpage-assembly.v2",
                "figure": "ARISTA Figure 5, panels a-e",
            }
        ),
        encoding="utf-8",
    )
    font_path = font_manager.findfont(
        font_manager.FontProperties(family="STIXGeneral", weight="bold"),
        fallback_to_default=False,
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(PACKAGE_ROOT / "scripts/results/build_main_figure_5_vector.py"),
            "--base-pdf",
            str(base_pdf),
            "--base-manifest",
            str(base_manifest),
            "--font-file",
            font_path,
            "--output-dir",
            str(output),
        ],
        cwd=PACKAGE_ROOT,
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(PACKAGE_ROOT)},
    )
    summary = json.loads(completed.stdout)
    assert summary["scientific_label_release"] == "v5"
    assert summary["outside_label_equivalence"]["passed"] is True
    assert summary["outside_label_equivalence"]["outside_label_changed_pixels"] == 0
    assert str(tmp_path) not in completed.stdout
    result_pdf = output / summary["files"]["pdf"]
    result_png = output / summary["files"]["png"]
    with fitz.open(result_pdf) as result:
        text = result[0].get_text("text")
        assert "Spatial velocity cosine similarity" in text
        assert "interaction vs full spatial velocity" in text
        assert "migration" not in text
    with Image.open(result_png) as image:
        assert image.size == (2481, 3508)


def test_arista_supplementary_notebook_uses_current_numbering() -> None:
    path = (
        PACKAGE_ROOT
        / "docs/tutorials/paper_figures/arista_figures.ipynb"
    )
    notebook = json.loads(path.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", ())) for cell in notebook["cells"]
    )
    assert "Supplementary Figures S17–S22" in source
    assert "from CytoBridge.results import" in source
    assert 'output_dir = Path("outputs") / "arista_supplementary_figures"' in source
    assert "plot_arista_supplementary_figures" in source
    assert "full_recompute_inputs" in source
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None
            assert cell["outputs"] == []
