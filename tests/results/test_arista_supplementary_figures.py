from __future__ import annotations

import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys

import pymupdf as fitz
import matplotlib as mpl
from matplotlib import font_manager
import pandas as pd
from PIL import Image, ImageChops
import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE_ROOT))

from CytoBridge.results.arista_supplementary_figures import (  # noqa: E402
    ARISTA_RELEASE_DIRECTORY,
    ARISTA_RELEASE_ENVIRONMENT_VARIABLE,
    FIGURE_ORDER,
    calculate_arista_ligand_receptor_panels,
    calculate_arista_supplementary_pages,
    export_arista_reference_pages,
    load_arista_figure_release,
    load_arista_supplementary_figures,
    plot_arista_ligand_receptor_figures,
    plot_arista_supplementary_figures,
    resolve_arista_release_dir,
    select_arista_supplementary_pages,
    write_arista_source_index,
    write_arista_ligand_receptor_tables,
    write_arista_supplementary_tables,
)


FORMAL_RELEASE_ROOT = PACKAGE_ROOT / "release_artifacts" / ARISTA_RELEASE_DIRECTORY


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
        "8d4787c1",
        "e36de1fc",
    ]
    assert [(page.width_pixels, page.height_pixels) for page in pages] == [
        (2106, 2093),
        (3751, 3606),
        (2333, 2400),
        (2400, 1554),
        (2372, 1322),
        (3222, 3857),
    ]
    assert set(data.tables) == set(data.manifest["tables"])
    assert len(data.tables) == 18


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
    assert clusters["n_pairs"].first().to_dict() == {1: 217, 2: 314}
    assert len(tables["ligand_receptor_cluster_assignments"]) == 531
    assert len(tables["ligand_receptor_normalized_profiles"]) == 531
    assert len(tables["ligand_receptor_all_pair_timecourse"]) == 531 * 9
    k_selection = tables["ligand_receptor_k_selection"].sort_values(
        ["silhouette", "k"], ascending=[False, True]
    )
    assert int(k_selection.iloc[0]["k"]) == 2
    roster = tables["ligand_receptor_display_roster"]
    assert len(roster) == 50
    assert roster.groupby("cluster").size().to_dict() == {1: 25, 2: 25}
    assert len(tables["ligand_receptor_pair_timecourse"]) == 450


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


def test_formal_arista_release_uses_current_numbers_and_checkout_sources() -> None:
    release = load_arista_figure_release(FORMAL_RELEASE_ROOT)
    index = release.source_index
    assert index["paper_location"].tolist() == [
        f"Supplementary Figure S{number}" for number in range(19, 25)
    ]
    assert index["release_location"].tolist() == [
        f"Supplementary Figure S{number}" for number in range(12, 18)
    ]
    assert index["content"].tolist() == [
        "Spatial interpolation",
        "Growth",
        "Lineage and composition",
        "Gene programs and GO enrichment",
        "Ligand-receptor clusters",
        "Ligand-receptor small multiples",
    ]
    for column in (
        "formal_pdf",
        "formal_svg",
        "formal_png",
        "release_build_snapshot",
        "release_manifest",
        "downstream_inputs",
        "checkpoint_inputs",
    ):
        for value in index[column]:
            for relative_path in filter(None, str(value).split(";")):
                path = PurePosixPath(relative_path)
                assert not path.is_absolute()
                assert ".." not in path.parts
                assert (release.root / relative_path).is_file()
    for column in ("canonical_scripts", "calculation_entrypoints"):
        for value in index[column]:
            for relative_path in str(value).split(";"):
                path = PurePosixPath(relative_path)
                assert not path.is_absolute()
                assert ".." not in path.parts
                assert (PACKAGE_ROOT / relative_path).is_file()

    compact = load_arista_supplementary_figures()
    by_figure = index.set_index("paper_location")
    for figure in FIGURE_ORDER:
        formal_png = (
            release.root / by_figure.loc[f"Supplementary Figure {figure}", "formal_png"]
        )
        assert compact.raster_paths[figure].read_bytes() == formal_png.read_bytes()


def test_formal_arista_release_records_vector_boundaries() -> None:
    index = load_arista_figure_release(FORMAL_RELEASE_ROOT).source_index.set_index(
        "paper_location"
    )
    assert index.loc["Supplementary Figure S22", "vector_scope"] == (
        "raster composite PDF; four retained panel SVGs, two with an embedded "
        "raster layer"
    )
    assert len(index.loc["Supplementary Figure S22", "formal_svg"].split(";")) == 4
    assert index.loc["Supplementary Figure S20", "vector_scope"] == (
        "full-page PDF and SVG with nine embedded raster layers"
    )
    assert index.loc["Supplementary Figure S23", "release_build_snapshot"].endswith(
        "build_s16_kmeans_legacy_style.py"
    )
    assert "recluster_arista_lr_patterns.py" in index.loc[
        "Supplementary Figure S23", "canonical_scripts"
    ]
    assert index.loc["Supplementary Figure S24", "release_build_snapshot"].endswith(
        "build_s17_balanced_representative_legacy_style.py"
    )
    assert index.loc["Supplementary Figure S20", "build_scope"] == (
        "release snapshot is the exact page builder; repository script adds "
        "optional display settings"
    )
    assert index.loc["Supplementary Figure S21", "input_scope"] == (
        "release retains derived fixed-particle tables; the upstream "
        "fixed-particle file is external"
    )


def test_formal_arista_release_resolution_and_index_writer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        ARISTA_RELEASE_ENVIRONMENT_VARIABLE,
        str(FORMAL_RELEASE_ROOT),
    )
    assert resolve_arista_release_dir() == FORMAL_RELEASE_ROOT.resolve()
    release = load_arista_figure_release()
    output = write_arista_source_index(release, tmp_path)
    written = pd.read_csv(output, keep_default_na=False)
    assert (
        written["paper_location"].tolist()
        == release.source_index["paper_location"].tolist()
    )


def test_arista_supplementary_tables_are_written(tmp_path: Path) -> None:
    data = load_arista_supplementary_figures()
    pages = calculate_arista_supplementary_pages(data)
    written = write_arista_supplementary_tables(data, pages, tmp_path)
    assert len(written) == 21
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
            assert document[0].rect.height == pytest.approx(
                page.height_points, abs=0.01
            )
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
        if page.figure == "S21":
            assert rendered.size == (source_rgb.width + 1, source_rgb.height)
        else:
            assert rendered.size == source_rgb.size
            assert ImageChops.difference(source_rgb, rendered).getbbox() is None


def test_corrected_arista_lr_figures_are_drawn_from_tables(tmp_path: Path) -> None:
    data = load_arista_supplementary_figures()
    panels = calculate_arista_ligand_receptor_panels(data)
    written = plot_arista_ligand_receptor_figures(data, tmp_path, panels)
    assert set(written) == {"S23", "S24"}
    formal_geometry = {
        "S23": (569.192, 317.15225),
        "S24": (1288.692187, 1542.593706),
    }
    formal_png_geometry = {
        "S23": (2371, 1321),
        "S24": (3221, 3856),
    }
    for figure, (pdf_path, png_path) in written.items():
        assert pdf_path.is_file()
        assert png_path.is_file()
        with fitz.open(pdf_path) as document:
            assert document.page_count == 1
            page = document[0]
            assert page.rect.width == pytest.approx(
                formal_geometry[figure][0], abs=0.01
            )
            assert page.rect.height == pytest.approx(
                formal_geometry[figure][1], abs=0.01
            )
            embedded_fonts = {font[3] for font in page.get_fonts(full=True)}
            assert any(
                family in font_name
                for font_name in embedded_fonts
                for family in ("Arial", "LiberationSans", "DejaVuSans")
            )
        with Image.open(png_path) as image:
            assert image.size == formal_png_geometry[figure]
    assert written["S23"][0].name == "FigureS23_ARISTA_redrawn.pdf"
    assert written["S24"][0].name == "FigureS24_ARISTA_redrawn.pdf"
    assert written["S23"][1].read_bytes() != data.raster_paths["S23"].read_bytes()
    assert written["S24"][1].read_bytes() != data.raster_paths["S24"].read_bytes()


def test_arista_lr_calculation_matches_formal_figure_tables() -> None:
    data = load_arista_supplementary_figures()
    panels = calculate_arista_ligand_receptor_panels(data)

    assert panels.assignments.groupby("cluster").size().to_dict() == {1: 217, 2: 314}
    assert panels.display_roster.groupby("cluster").size().to_dict() == {1: 25, 2: 25}
    assert len(panels.display_timecourse) == 450
    for actual, table_id in (
        (panels.prototypes, "ligand_receptor_cluster_prototypes"),
        (panels.assignments, "ligand_receptor_cluster_assignments"),
        (panels.normalized_profiles, "ligand_receptor_normalized_profiles"),
        (panels.diagnostics, "ligand_receptor_cluster_diagnostics"),
        (panels.display_roster, "ligand_receptor_display_roster"),
        (panels.display_timecourse, "ligand_receptor_pair_timecourse"),
    ):
        pd.testing.assert_frame_equal(
            actual,
            data.tables[table_id],
            check_exact=False,
            rtol=1e-12,
            atol=1e-12,
        )

    # Secondary k>2 solutions can change across scikit-learn releases.  The
    # scientific contract is that k=2 remains best and its result is identical.
    calculated_k = panels.k_selection.sort_values(
        ["silhouette", "k"], ascending=[False, True]
    )
    formal_k = data.tables["ligand_receptor_k_selection"].sort_values(
        ["silhouette", "k"], ascending=[False, True]
    )
    assert int(calculated_k.iloc[0]["k"]) == 2
    assert int(formal_k.iloc[0]["k"]) == 2
    pd.testing.assert_series_equal(
        panels.k_selection.set_index("k").loc[2],
        data.tables["ligand_receptor_k_selection"].set_index("k").loc[2],
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )


def test_arista_lr_writer_uses_calculated_panels(tmp_path: Path) -> None:
    data = load_arista_supplementary_figures()
    panels = calculate_arista_ligand_receptor_panels(data)
    written = write_arista_ligand_receptor_tables(data, tmp_path, panels)
    assert set(written) == {
        "ligand_receptor_all_pair_timecourse",
        "ligand_receptor_cluster_prototypes",
        "ligand_receptor_cluster_assignments",
        "ligand_receptor_normalized_profiles",
        "ligand_receptor_k_selection",
        "ligand_receptor_cluster_diagnostics",
        "ligand_receptor_display_roster",
        "ligand_receptor_pair_timecourse",
    }
    pd.testing.assert_frame_equal(
        pd.read_csv(written["ligand_receptor_cluster_assignments"]),
        panels.assignments,
    )
    pd.testing.assert_frame_equal(
        pd.read_csv(written["ligand_receptor_display_roster"]),
        panels.display_roster,
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )


def test_arista_supplementary_selection_is_current_and_unique() -> None:
    pages = calculate_arista_supplementary_pages(load_arista_supplementary_figures())
    assert [
        page.figure
        for page in select_arista_supplementary_pages(
            pages,
            ("S24", "S19"),
        )
    ] == ["S19", "S24"]
    with pytest.raises(ValueError, match="duplicates"):
        select_arista_supplementary_pages(pages, ("S19", "S19"))
    with pytest.raises(ValueError, match="Unknown"):
        select_arista_supplementary_pages(pages, ("S18",))


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
            "S19",
            "S24",
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
    assert summary["figures"] == ["S19", "S24"]
    assert set(summary["files"]) == {"S19", "S24"}
    assert summary["formal_source_index"] == "arista_formal_source_index.csv"
    assert (output / summary["formal_source_index"]).is_file()
    assert str(tmp_path) not in completed.stdout


def test_arista_cli_keeps_compact_fallback_without_repository_release(
    tmp_path: Path,
) -> None:
    output = tmp_path / "compact-cli"
    missing_release = tmp_path / "release-not-installed"
    completed = subprocess.run(
        [
            sys.executable,
            str(PACKAGE_ROOT / "scripts/results/plot_arista_figures.py"),
            "--output-dir",
            str(output),
            "--figures",
            "S19",
        ],
        cwd=PACKAGE_ROOT,
        check=True,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            ARISTA_RELEASE_ENVIRONMENT_VARIABLE: str(missing_release),
            "MPLBACKEND": "Agg",
            "MPLCONFIGDIR": str(tmp_path / "compact-mpl"),
            "PYTHONPATH": str(PACKAGE_ROOT),
        },
    )
    summary = json.loads(completed.stdout)
    assert summary["figures"] == ["S19"]
    assert summary["formal_source_index"] is None
    assert (output / summary["files"]["S19"]["pdf"]).is_file()
    assert (output / summary["files"]["S19"]["png"]).is_file()


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
    path = PACKAGE_ROOT / "docs/tutorials/paper_figures/arista_figures.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", ())) for cell in notebook["cells"])
    assert "Supplementary Figures S19–S24" in source
    assert "from CytoBridge.results import" in source
    assert 'output_dir = Path("outputs") / "arista_supplementary_figures"' in source
    assert "load_arista_figure_release()" in source
    assert "formal_release.source_index" in source
    assert "formal_pdf" in source
    assert "formal_svg" in source
    assert "canonical_scripts" in source
    assert "release_build_snapshot" in source
    assert "build_scope" in source
    assert "calculation_entrypoints" in source
    assert "downstream_inputs" in source
    assert "input_scope" in source
    assert "checkpoint_inputs" in source
    assert "plot_arista_ligand_receptor_figures" in source
    assert "calculate_arista_ligand_receptor_panels" in source
    assert "table-driven scientific redraw" in source.lower()
    assert "Draw S23 and S24 from the calculated tables" in source
    assert "released reference pages" in source.lower()
    assert "export_arista_reference_pages" in source
    assert "plot_arista_supplementary_figures" not in source
    assert "write_arista_ligand_receptor_tables(data, output_dir, panels)" in source
    assert 'figures=("S19", "S20", "S21", "S22")' in source
    assert "full_recompute_inputs" in source
    assert "217" not in source
    assert "314" not in source
