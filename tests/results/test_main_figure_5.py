from __future__ import annotations

import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import zlib

import pymupdf as fitz
import matplotlib as mpl
import pandas as pd
from PIL import Image, ImageChops
import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE_ROOT))

from CytoBridge.results.main_figure_5 import (  # noqa: E402
    PANEL_ORDER,
    calculate_main_figure_5,
    export_main_figure_5_reference_page,
    load_main_figure_5,
    plot_main_figure_5,
    validate_main_figure_5_reference_page,
    write_main_figure_5_tables,
)


def _fixture_copy(tmp_path: Path) -> Path:
    source = load_main_figure_5().source_dir
    target = tmp_path / "main_figure_5"
    shutil.copytree(source, target)
    return target


def _crc32(path: Path) -> str:
    return f"{zlib.crc32(path.read_bytes()) & 0xFFFFFFFF:08x}"


def test_packaged_main_figure_5_contract() -> None:
    data = load_main_figure_5()
    page = validate_main_figure_5_reference_page(data)
    assert tuple(data.panel_index["panel"]) == PANEL_ORDER
    assert page.width_pixels == 2481
    assert page.height_pixels == 3508
    assert page.width_points == pytest.approx(595.276)
    assert page.height_points == pytest.approx(841.89)
    assert page.reference_dpi == 300
    assert page.raster_crc32 == "1331a768"
    assert page.panel_count == 5
    assert data.manifest["scientific_label_release"] == "v5"
    assert data.manifest["reader_action"] == "reference-export"
    assert data.manifest["numerical_recalculation"] is False
    assert data.manifest["scientific_labels"] == {
        "Spatial migration velocity": "Spatial velocity",
        "Spatial velocity cosine simlarity": "Spatial velocity cosine similarity",
        "interaction VS migration": "interaction vs full spatial velocity",
    }


def test_full_recompute_registry_uses_relative_paths() -> None:
    registry = load_main_figure_5().full_recompute_inputs
    assert set(registry["input_id"]) >= {
        "raw_h5ad",
        "aligned_h5ad",
        "model_dir",
        "slice_dir",
        "velocity_components",
        "growth_by_cell",
        "communication_by_celltype",
    }
    for value in registry["relative_path"]:
        path = PurePosixPath(value)
        assert not path.is_absolute()
        assert ".." not in path.parts
        assert not str(value).startswith(("/Users/", "/home/", "/data/"))
    source = registry.loc[registry["input_id"].eq("raw_h5ad"), "public_source"].item()
    assert source.startswith("https://")


def test_main_figure_5_tables_are_written(tmp_path: Path) -> None:
    data = load_main_figure_5()
    page = validate_main_figure_5_reference_page(data)
    paths = write_main_figure_5_tables(data, page, tmp_path)
    assert {name: path.name for name, path in paths.items()} == {
        "panels": "main_figure_5_panel_index.csv",
        "inputs": "main_figure_5_full_recompute_inputs.csv",
        "page": "main_figure_5_page.csv",
    }
    assert pd.read_csv(paths["panels"])["panel"].tolist() == list(PANEL_ORDER)
    assert pd.read_csv(paths["page"]).loc[0, "panel_count"] == 5


def test_main_figure_5_reference_export_is_png_exact_and_a4(tmp_path: Path) -> None:
    mpl.use("Agg", force=True)
    before = mpl.rcParams.copy()
    data = load_main_figure_5()
    page = validate_main_figure_5_reference_page(data)
    pdf, png = export_main_figure_5_reference_page(data, tmp_path, page)
    assert png.read_bytes() == data.raster_path.read_bytes()
    assert _crc32(png) == "1331a768"
    with Image.open(png) as image:
        assert image.size == (2481, 3508)
        image.verify()
    with fitz.open(pdf) as document:
        assert document.page_count == 1
        assert document[0].rect.width == pytest.approx(595.276, abs=0.01)
        assert document[0].rect.height == pytest.approx(841.89, abs=0.01)
        pixmap = document[0].get_pixmap(
            matrix=fitz.Matrix(page.reference_dpi / 72.0, page.reference_dpi / 72.0),
            alpha=False,
        )
        rendered = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
    with Image.open(png) as source:
        difference = ImageChops.difference(source.convert("RGB"), rendered)
    assert difference.getbbox() is None
    assert mpl.rcParams["font.family"] == before["font.family"]
    assert mpl.rcParams["font.size"] == before["font.size"]


def test_main_figure_5_module_import_is_matplotlib_lazy() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import CytoBridge.results.main_figure_5; "
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


def test_main_figure_5_cli_has_sanitized_summary(tmp_path: Path) -> None:
    output = tmp_path / "cli"
    completed = subprocess.run(
        [
            sys.executable,
            str(
                PACKAGE_ROOT
                / "scripts/results/export_main_figure_5_reference_page.py"
            ),
            "--output-dir",
            str(output),
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
    assert summary["analysis"] == "main_figure_5"
    assert summary["figure_action"] == "reference-export"
    assert summary["numerical_recalculation"] is False
    assert summary["scientific_label_release"] == "v5"
    assert summary["source"] == "packaged"
    assert summary["panel_count"] == 5
    assert summary["canvas_pixels"] == [2481, 3508]
    assert str(tmp_path) not in completed.stdout
    assert json.loads((output / "run_summary.json").read_text()) == summary


def test_main_figure_5_legacy_names_remain_available() -> None:
    data = load_main_figure_5()
    assert calculate_main_figure_5(data) == validate_main_figure_5_reference_page(data)
    assert callable(plot_main_figure_5)
    assert "Compatibility alias" in (plot_main_figure_5.__doc__ or "")


def test_changed_raster_is_rejected(tmp_path: Path) -> None:
    results_dir = _fixture_copy(tmp_path)
    path = results_dir / "main_figure_5_compact.png"
    payload = bytearray(path.read_bytes())
    payload[-20] ^= 1
    path.write_bytes(payload)
    with pytest.raises(ValueError, match="raster digest"):
        load_main_figure_5(results_dir)


@pytest.mark.parametrize(
    "absolute_path",
    ["/data/arista/Regeneration.h5ad", "C:\\arista\\Regeneration.h5ad"],
)
def test_absolute_registry_path_is_rejected(
    tmp_path: Path, absolute_path: str
) -> None:
    results_dir = _fixture_copy(tmp_path)
    path = results_dir / "full_recompute_inputs.csv"
    table = pd.read_csv(path, keep_default_na=False)
    table.loc[0, "relative_path"] = absolute_path
    table.to_csv(path, index=False)
    with pytest.raises(ValueError, match="non-relative input path"):
        load_main_figure_5(results_dir)
