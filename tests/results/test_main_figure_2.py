from __future__ import annotations

from importlib.resources import files
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pymupdf as fitz
import numpy as np
import pandas as pd
from PIL import Image
import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from CytoBridge.results.main_figure_2 import (  # noqa: E402
    assemble_main_figure_2,
    load_main_figure_2,
    plot_main_figure_2,
    summarize_main_figure_2_replicates,
)


def _fixture_copy(tmp_path: Path) -> Path:
    target = tmp_path / "main_figure_2"
    shutil.copytree(load_main_figure_2().source_dir, target)
    return target


def _render(path: Path, *, dpi: int = 144) -> np.ndarray:
    with fitz.open(path) as document:
        pixmap = document[0].get_pixmap(
            matrix=fitz.Matrix(dpi / 72, dpi / 72), alpha=False
        )
    return np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
        pixmap.height, pixmap.width, pixmap.n
    )[:, :, :3]


def test_packaged_main_figure_2_contract() -> None:
    data = load_main_figure_2()
    assert len(data.summary) == 6
    assert len(data.replicates) == 60
    assert len(data.baselines) == 9
    assert data.manifest["analysis"] == "main_figure_2"
    assert data.manifest["reader_action"] == (
        "result-summary-redraw + external-assembly"
    )
    assert data.manifest["full_rerun"]["included"] is False
    assert data.manifest["full_rerun"]["external_dependencies"]
    calculated = summarize_main_figure_2_replicates(data.replicates)
    assert np.allclose(
        calculated[["mean_w2", "sd_w2"]],
        data.summary[["mean_w2", "sd_w2"]],
        rtol=1e-12,
        atol=1e-14,
    )


def test_main_figure_2_resources_are_packaged() -> None:
    root = files("CytoBridge.results").joinpath("data", "main_figure_2")
    expected = {
        "baseline_w2.csv",
        "frozen_panels_a_to_d.pdf",
        "manifest.json",
        "w2_mean_sd_ci.csv",
        "w2_replicates_long.csv",
    }
    for name in expected:
        assert root.joinpath(name).is_file()


def test_main_figure_2_plot_dependency_is_declared() -> None:
    for filename in ("all.txt", "notebook.txt", "plot.txt"):
        text = (REPOSITORY_ROOT / "requirements" / filename).read_text(
            encoding="utf-8"
        )
        assert "PyMuPDF>=1.24,<2" in text


def test_main_figure_2_assembly_changes_only_panel_e(tmp_path: Path) -> None:
    data = load_main_figure_2()
    pdf, png = assemble_main_figure_2(data, tmp_path)
    assert pdf.is_file() and png.is_file()
    with Image.open(png) as image:
        assert image.size == (2481, 3508)
        image.verify()
    with fitz.open(pdf) as document:
        assert document.page_count == 1
        page = document[0]
        assert np.isclose(page.rect.width, 595.276, atol=0.02)
        assert np.isclose(page.rect.height, 841.89, atol=0.02)
        text = page.get_text("text").replace("\u00a0", " ")
    assert "Pearson’s r=0.96" in text
    assert "Wasserstein-2 distance" in text

    before = _render(data.frozen_panels_pdf)
    after = _render(pdf)
    assert before.shape == after.shape
    mask = np.zeros(before.shape[:2], dtype=bool)
    scale = 144 / 72
    mask[
        math.floor((841.89 - 225) * scale) :,
        math.floor(198 * scale) :,
    ] = True
    changed = np.any(before != after, axis=2)
    assert not changed[~mask].any()
    assert changed[mask].sum() > 1_000


def test_main_figure_2_rejects_summary_replicate_mismatch(tmp_path: Path) -> None:
    results_dir = _fixture_copy(tmp_path)
    path = results_dir / "w2_replicates_long.csv"
    table = pd.read_csv(path)
    table.loc[0, "w2"] += 0.1
    table.to_csv(path, index=False)
    with pytest.raises(ValueError, match="replicate calculations"):
        load_main_figure_2(results_dir)


def test_main_figure_2_cli(tmp_path: Path) -> None:
    output = tmp_path / "cli"
    completed = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "scripts/results/assemble_main_figure_2.py"),
            "--output-dir",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "MPLBACKEND": "Agg",
            "MPLCONFIGDIR": str(tmp_path / "mpl"),
            "PYTHONPATH": str(REPOSITORY_ROOT),
        },
    )
    summary = json.loads(completed.stdout)
    assert summary["analysis"] == "main_figure_2"
    assert summary["figure_action"] == "result-summary-redraw + external-assembly"
    assert summary["frozen_panels"] == ["a", "b", "c", "d"]
    assert summary["redrawn_panels"] == ["e"]
    assert Path(summary["pdf"]).is_file()
    assert Path(summary["png"]).is_file()
    assert set(summary["tables"]) == {"baselines", "replicates", "summary"}
    assert json.loads((output / "run_summary.json").read_text()) == summary


def test_main_figure_2_legacy_plot_name_remains_available() -> None:
    assert callable(plot_main_figure_2)
    assert "Compatibility alias" in (plot_main_figure_2.__doc__ or "")
