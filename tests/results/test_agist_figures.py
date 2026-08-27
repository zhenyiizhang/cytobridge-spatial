from __future__ import annotations

from importlib.resources import files
import json
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

from CytoBridge.results.agist_figures import (  # noqa: E402
    calculate_agist_figure_panels,
    load_agist_figures,
    plot_agist_figures,
)


def _fixture_copy(tmp_path: Path) -> Path:
    target = tmp_path / "agist_figures"
    shutil.copytree(load_agist_figures().source_dir, target)
    return target


def test_packaged_agist_figure_contract() -> None:
    data = load_agist_figures()
    panels = calculate_agist_figure_panels(data)

    assert data.manifest["analysis"] == "agist_figures"
    assert data.manifest["full_rerun"]["included"] is False
    assert len(data.velocity_per_cell) == 31_816
    assert data.observed_spatial.shape == (2243, 2)
    assert data.trajectory_ground_truth.shape == (201, 60, 4)
    assert np.array_equal(
        data.trajectory_indices,
        np.sort(np.random.default_rng(11).choice(400, size=60, replace=False)),
    )
    assert not data.full_recompute_inputs["relative_identifier"].str.startswith(
        "/"
    ).any()

    overall = panels.velocity_overall.set_index("velocity_space")
    assert np.isclose(overall.loc["physical", "mean"], 0.9263500515542803)
    assert np.isclose(overall.loc["gene", "mean"], 0.9442760367764322)
    assert np.isclose(
        data.growth_metrics.loc[
            data.growth_metrics["time"] > 0, "absolute_tmv"
        ].mean(),
        0.01508235216140747,
    )
    assert np.isclose(
        np.corrcoef(
            panels.potential_curve["true_potential"],
            panels.potential_curve["learned_potential"],
        )[0, 1],
        0.9799886066383152,
    )

    final_time = panels.ablation_summary[
        panels.ablation_summary["time"] == 4
    ].pivot(index="space", columns="condition", values="mean")
    differences = (
        final_time["interaction_off"] - final_time["interaction_on"]
    )
    assert np.isclose(differences["joint"], 0.10587785235314529)
    assert np.isclose(differences["spatial"], 0.06875923614865509)
    assert np.isclose(differences["gene"], 0.08201761622237475)


def test_agist_figure_resources_are_packaged() -> None:
    root = files("CytoBridge.results").joinpath("data", "agist_figures")
    expected = {
        "full_recompute_inputs.csv",
        "manifest.json",
        "s2_cluster_selection_diagnostics.csv",
        "s2_velocity_cosine_by_state_cluster.csv",
        "s2_velocity_cosine_by_time.csv",
        "s2_velocity_cosine_by_time_and_state_cluster.csv",
        "s2_velocity_cosine_overall.csv",
        "s2_velocity_cosine_per_cell.csv.gz",
        "s3_display_trajectories.npz",
        "s3_growth_mass_metrics.csv",
        "s3_interaction_ablation_metrics.csv",
        "s3_interaction_radial_curve.csv",
        "s3_observed_snapshots.npz",
    }
    assert {path.name for path in root.iterdir()} == expected


def test_agist_figures_render_vector_pdf_and_png(tmp_path: Path) -> None:
    data = load_agist_figures()
    panels = calculate_agist_figure_panels(data)
    figures = plot_agist_figures(data, panels, tmp_path)

    expected = {
        "s2": ((2646, 2320), (595.44, 522.0)),
        "s3": ((2646, 3740), (595.44, 841.68)),
    }
    for figure, (png_size, page_size) in expected.items():
        pdf, png = figures[figure]
        with Image.open(png) as image:
            assert image.size == png_size
            image.verify()
        with fitz.open(pdf) as document:
            assert document.page_count == 1
            assert np.isclose(document[0].rect.width, page_size[0], atol=0.02)
            assert np.isclose(document[0].rect.height, page_size[1], atol=0.02)
            text = document[0].get_text()
            assert len(document[0].get_drawings()) > 20
        assert "CytoBridge" in text or figure == "s2"


def test_agist_figures_reject_cell_summary_mismatch(tmp_path: Path) -> None:
    results_dir = _fixture_copy(tmp_path)
    path = results_dir / "s2_velocity_cosine_per_cell.csv.gz"
    table = pd.read_csv(path)
    table.loc[0, "physical_cosine"] -= 0.1
    table.to_csv(path, index=False, compression="gzip")
    with pytest.raises(ValueError, match="cell-level calculations"):
        load_agist_figures(results_dir)


def test_agist_figures_cli(tmp_path: Path) -> None:
    output = tmp_path / "cli"
    completed = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "scripts/results/plot_agist_figures.py"),
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
    assert summary["analysis"] == "agist_figures"
    assert set(summary["figures"]) == {"s2", "s3"}
    assert len(summary["tables"]) == 7
    for paths in summary["figures"].values():
        assert Path(paths["pdf"]).is_file()
        assert Path(paths["png"]).is_file()
    assert json.loads((output / "run_summary.json").read_text()) == summary
