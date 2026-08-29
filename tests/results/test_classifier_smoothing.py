from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
from PIL import Image
import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from CytoBridge.results.classifier_smoothing import (  # noqa: E402
    classifier_smoothing_statistics,
    load_classifier_smoothing_results,
    plot_classifier_smoothing,
)


def _fixture_copy(tmp_path: Path) -> Path:
    source = load_classifier_smoothing_results().source_dir
    target = tmp_path / "classifier_smoothing"
    shutil.copytree(source, target)
    return target


def test_results_import_keeps_plotting_optional() -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPOSITORY_ROOT)
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import CytoBridge.results; print('matplotlib' in sys.modules)",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert completed.stdout.strip() == "False"


def test_packaged_classifier_smoothing_contract() -> None:
    results = load_classifier_smoothing_results()
    statistics = classifier_smoothing_statistics(results)
    assert results.source_dir.name == "classifier_smoothing"
    assert len(results.metrics) == 25
    assert len(results.frames) == 45
    assert len(results.intervals) == 20
    assert len(results.composition) == len(results.transition) == 5
    assert results.manifest["analysis"] == "classifier_smoothing"
    assert np.isclose(
        statistics["zebrafish_composition_tv_percent_k10"],
        7.649366770733167,
    )
    assert np.isclose(
        statistics["zebrafish_transition_percentage_point_delta_k10_vs_k1"],
        0.04440497335701821,
    )
    manifest_text = (results.source_dir / "manifest.json").read_text(encoding="utf-8")
    assert "/Users/" not in manifest_text
    assert "sha256" not in manifest_text.lower()


def test_classifier_smoothing_plot_is_agg_safe_and_local(tmp_path: Path) -> None:
    import matplotlib as mpl

    mpl.use("Agg", force=True)
    before = mpl.rcParams.copy()
    pdf, png = plot_classifier_smoothing(
        load_classifier_smoothing_results(),
        tmp_path,
    )
    assert pdf.is_file() and pdf.stat().st_size > 0
    with Image.open(png) as image:
        assert image.size == (3740, 2646)
        image.verify()
    assert mpl.rcParams["font.family"] == before["font.family"]
    assert mpl.rcParams["font.size"] == before["font.size"]


def test_classifier_smoothing_cli_summary(tmp_path: Path) -> None:
    output = tmp_path / "cli"
    completed = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "scripts/results/plot_classifier_smoothing.py"),
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
    assert set(summary) == {"analysis", "input_directory", "pdf", "png", "tables"}
    assert summary["analysis"] == "classifier_smoothing"
    assert json.loads((output / "run_summary.json").read_text()) == summary


def test_classifier_smoothing_missing_column(tmp_path: Path) -> None:
    results_dir = _fixture_copy(tmp_path)
    path = results_dir / "five_dataset_k_metrics.csv"
    pd.read_csv(path).drop(columns=["balanced_accuracy"]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        load_classifier_smoothing_results(results_dir)


def test_classifier_smoothing_duplicate_key(tmp_path: Path) -> None:
    results_dir = _fixture_copy(tmp_path)
    path = results_dir / "five_dataset_k_metrics.csv"
    table = pd.read_csv(path)
    pd.concat([table, table.iloc[[0]]], ignore_index=True).to_csv(path, index=False)
    with pytest.raises(ValueError, match="duplicate dataset-k"):
        load_classifier_smoothing_results(results_dir)


def test_classifier_smoothing_nonfinite_value(tmp_path: Path) -> None:
    results_dir = _fixture_copy(tmp_path)
    path = results_dir / "five_dataset_k_metrics.csv"
    table = pd.read_csv(path)
    table.loc[0, "balanced_accuracy"] = np.nan
    table.to_csv(path, index=False)
    with pytest.raises(ValueError, match="non-finite"):
        load_classifier_smoothing_results(results_dir)


def test_classifier_smoothing_unknown_dataset(tmp_path: Path) -> None:
    results_dir = _fixture_copy(tmp_path)
    path = results_dir / "five_dataset_k_metrics.csv"
    table = pd.read_csv(path)
    table.loc[0, "dataset"] = "unknown_dataset"
    table.to_csv(path, index=False)
    with pytest.raises(ValueError, match="unknown datasets"):
        load_classifier_smoothing_results(results_dir)
