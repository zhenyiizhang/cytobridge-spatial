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

from CytoBridge.results.interaction_evidence import (  # noqa: E402
    interaction_evidence_statistics,
    load_interaction_evidence_results,
    plot_interaction_evidence,
)


def _fixture_copy(tmp_path: Path) -> Path:
    source = load_interaction_evidence_results().source_dir
    target = tmp_path / "interaction_evidence"
    shutil.copytree(source, target)
    return target


def test_packaged_interaction_evidence_contract() -> None:
    results = load_interaction_evidence_results()
    statistics = interaction_evidence_statistics(results)
    assert results.source_dir.name == "interaction_evidence"
    assert len(results.no_lr) == 48
    assert len(results.stvcr) == 33
    assert len(results.panel_summary) == 30
    assert "source_scope" not in results.no_lr
    assert "source_scope" not in results.stvcr
    assert "outcome" not in results.stvcr
    assert results.manifest["analysis"] == "interaction_evidence"
    assert np.isclose(
        statistics["no_lr_mean_relative_change_arista"],
        1.772661196476827,
    )
    assert np.isclose(
        statistics["no_lr_mean_relative_change_chicken_heart"],
        0.01817799522095409,
    )
    manifest_text = (results.source_dir / "manifest.json").read_text(encoding="utf-8")
    assert "/Users/" not in manifest_text
    assert "sha256" not in manifest_text.lower()
    assert "2026" not in manifest_text


def test_interaction_evidence_plot_is_agg_safe_and_local(tmp_path: Path) -> None:
    import matplotlib as mpl

    mpl.use("Agg", force=True)
    before = mpl.rcParams.copy()
    pdf, png = plot_interaction_evidence(
        load_interaction_evidence_results(),
        tmp_path,
    )
    assert pdf.is_file() and pdf.stat().st_size > 0
    with Image.open(png) as image:
        assert image.size == (2646, 3740)
        image.verify()
    assert mpl.rcParams["font.family"] == before["font.family"]
    assert mpl.rcParams["font.size"] == before["font.size"]


def test_interaction_evidence_cli_summary(tmp_path: Path) -> None:
    output = tmp_path / "cli"
    completed = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "scripts/results/plot_interaction_evidence.py"),
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
    assert summary["analysis"] == "interaction_evidence"
    assert json.loads((output / "run_summary.json").read_text()) == summary


def test_interaction_evidence_missing_column(tmp_path: Path) -> None:
    results_dir = _fixture_copy(tmp_path)
    path = results_dir / "no_lr_paired_target_deltas.csv"
    pd.read_csv(path).drop(columns=["full"]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        load_interaction_evidence_results(results_dir)


def test_interaction_evidence_duplicate_key(tmp_path: Path) -> None:
    results_dir = _fixture_copy(tmp_path)
    path = results_dir / "stvcr_paired_target_deltas.csv"
    table = pd.read_csv(path)
    pd.concat([table, table.iloc[[0]]], ignore_index=True).to_csv(path, index=False)
    with pytest.raises(ValueError, match="duplicate dataset-target-space"):
        load_interaction_evidence_results(results_dir)


def test_interaction_evidence_nonfinite_value(tmp_path: Path) -> None:
    results_dir = _fixture_copy(tmp_path)
    path = results_dir / "no_lr_paired_target_deltas.csv"
    table = pd.read_csv(path)
    table.loc[0, "full"] = np.nan
    table.to_csv(path, index=False)
    with pytest.raises(ValueError, match="non-finite"):
        load_interaction_evidence_results(results_dir)


def test_interaction_evidence_unknown_dataset(tmp_path: Path) -> None:
    results_dir = _fixture_copy(tmp_path)
    path = results_dir / "stvcr_paired_target_deltas.csv"
    table = pd.read_csv(path)
    table.loc[0, "dataset"] = "unknown_dataset"
    table.to_csv(path, index=False)
    with pytest.raises(ValueError, match="unknown datasets"):
        load_interaction_evidence_results(results_dir)
