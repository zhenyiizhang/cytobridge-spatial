from __future__ import annotations

from importlib.resources import files
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

from CytoBridge.results.lr_complex_aggregation import (  # noqa: E402
    load_lr_complex_aggregation_results,
    plot_lr_complex_aggregation,
)


def _fixture_copy(tmp_path: Path) -> Path:
    source = load_lr_complex_aggregation_results().source_dir
    target = tmp_path / "lr_complex_aggregation"
    shutil.copytree(source, target)
    return target


def test_packaged_lr_complex_aggregation_contract() -> None:
    results = load_lr_complex_aggregation_results()
    assert results.source_dir.name == "lr_complex_aggregation"
    assert len(results.paired_scores) == 38_781
    assert len(results.per_time_summary) == 64
    assert len(results.dataset_summary) == 4
    assert results.manifest["analysis"] == "lr_complex_aggregation"

    summary = results.dataset_summary.set_index("dataset")
    assert int(summary.loc["ARISTA", "n_scored_pairs"]) == 531
    assert int(summary.loc["ARISTA", "n_multisubunit_pairs"]) == 293
    assert np.isclose(
        summary.loc["ARISTA", "pooled_spearman"],
        0.9601058481896767,
    )
    assert np.isclose(
        summary.loc["ARISTA", "min_top10_jaccard"],
        2.0 / 3.0,
    )

    zebrafish = results.per_time_summary.loc[
        results.per_time_summary["dataset"].eq("Zebrafish")
        & results.per_time_summary["scope"].eq("all_scored_pairs")
    ]
    assert zebrafish.loc[zebrafish["time"].isin([1.0, 1.5]), "spearman"].isna().all()
    assert zebrafish["normalized_time"].min() == 0.0
    assert zebrafish["normalized_time"].max() == 1.0
    assert not np.isclose(
        summary.loc["Zebrafish", "pooled_spearman"],
        zebrafish["spearman"].mean(),
    )

    manifest_text = (results.source_dir / "manifest.json").read_text(encoding="utf-8")
    assert "/Users/" not in manifest_text
    assert "sha256" not in manifest_text.lower()
    assert "2026" not in manifest_text


def test_lr_complex_aggregation_package_resources() -> None:
    root = files("CytoBridge.results").joinpath("data", "lr_complex_aggregation")
    assert root.joinpath("manifest.json").is_file()
    for dataset in ("zebrafish", "mosta", "arista", "chicken_heart"):
        resource = root.joinpath(dataset, "paired_scores.csv")
        assert resource.is_file()
        with resource.open("rb") as handle:
            assert handle.readline().startswith(b"time,pair,score_min")


def test_lr_complex_aggregation_plot_is_agg_safe_and_local(tmp_path: Path) -> None:
    import matplotlib as mpl

    mpl.use("Agg", force=True)
    before = mpl.rcParams.copy()
    pdf, png = plot_lr_complex_aggregation(
        load_lr_complex_aggregation_results(),
        tmp_path,
    )
    assert pdf.is_file() and pdf.stat().st_size > 0
    with Image.open(png) as image:
        assert image.size == (2646, 3740)
        image.verify()
    assert mpl.rcParams["font.family"] == before["font.family"]
    assert mpl.rcParams["font.size"] == before["font.size"]


def test_lr_complex_aggregation_cli_summary(tmp_path: Path) -> None:
    output = tmp_path / "cli"
    completed = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "scripts/results/plot_lr_complex_aggregation.py"),
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
    assert summary["analysis"] == "lr_complex_aggregation"
    assert json.loads((output / "run_summary.json").read_text()) == summary


def test_lr_complex_aggregation_missing_column(tmp_path: Path) -> None:
    results_dir = _fixture_copy(tmp_path)
    path = results_dir / "arista/paired_scores.csv"
    pd.read_csv(path).drop(columns=["score_min"]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        load_lr_complex_aggregation_results(results_dir)


def test_lr_complex_aggregation_duplicate_key(tmp_path: Path) -> None:
    results_dir = _fixture_copy(tmp_path)
    path = results_dir / "mosta/paired_scores.csv"
    table = pd.read_csv(path)
    pd.concat([table, table.iloc[[0]]], ignore_index=True).to_csv(path, index=False)
    with pytest.raises(ValueError, match="duplicate dataset-time-pair"):
        load_lr_complex_aggregation_results(results_dir)


def test_lr_complex_aggregation_nonfinite_score(tmp_path: Path) -> None:
    results_dir = _fixture_copy(tmp_path)
    path = results_dir / "zebrafish/paired_scores.csv"
    table = pd.read_csv(path)
    table.loc[0, "score_geometric_mean"] = np.nan
    table.to_csv(path, index=False)
    with pytest.raises(ValueError, match="non-finite"):
        load_lr_complex_aggregation_results(results_dir)


def test_lr_complex_aggregation_negative_score(tmp_path: Path) -> None:
    results_dir = _fixture_copy(tmp_path)
    path = results_dir / "chicken_heart/paired_scores.csv"
    table = pd.read_csv(path)
    table.loc[0, "score_min"] = -1.0
    table.to_csv(path, index=False)
    with pytest.raises(ValueError, match="negative LR scores"):
        load_lr_complex_aggregation_results(results_dir)


def test_lr_complex_aggregation_constant_pair_annotation(tmp_path: Path) -> None:
    results_dir = _fixture_copy(tmp_path)
    path = results_dir / "arista/paired_scores.csv"
    table = pd.read_csv(path)
    pair = table.iloc[0]["pair"]
    row = table.index[table["pair"].eq(pair)][1]
    table.loc[row, "is_multisubunit"] = not bool(table.loc[row, "is_multisubunit"])
    table.to_csv(path, index=False)
    with pytest.raises(ValueError, match="changes is_multisubunit"):
        load_lr_complex_aggregation_results(results_dir)
