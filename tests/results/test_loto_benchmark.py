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

from CytoBridge.results.loto_benchmark import (  # noqa: E402
    compute_paired_loto_ratios,
    load_loto_benchmark,
    plot_loto_benchmark,
    summarize_loto_ratios,
)


def _fixture_copy(tmp_path: Path) -> Path:
    source = load_loto_benchmark().source_dir
    target = tmp_path / "loto_benchmark"
    shutil.copytree(source, target)
    return target


def _sorted(table: pd.DataFrame) -> pd.DataFrame:
    keys = [
        name
        for name in ("method", "dataset", "target", "space")
        if name in table.columns
    ]
    return table.sort_values(keys).reset_index(drop=True)


def test_packaged_loto_benchmark_contract() -> None:
    data = load_loto_benchmark()
    assert len(data.target_means) == 231
    assert len(data.paired_ratios) == 198
    assert len(data.dataset_summary) == 40
    assert len(data.native_support) == 99
    assert data.protocol["projection"] == {
        "repeats_per_dataset_target_space": 5,
        "directions_per_repeat": 1024,
        "sharing_scope": "dataset-target-space-repeat",
        "applicable_methods_share_directions_within_repeat": True,
        "shared_across_datasets": False,
        "method_specific_directions": False,
    }

    assert list(data.paired_ratios.columns) == [
        "method",
        "display_name",
        "dataset",
        "target",
        "space",
        "cytobridge_sliced_w2",
        "method_sliced_w2",
        "method_to_cytobridge_ratio",
    ]
    assert list(data.dataset_summary.columns) == [
        "method",
        "display_name",
        "dataset",
        "relative_sliced_w2",
        "median_relative_sliced_w2",
        "n_paired_comparisons",
    ]
    stvcr_arista = data.dataset_summary.loc[
        data.dataset_summary["method"].eq("stvcr")
        & data.dataset_summary["dataset"].eq("arista")
    ].iloc[0]
    assert np.isclose(stvcr_arista["relative_sliced_w2"], 1.1184797905976476)
    assert int(stvcr_arista["n_paired_comparisons"]) == 9


def test_loto_ratios_use_equal_matched_cell_weights() -> None:
    target_means = pd.DataFrame(
        [
            ("d", 1, "CytoBridge-0.015", "CytoBridge", "state", 1.0),
            ("d", 2, "CytoBridge-0.015", "CytoBridge", "state", 100.0),
            ("d", 1, "stvcr", "stVCR", "state", 2.0),
            ("d", 2, "stvcr", "stVCR", "state", 100.0),
        ],
        columns=[
            "dataset",
            "target",
            "method",
            "display_name",
            "space",
            "sliced_w2",
        ],
    )
    paired = compute_paired_loto_ratios(target_means)
    summary = summarize_loto_ratios(paired)
    assert np.isclose(summary.loc[0, "relative_sliced_w2"], 1.5)
    assert not np.isclose(
        summary.loc[0, "relative_sliced_w2"],
        target_means.loc[target_means["method"].eq("stvcr"), "sliced_w2"].mean()
        / target_means.loc[
            target_means["method"].eq("CytoBridge-0.015"), "sliced_w2"
        ].mean(),
    )


def test_state_only_methods_keep_only_applicable_cells() -> None:
    data = load_loto_benchmark()
    target_counts = {
        "zebrafish": 3,
        "mosta": 2,
        "arista": 3,
        "admouse": 1,
        "chicken_heart": 2,
    }
    for method in ("mioflow", "stories", "wot"):
        rows = data.paired_ratios.loc[data.paired_ratios["method"].eq(method)]
        assert set(rows["space"]) == {"state"}
        assert rows.groupby("dataset").size().to_dict() == target_counts
    full = data.paired_ratios.loc[data.paired_ratios["method"].eq("stvcr")]
    assert set(full["space"]) == {"joint", "spatial", "state"}
    assert full.groupby("dataset").size().to_dict() == {
        name: count * 3 for name, count in target_counts.items()
    }


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda table: table.drop(columns=["sliced_w2"]), "missing columns"),
        (
            lambda table: pd.concat([table, table.iloc[[0]]], ignore_index=True),
            "duplicate dataset-target-method-space",
        ),
        (
            lambda table: table.assign(
                sliced_w2=table["sliced_w2"].mask(table.index == 0, np.nan)
            ),
            "non-finite",
        ),
        (
            lambda table: table.assign(
                sliced_w2=table["sliced_w2"].mask(table.index == 0, 0.0)
            ),
            "non-positive",
        ),
        (
            lambda table: table.assign(
                n_projection_repeats=table["n_projection_repeats"].mask(
                    table.index == 0, 4
                )
            ),
            "five projection repeats",
        ),
        (
            lambda table: table.assign(
                dataset=table["dataset"].mask(table.index == 0, "unknown")
            ),
            "unknown datasets",
        ),
    ],
)
def test_loto_target_mean_errors(tmp_path: Path, mutation, message: str) -> None:
    results_dir = _fixture_copy(tmp_path)
    path = results_dir / "loto_target_stage_means.csv"
    mutation(pd.read_csv(path)).to_csv(path, index=False)
    with pytest.raises(ValueError, match=message):
        load_loto_benchmark(results_dir)


def test_loto_method_applicability_error(tmp_path: Path) -> None:
    results_dir = _fixture_copy(tmp_path)
    path = results_dir / "loto_target_stage_means.csv"
    table = pd.read_csv(path)
    row = table.loc[table["method"].eq("mioflow")].iloc[0].copy()
    row["space"] = "joint"
    pd.concat([table, row.to_frame().T], ignore_index=True).to_csv(path, index=False)
    with pytest.raises(ValueError, match="space contract"):
        load_loto_benchmark(results_dir)


def test_loto_native_support_contract_and_stvcr_variation(tmp_path: Path) -> None:
    data = load_loto_benchmark()
    support = data.native_support
    assert support["initial_source_roster_n"].eq(5000).all()
    assert not support["target_size_resampling"].any()
    assert support["sliced_w2_predicted_weights"].eq("normalized_before_metric").all()
    assert support["sliced_w2_support"].eq("all_native_predicted_points").all()

    results_dir = _fixture_copy(tmp_path)
    path = results_dir / "native_output_support.csv"
    varied = pd.read_csv(path)
    index = varied.index[varied["method"].eq("stvcr")][0]
    varied.loc[index, "native_output_n"] = 123
    varied.loc[index, "output_support_differs_from_initial"] = True
    varied.to_csv(path, index=False)
    assert (
        load_loto_benchmark(results_dir).native_support.loc[index, "native_output_n"]
        == 123
    )


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("target_size_resampling", True, "target-size resampling"),
        ("sliced_w2_predicted_weights", "raw", "normalize predicted weights"),
        ("initial_source_roster_n", 4999, "start every method from 5,000"),
    ],
)
def test_loto_native_support_errors(
    tmp_path: Path, column: str, value: object, message: str
) -> None:
    results_dir = _fixture_copy(tmp_path)
    path = results_dir / "native_output_support.csv"
    table = pd.read_csv(path)
    table.loc[0, column] = value
    table.to_csv(path, index=False)
    with pytest.raises(ValueError, match=message):
        load_loto_benchmark(results_dir)


def test_loto_fixed_method_support_error(tmp_path: Path) -> None:
    results_dir = _fixture_copy(tmp_path)
    path = results_dir / "native_output_support.csv"
    table = pd.read_csv(path)
    row = table.index[table["method"].eq("moscot")][0]
    table.loc[row, "native_output_n"] = 4999
    table.loc[row, "output_support_differs_from_initial"] = True
    table.to_csv(path, index=False)
    with pytest.raises(ValueError, match="outside stVCR"):
        load_loto_benchmark(results_dir)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("directions_per_repeat", 256),
        ("repeats_per_dataset_target_space", 4),
    ],
)
def test_loto_projection_protocol_errors(
    tmp_path: Path, field: str, value: int
) -> None:
    results_dir = _fixture_copy(tmp_path)
    path = results_dir / "protocol.json"
    protocol = json.loads(path.read_text())
    protocol["projection"][field] = value
    path.write_text(json.dumps(protocol), encoding="utf-8")
    with pytest.raises(ValueError, match="projection protocol"):
        load_loto_benchmark(results_dir)


def test_loto_resources_and_lazy_plotting() -> None:
    root = files("CytoBridge.results").joinpath("data", "loto_benchmark")
    expected = {
        "manifest.json",
        "protocol.json",
        "loto_target_stage_means.csv",
        "native_output_support.csv",
    }
    for relative_name in expected:
        assert root.joinpath(*relative_name.split("/")).is_file()
    for relative_name in expected:
        resource = root.joinpath(*relative_name.split("/"))
        text = resource.read_text(encoding="utf-8")
        assert "/Users/" not in text
        assert "sha256" not in text.lower()

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import CytoBridge.results; print('matplotlib' in sys.modules)",
        ],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(REPOSITORY_ROOT)},
    )
    assert completed.stdout.strip() == "False"


def test_loto_plot_is_a4_agg_safe_and_local(tmp_path: Path) -> None:
    import matplotlib as mpl

    mpl.use("Agg", force=True)
    before = mpl.rcParams.copy()
    pdf, png = plot_loto_benchmark(load_loto_benchmark(), tmp_path)
    assert pdf.is_file() and pdf.stat().st_size > 0
    with Image.open(png) as image:
        assert image.size == (2646, 3740)
        assert image.info.get("Software") == "CytoBridge"
        image.verify()
    with fitz.open(pdf) as document:
        assert document.page_count == 1
        assert document.metadata.get("creator") == "CytoBridge"
        page = document[0].rect
        assert np.isclose(page.width, 595.44, atol=0.1)
        assert np.isclose(page.height, 841.68, atol=0.1)
    assert mpl.rcParams["font.family"] == before["font.family"]
    assert mpl.rcParams["font.size"] == before["font.size"]


def test_loto_cli_summary(tmp_path: Path) -> None:
    output = tmp_path / "cli"
    completed = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "scripts/results/plot_loto_benchmark.py"),
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
    assert summary["analysis"] == "loto_benchmark"
    assert (output / "paired_loto_ratios.csv").is_file()
    assert (output / "loto_dataset_summary.csv").is_file()
    assert json.loads((output / "run_summary.json").read_text()) == summary
