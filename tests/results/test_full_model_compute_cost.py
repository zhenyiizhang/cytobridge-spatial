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
import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from CytoBridge.results.compute_cost import (  # noqa: E402
    DISPLAY_COLUMNS,
    RAW_COLUMNS,
    format_full_model_compute_cost,
    load_full_model_compute_cost,
    write_full_model_compute_cost_tables,
)


EXPECTED_RAW = pd.DataFrame(
    [
        (
            "admouse",
            3,
            "2.5, 5.7, 17.9 months",
            172092,
            696.4544131811708,
            2446.66015625,
            708.55419921875,
        ),
        (
            "arista",
            5,
            "2, 5, 10, 15, 20 DPI",
            46199,
            1324.5375007176772,
            9262.1328125,
            870.3583984375,
        ),
        (
            "chicken_heart",
            4,
            "D4, D7, D10, D14",
            3550,
            761.1574015710503,
            2833.04296875,
            1922.01904296875,
        ),
        (
            "mosta",
            4,
            "E12.5, E13.5, E14.5, E15.5",
            344603,
            1997.5040369811468,
            17023.75,
            1051.01806640625,
        ),
        (
            "zebrafish",
            5,
            "5.25, 10, 12, 18, 24 hpf",
            11999,
            1239.2258655983023,
            2174.1328125,
            577.80859375,
        ),
    ],
    columns=RAW_COLUMNS,
)

EXPECTED_DISPLAY = pd.DataFrame(
    [
        ("AD mouse", "3: 2.5, 5.7, 17.9 months", "172,092", "11.6", "2.39", "0.69"),
        ("ARISTA", "5: 2, 5, 10, 15, 20 DPI", "46,199", "22.1", "9.05", "0.85"),
        ("Chicken heart", "4: D4, D7, D10, D14", "3,550", "12.7", "2.77", "1.88"),
        ("MOSTA", "4: E12.5, E13.5, E14.5, E15.5", "344,603", "33.3", "16.62", "1.03"),
        ("Zebrafish", "5: 5.25, 10, 12, 18, 24 hpf", "11,999", "20.7", "2.12", "0.56"),
    ],
    columns=DISPLAY_COLUMNS,
)


def _fixture_copy(tmp_path: Path) -> Path:
    source = load_full_model_compute_cost().source_dir
    target = tmp_path / "full_model_compute_cost"
    shutil.copytree(source, target)
    return target


def test_packaged_full_model_compute_cost() -> None:
    results = load_full_model_compute_cost()
    pd.testing.assert_frame_equal(results.measurements, EXPECTED_RAW, check_exact=True)
    measurement = results.manifest["measurement"]
    assert measurement["training_stages"] == 6
    assert measurement["hardware"] == {
        "gpu_count": 1,
        "gpu_model": "NVIDIA GeForce RTX 4090 D",
    }
    assert measurement["training_time"]["scope"] == "TrainingPipeline.train"
    assert measurement["host_memory"]["unit"] == "MiB"
    assert measurement["gpu_memory"]["unit"] == "MiB"


def test_full_model_compute_cost_display() -> None:
    table = format_full_model_compute_cost(load_full_model_compute_cost())
    pd.testing.assert_frame_equal(table, EXPECTED_DISPLAY, check_exact=True)
    assert table.columns[1] == "Time points used for training"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda table: table.drop(columns=["training_time_seconds"]),
            "missing columns",
        ),
        (
            lambda table: pd.concat([table, table.iloc[[0]]], ignore_index=True),
            "duplicate datasets",
        ),
        (
            lambda table: table.assign(
                dataset=table["dataset"].mask(table.index == 0, "unknown")
            ),
            "five-dataset roster",
        ),
        (
            lambda table: table.assign(
                training_time_seconds=table["training_time_seconds"].mask(
                    table.index == 0, np.nan
                )
            ),
            "non-finite",
        ),
        (
            lambda table: table.assign(
                peak_host_memory_mib=table["peak_host_memory_mib"].mask(
                    table.index == 0, 0.0
                )
            ),
            "non-positive",
        ),
        (
            lambda table: table.assign(
                time_points_used_for_training=table[
                    "time_points_used_for_training"
                ].mask(table.index == 0, 2)
            ),
            "do not match the label lists",
        ),
    ],
)
def test_full_model_compute_cost_input_errors(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    results_dir = _fixture_copy(tmp_path)
    path = results_dir / "full_model_compute_cost.csv"
    mutation(pd.read_csv(path)).to_csv(path, index=False)
    with pytest.raises(ValueError, match=message):
        load_full_model_compute_cost(results_dir)


def test_full_model_compute_cost_manifest_error(tmp_path: Path) -> None:
    results_dir = _fixture_copy(tmp_path)
    path = results_dir / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["measurement"]["training_time"]["scope"] = "other"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="measurement contract"):
        load_full_model_compute_cost(results_dir)


def test_full_model_compute_cost_writer(tmp_path: Path) -> None:
    results = load_full_model_compute_cost()
    paths = write_full_model_compute_cost_tables(results, tmp_path)
    assert set(paths) == {"raw", "display", "markdown"}
    pd.testing.assert_frame_equal(
        pd.read_csv(paths["raw"], float_precision="round_trip"),
        EXPECTED_RAW,
        check_exact=True,
    )
    pd.testing.assert_frame_equal(
        pd.read_csv(paths["display"], dtype=str),
        EXPECTED_DISPLAY,
        check_exact=True,
    )
    markdown = paths["markdown"].read_text(encoding="utf-8")
    assert "| Time points used for training |" in markdown
    assert "| Chicken heart | 4: D4, D7, D10, D14 |" in markdown


def test_full_model_compute_cost_resources_and_import() -> None:
    root = files("CytoBridge.results").joinpath("data", "full_model_compute_cost")
    assert root.joinpath("full_model_compute_cost.csv").is_file()
    assert root.joinpath("manifest.json").is_file()

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import CytoBridge.results.compute_cost; "
                "print('matplotlib' in sys.modules)"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(REPOSITORY_ROOT)},
    )
    assert completed.stdout.strip() == "False"


def test_full_model_compute_cost_cli(tmp_path: Path) -> None:
    output = tmp_path / "cli"
    completed = subprocess.run(
        [
            sys.executable,
            str(
                REPOSITORY_ROOT
                / "scripts/results/build_full_model_compute_cost_table.py"
            ),
            "--output-dir",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(REPOSITORY_ROOT)},
    )
    summary = json.loads(completed.stdout)
    assert summary == {
        "analysis": "full_model_compute_cost",
        "outputs": {
            "display": "full_model_compute_cost_table.csv",
            "markdown": "full_model_compute_cost_table.md",
            "raw": "full_model_compute_cost.csv",
        },
        "rows": 5,
    }
    assert json.loads((output / "run_summary.json").read_text()) == summary
    assert all(not Path(name).is_absolute() for name in summary["outputs"].values())
