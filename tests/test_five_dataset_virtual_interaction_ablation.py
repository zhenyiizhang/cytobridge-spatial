from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "run_five_dataset_virtual_interaction_ablation.py"
)
SPEC = importlib.util.spec_from_file_location(
    "run_five_dataset_virtual_interaction_ablation", SCRIPT
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_paired_displacement_metrics_separates_spatial_and_expression():
    on = (
        np.zeros((3, 4), dtype=np.float32),
        np.zeros((3, 4), dtype=np.float32),
    )
    off = (
        np.zeros((3, 4), dtype=np.float32),
        np.asarray(
            [[3.0, 4.0, 0.0, 0.0], [0.0, 0.0, 6.0, 8.0], [0.0, 0.0, 0.0, 0.0]],
            dtype=np.float32,
        ),
    )
    table = MODULE.paired_displacement_metrics(on, off, [0.0, 1.0])
    final = table.loc[table["time"] == 1.0].set_index("space")
    assert final.loc["spatial", "mean_displacement"] == pytest.approx(5.0 / 3.0)
    assert final.loc["expression", "mean_displacement"] == pytest.approx(10.0 / 3.0)
    assert final.loc["joint", "rms_displacement"] == pytest.approx(
        np.sqrt((25.0 + 100.0) / 3.0)
    )


def test_paired_displacement_rejects_population_or_time_mismatch():
    frame = np.zeros((3, 52), dtype=np.float32)
    with pytest.raises(ValueError, match="time grid"):
        MODULE.paired_displacement_metrics((frame,), (frame, frame), [0.0])
    with pytest.raises(ValueError, match="shape mismatch"):
        MODULE.paired_displacement_metrics(
            (frame,), (np.zeros((2, 52), dtype=np.float32),), [0.0]
        )


def test_coupled_distribution_metrics_has_zero_sampling_floor():
    points = np.arange(120, dtype=float).reshape(20, 6)
    table = MODULE.coupled_distribution_metrics(
        [points], [points.copy()], [0.0], spatial_dim=2, max_ot_points=5
    )
    assert table["w1"].eq(0.0).all()
    assert table["w2"].eq(0.0).all()
    assert table["ot_ablation_points"].eq(5).all()
    assert (
        table["ot_sampling"].eq("shared_paired_row_indices_without_replacement").all()
    )


def test_time_grid_uses_observed_and_interpolated_without_reanchoring():
    config = {
        "downstream": {
            "observed": [0.0, 1.0, 2.0],
            "interpolated": [0.5, 1.5],
        }
    }
    assert MODULE._time_grid(config) == (0.0, 0.5, 1.0, 1.5, 2.0)


def test_output_root_must_be_new_or_empty(tmp_path):
    output = tmp_path / "run"
    output.mkdir()
    (output / "stale.txt").write_text("stale", encoding="utf-8")
    with pytest.raises(FileExistsError, match="new or empty"):
        MODULE._require_empty(output)


def test_cli_declares_run_and_report_contracts(tmp_path):
    run_args = MODULE.parse_args(
        [
            "run",
            "--dataset",
            "zebrafish",
            "--aligned-h5ad",
            str(tmp_path / "a.h5ad"),
            "--expected-aligned-sha256",
            "0" * 64,
            "--model-dir",
            str(tmp_path / "model"),
            "--expected-training-summary-sha256",
            "1" * 64,
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )
    assert run_args.command == "run"
    assert run_args.dataset == "zebrafish"
    report_args = MODULE.parse_args(
        [
            "report",
            "--run-root",
            str(tmp_path / "root"),
            "--output-dir",
            str(tmp_path / "report"),
        ]
    )
    assert report_args.command == "report"
