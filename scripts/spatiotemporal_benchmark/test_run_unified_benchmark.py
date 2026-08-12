from __future__ import annotations

import csv
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from scripts.spatiotemporal_benchmark import run_unified_benchmark as runner


def test_dataset_matrix_has_the_complete_target_plan():
    configs = runner.load_datasets(list(runner.DATASETS))
    assert configs["zebrafish"]["loto_targets"] == [1, 2, 3]
    assert configs["zebrafish"]["full_data_targets"] == [1, 2, 3, 4]
    assert configs["mosta"]["loto_targets"] == [1, 2]
    assert configs["mosta"]["full_data_targets"] == [1, 2, 3]
    assert configs["arista"]["full_data_targets"] == [1, 2, 3, 4]
    assert configs["admouse"]["loto_targets"] == [1]
    assert configs["admouse"]["full_data_targets"] == [1, 2]


def test_mosta_full_dry_run_includes_t3(tmp_path, capsys):
    runner.main(
        [
            "--datasets",
            "mosta",
            "--formal-root",
            str(tmp_path / "formal"),
            "--run-root",
            str(tmp_path / "benchmark"),
            "--dry-run",
            "run",
            "--methods",
            "stvcr",
            "--tracks",
            "full_data",
        ]
    )
    output = capsys.readouterr().out
    assert "--target-time 1" in output
    assert "--target-time 2" in output
    assert "--target-time 3" in output


def test_not_applicable_rows_are_written_without_running_a_job(tmp_path):
    runner.main(
        [
            "--datasets",
            "admouse",
            "--run-root",
            str(tmp_path),
            "run",
            "--methods",
            "spatrack",
            "--tracks",
            "loto",
            "full_data",
        ]
    )
    path = tmp_path / "admouse" / "status" / "method_target_status.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [(row["track"], row["target"]) for row in rows] == [
        ("loto", "1"),
        ("full_data", "1"),
        ("full_data", "2"),
    ]
    assert {row["status"] for row in rows} == {"not_applicable"}


def test_execute_maps_missing_timeout_oom_and_failure(tmp_path):
    missing = tmp_path / "missing"
    assert (
        runner.execute([], [missing], 1, tmp_path / "missing.log")[0] == "not_available"
    )

    completed = SimpleNamespace(returncode=0, stdout="ok")
    with mock.patch.object(runner.subprocess, "run", return_value=completed):
        assert runner.execute([["job"]], [], 1, tmp_path / "ok.log")[0] == "completed"

    oom = SimpleNamespace(returncode=1, stdout="CUDA out of memory")
    with mock.patch.object(runner.subprocess, "run", return_value=oom):
        assert runner.execute([["job"]], [], 1, tmp_path / "oom.log")[0] == "oom"

    failed = SimpleNamespace(returncode=2, stdout="bad input")
    with mock.patch.object(runner.subprocess, "run", return_value=failed):
        assert runner.execute([["job"]], [], 1, tmp_path / "failed.log")[0] == "failed"

    with mock.patch.object(
        runner.subprocess,
        "run",
        side_effect=subprocess.TimeoutExpired("job", 1, output="slow"),
    ):
        assert (
            runner.execute([["job"]], [], 1, tmp_path / "timeout.log")[0] == "timeout"
        )
