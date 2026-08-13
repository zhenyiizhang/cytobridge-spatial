from __future__ import annotations

import csv
import multiprocessing
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from scripts.spatiotemporal_benchmark import run_unified_benchmark as runner


def _merge_one_status(path, method):
    runner.merge_status_rows(
        path,
        [
            {
                "track": "loto",
                "target": 1,
                "method": method,
                "status": "completed",
                "reason": "",
                "elapsed_seconds": 1.0,
            }
        ],
    )


def test_dataset_matrix_has_the_complete_target_plan():
    configs = runner.load_datasets(list(runner.DATASETS))
    assert configs["zebrafish"]["loto_targets"] == [1, 2, 3]
    assert configs["zebrafish"]["full_data_targets"] == [1, 2, 3, 4]
    assert configs["mosta"]["loto_targets"] == [1, 2]
    assert configs["mosta"]["full_data_targets"] == [1, 2, 3]
    assert configs["arista"]["full_data_targets"] == [1, 2, 3, 4]
    assert configs["admouse"]["loto_targets"] == [1]
    assert configs["admouse"]["full_data_targets"] == [1, 2]


def test_cytobridge_loto_preparation_receives_the_training_profile(tmp_path):
    cfg = runner.load_datasets(["admouse"])["admouse"]
    commands = runner.cytobridge_commands(
        Path("python"),
        cfg,
        tmp_path / "formal/admouse",
        tmp_path / "benchmark/admouse/inputs/manifest.json",
        tmp_path / "benchmark/admouse",
        "loto_t1",
        [1],
        "cpu",
    )
    prepare = commands[0]
    self_config = tmp_path / "formal/admouse/training/config.yaml"
    assert "prepare-loto" in prepare
    assert prepare[prepare.index("--training-config") + 1] == str(self_config)
    assert "--database" in prepare

    learned_cfg = runner.load_datasets(["zebrafish"])["zebrafish"]
    learned = runner.cytobridge_commands(
        Path("python"),
        learned_cfg,
        tmp_path / "formal/zebrafish",
        tmp_path / "benchmark/zebrafish/inputs/manifest.json",
        tmp_path / "benchmark/zebrafish",
        "loto_t1",
        [1],
        "cpu",
    )[0]
    assert "--database" in learned

    radius_cfg = dict(cfg)
    radius_cfg["benchmark"] = dict(cfg["benchmark"], edge_prior_mode="all_spatial")
    radius = runner.cytobridge_commands(
        Path("python"),
        radius_cfg,
        tmp_path / "formal/admouse",
        tmp_path / "benchmark/admouse/inputs/manifest.json",
        tmp_path / "benchmark/admouse",
        "loto_t1",
        [1],
        "cpu",
    )[0]
    assert "--database" not in radius


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
        ("full_data", "1"),
        ("full_data", "2"),
        ("loto", "1"),
    ]
    assert {row["status"] for row in rows} == {"not_applicable"}


def test_primary_evaluation_uses_the_shared_registry_and_excludes_sensitivity_only(
    tmp_path, capsys
):
    runner.main(
        [
            "--datasets",
            "admouse",
            "--run-root",
            str(tmp_path),
            "--dry-run",
            "evaluate",
            "--tracks",
            "loto",
        ]
    )
    output = capsys.readouterr().out
    assert "--method-registry" in output
    assert str(runner.METHOD_REGISTRY) in output
    assert "CytoBridge-0.015" in output
    assert "random_independent_pairs" in output
    assert "spatrack" not in output.lower()


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


def test_partial_full_run_keeps_completed_target_and_failed_later_target(
    tmp_path, monkeypatch
):
    cfg = runner.load_datasets(["admouse"])["admouse"]
    args = SimpleNamespace(
        run_root=tmp_path,
        formal_root=tmp_path / "formal",
        methods=["stories"],
        tracks=["full_data"],
        software_root=tmp_path / "software",
        device="cpu",
        timeout=1,
        dry_run=False,
    )
    completed = tmp_path / "admouse/predictions/full_data/stories/t1"
    completed.mkdir(parents=True)
    (completed / "prediction.npz").write_bytes(b"prediction")
    (completed / "summary.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        runner, "execute", lambda *unused: ("timeout", "timeout after t1")
    )

    runner.run_dataset("admouse", cfg, args, {}, {"stories": tmp_path})

    rows = list(
        csv.DictReader(
            (tmp_path / "admouse/status/method_target_status.csv").open(
                newline="", encoding="utf-8"
            )
        )
    )
    assert [(row["target"], row["status"]) for row in rows] == [
        ("1", "completed"),
        ("2", "timeout"),
    ]


def test_status_updates_merge_instead_of_erasing_other_methods(tmp_path):
    path = tmp_path / "status.csv"
    runner.merge_status_rows(
        path,
        [
            {
                "track": "loto",
                "target": 1,
                "method": "stvcr",
                "status": "completed",
                "reason": "",
                "elapsed_seconds": 2.0,
            }
        ],
    )
    runner.merge_status_rows(
        path,
        [
            {
                "track": "loto",
                "target": 1,
                "method": "stories",
                "status": "timeout",
                "reason": "budget",
                "elapsed_seconds": 3600.0,
            }
        ],
    )
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert {(row["method"], row["status"]) for row in rows} == {
        ("stvcr", "completed"),
        ("stories", "timeout"),
    }


def test_parallel_status_updates_do_not_drop_methods(tmp_path):
    path = tmp_path / "status.csv"
    processes = [
        multiprocessing.Process(target=_merge_one_status, args=(path, method))
        for method in ("stvcr", "stories", "mioflow", "moscot")
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=10)
        assert process.exitcode == 0
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["method"] for row in rows} == {
        "stvcr",
        "stories",
        "mioflow",
        "moscot",
    }
