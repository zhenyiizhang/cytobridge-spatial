from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

from CytoBridge.pl.training import (
    plot_training_history,
    summarize_training_history,
)


def _history() -> pd.DataFrame:
    frame = pd.DataFrame(
        [
            {
                "stage_index": 0,
                "stage": "Pretrain",
                "mode": "neural_ode",
                "epoch": 1,
                "loss": 3.0,
                "checkpoint_metric": "forward_last_ot",
                "checkpoint_value": 2.5,
                "is_best": True,
                "learning_rate": 1e-4,
                "save_strategy": "best",
                "mean_ot_loss": 0.4,
            },
            {
                "stage_index": 0,
                "stage": "Pretrain",
                "mode": "neural_ode",
                "epoch": 2,
                "loss": 2.0,
                "checkpoint_metric": "forward_last_ot",
                "checkpoint_value": 1.5,
                "is_best": True,
                "learning_rate": 1e-4,
                "save_strategy": "best",
                "mean_ot_loss": 0.3,
            },
            {
                "stage_index": 0,
                "stage": "Pretrain",
                "mode": "neural_ode",
                "epoch": 3,
                "loss": 2.5,
                "checkpoint_metric": "forward_last_ot",
                "checkpoint_value": 2.0,
                "is_best": False,
                "learning_rate": 1e-4,
                "save_strategy": "best",
                "mean_ot_loss": 0.35,
            },
            {
                "stage_index": 1,
                "stage": "Train_Score",
                "mode": "score_matching",
                "epoch": 1,
                "loss": 1.0,
                "checkpoint_metric": "total_loss",
                "checkpoint_value": 1.0,
                "is_best": True,
                "learning_rate": 1e-4,
                "save_strategy": "last",
            },
            {
                "stage_index": 1,
                "stage": "Train_Score",
                "mode": "score_matching",
                "epoch": 2,
                "loss": 1.2,
                "checkpoint_metric": "total_loss",
                "checkpoint_value": 1.2,
                "is_best": False,
                "learning_rate": 1e-4,
                "save_strategy": "last",
            },
        ]
    )
    frame["is_selected_checkpoint"] = [False, True, False, False, True]
    frame["batch_size"] = [128, 128, 128, 64, 64]
    frame["epoch_wall_time_seconds"] = [1.0, 1.1, 1.2, 0.5, 0.6]
    frame["optimizer_steps_epoch"] = [3, 3, 3, 1, 1]
    frame["stage_wall_time_seconds"] = [3.5, 3.5, 3.5, 1.2, 1.2]
    frame["stage_learning_rate_start"] = [1e-4] * 5
    frame["stage_learning_rate_end"] = [5e-5, 5e-5, 5e-5, 1e-4, 1e-4]
    frame["stage_optimizer_steps"] = [9, 9, 9, 2, 2]
    return frame


def test_training_history_summary_respects_best_and_last_checkpoint_rules():
    summary = summarize_training_history(_history()).set_index("stage")
    assert summary.loc["Pretrain", "selected_checkpoint_epoch"] == 2
    assert summary.loc["Pretrain", "selected_checkpoint_value"] == pytest.approx(1.5)
    assert summary.loc["Train_Score", "selected_checkpoint_epoch"] == 2
    assert summary.loc["Train_Score", "selected_checkpoint_value"] == pytest.approx(1.2)
    assert summary.loc["Train_Score", "save_strategy"] == "last"
    assert summary.loc["Pretrain", "start_learning_rate"] == pytest.approx(1e-4)
    assert summary.loc["Pretrain", "end_learning_rate"] == pytest.approx(5e-5)
    assert summary.loc["Pretrain", "stage_wall_time_seconds"] == pytest.approx(3.5)
    assert summary.loc["Pretrain", "epoch_wall_time_seconds_sum"] == pytest.approx(3.3)
    assert summary.loc["Pretrain", "optimizer_step_count"] == 9
    assert summary.loc["Pretrain", "batch_size"] == 128
    assert summary.loc["Pretrain", "checkpoint_selection_source"] == (
        "explicit_history_flag"
    )
    assert summary.loc["Pretrain", "learning_rate_endpoint_scope"] == (
        "optimizer_state_before_and_after_stage"
    )


def test_schema_v1_history_infers_selected_checkpoint_without_rewriting_is_best():
    legacy = _history().drop(
        columns=[
            "is_selected_checkpoint",
            "batch_size",
            "epoch_wall_time_seconds",
            "optimizer_steps_epoch",
            "stage_wall_time_seconds",
            "stage_learning_rate_start",
            "stage_learning_rate_end",
            "stage_optimizer_steps",
        ]
    )
    summary = summarize_training_history(legacy).set_index("stage")
    assert summary.loc["Pretrain", "selected_checkpoint_epoch"] == 2
    assert summary.loc["Train_Score", "selected_checkpoint_epoch"] == 2
    assert summary.loc["Pretrain", "checkpoint_selection_source"].startswith(
        "inferred_from_legacy"
    )
    assert summary.loc["Pretrain", "learning_rate_endpoint_scope"].startswith(
        "first_and_last_recorded_epoch"
    )
    assert pd.isna(summary.loc["Pretrain", "optimizer_step_count"])


def test_explicit_unselected_history_does_not_fabricate_a_checkpoint():
    interrupted = _history().loc[lambda frame: frame["stage_index"] == 0].copy()
    interrupted["is_selected_checkpoint"] = False
    summary = summarize_training_history(interrupted).iloc[0]
    assert pd.isna(summary["selected_checkpoint_epoch"])
    assert pd.isna(summary["selected_checkpoint_value"])


def test_history_rejects_multiple_explicit_selected_checkpoints():
    invalid = _history()
    invalid.loc[invalid["stage_index"] == 0, "is_selected_checkpoint"] = [
        True,
        True,
        False,
    ]
    with pytest.raises(ValueError, match="multiple selected checkpoints"):
        summarize_training_history(invalid)


def test_training_history_report_carries_measured_resource_nulls(tmp_path):
    history_path = tmp_path / "training_history.csv"
    _history().to_csv(history_path, index=False)
    run_summary_path = tmp_path / "training_run_summary.json"
    run_summary_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "scope": "training only",
                "resources": {
                    "cpu_max_rss_mib": 100.0,
                    "cuda_peak_allocated_mib": None,
                    "cuda_peak_reserved_mib": None,
                },
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "report"
    script = Path(__file__).resolve().parents[1] / "scripts" / "summarize_training_history.py"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--history",
            str(history_path),
            "--run-summary",
            str(run_summary_path),
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    carried = json.loads(
        (output_dir / "training_resource_summary.json").read_text(encoding="utf-8")
    )
    assert carried["resources"]["cuda_peak_allocated_mib"] is None
    manifest = json.loads(
        (output_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["input"]["training_run_summary"]["sha256"]


def test_training_history_plot_writes_stage_separated_panel(tmp_path):
    output = plot_training_history(
        _history(),
        tmp_path / "training_history.png",
        title="Toy training",
    )
    assert output.exists()
    assert output.stat().st_size > 1000


def test_training_history_rejects_incomplete_schema():
    with pytest.raises(ValueError, match="missing columns"):
        summarize_training_history(pd.DataFrame({"stage": ["x"]}))
