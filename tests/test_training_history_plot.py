from __future__ import annotations

import pandas as pd
import pytest

from CytoBridge.pl.training import (
    plot_training_history,
    summarize_training_history,
)


def _history() -> pd.DataFrame:
    return pd.DataFrame(
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


def test_training_history_summary_respects_best_and_last_checkpoint_rules():
    summary = summarize_training_history(_history()).set_index("stage")
    assert summary.loc["Pretrain", "selected_checkpoint_epoch"] == 2
    assert summary.loc["Pretrain", "selected_checkpoint_value"] == pytest.approx(1.5)
    assert summary.loc["Train_Score", "selected_checkpoint_epoch"] == 2
    assert summary.loc["Train_Score", "selected_checkpoint_value"] == pytest.approx(1.2)
    assert summary.loc["Train_Score", "save_strategy"] == "last"


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
