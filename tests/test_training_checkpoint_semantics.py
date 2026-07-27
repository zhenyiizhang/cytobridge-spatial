from __future__ import annotations

import types
from pathlib import Path

import pandas as pd
import torch
import yaml

import CytoBridge.tl.train.trainer as trainer_module
from CytoBridge.tl.train.trainer import TrainingPipeline


class _ScalarModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([0.0]))


class _Logger:
    @staticmethod
    def info(_message: str) -> None:
        return None


def _bare_pipeline(tmp_path) -> TrainingPipeline:
    pipeline = object.__new__(TrainingPipeline)
    pipeline.model = _ScalarModel()
    pipeline.config = {"ckpt_dir": str(tmp_path)}
    pipeline.logger = _Logger()
    pipeline.scheduler = None
    pipeline.ode_func = None
    return pipeline


def test_last_checkpoint_does_not_add_an_optimizer_epoch(tmp_path) -> None:
    pipeline = _bare_pipeline(tmp_path)
    calls = []

    def fake_epoch(self, stage_params, data, time_points, ode_func):
        calls.append(len(calls) + 1)
        with torch.no_grad():
            self.model.weight.add_(1.0)
        self._last_neural_ode_epoch = {
            "forward_last_ot": float(calls[-1]) / 10.0,
            "state_after_forward": None,
            "mean_ot_loss": float(calls[-1]) / 20.0,
            "mean_mass_loss": 0.0,
            "mean_energy_loss": 0.0,
            "mean_density_loss": 0.0,
            "mean_pinn_loss": 0.0,
            "n_intervals": 4,
        }
        return float(calls[-1])

    pipeline.train_neural_ode_epoch = types.MethodType(fake_epoch, pipeline)
    pipeline.run_neural_ode_stage(
        {
            "name": "Finetune",
            "epochs": 2,
            "save_strategy": "last",
            "train_strategy": "v",
        },
        data=[],
        time_points=[],
    )

    assert calls == [1, 2]
    assert pipeline.model.weight.item() == 2.0
    state = torch.load(tmp_path / "Finetune" / "last_model.pth", weights_only=True)
    assert state["weight"].item() == 2.0
    history = pipeline.training_history_frame()
    assert history["epoch"].tolist() == [1, 2]
    assert history["loss"].tolist() == [1.0, 2.0]
    assert history["checkpoint_metric"].tolist() == [
        "average_loss",
        "average_loss",
    ]
    assert history["forward_last_ot"].tolist() == [0.1, 0.2]
    assert history["mean_ot_loss"].tolist() == [0.05, 0.1]
    assert history["n_intervals"].tolist() == [4.0, 4.0]
    pipeline._save_training_history()
    saved_history = pd.read_csv(tmp_path / "training_history.csv")
    assert saved_history["epoch"].tolist() == [1, 2]


def test_legacy_checkpoint_uses_forward_snapshot(tmp_path) -> None:
    pipeline = _bare_pipeline(tmp_path)
    forward_metrics = iter([3.0, 1.0, 2.0])

    def fake_epoch(self, stage_params, data, time_points, ode_func):
        with torch.no_grad():
            # Represent the state immediately after the forward pass.
            self.model.weight.add_(1.0)
        forward_state = {"weight": self.model.weight.detach().clone()}
        metric = next(forward_metrics)
        # Represent additional reverse-pass updates that must not enter the
        # legacy intermediate-stage checkpoint.
        with torch.no_grad():
            self.model.weight.add_(10.0)
        self._last_neural_ode_epoch = {
            "forward_last_ot": metric,
            "state_after_forward": forward_state,
        }
        return metric + 10.0

    pipeline.train_neural_ode_epoch = types.MethodType(fake_epoch, pipeline)
    pipeline.run_neural_ode_stage(
        {
            "name": "Refine",
            "epochs": 3,
            "save_strategy": "best",
            "checkpoint_metric": "legacy_forward_last_ot",
            "train_strategy": "v",
        },
        data=[],
        time_points=[],
    )

    # Forward snapshots were 1, 12, and 23; the lowest forward OT metric was
    # the second epoch, so the selected state is 12 rather than a reverse state.
    assert pipeline.model.weight.item() == 12.0
    state = torch.load(tmp_path / "Refine" / "best_model.pth", weights_only=True)
    assert state["weight"].item() == 12.0


class _ScoreModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.score_net = torch.nn.Linear(1, 1, bias=False)

    def compute_score(self, *, t, x, create_graph):
        value = self.score_net(x)
        return value, value


def test_score_matching_records_total_score_and_penalty_each_epoch(
    monkeypatch, tmp_path
) -> None:
    class _FakeMatcher:
        def __init__(self, sigma):
            self.sigma = sigma

        @staticmethod
        def compute_lambda(values):
            return torch.ones_like(values)

    def fake_batch(_matcher, _x, _trajectory, batch_size, _time, return_noise):
        assert return_noise is True
        t = torch.zeros(batch_size)
        x = torch.ones(batch_size, 1)
        eps = torch.zeros(batch_size, 1)
        return t, x, None, eps

    monkeypatch.setattr(
        trainer_module, "SchrodingerBridgeConditionalFlowMatcher", _FakeMatcher
    )
    monkeypatch.setattr(trainer_module, "get_batch_size", fake_batch)

    pipeline = object.__new__(TrainingPipeline)
    pipeline.model = _ScoreModel()
    pipeline.config = {"ckpt_dir": str(tmp_path)}
    pipeline.device = torch.device("cpu")
    pipeline.optimizer = torch.optim.SGD(
        pipeline.model.score_net.parameters(), lr=0.05
    )
    pipeline.scheduler = None
    pipeline.logger = _Logger()
    pipeline.training_history = []
    pipeline._active_stage_index = 3
    pipeline._generate_score_trajectory = types.MethodType(
        lambda self, data, time_points: [], pipeline
    )

    pipeline.run_score_matching_stage(
        {
            "name": "Train_Score",
            "mode": "score_matching",
            "epochs": 2,
            "batch_size": 2,
            "sigma": 0.03,
            "lambda_penalty": 0.0,
            "save_strategy": "last",
        },
        data=[torch.zeros(2, 1), torch.ones(2, 1)],
        time_points=[0.0, 1.0],
    )

    history = pipeline.training_history_frame()
    assert history["epoch"].tolist() == [1, 2]
    assert history["stage_index"].tolist() == [3, 3]
    assert history["checkpoint_metric"].tolist() == ["total_loss", "total_loss"]
    assert history["loss"].tolist() == history["score_loss"].tolist()
    assert history["penalty"].tolist() == [0.0, 0.0]


def test_arista_full_finetune_uses_legacy_best_forward_ot() -> None:
    config_path = (
        Path(__file__).resolve().parents[1]
        / "CytoBridge"
        / "configs"
        / "arista_spatial_full.yaml"
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    stages = {
        stage["name"]: stage for stage in config["training"]["plan"]
    }
    finetune = stages["Finetune"]

    assert finetune["checkpoint_metric"] == "legacy_forward_last_ot"
    assert finetune["save_strategy"] == "best"
    assert finetune["scheduler_metric"] == "forward_last_ot"
    assert finetune["scheduler_step_before_reverse"] is True
