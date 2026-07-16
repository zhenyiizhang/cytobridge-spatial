from __future__ import annotations

import types
from pathlib import Path

import torch
import yaml

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
