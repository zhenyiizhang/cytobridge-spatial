from __future__ import annotations

import json
import hashlib

import numpy as np
import pytest
import torch
from torch import nn

from CytoBridge.tl.downstream.functional_ablation import (
    run_frozen_checkpoint_ablations,
    save_frozen_checkpoint_ablation_result,
    temporary_lr_gate_mode,
)
from CytoBridge.tl.downstream.checkpoint import LoadedModel
from CytoBridge.tl.downstream.perturbation import FixedCohortRollout


class _ModernInteraction(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.link_predictor = nn.Linear(2, 1)
        self.edge_prior_mode = "learned"
        self.edge_predictor_thre = 0.63


class _LegacyInteraction(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.link_predictor = nn.Linear(2, 1)
        self.edge_predictor_thre = 0.45


class _FullModel(nn.Module):
    def __init__(self, interaction_net: nn.Module | None = None) -> None:
        super().__init__()
        self.components = ["velocity", "growth", "score", "interaction"]
        self.interaction_net = interaction_net or _ModernInteraction()
        self.sentinel_weight = nn.Parameter(torch.tensor([2.0]))


class _ToyGateInteraction(nn.Module):
    requires_time = True

    def __init__(self) -> None:
        super().__init__()
        self.link_predictor = nn.Linear(2, 1)
        self.edge_prior_mode = "learned"
        self.edge_predictor_thre = 0.63

    def forward(self, x, _lnw, _t):
        scale = 1.0 if self.edge_prior_mode == "learned" else 2.0
        return torch.full_like(x, scale)


class _ToyFunctionalModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.components = ["velocity", "score", "interaction"]
        self.interaction_net = _ToyGateInteraction()
        self.use_growth_in_ode_inter = True

    def predict_velocity(self, *, t, x):
        del t
        return torch.zeros_like(x)

    def compute_score(self, *, t, x, create_graph=False):
        del t, create_graph
        score = (x * 0.0).sum(dim=1, keepdim=True)
        return score, torch.zeros_like(x)


def test_temporary_lr_gate_mode_restores_modern_model_on_error() -> None:
    model = _FullModel()

    with pytest.raises(RuntimeError, match="stop"):
        with temporary_lr_gate_mode(model, "all_spatial") as record:
            assert model.interaction_net.edge_prior_mode == "all_spatial"
            assert record.implementation == "edge_prior_mode=all_spatial"
            raise RuntimeError("stop")

    assert model.interaction_net.edge_prior_mode == "learned"
    assert model.interaction_net.edge_predictor_thre == pytest.approx(0.63)


def test_temporary_lr_gate_mode_uses_zero_threshold_for_legacy_model() -> None:
    model = _FullModel(_LegacyInteraction())

    with temporary_lr_gate_mode(model, "all_spatial") as record:
        assert model.interaction_net.edge_predictor_thre == pytest.approx(0.0)
        assert record.implementation == "legacy_edge_predictor_threshold=0"

    assert model.interaction_net.edge_predictor_thre == pytest.approx(0.45)


def test_frozen_ablation_runner_matches_model_start_seed_and_time_grid(
    monkeypatch,
) -> None:
    import CytoBridge.tl.downstream.functional_ablation as module

    model = _FullModel()
    initial = np.arange(24, dtype=np.float32).reshape(6, 4)
    calls: list[dict[str, object]] = []

    def fake_rollout(
        points,
        live_model,
        *,
        start_time,
        end_time,
        dt,
        interaction_m,
        grouping_seed,
        device,
        spatial_dim,
        interaction_enabled,
        sigma,
    ):
        np.random.random()
        torch.rand(1)
        mode = live_model.interaction_net.edge_prior_mode
        calls.append(
            {
                "points": points.copy(),
                "model_id": id(live_model),
                "mode": mode,
                "seed": grouping_seed,
                "interaction_enabled": interaction_enabled,
            }
        )
        offset = {
            ("learned", True): 1.0,
            ("learned", False): 2.0,
            ("all_spatial", True): 3.0,
        }[(mode, bool(interaction_enabled))]
        return FixedCohortRollout(
            times=np.asarray([start_time, end_time], dtype=np.float64),
            points=np.stack((points, points + offset)).astype(np.float32),
            interaction_enabled=bool(interaction_enabled),
            grouping_seed=int(grouping_seed),
            sigma=float(sigma),
        )

    monkeypatch.setattr(module, "deterministic_fixed_cohort_rollout", fake_rollout)
    weight_before = model.sentinel_weight.detach().clone()
    numpy_state_before = np.random.get_state()
    torch_state_before = torch.random.get_rng_state()

    result = run_frozen_checkpoint_ablations(
        initial,
        model,
        start_time=0.0,
        end_time=1.0,
        dt=0.1,
        interaction_m=4,
        grouping_seed=17,
        device="cpu",
        spatial_dim=2,
    )

    assert [call["mode"] for call in calls] == [
        "learned",
        "learned",
        "all_spatial",
    ]
    assert [call["interaction_enabled"] for call in calls] == [
        True,
        False,
        True,
    ]
    assert {call["seed"] for call in calls} == {17}
    assert {call["model_id"] for call in calls} == {id(model)}
    for call in calls:
        np.testing.assert_array_equal(call["points"], initial)
    for rollout in result.rollouts.values():
        np.testing.assert_array_equal(rollout.points[0], initial)
    assert model.interaction_net.edge_prior_mode == "learned"
    torch.testing.assert_close(model.sentinel_weight, weight_before)
    assert np.random.get_state()[0] == numpy_state_before[0]
    np.testing.assert_array_equal(np.random.get_state()[1], numpy_state_before[1])
    torch.testing.assert_close(torch.random.get_rng_state(), torch_state_before)

    matched = result.manifest["matched_controls"]
    assert matched["same_checkpoint_weights"] is True
    assert matched["same_grouping_seed"] is True
    assert result.manifest["rollout"]["growth_or_particle_resampling"] is False
    assert result.manifest["conditions"]["lr_gate_off"]["weights_retrained"] is False


def test_frozen_ablation_actual_fixed_cohort_smoke() -> None:
    model = _ToyFunctionalModel()
    initial = np.zeros((4, 3), dtype=np.float32)

    result = run_frozen_checkpoint_ablations(
        initial,
        model,
        start_time=0.0,
        end_time=0.2,
        dt=0.1,
        interaction_m=4,
        grouping_seed=7,
        spatial_dim=2,
    )

    np.testing.assert_allclose(result.rollouts["full"].points[-1], 0.2)
    np.testing.assert_allclose(
        result.rollouts["interaction_off"].points[-1],
        0.0,
    )
    np.testing.assert_allclose(result.rollouts["lr_gate_off"].points[-1], 0.4)
    assert model.interaction_net.edge_prior_mode == "learned"


def test_trained_control_rejects_separately_built_all_spatial_model() -> None:
    model = _FullModel()
    model.interaction_net.edge_prior_mode = "all_spatial"

    with pytest.raises(ValueError, match="already configured"):
        with temporary_lr_gate_mode(model, "trained"):
            pass


def test_save_frozen_checkpoint_ablation_result(tmp_path, monkeypatch) -> None:
    import CytoBridge.tl.downstream.functional_ablation as module

    model = _FullModel()
    initial = np.zeros((3, 4), dtype=np.float32)

    def fake_rollout(points, _model, **kwargs):
        return FixedCohortRollout(
            times=np.asarray([kwargs["start_time"], kwargs["end_time"]]),
            points=np.stack((points, points)),
            interaction_enabled=bool(kwargs["interaction_enabled"]),
            grouping_seed=int(kwargs["grouping_seed"]),
            sigma=0.0,
        )

    monkeypatch.setattr(module, "deterministic_fixed_cohort_rollout", fake_rollout)
    result = run_frozen_checkpoint_ablations(
        initial,
        model,
        start_time=0.0,
        end_time=1.0,
        dt=0.5,
        interaction_m=3,
        grouping_seed=9,
    )
    paths = save_frozen_checkpoint_ablation_result(result, tmp_path)

    assert set(paths) == {"full", "interaction_off", "lr_gate_off", "manifest"}
    saved = np.load(paths["lr_gate_off"])
    np.testing.assert_array_equal(saved["points"][0], initial)
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    assert manifest["condition_order"] == [
        "full",
        "interaction_off",
        "lr_gate_off",
    ]


def test_loaded_checkpoint_file_hashes_are_recorded(tmp_path, monkeypatch) -> None:
    import CytoBridge.tl.downstream.functional_ablation as module

    weight_path = tmp_path / "model.pth"
    score_path = tmp_path / "score.pth"
    weight_path.write_bytes(b"fitted-full-model")
    score_path.write_bytes(b"fitted-score-model")
    model = _FullModel()
    loaded = LoadedModel(
        model=model,
        config={},
        weight_stage="Finetune",
        score_stage="Train_Score_Final",
        weight_path=weight_path,
        score_path=score_path,
    )

    def fake_rollout(points, _model, **kwargs):
        return FixedCohortRollout(
            times=np.asarray([kwargs["start_time"], kwargs["end_time"]]),
            points=np.stack((points, points)),
            interaction_enabled=bool(kwargs["interaction_enabled"]),
            grouping_seed=int(kwargs["grouping_seed"]),
            sigma=0.0,
        )

    monkeypatch.setattr(module, "deterministic_fixed_cohort_rollout", fake_rollout)
    result = run_frozen_checkpoint_ablations(
        np.zeros((3, 4), dtype=np.float32),
        loaded,
        start_time=0.0,
        end_time=1.0,
        dt=0.5,
        interaction_m=3,
        grouping_seed=9,
    )
    checkpoint = result.manifest["checkpoint"]
    assert checkpoint["weight_sha256"] == hashlib.sha256(
        b"fitted-full-model"
    ).hexdigest()
    assert checkpoint["score_sha256"] == hashlib.sha256(
        b"fitted-score-model"
    ).hexdigest()
