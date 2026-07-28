from __future__ import annotations

import json
import hashlib
import math

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from CytoBridge.tl.downstream.functional_ablation import (
    ContinuousFrozenCheckpointAblationResult,
    run_continuous_frozen_checkpoint_ablations,
    run_frozen_checkpoint_ablations,
    save_continuous_frozen_checkpoint_ablation_result,
    save_frozen_checkpoint_ablation_result,
    temporary_lr_gate_mode,
)
from CytoBridge.tl.downstream.checkpoint import LoadedModel
from CytoBridge.tl.downstream.evaluation import DistributionEvaluationResult
from CytoBridge.tl.downstream.perturbation import FixedCohortRollout
from CytoBridge.tl.downstream.simulation import _euler_sdeint, simulate_sde_points


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


class _ZeroTimeInteraction(nn.Module):
    requires_time = True

    def __init__(self) -> None:
        super().__init__()
        self.link_predictor = nn.Linear(2, 1)
        self.edge_prior_mode = "learned"
        self.edge_predictor_thre = 0.5

    def forward(self, x, _lnw, _t):
        return torch.zeros_like(x)


class _ToyContinuousModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.components = ["velocity", "growth", "score", "interaction"]
        self.interaction_net = _ZeroTimeInteraction()
        self.use_growth_in_ode_inter = True

    def predict_velocity(self, *, t, x):
        del t
        return torch.zeros_like(x)

    def predict_growth(self, *, t, x):
        del t
        return torch.full(
            (x.shape[0], 1),
            0.2,
            dtype=x.dtype,
            device=x.device,
        )

    def compute_score(self, *, t, x, create_graph=False):
        del t, create_graph
        score = (x * 0.0).sum(dim=1, keepdim=True)
        return score, torch.zeros_like(x)


class _UnitDriftGrowthSDE:
    def f(self, _t, state):
        z, lnw = state
        return torch.ones_like(z), torch.full_like(lnw, 0.5)

    def g(self, _t, z):
        return torch.zeros_like(z)


class _NoisyConstantSDE:
    def f(self, _t, state):
        z, lnw = state
        return torch.full_like(z, 0.2), torch.full_like(lnw, 0.1)

    def g(self, _t, z):
        return torch.full_like(z, 0.3)


def _legacy_aligned_euler(sde, initial_state, dt, ts):
    current_state = initial_state
    current_time = float(ts[0].item())
    final_time = float(ts[-1].item())
    requested = [float(value) for value in ts.tolist()]
    output_states = []
    next_output_idx = 0
    while current_time <= final_time + 1e-8:
        if current_time >= requested[next_output_idx] - 1e-8:
            output_states.append(current_state)
            next_output_idx += 1
            if next_output_idx >= len(requested):
                break
        t_tensor = torch.tensor(
            [current_time],
            dtype=torch.float32,
            device=current_state[0].device,
        )
        f_z, f_lnw = sde.f(t_tensor, current_state)
        noise = torch.randn_like(current_state[0]) * math.sqrt(dt)
        new_z = current_state[0] + f_z * dt + sde.g(
            t_tensor,
            current_state[0],
        ) * noise
        new_lnw = current_state[1] + f_lnw * dt
        current_state = (new_z, new_lnw)
        current_time += dt
    return (
        torch.stack([state[0] for state in output_states]),
        torch.stack([state[1] for state in output_states]),
    )


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


def test_simulate_sde_points_common_noise_is_independent_of_interaction_rng() -> None:
    ad = pytest.importorskip("anndata")
    model = _ToyContinuousModel()
    n_per_time = 6
    n_obs = 3 * n_per_time
    adata = ad.AnnData(
        X=np.zeros((n_obs, 2), dtype=np.float32),
        obs=pd.DataFrame(
            {
                "time_point_processed": np.repeat(
                    [0.0, 1.0, 2.0],
                    n_per_time,
                )
            }
        ),
    )
    adata.obsm["X_latent"] = np.arange(
        n_obs * 2,
        dtype=np.float32,
    ).reshape(n_obs, 2)
    adata.obsm["spatial_aligned"] = np.arange(
        n_obs * 2,
        dtype=np.float32,
    ).reshape(n_obs, 2) / 10.0

    kwargs = {
        "adata": adata,
        "model": model,
        "dim": 4,
        "time_index": 0,
        "n_samples": 4,
        "ts_points": [0.0, 1.0, 2.0],
        "dt": 0.25,
        "sigma": 0.1,
        "include_score": True,
        "interaction_m": 4,
        "noise_seed": 123,
        "device": "cpu",
        "time_key": "time_point_processed",
        "obsm_key": "X_latent",
        "spatial_key": "spatial_aligned",
        "concat_spatial": True,
        "verbose": False,
    }
    torch.manual_seed(71)
    full_points, full_weights = simulate_sde_points(
        **kwargs,
        include_interaction=True,
    )
    torch.manual_seed(71)
    off_points, off_weights = simulate_sde_points(
        **kwargs,
        include_interaction=False,
    )

    for full, off in zip(full_points, off_points):
        np.testing.assert_array_equal(full, off)
    np.testing.assert_array_equal(full_weights, off_weights)
    assert float(np.asarray(full_weights[-1]).sum()) > 1.0


def test_euler_sdeint_hits_nondivisible_requested_boundaries_exactly() -> None:
    initial = (
        torch.zeros((2, 1), dtype=torch.float32),
        torch.zeros((2, 1), dtype=torch.float32),
    )
    points, log_weights = _euler_sdeint(
        _UnitDriftGrowthSDE(),
        initial,
        dt=0.3,
        ts=torch.tensor([0.0, 0.5, 1.0], dtype=torch.float32),
    )

    torch.testing.assert_close(
        points[:, 0, 0],
        torch.tensor([0.0, 0.5, 1.0]),
    )
    torch.testing.assert_close(
        log_weights[:, 0, 0],
        torch.tensor([0.0, 0.25, 0.5]),
    )


def test_euler_sdeint_preserves_aligned_global_rng_path() -> None:
    initial = (
        torch.zeros((3, 2), dtype=torch.float32),
        torch.zeros((3, 1), dtype=torch.float32),
    )
    ts = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float32)
    torch.manual_seed(67)
    expected = _legacy_aligned_euler(
        _NoisyConstantSDE(),
        initial,
        dt=0.25,
        ts=ts,
    )
    torch.manual_seed(67)
    actual = _euler_sdeint(
        _NoisyConstantSDE(),
        initial,
        dt=0.25,
        ts=ts,
    )

    torch.testing.assert_close(actual[0], expected[0], rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual[1], expected[1], rtol=0.0, atol=0.0)


def test_continuous_frozen_runner_is_t0_to_all_targets_without_restart(
    monkeypatch,
) -> None:
    ad = pytest.importorskip("anndata")
    import CytoBridge.tl.downstream.functional_ablation as module

    model = _FullModel()
    n_per_time = 3
    adata = ad.AnnData(
        X=np.zeros((15, 2), dtype=np.float32),
        obs=pd.DataFrame(
            {
                "time_point_processed": np.repeat(
                    [0.0, 1.0, 2.0, 3.0, 4.0],
                    n_per_time,
                )
            }
        ),
    )
    adata.obsm["X_latent"] = np.zeros((15, 2), dtype=np.float32)
    adata.obsm["spatial_aligned"] = np.zeros((15, 2), dtype=np.float32)
    calls: list[dict[str, object]] = []

    def fake_evaluate(_adata, live_model, **kwargs):
        mode = live_model.interaction_net.edge_prior_mode
        calls.append(
            {
                "mode": mode,
                "include_interaction": kwargs["include_interaction"],
                "include_score": kwargs["include_score"],
                "noise_seed": kwargs["noise_seed"],
                "time_points": tuple(kwargs["time_points"]),
            }
        )
        times = tuple(float(value) for value in kwargs["time_points"])
        initial = np.arange(12, dtype=np.float32).reshape(3, 4)
        offset = {
            ("learned", True): 1.0,
            ("learned", False): 2.0,
            ("all_spatial", True): 3.0,
        }[(mode, bool(kwargs["include_interaction"]))]
        points = {
            time: initial.copy() if index == 0 else initial + offset * index
            for index, time in enumerate(times)
        }
        weights = {
            time: np.full(3, np.exp(0.1 * time) / 3.0, dtype=np.float64)
            for time in times
        }
        observed = {time: initial.copy() for time in times}
        return DistributionEvaluationResult(
            time_points=times,
            spatial_dim=2,
            predicted_points=points,
            predicted_weights=weights,
            observed_points=observed,
            metrics=pd.DataFrame(
                [
                    {
                        "time": time,
                        "space": "joint",
                        "w1": offset,
                        "w2": offset,
                        "tmv": 0.0,
                    }
                    for time in times[1:]
                ]
            ),
            settings={},
        )

    monkeypatch.setattr(module, "evaluate_model_distributions", fake_evaluate)
    result = run_continuous_frozen_checkpoint_ablations(
        adata,
        model,
        n_samples=3,
        dt=0.01,
        sigma=0.03,
        interaction_m=3,
        device="cpu",
        random_seed=19,
        verbose=False,
    )

    assert [call["mode"] for call in calls] == [
        "learned",
        "learned",
        "all_spatial",
    ]
    assert [call["include_interaction"] for call in calls] == [
        True,
        False,
        True,
    ]
    assert {call["include_score"] for call in calls} == {True}
    assert {call["noise_seed"] for call in calls} == {19}
    assert {call["time_points"] for call in calls} == {
        (0.0, 1.0, 2.0, 3.0, 4.0)
    }
    assert result.manifest["observed_intermediate_restart"] is False
    assert result.manifest["evaluation_times"] == [1.0, 2.0, 3.0, 4.0]
    assert result.manifest["rollout"]["include_growth"] is True
    assert result.manifest["matched_controls"]["common_brownian_noise"] is True
    assert model.interaction_net.edge_prior_mode == "learned"


def test_continuous_runner_rejects_late_restart(monkeypatch) -> None:
    ad = pytest.importorskip("anndata")
    model = _FullModel()
    adata = ad.AnnData(
        X=np.zeros((6, 2), dtype=np.float32),
        obs=pd.DataFrame(
            {"time_point_processed": np.repeat([0.0, 1.0, 2.0], 2)}
        ),
    )
    adata.obsm["X_latent"] = np.zeros((6, 2), dtype=np.float32)
    adata.obsm["spatial_aligned"] = np.zeros((6, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="earliest observed time"):
        run_continuous_frozen_checkpoint_ablations(
            adata,
            model,
            time_points=[1.0, 2.0],
            n_samples=2,
            interaction_m=2,
            device="cpu",
            verbose=False,
        )


def test_continuous_runner_rejects_incomplete_observed_grid() -> None:
    ad = pytest.importorskip("anndata")
    model = _FullModel()
    adata = ad.AnnData(
        X=np.zeros((6, 2), dtype=np.float32),
        obs=pd.DataFrame(
            {"time_point_processed": np.repeat([0.0, 1.0, 2.0], 2)}
        ),
    )
    adata.obsm["X_latent"] = np.zeros((6, 2), dtype=np.float32)
    adata.obsm["spatial_aligned"] = np.zeros((6, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="complete sorted observed time grid"):
        run_continuous_frozen_checkpoint_ablations(
            adata,
            model,
            time_points=[0.0, 2.0],
            n_samples=2,
            interaction_m=2,
            device="cpu",
            verbose=False,
        )


def test_save_continuous_frozen_checkpoint_ablation_result(tmp_path) -> None:
    times = (0.0, 1.0, 2.0)
    evaluations = {}
    for name, offset in {"full": 0.0, "interaction_off": 1.0}.items():
        predicted = {
            time: np.full((3, 4), offset + time, dtype=np.float32)
            for time in times
        }
        weights = {
            time: np.full(3, np.exp(time) / 3.0, dtype=np.float64)
            for time in times
        }
        evaluations[name] = DistributionEvaluationResult(
            time_points=times,
            spatial_dim=2,
            predicted_points=predicted,
            predicted_weights=weights,
            observed_points={
                time: np.zeros((2 + int(time), 4), dtype=np.float32)
                for time in times
            },
            metrics=pd.DataFrame(
                [{"time": 1.0, "space": "joint", "w1": offset, "w2": offset}]
            ),
            settings={},
        )
    result = ContinuousFrozenCheckpointAblationResult(
        evaluations=evaluations,
        manifest={"analysis": "test"},
    )

    paths = save_continuous_frozen_checkpoint_ablation_result(
        result,
        tmp_path,
    )

    saved = np.load(paths["full_trajectory"])
    np.testing.assert_array_equal(saved["times"], times)
    assert saved["points"].shape == (3, 3, 4)
    assert saved["weights"].shape == (3, 3)
    combined = pd.read_csv(paths["metrics"])
    assert set(combined["condition"]) == {"full", "interaction_off"}
    observed = np.load(paths["observed"])
    assert set(observed.files) == {
        "times",
        "observed_0",
        "observed_1",
        "observed_2",
    }
    assert observed["observed_0"].shape == (2, 4)
    assert observed["observed_2"].shape == (4, 4)
