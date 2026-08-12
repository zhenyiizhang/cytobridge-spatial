import math
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from CytoBridge.tl.downstream import simulation
from CytoBridge.tl.downstream.simulation import _euler_sdeint_split


class _UnitDriftNoGrowthSDE:
    def f(self, _t, state):
        z, lnw = state
        return torch.ones_like(z), torch.zeros_like(lnw)

    def g(self, _t, z):
        return torch.zeros_like(z)


class _ConstantGrowthSDE:
    def f(self, _t, state):
        z, lnw = state
        return torch.zeros_like(z), torch.ones_like(lnw)

    def g(self, _t, z):
        return torch.zeros_like(z)


def _initial_state():
    z = torch.tensor([[0.0], [2.0]], dtype=torch.float32)
    lnw = torch.full((2, 1), math.log(0.5), dtype=torch.float32)
    return z, lnw


def test_euler_split_records_initial_state_before_first_step():
    initial = _initial_state()
    states, _ = _euler_sdeint_split(
        _UnitDriftNoGrowthSDE(),
        initial,
        dt=0.1,
        ts=torch.tensor([0.0, 0.5, 1.0]),
        noise_std=0.0,
    )

    torch.testing.assert_close(states[0], initial[0])
    torch.testing.assert_close(states[1], initial[0] + 0.5)
    torch.testing.assert_close(states[2], initial[0] + 1.0)
    assert [state.shape[0] for state in states] == [2, 2, 2]


def test_euler_split_piecewise_shared_boundary_is_exact():
    initial = _initial_state()
    first_states, first_weights = _euler_sdeint_split(
        _UnitDriftNoGrowthSDE(),
        initial,
        dt=0.1,
        ts=torch.tensor([0.0, 0.5]),
        noise_std=0.0,
    )
    second_initial = (first_states[-1], first_weights[-1])
    second_states, _ = _euler_sdeint_split(
        _UnitDriftNoGrowthSDE(),
        second_initial,
        dt=0.1,
        ts=torch.tensor([0.5, 1.0]),
        noise_std=0.0,
    )

    torch.testing.assert_close(second_states[0], first_states[-1])
    torch.testing.assert_close(second_states[-1], initial[0] + 1.0)


def test_fixed_split_event_grid_is_output_grid_invariant():
    z = torch.arange(128, dtype=torch.float32).reshape(-1, 1)
    initial = (
        z,
        torch.full((len(z), 1), math.log(1.0 / len(z)), dtype=torch.float32),
    )
    torch.manual_seed(42)
    coarse, coarse_weights = _euler_sdeint_split(
        _ConstantGrowthSDE(),
        initial,
        dt=0.05,
        ts=torch.tensor([0.0, 0.5, 1.0]),
        noise_std=0.0,
        resample_dt=0.1,
    )
    torch.manual_seed(42)
    dense, dense_weights = _euler_sdeint_split(
        _ConstantGrowthSDE(),
        initial,
        dt=0.05,
        ts=torch.tensor([value / 10 for value in range(11)]),
        noise_std=0.0,
        resample_dt=0.1,
    )

    torch.testing.assert_close(coarse[0], dense[0])
    torch.testing.assert_close(coarse[1], dense[5])
    torch.testing.assert_close(coarse[2], dense[10])
    torch.testing.assert_close(coarse_weights[1], dense_weights[5])
    torch.testing.assert_close(coarse_weights[2], dense_weights[10])


def test_split_integrator_rejects_empty_nonfinite_and_invalid_grids():
    initial = _initial_state()
    with pytest.raises(ValueError, match="at least one"):
        _euler_sdeint_split(
            _UnitDriftNoGrowthSDE(), initial, dt=0.1, ts=torch.tensor([])
        )
    with pytest.raises(ValueError, match="finite"):
        _euler_sdeint_split(
            _UnitDriftNoGrowthSDE(),
            initial,
            dt=float("nan"),
            ts=torch.tensor([0.0, 1.0]),
        )
    with pytest.raises(ValueError, match="resample_dt must be finite"):
        _euler_sdeint_split(
            _UnitDriftNoGrowthSDE(),
            initial,
            dt=0.1,
            ts=torch.tensor([0.0, 1.0]),
            resample_dt=float("inf"),
        )


def test_split_integrator_particle_limit_fails_before_repeat_allocation():
    with pytest.raises(RuntimeError, match="particle limit exceeded"):
        _euler_sdeint_split(
            _ConstantGrowthSDE(),
            _initial_state(),
            dt=0.1,
            ts=torch.tensor([0.0, 1.0]),
            resample_dt=1.0,
            max_particles=1,
        )


def test_split_particle_count_encodes_equal_mass_after_resampling():
    z = torch.tensor([[0.0], [1.0]], dtype=torch.float32)
    previous_weights = torch.full((2, 1), 0.5, dtype=torch.float32)
    grown_lnw = torch.zeros((2, 1), dtype=torch.float32)  # mass doubled per source

    (resampled_z, resampled_lnw), weights = simulation._apply_split_event(
        (z, grown_lnw),
        previous_weights=previous_weights,
        initial_count=2,
        noise_std=0.0,
    )

    assert resampled_z.shape[0] == 4
    np.testing.assert_allclose(torch.exp(resampled_lnw).numpy(), 0.5)
    np.testing.assert_allclose(weights.numpy(), 0.5)
    assert weights.sum().item() == pytest.approx(2.0)


class _ConstantGrowthModel:
    components = ("velocity", "growth")
    interaction_net = None

    @staticmethod
    def predict_velocity(*, t, x):
        del t
        return torch.zeros_like(x)

    @staticmethod
    def predict_growth(*, t, x):
        del t
        return torch.full((x.shape[0], 1), 4.0, dtype=x.dtype, device=x.device)


def test_anndata_split_simulation_applies_growth_alpha(monkeypatch):
    recorded_growth = []

    def inspect_first_derivative(sde, initial_state, **_kwargs):
        _, dlnw = sde.f(torch.tensor([0.0]), initial_state)
        recorded_growth.append(dlnw.detach().cpu().numpy())
        return [initial_state[0]], [initial_state[1]]

    monkeypatch.setattr(simulation, "_euler_sdeint_split", inspect_first_derivative)
    adata = SimpleNamespace(
        obs=pd.DataFrame({"time": [0.0, 0.0]}),
        obsm={"X_latent": np.asarray([[0.0], [1.0]], dtype=np.float32)},
        n_obs=2,
    )
    common = dict(
        adata=adata,
        model=_ConstantGrowthModel(),
        time_key="time",
        obsm_key="X_latent",
        concat_spatial=False,
        ts_points=[0.0],
        device="cpu",
    )

    simulation.simulate_sde_points_split(**common, growth_alpha=0.25)
    simulation.simulate_sde_points_split(**common)

    np.testing.assert_allclose(recorded_growth[0], 1.0)
    np.testing.assert_allclose(recorded_growth[1], 4.0)


def test_visualization_only_warp_uses_one_global_model_trajectory(monkeypatch):
    segment_starts = []

    def fake_simulate(*, x0, ts_points, **_kwargs):
        start = np.asarray(x0, dtype=np.float32).copy()
        segment_starts.append(start)
        return np.array([start.copy() for _ in ts_points], dtype=object)

    def fake_sample(
        _df,
        *,
        time_value,
        feature_cols,
        label_col,
        n_samples_cap,
        rng,
    ):
        del feature_cols, label_col, rng
        xy = np.asarray([float(time_value), 2.0 * float(time_value)], dtype=np.float32)
        return (
            np.column_stack(
                (
                    np.repeat(xy[0], n_samples_cap),
                    np.repeat(xy[1], n_samples_cap),
                    np.zeros(n_samples_cap, dtype=np.float32),
                )
            ).astype(np.float32),
            np.asarray(["A"] * n_samples_cap),
        )

    monkeypatch.setattr(
        simulation,
        "simulate_sde_points_split_from_x0",
        fake_simulate,
    )
    monkeypatch.setattr(simulation, "sample_observed_x0", fake_sample)

    warped, prewarp = simulation.simulate_piecewise_spatially_warped_split(
        x0=np.zeros((2, 3), dtype=np.float32),
        f_net=None,
        score_net=None,
        observed_time_points=[0.0, 1.0, 2.0],
        ts_points=[0.0, 0.5, 1.0, 1.5, 2.0],
        df=pd.DataFrame({"samples": [0.0, 0.0, 1.0, 1.0, 2.0, 2.0]}),
        feature_cols_full=["x1", "x2", "x3"],
        label_col="Annotation",
        dt=0.1,
        sigma=0.03,
        sigma_by_dim=None,
        growth_alpha=1.0,
        interaction_m=4,
        device="cpu",
        rng=np.random.default_rng(0),
        k=1,
        eps=1e-6,
        return_prewarp=True,
        warp_visualization_only=True,
    )

    assert len(segment_starts) == 1
    np.testing.assert_array_equal(segment_starts[0], prewarp[0])
    np.testing.assert_array_equal(warped[0], prewarp[0])
    np.testing.assert_array_equal(warped[2], prewarp[2])
    np.testing.assert_array_equal(warped[4], prewarp[4])
    np.testing.assert_array_equal(
        np.asarray(warped[1])[:, :2],
        np.asarray([[0.5, 1.0], [0.5, 1.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        np.asarray(warped[3])[:, :2],
        np.asarray([[1.5, 3.0], [1.5, 3.0]], dtype=np.float32),
    )
    assert not np.array_equal(warped[1], prewarp[1])


def test_spatial_warp_breaks_exact_anchor_ties_by_anchor_index() -> None:
    displacement = simulation._compute_spatial_warp_displacements(
        np.asarray([[0.0, 0.0]], dtype=np.float32),
        np.asarray([[-1.0, 0.0], [1.0, 0.0]], dtype=np.float32),
        np.asarray([[-2.0, 0.0], [3.0, 0.0]], dtype=np.float32),
        k=1,
        eps=1e-6,
    )

    np.testing.assert_allclose(displacement, [[-1.0, 0.0]])
    with pytest.raises(ValueError, match="eps must be finite and positive"):
        simulation._compute_spatial_warp_displacements(
            np.asarray([[0.0, 0.0]], dtype=np.float32),
            np.asarray([[-1.0, 0.0]], dtype=np.float32),
            np.asarray([[-2.0, 0.0]], dtype=np.float32),
            k=1,
            eps=0.0,
        )
