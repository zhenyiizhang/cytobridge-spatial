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


def test_split_integrator_tracks_inherited_lineages_and_daughter_noise():
    initial = _initial_state()
    lineage_ids = torch.tensor([10, 20], dtype=torch.long)

    torch.manual_seed(7)
    states_zero, _, lineages_zero = _euler_sdeint_split(
        _ConstantGrowthSDE(),
        initial,
        dt=1.0,
        ts=torch.tensor([0.0, 1.0]),
        noise_std=0.0,
        resample_dt=1.0,
        initial_lineage_ids=lineage_ids,
        return_lineage_ids=True,
    )
    torch.manual_seed(7)
    states_noisy, _, lineages_noisy = _euler_sdeint_split(
        _ConstantGrowthSDE(),
        initial,
        dt=1.0,
        ts=torch.tensor([0.0, 1.0]),
        noise_std=0.1,
        resample_dt=1.0,
        initial_lineage_ids=lineage_ids,
        return_lineage_ids=True,
    )

    torch.testing.assert_close(lineages_zero[0], lineage_ids)
    torch.testing.assert_close(lineages_zero[1], lineages_noisy[1])
    assert len(lineages_zero) == len(states_zero) == 2
    assert all(
        lineage.shape[0] == state.shape[0]
        for lineage, state in zip(lineages_zero, states_zero)
    )
    assert set(lineages_zero[1].tolist()) == {10, 20}
    assert states_zero[1].shape == states_noisy[1].shape
    assert not torch.allclose(states_zero[1], states_noisy[1])


def test_split_event_repeats_parent_lineage_and_drops_extinct_lineage():
    z = torch.tensor([[1.0], [9.0]], dtype=torch.float32)
    previous_weights = torch.full((2, 1), 0.5, dtype=torch.float32)
    next_weights = torch.tensor([[1.0], [5e-31]], dtype=torch.float32)
    lineage_ids = torch.tensor([101, 202], dtype=torch.long)

    (resampled_z, _), _, resampled_lineages = simulation._apply_split_event(
        (z, torch.log(next_weights)),
        previous_weights=previous_weights,
        initial_count=2,
        noise_std=0.0,
        lineage_ids=lineage_ids,
    )

    torch.testing.assert_close(resampled_z, torch.tensor([[1.0], [1.0]]))
    torch.testing.assert_close(resampled_lineages, torch.tensor([101, 101]))


@pytest.mark.parametrize("noise_std", [-0.1, float("nan"), float("inf")])
def test_split_integrator_rejects_invalid_daughter_noise(noise_std):
    with pytest.raises(ValueError, match="noise_std must be finite and >= 0"):
        _euler_sdeint_split(
            _UnitDriftNoGrowthSDE(),
            _initial_state(),
            dt=0.1,
            ts=torch.tensor([0.0, 1.0]),
            noise_std=noise_std,
        )


def test_split_from_x0_defaults_to_zero_daughter_noise_and_returns_lineages(
    monkeypatch,
):
    calls = []

    def fake_integrator(
        _sde,
        initial_state,
        *,
        noise_std,
        initial_lineage_ids,
        return_lineage_ids,
        **_kwargs,
    ):
        calls.append((noise_std, initial_lineage_ids, return_lineage_ids))
        states = [initial_state[0], initial_state[0]]
        weights = [initial_state[1], initial_state[1]]
        if not return_lineage_ids:
            return states, weights
        lineages = [initial_lineage_ids, initial_lineage_ids]
        return states, weights, lineages

    monkeypatch.setattr(simulation, "_euler_sdeint_split", fake_integrator)
    f_net = SimpleNamespace(
        v_net=object(),
        g_net=object(),
        interaction_net=object(),
    )
    score_net = object()
    common = dict(
        x0=np.asarray([[0.0], [1.0]], dtype=np.float32),
        f_net=f_net,
        score_net=score_net,
        ts_points=[0.0, 1.0],
        dt=0.1,
        sigma=0.03,
        sigma_by_dim=None,
        growth_alpha=1.0,
        interaction_m=2,
        device="cpu",
        verbose=False,
    )

    points = simulation.simulate_sde_points_split_from_x0(**common)
    points_with_ids, lineage_ids = simulation.simulate_sde_points_split_from_x0(
        **common,
        daughter_noise_std=0.06,
        initial_lineage_ids=[11, 22],
        return_lineage_ids=True,
    )

    assert len(points) == len(points_with_ids) == 2
    assert points.shape == points_with_ids.shape == (2,)
    assert lineage_ids.shape == (2,)
    assert all(np.issubdtype(frame.dtype, np.integer) for frame in lineage_ids)
    assert calls[0] == (0.0, None, False)
    assert calls[1][0] == 0.06
    torch.testing.assert_close(calls[1][1], torch.tensor([11, 22]))
    assert calls[1][2] is True
    np.testing.assert_array_equal(lineage_ids[0], np.asarray([11, 22]))


def test_split_from_x0_without_interaction_integrates_velocity_score_and_growth():
    class FNet:
        interaction_net = None

        @staticmethod
        def v_net(_t, z):
            return torch.ones_like(z)

        @staticmethod
        def g_net(_t, z):
            return torch.ones((z.shape[0], 1), device=z.device, dtype=z.dtype)

    class Score:
        @staticmethod
        def compute_gradient(_t, z):
            return torch.full_like(z, 2.0)

    points = simulation.simulate_sde_points_split_from_x0(
        x0=np.asarray([[0.0], [1.0]], dtype=np.float32),
        f_net=FNet(),
        score_net=Score(),
        ts_points=[0.0, 1.0],
        dt=0.25,
        sigma=0.0,
        sigma_by_dim=None,
        growth_alpha=1.0,
        interaction_m=1,
        device="cpu",
        verbose=False,
        resample_dt=1.0,
        daughter_noise_std=0.0,
    )

    assert points.shape == (2,)
    np.testing.assert_allclose(points[0], [[0.0], [1.0]], atol=1e-6)
    assert points[1].shape[0] > points[0].shape[0]
    assert set(np.round(points[1].reshape(-1), 6)) == {3.0, 4.0}


def test_interaction_grouping_rng_does_not_advance_population_rng_stream():
    class ZeroInteraction:
        requires_time = True

        @staticmethod
        def __call__(z, _lnw, _t):
            return torch.zeros_like(z)

    class FNet:
        def __init__(self, interaction_net):
            self.interaction_net = interaction_net

        @staticmethod
        def v_net(_t, z):
            return torch.zeros_like(z)

        @staticmethod
        def g_net(_t, z):
            return torch.zeros((z.shape[0], 1), device=z.device, dtype=z.dtype)

    class Score:
        @staticmethod
        def compute_gradient(_t, z):
            return torch.zeros_like(z)

    common = dict(
        x0=np.arange(8, dtype=np.float32).reshape(4, 2),
        score_net=Score(),
        ts_points=[0.0, 0.1, 0.2],
        dt=0.1,
        sigma=0.4,
        sigma_by_dim=None,
        growth_alpha=1.0,
        interaction_m=2,
        device="cpu",
        verbose=False,
        resample_dt=0.1,
        daughter_noise_std=0.0,
        interaction_seed=10_042,
    )

    torch.manual_seed(42)
    without_interaction = simulation.simulate_sde_points_split_from_x0(
        **common,
        f_net=FNet(None),
    )
    torch.manual_seed(42)
    with_zero_interaction = simulation.simulate_sde_points_split_from_x0(
        **common,
        f_net=FNet(ZeroInteraction()),
    )

    assert without_interaction.shape == with_zero_interaction.shape == (3,)
    for expected, actual in zip(without_interaction, with_zero_interaction):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    ("interaction_seed", "error_type"),
    ((True, TypeError), (-1, ValueError), (1.5, ValueError)),
)
def test_split_from_x0_rejects_invalid_interaction_seed(
    interaction_seed,
    error_type,
):
    with pytest.raises(error_type, match="interaction_seed"):
        simulation.simulate_sde_points_split_from_x0(
            x0=np.asarray([[0.0], [1.0]], dtype=np.float32),
            f_net=SimpleNamespace(
                v_net=object(),
                g_net=object(),
                interaction_net=None,
            ),
            score_net=object(),
            ts_points=[0.0, 1.0],
            dt=0.1,
            sigma=0.03,
            sigma_by_dim=None,
            growth_alpha=1.0,
            interaction_m=2,
            device="cpu",
            verbose=False,
            interaction_seed=interaction_seed,
        )


@pytest.mark.parametrize(
    ("overrides", "error_type", "match"),
    (
        ({"daughter_noise_std": -0.01}, ValueError, "finite and >= 0"),
        ({"daughter_noise_std": float("nan")}, ValueError, "finite and >= 0"),
        ({"daughter_noise_std": float("inf")}, ValueError, "finite and >= 0"),
        (
            {"initial_lineage_ids": [1]},
            ValueError,
            "one value per initial particle",
        ),
        (
            {"initial_lineage_ids": [1.0, 2.0]},
            TypeError,
            "integer identifiers",
        ),
    ),
)
def test_split_from_x0_rejects_invalid_daughter_noise_and_lineage_roster(
    overrides,
    error_type,
    match,
):
    kwargs = dict(
        x0=np.asarray([[0.0], [1.0]], dtype=np.float32),
        f_net=object(),
        score_net=object(),
        ts_points=[0.0, 1.0],
        dt=0.1,
        sigma=0.03,
        sigma_by_dim=None,
        growth_alpha=1.0,
        interaction_m=2,
        device="cpu",
        verbose=False,
    )
    kwargs.update(overrides)

    with pytest.raises(error_type, match=match):
        simulation.simulate_sde_points_split_from_x0(**kwargs)


def test_explosive_growth_fails_before_repeat_allocation(monkeypatch):
    class ExplosiveGrowthSDE:
        def f(self, _t, state):
            z, lnw = state
            return torch.zeros_like(z), torch.full_like(lnw, 1000.0)

        def g(self, _t, z):
            return torch.zeros_like(z)

    def unexpected_repeat(*_args, **_kwargs):
        raise AssertionError("repeat_interleave must not run above the ceiling")

    monkeypatch.setattr(torch, "repeat_interleave", unexpected_repeat)
    with pytest.raises(
        RuntimeError,
        match=r"particle limit exceeded before allocation at t=0.1",
    ) as error:
        _euler_sdeint_split(
            ExplosiveGrowthSDE(),
            _initial_state(),
            dt=0.1,
            ts=torch.tensor([0.0, 0.1]),
            resample_dt=0.1,
            max_particles=100_000,
        )

    assert "never downsamples particles" in str(error.value)


def test_particle_ceiling_applies_to_the_initial_state_without_downsampling():
    with pytest.raises(RuntimeError, match="limit exceeded at initial state"):
        _euler_sdeint_split(
            _UnitDriftNoGrowthSDE(),
            _initial_state(),
            dt=0.1,
            ts=torch.tensor([0.0]),
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
        max_particles=4,
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


def _reverse_time_dataframe():
    return pd.DataFrame(
        {
            "samples": [4.0, 4.0, 0.0, 0.0],
            "x1": [40.0, 41.0, 0.0, 1.0],
        }
    )


def test_legacy_simulation_time_index_uses_chronological_order(monkeypatch):
    captured = {}

    def capture_initial_state(_sde, initial_state, **_kwargs):
        z, lnw = initial_state
        captured["x0"] = z.detach().cpu().numpy().copy()
        return z.unsqueeze(0), lnw.unsqueeze(0)

    monkeypatch.setattr(simulation, "_euler_sdeint", capture_initial_state)
    f_net = SimpleNamespace(
        v_net=lambda _t, z: torch.zeros_like(z),
        g_net=lambda _t, z: torch.zeros((z.shape[0], 1), dtype=z.dtype),
        interaction_net=None,
    )

    simulation.simulate_sde_points(
        df=_reverse_time_dataframe(),
        f_net=f_net,
        score_net=object(),
        dim=1,
        time_index=0,
        n_samples=10,
        ts_points=[0.0],
        device="cpu",
        verbose=False,
    )

    np.testing.assert_array_equal(captured["x0"], [[0.0], [1.0]])


def test_legacy_split_simulation_time_index_uses_chronological_order(monkeypatch):
    captured = {}

    def capture_x0(*, x0, **_kwargs):
        captured["x0"] = np.asarray(x0).copy()
        return np.asarray([x0], dtype=object)

    monkeypatch.setattr(simulation, "simulate_sde_points_split_from_x0", capture_x0)

    simulation.simulate_sde_points_split(
        df=_reverse_time_dataframe(),
        f_net=object(),
        score_net=object(),
        dim=1,
        time_index=0,
        n_samples=10,
        ts_points=[0.0],
        device="cpu",
        verbose=False,
    )

    np.testing.assert_array_equal(captured["x0"], [[0.0], [1.0]])


def test_visualization_only_warp_uses_one_global_model_trajectory(monkeypatch):
    segment_starts = []
    daughter_noises = []

    def fake_simulate(*, x0, ts_points, daughter_noise_std, **_kwargs):
        start = np.asarray(x0, dtype=np.float32).copy()
        segment_starts.append(start)
        daughter_noises.append(daughter_noise_std)
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
        daughter_noise_std=0.06,
    )

    assert len(segment_starts) == 1
    assert daughter_noises == [0.06]
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
