import json
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import torch

import CytoBridge.tl.downstream.ablation as ablation_module
from CytoBridge.tl.downstream.evaluation import _prepare_ot_samples
from CytoBridge.tl import (
    VirtualAblationResult,
    VirtualInteractionAblationResult,
    compute_virtual_ablation_metrics,
    run_virtual_cell_type_ablation,
    run_virtual_interaction_ablation,
)


def _trajectory(*frames):
    values = np.empty(len(frames), dtype=object)
    values[:] = [np.asarray(frame, dtype=np.float32) for frame in frames]
    return values


def _toy_adata():
    obs = pd.DataFrame(
        {
            "time_point_processed": [0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
            "Annotation": ["A", "A", "B", "C", "B", "C"],
        },
        index=[f"cell_{index}" for index in range(6)],
    )
    adata = ad.AnnData(X=np.zeros((6, 1), dtype=np.float32), obs=obs)
    adata.obsm["spatial_aligned"] = np.asarray(
        [[0, 0], [1, 0], [2, 0], [3, 0], [2, 1], [3, 1]],
        dtype=np.float32,
    )
    adata.obsm["X_latent"] = np.arange(6, dtype=np.float32).reshape(-1, 1)
    return adata


def test_compute_virtual_ablation_metrics_reports_spaces_counts_and_shifts():
    baseline = _trajectory(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 2.0]],
        [[1.0, 0.0, 1.0], [3.0, 0.0, 3.0]],
    )
    variant = _trajectory(
        [[1.0, 0.0, 1.0]],
        [[4.0, 0.0, 4.0], [6.0, 0.0, 6.0], [8.0, 0.0, 8.0]],
    )

    metrics = compute_virtual_ablation_metrics(
        baseline,
        {"remove_A": variant},
        [0.0, 1.0],
        spatial_dim=2,
    )

    assert set(metrics["space"]) == {"joint", "spatial", "latent"}
    row = metrics.query("time == 1.0 and space == 'spatial'").iloc[0]
    assert row["n_baseline"] == 2
    assert row["n_ablation"] == 3
    assert row["count_delta"] == 1
    assert row["count_ratio"] == pytest.approx(1.5)
    # Baseline centroid x=2; ablated centroid x=6.
    assert row["centroid_shift"] == pytest.approx(4.0)
    assert row["w1"] == pytest.approx(4.0)
    assert row["w2"] == pytest.approx(np.sqrt(17.0))
    assert row["ot_ablation_points"] == 3
    assert row["ot_baseline_points"] == 2
    assert row["baseline_rms_radius"] == pytest.approx(1.0)
    assert row["ablation_rms_radius"] == pytest.approx(np.sqrt(8.0 / 3.0))


def test_uniform_ot_clouds_are_capped_without_replacement():
    predicted = np.arange(20, dtype=float).reshape(-1, 1)
    observed = np.arange(100, 120, dtype=float).reshape(-1, 1)
    pred, obs, pred_weights, obs_weights = _prepare_ot_samples(
        predicted,
        observed,
        None,
        max_ot_points=8,
        rng=np.random.default_rng(7),
    )

    assert np.unique(pred, axis=0).shape[0] == 8
    assert np.unique(obs, axis=0).shape[0] == 8
    np.testing.assert_allclose(pred_weights, np.full(8, 1.0 / 8.0))
    np.testing.assert_allclose(obs_weights, np.full(8, 1.0 / 8.0))


def test_virtual_ablation_workflow_shares_cohort_exports_data_and_snapshots(
    monkeypatch, tmp_path
):
    adata = _toy_adata()
    simulated_x0 = []
    simulation_options = []

    def fake_simulate_sde_points_split_from_x0(
        *, x0, ts_points, growth_alpha, sigma_by_dim, **_kwargs
    ):
        initial = np.asarray(x0, dtype=np.float32)
        simulated_x0.append(initial.copy())
        simulation_options.append((growth_alpha, sigma_by_dim))
        return _trajectory(
            *(initial + float(time_value) for time_value in ts_points)
        )

    def labeler(points, _time_points):
        return [np.repeat("predicted", len(frame)) for frame in points]

    monkeypatch.setattr(
        ablation_module,
        "simulate_sde_points_split_from_x0",
        fake_simulate_sde_points_split_from_x0,
    )

    runtime = SimpleNamespace(model=object(), f_net=object(), score_net=object())
    result = run_virtual_cell_type_ablation(
        adata,
        runtime,
        ablations={"remove_A": ["A"], "remove_A_and_B": ["A", "B"]},
        time_points=[0.0, 0.5, 1.0],
        output_dir=tmp_path,
        n_samples=None,
        dt=0.05,
        sigma=0.03,
        growth_alpha=0.5,
        device="cpu",
        trajectory_labeler=labeler,
        snapshot_times=[0.0, 1.0],
        snapshot_formats=("png",),
        verbose=False,
    )

    assert isinstance(result, VirtualAblationResult)
    assert result.initial_obs_names == ("cell_0", "cell_1", "cell_2", "cell_3")
    np.testing.assert_array_equal(
        simulated_x0[0],
        [[0, 0, 0], [1, 0, 1], [2, 0, 2], [3, 0, 3]],
    )
    np.testing.assert_array_equal(simulated_x0[1], [[2, 0, 2], [3, 0, 3]])
    np.testing.assert_array_equal(simulated_x0[2], [[3, 0, 3]])
    assert simulation_options == [(0.5, None)] * 3
    assert [len(frame) for frame in result.baseline_points] == [4, 4, 4]
    assert [len(frame) for frame in result.ablation_points["remove_A"]] == [2, 2, 2]
    assert result.settings["variant_initial_counts"] == {
        "remove_A": 2,
        "remove_A_and_B": 1,
    }
    assert result.settings["growth_alpha"] == pytest.approx(0.5)
    assert result.settings["max_ot_points"] == 1024
    assert result.settings["simulation"].endswith("no replacement")

    expected = [
        tmp_path / "trajectories" / "baseline_points.npy",
        tmp_path / "trajectories" / "remove_A_points.npy",
        tmp_path / "ablation_metrics.csv",
        tmp_path / "label_composition.csv",
        tmp_path / "initial_cohort.csv",
        tmp_path / "manifest.json",
        tmp_path / "snapshots" / "remove_A" / "frame_000_t_0.png",
        tmp_path / "snapshots" / "remove_A" / "frame_002_t_1.png",
        tmp_path / "snapshots" / "label_legend.png",
    ]
    assert all(path.exists() for path in expected)

    saved = np.load(
        tmp_path / "trajectories" / "baseline_points.npy", allow_pickle=True
    )
    assert saved.shape == (3,)
    assert np.asarray(saved[0]).shape == (4, 3)

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["schema_version"] == 1
    assert manifest["trajectory_shapes"]["remove_A"] == [[2, 3]] * 3

    cohort = pd.read_csv(tmp_path / "initial_cohort.csv")
    assert cohort["kept__remove_A"].tolist() == [False, False, True, True]
    composition = pd.read_csv(tmp_path / "label_composition.csv")
    row = composition.query(
        "variant == 'remove_A' and time == 1.0 and label == 'predicted'"
    ).iloc[0]
    assert row["baseline_count"] == 4
    assert row["ablation_count"] == 2


def test_virtual_ablation_rejects_unknown_or_total_removal_before_simulation():
    adata = _toy_adata()

    with pytest.raises(ValueError, match="absent at start time"):
        run_virtual_cell_type_ablation(
            adata,
            object(),
            ablations={"unknown": ["not-a-label"]},
            time_points=[0.0, 1.0],
            output_dir=None,
            device="cpu",
        )

    with pytest.raises(ValueError, match="removes every cell"):
        run_virtual_cell_type_ablation(
            adata,
            object(),
            ablations={"all": ["A", "B", "C"]},
            time_points=[0.0, 1.0],
            output_dir=None,
            device="cpu",
        )


def test_virtual_ablation_mass_control_uses_equal_independent_cohorts(
    monkeypatch, tmp_path
):
    adata = _toy_adata()
    simulated_x0 = []

    def fake_simulate_sde_points_split_from_x0(*, x0, ts_points, **_kwargs):
        initial = np.asarray(x0, dtype=np.float32)
        simulated_x0.append(initial.copy())
        return _trajectory(*(initial.copy() for _ in ts_points))

    monkeypatch.setattr(
        ablation_module,
        "simulate_sde_points_split_from_x0",
        fake_simulate_sde_points_split_from_x0,
    )
    runtime = SimpleNamespace(model=object(), f_net=object(), score_net=object())

    result = run_virtual_cell_type_ablation(
        adata,
        runtime,
        ablations={"remove_A": ["A"]},
        time_points=[0.0, 0.5, 1.0],
        output_dir=tmp_path,
        n_samples=3,
        mass_control=True,
        random_seed=7,
        device="cpu",
        save_snapshots=False,
        verbose=False,
    )

    # Full pool=4 and post-removal pool=2, so both branches start with two
    # independently sampled cells even though the requested cap is three.
    assert simulated_x0[0].shape == (2, 3)
    assert simulated_x0[1].shape == (2, 3)
    assert not np.isin(simulated_x0[1][:, 2], [0.0, 1.0]).any()
    historical_rng = np.random.default_rng(7)
    historical_rng.choice(np.asarray([0, 1]), size=2, replace=False)
    expected_baseline_positions = historical_rng.choice(
        np.arange(4), size=2, replace=False
    )
    expected_ablation_positions = historical_rng.choice(
        np.asarray([2, 3]), size=2, replace=False
    )
    full_features = np.column_stack(
        [
            adata.obsm["spatial_aligned"][:4],
            adata.obsm["X_latent"][:4],
        ]
    )
    np.testing.assert_array_equal(
        simulated_x0[0], full_features[expected_baseline_positions]
    )
    np.testing.assert_array_equal(
        simulated_x0[1], full_features[expected_ablation_positions]
    )
    assert result.settings["mass_control"] is True
    assert result.settings["cohort_sampling_mode"] == (
        "matched_independent_no_replacement"
    )
    assert result.settings["baseline_pool_size"] == 4
    assert result.settings["variant_pool_sizes"] == {"remove_A": 2}
    assert result.settings["matched_initial_particle_count"] == 2
    assert result.settings["sampling_draw_order"] == [
        "remove_pool_label:A",
        "baseline",
        "remove_A",
    ]
    assert "historical RNG consumption" in result.settings[
        "ablation_pool_construction"
    ]
    assert result.settings["simulation_seeds"] == {
        "baseline": 7,
        "remove_A": 7,
    }
    assert "not cell ID" in result.settings["random_stream_coupling"]
    assert set(result.metrics["count_delta"]) == {0}

    cohort = pd.read_csv(tmp_path / "initial_cohort.csv")
    assert len(cohort) == 4
    assert int(cohort["selected_in_baseline"].sum()) == 2
    assert int(cohort["selected_in__remove_A"].sum()) == 2
    assert not cohort.loc[cohort["initial_label"].eq("A"), "selected_in__remove_A"].any()

    sampling = pd.read_csv(tmp_path / "cohort_sampling.csv")
    assert sampling.groupby("branch").size().to_dict() == {
        "baseline": 2,
        "remove_A": 2,
    }
    assert set(sampling.query("branch == 'remove_A'")["initial_label"]) == {
        "B",
        "C",
    }

    # Cohort sampling is deterministic and independent of SDE random state.
    first_baseline, first_ablation = [values.copy() for values in simulated_x0]
    run_virtual_cell_type_ablation(
        adata,
        runtime,
        ablations={"remove_A": ["A"]},
        time_points=[0.0, 0.5, 1.0],
        output_dir=None,
        n_samples=3,
        mass_control=True,
        random_seed=7,
        device="cpu",
        save_snapshots=False,
        verbose=False,
    )
    np.testing.assert_array_equal(simulated_x0[2], first_baseline)
    np.testing.assert_array_equal(simulated_x0[3], first_ablation)


def test_virtual_ablation_mass_control_requires_one_ablation():
    with pytest.raises(ValueError, match="exactly one ablation"):
        run_virtual_cell_type_ablation(
            _toy_adata(),
            object(),
            ablations={"remove_A": ["A"], "remove_B": ["B"]},
            time_points=[0.0, 1.0],
            mass_control=True,
            output_dir=None,
            device="cpu",
        )


def test_virtual_interaction_ablation_pairs_x0_seed_and_zero_force(
    monkeypatch, tmp_path
):
    calls = []

    class FakeInteraction:
        requires_time = True

        def __call__(self, x, _lnw, _t):
            return torch.ones_like(x)

    f_net = SimpleNamespace(
        v_net=object(),
        g_net=object(),
        interaction_net=FakeInteraction(),
    )
    runtime = SimpleNamespace(model=object(), f_net=f_net, score_net=object())

    def fake_simulate_sde_points_split_from_x0(
        *, x0, f_net, score_net, ts_points, dt, sigma, interaction_m, **_kwargs
    ):
        initial = np.asarray(x0, dtype=np.float32)
        tensor = torch.as_tensor(initial)
        lnw = torch.zeros((len(initial), 1), dtype=tensor.dtype)
        force = f_net.interaction_net(tensor, lnw, torch.tensor(0.0))
        calls.append(
            {
                "x0": initial.copy(),
                "force": force.detach().cpu().numpy(),
                "numpy_random": np.random.random(4),
                "torch_random": torch.rand(4).numpy(),
                "score_net": score_net,
                "dt": dt,
                "sigma": sigma,
                "interaction_m": interaction_m,
            }
        )
        offset = float(force.mean())
        return _trajectory(
            *(initial + offset + float(time_value) for time_value in ts_points)
        )

    monkeypatch.setattr(
        ablation_module,
        "simulate_sde_points_split_from_x0",
        fake_simulate_sde_points_split_from_x0,
    )
    x0 = np.asarray(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 2.0], [2.0, 0.0, 3.0]],
        dtype=np.float32,
    )
    result = run_virtual_interaction_ablation(
        x0,
        runtime,
        time_points=[0.0, 0.5, 1.0],
        output_dir=tmp_path,
        random_seed=17,
        dt=0.05,
        sigma=0.03,
        interaction_m=2,
        spatial_dim=2,
        device="cpu",
        verbose=False,
    )

    assert isinstance(result, VirtualInteractionAblationResult)
    np.testing.assert_array_equal(calls[0]["x0"], x0)
    np.testing.assert_array_equal(calls[1]["x0"], x0)
    np.testing.assert_array_equal(calls[0]["force"], np.ones_like(x0))
    np.testing.assert_array_equal(calls[1]["force"], np.zeros_like(x0))
    np.testing.assert_array_equal(calls[0]["numpy_random"], calls[1]["numpy_random"])
    np.testing.assert_array_equal(calls[0]["torch_random"], calls[1]["torch_random"])
    assert calls[0]["score_net"] is calls[1]["score_net"]
    assert calls[0]["dt"] == calls[1]["dt"] == pytest.approx(0.05)
    assert calls[0]["sigma"] == calls[1]["sigma"] == pytest.approx(0.03)
    assert calls[0]["interaction_m"] == calls[1]["interaction_m"] == 2
    assert result.settings["same_initial_state"] is True
    assert result.settings["baseline_interaction_scale"] == 1.0
    assert result.settings["ablated_interaction_scale"] == 0.0
    assert "both branches execute cal_interaction" in result.settings[
        "random_stream_control"
    ]
    assert set(result.metrics["variant"]) == {"interaction_off"}
    assert (tmp_path / "trajectories" / "initial_x0.npy").exists()
    assert (tmp_path / "trajectories" / "interaction_on_points.npy").exists()
    assert (tmp_path / "trajectories" / "interaction_off_points.npy").exists()
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["trajectory_shapes"]["interaction_on"] == [[3, 3]] * 3
    assert manifest["trajectory_shapes"]["interaction_off"] == [[3, 3]] * 3
