import json
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest

import CytoBridge.tl.downstream.ablation as ablation_module
from CytoBridge.tl.downstream.evaluation import _prepare_ot_samples
from CytoBridge.tl import (
    VirtualAblationResult,
    compute_virtual_ablation_metrics,
    run_virtual_cell_type_ablation,
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
