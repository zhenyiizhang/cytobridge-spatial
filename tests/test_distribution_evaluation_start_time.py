from __future__ import annotations

import anndata as ad
import numpy as np

from CytoBridge.tl.downstream import evaluation


def test_target_only_evaluation_still_simulates_from_observed_initial_time(monkeypatch):
    times = np.repeat([0.0, 1.0, 2.0], [2, 4, 6])
    adata = ad.AnnData(X=np.zeros((len(times), 1), dtype=np.float32))
    adata.obs["time"] = times
    adata.obsm["X_latent"] = np.zeros((len(times), 1), dtype=np.float32)
    adata.obsm["spatial_aligned"] = np.zeros((len(times), 2), dtype=np.float32)

    def fake_simulation(**kwargs):
        assert kwargs["ts_points"] == [0.0, 1.0, 2.0]
        points = np.asarray(
            [
                np.full((2, 3), time_value, dtype=np.float32)
                for time_value in kwargs["ts_points"]
            ],
            dtype=object,
        )
        weights = np.asarray(
            [
                np.full(2, (time_value + 1.0) / 2.0, dtype=np.float64)
                for time_value in kwargs["ts_points"]
            ]
        )
        return points, weights

    monkeypatch.setattr(evaluation, "simulate_sde_points", fake_simulation)
    monkeypatch.setattr(
        evaluation,
        "compute_distribution_metrics",
        lambda *args, **kwargs: {
            "w1": 0.0,
            "w2": 0.0,
            "ot_predicted_points": 2,
            "ot_observed_points": 2,
        },
    )
    monkeypatch.setattr(
        evaluation,
        "compute_local_structure_metrics",
        lambda *args, **kwargs: {},
    )

    result = evaluation.evaluate_model_distributions(
        adata,
        object(),
        time_points=[2.0, 1.0],
        time_key="time",
        device="cpu",
        verbose=False,
    )

    assert result.time_points == (1.0, 2.0)
    np.testing.assert_allclose(result.predicted_points[1.0], 1.0)
    np.testing.assert_allclose(result.predicted_points[2.0], 2.0)
    assert result.settings["simulation_initial_time"] == 0.0
    assert result.metrics.groupby("time")["observed_mass_relative"].first().to_dict() == {
        1.0: 2.0,
        2.0: 3.0,
    }
    assert result.metrics.groupby("time")["tmv"].first().to_dict() == {
        1.0: 0.0,
        2.0: 0.0,
    }

