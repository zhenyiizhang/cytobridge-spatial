from __future__ import annotations

import numpy as np

from CytoBridge.tl.downstream.evaluation import compute_distribution_metrics


def test_exact_ot_coalesces_duplicate_support_without_changing_measure(
    monkeypatch,
) -> None:
    import ot

    predicted = np.array([[0.0], [0.0], [2.0]], dtype=np.float64)
    observed = np.array([[0.0], [1.0], [1.0]], dtype=np.float64)
    predicted_weights = np.array([0.25, 0.25, 0.5], dtype=np.float64)

    original_emd2 = ot.emd2
    support_shapes: list[tuple[int, int]] = []

    def record_support(a, b, cost, **kwargs):
        support_shapes.append(cost.shape)
        return original_emd2(a, b, cost, **kwargs)

    monkeypatch.setattr(ot, "emd2", record_support)
    metrics = compute_distribution_metrics(
        predicted,
        observed,
        predicted_weights=predicted_weights,
        max_ot_points=None,
    )

    assert support_shapes == [(2, 2), (2, 2)]
    assert metrics["w1"] == 2.0 / 3.0
    assert metrics["w2"] == np.sqrt(2.0 / 3.0)
    assert metrics["ot_predicted_points"] == 3
    assert metrics["ot_observed_points"] == 3
