from __future__ import annotations

import pandas as pd

from CytoBridge.tl import (
    compare_distribution_metric_tables,
    compute_local_structure_metrics,
    save_distribution_metric_comparison,
)


def _metrics(offset: float) -> pd.DataFrame:
    rows = []
    for time in (1.0, 2.0):
        for space in ("joint", "pca", "spatial"):
            rows.append(
                {
                    "time": time,
                    "space": space,
                    "w1": time + offset,
                    "w2": time + 0.5 + offset,
                    "tmv": 0.1 + offset,
                    "nn_dispersion_ratio": 1.0 + offset,
                    "support_recall_at_observed_q95": 0.9 + offset,
                    "support_precision_at_observed_q95": 0.95 + offset,
                    "clump_fraction_at_0_1_observed_nn": 0.02 + offset,
                }
            )
    return pd.DataFrame(rows)


def test_five_model_metric_comparison_and_figure(tmp_path) -> None:
    tables = {
        "published_saved_model": _metrics(0.0),
        "current_six_stage_legacy_input": _metrics(-0.01),
        "current_six_stage_current_preprocess_published_thresholds": _metrics(-0.02),
        "current_six_stage_current_preprocess_auto_thresholds": _metrics(-0.03),
        "current_six_stage_strict_frozen_auto_edge_published_thresholds": _metrics(-0.04),
    }
    comparison = compare_distribution_metric_tables(
        tables,
        baseline="published_saved_model",
    )
    paths = save_distribution_metric_comparison(comparison, tmp_path)

    assert comparison.metrics["model"].nunique() == 5
    assert comparison.paired_deltas["candidate"].nunique() == 4
    assert (comparison.paired_deltas["w1_delta"] < 0).all()
    assert set(paths) == {
        "metrics",
        "summary",
        "paired_deltas",
        "figure",
        "local_structure_figure",
    }
    assert all((tmp_path / name).is_file() for name in (
        "model_metrics_long.csv",
        "model_metrics_mean_by_space.csv",
        "model_metrics_paired_deltas.csv",
        "model_metric_comparison.svg",
        "model_local_structure_comparison.svg",
    ))


def test_local_structure_metrics_detect_particle_collapse() -> None:
    observed = pd.DataFrame(
        [(x, y) for x in range(10) for y in range(10)],
        columns=["x", "y"],
    ).to_numpy(dtype=float)
    predicted = observed.copy()
    predicted[:50] = predicted[0]

    metrics = compute_local_structure_metrics(
        predicted,
        observed,
        max_points=None,
    )

    assert metrics["nn_dispersion_ratio"] < 1.0
    assert metrics["clump_fraction_at_0_1_observed_nn"] >= 0.5
    assert 0.0 <= metrics["support_recall_at_observed_q95"] <= 1.0
    assert 0.0 <= metrics["support_precision_at_observed_q95"] <= 1.0
