from __future__ import annotations

import json

import numpy as np
import pytest

from CytoBridge.tl.downstream.benchmark import (
    FrozenBenchmarkTransform,
    benchmark_projection_seed,
    evaluate_spatiotemporal_prediction,
    fit_frozen_benchmark_transform,
)


def _training_arrays() -> tuple[np.ndarray, np.ndarray]:
    state = np.asarray(
        [
            [-3.0, 10.0, 2.0],
            [-1.0, 14.0, 8.0],
            [1.0, 18.0, 5.0],
            [3.0, 22.0, 11.0],
        ]
    )
    spatial = np.asarray(
        [
            [-4.0, -1.0],
            [-1.0, 2.0],
            [2.0, -3.0],
            [7.0, 2.0],
        ]
    )
    return state, spatial


def test_frozen_transform_uses_training_rows_and_roundtrips_json() -> None:
    train_state, train_spatial = _training_arrays()
    transform = fit_frozen_benchmark_transform(train_state, train_spatial)
    before = transform.to_json()

    # Transforming an extreme held-out row cannot update frozen train moments.
    transform.transform_state(np.asarray([[1e9, -1e9, 5e8]]))
    transform.transform_spatial(np.asarray([[2e7, -3e7]]))
    assert transform.to_json() == before

    restored = FrozenBenchmarkTransform.from_json(before)
    np.testing.assert_allclose(
        restored.transform_joint(train_state, train_spatial),
        transform.transform_joint(train_state, train_spatial),
    )
    assert json.loads(before)["state_dim"] == train_state.shape[1]
    assert json.loads(before)["spatial_dim"] == train_spatial.shape[1]

    corrupted = transform.to_dict()
    corrupted["state_dim"] = train_state.shape[1] + 1
    with pytest.raises(ValueError, match="state_dim"):
        FrozenBenchmarkTransform.from_dict(corrupted)


def test_transform_balances_state_and_spatial_blocks_without_shape_warp() -> None:
    train_state, train_spatial = _training_arrays()
    transform = FrozenBenchmarkTransform.fit(train_state, train_spatial)
    state_block = transform.transform_state(train_state)
    spatial_block = transform.transform_spatial(train_spatial)

    # Each block has unit expected squared norm on the rows used for fitting.
    assert np.mean(np.sum(state_block**2, axis=1)) == pytest.approx(1.0)
    assert np.mean(np.sum(spatial_block**2, axis=1)) == pytest.approx(1.0)

    # A single spatial divisor keeps relative coordinate scales/geometric shape.
    raw_delta = train_spatial[3] - train_spatial[0]
    transformed_delta = spatial_block[3] - spatial_block[0]
    assert transformed_delta[0] / transformed_delta[1] == pytest.approx(
        raw_delta[0] / raw_delta[1]
    )
    joint = transform.transform_joint(train_state, train_spatial)
    np.testing.assert_allclose(joint[:, : train_state.shape[1]], state_block)
    np.testing.assert_allclose(joint[:, train_state.shape[1] :], spatial_block)


def test_state_only_evaluation_emits_only_state_and_keeps_unequal_counts() -> None:
    train_state, _ = _training_arrays()
    transform = FrozenBenchmarkTransform.fit(train_state)
    predicted = np.vstack((train_state, [[5.0, 26.0, 14.0]]))
    observed = train_state[:3]

    metrics = evaluate_spatiotemporal_prediction(
        transform=transform,
        benchmark="zebrafish",
        split="holdout_12hpf",
        method="STORIES_state_only",
        predicted_state=predicted,
        observed_state=observed,
        n_projections=16,
        projection_repeats=3,
        max_ot_points=2,
    )

    assert metrics.shape[0] == 3
    assert set(metrics["space"]) == {"state"}
    assert set(metrics["n_predicted"]) == {5}
    assert set(metrics["n_observed"]) == {3}
    assert set(metrics["exact_ot_predicted_points"]) == {2}
    assert set(metrics["exact_ot_observed_points"]) == {2}
    assert set(metrics["primary_metric"]) == {"sliced_w2"}
    np.testing.assert_allclose(metrics["primary_value"], metrics["sliced_w2"])


def test_projection_basis_is_shared_across_method_names() -> None:
    train_state, train_spatial = _training_arrays()
    transform = FrozenBenchmarkTransform.fit(train_state, train_spatial)
    common = dict(
        transform=transform,
        benchmark="zebrafish",
        split="holdout_18hpf",
        predicted_state=train_state + np.asarray([0.2, -0.1, 0.3]),
        observed_state=train_state,
        predicted_spatial=train_spatial + np.asarray([0.1, -0.2]),
        observed_spatial=train_spatial,
        n_projections=24,
        projection_repeats=2,
        max_ot_points=None,
    )
    method_a = evaluate_spatiotemporal_prediction(method="stVCR", **common)
    method_b = evaluate_spatiotemporal_prediction(method="MOSCOT", **common)

    columns = ["space", "projection_repeat"]
    method_a = method_a.sort_values(columns).reset_index(drop=True)
    method_b = method_b.sort_values(columns).reset_index(drop=True)
    assert method_a["projection_seed"].tolist() == method_b["projection_seed"].tolist()
    assert (
        method_a["projection_sha256"].tolist() == method_b["projection_sha256"].tolist()
    )
    np.testing.assert_allclose(method_a["sliced_w2"], method_b["sliced_w2"])
    assert benchmark_projection_seed(
        "zebrafish", "holdout_18hpf", "state", 0
    ) != benchmark_projection_seed("zebrafish", "holdout_18hpf", "state", 1)


def test_predicted_weights_are_used_by_sliced_and_exact_ot() -> None:
    train_state = np.asarray([[-10.0], [0.0], [10.0], [20.0]])
    transform = FrozenBenchmarkTransform.fit(train_state)
    observed = np.asarray([[0.0], [10.0]])
    predicted = np.asarray([[0.0], [10.0], [1e5]])

    metrics = evaluate_spatiotemporal_prediction(
        transform=transform,
        benchmark="weight_test",
        split="all_times",
        method="weighted_particles",
        predicted_state=predicted,
        observed_state=observed,
        predicted_weights=np.asarray([2.0, 2.0, 0.0]),
        n_projections=8,
        projection_repeats=2,
        max_ot_points=None,
    )

    assert metrics["predicted_weight_sum"].tolist() == [4.0, 4.0]
    np.testing.assert_allclose(metrics["sliced_w2"], 0.0, atol=1e-12)
    np.testing.assert_allclose(metrics["exact_w1"], 0.0, atol=1e-12)
    np.testing.assert_allclose(metrics["exact_w2"], 0.0, atol=1e-12)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"predicted_state": [[np.nan, 0.0, 1.0]]}, "finite"),
        ({"predicted_state": [[0.0, 1.0]]}, "expected 3"),
        ({"predicted_weights": [1.0, -1.0, 1.0, 1.0]}, "non-negative"),
        ({"predicted_weights": [0.0, 0.0, 0.0, 0.0]}, "positive sum"),
        ({"predicted_weights": [1.0, 1.0]}, "expected 4"),
        ({"predicted_weights": [[1.0, 1.0, 1.0, 1.0]]}, "one-dimensional"),
        ({"predicted_spatial": [[0.0, 0.0]]}, "both be set"),
        ({"method": None}, "must be a string"),
        ({"split": "   "}, "non-empty string"),
        ({"n_projections": 0}, "positive integer"),
        ({"projection_repeats": 1.5}, "positive integer"),
        ({"max_ot_points": -1}, "positive integer"),
    ],
)
def test_evaluator_rejects_invalid_inputs(overrides, message: str) -> None:
    train_state, _ = _training_arrays()
    transform = FrozenBenchmarkTransform.fit(train_state)
    arguments = dict(
        transform=transform,
        benchmark="zebrafish",
        split="all_times",
        method="linear",
        predicted_state=train_state,
        observed_state=train_state,
        n_projections=8,
        projection_repeats=1,
        max_ot_points=None,
    )
    arguments.update(overrides)
    with pytest.raises((TypeError, ValueError), match=message):
        evaluate_spatiotemporal_prediction(**arguments)


def test_evaluator_rejects_spatial_row_mismatch_and_state_only_transform() -> None:
    train_state, train_spatial = _training_arrays()
    transform = FrozenBenchmarkTransform.fit(train_state, train_spatial)
    with pytest.raises(ValueError, match="predicted_state and predicted_spatial"):
        evaluate_spatiotemporal_prediction(
            transform=transform,
            benchmark="zebrafish",
            split="all_times",
            method="linear",
            predicted_state=train_state,
            observed_state=train_state,
            predicted_spatial=train_spatial[:2],
            observed_spatial=train_spatial,
            projection_repeats=1,
        )

    state_only_transform = FrozenBenchmarkTransform.fit(train_state)
    with pytest.raises(ValueError, match="without spatial"):
        evaluate_spatiotemporal_prediction(
            transform=state_only_transform,
            benchmark="zebrafish",
            split="all_times",
            method="linear",
            predicted_state=train_state,
            observed_state=train_state,
            predicted_spatial=train_spatial,
            observed_spatial=train_spatial,
            projection_repeats=1,
        )
