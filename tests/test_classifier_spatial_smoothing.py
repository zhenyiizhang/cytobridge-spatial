from __future__ import annotations

import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder

from CytoBridge.tl.downstream.classification import (
    analyze_spatial_label_sensitivity,
    predict_labels_for_points,
    predict_labels_for_trajectories,
    smooth_spatial_labels,
)


def test_spatial_smoothing_k_at_most_one_is_identity_and_does_not_mutate_inputs():
    labels = np.asarray(["A", "B", "A", "B"])
    coords = np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    labels_before = labels.copy()
    coords_before = coords.copy()

    np.testing.assert_array_equal(smooth_spatial_labels(labels, coords, k=0), labels)
    np.testing.assert_array_equal(smooth_spatial_labels(labels, coords, k=1), labels)
    np.testing.assert_array_equal(labels, labels_before)
    np.testing.assert_array_equal(coords, coords_before)


def test_spatial_smoothing_clamps_k_above_n_and_suppresses_rare_label():
    labels = np.asarray(["A", "A", "A", "B"])
    coords = np.arange(4, dtype=np.float64).reshape(-1, 1)

    refined = smooth_spatial_labels(labels, coords, k=99)

    np.testing.assert_array_equal(refined, ["A", "A", "A", "A"])
    report = analyze_spatial_label_sensitivity(labels, coords, k_values=(99,))
    record = report["results"][0]
    assert record["requested_k"] == 99
    assert record["effective_k"] == 4
    assert record["per_type"]["B"]["raw_count"] == 1
    assert record["per_type"]["B"]["smoothed_count"] == 0
    assert record["per_type"]["B"]["retention_fraction"] == 0.0


def test_explicit_spatial_coordinates_change_only_the_smoothing_geometry():
    labels = np.asarray(["A", "A", "A", "B", "B", "B"])
    segregated = np.asarray([[0.0], [1.0], [2.0], [10.0], [11.0], [12.0]])
    interleaved = np.asarray([[0.0], [2.0], [4.0], [1.0], [3.0], [5.0]])

    segregated_result = smooth_spatial_labels(labels, segregated, k=3)
    interleaved_result = smooth_spatial_labels(labels, interleaved, k=3)

    np.testing.assert_array_equal(segregated_result, labels)
    assert np.any(interleaved_result != labels)


def test_even_k_ties_are_deterministic_and_report_legacy_policy_metadata():
    labels = np.asarray(["B", "A"])
    coords = np.asarray([[0.0], [1.0]])

    first = smooth_spatial_labels(labels, coords, k=2)
    second = smooth_spatial_labels(labels, coords, k=2)
    report = analyze_spatial_label_sensitivity(
        labels,
        coords,
        k_values=(2,),
        boundary_mask=np.asarray([True, False]),
    )
    record = report["results"][0]

    np.testing.assert_array_equal(first, ["A", "A"])
    np.testing.assert_array_equal(second, first)
    assert record["even_effective_k"] is True
    assert record["n_vote_ties"] == 2
    assert record["tie_policy"] == "sklearn_legacy"
    assert record["neighbor_algorithm"] == (
        "scipy.spatial.cKDTree_exact_boundary_ties"
    )
    assert record["self_inclusion_contract"] == "forced_query_row"
    assert record["composition_total_variation"] == 0.5
    assert record["max_absolute_change_pp"] == 50.0
    assert record["boundary_flip_rate"] == 1.0
    assert record["interior_flip_rate"] == 0.0


class _SignClassifier(torch.nn.Module):
    def forward(self, values: torch.Tensor) -> torch.Tensor:
        feature = values[:, 0]
        return torch.stack((-feature, feature), dim=1)


def test_point_prediction_honors_explicit_feature_and_spatial_coordinates():
    points = np.asarray(
        [
            [0.0, 0.0, -3.0],
            [1.0, 0.0, -2.0],
            [2.0, 0.0, -1.0],
            [10.0, 0.0, 1.0],
            [11.0, 0.0, 2.0],
            [12.0, 0.0, 3.0],
        ],
        dtype=np.float32,
    )
    explicit_spatial = np.asarray(
        [[0.0], [2.0], [4.0], [1.0], [3.0], [5.0]], dtype=np.float32
    )
    points_before = points.copy()
    spatial_before = explicit_spatial.copy()
    encoder = LabelEncoder().fit(["A", "B"])

    predicted = predict_labels_for_points(
        points=points,
        time_value=0.0,
        model=_SignClassifier(),
        label_encoder=encoder,
        feature_dim=1,
        device="cpu",
        knn_neighbors=3,
        include_time_feature=False,
        feature_indices=(2,),
        spatial_coords=explicit_spatial,
    )
    trajectory_predicted = predict_labels_for_trajectories(
        sde_points=np.asarray([points]),
        ts_points=(0.0,),
        model=_SignClassifier(),
        label_encoder=encoder,
        feature_dim=1,
        device="cpu",
        knn_neighbors=3,
        include_time_feature=False,
        feature_indices=(2,),
        spatial_coords=explicit_spatial,
    )[0]
    raw = np.asarray(["A", "A", "A", "B", "B", "B"])
    expected = smooth_spatial_labels(raw, explicit_spatial, k=3)

    np.testing.assert_array_equal(predicted, expected)
    np.testing.assert_array_equal(trajectory_predicted, expected)
    assert np.any(predicted != raw)
    np.testing.assert_array_equal(points, points_before)
    np.testing.assert_array_equal(explicit_spatial, spatial_before)
