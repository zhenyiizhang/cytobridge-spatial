from __future__ import annotations

import numpy as np

from CytoBridge.pp.edge_prediction import (
    _balanced_training_pool,
    _binary_metrics,
    _build_leakage_free_split,
    _find_best_threshold,
    _unique_directed_edges,
    _validate_positive_spatial_contract,
    vectorized_negative_sampling,
)


def _balanced_edges_for_times(n_times: int, n_nodes: int = 8):
    edges = []
    labels = []
    times = []
    for time_idx in range(n_times):
        for node in range(n_nodes):
            edges.append([node, (node + 1) % n_nodes])
            labels.append(1)
            edges.append([node, (node + 3) % n_nodes])
            labels.append(0)
            times.extend([time_idx, time_idx])
    return (
        np.asarray(edges, dtype=np.int32),
        np.asarray(labels, dtype=np.float32),
        np.asarray(times, dtype=np.int32),
    )


def test_positive_lr_multiedges_are_unique_per_directed_cell_pair():
    edges = np.asarray([[0, 1], [0, 1], [1, 0], [0, 2], [0, 2]], dtype=np.int32)
    unique, counts = _unique_directed_edges(edges)

    assert unique.tolist() == [[0, 1], [0, 2], [1, 0]]
    assert counts == {"raw": 5, "unique": 3, "duplicates_removed": 2}


def test_negative_candidates_follow_the_same_strict_spatial_contract():
    coordinates = np.asarray(
        [[0.0, 0.0], [0.0, 0.0], [0.5, 0.0], [1.0, 0.0]], dtype=np.float64
    )
    sampled = vectorized_negative_sampling(
        positive_edges=np.empty((0, 2), dtype=np.int32),
        num_nodes=4,
        num_neg_samples=100,
        spatial_coords=coordinates,
        distance_threshold=1.0,
    )

    assert {tuple(edge) for edge in sampled.tolist()} == {
        (0, 2),
        (1, 2),
        (2, 0),
        (2, 1),
        (2, 3),
        (3, 2),
    }


def test_training_pool_is_balanced_without_changing_validation_prevalence():
    labels = np.asarray([1, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    balanced = _balanced_training_pool(np.arange(labels.size), labels, random_seed=42)

    assert balanced.size == 4
    assert int(np.sum(labels[balanced] == 1)) == 2
    assert int(np.sum(labels[balanced] == 0)) == 2
    assert float(np.mean(labels)) == 0.25


def test_group_split_holds_out_complete_time_slices():
    edges, labels, times = _balanced_edges_for_times(4)
    indices, metadata = _build_leakage_free_split(
        edges,
        labels,
        times,
        [8, 8, 8, 8],
        split_strategy="group_or_node",
        random_seed=42,
        validation_ratio=0.1,
        test_ratio=0.2,
    )

    assert metadata["strategy"] == "time_group_holdout"
    split_times = {name: set(times[index].tolist()) for name, index in indices.items()}
    assert split_times["train"].isdisjoint(split_times["validation"])
    assert split_times["train"].isdisjoint(split_times["test"])
    assert split_times["validation"].isdisjoint(split_times["test"])
    assert "not future-time" in metadata["interpretation"]


def test_node_split_uses_every_observed_time_in_production_partitions():
    edges, labels, times = _balanced_edges_for_times(3, n_nodes=18)
    indices, metadata = _build_leakage_free_split(
        edges,
        labels,
        times,
        [18, 18, 18],
        split_strategy="node_disjoint",
        random_seed=42,
        validation_ratio=0.2,
        test_ratio=0.2,
    )

    assert metadata["strategy"] == "node_disjoint_holdout"
    for split_name in ("train", "validation", "test"):
        assert set(times[indices[split_name]].tolist()) == {0, 1, 2}


def test_node_split_has_no_node_overlap():
    edges, labels, times = _balanced_edges_for_times(1, n_nodes=12)
    indices, metadata = _build_leakage_free_split(
        edges,
        labels,
        times,
        [12],
        split_strategy="node_disjoint",
        random_seed=7,
        validation_ratio=0.25,
        test_ratio=0.25,
    )

    assert metadata["strategy"] == "node_disjoint_holdout"
    nodes = {
        name: set(edges[index].reshape(-1).tolist()) for name, index in indices.items()
    }
    assert nodes["train"].isdisjoint(nodes["validation"])
    assert nodes["train"].isdisjoint(nodes["test"])
    assert nodes["validation"].isdisjoint(nodes["test"])
    assert metadata["cross_split_edges_excluded"] > 0


def test_positive_edges_must_belong_to_radius_candidate_universe():
    coordinates = np.asarray([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]])
    _validate_positive_spatial_contract(
        np.asarray([[0, 1], [1, 0]], dtype=np.int32),
        coordinates,
        1.0,
        time_value=0.0,
    )

    for invalid in (
        np.asarray([[0, 0]], dtype=np.int32),
        np.asarray([[0, 2]], dtype=np.int32),
    ):
        try:
            _validate_positive_spatial_contract(
                invalid, coordinates, 1.0, time_value=0.0
            )
        except ValueError as error:
            assert "same spatial candidate universe" in str(error)
        else:
            raise AssertionError("invalid positive edge passed the spatial contract")


def test_exact_f1_threshold_and_sparse_metrics_are_reported():
    labels = np.asarray([1, 0, 1, 0, 0, 0], dtype=np.int64)
    probabilities = np.asarray([0.9, 0.8, 0.7, 0.6, 0.2, 0.1])

    threshold, best_f1, _accuracy = _find_best_threshold(labels, probabilities)
    metrics = _binary_metrics(labels, probabilities, threshold)

    assert threshold == 0.7
    assert np.isclose(best_f1, 0.8)
    assert metrics["positive_fraction"] == 2 / 6
    assert metrics["precision"] == 2 / 3
    assert metrics["recall"] == 1.0
    assert metrics["predicted_edge_fraction"] == 3 / 6
    assert metrics["average_precision"] > metrics["positive_fraction"]
