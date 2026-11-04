from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from CytoBridge.pp import AlignConfig


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_mosta_alignment_sensitivity.py"
SPEC = importlib.util.spec_from_file_location("mosta_alignment_sensitivity", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_uncapped_config_preserves_formal_values_and_only_removes_fit_cap():
    config = AlignConfig(
        phase1_epochs=17,
        phase2_epochs=9,
        random_seed=123,
        max_cells_per_timepoint=20000,
        input_spatial_key="spatial",
        shared_scale=None,
    )
    stored = {
        "config": {
            key: ("none" if value is None else value)
            for key, value in config.__dict__.items()
        }
    }
    uncapped = MODULE._uncapped_config(stored, random_seed=None)

    assert uncapped.phase1_epochs == 17
    assert uncapped.phase2_epochs == 9
    assert uncapped.random_seed == 123
    assert uncapped.shared_scale is None
    assert uncapped.max_cells_per_timepoint is None
    assert uncapped.input_spatial_key == "spatial_original"


def test_orthogonal_procrustes_recovers_rigid_transform_without_scaling():
    rng = np.random.default_rng(11)
    reference = rng.normal(size=(100, 2))
    theta = 0.63
    rotation = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    candidate = reference @ rotation + np.array([3.0, -7.0])

    aligned, metadata = MODULE._orthogonal_procrustes(reference, candidate)

    np.testing.assert_allclose(aligned, reference, atol=1e-10)
    assert np.isclose(abs(metadata["rotation_determinant"]), 1.0)


def test_equal_group_weights_prevent_large_timepoint_from_dominating():
    groups = np.asarray([0] * 90 + [1] * 10)
    weights = MODULE._equal_group_weights(groups)

    assert weights.sum() == pytest.approx(1.0)
    assert weights[groups == 0].sum() == pytest.approx(0.5)
    assert weights[groups == 1].sum() == pytest.approx(0.5)

    rng = np.random.default_rng(71)
    reference = rng.normal(size=(100, 2))
    theta = -0.41
    rotation = np.asarray(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    candidate = reference @ rotation + np.asarray([-4.0, 3.0])
    aligned, metadata = MODULE._orthogonal_procrustes(
        reference,
        candidate,
        weights=weights,
    )
    np.testing.assert_allclose(aligned, reference, atol=1e-10)
    assert metadata["weighting"] == "caller_supplied"


def test_knn_jaccard_is_one_for_rigidly_equivalent_coordinates():
    rng = np.random.default_rng(19)
    reference = rng.normal(size=(60, 2))
    candidate = reference[:, ::-1] + np.array([5.0, 2.0])

    scores = MODULE._knn_jaccard(reference, candidate, k=10)

    np.testing.assert_allclose(scores, 1.0)


def test_knn_explicitly_excludes_self_when_coordinates_have_ties():
    coordinates = np.asarray(
        [[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [2.0, 2.0]]
    )
    neighbors = MODULE._knn_indices(coordinates, k=2)

    assert neighbors.shape == (5, 2)
    assert not np.any(neighbors == np.arange(5)[:, None])
    assert all(len(set(row)) == 2 for row in neighbors.tolist())
    np.testing.assert_allclose(
        MODULE._knn_jaccard(coordinates, coordinates, k=2),
        1.0,
    )


def test_fresh_threshold_matches_formal_nn1_formula():
    rows = pd.DataFrame(
        {
            "capped_nn1_median": [1.0, 2.0, 3.0, 4.0],
            "uncapped_nn1_median": [2.0, 3.0, 4.0, 5.0],
        }
    )

    assert np.isclose(MODULE._fresh_threshold(rows, "capped"), 12.0)
    assert np.isclose(MODULE._fresh_threshold(rows, "uncapped"), 16.8)
    assert np.isclose(
        MODULE._fresh_threshold(
            rows,
            "capped",
            recommended_spot_scale=2.0,
            neighborhood_factor=3.0,
        ),
        15.0,
    )


def test_cutoff_contract_requires_stored_formula_inputs():
    assert MODULE._stored_threshold_contract(
        {
            "recommended_spot_scale": 1.2,
            "neighborhood_factor": 4.0,
            "neighborhood_threshold": 0.137,
        }
    ) == pytest.approx((1.2, 4.0, 0.137))
    with pytest.raises(ValueError, match="lacks cutoff provenance"):
        MODULE._stored_threshold_contract({"neighborhood_threshold": 0.137})


def test_acceptance_requires_each_timepoint_not_only_pooled_metrics():
    stage_metrics = pd.DataFrame(
        {
            "knn_jaccard_median": [1.0, 0.80],
            "nn1_median_relative_difference": [0.0, 0.20],
            "displacement_over_capped_nn1_median": [0.0, 0.80],
            "capped_zero_nn1_count": [0, 0],
            "uncapped_zero_nn1_count": [0, 0],
        }
    )
    # Cell-pooled medians pass because a large well-aligned stage dominates.
    gates = MODULE._acceptance_gates(
        stage_metrics,
        pooled_jaccard=np.asarray([1.0] * 100 + [0.8] * 5),
        pooled_displacement_ratio=np.asarray([0.0] * 100 + [0.8] * 5),
        cutoff_relative_difference=0.0,
        stored_cutoff_relative_error=0.0,
    )
    assert gates["pooled_median_knn_jaccard_at_least_0.90"]
    assert gates["pooled_median_displacement_over_nn1_at_most_0.5"]
    assert not gates["every_timepoint_median_knn_jaccard_at_least_0.90"]
    assert not gates[
        "every_timepoint_nn1_median_relative_difference_at_most_0.05"
    ]
    assert not gates[
        "every_timepoint_median_displacement_over_nn1_at_most_0.5"
    ]


def test_launchers_share_gpu_lock_and_refuse_partial_or_dirty_runs():
    root = SCRIPT.parents[1]
    sensitivity = (root / "scripts" / "launch_mosta_alignment_sensitivity.sh").read_text()
    full = (root / "scripts" / "launch_mosta_formal_full_alignment.sh").read_text()

    assert 'locks/gpu${GPU_INDEX}.lock' in sensitivity
    assert 'locks/gpu${GPU_ID}.lock' in full
    assert "--no-optional-locks" in sensitivity
    assert "status --porcelain --untracked-files=all" in sensitivity
    assert "--no-optional-locks" in full
    assert "status --porcelain --untracked-files=all" in full
    for partial in ("input_contract", "preprocess", "training", "evaluation", "downstream"):
        assert f'${{RUN}}/{partial}' in full
