from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
import pytest

from CytoBridge.pl import plot_developmental_wave_heatmap
from CytoBridge.tl import (
    analyze_developmental_wave,
    make_gene_set_library,
    overrepresentation_analysis,
    project_velocity_to_embedding,
)


def _wave_profiles() -> pd.DataFrame:
    times = np.arange(5, dtype=float)
    rows: dict[str, np.ndarray] = {}
    for phase, peak in enumerate((0, 2, 4), start=1):
        for member in range(6):
            # Amplitude differences exercise deterministic variance tie breaks;
            # row standardization leaves the peak assignment unchanged.
            amplitude = 1.0 + 0.1 * member
            rows[f"p{phase}_g{member}"] = amplitude * np.exp(
                -0.5 * np.square((times - peak) / 0.55)
            )
    return pd.DataFrame.from_dict(rows, orient="index", columns=times)


def test_developmental_wave_peak_order_dp_and_standardization() -> None:
    profiles = _wave_profiles()
    result = analyze_developmental_wave(
        profiles,
        n_top_profiles=None,
        n_phases=3,
        min_phase_size=4,
    )
    assignments = result.assignments
    assert assignments.columns.tolist() == [
        "profile",
        "wave_rank",
        "temporal_variance",
        "peak_index",
        "peak_time",
        "phase",
    ]
    assert assignments["peak_time"].is_monotonic_increasing
    assert assignments.groupby("phase", sort=True).size().tolist() == [6, 6, 6]
    expected = {
        **{f"p1_g{i}": 1 for i in range(6)},
        **{f"p2_g{i}": 2 for i in range(6)},
        **{f"p3_g{i}": 3 for i in range(6)},
    }
    assert assignments.set_index("profile")["phase"].to_dict() == expected
    np.testing.assert_allclose(
        result.standardized_profiles.mean(axis=1).to_numpy(),
        0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        result.standardized_profiles.std(axis=1, ddof=0).to_numpy(),
        1.0,
        atol=1e-12,
    )
    assert set(result.prototypes.columns) == {
        "phase",
        "time",
        "mean",
        "std",
        "n_profiles",
    }
    assert result.prototypes.shape[0] == 3 * profiles.shape[1]
    assert result.diagnostics["within_peak_sse"].sum() == pytest.approx(0.0)
    assert result.diagnostics["total_within_peak_sse"].nunique() == 1


def test_developmental_wave_is_deterministic_under_row_and_time_permutation() -> None:
    profiles = _wave_profiles()
    expected = analyze_developmental_wave(
        profiles,
        n_top_profiles=15,
        n_phases=3,
        min_phase_size=3,
    )
    permuted = profiles.sample(frac=1.0, random_state=23).iloc[:, [4, 0, 3, 1, 2]]
    actual = analyze_developmental_wave(
        permuted,
        n_top_profiles=15,
        n_phases=3,
        min_phase_size=3,
    )
    pd.testing.assert_frame_equal(expected.assignments, actual.assignments)
    pd.testing.assert_frame_equal(expected.ordered_profiles, actual.ordered_profiles)
    pd.testing.assert_frame_equal(expected.diagnostics, actual.diagnostics)
    assert expected.settings == actual.settings
    assert len(expected.settings["phase_boundaries"]) == 3


def test_developmental_wave_dp_objective_is_globally_optimal() -> None:
    result = analyze_developmental_wave(
        _wave_profiles(),
        n_top_profiles=12,
        n_phases=3,
        min_phase_size=2,
    )
    peaks = result.assignments["peak_time"].to_numpy(dtype=float)
    observed = float(result.diagnostics["total_within_peak_sse"].iloc[0])
    brute_force = np.inf
    n = len(peaks)
    for first, second in combinations(range(2, n - 1), 2):
        if first < 2 or second - first < 2 or n - second < 2:
            continue
        score = 0.0
        for start, end in ((0, first), (first, second), (second, n)):
            segment = peaks[start:end]
            score += float(np.square(segment - segment.mean()).sum())
        brute_force = min(brute_force, score)
    assert observed == pytest.approx(brute_force, abs=1e-12)


def test_developmental_wave_errors_and_plot(tmp_path) -> None:
    profiles = _wave_profiles()
    with pytest.raises(ValueError, match="Cannot divide"):
        analyze_developmental_wave(
            profiles.iloc[:8],
            n_phases=3,
            min_phase_size=3,
        )
    bad = profiles.iloc[:6].copy()
    bad.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        analyze_developmental_wave(bad, n_phases=2, min_phase_size=2)
    duplicate_times = profiles.iloc[:6].copy()
    duplicate_times.columns = [0, 0.0, 1, 2, 3]
    with pytest.raises(ValueError, match="unique numeric"):
        analyze_developmental_wave(
            duplicate_times,
            n_phases=2,
            min_phase_size=2,
        )

    result = analyze_developmental_wave(
        profiles,
        n_phases=3,
        min_phase_size=4,
    )
    output = tmp_path / "developmental_wave.png"
    returned = plot_developmental_wave_heatmap(result, out_path=output)
    assert returned == output.resolve()
    assert output.stat().st_size > 1000


def _projection_fixture():
    coordinates = np.asarray(
        [[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]],
        dtype=float,
    )
    embedding = coordinates.copy()
    velocity = np.tile([1.0, 0.0], (len(coordinates), 1))
    neighbors = np.asarray(
        [
            [j for j in range(len(coordinates)) if j != i]
            for i in range(len(coordinates))
        ],
        dtype=int,
    )
    return coordinates, velocity, embedding, neighbors


def test_velocity_projection_matches_cosine_softmax_definition() -> None:
    coordinates, velocity, embedding, neighbors = _projection_fixture()
    result = project_velocity_to_embedding(
        coordinates,
        velocity,
        embedding,
        neighbor_indices=neighbors,
        temperature=1.0,
    )
    expected_probability = np.exp([1.0, -1.0, 0.0, 0.0])
    expected_probability /= expected_probability.sum()
    np.testing.assert_allclose(
        result.transition_probabilities[0],
        expected_probability,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        result.projected_velocity[0],
        [expected_probability[0] - expected_probability[1], 0.0],
        atol=1e-12,
    )
    np.testing.assert_allclose(result.transition_probabilities.sum(axis=1), 1.0)
    np.testing.assert_allclose(result.transition_weights.sum(axis=1), 0.0, atol=1e-15)
    target_displacements = embedding[neighbors] - embedding[:, None, :]
    target_norm = np.linalg.norm(target_displacements, axis=2, keepdims=True)
    target_displacements = target_displacements / target_norm
    np.testing.assert_allclose(
        result.projected_velocity,
        np.einsum("nk,nkd->nd", result.transition_weights, target_displacements),
        atol=1e-12,
    )
    assert result.diagnostics["neighbor_source"] == "caller_supplied"
    assert result.diagnostics["n_neighbors_parameter_used"] is False


def test_velocity_projection_scale_translation_and_rotation_invariants() -> None:
    coordinates, velocity, embedding, neighbors = _projection_fixture()
    reference = project_velocity_to_embedding(
        coordinates,
        velocity,
        embedding,
        neighbor_indices=neighbors,
    )
    scaled_velocity = project_velocity_to_embedding(
        coordinates,
        velocity * 37.0,
        embedding,
        neighbor_indices=neighbors,
    )
    np.testing.assert_allclose(
        scaled_velocity.projected_velocity,
        reference.projected_velocity,
        atol=1e-12,
    )

    angle = 0.63
    rotation = np.asarray(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    transformed_embedding = 8.0 * (embedding @ rotation) + np.asarray([12.0, -7.0])
    transformed = project_velocity_to_embedding(
        coordinates,
        velocity,
        transformed_embedding,
        neighbor_indices=neighbors,
    )
    np.testing.assert_allclose(
        transformed.projected_velocity,
        reference.projected_velocity @ rotation,
        atol=1e-12,
    )


def test_velocity_projection_zero_velocity_and_latent_knn_contract() -> None:
    coordinates = np.asarray([[0.0], [0.1], [2.0], [5.0]])
    velocity = np.zeros_like(coordinates)
    # Embedding proximity deliberately contradicts latent proximity.
    embedding = np.asarray([[0, 0], [100, 0], [0.01, 0], [0.02, 0]], dtype=float)
    result = project_velocity_to_embedding(
        coordinates,
        velocity,
        embedding,
        n_neighbors=1,
    )
    assert result.neighbor_indices[0, 0] == 1
    np.testing.assert_array_equal(result.projected_velocity, np.zeros((4, 2)))
    np.testing.assert_array_equal(result.transition_probabilities, np.zeros((4, 1)))
    np.testing.assert_array_equal(result.transition_weights, np.zeros((4, 1)))
    np.testing.assert_array_equal(result.cosine_similarities, np.zeros((4, 1)))
    assert result.diagnostics["neighbor_source"] == "knn_latent_coordinates"
    assert result.diagnostics["knn_backend"] == (
        "scipy.spatial.cKDTree_exact_boundary_ties"
    )
    assert result.diagnostics["n_neighbors_parameter_used"] is True
    assert result.diagnostics["n_zero_velocity"] == 4
    assert result.diagnostics["n_rows_without_transition"] == 4

    uncentered = project_velocity_to_embedding(
        coordinates,
        velocity,
        embedding,
        n_neighbors=1,
        center_probabilities=False,
    )
    np.testing.assert_array_equal(
        uncentered.transition_probabilities,
        np.zeros((4, 1)),
    )
    np.testing.assert_array_equal(uncentered.transition_weights, np.zeros((4, 1)))
    np.testing.assert_array_equal(uncentered.projected_velocity, np.zeros((4, 2)))


def test_velocity_projection_knn_breaks_exact_distance_ties_by_cell_index() -> None:
    coordinates = np.asarray([[0.0], [1.0], [-1.0], [3.0]])
    result = project_velocity_to_embedding(
        coordinates,
        np.ones_like(coordinates),
        np.column_stack((coordinates[:, 0], np.zeros(4))),
        n_neighbors=1,
    )

    assert result.neighbor_indices[0, 0] == 1
    assert not np.any(
        result.neighbor_indices == np.arange(len(coordinates), dtype=int)[:, None]
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"temperature": 0.0}, "temperature"),
        ({"neighbor_indices": np.asarray([[0], [0], [0], [0], [0]])}, "self"),
        (
            {"neighbor_indices": np.asarray([[1, 1], [0, 2], [0, 1], [0, 1], [0, 1]])},
            "duplicate",
        ),
    ],
)
def test_velocity_projection_rejects_invalid_contracts(kwargs, message) -> None:
    coordinates, velocity, embedding, _ = _projection_fixture()
    with pytest.raises(ValueError, match=message):
        project_velocity_to_embedding(
            coordinates,
            velocity,
            embedding,
            **kwargs,
        )


def test_existing_ora_is_available_from_public_tool_namespace() -> None:
    library = make_gene_set_library(
        {"signal": ["A", "B", "C"], "other": ["D", "E", "F"]}
    )
    result = overrepresentation_analysis(
        ["A", "B"],
        library,
        background_genes=list("ABCDEF"),
        min_set_size=2,
    )
    assert result.loc[0, "term"] == "signal"
    assert result.loc[0, "overlap_count"] == 2
