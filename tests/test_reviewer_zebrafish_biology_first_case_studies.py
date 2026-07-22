from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys

import anndata as ad
import numpy as np
import pandas as pd
import pytest


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from reviewer_zebrafish_ccc import biology_first_case_studies as SCREEN  # noqa: E402
from reviewer_zebrafish_ccc import delta_notch_case_study as NOTCH  # noqa: E402


def test_exact_circuit_density_averages_seeds_then_uses_possible_pair_denominator() -> None:
    edges = pd.DataFrame(
        [
            {
                "stage": 1.0,
                "stage_label": "10hpf",
                "grouping_seed": 101,
                "source_index": 0,
                "target_index": 2,
                "sender_type": "A",
                "receiver_type": "B",
                "attention_abs_mean": 2.0,
                "edge_message_norm_joint": 4.0,
                "spatial_distance": 0.1,
            },
            {
                "stage": 1.0,
                "stage_label": "10hpf",
                "grouping_seed": 202,
                "source_index": 0,
                "target_index": 2,
                "sender_type": "A",
                "receiver_type": "B",
                "attention_abs_mean": 6.0,
                "edge_message_norm_joint": 8.0,
                "spatial_distance": 0.2,
            },
        ]
    )
    axes = pd.DataFrame(
        {
            "ligand": ["lig"],
            "receptor": ["rec"],
            "axis_id": ["lig->rec"],
            "evidence_scope": ["independent biology"],
            "claim_guardrail": ["supportive only"],
        }
    )
    activities = {
        "lig": np.asarray([1.0, 0.0, 0.0, 0.0, 0.0]),
        "rec": np.asarray([0.0, 0.0, 1.0, 0.0, 0.0]),
    }
    counts = pd.DataFrame(
        {
            "stage": [1.0, 1.0],
            "cell_type": ["A", "B"],
            "n_cells": [2, 3],
        }
    )
    expression = pd.DataFrame(
        [
            {
                "stage": 1.0,
                "cell_type": "A",
                "axis_id": "lig->rec",
                "role": "ligand",
                "mean_scaled_expression": 0.5,
                "positive_fraction": 0.5,
                "n_cells": 2,
            },
            {
                "stage": 1.0,
                "cell_type": "B",
                "axis_id": "lig->rec",
                "role": "receptor",
                "mean_scaled_expression": 1.0 / 3.0,
                "positive_fraction": 1.0 / 3.0,
                "n_cells": 3,
            },
        ]
    )

    result = SCREEN.score_exact_circuits(
        edges, axes, activities, counts, expression
    )
    row = result.loc[
        result["sender_type"].eq("A") & result["receiver_type"].eq("B")
    ].iloc[0]

    assert row.n_grouping_seeds == 2
    assert row.n_possible_distinct_cell_pairs == 6
    assert row.n_active_occurrences == 2
    assert row.n_active_unique_edges == 1
    assert row.attention_lr_sum_occurrences == 8.0
    assert row.exact_message_lr_sum_occurrences == 12.0
    assert row.cytobridge_attention_lr_density == pytest.approx(8.0 / 2.0 / 6.0)
    assert row.cytobridge_exact_message_lr_density == pytest.approx(
        12.0 / 2.0 / 6.0
    )
    assert row.cytobridge_lr_only_density == pytest.approx(2.0 / 2.0 / 6.0)


def test_homotypic_density_excludes_self_pairs_from_denominator() -> None:
    edges = pd.DataFrame(
        [
            {
                "stage": 1.0,
                "stage_label": "10hpf",
                "grouping_seed": 101,
                "source_index": 0,
                "target_index": 1,
                "sender_type": "A",
                "receiver_type": "A",
                "attention_abs_mean": 2.0,
                "edge_message_norm_joint": 3.0,
                "spatial_distance": 0.1,
            }
        ]
    )
    axes = pd.DataFrame(
        {
            "ligand": ["lig"],
            "receptor": ["rec"],
            "axis_id": ["lig->rec"],
            "evidence_scope": ["biology"],
            "claim_guardrail": ["observational"],
        }
    )
    activities = {"lig": np.ones(3), "rec": np.ones(3)}
    counts = pd.DataFrame({"stage": [1.0], "cell_type": ["A"], "n_cells": [3]})
    expression = pd.DataFrame(
        [
            {
                "stage": 1.0,
                "cell_type": "A",
                "axis_id": "lig->rec",
                "role": role,
                "mean_scaled_expression": 1.0,
                "positive_fraction": 1.0,
                "n_cells": 3,
            }
            for role in ("ligand", "receptor")
        ]
    )

    row = SCREEN.score_exact_circuits(
        edges, axes, activities, counts, expression
    ).iloc[0]

    assert row.n_shared_sender_receiver_cells == 3
    assert row.n_possible_distinct_cell_pairs == 6
    assert row.cytobridge_attention_lr_density == pytest.approx(2.0 / 6.0)


def test_commot_duplicate_provenance_rows_must_be_numerically_identical(
    tmp_path: Path,
) -> None:
    path = tmp_path / "commot_lr.csv"
    pd.DataFrame(
        [
            {
                "stage": 4.0,
                "sender_type": "A",
                "receiver_type": "B",
                "ligand": "dla",
                "receptor": "notch1a",
                "score": 5.0,
                "abundance_controlled_score": 0.2,
                "score_distinct_cell_pairs": 4.0,
                "abundance_controlled_distinct_cell_score": 0.16,
                "n_possible_distinct_cell_pairs": 25,
                "database_rows": "9",
            },
            {
                "stage": 4.0,
                "sender_type": "A",
                "receiver_type": "B",
                "ligand": "dla",
                "receptor": "notch1a",
                "score": 5.0,
                "abundance_controlled_score": 0.2,
                "score_distinct_cell_pairs": 4.0,
                "abundance_controlled_distinct_cell_score": 0.16,
                "n_possible_distinct_cell_pairs": 25,
                "database_rows": "4",
            },
        ]
    ).to_csv(path, index=False)

    result = SCREEN.collapse_commot(path)

    assert len(result) == 1
    assert result.loc[0, "commot_total_flow"] == 4.0
    assert result.loc[0, "commot_abundance_controlled_score"] == 0.16
    assert result.loc[0, "commot_database_rows"] == "4;9"

    conflicting = pd.read_csv(path)
    conflicting.loc[1, "score_distinct_cell_pairs"] = 4.5
    conflicting.to_csv(path, index=False)
    with pytest.raises(ValueError, match="duplicate provenance rows disagree"):
        SCREEN.collapse_commot(path)


def test_support_filtered_competition_ranks_and_structural_zero_completion() -> None:
    contexts = [("A", "A"), ("A", "B"), ("B", "A"), ("B", "B")]
    cytobridge = pd.DataFrame(
        [
            {
                "stage": 4.0,
                "axis_id": "dla->notch1a",
                "ligand": "dla",
                "receptor": "notch1a",
                "sender_type": sender,
                "receiver_type": receiver,
                "n_active_unique_edges": active,
                "n_sender_cells": 10,
                "n_receiver_cells": 10,
                "n_possible_distinct_cell_pairs": 90 if sender == receiver else 100,
                "cytobridge_attention_lr_density": score,
                "cytobridge_exact_message_lr_density": score,
                "cytobridge_lr_only_density": score,
            }
            for (sender, receiver), score, active in zip(
                contexts, [10.0, 5.0, 5.0, 100.0], [5, 5, 5, 1]
            )
        ]
    )
    commot = pd.DataFrame(
        [
            {
                "stage": 4.0,
                "axis_id": "dla->notch1a",
                "ligand": "dla",
                "receptor": "notch1a",
                "sender_type": "A",
                "receiver_type": "A",
                "commot_total_flow": 8.0,
                "commot_total_flow_min": 8.0,
                "commot_abundance_controlled_score": 8.0 / 90.0,
                "commot_abundance_controlled_score_min": 8.0 / 90.0,
                "commot_n_possible_distinct_cell_pairs": 90,
                "commot_n_possible_distinct_cell_pairs_min": 90,
                "commot_database_rows": "1",
            }
        ]
    )
    availability = pd.DataFrame(
        {
            "stage": [4.0],
            "axis_id": ["dla->notch1a"],
            "ligand": ["dla"],
            "receptor": ["notch1a"],
            "commot_axis_stage_available": [True],
            "commot_matrix_keys": ["commot-test-dla-notch1a"],
        }
    )

    ranked = SCREEN.join_and_rank(
        cytobridge,
        commot,
        availability,
        min_active_edges=5,
        min_cells_per_side=10,
    )
    evaluated = ranked.loc[ranked["is_evaluated_context"]].set_index(
        ["sender_type", "receiver_type"]
    )

    assert len(evaluated) == 3
    assert evaluated["n_evaluated_contexts"].eq(3).all()
    assert evaluated.loc[("A", "A"), "attention_context_rank_from_top"] == 1
    assert evaluated.loc[("A", "B"), "attention_context_rank_from_top"] == 2
    assert evaluated.loc[("B", "A"), "attention_context_rank_from_top"] == 2
    assert evaluated.loc[("A", "B"), "attention_context_tie_count"] == 2
    assert evaluated.loc[("A", "B"), "attention_context_rank_fraction"] == pytest.approx(2 / 3)
    assert evaluated.loc[("A", "B"), "attention_context_top_percent"] == pytest.approx(100 / 3)
    assert evaluated.loc[("B", "A"), "commot_abundance_controlled_score"] == 0.0
    assert bool(
        evaluated.loc[
            ("B", "A"), "commot_context_was_sparse_structural_zero"
        ]
    )

    unavailable_cb = cytobridge.iloc[[0]].assign(
        axis_id="dld->notch3", ligand="dld", receptor="notch3"
    )
    unavailable = SCREEN.join_and_rank(
        unavailable_cb,
        commot.iloc[0:0],
        pd.DataFrame(
            {
                "stage": [4.0],
                "axis_id": ["dld->notch3"],
                "ligand": ["dld"],
                "receptor": ["notch3"],
                "commot_axis_stage_available": [False],
                "commot_matrix_keys": ["commot-test-dld-notch3"],
            }
        ),
        min_active_edges=5,
        min_cells_per_side=10,
    ).iloc[0]
    assert not bool(unavailable.is_evaluated_context)
    assert np.isnan(unavailable.commot_abundance_controlled_score)
    assert np.isnan(unavailable.attention_context_rank_from_top)

    with pytest.raises(ValueError, match="score rows exist"):
        SCREEN.join_and_rank(
            cytobridge.iloc[[0]],
            commot,
            availability.assign(commot_axis_stage_available=False),
            min_active_edges=5,
            min_cells_per_side=10,
        )


def _mapping_data() -> ad.AnnData:
    obs = pd.DataFrame(
        {
            "time_point_processed": [4.0] * 8,
            "time": ["24hpf"] * 8,
            "Annotation": ["A"] * 4 + ["B"] * 4,
        },
        index=[f"cell_{index}" for index in range(8)],
    )
    return ad.AnnData(
        X=np.zeros((8, 1), dtype=np.float32),
        obs=obs,
        var=pd.DataFrame(index=["dummy"]),
    )


def _write_observed_cells(path: Path, data: ad.AnnData) -> Path:
    observed = pd.DataFrame(
        {
            "global_index": np.arange(data.n_obs, dtype=int),
            "obs_name": data.obs_names.astype(str),
            "stage": data.obs["time_point_processed"].to_numpy(float),
            "stage_label": data.obs["time"].astype(str).to_numpy(),
            "cell_type": data.obs["Annotation"].astype(str).to_numpy(),
        }
    )
    observed.to_csv(path, index=False)
    return path


def _write_stage_seed(
    stage_dir: Path, seed: int, edge_output: np.ndarray
) -> None:
    n_rows = int(edge_output.shape[0])
    frame = pd.DataFrame(
        {
            "stage": [4.0] * n_rows,
            "stage_label": ["24hpf"] * n_rows,
            "grouping_seed": [seed] * n_rows,
            "source_index": np.arange(n_rows, dtype=int),
            "target_index": np.arange(4, 4 + n_rows, dtype=int),
            "source_index_stage": np.arange(n_rows, dtype=int),
            "target_index_stage": np.arange(4, 4 + n_rows, dtype=int),
            "sender_type": ["A"] * n_rows,
            "receiver_type": ["B"] * n_rows,
            "attention_abs_mean": np.linspace(1.0, 2.0, n_rows),
            "edge_message_norm_joint": np.linalg.norm(edge_output, axis=1),
            "spatial_distance": np.linspace(0.1, 0.2, n_rows),
            "edge_predictor_probability": [0.9] * n_rows,
            "source_mass_fraction": [0.5] * n_rows,
        }
    )
    frame.to_csv(stage_dir / f"edges_seed_{seed}.csv.gz", index=False)
    np.savez_compressed(
        stage_dir / f"exact_arrays_seed_{seed}.npz",
        edge_output=edge_output,
        global_indices=np.arange(8, dtype=int),
    )


def test_exact_arrays_remain_row_aligned_with_each_seed_edge_table(
    tmp_path: Path,
) -> None:
    stage_dir = tmp_path / "stage_4_24hpf"
    stage_dir.mkdir()
    arrays_expected = {
        101: np.asarray([[0.0, 0.0, 1.0, 2.0], [0.0, 0.0, 3.0, 4.0]]),
        202: np.asarray([[0.0, 0.0, 5.0, 6.0], [0.0, 0.0, 7.0, 8.0]]),
    }
    for seed, edge_output in arrays_expected.items():
        _write_stage_seed(stage_dir, seed, edge_output)

    data = _mapping_data()
    observed = _write_observed_cells(tmp_path / "observed_cells.csv.gz", data)
    edges, arrays, inventory, resolution = NOTCH.load_stage_edges(
        tmp_path, 4.0, "24hpf", data=data, observed_cells=observed
    )

    assert set(arrays) == {101, 202}
    assert len(inventory) == 2
    assert resolution["global_index_order_assumed_without_validation"] is False
    for seed, expected in arrays_expected.items():
        np.testing.assert_array_equal(arrays[seed], expected)
        assert edges.loc[
            edges["grouping_seed"].eq(seed), "edge_row_within_seed"
        ].tolist() == [0, 1]

    annotated = NOTCH.annotate_edges(
        edges,
        arrays,
        ligand_activity=np.ones(8),
        receptor_activity=np.ones(8),
        receiver_fold={4: 0, 5: 0},
        directions={0: (np.asarray([1.0, 10.0]), 0.0)},
        spatial_dim=2,
    )
    np.testing.assert_allclose(
        annotated["response_direction_projection"], [21.0, 43.0, 65.0, 87.0]
    )


def test_stage_edge_loader_rejects_edge_array_row_mismatch(tmp_path: Path) -> None:
    stage_dir = tmp_path / "stage_4_24hpf"
    stage_dir.mkdir()
    _write_stage_seed(
        stage_dir,
        101,
        np.asarray([[0.0, 0.0, 1.0, 2.0], [0.0, 0.0, 3.0, 4.0]]),
    )
    np.savez_compressed(
        stage_dir / "exact_arrays_seed_101.npz",
        edge_output=np.asarray([[0.0, 0.0, 1.0, 2.0]]),
        global_indices=np.arange(8, dtype=int),
    )
    data = _mapping_data()
    observed = _write_observed_cells(tmp_path / "observed_cells.csv.gz", data)

    with pytest.raises(ValueError, match="Edge/array row mismatch for seed 101"):
        NOTCH.load_stage_edges(
            tmp_path, 4.0, "24hpf", data=data, observed_cells=observed
        )


def test_screen_and_case_map_attribution_indices_by_obs_name_after_reorder(
    tmp_path: Path,
) -> None:
    stage_dir = tmp_path / "stage_4_24hpf"
    stage_dir.mkdir()
    _write_stage_seed(
        stage_dir,
        101,
        np.asarray([[0.0, 0.0, 1.0, 2.0], [0.0, 0.0, 3.0, 4.0]]),
    )
    original = _mapping_data()
    observed = _write_observed_cells(tmp_path / "observed_cells.csv.gz", original)
    reordered = original[[1, 0, 2, 3, 4, 5, 6, 7]].copy()

    screen_edges, _, screen_resolution = SCREEN.load_edges(
        tmp_path, reordered, observed
    )
    case_edges, _, _, case_resolution = NOTCH.load_stage_edges(
        tmp_path, 4.0, "24hpf", data=reordered, observed_cells=observed
    )

    assert screen_edges.loc[0, "source_index_attribution"] == 0
    assert screen_edges.loc[0, "source_index"] == 1
    assert case_edges.loc[0, "source_index_attribution"] == 0
    assert case_edges.loc[0, "source_index"] == 1
    assert screen_resolution["global_index_order_assumed_without_validation"] is False
    assert case_resolution["global_index_order_assumed_without_validation"] is False


def test_crossfit_response_directions_have_state_shape_and_project_messages() -> None:
    rng = np.random.default_rng(19)
    state = rng.normal(size=(30, 4))
    response = 1.5 * state[:, 0] - 0.75 * state[:, 1] + 0.2 * state[:, 3]
    receiver_indices = np.arange(5, 25, dtype=int)

    crossfit, directions = NOTCH.crossfit_response_direction(
        state, response, receiver_indices, seed=37, n_splits=5
    )

    assert set(crossfit["global_index"]) == set(receiver_indices)
    assert set(crossfit["fold"]) == set(range(5))
    assert set(directions) == set(range(5))
    assert np.isfinite(crossfit["predicted_response"]).all()
    for row in crossfit.itertuples(index=False):
        coefficient, intercept = directions[int(row.fold)]
        assert coefficient.shape == (state.shape[1],)
        assert np.isfinite(coefficient).all()
        expected = state[int(row.global_index)] @ coefficient + intercept
        assert row.predicted_response == pytest.approx(expected)

        exact_edge_output = np.asarray(
            [[0.0, 0.0, 0.1, 0.2, 0.3, 0.4]], dtype=float
        )
        projection = exact_edge_output[:, 2:] @ coefficient
        assert projection.shape == (1,)
        assert np.isfinite(projection).all()


def test_matched_projection_null_collapses_seeds_and_aggregates_by_receiver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        # One selected directed edge repeated across two technical seeds.
        (101, 0, 10, 1.0, 1.0),
        (202, 0, 10, 3.0, 1.0),
        # A second selected edge to the same receiver.
        (101, 1, 10, 2.0, 1.0),
        # One repeated non-compatible control to receiver 10.
        (101, 2, 10, -1.0, 0.0),
        (202, 2, 10, -3.0, 0.0),
        # A selected/control pair to a second receiver.
        (101, 3, 11, 10.0, 1.0),
        (101, 4, 11, 0.0, 0.0),
    ]
    annotated = pd.DataFrame(
        [
            {
                "grouping_seed": seed,
                "source_index": source,
                "target_index": receiver,
                "sender_type": "A",
                "receiver_type": "B",
                "response_direction_projection": projection,
                "lr_activity": lr,
                "spatial_distance": 1.0,
                "edge_predictor_probability": 0.9,
                "source_mass_fraction": 0.5,
                "edge_message_norm_joint": 2.0,
            }
            for seed, source, receiver, projection, lr in rows
        ]
    )

    class RecordingGenerator:
        def __init__(self) -> None:
            self.calls: list[tuple[tuple[float, ...], int, bool]] = []

        def choice(self, values, *, size: int, replace: bool):
            array = np.asarray(values, dtype=float)
            self.calls.append((tuple(array.tolist()), int(size), bool(replace)))
            return array[: int(size)]

    recorder = RecordingGenerator()
    monkeypatch.setattr(NOTCH.np.random, "default_rng", lambda seed: recorder)

    null, summary, receiver_summary = NOTCH.matched_projection_null(
        annotated,
        sender_type="A",
        receiver_type="B",
        n_permutations=2,
        seed=91,
    )

    expected_calls = Counter(
        {
            ((-2.0,), 1, True): 4,
            ((0.0,), 1, True): 2,
        }
    )
    assert Counter(recorder.calls) == expected_calls
    assert len(null) == 2
    assert summary["n_lr_compatible_edge_occurrences"] == 4
    assert summary["n_lr_compatible_unique_edges"] == 3
    assert summary["n_lr_compatible_unique_edges_matched"] == 3
    assert summary["n_lr_compatible_unique_edges_dropped_no_control"] == 0
    assert summary["n_matched_receiver_clusters"] == 2
    assert summary["n_all_same_circuit_edge_occurrences"] == 7
    assert summary["n_all_same_circuit_unique_edges"] == 5
    assert summary["observed_mean_projection"] == pytest.approx(6.0)
    assert summary["null_mean"] == pytest.approx(-1.0)
    assert len(receiver_summary) == 2
