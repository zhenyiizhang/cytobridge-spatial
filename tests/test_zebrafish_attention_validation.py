from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from CytoBridge.zebrafish_attention_validation import (
    adaptive_pair_strata,
    build_pair_lr_activity_matrix,
    collapse_commot_lr_scores,
    collapse_lr_database,
    collapse_nichenet_lr_scores,
    complete_directed_pair_table,
    controlled_pair_concordance,
    external_ranks_for_selected_pairs,
    jointly_supported_lr_targets,
    lr_scores_from_pair_modifiers,
    modifier_permutation_test,
    pair_method_concordance,
    paper_reference_enrichment,
    positive_rank_weights,
    rank_metrics,
    scaled_expression_by_type,
    select_pairs_by_cytobridge_only,
    shared_lr_rank_metrics,
)


def _pair_table() -> pd.DataFrame:
    rows = []
    values = {
        ("A", "A"): (0.1, 0.2, 10, 10, 0.1),
        ("A", "B"): (0.9, 0.8, 10, 20, 0.2),
        ("B", "A"): (0.4, 0.3, 20, 10, 0.3),
        ("B", "B"): (0.2, 0.1, 20, 20, 0.4),
    }
    for (sender, receiver), (
        attention,
        message,
        n_sender,
        n_receiver,
        distance,
    ) in values.items():
        rows.append(
            {
                "sender_type": sender,
                "receiver_type": receiver,
                "G_AB_attention_mean_mean": attention,
                "D_AB_joint_mean": message,
                "n_sender_cells_mean": n_sender,
                "n_receiver_cells_mean": n_receiver,
                "spatial_distance_mean_mean": distance,
            }
        )
    return pd.DataFrame(rows)


def test_complete_grid_and_pair_concordance_include_zeros() -> None:
    cytobridge = _pair_table()
    external = pd.DataFrame(
        {
            "sender_type": ["A", "B"],
            "receiver_type": ["B", "A"],
            "score": [3.0, 1.0],
        }
    )
    complete = complete_directed_pair_table(external, score_column="score")
    assert len(complete) == 4
    assert sorted(complete.score) == [0.0, 0.0, 1.0, 3.0]
    metrics = pair_method_concordance(
        cytobridge,
        external,
        cytobridge_score="D_AB_joint_mean",
        external_score="score",
        top_fraction=0.5,
    )
    assert metrics.n_shared == 4
    assert metrics.spearman_rho > 0.8
    assert metrics.top_overlap_n == 1


def test_controlled_pair_concordance_is_deterministic() -> None:
    rows = []
    for sender_index, sender in enumerate(("A", "B", "C")):
        for receiver_index, receiver in enumerate(("A", "B", "C")):
            score = float((sender_index + 1) ** 2 + 0.7 * receiver_index)
            rows.append(
                {
                    "sender_type": sender,
                    "receiver_type": receiver,
                    "G_AB_attention_mean_mean": score,
                    "n_sender_cells_mean": 10 + sender_index,
                    "n_receiver_cells_mean": 12 + receiver_index,
                    "spatial_distance_mean_mean": abs(sender_index - receiver_index)
                    + 0.1 * sender_index,
                }
            )
    cytobridge = pd.DataFrame(rows)
    external = cytobridge[["sender_type", "receiver_type"]].assign(
        score=cytobridge["G_AB_attention_mean_mean"] ** 1.3
    )
    first = controlled_pair_concordance(
        cytobridge,
        external,
        cytobridge_score="G_AB_attention_mean_mean",
        external_score="score",
        permutations=20,
        seed=7,
    )
    second = controlled_pair_concordance(
        cytobridge,
        external,
        cytobridge_score="G_AB_attention_mean_mean",
        external_score="score",
        permutations=20,
        seed=7,
    )
    assert first == second
    assert first["n_pairs"] == 9
    assert first["adjusted_spearman_rho"] > 0.5


def test_cytobridge_selection_is_independent_of_external_scores() -> None:
    cytobridge = _pair_table()
    selected = select_pairs_by_cytobridge_only(
        cytobridge, score_column="D_AB_joint_mean", n_pairs=2
    )
    assert selected[["sender_type", "receiver_type"]].values.tolist() == [
        ["A", "B"],
        ["B", "A"],
    ]
    external_a = pd.DataFrame(
        {
            "sender_type": ["A", "A", "B", "B"],
            "receiver_type": ["A", "B", "A", "B"],
            "score": [100.0, 0.0, 1.0, 50.0],
        }
    )
    external_b = external_a.assign(score=[0.0, 100.0, 50.0, 1.0])
    result_a = external_ranks_for_selected_pairs(
        selected, external_a, external_score="score", external_method="method-a"
    )
    result_b = external_ranks_for_selected_pairs(
        selected, external_b, external_score="score", external_method="method-b"
    )
    assert result_a.cytobridge_selection_rank.tolist() == [1, 2]
    assert result_b.cytobridge_selection_rank.tolist() == [1, 2]
    assert not np.allclose(
        result_a.external_rank_percentile, result_b.external_rank_percentile
    )


def test_lr_mapping_uses_all_directed_pairs_and_keeps_lr_only_baseline() -> None:
    expression = pd.DataFrame(
        {
            "lig1": [10.0, 0.0, 5.0, 0.0],
            "rec1": [0.0, 6.0, 0.0, 3.0],
            "lig2": [0.0, 8.0, 0.0, 4.0],
            "rec2": [7.0, 0.0, 3.5, 0.0],
        }
    )
    labels = ["A", "B", "A", "B"]
    means = scaled_expression_by_type(expression, labels)
    database = collapse_lr_database(
        pd.DataFrame(
            {
                "ligand": ["lig1", "lig2", "missing"],
                "receptor": ["rec1", "rec2", "rec1"],
                "pathway": ["P1", "P2", "P3"],
                "category": ["secreted", "contact", "secreted"],
            }
        )
    )
    pairs = _pair_table()
    activity, represented = build_pair_lr_activity_matrix(pairs, means, database)
    assert activity.shape == (4, 2)
    assert represented.lr_id.tolist() == ["lig1->rec1", "lig2->rec2"]
    modifiers = {
        "attention": np.asarray([0.1, 1.0, 0.5, 0.2]),
        "exact_message": np.asarray([0.2, 1.0, 0.4, 0.1]),
    }
    scored = lr_scores_from_pair_modifiers(activity, represented, modifiers)
    assert {
        "lr_only_score",
        "attention_score",
        "exact_message_score",
        "attention_minus_lr_only",
    }.issubset(scored.columns)
    assert np.isfinite(scored.filter(like="score").to_numpy()).all()


def test_external_lr_collapse_and_shared_metrics() -> None:
    cytobridge = pd.DataFrame(
        {
            "lr_id": ["a->x", "b->y", "c->z"],
            "ligand": ["a", "b", "c"],
            "receptor": ["x", "y", "z"],
            "pathways": ["P", "Q", "R"],
            "lr_only_score": [1.0, 2.0, 3.0],
            "attention_score": [1.0, 3.0, 2.0],
            "exact_message_score": [1.0, 2.0, 4.0],
        }
    )
    commot = collapse_commot_lr_scores(
        pd.DataFrame(
            {
                "ligand": ["A", "A", "B", "C"],
                "receptor": ["X", "X", "Y", "Z"],
                "abundance_controlled_distinct_cell_score": [0.5, 0.5, 2.0, 4.0],
            }
        )
    )
    merged, metrics = shared_lr_rank_metrics(
        cytobridge, commot, external_score="commot_score", top_fraction=1 / 3
    )
    assert len(merged) == 3
    assert set(metrics.cytobridge_view) == {"lr_only", "attention", "exact_message"}
    assert metrics.set_index("cytobridge_view").loc[
        "exact_message", "spearman_rho"
    ] == pytest.approx(1.0)
    nichenet = collapse_nichenet_lr_scores(
        pd.DataFrame(
            {
                "ligand": ["A", "A", "B", "C"],
                "receptor": ["X", "X", "Y", "Z"],
                "lr_evidence": [0.2, 0.4, 0.5, 0.6],
            }
        )
    )
    assert nichenet.set_index("lr_id").loc["a->x", "nichenet_score"] == pytest.approx(
        0.3
    )


def test_modifier_permutation_is_deterministic_and_audited() -> None:
    pairs = _pair_table()
    activity = np.asarray(
        [
            [1.0, 0.0, 0.2],
            [0.8, 0.1, 0.0],
            [0.0, 0.7, 0.2],
            [0.1, 1.0, 0.0],
        ]
    )
    modifier = np.asarray([0.1, 1.0, 0.8, 0.2])
    strata = adaptive_pair_strata(pairs)
    result_a = modifier_permutation_test(
        activity,
        modifier,
        [0.9, 0.4, 0.1],
        [0, 1, 2],
        strata,
        permutations=20,
        seed=17,
        top_fraction=1 / 3,
    )
    result_b = modifier_permutation_test(
        activity,
        modifier,
        [0.9, 0.4, 0.1],
        [0, 1, 2],
        strata,
        permutations=20,
        seed=17,
        top_fraction=1 / 3,
    )
    assert result_a == result_b
    assert result_a["n_permutations"] == 20
    assert 0 <= result_a["spearman_empirical_p_upper"] <= 1


def test_paper_reference_and_target_bridge_are_explicit() -> None:
    scores = pd.DataFrame(
        {
            "lr_id": ["a->x", "b->y", "c->z", "d->w", "e->v"],
            "ligand": ["a", "b", "c", "d", "e"],
            "receptor": ["x", "y", "z", "w", "v"],
            "pathways": ["P", "P", "Q", "R", "S"],
            "lr_only_score": [0.2, 0.1, 0.8, 0.4, 0.3],
            "attention_score": [0.9, 0.8, 0.2, 0.1, 0.3],
            "exact_message_score": [0.9, 0.7, 0.2, 0.1, 0.3],
        }
    )
    paper = pd.DataFrame(
        {
            "paper_display_order": [1, 2],
            "ligand": ["a", "b"],
            "receptor": ["x", "y"],
        }
    )
    annotated, enrichment = paper_reference_enrichment(
        scores,
        paper,
        score_columns=["lr_only_score", "exact_message_score"],
        top_fraction=0.4,
    )
    assert annotated.paper_2022_reference.sum() == 2
    exact = enrichment.set_index("score_column").loc["exact_message_score"]
    assert exact.paper_reference_auc == pytest.approx(1.0)

    commot = pd.DataFrame(
        {"lr_id": scores.lr_id, "commot_score": [0.9, 0.8, 0.1, 0.2, 0.3]}
    )
    nichenet = pd.DataFrame(
        {"lr_id": scores.lr_id, "nichenet_score": [0.9, 0.7, 0.1, 0.2, 0.3]}
    )
    targets = pd.DataFrame(
        {
            "ligand": ["a", "a", "b"],
            "receptor": ["x", "x", "y"],
            "target": ["t1", "t2", "t3"],
            "ligand_target_evidence": [0.9, 0.5, 0.8],
        }
    )
    shared, links = jointly_supported_lr_targets(
        scores,
        commot,
        nichenet,
        targets,
        cytobridge_view="exact_message",
        top_fraction=0.4,
    )
    assert shared.jointly_supported.sum() == 2
    assert links[["lr_id", "target"]].values.tolist() == [
        ["a->x", "t1"],
        ["a->x", "t2"],
        ["b->y", "t3"],
    ]


def test_original_paper_reference_contains_exactly_21_unique_axes() -> None:
    path = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "scripts"
        / "reviewer_zebrafish_ccc"
        / "original_paper_21_lr.csv"
    )
    reference = pd.read_csv(path)
    assert len(reference) == 21
    assert reference.paper_display_order.tolist() == list(range(1, 22))
    assert not reference[["ligand", "receptor"]].duplicated().any()
    assert set(reference.source_figure) == {"Figure 5B"}


def test_rank_metrics_rejects_unitless_degenerate_comparison() -> None:
    metrics = rank_metrics([1, 2, 3], [3, 2, 1], top_fraction=1 / 3)
    assert metrics.spearman_rho == pytest.approx(-1.0)
    with pytest.raises(ValueError, match="at least three"):
        rank_metrics([1, 2], [1, 2])


def test_positive_rank_weights_preserve_structural_zeros() -> None:
    values = positive_rank_weights([0.0, 4.0, 2.0, 0.0])
    assert values.tolist() == [0.0, 1.0, 0.5, 0.0]
