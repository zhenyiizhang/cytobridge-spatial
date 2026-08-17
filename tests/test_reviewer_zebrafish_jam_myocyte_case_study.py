from __future__ import annotations

from pathlib import Path
import sys

import anndata as ad
import numpy as np
import pandas as pd
import pytest


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from reviewer_zebrafish_ccc import jam_myocyte_case_study as JAM  # noqa: E402


def _adata(
    obs_names: list[str],
    annotations: list[str],
    *,
    stages: list[float] | None = None,
    stage_labels: list[str] | None = None,
) -> ad.AnnData:
    n = len(obs_names)
    obs = pd.DataFrame(
        {
            "time_point_processed": stages or [3.0] * n,
            "time": stage_labels or ["18hpf"] * n,
            "Annotation": annotations,
        },
        index=obs_names,
    )
    return ad.AnnData(
        X=np.zeros((n, 4), dtype=np.float32),
        obs=obs,
        var=pd.DataFrame(index=["jam2a", "jam3b", "myog", "mymk"]),
    )


def _observed() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "global_index": [0, 1, 2],
            "obs_name": ["cell_a", "cell_b", "cell_c"],
            "stage": [3.0, 3.0, 3.0],
            "stage_label": ["18hpf", "18hpf", "18hpf"],
            "cell_type": ["A", "B", "C"],
        }
    )


def test_expected_grouping_seeds_are_exactly_five_unique() -> None:
    assert JAM.parse_seeds("101,202,303,404,505") == (101, 202, 303, 404, 505)
    with pytest.raises(ValueError, match="Exactly five"):
        JAM.parse_seeds("101,202,303,404")
    with pytest.raises(ValueError, match="Exactly five"):
        JAM.parse_seeds("101,101,202,303,404")


def test_external_methods_are_optional_for_cytobridge_only_audit() -> None:
    args = JAM.parser().parse_args(
        [
            "--h5ad",
            "input.h5ad",
            "--edge-dir",
            "edges",
            "--observed-cells",
            "observed.csv.gz",
            "--output-dir",
            "output",
        ]
    )
    assert args.external_spec == []
    assert JAM.parse_external_specs(args.external_spec) == []


def test_unrelated_case_insensitive_gene_duplicate_is_tolerated() -> None:
    data = ad.AnnData(
        X=np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 2.0]], dtype=np.float32),
        var=pd.DataFrame(index=["jam2a", "abcc5", "ABCC5"]),
    )
    values, audit = JAM.load_gene_values(data, ["jam2a"])
    assert values["jam2a"].tolist() == [1.0, 0.0]
    assert audit.loc[audit["gene"].eq("jam2a"), "available"].item()


def test_requested_case_insensitive_gene_duplicate_fails_closed() -> None:
    data = ad.AnnData(
        X=np.ones((2, 3), dtype=np.float32),
        var=pd.DataFrame(index=["jam2a", "JAM2A", "abcc5"]),
    )
    with pytest.raises(ValueError, match="Required gene 'jam2a'.*2 case-insensitive"):
        JAM.load_gene_values(data, ["jam2a"])


def test_obs_name_mapping_defeats_h5ad_index_order_trap() -> None:
    # H5AD row order is C,A,B, whereas attribution global order is A,B,C.
    data = _adata(["cell_c", "cell_a", "cell_b"], ["C", "A", "B"])
    table, global_to_h5, global_to_stage = JAM.map_observed_cells(
        data,
        _observed(),
        stage=3.0,
        stage_label="18hpf",
        time_key="time_point_processed",
        time_label_key="time",
        annotation_key="Annotation",
    )
    assert table.set_index("global_index").loc[0, "h5ad_index"] == 1
    edges = pd.DataFrame(
        {
            "source_index": [0],
            "target_index": [1],
            "source_index_stage": [0],
            "target_index_stage": [1],
            "sender_type": ["A"],
            "receiver_type": ["B"],
        }
    )
    resolved = JAM.resolve_edge_endpoints(
        edges,
        data,
        observed_table=table,
        global_to_h5=global_to_h5,
        global_to_stage=global_to_stage,
        stage=3.0,
        stage_label="18hpf",
        time_key="time_point_processed",
        annotation_key="Annotation",
    )
    assert resolved["_source_h5ad_index"].tolist() == [1]
    assert resolved["_target_h5ad_index"].tolist() == [2]


def test_stage_local_disagreement_fails_closed_after_obs_name_mapping() -> None:
    data = _adata(["cell_c", "cell_a", "cell_b"], ["C", "A", "B"])
    table, global_to_h5, global_to_stage = JAM.map_observed_cells(
        data,
        _observed(),
        stage=3.0,
        stage_label="18hpf",
        time_key="time_point_processed",
        time_label_key="time",
        annotation_key="Annotation",
    )
    edges = pd.DataFrame(
        {
            "source_index": [0],
            "target_index": [1],
            "source_index_stage": [1],
            "target_index_stage": [0],
            "sender_type": ["A"],
            "receiver_type": ["B"],
        }
    )
    with pytest.raises(ValueError, match="Stage-local edge indices disagree"):
        JAM.resolve_edge_endpoints(
            edges,
            data,
            observed_table=table,
            global_to_h5=global_to_h5,
            global_to_stage=global_to_stage,
            stage=3.0,
            stage_label="18hpf",
            time_key="time_point_processed",
            annotation_key="Annotation",
        )


def test_distinct_denominator_and_five_seed_zero_imputation() -> None:
    axes = pd.DataFrame(
        {
            "axis_id": ["jam2a->jam3b", "jam3b->jam2a"],
            "ligand": ["jam2a", "jam3b"],
            "receptor": ["jam3b", "jam2a"],
        }
    )
    data = _adata(["a", "b", "c"], ["Somite", "Somite", "Somite"])
    universe = JAM.context_universe(
        data,
        stage=3.0,
        stage_label="18hpf",
        time_key="time_point_processed",
        annotation_key="Annotation",
        axes=axes,
    )
    # One distinct edge in one seed plus a high-valued self edge. The self edge
    # must not enter either numerator or the n*(n-1) denominator.
    edges = pd.DataFrame(
        {
            "grouping_seed": [101, 101],
            "_source_h5ad_index": [0, 2],
            "_target_h5ad_index": [1, 2],
            "sender_type": ["Somite", "Somite"],
            "receiver_type": ["Somite", "Somite"],
            "attention_abs_mean": [2.0, 100.0],
            "edge_message_norm_joint": [3.0, 100.0],
        }
    )
    activities = {
        "jam2a": np.asarray([1.0, 0.0, 1.0]),
        "jam3b": np.asarray([0.0, 1.0, 1.0]),
    }
    scored = JAM.score_cytobridge_contexts(
        edges,
        universe,
        activities,
        expected_seeds=(101, 202, 303, 404, 505),
    )
    row = scored.loc[scored["axis_id"].eq("jam2a->jam3b")].iloc[0]
    assert row["n_possible_distinct_cell_pairs"] == 3 * 3 - 3
    assert row["n_grouping_seeds"] == 5
    assert row["n_seeds_with_any_edge"] == 1
    assert row["n_context_seed_zeros"] == 4
    assert row["n_active_unique_edges"] == 1
    assert row["cytobridge_raw_attention_magnitude_density"] == pytest.approx(2 / 30)
    assert row["cytobridge_lr_only_density"] == pytest.approx(1 / 30)
    assert row["cytobridge_attention_lr_density"] == pytest.approx(2 / 30)
    assert row["cytobridge_exact_message_lr_density"] == pytest.approx(3 / 30)


def test_distinct_pair_count_uses_set_intersection() -> None:
    assert JAM.distinct_pair_count(3, 3, 3) == 6
    assert JAM.distinct_pair_count(3, 2, 0) == 6
    assert JAM.distinct_pair_count(4, 5, 2) == 18
    with pytest.raises(ValueError, match="intersection"):
        JAM.distinct_pair_count(2, 3, 4)


def test_support_filter_precedes_min_rank_and_zeros_and_ties_are_retained() -> None:
    frame = pd.DataFrame(
        {
            "stage": [3.0] * 4,
            "axis_id": ["jam2a->jam3b"] * 4,
            "score": [2.0, 2.0, 0.0, 100.0],
            "passes_context_support_filter": [True, True, True, False],
        }
    )
    ranked = JAM.attach_ranks(frame, {"score": "test"})
    assert ranked["test_rank_from_top"].iloc[:3].tolist() == [1.0, 1.0, 3.0]
    assert ranked["test_n_ranked_contexts"].iloc[:3].tolist() == [3.0, 3.0, 3.0]
    assert ranked["test_tie_count"].iloc[:3].tolist() == [2.0, 2.0, 1.0]
    assert ranked["test_top_tail_percent"].iloc[2] == pytest.approx(200 / 3)
    assert ranked["test_rank_over_n"].iloc[:3].tolist() == ["1/3", "1/3", "3/3"]
    assert pd.isna(ranked["test_rank_from_top"].iloc[3])


def test_raw_type_pair_rank_universe_is_independent_of_jam_support() -> None:
    axes = pd.DataFrame(
        {
            "axis_id": ["jam2a->jam3b", "jam3b->jam2a"],
            "ligand": ["jam2a", "jam3b"],
            "receptor": ["jam3b", "jam2a"],
        }
    )
    data = _adata(["a0", "a1", "b0", "b1"], ["A", "A", "B", "B"])
    universe = JAM.context_universe(
        data,
        stage=3.0,
        stage_label="18hpf",
        time_key="time_point_processed",
        annotation_key="Annotation",
        axes=axes,
    )
    edges = pd.DataFrame(
        {
            "grouping_seed": [101],
            "_source_h5ad_index": [0],
            "_target_h5ad_index": [1],
            "sender_type": ["A"],
            "receiver_type": ["A"],
            "attention_abs_mean": [2.0],
        }
    )
    ranked = JAM.score_raw_type_pair_universe(
        edges,
        universe,
        expected_seeds=(101, 202, 303, 404, 505),
    )
    assert len(ranked) == 4  # 2 x 2 full directed square
    assert set(ranked["raw_attention_full_type_pair_n_ranked_contexts"]) == {4.0}
    zero_rows = ranked["n_raw_edge_occurrences"].eq(0)
    assert zero_rows.sum() == 3
    assert ranked.loc[zero_rows, "raw_attention_full_type_pair_density"].eq(0).all()


def test_external_stage_prefers_internal_stage_over_biological_time() -> None:
    frame = pd.DataFrame({"stage": [3.0, 4.0], "stage_time": [18.0, 24.0]})
    mask, basis = JAM.normalized_stage(
        frame,
        "COMMOT",
        internal_stage=3.0,
        biological_time_hpf=18.0,
    )
    assert basis == "internal_stage"
    assert mask.tolist() == [True, False]

    fallback = pd.DataFrame(
        {"stage": ["18hpf", "24hpf"], "stage_time": [18.0, 24.0]}
    )
    fallback_mask, fallback_basis = JAM.normalized_stage(
        fallback,
        "legacy external",
        internal_stage=3.0,
        biological_time_hpf=18.0,
    )
    assert fallback_basis == "biological_time_hpf_fallback"
    assert fallback_mask.tolist() == [True, False]


def _external_contexts() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "stage": [3.0, 3.0, 3.0],
            "axis_id": ["jam2a->jam3b", "jam2a->jam3b", "jam3b->jam2a"],
            "ligand": ["jam2a", "jam2a", "jam3b"],
            "receptor": ["jam3b", "jam3b", "jam2a"],
            "sender_type": ["Somite", "A", "Somite"],
            "receiver_type": ["Somite", "B", "Somite"],
            "n_possible_distinct_cell_pairs": [6, 8, 6],
            "passes_context_support_filter": [True, True, True],
        }
    )


def test_sparse_external_valid_missing_row_is_zero_unavailable_axis_is_na() -> None:
    contexts = _external_contexts()
    scores = pd.DataFrame(
        {
            "axis_id": ["jam2a->jam3b"],
            "ligand": ["jam2a"],
            "receptor": ["jam3b"],
            "sender_type": ["Somite"],
            "receiver_type": ["Somite"],
            "external_input_score": [0.5],
            "external_duplicate_provenance_rows_collapsed": [1],
            "external_input_distinct_denominator": [6.0],
        }
    )
    availability = pd.DataFrame(
        {
            "axis_id": ["jam2a->jam3b", "jam3b->jam2a"],
            "ligand": ["jam2a", "jam3b"],
            "receptor": ["jam3b", "jam2a"],
            "external_axis_available": [True, False],
            "external_availability_provenance_rows": [1, 1],
            "external_matrix_keys": ["key", ""],
        }
    )
    spec = JAM.ExternalSpec(
        method="COMMOT",
        table=Path("scores.csv"),
        availability=Path("availability.csv"),
        score_column="abundance_controlled_distinct_cell_score",
        score_mode="distinct_density",
    )
    joined = JAM.join_external_sparse(contexts, scores, availability, spec=spec)
    assert joined["commot_score"].tolist()[:2] == [0.5, 0.0]
    assert joined["commot_sparse_context_completed_as_zero"].tolist()[:2] == [False, True]
    assert pd.isna(joined["commot_score"].iloc[2])


def test_external_distinct_denominator_mismatch_fails_closed() -> None:
    contexts = _external_contexts().iloc[[0]].copy()
    scores = pd.DataFrame(
        {
            "axis_id": ["jam2a->jam3b"],
            "ligand": ["jam2a"],
            "receptor": ["jam3b"],
            "sender_type": ["Somite"],
            "receiver_type": ["Somite"],
            "external_input_score": [1.0],
            "external_duplicate_provenance_rows_collapsed": [1],
            "external_input_distinct_denominator": [9.0],
        }
    )
    availability = pd.DataFrame(
        {
            "axis_id": ["jam2a->jam3b"],
            "ligand": ["jam2a"],
            "receptor": ["jam3b"],
            "external_axis_available": [True],
            "external_availability_provenance_rows": [1],
            "external_matrix_keys": ["key"],
        }
    )
    spec = JAM.ExternalSpec(
        "COMMOT", Path("scores"), Path("availability"), "score", "distinct_mass"
    )
    with pytest.raises(ValueError, match="denominator disagrees"):
        JAM.join_external_sparse(contexts, scores, availability, spec=spec)


def test_compatible_undirected_neighbor_pair_is_counted_once() -> None:
    pairs = np.asarray([[0, 1], [0, 2]], dtype=int)
    jam2a = np.asarray([True, False, True])
    jam3b = np.asarray([True, False, True])
    # Both orientations are true for 0--2, but the undirected pair is counted once.
    assert JAM.compatible_neighbor_count(pairs, jam2a, jam3b) == 1


def test_missing_control_metrics_are_explicit_na_not_fabricated() -> None:
    controls = JAM.load_control_artifact(None)
    assert set(controls["control"]) == {
        "trained",
        "pre_interaction",
        "randomized_interaction_seed17",
    }
    assert not controls["control_metrics_available"].any()
    assert controls["unavailable_reason"].str.len().gt(0).all()


def test_control_condition_aliases_are_canonicalized(tmp_path: Path) -> None:
    source = tmp_path / "controls.csv"
    pd.DataFrame(
        {
            "condition": ["trained", "pre_interaction", "random"],
            "jam_compatible_edge_percentile_mean": [0.67, 0.66, 0.57],
        }
    ).to_csv(source, index=False)

    controls = JAM.load_control_artifact(source)

    assert controls["control"].tolist() == [
        "trained",
        "pre_interaction",
        "randomized_interaction_seed17",
    ]
    assert controls["control_metrics_available"].all()
    assert controls["unavailable_reason"].eq("").all()
