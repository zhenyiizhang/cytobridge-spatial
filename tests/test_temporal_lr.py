from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from CytoBridge.tl.downstream.lr_projection import (
    _combine_complex,
    compute_focal_lr_type_hotspots,
    project_communication_to_lr_timecourses,
)
from CytoBridge.tl.downstream.temporal import (
    cluster_temporal_profiles,
    inverse_pca_states,
    load_pca_reconstruction_spec,
    pca_reconstruction_feature_coverage,
    simplify_gene_names,
    summarize_temporal_gene_patterns,
)


def _reference_adata() -> ad.AnnData:
    # Column means are [1, 2, 3], which is the recoverable PCA center.
    reference = ad.AnnData(
        X=np.asarray(
            [
                [0.8, 1.8, 2.8],
                [1.2, 2.2, 3.2],
                [1.0, 2.0, 3.0],
                [1.0, 2.0, 3.0],
            ],
            dtype=np.float32,
        )
    )
    reference.var_names = [
        "L1 | AMEX0001",
        "LOC1[nr]|R1[hs] | AMEX0002",
        "G3 | AMEX0003",
    ]
    reference.varm["PCs"] = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, -0.5],
        ],
        dtype=np.float32,
    )
    center = np.asarray(reference.X).mean(axis=0, keepdims=True)
    reference.var["pca_center"] = center.reshape(-1)
    reference.obsm["X_pca"] = (
        (np.asarray(reference.X) - center) @ reference.varm["PCs"]
    ).astype(np.float32)
    return reference


def _trajectory_slices() -> dict[str, ad.AnnData]:
    slices = {}
    for time, pcs in {
        0.0: [[0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [0.1, 0.1]],
        0.5: [[0.2, 0.1], [0.3, 0.1], [0.2, 0.2], [0.3, 0.2]],
        1.0: [[0.4, 0.2], [0.5, 0.2], [0.4, 0.3], [0.5, 0.3]],
    }.items():
        states = np.column_stack(
            (
                np.zeros((4, 2), dtype=np.float32),
                np.asarray(pcs, dtype=np.float32),
            )
        )
        current = ad.AnnData(X=states)
        current.obs["Annotation"] = ["A", "A", "B", "B"]
        slices[str(time)] = current
    return slices


def _observed_expression() -> ad.AnnData:
    # Deliberately use a different cell count/order than the trajectory slices
    # and reverse the PCA feature order. Hybrid projection must align by time,
    # the observed AnnData's own labels, and exact var names (never row order).
    observed = ad.AnnData(X=np.zeros((6, 3), dtype=np.float32))
    observed.var_names = [
        "G3 | AMEX0003",
        "LOC1[nr]|R1[hs] | AMEX0002",
        "L1 | AMEX0001",
    ]
    observed.obs["stage"] = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    observed.obs["cell_type"] = ["B", "A", "B", "B", "A", "B"]
    # Rows are [G3, R1, L1]. Type means in [L1, R1] order are:
    # t0: A=[11, 2], B=[3, 21]; t1: A=[6, 2], B=[1, 9].
    observed.layers["counts"] = np.asarray(
        [
            [0.0, 20.0, 2.0],
            [0.0, 2.0, 11.0],
            [0.0, 22.0, 4.0],
            [0.0, 8.0, 0.0],
            [0.0, 2.0, 6.0],
            [0.0, 10.0, 2.0],
        ],
        dtype=np.float32,
    )
    return observed


def test_inverse_pca_and_gene_aliases() -> None:
    reference = _reference_adata()
    reconstructed = inverse_pca_states(
        reference,
        np.asarray([[0.0, 0.0, 0.2, 0.4]], dtype=np.float32),
    )
    np.testing.assert_allclose(reconstructed[0], [1.2, 2.4, 2.9], atol=1e-6)
    aliases = simplify_gene_names(reference.var_names, preferred_species_tag="hs")
    assert aliases["gene_symbol"].tolist() == ["L1", "R1", "G3"]


def test_complex_geometric_mean_is_zero_preserving_and_distinct_from_min() -> None:
    vectors = {
        "A": np.asarray([1.0, 4.0, 0.0]),
        "B": np.asarray([4.0, 9.0, 2.0]),
    }
    minimum, missing = _combine_complex(
        "A_B", vectors, mode="min", require_all_subunits=True
    )
    geometric, geometric_missing = _combine_complex(
        "A_B", vectors, mode="geometric_mean", require_all_subunits=True
    )

    assert missing == geometric_missing == []
    np.testing.assert_allclose(minimum, [1.0, 4.0, 0.0])
    np.testing.assert_allclose(geometric, [2.0, 6.0, 0.0])

    with pytest.raises(ValueError, match="requires non-negative"):
        _combine_complex(
            "A_B",
            {"A": np.asarray([-1.0]), "B": np.asarray([2.0])},
            mode="geometric_mean",
            require_all_subunits=True,
        )

    with pytest.raises(ValueError, match="non-finite"):
        _combine_complex(
            "A_B",
            {"A": np.asarray([np.nan]), "B": np.asarray([2.0])},
            mode="min",
            require_all_subunits=True,
        )


def test_persisted_pca_center_survives_reference_population_subsetting() -> None:
    reference = _reference_adata()
    fit_center = np.asarray([10.0, 20.0, 30.0], dtype=np.float32)
    reference.var["pca_center"] = fit_center
    subset = reference[[0, 1]].copy()
    subset.layers["different_population"] = np.full(
        subset.shape,
        999.0,
        dtype=np.float32,
    )

    reconstructed = inverse_pca_states(
        subset,
        np.asarray([[0.0, 0.0, 0.2, 0.4]], dtype=np.float32),
        layer="different_population",
    )
    np.testing.assert_allclose(reconstructed[0], [10.2, 20.4, 29.9], atol=1e-6)


def test_explicit_pca_contract_and_historical_symbol_policy() -> None:
    loadings = pd.DataFrame(
        {
            "feature": ["OLD1[nr]|HUMAN1[hs] | AMEX1", "L2 | AMEX2"],
            "PC1": [1.0, 0.5],
            "PC2": [0.0, -0.5],
        }
    )
    center = pd.DataFrame(
        {
            "feature": ["L2 | AMEX2", "OLD1[nr]|HUMAN1[hs] | AMEX1"],
            "mean": [2.0, 1.0],
        }
    )
    reconstruction = load_pca_reconstruction_spec(loadings, center)
    reference = ad.AnnData(X=np.zeros((1, 2), dtype=np.float32))
    reconstructed = inverse_pca_states(
        reference,
        np.asarray([[0.0, 0.0, 0.2, 0.4]], dtype=np.float32),
        reconstruction=reconstruction,
    )
    np.testing.assert_allclose(reconstructed[0], [1.2, 1.9], atol=1e-6)
    historical = simplify_gene_names(reconstruction.feature_names)
    preferred_human = simplify_gene_names(
        reconstruction.feature_names, preferred_species_tag="hs"
    )
    assert historical["gene_symbol"].tolist() == ["OLD1", "L2"]
    assert preferred_human["gene_symbol"].tolist() == ["HUMAN1", "L2"]


def test_ward_dendrogram_cluster_order_is_recorded() -> None:
    profiles = pd.DataFrame(
        [[0.0, 1.0, 0.0], [0.0, 0.9, 0.1], [1.0, 0.0, 0.0], [0.9, 0.1, 0.0]],
        index=["a", "b", "c", "d"],
        columns=[0.0, 0.5, 1.0],
    )
    result = cluster_temporal_profiles(
        profiles,
        n_clusters=2,
        normalization="minmax",
        method="ward",
        cluster_order="dendrogram",
    )
    assignments = result.assignments.set_index("profile")["cluster"]
    assert assignments["a"] == assignments["b"]
    assert assignments["c"] == assignments["d"]
    assert assignments["a"] != assignments["c"]
    assert result.diagnostics.loc[0, "linkage_method"] == "ward"
    assert result.diagnostics.loc[0, "cluster_order"] == "dendrogram"
    assert sorted(result.assignments["dendrogram_rank"].tolist()) == [0, 1, 2, 3]


def test_kmeans_temporal_clustering_is_deterministic() -> None:
    profiles = pd.DataFrame(
        [
            [0.0, 0.8, 1.0, 0.2],
            [0.0, 0.7, 0.9, 0.3],
            [1.0, 0.8, 0.1, 0.0],
            [0.9, 1.0, 0.2, 0.0],
        ],
        index=["late_a", "late_b", "early_a", "early_b"],
        columns=[0.0, 0.5, 1.0, 1.5],
    )
    first = cluster_temporal_profiles(
        profiles,
        n_clusters=2,
        normalization="minmax",
        method="kmeans",
        cluster_order="peak_time",
    )
    second = cluster_temporal_profiles(
        profiles,
        n_clusters=2,
        normalization="minmax",
        method="kmeans",
        cluster_order="peak_time",
    )

    pd.testing.assert_frame_equal(first.assignments, second.assignments)
    pd.testing.assert_frame_equal(first.prototypes, second.prototypes)
    assignments = first.assignments.set_index("profile")["cluster"]
    assert assignments["early_a"] == assignments["early_b"] == 1
    assert assignments["late_a"] == assignments["late_b"] == 2
    diagnostics = first.diagnostics.iloc[0]
    assert diagnostics["linkage_method"] == "kmeans"
    assert diagnostics["cut_strategy"] == "sklearn_kmeans_pp_n_init_100_seed_0"
    assert diagnostics["minimum_cluster_size"] == 2
    assert diagnostics["maximum_cluster_size"] == 2

    with pytest.raises(ValueError, match="unavailable"):
        cluster_temporal_profiles(
            profiles,
            n_clusters=2,
            normalization="minmax",
            method="kmeans",
            cluster_order="dendrogram",
        )


def test_temporal_clustering_zero_distance_ties_still_cut_to_exact_k() -> None:
    profiles = pd.DataFrame(
        np.zeros((6, 4), dtype=np.float64),
        index=[f"p{index}" for index in range(6)],
        columns=[0.0, 0.5, 1.0, 1.5],
    )
    first = cluster_temporal_profiles(
        profiles,
        n_clusters=3,
        normalization="minmax",
        method="average",
        cluster_order="peak_time",
    )
    second = cluster_temporal_profiles(
        profiles,
        n_clusters=3,
        normalization="minmax",
        method="average",
        cluster_order="peak_time",
    )

    assert sorted(first.assignments["cluster"].unique().tolist()) == [1, 2, 3]
    assert first.assignments.equals(second.assignments)
    assert sorted(first.prototypes["cluster"].unique().tolist()) == [1, 2, 3]
    diagnostics = first.diagnostics.iloc[0]
    assert diagnostics["chosen_clusters"] == 3
    assert diagnostics["clusters_found"] == 3
    assert diagnostics["cut_strategy"] == "scipy_cut_tree_exact_n_clusters"
    assert diagnostics["n_zero_distance_merges"] == 5


def test_temporal_gene_and_lr_projection() -> None:
    reference = _reference_adata()
    slices = _trajectory_slices()
    gene_result = summarize_temporal_gene_patterns(
        slices,
        reference,
        n_top_genes=2,
        n_cluster_genes=3,
        n_clusters=2,
        preferred_species_tag="hs",
    )
    assert gene_result.expression.shape == (3, 3)
    assert gene_result.top_variable_genes.shape[0] == 2
    assert gene_result.clustering.assignments.shape[0] == 3
    assert gene_result.settings["n_cluster_genes"] == 3
    assert set(gene_result.clustering.assignments["cluster"]) == {1, 2}

    communications = {
        key: {
            "types": np.asarray(["A", "B"], dtype=object),
            "M_per_source": np.asarray([[0.0, 1.0], [0.5, 0.0]]),
        }
        for key in slices
    }
    lr_result = project_communication_to_lr_timecourses(
        slices,
        reference,
        communications,
        pd.DataFrame({"ligand": ["L1"], "receptor": ["R1"]}),
        preferred_species_tag="hs",
        n_clusters=1,
    )
    assert lr_result.pair_timecourse.shape[0] == 3
    assert (lr_result.pair_timecourse["score"] > 0).all()
    assert lr_result.coverage["n_lr_pairs_scored"].tolist() == [1, 1, 1]
    assert lr_result.coverage["expression_source"].tolist() == [
        "inverse_pca",
        "inverse_pca",
        "inverse_pca",
    ]
    expected_l1 = np.mean(np.expm1([1.0, 1.1]))
    expected_t0 = expected_l1 * (np.expm1(2.1) + 0.5 * np.expm1(2.0))
    score_t0 = lr_result.pair_timecourse.set_index("time").loc[0.0, "score"]
    assert score_t0 == pytest.approx(expected_t0)
    assert lr_result.settings["observed_expression"] is None
    assert lr_result.pattern_summary.loc[0, "cluster"] == 1


def test_lr_pair_identity_does_not_collapse_ambiguous_display_labels() -> None:
    reference = ad.AnnData(X=np.zeros((1, 3), dtype=np.float32))
    reference.var_names = ["A", "B", "C"]
    reference.var["pca_center"] = np.zeros(3, dtype=np.float32)
    slices = {}
    communications = {}
    for time in (0.0, 1.0):
        current = ad.AnnData(X=np.zeros((1, 2), dtype=np.float32))
        current.obs["Annotation"] = ["T"]
        slices[str(time)] = current
        communications[str(time)] = {
            "types": np.asarray(["T"], dtype=object),
            "M_per_source": np.asarray([[1.0]], dtype=np.float64),
        }

    observed = ad.AnnData(
        X=np.asarray([[2.0, 3.0, 5.0], [4.0, 9.0, 16.0]], dtype=np.float32)
    )
    observed.var_names = reference.var_names.copy()
    observed.obs["time"] = [0.0, 1.0]
    observed.obs["Annotation"] = ["T", "T"]

    result = project_communication_to_lr_timecourses(
        slices,
        reference,
        communications,
        pd.DataFrame(
            {
                "ligand": ["A_B", "A"],
                "receptor": ["C", "B_C"],
            }
        ),
        observed_adata=observed,
        observed_time_key="time",
        observed_time_points=[0.0, 1.0],
        observed_expression_space="count",
        expression_space="count",
        complex_mode="geomean",
        require_all_subunits=True,
        n_clusters=2,
        return_type_matrices=True,
    )

    expected_ids = {'["A_B","C"]', '["A","B_C"]'}
    assert result.pair_timecourse["pair"].unique().tolist() == ["A_B_C"]
    assert set(result.pair_timecourse["pair_id"]) == expected_ids
    assert set(
        result.pair_timecourse[["ligand", "receptor"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    ) == {("A_B", "C"), ("A", "B_C")}
    scores = result.pair_timecourse.set_index(["ligand", "receptor", "time"])["score"]
    assert scores.loc[("A_B", "C", 0.0)] == pytest.approx(np.sqrt(6.0) * 5.0)
    assert scores.loc[("A", "B_C", 0.0)] == pytest.approx(2.0 * np.sqrt(15.0))
    assert scores.loc[("A_B", "C", 1.0)] == pytest.approx(96.0)
    assert scores.loc[("A", "B_C", 1.0)] == pytest.approx(48.0)

    pair_audit = result.trajectory_coverage.loc[
        result.trajectory_coverage["trajectory_kind"] == "pair"
    ]
    assert pair_audit["pair"].nunique() == 1
    assert set(pair_audit["pair_id"]) == expected_ids
    assert pair_audit["retained"].all()
    assert result.celltype_timecourse["pair_id"].nunique() == 2
    assert result.type_matrix["pair_id"].nunique() == 2
    assert result.pattern_summary["pair_id"].nunique() == 2
    assert result.clustering.assignments["pair_id"].nunique() == 2
    assert set(result.clustering.assignments["profile"]) == {"A_B_C"}
    assert result.coverage["n_lr_pairs_scored"].tolist() == [2, 2]
    assert result.coverage["n_duplicate_pairs"].tolist() == [0, 0]
    assert result.settings["complex_mode"] == "geometric_mean"
    contract = result.settings["temporal_eligibility_contract"]
    assert contract["database_unique_pairs"] == 2
    assert contract["retained_complete_nonzero_pairs"] == 2
    assert contract["pair_identity"]["internal"] == "structured_tuple_ligand_receptor"


def test_hybrid_lr_projection_uses_observed_expression_only_at_declared_times() -> None:
    reference = _reference_adata()
    slices = _trajectory_slices()
    communications = {
        key: {
            "types": np.asarray(["A", "B"], dtype=object),
            "M_per_source": np.asarray([[0.0, 1.0], [0.5, 0.0]]),
        }
        for key in slices
    }
    lr_database = pd.DataFrame({"ligand": ["L1"], "receptor": ["R1"]})
    generated_only = project_communication_to_lr_timecourses(
        slices,
        reference,
        communications,
        lr_database,
        preferred_species_tag="hs",
        n_clusters=1,
    )
    hybrid = project_communication_to_lr_timecourses(
        slices,
        reference,
        communications,
        lr_database,
        preferred_species_tag="hs",
        n_clusters=1,
        observed_adata=_observed_expression(),
        observed_time_key="stage",
        observed_time_points=[0.0, 1.0],
        observed_annotation_key="cell_type",
        observed_layer="counts",
        observed_expression_space="count",
    )

    scores = hybrid.pair_timecourse.set_index("time")["score"]
    assert scores.loc[0.0] == pytest.approx(234.0)
    assert scores.loc[1.0] == pytest.approx(55.0)
    generated_scores = generated_only.pair_timecourse.set_index("time")["score"]
    assert scores.loc[0.5] == pytest.approx(generated_scores.loc[0.5])

    coverage = hybrid.coverage.set_index("time")
    assert coverage["expression_source"].tolist() == [
        "observed",
        "inverse_pca",
        "observed",
    ]
    assert coverage["n_cells"].tolist() == [4, 4, 4]
    assert coverage["n_expression_cells"].tolist() == [3, 4, 3]
    observed_settings = hybrid.settings["observed_expression"]
    assert observed_settings["gene_alignment"] == "var_name"
    assert observed_settings["layer"] == "counts"
    assert observed_settings["source_space"] == "count"
    assert (
        observed_settings["cell_alignment"]
        == "within_observed_adata_by_time_and_annotation"
    )


def test_hybrid_lr_projection_missing_observed_time_policy() -> None:
    reference = _reference_adata()
    slices = _trajectory_slices()
    communications = {
        key: {
            "types": np.asarray(["A", "B"], dtype=object),
            "M_per_source": np.asarray([[0.0, 1.0], [0.5, 0.0]]),
        }
        for key in slices
    }
    observed = _observed_expression()
    observed = observed[observed.obs["stage"].to_numpy() == 0.0].copy()
    kwargs = {
        "time_points": [0.0, 1.0],
        "preferred_species_tag": "hs",
        "n_clusters": 1,
        "observed_adata": observed,
        "observed_time_key": "stage",
        "observed_time_points": [0.0, 1.0],
        "observed_annotation_key": "cell_type",
        "observed_layer": "counts",
        "observed_expression_space": "count",
    }
    lr_database = pd.DataFrame({"ligand": ["L1"], "receptor": ["R1"]})
    with pytest.raises(ValueError, match="No observed_adata rows matched"):
        project_communication_to_lr_timecourses(
            slices,
            reference,
            communications,
            lr_database,
            **kwargs,
        )

    result = project_communication_to_lr_timecourses(
        slices,
        reference,
        communications,
        lr_database,
        observed_missing_time_policy="generated",
        **kwargs,
    )
    coverage = result.coverage.set_index("time")
    assert coverage["expression_source"].tolist() == ["observed", "inverse_pca"]
    assert coverage["observed_missing_fallback"].tolist() == [False, True]


def test_hybrid_lr_missing_observed_celltype_is_unavailable_not_zero_filled() -> None:
    reference = _reference_adata()
    slices = _trajectory_slices()
    communications = {
        key: {
            "types": np.asarray(["A", "B"], dtype=object),
            "M_per_source": np.asarray([[0.0, 1.0], [0.5, 0.0]]),
        }
        for key in slices
    }
    observed = _observed_expression()
    keep = ~(
        np.isclose(observed.obs["stage"].to_numpy(dtype=float), 1.0)
        & (observed.obs["cell_type"].astype(str).to_numpy() == "B")
    )
    observed = observed[keep].copy()
    result = project_communication_to_lr_timecourses(
        slices,
        reference,
        communications,
        pd.DataFrame({"ligand": ["L1"], "receptor": ["R1"]}),
        preferred_species_tag="hs",
        n_clusters=1,
        observed_adata=observed,
        observed_time_key="stage",
        observed_time_points=[0.0, 1.0],
        observed_annotation_key="cell_type",
        observed_layer="counts",
        observed_expression_space="count",
        return_type_matrices=True,
    )

    # Pair-level aggregation remains defined on supported sender/receiver
    # populations at all three times; the absent B population is not treated
    # as observed zero expression.
    assert result.pair_timecourse["time"].tolist() == [0.0, 0.5, 1.0]
    t1_pair = result.pair_timecourse.set_index("time").loc[1.0]
    assert t1_pair["n_expression_supported_cell_types"] == 1
    assert t1_pair["n_expression_unsupported_cell_types"] == 1
    coverage = result.coverage.set_index("time")
    assert coverage.loc[1.0, "n_expression_supported_cell_types"] == 1
    assert coverage.loc[1.0, "expression_unsupported_cell_types"] == "B"

    audit = result.trajectory_coverage.loc[
        result.trajectory_coverage["trajectory_kind"] == "pair_celltype"
    ].set_index(["pair", "cell_type"])
    assert bool(audit.loc[("L1_R1", "A"), "retained"])
    assert not bool(audit.loc[("L1_R1", "B"), "retained"])
    assert audit.loc[("L1_R1", "B"), "expression_unavailable_times"] == "[1.0]"
    assert audit.loc[("L1_R1", "B"), "drop_reason"] == "missing_expression_support"
    assert set(result.celltype_timecourse["cell_type"]) == {"A"}
    t1_matrix = result.type_matrix.loc[result.type_matrix["time"] == 1.0]
    unsupported = t1_matrix.loc[~t1_matrix["expression_supported_edge"].astype(bool)]
    assert not unsupported.empty
    assert unsupported["lr_score"].isna().all()
    support_contract = result.settings["expression_support_contract"]
    assert support_contract["unsupported_type_value"] == (
        "unavailable_not_observed_zero"
    )


def test_hybrid_lr_projection_rejects_missing_reconstruction_genes() -> None:
    observed = _observed_expression()[:, [0, 1]].copy()
    with pytest.raises(ValueError, match="missing 1 PCA reconstruction features"):
        project_communication_to_lr_timecourses(
            {"0.0": _trajectory_slices()["0.0"]},
            _reference_adata(),
            {
                "0.0": {
                    "types": np.asarray(["A", "B"], dtype=object),
                    "M_per_source": np.asarray([[0.0, 1.0], [0.5, 0.0]]),
                }
            },
            pd.DataFrame({"ligand": ["L1"], "receptor": ["R1"]}),
            time_points=[0.0],
            observed_adata=observed,
            observed_time_key="stage",
            observed_annotation_key="cell_type",
            observed_layer="counts",
        )


def test_pca_feature_coverage_and_temporal_summary_exclude_center_only_genes() -> None:
    reference = _reference_adata()
    reference.varm["PCs"][2] = [0.0, 1e-10]
    coverage = pca_reconstruction_feature_coverage(
        reference.var_names,
        reference.varm["PCs"],
    )
    assert coverage["active"].tolist() == [True, True, False]
    assert coverage.loc[2, "max_abs_loading"] == pytest.approx(1e-10)

    result = summarize_temporal_gene_patterns(
        _trajectory_slices(),
        reference,
        n_top_genes=2,
        n_clusters=1,
    )
    assert result.expression.shape == (2, 3)
    assert "G3" not in result.expression.index
    assert result.gene_name_map["pca_active"].all()
    assert result.settings["pca_features_inactive"] == 1


def _count_aggregation_fixture():
    reference = ad.AnnData(X=np.zeros((2, 2), dtype=np.float32))
    reference.var_names = ["Lig", "Rec"]
    reference.varm["PCs"] = np.eye(2, dtype=np.float32)
    reference.var["pca_center"] = np.zeros(2, dtype=np.float32)

    log_states = np.asarray(
        [
            [0.0, 0.0],
            [np.log1p(8.0), 0.0],
            [0.0, 0.0],
            [0.0, np.log1p(6.0)],
        ],
        dtype=np.float32,
    )
    slices = {}
    communications = {}
    for time in (0.0, 1.0):
        current = ad.AnnData(
            X=np.column_stack((np.zeros((4, 2), dtype=np.float32), log_states))
        )
        current.obs["Annotation"] = ["A", "A", "B", "B"]
        slices[str(time)] = current
        communications[str(time)] = {
            "types": np.asarray(["A", "B"], dtype=object),
            "M_per_source": np.asarray([[0.0, 1.0], [0.0, 0.0]]),
        }

    observed = ad.AnnData(X=np.vstack([log_states, log_states]))
    observed.var_names = reference.var_names.copy()
    observed.obs["time"] = [0.0] * 4 + [1.0] * 4
    observed.obs["Annotation"] = ["A", "A", "B", "B"] * 2
    return reference, slices, communications, observed


def test_count_space_converts_each_cell_before_arithmetic_celltype_mean() -> None:
    reference, slices, communications, observed = _count_aggregation_fixture()
    database = pd.DataFrame({"ligand": ["Lig"], "receptor": ["Rec"]})
    generated = project_communication_to_lr_timecourses(
        slices,
        reference,
        communications,
        database,
        n_clusters=1,
        expression_space="count",
    )
    # The all-observed path must not require or apply a PCA-loading contract.
    del reference.varm["PCs"]
    from scipy import sparse

    for current in slices.values():
        current.X = sparse.csr_matrix(current.X)
    observed_result = project_communication_to_lr_timecourses(
        slices,
        reference,
        communications,
        database,
        n_clusters=1,
        expression_space="count",
        observed_adata=observed,
        observed_time_key="time",
        observed_expression_space="log1p",
    )

    # mean([0, 8]) * mean([0, 6]) = 4 * 3, not the product of
    # expm1(mean(log1p(.))) geometric pseudobulks.
    np.testing.assert_allclose(generated.pair_timecourse["score"], [12.0, 12.0])
    np.testing.assert_allclose(
        observed_result.pair_timecourse["score"],
        generated.pair_timecourse["score"],
    )
    assert (
        generated.settings["celltype_expression_aggregation"]
        == "per-cell source-to-target conversion followed by arithmetic cell-type mean"
    )
    assert observed_result.settings["uses_inverse_pca"] is False
    assert (
        observed_result.settings["feature_universe"]["pca_active_filter_applied"]
        is False
    )


def test_generated_log1p_lr_clips_negative_inverse_pca_expression_per_cell() -> None:
    reference = ad.AnnData(X=np.zeros((2, 2), dtype=np.float32))
    reference.var_names = ["Lig", "Rec"]
    reference.varm["PCs"] = np.eye(2, dtype=np.float32)
    reference.var["pca_center"] = np.zeros(2, dtype=np.float32)
    current = ad.AnnData(
        X=np.asarray(
            [
                [0.0, 0.0, -2.0, 0.0],
                [0.0, 0.0, 2.0, 0.0],
                [0.0, 0.0, 0.0, 3.0],
            ],
            dtype=np.float32,
        )
    )
    current.obs["Annotation"] = ["A", "A", "B"]
    slices = {str(time): current.copy() for time in (0.0, 1.0)}
    communications = {
        str(time): {
            "types": np.asarray(["A", "B"], dtype=object),
            "M_per_source": np.asarray([[0.0, 1.0], [0.0, 0.0]]),
        }
        for time in (0.0, 1.0)
    }

    result = project_communication_to_lr_timecourses(
        slices,
        reference,
        communications,
        pd.DataFrame({"ligand": ["Lig"], "receptor": ["Rec"]}),
        expression_space="log1p",
        n_clusters=1,
        return_type_matrices=True,
    )

    edge = result.type_matrix.query(
        "time == 0.0 and sender_type == 'A' and receiver_type == 'B'"
    ).iloc[0]
    assert edge["ligand_mean"] == pytest.approx(1.0)
    assert edge["receptor_mean"] == pytest.approx(3.0)
    assert edge["lr_score"] == pytest.approx(3.0)


def test_temporal_gene_default_clips_negative_log1p_per_cell() -> None:
    reference = ad.AnnData(X=np.zeros((2, 2), dtype=np.float32))
    reference.var_names = ["G1", "G2"]
    reference.varm["PCs"] = np.eye(2, dtype=np.float32)
    reference.var["pca_center"] = np.zeros(2, dtype=np.float32)
    slices = {}
    for time, values in (
        (0.0, [[-2.0, 0.0], [2.0, 3.0]]),
        (1.0, [[-4.0, 1.0], [4.0, 5.0]]),
    ):
        state = np.column_stack((np.zeros((2, 2)), np.asarray(values)))
        slices[str(time)] = ad.AnnData(X=state.astype(np.float32))

    result = summarize_temporal_gene_patterns(
        slices,
        reference,
        n_top_genes=2,
        n_cluster_genes=2,
        n_clusters=1,
    )

    np.testing.assert_allclose(result.expression.loc["G1"], [1.0, 2.0])
    np.testing.assert_allclose(result.signed_expression.loc["G1"], [0.0, 0.0])
    np.testing.assert_allclose(
        result.clustering.normalized_profiles.loc["G1"], [-1.0, 1.0]
    )
    assert result.settings["clip_min"] == 0.0


def test_strict_lr_complex_reports_missing_and_center_only_subunits() -> None:
    reference = ad.AnnData(X=np.ones((2, 3), dtype=np.float32))
    reference.var_names = ["Wnt3a", "Fzd7", "CenterOnly"]
    reference.varm["PCs"] = np.asarray(
        [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
        dtype=np.float32,
    )
    reference.var["pca_center"] = [1.0, 1.0, 10.0]
    slices = {}
    communications = {}
    for time in (0.0, 1.0):
        current = ad.AnnData(
            X=np.asarray(
                [
                    [0.0, 0.0, 0.1, 0.0],
                    [0.0, 0.0, 0.2, 0.0],
                    [0.0, 0.0, 0.0, 0.1],
                    [0.0, 0.0, 0.0, 0.2],
                ],
                dtype=np.float32,
            )
        )
        current.obs["Annotation"] = ["A", "A", "B", "B"]
        slices[str(time)] = current
        communications[str(time)] = {
            "types": np.asarray(["A", "B"], dtype=object),
            "M_per_source": np.asarray([[0.0, 1.0], [0.0, 0.0]]),
        }
    result = project_communication_to_lr_timecourses(
        slices,
        reference,
        communications,
        pd.DataFrame(
            {
                "ligand": ["Wnt3a", "Wnt3a", "Wnt3a"],
                "receptor": ["Fzd7", "Fzd7_Lrp6", "Fzd7_CenterOnly"],
            }
        ),
        n_clusters=1,
    )
    assert result.pair_timecourse["pair"].unique().tolist() == ["Wnt3a_Fzd7"]
    assert result.coverage["n_lr_pairs_scored"].tolist() == [1, 1]
    assert result.coverage["n_lr_pairs_with_unreconstructable_subunit"].tolist() == [
        1,
        1,
    ]
    assert result.coverage["unreconstructable_subunits"].tolist() == [
        "CenterOnly",
        "CenterOnly",
    ]
    assert result.coverage["n_lr_pairs_with_missing_subunit"].tolist() == [1, 1]
    assert result.coverage["missing_subunits"].tolist() == ["Lrp6", "Lrp6"]
    assert result.coverage["n_pca_features_inactive"].tolist() == [1, 1]


def test_hybrid_lr_uses_one_active_pca_subunit_universe_at_all_times() -> None:
    reference = ad.AnnData(X=np.zeros((2, 3), dtype=np.float32))
    reference.var_names = ["L1", "R1", "CenterOnly"]
    reference.varm["PCs"] = np.asarray(
        [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
        dtype=np.float32,
    )
    reference.var["pca_center"] = [1.0, 1.0, 10.0]
    slices = {}
    communications = {}
    for time in (0.0, 0.5, 1.0):
        current = ad.AnnData(
            X=np.asarray(
                [
                    [0.0, 0.0, 0.1, 0.1],
                    [0.0, 0.0, 0.2, 0.2],
                    [0.0, 0.0, 0.1, 0.1],
                    [0.0, 0.0, 0.2, 0.2],
                ],
                dtype=np.float32,
            )
        )
        current.obs["Annotation"] = ["A", "A", "B", "B"]
        slices[str(time)] = current
        communications[str(time)] = {
            "types": np.asarray(["A", "B"], dtype=object),
            "M_per_source": np.asarray([[0.0, 1.0], [0.0, 0.0]]),
        }

    observed = ad.AnnData(
        X=np.asarray(
            [
                [2.0, 1.0, 100.0],
                [1.0, 2.0, 100.0],
                [2.0, 1.0, 100.0],
                [1.0, 2.0, 100.0],
            ],
            dtype=np.float32,
        )
    )
    observed.var_names = reference.var_names.copy()
    observed.obs["time"] = [0.0, 0.0, 1.0, 1.0]
    observed.obs["Annotation"] = ["A", "B", "A", "B"]
    result = project_communication_to_lr_timecourses(
        slices,
        reference,
        communications,
        pd.DataFrame(
            {
                "ligand": ["L1", "L1"],
                "receptor": ["R1", "CenterOnly"],
            }
        ),
        n_clusters=1,
        require_all_subunits=True,
        observed_adata=observed,
        observed_time_key="time",
        observed_time_points=[0.0, 1.0],
        observed_expression_space="count",
    )

    assert result.pair_timecourse["pair"].unique().tolist() == ["L1_R1"]
    assert result.pair_timecourse["time"].tolist() == [0.0, 0.5, 1.0]
    pair_audit = result.trajectory_coverage.loc[
        result.trajectory_coverage["trajectory_kind"] == "pair"
    ].set_index("pair")
    assert bool(pair_audit.loc["L1_R1", "retained"])
    assert not bool(pair_audit.loc["L1_CenterOnly", "retained"])
    assert (
        pair_audit.loc["L1_CenterOnly", "drop_reason"]
        == "not_scoreable_in_uniform_active_pca_universe"
    )
    assert pair_audit.loc["L1_CenterOnly", "inactive_pca_subunits"] == "CenterOnly"
    assert set(result.clustering.assignments["profile"]) == {"L1_R1"}
    assert set(result.dropped_trajectories["pair"]) == {"L1_CenterOnly"}
    contract = result.settings["temporal_eligibility_contract"]
    assert contract["observed_and_generated_feature_universe_identical"] is True
    assert contract["missing_time_policy"] == "drop_and_audit_never_zero_fill"
    assert result.coverage["n_observed_lr_features"].tolist() == [2, 2, 2]


def test_pair_celltype_outputs_drop_incomplete_requested_time_grids() -> None:
    reference = _reference_adata()
    slices = _trajectory_slices()
    communications = {
        "0.0": {
            "types": np.asarray(["A", "B"], dtype=object),
            "M_per_source": np.asarray([[0.0, 1.0], [0.5, 0.0]]),
        },
        "0.5": {
            "types": np.asarray(["A"], dtype=object),
            "M_per_source": np.asarray([[1.0]]),
        },
        "1.0": {
            "types": np.asarray(["A", "B"], dtype=object),
            "M_per_source": np.asarray([[0.0, 1.0], [0.5, 0.0]]),
        },
    }
    result = project_communication_to_lr_timecourses(
        slices,
        reference,
        communications,
        pd.DataFrame({"ligand": ["L1"], "receptor": ["R1"]}),
        preferred_species_tag="hs",
        n_clusters=1,
    )

    assert result.pair_timecourse.groupby("pair")["time"].nunique().tolist() == [3]
    celltype_audit = result.trajectory_coverage.loc[
        result.trajectory_coverage["trajectory_kind"] == "pair_celltype"
    ].set_index(["pair", "cell_type"])
    assert bool(celltype_audit.loc[("L1_R1", "A"), "retained"])
    assert not bool(celltype_audit.loc[("L1_R1", "B"), "retained"])
    assert celltype_audit.loc[("L1_R1", "B"), "missing_times"] == "[0.5]"
    assert set(result.celltype_timecourse["cell_type"]) == {"A"}
    assert result.celltype_timecourse.groupby(["pair", "cell_type"])[
        "time"
    ].nunique().tolist() == [3]


def test_focal_lr_type_hotspot_uses_article_estimand_and_constant_type_mapping() -> (
    None
):
    reference = _reference_adata()
    slices = _trajectory_slices()
    communications = {
        key: {
            "types": np.asarray(["A", "B"], dtype=object),
            "M_per_source": np.asarray([[0.0, 1.0], [0.5, 0.0]]),
        }
        for key in slices
    }
    display_slices = {}
    for key, state in slices.items():
        display = ad.AnnData(X=np.vstack([state.X, state.X]))
        display.obs["Annotation"] = state.obs["Annotation"].astype(str).tolist() * 2
        display.obs_names = [f"{key}_display_{index}" for index in range(display.n_obs)]
        display_slices[key] = display
    result = compute_focal_lr_type_hotspots(
        slices,
        reference,
        communications,
        ligand="L1",
        receptor="R1",
        preferred_species_tag="hs",
        observed_adata=_observed_expression(),
        observed_time_key="stage",
        observed_time_points=[0.0, 1.0],
        observed_annotation_key="cell_type",
        observed_layer="counts",
        observed_expression_space="count",
        cell_mapping_adata_dict=display_slices,
    )

    t0 = result.type_matrix.loc[result.type_matrix["time"] == 0.0].set_index(
        ["sender_type", "receiver_type"]
    )
    assert t0.loc[("A", "B"), "ligand_mean"] == pytest.approx(11.0)
    assert t0.loc[("A", "B"), "receptor_mean"] == pytest.approx(21.0)
    assert t0.loc[("A", "B"), "communication_weight"] == pytest.approx(1.0)
    assert t0.loc[("A", "B"), "lr_score"] == pytest.approx(231.0)
    assert t0.loc[("B", "A"), "lr_score"] == pytest.approx(3.0)
    np.testing.assert_allclose(
        result.type_matrix["lr_score"],
        result.type_matrix["ligand_mean"]
        * result.type_matrix["receptor_mean"]
        * result.type_matrix["communication_weight"],
    )
    assert len(result.type_matrix) == 3 * 2 * 2
    assert len(result.cell_mapping) == 3 * 8
    assert result.type_scores.groupby("time")["cell_type"].nunique().tolist() == [
        2,
        2,
        2,
    ]
    for (_, cell_type), subset in result.cell_mapping.groupby(["time", "cell_type"]):
        assert subset["incoming"].nunique() == 1
        assert subset["outgoing"].nunique() == 1
        assert subset["total_raw"].nunique() == 1
        expected = result.type_scores.loc[
            (result.type_scores["time"] == subset["time"].iloc[0])
            & (result.type_scores["cell_type"] == cell_type),
            "total",
        ].iloc[0]
        assert subset["total_raw"].iloc[0] == pytest.approx(expected)
    assert result.audit["max_formula_abs_error"].max() == pytest.approx(0.0)
    assert result.audit["within_type_cell_scores_constant"].all()
    assert result.audit["n_compute_cells"].tolist() == [4, 4, 4]
    assert result.audit["n_display_cells"].tolist() == [8, 8, 8]
    for key, display in display_slices.items():
        observed_ids = result.cell_mapping.loc[
            np.isclose(result.cell_mapping["time"], float(key)), "cell_id"
        ].tolist()
        assert observed_ids == display.obs_names.astype(str).tolist()
    assert result.settings["aggregation_level"] == "cell_type"
    assert result.settings["per_edge_attention_hotspot"] is False
    assert result.settings["strict_all_subunit_corrected_reanalysis"] is True


def test_focal_lr_type_hotspot_strict_complex_never_omits_requested_subunit() -> None:
    reference = _reference_adata()
    slices = _trajectory_slices()
    communications = {
        key: {
            "types": np.asarray(["A", "B"], dtype=object),
            "M_per_source": np.asarray([[0.0, 1.0], [0.5, 0.0]]),
        }
        for key in slices
    }
    with pytest.raises(ValueError, match="No ligand-receptor pairs could be scored"):
        compute_focal_lr_type_hotspots(
            slices,
            reference,
            communications,
            ligand="L1",
            receptor="R1_MissingSubunit",
            preferred_species_tag="hs",
            require_all_subunits=True,
        )


def test_focal_lr_type_hotspot_keeps_missing_type_unavailable_not_zero() -> None:
    reference = _reference_adata()
    compute = ad.AnnData(X=np.asarray([[0.0, 0.0, 1.0, 1.0]], dtype=np.float32))
    compute.obs["Annotation"] = ["A"]
    compute.obs_names = ["compute_a"]
    display = ad.AnnData(
        X=np.asarray(
            [[0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
            dtype=np.float32,
        )
    )
    display.obs["Annotation"] = ["A", "B"]
    display.obs_names = ["display_a", "display_b"]
    communications = {
        key: {
            "types": np.asarray(["A", "B"], dtype=object),
            "M_per_source": np.ones((2, 2), dtype=np.float64),
        }
        for key in ("0.0", "1.0")
    }

    result = compute_focal_lr_type_hotspots(
        {"0.0": compute, "1.0": compute.copy()},
        reference,
        communications,
        ligand="L1",
        receptor="R1",
        preferred_species_tag="hs",
        cell_mapping_adata_dict={"0.0": display, "1.0": display.copy()},
    )

    scores = result.type_scores.set_index("cell_type")
    assert np.isfinite(scores.loc["A", "total"]).all()
    assert scores.loc["B", "incoming"].isna().all()
    assert scores.loc["B", "outgoing"].isna().all()
    assert scores.loc["B", "total"].isna().all()
    assert not scores.loc["B", "expression_supported_as_sender"].any()
    assert not scores.loc["B", "expression_supported_as_receiver"].any()
    mapped_b = result.cell_mapping.loc[result.cell_mapping["cell_type"] == "B"]
    assert mapped_b[["incoming", "outgoing", "total_raw"]].isna().all().all()
    assert result.settings["unsupported_expression_value"] == (
        "unavailable_nan_never_observed_zero"
    )
