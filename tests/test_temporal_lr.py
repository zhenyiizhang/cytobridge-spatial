from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from CytoBridge.tl.downstream.lr_projection import (
    project_communication_to_lr_timecourses,
)
from CytoBridge.tl.downstream.temporal import (
    cluster_temporal_profiles,
    inverse_pca_states,
    load_pca_reconstruction_spec,
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


def test_persisted_pca_center_survives_reference_population_subsetting() -> None:
    reference = _reference_adata()
    fit_center = np.asarray([10.0, 20.0, 30.0], dtype=np.float32)
    reference.var["pca_center"] = fit_center
    subset = reference[[0, 1]].copy()

    reconstructed = inverse_pca_states(
        subset,
        np.asarray([[0.0, 0.0, 0.2, 0.4]], dtype=np.float32),
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
    expected_t0 = np.expm1(1.05) * (
        np.expm1(2.1) + 0.5 * np.expm1(2.0)
    )
    score_t0 = lr_result.pair_timecourse.set_index("time").loc[0.0, "score"]
    assert score_t0 == pytest.approx(expected_t0)
    assert lr_result.settings["observed_expression"] is None
    assert lr_result.pattern_summary.loc[0, "cluster"] == 1


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
