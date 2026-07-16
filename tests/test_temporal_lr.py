from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd

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


def test_inverse_pca_and_gene_aliases() -> None:
    reference = _reference_adata()
    reconstructed = inverse_pca_states(
        reference,
        np.asarray([[0.0, 0.0, 0.2, 0.4]], dtype=np.float32),
    )
    np.testing.assert_allclose(reconstructed[0], [1.2, 2.4, 2.9], atol=1e-6)
    aliases = simplify_gene_names(reference.var_names, preferred_species_tag="hs")
    assert aliases["gene_symbol"].tolist() == ["L1", "R1", "G3"]


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
    assert lr_result.pattern_summary.loc[0, "cluster"] == 1
