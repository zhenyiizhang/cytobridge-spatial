from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from CytoBridge.tl import summarize_temporal_gene_patterns


def _candidate_fixture(
    *,
    n_hvg: int = 2000,
    n_forced: int = 747,
) -> tuple[ad.AnnData, dict[str, ad.AnnData], list[str], list[str]]:
    hvg_names = [f"hvg_{index:04d}" for index in range(n_hvg)]
    forced_names = [f"forced_{index:04d}" for index in range(n_forced)]
    reference = ad.AnnData(X=np.zeros((2, n_hvg + n_forced), dtype=np.float32))
    reference.var_names = hvg_names + forced_names
    loadings = np.zeros((n_hvg + n_forced, 2), dtype=np.float32)
    loadings[:n_hvg, 0] = np.linspace(0.001, 0.02, n_hvg)
    loadings[n_hvg:, 0] = np.linspace(0.2, 1.0, n_forced)
    # Keep the second PC non-degenerate while the test trajectory varies PC1.
    loadings[:, 1] = 0.001
    reference.varm["PCs"] = loadings
    reference.var["pca_center"] = np.zeros(
        n_hvg + n_forced,
        dtype=np.float32,
    )
    reference.var["pca_center"] = np.zeros(n_hvg + n_forced, dtype=np.float32)

    slices: dict[str, ad.AnnData] = {}
    for time in (0.0, 1.0, 2.0):
        pcs = np.tile(np.asarray([time, 0.0], dtype=np.float32), (3, 1))
        states = np.column_stack((np.zeros((3, 2), dtype=np.float32), pcs))
        slices[str(time)] = ad.AnnData(X=states)
    return reference, slices, hvg_names, forced_names


def _summarize(reference, slices, **kwargs):
    return summarize_temporal_gene_patterns(
        slices,
        reference,
        n_top_genes=10,
        n_cluster_genes=10,
        n_clusters=1,
        **kwargs,
    )


def test_candidate_universe_excludes_high_variance_forced_pca_features() -> None:
    reference, slices, hvg_names, forced_names = _candidate_fixture()
    unrestricted = _summarize(reference, slices)
    assert unrestricted.expression.shape[0] == 2747
    assert set(unrestricted.top_variable_genes["gene"]).issubset(forced_names)

    # Reversed input proves candidate_features is a set contract; output stays
    # aligned to PCA reconstruction feature order.
    restricted = _summarize(
        reference,
        slices,
        candidate_features=list(reversed(hvg_names)),
    )
    assert restricted.expression.index.tolist() == hvg_names
    assert restricted.expression.shape[0] == 2000
    assert not any(
        gene.startswith("forced_")
        for gene in restricted.top_variable_genes["gene"].astype(str)
    )
    provenance = restricted.settings["candidate_features"]
    assert provenance["policy"] == "strict"
    assert provenance["match_space"] == "exact_pca_reconstruction_feature_name"
    assert provenance["requested_count"] == 2000
    assert provenance["used_count"] == 2000
    assert provenance["missing_count"] == 0
    assert provenance["inactive_count"] == 0
    assert provenance["used"] == hvg_names
    assert restricted.settings["pca_features_active"] == 2747
    contract = restricted.settings["pca_contract"]
    assert contract["center_source"] == "reference_adata.var['pca_center']"
    assert contract["feature_count"] == 2747
    assert contract["component_count"] == 2
    assert restricted.settings["expression_contract"] == {
        "source_policy": "inverse_pca_all_timepoints",
        "output_space": "mean_per_cell_clipped_log1p_expression",
        "signed_output_space": "mean_signed_log1p_expression",
        "aggregation_identity": (
            "nonlinear clipping applied per cell before arithmetic mean; "
            "inverse_pca(mean) identity is intentionally not used"
        ),
        "count_space_conversion": "not_applied",
    }

    changed_reference = reference.copy()
    changed_loadings = np.asarray(changed_reference.varm["PCs"]).copy()
    changed_loadings[0, 0] *= 2.0
    changed_reference.varm["PCs"] = changed_loadings
    changed = _summarize(
        changed_reference,
        slices,
        candidate_features=hvg_names,
    )
    assert not np.array_equal(
        changed.expression.loc[hvg_names[0]].to_numpy(),
        restricted.expression.loc[hvg_names[0]].to_numpy(),
    )


def test_candidate_features_reject_duplicates_and_missing_names() -> None:
    reference, slices, hvg_names, _ = _candidate_fixture(n_hvg=4, n_forced=2)
    with pytest.raises(ValueError, match="unique.*duplicates"):
        _summarize(
            reference,
            slices,
            candidate_features=[hvg_names[0], hvg_names[0]],
        )
    with pytest.raises(ValueError, match="absent.*examples"):
        _summarize(
            reference,
            slices,
            candidate_features=[hvg_names[0], "not_in_pca"],
        )
    with pytest.raises(ValueError, match="at least one"):
        _summarize(reference, slices, candidate_features=[])
    with pytest.raises(TypeError, match="not a single string"):
        _summarize(reference, slices, candidate_features=hvg_names[0])


def test_candidate_features_reject_center_only_feature_even_in_compat_mode() -> None:
    reference, slices, hvg_names, _ = _candidate_fixture(n_hvg=4, n_forced=2)
    loadings = np.asarray(reference.varm["PCs"]).copy()
    loadings[0] = 0.0
    reference.varm["PCs"] = loadings
    with pytest.raises(ValueError, match="inactive center-only PCA features"):
        _summarize(
            reference,
            slices,
            candidate_features=[hvg_names[0]],
            active_features_only=False,
        )


def test_candidate_features_none_preserves_existing_result_contract() -> None:
    reference, slices, _, _ = _candidate_fixture(n_hvg=5, n_forced=3)
    implicit = _summarize(reference, slices)
    explicit = _summarize(reference, slices, candidate_features=None)
    pd.testing.assert_frame_equal(implicit.expression, explicit.expression)
    pd.testing.assert_frame_equal(
        implicit.top_variable_genes,
        explicit.top_variable_genes,
    )
    pd.testing.assert_frame_equal(
        implicit.clustering.assignments,
        explicit.clustering.assignments,
    )
    assert implicit.settings == explicit.settings
    provenance = explicit.settings["candidate_features"]
    assert provenance["policy"] == "not_requested"
    assert provenance["requested"] is None
    assert provenance["requested_count"] is None
    assert provenance["used_count"] == 8
