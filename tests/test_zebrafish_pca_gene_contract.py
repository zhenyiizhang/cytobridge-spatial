"""Numerical contract tests for PCA-state to processed-gene reconstruction.

The fixture is deliberately dataset-agnostic.  It captures the contract used
by the zebrafish workflow without depending on the large manuscript inputs.
"""

from __future__ import annotations

import anndata as ad
import numpy as np

from CytoBridge.tl.downstream.temporal import (
    inverse_pca_states,
    pca_reconstruction_feature_coverage,
    summarize_temporal_gene_patterns,
)


def _reference() -> ad.AnnData:
    reference = ad.AnnData(
        X=np.asarray(
            [
                [101.0, 201.0, 301.0],
                [103.0, 203.0, 303.0],
            ],
            dtype=np.float32,
        )
    )
    reference.var_names = ["active_a", "active_b", "center_only"]
    reference.varm["PCs"] = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
        ],
        dtype=np.float32,
    )
    # This is intentionally different from the mean of reference.X.  A
    # subsetted AnnData must continue to use the center fitted on the original
    # PCA reference population.
    reference.var["pca_center"] = np.asarray([10.0, 20.0, 30.0], dtype=np.float32)
    return reference


def _joint_states(pc_rows: list[list[float]]) -> ad.AnnData:
    pcs = np.asarray(pc_rows, dtype=np.float32)
    spatial = np.zeros((pcs.shape[0], 2), dtype=np.float32)
    return ad.AnnData(X=np.column_stack([spatial, pcs]))


def test_persisted_center_inverse_and_forward_round_trip() -> None:
    reference = _reference()[[0]].copy()
    states = np.asarray(
        [
            [7.0, 8.0, 1.5, -2.0],
            [9.0, 10.0, -0.5, 3.0],
        ],
        dtype=np.float32,
    )

    reconstructed = inverse_pca_states(reference, states)
    np.testing.assert_allclose(
        reconstructed,
        [[11.5, 18.0, 30.0], [9.5, 23.0, 30.0]],
        atol=1e-7,
    )

    center = reference.var["pca_center"].to_numpy(dtype=np.float32)
    loadings = np.asarray(reference.varm["PCs"], dtype=np.float32)
    projected_again = (reconstructed - center[None, :]) @ loadings
    np.testing.assert_allclose(projected_again, states[:, 2:4], atol=1e-7)


def test_active_feature_coverage_excludes_center_only_gene() -> None:
    reference = _reference()
    coverage = pca_reconstruction_feature_coverage(
        reference.var_names, reference.varm["PCs"]
    )

    assert coverage["feature_name"].tolist() == [
        "active_a",
        "active_b",
        "center_only",
    ]
    assert coverage["active"].tolist() == [True, True, False]

    states = {
        "0.0": _joint_states([[1.0, 2.0], [3.0, 4.0]]),
        "0.5": _joint_states([[5.0, 6.0], [7.0, 8.0]]),
    }
    active_only = summarize_temporal_gene_patterns(
        states,
        reference,
        n_top_genes=2,
        n_clusters=1,
    )
    assert active_only.expression.index.tolist() == ["active_a", "active_b"]
    assert active_only.gene_name_map["pca_active"].tolist() == [True, True]
    assert active_only.settings["pca_features_total"] == 3
    assert active_only.settings["pca_features_active"] == 2
    assert active_only.settings["pca_features_inactive"] == 1

    including_center_only = summarize_temporal_gene_patterns(
        states,
        reference,
        n_top_genes=3,
        n_clusters=1,
        active_features_only=False,
    )
    assert including_center_only.gene_name_map["pca_active"].tolist() == [
        True,
        True,
        False,
    ]
    np.testing.assert_allclose(
        including_center_only.expression.loc["center_only"],
        [30.0, 30.0],
        atol=1e-7,
    )


def test_observed_and_generated_states_share_one_inverse_pca_map() -> None:
    reference = _reference()
    # The integer-time entry represents an observed latent state and the
    # half-time entry a generated state.  Both enter through the identical
    # joint-state/PCA reconstruction contract.
    states = {
        "0.0": _joint_states([[0.0, 2.0], [2.0, 4.0]]),
        "0.5": _joint_states([[4.0, 6.0], [6.0, 8.0]]),
    }
    result = summarize_temporal_gene_patterns(
        states,
        reference,
        n_top_genes=2,
        n_clusters=1,
    )

    expected_columns = []
    for key in ("0.0", "0.5"):
        mean_joint_state = np.asarray(states[key].X).mean(axis=0, keepdims=True)
        expected_columns.append(inverse_pca_states(reference, mean_joint_state)[0, :2])
    expected = np.stack(expected_columns, axis=1)
    np.testing.assert_allclose(result.expression.to_numpy(), expected, atol=1e-7)
    np.testing.assert_allclose(result.signed_expression.to_numpy(), expected, atol=1e-7)
    assert result.settings["expression_space"] == (
        "per-cell clipped rank-retained inverse-PCA processed log1p"
    )
    assert result.settings["signed_expression_space"] == (
        "signed rank-retained inverse-PCA processed log1p score"
    )
    assert result.settings["observed_generated_comparability"] == (
        "all time points use the same inverse-PCA map and persisted fit center"
    )
    assert result.settings["aggregation"] == (
        "per-cell inverse-PCA, optional clipping, arithmetic mean"
    )
    assert result.settings["clip_min"] == 0.0
    assert (result.reconstruction_diagnostics["n_values_below_clip_min"] == 0).all()


def test_temporal_summary_clips_per_cell_and_reports_signed_diagnostics() -> None:
    reference = _reference()
    states = {
        # active_a reconstructs to [-10, 30].  Its signed mean is 10, while
        # clip-each-cell-then-mean is 15; clipping the mean would wrongly give
        # 10 and is therefore distinguishable in this fixture.
        "0.0": _joint_states([[-20.0, 0.0], [20.0, 0.0]]),
        "0.5": _joint_states([[-10.0, 0.0], [30.0, 0.0]]),
    }
    result = summarize_temporal_gene_patterns(
        states,
        reference,
        n_top_genes=2,
        n_clusters=1,
        reconstruction_batch_size=1,
    )

    assert result.signed_expression.loc["active_a", 0.0] == 10.0
    assert result.expression.loc["active_a", 0.0] == 15.0
    assert result.expression.loc["active_a", 0.0] != max(
        result.signed_expression.loc["active_a", 0.0], 0.0
    )

    diagnostics = result.reconstruction_diagnostics.set_index("time")
    assert diagnostics.loc[0.0, "n_cells"] == 2
    assert diagnostics.loc[0.0, "n_features"] == 2
    assert diagnostics.loc[0.0, "n_values"] == 4
    assert diagnostics.loc[0.0, "preclip_min"] == -10.0
    assert diagnostics.loc[0.0, "n_values_below_clip_min"] == 1
    assert diagnostics.loc[0.0, "fraction_below_clip_min"] == 0.25
    assert diagnostics.loc[0.0, "clip_min"] == 0.0
    assert diagnostics.loc[0.0, "postclip_min"] == 0.0
