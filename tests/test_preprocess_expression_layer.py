from __future__ import annotations

import numpy as np
import pytest
from anndata import AnnData
from scipy import sparse

from CytoBridge.pp import preprocess


def _example_adata() -> AnnData:
    counts = np.asarray(
        [
            [1.0, 3.0, 0.0],
            [2.0, 0.0, 2.0],
            [0.0, 1.0, 4.0],
        ],
        dtype=np.float32,
    )
    adata = AnnData(X=sparse.csr_matrix(np.log1p(counts)))
    adata.layers["counts"] = sparse.csr_matrix(counts)
    adata.obs["time"] = ["2DPI", "5DPI", "10DPI"]
    return adata


def test_preprocess_can_restore_counts_before_normalize_log1p() -> None:
    adata = _example_adata()
    counts = adata.layers["counts"].toarray().copy()

    result = preprocess(
        adata,
        time_key="time",
        normalization_target_sum=10.0,
        dim_reduction="none",
        select_hvg=False,
        expression_layer="counts",
    )

    expected = np.log1p(counts * (10.0 / counts.sum(axis=1))[:, None])
    np.testing.assert_allclose(result.X.toarray(), expected, rtol=1e-6, atol=1e-7)
    np.testing.assert_array_equal(result.layers["counts"].toarray(), counts)
    assert result.uns["preprocess_info"]["expression_source"] == "layers['counts']"
    assert result.uns["preprocess_info"]["expression_layer"] == "counts"


def test_preprocess_existing_x_behavior_remains_explicit() -> None:
    adata = _example_adata()
    original_x = adata.X.toarray().copy()

    result = preprocess(
        adata,
        time_key="time",
        normalization=False,
        log1p=False,
        dim_reduction="none",
        select_hvg=False,
    )

    np.testing.assert_array_equal(result.X.toarray(), original_x)
    assert result.uns["preprocess_info"]["expression_source"] == "X"
    assert result.uns["preprocess_info"]["expression_layer"] == "none"


def test_preprocess_blocks_silent_double_log() -> None:
    adata = _example_adata()

    with pytest.raises(ValueError, match="double-transform expression"):
        preprocess(
            adata,
            time_key="time",
            dim_reduction="none",
            select_hvg=False,
        )


def test_preprocess_detects_near_log1p_relation_with_small_mismatch() -> None:
    adata = _example_adata()
    # Keep the relation overwhelmingly log1p while introducing a small legacy
    # mismatch, matching the ARISTA source-file audit contract.
    counts = np.ones((2, 6000), dtype=np.float32)
    transformed = np.log1p(counts)
    transformed[0, 0] += 0.5
    adata = AnnData(X=sparse.csr_matrix(transformed))
    adata.layers["counts"] = sparse.csr_matrix(counts)
    adata.obs["time"] = ["2DPI", "5DPI"]

    with pytest.raises(ValueError, match="double-transform expression"):
        preprocess(
            adata,
            time_key="time",
            dim_reduction="none",
            select_hvg=False,
        )


def test_counts_source_clears_stale_log1p_marker() -> None:
    adata = _example_adata()
    adata.uns["log1p"] = {"base": None}

    result = preprocess(
        adata,
        time_key="time",
        normalization_target_sum=10.0,
        dim_reduction="none",
        select_hvg=False,
        expression_layer="counts",
    )

    assert result.uns["preprocess_info"]["preexisting_log1p_marker"] is True
    assert result.uns["preprocess_info"]["input_x_state_detected"]["state"] in {
        "near_log1p_of_counts",
        "transformed_from_metadata",
    }
    assert result.uns["preprocess_info"]["selected_expression_stats"]["integer_like_fraction"] == 1.0


def test_preprocess_rejects_missing_expression_layer() -> None:
    adata = _example_adata()

    with pytest.raises(KeyError, match="expression_layer 'raw_counts'"):
        preprocess(
            adata,
            time_key="time",
            dim_reduction="none",
            select_hvg=False,
            expression_layer="raw_counts",
        )
