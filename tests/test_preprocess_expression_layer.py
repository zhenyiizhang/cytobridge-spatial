from __future__ import annotations

import json

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


def test_preprocess_persists_exact_pca_fit_center() -> None:
    counts = np.asarray(
        [
            [1.0, 2.0, 0.0, 3.0],
            [2.0, 1.0, 1.0, 0.0],
            [0.0, 3.0, 2.0, 1.0],
            [4.0, 0.0, 1.0, 2.0],
            [1.0, 1.0, 4.0, 0.0],
        ],
        dtype=np.float32,
    )
    adata = AnnData(X=sparse.csr_matrix(np.log1p(counts)))
    adata.layers["counts"] = sparse.csr_matrix(counts)
    adata.obs["time"] = ["t0", "t0", "t1", "t1", "t2"]

    result = preprocess(
        adata,
        time_key="time",
        normalization_target_sum=10.0,
        n_pcs=2,
        select_hvg=False,
        expression_layer="counts",
    )

    processed = np.log1p(counts * (10.0 / counts.sum(axis=1))[:, None])
    np.testing.assert_allclose(
        result.var["pca_center"].to_numpy(),
        processed.mean(axis=0),
        rtol=1e-6,
        atol=1e-7,
    )
    assert result.uns["pca_center_info"]["n_obs_fit"] == 5


def test_preprocess_accepts_none_dim_reduction() -> None:
    adata = _example_adata()
    original_x = adata.X.toarray().copy()

    result = preprocess(
        adata,
        time_key="time",
        normalization=False,
        log1p=False,
        dim_reduction=None,
        select_hvg=False,
    )

    np.testing.assert_allclose(result.obsm["X_latent"], original_x)
    assert result.uns["preprocess_info"]["dim_reduction"] == "none"


def test_explicit_layer_gets_strict_raw_count_validation_in_auto_mode() -> None:
    adata = _example_adata()
    non_counts = adata.layers["counts"].copy().tolil()
    non_counts[1, 1] = 0.25
    adata.layers["normalized_but_misnamed"] = non_counts.tocsr()

    with pytest.raises(ValueError, match="strict raw-count-like validation"):
        preprocess(
            adata,
            time_key="time",
            dim_reduction="none",
            select_hvg=False,
            expression_layer="normalized_but_misnamed",
        )


def test_raw_count_validation_off_is_explicit_compatibility_escape_hatch() -> None:
    adata = _example_adata()
    adata.layers["documented_non_counts"] = adata.layers["counts"].multiply(0.5)

    result = preprocess(
        adata,
        time_key="time",
        normalization=False,
        log1p=False,
        dim_reduction="none",
        select_hvg=False,
        expression_layer="documented_non_counts",
        raw_count_validation="off",
    )

    assert result.uns["preprocess_info"]["raw_count_validation_effective"] == "basic"
    assert result.uns["preprocess_info"]["raw_counts_layer"] == "documented_non_counts"


def test_explicit_raw_layer_remains_canonical_when_counts_layer_is_stale() -> None:
    adata = _example_adata()
    raw_counts = adata.layers["counts"].copy()
    stale_counts = sparse.csr_matrix(np.full(adata.shape, 99.0, dtype=np.float32))
    adata.layers["raw_counts"] = raw_counts
    adata.layers["counts"] = stale_counts

    result = preprocess(
        adata,
        time_key="time",
        normalization_target_sum=10.0,
        dim_reduction="none",
        select_hvg=False,
        expression_layer="raw_counts",
    )

    np.testing.assert_array_equal(result.layers["counts"].toarray(), stale_counts.toarray())
    np.testing.assert_array_equal(result.layers["raw_counts"].toarray(), raw_counts.toarray())
    info = result.uns["preprocess_info"]
    assert info["raw_counts_layer"] == "raw_counts"
    assert info["counts_layer"] == "raw_counts"
    assert info["raw_count_validation_effective"] == "strict"


def test_strict_dense_validation_checks_rows_beyond_diagnostic_sample() -> None:
    raw_counts = np.ones((300, 2), dtype=np.float32)
    raw_counts[257, 1] = 1.25
    adata = AnnData(X=np.log1p(raw_counts))
    adata.layers["raw_counts"] = raw_counts
    adata.obs["time"] = ["t0"] * 150 + ["t1"] * 150

    with pytest.raises(ValueError, match="1 values outside integer tolerance"):
        preprocess(
            adata,
            time_key="time",
            dim_reduction="none",
            select_hvg=False,
            expression_layer="raw_counts",
        )


def test_numeric_time_mapping_provenance_is_h5ad_safe(tmp_path) -> None:
    adata = AnnData(X=np.ones((4, 2), dtype=np.float32))
    adata.obs["time"] = [1, 1, 2, 2]

    result = preprocess(
        adata,
        time_key="time",
        time_mapping={"1": 0.0, "2": 1.0},
        normalization=False,
        log1p=False,
        dim_reduction=None,
        select_hvg=False,
    )
    output_path = tmp_path / "numeric_time_mapping.h5ad"
    result.write_h5ad(output_path)

    records = json.loads(result.uns["preprocess_info"]["time_mapping_json"])
    assert records == [
        {"source": "1", "source_type": "int", "target": 0.0},
        {"source": "2", "source_type": "int", "target": 1.0},
    ]
    assert result.obs["time_point_processed"].tolist() == [0.0, 0.0, 1.0, 1.0]
