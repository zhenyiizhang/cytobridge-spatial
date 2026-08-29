from __future__ import annotations

import json

import numpy as np
import pandas as pd
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


def test_preprocess_forwards_batch_key_to_hvg_selection(monkeypatch) -> None:
    adata = AnnData(
        X=np.asarray(
            [
                [1.0, 0.0, 2.0],
                [0.0, 3.0, 1.0],
                [4.0, 1.0, 0.0],
                [1.0, 2.0, 1.0],
            ],
            dtype=np.float32,
        ),
        obs=pd.DataFrame(
            {"time": [0, 0, 1, 1], "Batch": ["a", "a", "b", "b"]},
            index=["c0", "c1", "c2", "c3"],
        ),
    )
    captured = {}

    def fake_hvg(target, *, n_top_genes, batch_key):
        captured["n_top_genes"] = n_top_genes
        captured["batch_key"] = batch_key
        target.var["highly_variable"] = [True, False, True]

    monkeypatch.setattr("scanpy.pp.highly_variable_genes", fake_hvg)

    result = preprocess(
        adata,
        time_key="time",
        normalization=False,
        log1p=False,
        select_hvg=True,
        n_top_genes=2,
        dim_reduction="none",
        hvg_batch_key="Batch",
    )

    assert captured == {"n_top_genes": 2, "batch_key": "Batch"}
    assert result.uns["preprocess_info"]["hvg_batch_key"] == "Batch"


def test_preprocess_can_rank_hvgs_before_clean_latent_normalization(monkeypatch) -> None:
    counts = np.asarray(
        [
            [4.0, 1.0, 2.0, 0.0],
            [1.0, 5.0, 3.0, 1.0],
            [2.0, 0.0, 6.0, 2.0],
            [3.0, 2.0, 1.0, 4.0],
        ],
        dtype=np.float32,
    )
    adata = AnnData(
        X=sparse.csr_matrix(np.log1p(counts)),
        obs=pd.DataFrame(
            {
                "time": ["fit-a", "fit-a", "fit-b", "reference-only"],
                "Batch": ["a", "a", "b", "reference"],
            },
            index=["c0", "c1", "c2", "c3"],
        ),
        var=pd.DataFrame(index=["g0", "g1", "g2", "g3"]),
    )
    adata.layers["counts"] = sparse.csr_matrix(counts)
    captured = {}

    def fake_hvg(target, *, n_top_genes, batch_key):
        captured["n_obs"] = target.n_obs
        captured["matrix"] = target.X.toarray().copy()
        captured["batch_key"] = batch_key
        target.var["highly_variable"] = [True, False, True, False]

    monkeypatch.setattr("scanpy.pp.highly_variable_genes", fake_hvg)

    result = preprocess(
        adata,
        time_key="time",
        normalization=True,
        normalization_target_sum=None,
        log1p=True,
        select_hvg=True,
        n_top_genes=2,
        dim_reduction="none",
        expression_layer="counts",
        raw_count_validation="strict",
        hvg_batch_key="Batch",
        hvg_selection_transform="log1p_counts",
        normalization_reference="latent_features",
        required_latent_features=("g3",),
        latent_fit_obs_values=("fit-a", "fit-b"),
    )

    np.testing.assert_allclose(captured["matrix"], np.log1p(counts))
    assert captured["n_obs"] == 4
    assert captured["batch_key"] == "Batch"
    selected_counts = counts[:3]
    reference_totals = selected_counts[:, [0, 2, 3]].sum(axis=1)
    target = float(np.median(reference_totals))
    expected = np.log1p(
        selected_counts * (target / reference_totals)[:, None]
    )
    np.testing.assert_allclose(result.X.toarray(), expected, rtol=1e-6, atol=1e-7)
    assert result.obs_names.tolist() == ["c0", "c1", "c2"]
    assert result.var["highly_variable"].tolist() == [True, False, True, True]
    info = result.uns["preprocess_info"]
    assert info["hvg_selection_transform"] == "log1p_counts"
    assert info["normalization_reference"] == "latent_features"
    assert info["normalization_reference_feature_count"] == 3
    assert info["normalization_target_sum_resolved"] == target
    assert info["hvg_fit_n_obs"] == 4
    assert info["latent_fit_n_obs"] == 3
    assert info["latent_fit_obs_values"] == ["fit-a", "fit-b"]
    assert info["n_statistical_hvgs"] == 2
    assert info["n_latent_fit_features"] == 3


def test_latent_fit_scope_requires_pre_normalization_hvg_ranking() -> None:
    with pytest.raises(ValueError, match="latent_fit_obs_values"):
        preprocess(
            _example_adata(),
            time_key="time",
            dim_reduction="none",
            hvg_selection_transform="post_transform",
            latent_fit_obs_values=("2DPI",),
        )


def test_preprocess_filters_label_blind_spatial_outlier_before_latent_fit() -> None:
    counts = np.asarray(
        [
            [1.0, 2.0],
            [2.0, 1.0],
            [1.0, 3.0],
            [3.0, 1.0],
            [2.0, 2.0],
            [4.0, 1.0],
        ],
        dtype=np.float32,
    )
    obs = pd.DataFrame(
        {
            "time": ["t0"] * 6,
            "Batch": ["batch-a"] * 6,
            "CellID": [f"cell-{index}" for index in range(6)],
            "Annotation": ["common"] * 5 + ["rare"],
        },
        index=[f"row-{index}" for index in range(6)],
    )
    adata = AnnData(X=counts, obs=obs)
    adata.obsm["spatial"] = np.asarray(
        [[0.0, 0.0], [1.0, 0.0], [2.1, 0.0], [3.3, 0.0], [4.7, 0.0], [50.0, 0.0]],
        dtype=np.float32,
    )

    result = preprocess(
        adata,
        time_key="time",
        normalization=False,
        log1p=False,
        select_hvg=False,
        dim_reduction="none",
        observation_id_keys=("Batch", "CellID"),
        spatial_outlier_filter=True,
        spatial_outlier_key="spatial",
        spatial_outlier_group_key="Batch",
        spatial_outlier_nn_mad_z_threshold=50.0,
    )

    assert result.n_obs == 5
    assert "Batch=batch-a|CellID=cell-5" not in result.obs_names
    info = json.loads(result.uns["preprocess_info"]["spatial_outlier_filter_json"])
    assert info["label_blind"] is True
    assert info["n_input"] == 6
    assert info["n_removed"] == 1
    assert info["n_retained"] == 5
    assert info["removed_observations"][0]["obs_name"] == (
        "Batch=batch-a|CellID=cell-5"
    )
    assert info["removed_observations"][0]["robust_nn_z"] > 50.0


def test_spatial_outlier_filter_is_disabled_by_default() -> None:
    adata = _example_adata()
    adata.obsm["spatial"] = np.asarray(
        [[0.0, 0.0], [1.0, 0.0], [100.0, 0.0]], dtype=np.float32
    )
    result = preprocess(
        adata,
        time_key="time",
        normalization=False,
        log1p=False,
        select_hvg=False,
        dim_reduction="none",
    )
    assert result.n_obs == 3
    info = json.loads(result.uns["preprocess_info"]["spatial_outlier_filter_json"])
    assert info == {
        "enabled": False,
        "method": "not_applied",
        "n_input": 3,
        "n_removed": 0,
        "n_retained": 3,
    }


def test_preprocess_rejects_missing_hvg_batch_column() -> None:
    with pytest.raises(KeyError, match="hvg_batch_key"):
        preprocess(
            _example_adata(),
            time_key="time",
            normalization=False,
            log1p=False,
            select_hvg=True,
            dim_reduction="none",
            hvg_batch_key="missing_batch",
        )


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


def test_preprocess_requires_declared_identity_for_duplicate_names() -> None:
    adata = AnnData(
        X=np.ones((4, 2), dtype=np.float32),
        obs={
            "time": ["t0", "t0", "t1", "t1"],
            "Batch": ["a", "b", "a", "b"],
            "CellID": ["cell", "cell", "other", "third"],
        },
    )
    adata.obs_names = ["cell", "cell", "other", "third"]

    with pytest.raises(ValueError, match="observation_id_keys"):
        preprocess(
            adata.copy(),
            time_key="time",
            normalization=False,
            log1p=False,
            dim_reduction="none",
            select_hvg=False,
        )

    result = preprocess(
        adata,
        time_key="time",
        normalization=False,
        log1p=False,
        dim_reduction="none",
        select_hvg=False,
        observation_id_keys=("Batch", "CellID"),
    )

    assert result.obs_names.is_unique
    assert result.obs_names.tolist() == [
        "Batch=a|CellID=cell",
        "Batch=b|CellID=cell",
        "Batch=a|CellID=other",
        "Batch=b|CellID=third",
    ]
    assert result.obs["original_obs_name"].tolist() == [
        "cell",
        "cell",
        "other",
        "third",
    ]
    assert result.uns["preprocess_info"]["observation_names"] == {
        "input_names_unique": False,
        "duplicate_rows": 2,
        "duplicate_values": 1,
        "strategy": "composite_obs_columns",
        "identity_keys": ["Batch", "CellID"],
        "original_name_column": "original_obs_name",
    }


def test_preprocess_leaves_unique_observation_names_unchanged() -> None:
    adata = _example_adata()
    adata.obs_names = ["cell-a", "cell-b", "cell-c"]

    result = preprocess(
        adata,
        time_key="time",
        normalization=False,
        log1p=False,
        dim_reduction="none",
        select_hvg=False,
    )

    assert result.obs_names.tolist() == ["cell-a", "cell-b", "cell-c"]
    assert "original_obs_name" not in result.obs
    assert result.uns["preprocess_info"]["observation_names"] == {
        "input_names_unique": True,
        "duplicate_rows": 0,
        "duplicate_values": 0,
        "strategy": "existing_index",
        "identity_keys": [],
        "original_name_column": "none",
    }


def test_composite_observation_identity_is_stable_under_row_reordering() -> None:
    adata = AnnData(
        X=np.ones((3, 1), dtype=np.float32),
        obs={
            "time": ["t0", "t0", "t1"],
            "sample": ["sample|a", "sample|a", "sample=b"],
            "cell_id": ["cell=1", "cell|2", "cell\\3"],
        },
    )
    adata.obs_names = ["cell", "cell", "cell"]

    first = preprocess(
        adata.copy(),
        time_key="time",
        normalization=False,
        log1p=False,
        dim_reduction="none",
        select_hvg=False,
        observation_id_keys=("sample", "cell_id"),
    )
    reordered = preprocess(
        adata[[2, 0, 1]].copy(),
        time_key="time",
        normalization=False,
        log1p=False,
        dim_reduction="none",
        select_hvg=False,
        observation_id_keys=("sample", "cell_id"),
    )

    assert set(first.obs_names) == set(reordered.obs_names)
    assert all("%" in name for name in first.obs_names)


@pytest.mark.parametrize(
    ("sample_values", "match"),
    ((["s", None], "is missing"), (["s", "s"], "do not form a unique")),
)
def test_composite_observation_identity_rejects_invalid_columns(
    sample_values,
    match,
) -> None:
    adata = AnnData(
        X=np.ones((2, 1), dtype=np.float32),
        obs={
            "time": ["t0", "t1"],
            "sample": sample_values,
            "cell_id": ["cell", "cell"],
        },
    )
    adata.obs_names = ["cell", "cell"]

    with pytest.raises(ValueError, match=match):
        preprocess(
            adata,
            time_key="time",
            normalization=False,
            log1p=False,
            dim_reduction="none",
            select_hvg=False,
            observation_id_keys=("sample", "cell_id"),
        )


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
    assert (
        result.uns["preprocess_info"]["selected_expression_stats"][
            "integer_like_fraction"
        ]
        == 1.0
    )


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

    np.testing.assert_array_equal(
        result.layers["counts"].toarray(), stale_counts.toarray()
    )
    np.testing.assert_array_equal(
        result.layers["raw_counts"].toarray(), raw_counts.toarray()
    )
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


def test_json_numeric_time_mapping_matches_float_observations() -> None:
    adata = AnnData(
        X=np.ones((3, 1), dtype=np.float32),
        obs={"time": np.asarray([1.0, 2.0, 3.0], dtype=np.float64)},
    )

    result = preprocess(
        adata,
        time_key="time",
        time_mapping={"1": 0.0, "2": 1.0, "3": 2.0},
        normalization=False,
        log1p=False,
        dim_reduction="none",
        select_hvg=False,
    )

    assert result.obs["time_point_processed"].tolist() == [0.0, 1.0, 2.0]


def test_numeric_time_mapping_rejects_ambiguous_string_keys() -> None:
    adata = AnnData(
        X=np.ones((1, 1), dtype=np.float32),
        obs={"time": np.asarray([1.0], dtype=np.float64)},
    )

    with pytest.raises(ValueError, match="ambiguous equivalent keys"):
        preprocess(
            adata,
            time_key="time",
            time_mapping={"1": 0.0, "1.0": 0.0},
            normalization=False,
            log1p=False,
            dim_reduction="none",
            select_hvg=False,
        )
