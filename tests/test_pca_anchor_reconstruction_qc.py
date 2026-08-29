from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from CytoBridge.tl import (
    evaluate_pca_anchor_reconstruction,
    make_pca_reconstruction_spec,
)


def _qc_fixture() -> tuple[ad.AnnData, np.ndarray, np.ndarray, np.ndarray]:
    feature_names = ["A", "B", "C"]
    loadings = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, -1.0],
        ],
        dtype=np.float32,
    )
    center = np.asarray([0.1, 0.2, 0.0], dtype=np.float32)
    latent = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
        ],
        dtype=np.float32,
    )
    # The implementation canonicalizes the persisted float32 PCA contract to
    # float64 before multiplying; mirror that exact contract in expectations.
    reconstructed = (
        latent.astype(np.float64) @ loadings.astype(np.float64).T
        + center.astype(np.float64)[None, :]
    )
    observed = np.asarray(
        [
            [0.2, 0.1, 0.0],
            [0.9, 0.2, 1.2],
            [0.1, 1.4, 0.0],
            [0.0, 0.0, 0.3],
        ],
        dtype=np.float32,
    )
    adata = ad.AnnData(X=np.full_like(observed, 123.0))
    adata.var_names = feature_names
    adata.obs["stage"] = [1.0, 1.0, 2.0, 2.0]
    adata.obsm["X_pca"] = latent
    adata.varm["PCs"] = loadings
    adata.var["pca_center"] = center
    adata.layers["log1p"] = sparse.csr_matrix(observed)
    return (
        adata,
        observed.astype(np.float64),
        reconstructed.astype(np.float64),
        loadings,
    )


def _expected_aggregate(
    observed: np.ndarray, reconstructed: np.ndarray
) -> pd.DataFrame:
    rows = []
    for time, indices in ((1.0, np.asarray([0, 1])), (2.0, np.asarray([2, 3]))):
        current_observed = observed[indices]
        current_reconstructed = reconstructed[indices]
        error = current_reconstructed - current_observed
        sse = float(np.square(error).sum())
        feature_sst = np.square(
            current_observed - current_observed.mean(axis=0, keepdims=True)
        ).sum(axis=0)
        rows.append(
            {
                "time": time,
                "rmse": np.sqrt(sse / error.size),
                "mae": np.abs(error).mean(),
                "r2": 1.0 - sse / feature_sst.sum(),
                "negative_reconstructed_fraction": np.mean(current_reconstructed < 0.0),
            }
        )
    return pd.DataFrame(rows)


def test_anchor_reconstruction_qc_is_chunked_exact_and_auditable() -> None:
    adata, observed, reconstructed, _ = _qc_fixture()
    chunked = evaluate_pca_anchor_reconstruction(
        adata,
        latent_key="X_pca",
        time_key="stage",
        expression_layer="log1p",
        chunk_size=1,
    )
    batched = evaluate_pca_anchor_reconstruction(
        adata,
        latent_key="X_pca",
        time_key="stage",
        expression_layer="log1p",
        chunk_size=3,
    )
    pd.testing.assert_frame_equal(
        chunked.aggregate_metrics,
        batched.aggregate_metrics,
        check_exact=False,
        rtol=1e-13,
        atol=1e-13,
    )
    pd.testing.assert_frame_equal(
        chunked.per_feature_metrics,
        batched.per_feature_metrics,
        check_exact=False,
        rtol=1e-13,
        atol=1e-13,
    )

    expected = _expected_aggregate(observed, reconstructed)
    observed_aggregate = chunked.aggregate_metrics
    assert observed_aggregate["time"].tolist() == [1.0, 2.0]
    assert observed_aggregate["n_cells_effective"].tolist() == [2, 2]
    assert observed_aggregate["n_features_effective"].tolist() == [3, 3]
    assert observed_aggregate["n_values_effective"].tolist() == [6, 6]
    for column in ("rmse", "mae", "r2", "negative_reconstructed_fraction"):
        np.testing.assert_allclose(
            observed_aggregate[column],
            expected[column],
            rtol=1e-7,
            atol=1e-10,
        )
    assert observed_aggregate["scale"].tolist() == ["log1p", "log1p"]

    per_feature = chunked.per_feature_metrics
    assert per_feature.shape == (6, 14)
    assert per_feature.groupby("time", sort=True)["feature"].apply(list).to_dict() == {
        1.0: ["A", "B", "C"],
        2.0: ["A", "B", "C"],
    }
    for time, indices in ((1.0, [0, 1]), (2.0, [2, 3])):
        error = reconstructed[indices] - observed[indices]
        current = per_feature.loc[per_feature["time"] == time]
        np.testing.assert_allclose(
            current["rmse"],
            np.sqrt(np.square(error).mean(0)),
            atol=1e-7,
        )
        np.testing.assert_allclose(current["mae"], np.abs(error).mean(0), atol=1e-7)
        current_observed = observed[indices]
        current_reconstructed = reconstructed[indices]
        expected_correlation = np.asarray(
            [
                np.corrcoef(current_observed[:, j], current_reconstructed[:, j])[0, 1]
                if np.std(current_observed[:, j]) > 0
                and np.std(current_reconstructed[:, j]) > 0
                else np.nan
                for j in range(current_observed.shape[1])
            ]
        )
        np.testing.assert_allclose(
            current["observed_mean"], current_observed.mean(axis=0), atol=1e-12
        )
        np.testing.assert_allclose(
            current["reconstructed_mean"],
            current_reconstructed.mean(axis=0),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            current["observed_std"], current_observed.std(axis=0), atol=1e-12
        )
        np.testing.assert_allclose(
            current["reconstructed_std"],
            current_reconstructed.std(axis=0),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            current["bias"],
            current_reconstructed.mean(axis=0) - current_observed.mean(axis=0),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            current["correlation"], expected_correlation, atol=1e-12, equal_nan=True
        )
        np.testing.assert_allclose(
            current["std_ratio"],
            np.divide(
                current_reconstructed.std(axis=0),
                current_observed.std(axis=0),
                out=np.full(current_observed.shape[1], np.nan),
                where=current_observed.std(axis=0) > np.finfo(float).eps,
            ),
            atol=1e-12,
            equal_nan=True,
        )

    settings = chunked.settings
    assert settings["algorithm"] == "chunked_observed_vs_exact_center_inverse_pca"
    assert settings["scale"] == "log1p"
    assert settings["expression_source"] == "adata.layers['log1p']"
    assert settings["n_cells_effective"] == 4
    assert settings["n_features_effective"] == 3
    assert settings["n_components"] == 2
    assert settings["n_active_features"] == 3
    assert settings["n_inactive_features"] == 0
    assert settings["require_active_features"] is True
    assert settings["subset_policy"] == "caller_supplied_adata_view"
    assert settings["n_chunks_evaluated"] == 4


def test_anchor_reconstruction_qc_strict_feature_and_center_contracts() -> None:
    adata, _, _, loadings = _qc_fixture()
    wrong_order = make_pca_reconstruction_spec(
        ["B", "A", "C"],
        loadings[[1, 0, 2]],
        np.asarray([0.2, 0.1, 0.0]),
    )
    with pytest.raises(ValueError, match="feature order.*different order"):
        evaluate_pca_anchor_reconstruction(
            adata,
            latent_key="X_pca",
            time_key="stage",
            expression_layer="log1p",
            pca_reconstruction=wrong_order,
        )

    missing_center = adata.copy()
    del missing_center.var["pca_center"]
    with pytest.raises(KeyError, match="does not infer a center"):
        evaluate_pca_anchor_reconstruction(
            missing_center,
            latent_key="X_pca",
            time_key="stage",
            expression_layer="log1p",
        )

    wrong_latent_width = adata.copy()
    wrong_latent_width.obsm["X_pca"] = np.column_stack(
        [wrong_latent_width.obsm["X_pca"], np.zeros(adata.n_obs)]
    )
    with pytest.raises(ValueError, match="exact equality"):
        evaluate_pca_anchor_reconstruction(
            wrong_latent_width,
            latent_key="X_pca",
            time_key="stage",
            expression_layer="log1p",
        )

    with pytest.raises(ValueError, match="expression_space must be 'log1p'"):
        evaluate_pca_anchor_reconstruction(
            adata,
            latent_key="X_pca",
            time_key="stage",
            expression_layer="log1p",
            expression_space="count",
        )


def test_anchor_reconstruction_qc_rejects_center_only_features_by_default() -> None:
    adata, _, _, _ = _qc_fixture()
    adata.varm["PCs"][2] = 0.0

    with pytest.raises(ValueError, match="active PCA features.*original 2,000 HVGs"):
        evaluate_pca_anchor_reconstruction(
            adata,
            latent_key="X_pca",
            time_key="stage",
            expression_layer="log1p",
        )

    active_view = adata[:, ["A", "B"]].copy()
    subset_result = evaluate_pca_anchor_reconstruction(
        active_view,
        latent_key="X_pca",
        time_key="stage",
        expression_layer="log1p",
    )
    assert subset_result.settings["n_features_effective"] == 2
    assert subset_result.settings["n_active_features"] == 2
    assert subset_result.settings["n_inactive_features"] == 0

    diagnostic = evaluate_pca_anchor_reconstruction(
        adata,
        latent_key="X_pca",
        time_key="stage",
        expression_layer="log1p",
        require_active_features=False,
    )
    assert diagnostic.settings["require_active_features"] is False
    assert diagnostic.settings["n_active_features"] == 2
    assert diagnostic.settings["n_inactive_features"] == 1


def test_anchor_reconstruction_qc_explicit_matching_contract() -> None:
    adata, _, _, loadings = _qc_fixture()
    reconstruction = make_pca_reconstruction_spec(
        adata.var_names,
        loadings,
        adata.var["pca_center"].to_numpy(),
        metadata={"source": "unit-test"},
    )
    result = evaluate_pca_anchor_reconstruction(
        adata,
        latent_key="X_pca",
        time_key="stage",
        expression_layer="log1p",
        pca_reconstruction=reconstruction,
        chunk_size=2,
    )
    assert result.settings["center_source"] == "explicit_pca_reconstruction_spec"
    assert result.settings["pca_reconstruction"] == {"source": "unit-test"}
    assert result.settings["loadings_key"] is None
    assert result.settings["center_var_key"] is None
