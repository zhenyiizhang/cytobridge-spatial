"""Model-state adaptation must preserve the training feature order."""
import numpy as np
import pandas as pd
import pytest

from CytoBridge.tl import model_state_adata


class BackedInput:
    obs = pd.DataFrame({"t": [2.0, 0.0], "type": ["a", "b"]}, index=["c1", "c2"])
    obs_names = obs.index
    obsm = {"pca": np.array([[5, 6, 7], [8, 9, 10]]),
            "xy": np.array([[1, 2], [3, 4]])}

    @property
    def X(self):
        raise AssertionError("A backed gene-expression matrix should not be read.")


def test_model_state_preserves_order_and_does_not_read_expression():
    result = model_state_adata(BackedInput(), time_key="t", latent_key="pca", spatial_key="xy")
    np.testing.assert_array_equal(result.X, [[1, 2, 5, 6, 7], [3, 4, 8, 9, 10]])
    assert result.obs_names.tolist() == ["c1", "c2"]
    assert result.obs["time_point_processed"].tolist() == [2.0, 0.0]
    assert result.obs["type"].tolist() == ["a", "b"]
    result.obsm["X_latent"][0, 0] = -1
    assert BackedInput.obsm["pca"][0, 0] == 5


def test_nonspatial_model_states():
    result = model_state_adata(BackedInput(), time_key="t", latent_key="pca", spatial_key=None)
    np.testing.assert_array_equal(result.X, BackedInput.obsm["pca"])
    assert "spatial_aligned" not in result.obsm


def test_requires_existing_latent_features():
    with pytest.raises(KeyError, match="Missing preprocessed features"):
        model_state_adata(BackedInput(), time_key="t", spatial_key="xy")


def test_rejects_nonfinite_model_state():
    data = BackedInput()
    data.obsm = {"pca": np.array([[np.nan], [1]]), "xy": BackedInput.obsm["xy"]}
    with pytest.raises(ValueError, match="finite"):
        model_state_adata(data, time_key="t", latent_key="pca", spatial_key="xy")
