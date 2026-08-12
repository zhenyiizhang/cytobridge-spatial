from __future__ import annotations

import anndata as ad
import numpy as np
import pytest

from CytoBridge.tl.train.fit import (
    _build_model_input,
    _resolve_spatial_dim_config,
)


def _adata_with_three_dimensional_space() -> ad.AnnData:
    adata = ad.AnnData(X=np.zeros((4, 2), dtype=np.float32))
    adata.obsm["X_latent"] = np.arange(8, dtype=np.float32).reshape(4, 2)
    adata.obsm["spatial_aligned"] = np.arange(12, dtype=np.float32).reshape(4, 3)
    return adata


def test_nonspatial_input_records_zero_spatial_dimensions():
    model_input, spatial_dim = _build_model_input(
        _adata_with_three_dimensional_space(),
        is_spatial=False,
    )
    config = {"model": {}}

    _resolve_spatial_dim_config(config, spatial_dim)

    assert model_input.shape == (4, 2)
    assert config["spatial_dim"] == 0
    assert config["model"]["spatial_dim"] == 0


def test_three_dimensional_input_governs_model_and_ot_config():
    model_input, spatial_dim = _build_model_input(
        _adata_with_three_dimensional_space(),
        is_spatial=True,
    )
    config = {"model": {}}

    _resolve_spatial_dim_config(config, spatial_dim)

    assert model_input.shape == (4, 5)
    assert config["spatial_dim"] == 3
    assert config["model"]["spatial_dim"] == 3


@pytest.mark.parametrize(
    "config",
    (
        {"spatial_dim": 2, "model": {}},
        {"model": {"spatial_dim": 2}},
        {"spatial_dim": 3, "model": {"spatial_dim": 2}},
    ),
)
def test_explicit_spatial_dimension_must_match_actual_input(config):
    with pytest.raises(ValueError, match="conflicts with spatial_dim=3"):
        _resolve_spatial_dim_config(config, 3)
