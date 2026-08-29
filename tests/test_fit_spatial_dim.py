from __future__ import annotations

import anndata as ad
import numpy as np
import pytest
from types import SimpleNamespace

from CytoBridge.tl.train.fit import (
    _apply_runtime_overrides,
    _build_model_input,
    _edge_prior_artifact_metadata,
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


def test_all_spatial_artifact_metadata_omits_inert_predictor_defaults():
    model = SimpleNamespace(
        components=["interaction"],
        interaction_type="gnn",
        interaction_net=SimpleNamespace(
            edge_prior_mode="all_spatial",
            edge_predictor_thre=0.5,
        ),
    )

    metadata = _edge_prior_artifact_metadata(
        model,
        {"model": {"interaction_net": {"edge_prior_mode": "all_spatial"}}},
        used_edge_predictor_path=None,
        used_edge_predictor_threshold=None,
    )

    assert metadata == {
        "edge_prior_mode": "all_spatial",
        "edge_predictor_path": None,
        "edge_predictor_threshold": None,
        "uses_learned_edge_prior": False,
    }


def test_learned_artifact_metadata_requires_real_path_and_threshold():
    model = SimpleNamespace(
        components=["interaction"],
        interaction_type="gnn",
        interaction_net=SimpleNamespace(
            edge_prior_mode="learned",
            edge_predictor_thre=0.62,
        ),
    )
    config = {
        "model": {
            "interaction_net": {
                "edge_prior_mode": "learned",
                "edge_predictor_path": "/models/edge.pt",
            }
        }
    }

    metadata = _edge_prior_artifact_metadata(
        model,
        config,
        used_edge_predictor_path=None,
        used_edge_predictor_threshold=None,
    )
    assert metadata["edge_predictor_path"] == "/models/edge.pt"
    assert metadata["edge_predictor_threshold"] == pytest.approx(0.62)

    config["model"]["interaction_net"].pop("edge_predictor_path")
    with pytest.raises(ValueError, match="must record the edge predictor path"):
        _edge_prior_artifact_metadata(
            model,
            config,
            used_edge_predictor_path=None,
            used_edge_predictor_threshold=None,
        )


def test_all_spatial_fit_rejects_inert_predictor_config_and_runtime_values():
    adata = _adata_with_three_dimensional_space()
    base = {
        "model": {
            "components": ["interaction"],
            "interaction_type": "gnn",
            "interaction_net": {"edge_prior_mode": "all_spatial"},
        }
    }

    with pytest.raises(ValueError, match="edge_prior_mode='all_spatial'"):
        _apply_runtime_overrides(
            base,
            adata,
            edge_predictor_threshold=0.5,
        )

    stale = {
        "model": {
            "components": ["interaction"],
            "interaction_type": "gnn",
            "interaction_net": {
                "edge_prior_mode": "all_spatial",
                "edge_predictor_path": "/ignored/edge.pt",
            },
        }
    }
    with pytest.raises(ValueError, match="records inert predictor settings"):
        _apply_runtime_overrides(stale, adata)

    for namespace in ("fit_params", "cytobridge_fit", "training_params"):
        from_adata = _adata_with_three_dimensional_space()
        from_adata.uns[namespace] = {
            "edge_predictor_path": "/ignored/edge.pt",
            "edge_predictor_threshold": 0.5,
        }
        with pytest.raises(ValueError, match="input AnnData records inert"):
            _apply_runtime_overrides(
                {
                    "model": {
                        "components": ["interaction"],
                        "interaction_type": "gnn",
                        "interaction_net": {"edge_prior_mode": "all_spatial"},
                    }
                },
                from_adata,
            )

    graph_adata = _adata_with_three_dimensional_space()
    graph_adata.uns["interaction_graph"] = {
        "edge_predictor_path": "/ignored/edge.pt",
        "edge_predictor_threshold": 0.5,
    }
    used = _apply_runtime_overrides(
        {
            "model": {
                "components": ["interaction"],
                "interaction_type": "gnn",
                "interaction_net": {"edge_prior_mode": "all_spatial"},
            }
        },
        graph_adata,
    )
    assert "interaction_graph" not in graph_adata.uns
    assert used["removed_input_interaction_graph_metadata"] is True
