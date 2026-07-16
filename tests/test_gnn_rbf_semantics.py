from __future__ import annotations

import pytest
import torch

pytest.importorskip("torch_geometric")

from CytoBridge.tl.graph.spatial_gnn import GNNInteraction, LinkPredictorMLP


def _edge_predictor(tmp_path, *, dim: int = 52):
    path = tmp_path / "edge.pt"
    torch.save(LinkPredictorMLP(input_dim=dim * 2).state_dict(), path)
    return path


def test_gnn_rbf_is_fixed_by_default_for_legacy_parity(tmp_path) -> None:
    model = GNNInteraction(
        in_out_dim=52,
        hidden_dim=16,
        num_heads=4,
        num_layers=1,
        num_rbf=8,
        edge_predictor_path=str(_edge_predictor(tmp_path)),
    )

    parameter_names = set(dict(model.named_parameters()))
    buffer_names = set(dict(model.named_buffers()))
    assert "rbf_expansion.means" not in parameter_names
    assert "rbf_expansion.betas" not in parameter_names
    assert "rbf_expansion.means" in buffer_names
    assert "rbf_expansion.betas" in buffer_names


def test_gnn_rbf_trainable_is_explicit_opt_in(tmp_path) -> None:
    model = GNNInteraction(
        in_out_dim=52,
        hidden_dim=16,
        num_heads=4,
        num_layers=1,
        num_rbf=8,
        rbf_trainable=True,
        edge_predictor_path=str(_edge_predictor(tmp_path)),
    )

    parameter_names = set(dict(model.named_parameters()))
    assert "rbf_expansion.means" in parameter_names
    assert "rbf_expansion.betas" in parameter_names
