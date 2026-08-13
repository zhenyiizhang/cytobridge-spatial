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


def test_all_spatial_ablation_removes_only_pretrained_edge_gate() -> None:
    model = GNNInteraction(
        in_out_dim=4,
        hidden_dim=8,
        num_heads=2,
        num_layers=1,
        num_rbf=4,
        cutoff=0.5,
        edge_prior_mode="all_spatial",
        edge_predictor_path=None,
    )
    points = torch.tensor(
        [
            [0.0, 0.0, 0.2, 0.4],
            [0.1, 0.0, 0.3, 0.5],
            [1.0, 0.0, 0.7, 0.9],
        ],
        dtype=torch.float32,
    )
    log_weights = torch.log(torch.full((3, 1), 1.0 / 3.0))
    output = model(points, log_weights, torch.zeros(3, 1))

    assert output.shape == points.shape
    assert not hasattr(model, "link_predictor")
    assert {tuple(edge) for edge in model.edge_index.t().detach().cpu().tolist()} == {
        (0, 1),
        (1, 0),
    }

    output.square().mean().backward()
    assert any(
        parameter.grad is not None
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    reloaded = GNNInteraction(
        in_out_dim=4,
        hidden_dim=8,
        num_heads=2,
        num_layers=1,
        num_rbf=4,
        cutoff=0.5,
        edge_prior_mode="all_spatial",
        edge_predictor_path=None,
    )
    reloaded.load_state_dict(model.state_dict(), strict=True)
    assert not hasattr(reloaded, "link_predictor")


def test_edge_prior_mode_is_validated_before_loading_predictor() -> None:
    with pytest.raises(ValueError, match="edge_prior_mode"):
        GNNInteraction(
            in_out_dim=4,
            hidden_dim=8,
            num_heads=2,
            num_layers=1,
            edge_prior_mode="not-a-mode",
        )
