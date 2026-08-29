from __future__ import annotations

import pytest
import torch

pytest.importorskip("torch_geometric")

from CytoBridge.tl.graph.spatial_gnn import (
    GNNInteraction,
    GraphAttentionLayer,
    LinkPredictorMLP,
)


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


def _attention_inputs(*, n_nodes: int = 7, n_edges: int = 23):
    generator = torch.Generator().manual_seed(17)
    x = torch.randn(n_nodes, 8, generator=generator)
    vec = torch.randn(n_nodes, 2, 8, generator=generator)
    lnw = torch.log(torch.full((n_nodes, 1), 1.0 / n_nodes))
    source = torch.arange(n_edges) % n_nodes
    target = (torch.arange(n_edges) * 3 + 1) % n_nodes
    edge_index = torch.stack([source, target], dim=0)
    edge_attr = torch.randn(n_edges, 8, generator=generator)
    edge_vec = torch.randn(n_edges, 2, generator=generator)
    return x, vec, lnw, edge_index, edge_attr, edge_vec


def test_no_grad_edge_chunking_matches_historical_propagation() -> None:
    layer = GraphAttentionLayer(hidden_dim=8, num_heads=2)
    inputs = _attention_inputs()

    with torch.no_grad():
        layer.inference_edge_chunk_size = 10_000
        expected_x, expected_vec = layer(*inputs)
        layer.inference_edge_chunk_size = 5
        actual_x, actual_vec = layer(*inputs)

    torch.testing.assert_close(actual_x, expected_x, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(actual_vec, expected_vec, rtol=1e-5, atol=1e-6)


def test_no_grad_edge_chunking_is_repeatable_with_repeated_targets() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    layer = GraphAttentionLayer(hidden_dim=8, num_heads=2).to(device)
    inputs = tuple(
        value.to(device) for value in _attention_inputs(n_nodes=7, n_edges=5_003)
    )
    layer.inference_edge_chunk_size = 127

    with torch.no_grad():
        first_x, first_vec = layer(*inputs)
        second_x, second_vec = layer(*inputs)

    torch.testing.assert_close(first_x, second_x, rtol=0.0, atol=0.0)
    torch.testing.assert_close(first_vec, second_vec, rtol=0.0, atol=0.0)


def test_grad_enabled_attention_keeps_historical_propagation(monkeypatch) -> None:
    layer = GraphAttentionLayer(hidden_dim=8, num_heads=2)
    layer.inference_edge_chunk_size = 1
    inputs = list(_attention_inputs())
    inputs[0].requires_grad_(True)

    def fail_chunked(**_kwargs):
        raise AssertionError("Training must not use inference edge chunks")

    monkeypatch.setattr(layer, "_propagate_inference_chunks", fail_chunked)
    output_x, output_vec = layer(*inputs)
    (output_x.square().mean() + output_vec.square().mean()).backward()

    assert inputs[0].grad is not None
