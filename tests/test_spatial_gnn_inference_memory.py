from __future__ import annotations

import torch
from torch import nn

import CytoBridge.tl.graph.spatial_gnn as spatial_gnn


def _layer_without_torch_geometric() -> spatial_gnn.GraphAttentionLayer:
    layer = object.__new__(spatial_gnn.GraphAttentionLayer)
    nn.Module.__init__(layer)
    layer.hidden_dim = 8
    layer.num_heads = 2
    layer.head_dim = 4
    layer.activation = nn.LeakyReLU()
    layer.attn_activation = nn.SiLU()
    layer.q_proj = nn.Linear(8, 8)
    layer.k_proj = nn.Linear(8, 8)
    layer.v_proj = nn.Linear(8, 8)
    layer.dk_proj = nn.Linear(8, 8)
    layer.dv_proj = nn.Linear(8, 8)
    layer.s_proj = nn.Linear(8, 16)
    layer.layernorm = nn.LayerNorm(8)
    layer.res_proj = nn.Linear(8, 8)
    layer.vec_proj = nn.Linear(8, 24, bias=False)
    layer.o_proj = nn.Linear(8, 24)
    layer.out_transform = nn.Sequential(
        nn.Linear(4, 8),
        layer.activation,
        nn.Linear(8, 4),
    )
    return layer


def test_inference_attention_chunks_edges_without_changing_messages():
    # The source module remains importable without torch-geometric so package
    # contract tests can exercise the memory-bounded message implementation.
    # Its fallback base is nn.Module; bypass only the constructor dependency
    # guard and never call the unavailable PyG propagate method.
    torch.manual_seed(17)
    layer = _layer_without_torch_geometric()

    x = torch.randn(5, 8)
    vec = torch.randn(5, 2, 8)
    lnw = torch.log(torch.tensor([[0.1], [0.2], [0.3], [0.15], [0.25]]))
    edge_index = torch.tensor(
        [
            [0, 1, 2, 3, 4, 0, 2, 4, 1, 3, 0, 4],
            [1, 2, 3, 4, 0, 2, 4, 1, 3, 0, 4, 2],
        ],
        dtype=torch.long,
    )
    edge_attr = torch.randn(edge_index.shape[1], 8)
    edge_vec = torch.randn(edge_index.shape[1], 2)

    x_res = layer.res_proj(x).reshape(-1, layer.num_heads, layer.head_dim)
    normalized = layer.layernorm(x)
    q = layer.q_proj(normalized).reshape(-1, layer.num_heads, layer.head_dim)
    k = layer.k_proj(normalized).reshape(-1, layer.num_heads, layer.head_dim)
    v = layer.v_proj(normalized).reshape(-1, layer.num_heads, layer.head_dim)
    dk = layer.dk_proj(edge_attr).reshape(-1, layer.num_heads, layer.head_dim)
    dv = layer.dv_proj(edge_attr).reshape(-1, layer.num_heads, layer.head_dim)
    w = torch.exp(lnw) * lnw.shape[0]
    source, target = edge_index
    layer.return_attn = True
    message_x, message_vec, message_weight = layer.message(
        q[target],
        k[source],
        v[source],
        vec[source],
        w[source],
        x_res[target],
        dk,
        dv,
        edge_attr,
        edge_vec,
    )
    expected_x = layer.manual_scatter_mean(
        message_x,
        message_weight,
        target,
        dim=0,
        output_size=x.shape[0],
    )
    expected_vec = layer.manual_scatter_mean(
        message_vec,
        message_weight,
        target,
        dim=0,
        output_size=x.shape[0],
    )
    expected_attention = layer.attn.detach().clone()

    observed_batches: list[int] = []

    def record_batch(_module, inputs, _output):
        observed_batches.append(int(inputs[0].shape[0]))

    hook = layer.out_transform[0].register_forward_hook(record_batch)
    layer.inference_edge_chunk_size = 3
    try:
        with torch.no_grad():
            actual_x, actual_vec = layer(
                x,
                vec,
                lnw,
                edge_index,
                edge_attr,
                edge_vec,
                return_attn=True,
            )
    finally:
        hook.remove()

    assert len(observed_batches) == 4
    assert max(observed_batches) <= layer.inference_edge_chunk_size
    torch.testing.assert_close(actual_x, expected_x.detach(), atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(actual_vec, expected_vec.detach(), atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(layer.attn, expected_attention, atol=2e-6, rtol=2e-6)


def test_inference_chunk_size_is_positive():
    assert spatial_gnn.GraphAttentionLayer.inference_edge_chunk_size > 0
