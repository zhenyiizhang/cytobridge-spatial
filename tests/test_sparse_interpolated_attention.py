from __future__ import annotations

import anndata as ad
import numpy as np
import torch
from torch import nn

from CytoBridge.tl.downstream.attention import (
    analyze_attention_by_celltype,
    save_interpolated_attention,
)


class _Rbf(nn.Module):
    def forward(self, distance):
        return torch.stack((distance, distance.square(), torch.exp(-distance)), dim=1)


class _AttentionLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_heads = 2
        self.head_dim = 3
        self.layernorm = nn.LayerNorm(6)
        self.q_proj = nn.Linear(6, 6)
        self.k_proj = nn.Linear(6, 6)
        self.dk_proj = nn.Linear(6, 6)
        self.attn_activation = nn.SiLU()


class _InteractionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.use_spatial = True
        self.cutoff = 0.42
        self.edge_prior_mode = "learned"
        self.edge_predictor_thre = 0.5
        self.link_predictor = nn.Linear(10, 1)
        self.gene_embed = nn.Sequential(nn.Linear(3, 6), nn.Tanh(), nn.Linear(6, 6))
        self.rbf_expansion = _Rbf()
        self.distance_projection = nn.Linear(3, 6)
        self.gnn_layers = nn.ModuleList([_AttentionLayer()])


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.interaction_net = _InteractionNet()


def _dense_reference(interaction_net, data):
    """The original dense N x N implementation, retained only as a test oracle."""
    distance_matrix = torch.linalg.vector_norm(
        data[:, None, :2] - data[None, :, :2], dim=2
    )
    source, target = torch.where(distance_matrix < interaction_net.cutoff)
    probability = torch.sigmoid(
        interaction_net.link_predictor(
            torch.cat((data[source], data[target]), dim=1)
        )
    ).reshape(-1)
    keep = probability >= interaction_net.edge_predictor_thre
    source, target = source[keep], target[keep]
    distance = distance_matrix[source, target]
    keep = (source != target) & (distance > 1e-6)
    source, target, distance = source[keep], target[keep], distance[keep]

    x_embed = interaction_net.gene_embed(data[:, 2:])
    layer = interaction_net.gnn_layers[0]
    x_norm = layer.layernorm(x_embed)
    q = layer.q_proj(x_norm).reshape(-1, layer.num_heads, layer.head_dim)
    k = layer.k_proj(x_norm).reshape(-1, layer.num_heads, layer.head_dim)
    rbf = interaction_net.rbf_expansion(distance)
    edge_attr = (
        x_embed[source] + x_embed[target]
    ) * interaction_net.distance_projection(rbf)
    dk = layer.dk_proj(edge_attr).reshape(-1, layer.num_heads, layer.head_dim)
    attention = layer.attn_activation(
        (q[target] * k[source] * dk).sum(dim=-1)
    ).abs().mean(dim=1)
    return torch.stack((source, target)).numpy(), attention.detach().numpy()


def test_sparse_attention_matches_dense_edges_values_and_celltype_summary():
    torch.manual_seed(7)
    points = np.array(
        [
            [0.00, 0.00, 0.2, -0.1, 0.8],
            [0.12, 0.03, 0.4, 0.3, -0.2],
            [0.35, 0.02, -0.5, 0.7, 0.1],
            [0.61, 0.06, 0.9, -0.4, 0.2],
            [0.35, 0.34, -0.3, 0.2, 0.6],
            [0.89, 0.75, 0.1, 0.5, -0.7],
        ],
        dtype=np.float32,
    )
    labels = np.array(["A", "A", "B", "B", "C", "C"])
    adata = ad.AnnData(X=points)
    model = _Model().eval()

    expected_edges, expected_attention = _dense_reference(
        model.interaction_net, torch.from_numpy(points)
    )
    result = save_interpolated_attention(
        adata,
        time_value=0.5,
        model=model,
        save_files=False,
        save_dense_matrix=False,
        edge_batch_size=3,
    )

    np.testing.assert_array_equal(result["edge_index"], expected_edges)
    np.testing.assert_allclose(result["attn_mean"], expected_attention, rtol=1e-6, atol=1e-7)
    assert "attn_matrix" not in result

    expected_summary = analyze_attention_by_celltype(
        expected_edges,
        expected_attention,
        labels,
        winsor_quantile=None,
        distance_bins=None,
        show_plots=False,
    )
    sparse_summary = analyze_attention_by_celltype(
        result["edge_index"],
        result["attn_mean"],
        labels,
        winsor_quantile=None,
        distance_bins=None,
        show_plots=False,
    )
    for name in ("M_sum", "M_per_source", "M_row", "M_mean", "asym"):
        np.testing.assert_allclose(
            sparse_summary[name], expected_summary[name], rtol=1e-6, atol=1e-8
        )


def test_distance_quartiles_include_every_edge_once():
    edge_index = np.array([[0, 0, 1, 2], [1, 2, 2, 3]])
    attention = np.array([1.0, 2.0, 3.0, 4.0])
    labels = np.array(["A", "A", "B", "B"])
    coordinates = np.array([[0.0, 0.0], [0.1, 0.0], [0.3, 0.0], [0.8, 0.0]])

    result = analyze_attention_by_celltype(
        edge_index,
        attention,
        labels,
        spatial_coord=coordinates,
        winsor_quantile=None,
        distance_bins="quartile",
        show_plots=False,
    )

    assert len(result["M_per_source_bybin"]) == 4
    reconstructed_sum = sum(result["M_per_source_bybin"]) * np.array([[2.0], [2.0]])
    np.testing.assert_allclose(reconstructed_sum, result["M_sum"])


def test_attention_rows_are_senders_and_columns_are_receivers():
    result = analyze_attention_by_celltype(
        np.asarray([[0, 1, 2], [2, 2, 0]], dtype=np.int64),
        np.asarray([1.0, 3.0, 2.0]),
        np.asarray(["A", "A", "B"]),
        winsor_quantile=None,
        distance_bins=None,
        show_plots=False,
    )

    assert result["types"].tolist() == ["A", "B"]
    np.testing.assert_allclose(result["M_per_source"], [[0.0, 2.0], [2.0, 0.0]])
