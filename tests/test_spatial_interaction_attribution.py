from __future__ import annotations

import numpy as np
import pytest
import importlib.util
from pathlib import Path
import sys

torch = pytest.importorskip("torch")
from torch import nn

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "CytoBridge"
    / "tl"
    / "downstream"
    / "spatial_interaction_attribution.py"
)
SPEC = importlib.util.spec_from_file_location(
    "spatial_interaction_attribution_under_test", MODULE_PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
analyze_spatial_gnn_by_celltype = MODULE.analyze_spatial_gnn_by_celltype
decompose_spatial_gnn_group = MODULE.decompose_spatial_gnn_group
make_interaction_groups = MODULE.make_interaction_groups


class _FakeLayer(nn.Module):
    def __init__(self, hidden_dim=8, num_heads=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.activation = nn.Tanh()
        self.attn_activation = nn.SiLU()
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dk_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dv_proj = nn.Linear(hidden_dim, hidden_dim)
        self.s_proj = nn.Linear(hidden_dim, hidden_dim * 2)
        self.layernorm = nn.LayerNorm(hidden_dim)
        self.res_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_transform = nn.Sequential(
            nn.Linear(self.head_dim, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim, self.head_dim),
        )
        self.attn = None


class _FakePredictor(nn.Module):
    def forward(self, pair):
        # Every non-self spatial candidate is selected.
        return torch.full((pair.shape[0], 1), 8.0, device=pair.device, dtype=pair.dtype)


class _FakeRBF(nn.Module):
    def forward(self, distance):
        return torch.stack((distance, distance.square()), dim=1)


class _FakeSpatialNet(nn.Module):
    def __init__(self, num_layers=1):
        super().__init__()
        hidden = 8
        self.use_spatial = True
        self.cutoff = 10.0
        self.edge_predictor_thre = 0.5
        self.link_predictor = _FakePredictor()
        self.gene_embed = nn.Sequential(nn.Linear(3, hidden), nn.Tanh(), nn.Linear(hidden, hidden))
        self.distance_projection = nn.Linear(2, hidden)
        self.rbf_expansion = _FakeRBF()
        self.gnn_layers = nn.ModuleList([_FakeLayer(hidden, 2) for _ in range(num_layers)])
        self.gene_readout = nn.Sequential(nn.Linear(hidden, 3))
        self.edge_index = None

    def forward(self, x, lnw, t, return_attn=False):
        if len(self.gnn_layers) != 1:
            raise NotImplementedError
        layer = self.gnn_layers[0]
        n = x.shape[0]
        rows, cols = torch.where(torch.ones((n, n), dtype=torch.bool, device=x.device))
        keep = rows != cols
        source, target = rows[keep], cols[keep]
        self.edge_index = torch.stack((source, target))
        distance = torch.linalg.vector_norm(x[source, :2] - x[target, :2], dim=1)
        direction = (x[source, :2] - x[target, :2]) / distance[:, None]
        embed = self.gene_embed(x[:, 2:])
        edge_attr = (embed[source] + embed[target]) * self.distance_projection(
            self.rbf_expansion(distance)
        )
        residual = layer.res_proj(embed).reshape(-1, layer.num_heads, layer.head_dim)
        normalized = layer.layernorm(embed)
        q = layer.q_proj(normalized).reshape(-1, layer.num_heads, layer.head_dim)
        k = layer.k_proj(normalized).reshape(-1, layer.num_heads, layer.head_dim)
        v = layer.v_proj(normalized).reshape(-1, layer.num_heads, layer.head_dim)
        dk = layer.dk_proj(edge_attr).reshape(-1, layer.num_heads, layer.head_dim)
        dv = layer.dv_proj(edge_attr).reshape(-1, layer.num_heads, layer.head_dim)
        attention = layer.attn_activation((q[target] * k[source] * dk).sum(-1))
        layer.attn = attention
        message = layer.out_transform(v[source] * dv + residual[target])
        message = (message * attention[..., None]).reshape(-1, layer.hidden_dim)
        _, scale = torch.split(layer.activation(layer.s_proj(message)), layer.hidden_dim, dim=1)
        vector = scale[:, None, :] * direction[:, :, None]
        mass = torch.exp(lnw) * n
        edge_mass = mass[source]
        denominator = torch.zeros((n, 1), dtype=x.dtype, device=x.device)
        denominator.index_add_(0, target, edge_mass)
        fraction = edge_mass / denominator[target]
        scalar_aggregate = torch.zeros((n, layer.hidden_dim), dtype=x.dtype, device=x.device)
        scalar_aggregate.index_add_(0, target, message * fraction)
        vector_aggregate = torch.zeros((n, 2, layer.hidden_dim), dtype=x.dtype, device=x.device)
        vector_aggregate.index_add_(0, target, vector * fraction[:, None, :])
        return torch.cat((vector_aggregate.mean(-1), self.gene_readout(scalar_aggregate)), dim=1)


def _fixture():
    torch.manual_seed(17)
    net = _FakeSpatialNet()
    x = torch.tensor(
        [
            [0.0, 0.0, 0.2, -0.3, 0.5],
            [1.0, 0.2, -0.1, 0.6, 0.4],
            [0.3, 1.1, 0.7, 0.2, -0.4],
            [1.2, 1.4, -0.5, 0.1, 0.8],
        ],
        dtype=torch.float32,
    )
    lnw = torch.log(torch.tensor([[0.1], [0.2], [0.3], [0.4]], dtype=torch.float32))
    return net, x, lnw


def test_exact_edges_reconstruct_official_spatial_output():
    net, x, lnw = _fixture()
    result = decompose_spatial_gnn_group(net, x, lnw, torch.tensor(0.5))
    assert result.edge_index.shape == (2, 12)
    assert result.edge_output.shape == (12, 5)
    torch.testing.assert_close(result.reconstructed, result.output, atol=2e-6, rtol=2e-6)
    assert result.max_abs_residual < 2e-6
    # Readout bias is a receiver baseline, never an edge/sender contribution.
    torch.testing.assert_close(result.baseline[:, :2], torch.zeros_like(result.baseline[:, :2]))
    expected_bias = net.gene_readout[0].bias.expand(4, -1)
    torch.testing.assert_close(result.baseline[:, 2:], expected_bias)


def test_grouped_D_AB_includes_all_receivers_and_preserves_edges():
    net, x, lnw = _fixture()
    labels = np.array(["A", "A", "B", "B"])
    grouped = analyze_spatial_gnn_by_celltype(
        net,
        x,
        lnw,
        torch.tensor(0.5),
        labels,
        group_size=4,
        grouping_seed=101,
    )
    assert grouped.edge_table.shape[0] == 12
    assert grouped.edge_output.shape == (12, 5)
    assert grouped.attention_signed.shape == (12, 2)
    table = grouped.type_pair_table.set_index(["sender_type", "receiver_type"])
    assert table.loc[("A", "B"), "edge_count"] == 4
    assert table.loc[("A", "B"), "D_AB_joint"] > 0
    assert grouped.reconstruction_table["max_abs_residual"].max() < 2e-6


def test_group_sizes_match_released_remainder_contract():
    groups = make_interaction_groups(5271, 1024, random_state=7)
    assert [len(group) for group in groups] == [1024, 1024, 1024, 1024, 1175]
    assert np.unique(np.concatenate(groups)).size == 5271


def test_multilayer_model_is_rejected_fail_closed():
    net = _FakeSpatialNet(num_layers=2)
    x = torch.randn(4, 5)
    lnw = torch.full((4, 1), -np.log(4), dtype=torch.float32)
    with pytest.raises(ValueError, match="exactly one GNN layer"):
        decompose_spatial_gnn_group(net, x, lnw, torch.tensor(0.5))
