import os
from typing import Optional, Tuple

import torch
import torch.nn as nn

from CytoBridge.tl.core.interaction import ExpNormalSmearing

try:
    from torch_geometric.nn import MessagePassing
except ImportError as exc:  # pragma: no cover - handled at runtime
    MessagePassing = None
    _TORCH_GEOMETRIC_ERROR = exc


class GNNInteraction(nn.Module):
    requires_time = True

    def __init__(
        self,
        in_out_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        activation: str = "Tanh",
        num_rbf: int = 8,
        cutoff: float = 0.2,
        use_spatial: bool = True,
        edge_predictor_path: Optional[str] = None,
        edge_predictor_thre: float = 0.5,
        edge_predictor_root: Optional[str] = None,
    ):
        super().__init__()
        if MessagePassing is None:
            raise ImportError(
                "torch_geometric is required for GNNInteraction. "
                "Install torch-geometric to use the spatial interaction model."
            ) from _TORCH_GEOMETRIC_ERROR

        if activation.lower() == "tanh":
            self.activation = nn.Tanh()
        elif activation.lower() == "relu":
            self.activation = nn.ReLU()
        elif activation.lower() == "gelu":
            self.activation = nn.GELU()
        elif activation.lower() == "leakyrelu":
            self.activation = nn.LeakyReLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self.in_out_dim = in_out_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.use_spatial = use_spatial
        self.cutoff = cutoff
        self.edge_predictor_thre = edge_predictor_thre

        self.rbf_expansion = ExpNormalSmearing(cutoff=cutoff, num_rbf=num_rbf, trainable=True)

        if edge_predictor_path is None:
            raise ValueError("edge_predictor_path must be provided for GNNInteraction.")
        predictor_path = os.path.expanduser(edge_predictor_path)
        if edge_predictor_root is not None and not os.path.isabs(predictor_path):
            predictor_path = os.path.join(edge_predictor_root, predictor_path)
        if not os.path.isabs(predictor_path):
            predictor_path = os.path.abspath(predictor_path)

        self.link_predictor = LinkPredictorMLP(input_dim=in_out_dim * 2)
        self.link_predictor.load_state_dict(torch.load(predictor_path, map_location=torch.device("cpu")))
        for param in self.link_predictor.parameters():
            param.requires_grad = False
        self.link_predictor.eval()

        self.gene_embed = nn.Sequential(
            nn.Linear(in_out_dim - 2, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.distance_projection = nn.Linear(num_rbf, hidden_dim)
        self.gnn_layers = nn.ModuleList(
            [GraphAttentionLayer(hidden_dim, num_heads, activation=activation) for _ in range(num_layers)]
        )

        self.gene_readout = nn.Sequential(
            nn.Linear(hidden_dim, in_out_dim - 2),
        )

    def forward(self, x: torch.Tensor, lnw: torch.Tensor, t: torch.Tensor, return_attn: bool = False) -> torch.Tensor:
        if self.use_spatial:
            x_expanded = x[:, :2].unsqueeze(1)
            x_expanded_t = x[:, :2].unsqueeze(0)
            pairwise_distances = torch.norm(x_expanded - x_expanded_t, dim=2)
            mask = (pairwise_distances < self.cutoff).float()
        else:
            x_expanded = x.unsqueeze(1)
            x_expanded_t = x.unsqueeze(0)
            pairwise_distances = torch.norm(x_expanded - x_expanded_t, dim=2)
            mask = (pairwise_distances < self.cutoff).float()

        rows, cols = torch.where(mask)
        features_i = x[rows]
        features_j = x[cols]
        pair_features = torch.cat([features_i, features_j], dim=1)
        pred_probs = self.link_predictor(pair_features)
        pred_probs = torch.sigmoid(pred_probs).reshape(-1)
        connected = pred_probs >= self.edge_predictor_thre

        edge_index = torch.stack([rows[connected], cols[connected]], dim=0)
        indices = edge_index[0] != edge_index[1]
        edge_index = edge_index[:, indices]
        r_ij = pairwise_distances[edge_index[0], edge_index[1]]
        edge_index = edge_index[:, r_ij > 1e-6]
        r_ij = r_ij[r_ij > 1e-6]
        # Cache for downstream attention/communication analysis (does not affect forward output).
        self.edge_index = edge_index.detach()

        x_embed = self.gene_embed(x[:, 2:])
        vec = torch.zeros(x_embed.size(0), 2, x_embed.size(1), device=x.device)

        vec_ij = (x[edge_index[0], :2] - x[edge_index[1], :2]) / r_ij.unsqueeze(1)
        rbf_ij = self.rbf_expansion(r_ij)
        edge_attr = (x_embed[edge_index[0]] + x_embed[edge_index[1]]) * self.distance_projection(rbf_ij)

        for layer in self.gnn_layers:
            x_embed, vec = layer(x_embed, vec, lnw, edge_index, edge_attr, vec_ij, return_attn=return_attn)

        x_spatial = vec.mean(dim=-1)
        x_gene = self.gene_readout(x_embed)
        x_out = torch.cat([x_spatial, x_gene], dim=1)

        return x_out


class GraphAttentionLayer(MessagePassing):
    def __init__(self, hidden_dim: int, num_heads: int, activation: str = "Tanh"):
        super().__init__(node_dim=0)
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.hidden_dim // self.num_heads
        if activation.lower() == "tanh":
            self.activation = nn.Tanh()
        elif activation.lower() == "relu":
            self.activation = nn.ReLU()
        elif activation.lower() == "gelu":
            self.activation = nn.GELU()
        elif activation.lower() == "leakyrelu":
            self.activation = nn.LeakyReLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        self.attn_activation = nn.SiLU()
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dk_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dv_proj = nn.Linear(hidden_dim, hidden_dim)
        self.s_proj = nn.Linear(hidden_dim, hidden_dim * 2)
        self.layernorm = nn.LayerNorm(hidden_dim)
        self.res_proj = nn.Linear(hidden_dim, hidden_dim)
        self.vec_proj = nn.Linear(hidden_dim, hidden_dim * 3, bias=False)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim * 3)
        self.out_transform = nn.Sequential(
            nn.Linear(hidden_dim // num_heads, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim, hidden_dim // num_heads),
        )

    def forward(self, x, vec, lnw, edge_index, edge_attr, edge_vec, return_attn: bool = False):
        x_res = self.res_proj(x).reshape(-1, self.num_heads, self.head_dim)
        x = self.layernorm(x)
        q = self.q_proj(x).reshape(-1, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(-1, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(-1, self.num_heads, self.head_dim)
        dk = self.dk_proj(edge_attr).reshape(-1, self.num_heads, self.head_dim)
        dv = self.dv_proj(edge_attr).reshape(-1, self.num_heads, self.head_dim)
        w = torch.exp(lnw) * lnw.shape[0]

        self.return_attn = return_attn
        x, vec_out = self.propagate(
            edge_index,
            q=q,
            k=k,
            v=v,
            w=w,
            x_orig=x_res,
            dk=dk,
            dv=dv,
            vec=vec,
            r_ij=edge_attr,
            d_ij=edge_vec,
            size=None,
        )
        return x, vec_out

    def message(self, q_i, k_j, v_j, vec_j, w_j, x_orig_i, dk, dv, r_ij, d_ij):
        attn = (q_i * k_j * dk).sum(dim=-1)
        attn = self.attn_activation(attn)
        if self.return_attn:
            self.attn = attn
        v_j = v_j * dv + x_orig_i
        v_j = self.out_transform(v_j)
        v_j = (v_j * attn.unsqueeze(2)).view(-1, self.hidden_dim)

        s1, s2 = torch.split(self.activation(self.s_proj(v_j)), self.hidden_dim, dim=1)
        vec_j = vec_j * s1.unsqueeze(1) + s2.unsqueeze(1) * d_ij.unsqueeze(2)

        v_j = v_j * w_j
        vec_j = vec_j * w_j.unsqueeze(1)
        return v_j, vec_j, w_j

    def manual_scatter_mean(self, src: torch.Tensor, weight: torch.Tensor, index: torch.Tensor, dim: int, output_size: int) -> torch.Tensor:
        output_shape = list(src.shape)
        output_shape[dim] = output_size

        index_expand_shape = [1] * src.dim()
        index_expand_shape[dim] = -1
        index_expanded = index.view(index_expand_shape).expand_as(src)

        output_sum = torch.zeros(output_shape, dtype=src.dtype, device=src.device)
        output_sum.scatter_add_(dim, index_expanded, src)

        count = torch.zeros(output_shape, dtype=src.dtype, device=src.device)
        weight_expand_shape = [-1] + [1] * (src.dim() - 1)
        weight_expanded = weight.view(weight_expand_shape).expand_as(src)
        count.scatter_add_(dim, index_expanded, weight_expanded)

        count_safe = count.clone()
        count_safe[count_safe == 0] = 1
        return output_sum / count_safe

    def aggregate(self, features: Tuple[torch.Tensor, torch.Tensor], index: torch.Tensor, ptr, dim_size: Optional[int]):
        x, vec, w = features
        aggregation_dim = self.node_dim
        aggregated_x = self.manual_scatter_mean(x, w, index, dim=aggregation_dim, output_size=dim_size)
        aggregated_vec = self.manual_scatter_mean(vec, w, index, dim=aggregation_dim, output_size=dim_size)
        return aggregated_x, aggregated_vec

    def update(self, inputs: Tuple[torch.Tensor, torch.Tensor]):
        return inputs


class LinkPredictorMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        return self.network(x)
