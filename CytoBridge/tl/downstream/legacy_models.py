from __future__ import annotations

import math
import os
from typing import Optional, Tuple

import torch
import torch.nn as nn

try:
    from torch_geometric.nn import MessagePassing
except ImportError as exc:  # pragma: no cover - handled at runtime
    MessagePassing = None
    _TORCH_GEOMETRIC_ERROR = exc

_MessagePassingBase = MessagePassing if MessagePassing is not None else nn.Module


def _legacy_activation(name: str) -> nn.Module:
    key = name.lower()
    if key == "tanh":
        return nn.Tanh()
    if key == "relu":
        return nn.ReLU()
    if key == "elu":
        return nn.ELU()
    if key == "gelu":
        return nn.GELU()
    if key == "leakyrelu":
        return nn.LeakyReLU()
    raise ValueError(f"Unsupported activation: {name}")


class LegacyVelocityNet(nn.Module):
    def __init__(self, in_out_dim, hidden_dim, n_hiddens, activation="Tanh", use_spatial=False):
        super().__init__()
        layers = [in_out_dim + 1]
        for _ in range(n_hiddens):
            layers.append(hidden_dim)
        layers.append(in_out_dim)

        self.activation = _legacy_activation(activation)
        self.use_spatial = use_spatial

        if use_spatial:
            self.spatial_net = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(layers[i], layers[i + 1]),
                        self.activation,
                    )
                    for i in range(len(layers) - 2)
                ]
            )
            self.spatial_out = nn.Linear(layers[-2], 2)
            self.gene_net = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(layers[i], layers[i + 1]),
                        self.activation,
                    )
                    for i in range(len(layers) - 2)
                ]
            )
            self.gene_out = nn.Linear(layers[-2], in_out_dim - 2)
        else:
            self.net = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(layers[i], layers[i + 1]),
                        self.activation,
                    )
                    for i in range(len(layers) - 2)
                ]
            )
            self.out = nn.Linear(layers[-2], layers[-1])

    def forward(self, t, x):
        num = x.shape[0]
        t = t.expand(num, 1)
        state = torch.cat((t, x), dim=1)
        if self.use_spatial:
            ii = 0
            for layer in self.spatial_net:
                x = layer(state) if ii == 0 else layer(x)
                ii += 1
            spatial_x = self.spatial_out(x)

            ii = 0
            for layer in self.gene_net:
                x = layer(state) if ii == 0 else layer(x)
                ii += 1
            gene_x = self.gene_out(x)
            x = torch.cat([spatial_x, gene_x], dim=1)
        else:
            ii = 0
            for layer in self.net:
                x = layer(state) if ii == 0 else layer(x)
                ii += 1
            x = self.out(x)
        return x


class LegacyGrowthNet(nn.Module):
    def __init__(self, in_out_dim, hidden_dim, activation="Tanh"):
        super().__init__()
        self.activation = _legacy_activation(activation)
        self.net = nn.Sequential(
            nn.Linear(in_out_dim + 1, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, t, x):
        num = x.shape[0]
        t = t.expand(num, 1)
        state = torch.cat((t, x), dim=1)
        return self.net(state)


class LegacyScoreNet(nn.Module):
    def __init__(self, in_out_dim, hidden_dim, activation="Tanh"):
        super().__init__()
        self.activation = _legacy_activation(activation)
        self.net = nn.Sequential(
            nn.Linear(in_out_dim + 1, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, t, x):
        num = x.shape[0]
        t = t.expand(num, 1)
        state = torch.cat((t, x), dim=1)
        return self.net(state)


class LegacyScoreNet2(nn.Module):
    def __init__(self, in_out_dim, hidden_dim, activation="Tanh"):
        super().__init__()
        self.activation = _legacy_activation(activation)
        self.net = nn.Sequential(
            nn.Linear(in_out_dim + 1, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, t, x):
        state = torch.cat((t, x), dim=1)
        return self.net(state)

    def compute_gradient(self, t, x):
        x = x.requires_grad_(True)
        output = self.forward(t, x)
        gradient = torch.autograd.grad(
            outputs=output,
            inputs=x,
            grad_outputs=torch.ones_like(output),
            create_graph=True,
        )[0]
        return gradient


class LegacyInDeDiffusionNet(nn.Module):
    def __init__(self, in_out_dim, hidden_dim, activation="Tanh"):
        super().__init__()
        self.activation = _legacy_activation(activation)
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, t, x):
        num = x.shape[0]
        t = t.expand(num, 1)
        return self.net(t)


class LegacyLinkPredictorMLP(nn.Module):
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


class LegacyCosineCutoff(nn.Module):
    def __init__(self, cutoff):
        super().__init__()
        self.cutoff = cutoff

    def forward(self, distances):
        cutoffs = 0.5 * (torch.cos(distances * math.pi / self.cutoff) + 1.0)
        cutoffs = cutoffs * (distances < self.cutoff).float()
        return cutoffs


class LegacyExpNormalSmearing(nn.Module):
    def __init__(self, cutoff=5.0, cutoff_sr=0.0, num_rbf=50, trainable=False):
        super().__init__()
        self.cutoff = cutoff
        self.cutoff_sr = cutoff_sr
        self.num_rbf = num_rbf
        self.trainable = trainable
        self.alpha = 1.0
        self.cutoff_fn = LegacyCosineCutoff(cutoff)
        means, betas = self._initial_params()
        if trainable:
            self.register_parameter("means", nn.Parameter(means))
            self.register_parameter("betas", nn.Parameter(betas))
        else:
            self.register_buffer("means", means)
            self.register_buffer("betas", betas)

    def _initial_params(self):
        if self.cutoff == 0:
            return torch.zeros(self.num_rbf), torch.tensor(0.0)
        start_value = torch.exp(torch.scalar_tensor(-self.cutoff))
        end_value = torch.exp(torch.scalar_tensor(-self.cutoff_sr))
        means = torch.linspace(start_value, end_value, self.num_rbf)
        betas = torch.tensor([(2 / self.num_rbf * (end_value - start_value)) ** -2] * self.num_rbf)
        return means, betas

    def reset_parameters(self):
        means, betas = self._initial_params()
        self.means.data.copy_(means)
        self.betas.data.copy_(betas)

    def forward(self, dist):
        dist = dist.unsqueeze(-1)
        return self.cutoff_fn(dist) * torch.exp(
            -self.betas * (torch.exp(self.alpha * (-dist)) - self.means) ** 2
        )


class LegacyGraphAttentionLayer(_MessagePassingBase):
    def __init__(self, hidden_dim, num_heads, activation="Tanh"):
        if MessagePassing is None:
            raise ImportError(
                "torch_geometric is required for legacy GNN interaction loading. "
                "Install it with: pip install 'CytoBridge[graph]'"
            ) from _TORCH_GEOMETRIC_ERROR
        super().__init__(node_dim=0)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.activation = _legacy_activation(activation)
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

    def forward(self, x, vec, lnw, edge_index, edge_attr, edge_vec, return_attn=False):
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

    def aggregate(
        self,
        features: Tuple[torch.Tensor, torch.Tensor],
        index: torch.Tensor,
        ptr: Optional[torch.Tensor],
        dim_size: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, vec, w = features
        aggregation_dim = self.node_dim
        aggregated_x = self.manual_scatter_mean(x, w, index, dim=aggregation_dim, output_size=dim_size)
        aggregated_vec = self.manual_scatter_mean(vec, w, index, dim=aggregation_dim, output_size=dim_size)
        return aggregated_x, aggregated_vec

    def update(self, inputs: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        return inputs


class LegacyGNNInteraction(nn.Module):
    requires_time = True

    def __init__(
        self,
        in_out_dim,
        hidden_dim,
        num_heads,
        num_layers,
        activation="Tanh",
        num_rbf=8,
        cutoff=0.2,
        use_spatial=True,
        edge_predictor_path=None,
        edge_predictor_thre=0.5,
        edge_predictor_root: Optional[str] = None,
    ):
        super().__init__()
        if MessagePassing is None:
            raise ImportError(
                "torch_geometric is required for legacy GNN interaction loading. "
                "Install it with: pip install 'CytoBridge[graph]'"
            ) from _TORCH_GEOMETRIC_ERROR

        self.activation = _legacy_activation(activation)
        self.in_out_dim = in_out_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.use_spatial = use_spatial
        self.cutoff = cutoff
        self.rbf_expansion = LegacyExpNormalSmearing(cutoff=cutoff, num_rbf=num_rbf)

        if edge_predictor_path is None:
            raise ValueError("edge_predictor_path is required for legacy GNN interaction loading.")
        predictor_path = os.path.expanduser(str(edge_predictor_path))
        if not os.path.isabs(predictor_path):
            if edge_predictor_root is None:
                raise ValueError("edge_predictor_root is required for relative edge predictor paths.")
            predictor_path = os.path.join(edge_predictor_root, os.path.basename(predictor_path))
        self.link_predictor = LegacyLinkPredictorMLP(input_dim=in_out_dim * 2)
        self.link_predictor.load_state_dict(
            torch.load(predictor_path, map_location=torch.device("cpu"))
        )
        for param in self.link_predictor.parameters():
            param.requires_grad = False
        self.link_predictor.eval()

        self.gene_embed = nn.Sequential(
            nn.Linear(in_out_dim - 2, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.edge_predictor_thre = edge_predictor_thre
        self.distance_projection = nn.Linear(num_rbf, hidden_dim)
        self.gnn_layers = nn.ModuleList(
            [LegacyGraphAttentionLayer(hidden_dim, num_heads, activation=activation) for _ in range(num_layers)]
        )
        self.gene_readout = nn.Sequential(nn.Linear(hidden_dim, in_out_dim - 2))

    def forward(self, x, lnw, t, return_attn=False):
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
        self.edge_index = edge_index
        del mask

        num = x.shape[0]
        t = t.expand(num, 1)
        x_embed = self.gene_embed(x[:, 2:])
        vec = torch.zeros(x_embed.size(0), 2, x_embed.size(1), device=x.device)
        vec_ij = (x[edge_index[0], :2] - x[edge_index[1], :2]) / r_ij.unsqueeze(1)
        rbf_ij = self.rbf_expansion(r_ij)
        edge_attr = (x_embed[edge_index[0]] + x_embed[edge_index[1]]) * self.distance_projection(rbf_ij)

        for layer in self.gnn_layers:
            x_embed, vec = layer(x_embed, vec, lnw, edge_index, edge_attr, vec_ij, return_attn=return_attn)

        x_spatial = vec.mean(dim=-1)
        x_gene = self.gene_readout(x_embed)
        return torch.cat([x_spatial, x_gene], dim=1)


class LegacyFNetInteraction(nn.Module):
    def __init__(
        self,
        in_out_dim,
        hidden_dim,
        n_hiddens,
        activation,
        num_rbf=8,
        thre=100,
        dim_reduce=False,
        use_spatial=False,
        num_heads=8,
        num_layers=1,
        edge_predictor_path=None,
        edge_predictor_thre=0.5,
        edge_predictor_root: Optional[str] = None,
    ):
        super().__init__()
        self.in_out_dim = in_out_dim
        self.hidden_dim = hidden_dim
        self.v_net = LegacyVelocityNet(in_out_dim, hidden_dim, n_hiddens, activation, use_spatial=use_spatial)
        self.g_net = LegacyGrowthNet(in_out_dim, hidden_dim, activation)
        self.s_net = LegacyScoreNet(in_out_dim, hidden_dim, activation)
        self.d_net = LegacyInDeDiffusionNet(in_out_dim, hidden_dim, activation)
        self.interaction_net = LegacyGNNInteraction(
            in_out_dim,
            hidden_dim,
            num_heads,
            num_layers,
            activation=activation,
            num_rbf=num_rbf,
            cutoff=thre,
            use_spatial=use_spatial,
            edge_predictor_path=edge_predictor_path,
            edge_predictor_thre=edge_predictor_thre,
            edge_predictor_root=edge_predictor_root,
        )

    def forward(self, t, z):
        with torch.set_grad_enabled(True):
            z.requires_grad_(True)
            t.requires_grad_(True)
            v = self.v_net(t, z).float()
            g = self.g_net(t, z).float()
            s = self.s_net(t, z).float()
            d = self.d_net(t, z).float()
        return v, g, s, d


class LegacyDynamicalModel(nn.Module):
    def __init__(self, legacy_cfg: dict, *, edge_predictor_root: str):
        super().__init__()
        model_cfg = legacy_cfg["model"]
        latent_dim = int(legacy_cfg["data"]["dim"])
        self.latent_dim = latent_dim
        self.components = ["velocity", "growth", "score", "interaction"]
        self.interaction_group_size = 1024
        self.legacy_config = legacy_cfg

        self.f_net = LegacyFNetInteraction(
            in_out_dim=model_cfg["in_out_dim"],
            hidden_dim=model_cfg["hidden_dim"],
            n_hiddens=model_cfg["n_hiddens"],
            activation=model_cfg["activation"],
            num_rbf=8,
            thre=model_cfg["thre"],
            use_spatial=bool(model_cfg.get("use_spatial", True)),
            num_heads=8,
            num_layers=1,
            edge_predictor_path=model_cfg.get("edge_predictor_path"),
            edge_predictor_thre=model_cfg.get("edge_predictor_thre", 0.5),
            edge_predictor_root=edge_predictor_root,
        )
        self.score_model = LegacyScoreNet2(
            in_out_dim=model_cfg["in_out_dim"],
            hidden_dim=model_cfg["score_hidden_dim"],
            activation=model_cfg["activation"],
        )

    @property
    def v_net(self):
        return self.f_net.v_net

    @property
    def g_net(self):
        return self.f_net.g_net

    @property
    def interaction_net(self):
        return self.f_net.interaction_net

    def _expand_time(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(t):
            t = torch.tensor(t, dtype=x.dtype, device=x.device)
        else:
            t = t.to(device=x.device, dtype=x.dtype)
        if t.dim() == 0:
            t = t.view(1, 1)
        elif t.dim() == 1:
            t = t.unsqueeze(1)
        if t.shape[0] == 1 and x.shape[0] > 1:
            t = t.expand(x.shape[0], 1)
        return t

    def predict_velocity(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.v_net(self._expand_time(t, x), x)

    def predict_growth(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.g_net(self._expand_time(t, x), x)

    def predict_score(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.score_model(self._expand_time(t, x), x)

    def compute_score(self, t, x, create_graph: bool = True):
        x = x.requires_grad_(True)
        out_score = self.predict_score(t=t, x=x)
        gradient = torch.autograd.grad(
            outputs=out_score,
            inputs=x,
            grad_outputs=torch.ones_like(out_score),
            create_graph=create_graph,
        )[0]
        return out_score, gradient
