"""Read the S4/S5 state-space checkpoints with their original attention layer.

This is the 18 July 2026 non-spatial architecture, not the spatial/gene
backbone. Checkpoints are loaded strictly, including the embedded edge prior.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Optional, Tuple
import torch
from torch import nn
import yaml
from torch_geometric.nn import MessagePassing
from CytoBridge.tl.core.interaction import ExpNormalSmearing
from CytoBridge.tl.core.models import DynamicalModel, HyperNetwork, SpatialVelocityNet
from CytoBridge.tl.graph.spatial_gnn import LinkPredictorMLP
from CytoBridge.tl.downstream.checkpoint import (
    LoadedModel, _load_state_dict, _resolve_stage_checkpoint, _score_stage_candidates,
)
_MessagePassingBase = MessagePassing

class StateGraphAttentionLayer(_MessagePassingBase):
    """Distance-conditioned GNN layer for an undifferentiated state space.

    This keeps the scalar RBF attention and particle-mass aggregation used by
    :class:`GraphAttentionLayer`, but omits the two-dimensional equivariant
    vector channel because a PCA/state space has no privileged coordinate
    axes.  Avoiding an ``N x D x H`` vector tensor also keeps 50-dimensional
    expression models tractable.
    """

    def __init__(self, hidden_dim: int, num_heads: int, activation: str = "Tanh"):
        if MessagePassing is None:
            raise ImportError(
                "torch_geometric is required for GNNInteraction."
            ) from _TORCH_GEOMETRIC_ERROR
        super().__init__(node_dim=0)
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")
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
        self.layernorm = nn.LayerNorm(hidden_dim)
        self.out_transform = nn.Sequential(
            nn.Linear(self.head_dim, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim, self.head_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        lnw: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        return_attn: bool = False,
    ) -> torch.Tensor:
        if edge_index.numel() == 0:
            if return_attn:
                self.attn = x.new_zeros((0, self.num_heads))
            return x.new_zeros((x.shape[0], self.hidden_dim))

        x_norm = self.layernorm(x)
        q = self.q_proj(x_norm).reshape(-1, self.num_heads, self.head_dim)
        k = self.k_proj(x_norm).reshape(-1, self.num_heads, self.head_dim)
        v = self.v_proj(x_norm).reshape(-1, self.num_heads, self.head_dim)
        dk = self.dk_proj(edge_attr).reshape(-1, self.num_heads, self.head_dim)
        dv = self.dv_proj(edge_attr).reshape(-1, self.num_heads, self.head_dim)
        w = torch.exp(lnw).reshape(-1) * lnw.shape[0]

        self.return_attn = bool(return_attn)
        return self.propagate(
            edge_index,
            q=q,
            k=k,
            v=v,
            w=w,
            dk=dk,
            dv=dv,
            size=(x.shape[0], x.shape[0]),
        )

    def message(self, q_i, k_j, v_j, w_j, dk, dv):
        attn = self.attn_activation((q_i * k_j * dk).sum(dim=-1))
        if self.return_attn:
            self.attn = attn
        # There is deliberately no target-only residual here: every state
        # interaction is constructed from a neighbour value and edge context.
        message = self.out_transform(v_j * dv)
        message = (message * attn.unsqueeze(-1)).reshape(-1, self.hidden_dim)
        return message * w_j.unsqueeze(-1), w_j

    def aggregate(
        self,
        features: Tuple[torch.Tensor, torch.Tensor],
        index: torch.Tensor,
        ptr,
        dim_size: Optional[int],
    ) -> torch.Tensor:
        message, weight = features
        output_size = int(dim_size or 0)
        output = torch.zeros(
            output_size,
            self.hidden_dim,
            dtype=message.dtype,
            device=message.device,
        )
        weight_sum = torch.zeros(
            output_size,
            1,
            dtype=message.dtype,
            device=message.device,
        )
        output.index_add_(0, index, message)
        weight_sum.index_add_(0, index, weight.unsqueeze(-1))
        return output / weight_sum.clamp_min(torch.finfo(message.dtype).eps)



class StateInteraction(nn.Module):
    requires_time = True

    def __init__(self, dimension, config):
        super().__init__()
        if not config.get("state_space") or config.get("use_spatial", True):
            raise ValueError("S4/S5 require the original undifferentiated state-space model")
        if config.get("force_mode", "attention") != "attention":
            raise ValueError("S4/S5 use attention, not a radial-force head")
        self.state_space, self.use_spatial = True, False
        self.force_mode = "attention"
        self.in_out_dim = dimension
        self.cutoff = float(config["cutoff"])
        self.rbf_cutoff = 1.0
        self.edge_mode = config["edge_mode"]
        self.edge_predictor_thre = float(config.get("edge_predictor_thre", .5))
        hidden = int(config["hidden_dim"])
        activations = {"leakyrelu": nn.LeakyReLU, "relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU}
        self.rbf_expansion = ExpNormalSmearing(cutoff=1., num_rbf=config["num_rbf"], trainable=config.get("rbf_trainable", False))
        if self.edge_mode == "predictor":
            self.link_predictor = LinkPredictorMLP(input_dim=2 * dimension)
            for parameter in self.link_predictor.parameters():
                parameter.requires_grad_(False)
        elif self.edge_mode != "radius":
            raise ValueError("Expected radius or predictor edge mode")
        self.distance_projection = nn.Linear(config["num_rbf"], hidden)
        self.state_embed = nn.Sequential(nn.Linear(dimension, hidden), activations[config["activation"].lower()](), nn.Linear(hidden, hidden))
        self.gnn_layers = nn.ModuleList([StateGraphAttentionLayer(hidden, config["num_heads"], config["activation"]) for _ in range(config["num_layers"])])
        self.state_readout = nn.Linear(hidden, dimension, bias=False)

    def forward(self, x, lnw, t, return_attn=False):
        distance = torch.norm(x.unsqueeze(1) - x.unsqueeze(0), dim=2)
        source, target = torch.where(distance < self.cutoff)
        if self.edge_mode == "predictor":
            pairs = torch.cat([x[source], x[target]], dim=1)
            connected = torch.sigmoid(self.link_predictor(pairs)).reshape(-1) >= self.edge_predictor_thre
            source, target = source[connected], target[connected]
        keep = (source != target) & (distance[source, target] > 1e-6)
        source, target = source[keep], target[keep]
        self.edge_index = torch.stack([source, target]).detach()
        radius = torch.clamp(distance[source, target] / self.cutoff, 0., 1.)
        embedded = self.state_embed(x)
        edge_attr = (embedded[source] + embedded[target]) * self.distance_projection(self.rbf_expansion(radius))
        for layer in self.gnn_layers:
            embedded = layer(embedded, lnw, self.edge_index, edge_attr, return_attn=return_attn)
        return self.state_readout(embedded)


class StateModel(DynamicalModel):
    def __init__(self, dimension, config):
        nn.Module.__init__(self)
        self.config = config
        self.latent_dim = int(dimension)
        self.net_input_dim = dimension + 1
        self.components = list(config["components"])
        self.interaction_type = config.get("interaction_type", "gnn")
        self.interaction_group_size = int(config.get("interaction_group_size", 16))
        self.use_growth_in_ode_inter = True
        for name in ("velocity", "growth", "score"):
            if name not in self.components:
                setattr(self, f"{name}_net", None)
                continue
            options = deepcopy(config[f"{name}_net"])
            if options.pop("condition_on_time", True) is not True:
                raise ValueError("These paper checkpoints condition all networks on time")
            if name == "velocity":
                if options.pop("use_spatial", False):
                    raise ValueError("State-space velocity must not split spatial/gene axes")
                network = HyperNetwork(input_dim=dimension + 1, output_dim=dimension, **options)
            else:
                network = HyperNetwork(input_dim=dimension + 1, output_dim=1, **options)
            setattr(self, f"{name}_net", network)
        self.interaction_net = StateInteraction(dimension, config["interaction_net"]) if "interaction" in self.components else None


def load_state_model(model_dir, *, dim=50, device="cpu", stage="Finetune", score_stage_prefer=None, edge_predictor_path=None):
    root, device = Path(model_dir), torch.device(device)
    config = yaml.safe_load((root / "config.yaml").read_text())
    weights = _resolve_stage_checkpoint(root, stage, config)
    if weights is None:
        raise FileNotFoundError(f"No {stage} checkpoint under {root}")
    model = StateModel(dim, config["model"]).to(device)
    state = _load_state_dict(weights, device)
    if model.interaction_net is not None and model.interaction_net.edge_mode == "predictor":
        embedded = any(".link_predictor." in key for key in state)
        if not embedded:
            prior = edge_predictor_path or config["model"]["interaction_net"].get("edge_predictor_path")
            if prior is None:
                raise ValueError("Checkpoint lacks embedded edge prior; pass edge_predictor_path")
            model.interaction_net.link_predictor.load_state_dict(_load_state_dict(Path(prior), device), strict=True)
            state = {**state, **{key: value for key, value in model.state_dict().items() if ".link_predictor." in key}}
    model.load_state_dict(state, strict=True)
    score_stage, score_path = None, None
    if "score" in model.components:
        for candidate in _score_stage_candidates(config, score_stage_prefer):
            path = root / candidate / "score_model.pth"
            if path.exists():
                model.score_net.load_state_dict(_load_state_dict(path, device), strict=True)
                score_stage, score_path = candidate, path
                break
    model.eval()
    return LoadedModel(model, config, stage, score_stage, weights, score_path)
