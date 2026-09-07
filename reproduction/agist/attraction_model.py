"""Load the S3 radial-force checkpoint without changing the spatial model API.

The paper's Gaussian radial basis and group-mean central-force head are retained
from the attraction benchmark runtime. This loader accepts that architecture
only and uses strict checkpoint keys; it never converts weights or refits them.
"""
from __future__ import annotations

import math
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
import yaml

from CytoBridge.tl.core.models import DynamicalModel, HyperNetwork, SpatialVelocityNet
from CytoBridge.tl.downstream.checkpoint import LoadedModel, _load_state_dict, _resolve_stage_checkpoint


class GaussianRadialBasis(nn.Module):
    def __init__(self, cutoff, num_rbf):
        super().__init__()
        self.register_buffer("centers", torch.linspace(0.0, float(cutoff), int(num_rbf)))
        self.register_buffer("log_inverse_width", torch.tensor(math.log((num_rbf - 1) / cutoff)))

    def forward(self, distance):
        scaled = (distance.unsqueeze(-1) - self.centers) * torch.exp(self.log_inverse_width)
        return torch.exp(-0.5 * scaled.square())


class RadialInteraction(nn.Module):
    requires_time = True

    def __init__(self, config):
        super().__init__()
        required = {"force_mode": "pairwise_radial", "edge_mode": "radius",
                    "aggregation_mode": "group_mean", "radial_basis_mode": "gaussian",
                    "radial_envelope_mode": "none", "radial_structure": "unconstrained",
                    "radial_gene_projection_mode": "fixed_matrix"}
        if any(config.get(k) != v for k, v in required.items()) or config.get("rbf_trainable", False):
            raise ValueError("This loader requires the paper's fixed-matrix Gaussian radial-force architecture")
        self.cutoff = float(config["cutoff"])
        self.in_out_dim = 4
        self.force_mode = "pairwise_radial"
        self.aggregation_mode = "group_mean"
        self.radial_gene_projection_mode = "fixed"
        self.register_buffer("gene_force_projection_matrix", torch.tensor(config["radial_gene_projection_matrix"], dtype=torch.float32))
        self.rbf_expansion = GaussianRadialBasis(self.cutoff, int(config["num_rbf"]))
        hidden = int(config["hidden_dim"])
        if config["activation"] != "leakyrelu":
            raise ValueError("The selected S3 checkpoint uses leakyrelu")
        self.radial_force_net = nn.Sequential(
            nn.Linear(int(config["num_rbf"]), hidden), nn.LeakyReLU(),
            nn.Linear(hidden, hidden), nn.LeakyReLU(), nn.Linear(hidden, 1, bias=False),
        )

    def forward(self, x, lnw, t):
        distance = torch.norm(x[:, :2].unsqueeze(1) - x[:, :2].unsqueeze(0), dim=2)
        source, target = torch.where(distance < self.cutoff)
        keep = source != target
        source, target = source[keep], target[keep]
        radius = distance[source, target]
        keep = radius > 1e-6
        source, target, radius = source[keep], target[keep], radius[keep]
        self.edge_index = torch.stack((source, target)).detach()
        output = torch.zeros(x.shape[0], 2, dtype=x.dtype, device=x.device)
        if len(radius):
            coefficient = self.radial_force_net(self.rbf_expansion(radius)).reshape(-1)
            direction = (x[source, :2] - x[target, :2]) / radius.unsqueeze(1)
            particle_mass = torch.exp(lnw).reshape(-1) * lnw.shape[0]
            message = coefficient.unsqueeze(1) * direction * particle_mass[source].unsqueeze(1)
            output.index_add_(0, target, message)
            output = output / float(max(x.shape[0] - 1, 1))
        force = torch.zeros_like(x)
        force[:, :2] = output
        force[:, 2:] = F.linear(output, self.gene_force_projection_matrix)
        return force


class AttractionModel(DynamicalModel):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.config = config
        self.latent_dim = 4
        self.net_input_dim = 5
        self.components = list(config["components"])
        self.interaction_type = "gnn"
        self.interaction_group_size = int(config["interaction_group_size"])
        self.use_growth_in_ode_inter = True
        for name in ("velocity", "growth", "score"):
            options = dict(config[f"{name}_net"])
            if options.pop("condition_on_time", True) is not True:
                raise ValueError("The S3 checkpoint conditions all networks on time")
            if name == "velocity":
                network = SpatialVelocityNet(in_out_dim=4, spatial_dim=2, **options)
            else:
                network = HyperNetwork(input_dim=5, output_dim=1, **options)
            setattr(self, f"{name}_net", network)
        self.interaction_net = RadialInteraction(config["interaction_net"])


def load_attraction_model(model_dir, *, dim=4, device="cuda", stage="Score_Refine", **kwargs):
    if dim != 4:
        raise ValueError("S3 radial checkpoint has two spatial and two gene dimensions")
    root = Path(model_dir)
    config = yaml.safe_load((root / "config.yaml").read_text())
    model = AttractionModel(config["model"]).to(device)
    weights = _resolve_stage_checkpoint(root, stage, config)
    if weights is None:
        raise FileNotFoundError(f"No {stage} checkpoint under {root}")
    model.load_state_dict(_load_state_dict(weights, torch.device(device)), strict=True)
    model.eval()
    return LoadedModel(model, config, stage, None, weights, None)


def load_benchmark_model(model_dir, **kwargs):
    """Keep the dedicated compatibility path limited to the S3 architecture."""
    config = yaml.safe_load((Path(model_dir) / "config.yaml").read_text())
    interaction = config["model"].get("interaction_net", {})
    if interaction.get("force_mode") == "pairwise_radial" and interaction.get("radial_gene_projection_mode") == "fixed_matrix":
        return load_attraction_model(model_dir, **kwargs)
    from CytoBridge.tl.downstream.checkpoint import load_dynamical_model_from_dir
    return load_dynamical_model_from_dir(model_dir, **kwargs)
