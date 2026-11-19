"""Exact attribution for the released one-layer spatial GNN interaction.

The attention gate saved by :mod:`CytoBridge.tl.downstream.attention` is only
one factor in a graph message.  This module decomposes the *complete* spatial
GNN interaction output into directed edge contributions and verifies that the
contributions, together with the gene-readout bias, reconstruct the official
forward output.

The implementation is deliberately fail-closed.  It supports the released
one-layer ``GNNInteraction`` architecture and rejects multi-layer or structurally
different models, for which an edge contribution from an early layer is no
longer a simple additive term in the final output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "SpatialExactDecomposition",
    "SpatialGroupedAttribution",
    "analyze_spatial_gnn_by_celltype",
    "decompose_spatial_gnn_group",
    "make_interaction_groups",
    "validate_spatial_exact_decomposition_model",
]


@dataclass(frozen=True)
class SpatialExactDecomposition:
    """Exact output decomposition for one interaction group.

    ``edge_output`` is aligned to ``edge_index`` and has the same feature
    dimension as the interaction output.  For every receiver ``i``:

    ``output[i] == baseline[i] + edge_output[target == i].sum(0)``.

    The baseline is zero in the two spatial dimensions and equals the linear
    gene-readout bias in the remaining dimensions.  It is intentionally not
    assigned to any sender.
    """

    output: object
    baseline: object
    reconstructed: object
    edge_index: object
    edge_output: object
    attention_signed: object
    attention_abs_mean: object
    source_mass_fraction: object
    edge_predictor_probability: object
    edge_distance: object
    max_abs_residual: float
    relative_l2_residual: float


@dataclass(frozen=True)
class SpatialGroupedAttribution:
    """Cell-type attribution aggregated over one deterministic grouping."""

    type_pair_table: pd.DataFrame
    edge_table: pd.DataFrame
    edge_output: np.ndarray
    attention_signed: np.ndarray
    reconstruction_table: pd.DataFrame
    groups: tuple[np.ndarray, ...]


def _require_attrs(owner, names: Sequence[str], *, owner_name: str) -> None:
    missing = [name for name in names if not hasattr(owner, name)]
    if missing:
        raise TypeError(f"{owner_name} is missing required attributes: {missing}.")


def _single_linear(module):
    """Return the sole Linear-like module in a readout container."""
    import torch.nn as nn

    if isinstance(module, nn.Linear):
        return module
    children = list(module.children())
    if len(children) == 1 and isinstance(children[0], nn.Linear):
        return children[0]
    raise TypeError(
        "Exact spatial decomposition requires gene_readout to contain exactly "
        "one torch.nn.Linear module."
    )


def validate_spatial_exact_decomposition_model(interaction_net) -> None:
    """Validate the architecture contract used by exact decomposition."""
    _require_attrs(
        interaction_net,
        (
            "use_spatial",
            "gnn_layers",
            "gene_embed",
            "gene_readout",
            "distance_projection",
            "rbf_expansion",
            "link_predictor",
            "edge_predictor_thre",
        ),
        owner_name="interaction_net",
    )
    if not bool(interaction_net.use_spatial):
        raise ValueError(
            "This API is for the spatial GNN. Use the state-space attribution "
            "API for use_spatial=False models."
        )
    if len(interaction_net.gnn_layers) != 1:
        raise ValueError(
            "Exact spatial edge attribution currently supports exactly one GNN "
            f"layer; observed {len(interaction_net.gnn_layers)}."
        )
    layer = interaction_net.gnn_layers[0]
    _require_attrs(
        layer,
        (
            "hidden_dim",
            "num_heads",
            "head_dim",
            "res_proj",
            "layernorm",
            "q_proj",
            "k_proj",
            "v_proj",
            "dk_proj",
            "dv_proj",
            "out_transform",
            "attn_activation",
            "s_proj",
            "activation",
        ),
        owner_name="gnn_layers[0]",
    )
    if int(layer.hidden_dim) != int(layer.num_heads) * int(layer.head_dim):
        raise ValueError(
            "hidden_dim must equal num_heads * head_dim for exact attribution."
        )
    _single_linear(interaction_net.gene_readout)


def _as_column_lnw(lnw, *, n_cells: int, device, dtype):
    import torch

    values = torch.as_tensor(lnw, device=device, dtype=dtype)
    if values.ndim == 1:
        values = values[:, None]
    if values.shape != (n_cells, 1):
        raise ValueError(
            f"lnw must have shape ({n_cells}, 1) or ({n_cells},), got "
            f"{tuple(values.shape)}."
        )
    return values


def decompose_spatial_gnn_group(
    interaction_net,
    x,
    lnw,
    t,
    *,
    reconstruction_atol: float = 2e-5,
    reconstruction_rtol: float = 2e-5,
) -> SpatialExactDecomposition:
    """Decompose one spatial-GNN group into exact directed-edge messages.

    Parameters
    ----------
    interaction_net
        Released one-layer spatial ``GNNInteraction`` instance.
    x
        Joint coordinates with two spatial columns followed by model-state
        columns.
    lnw
        Per-particle log mass in the convention passed directly to
        ``GNNInteraction``.
    t
        Time value.  The released interaction network accepts it for API
        consistency but does not otherwise transform messages by time.
    reconstruction_atol, reconstruction_rtol
        Fail-closed numerical reconstruction tolerances.
    """
    import torch
    import torch.nn.functional as F

    validate_spatial_exact_decomposition_model(interaction_net)
    x = torch.as_tensor(x)
    if not torch.is_floating_point(x):
        x = x.float()
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 3:
        raise ValueError("x must be a floating N x (2 + state_dim) matrix with N >= 2.")
    lnw = _as_column_lnw(
        lnw, n_cells=int(x.shape[0]), device=x.device, dtype=x.dtype
    )
    t = torch.as_tensor(t, device=x.device, dtype=x.dtype)

    was_training = bool(interaction_net.training)
    interaction_net.eval()
    try:
        with torch.no_grad():
            output = interaction_net(x, lnw, t, return_attn=True)
            edge_index = interaction_net.edge_index.detach().clone()
            if edge_index.ndim != 2 or edge_index.shape[0] != 2:
                raise RuntimeError("interaction_net.edge_index must have shape (2, E).")

            source = edge_index[0]
            target = edge_index[1]
            n_cells = int(x.shape[0])
            x_embed = interaction_net.gene_embed(x[:, 2:])
            layer = interaction_net.gnn_layers[0]

            distance = torch.linalg.vector_norm(
                x[source, :2] - x[target, :2], dim=1
            )
            if bool(torch.any(distance <= 1e-6)):
                raise RuntimeError("Official edge list unexpectedly contains zero-distance edges.")
            direction = (x[source, :2] - x[target, :2]) / distance[:, None]
            rbf = interaction_net.rbf_expansion(distance)
            edge_attr = (
                x_embed[source] + x_embed[target]
            ) * interaction_net.distance_projection(rbf)

            x_res = layer.res_proj(x_embed).reshape(
                -1, int(layer.num_heads), int(layer.head_dim)
            )
            x_norm = layer.layernorm(x_embed)
            q = layer.q_proj(x_norm).reshape(
                -1, int(layer.num_heads), int(layer.head_dim)
            )
            k = layer.k_proj(x_norm).reshape(
                -1, int(layer.num_heads), int(layer.head_dim)
            )
            v = layer.v_proj(x_norm).reshape(
                -1, int(layer.num_heads), int(layer.head_dim)
            )
            dk = layer.dk_proj(edge_attr).reshape(
                -1, int(layer.num_heads), int(layer.head_dim)
            )
            dv = layer.dv_proj(edge_attr).reshape(
                -1, int(layer.num_heads), int(layer.head_dim)
            )

            attention = layer.attn_activation(
                (q[target] * k[source] * dk).sum(dim=-1)
            )
            cached_attention = getattr(layer, "attn", None)
            if cached_attention is None or cached_attention.shape != attention.shape:
                raise RuntimeError("Official forward did not expose aligned per-edge attention.")
            if not torch.allclose(
                attention,
                cached_attention,
                atol=reconstruction_atol,
                rtol=reconstruction_rtol,
            ):
                raise RuntimeError("Recomputed attention is not aligned to the official edge list.")

            scalar_message = v[source] * dv + x_res[target]
            scalar_message = layer.out_transform(scalar_message)
            scalar_message = (
                scalar_message * attention.unsqueeze(-1)
            ).reshape(-1, int(layer.hidden_dim))
            _, spatial_scale = torch.split(
                layer.activation(layer.s_proj(scalar_message)),
                int(layer.hidden_dim),
                dim=1,
            )
            vector_message = spatial_scale[:, None, :] * direction[:, :, None]

            source_mass = torch.exp(lnw) * float(n_cells)
            edge_mass = source_mass[source]
            receiver_mass = torch.zeros(
                (n_cells, 1), device=x.device, dtype=x.dtype
            )
            if target.numel():
                receiver_mass.index_add_(0, target, edge_mass)
                mass_fraction = edge_mass / receiver_mass[target]
            else:
                mass_fraction = torch.empty((0, 1), device=x.device, dtype=x.dtype)

            scalar_edge = scalar_message * mass_fraction
            vector_edge = vector_message * mass_fraction[:, None, :]
            spatial_edge = vector_edge.mean(dim=-1)

            readout = _single_linear(interaction_net.gene_readout)
            gene_edge = F.linear(scalar_edge, readout.weight, bias=None)
            edge_output = torch.cat((spatial_edge, gene_edge), dim=1)

            baseline = torch.zeros_like(output)
            if readout.bias is not None:
                baseline[:, 2:] = readout.bias[None, :]
            reconstructed = baseline.clone()
            if target.numel():
                reconstructed.index_add_(0, target, edge_output)

            pair_features = torch.cat((x[source], x[target]), dim=1)
            predictor_probability = torch.sigmoid(
                interaction_net.link_predictor(pair_features)
            ).reshape(-1)
            if predictor_probability.numel() and bool(
                torch.any(
                    predictor_probability
                    < float(interaction_net.edge_predictor_thre) - 1e-7
                )
            ):
                raise RuntimeError("Official edge list contains an edge below predictor threshold.")

            residual = reconstructed - output
            max_abs = float(residual.abs().max().item()) if residual.numel() else 0.0
            relative_l2 = float(
                torch.linalg.vector_norm(residual)
                / torch.clamp(torch.linalg.vector_norm(output), min=1e-12)
            )
            if not torch.allclose(
                reconstructed,
                output,
                atol=reconstruction_atol,
                rtol=reconstruction_rtol,
            ):
                raise RuntimeError(
                    "Exact spatial edge decomposition failed reconstruction: "
                    f"max_abs={max_abs:.6g}, relative_l2={relative_l2:.6g}."
                )

            return SpatialExactDecomposition(
                output=output.detach(),
                baseline=baseline.detach(),
                reconstructed=reconstructed.detach(),
                edge_index=edge_index.detach(),
                edge_output=edge_output.detach(),
                attention_signed=attention.detach(),
                attention_abs_mean=attention.abs().mean(dim=1).detach(),
                source_mass_fraction=mass_fraction.reshape(-1).detach(),
                edge_predictor_probability=predictor_probability.detach(),
                edge_distance=distance.detach(),
                max_abs_residual=max_abs,
                relative_l2_residual=relative_l2,
            )
    finally:
        interaction_net.train(was_training)


def make_interaction_groups(
    n_cells: int,
    group_size: int,
    *,
    random_state: int,
) -> tuple[np.ndarray, ...]:
    """Create explicit groups with the released ``cal_interaction_gnn`` sizes.

    When there is a remainder, the released runtime combines it with one full
    group.  For example, 5,271 cells at group size 1,024 become four groups of
    1,024 and one group of 1,175, rather than five full groups plus 151 cells.
    """
    n_cells = int(n_cells)
    group_size = int(group_size)
    if n_cells < 2:
        raise ValueError("At least two cells are required.")
    if group_size < 2:
        raise ValueError("group_size must be at least two.")
    permutation = np.random.default_rng(int(random_state)).permutation(n_cells)
    if n_cells % group_size == 0:
        n_full = n_cells // group_size
        remainder = 0
    elif n_cells < group_size:
        n_full = 0
        remainder = n_cells
    else:
        n_full = n_cells // group_size - 1
        remainder = n_cells % group_size + group_size
    groups = [
        permutation[index * group_size : (index + 1) * group_size]
        for index in range(n_full)
    ]
    if remainder:
        groups.append(permutation[n_full * group_size :])
    if any(group.size < 2 for group in groups):
        raise RuntimeError("Interaction grouping produced an isolated cell.")
    observed = np.concatenate(groups)
    if observed.size != n_cells or np.unique(observed).size != n_cells:
        raise RuntimeError("Interaction groups do not form a partition of all cells.")
    return tuple(np.asarray(group, dtype=np.int64) for group in groups)


def _space_slices(feature_dim: int, spatial_dim: int) -> dict[str, slice]:
    spaces = {"joint": slice(0, feature_dim)}
    if spatial_dim:
        spaces["spatial"] = slice(0, spatial_dim)
    if spatial_dim < feature_dim:
        spaces["state"] = slice(spatial_dim, feature_dim)
    return spaces


def analyze_spatial_gnn_by_celltype(
    interaction_net,
    x,
    lnw,
    t,
    labels: Sequence[object],
    *,
    group_size: int,
    grouping_seed: int,
    spatial_dim: int = 2,
) -> SpatialGroupedAttribution:
    """Aggregate exact edge messages and attention by sender/receiver type.

    The returned ``D_AB`` is a mean-of-receiver-norms, including zero for a
    receiver with no incoming edge from sender type A.  This matches the formal
    state-space definition while retaining separate joint, spatial, and state
    norms for a spatial model.
    """
    import torch

    x = torch.as_tensor(x)
    if not torch.is_floating_point(x):
        x = x.float()
    n_cells, feature_dim = map(int, x.shape)
    labels = np.asarray(labels).astype(str)
    if labels.shape != (n_cells,):
        raise ValueError(f"labels must have shape ({n_cells},), got {labels.shape}.")
    lnw = _as_column_lnw(
        lnw, n_cells=n_cells, device=x.device, dtype=x.dtype
    )
    types, type_id = np.unique(labels, return_inverse=True)
    n_types = int(types.size)
    receiver_counts = np.bincount(type_id, minlength=n_types).astype(float)
    spaces = _space_slices(feature_dim, int(spatial_dim))
    groups = make_interaction_groups(
        n_cells, group_size, random_state=int(grouping_seed)
    )
    global_mass = torch.exp(lnw) * float(n_cells)

    gate_sum = np.zeros((n_types, n_types), dtype=np.float64)
    gate_count = np.zeros((n_types, n_types), dtype=np.int64)
    mass_gate_sum = np.zeros((n_types, n_types), dtype=np.float64)
    predictor_sum = np.zeros((n_types, n_types), dtype=np.float64)
    distance_sum = np.zeros((n_types, n_types), dtype=np.float64)
    drift_norm_sum = {
        name: np.zeros((n_types, n_types), dtype=np.float64) for name in spaces
    }
    edge_frames: list[pd.DataFrame] = []
    edge_outputs: list[np.ndarray] = []
    signed_attentions: list[np.ndarray] = []
    reconstruction_rows: list[dict[str, object]] = []

    for group_index, indices in enumerate(groups):
        index_tensor = torch.as_tensor(indices, device=x.device, dtype=torch.long)
        group_mass = global_mass[index_tensor]
        group_lnw = torch.log(group_mass / float(indices.size))
        result = decompose_spatial_gnn_group(
            interaction_net,
            x[index_tensor],
            group_lnw,
            t,
        )
        local_edges = result.edge_index.detach().cpu().numpy()
        source_global = indices[local_edges[0]]
        target_global = indices[local_edges[1]]
        source_type = type_id[source_global]
        target_type = type_id[target_global]
        gate = result.attention_abs_mean.detach().cpu().numpy().astype(float)
        fraction = result.source_mass_fraction.detach().cpu().numpy().astype(float)
        predictor = (
            result.edge_predictor_probability.detach().cpu().numpy().astype(float)
        )
        distance = result.edge_distance.detach().cpu().numpy().astype(float)
        contribution = result.edge_output.detach().cpu().numpy().astype(np.float32)
        signed = result.attention_signed.detach().cpu().numpy().astype(np.float32)

        np.add.at(gate_sum, (source_type, target_type), gate)
        np.add.at(gate_count, (source_type, target_type), 1)
        np.add.at(mass_gate_sum, (source_type, target_type), gate * fraction)
        np.add.at(predictor_sum, (source_type, target_type), predictor)
        np.add.at(distance_sum, (source_type, target_type), distance)

        local_target_types = type_id[indices]
        for sender_type in np.unique(source_type):
            mask = source_type == sender_type
            sender_contribution = np.zeros(
                (indices.size, feature_dim), dtype=np.float64
            )
            np.add.at(
                sender_contribution,
                local_edges[1, mask],
                contribution[mask].astype(np.float64),
            )
            for space_name, columns in spaces.items():
                norms = np.linalg.norm(sender_contribution[:, columns], axis=1)
                drift_norm_sum[space_name][sender_type] += np.bincount(
                    local_target_types,
                    weights=norms,
                    minlength=n_types,
                )

        edge_frames.append(
            pd.DataFrame(
                {
                    "grouping_seed": int(grouping_seed),
                    "group_index": int(group_index),
                    "source_index": source_global,
                    "target_index": target_global,
                    "sender_type": labels[source_global],
                    "receiver_type": labels[target_global],
                    "attention_abs_mean": gate,
                    "source_mass_fraction": fraction,
                    "mass_weighted_attention": gate * fraction,
                    "edge_predictor_probability": predictor,
                    "spatial_distance": distance,
                    "edge_message_norm_joint": np.linalg.norm(contribution, axis=1),
                    "edge_message_norm_spatial": np.linalg.norm(
                        contribution[:, :spatial_dim], axis=1
                    ),
                    "edge_message_norm_state": np.linalg.norm(
                        contribution[:, spatial_dim:], axis=1
                    ),
                }
            )
        )
        edge_outputs.append(contribution)
        signed_attentions.append(signed)
        gene_bias = result.baseline[0, spatial_dim:].detach().cpu().numpy()
        output = result.output.detach().cpu().numpy()
        bias_l2 = float(np.linalg.norm(gene_bias))
        output_l2 = float(np.linalg.norm(output))
        reconstruction_rows.append(
            {
                "grouping_seed": int(grouping_seed),
                "group_index": int(group_index),
                "n_cells": int(indices.size),
                "n_edges": int(local_edges.shape[1]),
                "max_abs_residual": result.max_abs_residual,
                "relative_l2_residual": result.relative_l2_residual,
                "gene_readout_bias_l2_per_cell": bias_l2,
                "interaction_output_l2": output_l2,
                "bias_to_output_l2_ratio": (
                    float(np.sqrt(indices.size) * bias_l2 / output_l2)
                    if output_l2 > 0
                    else np.nan
                ),
            }
        )

    rows: list[dict[str, object]] = []
    for sender in range(n_types):
        for receiver in range(n_types):
            count = int(gate_count[sender, receiver])
            row = {
                "grouping_seed": int(grouping_seed),
                "sender_type": str(types[sender]),
                "receiver_type": str(types[receiver]),
                "n_sender_cells": int(np.sum(type_id == sender)),
                "n_receiver_cells": int(receiver_counts[receiver]),
                "edge_count": count,
                "G_AB_attention_mean": (
                    float(gate_sum[sender, receiver] / count) if count else 0.0
                ),
                "A_AB_mass_attention": (
                    float(mass_gate_sum[sender, receiver] / receiver_counts[receiver])
                    if receiver_counts[receiver] > 0
                    else 0.0
                ),
                "edge_predictor_probability_mean": (
                    float(predictor_sum[sender, receiver] / count) if count else 0.0
                ),
                "spatial_distance_mean": (
                    float(distance_sum[sender, receiver] / count) if count else np.nan
                ),
            }
            for space_name in spaces:
                row[f"D_AB_{space_name}"] = (
                    float(
                        drift_norm_sum[space_name][sender, receiver]
                        / receiver_counts[receiver]
                    )
                    if receiver_counts[receiver] > 0
                    else 0.0
                )
            rows.append(row)

    edge_table = pd.concat(edge_frames, ignore_index=True) if edge_frames else pd.DataFrame()
    return SpatialGroupedAttribution(
        type_pair_table=pd.DataFrame(rows),
        edge_table=edge_table,
        edge_output=(
            np.concatenate(edge_outputs, axis=0)
            if edge_outputs
            else np.empty((0, feature_dim), dtype=np.float32)
        ),
        attention_signed=(
            np.concatenate(signed_attentions, axis=0)
            if signed_attentions
            else np.empty(
                (0, int(interaction_net.gnn_layers[0].num_heads)), dtype=np.float32
            )
        ),
        reconstruction_table=pd.DataFrame(reconstruction_rows),
        groups=groups,
    )
