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

_MessagePassingBase = MessagePassing if MessagePassing is not None else nn.Module


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
        spatial_dim: int = 2,
        rbf_trainable: bool = False,
        edge_predictor_path: Optional[str] = None,
        edge_predictor_thre: float = 0.5,
        edge_predictor_root: Optional[str] = None,
        edge_prior_mode: str = "learned",
        load_edge_predictor_from_path: bool = True,
    ):
        super().__init__()
        if MessagePassing is None:
            raise ImportError(
                "torch_geometric is required for GNNInteraction. "
                "Install it with: pip install 'CytoBridge[graph]'"
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
        self.spatial_dim = int(spatial_dim)
        if self.spatial_dim < 0 or self.spatial_dim >= int(in_out_dim):
            raise ValueError(
                "spatial_dim must be non-negative and smaller than in_out_dim, "
                f"got spatial_dim={self.spatial_dim}, in_out_dim={in_out_dim}."
            )
        if self.use_spatial and self.spatial_dim == 0:
            raise ValueError(
                "GNNInteraction use_spatial=True requires spatial coordinates; "
                "the actual model input has spatial_dim=0."
            )
        self.cutoff = cutoff
        self.edge_predictor_thre = edge_predictor_thre
        self.edge_prior_mode = str(edge_prior_mode).lower()
        if self.edge_prior_mode not in {"learned", "all_spatial"}:
            raise ValueError("edge_prior_mode must be 'learned' or 'all_spatial'.")

        # The released ARISTA/DeepRUOT GNN kept the RBF centers and widths
        # fixed. Preserve that behavior by default; a trainable RBF remains an
        # explicit model ablation rather than an accidental semantic change.
        self.rbf_trainable = bool(rbf_trainable)
        self.rbf_expansion = ExpNormalSmearing(
            cutoff=cutoff,
            num_rbf=num_rbf,
            trainable=self.rbf_trainable,
        )

        if self.edge_prior_mode == "learned":
            # The predictor is frozen auxiliary state, not part of the matched
            # trainable interaction-backbone initialization.  Keep its random
            # initialization (which is overwritten for production models) from
            # advancing the CPU stream used by gene_embed/GAT/readout.  This
            # makes learned and all-spatial ablations share byte-identical
            # trainable GNN initial weights under the same outer seed.
            with torch.random.fork_rng(devices=[]):
                self.link_predictor = LinkPredictorMLP(input_dim=in_out_dim * 2)
                if load_edge_predictor_from_path:
                    if edge_predictor_path is None:
                        raise ValueError(
                            "edge_predictor_path must be provided when "
                            "edge_prior_mode='learned'."
                        )
                    predictor_path = os.path.expanduser(edge_predictor_path)
                    if edge_predictor_root is not None and not os.path.isabs(
                        predictor_path
                    ):
                        predictor_path = os.path.join(
                            edge_predictor_root, predictor_path
                        )
                    if not os.path.isabs(predictor_path):
                        predictor_path = os.path.abspath(predictor_path)
                    self.link_predictor.load_state_dict(
                        torch.load(
                            predictor_path,
                            map_location=torch.device("cpu"),
                            weights_only=True,
                        )
                    )
            for param in self.link_predictor.parameters():
                param.requires_grad = False
            self.link_predictor.eval()

        self.gene_embed = nn.Sequential(
            nn.Linear(in_out_dim - self.spatial_dim, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.distance_projection = nn.Linear(num_rbf, hidden_dim)
        self.gnn_layers = nn.ModuleList(
            [
                GraphAttentionLayer(hidden_dim, num_heads, activation=activation)
                for _ in range(num_layers)
            ]
        )

        self.gene_readout = nn.Sequential(
            nn.Linear(hidden_dim, in_out_dim - self.spatial_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        lnw: torch.Tensor,
        t: torch.Tensor,
        return_attn: bool = False,
    ) -> torch.Tensor:
        if self.use_spatial:
            spatial = x[:, : self.spatial_dim]
            x_expanded = spatial.unsqueeze(1)
            x_expanded_t = spatial.unsqueeze(0)
            pairwise_distances = torch.norm(x_expanded - x_expanded_t, dim=2)
            mask = (pairwise_distances < self.cutoff).float()
        else:
            x_expanded = x.unsqueeze(1)
            x_expanded_t = x.unsqueeze(0)
            pairwise_distances = torch.norm(x_expanded - x_expanded_t, dim=2)
            mask = (pairwise_distances < self.cutoff).float()

        rows, cols = torch.where(mask)
        if self.edge_prior_mode == "learned":
            features_i = x[rows]
            features_j = x[cols]
            pair_features = torch.cat([features_i, features_j], dim=1)
            pred_probs = self.link_predictor(pair_features)
            pred_probs = torch.sigmoid(pred_probs).reshape(-1)
            connected = pred_probs >= self.edge_predictor_thre
        else:
            # Matched reviewer ablation: retain the complete interaction GNN,
            # distance cutoff, RBF features, and trainable attention/readout,
            # while removing only the pretrained LR-informed edge gate.
            connected = torch.ones(rows.shape[0], dtype=torch.bool, device=x.device)

        edge_index = torch.stack([rows[connected], cols[connected]], dim=0)
        indices = edge_index[0] != edge_index[1]
        edge_index = edge_index[:, indices]
        r_ij = pairwise_distances[edge_index[0], edge_index[1]]
        edge_index = edge_index[:, r_ij > 1e-6]
        r_ij = r_ij[r_ij > 1e-6]
        # Cache for downstream attention/communication analysis (does not affect forward output).
        self.edge_index = edge_index.detach()

        x_embed = self.gene_embed(x[:, self.spatial_dim :])
        vec = torch.zeros(
            x_embed.size(0),
            self.spatial_dim,
            x_embed.size(1),
            device=x.device,
        )

        vec_ij = (
            x[edge_index[0], : self.spatial_dim] - x[edge_index[1], : self.spatial_dim]
        ) / r_ij.unsqueeze(1)
        rbf_ij = self.rbf_expansion(r_ij)
        edge_attr = (
            x_embed[edge_index[0]] + x_embed[edge_index[1]]
        ) * self.distance_projection(rbf_ij)

        for layer in self.gnn_layers:
            x_embed, vec = layer(
                x_embed,
                vec,
                lnw,
                edge_index,
                edge_attr,
                vec_ij,
                return_attn=return_attn,
            )

        x_spatial = vec.mean(dim=-1)
        x_gene = self.gene_readout(x_embed)
        x_out = torch.cat([x_spatial, x_gene], dim=1)

        return x_out


class GraphAttentionLayer(_MessagePassingBase):
    # Inference can contain hundreds of thousands of accepted edges.  PyG's
    # ordinary propagate path materializes every projected edge message at
    # once, which exceeds 24 GB for the formal 1024-particle interaction
    # groups.  Training deliberately keeps the historical path byte-for-byte;
    # no-grad inference uses contiguous chunks and accumulates the identical
    # weighted scatter-mean sufficient statistics.
    inference_edge_chunk_size = 32_768

    def __init__(self, hidden_dim: int, num_heads: int, activation: str = "Tanh"):
        if MessagePassing is None:
            raise ImportError(
                "torch_geometric is required for GNNInteraction. "
                "Install it with: pip install 'CytoBridge[graph]'"
            ) from _TORCH_GEOMETRIC_ERROR
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

    def forward(
        self, x, vec, lnw, edge_index, edge_attr, edge_vec, return_attn: bool = False
    ):
        x_res = self.res_proj(x).reshape(-1, self.num_heads, self.head_dim)
        x = self.layernorm(x)
        q = self.q_proj(x).reshape(-1, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(-1, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(-1, self.num_heads, self.head_dim)
        w = torch.exp(lnw) * lnw.shape[0]

        self.return_attn = return_attn
        if not torch.is_grad_enabled() and int(edge_index.shape[1]) > int(
            self.inference_edge_chunk_size
        ):
            return self._propagate_inference_chunks(
                edge_index=edge_index,
                q=q,
                k=k,
                v=v,
                w=w,
                x_orig=x_res,
                edge_attr=edge_attr,
                vec=vec,
                edge_vec=edge_vec,
                return_attn=return_attn,
            )

        dk = self.dk_proj(edge_attr).reshape(-1, self.num_heads, self.head_dim)
        dv = self.dv_proj(edge_attr).reshape(-1, self.num_heads, self.head_dim)
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

    def _propagate_inference_chunks(
        self,
        *,
        edge_index,
        q,
        k,
        v,
        w,
        x_orig,
        edge_attr,
        vec,
        edge_vec,
        return_attn: bool,
    ):
        """No-grad equivalent of ``propagate`` with bounded edge memory."""

        if torch.is_grad_enabled():
            raise RuntimeError(
                "Chunked graph attention is inference-only; grad-enabled calls "
                "must retain the historical MessagePassing path."
            )
        n_nodes = int(q.shape[0])
        edge_count = int(edge_index.shape[1])
        chunk_size = int(self.inference_edge_chunk_size)
        if chunk_size <= 0:
            raise ValueError("inference_edge_chunk_size must be > 0.")

        # Accumulate accepted edge messages in a fixed target-sorted order and
        # in float64.  CUDA ``index_add_`` with repeated target indices uses
        # atomic additions whose order can change when apparently unrelated
        # synchronization is introduced.  The per-layer discrepancy is tiny,
        # but split-population inference can amplify it through the coupled
        # interaction -> growth -> branching feedback.  Reducing each target
        # segment first leaves only unique-target writes and makes the bounded-
        # memory path numerically stable without changing the message formula.
        accumulator_dtype = torch.float64
        x_sum = torch.zeros(
            (n_nodes, self.hidden_dim),
            dtype=accumulator_dtype,
            device=q.device,
        )
        vec_sum = torch.zeros(
            (n_nodes, vec.shape[1], self.hidden_dim),
            dtype=accumulator_dtype,
            device=vec.device,
        )
        weight_sum = torch.zeros((n_nodes, 1), dtype=accumulator_dtype, device=w.device)
        attention_chunks = [] if return_attn else None

        for start in range(0, edge_count, chunk_size):
            stop = min(start + chunk_size, edge_count)
            source = edge_index[0, start:stop]
            target = edge_index[1, start:stop]
            chunk_attr = edge_attr[start:stop]
            dk = self.dk_proj(chunk_attr).reshape(-1, self.num_heads, self.head_dim)
            dv = self.dv_proj(chunk_attr).reshape(-1, self.num_heads, self.head_dim)
            x_message, vec_message, weight_message = self.message(
                q_i=q[target],
                k_j=k[source],
                v_j=v[source],
                vec_j=vec[source],
                w_j=w[source],
                x_orig_i=x_orig[target],
                dk=dk,
                dv=dv,
                r_ij=chunk_attr,
                d_ij=edge_vec[start:stop],
            )
            order = torch.argsort(target, stable=True)
            sorted_target = target[order]
            unique_target, counts = torch.unique_consecutive(
                sorted_target, return_counts=True
            )
            x_segment = torch.segment_reduce(
                x_message[order].to(accumulator_dtype),
                "sum",
                lengths=counts,
            )
            vec_segment = torch.segment_reduce(
                vec_message[order].to(accumulator_dtype),
                "sum",
                lengths=counts,
            )
            weight_segment = torch.segment_reduce(
                weight_message[order].to(accumulator_dtype),
                "sum",
                lengths=counts,
            )
            x_sum[unique_target] = x_sum[unique_target] + x_segment
            vec_sum[unique_target] = vec_sum[unique_target] + vec_segment
            weight_sum[unique_target] = weight_sum[unique_target] + weight_segment
            if attention_chunks is not None:
                attention_chunks.append(self.attn)

        denominator = weight_sum.clone()
        denominator[denominator == 0] = 1
        if attention_chunks is not None:
            self.attn = torch.cat(attention_chunks, dim=0)
        return (
            (x_sum / denominator).to(q.dtype),
            (vec_sum / denominator.unsqueeze(1)).to(vec.dtype),
        )

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

    def manual_scatter_mean(
        self,
        src: torch.Tensor,
        weight: torch.Tensor,
        index: torch.Tensor,
        dim: int,
        output_size: int,
    ) -> torch.Tensor:
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
        ptr,
        dim_size: Optional[int],
    ):
        x, vec, w = features
        aggregation_dim = self.node_dim
        aggregated_x = self.manual_scatter_mean(
            x, w, index, dim=aggregation_dim, output_size=dim_size
        )
        aggregated_vec = self.manual_scatter_mean(
            vec, w, index, dim=aggregation_dim, output_size=dim_size
        )
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
