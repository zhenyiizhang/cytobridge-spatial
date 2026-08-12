"""Attention analysis for cell-cell communication.

This module provides functions for computing and analyzing attention weights
between cells, including cell-type level aggregation and permutation testing.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, TYPE_CHECKING, Union

import numpy as np

if TYPE_CHECKING:
    import anndata as ad
    import torch

__all__ = [
    "save_interpolated_attention",
    "analyze_attention_by_celltype",
]


def _prepare_model_input_from_adata(adata) -> np.ndarray:
    """Prepare model input in the same convention as training/downstream dynamics."""
    if "X_latent" in adata.obsm:
        latent = np.asarray(adata.obsm["X_latent"], dtype=np.float32)
    else:
        X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
        latent = np.asarray(X, dtype=np.float32)

    if "spatial_aligned" in adata.obsm:
        spatial = np.asarray(adata.obsm["spatial_aligned"], dtype=np.float32)
        if spatial.shape[0] != latent.shape[0]:
            raise ValueError(
                f"Row mismatch between spatial_aligned ({spatial.shape[0]}) and latent ({latent.shape[0]})."
            )
        return np.hstack((spatial, latent)).astype(np.float32)
    return latent


def _radius_neighbor_candidates(
    model_input: np.ndarray,
    *,
    cutoff: float,
    use_spatial: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Return directed radius-neighbor candidates in dense-forward order."""
    from scipy.spatial import cKDTree

    coordinates = model_input[:, :2] if use_spatial else model_input
    # Query slightly beyond the cutoff, then reproduce the model's strict
    # float32 distance test below.  The expansion only protects borderline
    # pairs from float64/float32 round-off in cKDTree.
    query_radius = float(cutoff) + max(1e-7, abs(float(cutoff)) * 1e-6)
    pairs = cKDTree(coordinates).query_pairs(query_radius, output_type="ndarray")
    if pairs.size == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty

    source = np.concatenate((pairs[:, 0], pairs[:, 1])).astype(np.int64)
    target = np.concatenate((pairs[:, 1], pairs[:, 0])).astype(np.int64)
    order = np.lexsort((target, source))
    return source[order], target[order]


def _select_interaction_edges(
    interaction_net,
    data,
    source: np.ndarray,
    target: np.ndarray,
    *,
    edge_batch_size: int,
):
    """Apply the exact cutoff and learned edge prior without dense N x N tensors."""
    import torch

    coordinate_dim = 2 if bool(interaction_net.use_spatial) else data.shape[1]
    learned_prior = getattr(interaction_net, "edge_prior_mode", "learned") == "learned"
    selected_source = []
    selected_target = []

    for start in range(0, source.size, edge_batch_size):
        stop = min(start + edge_batch_size, source.size)
        source_batch = torch.as_tensor(
            source[start:stop], dtype=torch.long, device=data.device
        )
        target_batch = torch.as_tensor(
            target[start:stop], dtype=torch.long, device=data.device
        )
        distance = torch.linalg.vector_norm(
            data[source_batch, :coordinate_dim] - data[target_batch, :coordinate_dim],
            dim=1,
        )
        keep = (distance < float(interaction_net.cutoff)) & (distance > 1e-6)

        if learned_prior:
            pair_features = torch.cat(
                (data[source_batch], data[target_batch]), dim=1
            )
            probability = torch.sigmoid(
                interaction_net.link_predictor(pair_features)
            ).reshape(-1)
            keep &= probability >= float(interaction_net.edge_predictor_thre)

        selected_source.append(source_batch[keep])
        selected_target.append(target_batch[keep])

    if not selected_source:
        empty = torch.empty(0, dtype=torch.long, device=data.device)
        return empty, empty
    return torch.cat(selected_source), torch.cat(selected_target)


def _first_layer_attention(
    interaction_net,
    data,
    source,
    target,
    *,
    edge_batch_size: int,
):
    """Evaluate the first GNN layer's attention gate in edge batches.

    ``edge_index[0]`` is the PyG message source (sender, ``j``) and
    ``edge_index[1]`` is the message target (receiver, ``i``).
    """
    import torch

    x_embed = interaction_net.gene_embed(data[:, 2:])
    layer = interaction_net.gnn_layers[0]
    x_norm = layer.layernorm(x_embed)
    q = layer.q_proj(x_norm).reshape(
        -1, int(layer.num_heads), int(layer.head_dim)
    )
    k = layer.k_proj(x_norm).reshape(
        -1, int(layer.num_heads), int(layer.head_dim)
    )
    coordinate_dim = 2 if bool(interaction_net.use_spatial) else data.shape[1]
    attention = []

    for start in range(0, source.numel(), edge_batch_size):
        stop = min(start + edge_batch_size, source.numel())
        source_batch = source[start:stop]
        target_batch = target[start:stop]
        distance = torch.linalg.vector_norm(
            data[source_batch, :coordinate_dim] - data[target_batch, :coordinate_dim],
            dim=1,
        )
        rbf = interaction_net.rbf_expansion(distance)
        edge_attr = (
            x_embed[source_batch] + x_embed[target_batch]
        ) * interaction_net.distance_projection(rbf)
        dk = layer.dk_proj(edge_attr).reshape(
            -1, int(layer.num_heads), int(layer.head_dim)
        )
        gate = layer.attn_activation(
            (q[target_batch] * k[source_batch] * dk).sum(dim=-1)
        )
        attention.append(gate.abs().mean(dim=1))

    if not attention:
        return torch.empty(0, dtype=data.dtype, device=data.device)
    return torch.cat(attention)


def save_interpolated_attention(
    adata: "ad.AnnData",
    time_value: float,
    model=None,
    f_net=None,
    device: str = "cpu",
    out_dir: Optional[str] = None,
    save_files: bool = True,
    save_dense_matrix: bool = False,
    edge_batch_size: int = 131_072,
) -> Dict[str, np.ndarray]:
    """Compute first-layer attention on sparse radius-neighbor edges.
    
    Parameters
    ----------
    adata : AnnData
        AnnData with model input features (prefers obsm['X_latent'] and optional obsm['spatial_aligned']).
    time_value : float
        Time point for attention computation.
    model : DynamicalModel, optional
        Trained CytoBridge DynamicalModel with interaction_net.
    f_net : torch.nn.Module, optional
        Legacy interaction/ODE module used by the original MOSTA helper.
    device : str
        Device for computation.
    out_dir : str, optional
        Output directory for saving arrays. If None, uses current directory.
    save_files : bool
        Whether to save arrays to disk.
    save_dense_matrix : bool
        Whether to materialize the dense N x N attention matrix. Disabled by
        default because the edge representation contains the same information.
    edge_batch_size : int
        Number of candidate edges evaluated per learned-prior/attention batch.
        
    Returns
    -------
    dict
        Dictionary with ``attn_mean`` and ``edge_index``; ``attn_matrix`` is
        included only when explicitly requested.
    """
    import torch

    if f_net is not None:
        interaction_net_owner = f_net
    elif model is not None:
        interaction_net_owner = model
    else:
        raise ValueError("Provide either `model` or `f_net`.")

    interaction_net = getattr(interaction_net_owner, "interaction_net", None)
    if interaction_net is None and hasattr(interaction_net_owner, "ode_func"):
        f_net = interaction_net_owner.ode_func
        interaction_net = getattr(f_net, "interaction_net", None)

    if interaction_net is None:
        raise ValueError("Model does not have an interaction_net component")

    interaction_net = interaction_net.to(device)
    model_input = _prepare_model_input_from_adata(adata)
    data = torch.tensor(model_input, dtype=torch.float32, device=device)
    n_particles = data.shape[0]
    if edge_batch_size <= 0:
        raise ValueError("edge_batch_size must be positive.")

    candidate_source, candidate_target = _radius_neighbor_candidates(
        model_input,
        cutoff=float(interaction_net.cutoff),
        use_spatial=bool(interaction_net.use_spatial),
    )
    was_training = bool(interaction_net.training)
    interaction_net.eval()
    try:
        with torch.no_grad():
            source, target = _select_interaction_edges(
                interaction_net,
                data,
                candidate_source,
                candidate_target,
                edge_batch_size=int(edge_batch_size),
            )
            attn = _first_layer_attention(
                interaction_net,
                data,
                source,
                target,
                edge_batch_size=int(edge_batch_size),
            )
    finally:
        interaction_net.train(was_training)

    edge_index = torch.stack((source, target)).cpu().numpy()
    attn_mean = attn.cpu().numpy()

    out = {"attn_mean": attn_mean, "edge_index": edge_index}
    attn_matrix = None
    if save_dense_matrix:
        attn_matrix = np.zeros((n_particles, n_particles), dtype=float)
        attn_matrix[edge_index[0], edge_index[1]] = attn_mean
        out["attn_matrix"] = attn_matrix

    if save_files:
        import os

        if out_dir is None:
            out_dir = os.getcwd()
        os.makedirs(out_dir, exist_ok=True)
        if save_dense_matrix and attn_matrix is not None:
            np.save(os.path.join(out_dir, f"attn_interp_t{time_value}.npy"), attn_matrix)
        np.save(os.path.join(out_dir, f"attn_mean_interp_t{time_value}.npy"), attn_mean)
        np.save(os.path.join(out_dir, f"edge_index_interp_t{time_value}.npy"), edge_index)

    return out


def analyze_attention_by_celltype(
    edge_index: np.ndarray,
    attn: np.ndarray,
    labels: np.ndarray,
    spatial_coord: Optional[np.ndarray] = None,
    time_title: Optional[str] = None,
    remove_self_loop: bool = True,
    winsor_quantile: float = 0.995,
    distance_bins: Union[str, Sequence[float]] = "quartile",
    n_permutations: int = 0,
    random_state: int = 0,
    show_plots: bool = True,
    plot: Optional[bool] = None,
) -> Dict:
    """Aggregate attention weights at cell-type level with optional distance stratification.
    
    This function analyzes attention patterns between cell types, computing
    various aggregation metrics and optionally performing permutation testing.
    
    Parameters
    ----------
    edge_index : np.ndarray
        Edge indices of shape (2, n_edges), with sender/source in row 0 and
        receiver/target in row 1 (PyG ``source_to_target`` convention).
    attn : np.ndarray
        Attention weights of shape (n_edges,).
    labels : np.ndarray
        Cell type labels of shape (n_cells,).
    spatial_coord : np.ndarray, optional
        Spatial coordinates of shape (n_cells, 2) for distance stratification.
    time_title : str, optional
        Title suffix for plots.
    remove_self_loop : bool
        Whether to remove self-loops from analysis.
    winsor_quantile : float
        Quantile for Winsorizing attention values.
    distance_bins : str or Sequence[float]
        Either "quartile" or explicit bin edges.
    n_permutations : int
        Number of permutations for statistical testing.
    random_state : int
        Random seed for permutation testing.
    show_plots : bool
        Whether to display heatmap plots.
        
    Returns
    -------
    dict
        Dictionary containing:
        - types: unique cell types
        - M_sum: sum of attention per type pair
        - M_per_source: attention per source cell
        - M_row: row-normalized attention
        - M_mean: mean attention per edge
        - asym: directional asymmetry matrix
        - type_stats: DataFrame with in/out strength statistics
        - fdr, sig_mask: FDR and significance (if permutation testing)
        - distance_bins, M_per_source_bybin: distance-stratified results
    """
    import pandas as pd
    from scipy.sparse import coo_matrix
    
    edge_index = np.asarray(edge_index)
    attn = np.asarray(attn).astype(float)
    labels = np.asarray(labels)
    
    if edge_index.shape[0] != 2:
        raise ValueError("edge_index should have shape (2, E)")
    if attn.shape[0] != edge_index.shape[1]:
        raise ValueError("attn length must match number of edges")
    if spatial_coord is not None:
        spatial_coord = np.asarray(spatial_coord)
        if spatial_coord.shape[0] != labels.shape[0] or spatial_coord.ndim != 2:
            raise ValueError("spatial_coord must be N×D and align with labels order")
    
    send = edge_index[0].copy()
    recv = edge_index[1].copy()
    w = attn.copy()
    
    if remove_self_loop:
        m = send != recv
        send, recv, w = send[m], recv[m], w[m]
    
    if w.size > 0 and winsor_quantile is not None and 0.9 < winsor_quantile < 1.0:
        hi = np.quantile(w, winsor_quantile)
        w = np.minimum(w, hi)
    
    types, type_id = np.unique(labels, return_inverse=True)
    T = len(types)
    n_per_type = np.bincount(type_id, minlength=T).astype(float)
    n_per_type[n_per_type == 0] = 1.0
    
    M_sum = coo_matrix((w, (type_id[send], type_id[recv])), shape=(T, T)).toarray()
    M_per_source = M_sum / n_per_type[:, None]
    row_sums = M_sum.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    M_row = M_sum / row_sums
    
    edge_counts = coo_matrix((np.ones_like(w), (type_id[send], type_id[recv])), shape=(T, T)).toarray()
    edge_counts_safe = edge_counts.copy()
    edge_counts_safe[edge_counts_safe == 0] = 1.0
    M_mean = M_sum / edge_counts_safe
    
    out_strength = M_sum.sum(axis=1)
    in_strength = M_sum.sum(axis=0)
    type_stats = pd.DataFrame({
        "type": types,
        "out_strength": out_strength,
        "in_strength": in_strength,
        "net_out_minus_in": out_strength - in_strength,
    }).sort_values("net_out_minus_in", ascending=False)
    
    asym = M_per_source - M_per_source.T
    
    # Permutation testing
    fdr = None
    sig_mask = None
    if n_permutations and n_permutations > 0:
        rng = np.random.default_rng(random_state)
        null = np.zeros((T, T, n_permutations))
        for b in range(n_permutations):
            shuf = rng.permutation(type_id)
            Mb = coo_matrix((w, (shuf[send], shuf[recv])), shape=(T, T)).toarray()
            null[..., b] = Mb / n_per_type[:, None]
        obs = M_per_source
        p = (null >= obs[..., None]).mean(axis=2)
        p_flat = p.ravel()
        order = np.argsort(p_flat)
        adjusted_sorted = (
            p_flat[order] * p_flat.size / np.arange(1, p_flat.size + 1)
        )
        adjusted_sorted = np.minimum.accumulate(adjusted_sorted[::-1])[::-1]
        fdr_flat = np.empty_like(p_flat)
        fdr_flat[order] = np.minimum(adjusted_sorted, 1.0)
        fdr = fdr_flat.reshape(T, T)
        sig_mask = fdr < 0.05
    
    # Distance stratification
    distance_panels = None
    if spatial_coord is not None and w.size > 0:
        pair_dist = np.linalg.norm(spatial_coord[recv] - spatial_coord[send], axis=1)
        if distance_bins == "quartile":
            bins = np.quantile(pair_dist, [0, 0.25, 0.5, 0.75, 1.0])
        elif isinstance(distance_bins, (list, tuple, np.ndarray)):
            bins = np.asarray(distance_bins, dtype=float)
            if not np.all(np.diff(bins) > 0):
                raise ValueError("distance_bins must be strictly increasing")
        else:
            bins = None
        
        if bins is not None:
            # ``bins`` are interval boundaries, so K boundaries define K-1
            # panels.  Searching only the interior boundaries includes both
            # the global minimum and maximum exactly once.
            bin_id = np.searchsorted(bins[1:-1], pair_dist, side="right")
            Mps_list = []
            for b in range(len(bins) - 1):
                m = bin_id == b
                if m.sum() > 0:
                    Ms = coo_matrix((w[m], (type_id[send[m]], type_id[recv[m]])), shape=(T, T)).toarray()
                else:
                    Ms = np.zeros((T, T))
                Mps = Ms / n_per_type[:, None]
                Mps_list.append(Mps)
            distance_panels = {"bins": bins, "M_per_source_bybin": Mps_list}
    
    # Plotting
    if plot is not None:
        show_plots = bool(plot)

    if show_plots:
        try:
            import seaborn as sns
            import matplotlib.pyplot as plt
            
            sns.set_theme(
                style="whitegrid",
                font_scale=1.1,
                rc={"axes.facecolor": "white", "figure.facecolor": "white"},
            )
            
            cmap_main = sns.color_palette("PuBu", as_cmap=True)
            cmap_asym = sns.diverging_palette(250, 10, as_cmap=True)
            title_prefix = f" (time={time_title})" if time_title is not None else ""
            
            plt.figure(figsize=(6.8, 5.6))
            ax = sns.heatmap(
                M_per_source,
                xticklabels=types,
                yticklabels=types,
                cmap=cmap_main,
                square=True,
                cbar_kws={"label": "Attention (A→B)"},
                linewidths=0.4,
                linecolor="white",
            )
            plt.title(f"Per-source-cell attention A→B{title_prefix}")
            plt.xlabel("Receiver type (B)")
            plt.ylabel("Sender type (A)")
            if sig_mask is not None:
                yy, xx = np.where(sig_mask)
                for i, j in zip(yy, xx):
                    ax.text(j + 0.5, i + 0.5, "•", ha="center", va="center", fontsize=9, color="k")
                plt.suptitle("• : FDR < 0.05", y=1.02, fontsize=10)
            plt.tight_layout()
            plt.show()
            
            plt.figure(figsize=(6.8, 5.6))
            sns.heatmap(
                asym,
                xticklabels=types,
                yticklabels=types,
                cmap=cmap_asym,
                center=0,
                square=True,
                linewidths=0.4,
                linecolor="white",
                cbar_kws={"label": "Asymmetry (A→B minus B→A)"},
            )
            plt.title(f"Directional asymmetry (per-source normalized){title_prefix}")
            plt.xlabel("B")
            plt.ylabel("A")
            plt.tight_layout()
            plt.show()
        except ImportError:
            pass  # Skip plotting if seaborn not available
    
    result = {
        "types": types,
        "M_sum": M_sum,
        "M_per_source": M_per_source,
        "M_row": M_row,
        "M_mean": M_mean,
        "asym": asym,
        "type_stats": type_stats,
        "edge_counts": edge_counts,
    }
    if fdr is not None:
        result["fdr"] = fdr
        result["sig_mask"] = sig_mask
    if distance_panels is not None:
        result["distance_bins"] = distance_panels["bins"]
        result["M_per_source_bybin"] = distance_panels["M_per_source_bybin"]
    
    return result
