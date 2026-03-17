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


def save_interpolated_attention(
    adata: "ad.AnnData",
    time_value: float,
    model,
    device: str = "cpu",
    out_dir: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """Compute and save attention weights at an interpolated time point.
    
    Parameters
    ----------
    adata : AnnData
        AnnData with model input features (prefers obsm['X_latent'] and optional obsm['spatial_aligned']).
    time_value : float
        Time point for attention computation.
    model : DynamicalModel
        Trained CytoBridge DynamicalModel with interaction_net.
    device : str
        Device for computation.
    out_dir : str, optional
        Output directory for saving arrays. If None, uses current directory.
        
    Returns
    -------
    dict
        Dictionary with keys: 'attn_matrix', 'attn_mean', 'edge_index'.
    """
    import os
    import torch
    
    # Extract interaction network from model (DynamicalModel expected).
    interaction_net = getattr(model, "interaction_net", None)
    if interaction_net is None and hasattr(model, "ode_func"):
        f_net = model.ode_func
        interaction_net = getattr(f_net, "interaction_net", None)
    
    if interaction_net is None:
        raise ValueError("Model does not have an interaction_net component")
    
    interaction_net = interaction_net.to(device)
    model_input = _prepare_model_input_from_adata(adata)
    data = torch.tensor(model_input, dtype=torch.float32, device=device)
    n_particles = data.shape[0]
    lnw0 = torch.log(torch.ones(n_particles, 1, device=device) / n_particles)
    time_tensor = torch.tensor(time_value, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        _ = interaction_net(data, lnw0, time_tensor, return_attn=True)
        attn = interaction_net.gnn_layers[0].attn
    
    attn = torch.abs(attn)
    attn_mean = attn.mean(dim=1).cpu().numpy()
    if not hasattr(interaction_net, "edge_index") or interaction_net.edge_index is None:
        raise AttributeError(
            "interaction_net.edge_index is missing. "
            "Ensure GNNInteraction caches edge_index during forward."
        )
    edge_index = interaction_net.edge_index.detach().cpu().numpy()
    # Keep attn/edge alignment when removing self-loops (bug fix; edge_index is usually self-loop free already).
    m = edge_index[0] != edge_index[1]
    edge_index = edge_index[:, m]
    attn_mean = attn_mean[m]
    
    attn_matrix = np.zeros((n_particles, n_particles))
    attn_matrix[edge_index[0], edge_index[1]] = attn_mean
    
    if out_dir is None:
        out_dir = os.getcwd()
    os.makedirs(out_dir, exist_ok=True)
    
    np.save(os.path.join(out_dir, f"attn_interp_t{time_value}.npy"), attn_matrix)
    np.save(os.path.join(out_dir, f"attn_mean_interp_t{time_value}.npy"), attn_mean)
    np.save(os.path.join(out_dir, f"edge_index_interp_t{time_value}.npy"), edge_index)
    
    return {"attn_matrix": attn_matrix, "attn_mean": attn_mean, "edge_index": edge_index}


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
) -> Dict:
    """Aggregate attention weights at cell-type level with optional distance stratification.
    
    This function analyzes attention patterns between cell types, computing
    various aggregation metrics and optionally performing permutation testing.
    
    Parameters
    ----------
    edge_index : np.ndarray
        Edge indices of shape (2, n_edges).
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
    
    if winsor_quantile is not None and 0.9 < winsor_quantile < 1.0:
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
        rank = np.empty_like(order)
        rank[order] = np.arange(1, p_flat.size + 1)
        fdr_flat = p_flat * p_flat.size / np.maximum(rank, 1)
        fdr = fdr_flat.reshape(T, T)
        sig_mask = fdr < 0.05
    
    # Distance stratification
    distance_panels = None
    if spatial_coord is not None:
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
            bin_id = np.digitize(pair_dist, bins, right=True)
            Mps_list = []
            for b in range(1, len(bins) + 1):
                m = bin_id == b
                if m.sum() > 0:
                    Ms = coo_matrix((w[m], (type_id[send[m]], type_id[recv[m]])), shape=(T, T)).toarray()
                else:
                    Ms = np.zeros((T, T))
                Mps = Ms / n_per_type[:, None]
                Mps_list.append(Mps)
            distance_panels = {"bins": bins, "M_per_source_bybin": Mps_list}
    
    # Plotting
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
