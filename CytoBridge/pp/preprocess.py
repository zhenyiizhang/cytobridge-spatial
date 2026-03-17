import scanpy as sc
from anndata import AnnData
import pandas as pd
import numpy as np
import logging
import re
from typing import Optional, Dict

_TIME_PATTERN = re.compile(r"-?\d+(?:\.\d+)?")


def _auto_time_order(unique_times) -> list:
    """Robust auto ordering for mixed/string time labels."""
    unique_times = list(unique_times)
    series = pd.Series(unique_times)
    if pd.api.types.is_numeric_dtype(series):
        return sorted(unique_times, key=float)

    parsed = []
    for idx, value in enumerate(unique_times):
        match = _TIME_PATTERN.search(str(value))
        parsed.append((float(match.group()) if match else None, idx, value))

    # If every label has an embedded number (e.g. 24hpf, E10.5), sort by that.
    if all(item[0] is not None for item in parsed):
        parsed.sort(key=lambda x: (x[0], x[1]))
        return [item[2] for item in parsed]

    # Fallback to observed order to avoid lexicographic mis-ordering.
    return unique_times


def preprocess(
    adata: AnnData,
    time_key: str,
    n_top_genes: int = 2000,
    dim_reduction: str = 'pca',
    n_pcs: int = 50,
    time_mapping: Optional[Dict[str, float]] = None,
    normalization: bool = True,
    log1p: bool = True,
    select_hvg: bool = True,
) -> AnnData:
    """
    Preprocess step for dynamical optimal transport analysis.

    This function implements a standard preprocessing workflow. Key steps include:
    1. Mapping categorical time points to a numerical scale.
    2. Normalizing and log-transforming the data.
    3. Selecting highly variable genes (HVGs).
    4. Scaling the data and performing PCA.

    Parameters
    ----------
    adata
        The annotated data matrix.
    time_key
        The key in `adata.obs` that specifies the time point of each cell.
    n_top_genes
        Number of highly variable genes to select.
    dim_reduction
        The dimension reduction method to use.
    n_pcs
        Number of principal components to compute.
    time_mapping
        A dictionary to map non-numeric time points to numeric values.
        Example: {'Day0': 0, 'Day2': 2.0, 'Day7': 7.0}.
        If None (default), time points are automatically sorted and mapped to
        0, 1, 2, ...
    normalization
        If True, normalize the data.
    log1p
        If True, apply log1p transformation to the data.
    select_hvg
        If True, select highly variable genes.
    Returns
    -------
    AnnData
        Processed AnnData used for training/downstream.
        `adata.X` is kept as gene-level expression matrix (after optional normalization/log1p/HVG).
        Reduced representation is stored in `adata.obsm['X_latent']`.
    """
    # --- Input Validation and Setup ---
    if time_key not in adata.obs.keys():
        raise KeyError(
            f"The specified time_key '{time_key}' was not found in adata.obs. "
            f"Available keys are: {list(adata.obs.keys())}"
        )
    print(f"Using '{time_key}' as the time point identifier.")

    # --- Time Point Mapping Logic ---
    unique_times = adata.obs[time_key].unique()

    # The processed time key is current hard-coded for convenience
    time_key_added = 'time_point_processed'
    if time_mapping:
        print(f"Using user-provided time mapping.")
        # Check if all time points in data are present in the mapping
        missing_keys = [t for t in unique_times if t not in time_mapping]
        if missing_keys:
            raise ValueError(
                f"The following time points in adata.obs['{time_key}'] are "
                f"not present in the provided time_mapping: {missing_keys}"
            )
        # Apply the user-defined mapping
        adata.obs[time_key_added] = adata.obs[time_key].map(time_mapping).astype(float)

    else:
        print("No time mapping provided. Generating automatic mapping.")
        # Automatically map time points with robust numeric-aware ordering.
        sorted_times = _auto_time_order(unique_times)
        auto_mapping = {time_point: i for i, time_point in enumerate(sorted_times)}
        print(f"Automatically generated time mapping: {auto_mapping}")
        # Apply the automatic mapping
        adata.obs[time_key_added] = adata.obs[time_key].map(auto_mapping).astype(float)

    print(f"Numerical time points stored in `adata.obs['{time_key_added}']`.")

    # --- Standard Preprocessing Steps ---
    # Keep raw counts for downstream modules that need gene-space count values
    # (e.g., ligand-receptor interaction graph construction).
    if "counts" not in adata.layers:
        adata.layers["counts"] = adata.X.copy()

    if normalization:
        print("Normalizing total counts and applying log1p transformation.")
        sc.pp.normalize_total(adata, target_sum=1e4)

    if log1p:
        sc.pp.log1p(adata)

    # Record the HVG mask (for later storage in original_gene_info)
    hvg_mask = None
    if select_hvg:
        print(f"Selecting top {n_top_genes} highly variable genes.")
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=n_top_genes,
        )
        hvg_mask = adata.var.highly_variable.copy()
        print(f"HVG marked: {int(np.sum(hvg_mask.values))} genes (no subsetting of adata.X).")

    # --- Dimension Reduction ---
    # ---------- : PCA | UMAP | none ----------
    if dim_reduction.lower() == 'pca':
        sc.pp.pca(
            adata,
            n_comps=n_pcs,
            svd_solver='arpack',
            use_highly_variable=bool(select_hvg),
        )
        adata.obsm['X_latent'] = np.asarray(adata.obsm['X_pca'], dtype=np.float32)

    elif dim_reduction.lower() == 'umap':
        sc.pp.pca(
            adata,
            n_comps=n_pcs,
            svd_solver='arpack',
            use_highly_variable=bool(select_hvg),
        )
        sc.pp.neighbors(adata, n_pcs=n_pcs)
        sc.tl.umap(adata)
        adata.obsm['X_latent'] = np.asarray(adata.obsm['X_umap'], dtype=np.float32)

    elif dim_reduction.lower() == 'none' or dim_reduction is None:
        # Convert to dense float array so downstream torch.tensor(...) is safe.
        if hasattr(adata.X, "toarray"):
            adata.obsm['X_latent'] = adata.X.toarray().astype(np.float32)
        else:
            adata.obsm['X_latent'] = np.asarray(adata.X, dtype=np.float32)
        print("Dimension reduction set to 'none'.")

    else:
        raise ValueError(f"Invalid dimension reduction method: {dim_reduction}")

    # Store preprocessing provenance in-place.
    preprocess_info = {
        'time_key': time_key,
        'time_key_added': time_key_added,
        'normalization': bool(normalization),
        'log1p': bool(log1p),
        'select_hvg': bool(select_hvg),
        'n_top_genes': int(n_top_genes),
        'dim_reduction': str(dim_reduction).lower() if dim_reduction is not None else 'none',
        'n_pcs': int(n_pcs),
        'hvg_for_latent_only': bool(select_hvg),
        'x_representation': 'gene_expression',
        'latent_key': 'X_latent',
        'counts_layer': 'counts',
    }
    adata.uns['preprocess_info'] = preprocess_info
    original_gene_info = {
        'var': adata.var.copy(deep=True),
        'var_names': adata.var_names.tolist(),
        'X_shape': np.array(adata.X.shape),
        'select_hvg': select_hvg,
    }
    if hvg_mask is not None:
        original_gene_info['hvg_mask'] = hvg_mask.values.astype(bool)
    adata.uns['original_gene_info'] = original_gene_info

    print(
        "Preprocess output: "
        f"X(gene) shape={adata.X.shape}, X_latent shape={adata.obsm['X_latent'].shape}"
    )

    print("Preprocessing recipe finished.")
    return adata
