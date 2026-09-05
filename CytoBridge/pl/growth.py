"""Growth-rate visualization.

This module provides functions for plotting growth-rate (g-value) visualizations
and gene expression trends.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd
    import anndata as ad

__all__ = [
    "plot_g_values",
    "plot_growth_per_time",
    "plot_growth_per_time_from_adata",
    "gene_velocity_embeddings",
    "gene_velocity_embeddings_from_adata",
    "plot_gene_expression_trends",
    "plot_cell_counts_over_time",
    "plot_growth_interaction_bubble",
    "plot_growth_timepoint_grid",
]


def plot_growth_timepoint_grid(
    adata_dict,
    *,
    time_points,
    time_keys=None,
    out_path: str,
    value_key: str = "growth_rate",
    spatial_key: str = "spatial",
    source_by_time: dict | None = None,
    n_cols: int = 3,
    cmap: str = "viridis",
    point_size: float = 2.0,
    lower_quantile: float = 0.05,
    upper_quantile: float = 0.95,
    scale_mode: str = "panel_limits",
    shared_colorbar: bool = False,
    colorbar_label: str | None = None,
    title: str | None = None,
    show: bool = False,
):
    """Plot spatial growth maps on an observed/interpolated time grid.

    ``scale_mode='panel_limits'`` preserves the historical behavior: raw values
    are shown with independent robust limits in each panel.  ``'per_time_0_1'``
    robust-scales every time point to 0--1 before plotting, while
    ``'global_limits'`` uses one pair of robust raw-value limits across all
    panels.  The latter two modes can use one genuinely shared colorbar.
    Set ``show=True`` to also display the calculated plot in a notebook.
    """
    from pathlib import Path
    import math

    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    times = list(time_points)
    keys = [str(value) for value in times] if time_keys is None else list(time_keys)
    if len(times) != len(keys):
        raise ValueError("time_points and time_keys must have the same length.")
    if int(n_cols) <= 0:
        raise ValueError("n_cols must be positive.")
    scale_mode = str(scale_mode).strip().lower()
    allowed_scale_modes = {"panel_limits", "per_time_0_1", "global_limits"}
    if scale_mode not in allowed_scale_modes:
        raise ValueError(
            f"scale_mode must be one of {sorted(allowed_scale_modes)}, "
            f"got {scale_mode!r}."
        )
    if not (0.0 <= float(lower_quantile) < float(upper_quantile) <= 1.0):
        raise ValueError(
            "lower_quantile and upper_quantile must satisfy "
            "0 <= lower < upper <= 1."
        )
    if bool(shared_colorbar) and scale_mode == "panel_limits" and len(times) > 1:
        raise ValueError(
            "shared_colorbar=True is incompatible with per-panel raw limits; "
            "use scale_mode='per_time_0_1' or 'global_limits'."
        )

    records = []
    for time_value, key in zip(times, keys):
        if key not in adata_dict:
            raise KeyError(f"Missing timepoint key in adata_dict: {key!r}.")
        adata_t = adata_dict[key]
        if value_key not in adata_t.obs:
            raise KeyError(f"adata_dict[{key!r}].obs is missing {value_key!r}.")
        values = np.asarray(adata_t.obs[value_key], dtype=float).reshape(-1)
        if values.size == 0 or not np.isfinite(values).all():
            raise ValueError(
                f"adata_dict[{key!r}].obs[{value_key!r}] must be non-empty and finite."
            )
        if spatial_key in adata_t.obsm:
            coords = np.asarray(adata_t.obsm[spatial_key], dtype=float)
        else:
            matrix = (
                adata_t.X.toarray()
                if hasattr(adata_t.X, "toarray")
                else np.asarray(adata_t.X)
            )
            coords = np.asarray(matrix, dtype=float)[:, :2]
        if (
            coords.ndim != 2
            or coords.shape[0] != values.shape[0]
            or coords.shape[1] < 2
        ):
            raise ValueError(
                f"Spatial coordinates for {key!r} must have shape "
                f"({values.shape[0]}, D) with D>=2, got {coords.shape}."
            )
        plot_coords = coords[:, :2]
        if not np.isfinite(plot_coords).all():
            raise ValueError(f"Spatial coordinates for {key!r} contain non-finite values.")
        records.append((time_value, key, plot_coords, values))

    global_limits = None
    if scale_mode == "global_limits":
        all_values = np.concatenate([record[3] for record in records])
        global_limits = tuple(
            float(value)
            for value in np.quantile(
                all_values, [float(lower_quantile), float(upper_quantile)]
            )
        )
        if np.isclose(global_limits[0], global_limits[1]):
            global_limits = (global_limits[0], global_limits[0] + 1e-8)

    n_rows = int(math.ceil(len(times) / int(n_cols)))
    with plt.rc_context(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "text.color": "black",
            "axes.labelcolor": "black",
        }
    ):
        fig, axes = plt.subplots(
            n_rows,
            int(n_cols),
            figsize=(4.2 * int(n_cols), 4.2 * n_rows),
            squeeze=False,
            dpi=300,
        )
        for ax in axes.flat:
            ax.axis("off")
        scatter = None
        for ax, (time_value, key, coords, raw_values) in zip(axes.flat, records):
            if scale_mode == "per_time_0_1":
                low, high = np.quantile(
                    raw_values, [float(lower_quantile), float(upper_quantile)]
                )
                values = np.clip(
                    (raw_values - float(low)) / max(float(high - low), 1e-8),
                    0.0,
                    1.0,
                )
                vmin, vmax = 0.0, 1.0
            elif scale_mode == "global_limits":
                values = raw_values
                assert global_limits is not None
                vmin, vmax = global_limits
            else:
                values = raw_values
                vmin, vmax = (
                    float(value)
                    for value in np.quantile(
                        raw_values,
                        [float(lower_quantile), float(upper_quantile)],
                    )
                )
                if np.isclose(vmin, vmax):
                    vmax = vmin + 1e-8
            scatter = ax.scatter(
                coords[:, 0],
                coords[:, 1],
                c=values,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                s=float(point_size),
                linewidths=0,
                alpha=0.85,
            )
            source = source_by_time.get(time_value) if source_by_time else None
            suffix = f" ({source})" if source else ""
            ax.set_title(f"t={float(time_value):g}{suffix}", fontsize=10)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.axis("on")
            for spine in ax.spines.values():
                spine.set_visible(False)
            if not shared_colorbar:
                colorbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.02)
                colorbar.ax.tick_params(labelsize=7)
                if colorbar_label:
                    colorbar.set_label(str(colorbar_label), fontsize=8)
        if shared_colorbar and scatter is not None:
            if scale_mode == "per_time_0_1":
                shared_limits = (0.0, 1.0)
            else:
                assert global_limits is not None
                shared_limits = global_limits
            colorbar = fig.colorbar(
                ScalarMappable(
                    norm=Normalize(vmin=shared_limits[0], vmax=shared_limits[1]),
                    cmap=cmap,
                ),
                ax=list(axes.flat[: len(records)]),
                fraction=0.025,
                pad=0.02,
            )
            colorbar.ax.tick_params(labelsize=7)
            colorbar.set_label(
                str(colorbar_label or value_key),
                fontsize=8,
            )
        if title:
            fig.suptitle(title, fontsize=13)
        if shared_colorbar:
            fig.subplots_adjust(
                left=0.04,
                right=0.9,
                bottom=0.04,
                top=0.94 if title else 0.98,
                wspace=0.08,
                hspace=0.16,
            )
        else:
            fig.tight_layout()
        path = Path(out_path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight", facecolor="white")
        if show:
            plt.show()
        plt.close(fig)
    return path


def plot_growth_interaction_bubble(
    grouped,
    *,
    out_path: str,
    min_dot_size: float = 20.0,
    max_dot_size: float = 400.0,
    cmap: str = "plasma",
):
    """Plot one growth/interaction bubble per ``(time, celltype)`` group."""
    from pathlib import Path

    import matplotlib.pyplot as plt
    import pandas as pd

    table = pd.DataFrame(grouped).copy()
    required = {"time", "celltype", "growth_mean", "interaction_mean", "n"}
    missing = sorted(required.difference(table.columns))
    if missing:
        raise KeyError(f"grouped table is missing columns: {missing}")
    if table.empty:
        raise ValueError("grouped table must be non-empty.")

    time_values = sorted(table["time"].astype(float).unique())
    time_to_index = {value: index for index, value in enumerate(time_values)}
    colors = table["time"].astype(float).map(time_to_index)
    sizes = np.clip(
        table["n"].to_numpy(dtype=float),
        float(min_dot_size),
        float(max_dot_size),
    )

    with plt.rc_context(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "text.color": "black",
            "axes.labelcolor": "black",
            "axes.edgecolor": "black",
        }
    ):
        fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=300)
        scatter = ax.scatter(
            table["interaction_mean"],
            table["growth_mean"],
            s=sizes,
            c=colors,
            cmap=cmap,
            alpha=0.82,
            edgecolors="white",
            linewidths=0.35,
        )
        ax.set_xlabel("Mean interaction magnitude")
        ax.set_ylabel("Mean growth (g)")
        ax.grid(False)
        colorbar = fig.colorbar(scatter, ax=ax, pad=0.02)
        colorbar.set_label("Time index")
        fig.tight_layout()
        path = Path(out_path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    return path


def _coerce_feature_matrix_from_adata(
    adata,
    *,
    obsm_key: str,
    spatial_key: str,
    concat_spatial: Optional[bool],
) -> tuple[np.ndarray, int]:
    if obsm_key in adata.obsm:
        latent = np.asarray(adata.obsm[obsm_key], dtype=np.float32)
    else:
        latent = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
        latent = np.asarray(latent, dtype=np.float32)

    use_spatial = bool(concat_spatial) if concat_spatial is not None else (spatial_key in adata.obsm)
    if not use_spatial:
        return latent, 0

    if spatial_key not in adata.obsm:
        raise KeyError(f"concat_spatial=True but adata.obsm['{spatial_key}'] is missing.")
    spatial = np.asarray(adata.obsm[spatial_key], dtype=np.float32)
    if spatial.shape[0] != latent.shape[0]:
        raise ValueError(
            f"Row mismatch between '{spatial_key}' ({spatial.shape[0]}) and "
            f"'{obsm_key}' ({latent.shape[0]})."
        )
    return np.hstack((spatial, latent)).astype(np.float32), int(spatial.shape[1])


def _prepare_growth_arrays_from_adata(
    adata,
    *,
    dim: int,
    time_key: Optional[str],
    obsm_key: str,
    spatial_key: str,
    concat_spatial: Optional[bool],
) -> tuple[np.ndarray, np.ndarray]:
    from CytoBridge.tl.downstream.downstream_data import infer_time_key, parse_time_value

    if not hasattr(adata, "obs") or not hasattr(adata, "obsm"):
        raise TypeError(f"Expected AnnData-like object, got {type(adata)}")

    resolved_time_key = infer_time_key(adata.obs, preferred=time_key)
    times = np.asarray([parse_time_value(v) for v in adata.obs[resolved_time_key].values], dtype=np.float64)
    X, _ = _coerce_feature_matrix_from_adata(
        adata,
        obsm_key=obsm_key,
        spatial_key=spatial_key,
        concat_spatial=concat_spatial,
    )
    if int(dim) > X.shape[1]:
        raise ValueError(f"Requested dim={dim}, but feature matrix has dim={X.shape[1]}.")
    return X[:, : int(dim)], times


def plot_g_values(
    df: "pd.DataFrame",
    dim: int,
    model,
    time_index: int = 0,
    dim_reducer=None,
    device: str = "cpu",
    out_path: Optional[str] = None,
    figsize: tuple = (6, 6),
    point_size: float = 0.5,
    alpha: float = 0.7,
    cmap: str = "rainbow",
    samples_column: str = "samples",
):
    """Plot growth-rate g-values as a scatter plot.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with samples column and feature columns.
    dim : int
        Number of feature dimensions.
    model : DynamicalModel
        Trained CytoBridge model with g_net.
    time_index : int
        Index of time point to visualize.
    dim_reducer : object, optional
        Dimensionality reducer with transform method.
    device : str
        Device for computation.
    out_path : str, optional
        Path to save figure.
    figsize : tuple
        Figure size.
    point_size : float
        Scatter point size.
    alpha : float
        Point transparency.
    cmap : str
        Colormap name.
    samples_column : str
        Name of time/samples column.

    Returns
    -------
    str or None
        Path to saved figure if out_path provided.
    """
    import matplotlib.pyplot as plt
    import torch

    from CytoBridge.tl.downstream.downstream_data import infer_feature_columns

    time_points = sorted(df[samples_column].unique())
    if time_index < 0 or time_index >= len(time_points):
        raise ValueError(f"time_index={time_index} out of range [0, {len(time_points)-1}]")

    feature_cols = list(infer_feature_columns(df=df, samples_column=samples_column))[:dim]
    if len(feature_cols) < dim:
        raise ValueError(f"Requested dim={dim}, found only {len(feature_cols)} feature columns")

    if "growth" not in getattr(model, "components", []):
        raise ValueError("Model does not have growth_net component")
    model = model.to(device)

    data_by_time = {}
    for time in [time_points[time_index]]:
        subset = df[df[samples_column] == time]
        data = torch.tensor(subset[feature_cols].values, dtype=torch.float32, device=device)
        t_expand = torch.full((data.shape[0], 1), float(time), dtype=torch.float32, device=device)
        with torch.no_grad():
            g_values = model.predict_growth(t=t_expand, x=data)
        data_by_time[time] = {"data": subset, "g_values": g_values.detach().cpu().numpy().flatten()}

    all_g_values = np.concatenate([content["g_values"] for content in data_by_time.values()])
    vmax_value = np.percentile(all_g_values, 95)
    norm = plt.Normalize(vmin=0, vmax=vmax_value, clip=True)

    fig, ax = plt.subplots(figsize=figsize)
    for time, content in data_by_time.items():
        subset = content["data"]
        g_vals = content["g_values"]
        new_data = subset[feature_cols]
        if dim_reducer is not None:
            data_reduced = dim_reducer.transform(new_data)
        else:
            data_reduced = new_data.iloc[:, :2].values
        x = data_reduced[:, 0]
        y = data_reduced[:, 1]
        colors = plt.cm.get_cmap(cmap)(norm(g_vals))
        ax.scatter(x, y, c=colors, label=f"Time {time}", s=point_size, alpha=alpha, marker="o")

    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.legend()

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array(all_g_values)
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Predicted growth rate")

    if out_path:
        fig.savefig(out_path, format="pdf" if out_path.endswith(".pdf") else "png", bbox_inches="tight")

    plt.show()
    return out_path


def plot_growth_per_time(
    df: "pd.DataFrame",
    dim: int,
    model,
    out_dir: str,
    device: str = "cpu",
    point_size: float = 0.6,
    point_alpha: float = 0.9,
    samples_column: str = "samples",
    cmap: str = "rainbow",
):
    """Plot growth-rate scatter for each time point (Nature-style).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with samples column and feature columns.
    dim : int
        Number of feature dimensions.
    model : DynamicalModel
        Trained CytoBridge model.
    out_dir : str
        Output directory for figures.
    device : str
        Device for computation.
    point_size : float
        Scatter point size.
    point_alpha : float
        Scatter alpha.
    samples_column : str
        Name of time/samples column.
    cmap : str
        Colormap name.

    Returns
    -------
    list
        List of output file paths.
    """
    import os
    import matplotlib.pyplot as plt
    import torch
    from CytoBridge.tl.downstream.downstream_data import infer_feature_columns

    os.makedirs(out_dir, exist_ok=True)

    if "growth" not in getattr(model, "components", []):
        raise ValueError("Model does not have growth_net component")
    model = model.to(device)

    time_points = sorted(df[samples_column].unique())
    feature_cols = list(infer_feature_columns(df=df, samples_column=samples_column))[:dim]
    if len(feature_cols) < dim:
        raise ValueError(f"Requested dim={dim}, found only {len(feature_cols)} feature columns")
    if len(feature_cols) < 2:
        raise ValueError("At least 2 feature columns are required for spatial growth plotting")
    output_paths = []

    for t_idx, t_val in enumerate(time_points):
        subset = df[df[samples_column] == t_val]
        data = torch.tensor(subset[feature_cols].values, dtype=torch.float32, device=device)
        t_expand = torch.full((data.shape[0], 1), float(t_val), dtype=torch.float32, device=device)
        with torch.no_grad():
            g_vals = model.predict_growth(t=t_expand, x=data).detach().cpu().numpy().squeeze()

        # ST-1104 style: robust 5-95 scaling per timepoint
        g_min, g_max = np.percentile(g_vals, [5, 95])
        g_scaled = np.clip((g_vals - g_min) / max(g_max - g_min, 1e-6), 0, 1)

        fig, ax = plt.subplots(figsize=(5, 5), dpi=300)
        scatt = ax.scatter(
            subset[feature_cols[0]].values,
            subset[feature_cols[1]].values,
            c=g_scaled,
            s=point_size,
            cmap="RdYlBu_r",
            linewidths=0,
            alpha=float(point_alpha),
        )
        ax.set_title(f"Growth g (t={t_val})", fontsize=10)
        ax.axis("off")
        ax.set_aspect("equal")
        cbar = plt.colorbar(scatt, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("g (scaled 5-95%)")
        out_path = os.path.join(out_dir, f"growth_t{t_idx}.pdf")
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        output_paths.append(out_path)

    return output_paths


def plot_growth_per_time_from_adata(
    adata: "ad.AnnData",
    dim: int,
    model,
    out_dir: str,
    device: str = "cpu",
    point_size: float = 0.6,
    point_alpha: float = 0.9,
    time_key: Optional[str] = None,
    obsm_key: str = "X_latent",
    spatial_key: str = "spatial_aligned",
    concat_spatial: Optional[bool] = None,
):
    """AnnData-first growth plotting wrapper."""
    import os
    import matplotlib.pyplot as plt
    import torch

    os.makedirs(out_dir, exist_ok=True)
    if "growth" not in getattr(model, "components", []):
        raise ValueError("Model does not have growth_net component")
    model = model.to(device)

    X, times = _prepare_growth_arrays_from_adata(
        adata,
        dim=dim,
        time_key=time_key,
        obsm_key=obsm_key,
        spatial_key=spatial_key,
        concat_spatial=concat_spatial,
    )
    if X.shape[1] < 2:
        raise ValueError("At least 2 feature dimensions are required for spatial growth plotting")

    output_paths = []
    time_points = sorted(np.unique(times).tolist())
    for t_idx, t_val in enumerate(time_points):
        mask = np.isclose(times, float(t_val), rtol=0.0, atol=1e-9)
        if not np.any(mask):
            continue
        data = torch.tensor(X[mask], dtype=torch.float32, device=device)
        t_expand = torch.full((data.shape[0], 1), float(t_val), dtype=torch.float32, device=device)
        with torch.no_grad():
            g_vals = model.predict_growth(t=t_expand, x=data).detach().cpu().numpy().squeeze()

        g_min, g_max = np.percentile(g_vals, [5, 95])
        g_scaled = np.clip((g_vals - g_min) / max(g_max - g_min, 1e-6), 0, 1)

        fig, ax = plt.subplots(figsize=(5, 5), dpi=300)
        scatt = ax.scatter(
            X[mask, 0],
            X[mask, 1],
            c=g_scaled,
            s=point_size,
            cmap="RdYlBu_r",
            linewidths=0,
            alpha=float(point_alpha),
        )
        ax.set_title(f"Growth g (t={t_val})", fontsize=10)
        ax.axis("off")
        ax.set_aspect("equal")
        cbar = plt.colorbar(scatt, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("g (scaled 5-95%)")
        out_path = os.path.join(out_dir, f"growth_t{t_idx}.pdf")
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        output_paths.append(out_path)

    return output_paths


def gene_velocity_embeddings(
    df: "pd.DataFrame",
    dim: int,
    model,
    out_dir: str,
    label_to_color: Optional[Dict[str, str]] = None,
    keep_cell_types: Optional[Sequence[str]] = None,
    device: str = "cpu",
    samples_column: str = "samples",
    annotation_column: str = "Annotation",
) -> list[str]:
    """Compute full velocity and plot gene-velocity streamlines in UMAP/PCA (ST-1104 style).

    Notes
    -----
    - Uses the full velocity decomposition: drift + interaction + score_gradient.
    - Builds an AnnData in gene space (drops the first 2 spatial dims).
    """
    import os

    import anndata as ad
    import numpy as np
    import scanpy as sc
    import scvelo as scv

    from CytoBridge.tl import compute_velocity_components, compute_umap_embedding
    from CytoBridge.tl.downstream.downstream_data import infer_feature_columns

    os.makedirs(out_dir, exist_ok=True)

    feature_cols = list(infer_feature_columns(df=df, samples_column=samples_column))[:dim]
    if len(feature_cols) < dim:
        raise ValueError(f"Requested dim={dim}, found only {len(feature_cols)} feature columns")
    if len(feature_cols) < 3:
        raise ValueError("Need at least 3 feature dimensions to compute gene velocity embeddings")

    all_data = df[feature_cols].values
    gene_data = np.nan_to_num(all_data[:, 2:], nan=0.0, posinf=0.0, neginf=0.0)

    vel_full_all = np.zeros_like(all_data, dtype=np.float32)
    for t in sorted(df[samples_column].unique()):
        mask = df[samples_column] == t
        data_t = all_data[mask]
        vel_t = compute_velocity_components(
            data=data_t,
            time_value=float(t),
            model=model,
            device=device,
        )["full"]
        vel_full_all[mask] = vel_t

    gene_vel_full = np.nan_to_num(vel_full_all[:, 2:], nan=0.0, posinf=0.0, neginf=0.0)

    umap_coords, _ = compute_umap_embedding(gene_data, n_neighbors=30, min_dist=0.3, seed=0)
    if not np.isfinite(umap_coords).all():
        umap_coords = np.nan_to_num(umap_coords, nan=0.0, posinf=0.0, neginf=0.0)

    adata = ad.AnnData(X=gene_data)
    adata.layers["spliced"] = gene_data
    adata.layers["Ms"] = gene_data
    adata.layers["velocity"] = gene_vel_full
    adata.obsm["X_umap"] = umap_coords

    sc.tl.pca(adata, n_comps=2, svd_solver="arpack")
    if "X_pca" in adata.obsm and not np.isfinite(adata.obsm["X_pca"]).all():
        adata.obsm["X_pca"] = np.nan_to_num(adata.obsm["X_pca"], nan=0.0, posinf=0.0, neginf=0.0)

    adata.obs["timepoint"] = df[samples_column].astype(str).values
    adata.obs["time"] = df[samples_column].astype(float).values
    if annotation_column in df.columns:
        adata.obs["cell_type"] = df[annotation_column].astype(str).values

    if label_to_color and "cell_type" in adata.obs:
        adata.obs["cell_type"] = adata.obs["cell_type"].astype("category")
        cats = list(adata.obs["cell_type"].cat.categories)
        palette = [label_to_color.get(c, "#888888") for c in cats]
        adata.uns["cell_type_colors"] = palette

    sc.pp.neighbors(adata, n_neighbors=30, use_rep="X")
    scv.tl.velocity_graph(adata, vkey="velocity", xkey="Ms")
    scv.settings.set_figure_params("scvelo")

    plots: list[str] = []
    color_items = [("timepoint", "timepoint", "timepoint"), ("time", "time", "time")]
    if "cell_type" in adata.obs:
        color_items.append(("cell_type", "cell type", "celltype"))

    keep_set = {str(x) for x in keep_cell_types} if keep_cell_types else None

    def _prepare_celltype_plot(plot_adata: ad.AnnData) -> tuple[ad.AnnData, str]:
        if "cell_type" not in plot_adata.obs:
            return plot_adata, "cell_type"
        if not keep_set:
            return plot_adata, "cell_type"

        plot_key = "cell_type_keep"
        labels = plot_adata.obs["cell_type"].astype(str)
        plot_adata.obs[plot_key] = labels.where(labels.isin(keep_set), other="Other").astype("category")

        desired = [c for c in sorted(keep_set) if c in plot_adata.obs[plot_key].cat.categories]
        if "Other" in plot_adata.obs[plot_key].cat.categories:
            desired.append("Other")
        if desired:
            plot_adata.obs[plot_key] = plot_adata.obs[plot_key].cat.reorder_categories(desired, ordered=False)
        if label_to_color:
            palette = [label_to_color.get(c, "#888888") for c in desired]
        else:
            base = sc.pl.palettes.default_20
            palette = [base[i % len(base)] for i in range(len(desired))]
        if "Other" in desired:
            palette[desired.index("Other")] = "#D0D0D0"
        plot_adata.uns[f"{plot_key}_colors"] = palette
        return plot_adata, plot_key

    for basis, basis_label in [("umap", "UMAP"), ("pca", "PCA")]:
        scv.tl.velocity_embedding(adata, basis=basis, vkey="velocity")
        vel_key = f"velocity_{basis}"
        if vel_key in adata.obsm:
            adata.obsm[vel_key] = np.nan_to_num(adata.obsm[vel_key], nan=0.0, posinf=0.0, neginf=0.0)

        for key, title_label, fname_label in color_items:
            plot_adata = adata
            color_key = key
            if key == "cell_type":
                plot_adata, color_key = _prepare_celltype_plot(adata.copy())

            fname = os.path.join(out_dir, f"velocity_gene_full_{basis}_{fname_label}.pdf")
            ax = scv.pl.velocity_embedding_stream(
                plot_adata,
                basis=basis,
                color=color_key,
                density=2,
                figsize=(6, 6),
                title=f"Gene velocity full ({basis_label}, {title_label})",
                show=False,
                legend_loc="right",
            )
            try:
                fig = ax.figure if hasattr(ax, "figure") else None
                if fig is not None:
                    fig.savefig(fname, bbox_inches="tight")
                else:
                    import matplotlib.pyplot as plt

                    plt.savefig(fname, bbox_inches="tight")
            finally:
                import matplotlib.pyplot as plt

                plt.close("all")
            plots.append(fname)

    return plots


def gene_velocity_embeddings_from_adata(
    adata: "ad.AnnData",
    dim: int,
    model,
    out_dir: str,
    label_to_color: Optional[Dict[str, str]] = None,
    keep_cell_types: Optional[Sequence[str]] = None,
    device: str = "cpu",
    time_key: Optional[str] = None,
    obsm_key: str = "X_latent",
    spatial_key: str = "spatial_aligned",
    concat_spatial: Optional[bool] = None,
    annotation_column: str = "Annotation",
    bases: Optional[Sequence[str]] = None,
    color_keys: Optional[Sequence[str]] = None,
    celltype_on_data: bool = False,
    output_format: str = "pdf",
    reuse_velocity_if_present: bool = False,
    n_neighbors: int = 30,
) -> list[str]:
    """AnnData-first gene-velocity embedding plotter.

    The velocity graph is always built in the full expression-associated
    latent space. ``bases`` only controls the final 2-D projection(s).
    """
    import os

    import anndata as ad
    import numpy as np
    import scanpy as sc
    import scvelo as scv

    from CytoBridge.tl import compute_umap_embedding, compute_velocity_components_from_adata
    from CytoBridge.tl.downstream.downstream_data import infer_time_key, parse_time_value

    os.makedirs(out_dir, exist_ok=True)
    selected_bases = tuple(bases or ("umap", "pca"))
    unknown_bases = sorted(set(selected_bases) - {"umap", "pca"})
    if unknown_bases:
        raise ValueError(f"Unsupported gene-velocity bases: {unknown_bases}")
    selected_color_keys = tuple(
        color_keys or ("timepoint", "time", "cell_type")
    )
    unknown_color_keys = sorted(
        set(selected_color_keys) - {"timepoint", "time", "cell_type"}
    )
    if unknown_color_keys:
        raise ValueError(f"Unsupported gene-velocity color keys: {unknown_color_keys}")
    output_format = str(output_format).strip().lower().lstrip(".")
    if output_format not in {"pdf", "svg", "png"}:
        raise ValueError("output_format must be one of {'pdf', 'svg', 'png'}.")
    if int(n_neighbors) <= 0:
        raise ValueError("n_neighbors must be > 0.")

    X, _ = _coerce_feature_matrix_from_adata(
        adata,
        obsm_key=obsm_key,
        spatial_key=spatial_key,
        concat_spatial=concat_spatial,
    )
    if int(dim) > X.shape[1]:
        raise ValueError(f"Requested dim={dim}, but feature matrix has dim={X.shape[1]}.")
    X = X[:, : int(dim)]
    if X.shape[1] < 3:
        raise ValueError("Need at least 3 feature dimensions to compute gene velocity embeddings")

    comp = compute_velocity_components_from_adata(
        adata=adata,
        model=model,
        dim=int(dim),
        device=device,
        time_key=time_key,
        obsm_key=obsm_key,
        spatial_key=spatial_key,
        concat_spatial=concat_spatial,
        write_to_adata=True,
        reuse_if_present=bool(reuse_velocity_if_present),
    )
    vel_full_all = comp["full"]

    gene_data = np.nan_to_num(X[:, 2:], nan=0.0, posinf=0.0, neginf=0.0)
    gene_vel_full = np.nan_to_num(vel_full_all[:, 2:], nan=0.0, posinf=0.0, neginf=0.0)

    plot_adata = ad.AnnData(X=gene_data)
    plot_adata.layers["spliced"] = gene_data
    plot_adata.layers["Ms"] = gene_data
    plot_adata.layers["velocity"] = gene_vel_full
    if "umap" in selected_bases:
        umap_coords, _ = compute_umap_embedding(
            gene_data, n_neighbors=int(n_neighbors), min_dist=0.3, seed=0
        )
        if not np.isfinite(umap_coords).all():
            umap_coords = np.nan_to_num(
                umap_coords, nan=0.0, posinf=0.0, neginf=0.0
            )
        plot_adata.obsm["X_umap"] = umap_coords

    if "pca" in selected_bases:
        sc.tl.pca(plot_adata, n_comps=2, svd_solver="arpack")
        if "X_pca" in plot_adata.obsm and not np.isfinite(
            plot_adata.obsm["X_pca"]
        ).all():
            plot_adata.obsm["X_pca"] = np.nan_to_num(
                plot_adata.obsm["X_pca"], nan=0.0, posinf=0.0, neginf=0.0
            )

    resolved_time_key = infer_time_key(adata.obs, preferred=time_key)
    plot_adata.obs["timepoint"] = adata.obs[resolved_time_key].astype(str).values
    plot_adata.obs["time"] = np.asarray(
        [parse_time_value(v) for v in adata.obs[resolved_time_key].values], dtype=float
    )
    if annotation_column in adata.obs.columns:
        plot_adata.obs["cell_type"] = adata.obs[annotation_column].astype(str).values

    if label_to_color and "cell_type" in plot_adata.obs:
        plot_adata.obs["cell_type"] = plot_adata.obs["cell_type"].astype("category")
        cats = list(plot_adata.obs["cell_type"].cat.categories)
        palette = [label_to_color.get(c, "#888888") for c in cats]
        plot_adata.uns["cell_type_colors"] = palette

    sc.pp.neighbors(plot_adata, n_neighbors=int(n_neighbors), use_rep="X")
    scv.tl.velocity_graph(plot_adata, vkey="velocity", xkey="Ms")
    scv.settings.set_figure_params("scvelo")

    plots: list[str] = []
    color_item_map = {
        "timepoint": ("timepoint", "timepoint", "timepoint"),
        "time": ("time", "time", "time"),
        "cell_type": ("cell_type", "cell type", "celltype"),
    }
    color_items = [
        color_item_map[key]
        for key in selected_color_keys
        if key != "cell_type" or "cell_type" in plot_adata.obs
    ]

    keep_set = {str(x) for x in keep_cell_types} if keep_cell_types else None

    def _prepare_celltype_plot(inner_adata: ad.AnnData) -> tuple[ad.AnnData, str]:
        if "cell_type" not in inner_adata.obs:
            return inner_adata, "cell_type"
        if not keep_set:
            return inner_adata, "cell_type"

        plot_key = "cell_type_keep"
        labels = inner_adata.obs["cell_type"].astype(str)
        inner_adata.obs[plot_key] = labels.where(labels.isin(keep_set), other="Other").astype("category")
        desired = [c for c in sorted(keep_set) if c in inner_adata.obs[plot_key].cat.categories]
        if "Other" in inner_adata.obs[plot_key].cat.categories:
            desired.append("Other")
        if desired:
            inner_adata.obs[plot_key] = inner_adata.obs[plot_key].cat.reorder_categories(desired, ordered=False)
        if label_to_color:
            palette = [label_to_color.get(c, "#888888") for c in desired]
        else:
            base = sc.pl.palettes.default_20
            palette = [base[i % len(base)] for i in range(len(desired))]
        if "Other" in desired:
            palette[desired.index("Other")] = "#D0D0D0"
        inner_adata.uns[f"{plot_key}_colors"] = palette
        return inner_adata, plot_key

    basis_labels = {"umap": "UMAP", "pca": "PCA"}
    for basis in selected_bases:
        basis_label = basis_labels[basis]
        scv.tl.velocity_embedding(plot_adata, basis=basis, vkey="velocity")
        vel_key = f"velocity_{basis}"
        if vel_key in plot_adata.obsm:
            plot_adata.obsm[vel_key] = np.nan_to_num(plot_adata.obsm[vel_key], nan=0.0, posinf=0.0, neginf=0.0)

        for key, title_label, fname_label in color_items:
            cur_adata = plot_adata
            color_key = key
            if key == "cell_type":
                cur_adata, color_key = _prepare_celltype_plot(plot_adata.copy())

            fname = os.path.join(
                out_dir,
                f"velocity_gene_full_{basis}_{fname_label}.{output_format}",
            )
            ax = scv.pl.velocity_embedding_stream(
                cur_adata,
                basis=basis,
                color=color_key,
                density=2,
                figsize=(6, 6),
                title=f"Gene velocity full ({basis_label}, {title_label})",
                show=False,
                legend_loc="right",
            )
            try:
                fig = ax.figure if hasattr(ax, "figure") else None
                if fig is not None:
                    fig.savefig(fname, bbox_inches="tight")
                else:
                    import matplotlib.pyplot as plt

                    plt.savefig(fname, bbox_inches="tight")
            finally:
                import matplotlib.pyplot as plt

                plt.close("all")
            plots.append(fname)

            if key == "cell_type" and celltype_on_data:
                ondata_fname = os.path.join(
                    out_dir,
                    f"velocity_gene_full_{basis}_{fname_label}_ondata.{output_format}",
                )
                ondata_ax = scv.pl.velocity_embedding_stream(
                    cur_adata,
                    basis=basis,
                    color=color_key,
                    density=2,
                    figsize=(6, 6),
                    title=(
                        f"Gene velocity full ({basis_label}, {title_label}, "
                        "on-data labels)"
                    ),
                    show=False,
                    legend_loc="on data",
                )
                try:
                    ondata_fig = (
                        ondata_ax.figure if hasattr(ondata_ax, "figure") else None
                    )
                    if ondata_fig is not None:
                        ondata_fig.savefig(ondata_fname, bbox_inches="tight")
                    else:
                        import matplotlib.pyplot as plt

                        plt.savefig(ondata_fname, bbox_inches="tight")
                finally:
                    import matplotlib.pyplot as plt

                    plt.close("all")
                plots.append(ondata_fname)

    return plots


def plot_gene_expression_trends(
    adata: "ad.AnnData",
    genes: Optional[Sequence[str]] = None,
    time_key: str = "time",
    groupby: str = "Annotation",
    top_hvg: int = 5,
    out_dir: str = ".",
):
    """Plot gene expression trends across time.

    Parameters
    ----------
    adata : AnnData
        AnnData object with gene expression.
    genes : Sequence[str], optional
        Specific genes to plot. If None, uses top HVGs.
    time_key : str
        Key for time in adata.obs.
    groupby : str
        Key for grouping cells.
    top_hvg : int
        Number of top HVGs if genes not specified.
    out_dir : str
        Output directory.

    Returns
    -------
    list
        List of output file paths.
    """
    import os
    import matplotlib.pyplot as plt
    import pandas as pd

    os.makedirs(out_dir, exist_ok=True)

    if genes is None:
        if "highly_variable" in adata.var.columns:
            hvg_mask = adata.var["highly_variable"]
            genes = adata.var_names[hvg_mask][:top_hvg].tolist()
        else:
            genes = adata.var_names[:top_hvg].tolist()

    output_paths = []
    for gene in genes:
        if gene not in adata.var_names:
            continue

        expr = adata[:, gene].X
        if hasattr(expr, "toarray"):
            expr = expr.toarray().flatten()
        else:
            expr = np.array(expr).flatten()

        plot_df = pd.DataFrame({
            time_key: adata.obs[time_key].values,
            groupby: adata.obs[groupby].values,
            "expression": expr,
        })

        mean_df = plot_df.groupby([time_key, groupby])["expression"].mean().reset_index()

        fig, ax = plt.subplots(figsize=(8, 5))
        for group_name in mean_df[groupby].unique():
            group_data = mean_df[mean_df[groupby] == group_name]
            ax.plot(group_data[time_key], group_data["expression"], marker="o", label=group_name)

        ax.set_xlabel("Time")
        ax.set_ylabel("Mean Expression")
        ax.set_title(f"{gene} Expression Trend")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        out_path = os.path.join(out_dir, f"gene_trend_{gene}.pdf")
        fig.savefig(out_path, bbox_inches="tight")
        output_paths.append(out_path)
        plt.close(fig)

    return output_paths


def plot_cell_counts_over_time(
    adata: "ad.AnnData",
    time_key: str = "time",
    groupby: str = "Annotation",
    out_path: Optional[str] = None,
    figsize: tuple = (10, 6),
):
    """Plot cell counts per group across time.

    Parameters
    ----------
    adata : AnnData
        AnnData object.
    time_key : str
        Key for time in adata.obs.
    groupby : str
        Key for grouping cells.
    out_path : str, optional
        Output file path.
    figsize : tuple
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
        Matplotlib figure.
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    counts_df = adata.obs.groupby([time_key, groupby]).size().reset_index(name="count")

    fig, ax = plt.subplots(figsize=figsize)
    for group_name in counts_df[groupby].unique():
        group_data = counts_df[counts_df[groupby] == group_name]
        ax.plot(group_data[time_key], group_data["count"], marker="o", label=group_name)

    ax.set_xlabel("Time")
    ax.set_ylabel("Cell Count")
    ax.set_title("Cell Counts Over Time")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, bbox_inches="tight")

    return fig
