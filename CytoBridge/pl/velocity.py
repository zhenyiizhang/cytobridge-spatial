"""Velocity visualization utilities (scVelo-style).

This module is ported from the ST-1104 downstream implementation and is used
for post-hoc visualization only (does not affect training).
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "plot_velocity_component",
    "plot_intrinsic_interaction_direction_correlation",
    "plot_intrinsic_interaction_direction_correlation_from_adata",
]


def plot_velocity_component(
    coords: "np.ndarray",
    velocity: "np.ndarray",
    feature_matrix: Optional["np.ndarray"] = None,
    labels: Optional[Sequence[str]] = None,
    label_to_color: Optional[Dict[str, str]] = None,
    title: str = "",
    out_path: Optional[str] = None,
    density: float = 2.0,
    basis: str = "spatial",
    show_legend: bool = False,
):
    """Plot a single velocity field using scVelo streamlines.

    Parameters
    ----------
    coords
        Coordinates of shape (N, 2) to plot in.
    velocity
        Velocity vectors.
        - If `feature_matrix` is None: shape must be (N, 2).
        - If `feature_matrix` is provided: shape must match `feature_matrix` (e.g., N x 50).
    feature_matrix
        Optional high-dimensional feature space used to build velocity graph.
        When provided, streamlines can be projected to `coords` (spatial basis) while
        velocity is computed in this high-dimensional space.
    labels
        Optional categorical labels for coloring.
    label_to_color
        Optional mapping label -> hex/rgb.
    out_path
        If provided, saves the figure (pdf/svg/png based on suffix).

    Notes
    -----
    scVelo cannot construct a meaningful streamline grid when the raw field is
    nearly zero or fewer than 50 finite, non-zero embedded velocity vectors are
    available. In those legitimate edge cases (for example, a small sample
    with no predicted interaction edges), this function renders a labeled
    scatter/quiver fallback and records the reason in
    ``adata.uns["velocity_plot_fallback"]``.
    """
    import anndata as ad
    import matplotlib.pyplot as plt
    import scanpy as sc
    import scvelo as scv

    coords = np.asarray(coords, dtype=float)
    velocity = np.asarray(velocity, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coords must be of shape (N, 2)")
    if feature_matrix is None:
        if velocity.shape != coords.shape:
            raise ValueError("velocity must have the same shape as coords when feature_matrix is None")
        graph_X = coords
    else:
        graph_X = np.asarray(feature_matrix, dtype=float)
        if graph_X.ndim != 2:
            raise ValueError("feature_matrix must be a 2D array")
        if graph_X.shape[0] != coords.shape[0]:
            raise ValueError("feature_matrix and coords must have the same number of rows")
        if velocity.shape != graph_X.shape:
            raise ValueError("velocity must have the same shape as feature_matrix when feature_matrix is provided")

    adata = ad.AnnData(X=graph_X)
    adata.obsm["X_spatial"] = coords
    adata.layers["Ms"] = graph_X
    adata.layers["velocity"] = velocity

    palette_list = None
    if labels is not None:
        labels_arr = np.asarray(labels).astype(str)
        if label_to_color is None:
            uniq = sorted(set(labels_arr))
            cmap = plt.get_cmap("tab20", len(uniq))
            label_to_color = {
                u: "#{:02x}{:02x}{:02x}".format(
                    int(cmap(i)[0] * 255), int(cmap(i)[1] * 255), int(cmap(i)[2] * 255)
                )
                for i, u in enumerate(uniq)
            }

        categories = [c for c in label_to_color.keys() if c in set(labels_arr)]
        if not categories:
            categories = sorted(set(labels_arr))
        adata.obs["Annotation"] = labels_arr
        adata.obs["Annotation"] = adata.obs["Annotation"].astype("category")
        adata.obs["Annotation"] = adata.obs["Annotation"].cat.reorder_categories(categories, ordered=True)
        palette_list = [label_to_color.get(cat, "#888888") for cat in categories]
        adata.uns["Annotation_colors"] = palette_list

    def _render_velocity_fallback(reason: str, valid_vectors: np.ndarray):
        from matplotlib.lines import Line2D

        fig, ax = plt.subplots(figsize=(6, 6))
        if labels is None:
            point_colors = "#7f7f7f"
        else:
            point_colors = [label_to_color.get(label, "#888888") for label in labels_arr]
        ax.scatter(coords[:, 0], coords[:, 1], c=point_colors, s=8, linewidths=0, alpha=0.9)
        if feature_matrix is None and np.any(valid_vectors):
            ax.quiver(
                coords[valid_vectors, 0],
                coords[valid_vectors, 1],
                velocity[valid_vectors, 0],
                velocity[valid_vectors, 1],
                angles="xy",
                scale_units="xy",
                scale=1.0,
                color="#222222",
            )
        if labels is not None and show_legend:
            handles = [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="",
                    markerfacecolor=label_to_color.get(category, "#888888"),
                    markeredgecolor="none",
                    label=category,
                )
                for category in categories
            ]
            ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        adata.uns["velocity_plot_fallback"] = reason
        if out_path:
            fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return adata

    valid_vectors = np.isfinite(velocity).all(axis=1) & (
        np.linalg.norm(velocity, axis=1) > 1e-12
    )
    if int(valid_vectors.sum()) < 2:
        return _render_velocity_fallback("near_zero_velocity", valid_vectors)

    sc.pp.neighbors(adata, n_neighbors=30, use_rep="X")
    scv.tl.velocity_graph(adata, vkey="velocity")
    scv.tl.velocity_embedding(adata, basis=basis, vkey="velocity")
    embedded_velocity = np.asarray(adata.obsm.get(f"velocity_{basis}"), dtype=float)
    valid_embedded = np.isfinite(embedded_velocity).all(axis=1) & (
        np.linalg.norm(embedded_velocity, axis=1) > 1e-12
    )
    if int(valid_embedded.sum()) < 50:
        return _render_velocity_fallback("insufficient_embedded_velocity", valid_vectors)
    scv.settings.set_figure_params("scvelo")

    fig, ax = plt.subplots(figsize=(6, 6))
    legend_loc = "right margin" if show_legend else "none"
    scv.pl.velocity_embedding_stream(
        adata,
        basis=basis,
        color="Annotation" if labels is not None else None,
        palette=palette_list,
        density=density,
        ax=ax,
        show=False,
        title=title,
        legend_loc=legend_loc,
    )
    if out_path:
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return adata


def plot_intrinsic_interaction_direction_correlation(
    df,
    dim: int,
    model,
    out_path: str,
    device: str = "cpu",
    samples_column: str = "samples",
):
    """Plot per-timepoint cosine similarity between intrinsic and interaction velocities.

    This is a direct port of the final block in ``velocity_migration.ipynb``.
    """
    import math
    import os

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    from CytoBridge.tl import compute_velocity_components

    from CytoBridge.tl.downstream.downstream_data import infer_feature_columns

    rows = []
    timepoints = sorted(df[samples_column].unique())
    feature_cols = list(infer_feature_columns(df=df, samples_column=samples_column))[:dim]
    if len(feature_cols) < dim:
        raise ValueError(f"Requested dim={dim}, found only {len(feature_cols)} feature columns")
    if len(feature_cols) < 2:
        raise ValueError("At least 2 feature columns are required for spatial correlation plotting")

    for t in timepoints:
        df_t = df[df[samples_column] == t]
        if df_t.empty:
            continue
        data_t = df_t[feature_cols].values
        coords = data_t[:, :2]
        vel = compute_velocity_components(data=data_t, time_value=float(t), model=model, device=device)
        v_intr = vel["drift"][:, :2]
        v_inter = vel["interaction"][:, :2]
        dot = np.einsum("ij,ij->i", v_intr, v_inter)
        denom = np.linalg.norm(v_intr, axis=1) * np.linalg.norm(v_inter, axis=1)
        cos = np.divide(dot, denom, out=np.zeros_like(dot), where=denom > 0)
        rows.append(
            pd.DataFrame(
                {
                    "timepoint": str(t),
                    "x": coords[:, 0],
                    "y": coords[:, 1],
                    "cos": cos,
                }
            )
        )

    if not rows:
        raise ValueError("No valid rows were generated for correlation plotting.")
    df_cos = pd.concat(rows, ignore_index=True)
    df_cos = df_cos[np.isfinite(df_cos["cos"])]

    sns.set_theme(style="white", context="talk")
    ncols = min(3, len(timepoints))
    nrows = int(math.ceil(len(timepoints) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.8 * nrows), dpi=150, squeeze=False)

    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "intrinsic_interaction_corr", ["#2F5DFF", "#F7F7F7", "#E63946"]
    )
    norm = mpl.colors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)

    for i, t in enumerate(timepoints):
        r, c = divmod(i, ncols)
        ax = axes[r][c]
        sub = df_cos[df_cos["timepoint"] == str(t)]
        ax.scatter(sub["x"], sub["y"], c=sub["cos"], s=6, linewidths=0, cmap=cmap, norm=norm, alpha=0.9)
        ax.set_title(f"t = {t}")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")

    for j in range(len(timepoints), nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r][c].axis("off")

    cax = fig.add_axes([0.92, 0.18, 0.02, 0.64])
    cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm)
    cb.set_label("Cosine similarity (intrinsic vs interaction)")

    fig.suptitle("Spatial intrinsic vs interaction velocity direction correlation", y=0.98)
    plt.tight_layout(rect=[0, 0, 0.9, 0.96])

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_intrinsic_interaction_direction_correlation_from_adata(
    adata,
    out_path: str,
    *,
    time_key: str | None = None,
    spatial_key: str = "spatial_aligned",
    drift_key: str = "velocity_model",
    interaction_key: str = "interaction_model",
):
    """Plot per-timepoint cosine similarity using precomputed components in AnnData."""
    import math
    import os

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    from CytoBridge.tl.downstream.downstream_data import infer_time_key, parse_time_value

    if not hasattr(adata, "obs") or not hasattr(adata, "obsm"):
        raise TypeError(f"Expected AnnData-like object, got {type(adata)}")
    if spatial_key not in adata.obsm:
        raise KeyError(f"adata.obsm['{spatial_key}'] is required.")
    if drift_key not in adata.obsm:
        raise KeyError(f"adata.obsm['{drift_key}'] is required.")
    if interaction_key not in adata.obsm:
        raise KeyError(f"adata.obsm['{interaction_key}'] is required.")

    resolved_time_key = infer_time_key(adata.obs, preferred=time_key)
    times = np.asarray([parse_time_value(v) for v in adata.obs[resolved_time_key].values], dtype=float)
    timepoints = sorted(np.unique(times).tolist())

    coords = np.asarray(adata.obsm[spatial_key], dtype=float)
    drift = np.asarray(adata.obsm[drift_key], dtype=float)
    interaction = np.asarray(adata.obsm[interaction_key], dtype=float)
    if coords.shape[1] < 2:
        raise ValueError(f"'{spatial_key}' must have at least 2 dimensions.")
    if drift.shape[0] != coords.shape[0] or interaction.shape[0] != coords.shape[0]:
        raise ValueError("Row count mismatch among spatial/drift/interaction arrays.")
    if drift.shape[1] < 2 or interaction.shape[1] < 2:
        raise ValueError("drift/interaction vectors must have at least 2 dimensions.")

    rows = []
    for t in timepoints:
        mask = np.isclose(times, float(t), rtol=0.0, atol=1e-9)
        if not np.any(mask):
            continue
        v_intr = drift[mask, :2]
        v_inter = interaction[mask, :2]
        dot = np.einsum("ij,ij->i", v_intr, v_inter)
        denom = np.linalg.norm(v_intr, axis=1) * np.linalg.norm(v_inter, axis=1)
        cos = np.divide(dot, denom, out=np.zeros_like(dot), where=denom > 0)
        rows.append(
            pd.DataFrame(
                {
                    "timepoint": str(t),
                    "x": coords[mask, 0],
                    "y": coords[mask, 1],
                    "cos": cos,
                }
            )
        )

    if not rows:
        raise ValueError("No valid rows were generated for correlation plotting.")
    df_cos = pd.concat(rows, ignore_index=True)
    df_cos = df_cos[np.isfinite(df_cos["cos"])]

    sns.set_theme(style="white", context="talk")
    ncols = min(3, len(timepoints))
    nrows = int(math.ceil(len(timepoints) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.8 * nrows), dpi=150, squeeze=False)

    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "intrinsic_interaction_corr", ["#2F5DFF", "#F7F7F7", "#E63946"]
    )
    norm = mpl.colors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)

    for i, t in enumerate(timepoints):
        r, c = divmod(i, ncols)
        ax = axes[r][c]
        sub = df_cos[df_cos["timepoint"] == str(t)]
        ax.scatter(sub["x"], sub["y"], c=sub["cos"], s=6, linewidths=0, cmap=cmap, norm=norm, alpha=0.9)
        ax.set_title(f"t = {t}")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")

    for j in range(len(timepoints), nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r][c].axis("off")

    cax = fig.add_axes([0.92, 0.18, 0.02, 0.64])
    cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm)
    cb.set_label("Cosine similarity (intrinsic vs interaction)")

    fig.suptitle("Spatial intrinsic vs interaction velocity direction correlation", y=0.98)
    plt.tight_layout(rect=[0, 0, 0.9, 0.96])

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path
