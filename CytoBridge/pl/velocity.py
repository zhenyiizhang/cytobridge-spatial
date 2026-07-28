"""Velocity visualization utilities (scVelo-style).

This module is ported from the ST-1104 downstream implementation and is used
for post-hoc visualization only (does not affect training).
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "plot_velocity_component",
    "plot_intrinsic_interaction_direction_correlation",
    "plot_intrinsic_interaction_direction_correlation_from_adata",
    "SpatialDirectionCorrelationResult",
    "embed_velocity_to_spatial",
    "plot_spatial_component_direction_correlation_roi_from_adata",
]


@dataclass(frozen=True)
class SpatialDirectionCorrelationResult:
    """Projected component directions and the selected spatial ROI."""

    table: object
    embedded_a: np.ndarray
    embedded_b: np.ndarray
    roi_bounds: tuple[float, float, float, float]
    figure_path: Path
    csv_path: Optional[Path]


def embed_velocity_to_spatial(
    feature_matrix: np.ndarray,
    coordinates: np.ndarray,
    velocity: np.ndarray,
    *,
    n_neighbors: int = 30,
    neighbor_rep: str = "X_spatial",
) -> np.ndarray:
    """Project a high-dimensional velocity field into a 2D spatial basis."""
    import anndata as ad
    import scanpy as sc
    import scvelo as scv

    feature_matrix = np.asarray(feature_matrix, dtype=np.float32)
    coordinates = np.asarray(coordinates, dtype=np.float32)
    velocity = np.asarray(velocity, dtype=np.float32)
    if feature_matrix.ndim != 2 or coordinates.ndim != 2 or velocity.ndim != 2:
        raise ValueError("feature_matrix, coordinates, and velocity must be 2D.")
    if coordinates.shape[1] != 2:
        raise ValueError("coordinates must have shape (n_cells, 2).")
    if feature_matrix.shape != velocity.shape:
        raise ValueError("feature_matrix and velocity must have identical shapes.")
    if feature_matrix.shape[0] != coordinates.shape[0]:
        raise ValueError("All arrays must have the same number of rows.")
    if feature_matrix.shape[0] < 2:
        raise ValueError("At least two cells are required for velocity embedding.")
    finite_nonzero = np.isfinite(velocity).all(axis=1) & (
        np.linalg.norm(velocity, axis=1) > 1e-12
    )
    if int(finite_nonzero.sum()) < 2:
        return np.zeros_like(coordinates, dtype=np.float32)

    plot_adata = ad.AnnData(X=feature_matrix.copy())
    plot_adata.obsm["X_spatial"] = coordinates.copy()
    plot_adata.layers["Ms"] = feature_matrix.copy()
    plot_adata.layers["velocity"] = velocity.copy()
    use_rep = neighbor_rep
    if use_rep == "X" or use_rep is None:
        use_rep = "X"
    elif use_rep not in plot_adata.obsm:
        raise KeyError(
            f"neighbor_rep='{neighbor_rep}' is not available; use 'X' or 'X_spatial'."
        )
    n_neighbors = max(1, min(int(n_neighbors), feature_matrix.shape[0] - 1))
    sc.pp.neighbors(plot_adata, n_neighbors=n_neighbors, use_rep=use_rep)
    scv.tl.velocity_graph(plot_adata, vkey="velocity", xkey="Ms")
    scv.tl.velocity_embedding(
        plot_adata,
        basis="spatial",
        vkey="velocity",
    )
    return np.asarray(plot_adata.obsm["velocity_spatial"], dtype=np.float32)


def plot_spatial_component_direction_correlation_roi_from_adata(
    adata,
    out_path: str,
    *,
    target_timepoint: float,
    component_a_key: str = "full_drift_model",
    component_b_key: str = "interaction_model",
    component_a_label: str = "full",
    component_b_label: str = "interaction",
    time_key: Optional[str] = None,
    annotation_key: str = "Annotation",
    focus_label_keyword: Optional[str] = None,
    pad_ratio: float = 0.15,
    n_neighbors: int = 30,
    obsm_key: str = "X_latent",
    spatial_key: str = "spatial_aligned",
    concat_spatial: bool = True,
    csv_path: Optional[str] = None,
) -> SpatialDirectionCorrelationResult:
    """Plot scVelo-projected directional similarity inside a label-defined ROI."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import pandas as pd

    from CytoBridge.tl.downstream.downstream_data import infer_time_key, parse_time_value

    required_obsm = [spatial_key, component_a_key, component_b_key]
    missing = [key for key in required_obsm if key not in adata.obsm]
    if missing:
        raise KeyError(f"adata.obsm is missing required arrays: {missing}")
    resolved_time_key = infer_time_key(adata.obs, preferred=time_key)
    times = np.asarray(
        [parse_time_value(value) for value in adata.obs[resolved_time_key]],
        dtype=float,
    )
    available = np.unique(times)
    selected_time = float(
        available[np.argmin(np.abs(available - float(target_timepoint)))]
    )
    mask = np.isclose(times, selected_time, rtol=0.0, atol=1e-9)
    coordinates = np.asarray(adata.obsm[spatial_key], dtype=np.float32)[mask, :2]
    component_a = np.asarray(adata.obsm[component_a_key], dtype=np.float32)[mask]
    component_b = np.asarray(adata.obsm[component_b_key], dtype=np.float32)[mask]
    if obsm_key in adata.obsm:
        latent = np.asarray(adata.obsm[obsm_key], dtype=np.float32)[mask]
    else:
        latent_raw = adata.X[mask]
        if hasattr(latent_raw, "toarray"):
            latent_raw = latent_raw.toarray()
        latent = np.asarray(latent_raw, dtype=np.float32)
    features = (
        np.hstack((coordinates, latent)).astype(np.float32)
        if concat_spatial
        else latent
    )
    if component_a.shape != features.shape or component_b.shape != features.shape:
        raise ValueError(
            "Component arrays must match the selected model feature matrix; "
            f"features={features.shape}, {component_a_key}={component_a.shape}, "
            f"{component_b_key}={component_b.shape}."
        )

    embedded_a = embed_velocity_to_spatial(
        features,
        coordinates,
        component_a,
        n_neighbors=n_neighbors,
        neighbor_rep="X_spatial",
    )
    embedded_b = embed_velocity_to_spatial(
        features,
        coordinates,
        component_b,
        n_neighbors=n_neighbors,
        neighbor_rep="X_spatial",
    )
    dot = np.einsum("ij,ij->i", embedded_a, embedded_b)
    denominator = np.linalg.norm(embedded_a, axis=1) * np.linalg.norm(
        embedded_b, axis=1
    )
    cosine = np.divide(
        dot,
        denominator,
        out=np.zeros_like(dot),
        where=denominator > 0,
    )

    labels = (
        adata.obs.loc[mask, annotation_key].astype(str).to_numpy()
        if annotation_key in adata.obs.columns
        else np.full(coordinates.shape[0], "unknown", dtype=object)
    )
    focus_mask = np.zeros(coordinates.shape[0], dtype=bool)
    if focus_label_keyword:
        focus_mask = np.char.find(
            np.char.lower(labels.astype(str)),
            str(focus_label_keyword).lower(),
        ) >= 0
    if np.any(focus_mask):
        focus_coordinates = coordinates[focus_mask]
        x0, y0 = focus_coordinates.min(axis=0)
        x1, y1 = focus_coordinates.max(axis=0)
        dx = max(float(x1 - x0), 1e-6)
        dy = max(float(y1 - y0), 1e-6)
        x_min, x_max = x0 - pad_ratio * dx, x1 + pad_ratio * dx
        y_min, y_max = y0 - pad_ratio * dy, y1 + pad_ratio * dy
    else:
        x_min, x_max = np.quantile(coordinates[:, 0], [0.72, 0.92])
        y_min, y_max = np.quantile(coordinates[:, 1], [0.40, 0.70])
    roi_mask = (
        (coordinates[:, 0] >= x_min)
        & (coordinates[:, 0] <= x_max)
        & (coordinates[:, 1] >= y_min)
        & (coordinates[:, 1] <= y_max)
    )
    table = pd.DataFrame(
        {
            "timepoint": selected_time,
            "x": coordinates[roi_mask, 0],
            "y": coordinates[roi_mask, 1],
            "cosine": cosine[roi_mask],
            "celltype": labels[roi_mask],
        }
    )
    if table.empty:
        raise ValueError("The selected ROI contains no cells.")

    figure_path = Path(out_path).expanduser().resolve()
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    csv_output = Path(csv_path).expanduser().resolve() if csv_path else None
    if csv_output is not None:
        csv_output.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(csv_output, index=False)

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
        fig, ax = plt.subplots(figsize=(6.6, 6.0), dpi=150)
        cmap = plt.cm.plasma
        norm = mpl.colors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
        scatter = ax.scatter(
            table["x"],
            table["y"],
            c=table["cosine"],
            s=22,
            linewidths=0,
            cmap=cmap,
            norm=norm,
            alpha=0.95,
        )
        ax.set_xlabel("X (spatial)")
        ax.set_ylabel("Y (spatial)")
        ax.set_aspect("equal")
        ax.set_title(
            f"ROI t={selected_time:g} | {component_a_label} vs {component_b_label}"
        )
        colorbar = fig.colorbar(scatter, ax=ax, pad=0.02)
        colorbar.set_label(
            f"Cosine similarity ({component_a_label} vs {component_b_label})"
        )
        fig.savefig(figure_path, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    return SpatialDirectionCorrelationResult(
        table=table,
        embedded_a=embedded_a,
        embedded_b=embedded_b,
        roi_bounds=(float(x_min), float(x_max), float(y_min), float(y_max)),
        figure_path=figure_path,
        csv_path=csv_output,
    )


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
    n_neighbors: int = 30,
):
    """Plot a model-derived velocity component in aligned spatial coordinates.

    Parameters
    ----------
    coords
        Coordinates of shape (N, 2) to plot in.
    velocity
        Velocity vectors.
        - If `feature_matrix` is None: shape must be (N, 2), and the vectors are
          rendered directly as a locally smoothed aligned-spatial vector field.
        - If `feature_matrix` is provided: shape must match `feature_matrix`
          (for example, N x 50 PCA-state derivatives). scVelo constructs the
          transition graph in that state space and projects it to `coords`.
    feature_matrix
        Optional high-dimensional state space used to build the scVelo
        transition graph. When provided, the model-derived state derivative is
        projected to `coords`. This is not splicing-based RNA velocity unless
        the caller explicitly supplies such a derivative.
    labels
        Optional categorical labels for coloring.
    label_to_color
        Optional mapping label -> hex/rgb.
    out_path
        If provided, saves the figure (pdf/svg/png based on suffix).

    Notes
    -----
    Direct 2D model drift and state-to-space projection are intentionally kept
    separate. scVelo cannot construct a meaningful streamline grid when fewer
    than 50 finite, non-zero embedded state velocities are available. In those
    legitimate edge cases, this function renders a labeled quiver fallback and
    records the reason in ``adata.uns["velocity_plot_fallback"]``. If only the
    vector PDF/SVG backend rejects an otherwise valid scVelo streamline path,
    the already-computed scVelo streamlines are rasterized into that container;
    they are not replaced by per-cell arrows.
    """
    import anndata as ad
    import matplotlib.pyplot as plt
    import scanpy as sc
    import scvelo as scv

    coords = np.asarray(coords, dtype=float)
    velocity = np.asarray(velocity, dtype=float)
    if int(n_neighbors) <= 0:
        raise ValueError("n_neighbors must be > 0.")
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coords must be of shape (N, 2)")
    direct_spatial = feature_matrix is None
    if direct_spatial:
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
    adata.obsm[f"X_{basis}"] = coords
    adata.layers["Ms"] = graph_X
    adata.layers["velocity"] = velocity
    adata.uns["velocity_projection_mode"] = (
        "direct_aligned_spatial_drift"
        if direct_spatial
        else "model_state_velocity_to_aligned_spatial"
    )

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

    def _render_velocity_fallback(
        reason: str,
        valid_vectors: np.ndarray,
        projected_velocity: Optional[np.ndarray] = None,
    ):
        from matplotlib.lines import Line2D

        fig, ax = plt.subplots(figsize=(6, 6))
        if labels is None:
            point_colors = "#7f7f7f"
        else:
            point_colors = [label_to_color.get(label, "#888888") for label in labels_arr]
        ax.scatter(coords[:, 0], coords[:, 1], c=point_colors, s=8, linewidths=0, alpha=0.9)
        quiver_velocity = (
            np.asarray(projected_velocity, dtype=float)
            if projected_velocity is not None
            else (velocity if feature_matrix is None else None)
        )
        if quiver_velocity is not None:
            quiver_mask = np.isfinite(quiver_velocity).all(axis=1) & (
                np.linalg.norm(quiver_velocity, axis=1) > 1e-12
            )
        else:
            quiver_mask = np.asarray(valid_vectors, dtype=bool)
        if quiver_velocity is not None and np.any(quiver_mask):
            ax.quiver(
                coords[quiver_mask, 0],
                coords[quiver_mask, 1],
                quiver_velocity[quiver_mask, 0],
                quiver_velocity[quiver_mask, 1],
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

    def _render_direct_spatial_field():
        """Render explicit 2D model drift without reinterpreting it via scVelo."""
        from matplotlib.lines import Line2D
        from scipy.spatial import cKDTree

        fig, ax = plt.subplots(figsize=(6, 6))
        if labels is None:
            point_colors = "#b8b8b8"
        else:
            point_colors = [
                label_to_color.get(label, "#b8b8b8") for label in labels_arr
            ]
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=point_colors,
            s=7,
            linewidths=0,
            alpha=0.55,
            rasterized=True,
        )
        finite = np.isfinite(coords).all(axis=1) & np.isfinite(velocity).all(axis=1)
        nonzero = finite & (np.linalg.norm(velocity, axis=1) > 1e-12)
        if int(nonzero.sum()) >= 2:
            xy = coords[finite]
            vv = velocity[finite]
            grid_size = int(np.clip(np.sqrt(len(xy)) / 2.0, 14, 30))
            x_grid = np.linspace(
                float(np.quantile(xy[:, 0], 0.01)),
                float(np.quantile(xy[:, 0], 0.99)),
                grid_size,
            )
            y_grid = np.linspace(
                float(np.quantile(xy[:, 1], 0.01)),
                float(np.quantile(xy[:, 1], 0.99)),
                grid_size,
            )
            grid_x, grid_y = np.meshgrid(x_grid, y_grid)
            query = np.column_stack((grid_x.ravel(), grid_y.ravel()))
            tree = cKDTree(xy)
            k = min(max(3, int(n_neighbors)), len(xy))
            distances, indices = tree.query(query, k=k)
            if k == 1:
                distances = distances[:, None]
                indices = indices[:, None]
            positive_distances = distances[np.isfinite(distances) & (distances > 0)]
            bandwidth = (
                float(np.median(distances[:, -1]))
                if positive_distances.size
                else 1.0
            )
            bandwidth = max(bandwidth, np.finfo(float).eps)
            weights = np.exp(-0.5 * np.square(distances / bandwidth))
            weight_sum = weights.sum(axis=1)
            local_velocity = np.divide(
                np.einsum("ij,ijk->ik", weights, vv[indices]),
                weight_sum[:, None],
                out=np.zeros((len(query), 2), dtype=float),
                where=weight_sum[:, None] > 0,
            )
            if len(xy) > 2:
                cell_spacing = tree.query(xy, k=2)[0][:, 1]
                support_radius = max(
                    3.0 * float(np.median(cell_spacing[np.isfinite(cell_spacing)])),
                    np.finfo(float).eps,
                )
            else:
                support_radius = float("inf")
            magnitude = np.linalg.norm(local_velocity, axis=1)
            supported = (
                np.isfinite(local_velocity).all(axis=1)
                & (distances[:, 0] <= support_radius)
                & (magnitude > 1e-12)
            )
            if np.any(supported):
                ax.quiver(
                    query[supported, 0],
                    query[supported, 1],
                    local_velocity[supported, 0],
                    local_velocity[supported, 1],
                    angles="xy",
                    scale_units="xy",
                    scale=None,
                    color="#202020",
                    width=0.0032,
                    alpha=0.85,
                    zorder=5,
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
            ax.legend(
                handles=handles,
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                frameon=False,
            )
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.axis("off")
        adata.uns["velocity_plot_render"] = "direct_smoothed_quiver"
        if out_path:
            fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return adata

    def _save_scvelo_stream(fig) -> Optional[str]:
        """Save scVelo streams, rasterizing only when a vector backend rejects NaNs."""
        if not out_path:
            return "scvelo_stream_not_saved"
        try:
            fig.savefig(out_path, dpi=300, bbox_inches="tight")
            return "scvelo_stream_vector"
        except ValueError as exc:
            if "finite" not in str(exc).lower():
                raise
            # Matplotlib's PDF/SVG path writer can reject a non-finite vertex
            # left by the streamline interpolator even though the Agg renderer
            # displays the valid streamlines correctly. Preserve that exact
            # scVelo panel as a high-resolution raster inside the requested
            # output container.
            buffer = io.BytesIO()
            try:
                fig.savefig(
                    buffer,
                    format="png",
                    dpi=300,
                    bbox_inches="tight",
                    facecolor=fig.get_facecolor(),
                )
                buffer.seek(0)
                raster = plt.imread(buffer, format="png")
                width, height = fig.get_size_inches()
                raster_fig = plt.figure(figsize=(width, height), dpi=300)
                raster_ax = raster_fig.add_axes((0, 0, 1, 1))
                raster_ax.imshow(raster)
                raster_ax.axis("off")
                raster_fig.savefig(
                    out_path,
                    dpi=300,
                    bbox_inches="tight",
                    pad_inches=0,
                )
                plt.close(raster_fig)
                return "scvelo_stream_rasterized"
            except Exception:
                return None
            finally:
                buffer.close()

    valid_vectors = np.isfinite(velocity).all(axis=1) & (
        np.linalg.norm(velocity, axis=1) > 1e-12
    )
    if int(valid_vectors.sum()) < 2:
        return _render_velocity_fallback("near_zero_velocity", valid_vectors)

    if direct_spatial:
        return _render_direct_spatial_field()

    sc.pp.neighbors(adata, n_neighbors=int(n_neighbors), use_rep="X")
    scv.tl.velocity_graph(adata, vkey="velocity", xkey="Ms")
    scv.tl.velocity_embedding(adata, basis=basis, vkey="velocity")
    embedded_velocity = np.asarray(adata.obsm.get(f"velocity_{basis}"), dtype=float)
    valid_embedded = np.isfinite(embedded_velocity).all(axis=1) & (
        np.linalg.norm(embedded_velocity, axis=1) > 1e-12
    )
    if int(valid_embedded.sum()) < 50:
        return _render_velocity_fallback(
            "insufficient_embedded_velocity",
            valid_embedded,
            projected_velocity=embedded_velocity,
        )
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
    render_mode = _save_scvelo_stream(fig)
    plt.close(fig)
    if render_mode is None:
        return _render_velocity_fallback(
            "nonfinite_streamline_raster_render",
            valid_embedded,
            projected_velocity=embedded_velocity,
        )
    adata.uns["velocity_plot_render"] = render_mode
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
