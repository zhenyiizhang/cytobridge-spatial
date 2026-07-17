"""SDE trajectory visualization.

This module provides functions for plotting SDE simulation results,
comparing predicted trajectories with real data.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

__all__ = [
    "plot_sde_vs_real",
    "plot_sde_vs_real_from_adata",
    "plot_trajectory_gif",
    "plot_trajectory_grid",
]


def plot_sde_vs_real(
    df: "pd.DataFrame",
    sde_points: np.ndarray,
    time_values: Sequence[float],
    dim_pairs: Sequence[Tuple[int, int]] = ((0, 1),),
    annotation_key: Optional[str] = None,
    out_prefix: Optional[str] = None,
    samples_column: str = "samples",
    figsize: Tuple[int, int] = (6, 6),
    real_color: str = "red",
    sim_color: str = "blue",
    point_size: float = 3.0,
    alpha: float = 0.6,
):
    """Plot SDE simulation results compared to real data.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with real data.
    sde_points : np.ndarray
        Array of simulated points at each time point.
    time_values : Sequence[float]
        Time values corresponding to sde_points.
    dim_pairs : Sequence[Tuple[int, int]]
        Pairs of dimensions to plot.
    annotation_key : str, optional
        Column for coloring by annotation.
    out_prefix : str, optional
        Prefix for output file paths.
    samples_column : str
        Name of time/samples column in df.
    figsize : Tuple[int, int]
        Figure size.
    real_color : str
        Color for real data points.
    sim_color : str
        Color for simulated data points.
    point_size : float
        Size of scatter points.
    alpha : float
        Alpha transparency.
    """
    import matplotlib.pyplot as plt
    
    from CytoBridge.tl.downstream.downstream_data import infer_feature_columns

    feature_cols = list(infer_feature_columns(df=df, samples_column=samples_column))
    figs = []
    for t_idx, t_val in enumerate(time_values):
        real = df[df[samples_column] == t_val]
        if real.empty:
            continue
        if len(feature_cols) == 0:
            raise ValueError("No feature columns available for trajectory plotting.")
        sim = np.asarray(sde_points[t_idx], dtype=float)

        for d1, d2 in dim_pairs:
            if d1 >= len(feature_cols) or d2 >= len(feature_cols):
                raise ValueError(
                    f"Requested dim pair ({d1}, {d2}) exceeds available feature dims {len(feature_cols)}"
                )
            fig, ax = plt.subplots(figsize=figsize)
            ax.scatter(
                real[feature_cols[d1]].values,
                real[feature_cols[d2]].values,
                c=real_color,
                s=point_size,
                alpha=alpha,
                label="real",
            )
            ax.scatter(
                sim[:, d1],
                sim[:, d2],
                c=sim_color,
                s=point_size,
                alpha=alpha,
                label="sim",
            )
            ax.set_title(f"SDE vs real (t={t_val}) dims {d1+1},{d2+1}")
            ax.legend(loc="best")
            ax.set_xlabel(f"x{d1+1}")
            ax.set_ylabel(f"x{d2+1}")
            
            if out_prefix:
                out_path = f"{out_prefix}_t{t_val}_d{d1+1}_{d2+1}.png"
                fig.savefig(out_path, dpi=300, bbox_inches="tight")
            
            figs.append(fig)
            plt.show()
    
    return figs


def plot_sde_vs_real_from_adata(
    adata,
    sde_points: np.ndarray,
    time_values: Sequence[float],
    *,
    dim_pairs: Sequence[Tuple[int, int]] = ((0, 1),),
    out_prefix: Optional[str] = None,
    time_key: Optional[str] = None,
    obsm_key: str = "X_latent",
    spatial_key: str = "spatial_aligned",
    concat_spatial: Optional[bool] = None,
    figsize: Tuple[int, int] = (6, 6),
    real_color: str = "red",
    sim_color: str = "blue",
    point_size: float = 3.0,
    alpha: float = 0.6,
):
    """AnnData-first version of ``plot_sde_vs_real``."""
    import matplotlib.pyplot as plt

    from CytoBridge.tl.downstream.downstream_data import infer_time_key, parse_time_value

    if not hasattr(adata, "obs") or not hasattr(adata, "obsm"):
        raise TypeError(f"Expected AnnData-like input, got {type(adata)}")

    resolved_time_key = infer_time_key(adata.obs, preferred=time_key)
    obs_times = np.asarray([parse_time_value(v) for v in adata.obs[resolved_time_key].values], dtype=np.float64)

    if obsm_key in adata.obsm:
        latent = np.asarray(adata.obsm[obsm_key], dtype=np.float32)
    else:
        latent = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
        latent = np.asarray(latent, dtype=np.float32)

    use_spatial = bool(concat_spatial) if concat_spatial is not None else (spatial_key in adata.obsm)
    if use_spatial:
        if spatial_key not in adata.obsm:
            raise KeyError(f"concat_spatial=True but adata.obsm['{spatial_key}'] is missing.")
        spatial = np.asarray(adata.obsm[spatial_key], dtype=np.float32)
        if spatial.shape[0] != latent.shape[0]:
            raise ValueError(
                f"Row mismatch between '{spatial_key}' ({spatial.shape[0]}) and "
                f"'{obsm_key}' ({latent.shape[0]})."
            )
        features = np.hstack((spatial, latent)).astype(np.float32)
    else:
        features = latent.astype(np.float32)

    figs = []
    n_dim = features.shape[1]
    for t_idx, t_val in enumerate(time_values):
        mask = np.isclose(obs_times, float(t_val), rtol=0.0, atol=1e-9)
        if not np.any(mask):
            continue
        real = features[mask]
        sim = np.asarray(sde_points[t_idx], dtype=float)

        for d1, d2 in dim_pairs:
            if d1 >= n_dim or d2 >= n_dim:
                raise ValueError(f"Requested dim pair ({d1}, {d2}) exceeds available dims {n_dim}")
            fig, ax = plt.subplots(figsize=figsize)
            ax.scatter(real[:, d1], real[:, d2], c=real_color, s=point_size, alpha=alpha, label="real")
            ax.scatter(sim[:, d1], sim[:, d2], c=sim_color, s=point_size, alpha=alpha, label="sim")
            ax.set_title(f"SDE vs real (t={t_val}) dims {d1+1},{d2+1}")
            ax.legend(loc="best")
            ax.set_xlabel(f"x{d1+1}")
            ax.set_ylabel(f"x{d2+1}")

            if out_prefix:
                out_path = f"{out_prefix}_t{t_val}_d{d1+1}_{d2+1}.png"
                fig.savefig(out_path, dpi=300, bbox_inches="tight")
            figs.append(fig)
            plt.show()
    return figs


def plot_trajectory_grid(
    sde_points: np.ndarray,
    time_values: Sequence[float],
    dim_pairs: Sequence[Tuple[int, int]] = ((0, 1),),
    labels_list: Optional[Sequence[Sequence[str]]] = None,
    label_to_color: Optional[dict] = None,
    out_path: Optional[str] = None,
    figsize_per_panel: Tuple[int, int] = (4, 4),
    point_size: float = 1.0,
    alpha: float = 0.6,
    title: str = "SDE Trajectories",
):
    """Plot SDE trajectories in a grid layout.
    
    Parameters
    ----------
    sde_points : np.ndarray
        Array of simulated points at each time point.
    time_values : Sequence[float]
        Time values corresponding to sde_points.
    dim_pairs : Sequence[Tuple[int, int]]
        Pairs of dimensions to plot.
    labels_list : Sequence[Sequence[str]], optional
        Labels for each time point for coloring.
    label_to_color : dict, optional
        Mapping of labels to colors.
    out_path : str, optional
        Output file path.
    figsize_per_panel : Tuple[int, int]
        Size of each subplot.
    point_size : float
        Size of scatter points.
    alpha : float
        Alpha transparency.
    title : str
        Figure title.
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        Matplotlib figure object.
    """
    import matplotlib.pyplot as plt
    
    n_times = len(time_values)
    n_dim_pairs = len(dim_pairs)
    
    fig, axes = plt.subplots(
        n_dim_pairs,
        n_times,
        figsize=(figsize_per_panel[0] * n_times, figsize_per_panel[1] * n_dim_pairs),
        squeeze=False,
    )
    
    for t_idx, t_val in enumerate(time_values):
        points = np.asarray(sde_points[t_idx], dtype=float)
        
        colors = None
        if labels_list is not None and t_idx < len(labels_list):
            labels = labels_list[t_idx]
            if label_to_color is not None:
                colors = [label_to_color.get(lbl, "#888888") for lbl in labels]
            else:
                unique_labels = sorted(set(labels))
                cmap = plt.get_cmap("tab20", len(unique_labels))
                label_map = {lbl: cmap(i) for i, lbl in enumerate(unique_labels)}
                colors = [label_map[lbl] for lbl in labels]
        
        for dp_idx, (d1, d2) in enumerate(dim_pairs):
            ax = axes[dp_idx, t_idx]
            ax.scatter(
                points[:, d1],
                points[:, d2],
                c=colors,
                s=point_size,
                alpha=alpha,
            )
            ax.set_xlabel(f"x{d1+1}")
            ax.set_ylabel(f"x{d2+1}")
            ax.set_title(f"t={t_val}")
    
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    
    if out_path:
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
    
    return fig


def plot_trajectory_gif(
    sde_points: np.ndarray,
    time_values: Sequence[float],
    labels_list: Optional[Sequence[Sequence[str]]] = None,
    label_to_color: Optional[dict] = None,
    out_path: Optional[str] = None,
    dim_pair: Tuple[int, int] = (0, 1),
    point_size: float = 6.0,
    alpha: float = 0.85,
    fps: int = 4,
):
    """Render a GIF or MP4 trajectory animation colored by predicted labels.

    The writer is selected from ``out_path``: ``.gif`` uses Pillow and ``.mp4``
    uses Matplotlib's ffmpeg writer.
    """
    import os

    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter

    d1, d2 = int(dim_pair[0]), int(dim_pair[1])
    frames = [np.asarray(sde_points[i], dtype=float) for i in range(len(time_values))]
    if not frames:
        raise ValueError("sde_points is empty.")

    all_x = []
    all_y = []
    for points in frames:
        if points.ndim != 2 or max(d1, d2) >= points.shape[1]:
            raise ValueError(f"dim_pair {dim_pair} is out of range for trajectory points.")
        all_x.append(points[:, d1])
        all_y.append(points[:, d2])
    x_all = np.concatenate(all_x)
    y_all = np.concatenate(all_y)
    x_min, x_max = float(np.min(x_all)), float(np.max(x_all))
    y_min, y_max = float(np.min(y_all)), float(np.max(y_all))
    x_pad = max(1e-6, (x_max - x_min) * 0.05)
    y_pad = max(1e-6, (y_max - y_min) * 0.05)

    if labels_list is not None and label_to_color is None:
        uniq = []
        seen = set()
        for labels in labels_list:
            for label in labels:
                key = str(label)
                if key not in seen:
                    seen.add(key)
                    uniq.append(key)
        cmap = plt.get_cmap("tab20")
        label_to_color = {
            lab: "#{:02x}{:02x}{:02x}".format(
                int(cmap(i % cmap.N)[0] * 255),
                int(cmap(i % cmap.N)[1] * 255),
                int(cmap(i % cmap.N)[2] * 255),
            )
            for i, lab in enumerate(uniq)
        }

    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    scatter = ax.scatter([], [], s=point_size, alpha=alpha, linewidths=0)
    title = ax.set_title("")
    ax.set_xlabel(f"x{d1+1}")
    ax.set_ylabel(f"x{d2+1}")
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.set_aspect("equal")

    def _frame_colors(frame_idx: int, n_points: int):
        if labels_list is None or frame_idx >= len(labels_list):
            return "#1f77b4"
        labels = np.asarray(labels_list[frame_idx]).astype(str)
        if labels.shape[0] != n_points:
            n = min(int(labels.shape[0]), int(n_points))
            labels = labels[:n]
            return [label_to_color.get(lab, "#888888") for lab in labels], n
        return [label_to_color.get(lab, "#888888") for lab in labels], n_points

    def _update(frame_idx: int):
        points = frames[frame_idx]
        n_points = points.shape[0]
        x = points[:, d1]
        y = points[:, d2]
        colors = "#1f77b4"
        if labels_list is not None:
            colors, n_keep = _frame_colors(frame_idx, n_points)
            x = x[:n_keep]
            y = y[:n_keep]
        scatter.set_offsets(np.column_stack([x, y]))
        scatter.set_color(colors)
        title.set_text(f"t = {time_values[frame_idx]}")
        return scatter, title

    anim = FuncAnimation(fig, _update, frames=len(frames), interval=max(1, int(1000 / max(1, fps))), blit=False)

    if out_path:
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        suffix = os.path.splitext(str(out_path))[1].lower()
        if suffix == ".gif":
            writer = PillowWriter(fps=max(1, int(fps)))
        elif suffix == ".mp4":
            writer = FFMpegWriter(
                fps=max(1, int(fps)),
                codec="libx264",
                extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
            )
        else:
            raise ValueError("Animation out_path must end in .gif or .mp4.")
        anim.save(out_path, writer=writer)
        plt.close(fig)
        return out_path
    return anim
