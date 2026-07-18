"""SDE trajectory visualization.

This module provides functions for plotting SDE simulation results,
comparing predicted trajectories with real data.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

__all__ = [
    "plot_sde_vs_real",
    "plot_sde_vs_real_from_adata",
    "plot_trajectory_gif",
    "plot_trajectory_comparison_grid",
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
    n_cols: Optional[int] = None,
    show_axes: bool = True,
    show_legend: bool = False,
    equal_aspect: bool = False,
    legend_title: Optional[str] = None,
    legend_fontsize: float = 7.0,
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
    n_cols : int, optional
        Wrap panels into this many columns.  ``None`` preserves the historical
        layout with one row per dimension pair and one column per time point.
    show_axes : bool
        Whether to show coordinate axes and ticks.
    show_legend : bool
        Whether to add one figure-level label legend.
    equal_aspect : bool
        Whether to use equal x/y scaling in every panel.
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        Matplotlib figure object.
    """
    import math

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    
    n_times = len(time_values)
    n_dim_pairs = len(dim_pairs)
    
    if n_times == 0 or n_dim_pairs == 0:
        raise ValueError("time_values and dim_pairs must both be non-empty.")
    if len(sde_points) < n_times:
        raise ValueError(
            f"sde_points has {len(sde_points)} frames for {n_times} time values."
        )
    total_panels = n_times * n_dim_pairs
    layout_cols = n_times if n_cols is None else int(n_cols)
    if layout_cols <= 0:
        raise ValueError("n_cols must be positive when provided.")
    layout_cols = min(layout_cols, total_panels)
    layout_rows = int(math.ceil(total_panels / layout_cols))

    resolved_label_to_color = {
        str(label): color for label, color in dict(label_to_color or {}).items()
    }
    if labels_list is not None and not resolved_label_to_color:
        unique_labels = sorted(
            {
                str(label)
                for labels in labels_list[:n_times]
                for label in labels
            }
        )
        cmap = plt.get_cmap("tab20")
        resolved_label_to_color = {
            label: cmap(index % cmap.N) for index, label in enumerate(unique_labels)
        }

    legend_extra_width = 3.6 if show_legend and resolved_label_to_color else 0.0
    fig, axes = plt.subplots(
        layout_rows,
        layout_cols,
        figsize=(
            figsize_per_panel[0] * layout_cols + legend_extra_width,
            figsize_per_panel[1] * layout_rows,
        ),
        squeeze=False,
        facecolor="white",
    )

    panel_index = 0
    used_labels: set[str] = set()
    for dp_idx, (d1, d2) in enumerate(dim_pairs):
        for t_idx, t_val in enumerate(time_values):
            points = np.asarray(sde_points[t_idx], dtype=float)
            if points.ndim != 2 or max(d1, d2) >= points.shape[1]:
                raise ValueError(
                    f"Frame {t_idx} with shape {points.shape} cannot plot dimensions "
                    f"{(d1, d2)}."
                )
            colors = None
            if labels_list is not None and t_idx < len(labels_list):
                labels = np.asarray(labels_list[t_idx]).astype(str).reshape(-1)
                if labels.shape[0] != points.shape[0]:
                    raise ValueError(
                        f"Frame {t_idx} has {labels.shape[0]} labels for "
                        f"{points.shape[0]} points."
                    )
                used_labels.update(labels.tolist())
                colors = [resolved_label_to_color.get(lbl, "#888888") for lbl in labels]

            ax = axes.flat[panel_index]
            panel_index += 1
            ax.scatter(
                points[:, d1],
                points[:, d2],
                c=colors,
                s=point_size,
                alpha=alpha,
                linewidths=0,
                rasterized=points.shape[0] > 30000,
            )
            if show_axes:
                ax.set_xlabel(f"x{d1+1}")
                ax.set_ylabel(f"x{d2+1}")
            else:
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
            if equal_aspect:
                ax.set_aspect("equal")
            dimensions = f" dims {d1 + 1},{d2 + 1}" if n_dim_pairs > 1 else ""
            ax.set_title(f"t={t_val}{dimensions}")

    for ax in axes.flat[panel_index:]:
        ax.axis("off")

    if show_legend and resolved_label_to_color and used_labels:
        ordered_labels = [
            str(label)
            for label in resolved_label_to_color
            if str(label) in used_labels
        ]
        ordered_labels.extend(sorted(used_labels.difference(ordered_labels)))
        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor=resolved_label_to_color.get(label, "#888888"),
                markeredgecolor="black",
                markeredgewidth=0.4,
                markersize=5,
                label=label,
            )
            for label in ordered_labels
        ]
        fig.legend(
            handles=handles,
            loc="center right",
            bbox_to_anchor=(0.995, 0.5),
            frameon=False,
            title=legend_title,
            fontsize=float(legend_fontsize),
        )

    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0.0, 0.0, 0.78 if show_legend else 1.0, 0.97))
    
    if out_path:
        fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    
    return fig


def plot_trajectory_comparison_grid(
    trajectories: Mapping[str, Sequence[np.ndarray]],
    time_values: Sequence[float],
    *,
    out_path: Optional[str] = None,
    labels_by_condition: Optional[Mapping[str, Sequence[Sequence[str]]]] = None,
    label_to_color: Optional[Mapping[str, object]] = None,
    selected_times: Optional[Sequence[float]] = None,
    condition_titles: Optional[Mapping[str, str]] = None,
    dim_pair: Tuple[int, int] = (0, 1),
    point_size: float = 1.0,
    alpha: float = 0.8,
    figsize_per_panel: Tuple[float, float] = (3.0, 3.0),
    shared_axis_limits: bool = True,
    show_counts: bool = False,
    show_legend: bool = False,
    legend_title: Optional[str] = None,
    title: Optional[str] = None,
):
    """Plot matched trajectory conditions as time-by-condition panels.

    This dataset-agnostic comparison layout is useful for baseline/perturbation
    trajectories and any other matched simulation branches.  Rows are selected
    times and columns follow the insertion order of ``trajectories``.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    conditions = list(trajectories)
    times = [float(value) for value in time_values]
    if not conditions:
        raise ValueError("trajectories must contain at least one condition.")
    if not times:
        raise ValueError("time_values must be non-empty.")
    if len(dim_pair) != 2 or min(dim_pair) < 0:
        raise ValueError("dim_pair must contain two non-negative dimensions.")
    for condition, trajectory in trajectories.items():
        if len(trajectory) != len(times):
            raise ValueError(
                f"Trajectory {condition!r} has {len(trajectory)} frames for "
                f"{len(times)} time values."
            )

    if selected_times is None:
        selected_indices = list(range(len(times)))
    else:
        selected_indices = []
        for requested in selected_times:
            matches = [
                index
                for index, value in enumerate(times)
                if np.isclose(value, float(requested), rtol=0.0, atol=1e-8)
            ]
            if not matches:
                raise ValueError(f"Selected time {requested} is absent from time_values.")
            selected_indices.append(matches[0])
    if not selected_indices:
        raise ValueError("selected_times did not select any frames.")

    resolved_colors = {
        str(label): color for label, color in dict(label_to_color or {}).items()
    }
    if labels_by_condition is not None and not resolved_colors:
        unique_labels = sorted(
            {
                str(label)
                for condition in conditions
                for index in selected_indices
                for label in labels_by_condition[condition][index]
            }
        )
        cmap = plt.get_cmap("tab20")
        resolved_colors = {
            label: cmap(index % cmap.N) for index, label in enumerate(unique_labels)
        }

    all_plot_values: list[np.ndarray] = []
    for condition in conditions:
        for index in selected_indices:
            frame = np.asarray(trajectories[condition][index], dtype=float)
            if frame.ndim != 2 or max(dim_pair) >= frame.shape[1]:
                raise ValueError(
                    f"Trajectory {condition!r} frame {index} with shape {frame.shape} "
                    f"cannot plot dimensions {dim_pair}."
                )
            if frame.shape[0] > 0:
                all_plot_values.append(frame[:, dim_pair])
    axis_limits = None
    if shared_axis_limits and all_plot_values:
        values = np.concatenate(all_plot_values, axis=0)
        x_min, y_min = np.nanmin(values, axis=0)
        x_max, y_max = np.nanmax(values, axis=0)
        x_pad = max(1e-6, float(x_max - x_min) * 0.05)
        y_pad = max(1e-6, float(y_max - y_min) * 0.05)
        axis_limits = (x_min - x_pad, x_max + x_pad, y_min - y_pad, y_max + y_pad)

    legend_extra_width = 3.6 if show_legend and resolved_colors else 0.0
    fig, axes = plt.subplots(
        len(selected_indices),
        len(conditions),
        figsize=(
            float(figsize_per_panel[0]) * len(conditions) + legend_extra_width,
            float(figsize_per_panel[1]) * len(selected_indices),
        ),
        squeeze=False,
        facecolor="white",
    )
    used_labels: set[str] = set()
    for row, time_index in enumerate(selected_indices):
        for column, condition in enumerate(conditions):
            ax = axes[row, column]
            frame = np.asarray(trajectories[condition][time_index], dtype=float)
            point_colors: object = "#4c78a8"
            if labels_by_condition is not None:
                if condition not in labels_by_condition:
                    raise KeyError(f"labels_by_condition is missing {condition!r}.")
                labels = np.asarray(
                    labels_by_condition[condition][time_index]
                ).astype(str).reshape(-1)
                if labels.shape[0] != frame.shape[0]:
                    raise ValueError(
                        f"Trajectory {condition!r} frame {time_index} has "
                        f"{labels.shape[0]} labels for {frame.shape[0]} points."
                    )
                used_labels.update(labels.tolist())
                point_colors = [resolved_colors.get(label, "#888888") for label in labels]
            if frame.shape[0] > 0:
                ax.scatter(
                    frame[:, dim_pair[0]],
                    frame[:, dim_pair[1]],
                    c=point_colors,
                    s=float(point_size),
                    alpha=float(alpha),
                    linewidths=0,
                    rasterized=frame.shape[0] > 30000,
                )
            if axis_limits is not None:
                ax.set_xlim(axis_limits[0], axis_limits[1])
                ax.set_ylim(axis_limits[2], axis_limits[3])
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row == 0:
                heading = (condition_titles or {}).get(condition, condition)
                ax.set_title(
                    f"{heading}\n(n={frame.shape[0]})" if show_counts else heading,
                    fontsize=11,
                    fontweight="bold",
                )
            elif show_counts:
                ax.set_title(f"n={frame.shape[0]}", fontsize=8)
            if column == 0:
                ax.set_ylabel(
                    f"t={times[time_index]:g}",
                    rotation=0,
                    ha="right",
                    va="center",
                    labelpad=12,
                    fontsize=9,
                )

    if show_legend and resolved_colors and used_labels:
        ordered_labels = [label for label in resolved_colors if label in used_labels]
        ordered_labels.extend(sorted(used_labels.difference(ordered_labels)))
        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor=resolved_colors.get(label, "#888888"),
                markeredgecolor="none",
                label=label,
                markersize=5,
            )
            for label in ordered_labels
        ]
        fig.legend(
            handles=handles,
            loc="center right",
            bbox_to_anchor=(0.995, 0.5),
            frameon=False,
            title=legend_title,
            fontsize=7,
        )
    if title:
        fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0.0, 0.0, 0.78 if show_legend else 1.0, 0.98))
    if out_path:
        fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
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
