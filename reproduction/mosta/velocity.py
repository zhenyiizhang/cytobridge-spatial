"""Stream plotting used for Figure 4e."""
from __future__ import annotations
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scvelo as scv

def plot_single_velocity_field(
    adata,
    velocity_key: str,
    density: float,
    figsize: Tuple[int, int],
    flip_y: bool,
    flip_x: bool,
    title: str,
    color_key: str,
    mode: str = "default",
    remove_outliers: bool = True,
    timepoint_str: Optional[str] = None,
    plot_region: Optional[Sequence[float]] = None,
    palette: Optional[Dict[str, str]] = None,
    scvelo_default_style: bool = False,
    **kwargs,
):
    """Helper to stream-plot a single velocity field (intrinsic or interaction)."""
    alpha_val = kwargs.get("alpha", 0.25)

    if mode == "black":
        plt.style.use("dark_background")
        background_color, text_color = "black", "white"
    else:
        plt.style.use("default")
        background_color, text_color = "white", "black"

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(background_color)
    ax.set_facecolor(background_color)

    adata_plot = adata.copy()
    if remove_outliers:
        y = adata_plot.obsm["X_spatial"][:, 1]
        q1, q3 = np.percentile(y, [25, 75])
        iqr = q3 - q1
        mask = (y >= q1 - 1.5 * iqr) & (y <= q3 + 1.5 * iqr)
        adata_plot = adata_plot[mask].copy()

    if plot_region is not None:
        x_min, x_max, y_min, y_max = plot_region
        X = adata_plot.obsm["X_spatial"]
        mask = np.ones(len(X), dtype=bool)
        if x_min is not None:
            mask &= X[:, 0] > x_min
        if x_max is not None:
            mask &= X[:, 0] < x_max
        if y_min is not None:
            mask &= X[:, 1] > y_min
        if y_max is not None:
            mask &= X[:, 1] < y_max
        adata_plot = adata_plot[mask].copy()
        print(f"  Zoom-in subset: {len(adata_plot)} cells remaining.")

    if isinstance(palette, dict):
        palette = [palette[str(label)] for label in adata_plot.obs[color_key].cat.categories]
    point_size = 50 if plot_region is None else 60
    # scvelo's internal default chooses n_neighbors ~ int(n_obs/50), which can become 0
    # for very small subsets (e.g. after filtering/zoom). Guard to keep it >= 1.
    stream_n_neighbors = max(1, min(30, int(adata_plot.n_obs) - 1)) if int(adata_plot.n_obs) > 1 else 1

    if scvelo_default_style:
        # Keep scVelo defaults for a more canonical look.
        # Still pass a safe `n_neighbors` to avoid scvelo's internal default becoming 0
        # on very small subsets (which raises in sklearn).
        scv.pl.velocity_embedding_stream(
            adata_plot,
            basis="spatial",
            vkey=velocity_key,
            color=color_key,
            palette=palette,
            ax=ax,
            show=False,
            density=density,
            n_neighbors=stream_n_neighbors,
            title="",
        )
    else:
        scv.pl.velocity_embedding_stream(
            adata_plot,
            basis="spatial",
            vkey=velocity_key,
            color=color_key,
            palette=palette,
            ax=ax,
            show=False,
            density=density,
            smooth=0.8,
            min_mass=1,
            cutoff_perc=3,
            linewidth=1.5,
            arrow_size=1.2,
            n_neighbors=stream_n_neighbors,
            alpha=alpha_val,
            size=point_size,
            legend_loc="right margin",
            title="",
            frameon=False,
        )

    if flip_y:
        ax.invert_yaxis()
    if flip_x:
        ax.invert_xaxis()

    if plot_region is not None:
        x_min, x_max, y_min, y_max = plot_region
        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()
        ax.set_xlim(x_min if x_min is not None else cur_xlim[0], x_max if x_max is not None else cur_xlim[1])
        target_ymin = y_min if y_min is not None else (cur_ylim[0] if not flip_y else cur_ylim[1])
        target_ymax = y_max if y_max is not None else (cur_ylim[1] if not flip_y else cur_ylim[0])
        if flip_y:
            ax.set_ylim(target_ymax, target_ymin)
        else:
            ax.set_ylim(target_ymin, target_ymax)

    full_title = f"{title} - {timepoint_str}" if timepoint_str else title
    ax.set_title(full_title, fontsize=20, fontweight="bold", color=text_color, pad=20)
    for spine in ax.spines.values():
        spine.set_color(text_color)
    ax.tick_params(colors=text_color, labelsize=12)
    ax.xaxis.label.set_color(text_color)
    ax.yaxis.label.set_color(text_color)

    return fig, ax
