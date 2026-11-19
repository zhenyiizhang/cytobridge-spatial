"""Spatial populations, cell-type proportions, and counts over time."""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd


def plot_population_overview(
    spatial_populations: Mapping[float, object],
    proportions: pd.DataFrame,
    counts: pd.DataFrame,
    *,
    palette: Mapping[str, str],
    observed_times: Sequence[float],
    ages: Mapping[float, float],
    annotation_key: str,
    spatial_key: str = "spatial",
    max_points_per_stage: int = 10000,
    random_seed: int = 42,
):
    """Plot already calculated populations and counts on an A4 canvas.

    Spatial planes are equally spaced for display. Their labels show the
    supplied ages. The count curves use ages as a continuous horizontal axis.
    Dashed outlines mark times without observations in the spatial and
    proportion panels. Display sampling affects only the spatial scatter.

    Parameters
    ----------
    spatial_populations
        Mapping from model time to AnnData, containing coordinates and labels.
    proportions
        Cell-type fractions, indexed by model time. Each row must sum to one.
    counts
        Predicted cell-type counts, indexed by model time.
    palette
        Colours keyed by cell-type labels. Its order sets the legend order.
    observed_times
        Times at which measurements were available. This changes the frame
        style only, not the populations supplied by the caller.
    ages
        Model-time to age mapping, covering all three panels.
    annotation_key, spatial_key
        AnnData fields for labels and spatial coordinates.
    max_points_per_stage
        Maximum number of uniformly sampled spatial points per plane.
    random_seed
        Seed used only for display sampling.

    Returns
    -------
    fig, axes
        Matplotlib figure and the three axes. No model is evaluated here.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    if max_points_per_stage < 1:
        raise ValueError("max_points_per_stage must be positive")
    for name, table in (("proportions", proportions), ("counts", counts)):
        if table.empty or not table.index.is_unique or not table.columns.is_unique:
            raise ValueError(f"{name} must have unique, non-empty rows and columns")
        if not np.isfinite(table.to_numpy()).all() or (table.to_numpy() < 0).any():
            raise ValueError(f"{name} must contain finite non-negative values")
    if not np.allclose(proportions.sum(axis=1), 1):
        raise ValueError("Each proportion row must sum to one")
    needed_labels = set(proportions.columns) | set(counts.columns)
    times = sorted(spatial_populations)
    if not times:
        raise ValueError("At least one spatial population is required")
    for time in times:
        population = spatial_populations[time]
        needed_labels.update(population.obs[annotation_key].astype(str))
        coordinates = np.asarray(population.obsm[spatial_key])
        if coordinates.shape != (population.n_obs, 2) or not np.isfinite(coordinates).all():
            raise ValueError("Spatial coordinates must be a finite n_cells by 2 array")
    missing = needed_labels - set(palette)
    if missing:
        raise ValueError(f"Missing cell-type colours: {sorted(missing)}")
    all_times = set(times) | set(proportions.index) | set(counts.index)
    if all_times - set(ages):
        raise ValueError("ages must cover every plotted time")
    if any(not np.isfinite(ages[t]) for t in all_times):
        raise ValueError("ages must be finite")
    sorted_times = sorted(all_times)
    if any(ages[b] <= ages[a] for a, b in zip(sorted_times, sorted_times[1:])):
        raise ValueError("ages must increase with model time")

    labels = [label for label in palette if label in needed_labels]
    observed = set(map(float, observed_times))
    fig = plt.figure(figsize=(8.27, 11.69))
    ax_spatial = fig.add_axes([0.08, 0.52, 0.84, 0.43], projection="3d")
    ax_proportions = fig.add_axes([0.13, 0.285, 0.81, 0.15])
    ax_counts = fig.add_axes([0.13, 0.065, 0.81, 0.15])
    for letter, title, y in (("a", "Spatial populations", 0.97),
                             ("b", "Cell-type proportions", 0.463),
                             ("c", "Predicted cell numbers", 0.244)):
        fig.text(0.065, y, letter, fontsize=14, weight="bold")
        fig.text(0.11, y, title, fontsize=12, weight="bold")

    all_coordinates = np.concatenate([np.asarray(spatial_populations[t].obsm[spatial_key]) for t in times])
    low, high = all_coordinates.min(axis=0), all_coordinates.max(axis=0)
    padding = np.maximum(0.02 * (high - low), 1e-5)
    low, high = low - padding, high + padding
    rng = np.random.default_rng(random_seed)
    for level, time in enumerate(times):
        population = spatial_populations[time]
        indices = np.sort(rng.choice(population.n_obs, size=min(max_points_per_stage, population.n_obs), replace=False))
        coordinates = np.asarray(population.obsm[spatial_key])[indices]
        colours = [palette[label] for label in population.obs[annotation_key].astype(str).iloc[indices]]
        ax_spatial.scatter(coordinates[:, 0], coordinates[:, 1], np.full(len(indices), level),
                           c=colours, s=0.6, alpha=0.75, linewidths=0, depthshade=False)
        x = [low[0], high[0], high[0], low[0], low[0]]
        y = [low[1], low[1], high[1], high[1], low[1]]
        ax_spatial.plot(x, y, np.full(5, level), color="black", lw=0.6,
                        linestyle="-" if time in observed else (0, (4, 3)))
    ax_spatial.view_init(elev=18, azim=-60)
    ax_spatial.set_box_aspect((1, 1, 1.4))
    ax_spatial.set(xticks=[], yticks=[], zticks=range(len(times)),
                   zticklabels=[f"{ages[t]:.1f}" for t in times], zlabel="Age (months)")
    ax_spatial.tick_params(labelsize=9)
    ax_spatial.grid(False)
    for axis in (ax_spatial.xaxis, ax_spatial.yaxis, ax_spatial.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor((1, 1, 1, 0))
    handles = [Line2D([], [], marker="o", linestyle="none", color=palette[label],
                      markersize=4, label=label) for label in labels]
    fig.legend(handles=handles, loc="center", bbox_to_anchor=(0.51, 0.50),
               ncol=4, frameon=False, fontsize=9, columnspacing=0.9, handletextpad=0.3)

    proportions = proportions.reindex(columns=labels, fill_value=0).sort_index()
    positions = np.arange(len(proportions))
    bottom = np.zeros(len(proportions))
    for label in labels:
        values = 100 * proportions[label].to_numpy()
        ax_proportions.bar(positions, values, bottom=bottom, width=0.82,
                           color=palette[label], linewidth=0.3, edgecolor="white")
        bottom += values
    for position, time in zip(positions, proportions.index):
        if time not in observed:
            ax_proportions.add_patch(Rectangle((position - 0.45, 0), 0.9, 100,
                                               fill=False, ec="black", lw=0.45, ls=(0, (3, 2))))
    labelled = [i for i, t in enumerate(proportions.index)
                if t in observed or i == len(proportions) - 1]
    ax_proportions.set(xticks=labelled,
                       xticklabels=[f"{ages[proportions.index[i]]:.1f}" for i in labelled],
                       xlabel="Age (months)", ylabel="Cell proportion (%)", ylim=(0, 100))
    counts = counts.sort_index()
    for label in labels:
        if label in counts:
            ax_counts.plot([ages[t] for t in counts.index], counts[label],
                            color=palette[label], lw=1.7)
    observed_ages = [ages[t] for t in sorted(observed) if t in ages]
    ax_counts.set(xlabel="Age (months)", ylabel="Number of cells", xticks=observed_ages)
    ax_counts.set_ylim(bottom=0)
    for axis in (ax_proportions, ax_counts):
        axis.tick_params(labelsize=9)
        axis.spines[["top", "right"]].set_visible(False)
    return fig, (ax_spatial, ax_proportions, ax_counts)
