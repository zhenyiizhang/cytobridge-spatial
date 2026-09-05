"""Cell-type growth summaries and size-coded spatial maps."""
from __future__ import annotations

import numpy as np


def plot_growth_heatmap(matrix, *, observed_times=None, cmap="RdBu_r", ax=None):
    """Plot a cell-type-by-time table, such as median growth in Fig. S9a.

    Calculate the summary with ``groupby`` and ``unstack`` before calling this
    function. Rows and columns are plotted in the supplied order. Missing
    cell-type/time combinations remain blank. Returns the Matplotlib axis.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or not values.size or not np.isfinite(values).any():
        raise ValueError("matrix must contain at least one finite growth value")
    if np.isinf(values).any() or matrix.index.has_duplicates or matrix.columns.has_duplicates:
        raise ValueError("Use unique cell types and times, and finite values or NaN")
    if ax is None:
        _, ax = plt.subplots(figsize=(max(6, 0.48 * matrix.shape[1]), max(3, 0.32 * matrix.shape[0])))
    colour_map = plt.get_cmap(cmap).copy()
    colour_map.set_bad("white")
    image = ax.imshow(np.ma.masked_invalid(values), aspect="auto", cmap=colour_map)
    for row, column in np.argwhere(np.isfinite(values)):
        value = values[row, column]
        r, g, b, _ = image.cmap(image.norm(value))
        text_colour = "white" if 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.42 else "black"
        ax.text(column, row, f"{value:.2f}", ha="center", va="center", fontsize=9, color=text_colour)
    ax.set_xticks(np.arange(len(matrix.columns)), labels=matrix.columns)
    ax.set_yticks(np.arange(len(matrix.index)), labels=matrix.index)
    ax.set_xlabel("Time point")
    ax.set_ylabel("Cell type")
    ax.set_title("Median growth rate", fontsize=12)
    if observed_times is not None:
        observed = set(observed_times)
        if not observed.issubset(set(matrix.columns)):
            raise ValueError("observed_times contains a time absent from matrix")
        generated = [i for i, t in enumerate(matrix.columns) if t not in observed]
        for group in np.split(generated, np.where(np.diff(generated) != 1)[0] + 1):
            if len(group):
                ax.add_patch(Rectangle((group[0] - 0.5, -0.5), len(group), len(matrix.index),
                                       fill=False, edgecolor="#8a6b58", linewidth=1,
                                       linestyle=(0, (2, 2)), clip_on=False))
    ax.figure.colorbar(image, ax=ax, fraction=0.025, pad=0.015, label="Growth rate")
    return ax


def plot_growth_size_maps(
    growth, *, palette, time_order, observed_times=(), time_key="time_key",
    label_key="celltype", value_key="growth", x_key="x", y_key="y",
    size_range=(0.8, 18.0), alpha_range=(0.4, 1.0), figsize=None,
):
    """Plot growth by point size and cell type by colour, as in Fig. S9b.

    ``growth`` can be the table returned by
    :func:`CytoBridge.tl.evaluate_growth_by_timepoint`. One linear scale from
    the minimum to maximum growth value is shared by all stages. Coordinates
    share a scale and aspect ratio. Returns ``(figure, axes)``.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Rectangle

    columns = [time_key, label_key, value_key, x_key, y_key]
    missing = set(columns).difference(growth.columns)
    if missing:
        raise ValueError(f"Missing growth-table columns: {sorted(missing)}")
    times = list(time_order)
    if not times or len(set(times)) != len(times):
        raise ValueError("time_order must list each displayed time once")
    if set(growth[time_key]) != set(times):
        raise ValueError("time_order must contain all times in the supplied table")
    missing_colours = set(growth[label_key]).difference(palette)
    if missing_colours:
        raise ValueError(f"Supply colours for: {sorted(missing_colours)}")
    values = growth[[value_key, x_key, y_key]].to_numpy(dtype=float)
    if not np.isfinite(values).all() or growth[columns].isna().any().any():
        raise ValueError("Growth values, coordinates, time, and labels must be present and finite")
    if not (0 <= size_range[0] <= size_range[1]) or not (0 <= alpha_range[0] <= alpha_range[1] <= 1):
        raise ValueError("Use non-negative sizes and alpha values between zero and one")
    if not set(observed_times).issubset(set(times)):
        raise ValueError("observed_times contains a time absent from time_order")
    low, high = values[:, 0].min(), values[:, 0].max()

    def normalized(v):
        return np.clip((v - low) / (high - low + 1e-8), 0, 1)

    coord_min = values[:, 1:].min(axis=0)
    coord_scale = max(float(np.ptp(values[:, 1:], axis=0).max()), 1e-12)
    fig, axes = plt.subplots(1, len(times), squeeze=False,
                             figsize=figsize or (max(8, 1.35 * len(times)), 4.5))
    axes = axes[0]
    for ax, time in zip(axes, times):
        subset = growth.loc[growth[time_key].eq(time)]
        for label, cells in subset.groupby(label_key, sort=True, observed=True):
            strength = normalized(cells[value_key].to_numpy())
            coords = (cells[[x_key, y_key]].to_numpy() - coord_min) / coord_scale
            ax.scatter(coords[:, 0], coords[:, 1], color=palette[label],
                       s=size_range[0] + strength * (size_range[1] - size_range[0]),
                       alpha=alpha_range[0] + strength * (alpha_range[1] - alpha_range[0]),
                       linewidths=0, rasterized=False)
        ax.set(xlim=(-0.03, 1.03), ylim=(-0.03, 1.03), aspect="equal")
        ax.set_title(str(time), fontsize=9, color="black")
        ax.set_axis_off()
        if observed_times and time not in observed_times:
            ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, fill=False,
                                   edgecolor="#8a6b58", linestyle=(0, (2, 2)), linewidth=1))
    handles = [Patch(color=colour, label=label) for label, colour in palette.items()
               if label in set(growth[label_key])]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.12),
               ncol=min(4, len(handles)), frameon=False, fontsize=9)
    levels = np.unique(np.r_[low, np.quantile(values[:, 0], [0.05, 0.5, 0.95]), high])
    handles = [Line2D([], [], marker="o", linestyle="", color="black",
                      markersize=np.sqrt(size_range[0] + normalized(v) * (size_range[1] - size_range[0])),
                      label=f"{v:.2f}") for v in levels]
    fig.legend(handles=handles, title="Growth rate", loc="lower center", ncol=len(handles),
               frameon=False, fontsize=9, title_fontsize=9)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.42, wspace=0.02)
    return fig, axes
