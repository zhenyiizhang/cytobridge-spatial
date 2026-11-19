"""Draw the brain growth maps in Supplementary Figure S12."""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DISPLAY_TIMES = (0., .25, .5, .75, 1., 1.25, 1.75, 2., 2.25, 2.5, 2.75, 3.)


def plot_brain_growth(cells, *, vmin, vmax):
    """Plot per-cell growth with the common scale and layout used in the SI.

    ``cells`` contains ``time``, ``x``, ``y`` and ``growth`` for brain cells.
    Supply all thirteen quarter-step populations. The colour limits are the
    pooled 5th/95th percentiles recorded by the growth calculation. Twelve
    populations are displayed, omitting time 1.5 as in the manuscript.

    Returns the Matplotlib figure and the numerical summary of its panels.
    No model is evaluated and no existing image is read by this function.
    """
    required = ["time", "x", "y", "growth"]
    values = cells[required].to_numpy(float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Brain growth values must be nonempty and finite.")
    if not np.isfinite([vmin, vmax]).all() or vmin >= vmax:
        raise ValueError("Growth colour limits must be finite and increasing.")
    low, high = cells[["x", "y"]].min(), cells[["x", "y"]].max()
    padding = .04 * (high - low)
    size = 9 * 11.69 / 6.5  # 9 pt when placed at the SI text width.
    with mpl.rc_context({
        "font.family": "Arial", "font.size": size,
        "axes.titlesize": size, "axes.labelsize": size,
        "xtick.labelsize": size, "ytick.labelsize": size,
        "text.color": "black", "axes.titlecolor": "black",
        "axes.labelcolor": "black", "xtick.color": "black",
        "ytick.color": "black", "pdf.fonttype": 42, "ps.fonttype": 42,
    }):
        fig, axes = plt.subplots(3, 4, figsize=(11.69, 8.27))
        fig.subplots_adjust(left=.02, right=.89, bottom=.025, top=.95,
                            wspace=.12, hspace=.18)
        norm = mpl.colors.Normalize(vmin, vmax)
        rows = []
        for axis, time in zip(axes.flat, DISPLAY_TIMES):
            subset = cells.loc[np.isclose(cells.time, time)]
            if subset.empty:
                plt.close(fig)
                raise ValueError(f"Missing brain population at model time {time:g}.")
            axis.scatter(subset.x, subset.y, c=subset.growth.clip(vmin, vmax),
                         cmap="viridis", norm=norm, s=2.2, alpha=.92,
                         linewidths=0, rasterized=False)
            axis.set(xlim=(low.x-padding.x, high.x+padding.x),
                     ylim=(low.y-padding.y, high.y+padding.y))
            axis.set_aspect("equal")
            axis.set_axis_off()
            axis.set_title(f"t = {time:g}", fontweight="normal", pad=6)
            rows.append({"time": time, "cells": len(subset),
                         "growth_mean": subset.growth.mean(),
                         "growth_median": subset.growth.median()})
        colorbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap="viridis"),
                               cax=fig.add_axes([.917, .20, .014, .60]))
        colorbar.solids.set_rasterized(False)
        colorbar.solids.set_edgecolor("face")
        colorbar.set_label("Growth rate", labelpad=8)
        colorbar.set_ticks(np.arange(0, .51, .1))
        colorbar.ax.tick_params(width=.7, length=3.5, colors="black")
        colorbar.outline.set_linewidth(.7)
    return fig, pd.DataFrame(rows)
