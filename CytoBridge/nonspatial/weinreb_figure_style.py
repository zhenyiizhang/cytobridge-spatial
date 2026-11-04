"""Reusable Matplotlib style helpers for CytoBridge manuscript figures."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt


A4_PORTRAIT = (8.27, 11.69)

GT_COLOR = "#59616A"
CYTOBRIDGE_COLOR = "#07838B"
ABLATION_COLOR = "#CC6677"
GT_INTERACTION_COLOR = "#C95F72"
GRID_COLOR = "#D7DDE2"
TEXT_COLOR = "#111111"
HEADING_COLOR = "#111111"

PANEL_LABEL_SIZE = 14.0
GROUP_TITLE_SIZE = 12.0
PLOT_TEXT_SIZE = 9.0


def apply_style() -> None:
    """Apply the default manuscript typography and line settings."""

    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": PLOT_TEXT_SIZE,
            "axes.titlesize": PLOT_TEXT_SIZE,
            "axes.labelsize": PLOT_TEXT_SIZE,
            "xtick.labelsize": PLOT_TEXT_SIZE,
            "ytick.labelsize": PLOT_TEXT_SIZE,
            "legend.fontsize": PLOT_TEXT_SIZE,
            "axes.linewidth": 0.65,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "text.color": TEXT_COLOR,
            "axes.labelcolor": TEXT_COLOR,
            "axes.titlecolor": HEADING_COLOR,
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def panel_heading(
    ax: mpl.axes.Axes,
    label: str,
    title: str,
    *,
    label_x: float = 0.0,
    title_x: float = 0.045,
    y: float = 0.55,
) -> None:
    """Draw a lower-case panel label and sentence-case group title."""

    ax.axis("off")
    ax.text(
        label_x,
        y,
        label.lower(),
        fontsize=PANEL_LABEL_SIZE,
        fontweight="bold",
        va="center",
        ha="left",
        color=HEADING_COLOR,
    )
    ax.text(
        title_x,
        y,
        title,
        fontsize=GROUP_TITLE_SIZE,
        fontweight="bold",
        va="center",
        ha="left",
        color=HEADING_COLOR,
    )


def time_count_title(
    ax: mpl.axes.Axes,
    time: float | int | str,
    count: int,
    *,
    pad: float = 1.5,
) -> None:
    """Place sample count directly below the corresponding time label."""

    ax.set_title(
        f"t = {time}\nn = {count}",
        fontsize=PLOT_TEXT_SIZE,
        pad=pad,
        linespacing=0.95,
    )


def clean_axis(ax: mpl.axes.Axes, *, grid: bool = True) -> None:
    """Apply the standard open-axis treatment."""

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(color=GRID_COLOR, linewidth=0.45, alpha=0.55, zorder=0)
    else:
        ax.grid(False)
    ax.set_axisbelow(True)


def safe_legend(
    ax: mpl.axes.Axes,
    *,
    handles: Iterable | None = None,
    labels: Iterable[str] | None = None,
    title: str | None = None,
    loc: str = "best",
    **kwargs,
):
    """Create a compact frameless legend using the manuscript defaults.

    Visual inspection is still required. Move the legend if it overlaps data,
    error bars, titles, or annotations.
    """

    legend = ax.legend(
        handles=handles,
        labels=labels,
        title=title,
        loc=loc,
        frameon=False,
        fontsize=PLOT_TEXT_SIZE,
        **kwargs,
    )
    if legend is not None and legend.get_title() is not None:
        legend.get_title().set_fontsize(PLOT_TEXT_SIZE)
        legend.get_title().set_fontweight("normal")
    return legend


def save_figure(
    fig: plt.Figure,
    pdf_path: str | Path,
    png_path: str | Path,
    *,
    dpi: int = 320,
) -> None:
    """Save matching vector PDF and high-resolution PNG outputs."""

    pdf = Path(pdf_path)
    png = Path(png_path)
    pdf.parent.mkdir(parents=True, exist_ok=True)
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf, facecolor="white")
    fig.savefig(png, dpi=dpi, facecolor="white")
