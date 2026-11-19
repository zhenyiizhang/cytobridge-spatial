"""Frozen publication style for the scNT A4 figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


A4_PORTRAIT = (8.27, 11.69)
GT_COLOR = "#59616A"
CYTOBRIDGE_COLOR = "#07838B"
ABLATION_COLOR = "#CC6677"
GRID_COLOR = "#D7DDE2"
TEXT_COLOR = "#24313A"
HEADING_COLOR = "#102A43"


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 9.0,
            "axes.titlesize": 9.0,
            "axes.labelsize": 9.0,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
            "legend.fontsize": 9.0,
            "axes.linewidth": 0.65,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "text.color": TEXT_COLOR,
            "axes.labelcolor": TEXT_COLOR,
            "axes.titlecolor": TEXT_COLOR,
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def panel_heading(
    ax, label: str, title: str, *, title_x: float = 0.055, y: float = 0.55
) -> None:
    ax.axis("off")
    ax.text(
        0,
        y,
        label.lower(),
        fontsize=14,
        fontweight="bold",
        va="center",
        ha="left",
        color=HEADING_COLOR,
    )
    ax.text(
        title_x,
        y,
        title,
        fontsize=12,
        fontweight="bold",
        va="center",
        ha="left",
        color=HEADING_COLOR,
    )


def clean_axis(ax, *, grid: bool = True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(color=GRID_COLOR, linewidth=0.45, alpha=0.55, zorder=0)
    else:
        ax.grid(False)
    ax.set_axisbelow(True)


def save_figure(
    fig: plt.Figure, pdf_path: str | Path, png_path: str | Path, *, dpi: int = 320
) -> None:
    pdf = Path(pdf_path)
    png = Path(png_path)
    pdf.parent.mkdir(parents=True, exist_ok=True)
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf, facecolor="white")
    fig.savefig(png, dpi=dpi, facecolor="white")
