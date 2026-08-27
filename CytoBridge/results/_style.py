"""Matplotlib settings shared by paper-figure renderers."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


CLASSIFIER_COLORS = {
    1: "#59616A",
    5: "#AAB5BC",
    10: "#07838B",
    20: "#82B8BC",
    50: "#D7DDE2",
}
TEXT_COLOR = "#000000"
GRID_COLOR = "#D7DDE2"
CYTOBRIDGE_COLOR = "#07838B"
COMPARISON_COLOR = "#59616A"

INTERACTION_A4_PORTRAIT = (8.27, 11.69)
INTERACTION_FULL_COLOR = "#07838B"
INTERACTION_NO_LR_COLOR = "#CC6677"
INTERACTION_EXTERNAL_COLOR = "#59616A"
INTERACTION_SPACE_COLORS = {
    "joint": "#4C78A8",
    "spatial": "#E39D2D",
    "state": "#8F63A8",
}
INTERACTION_TEXT_COLOR = "#24313A"
INTERACTION_HEADING_COLOR = "#102A43"

LR_COMPLEX_COLORS = {
    "Zebrafish": "#0072B2",
    "MOSTA": "#009E73",
    "ARISTA": "#CC79A7",
    "Chicken Heart": "#D55E00",
}


CLASSIFIER_RC = {
    "font.family": "Arial",
    "font.size": 9.0,
    "axes.titlesize": 9.0,
    "axes.labelsize": 9.0,
    "axes.linewidth": 0.65,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.axisbelow": True,
    "xtick.labelsize": 9.0,
    "ytick.labelsize": 9.0,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "legend.fontsize": 9.0,
    "legend.frameon": False,
    "lines.linewidth": 1.5,
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
    "savefig.dpi": 320,
}

INTERACTION_RC = {
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
    "text.color": INTERACTION_TEXT_COLOR,
    "axes.labelcolor": INTERACTION_TEXT_COLOR,
    "axes.titlecolor": INTERACTION_TEXT_COLOR,
    "xtick.color": INTERACTION_TEXT_COLOR,
    "ytick.color": INTERACTION_TEXT_COLOR,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
}

LR_COMPLEX_RC = {
    "font.family": "Arial",
    "font.size": 9,
    "axes.titlesize": 12,
    "axes.labelsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.linewidth": 0.65,
    "axes.edgecolor": TEXT_COLOR,
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

LOTO_BENCHMARK_RC = {
    "font.family": "Arial",
    "font.size": 9.0,
    "axes.titlesize": 9.0,
    "axes.labelsize": 9.0,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
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


def clean_axis(ax: "Axes", *, horizontal_only: bool = True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(
        axis="y" if horizontal_only else "both",
        color=GRID_COLOR,
        linewidth=0.45,
        alpha=0.65,
        zorder=0,
    )
    ax.set_axisbelow(True)


def panel_heading(ax: "Axes", label: str, title: str) -> None:
    ax.axis("off")
    ax.text(
        0.0,
        0.52,
        label.lower(),
        fontsize=14,
        fontweight="bold",
        va="center",
        ha="left",
        color="black",
    )
    ax.text(
        0.055,
        0.52,
        title,
        fontsize=12,
        fontweight="bold",
        va="center",
        ha="left",
        color="black",
    )


def interaction_panel_heading(
    ax: "Axes",
    label: str,
    title: str,
    *,
    title_x: float = 0.060,
    y: float = 0.54,
) -> None:
    ax.axis("off")
    ax.text(
        0.0,
        y,
        label.lower(),
        fontsize=14.0,
        fontweight="bold",
        va="center",
        ha="left",
        color=INTERACTION_HEADING_COLOR,
    )
    ax.text(
        title_x,
        y,
        title,
        fontsize=12.0,
        fontweight="bold",
        va="center",
        ha="left",
        color=INTERACTION_HEADING_COLOR,
    )


def save_figure(
    figure: "Figure",
    pdf_path: str | Path,
    png_path: str | Path,
    *,
    dpi: int = 320,
    pdf_metadata: dict[str, str] | None = None,
    png_metadata: dict[str, str] | None = None,
) -> None:
    figure.savefig(Path(pdf_path), facecolor="white", metadata=pdf_metadata)
    figure.savefig(
        Path(png_path),
        dpi=dpi,
        facecolor="white",
        metadata=png_metadata,
    )
