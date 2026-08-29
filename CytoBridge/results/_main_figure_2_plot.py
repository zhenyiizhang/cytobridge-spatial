"""Assembler for Main Figure 2."""

from __future__ import annotations

from pathlib import Path
import tempfile

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from ._io import prepare_output_dir
from .main_figure_2 import METHODS_BY_SPACE, TIMES, MainFigure2Data


PAGE_SIZE = (595.276, 841.89)
OVERLAY_PAGE_SIZE = (594.72, 841.68)
METHOD_COLORS = {
    "STORIES": "#E1CCBB",
    "stVCR": "#C18071",
    "CytoBridge": "#8B3842",
}
FIGURE_2_RC = {
    "font.family": "Arial",
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 300,
}


def _add_white_box(figure: mpl.figure.Figure) -> None:
    page_width, page_height = PAGE_SIZE
    figure.add_artist(
        Rectangle(
            (198.0 / page_width, 0.0),
            (page_width - 198.0) / page_width,
            225.0 / page_height,
            transform=figure.transFigure,
            facecolor="white",
            edgecolor="none",
            zorder=0,
        )
    )


def _add_axis(
    figure: mpl.figure.Figure, box: tuple[float, float, float, float]
) -> plt.Axes:
    page_width, page_height = PAGE_SIZE
    x, y, width, height = box
    axis = figure.add_axes(
        [x / page_width, y / page_height, width / page_width, height / page_height],
        zorder=2,
    )
    axis.set_facecolor("none")
    return axis


def _panel_values(
    data: MainFigure2Data, *, space: str, method: str
) -> tuple[np.ndarray, np.ndarray]:
    if method == "CytoBridge":
        rows = data.summary.loc[data.summary["space"].eq(space)].set_index("time")
        return (
            rows.loc[list(TIMES), "mean_w2"].to_numpy(dtype=float),
            rows.loc[list(TIMES), "sd_w2"].to_numpy(dtype=float),
        )
    rows = data.baselines.loc[
        data.baselines["space"].eq(space)
        & data.baselines["method"].eq(method)
    ].set_index("time")
    return (
        rows.loc[list(TIMES), "w2"].to_numpy(dtype=float),
        np.zeros(len(TIMES), dtype=float),
    )


def _draw_w2_panel(
    axis: plt.Axes,
    data: MainFigure2Data,
    *,
    space: str,
    show_legend: bool,
) -> None:
    times = np.asarray(TIMES, dtype=float)
    methods = METHODS_BY_SPACE[space]
    width = 0.22 if len(methods) == 3 else 0.28
    offsets = (np.arange(len(methods)) - (len(methods) - 1) / 2) * width
    random = np.random.default_rng(20260811)

    for index, method in enumerate(methods):
        x_positions = times + offsets[index]
        values, errors = _panel_values(data, space=space, method=method)
        axis.bar(
            x_positions,
            values,
            width=width * 0.88,
            color=METHOD_COLORS[method],
            edgecolor="#2A2A2A",
            linewidth=0.35,
            label=method,
            zorder=2,
        )
        if method != "CytoBridge":
            continue
        axis.errorbar(
            x_positions,
            values,
            yerr=errors,
            fmt="none",
            ecolor="#282828",
            elinewidth=0.65,
            capsize=1.8,
            capthick=0.65,
            zorder=5,
        )
        for point_index, time in enumerate(TIMES):
            replicates = data.replicates.loc[
                data.replicates["space"].eq(space)
                & data.replicates["time"].eq(time),
                "w2",
            ].to_numpy(dtype=float)
            jitter = random.uniform(
                -width * 0.16, width * 0.16, size=replicates.size
            )
            axis.scatter(
                np.full(replicates.size, x_positions[point_index]) + jitter,
                replicates,
                s=2.5,
                facecolor="white",
                edgecolor=METHOD_COLORS["CytoBridge"],
                linewidth=0.35,
                zorder=6,
            )

    axis.set_xticks(times, [f"T={time}" for time in TIMES])
    axis.set_title(
        "Gene expression space" if space == "gene" else "Physical space", pad=3
    )
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(length=2, width=0.55, pad=1.5)
    axis.grid(False)
    if show_legend:
        axis.legend(
            frameon=False,
            loc="upper left",
            handlelength=0.8,
            handletextpad=0.35,
            borderpad=0,
        )


def _write_panel_e_overlay(data: MainFigure2Data, path: Path) -> None:
    page_width, page_height = PAGE_SIZE
    with mpl.rc_context(FIGURE_2_RC):
        figure = plt.figure(
            figsize=(OVERLAY_PAGE_SIZE[0] / 72, OVERLAY_PAGE_SIZE[1] / 72)
        )
        figure.patch.set_alpha(0)
        _add_white_box(figure)
        gene_axis = _add_axis(figure, (231, 42, 196, 145))
        physical_axis = _add_axis(figure, (446, 42, 135, 145))
        _draw_w2_panel(gene_axis, data, space="gene", show_legend=True)
        _draw_w2_panel(physical_axis, data, space="physical", show_legend=False)
        gene_axis.set_ylabel("Wasserstein-2 distance", labelpad=2)
        physical_axis.set_yticklabels([])
        physical_axis.tick_params(axis="y", length=0)
        figure.text(
            198 / page_width,
            215 / page_height,
            "e",
            fontsize=16,
            fontweight="bold",
        )
        figure.savefig(
            path,
            format="pdf",
            facecolor="none",
            edgecolor="none",
            metadata={
                "CreationDate": None,
                "ModDate": None,
                "Creator": "CytoBridge",
                "Producer": "Matplotlib",
            },
        )
        plt.close(figure)


def _fitz():
    try:
        import pymupdf
    except ImportError as error:  # pragma: no cover - optional dependency
        raise ImportError(
            "Main Figure 2 plotting requires PyMuPDF. Install CytoBridge[plot]."
        ) from error
    return pymupdf


def assemble_main_figure_2(
    data: MainFigure2Data, output_dir: str | Path, *, dpi: int = 300
) -> tuple[Path, Path]:
    """Combine frozen panels a--d with a redrawn panel e."""

    if dpi <= 0:
        raise ValueError("dpi must be positive")
    output = prepare_output_dir(output_dir)
    pdf_path = output / "main_figure_2.pdf"
    png_path = output / "main_figure_2.png"
    fitz = _fitz()

    with tempfile.TemporaryDirectory(prefix="cytobridge-main-figure-2-") as temporary:
        overlay_path = Path(temporary) / "panel_e.pdf"
        _write_panel_e_overlay(data, overlay_path)
        document = fitz.open(data.frozen_panels_pdf)
        overlay = fitz.open(overlay_path)
        try:
            if document.page_count != 1 or overlay.page_count != 1:
                raise ValueError("Main Figure 2 inputs must each contain one page")
            page = document[0]
            if not (
                np.isclose(page.rect.width, PAGE_SIZE[0], atol=0.02)
                and np.isclose(page.rect.height, PAGE_SIZE[1], atol=0.02)
            ):
                raise ValueError(f"Unexpected Main Figure 2 page size: {page.rect}")
            page.show_pdf_page(
                page.rect, overlay, 0, keep_proportion=False, overlay=True
            )
            document.save(
                pdf_path, garbage=4, deflate=True, no_new_id=True
            )
        finally:
            overlay.close()
            document.close()

    document = fitz.open(pdf_path)
    try:
        pixmap = document[0].get_pixmap(
            matrix=fitz.Matrix(dpi / 72, dpi / 72), alpha=False
        )
        pixmap.save(png_path)
    finally:
        document.close()
    return pdf_path, png_path


def plot_main_figure_2(
    data: MainFigure2Data, output_dir: str | Path, *, dpi: int = 300
) -> tuple[Path, Path]:
    """Compatibility alias for :func:`assemble_main_figure_2`."""

    return assemble_main_figure_2(data, output_dir, dpi=dpi)
