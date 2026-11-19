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
    if data.baseline_replicates is not None:
        rows=data.baseline_replicates
        selected=rows[rows.space.eq(space)&rows.method.eq(method)]
        stats=selected.groupby('time').w2.agg(['mean','std']).reindex(TIMES)
        return stats['mean'].to_numpy(),stats['std'].fillna(0).to_numpy()
    rows = data.baselines.loc[
        data.baselines["space"].eq(space)
        & data.baselines["method"].eq(method)
    ].set_index("time")
    return (
        rows.loc[list(TIMES), "w2"].to_numpy(dtype=float),
        np.zeros(len(TIMES), dtype=float),
    )


def _draw_w2_panel(axis, data, *, space, show_legend):
    width=.23 if space=='gene' else .3
    methods=METHODS_BY_SPACE[space]
    for j,method in enumerate(methods):
        values,errors=_panel_values(data,space=space,method=method)
        positions=np.arange(1,4)+(j-(len(methods)-1)/2)*width
        axis.bar(positions,values,width,color=METHOD_COLORS[method],label=method,
            edgecolor='black',linewidth=.4,yerr=errors,capsize=1.5,error_kw={'elinewidth':.65})
        source=data.replicates if method=='CytoBridge' else data.baseline_replicates
        if source is not None:
            for x,t in zip(positions,TIMES):
                points=source.loc[source.method.eq(method)&source.space.eq(space)&source.time.eq(t),'w2'].to_numpy()
                if len(points)>1:
                    axis.scatter(x+np.linspace(-width*.3,width*.3,len(points)),points,s=3.5,facecolors='none',edgecolors='black',linewidths=.35,zorder=5)
    axis.set_xticks([1,2,3],['T=1','T=2','T=3']);axis.tick_params(labelsize=11)
    axis.spines[['top','right']].set_visible(False)
    if space=='gene':
        axis.set_ylabel('Wasserstein-2 distance',fontsize=13,labelpad=5)
        axis.set_ylim(0,axis.get_ylim()[1]*1.14)
    if show_legend:axis.legend(loc='upper left',frameon=False,fontsize=10,handlelength=1.,labelspacing=.25)

def _write_panel_e_overlay(data, path):
    with mpl.rc_context(FIGURE_2_RC):
        figure=plt.figure(figsize=(PAGE_SIZE[0]/72,PAGE_SIZE[1]/72))
        figure.patch.set_alpha(0)
        figure.add_artist(Rectangle((.347,0),.653,.255,transform=figure.transFigure,facecolor='white',edgecolor='none',zorder=0))
        for space,box,title in [('gene',[.408,.047,.305,.171],'Gene expression space'),('physical',[.772,.047,.210,.171],'Physical space')]:
            axis=figure.add_axes(box)
            _draw_w2_panel(axis,data,space=space,show_legend=space=='gene')
            figure.text(box[0]+box[2]/2,.244,title,fontsize=14,fontweight='bold',ha='center',va='baseline')
        figure.text(.354,.244,'e',fontsize=18,fontweight='bold',ha='left',va='baseline')
        figure.savefig(path,facecolor='none',edgecolor='none')
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
