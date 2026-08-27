"""Raster-preserving reference-page exporter for ARISTA Main Figure 5."""

from __future__ import annotations

from pathlib import Path
import shutil
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
from PIL import Image

from ._io import prepare_output_dir
from .main_figure_5 import OUTPUT_STEM

if TYPE_CHECKING:
    from .main_figure_5 import MainFigure5Data, MainFigure5Page


def render_main_figure_5(
    data: "MainFigure5Data",
    page: "MainFigure5Page",
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """Write the packaged page without requiring an external layout program."""

    output = prepare_output_dir(output_dir)
    pdf_path = output / f"{OUTPUT_STEM}.pdf"
    png_path = output / f"{OUTPUT_STEM}.png"
    shutil.copyfile(data.raster_path, png_path)

    with Image.open(data.raster_path) as image:
        raster = image.convert("RGB")
        figure = plt.figure(
            figsize=(page.width_points / 72.0, page.height_points / 72.0),
            facecolor="white",
        )
        axis = figure.add_axes([0.0, 0.0, 1.0, 1.0])
        axis.imshow(raster, interpolation="none", aspect="auto")
        axis.set_axis_off()
        figure.savefig(
            pdf_path,
            format="pdf",
            dpi=page.reference_dpi,
            facecolor="white",
            edgecolor="none",
            metadata={
                "Title": "ARISTA Main Figure 5",
                "Subject": "Packaged scientific-label reference page",
                "Creator": "CytoBridge",
                "Producer": "CytoBridge",
                "CreationDate": None,
                "ModDate": None,
            },
        )
        plt.close(figure)
    return pdf_path, png_path
