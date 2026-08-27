"""Raster-preserving page writer for ARISTA Supplementary Figures S17--S22."""

from __future__ import annotations

from pathlib import Path
import shutil
from typing import TYPE_CHECKING, Iterable

import matplotlib.pyplot as plt
from PIL import Image

from ._io import prepare_output_dir

if TYPE_CHECKING:
    from .arista_supplementary_figures import (
        AristaSupplementaryData,
        AristaSupplementaryPage,
    )


def render_arista_supplementary_figures(
    data: "AristaSupplementaryData",
    pages: Iterable["AristaSupplementaryPage"],
    output_dir: str | Path,
) -> dict[str, tuple[Path, Path]]:
    """Write release page payloads without requiring an external layout program."""

    output = prepare_output_dir(output_dir)
    written: dict[str, tuple[Path, Path]] = {}
    for page in pages:
        png_path = output / page.compact_source
        pdf_path = output / f"{Path(page.compact_source).stem}.pdf"
        shutil.copyfile(data.raster_paths[page.figure], png_path)
        with Image.open(data.raster_paths[page.figure]) as image:
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
                    "Title": f"ARISTA Supplementary Figure {page.figure}",
                    "Subject": page.topic,
                    "Creator": "CytoBridge",
                    "Producer": "CytoBridge",
                    "CreationDate": None,
                    "ModDate": None,
                },
            )
            plt.close(figure)
        written[page.figure] = (pdf_path, png_path)
    return written
