"""Reference-page export and table-driven ARISTA ligand--receptor plots."""

from __future__ import annotations

from pathlib import Path
import math
import shutil
from typing import TYPE_CHECKING, Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from ._io import prepare_output_dir

if TYPE_CHECKING:
    from .arista_supplementary_figures import (
        AristaLigandReceptorPanels,
        AristaSupplementaryData,
        AristaSupplementaryPage,
    )


_S21_COLORS = {1: "#66c2a5", 2: "#fc8d62"}
_S22_COLORS = {1: "#1f77b4", 2: "#ff7f0e"}
_TIME_POINTS = np.arange(0.0, 4.0 + 0.5, 0.5)


def _require_columns(table: pd.DataFrame, columns: set[str], name: str) -> None:
    missing = sorted(columns.difference(table.columns))
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


def _save_plot(figure: plt.Figure, stem: Path, *, dpi: int) -> tuple[Path, Path]:
    pdf_path = stem.with_suffix(".pdf")
    png_path = stem.with_suffix(".png")
    metadata = {
        "Title": stem.name,
        "Creator": "CytoBridge",
        "Producer": "CytoBridge",
        "CreationDate": None,
        "ModDate": None,
    }
    figure.savefig(
        pdf_path,
        format="pdf",
        bbox_inches="tight",
        facecolor="white",
        metadata=metadata,
    )
    figure.savefig(
        png_path,
        format="png",
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)
    return pdf_path, png_path


def _plot_s21(prototypes: pd.DataFrame, output: Path) -> tuple[Path, Path]:
    _require_columns(
        prototypes,
        {"cluster", "time", "mean_normalized_score", "n_pairs"},
        "ligand_receptor_cluster_prototypes",
    )
    counts = (
        prototypes.groupby("cluster", sort=True)["n_pairs"].first().astype(int).to_dict()
    )
    if counts != {1: 217, 2: 314}:
        raise ValueError(f"Expected corrected ARISTA cluster sizes 217 and 314, found {counts}")
    if prototypes.groupby("cluster").size().to_dict() != {1: 9, 2: 9}:
        raise ValueError("Each corrected ARISTA prototype must contain nine time points")

    figure, axis = plt.subplots(figsize=(8.0, 4.5))
    for cluster, subset in prototypes.groupby("cluster", sort=True):
        subset = subset.sort_values("time")
        x = subset["time"].to_numpy(dtype=float)
        if not np.allclose(x, _TIME_POINTS, rtol=0.0, atol=1e-12):
            raise ValueError(f"Cluster {cluster} has an unexpected time grid")
        mean = subset["mean_normalized_score"].to_numpy(dtype=float)
        color = _S21_COLORS[int(cluster)]
        axis.plot(
            x,
            mean,
            marker="o",
            markersize=5,
            linewidth=2.6,
            color=color,
            label=f"Cluster {int(cluster)} (n={counts[int(cluster)]})",
        )
        axis.fill_between(x, 0, mean, color=color, alpha=0.12)
    axis.set_title("Communication pattern prototypes")
    axis.set_xlabel("Time")
    axis.set_ylabel("Mean normalized score")
    axis.set_ylim(0, 1.02)
    axis.grid(True, axis="y", alpha=0.2)
    axis.legend(frameon=False)
    figure.tight_layout()
    return _save_plot(figure, output / "FigureS21_ARISTA_redrawn", dpi=300)


def _plot_s22(
    roster: pd.DataFrame,
    timecourse: pd.DataFrame,
    output: Path,
) -> tuple[Path, Path]:
    _require_columns(
        roster,
        {"pair", "cluster", "display_order"},
        "ligand_receptor_display_roster",
    )
    _require_columns(
        timecourse,
        {"pair", "cluster", "time", "score", "display_order"},
        "ligand_receptor_pair_timecourse",
    )
    roster = roster.sort_values("display_order", kind="mergesort").copy()
    counts = roster.groupby("cluster").size().astype(int).to_dict()
    if len(roster) != 50 or counts != {1: 25, 2: 25}:
        raise ValueError(
            "Corrected ARISTA S22 requires 50 pairs, with 25 from each cluster"
        )
    if roster["pair"].nunique() != 50 or len(timecourse) != 450:
        raise ValueError("Corrected ARISTA S22 requires 50 unique pairs and 450 rows")

    columns = 5
    rows = int(math.ceil(len(roster) / columns))
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(3.6 * columns, 2.15 * rows),
        squeeze=False,
    )
    for axis in axes.flat:
        axis.axis("off")
    for axis, record in zip(axes.flat, roster.itertuples(index=False)):
        subset = timecourse.loc[timecourse["pair"].eq(record.pair)].sort_values("time")
        x = subset["time"].to_numpy(dtype=float)
        y = subset["score"].to_numpy(dtype=float)
        if len(subset) != 9 or not np.allclose(
            x, _TIME_POINTS, rtol=0.0, atol=1e-12
        ):
            raise ValueError(f"Incomplete corrected time course for {record.pair}")
        axis.axis("on")
        axis.set_title(str(record.pair), fontsize=8)
        axis.grid(True, axis="y", alpha=0.2)
        axis.set_xlabel("Time", fontsize=8)
        axis.set_ylabel("Score", fontsize=8)
        axis.tick_params(axis="both", labelsize=7)
        color = _S22_COLORS[int(record.cluster)]
        dense_x = np.linspace(x.min(), x.max(), 300)
        axis.plot(dense_x, np.interp(dense_x, x, y), color=color, linewidth=1.8)
        axis.scatter(x, y, color=color, s=12)
    figure.suptitle(
        "Representative LR pair trends (25 per pattern; n=50)",
        fontsize=13,
        y=0.995,
    )
    figure.tight_layout()
    return _save_plot(figure, output / "FigureS22_ARISTA_redrawn", dpi=180)


def render_arista_ligand_receptor_figures(
    panels: "AristaLigandReceptorPanels",
    output_dir: str | Path,
) -> dict[str, tuple[Path, Path]]:
    """Draw corrected S21 and S22 from tables calculated during this run."""

    output = prepare_output_dir(output_dir)
    style = {
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "font.family": "Arial",
        "font.sans-serif": ["Arial"],
        "font.size": 10,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
    with mpl.rc_context(style):
        s21 = _plot_s21(panels.prototypes, output)
        s22 = _plot_s22(
            panels.display_roster,
            panels.display_timecourse,
            output,
        )
    return {"S21": s21, "S22": s22}


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
