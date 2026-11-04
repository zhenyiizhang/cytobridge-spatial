"""Validate and export the packaged reference page for ARISTA Main Figure 5."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
import struct
from typing import Any
import zlib

import pandas as pd

from ._io import prepare_output_dir, read_json, require_files, resolve_results_dir


PANEL_ORDER = ("a", "b", "c", "d", "e")
OUTPUT_STEM = "Figure5_ARISTA_package_native_scientific_labels"
_FILES = (
    "main_figure_5_compact.png",
    "panel_index.csv",
    "full_recompute_inputs.csv",
    "manifest.json",
)
_PANEL_COLUMNS = (
    "panel",
    "title",
    "content",
    "compact_source",
    "full_calculation_inputs",
)
_REGISTRY_COLUMNS = (
    "input_id",
    "relative_path",
    "kind",
    "stage",
    "panels",
    "public_source",
    "description",
)


@dataclass(frozen=True)
class MainFigure5Data:
    """Packaged reference page and full-calculation input registry."""

    source_dir: Path
    manifest: dict[str, Any]
    raster_path: Path
    panel_index: pd.DataFrame
    full_recompute_inputs: pd.DataFrame


@dataclass(frozen=True)
class MainFigure5Page:
    """Validated properties of the packaged reference page."""

    width_pixels: int
    height_pixels: int
    width_points: float
    height_points: float
    reference_dpi: int
    raster_crc32: str
    panel_count: int


def _png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        header = handle.read(24)
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"Not a PNG file: {path}")
    if header[12:16] != b"IHDR":
        raise ValueError(f"PNG header is missing IHDR: {path}")
    return struct.unpack(">II", header[16:24])


def _crc32(path: Path) -> str:
    value = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value = zlib.crc32(block, value)
    return f"{value & 0xFFFFFFFF:08x}"


def _require_columns(table: pd.DataFrame, columns: tuple[str, ...], source: Path) -> None:
    missing = sorted(set(columns).difference(table.columns))
    if missing:
        raise ValueError(f"{source} is missing columns: {missing}")


def _validate_relative_path(value: str, source: Path) -> None:
    path = PurePosixPath(value)
    windows_path = PureWindowsPath(value)
    if (
        path.is_absolute()
        or windows_path.is_absolute()
        or ".." in path.parts
        or not path.parts
    ):
        raise ValueError(f"{source} contains a non-relative input path: {value}")


def _validate_manifest(
    manifest: dict[str, Any], raster_path: Path
) -> tuple[int, int, float, float, int, str]:
    if manifest.get("schema_version") != 1:
        raise ValueError("Main Figure 5 uses an unsupported schema version")
    if manifest.get("analysis") != "main_figure_5":
        raise ValueError("The packaged manifest does not describe Main Figure 5")
    if manifest.get("reader_action") != "reference-export":
        raise ValueError("Main Figure 5 must be marked as a reference export")
    if manifest.get("numerical_recalculation") is not False:
        raise ValueError("Main Figure 5 must not claim numerical recalculation")
    if manifest.get("scientific_label_release") != "v5":
        raise ValueError("Main Figure 5 does not use the scientific-label v5 release")
    render = manifest.get("render", {})
    canvas = tuple(render.get("canvas_pixels", ()))
    page = tuple(render.get("page_points", ()))
    if len(canvas) != 2 or len(page) != 2:
        raise ValueError("Main Figure 5 is missing page dimensions")
    width, height = _png_size(raster_path)
    if (width, height) != tuple(int(value) for value in canvas):
        raise ValueError("Main Figure 5 raster dimensions do not match the manifest")
    observed_crc = _crc32(raster_path)
    expected_crc = str(render.get("compact_raster_crc32", ""))
    if observed_crc != expected_crc:
        raise ValueError("Main Figure 5 raster digest does not match the manifest")
    reference_dpi = int(render.get("reference_dpi", 0))
    if reference_dpi <= 0:
        raise ValueError("Main Figure 5 reference DPI must be positive")
    labels = manifest.get("scientific_labels", {})
    expected_labels = {
        "Spatial migration velocity": "Spatial velocity",
        "Spatial velocity cosine simlarity": "Spatial velocity cosine similarity",
        "interaction VS migration": "interaction vs full spatial velocity",
    }
    if labels != expected_labels:
        raise ValueError("Main Figure 5 scientific-label mapping changed")
    return width, height, float(page[0]), float(page[1]), reference_dpi, observed_crc


def load_main_figure_5(results_dir: str | Path | None = None) -> MainFigure5Data:
    """Load the packaged page raster, panel index, and full-input registry."""

    source_dir = resolve_results_dir(results_dir, slug="main_figure_5")
    paths = require_files(source_dir, _FILES)
    manifest = read_json(paths["manifest.json"])
    _validate_manifest(manifest, paths["main_figure_5_compact.png"])

    panel_index = pd.read_csv(paths["panel_index.csv"], keep_default_na=False)
    _require_columns(panel_index, _PANEL_COLUMNS, paths["panel_index.csv"])
    if tuple(panel_index["panel"].astype(str)) != PANEL_ORDER:
        raise ValueError("Main Figure 5 panel order must be a through e")
    if panel_index["panel"].duplicated().any():
        raise ValueError("Main Figure 5 contains duplicate panel labels")
    if not panel_index["compact_source"].eq("main_figure_5_compact.png").all():
        raise ValueError("Main Figure 5 panel rows use an unexpected compact source")

    registry = pd.read_csv(paths["full_recompute_inputs.csv"], keep_default_na=False)
    _require_columns(registry, _REGISTRY_COLUMNS, paths["full_recompute_inputs.csv"])
    if registry["input_id"].duplicated().any():
        raise ValueError("Main Figure 5 input registry contains duplicate identifiers")
    for value in registry["relative_path"].astype(str):
        _validate_relative_path(value, paths["full_recompute_inputs.csv"])
    invalid_panels = sorted(
        {
            panel
            for value in registry["panels"].astype(str)
            for panel in value.split(";")
            if panel not in PANEL_ORDER
        }
    )
    if invalid_panels:
        raise ValueError(f"Main Figure 5 registry contains unknown panels: {invalid_panels}")
    urls = registry.loc[registry["public_source"].astype(bool), "public_source"]
    if not urls.astype(str).str.startswith("https://").all():
        raise ValueError("Main Figure 5 public sources must use HTTPS")

    return MainFigure5Data(
        source_dir=source_dir,
        manifest=manifest,
        raster_path=paths["main_figure_5_compact.png"],
        panel_index=panel_index,
        full_recompute_inputs=registry,
    )


def validate_main_figure_5_reference_page(
    data: MainFigure5Data,
) -> MainFigure5Page:
    """Validate the packaged page and return its recorded dimensions."""

    width, height, width_pt, height_pt, dpi, raster_crc = _validate_manifest(
        data.manifest, data.raster_path
    )
    required_inputs = {
        item
        for value in data.panel_index["full_calculation_inputs"].astype(str)
        for item in value.split(";")
    }
    known_inputs = set(data.full_recompute_inputs["input_id"].astype(str))
    unknown = sorted(required_inputs.difference(known_inputs))
    if unknown:
        raise ValueError(f"Main Figure 5 panel index references unknown inputs: {unknown}")
    return MainFigure5Page(
        width_pixels=width,
        height_pixels=height,
        width_points=width_pt,
        height_points=height_pt,
        reference_dpi=dpi,
        raster_crc32=raster_crc,
        panel_count=len(data.panel_index),
    )


def write_main_figure_5_tables(
    data: MainFigure5Data,
    page: MainFigure5Page,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write the panel mapping, external-input registry, and page summary."""

    output = prepare_output_dir(output_dir)
    panel_path = output / "main_figure_5_panel_index.csv"
    registry_path = output / "main_figure_5_full_recompute_inputs.csv"
    page_path = output / "main_figure_5_page.csv"
    data.panel_index.to_csv(panel_path, index=False)
    data.full_recompute_inputs.to_csv(registry_path, index=False)
    pd.DataFrame(
        [
            {
                "width_pixels": page.width_pixels,
                "height_pixels": page.height_pixels,
                "width_points": page.width_points,
                "height_points": page.height_points,
                "reference_dpi": page.reference_dpi,
                "raster_crc32": page.raster_crc32,
                "panel_count": page.panel_count,
            }
        ]
    ).to_csv(page_path, index=False)
    return {"panels": panel_path, "inputs": registry_path, "page": page_path}


def export_main_figure_5_reference_page(
    data: MainFigure5Data,
    output_dir: str | Path,
    page: MainFigure5Page | None = None,
) -> tuple[Path, Path]:
    """Export the packaged PNG and a raster-equivalent A4 PDF."""

    from ._main_figure_5_plot import render_main_figure_5

    return render_main_figure_5(
        data,
        page or validate_main_figure_5_reference_page(data),
        output_dir,
    )


def calculate_main_figure_5(data: MainFigure5Data) -> MainFigure5Page:
    """Compatibility alias for :func:`validate_main_figure_5_reference_page`."""

    return validate_main_figure_5_reference_page(data)


def plot_main_figure_5(
    data: MainFigure5Data,
    output_dir: str | Path,
    page: MainFigure5Page | None = None,
) -> tuple[Path, Path]:
    """Compatibility alias for :func:`export_main_figure_5_reference_page`."""

    return export_main_figure_5_reference_page(data, output_dir, page)


__all__ = [
    "MainFigure5Data",
    "MainFigure5Page",
    "PANEL_ORDER",
    "calculate_main_figure_5",
    "export_main_figure_5_reference_page",
    "load_main_figure_5",
    "plot_main_figure_5",
    "validate_main_figure_5_reference_page",
    "write_main_figure_5_tables",
]
