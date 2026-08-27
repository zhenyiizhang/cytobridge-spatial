"""Compact reader reproduction for ARISTA Supplementary Figures S17--S22."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
import shutil
import struct
from typing import Any, Iterable
import zlib

import pandas as pd

from ._io import prepare_output_dir, read_json, require_files, resolve_results_dir


FIGURE_ORDER = ("S17", "S18", "S19", "S20", "S21", "S22")
_CORE_FILES = ("manifest.json", "figure_index.csv", "full_recompute_inputs.csv")
_INDEX_COLUMNS = (
    "figure",
    "topic",
    "compact_source",
    "width_pixels",
    "height_pixels",
    "width_points",
    "height_points",
    "reference_dpi",
    "raster_crc32",
    "table_ids",
    "full_calculation_inputs",
)
_REGISTRY_COLUMNS = (
    "input_id",
    "relative_path",
    "kind",
    "stage",
    "figures",
    "public_source",
    "description",
)


@dataclass(frozen=True)
class AristaSupplementaryData:
    """Packaged pages, formal release tables, and external-input registry."""

    source_dir: Path
    manifest: dict[str, Any]
    figure_index: pd.DataFrame
    full_recompute_inputs: pd.DataFrame
    raster_paths: dict[str, Path]
    tables: dict[str, pd.DataFrame]


@dataclass(frozen=True)
class AristaSupplementaryPage:
    """Validated page properties used by the compact renderer."""

    figure: str
    topic: str
    compact_source: str
    width_pixels: int
    height_pixels: int
    width_points: float
    height_points: float
    reference_dpi: int
    raster_crc32: str


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
    posix_path = PurePosixPath(value)
    windows_path = PureWindowsPath(value)
    if (
        posix_path.is_absolute()
        or windows_path.is_absolute()
        or ".." in posix_path.parts
        or not posix_path.parts
    ):
        raise ValueError(f"{source} contains a non-relative input path: {value}")


def _split_items(value: object) -> tuple[str, ...]:
    return tuple(item for item in str(value).split(";") if item)


def _validate_manifest(manifest: dict[str, Any], source: Path) -> None:
    if manifest.get("schema_version") != 1:
        raise ValueError(f"{source} has an unsupported schema version")
    if manifest.get("analysis") != "arista_supplementary_figures":
        raise ValueError(f"{source} does not describe the ARISTA figure set")
    if tuple(manifest.get("figure_set", ())) != FIGURE_ORDER:
        raise ValueError(f"{source} has an unexpected figure order")
    files = manifest.get("files")
    tables = manifest.get("tables")
    if not isinstance(files, dict) or not files:
        raise ValueError(f"{source} has no packaged file registry")
    if not isinstance(tables, dict) or not tables:
        raise ValueError(f"{source} has no table registry")
    for filename in files:
        path = PurePosixPath(str(filename))
        if path.is_absolute() or len(path.parts) != 1:
            raise ValueError(f"{source} contains an invalid packaged filename")


def _validated_page(row: pd.Series, raster_path: Path) -> AristaSupplementaryPage:
    width, height = _png_size(raster_path)
    expected_size = (int(row["width_pixels"]), int(row["height_pixels"]))
    if (width, height) != expected_size:
        raise ValueError(f"{raster_path} dimensions do not match the figure index")
    observed_crc = _crc32(raster_path)
    if observed_crc != str(row["raster_crc32"]):
        raise ValueError(f"{raster_path} does not match the compact page contract")
    width_points = float(row["width_points"])
    height_points = float(row["height_points"])
    reference_dpi = int(row["reference_dpi"])
    if width_points <= 0 or height_points <= 0 or reference_dpi <= 0:
        raise ValueError(f"{raster_path} has invalid page properties")
    return AristaSupplementaryPage(
        figure=str(row["figure"]),
        topic=str(row["topic"]),
        compact_source=str(row["compact_source"]),
        width_pixels=width,
        height_pixels=height,
        width_points=width_points,
        height_points=height_points,
        reference_dpi=reference_dpi,
        raster_crc32=observed_crc,
    )


def load_arista_supplementary_figures(
    results_dir: str | Path | None = None,
) -> AristaSupplementaryData:
    """Load the six compact pages, their formal tables, and input registry."""

    source_dir = resolve_results_dir(
        results_dir,
        slug="arista_supplementary_figures",
    )
    core_paths = require_files(source_dir, _CORE_FILES)
    manifest = read_json(core_paths["manifest.json"])
    _validate_manifest(manifest, core_paths["manifest.json"])
    file_names = tuple(str(name) for name in manifest["files"])
    paths = require_files(source_dir, file_names)

    figure_index = pd.read_csv(paths["figure_index.csv"], keep_default_na=False)
    _require_columns(figure_index, _INDEX_COLUMNS, paths["figure_index.csv"])
    if tuple(figure_index["figure"].astype(str)) != FIGURE_ORDER:
        raise ValueError("ARISTA supplementary figure order must be S17 through S22")
    if figure_index["figure"].duplicated().any():
        raise ValueError("ARISTA supplementary figure identifiers must be unique")

    registry = pd.read_csv(paths["full_recompute_inputs.csv"], keep_default_na=False)
    _require_columns(registry, _REGISTRY_COLUMNS, paths["full_recompute_inputs.csv"])
    if registry["input_id"].duplicated().any():
        raise ValueError("ARISTA external-input identifiers must be unique")
    for value in registry["relative_path"].astype(str):
        _validate_relative_path(value, paths["full_recompute_inputs.csv"])
    invalid_figures = sorted(
        {
            figure
            for value in registry["figures"].astype(str)
            for figure in _split_items(value)
            if figure not in FIGURE_ORDER
        }
    )
    if invalid_figures:
        raise ValueError(f"ARISTA input registry contains unknown figures: {invalid_figures}")
    sources = registry.loc[
        registry["public_source"].astype(str).str.strip().ne(""),
        "public_source",
    ]
    if not sources.astype(str).str.startswith("https://").all():
        raise ValueError("ARISTA public sources must use HTTPS")

    table_specs = manifest["tables"]
    table_ids = set(table_specs)
    known_inputs = set(registry["input_id"].astype(str))
    raster_paths: dict[str, Path] = {}
    for _, row in figure_index.iterrows():
        figure = str(row["figure"])
        source_name = str(row["compact_source"])
        if source_name not in paths or not source_name.endswith(".png"):
            raise ValueError(f"{figure} has an invalid compact page source")
        raster_paths[figure] = paths[source_name]
        _validated_page(row, paths[source_name])
        unknown_tables = sorted(set(_split_items(row["table_ids"])).difference(table_ids))
        if unknown_tables:
            raise ValueError(f"{figure} references unknown tables: {unknown_tables}")
        unknown_inputs = sorted(
            set(_split_items(row["full_calculation_inputs"])).difference(known_inputs)
        )
        if unknown_inputs:
            raise ValueError(f"{figure} references unknown inputs: {unknown_inputs}")

    tables: dict[str, pd.DataFrame] = {}
    for table_id, specification in table_specs.items():
        if not isinstance(specification, dict):
            raise ValueError(f"ARISTA table {table_id} has an invalid specification")
        figure = str(specification.get("figure", ""))
        filename = str(specification.get("file", ""))
        if figure not in FIGURE_ORDER or filename not in paths or not filename.endswith(".csv"):
            raise ValueError(f"ARISTA table {table_id} has an invalid source")
        tables[str(table_id)] = pd.read_csv(paths[filename], keep_default_na=False)

    return AristaSupplementaryData(
        source_dir=source_dir,
        manifest=manifest,
        figure_index=figure_index,
        full_recompute_inputs=registry,
        raster_paths=raster_paths,
        tables=tables,
    )


def calculate_arista_supplementary_pages(
    data: AristaSupplementaryData,
) -> tuple[AristaSupplementaryPage, ...]:
    """Recalculate and validate the properties of all six compact pages."""

    return tuple(
        _validated_page(row, data.raster_paths[str(row["figure"])])
        for _, row in data.figure_index.iterrows()
    )


def select_arista_supplementary_pages(
    pages: Iterable[AristaSupplementaryPage],
    figures: Iterable[str] | None = None,
) -> tuple[AristaSupplementaryPage, ...]:
    """Select current ARISTA figure identifiers while preserving page order."""

    page_map = {page.figure: page for page in pages}
    requested = FIGURE_ORDER if figures is None else tuple(str(value) for value in figures)
    if len(requested) != len(set(requested)):
        raise ValueError("ARISTA figure selection contains duplicates")
    unknown = sorted(set(requested).difference(FIGURE_ORDER))
    if unknown:
        raise ValueError(f"Unknown ARISTA supplementary figures: {unknown}")
    return tuple(page_map[figure] for figure in FIGURE_ORDER if figure in requested)


def write_arista_supplementary_tables(
    data: AristaSupplementaryData,
    pages: Iterable[AristaSupplementaryPage],
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write the page index, input registry, and compact formal tables."""

    output = prepare_output_dir(output_dir)
    selected_pages = tuple(pages)
    page_path = output / "arista_supplementary_page_summary.csv"
    pd.DataFrame(
        [
            {
                "figure": page.figure,
                "topic": page.topic,
                "width_pixels": page.width_pixels,
                "height_pixels": page.height_pixels,
                "width_points": page.width_points,
                "height_points": page.height_points,
                "reference_dpi": page.reference_dpi,
                "raster_crc32": page.raster_crc32,
            }
            for page in selected_pages
        ]
    ).to_csv(page_path, index=False)
    index_path = output / "arista_supplementary_figure_index.csv"
    registry_path = output / "arista_supplementary_full_recompute_inputs.csv"
    data.figure_index.to_csv(index_path, index=False)
    data.full_recompute_inputs.to_csv(registry_path, index=False)
    written = {
        "page_summary": page_path,
        "figure_index": index_path,
        "full_recompute_inputs": registry_path,
    }
    for table_id, specification in data.manifest["tables"].items():
        filename = str(specification["file"])
        destination = output / filename
        shutil.copyfile(data.source_dir / filename, destination)
        written[str(table_id)] = destination
    return written


def plot_arista_supplementary_figures(
    data: AristaSupplementaryData,
    output_dir: str | Path,
    pages: Iterable[AristaSupplementaryPage] | None = None,
    figures: Iterable[str] | None = None,
) -> dict[str, tuple[Path, Path]]:
    """Write exact compact PNGs and raster-equivalent PDFs for selected pages."""

    from ._arista_supplementary_figures_plot import (
        render_arista_supplementary_figures,
    )

    calculated = (
        calculate_arista_supplementary_pages(data)
        if pages is None
        else tuple(pages)
    )
    selected = select_arista_supplementary_pages(calculated, figures)
    return render_arista_supplementary_figures(data, selected, output_dir)


__all__ = [
    "AristaSupplementaryData",
    "AristaSupplementaryPage",
    "FIGURE_ORDER",
    "calculate_arista_supplementary_pages",
    "load_arista_supplementary_figures",
    "plot_arista_supplementary_figures",
    "select_arista_supplementary_pages",
    "write_arista_supplementary_tables",
]
