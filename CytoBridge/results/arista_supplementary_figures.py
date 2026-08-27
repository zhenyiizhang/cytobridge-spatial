"""Compact reader reproduction for ARISTA Supplementary Figures S19--S24."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import shutil
import struct
from typing import Any, Iterable
import zlib

import numpy as np
import pandas as pd

from ._io import prepare_output_dir, read_json, require_files, resolve_results_dir


FIGURE_ORDER = ("S19", "S20", "S21", "S22", "S23", "S24")
ARISTA_RELEASE_DIRECTORY = "arista_package_native_spatialqc_z50_retrain_20260824_r1"
ARISTA_RELEASE_ENVIRONMENT_VARIABLE = "CYTOBRIDGE_ARISTA_RELEASE_DIR"
_CORE_FILES = ("manifest.json", "figure_index.csv", "full_recompute_inputs.csv")
_INDEX_COLUMNS = (
    "figure",
    "source_figure",
    "topic",
    "compact_source",
    "output_filename",
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
_LR_TABLE_IDS = (
    "ligand_receptor_all_pair_timecourse",
    "ligand_receptor_cluster_prototypes",
    "ligand_receptor_cluster_assignments",
    "ligand_receptor_normalized_profiles",
    "ligand_receptor_k_selection",
    "ligand_receptor_cluster_diagnostics",
    "ligand_receptor_display_roster",
    "ligand_receptor_pair_timecourse",
)

_CANONICAL_SCRIPT_DIRECTORY = "scripts/reviewer_arista_20260824"
_CALCULATION_ENTRYPOINTS = (
    "scripts/run_spatiotemporal_downstream.py",
    "CytoBridge/configs/arista_downstream.yaml",
)
_CHECKPOINT_INPUTS = (
    "main_run/provenance/config.yaml",
    "main_run/provenance/training_run_summary.json",
    "main_run/training/Finetune/best_model.pth",
    "main_run/training/Score_Refine/score_model.pth",
)
_SLICE_INPUTS = tuple(
    f"main_run/downstream/slice_data/time_{token}.h5ad"
    for token in ("0", "0p5", "1", "1p5", "2", "2p5", "3", "3p5", "4")
)
_S12_DIRECTORY = "S12_package_native_warpk1_oldstyle_v3_legacy_palette"
_S13_S14_DIRECTORY = "S13_S14_package_native_oldstyle_v3_legacy_palette"
_S15_S17_DIRECTORY = "S15_S17_package_native_strict_oldstyle_v1"
_S16_RECLUSTER_DIRECTORY = "S16_lr_kmeans_recluster_v1"
_S16_KMEANS_DIRECTORY = "S16_package_native_kmeans_oldstyle_v1_finalqa"
_S17_BALANCED_DIRECTORY = "S17_package_native_balanced25_oldstyle_v1"
_FORMAL_SOURCE_SPECS = (
    {
        "figure": "S19",
        "release_figure": "S12",
        "topic": "Spatial interpolation",
        "formal_pdf": (
            f"{_S12_DIRECTORY}/figures/pdf/"
            "FigureS12_ARISTA_package_native_warpk1_oldstyle_FINAL.pdf"
        ),
        "formal_svg": (
            f"{_S12_DIRECTORY}/figures/vector/"
            "FigureS12_ARISTA_package_native_warpk1_oldstyle_FINAL.svg"
        ),
        "formal_png": (
            f"{_S12_DIRECTORY}/figures/png/"
            "FigureS12_ARISTA_package_native_warpk1_oldstyle_FINAL.png"
        ),
        "vector_scope": "full-page PDF and SVG",
        "canonical_scripts": ";".join(
            (
                f"{_CANONICAL_SCRIPT_DIRECTORY}/"
                "build_s12_package_native_warpk1_oldstyle.py",
                f"{_CANONICAL_SCRIPT_DIRECTORY}/"
                "build_s12_s14_legacy_style_corrected.py",
            )
        ),
        "release_build_snapshot": (
            f"{_S12_DIRECTORY}/scripts/build_s12_package_native_warpk1_oldstyle.py"
        ),
        "build_scope": (
            "release snapshot is the exact page builder; repository copy matches it"
        ),
        "release_manifest": f"{_S12_DIRECTORY}/MANIFEST.json",
        "downstream_inputs": ";".join(
            (
                *_SLICE_INPUTS,
                "main_run/preprocess/aligned_cell_identity.csv",
                "main_run/downstream/label_to_color.json",
                f"{_S12_DIRECTORY}/tables/s12_panel_inventory_complete_display.csv",
            )
        ),
        "input_scope": (
            "release retains page slices and a derived table; spatial-warp "
            "intermediates are external"
        ),
    },
    {
        "figure": "S20",
        "release_figure": "S13",
        "topic": "Growth",
        "formal_pdf": (
            f"{_S13_S14_DIRECTORY}/figures/pdf/"
            "FigureS13_ARISTA_package_native_oldstyle_FINAL.pdf"
        ),
        "formal_svg": (
            f"{_S13_S14_DIRECTORY}/figures/vector/"
            "FigureS13_ARISTA_package_native_oldstyle_FINAL.svg"
        ),
        "formal_png": (
            f"{_S13_S14_DIRECTORY}/figures/png/"
            "FigureS13_ARISTA_package_native_oldstyle_FINAL.png"
        ),
        "vector_scope": "full-page PDF and SVG with nine embedded raster layers",
        "canonical_scripts": ";".join(
            (
                f"{_CANONICAL_SCRIPT_DIRECTORY}/"
                "build_s13_s14_package_native_oldstyle.py",
                f"{_CANONICAL_SCRIPT_DIRECTORY}/"
                "build_s12_s14_legacy_style_corrected.py",
            )
        ),
        "release_build_snapshot": (
            f"{_S13_S14_DIRECTORY}/scripts/build_s13_s14_package_native_oldstyle.py"
        ),
        "build_scope": (
            "release snapshot is the exact page builder; repository script adds "
            "optional display settings"
        ),
        "release_manifest": f"{_S13_S14_DIRECTORY}/MANIFEST.json",
        "downstream_inputs": ";".join(
            (
                "main_run/downstream/growth/growth_by_cell.csv",
                "main_run/downstream/label_to_color.json",
                f"{_S13_S14_DIRECTORY}/tables/"
                "s13_scale_and_display_counts_A_all_valid.csv",
                f"{_S13_S14_DIRECTORY}/tables/s13_seed42_display_sample.csv",
            )
        ),
        "input_scope": (
            "release retains growth and display tables; spatial-warp "
            "intermediates are external"
        ),
    },
    {
        "figure": "S21",
        "release_figure": "S14",
        "topic": "Lineage and composition",
        "formal_pdf": (
            f"{_S13_S14_DIRECTORY}/figures/pdf/"
            "FigureS14_ARISTA_package_native_oldstyle_FINAL.pdf"
        ),
        "formal_svg": (
            f"{_S13_S14_DIRECTORY}/figures/vector/"
            "FigureS14_ARISTA_package_native_oldstyle_FINAL.svg"
        ),
        "formal_png": (
            f"{_S13_S14_DIRECTORY}/figures/png/"
            "FigureS14_ARISTA_package_native_oldstyle_FINAL.png"
        ),
        "vector_scope": "full-page PDF and SVG",
        "canonical_scripts": ";".join(
            (
                f"{_CANONICAL_SCRIPT_DIRECTORY}/"
                "build_s13_s14_package_native_oldstyle.py",
                f"{_CANONICAL_SCRIPT_DIRECTORY}/"
                "build_s12_s14_legacy_style_corrected.py",
            )
        ),
        "release_build_snapshot": (
            f"{_S13_S14_DIRECTORY}/scripts/build_s13_s14_package_native_oldstyle.py"
        ),
        "build_scope": (
            "release snapshot is the exact page builder; repository script adds "
            "optional display settings"
        ),
        "release_manifest": f"{_S13_S14_DIRECTORY}/MANIFEST.json",
        "downstream_inputs": ";".join(
            (
                f"{_S13_S14_DIRECTORY}/tables/s14_fixed_particle_counts.csv",
                f"{_S13_S14_DIRECTORY}/tables/s14_fixed_particle_fractions.csv",
                f"{_S13_S14_DIRECTORY}/tables/"
                "s14b_corrected_top15_other_percent.csv",
            )
        ),
        "input_scope": (
            "release retains derived fixed-particle tables; the upstream "
            "fixed-particle file is external"
        ),
    },
    {
        "figure": "S22",
        "release_figure": "S15",
        "topic": "Gene programs and GO enrichment",
        "formal_pdf": (
            f"{_S15_S17_DIRECTORY}/figures/"
            "FigureS15_ARISTA_strict_corrected_legacy_style.pdf"
        ),
        "formal_svg": ";".join(
            f"{_S15_S17_DIRECTORY}/figures/panels/{name}"
            for name in (
                "S15a_top_variable_gene_trajectories.svg",
                "S15b_gene_pattern_curves.svg",
                "S15c_pattern_1_GO_barplot.svg",
                "S15d_pattern_2_GO_dotplot.svg",
            )
        ),
        "formal_png": (
            f"{_S15_S17_DIRECTORY}/figures/"
            "FigureS15_ARISTA_strict_corrected_legacy_style.png"
        ),
        "vector_scope": (
            "raster composite PDF; four retained panel SVGs, two with an "
            "embedded raster layer"
        ),
        "canonical_scripts": (
            f"{_CANONICAL_SCRIPT_DIRECTORY}/build_s15_s17_strict_legacy_style.py"
        ),
        "release_build_snapshot": "",
        "build_scope": "repository builder; no duplicate builder in the release",
        "release_manifest": f"{_S15_S17_DIRECTORY}/MANIFEST.json",
        "downstream_inputs": ";".join(
            f"{_S15_S17_DIRECTORY}/provenance/source_snapshots/{name}"
            for name in (
                "bank_summary.json",
                "gene_mean_expression.csv",
                "gene_reconstruction_diagnostics.csv",
            )
        ),
        "input_scope": "release retains the builder source snapshots",
    },
    {
        "figure": "S23",
        "release_figure": "S16",
        "topic": "Ligand-receptor clusters",
        "formal_pdf": (
            f"{_S16_KMEANS_DIRECTORY}/figures/"
            "FigureS16_ARISTA_package_native_kmeans_legacy_style.pdf"
        ),
        "formal_svg": (
            f"{_S16_KMEANS_DIRECTORY}/figures/"
            "FigureS16_ARISTA_package_native_kmeans_legacy_style.svg"
        ),
        "formal_png": (
            f"{_S16_KMEANS_DIRECTORY}/figures/"
            "FigureS16_ARISTA_package_native_kmeans_legacy_style.png"
        ),
        "vector_scope": "full-page PDF and SVG",
        "canonical_scripts": ";".join(
            (
                f"{_CANONICAL_SCRIPT_DIRECTORY}/recluster_arista_lr_patterns.py",
                f"{_CANONICAL_SCRIPT_DIRECTORY}/build_s16_kmeans_legacy_style.py",
            )
        ),
        "release_build_snapshot": (
            f"{_S16_KMEANS_DIRECTORY}/build_s16_kmeans_legacy_style.py"
        ),
        "build_scope": "release snapshot is the exact page builder",
        "release_manifest": f"{_S16_KMEANS_DIRECTORY}/MANIFEST.json",
        "downstream_inputs": ";".join(
            (
                f"{_S15_S17_DIRECTORY}/provenance/source_snapshots/"
                "lr_pair_timecourse.csv",
                f"{_S16_RECLUSTER_DIRECTORY}/S16_lr_kmeans_assignments.csv",
                f"{_S16_RECLUSTER_DIRECTORY}/S16_lr_kmeans_prototypes.csv",
                f"{_S16_RECLUSTER_DIRECTORY}/S16_lr_k_selection.csv",
            )
        ),
        "input_scope": (
            "release retains the strict pair scores, corrected deterministic "
            "k-means assignments, prototypes, and k-selection table"
        ),
    },
    {
        "figure": "S24",
        "release_figure": "S17",
        "topic": "Ligand-receptor small multiples",
        "formal_pdf": (
            f"{_S17_BALANCED_DIRECTORY}/figures/"
            "FigureS17_ARISTA_package_native_balanced_representative_legacy_style.pdf"
        ),
        "formal_svg": (
            f"{_S17_BALANCED_DIRECTORY}/figures/"
            "FigureS17_ARISTA_package_native_balanced_representative_legacy_style.svg"
        ),
        "formal_png": (
            f"{_S17_BALANCED_DIRECTORY}/figures/"
            "FigureS17_ARISTA_package_native_balanced_representative_legacy_style.png"
        ),
        "vector_scope": "full-page PDF and SVG",
        "canonical_scripts": ";".join(
            (
                f"{_CANONICAL_SCRIPT_DIRECTORY}/recluster_arista_lr_patterns.py",
                f"{_CANONICAL_SCRIPT_DIRECTORY}/"
                "build_s17_balanced_representative_legacy_style.py",
            )
        ),
        "release_build_snapshot": (
            f"{_S17_BALANCED_DIRECTORY}/"
            "build_s17_balanced_representative_legacy_style.py"
        ),
        "build_scope": "release snapshot is the exact page builder",
        "release_manifest": f"{_S17_BALANCED_DIRECTORY}/MANIFEST.json",
        "downstream_inputs": ";".join(
            (
                f"{_S15_S17_DIRECTORY}/provenance/source_snapshots/"
                "lr_pair_timecourse.csv",
                f"{_S16_RECLUSTER_DIRECTORY}/S16_lr_kmeans_assignments.csv",
                f"{_S16_RECLUSTER_DIRECTORY}/S16_lr_kmeans_normalized_profiles.csv",
                f"{_S17_BALANCED_DIRECTORY}/tables/"
                "S17_balanced_representative_roster.csv",
                f"{_S17_BALANCED_DIRECTORY}/tables/"
                "S17_balanced_representative_timecourse.csv",
            )
        ),
        "input_scope": (
            "release retains the corrected assignments, normalized profiles, "
            "balanced display roster, and all displayed time courses"
        ),
    },
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
    source_figure: str
    topic: str
    compact_source: str
    output_filename: str
    width_pixels: int
    height_pixels: int
    width_points: float
    height_points: float
    reference_dpi: int
    raster_crc32: str


@dataclass(frozen=True)
class AristaLigandReceptorPanels:
    """Recalculated tables used to draw corrected ARISTA S23 and S24."""

    prototypes: pd.DataFrame
    assignments: pd.DataFrame
    normalized_profiles: pd.DataFrame
    k_selection: pd.DataFrame
    diagnostics: pd.DataFrame
    display_roster: pd.DataFrame
    display_timecourse: pd.DataFrame


@dataclass(frozen=True)
class AristaFigureRelease:
    """Formal ARISTA page sources retained in the repository release."""

    root: Path
    source_index: pd.DataFrame


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


def _require_columns(
    table: pd.DataFrame, columns: tuple[str, ...], source: Path
) -> None:
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


def _release_path(root: Path, value: str, source: str) -> Path:
    _validate_relative_path(value, Path(source))
    path = root / value
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def resolve_arista_release_dir(release_dir: str | Path | None = None) -> Path:
    """Resolve the formal ARISTA release from an argument, environment, or checkout."""

    if release_dir is not None:
        selected: str | Path = release_dir
    else:
        environment_value = os.environ.get(ARISTA_RELEASE_ENVIRONMENT_VARIABLE)
        selected = (
            environment_value
            if environment_value
            else Path(__file__).resolve().parents[2]
            / "release_artifacts"
            / ARISTA_RELEASE_DIRECTORY
        )
    root = Path(selected).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(
            f"ARISTA formal release not found at {root}. Supply release_dir or set "
            f"{ARISTA_RELEASE_ENVIRONMENT_VARIABLE}."
        )
    return root


def _arista_formal_source_index(root: Path) -> pd.DataFrame:
    rows = []
    for specification in _FORMAL_SOURCE_SPECS:
        row = {
            "paper_location": f"Supplementary Figure {specification['figure']}",
            "release_location": (
                f"Supplementary Figure {specification['release_figure']}"
            ),
            "content": specification["topic"],
            "formal_pdf": specification["formal_pdf"],
            "formal_svg": specification["formal_svg"],
            "formal_png": specification["formal_png"],
            "vector_scope": specification["vector_scope"],
            "canonical_scripts": specification["canonical_scripts"],
            "calculation_entrypoints": ";".join(_CALCULATION_ENTRYPOINTS),
            "release_build_snapshot": specification["release_build_snapshot"],
            "build_scope": specification["build_scope"],
            "release_manifest": specification["release_manifest"],
            "downstream_inputs": specification["downstream_inputs"],
            "input_scope": specification["input_scope"],
            "checkpoint_inputs": ";".join(_CHECKPOINT_INPUTS),
            "compact_output": "formal PNG plus raster-equivalent PDF",
        }
        for column in (
            "formal_pdf",
            "formal_svg",
            "formal_png",
            "release_build_snapshot",
            "release_manifest",
            "downstream_inputs",
            "checkpoint_inputs",
        ):
            for relative_path in _split_items(row[column]):
                _release_path(root, relative_path, column)
        for column in ("canonical_scripts", "calculation_entrypoints"):
            for relative_path in _split_items(row[column]):
                _validate_relative_path(relative_path, Path(column))
        rows.append(row)
    return pd.DataFrame(rows)


def load_arista_figure_release(
    release_dir: str | Path | None = None,
) -> AristaFigureRelease:
    """Load formal release sources under the current S19--S24 numbering."""

    root = resolve_arista_release_dir(release_dir)
    for filename in ("README.md", "PROVENANCE.md", "FINAL_RELEASE_QA.md"):
        _release_path(root, filename, "release root")
    return AristaFigureRelease(
        root=root,
        source_index=_arista_formal_source_index(root),
    )


def write_arista_source_index(
    release: AristaFigureRelease,
    output_dir: str | Path,
) -> Path:
    """Write the current-number mapping to formal pages, code, and inputs."""

    output = prepare_output_dir(output_dir)
    path = output / "arista_formal_source_index.csv"
    release.source_index.to_csv(path, index=False)
    return path


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
        source_figure=str(row["source_figure"]),
        topic=str(row["topic"]),
        compact_source=str(row["compact_source"]),
        output_filename=str(row["output_filename"]),
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
        raise ValueError("ARISTA supplementary figure order must be S19 through S24")
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
        raise ValueError(
            f"ARISTA input registry contains unknown figures: {invalid_figures}"
        )
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
        unknown_tables = sorted(
            set(_split_items(row["table_ids"])).difference(table_ids)
        )
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
        if (
            figure not in FIGURE_ORDER
            or filename not in paths
            or not filename.endswith(".csv")
        ):
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


def calculate_arista_ligand_receptor_panels(
    data: AristaSupplementaryData,
) -> AristaLigandReceptorPanels:
    """Recluster all 531 strict LR profiles and select the displayed 50 pairs."""

    from CytoBridge.tl.downstream.temporal import cluster_temporal_profiles

    table_id = "ligand_receptor_all_pair_timecourse"
    if table_id not in data.tables:
        raise ValueError(f"ARISTA LR reproduction is missing table: {table_id}")
    timecourse = data.tables[table_id].copy()
    required = {"pair", "time", "score"}
    missing = sorted(required.difference(timecourse.columns))
    if missing:
        raise ValueError(f"ARISTA all-pair timecourse is missing columns: {missing}")
    if timecourse.duplicated(["pair", "time"]).any():
        raise ValueError("ARISTA all-pair timecourse contains duplicate pair/time rows")

    times = np.arange(0.0, 4.0 + 0.5, 0.5)
    observed_times = np.sort(timecourse["time"].to_numpy(dtype=float))
    observed_times = np.unique(observed_times)
    if not np.array_equal(observed_times, times):
        raise ValueError(
            f"Expected ARISTA times {times.tolist()}, found {observed_times.tolist()}"
        )
    profiles = timecourse.pivot(index="pair", columns="time", values="score")
    profiles = profiles.reindex(columns=times).sort_index()
    if profiles.shape != (531, 9) or profiles.isna().any().any():
        raise ValueError(
            f"Expected a complete 531-by-9 ARISTA LR matrix, found {profiles.shape}"
        )

    clustering = cluster_temporal_profiles(
        profiles,
        n_clusters=2,
        normalization="minmax",
        method="kmeans",
        cluster_order="peak_time",
    )
    assignments = clustering.assignments.rename(columns={"profile": "pair"})
    counts = assignments.groupby("cluster").size().astype(int).to_dict()
    if counts != {1: 217, 2: 314}:
        raise ValueError(
            f"Corrected ARISTA clustering must contain 217 and 314 pairs, found {counts}"
        )
    prototypes = clustering.prototypes.rename(
        columns={
            "mean": "mean_normalized_score",
            "std": "std_normalized_score",
            "n_profiles": "n_pairs",
        }
    )

    k_rows: list[dict[str, object]] = []
    for k in range(2, 9):
        result = cluster_temporal_profiles(
            profiles,
            n_clusters=k,
            normalization="minmax",
            method="kmeans",
            cluster_order="peak_time",
        )
        diagnostic = result.diagnostics.iloc[0]
        cluster_counts = result.assignments.groupby("cluster").size().astype(int)
        k_rows.append(
            {
                "k": k,
                "silhouette": float(diagnostic["silhouette"]),
                "minimum_cluster_size": int(cluster_counts.min()),
                "maximum_cluster_size": int(cluster_counts.max()),
                "cluster_counts": ";".join(
                    f"{int(cluster)}:{int(count)}"
                    for cluster, count in cluster_counts.items()
                ),
            }
        )
    k_selection = pd.DataFrame(k_rows)
    selected_k = int(
        k_selection.sort_values(
            ["silhouette", "k"], ascending=[False, True]
        ).iloc[0]["k"]
    )
    if selected_k != 2:
        raise ValueError(f"Expected k=2 to have the best silhouette, found k={selected_k}")

    normalized = clustering.normalized_profiles.copy()
    normalized.index.name = "pair"
    normalized.columns = [f"time_{float(value):.1f}" for value in normalized.columns]
    normalized_table = normalized.reset_index()
    assigned = normalized_table.merge(assignments, on="pair", validate="one_to_one")
    time_columns = [f"time_{value:.1f}" for value in times]
    blocks: list[pd.DataFrame] = []
    for cluster in (1, 2):
        block = assigned.loc[assigned["cluster"].eq(cluster)].copy()
        prototype = block[time_columns].mean(axis=0).to_numpy(dtype=float)
        delta = block[time_columns].to_numpy(dtype=float) - prototype[None, :]
        block["distance_to_pattern_prototype"] = np.sqrt(
            np.square(delta).sum(axis=1)
        )
        block = block.sort_values(
            ["distance_to_pattern_prototype", "pair"], kind="mergesort"
        ).reset_index(drop=True)
        block["representativeness_rank_within_pattern"] = np.arange(
            1, len(block) + 1
        )
        blocks.append(block.head(25))
    roster = pd.concat(blocks, ignore_index=True)
    roster = roster.sort_values(
        ["cluster", "representativeness_rank_within_pattern"], kind="mergesort"
    ).reset_index(drop=True)
    roster["display_order"] = np.arange(1, len(roster) + 1)
    if roster.groupby("cluster").size().to_dict() != {1: 25, 2: 25}:
        raise ValueError("ARISTA representative selection must retain 25 pairs per cluster")

    roster_columns = [
        "pair",
        "cluster",
        "representativeness_rank_within_pattern",
        "distance_to_pattern_prototype",
        "display_order",
    ]
    display_timecourse = timecourse.loc[timecourse["pair"].isin(roster["pair"])].merge(
        roster[roster_columns], on="pair", validate="many_to_one"
    )
    display_timecourse = display_timecourse.sort_values(
        ["display_order", "time"], kind="mergesort"
    ).reset_index(drop=True)
    if len(display_timecourse) != 450:
        raise ValueError(
            f"Expected 450 displayed ARISTA LR rows, found {len(display_timecourse)}"
        )

    return AristaLigandReceptorPanels(
        prototypes=prototypes.reset_index(drop=True),
        assignments=assignments.sort_values("pair", kind="mergesort").reset_index(
            drop=True
        ),
        normalized_profiles=normalized_table,
        k_selection=k_selection,
        diagnostics=clustering.diagnostics.copy(),
        display_roster=roster,
        display_timecourse=display_timecourse,
    )


def select_arista_supplementary_pages(
    pages: Iterable[AristaSupplementaryPage],
    figures: Iterable[str] | None = None,
) -> tuple[AristaSupplementaryPage, ...]:
    """Select current ARISTA figure identifiers while preserving page order."""

    page_map = {page.figure: page for page in pages}
    requested = (
        FIGURE_ORDER if figures is None else tuple(str(value) for value in figures)
    )
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


def write_arista_ligand_receptor_tables(
    data: AristaSupplementaryData,
    output_dir: str | Path,
    panels: AristaLigandReceptorPanels | None = None,
) -> dict[str, Path]:
    """Write the all-pair input and tables recalculated for S23 and S24."""

    output = prepare_output_dir(output_dir)
    calculated = (
        calculate_arista_ligand_receptor_panels(data) if panels is None else panels
    )
    calculated_tables = {
        "ligand_receptor_cluster_prototypes": calculated.prototypes,
        "ligand_receptor_cluster_assignments": calculated.assignments,
        "ligand_receptor_normalized_profiles": calculated.normalized_profiles,
        "ligand_receptor_k_selection": calculated.k_selection,
        "ligand_receptor_cluster_diagnostics": calculated.diagnostics,
        "ligand_receptor_display_roster": calculated.display_roster,
        "ligand_receptor_pair_timecourse": calculated.display_timecourse,
    }
    written: dict[str, Path] = {}
    for table_id in _LR_TABLE_IDS:
        if table_id not in data.tables:
            raise ValueError(f"ARISTA LR reproduction is missing table: {table_id}")
        specification = data.manifest["tables"][table_id]
        filename = str(specification["file"])
        destination = output / filename
        if table_id == "ligand_receptor_all_pair_timecourse":
            shutil.copyfile(data.source_dir / filename, destination)
        else:
            calculated_tables[table_id].to_csv(destination, index=False)
        written[table_id] = destination
    return written


def export_arista_reference_pages(
    data: AristaSupplementaryData,
    output_dir: str | Path,
    pages: Iterable[AristaSupplementaryPage] | None = None,
    figures: Iterable[str] | None = None,
) -> dict[str, tuple[Path, Path]]:
    """Export released page images for visual reference.

    This export preserves the released page appearance but does not recalculate
    the analyses or rebuild their vector layouts.  Use
    :func:`plot_arista_ligand_receptor_figures` to redraw S23 and S24 from the
    released numerical tables.
    """

    from ._arista_supplementary_figures_plot import (
        render_arista_supplementary_figures,
    )

    calculated = (
        calculate_arista_supplementary_pages(data) if pages is None else tuple(pages)
    )
    selected = select_arista_supplementary_pages(calculated, figures)
    return render_arista_supplementary_figures(data, selected, output_dir)


def plot_arista_supplementary_figures(
    data: AristaSupplementaryData,
    output_dir: str | Path,
    pages: Iterable[AristaSupplementaryPage] | None = None,
    figures: Iterable[str] | None = None,
) -> dict[str, tuple[Path, Path]]:
    """Compatibility alias for :func:`export_arista_reference_pages`."""

    return export_arista_reference_pages(data, output_dir, pages, figures)


def plot_arista_ligand_receptor_figures(
    data: AristaSupplementaryData,
    output_dir: str | Path,
    panels: AristaLigandReceptorPanels | None = None,
) -> dict[str, tuple[Path, Path]]:
    """Redraw corrected ARISTA S23 and S24 from all 531 strict LR profiles."""

    from ._arista_supplementary_figures_plot import (
        render_arista_ligand_receptor_figures,
    )

    calculated = (
        calculate_arista_ligand_receptor_panels(data) if panels is None else panels
    )
    return render_arista_ligand_receptor_figures(calculated, output_dir)


__all__ = [
    "ARISTA_RELEASE_DIRECTORY",
    "ARISTA_RELEASE_ENVIRONMENT_VARIABLE",
    "AristaFigureRelease",
    "AristaLigandReceptorPanels",
    "AristaSupplementaryData",
    "AristaSupplementaryPage",
    "FIGURE_ORDER",
    "calculate_arista_supplementary_pages",
    "calculate_arista_ligand_receptor_panels",
    "export_arista_reference_pages",
    "load_arista_figure_release",
    "load_arista_supplementary_figures",
    "plot_arista_supplementary_figures",
    "plot_arista_ligand_receptor_figures",
    "resolve_arista_release_dir",
    "select_arista_supplementary_pages",
    "write_arista_source_index",
    "write_arista_ligand_receptor_tables",
    "write_arista_supplementary_tables",
]
