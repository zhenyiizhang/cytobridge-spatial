#!/usr/bin/env python3
"""Rebuild ARISTA SI Figures S12--S14 with corrected state and legacy style.

The renderer deliberately separates scientific inputs from display policy:

* corrected S12 states, corrected dense-grid growth, and corrected fixed-particle
  lineage/composition are immutable inputs;
* the submitted S12/S13/S14 plotting grammar is copied from the historical
  renderer (layout, point sizes, sampling, titles, axes, fonts, and palette);
* no historical 46,189-row identity mask is applied;
* an optional, label-blind display-only spatial-isolation rule is rendered as a
  transparent B variant, while the A variant retains every valid observation;
* every row remains in all numerical tables and computations.

The script refuses to write into a non-empty output directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import anndata as ad
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import Normalize
from matplotlib.patches import Ellipse
from matplotlib.transforms import Bbox
from PIL import Image, ImageDraw, ImageFont
from pypdf import PdfReader, PdfWriter
from scipy.spatial import cKDTree


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CORRECTED_BUNDLE = PROJECT_ROOT / "output/arista_paper_equivalent_corrected_20260822_3c87a3e"
FULL_BANK = PROJECT_ROOT / "output/arista_full_model_figure_bank_corrected_20260822_3c87a3e/downstream"
FORMAL_V3 = CORRECTED_BUNDLE / "server_results/formal_paper_full46209_canonical983b_3c87a3e_v3"
S13_AUGMENTED = (
    CORRECTED_BUNDLE
    / "server_results/main5abce_s13_paper_contract_full46209_canonical983b_v3"
    / "tables/s13_growth_by_cell_full_compute_display_mask.csv"
)
PALETTE_JSON = PROJECT_ROOT / "repositories/cb_reproducibility/assets/arista/label_to_color.json"
LEGACY_EXTRA_EVIDENCE = CORRECTED_BUNDLE / "evidence/observed_display_exclusions_20.csv"

OLD_S12_REFERENCE = Path(
    "/Users/zhenyizhang/Desktop/202511/CytoBRIDGE/supp_text/figures/arista/"
    "interpolation_results_compact.jpg"
)
OLD_S13_REFERENCE = Path(
    "/Users/zhenyizhang/Desktop/202511/CytoBRIDGE/supp_text/figures/arista/"
    "arista_growth_maps_dense_compact.jpg"
)
OLD_S14_REFERENCE = Path(
    "/Users/zhenyizhang/Desktop/202511/CytoBRIDGE/supp_text/figures/arista/"
    "lineage_sankey_compact.jpg"
)
OLD_S12_RAW = PROJECT_ROOT / "results/arista_lineage_snapshot_focus_anchor_local/timepoint_svg/timepoint_mosaic.png"
OLD_S13_RAW = PROJECT_ROOT / "results/arista_review_dense_local/figures/arista_growth_maps_dense.png"
OLD_S14A_RAW = PROJECT_ROOT / "results/arista_lineage_snapshot_focus_anchor_local/lineage_sankey.svg"
OLD_S14B_RAW = PROJECT_ROOT / "results/arista_review_dense_local/figures/arista_cell_composition_stacked_bar.svg"
FORMAL_K10_WITNESS = PROJECT_ROOT / "results/cross_dataset_corrected_replicas_20260812/arista/figures/ARISTA_S16_formal_k10.png"
REVIEW_DENSE_MANIFEST = PROJECT_ROOT / "results/arista_review_dense_local/analysis_manifest.json"
ACCEPTED_V3 = PROJECT_ROOT / "output/arista_si_s12_s14_corrected_legacy_style_20260823_3c87a3e_v3"

DEFAULT_OUTPUT = PROJECT_ROOT / "output/arista_si_s12_s14_corrected_legacy_style_20260823_3c87a3e_v4"

OBSERVED_TIMES = (0.0, 1.0, 2.0, 3.0, 4.0)
MIDPOINT_TIMES = (0.5, 1.5, 2.5, 3.5)
DENSE_TIMES = tuple(sorted(OBSERVED_TIMES + MIDPOINT_TIMES))
S12_LAYOUT: tuple[tuple[float, str] | None, ...] = (
    (0.0, "Observed"),
    (0.0, "Generated"),
    (0.5, "Generated"),
    (1.0, "Observed"),
    (1.0, "Generated"),
    (1.5, "Generated"),
    (2.0, "Observed"),
    (2.0, "Generated"),
    (2.5, "Generated"),
    (3.0, "Observed"),
    (3.0, "Generated"),
    (3.5, "Generated"),
    (4.0, "Observed"),
    (4.0, "Generated"),
    None,
    None,
)

ISOLATION_Z_THRESHOLD = 20.0
ISOLATION_SENSITIVITY_THRESHOLDS = (15.0, 20.0, 30.0, 50.0, 100.0)
DISPLAY_SAMPLE_SEED = 42
DISPLAY_SAMPLE_CAP = 2500
FIXED_TIMESTAMP = "2026-08-23T00:00:00+00:00"
FIXED_PDF_DATE = datetime(2026, 8, 23, tzinfo=timezone.utc)
S12_SUBMITTED_CANVAS_PT = (505.44, 502.262564)
S13_SUBMITTED_CANVAS_PT = (900.132969, 865.422001)
S12_SUBMITTED_RASTER_PX = (2106, 2093)
S13_SUBMITTED_RASTER_PX = (3751, 3606)


@dataclass(frozen=True)
class SpatialPanel:
    time: float
    source: str
    x: np.ndarray
    y: np.ndarray
    labels: np.ndarray
    ids: np.ndarray
    input_path: Path
    coordinate_basis: str

    @property
    def key(self) -> tuple[float, str]:
        return (self.time, self.source)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--full-bank", type=Path, default=FULL_BANK)
    parser.add_argument("--formal-v3", type=Path, default=FORMAL_V3)
    parser.add_argument("--s13-augmented", type=Path, default=S13_AUGMENTED)
    parser.add_argument("--palette", type=Path, default=PALETTE_JSON)
    parser.add_argument("--legacy-extra-evidence", type=Path, default=LEGACY_EXTRA_EVIDENCE)
    parser.add_argument("--isolation-z-threshold", type=float, default=ISOLATION_Z_THRESHOLD)
    parser.add_argument(
        "--determinism-reference",
        type=Path,
        default=None,
        help="Optional completed bundle whose non-manifest artifacts must hash-match this rebuild.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_files(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(missing))


def initialize_output(output_dir: Path) -> dict[str, Path]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty immutable output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    directories = {
        "vector": output_dir / "figures/vector",
        "pdf": output_dir / "figures/pdf",
        "png": output_dir / "figures/png",
        "jpeg": output_dir / "figures/jpeg",
        "qa": output_dir / "qa",
        "tables": output_dir / "tables",
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    return directories


def canonical_observed_id(value: str) -> str:
    pieces = []
    for piece in str(value).split("|"):
        if piece.startswith("Batch="):
            piece = piece[len("Batch=") :]
        elif piece.startswith("CellID="):
            piece = piece[len("CellID=") :]
        pieces.append(piece)
    return "|".join(pieces)


def time_token(time: float) -> str:
    return str(int(time)) if float(time).is_integer() else str(time).replace(".", "p")


def load_palette(path: Path) -> dict[str, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not data:
        raise ValueError(f"Invalid palette: {path}")
    return {str(label): str(color) for label, color in data.items()}


def validate_labels(labels: Iterable[str], palette: dict[str, str], context: str) -> None:
    missing = sorted(set(map(str, labels)) - set(palette))
    if missing:
        raise KeyError(f"Labels absent from the canonical palette in {context}: {missing}")


def load_observed_panel(
    path: Path,
    time: float,
    palette: dict[str, str],
    *,
    source_ids: np.ndarray | None = None,
    source_labels: np.ndarray | None = None,
) -> SpatialPanel:
    adata = ad.read_h5ad(path)
    if "spatial" not in adata.obsm or "Annotation" not in adata.obs:
        raise KeyError(f"{path} must contain spatial and Annotation")
    coordinates = np.asarray(adata.obsm["spatial"], dtype=float)
    labels = adata.obs["Annotation"].astype(str).to_numpy()
    if "source_obs_id" in adata.obs:
        ids = adata.obs["source_obs_id"].astype(str).map(canonical_observed_id).to_numpy()
    elif source_ids is not None:
        ids = np.asarray(
            [canonical_observed_id(value) for value in source_ids], dtype=object
        )
    else:
        raise KeyError(
            f"{path} lacks source_obs_id and no external identity roster was supplied"
        )
    if len(ids) != adata.n_obs:
        raise ValueError(f"External identity count differs at t={time:g}: {len(ids)} != {adata.n_obs}")
    if source_labels is not None and not np.array_equal(
        labels.astype(str), np.asarray(source_labels).astype(str)
    ):
        raise ValueError(f"External identity labels are not row-identical at t={time:g}")
    if coordinates.shape != (adata.n_obs, 2) or not np.isfinite(coordinates).all():
        raise ValueError(f"Invalid observed spatial coordinates in {path}: {coordinates.shape}")
    if pd.Series(ids).duplicated().any():
        raise ValueError(f"Duplicate source_obs_id values in {path}")
    validate_labels(labels, palette, str(path))
    return SpatialPanel(
        time=float(time),
        source="Observed",
        x=coordinates[:, 0],
        y=coordinates[:, 1],
        labels=labels,
        ids=ids,
        input_path=path,
        coordinate_basis="accepted corrected H5AD obsm['spatial']",
    )


def parse_generated_panel(path: Path, time: float, palette: dict[str, str]) -> SpatialPanel:
    root = ET.parse(path).getroot()
    namespace = "{http://www.w3.org/2000/svg}"
    collections = [
        node for node in root.iter(namespace + "g") if node.get("id", "").startswith("PathCollection_")
    ]
    if not collections:
        raise ValueError(f"No PathCollection found in corrected generated panel: {path}")
    uses = list(collections[0].iter(namespace + "use"))
    if not uses:
        raise ValueError(f"No corrected generated cells found in {path}")
    color_to_label = {str(color).lower(): str(label) for label, color in palette.items()}
    fill_re = re.compile(r"(?:^|;)\s*fill:\s*(#[0-9a-fA-F]{6})")
    xs: list[float] = []
    ys: list[float] = []
    labels: list[str] = []
    for node in uses:
        match = fill_re.search(node.get("style", ""))
        if match is None:
            raise ValueError(f"Generated point lacks a fill color: {path}")
        color = match.group(1).lower()
        if color not in color_to_label:
            raise KeyError(f"Generated color {color} is absent from the canonical palette")
        xs.append(float(node.get("x")))
        ys.append(-float(node.get("y")))
        labels.append(color_to_label[color])
    labels_array = np.asarray(labels, dtype=object)
    validate_labels(labels_array, palette, str(path))
    return SpatialPanel(
        time=float(time),
        source="Generated",
        x=np.asarray(xs, dtype=float),
        y=np.asarray(ys, dtype=float),
        labels=labels_array,
        ids=np.asarray([f"generated:{time:.1f}:{index}" for index in range(len(labels))], dtype=object),
        input_path=path,
        coordinate_basis="corrected formal-v3 SVG PathCollection_1 geometry",
    )


def load_s12_panels(
    full_bank: Path,
    formal_v3: Path,
    palette: dict[str, str],
    identity_csv: Path | None = None,
    generated_source_palette: dict[str, str] | None = None,
) -> dict[tuple[float, str], SpatialPanel]:
    panels: dict[tuple[float, str], SpatialPanel] = {}
    slice_dir = full_bank / "slice_data"
    snapshot_dir = formal_v3 / "snapshots"
    identity = None
    if identity_csv is not None:
        identity = pd.read_csv(identity_csv)
        required = {"source_obs_id", "time_point_processed", "Annotation"}
        if not required.issubset(identity.columns):
            raise KeyError(
                f"Identity CSV is missing {sorted(required - set(identity.columns))}"
            )
    for time in OBSERVED_TIMES:
        path = slice_dir / f"time_{time_token(time)}.h5ad"
        source_ids = None
        source_labels = None
        if identity is not None:
            subset = identity[np.isclose(identity["time_point_processed"], float(time))]
            source_ids = subset["source_obs_id"].astype(str).to_numpy()
            source_labels = subset["Annotation"].astype(str).to_numpy()
        panel = load_observed_panel(
            path,
            time,
            palette,
            source_ids=source_ids,
            source_labels=source_labels,
        )
        panels[panel.key] = panel
    for time in DENSE_TIMES:
        if time in OBSERVED_TIMES:
            path = snapshot_dir / f"time_{time:.1f}__Generated.svg"
        else:
            path = snapshot_dir / f"time_{time:.1f}.svg"
        panel = parse_generated_panel(
            path,
            time,
            palette if generated_source_palette is None else generated_source_palette,
        )
        panels[panel.key] = panel
    expected = {item for item in S12_LAYOUT if item is not None}
    if set(panels) != expected:
        raise AssertionError(f"S12 panel inventory mismatch: {set(panels) ^ expected}")
    return panels


def configure_s12_legacy_style(font_family: str = "DejaVu Sans") -> None:
    matplotlib.rcdefaults()
    matplotlib.rcParams.update(
        {
            "font.family": str(font_family),
            "savefig.dpi": 300,
            "svg.fonttype": "path",
            "svg.hashsalt": "arista-s12-s14-legacy-style-3c87a3e",
        }
    )


def configure_review_legacy_style(font_family: str = "DejaVu Sans") -> None:
    matplotlib.rcdefaults()
    sns.set_theme(style="white", context="paper")
    matplotlib.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "font.size": 10,
            "font.family": str(font_family),
            "svg.fonttype": "path",
            "svg.hashsalt": "arista-s12-s14-legacy-style-3c87a3e",
        }
    )


def save_jpeg(png_path: Path, jpeg_path: Path) -> None:
    image = Image.open(png_path).convert("RGB")
    image.save(jpeg_path, format="JPEG", quality=95, subsampling=0, dpi=(300, 300), optimize=False)


def submitted_canvas_bbox(fig: plt.Figure, canvas_pt: tuple[float, float]) -> Bbox:
    """Return a fixed legacy canvas around the unchanged tight plot content.

    The historical renderers used ``bbox_inches='tight'``.  A tight bounding
    box is data-dependent when equal-aspect spatial axes receive corrected
    coordinates, so the same renderer can otherwise change the outer canvas.
    We preserve the tight content at one point per point and only add symmetric
    white margin up to the physical canvas measured from the submitted SVG.
    """

    fig.canvas.draw()
    tight = fig.get_tightbbox(fig.canvas.get_renderer()).padded(0.1)
    target_width = float(canvas_pt[0]) / 72.0
    target_height = float(canvas_pt[1]) / 72.0
    tolerance = 1e-6
    if tight.width > target_width + tolerance or tight.height > target_height + tolerance:
        raise AssertionError(
            "Corrected tight content exceeds submitted canvas: "
            f"tight={tight.width * 72:.6f}x{tight.height * 72:.6f} pt, "
            f"target={canvas_pt[0]:.6f}x{canvas_pt[1]:.6f} pt"
        )
    return Bbox.from_bounds(
        tight.x0 - (target_width - tight.width) / 2.0,
        tight.y0 - (target_height - tight.height) / 2.0,
        target_width,
        target_height,
    )


def submitted_raster_bbox(fig: plt.Figure, raster_px: tuple[int, int], dpi: int = 300) -> Bbox:
    """Return an exact raster-sized canvas around the unchanged tight content."""

    fig.canvas.draw()
    tight = fig.get_tightbbox(fig.canvas.get_renderer()).padded(0.1)
    # A tiny epsilon avoids platform-dependent truncation of an integer-valued
    # floating-point pixel extent inside Matplotlib's Agg backend.
    target_width = (float(raster_px[0]) + 1e-6) / float(dpi)
    target_height = (float(raster_px[1]) + 1e-6) / float(dpi)
    if tight.width > target_width or tight.height > target_height:
        raise AssertionError(
            "Corrected tight content exceeds submitted raster canvas: "
            f"tight={tight.width * dpi:.6f}x{tight.height * dpi:.6f} px, "
            f"target={raster_px[0]}x{raster_px[1]} px"
        )
    return Bbox.from_bounds(
        tight.x0 - (target_width - tight.width) / 2.0,
        tight.y0 - (target_height - tight.height) / 2.0,
        target_width,
        target_height,
    )


def save_mpl_figure(
    fig: plt.Figure,
    name: str,
    directories: dict[str, Path],
    *,
    submitted_canvas_pt: tuple[float, float] | None = None,
    submitted_raster_px: tuple[int, int] | None = None,
) -> dict[str, Path]:
    paths = {
        "svg": directories["vector"] / f"{name}.svg",
        "pdf": directories["pdf"] / f"{name}.pdf",
        "png": directories["png"] / f"{name}.png",
        "jpg": directories["jpeg"] / f"{name}.jpg",
    }
    bbox_inches: str | Bbox = (
        submitted_canvas_bbox(fig, submitted_canvas_pt)
        if submitted_canvas_pt is not None
        else "tight"
    )
    fig.savefig(
        paths["svg"],
        format="svg",
        facecolor="white",
        bbox_inches=bbox_inches,
        metadata={"Date": FIXED_TIMESTAMP, "Creator": "ARISTA corrected legacy-style renderer"},
    )
    fig.savefig(
        paths["pdf"],
        format="pdf",
        facecolor="white",
        bbox_inches=bbox_inches,
        metadata={
            "Creator": "ARISTA corrected legacy-style renderer",
            "CreationDate": FIXED_PDF_DATE,
            "ModDate": FIXED_PDF_DATE,
        },
    )
    raster_bbox: str | Bbox = (
        submitted_raster_bbox(fig, submitted_raster_px)
        if submitted_raster_px is not None
        else bbox_inches
    )
    fig.savefig(paths["png"], format="png", facecolor="white", bbox_inches=raster_bbox, dpi=300)
    save_jpeg(paths["png"], paths["jpg"])
    return paths


def batch_time_matches(source_id: str, time: float) -> bool:
    match = re.search(r"Injury_(\d+)DPI", str(source_id))
    if match is None:
        return False
    expected = {2: 0.0, 5: 1.0, 10: 2.0, 15: 3.0, 20: 4.0}
    return expected.get(int(match.group(1))) == float(time)


def build_outlier_audit(
    panels: dict[tuple[float, str], SpatialPanel],
    palette: dict[str, str],
    legacy_extra_evidence: Path,
    tables_dir: Path,
    threshold: float,
) -> tuple[pd.DataFrame, set[tuple[float, str]], dict]:
    legacy_table = pd.read_csv(legacy_extra_evidence)
    legacy_ids = set(legacy_table["composite_id"].astype(str).map(canonical_observed_id))
    records: list[dict] = []
    threshold_rows: list[dict] = []
    panel_items = [item for item in S12_LAYOUT if item is not None]
    for panel_index, (time, source) in enumerate(panel_items):
        panel = panels[(time, source)]
        coordinates = np.column_stack([panel.x, panel.y])
        nn_distance = cKDTree(coordinates).query(coordinates, k=2)[0][:, 1]
        median = float(np.median(nn_distance))
        mad = float(np.median(np.abs(nn_distance - median)))
        scale = 1.4826 * mad
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"Cannot define robust nearest-neighbor scale at t={time}, source={source}")
        robust_z = (nn_distance - median) / scale
        for index, (source_id, label, x, y, distance, z_value) in enumerate(
            zip(panel.ids, panel.labels, panel.x, panel.y, nn_distance, robust_z)
        ):
            finite = bool(np.isfinite([x, y]).all())
            if source == "Observed":
                source_id_valid = batch_time_matches(str(source_id), time)
            else:
                source_id_valid = str(source_id) == f"generated:{time:.1f}:{index}"
            palette_valid = str(label) in palette
            malformed_or_wrong_source = not (finite and source_id_valid and palette_valid)
            row = {
                "s12_layout_index": int(panel_index),
                "time": float(time),
                "source": str(source),
                "row_index_within_panel": int(index),
                "source_id": str(source_id),
                "celltype": str(label),
                "x": float(x),
                "y": float(y),
                "nearest_neighbor_distance": float(distance),
                "panel_nn_median": median,
                "panel_nn_mad": mad,
                "robust_nn_z": float(z_value),
                "finite_coordinates": finite,
                "source_id_matches_panel": source_id_valid,
                "canonical_palette_label": palette_valid,
                "malformed_or_wrong_source": malformed_or_wrong_source,
                "legacy_46189_roster_extra": (
                    source == "Observed" and canonical_observed_id(str(source_id)) in legacy_ids
                ),
                "objective_display_isolation_flag": bool(z_value > threshold),
                "computation_excluded": False,
            }
            for candidate_threshold in ISOLATION_SENSITIVITY_THRESHOLDS:
                row[f"flag_z_gt_{int(candidate_threshold)}"] = bool(z_value > candidate_threshold)
                if z_value > candidate_threshold:
                    threshold_rows.append(
                        {
                            "threshold": float(candidate_threshold),
                            "time": float(time),
                            "source": str(source),
                            "source_id": str(source_id),
                            "celltype": str(label),
                            "x": float(x),
                            "y": float(y),
                            "nearest_neighbor_distance": float(distance),
                            "robust_nn_z": float(z_value),
                        }
                    )
            records.append(row)
    audit = pd.DataFrame(records).sort_values(["s12_layout_index", "row_index_within_panel"]).reset_index(drop=True)
    if audit.duplicated(["time", "source", "source_id"]).any():
        raise AssertionError("S12 spatial-isolation audit contains duplicate panel/source IDs")
    if audit["malformed_or_wrong_source"].any():
        raise AssertionError("Objective malformed/wrong-source validation failed")
    flagged = audit[audit["objective_display_isolation_flag"]].copy()
    flag_keys = set(zip(flagged["time"].astype(float), flagged["source_id"].astype(str)))
    audit.to_csv(tables_dir / "s12_spatial_isolation_audit_all_14_panels.csv", index=False)
    flagged.to_csv(tables_dir / f"s12_spatial_isolation_flags_all_panels_nnmad_zgt{int(threshold)}.csv", index=False)
    observed_audit = audit[audit["source"] == "Observed"].copy()
    observed_flags = flagged[flagged["source"] == "Observed"].copy()
    generated_flags = flagged[flagged["source"] == "Generated"].copy()
    observed_audit.to_csv(tables_dir / "observed_outlier_audit_full_46209.csv", index=False)
    observed_flags.to_csv(tables_dir / f"observed_outlier_flags_nnmad_zgt{int(threshold)}.csv", index=False)
    generated_flags.to_csv(tables_dir / f"generated_outlier_flags_nnmad_zgt{int(threshold)}.csv", index=False)
    sensitivity = pd.DataFrame(threshold_rows).sort_values(
        ["threshold", "time", "source", "robust_nn_z"], ascending=[True, True, True, False]
    )
    sensitivity.to_csv(tables_dir / "s12_spatial_isolation_threshold_sensitivity.csv", index=False)
    top = (
        audit.sort_values(["s12_layout_index", "robust_nn_z"], ascending=[True, False])
        .groupby("s12_layout_index", sort=True)
        .head(10)
        .reset_index(drop=True)
    )
    top.to_csv(tables_dir / "s12_spatial_isolation_top10_each_panel.csv", index=False)
    flags_by_panel = {
        f"{float(time):.1f}|{source}": int(count)
        for (time, source), count in flagged.groupby(["time", "source"], sort=True).size().items()
    }
    summary = {
        "rule": "within each of all 14 S12 panels, flag robust 1-NN z > threshold; z=(d-median(d))/(1.4826*MAD(d))",
        "label_blind": True,
        "threshold": float(threshold),
        "n_s12_compute_rows": int(len(audit)),
        "n_observed_compute_rows": int((audit["source"] == "Observed").sum()),
        "n_generated_compute_rows": int((audit["source"] == "Generated").sum()),
        "n_malformed_or_wrong_source": int(audit["malformed_or_wrong_source"].sum()),
        "n_display_flags": int(len(flagged)),
        "n_observed_display_flags": int(len(observed_flags)),
        "n_generated_display_flags": int(len(generated_flags)),
        "flags_by_panel": flags_by_panel,
        "threshold_sensitivity_counts": {
            str(value): int(audit[f"flag_z_gt_{int(value)}"].sum())
            for value in ISOLATION_SENSITIVITY_THRESHOLDS
        },
        "computation_policy": "all 128,330 S12 rows retained: 46,209 observed plus 82,121 split-SDE generated",
        "legacy_20_policy": "annotated for audit only; never used as a display mask",
    }
    (tables_dir / "s12_spatial_isolation_audit_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return audit, flag_keys, summary


def plot_s12(
    panels: dict[tuple[float, str], SpatialPanel],
    palette: dict[str, str],
    flag_keys: set[tuple[float, str]],
    hide_objective_flags: bool,
    name: str,
    directories: dict[str, Path],
    *,
    font_family: str = "DejaVu Sans",
) -> tuple[dict[str, Path], pd.DataFrame]:
    configure_s12_legacy_style(font_family)
    fig, axes = plt.subplots(4, 4, figsize=(8.8, 8.8), dpi=300)
    inventory: list[dict] = []
    for layout_index, (ax, item) in enumerate(zip(axes.flat, S12_LAYOUT)):
        if item is None:
            ax.axis("off")
            ax.set_facecolor("white")
            continue
        panel = panels[item]
        keep = np.ones(len(panel.x), dtype=bool)
        if hide_objective_flags:
            keep = np.asarray([(panel.time, str(source_id)) not in flag_keys for source_id in panel.ids])
        colors = [palette.get(str(label), "#888888") for label in panel.labels[keep]]
        ax.set_facecolor("white")
        ax.scatter(
            panel.x[keep],
            panel.y[keep],
            s=2.5,
            c=colors,
            linewidths=0,
            alpha=0.9,
            rasterized=False,
        )
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(f"t = {panel.time:.1f} | {panel.source}", color="#1a1a1a", fontsize=8, pad=3)
        inventory.append(
            {
                "layout_index": int(layout_index),
                "time": float(panel.time),
                "source": panel.source,
                "n_compute": int(len(panel.x)),
                "n_display": int(keep.sum()),
                "n_objective_display_hidden": int((~keep).sum()),
                "input_path": str(panel.input_path.resolve()),
                "coordinate_basis": panel.coordinate_basis,
            }
        )
    paths = save_mpl_figure(
        fig,
        name,
        directories,
        submitted_canvas_pt=S12_SUBMITTED_CANVAS_PT,
        submitted_raster_px=S12_SUBMITTED_RASTER_PX,
    )
    plt.close(fig)
    return paths, pd.DataFrame(inventory)


def validate_s13_sources(formal_growth: Path, augmented_path: Path) -> pd.DataFrame:
    raw = pd.read_csv(formal_growth).sort_values(["time", "cell_index"]).reset_index(drop=True)
    augmented = pd.read_csv(augmented_path).sort_values(["time", "cell_index"]).reset_index(drop=True)
    required = {
        "time",
        "time_key",
        "cell_index",
        "x",
        "y",
        "growth",
        "celltype",
        "source",
        "composite_id",
    }
    missing = required - set(augmented.columns)
    if missing:
        raise KeyError(f"Augmented S13 table lacks columns: {sorted(missing)}")
    if len(raw) != 82329 or len(raw) != len(augmented):
        raise AssertionError(f"Unexpected corrected S13 row counts: {len(raw)} and {len(augmented)}")
    for column in ("time", "time_key", "cell_index", "x", "y", "growth"):
        if not np.allclose(raw[column], augmented[column], rtol=0.0, atol=1e-10, equal_nan=True):
            raise AssertionError(f"Corrected and augmented S13 tables differ in {column}")
    if not np.array_equal(raw["celltype"].astype(str), augmented["celltype"].astype(str)):
        raise AssertionError("Corrected and augmented S13 cell-type labels differ")
    if tuple(sorted(augmented["time"].astype(float).unique())) != DENSE_TIMES:
        raise AssertionError("Corrected S13 dense time grid changed")
    return augmented


def build_s13_display_sample(
    table: pd.DataFrame,
    s12_audit: pd.DataFrame,
    tables_dir: Path,
) -> pd.DataFrame:
    flagged = s12_audit[s12_audit["objective_display_isolation_flag"]].copy()
    observed_flag_keys = {
        (float(row.time), canonical_observed_id(str(row.source_id)))
        for row in flagged.itertuples()
        if row.source == "Observed"
    }
    generated_flag_keys = {
        (float(row.time), int(row.row_index_within_panel))
        for row in flagged.itertuples()
        if row.source == "Generated"
    }
    rng = np.random.default_rng(DISPLAY_SAMPLE_SEED)
    sampled: list[pd.DataFrame] = []
    for time, sub in table.groupby("time", sort=True):
        sub = sub.sort_values("cell_index").reset_index(drop=True)
        n_display = min(DISPLAY_SAMPLE_CAP, len(sub))
        chosen = np.sort(rng.choice(len(sub), size=n_display, replace=False))
        selected = sub.iloc[chosen].copy()
        selected["n_compute_panel"] = int(len(sub))
        selected["display_sample_rank"] = np.arange(len(selected), dtype=int)
        objective_flags: list[bool] = []
        mapping_ids: list[str] = []
        for row in selected.itertuples():
            if str(row.source).lower() == "observed":
                mapping_id = canonical_observed_id(str(row.composite_id))
                is_flagged = (float(time), mapping_id) in observed_flag_keys
            else:
                mapping_id = f"generated:{float(time):.1f}:{int(row.cell_index)}"
                is_flagged = (float(time), int(row.cell_index)) in generated_flag_keys
            objective_flags.append(bool(is_flagged))
            mapping_ids.append(mapping_id)
        selected["objective_isolation_flag"] = objective_flags
        selected["s12_source_row_id"] = mapping_ids
        sampled.append(selected)
    result = pd.concat(sampled, ignore_index=True)
    if len(result) != DISPLAY_SAMPLE_CAP * len(DENSE_TIMES):
        raise AssertionError(f"Unexpected S13 legacy display sample size: {len(result)}")
    result.to_csv(tables_dir / "s13_seed42_display_sample_all_valid.csv", index=False)
    return result


def write_s12_s13_flag_mapping(
    s12_audit: pd.DataFrame,
    s13_sample: pd.DataFrame,
    tables_dir: Path,
) -> pd.DataFrame:
    rows: list[dict] = []
    flagged = s12_audit[s12_audit["objective_display_isolation_flag"]].copy()
    for item in flagged.itertuples():
        if item.source == "Observed":
            matches = s13_sample[
                (s13_sample["time"].astype(float) == float(item.time))
                & (s13_sample["s12_source_row_id"].astype(str) == str(item.source_id))
            ]
            mapping_basis = "canonical observed source ID"
        else:
            matches = s13_sample[
                (s13_sample["time"].astype(float) == float(item.time))
                & (s13_sample["cell_index"].astype(int) == int(item.row_index_within_panel))
                & (s13_sample["source"].astype(str).str.lower() == "simulated")
            ]
            mapping_basis = "generated SVG row index equals corrected dense-grid cell_index"
        if len(matches) > 1:
            raise AssertionError(f"S12/S13 objective-flag mapping is not one-to-zero-or-one: {item.source_id}")
        match = matches.iloc[0] if len(matches) == 1 else None
        rows.append(
            {
                "time": float(item.time),
                "s12_source": str(item.source),
                "s12_source_id": str(item.source_id),
                "s12_row_index_within_panel": int(item.row_index_within_panel),
                "celltype": str(item.celltype),
                "s12_x": float(item.x),
                "s12_y": float(item.y),
                "s12_nearest_neighbor_distance": float(item.nearest_neighbor_distance),
                "s12_robust_nn_z": float(item.robust_nn_z),
                "mapping_basis": mapping_basis,
                "in_s13_seed42_sample": match is not None,
                "s13_display_sample_rank": None if match is None else int(match["display_sample_rank"]),
                "s13_cell_index": None if match is None else int(match["cell_index"]),
                "s13_x": None if match is None else float(match["x"]),
                "s13_y": None if match is None else float(match["y"]),
                "s13_growth": None if match is None else float(match["growth"]),
                "s13_objective_isolation_flag": False if match is None else bool(match["objective_isolation_flag"]),
            }
        )
    mapping = pd.DataFrame(rows)
    mapping.to_csv(tables_dir / "s12_s13_objective_display_flag_mapping.csv", index=False)
    return mapping


def plot_s13(
    sample: pd.DataFrame,
    hide_objective_flags: bool,
    name: str,
    directories: dict[str, Path],
    tables_dir: Path,
    *,
    fit_package_native_canvas: bool = False,
    annotate_injury_reference: bool = False,
    annotate_injury_all_panels: bool = False,
    font_family: str = "DejaVu Sans",
) -> tuple[dict[str, Path], pd.DataFrame]:
    configure_review_legacy_style(font_family)
    # The new package-native spatial ranges make equal-aspect panels extend
    # about one point beyond the submitted fixed canvas.  A 0.02-inch vertical
    # figure adjustment restores the measured outer canvas without changing
    # fonts, markers, colormap, normalization, or panel grammar.  Historical
    # callers retain the exact original 12.6 x 12.6-inch figure.
    if fit_package_native_canvas:
        # Arial has a slightly taller tight bounding box than DejaVu Sans for
        # this fixed legacy page.  Compensate by exactly 0.02 inch so the
        # scientific axes and submitted outer canvas remain unchanged.
        figure_height = 12.5535 if str(font_family).lower() == "arial" else 12.58
    else:
        figure_height = 12.6
    fig, axes = plt.subplots(3, 3, figsize=(12.6, figure_height), squeeze=False)
    scale_rows: list[dict] = []
    for ax in axes.flat:
        ax.axis("off")
    for ax, (time, sub) in zip(axes.flat, sample.groupby("time", sort=True)):
        shown = sub[~sub["objective_isolation_flag"]].copy() if hide_objective_flags else sub.copy()
        # Policy B is display-only: use the exact same old seed-42 sample to
        # freeze normalization, then suppress flagged glyphs.  This keeps the
        # colormap contract identical between A and B.
        values = sub["growth"].to_numpy(dtype=float)
        q05, q95 = np.percentile(values, [5, 95])
        if q05 == q95:
            q95 = q05 + np.finfo(float).eps
        norm = Normalize(vmin=float(q05), vmax=float(q95), clip=True)
        scatter = ax.scatter(
            shown["x"],
            shown["y"],
            c=shown["growth"],
            cmap="viridis",
            s=2.0,
            linewidths=0,
            alpha=0.85,
            norm=norm,
        )
        source = "observed" if float(time) in OBSERVED_TIMES else "simulated"
        ax.set_title(f"t={float(time):.1f} ({source})", fontsize=10)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.02)
        cbar.ax.tick_params(labelsize=7)
        annotate_this_panel = bool(
            annotate_injury_reference
            and (
                annotate_injury_all_panels
                or np.isclose(float(time), 0.0)
            )
        )
        if annotate_this_panel:
            # Anatomical locator only: the visible orientation places the
            # injured half-brain on the right, with the resection in its lower
            # portion.  The outline is deliberately schematic and is not
            # estimated from the growth-rate colors.
            injury = Ellipse(
                (0.69, 0.17),
                width=0.28,
                height=0.18,
                angle=-8.0,
                transform=ax.transAxes,
                fill=False,
                edgecolor="#242b83",
                linewidth=1.4,
                linestyle=(0, (4, 3)),
                zorder=20,
            )
            ax.add_patch(injury)
            if np.isclose(float(time), 0.0):
                ax.annotate(
                    "Right hemisphere",
                    xy=(0.76, 0.69),
                    xycoords="axes fraction",
                    xytext=(0.58, 0.92),
                    textcoords="axes fraction",
                    fontsize=8,
                    color="#1f1f1f",
                    ha="left",
                    va="top",
                    arrowprops={"arrowstyle": "-|>", "color": "#1f1f1f", "lw": 0.9},
                    annotation_clip=False,
                    zorder=21,
                )
            ax.annotate(
                "Injury region",
                xy=(0.69, 0.17),
                xycoords="axes fraction",
                xytext=(0.46, 0.035),
                textcoords="axes fraction",
                fontsize=8,
                fontweight="bold",
                color="#242b83",
                ha="left",
                va="bottom",
                arrowprops={"arrowstyle": "-|>", "color": "#242b83", "lw": 1.0},
                annotation_clip=False,
                zorder=21,
            )
        scale_rows.append(
            {
                "time": float(time),
                "source": source,
                "n_compute": int(sub["n_compute_panel"].iloc[0]),
                "n_seed42_sample": int(len(sub)),
                "n_display": int(len(shown)),
                "n_objective_display_hidden": int(len(sub) - len(shown)),
                "q05_display_sample": float(q05),
                "q95_display_sample": float(q95),
            }
        )
    fig.suptitle("Arista growth-rate maps across dense time grid", fontsize=13)
    fig.tight_layout()
    paths = save_mpl_figure(
        fig,
        name,
        directories,
        submitted_canvas_pt=S13_SUBMITTED_CANVAS_PT,
        submitted_raster_px=S13_SUBMITTED_RASTER_PX,
    )
    plt.close(fig)
    scales = pd.DataFrame(scale_rows)
    suffix = "B_nnmad20" if hide_objective_flags else "A_all_valid"
    scales.to_csv(tables_dir / f"s13_scale_and_display_counts_{suffix}.csv", index=False)
    return paths, scales


def calculate_corrected_composition(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    table = pd.read_csv(path)
    required = {"time", "celltype", "count", "fraction", "total"}
    missing = required - set(table.columns)
    if missing:
        raise KeyError(f"Corrected composition table lacks columns: {sorted(missing)}")
    count_table = table.pivot(index="time", columns="celltype", values="count").fillna(0).astype(int).sort_index()
    fraction_table = table.pivot(index="time", columns="celltype", values="fraction").fillna(0.0).sort_index()
    if tuple(count_table.index.astype(float)) != DENSE_TIMES:
        raise AssertionError("Corrected composition time grid changed")
    if not np.array_equal(count_table.sum(axis=1).to_numpy(), np.full(len(DENSE_TIMES), 7668)):
        raise AssertionError("S14 corrected fixed-particle cohort is not 7,668 at every time")
    if not np.allclose(fraction_table.sum(axis=1), 1.0, rtol=0.0, atol=1e-8):
        raise AssertionError("Corrected composition fractions do not sum to one")
    return count_table, fraction_table


def plot_s14b(
    fraction_table: pd.DataFrame,
    palette: dict[str, str],
    name: str,
    directories: dict[str, Path],
    tables_dir: Path,
) -> tuple[dict[str, Path], pd.DataFrame]:
    configure_review_legacy_style()
    validate_labels(fraction_table.columns, palette, "corrected S14b composition")
    global_order = fraction_table.mean(axis=0).sort_values(ascending=False)
    selected = list(global_order.head(min(15, len(global_order))).index)
    display = fraction_table.copy()
    if len(selected) < display.shape[1]:
        display["Other"] = display.drop(columns=selected).sum(axis=1)
        display = display[selected + ["Other"]]
    else:
        display = display[selected]
    display_pct = display * 100.0
    colors = [palette.get(label, "#c9c3b8" if label == "Other" else "#808080") for label in display_pct.columns]
    fig, ax = plt.subplots(figsize=(11.0, 4.8), facecolor="white")
    bottom = np.zeros(display_pct.shape[0], dtype=float)
    x = np.arange(display_pct.shape[0], dtype=float)
    for cell_type, color in zip(display_pct.columns, colors):
        values = display_pct[cell_type].to_numpy(dtype=float)
        ax.bar(
            x,
            values,
            bottom=bottom,
            width=0.76,
            color=color,
            edgecolor="white",
            linewidth=0.6,
            label=cell_type,
        )
        bottom += values
    ax.set_xlim(-0.55, len(x) - 0.45)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Cell proportion (%)")
    ax.set_xlabel("Time")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{time:.2f}" for time in display_pct.index], rotation=0)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.grid(axis="y", color="#e9e3d8", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(
        "Arista cell composition across observed and interpolated time points",
        loc="left",
        fontsize=12.5,
    )
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
        title="Cell type",
        fontsize=8,
        title_fontsize=8.5,
    )
    fig.tight_layout()
    paths = save_mpl_figure(fig, name, directories)
    plt.close(fig)
    display_pct.to_csv(tables_dir / "s14b_corrected_top15_other_percent.csv", index_label="time")
    return paths, display_pct


def copy_s14a_sources(formal_v3: Path, name: str, directories: dict[str, Path]) -> dict[str, Path]:
    paths = {
        "svg": directories["vector"] / f"{name}.svg",
        "pdf": directories["pdf"] / f"{name}.pdf",
        "png": directories["png"] / f"{name}.png",
        "jpg": directories["jpeg"] / f"{name}.jpg",
    }
    shutil.copy2(formal_v3 / "lineage_sankey.svg", paths["svg"])
    shutil.copy2(formal_v3 / "lineage_sankey.pdf", paths["pdf"])
    shutil.copy2(formal_v3 / "lineage_sankey.png", paths["png"])
    save_jpeg(paths["png"], paths["jpg"])
    svg_text = paths["svg"].read_text(encoding="utf-8")
    required_fragments = [
        'viewBox="0 0 1600 1000"',
        "Cell Fate Transitions",
        'font-family: Arial',
        'class="sankey-link"',
        'class="sankey-node"',
        ">0.0<",
        ">4.0<",
    ]
    missing = [fragment for fragment in required_fragments if fragment not in svg_text]
    if missing:
        raise AssertionError(f"Corrected S14a no longer satisfies the legacy Plotly contract: {missing}")
    return paths


def nested_svg(path: Path, x: float, y: float, width: float, height: float, view_box: str | None = None) -> ET.Element:
    node = ET.parse(path).getroot()
    node.set("x", f"{x:.6f}")
    node.set("y", f"{y:.6f}")
    node.set("width", f"{width:.6f}")
    node.set("height", f"{height:.6f}")
    if view_box is not None:
        node.set("viewBox", view_box)
    node.set("preserveAspectRatio", "xMidYMid meet")
    return node


def render_s14_composite(
    s14a_svg: Path,
    s14b_svg: Path,
    name: str,
    directories: dict[str, Path],
) -> dict[str, Path]:
    svg_ns = "http://www.w3.org/2000/svg"
    ET.register_namespace("", svg_ns)
    ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")
    root = ET.Element(
        f"{{{svg_ns}}}svg",
        {
            "width": "560pt",
            "height": "576pt",
            "viewBox": "0 0 560 576",
            "version": "1.1",
        },
    )
    ET.SubElement(root, f"{{{svg_ns}}}rect", {"x": "0", "y": "0", "width": "560", "height": "576", "fill": "white"})
    # The old compact composite cropped the Plotly top/bottom whitespace and
    # added the title/panel label in the composite canvas.
    root.append(nested_svg(s14a_svg, 0.0, 0.0, 560.0, 297.5, view_box="0 95 1600 850"))
    title = ET.SubElement(
        root,
        f"{{{svg_ns}}}text",
        {
            "x": "280",
            "y": "13",
            "text-anchor": "middle",
            "font-family": "Arial, sans-serif",
            "font-size": "8",
            "fill": "#2a3f5f",
        },
    )
    title.text = "Cell Fate Transitions"
    panel_a = ET.SubElement(
        root,
        f"{{{svg_ns}}}text",
        {"x": "0", "y": "18", "font-family": "Arial, sans-serif", "font-size": "20", "font-weight": "bold", "fill": "black"},
    )
    panel_a.text = "a"
    root.append(nested_svg(s14b_svg, 24.0, 337.0, 530.0, 228.0))
    panel_b = ET.SubElement(
        root,
        f"{{{svg_ns}}}text",
        {"x": "0", "y": "349", "font-family": "Arial, sans-serif", "font-size": "20", "font-weight": "bold", "fill": "black"},
    )
    panel_b.text = "b"

    paths = {
        "svg": directories["vector"] / f"{name}.svg",
        "pdf": directories["pdf"] / f"{name}.pdf",
        "png": directories["png"] / f"{name}.png",
        "jpg": directories["jpeg"] / f"{name}.jpg",
    }
    ET.ElementTree(root).write(paths["svg"], encoding="utf-8", xml_declaration=True)
    renderer = shutil.which("rsvg-convert")
    if renderer is None:
        raise RuntimeError("rsvg-convert is required for standalone S14 PDF/PNG rendering")
    subprocess.run([renderer, "-f", "pdf", "-o", str(paths["pdf"]), str(paths["svg"])], check=True)
    normalize_pdf_metadata(paths["pdf"])
    subprocess.run(
        [renderer, "-w", "2333", "-h", "2400", "-o", str(paths["png"]), str(paths["svg"])],
        check=True,
    )
    save_jpeg(paths["png"], paths["jpg"])
    return paths


def make_contact_sheet(
    left_path: Path,
    right_path: Path,
    out_path: Path,
    left_label: str,
    right_label: str,
    target_width_each: int = 1150,
) -> None:
    images = [Image.open(left_path).convert("RGB"), Image.open(right_path).convert("RGB")]
    resized = []
    for image in images:
        height = int(round(image.height * target_width_each / image.width))
        resized.append(image.resize((target_width_each, height), Image.Resampling.LANCZOS))
    header = 70
    canvas = Image.new("RGB", (target_width_each * 2, max(image.height for image in resized) + header), "white")
    font_path = Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans.ttf"
    font = ImageFont.truetype(str(font_path), 28)
    draw = ImageDraw.Draw(canvas)
    for column, (image, label) in enumerate(zip(resized, (left_label, right_label))):
        x = column * target_width_each
        canvas.paste(image, (x, header))
        box = draw.textbbox((0, 0), label, font=font)
        draw.text((x + (target_width_each - (box[2] - box[0])) / 2, 18), label, fill="black", font=font)
    canvas.save(out_path, format="PNG", dpi=(150, 150))


def image_metrics(path: Path) -> dict:
    image = Image.open(path)
    return {"width_px": int(image.width), "height_px": int(image.height), "mode": image.mode}


def pdf_metrics(path: Path) -> dict:
    reader = PdfReader(str(path))
    page = reader.pages[0]
    return {
        "pages": len(reader.pages),
        "width_pt": float(page.mediabox.width),
        "height_pt": float(page.mediabox.height),
    }


def normalize_pdf_metadata(path: Path) -> None:
    """Remove renderer wall-clock metadata while preserving vector content."""

    reader = PdfReader(str(path))
    writer = PdfWriter(clone_from=reader)
    writer.metadata = None
    writer.add_metadata(
        {
            "/Creator": "ARISTA corrected legacy-style renderer",
            "/Producer": "rsvg-convert/cairo normalized by pypdf",
            "/CreationDate": "D:20260823000000+00'00'",
            "/ModDate": "D:20260823000000+00'00'",
        }
    )
    # Cairo's trailer ID incorporates volatile render metadata.  Omitting it
    # makes independent renders byte-identical after the fixed Info dictionary.
    writer._ID = None
    temporary = path.with_suffix(path.suffix + ".normalized")
    with temporary.open("wb") as handle:
        writer.write(handle)
    temporary.replace(path)


def copy_figure_alias(
    source_paths: dict[str, Path],
    alias_name: str,
    directories: dict[str, Path],
) -> dict[str, Path]:
    destination = {
        "svg": directories["vector"] / f"{alias_name}.svg",
        "pdf": directories["pdf"] / f"{alias_name}.pdf",
        "png": directories["png"] / f"{alias_name}.png",
        "jpg": directories["jpeg"] / f"{alias_name}.jpg",
    }
    for extension, path in destination.items():
        shutil.copy2(source_paths[extension], path)
    return destination


def copy_named_figure_from_bundle(
    accepted_bundle: Path,
    stem: str,
    directories: dict[str, Path],
) -> dict[str, Path]:
    source = {
        "svg": accepted_bundle / "figures/vector" / f"{stem}.svg",
        "pdf": accepted_bundle / "figures/pdf" / f"{stem}.pdf",
        "png": accepted_bundle / "figures/png" / f"{stem}.png",
        "jpg": accepted_bundle / "figures/jpeg" / f"{stem}.jpg",
    }
    require_files(source.values())
    destination = {
        "svg": directories["vector"] / f"{stem}.svg",
        "pdf": directories["pdf"] / f"{stem}.pdf",
        "png": directories["png"] / f"{stem}.png",
        "jpg": directories["jpeg"] / f"{stem}.jpg",
    }
    for extension in source:
        shutil.copy2(source[extension], destination[extension])
        if sha256(source[extension]) != sha256(destination[extension]):
            raise AssertionError(f"Copied figure is not byte-identical: {source[extension]}")
    return destination


def copy_v3_s14_and_s13a(
    accepted_bundle: Path,
    directories: dict[str, Path],
) -> tuple[dict[str, Path], dict[str, Path], dict[str, dict]]:
    s13_a_name = "FigureS13_ARISTA_corrected_oldstyle_A_all_valid_seed42_n2500"
    s14_stems = (
        "PanelS14a_ARISTA_corrected_oldstyle_fixed_particle_lineage",
        "PanelS14b_ARISTA_corrected_oldstyle_fixed7668",
        "FigureS14_ARISTA_corrected_oldstyle_lineage_composition",
    )
    s13_a_paths = copy_named_figure_from_bundle(accepted_bundle, s13_a_name, directories)
    copied: dict[str, dict] = {}
    for extension, destination in s13_a_paths.items():
        source = accepted_bundle / "figures" / ("vector" if extension == "svg" else "jpeg" if extension == "jpg" else extension) / destination.name
        copied[str(destination)] = {"source": str(source), "sha256": sha256(destination)}
    s14_paths_by_stem: dict[str, dict[str, Path]] = {}
    for stem in s14_stems:
        paths = copy_named_figure_from_bundle(accepted_bundle, stem, directories)
        s14_paths_by_stem[stem] = paths
        for extension, destination in paths.items():
            source = accepted_bundle / "figures" / ("vector" if extension == "svg" else "jpeg" if extension == "jpg" else extension) / destination.name
            copied[str(destination)] = {"source": str(source), "sha256": sha256(destination)}
    for table_name in (
        "s13_scale_and_display_counts_A_all_valid.csv",
        "s14_corrected_fixed_particle_counts_full.csv",
        "s14_corrected_fixed_particle_fractions_full.csv",
        "s14b_corrected_top15_other_percent.csv",
    ):
        source = accepted_bundle / "tables" / table_name
        destination = directories["tables"] / table_name
        require_files([source])
        shutil.copy2(source, destination)
        if sha256(source) != sha256(destination):
            raise AssertionError(f"Copied table is not byte-identical: {source}")
        copied[str(destination)] = {"source": str(source), "sha256": sha256(destination)}
    return (
        s13_a_paths,
        s14_paths_by_stem["FigureS14_ARISTA_corrected_oldstyle_lineage_composition"],
        copied,
    )


def write_determinism_report(output_dir: Path, reference_dir: Path) -> dict:
    reference_dir = reference_dir.expanduser().resolve()
    if not reference_dir.is_dir():
        raise FileNotFoundError(f"Determinism reference is not a directory: {reference_dir}")
    excluded = {"MANIFEST.json", "DETERMINISM_REPORT.json"}
    current = {
        str(path.relative_to(output_dir)): path
        for path in output_dir.rglob("*")
        if path.is_file() and path.name not in excluded
    }
    reference = {
        str(path.relative_to(reference_dir)): path
        for path in reference_dir.rglob("*")
        if path.is_file() and path.name not in excluded
    }
    if set(current) != set(reference):
        raise AssertionError(
            "Determinism artifact inventory differs: "
            f"current_only={sorted(set(current) - set(reference))}, "
            f"reference_only={sorted(set(reference) - set(current))}"
        )
    mismatches = [
        relative
        for relative in sorted(current)
        if sha256(current[relative]) != sha256(reference[relative])
    ]
    report = {
        "status": "PASS" if not mismatches else "FAIL",
        "reference_bundle": str(reference_dir),
        "comparison_policy": "SHA-256 of every artifact except MANIFEST.json and DETERMINISM_REPORT.json",
        "n_artifacts_compared": len(current),
        "n_hash_mismatches": len(mismatches),
        "mismatches": mismatches,
    }
    if mismatches:
        raise AssertionError(f"Independent rebuild is not deterministic: {mismatches}")
    (output_dir / "DETERMINISM_REPORT.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def write_sha256s(output_dir: Path) -> Path:
    excluded = {"MANIFEST.json", "DETERMINISM_REPORT.json", "SHA256SUMS"}
    files = sorted(
        path for path in output_dir.rglob("*") if path.is_file() and path.name not in excluded
    )
    output = output_dir / "SHA256SUMS"
    output.write_text(
        "".join(f"{sha256(path)}  {path.relative_to(output_dir)}\n" for path in files),
        encoding="utf-8",
    )
    return output


def archive_script_snapshot(output_dir: Path) -> Path:
    destination = output_dir / "scripts" / Path(__file__).name
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__).resolve(), destination)
    if sha256(Path(__file__).resolve()) != sha256(destination):
        raise AssertionError("Archived plotting script snapshot differs from the executed builder")
    return destination


def write_provenance(
    output_dir: Path,
    script_snapshot: Path,
    s12_final_paths: dict[str, Path],
    s13_final_paths: dict[str, Path],
    s14_paths: dict[str, Path],
    formal_v3: Path,
    formal_data: dict,
) -> Path:
    stable_bundle = DEFAULT_OUTPUT.resolve()
    checkpoint = formal_data["model"]["weight_checkpoint_sha256"]
    provenance = f"""# Figure provenance

Archived on: `2026-08-23`

Manuscript figure: `ARISTA Supplementary Figures S12--S14`

Scientific claim: Corrected ARISTA spatial, growth, lineage, and composition results are shown with the plotting grammar of the submitted supplementary figures.

## Files

- Vector figures: `{stable_bundle / 'figures/pdf/FigureS12_ARISTA_corrected_oldstyle_FINAL.pdf'}`, `{stable_bundle / 'figures/pdf/FigureS13_ARISTA_corrected_oldstyle_FINAL.pdf'}`, `{stable_bundle / 'figures/pdf/FigureS14_ARISTA_corrected_oldstyle_lineage_composition.pdf'}`
- PNG previews: `{stable_bundle / 'figures/png'}`
- Plotting script: `{stable_bundle / 'scripts' / script_snapshot.name}`
- Caption/source layout: `/Users/zhenyizhang/Desktop/202511/nbme预投稿/投稿/投稿修改/si.pdf`, pages 38--40
- Compiled manuscript or SI: `/Users/zhenyizhang/Desktop/202511/nbme预投稿/投稿/投稿修改/si.pdf`

## Selected experiment

- Local run: `{formal_v3}`
- Server run: accepted corrected formal-v3 state recorded in `{formal_v3 / 'run_manifest.json'}`
- Configuration: `{formal_v3 / 'run_manifest.json'}`
- Manifest: `{formal_v3 / 'run_manifest.json'}`
- Checkpoint SHA-256: `{checkpoint}`
- Training stages and epoch counts: unchanged from the accepted corrected model package

## Panel sources

| Panel | Content | Source files | Calculation |
|---|---|---|---|
| S12 | Observed and generated spatial snapshots | corrected H5AD slices and formal-v3 snapshot source rows | exact old 14-panel scatter grammar; label-blind per-panel robust 1-NN z>20 display filter |
| S13 | Corrected growth-rate grid | `growth_dense_time_grid.csv` and its exact-row augmented table | exact old seed-42 n<=2500 sample and viridis q05--q95 normalization; mapped objective source rows hidden only at display |
| S14a | Cell-fate transitions | accepted v3 fixed-particle lineage figure | byte-identical accepted v3 artifact |
| S14b | Cell-type composition | accepted v3 fixed-particle composition figure | byte-identical accepted v3 artifact |

## Evaluation protocol

- Initial cells or particles: 7,668 persistent particles for lineage; complete corrected panel populations for S12/S13
- Evaluation weights: unchanged from accepted formal-v3
- Growth handling: all 82,329 S13 rows retained; display sampling and filtering do not change computation
- Time step and diffusion scale: unchanged from accepted formal-v3 manifest
- Seeds: classifier/simulation seeds pinned in the accepted formal-v3 manifest; display sample seed 42
- Uncertainty summary: not applicable to these deterministic display panels

## Rebuild command

```bash
MPLCONFIGDIR=/tmp/arista_mplconfig python {Path(__file__).resolve()} --output-dir {stable_bundle} --isolation-z-threshold 20
```

## Interpretation

The publication aliases use one coordinate-only isolation rule consistently across S12 and the corresponding source rows in S13. All observations and generated rows remain in numerical computations. S14 is unchanged from accepted v3.

## SHA-256

- S12 PDF: `{sha256(s12_final_paths['pdf'])}`
- S12 PNG: `{sha256(s12_final_paths['png'])}`
- S13 PDF: `{sha256(s13_final_paths['pdf'])}`
- S13 PNG: `{sha256(s13_final_paths['png'])}`
- S14 PDF: `{sha256(s14_paths['pdf'])}`
- S14 PNG: `{sha256(s14_paths['png'])}`
- Plotting script: `{sha256(script_snapshot)}`
"""
    destination = output_dir / "PROVENANCE.md"
    destination.write_text(provenance, encoding="utf-8")
    return destination


def reference_derivative_metrics(reference: Path, raw: Path) -> dict:
    ref = np.asarray(Image.open(reference).convert("RGB"), dtype=np.float32)
    src = np.asarray(Image.open(raw).convert("RGB"), dtype=np.float32)
    if ref.shape != src.shape:
        return {"same_shape": False, "reference_shape": list(ref.shape), "raw_shape": list(src.shape)}
    difference = ref - src
    return {
        "same_shape": True,
        "shape": list(ref.shape),
        "mae_rgb": float(np.abs(difference).mean()),
        "rmse_rgb": float(np.sqrt(np.square(difference).mean())),
        "luminance_correlation": float(
            np.corrcoef(ref.reshape(-1, 3).mean(axis=1), src.reshape(-1, 3).mean(axis=1))[0, 1]
        ),
    }


def write_audit_report(
    output_dir: Path,
    outlier_summary: dict,
    s12_inventory_a: pd.DataFrame,
    s12_inventory_b: pd.DataFrame,
    s13_scales_a: pd.DataFrame,
    s13_scales_b: pd.DataFrame,
) -> None:
    report = f"""# ARISTA SI S12--S14 corrected-state / legacy-style audit

## Governing rule

Newest accepted corrected numerical state is retained. The submitted SI plotting grammar is copied exactly where it is code-defined. Historical numerical tables are used only as visual witnesses. This v4 bundle supersedes v3 for S12 and S13 display filtering; S14 is copied byte-for-byte from accepted v3.

## S12 interpolation mosaic

- Corrected source: full corrected observed H5AD slices (46,209 cells total) plus corrected formal-v3 split-SDE generated snapshots from model commit `3c87a3e`.
- Locked legacy grammar: 14 panels in the submitted 4x4 order; 8.8x8.8-inch canvas; one raw-order scatter per panel; point area 2.5 pt^2; alpha 0.9; DejaVu Sans 8-point titles; equal aspect; no axes/legend; tight bounding box.
- Prior corrected-replica failure: that renderer changed point area to 1.25 pt^2, canvas to 9.0x8.45 inches, subplot margins, and category drawing order.
- Outlier policy A retains every valid row as audit evidence. Policy B is the frozen publication renderer: the same label-blind robust 1-NN rule at z>{outlier_summary['threshold']:.0f} is applied independently to every one of the 14 displayed panels, without cell-type labels. It flags exactly {outlier_summary['n_display_flags']} rows ({outlier_summary['flags_by_panel']}); all remain in every computation and table.
- The old 46,189-row roster mask is not applied.
- `FigureS12_ARISTA_corrected_oldstyle_FINAL.*` is a byte-identical alias of policy B.
- The sole generated flag is `generated:0.5:3291` (`nptxEX`), at SVG display coordinates (88.014003, -188.427663), robust z=22.129868. This removes the isolated magenta glyph below the t=0.5 generated tissue without altering the generated state.

## S13 growth grid

- Corrected source: formal-v3 `growth_dense_time_grid.csv`, 82,329 corrected rows across nine time points.
- Locked legacy grammar: seed-42 sorted uniform display sample capped at 2,500 rows per panel; 3x3 12.6x12.6-inch canvas; viridis; independent per-panel 5th--95th percentile scaling on the displayed sample; 2.0 pt^2 points; alpha 0.85; individual colorbars; submitted titles.
- Prior bundle failure: it plotted essentially all 82,329 rows and used full-cohort scale limits, producing visibly denser maps than the submitted SI.
- No targeted 20-row mask is applied. Policy B removes only objective source rows that intersect the already frozen seed-42 display sample; its color normalization remains frozen to that full sample.
- The t=0.5 generated source row maps exactly to corrected dense-grid `cell_index=3291`, display rank 1092, raw coordinate (-0.274431, -0.744711), growth 0.068518. S13 policy B therefore hides four glyphs total: three observed rows plus this simulated row. All 82,329 rows remain in computation.
- `FigureS13_ARISTA_corrected_oldstyle_FINAL.*` is a byte-identical alias of policy B.

## S14 lineage and composition

- Accepted v3 S14 is copied byte-for-byte, including the complete 7,668-particle non-split fixed-particle lineage cohort. Persistent identity is mandatory; split birth/death rows cannot be paired across time as lineage.
- S14a legacy grammar: the existing corrected Plotly `nature-methods` Sankey (Arial, node/link colors, 0.4 link alpha, nine time labels, title) is reused without numerical alteration.
- Corrected S14b source: the same complete 7,668-particle fixed cohort at all nine times, not the legacy review suite's mixed observed/3,072-particle population table.
- S14b legacy grammar: exact historical 11.0x4.8-inch stacked-bar function, top 15 by across-time mean plus Other, white borders, 20% y ticks, old grid/title/legend geometry.
- The old near-square 2-panel composite geometry is restored; the prior 5000x5709 assembly contained excessive vertical whitespace.

## Historical witnesses that must not supply numbers

- `results/arista_review_dense_local`: legacy 3,072-particle simulation and 300-epoch classifier; it is the exact S13/S14b style witness only.
- `ARISTA_S16_formal_k10.png`: filename/content mismatch; its content is the 14-panel S12 interpolation mosaic. Its run summary says raw MLP k=1 and an older r10 rollout, so it is a layout/style witness only.
- Submitted compact S12/S13 JPEGs are near-exact JPEG derivatives of the cited local PNG renderers; their pixel dimensions are therefore authoritative style targets.

## Display counts

- S12 A observed/generated totals are archived in `tables/s12_panel_inventory_A_all_valid.csv`; S12 B in `tables/s12_panel_inventory_B_nnmad20.csv`. The complete 128,330-row all-panel audit (46,209 observed plus 82,121 split-SDE generated) and exact 10-row flag table are archived separately.
- S13 A/B counts and color limits are archived in the corresponding scale tables. The complete corrected input remains hash-pinned in the manifest.
- The publication aliases select B; A and both contact sheets remain only as transparent QA evidence.
"""
    (output_dir / "AUDIT_REPORT.md").write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not np.isclose(float(args.isolation_z_threshold), ISOLATION_Z_THRESHOLD, rtol=0.0, atol=0.0):
        raise ValueError(
            f"The frozen publication threshold is {ISOLATION_Z_THRESHOLD:g}; "
            "use a separate exploratory renderer for threshold sensitivity."
        )
    output_dir = args.output_dir.expanduser().resolve()
    full_bank = args.full_bank.expanduser().resolve()
    formal_v3 = args.formal_v3.expanduser().resolve()
    augmented_path = args.s13_augmented.expanduser().resolve()
    palette_path = args.palette.expanduser().resolve()
    legacy_extra_path = args.legacy_extra_evidence.expanduser().resolve()
    accepted_v3 = ACCEPTED_V3.resolve()
    formal_growth = formal_v3 / "growth_dense_time_grid.csv"
    formal_composition = formal_v3 / "celltype_composition.csv"
    formal_manifest = formal_v3 / "run_manifest.json"

    input_paths = [
        palette_path,
        legacy_extra_path,
        augmented_path,
        formal_growth,
        formal_composition,
        formal_manifest,
        formal_v3 / "lineage_sankey.svg",
        formal_v3 / "lineage_sankey.pdf",
        formal_v3 / "lineage_sankey.png",
        OLD_S12_REFERENCE,
        OLD_S13_REFERENCE,
        OLD_S14_REFERENCE,
        OLD_S12_RAW,
        OLD_S13_RAW,
        OLD_S14A_RAW,
        OLD_S14B_RAW,
        FORMAL_K10_WITNESS,
        REVIEW_DENSE_MANIFEST,
        accepted_v3 / "MANIFEST.json",
        accepted_v3 / "QA_REPORT.json",
        accepted_v3 / "DETERMINISM_REPORT.json",
    ]
    input_paths.extend(full_bank / "slice_data" / f"time_{time_token(time)}.h5ad" for time in OBSERVED_TIMES)
    input_paths.extend(
        formal_v3 / "snapshots" / (f"time_{time:.1f}__Generated.svg" if time in OBSERVED_TIMES else f"time_{time:.1f}.svg")
        for time in DENSE_TIMES
    )
    require_files(input_paths)
    directories = initialize_output(output_dir)
    palette = load_palette(palette_path)
    panels = load_s12_panels(full_bank, formal_v3, palette)

    outlier_audit, flag_keys, outlier_summary = build_outlier_audit(
        panels,
        palette,
        legacy_extra_path,
        directories["tables"],
        float(args.isolation_z_threshold),
    )
    if len(outlier_audit) != 128330:
        raise AssertionError(f"S12 audit must retain all 128,330 rows, found {len(outlier_audit)}")
    if outlier_summary["n_display_flags"] != 10 or outlier_summary["flags_by_panel"] != {
        "0.5|Generated": 1,
        "2.0|Observed": 4,
        "3.0|Observed": 2,
        "4.0|Observed": 3,
    }:
        raise AssertionError(f"Frozen objective outlier roster changed: {outlier_summary}")
    generated_flags = outlier_audit[
        outlier_audit["objective_display_isolation_flag"]
        & (outlier_audit["source"] == "Generated")
    ]
    if len(generated_flags) != 1:
        raise AssertionError(f"Expected one generated display flag, found {len(generated_flags)}")
    generated_flag = generated_flags.iloc[0]
    expected_generated = {
        "time": 0.5,
        "row_index_within_panel": 3291,
        "source_id": "generated:0.5:3291",
        "celltype": "nptxEX",
        "x": 88.014003,
        "y": -188.427663,
        "nearest_neighbor_distance": 11.14320803450123,
        "robust_nn_z": 22.129868257630903,
    }
    for key in ("time", "x", "y", "nearest_neighbor_distance", "robust_nn_z"):
        if not np.isclose(float(generated_flag[key]), float(expected_generated[key]), rtol=0.0, atol=1e-10):
            raise AssertionError(f"Generated objective flag {key} changed: {generated_flag[key]}")
    for key in ("row_index_within_panel", "source_id", "celltype"):
        if generated_flag[key] != expected_generated[key]:
            raise AssertionError(f"Generated objective flag {key} changed: {generated_flag[key]}")

    s12_a_name = "FigureS12_ARISTA_corrected_oldstyle_A_all_valid"
    s12_b_name = "FigureS12_ARISTA_corrected_oldstyle_B_nnmad20_display_only"
    s12_a_paths, s12_inventory_a = plot_s12(panels, palette, flag_keys, False, s12_a_name, directories)
    s12_b_paths, s12_inventory_b = plot_s12(panels, palette, flag_keys, True, s12_b_name, directories)
    s12_final_name = "FigureS12_ARISTA_corrected_oldstyle_FINAL"
    s12_final_paths = copy_figure_alias(s12_b_paths, s12_final_name, directories)
    s12_inventory_a.to_csv(directories["tables"] / "s12_panel_inventory_A_all_valid.csv", index=False)
    s12_inventory_b.to_csv(directories["tables"] / "s12_panel_inventory_B_nnmad20.csv", index=False)

    s13_a_name = "FigureS13_ARISTA_corrected_oldstyle_A_all_valid_seed42_n2500"
    s13_b_name = "FigureS13_ARISTA_corrected_oldstyle_B_nnmad20_seed42_n2500"
    s14_name = "FigureS14_ARISTA_corrected_oldstyle_lineage_composition"
    s13_a_paths, s14_paths, copied_from_v3 = copy_v3_s14_and_s13a(accepted_v3, directories)
    s13_table = validate_s13_sources(formal_growth, augmented_path)
    s13_sample = build_s13_display_sample(s13_table, outlier_audit, directories["tables"])
    flag_mapping = write_s12_s13_flag_mapping(outlier_audit, s13_sample, directories["tables"])
    if int(s13_sample["objective_isolation_flag"].sum()) != 4:
        raise AssertionError("S13 seed-42 sample must contain exactly four objective display flags")
    generated_mapping = flag_mapping[flag_mapping["s12_source"] == "Generated"]
    if len(generated_mapping) != 1 or not bool(generated_mapping.iloc[0]["in_s13_seed42_sample"]):
        raise AssertionError("Generated S12 objective flag failed to map into the S13 seed-42 sample")
    if int(generated_mapping.iloc[0]["s13_cell_index"]) != 3291 or int(
        generated_mapping.iloc[0]["s13_display_sample_rank"]
    ) != 1092:
        raise AssertionError("Generated S12/S13 source-row mapping changed")
    s13_scales_a = pd.read_csv(directories["tables"] / "s13_scale_and_display_counts_A_all_valid.csv")
    s13_b_paths, s13_scales_b = plot_s13(s13_sample, True, s13_b_name, directories, directories["tables"])
    s13_final_name = "FigureS13_ARISTA_corrected_oldstyle_FINAL"
    s13_final_paths = copy_figure_alias(s13_b_paths, s13_final_name, directories)
    s14b_display = pd.read_csv(
        directories["tables"] / "s14b_corrected_top15_other_percent.csv", index_col="time"
    )

    make_contact_sheet(
        s12_a_paths["png"],
        s12_b_paths["png"],
        directories["qa"] / "S12_A_all_valid_vs_B_objective_nnmad20.png",
        "A — all valid observed/generated rows",
        "B — same per-panel robust 1-NN z > 20",
    )
    make_contact_sheet(
        s13_a_paths["png"],
        s13_b_paths["png"],
        directories["qa"] / "S13_A_all_valid_vs_B_objective_nnmad20.png",
        "A — exact old seed-42 n≤2500 sample",
        "B — mapped source-row robust 1-NN z > 20",
    )

    write_audit_report(
        output_dir,
        outlier_summary,
        s12_inventory_a,
        s12_inventory_b,
        s13_scales_a,
        s13_scales_b,
    )
    formal_data = json.loads(formal_manifest.read_text(encoding="utf-8"))
    script_snapshot = archive_script_snapshot(output_dir)
    provenance_path = write_provenance(
        output_dir,
        script_snapshot,
        s12_final_paths,
        s13_final_paths,
        s14_paths,
        formal_v3,
        formal_data,
    )

    expected_png_dimensions = {
        s12_a_name: [2106, 2093],
        s12_b_name: [2106, 2093],
        s12_final_name: [2106, 2093],
        s13_a_name: [3751, 3606],
        s13_b_name: [3751, 3606],
        s13_final_name: [3751, 3606],
        s14_name: [2333, 2400],
    }
    actual_png_dimensions = {
        s12_a_name: list(Image.open(s12_a_paths["png"]).size),
        s12_b_name: list(Image.open(s12_b_paths["png"]).size),
        s12_final_name: list(Image.open(s12_final_paths["png"]).size),
        s13_a_name: list(Image.open(s13_a_paths["png"]).size),
        s13_b_name: list(Image.open(s13_b_paths["png"]).size),
        s13_final_name: list(Image.open(s13_final_paths["png"]).size),
        s14_name: list(Image.open(s14_paths["png"]).size),
    }
    dimension_checks = {
        name: actual_png_dimensions[name] == expected
        for name, expected in expected_png_dimensions.items()
    }
    if not all(dimension_checks.values()):
        raise AssertionError(
            f"Legacy canvas dimensions changed: expected={expected_png_dimensions}, actual={actual_png_dimensions}"
        )
    if int(s12_inventory_b["n_objective_display_hidden"].sum()) != 10:
        raise AssertionError("S12 publication renderer must hide exactly 10 objective glyphs")
    if int(s13_scales_b["n_objective_display_hidden"].sum()) != 4 or int(
        s13_scales_b["n_display"].sum()
    ) != 22496:
        raise AssertionError("S13 publication renderer must display 22,496 rows after four objective flags")
    if not np.allclose(
        s13_scales_a[["q05_display_sample", "q95_display_sample"]].to_numpy(),
        s13_scales_b[["q05_display_sample", "q95_display_sample"]].to_numpy(),
        rtol=0.0,
        atol=1e-15,
    ):
        raise AssertionError("S13 A/B color normalization changed")

    qa = {
        "status": "PASS",
        "corrected_numeric_sources": {
            "model_package_commit": "3c87a3e",
            "observed_compute_rows": 46209,
            "s12_generated_compute_rows": 82121,
            "s12_total_compute_rows": 128330,
            "s13_compute_rows": 82329,
            "s14_fixed_particles_each_time": 7668,
            "lineage_identity": "non_split_fixed_particles",
        },
        "outlier_audit": outlier_summary,
        "s12": {
            "layout_slots": len(S12_LAYOUT),
            "nonempty_panels": len([item for item in S12_LAYOUT if item is not None]),
            "variant_A_all_valid_observed_display_rows": int(
                s12_inventory_a.loc[s12_inventory_a["source"] == "Observed", "n_display"].sum()
            ),
            "variant_A_all_valid_generated_display_rows": int(
                s12_inventory_a.loc[s12_inventory_a["source"] == "Generated", "n_display"].sum()
            ),
            "variant_B_objective_hidden": int(s12_inventory_b["n_objective_display_hidden"].sum()),
            "variant_B_objective_hidden_by_source": {
                str(source): int(value)
                for source, value in s12_inventory_b.groupby("source")["n_objective_display_hidden"].sum().items()
            },
            "publication_alias": s12_final_name,
            "publication_alias_byte_identical_to_B": all(
                sha256(s12_final_paths[extension]) == sha256(s12_b_paths[extension])
                for extension in ("svg", "pdf", "png", "jpg")
            ),
        },
        "s13": {
            "seed": DISPLAY_SAMPLE_SEED,
            "cap_per_panel": DISPLAY_SAMPLE_CAP,
            "variant_A_display_rows": int(s13_scales_a["n_display"].sum()),
            "variant_B_display_rows": int(s13_scales_b["n_display"].sum()),
            "variant_B_objective_hidden": int(s13_scales_b["n_objective_display_hidden"].sum()),
            "mapped_s12_flags_in_seed42_sample": int(flag_mapping["in_s13_seed42_sample"].sum()),
            "generated_source_row_mapping": {
                "s12_source_id": str(generated_mapping.iloc[0]["s12_source_id"]),
                "s13_cell_index": int(generated_mapping.iloc[0]["s13_cell_index"]),
                "s13_display_sample_rank": int(generated_mapping.iloc[0]["s13_display_sample_rank"]),
                "s13_x": float(generated_mapping.iloc[0]["s13_x"]),
                "s13_y": float(generated_mapping.iloc[0]["s13_y"]),
                "s13_growth": float(generated_mapping.iloc[0]["s13_growth"]),
            },
            "A_B_color_limits_identical_within_csv_roundtrip": bool(
                np.allclose(
                    s13_scales_a[["q05_display_sample", "q95_display_sample"]].to_numpy(),
                    s13_scales_b[["q05_display_sample", "q95_display_sample"]].to_numpy(),
                    rtol=0.0,
                    atol=1e-15,
                )
            ),
            "publication_alias": s13_final_name,
            "publication_alias_byte_identical_to_B": all(
                sha256(s13_final_paths[extension]) == sha256(s13_b_paths[extension])
                for extension in ("svg", "pdf", "png", "jpg")
            ),
        },
        "s14": {
            "composition_row_sums_100": bool(np.allclose(s14b_display.sum(axis=1), 100.0, atol=1e-8)),
            "pdf": pdf_metrics(s14_paths["pdf"]),
            "copied_byte_identical_from_accepted_v3": all(
                sha256(path) == sha256(accepted_v3 / path.relative_to(output_dir))
                for path in output_dir.rglob("*")
                if path.is_file()
                and (
                    path.name.startswith("FigureS14_")
                    or path.name.startswith("PanelS14")
                    or path.name.startswith("s14_")
                    or path.name.startswith("s14b_")
                )
            ),
        },
        "accepted_v3_copy_count": len(copied_from_v3),
        "provenance": str(provenance_path.name),
        "legacy_canvas_dimension_checks": dimension_checks,
        "png_metrics": {
            name: image_metrics(path)
            for name, path in {
                s12_a_name: s12_a_paths["png"],
                s12_b_name: s12_b_paths["png"],
                s12_final_name: s12_final_paths["png"],
                s13_a_name: s13_a_paths["png"],
                s13_b_name: s13_b_paths["png"],
                s13_final_name: s13_final_paths["png"],
                s14_name: s14_paths["png"],
            }.items()
        },
        "submitted_asset_derivative_checks": {
            "S12_reference_vs_old_raw": reference_derivative_metrics(OLD_S12_REFERENCE, OLD_S12_RAW),
            "S13_reference_vs_old_raw": reference_derivative_metrics(OLD_S13_REFERENCE, OLD_S13_RAW),
        },
        "single_page_pdf_checks": {
            path.name: pdf_metrics(path)
            for path in sorted(directories["pdf"].glob("*.pdf"))
        },
    }
    (output_dir / "QA_REPORT.json").write_text(json.dumps(qa, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    write_sha256s(output_dir)
    if args.determinism_reference is not None:
        write_determinism_report(output_dir, args.determinism_reference)

    output_files = sorted(path for path in output_dir.rglob("*") if path.is_file())
    manifest = {
        "created_utc": FIXED_TIMESTAMP,
        "workflow": "ARISTA S12-S14 corrected computation with exact submitted plotting grammar and source-row-consistent objective display filtering",
        "immutable_output": str(output_dir),
        "supersedes": str(accepted_v3),
        "supersession_scope": "S12 and S13 publication display filtering only; S14 is byte-identical to accepted v3",
        "scientific_contract": {
            "accepted_model_package_commit": "3c87a3e",
            "weight_checkpoint_sha256": formal_data["model"]["weight_checkpoint_sha256"],
            "score_checkpoint_sha256": formal_data["model"]["score_checkpoint_sha256"],
            "trajectory_semantics": formal_data["trajectory_semantics"],
            "classifier_knn_neighbors": formal_data["classifier_knn_neighbors"],
            "simulation_seeds": formal_data["simulation_seeds"],
            "all_valid_rows_retained_in_computation": True,
            "s12_compute_rows": 128330,
            "s13_compute_rows": 82329,
        },
        "display_contract": {
            "S12": {
                "legacy_canvas_inches": [8.8, 8.8],
                "layout": [None if item is None else {"time": item[0], "source": item[1]} for item in S12_LAYOUT],
                "point_area_pt2": 2.5,
                "alpha": 0.9,
                "font": "DejaVu Sans",
                "variant_A": "all valid observed and generated rows visible",
                "variant_B": f"same label-blind per-panel display-only robust 1-NN z>{args.isolation_z_threshold:g} across all 14 panels",
                "objective_display_flags": 10,
                "objective_display_flags_by_source": {"Observed": 9, "Generated": 1},
                "publication_FINAL": "byte-identical alias of variant B",
            },
            "S13": {
                "legacy_seed": DISPLAY_SAMPLE_SEED,
                "legacy_uniform_cap_per_panel": DISPLAY_SAMPLE_CAP,
                "legacy_canvas_inches": [12.6, 12.6],
                "colormap": "viridis",
                "per_panel_scale": "5th-95th percentiles of the full seed-42 sample before display-only suppression",
                "point_area_pt2": 2.0,
                "alpha": 0.85,
                "objective_source_rows_in_seed42_sample": 4,
                "generated_mapping": "generated:0.5:3291 -> t=0.5 cell_index=3291, display_sample_rank=1092",
                "publication_FINAL": "byte-identical alias of variant B",
            },
            "S14": {
                "S14a": "corrected nature-methods Plotly Sankey, cropped into submitted compact grammar",
                "S14b": "exact historical 11.0x4.8 stacked-bar function",
                "composite_canvas_pt": [560, 576],
                "review_png_px": [2333, 2400],
                "artifact_policy": "all S14 figure and table artifacts copied byte-identically from accepted v3",
            },
        },
        "historical_numeric_non_use": {
            "arista_review_dense_local": "style witness only; legacy 3072-particle/300-epoch computation not reused",
            "ARISTA_S16_formal_k10.png": "misnamed S12-content visual witness only; older r10/raw-MLP-k1 numbers not reused",
            "legacy_46189_roster_mask": "not applied",
        },
        "inputs": {str(path.resolve()): sha256(path) for path in sorted(set(input_paths))},
        "copied_artifacts_from_accepted_v3": {
            str(Path(destination).relative_to(output_dir)): data
            for destination, data in sorted(copied_from_v3.items())
        },
        "plotting_script_snapshot": {
            "path": str(script_snapshot.relative_to(output_dir)),
            "sha256": sha256(script_snapshot),
        },
        "outputs": {
            str(path.relative_to(output_dir)): {
                "sha256": sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for path in output_files
            if path.name != "MANIFEST.json"
        },
        "rebuild_command": (
            f"python {Path(__file__).resolve()} --output-dir {output_dir} "
            f"--isolation-z-threshold {args.isolation_z_threshold:g}"
        ),
    }
    (output_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote immutable ARISTA SI bundle: {output_dir}")
    print(f"S12 A: {s12_a_paths['pdf']}")
    print(f"S12 B: {s12_b_paths['pdf']}")
    print(f"S12 FINAL: {s12_final_paths['pdf']}")
    print(f"S13 A: {s13_a_paths['pdf']}")
    print(f"S13 B: {s13_b_paths['pdf']}")
    print(f"S13 FINAL: {s13_final_paths['pdf']}")
    print(f"S14: {s14_paths['pdf']}")


if __name__ == "__main__":
    main()
