#!/usr/bin/env python3
"""Assemble corrected MOSTA Fig. 4c values in the original Illustrator grammar.

This is a render-only program.  It never loads a model, reruns a trajectory, or
reclassifies a cell.  Numerical truth comes from one frozen, hashed render-state
NPZ exported from the accepted package run.  Visual truth comes from the PDF-
compatible original Illustrator file and the historical plotting script.

The dense spatial scatter is intentionally rasterized, matching the original
Illustrator asset.  Text, centroid markers, and transition ribbons remain
vector objects in PDF/SVG output.  Spatial coordinates receive only the common
equal-aspect display transform and the historical x-translation that places the
target slice beside the source slice; no rotation, anisotropic stretch, warp,
or category-specific displacement is applied.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import date
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
from typing import Any

import numpy as np


SOURCE_LABEL = "Cartilage primordium"
EXPECTED_TOP3 = ("Cartilage", "Cartilage primordium", "Connective tissue")

# Measured directly from Figure_mouse1.ai (PDF-compatible Illustrator file).
ORIGINAL_AI_SHA256 = "340a5ed88dc911d6923bc6b21cf1ceb39fdbef16edf2e325822a5b422045cbc2"
ORIGINAL_SCATTER_XREF = 151
ORIGINAL_SCATTER_SIZE = (1273, 844)
ORIGINAL_SCATTER_BBOX = (
    316.0863952636719,
    237.58531188964844,
    583.919189453125,
    415.15863037109375,
)

# Tight standalone crop around panel c in the original A4 artboard, in PDF pt.
PANEL_RECT = (316.0863952636719, 202.0, 595.0, 440.0)

# Text baselines from the original AI, in page coordinates (PDF pt).
TEXT_BASELINES = {
    "panel": (330.0672302246094, 219.66302490234375),
    "title": (344.2571105957031, 220.50885009765625),
    "generated": (349.2779846191406, 239.80859375),
    "observed": (494.4375, 239.80499267578125),
    "stage_source": (361.99798583984375, 432.4085998535156),
    "stage_target": (504.80999755859375, 432.4085693359375),
    "source_label": (332.6986999511719, 336.1658935546875),
    "pct_connective_tissue": (375.376708984375, 367.88189697265625),
    "pct_cartilage_primordium": (442.7687072753906, 342.2679138183594),
    "pct_cartilage": (515.0927124023438, 348.2709045410156),
    "label_cartilage_primordium": (462.01971435546875, 319.5609130859375),
    "label_cartilage": (550.4987182617188, 333.4208984375),
    "label_connective_tissue_1": (505.6723937988281, 367.8855895996094),
    "label_connective_tissue_2": (505.6723937988281, 378.68560791015625),
}

# Original AI vector grammar, retained as audit evidence.  New curve endpoints
# are computed from the current numerical centroids rather than copied from here.
ORIGINAL_VECTOR_REFERENCE = {
    "ribbon_color": "#3cb44b",
    "ribbon_width_rule": "2 + 18 * transition_probability",
    "ribbon_curvature": 0.20,
    "source_marker_diameter_pt": 3.92,
    "target_marker_diameter_pt": 3.396,
    "source_marker_edge_pt": 0.6,
    "target_marker_edge_pt": 0.5,
    "old_ribbons": [
        {
            "target": "Cartilage primordium",
            "start": [380.78851318359375, 345.04962158203125],
            "end": [525.740478515625, 328.2066345214844],
            "width_pt": 9.633999824523926,
        },
        {
            "target": "Connective tissue",
            "start": [380.6860046386719, 345.28521728515625],
            "end": [499.20599365234375, 366.7462158203125],
            "width_pt": 6.770999908447266,
        },
        {
            "target": "Cartilage",
            "start": [380.7626037597656, 345.134521484375],
            "end": [555.7515869140625, 342.5495300292969],
            "width_pt": 5.1620001792907715,
        },
    ],
}

# For each original AI cubic, subtract the straight-line control points at 1/3
# and 2/3.  The two residuals are nearly identical, confirming that the old
# curves were a consistent routed visual arc rather than a biological path.
# Applying the same category-specific residual to the new straight connection
# preserves the AI lane geometry while keeping both current centroids exact.
ORIGINAL_CURVE_RESIDUAL_PT = {
    "Cartilage primordium": (2.20851643880206, 19.104284922281883),
    "Connective tissue": (-2.817316691080743, 15.59332784016928),
    "Cartilage": (0.3396759033203125, 23.121673583984375),
}

# Direct category-specific cubic templates measured from the old AI: control-x
# locations are normalized along each old source-to-target chord, while control-y
# offsets remain in page points relative to the old source marker.  This retains
# the recognizable upper/middle/lower ribbon lanes even when the new E15.5 target
# centroids are spatially closer than the old E14.5 centroids.
ORIGINAL_CUBIC_LANE_TEMPLATE = {
    "Cartilage primordium": {
        "control_x_fraction": (0.3486474773343102, 0.6819922373947778),
        "control_y_offset_from_source_pt": (13.727890014648438, 8.112899780273438),
    },
    "Connective tissue": {
        "control_x_fraction": (0.3095069634577474, 0.6428369886867553),
        "control_y_offset_from_source_pt": (23.208480834960938, 30.362472534179688),
    },
    "Cartilage": {
        "control_x_fraction": (0.3352872982134699, 0.6686402518876422),
        "control_y_offset_from_source_pt": (22.567291259765625, 21.706298828125),
    },
}

ORIGINAL_CUBIC_POINTS_PAGE = {
    "Cartilage primordium": (
        (380.78851318359375, 345.04962158203125),
        (431.3265075683594, 358.54962158203125),
        (479.64453125, 352.93463134765625),
        (525.740478515625, 328.2066345214844),
    ),
    "Connective tissue": (
        (380.6860046386719, 345.28521728515625),
        (417.3690185546875, 368.03021240234375),
        (456.8760070800781, 375.1842041015625),
        (499.20599365234375, 366.7462158203125),
    ),
    "Cartilage": (
        (380.7626037597656, 345.134521484375),
        (439.43359375, 367.3895263671875),
        (497.76361083984375, 366.5285339355469),
        (555.7515869140625, 342.5495300292969),
    ),
}

FONT_REGULAR = Path("/System/Library/Fonts/Supplemental/Arial.ttf")
FONT_BOLD = Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf")


@dataclass(frozen=True)
class Transition:
    target_label: str
    count: int
    probability: float
    centroid_xy: tuple[float, float]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def strip_hex_alpha(color: str) -> str:
    value = str(color).strip()
    if value.startswith("#") and len(value) == 9:
        return value[:7]
    return value


def page_to_panel(point: tuple[float, float]) -> tuple[float, float]:
    return (point[0] - PANEL_RECT[0], point[1] - PANEL_RECT[1])


def cubic_point(
    points: tuple[tuple[float, float], ...], t: float
) -> tuple[float, float]:
    omt = 1.0 - t
    weights = (omt**3, 3.0 * omt**2 * t, 3.0 * omt * t**2, t**3)
    return (
        float(sum(weight * point[0] for weight, point in zip(weights, points))),
        float(sum(weight * point[1] for weight, point in zip(weights, points))),
    )


def cubic_y_at_x(points: tuple[tuple[float, float], ...], x: float) -> float:
    if not (points[0][0] <= x <= points[-1][0]):
        raise ValueError(f"x={x} is outside monotone cubic chord {points[0][0]}..{points[-1][0]}")
    lower, upper = 0.0, 1.0
    for _ in range(80):
        middle = 0.5 * (lower + upper)
        middle_x, _ = cubic_point(points, middle)
        if middle_x < x:
            lower = middle
        else:
            upper = middle
    return cubic_point(points, 0.5 * (lower + upper))[1]


def validate_original_ai(path: Path) -> dict[str, Any]:
    import fitz

    if sha256(path) != ORIGINAL_AI_SHA256:
        raise RuntimeError("Original AI hash changed; refusing an unaudited style source.")
    document = fitz.open(path)
    if document.page_count != 1:
        raise RuntimeError("Expected the original AI to contain exactly one PDF page.")
    page = document[0]
    matches = [
        info for info in page.get_image_info(xrefs=True)
        if int(info.get("xref", -1)) == ORIGINAL_SCATTER_XREF
    ]
    if len(matches) != 1:
        raise RuntimeError("Could not uniquely locate the original Fig. 4c scatter image.")
    image = matches[0]
    bbox = tuple(float(value) for value in image["bbox"])
    if tuple(int(image[key]) for key in ("width", "height")) != ORIGINAL_SCATTER_SIZE:
        raise RuntimeError("Original Fig. 4c scatter image dimensions changed.")
    if not np.allclose(bbox, ORIGINAL_SCATTER_BBOX, atol=1e-5, rtol=0):
        raise RuntimeError("Original Fig. 4c scatter image placement changed.")
    result = {
        "page_size_pt": [float(page.rect.width), float(page.rect.height)],
        "scatter_xref": ORIGINAL_SCATTER_XREF,
        "scatter_size_px": list(ORIGINAL_SCATTER_SIZE),
        "scatter_bbox_pt": list(bbox),
        "drawing_count": len(page.get_drawings()),
        "image_count": len(page.get_images(full=True)),
    }
    document.close()
    return result


def load_numeric_state(
    render_state: Path,
    render_manifest_path: Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any], list[Transition]]:
    manifest = json.loads(render_manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "COMPLETE":
        raise RuntimeError("Server render-state manifest is not COMPLETE.")
    expected_hash = manifest["outputs"]["render_state_sha256"]
    if sha256(render_state) != expected_hash:
        raise RuntimeError("Render-state SHA-256 does not match the server manifest.")
    if manifest.get("classifier") != "latest accepted 52D cache":
        raise RuntimeError("Render state is not tied to the latest accepted 52D classifier.")
    if int(manifest.get("classifier_knn_neighbors", -1)) != 10:
        raise RuntimeError("This assembly requires the user-selected k=10 identity rule.")
    if float(manifest.get("source_time", np.nan)) != 2.5:
        raise RuntimeError("Unexpected source time; expected t=2.5 (E15.0).")
    if float(manifest.get("target_time", np.nan)) != 3.0:
        raise RuntimeError("Unexpected target time; expected t=3.0 (E15.5).")
    if manifest.get("trajectory") != "global_t0 fixed particle; no restart; no warp":
        raise RuntimeError("Unexpected trajectory provenance.")

    with np.load(render_state, allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]) for key in archive.files}
    required = {
        "source_background_spatial",
        "selected_source_spatial",
        "target_spatial",
        "target_labels",
        "observed_target_spatial",
        "selected_lineage_id",
    }
    if set(arrays) != required:
        raise RuntimeError(f"Unexpected render-state fields: {sorted(arrays)}")

    source = np.asarray(arrays["selected_source_spatial"][:, :2], dtype=np.float32)
    target = np.asarray(arrays["target_spatial"][:, :2], dtype=np.float32)
    labels = np.asarray(arrays["target_labels"]).astype(str)
    lineage_id = np.asarray(arrays["selected_lineage_id"], dtype=np.int64)
    if not (len(source) == len(target) == len(labels) == len(lineage_id)):
        raise RuntimeError("Selected source/target arrays are not row-aligned.")
    if len(np.unique(lineage_id)) != len(lineage_id):
        raise RuntimeError("Lineage identifiers are not unique.")
    if len(source) != int(manifest["source_count"]):
        raise RuntimeError("Source count disagrees with the server manifest.")

    transitions: list[Transition] = []
    unique, counts = np.unique(labels, return_counts=True)
    for label, count in sorted(
        zip(unique.tolist(), counts.tolist()), key=lambda item: item[1], reverse=True
    ):
        mask = labels == label
        centroid = target[mask].mean(axis=0)
        transitions.append(
            Transition(
                target_label=str(label),
                count=int(count),
                probability=float(count) / float(len(labels)),
                centroid_xy=(float(centroid[0]), float(centroid[1])),
            )
        )
    if tuple(item.target_label for item in transitions[:3]) != EXPECTED_TOP3:
        raise RuntimeError(
            "Validated top-three target order changed: "
            f"{tuple(item.target_label for item in transitions[:3])}"
        )
    for computed, recorded in zip(transitions[:3], manifest["top3"]):
        if computed.target_label != recorded["target_label"]:
            raise RuntimeError("Top-three label differs from the server manifest.")
        if computed.count != int(recorded["count"]):
            raise RuntimeError("Top-three count differs from the server manifest.")
        if not np.isclose(computed.probability, float(recorded["probability"]), atol=1e-15):
            raise RuntimeError("Top-three probability differs from the server manifest.")

    normalized = {
        "source_background_spatial": np.asarray(
            arrays["source_background_spatial"][:, :2], dtype=np.float32
        ),
        "selected_source_spatial": source,
        "target_spatial": target,
        "target_labels": labels,
        "observed_target_spatial": np.asarray(
            arrays["observed_target_spatial"][:, :2], dtype=np.float32
        ),
        "selected_lineage_id": lineage_id,
    }
    return normalized, manifest, transitions


def create_scatter_layer(
    arrays: dict[str, np.ndarray],
    transitions: list[Transition],
    palette: dict[str, str],
    output: Path,
) -> tuple[dict[str, tuple[float, float]], dict[str, Any]]:
    # The seed, point sizes, colors, alpha, and 20k background cap are copied
    # from evaluation/mosta/code/mosta_cartilage_lineage_transition_0_1_local.py.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    seed = 42
    background_cap = 20_000
    background_color = "#6f6f6f"
    background_alpha = 0.25
    background_size = 2.0
    lineage_size = 4.0
    lineage_alpha = 0.90
    gap_fraction = 0.15

    source_background = arrays["source_background_spatial"]
    source_selected = arrays["selected_source_spatial"]
    target = arrays["target_spatial"]
    labels = arrays["target_labels"]
    observed_target = arrays["observed_target_spatial"]
    all_spatial = (source_background, source_selected, target, observed_target)

    x_min = float(min(item[:, 0].min() for item in all_spatial))
    x_max = float(max(item[:, 0].max() for item in all_spatial))
    y_min = float(min(item[:, 1].min() for item in all_spatial))
    y_max = float(max(item[:, 1].max() for item in all_spatial))
    x_span = max(1e-6, x_max - x_min)
    gap = gap_fraction * x_span
    shift = x_span + gap

    rng = np.random.default_rng(seed)
    if len(source_background) > background_cap:
        source_indices = rng.choice(len(source_background), background_cap, replace=False)
        source_background_visible = source_background[source_indices]
    else:
        source_background_visible = source_background
    if len(observed_target) > background_cap:
        target_indices = rng.choice(len(observed_target), background_cap, replace=False)
        observed_target_visible = observed_target[target_indices]
    else:
        observed_target_visible = observed_target

    kept = {item.target_label for item in transitions[:3]}
    target_keep = np.asarray([label in kept for label in labels], dtype=bool)
    target_visible = target[target_keep].copy()
    target_visible[:, 0] += shift
    observed_target_shifted = observed_target_visible.copy()
    observed_target_shifted[:, 0] += shift

    width_px, height_px = ORIGINAL_SCATTER_SIZE
    dpi = 150
    fig = plt.figure(figsize=(width_px / dpi, height_px / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.scatter(
        source_background_visible[:, 0], source_background_visible[:, 1],
        s=background_size, c=background_color, alpha=background_alpha,
        linewidths=0, rasterized=True,
    )
    ax.scatter(
        observed_target_shifted[:, 0], observed_target_shifted[:, 1],
        s=background_size, c=background_color, alpha=background_alpha,
        linewidths=0, rasterized=True,
    )
    ax.scatter(
        source_selected[:, 0], source_selected[:, 1],
        s=lineage_size, c=strip_hex_alpha(palette[SOURCE_LABEL]),
        alpha=lineage_alpha, linewidths=0, rasterized=True,
    )
    target_colors = [strip_hex_alpha(palette.get(label, "#888888")) for label in labels[target_keep]]
    ax.scatter(
        target_visible[:, 0], target_visible[:, 1],
        s=lineage_size, c=target_colors, alpha=lineage_alpha,
        linewidths=0, rasterized=True,
    )
    ax.set_xlim(x_min, x_min + 2.0 * x_span + gap)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    fig.canvas.draw()

    source_centroid = source_selected.mean(axis=0)
    data_centroids: dict[str, tuple[float, float]] = {
        SOURCE_LABEL + "__source": (float(source_centroid[0]), float(source_centroid[1]))
    }
    for transition in transitions[:3]:
        data_centroids[transition.target_label] = transition.centroid_xy

    page_centroids: dict[str, tuple[float, float]] = {}
    scatter_x0, scatter_y0, scatter_x1, scatter_y1 = ORIGINAL_SCATTER_BBOX
    for label, centroid in data_centroids.items():
        display_xy = np.asarray(centroid, dtype=float).copy()
        if not label.endswith("__source"):
            display_xy[0] += shift
        pixel_x, pixel_y = ax.transData.transform(display_xy)
        page_x = scatter_x0 + pixel_x / width_px * (scatter_x1 - scatter_x0)
        page_y = scatter_y1 - pixel_y / height_px * (scatter_y1 - scatter_y0)
        page_centroids[label] = (float(page_x), float(page_y))

    fig.savefig(output, dpi=dpi, facecolor="white", transparent=False)
    plt.close(fig)

    from PIL import Image

    with Image.open(output) as image:
        if image.size != ORIGINAL_SCATTER_SIZE:
            raise RuntimeError(
                f"Scatter layer is {image.size}, expected {ORIGINAL_SCATTER_SIZE}."
            )

    transform = {
        "seed": seed,
        "background_cap_per_slice": background_cap,
        "background_color": background_color,
        "background_alpha": background_alpha,
        "background_size_pt2": background_size,
        "lineage_size_pt2": lineage_size,
        "lineage_alpha": lineage_alpha,
        "gap_fraction": gap_fraction,
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
        "x_span": x_span,
        "target_x_translation": shift,
        "aspect": "equal",
        "rotation": 0,
        "anisotropic_scaling": False,
        "warp": False,
        "source_background_total": int(len(source_background)),
        "source_background_displayed": int(len(source_background_visible)),
        "observed_target_total": int(len(observed_target)),
        "observed_target_displayed": int(len(observed_target_visible)),
        "selected_source_displayed": int(len(source_selected)),
        "selected_target_top3_displayed": int(target_keep.sum()),
        "centroids_data": {label: list(value) for label, value in data_centroids.items()},
        "centroids_page_pt": {label: list(value) for label, value in page_centroids.items()},
    }
    return page_centroids, transform


def assemble_panel(
    scatter_layer: Path,
    page_centroids: dict[str, tuple[float, float]],
    transitions: list[Transition],
    palette: dict[str, str],
    outputs: dict[str, Path],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    from matplotlib.path import Path as MplPath
    from matplotlib.patches import Circle, PathPatch

    matplotlib.rcParams.update(
        {
            "font.family": "Arial",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.unicode_minus": False,
        }
    )
    regular = FontProperties(fname=str(FONT_REGULAR))
    bold = FontProperties(fname=str(FONT_BOLD))

    panel_width = PANEL_RECT[2] - PANEL_RECT[0]
    panel_height = PANEL_RECT[3] - PANEL_RECT[1]
    # Use the effective resolution of the original 1273x844 Illustrator raster
    # (about 342 dpi) so the PDF does not silently downsample the scatter layer.
    composition_dpi = 342.23
    fig = plt.figure(
        figsize=(panel_width / 72.0, panel_height / 72.0),
        dpi=composition_dpi,
    )
    ax = fig.add_axes([0, 0, 1, 1])
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, panel_width)
    ax.set_ylim(panel_height, 0)
    ax.axis("off")

    image = mpimg.imread(scatter_layer)
    scatter_x0, scatter_y0 = page_to_panel((ORIGINAL_SCATTER_BBOX[0], ORIGINAL_SCATTER_BBOX[1]))
    scatter_x1, scatter_y1 = page_to_panel((ORIGINAL_SCATTER_BBOX[2], ORIGINAL_SCATTER_BBOX[3]))
    ax.imshow(
        image,
        extent=(scatter_x0, scatter_x1, scatter_y1, scatter_y0),
        origin="upper",
        interpolation="nearest",
        zorder=1,
    )

    source_page = page_centroids[SOURCE_LABEL + "__source"]
    source = page_to_panel(source_page)
    transition_by_label = {item.target_label: item for item in transitions[:3]}
    source_color = strip_hex_alpha(palette[SOURCE_LABEL])

    # Preserve the original AI's same-color, no-arrow cubic lane grammar.  The
    # endpoints are current numerical centroids; only the old lane residual is
    # transferred to the otherwise straight connection.
    current_cubic_page: dict[str, tuple[tuple[float, float], ...]] = {}
    for label in ("Cartilage primordium", "Connective tissue", "Cartilage"):
        transition = transition_by_label[label]
        target = page_to_panel(page_centroids[label])
        lane = ORIGINAL_CUBIC_LANE_TEMPLATE[label]
        x_fraction_1, x_fraction_2 = lane["control_x_fraction"]
        y_offset_1, y_offset_2 = lane["control_y_offset_from_source_pt"]
        if label == "Connective tissue":
            # The E15.5 CT and Cartilage endpoints are 14.85 pt closer vertically
            # than in the old AI.  Add that missing *lane* separation gradually
            # to the CT control points (1/3 and 2/3), but leave the endpoint exact.
            old_separation = (
                ORIGINAL_CUBIC_POINTS_PAGE["Connective tissue"][-1][1]
                - ORIGINAL_CUBIC_POINTS_PAGE["Cartilage"][-1][1]
            )
            current_separation = (
                page_centroids["Connective tissue"][1]
                - page_centroids["Cartilage"][1]
            )
            missing_lane_separation = max(0.0, old_separation - current_separation)
            y_offset_1 += missing_lane_separation / 3.0
            y_offset_2 += 2.0 * missing_lane_separation / 3.0
        control_1 = (
            source[0] + (target[0] - source[0]) * x_fraction_1,
            source[1] + y_offset_1,
        )
        control_2 = (
            source[0] + (target[0] - source[0]) * x_fraction_2,
            source[1] + y_offset_2,
        )
        path = MplPath(
            [source, control_1, control_2, target],
            [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4],
        )
        current_cubic_page[label] = tuple(
            (point[0] + PANEL_RECT[0], point[1] + PANEL_RECT[1])
            for point in (source, control_1, control_2, target)
        )
        ribbon = PathPatch(
            path,
            fill=False,
            linewidth=2.0 + 18.0 * transition.probability,
            edgecolor=source_color,
            alpha=1.0,
            capstyle="round",
            joinstyle="round",
            zorder=3,
        )
        ax.add_patch(ribbon)

    # Vector markers use the diameters and black outlines measured from the AI.
    ax.add_patch(
        Circle(
            source,
            radius=ORIGINAL_VECTOR_REFERENCE["source_marker_diameter_pt"] / 2.0,
            facecolor=source_color,
            edgecolor="black",
            linewidth=ORIGINAL_VECTOR_REFERENCE["source_marker_edge_pt"],
            zorder=5,
        )
    )
    for label in ("Cartilage primordium", "Connective tissue", "Cartilage"):
        target = page_to_panel(page_centroids[label])
        ax.add_patch(
            Circle(
                target,
                radius=ORIGINAL_VECTOR_REFERENCE["target_marker_diameter_pt"] / 2.0,
                facecolor=strip_hex_alpha(palette[label]),
                edgecolor="black",
                linewidth=ORIGINAL_VECTOR_REFERENCE["target_marker_edge_pt"],
                zorder=5,
            )
        )

    def original_text(
        key: str,
        value: str,
        size: float,
        font: FontProperties = regular,
        color: str = "#000000",
        zorder: int = 7,
    ) -> None:
        x, y = page_to_panel(TEXT_BASELINES[key])
        ax.text(
            x, y, value,
            fontsize=size,
            fontproperties=font,
            color=color,
            ha="left",
            va="baseline",
            zorder=zorder,
        )

    # Exact original AI typography and label positions; only stage and values change.
    original_text("panel", "c", 14, bold)
    original_text("title", "Predicted transition probability", 14, bold)
    original_text("generated", "Generated", 12)
    original_text("observed", "Observed", 12)
    original_text("stage_source", "E15.0", 12, color="#231815")
    original_text("stage_target", "E15.5", 12)
    original_text("source_label", "Cartilage primordium", 9)
    # Keep the old x anchor but retain the exact old label-to-centroid vertical
    # offset after the current CP target centroid moves upward.
    old_cp_dot_y = 327.79119873046875
    old_cp_label_x, old_cp_label_y = TEXT_BASELINES["label_cartilage_primordium"]
    current_cp_label_page = (
        old_cp_label_x,
        page_centroids["Cartilage primordium"][1] + (old_cp_label_y - old_cp_dot_y),
    )
    current_cp_label_x, current_cp_label_y = page_to_panel(current_cp_label_page)
    ax.text(
        current_cp_label_x,
        current_cp_label_y,
        "Cartilage primordium",
        fontsize=9,
        fontproperties=regular,
        color="#000000",
        ha="left",
        va="baseline",
        zorder=7,
    )
    original_text("label_cartilage", "Cartilage", 9)
    original_text("label_connective_tissue_1", "Connective ", 9)
    original_text("label_connective_tissue_2", "tissue", 9)

    probability_keys = {
        "Cartilage primordium": "pct_cartilage_primordium",
        "Connective tissue": "pct_connective_tissue",
        "Cartilage": "pct_cartilage",
    }
    for label, key in probability_keys.items():
        if label == "Connective tissue":
            # The old 26.5% anchor was tied to the old source marker.  Preserve
            # that exact marker-relative offset after the current centroid moves.
            old_source = (379.9425964355469, 344.8217315673828)
            old_anchor = TEXT_BASELINES[key]
            current_x = source_page[0] + (old_anchor[0] - old_source[0])
            current_y = source_page[1] + (old_anchor[1] - old_source[1])
            x, y = page_to_panel((current_x, current_y))
            ax.text(
                x, y,
                f"{100.0 * transition_by_label[label].probability:.1f}%",
                fontsize=9,
                fontproperties=regular,
                color="#000000",
                ha="left",
                va="baseline",
                zorder=7,
            )
        else:
            # Keep the old x anchor and the old vertical offset above its own
            # ribbon, now evaluated on the routed current cubic.
            anchor_x, old_anchor_y = TEXT_BASELINES[key]
            old_curve_y = cubic_y_at_x(ORIGINAL_CUBIC_POINTS_PAGE[label], anchor_x)
            current_curve_y = cubic_y_at_x(current_cubic_page[label], anchor_x)
            current_anchor = (anchor_x, current_curve_y - (old_curve_y - old_anchor_y))
            x, y = page_to_panel(current_anchor)
            ax.text(
                x, y,
                f"{100.0 * transition_by_label[label].probability:.1f}%",
                fontsize=9,
                fontproperties=regular,
                color="#000000",
                ha="left",
                va="baseline",
                zorder=7,
            )

    fig.savefig(outputs["pdf"], facecolor="white")
    fig.savefig(outputs["svg"], facecolor="white")
    fig.savefig(outputs["png"], dpi=480, facecolor="white")
    plt.close(fig)


def freeze_tree(output_dir: Path) -> None:
    for path in output_dir.rglob("*"):
        if path.is_file():
            path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        elif path.is_dir():
            path.chmod(stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
    output_dir.chmod(
        stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--render-state", type=Path, required=True)
    parser.add_argument("--render-state-manifest", type=Path, required=True)
    parser.add_argument("--palette", type=Path, required=True)
    parser.add_argument("--legacy-script", type=Path, required=True)
    parser.add_argument("--legacy-reference-png", type=Path, required=True)
    parser.add_argument("--original-ai", type=Path, required=True)
    parser.add_argument("--manuscript-pdf", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite immutable output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)

    paths = {
        "render_state": args.render_state.expanduser().resolve(),
        "render_state_manifest": args.render_state_manifest.expanduser().resolve(),
        "palette": args.palette.expanduser().resolve(),
        "legacy_script": args.legacy_script.expanduser().resolve(),
        "legacy_reference_png": args.legacy_reference_png.expanduser().resolve(),
        "original_ai": args.original_ai.expanduser().resolve(),
        "manuscript_pdf": args.manuscript_pdf.expanduser().resolve(),
    }
    for label, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label}: {path}")
    if not FONT_REGULAR.is_file() or not FONT_BOLD.is_file():
        raise FileNotFoundError("Required original Arial fonts are unavailable.")

    original_ai_measurements = validate_original_ai(paths["original_ai"])
    arrays, server_manifest, transitions = load_numeric_state(
        paths["render_state"], paths["render_state_manifest"]
    )
    palette = json.loads(paths["palette"].read_text(encoding="utf-8"))
    for label in (SOURCE_LABEL, *EXPECTED_TOP3):
        if label not in palette:
            raise RuntimeError(f"MOSTA palette lacks required label: {label}")

    stem = "Fig4c_MOSTA_cartilage_lineage_E15p0_to_E15p5_global_t0_50k_latest52D_k10_original_AI_equivalent_v5"
    scatter_layer = output_dir / f"{stem}__scatter_layer_1273x844.png"
    outputs = {extension: output_dir / f"{stem}.{extension}" for extension in ("pdf", "svg", "png")}
    page_centroids, display_transform = create_scatter_layer(
        arrays, transitions, palette, scatter_layer
    )
    assemble_panel(scatter_layer, page_centroids, transitions, palette, outputs)

    transitions_csv = output_dir / f"{stem}__displayed_transitions.csv"
    with transitions_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("rank", "target_label", "count", "probability", "percent", "centroid_x", "centroid_y"),
        )
        writer.writeheader()
        for rank, item in enumerate(transitions[:3], start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "target_label": item.target_label,
                    "count": item.count,
                    "probability": f"{item.probability:.17g}",
                    "percent": f"{100.0 * item.probability:.12g}",
                    "centroid_x": f"{item.centroid_xy[0]:.12g}",
                    "centroid_y": f"{item.centroid_xy[1]:.12g}",
                }
            )

    provenance_dir = output_dir / "provenance"
    provenance_dir.mkdir()
    script_snapshot = provenance_dir / Path(__file__).name
    shutil.copy2(Path(__file__).resolve(), script_snapshot)

    rebuild_command = (
        f"MPLCONFIGDIR=/tmp/mosta_fig4c_mplconfig python {Path(__file__).resolve()} "
        f"--render-state {paths['render_state']} "
        f"--render-state-manifest {paths['render_state_manifest']} "
        f"--palette {paths['palette']} "
        f"--legacy-script {paths['legacy_script']} "
        f"--legacy-reference-png {paths['legacy_reference_png']} "
        f"--original-ai {paths['original_ai']} "
        f"--manuscript-pdf {paths['manuscript_pdf']} "
        "--output-dir <new-immutable-output-directory>"
    )

    manifest = {
        "status": "COMPLETE",
        "figure": "Main text Fig. 4c",
        "selection": "E15.0 to E15.5, latest accepted 52D classifier, spatial vote k=10",
        "scientific_claim": (
            "A Cartilage primordium source population at E15.0 distributes at E15.5 "
            "among Cartilage, Cartilage primordium, and Connective tissue as the three "
            "largest k=10-smoothed predicted identities."
        ),
        "render_only": True,
        "model_loaded": False,
        "simulation_run": False,
        "classification_run": False,
        "historical_percentages_loaded": False,
        "server_numeric_manifest": {
            "path": str(paths["render_state_manifest"]),
            "sha256": sha256(paths["render_state_manifest"]),
            "status": server_manifest["status"],
            "classifier": server_manifest["classifier"],
            "classifier_knn_neighbors": server_manifest["classifier_knn_neighbors"],
            "source_time": server_manifest["source_time"],
            "target_time": server_manifest["target_time"],
            "particle_cap": server_manifest["particle_cap"],
            "source_count": server_manifest["source_count"],
            "trajectory": server_manifest["trajectory"],
        },
        "inputs": {
            label: {"path": str(path), "sha256": sha256(path)}
            for label, path in paths.items()
        },
        "displayed_transitions": [
            {
                "rank": rank,
                "target_label": item.target_label,
                "count": item.count,
                "probability": item.probability,
                "percent": 100.0 * item.probability,
                "centroid_xy": list(item.centroid_xy),
                "centroid_page_pt": list(page_centroids[item.target_label]),
                "ribbon_width_pt": 2.0 + 18.0 * item.probability,
            }
            for rank, item in enumerate(transitions[:3], start=1)
        ],
        "source": {
            "label": SOURCE_LABEL,
            "count": int(len(arrays["selected_source_spatial"])),
            "centroid_xy": display_transform["centroids_data"][SOURCE_LABEL + "__source"],
            "centroid_page_pt": list(page_centroids[SOURCE_LABEL + "__source"]),
        },
        "style_contract": {
            "source": "Figure_mouse1.ai plus historical MOSTA lineage plotting script",
            "panel_rect_pt": list(PANEL_RECT),
            "original_ai_measurements": original_ai_measurements,
            "original_text_baselines_pt": {key: list(value) for key, value in TEXT_BASELINES.items()},
            "original_vector_reference": ORIGINAL_VECTOR_REFERENCE,
            "original_curve_residual_pt": {
                key: list(value) for key, value in ORIGINAL_CURVE_RESIDUAL_PT.items()
            },
            "original_cubic_lane_template": ORIGINAL_CUBIC_LANE_TEMPLATE,
            "original_cubic_points_page": ORIGINAL_CUBIC_POINTS_PAGE,
            "font": "Arial / Arial Bold",
            "dense_spatial_scatter": "single intentionally rasterized layer, matching original AI",
            "vector_layers": ["transition ribbons", "centroid markers", "all text"],
            "background_and_point_style": "historical plotting-script constants",
            "category_palette": str(paths["palette"]),
            "display_transform": display_transform,
            "numeric_centroids_used_as_ribbon_endpoints": True,
            "ribbon_routing": (
                "category-specific cubic lane template measured from original AI; "
                "missing old CT-versus-Cartilage lane separation is restored only "
                "in control points; visual routing only; endpoints remain current centroids"
            ),
            "percentage_labels": (
                "old x anchors and curve-relative vertical offsets; CT retains the "
                "old source-marker-relative offset"
            ),
            "category_labels": (
                "original AI anchors; CP retains the original centroid-relative "
                "vertical offset after its current centroid moves upward"
            ),
            "rotation": False,
            "anisotropic_stretch": False,
            "warp": False,
        },
        "interpretation_limit": (
            "k=10 is a post-hoc spatial majority vote applied to the latest classifier; "
            "it is retained because the user requested a classifier-consistent k=10 panel. "
            "The stage and rank order differ from the historical E14.0 to E14.5 panel and "
            "must be described as E15.0 to E15.5."
        ),
        "outputs": {},
        "rebuild_command": rebuild_command,
    }
    for label, path in {
        **outputs,
        "scatter_layer": scatter_layer,
        "displayed_transitions_csv": transitions_csv,
        "plotting_script_snapshot": script_snapshot,
    }.items():
        manifest["outputs"][label] = {"path": path.name if path.parent == output_dir else str(path.relative_to(output_dir)), "sha256": sha256(path)}

    manifest_path = output_dir / "manifest.json"
    write_json(manifest_path, manifest)

    provenance_md = provenance_dir / "figure-provenance.md"
    provenance_md.write_text(
        "\n".join(
            [
                "# Figure provenance",
                "",
                f"Archived on: `{date.today().isoformat()}`",
                "",
                "Manuscript figure: `Main text Fig. 4c — Predicted transition probability`",
                "",
                "Scientific claim: `At E15.0→E15.5, the three largest k=10-smoothed predicted identities from Cartilage primordium are Cartilage, Cartilage primordium, and Connective tissue.`",
                "",
                "## Files",
                "",
                f"- Vector figure: `{outputs['pdf']}` and `{outputs['svg']}`",
                f"- PNG preview: `{outputs['png']}`",
                f"- Plotting script: `{script_snapshot}`",
                f"- Caption source: `{paths['manuscript_pdf']}`",
                f"- Compiled manuscript: `{paths['manuscript_pdf']}`",
                "",
                "## Selected experiment",
                "",
                f"- Local render input: `{paths['render_state']}`",
                "- Server run: `/data/cytobridge/projects/CytoBridge-ST-1104/runs/mosta-paper-figures/fig4c-k10-E15-to-E15p5-render-export-global-t0-50k-2b3c79e-seed42-20260825-v6`",
                f"- Manifest: `{paths['render_state_manifest']}`",
                "- Classifier: latest accepted 52D cache; k=10 spatial majority vote",
                "- Training: not rerun by this render-only assembly",
                "",
                "## Source paths",
                "",
                f"- Frozen numerical render state: `{paths['render_state']}`",
                f"- Server render manifest: `{paths['render_state_manifest']}`",
                f"- Original Illustrator style source: `{paths['original_ai']}`",
                f"- Historical plotting script: `{paths['legacy_script']}`",
                f"- MOSTA palette: `{paths['palette']}`",
                "",
                "## Panel sources",
                "",
                "| Panel | Content | Source files | Calculation |",
                "|---|---|---|---|",
                f"| c | E15.0→E15.5 cartilage-lineage transition | `{paths['render_state']}`, `{paths['original_ai']}` | global-t0 fixed 50k trajectory; latest 52D classifier; k=10 vote |",
                "",
                "## Evaluation protocol",
                "",
                "- Initial particles: `50,000 global-t0 fixed particles`",
                "- Displayed source particles: `1,282 Cartilage primordium particles at E15.0`",
                "- Identity rule: `latest 52D classifier + k=10 spatial majority vote`",
                "- Time step and diffusion scale: `dt=0.05, sigma=0`",
                "- Seed: `42`",
                "- No restart, rotation, anisotropic stretch, or warp",
                "",
                "## Rebuild command",
                "",
                "```bash",
                rebuild_command,
                "```",
                "",
                "## Interpretation",
                "",
                "`This is an E15.0→E15.5 result. It preserves the requested three biological categories but not the historical E14.0→E14.5 stage or rank order. k=10 is post-hoc spatial smoothing and does not retrain or replace the classifier.`",
                "",
                "## SHA-256",
                "",
                f"- Figure PDF: `{sha256(outputs['pdf'])}`",
                f"- Figure PNG: `{sha256(outputs['png'])}`",
                f"- Plotting script: `{sha256(script_snapshot)}`",
                "",
            ]
        ),
        encoding="utf-8",
    )

    checksums = {
        str(path.relative_to(output_dir)): sha256(path)
        for path in sorted(output_dir.rglob("*"))
        if path.is_file() and path.name not in {"SHA256SUMS.json", "COMPLETE"}
    }
    write_json(output_dir / "SHA256SUMS.json", checksums)
    (output_dir / "COMPLETE").write_text("COMPLETE\n", encoding="utf-8")
    freeze_tree(output_dir)


if __name__ == "__main__":
    main()
