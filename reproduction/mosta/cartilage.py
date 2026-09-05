"""Cartilage-lineage drawing functions from the original Figure 4c source."""
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

ORIGINAL_SCATTER_XREF = 151

ORIGINAL_SCATTER_SIZE = (1273, 844)

ORIGINAL_SCATTER_BBOX = (
    316.0863952636719,
    237.58531188964844,
    583.919189453125,
    415.15863037109375,
)

PANEL_RECT = (316.0863952636719, 202.0, 595.0, 440.0)

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

ORIGINAL_CURVE_RESIDUAL_PT = {
    "Cartilage primordium": (2.20851643880206, 19.104284922281883),
    "Connective tissue": (-2.817316691080743, 15.59332784016928),
    "Cartilage": (0.3396759033203125, 23.121673583984375),
}

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

@dataclass(frozen=True)
class Transition:
    target_label: str
    count: int
    probability: float
    centroid_xy: tuple[float, float]

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
    regular = FontProperties(family="Arial")
    bold = FontProperties(family="Arial", weight="bold")

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
