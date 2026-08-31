from __future__ import annotations

"""Draw Supplementary Figures S7 and S8 from the archived numerical results."""

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
RELEASE_DIR = SCRIPT_DIR.parent
RESULTS_DIR = RELEASE_DIR / "data"
FIGURE_DIR = RELEASE_DIR / "figures"
STYLE_DIR = SCRIPT_DIR

from figure_style import (
    A4_PORTRAIT,
    GRID_COLOR,
    HEADING_COLOR,
    PLOT_TEXT_SIZE,
    apply_style,
    clean_axis,
    panel_heading,
    save_figure,
)


TIME_ORDER = ("D4", "D7", "D10", "D14")
TIME_COLORS = {
    "D4": "#4C78A8",
    "D7": "#4CB5AE",
    "D10": "#E99572",
    "D14": "#F2C65D",
}
ORIGINAL_INPUT_COLOR = "#B7BDC5"
PERTURBED_INPUT_COLOR = "#F28E2B"
TRANSLATION_COLOR = "#2A9D8F"
ROTATION_COLOR = "#E07A5F"
DISPLAY_VARIANTS = (
    ("baseline_repeat", "Unperturbed"),
    ("translate_low", "Translation ≤0.36×"),
    ("translate_moderate", "Translation ≤0.71×"),
    ("rotate_low", "Rotation ≤3.1°"),
    ("rotate_moderate", "Rotation ≤6.2°"),
    ("translate_rotate_low", "Combined\n≤0.36×, ≤3.1°"),
    ("translate_rotate_moderate", "Combined\n≤0.71×, ≤6.2°"),
)


def configure_style() -> None:
    """Apply the CytoBridge manuscript style with the local Arial files."""

    for path in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial Italic.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold Italic.ttf",
    ):
        font_path = Path(path)
        if font_path.exists():
            mpl.font_manager.fontManager.addfont(font_path)
    mpl.style.use(STYLE_DIR / "cytobridge-paper.mplstyle")
    apply_style()
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "svg.fonttype": "none",
            "axes.unicode_minus": False,
        }
    )


def load_plot_inputs() -> dict[str, np.ndarray]:
    bundle_path = RESULTS_DIR / "plot_inputs.npz"
    with np.load(bundle_path, allow_pickle=False) as bundle:
        arrays = {key: bundle[key] for key in bundle.files}
    n = len(arrays["timepoint"])
    for key, value in arrays.items():
        if key.endswith("_xy") and value.shape != (n, 2):
            raise ValueError(f"Unexpected coordinate shape for {key}: {value.shape}")
    return arrays


def scatter_stages(
    ax: mpl.axes.Axes,
    xy: np.ndarray,
    timepoint: np.ndarray,
    *,
    size: float,
    alpha: float = 0.80,
) -> None:
    for stage in TIME_ORDER:
        mask = timepoint == stage
        ax.scatter(
            xy[mask, 0],
            xy[mask, 1],
            s=size,
            color=TIME_COLORS[stage],
            edgecolors="none",
            alpha=alpha,
            rasterized=False,
        )
    spatial_axis(ax)


def spatial_axis(ax: mpl.axes.Axes) -> None:
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def time_legend_handles(marker_size: float = 4.8) -> list[Line2D]:
    return [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markersize=marker_size,
            markerfacecolor=TIME_COLORS[stage],
            markeredgewidth=0,
        )
        for stage in TIME_ORDER
    ]


def value_matrix(table: pd.DataFrame, value: str) -> np.ndarray:
    variant_order = [variant for variant, _ in DISPLAY_VARIANTS]
    matrix = (
        table.pivot(index="variant", columns="timepoint", values=value)
        .reindex(index=variant_order, columns=TIME_ORDER)
        .to_numpy(dtype=float)
    )
    if matrix.shape != (len(DISPLAY_VARIANTS), len(TIME_ORDER)):
        raise ValueError(f"Unexpected matrix shape for {value}: {matrix.shape}")
    return matrix


def annotation_color(cmap, norm: Normalize, value: float) -> str:
    red, green, blue, _ = cmap(norm(value))
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return "white" if luminance < 0.48 else "#24313A"


def heatmap(
    ax: mpl.axes.Axes,
    table: pd.DataFrame,
    value: str,
    title: str,
    *,
    vmin: float,
    vmax: float,
    percent: bool = False,
    show_row_labels: bool = True,
) -> None:
    matrix = value_matrix(table, value)
    cmap = plt.get_cmap("viridis")
    norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
    image = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")
    ax.set_title(title, fontsize=PLOT_TEXT_SIZE, pad=4.5)
    ax.set_xticks(np.arange(len(TIME_ORDER)), TIME_ORDER)
    ax.set_yticks(np.arange(len(DISPLAY_VARIANTS)))
    if show_row_labels:
        ax.set_yticklabels([label for _, label in DISPLAY_VARIANTS])
    else:
        ax.set_yticklabels([])
    ax.tick_params(length=0, pad=2)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            cell_value = matrix[row, column]
            if not np.isfinite(cell_value):
                label = "NA"
                color = "white"
            elif percent:
                label = f"{100 * cell_value:.1f}"
                color = annotation_color(cmap, norm, cell_value)
            else:
                label = f"{cell_value:.2f}"
                if label == "1.00" and cell_value < 1.0:
                    label = "1.00*"
                color = annotation_color(cmap, norm, cell_value)
            ax.text(
                column,
                row,
                label,
                ha="center",
                va="center",
                fontsize=7.5,
                color=color,
            )
    for spine in ax.spines.values():
        spine.set_visible(False)
    ticks = [vmin, (vmin + vmax) / 2, vmax]
    colorbar = ax.figure.colorbar(
        image,
        ax=ax,
        fraction=0.042,
        pad=0.025,
        ticks=ticks,
    )
    if percent:
        colorbar.ax.set_yticklabels([f"{100 * tick:.1f}" for tick in ticks])
        colorbar.set_label("%", rotation=0, labelpad=6, fontsize=8)
    colorbar.ax.tick_params(labelsize=7.5, length=2, pad=1.5)
    colorbar.outline.set_linewidth(0.45)


def perturbation_sizes() -> tuple[dict[str, float], dict[str, float]]:
    records: list[dict] = []
    for filename in ("lower_input_manifest.json", "input_manifest.json"):
        document = json.loads(
            (RESULTS_DIR.parent / "manifests" / filename).read_text(encoding="utf-8")
        )
        for variant in document["variants"]:
            stages = variant.get("stages", variant.get("stage_records", []))
            for stage in stages:
                records.append({"variant": variant["variant"], **stage})
    frame = pd.DataFrame(records)
    frame["translation"] = np.hypot(frame["translate_x_nn"], frame["translate_y_nn"])
    frame["rotation"] = frame["rotation_deg"].abs()
    translation = frame.groupby("variant")["translation"].max().to_dict()
    rotation = frame.groupby("variant")["rotation"].max().to_dict()
    return translation, rotation


def perturbation_bars(
    ax: mpl.axes.Axes,
    values: list[float],
    *,
    categories: list[str],
    ylabel: str,
    title: str,
    digits: int,
    color: str,
) -> None:
    positions = np.arange(len(categories), dtype=float)
    bars = ax.bar(
        positions,
        values,
        width=0.66,
        color=color,
        edgecolor="white",
        linewidth=0.4,
        zorder=2,
    )
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.{digits}f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#24313A",
        )
    ax.set_title(title, pad=4)
    ax.set_ylabel(ylabel)
    ax.set_xticks(positions, categories)
    upper = max(values)
    ax.set_ylim(0, upper * 1.24)
    clean_axis(ax, grid=False)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.45, alpha=0.65, zorder=0)


def plot_s7(arrays: dict[str, np.ndarray]) -> tuple[Path, Path]:
    coordinates = pd.read_csv(RESULTS_DIR / "coordinate_metrics.csv")
    velocity = pd.read_csv(RESULTS_DIR / "velocity_metrics_pooled.csv")
    interactions = pd.read_csv(RESULTS_DIR / "interaction_metrics.csv")
    translation, rotation = perturbation_sizes()

    fig = plt.figure(figsize=A4_PORTRAIT, constrained_layout=False)
    outer = fig.add_gridspec(
        7,
        1,
        height_ratios=(0.23, 1.66, 0.23, 4.12, 0.17, 0.23, 1.46),
        left=0.205,
        right=0.965,
        top=0.972,
        bottom=0.058,
        hspace=0.18,
    )

    heading_a = fig.add_subplot(outer[0])
    panel_heading(heading_a, "a", "Chicken-heart sections", title_x=0.05)
    section_grid = outer[1].subgridspec(1, 2, wspace=0.12)
    ax_input = fig.add_subplot(section_grid[0, 0])
    ax_aligned = fig.add_subplot(section_grid[0, 1])
    scatter_stages(
        ax_input,
        arrays["source_input_xy"],
        arrays["timepoint"],
        size=3.3,
    )
    scatter_stages(
        ax_aligned,
        arrays["accepted_aligned_xy"],
        arrays["timepoint"],
        size=3.3,
    )
    ax_input.set_title("Input sections", pad=3)
    ax_aligned.set_title("Aligned sections", pad=3)
    fig.legend(
        time_legend_handles(),
        TIME_ORDER,
        ncol=4,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.59, 0.817),
        columnspacing=0.9,
        handletextpad=0.25,
        borderaxespad=0,
    )

    heading_b = fig.add_subplot(outer[2])
    panel_heading(
        heading_b,
        "b",
        "Alignment and downstream sensitivity",
        title_x=0.05,
    )
    heat_grid = outer[3].subgridspec(2, 2, hspace=0.28, wspace=0.30)
    heatmap(
        fig.add_subplot(heat_grid[0, 0]),
        coordinates,
        "rigid_adjusted_rmsd_fraction_of_baseline_radius",
        "Aligned-coordinate residual",
        vmin=0.0,
        vmax=0.05,
        percent=True,
        show_row_labels=True,
    )
    heatmap(
        fig.add_subplot(heat_grid[0, 1]),
        velocity[velocity["velocity_key"] == "full"],
        "median_cosine_rigid_adjusted",
        "Full-velocity cosine",
        vmin=0.0,
        vmax=1.0,
        show_row_labels=False,
    )
    heatmap(
        fig.add_subplot(heat_grid[1, 0]),
        velocity[velocity["velocity_key"] == "interaction"],
        "median_cosine_rigid_adjusted",
        "Interaction-velocity cosine",
        vmin=0.0,
        vmax=1.0,
        show_row_labels=True,
    )
    heatmap(
        fig.add_subplot(heat_grid[1, 1]),
        interactions,
        "attention_weighted_jaccard_union",
        "Interaction-weight overlap",
        vmin=0.0,
        vmax=1.0,
        show_row_labels=False,
    )

    rounding_note = fig.add_subplot(outer[4])
    rounding_note.axis("off")
    rounding_note.text(
        0.0,
        0.50,
        "* Values are rounded to two decimal places; 1.00 may denote a value below one.",
        ha="left",
        va="center",
        fontsize=PLOT_TEXT_SIZE,
        color="#59616A",
    )

    heading_c = fig.add_subplot(outer[5])
    panel_heading(heading_c, "c", "Perturbation sizes", title_x=0.05)
    size_grid = outer[6].subgridspec(1, 2, wspace=0.34)
    ax_translation = fig.add_subplot(size_grid[0, 0])
    ax_rotation = fig.add_subplot(size_grid[0, 1])
    perturbation_bars(
        ax_translation,
        [
            translation["translate_low"],
            translation["translate_moderate"],
            translation["translate_rotate_low"],
            translation["translate_rotate_moderate"],
        ],
        categories=["Translation\nlower", "Translation\nhigher", "Combined\nlower", "Combined\nhigher"],
        ylabel="Maximum shift\n(median-NN units)",
        title="Translation",
        digits=2,
        color=TRANSLATION_COLOR,
    )
    perturbation_bars(
        ax_rotation,
        [
            rotation["rotate_low"],
            rotation["rotate_moderate"],
            rotation["translate_rotate_low"],
            rotation["translate_rotate_moderate"],
        ],
        categories=["Rotation\nlower", "Rotation\nhigher", "Combined\nlower", "Combined\nhigher"],
        ylabel="Maximum rotation (°)",
        title="Rotation",
        digits=1,
        color=ROTATION_COLOR,
    )

    pdf = FIGURE_DIR / "heart_alignment_sensitivity_S7_final.pdf"
    png = FIGURE_DIR / "heart_alignment_sensitivity_S7_final.png"
    save_figure(fig, pdf, png, dpi=320)
    plt.close(fig)
    return pdf, png


def bounds(arrays: list[np.ndarray], pad_fraction: float = 0.045) -> tuple[float, ...]:
    stacked = np.vstack(arrays)
    xmin, ymin = stacked.min(axis=0)
    xmax, ymax = stacked.max(axis=0)
    dx = max(xmax - xmin, 1e-12)
    dy = max(ymax - ymin, 1e-12)
    return (
        xmin - pad_fraction * dx,
        xmax + pad_fraction * dx,
        ymin - pad_fraction * dy,
        ymax + pad_fraction * dy,
    )


def set_bounds(ax: mpl.axes.Axes, limits: tuple[float, ...]) -> None:
    xmin, xmax, ymin, ymax = limits
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)


def plot_s8(arrays: dict[str, np.ndarray]) -> tuple[Path, Path]:
    source_xy = arrays["source_input_xy"]
    timepoint = arrays["timepoint"]
    stage_bounds: dict[str, tuple[float, ...]] = {}
    for stage in TIME_ORDER:
        stage_mask = timepoint == stage
        stage_arrays = [source_xy[stage_mask]]
        for variant, _ in DISPLAY_VARIANTS[1:]:
            stage_arrays.append(arrays[f"{variant}__input_xy"][stage_mask])
        stage_bounds[stage] = bounds(stage_arrays)
    aligned_bounds = bounds(
        [arrays[f"{variant}__aligned_xy"] for variant, _ in DISPLAY_VARIANTS]
    )

    fig = plt.figure(figsize=A4_PORTRAIT, constrained_layout=False)
    grid = fig.add_gridspec(
        9,
        6,
        width_ratios=(1.33, 0.96, 0.96, 0.96, 0.96, 1.58),
        height_ratios=(0.30, 0.23, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        left=0.035,
        right=0.975,
        top=0.975,
        bottom=0.075,
        wspace=0.06,
        hspace=0.09,
    )

    heading_input = fig.add_subplot(grid[0, :5])
    panel_heading(heading_input, "a", "Input coordinate perturbations", title_x=0.065)
    heading_aligned = fig.add_subplot(grid[0, 5])
    panel_heading(
        heading_aligned,
        "b",
        "Aligned sections",
        label_x=0.0,
        title_x=0.16,
    )

    for column, stage in enumerate(TIME_ORDER, start=1):
        title_axis = fig.add_subplot(grid[1, column])
        title_axis.axis("off")
        title_axis.text(
            0.5,
            0.52,
            stage,
            ha="center",
            va="center",
            fontsize=PLOT_TEXT_SIZE,
            color=HEADING_COLOR,
        )
    aligned_title_axis = fig.add_subplot(grid[1, 5])
    aligned_title_axis.axis("off")
    aligned_title_axis.text(
        0.5,
        0.52,
        "All stages",
        ha="center",
        va="center",
        fontsize=PLOT_TEXT_SIZE,
        color=HEADING_COLOR,
    )

    for row, (variant, row_label) in enumerate(DISPLAY_VARIANTS, start=2):
        label_axis = fig.add_subplot(grid[row, 0])
        label_axis.axis("off")
        label_axis.text(
            0.98,
            0.5,
            row_label,
            ha="right",
            va="center",
            fontsize=PLOT_TEXT_SIZE,
            color="#24313A",
            linespacing=1.05,
        )
        input_xy = arrays[f"{variant}__input_xy"]
        for column, stage in enumerate(TIME_ORDER, start=1):
            ax = fig.add_subplot(grid[row, column])
            stage_mask = timepoint == stage
            ax.scatter(
                source_xy[stage_mask, 0],
                source_xy[stage_mask, 1],
                s=1.45,
                color=ORIGINAL_INPUT_COLOR,
                edgecolors="none",
                alpha=0.48,
                rasterized=False,
            )
            if variant != "baseline_repeat":
                ax.scatter(
                    input_xy[stage_mask, 0],
                    input_xy[stage_mask, 1],
                    s=1.45,
                    color=PERTURBED_INPUT_COLOR,
                    edgecolors="none",
                    alpha=0.66,
                    rasterized=False,
                )
            set_bounds(ax, stage_bounds[stage])
            spatial_axis(ax)

        aligned_axis = fig.add_subplot(grid[row, 5])
        scatter_stages(
            aligned_axis,
            arrays[f"{variant}__aligned_xy"],
            timepoint,
            size=1.55,
            alpha=0.78,
        )
        set_bounds(aligned_axis, aligned_bounds)

    input_handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markersize=4.6,
            markerfacecolor=ORIGINAL_INPUT_COLOR,
            markeredgewidth=0,
        ),
        Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markersize=4.6,
            markerfacecolor=PERTURBED_INPUT_COLOR,
            markeredgewidth=0,
        ),
    ]
    fig.legend(
        input_handles,
        ("Original", "Perturbed"),
        ncol=2,
        frameon=False,
        loc="lower left",
        bbox_to_anchor=(0.22, 0.022),
        columnspacing=1.0,
        handletextpad=0.35,
    )
    fig.legend(
        time_legend_handles(marker_size=4.6),
        TIME_ORDER,
        ncol=4,
        frameon=False,
        loc="lower right",
        bbox_to_anchor=(0.975, 0.022),
        columnspacing=0.7,
        handletextpad=0.25,
    )

    pdf = FIGURE_DIR / "heart_alignment_sensitivity_S8_final.pdf"
    png = FIGURE_DIR / "heart_alignment_sensitivity_S8_final.png"
    save_figure(fig, pdf, png, dpi=320)
    plt.close(fig)
    return pdf, png


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Directory containing plot_inputs.npz and the archived metric CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=FIGURE_DIR,
        help="Directory for the PDF and PNG files.",
    )
    return parser.parse_args()


def main() -> None:
    global RESULTS_DIR, FIGURE_DIR
    args = parse_args()
    RESULTS_DIR = args.data_dir.expanduser().resolve()
    FIGURE_DIR = args.output_dir.expanduser().resolve()
    configure_style()
    arrays = load_plot_inputs()
    paths = (*plot_s7(arrays), *plot_s8(arrays))
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
