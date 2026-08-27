"""Shared Matplotlib renderer for zebrafish S27--S34."""

from __future__ import annotations

from pathlib import Path
import textwrap
from typing import TYPE_CHECKING

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

if TYPE_CHECKING:
    from .zebrafish_si import ZebrafishSIPanels, ZebrafishSIResults


A4_PORTRAIT = (8.27, 11.69)
S27_GROUPS = ("observed", "generated")
S29_CONDITIONS = ("Baseline", "YSL removal", "EVL removal")
ABLATION_SPECS = (
    ("remove_YSL", "YSL removal", "#0072B2", "o"),
    ("remove_EVL", "EVL removal", "#D55E00", "s"),
)
NOISE_COLORS = {
    0.0: "#5F6368",
    0.01: "#0072B2",
    0.03: "#D55E00",
    0.06: "#CC79A7",
}
LOSS_STYLES = {
    "formal_alpha_control": {
        "label": r"$\alpha_{\mathrm{expr}}=0.015$",
        "color": "#1F4E79",
        "hatch": None,
    },
    "formal": {
        "ratio_label": r"$\lambda_{OT}:\lambda_{mass}=1:1$",
        "color": "#1F4E79",
        "hatch": None,
    },
    "alpha_expr_005": {
        "label": r"$\alpha_{\mathrm{expr}}=0.05$",
        "color": "#D55E00",
        "hatch": "///",
    },
    "ot_mass_10_to_1": {
        "ratio_label": r"$\lambda_{OT}:\lambda_{mass}=10:1$",
        "color": "#E69F00",
        "hatch": "\\\\",
    },
    "ot_mass_1_to_10": {
        "ratio_label": r"$\lambda_{OT}:\lambda_{mass}=1:10$",
        "color": "#009E73",
        "hatch": "..",
    },
}
LOSS_SPACES = (
    ("joint", "Joint state"),
    ("pca", "Expression state"),
    ("spatial", "Physical space"),
)
STEMS = {
    "s27": "zebrafish_s27_observed_generated",
    "s28": "zebrafish_s28_growth_observed",
    "s29": "zebrafish_s29_virtual_removal_morphology",
    "s30": "zebrafish_s30_virtual_removal_quantitative",
    "s31": "zebrafish_s31_gene_dynamics",
    "s32": "zebrafish_s32_loss_weight_sensitivity",
    "s33": "zebrafish_s33_daughter_noise",
    "s34": "zebrafish_s34_inverse_pca_sanity",
}
_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.facecolor": "white",
    "savefig.transparent": False,
}


def _save(
    figure: plt.Figure,
    output: Path,
    figure_id: str,
    *,
    dpi: int = 320,
    tight: bool = False,
) -> tuple[Path, Path]:
    stem = output / STEMS[figure_id]
    pdf = stem.with_suffix(".pdf")
    png = stem.with_suffix(".png")
    kwargs = {"bbox_inches": "tight"} if tight else {}
    figure.savefig(pdf, **kwargs)
    figure.savefig(png, dpi=dpi, **kwargs)
    plt.close(figure)
    return pdf, png


def _scatter_labels(
    axis: plt.Axes,
    xy: np.ndarray,
    labels: np.ndarray,
    colors: dict[str, str],
    *,
    size: float,
    order: list[str] | tuple[str, ...] | None = None,
    alpha: float = 0.82,
) -> None:
    sequence = order if order is not None else sorted(set(labels.astype(str)))
    for label in sequence:
        keep = labels.astype(str) == label
        if np.any(keep):
            axis.scatter(
                xy[keep, 0],
                xy[keep, 1],
                s=size,
                c=colors.get(label, "#BDBDBD"),
                linewidths=0,
                alpha=alpha,
                rasterized=False,
            )
    axis.set_aspect("equal", adjustable="box")
    axis.set_axis_off()


def _limits(arrays: list[np.ndarray], pad: float = 0.04):
    values = np.vstack(arrays)
    lower = values.min(axis=0)
    upper = values.max(axis=0)
    margin = np.maximum((upper - lower) * pad, 1e-6)
    return (
        (float(lower[0] - margin[0]), float(upper[0] + margin[0])),
        (float(lower[1] - margin[1]), float(upper[1] + margin[1])),
    )


def _render_s27(results: "ZebrafishSIResults", output: Path):
    packed = results.observed_generated
    points = packed.xy
    lower = points.min(axis=0)
    upper = points.max(axis=0)
    center = 0.5 * (lower + upper)
    span = float(max(*(upper - lower), 1e-6)) * 1.06
    xlim = (center[0] - span / 2, center[0] + span / 2)
    ylim = (center[1] - span / 2, center[1] + span / 2)
    colors = dict(results.observed_generated_colors)

    figure = plt.figure(figsize=A4_PORTRAIT)
    outer = figure.add_gridspec(
        2,
        2,
        height_ratios=(0.75, 2.35),
        width_ratios=(3.0, 1.05),
        left=0.04,
        right=0.985,
        bottom=0.035,
        top=0.96,
        wspace=0.06,
        hspace=0.12,
    )
    observed_grid = outer[0, 0].subgridspec(1, 5, wspace=0.03)
    generated_grid = outer[1, 0].subgridspec(3, 3, wspace=0.03, hspace=0.06)
    for index, time in enumerate(range(5)):
        axis = figure.add_subplot(observed_grid[0, index])
        xy, labels = packed.frame("observed", float(time))
        _scatter_labels(
            axis, xy, labels, colors, size=2.5, order=list(colors), alpha=0.88
        )
        axis.set_xlim(xlim)
        axis.set_ylim(ylim)
        axis.set_title(f"t = {time:g}", pad=2)
    for index, time in enumerate(np.arange(0, 4.5, 0.5)):
        axis = figure.add_subplot(generated_grid[index // 3, index % 3])
        xy, labels = packed.frame("generated", float(time))
        _scatter_labels(
            axis, xy, labels, colors, size=2.5, order=list(colors), alpha=0.88
        )
        axis.set_xlim(xlim)
        axis.set_ylim(ylim)
        axis.set_title(f"t = {time:g}", pad=2)
        axis.text(0.02, 0.02, f"n = {len(xy):,}", transform=axis.transAxes, fontsize=6.5)
    legend_axis = figure.add_subplot(outer[:, 1])
    legend_axis.set_axis_off()
    legend_axis.legend(
        handles=[
            Line2D(
                [],
                [],
                linestyle="",
                marker="o",
                markersize=4,
                markerfacecolor=color,
                markeredgewidth=0,
                label=label,
            )
            for label, color in colors.items()
        ],
        title="Cell type",
        loc="center left",
        frameon=False,
        fontsize=5.8,
        title_fontsize=8,
        handletextpad=0.4,
        labelspacing=0.34,
    )
    figure.text(0.04, 0.985, "a", fontsize=14, fontweight="bold", va="top")
    figure.text(
        0.075,
        0.985,
        "Observed developmental stages",
        fontsize=12,
        fontweight="bold",
        va="top",
    )
    figure.text(0.04, 0.738, "b", fontsize=14, fontweight="bold", va="top")
    figure.text(
        0.075,
        0.738,
        "Model-generated developmental trajectory",
        fontsize=12,
        fontweight="bold",
        va="top",
    )
    return _save(figure, output, "s27")


def _render_s28(
    results: "ZebrafishSIResults", panels: "ZebrafishSIPanels", output: Path
):
    figure, axes = plt.subplots(3, 2, figsize=A4_PORTRAIT)
    axes = axes.ravel()
    plotted = None
    for index, time in enumerate(range(5)):
        axis = axes[index]
        subset = panels.growth_scaled.loc[
            np.isclose(panels.growth_scaled["time"], float(time))
        ]
        plotted = axis.scatter(
            subset["x"],
            subset["y"],
            c=subset["growth_scaled"],
            cmap="RdYlBu_r",
            vmin=0,
            vmax=1,
            s=2.0,
            linewidths=0,
            rasterized=False,
        )
        axis.set_title(f"t = {time:g}", fontsize=12, fontweight="bold", pad=3)
        axis.set_aspect("equal", adjustable="box")
        axis.set_axis_off()
    axes[-1].axis("off")
    colorbar = figure.colorbar(plotted, ax=axes[:5].tolist(), fraction=0.025, pad=0.025)
    colorbar.solids.set_rasterized(False)
    colorbar.set_label("Growth (within-time 5th–95th percentile scale)")
    figure.suptitle(
        "Growth-rate maps across observed zebrafish stages",
        fontsize=14,
        fontweight="bold",
        y=0.985,
    )
    figure.subplots_adjust(
        left=0.045,
        right=0.88,
        top=0.93,
        bottom=0.035,
        wspace=0.06,
        hspace=0.11,
    )
    return _save(figure, output, "s28")


def _render_s29(results: "ZebrafishSIResults", output: Path):
    packed = results.virtual_removal
    arrays = [xy for _, _, xy, _ in packed.iter_frames()]
    xlim, ylim = _limits(arrays)
    colors = dict(results.celltype_colors)
    figure = plt.figure(figsize=A4_PORTRAIT)
    grid = figure.add_gridspec(
        5, 4, width_ratios=[1, 1, 1, 1.12], wspace=0.02, hspace=0.05
    )
    for column, condition in enumerate(S29_CONDITIONS):
        for row, time in enumerate(range(5)):
            axis = figure.add_subplot(grid[row, column])
            xy, labels = packed.frame(condition, float(time))
            _scatter_labels(axis, xy, labels, colors, size=1.65)
            axis.set_xlim(xlim)
            axis.set_ylim(ylim)
            axis.text(
                0.02,
                0.02,
                f"n = {len(xy):,}",
                transform=axis.transAxes,
                fontsize=7,
                va="bottom",
            )
            if column == 0:
                axis.text(
                    -0.07,
                    0.5,
                    f"t = {time:g}",
                    transform=axis.transAxes,
                    rotation=90,
                    ha="center",
                    va="center",
                    fontsize=9,
                )
            if row == 0:
                axis.set_title(condition, fontsize=12, fontweight="bold")
    legend_axis = figure.add_subplot(grid[:, 3])
    legend_axis.set_axis_off()
    present = sorted(
        set(packed.label_names[packed.label_id].astype(str))
    )
    legend_axis.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markersize=4,
                markerfacecolor=colors.get(label, "#BDBDBD"),
                markeredgecolor="none",
                label="\n".join(textwrap.wrap(label, width=29)),
            )
            for label in present
        ],
        title="Classifier-assigned cell state",
        loc="center left",
        frameon=False,
        fontsize=5.35,
        title_fontsize=7.3,
        handletextpad=0.35,
        labelspacing=0.30,
    )
    figure.text(0.015, 0.985, "a", fontsize=14, fontweight="bold", va="top")
    figure.text(
        0.06,
        0.985,
        "Development after virtual YSL and EVL removal",
        fontsize=12,
        fontweight="bold",
        va="top",
    )
    figure.subplots_adjust(left=0.055, right=0.985, top=0.945, bottom=0.025)
    return _save(figure, output, "s29")


def _clean_axis(axis: plt.Axes, *, grid: bool = True) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    if grid:
        axis.grid(color="#D7DDE2", linewidth=0.45, alpha=0.55, zorder=0)
    axis.set_axisbelow(True)


def _render_s30(
    results: "ZebrafishSIResults", panels: "ZebrafishSIPanels", output: Path
):
    figure = plt.figure(figsize=(8.19, 5.7))
    grid = figure.add_gridspec(
        2,
        6,
        height_ratios=[1.05, 1.0],
        hspace=0.52,
        wspace=0.68,
        left=0.075,
        right=0.985,
        top=0.88,
        bottom=0.105,
    )
    figure.text(0.075, 0.955, "a", fontsize=14, fontweight="bold", va="top")
    figure.text(
        0.105,
        0.955,
        "Endpoint spatial distributions (t = 4)",
        fontsize=12,
        fontweight="bold",
        va="top",
    )
    snapshot_axes = [
        figure.add_subplot(grid[0, 0:2]),
        figure.add_subplot(grid[0, 2:4]),
        figure.add_subplot(grid[0, 4:6]),
    ]
    snapshots = (
        (results.endpoint_baseline_xy, "Baseline", "#59616A"),
        (results.endpoint_ysl_xy, "YSL removal", "#0072B2"),
        (results.endpoint_evl_xy, "EVL removal", "#D55E00"),
    )
    for axis, (xy, title, color) in zip(snapshot_axes, snapshots, strict=True):
        axis.scatter(
            xy[:, 0],
            xy[:, 1],
            s=2.2,
            c=color,
            alpha=0.70,
            linewidths=0,
            rasterized=False,
        )
        axis.set_title(f"{title}\nn = {len(xy):,}", pad=3, linespacing=1.05)
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)
    stacked = np.vstack([item[0] for item in snapshots])
    xmin, ymin = np.quantile(stacked, 0.002, axis=0)
    xmax, ymax = np.quantile(stacked, 0.998, axis=0)
    xpad = 0.035 * max(xmax - xmin, 1e-8)
    ypad = 0.035 * max(ymax - ymin, 1e-8)
    for axis in snapshot_axes:
        axis.set_xlim(float(xmin - xpad), float(xmax + xpad))
        axis.set_ylim(float(ymin - ypad), float(ymax + ypad))
        axis.set_aspect("equal", adjustable="box")

    curve_axis = figure.add_subplot(grid[1, 0:3])
    centroid_axis = figure.add_subplot(grid[1, 3:6])
    curve_axis.text(
        0.0, 1.09, "b", transform=curve_axis.transAxes, fontsize=14,
        fontweight="bold", va="bottom",
    )
    curve_axis.text(
        0.10, 1.09, "Spatial W1", transform=curve_axis.transAxes, fontsize=12,
        fontweight="bold", va="bottom",
    )
    for variant, label, color, marker in ABLATION_SPECS:
        subset = results.ablation_w1_curve.loc[
            results.ablation_w1_curve["variant"].eq(variant)
        ].sort_values("time")
        time = subset["time"].to_numpy(float)
        mean = subset["mean"].to_numpy(float)
        sem = subset["sem"].to_numpy(float)
        curve_axis.fill_between(
            time, mean - sem, mean + sem, color=color, alpha=0.16, linewidth=0
        )
        curve_axis.plot(time, mean, color=color, linewidth=1.8, label=label, zorder=3)
        observed = np.isclose(
            time[:, None], np.arange(5, dtype=float)[None, :], atol=1e-9
        ).any(axis=1)
        curve_axis.plot(
            time[observed], mean[observed], linestyle="none", marker=marker,
            markersize=4, markerfacecolor=color, markeredgecolor="white",
            markeredgewidth=0.45, zorder=4,
        )
    curve_axis.set_xlim(-0.03, 4.03)
    curve_axis.set_ylim(bottom=0)
    curve_axis.set_xticks(np.arange(5, dtype=float))
    curve_axis.set_xlabel("Developmental stage")
    curve_axis.set_ylabel("W1 distance from baseline")
    _clean_axis(curve_axis)
    curve_axis.legend(frameon=False, loc="upper left", handlelength=2.2)

    centroid_axis.text(
        0.0, 1.09, "c", transform=centroid_axis.transAxes, fontsize=14,
        fontweight="bold", va="bottom",
    )
    centroid_axis.text(
        0.10, 1.09, "Endpoint centroid shift (t = 4)",
        transform=centroid_axis.transAxes, fontsize=12, fontweight="bold", va="bottom",
    )
    summaries = panels.ablation_centroid_summary.set_index("variant")
    raw = results.ablation_centroid_by_seed.pivot(
        index="seed", columns="variant", values="centroid_shift"
    ).sort_index()
    ymax = max(
        float(raw.to_numpy(float).max()),
        float(summaries["ci95_high"].max()),
    ) + 0.025
    centroid_axis.set_ylim(0, ymax)
    for x, (variant, label, color, marker) in enumerate(ABLATION_SPECS):
        row = summaries.loc[variant]
        values = raw[variant].to_numpy(float)
        mean = float(row["mean"])
        centroid_axis.bar(x, mean, width=0.54, color=color, alpha=0.62, linewidth=0)
        centroid_axis.errorbar(
            x,
            mean,
            yerr=np.asarray(
                [[mean - float(row["ci95_low"])], [float(row["ci95_high"]) - mean]]
            ),
            color="#24313A",
            linewidth=0.9,
            capsize=3,
            capthick=0.9,
            zorder=4,
        )
        centroid_axis.scatter(
            np.full(values.size, x), values, s=25, marker=marker, facecolor="white",
            edgecolor=color, linewidth=1, zorder=5,
        )
        centroid_axis.text(
            x, 0.5 * mean, f"{mean:.3f}", color="white", fontsize=8.3,
            fontweight="bold", ha="center", va="center",
            bbox={"facecolor": color, "edgecolor": "none", "pad": 0.45},
            zorder=6,
        )
    centroid_axis.set_xlim(-0.55, 1.55)
    centroid_axis.set_xticks([0, 1], ["YSL removal", "EVL removal"])
    centroid_axis.set_ylabel("Centroid shift from baseline")
    _clean_axis(centroid_axis)
    return _save(figure, output, "s30")


def _render_s31(panels: "ZebrafishSIPanels", output: Path):
    matrix = panels.gene_zscores
    split = len(matrix) // 2
    blocks = (matrix.iloc[:split], matrix.iloc[split:])
    figure = plt.figure(figsize=A4_PORTRAIT)
    grid = figure.add_gridspec(
        1,
        3,
        width_ratios=[1.0, 1.0, 0.045],
        left=0.095,
        right=0.94,
        top=0.94,
        bottom=0.08,
        wspace=0.16,
    )
    axes = [figure.add_subplot(grid[0, 0]), figure.add_subplot(grid[0, 1])]
    colorbar_axis = figure.add_subplot(grid[0, 2])
    image = None
    for index, (axis, block) in enumerate(zip(axes, blocks, strict=True)):
        image = axis.pcolormesh(
            np.arange(block.shape[1] + 1) - 0.5,
            np.arange(block.shape[0] + 1) - 0.5,
            block.to_numpy(float),
            shading="flat",
            cmap="RdBu_r",
            vmin=-2,
            vmax=2,
            rasterized=False,
        )
        axis.set_xticks(
            np.arange(block.shape[1]),
            [f"{value:g}" for value in block.columns.astype(float)],
            rotation=45,
            ha="right",
        )
        rows = np.arange(len(block))
        axis.set_yticks(rows, block.index.astype(str), fontsize=3.2)
        axis.set_ylim(len(block) - 0.5, -0.5)
        axis.set_xlabel("Time")
        if index == 0:
            axis.set_ylabel("Genes")
        else:
            axis.yaxis.tick_right()
            axis.tick_params(axis="y", labelright=True, labelleft=False, pad=1.5)
    colorbar = figure.colorbar(image, cax=colorbar_axis)
    colorbar.solids.set_rasterized(False)
    colorbar.set_label("Within-gene temporal z score")
    figure.suptitle("YSL-lineage gene dynamics", fontsize=14, fontweight="bold", y=0.985)
    return _save(figure, output, "s31")


def _loss_legend(conditions: list[str], key: str):
    return [
        Patch(
            facecolor=LOSS_STYLES[name]["color"],
            edgecolor="black",
            linewidth=0.5,
            hatch=LOSS_STYLES[name]["hatch"],
            label=LOSS_STYLES[name][key],
        )
        for name in conditions
    ]


def _render_s32(results: "ZebrafishSIResults", output: Path):
    frame = results.loss_weight_metrics
    figure, axes = plt.subplots(2, 3, figsize=(11.69, 8.27), sharex=True)
    figure.subplots_adjust(
        left=0.075, right=0.985, bottom=0.09, top=0.88, hspace=0.48, wspace=0.30
    )
    row_specs = (
        (["formal_alpha_control", "alpha_expr_005"], "label"),
        (["ot_mass_10_to_1", "formal", "ot_mass_1_to_10"], "ratio_label"),
    )
    for column, (space, title) in enumerate(LOSS_SPACES):
        ymax = float(frame.loc[frame["space"].eq(space), "w1"].max()) * 1.15
        for row, (conditions, _) in enumerate(row_specs):
            axis = axes[row, column]
            x = np.arange(4, dtype=float)
            width = 0.72 / len(conditions)
            for index, condition in enumerate(conditions):
                data = frame.loc[
                    frame["condition"].eq(condition) & frame["space"].eq(space)
                ].sort_values("time")
                offset = (index - (len(conditions) - 1) / 2) * width
                style = LOSS_STYLES[condition]
                axis.bar(
                    x + offset,
                    data["w1"],
                    width=width,
                    color=style["color"],
                    edgecolor="black",
                    linewidth=0.5,
                    hatch=style["hatch"],
                    zorder=3,
                )
            axis.set_ylim(0, ymax)
            axis.set_title(title, pad=5, fontsize=10, fontweight="bold")
            axis.set_xticks(x, ["1", "2", "3", "4"])
            if row == 1:
                axis.set_xlabel("Model time")
            if column == 0:
                axis.set_ylabel("W1 (lower is better)")
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            axis.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.75, zorder=0)
    figure.text(0.016, 0.965, "a", fontsize=14, fontweight="bold", va="top")
    figure.text(
        0.048, 0.965, r"Sensitivity to $\alpha_{\mathrm{expr}}$",
        fontsize=12, fontweight="bold", va="top",
    )
    figure.legend(
        handles=_loss_legend(*row_specs[0]), loc="upper right",
        bbox_to_anchor=(0.985, 0.972), ncol=2, frameon=False,
    )
    figure.text(0.016, 0.515, "b", fontsize=14, fontweight="bold", va="top")
    figure.text(
        0.048, 0.515, r"Sensitivity to $\lambda_{OT}:\lambda_{mass}$",
        fontsize=12, fontweight="bold", va="top",
    )
    figure.legend(
        handles=_loss_legend(*row_specs[1]), loc="center right",
        bbox_to_anchor=(0.985, 0.516), ncol=3, frameon=False,
    )
    return _save(figure, output, "s32", dpi=480)


def _wrapped(value: str, width: int) -> str:
    return "\n".join(textwrap.wrap(str(value), width=width, break_long_words=False))


def _panel_label(axis: plt.Axes, label: str, *, x: float, y: float) -> None:
    axis.text(
        x, y, label, transform=axis.transAxes, fontsize=14, fontweight="bold",
        ha="left", va="top", clip_on=False,
    )


def _render_s33(
    results: "ZebrafishSIResults", panels: "ZebrafishSIPanels", output: Path
):
    composition = results.daughter_composition
    summary = panels.daughter_composition_summary
    observed = results.daughter_observed_composition
    lineage_values = results.daughter_lineage_values
    lineage_summary = panels.daughter_lineage_summary
    selected = results.daughter_lineage_pairs
    sensitivity = results.daughter_sensitivity
    particles = results.daughter_particle_counts
    noises = sorted(composition["daughter_noise_std"].unique())
    times = sorted(composition["time"].unique())
    observed_times = np.sort(observed["time"].unique().astype(float))

    figure = plt.figure(figsize=(7.35, 8.15), facecolor="white")
    outer = figure.add_gridspec(
        2, 3, height_ratios=[1.42, 1.0], width_ratios=[1.22, 1.0, 1.0],
        left=0.085, right=0.985, bottom=0.09, top=0.865,
        hspace=0.50, wspace=0.56,
    )
    top_grid = outer[0, :].subgridspec(2, 3, hspace=0.54, wspace=0.36)
    top_axes = [
        figure.add_subplot(top_grid[row, column])
        for row in range(2)
        for column in range(3)
    ]
    for axis, celltype in zip(top_axes, panels.daughter_top_celltypes, strict=True):
        observed_line = (
            observed.loc[observed["celltype"].eq(celltype)]
            .set_index("time")["fraction"]
            .reindex(observed_times, fill_value=0.0)
        )
        axis.plot(
            observed_times, observed_line.to_numpy(float), color="black",
            linestyle=":", marker="s", markersize=3, linewidth=1.15, zorder=5,
        )
        plot_order = [value for value in noises if not np.isclose(value, 0.0)] + [0.0]
        for noise in plot_order:
            subset = summary.loc[
                np.isclose(summary["daughter_noise_std"], noise)
                & summary["celltype"].eq(celltype)
            ].set_index("time").reindex(times).fillna(0.0)
            mean = subset["mean"].to_numpy(float)
            sem = subset["sem"].to_numpy(float)
            color = NOISE_COLORS[float(noise)]
            reference = np.isclose(noise, 0.0)
            axis.plot(
                times, mean, color=color, linewidth=1.5 if reference else 1.15,
                linestyle="--" if reference else "-", marker="o",
                markersize=3.2 if reference else 2.6,
                markerfacecolor="white" if reference else color,
                markeredgecolor=color, markeredgewidth=0.8 if reference else 0,
                zorder=4 if reference else 3,
            )
            axis.fill_between(
                times, mean - sem, mean + sem, color=color, alpha=0.12, linewidth=0
            )
        axis.set_title(_wrapped(celltype, 28), pad=4)
        axis.set_xlim(min(times), max(times))
        axis.set_xticks(times)
        axis.set_ylim(bottom=0)
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#E8E8E8", linewidth=0.6)
    for axis in top_axes[3:]:
        axis.set_xlabel("Developmental stage")
    top_axes[0].set_ylabel("Cell fraction")
    top_axes[3].set_ylabel("Cell fraction")
    figure.text(0.035, 0.914, "a", fontsize=14, fontweight="bold", va="center")
    figure.text(
        0.085, 0.914, "Cell-type composition trajectories",
        fontsize=12, fontweight="bold", va="center",
    )
    handles = [
        Line2D([], [], color="black", linestyle=":", marker="s", markersize=4, label="Observed")
    ] + [
        Line2D(
            [], [], color=NOISE_COLORS[float(noise)], marker="o",
            linewidth=1.5 if np.isclose(noise, 0.0) else 1.15,
            linestyle="--" if np.isclose(noise, 0.0) else "-",
            markerfacecolor="white" if np.isclose(noise, 0.0) else NOISE_COLORS[float(noise)],
            markeredgewidth=0.8 if np.isclose(noise, 0.0) else 0,
            markersize=4, label=rf"$\sigma_{{noise}}$={noise:g}",
        )
        for noise in noises
    ]
    figure.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.57, 0.985),
        ncol=len(handles), frameon=False, handlelength=2, columnspacing=1,
    )

    lineage_axis = figure.add_subplot(outer[1, 0])
    ordered_pairs = list(
        zip(selected["source_celltype"].astype(str), selected["target_celltype"].astype(str), strict=True)
    )[::-1]
    y_base = np.arange(len(ordered_pairs), dtype=float)
    offsets = np.linspace(-0.24, 0.24, len(noises))
    for offset, noise in zip(offsets, noises, strict=True):
        color = NOISE_COLORS[float(noise)]
        for y, (source, target) in zip(y_base, ordered_pairs, strict=True):
            raw = lineage_values.loc[
                np.isclose(lineage_values["daughter_noise_std"], noise)
                & lineage_values["source_celltype"].eq(source)
                & lineage_values["target_celltype"].eq(target)
            ]
            row = lineage_summary.loc[
                np.isclose(lineage_summary["daughter_noise_std"], noise)
                & lineage_summary["source_celltype"].eq(source)
                & lineage_summary["target_celltype"].eq(target)
            ].iloc[0]
            lineage_axis.scatter(
                raw["fraction"], np.full(len(raw), y + offset), s=9,
                color=color, alpha=0.28, linewidths=0, zorder=2,
            )
            lineage_axis.errorbar(
                float(row["mean"]), y + offset, xerr=float(row["sem"]), fmt="o",
                markersize=3.4, color=color, capsize=1.5, linewidth=0.9, zorder=3,
            )
    lineage_axis.set_yticks(y_base)
    lineage_axis.set_yticklabels(
        [_wrapped(f"{source} → {target}", 31) for source, target in ordered_pairs],
        fontsize=6.8,
    )
    lineage_axis.set_xlabel("Descendant fraction within t0 lineage")
    lineage_axis.set_title(
        "Major model-inferred\nt0→t4 fates", fontsize=12,
        fontweight="bold", loc="left", pad=8,
    )
    lineage_axis.spines[["top", "right"]].set_visible(False)
    lineage_axis.grid(axis="x", color="#E8E8E8", linewidth=0.6)
    _panel_label(lineage_axis, "b", x=-0.32, y=1.18)

    sensitivity_axis = figure.add_subplot(outer[1, 1])
    final = sensitivity.loc[np.isclose(sensitivity["time"], 4.0)]
    metrics = (
        ("composition_tv_from_reference", "Cell-type composition", "o", "#009E73"),
        ("lineage_weighted_tv_from_reference", "Lineage transitions", "s", "#6A3D9A"),
    )
    for column, label, marker, color in metrics:
        means, sems = [], []
        for noise in noises:
            values = 100 * final.loc[
                np.isclose(final["daughter_noise_std"], noise), column
            ]
            means.append(float(values.mean()))
            sems.append(float(values.sem(ddof=1)))
            sensitivity_axis.scatter(
                np.full(len(values), noise) + np.linspace(-0.0014, 0.0014, len(values)),
                values, s=11, color=color, alpha=0.28, linewidths=0,
            )
        sensitivity_axis.errorbar(
            noises, means, yerr=sems, color=color, marker=marker,
            markersize=4, linewidth=1.25, capsize=2, label=label, zorder=4,
        )
    sensitivity_axis.set_xticks(noises)
    sensitivity_axis.set_xticklabels(["0", "0.01", "0.03", "0.06"])
    sensitivity_axis.set_xlabel(r"Daughter perturbation $\sigma_{noise}$")
    sensitivity_axis.set_ylabel(r"Distribution change from $\sigma_{noise}=0$ (%)")
    sensitivity_axis.set_title(
        "Distribution changes", fontsize=12, fontweight="bold", loc="left", pad=8
    )
    sensitivity_axis.legend(frameon=False, loc="upper left")
    sensitivity_axis.spines[["top", "right"]].set_visible(False)
    sensitivity_axis.grid(axis="y", color="#E8E8E8", linewidth=0.6)
    _panel_label(sensitivity_axis, "c", x=-0.28, y=1.18)

    particle_axis = figure.add_subplot(outer[1, 2])
    initial_n = float(
        particles.loc[np.isclose(particles["time"], min(times)), "n_particles"].iloc[0]
    )
    for noise in [value for value in noises if not np.isclose(value, 0.0)] + [0.0]:
        subset = particles.loc[np.isclose(particles["daughter_noise_std"], noise)].copy()
        subset["fold"] = subset["n_particles"] / initial_n
        reference = np.isclose(noise, 0.0)
        for _, seed_values in subset.groupby("seed"):
            particle_axis.plot(
                seed_values["time"], seed_values["fold"], color=NOISE_COLORS[float(noise)],
                alpha=0.13, linewidth=0.8 if reference else 0.7,
                linestyle="--" if reference else "-",
            )
        grouped = subset.groupby("time")["fold"].agg(["mean", "sem"]).reindex(times)
        particle_axis.errorbar(
            times, grouped["mean"], yerr=grouped["sem"], color=NOISE_COLORS[float(noise)],
            marker="o", markersize=3, linewidth=1.5 if reference else 1.05,
            linestyle="--" if reference else "-",
            markerfacecolor="white" if reference else NOISE_COLORS[float(noise)],
            markeredgewidth=0.8 if reference else 0, capsize=1.5,
            zorder=4 if reference else 3,
        )
    observed_counts = observed.groupby("time")["n_cells"].first().sort_index()
    particle_axis.plot(
        observed_counts.index.to_numpy(float),
        observed_counts.to_numpy(float) / float(observed_counts.iloc[0]),
        color="black", linestyle=":", marker="s", markersize=3.2, linewidth=1.15,
    )
    particle_axis.set_xlabel("Developmental stage")
    particle_axis.set_ylabel("Population size / t0")
    particle_axis.set_xticks(times)
    particle_axis.set_title(
        "Population growth", fontsize=12, fontweight="bold", loc="left", pad=8
    )
    particle_axis.spines[["top", "right"]].set_visible(False)
    particle_axis.grid(axis="y", color="#E8E8E8", linewidth=0.6)
    _panel_label(particle_axis, "d", x=-0.28, y=1.18)
    return _save(figure, output, "s33", tight=True)


def _render_s34(
    results: "ZebrafishSIResults", panels: "ZebrafishSIPanels", output: Path
):
    observed = results.observed_expression
    reconstructed = results.reconstructed_expression
    top_mask = observed.index.astype(str).isin(results.top_variable_genes)
    metrics = panels.inverse_pca_metrics.set_index("time")
    figure, axes = plt.subplots(3, 2, figsize=A4_PORTRAIT)
    axes = axes.ravel()
    for index, time in enumerate(observed.columns.astype(float)):
        axis = axes[index]
        x = observed[time].to_numpy(float)
        y = reconstructed[time].to_numpy(float)
        lower = min(float(x.min()), float(y.min()))
        upper = max(float(x.max()), float(y.max()))
        pad = max(0.03 * (upper - lower), 1e-3)
        low, high = lower - pad, upper + pad
        axis.scatter(
            x[~top_mask], y[~top_mask], s=5, c="#7AA6C2", alpha=0.30,
            linewidths=0, rasterized=False,
        )
        axis.scatter(
            x[top_mask], y[top_mask], s=7, c="#D55E00", alpha=0.62,
            linewidths=0, rasterized=False,
        )
        axis.plot([low, high], [low, high], color="#333333", linewidth=1, linestyle="--")
        row = metrics.loc[time]
        axis.set_title(
            f"t = {time:g}   r = {row['pearson_r']:.2f}, RMSE = {row['rmse']:.2f}",
            fontsize=12,
            fontweight="bold",
        )
        axis.set_xlim(low, high)
        axis.set_ylim(low, high)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("Exact observed mean log1p")
        axis.set_ylabel("Inverse-PCA mean log1p")
        axis.spines[["top", "right"]].set_visible(False)
    axes[-1].axis("off")
    axes[-1].legend(
        handles=[
            Line2D(
                [0], [0], marker="o", linestyle="", markersize=5,
                markerfacecolor="#7AA6C2", markeredgecolor="none", alpha=0.6,
                label="All active genes",
            ),
            Line2D(
                [0], [0], marker="o", linestyle="", markersize=5,
                markerfacecolor="#D55E00", markeredgecolor="none", alpha=0.8,
                label="Top 250 temporal genes",
            ),
        ],
        loc="center",
        frameon=False,
    )
    figure.suptitle(
        "Observed expression versus inverse-PCA reconstruction",
        fontsize=14,
        fontweight="bold",
        y=0.985,
    )
    figure.subplots_adjust(
        left=0.09, right=0.97, top=0.93, bottom=0.06, wspace=0.28, hspace=0.28
    )
    return _save(figure, output, "s34")


def render_zebrafish_si(
    results: "ZebrafishSIResults",
    panels: "ZebrafishSIPanels",
    output_dir: str | Path,
    figures: tuple[str, ...],
) -> dict[str, tuple[Path, Path]]:
    """Render selected figure pairs with one local style context."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    renderers = {
        "s27": lambda: _render_s27(results, output),
        "s28": lambda: _render_s28(results, panels, output),
        "s29": lambda: _render_s29(results, output),
        "s30": lambda: _render_s30(results, panels, output),
        "s31": lambda: _render_s31(panels, output),
        "s32": lambda: _render_s32(results, output),
        "s33": lambda: _render_s33(results, panels, output),
        "s34": lambda: _render_s34(results, panels, output),
    }
    with mpl.rc_context(_RC):
        return {figure_id: renderers[figure_id]() for figure_id in figures}
