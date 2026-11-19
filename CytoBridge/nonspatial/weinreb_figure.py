"""Build the accepted Weinreb A4 figure from a verified panel-data bundle.

No pre-rendered PNG or SVG panel is used.  SPRING is display-only, and every
model field is a direct PC1--PC2 projection of the fitted CytoBridge model; it
is not RNA velocity. Bundle selection and SHA-256 validation are owned by
``CytoBridge.nonspatial.figures``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from .weinreb_figure_style import (
    A4_PORTRAIT,
    ABLATION_COLOR,
    CYTOBRIDGE_COLOR,
    GRID_COLOR,
    GT_COLOR,
    HEADING_COLOR,
    TEXT_COLOR,
    apply_style,
    clean_axis,
    panel_heading,
    save_figure,
)


HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent
PANEL_DATA = BUNDLE / "panel_data"
METRICS = BUNDLE / "metrics"

FIGURE_STEM = "weinreb_nonspatial_interaction_a4"

TEAL = CYTOBRIDGE_COLOR
CORAL = ABLATION_COLOR
PURPLE = "#6F5AA8"
ROSE = "#B64E6C"
LIGHT_TEAL = "#87C9CC"
LIGHT_GRAY = "#E7EBEE"
MID_GRAY = "#8A939A"
DARK_GRAY = "#111111"

CELL_COLORS = {
    "Undifferentiated": "#7A7A7A",
    "Monocyte": "#2878B5",
    "Neutrophil": "#E45756",
    "Baso": "#8E6C8A",
    "Erythroid": "#C44E52",
    "Meg": "#CCB974",
    "Mast": "#9C755F",
    "Lymphoid": "#4C9F70",
    "Eos": "#F28E2B",
    "Ccr7_DC": "#59A14F",
    "pDC": "#76B7B2",
}

CELL_ORDER = [
    "Undifferentiated",
    "Monocyte",
    "Neutrophil",
    "Baso",
    "Mast",
    "Meg",
    "Erythroid",
    "Lymphoid",
    "Eos",
    "Ccr7_DC",
    "pDC",
]

COPIED_NAMES = {
    "model_grids": "model_field_grids.npz",
    "distribution": "distribution_per_repeat.csv",
    "distribution_composite": "distribution_w1_w2_composite_per_repeat.csv",
    "clone_seed": "clone_fate_per_training_seed.csv",
    "clone_ensemble": "clone_fate_ensemble.csv",
    "cellchat_joined": "cellchat_joined_directed_edges.csv",
    "association": "cellchat_association_metrics.csv",
    "seed_stability": "interaction_training_seed_rank_stability.csv",
    "cytobridge_pathways": "cytobridge_pathway_scores.csv",
}


def require_panel_data() -> None:
    expected = [
        "observed_cells.csv.gz",
        "source_manifest.json",
        *COPIED_NAMES.values(),
    ]
    missing = [name for name in expected if not (PANEL_DATA / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "The validated archived panel-data bundle is incomplete. Missing: "
            + ", ".join(missing)
        )


def panel_container(fig: plt.Figure, spec, label: str, title: str, *, body_rows=1):
    inner = spec.subgridspec(
        2,
        1,
        height_ratios=[0.15, body_rows],
        hspace=0.025,
    )
    heading = fig.add_subplot(inner[0])
    panel_heading(heading, label, title, title_x=0.062 if len(title) > 25 else 0.07)
    return inner[1], heading


def draw_rounded_box(
    ax,
    xy,
    width,
    height,
    text_value,
    *,
    facecolor="white",
    edgecolor=GRID_COLOR,
    textcolor=TEXT_COLOR,
    linewidth=0.8,
    fontsize=8.2,
    weight="normal",
):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text_value,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=textcolor,
        fontweight=weight,
        linespacing=1.15,
    )
    return patch


def draw_vertical_arrow(ax, x, y0, y1, color=GT_COLOR):
    ax.add_patch(
        FancyArrowPatch(
            (x, y0),
            (x, y1),
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=0.8,
            color=color,
        )
    )


def draw_design_panel(ax, cells: pd.DataFrame) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    draw_rounded_box(
        ax,
        (0.08, 0.72),
        0.84,
        0.17,
        "Time-resolved scRNA-seq\nDays 2, 4 and 6 · clone barcodes",
        facecolor="#F6F8FA",
        edgecolor="#BEC7CE",
        fontsize=8.3,
        weight="bold",
    )
    draw_vertical_arrow(ax, 0.50, 0.72, 0.63)
    draw_rounded_box(
        ax,
        (0.13, 0.45),
        0.74,
        0.16,
        "Expression state\n2,000 HVGs  →  50 PCs",
        facecolor="white",
        edgecolor="#9FAAB2",
        fontsize=8.3,
    )
    draw_vertical_arrow(ax, 0.50, 0.45, 0.36, color=TEAL)
    draw_rounded_box(
        ax,
        (0.09, 0.16),
        0.82,
        0.18,
        "LR-informed CytoBridge GNN–SDE\n193 LR pairs · 52 pathways",
        facecolor="#E7F3F4",
        edgecolor=TEAL,
        textcolor=HEADING_COLOR,
        linewidth=1.1,
        fontsize=8.4,
        weight="bold",
    )


def spring_limits(cells: pd.DataFrame):
    xlo, xhi = cells["spring_x"].quantile([0.002, 0.998])
    ylo, yhi = cells["spring_y"].quantile([0.002, 0.998])
    xpad = 0.035 * (xhi - xlo)
    ypad = 0.035 * (yhi - ylo)
    return (xlo - xpad, xhi + xpad), (ylo - ypad, yhi + ypad)


def draw_spring_panel(fig, spec, cells: pd.DataFrame) -> None:
    inner = spec.subgridspec(
        4,
        1,
        height_ratios=[0.29, 1.0, 0.10, 0.22],
        hspace=0.035,
    )
    head = fig.add_subplot(inner[0])
    panel_heading(
        head,
        "a",
        "Observed cell states across time",
        title_x=0.078,
        y=0.74,
    )
    maps = inner[1].subgridspec(1, 3, wspace=0.035)
    xlim, ylim = spring_limits(cells)
    rng = np.random.default_rng(20260810)
    map_axes = []
    for column, day in enumerate([2, 4, 6]):
        ax = fig.add_subplot(maps[0, column])
        map_axes.append(ax)
        subset = cells[cells["day"] == day]
        # Plot common populations first and rare populations last so rare cells remain visible.
        counts = subset["cell_type"].value_counts()
        order = counts.sort_values(ascending=False).index.tolist()
        for cell_type in order:
            points = subset[subset["cell_type"] == cell_type]
            index = np.arange(len(points))
            rng.shuffle(index)
            points = points.iloc[index]
            ax.scatter(
                points["spring_x"],
                points["spring_y"],
                s=2.0,
                c=CELL_COLORS[cell_type],
                alpha=0.70,
                linewidths=0,
                rasterized=False,
            )
        ax.set_title(
            f"Day {day}\nn = {len(subset):,}", fontsize=8.5, pad=0.0, linespacing=0.92
        )
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        if column == 0:
            ax.set_ylabel("SPRING 2", fontsize=7.8, labelpad=2.0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(True)
        ax.spines["left"].set_visible(True)
        ax.spines["bottom"].set_color("#5A6268")
        ax.spines["left"].set_color("#5A6268")
        ax.spines["bottom"].set_linewidth(0.65)
        ax.spines["left"].set_linewidth(0.65)
    coordinate_ax = fig.add_subplot(inner[2])
    coordinate_ax.axis("off")
    coordinate_ax.text(
        0.5,
        0.52,
        "SPRING 1",
        ha="center",
        va="center",
        fontsize=7.8,
        color=HEADING_COLOR,
    )
    legend_ax = fig.add_subplot(inner[3])
    legend_ax.axis("off")
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markersize=4.2,
            color=CELL_COLORS[name],
            label=name,
        )
        for name in CELL_ORDER
    ]
    legend_ax.legend(
        handles=handles,
        ncol=6,
        loc="center",
        frameon=False,
        fontsize=7.6,
        columnspacing=0.9,
        handletextpad=0.35,
        borderaxespad=0,
    )


def deterministic_background(subset: pd.DataFrame, n: int = 2600) -> pd.DataFrame:
    if len(subset) <= n:
        return subset
    return subset.sample(n=n, random_state=20260810)


def draw_stream_field(
    ax, grids, cells, *, day: int, field: str, color: str, speed_limit: float
):
    subset = deterministic_background(cells[cells["day"] == day])
    counts = subset["cell_type"].value_counts()
    for cell_type in counts.sort_values(ascending=False).index:
        points = subset[subset["cell_type"] == cell_type]
        ax.scatter(
            points["pc1"],
            points["pc2"],
            s=1.45,
            color=CELL_COLORS[cell_type],
            alpha=0.24,
            linewidths=0,
            rasterized=False,
            zorder=0,
        )
    prefix = f"day{day}"
    x = grids[f"{prefix}_x_axis"]
    y = grids[f"{prefix}_y_axis"]
    mask = grids[f"{prefix}_support_mask"].astype(bool)
    u = np.ma.masked_where(~mask, grids[f"{prefix}_{field}_u"])
    v = np.ma.masked_where(~mask, grids[f"{prefix}_{field}_v"])
    speed = np.ma.masked_where(~mask, grids[f"{prefix}_{field}_speed"])
    linewidth = 0.32 + 0.80 * np.clip(speed.filled(0) / speed_limit, 0, 1)
    ax.streamplot(
        x,
        y,
        u,
        v,
        color=color,
        density=0.72,
        linewidth=linewidth,
        arrowsize=0.55,
        minlength=0.08,
        maxlength=2.7,
        zorder=2,
    )
    ax.set_xlim(float(x.min()), float(x.max()))
    ax.set_ylim(float(y.min()), float(y.max()))
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(labelsize=7, pad=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)


def draw_model_field_panel(fig, spec, cells: pd.DataFrame, grids) -> None:
    body, heading = panel_container(
        fig, spec, "b", "Interaction-informed dynamics in PCA state space"
    )
    axes_spec = body.subgridspec(1, 4, wspace=0.16)
    full_speeds = []
    for day in [2, 4, 6]:
        mask = grids[f"day{day}_support_mask"].astype(bool)
        full_speeds.append(grids[f"day{day}_lr_full_drift_speed"][mask])
    full_limit = float(np.quantile(np.concatenate(full_speeds), 0.92))
    interaction_mask = grids["day6_support_mask"].astype(bool)
    interaction_limit = float(
        np.quantile(grids["day6_lr_interaction_speed"][interaction_mask], 0.92)
    )
    specifications = [
        (2, "lr_full_drift", TEAL, "Day 2 · full drift"),
        (4, "lr_full_drift", TEAL, "Day 4 · full drift"),
        (6, "lr_full_drift", TEAL, "Day 6 · full drift"),
        (6, "lr_interaction", ROSE, "Day 6 · interaction component"),
    ]
    for index, (day, field, color, title) in enumerate(specifications):
        ax = fig.add_subplot(axes_spec[0, index])
        draw_stream_field(
            ax,
            grids,
            cells,
            day=day,
            field=field,
            color=color,
            speed_limit=full_limit if field == "lr_full_drift" else interaction_limit,
        )
        ax.set_title(title, fontsize=8.5, pad=2)
        ax.set_xlabel("PC1", fontsize=8)
        if index == 0:
            ax.set_ylabel("PC2", fontsize=8)
        else:
            ax.set_yticklabels([])


def distribution_effects(distribution: pd.DataFrame) -> pd.DataFrame:
    subset = distribution[
        distribution["condition"].isin(["lr_predictor_noise", "no_interaction_noise"])
        & distribution["metric"].isin(["w1", "w2"])
        & (distribution["space"] == "pca")
    ].copy()
    pivot = subset.pivot_table(
        index=["training_seed", "inference_seed", "day", "metric"],
        columns="condition",
        values="value",
        aggfunc="first",
    ).reset_index()
    required = {"lr_predictor_noise", "no_interaction_noise"}
    if not required.issubset(pivot.columns) or len(pivot) != 3 * 50 * 4:
        raise ValueError("Primary W1/W2 paired inference table is incomplete.")
    pivot["relative_reduction"] = (
        pivot["no_interaction_noise"] - pivot["lr_predictor_noise"]
    ) / pivot["no_interaction_noise"]
    effects = pivot.groupby(["training_seed", "day", "metric"], as_index=False).agg(
        mean_relative_reduction=("relative_reduction", "mean"),
        inference_repeat_sd=("relative_reduction", "std"),
        n_inference_repeats=("inference_seed", "nunique"),
    )
    effects["relative_reduction_pct"] = 100 * effects["mean_relative_reduction"]
    return effects


def draw_distribution_ablation(
    ax, effects: pd.DataFrame, composite: pd.DataFrame
) -> float:
    endpoint_order = [
        ("Day 4", "w1"),
        ("Day 4", "w2"),
        ("Day 6", "w1"),
        ("Day 6", "w2"),
    ]
    labels = ["D4 · W1", "D4 · W2", "D6 · W1", "D6 · W2"]
    y = np.arange(len(endpoint_order))[::-1]
    seed_markers = {42: "o", 43: "s", 44: "^"}
    offsets = {42: -0.10, 43: 0.0, 44: 0.10}
    for yi, endpoint in zip(y, endpoint_order):
        block = effects[
            (effects["day"] == endpoint[0]) & (effects["metric"] == endpoint[1])
        ]
        values = []
        for row in block.itertuples(index=False):
            values.append(row.relative_reduction_pct)
            ax.scatter(
                row.relative_reduction_pct,
                yi + offsets[int(row.training_seed)],
                s=20,
                marker=seed_markers[int(row.training_seed)],
                facecolor="white",
                edgecolor=MID_GRAY,
                linewidth=0.8,
                zorder=3,
            )
        mean_value = float(np.mean(values))
        ax.scatter(
            mean_value,
            yi,
            s=37,
            marker="D",
            color=TEAL,
            edgecolor="white",
            linewidth=0.5,
            zorder=4,
        )
    ax.axvspan(0, 6.4, color=TEAL, alpha=0.045, zorder=-3)
    ax.axvline(0, color=GT_COLOR, lw=0.8)
    ax.set_yticks(y, labels=labels)
    ax.set_xlim(-0.45, 6.35)
    ax.set_xlabel(
        "Relative W1/W2 reduction (%)\npositive = with interaction better", labelpad=2
    )
    clean_axis(ax, grid=True)
    ax.grid(axis="y", visible=False)
    composite_mean = float(composite["mean_relative_improvement"].mean()) * 100
    ax.text(
        0.98,
        0.96,
        f"Equal-weight composite: {composite_mean:.2f}% lower",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.2,
        color=HEADING_COLOR,
        fontweight="bold",
        zorder=10,
        bbox={
            "boxstyle": "round,pad=0.18",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 1.0,
        },
    )
    handles = [
        Line2D(
            [],
            [],
            marker=marker,
            linestyle="",
            markerfacecolor="white",
            markeredgecolor=MID_GRAY,
            label=f"seed {seed}",
        )
        for seed, marker in seed_markers.items()
    ] + [Line2D([], [], marker="D", linestyle="", color=TEAL, label="seed mean")]
    ax.legend(
        handles=handles,
        loc="lower right",
        ncol=2,
        frameon=False,
        fontsize=7.2,
        handletextpad=0.3,
        columnspacing=0.7,
    )
    return composite_mean


def clone_effects(seed_table: pd.DataFrame, ensemble: pd.DataFrame) -> pd.DataFrame:
    metrics = {
        "TV agreement": "clone_macro_tv_agreement",
        "JS similarity": "clone_macro_js_similarity",
        "Dominant fate": "clone_macro_dominant_fate_match",
    }
    rows = []
    for seed in sorted(seed_table["training_seed"].unique()):
        block = seed_table[seed_table["training_seed"] == seed].set_index("condition")
        for label, column in metrics.items():
            rows.append(
                {
                    "estimate": f"seed {int(seed)}",
                    "training_seed": int(seed),
                    "metric": label,
                    "with_interaction": float(
                        block.loc["full_interaction_noise", column]
                    ),
                    "without_interaction": float(
                        block.loc["no_interaction_noise", column]
                    ),
                    "agreement_gain_pp": 100
                    * float(
                        block.loc["full_interaction_noise", column]
                        - block.loc["no_interaction_noise", column]
                    ),
                }
            )
    block = ensemble.set_index("condition")
    for label, column in metrics.items():
        rows.append(
            {
                "estimate": "3-seed ensemble",
                "training_seed": np.nan,
                "metric": label,
                "with_interaction": float(block.loc["full_interaction_noise", column]),
                "without_interaction": float(block.loc["no_interaction_noise", column]),
                "agreement_gain_pp": 100
                * float(
                    block.loc["full_interaction_noise", column]
                    - block.loc["no_interaction_noise", column]
                ),
            }
        )
    return pd.DataFrame(rows)


def draw_clone_ablation(ax, effects: pd.DataFrame, ensemble: pd.DataFrame) -> None:
    metrics = ["TV agreement", "JS similarity", "Dominant fate"]
    y = np.arange(len(metrics))[::-1]
    seed_markers = {42: "o", 43: "s", 44: "^"}
    offsets = {42: -0.10, 43: 0.0, 44: 0.10}
    for yi, metric in zip(y, metrics):
        block = effects[
            (effects["metric"] == metric) & effects["training_seed"].notna()
        ]
        for row in block.itertuples(index=False):
            seed = int(row.training_seed)
            ax.scatter(
                row.agreement_gain_pp,
                yi + offsets[seed],
                s=20,
                marker=seed_markers[seed],
                facecolor="white",
                edgecolor=MID_GRAY,
                linewidth=0.8,
                zorder=3,
            )
        ensemble_row = effects[
            (effects["metric"] == metric) & effects["training_seed"].isna()
        ].iloc[0]
        ax.scatter(
            ensemble_row["agreement_gain_pp"],
            yi,
            s=39,
            marker="D",
            color=TEAL,
            edgecolor="white",
            linewidth=0.5,
            zorder=4,
        )
    ax.axvspan(0, 7.7, color=TEAL, alpha=0.045, zorder=-3)
    ax.axvline(0, color=GT_COLOR, lw=0.8)
    ax.set_yticks(y, labels=metrics)
    ax.set_xlim(-2.0, 7.7)
    ax.set_xlabel(
        "Agreement gain (percentage points)\npositive = with interaction better",
        labelpad=2,
    )
    clean_axis(ax, grid=True)
    ax.grid(axis="y", visible=False)
    handles = [
        Line2D(
            [],
            [],
            marker=marker,
            linestyle="",
            markerfacecolor="white",
            markeredgecolor=MID_GRAY,
            label=f"seed {seed}",
        )
        for seed, marker in seed_markers.items()
    ] + [Line2D([], [], marker="D", linestyle="", color=TEAL, label="3-seed ensemble")]
    ax.legend(
        handles=handles,
        loc="lower right",
        ncol=2,
        frameon=False,
        fontsize=7.2,
        handletextpad=0.3,
        columnspacing=0.7,
    )


def draw_distribution_dumbbells(
    ax,
    distribution: pd.DataFrame,
    composite: pd.DataFrame,
) -> float:
    """Direct Full-versus-No-interaction comparison in the original W units."""

    subset = distribution[
        distribution["condition"].isin(["lr_predictor_noise", "no_interaction_noise"])
        & distribution["metric"].isin(["w1", "w2"])
        & (distribution["space"] == "pca")
    ].copy()
    seed_means = subset.groupby(
        ["training_seed", "condition", "day", "metric"], as_index=False
    ).value.mean()
    endpoint_order = [
        ("Day 4", "w1"),
        ("Day 4", "w2"),
        ("Day 6", "w1"),
        ("Day 6", "w2"),
    ]
    labels = ["D4 · W1", "D4 · W2", "D6 · W1", "D6 · W2"]
    y = np.arange(len(endpoint_order))[::-1]
    offsets = {42: -0.11, 43: 0.0, 44: 0.11}

    for yi, endpoint in zip(y, endpoint_order):
        block = seed_means[
            (seed_means["day"] == endpoint[0]) & (seed_means["metric"] == endpoint[1])
        ]
        pivot = block.pivot(index="training_seed", columns="condition", values="value")
        for seed, row in pivot.iterrows():
            yy = yi + offsets[int(seed)]
            full = float(row["lr_predictor_noise"])
            no_interaction = float(row["no_interaction_noise"])
            ax.plot([full, no_interaction], [yy, yy], color="#B8C0C6", lw=0.7, zorder=1)
            ax.scatter(
                full,
                yy,
                s=14,
                marker="o",
                color=TEAL,
                alpha=0.48,
                linewidth=0,
                zorder=2,
            )
            ax.scatter(
                no_interaction,
                yy,
                s=14,
                marker="s",
                facecolor="white",
                edgecolor=CORAL,
                alpha=0.58,
                linewidth=0.7,
                zorder=2,
            )

        full_mean = float(pivot["lr_predictor_noise"].mean())
        no_mean = float(pivot["no_interaction_noise"].mean())
        arrow = FancyArrowPatch(
            (no_mean, yi),
            (full_mean, yi),
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=1.7,
            color=TEAL,
            zorder=4,
        )
        ax.add_patch(arrow)
        ax.scatter(
            no_mean,
            yi,
            s=42,
            marker="s",
            color=CORAL,
            edgecolor="white",
            linewidth=0.6,
            zorder=5,
        )
        ax.scatter(
            full_mean,
            yi,
            s=48,
            marker="o",
            color=TEAL,
            edgecolor="white",
            linewidth=0.6,
            zorder=6,
        )

    composite_mean = float(composite["mean_relative_improvement"].mean()) * 100
    ax.set_yticks(y, labels=labels)
    ax.set_xlim(7.02, 8.20)
    ax.set_xlabel("Weighted PCA distance  (lower is better)", labelpad=3)
    clean_axis(ax, grid=True)
    ax.grid(axis="y", visible=False)
    return composite_mean


def draw_clone_metric_cards(fig: plt.Figure, spec, ensemble: pd.DataFrame) -> None:
    """Show the ensemble comparison as four large, directly labeled cards."""

    block = ensemble.set_index("condition")
    metrics = [
        ("TV agreement", "clone_macro_tv_agreement"),
        ("JS similarity", "clone_macro_js_similarity"),
        ("Dominant-fate match", "clone_macro_dominant_fate_match"),
        ("Probability Pearson", "clone_fate_pair_pearson"),
    ]
    grid = spec.subgridspec(2, 2, hspace=0.12, wspace=0.08)
    for index, (label, column) in enumerate(metrics):
        ax = fig.add_subplot(grid[index // 2, index % 2])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.add_patch(
            FancyBboxPatch(
                (0.015, 0.035),
                0.97,
                0.91,
                boxstyle="round,pad=0.018,rounding_size=0.035",
                facecolor="#F7FAFB",
                edgecolor="#D8E0E5",
                linewidth=0.7,
            )
        )
        full = float(block.loc["full_interaction_noise", column])
        no_interaction = float(block.loc["no_interaction_noise", column])
        gain_pp = 100 * (full - no_interaction)
        ax.text(
            0.06,
            0.86,
            label,
            ha="left",
            va="top",
            fontsize=8.3,
            color=HEADING_COLOR,
            fontweight="bold",
        )
        ax.text(
            0.20,
            0.62,
            "No interaction",
            ha="center",
            va="center",
            fontsize=6.9,
            color=DARK_GRAY,
        )
        ax.text(
            0.20,
            0.38,
            f"{no_interaction:.3f}",
            ha="center",
            va="center",
            fontsize=12.2,
            color=CORAL,
            fontweight="bold",
        )
        ax.add_patch(
            FancyArrowPatch(
                (0.40, 0.43),
                (0.63, 0.43),
                arrowstyle="-|>",
                mutation_scale=10,
                linewidth=1.4,
                color=TEAL,
            )
        )
        ax.text(
            0.79, 0.62, "Full", ha="center", va="center", fontsize=6.9, color=DARK_GRAY
        )
        ax.text(
            0.79,
            0.38,
            f"{full:.3f}",
            ha="center",
            va="center",
            fontsize=12.2,
            color=TEAL,
            fontweight="bold",
        )
        ax.text(
            0.50,
            0.12,
            f"+{gain_pp:.2f} percentage points",
            ha="center",
            va="center",
            fontsize=7.0,
            color=TEAL,
            fontweight="bold",
        )


def draw_clone_probability_dumbbells(
    fig: plt.Figure,
    spec,
    ensemble: pd.DataFrame,
) -> None:
    metrics = [
        ("TV agreement", "clone_macro_tv_agreement", (0.445, 0.492)),
        ("JS similarity", "clone_macro_js_similarity", (0.545, 0.578)),
    ]
    axes_spec = spec.subgridspec(1, 2, wspace=0.24)
    block = ensemble.set_index("condition")
    for column, (label, metric, xlim) in enumerate(metrics):
        ax = fig.add_subplot(axes_spec[0, column])
        full = float(block.loc["full_interaction_noise", metric])
        no_interaction = float(block.loc["no_interaction_noise", metric])
        yy = 0.5
        ax.add_patch(
            FancyArrowPatch(
                (no_interaction, yy),
                (full, yy),
                arrowstyle="-|>",
                mutation_scale=10,
                linewidth=1.9,
                color=TEAL,
                zorder=3,
            )
        )
        ax.scatter(
            no_interaction,
            yy,
            s=48,
            marker="s",
            color=CORAL,
            edgecolor="white",
            linewidth=0.6,
            zorder=4,
        )
        ax.scatter(
            full,
            yy,
            s=54,
            marker="o",
            color=TEAL,
            edgecolor="white",
            linewidth=0.6,
            zorder=5,
        )
        ax.set_xlim(*xlim)
        ax.set_ylim(0.0, 1.0)
        ax.set_yticks([yy])
        if column == 0:
            ax.set_yticklabels(["Ensemble"])
        else:
            ax.set_yticklabels([])
        ax.set_xlabel(f"{label}  ↑", labelpad=2, fontweight="bold")
        clean_axis(ax, grid=True)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="x", labelsize=7.2)


def day6_network_edges(joined: pd.DataFrame, *, top_n: int = 30) -> pd.DataFrame:
    day6 = joined[(joined["time"] == 6) & joined["heterotypic"].astype(bool)].copy()
    rows = []
    specifications = [
        ("CytoBridge D", "message_D_AB_raw"),
        ("CellChat", "cellchat_native_raw"),
    ]
    for method, score in specifications:
        selected = day6[day6[score] > 0].nlargest(top_n, score).copy()
        selected["method"] = method
        selected["display_score"] = selected[score]
        selected["display_rank"] = np.arange(1, len(selected) + 1)
        selected["display_top_n"] = top_n
        rows.append(
            selected[
                [
                    "method",
                    "sender_type",
                    "receiver_type",
                    "display_score",
                    "display_rank",
                    "display_top_n",
                ]
            ]
        )
    return pd.concat(rows, ignore_index=True)


def circle_positions(types: list[str]) -> dict[str, np.ndarray]:
    angles = np.linspace(np.pi / 2, np.pi / 2 - 2 * np.pi, len(types), endpoint=False)
    return {
        name: 0.82 * np.array([np.cos(angle), np.sin(angle)])
        for name, angle in zip(types, angles)
    }


def draw_circle_network(
    ax, edges: pd.DataFrame, joined: pd.DataFrame, *, method: str, title: str
) -> None:
    node_order = [
        "Undifferentiated",
        "Neutrophil",
        "Monocyte",
        "Meg",
        "Mast",
        "Lymphoid",
        "Erythroid",
        "Eos",
        "Ccr7_DC",
        "Baso",
        "pDC",
    ]
    positions = circle_positions(node_order)
    selected = edges[edges["method"] == method].sort_values(
        "display_rank", ascending=False
    )
    max_rank = max(float(selected["display_rank"].max()), 1.0)
    for row in selected.itertuples(index=False):
        source = positions[row.sender_type]
        target = positions[row.receiver_type]
        vector = target - source
        length = float(np.linalg.norm(vector))
        if length <= 0:
            continue
        unit = vector / length
        start = source + 0.075 * unit
        end = target - 0.075 * unit
        strength = 1.0 - (float(row.display_rank) - 1.0) / max(max_rank - 1.0, 1.0)
        curvature = (
            0.075
            if node_order.index(row.sender_type) < node_order.index(row.receiver_type)
            else -0.075
        )
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=5.5 + 2.8 * strength,
                connectionstyle=f"arc3,rad={curvature}",
                linewidth=0.35 + 1.55 * strength,
                color=CELL_COLORS[row.sender_type],
                alpha=0.20 + 0.58 * strength,
                zorder=1,
            )
        )

    counts = (
        joined[joined["time"] == 6][["sender_type", "n_sender_cells"]]
        .drop_duplicates("sender_type")
        .set_index("sender_type")["n_sender_cells"]
    )
    log_counts = np.log1p(counts.reindex(node_order).to_numpy(dtype=float))
    scaled = (log_counts - log_counts.min()) / max(
        float(np.ptp(log_counts)), np.finfo(float).eps
    )
    sizes = 85 + 120 * scaled
    for name, size in zip(node_order, sizes):
        point = positions[name]
        ax.scatter(
            point[0],
            point[1],
            s=float(size),
            color=CELL_COLORS[name],
            edgecolor="white",
            linewidth=0.9,
            zorder=4,
        )
        label_point = 1.16 * point
        horizontal = (
            "left"
            if label_point[0] > 0.08
            else "right"
            if label_point[0] < -0.08
            else "center"
        )
        vertical = (
            "bottom"
            if label_point[1] > 0.15
            else "top"
            if label_point[1] < -0.15
            else "center"
        )
        display_name = "Undiff." if name == "Undifferentiated" else name
        ax.text(
            label_point[0],
            label_point[1],
            display_name,
            ha=horizontal,
            va=vertical,
            fontsize=6.5,
            color=TEXT_COLOR,
        )
    ax.set_xlim(-1.20, 1.20)
    ax.set_ylim(-1.20, 1.20)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=8.8, pad=1.5, color=HEADING_COLOR, fontweight="bold")


def summarize_concordance(
    association: pd.DataFrame,
    stability: pd.DataFrame,
    network_edges: pd.DataFrame,
) -> dict:
    raw = association[
        (association["method"] == "D_exact_message")
        & (association["universe"] == "all_edges")
        & (association["control_set"] == "unadjusted")
    ].sort_values("time")
    if len(raw) != 3:
        raise ValueError("Expected three raw all-edge D--CellChat association rows.")
    stable = stability[
        (stability["method"] == "D_exact_message")
        & (stability["universe"] == "all_edges")
    ]
    stable_min = float(stable["spearman_rho"].min())
    stable_max = float(stable["spearman_rho"].max())
    mean_rho = float(raw["spearman_rho"].mean())
    mean_tau = float(raw["kendall_tau_b_unadjusted"].mean())
    display_counts = network_edges.groupby("method").size().to_dict()

    day6 = raw[raw["time"] == 6].iloc[0]
    return {
        "displayed_day": 6,
        "displayed_spearman_rho": float(day6["spearman_rho"]),
        "displayed_kendall_tau_b": float(day6["kendall_tau_b_unadjusted"]),
        "displayed_n_contexts": int(day6["n_edges"]),
        "mean_spearman_rho_equal_time": mean_rho,
        "mean_kendall_tau_b_equal_time": mean_tau,
        "display_edges_per_method": {
            str(key): int(value) for key, value in display_counts.items()
        },
        "seed_stability_min": stable_min,
        "seed_stability_max": stable_max,
        "per_day": [
            {
                "day": int(row.time),
                "spearman_rho": float(row.spearman_rho),
                "kendall_tau_b": float(row.kendall_tau_b_unadjusted),
                "n_contexts": int(row.n_edges),
            }
            for row in raw.itertuples(index=False)
        ],
    }


def select_day6_cytobridge_pathways(
    pathways: pd.DataFrame, *, top_n: int = 8
) -> pd.DataFrame:
    """Aggregate Day-6 CytoBridge LR-annotated interaction score by pathway."""

    required = {
        "time",
        "sender_type",
        "receiver_type",
        "pathway",
        "S_AB_pathway",
    }
    missing = required - set(pathways.columns)
    if missing:
        raise ValueError(f"CytoBridge pathway table lacks columns: {sorted(missing)}")
    day6 = pathways[
        (pathways["time"] == 6)
        & (pathways["sender_type"] != pathways["receiver_type"])
        & (pathways["S_AB_pathway"] > 0)
    ].copy()
    total_score = float(day6["S_AB_pathway"].sum())
    if not np.isfinite(total_score) or total_score <= 0:
        raise ValueError("Day-6 CytoBridge pathway score has no positive mass.")
    selected = (
        day6.groupby("pathway", as_index=False)
        .agg(
            summed_cytobridge_score=("S_AB_pathway", "sum"),
            n_positive_cell_type_pairs=("S_AB_pathway", "size"),
        )
        .sort_values(
            ["summed_cytobridge_score", "pathway"],
            ascending=[False, True],
        )
        .head(top_n)
        .copy()
    )
    if selected.empty:
        raise ValueError("No positive Day-6 CytoBridge pathway scores were found.")
    selected["share_of_day6_score_pct"] = (
        100.0 * selected["summed_cytobridge_score"] / total_score
    )
    selected["display_rank"] = np.arange(1, len(selected) + 1)
    return selected


def draw_cytobridge_pathways(ax, selected: pd.DataFrame) -> None:
    plotted = selected.sort_values("share_of_day6_score_pct", ascending=True)
    y = np.arange(len(plotted))
    values = plotted["share_of_day6_score_pct"].to_numpy(dtype=float)
    pathway_labels = {
        "GRN": "GRN (granulin)",
        "GALECTIN": "Galectin",
        "CCL": "CCL chemokines",
        "ANNEXIN": "Annexin",
    }
    ax.barh(y, values, height=0.62, color=TEAL, alpha=0.88)
    ax.set_yticks(
        y,
        labels=[
            pathway_labels.get(str(name), str(name)) for name in plotted["pathway"]
        ],
    )
    ax.set_xlim(0, max(values) * 1.18)
    ax.set_xlabel("Relative interaction signal (%)")
    for yy, value in zip(y, values):
        ax.text(
            value + max(values) * 0.025,
            yy,
            f"{value:.1f}",
            ha="left",
            va="center",
            fontsize=7.0,
            color=HEADING_COLOR,
        )
    clean_axis(ax, grid=True)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="both", labelsize=7.0)


def draw_rank_scatter(ax, joined: pd.DataFrame, association: pd.DataFrame) -> dict:
    day6 = joined[joined["time"] == 6].copy()
    if len(day6) != 121:
        raise ValueError(f"Expected 121 Day-6 directed contexts, found {len(day6)}")
    hetero = day6[day6["heterotypic"].astype(bool)]
    self_edges = day6[~day6["heterotypic"].astype(bool)]
    ax.scatter(
        hetero["cellchat_raw_rank_percentile"],
        hetero["message_raw_rank_percentile"],
        s=18,
        color=LIGHT_TEAL,
        alpha=0.70,
        edgecolor="white",
        linewidth=0.25,
        label="heterotypic",
        zorder=2,
    )
    ax.scatter(
        self_edges["cellchat_raw_rank_percentile"],
        self_edges["message_raw_rank_percentile"],
        s=20,
        marker="s",
        color="#A5ADB3",
        alpha=0.80,
        edgecolor="white",
        linewidth=0.25,
        label="same type",
        zorder=2,
    )
    ax.plot([0, 1], [0, 1], linestyle=(0, (3, 2)), color="#AEB6BC", lw=0.7, zorder=0)
    focus = day6[
        (day6["sender_type"] == "Baso") & (day6["receiver_type"] == "Monocyte")
    ].iloc[0]
    cellchat_rank = int(
        day6["cellchat_native_raw"].rank(method="min", ascending=False).loc[focus.name]
    )
    message_rank = int(
        day6["message_D_AB_raw"].rank(method="min", ascending=False).loc[focus.name]
    )
    ax.scatter(
        focus["cellchat_raw_rank_percentile"],
        focus["message_raw_rank_percentile"],
        s=58,
        marker="*",
        color=CORAL,
        edgecolor="white",
        linewidth=0.5,
        zorder=4,
        label="Baso→Monocyte",
    )
    ax.annotate(
        f"Baso→Monocyte\nrank {cellchat_rank} in CellChat; rank {message_rank} in D",
        xy=(
            focus["cellchat_raw_rank_percentile"],
            focus["message_raw_rank_percentile"],
        ),
        xytext=(0.53, 0.73),
        textcoords="axes fraction",
        arrowprops={"arrowstyle": "-", "lw": 0.65, "color": CORAL},
        fontsize=7.4,
        color=DARK_GRAY,
        ha="left",
        va="top",
    )
    ax.set_xlim(-0.02, 1.03)
    ax.set_ylim(-0.02, 1.03)
    ax.set_xlabel("CellChat edge-rank percentile")
    ax.set_ylabel("CytoBridge D edge-rank percentile")
    ax.set_title(
        f"Day 6 · all {len(day6)} directed cell-type pairs", fontsize=8.5, pad=2
    )
    clean_axis(ax, grid=True)
    metric = association[
        (association["time"] == 6)
        & (association["method"] == "D_exact_message")
        & (association["universe"] == "all_edges")
        & (association["control_set"] == "unadjusted")
    ]
    if len(metric) != 1:
        raise ValueError(
            "Could not identify the unique Day-6 raw D--CellChat metric row."
        )
    rho = float(metric.iloc[0]["spearman_rho"])
    tau = float(metric.iloc[0]["kendall_tau_b_unadjusted"])
    ax.text(
        0.03,
        0.97,
        f"ρ = {rho:.3f}  ·  τ = {tau:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.2,
        color=HEADING_COLOR,
        fontweight="bold",
        bbox={
            "boxstyle": "round,pad=0.18",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.88,
        },
    )
    ax.legend(loc="lower right", frameon=False, fontsize=7.2, handletextpad=0.35)
    return {
        "n_edges": 121,
        "spearman_rho": rho,
        "kendall_tau_b": tau,
        "focus_sender": "Baso",
        "focus_receiver": "Monocyte",
        "focus_D_rank": message_rank,
        "focus_CellChat_rank": cellchat_rank,
        "focus_D_rank_percentile": float(focus["message_raw_rank_percentile"]),
        "focus_CellChat_rank_percentile": float(focus["cellchat_raw_rank_percentile"]),
    }


def selected_associations(association: pd.DataFrame) -> pd.DataFrame:
    selected = association[
        (association["method"] == "D_exact_message")
        & (association["universe"] == "all_edges")
        & association["control_set"].isin(
            ["unadjusted", "Q_abundance_sender_receiver_FE"]
        )
    ].copy()
    if len(selected) != 6:
        raise ValueError(
            f"Expected six selected CellChat association rows, found {len(selected)}"
        )
    selected["display_control"] = selected["control_set"].map(
        {
            "unadjusted": "Raw rank",
            "Q_abundance_sender_receiver_FE": "After Q + abundance + type FE",
        }
    )
    return selected


def draw_association_summary(
    ax, association: pd.DataFrame, stability: pd.DataFrame
) -> dict:
    selected = selected_associations(association)
    styles = {
        "Raw rank": (TEAL, "o", -0.08),
        "After Q + abundance + type FE": (CORAL, "s", 0.08),
    }
    for label, (color, marker, offset) in styles.items():
        block = selected[selected["display_control"] == label].sort_values("time")
        x = block["time"].to_numpy(dtype=float) + offset
        y = block["spearman_rho"].to_numpy(dtype=float)
        low = block["bootstrap_rho_ci_low"].to_numpy(dtype=float)
        high = block["bootstrap_rho_ci_high"].to_numpy(dtype=float)
        ax.errorbar(
            x,
            y,
            yerr=np.vstack([y - low, high - y]),
            color=color,
            marker=marker,
            markersize=4.5,
            linewidth=1.0,
            capsize=2.2,
            label=label,
            zorder=3,
        )
    ax.axhline(0, color=GT_COLOR, lw=0.8)
    ax.set_xticks([2, 4, 6], labels=["Day 2", "Day 4", "Day 6"])
    ax.set_ylim(-0.58, 1.02)
    ax.set_ylabel("Spearman ρ with CellChat")
    ax.set_title("Raw concordance and shared-prior sensitivity", fontsize=8.5, pad=2)
    clean_axis(ax, grid=True)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower right", frameon=False, fontsize=7.1, handletextpad=0.4)

    stable = stability[
        (stability["method"] == "D_exact_message")
        & (stability["universe"] == "all_edges")
    ]
    stable_min = float(stable["spearman_rho"].min())
    stable_max = float(stable["spearman_rho"].max())
    ax.text(
        0.03,
        0.97,
        f"Across independently trained seeds\nD edge ranks: ρ = {stable_min:.3f}–{stable_max:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.7,
        color=HEADING_COLOR,
        fontweight="bold",
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "#F5F8FA",
            "edgecolor": "#CED5DA",
            "lw": 0.6,
        },
    )
    ax.text(
        0.03,
        0.04,
        "Cluster-bootstrap 95% CI\nCompatibility ≠ causal validation",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.2,
        color=DARK_GRAY,
    )
    return {"seed_stability_min": stable_min, "seed_stability_max": stable_max}


def write_metrics(
    dist_effect: pd.DataFrame,
    clone_effect: pd.DataFrame,
    association: pd.DataFrame,
    composite_mean: float,
    network_edges: pd.DataFrame,
    concordance_summary: dict,
    pathway_summary: pd.DataFrame,
) -> None:
    METRICS.mkdir(parents=True, exist_ok=True)
    dist_effect.to_csv(METRICS / "distribution_primary_seed_effects.csv", index=False)
    clone_effect.to_csv(
        METRICS / "clone_fate_primary_seed_and_ensemble_effects.csv", index=False
    )
    network_edges.to_csv(METRICS / "day6_circle_network_display_edges.csv", index=False)
    pathway_summary.to_csv(METRICS / "cytobridge_day6_top_pathways.csv", index=False)
    selected = selected_associations(association)
    selected.to_csv(METRICS / "cellchat_raw_and_adjusted_associations.csv", index=False)
    summary = {
        "distribution": {
            "primary_comparison": "full interaction + score/noise versus independently trained no interaction + score/noise",
            "training_seeds": [42, 43, 44],
            "paired_inference_repeats_per_seed": 50,
            "equal_weight_W1_W2_relative_reduction_pct": composite_mean,
        },
        "clone_fate": {
            "training_seeds": [42, 43, 44],
            "ensemble_is_probability_average_not_independent_replicate": True,
        },
        "cellchat": {
            **concordance_summary,
            "circle_network_day": 6,
            "circle_selection": "independent top-30 positive heterotypic directed edges within each method",
            "circle_edge_width": "within-method display rank only; raw units are not compared",
            "concordance_universe": "all available directed cell-type contexts at each day",
            "time_aggregation": "unweighted arithmetic mean of Day 2, Day 4, and Day 6 coefficients",
            "same_expression_and_shared_lr_prior": True,
            "independent_or_causal_validation": False,
        },
        "cytobridge_pathway_readout": {
            "displayed_pathways": pathway_summary["pathway"].astype(str).tolist(),
            "day": 6,
            "selection": (
                "eight pathways with the largest summed positive CytoBridge "
                "S_AB_pathway across directed pairs of different cell types"
            ),
            "score": "S_AB_pathway = D_AB_mean * Q_AB_pathway",
            "bar_unit": "percentage of the summed positive Day-6 S_AB_pathway across all pathways and heterotypic directed cell-type pairs",
            "cellchat_or_nichenet_inference_used": False,
            "is_causal_validation": False,
        },
        "model_field": {
            "coordinate_system": "direct PC1-PC2 component of the 50-PC model state",
            "projection": "support-masked smoothed conditional mean of saved PC1-PC2 vector components",
            "spring_used_for_training": False,
            "is_rna_velocity": False,
            "full_drift": "v + grad(s) + i",
            "interaction_component": "i",
        },
    }
    (METRICS / "figure_summary_metrics.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_figure(dpi: int) -> None:
    apply_style()
    mpl.rcParams.update(
        {
            "axes.titleweight": "normal",
            "legend.handlelength": 1.25,
            "figure.dpi": 120,
            "savefig.bbox": None,
        }
    )
    cells = pd.read_csv(PANEL_DATA / "observed_cells.csv.gz")
    grids = np.load(PANEL_DATA / "model_field_grids.npz")
    distribution = pd.read_csv(PANEL_DATA / "distribution_per_repeat.csv")
    composite = pd.read_csv(PANEL_DATA / "distribution_w1_w2_composite_per_repeat.csv")
    clone_seed = pd.read_csv(PANEL_DATA / "clone_fate_per_training_seed.csv")
    clone_ensemble = pd.read_csv(PANEL_DATA / "clone_fate_ensemble.csv")
    joined = pd.read_csv(PANEL_DATA / "cellchat_joined_directed_edges.csv")
    association = pd.read_csv(PANEL_DATA / "cellchat_association_metrics.csv")
    stability = pd.read_csv(PANEL_DATA / "interaction_training_seed_rank_stability.csv")
    pathways = pd.read_csv(PANEL_DATA / "cytobridge_pathway_scores.csv")

    dist_effect = distribution_effects(distribution)
    clone_effect = clone_effects(clone_seed, clone_ensemble)

    fig = plt.figure(figsize=A4_PORTRAIT)
    outer = fig.add_gridspec(
        4,
        1,
        left=0.075,
        right=0.975,
        top=0.985,
        bottom=0.045,
        height_ratios=[1.85, 2.20, 1.55, 2.15],
        hspace=0.25,
    )

    draw_spring_panel(fig, outer[0], cells)

    draw_model_field_panel(fig, outer[1], cells, grids)

    middle = outer[2].subgridspec(1, 2, wspace=0.20)
    condition_handles = [
        Line2D(
            [], [], marker="o", linestyle="", color=TEAL, markersize=4.6, label="Full"
        ),
        Line2D(
            [],
            [],
            marker="s",
            linestyle="",
            color=CORAL,
            markersize=4.4,
            label="No interaction",
        ),
    ]
    d_body, d_heading = panel_container(fig, middle[0], "c", "Distribution error")
    d_heading.legend(
        handles=condition_handles,
        loc="center right",
        ncol=2,
        frameon=False,
        fontsize=6.5,
        handletextpad=0.22,
        columnspacing=0.5,
        borderaxespad=0,
    )
    ax_d = fig.add_subplot(d_body)
    composite_mean = draw_distribution_dumbbells(ax_d, distribution, composite)
    e_body, e_heading = panel_container(fig, middle[1], "d", "Clone-fate agreement")
    e_heading.legend(
        handles=condition_handles,
        loc="center right",
        ncol=2,
        frameon=False,
        fontsize=6.5,
        handletextpad=0.22,
        columnspacing=0.5,
        borderaxespad=0,
    )
    draw_clone_probability_dumbbells(fig, e_body, clone_ensemble)

    network_edges = day6_network_edges(joined, top_n=30)
    concordance_summary = summarize_concordance(association, stability, network_edges)
    pathway_summary = select_day6_cytobridge_pathways(pathways)
    bottom_row = outer[3].subgridspec(1, 2, width_ratios=[2.08, 0.92], wspace=0.18)
    f_body, f_heading = panel_container(
        fig, bottom_row[0], "e", "Day 6 directed cell–cell interactions"
    )
    f_heading.text(
        0.995,
        0.50,
        f"Spearman ρ = {concordance_summary['displayed_spearman_rho']:.3f}",
        ha="right",
        va="center",
        fontsize=7.6,
        color=HEADING_COLOR,
        fontweight="bold",
    )
    networks = f_body.subgridspec(1, 2, wspace=0.12)
    ax_f1 = fig.add_subplot(networks[0, 0])
    draw_circle_network(
        ax_f1,
        network_edges,
        joined,
        method="CytoBridge D",
        title="CytoBridge interaction network",
    )
    ax_f2 = fig.add_subplot(networks[0, 1])
    draw_circle_network(
        ax_f2,
        network_edges,
        joined,
        method="CellChat",
        title="CellChat communication network",
    )
    pathway_body, _ = panel_container(
        fig, bottom_row[1], "f", "Day 6 signaling pathways"
    )
    pathway_ax = fig.add_subplot(pathway_body)
    draw_cytobridge_pathways(pathway_ax, pathway_summary)

    pdf = BUNDLE / f"{FIGURE_STEM}.pdf"
    png = BUNDLE / f"{FIGURE_STEM}.png"
    save_figure(fig, pdf, png, dpi=dpi)
    plt.close(fig)
    write_metrics(
        dist_effect,
        clone_effect,
        association,
        composite_mean,
        network_edges,
        concordance_summary,
        pathway_summary,
    )
