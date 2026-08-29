#!/usr/bin/env python3
"""Rebuild AdMouse Figure c/d with weighted linkage only.

This version uses the formal r2 Microglia profiles and the same plotting
parameters. The hierarchical clustering linkage method used to assign four
temporal programs is ``weighted``. Program boundaries use the requested red
dashed Notebook style.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import cut_tree, leaves_list, linkage
from sklearn.metrics import silhouette_score


ROOT_DIR = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT_DIR / "figures"
DATA_DIR = ROOT_DIR / "data"
PROFILE_FILE = DATA_DIR / "gene_zscore_profiles.csv"
METADATA_FILE = DATA_DIR / "gene_temporal_metadata.csv"

N_CLUSTERS = 4
METHOD = "weighted"
DISPLAY_MAX_TIME = 2.4
MODEL_TIME_GRID = np.round(np.arange(0.0, DISPLAY_MAX_TIME + 0.001, 0.1), 1)
AGE_MONTH_GRID = np.array(
    [
        2.5, 2.8, 3.1, 3.5, 3.8, 4.1, 4.4, 4.7, 5.1, 5.4,
        5.7, 6.9, 8.1, 9.4, 10.6, 11.8, 13.0, 14.2, 15.5, 16.7,
        17.9, 18.7, 19.4, 20.2, 21.0,
    ],
    dtype=float,
)
DISPLAY_AGE_TICKS = np.arange(2.5, 20.0 + 0.001, 2.5)

PROGRAM_LABELS = {
    1: "Early-high / declining",
    2: "Mid-course-low / U-shaped",
    3: "Mid-course transient",
    4: "Late-high / rising",
}
PROGRAM_COLORS = {1: "#4C78A8", 2: "#F28E2B", 3: "#59A14F", 4: "#E15759"}
CURVE_COLORS = {1: "#2C7FB8", 2: "#D62728", 3: "#CC79A7", 4: "#33B5CC"}


def model_time_to_age_months(model_time: np.ndarray) -> np.ndarray:
    """Apply the requested explicit t=0.0--2.4 display-age mapping."""

    model_time = np.asarray(model_time, dtype=float)
    if np.any(model_time < MODEL_TIME_GRID[0]) or np.any(
        model_time > MODEL_TIME_GRID[-1]
    ):
        raise ValueError("Requested model time falls outside the t=0.0--2.4 map.")
    return np.interp(model_time, MODEL_TIME_GRID, AGE_MONTH_GRID)


def style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial"],
            "font.size": 8.5,
            "axes.titlesize": 9,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def panel_label(ax: mpl.axes.Axes, label: str) -> None:
    ax.text(
        -0.11,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def clean_axis(ax: mpl.axes.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(False)


def cluster_profiles(
    profiles: pd.DataFrame, method: str
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Match the package's exact four-cluster, peak-time ordered algorithm."""

    values = profiles.to_numpy(dtype=np.float64)
    hierarchy = linkage(values, method=method, metric="euclidean")
    raw_labels = cut_tree(hierarchy, n_clusters=[N_CLUSTERS]).reshape(-1) + 1

    peak_order: list[tuple[int, int]] = []
    for raw_label in sorted(np.unique(raw_labels)):
        prototype = values[raw_labels == raw_label].mean(axis=0)
        peak_order.append((int(np.argmax(prototype)), int(raw_label)))
    remap = {raw: index + 1 for index, (_, raw) in enumerate(sorted(peak_order))}
    labels = np.asarray([remap[int(raw)] for raw in raw_labels], dtype=int)

    metadata = pd.read_csv(METADATA_FILE)
    assignments = pd.DataFrame(
        {"gene": profiles.index.astype(str), "cluster": labels}
    ).merge(metadata, on="gene", how="left", validate="1:1")

    prototype_rows: list[dict[str, object]] = []
    times = profiles.columns.astype(float).to_numpy()
    for cluster in sorted(np.unique(labels)):
        subset = values[labels == cluster]
        for column_index, time in enumerate(times):
            prototype_rows.append(
                {
                    "cluster": int(cluster),
                    "time": float(time),
                    "mean": float(subset[:, column_index].mean()),
                    "std": float(subset[:, column_index].std()),
                    "n_genes": int(subset.shape[0]),
                }
            )
    prototypes = pd.DataFrame(prototype_rows)
    diagnostics = {
        "requested_clusters": N_CLUSTERS,
        "clusters_found": int(len(np.unique(labels))),
        "normalization": "gene-wise zscore (formal r2 profiles, unchanged)",
        "linkage_method": method,
        "metric": "euclidean",
        "cluster_order": "peak_time",
        "cut_strategy": "scipy_cut_tree_exact_n_clusters",
        "silhouette": float(silhouette_score(values, labels)),
        "cluster_sizes": {
            str(cluster): int(np.count_nonzero(labels == cluster))
            for cluster in sorted(np.unique(labels))
        },
    }
    return assignments, prototypes, diagnostics


def ordered_program_matrix(
    profiles: pd.DataFrame, assignments: pd.DataFrame
) -> tuple[pd.DataFrame, list[dict[str, int]]]:
    """Keep the formal plotting code's average-linkage within-block ordering."""

    order: list[str] = []
    blocks: list[dict[str, int]] = []
    cursor = 0
    for cluster in sorted(assignments["cluster"].unique()):
        genes = assignments.loc[assignments["cluster"].eq(cluster), "gene"].tolist()
        matrix = profiles.loc[genes].to_numpy(float)
        if len(genes) > 1:
            indices = leaves_list(linkage(matrix, method="average", metric="euclidean"))
            genes = [genes[index] for index in indices]
        order.extend(genes)
        blocks.append(
            {
                "cluster": int(cluster),
                "start": cursor,
                "stop": cursor + len(genes),
                "n": len(genes),
            }
        )
        cursor += len(genes)
    return profiles.loc[order], blocks


def draw_program_heatmap(
    ax: mpl.axes.Axes, profiles: pd.DataFrame, assignments: pd.DataFrame
) -> mpl.image.AxesImage:
    matrix, blocks = ordered_program_matrix(profiles, assignments)
    image = ax.imshow(
        matrix.to_numpy(float),
        aspect="auto",
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-2, vcenter=0, vmax=2),
        interpolation="nearest",
        rasterized=True,
    )
    times = matrix.columns.astype(float).to_numpy()
    ticks = np.arange(0, len(times), 5)
    if ticks[-1] != len(times) - 1:
        ticks = np.append(ticks, len(times) - 1)
    ax.set_xticks(ticks, [f"{times[index]:.1f}" for index in ticks])
    ax.set_xlabel("Model time")
    ax.set_ylabel("Active PCA genes")
    ax.set_yticks([])
    ax.set_title("Microglia reconstructed gene programs", loc="left", pad=5)
    for block in blocks:
        start, stop, cluster = block["start"], block["stop"], block["cluster"]
        if start:
            ax.axhline(
                start - 0.5,
                color="red",
                linestyle="--",
                linewidth=1.2,
                alpha=0.85,
            )
        ax.text(
            len(times) - 0.2,
            (start + stop - 1) / 2,
            f"P{cluster}  n={block['n']}",
            ha="right",
            va="center",
            fontsize=7.2,
            fontweight="bold",
            color=PROGRAM_COLORS[cluster],
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.2},
        )
    return image


def draw_program_curves(
    axes: list[mpl.axes.Axes],
    prototypes: pd.DataFrame,
    pattern_title_mode: str = "full",
) -> None:
    for cluster, ax in zip(sorted(prototypes["cluster"].unique()), axes):
        subset = prototypes[
            prototypes["cluster"].eq(cluster)
            & prototypes["time"].le(DISPLAY_MAX_TIME)
        ].sort_values("time")
        color = CURVE_COLORS[int(cluster)]
        ages = model_time_to_age_months(subset["time"].to_numpy(float))
        ax.fill_between(
            ages,
            subset["mean"] - subset["std"],
            subset["mean"] + subset["std"],
            color=color,
            alpha=0.18,
            linewidth=0,
        )
        ax.plot(ages, subset["mean"], color=color, lw=2.2)
        ax.axhline(0, color="#9A9A9A", lw=1.0, ls="--", alpha=0.8)
        display_end_age = float(
            model_time_to_age_months(np.array([DISPLAY_MAX_TIME]))[0]
        )
        ax.set_xlim(AGE_MONTH_GRID[0], display_end_age)
        ax.set_ylim(-2.6, 2.6)
        n_genes = int(subset["n_genes"].iloc[0])
        if pattern_title_mode == "full":
            ax.set_title(
                f"Pattern {int(cluster)} (n={n_genes})\n"
                f"({PROGRAM_LABELS[int(cluster)]})",
                loc="center",
                fontsize=8.5,
                pad=3,
            )
        elif pattern_title_mode == "short":
            ax.set_title(
                f"Pattern {int(cluster)} (n={n_genes})",
                loc="center",
                fontsize=8.5,
                pad=3,
            )
        age_ticks = np.append(DISPLAY_AGE_TICKS, display_end_age)
        ax.set_xticks(
            age_ticks,
            [f"{tick:.1f}" for tick in age_ticks],
            rotation=60,
        )
        ax.set_yticks([-2, -1, 0, 1, 2], [f"{tick:.1f}" for tick in [-2, -1, 0, 1, 2]])
        ax.set_xlabel("Age (months)")
        ax.set_ylabel("Z-score")
        ax.tick_params(axis="both", labelsize=6.5, width=0.8, length=2.5, pad=1)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(0.9)
        ax.grid(False)


def make_curve_figure(method: str, prototypes: pd.DataFrame) -> None:
    """Draw the weighted result in the original AdMouse Figure-d layout."""

    fig, axes_array = plt.subplots(
        2,
        2,
        figsize=(4.9, 5.45),
        gridspec_kw={"wspace": 0.43, "hspace": 0.58},
    )
    axes = list(axes_array.ravel())
    draw_program_curves(axes, prototypes, pattern_title_mode="full")
    panel_label(axes[0], "d")
    fig.subplots_adjust(left=0.14, right=0.97, top=0.91, bottom=0.12)
    stem = f"admouse_gene_program_dynamics_{method}_linkage"
    fig.savefig(FIGURE_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIGURE_DIR / f"{stem}.png", dpi=320, bbox_inches="tight")
    plt.close(fig)


def make_figure(
    method: str,
    profiles: pd.DataFrame,
    assignments: pd.DataFrame,
    prototypes: pd.DataFrame,
) -> None:
    # Preserve clustering and z-score values from the complete formal series.
    display_columns = [
        column for column in profiles.columns if float(column) <= DISPLAY_MAX_TIME
    ]
    display_profiles = profiles.loc[:, display_columns]
    display_prototypes = prototypes[prototypes["time"].le(DISPLAY_MAX_TIME)]

    fig = plt.figure(figsize=(8.27, 5.3))
    grid = fig.add_gridspec(
        2,
        3,
        width_ratios=[1.75, 0.78, 0.78],
        left=0.08,
        right=0.98,
        top=0.93,
        bottom=0.13,
        hspace=0.42,
        wspace=0.38,
    )
    heat = fig.add_subplot(grid[:, 0])
    image = draw_program_heatmap(heat, display_profiles, assignments)
    panel_label(heat, "c")
    curve_axes = [fig.add_subplot(grid[row, column]) for row in range(2) for column in (1, 2)]
    draw_program_curves(curve_axes, display_prototypes, pattern_title_mode="short")
    panel_label(curve_axes[0], "d")
    colorbar = fig.colorbar(image, ax=heat, fraction=0.035, pad=0.02)
    colorbar.set_label("Gene-wise z-score")
    stem = f"admouse_gene_program_dynamics_{method}_linkage_heatmap_and_curves"
    fig.savefig(FIGURE_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIGURE_DIR / f"{stem}.png", dpi=320, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    style()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    profiles = pd.read_csv(PROFILE_FILE).set_index("gene")
    if not np.isfinite(profiles.to_numpy(float)).all():
        raise ValueError("Formal r2 z-score profiles contain non-finite values.")

    diagnostics: dict[str, object] = {
        "source_profile": str(PROFILE_FILE),
        "n_genes": int(profiles.shape[0]),
        "n_times": int(profiles.shape[1]),
        "method": {},
    }
    assignments, prototypes, method_diagnostics = cluster_profiles(profiles, METHOD)
    pd.DataFrame(
        {"model_time": MODEL_TIME_GRID, "age_months": AGE_MONTH_GRID}
    ).to_csv(DATA_DIR / "model_time_to_age_mapping_t0_to_t2p4.csv", index=False)
    assignments.to_csv(
        DATA_DIR / "gene_cluster_assignments_weighted.csv", index=False
    )
    prototypes.to_csv(
        DATA_DIR / "gene_cluster_prototypes_weighted.csv", index=False
    )
    make_figure(METHOD, profiles, assignments, prototypes)
    make_curve_figure(METHOD, prototypes)
    diagnostics["method"] = method_diagnostics

    (DATA_DIR / "diagnostics_weighted.json").write_text(
        json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
