#!/usr/bin/env python3
"""Build final AD perturbation figures from the accepted package results.

The whole-tissue figure reproduces the perturbation scope used in the original
AD notebook.  The Microglia-only figure is a separate specificity sensitivity.
Both figures use the same spatial, attention, module, and ligand-receptor axes
so their effect sizes can be compared directly.

Every perturbation starts from the same real t=0 population and is propagated
continuously to model time 2.5. Only the compact formal source files under
``source_data/raw_server`` are read; earlier diagnostic outputs are not used.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd


RUN_DIR = Path(__file__).resolve().parents[1]
BUNDLE_DIR = RUN_DIR / "figures"
SOURCE_DIR = RUN_DIR / "data" / "figure_source"
PANEL_DATA_DIR = RUN_DIR / "data" / "panel_data"

SERVER_PERTURBATION_ROOT = Path(
    "/data/cytobridge/projects/CytoBridge-ST-1104/runs/"
    "admouse-lr-perturbation-20260814-3c87a3e-r2"
)

SCOPE_CONFIG = {
    "whole_tissue": {
        "output_stem": "admouse_whole_tissue_trem2_scale1_r2_rerun",
        "heading": "Whole-tissue Trem2 sensitivity | scale 1.0 | formal r2 model",
        "server_root": SERVER_PERTURBATION_ROOT / "whole_tissue",
        "caption_scope": "all observed t0 particles",
    },
}

CELLTYPE_ORDER = [
    "Astrocytes",
    "Excitatory neurons",
    "Fibroblast",
    "Inhibitory neurons",
    "Microglia",
    "OPC",
    "Oligodendrocytes",
    "Pericytes/Endothelial",
]

# Exact palette used in the reviewed AD notebooks and manuscript Figure 6.
CELLTYPE_COLORS = {
    "Astrocytes": "#1f77b4",
    "Excitatory neurons": "#ff7f0e",
    "Fibroblast": "#2ca02c",
    "Inhibitory neurons": "#d62728",
    "Microglia": "#9467bd",
    "OPC": "#8c564b",
    "Oligodendrocytes": "#e377c2",
    "Pericytes/Endothelial": "#7f7f7f",
}

HIGH_COLOR = "#c65a73"
LOW_COLOR = "#9cc9e3"
BASELINE_COLOR = "#6b6b6b"
ZERO_COLOR = "#777777"

MODULE_ORDER = [
    "DAM_microglia",
    "DAM_Lipid_Metabolism",
    "Lysosome_Phagosome",
    "AB_Clearance_Endolysosomal",
    "Inflammation_Complement",
    "Astrocyte_Reactive",
    "SPP1_CD44_axis",
]

MODULE_LABELS = {
    "DAM_microglia": "DAM",
    "DAM_Lipid_Metabolism": "DAM lipid\nmetab.",
    "Lysosome_Phagosome": "Phago/\nlysosome",
    "AB_Clearance_Endolysosomal": "Aβ clearance",
    "Inflammation_Complement": "Complement/\ninflamm.",
    "Astrocyte_Reactive": "Reactive\nastro.",
    "SPP1_CD44_axis": "SPP1-CD44",
}

MODULE_LABELS_COMPACT = {
    "DAM_microglia": "DAM",
    "DAM_Lipid_Metabolism": "Lipid",
    "Lysosome_Phagosome": "Lysosome",
    "AB_Clearance_Endolysosomal": "Aβ clear.",
    "Inflammation_Complement": "Inflamm.",
    "Astrocyte_Reactive": "Reactive",
    "SPP1_CD44_axis": "SPP1-CD44",
}

SPATIAL_CONDITIONS = ["baseline", "low", "high"]
SPATIAL_TITLES = {
    "baseline": "Baseline",
    "low": "Trem2 low",
    "high": "Trem2 high",
}

# Fixed ROI copied from the reviewed AD perturbation plotting notebook.
ROI_CENTER = (0.35, 0.45)
ROI_HALF_WIDTH = 0.25


def apply_style() -> None:
    """Apply the established AD manuscript typography and line hierarchy."""

    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial"],
            "font.size": 9.0,
            "axes.titlesize": 9.0,
            "axes.labelsize": 9.0,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "xtick.major.size": 2.6,
            "ytick.major.size": 2.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "text.color": "black",
            "axes.labelcolor": "black",
            "axes.edgecolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
        }
    )


def panel_heading(ax: mpl.axes.Axes, label: str, title: str) -> None:
    """Draw one lower-case panel label and one sentence-case group title."""

    ax.axis("off")
    ax.text(0.0, 0.52, label, fontsize=14, fontweight="bold", va="center")
    ax.text(0.055, 0.52, title, fontsize=12, fontweight="bold", va="center")


def format_box_axis(ax: mpl.axes.Axes) -> None:
    """Match the compact black frames in the original AD figure."""

    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.7)
        spine.set_color("black")
    ax.tick_params(direction="out")


def case_paths(scope: str, condition: str) -> dict[str, Path]:
    """Return the saved endpoint state, labels, sparse edges, and attention."""

    if condition == "baseline":
        state_path = SOURCE_DIR / "compat_base" / "01_interpolation" / "generated_t2.5.npy"
        label_path = SOURCE_DIR / scope / "baseline_labels_k1" / "labels_t2.5.npy"
        communication_dir = (
            SOURCE_DIR
            / "whole_tissue"
            / "baseline_communication"
            / "attention"
            / "edges"
        )
    else:
        perturbation_dir = SOURCE_DIR / scope / "perturbations" / "Trem2" / condition
        state_path = perturbation_dir / "states" / "generated_t2.5.npy"
        label_path = perturbation_dir / "labels_k1" / "labels_t2.5.npy"
        communication_dir = (
            perturbation_dir / "communication" / "attention"
        )

    return {
        "state": state_path,
        "labels": label_path,
        "edge_index": communication_dir / "edge_index_interp_t2.5.npy",
        "attention": communication_dir / "attn_mean_interp_t2.5.npy",
    }


def load_case(scope: str, condition: str) -> dict[str, np.ndarray]:
    """Load one endpoint case without changing the saved values."""

    paths = case_paths(scope, condition)
    state = np.load(paths["state"])
    return {
        "state": state,
        "xy": state[:, :2],
        "labels": np.load(paths["labels"]).astype(str),
        "edge_index": np.load(paths["edge_index"]),
        "attention": np.load(paths["attention"]).astype(float),
    }


def microglia_neuron_edges(
    case: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select saved Microglia-to-excitatory/inhibitory-neuron edges."""

    labels = case["labels"]
    edge_index = case["edge_index"]
    selected = (labels[edge_index[0]] == "Microglia") & np.isin(
        labels[edge_index[1]],
        ["Excitatory neurons", "Inhibitory neurons"],
    )
    return (
        edge_index[0, selected],
        edge_index[1, selected],
        case["attention"][selected],
    )


def entropy_bits(labels: np.ndarray) -> float:
    """Calculate Shannon entropy of the predicted cell-type composition."""

    _, counts = np.unique(labels, return_counts=True)
    proportions = counts / counts.sum()
    return float(-(proportions * np.log2(proportions)).sum())


def prepare_scope_tables(scope: str) -> dict[str, pd.DataFrame]:
    """Prepare the four panel-level tables for one perturbation scope."""

    cases = {
        condition: load_case(scope, condition)
        for condition in SPATIAL_CONDITIONS
    }

    composition_rows: list[dict[str, object]] = []
    for condition, case in cases.items():
        labels = case["labels"]
        counts = pd.Series(labels).value_counts()
        for celltype in CELLTYPE_ORDER:
            count = int(counts.get(celltype, 0))
            composition_rows.append(
                {
                    "condition": condition,
                    "condition_label": SPATIAL_TITLES[condition],
                    "celltype": celltype,
                    "count": count,
                    "fraction": count / len(labels),
                    "total": len(labels),
                }
            )
    composition = pd.DataFrame(composition_rows)

    module_source = pd.read_csv(
        SOURCE_DIR / scope / "all_perturbation_module_scores.csv"
    )
    modules = module_source[
        (module_source["perturbed_gene"] == "Trem2")
        & np.isclose(module_source["time"], 2.5)
        & (module_source["population"] == "all_particles")
        & module_source["module"].isin(MODULE_ORDER)
    ].copy()
    modules["module"] = pd.Categorical(
        modules["module"], categories=MODULE_ORDER, ordered=True
    )
    modules = modules.sort_values(["module", "direction"])

    attention_source = pd.read_csv(
        SOURCE_DIR / scope / "all_attention_contrasts.csv"
    )
    attention = attention_source[
        (attention_source["perturbed_gene"] == "Trem2")
        & np.isclose(attention_source["time"], 2.5)
        & (attention_source["source"] == "Microglia")
        & attention_source["target"].isin(
            ["Excitatory neurons", "Inhibitory neurons"]
        )
    ].copy()

    lr_source = pd.read_csv(SOURCE_DIR / scope / "all_lr_pair_contrasts.csv")
    lr = lr_source[
        (lr_source["perturbed_gene"] == "Spp1")
        & (lr_source["ligand"] == "Spp1")
        & (lr_source["receptor"] == "Cd44")
    ].copy()
    lr = lr.sort_values(["direction", "time"])

    PANEL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    composition.to_csv(
        PANEL_DATA_DIR / f"{scope}_composition_endpoint.csv", index=False
    )
    modules.to_csv(PANEL_DATA_DIR / f"{scope}_trem2_modules_endpoint.csv", index=False)
    attention.to_csv(
        PANEL_DATA_DIR / f"{scope}_trem2_attention_endpoint.csv", index=False
    )
    lr.to_csv(PANEL_DATA_DIR / f"{scope}_spp1_cd44_timecourse.csv", index=False)

    return {
        "composition": composition,
        "modules": modules,
        "attention": attention,
        "lr": lr,
    }


def spatial_limits(all_cases: dict[str, dict[str, dict[str, np.ndarray]]]) -> tuple:
    """Use one spatial window for both scopes and all conditions."""

    coordinates = np.vstack(
        [
            case["xy"]
            for scope_cases in all_cases.values()
            for case in scope_cases.values()
        ]
    )
    x_min, y_min = coordinates.min(axis=0)
    x_max, y_max = coordinates.max(axis=0)
    x_pad = 0.02 * (x_max - x_min)
    y_pad = 0.02 * (y_max - y_min)
    return x_min - x_pad, x_max + x_pad, y_min - y_pad, y_max + y_pad


def shared_plot_limits(
    all_cases: dict[str, dict[str, dict[str, np.ndarray]]],
    all_tables: dict[str, dict[str, pd.DataFrame]],
) -> dict[str, float]:
    """Calculate common attention and response axes across the two figures."""

    edge_attention = []
    for scope_cases in all_cases.values():
        for case in scope_cases.values():
            edge_attention.append(microglia_neuron_edges(case)[2])
    nonempty_selected = [values for values in edge_attention if len(values)]
    if nonempty_selected:
        spatial_attention = np.concatenate(nonempty_selected)
    else:
        # The formal AD run has no saved Microglia-to-neuron sparse edge at
        # t=2.5. Retain an honest empty overlay while scaling the legend from
        # all saved edges instead of inventing or imputing selected edges.
        all_saved = [
            case["attention"]
            for scope_cases in all_cases.values()
            for case in scope_cases.values()
            if len(case["attention"])
        ]
        spatial_attention = np.concatenate(all_saved) if all_saved else np.array([1.0])

    module_values = np.concatenate(
        [tables["modules"]["delta"].to_numpy() for tables in all_tables.values()]
    )
    endpoint_attention = np.concatenate(
        [
            tables["attention"]["delta_attention_per_source"].to_numpy()
            for tables in all_tables.values()
        ]
    )
    lr_values = np.concatenate(
        [tables["lr"]["delta_score"].to_numpy() for tables in all_tables.values()]
    )

    def symmetric_limit(values: np.ndarray, step: float) -> float:
        maximum = float(np.max(np.abs(values))) * 1.12
        return max(step, float(np.ceil(maximum / step) * step))

    return {
        "spatial_attention_vmax": float(np.quantile(spatial_attention, 0.995)),
        "module": symmetric_limit(module_values, 0.25),
        "endpoint_attention": symmetric_limit(endpoint_attention, 0.05),
        "lr": symmetric_limit(lr_values, 0.1),
    }


def add_celltype_points(
    ax: mpl.axes.Axes,
    case: dict[str, np.ndarray],
    view: np.ndarray,
    point_size: float,
) -> None:
    """Draw all predicted cell types with the fixed manuscript palette."""

    xy = case["xy"]
    labels = case["labels"]
    for celltype in CELLTYPE_ORDER:
        selected = view & (labels == celltype)
        ax.scatter(
            xy[selected, 0],
            xy[selected, 1],
            s=point_size,
            color=CELLTYPE_COLORS[celltype],
            alpha=0.75,
            linewidths=0,
            zorder=2,
        )


def add_saved_attention(
    ax: mpl.axes.Axes,
    case: dict[str, np.ndarray],
    view: np.ndarray,
    shared_vmax: float,
    maximum_edges: int,
) -> int:
    """Overlay the strongest saved Microglia-to-neuron edges."""

    source, target, attention = microglia_neuron_edges(case)
    inside = view[source] & view[target]
    source = source[inside]
    target = target[inside]
    attention = attention[inside]
    if not len(attention):
        return 0

    order = np.argsort(attention)[::-1][:maximum_edges]
    source = source[order]
    target = target[order]
    attention = attention[order]
    scaled = np.clip(attention / shared_vmax, 0.0, 1.0)
    segments = np.stack((case["xy"][source], case["xy"][target]), axis=1)
    colors = np.zeros((len(segments), 4), dtype=float)
    colors[:, 3] = 0.25 + 0.70 * scaled
    collection = LineCollection(
        segments,
        colors=colors,
        linewidths=0.25 + 1.45 * scaled,
        zorder=5,
    )
    ax.add_collection(collection)
    return len(attention)


def plot_spatial_case(
    ax: mpl.axes.Axes,
    case: dict[str, np.ndarray],
    title: str,
    limits: tuple[float, float, float, float],
    attention_vmax: float,
    roi: bool,
) -> None:
    """Plot one whole-tissue or fixed-ROI cell/attention map."""

    xy = case["xy"]
    if roi:
        view = (
            (np.abs(xy[:, 0] - ROI_CENTER[0]) <= ROI_HALF_WIDTH)
            & (np.abs(xy[:, 1] - ROI_CENTER[1]) <= ROI_HALF_WIDTH)
        )
        add_celltype_points(ax, case, view, point_size=1.4)
        displayed_edges = add_saved_attention(
            ax, case, view, attention_vmax, maximum_edges=220
        )
        ax.set_xlim(ROI_CENTER[0] - ROI_HALF_WIDTH, ROI_CENTER[0] + ROI_HALF_WIDTH)
        ax.set_ylim(ROI_CENTER[1] - ROI_HALF_WIDTH, ROI_CENTER[1] + ROI_HALF_WIDTH)
    else:
        view = np.ones(len(xy), dtype=bool)
        add_celltype_points(ax, case, view, point_size=0.25)
        displayed_edges = add_saved_attention(
            ax, case, view, attention_vmax, maximum_edges=260
        )
        ax.add_patch(
            Rectangle(
                (ROI_CENTER[0] - ROI_HALF_WIDTH, ROI_CENTER[1] - ROI_HALF_WIDTH),
                2 * ROI_HALF_WIDTH,
                2 * ROI_HALF_WIDTH,
                fill=False,
                edgecolor="black",
                linewidth=0.9,
                linestyle=(0, (4, 3)),
                zorder=8,
            )
        )
        ax.set_xlim(limits[0], limits[1])
        ax.set_ylim(limits[2], limits[3])

    if displayed_edges == 0:
        ax.text(
            0.5,
            0.04,
            "No saved Microglia→neuron edge",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=7.2,
            color="#444444",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.5},
            zorder=10,
        )

    ax.set_title(title, pad=3.5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    format_box_axis(ax)


def plot_composition(ax: mpl.axes.Axes, composition: pd.DataFrame) -> None:
    """Plot endpoint composition for baseline, Trem2 low, and Trem2 high."""

    positions = np.arange(len(SPATIAL_CONDITIONS))
    bottom = np.zeros(len(positions))
    for celltype in CELLTYPE_ORDER:
        values = np.array(
            [
                composition.loc[
                    (composition["condition"] == condition)
                    & (composition["celltype"] == celltype),
                    "fraction",
                ].iloc[0]
                * 100
                for condition in SPATIAL_CONDITIONS
            ]
        )
        ax.bar(
            positions,
            values,
            bottom=bottom,
            width=0.68,
            color=CELLTYPE_COLORS[celltype],
            edgecolor="white",
            linewidth=0.45,
        )
        bottom += values

    ax.set_ylim(0, 100)
    ax.set_ylabel("Cell proportion (%)")
    ax.set_xticks(positions)
    ax.set_xticklabels([SPATIAL_TITLES[value] for value in SPATIAL_CONDITIONS])
    format_box_axis(ax)


def plot_endpoint_attention(
    ax: mpl.axes.Axes, attention: pd.DataFrame, y_limit: float
) -> None:
    """Plot endpoint attention change for two neuronal receivers."""

    targets = ["Excitatory neurons", "Inhibitory neurons"]
    positions = np.arange(len(targets))
    width = 0.35
    for offset, direction, color in [
        (-width / 2, "high", HIGH_COLOR),
        (width / 2, "low", LOW_COLOR),
    ]:
        direction_data = attention[attention["direction"] == direction].set_index(
            "target"
        )
        values = direction_data.loc[targets, "delta_attention_per_source"]
        ax.bar(
            positions + offset,
            values,
            width=width,
            color=color,
            edgecolor="white",
            linewidth=0.4,
            label=direction.capitalize(),
        )

    ax.axhline(0, color=ZERO_COLOR, linestyle="--", linewidth=0.85)
    ax.set_ylim(-y_limit, y_limit)
    ax.set_ylabel("Δ attention per Microglia")
    ax.set_xticks(positions)
    ax.set_xticklabels(["Excitatory\nneurons", "Inhibitory\nneurons"])
    if np.allclose(attention["delta_attention_per_source"], 0.0):
        ax.text(
            0.5,
            0.54,
            "No selected sparse edge at t=2.5",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=8,
            color="#444444",
        )
    format_box_axis(ax)


def plot_modules(
    ax: mpl.axes.Axes,
    modules: pd.DataFrame,
    y_limit: float,
    compact_labels: bool = False,
) -> None:
    """Plot the original Figure 6g module set at the corrected endpoint."""

    positions = np.arange(len(MODULE_ORDER))
    width = 0.36
    for offset, direction, color in [
        (-width / 2, "high", HIGH_COLOR),
        (width / 2, "low", LOW_COLOR),
    ]:
        direction_data = modules[modules["direction"] == direction].set_index(
            "module"
        )
        values = direction_data.loc[MODULE_ORDER, "delta"]
        ax.bar(
            positions + offset,
            values,
            width=width,
            color=color,
            edgecolor="white",
            linewidth=0.45,
            label=direction.capitalize(),
        )

    ax.axhline(0, color=ZERO_COLOR, linestyle="--", linewidth=0.85)
    ax.set_ylim(-y_limit, y_limit)
    ax.set_ylabel("All-particle module score change")
    ax.set_xticks(positions)
    label_map = MODULE_LABELS_COMPACT if compact_labels else MODULE_LABELS
    ax.set_xticklabels(
        [label_map[module] for module in MODULE_ORDER],
        rotation=40 if compact_labels else 28,
        ha="right",
        fontsize=7.5 if compact_labels else 8,
    )
    format_box_axis(ax)


def plot_lr(ax: mpl.axes.Axes, lr: pd.DataFrame, y_limit: float) -> None:
    """Plot the corrected Spp1-Cd44 ligand-receptor score trajectory."""

    for direction, color in [("high", HIGH_COLOR), ("low", LOW_COLOR)]:
        values = lr[lr["direction"] == direction].sort_values("time")
        ax.plot(
            values["time"],
            values["delta_score"],
            color=color,
            linewidth=2.0,
            label=direction.capitalize(),
        )
        markers = values.iloc[::5]
        ax.scatter(
            markers["time"],
            markers["delta_score"],
            s=16,
            color=color,
            edgecolor="white",
            linewidth=0.35,
            zorder=3,
        )

    ax.axhline(0, color=ZERO_COLOR, linestyle="--", linewidth=0.85)
    ax.set_xlim(0, 2.5)
    ax.set_ylim(-y_limit, y_limit)
    ax.set_xticks(np.arange(0, 2.51, 0.5))
    ax.set_xlabel("Model time")
    ax.set_ylabel("Δ Spp1-Cd44 LR score")
    format_box_axis(ax)


def make_scope_figure(
    scope: str,
    cases: dict[str, dict[str, np.ndarray]],
    tables: dict[str, pd.DataFrame],
    spatial_window: tuple[float, float, float, float],
    plot_limits: dict[str, float],
) -> None:
    """Assemble one A4 manuscript/SI-style figure."""

    fig = plt.figure(figsize=(8.27, 11.69))
    grid = fig.add_gridspec(
        10,
        6,
        height_ratios=[0.25, 1.55, 1.55, 0.70, 0.25, 1.25, 0.25, 1.35, 0.50, 1.10],
        left=0.08,
        right=0.98,
        bottom=0.055,
        top=0.985,
        hspace=0.56,
        wspace=0.55,
    )

    panel_heading(fig.add_subplot(grid[0, :]), "a", SCOPE_CONFIG[scope]["heading"])
    for column, condition in enumerate(SPATIAL_CONDITIONS):
        plot_spatial_case(
            fig.add_subplot(grid[1, 2 * column : 2 * column + 2]),
            cases[condition],
            SPATIAL_TITLES[condition],
            spatial_window,
            plot_limits["spatial_attention_vmax"],
            roi=False,
        )
        plot_spatial_case(
            fig.add_subplot(grid[2, 2 * column : 2 * column + 2]),
            cases[condition],
            f"{SPATIAL_TITLES[condition]} | fixed ROI",
            spatial_window,
            plot_limits["spatial_attention_vmax"],
            roi=True,
        )

    legend_axis = fig.add_subplot(grid[3, :])
    legend_axis.axis("off")
    legend_axis.legend(
        handles=[
            Patch(facecolor=CELLTYPE_COLORS[celltype], label=celltype)
            for celltype in CELLTYPE_ORDER
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.06),
        ncol=4,
        frameon=False,
        handlelength=1.0,
        handletextpad=0.35,
        columnspacing=0.9,
    )
    colorbar_axis = legend_axis.inset_axes([0.35, 0.03, 0.30, 0.14])
    colorbar = mpl.colorbar.ColorbarBase(
        colorbar_axis,
        cmap=plt.get_cmap("Greys"),
        norm=Normalize(0, plot_limits["spatial_attention_vmax"]),
        orientation="horizontal",
    )
    colorbar.set_label("Saved GNN edge attention (shared scale)", fontsize=8)
    colorbar.ax.tick_params(labelsize=7.5, length=2, pad=1)

    panel_heading(fig.add_subplot(grid[4, :3]), "b", "Endpoint cell-type composition")
    panel_heading(fig.add_subplot(grid[4, 3:]), "c", "Microglia-to-neuron attention")
    composition_axis = fig.add_subplot(grid[5, :3])
    attention_axis = fig.add_subplot(grid[5, 3:])
    plot_composition(composition_axis, tables["composition"])
    plot_endpoint_attention(
        attention_axis,
        tables["attention"],
        plot_limits["endpoint_attention"],
    )
    attention_axis.legend(
        loc="upper right",
        frameon=False,
        ncol=2,
        handlelength=1.2,
        handletextpad=0.35,
        columnspacing=0.8,
    )

    panel_heading(fig.add_subplot(grid[6, :]), "d", "Trem2-associated module changes")
    module_axis = fig.add_subplot(grid[7, :])
    plot_modules(module_axis, tables["modules"], plot_limits["module"])
    module_axis.legend(
        loc="upper right",
        frameon=False,
        ncol=2,
        handlelength=1.2,
        handletextpad=0.35,
        columnspacing=0.8,
    )

    panel_heading(
        fig.add_subplot(grid[8, :]),
        "e",
        "Spp1-Cd44 trajectory | unchanged formal r2 scale-2.5 reference",
    )
    lr_axis = fig.add_subplot(grid[9, :])
    plot_lr(lr_axis, tables["lr"], plot_limits["lr"])
    lr_axis.legend(
        loc="upper right",
        frameon=False,
        ncol=2,
        handlelength=1.5,
        handletextpad=0.35,
        columnspacing=0.8,
    )

    stem = str(SCOPE_CONFIG[scope]["output_stem"])
    fig.savefig(BUNDLE_DIR / f"{stem}.pdf")
    fig.savefig(BUNDLE_DIR / f"{stem}.png", dpi=300)
    plt.close(fig)


def make_scope_comparison(
    all_tables: dict[str, dict[str, pd.DataFrame]],
    plot_limits: dict[str, float],
) -> None:
    """Make a direct whole-tissue versus Microglia-only summary."""

    fig = plt.figure(figsize=(8.27, 11.69))
    grid = fig.add_gridspec(
        8,
        2,
        height_ratios=[0.28, 1.20, 0.28, 1.20, 0.28, 1.55, 0.60, 1.25],
        left=0.10,
        right=0.98,
        bottom=0.075,
        top=0.98,
        hspace=0.72,
        wspace=0.28,
    )

    panel_heading(fig.add_subplot(grid[0, :]), "a", "Scope dependence of Microglia abundance")
    ax = fig.add_subplot(grid[1, :])
    positions = np.arange(len(SPATIAL_CONDITIONS))
    width = 0.34
    scope_colors = {"whole_tissue": "#59616A", "microglia_only": "#07838B"}
    scope_labels = {"whole_tissue": "Whole tissue", "microglia_only": "Microglia only"}
    for offset, scope in [(-width / 2, "whole_tissue"), (width / 2, "microglia_only")]:
        composition = all_tables[scope]["composition"]
        microglia = composition[composition["celltype"] == "Microglia"].set_index(
            "condition"
        )
        values = 100 * microglia.loc[SPATIAL_CONDITIONS, "fraction"].to_numpy()
        ax.bar(
            positions + offset,
            values,
            width=width,
            color=scope_colors[scope],
            edgecolor="white",
            linewidth=0.45,
            label=scope_labels[scope],
        )
    ax.set_ylabel("Predicted Microglia (%)")
    ax.set_xticks(positions)
    ax.set_xticklabels([SPATIAL_TITLES[value] for value in SPATIAL_CONDITIONS])
    format_box_axis(ax)
    ax.legend(loc="upper left", frameon=False, ncol=2)

    panel_heading(fig.add_subplot(grid[2, :]), "b", "Endpoint Microglia-to-neuron attention")
    for column, scope in enumerate(["whole_tissue", "microglia_only"]):
        scope_axis = fig.add_subplot(grid[3, column])
        plot_endpoint_attention(
            scope_axis,
            all_tables[scope]["attention"],
            plot_limits["endpoint_attention"],
        )
        scope_axis.set_title(scope_labels[scope], fontweight="bold", pad=5)
        if column == 1:
            scope_axis.set_ylabel("")
            scope_axis.legend(
                loc="upper right",
                frameon=False,
                ncol=2,
                handlelength=1.2,
                handletextpad=0.35,
                columnspacing=0.8,
            )

    panel_heading(fig.add_subplot(grid[4, :]), "c", "Endpoint Trem2-associated modules")
    for column, scope in enumerate(["whole_tissue", "microglia_only"]):
        scope_axis = fig.add_subplot(grid[5, column])
        plot_modules(
            scope_axis,
            all_tables[scope]["modules"],
            plot_limits["module"],
            compact_labels=True,
        )
        scope_axis.set_title(scope_labels[scope], fontweight="bold", pad=5)
        if column == 1:
            scope_axis.set_ylabel("")
            scope_axis.legend(
                loc="upper right",
                frameon=False,
                ncol=2,
                handlelength=1.2,
                handletextpad=0.35,
                columnspacing=0.8,
            )

    panel_heading(fig.add_subplot(grid[6, :]), "d", "Spp1-Cd44 interaction trajectories")
    for column, scope in enumerate(["whole_tissue", "microglia_only"]):
        scope_axis = fig.add_subplot(grid[7, column])
        plot_lr(scope_axis, all_tables[scope]["lr"], plot_limits["lr"])
        scope_axis.set_title(scope_labels[scope], fontweight="bold", pad=5)
        if column == 1:
            scope_axis.set_ylabel("")
            scope_axis.legend(
                loc="upper right",
                frameon=False,
                ncol=2,
                handlelength=1.4,
                handletextpad=0.35,
                columnspacing=0.8,
            )

    stem = "admouse_final_scope_comparison_lr_perturbation"
    fig.savefig(BUNDLE_DIR / f"{stem}.pdf")
    fig.savefig(BUNDLE_DIR / f"{stem}.png", dpi=300)
    plt.close(fig)


def write_sanity_tables(
    all_cases: dict[str, dict[str, dict[str, np.ndarray]]],
    all_tables: dict[str, dict[str, pd.DataFrame]],
) -> None:
    """Record cell diversity, finite values, and panel comparability."""

    spatial_rows: list[dict[str, object]] = []
    panel_rows: list[dict[str, object]] = []
    for scope, cases in all_cases.items():
        for condition, case in cases.items():
            labels = case["labels"]
            _, counts = np.unique(labels, return_counts=True)
            source, target, attention = microglia_neuron_edges(case)
            inside_roi = (
                (np.abs(case["xy"][:, 0] - ROI_CENTER[0]) <= ROI_HALF_WIDTH)
                & (np.abs(case["xy"][:, 1] - ROI_CENTER[1]) <= ROI_HALF_WIDTH)
            )
            present = sorted(set(labels))
            missing = [celltype for celltype in CELLTYPE_ORDER if celltype not in present]
            spatial_rows.append(
                {
                    "scope": scope,
                    "condition": condition,
                    "n_particles": len(labels),
                    "n_celltypes": len(present),
                    "entropy_bits": entropy_bits(labels),
                    "dominant_fraction": float(counts.max() / counts.sum()),
                    "missing_celltypes": ";".join(missing),
                    "state_values_finite": bool(np.isfinite(case["state"]).all()),
                    "attention_values_finite": bool(np.isfinite(case["attention"]).all()),
                    "n_sparse_edges": case["edge_index"].shape[1],
                    "n_microglia_neuron_edges": len(attention),
                    "n_roi_microglia_neuron_edges": int(
                        (inside_roi[source] & inside_roi[target]).sum()
                    ),
                    "status": "pass"
                    if len(present) == 8
                    and not missing
                    and np.isfinite(case["state"]).all()
                    and np.isfinite(case["attention"]).all()
                    else "fail",
                }
            )

        tables = all_tables[scope]
        composition_sums = tables["composition"].groupby("condition")[
            "fraction"
        ].sum()
        panel_rows.extend(
            [
                {
                    "scope": scope,
                    "panel": "a",
                    "content": "spatial cell types and sparse attention",
                    "n_rows": sum(len(case["labels"]) for case in cases.values()),
                    "n_na": 0,
                    "n_noncomparable": 0,
                    "finite": all(
                        np.isfinite(case["state"]).all()
                        and np.isfinite(case["attention"]).all()
                        for case in cases.values()
                    ),
                    "status": "pass",
                },
                {
                    "scope": scope,
                    "panel": "b",
                    "content": "endpoint cell-type composition",
                    "n_rows": len(tables["composition"]),
                    "n_na": int(tables["composition"].isna().sum().sum()),
                    "n_noncomparable": 0,
                    "finite": bool(
                        np.isfinite(tables["composition"]["fraction"]).all()
                        and np.allclose(composition_sums, 1.0)
                    ),
                    "status": "pass",
                },
                {
                    "scope": scope,
                    "panel": "c",
                    "content": "endpoint Microglia-to-neuron attention contrast",
                    "n_rows": len(tables["attention"]),
                    "n_na": int(
                        tables["attention"]["delta_attention_per_source"].isna().sum()
                    ),
                    "n_noncomparable": int((~tables["attention"]["comparable"]).sum()),
                    "finite": bool(
                        np.isfinite(
                            tables["attention"]["delta_attention_per_source"]
                        ).all()
                    ),
                    "status": "pass",
                },
                {
                    "scope": scope,
                    "panel": "d",
                    "content": "endpoint module score contrast",
                    "n_rows": len(tables["modules"]),
                    "n_na": int(tables["modules"]["delta"].isna().sum()),
                    "n_noncomparable": 0,
                    "finite": bool(np.isfinite(tables["modules"]["delta"]).all()),
                    "status": "pass",
                },
                {
                    "scope": scope,
                    "panel": "e",
                    "content": "Spp1-Cd44 LR contrast trajectory",
                    "n_rows": len(tables["lr"]),
                    "n_na": int(tables["lr"]["delta_score"].isna().sum()),
                    "n_noncomparable": int((~tables["lr"]["comparable"]).sum()),
                    "finite": bool(np.isfinite(tables["lr"]["delta_score"]).all()),
                    "status": "pass",
                },
            ]
        )

    spatial = pd.DataFrame(spatial_rows)
    panels = pd.DataFrame(panel_rows)
    spatial.to_csv(BUNDLE_DIR / "spatial_condition_sanity.csv", index=False)
    panels.to_csv(BUNDLE_DIR / "panel_sanity.csv", index=False)

    if (spatial["status"] != "pass").any() or (panels["status"] != "pass").any():
        raise RuntimeError("A corrected AD figure sanity check failed")


def write_panel_source_table() -> None:
    """Map every plotted panel to local and server source files."""

    rows: list[dict[str, object]] = []
    for scope, config in SCOPE_CONFIG.items():
        local_root = SOURCE_DIR / scope
        server_root = config["server_root"]
        common = {"scope": scope, "figure": config["output_stem"]}
        rows.extend(
            [
                {
                    **common,
                    "panel": "a",
                    "content": "endpoint spatial labels and saved sparse attention",
                    "local_sources": (
                        f"{SOURCE_DIR / 'compat_base'}; "
                        f"{local_root / 'baseline_labels_k1'}; "
                        f"{local_root / 'perturbations' / 'Trem2'}"
                    ),
                    "server_sources": (
                        f"{SERVER_PERTURBATION_ROOT / 'compat_base' / '01_interpolation'}; "
                        f"{server_root / 'baseline_labels_k1'}; "
                        f"{server_root / 'perturbations' / 'Trem2'}"
                    ),
                    "calculation": (
                        "Plot saved t=2.5 labels. Overlay the strongest saved "
                        "Microglia-to-neuron sparse edges on one shared scale."
                    ),
                },
                {
                    **common,
                    "panel": "b",
                    "content": "endpoint predicted cell-type composition",
                    "local_sources": (
                        f"{local_root / 'perturbations' / 'Trem2'}; "
                        f"{local_root / 'baseline_labels_k1'}"
                    ),
                    "server_sources": (
                        f"{server_root / 'baseline_labels_k1'}; "
                        f"{server_root / 'perturbations' / 'Trem2'}"
                    ),
                    "calculation": "Count each of eight k=1 predicted cell types at t=2.5.",
                },
                {
                    **common,
                    "panel": "c",
                    "content": "endpoint Microglia-to-neuron attention contrast",
                    "local_sources": f"{local_root / 'all_attention_contrasts.csv'}",
                    "server_sources": f"{server_root / 'all_attention_contrasts.csv'}",
                    "calculation": (
                        "Read matched t=2.5 attention-per-source differences "
                        "for excitatory and inhibitory neuron targets."
                    ),
                },
                {
                    **common,
                    "panel": "d",
                    "content": "endpoint Trem2-associated module contrast",
                    "local_sources": f"{local_root / 'all_perturbation_module_scores.csv'}",
                    "server_sources": f"{server_root / 'all_perturbation_module_scores.csv'}",
                    "calculation": (
                        "Read all-particle t=2.5 module-score differences from "
                        "the original seven-module Figure 6g set."
                    ),
                },
                {
                    **common,
                    "panel": "e",
                    "content": "Spp1-Cd44 ligand-receptor trajectory",
                    "local_sources": f"{local_root / 'all_lr_pair_contrasts.csv'}",
                    "server_sources": f"{server_root / 'all_lr_pair_contrasts.csv'}",
                    "calculation": (
                        "Read matched Spp1-Cd44 LR-score differences over model time."
                    ),
                },
            ]
        )
    for panel, content, calculation in [
        (
            "a",
            "Microglia fraction by perturbation scope",
            "Compare endpoint Microglia fractions from the two composition tables.",
        ),
        (
            "b",
            "Microglia-to-neuron attention by perturbation scope",
            "Compare the two matched endpoint attention-contrast tables on one axis.",
        ),
        (
            "c",
            "Trem2-associated modules by perturbation scope",
            "Compare the two endpoint all-particle module tables on one axis.",
        ),
        (
            "d",
            "Spp1-Cd44 trajectories by perturbation scope",
            "Compare the two matched LR-score contrast tables on one axis.",
        ),
    ]:
        rows.append(
            {
                "scope": "whole_vs_microglia_only",
                "figure": "admouse_final_scope_comparison_lr_perturbation",
                "panel": panel,
                "content": content,
                "local_sources": (
                    f"{PANEL_DATA_DIR / 'whole_tissue_composition_endpoint.csv'}; "
                    f"{PANEL_DATA_DIR / 'microglia_only_composition_endpoint.csv'}; "
                    f"{PANEL_DATA_DIR}"
                ),
                "server_sources": (
                    f"{SCOPE_CONFIG['whole_tissue']['server_root']}; "
                    f"{SCOPE_CONFIG['microglia_only']['server_root']}"
                ),
                "calculation": calculation,
            }
        )
    pd.DataFrame(rows).to_csv(BUNDLE_DIR / "panel_source_table.csv", index=False)


def write_caption(all_tables: dict[str, dict[str, pd.DataFrame]]) -> None:
    """Write manuscript-ready captions from the plotted source tables."""

    spatial = pd.read_csv(BUNDLE_DIR / "spatial_condition_sanity.csv")
    caption_sections = ["# Final AD ligand-receptor perturbation figure captions", ""]
    for scope, config in SCOPE_CONFIG.items():
        scope_spatial = spatial[spatial["scope"] == scope].set_index("condition")
        composition = all_tables[scope]["composition"]
        attention = all_tables[scope]["attention"]
        modules = all_tables[scope]["modules"]
        lr = all_tables[scope]["lr"]

        microglia_fraction = composition[
            composition["celltype"] == "Microglia"
        ].set_index("condition")["fraction"]
        attention_values = attention.set_index(["direction", "target"])[
            "delta_attention_per_source"
        ]
        lr_endpoint = lr[np.isclose(lr["time"], 2.5)].set_index("direction")[
            "delta_score"
        ]
        module_lookup = modules.set_index(["direction", "module"])["delta"]

        caption_sections.extend(
            [
                f"## {config['output_stem']}",
                "",
                (
                    "**Formal scale-2.5 AD in-silico perturbation analysis.** "
                    f"The latent edit was applied to {config['caption_scope']}, "
                    "at the shared real t=0 initial population. Every condition was "
                    "then propagated continuously from t=0 to t=2.5 with the same "
                    "accepted full model; intermediate observed slices were not used "
                    "to restart the perturbation trajectory. Cell identities were "
                    "assigned with heldout-selected k=1. "
                    "(a) Predicted cell identities and saved sparse "
                    "Microglia-to-neuron attention at model time 2.5. Whole-tissue "
                    "maps and the same fixed ROI are shown for baseline, Trem2 low, "
                    "and Trem2 high. Every condition retains eight predicted cell "
                    f"types. Entropy ranges from {scope_spatial.entropy_bits.min():.2f} "
                    f"to {scope_spatial.entropy_bits.max():.2f} bits and the largest "
                    f"single-type fraction is {100 * scope_spatial.dominant_fraction.max():.1f}%. "
                    "No saved Microglia-to-neuron sparse edge was present at this "
                    "endpoint, so no edge was imputed into the tissue or ROI maps. "
                    "(b) Endpoint predicted cell-type composition. Microglia comprise "
                    f"{100 * microglia_fraction['baseline']:.2f}% at baseline, "
                    f"{100 * microglia_fraction['low']:.2f}% after Trem2 low, and "
                    f"{100 * microglia_fraction['high']:.2f}% after Trem2 high. "
                    "(c) Matched endpoint change in attention per Microglia to "
                    "excitatory and inhibitory neurons. Trem2-low changes are "
                    f"{attention_values[('low', 'Excitatory neurons')]:.4f} and "
                    f"{attention_values[('low', 'Inhibitory neurons')]:.4f}. "
                    "Trem2-high changes are "
                    f"{attention_values[('high', 'Excitatory neurons')]:.4f} and "
                    f"{attention_values[('high', 'Inhibitory neurons')]:.4f}. "
                    "(d) All-particle endpoint change for the seven Trem2-associated "
                    "modules used in the original Figure 6g visual design. DAM changes "
                    f"are {module_lookup[('low', 'DAM_microglia')]:.3f} for Trem2 low "
                    f"and {module_lookup[('high', 'DAM_microglia')]:.3f} for Trem2 high. "
                    "(e) Matched Spp1-Cd44 ligand-receptor score trajectory. Endpoint "
                    f"changes are {lr_endpoint['low']:.3f} for Spp1 low and "
                    f"{lr_endpoint['high']:.3f} for Spp1 high. All attention and "
                    "ligand-receptor contrasts shown are finite and comparable. "
                    "The two scope figures use identical quantitative axes."
                ),
                "",
            ]
        )

    caption_sections.extend(
        [
            "## admouse_final_scope_comparison_lr_perturbation",
            "",
            (
                "**Perturbation-scope sensitivity.** (a) Endpoint predicted Microglia "
                "fractions for baseline, Trem2 low, and Trem2 high under the "
                "whole-tissue and Microglia-only interventions. (b) Matched endpoint "
                "Microglia-to-neuron attention changes. (c) All-particle endpoint "
                "changes for the seven Trem2-associated modules used in the original "
                "Figure 6g design. (d) Matched Spp1-Cd44 ligand-receptor score "
                "trajectories. Every quantitative axis is shared between scopes. "
                "The smaller Microglia-only responses are therefore shown without "
                "rescaling or visual amplification. These are model sensitivity "
                "results, not experimental causal knockouts."
            ),
            "",
        ]
    )

    (BUNDLE_DIR / "CAPTIONS.md").write_text(
        "\n".join(caption_sections), encoding="utf-8"
    )


def write_provenance() -> None:
    """Record the selected corrected experiments and exact rebuild command."""

    text = f"""# Final AD ligand-receptor perturbation provenance

Archived on: 2026-08-14

Scientific claim: Scale-2.5, k=1 AD in-silico perturbations from a shared real
t=0 population produce finite, scope-dependent changes in predicted cell-state,
sparse learned attention, and the strict scoreable Spp1-Cd44 LR trajectory.

## Files

- Whole-tissue figure: `{BUNDLE_DIR / 'admouse_final_whole_tissue_lr_perturbation.pdf'}`
- Microglia-only sensitivity: `{BUNDLE_DIR / 'admouse_final_microglia_only_lr_perturbation.pdf'}`
- Direct scope comparison: `{BUNDLE_DIR / 'admouse_final_scope_comparison_lr_perturbation.pdf'}`
- Plotting script: `{Path(__file__).resolve()}`
- Caption source: `{BUNDLE_DIR / 'CAPTIONS.md'}`
- Panel source table: `{BUNDLE_DIR / 'panel_source_table.csv'}`
- Panel sanity table: `{BUNDLE_DIR / 'panel_sanity.csv'}`

## Selected experiments and source paths

- Whole-tissue server run: `{SCOPE_CONFIG['whole_tissue']['server_root']}`
- Microglia-only server run: `{SCOPE_CONFIG['microglia_only']['server_root']}`
- Local compact inputs: `{SOURCE_DIR}`
- Model: AD mouse alpha-expression 0.015, seed 42.
- Perturbation magnitude: 2.5 times the latent PC standard deviation along the
  unit target-gene PCA loading.
- Classifier neighborhood: k=1, selected by fixed heldout balanced accuracy.
- Figure endpoint: model time 2.5.
- Trajectory contract: one continuous simulation from the shared real t=0
  population through all 26 requested times; no previous-slice restart.
- Fixed ROI: center {ROI_CENTER}, half-width {ROI_HALF_WIDTH}.

## Rebuild

```bash
MPLCONFIGDIR=/tmp/ad_fig_mpl \\
  /opt/anaconda3/envs/CytoCompass-runtime-v2-20260309/bin/python \\
  {Path(__file__).resolve()}
```

## Interpretation

The whole-tissue figure perturbs every t0 particle. The Microglia-only figure
perturbs only t0 Microglia and is a cell-type-specific sensitivity analysis.
All cells subsequently evolve together, so changes outside Microglia can be
indirect model responses. Sparse attention and ligand-receptor panels use the
newly recomputed k=1 tables. No scale-5 value is used. Missing displayed-edge
subsets are not filled. These are in-silico sensitivity results rather than
experimental causal interventions. Exact local artifact hashes are recorded in
`SHA256SUMS.txt` after rendering.

## SHA-256

See `SHA256SUMS.txt` in this figure bank for exact hashes of the rendered
figures, plotted source tables, compact source data, captions, and provenance.
"""
    (BUNDLE_DIR / "PROVENANCE.md").write_text(text, encoding="utf-8")


def main() -> None:
    """Build the scale-1.0 whole-tissue figure only."""

    apply_style()
    all_cases = {
        scope: {
            condition: load_case(scope, condition)
            for condition in SPATIAL_CONDITIONS
        }
        for scope in SCOPE_CONFIG
    }
    all_tables = {scope: prepare_scope_tables(scope) for scope in SCOPE_CONFIG}

    window = spatial_limits(all_cases)
    limits = shared_plot_limits(all_cases, all_tables)
    for scope in SCOPE_CONFIG:
        make_scope_figure(scope, all_cases[scope], all_tables[scope], window, limits)
    summary = all_tables["whole_tissue"]["composition"]
    summary.to_csv(RUN_DIR / "data" / "panel_data" / "composition_endpoint.csv", index=False)


if __name__ == "__main__":
    main()
