"""Dataset-agnostic plots for temporal gene and interaction programs."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping, Optional, Sequence, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from CytoBridge.tl.downstream.temporal import DevelopmentalWaveResult

__all__ = [
    "plot_developmental_wave_heatmap",
    "plot_temporal_gene_heatmap",
    "plot_temporal_pattern_prototypes",
    "plot_temporal_profile_small_multiples",
]


def _output_path(value: str | Path) -> Path:
    path = Path(value).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def plot_developmental_wave_heatmap(
    result: "DevelopmentalWaveResult",
    *,
    out_path: Optional[str | Path] = None,
    ax=None,
    cmap: str = "RdBu_r",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    phase_colors: Optional[Mapping[int, object] | Sequence[object]] = None,
    show_colorbar: bool = True,
    title: Optional[str] = "Developmental gene wave",
):
    """Render a peak-ordered wave with explicit contiguous phase annotations.

    The numerical ordering and segmentation are taken verbatim from
    :func:`CytoBridge.tl.analyze_developmental_wave`; plotting never reclusters
    or reorders features. If ``out_path`` is supplied, the figure is saved and
    the resolved :class:`~pathlib.Path` is returned. Otherwise the Matplotlib
    axes is returned for caller-side composition.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    if not hasattr(result, "ordered_profiles") or not hasattr(
        result, "assignments"
    ):
        raise TypeError(
            "result must be a DevelopmentalWaveResult from "
            "analyze_developmental_wave."
        )
    profiles = pd.DataFrame(result.ordered_profiles).copy()
    assignments = pd.DataFrame(result.assignments).copy()
    if profiles.empty or profiles.shape[1] < 2:
        raise ValueError("result.ordered_profiles must be a non-empty time matrix.")
    required = {"profile", "wave_rank", "phase"}
    missing = sorted(required.difference(assignments.columns))
    if missing:
        raise KeyError(f"result.assignments is missing columns: {missing}")
    assignments = assignments.sort_values("wave_rank", kind="mergesort")
    expected_names = profiles.index.astype(str).tolist()
    if assignments["profile"].astype(str).tolist() != expected_names:
        raise ValueError(
            "assignments ordered by wave_rank must match ordered_profiles rows."
        )
    phases = assignments["phase"].to_numpy(dtype=int)
    if np.any(phases <= 0) or np.any(np.diff(phases) < 0):
        raise ValueError("assignments phases must be positive and contiguous in order.")
    values = profiles.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("result.ordered_profiles contains non-finite values.")
    if vmin is None and vmax is None:
        magnitude = float(np.quantile(np.abs(values), 0.99))
        magnitude = max(magnitude, np.finfo(float).eps)
        vmin, vmax = -magnitude, magnitude
    elif vmin is None:
        vmax = float(vmax)
        vmin = -abs(vmax)
    elif vmax is None:
        vmin = float(vmin)
        vmax = abs(vmin)
    else:
        vmin, vmax = float(vmin), float(vmax)
    if not np.isfinite([vmin, vmax]).all() or not float(vmin) < float(vmax):
        raise ValueError("vmin and vmax must be finite with vmin < vmax.")

    created_figure = ax is None
    if created_figure:
        height = max(4.5, min(12.0, 0.035 * len(profiles) + 2.5))
        _, ax = plt.subplots(figsize=(8.2, height), facecolor="white")
    figure = ax.figure
    image = ax.imshow(
        values,
        aspect="auto",
        interpolation="nearest",
        origin="upper",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    try:
        times = np.asarray(profiles.columns, dtype=float)
        time_labels = [f"{value:g}" for value in times]
    except (TypeError, ValueError):
        time_labels = list(map(str, profiles.columns))
    ax.set_xticks(np.arange(len(time_labels)))
    ax.set_xticklabels(time_labels, rotation=45, ha="right")
    ax.set_xlabel("Time")
    ax.set_ylabel("Peak-ordered profiles / phases")
    if title:
        ax.set_title(str(title))

    unique_phases = np.unique(phases)
    if phase_colors is None:
        palette = plt.get_cmap("Set2")
        color_lookup = {
            int(phase): palette(index % palette.N)
            for index, phase in enumerate(unique_phases)
        }
    elif isinstance(phase_colors, Mapping):
        missing_colors = [
            int(phase) for phase in unique_phases if int(phase) not in phase_colors
        ]
        if missing_colors:
            raise KeyError(f"phase_colors is missing phases: {missing_colors}")
        color_lookup = {
            int(phase): phase_colors[int(phase)] for phase in unique_phases
        }
    else:
        supplied = list(phase_colors)
        if len(supplied) < len(unique_phases):
            raise ValueError("phase_colors does not contain enough colors.")
        color_lookup = {
            int(phase): supplied[index]
            for index, phase in enumerate(unique_phases)
        }

    phase_centers: list[float] = []
    phase_labels: list[str] = []
    n_profiles = int(len(profiles))
    for phase in unique_phases:
        positions = np.flatnonzero(phases == phase)
        start, end = int(positions[0]), int(positions[-1]) + 1
        if not np.array_equal(positions, np.arange(start, end)):
            raise ValueError(f"Phase {int(phase)} is not a contiguous row segment.")
        if start:
            ax.axhline(start - 0.5, color="white", linewidth=1.1)
        phase_centers.append((start + end - 1) / 2.0)
        phase_labels.append(f"Phase {int(phase)} (n={end - start})")
        # A narrow strip outside the left spine preserves every heatmap column.
        ax.add_patch(
            Rectangle(
                (-0.045, 1.0 - end / n_profiles),
                0.025,
                (end - start) / n_profiles,
                transform=ax.transAxes,
                facecolor=color_lookup[int(phase)],
                edgecolor="none",
                clip_on=False,
            )
        )
    ax.set_yticks(phase_centers)
    ax.set_yticklabels(phase_labels)
    if bool(show_colorbar):
        colorbar = figure.colorbar(image, ax=ax, pad=0.02, fraction=0.035)
        colorbar.set_label("Row-standardized value")
    figure.tight_layout()

    if out_path is not None:
        path = _output_path(out_path)
        figure.savefig(path, bbox_inches="tight", facecolor="white")
        if created_figure:
            plt.close(figure)
        return path
    return ax


def plot_temporal_pattern_prototypes(
    prototypes: pd.DataFrame,
    *,
    out_path: str | Path,
    title: str = "Temporal pattern prototypes",
    y_label: str = "Mean normalized value",
) -> Path:
    """Plot mean +/- standard deviation for clustered temporal profiles."""
    import matplotlib.pyplot as plt

    required = {"cluster", "time", "mean", "std", "n_profiles"}
    missing = sorted(required.difference(prototypes.columns))
    if missing:
        raise KeyError(f"prototypes is missing columns: {missing}")
    fig, ax = plt.subplots(figsize=(7.2, 4.4), facecolor="white")
    palette = plt.get_cmap("Set2")
    for idx, (cluster, subset) in enumerate(
        prototypes.groupby("cluster", sort=True)
    ):
        subset = subset.sort_values("time")
        color = palette(idx % palette.N)
        time = subset["time"].to_numpy(dtype=float)
        mean = subset["mean"].to_numpy(dtype=float)
        std = subset["std"].to_numpy(dtype=float)
        n_profiles = int(subset["n_profiles"].iloc[0])
        ax.plot(
            time,
            mean,
            marker="o",
            linewidth=2.2,
            color=color,
            label=f"Pattern {int(cluster)} (n={n_profiles})",
        )
        ax.fill_between(time, mean - std, mean + std, color=color, alpha=0.16)
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel(y_label)
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False)
    fig.tight_layout()
    path = _output_path(out_path)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_temporal_profile_small_multiples(
    normalized_profiles: pd.DataFrame,
    assignments: pd.DataFrame,
    *,
    out_path: str | Path,
    title: str = "Temporal profiles",
    max_profiles: Optional[int] = None,
    columns: int = 5,
) -> Path:
    """Plot one normalized temporal trajectory per profile."""
    import matplotlib.pyplot as plt

    if not {"profile", "cluster"}.issubset(assignments.columns):
        raise KeyError("assignments must contain 'profile' and 'cluster'.")
    if int(columns) <= 0:
        raise ValueError("columns must be positive.")
    profiles = pd.DataFrame(normalized_profiles).copy()
    profiles.index = profiles.index.astype(str)
    assignment = assignments.set_index("profile")["cluster"]
    selected = [name for name in profiles.index.astype(str) if name in assignment]
    if max_profiles is not None:
        selected = selected[: int(max_profiles)]
    if not selected:
        raise ValueError("No assigned profiles are available to plot.")
    rows = int(math.ceil(len(selected) / int(columns)))
    fig, axes = plt.subplots(
        rows,
        int(columns),
        figsize=(3.1 * int(columns), 2.2 * rows),
        squeeze=False,
        facecolor="white",
    )
    palette = plt.get_cmap("tab10")
    for ax in axes.flat:
        ax.axis("off")
    time = np.asarray(profiles.columns, dtype=float)
    for ax, name in zip(axes.flat, selected):
        cluster = int(assignment.loc[name])
        values = profiles.loc[name].to_numpy(dtype=float)
        ax.axis("on")
        ax.plot(time, values, marker="o", linewidth=1.5, color=palette((cluster - 1) % 10))
        ax.set_title(str(name), fontsize=7.5)
        ax.tick_params(labelsize=7)
        ax.grid(axis="y", alpha=0.18)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    path = _output_path(out_path)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_temporal_gene_heatmap(
    expression: pd.DataFrame,
    top_variable_genes: pd.DataFrame,
    *,
    out_path: str | Path,
    top_n: int = 60,
    title: str = "Temporal gene programs",
    panel_columns: int = 1,
) -> Path:
    """Plot gene-wise z-scored profiles in one or more contiguous panels.

    ``panel_columns`` only changes the page layout.  Genes are globally ordered
    once (preferentially by ``dendrogram_rank``) and then split into contiguous,
    near-equal blocks that share the same color scale.  The default single-panel
    layout is backward compatible.
    """
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    import seaborn as sns

    panel_columns = int(panel_columns)
    if panel_columns <= 0:
        raise ValueError("panel_columns must be positive.")
    required = {"gene", "variance", "cluster"}
    missing = sorted(required.difference(top_variable_genes.columns))
    if missing:
        raise KeyError(f"top_variable_genes is missing columns: {missing}")
    if "dendrogram_rank" in top_variable_genes.columns:
        ordered_table = top_variable_genes.sort_values("dendrogram_rank")
    else:
        ordered_table = top_variable_genes.sort_values(
            ["cluster", "variance"], ascending=[True, False]
        )
    ordered = ordered_table["gene"].astype(str)
    genes = [gene for gene in ordered if gene in expression.index][: int(top_n)]
    if not genes:
        raise ValueError("No selected genes are present in expression.")
    values = expression.loc[genes].astype(float)
    zscore = values.sub(values.mean(axis=1), axis=0).div(
        values.std(axis=1, ddof=0).replace(0.0, np.nan), axis=0
    ).fillna(0.0)
    n_panels = min(panel_columns, len(genes))
    blocks = [
        zscore.iloc[index]
        for index in np.array_split(np.arange(len(zscore)), n_panels)
        if len(index)
    ]
    block_height = max(len(block) for block in blocks)
    figure_width = 7.6 if n_panels == 1 else 3.8 * n_panels + 0.8
    figure_height = max(4.5, 0.15 * block_height)
    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=(figure_width, figure_height),
        facecolor="white",
        squeeze=False,
        constrained_layout=True,
    )
    for panel_index, (ax, block) in enumerate(zip(axes.flat, blocks)):
        sns.heatmap(
            block,
            cmap="RdBu_r",
            center=0.0,
            vmin=-2.0,
            vmax=2.0,
            ax=ax,
            cbar=False,
            yticklabels=True,
        )
        ax.set_xlabel("Time")
        ax.set_ylabel("Gene" if panel_index == 0 else "")
        if n_panels > 1:
            ax.tick_params(axis="y", labelsize=4.5)
    colorbar = fig.colorbar(
        ScalarMappable(norm=Normalize(vmin=-2.0, vmax=2.0), cmap="RdBu_r"),
        ax=list(axes.flat),
        fraction=0.025,
        pad=0.02,
    )
    colorbar.set_label("Gene-wise z-score")
    fig.suptitle(title)
    path = _output_path(out_path)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path
