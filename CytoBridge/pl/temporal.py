"""Dataset-agnostic plots for temporal gene and interaction programs."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

__all__ = [
    "plot_temporal_gene_heatmap",
    "plot_temporal_pattern_prototypes",
    "plot_temporal_profile_small_multiples",
]


def _output_path(value: str | Path) -> Path:
    path = Path(value).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


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
) -> Path:
    """Plot a gene-wise z-scored heatmap ordered by pattern and variance."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    required = {"gene", "variance", "cluster"}
    missing = sorted(required.difference(top_variable_genes.columns))
    if missing:
        raise KeyError(f"top_variable_genes is missing columns: {missing}")
    ordered = top_variable_genes.sort_values(
        ["cluster", "variance"], ascending=[True, False]
    )["gene"].astype(str)
    genes = [gene for gene in ordered if gene in expression.index][: int(top_n)]
    if not genes:
        raise ValueError("No selected genes are present in expression.")
    values = expression.loc[genes].astype(float)
    zscore = values.sub(values.mean(axis=1), axis=0).div(
        values.std(axis=1, ddof=0).replace(0.0, np.nan), axis=0
    ).fillna(0.0)
    fig, ax = plt.subplots(
        figsize=(7.6, max(4.5, 0.15 * len(genes))), facecolor="white"
    )
    sns.heatmap(
        zscore,
        cmap="RdBu_r",
        center=0.0,
        vmin=-2.0,
        vmax=2.0,
        ax=ax,
        cbar_kws={"label": "Gene-wise z-score", "shrink": 0.5},
    )
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Gene")
    fig.tight_layout()
    path = _output_path(out_path)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path
