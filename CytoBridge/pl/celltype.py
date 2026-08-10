"""Cell-type composition visualizations."""

from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = ["plot_celltype_composition"]


def plot_celltype_composition(
    summary,
    *,
    out_path: str | Path,
    label_to_color: dict[str, str] | None = None,
    celltype_order: list[str] | tuple[str, ...] | None = None,
    time_labels: dict | None = None,
    title: str | None = None,
    show_legend: bool = True,
):
    """Plot a stacked fraction bar chart from label-composition summary rows."""
    import matplotlib.pyplot as plt
    import pandas as pd

    table = pd.DataFrame(summary).copy()
    required = {"time_index", "time", "celltype", "fraction"}
    missing = sorted(required.difference(table.columns))
    if missing:
        raise KeyError(f"composition summary is missing columns: {missing}")
    if table.empty:
        raise ValueError("composition summary must be non-empty.")

    time_table = (
        table[["time_index", "time"]]
        .drop_duplicates()
        .sort_values("time_index", kind="stable")
    )
    time_indices = time_table["time_index"].to_list()
    observed_types = table["celltype"].astype(str).unique().tolist()
    if celltype_order is None:
        preferred = list(label_to_color or {})
        celltypes = [name for name in preferred if name in observed_types]
        celltypes += sorted(set(observed_types).difference(celltypes))
    else:
        celltypes = [str(name) for name in celltype_order if str(name) in observed_types]
        celltypes += sorted(set(observed_types).difference(celltypes))

    pivot = (
        table.pivot_table(
            index="time_index",
            columns="celltype",
            values="fraction",
            aggfunc="sum",
            fill_value=0.0,
        )
        .reindex(index=time_indices, columns=celltypes, fill_value=0.0)
    )
    row_sums = pivot.sum(axis=1).to_numpy(dtype=float)
    if not np.allclose(row_sums, 1.0, atol=1e-6):
        raise ValueError("Composition fractions must sum to one at every timepoint.")

    x = np.arange(len(time_indices))
    bottom = np.zeros(len(time_indices), dtype=float)
    palette = label_to_color or {}
    fallback = plt.get_cmap("tab20")
    with plt.rc_context(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "text.color": "black",
            "axes.labelcolor": "black",
            "axes.edgecolor": "black",
        }
    ):
        fig, ax = plt.subplots(figsize=(8.2, 4.8), dpi=300)
        for index, celltype in enumerate(celltypes):
            values = pivot[celltype].to_numpy(dtype=float)
            ax.bar(
                x,
                values,
                bottom=bottom,
                width=0.82,
                color=palette.get(celltype, fallback(index % fallback.N)),
                label=celltype,
                linewidth=0,
            )
            bottom += values
        tick_labels = []
        for value in time_table["time"].to_list():
            tick_labels.append(str(time_labels.get(value, value) if time_labels else value))
        ax.set_xticks(x, tick_labels)
        ax.set_xlabel("Time")
        ax.set_ylabel("Cell-type fraction")
        ax.set_ylim(0.0, 1.0)
        ax.grid(False)
        if title:
            ax.set_title(title)
        if show_legend:
            ax.legend(
                bbox_to_anchor=(1.02, 1.0),
                loc="upper left",
                frameon=False,
                fontsize=7,
            )
        fig.tight_layout()
        path = Path(out_path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    return path
