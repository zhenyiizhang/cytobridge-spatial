"""Draw S36 from the caller's per-time model-evaluation table."""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


CONDITIONS = {
    "formal_alpha_control": (r"$\alpha_{\mathrm{expr}}=0.015$", "#2166AC"),
    "alpha_expr_005": (r"$\alpha_{\mathrm{expr}}=0.05$", "#B2182B"),
    "ot_mass_10_to_1": (r"$\lambda_{OT}:\lambda_{mass}=10:1$", "#B2182B"),
    "formal": (r"$\lambda_{OT}:\lambda_{mass}=1:1$", "#2166AC"),
    "ot_mass_1_to_10": (r"$\lambda_{OT}:\lambda_{mass}=1:10$", "#8C8C8C"),
}
SPACES = (("joint", "Joint state"), ("pca", "Expression state"), ("spatial", "Physical space"))
STYLE = {
    "font.family": "Arial", "font.size": 9, "axes.titlesize": 10,
    "axes.labelsize": 9, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5, "axes.linewidth": .65, "axes.axisbelow": True,
    "xtick.major.width": .6, "ytick.major.width": .6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "text.color": "black", "axes.labelcolor": "black", "axes.titlecolor": "black",
    "xtick.color": "black", "ytick.color": "black",
    "pdf.fonttype": 42, "ps.fonttype": 42, "mathtext.fontset": "custom",
    "mathtext.rm": "Arial", "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
}


def validate_metrics(frame):
    """Require one W1 value for each setting, time 1–4, and output space."""
    table = frame.loc[:, ["condition", "time", "space", "w1"]].copy()
    table["condition"] = table.condition.replace({
        "reference_alpha": "formal_alpha_control", "reference_ratio": "formal",
    })
    for column in ("time", "w1"):
        table[column] = pd.to_numeric(table[column], errors="raise")
    if not np.isfinite(table[["time", "w1"]].to_numpy()).all() or (table.w1 < 0).any():
        raise ValueError("W1 values must be finite and non-negative.")
    expected = {(condition, float(time), space) for condition in CONDITIONS
                for time in (1, 2, 3, 4) for space, _ in SPACES}
    keys = ["condition", "time", "space"]
    if table.duplicated(keys).any() or set(table[keys].itertuples(index=False, name=None)) != expected:
        raise ValueError("S36 needs all five setting labels, times 1–4 and three output spaces exactly once.")
    return table


def make_figure(frame):
    """Return the current six-panel S36 layout without loading other results."""
    table = validate_metrics(frame)
    with mpl.rc_context(STYLE):
        fig, axes = plt.subplots(2, 3, figsize=(8.8, 6.4), sharex=True)
        fig.subplots_adjust(left=.075, right=.985, bottom=.09, top=.88, hspace=.48, wspace=.30)
        rows = (("formal_alpha_control", "alpha_expr_005"),
                ("ot_mass_10_to_1", "formal", "ot_mass_1_to_10"))
        for col, (space, title) in enumerate(SPACES):
            ymax = max(float(table.loc[table.space.eq(space), "w1"].max()) * 1.15, 1e-12)
            for row, conditions in enumerate(rows):
                ax = axes[row, col]
                width = .72 / len(conditions)
                for i, name in enumerate(conditions):
                    values = table.loc[table.condition.eq(name) & table.space.eq(space)].sort_values("time")
                    offset = (i - (len(conditions) - 1) / 2) * width
                    ax.bar(np.arange(4) + offset, values.w1, width=width*.76,
                           color=CONDITIONS[name][1], edgecolor="black", linewidth=.4, zorder=3)
                ax.set(ylim=(0, ymax), xticks=np.arange(4), xticklabels=["1", "2", "3", "4"])
                ax.set_title(title, pad=5, fontweight="normal")
                if row == 1:
                    ax.set_xlabel("Model time")
                if col == 0:
                    ax.set_ylabel("W1")
                ax.spines[["top", "right"]].set_visible(False)
                ax.grid(axis="y", color="#E7E7E7", linewidth=.45)
        for row, (label, title, y) in enumerate((
            ("a", r"Sensitivity to $\alpha_{\mathrm{expr}}$", .965),
            ("b", r"Sensitivity to $\lambda_{OT}:\lambda_{mass}$", .515),
        )):
            fig.text(.016, y, label, fontsize=14, fontweight="bold", va="top")
            fig.text(.048, y, title, fontsize=12, fontweight="bold", va="top")
            handles = [Patch(facecolor=CONDITIONS[name][1], edgecolor="black",
                             linewidth=.5, label=CONDITIONS[name][0]) for name in rows[row]]
            fig.legend(handles=handles, loc="upper right" if row == 0 else "center right",
                       bbox_to_anchor=(.985, .972 if row == 0 else .516),
                       ncol=len(handles), frameon=False)
    return fig


def draw(frame, output_dir):
    """Save the table, time averages, and PDF/PNG drawn from that same table."""
    table = validate_metrics(frame)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output / "loss_weight_metrics.csv", index=False)
    table.groupby(["condition", "space"], sort=False).w1.mean().reset_index(name="mean_w1").to_csv(
        output / "mean_w1.csv", index=False)
    with mpl.rc_context(STYLE):
        figure = make_figure(table)
        paths = tuple(output / f"zebrafish_s36_loss_weight_sensitivity.{ext}" for ext in ("pdf", "png"))
        for path in paths:
            figure.savefig(path, dpi=320, facecolor="white")
        plt.close(figure)
    return paths
