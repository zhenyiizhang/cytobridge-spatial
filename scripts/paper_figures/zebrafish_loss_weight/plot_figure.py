#!/usr/bin/env python3
"""Draw zebrafish loss-weight sensitivity from the evaluation tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DISPLAY = {
    "reference_alpha": (r"$\alpha_{expr}=0.015$", "#1F4E79", None),
    "alpha_expr_005": (r"$\alpha_{expr}=0.05$", "#D55E00", "///"),
    "ot_mass_10_to_1": (r"$\lambda_{OT}:\lambda_{mass}=10:1$", "#E69F00", "\\\\"),
    "reference_ratio": (r"$\lambda_{OT}:\lambda_{mass}=1:1$", "#1F4E79", None),
    "ot_mass_1_to_10": (r"$\lambda_{OT}:\lambda_{mass}=1:10$", "#009E73", ".."),
}
SPACES = (("joint", "Joint state"), ("pca", "Expression state"), ("spatial", "Physical space"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--alpha-metrics",
        required=True,
        type=Path,
        help="CSV with columns model, time, space, and w1 for the two expression weights",
    )
    parser.add_argument(
        "--evaluation-root",
        required=True,
        type=Path,
        help="directory containing reference, ot_mass_10_to_1, and ot_mass_1_to_10 evaluations",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def load_results(alpha_metrics: Path, evaluation_root: Path) -> pd.DataFrame:
    alpha = pd.read_csv(alpha_metrics)
    required = {"model", "time", "space", "w1"}
    if not required.issubset(alpha.columns):
        raise ValueError(f"{alpha_metrics} must contain {sorted(required)}")
    alpha = alpha.loc[:, ["time", "space", "w1", "model"]].rename(
        columns={"model": "condition"}
    )
    alpha["condition"] = alpha["condition"].replace(
        {"alpha_express_0015": "reference_alpha"}
    )

    frames = [alpha]
    for folder, condition in (
        ("reference", "reference_ratio"),
        ("ot_mass_10_to_1", "ot_mass_10_to_1"),
        ("ot_mass_1_to_10", "ot_mass_1_to_10"),
    ):
        path = evaluation_root / folder / "distribution_metrics.csv"
        frame = pd.read_csv(path)
        if not {"time", "space", "w1"}.issubset(frame.columns):
            raise ValueError(f"{path} must contain time, space, and w1")
        frame = frame.loc[:, ["time", "space", "w1"]].copy()
        frame["condition"] = condition
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    if not np.isfinite(result["w1"]).all() or (result["w1"] < 0).any():
        raise ValueError("W1 values must be finite and non-negative")
    return result


def style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def draw(frame: pd.DataFrame, output_dir: Path) -> tuple[Path, Path]:
    style()
    fig, axes = plt.subplots(2, 3, figsize=(11.69, 8.27), sharex=True)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.09, top=0.88, hspace=0.48, wspace=0.30)
    rows = (
        ("Expression-loss weight", ("reference_alpha", "alpha_expr_005")),
        ("OT-to-mass loss ratio", ("ot_mass_10_to_1", "reference_ratio", "ot_mass_1_to_10")),
    )
    for column, (space, title) in enumerate(SPACES):
        space_values = frame.loc[frame["space"].eq(space), "w1"]
        ymax = float(space_values.max()) * 1.15
        for row_index, (row_title, conditions) in enumerate(rows):
            ax = axes[row_index, column]
            times = sorted(frame.loc[frame["space"].eq(space), "time"].astype(float).unique())
            x = np.arange(len(times), dtype=float)
            width = 0.72 / len(conditions)
            for index, condition in enumerate(conditions):
                part = frame.loc[
                    frame["space"].eq(space) & frame["condition"].eq(condition)
                ].sort_values("time")
                if list(part["time"].astype(float)) != times:
                    raise ValueError(f"Missing time point for {condition} in {space}")
                label, color, hatch = DISPLAY[condition]
                offset = (index - (len(conditions) - 1) / 2) * width
                ax.bar(
                    x + offset,
                    part["w1"],
                    width=width,
                    label=label,
                    color=color,
                    edgecolor="black",
                    linewidth=0.5,
                    hatch=hatch,
                )
            ax.set_title(title)
            ax.set_ylim(0, ymax)
            ax.set_xticks(x, [f"{value:g}" for value in times])
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.75)
            if column == 0:
                ax.set_ylabel(f"{row_title}\nWasserstein-1")
            if row_index == 1:
                ax.set_xlabel("Time")
            if column == 2:
                ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf = output_dir / "zebrafish_loss_weight_sensitivity.pdf"
    png = output_dir / "zebrafish_loss_weight_sensitivity.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return pdf, png


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    frame = load_results(
        args.alpha_metrics.expanduser().resolve(strict=True),
        args.evaluation_root.expanduser().resolve(strict=True),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "loss_weight_metrics.csv", index=False)
    summary = frame.groupby(["condition", "space"], sort=False)["w1"].mean().reset_index(name="mean_w1")
    summary.to_csv(output_dir / "summary_statistics.csv", index=False)
    for path in draw(frame, output_dir):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
