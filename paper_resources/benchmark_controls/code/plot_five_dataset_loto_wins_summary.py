#!/usr/bin/env python3
"""Draw the five-dataset leave-one-time-point-out benchmark summary."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl  # noqa: E402
import matplotlib.lines as mlines  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from cytobridge_figure_style import (  # noqa: E402
    A4_PORTRAIT,
    CYTOBRIDGE_COLOR,
    GRID_COLOR,
    PLOT_TEXT_SIZE,
    apply_style,
    clean_axis,
    panel_heading,
    save_figure,
)


HERE = Path(__file__).resolve().parent
BUNDLE_ROOT = HERE.parent
DEFAULT_INPUT = BUNDLE_ROOT / "source_data/loto_target_stage_means.csv"
DEFAULT_LINEAR_OT_ROOT = BUNDLE_ROOT / "source_data/linear_ot_metrics"
DEFAULT_OUTPUT = BUNDLE_ROOT / "figure/five_dataset_loto_wins_summary.pdf"
DEFAULT_SPATRACK = BUNDLE_ROOT / "source_data/spatrack_metrics.csv"

DATASETS = ("zebrafish", "mosta", "arista", "admouse", "chicken_heart")
DATASET_LABELS = {
    "zebrafish": "Zebrafish",
    "mosta": "MOSTA",
    "arista": "ARISTA",
    "admouse": "AD mouse",
    "chicken_heart": "Chicken heart",
}
METHODS = (
    "CytoBridge-0.015",
    "exact_ot_displacement",
    "stvcr",
    "stories",
    "mioflow",
    "moscot",
    "paste",
    "spateo",
    "spatrack_pca",
    "wot",
    "random_independent_pairs",
)
DISPLAY = {
    "CytoBridge-0.015": "CytoBridge",
    "exact_ot_displacement": "Linear OT displacement",
    "stvcr": "stVCR",
    "stories": "STORIES",
    "mioflow": "MIOFlow",
    "moscot": "MOSCOT",
    "paste": "PASTE",
    "spateo": "Spateo",
    "wot": "Waddington-OT",
    "random_independent_pairs": "Random interpolation",
}
DISPLAY["spatrack_pca"] = "SpaTrack"
CONTROL_METHODS = {"exact_ot_displacement", "random_independent_pairs"}
MODEL_METHODS = tuple(method for method in METHODS if method not in CONTROL_METHODS)
BASE_SPACES = ("joint", "spatial", "state")
DISPLAY_SPACES = ("overall", *BASE_SPACES)
SPACE_LABELS = {
    "overall": "Overall",
    "joint": "Joint",
    "spatial": "Spatial",
    "state": "State",
}
OTHER_COLOR = "#7C8790"
SUMMARY_COLOR = "#B7C0C7"
BLACK = "#000000"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_targets(path: Path, linear_ot_root: Path) -> pd.DataFrame:
    targets = pd.read_csv(path)
    required = {"dataset", "target", "method", "space", "sliced_w2"}
    missing = sorted(required.difference(targets.columns))
    if missing:
        raise ValueError(f"missing columns in {path}: {missing}")
    if set(targets["dataset"]) != set(DATASETS):
        raise ValueError("the input does not contain the five accepted datasets")
    accepted_methods = set(METHODS).difference({"exact_ot_displacement", "spatrack_pca"})
    if set(targets["method"]) != accepted_methods:
        raise ValueError("the input does not contain the accepted nine-method set")

    linear_ot_frames: list[pd.DataFrame] = []
    for dataset in DATASETS:
        control_path = linear_ot_root / dataset / "loto_metrics_long.csv"
        control = pd.read_csv(control_path)
        required_control = {"target", "method", "space", "sliced_w2", "projection_repeat"}
        missing_control = sorted(required_control.difference(control.columns))
        if missing_control:
            raise ValueError(f"missing columns in {control_path}: {missing_control}")
        control["method"] = "exact_ot_displacement"
        repeats = control.groupby(["target", "space"])["projection_repeat"].nunique()
        if not repeats.eq(5).all():
            raise ValueError(f"{dataset}: Linear OT does not have five projection repeats")
        target_means = (
            control.groupby(["target", "method", "space"], as_index=False)
            .agg(
                sliced_w2=("sliced_w2", "mean"),
                projection_sd=("sliced_w2", "std"),
                n_projection_repeats=("projection_repeat", "nunique"),
            )
        )
        target_means.insert(0, "dataset", dataset)
        linear_ot_frames.append(target_means)
    targets = pd.concat([targets, *linear_ot_frames], ignore_index=True, sort=False)
    targets["display_name"] = targets["method"].map(DISPLAY)
    if targets["display_name"].isna().any():
        raise ValueError("unrecognized method in the combined benchmark")
    counts = targets.groupby(["dataset", "target", "space"]).size()
    if counts.size != 33:
        raise ValueError(f"expected 33 held-out comparisons, found {counts.size}")
    if not np.isfinite(targets["sliced_w2"]).all():
        raise ValueError("Sliced-W2 values must be finite")
    return targets


def rank_targets(targets: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ranked = targets.copy()
    ranked["rank"] = ranked.groupby(["dataset", "target", "space"])[
        "sliced_w2"
    ].rank(method="min", ascending=True)
    win_counts = (
        ranked.loc[ranked["rank"].eq(1)]
        .groupby("method")
        .size()
        .reindex(METHODS, fill_value=0)
        .rename("win_count")
        .reset_index()
    )
    return ranked, win_counts


def relative_values(targets: pd.DataFrame) -> pd.DataFrame:
    reference = targets.loc[
        targets["method"].eq("CytoBridge-0.015"),
        ["dataset", "target", "space", "sliced_w2"],
    ].rename(columns={"sliced_w2": "cytobridge_sliced_w2"})
    if reference.duplicated(["dataset", "target", "space"]).any():
        raise ValueError("duplicate CytoBridge reference row")
    paired = targets.loc[targets["method"].isin((*MODEL_METHODS, "spatrack_pca"))].merge(
        reference,
        on=["dataset", "target", "space"],
        how="left",
        validate="many_to_one",
    )
    if paired["cytobridge_sliced_w2"].isna().any():
        raise ValueError("incomplete CytoBridge pairing")
    paired["relative_difference_pct"] = 100.0 * (
        paired["sliced_w2"] / paired["cytobridge_sliced_w2"] - 1.0
    )
    return paired


def plot_dataset_space(
    axis: plt.Axes,
    relative: pd.DataFrame,
    dataset: str,
    space: str,
    *,
    show_method_labels: bool,
) -> None:
    panel = relative.loc[relative["dataset"].eq(dataset)].copy()
    plotted_values: list[float] = [0.0]

    panel_methods = MODEL_METHODS
    for y, method in enumerate(panel_methods):
        method_rows = panel.loc[panel["method"].eq(method)].copy()
        if space == "overall":
            by_target = method_rows.pivot(
                index="target",
                columns="space",
                values="relative_difference_pct",
            )
            if all(name in by_target.columns for name in BASE_SPACES) and not (
                by_target.loc[:, list(BASE_SPACES)].isna().any().any()
            ):
                values = by_target.loc[:, list(BASE_SPACES)].mean(axis=1).to_numpy(float)
            else:
                values = np.asarray([], dtype=float)
        else:
            values = method_rows.loc[
                method_rows["space"].eq(space), "relative_difference_pct"
            ].to_numpy(float)
        if len(values) == 0:
            continue

        mean = float(np.mean(values))
        sem = (
            float(np.std(values, ddof=1) / np.sqrt(len(values)))
            if len(values) > 1
            else 0.0
        )
        is_cytobridge = method == "CytoBridge-0.015"
        color = CYTOBRIDGE_COLOR if is_cytobridge else OTHER_COLOR
        axis.errorbar(
            mean,
            y,
            xerr=sem,
            fmt="D",
            markersize=4.8 if is_cytobridge else 3.9,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.5,
            color=color,
            ecolor=BLACK if not is_cytobridge else color,
            elinewidth=0.9,
            capsize=2.0,
            capthick=0.8,
            alpha=1.0 if is_cytobridge else 0.70,
            zorder=3,
        )
        offsets = np.linspace(-0.13, 0.13, len(values)) if len(values) > 1 else [0.0]
        axis.scatter(
            values,
            y + np.asarray(offsets),
            marker="o",
            s=15,
            facecolor="white",
            edgecolor=color if is_cytobridge else BLACK,
            linewidth=0.55,
            zorder=4,
        )
        plotted_values.extend(values.tolist())
        plotted_values.extend([mean - sem, mean + sem])

    lower = min(plotted_values)
    upper = max(plotted_values)
    span = max(upper - lower, 4.0)
    axis.set_xlim(lower - 0.10 * span, upper + 0.10 * span)
    axis.set_ylim(-0.55, len(panel_methods) - 0.45)
    axis.invert_yaxis()
    axis.set_yticks(np.arange(len(panel_methods)))
    if show_method_labels:
        axis.set_yticklabels([DISPLAY[method] for method in panel_methods])
        for label, method in zip(axis.get_yticklabels(), panel_methods):
            label.set_color(CYTOBRIDGE_COLOR if method == "CytoBridge-0.015" else BLACK)
            label.set_fontweight("bold" if method == "CytoBridge-0.015" else "normal")
    else:
        axis.set_yticklabels([])
    axis.tick_params(axis="y", length=0, pad=3.5)
    axis.tick_params(axis="x", labelsize=PLOT_TEXT_SIZE, colors=BLACK)
    clean_axis(axis, grid=False)
    axis.spines["left"].set_color(BLACK)
    axis.spines["bottom"].set_color(BLACK)
    axis.grid(axis="x", color=GRID_COLOR, linewidth=0.45, alpha=0.55, zorder=0)
    axis.set_facecolor("white")
    axis.set_title(
        SPACE_LABELS[space],
        pad=2.5,
        color=BLACK,
        fontweight="bold" if space == "overall" else "normal",
    )


def draw_figure(
    targets: pd.DataFrame,
    win_counts: pd.DataFrame,
    output_pdf: Path,
) -> tuple[Path, Path]:
    apply_style()
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "text.color": BLACK,
            "axes.labelcolor": BLACK,
            "axes.titlecolor": BLACK,
            "xtick.color": BLACK,
            "ytick.color": BLACK,
        }
    )
    relative = relative_values(targets)
    figure = plt.figure(figsize=A4_PORTRAIT)
    outer = figure.add_gridspec(
        6,
        1,
        left=0.245,
        right=0.975,
        top=0.925,
        bottom=0.06,
        height_ratios=(0.92, 1.0, 1.0, 1.0, 1.0, 1.0),
        hspace=0.29,
    )

    bar_block = outer[0, 0].subgridspec(2, 1, height_ratios=(0.20, 1.0), hspace=0.10)
    bar_heading = figure.add_subplot(bar_block[0, 0])
    panel_heading(bar_heading, "a", "Benchmark results across datasets")
    for item in bar_heading.texts:
        item.set_color(BLACK)
    bar_axis = figure.add_subplot(bar_block[1, 0])
    win_lookup = win_counts.set_index("method")["win_count"].to_dict()
    bar_methods = sorted(
        (
            method
            for method in METHODS
            if int(win_lookup.get(method, 0)) > 0 or method == "random_independent_pairs"
        ),
        key=lambda method: (-int(win_lookup.get(method, 0)), METHODS.index(method)),
    )
    y_positions = np.arange(len(bar_methods))
    totals = np.asarray([int(win_lookup.get(method, 0)) for method in bar_methods])
    for y_position, method, total in zip(y_positions, bar_methods, totals):
        is_cytobridge = method == "CytoBridge-0.015"
        is_control = method in CONTROL_METHODS
        bar_axis.barh(
            y_position,
            total,
            height=0.62,
            color=CYTOBRIDGE_COLOR if is_cytobridge else ("white" if is_control else SUMMARY_COLOR),
            edgecolor=OTHER_COLOR if is_control else "white",
            hatch="////" if method == "exact_ot_displacement" else None,
            linewidth=0.9 if is_control else 0.6,
            zorder=3,
        )
        bar_axis.text(
            float(total) + 0.35,
            y_position,
            str(int(total)),
            ha="left",
            va="center",
            fontsize=PLOT_TEXT_SIZE,
            color=BLACK,
        )
    bar_axis.set_yticks(y_positions, [DISPLAY[method] for method in bar_methods])
    for label, method in zip(bar_axis.get_yticklabels(), bar_methods):
        label.set_color(CYTOBRIDGE_COLOR if method == "CytoBridge-0.015" else BLACK)
        label.set_fontweight("bold" if method == "CytoBridge-0.015" else "normal")
    bar_axis.invert_yaxis()
    bar_axis.set_xlabel("Number of benchmark settings")
    x_max = max(22.0, float(totals.max()) + 1.8)
    bar_axis.set_xlim(0, x_max)
    bar_axis.set_xticks((0, 5, 10, 15, 20))
    bar_axis.tick_params(axis="y", length=0, pad=4.0)
    clean_axis(bar_axis, grid=False)
    bar_axis.spines["left"].set_color(BLACK)
    bar_axis.spines["bottom"].set_color(BLACK)
    bar_axis.grid(axis="x", color=GRID_COLOR, linewidth=0.45, alpha=0.55, zorder=0)

    for row, dataset in enumerate(DATASETS):
        block = outer[row + 1, 0].subgridspec(
            2,
            4,
            height_ratios=(0.19, 1.0),
            width_ratios=(1.08, 1.0, 1.0, 1.0),
            hspace=0.20,
            wspace=0.30,
        )
        heading = figure.add_subplot(block[0, :])
        panel_heading(heading, chr(ord("b") + row), DATASET_LABELS[dataset])
        for item in heading.texts:
            item.set_color(BLACK)
        axes = [figure.add_subplot(block[1, column]) for column in range(4)]
        for column, (axis, space) in enumerate(zip(axes, DISPLAY_SPACES)):
            plot_dataset_space(
                axis,
                relative,
                dataset,
                space,
                show_method_labels=column == 0,
            )

    figure.text(
        0.61,
        0.021,
        "Relative error vs CytoBridge (%)",
        ha="center",
        va="center",
        fontsize=PLOT_TEXT_SIZE,
        color=BLACK,
    )
    handles = [
        mlines.Line2D(
            [],
            [],
            marker="o",
            markerfacecolor="white",
            markeredgecolor=BLACK,
            color="none",
            linestyle="none",
            markersize=4.5,
            label="Held-out time point",
        ),
        mlines.Line2D(
            [],
            [],
            marker="D",
            markerfacecolor=BLACK,
            markeredgecolor="white",
            color=BLACK,
            linestyle="-",
            linewidth=0.9,
            markersize=4.5,
            label="Mean ± SEM",
        ),
    ]
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.61, 0.986),
        ncol=2,
        frameon=False,
        fontsize=9.0,
        handletextpad=0.45,
        columnspacing=1.0,
    )
    figure.text(
        0.975,
        0.948,
        "Positive values indicate lower CytoBridge error",
        ha="right",
        va="center",
        fontsize=8.5,
        color=BLACK,
    )

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    output_png = output_pdf.with_suffix(".png")
    save_figure(figure, output_pdf, output_png, dpi=320)
    plt.close(figure)
    return output_pdf, output_png


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--spatrack-metrics", type=Path, default=DEFAULT_SPATRACK)
    parser.add_argument("--linear-ot-root", type=Path, default=DEFAULT_LINEAR_OT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--table-output-dir",
        type=Path,
        default=BUNDLE_ROOT / "source_data",
        help="Directory for the recalculated ranking and relative-error tables.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=BUNDLE_ROOT / "manifests/figure_manifest.json",
        help="Path for the run manifest.",
    )
    args = parser.parse_args()

    table_output_dir = args.table_output_dir.resolve()
    table_output_dir.mkdir(parents=True, exist_ok=True)
    targets = load_targets(args.input.resolve(), args.linear_ot_root.resolve())
    targets.to_csv(
        table_output_dir / "loto_target_stage_means_with_linear_ot.csv",
        index=False,
    )
    # Add all five datasets before calculating ranks or relative errors.
    spatrack = pd.read_csv(args.spatrack_metrics)
    assert set(spatrack["dataset"]) == set(DATASETS)
    assert set(spatrack["space"]) == set(BASE_SPACES)
    assert spatrack.groupby(["dataset", "target", "space"])["projection_repeat"].nunique().eq(5).all()
    spatrack = spatrack.groupby(["dataset", "target", "space"], as_index=False).agg(sliced_w2=("sliced_w2", "mean"))
    assert len(spatrack) == 33
    spatrack["method"] = "spatrack_pca"
    spatrack["display_name"] = DISPLAY["spatrack_pca"]
    targets = pd.concat([targets, spatrack], ignore_index=True)
    ranked, win_counts = rank_targets(targets)
    targets.to_csv(table_output_dir / "loto_target_stage_means_with_spatrack.csv", index=False)
    relative = relative_values(targets)
    ranked.to_csv(table_output_dir / "loto_target_stage_ranks.csv", index=False)
    win_counts.to_csv(table_output_dir / "loto_win_counts.csv", index=False)
    relative.to_csv(
        table_output_dir / "loto_model_relative_to_cytobridge.csv",
        index=False,
    )
    pdf, png = draw_figure(targets, win_counts, args.output.resolve())

    cytobridge_wins = int(
        win_counts.loc[win_counts["method"].eq("CytoBridge-0.015"), "win_count"].iloc[0]
    )
    manifest = {
        "schema_version": "1.0.0",
        "status": "complete",
        "source": "refreshed five-dataset benchmark with accepted ARISTA and chicken-heart retraining results",
        "comparisons": 33,
        "spatrack_scope": "Five datasets, 33 target-space comparisons, all panels",
        "spatrack_input": str(args.spatrack_metrics.resolve()),
        "metric": "Sliced-W2; lower is better",
        "projection_repeats": 5,
        "cytobridge_first_place_count": cytobridge_wins,
        "linear_ot_first_place_count": int(
            win_counts.loc[
                win_counts["method"].eq("exact_ot_displacement"), "win_count"
            ].iloc[0]
        ),
        "input": {"path": str(args.input.resolve()), "sha256": sha256_file(args.input.resolve())},
        "linear_ot_inputs": {
            dataset: {
                "path": str(
                    args.linear_ot_root.resolve() / dataset / "loto_metrics_long.csv"
                ),
                "sha256": sha256_file(
                    args.linear_ot_root.resolve() / dataset / "loto_metrics_long.csv"
                ),
            }
            for dataset in DATASETS
        },
        "outputs": {
            "pdf": {"path": str(pdf), "sha256": sha256_file(pdf)},
            "png": {"path": str(png), "sha256": sha256_file(png)},
            "win_counts": {
                "path": str(table_output_dir / "loto_win_counts.csv"),
                "sha256": sha256_file(table_output_dir / "loto_win_counts.csv"),
            },
        },
    }
    manifest_path = args.manifest.resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
