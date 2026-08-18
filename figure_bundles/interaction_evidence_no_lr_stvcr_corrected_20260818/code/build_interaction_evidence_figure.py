#!/usr/bin/env python3
"""Build the corrected combined no-LR and final stVCR evidence figure."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import cytobridge_figure_style as cb_style


DATASET_ORDER_NO_LR = [
    "zebrafish",
    "mosta",
    "arista",
    "admouse",
    "chicken_heart",
]
DATASET_ORDER_STVCR = [
    "zebrafish",
    "mosta",
    "arista",
    "admouse",
    "chicken_heart",
]
DATASET_LABELS = {
    "zebrafish": "Zebrafish",
    "mosta": "MOSTA",
    "arista": "Arista",
    "admouse": "AD mouse",
    "chicken_heart": "Chicken\nheart",
}
SPACE_ORDER = ["joint", "spatial", "state"]
SPACE_LABELS = {"joint": "Joint", "spatial": "Spatial", "state": "Gene state"}

FULL_COLOR = "#07838B"
NO_LR_COLOR = "#CC6677"
STVCR_COLOR = "#59616A"
SPACE_COLORS = {"joint": "#4C78A8", "spatial": "#E39D2D", "state": "#8F63A8"}
GRID_COLOR = "#D7DDE2"
TEXT_COLOR = "#000000"
FINAL_LOTO_SHA256 = "43f3acfa5d508b8d72f0cc02a03c121df58cf2fb2137c54e677f217dbdc4c038"
EXPECTED_TARGETS = {
    "zebrafish": {1, 2, 3},
    "mosta": {1, 2},
    "arista": {1, 2, 3},
    "admouse": {1},
    "chicken_heart": {1, 2},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sem(values: pd.Series) -> float:
    clean = values.dropna().astype(float)
    if len(clean) <= 1:
        return 0.0
    return float(clean.std(ddof=1) / np.sqrt(len(clean)))


def load_no_lr(source_dir: Path) -> pd.DataFrame:
    path = source_dir / "formal_no_lr_paired_target_deltas.csv"
    table = pd.read_csv(path)
    table = table.loc[table["metric"].eq("sliced_w2")].copy()
    expected = {
        "dataset",
        "target",
        "space",
        "full",
        "no_lr_prior",
        "no_lr_prior_relative_to_full",
    }
    missing = expected.difference(table.columns)
    if missing:
        raise ValueError(f"no-LR table is missing columns: {sorted(missing)}")
    if set(table["dataset"]) != set(DATASET_ORDER_NO_LR[:-1]):
        raise ValueError("formal no-LR table does not contain the exact four datasets")
    if set(table["space"]) != set(SPACE_ORDER):
        raise ValueError(
            "no-LR table does not contain joint, spatial, and state spaces"
        )
    chicken_dir = source_dir / "chicken_heart_no_lr"
    chicken_parts = []
    metric_keys = [
        "target",
        "space",
        "projection_repeat",
        "projection_seed",
        "projection_sha256",
    ]
    for arm, filename in (
        ("full", "full_data_metrics_long.csv"),
        ("no_lr_prior", "no_lr_prior_metrics_long.csv"),
    ):
        metric_path = chicken_dir / filename
        metrics = pd.read_csv(metric_path)
        needed = set(metric_keys) | {"sliced_w2"}
        missing = needed.difference(metrics.columns)
        if missing:
            raise ValueError(
                f"{metric_path.name} is missing columns: {sorted(missing)}"
            )
        if set(metrics["target"]) != {1, 2, 3}:
            raise ValueError(f"{metric_path.name} does not contain targets 1, 2, 3")
        if set(metrics["space"]) != set(SPACE_ORDER):
            raise ValueError(f"{metric_path.name} has an unexpected evaluation space")
        if set(metrics["projection_repeat"]) != set(range(5)):
            raise ValueError(f"{metric_path.name} does not contain five repeats")
        if len(metrics) != 45 or metrics.duplicated(metric_keys).any():
            raise ValueError(f"{metric_path.name} is not an exact 45-row score table")
        metrics = (
            metrics.groupby(["target", "space"], as_index=False)["sliced_w2"]
            .mean()
            .rename(columns={"sliced_w2": arm})
        )
        chicken_parts.append(metrics)

    chicken = chicken_parts[0].merge(
        chicken_parts[1], on=["target", "space"], validate="one_to_one"
    )
    chicken.insert(0, "dataset", "chicken_heart")
    chicken["metric"] = "sliced_w2"
    chicken["no_lr_prior_minus_full"] = chicken["no_lr_prior"] - chicken["full"]
    chicken["no_lr_prior_relative_to_full"] = (
        chicken["no_lr_prior_minus_full"] / chicken["full"]
    )
    table = pd.concat([table, chicken], ignore_index=True, sort=False)

    if set(table["dataset"]) != set(DATASET_ORDER_NO_LR):
        raise ValueError("no-LR evidence does not contain all five datasets")
    duplicates = table.duplicated(["dataset", "target", "space"])
    if duplicates.any():
        raise ValueError("no-LR table contains duplicate dataset-target-space rows")
    return table


def load_stvcr(source_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = source_dir / "loto_target_stage_means.csv"
    if sha256(path) != FINAL_LOTO_SHA256:
        raise ValueError(
            "LOTO input is not the frozen final five-dataset table; "
            "the archived 2026-08-15 target summaries must not be reused"
        )
    table = pd.read_csv(path)
    required = {
        "dataset",
        "target",
        "method",
        "space",
        "sliced_w2",
        "n_projection_repeats",
    }
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
    if set(table["dataset"]) != set(DATASET_ORDER_STVCR):
        raise ValueError("final LOTO table does not contain the exact five datasets")
    if not table["n_projection_repeats"].eq(5).all():
        raise ValueError("final LOTO table must contain five projection repeats")
    if not np.isfinite(table["sliced_w2"]).all() or not table["sliced_w2"].gt(0).all():
        raise ValueError("final LOTO sliced-W2 values must be positive and finite")
    for dataset, targets in EXPECTED_TARGETS.items():
        observed = set(table.loc[table["dataset"].eq(dataset), "target"].astype(int))
        if observed != targets:
            raise ValueError(
                f"{dataset}: expected targets {sorted(targets)}, got {sorted(observed)}"
            )

    long = table.loc[
        table["method"].isin(["CytoBridge-0.015", "stvcr"]),
        ["dataset", "target", "method", "space", "sliced_w2"],
    ].copy()
    long["method"] = long["method"].replace({"stvcr": "stVCR"})
    if set(long["dataset"]) != set(DATASET_ORDER_STVCR):
        raise ValueError(
            "stVCR table does not contain the exact five benchmark datasets"
        )
    if set(long["space"]) != set(SPACE_ORDER):
        raise ValueError(
            "stVCR table does not contain joint, spatial, and state spaces"
        )
    duplicates = long.duplicated(["dataset", "target", "space", "method"])
    if duplicates.any():
        raise ValueError(
            "stVCR table contains duplicate dataset-target-space-method rows"
        )

    paired = (
        long.pivot(
            index=["dataset", "target", "space"],
            columns="method",
            values="sliced_w2",
        )
        .reset_index()
        .rename_axis(columns=None)
    )
    if paired[["CytoBridge-0.015", "stVCR"]].isna().any().any():
        raise ValueError("stVCR comparison is not paired for every target and space")
    if len(paired) != 33:
        raise ValueError(f"expected 33 paired LOTO cells, got {len(paired)}")
    if (paired["CytoBridge-0.015"] == paired["stVCR"]).any():
        raise ValueError("unexpected exact CytoBridge/stVCR tie")
    wins = int((paired["CytoBridge-0.015"] < paired["stVCR"]).sum())
    if wins != 25:
        raise ValueError(
            f"expected the current formal 25/33 CytoBridge wins, got {wins}/33; "
            "27/33 identifies the superseded archived table"
        )
    paired["stvcr_minus_cytobridge"] = paired["stVCR"] - paired["CytoBridge-0.015"]
    paired["stvcr_relative_to_cytobridge"] = (
        paired["stvcr_minus_cytobridge"] / paired["CytoBridge-0.015"]
    )
    return long, paired


def summarize_effects(no_lr: pd.DataFrame, stvcr: pd.DataFrame) -> pd.DataFrame:
    no_lr_summary = (
        no_lr.groupby(["dataset", "space"], sort=False)["no_lr_prior_relative_to_full"]
        .agg(n_targets="size", mean_relative_difference="mean", sem=sem)
        .reset_index()
    )
    no_lr_summary.insert(0, "comparison", "No LR prior minus full model")

    stvcr_summary = (
        stvcr.groupby(["dataset", "space"], sort=False)["stvcr_relative_to_cytobridge"]
        .agg(n_targets="size", mean_relative_difference="mean", sem=sem)
        .reset_index()
    )
    stvcr_summary.insert(0, "comparison", "stVCR minus CytoBridge")
    return pd.concat([no_lr_summary, stvcr_summary], ignore_index=True)


def style_axis(ax: mpl.axes.Axes, *, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    ax.grid(axis=grid_axis, color=GRID_COLOR, linewidth=0.5, alpha=0.75, zorder=0)
    ax.tick_params(colors=TEXT_COLOR)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)


def plot_normalized_aggregate(
    ax: mpl.axes.Axes,
    table: pd.DataFrame,
    datasets: list[str],
    *,
    value_col: str,
    baseline_label: str,
    comparison_label: str,
    baseline_color: str,
    comparison_color: str,
) -> None:
    x_base = np.arange(len(datasets), dtype=float)
    width = 0.34
    relative = np.array(
        [
            float(table.loc[table["dataset"].eq(dataset), value_col].mean())
            for dataset in datasets
        ]
    )
    baseline = np.ones(len(datasets), dtype=float)
    comparison = 1.0 + relative
    ax.bar(
        x_base - width / 2,
        baseline,
        width=width,
        color=baseline_color,
        edgecolor="white",
        linewidth=0.55,
        label=baseline_label,
        zorder=2,
    )
    ax.bar(
        x_base + width / 2,
        comparison,
        width=width,
        color=comparison_color,
        edgecolor="white",
        linewidth=0.55,
        label=comparison_label,
        zorder=2,
    )
    ax.set_xticks(x_base)
    ax.set_xticklabels([DATASET_LABELS[item] for item in datasets])
    ax.set_ylabel("Relative reconstruction error\n(baseline model = 1)")
    ax.set_ylim(0, max(comparison) * 1.08)
    ax.margins(x=0.04)
    style_axis(ax)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.0, 1.02),
        frameon=False,
        ncol=2,
        handlelength=1.2,
        columnspacing=1.1,
        borderaxespad=0,
    )


def plot_relative_effects(
    ax: mpl.axes.Axes,
    table: pd.DataFrame,
    datasets: list[str],
    *,
    value_col: str,
    reference_label: str,
) -> None:
    x_base = np.arange(len(datasets), dtype=float)
    offsets = {"joint": -0.23, "spatial": 0.0, "state": 0.23}
    bar_width = 0.19
    for space in SPACE_ORDER:
        means = []
        errors = []
        for dataset in datasets:
            values = table.loc[
                table["dataset"].eq(dataset) & table["space"].eq(space), value_col
            ].astype(float)
            means.append(float(values.mean() * 100.0))
            errors.append(float(sem(values) * 100.0))
        positions = x_base + offsets[space]
        ax.bar(
            positions,
            means,
            width=bar_width,
            color=SPACE_COLORS[space],
            alpha=0.88,
            edgecolor="white",
            linewidth=0.5,
            zorder=2,
            label=SPACE_LABELS[space],
        )
        ax.errorbar(
            positions,
            means,
            yerr=errors,
            fmt="none",
            ecolor=TEXT_COLOR,
            elinewidth=0.7,
            capsize=2.0,
            capthick=0.7,
            zorder=4,
        )
        for dataset_index, dataset in enumerate(datasets):
            values = table.loc[
                table["dataset"].eq(dataset) & table["space"].eq(space), value_col
            ].astype(float)
            if len(values) == 1:
                jitter = np.array([0.0])
            else:
                jitter = np.linspace(-0.035, 0.035, len(values))
            ax.scatter(
                positions[dataset_index] + jitter,
                values.to_numpy() * 100.0,
                s=13,
                facecolor="white",
                edgecolor=TEXT_COLOR,
                linewidth=0.55,
                zorder=5,
            )

    ax.axhline(0, color=TEXT_COLOR, linewidth=0.7, zorder=1)
    ax.set_xticks(x_base)
    ax.set_xticklabels([DATASET_LABELS[item] for item in datasets])
    ax.set_ylabel(f"Error change vs {reference_label} (%)")
    ax.text(
        0.99,
        0.98,
        f"Above 0 favors {reference_label}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.2,
        color=TEXT_COLOR,
    )
    ax.margins(x=0.03)
    style_axis(ax)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.0, 1.02),
        frameon=False,
        ncol=3,
        handlelength=1.2,
        columnspacing=0.9,
        borderaxespad=0,
    )


def add_heading(fig: plt.Figure, ax: mpl.axes.Axes, label: str, title: str) -> None:
    cb_style.panel_heading(ax, label, title, title_x=0.060, y=0.54)


def make_figure(no_lr: pd.DataFrame, stvcr: pd.DataFrame) -> plt.Figure:
    cb_style.apply_style()
    cb_style.HEADING_COLOR = TEXT_COLOR
    mpl.rcParams.update(
        {
            "text.color": TEXT_COLOR,
            "axes.labelcolor": TEXT_COLOR,
            "axes.titlecolor": TEXT_COLOR,
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
        }
    )

    fig = plt.figure(figsize=cb_style.A4_PORTRAIT)
    outer = fig.add_gridspec(
        nrows=2,
        ncols=2,
        left=0.09,
        right=0.97,
        bottom=0.07,
        top=0.975,
        wspace=0.24,
        hspace=0.25,
    )

    panels = []
    for spec in outer:
        inner = spec.subgridspec(
            nrows=2, ncols=1, height_ratios=[0.12, 0.88], hspace=0.03
        )
        panels.append((fig.add_subplot(inner[0]), fig.add_subplot(inner[1])))

    heading_a, ax_a = panels[0]
    add_heading(fig, heading_a, "a", "LR-prior ablation")
    plot_normalized_aggregate(
        ax_a,
        no_lr,
        DATASET_ORDER_NO_LR,
        value_col="no_lr_prior_relative_to_full",
        baseline_label="Full model",
        comparison_label="No LR prior",
        baseline_color=FULL_COLOR,
        comparison_color=NO_LR_COLOR,
    )

    heading_b, ax_b = panels[1]
    add_heading(fig, heading_b, "b", "No-LR effect by evaluation space")
    plot_relative_effects(
        ax_b,
        no_lr,
        DATASET_ORDER_NO_LR,
        value_col="no_lr_prior_relative_to_full",
        reference_label="Full model",
    )

    heading_c, ax_c = panels[2]
    add_heading(fig, heading_c, "c", "External interaction-free baseline")
    plot_normalized_aggregate(
        ax_c,
        stvcr,
        DATASET_ORDER_STVCR,
        value_col="stvcr_relative_to_cytobridge",
        baseline_label="CytoBridge",
        comparison_label="stVCR",
        baseline_color=FULL_COLOR,
        comparison_color=STVCR_COLOR,
    )

    heading_d, ax_d = panels[3]
    add_heading(fig, heading_d, "d", "stVCR comparison by evaluation space")
    plot_relative_effects(
        ax_d,
        stvcr,
        DATASET_ORDER_STVCR,
        value_col="stvcr_relative_to_cytobridge",
        reference_label="CytoBridge",
    )

    return fig


def write_caption(path: Path) -> None:
    text = """**Ligand–receptor priors and interaction-aware modeling contribute to spatiotemporal reconstruction.** (a) Dataset-level reconstruction error for independently trained full and no-LR-prior CytoBridge models, normalized to the full model. For every matched target and evaluation space, the no-LR Sliced-W2 is divided by the corresponding full-model value; these paired ratios are then averaged with equal weight within each dataset. (b) The same no-LR effect resolved by evaluation space. (c) Dataset-level held-out reconstruction error for CytoBridge and stVCR, normalized to CytoBridge. The stVCR-to-CytoBridge Sliced-W2 ratio is calculated within each matched dataset, held-out target, and evaluation space and then averaged with equal weight within each dataset. In the current final five-dataset LOTO results, CytoBridge has lower Sliced-W2 in 25 of 33 paired cells (joint 8/11, spatial 9/11, and state 8/11; no ties). stVCR models spatial and gene-state dynamics without an explicit cell-cell interaction term. (d) The stVCR comparison resolved by evaluation space. Positive differences indicate higher error for the comparison method and therefore favor the reference model named on the axis. In (b,d), bars show the mean across target stages, error bars show the s.e.m., and white points show individual targets. Target-stage variation is not biological replication. The no-LR comparison is a matched retraining ablation. The stVCR comparison is an external method comparison and is not interpreted as a single-component causal ablation.\n"""
    path.write_text(text, encoding="utf-8")


def write_provenance(
    path: Path,
    root: Path,
    script_path: Path,
    pdf_path: Path,
    png_path: Path,
    source_files: list[Path],
) -> None:
    def archived_path(item: Path) -> Path:
        return item.resolve().relative_to(root.resolve())

    source_lines = "\n".join(
        f"- `{archived_path(item)}` — `{sha256(item)}`" for item in source_files
    )
    archived_script = archived_path(script_path)
    archived_pdf = archived_path(pdf_path)
    archived_png = archived_path(png_path)
    text = f"""# Figure provenance

Archived on: `2026-08-18`

Scientific claim: LR-informed graph construction contributes to the matched CytoBridge reconstruction, while comparison with stVCR provides complementary evidence for the value of explicit interaction-aware modeling.

## Files

- Vector figure: `{archived_pdf}`
- PNG preview: `{archived_png}`
- Plotting script: `{archived_script}`
- Caption source: `figure/caption.md`

## Selected experiments

- Matched no-LR evidence: `/data/cytobridge/projects/CytoBridge-ST-1104/runs/corrected-matched-ablation-benchmark-20260813-3a380c5-r1/report`
- Chicken-heart matched no-LR evidence: `/data/cytobridge/projects/CytoBridge-ST-1104/runs/chicken-heart-matched-no-lr-20260815-4db603e-r1`
- stVCR evidence: current final unified held-out LOTO target-stage table for Zebrafish, MOSTA, ARISTA, AD mouse, and chicken heart.

## Source paths

- Archived panel inputs: `source_data/`
- Formal no-LR server report: `/data/cytobridge/projects/CytoBridge-ST-1104/runs/corrected-matched-ablation-benchmark-20260813-3a380c5-r1/report`
- Local stVCR benchmark table: `source_data/loto_target_stage_means.csv`. This single table replaces the superseded five archived dataset-specific summaries.

## Panel sources

| Panel | Content | Calculation |
|---|---|---|
| a | Full model versus no LR prior | Dataset mean of target- and space-specific relative sliced-W2 differences, displayed with full model = 1. |
| b | LR-prior effect by space | Mean and s.e.m. of `100 * (no-LR - full) / full` across targets. |
| c | CytoBridge versus stVCR | Dataset mean of target- and space-specific relative sliced-W2 differences, displayed with CytoBridge = 1. |
| d | External-baseline effect by space | Mean and s.e.m. of `100 * (stVCR - CytoBridge) / CytoBridge` across targets. |

## Evaluation protocol

- Primary metric: sliced W2. Lower values indicate better reconstruction.
- no-LR scope: full-data in-sample benchmark, 16 target stages across five datasets.
- stVCR scope: held-out LOTO benchmark, 11 target stages across five datasets.
- Final stVCR comparison: CytoBridge lower in 25/33 paired dataset-target-space cells; no ties.
- LOTO projection protocol: 1,024 directions per repeat and five shared repeats.
- Projection uncertainty is not used as biological replication. Error bars summarize variation across target stages.

## Source-file SHA-256

{source_lines}

## Rebuild command

```bash
python {archived_script}
```

## Interpretation

The no-LR comparison is a matched retraining ablation. stVCR is an external baseline without an explicit cell-cell interaction term, not a component-wise CytoBridge ablation. The two evidence classes are therefore displayed separately and are not pooled into a single effect estimate.

## SHA-256

- Figure PDF: `{sha256(pdf_path)}`
- Figure PNG: `{sha256(png_path)}`
- Plotting script: `{sha256(script_path)}`
"""
    path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=default_root)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    source_dir = root / "source_data"
    figure_dir = root / "figure"
    figure_dir.mkdir(parents=True, exist_ok=True)

    no_lr = load_no_lr(source_dir)
    stvcr_long, stvcr_paired = load_stvcr(source_dir)
    summary = summarize_effects(no_lr, stvcr_paired)

    stvcr_long.to_csv(source_dir / "stvcr_target_metrics_long.csv", index=False)
    stvcr_paired.to_csv(source_dir / "stvcr_paired_target_deltas.csv", index=False)
    summary.to_csv(source_dir / "panel_summary.csv", index=False)

    fig = make_figure(no_lr, stvcr_paired)
    pdf_path = figure_dir / "interaction_evidence_no_lr_stvcr_corrected_final.pdf"
    png_path = figure_dir / "interaction_evidence_no_lr_stvcr_corrected_final.png"
    cb_style.save_figure(fig, pdf_path, png_path, dpi=320)
    plt.close(fig)

    caption_path = figure_dir / "caption.md"
    write_caption(caption_path)

    source_files = [
        source_dir / "formal_no_lr_paired_target_deltas.csv",
        source_dir / "chicken_heart_no_lr" / "full_data_metrics_long.csv",
        source_dir / "chicken_heart_no_lr" / "no_lr_prior_metrics_long.csv",
        source_dir / "loto_target_stage_means.csv",
    ]
    write_provenance(
        root / "PROVENANCE.md",
        root,
        Path(__file__).resolve(),
        pdf_path,
        png_path,
        source_files,
    )


if __name__ == "__main__":
    main()
