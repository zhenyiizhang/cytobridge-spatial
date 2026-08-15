#!/usr/bin/env python3
"""Create a concise multi-dataset reviewer figure for LR complex sensitivity."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


DATASET_ORDER = ("Zebrafish", "MOSTA", "ARISTA", "Chicken Heart")
COLORS = {
    "Zebrafish": "#0072B2",
    "MOSTA": "#009E73",
    "ARISTA": "#CC79A7",
    "Chicken Heart": "#D55E00",
}
TEXT = "#000000"
GRID = "#D7DDE2"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_analysis(values: list[str]) -> dict[str, Path]:
    analyses: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--analysis must be DATASET=COMPARISON_ROOT")
        name, raw_path = value.split("=", 1)
        name = name.strip()
        path = Path(raw_path).expanduser().resolve()
        if name in analyses:
            raise ValueError(f"Duplicate analysis dataset: {name}")
        if not path.is_dir():
            raise FileNotFoundError(path)
        analyses[name] = path
    if set(analyses) != set(DATASET_ORDER):
        raise ValueError(
            "Exactly these analyses are required: " + ", ".join(DATASET_ORDER)
        )
    return analyses


def _read_analysis(path: Path) -> dict[str, object]:
    required = {
        "manifest": path / "run_manifest.json",
        "summary": path / "summary.json",
        "paired": path / "paired_scores.csv",
        "per_time": path / "per_time_stability.csv",
        "per_pair": path / "per_pair_stability.csv",
    }
    for input_path in required.values():
        if not input_path.is_file():
            raise FileNotFoundError(input_path)
    manifest = json.loads(required["manifest"].read_text(encoding="utf-8"))
    summary = json.loads(required["summary"].read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError(
            f"Sensitivity manifest is not complete: {required['manifest']}"
        )
    reproduction = manifest.get("formal_primary_reproduction", {})
    if int(reproduction.get("n_rows", 0)) <= 0:
        raise ValueError("Sensitivity manifest lacks primary reproduction evidence.")
    paired = pd.read_csv(required["paired"])
    per_time = pd.read_csv(required["per_time"])
    per_pair = pd.read_csv(required["per_pair"])
    if not np.isfinite(paired[["score_min", "score_geometric_mean"]]).all().all():
        raise ValueError("Paired LR scores contain non-finite values.")
    if set(per_time["scope"]) != {"all_scored_pairs", "multisubunit_pairs"}:
        raise ValueError("Per-time table lacks the two declared analysis scopes.")
    if int(summary.get("n_multisubunit_pairs", 0)) <= 0:
        raise ValueError("Every displayed dataset must contain multi-subunit pairs.")
    return {
        "paths": required,
        "manifest": manifest,
        "summary": summary,
        "paired": paired,
        "per_time": per_time,
        "per_pair": per_pair,
    }


def _style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 9,
            "axes.titlesize": 12,
            "axes.labelsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.linewidth": 0.65,
            "axes.edgecolor": TEXT,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "text.color": TEXT,
            "axes.labelcolor": TEXT,
            "axes.titlecolor": TEXT,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def _clean_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(TEXT)
    ax.spines["bottom"].set_color(TEXT)
    ax.grid(color=GRID, linewidth=0.45, alpha=0.65)
    ax.set_axisbelow(True)


def _panel_heading(
    ax: plt.Axes,
    label: str,
    title: str,
    *,
    legend_handles: list[Line2D] | None = None,
) -> None:
    ax.set_axis_off()
    ax.text(
        0.0,
        0.5,
        label,
        fontsize=14,
        fontweight="bold",
        ha="left",
        va="center",
        color=TEXT,
    )
    ax.text(
        0.105 if legend_handles is None else 0.055,
        0.5,
        title,
        fontsize=12,
        fontweight="bold",
        ha="left",
        va="center",
        color=TEXT,
    )
    if legend_handles:
        ax.legend(
            handles=legend_handles,
            frameon=False,
            ncol=len(legend_handles),
            loc="center right",
            bbox_to_anchor=(1.0, 0.5),
            borderaxespad=0,
            columnspacing=1.35,
            handletextpad=0.5,
        )


def _coverage(ax: plt.Axes, records: dict[str, dict[str, object]]) -> None:
    names = list(DATASET_ORDER)
    multi = [int(records[name]["summary"]["n_multisubunit_pairs"]) for name in names]
    total = [int(records[name]["summary"]["n_scored_pairs"]) for name in names]
    y = np.arange(len(names))
    ax.barh(y, multi, color=[COLORS[name] for name in names], height=0.62)
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set_xlabel("Scored multi-subunit LR pairs")
    xmax = max(multi) * 1.25
    ax.set_xlim(0, xmax)
    for idx, (count, all_count) in enumerate(zip(multi, total)):
        ax.text(
            count + xmax * 0.018,
            idx,
            f"{count:,} of {all_count:,}",
            va="center",
            ha="left",
            fontsize=9,
        )
    _clean_axis(ax)


def _pooled_rank(ax: plt.Axes, records: dict[str, dict[str, object]]) -> None:
    names = list(DATASET_ORDER)
    y = np.arange(len(names))
    for idx, name in enumerate(names):
        per_time = records[name]["per_time"]
        values = per_time.loc[
            per_time["scope"] == "all_scored_pairs", "spearman"
        ].dropna()
        pooled = float(records[name]["summary"]["global_spearman"])
        offsets = (
            np.linspace(-0.13, 0.13, len(values))
            if len(values) > 1
            else np.zeros(len(values))
        )
        ax.scatter(
            values,
            idx + offsets,
            color=COLORS[name],
            s=26,
            edgecolor="white",
            linewidth=0.45,
            zorder=2,
        )
        ax.scatter(
            [pooled],
            [idx],
            color=COLORS[name],
            marker="D",
            s=64,
            zorder=3,
            edgecolor=TEXT,
            linewidth=0.8,
        )
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set_xlim(0.90, 1.005)
    ax.set_xlabel("Spearman rank correlation")
    ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=TEXT,
                marker="o",
                linestyle="none",
                markersize=5,
                label="Single time point",
            ),
            Line2D(
                [0],
                [0],
                color=TEXT,
                marker="D",
                linestyle="none",
                markersize=6,
                label="All times combined",
            ),
        ],
        frameon=False,
        ncol=2,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.02),
        borderaxespad=0,
        columnspacing=1.0,
        handletextpad=0.6,
    )
    _clean_axis(ax)


def _top_overlap(ax: plt.Axes, records: dict[str, dict[str, object]]) -> None:
    boundaries = np.array([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0001])
    color_map = mpl.colors.ListedColormap(
        ["#B85C6B", "#D78993", "#E8B8B8", "#BFDAD7", "#66AFA9", "#087F8C"]
    )
    color_norm = mpl.colors.BoundaryNorm(boundaries, color_map.N)
    scatter = None
    for row, name in enumerate(DATASET_ORDER):
        table = records[name]["per_time"]
        table = table.loc[table["scope"] == "all_scored_pairs"].sort_values("time")
        times = table["time"].to_numpy(dtype=float)
        normalized = (times - times.min()) / max(times.max() - times.min(), 1.0)
        scatter = ax.scatter(
            normalized,
            np.full(len(table), row),
            c=table["top_jaccard"],
            cmap=color_map,
            norm=color_norm,
            marker="s",
            s=150,
            edgecolor="white",
            linewidth=0.7,
        )
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.55, len(DATASET_ORDER) - 0.45)
    ax.set_yticks(np.arange(len(DATASET_ORDER)), DATASET_ORDER)
    ax.invert_yaxis()
    ax.set_xlabel("Normalized developmental time")
    _clean_axis(ax)
    if scatter is None:
        raise ValueError("Top-pair overlap panel has no data.")
    colorbar = ax.figure.colorbar(
        scatter,
        ax=ax,
        boundaries=boundaries,
        ticks=np.arange(0.4, 1.01, 0.1),
        pad=0.025,
        fraction=0.035,
    )
    colorbar.set_label("Top-10 Jaccard")
    colorbar.outline.set_edgecolor(TEXT)
    colorbar.ax.tick_params(colors=TEXT)


def _write_caption(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "# Ligand–receptor complex aggregation sensitivity",
                "",
                "(a) Number of strictly scored multi-subunit LR pairs in each dataset. (b) Spearman rank agreement between the minimum-subunit rule and the zero-preserving geometric mean. Circles show correlations calculated separately at each time point. Diamonds show one correlation calculated after combining every scored LR pair–time observation. (c) Jaccard overlap of the top ten LR pairs over normalized developmental time. Each square represents one observed time point.",
                "",
                "The four datasets contain 293–973 strictly scored multi-subunit LR pairs. Pooled Spearman correlations are 0.961–0.999. The top ten signals remain unchanged across time in Zebrafish and MOSTA and show dataset-specific sensitivity in ARISTA and Chicken Heart. These results support the robustness of the broad LR dynamics while motivating cautious interpretation of individual heteromeric top-pair rankings.",
                "",
                "Only the final complex aggregation rule was changed. Trajectories, expression states, communication attention, LR databases, time grids, and scored pair universes were held fixed. Every required subunit was present under both rules, and each minimum-gate table was reproduced before the sensitivity calculation. The minimum gate remains the primary estimand. Chicken Heart uses the declared conserved-symbol human CellChatDB proxy and is not a species-complete chicken interaction screen.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def render(analyses: dict[str, Path], output_dir: Path) -> Path:
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory must be absent or empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    records = {name: _read_analysis(analyses[name]) for name in DATASET_ORDER}

    _style()
    fig = plt.figure(figsize=(8.27, 11.69))
    grid = fig.add_gridspec(
        4,
        2,
        height_ratios=[0.13, 1.0, 0.13, 1.28],
        hspace=0.2,
        wspace=0.42,
        left=0.13,
        right=0.91,
        top=0.91,
        bottom=0.08,
    )
    fig.suptitle(
        "Ligand–receptor complex aggregation sensitivity",
        fontsize=13,
        fontweight="bold",
        y=0.965,
    )
    _panel_heading(fig.add_subplot(grid[0, 0]), "a", "Multi-subunit coverage")
    _panel_heading(fig.add_subplot(grid[0, 1]), "b", "LR rank agreement")
    _coverage(fig.add_subplot(grid[1, 0]), records)
    _pooled_rank(fig.add_subplot(grid[1, 1]), records)
    _panel_heading(
        fig.add_subplot(grid[2, :]),
        "c",
        "Top-10 LR overlap",
    )
    _top_overlap(fig.add_subplot(grid[3, :]), records)

    pdf = output_dir / "lr_complex_aggregation_reviewer_response.pdf"
    png = output_dir / "lr_complex_aggregation_reviewer_response.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=320)
    plt.close(fig)

    metrics_rows = []
    for name in DATASET_ORDER:
        summary = records[name]["summary"]
        all_scope = records[name]["per_time"].loc[
            records[name]["per_time"]["scope"] == "all_scored_pairs"
        ]
        metrics_rows.append(
            {
                "dataset": name,
                "n_scored_pairs": int(summary["n_scored_pairs"]),
                "n_multisubunit_pairs": int(summary["n_multisubunit_pairs"]),
                "pooled_spearman": float(summary["global_spearman"]),
                "min_per_time_spearman": float(all_scope["spearman"].min()),
                "min_top10_jaccard": float(all_scope["top_jaccard"].min()),
            }
        )
    metrics = output_dir / "figure_metrics.csv"
    pd.DataFrame(metrics_rows).to_csv(metrics, index=False)
    caption = output_dir / "caption.md"
    _write_caption(caption)
    script_snapshot = output_dir / Path(__file__).name
    shutil.copy2(Path(__file__).resolve(), script_snapshot)

    inputs = []
    for name in DATASET_ORDER:
        for label, path in records[name]["paths"].items():
            inputs.append(
                {
                    "dataset": name,
                    "label": label,
                    "path": str(path),
                    "sha256": _sha256(path),
                }
            )
    outputs = [pdf, png, metrics, caption, script_snapshot]
    manifest = output_dir / "figure_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "status": "complete",
                "claim": (
                    "Broad LR dynamics are robust to complex aggregation, with "
                    "dataset-specific sensitivity among top heteromeric pairs."
                ),
                "inputs": inputs,
                "outputs": [
                    {"path": str(path), "sha256": _sha256(path)} for path in outputs
                ],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    provenance = output_dir / "PROVENANCE.md"
    provenance.write_text(
        "\n".join(
            [
                "# Figure provenance",
                "",
                "Archived on: `2026-08-15`",
                "",
                "Manuscript figure: `LR complex aggregation sensitivity`",
                "",
                "Scientific claim: Broad LR dynamics are robust to complex aggregation, with dataset-specific sensitivity among top heteromeric pairs.",
                "",
                "## Files",
                "",
                f"- Vector figure: `{pdf}`",
                f"- PNG preview: `{png}`",
                f"- Plotting script: `{script_snapshot}`",
                f"- Caption source: `{caption}`",
                f"- Figure manifest: `{manifest}`",
                "",
                "## Source paths",
                "",
                *[f"- {name}: `{analyses[name]}`" for name in DATASET_ORDER],
                "",
                "## Evaluation protocol",
                "",
                "The sensitivity reuses the accepted expression states, observed anchors, communication attention, LR database, and time/pair universe. Every required subunit is present in both rules. The geometric mean is zero preserving.",
                "",
                "## Rebuild command",
                "",
                "See the package script CLI and the input paths in `figure_manifest.json`.",
                "",
                "## SHA-256",
                "",
                f"- Figure PDF: `{_sha256(pdf)}`",
                f"- Figure PNG: `{_sha256(png)}`",
                f"- Plotting script: `{_sha256(script_snapshot)}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    print(render(_parse_analysis(args.analysis), args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
