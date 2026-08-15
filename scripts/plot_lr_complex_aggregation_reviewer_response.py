#!/usr/bin/env python3
"""Create the complete multi-dataset reviewer-response figure for LR complexes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd


DATASET_ORDER = ("Zebrafish", "MOSTA", "ARISTA", "Chicken Heart")
COLORS = {
    "Zebrafish": "#0072B2",
    "MOSTA": "#009E73",
    "ARISTA": "#CC79A7",
    "Chicken Heart": "#D55E00",
    "AD mouse": "#59616A",
}
TEXT = "#24313A"
GRID = "#D7DDE2"
PRIMARY = "#07838B"
SENSITIVITY = "#E69F00"


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
    if tuple(name for name in DATASET_ORDER if name in analyses) != DATASET_ORDER:
        raise ValueError(
            "Exactly these analyses are required: " + ", ".join(DATASET_ORDER)
        )
    if set(analyses) != set(DATASET_ORDER):
        raise ValueError("Unexpected analysis dataset names.")
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
    if not manifest.get("formal_primary_reproduction"):
        raise ValueError("Sensitivity manifest lacks primary reproduction evidence.")
    paired = pd.read_csv(required["paired"])
    per_time = pd.read_csv(required["per_time"])
    per_pair = pd.read_csv(required["per_pair"])
    if not np.isfinite(paired[["score_min", "score_geometric_mean"]]).all().all():
        raise ValueError("Paired LR scores contain non-finite values.")
    if set(per_time["scope"]) != {"all_scored_pairs", "multisubunit_pairs"}:
        raise ValueError("Per-time table lacks the two declared analysis scopes.")
    return {
        "paths": required,
        "manifest": manifest,
        "summary": summary,
        "paired": paired,
        "per_time": per_time,
        "per_pair": per_pair,
    }


def _admouse_counts(pair_table: Path) -> tuple[int, int]:
    table = pd.read_csv(pair_table)
    required = {"ligand", "receptor"}
    if missing := sorted(required - set(table.columns)):
        raise ValueError(f"AD pair table is missing columns: {missing}")
    pairs = table[["ligand", "receptor"]].drop_duplicates()
    multi = pairs.loc[
        pairs["ligand"].astype(str).str.contains("_", regex=False)
        | pairs["receptor"].astype(str).str.contains("_", regex=False)
    ]
    return int(len(pairs)), int(len(multi))


def _style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.linewidth": 0.65,
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


def _panel_title(ax: plt.Axes, label: str, title: str) -> None:
    ax.text(
        0.0,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        va="bottom",
        ha="left",
    )
    ax.text(
        0.09,
        1.08,
        title,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def _clean_axis(ax: plt.Axes, *, grid: bool = True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(grid, color=GRID, linewidth=0.45, alpha=0.65)
    ax.set_axisbelow(True)


def _box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    *,
    edge: str,
    fill: str = "white",
    fontsize: float = 9,
    weight: str = "normal",
) -> None:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.015,rounding_size=0.015",
        linewidth=1.25,
        edgecolor=edge,
        facecolor=fill,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        color=TEXT,
    )


def _schematic(ax: plt.Axes) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    _panel_title(ax, "a", "Where the challenged assumption enters")

    _box(ax, (0.02, 0.35), 0.13, 0.28, "Ligand\nL = 5", edge="#59616A")
    _box(ax, (0.20, 0.57), 0.13, 0.22, "Subunit\nR1 = 10", edge="#59616A")
    _box(ax, (0.20, 0.21), 0.13, 0.22, "Subunit\nR2 = 2", edge="#59616A")
    ax.text(
        0.265,
        0.50,
        "Both required",
        ha="center",
        va="center",
        fontsize=9,
        fontweight="bold",
    )
    for start, end in (
        ((0.34, 0.68), (0.43, 0.68)),
        ((0.34, 0.32), (0.43, 0.32)),
    ):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="->", mutation_scale=10))

    _box(
        ax,
        (0.44, 0.55),
        0.22,
        0.28,
        "Primary AND gate\nmin(10, 2) = 2",
        edge=PRIMARY,
        fill="#E6F3F4",
        weight="bold",
    )
    _box(
        ax,
        (0.44, 0.17),
        0.22,
        0.28,
        "Sensitivity rule\n√(10 × 2) = 4.47",
        edge=SENSITIVITY,
        fill="#FFF4D6",
        weight="bold",
    )
    for y in (0.69, 0.31):
        ax.add_patch(
            FancyArrowPatch((0.67, y), (0.74, y), arrowstyle="->", mutation_scale=10)
        )
    _box(
        ax,
        (0.75, 0.33),
        0.23,
        0.34,
        "LR score\n= ligand expression\n× complex expression\n× communication attention",
        edge="#59616A",
    )
    ax.text(
        0.67,
        0.04,
        "Both rules require all subunits and return zero if any required subunit is zero.",
        ha="center",
        va="bottom",
        fontsize=9,
        color=TEXT,
    )


def _coverage(ax: plt.Axes, records: dict[str, dict[str, object]], ad_counts) -> None:
    _panel_title(ax, "b", "Multi-subunit evidence")
    names = list(DATASET_ORDER) + ["AD mouse"]
    multi = [
        int(records[name]["summary"]["n_multisubunit_pairs"]) for name in DATASET_ORDER
    ]
    total = [int(records[name]["summary"]["n_scored_pairs"]) for name in DATASET_ORDER]
    total.append(ad_counts[0])
    multi.append(ad_counts[1])
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
            fontsize=8.5,
        )
    _clean_axis(ax, grid=True)


def _rank_stability(ax: plt.Axes, records: dict[str, dict[str, object]]) -> None:
    _panel_title(ax, "c", "Overall LR ranking")
    names = list(DATASET_ORDER)
    y = np.arange(len(names))
    for idx, name in enumerate(names):
        per_time = records[name]["per_time"]
        values = per_time.loc[per_time["scope"] == "all_scored_pairs", "spearman"]
        pooled = float(records[name]["summary"]["global_spearman"])
        ax.plot(
            [values.min(), values.max()],
            [idx, idx],
            color=COLORS[name],
            linewidth=2.2,
            solid_capstyle="round",
        )
        ax.scatter(
            [pooled], [idx], color=COLORS[name], s=42, zorder=3, edgecolor="white"
        )
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set_xlim(0.90, 1.005)
    ax.set_xlabel("Spearman rank correlation")
    ax.text(
        0.03,
        0.96,
        "Line: range across time\nPoint: pooled correlation",
        transform=ax.transAxes,
        fontsize=8.3,
        ha="left",
        va="top",
    )
    _clean_axis(ax, grid=True)


def _top_overlap(ax: plt.Axes, records: dict[str, dict[str, object]]) -> None:
    _panel_title(ax, "d", "Top-10 LR stability over time")
    for name in DATASET_ORDER:
        table = records[name]["per_time"]
        table = table.loc[table["scope"] == "all_scored_pairs"].sort_values("time")
        times = table["time"].to_numpy(dtype=float)
        normalized = (times - times.min()) / max(times.max() - times.min(), 1.0)
        ax.plot(
            normalized,
            table["top_jaccard"],
            marker="o",
            markersize=4,
            linewidth=1.6,
            color=COLORS[name],
            label=name,
        )
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0.35, 1.04)
    ax.set_xlabel("Normalized developmental time")
    ax.set_ylabel("Top-10 Jaccard overlap")
    ax.legend(
        frameon=False,
        ncol=4,
        loc="center",
        bbox_to_anchor=(0.69, 1.17),
        columnspacing=1.2,
        handletextpad=0.45,
    )
    _clean_axis(ax, grid=True)


def _display_pair(value: str) -> str:
    tokens = str(value).split("_")
    if len(tokens) <= 2:
        return f"{tokens[0]} → {tokens[1]}"
    return f"{tokens[0]} → {'+'.join(tokens[1:])}"


def _top_ten(table: pd.DataFrame, score: str) -> list[str]:
    return (
        table.sort_values([score, "pair"], ascending=[False, True])
        .head(10)["pair"]
        .astype(str)
        .tolist()
    )


def _rank_list(
    ax: plt.Axes,
    record: dict[str, object],
    *,
    label: str,
    title: str,
) -> dict[str, object]:
    per_time = record["per_time"]
    all_scope = per_time.loc[per_time["scope"] == "all_scored_pairs"]
    worst = all_scope.sort_values(["top_jaccard", "time"]).iloc[0]
    time_value = float(worst["time"])
    table = record["paired"].loc[np.isclose(record["paired"]["time"], time_value)]
    minimum = _top_ten(table, "score_min")
    geometric = _top_ten(table, "score_geometric_mean")
    min_rank = {pair: idx + 1 for idx, pair in enumerate(minimum)}
    geo_rank = {pair: idx + 1 for idx, pair in enumerate(geometric)}
    shared = set(minimum) & set(geometric)

    ax.set_xlim(0, 1)
    ax.set_ylim(10.8, 0.0)
    ax.axis("off")
    _panel_title(ax, label, title)
    ax.text(0.02, 0.35, "Minimum gate", fontsize=9, fontweight="bold", va="bottom")
    ax.text(
        0.98,
        0.35,
        "Geometric mean",
        fontsize=9,
        fontweight="bold",
        ha="right",
        va="bottom",
    )
    for pair in shared:
        ax.plot(
            [0.39, 0.61],
            [min_rank[pair], geo_rank[pair]],
            color="#B3BBC2",
            linewidth=0.8,
            zorder=1,
        )
    for rank, pair in enumerate(minimum, start=1):
        color = TEXT if pair in shared else "#A33A3A"
        ax.text(
            0.02,
            rank,
            f"{rank:>2}  {_display_pair(pair)}",
            fontsize=8.2,
            ha="left",
            va="center",
            color=color,
        )
    for rank, pair in enumerate(geometric, start=1):
        color = TEXT if pair in shared else SENSITIVITY
        weight = "normal" if pair in shared else "bold"
        ax.text(
            0.98,
            rank,
            f"{_display_pair(pair)}  {rank:<2}",
            fontsize=8.2,
            ha="right",
            va="center",
            color=color,
            fontweight=weight,
        )
    ax.text(
        0.50,
        10.55,
        f"t = {time_value:g}   top-10 Jaccard = {float(worst['top_jaccard']):.2f}",
        ha="center",
        va="center",
        fontsize=8.5,
    )
    return {
        "time": time_value,
        "top_jaccard": float(worst["top_jaccard"]),
        "minimum_top10": minimum,
        "geometric_top10": geometric,
    }


def _write_caption(path: Path, rank_details: dict[str, dict[str, object]]) -> None:
    path.write_text(
        "\n".join(
            [
                "# Ligand–receptor complex aggregation sensitivity",
                "",
                "**(a)** Multi-subunit ligand or receptor expression is combined by the minimum subunit for the primary analysis and by a zero-preserving geometric mean for sensitivity analysis. Both rules require every subunit. **(b)** Number of strictly scored multi-subunit LR pairs in each dataset. AD mouse has no eligible multi-subunit pair. **(c)** Pooled Spearman correlation and the range of per-time correlations between the two aggregation rules. **(d)** Jaccard overlap of the ten strongest LR pairs over normalized developmental time. **(e)** ARISTA top-ten lists at the most sensitive time point. **(f)** Chicken Heart top-ten lists at the most sensitive time point.",
                "",
                "Trajectories, expression states, communication attention, LR databases, time grids, and the scored pair universe were held fixed. The minimum-gate tables were reproduced before changing the aggregation. Overall LR rankings remain strongly correlated, while top heteromeric complexes are more sensitive in ARISTA and Chicken Heart. Chicken Heart uses the declared conserved-symbol human CellChatDB proxy and is not a species-complete chicken interaction screen.",
                "",
                f"ARISTA panel time: {rank_details['ARISTA']['time']:g}. Chicken Heart panel time: {rank_details['Chicken Heart']['time']:g}.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def render(analyses: dict[str, Path], admouse_table: Path, output_dir: Path) -> Path:
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory must be absent or empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    records = {name: _read_analysis(analyses[name]) for name in DATASET_ORDER}
    admouse_table = admouse_table.expanduser().resolve()
    if not admouse_table.is_file():
        raise FileNotFoundError(admouse_table)
    ad_counts = _admouse_counts(admouse_table)
    if ad_counts != (7, 0):
        raise ValueError(
            f"Expected the formal AD scope (7 pairs, 0 multi), got {ad_counts}"
        )

    _style()
    fig = plt.figure(figsize=(8.27, 11.69))
    grid = fig.add_gridspec(
        5,
        2,
        height_ratios=[1.18, 1.0, 0.95, 1.25, 1.25],
        hspace=0.68,
        wspace=0.42,
        left=0.125,
        right=0.96,
        top=0.94,
        bottom=0.055,
    )
    fig.suptitle(
        "Sensitivity of ligand–receptor dynamics to complex aggregation",
        fontsize=13,
        fontweight="bold",
        y=0.982,
    )
    ax_a = fig.add_subplot(grid[0, :])
    ax_b = fig.add_subplot(grid[1, 0])
    ax_c = fig.add_subplot(grid[1, 1])
    ax_d = fig.add_subplot(grid[2, :])
    ax_e = fig.add_subplot(grid[3, :])
    ax_f = fig.add_subplot(grid[4, :])
    _schematic(ax_a)
    _coverage(ax_b, records, ad_counts)
    _rank_stability(ax_c, records)
    _top_overlap(ax_d, records)
    rank_details = {
        "ARISTA": _rank_list(
            ax_e,
            records["ARISTA"],
            label="e",
            title="ARISTA top-pair changes",
        ),
        "Chicken Heart": _rank_list(
            ax_f,
            records["Chicken Heart"],
            label="f",
            title="Chicken Heart top-pair changes",
        ),
    }

    pdf = output_dir / "lr_complex_aggregation_reviewer_response.pdf"
    png = output_dir / "lr_complex_aggregation_reviewer_response.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=320)
    plt.close(fig)

    metrics_rows = []
    for name in DATASET_ORDER:
        summary = records[name]["summary"]
        per_time = records[name]["per_time"]
        all_scope = per_time.loc[per_time["scope"] == "all_scored_pairs"]
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
    metrics_rows.append(
        {
            "dataset": "AD mouse",
            "n_scored_pairs": 7,
            "n_multisubunit_pairs": 0,
            "pooled_spearman": 1.0,
            "min_per_time_spearman": 1.0,
            "min_top10_jaccard": 1.0,
        }
    )
    metrics = output_dir / "figure_metrics.csv"
    pd.DataFrame(metrics_rows).to_csv(metrics, index=False)
    panel_sources = output_dir / "panel_sources.csv"
    pd.DataFrame(
        [
            {"panel": "a", "source": "declared min and geometric-mean formulas"},
            {
                "panel": "b",
                "source": "four comparison summaries plus formal AD pair table",
            },
            {"panel": "c", "source": "per_time_stability.csv and summary.json"},
            {"panel": "d", "source": "per_time_stability.csv"},
            {"panel": "e", "source": "ARISTA paired_scores.csv"},
            {"panel": "f", "source": "Chicken Heart paired_scores.csv"},
        ]
    ).to_csv(panel_sources, index=False)
    caption = output_dir / "caption.md"
    _write_caption(caption, rank_details)
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
    inputs.append(
        {
            "dataset": "AD mouse",
            "label": "formal_pair_timecourse",
            "path": str(admouse_table),
            "sha256": _sha256(admouse_table),
        }
    )
    outputs = [pdf, png, metrics, panel_sources, caption, script_snapshot]
    manifest = output_dir / "figure_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "complete",
                "claim": (
                    "Broad LR dynamics are rank-stable to complex aggregation, "
                    "while top heteromeric pairs show dataset-specific sensitivity."
                ),
                "inputs": inputs,
                "rank_details": rank_details,
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
                "Scientific claim: Broad LR dynamics are rank-stable to complex aggregation, while top heteromeric pairs show dataset-specific sensitivity.",
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
                f"- AD mouse formal pair table: `{admouse_table}`",
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
    parser.add_argument("--admouse-pair-table", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    print(
        render(
            _parse_analysis(args.analysis),
            args.admouse_pair_table,
            args.output_dir,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
