#!/usr/bin/env python3
"""Compare strict LR dynamics under min and geometric-mean complex gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from CytoBridge.tl.downstream.lr_projection import (  # noqa: E402
    load_ligand_receptor_database,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_state() -> dict[str, object]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return {"commit": commit, "dirty": dirty}
    except Exception:
        return {"commit": None, "dirty": None}


def _read_score_table(path: Path, score_name: str) -> pd.DataFrame:
    table = pd.read_csv(path)
    required = {"time", "pair", "score"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}.")
    table = table[["time", "pair", "score"]].copy()
    table["time"] = pd.to_numeric(table["time"], errors="raise")
    table["pair"] = table["pair"].astype(str)
    table[score_name] = pd.to_numeric(table.pop("score"), errors="raise")
    if table.duplicated(["time", "pair"]).any():
        raise ValueError(f"{path} contains duplicate time/pair rows.")
    if not np.isfinite(table[score_name]).all() or (table[score_name] < 0).any():
        raise ValueError(f"{path} scores must be finite and non-negative.")
    return table


def _pair_annotations(lr_database: Path) -> pd.DataFrame:
    database = load_ligand_receptor_database(lr_database)
    database["pair"] = database["ligand"] + "_" + database["receptor"]
    database["ligand_n_subunits"] = database["ligand"].map(
        lambda value: len([token for token in str(value).split("_") if token])
    )
    database["receptor_n_subunits"] = database["receptor"].map(
        lambda value: len([token for token in str(value).split("_") if token])
    )
    database["is_multisubunit"] = (
        (database["ligand_n_subunits"] > 1)
        | (database["receptor_n_subunits"] > 1)
    )
    conflicting = (
        database.groupby("pair")["is_multisubunit"].nunique().loc[lambda x: x > 1]
    )
    if not conflicting.empty:
        raise ValueError(
            "LR database gives conflicting subunit definitions for pairs: "
            f"{conflicting.index.tolist()[:10]}."
        )
    return database.drop_duplicates("pair")[
        [
            "pair",
            "ligand",
            "receptor",
            "ligand_n_subunits",
            "receptor_n_subunits",
            "is_multisubunit",
        ]
    ]


def _safe_correlation(x: np.ndarray, y: np.ndarray, method: str) -> float:
    if x.size < 2 or np.all(x == x[0]) or np.all(y == y[0]):
        return float("nan")
    if method == "spearman":
        return float(spearmanr(x, y).statistic)
    return float(pearsonr(x, y).statistic)


def _top_overlap(
    subset: pd.DataFrame, *, top_fraction: float, top_k: int
) -> tuple[int, int, float]:
    n = int(subset.shape[0])
    if n == 0:
        return 0, 0, float("nan")
    selected_n = min(n, max(1, min(int(top_k), math.ceil(top_fraction * n))))
    top_min = set(
        subset.nlargest(selected_n, "score_min", keep="all")
        .head(selected_n)["pair"]
    )
    top_geometric = set(
        subset.nlargest(selected_n, "score_geometric_mean", keep="all")
        .head(selected_n)["pair"]
    )
    union = top_min | top_geometric
    intersection = top_min & top_geometric
    return selected_n, len(intersection), float(len(intersection) / len(union))


def compare_tables(
    min_table: Path,
    geometric_table: Path,
    lr_database: Path,
    *,
    top_fraction: float = 0.2,
    top_k: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    if not (0 < float(top_fraction) <= 1):
        raise ValueError("top_fraction must be in (0, 1].")
    if int(top_k) <= 0:
        raise ValueError("top_k must be positive.")
    minimum = _read_score_table(min_table, "score_min")
    geometric = _read_score_table(
        geometric_table, "score_geometric_mean"
    )
    merged = minimum.merge(
        geometric,
        on=["time", "pair"],
        how="outer",
        validate="1:1",
        indicator=True,
    )
    if not merged["_merge"].eq("both").all():
        unmatched = merged.loc[
            ~merged["_merge"].eq("both"), ["time", "pair", "_merge"]
        ]
        raise ValueError(
            "Min and geometric-mean runs do not score the same time/pair "
            f"universe; examples={unmatched.head().to_dict(orient='records')}."
        )
    merged = merged.drop(columns="_merge").merge(
        _pair_annotations(lr_database),
        on="pair",
        how="left",
        validate="many_to_one",
    )
    if merged["is_multisubunit"].isna().any():
        missing_pairs = sorted(
            merged.loc[merged["is_multisubunit"].isna(), "pair"].unique()
        )
        raise ValueError(
            "Scored pairs are absent from LR database: "
            f"{missing_pairs[:10]}."
        )
    merged["score_difference"] = (
        merged["score_geometric_mean"] - merged["score_min"]
    )
    merged["absolute_difference"] = merged["score_difference"].abs()
    denominator = np.maximum(
        np.maximum(
            merged["score_geometric_mean"].abs(),
            merged["score_min"].abs(),
        ),
        np.finfo(float).eps,
    )
    merged["symmetric_relative_difference"] = (
        merged["absolute_difference"] / denominator
    )

    per_time_rows: list[dict[str, object]] = []
    for time_value in sorted(merged["time"].unique()):
        current = merged.loc[merged["time"] == time_value]
        for scope, subset in (
            ("all_scored_pairs", current),
            (
                "multisubunit_pairs",
                current.loc[current["is_multisubunit"].astype(bool)],
            ),
        ):
            x = subset["score_min"].to_numpy(dtype=float)
            y = subset["score_geometric_mean"].to_numpy(dtype=float)
            selected_n, overlap_n, jaccard = _top_overlap(
                subset, top_fraction=top_fraction, top_k=top_k
            )
            per_time_rows.append(
                {
                    "time": float(time_value),
                    "scope": scope,
                    "n_pairs": int(subset.shape[0]),
                    "spearman": _safe_correlation(x, y, "spearman"),
                    "pearson": _safe_correlation(x, y, "pearson"),
                    "top_n": selected_n,
                    "top_overlap_n": overlap_n,
                    "top_jaccard": jaccard,
                    "mean_absolute_difference": (
                        float(subset["absolute_difference"].mean())
                        if not subset.empty
                        else np.nan
                    ),
                    "max_absolute_difference": (
                        float(subset["absolute_difference"].max())
                        if not subset.empty
                        else np.nan
                    ),
                    "mean_symmetric_relative_difference": (
                        float(
                            subset["symmetric_relative_difference"].mean()
                        )
                        if not subset.empty
                        else np.nan
                    ),
                    "max_symmetric_relative_difference": (
                        float(
                            subset["symmetric_relative_difference"].max()
                        )
                        if not subset.empty
                        else np.nan
                    ),
                }
            )
    per_time = pd.DataFrame(per_time_rows)

    per_pair_rows = []
    for pair, subset in merged.groupby("pair", sort=True):
        subset = subset.sort_values("time")
        x = subset["score_min"].to_numpy(dtype=float)
        y = subset["score_geometric_mean"].to_numpy(dtype=float)
        per_pair_rows.append(
            {
                "pair": str(pair),
                "ligand": str(subset.iloc[0]["ligand"]),
                "receptor": str(subset.iloc[0]["receptor"]),
                "is_multisubunit": bool(subset.iloc[0]["is_multisubunit"]),
                "n_times": int(subset.shape[0]),
                "trajectory_spearman": _safe_correlation(x, y, "spearman"),
                "trajectory_pearson": _safe_correlation(x, y, "pearson"),
                "mean_absolute_difference": float(
                    subset["absolute_difference"].mean()
                ),
                "max_absolute_difference": float(
                    subset["absolute_difference"].max()
                ),
                "mean_symmetric_relative_difference": float(
                    subset["symmetric_relative_difference"].mean()
                ),
                "max_symmetric_relative_difference": float(
                    subset["symmetric_relative_difference"].max()
                ),
            }
        )
    per_pair = pd.DataFrame(per_pair_rows)

    x_all = merged["score_min"].to_numpy(dtype=float)
    y_all = merged["score_geometric_mean"].to_numpy(dtype=float)
    multi = merged.loc[merged["is_multisubunit"].astype(bool)]
    summary = {
        "n_times": int(merged["time"].nunique()),
        "n_scored_pairs": int(merged["pair"].nunique()),
        "n_multisubunit_pairs": int(
            merged.loc[merged["is_multisubunit"].astype(bool), "pair"].nunique()
        ),
        "global_spearman": _safe_correlation(x_all, y_all, "spearman"),
        "global_pearson": _safe_correlation(x_all, y_all, "pearson"),
        "global_max_symmetric_relative_difference": float(
            merged["symmetric_relative_difference"].max()
        ),
        "multisubunit_max_symmetric_relative_difference": (
            float(multi["symmetric_relative_difference"].max())
            if not multi.empty
            else None
        ),
        "primary_result_is_mathematically_invariant": bool(
            merged["absolute_difference"].max() <= 1e-12
        ),
        "top_fraction": float(top_fraction),
        "top_k_cap": int(top_k),
    }
    return merged, per_time, per_pair, summary


def _plot(
    merged: pd.DataFrame,
    per_time: pd.DataFrame,
    summary: dict[str, object],
    path: Path,
) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.5))
    colors = np.where(merged["is_multisubunit"], "#D95F02", "#3569A8")
    x = np.log1p(merged["score_min"].to_numpy(dtype=float))
    y = np.log1p(merged["score_geometric_mean"].to_numpy(dtype=float))
    axes[0, 0].scatter(x, y, c=colors, s=20, alpha=0.72, linewidths=0)
    lower = float(min(x.min(), y.min()))
    upper = float(max(x.max(), y.max()))
    axes[0, 0].plot([lower, upper], [lower, upper], "--", color="black", lw=1)
    axes[0, 0].set(
        xlabel="log1p score: minimum gate",
        ylabel="log1p score: geometric mean",
        title="All time × LR observations",
    )

    all_scope = per_time.loc[per_time["scope"] == "all_scored_pairs"]
    axes[0, 1].plot(
        all_scope["time"],
        all_scope["spearman"],
        marker="o",
        color="#3569A8",
    )
    axes[0, 1].axhline(1.0, ls="--", lw=1, color="black")
    axes[0, 1].set(
        ylim=(-0.05, 1.05),
        xlabel="Model time",
        ylabel="Spearman correlation",
        title="LR rank stability at every time",
    )

    axes[1, 0].plot(
        all_scope["time"],
        all_scope["top_jaccard"],
        marker="o",
        color="#2A9D8F",
    )
    axes[1, 0].axhline(1.0, ls="--", lw=1, color="black")
    axes[1, 0].set(
        ylim=(-0.05, 1.05),
        xlabel="Model time",
        ylabel="Top-set Jaccard",
        title="Top-signal overlap",
    )

    pair_difference = (
        merged.groupby(["pair", "is_multisubunit"], as_index=False)[
            "symmetric_relative_difference"
        ]
        .max()
        .sort_values("symmetric_relative_difference", ascending=False)
        .head(12)
        .sort_values("symmetric_relative_difference")
    )
    axes[1, 1].barh(
        pair_difference["pair"],
        pair_difference["symmetric_relative_difference"],
        color=np.where(pair_difference["is_multisubunit"], "#D95F02", "#3569A8"),
    )
    axes[1, 1].set(
        xlabel="Maximum symmetric relative difference",
        title="Most aggregation-sensitive pairs",
    )
    if int(summary["n_multisubunit_pairs"]) == 0:
        axes[1, 1].text(
            0.5,
            0.5,
            "No eligible multi-subunit LR pairs\nin the primary feature contract",
            transform=axes[1, 1].transAxes,
            ha="center",
            va="center",
            bbox={"facecolor": "white", "edgecolor": "#777777", "alpha": 0.95},
        )
    for axis in axes.ravel():
        axis.grid(alpha=0.18, linewidth=0.6)
    fig.suptitle("LR complex aggregation sensitivity: min vs geometric mean")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-table", type=Path, required=True)
    parser.add_argument("--geometric-table", type=Path, required=True)
    parser.add_argument("--lr-database", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-fraction", type=float, default=0.2)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args(argv)

    input_paths = [
        args.min_table.expanduser().resolve(),
        args.geometric_table.expanduser().resolve(),
        args.lr_database.expanduser().resolve(),
    ]
    for path in input_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    merged, per_time, per_pair, summary = compare_tables(
        input_paths[0],
        input_paths[1],
        input_paths[2],
        top_fraction=args.top_fraction,
        top_k=args.top_k,
    )
    outputs = []
    for filename, table in (
        ("paired_scores.csv", merged),
        ("per_time_stability.csv", per_time),
        ("per_pair_stability.csv", per_pair),
    ):
        path = output_dir / filename
        table.to_csv(path, index=False)
        outputs.append(path)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    outputs.append(summary_path)
    for suffix in ("png", "pdf"):
        outputs.append(
            _plot(
                merged,
                per_time,
                summary,
                output_dir / f"lr_complex_aggregation_sensitivity.{suffix}",
            )
        )
    readme = output_dir / "README.md"
    interpretation = (
        "No multi-subunit ligand or receptor passed the strict primary feature "
        "contract, so min and geometric mean are identical for every reported "
        "primary LR trajectory."
        if summary["n_multisubunit_pairs"] == 0
        else "Multi-subunit complexes are present; use the per-time and per-pair "
        "tables to report rank, top-signal, and magnitude sensitivity."
    )
    readme.write_text(
        "\n".join(
            [
                "# LR complex aggregation sensitivity",
                "",
                interpretation,
                "",
                "The two inputs must have the same strict time/pair universe. "
                "The script never fills missing pairs with zero.",
                "",
                f"- Scored pairs: {summary['n_scored_pairs']}",
                f"- Multi-subunit pairs: {summary['n_multisubunit_pairs']}",
                f"- Global Spearman: {summary['global_spearman']}",
                f"- Maximum relative difference: "
                f"{summary['global_max_symmetric_relative_difference']}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    outputs.append(readme)

    manifest = {
        "schema_version": 1,
        "command": [str(value) for value in sys.argv],
        "git": _git_state(),
        "inputs": [
            {"path": str(path), "sha256": _sha256(path)} for path in input_paths
        ],
        "parameters": {
            "top_fraction": float(args.top_fraction),
            "top_k": int(args.top_k),
            "require_identical_time_pair_universe": True,
        },
        "outputs": [
            {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in outputs
        ],
    }
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
