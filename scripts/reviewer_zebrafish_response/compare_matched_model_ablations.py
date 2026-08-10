#!/usr/bin/env python3
"""Compare matched CytoBridge model ablations from distribution evaluations.

Each input must be a ``distribution_metrics.csv`` produced by the public
``evaluate_model_distributions`` / ``save_distribution_evaluation`` workflow.
The command validates a common time/space evaluation grid, reports raw
condition metrics and changes relative to a nominated full-model reference,
and creates reviewer-facing W1/W2 and total-mass-variation figures.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    return {
        "path": str(resolved),
        "bytes": int(resolved.stat().st_size),
        "sha256": _sha256(resolved),
    }


def _git_state() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--short"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        return {"commit": commit, "dirty": bool(status)}
    except (OSError, subprocess.CalledProcessError) as error:
        return {"commit": None, "dirty": None, "error": str(error)}


def _parse_named_paths(specs: Sequence[str]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(
                f"Invalid --condition {spec!r}; expected NAME=PATH."
            )
        raw_name, raw_path = spec.split("=", 1)
        name = raw_name.strip()
        path = Path(raw_path.strip()).expanduser().resolve()
        if not name or not raw_path.strip():
            raise ValueError("Condition NAME and PATH must both be non-empty.")
        if name in paths:
            raise ValueError(f"Duplicate condition name: {name!r}.")
        if not path.is_file():
            raise FileNotFoundError(path)
        paths[name] = path
    if len(paths) < 2:
        raise ValueError("At least two --condition NAME=PATH inputs are required.")
    return paths


def _load_and_validate(
    paths: dict[str, Path],
) -> tuple[pd.DataFrame, list[str], list[float]]:
    required = {"time", "space", "w1", "w2", "tmv"}
    frames: list[pd.DataFrame] = []
    reference_grid: list[tuple[float, str]] | None = None
    spaces: list[str] = []
    times: list[float] = []
    for name, path in paths.items():
        frame = pd.read_csv(path)
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"{path} is missing columns: {missing}.")
        if frame.duplicated(["time", "space"]).any():
            raise ValueError(f"{path} has duplicate time/space rows.")
        for column in ("time", "w1", "w2", "tmv"):
            frame[column] = pd.to_numeric(frame[column], errors="raise")
        if not np.isfinite(frame[["time", "w1", "w2", "tmv"]]).all().all():
            raise ValueError(f"{path} contains non-finite required metrics.")
        frame["space"] = frame["space"].astype(str)
        grid = list(
            frame.sort_values(["time", "space"])[["time", "space"]]
            .itertuples(index=False, name=None)
        )
        if reference_grid is None:
            reference_grid = grid
            spaces = frame["space"].drop_duplicates().tolist()
            times = sorted(frame["time"].unique().astype(float).tolist())
        elif grid != reference_grid:
            raise ValueError(
                f"Condition {name!r} does not share the same time/space grid."
            )
        frame.insert(0, "condition", name)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True), spaces, times


def summarize_conditions(
    metrics: pd.DataFrame, *, reference: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if reference not in set(metrics["condition"].astype(str)):
        raise ValueError(f"Unknown reference condition: {reference!r}.")
    summary = (
        metrics.groupby(["condition", "space"], sort=False, observed=True)
        .agg(
            n_time_points=("time", "nunique"),
            mean_w1=("w1", "mean"),
            mean_w2=("w2", "mean"),
            endpoint_w1=("w1", "last"),
            endpoint_w2=("w2", "last"),
        )
        .reset_index()
    )
    long = summary.melt(
        id_vars=["condition", "space", "n_time_points"],
        value_vars=["mean_w1", "mean_w2", "endpoint_w1", "endpoint_w2"],
        var_name="metric",
        value_name="value",
    )
    reference_rows = (
        long.loc[long["condition"].eq(reference), ["space", "metric", "value"]]
        .rename(columns={"value": "reference_value"})
        .copy()
    )
    relative = long.merge(
        reference_rows, on=["space", "metric"], how="left", validate="many_to_one"
    )
    relative["absolute_change_vs_reference"] = (
        relative["value"] - relative["reference_value"]
    )
    relative["percent_change_vs_reference"] = np.where(
        relative["reference_value"].ne(0),
        100.0
        * relative["absolute_change_vs_reference"]
        / relative["reference_value"].abs(),
        np.nan,
    )

    mass = metrics[["condition", "time", "tmv"]].drop_duplicates()
    counts = mass.groupby(["condition", "time"]).size()
    if not counts.eq(1).all():
        raise ValueError("TMV differs across duplicated space rows.")
    mass_summary = (
        mass.groupby("condition", sort=False, observed=True)
        .agg(
            n_time_points=("time", "nunique"),
            mean_tmv=("tmv", "mean"),
            endpoint_tmv=("tmv", "last"),
            max_tmv=("tmv", "max"),
        )
        .reset_index()
    )
    reference_mass = mass_summary.loc[
        mass_summary["condition"].eq(reference), "mean_tmv"
    ]
    if len(reference_mass) != 1:
        raise ValueError("Reference condition must have one mass summary row.")
    reference_mean_tmv = float(reference_mass.iloc[0])
    mass_summary["mean_tmv_change_vs_reference"] = (
        mass_summary["mean_tmv"] - reference_mean_tmv
    )
    return summary, relative, mass_summary


def _condition_colors(names: Sequence[str]) -> dict[str, Any]:
    palette = plt.get_cmap("tab10")
    return {name: palette(index % 10) for index, name in enumerate(names)}


def plot_wasserstein(
    summary: pd.DataFrame,
    output: Path,
    *,
    condition_order: Sequence[str],
    space_order: Sequence[str],
) -> None:
    colors = _condition_colors(condition_order)
    fig, axes = plt.subplots(
        len(space_order),
        2,
        figsize=(10.5, max(3.2, 2.9 * len(space_order))),
        squeeze=False,
    )
    for row, space in enumerate(space_order):
        subset = summary.loc[summary["space"].eq(space)].set_index("condition")
        for column, metric in enumerate(("mean_w1", "mean_w2")):
            axis = axes[row, column]
            values = [float(subset.loc[name, metric]) for name in condition_order]
            positions = np.arange(len(condition_order))
            bars = axis.bar(
                positions,
                values,
                color=[colors[name] for name in condition_order],
                width=0.72,
            )
            axis.bar_label(
                bars,
                labels=[f"{value:.3g}" for value in values],
                padding=2,
                fontsize=8,
            )
            axis.set_xticks(positions, condition_order, rotation=18, ha="right")
            axis.set_ylabel(f"Mean {metric[-2:].upper()} (lower is better)")
            axis.set_title(f"{space}: {metric[-2:].upper()}")
            axis.set_ylim(bottom=0)
            axis.grid(axis="y", alpha=0.22)
            axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Matched model ablations: generated-versus-observed distance",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_mass(
    mass_summary: pd.DataFrame,
    output: Path,
    *,
    condition_order: Sequence[str],
) -> None:
    colors = _condition_colors(condition_order)
    indexed = mass_summary.set_index("condition")
    values = [float(indexed.loc[name, "mean_tmv"]) for name in condition_order]
    fig, axis = plt.subplots(figsize=(6.8, 4.5))
    positions = np.arange(len(condition_order))
    bars = axis.bar(
        positions,
        values,
        color=[colors[name] for name in condition_order],
        width=0.68,
    )
    axis.bar_label(
        bars,
        labels=[f"{value:.3g}" for value in values],
        padding=3,
        fontsize=9,
    )
    axis.set_xticks(positions, condition_order, rotation=18, ha="right")
    axis.set_ylabel("Mean TMV (lower is better)")
    axis.set_title("Matched model ablations: total-mass variation")
    axis.set_ylim(bottom=0)
    axis.grid(axis="y", alpha=0.22)
    axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--condition",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="Repeat for the full model and every matched ablation.",
    )
    parser.add_argument(
        "--reference",
        required=True,
        help="Condition name used as the full-model reference.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def run(args: argparse.Namespace, *, command_argv: Sequence[str]) -> dict[str, Any]:
    paths = _parse_named_paths(args.condition)
    if args.reference not in paths:
        raise ValueError("--reference must match one --condition name.")
    output_dir = args.output_dir.expanduser().resolve()
    known = (
        "matched_ablation_metrics.csv",
        "matched_ablation_summary.csv",
        "relative_to_reference.csv",
        "mass_summary.csv",
        "matched_ablation_w1_w2.png",
        "matched_ablation_w1_w2.pdf",
        "matched_ablation_tmv.png",
        "matched_ablation_tmv.pdf",
        "run_manifest.json",
    )
    existing = [output_dir / name for name in known if (output_dir / name).exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing outputs: "
            + ", ".join(str(path) for path in existing)
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics, spaces, times = _load_and_validate(paths)
    summary, relative, mass_summary = summarize_conditions(
        metrics, reference=str(args.reference)
    )
    condition_order = list(paths)
    output_paths = {
        "metrics": output_dir / "matched_ablation_metrics.csv",
        "summary": output_dir / "matched_ablation_summary.csv",
        "relative": output_dir / "relative_to_reference.csv",
        "mass_summary": output_dir / "mass_summary.csv",
        "w1_w2_png": output_dir / "matched_ablation_w1_w2.png",
        "w1_w2_pdf": output_dir / "matched_ablation_w1_w2.pdf",
        "tmv_png": output_dir / "matched_ablation_tmv.png",
        "tmv_pdf": output_dir / "matched_ablation_tmv.pdf",
    }
    metrics.to_csv(output_paths["metrics"], index=False, float_format="%.12g")
    summary.to_csv(output_paths["summary"], index=False, float_format="%.12g")
    relative.to_csv(output_paths["relative"], index=False, float_format="%.12g")
    mass_summary.to_csv(
        output_paths["mass_summary"], index=False, float_format="%.12g"
    )
    for suffix in ("png", "pdf"):
        plot_wasserstein(
            summary,
            output_paths[f"w1_w2_{suffix}"],
            condition_order=condition_order,
            space_order=spaces,
        )
        plot_mass(
            mass_summary,
            output_paths[f"tmv_{suffix}"],
            condition_order=condition_order,
        )

    manifest_path = output_dir / "run_manifest.json"
    manifest = {
        "schema_version": 1,
        "analysis": "matched_model_ablation_distribution_comparison",
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": {
            "argv": [str(value) for value in command_argv],
            "shell": shlex.join([str(value) for value in command_argv]),
            "cwd": str(Path.cwd().resolve()),
        },
        "code": {
            "script": _file_record(SCRIPT_PATH),
            "git": _git_state(),
        },
        "inputs": {
            name: _file_record(path) for name, path in paths.items()
        },
        "parameters": {
            "reference": str(args.reference),
            "condition_order": condition_order,
            "space_order": spaces,
            "times": times,
        },
        "interpretation": {
            "matched_design": (
                "Inputs must come from models trained and evaluated with the "
                "same data, random seed, optimization budget, and evaluation "
                "settings; this command validates only the evaluation grid."
            ),
            "direction": "Lower W1, W2, and TMV are better.",
            "causal_scope": (
                "A matched computational ablation estimates model-component "
                "contribution; it is not an experimental biological perturbation."
            ),
        },
        "outputs": {
            name: _file_record(path) for name, path in output_paths.items()
        },
        "manifest": {"path": str(manifest_path), "self_hash_omitted": True},
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(arguments)
    command = [sys.executable, str(SCRIPT_PATH), *arguments]
    manifest = run(args, command_argv=command)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "output_dir": str(args.output_dir.expanduser().resolve()),
                "outputs": manifest["outputs"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
