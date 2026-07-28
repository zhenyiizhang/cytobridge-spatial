#!/usr/bin/env python3
"""Run and report whole-trajectory same-checkpoint functional ablations.

The fitted full checkpoint is evaluated from the earliest observed stage to
every later observed stage in one continuous, non-split weighted SDE rollout.
All conditions use common Brownian increments within each rollout seed.

This command deliberately separates two random quantities:

* ``--seeds`` control stochastic SDE rollouts;
* ``--ot-sampling-seed`` controls bounded exact-OT support sampling and is
  held fixed across conditions and rollout seeds.

The latter makes the reported across-seed variation an SDE sensitivity
summary rather than a mixture of SDE and evaluation-subsampling variation.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import traceback
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import CytoBridge as cb  # noqa: E402
from CytoBridge.tl.downstream.evaluation import (  # noqa: E402
    DistributionEvaluationResult,
    compute_distribution_metrics,
)


DEFAULT_SEEDS = (17, 23, 42, 101, 202)
DEFAULT_CONDITIONS = ("full", "interaction_off", "lr_gate_off")
SPACE_ORDER = ("joint", "state", "spatial")
CONDITION_LABELS = {
    "full": "Full",
    "interaction_off": "Interaction OFF",
    "lr_gate_off": "All-spatial gate",
}
CONDITION_COLORS = {
    "full": "#4C78A8",
    "interaction_off": "#F58518",
    "lr_gate_off": "#E45756",
}
SPACE_LABELS = {
    "joint": "Joint (2D spatial + PCA state)",
    "state": "PCA state",
    "spatial": "Spatial (2D)",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--adata", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--time-points", type=float, nargs="+")
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=list(DEFAULT_CONDITIONS),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--primary-seed", type=int, default=42)
    parser.add_argument("--ot-sampling-seed", type=int, default=42)
    parser.add_argument("--n-samples", type=int, default=5000)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--sigma", type=float, default=0.03)
    parser.add_argument("--interaction-m", type=int, default=1024)
    parser.add_argument("--max-ot-points", type=int, default=1024)
    parser.add_argument("--structure-max-points", type=int, default=5000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--time-key", default="time_point_processed")
    parser.add_argument("--obsm-key", default="X_latent")
    parser.add_argument("--spatial-key", default="spatial_aligned")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing non-empty output directory.",
    )
    return parser


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
    def run(*arguments: str) -> str:
        return subprocess.run(
            ["git", *arguments],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    try:
        status = run("status", "--short")
        return {
            "repo_root": str(REPO_ROOT),
            "commit": run("rev-parse", "HEAD"),
            "dirty": bool(status),
            "status_short": status.splitlines(),
        }
    except (OSError, subprocess.CalledProcessError) as error:
        return {"repo_root": str(REPO_ROOT), "error": str(error)}


def _prepare_output_dir(path: Path, *, overwrite: bool) -> Path:
    root = path.expanduser().resolve()
    if root.exists() and any(root.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory is non-empty: {root}. "
                "Use --overwrite explicitly."
            )
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    return root


def _ordered_unique_ints(values: Sequence[int]) -> tuple[int, ...]:
    result: list[int] = []
    for value in values:
        parsed = int(value)
        if parsed not in result:
            result.append(parsed)
    if not result:
        raise ValueError("At least one rollout seed is required.")
    return tuple(result)


def _condition_label(name: str) -> str:
    return CONDITION_LABELS.get(name, name.replace("_", " ").title())


def _condition_color(name: str) -> str:
    return CONDITION_COLORS.get(name, "#777777")


def _space_slices(
    evaluation: DistributionEvaluationResult,
) -> dict[str, slice]:
    first_time = float(evaluation.time_points[0])
    dim = int(np.asarray(evaluation.predicted_points[first_time]).shape[1])
    spatial_dim = int(evaluation.spatial_dim)
    if spatial_dim <= 0 or spatial_dim >= dim:
        raise ValueError(
            f"Expected joint spatial/state output; spatial_dim={spatial_dim}, dim={dim}."
        )
    return {
        "joint": slice(0, dim),
        "spatial": slice(0, spatial_dim),
        "state": slice(spatial_dim, dim),
    }


def _derived_ot_seed(base_seed: int, time_index: int, space_index: int) -> int:
    return int(base_seed) + int(time_index) * 101 + int(space_index)


def _evaluate_result_with_fixed_ot_seed(
    result,
    *,
    rollout_seed: int,
    primary_seed: int,
    max_ot_points: int | None,
    ot_sampling_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recompute exact OT with one evaluation seed shared across rollouts."""

    rows: list[dict[str, Any]] = []
    mass_rows: list[dict[str, Any]] = []
    for condition, evaluation in result.evaluations.items():
        spaces = _space_slices(evaluation)
        times = tuple(float(value) for value in evaluation.time_points)
        source_time = times[0]
        initial_observed_count = int(
            np.asarray(evaluation.observed_points[source_time]).shape[0]
        )
        if initial_observed_count <= 0:
            raise ValueError("The observed source stage is empty.")

        for time_index, time_value in enumerate(times[1:], start=1):
            predicted = np.asarray(
                evaluation.predicted_points[time_value],
                dtype=np.float64,
            )
            observed = np.asarray(
                evaluation.observed_points[time_value],
                dtype=np.float64,
            )
            weights = np.asarray(
                evaluation.predicted_weights[time_value],
                dtype=np.float64,
            ).reshape(-1)
            if predicted.shape[0] != weights.shape[0]:
                raise ValueError(
                    f"{condition}/t={time_value:g}: weights do not align to points."
                )
            if not np.isfinite(weights).all():
                raise ValueError(
                    f"{condition}/t={time_value:g}: predicted weights contain "
                    "non-finite values."
                )
            predicted_mass = float(weights.sum())
            if not np.isfinite(predicted_mass) or predicted_mass <= 0:
                raise ValueError(
                    f"{condition}/t={time_value:g}: predicted mass must be "
                    f"finite and positive, got {predicted_mass!r}."
                )
            observed_mass_relative = float(
                observed.shape[0] / initial_observed_count
            )
            tmv_absolute = float(
                abs(predicted_mass - observed_mass_relative)
            )
            tmv = float(tmv_absolute / observed_mass_relative)
            normalized = weights / predicted_mass
            effective_particles = float(1.0 / np.square(normalized).sum())
            mass_rows.append(
                {
                    "condition": condition,
                    "rollout_seed": int(rollout_seed),
                    "is_primary_seed": bool(rollout_seed == primary_seed),
                    "source_time": source_time,
                    "time": time_value,
                    "n_predicted": int(predicted.shape[0]),
                    "n_observed": int(observed.shape[0]),
                    "predicted_mass": predicted_mass,
                    "observed_mass_relative": observed_mass_relative,
                    "tmv_absolute": tmv_absolute,
                    "tmv": tmv,
                    "normalized_weight_effective_particles": effective_particles,
                }
            )

            for space_index, (space, columns) in enumerate(spaces.items()):
                metric_seed = _derived_ot_seed(
                    ot_sampling_seed,
                    time_index,
                    space_index,
                )
                metric = compute_distribution_metrics(
                    predicted[:, columns],
                    observed[:, columns],
                    predicted_weights=weights,
                    max_ot_points=max_ot_points,
                    random_seed=metric_seed,
                )
                rows.append(
                    {
                        "condition": condition,
                        "rollout_seed": int(rollout_seed),
                        "is_primary_seed": bool(
                            rollout_seed == primary_seed
                        ),
                        "source_time": source_time,
                        "time": time_value,
                        "space": space,
                        "w1": float(metric["w1"]),
                        "w2": float(metric["w2"]),
                        "n_predicted": int(predicted.shape[0]),
                        "n_observed": int(observed.shape[0]),
                        "predicted_mass": predicted_mass,
                        "observed_mass_relative": observed_mass_relative,
                        "tmv_absolute": tmv_absolute,
                        "tmv": tmv,
                        "ot_predicted_points": int(
                            metric["ot_predicted_points"]
                        ),
                        "ot_observed_points": int(
                            metric["ot_observed_points"]
                        ),
                        "ot_sampling_seed": metric_seed,
                        "predicted_mass_policy": (
                            "native_unnormalised_for_tmv;"
                            "normalised_probability_for_ot"
                        ),
                        "observed_mass_policy": "uniform_empirical_for_ot",
                        "coordinate_policy": "native_aligned_unstandardized",
                    }
                )
    metrics = pd.DataFrame(rows)
    mass = pd.DataFrame(mass_rows)
    return metrics, mass


def _write_seed_failure(
    output_dir: Path,
    *,
    seed: int,
    conditions: Sequence[str],
    error: BaseException,
) -> tuple[Path, Path]:
    """Persist enough context to diagnose a failed, potentially costly seed."""

    failure_dir = output_dir / "failures"
    failure_dir.mkdir(parents=True, exist_ok=True)
    json_path = failure_dir / f"seed_{int(seed)}_failure.json"
    traceback_path = failure_dir / f"seed_{int(seed)}_traceback.log"
    trace = traceback.format_exc()
    traceback_path.write_text(trace, encoding="utf-8")
    record = {
        "status": "failed",
        "stage": "continuous_frozen_checkpoint_rollout",
        "rollout_seed": int(seed),
        "requested_conditions": [str(value) for value in conditions],
        "all_spatial_stress_condition_requested": "lr_gate_off" in conditions,
        "exception_type": type(error).__name__,
        "exception_message": str(error),
        "traceback": str(traceback_path),
        "note": (
            "The API evaluates the requested conditions in order. "
            "lr_gate_off is the all-spatial stress condition and can be much "
            "more resource-intensive because it admits every within-cutoff "
            "candidate edge. The traceback is preserved even though the "
            "overall report is incomplete."
        ),
    }
    json_path.write_text(
        json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return json_path, traceback_path


def _safe_percent_change(delta: pd.Series, reference: pd.Series) -> np.ndarray:
    numerator = delta.to_numpy(dtype=float)
    denominator = reference.to_numpy(dtype=float)
    return np.divide(
        100.0 * numerator,
        denominator,
        out=np.full_like(numerator, np.nan),
        where=np.abs(denominator) > np.finfo(float).eps,
    )


def _paired_deltas(
    metrics: pd.DataFrame,
    mass: pd.DataFrame,
    *,
    reference: str = "full",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["rollout_seed", "time", "space"]
    reference_metrics = (
        metrics.loc[
            metrics["condition"].eq(reference),
            keys + ["w1", "w2"],
        ]
        .rename(columns={"w1": "full_w1", "w2": "full_w2"})
        .copy()
    )
    if reference_metrics.duplicated(keys).any():
        raise ValueError("Full metrics contain duplicate paired keys.")
    paired = metrics.merge(
        reference_metrics,
        on=keys,
        how="left",
        validate="many_to_one",
    )
    if paired[["full_w1", "full_w2"]].isna().any().any():
        raise ValueError("Full W1/W2 reference grid is incomplete.")
    for metric in ("w1", "w2"):
        delta_name = f"{metric}_delta_vs_full"
        paired[delta_name] = paired[metric] - paired[f"full_{metric}"]
        paired[f"{metric}_percent_change_vs_full"] = _safe_percent_change(
            paired[delta_name],
            paired[f"full_{metric}"],
        )
        paired[f"{metric}_lower_than_full"] = paired[delta_name] < 0

    mass_keys = ["rollout_seed", "time"]
    reference_mass = (
        mass.loc[mass["condition"].eq(reference), mass_keys + ["tmv"]]
        .rename(columns={"tmv": "full_tmv"})
        .copy()
    )
    if reference_mass.duplicated(mass_keys).any():
        raise ValueError("Full mass metrics contain duplicate paired keys.")
    paired_mass = mass.merge(
        reference_mass,
        on=mass_keys,
        how="left",
        validate="many_to_one",
    )
    if paired_mass["full_tmv"].isna().any():
        raise ValueError("Full TMV reference grid is incomplete.")
    paired_mass["tmv_delta_vs_full"] = (
        paired_mass["tmv"] - paired_mass["full_tmv"]
    )
    paired_mass["tmv_percent_change_vs_full"] = _safe_percent_change(
        paired_mass["tmv_delta_vs_full"],
        paired_mass["full_tmv"],
    )
    paired_mass["tmv_lower_than_full"] = (
        paired_mass["tmv_delta_vs_full"] < 0
    )
    return paired, paired_mass


def _sensitivity_summary(
    metrics: pd.DataFrame,
    mass: pd.DataFrame,
) -> pd.DataFrame:
    long = metrics.melt(
        id_vars=[
            "condition",
            "rollout_seed",
            "is_primary_seed",
            "time",
            "space",
        ],
        value_vars=["w1", "w2"],
        var_name="metric",
        value_name="value",
    )
    mass_long = mass[
        ["condition", "rollout_seed", "is_primary_seed", "time", "tmv"]
    ].copy()
    mass_long["space"] = "mass"
    mass_long["metric"] = "tmv"
    mass_long = mass_long.rename(columns={"tmv": "value"})
    long = pd.concat([long, mass_long[long.columns]], ignore_index=True)

    def aggregate(frame: pd.DataFrame, *, time_label: str | None) -> pd.DataFrame:
        records: list[dict[str, Any]] = []
        for (condition, time, space, metric), group in frame.groupby(
            ["condition", "time", "space", "metric"],
            sort=False,
            observed=True,
        ):
            primary = group.loc[group["is_primary_seed"], "value"]
            values = group["value"].to_numpy(dtype=float)
            records.append(
                {
                    "condition": condition,
                    "time": str(time) if time_label is None else time_label,
                    "space": space,
                    "metric": metric,
                    "primary_value": (
                        float(primary.iloc[0]) if len(primary) == 1 else np.nan
                    ),
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "sd": (
                        float(np.std(values, ddof=1))
                        if len(values) > 1
                        else 0.0
                    ),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "n_rollout_seeds": int(len(values)),
                }
            )
        return pd.DataFrame(records)

    per_time = aggregate(long, time_label=None)
    seed_means = (
        long.groupby(
            ["condition", "rollout_seed", "is_primary_seed", "space", "metric"],
            sort=False,
            observed=True,
        )["value"]
        .mean()
        .reset_index()
    )
    seed_means["time"] = "mean_t1_to_t4"
    across_time = aggregate(seed_means, time_label="mean_t1_to_t4")
    return pd.concat([per_time, across_time], ignore_index=True)


def _paired_delta_summary(
    paired: pd.DataFrame,
    paired_mass: pd.DataFrame,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for metric in ("w1", "w2"):
        delta = f"{metric}_delta_vs_full"
        percent = f"{metric}_percent_change_vs_full"
        base = paired[
            [
                "condition",
                "rollout_seed",
                "time",
                "space",
                delta,
                percent,
            ]
        ].copy()
        base["metric"] = metric
        base = base.rename(
            columns={delta: "delta", percent: "percent_change"}
        )
        for (condition, time, space), group in base.groupby(
            ["condition", "time", "space"],
            sort=False,
            observed=True,
        ):
            values = group["delta"].to_numpy(dtype=float)
            percents = group["percent_change"].to_numpy(dtype=float)
            records.append(
                {
                    "condition": condition,
                    "time": str(time),
                    "space": space,
                    "metric": metric,
                    "mean_delta_vs_full": float(np.mean(values)),
                    "median_delta_vs_full": float(np.median(values)),
                    "min_delta_vs_full": float(np.min(values)),
                    "max_delta_vs_full": float(np.max(values)),
                    "sd_delta_vs_full": (
                        float(np.std(values, ddof=1))
                        if len(values) > 1
                        else 0.0
                    ),
                    "mean_percent_change_vs_full": float(
                        np.nanmean(percents)
                    ),
                    "median_percent_change_vs_full": float(
                        np.nanmedian(percents)
                    ),
                    "n_lower_than_full": int(np.sum(values < 0)),
                    "n_higher_than_full": int(np.sum(values > 0)),
                    "n_rollout_seeds": int(len(values)),
                }
            )

        seed_means = (
            base.groupby(
                ["condition", "rollout_seed", "space", "metric"],
                sort=False,
                observed=True,
            )[["delta", "percent_change"]]
            .mean()
            .reset_index()
        )
        for (condition, space), group in seed_means.groupby(
            ["condition", "space"],
            sort=False,
            observed=True,
        ):
            values = group["delta"].to_numpy(dtype=float)
            percents = group["percent_change"].to_numpy(dtype=float)
            records.append(
                {
                    "condition": condition,
                    "time": "mean_t1_to_t4",
                    "space": space,
                    "metric": metric,
                    "mean_delta_vs_full": float(np.mean(values)),
                    "median_delta_vs_full": float(np.median(values)),
                    "min_delta_vs_full": float(np.min(values)),
                    "max_delta_vs_full": float(np.max(values)),
                    "sd_delta_vs_full": (
                        float(np.std(values, ddof=1))
                        if len(values) > 1
                        else 0.0
                    ),
                    "mean_percent_change_vs_full": float(
                        np.nanmean(percents)
                    ),
                    "median_percent_change_vs_full": float(
                        np.nanmedian(percents)
                    ),
                    "n_lower_than_full": int(np.sum(values < 0)),
                    "n_higher_than_full": int(np.sum(values > 0)),
                    "n_rollout_seeds": int(len(values)),
                }
            )

    mass_base = paired_mass[
        [
            "condition",
            "rollout_seed",
            "time",
            "tmv_delta_vs_full",
            "tmv_percent_change_vs_full",
        ]
    ].rename(
        columns={
            "tmv_delta_vs_full": "delta",
            "tmv_percent_change_vs_full": "percent_change",
        }
    )
    for (condition, time), group in mass_base.groupby(
        ["condition", "time"],
        sort=False,
        observed=True,
    ):
        values = group["delta"].to_numpy(dtype=float)
        records.append(
            {
                "condition": condition,
                "time": str(time),
                "space": "mass",
                "metric": "tmv",
                "mean_delta_vs_full": float(np.mean(values)),
                "median_delta_vs_full": float(np.median(values)),
                "min_delta_vs_full": float(np.min(values)),
                "max_delta_vs_full": float(np.max(values)),
                "sd_delta_vs_full": (
                    float(np.std(values, ddof=1))
                    if len(values) > 1
                    else 0.0
                ),
                "mean_percent_change_vs_full": float(
                    np.nanmean(group["percent_change"].to_numpy(dtype=float))
                ),
                "median_percent_change_vs_full": float(
                    np.nanmedian(
                        group["percent_change"].to_numpy(dtype=float)
                    )
                ),
                "n_lower_than_full": int(np.sum(values < 0)),
                "n_higher_than_full": int(np.sum(values > 0)),
                "n_rollout_seeds": int(len(values)),
            }
        )
    mass_seed_means = (
        mass_base.groupby(["condition", "rollout_seed"], sort=False)[
            ["delta", "percent_change"]
        ]
        .mean()
        .reset_index()
    )
    for condition, group in mass_seed_means.groupby("condition", sort=False):
        values = group["delta"].to_numpy(dtype=float)
        records.append(
            {
                "condition": condition,
                "time": "mean_t1_to_t4",
                "space": "mass",
                "metric": "tmv",
                "mean_delta_vs_full": float(np.mean(values)),
                "median_delta_vs_full": float(np.median(values)),
                "min_delta_vs_full": float(np.min(values)),
                "max_delta_vs_full": float(np.max(values)),
                "sd_delta_vs_full": (
                    float(np.std(values, ddof=1))
                    if len(values) > 1
                    else 0.0
                ),
                "mean_percent_change_vs_full": float(
                    np.nanmean(group["percent_change"].to_numpy(dtype=float))
                ),
                "median_percent_change_vs_full": float(
                    np.nanmedian(
                        group["percent_change"].to_numpy(dtype=float)
                    )
                ),
                "n_lower_than_full": int(np.sum(values < 0)),
                "n_higher_than_full": int(np.sum(values > 0)),
                "n_rollout_seeds": int(len(values)),
            }
        )
    return pd.DataFrame(records)


def _save_figure(fig: plt.Figure, root: Path, stem: str) -> list[Path]:
    root.mkdir(parents=True, exist_ok=True)
    paths = [root / f"{stem}.png", root / f"{stem}.pdf"]
    fig.savefig(paths[0], dpi=260, bbox_inches="tight", facecolor="white")
    fig.savefig(paths[1], bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return paths


def _plot_wasserstein_curves(
    metrics: pd.DataFrame,
    *,
    conditions: Sequence[str],
    output_dir: Path,
    stem: str,
    log_scale: bool,
    title: str,
    central_tendency: str = "mean",
) -> list[Path]:
    if central_tendency not in {"mean", "median"}:
        raise ValueError(
            "central_tendency must be either 'mean' or 'median'."
        )
    fig, axes = plt.subplots(2, 3, figsize=(14.2, 7.4), squeeze=False)
    for row, metric in enumerate(("w1", "w2")):
        for column, space in enumerate(SPACE_ORDER):
            axis = axes[row, column]
            subset = metrics.loc[metrics["space"].eq(space)]
            for condition in conditions:
                condition_table = subset.loc[
                    subset["condition"].eq(condition)
                ]
                summary = condition_table.groupby(
                    "time", sort=True
                )[metric].agg(["mean", "median", "std"]).reset_index()
                if summary.empty:
                    continue
                summary["std"] = summary["std"].fillna(0.0)
                x = summary["time"].to_numpy(dtype=float)
                center = summary[central_tendency].to_numpy(dtype=float)
                sd = summary["std"].to_numpy(dtype=float)
                color = _condition_color(condition)
                axis.plot(
                    x,
                    center,
                    marker="o",
                    linewidth=2.0,
                    color=color,
                    label=(
                        _condition_label(condition)
                        + (
                            " (median)"
                            if central_tendency == "median"
                            else ""
                        )
                    ),
                )
                if not log_scale and central_tendency == "mean":
                    axis.fill_between(
                        x,
                        np.maximum(center - sd, 0.0),
                        center + sd,
                        color=color,
                        alpha=0.15,
                        linewidth=0,
                    )
                for seed, seed_table in condition_table.groupby(
                    "rollout_seed",
                    sort=False,
                ):
                    axis.plot(
                        seed_table["time"],
                        seed_table[metric],
                        color=color,
                        alpha=0.14,
                        linewidth=0.7,
                    )
            if log_scale:
                axis.set_yscale("log")
            axis.set_title(f"{SPACE_LABELS[space]} — {metric.upper()}")
            axis.set_xlabel("Observed target stage")
            axis.set_ylabel(f"{metric.upper()} (lower is better)")
            axis.set_xticks(sorted(subset["time"].unique()))
            axis.grid(alpha=0.22)
            axis.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=max(1, len(conditions)),
            frameon=False,
            bbox_to_anchor=(0.5, 1.02),
        )
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.065)
    fig.tight_layout()
    return _save_figure(fig, output_dir, stem)


def _plot_tmv(
    mass: pd.DataFrame,
    *,
    conditions: Sequence[str],
    output_dir: Path,
) -> list[Path]:
    main_conditions = [
        name for name in conditions if name != "lr_gate_off"
    ]
    if not main_conditions:
        main_conditions = list(conditions)
    panels = [
        ("Primary comparison (mean ± SD)", main_conditions, False),
        (
            "All-spatial stress test (median + seeds)",
            list(conditions),
            True,
        ),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.4), squeeze=False)
    for axis, (title, active, stress) in zip(axes[0], panels):
        for condition in active:
            table = mass.loc[mass["condition"].eq(condition)]
            summary = table.groupby("time", sort=True)["tmv"].agg(
                ["mean", "median", "std"]
            ).reset_index()
            if summary.empty:
                continue
            summary["std"] = summary["std"].fillna(0.0)
            x = summary["time"].to_numpy(dtype=float)
            center_column = "median" if stress else "mean"
            center = summary[center_column].to_numpy(dtype=float)
            sd = summary["std"].to_numpy(dtype=float)
            color = _condition_color(condition)
            axis.plot(
                x,
                center,
                marker="o",
                linewidth=2,
                color=color,
                label=(
                    _condition_label(condition)
                    + (" (median)" if stress else "")
                ),
            )
            if not stress:
                axis.fill_between(
                    x,
                    np.maximum(center - sd, 0.0),
                    center + sd,
                    color=color,
                    alpha=0.15,
                    linewidth=0,
                )
            else:
                for _, seed_table in table.groupby(
                    "rollout_seed",
                    sort=False,
                ):
                    axis.plot(
                        seed_table["time"],
                        seed_table["tmv"],
                        color=color,
                        alpha=0.14,
                        linewidth=0.7,
                    )
        if stress:
            axis.set_yscale("symlog", linthresh=1e-4)
        axis.set_title(title)
        axis.set_xlabel("Observed target stage")
        axis.set_ylabel("TMV (lower is better)")
        axis.set_xticks(sorted(mass["time"].unique()))
        axis.grid(alpha=0.22)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False, fontsize=8)
    fig.suptitle(
        "Whole-trajectory total-mass variation",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    return _save_figure(fig, output_dir, "tmv_time_curves")


def _stress_extreme_markdown(
    paired: pd.DataFrame | None,
    paired_mass: pd.DataFrame | None,
) -> str:
    if paired is None:
        return ""
    stress = paired.loc[
        paired["condition"].eq("lr_gate_off")
        & paired["space"].eq("joint")
    ].copy()
    if stress.empty:
        return ""
    candidates: list[tuple[float, str, Any]] = []
    for metric in ("w1", "w2"):
        delta_column = f"{metric}_delta_vs_full"
        for index, value in stress[delta_column].items():
            candidates.append((abs(float(value)), metric, index))
    _, metric, index = max(candidates, key=lambda item: item[0])
    row = stress.loc[index]
    value_column = metric
    full_column = f"full_{metric}"
    delta_column = f"{metric}_delta_vs_full"
    same_endpoint = stress.loc[stress["time"].eq(row["time"])]
    stable = same_endpoint.loc[
        ~same_endpoint["rollout_seed"].eq(row["rollout_seed"]),
        value_column,
    ].to_numpy(dtype=float)
    stable_text = (
        "没有其他 seed 可比较"
        if stable.size == 0
        else f"其余 seeds 为 [{np.min(stable):.5g}, {np.max(stable):.5g}]"
    )
    mass_text = ""
    if paired_mass is not None:
        mass_match = paired_mass.loc[
            paired_mass["condition"].eq("lr_gate_off")
            & paired_mass["rollout_seed"].eq(row["rollout_seed"])
            & paired_mass["time"].eq(row["time"])
        ]
        if len(mass_match) == 1:
            mass_row = mass_match.iloc[0]
            mass_text = (
                f"；同一 endpoint 的 TMV={float(mass_row['tmv']):.5g}"
                f"（Full={float(mass_row['full_tmv']):.5g}）"
            )
    return (
        "\n## All-spatial extreme-rollout audit\n\n"
        "为避免中位数掩盖稳定性问题，自动报告最大的 joint W1/W2 paired "
        "deviation：rollout seed "
        f"`{int(row['rollout_seed'])}`、t=`{float(row['time']):g}`、"
        f"{metric.upper()}=`{float(row[value_column]):.5g}`"
        f"（Full=`{float(row[full_column]):.5g}`，"
        f"Δ=`{float(row[delta_column]):+.5g}`）；{stable_text}{mass_text}。"
        "该 seed 保留在全部汇总中，不作事后删除。\n"
    )


def _interpretation_markdown(
    delta_summary: pd.DataFrame,
    *,
    conditions: Sequence[str],
    seeds: Sequence[int],
    primary_seed: int,
    ot_sampling_seed: int,
    paired: pd.DataFrame | None = None,
    paired_mass: pd.DataFrame | None = None,
) -> str:
    aggregate = delta_summary.loc[
        delta_summary["time"].eq("mean_t1_to_t4")
        & ~delta_summary["condition"].eq("full")
    ].copy()
    table_lines = [
        "| Condition | Space | Metric | Mean Δ | Median Δ | Seed-mean range | Median % | Higher in seeds |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate.itertuples(index=False):
        median_percent = (
            "—"
            if str(row.metric) == "tmv"
            else f"{float(row.median_percent_change_vs_full):+.2f}%"
        )
        table_lines.append(
            "| "
            f"{_condition_label(str(row.condition))} | {row.space} | "
            f"{str(row.metric).upper()} | {float(row.mean_delta_vs_full):+.5g} | "
            f"{float(row.median_delta_vs_full):+.5g} | "
            f"[{float(row.min_delta_vs_full):+.5g}, "
            f"{float(row.max_delta_vs_full):+.5g}] | "
            f"{median_percent} | "
            f"{int(row.n_higher_than_full)}/{int(row.n_rollout_seeds)} |"
        )
    table = "\n".join(table_lines)
    condition_text = ", ".join(_condition_label(value) for value in conditions)
    stress_audit = _stress_extreme_markdown(paired, paired_mass)
    return f"""# t0→t4 same-checkpoint functional ablation

## 这次到底算的是什么

同一个训练完成的 Full checkpoint 从真实最早时间点只初始化一次，随后连续
rollout 到最后一个时间点。真实中间时间点只用于评估，没有用真实 t1/t2/t3
重新启动模型。条件为：{condition_text}。

- `Interaction OFF`：推理时只去掉 interaction velocity；drift、score、growth、
  checkpoint 和 SDE 噪声都保留。
- `All-spatial gate`：保留 interaction GNN，但绕过训练好的 LR-informed gate，
  让空间 cutoff 内的候选边全部通过。它改变了边密度，不能称为纯 LR identity
  ablation，也不能简称为 “LR OFF”。

这个 runner 评价一个固定 checkpoint 的 inference-time functional dependence；
该 checkpoint 是 full-data、held-out 还是其他训练协议，属于外部训练 metadata，
不能由本报告自动推断。它从 source time 选取的细胞出发，始终追踪同一批
non-split weighted particles；后续增殖由 growth mass 表示，不是增加粒子数。

## 指标定义

每个生成细胞携带模型原生的未归一化 growth mass `w_i`。

- W1/W2：先将 `w_i` 归一化为概率，再与真实细胞的均匀经验分布做 exact EMD。
  因此 W1/W2 评价“质量在状态/空间中的位置”，不评价总质量是否正确。
- TMV：`abs(sum(w) - N_t/N_0) / (N_t/N_0)`，单独评价总质量。
  当 Full TMV 接近零时，百分比变化会被小分母放大，因此 TMV 应优先报告绝对
  delta，而不是把百分比作为 headline。
- Joint 是未经标准化的 `[2D aligned spatial, 50D PCA state]`。它通常受 PCA
  state 主导，因此必须同时报告 State 和 Spatial，不能把 Joint 当成两块等权。
- 三个空间单位不同，Joint/State/Spatial 的绝对数值不能横向比较。

## 随机性

- checkpoint 的训练 seed 没有变化；这不是多次独立训练。
- rollout seeds：`{list(seeds)}`，primary seed：`{primary_seed}`。
- primary seed 只用于保存一套可逐粒子检查的 trajectory；主比较曲线仍汇总所有
  rollout seeds，文件名中的 `primary` 指不含 all-spatial stress condition 的
  主比较 panel。
- 同一个 rollout seed 内，各条件使用相同 Brownian increments，所以可做 paired
  delta。
- exact-OT support seed 固定为 `{ot_sampling_seed}`（并按 time/space 做确定性
  offset），不随 rollout seed 改变。误差条因此主要反映 SDE rollout 敏感性，
  不是把 OT 子采样误差混进去。

## t1–t4 平均结果

负的 Δ 或百分比表示该 counterfactual 的数值低于 Full；这只是终点分布更接近，
不自动等于该生物学模块“更好”或“更坏”。Mean、median 和 seed-mean range
必须一起读；若 all-spatial 的均值被单个发散 seed 支配，应以中位数描述典型
rollout，并把该发散明确报告为稳定性结果，不能只报均值或删除该 seed。

{table}
{stress_audit}

## 如何解读

1. 先看逐时间点曲线，判断差异是持续存在，还是只由 t4 的累计误差驱动。
   主比较粗线是 mean；all-spatial stress 图粗线是 median，淡色细线是每个 seed。
2. 再看 paired delta 及 `Higher in seeds`。如果这里显示
   `{len(seeds)}/{len(seeds)}`，表示 counterfactual 误差在所有 rollout seeds
   中都高于 Full；全 seeds 同方向比单个 seed 的 barplot 更可靠，但这些 seed
   仍只是固定 checkpoint 的随机 rollout，不是训练重复。
3. All-spatial 若显著恶化，最多说明训练好的 gate 对限制过多 message passing 和
   稳定动力学重要；由于边数同时改变，不能据此单独声称 LR identity 因果成立。
4. 这套 whole-trajectory weighted-SDE 结果是 same-checkpoint functional
   sensitivity。重新训练的 no-interaction/no-LR-prior 是 architecture ablation，
   回答的问题不同，两者应并列而不能混成同一误差条。

## 旧 t3→t4 结果的正式位置

旧结果应标为 **late-stage deterministic fixed-cohort sensitivity
(t3→t4 only)**。它没有 growth、噪声、resampling 或真实组成变化，只适合说明
晚期局部开关会怎样改变固定细胞；不得作为 t0→t4 主性能 benchmark，也不得把其
绝对 W1/W2 与本报告混算。旧局部结果与连续结果方向不同并非矛盾：前者从
measured t3 重新初始化并绕过 t0→t3 的累计误差，后者测量从 t0 开始累计的
whole-trajectory distribution performance。
"""


def _validate_seed_result(
    result,
    *,
    expected_conditions: Sequence[str],
    expected_seed: int,
) -> None:
    manifest = result.manifest
    if manifest.get("analysis") != (
        "continuous_frozen_checkpoint_functional_ablation"
    ):
        raise ValueError("Unexpected continuous-ablation API manifest.")
    if list(manifest.get("condition_order", ())) != list(expected_conditions):
        raise ValueError("Continuous-ablation condition order changed.")
    controls = manifest.get("matched_controls", {})
    if not bool(controls.get("common_brownian_noise")):
        raise ValueError("The API did not certify common Brownian noise.")
    if int(controls.get("brownian_noise_seed")) != int(expected_seed):
        raise ValueError("The API Brownian seed differs from the rollout seed.")
    if bool(manifest.get("observed_intermediate_restart", True)):
        raise ValueError("The API restarted from an intermediate observation.")


def run(
    args: argparse.Namespace,
    *,
    command_argv: Sequence[str],
) -> dict[str, Any]:
    adata_path = args.adata.expanduser().resolve()
    model_dir = args.model_dir.expanduser().resolve()
    if not adata_path.is_file():
        raise FileNotFoundError(adata_path)
    if not model_dir.is_dir():
        raise FileNotFoundError(model_dir)
    output_dir = _prepare_output_dir(
        args.output_dir,
        overwrite=bool(args.overwrite),
    )
    seeds = _ordered_unique_ints(args.seeds)
    primary_seed = int(args.primary_seed)
    if primary_seed not in seeds:
        raise ValueError("--primary-seed must be included in --seeds.")
    conditions = tuple(str(value) for value in args.conditions)
    if not conditions or len(set(conditions)) != len(conditions):
        raise ValueError("--conditions must be non-empty and unique.")
    if "full" not in conditions:
        raise ValueError("--conditions must include 'full'.")
    if int(args.max_ot_points) <= 0:
        raise ValueError("--max-ot-points must be positive.")

    import scanpy as sc

    adata = sc.read_h5ad(adata_path)
    if args.obsm_key not in adata.obsm:
        raise KeyError(f"adata.obsm[{args.obsm_key!r}] is missing.")
    if args.spatial_key not in adata.obsm:
        raise KeyError(f"adata.obsm[{args.spatial_key!r}] is missing.")
    state = np.asarray(adata.obsm[args.obsm_key])
    spatial = np.asarray(adata.obsm[args.spatial_key])
    if state.ndim != 2 or spatial.ndim != 2 or state.shape[0] != spatial.shape[0]:
        raise ValueError("State and spatial matrices are not row-aligned.")
    joint_dim = int(state.shape[1] + spatial.shape[1])
    loaded = cb.tl.load_dynamical_model_from_dir(
        model_dir,
        dim=joint_dim,
        device=str(args.device),
    )

    seed_manifest_dir = output_dir / "seed_manifests"
    seed_manifest_dir.mkdir(parents=True, exist_ok=True)
    metric_tables: list[pd.DataFrame] = []
    mass_tables: list[pd.DataFrame] = []
    seed_manifest_paths: list[Path] = []
    primary_paths: Mapping[str, Path] | None = None
    primary_api_manifest: Mapping[str, Any] | None = None

    for seed in seeds:
        try:
            result = cb.tl.run_continuous_frozen_checkpoint_ablations(
                adata,
                loaded,
                time_points=args.time_points,
                n_samples=int(args.n_samples),
                dt=float(args.dt),
                sigma=float(args.sigma),
                interaction_m=int(args.interaction_m),
                conditions=conditions,
                max_ot_points=int(args.max_ot_points),
                structure_max_points=int(args.structure_max_points),
                device=str(args.device),
                time_key=args.time_key,
                obsm_key=args.obsm_key,
                spatial_key=args.spatial_key,
                concat_spatial=True,
                random_seed=int(seed),
                verbose=not bool(args.quiet),
            )
        except Exception as error:
            failure_json, failure_traceback = _write_seed_failure(
                output_dir,
                seed=seed,
                conditions=conditions,
                error=error,
            )
            raise RuntimeError(
                f"Continuous rollout failed for seed {seed}; diagnostics were "
                f"saved to {failure_json} and {failure_traceback}. If "
                "'lr_gate_off' was requested, remember that it is the "
                "resource-intensive All-spatial stress condition."
            ) from error
        _validate_seed_result(
            result,
            expected_conditions=conditions,
            expected_seed=seed,
        )
        seed_manifest_path = seed_manifest_dir / f"seed_{seed}.json"
        seed_manifest_path.write_text(
            json.dumps(
                result.manifest,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        seed_manifest_paths.append(seed_manifest_path)
        metrics, mass = _evaluate_result_with_fixed_ot_seed(
            result,
            rollout_seed=seed,
            primary_seed=primary_seed,
            max_ot_points=int(args.max_ot_points),
            ot_sampling_seed=int(args.ot_sampling_seed),
        )
        metric_tables.append(metrics)
        mass_tables.append(mass)
        if seed == primary_seed:
            primary_root = output_dir / f"primary_seed_{primary_seed}"
            primary_paths = (
                cb.tl.save_continuous_frozen_checkpoint_ablation_result(
                    result,
                    primary_root,
                )
            )
            primary_api_manifest = result.manifest

    if primary_paths is None or primary_api_manifest is None:
        raise RuntimeError("Primary-seed result was not saved.")
    metrics = pd.concat(metric_tables, ignore_index=True)
    mass = pd.concat(mass_tables, ignore_index=True)
    paired, paired_mass = _paired_deltas(metrics, mass)
    sensitivity = _sensitivity_summary(metrics, mass)
    delta_summary = _paired_delta_summary(paired, paired_mass)

    paths: dict[str, Path] = {
        "metrics": output_dir / "metrics_all_rollout_seeds.csv",
        "mass": output_dir / "mass_metrics_all_rollout_seeds.csv",
        "paired_wasserstein": (
            output_dir / "paired_wasserstein_deltas_vs_full.csv"
        ),
        "paired_tmv": output_dir / "paired_tmv_deltas_vs_full.csv",
        "sensitivity": output_dir / "sampling_sensitivity_summary.csv",
        "paired_summary": output_dir / "paired_delta_summary.csv",
        "interpretation": output_dir / "INTERPRETATION_CN.md",
    }
    metrics.to_csv(paths["metrics"], index=False, float_format="%.12g")
    mass.to_csv(paths["mass"], index=False, float_format="%.12g")
    paired.to_csv(
        paths["paired_wasserstein"],
        index=False,
        float_format="%.12g",
    )
    paired_mass.to_csv(
        paths["paired_tmv"],
        index=False,
        float_format="%.12g",
    )
    sensitivity.to_csv(
        paths["sensitivity"],
        index=False,
        float_format="%.12g",
    )
    delta_summary.to_csv(
        paths["paired_summary"],
        index=False,
        float_format="%.12g",
    )

    main_conditions = [
        condition for condition in conditions if condition != "lr_gate_off"
    ]
    if not main_conditions:
        main_conditions = list(conditions)
    main_figures = _plot_wasserstein_curves(
        metrics,
        conditions=main_conditions,
        output_dir=output_dir,
        stem="w1_w2_time_curves_primary",
        log_scale=False,
        title="Same-checkpoint t0→t4 functional ablation",
    )
    stress_figures = _plot_wasserstein_curves(
        metrics,
        conditions=conditions,
        output_dir=output_dir,
        stem="w1_w2_time_curves_all_spatial_stress_log",
        log_scale=True,
        title=(
            "All-spatial gate stress test — median + individual seeds "
            "(log scale)"
        ),
        central_tendency="median",
    )
    tmv_figures = _plot_tmv(
        mass,
        conditions=conditions,
        output_dir=output_dir,
    )
    paths.update(
        {
            "w1_w2_primary_png": main_figures[0],
            "w1_w2_primary_pdf": main_figures[1],
            "w1_w2_stress_png": stress_figures[0],
            "w1_w2_stress_pdf": stress_figures[1],
            "tmv_png": tmv_figures[0],
            "tmv_pdf": tmv_figures[1],
        }
    )
    paths["interpretation"].write_text(
        _interpretation_markdown(
            delta_summary,
            conditions=conditions,
            seeds=seeds,
            primary_seed=primary_seed,
            ot_sampling_seed=int(args.ot_sampling_seed),
            paired=paired,
            paired_mass=paired_mass,
        ),
        encoding="utf-8",
    )

    manifest_path = output_dir / "run_manifest.json"
    manifest = {
        "schema_version": 1,
        "analysis": (
            "continuous_frozen_checkpoint_ablation_multiseed_report"
        ),
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
            "adata": _file_record(adata_path),
            "model_dir": str(model_dir),
            "checkpoint": dict(primary_api_manifest["checkpoint"]),
        },
        "parameters": {
            "conditions": list(conditions),
            "rollout_seeds": list(seeds),
            "primary_seed": primary_seed,
            "ot_sampling_seed_base": int(args.ot_sampling_seed),
            "ot_sampling_seed_policy": (
                "fixed across conditions and rollout seeds; "
                "base + 101*time_index + space_index"
            ),
            "n_samples_cap": int(args.n_samples),
            "dt": float(args.dt),
            "sigma": float(args.sigma),
            "interaction_m": int(args.interaction_m),
            "max_ot_points": int(args.max_ot_points),
            "time_points": (
                None
                if args.time_points is None
                else [float(value) for value in args.time_points]
            ),
            "device": str(args.device),
        },
        "design": {
            "source": "earliest observed time only",
            "observed_intermediate_restart": False,
            "continuous_non_split_weighted_sde": True,
            "common_brownian_noise_within_seed": True,
            "multiple_training_seeds": False,
            "seed_interpretation": (
                "Across-seed SD measures stochastic rollout sensitivity of "
                "one fixed trained checkpoint."
            ),
            "spatial_warp": False,
        },
        "metrics": {
            "spaces": list(SPACE_ORDER),
            "coordinate_policy": "native_aligned_unstandardized",
            "joint_definition": (
                "raw concatenation of aligned spatial coordinates and PCA state"
            ),
            "w1": "exact EMD with Euclidean ground cost",
            "w2": "sqrt(exact EMD with squared-Euclidean ground cost)",
            "predicted_ot_mass": "native growth weights normalized to probability",
            "observed_ot_mass": "uniform empirical probability",
            "tmv": (
                "abs(sum(native weights)-N_t/N_0)/(N_t/N_0)"
            ),
            "lower_is_better": True,
        },
        "condition_scope": dict(primary_api_manifest["conditions"]),
        "lr_gate_caveat": dict(primary_api_manifest["lr_gate_null_scope"]),
        "legacy_t3_t4_status": (
            "late-stage deterministic fixed-cohort sensitivity only; "
            "not part of the whole-trajectory main performance benchmark"
        ),
        "seed_manifests": [
            _file_record(path) for path in seed_manifest_paths
        ],
        "primary_seed_outputs": {
            name: _file_record(path)
            for name, path in primary_paths.items()
        },
        "outputs": {
            name: _file_record(path) for name, path in paths.items()
        },
        "manifest": {
            "path": str(manifest_path),
            "self_hash_omitted": True,
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    args = _parser().parse_args(arguments)
    command = [sys.executable, str(SCRIPT_PATH), *arguments]
    manifest = run(args, command_argv=command)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "output_dir": manifest["manifest"]["path"],
                "conditions": manifest["parameters"]["conditions"],
                "rollout_seeds": manifest["parameters"]["rollout_seeds"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
