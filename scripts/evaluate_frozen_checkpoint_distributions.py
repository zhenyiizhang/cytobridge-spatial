#!/usr/bin/env python3
"""Evaluate saved frozen-checkpoint ablations with exact W1/W2.

This command compares each fixed-cohort endpoint with an observed target stage
in joint, spatial, and PCA-state spaces.  It also compares every inference-time
condition directly with the full rollout using a shared row subsample, because
the saved trajectories preserve cell identity.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import CytoBridge as cb  # noqa: E402


PRIMARY_FILENAME = "distribution_metrics_primary_seed.csv"
OBSERVED_FILENAME = "condition_vs_observed_metrics_all_seeds.csv"
FULL_FILENAME = "condition_vs_full_metrics_all_seeds.csv"
SENSITIVITY_FILENAME = "sampling_sensitivity_summary.csv"
OBSERVED_FIGURE = "condition_vs_observed_w1_w2"
FULL_FIGURE = "condition_vs_full_w1_w2"
INTERPRETATION_FILENAME = "INTERPRETATION_CN.md"
MANIFEST_FILENAME = "run_manifest.json"

_CONDITION_LABELS = {
    "full": "Full",
    "interaction_off": "Interaction OFF",
    "lr_gate_off": "All-spatial gate",
}
_CONDITION_COLORS = {
    "full": "#4C78A8",
    "interaction_off": "#F58518",
    "lr_gate_off": "#E45756",
}
_SPACE_LABELS = {
    "joint": "Joint (spatial + PCA state)",
    "spatial": "Spatial (2D)",
    "state": "PCA state",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--ablation-dir",
        required=True,
        type=Path,
        help="Directory containing full.npz and frozen-ablation condition NPZs.",
    )
    parser.add_argument("--adata", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=None,
        help=(
            "Condition names to evaluate. Default reads condition_order from "
            "the ablation manifest."
        ),
    )
    parser.add_argument("--full-condition", default="full")
    parser.add_argument("--target-time", type=float, required=True)
    parser.add_argument("--time-key", default=None)
    parser.add_argument("--obsm-key", default="X_latent")
    parser.add_argument("--spatial-key", default="spatial_aligned")
    parser.add_argument("--spatial-dim", type=int, default=2)
    parser.add_argument("--max-ot-points", type=int, default=1024)
    parser.add_argument("--primary-seed", type=int, default=42)
    parser.add_argument(
        "--sensitivity-seeds",
        type=int,
        nargs="+",
        default=[17, 23, 42, 101, 202],
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing non-empty output directory.",
    )
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
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
    if root.exists() and any(root.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory is non-empty: {root}. Use --overwrite explicitly."
        )
    root.mkdir(parents=True, exist_ok=True)
    return root


def _load_ablation_manifest(root: Path) -> tuple[dict[str, Any], Path]:
    path = root / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Frozen-ablation manifest is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8")), path


def _resolve_conditions(
    requested: Sequence[str] | None,
    manifest: Mapping[str, Any],
) -> list[str]:
    values = (
        list(requested)
        if requested is not None
        else list(manifest.get("condition_order", ()))
    )
    if not values:
        raise ValueError(
            "No conditions supplied and manifest.condition_order is empty."
        )
    if len(set(values)) != len(values):
        raise ValueError(f"Duplicate conditions: {values}.")
    return values


def _load_condition_endpoint(
    path: Path,
    *,
    target_time: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Condition trajectory is missing: {path}")
    with np.load(path) as saved:
        if "points" not in saved or "times" not in saved:
            raise KeyError(f"{path} must contain 'points' and 'times'.")
        points = np.asarray(saved["points"], dtype=np.float64)
        times = np.asarray(saved["times"], dtype=np.float64).reshape(-1)
    if points.ndim != 3 or len(points) != len(times):
        raise ValueError(
            f"{path} must store points[T,N,D] aligned to times[T]; got "
            f"{points.shape} and {times.shape}."
        )
    matches = np.flatnonzero(np.isclose(times, target_time, rtol=0.0, atol=1e-8))
    if len(matches) != 1:
        raise ValueError(
            f"{path} has {len(matches)} frames matching target time {target_time}; "
            f"available range is [{times.min()}, {times.max()}]."
        )
    frame_index = int(matches[0])
    return points[frame_index], {
        "target_frame_index": frame_index,
        "trajectory_start_time": float(times[0]),
        "trajectory_end_time": float(times[-1]),
        "trajectory_frames": int(len(times)),
        "endpoint_cells": int(points.shape[1]),
        "endpoint_dim": int(points.shape[2]),
    }


def _load_observed_endpoint(
    adata_path: Path,
    *,
    target_time: float,
    time_key: str | None,
    obsm_key: str,
    spatial_key: str,
) -> tuple[np.ndarray, str, int]:
    import scanpy as sc

    adata = sc.read_h5ad(adata_path)
    frame, resolved_time_key = cb.tl.adata_to_aligned_dataframe(
        adata,
        time_key=time_key,
        obsm_key=obsm_key,
        spatial_key=spatial_key,
        concat_spatial=True,
    )
    features = list(cb.tl.infer_feature_columns(frame))
    mask = np.isclose(
        frame["samples"].to_numpy(dtype=float),
        float(target_time),
        rtol=0.0,
        atol=1e-8,
    )
    if not np.any(mask):
        available = sorted(frame["samples"].unique().tolist())
        raise ValueError(
            f"No observed cells match target time {target_time}; available={available}."
        )
    return (
        frame.loc[mask, features].to_numpy(dtype=np.float64),
        str(resolved_time_key),
        int(mask.sum()),
    )


def _condition_label(name: str) -> str:
    return _CONDITION_LABELS.get(name, name.replace("_", " ").title())


def _condition_color(name: str) -> str:
    return _CONDITION_COLORS.get(name, "#777777")


def _metric_axis_label(metric: str) -> str:
    return "Exact EMD W1" if metric == "w1" else "Exact EMD W2"


def _save_figure(fig: plt.Figure, root: Path, stem: str) -> list[Path]:
    paths = [root / f"{stem}.png", root / f"{stem}.pdf"]
    fig.savefig(paths[0], dpi=240, bbox_inches="tight", facecolor="white")
    fig.savefig(paths[1], bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return paths


def _plot_condition_vs_observed(
    table: pd.DataFrame,
    *,
    condition_order: Sequence[str],
    full_condition: str,
    primary_seed: int,
    target_time: float,
    output_dir: Path,
) -> list[Path]:
    spaces = ["joint", "spatial", "state"]
    metrics = ["w1", "w2"]
    primary = table.loc[table["sampling_seed"] == int(primary_seed)].copy()
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.2), squeeze=False)
    x = np.arange(len(condition_order), dtype=float)

    for row, metric in enumerate(metrics):
        for column, space in enumerate(spaces):
            axis = axes[row, column]
            space_all = table.loc[table["space"] == space]
            space_primary = primary.loc[primary["space"] == space].set_index(
                "condition"
            )
            values = np.asarray(
                [space_primary.loc[name, metric] for name in condition_order],
                dtype=float,
            )
            axis.bar(
                x,
                values,
                width=0.66,
                color=[_condition_color(name) for name in condition_order],
                alpha=0.86,
                edgecolor="white",
                linewidth=0.8,
                zorder=2,
            )
            for index, name in enumerate(condition_order):
                samples = space_all.loc[
                    space_all["condition"] == name, metric
                ].to_numpy(dtype=float)
                offsets = (
                    np.linspace(-0.10, 0.10, len(samples))
                    if len(samples) > 1
                    else np.zeros(1)
                )
                axis.scatter(
                    index + offsets,
                    samples,
                    s=18,
                    facecolor="white",
                    edgecolor=_condition_color(name),
                    linewidth=0.9,
                    zorder=3,
                )
                pct = float(
                    space_primary.loc[
                        name,
                        f"{metric}_percent_change_vs_full",
                    ]
                )
                label = f"{values[index]:.3g}"
                if name != full_condition and np.isfinite(pct):
                    label += f"\n({pct:+.1f}%)"
                label_y = max(values[index], float(np.max(samples)))
                axis.annotate(
                    label,
                    (index, label_y),
                    xytext=(0, 7),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
            axis.set_xticks(x)
            axis.set_xticklabels(
                [_condition_label(name) for name in condition_order],
                rotation=18,
                ha="right",
            )
            axis.set_ylabel(_metric_axis_label(metric))
            axis.set_title(_SPACE_LABELS[space])
            axis.spines[["top", "right"]].set_visible(False)
            axis.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.7)
            axis.set_axisbelow(True)
            axis.margins(y=0.22)

    fig.suptitle(
        f"Frozen-checkpoint endpoint vs observed t{target_time:g}",
        fontsize=14,
        y=0.995,
    )
    fig.text(
        0.5,
        0.005,
        (
            f"Bars: prespecified seed {primary_seed}; open points: fixed-seed "
            "sampling sensitivity. Percentages are changes relative to Full "
            "(lower is better). Uniform empirical mass."
        ),
        ha="center",
        va="bottom",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.055, 1, 0.96))
    return _save_figure(fig, output_dir, OBSERVED_FIGURE)


def _plot_condition_vs_full(
    table: pd.DataFrame,
    *,
    condition_order: Sequence[str],
    full_condition: str,
    primary_seed: int,
    output_dir: Path,
) -> list[Path]:
    conditions = [name for name in condition_order if name != full_condition]
    spaces = ["joint", "spatial", "state"]
    metrics = ["w1", "w2"]
    primary = table.loc[table["sampling_seed"] == int(primary_seed)].copy()
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 7.0), squeeze=False)
    x = np.arange(len(conditions), dtype=float)

    for row, metric in enumerate(metrics):
        bound_column = f"identity_coupling_{metric}_upper_bound"
        for column, space in enumerate(spaces):
            axis = axes[row, column]
            space_all = table.loc[table["space"] == space]
            space_primary = primary.loc[primary["space"] == space].set_index(
                "condition"
            )
            values = np.asarray(
                [space_primary.loc[name, metric] for name in conditions],
                dtype=float,
            )
            bounds = np.asarray(
                [space_primary.loc[name, bound_column] for name in conditions],
                dtype=float,
            )
            axis.bar(
                x,
                values,
                width=0.62,
                color=[_condition_color(name) for name in conditions],
                alpha=0.86,
                edgecolor="white",
                linewidth=0.8,
                label="Exact OT",
                zorder=2,
            )
            axis.scatter(
                x,
                bounds,
                marker="D",
                s=38,
                facecolor="white",
                edgecolor="#222222",
                linewidth=1.0,
                label="Identity-coupling bound",
                zorder=4,
            )
            for index, name in enumerate(conditions):
                samples = space_all.loc[
                    space_all["condition"] == name, metric
                ].to_numpy(dtype=float)
                offsets = (
                    np.linspace(-0.08, 0.08, len(samples))
                    if len(samples) > 1
                    else np.zeros(1)
                )
                axis.scatter(
                    index + offsets,
                    samples,
                    s=18,
                    facecolor="white",
                    edgecolor=_condition_color(name),
                    linewidth=0.9,
                    zorder=3,
                )
                label_y = max(
                    values[index],
                    bounds[index],
                    float(np.max(samples)),
                )
                axis.annotate(
                    f"{values[index]:.3g}",
                    (index, label_y),
                    xytext=(0, 7),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
            axis.set_xticks(x)
            axis.set_xticklabels(
                [_condition_label(name) for name in conditions],
                rotation=15,
                ha="right",
            )
            axis.set_ylabel(_metric_axis_label(metric))
            axis.set_title(_SPACE_LABELS[space])
            axis.spines[["top", "right"]].set_visible(False)
            axis.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.7)
            axis.set_axisbelow(True)
            axis.margins(y=0.22)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        frameon=False,
        ncol=2,
    )
    fig.suptitle(
        "Same fixed cohort: inference switch vs Full",
        fontsize=14,
        y=0.995,
    )
    fig.text(
        0.5,
        0.005,
        (
            f"Bars: exact OT on shared row subset, seed {primary_seed}; "
            "diamonds: known same-cell identity-coupling upper bound. "
            "Open points show fixed-seed sampling sensitivity."
        ),
        ha="center",
        va="bottom",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.055, 1, 0.89))
    return _save_figure(fig, output_dir, FULL_FIGURE)


def _format_primary_table(table: pd.DataFrame) -> str:
    columns = [
        "condition",
        "space",
        "w1",
        "w1_percent_change_vs_full",
        "w2",
        "w2_percent_change_vs_full",
    ]
    rows = [
        "| Condition | Space | W1 | ΔW1 vs Full | W2 | ΔW2 vs Full |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for record in table[columns].to_dict("records"):
        rows.append(
            "| {condition} | {space} | {w1:.6g} | {w1_pct:+.2f}% | "
            "{w2:.6g} | {w2_pct:+.2f}% |".format(
                condition=_condition_label(str(record["condition"])),
                space=str(record["space"]),
                w1=float(record["w1"]),
                w1_pct=float(record["w1_percent_change_vs_full"]),
                w2=float(record["w2"]),
                w2_pct=float(record["w2_percent_change_vs_full"]),
            )
        )
    return "\n".join(rows)


def _write_interpretation(
    output_dir: Path,
    *,
    observed_primary: pd.DataFrame,
    primary_seed: int,
    target_time: float,
) -> Path:
    path = output_dir / INTERPRETATION_FILENAME
    body = f"""# Frozen-checkpoint distribution evaluation

## Primary result

Seed `{primary_seed}` is the prespecified primary result. W1/W2 below compare
each deterministic fixed-cohort endpoint with observed t{target_time:g};
lower is better.

{_format_primary_table(observed_primary)}

## What can and cannot be concluded

- `condition_vs_observed` asks whether an inference-time switch moves the saved
  fixed cohort closer to or farther from the observed endpoint distribution.
- `condition_vs_full` asks how strongly that switch changes the same fitted
  model. It uses the same sampled cell rows on both sides; the identity-coupling
  mean/RMS values are upper bounds on exact OT W1/W2, and the paired median is a
  cell-level effect-size summary.
- These deterministic trajectories retain the starting cohort and use no
  growth, splitting, resampling, noise, or spatial warp. Their absolute W1/W2
  therefore must not be numerically mixed with the retrained distribution
  evaluation, which can have a different particle/composition estimand.
- `lr_gate_off` admits every within-cutoff spatial candidate edge. Its result is
  confounded by the changed edge density and must not be described as a pure LR
  identity ablation.
- Joint, spatial, and PCA-state values live on different scales and should not
  be compared to one another.
"""
    path.write_text(body, encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.max_ot_points <= 0:
        raise ValueError("--max-ot-points must be positive.")
    ablation_dir = args.ablation_dir.expanduser().resolve()
    adata_path = args.adata.expanduser().resolve()
    if not adata_path.is_file():
        raise FileNotFoundError(f"AnnData input is missing: {adata_path}")
    output_dir = _prepare_output_dir(
        args.output_dir,
        overwrite=bool(args.overwrite),
    )
    ablation_manifest, ablation_manifest_path = _load_ablation_manifest(
        ablation_dir
    )
    conditions = _resolve_conditions(args.conditions, ablation_manifest)
    if args.full_condition not in conditions:
        raise ValueError(
            f"--full-condition {args.full_condition!r} is not in {conditions}."
        )

    condition_endpoints: dict[str, np.ndarray] = {}
    condition_files: dict[str, dict[str, Any]] = {}
    endpoint_metadata: dict[str, dict[str, Any]] = {}
    for condition in conditions:
        path = ablation_dir / f"{condition}.npz"
        endpoint, metadata = _load_condition_endpoint(
            path,
            target_time=float(args.target_time),
        )
        condition_endpoints[condition] = endpoint
        condition_files[condition] = _file_record(path)
        endpoint_metadata[condition] = metadata

    observed_endpoint, resolved_time_key, n_observed = _load_observed_endpoint(
        adata_path,
        target_time=float(args.target_time),
        time_key=args.time_key,
        obsm_key=args.obsm_key,
        spatial_key=args.spatial_key,
    )
    evaluation = cb.tl.evaluate_frozen_ablation_distributions(
        condition_endpoints,
        observed_endpoint,
        spatial_dim=int(args.spatial_dim),
        full_condition=str(args.full_condition),
        max_ot_points=int(args.max_ot_points),
        primary_seed=int(args.primary_seed),
        sensitivity_seeds=args.sensitivity_seeds,
    )

    observed_path = output_dir / OBSERVED_FILENAME
    full_path = output_dir / FULL_FILENAME
    sensitivity_path = output_dir / SENSITIVITY_FILENAME
    evaluation.condition_vs_observed.to_csv(observed_path, index=False)
    evaluation.condition_vs_full.to_csv(full_path, index=False)
    evaluation.sensitivity_summary.to_csv(sensitivity_path, index=False)

    primary = pd.concat(
        [
            evaluation.condition_vs_observed.loc[
                evaluation.condition_vs_observed["is_primary_seed"]
            ],
            evaluation.condition_vs_full.loc[
                evaluation.condition_vs_full["is_primary_seed"]
            ],
        ],
        ignore_index=True,
        sort=False,
    )
    primary_path = output_dir / PRIMARY_FILENAME
    primary.to_csv(primary_path, index=False)
    observed_primary = evaluation.condition_vs_observed.loc[
        evaluation.condition_vs_observed["is_primary_seed"]
    ].copy()

    output_paths: list[Path] = [
        primary_path,
        observed_path,
        full_path,
        sensitivity_path,
    ]
    output_paths.extend(
        _plot_condition_vs_observed(
            evaluation.condition_vs_observed,
            condition_order=conditions,
            full_condition=str(args.full_condition),
            primary_seed=int(args.primary_seed),
            target_time=float(args.target_time),
            output_dir=output_dir,
        )
    )
    output_paths.extend(
        _plot_condition_vs_full(
            evaluation.condition_vs_full,
            condition_order=conditions,
            full_condition=str(args.full_condition),
            primary_seed=int(args.primary_seed),
            output_dir=output_dir,
        )
    )
    output_paths.append(
        _write_interpretation(
            output_dir,
            observed_primary=observed_primary,
            primary_seed=int(args.primary_seed),
            target_time=float(args.target_time),
        )
    )

    manifest = {
        "analysis": "frozen_checkpoint_distribution_evaluation",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(sys.argv if argv is None else [sys.argv[0], *argv]),
        "code": {
            "script": _file_record(Path(__file__).resolve()),
            "api_module": _file_record(
                REPO_ROOT
                / "CytoBridge"
                / "tl"
                / "downstream"
                / "functional_ablation_evaluation.py"
            ),
            "git": _git_state(),
        },
        "inputs": {
            "ablation_manifest": _file_record(ablation_manifest_path),
            "ablation_manifest_analysis": ablation_manifest.get("analysis"),
            "condition_files": condition_files,
            "condition_endpoint_metadata": endpoint_metadata,
            "adata": _file_record(adata_path),
            "resolved_time_key": resolved_time_key,
            "obsm_key": str(args.obsm_key),
            "spatial_key": str(args.spatial_key),
            "target_time": float(args.target_time),
            "n_observed_target_cells": int(n_observed),
        },
        "settings": dict(evaluation.settings),
        "primary_result": {
            "sampling_seed": int(args.primary_seed),
            "table": PRIMARY_FILENAME,
            "condition_vs_observed_percent_change_definition": (
                "100 * (condition metric - full metric) / full metric; "
                "positive is worse and negative is better"
            ),
        },
        "caveats": {
            "fixed_cohort_estimand": (
                "Deterministic fixed cohort with no growth, split, resampling, "
                "noise, or spatial warp. Absolute W1/W2 must not be numerically "
                "mixed with retrained evaluations that use another composition "
                "or particle estimand."
            ),
            "target_composition": (
                "The rollout retains the source-stage cohort while the observed "
                "target contains the target-stage cell composition."
            ),
            "lr_gate_off": (
                "All within-cutoff candidates are admitted, changing edge "
                "density as well as LR-gate identity."
            ),
            "cross_space_scale": (
                "Joint, spatial, and PCA-state metrics have different units/"
                "scales and are not comparable across spaces."
            ),
        },
        "outputs": {
            path.name: _file_record(path)
            for path in output_paths
        },
    }
    manifest_path = output_dir / MANIFEST_FILENAME
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"Saved frozen-checkpoint distribution evaluation to {output_dir}")
    print(f"Primary metrics: {primary_path}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
