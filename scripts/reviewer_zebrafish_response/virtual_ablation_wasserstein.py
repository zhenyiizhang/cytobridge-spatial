#!/usr/bin/env python3
"""Summarize saved virtual-ablation trajectories with Wasserstein metrics.

The command is intentionally dataset agnostic.  It accepts one baseline
trajectory, one or more named variant trajectories, and an explicit time grid.
It delegates all distribution calculations to the public CytoBridge API and
adds reviewer-facing summaries, a spatial W1/W2 time-course figure, and a
hash-complete provenance manifest.

Example
-------
python scripts/reviewer_zebrafish_response/virtual_ablation_wasserstein.py \
  --baseline-points baseline_points.npy \
  --variant remove_YSL=remove_YSL_points.npy \
  --variant remove_EVL=remove_EVL_points.npy \
  --time-start 0 --time-stop 4 --time-step 0.05 \
  --output-dir results/virtual_ablation_wasserstein
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
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

import CytoBridge as cb


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
METRICS_FILENAME = "virtual_ablation_metrics.csv"
SUMMARY_FILENAME = "virtual_ablation_wasserstein_summary.csv"
FIGURE_FILENAME = "spatial_w1_w2_time_curves.png"
MANIFEST_FILENAME = "run_manifest.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute baseline-versus-variant virtual-ablation W1/W2 metrics "
            "from saved trajectory arrays."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--baseline-points",
        "--baseline",
        dest="baseline_points",
        required=True,
        type=Path,
        help="Baseline trajectory saved as a .npy array.",
    )
    parser.add_argument(
        "--variant",
        required=True,
        action="append",
        metavar="NAME=PATH",
        help=(
            "Named ablation trajectory. Repeat once per condition, for example "
            "--variant remove_YSL=/path/remove_YSL_points.npy."
        ),
    )
    parser.add_argument(
        "--time-grid",
        type=Path,
        default=None,
        help=(
            "Time values in .npy, .npz, .json, .csv, .tsv, or plain-text "
            "format. Mutually exclusive with --times and the regular-grid "
            "arguments."
        ),
    )
    parser.add_argument(
        "--times",
        default=None,
        help=(
            "Comma-separated explicit time values, for example "
            "'0,0.5,1,1.5'."
        ),
    )
    parser.add_argument("--time-start", type=float, default=None)
    parser.add_argument("--time-stop", type=float, default=None)
    parser.add_argument("--time-step", type=float, default=None)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--spatial-dim",
        type=int,
        default=2,
        help="Number of leading trajectory columns treated as spatial.",
    )
    parser.add_argument(
        "--max-ot-points",
        type=_optional_positive_int,
        default=1024,
        help=(
            "Maximum points retained per empirical cloud for exact OT; use "
            "'none' to retain every point."
        ),
    )
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--plot-title",
        default="Virtual ablation: spatial distribution shift",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of this command's known output files.",
    )
    return parser


def _optional_positive_int(value: str) -> int | None:
    if str(value).strip().casefold() in {"none", "null", "all"}:
        return None
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive or 'none'")
    return parsed


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


def _git_state(repo_root: Path) -> dict[str, Any]:
    def run(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    try:
        status = run("status", "--short")
        return {
            "repo_root": str(repo_root.resolve()),
            "commit": run("rev-parse", "HEAD"),
            "dirty": bool(status),
            "status_short": status.splitlines(),
        }
    except (OSError, subprocess.CalledProcessError) as error:
        return {
            "repo_root": str(repo_root.resolve()),
            "commit": None,
            "dirty": None,
            "error": str(error),
        }


def _parse_variant_specs(specs: Sequence[str]) -> dict[str, Path]:
    variants: dict[str, Path] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(
                f"Invalid --variant {spec!r}; expected the form NAME=PATH."
            )
        raw_name, raw_path = spec.split("=", 1)
        name = raw_name.strip()
        path_text = raw_path.strip()
        if not name or not path_text:
            raise ValueError(
                f"Invalid --variant {spec!r}; NAME and PATH must be non-empty."
            )
        if name in variants:
            raise ValueError(f"Duplicate --variant name: {name!r}.")
        path = Path(path_text).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Variant trajectory does not exist: {path}")
        variants[name] = path
    if not variants:
        raise ValueError("At least one --variant NAME=PATH is required.")
    return variants


def _load_trajectory(path: Path, *, label: str) -> np.ndarray:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} trajectory does not exist: {resolved}")
    if resolved.suffix.casefold() != ".npy":
        raise ValueError(f"{label} trajectory must be a .npy file: {resolved}")
    trajectory = np.load(resolved, allow_pickle=True)
    if trajectory.ndim == 0:
        raise ValueError(f"{label} trajectory must contain a sequence of frames.")
    if len(trajectory) == 0:
        raise ValueError(f"{label} trajectory contains no frames.")
    return trajectory


def _load_time_grid_file(path: Path) -> np.ndarray:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Time-grid file does not exist: {resolved}")
    suffix = resolved.suffix.casefold()
    if suffix == ".npy":
        return np.asarray(np.load(resolved, allow_pickle=False), dtype=float)
    if suffix == ".npz":
        with np.load(resolved, allow_pickle=False) as archive:
            preferred = [
                key for key in ("time_points", "times", "time") if key in archive
            ]
            if preferred:
                return np.asarray(archive[preferred[0]], dtype=float)
            if len(archive.files) != 1:
                raise ValueError(
                    "A .npz time grid must contain one array or a "
                    "'time_points'/'times'/'time' array."
                )
            return np.asarray(archive[archive.files[0]], dtype=float)
    if suffix == ".json":
        payload = json.loads(resolved.read_text(encoding="utf-8"))
        if isinstance(payload, Mapping):
            for key in ("time_points", "times", "time"):
                if key in payload:
                    payload = payload[key]
                    break
            else:
                raise ValueError(
                    "JSON time grid must be a list or contain "
                    "'time_points', 'times', or 'time'."
                )
        return np.asarray(payload, dtype=float)
    if suffix in {".csv", ".tsv"}:
        separator = "\t" if suffix == ".tsv" else ","
        frame = pd.read_csv(resolved, sep=separator)
        preferred = [
            column
            for column in ("time", "time_point", "time_points", "times")
            if column in frame.columns
        ]
        if preferred:
            return frame[preferred[0]].to_numpy(dtype=float)
        if frame.shape[1] != 1:
            raise ValueError(
                "CSV/TSV time grid must have one column or a named time column."
            )
        return frame.iloc[:, 0].to_numpy(dtype=float)
    return np.asarray(np.loadtxt(resolved, dtype=float), dtype=float)


def _regular_time_grid(start: float, stop: float, step: float) -> np.ndarray:
    if not np.isfinite([start, stop, step]).all():
        raise ValueError("Regular-grid start, stop, and step must be finite.")
    if step <= 0:
        raise ValueError("--time-step must be positive.")
    if stop <= start:
        raise ValueError("--time-stop must be greater than --time-start.")
    intervals = (stop - start) / step
    rounded = int(round(intervals))
    if rounded < 1 or not np.isclose(
        intervals, rounded, rtol=1e-9, atol=1e-9
    ):
        raise ValueError(
            "The interval from --time-start to --time-stop must be an integer "
            "multiple of --time-step."
        )
    return np.linspace(start, stop, rounded + 1, dtype=float)


def _validate_time_points(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 2 and 1 in array.shape:
        array = array.reshape(-1)
    if array.ndim != 1:
        raise ValueError("Time grid must be one-dimensional.")
    if array.size < 2:
        raise ValueError("Time grid must contain at least two points for AUC.")
    if not np.isfinite(array).all():
        raise ValueError("Time grid contains non-finite values.")
    if not np.all(np.diff(array) > 0):
        raise ValueError("Time grid must be strictly increasing.")
    return array.astype(float, copy=False)


def _resolve_time_grid(args: argparse.Namespace) -> tuple[np.ndarray, dict[str, Any]]:
    has_file = args.time_grid is not None
    has_explicit = args.times is not None
    regular_values = (args.time_start, args.time_stop, args.time_step)
    has_regular = any(value is not None for value in regular_values)
    if sum((has_file, has_explicit, has_regular)) != 1:
        raise ValueError(
            "Specify exactly one time-grid source: --time-grid, --times, or "
            "all of --time-start/--time-stop/--time-step."
        )

    source: dict[str, Any]
    if has_file:
        path = args.time_grid.expanduser().resolve()
        values = _load_time_grid_file(path)
        source = {"kind": "file", "file": _file_record(path)}
    elif has_explicit:
        tokens = [token.strip() for token in str(args.times).split(",")]
        if not tokens or any(not token for token in tokens):
            raise ValueError("--times must be a comma-separated list of numbers.")
        values = np.asarray([float(token) for token in tokens], dtype=float)
        source = {"kind": "explicit_cli"}
    else:
        if not all(value is not None for value in regular_values):
            raise ValueError(
                "--time-start, --time-stop, and --time-step must be supplied "
                "together."
            )
        values = _regular_time_grid(*regular_values)
        source = {
            "kind": "regular_cli",
            "start": float(args.time_start),
            "stop": float(args.time_stop),
            "step": float(args.time_step),
        }

    values = _validate_time_points(values)
    canonical = np.asarray(values, dtype="<f8").tobytes(order="C")
    source.update(
        {
            "n_time_points": int(values.size),
            "values": values.tolist(),
            "values_sha256_float64_le": hashlib.sha256(canonical).hexdigest(),
        }
    )
    return values, source


def summarize_wasserstein_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    required = {
        "variant",
        "time_index",
        "time",
        "space",
        "n_baseline",
        "n_ablation",
        "w1",
        "w2",
    }
    missing = sorted(required.difference(metrics.columns))
    if missing:
        raise ValueError(f"Metrics table is missing required columns: {missing}")

    rows: list[dict[str, Any]] = []
    for (variant, space), group in metrics.groupby(
        ["variant", "space"], sort=False, observed=True
    ):
        ordered = group.sort_values(["time_index", "time"], kind="stable")
        if ordered["time"].duplicated().any():
            raise ValueError(
                f"Duplicate time rows for variant={variant!r}, space={space!r}."
            )
        times = ordered["time"].to_numpy(dtype=float)
        baseline_n_t0 = int(ordered["n_baseline"].iloc[0])
        variant_n_t0 = int(ordered["n_ablation"].iloc[0])
        removed_n_t0 = int(baseline_n_t0 - variant_n_t0)
        removed_fraction_t0 = (
            float(removed_n_t0 / baseline_n_t0)
            if baseline_n_t0 > 0 and removed_n_t0 > 0
            else float("nan")
        )
        for metric in ("w1", "w2"):
            values = ordered[metric].to_numpy(dtype=float)
            finite = np.isfinite(values)
            t0_value = float(values[0])
            endpoint_value = float(values[-1])
            time_span = float(times[-1] - times[0])
            if finite.all():
                trapezoid = getattr(np, "trapezoid", np.trapz)
                auc = float(trapezoid(values, times))
                auc_change = float(auc - t0_value * time_span)
                time_average = float(auc / time_span)
            else:
                auc = float("nan")
                auc_change = float("nan")
                time_average = float("nan")
            rows.append(
                {
                    "variant": str(variant),
                    "space": str(space),
                    "metric": metric,
                    "n_time_points": int(len(values)),
                    "n_finite_time_points": int(finite.sum()),
                    "time_t0": float(times[0]),
                    "time_endpoint": float(times[-1]),
                    "time_span": time_span,
                    "baseline_n_t0": baseline_n_t0,
                    "variant_n_t0": variant_n_t0,
                    "removed_n_t0": removed_n_t0,
                    "removed_fraction_t0": removed_fraction_t0,
                    "value_t0": t0_value,
                    "value_endpoint": endpoint_value,
                    "endpoint_change_from_t0": float(
                        endpoint_value - t0_value
                    ),
                    "auc": auc,
                    "auc_change_from_t0": auc_change,
                    "time_average": time_average,
                    "endpoint_change_per_removed_fraction": (
                        float((endpoint_value - t0_value) / removed_fraction_t0)
                        if np.isfinite(removed_fraction_t0)
                        else float("nan")
                    ),
                    "auc_change_per_removed_fraction": (
                        float(auc_change / removed_fraction_t0)
                        if np.isfinite(removed_fraction_t0)
                        else float("nan")
                    ),
                }
            )
    return pd.DataFrame(rows)


def plot_spatial_wasserstein(
    metrics: pd.DataFrame,
    output_path: Path,
    *,
    variant_order: Sequence[str],
    title: str,
) -> None:
    spatial = metrics.loc[metrics["space"].eq("spatial")].copy()
    if spatial.empty:
        raise ValueError(
            "No spatial rows were produced; --spatial-dim must be at least 1."
        )

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.5), sharex=True)
    colors = plt.get_cmap("tab10")
    markers = ("o", "s", "^", "D", "v", "P", "X", "<", ">")
    line_styles = ("-", "--", "-.", ":")
    for index, variant in enumerate(variant_order):
        variant_rows = spatial.loc[spatial["variant"].eq(variant)].sort_values(
            ["time_index", "time"], kind="stable"
        )
        if variant_rows.empty:
            raise ValueError(f"No spatial metrics found for variant {variant!r}.")
        mark_every = max(1, int(np.ceil(len(variant_rows) / 12)))
        for axis, metric, label in zip(
            axes, ("w1", "w2"), ("W1", "W2"), strict=True
        ):
            axis.plot(
                variant_rows["time"],
                variant_rows[metric],
                label=str(variant),
                color=colors(index % 10),
                linestyle=line_styles[(index // 10) % len(line_styles)],
                linewidth=2.2,
                marker=markers[index % len(markers)],
                markersize=4.2,
                markevery=mark_every,
            )
            axis.set_title(f"Spatial {label}")
            axis.set_xlabel("Model time")
            axis.set_ylabel(f"{label} distance from baseline")
            axis.set_ylim(bottom=0)
            axis.grid(True, color="#d9d9d9", linewidth=0.7, alpha=0.8)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.96),
        ncol=min(4, max(1, len(labels))),
        frameon=False,
        title="Ablation variant",
    )
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.04)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        metadata={"Title": title},
    )
    plt.close(fig)


def _prepare_output_dir(output_dir: Path, *, overwrite: bool) -> Path:
    resolved = output_dir.expanduser().resolve()
    known_outputs = (
        METRICS_FILENAME,
        SUMMARY_FILENAME,
        FIGURE_FILENAME,
        MANIFEST_FILENAME,
    )
    existing = [resolved / name for name in known_outputs if (resolved / name).exists()]
    if existing and not overwrite:
        rendered = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"Refusing to overwrite existing outputs: {rendered}. "
            "Use --overwrite to replace them."
        )
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def run(args: argparse.Namespace, *, command_argv: Sequence[str]) -> dict[str, Any]:
    baseline_path = args.baseline_points.expanduser().resolve()
    variants = _parse_variant_specs(args.variant)
    time_points, time_source = _resolve_time_grid(args)
    if args.spatial_dim <= 0:
        raise ValueError("--spatial-dim must be positive for the spatial figure.")

    baseline = _load_trajectory(baseline_path, label="Baseline")
    variant_trajectories = {
        name: _load_trajectory(path, label=f"Variant {name!r}")
        for name, path in variants.items()
    }
    expected_frames = int(time_points.size)
    trajectory_lengths = {
        "baseline": int(len(baseline)),
        **{
            name: int(len(trajectory))
            for name, trajectory in variant_trajectories.items()
        },
    }
    wrong_lengths = {
        name: length
        for name, length in trajectory_lengths.items()
        if length != expected_frames
    }
    if wrong_lengths:
        raise ValueError(
            f"Trajectory/time-grid length mismatch; expected {expected_frames} "
            f"frames, found {wrong_lengths}."
        )

    git_state = _git_state(REPO_ROOT)
    metrics = cb.tl.compute_virtual_ablation_metrics(
        baseline,
        variant_trajectories,
        time_points,
        spatial_dim=int(args.spatial_dim),
        max_ot_points=args.max_ot_points,
        random_seed=int(args.random_seed),
    )
    summary = summarize_wasserstein_metrics(metrics)
    output_dir = _prepare_output_dir(args.output_dir, overwrite=args.overwrite)
    metrics_path = output_dir / METRICS_FILENAME
    summary_path = output_dir / SUMMARY_FILENAME
    figure_path = output_dir / FIGURE_FILENAME
    manifest_path = output_dir / MANIFEST_FILENAME

    metrics.to_csv(metrics_path, index=False, float_format="%.12g")
    summary.to_csv(summary_path, index=False, float_format="%.12g")
    plot_spatial_wasserstein(
        metrics,
        figure_path,
        variant_order=list(variants),
        title=str(args.plot_title),
    )

    output_records = {
        "metrics": _file_record(metrics_path),
        "summary": _file_record(summary_path),
        "spatial_time_curves": _file_record(figure_path),
    }
    command = [str(value) for value in command_argv]
    manifest = {
        "schema_version": 1,
        "analysis": "dataset_agnostic_virtual_ablation_wasserstein",
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": {
            "argv": command,
            "shell": shlex.join(command),
            "cwd": str(Path.cwd().resolve()),
        },
        "code": {
            "script": _file_record(SCRIPT_PATH),
            "git": git_state,
        },
        "inputs": {
            "baseline_points": _file_record(baseline_path),
            "variants": {
                name: _file_record(path) for name, path in variants.items()
            },
            "time_grid": time_source,
        },
        "parameters": {
            "spatial_dim": int(args.spatial_dim),
            "max_ot_points": (
                None
                if args.max_ot_points is None
                else int(args.max_ot_points)
            ),
            "random_seed": int(args.random_seed),
            "plot_title": str(args.plot_title),
            "variant_order": list(variants),
        },
        "metric_definition": {
            "implementation": "cb.tl.compute_virtual_ablation_metrics",
            "comparison": "variant empirical distribution versus matched baseline",
            "weights": "uniform empirical weights",
            "spaces": metrics["space"].drop_duplicates().astype(str).tolist(),
            "summary": {
                "t0": "the first supplied time-grid point",
                "endpoint": "the last supplied time-grid point",
                "endpoint_change_from_t0": "endpoint value minus t0 value",
                "auc": "trapezoidal integral of the raw distance over model time",
                "auc_change_from_t0": (
                    "raw AUC minus t0 value multiplied by elapsed model time"
                ),
                "removed_fraction_t0": (
                    "(baseline t0 count minus variant t0 count) divided by "
                    "baseline t0 count"
                ),
                "per_removed_fraction_columns": (
                    "descriptive scale normalization only; Wasserstein "
                    "divergence is not assumed to be linear in the removed "
                    "fraction and these columns are not causal effect estimates"
                ),
            },
        },
        "counts": {
            "variants": int(len(variants)),
            "time_points": int(len(time_points)),
            "metric_rows": int(len(metrics)),
            "summary_rows": int(len(summary)),
            "trajectory_frames": trajectory_lengths,
        },
        "outputs": output_records,
        "manifest": {
            "path": str(manifest_path),
            "self_hash_omitted": True,
        },
    }
    manifest_path.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
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
