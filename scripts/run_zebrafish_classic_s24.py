#!/usr/bin/env python3
"""Run and report the classic unequal-population zebrafish S24 sensitivity.

This runner restores the estimand used by the original manuscript panel: one
complete observed t=0 cohort is propagated as the baseline, while the YSL and
EVL branches are exact subsets formed by deleting those labels.  Learned
growth-driven split/extinction remains enabled, so branch particle counts are
allowed to diverge.  This is a virtual-removal model sensitivity, not a causal
knockout or a biological-replicate experiment.

The command is deliberately separate from ``run_zebrafish_paper_downstream``.
Its corrected equal-N, fixed-population S24 remains a useful shape control.
Here, ``run-seed`` produces one formal simulation-seed result and ``report``
combines seeds 42--46 only after applying the predeclared latent-support gate.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[1]
_SCRIPTS_DIR = SCRIPT_PATH.parent
for _source_root in (str(_SCRIPTS_DIR), str(REPO_ROOT)):
    if _source_root in sys.path:
        sys.path.remove(_source_root)
    sys.path.insert(0, _source_root)

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

import CytoBridge as cb
import run_zebrafish_paper_downstream as paper


SCHEMA_VERSION = 1
FORMAL_SEEDS = (42, 43, 44, 45, 46)
TRAJECTORY_FILENAMES = {
    "baseline": "baseline_points.npy",
    "remove_YSL": "remove_YSL_points.npy",
    "remove_EVL": "remove_EVL_points.npy",
}
REPLAY_STATE_RTOL = 1e-6
REPLAY_STATE_ATOL = 1e-5
REPLAY_METRIC_RTOL = 1e-8
REPLAY_METRIC_ATOL = 1e-8
REPLAY_TIME_ATOL = 1e-12
REPLAY_METRIC_CATEGORICAL_COLUMNS = ("variant", "space")
REPLAY_METRIC_DISCRETE_COLUMNS = (
    "time_index",
    "n_baseline",
    "n_ablation",
    "count_delta",
    "ot_baseline_points",
    "ot_ablation_points",
    "ot_random_seed",
)
REPLAY_SEED = 42
YSL_LABEL = "Yolk Syncytial Layer"
EVL_LABEL = "EVL"
ABLATIONS = {"remove_YSL": (YSL_LABEL,), "remove_EVL": (EVL_LABEL,)}
EXPECTED_T0_COUNTS = {"total": 563, YSL_LABEL: 29, EVL_LABEL: 272}
DT = 0.005
OUTPUT_STEP = 0.05
RESAMPLE_DT = 0.05
SIGMA = 0.03
GROWTH_ALPHA = 1.0
INTERACTION_M = 1024
MAX_PARTICLES = 100_000
MAX_OT_POINTS = 1024
SUPPORT_MAX_OUTSIDE_FRACTION = 0.01
SUPPORT_MAX_NORM_MULTIPLIER = 2.0
OBSERVED_ENDPOINTS = (0.0, 1.0, 2.0, 3.0, 4.0)
CONDITION_COLORS = {
    "baseline": "#59616A",
    "remove_YSL": "#0072B2",
    "remove_EVL": "#D55E00",
}
CONDITION_LABELS = {
    "baseline": "Baseline",
    "remove_YSL": "YSL removal",
    "remove_EVL": "EVL removal",
}


def _require_import_origins() -> None:
    bindings = {
        "CytoBridge": Path(cb.__file__).resolve(),
        "paper runner": Path(paper.__file__).resolve(),
    }
    invalid = {
        name: str(path)
        for name, path in bindings.items()
        if not path.is_relative_to(REPO_ROOT)
    }
    if invalid:
        raise RuntimeError(
            f"Classic S24 imported code outside its release root {REPO_ROOT}: {invalid}"
        )


def _time_grid(end_time: float = 4.0) -> np.ndarray:
    end_time = float(end_time)
    if end_time not in {3.0, 4.0}:
        raise ValueError("Classic S24 end_time must be exactly 3 or 4.")
    return np.linspace(
        0.0,
        end_time,
        int(round(end_time / OUTPUT_STEP)) + 1,
        dtype=np.float64,
    )


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "path": str(resolved),
        "bytes": int(resolved.stat().st_size),
        "sha256": _sha256(resolved),
    }


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _write_sha256_sidecar(path: Path) -> Path:
    sidecar = path.with_name(path.name + ".sha256")
    sidecar.write_text(f"{_sha256(path)}  {path.name}\n", encoding="utf-8")
    return sidecar


def _require_sha256_sidecar(path: Path) -> None:
    sidecar = path.with_name(path.name + ".sha256")
    if not sidecar.is_file():
        raise FileNotFoundError(f"Missing SHA-256 sidecar: {sidecar}")
    fields = sidecar.read_text(encoding="utf-8").strip().split()
    if len(fields) != 2 or fields[1] != path.name or fields[0] != _sha256(path):
        raise RuntimeError(f"Stale or malformed SHA-256 sidecar: {sidecar}")


def _fresh_directory(path: Path, *, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.exists() and any(resolved.iterdir()):
        raise FileExistsError(f"Refusing non-empty {label}: {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _paper_context(args: argparse.Namespace, seed_dir: Path):
    argv = [
        "--aligned-h5ad",
        str(args.aligned_h5ad),
        "--model-dir",
        str(args.model_dir),
        "--acceptance-report",
        str(args.acceptance_report),
        "--expected-acceptance-sha256",
        str(args.expected_acceptance_sha256),
        "--output-dir",
        str(seed_dir / "_context"),
        "--profile",
        "full",
        "--device",
        str(args.device),
        "--random-seed",
        str(args.seed),
        "--sde-dt",
        str(DT),
        "--sde-sigma",
        str(SIGMA),
        "--growth-alpha",
        str(GROWTH_ALPHA),
        "--sde-max-particles",
        str(MAX_PARTICLES),
        "--ablation-step",
        str(OUTPUT_STEP),
    ]
    if args.shared_cache_dir is not None:
        argv.extend(["--shared-cache-dir", str(args.shared_cache_dir)])
    parsed = paper._build_parser().parse_args(argv)
    ctx = paper._build_context(parsed)
    git_state = ctx.common_signature.get("git", {})
    commit = str(git_state.get("commit", "")) if isinstance(git_state, Mapping) else ""
    if (
        not isinstance(git_state, Mapping)
        or git_state.get("dirty") is not False
        or len(commit) != 40
    ):
        raise RuntimeError(
            "Formal classic S24 requires a clean exact Git checkout with a "
            f"40-character commit; got {git_state!r}."
        )
    return ctx


def _validate_formal_t0(ctx: Any) -> None:
    times = np.asarray(ctx.adata.obs[ctx.args.time_key], dtype=float)
    mask = np.isclose(times, 0.0, rtol=0.0, atol=1e-9)
    labels = ctx.adata.obs.loc[mask, ctx.args.annotation_key].astype(str)
    actual = {
        "total": int(mask.sum()),
        YSL_LABEL: int((labels == YSL_LABEL).sum()),
        EVL_LABEL: int((labels == EVL_LABEL).sum()),
    }
    if actual != EXPECTED_T0_COUNTS:
        raise RuntimeError(
            "The accepted zebrafish t=0 cohort does not match the frozen classic "
            f"S24 contract: expected {EXPECTED_T0_COUNTS}, got {actual}."
        )


def _validate_result_contract(result: Any, *, seed: int, end_time: float = 4.0) -> None:
    expected_times = _time_grid(end_time)
    if not np.array_equal(np.asarray(result.time_points), expected_times):
        raise RuntimeError("Classic S24 returned a non-canonical output grid.")
    settings = result.settings
    exact = {
        "mass_control": False,
        "growth_alpha": GROWTH_ALPHA,
        "dt": DT,
        "resample_dt": RESAMPLE_DT,
        "sigma": SIGMA,
        "interaction_m": INTERACTION_M,
        "max_particles": MAX_PARTICLES,
        "random_seed": int(seed),
        "interaction_seed": int(seed) + 10_001,
        "common_random_seed": True,
    }
    for key, expected in exact.items():
        actual = settings.get(key)
        if isinstance(expected, float):
            valid = actual is not None and np.isclose(
                float(actual), expected, rtol=0.0, atol=1e-12
            )
        else:
            valid = actual == expected
        if not valid:
            raise RuntimeError(
                f"Classic S24 setting {key!r} changed: expected={expected!r}, "
                f"actual={actual!r}."
            )
    initial_counts = settings.get("variant_initial_counts", {})
    if int(settings.get("n_initial", -1)) != 563 or initial_counts != {
        "remove_YSL": 534,
        "remove_EVL": 291,
    }:
        raise RuntimeError(
            "Classic S24 did not preserve the full 563-cell baseline and exact "
            f"YSL/EVL subsets: settings={settings}."
        )
    if set(result.ablation_points) != set(ABLATIONS):
        raise RuntimeError("Classic S24 returned the wrong ablation branches.")


def run_seed(args: argparse.Namespace) -> int:
    seed = int(args.seed)
    if seed not in FORMAL_SEEDS:
        raise ValueError(f"--seed must be one of {FORMAL_SEEDS}, got {seed}.")
    seed_dir = _fresh_directory(Path(args.output_dir), label="seed output directory")
    ctx = _paper_context(args, seed_dir)
    _validate_formal_t0(ctx)
    end_time = float(args.end_time)
    time_points = _time_grid(end_time)
    experiment_dir = seed_dir / "experiment"
    result = cb.tl.run_virtual_cell_type_ablation(
        ctx.adata,
        ctx.runtime,
        ablations=ABLATIONS,
        time_points=time_points,
        output_dir=experiment_dir,
        time_index=0,
        n_samples=None,
        dt=DT,
        resample_dt=RESAMPLE_DT,
        sigma=SIGMA,
        growth_alpha=GROWTH_ALPHA,
        interaction_m=INTERACTION_M,
        max_particles=MAX_PARTICLES,
        device=ctx.args.device,
        time_key=ctx.args.time_key,
        annotation_key=ctx.args.annotation_key,
        obsm_key=ctx.args.latent_key,
        spatial_key=ctx.args.spatial_key,
        concat_spatial=True,
        spatial_dim=2,
        random_seed=seed,
        interaction_seed=seed + 10_001,
        common_random_seed=True,
        max_ot_points=MAX_OT_POINTS,
        mass_control=False,
        trajectory_labeler=None,
        save_data=True,
        save_snapshots=False,
        verbose=True,
    )
    _validate_result_contract(result, seed=seed, end_time=end_time)
    trajectories = {
        "baseline": result.baseline_points,
        "remove_YSL": result.ablation_points["remove_YSL"],
        "remove_EVL": result.ablation_points["remove_EVL"],
    }
    audit, audit_summary = paper._compute_trajectory_support_audit(
        np.asarray(ctx.adata.obsm[ctx.args.latent_key], dtype=np.float32),
        trajectories,
        time_points,
        spatial_dim=2,
        max_outside_fraction=SUPPORT_MAX_OUTSIDE_FRACTION,
        max_norm_multiplier=SUPPORT_MAX_NORM_MULTIPLIER,
    )
    audit_path = seed_dir / "trajectory_support_audit.csv"
    audit_json_path = seed_dir / "trajectory_support_audit.json"
    audit.to_csv(audit_path, index=False, float_format="%.12g")
    _write_json(audit_json_path, audit_summary)

    output_files = sorted(
        path
        for path in experiment_dir.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "analysis": "zebrafish_classic_s24_unequal_population_virtual_removal",
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "protocol": {
            "initialization": "one complete observed t=0 baseline cohort",
            "baseline_initial_n": 563,
            "variant_initial_n": {"remove_YSL": 534, "remove_EVL": 291},
            "ablation": "exact target-label subset deletion; no replacement",
            "mass_control": False,
            "growth_alpha": GROWTH_ALPHA,
            "growth_resampling": "enabled at fixed 0.05 event intervals",
            "sigma": SIGMA,
            "dt": DT,
            "resample_dt": RESAMPLE_DT,
            "time_points": time_points.tolist(),
            "end_time": end_time,
            "continuous_global_t0": True,
            "reanchoring": False,
            "spatial_warp": False,
            "state_clipping_or_outlier_removal": False,
            "trajectory_downsampling": False,
            "ot_max_points": MAX_OT_POINTS,
            "ot_subsampling": (
                "deterministic metric-only empirical-cloud subsampling; saved "
                "trajectories and plotted states retain every particle"
            ),
            "interaction_m": INTERACTION_M,
            "interaction_seed": seed + 10_001,
            "max_particles": MAX_PARTICLES,
            "common_random_seed": True,
            "interpretation": (
                "one-checkpoint virtual-removal model sensitivity; not a causal "
                "knockout, biological replicate, or uncertainty over training"
            ),
        },
        "inputs": {
            "aligned_h5ad": _file_record(ctx.args.aligned_h5ad),
            "model_config": _file_record(ctx.args.model_dir / "config.yaml"),
            "weight": _file_record(ctx.common_signature["weight_path"]),
            "score": _file_record(ctx.common_signature["score_path"]),
            "acceptance": ctx.common_signature[paper.MATCHED_ACCEPTANCE_KEY],
        },
        "code": {
            "runner": _file_record(SCRIPT_PATH),
            "git": ctx.common_signature["git"],
        },
        "plot_metadata": {"label_to_color": dict(ctx.label_to_color)},
        "support_audit": {
            "status": audit_summary["status"],
            "csv": _file_record(audit_path),
            "json": _file_record(audit_json_path),
            "publication_endpoint_selected_later_across_all_seeds": True,
        },
        "outputs": [_file_record(path) for path in output_files],
    }
    manifest_path = seed_dir / "run_summary.json"
    _write_json(manifest_path, manifest)
    _write_sha256_sidecar(manifest_path)
    print(
        json.dumps(
            {"status": "complete", "seed": seed, "run_summary": str(manifest_path)}
        )
    )
    return 0


def _load_seed(seed_dir: Path, *, expected_seed: int) -> dict[str, Any]:
    summary_path = seed_dir / "run_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing seed summary: {summary_path}")
    _require_sha256_sidecar(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        summary.get("status") != "complete"
        or int(summary.get("seed", -1)) != expected_seed
    ):
        raise RuntimeError(f"Invalid seed summary: {summary_path}")
    protocol = summary.get("protocol", {})
    required = {
        "baseline_initial_n": 563,
        "mass_control": False,
        "growth_alpha": GROWTH_ALPHA,
        "sigma": SIGMA,
        "dt": DT,
        "resample_dt": RESAMPLE_DT,
        "continuous_global_t0": True,
        "reanchoring": False,
        "spatial_warp": False,
        "state_clipping_or_outlier_removal": False,
        "trajectory_downsampling": False,
        "ot_max_points": MAX_OT_POINTS,
        "end_time": protocol.get("end_time"),
    }
    for key, value in required.items():
        if protocol.get(key) != value:
            raise RuntimeError(f"Seed {expected_seed} protocol mismatch for {key}.")
    recorded_runner = summary.get("code", {}).get("runner", {})
    if recorded_runner.get("sha256") != _sha256(SCRIPT_PATH):
        raise RuntimeError(
            f"Seed {expected_seed} was produced by different classic S24 runner bytes."
        )
    recorded_git = summary.get("code", {}).get("git", {})
    current_git = paper._git_revision()
    if (
        not isinstance(recorded_git, Mapping)
        or recorded_git.get("dirty") is not False
        or len(str(recorded_git.get("commit", ""))) != 40
        or recorded_git.get("commit") != current_git.get("commit")
        or current_git.get("dirty") is not False
    ):
        raise RuntimeError(
            f"Seed {expected_seed} does not bind the current clean release commit: "
            f"recorded={recorded_git}, current={current_git}."
        )
    for record in summary.get("outputs", []):
        path = Path(record["path"])
        if "experiment" not in path.parts:
            raise RuntimeError(
                f"Seed {expected_seed} output is outside the experiment bundle: {path}"
            )
        component_index = len(path.parts) - 1 - path.parts[::-1].index("experiment")
        relative = Path(*path.parts[component_index + 1 :])
        local = seed_dir / "experiment" / relative
        if not local.is_file() or _sha256(local) != record["sha256"]:
            raise RuntimeError(
                f"Seed {expected_seed} output changed or is missing: {local}"
            )
    support_records = summary.get("support_audit", {})
    for key, filename in (
        ("csv", "trajectory_support_audit.csv"),
        ("json", "trajectory_support_audit.json"),
    ):
        record = support_records.get(key, {})
        local = seed_dir / filename
        if (
            not isinstance(record, Mapping)
            or not local.is_file()
            or _sha256(local) != record.get("sha256")
        ):
            raise RuntimeError(
                f"Seed {expected_seed} support audit {key} changed or is missing: {local}"
            )
    trajectory_dir = seed_dir / "experiment" / "trajectories"
    trajectories = {
        name: np.load(trajectory_dir / filename, allow_pickle=True)
        for name, filename in TRAJECTORY_FILENAMES.items()
    }
    metrics = pd.read_csv(seed_dir / "experiment" / "ablation_metrics.csv")
    audit = pd.read_csv(seed_dir / "trajectory_support_audit.csv")
    required_audit_columns = {
        "condition",
        "time",
        "passes_publication_support_gate",
    }
    if not required_audit_columns.issubset(audit.columns):
        raise RuntimeError(f"Seed {expected_seed} support audit has the wrong schema.")
    if set(audit["condition"].astype(str)) != {
        "baseline",
        "remove_YSL",
        "remove_EVL",
    }:
        raise RuntimeError(f"Seed {expected_seed} support audit has wrong conditions.")
    expected_times = _time_grid(float(protocol.get("end_time")))
    for condition, rows in audit.groupby("condition", sort=False):
        actual_times = rows.sort_values("time", kind="stable")["time"].to_numpy(
            dtype=float
        )
        if not np.allclose(actual_times, expected_times, rtol=0.0, atol=1e-12):
            raise RuntimeError(
                f"Seed {expected_seed} support audit has wrong times for {condition}."
            )
    return {
        "summary": summary,
        "trajectories": trajectories,
        "metrics": metrics,
        "audit": audit,
        "summary_path": summary_path,
    }


def _latest_common_endpoint(seed_runs: Mapping[int, Mapping[str, Any]]) -> float:
    latest = None
    maximum_common_time = min(
        float(run["audit"]["time"].max()) for run in seed_runs.values()
    )
    for endpoint in (
        value for value in OBSERVED_ENDPOINTS if value <= maximum_common_time + 1e-9
    ):
        passed = True
        for run in seed_runs.values():
            rows = run["audit"]
            through_endpoint = rows.loc[rows["time"] <= endpoint + 1e-9]
            expected_frames = 3 * (int(round(endpoint / OUTPUT_STEP)) + 1)
            if (
                len(through_endpoint) != expected_frames
                or not through_endpoint["passes_publication_support_gate"]
                .astype(bool)
                .all()
            ):
                passed = False
                break
        if passed:
            latest = endpoint
    if latest is None:
        raise RuntimeError(
            "No observed endpoint passes the support gate across all seeds."
        )
    return float(latest)


def _trajectory_hashes(trajectory_dir: Path) -> dict[str, str]:
    return {
        name: _sha256(trajectory_dir / filename)
        for name, filename in TRAJECTORY_FILENAMES.items()
    }


def _require_seed42_replay(run_root: Path, replay_dir: Path) -> dict[str, Any]:
    primary_experiment = run_root / "seeds" / "seed_42" / "experiment"
    replay_experiment = replay_dir / "experiment"
    primary_trajectories = primary_experiment / "trajectories"
    replay_trajectories = replay_experiment / "trajectories"
    primary_hashes = _trajectory_hashes(primary_trajectories)
    replay_hashes = _trajectory_hashes(replay_trajectories)

    trajectory_audit: dict[str, dict[str, Any]] = {}
    for name, filename in TRAJECTORY_FILENAMES.items():
        primary = np.load(primary_trajectories / filename, allow_pickle=True)
        replay = np.load(replay_trajectories / filename, allow_pickle=True)
        if primary.ndim != 1 or replay.ndim != 1:
            raise RuntimeError(
                f"Seed 42 replay trajectory {name} is not a one-dimensional "
                "frame container."
            )
        if len(primary) != len(replay):
            raise RuntimeError(
                f"Seed 42 replay trajectory {name} frame count differs: "
                f"primary={len(primary)}, replay={len(replay)}."
            )

        squared_error_sum = 0.0
        compared_values = 0
        max_abs_difference = 0.0
        maximum_particle_count = 0
        for frame_index, (primary_frame, replay_frame) in enumerate(
            zip(primary, replay)
        ):
            primary_values = np.asarray(primary_frame)
            replay_values = np.asarray(replay_frame)
            if primary_values.ndim != 2 or replay_values.ndim != 2:
                raise RuntimeError(
                    f"Seed 42 replay trajectory {name} frame {frame_index} is "
                    "not two-dimensional."
                )
            if primary_values.shape != replay_values.shape:
                raise RuntimeError(
                    f"Seed 42 replay trajectory {name} frame {frame_index} "
                    "shape/count differs: "
                    f"primary={primary_values.shape}, replay={replay_values.shape}."
                )
            if not (
                np.issubdtype(primary_values.dtype, np.number)
                and np.issubdtype(replay_values.dtype, np.number)
            ):
                raise RuntimeError(
                    f"Seed 42 replay trajectory {name} frame {frame_index} is "
                    "not numeric."
                )
            if not (
                np.isfinite(primary_values).all() and np.isfinite(replay_values).all()
            ):
                raise RuntimeError(
                    f"Seed 42 replay trajectory {name} frame {frame_index} "
                    "contains non-finite values."
                )
            if not np.allclose(
                primary_values,
                replay_values,
                rtol=REPLAY_STATE_RTOL,
                atol=REPLAY_STATE_ATOL,
            ):
                difference = np.abs(
                    primary_values.astype(np.float64) - replay_values.astype(np.float64)
                )
                raise RuntimeError(
                    f"Seed 42 replay trajectory {name} frame {frame_index} "
                    "exceeds the numerical replay tolerance: "
                    f"max_abs={float(difference.max()):.12g}, "
                    f"rtol={REPLAY_STATE_RTOL:g}, atol={REPLAY_STATE_ATOL:g}."
                )
            difference = primary_values.astype(np.float64) - replay_values.astype(
                np.float64
            )
            if difference.size:
                max_abs_difference = max(
                    max_abs_difference, float(np.max(np.abs(difference)))
                )
                squared_error_sum += float(np.sum(np.square(difference)))
                compared_values += int(difference.size)
            maximum_particle_count = max(
                maximum_particle_count, int(primary_values.shape[0])
            )

        trajectory_audit[name] = {
            "primary_sha256": primary_hashes[name],
            "replay_sha256": replay_hashes[name],
            "byte_exact": primary_hashes[name] == replay_hashes[name],
            "n_frames": int(len(primary)),
            "frame_shapes_exact": True,
            "particle_counts_exact": True,
            "finite": True,
            "maximum_particle_count": maximum_particle_count,
            "max_abs_difference": max_abs_difference,
            "rmse_difference": (
                float(np.sqrt(squared_error_sum / compared_values))
                if compared_values
                else 0.0
            ),
        }

    primary_metrics_path = primary_experiment / "ablation_metrics.csv"
    replay_metrics_path = replay_experiment / "ablation_metrics.csv"
    primary_metrics = pd.read_csv(primary_metrics_path)
    replay_metrics = pd.read_csv(replay_metrics_path)
    if list(primary_metrics.columns) != list(replay_metrics.columns):
        raise RuntimeError(
            "Seed 42 replay metric schema differs from the primary run: "
            f"primary={list(primary_metrics.columns)}, "
            f"replay={list(replay_metrics.columns)}."
        )
    if len(primary_metrics) != len(replay_metrics):
        raise RuntimeError(
            "Seed 42 replay metric row count differs from the primary run: "
            f"primary={len(primary_metrics)}, replay={len(replay_metrics)}."
        )
    if primary_metrics.empty:
        raise RuntimeError("Seed 42 replay metrics are empty.")
    required_metric_columns = {
        *REPLAY_METRIC_CATEGORICAL_COLUMNS,
        *REPLAY_METRIC_DISCRETE_COLUMNS,
        "time",
    }
    missing_metric_columns = sorted(
        required_metric_columns - set(primary_metrics.columns)
    )
    if missing_metric_columns:
        raise RuntimeError(
            "Seed 42 replay metrics lack required contract columns: "
            f"{missing_metric_columns}."
        )

    for column in REPLAY_METRIC_CATEGORICAL_COLUMNS:
        primary_values = primary_metrics[column]
        replay_values = replay_metrics[column]
        if (
            primary_values.isna().any()
            or replay_values.isna().any()
            or not np.array_equal(
                primary_values.to_numpy(dtype=object),
                replay_values.to_numpy(dtype=object),
            )
        ):
            raise RuntimeError(
                f"Seed 42 replay categorical metric column {column!r} differs."
            )

    for column in REPLAY_METRIC_DISCRETE_COLUMNS:
        if not (
            pd.api.types.is_numeric_dtype(primary_metrics[column])
            and pd.api.types.is_numeric_dtype(replay_metrics[column])
        ):
            raise RuntimeError(
                f"Seed 42 replay discrete metric column {column!r} is not numeric."
            )
        primary_values = primary_metrics[column].to_numpy(dtype=np.float64)
        replay_values = replay_metrics[column].to_numpy(dtype=np.float64)
        if not (
            np.isfinite(primary_values).all()
            and np.isfinite(replay_values).all()
            and np.equal(primary_values, np.rint(primary_values)).all()
            and np.equal(replay_values, np.rint(replay_values)).all()
            and np.array_equal(primary_values, replay_values)
        ):
            raise RuntimeError(
                f"Seed 42 replay discrete metric column {column!r} differs."
            )

    primary_time = primary_metrics["time"].to_numpy(dtype=np.float64)
    replay_time = replay_metrics["time"].to_numpy(dtype=np.float64)
    if not (
        np.isfinite(primary_time).all()
        and np.isfinite(replay_time).all()
        and np.allclose(
            primary_time,
            replay_time,
            rtol=0.0,
            atol=REPLAY_TIME_ATOL,
        )
    ):
        raise RuntimeError(
            "Seed 42 replay metric time column differs beyond "
            f"atol={REPLAY_TIME_ATOL:g}."
        )

    excluded_columns = {
        *REPLAY_METRIC_CATEGORICAL_COLUMNS,
        *REPLAY_METRIC_DISCRETE_COLUMNS,
        "time",
    }
    numeric_columns = [
        column for column in primary_metrics.columns if column not in excluded_columns
    ]
    if not numeric_columns:
        raise RuntimeError("Seed 42 replay metrics contain no continuous columns.")
    numeric_audit: dict[str, dict[str, float]] = {}
    for column in numeric_columns:
        if not (
            pd.api.types.is_numeric_dtype(primary_metrics[column])
            and pd.api.types.is_numeric_dtype(replay_metrics[column])
        ):
            raise RuntimeError(
                f"Seed 42 replay metric column {column!r} is not numeric."
            )
        primary_values = primary_metrics[column].to_numpy(dtype=np.float64)
        replay_values = replay_metrics[column].to_numpy(dtype=np.float64)
        if not (np.isfinite(primary_values).all() and np.isfinite(replay_values).all()):
            raise RuntimeError(
                f"Seed 42 replay metric column {column!r} contains non-finite values."
            )
        difference = primary_values - replay_values
        if not np.allclose(
            primary_values,
            replay_values,
            rtol=REPLAY_METRIC_RTOL,
            atol=REPLAY_METRIC_ATOL,
        ):
            raise RuntimeError(
                f"Seed 42 replay metric column {column!r} exceeds tolerance: "
                f"max_abs={float(np.max(np.abs(difference))):.12g}, "
                f"rtol={REPLAY_METRIC_RTOL:g}, atol={REPLAY_METRIC_ATOL:g}."
            )
        numeric_audit[column] = {
            "max_abs_difference": float(np.max(np.abs(difference))),
            "rmse_difference": float(np.sqrt(np.mean(np.square(difference)))),
        }

    metric_audit = {
        "primary_sha256": _sha256(primary_metrics_path),
        "replay_sha256": _sha256(replay_metrics_path),
        "byte_exact": _sha256(primary_metrics_path) == _sha256(replay_metrics_path),
        "schema_exact": True,
        "n_rows": int(len(primary_metrics)),
        "categorical_columns_exact": list(REPLAY_METRIC_CATEGORICAL_COLUMNS),
        "discrete_columns_exact": list(REPLAY_METRIC_DISCRETE_COLUMNS),
        "time": {
            "atol": REPLAY_TIME_ATOL,
            "max_abs_difference": float(np.max(np.abs(primary_time - replay_time))),
        },
        "numeric_columns": numeric_audit,
    }
    return {
        "status": "PASS",
        "comparison": "fail-closed numerical replay audit",
        "trajectory_tolerance": {
            "rtol": REPLAY_STATE_RTOL,
            "atol": REPLAY_STATE_ATOL,
        },
        "metric_tolerance": {
            "rtol": REPLAY_METRIC_RTOL,
            "atol": REPLAY_METRIC_ATOL,
            "time_atol": REPLAY_TIME_ATOL,
        },
        "primary": primary_hashes,
        "repeat": replay_hashes,
        "trajectories": trajectory_audit,
        "metrics": metric_audit,
    }


def _paper_rc() -> dict[str, Any]:
    return {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial"],
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }


def _common_limits(frames: Sequence[np.ndarray]) -> tuple[float, float, float, float]:
    stacked = np.vstack([np.asarray(frame)[:, :2] for frame in frames])
    if not np.isfinite(stacked).all():
        raise RuntimeError("Cannot plot non-finite spatial coordinates.")
    x0, y0 = np.min(stacked, axis=0)
    x1, y1 = np.max(stacked, axis=0)
    pad_x = max((x1 - x0) * 0.035, 1e-6)
    pad_y = max((y1 - y0) * 0.035, 1e-6)
    return x0 - pad_x, x1 + pad_x, y0 - pad_y, y1 + pad_y


def _aggregate_curves(seed_runs: Mapping[int, Mapping[str, Any]]) -> pd.DataFrame:
    frames = []
    for seed, run in seed_runs.items():
        frame = run["metrics"].copy()
        frame["seed"] = int(seed)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _sem(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=float)
    return float(np.std(array, ddof=1) / np.sqrt(array.size)) if array.size > 1 else 0.0


def _plot_quantitative(
    seed_runs: Mapping[int, Mapping[str, Any]], endpoint: float, output: Path
) -> None:
    index = int(round(endpoint / OUTPUT_STEP))
    seed42 = seed_runs[42]["trajectories"]
    endpoint_frames = [seed42[name][index] for name in CONDITION_LABELS]
    limits = _common_limits(endpoint_frames)
    combined = _aggregate_curves(seed_runs)
    spatial = combined.loc[
        combined["space"].eq("spatial") & (combined["time"] <= endpoint + 1e-9)
    ].copy()

    with mpl.rc_context(_paper_rc()):
        fig = plt.figure(figsize=(10.6, 6.3), constrained_layout=False)
        grid = fig.add_gridspec(2, 6, height_ratios=(1.1, 0.9), hspace=0.38, wspace=0.9)
        for panel_index, name in enumerate(CONDITION_LABELS):
            ax = fig.add_subplot(grid[0, 2 * panel_index : 2 * panel_index + 2])
            points = np.asarray(seed42[name][index])
            ax.scatter(
                points[:, 0],
                points[:, 1],
                s=2.2,
                alpha=0.70,
                c=CONDITION_COLORS[name],
                linewidths=0,
                rasterized=False,
            )
            ax.set_title(f"{CONDITION_LABELS[name]}\nn = {len(points):,}", pad=3)
            ax.set(xlim=limits[:2], ylim=limits[2:])
            ax.set_aspect("equal", adjustable="box")
            ax.axis("off")
        fig.text(0.02, 0.965, "a", fontsize=14, fontweight="bold", va="top")
        fig.text(
            0.055,
            0.965,
            f"Spatial distributions at t = {endpoint:g}",
            fontsize=12,
            fontweight="bold",
            va="top",
        )

        ax_w1 = fig.add_subplot(grid[1, 0:2])
        ax_n = fig.add_subplot(grid[1, 2:4])
        ax_c = fig.add_subplot(grid[1, 4:6])
        for variant in ("remove_YSL", "remove_EVL"):
            rows = spatial.loc[spatial["variant"].eq(variant)]
            grouped = rows.groupby("time", sort=True)["w1"]
            times = np.asarray(list(grouped.groups), dtype=float)
            means = np.asarray([grouped.get_group(t).mean() for t in times])
            sems = np.asarray([_sem(grouped.get_group(t)) for t in times])
            color = CONDITION_COLORS[variant]
            ax_w1.plot(
                times, means, color=color, lw=1.8, label=CONDITION_LABELS[variant]
            )
            ax_w1.fill_between(
                times, means - sems, means + sems, color=color, alpha=0.18, linewidth=0
            )
        ax_w1.set(
            xlabel="Developmental stage",
            ylabel="Spatial W1 from baseline",
            xlim=(0, endpoint),
        )
        ax_w1.legend(frameon=False)
        ax_w1.spines[["top", "right"]].set_visible(False)
        ax_w1.grid(color="#DCE1E5", lw=0.6, alpha=0.7)
        ax_w1.text(
            -0.16, 1.08, "b", transform=ax_w1.transAxes, fontsize=14, fontweight="bold"
        )
        ax_w1.set_title("Spatial W1", loc="left", fontweight="bold")

        for name in CONDITION_LABELS:
            values_by_seed = []
            for seed, run in seed_runs.items():
                values_by_seed.append(
                    [len(frame) for frame in run["trajectories"][name][: index + 1]]
                )
            counts = np.asarray(values_by_seed, dtype=float)
            times = _time_grid(endpoint)
            mean = counts.mean(axis=0)
            sem = counts.std(axis=0, ddof=1) / np.sqrt(counts.shape[0])
            color = CONDITION_COLORS[name]
            ax_n.plot(times, mean, color=color, lw=1.7, label=CONDITION_LABELS[name])
            ax_n.fill_between(
                times, mean - sem, mean + sem, color=color, alpha=0.15, linewidth=0
            )
        ax_n.set(
            xlabel="Developmental stage", ylabel="Particle count", xlim=(0, endpoint)
        )
        ax_n.legend(frameon=False)
        ax_n.spines[["top", "right"]].set_visible(False)
        ax_n.grid(color="#DCE1E5", lw=0.6, alpha=0.7)
        ax_n.text(
            -0.16, 1.08, "c", transform=ax_n.transAxes, fontsize=14, fontweight="bold"
        )
        ax_n.set_title("Learned population trajectory", loc="left", fontweight="bold")

        variants = ("remove_YSL", "remove_EVL")
        positions = np.arange(2)
        for pos, variant in zip(positions, variants):
            values = spatial.loc[
                spatial["variant"].eq(variant)
                & np.isclose(spatial["time"], endpoint, atol=1e-9),
                "centroid_shift",
            ].to_numpy(dtype=float)
            mean = float(values.mean())
            ci = 2.776445 * _sem(values)
            color = CONDITION_COLORS[variant]
            ax_c.bar(pos, mean, color=color, alpha=0.62, width=0.55)
            ax_c.errorbar(
                pos, mean, yerr=ci, fmt="none", color="#26323A", capsize=3, lw=1
            )
            offsets = np.linspace(-0.055, 0.055, len(values))
            ax_c.scatter(
                np.full_like(values, pos) + offsets,
                values,
                s=26,
                facecolor="white",
                edgecolor=color,
                lw=1.2,
                zorder=3,
                rasterized=False,
            )
        ax_c.set_xticks(positions, ["YSL removal", "EVL removal"])
        ax_c.set_ylabel("Centroid shift from baseline")
        ax_c.spines[["top", "right"]].set_visible(False)
        ax_c.grid(axis="y", color="#DCE1E5", lw=0.6, alpha=0.7)
        ax_c.text(
            -0.16, 1.08, "d", transform=ax_c.transAxes, fontsize=14, fontweight="bold"
        )
        ax_c.set_title(
            f"Centroid shift at t = {endpoint:g}", loc="left", fontweight="bold"
        )

        fig.subplots_adjust(left=0.075, right=0.985, bottom=0.10, top=0.91)
        fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
        fig.savefig(
            output.with_suffix(".png"), dpi=320, bbox_inches="tight", facecolor="white"
        )
        plt.close(fig)


def _load_trajectory_labeler(
    classifier_cache: Path, classifier_pca: Path, *, device: str
):
    cache = cb.tl.load_cached_mlp_classifier(str(classifier_cache), device=device)
    classifier_contract = {
        "cache_tag": cache.metadata.get("cache_tag"),
        "feature_dim": cache.feature_dim,
        "include_time_feature": cache.include_time_feature,
        "label_col": cache.label_col,
    }
    expected_classifier_contract = {
        "cache_tag": "zebrafish-paper-ablation-spatial2-pca10",
        "feature_dim": 12,
        "include_time_feature": True,
        "label_col": "Annotation",
    }
    if classifier_contract != expected_classifier_contract:
        raise RuntimeError(
            "Classic S24 requires the formal ablation classifier, not the main "
            f"trajectory classifier: expected={expected_classifier_contract}, "
            f"actual={classifier_contract}."
        )
    with np.load(classifier_pca, allow_pickle=False) as archive:
        components = np.asarray(archive["components"], dtype=np.float32)
        mean = np.asarray(archive["mean"], dtype=np.float32)
    if components.shape != (10, 50) or mean.shape != (50,):
        raise RuntimeError(
            "Classic S24 classifier PCA must contain components 10x50 and mean 50; "
            f"got {components.shape} and {mean.shape}."
        )

    def label(frame: np.ndarray, time_value: float) -> np.ndarray:
        points = np.asarray(frame, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 52:
            raise RuntimeError(
                f"Classic S24 label input must be N x 52, got {points.shape}."
            )
        reduced = (points[:, 2:] - mean) @ components.T
        features = np.hstack((points[:, :2], reduced)).astype(np.float32)
        return cb.tl.predict_labels_for_points(
            points=features,
            time_value=float(time_value),
            model=cache.model,
            label_encoder=cache.label_encoder,
            feature_dim=12,
            device=device,
            knn_neighbors=10,
            include_time_feature=True,
            spatial_coords=points[:, :2],
        )

    return label


def _plot_time_grid(
    seed_runs: Mapping[int, Mapping[str, Any]],
    endpoint: float,
    output: Path,
    *,
    trajectory_labeler=None,
    label_to_color: Mapping[str, str] | None = None,
) -> None:
    display_times = [value for value in OBSERVED_ENDPOINTS if value <= endpoint + 1e-9]
    seed42 = seed_runs[42]["trajectories"]
    frames = [
        np.asarray(seed42[name][int(round(time / OUTPUT_STEP))])
        for time in display_times
        for name in CONDITION_LABELS
    ]
    limits = _common_limits(frames)
    labels_seen: set[str] = set()
    with mpl.rc_context(_paper_rc()):
        fig, axes = plt.subplots(
            len(display_times),
            3,
            figsize=(7.2, 2.0 * len(display_times)),
            squeeze=False,
        )
        for row, time in enumerate(display_times):
            index = int(round(time / OUTPUT_STEP))
            for col, name in enumerate(CONDITION_LABELS):
                ax = axes[row, col]
                points = np.asarray(seed42[name][index])
                if trajectory_labeler is None:
                    colors = CONDITION_COLORS[name]
                else:
                    labels = np.asarray(trajectory_labeler(points, time)).astype(str)
                    labels_seen.update(labels.tolist())
                    colors = [
                        (label_to_color or {}).get(label, "#9E9E9E") for label in labels
                    ]
                ax.scatter(
                    points[:, 0],
                    points[:, 1],
                    s=1.5,
                    alpha=0.75,
                    c=colors,
                    linewidths=0,
                    rasterized=False,
                )
                if row == 0:
                    ax.set_title(CONDITION_LABELS[name], fontweight="bold")
                if col == 0:
                    ax.text(
                        -0.02,
                        0.5,
                        f"t = {time:g}\nn = {len(points):,}",
                        transform=ax.transAxes,
                        ha="right",
                        va="center",
                        fontsize=8,
                    )
                else:
                    ax.text(
                        0.02,
                        0.02,
                        f"n = {len(points):,}",
                        transform=ax.transAxes,
                        ha="left",
                        va="bottom",
                        fontsize=7,
                    )
                ax.set(xlim=limits[:2], ylim=limits[2:])
                ax.set_aspect("equal", adjustable="box")
                ax.axis("off")
        if trajectory_labeler is not None and labels_seen:
            handles = [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="none",
                    markersize=3.5,
                    markerfacecolor=(label_to_color or {}).get(label, "#9E9E9E"),
                    markeredgewidth=0,
                    label=label,
                )
                for label in sorted(labels_seen)
            ]
            fig.legend(
                handles=handles,
                loc="center left",
                bbox_to_anchor=(0.995, 0.5),
                frameon=False,
                fontsize=5.5,
                ncol=1,
            )
        fig.subplots_adjust(
            left=0.12,
            right=(0.80 if trajectory_labeler is not None else 0.99),
            bottom=0.02,
            top=0.96,
            hspace=0.05,
            wspace=0.05,
        )
        fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
        fig.savefig(
            output.with_suffix(".png"), dpi=320, bbox_inches="tight", facecolor="white"
        )
        plt.close(fig)


def report(args: argparse.Namespace) -> int:
    run_root = Path(args.run_root).expanduser().resolve()
    replay_dir = Path(args.seed42_repeat).expanduser().resolve()
    primary_seed42_dir = (run_root / "seeds" / "seed_42").resolve()
    if replay_dir == primary_seed42_dir:
        raise ValueError(
            "--seed42-repeat must be an independently executed directory, not "
            "the primary seed_42 directory."
        )
    output_dir = _fresh_directory(
        Path(args.output_dir), label="report output directory"
    )
    seed_runs = {
        seed: _load_seed(run_root / "seeds" / f"seed_{seed}", expected_seed=seed)
        for seed in FORMAL_SEEDS
    }
    replay_run = _load_seed(replay_dir, expected_seed=42)
    replay = _require_seed42_replay(run_root, replay_dir)

    reference_inputs = seed_runs[42]["summary"]["inputs"]
    reference_end_time = float(seed_runs[42]["summary"]["protocol"]["end_time"])
    for seed, run in seed_runs.items():
        if run["summary"]["inputs"] != reference_inputs:
            raise RuntimeError(f"Seed {seed} does not bind the same accepted inputs.")
        if float(run["summary"]["protocol"]["end_time"]) != reference_end_time:
            raise RuntimeError(f"Seed {seed} uses a different simulation horizon.")
    if replay_run["summary"]["inputs"] != reference_inputs:
        raise RuntimeError("The seed-42 replay does not bind the accepted inputs.")
    if float(replay_run["summary"]["protocol"]["end_time"]) != reference_end_time:
        raise RuntimeError("The seed-42 replay uses a different simulation horizon.")
    endpoint = _latest_common_endpoint(seed_runs)
    quantitative_stem = output_dir / "zebrafish_classic_s24_unequalN_five_seed"
    time_grid_stem = output_dir / "zebrafish_classic_s24_morphology_grid"
    classifier_cache = Path(args.classifier_cache).expanduser().resolve()
    classifier_pca = Path(args.classifier_pca).expanduser().resolve()
    labeler = _load_trajectory_labeler(
        classifier_cache, classifier_pca, device=str(args.classifier_device)
    )
    label_to_color = (
        seed_runs[42]["summary"].get("plot_metadata", {}).get("label_to_color", {})
    )
    if not isinstance(label_to_color, Mapping) or not label_to_color:
        raise RuntimeError("Seed summary lacks the accepted cell-type color mapping.")
    _plot_quantitative(seed_runs, endpoint, quantitative_stem)
    _plot_time_grid(
        seed_runs,
        endpoint,
        time_grid_stem,
        trajectory_labeler=labeler,
        label_to_color=label_to_color,
    )

    combined = _aggregate_curves(seed_runs)
    combined_path = output_dir / "all_seed_metrics.csv"
    combined.to_csv(combined_path, index=False, float_format="%.12g")
    endpoint_rows = combined.loc[
        combined["space"].eq("spatial")
        & np.isclose(combined["time"], endpoint, atol=1e-9)
    ]
    summary_rows = []
    for variant, rows in endpoint_rows.groupby("variant", sort=False):
        summary_rows.append(
            {
                "variant": variant,
                "endpoint": endpoint,
                "n_seeds": int(rows["seed"].nunique()),
                "spatial_w1_mean": float(rows["w1"].mean()),
                "spatial_w1_sem": _sem(rows["w1"]),
                "spatial_w2_mean": float(rows["w2"].mean()),
                "spatial_w2_sem": _sem(rows["w2"]),
                "centroid_shift_mean": float(rows["centroid_shift"].mean()),
                "centroid_shift_sem": _sem(rows["centroid_shift"]),
                "baseline_n_mean": float(rows["n_baseline"].mean()),
                "ablation_n_mean": float(rows["n_ablation"].mean()),
                "count_ratio_mean": float(rows["count_ratio"].mean()),
            }
        )
    endpoint_summary = pd.DataFrame(summary_rows)
    endpoint_path = output_dir / "endpoint_summary.csv"
    endpoint_summary.to_csv(endpoint_path, index=False, float_format="%.12g")

    caption = {
        "title": "Classic unequal-population zebrafish virtual-removal sensitivity",
        "text": (
            f"Five simulation seeds (42–46) propagate one complete 563-cell t=0 "
            f"cohort continuously to t={endpoint:g}. The YSL and EVL branches are "
            "exact initial subsets after removing 29 YSL cells (n=534) or 272 EVL "
            "cells (n=291), respectively. Learned growth-driven split/extinction, "
            "diffusion, score correction, velocity drift, and interaction remain "
            "enabled; no observed slice re-anchors or warps a trajectory. Spatial "
            "snapshots and the morphology grid show the preregistered seed-42 run. "
            "W1 uses uniform empirical OT on deterministic supports capped at "
            "1,024 points per cloud and therefore measures shape/location rather "
            "than total population; N(t) reports the learned "
            "population response separately. Spatial-W1 and N(t) curves show the "
            "five-seed mean ± SEM; "
            "centroid bars show the mean with a two-sided 95% t interval and all "
            "individual seed values. Seeds quantify simulation-stream "
            "variability conditional on one frozen checkpoint, not biological or "
            "training uncertainty. This is a virtual-removal model sensitivity, "
            "not a causal knockout. The formal endpoint is the latest observed time "
            "for which every frame through that endpoint passes the predeclared "
            "latent-support gate for all 15 trajectories."
        ),
        "endpoint": endpoint,
        "t4_used": bool(np.isclose(endpoint, 4.0)),
        "matched_equal_n_control_is_separate": True,
    }
    caption_path = output_dir / "CAPTION.json"
    _write_json(caption_path, caption)

    output_paths = [
        quantitative_stem.with_suffix(".pdf"),
        quantitative_stem.with_suffix(".png"),
        time_grid_stem.with_suffix(".pdf"),
        time_grid_stem.with_suffix(".png"),
        combined_path,
        endpoint_path,
        caption_path,
    ]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "analysis": "zebrafish_classic_s24_unequal_population_five_seed_report",
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "formal_endpoint": endpoint,
        "support_gate": {
            "all_15_trajectories_required": True,
            "nonfinite_allowed": False,
            "maximum_fraction_outside_observed_latent_norm_max": SUPPORT_MAX_OUTSIDE_FRACTION,
            "maximum_generated_latent_norm_multiplier": SUPPORT_MAX_NORM_MULTIPLIER,
            "no_clipping_or_outlier_removal": True,
        },
        "seed42_deterministic_replay": replay,
        "input_seed_summaries": {
            str(seed): _file_record(run["summary_path"])
            for seed, run in seed_runs.items()
        },
        "classifier": {
            "cache": _file_record(classifier_cache),
            "pca": _file_record(classifier_pca),
            "feature_contract": "time + spatial2 + PCA10(original latent50)",
            "spatial_knn_neighbors": 10,
        },
        "outputs": [_file_record(path) for path in output_paths],
        "code": {"runner": _file_record(SCRIPT_PATH)},
    }
    manifest_path = output_dir / "report_manifest.json"
    _write_json(manifest_path, manifest)
    _write_sha256_sidecar(manifest_path)
    print(
        json.dumps(
            {
                "status": "complete",
                "endpoint": endpoint,
                "report_manifest": str(manifest_path),
            }
        )
    )
    return 0


def _add_common_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--aligned-h5ad", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--acceptance-report", type=Path, required=True)
    parser.add_argument("--expected-acceptance-sha256", required=True)
    parser.add_argument("--shared-cache-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    seed_parser = subparsers.add_parser(
        "run-seed", help="Run one formal simulation seed."
    )
    _add_common_inputs(seed_parser)
    seed_parser.add_argument("--seed", type=int, required=True)
    seed_parser.add_argument(
        "--end-time", type=float, choices=(3.0, 4.0), required=True
    )
    seed_parser.add_argument("--output-dir", type=Path, required=True)
    report_parser = subparsers.add_parser(
        "report", help="Validate and plot seeds 42–46."
    )
    report_parser.add_argument("--run-root", type=Path, required=True)
    report_parser.add_argument("--seed42-repeat", type=Path, required=True)
    report_parser.add_argument("--classifier-cache", type=Path, required=True)
    report_parser.add_argument("--classifier-pca", type=Path, required=True)
    report_parser.add_argument("--classifier-device", default="cpu")
    report_parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    _require_import_origins()
    args = build_parser().parse_args(argv)
    if args.command == "run-seed":
        return run_seed(args)
    if args.command == "report":
        return report(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
