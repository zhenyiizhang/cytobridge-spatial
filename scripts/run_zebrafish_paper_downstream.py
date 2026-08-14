#!/usr/bin/env python3
"""Reproduce the zebrafish manuscript downstream analyses from a native model.

This runner consumes the clean-counts aligned AnnData and a *current* CytoBridge
six-stage training directory.  It intentionally does not read historical
interpolated H5AD files, cached communication pickles, or copied manuscript
PDFs.  Every biological result is regenerated through public CytoBridge APIs.

The manuscript stages use two explicitly separated state contracts:

* S22 is a single unwarped global-t0 fixed-population state transport on a
  fixed dense grid.  One real t=0 cohort evolves continuously through t=4
  under drift, score, interaction, and diffusion; learned growth-driven
  birth/extinction is disabled.  Real integer-time slices are exported only as
  a separate reference series and never replace generated trajectory frames.
* S25 and communication retain their historical hybrid reconstruction contract
  for now: observed cells at integer times and interval-local generated cells
  at intermediate times.  They do not implicitly consume the S22 global-t0
  bundle, because doing so would silently change their scientific estimand.
* S24 uses separate YSL- and EVL-excluded, equal-N, deterministic
  fixed-population global-t0 cohorts through observed t=3.  Diffusion and
  learned growth are disabled while drift, score, and interaction remain
  active.  It is a preterminal spatial model-sensitivity analysis, not
  terminal t=4 evidence, total-mass deletion, a causal knockout, canonical
  reconstruction evidence, or a lineage analysis.
* Communication uses a hybrid state population: observed cells at integer
  times and interval-local generated cells at intermediate times. This
  state-source choice is separate from the LR expression measurement policy.

Stages are resumable.  A completed stage is skipped when its input/settings
signature and every recorded output still match; pass ``--force`` to rerun it.

Examples
--------
Run all manuscript analyses::

    python scripts/run_zebrafish_paper_downstream.py \
      --aligned-h5ad MATCHED_RUN/zebrafish/preprocess/zebrafish_aligned.h5ad \
      --model-dir MATCHED_RUN/zebrafish/training \
      --acceptance-report MATCHED_RUN/matched_ablation_acceptance.json \
      --expected-acceptance-sha256 <exact-sha256> \
      --lr-database /accepted/assets/CellChatDB.ligrec.zebrafish.csv \
      --output-dir MATCHED_RUN/zebrafish/paper_downstream \
      --stage all --profile full --device cuda

Resume only S25 and communication::

    python scripts/run_zebrafish_paper_downstream.py ... \
      --stage s25,communication

The default shared classifier cache is placed beside the preprocessing folder,
so two alpha-expression conditions trained from the same aligned H5AD reuse the
same label classifiers.  Use ``--shared-cache-dir`` to choose another location.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Callable, Mapping, Optional, Sequence

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import CytoBridge as cb  # noqa: E402


ALL_STAGES = (
    "classifier",
    "velocity",
    "s22",
    "growth",
    "ablation",
    "s25",
    "communication",
)
OBSERVED_TIMES = (0.0, 1.0, 2.0, 3.0, 4.0)
HALF_TIMES = tuple(float(value) for value in np.arange(0.0, 4.0 + 0.5, 0.5))
MAIN_CLASSIFIER_CACHE_TAG = "zebrafish-paper-main-spatial2-latent10"
S22_FIXED_GROWTH_ALPHA = 0.0
S22_TRAJECTORY_MODE = "global_t0_fixed_population_state_transport"
S22_TRAJECTORY_SCOPE = (
    "single continuous unwarped fixed-population state transport initialized "
    "once from the real t=0 cohort and propagated through t=4 with learned "
    "drift, score, interaction, and stochastic diffusion retained but learned "
    "growth-driven birth/extinction disabled; every displayed trajectory frame, "
    "including integer times after t=0, is generated from that same t=0 "
    "initialization; observed integer-time slices are separate references only; "
    "not an abundance forecast or reconstruction of observed stages"
)
S25_COMMUNICATION_TRAJECTORY_SCOPE = (
    "piecewise observed-anchored interval-local one-sided forward simulation; "
    "each generated slice starts from the preceding observed anchor and is not "
    "conditioned on the following observed endpoint; not global-t0 and not "
    "lineage-continuous"
)
S22_MOSAIC_COLUMNS = 3
S25_HEATMAP_COLUMNS = 2
S24_PROTOCOL = "preterminal_t3_sigma0"
S24_END_TIME = 3.0
S24_PUBLICATION_TIMES = (0.0, 1.0, 2.0, 3.0)
S24_FIXED_DT = 0.005
S24_FIXED_SIGMA = 0.0
S24_FIXED_GROWTH_ALPHA = 0.0
S24_INTERACTION_M = 1024
S24_INTERACTION_SEED_OFFSET = 10_001
S24_SUPPORT_MAX_OUTSIDE_FRACTION = 0.01
S24_SUPPORT_MAX_NORM_MULTIPLIER = 2.0
MATCHED_ACCEPTANCE_KEY = "canonical_matched_acceptance"
MATCHED_ACCEPTANCE_REQUIRED_EXACT = {
    "status": "PASS",
    "datasets": {"zebrafish": {"status": "PASS"}},
}
IMPLEMENTATION_SOURCE_RELATIVE_PATHS = tuple(
    sorted(
        path.relative_to(REPO_ROOT).as_posix()
        for path in (REPO_ROOT / "CytoBridge").rglob("*.py")
        if path.is_file()
    )
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_ready(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_json_ready(item) for item in value.tolist()]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(item) for item in value]
    return value


def _stable_hash(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        _json_ready(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _git_revision() -> dict[str, object]:
    try:
        revision = subprocess.run(
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
        return {"commit": revision, "dirty": dirty}
    except Exception:
        return {"commit": None, "dirty": None}


def _require_file(path: str | Path, description: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Missing {description}: {resolved}")
    return resolved


def _require_dir(path: str | Path, description: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"Missing {description}: {resolved}")
    return resolved


def _lexical_absolute_path(path: str | Path) -> Path:
    """Return an absolute normalized path without resolving its final symlink."""

    return Path(os.path.abspath(os.fspath(Path(path).expanduser())))


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _load_matched_acceptance_report(
    path: str | Path,
    expected_sha256: str,
) -> tuple[Path, str, Mapping[str, Any], Path]:
    """Load the exact canonical matched acceptance report, failing closed."""

    report_path = _require_file(path, "canonical matched acceptance report")
    expected = str(expected_sha256).strip().lower()
    if len(expected) != 64 or any(
        character not in "0123456789abcdef" for character in expected
    ):
        raise ValueError("--expected-acceptance-sha256 must be exactly 64 hex digits")
    observed = _sha256(report_path)
    if observed != expected:
        raise RuntimeError(
            "Canonical matched acceptance SHA-256 mismatch: "
            f"expected {expected}, observed {observed}"
        )
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Cannot read canonical matched acceptance JSON {report_path}: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise RuntimeError("Canonical matched acceptance report must be a JSON object")
    datasets = payload.get("datasets")
    zebrafish = datasets.get("zebrafish") if isinstance(datasets, Mapping) else None
    if payload.get("status") != "PASS":
        raise RuntimeError("Canonical matched acceptance report is not overall PASS")
    if not isinstance(zebrafish, Mapping) or zebrafish.get("status") != "PASS":
        raise RuntimeError(
            "Canonical matched acceptance report does not record "
            "datasets.zebrafish status PASS"
        )
    raw_run_root = payload.get("run_root")
    if not isinstance(raw_run_root, str) or not raw_run_root.strip():
        raise RuntimeError(
            "Canonical matched acceptance report lacks a non-empty run_root"
        )
    declared_root = Path(raw_run_root).expanduser()
    if not declared_root.is_absolute():
        raise RuntimeError("Canonical matched acceptance run_root must be absolute")
    run_root = declared_root.resolve()
    if not run_root.is_dir():
        raise RuntimeError(
            f"Canonical matched acceptance run_root is missing: {run_root}"
        )
    return report_path, observed, payload, run_root


def _require_zebrafish_matched_path(
    path: str | Path,
    *,
    run_root: Path,
    description: str,
    allow_final_symlink: bool,
) -> Path:
    """Require a formal input entry under the accepted full zebrafish profile."""

    candidate = _lexical_absolute_path(path)
    try:
        relative = candidate.relative_to(run_root)
    except ValueError as exc:
        raise RuntimeError(
            f"{description} is outside canonical matched run_root {run_root}: "
            f"{candidate}"
        ) from exc
    if not relative.parts or relative.parts[0] != "zebrafish":
        raise RuntimeError(
            f"{description} must belong to the full zebrafish profile under "
            f"canonical matched run_root {run_root}: {candidate}"
        )
    profile_root = run_root / "zebrafish"
    resolved_scope = (
        candidate.parent.resolve() if allow_final_symlink else candidate.resolve()
    )
    if not _is_relative_to(resolved_scope, profile_root):
        raise RuntimeError(
            f"{description} escapes the full zebrafish profile through a symlink: "
            f"{candidate}"
        )
    return candidate


def _require_formal_acceptance_cli(args: argparse.Namespace) -> None:
    report = getattr(args, "acceptance_report", None)
    digest = getattr(args, "expected_acceptance_sha256", None)
    required = str(getattr(args, "profile", "full")) == "full"
    if required and (report is None or digest is None):
        raise ValueError(
            "--acceptance-report and --expected-acceptance-sha256 are required "
            "for --profile full"
        )
    if (report is None) != (digest is None):
        raise ValueError(
            "--acceptance-report and --expected-acceptance-sha256 must be "
            "provided together"
        )


def _build_matched_acceptance_binding(
    args: argparse.Namespace,
    *,
    aligned_h5ad: str | Path,
    model_dir: str | Path,
) -> dict[str, object] | None:
    """Bind formal downstream work to one exact accepted matched run."""

    _require_formal_acceptance_cli(args)
    if args.acceptance_report is None:
        return None
    report_path, digest, _payload, run_root = _load_matched_acceptance_report(
        args.acceptance_report,
        args.expected_acceptance_sha256,
    )
    aligned_entry = _require_zebrafish_matched_path(
        aligned_h5ad,
        run_root=run_root,
        description="aligned zebrafish H5AD path",
        allow_final_symlink=True,
    )
    model_entry = _require_zebrafish_matched_path(
        model_dir,
        run_root=run_root,
        description="native six-stage model directory",
        allow_final_symlink=False,
    )
    return {
        "path": str(report_path),
        "sha256": digest,
        "required_exact": _json_ready(MATCHED_ACCEPTANCE_REQUIRED_EXACT),
        "observed_run_root": str(run_root),
        "matched_profile": "zebrafish",
        "aligned_h5ad_entry": str(aligned_entry),
        "model_dir_entry": str(model_entry),
    }


def _require_acceptance_binding_current(binding: Mapping[str, object]) -> None:
    """Rehash a signed acceptance record before reuse or publication."""

    if binding.get("required_exact") != MATCHED_ACCEPTANCE_REQUIRED_EXACT:
        raise RuntimeError(
            "Canonical matched acceptance binding has stale required_exact assertions"
        )
    report_path, digest, _payload, run_root = _load_matched_acceptance_report(
        str(binding.get("path", "")),
        str(binding.get("sha256", "")),
    )
    if str(report_path) != str(binding.get("path")) or digest != binding.get("sha256"):
        raise RuntimeError("Canonical matched acceptance path/SHA binding changed")
    if str(run_root) != binding.get("observed_run_root"):
        raise RuntimeError("Canonical matched acceptance run_root binding changed")
    _require_zebrafish_matched_path(
        str(binding.get("aligned_h5ad_entry", "")),
        run_root=run_root,
        description="bound aligned zebrafish H5AD path",
        allow_final_symlink=True,
    )
    _require_zebrafish_matched_path(
        str(binding.get("model_dir_entry", "")),
        run_root=run_root,
        description="bound native six-stage model directory",
        allow_final_symlink=False,
    )


def _context_acceptance_binding(ctx: object) -> Mapping[str, object] | None:
    common = getattr(ctx, "common_signature", {})
    binding = (
        common.get(MATCHED_ACCEPTANCE_KEY) if isinstance(common, Mapping) else None
    )
    if binding is None:
        return None
    if not isinstance(binding, Mapping):
        raise RuntimeError("Canonical matched acceptance binding must be an object")
    _require_acceptance_binding_current(binding)
    return binding


def _time_grid(start: float, end: float, step: float) -> list[float]:
    if not np.isfinite(step) or float(step) <= 0:
        raise ValueError("time-grid step must be finite and > 0")
    span = float(end) - float(start)
    n_steps = int(round(span / float(step)))
    if not np.isclose(n_steps * float(step), span, rtol=0.0, atol=1e-8):
        raise ValueError(f"step={step} does not exactly tile interval [{start}, {end}]")
    return [
        float(round(float(start) + i * float(step), 10)) for i in range(n_steps + 1)
    ]


def _parse_stages(raw: str) -> list[str]:
    values = [value.strip().lower() for value in str(raw).split(",") if value.strip()]
    if not values:
        raise ValueError("--stage must not be empty")
    if "all" in values:
        if len(values) != 1:
            raise ValueError(
                "Use --stage all by itself, or provide a comma-separated subset"
            )
        return list(ALL_STAGES)
    unknown = sorted(set(values).difference(ALL_STAGES))
    if unknown:
        raise ValueError(f"Unknown stages {unknown}; valid stages: {list(ALL_STAGES)}")
    selected = set(values)
    # Always execute in dependency order, even if a caller writes
    # ``--stage communication,s25``.
    return [stage for stage in ALL_STAGES if stage in selected]


def _parse_csv_strings(raw: str) -> list[str]:
    values = [value.strip().lower() for value in str(raw).split(",") if value.strip()]
    if not values:
        raise ValueError("Expected at least one comma-separated value")
    return list(dict.fromkeys(values))


def _joint_features(
    adata: ad.AnnData,
    *,
    latent_key: str,
    spatial_key: str,
) -> np.ndarray:
    if latent_key not in adata.obsm:
        raise KeyError(f"adata.obsm['{latent_key}'] is required")
    if spatial_key not in adata.obsm:
        raise KeyError(f"adata.obsm['{spatial_key}'] is required")
    spatial = np.asarray(adata.obsm[spatial_key], dtype=np.float32)
    latent = np.asarray(adata.obsm[latent_key], dtype=np.float32)
    if spatial.ndim != 2 or spatial.shape[1] != 2:
        raise ValueError(
            f"Zebrafish spatial coordinates must be N x 2, got {spatial.shape}"
        )
    if spatial.shape[0] != latent.shape[0]:
        raise ValueError("Spatial and latent row counts do not match")
    return np.hstack((spatial, latent)).astype(np.float32, copy=False)


def _label_colors(
    adata: ad.AnnData,
    *,
    annotation_key: str,
    color_key: Optional[str],
) -> dict[str, str]:
    labels = adata.obs[annotation_key].astype(str).to_numpy()
    if color_key and color_key in adata.obs:
        table = pd.DataFrame(
            {
                "label": labels,
                "color": adata.obs[color_key].astype(str).to_numpy(),
            }
        )
        mapping: dict[str, str] = {}
        for label, subset in table.groupby("label", sort=False):
            colors = subset["color"].dropna().astype(str)
            colors = colors.loc[
                colors.str.strip().ne("") & colors.str.lower().ne("nan")
            ]
            if not colors.empty:
                mapping[str(label)] = str(colors.mode().iloc[0])
        if set(np.unique(labels)).issubset(mapping):
            return mapping
    return cb.tl.load_label_to_color(labels)


def _minimal_state_adata(
    points: np.ndarray,
    labels: Sequence[object],
    *,
    annotation_key: str,
) -> ad.AnnData:
    points = np.asarray(points, dtype=np.float32)
    labels = np.asarray(labels).astype(str)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"State points must be N x D (D>=3), got {points.shape}")
    if labels.shape[0] != points.shape[0]:
        raise ValueError("State point and label counts do not match")
    result = ad.AnnData(X=points)
    result.obs[annotation_key] = labels
    result.obsm["spatial"] = points[:, :2].copy()
    return result


def _write_state_bundle(
    adata_dict: Mapping[str, ad.AnnData],
    time_points: Sequence[float],
    output_dir: str | Path,
    *,
    annotation_key: str,
    source_by_time: Optional[Mapping[float, str]] = None,
) -> list[Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    files: list[Path] = []
    for index, time_value in enumerate(time_points):
        key = str(float(time_value))
        if key not in adata_dict:
            raise KeyError(f"State dictionary is missing time key {key!r}")
        state = adata_dict[key]
        points = np.asarray(state.X, dtype=np.float32)
        # Store fixed-width Unicode, not an object array, so bundles remain
        # loadable with NumPy's safe ``allow_pickle=False`` default.
        labels = np.asarray(state.obs[annotation_key].astype(str).tolist(), dtype=str)
        filename = f"frame_{index:03d}.npz"
        path = output / filename
        np.savez_compressed(path, points=points, labels=labels)
        files.append(path)
        records.append(
            {
                "index": int(index),
                "time": float(time_value),
                "key": key,
                "file": filename,
                "sha256": _sha256(path),
                "n_cells": int(points.shape[0]),
                "feature_dim": int(points.shape[1]),
                "source": (
                    None
                    if source_by_time is None
                    else str(source_by_time[float(time_value)])
                ),
            }
        )
    index_path = output / "index.json"
    index_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "annotation_key": annotation_key,
                "frames": records,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    files.append(index_path)
    return files


def _read_state_bundle(
    bundle_dir: str | Path,
    *,
    annotation_key: str,
) -> tuple[dict[str, ad.AnnData], list[float], dict[float, str]]:
    bundle = Path(bundle_dir)
    index_path = _require_file(bundle / "index.json", "trajectory-state index")
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    states: dict[str, ad.AnnData] = {}
    times: list[float] = []
    sources: dict[float, str] = {}
    for record in payload["frames"]:
        time_value = float(record["time"])
        frame_path = _require_file(bundle / record["file"], "trajectory frame")
        expected_sha256 = record.get("sha256")
        if expected_sha256 is not None and _sha256(frame_path) != str(expected_sha256):
            raise RuntimeError(f"Trajectory frame checksum mismatch: {frame_path}")
        frame = np.load(frame_path)
        points = np.asarray(frame["points"], dtype=np.float32)
        labels = np.asarray(frame["labels"]).astype(str)
        states[str(time_value)] = _minimal_state_adata(
            points, labels, annotation_key=annotation_key
        )
        times.append(time_value)
        if record.get("source") is not None:
            sources[time_value] = str(record["source"])
    return states, times, sources


@dataclass
class RunContext:
    args: argparse.Namespace
    adata: ad.AnnData
    df: pd.DataFrame
    loaded: object
    runtime: object
    dim: int
    spatial_dim: int
    output_dir: Path
    shared_cache_dir: Path
    label_to_color: dict[str, str]
    common_signature: dict[str, object]


def _stage_manifest_path(ctx: RunContext, stage: str) -> Path:
    return ctx.output_dir / stage / "stage_manifest.json"


def _recorded_outputs_exist(manifest: Mapping[str, object]) -> bool:
    outputs = manifest.get("outputs", [])
    if not outputs:
        return False
    artifacts = manifest.get("output_artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != len(outputs):
        # Backward-compatible handling for manifests written before checksums
        # were recorded.  Empty files are never considered resumable outputs.
        return all(
            Path(str(path)).is_file() and Path(str(path)).stat().st_size > 0
            for path in outputs
        )
    for record in artifacts:
        if not isinstance(record, Mapping):
            return False
        path = Path(str(record.get("path", "")))
        if not path.is_file() or path.stat().st_size <= 0:
            return False
        if int(record.get("size_bytes", -1)) != int(path.stat().st_size):
            return False
        if str(record.get("sha256", "")) != _sha256(path):
            return False
    return True


def _require_manifest_current_common_signature(
    ctx: RunContext,
    manifest: Mapping[str, object],
    manifest_path: str | Path,
    *,
    stage: str,
) -> None:
    """Require a stage manifest signature to bind the current common contract."""

    source = Path(manifest_path)
    if manifest.get("status") != "complete":
        raise RuntimeError(f"Upstream stage {stage!r} is not complete: {source}")
    expected_signature = _stable_hash(
        {
            "stage": stage,
            "common": ctx.common_signature,
            "settings": manifest.get("settings", {}),
        }
    )
    if manifest.get("signature") != expected_signature:
        raise RuntimeError(
            f"Upstream stage {stage!r} was produced by different data/model/code "
            f"settings. Rerun that stage before consuming it: {source}"
        )
    current_acceptance = _context_acceptance_binding(ctx)
    recorded_acceptance = manifest.get(MATCHED_ACCEPTANCE_KEY)
    if current_acceptance is not None and recorded_acceptance != current_acceptance:
        raise RuntimeError(
            f"Upstream stage {stage!r} is missing or was produced under a stale "
            f"canonical matched acceptance binding: {source}"
        )
    if current_acceptance is None and recorded_acceptance is not None:
        raise RuntimeError(
            f"Upstream stage {stage!r} has an unexpected canonical matched "
            f"acceptance binding: {source}"
        )


def _require_current_stage_manifest(ctx: RunContext, stage: str) -> dict[str, object]:
    """Load an upstream stage only when it matches the current run contract."""
    manifest_path = _stage_manifest_path(ctx, stage)
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Current {stage!r} stage manifest is required: {manifest_path}."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require_manifest_current_common_signature(
        ctx, manifest, manifest_path, stage=stage
    )
    if not _recorded_outputs_exist(manifest):
        raise RuntimeError(
            f"Upstream stage {stage!r} has missing or modified outputs: {manifest_path}"
        )
    return manifest


def _expected_s22_state_settings(
    ctx: RunContext, *, growth_alpha: Optional[float] = None
) -> dict[str, object]:
    """Return the current settings that determine reusable canonical S22 states."""

    effective_growth_alpha = (
        float(ctx.args.growth_alpha) if growth_alpha is None else float(growth_alpha)
    )
    return {
        "dt": float(ctx.args.sde_dt),
        "split_resample_dt": float(ctx.args.sde_dt),
        "sigma": float(ctx.args.sde_sigma),
        "daughter_noise_std": 0.0,
        "growth_alpha": effective_growth_alpha,
        "interaction_m": int(ctx.args.interaction_m),
        "sde_n_samples": (
            int(ctx.args.smoke_n_samples)
            if ctx.args.profile == "smoke"
            else ctx.args.sde_n_samples
        ),
        "max_particles": int(ctx.args.sde_max_particles),
        "classifier": _main_classifier_settings(ctx),
    }


def _contract_mismatch_paths(
    recorded: object, expected: object, *, prefix: str = ""
) -> list[str]:
    if isinstance(recorded, Mapping) and isinstance(expected, Mapping):
        mismatches: list[str] = []
        for key in sorted(set(recorded).union(expected)):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in recorded or key not in expected:
                mismatches.append(path)
            else:
                mismatches.extend(
                    _contract_mismatch_paths(recorded[key], expected[key], prefix=path)
                )
        return mismatches
    return [] if recorded == expected else [prefix]


def _require_s22_state_settings_match(
    ctx: RunContext,
    manifest: Mapping[str, object],
    manifest_path: str | Path,
) -> None:
    """Reject canonical S22 states generated under different current settings."""

    settings = manifest.get("settings")
    expected = _expected_s22_state_settings(ctx)
    recorded = (
        {key: settings.get(key) for key in expected}
        if isinstance(settings, Mapping)
        else {}
    )
    mismatches = _contract_mismatch_paths(recorded, expected)
    if mismatches:
        raise RuntimeError(
            "S22 canonical states were produced with state-affecting settings "
            "that differ from the current S25 request "
            f"({', '.join(mismatches)}): {Path(manifest_path)}"
        )


def _require_s25_interval_local_manifest_semantics(
    manifest: Mapping[str, object], manifest_path: str | Path
) -> None:
    """Reject external S25 bundles without proven interval-local semantics."""

    source = Path(manifest_path)
    settings = manifest.get("settings")
    details = manifest.get("details")
    if not isinstance(settings, Mapping) or not isinstance(details, Mapping):
        raise RuntimeError(
            "S22 manifest lacks settings/details needed to prove canonical "
            f"trajectory semantics: {source}"
        )
    display_warp = settings.get("display_warp")
    failures = []
    if (
        settings.get("trajectory_mode")
        != "piecewise_observed_anchored_interval_forward_simulation"
    ):
        failures.append("trajectory_mode")
    if settings.get("split_sde_piecewise") is not True:
        failures.append("split_sde_piecewise")
    if settings.get("piecewise_observed_sample_mode") != "per_timepoint":
        failures.append("piecewise_observed_sample_mode")
    if settings.get("piecewise_include_end") is not False:
        failures.append("piecewise_include_end")
    if settings.get("daughter_noise_std") != 0.0:
        failures.append("daughter_noise_std")
    if (
        not isinstance(display_warp, Mapping)
        or display_warp.get("applied") is not False
    ):
        failures.append("display_warp.applied")
    if settings.get("simulation") != S25_COMMUNICATION_TRAJECTORY_SCOPE:
        failures.append("simulation/trajectory_scope")
    if details.get("trajectory_scope") != S25_COMMUNICATION_TRAJECTORY_SCOPE:
        failures.append("details.trajectory_scope")
    if details.get("display_warp_applied") is not False:
        failures.append("details.display_warp_applied")
    if failures:
        raise RuntimeError(
            "S25 source bundle is not proven to use interval-local, one-sided "
            "observed-anchor semantics; refusing incompatible/global-t0 "
            f"states ({', '.join(failures)}): {source}"
        )


def _execute_stage(
    ctx: RunContext,
    stage: str,
    settings: Mapping[str, object],
    action: Callable[[Path], tuple[Sequence[str | Path], Mapping[str, object]]],
) -> dict[str, object]:
    current_acceptance = _context_acceptance_binding(ctx)
    stage_dir = ctx.output_dir / stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    signature = _stable_hash(
        {
            "stage": stage,
            "common": ctx.common_signature,
            "settings": settings,
        }
    )
    manifest_path = _stage_manifest_path(ctx, stage)
    if manifest_path.exists() and not ctx.args.force:
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            prior.get("status") == "complete"
            and prior.get("signature") == signature
            and _recorded_outputs_exist(prior)
        ):
            if prior.get(MATCHED_ACCEPTANCE_KEY) != current_acceptance:
                raise RuntimeError(
                    f"Refusing to resume {stage!r} with a missing or stale "
                    "canonical matched acceptance binding"
                )
            print(f"[resume] {stage}: signature matches; keeping existing outputs")
            return prior
        print(
            f"[resume] {stage}: inputs/settings changed or outputs missing; rerunning"
        )

    started = time.time()
    print(f"[stage] {stage}: start")
    outputs, details = action(stage_dir)
    output_paths = [str(Path(path).expanduser().resolve()) for path in outputs]
    missing = [path for path in output_paths if not Path(path).is_file()]
    if missing:
        raise RuntimeError(f"Stage {stage!r} reported missing outputs: {missing}")
    empty = [path for path in output_paths if Path(path).stat().st_size <= 0]
    if empty:
        raise RuntimeError(f"Stage {stage!r} reported empty outputs: {empty}")
    output_artifacts = [
        {
            "path": path,
            "size_bytes": int(Path(path).stat().st_size),
            "sha256": _sha256(Path(path)),
        }
        for path in output_paths
    ]
    _context_acceptance_binding(ctx)
    manifest = {
        "schema_version": 1,
        "stage": stage,
        "status": "complete",
        "signature": signature,
        "started_at": datetime.fromtimestamp(started, tz=timezone.utc).isoformat(),
        "completed_at": _utc_now(),
        "elapsed_seconds": float(time.time() - started),
        "settings": _json_ready(settings),
        "details": _json_ready(details),
        "outputs": output_paths,
        "output_artifacts": output_artifacts,
        MATCHED_ACCEPTANCE_KEY: _json_ready(current_acceptance),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"[stage] {stage}: done in {manifest['elapsed_seconds']:.1f}s")
    return manifest


def _main_classifier_settings(ctx: RunContext) -> dict[str, object]:
    epochs = 2 if ctx.args.profile == "smoke" else int(ctx.args.classifier_epochs)
    return {
        "contract": "time + aligned spatial(2) + leading latent PCs(10)",
        "n_joint_features": 12,
        "hidden_size": 128,
        "epochs": epochs,
        "learning_rate": 1e-3,
        "test_size": 0.1,
        "best_epoch_metric": "bacc",
        "train_on_full_data": False,
        "seed": int(ctx.args.random_seed),
        "cache_dir": str(ctx.shared_cache_dir / "trajectory_classifier"),
        "cache_tag": MAIN_CLASSIFIER_CACHE_TAG,
    }


def _train_main_classifier(ctx: RunContext):
    settings = _main_classifier_settings(ctx)
    cached, cache_path = cb.tl.train_cached_mlp_classifier_from_adata(
        ctx.adata,
        cache_dir=settings["cache_dir"],
        cache_tag=MAIN_CLASSIFIER_CACHE_TAG,
        label_col=ctx.args.annotation_key,
        time_key=ctx.args.time_key,
        obsm_key=ctx.args.latent_key,
        spatial_key=ctx.args.spatial_key,
        concat_spatial=True,
        hidden_size=int(settings["hidden_size"]),
        epochs=int(settings["epochs"]),
        lr=float(settings["learning_rate"]),
        test_size=float(settings["test_size"]),
        seed=int(settings["seed"]),
        device=ctx.args.device,
        include_time_feature=True,
        n_features=int(settings["n_joint_features"]),
        best_epoch_metric=str(settings["best_epoch_metric"]),
        train_on_full_data=bool(settings["train_on_full_data"]),
    )
    return cached, Path(cache_path)


def _stage_classifier(ctx: RunContext) -> dict[str, object]:
    settings = _main_classifier_settings(ctx)

    def action(stage_dir: Path):
        cached, cache_path = _train_main_classifier(ctx)
        summary_path = stage_dir / "classifier_summary.json"
        summary = {
            "cache_path": str(cache_path),
            "accuracy": cached.accuracy,
            "balanced_accuracy": cached.balanced_accuracy,
            "metadata": cached.metadata,
            "evaluation": cached.evaluation,
        }
        summary_path.write_text(
            json.dumps(_json_ready(summary), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return [cache_path, summary_path], summary

    return _execute_stage(ctx, "classifier", settings, action)


def _stage_velocity(ctx: RunContext) -> dict[str, object]:
    interaction_cfg = ctx.loaded.config["model"]["interaction_net"]
    settings = {
        "time_points": [0.0, 2.0, 4.0],
        "components": {
            "drift": "intrinsic",
            "interaction": "interaction",
            "score": "score",
            "full": "full",
        },
        "projection_modes": {
            "spatial_direct": "component[:, :2], feature_matrix=None",
            "latent_to_spatial": (
                "component[:, 2:] with feature_matrix=joint_features[:, 2:], "
                "projected onto aligned spatial coordinates"
            ),
        },
        "interaction_m": int(ctx.args.interaction_m),
        "interaction_cutoff": float(interaction_cfg["cutoff"]),
        "n_neighbors": int(ctx.args.velocity_neighbors),
    }

    def action(stage_dir: Path):
        import matplotlib.pyplot as plt

        components = cb.tl.compute_velocity_components_from_adata(
            ctx.adata,
            ctx.loaded.model,
            dim=ctx.dim,
            interaction_m=int(ctx.args.interaction_m),
            interaction_threshold=float(interaction_cfg["cutoff"]),
            device=ctx.args.device,
            time_key=ctx.args.time_key,
            obsm_key=ctx.args.latent_key,
            spatial_key=ctx.args.spatial_key,
            concat_spatial=True,
            write_to_adata=False,
            reuse_if_present=False,
        )
        data_path = stage_dir / "velocity_components.npz"
        np.savez_compressed(
            data_path,
            times=components["times"],
            features=components["features"],
            drift=components["drift"],
            interaction=components["interaction"],
            score=components["score"],
            full=components["full"],
        )
        outputs: list[Path] = [data_path]
        plot_fallbacks: list[dict[str, str]] = []
        labels_all = ctx.adata.obs[ctx.args.annotation_key].astype(str).to_numpy()
        for time_value in settings["time_points"]:
            mask = np.isclose(
                components["times"], float(time_value), rtol=0.0, atol=1e-9
            )
            if not np.any(mask):
                raise ValueError(f"No cells at requested velocity time {time_value}")
            coords = components["features"][mask, :2]
            features = components["features"][mask]
            labels = labels_all[mask]
            for name, panel_label in settings["components"].items():
                direct_path = (
                    stage_dir / f"spatial_direct_{panel_label}_t{time_value:g}.pdf"
                )
                direct_plot = cb.pl.plot_velocity_component(
                    coords=coords,
                    velocity=components[name][mask, :2],
                    feature_matrix=None,
                    labels=labels,
                    label_to_color=ctx.label_to_color,
                    title=f"{panel_label} velocity (spatial direct), t={time_value:g}",
                    out_path=str(direct_path),
                    density=2.0,
                    basis="spatial",
                    show_legend=False,
                    n_neighbors=int(ctx.args.velocity_neighbors),
                )
                direct_reason = getattr(direct_plot, "uns", {}).get(
                    "velocity_plot_fallback"
                )
                if direct_reason:
                    plot_fallbacks.append(
                        {"file": str(direct_path), "reason": str(direct_reason)}
                    )
                plt.close("all")
                outputs.append(direct_path)

                latent_path = (
                    stage_dir / f"latent_to_spatial_{panel_label}_t{time_value:g}.pdf"
                )
                latent_plot = cb.pl.plot_velocity_component(
                    coords=coords,
                    velocity=components[name][mask, 2:],
                    feature_matrix=features[:, 2:],
                    labels=labels,
                    label_to_color=ctx.label_to_color,
                    title=(
                        f"{panel_label} velocity (latent to spatial), "
                        f"t={time_value:g}"
                    ),
                    out_path=str(latent_path),
                    density=2.0,
                    basis="spatial",
                    show_legend=False,
                    n_neighbors=int(ctx.args.velocity_neighbors),
                )
                latent_reason = getattr(latent_plot, "uns", {}).get(
                    "velocity_plot_fallback"
                )
                if latent_reason:
                    plot_fallbacks.append(
                        {"file": str(latent_path), "reason": str(latent_reason)}
                    )
                plt.close("all")
                outputs.append(latent_path)
        identity_error = float(
            np.max(
                np.abs(
                    components["full"]
                    - components["drift"]
                    - components["interaction"]
                    - components["score"]
                )
            )
        )
        return outputs, {
            "n_cells": int(ctx.adata.n_obs),
            "full_identity_max_error": identity_error,
            "all_finite": bool(
                all(
                    np.isfinite(components[name]).all()
                    for name in settings["components"]
                )
            ),
            "plot_fallbacks": plot_fallbacks,
        }

    return _execute_stage(ctx, "velocity", settings, action)


def _run_interpolation(
    ctx: RunContext,
    *,
    output_dir: Path,
    time_points: Sequence[float],
    trajectory_mode: str,
    split_growth_alpha: float,
    display_piecewise_warp: bool,
):
    valid_modes = {S22_TRAJECTORY_MODE, "interval_local_observed_anchored"}
    if trajectory_mode not in valid_modes:
        raise ValueError(
            f"trajectory_mode must be exactly one of {sorted(valid_modes)}, "
            f"got {trajectory_mode!r}"
        )
    global_t0 = trajectory_mode == S22_TRAJECTORY_MODE
    if display_piecewise_warp:
        raise ValueError(
            "Canonical zebrafish paper reconstruction does not permit the legacy "
            "endpoint-directed display warp. Consume the canonical unwarped "
            "model states instead."
        )
    interp = [
        float(value)
        for value in time_points
        if not any(
            np.isclose(value, obs, rtol=0.0, atol=1e-9) for obs in OBSERVED_TIMES
        )
    ]
    classifier_settings = _main_classifier_settings(ctx)
    return cb.tl.run_interpolation_workflow(
        df=ctx.df,
        dim=ctx.dim,
        annotation_key=ctx.args.annotation_key,
        runtime=ctx.runtime,
        device=ctx.args.device,
        output_dir=str(output_dir),
        requested_plot_points=[float(value) for value in time_points],
        interp_time_points=interp,
        no_interp=False,
        use_real_for_observed=not global_t0,
        classifier_cache_dir=str(ctx.shared_cache_dir / "trajectory_classifier"),
        classifier_cache_tag=MAIN_CLASSIFIER_CACHE_TAG,
        classifier_adata=ctx.adata,
        classifier_time_key=ctx.args.time_key,
        classifier_obsm_key=ctx.args.latent_key,
        classifier_spatial_key=ctx.args.spatial_key,
        classifier_concat_spatial=True,
        classifier_epochs=int(classifier_settings["epochs"]),
        classifier_hidden_size=int(classifier_settings["hidden_size"]),
        classifier_lr=float(classifier_settings["learning_rate"]),
        classifier_test_size=float(classifier_settings["test_size"]),
        classifier_train_on_full_data=False,
        classifier_best_metric="bacc",
        classifier_n_pcs=12,
        classifier_knn_neighbors=10,
        sde_n_samples=(
            int(ctx.args.smoke_n_samples)
            if ctx.args.profile == "smoke"
            else ctx.args.sde_n_samples
        ),
        skip_nonsplit_sde=True,
        split_sde_dt=float(ctx.args.sde_dt),
        split_sigma_scalar=float(ctx.args.sde_sigma),
        split_daughter_noise_std=0.0,
        split_growth_alpha=float(split_growth_alpha),
        split_interaction_m=int(ctx.args.interaction_m),
        split_resample_dt=float(ctx.args.sde_dt),
        split_max_particles=int(ctx.args.sde_max_particles),
        split_sde_piecewise=not global_t0,
        split_sde_piecewise_include_end=False,
        piecewise_observed_sample_mode=("t0_fixed" if global_t0 else "per_timepoint"),
        spatial_warp_to_observed=False,
        spatial_warp_to_observed_piecewise=False,
        spatial_warp_visualization_only=False,
        spatial_warp_k=8,
        spatial_warp_eps=1e-6,
        random_seed=int(ctx.args.random_seed),
    )


def _require_global_t0_generated_states(
    states: Mapping[str, ad.AnnData],
    time_points: Sequence[float],
    *,
    observed_t0_points: np.ndarray,
) -> None:
    """Fail closed unless every requested frame belongs to one global-t0 path."""

    times = [float(value) for value in time_points]
    if not times or not np.isclose(times[0], 0.0, rtol=0.0, atol=1e-9):
        raise RuntimeError("S22 global-t0 simulation grid must start at t=0.")
    if not np.isclose(times[-1], 4.0, rtol=0.0, atol=1e-9):
        raise RuntimeError("S22 global-t0 simulation grid must end at t=4.")
    if any(right <= left for left, right in zip(times, times[1:])):
        raise RuntimeError("S22 global-t0 simulation grid must be strictly increasing.")

    feature_dim: Optional[int] = None
    particle_count: Optional[int] = None
    for time_value in times:
        key = str(float(time_value))
        if key not in states:
            raise RuntimeError(f"S22 global-t0 result is missing time {key}.")
        state = states[key]
        points = np.asarray(state.X, dtype=np.float32)
        if points.ndim != 2 or points.shape[0] == 0:
            raise RuntimeError(
                f"S22 global-t0 frame t={time_value:g} must be a non-empty matrix."
            )
        if feature_dim is None:
            feature_dim = int(points.shape[1])
            particle_count = int(points.shape[0])
        if points.shape[1] != feature_dim:
            raise RuntimeError(
                "S22 global-t0 frames changed feature dimension: "
                f"time={time_value:g}, expected={feature_dim}, actual={points.shape[1]}."
            )
        if points.shape[0] != particle_count:
            raise RuntimeError(
                "S22 fixed-population transport changed particle count: "
                f"time={time_value:g}, expected={particle_count}, "
                f"actual={points.shape[0]}."
            )
        if not np.isfinite(points).all():
            raise RuntimeError(
                f"S22 global-t0 frame t={time_value:g} contains non-finite values."
            )
        actual_origin = str(state.uns.get("slice_origin", ""))
        if actual_origin != "generated_global_t0":
            raise RuntimeError(
                "S22 trajectory contains an observed-substituted or re-anchored "
                f"frame: time={time_value:g}, expected origin="
                f"'generated_global_t0', actual={actual_origin!r}."
            )
        actual_anchor = state.uns.get("source_anchor_time")
        if actual_anchor is None or not np.isclose(
            float(actual_anchor), 0.0, rtol=0.0, atol=1e-9
        ):
            raise RuntimeError(
                "S22 trajectory was not propagated from the single t=0 anchor: "
                f"time={time_value:g}, source_anchor_time={actual_anchor!r}."
            )

    generated_t0 = np.ascontiguousarray(np.asarray(states["0.0"].X, dtype=np.float32))
    observed_t0 = np.ascontiguousarray(np.asarray(observed_t0_points, dtype=np.float32))
    if observed_t0.ndim != 2 or observed_t0.shape[1] != generated_t0.shape[1]:
        raise RuntimeError(
            "Observed t=0 reference does not match the generated state dimension."
        )
    observed_rows = {row.tobytes() for row in observed_t0}
    if any(row.tobytes() not in observed_rows for row in generated_t0):
        raise RuntimeError(
            "S22 t=0 state is not an exact sample of the real observed t=0 "
            "population."
        )


def _stage_s22(ctx: RunContext) -> dict[str, object]:
    video_step = 1.0 if ctx.args.profile == "smoke" else float(ctx.args.video_step)
    video_times = _time_grid(0.0, 4.0, video_step)
    if ctx.args.profile == "smoke":
        # Exercise the entire global-t0 trajectory, including every integer time.
        video_times = list(HALF_TIMES)
        simulation_times = list(HALF_TIMES)
    else:
        simulation_times = _time_grid(0.0, 4.0, float(ctx.args.s22_simulation_step))
        missing_render_times = [
            value
            for value in [*HALF_TIMES, *video_times]
            if not any(
                np.isclose(value, simulated, rtol=0.0, atol=1e-9)
                for simulated in simulation_times
            )
        ]
        if missing_render_times:
            raise ValueError(
                "S22 mosaic/video times must be a subset of the fixed simulation "
                f"grid; missing={sorted(set(missing_render_times))}. Choose "
                "--video-step as an integer multiple of --s22-simulation-step."
            )
    formats = _parse_csv_strings(ctx.args.video_formats)
    unsupported = sorted(set(formats).difference({"gif", "mp4"}))
    if unsupported:
        raise ValueError(f"Unsupported --video-formats values: {unsupported}")
    settings = {
        "mosaic_times": list(HALF_TIMES),
        "video_times": video_times,
        "trajectory_frames": "generated_at_every_time_including_integer_times",
        "observed_integer_frames": "separate_reference_only",
        "use_real_for_observed_trajectory_frames": False,
        "simulation": S22_TRAJECTORY_SCOPE,
        "trajectory_mode": S22_TRAJECTORY_MODE,
        "population_mode": "fixed_population_state_transport",
        "particle_count_contract": "exactly constant at the sampled t=0 count",
        "retained_dynamics": [
            "learned_velocity_drift",
            "learned_score_gradient",
            "learned_interaction_force_with_uniform_particle_mass",
            "stochastic_diffusion",
        ],
        "disabled_dynamics": [
            "learned_growth_driven_birth_extinction",
            "growth_weight_feedback_into_interaction_mass",
            "cell_abundance_forecasting",
        ],
        "trained_growth_head": {
            "present_in_checkpoint": True,
            "applied_to_s22_transport": False,
            "reported_separately": "S23 observed-state growth maps",
        },
        "scientific_claim": (
            "conditional fixed-population state transport from one t=0 cohort; "
            "not an abundance forecast, adjacent-anchor interpolation, or "
            "reconstruction of observed stages"
        ),
        "split_sde_piecewise": False,
        "piecewise_observed_sample_mode": None,
        "piecewise_include_end": None,
        "simulation_grid": list(simulation_times),
        "simulation_step": (
            None if ctx.args.profile == "smoke" else float(ctx.args.s22_simulation_step)
        ),
        "mosaic_is_subsample_of_single_global_t0_simulation": True,
        "mosaic_layout": {
            "columns": S22_MOSAIC_COLUMNS,
            "show_axes": False,
            "show_legend": True,
        },
        "canonical_state_consumers": [],
        "downstream_state_contract": {
            "s25": "retains separate interval-local hybrid reconstruction",
            "communication": "consumes the S25 hybrid reconstruction",
            "implicit_s22_reuse": False,
        },
        **_expected_s22_state_settings(ctx, growth_alpha=S22_FIXED_GROWTH_ALPHA),
        "display_warp": {
            "applied": False,
            "reason": (
                "disabled so the displayed coordinates remain the direct output "
                "of the single global-t0 fixed-population model transport"
            ),
        },
        "generated_label_knn_neighbors": 10,
        "generated_label_policy": (
            "shared Zebrafish annotation policy for every generated frame"
        ),
        "trajectory_support_audit": {
            "reference": "maximum observed norm in original latent coordinates",
            "maximum_fraction_outside_observed_max": (S24_SUPPORT_MAX_OUTSIDE_FRACTION),
            "maximum_generated_norm_multiplier": S24_SUPPORT_MAX_NORM_MULTIPLIER,
            "publication_blocking": False,
            "reason": (
                "S22 is an explicit global-t0 demonstration; any tail-support "
                "failure is recorded and must be disclosed rather than hidden, "
                "clipped, or silently converted into an observed-anchored path"
            ),
        },
        "video_fps": int(ctx.args.video_fps),
        "video_formats": formats,
        "point_size": float(ctx.args.point_size),
    }

    def action(stage_dir: Path):
        import matplotlib.pyplot as plt

        dense_result = _run_interpolation(
            ctx,
            output_dir=stage_dir / "workflow_shared_dense",
            time_points=simulation_times,
            trajectory_mode=S22_TRAJECTORY_MODE,
            split_growth_alpha=S22_FIXED_GROWTH_ALPHA,
            display_piecewise_warp=False,
        )
        actual_simulation_times = [float(value) for value in dense_result.ts_points]
        if actual_simulation_times != [float(value) for value in simulation_times]:
            raise RuntimeError(
                "S22 workflow returned a different simulation grid: "
                f"expected={simulation_times}, actual={actual_simulation_times}."
            )
        observed_states = _observed_state_dict(ctx)
        _require_global_t0_generated_states(
            dense_result.adata_dict,
            simulation_times,
            observed_t0_points=np.asarray(observed_states[0.0].X, dtype=np.float32),
        )
        support_frames = [
            np.asarray(
                dense_result.adata_dict[str(float(time_value))].X,
                dtype=np.float32,
            )
            for time_value in simulation_times
        ]
        support_audit, support_summary = _compute_trajectory_support_audit(
            np.asarray(ctx.adata.obsm[ctx.args.latent_key], dtype=np.float32),
            {"global_t0_fixed_population": support_frames},
            simulation_times,
            spatial_dim=2,
            max_outside_fraction=S24_SUPPORT_MAX_OUTSIDE_FRACTION,
            max_norm_multiplier=S24_SUPPORT_MAX_NORM_MULTIPLIER,
        )
        support_path = stage_dir / "S22_trajectory_support_audit.csv"
        support_audit.to_csv(support_path, index=False)
        support_summary_path = stage_dir / "S22_trajectory_support_audit.json"
        support_summary_path.write_text(
            json.dumps(_json_ready(support_summary), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        source_by_time = {
            float(value): (
                "sampled_observed_t0_initial_condition"
                if np.isclose(value, 0.0, rtol=0.0, atol=1e-9)
                else "generated_global_t0_fixed_population_state_transport"
            )
            for value in HALF_TIMES
        }
        outputs = _write_state_bundle(
            dense_result.adata_dict,
            HALF_TIMES,
            stage_dir / "mosaic_states",
            annotation_key=ctx.args.annotation_key,
            source_by_time=source_by_time,
        )
        outputs.extend([support_path, support_summary_path])
        canonical_states: dict[str, ad.AnnData] = {}
        for time_value in HALF_TIMES:
            key = str(float(time_value))
            if key not in dense_result.adata_dict:
                raise RuntimeError(f"Canonical S22 result is missing time {key}.")
            state = dense_result.adata_dict[key]
            expected_origin = "generated_global_t0"
            actual_origin = str(state.uns.get("slice_origin", ""))
            if actual_origin != expected_origin:
                raise RuntimeError(
                    "Canonical S22 slice provenance is not global-t0 generated: "
                    f"time={time_value}, expected origin={expected_origin!r}, "
                    f"actual={actual_origin!r}."
                )
            expected_anchor = 0.0
            actual_anchor = state.uns.get("source_anchor_time")
            if actual_anchor is None or not np.isclose(
                float(actual_anchor), expected_anchor, rtol=0.0, atol=1e-9
            ):
                raise RuntimeError(
                    "Canonical S22 slice was not propagated from t=0: "
                    f"time={time_value}, expected={expected_anchor}, "
                    f"actual={actual_anchor}."
                )
            canonical_states[str(float(time_value))] = _minimal_state_adata(
                np.asarray(state.X, dtype=np.float32),
                state.obs[ctx.args.annotation_key].astype(str).to_numpy(),
                annotation_key=ctx.args.annotation_key,
            )
        canonical_source_by_time = dict(source_by_time)
        global_outputs = _write_state_bundle(
            canonical_states,
            HALF_TIMES,
            stage_dir / "global_t0_fixed_population_states",
            annotation_key=ctx.args.annotation_key,
            source_by_time=canonical_source_by_time,
        )
        outputs.extend(global_outputs)
        observed_reference_by_key = {
            str(float(time_value)): observed_states[float(time_value)]
            for time_value in OBSERVED_TIMES
        }
        outputs.extend(
            _write_state_bundle(
                observed_reference_by_key,
                OBSERVED_TIMES,
                stage_dir / "observed_reference_states",
                annotation_key=ctx.args.annotation_key,
                source_by_time={
                    float(value): "observed_reference_only" for value in OBSERVED_TIMES
                },
            )
        )
        source_table = pd.DataFrame(
            {
                "time": list(HALF_TIMES),
                "trajectory_display_source": [
                    source_by_time[float(value)] for value in HALF_TIMES
                ],
                "canonical_state_source": [
                    canonical_source_by_time[float(value)] for value in HALF_TIMES
                ],
                "population_mode": [
                    "fixed_population_state_transport" for _ in HALF_TIMES
                ],
                "growth_alpha": [S22_FIXED_GROWTH_ALPHA for _ in HALF_TIMES],
                "source_anchor_time": [0.0 for _ in HALF_TIMES],
                "observed_reference_available": [
                    bool(float(value) in OBSERVED_TIMES) for value in HALF_TIMES
                ],
                "observed_reference_source": [
                    (
                        "observed_reference_only"
                        if float(value) in OBSERVED_TIMES
                        else None
                    )
                    for value in HALF_TIMES
                ],
                "s25_analysis_source": [
                    "separate_interval_local_hybrid_not_implicit_s22_reuse"
                    for _ in HALF_TIMES
                ],
                "communication_source": [
                    "separate_s25_hybrid_not_implicit_s22_reuse" for _ in HALF_TIMES
                ],
            }
        )
        source_path = stage_dir / "frame_sources.csv"
        source_table.to_csv(source_path, index=False)
        outputs.append(source_path)

        snapshot_dir = stage_dir / "mosaic_snapshots"
        cb.tl.save_timepoint_snapshots(
            adata_dict=dense_result.adata_dict,
            time_keys=[str(float(value)) for value in HALF_TIMES],
            annotation_key=ctx.args.annotation_key,
            label_to_color=ctx.label_to_color,
            snapshot_dir=str(snapshot_dir),
            background_color="white",
            font_color="black",
            snapshot_point_size=float(ctx.args.point_size),
            snapshot_alpha=0.9,
            mosaic_cols=S22_MOSAIC_COLUMNS,
            mosaic_cell_size=3.0,
            mosaic_show_title=True,
            save_pdf=True,
        )
        outputs.extend(sorted(snapshot_dir.glob("*")))

        mosaic_points = np.empty(len(HALF_TIMES), dtype=object)
        mosaic_labels: list[np.ndarray] = []
        for index, time_value in enumerate(HALF_TIMES):
            state = dense_result.adata_dict[str(float(time_value))]
            mosaic_points[index] = np.asarray(state.X, dtype=np.float32)
            mosaic_labels.append(
                state.obs[ctx.args.annotation_key].astype(str).to_numpy()
            )
        mosaic_pdf = (
            stage_dir / "S22_global_t0_fixed_population_state_transport_mosaic.pdf"
        )
        fig = cb.pl.plot_trajectory_grid(
            sde_points=mosaic_points,
            time_values=HALF_TIMES,
            dim_pairs=((0, 1),),
            labels_list=mosaic_labels,
            label_to_color=ctx.label_to_color,
            out_path=str(mosaic_pdf),
            figsize_per_panel=(2.6, 2.6),
            point_size=float(ctx.args.point_size),
            alpha=0.9,
            title="Global-t0 fixed-population state transport (growth disabled)",
            n_cols=S22_MOSAIC_COLUMNS,
            show_axes=False,
            show_legend=True,
            equal_aspect=True,
            legend_title="Cell type",
            legend_fontsize=6.0,
        )
        mosaic_png = (
            stage_dir / "S22_global_t0_fixed_population_state_transport_mosaic.png"
        )
        fig.savefig(mosaic_png, dpi=240, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        outputs.extend([mosaic_pdf, mosaic_png])

        reference_points = np.empty(len(OBSERVED_TIMES), dtype=object)
        reference_labels: list[np.ndarray] = []
        for index, time_value in enumerate(OBSERVED_TIMES):
            state = observed_states[float(time_value)]
            reference_points[index] = np.asarray(state.X, dtype=np.float32)
            reference_labels.append(
                state.obs[ctx.args.annotation_key].astype(str).to_numpy()
            )
        reference_pdf = stage_dir / "S22_observed_reference_mosaic.pdf"
        reference_fig = cb.pl.plot_trajectory_grid(
            sde_points=reference_points,
            time_values=OBSERVED_TIMES,
            dim_pairs=((0, 1),),
            labels_list=reference_labels,
            label_to_color=ctx.label_to_color,
            out_path=str(reference_pdf),
            figsize_per_panel=(2.6, 2.6),
            point_size=float(ctx.args.point_size),
            alpha=0.9,
            title="Observed zebrafish reference slices",
            n_cols=S22_MOSAIC_COLUMNS,
            show_axes=False,
            show_legend=True,
            equal_aspect=True,
            legend_title="Cell type",
            legend_fontsize=6.0,
        )
        reference_png = stage_dir / "S22_observed_reference_mosaic.png"
        reference_fig.savefig(
            reference_png, dpi=240, bbox_inches="tight", facecolor="white"
        )
        plt.close(reference_fig)
        outputs.extend([reference_pdf, reference_png])

        fixed_particle_count = int(
            dense_result.adata_dict[str(float(simulation_times[0]))].n_obs
        )
        panel_caption = (
            "One observed t=0 cohort is propagated continuously to t=4 with "
            "learned velocity drift, score-gradient correction, interaction "
            f"forces, and stochastic diffusion (fixed N={fixed_particle_count}). "
            "Learned growth-driven birth/extinction is disabled. Every post-t=0 "
            "frame is generated from the same initialization without real-slice "
            "re-anchoring; observed stages are shown only in the separate reference "
            "panel. This is fixed-population model state transport, not a cell-"
            "abundance forecast, adjacent-anchor interpolation, or reconstruction "
            "of observed stages."
        )
        caption_path = stage_dir / "S22_panel_caption.json"
        caption_path.write_text(
            json.dumps({"S22": panel_caption}, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        outputs.append(caption_path)

        dense_points = np.empty(len(video_times), dtype=object)
        dense_labels: list[np.ndarray] = []
        for index, time_value in enumerate(video_times):
            state = dense_result.adata_dict[str(float(time_value))]
            dense_points[index] = np.asarray(state.X, dtype=np.float32)
            dense_labels.append(
                state.obs[ctx.args.annotation_key].astype(str).to_numpy()
            )
        animation_errors: dict[str, str] = {}
        for extension in formats:
            animation_path = (
                stage_dir
                / f"S22_global_t0_fixed_population_state_transport_dense.{extension}"
            )
            if extension == "mp4" and shutil.which("ffmpeg") is None:
                animation_errors[extension] = "ffmpeg is not installed"
                continue
            try:
                cb.pl.plot_trajectory_gif(
                    sde_points=dense_points,
                    time_values=video_times,
                    labels_list=dense_labels,
                    label_to_color=ctx.label_to_color,
                    out_path=str(animation_path),
                    dim_pair=(0, 1),
                    point_size=max(1.0, float(ctx.args.point_size)),
                    alpha=0.9,
                    fps=int(ctx.args.video_fps),
                )
                outputs.append(animation_path)
            except Exception as exc:
                if extension != "mp4":
                    raise
                animation_errors[extension] = f"{type(exc).__name__}: {exc}"

        return outputs, {
            "classifier_cache_path": dense_result.classifier_cache_path,
            "classifier_accuracy": dense_result.classifier_accuracy,
            "classifier_balanced_accuracy": dense_result.classifier_balanced_accuracy,
            "mosaic_cell_counts": {
                str(float(t)): int(dense_result.adata_dict[str(float(t))].n_obs)
                for t in HALF_TIMES
            },
            "video_frame_count": int(len(video_times)),
            "simulation_frame_count": int(len(simulation_times)),
            "single_global_t0_simulation_for_mosaic_and_video": True,
            "population_mode": "fixed_population_state_transport",
            "fixed_particle_count": fixed_particle_count,
            "particle_count_constant_across_all_frames": True,
            "growth_alpha": S22_FIXED_GROWTH_ALPHA,
            "growth_head_applied_to_transport": False,
            "retained_dynamics": settings["retained_dynamics"],
            "disabled_dynamics": settings["disabled_dynamics"],
            "scientific_claim": settings["scientific_claim"],
            "panel_caption": panel_caption,
            "observed_integer_frames_substituted_into_trajectory": False,
            "observed_reference_times": list(OBSERVED_TIMES),
            "animation_errors": animation_errors,
            "display_warp_applied": False,
            "trajectory_scope": S22_TRAJECTORY_SCOPE,
            "trajectory_support_audit": support_summary,
            "trajectory_support_audit_publication_blocking": False,
            "global_t0_fixed_population_state_index": str(
                (
                    stage_dir / "global_t0_fixed_population_states" / "index.json"
                ).resolve()
            ),
            "global_t0_fixed_population_state_index_sha256": _sha256(
                stage_dir / "global_t0_fixed_population_states" / "index.json"
            ),
            "observed_reference_state_index": str(
                (stage_dir / "observed_reference_states" / "index.json").resolve()
            ),
            "observed_reference_state_index_sha256": _sha256(
                stage_dir / "observed_reference_states" / "index.json"
            ),
            "simulation_seeds": dense_result.simulation_seeds,
        }

    return _execute_stage(ctx, "s22", settings, action)


def _stage_growth(ctx: RunContext) -> dict[str, object]:
    settings = {
        "observed_times": list(OBSERVED_TIMES),
        "normalization": "independent per-time 5th-95th percentile scaling",
        "raw_growth_exported": True,
        "composite_layout": {
            "columns": 2,
            "shared_display_scale": [0.0, 1.0],
        },
        "point_size": float(ctx.args.point_size),
    }

    def action(stage_dir: Path):
        observed_float_keys = _observed_state_dict(ctx)
        observed_states = {
            str(float(time_value)): state
            for time_value, state in observed_float_keys.items()
        }
        raw = cb.tl.evaluate_growth_by_timepoint(
            observed_states,
            model=ctx.loaded.model,
            time_points=OBSERVED_TIMES,
            annotation_key=ctx.args.annotation_key,
            spatial_key="spatial",
            value_key="growth_rate",
            device=ctx.args.device,
        )
        raw_path = stage_dir / "growth_per_cell.csv"
        raw.to_csv(raw_path, index=False)
        outputs: list[Path] = [raw_path]

        common_plot_options = {
            "adata_dict": observed_states,
            "spatial_key": "spatial",
            "value_key": "growth_rate",
            "cmap": "RdYlBu_r",
            "point_size": float(ctx.args.point_size),
            "lower_quantile": 0.05,
            "upper_quantile": 0.95,
            "scale_mode": "per_time_0_1",
            "colorbar_label": "g (scaled 5-95%)",
        }
        for time_index, time_value in enumerate(OBSERVED_TIMES):
            path = stage_dir / f"growth_t{time_index}.pdf"
            cb.pl.plot_growth_timepoint_grid(
                **common_plot_options,
                time_points=[time_value],
                time_keys=[str(float(time_value))],
                out_path=str(path),
                n_cols=1,
                shared_colorbar=False,
            )
            outputs.append(path)

        for extension in ("pdf", "png"):
            path = stage_dir / f"S23_growth_observed_grid.{extension}"
            cb.pl.plot_growth_timepoint_grid(
                **common_plot_options,
                time_points=OBSERVED_TIMES,
                out_path=str(path),
                n_cols=2,
                shared_colorbar=True,
                title="Growth-rate maps across observed zebrafish stages",
            )
            outputs.append(path)
        return outputs, {
            "n_panels": len(OBSERVED_TIMES),
            "n_growth_values": int(len(raw)),
            "composite_outputs": [
                str(stage_dir / "S23_growth_observed_grid.pdf"),
                str(stage_dir / "S23_growth_observed_grid.png"),
            ],
        }

    return _execute_stage(ctx, "growth", settings, action)


def _ablation_classifier(ctx: RunContext, stage_dir: Path):
    from sklearn.decomposition import PCA

    latent = np.asarray(ctx.adata.obsm[ctx.args.latent_key], dtype=np.float32)
    if latent.shape[1] < 10:
        raise ValueError(
            f"Ablation classifier requires at least 10 latent PCs, got {latent.shape[1]}"
        )
    pca = PCA(n_components=10, random_state=int(ctx.args.random_seed))
    latent_pca10 = pca.fit_transform(latent).astype(np.float32)
    classifier_features = np.hstack(
        (
            np.asarray(ctx.adata.obsm[ctx.args.spatial_key], dtype=np.float32),
            latent_pca10,
        )
    ).astype(np.float32)
    classifier_adata = ad.AnnData(
        X=np.zeros((ctx.adata.n_obs, 0), dtype=np.float32),
        obs=ctx.adata.obs[[ctx.args.annotation_key, ctx.args.time_key]].copy(),
    )
    classifier_adata.obsm["X_ablation_classifier"] = classifier_features
    epochs = (
        2 if ctx.args.profile == "smoke" else int(ctx.args.ablation_classifier_epochs)
    )
    cached, cache_path = cb.tl.train_cached_mlp_classifier_from_adata(
        classifier_adata,
        cache_dir=ctx.shared_cache_dir / "ablation_classifier",
        cache_tag="zebrafish-paper-ablation-spatial2-pca10",
        label_col=ctx.args.annotation_key,
        time_key=ctx.args.time_key,
        obsm_key="X_ablation_classifier",
        concat_spatial=False,
        hidden_size=128,
        epochs=epochs,
        lr=1e-3,
        test_size=0.1,
        seed=int(ctx.args.random_seed),
        device=ctx.args.device,
        include_time_feature=True,
        n_features=12,
        best_epoch_metric="bacc",
        train_on_full_data=True,
    )
    pca_path = stage_dir / "ablation_classifier_pca10.npz"
    np.savez_compressed(
        pca_path,
        components=np.asarray(pca.components_, dtype=np.float32),
        mean=np.asarray(pca.mean_, dtype=np.float32),
        explained_variance=np.asarray(pca.explained_variance_, dtype=np.float32),
        explained_variance_ratio=np.asarray(
            pca.explained_variance_ratio_, dtype=np.float32
        ),
        singular_values=np.asarray(pca.singular_values_, dtype=np.float32),
        n_samples_seen=np.asarray([latent.shape[0]], dtype=np.int64),
    )

    def labeler(points, time_points):
        transformed = np.empty(len(points), dtype=object)
        for index, frame in enumerate(points):
            frame = np.asarray(frame, dtype=np.float32)
            transformed[index] = np.hstack(
                (frame[:, :2], pca.transform(frame[:, 2:]))
            ).astype(np.float32)
        return cb.tl.predict_labels_for_trajectories(
            sde_points=transformed,
            ts_points=time_points,
            model=cached.model,
            label_encoder=cached.label_encoder,
            feature_dim=12,
            device=ctx.args.device,
            knn_neighbors=10,
            include_time_feature=True,
        )

    return cached, Path(cache_path), pca_path, labeler


def _compute_trajectory_support_audit(
    observed_latent: np.ndarray,
    trajectories: Mapping[str, Sequence[np.ndarray]],
    time_points: Sequence[float],
    *,
    spatial_dim: int,
    max_outside_fraction: float = S24_SUPPORT_MAX_OUTSIDE_FRACTION,
    max_norm_multiplier: float = S24_SUPPORT_MAX_NORM_MULTIPLIER,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Pure latent-support audit for generated joint-space trajectories.

    The observed reference defines one transparent radial support boundary: the
    maximum Euclidean norm in the original latent coordinates.  Every generated
    frame is summarized without clipping or dropping outliers.  This helper only
    computes and classifies the audit; callers decide whether a failed audit is
    diagnostic or publication-blocking.
    """

    observed = np.asarray(observed_latent, dtype=np.float64)
    if observed.ndim != 2 or observed.shape[0] == 0 or observed.shape[1] == 0:
        raise ValueError(
            "observed_latent must be a non-empty two-dimensional matrix, "
            f"got {observed.shape}."
        )
    times = tuple(float(value) for value in time_points)
    if not times:
        raise ValueError("time_points must be non-empty for a support audit.")
    spatial_dim = int(spatial_dim)
    if spatial_dim < 0:
        raise ValueError("spatial_dim must be >= 0.")
    max_outside_fraction = float(max_outside_fraction)
    max_norm_multiplier = float(max_norm_multiplier)
    if not np.isfinite(max_outside_fraction) or not 0.0 <= max_outside_fraction <= 1.0:
        raise ValueError("max_outside_fraction must be finite and in [0, 1].")
    if not np.isfinite(max_norm_multiplier) or max_norm_multiplier <= 0.0:
        raise ValueError("max_norm_multiplier must be finite and > 0.")

    observed_finite_rows = np.isfinite(observed).all(axis=1)
    observed_is_finite = bool(observed_finite_rows.all())
    observed_norms = np.linalg.norm(observed[observed_finite_rows], axis=1)
    observed_norm_max = float(np.max(observed_norms)) if observed_norms.size else None
    observed_failure = None
    if not observed_is_finite:
        observed_failure = "observed_latent_contains_nonfinite_values"
    elif observed_norm_max is None:
        observed_failure = "observed_latent_has_no_finite_rows"

    rows: list[dict[str, object]] = []
    violations: list[dict[str, object]] = []
    for condition, raw_frames in trajectories.items():
        frames = tuple(raw_frames)
        if len(frames) != len(times):
            raise ValueError(
                f"Trajectory {condition!r} has {len(frames)} frames for "
                f"{len(times)} requested times."
            )
        for frame_index, (time_value, raw_frame) in enumerate(zip(times, frames)):
            frame = np.asarray(raw_frame, dtype=np.float64)
            expected_dim = spatial_dim + int(observed.shape[1])
            if frame.ndim != 2 or frame.shape[1] != expected_dim:
                raise ValueError(
                    f"Trajectory {condition!r} frame {frame_index} must have "
                    f"shape N x {expected_dim}, got {frame.shape}."
                )
            latent = frame[:, spatial_dim:]
            finite_rows = np.isfinite(frame).all(axis=1)
            n_points = int(latent.shape[0])
            n_finite = int(np.count_nonzero(finite_rows))
            n_nonfinite = int(n_points - n_finite)
            n_nonfinite_values = int(np.count_nonzero(~np.isfinite(frame)))
            finite_norms = np.linalg.norm(latent[finite_rows], axis=1)
            p99_norm = (
                float(np.quantile(finite_norms, 0.99)) if finite_norms.size else None
            )
            max_norm = float(np.max(finite_norms)) if finite_norms.size else None
            if observed_norm_max is not None and n_points > 0:
                outside_count = int(np.count_nonzero(finite_norms > observed_norm_max))
                outside_fraction = float(outside_count / n_points)
            else:
                outside_count = 0
                outside_fraction = None

            reasons: list[str] = []
            if n_points == 0:
                reasons.append("empty_frame")
            if n_nonfinite > 0:
                reasons.append("nonfinite_generated_values")
            if outside_fraction is not None and outside_fraction > max_outside_fraction:
                reasons.append("outside_observed_max_fraction")
            if (
                observed_norm_max is not None
                and max_norm is not None
                and max_norm > max_norm_multiplier * observed_norm_max
            ):
                reasons.append("maximum_norm_multiplier")
            if observed_failure is not None:
                reasons.append(observed_failure)
            passed = not reasons
            record = {
                "condition": str(condition),
                "frame_index": int(frame_index),
                "time": float(time_value),
                "n_points": n_points,
                "n_finite_points": n_finite,
                "n_nonfinite_points": n_nonfinite,
                "n_nonfinite_values": n_nonfinite_values,
                "latent_norm_p99": p99_norm,
                "latent_norm_max": max_norm,
                "observed_latent_norm_max": observed_norm_max,
                "n_outside_observed_max": outside_count,
                "fraction_outside_observed_max": outside_fraction,
                "max_outside_fraction_threshold": max_outside_fraction,
                "max_norm_multiplier_threshold": max_norm_multiplier,
                "passes_publication_support_gate": passed,
                "failure_reasons": ";".join(reasons),
            }
            rows.append(record)
            if not passed:
                violations.append(
                    {
                        "condition": str(condition),
                        "frame_index": int(frame_index),
                        "time": float(time_value),
                        "failure_reasons": list(reasons),
                        "fraction_outside_observed_max": outside_fraction,
                        "latent_norm_max": max_norm,
                    }
                )

    audit = pd.DataFrame(rows)
    summary: dict[str, object] = {
        "schema_version": 1,
        "status": "PASS" if not violations and observed_failure is None else "FAIL",
        "semantics": (
            "latent radial-support publication guard; reports all generated "
            "points without clipping or outlier removal"
        ),
        "observed_reference": {
            "n_points": int(observed.shape[0]),
            "latent_dim": int(observed.shape[1]),
            "all_finite": observed_is_finite,
            "latent_norm_max": observed_norm_max,
        },
        "thresholds": {
            "nonfinite_generated_values_allowed": False,
            "maximum_fraction_outside_observed_latent_norm_max": max_outside_fraction,
            "maximum_generated_latent_norm_multiplier": max_norm_multiplier,
            "comparison_operators": {
                "fraction": ">",
                "maximum_norm": ">",
            },
        },
        "n_frames": int(len(rows)),
        "n_failed_frames": int(len(violations)),
        "failed_frames": violations,
    }
    return audit, summary


def _require_trajectory_support_audit_pass(
    summary: Mapping[str, object], *, stage: str
) -> None:
    """Fail a publication stage when its precomputed support audit failed."""

    if summary.get("status") == "PASS":
        return
    failures = summary.get("failed_frames", [])
    examples = []
    if isinstance(failures, Sequence):
        for item in failures[:3]:
            if isinstance(item, Mapping):
                examples.append(
                    f"{item.get('condition')}@t={item.get('time')}:"
                    f"{','.join(str(value) for value in item.get('failure_reasons', []))}"
                )
    suffix = f" Examples: {'; '.join(examples)}." if examples else ""
    raise RuntimeError(
        f"{stage} failed the publication latent-support gate; no publication "
        f"panel was emitted.{suffix}"
    )


def _require_s24_preterminal_t3_sigma0_result(
    result,
    *,
    variant: str,
    time_points: Sequence[float],
    random_seed: int,
    interaction_seed: int,
    interaction_m: int,
) -> int:
    """Validate the deterministic preterminal spatial-sensitivity contract."""

    matched_n = int(len(result.initial_obs_names))
    if matched_n <= 0:
        raise RuntimeError(f"S24 {variant} returned an empty matched cohort.")
    expected_times = tuple(float(value) for value in time_points)
    actual_times = tuple(float(value) for value in result.time_points)
    if len(actual_times) != len(expected_times) or not np.allclose(
        actual_times,
        expected_times,
        rtol=0.0,
        atol=1e-10,
    ):
        raise RuntimeError(
            f"S24 {variant} returned the wrong output-time grid: "
            f"expected={expected_times}, actual={actual_times}."
        )
    if (
        not actual_times
        or not np.isclose(actual_times[0], 0.0, rtol=0.0, atol=1e-10)
        or not np.isclose(actual_times[-1], S24_END_TIME, rtol=0.0, atol=1e-10)
    ):
        raise RuntimeError(
            f"S24 {variant} must run globally from t=0 through preterminal "
            f"t={S24_END_TIME:g}, not from a re-anchor or to a terminal endpoint."
        )
    settings = result.settings
    variant_counts = settings.get("variant_initial_counts", {})
    variant_n = int(variant_counts.get(variant, -1))
    if variant_n != matched_n:
        raise RuntimeError(
            f"S24 {variant} is not equal-N at initialization: "
            f"baseline={matched_n}, variant={variant_n}."
        )
    if settings.get("mass_control") is not True:
        raise RuntimeError(f"S24 {variant} did not use mass_control=True.")
    if not np.isclose(
        float(settings.get("dt", np.nan)),
        S24_FIXED_DT,
        rtol=0.0,
        atol=0.0,
    ) or not np.isclose(
        float(settings.get("resample_dt", np.nan)),
        S24_FIXED_DT,
        rtol=0.0,
        atol=0.0,
    ):
        raise RuntimeError(
            f"S24 {variant} did not use dt=resample_dt={S24_FIXED_DT:g}."
        )
    if not np.isclose(
        float(settings.get("sigma", np.nan)),
        S24_FIXED_SIGMA,
        rtol=0.0,
        atol=0.0,
    ):
        raise RuntimeError(f"S24 {variant} did not use sigma=0.")
    sigma_by_dim = settings.get("sigma_by_dim")
    if sigma_by_dim is not None and not np.all(
        np.asarray(sigma_by_dim, dtype=np.float64) == S24_FIXED_SIGMA
    ):
        raise RuntimeError(f"S24 {variant} used nonzero dimension-wise diffusion.")
    if not np.isclose(
        float(settings.get("growth_alpha", np.nan)),
        S24_FIXED_GROWTH_ALPHA,
        rtol=0.0,
        atol=0.0,
    ):
        raise RuntimeError(f"S24 {variant} did not disable learned growth.")
    if int(settings.get("interaction_m", -1)) != int(interaction_m):
        raise RuntimeError(f"S24 {variant} used the wrong interaction_m.")
    if int(settings.get("interaction_seed", -1)) != int(interaction_seed):
        raise RuntimeError(f"S24 {variant} used the wrong interaction seed.")
    seeds = settings.get("simulation_seeds", {})
    if int(seeds.get("baseline", -1)) != int(random_seed) or int(
        seeds.get(variant, -1)
    ) != int(random_seed):
        raise RuntimeError(
            f"S24 {variant} did not use the same deterministic replay seed."
        )
    if int(interaction_m) <= matched_n:
        raise RuntimeError(
            f"S24 {variant} requires one full interaction context group: "
            f"interaction_m={interaction_m}, matched_n={matched_n}."
        )

    baseline_counts = [
        int(np.asarray(frame).shape[0]) for frame in result.baseline_points
    ]
    ablation_counts = [
        int(np.asarray(frame).shape[0]) for frame in result.ablation_points[variant]
    ]
    if len(baseline_counts) != len(expected_times) or len(ablation_counts) != len(
        expected_times
    ):
        raise RuntimeError(
            f"S24 {variant} did not return one fixed-population frame per "
            "requested preterminal output time."
        )
    if any(count != matched_n for count in baseline_counts + ablation_counts):
        raise RuntimeError(
            f"S24 {variant} changed particle count despite fixed-population "
            f"settings: baseline={baseline_counts}, variant={ablation_counts}."
        )
    return matched_n


def _stage_ablation(ctx: RunContext) -> dict[str, object]:
    step = 1.0 if ctx.args.profile == "smoke" else float(ctx.args.ablation_step)
    time_points = _time_grid(0.0, S24_END_TIME, step)
    interaction_seed = int(ctx.args.random_seed) + S24_INTERACTION_SEED_OFFSET
    experiments = (
        {
            "target": "YSL",
            "variant": "remove_YSL",
            "label": str(ctx.args.ysl_label),
            "formal_matched_n": 534,
        },
        {
            "target": "EVL",
            "variant": "remove_EVL",
            "label": str(ctx.args.evl_label),
            "formal_matched_n": 291,
        },
    )
    settings = {
        "time_points": time_points,
        "output_step": step,
        "publication_snapshot_times": list(S24_PUBLICATION_TIMES),
        "simulation": (
            "two target-specific global-t0 deterministic matched equal-N "
            "fixed-population spatial sensitivities; each branch is initialized "
            "once at t=0 and propagated continuously through preterminal t=3 with "
            "no re-anchoring or spatial warp"
        ),
        "canonical_reconstruction": False,
        "publication_protocol": S24_PROTOCOL,
        "counterfactual_scope": (
            "global_t0_preterminal_t3_deterministic_spatial_sensitivity"
        ),
        "interpretation": (
            "conditional spatial model sensitivity to a target-excluded initial "
            "cohort through observed t=3; not total-mass deletion, a causal knockout, "
            "a stochastic forecast, terminal t=4 evidence, full joint-state terminal "
            "evidence, lineage evidence, or an uncertainty estimate"
        ),
        "dt": S24_FIXED_DT,
        "split_resample_dt": S24_FIXED_DT,
        "sigma": S24_FIXED_SIGMA,
        "s24_fixed_numerics": {
            "source": "hard-coded publication protocol; CLI SDE values do not apply",
            "cli_sde_dt": float(ctx.args.sde_dt),
            "cli_sde_sigma": float(ctx.args.sde_sigma),
        },
        "growth_alpha": S24_FIXED_GROWTH_ALPHA,
        "dynamics_retained": [
            "learned_velocity_drift",
            "score_gradient_correction",
            "learned_interaction",
        ],
        "dynamics_disabled": [
            "stochastic_diffusion",
            "growth_driven_birth_extinction",
        ],
        "growth_resampling": (
            "disabled for publication analysis; particle count must remain fixed"
        ),
        "mass_control": True,
        "cohort_matching": (
            "independent no-replacement baseline and target-excluded draws at the "
            "same initial particle count"
        ),
        "interaction_m": S24_INTERACTION_M,
        "interaction_seed": interaction_seed,
        "interaction_context": (
            "one full context group per formal matched cohort because interaction_m "
            "exceeds both formal matched particle counts"
        ),
        "n_samples": (
            int(ctx.args.smoke_n_samples) if ctx.args.profile == "smoke" else None
        ),
        "snapshot_point_size": float(ctx.args.point_size),
        "composite_layout": {
            "rows": "observed times",
            "separate_panels": ["YSL", "EVL"],
            "columns_within_each_panel": ["matched_baseline", "target_excluded"],
            "shared_axis_limits_within_target": True,
            "cross_target_axis_sharing": False,
            "outlier_clipping_or_removal": False,
            "semantics": (
                "preterminal t=3, sigma=0 matched equal-N fixed-population spatial "
                "model sensitivity; not terminal or causal evidence"
            ),
        },
        "max_particles": int(ctx.args.sde_max_particles),
        "experiments": {
            str(spec["target"]): {
                "variant": str(spec["variant"]),
                "excluded_label": str(spec["label"]),
                "ablations_per_call": 1,
                "formal_matched_particle_count": int(spec["formal_matched_n"]),
            }
            for spec in experiments
        },
        "trajectory_support_audit": {
            "reference": "maximum observed norm in original latent coordinates",
            "nonfinite_allowed": False,
            "maximum_fraction_outside_observed_max": (S24_SUPPORT_MAX_OUTSIDE_FRACTION),
            "maximum_generated_norm_multiplier": S24_SUPPORT_MAX_NORM_MULTIPLIER,
            "required_conditions": [
                "YSL_matched_baseline",
                "YSL_remove_YSL",
                "EVL_matched_baseline",
                "EVL_remove_EVL",
            ],
            "publication_stage_fails_on_violation": True,
        },
        "classifier": {
            "contract": "time + spatial2 + fresh PCA10(original latent50)",
            "hidden_size": 128,
            "epochs": (
                2
                if ctx.args.profile == "smoke"
                else int(ctx.args.ablation_classifier_epochs)
            ),
            "best_epoch_metric": "bacc",
            "train_on_full_data": True,
            "knn_neighbors": 10,
        },
        "random_stream_coupling": (
            "Brownian diffusion is disabled at sigma=0. The same branch-level seed "
            "is retained for deterministic replay of independently sampled equal-N "
            "cohorts, and the explicit interaction-grouping seed is shared across "
            "baseline and target-excluded branches; single-seed conditional sensitivity"
        ),
        "terminal_t4_scope": {
            "included": False,
            "evaluated": False,
            "claimed": False,
            "reason": (
                "the preterminal protocol is defined through observed t=3; "
                "t=4 is not evaluated or claimed"
            ),
        },
        "superseded_legacy_result": {
            "description": (
                "combined unequal-N YSL/EVL virtual-removal run with learned "
                "growth/resampling enabled"
            ),
            "status": "superseded_diagnostic_only",
            "reused": False,
            "publication_eligible": False,
            "reason": (
                "particle-count, Monte Carlo context, and learned-growth amplification "
                "were confounded with target exclusion; its EVL branch left model "
                "support"
            ),
        },
    }

    def action(stage_dir: Path):
        import matplotlib as mpl
        import matplotlib.pyplot as plt

        if ctx.args.profile == "full":
            t0_mask = np.isclose(
                ctx.df["samples"].to_numpy(dtype=float),
                0.0,
                rtol=0.0,
                atol=1e-9,
            )
            t0_labels = ctx.df.loc[t0_mask, ctx.args.annotation_key].astype(str)
            t0_counts = t0_labels.value_counts().to_dict()
            expected = {
                "total": 563,
                str(ctx.args.ysl_label): 29,
                str(ctx.args.evl_label): 272,
            }
            actual = {
                "total": int(t0_mask.sum()),
                str(ctx.args.ysl_label): int(t0_counts.get(str(ctx.args.ysl_label), 0)),
                str(ctx.args.evl_label): int(t0_counts.get(str(ctx.args.evl_label), 0)),
            }
            if actual != expected:
                raise ValueError(
                    "The clean-counts t0 cohort does not match the frozen S24 "
                    f"contract: expected {expected}, got {actual}."
                )
        cached, cache_path, pca_path, labeler = _ablation_classifier(ctx, stage_dir)
        outputs: list[Path] = [cache_path, pca_path]
        results: dict[str, object] = {}
        matched_counts: dict[str, int] = {}
        publication_metrics: dict[str, str] = {}
        support_trajectories: dict[str, Sequence[np.ndarray]] = {}

        for spec in experiments:
            target = str(spec["target"])
            variant = str(spec["variant"])
            result = cb.tl.run_virtual_cell_type_ablation(
                ctx.adata,
                ctx.runtime,
                ablations={variant: [str(spec["label"])]},
                time_points=time_points,
                output_dir=stage_dir / f"experiment_{target}_{S24_PROTOCOL}",
                time_index=0,
                n_samples=(
                    int(ctx.args.smoke_n_samples)
                    if ctx.args.profile == "smoke"
                    else None
                ),
                dt=S24_FIXED_DT,
                resample_dt=S24_FIXED_DT,
                sigma=S24_FIXED_SIGMA,
                growth_alpha=S24_FIXED_GROWTH_ALPHA,
                interaction_m=S24_INTERACTION_M,
                interaction_seed=interaction_seed,
                max_particles=int(ctx.args.sde_max_particles),
                device=ctx.args.device,
                time_key=ctx.args.time_key,
                annotation_key=ctx.args.annotation_key,
                obsm_key=ctx.args.latent_key,
                spatial_key=ctx.args.spatial_key,
                concat_spatial=True,
                spatial_dim=2,
                random_seed=int(ctx.args.random_seed),
                common_random_seed=True,
                mass_control=True,
                trajectory_labeler=labeler,
                save_data=True,
                save_snapshots=False,
                label_to_color=ctx.label_to_color,
                verbose=True,
            )
            matched_n = _require_s24_preterminal_t3_sigma0_result(
                result,
                variant=variant,
                time_points=time_points,
                random_seed=int(ctx.args.random_seed),
                interaction_seed=interaction_seed,
                interaction_m=S24_INTERACTION_M,
            )
            if ctx.args.profile == "full" and matched_n != int(
                spec["formal_matched_n"]
            ):
                raise RuntimeError(
                    f"Unexpected formal S24 {target} matched cohort size: "
                    f"expected {spec['formal_matched_n']}, got {matched_n}."
                )
            results[target] = result
            matched_counts[target] = matched_n
            outputs.extend(Path(path) for path in result.files)

            if "space" not in result.metrics.columns:
                raise RuntimeError(
                    f"S24 {target} metrics do not declare a feature-space scope."
                )
            spatial_metrics = result.metrics.loc[
                result.metrics["space"].astype(str).eq("spatial")
            ].copy()
            if spatial_metrics.empty or spatial_metrics["time"].astype(float).max() > (
                S24_END_TIME + 1e-10
            ):
                raise RuntimeError(
                    f"S24 {target} did not return preterminal spatial metrics."
                )
            metrics_path = stage_dir / f"S24_{target}_{S24_PROTOCOL}_metrics.csv"
            spatial_metrics.to_csv(metrics_path, index=False)
            outputs.append(metrics_path)
            publication_metrics[target] = str(metrics_path)
            support_trajectories[f"{target}_matched_baseline"] = result.baseline_points
            support_trajectories[f"{target}_{variant}"] = result.ablation_points[
                variant
            ]

        audit, audit_summary = _compute_trajectory_support_audit(
            np.asarray(ctx.adata.obsm[ctx.args.latent_key], dtype=np.float32),
            support_trajectories,
            time_points,
            spatial_dim=2,
            max_outside_fraction=S24_SUPPORT_MAX_OUTSIDE_FRACTION,
            max_norm_multiplier=S24_SUPPORT_MAX_NORM_MULTIPLIER,
        )
        audit_path = stage_dir / f"S24_{S24_PROTOCOL}_trajectory_support_audit.csv"
        audit.to_csv(audit_path, index=False)
        audit_summary_path = (
            stage_dir / f"S24_{S24_PROTOCOL}_trajectory_support_audit.json"
        )
        audit_summary_path.write_text(
            json.dumps(_json_ready(audit_summary), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        outputs.extend([audit_path, audit_summary_path])
        _require_trajectory_support_audit_pass(
            audit_summary, stage=f"S24 {S24_PROTOCOL}"
        )

        captions: dict[str, str] = {}
        publication_grids: dict[str, list[str]] = {}
        for spec in experiments:
            target = str(spec["target"])
            variant = str(spec["variant"])
            result = results[target]
            matched_n = matched_counts[target]
            comparison_trajectories = {
                "baseline": result.baseline_points,
                variant: result.ablation_points[variant],
            }
            comparison_labels = None
            if result.baseline_labels is not None:
                comparison_labels = {
                    "baseline": result.baseline_labels,
                    variant: result.ablation_labels[variant],
                }
            condition_titles = {
                "baseline": f"{target}-matched baseline (N={matched_n})",
                variant: f"{target}-excluded matched cohort (N={matched_n})",
            }
            title = f"{target}-excluded spatial sensitivity through t=3 " "(sigma=0)"
            captions[target] = (
                f"The {S24_PROTOCOL} panel compares independently sampled baseline "
                f"and {target}-excluded t=0 cohorts matched at N={matched_n}. Both "
                "cohorts were propagated continuously from t=0 through observed t=3 "
                f"with dt={S24_FIXED_DT:g}, sigma=0, no growth-driven resampling, no "
                "re-anchoring, and no spatial warp. Learned velocity drift, "
                "score-gradient correction, and interactions were retained. Brownian "
                "diffusion was disabled, and one explicit interaction-grouping seed "
                "was shared for deterministic replay. This is a conditional spatial "
                "model sensitivity, not a stochastic forecast, total-mass deletion, "
                "causal knockout, or full joint-state terminal result. Terminal t=4 "
                "is not evaluated or claimed because this preterminal protocol is "
                "defined through observed t=3."
            )
            publication_grids[target] = []
            for extension in ("pdf", "png"):
                path = stage_dir / f"S24_{target}_{S24_PROTOCOL}_grid.{extension}"
                with mpl.rc_context(
                    {
                        "font.family": "Arial",
                        "font.size": 9.0,
                        "axes.titlesize": 9.0,
                        "axes.labelsize": 9.0,
                        "xtick.labelsize": 9.0,
                        "ytick.labelsize": 9.0,
                        "legend.fontsize": 9.0,
                        "pdf.fonttype": 42,
                        "ps.fonttype": 42,
                    }
                ):
                    figure = cb.pl.plot_trajectory_comparison_grid(
                        trajectories=comparison_trajectories,
                        time_values=time_points,
                        labels_by_condition=comparison_labels,
                        label_to_color=ctx.label_to_color,
                        selected_times=S24_PUBLICATION_TIMES,
                        condition_titles=condition_titles,
                        dim_pair=(0, 1),
                        point_size=float(ctx.args.point_size),
                        alpha=0.9,
                        shared_axis_limits=True,
                        show_counts=True,
                        show_legend=False,
                        title=title,
                        out_path=str(path),
                    )
                plt.close(figure)
                outputs.append(path)
                publication_grids[target].append(str(path))

        captions_path = stage_dir / f"S24_{S24_PROTOCOL}_panel_captions.json"
        captions_path.write_text(
            json.dumps(captions, indent=2, sort_keys=True), encoding="utf-8"
        )
        outputs.append(captions_path)
        return outputs, {
            "classifier_cache_path": str(cache_path),
            "classifier_accuracy": cached.accuracy,
            "classifier_balanced_accuracy": cached.balanced_accuracy,
            "matched_initial_particle_counts": matched_counts,
            "simulation_seeds": {
                target: results[target].settings["simulation_seeds"]
                for target in results
            },
            "interaction_seed": interaction_seed,
            "interaction_m": S24_INTERACTION_M,
            "publication_protocol": S24_PROTOCOL,
            "time_points": time_points,
            "publication_snapshot_times": list(S24_PUBLICATION_TIMES),
            "end_time": S24_END_TIME,
            "dt": S24_FIXED_DT,
            "resample_dt": S24_FIXED_DT,
            "sigma": S24_FIXED_SIGMA,
            "growth_alpha": S24_FIXED_GROWTH_ALPHA,
            "publication_metric_scope": "spatial only through observed t=3",
            "terminal_t4_included": False,
            "trajectory_support_audit": audit_summary,
            "publication_metrics": publication_metrics,
            "publication_grids": publication_grids,
            "panel_captions": captions,
            "superseded_legacy_result_reused": False,
        }

    return _execute_stage(ctx, "ablation", settings, action)


def _stage_s25(ctx: RunContext) -> dict[str, object]:
    top_n = 25 if ctx.args.profile == "smoke" else int(ctx.args.s25_top_genes)
    external_bundle_arg = getattr(ctx.args, "s25_canonical_state_bundle", None)
    external_bundle = (
        None
        if external_bundle_arg is None
        else _require_dir(
            external_bundle_arg,
            "external S25 canonical interval-local state bundle",
        )
    )
    # S22 now represents a different scientific estimand (one global-t0 path).
    # Reuse an interval-local bundle only when the caller opts in explicitly;
    # never infer compatibility from a neighboring S22 directory name.
    canonical_index = (
        external_bundle / "index.json" if external_bundle is not None else None
    )
    if external_bundle is not None:
        assert canonical_index is not None
        canonical_index = _require_file(
            canonical_index, "external S25 canonical state index"
        )
        upstream_s22_manifest = None
        external_manifest_path = _require_file(
            external_bundle.parent / "stage_manifest.json",
            "adjacent external S22 stage manifest",
        )
        external_s22_manifest = json.loads(
            external_manifest_path.read_text(encoding="utf-8")
        )
        _require_manifest_current_common_signature(
            ctx,
            external_s22_manifest,
            external_manifest_path,
            stage="s22",
        )
        _require_s25_interval_local_manifest_semantics(
            external_s22_manifest, external_manifest_path
        )
        _require_s22_state_settings_match(
            ctx, external_s22_manifest, external_manifest_path
        )
        if not _recorded_outputs_exist(external_s22_manifest):
            raise RuntimeError(
                "External S22 outputs no longer match its stage manifest: "
                f"{external_manifest_path}"
            )
        recorded_outputs = {
            str(Path(path).expanduser().resolve())
            for path in external_s22_manifest.get("outputs", [])
        }
        if str(canonical_index.resolve()) not in recorded_outputs:
            raise RuntimeError(
                "External canonical state index is not recorded by its adjacent "
                f"S22 manifest: {canonical_index}"
            )
    else:
        upstream_s22_manifest = None
        external_s22_manifest = None
        external_manifest_path = None
    settings = {
        "time_points": list(HALF_TIMES),
        "trajectory": (
            "observed integer states with actual annotations plus canonical "
            "interval-local one-sided half-time states simulated from the preceding "
            "observed anchor"
        ),
        "trajectory_scope": S25_COMMUNICATION_TRAJECTORY_SCOPE,
        "trajectory_mode": "piecewise_observed_anchored_interval_forward_simulation",
        "split_sde_piecewise": True,
        "piecewise_observed_sample_mode": "per_timepoint",
        "piecewise_include_end": False,
        "dt": float(ctx.args.sde_dt),
        "split_resample_dt": float(ctx.args.sde_dt),
        "sigma": float(ctx.args.sde_sigma),
        "growth_alpha": float(ctx.args.growth_alpha),
        "interaction_m": int(ctx.args.interaction_m),
        "sde_n_samples": (
            int(ctx.args.smoke_n_samples)
            if ctx.args.profile == "smoke"
            else ctx.args.sde_n_samples
        ),
        "max_particles": int(ctx.args.sde_max_particles),
        "cell_type_filter": ctx.args.ysl_label,
        "daughter_noise_std": 0.0,
        "target_classifier_knn_neighbors": int(ctx.args.s25_classifier_knn_neighbors),
        "target_classifier_policy": (
            "same k=10 spatially smoothed classifier labels used by the generated "
            "trajectory, composition, and communication analyses"
        ),
        "top_variable_genes": top_n,
        "gene_expression_space": (
            "per-cell rank-retained inverse-PCA processed log1p, clipped at zero "
            "before the mean for both observed and generated states"
        ),
        "signed_gene_diagnostic": (
            "unclipped inverse-PCA means and per-time negative-value diagnostics"
        ),
        "observed_gene_validation": (
            "exact observed integer-time YSL log1p means are exported separately; "
            "they are not mixed into the rank-50 temporal trajectory"
        ),
        "pca_center": "persisted fit-time adata.var['pca_center']",
        "pca_feature_policy": (
            "active retained-component loadings only; center-only genes excluded"
        ),
        "heatmap_panel_columns": S25_HEATMAP_COLUMNS,
        "preferred_species_tag": ctx.args.preferred_species_tag,
        "normalization": "gene-wise zscore",
        "linkage": "average/euclidean",
        "cluster_order": "dendrogram",
        "n_clusters": int(ctx.args.s25_n_clusters),
        "classifier": _main_classifier_settings(ctx),
        "canonical_trajectory": (
            {
                "path": str(canonical_index.resolve()),
                "sha256": _sha256(canonical_index),
                "source_stage": (
                    "external_validated_s22_bundle"
                    if external_bundle is not None
                    else "s22"
                ),
                "source_stage_signature": (
                    external_s22_manifest.get("signature")
                    if external_s22_manifest is not None
                    else (
                        upstream_s22_manifest.get("signature")
                        if upstream_s22_manifest is not None
                        else None
                    )
                ),
                "source_stage_manifest": (
                    str(external_manifest_path.resolve())
                    if external_manifest_path is not None
                    and external_manifest_path.is_file()
                    else None
                ),
                "source_stage_manifest_sha256": (
                    _sha256(external_manifest_path)
                    if external_manifest_path is not None
                    and external_manifest_path.is_file()
                    else None
                ),
            }
            if canonical_index is not None and canonical_index.exists()
            else None
        ),
        "missing_target_policy": (
            "smoke_only_first_8_cells_with_manifest_warning"
            if ctx.args.profile == "smoke"
            else "strict_error"
        ),
    }

    def action(stage_dir: Path):
        if canonical_index is not None and canonical_index.exists():
            states, canonical_times, canonical_sources = _read_state_bundle(
                canonical_index.parent,
                annotation_key=ctx.args.annotation_key,
            )
            if canonical_times != list(HALF_TIMES):
                raise RuntimeError(
                    "External interval-local slices do not match the S25 grid."
                )
            invalid_sources = {
                time_value: source
                for time_value, source in canonical_sources.items()
                if source
                not in {
                    "observed_actual_annotation",
                    "generated_interval_local_one_sided",
                }
            }
            if invalid_sources:
                raise RuntimeError(
                    f"Unexpected canonical trajectory sources: {invalid_sources}"
                )
            source_manifest = (
                external_s22_manifest
                if external_s22_manifest is not None
                else upstream_s22_manifest
            )
            s22_details = (
                {} if source_manifest is None else source_manifest.get("details", {})
            )
            classifier_cache_path = s22_details.get("classifier_cache_path")
            classifier_accuracy = s22_details.get("classifier_accuracy")
            classifier_balanced_accuracy = s22_details.get(
                "classifier_balanced_accuracy"
            )
            simulation_seeds = s22_details.get("simulation_seeds", {})
            trajectory_source = (
                "external_validated_s22_canonical_interval_local"
                if external_bundle is not None
                else "s22_canonical_interval_local"
            )
            if classifier_cache_path is None:
                cached, cache_path = _train_main_classifier(ctx)
                classifier_cache_path = str(cache_path)
                classifier_accuracy = cached.accuracy
                classifier_balanced_accuracy = cached.balanced_accuracy
        else:
            result = _run_interpolation(
                ctx,
                output_dir=stage_dir / "workflow",
                time_points=HALF_TIMES,
                trajectory_mode="interval_local_observed_anchored",
                split_growth_alpha=float(ctx.args.growth_alpha),
                display_piecewise_warp=False,
            )
            states = result.adata_dict
            classifier_cache_path = result.classifier_cache_path
            classifier_accuracy = result.classifier_accuracy
            classifier_balanced_accuracy = result.classifier_balanced_accuracy
            simulation_seeds = result.simulation_seeds
            trajectory_source = "stage_local_piecewise_observed_anchored"
        outputs = _write_state_bundle(
            states,
            HALF_TIMES,
            stage_dir / "generated_states",
            annotation_key=ctx.args.annotation_key,
            source_by_time={
                float(value): (
                    "observed_actual_annotation"
                    if float(value) in OBSERVED_TIMES
                    else "generated_interval_local_one_sided"
                )
                for value in HALF_TIMES
            },
        )
        observed_states = _observed_state_dict(ctx)
        cached_classifier = cb.tl.load_cached_mlp_classifier(
            str(classifier_cache_path), device=ctx.args.device
        )
        analysis_states: dict[str, ad.AnnData] = {}
        analysis_source_by_time: dict[float, str] = {}
        for time_value in HALF_TIMES:
            key = str(float(time_value))
            if float(time_value) in observed_states:
                analysis_states[key] = observed_states[float(time_value)]
                analysis_source_by_time[
                    float(time_value)
                ] = "observed_actual_annotation"
                continue
            points = np.asarray(states[key].X, dtype=np.float32)
            labels = cb.tl.predict_labels_for_points(
                points=points,
                time_value=float(time_value),
                model=cached_classifier.model,
                label_encoder=cached_classifier.label_encoder,
                feature_dim=int(_main_classifier_settings(ctx)["n_joint_features"]),
                device=ctx.args.device,
                knn_neighbors=int(ctx.args.s25_classifier_knn_neighbors),
                include_time_feature=cached_classifier.include_time_feature,
            )
            analysis_states[key] = _minimal_state_adata(
                points, labels, annotation_key=ctx.args.annotation_key
            )
            analysis_source_by_time[float(time_value)] = (
                "generated_interval_local_classifier_knn_"
                f"{int(ctx.args.s25_classifier_knn_neighbors)}"
            )
        outputs.extend(
            _write_state_bundle(
                analysis_states,
                HALF_TIMES,
                stage_dir / "hybrid_analysis_states",
                annotation_key=ctx.args.annotation_key,
                source_by_time=analysis_source_by_time,
            )
        )
        ysl_states: dict[str, ad.AnnData] = {}
        ysl_counts: dict[str, int] = {}
        smoke_fallbacks: list[dict[str, object]] = []
        for time_value in HALF_TIMES:
            key = str(float(time_value))
            state = analysis_states[key]
            mask = state.obs[ctx.args.annotation_key].astype(str).to_numpy() == str(
                ctx.args.ysl_label
            )
            if not np.any(mask):
                if ctx.args.profile != "smoke":
                    raise ValueError(
                        f"No predicted {ctx.args.ysl_label!r} cells at t={time_value}; "
                        "cannot reproduce the S25 cell-type-specific heatmap"
                    )
                # A two-epoch smoke classifier is deliberately too small to
                # guarantee every rare label.  Select a deterministic handful
                # only so CI/smoke runs can exercise inverse PCA, clustering,
                # and plotting.  Full scientific runs remain strict above.
                fallback_n = min(8, int(state.n_obs))
                if fallback_n == 0:
                    raise ValueError(
                        f"Generated state at t={time_value} is empty; "
                        "the S25 smoke mechanics cannot be exercised"
                    )
                mask = np.zeros(state.n_obs, dtype=bool)
                mask[:fallback_n] = True
                smoke_fallbacks.append(
                    {
                        "time": float(time_value),
                        "selected_first_n_cells": fallback_n,
                        "reason": f"no predicted {ctx.args.ysl_label!r} cells",
                    }
                )
            ysl_states[key] = state[mask].copy()
            ysl_counts[key] = int(np.sum(mask))

        temporal = cb.tl.summarize_temporal_gene_patterns(
            ysl_states,
            ctx.adata,
            time_points=HALF_TIMES,
            spatial_dim=2,
            loadings_key="PCs",
            reference_layer=None,
            n_top_genes=top_n,
            n_cluster_genes=top_n,
            n_clusters=int(ctx.args.s25_n_clusters),
            preferred_species_tag=ctx.args.preferred_species_tag,
            profile_normalization="zscore",
            profile_linkage_method="average",
            profile_cluster_order="dendrogram",
            active_features_only=True,
            clip_min=0.0,
        )
        reference_var_names = pd.Index(ctx.adata.var_names.astype(str))
        active_var_names = pd.Index(temporal.gene_name_map["var_name"].astype(str))
        if not reference_var_names.is_unique or not active_var_names.is_unique:
            raise ValueError(
                "S25 exact-observed validation requires unique reference feature names."
            )
        active_indexer = reference_var_names.get_indexer(active_var_names)
        if np.any(active_indexer < 0):
            missing = active_var_names[active_indexer < 0][:5].tolist()
            raise ValueError(
                "S25 active inverse-PCA features are missing from aligned expression; "
                f"examples={missing}."
            )
        observed_times = np.asarray(
            [
                cb.tl.parse_time_value(value)
                for value in ctx.adata.obs[ctx.args.time_key]
            ],
            dtype=np.float64,
        )
        observed_labels = ctx.adata.obs[ctx.args.annotation_key].astype(str).to_numpy()
        exact_columns: dict[float, np.ndarray] = {}
        validation_rows: list[dict[str, object]] = []
        top_gene_names = set(temporal.top_variable_genes["gene"].astype(str).tolist())
        top_mask = temporal.expression.index.astype(str).isin(top_gene_names)
        from scipy import sparse

        for time_value in OBSERVED_TIMES:
            observed_mask = np.isclose(
                observed_times, float(time_value), rtol=0.0, atol=1e-9
            ) & (observed_labels == str(ctx.args.ysl_label))
            if not observed_mask.any():
                raise ValueError(
                    f"No exact observed {ctx.args.ysl_label!r} cells at "
                    f"t={time_value} for S25 validation."
                )
            observed_matrix = ctx.adata.X[observed_mask][:, active_indexer]
            if sparse.issparse(observed_matrix):
                exact_mean = np.asarray(observed_matrix.mean(axis=0)).reshape(-1)
            else:
                exact_mean = np.asarray(observed_matrix, dtype=np.float64).mean(axis=0)
            exact_columns[float(time_value)] = exact_mean
            for decoder_name, decoded in (
                (
                    "signed_inverse_pca",
                    temporal.signed_expression[float(time_value)].to_numpy(
                        dtype=np.float64
                    ),
                ),
                (
                    "clipped_inverse_pca",
                    temporal.expression[float(time_value)].to_numpy(dtype=np.float64),
                ),
            ):
                for scope, scope_mask in (
                    ("all_active_features", np.ones(len(exact_mean), dtype=bool)),
                    ("top_temporal_variance", np.asarray(top_mask, dtype=bool)),
                ):
                    exact_values = exact_mean[scope_mask]
                    decoded_values = decoded[scope_mask]
                    delta = decoded_values - exact_values
                    correlation = (
                        float(np.corrcoef(exact_values, decoded_values)[0, 1])
                        if exact_values.size > 1
                        and np.std(exact_values) > 0
                        and np.std(decoded_values) > 0
                        else np.nan
                    )
                    validation_rows.append(
                        {
                            "time": float(time_value),
                            "decoder": decoder_name,
                            "scope": scope,
                            "n_cells": int(observed_mask.sum()),
                            "n_features": int(exact_values.size),
                            "rmse": float(np.sqrt(np.mean(delta**2))),
                            "mae": float(np.mean(np.abs(delta))),
                            "mean_bias": float(np.mean(delta)),
                            "pearson_r": correlation,
                        }
                    )
        exact_observed_expression = pd.DataFrame(
            exact_columns,
            index=temporal.expression.index.copy(),
        )
        observed_validation = pd.DataFrame(validation_rows)
        tables = {
            "mean_expression.csv": temporal.expression,
            "mean_inverse_pca_log1p_clipped.csv": temporal.expression,
            "mean_inverse_pca_log1p_signed.csv": temporal.signed_expression,
            "inverse_pca_reconstruction_diagnostics.csv": (
                temporal.reconstruction_diagnostics
            ),
            "observed_exact_log1p_anchors.csv": exact_observed_expression,
            "observed_vs_inverse_pca_metrics.csv": observed_validation,
            "top_variable_genes.csv": temporal.top_variable_genes,
            "normalized_profiles.csv": temporal.clustering.normalized_profiles,
            "cluster_assignments.csv": temporal.clustering.assignments,
            "cluster_prototypes.csv": temporal.clustering.prototypes,
            "cluster_diagnostics.csv": temporal.clustering.diagnostics,
            "gene_name_map.csv": temporal.gene_name_map,
        }
        for filename, table in tables.items():
            path = stage_dir / filename
            index = filename in {
                "mean_expression.csv",
                "mean_inverse_pca_log1p_clipped.csv",
                "mean_inverse_pca_log1p_signed.csv",
                "observed_exact_log1p_anchors.csv",
                "normalized_profiles.csv",
            }
            table.to_csv(path, index=index)
            outputs.append(path)
        import matplotlib.pyplot as plt

        figure, axes = plt.subplots(2, 3, figsize=(12.0, 7.6), facecolor="white")
        for axis, time_value in zip(axes.ravel(), OBSERVED_TIMES):
            exact_values = exact_observed_expression[float(time_value)].to_numpy(
                dtype=np.float64
            )
            decoded_values = temporal.expression[float(time_value)].to_numpy(
                dtype=np.float64
            )
            axis.scatter(
                exact_values,
                decoded_values,
                s=5,
                alpha=0.35,
                color="#4C78A8",
                linewidths=0,
            )
            axis.scatter(
                exact_values[np.asarray(top_mask, dtype=bool)],
                decoded_values[np.asarray(top_mask, dtype=bool)],
                s=7,
                alpha=0.55,
                color="#E45756",
                linewidths=0,
                label=f"top {top_n}",
            )
            lo = float(min(exact_values.min(), decoded_values.min()))
            hi = float(max(exact_values.max(), decoded_values.max()))
            axis.plot([lo, hi], [lo, hi], color="black", linewidth=0.8, alpha=0.7)
            metric_row = observed_validation.loc[
                (observed_validation["time"] == float(time_value))
                & (observed_validation["decoder"] == "clipped_inverse_pca")
                & (observed_validation["scope"] == "all_active_features")
            ].iloc[0]
            axis.set_title(
                f"t={time_value:g}: r={metric_row['pearson_r']:.3f}, "
                f"RMSE={metric_row['rmse']:.3f}"
            )
            axis.set_xlabel("exact observed mean log1p")
            axis.set_ylabel("clipped inverse-PCA mean")
        axes.ravel()[-1].axis("off")
        axes.ravel()[0].legend(frameon=False, loc="lower right")
        figure.suptitle(
            f"{ctx.args.ysl_label}: observed anchors vs rank-50 reconstruction"
        )
        figure.tight_layout()
        for suffix in ("pdf", "png"):
            validation_path = (
                stage_dir / f"S25_observed_vs_inverse_pca_validation.{suffix}"
            )
            figure.savefig(validation_path, dpi=300, bbox_inches="tight")
            outputs.append(validation_path)
        plt.close(figure)
        heatmap_pdf = stage_dir / "S25_YSL_top250_temporal_variance_heatmap.pdf"
        heatmap_png = stage_dir / "S25_YSL_top250_temporal_variance_heatmap.png"
        cb.pl.plot_temporal_gene_heatmap(
            temporal.expression,
            temporal.top_variable_genes,
            out_path=heatmap_pdf,
            top_n=top_n,
            title=f"{ctx.args.ysl_label}: top {top_n} temporal-variance genes",
            panel_columns=S25_HEATMAP_COLUMNS,
        )
        cb.pl.plot_temporal_gene_heatmap(
            temporal.expression,
            temporal.top_variable_genes,
            out_path=heatmap_png,
            top_n=top_n,
            title=f"{ctx.args.ysl_label}: top {top_n} temporal-variance genes",
            panel_columns=S25_HEATMAP_COLUMNS,
        )
        outputs.extend([heatmap_pdf, heatmap_png])
        settings_path = stage_dir / "temporal_settings.json"
        settings_path.write_text(
            json.dumps(_json_ready(temporal.settings), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        outputs.append(settings_path)
        return outputs, {
            "classifier_cache_path": classifier_cache_path,
            "classifier_accuracy": classifier_accuracy,
            "classifier_balanced_accuracy": classifier_balanced_accuracy,
            "trajectory_source": trajectory_source,
            "generated_cell_counts": {
                str(float(t)): int(states[str(float(t))].n_obs) for t in HALF_TIMES
            },
            "analysis_cell_counts": {
                str(float(t)): int(analysis_states[str(float(t))].n_obs)
                for t in HALF_TIMES
            },
            "analysis_state_sources": analysis_source_by_time,
            "ysl_cell_counts": ysl_counts,
            "gene_expression_contract": temporal.settings,
            "observed_inverse_pca_validation": validation_rows,
            "simulation_seeds": simulation_seeds,
            "smoke_only_target_fallbacks": smoke_fallbacks,
            "scientific_use_warning": (
                "Smoke-only fallback cells are not YSL and these outputs must not be "
                "used scientifically."
                if smoke_fallbacks
                else None
            ),
        }

    return _execute_stage(ctx, "s25", settings, action)


def _observed_state_dict(ctx: RunContext) -> dict[float, ad.AnnData]:
    joint = _joint_features(
        ctx.adata, latent_key=ctx.args.latent_key, spatial_key=ctx.args.spatial_key
    )
    raw_times = np.asarray(
        [cb.tl.parse_time_value(value) for value in ctx.adata.obs[ctx.args.time_key]],
        dtype=np.float64,
    )
    labels = ctx.adata.obs[ctx.args.annotation_key].astype(str).to_numpy()
    result: dict[float, ad.AnnData] = {}
    for time_value in OBSERVED_TIMES:
        mask = np.isclose(raw_times, time_value, rtol=0.0, atol=1e-9)
        if not np.any(mask):
            raise ValueError(f"No observed cells at canonical time {time_value}")
        result[time_value] = _minimal_state_adata(
            joint[mask], labels[mask], annotation_key=ctx.args.annotation_key
        )
    return result


def _build_explicitly_labeled_hybrid_states(
    generated_states: Mapping[str, ad.AnnData],
    observed_states: Mapping[float, ad.AnnData],
    *,
    time_points: Sequence[float],
    annotation_key: str,
    cached_classifier: object,
    classifier_feature_dim: int,
    device: str,
    knn_neighbors: int,
) -> tuple[
    dict[str, ad.AnnData],
    dict[float, str],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Build an observed/generated series with an explicit label policy.

    Generated-state annotations stored in an upstream trajectory bundle are
    provenance, not an analysis contract: they may have been produced for a
    display-only workflow with a different spatial-smoothing value.  This
    helper therefore re-predicts every generated frame with the classifier and
    ``knn_neighbors`` requested by the consuming analysis.  Observed frames
    retain their experimental annotations.
    """

    knn_neighbors = int(knn_neighbors)
    if knn_neighbors <= 0:
        raise ValueError("knn_neighbors must be > 0")
    hybrid: dict[str, ad.AnnData] = {}
    source_by_time: dict[float, str] = {}
    assignment_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    count_rows: list[dict[str, object]] = []
    for raw_time in time_points:
        time_value = float(raw_time)
        key = str(time_value)
        if time_value in observed_states:
            state = observed_states[time_value]
            labels = state.obs[annotation_key].astype(str).to_numpy()
            hybrid[key] = state
            source_by_time[time_value] = "observed_actual_annotation"
            summary_rows.append(
                {
                    "time": time_value,
                    "state_source": "observed",
                    "n_cells": int(state.n_obs),
                    "n_labels_changed": 0,
                    "fraction_labels_changed": 0.0,
                    "knn_neighbors": np.nan,
                    "points_preserved_exactly": True,
                }
            )
            for label, count in pd.Series(labels).value_counts().sort_index().items():
                count_rows.append(
                    {
                        "time": time_value,
                        "state_source": "observed",
                        "label_policy": "actual_annotation",
                        "label": str(label),
                        "n_cells": int(count),
                    }
                )
            continue
        if key not in generated_states:
            raise KeyError(f"Generated state dictionary is missing time key {key!r}")
        inherited_state = generated_states[key]
        if annotation_key not in inherited_state.obs:
            raise KeyError(
                f"Generated state {key!r} is missing annotation {annotation_key!r}"
            )
        points = np.asarray(inherited_state.X, dtype=np.float32)
        inherited_labels = inherited_state.obs[annotation_key].astype(str).to_numpy()
        analysis_labels = np.asarray(
            cb.tl.predict_labels_for_points(
                points=points,
                time_value=time_value,
                model=cached_classifier.model,
                label_encoder=cached_classifier.label_encoder,
                feature_dim=int(classifier_feature_dim),
                device=device,
                knn_neighbors=knn_neighbors,
                include_time_feature=cached_classifier.include_time_feature,
            )
        ).astype(str)
        if analysis_labels.shape != inherited_labels.shape:
            raise RuntimeError(
                "Classifier returned a different number of generated labels: "
                f"time={time_value}, expected={len(inherited_labels)}, "
                f"actual={len(analysis_labels)}"
            )
        changed = analysis_labels != inherited_labels
        hybrid[key] = _minimal_state_adata(
            points, analysis_labels, annotation_key=annotation_key
        )
        if not np.array_equal(np.asarray(hybrid[key].X), points):
            raise RuntimeError(
                f"Generated points changed while relabeling time {time_value}."
            )
        source_by_time[
            time_value
        ] = f"generated_interval_local_classifier_knn_{knn_neighbors}"
        summary_rows.append(
            {
                "time": time_value,
                "state_source": "generated_interval_local",
                "n_cells": int(len(analysis_labels)),
                "n_labels_changed": int(changed.sum()),
                "fraction_labels_changed": float(changed.mean()),
                "knn_neighbors": knn_neighbors,
                "points_preserved_exactly": True,
            }
        )
        assignment_rows.extend(
            {
                "time": time_value,
                "row_index": int(row_index),
                "inherited_label": str(inherited),
                "analysis_label": str(analysis),
                "changed": bool(is_changed),
                "knn_neighbors": knn_neighbors,
            }
            for row_index, (inherited, analysis, is_changed) in enumerate(
                zip(inherited_labels, analysis_labels, changed)
            )
        )
        for policy, labels in (
            ("inherited_bundle", inherited_labels),
            (f"classifier_knn_{knn_neighbors}", analysis_labels),
        ):
            for label, count in pd.Series(labels).value_counts().sort_index().items():
                count_rows.append(
                    {
                        "time": time_value,
                        "state_source": "generated_interval_local",
                        "label_policy": policy,
                        "label": str(label),
                        "n_cells": int(count),
                    }
                )
    return (
        hybrid,
        source_by_time,
        pd.DataFrame(assignment_rows),
        pd.DataFrame(summary_rows),
        pd.DataFrame(count_rows),
    )


def _compare_lr_measurement_contracts(
    hybrid_result: object,
    all_inverse_result: object,
    *,
    observed_times: Sequence[float],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame,]:
    """Compare exact-observed and all-inverse-PCA LR measurements.

    The legacy hybrid trajectory uses exact expression at observed integer
    times and inverse-PCA expression at generated half-times. The corrected
    default uses inverse-PCA expression throughout. This diagnostic holds
    states, labels, and communication matrices fixed and changes only that
    expression measurement operator.
    """

    observed = {float(value) for value in observed_times}
    hybrid_settings = getattr(hybrid_result, "settings", None)
    inverse_settings = getattr(all_inverse_result, "settings", None)
    if isinstance(hybrid_settings, Mapping) and isinstance(inverse_settings, Mapping):
        for contract_key in (
            "feature_universe",
            "pca_feature_coverage",
            "global_lr_coverage",
            "pca_center_source",
        ):
            if hybrid_settings.get(contract_key) != inverse_settings.get(contract_key):
                raise RuntimeError(
                    "Hybrid and all-inverse LR projections used different "
                    f"{contract_key!r} contracts."
                )
    pair = hybrid_result.pair_timecourse.merge(
        all_inverse_result.pair_timecourse,
        on=["pair", "time"],
        how="outer",
        suffixes=("_hybrid", "_all_inverse_pca"),
        indicator="pair_presence",
        validate="one_to_one",
    )
    pair["measurement_source_hybrid"] = np.where(
        pair["time"].astype(float).isin(observed),
        "exact_observed_expression",
        "inverse_pca_expression",
    )
    pair["score_delta_all_inverse_minus_hybrid"] = (
        pair["score_all_inverse_pca"] - pair["score_hybrid"]
    )
    pair["score_abs_delta"] = pair["score_delta_all_inverse_minus_hybrid"].abs()
    pair["score_ratio_all_inverse_over_hybrid"] = np.where(
        pair["score_hybrid"] > 0,
        pair["score_all_inverse_pca"] / pair["score_hybrid"],
        np.nan,
    )
    pair["score_log2_ratio_all_inverse_over_hybrid"] = np.where(
        (pair["score_hybrid"] > 0) & (pair["score_all_inverse_pca"] > 0),
        np.log2(pair["score_all_inverse_pca"] / pair["score_hybrid"]),
        np.nan,
    )
    symmetric_denominator = (
        pair["score_hybrid"].abs() + pair["score_all_inverse_pca"].abs()
    )
    pair["score_symmetric_relative_error"] = np.where(
        symmetric_denominator > 0,
        2.0 * pair["score_abs_delta"] / symmetric_denominator,
        0.0,
    )
    pair["zero_status"] = np.select(
        (
            pair["pair_presence"].eq("left_only"),
            pair["pair_presence"].eq("right_only"),
            pair["score_hybrid"].eq(0) & pair["score_all_inverse_pca"].eq(0),
            pair["score_hybrid"].eq(0) & pair["score_all_inverse_pca"].gt(0),
            pair["score_hybrid"].gt(0) & pair["score_all_inverse_pca"].eq(0),
        ),
        (
            "all_inverse_missing",
            "hybrid_missing",
            "both_zero",
            "hybrid_only_zero",
            "all_inverse_only_zero",
        ),
        default="both_nonzero",
    )

    celltype = hybrid_result.celltype_timecourse.merge(
        all_inverse_result.celltype_timecourse,
        on=["pair", "time", "cell_type"],
        how="outer",
        suffixes=("_hybrid", "_all_inverse_pca"),
        indicator="celltype_presence",
        validate="one_to_one",
    )
    celltype["measurement_source_hybrid"] = np.where(
        celltype["time"].astype(float).isin(observed),
        "exact_observed_expression",
        "inverse_pca_expression",
    )
    for column in ("incoming", "outgoing", "total"):
        celltype[f"{column}_delta_all_inverse_minus_hybrid"] = (
            celltype[f"{column}_all_inverse_pca"] - celltype[f"{column}_hybrid"]
        )
    common_celltypes = celltype["celltype_presence"].eq("both")
    if not np.array_equal(
        celltype.loc[common_celltypes, "n_cells_hybrid"].to_numpy(),
        celltype.loc[common_celltypes, "n_cells_all_inverse_pca"].to_numpy(),
    ):
        raise RuntimeError(
            "LR measurement comparison changed cell counts even though states and "
            "labels were held fixed."
        )

    metric_rows: list[dict[str, object]] = []
    for time_value, subset in pair.groupby("time", sort=True):
        common = subset["pair_presence"].eq("both")
        hybrid_values = subset.loc[common, "score_hybrid"].to_numpy(dtype=float)
        inverse_values = subset.loc[common, "score_all_inverse_pca"].to_numpy(
            dtype=float
        )
        delta = inverse_values - hybrid_values
        hybrid_sum = float(np.sum(hybrid_values))
        inverse_sum = float(np.sum(inverse_values))
        rmse = float(np.sqrt(np.mean(delta**2))) if len(delta) else np.nan
        hybrid_rms = (
            float(np.sqrt(np.mean(hybrid_values**2))) if len(delta) else np.nan
        )
        if len(delta) > 1 and np.std(hybrid_values) > 0 and np.std(inverse_values) > 0:
            pearson_r = float(np.corrcoef(hybrid_values, inverse_values)[0, 1])
            spearman_r = float(
                pd.Series(hybrid_values).corr(
                    pd.Series(inverse_values), method="spearman"
                )
            )
        else:
            pearson_r = np.nan
            spearman_r = np.nan
        metric_rows.append(
            {
                "time": float(time_value),
                "measurement_source_hybrid": (
                    "exact_observed_expression"
                    if float(time_value) in observed
                    else "inverse_pca_expression"
                ),
                "n_pairs_common": int(common.sum()),
                "n_pairs_hybrid_only": int(
                    subset["pair_presence"].eq("left_only").sum()
                ),
                "n_pairs_all_inverse_only": int(
                    subset["pair_presence"].eq("right_only").sum()
                ),
                "rmse": (rmse),
                "relative_rmse": (
                    float(rmse / hybrid_rms)
                    if len(delta) and hybrid_rms > 0
                    else np.nan
                ),
                "mae": float(np.mean(np.abs(delta))) if len(delta) else np.nan,
                "mean_bias_all_inverse_minus_hybrid": (
                    float(np.mean(delta)) if len(delta) else np.nan
                ),
                "max_abs_delta": (
                    float(np.max(np.abs(delta))) if len(delta) else np.nan
                ),
                "relative_l1": (
                    float(np.sum(np.abs(delta)) / np.sum(np.abs(hybrid_values)))
                    if len(delta) and np.sum(np.abs(hybrid_values)) > 0
                    else np.nan
                ),
                "hybrid_score_sum": hybrid_sum,
                "all_inverse_pca_score_sum": inverse_sum,
                "score_sum_ratio_all_inverse_over_hybrid": (
                    float(inverse_sum / hybrid_sum) if hybrid_sum > 0 else np.nan
                ),
                "n_zero_mismatches": int(
                    (np.equal(hybrid_values, 0) != np.equal(inverse_values, 0)).sum()
                ),
                "median_abs_log2_ratio": (
                    float(
                        np.median(
                            np.abs(
                                np.log2(
                                    inverse_values[
                                        (hybrid_values > 0) & (inverse_values > 0)
                                    ]
                                    / hybrid_values[
                                        (hybrid_values > 0) & (inverse_values > 0)
                                    ]
                                )
                            )
                        )
                    )
                    if np.any((hybrid_values > 0) & (inverse_values > 0))
                    else np.nan
                ),
                "top10_pair_overlap": (
                    int(
                        len(
                            set(
                                subset.loc[common].nlargest(
                                    min(10, int(common.sum())), "score_hybrid"
                                )["pair"]
                            )
                            & set(
                                subset.loc[common].nlargest(
                                    min(10, int(common.sum())),
                                    "score_all_inverse_pca",
                                )["pair"]
                            )
                        )
                    )
                    if common.any()
                    else 0
                ),
                "pearson_r": pearson_r,
                "spearman_r": spearman_r,
            }
        )
    metrics = pd.DataFrame(metric_rows)
    generated_metrics = metrics.loc[
        metrics["measurement_source_hybrid"].eq("inverse_pca_expression")
    ]
    if (generated_metrics["max_abs_delta"].fillna(0.0) > 1e-12).any():
        raise RuntimeError(
            "Common hybrid/all-inverse LR pairs disagree at generated times, where "
            "their expression contracts must be identical."
        )

    continuity_rows: list[dict[str, object]] = []
    available_times = sorted(pair["time"].dropna().astype(float).unique())
    generated_times = [value for value in available_times if value not in observed]
    for anchor_time in sorted(observed):
        left_candidates = [value for value in generated_times if value < anchor_time]
        right_candidates = [value for value in generated_times if value > anchor_time]
        left_time = max(left_candidates) if left_candidates else None
        right_time = min(right_candidates) if right_candidates else None
        if left_time is not None and right_time is not None:
            neighbor_mode = "two_sided_linear"
        elif left_time is not None:
            neighbor_mode = "one_sided_left"
        elif right_time is not None:
            neighbor_mode = "one_sided_right"
        else:
            continue
        anchor_rows = pair.loc[
            np.isclose(pair["time"].astype(float), anchor_time)
            & pair["pair_presence"].eq("both")
        ]
        for anchor in anchor_rows.itertuples(index=False):
            neighbor_scores: dict[str, float] = {}
            for side, neighbor_time in (("left", left_time), ("right", right_time)):
                if neighbor_time is None:
                    continue
                match = pair.loc[
                    pair["pair"].eq(anchor.pair)
                    & np.isclose(pair["time"].astype(float), neighbor_time)
                    & pair["pair_presence"].eq("both")
                ]
                if len(match) == 1:
                    neighbor_scores[side] = float(match.iloc[0]["score_hybrid"])
            if left_time is not None and right_time is not None:
                if set(neighbor_scores) != {"left", "right"}:
                    continue
                right_weight = (anchor_time - left_time) / (right_time - left_time)
                reference_score = (1.0 - right_weight) * neighbor_scores[
                    "left"
                ] + right_weight * neighbor_scores["right"]
            elif "left" in neighbor_scores:
                reference_score = neighbor_scores["left"]
            elif "right" in neighbor_scores:
                reference_score = neighbor_scores["right"]
            else:
                continue
            exact_score = float(anchor.score_hybrid)
            decoded_score = float(anchor.score_all_inverse_pca)
            exact_residual = exact_score - reference_score
            decoded_residual = decoded_score - reference_score
            exact_slope_jump = np.nan
            decoded_slope_jump = np.nan
            if set(neighbor_scores) == {"left", "right"}:
                exact_slope_jump = (neighbor_scores["right"] - exact_score) / (
                    right_time - anchor_time
                ) - (exact_score - neighbor_scores["left"]) / (anchor_time - left_time)
                decoded_slope_jump = (neighbor_scores["right"] - decoded_score) / (
                    right_time - anchor_time
                ) - (decoded_score - neighbor_scores["left"]) / (
                    anchor_time - left_time
                )
            continuity_rows.append(
                {
                    "pair": str(anchor.pair),
                    "anchor_time": anchor_time,
                    "neighbor_mode": neighbor_mode,
                    "left_generated_time": left_time,
                    "right_generated_time": right_time,
                    "neighbor_reference_score": reference_score,
                    "exact_observed_anchor_score": exact_score,
                    "inverse_pca_anchor_score": decoded_score,
                    "exact_anchor_residual": exact_residual,
                    "inverse_pca_anchor_residual": decoded_residual,
                    "abs_residual_exact_minus_inverse": (
                        abs(exact_residual) - abs(decoded_residual)
                    ),
                    "exact_anchor_slope_jump": exact_slope_jump,
                    "inverse_pca_anchor_slope_jump": decoded_slope_jump,
                }
            )
    continuity = pd.DataFrame(continuity_rows)
    continuity_metric_rows: list[dict[str, object]] = []
    if not continuity.empty:
        for anchor_time, subset in continuity.groupby("anchor_time", sort=True):
            continuity_metric_rows.append(
                {
                    "anchor_time": float(anchor_time),
                    "neighbor_mode": str(subset["neighbor_mode"].iloc[0]),
                    "n_pairs": int(len(subset)),
                    "mean_abs_residual_exact": float(
                        subset["exact_anchor_residual"].abs().mean()
                    ),
                    "mean_abs_residual_inverse_pca": float(
                        subset["inverse_pca_anchor_residual"].abs().mean()
                    ),
                    "median_abs_residual_exact": float(
                        subset["exact_anchor_residual"].abs().median()
                    ),
                    "median_abs_residual_inverse_pca": float(
                        subset["inverse_pca_anchor_residual"].abs().median()
                    ),
                    "fraction_inverse_pca_closer_to_generated_neighbors": float(
                        (subset["abs_residual_exact_minus_inverse"] > 0).mean()
                    ),
                    "mean_abs_slope_jump_exact": float(
                        subset["exact_anchor_slope_jump"].abs().mean()
                    ),
                    "mean_abs_slope_jump_inverse_pca": float(
                        subset["inverse_pca_anchor_slope_jump"].abs().mean()
                    ),
                }
            )
    continuity_metrics = pd.DataFrame(continuity_metric_rows)
    return pair, celltype, metrics, continuity, continuity_metrics


def _plot_communication_heatmaps(
    communications: Mapping[str, Mapping[str, object]],
    output_dir: Path,
    *,
    selected_times: Sequence[float],
    display_min: float,
) -> list[Path]:
    import matplotlib.pyplot as plt
    import seaborn as sns

    outputs = []
    for time_value in selected_times:
        record = communications[str(float(time_value))]
        values = np.asarray(record["M_per_source"], dtype=float).copy()
        if float(display_min) > 0:
            values[values < float(display_min)] = 0.0
        labels = [str(value) for value in np.asarray(record["types"]).tolist()]
        figure, axis = plt.subplots(figsize=(8.2, 7.2), facecolor="white")
        sns.heatmap(
            values,
            xticklabels=labels,
            yticklabels=labels,
            cmap="PuBu",
            square=True,
            ax=axis,
            cbar_kws={"label": "Attention per source cell"},
        )
        axis.set_title(f"Cell-type communication, t={time_value:g}")
        axis.set_xlabel("Receiver")
        axis.set_ylabel("Sender")
        figure.tight_layout()
        path = output_dir / f"communication_t{time_value:g}.pdf"
        figure.savefig(path, bbox_inches="tight", facecolor="white")
        plt.close(figure)
        outputs.append(path)
    return outputs


def _select_lr_pair(table: pd.DataFrame, requested: str) -> str:
    pairs = table["pair"].astype(str)
    matches = pairs.loc[pairs.str.casefold() == str(requested).casefold()].unique()
    if len(matches) == 1:
        return str(matches[0])
    ligand, _, receptor = str(requested).partition("_")
    nearby = sorted(
        {
            pair
            for pair in pairs.unique()
            if ligand.casefold() in pair.casefold()
            or receptor.casefold() in pair.casefold()
        }
    )[:30]
    raise KeyError(
        f"LR pair {requested!r} was not scored. Related scored pairs: {nearby}"
    )


def _plot_target_lr(
    pair_timecourse: pd.DataFrame,
    celltype_timecourse: pd.DataFrame,
    *,
    pair: str,
    output_dir: Path,
) -> list[Path]:
    import matplotlib.pyplot as plt
    import seaborn as sns

    pair_table = pair_timecourse.loc[pair_timecourse["pair"] == pair].copy()
    cell_table = celltype_timecourse.loc[celltype_timecourse["pair"] == pair].copy()
    pair_path = output_dir / f"{pair}_timecourse.csv"
    cell_path = output_dir / f"{pair}_celltype_timecourse.csv"
    pair_table.to_csv(pair_path, index=False)
    cell_table.to_csv(cell_path, index=False)

    figure, axes = plt.subplots(1, 2, figsize=(13.0, 4.8), facecolor="white")
    ordered = pair_table.sort_values("time")
    axes[0].plot(
        ordered["time"], ordered["score"], marker="o", linewidth=2.2, color="#2b6cb0"
    )
    axes[0].set_title(f"{pair}: total LR communication")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("LR score")
    axes[0].grid(axis="y", alpha=0.2)

    incoming = cell_table.pivot_table(
        index="cell_type",
        columns="time",
        values="incoming",
        aggfunc="sum",
        fill_value=0.0,
    )
    incoming = incoming.loc[incoming.max(axis=1).sort_values(ascending=False).index]
    sns.heatmap(
        incoming,
        cmap="magma",
        ax=axes[1],
        cbar_kws={"label": "Incoming LR score"},
    )
    axes[1].set_title(f"{pair}: receiving cell types")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Cell type")
    figure.tight_layout()
    figure_path = output_dir / f"{pair}_communication_panel.pdf"
    figure.savefig(figure_path, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return [pair_path, cell_path, figure_path]


def _stage_communication(ctx: RunContext) -> dict[str, object]:
    if ctx.args.lr_database is None:
        raise ValueError("--lr-database is required for the communication stage")
    lr_database = _require_file(ctx.args.lr_database, "zebrafish LR database")
    max_cells = (
        int(ctx.args.smoke_n_samples)
        if ctx.args.profile == "smoke"
        else ctx.args.communication_max_cells
    )
    s25_state_index = _require_file(
        ctx.output_dir / "s25" / "generated_states" / "index.json",
        "S25 no-warp generated-state index",
    )
    upstream_s25_manifest = _require_current_stage_manifest(ctx, "s25")
    classifier_cache_value = upstream_s25_manifest.get("details", {}).get(
        "classifier_cache_path"
    )
    if classifier_cache_value is None:
        raise RuntimeError(
            "The current S25 manifest does not record classifier_cache_path; "
            "rerun --stage s25,communication with the current runner."
        )
    classifier_cache = _require_file(
        classifier_cache_value, "S25 trajectory classifier cache"
    )
    communication_knn = int(ctx.args.communication_classifier_knn_neighbors)
    lr_expression_policy = str(ctx.args.lr_expression_time_policy)
    settings = {
        "time_points": list(HALF_TIMES),
        "states": (
            "observed integer frames plus unwarped interval-local one-sided "
            "half-time frames from the preceding observed anchor"
        ),
        "trajectory_scope": S25_COMMUNICATION_TRAJECTORY_SCOPE,
        "attention_matrix": "M_per_source",
        "remove_self_loop": False,
        "winsor_quantile": float(ctx.args.communication_winsor_quantile),
        "max_cells_per_timepoint": max_cells,
        "display_min_only": float(ctx.args.communication_display_min),
        "lr_database": str(lr_database),
        "lr_database_sha256": _sha256(lr_database),
        "target_pair": ctx.args.lr_pair,
        "s25_state_index": str(s25_state_index),
        "s25_state_index_sha256": _sha256(s25_state_index),
        "s25_stage_signature": upstream_s25_manifest.get("signature"),
        "generated_label_classifier": {
            "policy": (
                "explicitly re-predict every interval-local generated frame; never "
                "inherit display labels from the state bundle"
            ),
            "knn_neighbors": communication_knn,
            "default_semantics": (
                "k=10 is the production default; k=1/5/20/50 are explicit "
                "reviewer sensitivity conditions"
            ),
            "cache_path": str(classifier_cache),
            "cache_sha256": _sha256(classifier_cache),
            "observed_label_policy": "actual annotation",
        },
        "expression_space": (
            "arithmetic cell-type mean after per-cell conversion to normalized "
            "count-like abundance; not raw counts"
        ),
        "primary_lr_expression_time_policy": lr_expression_policy,
        "primary_lr_expression_contract": (
            "all times use per-cell clipped expm1(inverse-PCA log1p), including "
            "observed integer-state PCA coordinates"
            if lr_expression_policy == "all_inverse_pca"
            else "legacy hybrid: exact observed expression at integer times and "
            "inverse-PCA expression at generated half-times"
        ),
        "lr_expression_validation": (
            "both all-inverse-PCA and hybrid exact-observed projections are always "
            "exported; their difference isolates the measurement operator"
        ),
        "lr_feature_policy": (
            "one active-loading PCA feature universe across all observed/generated "
            "times; center-only subunits excluded globally"
        ),
        "complex_mode": str(ctx.args.lr_complex_mode),
        "complex_require_all_subunits": True,
        "preferred_species_tag": ctx.args.preferred_species_tag,
        "lr_n_clusters": int(ctx.args.lr_n_clusters),
        "random_seed": int(ctx.args.random_seed),
    }

    def action(stage_dir: Path):
        generated, generated_times, _ = _read_state_bundle(
            ctx.output_dir / "s25" / "generated_states",
            annotation_key=ctx.args.annotation_key,
        )
        if generated_times != list(HALF_TIMES):
            raise RuntimeError(
                "S25 generated-state time grid does not match the communication grid; "
                "rerun --stage s25,communication"
            )
        observed = _observed_state_dict(ctx)
        cached_classifier = cb.tl.load_cached_mlp_classifier(
            str(classifier_cache), device=ctx.args.device
        )
        expected_feature_dim = int(_main_classifier_settings(ctx)["n_joint_features"])
        if int(cached_classifier.feature_dim) != expected_feature_dim:
            raise RuntimeError(
                "Classifier feature contract differs from the zebrafish joint-state "
                f"contract: cache={cached_classifier.feature_dim}, "
                f"expected={expected_feature_dim}."
            )
        if not bool(cached_classifier.include_time_feature):
            raise RuntimeError("Communication classifier cache must include time.")
        if str(cached_classifier.label_col) != str(ctx.args.annotation_key):
            raise RuntimeError(
                "Communication classifier annotation mismatch: "
                f"cache={cached_classifier.label_col!r}, "
                f"requested={ctx.args.annotation_key!r}."
            )
        (
            hybrid,
            source_by_time,
            label_assignments,
            label_summary,
            label_counts,
        ) = _build_explicitly_labeled_hybrid_states(
            generated,
            observed,
            time_points=HALF_TIMES,
            annotation_key=ctx.args.annotation_key,
            cached_classifier=cached_classifier,
            classifier_feature_dim=int(cached_classifier.feature_dim),
            device=ctx.args.device,
            knn_neighbors=communication_knn,
        )
        outputs = _write_state_bundle(
            hybrid,
            HALF_TIMES,
            stage_dir / "hybrid_states",
            annotation_key=ctx.args.annotation_key,
            source_by_time=source_by_time,
        )
        for filename, table in (
            ("generated_label_assignments.csv", label_assignments),
            ("generated_label_reclassification_summary.csv", label_summary),
            ("communication_label_counts.csv", label_counts),
        ):
            path = stage_dir / filename
            table.to_csv(path, index=False)
            outputs.append(path)
        attention_dir = stage_dir / "attention"
        communication_pickle = stage_dir / "communications.pkl"
        communications = cb.tl.compute_timepoint_communications(
            adata_dict=hybrid,
            time_points=HALF_TIMES,
            annotation_key=ctx.args.annotation_key,
            f_net=ctx.runtime.f_net,
            device=ctx.args.device,
            out_dir=str(attention_dir),
            save_dense_attention_matrix=False,
            remove_self_loop=False,
            winsor_quantile=float(ctx.args.communication_winsor_quantile),
            save_pickle_path=str(communication_pickle),
            max_cells_per_timepoint=max_cells,
            random_seed=int(ctx.args.random_seed),
        )
        outputs.extend([communication_pickle, *sorted(attention_dir.glob("*"))])
        outputs.extend(
            _plot_communication_heatmaps(
                communications,
                stage_dir,
                selected_times=(0.0, 2.0, 4.0),
                display_min=float(ctx.args.communication_display_min),
            )
        )

        lr_common = {
            "time_points": HALF_TIMES,
            "annotation_key": ctx.args.annotation_key,
            "matrix_key": "M_per_source",
            "spatial_dim": 2,
            "loadings_key": "PCs",
            "reference_layer": None,
            "expression_space": "count",
            "complex_mode": str(ctx.args.lr_complex_mode),
            "require_all_subunits": True,
            "duplicate_policy": "first",
            "preferred_species_tag": ctx.args.preferred_species_tag,
            "n_clusters": int(ctx.args.lr_n_clusters),
            "profile_linkage_method": "average",
            "profile_cluster_order": "dendrogram",
        }
        hybrid_lr = cb.tl.project_communication_to_lr_timecourses(
            hybrid,
            ctx.adata,
            communications,
            lr_database,
            **lr_common,
            observed_adata=ctx.adata,
            observed_time_key=ctx.args.time_key,
            observed_time_points=OBSERVED_TIMES,
            observed_annotation_key=ctx.args.annotation_key,
            observed_layer=None,
            observed_expression_space="log1p",
            observed_missing_time_policy="error",
        )
        all_inverse_lr = cb.tl.project_communication_to_lr_timecourses(
            hybrid,
            ctx.adata,
            communications,
            lr_database,
            **lr_common,
        )
        primary_lr = (
            all_inverse_lr if lr_expression_policy == "all_inverse_pca" else hybrid_lr
        )
        (
            lr_measurement_pair,
            lr_measurement_celltype,
            lr_measurement_metrics,
            lr_anchor_continuity,
            lr_anchor_continuity_metrics,
        ) = _compare_lr_measurement_contracts(
            hybrid_lr,
            all_inverse_lr,
            observed_times=OBSERVED_TIMES,
        )
        tables = {
            "lr_pair_timecourse.csv": primary_lr.pair_timecourse,
            "lr_celltype_timecourse.csv": primary_lr.celltype_timecourse,
            "lr_pattern_summary.csv": primary_lr.pattern_summary,
            "lr_coverage.csv": primary_lr.coverage,
            "lr_normalized_profiles.csv": primary_lr.clustering.normalized_profiles,
            "lr_cluster_assignments.csv": primary_lr.clustering.assignments,
            "lr_cluster_prototypes.csv": primary_lr.clustering.prototypes,
            "lr_cluster_diagnostics.csv": primary_lr.clustering.diagnostics,
            "lr_all_inverse_pca_pair_timecourse.csv": (all_inverse_lr.pair_timecourse),
            "lr_all_inverse_pca_celltype_timecourse.csv": (
                all_inverse_lr.celltype_timecourse
            ),
            "lr_all_inverse_pca_pattern_summary.csv": all_inverse_lr.pattern_summary,
            "lr_all_inverse_pca_cluster_assignments.csv": (
                all_inverse_lr.clustering.assignments
            ),
            "lr_hybrid_exact_observed_pair_timecourse.csv": (hybrid_lr.pair_timecourse),
            "lr_hybrid_exact_observed_celltype_timecourse.csv": (
                hybrid_lr.celltype_timecourse
            ),
            "lr_hybrid_exact_observed_pattern_summary.csv": hybrid_lr.pattern_summary,
            "lr_hybrid_exact_observed_cluster_assignments.csv": (
                hybrid_lr.clustering.assignments
            ),
            "lr_hybrid_vs_all_inverse_pair_scores.csv": lr_measurement_pair,
            "lr_hybrid_vs_all_inverse_celltype_scores.csv": (lr_measurement_celltype),
            "lr_observed_vs_inverse_pca_metrics.csv": lr_measurement_metrics,
            "lr_anchor_source_switch_diagnostics.csv": lr_anchor_continuity,
            "lr_anchor_source_switch_metrics.csv": lr_anchor_continuity_metrics,
        }
        for filename, table in tables.items():
            path = stage_dir / filename
            table.to_csv(
                path,
                index=filename in {"lr_normalized_profiles.csv"},
            )
            outputs.append(path)
        target_pair = _select_lr_pair(primary_lr.pair_timecourse, ctx.args.lr_pair)
        outputs.extend(
            _plot_target_lr(
                primary_lr.pair_timecourse,
                primary_lr.celltype_timecourse,
                pair=target_pair,
                output_dir=stage_dir,
            )
        )
        settings_path = stage_dir / "lr_settings.json"
        settings_path.write_text(
            json.dumps(_json_ready(primary_lr.settings), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        outputs.append(settings_path)
        all_inverse_settings_path = stage_dir / "lr_all_inverse_pca_settings.json"
        all_inverse_settings_path.write_text(
            json.dumps(_json_ready(all_inverse_lr.settings), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        outputs.append(all_inverse_settings_path)
        hybrid_settings_path = stage_dir / "lr_hybrid_exact_observed_settings.json"
        hybrid_settings_path.write_text(
            json.dumps(_json_ready(hybrid_lr.settings), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        outputs.append(hybrid_settings_path)
        return outputs, {
            "target_pair_resolved": target_pair,
            "primary_lr_expression_time_policy": lr_expression_policy,
            "n_lr_pairs_scored": int(primary_lr.pair_timecourse["pair"].nunique()),
            "state_sources": source_by_time,
            "coverage": primary_lr.coverage.to_dict(orient="records"),
            "generated_label_classifier": {
                "knn_neighbors": communication_knn,
                "cache_path": str(classifier_cache),
                "cache_sha256": _sha256(classifier_cache),
                "feature_dim": int(cached_classifier.feature_dim),
                "feature_cols": list(cached_classifier.feature_cols),
                "label_col": str(cached_classifier.label_col),
                "include_time_feature": bool(cached_classifier.include_time_feature),
                "source_fingerprint": cached_classifier.metadata.get("source", {}).get(
                    "fingerprint"
                ),
            },
            "generated_label_reclassification": label_summary.to_dict(orient="records"),
            "communication_label_counts": label_counts.to_dict(orient="records"),
            "lr_measurement_contract_validation": (
                lr_measurement_metrics.to_dict(orient="records")
            ),
        }

    return _execute_stage(ctx, "communication", settings, action)


def _build_context(args: argparse.Namespace) -> RunContext:
    aligned_h5ad_entry = _lexical_absolute_path(args.aligned_h5ad)
    model_dir_entry = _lexical_absolute_path(args.model_dir)
    acceptance_binding = _build_matched_acceptance_binding(
        args,
        aligned_h5ad=aligned_h5ad_entry,
        model_dir=model_dir_entry,
    )
    args.aligned_h5ad = _require_file(args.aligned_h5ad, "aligned zebrafish H5AD")
    args.model_dir = _require_dir(args.model_dir, "native six-stage model directory")
    args.output_dir = Path(args.output_dir).expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    adata = sc.read_h5ad(args.aligned_h5ad)
    if args.annotation_key not in adata.obs:
        if "bin_annotation" in adata.obs:
            adata.obs[args.annotation_key] = adata.obs["bin_annotation"].astype(str)
        else:
            raise KeyError(
                f"Aligned H5AD lacks obs['{args.annotation_key}'] and 'bin_annotation'"
            )
    if args.time_key not in adata.obs:
        raise KeyError(f"Aligned H5AD lacks obs['{args.time_key}']")
    joint = _joint_features(
        adata, latent_key=args.latent_key, spatial_key=args.spatial_key
    )
    df, resolved_time_key = cb.tl.adata_to_aligned_dataframe(
        adata,
        time_key=args.time_key,
        obsm_key=args.latent_key,
        spatial_key=args.spatial_key,
        concat_spatial=True,
        annotation_key=args.annotation_key,
    )
    if resolved_time_key != args.time_key:
        raise RuntimeError(
            f"Requested time key {args.time_key!r} resolved as {resolved_time_key!r}"
        )
    observed = sorted(float(value) for value in df["samples"].unique())
    if observed != list(OBSERVED_TIMES):
        raise ValueError(
            f"Expected clean zebrafish model times {list(OBSERVED_TIMES)}, got {observed}"
        )
    loaded = cb.tl.load_dynamical_model_from_dir(
        args.model_dir, dim=int(joint.shape[1]), device=args.device
    )
    loaded.model.eval()
    runtime = cb.tl.build_dynamical_runtime(loaded)
    model_config_path = _require_file(
        args.model_dir / "config.yaml", "resolved model config"
    )
    resolved_interaction_config = loaded.config.get("model", {}).get(
        "interaction_net", {}
    )
    raw_edge_predictor = resolved_interaction_config.get("edge_predictor_path")
    edge_predictor_path = None
    if raw_edge_predictor:
        candidate = Path(str(raw_edge_predictor)).expanduser()
        raw_edge_root = resolved_interaction_config.get("edge_predictor_root")
        if not candidate.is_absolute() and raw_edge_root is not None:
            candidate = Path(str(raw_edge_root)).expanduser() / candidate
        edge_predictor_path = (
            candidate.resolve()
            if candidate.is_absolute()
            else (Path.cwd() / candidate).resolve()
        )
    if args.shared_cache_dir is None:
        # .../RUN/preprocess/zebrafish_aligned.h5ad -> .../RUN/shared_downstream_cache
        shared_cache_dir = args.aligned_h5ad.parent.parent / "shared_downstream_cache"
    else:
        shared_cache_dir = Path(args.shared_cache_dir).expanduser().resolve()
    shared_cache_dir.mkdir(parents=True, exist_ok=True)
    label_to_color = _label_colors(
        adata,
        annotation_key=args.annotation_key,
        color_key=args.color_key,
    )
    signature_sources = (
        Path(__file__).resolve(),
        *(REPO_ROOT / path for path in IMPLEMENTATION_SOURCE_RELATIVE_PATHS),
    )
    common_signature = {
        MATCHED_ACCEPTANCE_KEY: _json_ready(acceptance_binding),
        "aligned_h5ad": str(args.aligned_h5ad),
        "aligned_h5ad_matched_entry": str(aligned_h5ad_entry),
        "aligned_h5ad_sha256": _sha256(args.aligned_h5ad),
        "model_dir": str(args.model_dir),
        "model_dir_matched_entry": str(model_dir_entry),
        "weight_stage": loaded.weight_stage,
        "weight_path": str(loaded.weight_path),
        "weight_sha256": _sha256(loaded.weight_path),
        "score_stage": loaded.score_stage,
        "score_path": None if loaded.score_path is None else str(loaded.score_path),
        "score_sha256": (
            None if loaded.score_path is None else _sha256(loaded.score_path)
        ),
        "dim": int(joint.shape[1]),
        "spatial_dim": 2,
        "random_seed": int(args.random_seed),
        "device": str(args.device),
        "profile": str(args.profile),
        "shared_cache_dir": str(shared_cache_dir),
        "data_contract": {
            "annotation_key": str(args.annotation_key),
            "color_key": None if args.color_key is None else str(args.color_key),
            "time_key": str(args.time_key),
            "latent_key": str(args.latent_key),
            "spatial_key": str(args.spatial_key),
        },
        "model_config_path": str(model_config_path),
        "model_config_sha256": _sha256(model_config_path),
        "resolved_model_config_sha256": _stable_hash(loaded.config),
        "edge_predictor_path": (
            None if edge_predictor_path is None else str(edge_predictor_path)
        ),
        "edge_predictor_sha256": (
            None
            if edge_predictor_path is None
            else _sha256(_require_file(edge_predictor_path, "edge predictor"))
        ),
        "git": _git_revision(),
        "runner_path": str(Path(__file__).resolve()),
        "runner_sha256": _sha256(Path(__file__).resolve()),
        "implementation_sha256": {
            str(path.relative_to(REPO_ROOT)): _sha256(path)
            for path in signature_sources
        },
        "implementation_file_count": int(len(signature_sources)),
    }
    return RunContext(
        args=args,
        adata=adata,
        df=df,
        loaded=loaded,
        runtime=runtime,
        dim=int(joint.shape[1]),
        spatial_dim=2,
        output_dir=args.output_dir,
        shared_cache_dir=shared_cache_dir,
        label_to_color=label_to_color,
        common_signature=common_signature,
    )


def _write_root_manifest(
    ctx: RunContext,
    selected_stages: Sequence[str],
    stage_manifests: Mapping[str, Mapping[str, object]],
) -> Path:
    current_acceptance = _context_acceptance_binding(ctx)
    training_defaults = ctx.loaded.config.get("training", {}).get("defaults", {})
    interaction_cfg = ctx.loaded.config.get("model", {}).get("interaction_net", {})
    complete_manifests: dict[str, Mapping[str, object]] = {}
    for name in ALL_STAGES:
        try:
            manifest = _require_current_stage_manifest(ctx, name)
        except (OSError, ValueError, RuntimeError, json.JSONDecodeError):
            continue
        complete_manifests[name] = manifest
    complete_manifests.update(stage_manifests)
    stage_signatures = {
        name: manifest.get("signature") for name, manifest in complete_manifests.items()
    }
    payload = {
        "schema_version": 1,
        "workflow": "zebrafish_native_paper_downstream",
        "signature": _stable_hash(
            {
                "workflow": "zebrafish_native_paper_downstream",
                "common": ctx.common_signature,
                "stage_signatures": stage_signatures,
            }
        ),
        "completed_at": _utc_now(),
        "selected_stages_this_invocation": list(selected_stages),
        "completed_stages": list(complete_manifests),
        "common": ctx.common_signature,
        MATCHED_ACCEPTANCE_KEY: _json_ready(current_acceptance),
        "profile": ctx.args.profile,
        "shared_cache_dir": str(ctx.shared_cache_dir),
        "gpu_assignment": {
            "requested_device": ctx.args.device,
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "CYTOBRIDGE_ASSIGNED_GPU": os.environ.get("CYTOBRIDGE_ASSIGNED_GPU"),
        },
        "model": {
            "alpha_spatial": training_defaults.get("alpha_spatial"),
            "alpha_express": training_defaults.get("alpha_express"),
            "sigma": training_defaults.get("sigma"),
            "interaction_cutoff": interaction_cfg.get("cutoff"),
            "edge_predictor_path": interaction_cfg.get("edge_predictor_path"),
            "edge_predictor_threshold": interaction_cfg.get("edge_predictor_thre"),
        },
        "preprocess": {
            "n_obs": int(ctx.adata.n_obs),
            "n_vars": int(ctx.adata.n_vars),
            "time_counts": {
                str(key): int(value)
                for key, value in ctx.adata.obs[ctx.args.time_key]
                .value_counts()
                .sort_index()
                .items()
            },
            "preprocess_info": ctx.adata.uns.get("preprocess_info", {}),
            "pca_center_info": ctx.adata.uns.get("pca_center_info", {}),
        },
        "stage_manifests": {
            name: str(_stage_manifest_path(ctx, name)) for name in complete_manifests
        },
        "stage_signatures": stage_signatures,
    }
    path = ctx.output_dir / "run_manifest.json"
    path.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True), encoding="utf-8"
    )
    return path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--aligned-h5ad", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument(
        "--acceptance-report",
        type=Path,
        default=None,
        help=(
            "Exact canonical matched-ablation acceptance JSON. Required for "
            "--profile full and optional only for smoke tests."
        ),
    )
    parser.add_argument(
        "--expected-acceptance-sha256",
        default=None,
        help=(
            "Required exact SHA-256 of --acceptance-report. The report must be "
            "overall PASS with datasets.zebrafish PASS."
        ),
    )
    parser.add_argument(
        "--lr-database",
        type=Path,
        default=None,
        help="Required only for the communication stage.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--stage",
        default="all",
        help=("'all' or a comma-separated subset of: " + ", ".join(ALL_STAGES)),
    )
    parser.add_argument("--profile", choices=("full", "smoke"), default="full")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--shared-cache-dir", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--annotation-key", default="Annotation")
    parser.add_argument("--color-key", default="Color")
    parser.add_argument("--time-key", default="time_point_processed")
    parser.add_argument("--latent-key", default="X_latent")
    parser.add_argument("--spatial-key", default="spatial_aligned")

    parser.add_argument("--interaction-m", type=int, default=1024)
    parser.add_argument("--sde-dt", type=float, default=0.05)
    parser.add_argument("--sde-sigma", type=float, default=0.03)
    parser.add_argument(
        "--growth-alpha",
        type=float,
        default=1.0,
        help=(
            "Split-SDE growth multiplier; 1.0 is the recovered historical "
            "zebrafish workflow value."
        ),
    )
    parser.add_argument(
        "--sde-n-samples",
        type=int,
        default=None,
        help=(
            "Optional per-observed-anchor cap for canonical interpolation; "
            "default uses every cell at each observed anchor."
        ),
    )
    parser.add_argument(
        "--sde-max-particles",
        type=int,
        default=100000,
        help=(
            "Fail-fast ceiling checked before split-event allocation. This is a "
            "safety guard, not a downsampling target."
        ),
    )
    parser.add_argument("--smoke-n-samples", type=int, default=64)
    parser.add_argument("--point-size", type=float, default=2.0)

    parser.add_argument("--classifier-epochs", type=int, default=500)
    parser.add_argument(
        "--s22-simulation-step",
        type=float,
        default=0.1,
        help=(
            "Output grid for the single S22 global-t0 fixed-population state "
            "transport. Mosaic and video frames are selected from this continuous "
            "generated path; S22 hard-codes growth_alpha=0 and requires constant N."
        ),
    )
    parser.add_argument("--video-step", type=float, default=0.1)
    parser.add_argument("--video-fps", type=int, default=10)
    parser.add_argument("--video-formats", default="gif,mp4")
    parser.add_argument("--velocity-neighbors", type=int, default=30)

    parser.add_argument("--ablation-step", type=float, default=0.05)
    parser.add_argument("--ablation-classifier-epochs", type=int, default=500)
    parser.add_argument("--ysl-label", default="Yolk Syncytial Layer")
    parser.add_argument("--evl-label", default="EVL")

    parser.add_argument("--s25-top-genes", type=int, default=250)
    parser.add_argument("--s25-n-clusters", type=int, default=4)
    parser.add_argument(
        "--s25-canonical-state-bundle",
        type=Path,
        default=None,
        help=(
            "Optional historical interval-local state bundle for S25 only (the "
            "legacy directory name is canonical_prewarp_states). The index, frame "
            "hashes, and adjacent complete historical S22 manifest are validated. "
            "The current global-t0 S22 bundle is intentionally incompatible."
        ),
    )
    parser.add_argument(
        "--s25-classifier-knn-neighbors",
        type=int,
        default=10,
        help=(
            "Spatial label-smoothing k used for every generated Zebrafish cell-type "
            "annotation, including the S25 target-cell analysis. Default: 10."
        ),
    )
    parser.add_argument("--preferred-species-tag", default=None)

    parser.add_argument("--communication-max-cells", type=int, default=None)
    parser.add_argument(
        "--communication-classifier-knn-neighbors",
        type=int,
        default=10,
        help=(
            "Generated-frame label policy for communication/LR. It must match the "
            "generated-cell annotation policy; default: 10. Observed annotations "
            "are unchanged."
        ),
    )
    parser.add_argument("--communication-winsor-quantile", type=float, default=0.995)
    parser.add_argument(
        "--communication-display-min",
        type=float,
        default=0.0,
        help="Display-only cutoff for communication heatmaps; raw matrices are unchanged.",
    )
    parser.add_argument("--lr-pair", default="cxcl12a_cxcr4a")
    parser.add_argument("--lr-n-clusters", type=int, default=4)
    parser.add_argument(
        "--lr-complex-mode",
        choices=("min", "geometric_mean", "mean", "product"),
        default="min",
        help=(
            "Aggregation across subunits of a heteromeric ligand or receptor. "
            "Use geometric_mean for the reviewer sensitivity run; min preserves "
            "the manuscript AND-gate definition."
        ),
    )
    parser.add_argument(
        "--lr-expression-time-policy",
        choices=("all_inverse_pca", "hybrid_exact_observed"),
        default="all_inverse_pca",
        help=(
            "Expression measurement used by the primary LR time course. The "
            "default all_inverse_pca applies one comparable decoder at every time; "
            "hybrid_exact_observed is retained only for legacy manuscript parity. "
            "Both projections are exported regardless of this selection."
        ),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    if int(args.sde_max_particles) <= 0:
        raise ValueError("--sde-max-particles must be > 0")
    if int(args.s25_classifier_knn_neighbors) <= 0:
        raise ValueError("--s25-classifier-knn-neighbors must be > 0")
    if int(args.communication_classifier_knn_neighbors) <= 0:
        raise ValueError("--communication-classifier-knn-neighbors must be > 0")
    _require_formal_acceptance_cli(args)
    selected_stages = _parse_stages(args.stage)
    if "communication" in selected_stages and "s25" not in selected_stages:
        state_index = (
            Path(args.output_dir).expanduser().resolve()
            / "s25"
            / "generated_states"
            / "index.json"
        )
        if not state_index.exists():
            raise FileNotFoundError(
                "The communication stage requires S25 no-warp generated states. "
                "Run --stage s25,communication (or --stage all) first."
            )
    ctx = _build_context(args)
    runners = {
        "classifier": _stage_classifier,
        "velocity": _stage_velocity,
        "s22": _stage_s22,
        "growth": _stage_growth,
        "ablation": _stage_ablation,
        "s25": _stage_s25,
        "communication": _stage_communication,
    }
    manifests: dict[str, dict[str, object]] = {}
    for stage in selected_stages:
        manifests[stage] = runners[stage](ctx)
    root_manifest = _write_root_manifest(ctx, selected_stages, manifests)
    print(
        json.dumps(
            {
                "workflow": "zebrafish_native_paper_downstream",
                "output_dir": str(ctx.output_dir),
                "root_manifest": str(root_manifest),
                "stages": {
                    stage: str(_stage_manifest_path(ctx, stage))
                    for stage in selected_stages
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
