#!/usr/bin/env python3
"""Prepare and launch the formal four-dataset matched-ablation matrix.

This maintainer tool is deliberately stricter than a convenience batch script.
It binds a clean release checkout, the twelve packaged training configurations,
one immutable aligned H5AD per dataset, the four learned edge predictors, their
metadata and graph provenance, and an explicit GPU for every condition.  It
never preprocesses data and never silently reuses an output directory.

``dry-run`` and ``prepare`` are the normal first steps.  ``launch-one`` starts
exactly one already-planned train-only or downstream-only command in a detached
monitor process; ``render`` prints shell-quoted commands without executing them.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import secrets
import shlex
import subprocess
import sys
from typing import Any


SCHEMA_VERSION = 1
PLAN_NAME = "matched_ablation_matrix_manifest.json"
PLAN_DIGEST_NAME = f"{PLAN_NAME}.sha256"
LAUNCHER_DIR_NAME = "_matched_launcher"
DATASET_ORDER = ("zebrafish", "mosta", "arista", "admouse")
ARM_ORDER = ("full", "no_lr_prior", "no_interaction")
MATCHED_PROTOCOL = "isolated-interaction-crn-v1"
MATCHED_SEED = 42

DATASET_SPECS: dict[str, dict[str, Any]] = {
    "zebrafish": {
        "aligned_sha256": "14753bbfdd05c9971b4ed5db4a7e70693479c7b7074ed1ef1d6f3187e1119811",
        "family": "zebrafish-full-no-lr-no-interaction-v1",
        "threshold": 0.6063615679740906,
        "cutoff": 0.09606367405591873,
        "graph_slices": 5,
        "configs": {
            "full": "zebrafish_spatial_full_alpha_express_0015.yaml",
            "no_lr_prior": (
                "zebrafish_spatial_full_alpha_express_0015_no_lr_prior.yaml"
            ),
            "no_interaction": (
                "zebrafish_spatial_full_alpha_express_0015_no_interaction.yaml"
            ),
        },
    },
    "mosta": {
        "aligned_sha256": "8b9ca0ad3475040235036548d54b96272bf6c49f057f6c2a643152c11350ce25",
        "family": "mosta-full-no-lr-no-interaction-v1",
        "threshold": 0.1192110925912857,
        "cutoff": 0.02400244047956264,
        "graph_slices": 4,
        "configs": {
            "full": "mosta_spatial_full_alpha_express_0015.yaml",
            "no_lr_prior": ("mosta_spatial_full_alpha_express_0015_no_lr_prior.yaml"),
            "no_interaction": (
                "mosta_spatial_full_alpha_express_0015_no_interaction.yaml"
            ),
        },
    },
    "arista": {
        "aligned_sha256": "eb72988986af42aeb8853c253d07218a9cb6294615eff55178fc0b409823205d",
        "family": "arista-full-no-lr-no-interaction-v1",
        "threshold": 0.5884028673171997,
        "cutoff": 0.03154105148551745,
        "graph_slices": 5,
        "configs": {
            "full": "arista_spatial_full.yaml",
            "no_lr_prior": "arista_spatial_full_no_lr_prior.yaml",
            "no_interaction": "arista_spatial_full_no_interaction.yaml",
        },
    },
    "admouse": {
        "aligned_sha256": "26d9a68acde90afc09d11b9c17de38525e37b1ee6b2e0290ddbda3efbe9ab968",
        "family": "admouse-full-no-lr-no-interaction-v1",
        "threshold": 0.9956824779510498,
        "cutoff": 0.012106042891492197,
        "graph_slices": 3,
        "configs": {
            "full": "admouse_spatial_full_alpha_express_0015.yaml",
            "no_lr_prior": ("admouse_spatial_full_alpha_express_0015_no_lr_prior.yaml"),
            "no_interaction": (
                "admouse_spatial_full_alpha_express_0015_no_interaction.yaml"
            ),
        },
    },
}


def profile_name(dataset: str, arm: str) -> str:
    """Return the exact condition directory understood by the validator."""

    if dataset not in DATASET_SPECS:
        raise ValueError(f"Unknown dataset: {dataset!r}")
    if arm not in ARM_ORDER:
        raise ValueError(f"Unknown matched arm: {arm!r}")
    return dataset if arm == "full" else f"{dataset}_{arm}"


PROFILE_ORDER = tuple(
    profile_name(dataset, arm) for dataset in DATASET_ORDER for arm in ARM_ORDER
)
PROFILE_TO_DATASET_ARM = {
    profile_name(dataset, arm): (dataset, arm)
    for dataset in DATASET_ORDER
    for arm in ARM_ORDER
}


def _canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolved_file(path: str | Path, *, label: str) -> Path:
    supplied = Path(path).expanduser()
    if not supplied.exists():
        raise FileNotFoundError(f"{label} does not exist: {supplied}")
    resolved = supplied.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"{label} is not a regular file: {resolved}")
    return resolved


def _resolved_directory(path: str | Path, *, label: str) -> Path:
    supplied = Path(path).expanduser()
    if not supplied.exists():
        raise FileNotFoundError(f"{label} does not exist: {supplied}")
    resolved = supplied.resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError(f"{label} is not a directory: {resolved}")
    return resolved


def _file_identity(path: str | Path, *, label: str) -> dict[str, Any]:
    resolved = _resolved_file(path, label=label)
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size_bytes": int(stat.st_size),
        "sha256": _sha256_file(resolved),
    }


def _aggregate_records(records: Sequence[Mapping[str, Any]]) -> str:
    projected = [
        {
            "path": str(record["path"]),
            "size_bytes": int(record["size_bytes"]),
            "sha256": str(record["sha256"]),
        }
        for record in records
    ]
    return _sha256_bytes(
        json.dumps(projected, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )


def _tree_identity(
    root: str | Path,
    *,
    label: str,
    require_adjacency_records: int | None = None,
    dataset: str | None = None,
) -> dict[str, Any]:
    resolved = _resolved_directory(root, label=label)
    records: list[dict[str, Any]] = []
    adjacency_records = 0
    for candidate in sorted(resolved.rglob("*")):
        if candidate.is_symlink():
            raise ValueError(
                f"{label} must be an immutable real-file tree, not contain "
                f"symlinked entries: {candidate}"
            )
        if candidate.is_dir():
            continue
        target = candidate.resolve(strict=True)
        if not target.is_file():
            raise ValueError(f"{label} contains a non-file entry: {candidate}")
        relative = candidate.relative_to(resolved).as_posix()
        if candidate.name.endswith("_adjacency_records"):
            adjacency_records += 1
        stat = target.stat()
        records.append(
            {
                "path": relative,
                "size_bytes": int(stat.st_size),
                "sha256": _sha256_file(target),
            }
        )
    if not records:
        raise ValueError(f"{label} is empty: {resolved}")
    if (
        require_adjacency_records is not None
        and adjacency_records != require_adjacency_records
    ):
        raise ValueError(
            f"{label} must contain exactly {require_adjacency_records} "
            f"*_adjacency_records files; found {adjacency_records}."
        )
    if dataset is not None:
        expected_adjacency_paths = {
            f"{dataset}_t{index}/{dataset}_t{index}_adjacency_records"
            for index in range(int(require_adjacency_records or 0))
        }
        actual_adjacency_paths = {
            str(record["path"])
            for record in records
            if str(record["path"]).endswith("_adjacency_records")
        }
        if actual_adjacency_paths != expected_adjacency_paths:
            raise ValueError(
                f"{label} does not contain the exact dataset-bound canonical "
                f"slice paths: expected {sorted(expected_adjacency_paths)}, "
                f"got {sorted(actual_adjacency_paths)}."
            )
    return {
        "path": str(resolved),
        "file_count": len(records),
        "adjacency_record_count": adjacency_records,
        "sha256": _aggregate_records(records),
        "files": records,
    }


def _release_tree_identity(
    release_root: Path,
    *,
    python_only: bool,
) -> dict[str, Any]:
    package_root = release_root / "CytoBridge"
    if not package_root.is_dir():
        raise FileNotFoundError(
            f"Release root has no CytoBridge package directory: {package_root}"
        )
    candidates = list(package_root.rglob("*"))
    if not python_only:
        candidates.extend(
            [
                release_root / "scripts" / "run_matched_ablation_matrix.py",
                release_root / "scripts" / "validate_corrected_de_novo_run.py",
            ]
        )
    records: list[dict[str, Any]] = []
    for candidate in sorted(set(candidates)):
        if candidate.is_dir():
            continue
        if "__pycache__" in candidate.parts or candidate.suffix in {".pyc", ".pyo"}:
            continue
        if python_only and candidate.suffix != ".py":
            continue
        if candidate.is_symlink():
            raise ValueError(
                "The immutable release payload must not contain symlinked package "
                f"files: {candidate}"
            )
        if not candidate.is_file():
            continue
        relative = candidate.relative_to(release_root).as_posix()
        stat = candidate.stat()
        records.append(
            {
                "path": relative,
                "size_bytes": int(stat.st_size),
                "sha256": _sha256_file(candidate),
            }
        )
    if not records:
        raise ValueError("Release identity contains no files.")
    return {
        "file_count": len(records),
        "sha256": _aggregate_records(records),
        "files": records,
    }


def _run_git(release_root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(release_root), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(
            f"git {' '.join(arguments)} failed for {release_root}: {detail}"
        )
    return completed.stdout.strip()


def _git_release_identity(
    release_root: Path,
    *,
    expected_commit: str,
) -> dict[str, Any]:
    if len(expected_commit) != 40 or any(
        character not in "0123456789abcdef" for character in expected_commit.lower()
    ):
        raise ValueError("--release-commit must be a complete 40-character SHA-1.")
    expected_commit = expected_commit.lower()
    top_level = Path(_run_git(release_root, "rev-parse", "--show-toplevel")).resolve(
        strict=True
    )
    if top_level != release_root:
        raise ValueError(
            f"--release-root must be the Git top level: got {release_root}, "
            f"top level is {top_level}."
        )
    current_commit = _run_git(release_root, "rev-parse", "HEAD").lower()
    if current_commit != expected_commit:
        raise ValueError(
            f"Release commit mismatch: expected {expected_commit}, got {current_commit}."
        )
    dirty = _run_git(release_root, "status", "--porcelain", "--untracked-files=all")
    if dirty:
        shown = "\n".join(dirty.splitlines()[:20])
        raise ValueError(
            "The formal matrix requires a clean immutable release checkout; "
            f"Git reports changes:\n{shown}"
        )
    return {
        "top_level": str(top_level),
        "commit": current_commit,
        "clean": True,
    }


def _parse_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as error:  # pragma: no cover - exercised by source env gate
        raise RuntimeError(
            "PyYAML is required to validate the formal training configurations."
        ) from error
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Training config must contain a mapping: {path}")
    return value


def _close(actual: Any, expected: float) -> bool:
    try:
        value = float(actual)
    except (TypeError, ValueError):
        return False
    return math.isfinite(value) and math.isclose(
        value, float(expected), rel_tol=1e-12, abs_tol=1e-15
    )


def _validate_training_config(
    path: Path,
    *,
    dataset: str,
    arm: str,
) -> dict[str, Any]:
    config = _parse_yaml(path)
    spec = DATASET_SPECS[dataset]
    declaration = config.get("matched_ablation")
    expected_declaration = {
        "schema_version": 1,
        "family": spec["family"],
        "dataset": dataset,
        "arm": arm,
        "protocol": MATCHED_PROTOCOL,
        "shared_seed": MATCHED_SEED,
        "interaction_grouping_seed_offset": 10_000,
        "input_contract": "exact-shared-aligned-h5ad",
        "implementation_contract": "exact-shared-training-code-sha256",
    }
    if declaration != expected_declaration:
        raise ValueError(
            f"Canonical declaration drift in {path.name}: "
            f"expected {expected_declaration}, got {declaration}."
        )
    if int(config.get("seed", -1)) != MATCHED_SEED:
        raise ValueError(f"{path.name} must use seed {MATCHED_SEED}.")
    model = config.get("model")
    if not isinstance(model, dict):
        raise ValueError(f"{path.name} has no model mapping.")
    components = {
        str(component).strip().lower() for component in model.get("components", [])
    }
    interaction = model.get("interaction_net")
    if arm == "no_interaction":
        forbidden = {
            key
            for key in (
                "interaction_net",
                "interaction_type",
                "interaction_group_size",
            )
            if key in model
        }
        if "interaction" in components or forbidden:
            raise ValueError(
                f"{path.name} must contain neither an interaction component nor "
                f"inert interaction fields; found {sorted(forbidden)}."
            )
    else:
        if "interaction" not in components or not isinstance(interaction, dict):
            raise ValueError(f"{path.name} must contain the GNN interaction component.")
        if str(model.get("interaction_type", "")).lower() != "gnn":
            raise ValueError(f"{path.name} must use interaction_type=gnn.")
        if not _close(interaction.get("cutoff"), float(spec["cutoff"])):
            raise ValueError(f"{path.name} interaction cutoff drifted.")
        mode = str(interaction.get("edge_prior_mode", "")).lower()
        if arm == "full":
            if mode != "learned":
                raise ValueError(f"{path.name} full arm must use learned edge prior.")
            if not _close(interaction.get("edge_predictor_thre"), spec["threshold"]):
                raise ValueError(f"{path.name} learned threshold drifted.")
            if not interaction.get("edge_predictor_path"):
                raise ValueError(f"{path.name} must retain a predictor placeholder.")
        else:
            if mode != "all_spatial":
                raise ValueError(f"{path.name} no-LR-prior arm must use all_spatial.")
            stale = {
                key: interaction.get(key)
                for key in ("edge_predictor_path", "edge_predictor_thre")
                if interaction.get(key) is not None
            }
            if stale:
                raise ValueError(
                    f"{path.name} contains inert predictor settings: {stale}."
                )
    defaults = config.get("training", {}).get("defaults", {})
    if defaults.get("score_energy_objective") != "velocity_score_cross_term":
        raise ValueError(
            f"{path.name} must explicitly lock score_energy_objective="
            "velocity_score_cross_term for the matched comparison."
        )
    return config


def _validate_canonical_configs(release_root: Path) -> dict[str, dict[str, Any]]:
    configs_root = release_root / "CytoBridge" / "configs"
    identities: dict[str, dict[str, Any]] = {}
    seen_names: set[str] = set()
    for dataset in DATASET_ORDER:
        for arm in ARM_ORDER:
            name = str(DATASET_SPECS[dataset]["configs"][arm])
            if name in seen_names:
                raise ValueError(f"A canonical config is reused by two arms: {name}")
            seen_names.add(name)
            path = _resolved_file(
                configs_root / name, label=f"{dataset}/{arm} training config"
            )
            _validate_training_config(path, dataset=dataset, arm=arm)
            identities[profile_name(dataset, arm)] = _file_identity(
                path, label=f"{dataset}/{arm} training config"
            )
    if len(identities) != 12:
        raise AssertionError("Internal matrix error: expected exactly 12 configs.")
    return identities


def _validate_workflow_presets(release_root: Path) -> dict[str, dict[str, Any]]:
    identities: dict[str, dict[str, Any]] = {}
    for dataset in DATASET_ORDER:
        path = _resolved_file(
            release_root / "CytoBridge" / "workflow_configs" / f"{dataset}.json",
            label=f"{dataset} workflow preset",
        )
        try:
            preset = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid workflow preset JSON: {path}") from error
        if not isinstance(preset, dict):
            raise ValueError(f"Workflow preset must contain an object: {path}")
        spec = DATASET_SPECS[dataset]
        scientific = preset.get("scientific", {})
        train = preset.get("train", {})
        exact = (
            preset.get("dataset", {}).get("name") == dataset
            and scientific.get("seed") == MATCHED_SEED
            and _close(scientific.get("alpha_spatial"), 10.0)
            and _close(scientific.get("alpha_express"), 0.015)
            and train.get("config") == spec["configs"]["full"]
            and train.get("requires_edge_predictor") is True
            and _close(train.get("interaction_cutoff"), spec["cutoff"])
            and _close(train.get("edge_predictor_threshold"), spec["threshold"])
        )
        if not exact:
            raise ValueError(
                f"{dataset} workflow preset drifted from the formal package "
                "seed/alpha/main-config/cutoff/learned-threshold contract."
            )
        identities[dataset] = _file_identity(path, label=f"{dataset} workflow preset")
    return identities


def _validate_predictor_metadata(path: Path, *, dataset: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"Unreadable predictor metadata for {dataset}: {path}"
        ) from error
    if not isinstance(value, dict):
        raise ValueError(f"Predictor metadata for {dataset} must be a JSON object.")
    expected = float(DATASET_SPECS[dataset]["threshold"])
    for key in ("edge_predictor_threshold", "edge_predictor_threshold_selected"):
        if not _close(value.get(key), expected):
            raise ValueError(
                f"Predictor metadata {path} does not bind {key} to the frozen "
                f"validation-selected threshold {expected:.17g}."
            )
    if str(value.get("selection_source", "")).strip().lower() != "validation":
        raise ValueError(
            f"Predictor metadata {path} must record selection_source='validation'."
        )
    if not _close(value.get("distance_threshold"), DATASET_SPECS[dataset]["cutoff"]):
        raise ValueError(
            f"Predictor metadata {path} has the wrong spatial distance threshold."
        )
    split = value.get("split")
    if not isinstance(split, dict) or split.get("strategy") != "node_disjoint_holdout":
        raise ValueError(
            f"Predictor metadata {path} must record a node-disjoint holdout split."
        )
    try:
        random_seed = int(value.get("random_seed", -1))
    except (TypeError, ValueError):
        random_seed = -1
    if random_seed != MATCHED_SEED:
        raise ValueError(f"Predictor metadata {path} must record seed {MATCHED_SEED}.")
    return value


def _parse_assignments(
    values: Sequence[str],
    *,
    allowed_keys: Sequence[str],
    option: str,
) -> dict[str, str]:
    result: dict[str, str] = {}
    allowed = set(allowed_keys)
    for value in values:
        if "=" not in value:
            raise ValueError(f"{option} values must use KEY=VALUE; got {value!r}.")
        key, assigned = value.split("=", 1)
        key = key.strip()
        assigned = assigned.strip()
        if key not in allowed:
            raise ValueError(
                f"Unknown key {key!r} for {option}; expected {sorted(allowed)}."
            )
        if key in result:
            raise ValueError(f"Duplicate {option} assignment for {key!r}.")
        if not assigned:
            raise ValueError(f"Empty {option} assignment for {key!r}.")
        result[key] = assigned
    missing = allowed.difference(result)
    if missing:
        raise ValueError(f"Missing {option} assignments: {sorted(missing)}.")
    return result


def _gpu_assignments(values: Sequence[str]) -> dict[str, int]:
    raw = _parse_assignments(values, allowed_keys=PROFILE_ORDER, option="--gpu")
    result: dict[str, int] = {}
    for profile, value in raw.items():
        try:
            gpu = int(value)
        except ValueError as error:
            raise ValueError(
                f"GPU for {profile} must be a non-negative CUDA index; got {value!r}."
            ) from error
        if gpu < 0 or str(gpu) != value:
            raise ValueError(
                f"GPU for {profile} must be a canonical non-negative integer; "
                f"got {value!r}."
            )
        result[profile] = gpu
    return result


def _path_is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _bound_launcher_identity(release_root: Path) -> dict[str, Any]:
    """Bind the running launcher byte-for-byte to the selected release tree."""

    actual = Path(__file__).resolve(strict=True)
    expected = (release_root / "scripts" / "run_matched_ablation_matrix.py").resolve(
        strict=True
    )
    if actual != expected:
        raise ValueError(
            "The running launcher is not the launcher inside --release-root: "
            f"running={actual}, expected={expected}. Execute the release-owned "
            "script directly; launcher A may not drive release B."
        )
    return _file_identity(actual, label="release-owned matched matrix launcher")


def _command_record(
    argv: Sequence[str],
    *,
    release_root: Path,
    run_root: Path,
    profile: str,
    physical_gpu: int,
) -> dict[str, Any]:
    cache_root = run_root / LAUNCHER_DIR_NAME / "cache" / profile
    environment = {
        "PYTHONPATH": str(release_root),
        "CUDA_VISIBLE_DEVICES": str(physical_gpu),
        "CYTOBRIDGE_ASSIGNED_GPU": str(physical_gpu),
        "NUMBA_CACHE_DIR": str(cache_root / "numba"),
        "MPLCONFIGDIR": str(cache_root / "matplotlib"),
        "XDG_CACHE_HOME": str(cache_root / "xdg"),
        "PYTHONUNBUFFERED": "1",
        "JUPYTER_PLATFORM_DIRS": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONHASHSEED": str(MATCHED_SEED),
    }
    rendered = shlex.join(
        ["env", *(f"{key}={value}" for key, value in environment.items()), *argv]
    )
    return {
        "argv": list(argv),
        "cwd": str(release_root),
        "environment": environment,
        "shell": rendered,
    }


def _training_argv(
    *,
    python: Path,
    dataset: str,
    arm: str,
    config_path: Path,
    aligned_path: Path,
    profile_root: Path,
    predictor_path: Path | None,
) -> list[str]:
    argv = [
        str(python),
        "-m",
        "CytoBridge.cli",
        "workflow",
        "--config",
        dataset,
        "--step",
        "train",
        "--train",
        "--aligned-h5ad",
        str(aligned_path),
        "--training-config",
        str(config_path),
        "--output-dir",
        str(profile_root),
        "--device",
        "cuda:0",
    ]
    if arm == "full":
        if predictor_path is None:
            raise AssertionError("Internal matrix error: full arm lacks predictor.")
        argv.extend(
            [
                "--edge-predictor-path",
                str(predictor_path),
                "--edge-predictor-threshold",
                f"{float(DATASET_SPECS[dataset]['threshold']):.17g}",
            ]
        )
    elif predictor_path is not None:
        raise AssertionError("Internal matrix error: ablation received predictor.")
    return argv


def _downstream_argv(
    *,
    python: Path,
    dataset: str,
    arm: str,
    config_path: Path,
    aligned_path: Path,
    profile_root: Path,
    predictor_path: Path | None,
) -> list[str]:
    argv = [
        str(python),
        "-m",
        "CytoBridge.cli",
        "workflow",
        "--config",
        dataset,
        "--step",
        "downstream",
        "--aligned-h5ad",
        str(aligned_path),
        "--model-dir",
        str(profile_root / "training"),
        "--training-config",
        str(config_path),
        "--output-dir",
        str(profile_root),
        "--device",
        "cuda:0",
    ]
    if arm == "full":
        if predictor_path is None:
            raise AssertionError("Internal matrix error: full arm lacks predictor.")
        argv.extend(
            [
                "--edge-predictor-path",
                str(predictor_path),
                "--edge-predictor-threshold",
                f"{float(DATASET_SPECS[dataset]['threshold']):.17g}",
            ]
        )
    return argv


def _validator_argv(*, python: Path, release_root: Path, run_root: Path) -> list[str]:
    argv = [
        str(python),
        str(release_root / "scripts" / "validate_corrected_de_novo_run.py"),
        "--run-root",
        str(run_root),
        "--datasets",
        *PROFILE_ORDER,
        "--report",
        str(run_root / "matched_ablation_acceptance.json"),
    ]
    for dataset in DATASET_ORDER:
        argv.extend(["--matched-family", dataset])
    return argv


def build_plan(
    *,
    run_root: str | Path,
    release_root: str | Path,
    release_commit: str,
    python_executable: str | Path,
    aligned_h5ad: Mapping[str, str | Path],
    edge_predictor: Mapping[str, str | Path],
    input_graph_dir: Mapping[str, str | Path],
    gpu_by_profile: Mapping[str, int],
    git_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build and fully validate a non-mutating twelve-arm execution plan."""

    release = _resolved_directory(release_root, label="release root")
    output = Path(run_root).expanduser().resolve(strict=False)
    if _path_is_within(output, release) or _path_is_within(release, output):
        raise ValueError(
            "The output root and immutable release checkout must be disjoint."
        )
    if output.parent == output:
        raise ValueError("Refusing to use a filesystem root as --run-root.")
    python = _resolved_file(python_executable, label="Python executable")
    if not os.access(python, os.X_OK):
        raise PermissionError(f"Python executable is not executable: {python}")
    if set(aligned_h5ad) != set(DATASET_ORDER):
        raise ValueError("aligned_h5ad must contain exactly the four formal datasets.")
    if set(edge_predictor) != set(DATASET_ORDER):
        raise ValueError(
            "edge_predictor must contain exactly the four formal datasets."
        )
    if set(input_graph_dir) != set(DATASET_ORDER):
        raise ValueError(
            "input_graph_dir must contain exactly the four formal datasets."
        )
    if set(gpu_by_profile) != set(PROFILE_ORDER):
        raise ValueError("gpu_by_profile must contain exactly the twelve profiles.")
    for profile, gpu in gpu_by_profile.items():
        if isinstance(gpu, bool) or not isinstance(gpu, int) or gpu < 0:
            raise ValueError(f"Invalid GPU index for {profile}: {gpu!r}")

    actual_git = (
        dict(git_identity)
        if git_identity is not None
        else _git_release_identity(release, expected_commit=release_commit)
    )
    if (
        str(actual_git.get("commit", "")).lower() != release_commit.lower()
        or actual_git.get("clean") is not True
        or Path(str(actual_git.get("top_level", ""))).resolve(strict=False) != release
    ):
        raise ValueError("Injected or discovered Git release identity is not exact.")

    launcher_identity = _bound_launcher_identity(release)
    configs = _validate_canonical_configs(release)
    workflow_presets = _validate_workflow_presets(release)
    sources: dict[str, dict[str, Any]] = {}
    for dataset in DATASET_ORDER:
        aligned_identity = _file_identity(
            aligned_h5ad[dataset], label=f"{dataset} aligned H5AD"
        )
        expected_aligned_sha = str(DATASET_SPECS[dataset]["aligned_sha256"])
        if aligned_identity["sha256"] != expected_aligned_sha:
            raise ValueError(
                f"{dataset} aligned H5AD is not the exact accepted package input: "
                f"expected SHA-256 {expected_aligned_sha}, got "
                f"{aligned_identity['sha256']}."
            )
        predictor_identity = _file_identity(
            edge_predictor[dataset], label=f"{dataset} learned edge predictor"
        )
        predictor_path = Path(predictor_identity["path"])
        metadata_path = predictor_path.with_suffix(predictor_path.suffix + ".meta.json")
        metadata_identity = _file_identity(
            metadata_path, label=f"{dataset} edge predictor metadata"
        )
        _validate_predictor_metadata(metadata_path, dataset=dataset)
        graph_identity = _tree_identity(
            input_graph_dir[dataset],
            label=f"{dataset} learned-prior input graph",
            require_adjacency_records=int(DATASET_SPECS[dataset]["graph_slices"]),
            dataset=dataset,
        )
        sources[dataset] = {
            "aligned_h5ad": aligned_identity,
            "edge_predictor": predictor_identity,
            "edge_predictor_metadata": metadata_identity,
            "input_graph": graph_identity,
            "edge_predictor_threshold": float(DATASET_SPECS[dataset]["threshold"]),
            "threshold_source": "validation",
        }

    code_identity = _release_tree_identity(release, python_only=True)
    payload_identity = _release_tree_identity(release, python_only=False)
    arms: dict[str, dict[str, Any]] = {}
    for profile in PROFILE_ORDER:
        dataset, arm = PROFILE_TO_DATASET_ARM[profile]
        profile_root = output / profile
        aligned_path = profile_root / "preprocess" / f"{dataset}_aligned.h5ad"
        predictor_path = (
            profile_root / "preprocess" / "edge_classifier" / f"{dataset}_edge_model.pt"
            if arm == "full"
            else None
        )
        config_path = Path(configs[profile]["path"])
        train = _training_argv(
            python=python,
            dataset=dataset,
            arm=arm,
            config_path=config_path,
            aligned_path=aligned_path,
            profile_root=profile_root,
            predictor_path=predictor_path,
        )
        downstream = _downstream_argv(
            python=python,
            dataset=dataset,
            arm=arm,
            config_path=config_path,
            aligned_path=aligned_path,
            profile_root=profile_root,
            predictor_path=predictor_path,
        )
        arms[profile] = {
            "dataset": dataset,
            "arm": arm,
            "family": DATASET_SPECS[dataset]["family"],
            "protocol": MATCHED_PROTOCOL,
            "shared_seed": MATCHED_SEED,
            "gpu": int(gpu_by_profile[profile]),
            "device": "cuda:0",
            "training_config": configs[profile],
            "workflow_preset": workflow_presets[dataset],
            "paths": {
                "condition_root": str(profile_root),
                "aligned_h5ad": str(aligned_path),
                "training": str(profile_root / "training"),
                "downstream": str(profile_root / "downstream"),
                "edge_predictor": (
                    None if predictor_path is None else str(predictor_path)
                ),
                "input_graph": (
                    str(profile_root / "preprocess" / "input_graph")
                    if arm == "full"
                    else None
                ),
                "cache_root": str(output / LAUNCHER_DIR_NAME / "cache" / profile),
            },
            "commands": {
                "train": _command_record(
                    train,
                    release_root=release,
                    run_root=output,
                    profile=profile,
                    physical_gpu=int(gpu_by_profile[profile]),
                ),
                "downstream": _command_record(
                    downstream,
                    release_root=release,
                    run_root=output,
                    profile=profile,
                    physical_gpu=int(gpu_by_profile[profile]),
                ),
            },
        }

    plan = {
        "schema_version": SCHEMA_VERSION,
        "kind": "cytobridge-four-dataset-matched-ablation-matrix",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_root": str(output),
        "release": {
            "root": str(release),
            "commit": release_commit.lower(),
            "git": actual_git,
            "training_code": code_identity,
            "package_payload": payload_identity,
            "launcher": launcher_identity,
            "python_executable": _file_identity(python, label="Python executable"),
        },
        "matrix": {
            "datasets": list(DATASET_ORDER),
            "arms": list(ARM_ORDER),
            "profiles": list(PROFILE_ORDER),
            "fit_count": 12,
            "protocol": MATCHED_PROTOCOL,
            "shared_seed": MATCHED_SEED,
            "input_contract": "one-exact-shared-aligned-h5ad-per-dataset",
            "launch_contract": "train-only-then-downstream-only",
        },
        "sources": sources,
        "conditions": arms,
        "validator": _command_record(
            _validator_argv(python=python, release_root=release, run_root=output),
            release_root=release,
            run_root=output,
            profile="validator",
            physical_gpu=0,
        ),
        "planner": {
            "python": sys.version,
            "platform": platform.platform(),
        },
    }
    if len(plan["conditions"]) != 12:
        raise AssertionError("Internal matrix error: plan does not have 12 conditions.")
    return plan


def _plan_digest(plan: Mapping[str, Any]) -> str:
    return _sha256_bytes(_canonical_json_bytes(plan))


def prepare_run_root(plan: Mapping[str, Any]) -> tuple[Path, str]:
    """Create a new run root exactly once; never merge with existing content."""

    root = Path(str(plan["run_root"]))
    if root.exists() or root.is_symlink():
        raise FileExistsError(
            f"Refusing to prepare an existing output root, even if empty: {root}"
        )
    if not root.parent.is_dir():
        raise FileNotFoundError(
            f"The run-root parent must already exist: {root.parent}"
        )
    root.mkdir(mode=0o750)
    launcher = root / LAUNCHER_DIR_NAME
    launcher.mkdir(mode=0o750)
    (launcher / "logs").mkdir(mode=0o750)
    (launcher / "status").mkdir(mode=0o750)
    cache = launcher / "cache"
    cache.mkdir(mode=0o750)
    for profile in (*PROFILE_ORDER, "validator"):
        profile_cache = cache / profile
        profile_cache.mkdir(mode=0o750)
        for name in ("numba", "matplotlib", "xdg"):
            (profile_cache / name).mkdir(mode=0o750)

    for profile in PROFILE_ORDER:
        condition = plan["conditions"][profile]
        dataset = str(condition["dataset"])
        arm = str(condition["arm"])
        profile_root = root / profile
        profile_root.mkdir(mode=0o750)
        preprocess = profile_root / "preprocess"
        preprocess.mkdir(mode=0o750)
        aligned_link = preprocess / f"{dataset}_aligned.h5ad"
        aligned_link.symlink_to(plan["sources"][dataset]["aligned_h5ad"]["path"])
        if arm == "full":
            edge_dir = preprocess / "edge_classifier"
            edge_dir.mkdir(mode=0o750)
            edge_link = edge_dir / f"{dataset}_edge_model.pt"
            edge_link.symlink_to(plan["sources"][dataset]["edge_predictor"]["path"])
            edge_meta_link = edge_link.with_suffix(edge_link.suffix + ".meta.json")
            edge_meta_link.symlink_to(
                plan["sources"][dataset]["edge_predictor_metadata"]["path"]
            )
            graph_link = preprocess / "input_graph"
            graph_link.symlink_to(
                plan["sources"][dataset]["input_graph"]["path"],
                target_is_directory=True,
            )

    payload = _canonical_json_bytes(plan)
    digest = _sha256_bytes(payload)
    manifest_path = launcher / PLAN_NAME
    digest_path = launcher / PLAN_DIGEST_NAME
    with manifest_path.open("xb") as handle:
        handle.write(payload)
    with digest_path.open("x", encoding="ascii") as handle:
        handle.write(digest + "\n")
    manifest_path.chmod(0o440)
    digest_path.chmod(0o440)
    return root, digest


def _load_prepared_plan(run_root: str | Path) -> tuple[Path, dict[str, Any], str]:
    root = _resolved_directory(run_root, label="prepared run root")
    manifest_path = root / LAUNCHER_DIR_NAME / PLAN_NAME
    digest_path = root / LAUNCHER_DIR_NAME / PLAN_DIGEST_NAME
    if not manifest_path.is_file() or not digest_path.is_file():
        raise FileNotFoundError(
            f"Prepared manifest or digest is missing under {root / LAUNCHER_DIR_NAME}."
        )
    payload = manifest_path.read_bytes()
    expected = digest_path.read_text(encoding="ascii").strip().lower()
    actual = _sha256_bytes(payload)
    if expected != actual:
        raise ValueError(
            f"Prepared manifest SHA-256 mismatch: expected {expected}, got {actual}."
        )
    plan = json.loads(payload)
    if not isinstance(plan, dict) or plan.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Prepared manifest has an unsupported schema.")
    if Path(str(plan.get("run_root"))).resolve(strict=False) != root:
        raise ValueError("Prepared manifest run_root does not match its location.")
    if tuple(plan.get("matrix", {}).get("profiles", ())) != PROFILE_ORDER:
        raise ValueError("Prepared manifest profile matrix drifted.")
    return root, plan, actual


def _identity_matches(record: Mapping[str, Any], *, label: str) -> None:
    actual = _file_identity(str(record["path"]), label=label)
    if actual != dict(record):
        raise ValueError(f"{label} changed after matrix preparation.")


def verify_prepared_run(
    run_root: str | Path,
    *,
    git_identity: Mapping[str, Any] | None = None,
) -> tuple[Path, dict[str, Any], str]:
    """Re-hash every immutable dependency before rendering or launching."""

    root, plan, digest = _load_prepared_plan(run_root)
    release = _resolved_directory(plan["release"]["root"], label="release root")
    expected_commit = str(plan["release"]["commit"])
    current_git = (
        dict(git_identity)
        if git_identity is not None
        else _git_release_identity(release, expected_commit=expected_commit)
    )
    if current_git != plan["release"]["git"]:
        raise ValueError("Git release identity changed after matrix preparation.")
    if (
        _release_tree_identity(release, python_only=True)
        != plan["release"]["training_code"]
    ):
        raise ValueError("Release training code changed after matrix preparation.")
    if (
        _release_tree_identity(release, python_only=False)
        != plan["release"]["package_payload"]
    ):
        raise ValueError("Release package payload changed after matrix preparation.")
    current_launcher = _bound_launcher_identity(release)
    if current_launcher != plan["release"]["launcher"]:
        raise ValueError("Release-owned launcher changed after matrix preparation.")
    _identity_matches(plan["release"]["python_executable"], label="Python executable")
    cache_root = root / LAUNCHER_DIR_NAME / "cache"
    expected_cache_profiles = {*PROFILE_ORDER, "validator"}
    if cache_root.is_symlink() or not cache_root.is_dir():
        raise ValueError(f"Launcher cache root is missing or symlinked: {cache_root}")
    if {path.name for path in cache_root.iterdir()} != expected_cache_profiles:
        raise ValueError(
            "Launcher cache profile directories changed after preparation."
        )
    for profile in expected_cache_profiles:
        profile_cache = cache_root / profile
        if profile_cache.is_symlink() or not profile_cache.is_dir():
            raise ValueError(f"{profile} cache root is missing or symlinked.")
        expected_children = {"numba", "matplotlib", "xdg"}
        children = {path.name for path in profile_cache.iterdir()}
        if not expected_children <= children:
            raise ValueError(f"{profile} cache directories are incomplete: {children}.")
        for name in expected_children:
            cache_directory = profile_cache / name
            if cache_directory.is_symlink() or not cache_directory.is_dir():
                raise ValueError(f"{profile}/{name} cache is missing or symlinked.")
    for profile in PROFILE_ORDER:
        _identity_matches(
            plan["conditions"][profile]["training_config"],
            label=f"{profile} training config",
        )
        _identity_matches(
            plan["conditions"][profile]["workflow_preset"],
            label=f"{profile} workflow preset",
        )
    for dataset in DATASET_ORDER:
        source = plan["sources"][dataset]
        _identity_matches(source["aligned_h5ad"], label=f"{dataset} aligned H5AD")
        _identity_matches(
            source["edge_predictor"], label=f"{dataset} learned edge predictor"
        )
        _identity_matches(
            source["edge_predictor_metadata"],
            label=f"{dataset} edge predictor metadata",
        )
        current_graph = _tree_identity(
            source["input_graph"]["path"],
            label=f"{dataset} learned-prior input graph",
            require_adjacency_records=int(DATASET_SPECS[dataset]["graph_slices"]),
            dataset=dataset,
        )
        if current_graph != source["input_graph"]:
            raise ValueError(f"{dataset} input graph changed after preparation.")

    for profile in PROFILE_ORDER:
        dataset, arm = PROFILE_TO_DATASET_ARM[profile]
        preprocess = root / profile / "preprocess"
        aligned = preprocess / f"{dataset}_aligned.h5ad"
        expected_entries = {aligned.name}
        if arm == "full":
            expected_entries.update({"edge_classifier", "input_graph"})
        actual_entries = {path.name for path in preprocess.iterdir()}
        if actual_entries != expected_entries:
            raise ValueError(
                f"{profile} preprocess contents changed: expected "
                f"{sorted(expected_entries)}, got {sorted(actual_entries)}."
            )
        if not aligned.is_symlink() or aligned.resolve(strict=True) != Path(
            plan["sources"][dataset]["aligned_h5ad"]["path"]
        ):
            raise ValueError(f"{profile} aligned H5AD link changed.")
        forbidden = [
            preprocess / "edge_classifier",
            preprocess / "input_graph",
            preprocess / "metadata",
        ]
        if arm == "full":
            predictor = preprocess / "edge_classifier" / f"{dataset}_edge_model.pt"
            metadata = predictor.with_suffix(predictor.suffix + ".meta.json")
            graph = preprocess / "input_graph"
            if not predictor.is_symlink() or predictor.resolve(strict=True) != Path(
                plan["sources"][dataset]["edge_predictor"]["path"]
            ):
                raise ValueError(f"{profile} predictor link changed.")
            if not metadata.is_symlink() or metadata.resolve(strict=True) != Path(
                plan["sources"][dataset]["edge_predictor_metadata"]["path"]
            ):
                raise ValueError(f"{profile} predictor metadata link changed.")
            edge_entries = {path.name for path in predictor.parent.iterdir()}
            if edge_entries != {predictor.name, metadata.name}:
                raise ValueError(
                    f"{profile} edge_classifier contents changed: "
                    f"{sorted(edge_entries)}."
                )
            if not graph.is_symlink() or graph.resolve(strict=True) != Path(
                plan["sources"][dataset]["input_graph"]["path"]
            ):
                raise ValueError(f"{profile} input graph link changed.")
        elif any(path.exists() or path.is_symlink() for path in forbidden):
            raise ValueError(
                f"{profile} contains forbidden predictor/graph provenance artifacts."
            )
    return root, plan, digest


def _status_path(root: Path, profile: str, phase: str) -> Path:
    return root / LAUNCHER_DIR_NAME / "status" / f"{profile}.{phase}.json"


def _log_path(root: Path, profile: str, phase: str) -> Path:
    return root / LAUNCHER_DIR_NAME / "logs" / f"{profile}.{phase}.log"


def _gpu_reservation_path(root: Path, gpu: int) -> Path:
    return root / LAUNCHER_DIR_NAME / "status" / f"gpu-{gpu}.reservation.json"


def _reserve_gpu(
    root: Path,
    *,
    gpu: int,
    profile: str,
    phase: str,
    manifest_sha256: str,
    token: str,
) -> Path:
    """Atomically reserve one physical GPU before a detached monitor is spawned."""

    path = _gpu_reservation_path(root, gpu)
    reservation = {
        "schema_version": 1,
        "gpu": gpu,
        "profile": profile,
        "phase": phase,
        "manifest_sha256": manifest_sha256,
        "reservation_token": token,
        "state": "reserved",
        "reserved_at_utc": datetime.now(timezone.utc).isoformat(),
        "owner_pid": os.getpid(),
    }
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(reservation, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as error:
        raise RuntimeError(
            f"GPU cuda:{gpu} already has a formal reservation: {path}. "
            "Reservations are fail-closed; inspect status before using a new GPU."
        ) from error
    return path


def _owned_gpu_reservation(
    root: Path,
    *,
    gpu: int,
    profile: str,
    phase: str,
    manifest_sha256: str,
    token: str,
) -> dict[str, Any]:
    path = _gpu_reservation_path(root, gpu)
    reservation = _read_status(path)
    expected = {
        "gpu": gpu,
        "profile": profile,
        "phase": phase,
        "manifest_sha256": manifest_sha256,
    }
    if reservation is None or any(
        reservation.get(key) != value for key, value in expected.items()
    ):
        raise RuntimeError(f"GPU cuda:{gpu} reservation does not match this launch.")
    if not secrets.compare_digest(str(reservation.get("reservation_token", "")), token):
        raise RuntimeError(f"GPU cuda:{gpu} reservation token does not match.")
    return reservation


def _release_gpu_reservation(
    root: Path,
    *,
    gpu: int,
    profile: str,
    phase: str,
    manifest_sha256: str,
    token: str,
) -> None:
    _owned_gpu_reservation(
        root,
        gpu=gpu,
        profile=profile,
        phase=phase,
        manifest_sha256=manifest_sha256,
        token=token,
    )
    _gpu_reservation_path(root, gpu).unlink()


def _write_status(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    payload = _canonical_json_bytes(dict(value))
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _read_status(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Malformed launcher status: {path}")
    return value


def _pid_alive(pid: Any) -> bool:
    try:
        numeric = int(pid)
    except (TypeError, ValueError):
        return False
    if numeric <= 0:
        return False
    try:
        os.kill(numeric, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def status_snapshot(run_root: str | Path) -> dict[str, Any]:
    root, plan, digest = _load_prepared_plan(run_root)
    conditions: dict[str, Any] = {}
    for profile in PROFILE_ORDER:
        phases: dict[str, Any] = {}
        for phase in ("train", "downstream"):
            status = _read_status(_status_path(root, profile, phase))
            phases[phase] = status
        conditions[profile] = {
            "gpu": int(plan["conditions"][profile]["gpu"]),
            "train": phases["train"],
            "downstream": phases["downstream"],
            "training_summary_present": (
                root / profile / "training" / "training_run_summary.json"
            ).is_file(),
            "downstream_summary_present": (
                root / profile / "downstream" / "summary.json"
            ).is_file(),
            "monitor_alive": {
                phase: _pid_alive(
                    None if phases[phase] is None else phases[phase].get("monitor_pid")
                )
                for phase in ("train", "downstream")
            },
            "child_alive": {
                phase: _pid_alive(
                    None if phases[phase] is None else phases[phase].get("child_pid")
                )
                for phase in ("train", "downstream")
            },
        }
    return {
        "run_root": str(root),
        "manifest_sha256": digest,
        "conditions": conditions,
    }


def _assert_launchable(
    root: Path,
    plan: Mapping[str, Any],
    *,
    profile: str,
    phase: str,
    manifest_sha256: str | None = None,
) -> None:
    if profile not in PROFILE_ORDER:
        raise ValueError(f"Unknown profile: {profile!r}")
    if phase not in {"train", "downstream"}:
        raise ValueError(f"Unknown phase: {phase!r}")
    if _status_path(root, profile, phase).exists():
        raise FileExistsError(
            f"A {phase} launch record already exists for {profile}; formal runs "
            "are never resumed or overwritten by this launcher."
        )
    _assert_phase_outputs_fresh(
        root,
        profile=profile,
        phase=phase,
        manifest_sha256=manifest_sha256,
    )
    requested_gpu = int(plan["conditions"][profile]["gpu"])
    status_dir = root / LAUNCHER_DIR_NAME / "status"
    for path in status_dir.glob("*.json"):
        status = _read_status(path)
        if status is None or status.get("state") not in {"starting", "running"}:
            continue
        if int(status.get("gpu", -1)) == requested_gpu and (
            status.get("state") == "starting"
            or _pid_alive(status.get("monitor_pid"))
            or _pid_alive(status.get("child_pid"))
        ):
            raise RuntimeError(
                f"GPU cuda:{requested_gpu} is already reserved by "
                f"{status.get('profile')}/{status.get('phase')}."
            )


def _assert_phase_outputs_fresh(
    root: Path,
    *,
    profile: str,
    phase: str,
    manifest_sha256: str | None = None,
) -> None:
    """Recheck output freshness immediately before the workflow child starts."""

    condition_root = root / profile
    training = condition_root / "training"
    downstream = condition_root / "downstream"
    if phase == "train":
        if training.exists() or downstream.exists():
            raise FileExistsError(
                f"{profile} already has training/downstream output; use a new run root."
            )
    else:
        train_status = _read_status(_status_path(root, profile, "train"))
        status_ok = bool(
            train_status is not None
            and train_status.get("state") == "completed"
            and train_status.get("exit_code") == 0
            and train_status.get("profile") == profile
            and train_status.get("phase") == "train"
            and (
                manifest_sha256 is None
                or train_status.get("manifest_sha256") == manifest_sha256
            )
        )
        if not status_ok:
            raise RuntimeError(
                f"{profile} downstream requires this manifest's completed "
                "train launcher status with exit_code=0."
            )
        required = (
            training / "config.yaml",
            training / "training_run_summary.json",
            training / "training_history.csv",
            training / "adata.h5ad",
        )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"{profile} downstream requires a complete train-only fit; missing {missing}."
            )
        if downstream.exists():
            raise FileExistsError(
                f"{profile} downstream output already exists; use a new run root."
            )


def _phase_completion_missing(root: Path, *, profile: str, phase: str) -> list[str]:
    condition = root / profile
    if phase == "train":
        required = (
            condition / "training" / "config.yaml",
            condition / "training" / "training_run_summary.json",
            condition / "training" / "training_history.csv",
            condition / "training" / "adata.h5ad",
        )
    else:
        required = (condition / "downstream" / "summary.json",)
    return [str(path) for path in required if not path.is_file()]


def launch_one(
    run_root: str | Path,
    *,
    profile: str,
    phase: str,
    confirm_profile: str,
) -> dict[str, Any]:
    """Start one detached monitor after a literal per-profile confirmation."""

    if confirm_profile != profile:
        raise ValueError(
            "--confirm-profile must exactly repeat --profile to prevent an "
            "accidental expensive launch."
        )
    root, plan, digest = verify_prepared_run(run_root)
    _assert_launchable(
        root,
        plan,
        profile=profile,
        phase=phase,
        manifest_sha256=digest,
    )
    token = secrets.token_hex(16)
    status_path = _status_path(root, profile, phase)
    gpu = int(plan["conditions"][profile]["gpu"])
    _reserve_gpu(
        root,
        gpu=gpu,
        profile=profile,
        phase=phase,
        manifest_sha256=digest,
        token=token,
    )
    initial = {
        "schema_version": 1,
        "state": "starting",
        "profile": profile,
        "phase": phase,
        "gpu": gpu,
        "manifest_sha256": digest,
        "reservation_token": token,
        "requested_at_utc": datetime.now(timezone.utc).isoformat(),
        "monitor_pid": None,
        "child_pid": None,
        "exit_code": None,
        "log": str(_log_path(root, profile, phase)),
    }
    try:
        with status_path.open("x", encoding="utf-8") as handle:
            json.dump(initial, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
    except Exception:
        _release_gpu_reservation(
            root,
            gpu=gpu,
            profile=profile,
            phase=phase,
            manifest_sha256=digest,
            token=token,
        )
        raise
    monitor_argv = [
        str(plan["release"]["python_executable"]["path"]),
        str(plan["release"]["launcher"]["path"]),
        "_execute-one",
        "--run-root",
        str(root),
        "--profile",
        profile,
        "--phase",
        phase,
        "--reservation-token",
        token,
        "--manifest-sha256",
        digest,
    ]
    try:
        monitor = subprocess.Popen(
            monitor_argv,
            cwd=str(plan["release"]["root"]),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )
    except Exception as error:
        failed = dict(initial)
        failed.update(
            {
                "state": "launcher_failed",
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                "error": f"{type(error).__name__}: {error}",
            }
        )
        _write_status(status_path, failed)
        _release_gpu_reservation(
            root,
            gpu=gpu,
            profile=profile,
            phase=phase,
            manifest_sha256=digest,
            token=token,
        )
        raise
    current = _read_status(status_path)
    result = current if current is not None else initial
    result = dict(result)
    # Only the monitor writes durable state after spawn. This entirely removes
    # the parent/monitor read-then-write race that could regress running to
    # starting. The spawned PID remains useful in the immediate CLI response.
    result["spawned_monitor_pid"] = int(monitor.pid)
    return result


def _execute_one(
    run_root: str | Path,
    *,
    profile: str,
    phase: str,
    reservation_token: str,
    manifest_sha256: str,
) -> int:
    """Detached monitor implementation; not a user-facing entry point."""

    root, initial_plan, digest = _load_prepared_plan(run_root)
    if digest != manifest_sha256:
        raise ValueError("Monitor manifest SHA-256 does not match its reservation.")
    status_path = _status_path(root, profile, phase)
    reserved = _read_status(status_path)
    if (
        reserved is None
        or reserved.get("state") != "starting"
        or not secrets.compare_digest(
            str(reserved.get("reservation_token", "")), reservation_token
        )
    ):
        raise RuntimeError(
            "Launch reservation is absent, changed, or already consumed."
        )
    verifying = dict(reserved)
    verifying.update(
        {
            "state": "verifying",
            "monitor_pid": os.getpid(),
            "verification_started_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    _write_status(status_path, verifying)
    root, plan, verified_digest = verify_prepared_run(run_root)
    if verified_digest != digest or plan != initial_plan:
        raise ValueError("Prepared plan changed during monitor verification.")
    reserved = _read_status(status_path)
    if (
        reserved is None
        or reserved.get("state") != "verifying"
        or reserved.get("monitor_pid") != os.getpid()
        or not secrets.compare_digest(
            str(reserved.get("reservation_token", "")), reservation_token
        )
    ):
        raise RuntimeError(
            "Monitor status changed during immutable-input verification."
        )
    log_path = _log_path(root, profile, phase)
    command = plan["conditions"][profile]["commands"][phase]
    environment = os.environ.copy()
    environment.update(
        {str(key): str(value) for key, value in command["environment"].items()}
    )
    gpu = int(plan["conditions"][profile]["gpu"])
    _owned_gpu_reservation(
        root,
        gpu=gpu,
        profile=profile,
        phase=phase,
        manifest_sha256=digest,
        token=reservation_token,
    )
    gpu_lock_path = root / LAUNCHER_DIR_NAME / "status" / f"gpu-{gpu}.execution.lock"
    with gpu_lock_path.open("a+b") as gpu_lock:
        try:
            fcntl.flock(gpu_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            failed = dict(reserved)
            failed.update(
                {
                    "state": "failed",
                    "monitor_pid": os.getpid(),
                    "exit_code": 2,
                    "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                    "error": f"GPU cuda:{gpu} is locked by another matrix process.",
                }
            )
            _write_status(status_path, failed)
            raise RuntimeError(failed["error"]) from error
        # Close the launch/child-creation race. No unrelated process may insert
        # old output between the parent preflight and the real workflow start.
        _assert_phase_outputs_fresh(
            root,
            profile=profile,
            phase=phase,
            manifest_sha256=digest,
        )
        running = dict(reserved)
        running.update(
            {
                "state": "running",
                "monitor_pid": os.getpid(),
                "started_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        with log_path.open("xb") as log_handle:
            child = subprocess.Popen(
                [str(value) for value in command["argv"]],
                cwd=str(command["cwd"]),
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
            )
            running["child_pid"] = int(child.pid)
            _write_status(status_path, running)
            child_return_code = int(child.wait())
    missing_outputs = (
        _phase_completion_missing(root, profile=profile, phase=phase)
        if child_return_code == 0
        else []
    )
    return_code = (
        child_return_code if child_return_code != 0 or not missing_outputs else 3
    )
    finished = dict(running)
    finished.update(
        {
            "state": "completed" if return_code == 0 else "failed",
            "exit_code": return_code,
            "child_exit_code": child_return_code,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    if missing_outputs:
        finished[
            "error"
        ] = "Workflow exited zero but required phase outputs are missing: " + ", ".join(
            missing_outputs
        )
    _write_status(status_path, finished)
    _release_gpu_reservation(
        root,
        gpu=gpu,
        profile=profile,
        phase=phase,
        manifest_sha256=digest,
        token=reservation_token,
    )
    return return_code


def _record_monitor_failure(
    run_root: str | Path,
    *,
    profile: str,
    phase: str,
    reservation_token: str,
    error: Exception,
) -> None:
    """Best-effort conversion of monitor crashes into an auditable terminal state."""

    root = Path(run_root).expanduser().resolve(strict=False)
    status_path = _status_path(root, profile, phase)
    try:
        current = _read_status(status_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return
    if current is None or current.get("state") in {"completed", "failed"}:
        return
    if not secrets.compare_digest(
        str(current.get("reservation_token", "")), reservation_token
    ):
        return
    failed = dict(current)
    failed.update(
        {
            "state": "failed",
            "monitor_pid": os.getpid(),
            "exit_code": 2,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "error": f"{type(error).__name__}: {error}",
        }
    )
    try:
        _write_status(status_path, failed)
    except OSError:
        return
    try:
        gpu = int(current.get("gpu"))
        manifest_sha256 = str(current.get("manifest_sha256"))
        _release_gpu_reservation(
            root,
            gpu=gpu,
            profile=profile,
            phase=phase,
            manifest_sha256=manifest_sha256,
            token=reservation_token,
        )
    except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError):
        # Keep the reservation fail-closed if ownership cannot be proved.
        return


def _add_matrix_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument(
        "--release-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Clean Git top level containing this exact release.",
    )
    parser.add_argument("--release-commit", required=True)
    parser.add_argument(
        "--python-executable",
        required=True,
        type=Path,
        help=(
            "Server Python used for bound `python -m CytoBridge.cli workflow` "
            "commands."
        ),
    )
    parser.add_argument(
        "--aligned-h5ad",
        action="append",
        default=[],
        metavar="DATASET=PATH",
        help="Repeat exactly once for each of zebrafish, mosta, arista, admouse.",
    )
    parser.add_argument(
        "--edge-predictor",
        action="append",
        default=[],
        metavar="DATASET=PATH",
        help="Validation-selected learned predictor; repeat for all four datasets.",
    )
    parser.add_argument(
        "--input-graph-dir",
        action="append",
        default=[],
        metavar="DATASET=PATH",
        help="Exact LR graph provenance directory; repeat for all four datasets.",
    )
    parser.add_argument(
        "--gpu",
        action="append",
        default=[],
        metavar="PROFILE=INDEX",
        help="Repeat exactly once for each of the twelve validator profile names.",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("dry-run", "prepare"):
        child = subparsers.add_parser(name)
        _add_matrix_inputs(child)
        child.add_argument(
            "--json",
            action="store_true",
            help="Print the complete manifest rather than the command summary.",
        )
        child.add_argument(
            "--plan-out",
            type=Path,
            help="Write the plan with exclusive-create semantics (dry-run only).",
        )

    render = subparsers.add_parser("render")
    render.add_argument("--run-root", required=True, type=Path)
    render.add_argument("--profile", choices=PROFILE_ORDER)
    render.add_argument(
        "--phase", choices=("train", "downstream", "both"), default="both"
    )
    render.add_argument("--include-validator", action="store_true")

    launch = subparsers.add_parser("launch-one")
    launch.add_argument("--run-root", required=True, type=Path)
    launch.add_argument("--profile", required=True, choices=PROFILE_ORDER)
    launch.add_argument("--phase", required=True, choices=("train", "downstream"))
    launch.add_argument(
        "--confirm-profile",
        required=True,
        help="Must exactly repeat --profile; this is the expensive-action guard.",
    )

    status = subparsers.add_parser("status")
    status.add_argument("--run-root", required=True, type=Path)

    hidden = subparsers.add_parser("_execute-one", help=argparse.SUPPRESS)
    hidden.add_argument("--run-root", required=True, type=Path)
    hidden.add_argument("--profile", required=True, choices=PROFILE_ORDER)
    hidden.add_argument("--phase", required=True, choices=("train", "downstream"))
    hidden.add_argument("--reservation-token", required=True)
    hidden.add_argument("--manifest-sha256", required=True)
    return parser


def _plan_from_args(args: argparse.Namespace) -> dict[str, Any]:
    aligned = _parse_assignments(
        args.aligned_h5ad,
        allowed_keys=DATASET_ORDER,
        option="--aligned-h5ad",
    )
    predictors = _parse_assignments(
        args.edge_predictor,
        allowed_keys=DATASET_ORDER,
        option="--edge-predictor",
    )
    graph_dirs = _parse_assignments(
        args.input_graph_dir,
        allowed_keys=DATASET_ORDER,
        option="--input-graph-dir",
    )
    gpus = _gpu_assignments(args.gpu)
    return build_plan(
        run_root=args.run_root,
        release_root=args.release_root,
        release_commit=args.release_commit,
        python_executable=args.python_executable,
        aligned_h5ad=aligned,
        edge_predictor=predictors,
        input_graph_dir=graph_dirs,
        gpu_by_profile=gpus,
    )


def _print_commands(plan: Mapping[str, Any]) -> None:
    print(f"run root: {plan['run_root']}")
    print(f"release commit: {plan['release']['commit']}")
    print(f"training code SHA-256: {plan['release']['training_code']['sha256']}")
    print(f"manifest SHA-256: {_plan_digest(plan)}")
    for profile in PROFILE_ORDER:
        print(f"[{profile}] train")
        print(plan["conditions"][profile]["commands"]["train"]["shell"])
        print(f"[{profile}] downstream")
        print(plan["conditions"][profile]["commands"]["downstream"]["shell"])
    print("[validator]")
    print(plan["validator"]["shell"])


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command in {"dry-run", "prepare"}:
        plan = _plan_from_args(args)
        if args.plan_out is not None:
            if args.command != "dry-run":
                raise ValueError("--plan-out is only valid for a non-mutating dry-run.")
            destination = args.plan_out.expanduser().resolve(strict=False)
            with destination.open("xb") as handle:
                handle.write(_canonical_json_bytes(plan))
        if args.command == "prepare":
            root, digest = prepare_run_root(plan)
            print(f"Prepared fresh matched matrix: {root}")
            print(f"Immutable manifest SHA-256: {digest}")
        if args.json:
            print(_canonical_json_bytes(plan).decode("utf-8"), end="")
        else:
            _print_commands(plan)
        return 0
    if args.command == "render":
        _, plan, _ = verify_prepared_run(args.run_root)
        profiles = PROFILE_ORDER if args.profile is None else (args.profile,)
        phases = ("train", "downstream") if args.phase == "both" else (args.phase,)
        for profile in profiles:
            for phase in phases:
                print(plan["conditions"][profile]["commands"][phase]["shell"])
        if args.include_validator:
            print(plan["validator"]["shell"])
        return 0
    if args.command == "launch-one":
        status = launch_one(
            args.run_root,
            profile=args.profile,
            phase=args.phase,
            confirm_profile=args.confirm_profile,
        )
        print(json.dumps(status, indent=2, sort_keys=True, allow_nan=False))
        return 0
    if args.command == "status":
        print(
            json.dumps(
                status_snapshot(args.run_root),
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
        )
        return 0
    if args.command == "_execute-one":
        try:
            return _execute_one(
                args.run_root,
                profile=args.profile,
                phase=args.phase,
                reservation_token=args.reservation_token,
                manifest_sha256=args.manifest_sha256,
            )
        except Exception as error:
            _record_monitor_failure(
                args.run_root,
                profile=args.profile,
                phase=args.phase,
                reservation_token=args.reservation_token,
                error=error,
            )
            raise
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        FileNotFoundError,
        FileExistsError,
        PermissionError,
        RuntimeError,
        ValueError,
    ) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2) from error
