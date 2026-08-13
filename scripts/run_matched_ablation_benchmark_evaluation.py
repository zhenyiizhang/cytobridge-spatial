#!/usr/bin/env python3
"""Plan and report the matched-ablation full-data benchmark evaluation.

This entry point never treats the twelve generic downstream directories as
reconstruction evidence.  It plans one official unified-benchmark
``infer-full`` call and one frozen ``evaluate_predictions`` call for each of
the four data sets and three accepted training arms.  Preparation binds the
matched-family acceptance report, launcher manifest, package adapter/evaluator
bytes, benchmark inputs, twelve resolved configs, training summaries, and all
six checkpoint files per profile before creating a new evaluation root.

The tool does not launch work.  ``render`` prints the exact commands for an
external scheduler.  ``validate`` is read-only.  ``report`` requires all
twelve prediction and scoring contracts, then writes arm-labelled long tables,
paired full-versus-ablation deltas, a publication PDF/PNG, caption, provenance,
and a signed report manifest.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import stat
import subprocess
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402


SCHEMA_VERSION = 1
PLAN_KIND = "cytobridge-matched-ablation-full-data-benchmark-evaluation"
REPORT_KIND = f"{PLAN_KIND}-report"
CONTRACT_DIR = "_matched_benchmark_evaluation"
PLAN_NAME = "evaluation_plan.json"
PLAN_SIDECAR = f"{PLAN_NAME}.sha256"
DATASET_ORDER = ("zebrafish", "mosta", "arista", "admouse")
ARM_ORDER = ("full", "no_lr_prior", "no_interaction")
ARM_INTERACTION_MODE = {
    "full": "learned",
    "no_lr_prior": "all_spatial",
    "no_interaction": "none",
}
ARM_LABEL = {
    "full": "Full model",
    "no_lr_prior": "Radius-only interaction",
    "no_interaction": "No interaction",
}
DATASET_LABEL = {
    "zebrafish": "Zebrafish",
    "mosta": "MOSTA",
    "arista": "ARISTA",
    "admouse": "AD mouse",
}
SPACE_ORDER = ("joint", "state", "spatial")
SPACE_LABEL = {"joint": "Joint", "state": "State", "spatial": "Spatial"}
METRIC_ORDER = ("sliced_w2", "exact_w1", "exact_w2")
METHOD = "CytoBridge-0.015"
FULL_DATA_TARGETS = {
    "zebrafish": [1, 2, 3, 4],
    "mosta": [1, 2, 3],
    "arista": [1, 2, 3, 4],
    "admouse": [1, 2],
}
MATCHED_PROTOCOL = "isolated-interaction-crn-v1"
MATCHED_SEED = 42
PREDICTION_N = 5000
N_PROJECTIONS = 1024
PROJECTION_REPEATS = 5
MAX_OT_POINTS = 800
INFERENCE_DT = 0.01
INTERACTION_M = 1024
ALPHA_EXPRESS = 0.015
ALPHA_SPATIAL = 10.0
EXPECTED_IMPLEMENTATION = (
    f"CytoBridge alpha_spatial={ALPHA_SPATIAL:g}, " f"alpha_express={ALPHA_EXPRESS:g}"
)
INFERENCE_SIGMA = 0.03
INTERACTION_GROUPING_SEED = MATCHED_SEED + 10_000
OFFICIAL_SIMULATION_API = "CytoBridge.tl.downstream.simulation.simulate_sde_points"
OFFICIAL_SIMULATION_MODE = "continuous_non_split_weighted_sde"
STOCHASTIC_STREAM_CONTRACT = (
    "interaction grouping uses an independent torch.Generator; the global torch "
    "stream remains paired for Brownian diffusion"
)
WEIGHTS_SEMANTICS = "native_unnormalised_growth_mass"
INPUT_CONTRACT = "cytobridge-spatiotemporal-benchmark-input-v1"
BENCHMARK_PROJECTION_SEED_NAMESPACE = "cytobridge-spatiotemporal-benchmark-v1"
LAUNCHER_KIND = "cytobridge-four-dataset-matched-ablation-matrix"
LAUNCHER_SCHEMA_VERSION = 2
ADAPTER_FILES = (
    "scripts/spatiotemporal_benchmark/cytobridge/common.py",
    "scripts/spatiotemporal_benchmark/cytobridge/run_cytobridge.py",
    "CytoBridge/tl/core/interaction.py",
    "CytoBridge/tl/core/methods.py",
    "CytoBridge/tl/core/models.py",
    "CytoBridge/tl/graph/spatial_gnn.py",
    "CytoBridge/tl/downstream/checkpoint.py",
    "CytoBridge/tl/downstream/runtime.py",
    "CytoBridge/tl/downstream/simulation.py",
    "CytoBridge/tl/train/trainer.py",
    "CytoBridge/tl/train/fit.py",
    "CytoBridge/pp/edge_prediction.py",
    "CytoBridge/pp/interaction_graph.py",
)
EVALUATOR_FILES = (
    "scripts/spatiotemporal_benchmark/evaluate_predictions.py",
    "CytoBridge/tl/downstream/benchmark.py",
    "CytoBridge/tl/downstream/evaluation.py",
)
TEXT_COLOR = "#24313A"
GRID_COLOR = "#D7DDE2"
REFERENCE_COLOR = "#59616A"
NO_LR_COLOR = "#7A6BBE"
NO_INTERACTION_COLOR = "#CC6677"
FIGURE_BASENAME = "matched_ablation_full_data_benchmark"


class ContractError(ValueError):
    """An input, prediction, score, or report violates the frozen contract."""


def profile_name(dataset: str, arm: str) -> str:
    if dataset not in DATASET_ORDER or arm not in ARM_ORDER:
        raise ContractError(f"Unknown dataset/arm pair: {dataset!r}/{arm!r}")
    return dataset if arm == "full" else f"{dataset}_{arm}"


PROFILE_ORDER = tuple(
    profile_name(dataset, arm) for dataset in DATASET_ORDER for arm in ARM_ORDER
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalise_digest(value: Any, *, name: str) -> str:
    digest = str(value or "").strip().lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ContractError(f"{name} must be a lowercase 64-character SHA-256")
    return digest


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _stable_json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def _read_json(path: Path, *, description: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"Cannot read {description} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ContractError(f"{description} must contain a JSON object")
    return payload


def _read_yaml(path: Path, *, description: str) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ContractError(f"Cannot read {description} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ContractError(f"{description} must contain a YAML object")
    return payload


def _mapping(value: Any, *, description: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{description} must be an object")
    return value


def _verified_file(
    path: str | Path, expected_sha256: Any, *, description: str
) -> tuple[Path, str]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Missing {description}: {resolved}")
    expected = _normalise_digest(expected_sha256, name=f"{description} SHA-256")
    observed = _sha256(resolved)
    if observed != expected:
        raise ContractError(
            f"{description} SHA-256 mismatch: expected {expected}, observed {observed}"
        )
    return resolved, observed


def _file_identity(path: str | Path, *, description: str) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        raise FileNotFoundError(f"Missing or empty {description}: {resolved}")
    return {
        "path": str(resolved),
        "size_bytes": int(resolved.stat().st_size),
        "sha256": _sha256(resolved),
    }


def _absolute_lexical_path(path: str | Path) -> Path:
    """Return an absolute path without resolving symbolic links."""

    return Path(os.path.abspath(os.fspath(Path(path).expanduser())))


def _lstat_identity(path: Path, *, description: str) -> dict[str, Any]:
    try:
        entry_stat = path.lstat()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Missing {description}: {path}") from exc
    if stat.S_ISLNK(entry_stat.st_mode):
        file_type = "symlink"
    elif stat.S_ISREG(entry_stat.st_mode):
        file_type = "regular_file"
    elif stat.S_ISDIR(entry_stat.st_mode):
        file_type = "directory"
    else:
        file_type = "other"
    result: dict[str, Any] = {
        "path": str(path),
        "file_type": file_type,
        "mode": f"{stat.S_IMODE(entry_stat.st_mode):04o}",
        "size_bytes": int(entry_stat.st_size),
    }
    if file_type == "symlink":
        result["link_target"] = os.readlink(path)
    return result


def _symlink_chain_identity(path: Path, *, description: str) -> list[dict[str, Any]]:
    """Mirror the schema-2 launcher's lexical symlink-chain identity."""

    if not path.is_absolute():
        raise ContractError(f"{description} path must be absolute")
    resolved_prefix = Path(path.anchor)
    pending = list(path.parts[1:])
    seen_states: set[tuple[str, str, tuple[str, ...]]] = set()
    chain: list[dict[str, Any]] = []
    hops = 0
    while pending:
        component = pending.pop(0)
        candidate = resolved_prefix / component
        entry = _lstat_identity(candidate, description=description)
        if entry["file_type"] != "symlink":
            resolved_prefix = candidate
            continue
        hops += 1
        if hops > 64:
            raise ContractError(f"{description} has more than 64 symlink hops")
        raw_target_text = str(entry["link_target"])
        state = (str(candidate), raw_target_text, tuple(pending))
        if state in seen_states:
            raise ContractError(f"{description} contains a symlink loop")
        seen_states.add(state)
        chain.append(entry)
        raw_target = Path(raw_target_text)
        if raw_target.is_absolute():
            target = Path(os.path.normpath(os.fspath(raw_target)))
        else:
            target = Path(
                os.path.normpath(
                    os.path.join(os.fspath(candidate.parent), raw_target_text)
                )
            )
        pending = list(target.parts[1:]) + pending
        resolved_prefix = Path(target.anchor)
    return chain


def _python_runtime_probe(invocation: Path) -> dict[str, Any]:
    probe_source = (
        "import json, platform, sys; "
        "print(json.dumps({"
        "'executable': sys.executable, "
        "'prefix': sys.prefix, "
        "'base_prefix': sys.base_prefix, "
        "'exec_prefix': sys.exec_prefix, "
        "'base_exec_prefix': sys.base_exec_prefix, "
        "'implementation': platform.python_implementation(), "
        "'cache_tag': sys.implementation.cache_tag, "
        "'version': platform.python_version()"
        "}, sort_keys=True))"
    )
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTHONHASHSEED": str(MATCHED_SEED),
        }
    )
    try:
        completed = subprocess.run(
            [str(invocation), "-I", "-B", "-c", probe_source],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
            timeout=30,
        )
    except subprocess.TimeoutExpired as exc:
        raise ContractError(f"Python runtime probe timed out: {invocation}") from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise ContractError(f"Python runtime probe failed: {detail}")
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ContractError("Python runtime probe returned invalid JSON") from exc
    required = {
        "executable",
        "prefix",
        "base_prefix",
        "exec_prefix",
        "base_exec_prefix",
        "implementation",
        "cache_tag",
        "version",
    }
    if (
        not isinstance(value, dict)
        or set(value) != required
        or not all(isinstance(value[key], str) and value[key] for key in required)
    ):
        raise ContractError("Python runtime probe is incomplete")
    return value


def _python_environment_files(
    invocation: Path, runtime: Mapping[str, Any]
) -> list[dict[str, Any]]:
    candidates = (
        invocation.parent.parent / "pyvenv.cfg",
        Path(str(runtime["prefix"])) / "pyvenv.cfg",
    )
    identities: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in candidates:
        lexical = _absolute_lexical_path(candidate)
        key = str(lexical)
        if key in seen:
            continue
        seen.add(key)
        if not lexical.exists() and not lexical.is_symlink():
            continue
        identity = _file_identity(lexical, description="Python environment pyvenv.cfg")
        identity["invocation_path"] = key
        identities.append(identity)
    return identities


def _python_executable_identity(path: str | Path) -> dict[str, Any]:
    invocation = _absolute_lexical_path(path)
    invocation_lstat = _lstat_identity(invocation, description="Python executable")
    if not invocation.exists() or not os.access(invocation, os.X_OK):
        raise FileNotFoundError(f"Python executable is unavailable: {invocation}")
    resolved = invocation.resolve(strict=True)
    if not resolved.is_file():
        raise ContractError("Resolved Python executable is not a regular file")
    resolved_identity = _file_identity(
        resolved, description="resolved Python executable"
    )
    resolved_identity["lstat"] = _lstat_identity(
        resolved, description="resolved Python executable"
    )
    runtime = _python_runtime_probe(invocation)
    if Path(str(runtime["executable"])).resolve(strict=True) != resolved:
        raise ContractError("Python runtime probe resolved to another executable")
    return {
        "invocation_path": str(invocation),
        "invocation_lstat": invocation_lstat,
        "symlink_chain": _symlink_chain_identity(
            invocation, description="Python executable"
        ),
        "resolved_target": resolved_identity,
        "runtime": runtime,
        "environment_files": _python_environment_files(invocation, runtime),
    }


def _launcher_aggregate_records(records: Sequence[Mapping[str, Any]]) -> str:
    projected = [
        {
            "path": str(record["path"]),
            "size_bytes": int(record["size_bytes"]),
            "sha256": str(record["sha256"]),
        }
        for record in records
    ]
    return hashlib.sha256(
        json.dumps(projected, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _release_tree_identity(release_root: Path, *, python_only: bool) -> dict[str, Any]:
    """Recompute the exact schema-2 launcher release-tree identity."""

    package_root = release_root / "CytoBridge"
    if not package_root.is_dir():
        raise FileNotFoundError(f"Release lacks CytoBridge package: {package_root}")
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
            raise ContractError(f"Release payload contains symlink: {candidate}")
        if not candidate.is_file():
            continue
        records.append(
            {
                "path": candidate.relative_to(release_root).as_posix(),
                "size_bytes": int(candidate.stat().st_size),
                "sha256": _sha256(candidate),
            }
        )
    if not records:
        raise ContractError("Release identity contains no files")
    return {
        "file_count": len(records),
        "sha256": _launcher_aggregate_records(records),
        "files": records,
    }


def _git_release_identity(release_root: Path, commit: str) -> dict[str, Any]:
    if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit):
        raise ContractError("Launcher release commit is not a complete lowercase SHA-1")

    def run(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", "-C", str(release_root), *arguments],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise ContractError(
                f"Cannot verify launcher release Git identity: {detail}"
            )
        return completed.stdout.strip()

    top_level = Path(run("rev-parse", "--show-toplevel")).resolve(strict=True)
    observed_commit = run("rev-parse", "HEAD").lower()
    dirty = run("status", "--porcelain", "--untracked-files=all")
    if top_level != release_root or observed_commit != commit or dirty:
        raise ContractError(
            "Launcher release Git checkout is no longer exact and clean"
        )
    return {"top_level": str(top_level), "commit": observed_commit, "clean": True}


def _verify_release_binding(release_record: Any) -> tuple[Path, str]:
    release = _mapping(release_record, description="launcher release")
    required = {
        "root",
        "commit",
        "git",
        "training_code",
        "package_payload",
        "launcher",
        "python_executable",
    }
    if set(release) != required:
        raise ContractError("Launcher schema-2 release identity is incomplete")
    root = Path(str(release["root"])).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ContractError("Launcher release root is not a directory")
    commit = str(release["commit"]).lower()
    if _git_release_identity(root, commit) != release["git"]:
        raise ContractError("Launcher release Git identity differs from manifest")
    if _release_tree_identity(root, python_only=True) != release["training_code"]:
        raise ContractError("Launcher release training-code identity changed")
    if _release_tree_identity(root, python_only=False) != release["package_payload"]:
        raise ContractError("Launcher release package-payload identity changed")
    launcher_identity = _file_identity(
        root / "scripts" / "run_matched_ablation_matrix.py",
        description="release-owned matched launcher",
    )
    if launcher_identity != release["launcher"]:
        raise ContractError("Release-owned matched launcher identity changed")
    python_identity = _mapping(
        release["python_executable"], description="launcher Python identity"
    )
    invocation = python_identity.get("invocation_path")
    if not isinstance(invocation, str) or not invocation:
        raise ContractError("Launcher Python invocation path is malformed")
    if _python_executable_identity(invocation) != dict(python_identity):
        raise ContractError("Launcher Python lexical/runtime identity changed")
    return root, invocation


def _identity_matches(record: Any, *, description: str) -> None:
    identity = _mapping(record, description=f"{description} identity")
    required = {"path", "size_bytes", "sha256"}
    if not required <= set(identity):
        raise ContractError(f"{description} identity has unexpected fields")
    actual = _file_identity(identity["path"], description=description)
    if actual != {key: identity[key] for key in required}:
        raise ContractError(f"{description} changed after preparation")


def _sidecar_check(path: Path, digest: str, *, filename_required: bool) -> Path:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not sidecar.is_file():
        raise FileNotFoundError(f"Missing SHA-256 sidecar: {sidecar}")
    fields = sidecar.read_text(encoding="utf-8").strip().split()
    expected = [digest, path.name] if filename_required else [digest]
    if fields != expected:
        raise ContractError(f"SHA-256 sidecar does not bind {path}")
    return sidecar


def _assignments(values: Sequence[str], *, option: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        dataset, separator, item = str(value).partition("=")
        if not separator or dataset not in DATASET_ORDER or not item:
            raise ContractError(f"{option} expects DATASET=VALUE, found {value!r}")
        if dataset in result:
            raise ContractError(f"{option} repeats dataset {dataset!r}")
        result[dataset] = item
    if set(result) != set(DATASET_ORDER):
        raise ContractError(f"{option} must contain exactly {list(DATASET_ORDER)}")
    return result


def _verify_launcher_manifest(
    path: str | Path, expected_sha256: str
) -> tuple[Path, str, dict[str, Any], Path]:
    manifest_path, digest = _verified_file(
        path, expected_sha256, description="matched launcher manifest"
    )
    if manifest_path.name != "matched_ablation_matrix_manifest.json":
        raise ContractError("Launcher manifest has an unexpected filename")
    _sidecar_check(manifest_path, digest, filename_required=False)
    payload = _read_json(manifest_path, description="matched launcher manifest")
    if payload.get("schema_version") != LAUNCHER_SCHEMA_VERSION:
        raise ContractError("Launcher manifest schema_version is not 2")
    if payload.get("kind") != LAUNCHER_KIND:
        raise ContractError("Launcher manifest kind is not the matched matrix")
    _verify_release_binding(payload.get("release"))
    matrix = _mapping(payload.get("matrix"), description="launcher matrix")
    required_matrix = {
        "datasets": list(DATASET_ORDER),
        "arms": list(ARM_ORDER),
        "profiles": list(PROFILE_ORDER),
        "fit_count": 12,
        "protocol": MATCHED_PROTOCOL,
        "shared_seed": MATCHED_SEED,
    }
    for key, expected in required_matrix.items():
        if matrix.get(key) != expected:
            raise ContractError(f"Launcher matrix.{key} differs from {expected!r}")
    run_root = Path(str(payload.get("run_root", ""))).expanduser().resolve()
    if not run_root.is_dir() or manifest_path.parent.parent != run_root:
        raise ContractError(
            "Launcher run_root is missing or differs from manifest path"
        )
    conditions = _mapping(payload.get("conditions"), description="launcher conditions")
    if set(conditions) != set(PROFILE_ORDER):
        raise ContractError("Launcher manifest does not contain exactly 12 profiles")
    for dataset in DATASET_ORDER:
        for arm in ARM_ORDER:
            profile = profile_name(dataset, arm)
            condition = _mapping(
                conditions[profile], description=f"launcher condition {profile}"
            )
            if (
                condition.get("dataset") != dataset
                or condition.get("arm") != arm
                or condition.get("protocol") != MATCHED_PROTOCOL
                or condition.get("shared_seed") != MATCHED_SEED
            ):
                raise ContractError(
                    f"Launcher condition identity drifted for {profile}"
                )
            condition_root = Path(
                str(
                    _mapping(condition.get("paths"), description="condition paths").get(
                        "condition_root", ""
                    )
                )
            ).resolve()
            if condition_root != run_root / profile:
                raise ContractError(f"Launcher condition root drifted for {profile}")
    return manifest_path, digest, payload, run_root


def _verify_acceptance(
    path: str | Path,
    expected_sha256: str,
    *,
    launcher_root: Path,
) -> tuple[Path, str, dict[str, Any]]:
    report_path, digest = _verified_file(
        path, expected_sha256, description="matched-family acceptance report"
    )
    report = _read_json(report_path, description="matched-family acceptance report")
    if report.get("status") != "PASS":
        raise ContractError("Matched-family acceptance report is not overall PASS")
    if Path(str(report.get("run_root", ""))).expanduser().resolve() != launcher_root:
        raise ContractError("Acceptance run_root differs from launcher run_root")
    datasets = _mapping(report.get("datasets"), description="acceptance datasets")
    families = _mapping(
        report.get("matched_families"), description="acceptance matched_families"
    )
    if set(datasets) != set(PROFILE_ORDER) or set(families) != set(DATASET_ORDER):
        raise ContractError("Acceptance report does not cover the exact 12-arm matrix")
    for profile in PROFILE_ORDER:
        if _mapping(datasets[profile], description=profile).get("status") != "PASS":
            raise ContractError(f"Acceptance report does not accept profile {profile}")
    for dataset in DATASET_ORDER:
        if _mapping(families[dataset], description=dataset).get("status") != "PASS":
            raise ContractError(
                f"Acceptance report does not accept matched family {dataset}"
            )
    return report_path, digest, report


def _artifact_from_record(
    record: Any,
    *,
    manifest_path: Path,
    description: str,
) -> dict[str, Any]:
    value = _mapping(record, description=f"{description} record")
    relative = value.get("relative_path")
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ContractError(f"{description} must use a relative_path")
    root = manifest_path.parent.resolve()
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ContractError(f"{description} escapes benchmark input root") from exc
    expected = _normalise_digest(value.get("sha256"), name=f"{description} SHA-256")
    verified, _ = _verified_file(path, expected, description=description)
    size = value.get("size_bytes")
    if (
        isinstance(size, int)
        and not isinstance(size, bool)
        and size != verified.stat().st_size
    ):
        raise ContractError(f"{description} size differs from its manifest")
    return _file_identity(verified, description=description)


def _verify_benchmark_input(
    dataset: str,
    path: str | Path,
    expected_sha256: str,
    *,
    launcher_aligned_sha256: str,
) -> dict[str, Any]:
    manifest_path, digest = _verified_file(
        path, expected_sha256, description=f"{dataset} benchmark input manifest"
    )
    if manifest_path.name != "manifest.json":
        raise ContractError(f"{dataset} benchmark manifest must be named manifest.json")
    root_sidecar = _sidecar_check(manifest_path, digest, filename_required=True)
    payload = _read_json(manifest_path, description=f"{dataset} benchmark manifest")
    if (
        payload.get("contract_version") != INPUT_CONTRACT
        or payload.get("status") != "complete"
        or payload.get("dataset_id") != dataset
        or payload.get("prediction_n") != PREDICTION_N
    ):
        raise ContractError(f"{dataset} benchmark root contract is not frozen")
    source = _mapping(payload.get("source"), description=f"{dataset} input source")
    if source.get("h5ad_sha256") != launcher_aligned_sha256:
        raise ContractError(
            f"{dataset} benchmark source does not match the launcher-aligned input"
        )
    config_source_record = _mapping(
        payload.get("config_source"), description=f"{dataset} config_source"
    )
    config_source_path, _ = _verified_file(
        config_source_record.get("path"),
        config_source_record.get("sha256"),
        description=f"{dataset} benchmark config source",
    )
    config_source = _read_yaml(
        config_source_path, description=f"{dataset} benchmark config source"
    )
    resolved_config_identity = _artifact_from_record(
        payload.get("resolved_config"),
        manifest_path=manifest_path,
        description=f"{dataset} resolved benchmark config",
    )
    resolved_config = _read_yaml(
        Path(resolved_config_identity["path"]),
        description=f"{dataset} resolved benchmark config",
    )
    targets = list(FULL_DATA_TARGETS[dataset])
    for label, config in (
        ("config source", config_source),
        ("resolved config", resolved_config),
    ):
        if (
            config.get("dataset_id") != dataset
            or config.get("prediction_n") != PREDICTION_N
            or config.get("full_data_targets") != targets
        ):
            raise ContractError(f"{dataset} {label} differs from the frozen matrix")
    if payload.get("full_data_targets") != targets:
        raise ContractError(
            f"{dataset} full-data target set differs from the frozen matrix"
        )
    splits = _mapping(payload.get("splits"), description=f"{dataset} benchmark splits")
    full = _mapping(splits.get("full_data"), description=f"{dataset} full_data split")
    if (
        full.get("protocol") != "full_data"
        or full.get("held_out_benchmark_time") is not None
        or full.get("evaluation_targets") != targets
        or full.get("prediction_n") != PREDICTION_N
        or full.get("source_time") != 0
        or full.get("transductive_frozen_representation") is not True
        or full.get("representation_refit_per_fold") is not False
    ):
        raise ContractError(f"{dataset} full_data split contract is not frozen")
    artifacts: dict[str, dict[str, Any]] = {}
    train = _mapping(full.get("train"), description=f"{dataset} full_data train")
    for key in ("h5ad", "training_reference_npz", "source_roster_npz"):
        artifacts[f"train_{key}"] = _artifact_from_record(
            train.get(key),
            manifest_path=manifest_path,
            description=f"{dataset} full_data train {key}",
        )
    truth = _mapping(
        full.get("truth_by_time_npz"), description=f"{dataset} truth_by_time_npz"
    )
    if set(truth) != {str(target) for target in targets}:
        raise ContractError(f"{dataset} truth target artifacts are incomplete")
    for target in targets:
        artifacts[f"truth_t{target}"] = _artifact_from_record(
            truth[str(target)],
            manifest_path=manifest_path,
            description=f"{dataset} full_data truth t{target}",
        )
    split_manifest_record = full.get("manifest")
    split_sidecar_record = full.get("manifest_sha256_sidecar")
    artifacts["split_manifest"] = _artifact_from_record(
        split_manifest_record,
        manifest_path=manifest_path,
        description=f"{dataset} full_data split manifest",
    )
    artifacts["split_manifest_sidecar"] = _artifact_from_record(
        split_sidecar_record,
        manifest_path=manifest_path,
        description=f"{dataset} full_data split manifest sidecar",
    )
    split_path = Path(artifacts["split_manifest"]["path"])
    _sidecar_check(
        split_path, artifacts["split_manifest"]["sha256"], filename_required=True
    )
    return {
        "manifest": _file_identity(manifest_path, description=f"{dataset} manifest"),
        "manifest_sidecar": _file_identity(
            root_sidecar, description=f"{dataset} manifest sidecar"
        ),
        "config_source": _file_identity(
            config_source_path, description=f"{dataset} benchmark config source"
        ),
        "resolved_config": resolved_config_identity,
        "targets": list(targets),
        "source_aligned_sha256": launcher_aligned_sha256,
        "artifacts": artifacts,
    }


def _checkpoint_inventory(
    profile: str,
    arm: str,
    model_dir: Path,
    source_config_identity: Mapping[str, Any],
) -> dict[str, Any]:
    if not model_dir.is_dir():
        raise FileNotFoundError(
            f"Missing training directory for {profile}: {model_dir}"
        )
    _identity_matches(source_config_identity, description=f"{profile} source config")
    config_path = model_dir / "config.yaml"
    summary_path = model_dir / "training_run_summary.json"
    config = _read_yaml(config_path, description=f"{profile} resolved config")
    _read_json(summary_path, description=f"{profile} training summary")
    if config.get("seed") != MATCHED_SEED:
        raise ContractError(f"{profile} saved config seed is not {MATCHED_SEED}")
    model = _mapping(config.get("model"), description=f"{profile} model config")
    components = {str(value).strip().lower() for value in model.get("components", [])}
    expected_components = (
        {"velocity", "growth", "score"}
        if arm == "no_interaction"
        else {"velocity", "growth", "score", "interaction"}
    )
    if components != expected_components:
        raise ContractError(f"{profile} model components differ from arm contract")
    if arm == "no_interaction":
        if any(
            key in model
            for key in ("interaction_net", "interaction_type", "interaction_group_size")
        ):
            raise ContractError(f"{profile} carries forbidden interaction fields")
        interaction_mode = "none"
    else:
        interaction = _mapping(
            model.get("interaction_net"), description=f"{profile} interaction config"
        )
        interaction_mode = str(interaction.get("edge_prior_mode", "")).strip().lower()
        if interaction_mode != ARM_INTERACTION_MODE[arm]:
            raise ContractError(f"{profile} saved edge-prior mode differs from arm")
        if model.get("interaction_group_size") != INTERACTION_M:
            raise ContractError(
                f"{profile} interaction_group_size is not {INTERACTION_M}"
            )
        if arm == "no_lr_prior" and any(
            key in interaction for key in ("edge_predictor_path", "edge_predictor_thre")
        ):
            raise ContractError(
                f"{profile} radius-only config carries predictor fields"
            )
        if arm == "full":
            predictor_path = interaction.get("edge_predictor_path")
            predictor_threshold = interaction.get("edge_predictor_thre")
            if not isinstance(predictor_path, str) or not predictor_path.strip():
                raise ContractError(f"{profile} learned config lacks predictor path")
            try:
                threshold = float(predictor_threshold)
            except (TypeError, ValueError) as exc:
                raise ContractError(
                    f"{profile} learned config lacks predictor threshold"
                ) from exc
            if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
                raise ContractError(f"{profile} learned predictor threshold is invalid")
    training = _mapping(
        config.get("training"), description=f"{profile} training config"
    )
    stages = training.get("plan")
    if not isinstance(stages, list) or len(stages) != 6:
        raise ContractError(f"{profile} must contain exactly six training stages")
    checkpoints: dict[str, dict[str, Any]] = {}
    seen: set[str] = set()
    stage_contract: list[tuple[str, str]] = []
    for stage in stages:
        stage_config = _mapping(stage, description=f"{profile} training stage")
        name = str(stage_config.get("name", "")).strip()
        mode = str(stage_config.get("mode", "")).strip().lower()
        if not name or name in seen or mode not in {"neural_ode", "score_matching"}:
            raise ContractError(f"{profile} contains an invalid training stage")
        seen.add(name)
        stage_contract.append((name, mode))
        filename = (
            "score_model.pth"
            if mode == "score_matching"
            else (
                "last_model.pth"
                if str(stage_config.get("save_strategy", "best")).strip().lower()
                == "last"
                else "best_model.pth"
            )
        )
        checkpoints[name] = _file_identity(
            model_dir / name / filename,
            description=f"{profile} checkpoint {name}",
        )
    return {
        "profile": profile,
        "interaction_mode": interaction_mode,
        "model_dir": str(model_dir.resolve()),
        "source_config": dict(source_config_identity),
        "resolved_config": _file_identity(
            config_path, description=f"{profile} resolved config"
        ),
        "training_summary": _file_identity(
            summary_path, description=f"{profile} training summary"
        ),
        "stage_count": 6,
        "checkpoints": checkpoints,
        "inference_contract": {
            "components": [
                "velocity",
                "growth",
                "score",
                *([] if arm == "no_interaction" else ["interaction"]),
            ],
            "weight_stage": next(
                name
                for name, mode in reversed(stage_contract)
                if mode != "score_matching"
            ),
            "score_stage": next(
                name
                for name, mode in reversed(stage_contract)
                if mode == "score_matching"
            ),
            "interaction_m": None if arm == "no_interaction" else INTERACTION_M,
            "interaction_grouping_seed": (
                None if arm == "no_interaction" else INTERACTION_GROUPING_SEED
            ),
        },
    }


def _code_identity(
    release_root: Path, paths: Sequence[str], *, name: str
) -> dict[str, Any]:
    files = {
        relative: _file_identity(
            release_root / relative, description=f"{name} implementation {relative}"
        )
        for relative in paths
    }
    aggregate = _stable_json_sha256(
        {relative: record["sha256"] for relative, record in files.items()}
    )
    return {"schema_version": "1.0.0", "files": files, "aggregate_sha256": aggregate}


def _verify_python_binding(record: Any) -> str:
    identity = _mapping(record, description="launcher Python identity")
    invocation = identity.get("invocation_path")
    if not isinstance(invocation, str) or not invocation:
        raise ContractError("Launcher Python invocation path is invalid")
    if _python_executable_identity(invocation) != dict(identity):
        raise ContractError("Launcher Python lexical/runtime identity changed")
    return invocation


def _command_record(
    argv: Sequence[str], environment: Mapping[str, str], *, cwd: Path
) -> dict[str, Any]:
    env_words = [
        f"{key}={shlex.quote(value)}" for key, value in sorted(environment.items())
    ]
    command = " ".join(
        ["env", *env_words, *(shlex.quote(str(value)) for value in argv)]
    )
    shell = f"cd {shlex.quote(str(cwd))} && {command}"
    return {
        "argv": [str(value) for value in argv],
        "environment": dict(environment),
        "cwd": str(cwd),
        "shell": shell,
    }


def _profile_environment(
    *, profile: str, gpu: int, release_root: Path, evaluation_root: Path
) -> dict[str, str]:
    cache = evaluation_root / CONTRACT_DIR / "cache" / profile
    return {
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "CYTOBRIDGE_ASSIGNED_GPU": str(gpu),
        "PYTHONHASHSEED": str(MATCHED_SEED),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONUNBUFFERED": "1",
        "JUPYTER_PLATFORM_DIRS": "1",
        "PYTHONPATH": str(release_root),
        "NUMBA_CACHE_DIR": str(cache / "numba"),
        "MPLCONFIGDIR": str(cache / "matplotlib"),
        "XDG_CACHE_HOME": str(cache / "xdg"),
    }


def _score_environment(*, release_root: Path, evaluation_root: Path) -> dict[str, str]:
    cache = evaluation_root / CONTRACT_DIR / "cache" / "scoring"
    return {
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONHASHSEED": str(MATCHED_SEED),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONUNBUFFERED": "1",
        "JUPYTER_PLATFORM_DIRS": "1",
        "PYTHONPATH": str(release_root),
        "NUMBA_CACHE_DIR": str(cache / "numba"),
        "MPLCONFIGDIR": str(cache / "matplotlib"),
        "XDG_CACHE_HOME": str(cache / "xdg"),
    }


def build_plan(
    *,
    evaluation_root: str | Path,
    launcher_manifest: str | Path,
    expected_launcher_sha256: str,
    acceptance_report: str | Path,
    expected_acceptance_sha256: str,
    benchmark_inputs: Mapping[str, str | Path],
    expected_benchmark_sha256: Mapping[str, str],
) -> dict[str, Any]:
    manifest_path, launcher_sha, launcher, launcher_root = _verify_launcher_manifest(
        launcher_manifest, expected_launcher_sha256
    )
    acceptance_path, acceptance_sha, _ = _verify_acceptance(
        acceptance_report,
        expected_acceptance_sha256,
        launcher_root=launcher_root,
    )
    root = Path(evaluation_root).expanduser().resolve()
    if root.parent == root:
        raise ContractError("Refusing a filesystem root as evaluation root")
    release = _mapping(launcher.get("release"), description="launcher release")
    release_root = Path(str(release.get("root", ""))).expanduser().resolve()
    if not release_root.is_dir():
        raise FileNotFoundError(f"Launcher release root is missing: {release_root}")
    for disallowed in (launcher_root, release_root):
        nested = False
        for left, right in ((root, disallowed), (disallowed, root)):
            try:
                left.relative_to(right)
            except ValueError:
                continue
            nested = True
        if nested:
            raise ContractError(
                "Evaluation root must be disjoint from launcher/release roots"
            )
    python = _verify_python_binding(release.get("python_executable"))
    adapter_code = _code_identity(release_root, ADAPTER_FILES, name="adapter")
    evaluator_code = _code_identity(release_root, EVALUATOR_FILES, name="evaluator")
    sources = _mapping(launcher.get("sources"), description="launcher sources")
    conditions = _mapping(launcher.get("conditions"), description="launcher conditions")

    input_records: dict[str, dict[str, Any]] = {}
    for dataset in DATASET_ORDER:
        source = _mapping(
            sources.get(dataset), description=f"launcher source {dataset}"
        )
        aligned = _mapping(
            source.get("aligned_h5ad"), description=f"{dataset} aligned source"
        )
        _identity_matches(aligned, description=f"{dataset} launcher aligned H5AD")
        input_records[dataset] = _verify_benchmark_input(
            dataset,
            benchmark_inputs[dataset],
            expected_benchmark_sha256[dataset],
            launcher_aligned_sha256=str(aligned["sha256"]),
        )

    profiles: dict[str, dict[str, Any]] = {}
    for dataset in DATASET_ORDER:
        for arm in ARM_ORDER:
            profile = profile_name(dataset, arm)
            condition = _mapping(
                conditions[profile], description=f"condition {profile}"
            )
            paths = _mapping(condition.get("paths"), description=f"{profile} paths")
            config_identity = _mapping(
                condition.get("training_config"),
                description=f"{profile} training config",
            )
            inventory = _checkpoint_inventory(
                profile,
                arm,
                Path(str(paths.get("training", ""))).resolve(),
                config_identity,
            )
            input_link = Path(str(paths.get("aligned_h5ad", "")))
            if (
                not input_link.is_file()
                or _sha256(input_link.resolve())
                != input_records[dataset]["source_aligned_sha256"]
            ):
                raise ContractError(f"{profile} aligned input link changed")
            gpu = condition.get("gpu")
            if isinstance(gpu, bool) or not isinstance(gpu, int) or gpu < 0:
                raise ContractError(f"{profile} launcher GPU is invalid")
            predictions = root / "predictions" / dataset / arm
            scores = root / "scores" / dataset / arm
            environment = _profile_environment(
                profile=profile,
                gpu=gpu,
                release_root=release_root,
                evaluation_root=root,
            )
            infer_argv = [
                python,
                "-m",
                "scripts.spatiotemporal_benchmark.cytobridge.run_cytobridge",
                "infer-full",
                "--repo",
                str(release_root),
                "--input-manifest",
                input_records[dataset]["manifest"]["path"],
                "--split",
                "full_data",
                "--model-dir",
                inventory["model_dir"],
                "--training-config",
                config_identity["path"],
                "--output-dir",
                str(predictions),
                "--device",
                "cuda:0",
                "--prediction-n",
                str(PREDICTION_N),
                "--seed",
                str(MATCHED_SEED),
                "--sigma",
                str(INFERENCE_SIGMA),
                "--dt",
                str(INFERENCE_DT),
                "--interaction-m",
                str(INTERACTION_M),
            ]
            score_argv = [
                python,
                "-m",
                "scripts.spatiotemporal_benchmark.evaluate_predictions",
                "--input-manifest",
                input_records[dataset]["manifest"]["path"],
                "--predictions-root",
                str(predictions),
                "--track",
                "full_data",
                "--output-dir",
                str(scores),
                "--methods",
                METHOD,
                "--n-projections",
                str(N_PROJECTIONS),
                "--projection-repeats",
                str(PROJECTION_REPEATS),
                "--max-ot-points",
                str(MAX_OT_POINTS),
            ]
            profiles[profile] = {
                "dataset": dataset,
                "arm": arm,
                "arm_label": ARM_LABEL[arm],
                "interaction_mode": ARM_INTERACTION_MODE[arm],
                "gpu": gpu,
                "input_link": str(input_link.absolute()),
                "input_binding": {
                    "aligned_h5ad_sha256": input_records[dataset][
                        "source_aligned_sha256"
                    ],
                    "benchmark_manifest_sha256": input_records[dataset]["manifest"][
                        "sha256"
                    ],
                    "full_data_train_h5ad_sha256": input_records[dataset]["artifacts"][
                        "train_h5ad"
                    ]["sha256"],
                    "source_roster_sha256": input_records[dataset]["artifacts"][
                        "train_source_roster_npz"
                    ]["sha256"],
                },
                "inventory": inventory,
                "paths": {
                    "predictions": str(predictions),
                    "scores": str(scores),
                    "cache": str(root / CONTRACT_DIR / "cache" / profile),
                },
                "commands": {
                    "infer_full": _command_record(
                        infer_argv, environment, cwd=release_root
                    ),
                    "score_full_data": _command_record(
                        score_argv,
                        _score_environment(
                            release_root=release_root, evaluation_root=root
                        ),
                        cwd=release_root,
                    ),
                },
            }
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": PLAN_KIND,
        "created_at_utc": _utc_now(),
        "evaluation_root": str(root),
        "launcher": {
            "manifest": _file_identity(
                manifest_path, description="matched launcher manifest"
            ),
            "acceptance": _file_identity(
                acceptance_path, description="matched-family acceptance report"
            ),
            "run_root": str(launcher_root),
            "release_root": str(release_root),
            "release_commit": release.get("commit"),
            "python_executable": release.get("python_executable"),
            "launcher_manifest_sha256": launcher_sha,
            "acceptance_sha256": acceptance_sha,
        },
        "matrix": {
            "datasets": list(DATASET_ORDER),
            "arms": list(ARM_ORDER),
            "profiles": list(PROFILE_ORDER),
            "profile_count": 12,
            "target_prediction_count": sum(
                len(FULL_DATA_TARGETS[dataset]) for dataset in DATASET_ORDER
            )
            * len(ARM_ORDER),
            "protocol": MATCHED_PROTOCOL,
            "seed": MATCHED_SEED,
            "track": "full_data",
            "scientific_scope": "in-sample reconstruction; never LOTO generalization",
        },
        "benchmark_inputs": input_records,
        "implementation": {
            "adapter": adapter_code,
            "evaluator": evaluator_code,
            "orchestrator": _file_identity(
                Path(__file__).resolve(), description="evaluation orchestrator"
            ),
        },
        "settings": {
            "prediction_n": PREDICTION_N,
            "seed": MATCHED_SEED,
            "sigma": INFERENCE_SIGMA,
            "dt": INFERENCE_DT,
            "interaction_m": INTERACTION_M,
            "n_projections": N_PROJECTIONS,
            "projection_repeats": PROJECTION_REPEATS,
            "max_ot_points": MAX_OT_POINTS,
            "spaces": list(SPACE_ORDER),
            "metrics": list(METRIC_ORDER),
            "primary_metric": "sliced_w2",
            "paired_delta": "ablation minus full; positive means lower full-model error",
        },
        "profiles": profiles,
    }


def prepare_run_root(plan: Mapping[str, Any]) -> tuple[Path, str]:
    root = Path(str(plan["evaluation_root"]))
    if root.exists():
        raise FileExistsError(f"Refusing existing evaluation root: {root}")
    if not root.parent.is_dir():
        raise FileNotFoundError(f"Evaluation-root parent does not exist: {root.parent}")
    root.mkdir(mode=0o750)
    contract = root / CONTRACT_DIR
    contract.mkdir(mode=0o750)
    cache = contract / "cache"
    cache.mkdir(mode=0o750)
    for profile in (*PROFILE_ORDER, "scoring"):
        profile_cache = cache / profile
        profile_cache.mkdir(mode=0o750)
        for child in ("numba", "matplotlib", "xdg"):
            (profile_cache / child).mkdir(mode=0o750)
    payload = _canonical_json_bytes(dict(plan))
    digest = hashlib.sha256(payload).hexdigest()
    plan_path = contract / PLAN_NAME
    sidecar = contract / PLAN_SIDECAR
    with plan_path.open("xb") as handle:
        handle.write(payload)
    with sidecar.open("x", encoding="ascii") as handle:
        handle.write(f"{digest}  {PLAN_NAME}\n")
    plan_path.chmod(0o440)
    sidecar.chmod(0o440)
    return root, digest


def _load_plan(run_root: str | Path) -> tuple[Path, dict[str, Any], str]:
    root = Path(run_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Evaluation root is missing: {root}")
    plan_path = root / CONTRACT_DIR / PLAN_NAME
    sidecar = root / CONTRACT_DIR / PLAN_SIDECAR
    if not plan_path.is_file() or not sidecar.is_file():
        raise FileNotFoundError("Prepared evaluation plan or sidecar is missing")
    digest = _sha256(plan_path)
    if sidecar.read_text(encoding="ascii").strip().split() != [digest, PLAN_NAME]:
        raise ContractError("Prepared evaluation-plan sidecar mismatch")
    plan = _read_json(plan_path, description="prepared evaluation plan")
    if (
        plan.get("schema_version") != SCHEMA_VERSION
        or plan.get("kind") != PLAN_KIND
        or Path(str(plan.get("evaluation_root", ""))).resolve() != root
    ):
        raise ContractError("Prepared evaluation plan identity is invalid")
    if _mapping(plan.get("matrix"), description="evaluation matrix").get(
        "profiles"
    ) != list(PROFILE_ORDER):
        raise ContractError("Prepared evaluation profile matrix drifted")
    matrix = _mapping(plan.get("matrix"), description="evaluation matrix")
    if (
        matrix.get("datasets") != list(DATASET_ORDER)
        or matrix.get("arms") != list(ARM_ORDER)
        or matrix.get("profile_count") != 12
        or matrix.get("target_prediction_count") != 39
    ):
        raise ContractError("Prepared evaluation matrix dimensions drifted")
    settings = _mapping(plan.get("settings"), description="evaluation settings")
    expected_settings = {
        "prediction_n": PREDICTION_N,
        "seed": MATCHED_SEED,
        "sigma": INFERENCE_SIGMA,
        "dt": INFERENCE_DT,
        "interaction_m": INTERACTION_M,
        "n_projections": N_PROJECTIONS,
        "projection_repeats": PROJECTION_REPEATS,
        "max_ot_points": MAX_OT_POINTS,
        "spaces": list(SPACE_ORDER),
        "metrics": list(METRIC_ORDER),
        "primary_metric": "sliced_w2",
        "paired_delta": "ablation minus full; positive means lower full-model error",
    }
    if dict(settings) != expected_settings:
        raise ContractError("Prepared evaluation scientific settings drifted")
    return root, plan, digest


def verify_prepared_plan(run_root: str | Path) -> tuple[Path, dict[str, Any], str]:
    root, plan, digest = _load_plan(run_root)
    launcher = _mapping(plan.get("launcher"), description="plan launcher")
    _identity_matches(launcher.get("manifest"), description="launcher manifest")
    _identity_matches(launcher.get("acceptance"), description="acceptance report")
    _, launcher_sha, _, launcher_root = _verify_launcher_manifest(
        _mapping(launcher.get("manifest"), description="launcher manifest identity")[
            "path"
        ],
        launcher["launcher_manifest_sha256"],
    )
    if launcher_sha != launcher["launcher_manifest_sha256"]:
        raise ContractError("Launcher manifest digest drifted")
    _verify_acceptance(
        _mapping(launcher.get("acceptance"), description="acceptance identity")["path"],
        launcher["acceptance_sha256"],
        launcher_root=launcher_root,
    )
    _verify_python_binding(launcher.get("python_executable"))
    implementation = _mapping(plan.get("implementation"), description="implementation")
    for family in ("adapter", "evaluator"):
        code = _mapping(implementation.get(family), description=f"{family} code")
        for relative, record in _mapping(
            code.get("files"), description=f"{family} files"
        ).items():
            _identity_matches(record, description=f"{family} file {relative}")
        observed_aggregate = _stable_json_sha256(
            {
                relative: _mapping(record, description="code file")["sha256"]
                for relative, record in code["files"].items()
            }
        )
        if observed_aggregate != code.get("aggregate_sha256"):
            raise ContractError(f"{family} aggregate implementation hash drifted")
    _identity_matches(implementation.get("orchestrator"), description="orchestrator")
    for dataset, input_record in _mapping(
        plan.get("benchmark_inputs"), description="benchmark inputs"
    ).items():
        record = _mapping(input_record, description=f"{dataset} benchmark input")
        _identity_matches(record.get("manifest"), description=f"{dataset} manifest")
        _identity_matches(
            record.get("manifest_sidecar"), description=f"{dataset} manifest sidecar"
        )
        _identity_matches(
            record.get("config_source"), description=f"{dataset} config source"
        )
        _identity_matches(
            record.get("resolved_config"), description=f"{dataset} resolved config"
        )
        for name, artifact in _mapping(
            record.get("artifacts"), description="artifacts"
        ).items():
            _identity_matches(artifact, description=f"{dataset} artifact {name}")
    profiles = _mapping(plan.get("profiles"), description="plan profiles")
    if set(profiles) != set(PROFILE_ORDER):
        raise ContractError("Prepared plan profile set changed")
    for profile, profile_record in profiles.items():
        record = _mapping(profile_record, description=f"profile {profile}")
        inventory = _mapping(
            record.get("inventory"), description=f"{profile} inventory"
        )
        _identity_matches(
            inventory.get("source_config"), description=f"{profile} source config"
        )
        _identity_matches(
            inventory.get("resolved_config"), description=f"{profile} resolved config"
        )
        _identity_matches(
            inventory.get("training_summary"), description=f"{profile} training summary"
        )
        for stage, checkpoint in _mapping(
            inventory.get("checkpoints"), description=f"{profile} checkpoints"
        ).items():
            _identity_matches(checkpoint, description=f"{profile} checkpoint {stage}")
        input_link = Path(str(record.get("input_link", "")))
        dataset = str(record["dataset"])
        expected_input_binding = {
            "aligned_h5ad_sha256": plan["benchmark_inputs"][dataset][
                "source_aligned_sha256"
            ],
            "benchmark_manifest_sha256": plan["benchmark_inputs"][dataset]["manifest"][
                "sha256"
            ],
            "full_data_train_h5ad_sha256": plan["benchmark_inputs"][dataset][
                "artifacts"
            ]["train_h5ad"]["sha256"],
            "source_roster_sha256": plan["benchmark_inputs"][dataset]["artifacts"][
                "train_source_roster_npz"
            ]["sha256"],
        }
        if record.get("input_binding") != expected_input_binding:
            raise ContractError(f"{profile} input binding changed")
        if (
            not input_link.is_file()
            or _sha256(input_link.resolve())
            != plan["benchmark_inputs"][dataset]["source_aligned_sha256"]
        ):
            raise ContractError(f"{profile} aligned input link changed")
    return root, plan, digest


def _find_summary(prediction: Path) -> tuple[Path, dict[str, Any]]:
    canonical = prediction.parent / "summary.json"
    adjacent = prediction.with_suffix(".summary.json")
    if not canonical.is_file() or not adjacent.is_file():
        raise ContractError(f"Prediction summary pair is missing: {prediction.parent}")
    if canonical.read_bytes() != adjacent.read_bytes():
        raise ContractError(f"Prediction summary pair differs: {prediction.parent}")
    return canonical, _read_json(canonical, description="prediction summary")


def _sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _finite_matrix(archive: Any, key: str, *, description: str) -> np.ndarray:
    if key not in archive:
        raise ContractError(f"{description} lacks {key!r}")
    value = np.asarray(archive[key], dtype=np.float64)
    if value.ndim != 2 or value.shape[0] == 0 or value.shape[1] == 0:
        raise ContractError(f"{description} {key!r} is not a non-empty matrix")
    if not np.isfinite(value).all():
        raise ContractError(f"{description} {key!r} contains non-finite values")
    return value


def _training_reference_arrays(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            required = {"state", "spatial", "time", "row_id"}
            if not required <= set(archive.files):
                raise ContractError("Training reference lacks frozen array keys")
            state = _finite_matrix(archive, "state", description="training reference")
            spatial = _finite_matrix(
                archive, "spatial", description="training reference"
            )
            time = np.asarray(archive["time"], dtype=np.float64).reshape(-1)
            row_id = np.asarray(archive["row_id"]).astype(str)
    except (OSError, ValueError) as exc:
        raise ContractError(f"Cannot load training reference {path}: {exc}") from exc
    if (
        state.shape[0] != spatial.shape[0]
        or time.shape != (state.shape[0],)
        or row_id.shape != (state.shape[0],)
        or not np.isfinite(time).all()
        or len(set(row_id)) != len(row_id)
    ):
        raise ContractError("Training reference arrays have inconsistent rows")
    return {"state": state, "spatial": spatial, "time": time, "row_id": row_id}


def _truth_arrays(path: Path, *, target: int) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            state = _finite_matrix(archive, "state", description=f"truth t{target}")
            spatial = _finite_matrix(archive, "spatial", description=f"truth t{target}")
    except (OSError, ValueError) as exc:
        raise ContractError(f"Cannot load truth t{target} {path}: {exc}") from exc
    if state.shape[0] != spatial.shape[0]:
        raise ContractError(f"Truth t{target} state/spatial row counts differ")
    return {"state": state, "spatial": spatial}


def _float_matches(value: Any, expected: float, *, atol: float = 1e-12) -> bool:
    try:
        observed = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(observed) and math.isclose(
        observed, float(expected), rel_tol=0.0, abs_tol=atol
    )


def _validate_output_roster(
    *,
    prediction_root: Path,
    roster_record: Any,
    expected_roster: Mapping[str, Any],
    training: Mapping[str, np.ndarray],
) -> tuple[dict[str, Any], dict[str, Any]]:
    roster = _mapping(roster_record, description="prediction source roster")
    required = {
        "source_roster",
        "source_roster_sha256",
        "source_indices_sha256",
        "source_row_id_sha256",
        "source_time",
        "source_available_n",
        "prediction_n",
        "sampled_with_replacement",
        "canonical_input_roster",
        "canonical_input_roster_sha256",
        "shared_across_all_benchmark_families",
    }
    if set(roster) != required:
        raise ContractError("Prediction source-roster summary schema drifted")
    output_path = prediction_root / "source_roster.npz"
    canonical_path = Path(str(expected_roster["path"])).resolve()
    if (
        Path(str(roster.get("source_roster", ""))).resolve() != output_path.resolve()
        or Path(str(roster.get("canonical_input_roster", ""))).resolve()
        != canonical_path
        or roster.get("source_roster_sha256") != expected_roster["sha256"]
        or roster.get("canonical_input_roster_sha256") != expected_roster["sha256"]
        or roster.get("prediction_n") != PREDICTION_N
        or roster.get("shared_across_all_benchmark_families") is not True
        or not _float_matches(roster.get("source_time"), 0.0)
    ):
        raise ContractError("Prediction source roster is not path/hash bound")
    output_identity = _file_identity(output_path, description="output source roster")
    if output_identity["sha256"] != expected_roster["sha256"]:
        raise ContractError("Output roster is not a byte copy of canonical input")
    try:
        with np.load(output_path, allow_pickle=False) as archive:
            required_arrays = {
                "indices",
                "row_id",
                "source_time",
                "state",
                "spatial",
                "support_row_id",
                "support_indices",
            }
            if set(archive.files) != required_arrays:
                raise ContractError("Source roster NPZ schema drifted")
            indices = np.asarray(archive["indices"], dtype=np.int64)
            row_id = np.asarray(archive["row_id"]).astype(str)
            source_time = np.asarray(archive["source_time"], dtype=np.float64).reshape(
                -1
            )
            state = np.asarray(archive["state"], dtype=np.float64)
            spatial = np.asarray(archive["spatial"], dtype=np.float64)
    except (OSError, ValueError) as exc:
        raise ContractError(f"Cannot load output source roster: {exc}") from exc
    if (
        indices.shape != (PREDICTION_N,)
        or row_id.shape != (PREDICTION_N,)
        or source_time.shape != (1,)
        or state.shape != (PREDICTION_N, training["state"].shape[1])
        or spatial.shape != (PREDICTION_N, training["spatial"].shape[1])
        or np.any(indices < 0)
        or np.any(indices >= training["state"].shape[0])
        or not _float_matches(source_time[0], 0.0)
    ):
        raise ContractError("Output source roster shapes/indices/time are invalid")
    if (
        not np.array_equal(row_id, training["row_id"][indices])
        or not np.allclose(state, training["state"][indices], rtol=1e-6, atol=1e-6)
        or not np.allclose(spatial, training["spatial"][indices], rtol=1e-6, atol=1e-6)
        or not np.allclose(training["time"][indices], 0.0, rtol=0.0, atol=1e-8)
    ):
        raise ContractError("Output source roster rows differ from training reference")
    expected_indices_sha = _sha256_array(indices)
    expected_row_sha = _sha256_array(training["row_id"][indices].astype("U"))
    available = int(np.count_nonzero(np.isclose(training["time"], 0.0)))
    sampled_with_replacement = len(np.unique(indices)) < len(indices)
    if (
        roster.get("source_indices_sha256") != expected_indices_sha
        or roster.get("source_row_id_sha256") != expected_row_sha
        or roster.get("source_available_n") != available
        or roster.get("sampled_with_replacement") is not sampled_with_replacement
    ):
        raise ContractError("Prediction source-roster array evidence is invalid")
    return dict(roster), output_identity


def _prediction_arrays(
    path: Path,
    *,
    state_dim: int,
    spatial_dim: int,
    target: int,
) -> dict[str, Any]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            required = {"state", "spatial", "weights", "source_time", "target_time"}
            if set(archive.files) != required:
                raise ContractError(f"Prediction t{target} NPZ schema drifted")
            state = np.asarray(archive["state"], dtype=np.float64)
            spatial = np.asarray(archive["spatial"], dtype=np.float64)
            weights = np.asarray(archive["weights"], dtype=np.float64).reshape(-1)
            source_time = np.asarray(archive["source_time"], dtype=np.float64).reshape(
                -1
            )
            target_time = np.asarray(archive["target_time"], dtype=np.float64).reshape(
                -1
            )
    except (OSError, ValueError) as exc:
        raise ContractError(f"Cannot load prediction t{target}: {exc}") from exc
    if (
        state.shape != (PREDICTION_N, state_dim)
        or spatial.shape != (PREDICTION_N, spatial_dim)
        or weights.shape != (PREDICTION_N,)
        or source_time.shape != (1,)
        or target_time.shape != (1,)
        or not np.isfinite(state).all()
        or not np.isfinite(spatial).all()
        or not np.isfinite(weights).all()
        or np.any(weights < 0)
        or weights.sum() <= 0
        or not _float_matches(source_time[0], 0.0)
        or not _float_matches(target_time[0], target)
    ):
        raise ContractError(f"Prediction t{target} arrays violate native contract")
    return {
        "n_predicted": int(state.shape[0]),
        "predicted_mass": float(weights.sum()),
        "state_dim": state_dim,
        "spatial_dim": spatial_dim,
    }


def _validate_prediction_outputs(
    plan: Mapping[str, Any], profile: str, record: Mapping[str, Any]
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    dataset = str(record["dataset"])
    arm = str(record["arm"])
    targets = list(plan["benchmark_inputs"][dataset]["targets"])
    prediction_root = Path(str(record["paths"]["predictions"]))
    run_summary_path = prediction_root / "run_summary.json"
    run_summary = _read_json(run_summary_path, description=f"{profile} run summary")
    manifest_sha = plan["benchmark_inputs"][dataset]["manifest"]["sha256"]
    source_roster_sha = plan["benchmark_inputs"][dataset]["artifacts"][
        "train_source_roster_npz"
    ]["sha256"]
    expected_input = plan["benchmark_inputs"][dataset]
    training_identity = expected_input["artifacts"]["train_training_reference_npz"]
    training = _training_reference_arrays(Path(training_identity["path"]))
    roster, roster_identity = _validate_output_roster(
        prediction_root=prediction_root,
        roster_record=run_summary.get("source_roster"),
        expected_roster=expected_input["artifacts"]["train_source_roster_npz"],
        training=training,
    )
    if (
        run_summary.get("status") != "complete"
        or run_summary.get("method") != METHOD
        or run_summary.get("regime") != "full_data"
        or run_summary.get("split_id") != "full_data"
        or run_summary.get("source_time") != 0
        or run_summary.get("targets") != targets
        or run_summary.get("single_continuous_non_split_call") is not True
        or run_summary.get("intermediate_reset") is not False
        or run_summary.get("spatial_warp_applied") is not False
        or run_summary.get("prediction_n") != PREDICTION_N
        or run_summary.get("seed") != MATCHED_SEED
        or run_summary.get("input_manifest_sha256") != manifest_sha
    ):
        raise ContractError(f"{profile} full-data run summary contract is invalid")
    compact = run_summary.get("prediction_summaries")
    if (
        not isinstance(compact, list)
        or any(not isinstance(item, Mapping) for item in compact)
        or [item.get("target") for item in compact] != targets
    ):
        raise ContractError(f"{profile} compact prediction target list drifted")
    inventory = _mapping(record.get("inventory"), description=f"{profile} inventory")
    expected_checkpoints = {
        stage: checkpoint["sha256"]
        for stage, checkpoint in inventory["checkpoints"].items()
    }
    expected_adapter = plan["implementation"]["adapter"]
    expected_simulation = inventory["inference_contract"]
    expected_repo = {
        "repo": plan["launcher"]["release_root"],
        "git_commit": plan["launcher"]["release_commit"],
        "git_dirty": False,
    }
    expected_provenance = {
        "input_manifest": expected_input["manifest"]["path"],
        "input_manifest_sha256": manifest_sha,
        "train_h5ad": expected_input["artifacts"]["train_h5ad"]["path"],
        "train_h5ad_sha256": expected_input["artifacts"]["train_h5ad"]["sha256"],
        "training_reference_npz": training_identity["path"],
        "training_reference_sha256": training_identity["sha256"],
        "source_roster_npz": expected_input["artifacts"]["train_source_roster_npz"][
            "path"
        ],
        "source_roster_sha256": source_roster_sha,
        "truth_inputs_opened": False,
    }
    for key, expected in expected_provenance.items():
        observed = run_summary.get(key)
        if key.endswith(("manifest", "h5ad", "npz")):
            if Path(str(observed)).resolve() != Path(str(expected)).resolve():
                raise ContractError(f"{profile} run summary {key} is not input-bound")
        elif observed != expected:
            raise ContractError(f"{profile} run summary {key} is not input-bound")
    outputs: dict[int, dict[str, Any]] = {}
    official_signature: str | None = None
    for target in targets:
        prediction = prediction_root / f"t{target}" / "prediction.npz"
        if not prediction.is_file() or prediction.stat().st_size <= 0:
            raise FileNotFoundError(
                f"Missing {profile} prediction t{target}: {prediction}"
            )
        prediction_sha = _sha256(prediction)
        arrays = _prediction_arrays(
            prediction,
            state_dim=int(training["state"].shape[1]),
            spatial_dim=int(training["spatial"].shape[1]),
            target=target,
        )
        summary_path, summary = _find_summary(prediction)
        simulation = _mapping(
            summary.get("simulation"), description=f"{profile} simulation t{target}"
        )
        signature = simulation.get("official_api_signature")
        if not isinstance(signature, str) or not signature.startswith("("):
            raise ContractError(f"{profile} simulation API signature is missing")
        if official_signature is None:
            official_signature = signature
        elif signature != official_signature:
            raise ContractError(f"{profile} simulator signature changed across targets")
        expected_interaction = arm != "no_interaction"
        expected_edge_predictor = arm == "full"
        if (
            summary.get("status") != "complete"
            or summary.get("dataset") != dataset
            or summary.get("method") != METHOD
            or summary.get("regime") != "full_data"
            or summary.get("track") != "full_data"
            or summary.get("split_id") != "full_data"
            or summary.get("target") != target
            or summary.get("target_time") != target
            or summary.get("source_time") != 0
            or summary.get("output_scope") != "native_joint"
            or summary.get("native_vs_adapter") != "native_joint"
            or summary.get("spatial_warp_applied") is not False
            or summary.get("split_sde") is not False
            or summary.get("continuous_across_targets") is not True
            or summary.get("prediction_n") != PREDICTION_N
            or summary.get("seed") != MATCHED_SEED
            or summary.get("device") != "cuda:0"
            or summary.get("implementation") != EXPECTED_IMPLEMENTATION
            or summary.get("source_policy")
            != "fixed t0 bootstrap shared across all full-data targets; no intermediate reset"
            or summary.get("primary_benchmark_eligible") is not True
            or summary.get("native_joint") is not True
            or summary.get("native_mass") is not True
            or summary.get("native_growth") is not True
            or summary.get("weights_are_unnormalised") is not True
            or summary.get("spatial_warp") != "none"
            or summary.get("prediction_n_policy")
            != "fixed_from_train_contract_before_truth_access"
            or summary.get("truth_inputs_opened") is not False
            or not _float_matches(summary.get("alpha_express"), ALPHA_EXPRESS)
            or not _float_matches(summary.get("alpha_spatial"), ALPHA_SPATIAL)
            or not _float_matches(summary.get("sigma"), INFERENCE_SIGMA)
            or not _float_matches(summary.get("dt"), INFERENCE_DT)
            or summary.get("include_score") is not True
            or summary.get("include_interaction") is not expected_interaction
            or summary.get("interaction_mode") != ARM_INTERACTION_MODE[arm]
            or summary.get("edge_prior_mode") != ARM_INTERACTION_MODE[arm]
            or summary.get("edge_predictor_used") is not expected_edge_predictor
            or summary.get("interaction_m") != expected_simulation["interaction_m"]
            or not _float_matches(
                summary.get("predicted_mass"), arrays["predicted_mass"], atol=1e-10
            )
            or Path(str(summary.get("prediction_npz", ""))).resolve()
            != prediction.resolve()
            or summary.get("prediction_npz_sha256") != prediction_sha
            or summary.get("state_dim") != arrays["state_dim"]
            or summary.get("spatial_dim") != arrays["spatial_dim"]
            or summary.get("joint_dim") != arrays["state_dim"] + arrays["spatial_dim"]
            or summary.get("input_manifest_sha256") != manifest_sha
            or summary.get("config_sha256") != inventory["resolved_config"]["sha256"]
            or summary.get("checkpoint_sha256") != expected_checkpoints
            or summary.get("stage_complete") is not True
            or summary.get("stage_count") != 6
            or summary.get("source_roster") != roster
            or summary.get("repo") != expected_repo
            or simulation.get("official_api") != OFFICIAL_SIMULATION_API
            or simulation.get("simulation_mode") != OFFICIAL_SIMULATION_MODE
            or simulation.get("weight_stage") != expected_simulation["weight_stage"]
            or simulation.get("score_stage") != expected_simulation["score_stage"]
            or simulation.get("interaction_mode") != ARM_INTERACTION_MODE[arm]
            or simulation.get("edge_prior_mode") != ARM_INTERACTION_MODE[arm]
            or simulation.get("include_interaction") is not expected_interaction
            or simulation.get("edge_predictor_used") is not expected_edge_predictor
            or simulation.get("interaction_m") != expected_simulation["interaction_m"]
            or simulation.get("loaded_model_interaction_group_size")
            != expected_simulation["interaction_m"]
            or simulation.get("interaction_group_binding")
            != (
                "exact_checkpoint_model_match"
                if expected_interaction
                else "not_applicable_no_interaction_component"
            )
            or simulation.get("interaction_grouping_seed")
            != expected_simulation["interaction_grouping_seed"]
            or simulation.get("stochastic_stream_contract")
            != STOCHASTIC_STREAM_CONTRACT
            or simulation.get("dynamics_components")
            != expected_simulation["components"]
            or simulation.get("weights_semantics") != WEIGHTS_SEMANTICS
        ):
            raise ContractError(f"{profile} prediction summary t{target} is invalid")
        for key, expected in expected_provenance.items():
            observed = summary.get(key)
            if key.endswith(("manifest", "h5ad", "npz")):
                if Path(str(observed)).resolve() != Path(str(expected)).resolve():
                    raise ContractError(
                        f"{profile} prediction t{target} {key} is not input-bound"
                    )
            elif observed != expected:
                raise ContractError(
                    f"{profile} prediction t{target} {key} is not input-bound"
                )
        training_match = _mapping(
            summary.get("training_reference_match"),
            description=f"{profile} training match t{target}",
        )
        proof = training_match.get("proof")
        if proof == "benchmark_fit_summary":
            if (
                training_match.get("training_reference_sha256")
                != training_identity["sha256"]
            ):
                raise ContractError(
                    f"{profile} prediction training proof is not input-bound"
                )
        elif proof == "saved_adata_exact_frozen_arrays":
            row_proof = training_match.get("row_identity_proof")
            if row_proof == "contracted_row_id_exact_order":
                row_identity = training["row_id"].astype("U")
            elif row_proof == "legacy_obs_names_vs_benchmark_original_obs_name":
                try:
                    import anndata as ad

                    adata = ad.read_h5ad(
                        expected_input["artifacts"]["train_h5ad"]["path"],
                        backed="r",
                    )
                    try:
                        if "benchmark_original_obs_name" not in adata.obs:
                            raise ContractError(
                                f"{profile} benchmark input lacks legacy row identity"
                            )
                        row_identity = (
                            adata.obs["benchmark_original_obs_name"]
                            .astype(str)
                            .to_numpy(dtype=str)
                            .astype("U")
                        )
                    finally:
                        if adata.file is not None:
                            adata.file.close()
                except (OSError, ValueError) as exc:
                    raise ContractError(
                        f"{profile} cannot verify legacy training row identity: {exc}"
                    ) from exc
            else:
                raise ContractError(
                    f"{profile} saved-adata proof has unsupported row identity"
                )
            expected_array_sha = {
                "state": _sha256_array(np.asarray(training["state"], dtype=np.float32)),
                "spatial": _sha256_array(
                    np.asarray(training["spatial"], dtype=np.float32)
                ),
                "time": _sha256_array(np.asarray(training["time"], dtype=np.float64)),
                "row_identity": _sha256_array(row_identity),
            }
            if training_match.get("array_sha256") != expected_array_sha:
                raise ContractError(
                    f"{profile} saved-adata proof differs from training reference"
                )
        else:
            raise ContractError(
                f"{profile} prediction lacks a supported training proof"
            )
        adapter = _mapping(
            summary.get("adapter_implementation"),
            description=f"{profile} adapter implementation",
        )
        if (
            adapter.get("schema_version") != expected_adapter["schema_version"]
            or adapter.get("aggregate_sha256") != expected_adapter["aggregate_sha256"]
            or adapter.get("files")
            != {
                path: identity["sha256"]
                for path, identity in expected_adapter["files"].items()
            }
        ):
            raise ContractError(f"{profile} prediction used another adapter build")
        outputs[target] = {
            "prediction": _file_identity(
                prediction, description=f"{profile} prediction t{target}"
            ),
            "summary": _file_identity(
                summary_path, description=f"{profile} summary t{target}"
            ),
            **arrays,
        }
        compact_record = _mapping(
            compact[targets.index(target)],
            description=f"{profile} compact summary t{target}",
        )
        if (
            set(compact_record)
            != {"target", "prediction_npz", "prediction_npz_sha256", "predicted_mass"}
            or Path(str(compact_record.get("prediction_npz", ""))).resolve()
            != prediction.resolve()
            or compact_record.get("prediction_npz_sha256") != prediction_sha
            or not _float_matches(
                compact_record.get("predicted_mass"),
                arrays["predicted_mass"],
                atol=1e-10,
            )
        ):
            raise ContractError(f"{profile} compact prediction t{target} is not bound")
    return outputs, {
        "run_summary": _file_identity(
            run_summary_path, description=f"{profile} run summary"
        ),
        "source_roster": roster_identity,
    }


def _numeric_finite(
    frame: pd.DataFrame, columns: Sequence[str], *, profile: str
) -> None:
    for column in columns:
        if column not in frame:
            raise ContractError(f"{profile} score table lacks {column}")
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ContractError(f"{profile} score column {column} is non-finite")
        frame[column] = values


def _expected_projection_seed(dataset: str, space: str, repeat: int) -> int:
    """Mirror the bound benchmark seed contract without importing mutable code."""

    canonical = json.dumps(
        {
            "namespace": BENCHMARK_PROJECTION_SEED_NAMESPACE,
            "benchmark": dataset,
            "split": "full_data",
            "space": space,
            "repeat": repeat,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return int.from_bytes(
        hashlib.sha256(canonical.encode("utf-8")).digest()[:4], "little"
    )


def _expected_projection_sha256(dimension: int, seed: int) -> str:
    """Recreate POT 0.9.x's canonical random projection basis."""

    if dimension <= 0:
        raise ContractError("Projection dimension must be positive")
    projections = np.random.RandomState(seed).randn(dimension, N_PROJECTIONS)
    projections = projections / np.sqrt(np.sum(projections**2, axis=0, keepdims=True))
    canonical = np.ascontiguousarray(np.asarray(projections, dtype="<f8"))
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def _exact_ot_seed(dataset: str, space: str) -> int:
    canonical = json.dumps(
        {
            "namespace": f"{BENCHMARK_PROJECTION_SEED_NAMESPACE}-exact-ot",
            "benchmark": dataset,
            "split": "full_data",
            "space": space,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return int.from_bytes(
        hashlib.sha256(canonical.encode("utf-8")).digest()[:4], "little"
    )


def _prediction_metric_arrays(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            state = np.asarray(archive["state"], dtype=np.float64)
            spatial = np.asarray(archive["spatial"], dtype=np.float64)
            weights = np.asarray(archive["weights"], dtype=np.float64).reshape(-1)
    except (OSError, KeyError, ValueError) as exc:
        raise ContractError(f"Cannot reload metric inputs from {path}: {exc}") from exc
    return {"state": state, "spatial": spatial, "weights": weights}


def _recompute_metric_values(
    *,
    dataset: str,
    prediction_path: Path,
    training: Mapping[str, np.ndarray],
    truth: Mapping[str, np.ndarray],
) -> pd.DataFrame:
    """Independently recompute every reported OT value from bound NPZ arrays."""

    try:
        import ot
        from scipy.spatial.distance import cdist
    except ImportError as exc:
        raise ContractError(
            "POT and SciPy are required for score revalidation"
        ) from exc

    prediction = _prediction_metric_arrays(prediction_path)
    state_center = np.asarray(training["state"], dtype=np.float64).mean(axis=0)
    state_scale = np.asarray(training["state"], dtype=np.float64).std(axis=0, ddof=0)
    spatial_center = np.asarray(training["spatial"], dtype=np.float64).mean(axis=0)
    spatial_rms = float(
        np.sqrt(
            np.mean(
                (np.asarray(training["spatial"], dtype=np.float64) - spatial_center)
                ** 2
            )
        )
    )
    state_dim = int(state_center.size)
    spatial_dim = int(spatial_center.size)

    def state_transform(values: np.ndarray) -> np.ndarray:
        return (
            (np.asarray(values, dtype=np.float64) - state_center)
            / state_scale
            / math.sqrt(state_dim)
        )

    def spatial_transform(values: np.ndarray) -> np.ndarray:
        return (
            (np.asarray(values, dtype=np.float64) - spatial_center)
            / spatial_rms
            / math.sqrt(spatial_dim)
        )

    predicted_state = state_transform(prediction["state"])
    observed_state = state_transform(truth["state"])
    predicted_spatial = spatial_transform(prediction["spatial"])
    observed_spatial = spatial_transform(truth["spatial"])
    spaces = {
        "joint": (
            np.concatenate((predicted_state, predicted_spatial), axis=1),
            np.concatenate((observed_state, observed_spatial), axis=1),
        ),
        "state": (predicted_state, observed_state),
        "spatial": (predicted_spatial, observed_spatial),
    }
    raw_weights = np.asarray(prediction["weights"], dtype=np.float64)
    normalized_weights = raw_weights / float(raw_weights.sum())
    uniform_weights = np.full(
        raw_weights.shape[0], 1.0 / raw_weights.shape[0], dtype=np.float64
    )
    uniform_predicted = np.allclose(
        normalized_weights, uniform_weights, rtol=1e-12, atol=1e-15
    )
    rows: list[dict[str, Any]] = []
    for space, (predicted, observed) in spaces.items():
        rng = np.random.default_rng(_exact_ot_seed(dataset, space))
        exact_predicted = predicted
        exact_weights = normalized_weights
        if predicted.shape[0] > MAX_OT_POINTS:
            indices = rng.choice(
                predicted.shape[0],
                size=MAX_OT_POINTS,
                replace=not uniform_predicted,
                p=None if uniform_predicted else normalized_weights,
            )
            exact_predicted = predicted[indices]
            exact_weights = np.full(
                MAX_OT_POINTS, 1.0 / MAX_OT_POINTS, dtype=np.float64
            )
        exact_observed = observed
        if observed.shape[0] > MAX_OT_POINTS:
            indices = rng.choice(observed.shape[0], size=MAX_OT_POINTS, replace=False)
            exact_observed = observed[indices]
        observed_weights = np.full(
            exact_observed.shape[0],
            1.0 / exact_observed.shape[0],
            dtype=np.float64,
        )
        distances = cdist(exact_predicted, exact_observed, metric="euclidean")
        exact_w1 = float(
            ot.emd2(exact_weights, observed_weights, distances, numItermax=int(1e7))
        )
        exact_w2_sq = float(
            ot.emd2(
                exact_weights,
                observed_weights,
                distances**2,
                numItermax=int(1e7),
            )
        )
        exact_w2 = math.sqrt(max(exact_w2_sq, 0.0))
        sliced_observed_weights = np.full(
            observed.shape[0], 1.0 / observed.shape[0], dtype=np.float64
        )
        for repeat in range(PROJECTION_REPEATS):
            seed = _expected_projection_seed(dataset, space, repeat)
            sliced_w2 = float(
                ot.sliced_wasserstein_distance(
                    predicted,
                    observed,
                    a=normalized_weights,
                    b=sliced_observed_weights,
                    n_projections=N_PROJECTIONS,
                    p=2,
                    seed=seed,
                    log=False,
                )
            )
            rows.append(
                {
                    "space": space,
                    "projection_repeat": repeat,
                    "sliced_w2": sliced_w2,
                    "exact_w1": exact_w1,
                    "exact_w2": exact_w2,
                }
            )
    return pd.DataFrame.from_records(rows)


def _expected_transform_bytes(training: Mapping[str, np.ndarray]) -> bytes:
    state = np.asarray(training["state"], dtype=np.float64)
    spatial = np.asarray(training["spatial"], dtype=np.float64)
    state_center = state.mean(axis=0)
    state_scale = state.std(axis=0, ddof=0)
    spatial_center = spatial.mean(axis=0)
    spatial_rms_scale = float(np.sqrt(np.mean((spatial - spatial_center) ** 2)))
    if (
        np.any(~np.isfinite(state_scale))
        or np.any(state_scale <= 0)
        or not math.isfinite(spatial_rms_scale)
        or spatial_rms_scale <= 0
    ):
        raise ContractError("Training reference cannot define the frozen transform")
    payload = {
        "schema_version": 1,
        "state_dim": int(state.shape[1]),
        "spatial_dim": int(spatial.shape[1]),
        "state_center": state_center.tolist(),
        "state_scale": state_scale.tolist(),
        "spatial_center": spatial_center.tolist(),
        "spatial_rms_scale": spatial_rms_scale,
    }
    return (
        json.dumps(payload, allow_nan=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _validate_score_outputs(
    plan: Mapping[str, Any],
    profile: str,
    record: Mapping[str, Any],
    prediction_outputs: Mapping[int, Mapping[str, Any]],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    dataset = str(record["dataset"])
    targets = list(plan["benchmark_inputs"][dataset]["targets"])
    score_root = Path(str(record["paths"]["scores"]))
    csv_path = score_root / "full_data_metrics_long.csv"
    status_path = score_root / "full_data_method_target_status.csv"
    manifest_path = score_root / "full_data_evaluation_manifest.json"
    transform_path = score_root / "transforms" / "full_data.json"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing {profile} score table: {csv_path}")
    manifest = _read_json(manifest_path, description=f"{profile} evaluation manifest")
    expected_manifest = plan["benchmark_inputs"][dataset]["manifest"]
    if (
        manifest.get("schema_version") != "1.0.0"
        or manifest.get("status") != "complete"
        or manifest.get("track") != "full_data"
        or Path(str(manifest.get("input_manifest", ""))).resolve()
        != Path(expected_manifest["path"])
        or manifest.get("input_manifest_sha256") != expected_manifest["sha256"]
        or Path(str(manifest.get("predictions_root", ""))).resolve()
        != Path(str(record["paths"]["predictions"]))
        or manifest.get("n_projections") != N_PROJECTIONS
        or manifest.get("projection_repeats") != PROJECTION_REPEATS
        or manifest.get("max_ot_points") != MAX_OT_POINTS
        or manifest.get("methods") != [METHOD]
        or manifest.get("completed_methods") != [METHOD]
        or manifest.get("targets") != targets
        or manifest.get("spaces") != sorted(SPACE_ORDER)
        or manifest.get("status_table_source") is not None
        or Path(str(manifest.get("method_target_status_csv", ""))).resolve()
        != status_path
        or Path(str(manifest.get("metrics_long_csv", ""))).resolve() != csv_path
        or manifest.get("metrics_long_csv_sha256") != _sha256(csv_path)
        or manifest.get("n_rows")
        != len(targets) * len(SPACE_ORDER) * PROJECTION_REPEATS
    ):
        raise ContractError(f"{profile} evaluation manifest contract is invalid")
    if not status_path.is_file():
        raise FileNotFoundError(f"Missing {profile} method-target status table")
    status = pd.read_csv(status_path, keep_default_na=False)
    if (
        list(status.columns) != ["track", "target", "method", "status", "reason"]
        or len(status) != len(targets)
        or set(status["target"].astype(int)) != set(targets)
        or set(status["track"].astype(str)) != {"full_data"}
        or set(status["method"].astype(str)) != {METHOD}
        or set(status["status"].astype(str)) != {"completed"}
        or set(status["reason"].astype(str)) != {""}
        or manifest.get("method_target_status") != status.to_dict(orient="records")
    ):
        raise ContractError(f"{profile} method-target status is not complete")
    frame = pd.read_csv(csv_path)
    required = {
        "track",
        "target",
        "source_time",
        "benchmark",
        "split",
        "method",
        "space",
        "projection_repeat",
        "projection_seed",
        "projection_sha256",
        "n_projections",
        "primary_metric",
        "primary_value",
        "sliced_w2",
        "exact_w1",
        "exact_w2",
        "n_predicted",
        "n_observed",
        "predicted_weight_sum",
        "exact_ot_predicted_points",
        "exact_ot_observed_points",
        "output_scope",
        "native_vs_adapter",
        "prediction_path",
        "prediction_sha256",
        "prediction_summary",
        "prediction_summary_sha256",
        "input_manifest",
        "input_manifest_sha256",
        "training_reference",
        "training_reference_sha256",
        "source_roster",
        "source_roster_sha256",
        "truth_reference",
        "truth_reference_sha256",
        "transform_path",
        "transform_sha256",
        "tmv_available",
        "tmv",
        "tmv_absolute",
        "predicted_mass",
        "observed_mass_relative",
    }
    if set(frame.columns) < required or len(frame) != manifest["n_rows"]:
        raise ContractError(f"{profile} score table schema/row count is invalid")
    _numeric_finite(
        frame,
        (
            "target",
            "projection_repeat",
            "projection_seed",
            "n_projections",
            "sliced_w2",
            "exact_w1",
            "exact_w2",
            "n_predicted",
            "n_observed",
            "predicted_weight_sum",
            "exact_ot_predicted_points",
            "exact_ot_observed_points",
            "primary_value",
            "tmv",
            "tmv_absolute",
            "predicted_mass",
            "observed_mass_relative",
        ),
        profile=profile,
    )
    if (
        set(frame["track"].astype(str)) != {"full_data"}
        or set(frame["target"].astype(int)) != set(targets)
        or not frame["source_time"].eq(0).all()
        or set(frame["benchmark"].astype(str)) != {dataset}
        or set(frame["split"].astype(str)) != {"full_data"}
        or set(frame["method"].astype(str)) != {METHOD}
        or set(frame["space"].astype(str)) != set(SPACE_ORDER)
        or set(frame["projection_repeat"].astype(int)) != set(range(PROJECTION_REPEATS))
        or not frame["n_projections"].eq(N_PROJECTIONS).all()
        or not frame["n_predicted"].eq(PREDICTION_N).all()
        or set(frame["primary_metric"].astype(str)) != {"sliced_w2"}
        or not np.allclose(
            frame["primary_value"].to_numpy(dtype=float),
            frame["sliced_w2"].to_numpy(dtype=float),
            rtol=0.0,
            atol=1e-12,
        )
        or set(frame["output_scope"].astype(str)) != {"native_joint"}
        or set(frame["native_vs_adapter"].astype(str)) != {"native_joint"}
        or set(frame["input_manifest_sha256"].astype(str))
        != {expected_manifest["sha256"]}
        or {
            str(Path(value).expanduser().resolve())
            for value in frame["input_manifest"].astype(str)
        }
        != {str(Path(expected_manifest["path"]).resolve())}
        or not frame["tmv_available"].astype(str).str.lower().eq("true").all()
    ):
        raise ContractError(f"{profile} score table values violate full-data contract")
    if (
        frame[
            [
                "sliced_w2",
                "exact_w1",
                "exact_w2",
                "tmv",
                "tmv_absolute",
                "predicted_mass",
                "observed_mass_relative",
            ]
        ]
        < 0
    ).any().any() or not frame["observed_mass_relative"].gt(0).all():
        raise ContractError(f"{profile} score table contains a negative error")
    identity = ["target", "space", "projection_repeat"]
    if frame.duplicated(identity).any():
        raise ContractError(f"{profile} score table has duplicate paired identities")
    expected_identity = {
        (target, space, repeat)
        for target in targets
        for space in SPACE_ORDER
        for repeat in range(PROJECTION_REPEATS)
    }
    observed_identity = set(
        zip(
            frame["target"].astype(int),
            frame["space"].astype(str),
            frame["projection_repeat"].astype(int),
        )
    )
    if observed_identity != expected_identity:
        raise ContractError(f"{profile} score table does not cover the exact grid")
    expected_training = plan["benchmark_inputs"][dataset]["artifacts"][
        "train_training_reference_npz"
    ]
    expected_roster = plan["benchmark_inputs"][dataset]["artifacts"][
        "train_source_roster_npz"
    ]
    for column, expected in (
        ("training_reference", expected_training),
        ("source_roster", expected_roster),
    ):
        if {
            str(Path(value).expanduser().resolve())
            for value in frame[column].astype(str)
        } != {str(Path(expected["path"]).resolve())}:
            raise ContractError(f"{profile} {column} path is not input-bound")
    if set(frame["training_reference_sha256"].astype(str)) != {
        expected_training["sha256"]
    }:
        raise ContractError(f"{profile} training reference SHA-256 is not input-bound")
    if set(frame["source_roster_sha256"].astype(str)) != {expected_roster["sha256"]}:
        raise ContractError(f"{profile} source roster SHA-256 is not input-bound")
    training = _training_reference_arrays(Path(expected_training["path"]))
    expected_transform = _expected_transform_bytes(training)
    try:
        observed_transform = transform_path.read_bytes()
    except OSError as exc:
        raise ContractError(f"Cannot read {profile} frozen transform: {exc}") from exc
    if observed_transform != expected_transform:
        raise ContractError(
            f"{profile} transform was not exactly recomputed from training reference"
        )
    transform = _file_identity(
        transform_path, description=f"{profile} frozen full-data transform"
    )
    if set(frame["transform_sha256"].astype(str)) != {transform["sha256"]} or {
        str(Path(value).expanduser().resolve())
        for value in frame["transform_path"].astype(str)
    } != {str(transform_path.resolve())}:
        raise ContractError(f"{profile} score transform is not artifact-bound")
    for target in targets:
        target_rows = frame.loc[frame["target"].astype(int).eq(target)]
        prediction = prediction_outputs[target]["prediction"]
        summary = prediction_outputs[target]["summary"]
        truth = plan["benchmark_inputs"][dataset]["artifacts"][f"truth_t{target}"]
        truth_arrays = _truth_arrays(Path(truth["path"]), target=target)
        observed_n = int(truth_arrays["state"].shape[0])
        source_n = int(np.count_nonzero(np.isclose(training["time"], 0.0)))
        if source_n <= 0:
            raise ContractError(f"{profile} training reference lacks source time 0")
        predicted_mass = float(prediction_outputs[target]["predicted_mass"])
        observed_mass = float(observed_n / source_n)
        tmv_absolute = abs(predicted_mass - observed_mass)
        tmv = tmv_absolute / observed_mass
        recomputed_metrics = _recompute_metric_values(
            dataset=dataset,
            prediction_path=Path(prediction["path"]),
            training=training,
            truth=truth_arrays,
        ).set_index(["space", "projection_repeat"])
        for column, expected in (
            ("prediction_path", prediction),
            ("prediction_summary", summary),
            ("truth_reference", truth),
        ):
            if {
                str(Path(value).expanduser().resolve())
                for value in target_rows[column].astype(str)
            } != {str(Path(expected["path"]).resolve())}:
                raise ContractError(
                    f"{profile} t{target} {column} path is not artifact-bound"
                )
        for column, expected in (
            ("prediction_sha256", prediction),
            ("prediction_summary_sha256", summary),
            ("truth_reference_sha256", truth),
        ):
            if set(target_rows[column].astype(str)) != {expected["sha256"]}:
                raise ContractError(
                    f"{profile} t{target} {column} is not artifact-bound"
                )
        numeric_expectations = {
            "n_predicted": PREDICTION_N,
            "n_observed": observed_n,
            "predicted_weight_sum": predicted_mass,
            "exact_ot_predicted_points": min(PREDICTION_N, MAX_OT_POINTS),
            "exact_ot_observed_points": min(observed_n, MAX_OT_POINTS),
            "predicted_mass": predicted_mass,
            "observed_mass_relative": observed_mass,
            "tmv_absolute": tmv_absolute,
            "tmv": tmv,
        }
        for column, expected in numeric_expectations.items():
            values = target_rows[column].to_numpy(dtype=float)
            if not np.allclose(values, expected, rtol=0.0, atol=1e-10):
                raise ContractError(
                    f"{profile} t{target} {column} was not recomputed from NPZ/counts"
                )
        for space in SPACE_ORDER:
            for repeat in range(PROJECTION_REPEATS):
                projected = target_rows.loc[
                    target_rows["space"].astype(str).eq(space)
                    & target_rows["projection_repeat"].astype(int).eq(repeat)
                ]
                expected_seed = _expected_projection_seed(dataset, space, repeat)
                dimension = {
                    "state": int(training["state"].shape[1]),
                    "spatial": int(training["spatial"].shape[1]),
                    "joint": int(
                        training["state"].shape[1] + training["spatial"].shape[1]
                    ),
                }[space]
                if (
                    len(projected) != 1
                    or int(projected.iloc[0]["projection_seed"]) != expected_seed
                    or str(projected.iloc[0]["projection_sha256"])
                    != _expected_projection_sha256(dimension, expected_seed)
                ):
                    raise ContractError(
                        f"{profile} t{target} projection basis is not canonical"
                    )
                expected_metrics = recomputed_metrics.loc[(space, repeat)]
                for metric in METRIC_ORDER:
                    if not math.isclose(
                        float(projected.iloc[0][metric]),
                        float(expected_metrics[metric]),
                        rel_tol=1e-10,
                        abs_tol=1e-12,
                    ):
                        raise ContractError(
                            f"{profile} t{target} {metric} differs from NPZ recomputation"
                        )
    for (_, _), group in frame.groupby(["target", "space"]):
        if group["exact_w1"].nunique() != 1 or group["exact_w2"].nunique() != 1:
            raise ContractError(f"{profile} exact OT metric changes across projections")
    for _, group in frame.groupby("target"):
        for column in (
            "tmv",
            "tmv_absolute",
            "predicted_mass",
            "observed_mass_relative",
        ):
            if group[column].nunique() != 1:
                raise ContractError(
                    f"{profile} growth/mass field {column} changes within one target"
                )
    frame.insert(0, "dataset", dataset)
    frame.insert(1, "profile", profile)
    frame.insert(2, "arm", record["arm"])
    frame.insert(3, "arm_label", record["arm_label"])
    artifacts = {
        "metrics": _file_identity(csv_path, description=f"{profile} score table"),
        "status": _file_identity(status_path, description=f"{profile} status table"),
        "manifest": _file_identity(
            manifest_path, description=f"{profile} evaluation manifest"
        ),
        "transform": transform,
    }
    return frame, artifacts


def _validate_all_outputs(
    plan: Mapping[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frames: list[pd.DataFrame] = []
    artifacts: dict[str, Any] = {}
    for profile in PROFILE_ORDER:
        record = _mapping(plan["profiles"][profile], description=profile)
        prediction_artifacts, prediction_run_artifacts = _validate_prediction_outputs(
            plan, profile, record
        )
        score_frame, score_artifacts = _validate_score_outputs(
            plan, profile, record, prediction_artifacts
        )
        frames.append(score_frame)
        artifacts[profile] = {
            "predictions": prediction_artifacts,
            **prediction_run_artifacts,
            "scores": score_artifacts,
        }
    combined = pd.concat(frames, ignore_index=True)
    paired_keys = ["dataset", "target", "space", "projection_repeat"]
    for _, group in combined.groupby(paired_keys, sort=False):
        if set(group["arm"]) != set(ARM_ORDER):
            raise ContractError("A paired score identity lacks one of the three arms")
        for column in (
            "projection_seed",
            "projection_sha256",
            "n_projections",
            "n_observed",
            "input_manifest_sha256",
            "training_reference_sha256",
            "source_roster_sha256",
            "truth_reference_sha256",
            "transform_sha256",
        ):
            if group[column].nunique(dropna=False) != 1:
                raise ContractError(
                    f"Paired arm rows differ in frozen evaluator field {column}"
                )
    return combined, artifacts


def _paired_tables(
    combined: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    long_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    for (dataset, target, space), group in combined.groupby(
        ["dataset", "target", "space"], sort=True
    ):
        for metric in METRIC_ORDER:
            repeats = range(PROJECTION_REPEATS) if metric == "sliced_w2" else (0,)
            for repeat in repeats:
                repeat_group = (
                    group.loc[group["projection_repeat"].astype(int).eq(repeat)]
                    if metric == "sliced_w2"
                    else group.groupby("arm", as_index=False).first()
                )
                values = repeat_group.set_index("arm")[metric].astype(float).to_dict()
                if set(values) != set(ARM_ORDER) or values["full"] <= 0:
                    raise ContractError(
                        "Paired metric values are missing or full error is zero"
                    )
                row = {
                    "dataset": dataset,
                    "target": int(target),
                    "space": space,
                    "metric": metric,
                    "projection_repeat": repeat
                    if metric == "sliced_w2"
                    else "not_applicable",
                    "full": values["full"],
                    "no_lr_prior": values["no_lr_prior"],
                    "no_interaction": values["no_interaction"],
                }
                for arm in ARM_ORDER[1:]:
                    delta = values[arm] - values["full"]
                    row[f"{arm}_minus_full"] = delta
                    row[f"{arm}_relative_to_full"] = delta / values["full"]
                long_rows.append(row)
            arm_target = (
                group.groupby("arm", as_index=True)[metric]
                .mean()
                .astype(float)
                .to_dict()
            )
            target_row = {
                "dataset": dataset,
                "target": int(target),
                "space": space,
                "metric": metric,
                "full": arm_target["full"],
                "no_lr_prior": arm_target["no_lr_prior"],
                "no_interaction": arm_target["no_interaction"],
            }
            for arm in ARM_ORDER[1:]:
                delta = arm_target[arm] - arm_target["full"]
                target_row[f"{arm}_minus_full"] = delta
                target_row[f"{arm}_relative_to_full"] = delta / arm_target["full"]
            target_rows.append(target_row)
    paired_long = pd.DataFrame(long_rows).sort_values(
        ["dataset", "target", "space", "metric", "projection_repeat"],
        ignore_index=True,
    )
    target_deltas = pd.DataFrame(target_rows).sort_values(
        ["dataset", "target", "space", "metric"], ignore_index=True
    )
    summary_rows: list[dict[str, Any]] = []
    for identity, group in target_deltas.groupby(
        ["dataset", "space", "metric"], sort=True
    ):
        dataset, space, metric = identity
        for arm in ARM_ORDER[1:]:
            delta = group[f"{arm}_minus_full"].to_numpy(dtype=float)
            relative = group[f"{arm}_relative_to_full"].to_numpy(dtype=float)
            summary_rows.append(
                {
                    "dataset": dataset,
                    "space": space,
                    "metric": metric,
                    "comparison": f"{arm}_minus_full",
                    "n_targets": len(group),
                    "mean_delta": float(delta.mean()),
                    "sem_delta": float(delta.std(ddof=1) / math.sqrt(len(delta)))
                    if len(delta) > 1
                    else 0.0,
                    "mean_relative_delta": float(relative.mean()),
                    "sem_relative_delta": float(
                        relative.std(ddof=1) / math.sqrt(len(relative))
                    )
                    if len(relative) > 1
                    else 0.0,
                    "fraction_targets_positive": float(np.mean(delta > 0)),
                }
            )
    summary = pd.DataFrame(summary_rows).sort_values(
        ["dataset", "space", "metric", "comparison"], ignore_index=True
    )
    tmv_values = (
        combined.groupby(["dataset", "target", "arm"], as_index=False)["tmv"]
        .first()
        .pivot(index=["dataset", "target"], columns="arm", values="tmv")
        .reset_index()
    )
    tmv_values.columns.name = None
    for arm in ARM_ORDER[1:]:
        tmv_values[f"{arm}_minus_full"] = tmv_values[arm] - tmv_values["full"]
        tmv_values[f"{arm}_relative_to_full"] = np.where(
            tmv_values["full"] > 0,
            (tmv_values[arm] - tmv_values["full"]) / tmv_values["full"],
            np.nan,
        )
    return paired_long, target_deltas, summary, tmv_values


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 9,
            "axes.titlesize": 12,
            "axes.labelsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.labelcolor": TEXT_COLOR,
            "axes.edgecolor": TEXT_COLOR,
            "axes.titlecolor": TEXT_COLOR,
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
            "text.color": TEXT_COLOR,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def _save_figure(target_deltas: pd.DataFrame, pdf_path: Path, png_path: Path) -> None:
    _apply_style()
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(8.27, 11.69),
        sharey=True,
        gridspec_kw={
            "left": 0.10,
            "right": 0.96,
            "top": 0.88,
            "bottom": 0.10,
            "hspace": 0.34,
            "wspace": 0.25,
        },
    )
    plotted = target_deltas.loc[target_deltas["metric"].eq("sliced_w2")]
    all_relative = np.concatenate(
        [
            plotted[f"{arm}_relative_to_full"].to_numpy(dtype=float)
            for arm in ARM_ORDER[1:]
        ]
    )
    bound = max(0.05, float(np.max(np.abs(all_relative))) * 1.18)
    x = np.arange(len(SPACE_ORDER), dtype=float)
    offsets = {"no_lr_prior": -0.11, "no_interaction": 0.11}
    colors = {"no_lr_prior": NO_LR_COLOR, "no_interaction": NO_INTERACTION_COLOR}
    markers = {"no_lr_prior": "s", "no_interaction": "D"}
    for index, (dataset, axis) in enumerate(zip(DATASET_ORDER, axes.flat)):
        axis.text(
            -0.12,
            1.075,
            chr(ord("a") + index),
            transform=axis.transAxes,
            fontsize=14,
            fontweight="bold",
            va="bottom",
        )
        axis.set_title(
            DATASET_LABEL[dataset], loc="left", fontsize=12, fontweight="bold", pad=12
        )
        axis.axhline(0.0, color=REFERENCE_COLOR, linewidth=0.8, zorder=0)
        dataset_frame = plotted.loc[plotted["dataset"].eq(dataset)]
        for arm in ARM_ORDER[1:]:
            means: list[float] = []
            sems: list[float] = []
            for space in SPACE_ORDER:
                values = dataset_frame.loc[
                    dataset_frame["space"].eq(space), f"{arm}_relative_to_full"
                ].to_numpy(dtype=float)
                means.append(float(values.mean()))
                sems.append(
                    float(values.std(ddof=1) / math.sqrt(len(values)))
                    if len(values) > 1
                    else 0.0
                )
            axis.errorbar(
                x + offsets[arm],
                means,
                yerr=sems,
                color=colors[arm],
                marker=markers[arm],
                markerfacecolor="white",
                markeredgewidth=1.2,
                markersize=6,
                linewidth=1.5,
                capsize=3,
            )
        axis.set_xticks(x)
        axis.set_xticklabels([SPACE_LABEL[space] for space in SPACE_ORDER])
        axis.set_xlabel("Frozen benchmark space")
        axis.set_ylim(-bound, bound)
        axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        axis.grid(axis="y", color=GRID_COLOR, linewidth=0.7, alpha=0.8)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        if index % 2 == 0:
            axis.set_ylabel("Relative sliced-W2 change vs full")
    handles = [
        Line2D(
            [0],
            [0],
            color=NO_LR_COLOR,
            marker="s",
            markerfacecolor="white",
            markeredgewidth=1.2,
            label="Radius-only interaction − full",
        ),
        Line2D(
            [0],
            [0],
            color=NO_INTERACTION_COLOR,
            marker="D",
            markerfacecolor="white",
            markeredgewidth=1.2,
            label="No interaction − full",
        ),
    ]
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.53, 0.965),
        ncol=2,
        frameon=False,
        columnspacing=2.0,
    )
    figure.text(
        0.53,
        0.047,
        "Positive values indicate lower full-model reconstruction error. Mean ± SEM across target times.",
        ha="center",
        fontsize=9,
        color=REFERENCE_COLOR,
    )
    metadata = {
        "Title": "Matched-ablation full-data reconstruction benchmark",
        "Author": "CytoBridge",
        "Subject": "Paired full, radius-only, and no-interaction evaluation",
    }
    temporary_pdf = pdf_path.with_name(f".{pdf_path.name}.{os.getpid()}.tmp")
    temporary_png = png_path.with_name(f".{png_path.name}.{os.getpid()}.tmp")
    try:
        figure.savefig(temporary_pdf, format="pdf", metadata=metadata)
        figure.savefig(temporary_png, format="png", dpi=320, metadata=metadata)
        os.replace(temporary_pdf, pdf_path)
        os.replace(temporary_png, png_path)
    finally:
        plt.close(figure)
        temporary_pdf.unlink(missing_ok=True)
        temporary_png.unlink(missing_ok=True)


def _caption_text(target_deltas: pd.DataFrame) -> str:
    plotted = target_deltas.loc[target_deltas["metric"].eq("sliced_w2")]
    summaries = []
    for dataset in DATASET_ORDER:
        dataset_frame = plotted.loc[plotted["dataset"].eq(dataset)]
        values = []
        for arm in ARM_ORDER[1:]:
            mean = float(dataset_frame[f"{arm}_relative_to_full"].mean())
            values.append(f"{ARM_LABEL[arm].lower()} {100.0 * mean:+.1f}%")
        summaries.append(f"{DATASET_LABEL[dataset]}: " + ", ".join(values))
    return (
        "**Matched-ablation full-data reconstruction benchmark.** **a–d,** Paired "
        "relative changes in the primary sliced Wasserstein-2 error for Zebrafish, "
        "MOSTA, ARISTA, and AD mouse. Each arm uses the official unified-benchmark "
        "continuous full-data prediction from the same frozen source roster, seed, "
        "truth artifact, transform, and projection bases. Points and error bars are "
        "the mean and SEM across configured target times after averaging five "
        "projection repeats within each target. Positive values mean that the full "
        "learned-interaction model had lower reconstruction error than the indicated "
        "ablation. Negative values mean that the ablation had lower error. Mean "
        "changes pooled across the three displayed spaces were "
        + "; ".join(summaries)
        + ". Exact Wasserstein-1, exact Wasserstein-2, and native growth-mass TMV "
        "deltas are provided in the accompanying tables. These are in-sample "
        "full-data reconstruction results and must not be interpreted as "
        "leave-one-timepoint-out generalization or as a hypothesis test.\n"
    )


def _atomic_text(path: Path, value: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(value, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_text(path, _canonical_json_bytes(dict(value)).decode("utf-8"))


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        frame.to_csv(temporary, index=False, float_format="%.12g")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _output_artifact(path: Path, *, root: Path) -> dict[str, Any]:
    identity = _file_identity(path, description=f"report output {path.name}")
    return {
        "path": path.relative_to(root).as_posix(),
        "size_bytes": identity["size_bytes"],
        "sha256": identity["sha256"],
    }


def _git_state(repo: Path) -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=all"],
                cwd=repo,
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            ).stdout.strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}


def _provenance_text(
    *,
    root: Path,
    plan: Mapping[str, Any],
    plan_sha: str,
    output_dir: Path,
    artifacts: Mapping[str, Mapping[str, Any]],
) -> str:
    rebuild = " ".join(
        shlex.quote(value)
        for value in (
            sys.executable,
            str(Path(__file__).resolve()),
            "report",
            "--run-root",
            str(root),
            "--output-dir",
            str(output_dir.with_name(f"{output_dir.name}-rebuild")),
        )
    )
    lines = [
        "# Figure provenance",
        "",
        f"Generated (UTC): `{_utc_now()}`",
        "",
        "Scientific claim: Under one frozen full-data benchmark contract, quantify "
        "the reconstruction-error change caused by replacing the learned LR gate "
        "with radius-only interactions or removing interactions.",
        "",
        "## Source paths",
        "",
        f"- Prepared evaluation plan: `{root / CONTRACT_DIR / PLAN_NAME}`",
        f"- Evaluation plan SHA-256: `{plan_sha}`",
        f"- Matched launcher manifest: `{plan['launcher']['manifest']['path']}`",
        f"- Launcher manifest SHA-256: `{plan['launcher']['launcher_manifest_sha256']}`",
        f"- Matched acceptance report: `{plan['launcher']['acceptance']['path']}`",
        f"- Acceptance SHA-256: `{plan['launcher']['acceptance_sha256']}`",
        "- Twelve config/training-summary identities and 72 stage checkpoints are "
        "recorded in the prepared plan.",
        "- Four benchmark manifests and their full-data train, roster, truth, split "
        "manifest, and sidecar artifacts are hash-bound in the prepared plan.",
        "",
        "## Evaluation protocol",
        "",
        f"- Track: full_data in-sample reconstruction; never pooled with LOTO.",
        f"- Prediction population: {PREDICTION_N}; seed: {MATCHED_SEED}; dt: {INFERENCE_DT}.",
        f"- Primary metric: sliced W2 with {N_PROJECTIONS} projections and "
        f"{PROJECTION_REPEATS} shared repeats.",
        f"- Secondary exact OT support cap: {MAX_OT_POINTS} points per cloud.",
        "- Figure estimator: per-target mean across projection repeats, then mean ± "
        "SEM across configured target times.",
        "- Delta sign: ablation minus full. Positive means lower full-model error.",
        "",
        "## Rebuild",
        "",
        "```text",
        rebuild,
        "```",
        "",
        "## Interpretation",
        "",
        "The report supports only paired in-sample reconstruction comparisons of "
        "the three accepted arms. It is not evidence of held-out-time generalization "
        "and does not constitute a hypothesis test.",
        "",
        "## SHA-256",
        "",
        f"- Plotter/orchestrator: `{_sha256(Path(__file__).resolve())}`",
    ]
    for name, artifact in artifacts.items():
        lines.append(f"- `{artifact['path']}`: `{artifact['sha256']}` ({name})")
    return "\n".join(lines) + "\n"


def generate_report(
    root: Path,
    plan: Mapping[str, Any],
    plan_sha: str,
    combined: pd.DataFrame,
    source_artifacts: Mapping[str, Any],
    output_dir: Path,
) -> Path:
    if output_dir.exists():
        raise FileExistsError(f"Refusing existing report directory: {output_dir}")
    if output_dir.parent != root and root not in output_dir.parents:
        raise ContractError("Report output directory must remain under evaluation root")
    output_dir.mkdir(parents=True)
    paired_long, target_deltas, summary, tmv = _paired_tables(combined)
    paths = {
        "metrics_with_arm_long": output_dir / "metrics_with_arm_long.csv",
        "paired_deltas_long": output_dir / "paired_deltas_long.csv",
        "paired_target_deltas": output_dir / "paired_target_deltas.csv",
        "paired_delta_summary": output_dir / "paired_delta_summary.csv",
        "paired_tmv_deltas": output_dir / "paired_tmv_deltas.csv",
    }
    for path, frame in (
        (paths["metrics_with_arm_long"], combined),
        (paths["paired_deltas_long"], paired_long),
        (paths["paired_target_deltas"], target_deltas),
        (paths["paired_delta_summary"], summary),
        (paths["paired_tmv_deltas"], tmv),
    ):
        _atomic_csv(path, frame)
    pdf_path = output_dir / f"{FIGURE_BASENAME}.pdf"
    png_path = output_dir / f"{FIGURE_BASENAME}.png"
    _save_figure(target_deltas, pdf_path, png_path)
    caption_path = output_dir / "figure_caption.md"
    _atomic_text(caption_path, _caption_text(target_deltas))
    artifacts = {
        name: _output_artifact(path, root=output_dir) for name, path in paths.items()
    }
    artifacts.update(
        {
            "figure_pdf": _output_artifact(pdf_path, root=output_dir),
            "figure_png": _output_artifact(png_path, root=output_dir),
            "figure_caption": _output_artifact(caption_path, root=output_dir),
        }
    )
    provenance_path = output_dir / "PROVENANCE.md"
    _atomic_text(
        provenance_path,
        _provenance_text(
            root=root,
            plan=plan,
            plan_sha=plan_sha,
            output_dir=output_dir,
            artifacts=artifacts,
        ),
    )
    artifacts["provenance"] = _output_artifact(provenance_path, root=output_dir)
    covered = {
        "schema_version": 1,
        "kind": REPORT_KIND,
        "evaluation_plan_sha256": plan_sha,
        "launcher_manifest_sha256": plan["launcher"]["launcher_manifest_sha256"],
        "acceptance_sha256": plan["launcher"]["acceptance_sha256"],
        "settings": plan["settings"],
        "source_output_artifacts": source_artifacts,
        "outputs": artifacts,
        "code": {
            "orchestrator": _file_identity(
                Path(__file__).resolve(), description="report orchestrator"
            ),
            "git": _git_state(Path(__file__).resolve().parents[1]),
        },
    }
    report_manifest = {
        **covered,
        "status": "complete",
        "completed_at": _utc_now(),
        "signature": {
            "algorithm": "sha256-canonical-json",
            "value": _stable_json_sha256(covered),
            "covered_top_level_fields": list(covered),
            "excludes": ["status", "completed_at", "signature", "self hash"],
        },
    }
    manifest_path = output_dir / "report_manifest.json"
    _atomic_json(manifest_path, report_manifest)
    digest = _sha256(manifest_path)
    _atomic_text(
        output_dir / "report_manifest.sha256",
        f"{digest}  report_manifest.json\n",
    )
    return manifest_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="bind inputs and create a new plan root")
    prepare.add_argument("--run-root", required=True, type=Path)
    prepare.add_argument("--launcher-manifest", required=True, type=Path)
    prepare.add_argument("--expected-launcher-manifest-sha256", required=True)
    prepare.add_argument("--matched-acceptance", required=True, type=Path)
    prepare.add_argument("--expected-matched-acceptance-sha256", required=True)
    prepare.add_argument(
        "--benchmark-input",
        action="append",
        required=True,
        help="Repeat DATASET=/absolute/path/to/inputs/manifest.json four times.",
    )
    prepare.add_argument(
        "--expected-benchmark-input-sha256",
        action="append",
        required=True,
        help="Repeat DATASET=SHA256 four times.",
    )
    render = sub.add_parser("render", help="print commands without executing them")
    render.add_argument("--run-root", required=True, type=Path)
    render.add_argument("--phase", choices=("all", "infer", "score"), default="all")
    validate = sub.add_parser("validate", help="read-only validation of all outputs")
    validate.add_argument("--run-root", required=True, type=Path)
    report = sub.add_parser("report", help="validate and write quantitative report")
    report.add_argument("--run-root", required=True, type=Path)
    report.add_argument("--output-dir", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "prepare":
        benchmark_inputs = _assignments(
            args.benchmark_input, option="--benchmark-input"
        )
        benchmark_hashes = _assignments(
            args.expected_benchmark_input_sha256,
            option="--expected-benchmark-input-sha256",
        )
        plan = build_plan(
            evaluation_root=args.run_root,
            launcher_manifest=args.launcher_manifest,
            expected_launcher_sha256=args.expected_launcher_manifest_sha256,
            acceptance_report=args.matched_acceptance,
            expected_acceptance_sha256=args.expected_matched_acceptance_sha256,
            benchmark_inputs=benchmark_inputs,
            expected_benchmark_sha256=benchmark_hashes,
        )
        root, digest = prepare_run_root(plan)
        print(
            json.dumps(
                {"status": "prepared", "run_root": str(root), "plan_sha256": digest},
                indent=2,
            )
        )
        return 0
    root, plan, plan_sha = verify_prepared_plan(args.run_root)
    if args.command == "render":
        for profile in PROFILE_ORDER:
            commands = plan["profiles"][profile]["commands"]
            if args.phase in {"all", "infer"}:
                print(f"[{profile}] infer-full")
                print(commands["infer_full"]["shell"])
            if args.phase in {"all", "score"}:
                print(f"[{profile}] score-full-data")
                print(commands["score_full_data"]["shell"])
        return 0
    combined, artifacts = _validate_all_outputs(plan)
    if args.command == "validate":
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "plan_sha256": plan_sha,
                    "profiles": len(PROFILE_ORDER),
                    "metric_rows": len(combined),
                    "track": "full_data",
                },
                indent=2,
            )
        )
        return 0
    output_dir = (
        root / "report"
        if args.output_dir is None
        else args.output_dir.expanduser().resolve()
    )
    manifest = generate_report(root, plan, plan_sha, combined, artifacts, output_dir)
    print(f"Saved matched-ablation benchmark report: {manifest}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        ContractError,
        FileExistsError,
        FileNotFoundError,
        OSError,
        yaml.YAMLError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
