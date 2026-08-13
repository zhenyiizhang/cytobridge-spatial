#!/usr/bin/env python3
"""Prepare, run, and evaluate the four-dataset benchmark with one small CLI."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

# fmt: off
REPO = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO / "configs" / "unified_benchmark"
DATASETS = ("zebrafish", "mosta", "arista", "admouse")
DYNAMIC = ("stvcr", "stories", "mioflow")
STATIC = ("moscot", "wot", "paste", "spateo", "linear_centroid_shift", "random_independent_pairs")
PRIMARY_METHODS = ("cytobridge", *DYNAMIC, *STATIC)
METHODS = (*PRIMARY_METHODS, "spatrack")
METHOD_NAME = {"cytobridge": "CytoBridge-0.015", **{name: name for name in METHODS[1:]}}
EXTERNAL_STATIC = {"moscot", "wot", "paste", "spateo"}
METHOD_REGISTRY = REPO / "scripts" / "spatiotemporal_benchmark" / "method_registry.json"
STATIC_REGISTRY = REPO / "scripts" / "spatiotemporal_benchmark" / "static_baselines" / "method_registry.json"
OUTPUT_SCOPE_BY_METHOD = {
    "cytobridge": "native_joint",
    "stvcr": "native_joint",
    "stories": "native_state",
    "mioflow": "native_state",
    "moscot": "hybrid_joint",
    "wot": "hybrid_state",
    "paste": "hybrid_joint",
    "spateo": "hybrid_joint",
    "linear_centroid_shift": "native_joint",
    "random_independent_pairs": "native_joint",
}
NATIVE_VS_ADAPTER_BY_METHOD = {
    "cytobridge": "native_joint",
    "stvcr": "native_joint",
    "stories": "native_state",
    "mioflow": "native_state",
    "moscot": "hybrid_coupling_adapter",
    "wot": "hybrid_coupling_adapter",
    "paste": "hybrid_coupling_adapter",
    "spateo": "hybrid_coupling_adapter",
    "linear_centroid_shift": "explicit_control",
    "random_independent_pairs": "explicit_control",
}
EVALUATION_NON_NUMERIC_STATUSES = {
    "timeout",
    "oom",
    "failed",
    "not_available",
    "not_applicable",
}
EVALUATION_STATUS_ALIASES = {
    "out_of_memory": "oom",
    "unavailable": "not_available",
    "n/a": "not_applicable",
    "na": "not_applicable",
}
EVALUATION_TRACK_ALIASES = {
    "loto": "loto",
    "full_data": "full_data",
    "full-data": "full_data",
    "noholdout": "full_data",
    "no-holdout": "full_data",
}
DYNAMIC_REQUIRED_ARTIFACTS = {
    "stvcr": {"source_roster", "model", "rigid_transform", "fitted_train"},
    "stories": {"source_roster", "model"},
    "mioflow": {"source_roster", "model", "state_transform"},
}
DYNAMIC_DEFAULT_PARAMS = {
    "stvcr": {
        "n_epochs": 2000,
        "num_samples": 500,
        "use_alignment": False,
        "use_growth": True,
        "device": "cuda",
        "delta_t": 0.1,
        "weights_return_index": None,
    },
    "stories": {
        "max_iter": 100,
        "batch_size": 128,
        "restore": False,
        "keep_checkpoints": True,
    },
    "mioflow": {
        "n_epochs": 100,
        "hidden_dim": 64,
        "learning_rate": 1e-3,
        "lambda_ot": 1.0,
        "lambda_energy": 0.01,
        "sample_size": 512,
        "use_cuda": True,
        "n_bins": 101,
    },
}
DYNAMIC_SEED = 20_260_718
DYNAMIC_PINS = REPO / "scripts" / "spatiotemporal_benchmark" / "dynamic" / "method_pins.json"
DYNAMIC_ADAPTER_FILES = (
    "scripts/spatiotemporal_benchmark/dynamic/common.py",
    "scripts/spatiotemporal_benchmark/dynamic/run_dynamic.py",
)
STATIC_ADAPTER_FILES = (
    "scripts/spatiotemporal_benchmark/static_baselines/run.py",
    "scripts/spatiotemporal_benchmark/static_baselines/methods.py",
    "scripts/spatiotemporal_benchmark/static_baselines/coupling.py",
    "scripts/spatiotemporal_benchmark/static_baselines/data.py",
    "scripts/spatiotemporal_benchmark/static_baselines/provenance.py",
    "scripts/spatiotemporal_benchmark/static_baselines/registry.py",
)
CYTOBRIDGE_ADAPTER_FILES = (
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
DEFAULT_FORMAL_ROOT = Path(
    "/data/cytobridge/projects/CytoBridge-ST-1104/runs/"
    "corrected-matched-ablation-20260813-3c87a3e-r1"
)
DEFAULT_RUN_ROOT = Path(
    "/data/cytobridge/projects/CytoBridge-ST-1104/runs/"
    "corrected-benchmark-20260813-matched-3c87a3e-c4f8e203-r1"
)


def load_datasets(names):
    return {name: yaml.safe_load((CONFIG_DIR / f"{name}.yaml").read_text(encoding="utf-8")) for name in names}


def assignments(values):
    result = {}
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or name not in METHODS:
            raise ValueError(f"expected METHOD=PATH, found {value!r}")
        result[name] = Path(path).expanduser().resolve()
    return result


def command(module, python, *args):
    return [str(python), "-m", module, *(str(value) for value in args)]


def dynamic_commands(method, python, source, manifest, root, split, targets):
    module = "scripts.spatiotemporal_benchmark.dynamic.run_dynamic"
    fit = root / "fits" / method / split
    shared = ("--method", method, "--input-manifest", manifest, "--split-id", split, "--source-root", source)
    commands = [command(module, python, "fit", *shared, "--output-dir", fit)]
    track = "loto" if split.startswith("loto") else "full_data"
    for target in targets:
        output = root / "predictions" / track / method / f"t{target}"
        commands.append(command(module, python, "infer", *shared, "--fit-dir", fit,
                                "--target-time", target, "--output-dir", output))
    return commands


def static_commands(method, python, source, manifest, root, split, targets):
    module = "scripts.spatiotemporal_benchmark.static_baselines.run"
    track = "loto" if split.startswith("loto") else "full_data"
    output = root / "predictions" / track / method
    if track == "loto":
        output /= f"t{targets[0]}"
    args = ["run", "--method", method, "--evaluation-mode", "loto" if track == "loto" else "no-holdout",
            "--input-h5ad", root / "inputs" / split / "train.h5ad", "--input-manifest", manifest,
            "--output-dir", output, "--max-fit-n", 800]
    if track == "loto":
        args += ["--target-time", targets[0]]
    if source:
        args += ["--source-root", source]
    return [command(module, python, *args)]


def cytobridge_commands(python, cfg, formal, manifest, root, split, targets, device):
    module = "scripts.spatiotemporal_benchmark.cytobridge.run_cytobridge"
    model = formal / "training"
    training_config = formal / cfg["benchmark"]["training_config"]
    shared = ("--repo", REPO, "--input-manifest", manifest, "--split", split)
    if split == "full_data":
        output = root / "predictions" / "full_data" / "cytobridge"
        return [command(module, python, "validate-model", *shared, "--model-dir", model,
                        "--training-config", training_config),
                command(module, python, "infer-full", *shared, "--model-dir", model,
                        "--training-config", training_config, "--output-dir", output, "--device", device)]
    target, graph = targets[0], root / "graphs" / split
    loto_model = root / "fits" / "cytobridge" / split
    output = root / "predictions" / "loto" / "cytobridge" / f"t{target}"
    prepare_args = ["--training-config", training_config]
    if cfg["benchmark"].get("edge_prior_mode", "learned") == "learned":
        prepare_args += ["--database", REPO / cfg["benchmark"]["graph_database"]]
    return [command(module, python, "prepare-loto", *shared, *prepare_args,
                    "--expression-layer", cfg["benchmark"]["expression_layer"],
                    "--output-dir", graph, "--device", device),
            command(module, python, "fit-loto", *shared, "--training-config", training_config,
                    "--graph-dir", graph, "--output-dir", loto_model, "--device", device),
            command(module, python, "infer-loto", *shared, "--model-dir", loto_model,
                    "--training-config", training_config, "--output-dir", output, "--device", device)]


def jobs_for_dataset(name, cfg, args, pythons, sources):
    root, formal = args.run_root / name, args.formal_root / name
    manifest, jobs = root / "inputs" / "manifest.json", []
    for method in args.methods:
        for track in args.tracks:
            targets = cfg["loto_targets"] if track == "loto" else cfg["full_data_targets"]
            splits = [(f"loto_t{target}", [target]) for target in targets] if track == "loto" else [("full_data", targets)]
            for split, selected in splits:
                if method == "spatrack":
                    jobs.append((method, track, selected, [], [], "not_applicable")); continue
                python, source = pythons.get(method, Path(sys.executable)), sources.get(method, args.software_root / method)
                required = [python, manifest]
                if method == "cytobridge":
                    commands = cytobridge_commands(python, cfg, formal, manifest, root, split, selected, args.device)
                    required += [formal / "training", formal / cfg["benchmark"]["training_config"]]
                elif method in DYNAMIC:
                    commands = dynamic_commands(method, python, source, manifest, root, split, selected); required.append(source)
                else:
                    official_source = source if method in EXTERNAL_STATIC else None
                    commands = static_commands(method, python, official_source, manifest, root, split, selected)
                    if official_source: required.append(official_source)
                jobs.append((method, track, selected, commands, required, None))
    return jobs


def execute(commands, required, timeout, log_path):
    missing = [str(path) for path in required if not Path(path).exists()]
    if missing:
        return "not_available", "missing: " + ", ".join(missing)
    deadline, output = time.monotonic() + timeout, []
    try:
        for item in commands:
            run = subprocess.run(item, cwd=REPO, text=True, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, timeout=max(1, deadline - time.monotonic()))
            output.append(run.stdout)
            if run.returncode:
                status = "oom" if "out of memory" in "\n".join(output).lower() else "failed"
                break
        else:
            status = "completed"
    except subprocess.TimeoutExpired as error:
        fragment = error.stdout or ""
        output.append(fragment.decode() if isinstance(fragment, bytes) else fragment)
        status = "timeout"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("\n".join(output), encoding="utf-8")
    reason = "" if status == "completed" else f"{status}; see {log_path}"
    return status, reason


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


class ResumeValidationCache:
    """Memoize only stat-stable inputs and release/model artifacts.

    This cache is intentionally scoped to one ``run_dataset`` or ``evaluate``
    call. Prediction, summary, fit-manifest, and fit-artifact bytes must never
    use it because commands can create or replace those outputs mid-run.
    """

    def __init__(self):
        self._immutable_hashes = {}

    @staticmethod
    def _fingerprint(path):
        stat = path.stat()
        return (
            stat.st_dev,
            stat.st_ino,
            stat.st_size,
            stat.st_mtime_ns,
            stat.st_ctime_ns,
        )

    def immutable_sha256(self, path):
        path = Path(path).expanduser().resolve()
        before = self._fingerprint(path)
        cached = self._immutable_hashes.get(path)
        if cached is not None and cached[0] == before:
            return cached[1]
        digest = sha256_file(path)
        after = self._fingerprint(path)
        if before != after:
            raise OSError(f"immutable artifact changed while hashing: {path}")
        self._immutable_hashes[path] = (after, digest)
        return digest


def _immutable_sha256(path, validation_cache=None):
    if validation_cache is None:
        return sha256_file(path)
    if not isinstance(validation_cache, ResumeValidationCache):
        raise TypeError("validation_cache must be a ResumeValidationCache")
    return validation_cache.immutable_sha256(path)


def current_dynamic_adapter_implementation(validation_cache=None):
    """Hash the exact package-side dynamic adapter bytes used by this release."""

    files = {
        relative_path: _immutable_sha256(
            REPO / relative_path, validation_cache
        )
        for relative_path in DYNAMIC_ADAPTER_FILES
    }
    aggregate = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": "1.0.0",
        "files": files,
        "aggregate_sha256": aggregate,
    }


def _current_adapter_implementation(relative_paths, validation_cache=None):
    files = {
        path: _immutable_sha256(REPO / path, validation_cache)
        for path in relative_paths
    }
    aggregate = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": "1.0.0",
        "files": files,
        "aggregate_sha256": aggregate,
    }


def current_static_adapter_implementation(validation_cache=None):
    return _current_adapter_implementation(
        STATIC_ADAPTER_FILES, validation_cache
    )


def current_cytobridge_adapter_implementation(validation_cache=None):
    return _current_adapter_implementation(
        CYTOBRIDGE_ADAPTER_FILES, validation_cache
    )


def stable_seed(base_seed, *parts):
    payload = json.dumps(
        [int(base_seed), *map(str, parts)], separators=(",", ":")
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def _current_dataset_config(config):
    """Return the source and builder-resolved forms of the current dataset YAML."""

    dataset_id = config.get("dataset_id") if isinstance(config, dict) else None
    if dataset_id not in DATASETS:
        return None
    config_path = (CONFIG_DIR / f"{dataset_id}.yaml").expanduser().resolve()
    if not config_path.is_file():
        return None
    try:
        # Keep resume semantics identical to the input producer.  The fallback
        # preserves the documented ``python scripts/...`` entry point, where
        # the repository root is not necessarily importable as a package.
        try:
            from scripts.spatiotemporal_benchmark import build_inputs as builder
        except ModuleNotFoundError as error:
            if error.name not in {"scripts", "scripts.spatiotemporal_benchmark"}:
                raise
            import build_inputs as builder

        source_config = builder.load_config(config_path)
        resolved_config = builder.resolve_config(
            builder.parse_args(
                ["--config", str(config_path), "--validate-only"]
            )
        )
    except (ImportError, OSError, ValueError, TypeError, yaml.YAMLError):
        return None
    if source_config != config:
        return None
    return config_path, resolved_config


def _resolved_config_artifact_is_current(
    manifest_path, record, expected_config, validation_cache=None
):
    """Verify the producer's fixed resolved-config artifact and its semantics."""

    if not isinstance(record, dict):
        return False
    declared_path = record.get("path")
    relative_path = record.get("relative_path")
    declared_sha = record.get("sha256")
    declared_size = record.get("size_bytes")
    expected_path = (manifest_path.parent / "resolved_config.yaml").resolve()
    if (
        not isinstance(declared_path, str)
        or declared_path != str(expected_path)
        or not isinstance(relative_path, str)
        or relative_path != "resolved_config.yaml"
        or not isinstance(declared_sha, str)
        or len(declared_sha) != 64
        or not isinstance(declared_size, int)
        or isinstance(declared_size, bool)
        or not expected_path.is_file()
        or _immutable_sha256(expected_path, validation_cache) != declared_sha
        or declared_size != expected_path.stat().st_size
    ):
        return False
    payload = yaml.safe_load(expected_path.read_text(encoding="utf-8"))
    return isinstance(payload, dict) and payload == expected_config


def _current_prediction_contract(
    root, config, track, target, validation_cache=None
):
    """Bind resume checks to the current root manifest and dataset recipe."""

    manifest_path = root / "inputs" / "manifest.json"
    if not manifest_path.is_file() or not isinstance(config, dict):
        return None
    if track not in {"loto", "full_data"}:
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("status") != "complete":
        return None
    current_config = _current_dataset_config(config)
    if current_config is None:
        return None
    config_path, resolved_config = current_config
    config_source = manifest.get("config_source")
    if (
        not isinstance(config_source, dict)
        or config_source.get("path") != str(config_path)
        or config_source.get("sha256")
        != _immutable_sha256(config_path, validation_cache)
        or not _resolved_config_artifact_is_current(
            manifest_path,
            manifest.get("resolved_config"),
            resolved_config,
            validation_cache,
        )
    ):
        return None
    if manifest.get("dataset_id") != config.get("dataset_id"):
        return None
    expected_prediction_n = int(config["prediction_n"])
    expected_state_dim = int(config["state_dim"])
    expected_spatial_dim = int(config["spatial_dim"])
    if (
        expected_prediction_n <= 0
        or expected_state_dim <= 0
        or expected_spatial_dim <= 0
        or int(manifest.get("prediction_n", -1)) != expected_prediction_n
    ):
        return None
    split_id = "full_data" if track == "full_data" else f"loto_t{target}"
    splits = manifest.get("splits")
    split = splits.get(split_id) if isinstance(splits, dict) else None
    if not isinstance(split, dict):
        return None
    if int(split.get("prediction_n", -1)) != expected_prediction_n:
        return None
    targets = split.get("evaluation_targets")
    if not isinstance(targets, list) or not any(
        np.isclose(float(value), float(target), rtol=0.0, atol=1e-8)
        for value in targets
    ):
        return None
    train = split.get("train")
    if not isinstance(train, dict):
        return None

    def artifact_sha(name):
        artifact = train.get(name)
        if not isinstance(artifact, dict):
            return None
        digest = artifact.get("sha256")
        location = artifact.get("relative_path", artifact.get("path"))
        if not isinstance(digest, str) or not isinstance(location, str):
            return None
        artifact_path = Path(location).expanduser()
        if not artifact_path.is_absolute():
            artifact_path = manifest_path.parent / artifact_path
        if (
            not artifact_path.is_file()
            or _immutable_sha256(artifact_path, validation_cache)
            != digest.lower()
        ):
            return None
        return digest.lower()

    train_sha = artifact_sha("h5ad")
    reference_sha = artifact_sha("training_reference_npz")
    roster_sha = artifact_sha("source_roster_npz")
    try:
        source_time = float(split["source_time"])
    except (KeyError, TypeError, ValueError):
        return None
    if not np.isfinite(source_time):
        return None
    if not all(
        isinstance(digest, str) and len(digest) == 64
        for digest in (train_sha, reference_sha, roster_sha)
    ):
        return None
    return {
        "manifest_path": manifest_path.resolve(),
        "manifest_sha256": _immutable_sha256(
            manifest_path, validation_cache
        ),
        "dataset_id": str(config["dataset_id"]),
        "prediction_n": expected_prediction_n,
        "state_dim": expected_state_dim,
        "spatial_dim": expected_spatial_dim,
        "split_id": split_id,
        "source_time": source_time,
        "train_h5ad_sha256": train_sha,
        "training_reference_sha256": reference_sha,
        "source_roster_sha256": roster_sha,
        "cytobridge_interaction_mode": str(
            config.get("benchmark", {}).get("edge_prior_mode", "learned")
        )
        .strip()
        .lower(),
    }


def _summary_prediction_n(summary):
    for key in ("prediction_n", "prediction_n_contract", "prediction_n_per_time"):
        if key in summary:
            return int(summary[key])
    raise ValueError("prediction summary lacks the benchmark particle-count contract")


def _cytobridge_interaction_summary_is_consistent(summary):
    """Reject stale interaction claims, especially for the component ablation."""

    mode = str(summary.get("interaction_mode", "")).strip().lower()
    edge_mode = str(summary.get("edge_prior_mode", "")).strip().lower()
    include = summary.get("include_interaction")
    simulation = summary.get("simulation")
    if mode not in {"learned", "all_spatial", "none"} or mode != edge_mode:
        return False
    if not isinstance(simulation, dict):
        return False
    if (
        not np.isclose(float(summary.get("dt", np.nan)), 0.01, rtol=0.0, atol=1e-12)
        or not np.isclose(float(summary.get("sigma", np.nan)), 0.03, rtol=0.0, atol=1e-12)
        or not np.isclose(
            float(summary.get("alpha_express", np.nan)),
            0.015,
            rtol=0.0,
            atol=1e-12,
        )
        or not np.isclose(
            float(summary.get("alpha_spatial", np.nan)),
            10.0,
            rtol=0.0,
            atol=1e-12,
        )
        or summary.get("include_score") is not True
    ):
        return False
    if simulation.get("interaction_mode") != mode or simulation.get(
        "edge_prior_mode"
    ) != mode:
        return False
    if simulation.get("include_interaction") is not include:
        return False
    if simulation.get("stochastic_stream_contract") != (
        "interaction grouping uses an independent torch.Generator; the "
        "global torch stream remains paired for Brownian diffusion"
    ):
        return False
    if mode == "none":
        return (
            include is False
            and summary.get("interaction_m") is None
            and simulation.get("interaction_m") is None
            and simulation.get("loaded_model_interaction_group_size") is None
            and simulation.get("interaction_group_binding")
            == "not_applicable_no_interaction_component"
            and simulation.get("interaction_grouping_seed") is None
            and simulation.get("dynamics_components")
            == ["velocity", "growth", "score"]
            and summary.get("edge_predictor_used") is False
            and simulation.get("edge_predictor_used") is False
        )
    try:
        interaction_m = int(summary.get("interaction_m"))
    except (TypeError, ValueError):
        return False
    return (
        include is True
        and interaction_m == 1024
        and simulation.get("interaction_m") == interaction_m
        and simulation.get("loaded_model_interaction_group_size") == interaction_m
        and simulation.get("interaction_group_binding")
        == "exact_checkpoint_model_match"
        and simulation.get("interaction_grouping_seed") == 10_042
        and simulation.get("dynamics_components")
        == ["velocity", "growth", "score", "interaction"]
        and summary.get("edge_predictor_used") is (mode == "learned")
        and simulation.get("edge_predictor_used") is (mode == "learned")
    )


def _cytobridge_source_roster_is_current(summary, contract):
    roster = summary.get("source_roster")
    if not isinstance(roster, dict):
        return False
    output = Path(str(roster.get("source_roster", ""))).expanduser().resolve()
    canonical = Path(str(roster.get("canonical_input_roster", ""))).expanduser().resolve()
    manifest = json.loads(contract["manifest_path"].read_text(encoding="utf-8"))
    roster_record = manifest["splits"][contract["split_id"]]["train"][
        "source_roster_npz"
    ]
    expected_canonical = Path(
        str(roster_record.get("relative_path", roster_record.get("path", "")))
    ).expanduser()
    if not expected_canonical.is_absolute():
        expected_canonical = contract["manifest_path"].parent / expected_canonical
    return (
        output.is_file()
        and canonical.resolve() == expected_canonical.resolve()
        and output != canonical
        and sha256_file(output) == contract["source_roster_sha256"]
        and roster.get("source_roster_sha256") == contract["source_roster_sha256"]
        and roster.get("canonical_input_roster_sha256")
        == contract["source_roster_sha256"]
        and int(roster.get("prediction_n", -1)) == contract["prediction_n"]
        and np.isclose(
            float(roster.get("source_time", np.nan)),
            contract["source_time"],
            rtol=0.0,
            atol=1e-8,
        )
    )


def _cytobridge_full_run_summary_is_current(root, summary, contract, config):
    run_path = root / "predictions" / "full_data" / "cytobridge" / "run_summary.json"
    if not run_path.is_file():
        return False
    run = json.loads(run_path.read_text(encoding="utf-8"))
    expected_targets = [float(value) for value in config["full_data_targets"]]
    rows = run.get("prediction_summaries")
    if (
        run.get("status") != "complete"
        or run.get("method") != "CytoBridge-0.015"
        or run.get("regime") != "full_data"
        or run.get("split_id") != "full_data"
        or run.get("single_continuous_non_split_call") is not True
        or run.get("intermediate_reset") is not False
        or run.get("spatial_warp_applied") is not False
        or run.get("seed") != 42
        or run.get("prediction_n") != contract["prediction_n"]
        or not np.isclose(
            float(run.get("source_time", np.nan)),
            contract["source_time"],
            rtol=0.0,
            atol=1e-8,
        )
        or [float(value) for value in run.get("targets", [])] != expected_targets
        or run.get("input_manifest_sha256") != contract["manifest_sha256"]
        or run.get("train_h5ad_sha256") != contract["train_h5ad_sha256"]
        or run.get("training_reference_sha256")
        != contract["training_reference_sha256"]
        or run.get("source_roster_sha256") != contract["source_roster_sha256"]
        or not isinstance(rows, list)
        or len(rows) != len(expected_targets)
        or not _cytobridge_source_roster_is_current(run, contract)
    ):
        return False
    by_target = {float(row.get("target", np.nan)): row for row in rows if isinstance(row, dict)}
    if set(by_target) != set(expected_targets):
        return False
    for target, row in by_target.items():
        prediction = root / "predictions" / "full_data" / "cytobridge" / f"t{target:g}" / "prediction.npz"
        target_summary = prediction.parent / "summary.json"
        if (
            not prediction.is_file()
            or not target_summary.is_file()
            or Path(str(row.get("prediction_npz", ""))).resolve() != prediction.resolve()
            or row.get("prediction_npz_sha256") != sha256_file(prediction)
        ):
            return False
        target_payload = json.loads(target_summary.read_text(encoding="utf-8"))
        if not np.isclose(
            float(row.get("predicted_mass", np.nan)),
            float(target_payload.get("predicted_mass", np.nan)),
            rtol=1e-7,
            atol=1e-8,
        ):
            return False
    return True


def _current_cytobridge_model_contract(
    root,
    formal_dataset_root,
    config,
    track,
    split_id,
    validation_cache=None,
):
    """Hash the exact formal config/checkpoints that a prediction must reuse."""

    if formal_dataset_root is None:
        return None
    formal_dataset_root = Path(formal_dataset_root).expanduser().resolve()
    model_dir = (
        formal_dataset_root / "training"
        if track == "full_data"
        else Path(root).expanduser().resolve() / "fits" / "cytobridge" / split_id
    )
    training_config = (
        formal_dataset_root
        / str(config.get("benchmark", {}).get("training_config", ""))
    )
    saved_config = model_dir / "config.yaml"
    if not training_config.is_file() or not saved_config.is_file():
        return None
    saved = yaml.safe_load(saved_config.read_text(encoding="utf-8"))
    plan = saved.get("training", {}).get("plan") if isinstance(saved, dict) else None
    if not isinstance(plan, list) or not plan:
        return None
    checkpoints = {}
    for stage in plan:
        if not isinstance(stage, dict):
            return None
        name = str(stage.get("name", ""))
        if not name:
            return None
        filename = (
            "score_model.pth"
            if str(stage.get("mode", "")).lower() == "score_matching"
            else "last_model.pth"
            if str(stage.get("save_strategy", "best")).lower() == "last"
            else "best_model.pth"
        )
        checkpoint = model_dir / name / filename
        if not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
            return None
        checkpoints[name] = _immutable_sha256(
            checkpoint, validation_cache
        )
    saved_adata = model_dir / "adata.h5ad"
    if not saved_adata.is_file() or saved_adata.stat().st_size <= 0:
        return None
    return {
        "training_config_source_sha256": _immutable_sha256(
            training_config, validation_cache
        ),
        "saved_config_sha256": _immutable_sha256(
            saved_config, validation_cache
        ),
        "checkpoint_sha256": checkpoints,
        "saved_adata_path": saved_adata.resolve(),
        "saved_adata_sha256": _immutable_sha256(
            saved_adata, validation_cache
        ),
    }


def _static_run_manifest_is_current(
    root,
    method,
    track,
    target,
    contract,
    prediction,
    summary_path,
    validation_cache=None,
):
    """Bind every static target to its exact current adapter recipe and outputs."""

    target_dir = prediction.parent
    run_dir = target_dir if track == "loto" else target_dir.parent
    run_manifest = run_dir / "run_manifest.json"
    if method not in STATIC or not run_manifest.is_file() or not STATIC_REGISTRY.is_file():
        return False

    registry = json.loads(STATIC_REGISTRY.read_text(encoding="utf-8"))
    methods = registry.get("methods") if isinstance(registry, dict) else None
    method_spec = methods.get(method) if isinstance(methods, dict) else None
    representations = (
        method_spec.get("representations") if isinstance(method_spec, dict) else None
    )
    representation_spec = (
        representations.get("matched_state_spatial")
        if isinstance(representations, dict)
        else None
    )
    if not isinstance(method_spec, dict) or not isinstance(representation_spec, dict):
        return False
    expected_parameters = representation_spec.get("default_parameters")
    if not isinstance(expected_parameters, dict):
        return False

    manifest = json.loads(run_manifest.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        return False
    expected_mode = "loto" if track == "loto" else "no-holdout"
    protocol = manifest.get("protocol")
    inputs = manifest.get("input")
    output_scope = manifest.get("output_scope")
    if not all(isinstance(value, dict) for value in (protocol, inputs, output_scope)):
        return False
    if (
        manifest.get("schema_version") != "2.0.0"
        or manifest.get("status") != "complete"
        or manifest.get("dataset") != contract["dataset_id"]
        or manifest.get("method") != method
        or manifest.get("representation") != "matched_state_spatial"
        or manifest.get("method_spec") != method_spec
        or manifest.get("representation_spec") != representation_spec
        or manifest.get("parameters") != expected_parameters
        or manifest.get("seed") != 20_260_718
        or manifest.get("max_fit_n") != 800
        or manifest.get("adapter_implementation")
        != current_static_adapter_implementation(validation_cache)
        or manifest.get("dry_run") is not False
        or manifest.get("prediction_written") is not True
        or manifest.get("primary_benchmark_eligible") is not True
        or manifest.get("surrogate_attempted") is not False
        or protocol.get("mode") != expected_mode
        or int(protocol.get("prediction_n", -1)) != contract["prediction_n"]
        or protocol.get("truth_artifact_opened") is not False
        or protocol.get("truth_cell_count_read") is not False
        or protocol.get("target_n_used_for_prediction") is not False
        or output_scope.get("scope") != representation_spec.get("output_scope")
        or output_scope.get("hybrid_adapter")
        is not bool(representation_spec.get("hybrid", False))
        or int(output_scope.get("state_dimensions", -1)) != contract["state_dim"]
        or int(output_scope.get("spatial_dimensions", -1))
        != (0 if method == "wot" else contract["spatial_dim"])
        or output_scope.get("weights_exported") is not False
        or output_scope.get("growth_or_total_mass_evaluated") is not False
    ):
        return False
    if track == "loto":
        if (
            not np.isclose(
                float(protocol.get("requested_target_time", np.nan)),
                float(target),
                rtol=0.0,
                atol=1e-8,
            )
            or not np.isclose(
                float(protocol.get("loto_target", np.nan)),
                float(target),
                rtol=0.0,
                atol=1e-8,
            )
            or protocol.get("comparable_to_strict_loto") is not True
        ):
            return False
    elif (
        protocol.get("requested_target_time") is not None
        or protocol.get("loto_target") is not None
        or protocol.get("no_holdout_is_in_sample") is not True
    ):
        return False

    current_input_manifest = contract["manifest_path"]
    current_payload = json.loads(current_input_manifest.read_text(encoding="utf-8"))
    current_split = current_payload.get("splits", {}).get(contract["split_id"])
    current_train = current_split.get("train") if isinstance(current_split, dict) else None
    if not isinstance(current_train, dict):
        return False

    time_counts = current_split.get("train_time_counts")
    if not isinstance(time_counts, dict) or not time_counts:
        return False
    observed_time_grid = []
    for raw_time, count in time_counts.items():
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            return False
        try:
            numeric_time = float(raw_time)
        except (TypeError, ValueError):
            return False
        if not np.isfinite(numeric_time):
            return False
        if count > 0:
            observed_time_grid.append(numeric_time)
    observed_time_grid.sort()
    if len(observed_time_grid) < 2 or len(set(observed_time_grid)) != len(
        observed_time_grid
    ):
        return False
    declared_time_grid = protocol.get("time_values")
    if not isinstance(declared_time_grid, list) or any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        for value in declared_time_grid
    ):
        return False
    declared_time_grid = [float(value) for value in declared_time_grid]
    if (
        not all(np.isfinite(value) for value in declared_time_grid)
        or declared_time_grid != observed_time_grid
    ):
        return False
    if track == "loto":
        left = [value for value in observed_time_grid if value < float(target)]
        right = [value for value in observed_time_grid if value > float(target)]
        if (
            not left
            or not right
            or any(value == float(target) for value in observed_time_grid)
        ):
            return False
        expected_pairs = [(max(left), min(right))]
        if contract["source_time"] != expected_pairs[0][0]:
            return False
    else:
        expected_pairs = list(zip(observed_time_grid[:-1], observed_time_grid[1:]))
        if contract["source_time"] != observed_time_grid[0]:
            return False

    def current_artifact(name):
        record = current_train.get(name)
        if not isinstance(record, dict):
            return None
        raw_path = record.get("relative_path", record.get("path"))
        digest = record.get("sha256")
        if not isinstance(raw_path, str) or not isinstance(digest, str):
            return None
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = current_input_manifest.parent / path
        return path.resolve(), digest.lower()

    train_h5ad = current_artifact("h5ad")
    training_reference = current_artifact("training_reference_npz")
    source_roster = current_artifact("source_roster_npz")
    if None in (train_h5ad, training_reference, source_roster):
        return False
    if (
        Path(str(inputs.get("input_manifest", ""))).resolve()
        != current_input_manifest
        or inputs.get("input_manifest_sha256") != contract["manifest_sha256"]
        or inputs.get("input_manifest_split") != contract["split_id"]
        or inputs.get("input_manifest_h5ad_verified") is not True
        or Path(str(inputs.get("train_h5ad", ""))).resolve() != train_h5ad[0]
        or inputs.get("train_h5ad_sha256") != train_h5ad[1]
        or Path(str(inputs.get("training_reference", ""))).resolve()
        != training_reference[0]
        or inputs.get("training_reference_sha256") != training_reference[1]
        or inputs.get("input_manifest_training_reference_expected_sha256")
        != training_reference[1]
        or inputs.get("input_manifest_training_reference_verified") is not True
        or Path(str(inputs.get("source_roster", ""))).resolve() != source_roster[0]
        or inputs.get("source_roster_sha256") != source_roster[1]
    ):
        return False

    outputs = manifest.get("outputs")
    by_time = outputs.get("prediction_by_time") if isinstance(outputs, dict) else None
    if not isinstance(by_time, dict):
        return False
    matching = []
    for value, record in by_time.items():
        try:
            matches_target = np.isclose(
                float(value), float(target), rtol=0.0, atol=1e-8
            )
        except (TypeError, ValueError):
            matches_target = False
        if matches_target:
            matching.append(record)
    if len(matching) != 1 or not isinstance(matching[0], dict):
        return False
    output_record = matching[0]
    summary_record = output_record.get("summary")
    if (
        not isinstance(summary_record, dict)
        or Path(str(output_record.get("path", ""))).resolve() != prediction.resolve()
        or output_record.get("sha256") != sha256_file(prediction)
        or Path(str(summary_record.get("path", ""))).resolve()
        != summary_path.resolve()
        or summary_record.get("sha256") != sha256_file(summary_path)
    ):
        return False

    if method in EXTERNAL_STATIC:
        reference_commit = method_spec.get("reference_commit")
        official_runs = manifest.get("official_runs")
        if (
            not isinstance(reference_commit, str)
            or not reference_commit
            or not isinstance(official_runs, list)
            or len(official_runs) != len(expected_pairs)
            or not expected_pairs
            or manifest.get("control_run") is not None
        ):
            return False
        for run, (expected_from, expected_to) in zip(official_runs, expected_pairs):
            dependency = run.get("dependency") if isinstance(run, dict) else None
            if not isinstance(dependency, dict):
                return False
            source_root = Path(str(dependency.get("requested_source_root", ""))).resolve()
            module_path = Path(str(dependency.get("module_file", ""))).resolve()
            try:
                module_path.relative_to(source_root)
            except ValueError:
                return False
            if (
                isinstance(run.get("from"), bool)
                or not isinstance(run.get("from"), (int, float))
                or float(run["from"]) != expected_from
                or isinstance(run.get("to"), bool)
                or not isinstance(run.get("to"), (int, float))
                or float(run["to"]) != expected_to
                or run.get("parameters") != expected_parameters
                or run.get("official_api") != method_spec.get("official_api")
                or run.get("representation") != "matched_state_spatial"
                or dependency.get("available") is not True
                or dependency.get("source_mode") is not True
                or dependency.get("source_commit_verified") is not True
                or dependency.get("git_dirty") is not False
                or dependency.get("expected_git_commit") != reference_commit
                or dependency.get("git_commit") != reference_commit
                or dependency.get("distribution_metadata_required") is not False
                or dependency.get("module_from_requested_source") is not True
                or Path(str(dependency.get("git_toplevel", ""))).resolve()
                != source_root
                or dependency.get("compatibility_versions")
                != method_spec.get("compatibility_versions", {})
            ):
                return False
    else:
        control = manifest.get("control_run")
        if (
            not isinstance(control, dict)
            or control.get("control") != method
            or manifest.get("official_runs") not in (None, [])
        ):
            return False
    return True


def target_output_is_complete(
    root,
    method,
    track,
    target,
    config,
    formal_dataset_root=None,
    validation_cache=None,
):
    """Fail closed unless one target matches the current immutable contract."""

    target_dir = root / "predictions" / track / method / f"t{target}"
    prediction, summary_path = target_dir / "prediction.npz", target_dir / "summary.json"
    if not prediction.is_file() or not summary_path.is_file():
        return False
    try:
        contract = _current_prediction_contract(
            root, config, track, target, validation_cache
        )
        if contract is None:
            return False
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if not isinstance(summary, dict) or summary.get("status") not in {
            "complete",
            "completed",
        }:
            return False
        if summary.get("method") != METHOD_NAME[method]:
            return False
        if summary.get("track") != track or summary.get("regime") != track:
            return False
        if summary.get("dataset", summary.get("dataset_id")) != contract["dataset_id"]:
            return False
        expected_split_id = contract["split_id"]
        declared_split_id = summary.get(
            "split_id",
            summary.get(
                "input_manifest_split",
                expected_split_id if method not in {"cytobridge", *DYNAMIC} else None,
            ),
        )
        if declared_split_id != expected_split_id:
            return False
        for summary_key, contract_key in (
            ("train_h5ad_sha256", "train_h5ad_sha256"),
            ("training_reference_sha256", "training_reference_sha256"),
            ("source_roster_sha256", "source_roster_sha256"),
        ):
            if summary.get(summary_key) != contract[contract_key]:
                return False
        if summary.get("input_manifest_sha256") != contract["manifest_sha256"]:
            return False
        if summary.get("primary_benchmark_eligible") is not True:
            return False
        expected_scope = OUTPUT_SCOPE_BY_METHOD.get(method)
        if expected_scope is None or str(summary.get("output_scope", "")).lower() != expected_scope:
            return False
        effective_adapter = str(
            summary.get("native_vs_adapter", summary.get("output_scope", ""))
        ).lower()
        if effective_adapter != NATIVE_VS_ADAPTER_BY_METHOD[method]:
            return False
        if "source_time" not in summary or not np.isclose(
            float(summary["source_time"]),
            contract["source_time"],
            rtol=0.0,
            atol=1e-8,
        ):
            return False
        if method in {"cytobridge", *DYNAMIC}:
            if summary.get("truth_inputs_opened") is not False:
                return False
        else:
            if (
                summary.get("truth_artifact_opened") is not False
                or summary.get("target_n_used_for_prediction") is not False
            ):
                return False
        declared_manifest = summary.get("input_manifest")
        if declared_manifest is not None and Path(str(declared_manifest)).resolve() != contract[
            "manifest_path"
        ]:
            return False
        if _summary_prediction_n(summary) != contract["prediction_n"]:
            return False
        if "target_time" not in summary or not np.isclose(
            float(summary["target_time"]), float(target), rtol=0.0, atol=1e-8
        ):
            return False
        if "state_dim" in summary and int(summary["state_dim"]) != contract["state_dim"]:
            return False
        if "spatial_dim" in summary and int(summary["spatial_dim"]) != contract[
            "spatial_dim"
        ]:
            return False
        if method == "cytobridge" and (
            int(summary.get("seed", -1)) != 42
            or summary.get("interaction_mode")
            != contract["cytobridge_interaction_mode"]
            or not _cytobridge_interaction_summary_is_consistent(summary)
            or summary.get("adapter_implementation")
            != current_cytobridge_adapter_implementation(validation_cache)
        ):
            return False
        evaluator_summary = prediction.with_suffix(".summary.json")
        if method == "cytobridge":
            if (
                not evaluator_summary.is_file()
                or evaluator_summary.read_bytes() != summary_path.read_bytes()
            ):
                return False
        elif evaluator_summary.exists():
            return False
        if method in STATIC and not _static_run_manifest_is_current(
            root,
            method,
            track,
            target,
            contract,
            prediction,
            summary_path,
            validation_cache,
        ):
            return False
        if method == "cytobridge" and (
            summary.get("split_sde") is not False
            or summary.get("continuous_across_targets") is not (track == "full_data")
            or not _cytobridge_source_roster_is_current(summary, contract)
            or (
                track == "full_data"
                and not _cytobridge_full_run_summary_is_current(
                    root, summary, contract, config
                )
            )
        ):
            return False
        if method == "cytobridge" and formal_dataset_root is not None:
            model_contract = _current_cytobridge_model_contract(
                root,
                formal_dataset_root,
                config,
                track,
                expected_split_id,
                validation_cache,
            )
            if model_contract is None:
                return False
            if (
                summary.get("config_sha256")
                != model_contract["saved_config_sha256"]
                or summary.get("checkpoint_sha256")
                != model_contract["checkpoint_sha256"]
            ):
                return False
            reference_match = summary.get("training_reference_match")
            if not isinstance(reference_match, dict):
                return False
            if track == "full_data":
                if (
                    reference_match.get("proof")
                    != "saved_adata_exact_frozen_arrays"
                    or Path(str(reference_match.get("path", ""))).resolve()
                    != model_contract["saved_adata_path"]
                    or reference_match.get("sha256")
                    != model_contract["saved_adata_sha256"]
                ):
                    return False
            else:
                fit_summary = (
                    Path(root).expanduser().resolve()
                    / "fits"
                    / "cytobridge"
                    / expected_split_id
                    / "benchmark_fit_summary.json"
                )
                if (
                    reference_match.get("proof") != "benchmark_fit_summary"
                    or Path(str(reference_match.get("path", ""))).resolve()
                    != fit_summary.resolve()
                    or not fit_summary.is_file()
                    or reference_match.get("sha256") != sha256_file(fit_summary)
                ):
                    return False
                fit_payload = json.loads(fit_summary.read_text(encoding="utf-8"))
                prepare_summary = Path(
                    str(fit_payload.get("prepare_graph_summary", ""))
                ).expanduser().resolve()
                if (
                    fit_payload.get("training_config_source_sha256")
                    != model_contract["training_config_source_sha256"]
                    or fit_payload.get("saved_config_sha256")
                    != model_contract["saved_config_sha256"]
                    or fit_payload.get("checkpoint_sha256")
                    != model_contract["checkpoint_sha256"]
                    or not prepare_summary.is_file()
                    or fit_payload.get("prepare_graph_summary_sha256")
                    != sha256_file(prepare_summary)
                ):
                    return False
                direct_match = fit_payload.get("training_reference_match")
                if (
                    not isinstance(direct_match, dict)
                    or direct_match.get("proof")
                    != "saved_adata_exact_frozen_arrays"
                    or Path(str(direct_match.get("path", ""))).resolve()
                    != model_contract["saved_adata_path"]
                    or direct_match.get("sha256")
                    != model_contract["saved_adata_sha256"]
                ):
                    return False
        if method in DYNAMIC:
            fit_manifest = root / "fits" / method / expected_split_id / "fit_manifest.json"
            seed_token = "full_trajectory" if track == "full_data" else "loto_trajectory"
            expected_infer_seed = stable_seed(
                DYNAMIC_SEED,
                contract["dataset_id"],
                expected_split_id,
                method,
                seed_token,
            )
            if (
                not dynamic_fit_is_complete(
                    root,
                    method,
                    expected_split_id,
                    contract["manifest_path"],
                    validation_cache,
                )
                or not fit_manifest.is_file()
                or Path(str(summary.get("fit_manifest", ""))).resolve()
                != fit_manifest.resolve()
                or summary.get("fit_manifest_sha256") != sha256_file(fit_manifest)
                or summary.get("seed_base") != DYNAMIC_SEED
                or summary.get("params") != DYNAMIC_DEFAULT_PARAMS[method]
                or summary.get("adapter_implementation")
                != current_dynamic_adapter_implementation(validation_cache)
                or summary.get("infer_seed") != expected_infer_seed
                or summary.get("shared_full_trajectory_seed_across_targets")
                is not (track == "full_data")
            ):
                return False
        declared_path = summary.get("prediction_npz")
        if declared_path is None and isinstance(summary.get("prediction"), dict):
            declared_path = summary["prediction"].get("path")
        if declared_path is None or Path(str(declared_path)).resolve() != prediction.resolve():
            return False
        declared_sha = summary.get("prediction_npz_sha256") or summary.get(
            "prediction_sha256"
        )
        if declared_sha is None and isinstance(summary.get("prediction"), dict):
            declared_sha = summary["prediction"].get("sha256")
        if not isinstance(declared_sha, str) or declared_sha.lower() != sha256_file(prediction):
            return False
        with np.load(prediction, allow_pickle=False) as archive:
            keys = set(archive.files)
            if "state" not in keys:
                return False
            state = np.asarray(archive["state"])
            if (
                state.ndim != 2
                or state.shape[0] == 0
                or state.shape[1] != contract["state_dim"]
                or not np.isfinite(state).all()
            ):
                return False
            if method == "stvcr":
                if int(summary.get("native_output_n", -1)) != state.shape[0]:
                    return False
                count_changed = state.shape[0] != contract["prediction_n"]
                if summary.get("native_count_changed") is not count_changed:
                    return False
                if count_changed and summary.get("native_growth") is not True:
                    return False
            elif state.shape[0] != contract["prediction_n"]:
                return False
            if "spatial" in keys:
                spatial = np.asarray(archive["spatial"])
                if (
                    spatial.ndim != 2
                    or spatial.shape[0] != state.shape[0]
                    or spatial.shape[1] != contract["spatial_dim"]
                    or not np.isfinite(spatial).all()
                ):
                    return False
            expects_spatial = expected_scope in {"native_joint", "hybrid_joint"}
            if ("spatial" in keys) is not expects_spatial:
                return False
            expects_native_mass = method in {"cytobridge", "stvcr"}
            if (
                summary.get("native_mass") is not expects_native_mass
                or summary.get("native_growth") is not expects_native_mass
                or summary.get("weights_are_unnormalised") is not expects_native_mass
                or ("weights" in keys) is not expects_native_mass
            ):
                return False
            if "weights" in keys:
                weights = np.asarray(archive["weights"]).reshape(-1)
                if (
                    weights.shape != (state.shape[0],)
                    or not np.isfinite(weights).all()
                    or np.any(weights < 0)
                    or weights.sum() <= 0
                ):
                    return False
                weight_sum = float(weights.sum())
                if method == "cytobridge":
                    if not np.isclose(
                        float(summary.get("predicted_mass", np.nan)),
                        weight_sum,
                        rtol=1e-7,
                        atol=1e-8,
                    ):
                        return False
                elif method == "stvcr":
                    expected_weight = 1.0 / contract["prediction_n"]
                    if (
                        summary.get("weight_semantics")
                        != "uniform initial-particle mass 1/5000; "
                        "sum(weights)=native_n/5000"
                        or not np.allclose(
                            weights,
                            expected_weight,
                            rtol=1e-6,
                            atol=1e-9,
                        )
                        or not np.isclose(
                            float(summary.get("weight_sum", np.nan)),
                            weight_sum,
                            rtol=1e-7,
                            atol=1e-8,
                        )
                        or not np.isclose(
                            float(summary.get("growth_ratio", np.nan)),
                            state.shape[0] / contract["prediction_n"],
                            rtol=1e-7,
                            atol=1e-8,
                        )
                    ):
                        return False
    except (
        KeyError,
        OSError,
        ValueError,
        TypeError,
        json.JSONDecodeError,
        yaml.YAMLError,
    ):
        return False
    return True


def job_outputs_are_complete(
    root,
    method,
    track,
    targets,
    config,
    formal_dataset_root=None,
    validation_cache=None,
):
    """Return whether every target emitted the immutable prediction pair."""

    return all(
        target_output_is_complete(
            root,
            method,
            track,
            target,
            config,
            formal_dataset_root,
            validation_cache,
        )
        for target in targets
    )


def _directory_has_entries(path):
    path = Path(path)
    if not path.exists():
        return False
    if not path.is_dir():
        return True
    return next(path.iterdir(), None) is not None


def dynamic_fit_is_complete(
    root, method, split, input_manifest, validation_cache=None
):
    """Validate the completion markers and hash-bound files for a dynamic fit."""

    fit_dir = root / "fits" / method / split
    manifest_path, summary_path = fit_dir / "fit_manifest.json", fit_dir / "summary.json"
    if not manifest_path.is_file() or not summary_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if manifest != summary or manifest.get("status") != "complete":
            return False
        if manifest.get("method") != method or manifest.get("split_id") != split:
            return False
        if (
            manifest.get("seed_base") != DYNAMIC_SEED
            or manifest.get("params") != DYNAMIC_DEFAULT_PARAMS.get(method)
            or manifest.get("method_pin_registry_sha256")
            != _immutable_sha256(DYNAMIC_PINS, validation_cache)
        ):
            return False
        pin_registry = json.loads(DYNAMIC_PINS.read_text(encoding="utf-8"))
        current_pin = pin_registry.get("methods", {}).get(method)
        if (
            not isinstance(current_pin, dict)
            or manifest.get("method_pin") != current_pin
            or manifest.get("source_expected_git_commit") != current_pin.get("commit")
            or manifest.get("source_git_commit") != current_pin.get("commit")
            or manifest.get("source_tracked_tree_clean") is not True
            or manifest.get("source_worktree_clean") is not True
        ):
            return False
        if Path(str(manifest.get("fit_manifest", ""))).resolve() != manifest_path.resolve():
            return False
        if Path(str(manifest.get("summary", ""))).resolve() != summary_path.resolve():
            return False
        if (
            not Path(input_manifest).is_file()
            or manifest.get("input_manifest_sha256")
            != _immutable_sha256(input_manifest, validation_cache)
        ):
            return False
        input_payload = json.loads(Path(input_manifest).read_text(encoding="utf-8"))
        split_payload = input_payload.get("splits", {}).get(split)
        split_train = split_payload.get("train") if isinstance(split_payload, dict) else None
        if not isinstance(split_train, dict):
            return False
        dataset_id = input_payload.get("dataset_id", input_payload.get("dataset"))
        protocol = str(split_payload.get("protocol", ""))
        expected_regime = (
            "full_data"
            if split == "full_data" or protocol == "full_data"
            else "loto"
            if split.startswith("loto_t")
            or protocol == "leave_one_timepoint_out"
            else None
        )
        expected_fit_seed = stable_seed(
            DYNAMIC_SEED, dataset_id, split, method, "fit"
        )
        if (
            not isinstance(dataset_id, str)
            or not dataset_id
            or expected_regime is None
            or manifest.get("dataset") != dataset_id
            or manifest.get("regime") != expected_regime
            or isinstance(manifest.get("fit_seed"), bool)
            or manifest.get("fit_seed") != expected_fit_seed
            or manifest.get("adapter_implementation")
            != current_dynamic_adapter_implementation(validation_cache)
        ):
            return False
        canonical_artifacts = {
            "train_h5ad_sha256": split_train.get("h5ad", {}).get("sha256"),
            "training_reference_sha256": split_train.get(
                "training_reference_npz", {}
            ).get("sha256"),
            "source_roster_sha256": split_train.get("source_roster_npz", {}).get(
                "sha256"
            ),
        }
        if any(manifest.get(key) != digest for key, digest in canonical_artifacts.items()):
            return False
        artifacts = manifest.get("artifacts")
        required_artifacts = DYNAMIC_REQUIRED_ARTIFACTS.get(method)
        if (
            not isinstance(artifacts, dict)
            or required_artifacts is None
            or not required_artifacts <= set(artifacts)
        ):
            return False
        for name in required_artifacts:
            artifact = artifacts[name]
            if not isinstance(artifact, dict) or set(artifact) < {"path", "sha256"}:
                return False
            digest = artifact.get("sha256")
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                return False
            artifact_path = Path(str(artifact["path"]))
            if not artifact_path.is_absolute():
                artifact_path = fit_dir / artifact_path
            if not artifact_path.is_file() or digest != sha256_file(artifact_path):
                return False
        roster_artifact = artifacts["source_roster"]
        roster_contract = manifest.get("source_roster")
        canonical_roster = split_train["source_roster_npz"]
        canonical_roster_path = Path(
            str(canonical_roster.get("relative_path", canonical_roster.get("path", "")))
        ).expanduser()
        if not canonical_roster_path.is_absolute():
            canonical_roster_path = Path(input_manifest).parent / canonical_roster_path
        if (
            not isinstance(roster_contract, dict)
            or roster_contract.get("canonical_input_roster_sha256")
            != canonical_artifacts["source_roster_sha256"]
            or Path(str(roster_contract.get("canonical_input_roster", ""))).resolve()
            != canonical_roster_path.resolve()
            or roster_artifact.get("sha256")
            != canonical_artifacts["source_roster_sha256"]
        ):
            return False
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False
    return True


def commands_for_safe_resume(
    root,
    method,
    track,
    targets,
    commands,
    config,
    formal_dataset_root=None,
    validation_cache=None,
):
    """Return commands safe to run in-place, or a fail-closed resume reason."""

    complete = {
        target: target_output_is_complete(
            root,
            method,
            track,
            target,
            config,
            formal_dataset_root,
            validation_cache,
        )
        for target in targets
    }
    if all(complete.values()):
        return [], None

    invalid_nonempty = [
        target
        for target in targets
        if not complete[target]
        and _directory_has_entries(
            root / "predictions" / track / method / f"t{target}"
        )
    ]
    if invalid_nonempty:
        return [], (
            f"partial/corrupt {method} {track} target output(s) "
            f"{invalid_nonempty} cannot be resumed in place; use a new --run-root"
        )

    split = "full_data" if track == "full_data" else f"loto_t{targets[0]}"
    if method in DYNAMIC:
        fit_dir = root / "fits" / method / split
        if dynamic_fit_is_complete(
            root,
            method,
            split,
            root / "inputs" / "manifest.json",
            validation_cache,
        ):
            if len(commands) != len(targets) + 1:
                return [], f"internal error: unexpected {method} command plan"
            return [
                infer
                for target, infer in zip(targets, commands[1:])
                if not complete[target]
            ], None
        if _directory_has_entries(fit_dir) or any(complete.values()):
            return [], (
                f"partial/unverified {method} {split} fit cannot be resumed in place; "
                "use a new --run-root"
            )
        return commands, None

    shared_paths = (
        [root / "predictions" / track / method]
        if track == "full_data"
        else []
    )
    if method == "cytobridge" and track == "loto":
        shared_paths += [
            root / "graphs" / split,
            root / "fits" / "cytobridge" / split,
        ]
    if any(_directory_has_entries(path) for path in shared_paths):
        return [], (
            f"partial {method} {track} shared-trajectory job cannot be resumed in "
            "place; use a new --run-root"
        )
    return commands, None


def merge_status_rows(path, rows):
    """Update method/track/target rows without erasing separately run methods."""

    columns = ("track", "target", "method", "status", "reason", "elapsed_seconds")
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        merged = {}
        if path.is_file():
            with path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    merged[(row["track"], int(row["target"]), row["method"])] = row
        for row in rows:
            merged[(row["track"], int(row["target"]), row["method"])] = row
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            with temporary.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=columns)
                writer.writeheader()
                writer.writerows(
                    sorted(
                        merged.values(),
                        key=lambda row: (
                            row["track"],
                            int(row["target"]),
                            row["method"],
                        ),
                    )
                )
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()


def _evaluation_status_rows(path, *, track):
    """Read the exact explicit status contract used to justify an NA result."""

    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        raw_reader = csv.reader(handle)
        raw_header = next(raw_reader, None)
        if raw_header is None:
            raise RuntimeError("benchmark status table is empty")
        for line_number, raw_row in enumerate(raw_reader, start=2):
            if raw_row and len(raw_row) != len(raw_header):
                raise RuntimeError(
                    "benchmark status table row has a different number of fields "
                    f"than its header at line {line_number}"
                )
        handle.seek(0)
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        if len(fieldnames) != len(set(fieldnames)):
            duplicates = sorted(
                {name for name in fieldnames if fieldnames.count(name) > 1}
            )
            raise RuntimeError(
                f"benchmark status table contains duplicate columns {duplicates}"
            )
        required = {"target", "method", "status"}
        missing = sorted(required.difference(fieldnames))
        if missing:
            raise RuntimeError(f"benchmark status table is missing columns {missing}")
        rows = {}
        for row in reader:
            raw_track = str(row.get("track", track)).strip().lower().replace(" ", "_")
            try:
                row_track = EVALUATION_TRACK_ALIASES[raw_track]
            except KeyError as exc:
                raise RuntimeError(
                    f"unknown benchmark track/regime {row.get('track')!r}"
                ) from exc
            if row_track != track:
                continue
            try:
                numeric_target = float(row["target"])
                target = int(numeric_target)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"benchmark status target is not an integer: {row['target']!r}"
                ) from exc
            if not np.isfinite(numeric_target) or numeric_target != target:
                raise RuntimeError(
                    f"benchmark status target is not an integer: {row['target']!r}"
                )
            key = (str(row["method"]).strip(), target)
            if not key[0]:
                raise RuntimeError("benchmark status table contains an empty method")
            if key in rows:
                raise RuntimeError(
                    f"duplicate benchmark status row for {key[0]}/t{target}"
                )
            status = (
                str(row["status"])
                .strip()
                .lower()
                .replace("-", "_")
                .replace(" ", "_")
            )
            status = EVALUATION_STATUS_ALIASES.get(status, status)
            if status in {"complete", "success"}:
                status = "completed"
            if status not in {"completed", *EVALUATION_NON_NUMERIC_STATUSES}:
                raise RuntimeError(
                    f"unknown benchmark status {row['status']!r} for "
                    f"{key[0]}/t{target}"
                )
            rows[key] = status
    return rows


def run_or_print(commands, dry_run):
    for item in commands:
        print(shlex.join(item))
        if not dry_run:
            subprocess.run(item, cwd=REPO, check=True)


def prepare(name, _cfg, args):
    root = args.run_root / name
    build = command("scripts.spatiotemporal_benchmark.build_inputs", sys.executable,
                    "--config", CONFIG_DIR / f"{name}.yaml", "--h5ad",
                    args.formal_root / name / "preprocess" / f"{name}_aligned.h5ad", "--output-dir", root)
    if args.overwrite: build.append("--overwrite")
    verify = command("scripts.spatiotemporal_benchmark.verify_inputs", sys.executable, "--output-dir", root)
    run_or_print([build, verify], args.dry_run)


def run_dataset(name, cfg, args, pythons, sources):
    root = args.run_root / name
    formal = args.formal_root / name
    validation_cache = ResumeValidationCache()
    rows = []
    for method, track, targets, commands, required, fixed in jobs_for_dataset(name, cfg, args, pythons, sources):
        if args.dry_run:
            run_or_print(commands, True)
            continue
        started = time.monotonic()
        log = args.run_root / name / "logs" / f"{track}_{method}_{targets[0]}.log"
        if fixed:
            status, reason = fixed, "matched signed-PC benchmark is not applicable"
        elif job_outputs_are_complete(
            root,
            method,
            track,
            targets,
            cfg,
            formal,
            validation_cache,
        ):
            status, reason = "completed", ""
        else:
            runnable, resume_reason = commands_for_safe_resume(
                root,
                method,
                track,
                targets,
                commands,
                cfg,
                formal,
                validation_cache,
            )
            run_or_print(runnable, True)
            if resume_reason:
                status, reason = "failed", resume_reason
            else:
                status, reason = execute(runnable, required, args.timeout, log)
        elapsed = round(time.monotonic() - started, 3)
        for target in targets:
            target_complete = target_output_is_complete(
                root,
                method,
                track,
                target,
                cfg,
                formal,
                validation_cache,
            )
            target_status = "completed" if target_complete else status
            target_reason = "" if target_complete else reason
            if target_status == "completed" and not target_complete:
                target_status = "failed"
                target_reason = "job exited without prediction.npz and summary.json"
            rows.append({"track": track, "target": target, "method": METHOD_NAME[method], "status": target_status,
                         "reason": target_reason, "elapsed_seconds": elapsed})
    if not args.dry_run:
        path = args.run_root / name / "status" / "method_target_status.csv"
        merge_status_rows(path, rows)


def evaluate(name, cfg, args):
    root = args.run_root / name
    formal = args.formal_root / name
    validation_cache = ResumeValidationCache()
    for track in args.tracks:
        targets = cfg["loto_targets"] if track == "loto" else cfg["full_data_targets"]
        status_path = root / "status" / "method_target_status.csv"
        declared = (
            {}
            if args.dry_run
            else _evaluation_status_rows(status_path, track=track)
        )
        incomplete = []
        contradictions = []
        if not args.dry_run:
            for method in PRIMARY_METHODS:
                display_name = METHOD_NAME[method]
                for target in targets:
                    complete = target_output_is_complete(
                        root,
                        method,
                        track,
                        target,
                        cfg,
                        formal,
                        validation_cache,
                    )
                    status = declared.get((display_name, int(target)))
                    if complete:
                        if status in EVALUATION_NON_NUMERIC_STATUSES:
                            contradictions.append(
                                f"{method}:t{target} has a current prediction but "
                                f"status={status}"
                            )
                    elif status not in EVALUATION_NON_NUMERIC_STATUSES:
                        incomplete.append((method, target))
        if contradictions:
            raise RuntimeError(
                "Refusing evaluation because prediction artifacts contradict "
                "their declared execution status: " + "; ".join(contradictions)
            )
        if incomplete:
            detail = ", ".join(
                f"{method}:t{target}" for method, target in incomplete
            )
            raise RuntimeError(
                "Refusing evaluation because current immutable prediction "
                f"contracts are incomplete for {name}/{track}: {detail}"
            )
        output = root / "evaluation" / track
        score = command("scripts.spatiotemporal_benchmark.evaluate_predictions", sys.executable,
                        "--input-manifest", root / "inputs" / "manifest.json", "--predictions-root", root / "predictions" / track,
                        "--track", track, "--targets", *targets,
                        "--methods", *(METHOD_NAME[method] for method in PRIMARY_METHODS),
                        "--output-dir", output)
        if status_path.is_file():
            score.extend(["--status-table", str(status_path)])
        report = command("scripts.spatiotemporal_benchmark.summarize_results", sys.executable,
                         "--metrics-long", output / f"{track}_metrics_long.csv",
                         "--evaluation-manifest", output / f"{track}_evaluation_manifest.json",
                         "--method-registry", METHOD_REGISTRY,
                         "--output-dir", root / "reports" / track)
        run_or_print([score, report], args.dry_run)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--formal-root", type=Path, default=DEFAULT_FORMAL_ROOT)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--software-root", type=Path, default=REPO / "software")
    parser.add_argument("--dry-run", action="store_true")
    sub = parser.add_subparsers(dest="action", required=True)
    prep = sub.add_parser("prepare"); prep.add_argument("--overwrite", action="store_true")
    run = sub.add_parser("run"); run.add_argument("--methods", nargs="+", choices=METHODS, default=list(PRIMARY_METHODS))
    run.add_argument("--tracks", nargs="+", choices=("loto", "full_data"), default=["loto", "full_data"])
    run.add_argument("--timeout", type=int, default=3600); run.add_argument("--device", default="cuda")
    run.add_argument("--python", action="append", default=[]); run.add_argument("--source", action="append", default=[])
    report = sub.add_parser("evaluate")
    report.add_argument("--tracks", nargs="+", choices=("loto", "full_data"), default=["loto", "full_data"])
    args = parser.parse_args(argv); configs = load_datasets(args.datasets)
    if args.action == "prepare":
        for name, cfg in configs.items(): prepare(name, cfg, args)
    elif args.action == "run":
        pythons, sources = assignments(args.python), assignments(args.source)
        for name, cfg in configs.items(): run_dataset(name, cfg, args, pythons, sources)
    else:
        for name, cfg in configs.items(): evaluate(name, cfg, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
# fmt: on
