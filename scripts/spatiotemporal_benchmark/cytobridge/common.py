"""Strict contracts and provenance helpers for the CytoBridge adapter.

This module deliberately resolves only the training H5AD and training-reference
NPZ from the benchmark root manifest.  Truth artifacts are never resolved or
opened by training or inference code.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


METHOD = "CytoBridge-0.015"
CONTRACT_VERSION = "cytobridge-spatiotemporal-benchmark-input-v1"
CONTRACT_UNS_KEY = "cytobridge_benchmark_contract"
PREDICTION_N = 5000
SEED = 42
ALPHA_EXPRESS = 0.015
ALPHA_SPATIAL = 10.0
SIGMA = 0.03
LEGACY_FULL_EDGE_MODEL_SHA256 = (
    "ee3a4ef216ca8b268bcd6a57ed388fcca6abe88bb5652d16b8d69415e05cd4c1"
)

STAGE_PROFILE: tuple[dict[str, Any], ...] = (
    {
        "name": "Pretrain",
        "mode": "neural_ode",
        "epochs": 100,
        "batch_size": 512,
        "OT_loss": "weighted_emd_detach",
        "train_strategy": "v+g",
        "lambda_ot": 1.0,
        "lambda_mass": 0.01,
        "lambda_energy": 0.0,
        "global_mass": False,
        "reverse_mass_norm": True,
        "reverse_mass_offset": False,
        "checkpoint_metric": "legacy_forward_last_ot",
    },
    {
        "name": "Refine",
        "mode": "neural_ode",
        "epochs": 100,
        "batch_size": 512,
        "OT_loss": "weighted_emd",
        "train_strategy": "v+g",
        "lambda_ot": 1.0,
        "lambda_mass": 0.01,
        "lambda_energy": 0.0,
        "global_mass": True,
        "reverse_mass_norm": True,
        "reverse_mass_offset": False,
        "checkpoint_metric": "legacy_forward_last_ot",
    },
    {
        "name": "Init_interaction",
        "mode": "neural_ode",
        "epochs": 50,
        "batch_size": 512,
        "OT_loss": "weighted_emd",
        "train_strategy": "v+g+i",
        "lambda_ot": 10.0,
        "lambda_mass": 10.0,
        "lambda_energy": 0.01,
        "global_mass": True,
        "reverse_mass_norm": False,
        "reverse_mass_offset": True,
        "checkpoint_metric": "legacy_forward_last_ot",
    },
    {
        "name": "Train_Score",
        "mode": "score_matching",
        "epochs": 2001,
        "batch_size": 128,
        "train_strategy": "s",
        "sigma": SIGMA,
        "optimizer_type": "adamw",
        "lr": 0.0001,
        "lambda_penalty": 0,
        "save_strategy": "last",
    },
    {
        "name": "Finetune",
        "mode": "neural_ode",
        "epochs": 1000,
        "batch_size": 512,
        "OT_loss": "weighted_emd",
        "train_strategy": "v+g+i",
        "lr": 0.0001,
        "lambda_ot": 10.0,
        "lambda_mass": 10.0,
        "lambda_energy": 0.01,
        "global_mass": True,
        "score_use": True,
        "reverse_mass_norm": False,
        "reverse_mass_offset": True,
        "scheduler_type": "plateau",
        "scheduler_metric": "forward_last_ot",
        "scheduler_step_before_reverse": True,
        "max_grad_norm": 10.0,
        "checkpoint_metric": "legacy_forward_last_ot",
        "save_strategy": "best",
    },
    {
        "name": "Score_Refine",
        "mode": "score_matching",
        "epochs": 2001,
        "batch_size": 128,
        "train_strategy": "s",
        "sigma": SIGMA,
        "optimizer_type": "adamw",
        "lr": 0.0001,
        "lambda_penalty": 0,
        "save_strategy": "last",
    },
)

DEFAULT_PROFILE: dict[str, Any] = {
    "lr": 0.0001,
    "lambda_ot": 10.0,
    "lambda_mass": 10.0,
    "lambda_energy": 0.01,
    "sigma": SIGMA,
    "batch_size": 512,
    "alpha_spatial": ALPHA_SPATIAL,
    "alpha_express": ALPHA_EXPRESS,
    "global_mass": True,
}

MODEL_PROFILE: dict[str, Any] = {
    "components": ("velocity", "growth", "score", "interaction"),
    "interaction_type": "gnn",
    "interaction_group_size": 1024,
    "velocity_net": {
        "hidden_dim": 256,
        "n_layers": 5,
        "residual": False,
        "activation": "leaky_relu",
        "use_spatial": True,
    },
    "growth_net": {
        "hidden_dim": 256,
        "n_layers": 3,
        "residual": False,
        "activation": "leaky_relu",
    },
    "score_net": {
        "hidden_dim": 400,
        "n_layers": 3,
        "activation": "leaky_relu",
    },
    "interaction_net": {
        "hidden_dim": 256,
        "num_heads": 8,
        "num_layers": 1,
        "activation": "leakyrelu",
        "num_rbf": 8,
        "cutoff": 0.09606367405591873,
        "use_spatial": True,
        "rbf_trainable": False,
        "edge_predictor_path": "edge_classifier/zebrafish.pt",
        "edge_predictor_thre": 0.4999999701976776,
    },
}

CANONICAL_CKPT_DIR = "results/zebrafish_spatial_full_alpha_express_0015"
RUNTIME_MODEL_FIELDS = frozenset({"spatial_dim"})
RUNTIME_INTERACTION_FIELDS = frozenset(
    {"cutoff", "edge_predictor_path", "edge_predictor_thre"}
)


class ContractError(ValueError):
    """An immutable input or model artifact violates the benchmark contract."""


@dataclass(frozen=True)
class SplitInput:
    dataset_id: str
    split_id: str
    regime: str
    train_h5ad: Path
    training_reference_npz: Path
    source_roster_npz: Path
    observed_times: tuple[float, ...]
    evaluation_targets: tuple[float, ...]
    holdout_time: float | None
    prediction_n: int
    contract_uns_key: str
    root_manifest: Path
    root_manifest_sha256: str
    train_h5ad_sha256: str
    training_reference_sha256: str
    source_roster_sha256: str
    raw: Mapping[str, Any]


@dataclass(frozen=True)
class TrainingData:
    time: np.ndarray
    spatial: np.ndarray
    state: np.ndarray
    row_id: np.ndarray
    state_key: str
    spatial_key: str
    time_key: str
    row_id_key: str
    benchmark_original_obs_name: np.ndarray | None
    interaction_graph: Mapping[str, Any]
    h5ad_contract: Mapping[str, Any]

    @property
    def n_obs(self) -> int:
        return int(self.time.size)

    @property
    def state_dim(self) -> int:
        return int(self.state.shape[1])

    @property
    def spatial_dim(self) -> int:
        return int(self.spatial.shape[1])

    @property
    def joint_dim(self) -> int:
        return self.spatial_dim + self.state_dim


def canonical_time(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ContractError(f"time must be finite, found {value!r}")
    return result


def same_time(left: float, right: float) -> bool:
    return bool(np.isclose(left, right, rtol=0.0, atol=1e-8))


def plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return [plain(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return value


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _valid_digest(value: Any, label: str) -> str:
    digest = str(value or "").strip().lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ContractError(f"{label} lacks a valid SHA-256")
    return digest


def _artifact(entry: Any, manifest_dir: Path, label: str) -> tuple[Path, str]:
    if not isinstance(entry, Mapping):
        raise ContractError(f"{label} is not an artifact record")
    # Relative paths are the portable contract.  The absolute build-time path is
    # retained only as a fallback for early manifests without relative_path.
    raw = entry.get("relative_path") or entry.get("path")
    if raw is None:
        raise ContractError(f"{label} has no relative_path/path")
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = (manifest_dir / path).resolve()
    expected = _valid_digest(entry.get("sha256"), label)
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    observed = sha256_file(path)
    if observed != expected:
        raise ContractError(
            f"{label} SHA-256 mismatch: manifest={expected}, observed={observed}"
        )
    return path, observed


def _manifest_sidecar_check(path: Path, digest: str) -> None:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not sidecar.is_file():
        return
    token = sidecar.read_text(encoding="utf-8").strip().split()[0].lower()
    if token != digest:
        raise ContractError(f"manifest SHA sidecar does not match {path}")


def _times_from_counts(value: Any) -> tuple[float, ...]:
    if not isinstance(value, Mapping):
        raise ContractError("split train_time_counts must be a mapping")
    return tuple(
        sorted(canonical_time(key) for key, count in value.items() if int(count) > 0)
    )


def read_split_input(root_manifest: Path, split_id: str) -> SplitInput:
    """Resolve and hash only train-side artifacts for one benchmark split."""
    root_manifest = Path(root_manifest).expanduser().resolve()
    try:
        root = json.loads(root_manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read root manifest {root_manifest}: {exc}") from exc
    if not isinstance(root, Mapping):
        raise ContractError("root manifest must contain a JSON object")
    if str(root.get("contract_version", "")) != CONTRACT_VERSION:
        raise ContractError(
            f"expected contract_version={CONTRACT_VERSION!r}, "
            f"found {root.get('contract_version')!r}"
        )
    root_sha = sha256_file(root_manifest)
    _manifest_sidecar_check(root_manifest, root_sha)
    dataset_id = str(root.get("dataset_id", "")).strip()
    if not dataset_id:
        raise ContractError("root manifest lacks dataset_id")
    splits = root.get("splits")
    if not isinstance(splits, Mapping) or not isinstance(splits.get(split_id), Mapping):
        available = sorted(splits) if isinstance(splits, Mapping) else []
        raise ContractError(f"unknown split {split_id!r}; available={available}")
    entry = dict(splits[split_id])
    protocol = str(entry.get("protocol", ""))
    if split_id == "full_data" and protocol == "full_data":
        regime = "full_data"
    elif split_id.startswith("loto_t") and protocol == "leave_one_timepoint_out":
        regime = "loto"
    else:
        raise ContractError(f"split/protocol mismatch for {split_id!r}: {protocol!r}")

    train = entry.get("train")
    if not isinstance(train, Mapping):
        raise ContractError(f"split {split_id!r} lacks train artifacts")
    train_h5ad, train_sha = _artifact(
        train.get("h5ad"), root_manifest.parent, f"{split_id}/train.h5ad"
    )
    training_reference, reference_sha = _artifact(
        train.get("training_reference_npz"),
        root_manifest.parent,
        f"{split_id}/training_reference.npz",
    )
    source_roster, source_roster_sha = _artifact(
        train.get("source_roster_npz"),
        root_manifest.parent,
        f"{split_id}/source_roster.npz",
    )
    prediction_n = int(entry.get("prediction_n", root.get("prediction_n", 0)))
    if prediction_n != PREDICTION_N:
        raise ContractError(
            f"CytoBridge primary adapter requires prediction_n={PREDICTION_N}, "
            f"found {prediction_n}"
        )
    if entry.get("truth_cell_count_must_not_control_prediction_n") is not True:
        raise ContractError("split must prohibit truth-controlled prediction count")
    if entry.get("transductive_frozen_representation") is not True:
        raise ContractError("split must declare the frozen shared representation")
    if entry.get("representation_refit_per_fold") is not False:
        raise ContractError("representation refitting per fold is forbidden")

    observed = _times_from_counts(entry.get("train_time_counts"))
    targets_raw = entry.get("evaluation_targets")
    if not isinstance(targets_raw, Sequence) or isinstance(targets_raw, (str, bytes)):
        raise ContractError("split evaluation_targets must be a sequence")
    targets = tuple(canonical_time(value) for value in targets_raw)
    if not targets or len(set(targets)) != len(targets):
        raise ContractError("split evaluation_targets must be non-empty and unique")
    holdout_raw = entry.get("held_out_benchmark_time")
    holdout = None if holdout_raw is None else canonical_time(holdout_raw)
    if regime == "loto":
        if holdout is None or len(targets) != 1 or not same_time(targets[0], holdout):
            raise ContractError("LOTO split must expose exactly its held-out target")
        if any(same_time(holdout, value) for value in observed):
            raise ContractError("held-out target appears in train_time_counts")
        if not any(value < holdout for value in observed):
            raise ContractError("held-out target has no previous observed source stage")
        if entry.get("target_rows_physically_removed_from_train") is not True:
            raise ContractError("LOTO split does not assert physical target removal")
    else:
        if holdout is not None:
            raise ContractError("full_data split unexpectedly declares a held-out target")
        if not observed:
            raise ContractError("full_data split has no observed times")
        initial = min(observed)
        for target in targets:
            if target <= initial or not any(same_time(target, value) for value in observed):
                raise ContractError(f"full-data target t{target:g} is not an observed post-t0 stage")

    contract_key = str(entry.get("contract_uns_key", root.get("contract_uns_key", "")))
    if contract_key != CONTRACT_UNS_KEY:
        raise ContractError(
            f"contract_uns_key must be {CONTRACT_UNS_KEY!r}, found {contract_key!r}"
        )
    return SplitInput(
        dataset_id=dataset_id,
        split_id=split_id,
        regime=regime,
        train_h5ad=train_h5ad,
        training_reference_npz=training_reference,
        source_roster_npz=source_roster,
        observed_times=observed,
        evaluation_targets=targets,
        holdout_time=holdout,
        prediction_n=prediction_n,
        contract_uns_key=contract_key,
        root_manifest=root_manifest,
        root_manifest_sha256=root_sha,
        train_h5ad_sha256=train_sha,
        training_reference_sha256=reference_sha,
        source_roster_sha256=source_roster_sha,
        raw=entry,
    )


def _normalise_optional_time(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
        return None
    return canonical_time(value)


def _validated_matrix(name: str, value: Any, rows: int, columns: int) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float32)
    if matrix.shape != (rows, columns):
        raise ContractError(f"{name} must have shape {(rows, columns)}, found {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ContractError(f"{name} contains NaN or infinity")
    return matrix


def load_training_data(split: SplitInput) -> TrainingData:
    """Cross-check the train H5AD against its compact training reference."""
    import anndata as ad

    adata = ad.read_h5ad(split.train_h5ad, backed="r")
    try:
        raw_contract = adata.uns.get(split.contract_uns_key)
        if not isinstance(raw_contract, Mapping):
            raise ContractError(f"train H5AD lacks uns[{split.contract_uns_key!r}]")
        contract = plain(raw_contract)
        if str(contract.get("dataset_id", "")) != split.dataset_id:
            raise ContractError("H5AD/root-manifest dataset_id mismatch")
        if str(contract.get("split", "")) != split.split_id:
            raise ContractError("H5AD/root-manifest split mismatch")
        expected_role = "train" if split.regime == "loto" else "train_and_truth"
        if str(contract.get("role", "")) != expected_role:
            raise ContractError(
                f"H5AD role must be {expected_role!r}, found {contract.get('role')!r}"
            )
        if int(contract.get("prediction_n", 0)) != split.prediction_n:
            raise ContractError("H5AD prediction_n differs from train contract")
        if contract.get("truth_cell_count_must_not_control_prediction_n") is not True:
            raise ContractError("H5AD must prohibit truth-controlled prediction count")
        if contract.get("transductive_frozen_representation") is not True:
            raise ContractError("H5AD must declare a frozen shared representation")
        if contract.get("representation_refit_per_fold") is not False:
            raise ContractError("H5AD representation_refit_per_fold must be false")
        if bool(contract.get("target_removed")) != (split.regime == "loto"):
            raise ContractError("H5AD target_removed disagrees with regime")
        contract_holdout = _normalise_optional_time(
            contract.get("held_out_benchmark_time")
        )
        if split.regime == "loto":
            if contract_holdout is None or not same_time(contract_holdout, split.holdout_time):
                raise ContractError("H5AD held-out time disagrees with root manifest")
        elif contract_holdout is not None:
            raise ContractError("full_data H5AD declares a held-out target")

        state_key = str(contract.get("state_key", ""))
        spatial_key = str(contract.get("spatial_key", ""))
        time_key = str(contract.get("time_key", ""))
        row_id_key = str(contract.get("row_id_key", ""))
        if not all((state_key, spatial_key, time_key, row_id_key)):
            raise ContractError("H5AD contract lacks state/spatial/time/row_id keys")
        if state_key not in adata.obsm or spatial_key not in adata.obsm:
            raise ContractError("H5AD lacks contracted state/spatial matrices")
        if time_key not in adata.obs or row_id_key not in adata.obs:
            raise ContractError("H5AD lacks contracted time/row_id columns")
        rows = int(adata.n_obs)
        state_dim = int(contract.get("state_dim", adata.obsm[state_key].shape[1]))
        spatial_dim = int(contract.get("spatial_dim", adata.obsm[spatial_key].shape[1]))
        state = _validated_matrix(
            f"obsm[{state_key!r}]", adata.obsm[state_key], rows, state_dim
        )
        spatial = _validated_matrix(
            f"obsm[{spatial_key!r}]", adata.obsm[spatial_key], rows, spatial_dim
        )
        time = np.asarray(adata.obs[time_key], dtype=np.float64)
        row_id = adata.obs[row_id_key].astype(str).to_numpy(dtype=str)
        benchmark_original_obs_name = (
            adata.obs["benchmark_original_obs_name"].astype(str).to_numpy(dtype=str)
            if "benchmark_original_obs_name" in adata.obs
            else None
        )
        raw_interaction_graph = adata.uns.get("interaction_graph", {})
        interaction_graph = (
            plain(raw_interaction_graph)
            if isinstance(raw_interaction_graph, Mapping)
            else {}
        )
    finally:
        if adata.file is not None:
            adata.file.close()

    if time.shape != (rows,) or not np.isfinite(time).all():
        raise ContractError("training time vector is invalid")
    if row_id.shape != (rows,) or len(set(row_id)) != rows:
        raise ContractError("training row_id values must be unique")
    if benchmark_original_obs_name is not None and (
        benchmark_original_obs_name.shape != (rows,)
        or len(set(benchmark_original_obs_name)) != rows
    ):
        raise ContractError(
            "training benchmark_original_obs_name values must be unique and row-aligned"
        )
    actual_times = tuple(sorted({canonical_time(value) for value in time}))
    if len(actual_times) != len(split.observed_times) or any(
        not same_time(left, right)
        for left, right in zip(actual_times, split.observed_times)
    ):
        raise ContractError(
            f"H5AD times {actual_times} differ from manifest {split.observed_times}"
        )
    if split.regime == "loto" and np.any(
        np.isclose(time, split.holdout_time, rtol=0.0, atol=1e-8)
    ):
        raise ContractError("held-out rows physically occur in LOTO train H5AD")

    with np.load(split.training_reference_npz, allow_pickle=False) as archive:
        required = {"time", "spatial", "state", "row_id"}
        missing = sorted(required - set(archive.files))
        if missing:
            raise ContractError(f"training reference lacks keys {missing}")
        reference = {
            "time": np.asarray(archive["time"], dtype=np.float64),
            "spatial": np.asarray(archive["spatial"], dtype=np.float32),
            "state": np.asarray(archive["state"], dtype=np.float32),
            "row_id": np.asarray(archive["row_id"]).astype(str),
        }
    actual = {"time": time, "spatial": spatial, "state": state, "row_id": row_id}
    for label in ("time", "spatial", "state", "row_id"):
        left, right = reference[label], actual[label]
        if left.shape != right.shape:
            raise ContractError(f"training reference {label} shape differs from H5AD")
        equal = np.array_equal(left, right) if label == "row_id" else np.allclose(
            left, right, rtol=1e-6, atol=1e-6
        )
        if not equal:
            raise ContractError(f"training reference {label} differs from H5AD")
    return TrainingData(
        time=time,
        spatial=spatial,
        state=state,
        row_id=row_id,
        state_key=state_key,
        spatial_key=spatial_key,
        time_key=time_key,
        row_id_key=row_id_key,
        benchmark_original_obs_name=benchmark_original_obs_name,
        interaction_graph=interaction_graph,
        h5ad_contract=contract,
    )


def source_time(split: SplitInput) -> float:
    if split.regime == "full_data":
        return min(split.observed_times)
    assert split.holdout_time is not None
    return max(value for value in split.observed_times if value < split.holdout_time)


def bootstrap_indices(
    data: TrainingData, source: float, prediction_n: int = PREDICTION_N, seed: int = SEED
) -> np.ndarray:
    if prediction_n != PREDICTION_N:
        raise ContractError(f"prediction_n must remain fixed at {PREDICTION_N}")
    candidates = np.flatnonzero(np.isclose(data.time, source, rtol=0.0, atol=1e-8))
    if candidates.size == 0:
        raise ContractError(f"source time t{source:g} is absent")
    rng = np.random.default_rng(int(seed))
    return np.asarray(
        rng.choice(candidates, size=PREDICTION_N, replace=candidates.size < PREDICTION_N),
        dtype=np.int64,
    )


def load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ContractError(f"training config must be a YAML mapping: {path}")
    return dict(payload)


def _require_close(value: Any, expected: float, label: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ContractError(f"{label} is missing or non-numeric")
    try:
        observed = float(value)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{label} is missing or non-numeric") from exc
    if not np.isclose(observed, expected, rtol=0.0, atol=1e-12):
        raise ContractError(f"{label} must be {expected}, found {observed}")
    return observed


def _require_keys(
    value: Mapping[str, Any],
    expected: set[str],
    label: str,
    *,
    optional: set[str] | frozenset[str] = frozenset(),
) -> None:
    observed = set(value)
    missing = expected - observed
    extra = observed - expected - set(optional)
    if missing:
        raise ContractError(f"{label} lacks required fields {sorted(missing)}")
    if extra:
        rendered = sorted(repr(item) for item in extra)
        raise ContractError(f"{label} contains unsupported fields {rendered}")


def _require_exact(value: Any, expected: Any, label: str) -> None:
    if isinstance(expected, bool):
        matches = value is expected
    elif isinstance(expected, int):
        matches = isinstance(value, (int, np.integer)) and not isinstance(
            value, (bool, np.bool_)
        ) and int(value) == expected
    elif isinstance(expected, float):
        _require_close(value, expected, label)
        return
    elif isinstance(expected, tuple):
        matches = (
            isinstance(value, Sequence)
            and not isinstance(value, (str, bytes))
            and tuple(value) == expected
        )
    else:
        matches = value == expected
    if not matches:
        raise ContractError(f"{label} must be {expected!r}, found {value!r}")


def _validate_exact_mapping(
    value: Mapping[str, Any], expected: Mapping[str, Any], label: str
) -> None:
    _require_keys(value, set(expected), label)
    for key, expected_value in expected.items():
        _require_exact(value[key], expected_value, f"{label}.{key}")


def validate_training_config(
    config: Mapping[str, Any],
    *,
    runtime_resolved: bool = False,
    runtime_sigma: float = SIGMA,
) -> dict[str, Any]:
    """Enforce the complete published six-stage alpha_express=.015 profile.

    ``runtime_resolved=False`` is for the immutable canonical input YAML and
    requires every field and value exactly as published.  Saved models and the
    adapter's post-graph LOTO config use ``runtime_resolved=True``; that mode
    permits only the values the adapter explicitly resolves at runtime:
    checkpoint directory, spatial dimension, interaction cutoff/classifier,
    and the deterministic CLI sigma injected into every stage.
    """
    if not isinstance(config, Mapping):
        raise ContractError("training config must be a mapping")
    _require_keys(
        config,
        {"model", "ckpt_dir", "reverse", "seed", "training"},
        "config",
    )
    _require_exact(config["seed"], SEED, "config.seed")
    _require_exact(config["reverse"], True, "config.reverse")
    if runtime_resolved:
        _require_close(runtime_sigma, SIGMA, "runtime CLI sigma")
        if not isinstance(config["ckpt_dir"], str) or not config["ckpt_dir"].strip():
            raise ContractError("runtime-resolved config.ckpt_dir must be a non-empty string")
    else:
        _require_exact(config["ckpt_dir"], CANONICAL_CKPT_DIR, "config.ckpt_dir")

    training = config.get("training")
    if not isinstance(training, Mapping):
        raise ContractError("config lacks training mapping")
    _require_keys(training, {"defaults", "plan"}, "config.training")
    defaults = training.get("defaults")
    plan = training.get("plan")
    if (
        not isinstance(defaults, Mapping)
        or not isinstance(plan, Sequence)
        or isinstance(plan, (str, bytes))
    ):
        raise ContractError("config requires training.defaults and training.plan")
    _validate_exact_mapping(defaults, DEFAULT_PROFILE, "config.training.defaults")

    model = config.get("model")
    if not isinstance(model, Mapping):
        raise ContractError("config lacks model mapping")
    expected_model_fields = set(MODEL_PROFILE)
    if runtime_resolved:
        expected_model_fields.update(RUNTIME_MODEL_FIELDS)
    _require_keys(model, expected_model_fields, "config.model")
    _require_exact(model["components"], MODEL_PROFILE["components"], "config.model.components")
    _require_exact(
        model["interaction_type"],
        MODEL_PROFILE["interaction_type"],
        "config.model.interaction_type",
    )
    _require_exact(
        model["interaction_group_size"],
        MODEL_PROFILE["interaction_group_size"],
        "config.model.interaction_group_size",
    )
    for network in ("velocity_net", "growth_net", "score_net"):
        observed_network = model.get(network)
        if not isinstance(observed_network, Mapping):
            raise ContractError(f"config.model.{network} must be a mapping")
        _validate_exact_mapping(
            observed_network,
            MODEL_PROFILE[network],
            f"config.model.{network}",
        )
    interaction = model.get("interaction_net")
    if not isinstance(interaction, Mapping):
        raise ContractError("config.model.interaction_net must be a mapping")
    expected_interaction = MODEL_PROFILE["interaction_net"]
    _require_keys(interaction, set(expected_interaction), "config.model.interaction_net")
    for key, expected in expected_interaction.items():
        label = f"config.model.interaction_net.{key}"
        if runtime_resolved and key in RUNTIME_INTERACTION_FIELDS:
            if key == "cutoff":
                try:
                    cutoff = float(interaction[key])
                except (TypeError, ValueError) as exc:
                    raise ContractError(f"{label} must be finite and positive") from exc
                if not np.isfinite(cutoff) or cutoff <= 0:
                    raise ContractError(f"{label} must be finite and positive")
            elif key == "edge_predictor_path":
                if not isinstance(interaction[key], str) or not interaction[key].strip():
                    raise ContractError(f"{label} must be a non-empty string")
            else:
                try:
                    threshold = float(interaction[key])
                except (TypeError, ValueError) as exc:
                    raise ContractError(f"{label} must lie strictly between zero and one") from exc
                if not np.isfinite(threshold) or not 0 < threshold < 1:
                    raise ContractError(f"{label} must lie strictly between zero and one")
        else:
            _require_exact(interaction[key], expected, label)
    if runtime_resolved:
        _require_exact(model["spatial_dim"], 2, "config.model.spatial_dim")

    if len(plan) != len(STAGE_PROFILE):
        raise ContractError(f"training plan must contain exactly {len(STAGE_PROFILE)} stages")

    observed_profile: list[dict[str, Any]] = []
    for index, (stage, expected) in enumerate(zip(plan, STAGE_PROFILE)):
        if not isinstance(stage, Mapping):
            raise ContractError(f"training stage {index} is not a mapping")
        optional = (
            {"sigma"}
            if runtime_resolved and "sigma" not in expected
            else frozenset()
        )
        label = f"config.training.plan[{index}] ({expected['name']})"
        _require_keys(stage, set(expected), label, optional=optional)
        for key, expected_value in expected.items():
            _require_exact(stage[key], expected_value, f"{label}.{key}")
        if "sigma" in optional and "sigma" in stage:
            _require_close(stage["sigma"], runtime_sigma, f"{label}.sigma")
        observed_profile.append(plain(stage))

    runtime_fields: dict[str, Any] | None = None
    if runtime_resolved:
        runtime_fields = {
            "ckpt_dir": config["ckpt_dir"],
            "model.spatial_dim": model.get("spatial_dim"),
            "model.interaction_net.cutoff": interaction["cutoff"],
            "model.interaction_net.edge_predictor_path": interaction[
                "edge_predictor_path"
            ],
            "model.interaction_net.edge_predictor_thre": interaction[
                "edge_predictor_thre"
            ],
            "stage_sigma": runtime_sigma,
        }
    return {
        "seed": SEED,
        "reverse": True,
        "alpha_express": ALPHA_EXPRESS,
        "alpha_spatial": ALPHA_SPATIAL,
        "sigma": SIGMA,
        "components": list(MODEL_PROFILE["components"]),
        "defaults_profile": plain(defaults),
        "model_profile": plain(model),
        "stage_profile": observed_profile,
        "runtime_resolved": runtime_resolved,
        "runtime_resolved_fields": runtime_fields,
    }


def checkpoint_inventory(model_dir: Path) -> dict[str, Any]:
    """Validate all six stage outputs and return immutable hashes."""
    model_dir = Path(model_dir).expanduser().resolve()
    config_path = model_dir / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    config = load_yaml(config_path)
    profile = validate_training_config(config, runtime_resolved=True)
    plan = config["training"]["plan"]
    checkpoints: dict[str, dict[str, Any]] = {}
    for stage in plan:
        name = str(stage["name"])
        if str(stage["mode"]) == "score_matching":
            filename = "score_model.pth"
        else:
            filename = (
                "last_model.pth"
                if str(stage.get("save_strategy", "best")).lower() == "last"
                else "best_model.pth"
            )
        path = model_dir / name / filename
        if not path.is_file() or path.stat().st_size <= 0:
            raise ContractError(f"incomplete six-stage checkpoint: missing {path}")
        checkpoints[name] = {
            "path": str(path),
            "filename": filename,
            "sha256": sha256_file(path),
            "size_bytes": int(path.stat().st_size),
        }
    return {
        "model_dir": str(model_dir),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "training_profile": profile,
        "stage_complete": True,
        "stage_count": len(checkpoints),
        "checkpoints": checkpoints,
    }


def _checkpoint_runtime_binding(
    model_dir: Path,
    data: TrainingData,
    inventory: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind a saved runtime config and external edge model to audited values."""
    profile = inventory.get("training_profile")
    fields = (
        profile.get("runtime_resolved_fields")
        if isinstance(profile, Mapping)
        else None
    )
    if not isinstance(fields, Mapping):
        raise ContractError("checkpoint inventory lacks resolved runtime fields")
    configured_dir = Path(str(fields.get("ckpt_dir", ""))).expanduser().resolve()
    if configured_dir != model_dir:
        raise ContractError("saved config.ckpt_dir does not resolve to the model directory")
    _require_exact(
        fields.get("model.spatial_dim"),
        data.spatial_dim,
        "saved config.model.spatial_dim",
    )
    cutoff = _require_close(
        fields.get("model.interaction_net.cutoff"),
        float(expected["interaction_cutoff"]),
        "saved interaction cutoff",
    )
    threshold = _require_close(
        fields.get("model.interaction_net.edge_predictor_thre"),
        float(expected["edge_threshold"]),
        "saved edge predictor threshold",
    )
    raw_edge_path = fields.get("model.interaction_net.edge_predictor_path")
    if not isinstance(raw_edge_path, str) or not raw_edge_path.strip():
        raise ContractError("saved edge predictor path must be a non-empty string")
    configured_edge = Path(raw_edge_path).expanduser()
    if not configured_edge.is_absolute():
        raise ContractError("saved edge predictor path must be absolute")
    configured_edge = configured_edge.resolve()
    expected_edge = Path(str(expected["edge_model"])).expanduser().resolve()
    if configured_edge != expected_edge:
        raise ContractError("saved edge predictor path differs from audited provenance")
    if not configured_edge.is_file():
        raise ContractError("saved edge predictor artifact is missing")
    observed_edge_sha = sha256_file(configured_edge)
    expected_edge_sha = str(expected.get("edge_model_sha256", "")).lower()
    if observed_edge_sha != expected_edge_sha:
        raise ContractError("saved edge predictor artifact hash differs from provenance")
    return {
        "ckpt_dir": str(configured_dir),
        "spatial_dim": data.spatial_dim,
        "interaction_cutoff": cutoff,
        "edge_threshold": threshold,
        "edge_model": str(configured_edge),
        "edge_model_sha256": observed_edge_sha,
    }


def _legacy_full_runtime_expectation(data: TrainingData) -> dict[str, Any]:
    graph = data.interaction_graph
    required = {
        "neighborhood_threshold",
        "edge_predictor_threshold",
        "edge_predictor_path",
    }
    missing = required - set(graph)
    if missing:
        raise ContractError(
            "full-data input lacks frozen interaction provenance fields "
            f"{sorted(missing)}"
        )
    return {
        "interaction_cutoff": graph["neighborhood_threshold"],
        "edge_threshold": graph["edge_predictor_threshold"],
        "edge_model": graph["edge_predictor_path"],
        "edge_model_sha256": LEGACY_FULL_EDGE_MODEL_SHA256,
    }


def checkpoint_training_match(
    model_dir: Path,
    split: SplitInput,
    data: TrainingData,
    inventory: Mapping[str, Any] | None = None,
    runtime_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Prove that a reusable checkpoint was fitted to this training reference.

    New adapter fits have a benchmark_fit_summary.json linkage.  Older locked
    full-data fits are accepted only when their saved adata.h5ad has the exact
    frozen state, spatial, time, row identity, and row order of the training
    reference.
    """
    model_dir = Path(model_dir).expanduser().resolve()
    fit_summary = model_dir / "benchmark_fit_summary.json"
    if fit_summary.is_file():
        payload = json.loads(fit_summary.read_text(encoding="utf-8"))
        if str(payload.get("status")) != "complete":
            raise ContractError("model fit summary status is not complete")
        declared = str(payload.get("training_reference_sha256", "")).lower()
        manifest = str(payload.get("input_manifest_sha256", "")).lower()
        if declared != split.training_reference_sha256:
            raise ContractError("model fit summary training-reference SHA does not match")
        if manifest != split.root_manifest_sha256:
            raise ContractError("model fit summary input-manifest SHA does not match")
        if str(payload.get("split_id")) != split.split_id:
            raise ContractError("model fit summary split_id does not match")
        if str(payload.get("regime")) != split.regime:
            raise ContractError("model fit summary regime does not match")
        current = checkpoint_inventory(model_dir) if inventory is None else inventory
        declared_config = str(
            payload.get("saved_config_sha256", payload.get("config_sha256", ""))
        ).lower()
        if declared_config != str(current.get("config_sha256", "")).lower():
            raise ContractError("model config changed after benchmark fit")
        declared_checkpoints = payload.get("checkpoint_sha256")
        if not isinstance(declared_checkpoints, Mapping):
            raise ContractError("model fit summary lacks checkpoint hashes")
        current_checkpoints = {
            str(name): str(record["sha256"])
            for name, record in current["checkpoints"].items()
        }
        if {
            str(name): str(digest) for name, digest in declared_checkpoints.items()
        } != current_checkpoints:
            raise ContractError("model stage checkpoint hashes changed after benchmark fit")
        if split.regime == "full_data":
            expected_runtime = _legacy_full_runtime_expectation(data)
        else:
            prepare_summary = Path(
                str(payload.get("prepare_graph_summary", ""))
            ).expanduser().resolve()
            expected_prepare_sha = str(
                payload.get("prepare_graph_summary_sha256", "")
            ).lower()
            if (
                not prepare_summary.is_file()
                or sha256_file(prepare_summary) != expected_prepare_sha
            ):
                raise ContractError("prepared LOTO graph summary changed or disappeared")
            expected_runtime = {
                "interaction_cutoff": payload.get("interaction_cutoff"),
                "edge_threshold": payload.get("edge_threshold"),
                "edge_model": payload.get("edge_model"),
                "edge_model_sha256": payload.get("edge_model_sha256"),
            }
        runtime_report = _checkpoint_runtime_binding(
            model_dir, data, current, expected_runtime
        )
        original_match = payload.get("training_reference_match")
        if not isinstance(original_match, Mapping) or original_match.get("proof") != (
            "saved_adata_exact_frozen_arrays"
        ):
            raise ContractError(
                "model fit summary lacks a direct saved-adata training-reference proof"
            )
        row_proof = str(original_match.get("row_identity_proof", ""))
        if row_proof == "contracted_row_id_exact_order":
            expected_row_identity = data.row_id
        elif row_proof == "legacy_obs_names_vs_benchmark_original_obs_name":
            if data.benchmark_original_obs_name is None:
                raise ContractError(
                    "model fit summary declares a legacy row proof but the benchmark "
                    "input lacks benchmark_original_obs_name"
                )
            expected_row_identity = data.benchmark_original_obs_name
        else:
            raise ContractError("model fit summary lacks a supported row-identity proof")
        array_sha256 = original_match.get("array_sha256")
        declared_row_sha = (
            str(array_sha256.get("row_identity", "")).lower()
            if isinstance(array_sha256, Mapping)
            else ""
        )
        expected_row_sha = sha256_array(expected_row_identity.astype("U"))
        if declared_row_sha != expected_row_sha:
            raise ContractError(
                "model fit summary row-identity hash differs from the training reference"
            )
        saved = model_dir / "adata.h5ad"
        expected_saved_sha = str(original_match.get("sha256", "")).lower()
        if not saved.is_file() or sha256_file(saved) != expected_saved_sha:
            raise ContractError("model saved adata changed after training-reference proof")
        return {
            "proof": "benchmark_fit_summary",
            "path": str(fit_summary),
            "sha256": sha256_file(fit_summary),
            "training_reference_sha256": declared,
            "runtime_binding": runtime_report,
        }

    runtime_report: dict[str, Any] | None = None
    if inventory is not None:
        if runtime_binding is not None:
            expected_runtime = runtime_binding
        elif split.regime == "full_data":
            expected_runtime = _legacy_full_runtime_expectation(data)
        else:
            raise ContractError(
                "new LOTO checkpoint validation requires prepared graph runtime binding"
            )
        runtime_report = _checkpoint_runtime_binding(
            model_dir, data, inventory, expected_runtime
        )

    import anndata as ad

    saved = model_dir / "adata.h5ad"
    if not saved.is_file():
        raise ContractError(
            "checkpoint has neither benchmark_fit_summary.json nor saved adata.h5ad "
            "for training-reference verification"
        )
    adata = ad.read_h5ad(saved, backed="r")
    try:
        state_candidates = (
            data.state_key,
            str(data.h5ad_contract.get("source_state_key", "")),
            "X_latent",
        )
        spatial_candidates = (
            data.spatial_key,
            str(data.h5ad_contract.get("source_spatial_key", "")),
            "spatial_aligned",
        )
        time_candidates = (data.time_key, "benchmark_time", "time_point_processed")
        state_key = next((key for key in state_candidates if key and key in adata.obsm), None)
        spatial_key = next((key for key in spatial_candidates if key and key in adata.obsm), None)
        time_key = next((key for key in time_candidates if key and key in adata.obs), None)
        if state_key is None or spatial_key is None or time_key is None:
            raise ContractError("saved checkpoint adata lacks comparable state/spatial/time keys")
        observed_state = np.asarray(adata.obsm[state_key], dtype=np.float32)
        observed_spatial = np.asarray(adata.obsm[spatial_key], dtype=np.float32)
        observed_time = np.asarray(adata.obs[time_key], dtype=np.float64)
        if data.row_id_key in adata.obs:
            observed_row_identity = (
                adata.obs[data.row_id_key].astype(str).to_numpy(dtype=str)
            )
            expected_row_identity = data.row_id
            row_identity_proof = "contracted_row_id_exact_order"
            row_identity_key = f"obs/{data.row_id_key}"
        elif data.benchmark_original_obs_name is not None:
            observed_row_identity = np.asarray(adata.obs_names.astype(str), dtype=str)
            expected_row_identity = data.benchmark_original_obs_name
            row_identity_proof = "legacy_obs_names_vs_benchmark_original_obs_name"
            row_identity_key = "obs_names"
        else:
            raise ContractError(
                "saved checkpoint adata lacks contracted row_id and benchmark input "
                "lacks benchmark_original_obs_name for legacy row-identity proof"
            )
    finally:
        if adata.file is not None:
            adata.file.close()
    checks = {
        "state": (observed_state, data.state),
        "spatial": (observed_spatial, data.spatial),
        "time": (observed_time, data.time),
    }
    for label, (observed, expected) in checks.items():
        if observed.shape != expected.shape or not np.array_equal(observed, expected):
            raise ContractError(f"saved checkpoint adata {label} differs from training reference")
    if observed_row_identity.shape != expected_row_identity.shape or not np.array_equal(
        observed_row_identity, expected_row_identity
    ):
        raise ContractError(
            "saved checkpoint adata row identity/order differs from training reference"
        )
    report = {
        "proof": "saved_adata_exact_frozen_arrays",
        "path": str(saved),
        "sha256": sha256_file(saved),
        "state_key": state_key,
        "spatial_key": spatial_key,
        "time_key": time_key,
        "row_identity_proof": row_identity_proof,
        "row_identity_key": row_identity_key,
        "array_sha256": {
            "state": sha256_array(observed_state),
            "spatial": sha256_array(observed_spatial),
            "time": sha256_array(observed_time),
            "row_identity": sha256_array(observed_row_identity.astype("U")),
        },
    }
    if runtime_report is not None:
        report["runtime_binding"] = runtime_report
    return report


def repo_identity(repo: Path) -> dict[str, Any]:
    repo = Path(repo).expanduser().resolve()
    head = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    commit = head.stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain"],
        check=False,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "repo": str(repo),
        "git_commit": commit if len(commit) == 40 else None,
        "git_dirty": bool(status),
    }


def environment_provenance(device: str) -> dict[str, Any]:
    versions: dict[str, str | None] = {}
    for package in ("numpy", "anndata", "scanpy", "torch", "torchdiffeq"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    gpu: dict[str, Any] = {
        "requested_device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    try:
        import torch

        gpu["cuda_available"] = bool(torch.cuda.is_available())
        if str(device).startswith("cuda") and torch.cuda.is_available():
            index = torch.device(device).index
            index = torch.cuda.current_device() if index is None else int(index)
            props = torch.cuda.get_device_properties(index)
            gpu.update(
                {
                    "resolved_cuda_index": index,
                    "name": props.name,
                    "total_memory_bytes": int(props.total_memory),
                    "capability": list(torch.cuda.get_device_capability(index)),
                }
            )
    except Exception as exc:  # provenance must not mask a CPU preflight
        gpu["probe_error"] = repr(exc)
    return {
        "python_executable": os.path.realpath(sys.executable),
        "python_version": platform.python_version(),
        "dependency_versions": versions,
        "gpu": gpu,
    }


def input_provenance(split: SplitInput) -> dict[str, Any]:
    return {
        "input_manifest": str(split.root_manifest),
        "input_manifest_sha256": split.root_manifest_sha256,
        "train_h5ad": str(split.train_h5ad),
        "train_h5ad_sha256": split.train_h5ad_sha256,
        "training_reference_npz": str(split.training_reference_npz),
        "training_reference_sha256": split.training_reference_sha256,
        "source_roster_npz": str(split.source_roster_npz),
        "source_roster_sha256": split.source_roster_sha256,
        "truth_inputs_opened": False,
    }


def new_output_dir(path: Path) -> Path:
    path = Path(path).expanduser().resolve()
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(plain(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
