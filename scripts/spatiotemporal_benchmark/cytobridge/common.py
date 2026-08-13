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
BASE_COMPONENTS = ("velocity", "growth", "score")
COMPONENTS = (*BASE_COMPONENTS, "interaction")
STAGE_CONTRACT = (
    ("Pretrain", "neural_ode", "v+g"),
    ("Refine", "neural_ode", "v+g"),
    ("Init_interaction", "neural_ode", "v+g+i"),
    ("Train_Score", "score_matching", "s"),
    ("Finetune", "neural_ode", "v+g+i"),
    ("Score_Refine", "score_matching", "s"),
)
NO_INTERACTION_STAGE_CONTRACT = (
    ("Pretrain", "neural_ode", "v+g"),
    ("Refine", "neural_ode", "v+g"),
    ("Matched_stage_3_no_interaction", "neural_ode", "v+g"),
    ("Train_Score", "score_matching", "s"),
    ("Finetune_no_interaction", "neural_ode", "v+g"),
    ("Score_Refine", "score_matching", "s"),
)

# These values are resolved independently for each data set or LOTO fold.  They
# are excluded only when comparing a resolved fit with its package source YAML;
# all remaining scientific settings must match that source exactly.
RUNTIME_CONFIG_PATHS = {
    ("ckpt_dir",),
    ("spatial_dim",),
    ("model", "spatial_dim"),
    ("model", "interaction_net", "cutoff"),
    ("model", "interaction_net", "edge_predictor_path"),
    ("model", "interaction_net", "edge_predictor_thre"),
}


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
        raise ContractError(
            f"cannot read root manifest {root_manifest}: {exc}"
        ) from exc
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
            raise ContractError(
                "full_data split unexpectedly declares a held-out target"
            )
        if not observed:
            raise ContractError("full_data split has no observed times")
        initial = min(observed)
        for target in targets:
            if target <= initial or not any(
                same_time(target, value) for value in observed
            ):
                raise ContractError(
                    f"full-data target t{target:g} is not an observed post-t0 stage"
                )

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
        raise ContractError(
            f"{name} must have shape {(rows, columns)}, found {matrix.shape}"
        )
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
            if contract_holdout is None or not same_time(
                contract_holdout, split.holdout_time
            ):
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
        equal = (
            np.array_equal(left, right)
            if label == "row_id"
            else np.allclose(left, right, rtol=1e-6, atol=1e-6)
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
    data: TrainingData,
    source: float,
    prediction_n: int = PREDICTION_N,
    seed: int = SEED,
) -> np.ndarray:
    if prediction_n != PREDICTION_N:
        raise ContractError(f"prediction_n must remain fixed at {PREDICTION_N}")
    candidates = np.flatnonzero(np.isclose(data.time, source, rtol=0.0, atol=1e-8))
    if candidates.size == 0:
        raise ContractError(f"source time t{source:g} is absent")
    rng = np.random.default_rng(int(seed))
    return np.asarray(
        rng.choice(
            candidates, size=PREDICTION_N, replace=candidates.size < PREDICTION_N
        ),
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


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{label} must be a mapping")
    return value


def _require_positive_int(value: Any, label: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ContractError(f"{label} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{label} must be a positive integer") from exc
    if result <= 0 or result != value:
        raise ContractError(f"{label} must be a positive integer")
    return result


_OMIT_CONFIG_VALUE = object()


def _scientific_config(value: Any, path: tuple[str, ...] = ()) -> Any:
    """Drop only values resolved from the current data set or output folder."""
    if path in RUNTIME_CONFIG_PATHS or (
        len(path) == 4 and path[:2] == ("training", "plan") and path[-1] == "sigma"
    ):
        return _OMIT_CONFIG_VALUE
    if isinstance(value, Mapping):
        result = {}
        for key, child in value.items():
            projected = _scientific_config(child, (*path, str(key)))
            if projected is not _OMIT_CONFIG_VALUE:
                result[str(key)] = projected
        return result
    if isinstance(value, (list, tuple)):
        return [
            _scientific_config(child, (*path, str(index)))
            for index, child in enumerate(value)
        ]
    return plain(value)


def validate_training_config(
    config: Mapping[str, Any],
    *,
    runtime_resolved: bool = False,
    runtime_sigma: float = SIGMA,
    reference: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate the shared contract without imposing one data set's recipe.

    The package YAML (or a checkpoint's resolved ``config.yaml``) supplies the
    data-set-specific architecture, batch sizes, losses and schedule.  The
    benchmark fixes only the scientific settings shared by all four data sets:
    alpha=0.015, spatial weight 10, sigma 0.03, seed 42 and the six stage roles.
    When ``reference`` is supplied, every other scientific value must match it;
    only graph-derived values and output locations may differ.
    """
    if not isinstance(config, Mapping):
        raise ContractError("training config must be a mapping")
    for key in ("model", "ckpt_dir", "reverse", "seed", "training"):
        if key not in config:
            raise ContractError(f"config lacks required field {key!r}")
    if config["seed"] != SEED:
        raise ContractError(f"config.seed must be {SEED}, found {config['seed']!r}")
    if config["reverse"] is not True:
        raise ContractError("config.reverse must be true")
    if not isinstance(config["ckpt_dir"], str) or not config["ckpt_dir"].strip():
        raise ContractError("config.ckpt_dir must be a non-empty string")
    if runtime_resolved:
        _require_close(runtime_sigma, SIGMA, "runtime CLI sigma")

    training = _require_mapping(config["training"], "config.training")
    defaults = _require_mapping(training.get("defaults"), "config.training.defaults")
    plan = training.get("plan")
    if not isinstance(plan, Sequence) or isinstance(plan, (str, bytes)):
        raise ContractError("config.training.plan must be a sequence")
    _require_close(defaults.get("alpha_express"), ALPHA_EXPRESS, "alpha_express")
    _require_close(defaults.get("alpha_spatial"), ALPHA_SPATIAL, "alpha_spatial")
    _require_close(defaults.get("sigma"), SIGMA, "sigma")
    if defaults.get("global_mass") is not True:
        raise ContractError("config.training.defaults.global_mass must be true")

    model = _require_mapping(config["model"], "config.model")
    components = model.get("components")
    if not isinstance(components, Sequence) or isinstance(components, (str, bytes)):
        raise ContractError("config.model.components must be a sequence")
    components_tuple = tuple(str(component) for component in components)
    if components_tuple not in {COMPONENTS, BASE_COMPONENTS}:
        raise ContractError(
            "config.model.components must be either "
            f"{COMPONENTS!r} or {BASE_COMPONENTS!r}, found {components!r}"
        )
    for network_name in ("velocity_net", "growth_net", "score_net"):
        network = _require_mapping(
            model.get(network_name), f"config.model.{network_name}"
        )
        _require_positive_int(
            network.get("hidden_dim"), f"config.model.{network_name}.hidden_dim"
        )
        _require_positive_int(
            network.get("n_layers"), f"config.model.{network_name}.n_layers"
        )

    uses_interaction = components_tuple == COMPONENTS
    if uses_interaction:
        if model.get("interaction_type") != "gnn":
            raise ContractError("config.model.interaction_type must be 'gnn'")
        interaction_group_size = _require_positive_int(
            model.get("interaction_group_size"),
            "config.model.interaction_group_size",
        )
        interaction = _require_mapping(
            model.get("interaction_net"), "config.model.interaction_net"
        )
        cutoff = float(interaction.get("cutoff", np.nan))
        if not np.isfinite(cutoff) or cutoff <= 0:
            raise ContractError("config.model.interaction_net.cutoff must be positive")
        interaction_mode = (
            str(interaction.get("edge_prior_mode", "learned")).strip().lower()
        )
        if interaction_mode not in {"learned", "all_spatial"}:
            raise ContractError(
                "config.model.interaction_net.edge_prior_mode must be 'learned' or "
                f"'all_spatial', found {interaction_mode!r}"
            )
        if interaction_mode == "learned":
            try:
                threshold = float(interaction.get("edge_predictor_thre", np.nan))
            except (TypeError, ValueError) as exc:
                raise ContractError(
                    "config.model.interaction_net.edge_predictor_thre must lie "
                    "between zero and one when edge_prior_mode='learned'"
                ) from exc
            if not np.isfinite(threshold) or not 0 < threshold < 1:
                raise ContractError(
                    "config.model.interaction_net.edge_predictor_thre must lie "
                    "between zero and one when edge_prior_mode='learned'"
                )
            edge_path = interaction.get("edge_predictor_path")
            if not isinstance(edge_path, str) or not edge_path.strip():
                raise ContractError(
                    "config.model.interaction_net.edge_predictor_path must be a "
                    "non-empty string when edge_prior_mode='learned'"
                )
        else:
            inert_predictor_keys = sorted(
                key
                for key in (
                    "edge_predictor_path",
                    "edge_predictor_thre",
                    "edge_predictor_threshold",
                )
                if key in interaction
            )
            if inert_predictor_keys:
                raise ContractError(
                    "config.model.interaction_net edge_prior_mode='all_spatial' does "
                    "not use an edge predictor; remove inert predictor keys "
                    f"{inert_predictor_keys!r}"
                )
        stage_contract = STAGE_CONTRACT
    else:
        inert_interaction_keys = sorted(
            key
            for key in ("interaction_type", "interaction_group_size", "interaction_net")
            if key in model
        )
        if inert_interaction_keys:
            raise ContractError(
                "config.model.components excludes interaction; remove inert "
                f"interaction model fields {inert_interaction_keys!r}"
            )
        interaction_group_size = None
        interaction = {}
        cutoff = None
        interaction_mode = "none"
        stage_contract = NO_INTERACTION_STAGE_CONTRACT
    if runtime_resolved and int(model.get("spatial_dim", -1)) != 2:
        raise ContractError("runtime config.model.spatial_dim must be 2")

    if len(plan) != len(stage_contract):
        raise ContractError(
            f"training plan must contain exactly {len(stage_contract)} stages"
        )
    observed_profile: list[dict[str, Any]] = []
    for index, (stage, expected) in enumerate(zip(plan, stage_contract)):
        stage = _require_mapping(stage, f"config.training.plan[{index}]")
        expected_name, expected_mode, expected_strategy = expected
        for key, expected_value in (
            ("name", expected_name),
            ("mode", expected_mode),
            ("train_strategy", expected_strategy),
        ):
            if stage.get(key) != expected_value:
                raise ContractError(
                    f"config.training.plan[{index}].{key} must be "
                    f"{expected_value!r}, found {stage.get(key)!r}"
                )
        _require_positive_int(stage.get("epochs"), f"{expected_name}.epochs")
        _require_positive_int(stage.get("batch_size"), f"{expected_name}.batch_size")
        if "sigma" in stage:
            _require_close(stage["sigma"], runtime_sigma, f"{expected_name}.sigma")
        if interaction_mode == "none":
            if (
                expected_mode == "neural_ode"
                and stage.get("interaction_use") is not False
            ):
                raise ContractError(
                    f"{expected_name}.interaction_use must be false for a "
                    "no-interaction profile"
                )
            if expected_mode == "score_matching" and "interaction_use" in stage:
                raise ContractError(
                    f"{expected_name}.interaction_use is inert for a score-only stage"
                )
        observed_profile.append(plain(stage))

    scientific_profile = _scientific_config(config)
    if reference is not None:
        reference_profile = validate_training_config(reference)
        reference_model = _require_mapping(reference.get("model"), "reference.model")
        reference_interaction = reference_model.get("interaction_net", {})
        if interaction_mode == "all_spatial":
            reference_interaction = _require_mapping(
                reference_interaction,
                "reference.model.interaction_net",
            )
            if reference_profile["interaction_mode"] != "all_spatial":
                raise ContractError(
                    "resolved all_spatial training config differs in edge-prior mode "
                    "from its package YAML"
                )
            _require_close(
                cutoff,
                float(reference_interaction.get("cutoff", np.nan)),
                "resolved all_spatial interaction cutoff",
            )
        # Production checkpoints written before ``edge_prior_mode`` became an
        # explicit YAML field still have learned-prior semantics: ``learned``
        # was the GNN constructor default.  Normalize only that one historical
        # omission and only when the reference profile explicitly declares
        # learned.  An explicit actual mode, an all-spatial reference, or a
        # missing reference declaration must continue through the strict byte-
        # projected scientific comparison unchanged.
        if (
            interaction_mode != "none"
            and isinstance(reference_interaction, Mapping)
            and "edge_prior_mode" not in interaction
            and "edge_prior_mode" in reference_interaction
            and str(reference_interaction["edge_prior_mode"]).strip().lower()
            == "learned"
        ):
            scientific_profile["model"]["interaction_net"][
                "edge_prior_mode"
            ] = "learned"
        scientific_text = json.dumps(
            scientific_profile, sort_keys=True, separators=(",", ":")
        )
        reference_text = json.dumps(
            _scientific_config(reference), sort_keys=True, separators=(",", ":")
        )
        if scientific_text != reference_text:
            raise ContractError(
                "resolved training config changes scientific settings from its package YAML"
            )

    runtime_fields: dict[str, Any] | None = None
    if runtime_resolved:
        runtime_fields = {
            "ckpt_dir": config["ckpt_dir"],
            "model.spatial_dim": model.get("spatial_dim"),
            "interaction_mode": interaction_mode,
            "edge_prior_mode": interaction_mode,
            "stage_sigma": runtime_sigma,
        }
        if interaction_mode != "none":
            runtime_fields.update(
                {
                    "model.interaction_net.cutoff": interaction["cutoff"],
                    "model.interaction_net.edge_prior_mode": interaction_mode,
                }
            )
        if interaction_mode == "learned":
            runtime_fields.update(
                {
                    "model.interaction_net.edge_predictor_path": interaction[
                        "edge_predictor_path"
                    ],
                    "model.interaction_net.edge_predictor_thre": interaction[
                        "edge_predictor_thre"
                    ],
                }
            )
    return {
        "seed": SEED,
        "reverse": True,
        "alpha_express": ALPHA_EXPRESS,
        "alpha_spatial": ALPHA_SPATIAL,
        "sigma": SIGMA,
        "components": list(components_tuple),
        "interaction_mode": interaction_mode,
        "edge_prior_mode": interaction_mode,
        "uses_interaction": uses_interaction,
        "interaction_group_size": interaction_group_size,
        "expected_weight_stage": next(
            stage[0]
            for stage in reversed(stage_contract)
            if stage[1] != "score_matching" and stage[2] != "s"
        ),
        "expected_score_stage": next(
            stage[0]
            for stage in reversed(stage_contract)
            if stage[1] == "score_matching" or stage[2] == "s"
        ),
        "defaults_profile": plain(defaults),
        "model_profile": plain(model),
        "stage_profile": observed_profile,
        "scientific_profile": scientific_profile,
        "runtime_resolved": runtime_resolved,
        "runtime_resolved_fields": runtime_fields,
    }


def checkpoint_inventory(
    model_dir: Path, *, reference_config: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Validate all six stage outputs and return immutable hashes."""
    model_dir = Path(model_dir).expanduser().resolve()
    config_path = model_dir / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    config = load_yaml(config_path)
    profile = validate_training_config(
        config,
        runtime_resolved=True,
        reference=reference_config,
    )
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


def _runtime_interaction_mode(fields: Mapping[str, Any]) -> str:
    """Return the explicit interaction mode, preserving learned legacy defaults."""

    mode = fields.get(
        "interaction_mode",
        fields.get(
            "edge_prior_mode",
            fields.get("model.interaction_net.edge_prior_mode", "learned"),
        ),
    )
    result = str(mode).strip().lower()
    if result not in {"learned", "all_spatial", "none"}:
        raise ContractError("checkpoint inventory records an invalid interaction mode")
    return result


def _checkpoint_runtime_binding(
    model_dir: Path,
    data: TrainingData,
    inventory: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind a saved config to the graph values used for this benchmark fit.

    Current CytoBridge Finetune checkpoints carry the learned link-predictor
    parameters in their state dict.  The path recorded in ``config.yaml`` is
    useful provenance, but it is not a runtime dependency and may point to the
    machine on which the model was fitted.
    """
    profile = inventory.get("training_profile")
    fields = (
        profile.get("runtime_resolved_fields") if isinstance(profile, Mapping) else None
    )
    if not isinstance(fields, Mapping):
        raise ContractError("checkpoint inventory lacks resolved runtime fields")
    recorded_dir = str(fields.get("ckpt_dir", "")).strip()
    if not recorded_dir:
        raise ContractError("saved config.ckpt_dir must be non-empty")
    if int(fields.get("model.spatial_dim", -1)) != data.spatial_dim:
        raise ContractError(
            "saved config.model.spatial_dim differs from benchmark data"
        )
    interaction_mode = _runtime_interaction_mode(fields)
    expected_mode = (
        str(
            expected.get(
                "interaction_mode",
                expected.get("edge_prior_mode", interaction_mode),
            )
        )
        .strip()
        .lower()
    )
    if expected_mode != interaction_mode:
        raise ContractError("saved interaction mode differs from benchmark expectation")
    if interaction_mode == "none":
        inert_fields = sorted(
            key
            for key in (
                "model.interaction_net.cutoff",
                "model.interaction_net.edge_prior_mode",
                "model.interaction_net.edge_predictor_path",
                "model.interaction_net.edge_predictor_thre",
            )
            if key in fields
        )
        if inert_fields:
            raise ContractError(
                "no-interaction checkpoint inventory contains inert interaction "
                f"fields {inert_fields!r}"
            )
        return {
            "model_dir": str(model_dir),
            "recorded_ckpt_dir": recorded_dir,
            "spatial_dim": data.spatial_dim,
            "interaction_mode": "none",
            "edge_prior_mode": "none",
            "include_interaction": False,
        }
    cutoff = _require_close(
        fields.get("model.interaction_net.cutoff"),
        float(expected["interaction_cutoff"]),
        "saved interaction cutoff",
    )
    if interaction_mode == "all_spatial":
        inert_fields = sorted(
            key
            for key in (
                "model.interaction_net.edge_predictor_path",
                "model.interaction_net.edge_predictor_thre",
            )
            if key in fields
        )
        if inert_fields:
            raise ContractError(
                "all_spatial checkpoint inventory contains inert predictor fields "
                f"{inert_fields!r}"
            )
        return {
            "model_dir": str(model_dir),
            "recorded_ckpt_dir": recorded_dir,
            "spatial_dim": data.spatial_dim,
            "interaction_cutoff": cutoff,
            "interaction_mode": interaction_mode,
            "edge_prior_mode": interaction_mode,
            "include_interaction": True,
            "edge_threshold": None,
            "recorded_edge_predictor_path": None,
            "edge_predictor_source": None,
            "external_edge_predictor_required": False,
        }
    threshold = _require_close(
        fields.get("model.interaction_net.edge_predictor_thre"),
        float(expected["edge_threshold"]),
        "saved edge predictor threshold",
    )
    recorded_edge_path = fields.get("model.interaction_net.edge_predictor_path")
    return {
        "model_dir": str(model_dir),
        "recorded_ckpt_dir": recorded_dir,
        "spatial_dim": data.spatial_dim,
        "interaction_cutoff": cutoff,
        "interaction_mode": interaction_mode,
        "edge_prior_mode": interaction_mode,
        "include_interaction": True,
        "edge_threshold": threshold,
        "recorded_edge_predictor_path": str(recorded_edge_path),
        "edge_predictor_source": "embedded_finetune_checkpoint",
        "external_edge_predictor_required": False,
    }


def _full_runtime_expectation(
    data: TrainingData, inventory: Mapping[str, Any]
) -> dict[str, Any]:
    profile = inventory.get("training_profile")
    fields = (
        profile.get("runtime_resolved_fields") if isinstance(profile, Mapping) else None
    )
    if not isinstance(fields, Mapping):
        raise ContractError("checkpoint inventory lacks resolved runtime fields")
    interaction_mode = _runtime_interaction_mode(fields)
    if interaction_mode == "none":
        return {
            "interaction_mode": "none",
            "edge_prior_mode": "none",
        }
    graph = data.interaction_graph
    required = {"neighborhood_threshold"}
    if interaction_mode == "learned":
        required.add("edge_predictor_threshold")
    missing = required - set(graph)
    if missing and interaction_mode == "learned":
        raise ContractError(
            "full-data input lacks frozen interaction provenance fields "
            f"{sorted(missing)}"
        )
    cutoff = (
        fields.get("model.interaction_net.cutoff")
        if interaction_mode == "all_spatial"
        else graph.get("neighborhood_threshold")
    )
    result = {
        "interaction_cutoff": cutoff,
        "interaction_mode": interaction_mode,
        "edge_prior_mode": interaction_mode,
    }
    if interaction_mode == "learned":
        result["edge_threshold"] = graph["edge_predictor_threshold"]
    return result


def checkpoint_training_match(
    model_dir: Path,
    split: SplitInput,
    data: TrainingData,
    inventory: Mapping[str, Any] | None = None,
    runtime_binding: Mapping[str, Any] | None = None,
    reference_config_sha256: str | None = None,
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
            raise ContractError(
                "model fit summary training-reference SHA does not match"
            )
        if manifest != split.root_manifest_sha256:
            raise ContractError("model fit summary input-manifest SHA does not match")
        if str(payload.get("split_id")) != split.split_id:
            raise ContractError("model fit summary split_id does not match")
        if str(payload.get("regime")) != split.regime:
            raise ContractError("model fit summary regime does not match")
        if reference_config_sha256 is not None:
            expected_source_sha = _valid_digest(
                reference_config_sha256, "current training config"
            )
            declared_source_sha = _valid_digest(
                payload.get("training_config_source_sha256"),
                "model fit summary training config",
            )
            if declared_source_sha != expected_source_sha:
                raise ContractError(
                    "model fit summary training-config SHA differs from the current "
                    "training config"
                )
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
            raise ContractError(
                "model stage checkpoint hashes changed after benchmark fit"
            )
        if split.regime == "full_data":
            expected_runtime = _full_runtime_expectation(data, current)
        else:
            prepare_summary = (
                Path(str(payload.get("prepare_graph_summary", "")))
                .expanduser()
                .resolve()
            )
            expected_prepare_sha = str(
                payload.get("prepare_graph_summary_sha256", "")
            ).lower()
            if (
                not prepare_summary.is_file()
                or sha256_file(prepare_summary) != expected_prepare_sha
            ):
                raise ContractError("prepared LOTO mode summary changed or disappeared")
            profile = current.get("training_profile")
            fields = (
                profile.get("runtime_resolved_fields")
                if isinstance(profile, Mapping)
                else None
            )
            interaction_mode = (
                _runtime_interaction_mode(fields)
                if isinstance(fields, Mapping)
                else "learned"
            )
            expected_runtime = {
                "interaction_mode": interaction_mode,
                "edge_prior_mode": interaction_mode,
            }
            if interaction_mode != "none":
                expected_runtime["interaction_cutoff"] = payload.get(
                    "interaction_cutoff"
                )
            if interaction_mode == "learned":
                expected_runtime["edge_threshold"] = payload.get("edge_threshold")
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
            raise ContractError(
                "model fit summary lacks a supported row-identity proof"
            )
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
            raise ContractError(
                "model saved adata changed after training-reference proof"
            )
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
            expected_runtime = _full_runtime_expectation(data, inventory)
        else:
            raise ContractError(
                "new LOTO checkpoint validation requires prepared mode runtime binding"
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
        state_key = next(
            (key for key in state_candidates if key and key in adata.obsm), None
        )
        spatial_key = next(
            (key for key in spatial_candidates if key and key in adata.obsm), None
        )
        time_key = next(
            (key for key in time_candidates if key and key in adata.obs), None
        )
        if state_key is None or spatial_key is None or time_key is None:
            raise ContractError(
                "saved checkpoint adata lacks comparable state/spatial/time keys"
            )
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
            raise ContractError(
                f"saved checkpoint adata {label} differs from training reference"
            )
    if (
        observed_row_identity.shape != expected_row_identity.shape
        or not np.array_equal(observed_row_identity, expected_row_identity)
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
