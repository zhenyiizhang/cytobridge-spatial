"""Strict input contracts and provenance for dynamic benchmark adapters.

Only training artifacts are resolved here.  Truth artifacts may exist in the
root manifest, but this module deliberately never reads or resolves them.
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


METHODS = ("stvcr", "stories", "mioflow")
REGIMES = ("full_data", "loto")
CONTRACT_UNS_KEY = "cytobridge_benchmark_contract"
PREDICTION_N = 5000
DYNAMIC_ADAPTER_FILES = (
    "scripts/spatiotemporal_benchmark/dynamic/common.py",
    "scripts/spatiotemporal_benchmark/dynamic/run_dynamic.py",
)


class ContractError(ValueError):
    """The immutable benchmark input contract is missing or inconsistent."""


@dataclass(frozen=True)
class SplitContract:
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
    train_h5ad_declared_sha256: str
    training_reference_declared_sha256: str
    source_roster_declared_sha256: str
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


def canonical_time(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ContractError(f"time must be finite, found {value!r}")
    return result


def same_time(left: float, right: float) -> bool:
    return bool(np.isclose(left, right, rtol=0.0, atol=1e-8))


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def dynamic_adapter_implementation_identity() -> dict[str, Any]:
    """Return a portable byte-level identity for the in-repo dynamic adapter."""

    repository_root = Path(__file__).resolve().parents[3]
    files = {
        relative_path: sha256_file(repository_root / relative_path)
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


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def stable_seed(base_seed: int, *parts: object) -> int:
    payload = json.dumps(
        [int(base_seed), *map(str, parts)], separators=(",", ":")
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return [_plain(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _artifact_path(entry: Mapping[str, Any], manifest_dir: Path) -> Path:
    raw = entry.get("relative_path", entry.get("path"))
    if raw is None:
        raise ContractError("artifact is missing path")
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = (manifest_dir / path).resolve()
    return path


def _artifact_sha(entry: Mapping[str, Any], label: str) -> str:
    digest = str(entry.get("sha256", "")).lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ContractError(f"{label} is missing a valid SHA-256")
    return digest


def _times_from_counts(value: Any) -> tuple[float, ...]:
    if not isinstance(value, Mapping):
        raise ContractError("split train_time_counts must be a mapping")
    return tuple(
        sorted(canonical_time(key) for key, count in value.items() if int(count) > 0)
    )


def read_split_contract(root_manifest: Path, split_id: str) -> SplitContract:
    root_manifest = Path(root_manifest).expanduser().resolve()
    payload = json.loads(root_manifest.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ContractError("root input manifest must be a JSON object")
    splits = payload.get("splits")
    if not isinstance(splits, Mapping) or split_id not in splits:
        available = sorted(splits) if isinstance(splits, Mapping) else []
        raise ContractError(f"unknown split {split_id!r}; available={available}")
    entry = splits[split_id]
    if not isinstance(entry, Mapping):
        raise ContractError(f"split {split_id!r} must be an object")
    protocol = str(entry.get("protocol", ""))
    if split_id == "full_data" or protocol == "full_data":
        regime = "full_data"
    elif split_id.startswith("loto_t") or protocol == "leave_one_timepoint_out":
        regime = "loto"
    else:
        raise ContractError(f"cannot infer benchmark regime for split {split_id!r}")

    train = entry.get("train")
    if not isinstance(train, Mapping):
        raise ContractError(f"split {split_id!r} lacks train artifacts")
    h5ad_entry = train.get("h5ad")
    reference_entry = train.get("training_reference_npz")
    roster_entry = train.get("source_roster_npz")
    if (
        not isinstance(h5ad_entry, Mapping)
        or not isinstance(reference_entry, Mapping)
        or not isinstance(roster_entry, Mapping)
    ):
        raise ContractError(
            f"split {split_id!r} requires train H5AD, training reference, and canonical source roster"
        )
    manifest_dir = root_manifest.parent
    train_h5ad = _artifact_path(h5ad_entry, manifest_dir)
    training_reference = _artifact_path(reference_entry, manifest_dir)
    source_roster = _artifact_path(roster_entry, manifest_dir)
    for label, path in (
        ("train H5AD", train_h5ad),
        ("training reference NPZ", training_reference),
        ("source roster NPZ", source_roster),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} does not exist: {path}")

    observed_times = _times_from_counts(entry.get("train_time_counts"))
    targets_raw = entry.get("evaluation_targets")
    if not isinstance(targets_raw, Sequence) or isinstance(targets_raw, (str, bytes)):
        raise ContractError(f"split {split_id!r} requires evaluation_targets")
    targets = tuple(canonical_time(value) for value in targets_raw)
    if not targets or len(set(targets)) != len(targets):
        raise ContractError(f"split {split_id!r} evaluation_targets must be unique")
    holdout_raw = entry.get("held_out_benchmark_time")
    holdout = None if holdout_raw is None else canonical_time(holdout_raw)
    if regime == "loto":
        if holdout is None or len(targets) != 1 or not same_time(targets[0], holdout):
            raise ContractError("LOTO split must expose exactly its held-out target")
        if any(same_time(holdout, time) for time in observed_times):
            raise ContractError("LOTO held-out time occurs in train_time_counts")
        if not any(time < holdout for time in observed_times):
            raise ContractError("LOTO target has no previous observed source stage")
        if entry.get("target_rows_physically_removed_from_train") is not True:
            raise ContractError("LOTO manifest must declare physical target removal")
    else:
        if holdout is not None:
            raise ContractError("full_data split must not declare a held-out target")
        initial = min(observed_times)
        if any(target <= initial for target in targets):
            raise ContractError("full_data targets must occur after the initial stage")
        if any(
            not any(same_time(target, time) for time in observed_times)
            for target in targets
        ):
            raise ContractError("full_data targets must be observed training stages")

    prediction_n = int(entry.get("prediction_n", payload.get("prediction_n", 0)))
    if prediction_n != PREDICTION_N:
        raise ContractError(
            f"dynamic primary contract requires prediction_n={PREDICTION_N}, found {prediction_n}"
        )
    contract_key = str(
        entry.get(
            "contract_uns_key",
            payload.get("contract_uns_key", CONTRACT_UNS_KEY),
        )
    )
    if contract_key != CONTRACT_UNS_KEY:
        raise ContractError(
            f"contract_uns_key must be {CONTRACT_UNS_KEY!r}, found {contract_key!r}"
        )
    return SplitContract(
        dataset_id=str(
            payload.get(
                "dataset",
                payload.get(
                    "dataset_id",
                    entry.get("dataset", entry.get("dataset_id", "unknown")),
                ),
            )
        ),
        split_id=split_id,
        regime=regime,
        train_h5ad=train_h5ad,
        training_reference_npz=training_reference,
        source_roster_npz=source_roster,
        observed_times=observed_times,
        evaluation_targets=targets,
        holdout_time=holdout,
        prediction_n=prediction_n,
        contract_uns_key=contract_key,
        root_manifest=root_manifest,
        root_manifest_sha256=sha256_file(root_manifest),
        train_h5ad_declared_sha256=_artifact_sha(h5ad_entry, "train H5AD"),
        training_reference_declared_sha256=_artifact_sha(
            reference_entry, "training reference NPZ"
        ),
        source_roster_declared_sha256=_artifact_sha(roster_entry, "source roster NPZ"),
        raw=dict(entry),
    )


def _validated_matrix(name: str, value: Any, n_rows: int, dimension: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (n_rows, dimension):
        raise ContractError(
            f"{name} must have shape {(n_rows, dimension)}, found {array.shape}"
        )
    if not np.isfinite(array).all():
        raise ContractError(f"{name} contains NaN or infinity")
    return array


def _check_declared_hash(actual_path: Path, expected: str, label: str) -> str:
    actual = sha256_file(actual_path)
    if actual != expected:
        raise ContractError(f"{label} SHA-256 differs from root input manifest")
    return actual


def load_training_data(split: SplitContract) -> TrainingData:
    """Load and cross-check only train H5AD/NPZ artifacts."""
    import anndata as ad

    _check_declared_hash(
        split.train_h5ad, split.train_h5ad_declared_sha256, "train H5AD"
    )
    _check_declared_hash(
        split.training_reference_npz,
        split.training_reference_declared_sha256,
        "training reference NPZ",
    )
    _check_declared_hash(
        split.source_roster_npz,
        split.source_roster_declared_sha256,
        "source roster NPZ",
    )
    adata = ad.read_h5ad(split.train_h5ad, backed="r")
    try:
        raw_contract = adata.uns.get(split.contract_uns_key)
        if not isinstance(raw_contract, Mapping):
            raise ContractError(
                f"train H5AD lacks uns[{split.contract_uns_key!r}] contract"
            )
        contract = _plain(raw_contract)
        h5_dataset = str(contract.get("dataset", contract.get("dataset_id", "")))
        if h5_dataset != split.dataset_id:
            raise ContractError("H5AD/root-manifest dataset_id mismatch")
        if str(contract.get("split")) != split.split_id:
            raise ContractError("H5AD/root-manifest split mismatch")
        expected_role = "train" if split.regime == "loto" else "train_and_truth"
        if str(contract.get("role")) != expected_role:
            raise ContractError(
                f"H5AD role must be {expected_role!r}, found {contract.get('role')!r}"
            )
        if int(contract.get("prediction_n", 0)) != PREDICTION_N:
            raise ContractError("H5AD prediction_n differs from fixed primary contract")
        if (
            "truth_cell_count_must_not_control_prediction_n" in contract
            and contract.get("truth_cell_count_must_not_control_prediction_n")
            is not True
        ):
            raise ContractError("H5AD must prohibit truth-controlled prediction count")
        if contract.get("transductive_frozen_representation") is not True:
            raise ContractError("H5AD must declare a frozen shared representation")
        if (
            "representation_refit_per_fold" in contract
            and contract.get("representation_refit_per_fold") is not False
        ):
            raise ContractError(
                "per-method/per-fold representation refitting is forbidden"
            )
        if bool(contract.get("target_removed")) != (split.regime == "loto"):
            raise ContractError("H5AD target_removed flag disagrees with split regime")
        declared_holdout = contract.get("held_out_benchmark_time")
        if isinstance(declared_holdout, str) and declared_holdout.lower() in {
            "none",
            "null",
            "",
        }:
            declared_holdout = None
        if split.regime == "loto":
            if declared_holdout is None or not same_time(
                canonical_time(declared_holdout), float(split.holdout_time)
            ):
                raise ContractError("H5AD held-out target disagrees with root manifest")
        elif declared_holdout is not None:
            raise ContractError("full_data H5AD must not declare a held-out target")

        state_key = str(contract.get("state_key", ""))
        spatial_key = str(contract.get("spatial_key", ""))
        time_key = str(contract.get("time_key", ""))
        row_id_key = str(contract.get("row_id_key", ""))
        if not all((state_key, spatial_key, time_key, row_id_key)):
            raise ContractError("H5AD representation contract lacks required key names")
        if state_key not in adata.obsm or spatial_key not in adata.obsm:
            raise ContractError("H5AD lacks contracted state/spatial matrices")
        if time_key not in adata.obs or row_id_key not in adata.obs:
            raise ContractError("H5AD lacks contracted time/row_id columns")
        state_dim = int(
            contract.get("state_dim", np.asarray(adata.obsm[state_key]).shape[1])
        )
        spatial_dim = int(
            contract.get("spatial_dim", np.asarray(adata.obsm[spatial_key]).shape[1])
        )
        n_obs = int(adata.n_obs)
        state = _validated_matrix(
            f"obsm[{state_key!r}]", adata.obsm[state_key], n_obs, state_dim
        )
        spatial = _validated_matrix(
            f"obsm[{spatial_key!r}]", adata.obsm[spatial_key], n_obs, spatial_dim
        )
        time = np.asarray(adata.obs[time_key], dtype=np.float64)
        row_id = adata.obs[row_id_key].astype(str).to_numpy(dtype=str)
    finally:
        adata.file.close()

    if time.shape != (n_obs,) or not np.isfinite(time).all():
        raise ContractError("training time vector is invalid")
    if row_id.shape != (n_obs,) or len(set(row_id)) != n_obs:
        raise ContractError("training row_id values must be unique")
    actual_times = tuple(sorted({canonical_time(value) for value in time}))
    if len(actual_times) != len(split.observed_times) or any(
        not same_time(left, right)
        for left, right in zip(actual_times, split.observed_times)
    ):
        raise ContractError(
            f"H5AD times {actual_times} differ from manifest {split.observed_times}"
        )
    declared_time_values = contract.get("time_values")
    if declared_time_values is not None:
        contract_times = tuple(
            sorted({canonical_time(value) for value in declared_time_values})
        )
        if len(contract_times) != len(actual_times) or any(
            not same_time(left, right)
            for left, right in zip(contract_times, actual_times)
        ):
            raise ContractError("H5AD contract time_values differ from actual rows")
    if split.regime == "loto" and np.any(
        np.isclose(time, split.holdout_time, rtol=0.0, atol=1e-8)
    ):
        raise ContractError("held-out rows physically occur in LOTO train H5AD")

    with np.load(split.training_reference_npz, allow_pickle=False) as reference:
        required = {"time", "spatial", "state", "row_id"}
        missing = sorted(required - set(reference.files))
        if missing:
            raise ContractError(f"training reference NPZ lacks keys {missing}")
        ref_time = np.asarray(reference["time"], dtype=np.float64)
        ref_spatial = np.asarray(reference["spatial"], dtype=np.float32)
        ref_state = np.asarray(reference["state"], dtype=np.float32)
        ref_row = np.asarray(reference["row_id"]).astype(str)
    for label, left, right in (
        ("time", ref_time, time),
        ("spatial", ref_spatial, spatial),
        ("state", ref_state, state),
        ("row_id", ref_row, row_id),
    ):
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
        h5ad_contract=contract,
    )


def source_time_for_fit(split: SplitContract) -> float:
    if split.regime == "full_data":
        return min(split.observed_times)
    assert split.holdout_time is not None
    candidates = [time for time in split.observed_times if time < split.holdout_time]
    return max(candidates)


def validate_target(split: SplitContract, target_time: float) -> float:
    target = canonical_time(target_time)
    if not any(same_time(target, value) for value in split.evaluation_targets):
        raise ContractError(
            f"target {target} is outside split evaluation_targets={split.evaluation_targets}"
        )
    return target


def bootstrap_source_indices(
    data: TrainingData, source_time: float, *, prediction_n: int, seed: int
) -> np.ndarray:
    if prediction_n != PREDICTION_N:
        raise ContractError(
            f"primary dynamic benchmark fixes prediction_n={PREDICTION_N}"
        )
    candidates = np.flatnonzero(np.isclose(data.time, source_time, rtol=0.0, atol=1e-8))
    if not candidates.size:
        raise ContractError(f"source stage {source_time} contains no training rows")
    rng = np.random.default_rng(seed)
    return rng.choice(
        candidates,
        size=prediction_n,
        replace=candidates.size < prediction_n,
    ).astype(np.int64)


def physical_to_encoded_time(observed_times: tuple[float, ...], target: float) -> float:
    """Map canonical physical time to MIOFlow's ordinal time factorization."""
    times = np.asarray(observed_times, dtype=np.float64)
    target = canonical_time(target)
    exact = np.flatnonzero(np.isclose(times, target, rtol=0.0, atol=1e-8))
    if exact.size:
        return float(exact[0])
    left = np.flatnonzero(times < target)
    right = np.flatnonzero(times > target)
    if not left.size or not right.size:
        raise ContractError(
            f"target {target} cannot be interpolated within observed times {times.tolist()}"
        )
    lo, hi = int(left[-1]), int(right[0])
    alpha = (target - times[lo]) / (times[hi] - times[lo])
    return float(lo + alpha * (hi - lo))


def canonical_adata(data: TrainingData):
    import anndata as ad
    import pandas as pd

    obs = pd.DataFrame(
        {"time": data.time.astype(np.float32), "row_id": data.row_id.astype(str)},
        index=pd.Index(data.row_id.astype(str), name="row_id"),
    )
    result = ad.AnnData(
        X=data.state.astype(np.float32),
        obs=obs,
        var=pd.DataFrame(
            index=[f"state_{index:03d}" for index in range(data.state_dim)]
        ),
    )
    result.obsm["spatial"] = data.spatial.astype(np.float32)
    result.obsm["X_pca"] = data.state.astype(np.float32)
    result.obsm["X_stories"] = data.state.astype(np.float32)
    result.obsm["X_spatial_input"] = data.spatial.astype(np.float32)
    result.obsm["X_spatial_aligned"] = data.spatial.astype(np.float32)
    result.obsm["X_gene_input"] = data.state.astype(np.float32)
    result.obs["time_input"] = result.obs["time"].to_numpy()
    return result


def git_identity(source_root: Path) -> dict[str, Any]:
    source_root = Path(source_root).expanduser().resolve()
    toplevel = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "--show-toplevel"],
        check=False,
        capture_output=True,
        text=True,
    )
    if toplevel.returncode:
        raise ContractError(f"{source_root} is not an auditable git checkout")
    try:
        actual_toplevel = Path(toplevel.stdout.strip()).expanduser().resolve()
    except (OSError, RuntimeError) as exc:
        raise ContractError(f"{source_root} has an invalid git toplevel") from exc
    if actual_toplevel != source_root:
        raise ContractError(
            "official source root must be the exact git toplevel: "
            f"requested {source_root}, found {actual_toplevel}"
        )
    head = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    commit = head.stdout.strip()
    if head.returncode or len(commit) != 40:
        raise ContractError(f"{source_root} is not an auditable git checkout")
    status = subprocess.run(
        [
            "git",
            "-C",
            str(source_root),
            "status",
            "--porcelain",
            "--untracked-files=normal",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if status.returncode != 0:
        raise ContractError(f"cannot audit official source checkout: {source_root}")
    if status.stdout != "":
        raise ContractError(
            "official source checkout has tracked, staged, or untracked changes: "
            f"{source_root}"
        )
    remote = subprocess.run(
        ["git", "-C", str(source_root), "config", "--get", "remote.origin.url"],
        check=False,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "source_root": str(source_root),
        "source_git_commit": commit,
        "source_remote": remote or None,
        "source_tracked_tree_clean": True,
        "source_worktree_clean": True,
    }


def add_source_to_path(source_root: Path) -> Path:
    source_root = Path(source_root).expanduser().resolve()
    candidate = source_root / "src"
    import_root = candidate if candidate.is_dir() else source_root
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))
    return import_root


def package_version(module: Any, distribution_candidates: Sequence[str]) -> str | None:
    version = getattr(module, "__version__", None)
    if version is not None:
        return str(version)
    for name in distribution_candidates:
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return None


def environment_provenance(method: str, method_version: str | None) -> dict[str, Any]:
    distributions = {}
    for name in ("numpy", "anndata", "torch", "jax", "torchdiffeq"):
        try:
            distributions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            distributions[name] = None
    payload = {
        "python_executable": os.path.realpath(sys.executable),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "method": method,
        "method_version": method_version,
        "dependency_versions": distributions,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["environment_fingerprint_sha256"] = hashlib.sha256(encoded).hexdigest()
    return payload


def input_provenance(split: SplitContract) -> dict[str, Any]:
    return {
        "input_manifest": str(split.root_manifest),
        "input_manifest_sha256": split.root_manifest_sha256,
        "train_h5ad": str(split.train_h5ad),
        "train_h5ad_sha256": split.train_h5ad_declared_sha256,
        "training_reference_npz": str(split.training_reference_npz),
        "training_reference_sha256": split.training_reference_declared_sha256,
        "source_roster_npz": str(split.source_roster_npz),
        "source_roster_sha256": split.source_roster_declared_sha256,
        "truth_inputs_opened": False,
    }


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    Path(path).write_text(
        json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
