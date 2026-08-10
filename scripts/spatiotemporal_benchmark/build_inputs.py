#!/usr/bin/env python3
"""Build immutable, dataset-configured spatiotemporal benchmark inputs.

The builder creates one full-data split and one physical leave-one-timepoint-out
(LOTO) split per configured target.  A LOTO train H5AD never contains rows from
its held-out time.  State and spatial representations are copied from the source
H5AD; they are not refit per fold, so this is a transductive frozen-
representation benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import anndata as ad
import numpy as np
import pandas as pd
import yaml
from scipy import sparse


CONTRACT_VERSION = "cytobridge-spatiotemporal-benchmark-input-v1"
DEFAULT_CONTRACT_UNS_KEY = "cytobridge_benchmark_contract"
SOURCE_ROSTER_ALGORITHM = "ranked-support-bootstrap-v1"


class ContractError(ValueError):
    """Raised when an input violates the configured benchmark contract."""


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (pd.Series, pd.Index)):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot JSON-encode {type(value).__name__}")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_jsonable) + "\n",
        encoding="utf-8",
    )


def _write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        yaml.safe_dump(dict(payload), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def _write_sha_sidecar(path: Path) -> Path:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    sidecar.write_text(f"{sha256(path)}  {path.name}\n", encoding="utf-8")
    return sidecar


def _load_json_value(value: str) -> Any:
    text = str(value).strip()
    if text.startswith("@"):
        text = Path(text[1:]).expanduser().read_text(encoding="utf-8")
    elif not text.startswith(("{", "[")):
        candidate = Path(text).expanduser()
        if candidate.is_file():
            text = candidate.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ContractError(f"Invalid JSON value {value!r}: {exc}") from exc


def _parse_int_list(value: str | Sequence[int]) -> list[int]:
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("["):
            raw = _load_json_value(text)
        else:
            raw = [part.strip() for part in text.split(",") if part.strip()]
    else:
        raw = list(value)
    try:
        result = [int(item) for item in raw]
    except (TypeError, ValueError) as exc:
        raise ContractError(f"Expected an integer target list, found {raw!r}") from exc
    if not result or len(result) != len(set(result)):
        raise ContractError(f"Target list must be non-empty and unique, found {result!r}")
    return result


def _normalise_digest(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).lower().strip()
    if text in {"", "none", "null"}:
        return None
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise ContractError("expected_source_sha256 must be a 64-character hex digest or null")
    return text


def _slug(value: str) -> str:
    result = re.sub(r"[^a-zA-Z0-9]+", "_", str(value)).strip("_").lower()
    if not result:
        raise ContractError("dataset_id must contain at least one letter or number")
    return result


def _scalar_equal(left: Any, right: Any) -> bool:
    if pd.isna(left) or pd.isna(right):
        return False
    numeric_types = (int, float, np.integer, np.floating)
    if isinstance(left, numeric_types) and isinstance(right, numeric_types):
        return bool(np.isclose(float(left), float(right), rtol=0.0, atol=1e-10))
    return str(left) == str(right)


def _mapping_as_pairs(value: Any) -> list[tuple[Any, int]]:
    if isinstance(value, Mapping):
        raw_pairs = list(value.items())
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        raw_pairs = []
        for index, item in enumerate(value):
            if not isinstance(item, Sequence) or isinstance(item, (str, bytes)) or len(item) != 2:
                raise ContractError(
                    "time_map list form must contain [source, benchmark] pairs; "
                    f"entry {index} is {item!r}"
                )
            raw_pairs.append((item[0], item[1]))
    else:
        raise ContractError("time_map must be a mapping or a list of two-item pairs")
    if not raw_pairs:
        raise ContractError("time_map must not be empty")
    result: list[tuple[Any, int]] = []
    for source, target in raw_pairs:
        try:
            target_int = int(target)
        except (TypeError, ValueError) as exc:
            raise ContractError(f"Benchmark time must be an integer, found {target!r}") from exc
        if float(target_int) != float(target):
            raise ContractError(f"Benchmark time must be integer-like, found {target!r}")
        if any(_scalar_equal(source, prior_source) for prior_source, _ in result):
            raise ContractError(f"time_map contains equivalent duplicate source {source!r}")
        result.append((source, target_int))
    if len({target for _, target in result}) != len(result):
        raise ContractError("time_map must be one-to-one for this benchmark input contract")
    return result


def _canonical_times(values: Iterable[Any], pairs: Sequence[tuple[Any, int]]) -> np.ndarray:
    observed = list(values)
    mapped: list[int] = []
    matched_sources: set[int] = set()
    for value in observed:
        matches = [index for index, (source, _) in enumerate(pairs) if _scalar_equal(value, source)]
        if len(matches) != 1:
            raise ContractError(
                f"Observed time {value!r} matched {len(matches)} configured time_map entries"
            )
        index = matches[0]
        matched_sources.add(index)
        mapped.append(int(pairs[index][1]))
    if matched_sources != set(range(len(pairs))):
        missing = [pairs[index][0] for index in sorted(set(range(len(pairs))) - matched_sources)]
        raise ContractError(f"Configured source times are absent from the H5AD: {missing!r}")
    return np.asarray(mapped, dtype=np.int16)


def _time_pairs_jsonable(pairs: Sequence[tuple[Any, int]]) -> list[list[Any]]:
    return [[_jsonable_scalar(source), int(target)] for source, target in pairs]


def _jsonable_scalar(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def load_config(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ContractError(f"Configuration must be a YAML mapping: {path}")
    return deepcopy(dict(payload))


def resolve_config(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    overrides = {
        "input_h5ad": args.h5ad,
        "dataset_id": args.dataset,
        "time_key": args.time_key,
        "state_key": args.state_key,
        "spatial_key": args.spatial_key,
        "annotation_key": args.annotation_key,
        "prediction_n": args.prediction_n,
        "source_roster_support_n": args.source_roster_support_n,
        "source_roster_seed": args.source_roster_seed,
        "contract_uns_key": args.contract_uns_key,
        "expected_source_sha256": args.expected_source_sha256,
    }
    for key, value in overrides.items():
        if value is not None:
            config[key] = str(value) if isinstance(value, Path) else value
    if args.time_map is not None:
        config["time_map"] = _load_json_value(args.time_map)
    if args.loto_targets is not None:
        config["loto_targets"] = _parse_int_list(args.loto_targets)
    if args.full_data_targets is not None:
        config["full_data_targets"] = _parse_int_list(args.full_data_targets)
    if args.preprocess_contract_json is not None:
        value = _load_json_value(args.preprocess_contract_json)
        if not isinstance(value, Mapping):
            raise ContractError("--preprocess-contract-json must decode to an object")
        config["preprocess_contract"] = dict(value)

    required = [
        "dataset_id",
        "input_h5ad",
        "time_key",
        "time_map",
        "state_key",
        "state_dim",
        "spatial_key",
        "spatial_dim",
        "annotation_key",
        "loto_targets",
        "full_data_targets",
        "prediction_n",
        "preprocess_contract",
    ]
    missing = [key for key in required if key not in config]
    if missing:
        raise ContractError(f"Configuration is missing required keys: {missing}")
    config["dataset_id"] = str(config["dataset_id"])
    config["dataset_slug"] = _slug(config["dataset_id"])
    config["input_h5ad"] = str(Path(config["input_h5ad"]).expanduser().resolve())
    config["state_dim"] = int(config["state_dim"])
    config["spatial_dim"] = int(config["spatial_dim"])
    config["prediction_n"] = int(config["prediction_n"])
    config["source_roster_support_n"] = int(
        config.get("source_roster_support_n", 800)
    )
    config["source_roster_seed"] = int(config.get("source_roster_seed", 20260718))
    config["source_roster_algorithm"] = SOURCE_ROSTER_ALGORITHM
    config["loto_targets"] = _parse_int_list(config["loto_targets"])
    config["full_data_targets"] = _parse_int_list(config["full_data_targets"])
    config["expected_source_sha256"] = _normalise_digest(
        config.get("expected_source_sha256")
    )
    config["contract_uns_key"] = str(
        config.get("contract_uns_key", DEFAULT_CONTRACT_UNS_KEY)
    )
    config["benchmark_state_key"] = str(config.get("benchmark_state_key", "benchmark_state"))
    config["benchmark_spatial_key"] = str(
        config.get("benchmark_spatial_key", "benchmark_spatial")
    )
    config["row_id_key"] = str(config.get("row_id_key", "row_id"))
    config["benchmark_time_key"] = str(config.get("benchmark_time_key", "benchmark_time"))
    config["benchmark_annotation_key"] = str(
        config.get("benchmark_annotation_key", "benchmark_annotation")
    )
    if config["state_dim"] <= 0 or config["spatial_dim"] <= 0:
        raise ContractError("state_dim and spatial_dim must be positive")
    if config["prediction_n"] <= 0:
        raise ContractError("prediction_n must be positive")
    if config["source_roster_support_n"] <= 0:
        raise ContractError("source_roster_support_n must be positive")
    pairs = _mapping_as_pairs(config["time_map"])
    benchmark_times = [target for _, target in pairs]
    all_targets = set(config["loto_targets"]) | set(config["full_data_targets"])
    missing_targets = sorted(all_targets - set(benchmark_times))
    if missing_targets:
        raise ContractError(f"Evaluation targets are absent from time_map: {missing_targets}")
    config["time_map"] = _time_pairs_jsonable(pairs)
    config["benchmark_times"] = sorted(benchmark_times)
    contract = config["preprocess_contract"]
    if not isinstance(contract, Mapping):
        raise ContractError("preprocess_contract must be a mapping")
    contract = deepcopy(dict(contract))
    audits = contract.get("external_audits", [])
    if not isinstance(audits, Sequence) or isinstance(audits, (str, bytes)):
        raise ContractError("preprocess_contract.external_audits must be a sequence")
    normalized_audits: list[dict[str, Any]] = []
    config_dir = args.config.expanduser().resolve().parent
    for index, raw_audit in enumerate(audits):
        if not isinstance(raw_audit, Mapping):
            raise ContractError(f"external_audits[{index}] must be a mapping")
        audit = deepcopy(dict(raw_audit))
        if "path" not in audit or "sha256" not in audit:
            raise ContractError(f"external_audits[{index}] requires path and sha256")
        path = Path(str(audit["path"])).expanduser()
        audit["path"] = str(
            (path if path.is_absolute() else config_dir / path).resolve()
        )
        audit["sha256"] = _normalise_digest(audit["sha256"])
        if audit["sha256"] is None:
            raise ContractError(f"external_audits[{index}].sha256 is required")
        required_exact = audit.get("required_exact", {})
        if not isinstance(required_exact, Mapping):
            raise ContractError(
                f"external_audits[{index}].required_exact must be a mapping"
            )
        audit["required_exact"] = deepcopy(dict(required_exact))
        audit["name"] = str(audit.get("name", f"external_audit_{index}"))
        normalized_audits.append(audit)
    contract["external_audits"] = normalized_audits
    config["preprocess_contract"] = contract
    return config


def _take_rows(matrix: Any, rows: np.ndarray) -> np.ndarray:
    selected = matrix[rows]
    if hasattr(selected, "to_memory"):
        selected = selected.to_memory()
    if sparse.issparse(selected):
        selected = selected.toarray()
    return np.asarray(selected)


def _sample_rows(n_rows: int, maximum: int = 256) -> np.ndarray:
    if n_rows <= 0:
        return np.empty(0, dtype=np.int64)
    return np.unique(np.linspace(0, n_rows - 1, min(n_rows, maximum), dtype=np.int64))


def _matrix_stats(matrix: Any, n_rows: int, *, integer_like: bool) -> dict[str, Any]:
    values = _take_rows(matrix, _sample_rows(n_rows)).astype(np.float64, copy=False).ravel()
    finite = np.isfinite(values)
    finite_values = values[finite]
    result: dict[str, Any] = {
        "rows_sampled": int(min(n_rows, 256)),
        "values_checked": int(values.size),
        "all_finite": bool(np.all(finite)),
        "nonnegative": bool(np.all(finite_values >= 0)) if finite_values.size else True,
        "min": float(finite_values.min()) if finite_values.size else 0.0,
        "max": float(finite_values.max()) if finite_values.size else 0.0,
    }
    if integer_like:
        result["integer_like_fraction"] = (
            float(np.mean(np.isclose(finite_values, np.rint(finite_values), atol=1e-6)))
            if finite_values.size
            else 1.0
        )
    return result


def _require_finite_array(name: str, value: Any, shape: tuple[int, int]) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape:
        raise ContractError(f"{name} must have shape {shape}, found {array.shape}")
    if not np.issubdtype(array.dtype, np.number):
        raise ContractError(f"{name} must be numeric, found dtype {array.dtype}")
    if not np.isfinite(array).all():
        raise ContractError(f"{name} contains non-finite values")
    return array


def _values_equal(actual: Any, expected: Any) -> bool:
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping):
            return False
        return all(key in actual and _values_equal(actual[key], value) for key, value in expected.items())
    if isinstance(expected, Sequence) and not isinstance(expected, (str, bytes)):
        if isinstance(actual, np.ndarray):
            actual = actual.tolist()
        if not isinstance(actual, Sequence) or isinstance(actual, (str, bytes)):
            return False
        return len(actual) == len(expected) and all(
            _values_equal(left, right) for left, right in zip(actual, expected)
        )
    if isinstance(expected, (int, float, np.number)) and isinstance(actual, (int, float, np.number)):
        return bool(np.isclose(float(actual), float(expected), rtol=0.0, atol=1e-12))
    return actual == expected


def _validate_preprocess_provenance(adata: ad.AnnData, contract: Mapping[str, Any]) -> dict[str, Any]:
    uns_key = str(contract.get("uns_key", "preprocess_info"))
    raw = adata.uns.get(uns_key)
    if raw is None:
        raise ContractError(f"Missing uns[{uns_key!r}]; preprocessing provenance is required")
    if not isinstance(raw, Mapping):
        raise ContractError(f"uns[{uns_key!r}] must be a mapping")
    info = dict(raw)
    failures: list[str] = []
    exact = contract.get("required_exact", {})
    if not isinstance(exact, Mapping):
        raise ContractError("preprocess_contract.required_exact must be a mapping")
    for key, expected in exact.items():
        if key not in info:
            failures.append(f"missing {key!r}")
        elif not _values_equal(info[key], expected):
            failures.append(f"{key!r}: expected {expected!r}, found {info[key]!r}")
    contains = contract.get("required_contains", {})
    if not isinstance(contains, Mapping):
        raise ContractError("preprocess_contract.required_contains must be a mapping")
    for key, fragment in contains.items():
        if key not in info or str(fragment) not in str(info[key]):
            failures.append(f"{key!r} must contain {fragment!r}, found {info.get(key)!r}")
    if failures:
        raise ContractError(
            f"uns[{uns_key!r}] violates preprocessing provenance contract: " + "; ".join(failures)
        )
    return {
        "uns_key": uns_key,
        "status": "passed",
        "required_exact": deepcopy(dict(exact)),
        "required_contains": deepcopy(dict(contains)),
        "observed": info,
    }


def _validate_external_audits(contract: Mapping[str, Any]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for index, audit in enumerate(contract.get("external_audits", [])):
        path = Path(str(audit["path"])).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        observed_sha = sha256(path)
        expected_sha = str(audit["sha256"]).lower()
        if observed_sha != expected_sha:
            raise ContractError(
                f"external audit {path} SHA-256 mismatch: expected {expected_sha}, "
                f"observed {observed_sha}"
            )
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ContractError(f"cannot read external audit JSON {path}: {exc}") from exc
        if not isinstance(payload, Mapping):
            raise ContractError(f"external audit {path} must contain a JSON object")
        required_exact = audit.get("required_exact", {})
        failures = [
            f"{key!r}: expected {expected!r}, found {payload.get(key)!r}"
            for key, expected in required_exact.items()
            if key not in payload or not _values_equal(payload[key], expected)
        ]
        if failures:
            raise ContractError(
                f"external audit {path} violates required_exact: " + "; ".join(failures)
            )
        results.append(
            {
                "name": str(audit.get("name", f"external_audit_{index}")),
                "path": str(path),
                "sha256": observed_sha,
                "status": "passed",
                "required_exact": deepcopy(dict(required_exact)),
            }
        )
    return results


def inspect_input(config: Mapping[str, Any]) -> tuple[ad.AnnData, np.ndarray, dict[str, Any]]:
    h5ad_path = Path(config["input_h5ad"])
    if not h5ad_path.is_file():
        raise FileNotFoundError(h5ad_path)
    source = ad.read_h5ad(h5ad_path, backed="r")
    try:
        if source.n_obs == 0 or source.n_vars == 0:
            raise ContractError(f"Empty AnnData shape {source.shape}")
        for key in (config["time_key"], config["annotation_key"]):
            if key not in source.obs:
                raise ContractError(f"Missing obs[{key!r}]")
        if source.obs[config["annotation_key"]].isna().any():
            raise ContractError(f"obs[{config['annotation_key']!r}] contains missing values")
        if config["state_key"] not in source.obsm:
            raise ContractError(f"Missing obsm[{config['state_key']!r}]")
        if config["spatial_key"] not in source.obsm:
            raise ContractError(f"Missing obsm[{config['spatial_key']!r}]")
        preprocess = config["preprocess_contract"]
        required_layers = [str(key) for key in preprocess.get("required_layers", [])]
        missing_layers = [key for key in required_layers if key not in source.layers]
        if missing_layers:
            raise ContractError(f"Missing required layers: {missing_layers}")

        pairs = _mapping_as_pairs(config["time_map"])
        benchmark_time = _canonical_times(source.obs[config["time_key"]].tolist(), pairs)
        state = _require_finite_array(
            f"obsm[{config['state_key']!r}]",
            source.obsm[config["state_key"]],
            (source.n_obs, int(config["state_dim"])),
        )
        spatial_values = _require_finite_array(
            f"obsm[{config['spatial_key']!r}]",
            source.obsm[config["spatial_key"]],
            (source.n_obs, int(config["spatial_dim"])),
        )
        provenance = _validate_preprocess_provenance(source, preprocess)
        provenance["external_audits"] = _validate_external_audits(preprocess)
        matrix_checks = dict(preprocess.get("matrix_checks", {}))
        x_stats = _matrix_stats(source.X, source.n_obs, integer_like=False)
        if bool(matrix_checks.get("x_finite", True)) and not x_stats["all_finite"]:
            raise ContractError(f"X contains non-finite values: {x_stats}")
        if bool(matrix_checks.get("x_nonnegative", True)) and not x_stats["nonnegative"]:
            raise ContractError(f"X contains negative values: {x_stats}")
        counts_layer = str(preprocess.get("counts_layer", "counts"))
        counts_stats = None
        if counts_layer in source.layers:
            counts_stats = _matrix_stats(source.layers[counts_layer], source.n_obs, integer_like=True)
            if bool(matrix_checks.get("counts_finite", True)) and not counts_stats["all_finite"]:
                raise ContractError(f"layers[{counts_layer!r}] contains non-finite values")
            if bool(matrix_checks.get("counts_nonnegative", True)) and not counts_stats["nonnegative"]:
                raise ContractError(f"layers[{counts_layer!r}] contains negative values")
            threshold = float(matrix_checks.get("counts_integer_like_min_fraction", 0.9999))
            if counts_stats["integer_like_fraction"] < threshold:
                raise ContractError(
                    f"layers[{counts_layer!r}] is not integer-like in deterministic sample; "
                    f"fraction={counts_stats['integer_like_fraction']:.6f}, required={threshold:.6f}"
                )

        time_counts = {
            str(time): int(np.count_nonzero(benchmark_time == time))
            for time in config["benchmark_times"]
        }
        report = {
            "shape": [int(source.n_obs), int(source.n_vars)],
            "time_key": config["time_key"],
            "time_map": _time_pairs_jsonable(pairs),
            "time_counts": time_counts,
            "annotation_key": config["annotation_key"],
            "state_key": config["state_key"],
            "state_shape": list(state.shape),
            "spatial_key": config["spatial_key"],
            "spatial_shape": list(spatial_values.shape),
            "layers": sorted(str(key) for key in source.layers.keys()),
            "x_stats": x_stats,
            "counts_layer": counts_layer,
            "counts_stats": counts_stats,
            "preprocess_provenance": provenance,
        }
        return source, benchmark_time, report
    except Exception:
        source.file.close()
        raise


def _attach_contract_fields(
    source: ad.AnnData,
    benchmark_time: np.ndarray,
    config: Mapping[str, Any],
    source_sha: str,
) -> ad.AnnData:
    data = source.to_memory() if source.isbacked else source.copy()
    row_key = config["row_id_key"]
    time_key = config["benchmark_time_key"]
    annotation_key = config["benchmark_annotation_key"]
    row_ids = np.asarray(
        [f"{config['dataset_slug']}_r{index:08d}" for index in range(data.n_obs)],
        dtype=object,
    )
    data.obs["benchmark_original_obs_name"] = data.obs_names.astype(str).to_numpy(copy=True)
    data.obs["benchmark_source_time"] = data.obs[config["time_key"]].astype(str).to_numpy()
    data.obs[row_key] = row_ids
    data.obs[time_key] = benchmark_time.astype(np.int16)
    data.obs["time_point_processed"] = benchmark_time.astype(float)
    data.obs[annotation_key] = data.obs[config["annotation_key"]].astype(str).to_numpy()
    data.obs_names = pd.Index(row_ids, name=row_key)
    data.obsm[config["benchmark_state_key"]] = np.asarray(
        data.obsm[config["state_key"]], dtype=np.float32
    ).copy()
    data.obsm[config["benchmark_spatial_key"]] = np.asarray(
        data.obsm[config["spatial_key"]], dtype=np.float32
    ).copy()
    contract = {
        "version": CONTRACT_VERSION,
        "dataset_id": config["dataset_id"],
        "source_h5ad_sha256": source_sha,
        "expression_key": "X",
        "source_time_key": config["time_key"],
        "time_key": time_key,
        "time_map_json": json.dumps(config["time_map"], default=_jsonable),
        "source_state_key": config["state_key"],
        "state_key": config["benchmark_state_key"],
        "state_dim": int(config["state_dim"]),
        "source_spatial_key": config["spatial_key"],
        "spatial_key": config["benchmark_spatial_key"],
        "spatial_dim": int(config["spatial_dim"]),
        "source_annotation_key": config["annotation_key"],
        "annotation_key": annotation_key,
        "row_id_key": row_key,
        "loto_targets": np.asarray(config["loto_targets"], dtype=np.int16),
        "full_data_targets": np.asarray(config["full_data_targets"], dtype=np.int16),
        "prediction_n": int(config["prediction_n"]),
        "source_roster_support_n": int(config["source_roster_support_n"]),
        "source_roster_seed": int(config["source_roster_seed"]),
        "source_roster_algorithm": SOURCE_ROSTER_ALGORITHM,
        "prediction_n_policy": "fixed_before_truth_access",
        "truth_cell_count_must_not_control_prediction_n": True,
        "transductive_frozen_representation": True,
        "representation_refit_per_fold": False,
        "target_removed": False,
        "preprocess_provenance_contract_passed": True,
    }
    data.uns[config["contract_uns_key"]] = contract
    return data


def _normalise_obs_for_csv(series: pd.Series) -> pd.Series:
    if isinstance(series.dtype, pd.CategoricalDtype) or series.dtype == object:
        return series.astype(str)
    return series


def _model_frame(data: ad.AnnData, config: Mapping[str, Any]) -> pd.DataFrame:
    row_key = config["row_id_key"]
    time_key = config["benchmark_time_key"]
    annotation_key = config["benchmark_annotation_key"]
    spatial_values = np.asarray(data.obsm[config["benchmark_spatial_key"]], dtype=np.float32)
    state = np.asarray(data.obsm[config["benchmark_state_key"]], dtype=np.float32)
    columns: dict[str, Any] = {
        "row_id": data.obs[row_key].astype(str).to_numpy(),
        "source_time": data.obs["benchmark_source_time"].astype(str).to_numpy(),
        "benchmark_time": data.obs[time_key].to_numpy(np.int16),
        "samples": data.obs[time_key].to_numpy(np.int16),
        "annotation": data.obs[annotation_key].astype(str).to_numpy(),
    }
    if config["annotation_key"] != "Annotation":
        columns["Annotation"] = data.obs[annotation_key].astype(str).to_numpy()
    for key in data.obs.columns:
        if key not in columns:
            columns[str(key)] = _normalise_obs_for_csv(data.obs[key]).to_numpy()
    for index in range(int(config["spatial_dim"])):
        columns[f"spatial_{index + 1:02d}"] = spatial_values[:, index]
    for index in range(int(config["state_dim"])):
        columns[f"state_{index + 1:02d}"] = state[:, index]
    joint = np.column_stack([spatial_values, state])
    for index in range(joint.shape[1]):
        columns[f"x{index + 1}"] = joint[:, index]
    return pd.DataFrame(columns)


def _write_h5ad(path: Path, data: ad.AnnData) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    if temporary.exists():
        temporary.unlink()
    data.write_h5ad(temporary, compression="gzip")
    os.replace(temporary, path)


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    if temporary.exists():
        temporary.unlink()
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _write_reference_npz(path: Path, data: ad.AnnData, config: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    if temporary.exists():
        temporary.unlink()
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            spatial=np.asarray(data.obsm[config["benchmark_spatial_key"]], dtype=np.float32),
            state=np.asarray(data.obsm[config["benchmark_state_key"]], dtype=np.float32),
            time=data.obs[config["benchmark_time_key"]].to_numpy(dtype=np.int16),
            row_id=data.obs[config["row_id_key"]].astype(str).to_numpy(dtype=str),
            annotation=data.obs[config["benchmark_annotation_key"]]
            .astype(str)
            .to_numpy(dtype=str),
        )
    os.replace(temporary, path)


def _stable_roster_seed(base_seed: int, *parts: object) -> int:
    payload = "|".join(str(value) for value in (base_seed, *parts)).encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:16], 16) % (2**32 - 1)


def _row_ids_sha256(values: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in np.asarray(values, dtype=str):
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _ranked_support_indices(
    row_ids: np.ndarray,
    maximum: int,
    base_seed: int,
    source_time: int,
) -> np.ndarray:
    if len(row_ids) <= maximum:
        return np.arange(len(row_ids), dtype=np.int64)
    keys = np.asarray(
        [
            hashlib.sha256(
                f"{base_seed}|anchor|{source_time}|{row_id}".encode("utf-8")
            ).digest()
            for row_id in np.asarray(row_ids, dtype=str)
        ],
        dtype="S32",
    )
    return np.sort(
        np.argsort(keys, kind="stable")[: int(maximum)].astype(np.int64)
    )


def _write_source_roster_npz(
    path: Path,
    train: ad.AnnData,
    config: Mapping[str, Any],
    source_time: int,
) -> dict[str, Any]:
    times = train.obs[config["benchmark_time_key"]].to_numpy(dtype=int)
    candidates = np.flatnonzero(times == int(source_time))
    if candidates.size == 0:
        raise ContractError(f"source time t{source_time} is absent from training rows")
    row_ids = train.obs[config["row_id_key"]].astype(str).to_numpy()
    support_local = _ranked_support_indices(
        row_ids[candidates],
        int(config["source_roster_support_n"]),
        int(config["source_roster_seed"]),
        int(source_time),
    )
    support = candidates[support_local]
    support_ids = row_ids[support]
    support_digest = _row_ids_sha256(support_ids)
    bootstrap_seed = _stable_roster_seed(
        int(config["source_roster_seed"]),
        "source_roster",
        int(source_time),
        support_digest,
    )
    rng = np.random.default_rng(bootstrap_seed)
    selected_local = rng.choice(
        len(support),
        size=int(config["prediction_n"]),
        replace=len(support) < int(config["prediction_n"]),
    )
    indices = support[np.asarray(selected_local, dtype=np.int64)]
    selected_ids = row_ids[indices].astype(str)
    temporary = path.with_suffix(path.suffix + ".partial")
    if temporary.exists():
        temporary.unlink()
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            indices=np.asarray(indices, dtype=np.int64),
            row_id=selected_ids,
            source_time=np.asarray([source_time], dtype=np.int16),
            spatial=np.asarray(
                train.obsm[config["benchmark_spatial_key"]][indices],
                dtype=np.float32,
            ),
            state=np.asarray(
                train.obsm[config["benchmark_state_key"]][indices],
                dtype=np.float32,
            ),
            support_row_id=np.asarray(support_ids, dtype=str),
            support_indices=np.asarray(support, dtype=np.int64),
        )
    os.replace(temporary, path)
    return {
        "source_time": int(source_time),
        "support_n": int(len(support)),
        "available_n": int(len(candidates)),
        "prediction_n": int(len(indices)),
        "base_seed": int(config["source_roster_seed"]),
        "bootstrap_seed": int(bootstrap_seed),
        "support_row_id_sha256": support_digest,
        "roster_row_id_sha256": _row_ids_sha256(selected_ids),
        "algorithm": SOURCE_ROSTER_ALGORITHM,
    }


def _link_or_copy(source: Path, destination: Path) -> str:
    try:
        os.link(source, destination)
        return "hardlink_to_train"
    except OSError:
        shutil.copy2(source, destination)
        return "byte_copy_of_train"


def _artifact(path: Path, input_root: Path, **metadata: Any) -> dict[str, Any]:
    result = {
        "path": str(path.resolve()),
        "relative_path": str(path.resolve().relative_to(input_root.resolve())),
        "sha256": sha256(path),
        "size_bytes": int(path.stat().st_size),
    }
    result.update({key: _jsonable_scalar(value) for key, value in metadata.items()})
    return result


def _time_counts(data: ad.AnnData, config: Mapping[str, Any]) -> dict[str, int]:
    values = data.obs[config["benchmark_time_key"]].to_numpy(dtype=int)
    return {
        str(time): int(np.count_nonzero(values == time)) for time in config["benchmark_times"]
    }


def build_inputs(args: argparse.Namespace) -> dict[str, Any]:
    config = resolve_config(args)
    source, benchmark_time, inspection = inspect_input(config)
    h5ad_path = Path(config["input_h5ad"])
    source_sha = sha256(h5ad_path)
    expected_sha = config.get("expected_source_sha256")
    if expected_sha is not None and source_sha != expected_sha:
        source.file.close()
        raise ContractError(
            f"Source SHA-256 mismatch: expected {expected_sha}, found {source_sha}"
        )
    if args.validate_only:
        source.file.close()
        return {
            "status": "validated",
            "contract_version": CONTRACT_VERSION,
            "dataset_id": config["dataset_id"],
            "source_h5ad": str(h5ad_path),
            "source_h5ad_sha256": source_sha,
            "expected_source_sha256": expected_sha,
            "inspection": inspection,
            "resolved_config": config,
        }
    if args.output_dir is None:
        source.file.close()
        raise ContractError("--output-dir is required unless --validate-only is used")

    output_root = args.output_dir.expanduser().resolve()
    input_root = output_root / "inputs"
    if input_root.exists():
        if not args.overwrite and any(input_root.iterdir()):
            source.file.close()
            raise FileExistsError(
                f"Refusing to modify non-empty {input_root}; use a new output or --overwrite"
            )
        if args.overwrite:
            shutil.rmtree(input_root)
    input_root.mkdir(parents=True, exist_ok=True)

    data = _attach_contract_fields(source, benchmark_time, config, source_sha)
    source.file.close()
    if not data.obs_names.is_unique or not data.obs[config["row_id_key"]].is_unique:
        raise AssertionError("Generated benchmark row IDs are not unique")

    resolved_config_path = input_root / "resolved_config.yaml"
    _write_yaml(resolved_config_path, config)
    definitions_path = input_root / "column_definitions.json"
    _write_json(
        definitions_path,
        {
            "h5ad": {
                "X": "source gene expression, unchanged apart from row subsetting",
                "layers": "all source layers, including configured raw counts, preserved",
                f"obsm/{config['benchmark_spatial_key']}": (
                    f"copy of source obsm/{config['spatial_key']} ({config['spatial_dim']}D)"
                ),
                f"obsm/{config['benchmark_state_key']}": (
                    f"copy of source obsm/{config['state_key']} ({config['state_dim']}D)"
                ),
                f"obs/{config['row_id_key']}": "stable deterministic benchmark row identifier",
                f"obs/{config['benchmark_time_key']}": "configured canonical integer time",
                f"obs/{config['benchmark_annotation_key']}": "configured annotation as string",
                "metadata": "all source obs/var/uns/obsm/varm/obsp metadata are retained",
            },
            "csv": {
                "canonical": ["row_id", "source_time", "benchmark_time", "samples", "annotation"],
                "metadata": "all source obs columns plus canonical benchmark fields",
                "spatial": [
                    f"spatial_{index:02d}" for index in range(1, int(config["spatial_dim"]) + 1)
                ],
                "state": [
                    f"state_{index:02d}" for index in range(1, int(config["state_dim"]) + 1)
                ],
                "legacy_joint_aliases": (
                    f"x1..x{int(config['spatial_dim']) + int(config['state_dim'])}: spatial then state"
                ),
            },
            "npz": {
                "keys": ["spatial", "state", "time", "row_id", "annotation"],
                "prediction_n": int(config["prediction_n"]),
            },
        },
    )

    split_specs: list[tuple[str, int | None]] = [("full_data", None)] + [
        (f"loto_t{target}", target) for target in config["loto_targets"]
    ]
    root_splits: dict[str, Any] = {}
    for split_name, held_out in split_specs:
        split_dir = input_root / split_name
        split_dir.mkdir(parents=True, exist_ok=False)
        all_times = data.obs[config["benchmark_time_key"]].to_numpy(dtype=int)
        if held_out is None:
            train_mask = np.ones(data.n_obs, dtype=bool)
            truth_mask = np.ones(data.n_obs, dtype=bool)
            evaluation_targets = list(config["full_data_targets"])
        else:
            train_mask = all_times != held_out
            truth_mask = all_times == held_out
            evaluation_targets = [int(held_out)]
        train = data[train_mask].copy()
        train_contract = train.uns[config["contract_uns_key"]]
        train_contract["split"] = split_name
        train_contract["role"] = "train_and_truth" if held_out is None else "train"
        train_contract["target_removed"] = held_out is not None
        train_contract["held_out_benchmark_time"] = "none" if held_out is None else int(held_out)
        train_contract["evaluation_targets"] = np.asarray(evaluation_targets, dtype=np.int16)
        observed_train_times = sorted(
            int(value)
            for value in np.unique(
                train.obs[config["benchmark_time_key"]].to_numpy(dtype=int)
            )
        )
        source_time = (
            observed_train_times[0]
            if held_out is None
            else max(value for value in observed_train_times if value < int(held_out))
        )
        train_contract["source_time"] = int(source_time)

        train_h5ad = split_dir / "train.h5ad"
        train_csv = split_dir / "train.csv"
        training_reference = split_dir / "training_reference.npz"
        source_roster = split_dir / "source_roster.npz"
        _write_h5ad(train_h5ad, train)
        train_frame = _model_frame(train, config)
        _write_csv(train_csv, train_frame)
        _write_reference_npz(training_reference, train, config)
        source_roster_meta = _write_source_roster_npz(
            source_roster, train, config, source_time
        )

        truth_h5ad = split_dir / "truth.h5ad"
        truth_csv = split_dir / "truth.csv"
        truth_npz = split_dir / "truth.npz"
        if held_out is None:
            h5_mode = _link_or_copy(train_h5ad, truth_h5ad)
            csv_mode = _link_or_copy(train_csv, truth_csv)
            npz_mode = _link_or_copy(training_reference, truth_npz)
            storage_mode = f"h5ad:{h5_mode};csv:{csv_mode};npz:{npz_mode}"
            truth = train
            truth_frame = train_frame
        else:
            truth = data[truth_mask].copy()
            truth_contract = truth.uns[config["contract_uns_key"]]
            truth_contract["split"] = split_name
            truth_contract["role"] = "truth"
            truth_contract["target_removed"] = False
            truth_contract["held_out_benchmark_time"] = int(held_out)
            truth_contract["evaluation_targets"] = np.asarray(evaluation_targets, dtype=np.int16)
            _write_h5ad(truth_h5ad, truth)
            truth_frame = _model_frame(truth, config)
            _write_csv(truth_csv, truth_frame)
            _write_reference_npz(truth_npz, truth, config)
            storage_mode = "independent_train_and_truth"

        train_ids = set(train.obs[config["row_id_key"]].astype(str))
        truth_ids = set(truth.obs[config["row_id_key"]].astype(str))
        if held_out is not None and train_ids & truth_ids:
            raise AssertionError(f"{split_name}: row leakage between train and truth")
        if held_out is not None and held_out in set(
            train.obs[config["benchmark_time_key"]].to_numpy(dtype=int)
        ):
            raise AssertionError(f"{split_name}: held-out target remains in train H5AD")

        truth_by_time: dict[str, dict[str, Any]] = {}
        for target in evaluation_targets:
            mask = truth.obs[config["benchmark_time_key"]].to_numpy(dtype=int) == target
            stage = truth[mask].copy()
            if stage.n_obs == 0:
                raise AssertionError(f"{split_name}: evaluation target t{target} has no truth rows")
            stage_path = split_dir / f"truth_t{target}.npz"
            _write_reference_npz(stage_path, stage, config)
            truth_by_time[str(target)] = _artifact(
                stage_path, input_root, rows=int(stage.n_obs), target=int(target)
            )

        held_out_sources = (
            []
            if held_out is None
            else [source_time for source_time, target in config["time_map"] if int(target) == held_out]
        )
        split_manifest = {
            "contract_version": CONTRACT_VERSION,
            "dataset_id": config["dataset_id"],
            "contract_uns_key": config["contract_uns_key"],
            "split": split_name,
            "protocol": "full_data" if held_out is None else "leave_one_timepoint_out",
            "held_out_benchmark_time": held_out,
            "held_out_source_times": held_out_sources,
            "evaluation_targets": evaluation_targets,
            "storage_mode": storage_mode,
            "transductive_frozen_representation": True,
            "representation_refit_per_fold": False,
            "target_rows_physically_removed_from_train": held_out is not None,
            "prediction_n": int(config["prediction_n"]),
            "source_time": int(source_time),
            "source_roster_support_n": int(config["source_roster_support_n"]),
            "source_roster_seed": int(config["source_roster_seed"]),
            "source_roster_algorithm": SOURCE_ROSTER_ALGORITHM,
            "prediction_n_policy": "fixed_before_truth_access",
            "truth_cell_count_must_not_control_prediction_n": True,
            "train_time_counts": _time_counts(train, config),
            "truth_time_counts": _time_counts(truth, config),
            "train": {
                "h5ad": _artifact(
                    train_h5ad, input_root, rows=int(train.n_obs), columns=int(train.n_vars)
                ),
                "csv": _artifact(
                    train_csv, input_root, rows=int(len(train_frame)), columns=int(len(train_frame.columns))
                ),
                "training_reference_npz": _artifact(
                    training_reference, input_root, rows=int(train.n_obs)
                ),
                "source_roster_npz": _artifact(
                    source_roster, input_root, **source_roster_meta
                ),
            },
            "truth": {
                "h5ad": _artifact(
                    truth_h5ad, input_root, rows=int(truth.n_obs), columns=int(truth.n_vars)
                ),
                "csv": _artifact(
                    truth_csv, input_root, rows=int(len(truth_frame)), columns=int(len(truth_frame.columns))
                ),
                "truth_npz": _artifact(truth_npz, input_root, rows=int(truth.n_obs)),
            },
            "training_reference_npz": str(training_reference.resolve()),
            "source_roster_npz": _artifact(
                source_roster, input_root, **source_roster_meta
            ),
            "truth_npz": str(truth_npz.resolve()),
            "truth_by_time_npz": truth_by_time,
        }
        split_manifest_path = split_dir / "manifest.json"
        _write_json(split_manifest_path, split_manifest)
        split_manifest_sidecar = _write_sha_sidecar(split_manifest_path)
        root_splits[split_name] = {
            **split_manifest,
            "manifest": _artifact(split_manifest_path, input_root),
            "manifest_sha256_sidecar": _artifact(split_manifest_sidecar, input_root),
        }
        if held_out is not None:
            del truth
        del train

    root_manifest = {
        "contract_version": CONTRACT_VERSION,
        "status": "complete",
        "dataset_id": config["dataset_id"],
        "contract_uns_key": config["contract_uns_key"],
        "source": {
            "h5ad": str(h5ad_path.resolve()),
            "h5ad_sha256": source_sha,
            "expected_source_sha256": expected_sha,
            "expected_source_sha256_match": expected_sha is None or source_sha == expected_sha,
            "inspection": inspection,
        },
        "config_source": {
            "path": str(args.config.expanduser().resolve()),
            "sha256": sha256(args.config.expanduser().resolve()),
        },
        "resolved_config": _artifact(resolved_config_path, input_root),
        "column_definitions": _artifact(definitions_path, input_root),
        "time_map": config["time_map"],
        "benchmark_times": config["benchmark_times"],
        "loto_targets": config["loto_targets"],
        "full_data_targets": config["full_data_targets"],
        "prediction_n": int(config["prediction_n"]),
        "source_roster_support_n": int(config["source_roster_support_n"]),
        "source_roster_seed": int(config["source_roster_seed"]),
        "source_roster_algorithm": SOURCE_ROSTER_ALGORITHM,
        "prediction_n_policy": "fixed_before_truth_access",
        "truth_cell_count_must_not_control_prediction_n": True,
        "transductive_frozen_representation": True,
        "representation_refit_per_fold": False,
        "scientific_scope": (
            "LOTO physically removes target rows from method input while reusing state and spatial "
            "representations fitted during shared full-data preprocessing; it is transductive, "
            "not an inductive raw-expression holdout. Full-data evaluation is in-sample and must "
            "not be pooled with LOTO generalization results."
        ),
        "splits": root_splits,
    }
    root_manifest_path = input_root / "manifest.json"
    _write_json(root_manifest_path, root_manifest)
    root_sidecar = _write_sha_sidecar(root_manifest_path)
    root_manifest["manifest_path"] = str(root_manifest_path)
    root_manifest["manifest_sha256"] = sha256(root_manifest_path)
    root_manifest["manifest_sha256_sidecar"] = str(root_sidecar)
    return root_manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="dataset benchmark YAML")
    parser.add_argument("--h5ad", type=Path, help="override config input_h5ad")
    parser.add_argument("--output-dir", type=Path, help="benchmark run root")
    parser.add_argument("--dataset", help="override dataset_id")
    parser.add_argument("--time-key", help="override source obs time key")
    parser.add_argument("--time-map", help="override with JSON, JSON file, or @file")
    parser.add_argument("--state-key", help="override source obsm state key")
    parser.add_argument("--spatial-key", help="override source obsm spatial key")
    parser.add_argument("--annotation-key", help="override source obs annotation key")
    parser.add_argument("--loto-targets", help="comma list or JSON array")
    parser.add_argument("--full-data-targets", help="comma list or JSON array")
    parser.add_argument("--prediction-n", type=int)
    parser.add_argument("--source-roster-support-n", type=int)
    parser.add_argument("--source-roster-seed", type=int)
    parser.add_argument("--expected-source-sha256")
    parser.add_argument("--contract-uns-key")
    parser.add_argument(
        "--preprocess-contract-json",
        help="replace preprocess_contract with a JSON object, file, or @file",
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = build_inputs(args)
    except (ContractError, FileExistsError, FileNotFoundError, OSError, yaml.YAMLError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, default=_jsonable))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
