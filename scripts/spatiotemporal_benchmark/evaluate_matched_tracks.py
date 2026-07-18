#!/usr/bin/env python3
"""Matched LOTO-versus-full-data evaluation with one audited transform.

This is an opt-in companion to ``evaluate_predictions.py``.  It does not
change the primary per-track evaluator.  The matched design deliberately uses
only targets available in both tracks, fits one transform on explicitly named
anchor times whose rows are byte-identical in every participating training
split, and shares stochastic evaluation keys between the paired tracks.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
import sys
import zipfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
for search_path in (SCRIPT_DIR, REPO_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

import evaluate_predictions as primary  # noqa: E402
from CytoBridge.tl.downstream import benchmark as benchmark_metrics  # noqa: E402
from CytoBridge.tl.downstream import evaluation as evaluation_metrics  # noqa: E402
from CytoBridge.tl.downstream.benchmark import (  # noqa: E402
    FrozenBenchmarkTransform,
    evaluate_spatiotemporal_prediction,
    fit_frozen_benchmark_transform,
)


ContractError = primary.ContractError
TRACKS = ("loto", "full_data")
TRACK_SCOPES = {
    "loto": "held_out",
    "full_data": "in_sample",
}
_ARRAY_FIELDS = ("state", "spatial", "time", "row_id")
_CONCATENATED_HASH_FIELDS = ("row_id", "state", "spatial", "time")
_RUN_CONTRACT_SCHEMA_VERSION = "1.0.0"


def _array_signature(values: np.ndarray) -> dict[str, Any]:
    """Fingerprint an array's dtype, shape, and canonical C-order bytes."""

    array = np.asarray(values)
    if array.dtype.hasobject:
        raise ContractError("audit arrays must not use object dtype")
    contiguous = np.ascontiguousarray(array)
    return {
        "dtype": array.dtype.str,
        "shape": [int(value) for value in array.shape],
        "c_order_sha256": hashlib.sha256(contiguous.tobytes(order="C")).hexdigest(),
    }


def _bundle_signature(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    components = {key: _array_signature(arrays[key]) for key in _ARRAY_FIELDS}
    canonical = json.dumps(components, sort_keys=True, separators=(",", ":"))
    concatenated = hashlib.sha256()
    for key in _CONCATENATED_HASH_FIELDS:
        concatenated.update(np.ascontiguousarray(arrays[key]).tobytes(order="C"))
    return {
        "components": components,
        "bundle_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "concatenated_c_order_sha256": concatenated.hexdigest(),
        "concatenated_field_order": list(_CONCATENATED_HASH_FIELDS),
        "byte_definition": "dtype + shape + canonical C-order element bytes",
    }


def _one_dimensional(
    archive: Any,
    key: str,
    *,
    rows: int,
    path: Path,
) -> np.ndarray:
    if key not in archive:
        raise ContractError(f"{path}: missing {key!r}")
    values = np.asarray(archive[key])
    if values.ndim != 1 or values.shape[0] != rows:
        raise ContractError(
            f"{path}: {key!r} must have shape ({rows},), found {values.shape}"
        )
    if values.dtype.hasobject:
        raise ContractError(f"{path}: {key!r} must not use object dtype")
    return np.array(values, copy=True)


def _load_audit_arrays(path: Path) -> dict[str, np.ndarray]:
    """Load the benchmark arrays needed to prove row-level identity."""

    with np.load(path, allow_pickle=False) as archive:
        state = np.asarray(archive["state"]) if "state" in archive else None
        spatial = np.asarray(archive["spatial"]) if "spatial" in archive else None
        if state is None or spatial is None:
            raise ContractError(f"{path}: missing state or spatial")
        if state.ndim != 2 or spatial.ndim != 2 or state.shape[0] == 0:
            raise ContractError(f"{path}: state/spatial must be non-empty matrices")
        if state.shape[0] != spatial.shape[0]:
            raise ContractError(f"{path}: state/spatial row counts differ")
        if state.dtype.hasobject or spatial.dtype.hasobject:
            raise ContractError(f"{path}: state/spatial must not use object dtype")
        state_float = np.asarray(state, dtype=np.float64)
        spatial_float = np.asarray(spatial, dtype=np.float64)
        if not np.isfinite(state_float).all() or not np.isfinite(spatial_float).all():
            raise ContractError(f"{path}: state/spatial contain non-finite values")
        rows = int(state.shape[0])
        time = _one_dimensional(archive, "time", rows=rows, path=path)
        row_id = _one_dimensional(archive, "row_id", rows=rows, path=path)
    try:
        time_float = np.asarray(time, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{path}: time must be numeric") from exc
    if not np.isfinite(time_float).all():
        raise ContractError(f"{path}: time contains non-finite values")
    if len(np.unique(row_id)) != rows:
        raise ContractError(f"{path}: row_id must be unique")
    return {
        "state": np.array(state, copy=True),
        "spatial": np.array(spatial, copy=True),
        "time": time,
        "row_id": row_id,
    }


def _anchor_arrays(
    arrays: dict[str, np.ndarray],
    anchor_times: tuple[float, ...],
    *,
    split_id: str,
) -> dict[str, np.ndarray]:
    time_float = np.asarray(arrays["time"], dtype=np.float64)
    selected: dict[str, list[np.ndarray]] = {key: [] for key in _ARRAY_FIELDS}
    for anchor in anchor_times:
        mask = time_float == anchor
        if not np.any(mask):
            raise ContractError(
                f"anchor time {anchor:g} is absent from training split {split_id}"
            )
        for key in _ARRAY_FIELDS:
            selected[key].append(arrays[key][mask])
    return {key: np.concatenate(parts, axis=0) for key, parts in selected.items()}


def _require_identical_bundles(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    label: str,
) -> None:
    if left != right:
        mismatches = [
            key
            for key in _ARRAY_FIELDS
            if left["components"].get(key) != right["components"].get(key)
        ]
        raise ContractError(
            f"{label} are not byte-identical; mismatched arrays={mismatches}"
        )


def _require_array_bytes_identical(
    left: dict[str, np.ndarray],
    right: dict[str, np.ndarray],
    *,
    label: str,
) -> None:
    mismatches = []
    for key in _ARRAY_FIELDS:
        left_array = np.asarray(left[key])
        right_array = np.asarray(right[key])
        if (
            left_array.dtype.str != right_array.dtype.str
            or left_array.shape != right_array.shape
            or np.ascontiguousarray(left_array).tobytes(order="C")
            != np.ascontiguousarray(right_array).tobytes(order="C")
        ):
            mismatches.append(key)
    if mismatches:
        raise ContractError(
            f"{label} are not byte-identical; mismatched arrays={mismatches}"
        )


def _write_immutable_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != payload:
            raise ContractError(f"refusing to replace a different audited file: {path}")
        return
    temporary = path.with_name(f"{path.name}.partial.{os.getpid()}")
    temporary.write_text(payload, encoding="utf-8")
    try:
        os.link(temporary, path)
    except FileExistsError:
        if path.read_text(encoding="utf-8") != payload:
            raise ContractError(
                f"refusing to replace a concurrently created audited file: {path}"
            )
    finally:
        temporary.unlink(missing_ok=True)


def _strict_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _write_immutable_json(path: Path, payload: dict[str, Any]) -> None:
    _write_immutable_text(path, _strict_json(payload))


def _write_immutable_csv(path: Path, frame: pd.DataFrame) -> None:
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False)
    _write_immutable_text(path, buffer.getvalue())


def _write_final_manifest(path: Path, payload: dict[str, Any]) -> None:
    """Atomically publish a final manifest without any overwrite path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ContractError(f"final manifest exists and is immutable: {path}")
    temporary = path.with_name(f"{path.name}.partial.{os.getpid()}")
    temporary.write_text(_strict_json(payload), encoding="utf-8")
    try:
        os.link(temporary, path)
    except FileExistsError as exc:
        raise ContractError(f"final manifest exists and is immutable: {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def _code_dependencies() -> dict[str, dict[str, str]]:
    paths = {
        "matched_evaluator": Path(__file__).resolve(),
        "primary_evaluator": Path(primary.__file__).resolve(),
        "benchmark_metrics": Path(benchmark_metrics.__file__).resolve(),
        "distribution_metrics": Path(evaluation_metrics.__file__).resolve(),
    }
    return {
        name: {"path": str(path), "sha256": primary.sha256_file(path)}
        for name, path in paths.items()
    }


def _software_versions() -> dict[str, str]:
    import ot
    import scipy

    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable": str(Path(sys.executable).resolve()),
        "numpy": str(np.__version__),
        "pandas": str(pd.__version__),
        "scipy": str(scipy.__version__),
        "POT": str(ot.__version__),
    }


def _dataset_label(root: dict[str, Any]) -> str:
    return str(
        root.get(
            "dataset_id",
            root.get("dataset", root.get("dataset_name", "spatiotemporal")),
        )
    )


def _common_targets(root: dict[str, Any], requested: list[int] | None) -> list[int]:
    loto = set(primary._targets(root, "loto"))
    full = set(primary._targets(root, "full_data"))
    common = sorted(loto.intersection(full))
    if not common:
        raise ContractError("LOTO and full-data tracks have no common targets")
    if requested is None:
        return common
    values = [int(value) for value in requested]
    if not values or len(set(values)) != len(values):
        raise ContractError("--targets must be non-empty and unique")
    invalid = sorted(set(values).difference(common))
    if invalid:
        raise ContractError(
            f"targets {invalid} are not shared by LOTO and full-data tracks"
        )
    return values


def _anchor_times(values: Iterable[float]) -> tuple[float, ...]:
    anchors = tuple(float(value) for value in values)
    if not anchors or not np.isfinite(np.asarray(anchors)).all():
        raise ContractError("--anchor-times must contain finite values")
    if len(set(anchors)) != len(anchors):
        raise ContractError("--anchor-times must be unique")
    return anchors


def _run_contract_payload(
    *,
    args: argparse.Namespace,
    input_manifest: Path,
    manifest_sha: str,
    targets: list[int],
    anchors: tuple[float, ...],
    code_dependencies: dict[str, dict[str, str]],
    software_versions: dict[str, str],
) -> dict[str, Any]:
    return {
        "schema_version": _RUN_CONTRACT_SCHEMA_VERSION,
        "design": "matched_loto_vs_full_data",
        "input_manifest": str(input_manifest),
        "input_manifest_sha256": manifest_sha,
        "loto_predictions_root": str(args.loto_predictions_root.expanduser().resolve()),
        "full_data_predictions_root": str(
            args.full_data_predictions_root.expanduser().resolve()
        ),
        "targets": targets,
        "anchor_times": list(anchors),
        "methods": None if args.methods is None else list(args.methods),
        "include_nonprimary": bool(args.include_nonprimary),
        "n_projections": int(args.n_projections),
        "projection_repeats": int(args.projection_repeats),
        "max_ot_points": int(args.max_ot_points),
        "exact_ot_policy": "matched-separate-rng-v1",
        "code_dependencies": code_dependencies,
        "software_versions": software_versions,
    }


def _validate_requested_config(args: argparse.Namespace) -> None:
    if args.methods is not None:
        methods = [str(value) for value in args.methods]
        if not methods or any(not value.strip() for value in methods):
            raise ContractError("--methods must contain non-empty names")
        if len(set(methods)) != len(methods):
            raise ContractError("--methods must be unique")
    for name in ("n_projections", "projection_repeats", "max_ot_points"):
        value = getattr(args, name)
        if (
            isinstance(value, (bool, np.bool_))
            or int(value) != value
            or int(value) <= 0
        ):
            raise ContractError(
                f"--{name.replace('_', '-')} must be a positive integer"
            )


def _bind_run_contract(
    output_dir: Path,
    payload: dict[str, Any],
) -> tuple[Path, str]:
    """Bind a partial run before any derived artifact is written."""

    final_manifest = output_dir / "matched_evaluation_manifest.json"
    if final_manifest.exists():
        raise ContractError(
            f"completed matched run is immutable; final manifest exists: {final_manifest}"
        )
    contract_path = output_dir / "run_contract.json"
    expected = _strict_json(payload)
    if contract_path.exists():
        observed = contract_path.read_text(encoding="utf-8")
        if observed != expected:
            raise ContractError(
                "output directory is bound to a different matched run contract"
            )
    else:
        if output_dir.exists() and any(output_dir.iterdir()):
            raise ContractError(
                "non-empty output directory has no matched run contract; refusing "
                "to adopt or overwrite partial artifacts"
            )
        _write_immutable_json(contract_path, payload)
    return contract_path, primary.sha256_file(contract_path)


def _subset_arrays(
    arrays: dict[str, np.ndarray], mask: np.ndarray
) -> dict[str, np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (arrays["time"].shape[0],):
        raise ContractError("semantic split mask has the wrong shape")
    return {key: np.asarray(arrays[key])[mask] for key in _ARRAY_FIELDS}


def _row_id_set(values: np.ndarray) -> set[str]:
    return set(np.asarray(values).astype(str).tolist())


def _semantic_split_record(
    *,
    target: int,
    full_training: dict[str, np.ndarray],
    loto_training: dict[str, np.ndarray],
    loto_truth: dict[str, np.ndarray],
    full_truth: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Prove a LOTO NPZ is the exact target complement of full-data NPZ."""

    full_time = np.asarray(full_training["time"], dtype=np.float64)
    loto_time = np.asarray(loto_training["time"], dtype=np.float64)
    loto_truth_time = np.asarray(loto_truth["time"], dtype=np.float64)
    full_truth_time = np.asarray(full_truth["time"], dtype=np.float64)
    target_mask = full_time == float(target)
    if not np.any(target_mask):
        raise ContractError(f"full-data training reference has no target t{target}")
    if np.any(loto_time == float(target)):
        raise ContractError(f"LOTO training reference still contains target t{target}")
    if not np.all(loto_truth_time == float(target)):
        raise ContractError(f"LOTO truth contains rows outside target t{target}")
    if not np.all(full_truth_time == float(target)):
        raise ContractError(f"full-data truth contains rows outside target t{target}")

    expected_train = _subset_arrays(full_training, ~target_mask)
    expected_truth = _subset_arrays(full_training, target_mask)
    expected_train_signature = _bundle_signature(expected_train)
    expected_truth_signature = _bundle_signature(expected_truth)
    loto_train_signature = _bundle_signature(loto_training)
    loto_truth_signature = _bundle_signature(loto_truth)
    full_truth_signature = _bundle_signature(full_truth)
    _require_array_bytes_identical(
        expected_train,
        loto_training,
        label=f"full-data non-t{target} subset and LOTO training",
    )
    _require_array_bytes_identical(
        expected_truth,
        loto_truth,
        label=f"full-data t{target} subset and LOTO truth",
    )
    _require_array_bytes_identical(
        expected_truth,
        full_truth,
        label=f"full-data t{target} subset and full-data truth",
    )
    _require_identical_bundles(
        expected_train_signature,
        loto_train_signature,
        label=f"full-data non-t{target} subset and LOTO training",
    )
    _require_identical_bundles(
        expected_truth_signature,
        loto_truth_signature,
        label=f"full-data t{target} subset and LOTO truth",
    )
    _require_identical_bundles(
        expected_truth_signature,
        full_truth_signature,
        label=f"full-data t{target} subset and full-data truth",
    )

    train_ids = _row_id_set(loto_training["row_id"])
    truth_ids = _row_id_set(loto_truth["row_id"])
    full_ids = _row_id_set(full_training["row_id"])
    if train_ids.intersection(truth_ids):
        raise ContractError(f"LOTO train/truth row_id leakage at target t{target}")
    if train_ids.union(truth_ids) != full_ids:
        raise ContractError(
            f"LOTO train/truth row_id complement does not reconstruct full-data at t{target}"
        )
    if len(train_ids) + len(truth_ids) != len(full_ids):
        raise ContractError(
            f"LOTO row_id counts are not a disjoint complement at t{target}"
        )

    return {
        "target": int(target),
        "status": "complete",
        "loto_training_excludes_target": True,
        "loto_truth_contains_only_target": True,
        "full_data_truth_contains_only_target": True,
        "loto_train_truth_row_ids_disjoint": True,
        "loto_train_truth_row_ids_reconstruct_full_data": True,
        "loto_training_is_byte_exact_full_data_complement": True,
        "loto_truth_is_byte_exact_full_data_target_subset": True,
        "full_data_truth_is_byte_exact_full_data_target_subset": True,
        "arrays_verified": list(_ARRAY_FIELDS),
        "full_data_rows": int(full_time.shape[0]),
        "loto_training_rows": int(loto_time.shape[0]),
        "target_truth_rows": int(loto_truth_time.shape[0]),
        "complement_rows_sum": int(loto_time.shape[0] + loto_truth_time.shape[0]),
        "full_data_bundle_sha256": _bundle_signature(full_training)["bundle_sha256"],
        "non_target_complement_bundle_sha256": expected_train_signature[
            "bundle_sha256"
        ],
        "target_subset_bundle_sha256": expected_truth_signature["bundle_sha256"],
    }


def _load_cases(
    *,
    root: dict[str, Any],
    input_manifest: Path,
    targets: list[int],
    anchor_times: tuple[float, ...],
    output_dir: Path,
) -> tuple[
    dict[tuple[str, int], dict[str, Any]],
    FrozenBenchmarkTransform,
    Path,
    dict[str, Any],
]:
    cases: dict[tuple[str, int], dict[str, Any]] = {}
    split_anchors: dict[str, dict[str, Any]] = {}
    split_anchor_arrays: dict[str, dict[str, np.ndarray]] = {}
    split_paths: dict[str, dict[str, str]] = {}
    truth_pairs: dict[str, Any] = {}
    raw_references: dict[str, dict[str, np.ndarray]] = {}
    raw_truths: dict[tuple[str, int], dict[str, np.ndarray]] = {}

    for target in targets:
        per_track_truth: dict[str, dict[str, Any]] = {}
        for track in TRACKS:
            split_id, split = primary._split_record(root, track, target)
            training_path, training_sha = primary._artifact(
                primary._training_artifact(split),
                base=input_manifest.parent,
                label=f"{split_id}/training_reference",
            )
            roster_path, roster_sha = primary._artifact(
                primary._source_roster_artifact(split),
                base=input_manifest.parent,
                label=f"{split_id}/source_roster",
            )
            truth_path, truth_sha = primary._artifact(
                primary._truth_artifact(split, target),
                base=input_manifest.parent,
                label=f"{split_id}/truth_t{target}",
            )
            train_state, train_spatial, train_time = primary._load_reference(
                training_path
            )
            truth_state, truth_spatial = primary._load_truth(truth_path)
            raw_truth = _load_audit_arrays(truth_path)
            raw_truths[(track, target)] = raw_truth
            truth_signature = _bundle_signature(raw_truth)
            per_track_truth[track] = truth_signature
            cases[(track, target)] = {
                "track": track,
                "split_id": split_id,
                "training_path": training_path,
                "training_sha": training_sha,
                "source_roster_path": roster_path,
                "source_roster_sha": roster_sha,
                "truth_path": truth_path,
                "truth_sha": truth_sha,
                "train_time": train_time,
                "truth_state": truth_state,
                "truth_spatial": truth_spatial,
                "truth_signature": truth_signature,
            }
            if split_id not in split_anchors:
                raw_reference = _load_audit_arrays(training_path)
                raw_references[split_id] = raw_reference
                selected = _anchor_arrays(
                    raw_reference, anchor_times, split_id=split_id
                )
                split_anchor_arrays[split_id] = selected
                split_anchors[split_id] = _bundle_signature(selected)
                split_paths[split_id] = {
                    "training_reference": str(training_path),
                    "training_reference_sha256": training_sha,
                }
            elif split_paths[split_id]["training_reference_sha256"] != training_sha:
                raise ContractError(
                    f"split {split_id} resolves to inconsistent training references"
                )

        _require_identical_bundles(
            per_track_truth["loto"],
            per_track_truth["full_data"],
            label=f"LOTO/full-data truth for t{target}",
        )
        truth_pairs[str(target)] = {
            "byte_identical": True,
            "bundle_sha256": per_track_truth["loto"]["bundle_sha256"],
            "loto_truth": str(cases[("loto", target)]["truth_path"]),
            "loto_truth_sha256": cases[("loto", target)]["truth_sha"],
            "full_data_truth": str(cases[("full_data", target)]["truth_path"]),
            "full_data_truth_sha256": cases[("full_data", target)]["truth_sha"],
        }

    if "full_data" not in raw_references:
        raise ContractError(
            "matched evaluation requires a full_data training reference"
        )
    split_semantics = {
        str(target): _semantic_split_record(
            target=target,
            full_training=raw_references["full_data"],
            loto_training=raw_references[f"loto_t{target}"],
            loto_truth=raw_truths[("loto", target)],
            full_truth=raw_truths[("full_data", target)],
        )
        for target in targets
    }
    for target in targets:
        split_semantics[str(target)]["artifacts"] = {
            "loto_training_reference": str(cases[("loto", target)]["training_path"]),
            "loto_training_reference_sha256": cases[("loto", target)]["training_sha"],
            "full_data_training_reference": str(
                cases[("full_data", target)]["training_path"]
            ),
            "full_data_training_reference_sha256": cases[("full_data", target)][
                "training_sha"
            ],
            "loto_truth": str(cases[("loto", target)]["truth_path"]),
            "loto_truth_sha256": cases[("loto", target)]["truth_sha"],
            "full_data_truth": str(cases[("full_data", target)]["truth_path"]),
            "full_data_truth_sha256": cases[("full_data", target)]["truth_sha"],
        }
    split_audit = {
        "schema_version": "1.0.0",
        "status": "complete",
        "policy": (
            "For each target, LOTO training must be the byte-exact non-target "
            "subset of full-data training and both truth artifacts must be the "
            "byte-exact target subset."
        ),
        "arrays_verified": list(_ARRAY_FIELDS),
        "targets": split_semantics,
        "all_targets_semantically_verified": True,
    }
    split_audit_path = output_dir / "semantic_split_audit.json"
    _write_immutable_json(split_audit_path, split_audit)
    split_audit_sha = primary.sha256_file(split_audit_path)

    canonical_split = "full_data"
    if canonical_split not in split_anchors:
        canonical_split = sorted(split_anchors)[0]
    canonical_signature = split_anchors[canonical_split]
    for split_id, signature in split_anchors.items():
        _require_identical_bundles(
            canonical_signature,
            signature,
            label=f"anchor rows in {canonical_split} and {split_id}",
        )

    canonical_arrays = split_anchor_arrays[canonical_split]
    transform = fit_frozen_benchmark_transform(
        np.asarray(canonical_arrays["state"], dtype=np.float64),
        np.asarray(canonical_arrays["spatial"], dtype=np.float64),
    )
    transform_path = output_dir / "transforms" / "common_anchor_transform.json"
    _write_immutable_text(transform_path, transform.to_json() + "\n")
    transform_sha = primary.sha256_file(transform_path)
    for case in cases.values():
        case["transform_path"] = transform_path
        case["transform_sha"] = transform_sha

    audit = {
        "schema_version": "1.0.0",
        "status": "complete",
        "anchor_times": list(anchor_times),
        "canonical_split": canonical_split,
        "anchor_rows": int(canonical_arrays["state"].shape[0]),
        "anchor_bundle_sha256": canonical_signature["bundle_sha256"],
        "anchor_concatenated_c_order_sha256": canonical_signature[
            "concatenated_c_order_sha256"
        ],
        "anchor_concatenated_field_order": canonical_signature[
            "concatenated_field_order"
        ],
        "anchor_component_signatures": canonical_signature["components"],
        "byte_definition": canonical_signature["byte_definition"],
        "all_participating_training_splits_byte_identical": True,
        "participating_splits": {
            split_id: {
                **split_paths[split_id],
                "anchor_bundle_sha256": signature["bundle_sha256"],
                "anchor_concatenated_c_order_sha256": signature[
                    "concatenated_c_order_sha256"
                ],
            }
            for split_id, signature in sorted(split_anchors.items())
        },
        "paired_truth": truth_pairs,
        "semantic_split_audit": str(split_audit_path),
        "semantic_split_audit_sha256": split_audit_sha,
        "all_targets_semantically_verified": True,
        "transform": str(transform_path),
        "transform_sha256": transform_sha,
    }
    audit_path = output_dir / "common_anchor_audit.json"
    _write_immutable_json(audit_path, audit)
    audit["audit_path"] = str(audit_path)
    audit["audit_sha256"] = primary.sha256_file(audit_path)
    return cases, transform, transform_path, audit


def _scan_predictions(
    *,
    predictions_root: Path,
    track: str,
    targets: list[int],
    cases: dict[tuple[str, int], dict[str, Any]],
    manifest_sha: str,
    methods: set[str] | None,
    include_nonprimary: bool,
) -> dict[tuple[str, int], dict[str, Any]]:
    candidates = sorted(predictions_root.resolve().rglob("prediction.npz"))
    if not candidates:
        raise ContractError(f"no prediction.npz found below {predictions_root}")
    found: dict[tuple[str, int], dict[str, Any]] = {}
    for prediction_path in candidates:
        summary_candidates = (
            prediction_path.with_suffix(".summary.json"),
            prediction_path.parent / "summary.json",
            prediction_path.parent / "run_manifest.json",
        )
        summary_path = next(
            (path for path in summary_candidates if path.is_file()), None
        )
        if summary_path is None:
            raise ContractError(f"prediction has no summary JSON: {prediction_path}")
        try:
            prediction_bytes = prediction_path.read_bytes()
            summary_bytes = summary_path.read_bytes()
        except OSError as exc:
            raise ContractError(
                f"cannot capture prediction/summary byte snapshot for {prediction_path}: {exc}"
            ) from exc
        prediction_sha = hashlib.sha256(prediction_bytes).hexdigest()
        summary_sha = hashlib.sha256(summary_bytes).hexdigest()
        try:
            summary = json.loads(summary_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ContractError(
                f"cannot parse summary snapshot {summary_path}: {exc}"
            ) from exc
        if not isinstance(summary, dict):
            raise ContractError(f"{summary_path} must contain a JSON object")
        if primary._summary_track(summary) != track:
            continue
        target = primary._summary_target(summary)
        if target not in targets:
            continue
        method = primary._summary_method(summary)
        if methods is not None and method not in methods:
            continue
        if not include_nonprimary and not primary._primary_eligible(summary):
            continue
        key = (method, target)
        if key in found:
            raise ContractError(f"duplicate prediction for {method}/{track}/t{target}")
        case = cases[(track, target)]
        primary._verify_prediction_provenance(
            summary,
            manifest_sha=manifest_sha,
            training_sha=case["training_sha"],
            source_roster_sha=case["source_roster_sha"],
        )
        declared_sha = primary._summary_prediction_sha(summary)
        if declared_sha is None:
            raise ContractError(
                f"prediction summary does not record prediction SHA-256: {summary_path}"
            )
        if declared_sha != prediction_sha:
            raise ContractError(
                f"prediction SHA-256 does not match its summary: {prediction_path}"
            )
        try:
            with np.load(io.BytesIO(prediction_bytes), allow_pickle=False) as archive:
                state = primary._matrix(archive, "state")
                spatial = (
                    primary._matrix(archive, "spatial")
                    if "spatial" in archive
                    else None
                )
                if spatial is not None and spatial.shape[0] != state.shape[0]:
                    raise ContractError(
                        f"{prediction_path}: prediction state/spatial row counts differ"
                    )
                weights = None
                if "weights" in archive:
                    weights = np.asarray(archive["weights"], dtype=np.float64).reshape(
                        -1
                    )
                    if weights.shape != (state.shape[0],):
                        raise ContractError(
                            f"{prediction_path}: weights length does not match predictions"
                        )
                    if (
                        not np.isfinite(weights).all()
                        or np.any(weights < 0)
                        or weights.sum() <= 0
                    ):
                        raise ContractError(
                            f"{prediction_path}: weights must be finite, nonnegative "
                            "and nonzero"
                        )
        except (OSError, ValueError, zipfile.BadZipFile) as exc:
            if isinstance(exc, ContractError):
                raise
            raise ContractError(
                f"cannot parse prediction byte snapshot {prediction_path}: {exc}"
            ) from exc
        scope = primary._summary_scope(summary, spatial is not None)
        found[key] = {
            "method": method,
            "target": target,
            "path": prediction_path,
            "sha": prediction_sha,
            "summary_path": summary_path,
            "summary_sha": summary_sha,
            "summary": summary,
            "state": state,
            "spatial": spatial,
            "weights": weights,
            "scope": scope,
            "native_vs_adapter": str(
                summary.get("native_vs_adapter", summary.get("adapter_type", scope))
            ),
            "primary_benchmark_eligible": primary._primary_eligible(summary),
        }
    return found


def _method_grid(
    predictions: dict[str, dict[tuple[str, int], dict[str, Any]]],
    targets: list[int],
    requested: list[str] | None,
) -> list[str]:
    if requested is not None:
        methods = [str(value) for value in requested]
        if not methods or any(not value.strip() for value in methods):
            raise ContractError("--methods must contain non-empty names")
        if len(set(methods)) != len(methods):
            raise ContractError("--methods must be unique")
    else:
        methods = sorted(
            {
                method
                for track_predictions in predictions.values()
                for method, _ in track_predictions
            }
        )
    if not methods:
        raise ContractError("no eligible methods were found")
    missing = []
    for track in TRACKS:
        for method in methods:
            for target in targets:
                if (method, target) not in predictions[track]:
                    missing.append((method, track, target))
    if missing:
        rendered = ", ".join(
            f"{method}/{track}/t{target}" for method, track, target in missing
        )
        raise ContractError(
            f"incomplete matched method-by-track-by-target grid: {rendered}"
        )
    return methods


def _prediction_inventory(
    *,
    predictions: dict[str, dict[tuple[str, int], dict[str, Any]]],
    methods: list[str],
    targets: list[int],
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for method in methods:
        for track in TRACKS:
            for target in targets:
                prediction = predictions[track][(method, target)]
                summary = prediction["summary"]
                records.append(
                    {
                        "method": method,
                        "track": track,
                        "target": int(target),
                        "prediction_path": str(prediction["path"]),
                        "prediction_sha256": prediction["sha"],
                        "prediction_summary": str(prediction["summary_path"]),
                        "prediction_summary_sha256": prediction["summary_sha"],
                        "output_scope": prediction["scope"],
                        "native_vs_adapter": prediction["native_vs_adapter"],
                        "primary_benchmark_eligible": bool(
                            prediction["primary_benchmark_eligible"]
                        ),
                        "representation": str(summary.get("representation", "")),
                        "native_mass": bool(
                            summary.get("native_mass", False)
                            or summary.get("native_growth", False)
                        ),
                        "weights_are_unnormalised": bool(
                            summary.get("weights_are_unnormalised", False)
                        ),
                        "has_spatial": prediction["spatial"] is not None,
                        "has_weights": prediction["weights"] is not None,
                        "state_shape": [
                            int(value) for value in prediction["state"].shape
                        ],
                        "spatial_shape": (
                            None
                            if prediction["spatial"] is None
                            else [int(value) for value in prediction["spatial"].shape]
                        ),
                    }
                )
    records.sort(
        key=lambda record: (
            str(record["method"]),
            str(record["track"]),
            int(record["target"]),
            str(record["prediction_path"]),
        )
    )
    return {
        "schema_version": "1.0.0",
        "status": "complete",
        "snapshot_policy": (
            "Each prediction NPZ and summary JSON was read once as bytes; parsing and "
            "SHA-256 binding used that same in-memory byte snapshot."
        ),
        "sort_key": ["method", "track", "target", "prediction_path"],
        "n_records": int(len(records)),
        "records": records,
    }


def _verify_prediction_inventory_files(inventory: dict[str, Any]) -> None:
    records = inventory.get("records")
    if not isinstance(records, list) or not records:
        raise ContractError("prediction inventory has no records")
    for record in records:
        if not isinstance(record, dict):
            raise ContractError("prediction inventory record must be an object")
        for path_key, sha_key in (
            ("prediction_path", "prediction_sha256"),
            ("prediction_summary", "prediction_summary_sha256"),
        ):
            path = Path(str(record[path_key]))
            expected = str(record[sha_key])
            if not path.is_file():
                raise ContractError(
                    f"bound external prediction artifact is missing: {path}"
                )
            observed = primary.sha256_file(path)
            if observed != expected:
                raise ContractError(
                    f"bound external prediction artifact changed after snapshot: {path}"
                )


def _bind_prediction_inventory(
    *,
    output_dir: Path,
    inventory: dict[str, Any],
    base_run_contract_path: Path,
    base_run_contract_sha: str,
) -> tuple[Path, str, Path, str]:
    inventory_path = output_dir / "prediction_inventory.json"
    _write_immutable_json(inventory_path, inventory)
    inventory_sha = primary.sha256_file(inventory_path)
    bound_payload = {
        "schema_version": "1.0.0",
        "status": "bound",
        "base_run_contract": str(base_run_contract_path),
        "base_run_contract_sha256": base_run_contract_sha,
        "prediction_inventory": str(inventory_path),
        "prediction_inventory_sha256": inventory_sha,
    }
    bound_path = output_dir / "bound_run_contract.json"
    _write_immutable_json(bound_path, bound_payload)
    bound_sha = primary.sha256_file(bound_path)
    return inventory_path, inventory_sha, bound_path, bound_sha


def _verify_bound_inventory_from_manifest(manifest: dict[str, Any]) -> None:
    for path_key, sha_key in (
        ("run_contract", "run_contract_sha256"),
        ("prediction_inventory", "prediction_inventory_sha256"),
        ("bound_run_contract", "bound_run_contract_sha256"),
    ):
        path = Path(str(manifest[path_key]))
        if not path.is_file() or primary.sha256_file(path) != str(manifest[sha_key]):
            raise ContractError(
                f"bound matched-run artifact changed before publish: {path}"
            )
    inventory = primary._load_json(Path(str(manifest["prediction_inventory"])))
    _verify_prediction_inventory_files(inventory)


def _exact_ot_seed(benchmark: str, split: str, space: str) -> int:
    # The primary metric helper owns this exact seed implementation.  Calling it
    # here (rather than reimplementing its hash) keeps the recorded audit value
    # tied to the computation actually performed.
    return int(benchmark_metrics._exact_ot_seed(benchmark, split, space))


def _normalized_weights(
    weights: np.ndarray | None, n_points: int
) -> tuple[np.ndarray, float]:
    if weights is None:
        return np.full(n_points, 1.0 / n_points, dtype=np.float64), float(n_points)
    values = np.asarray(weights, dtype=np.float64).reshape(-1)
    if values.shape != (n_points,):
        raise ContractError("prediction weights do not match prediction rows")
    total = float(values.sum())
    if not np.isfinite(values).all() or np.any(values < 0) or total <= 0:
        raise ContractError(
            "prediction weights must be finite, nonnegative and nonzero"
        )
    return values / total, total


def _indices_sha256(indices: np.ndarray) -> str:
    canonical = np.ascontiguousarray(np.asarray(indices, dtype="<i8"))
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _transformed_spaces(
    *,
    transform: FrozenBenchmarkTransform,
    predicted_state: np.ndarray,
    observed_state: np.ndarray,
    predicted_spatial: np.ndarray | None,
    observed_spatial: np.ndarray | None,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    predicted_state_values = transform.transform_state(predicted_state)
    observed_state_values = transform.transform_state(observed_state)
    if predicted_spatial is None:
        return {"state": (predicted_state_values, observed_state_values)}
    if observed_spatial is None:
        raise ContractError("observed spatial values are missing")
    predicted_spatial_values = transform.transform_spatial(predicted_spatial)
    observed_spatial_values = transform.transform_spatial(observed_spatial)
    return {
        "joint": (
            np.concatenate((predicted_state_values, predicted_spatial_values), axis=1),
            np.concatenate((observed_state_values, observed_spatial_values), axis=1),
        ),
        "state": (predicted_state_values, observed_state_values),
        "spatial": (predicted_spatial_values, observed_spatial_values),
    }


def _observed_exact_indices(n_points: int, cap: int, seed: int) -> np.ndarray:
    if n_points > cap:
        # This RNG is intentionally separate from predicted-particle sampling.
        return np.asarray(
            np.random.default_rng(seed).choice(n_points, size=cap, replace=False),
            dtype=np.int64,
        )
    return np.arange(n_points, dtype=np.int64)


def _matched_exact_metrics(
    *,
    predicted: np.ndarray,
    observed: np.ndarray,
    predicted_weights: np.ndarray | None,
    max_ot_points: int,
    seed: int,
    observed_indices: np.ndarray,
) -> dict[str, Any]:
    """Compute exact OT with independent predicted and observed RNG streams."""

    import ot
    from scipy.spatial.distance import cdist

    predicted = np.asarray(predicted, dtype=np.float64)
    observed = np.asarray(observed, dtype=np.float64)
    cap = int(max_ot_points)
    if cap <= 0:
        raise ContractError("max_ot_points must be positive")
    weights, _ = _normalized_weights(predicted_weights, predicted.shape[0])
    if predicted.shape[0] > cap:
        predicted_indices = np.asarray(
            np.random.default_rng(seed).choice(
                predicted.shape[0], size=cap, replace=True, p=weights
            ),
            dtype=np.int64,
        )
        predicted_values = predicted[predicted_indices]
        exact_predicted_weights = np.full(cap, 1.0 / cap, dtype=np.float64)
    else:
        predicted_indices = np.arange(predicted.shape[0], dtype=np.int64)
        predicted_values = predicted
        exact_predicted_weights = weights
    observed_indices = np.asarray(observed_indices, dtype=np.int64)
    if observed_indices.ndim != 1 or observed_indices.size == 0:
        raise ContractError("observed exact-OT indices must be a non-empty vector")
    if np.any(observed_indices < 0) or np.any(observed_indices >= observed.shape[0]):
        raise ContractError("observed exact-OT indices are out of bounds")
    observed_values = observed[observed_indices]
    exact_observed_weights = np.full(
        observed_values.shape[0], 1.0 / observed_values.shape[0], dtype=np.float64
    )
    distances = cdist(predicted_values, observed_values, metric="euclidean")
    w1 = float(
        ot.emd2(
            exact_predicted_weights,
            exact_observed_weights,
            distances,
            numItermax=int(1e7),
        )
    )
    w2_squared = float(
        ot.emd2(
            exact_predicted_weights,
            exact_observed_weights,
            distances**2,
            numItermax=int(1e7),
        )
    )
    if not np.isfinite(w1) or not np.isfinite(w2_squared):
        raise ContractError("matched exact OT returned a non-finite value")
    return {
        "exact_w1": w1,
        "exact_w2": float(np.sqrt(max(w2_squared, 0.0))),
        "exact_ot_predicted_points": int(predicted_values.shape[0]),
        "exact_ot_observed_points": int(observed_values.shape[0]),
        "exact_ot_predicted_indices_sha256": _indices_sha256(predicted_indices),
        "exact_ot_observed_indices_sha256": _indices_sha256(observed_indices),
    }


def _evaluate_predictions(
    *,
    root: dict[str, Any],
    input_manifest: Path,
    manifest_sha: str,
    cases: dict[tuple[str, int], dict[str, Any]],
    transform: FrozenBenchmarkTransform,
    anchor_audit: dict[str, Any],
    predictions: dict[str, dict[tuple[str, int], dict[str, Any]]],
    methods: list[str],
    targets: list[int],
    n_projections: int,
    projection_repeats: int,
    max_ot_points: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    benchmark = _dataset_label(root)
    rows: list[pd.DataFrame] = []
    observed_sampling: dict[str, dict[str, Any]] = {}
    for target in targets:
        seed_split = f"matched_t{target}"
        for method in methods:
            for track in TRACKS:
                prediction = predictions[track][(method, target)]
                case = cases[(track, target)]
                spatial = prediction["spatial"]
                metrics = evaluate_spatiotemporal_prediction(
                    transform=transform,
                    benchmark=benchmark,
                    split=seed_split,
                    method=method,
                    predicted_state=prediction["state"],
                    observed_state=case["truth_state"],
                    predicted_spatial=spatial,
                    observed_spatial=(
                        case["truth_spatial"] if spatial is not None else None
                    ),
                    predicted_weights=prediction["weights"],
                    n_projections=n_projections,
                    projection_repeats=projection_repeats,
                    # The helper's exact columns are placeholders here. They are
                    # overwritten immediately below by the matched, separate-RNG
                    # implementation; cap the discarded EMD to avoid doing the
                    # formal 800x800 solve twice.
                    max_ot_points=1,
                )
                transformed_spaces = _transformed_spaces(
                    transform=transform,
                    predicted_state=prediction["state"],
                    observed_state=case["truth_state"],
                    predicted_spatial=spatial,
                    observed_spatial=(
                        case["truth_spatial"] if spatial is not None else None
                    ),
                )
                for space, (
                    predicted_values,
                    observed_values,
                ) in transformed_spaces.items():
                    seed = _exact_ot_seed(benchmark, seed_split, space)
                    sampling_key = f"t{target}/{space}"
                    if sampling_key not in observed_sampling:
                        observed_indices = _observed_exact_indices(
                            observed_values.shape[0], max_ot_points, seed
                        )
                        observed_sampling[sampling_key] = {
                            "target": int(target),
                            "space": space,
                            "seed": seed,
                            "observed_rows": int(observed_values.shape[0]),
                            "selected_rows": int(observed_indices.shape[0]),
                            "observed_indices": observed_indices.astype(int).tolist(),
                            "observed_indices_sha256": _indices_sha256(
                                observed_indices
                            ),
                        }
                    record = observed_sampling[sampling_key]
                    if record["seed"] != seed or record["observed_rows"] != int(
                        observed_values.shape[0]
                    ):
                        raise ContractError(
                            f"inconsistent observed exact-OT population for {sampling_key}"
                        )
                    observed_indices = np.asarray(
                        record["observed_indices"], dtype=np.int64
                    )
                    exact = _matched_exact_metrics(
                        predicted=predicted_values,
                        observed=observed_values,
                        predicted_weights=prediction["weights"],
                        max_ot_points=max_ot_points,
                        seed=seed,
                        observed_indices=observed_indices,
                    )
                    mask = metrics["space"] == space
                    for column, value in exact.items():
                        metrics.loc[mask, column] = value
                    metrics.loc[mask, "exact_ot_seed"] = seed
                    metrics.loc[mask, "exact_ot_predicted_seed"] = seed
                    metrics.loc[mask, "exact_ot_observed_seed"] = seed
                    metrics.loc[mask, "exact_ot_observed_indices_key"] = sampling_key
                    metrics.loc[mask, "exact_ot_separate_rng"] = True
                    metrics.loc[mask, "exact_ot_matched"] = True
                for seed_column in (
                    "exact_ot_seed",
                    "exact_ot_predicted_seed",
                    "exact_ot_observed_seed",
                ):
                    metrics[seed_column] = metrics[seed_column].astype(np.int64)
                metrics["exact_ot_separate_rng"] = metrics[
                    "exact_ot_separate_rng"
                ].astype(bool)
                metrics["exact_ot_matched"] = metrics["exact_ot_matched"].astype(bool)
                summary = prediction["summary"]
                tmv = primary._tmv_columns(
                    summary,
                    prediction["weights"],
                    track=track,
                    target=target,
                    training_times=case["train_time"],
                    target_count=case["truth_state"].shape[0],
                )
                metrics.insert(0, "track", track)
                metrics.insert(1, "evaluation_scope", TRACK_SCOPES[track])
                metrics.insert(2, "is_in_sample", track == "full_data")
                metrics.insert(3, "target", target)
                metrics["training_split"] = case["split_id"]
                metrics["seed_pairing_split"] = seed_split
                metrics["output_scope"] = prediction["scope"]
                metrics["native_vs_adapter"] = prediction["native_vs_adapter"]
                metrics["source_time"] = primary._source_time(
                    summary, track, target, case["train_time"]
                )
                metrics["prediction_path"] = str(prediction["path"])
                metrics["prediction_sha256"] = prediction["sha"]
                metrics["prediction_summary"] = str(prediction["summary_path"])
                metrics["prediction_summary_sha256"] = prediction["summary_sha"]
                metrics["input_manifest"] = str(input_manifest)
                metrics["input_manifest_sha256"] = manifest_sha
                metrics["training_reference"] = str(case["training_path"])
                metrics["training_reference_sha256"] = case["training_sha"]
                metrics["source_roster"] = str(case["source_roster_path"])
                metrics["source_roster_sha256"] = case["source_roster_sha"]
                metrics["truth_reference"] = str(case["truth_path"])
                metrics["truth_reference_sha256"] = case["truth_sha"]
                metrics["truth_bundle_sha256"] = case["truth_signature"][
                    "bundle_sha256"
                ]
                metrics["transform_path"] = str(case["transform_path"])
                metrics["transform_sha256"] = case["transform_sha"]
                metrics["anchor_bundle_sha256"] = anchor_audit["anchor_bundle_sha256"]
                for column, value in tmv.items():
                    metrics[column] = value
                rows.append(metrics)
    result = pd.concat(rows, ignore_index=True)
    result = result.sort_values(
        ["target", "space", "method", "track", "projection_repeat"],
        kind="stable",
    ).reset_index(drop=True)
    sampling_audit = {
        "schema_version": "1.0.0",
        "status": "complete",
        "policy": "matched exact OT with separate predicted and observed RNG streams",
        "seed_key": "dataset + matched target + space",
        "observed_indices_shared_across_methods_and_tracks": True,
        "observed_sampling_independent_of_predicted_rng_consumption": True,
        "primary_helper_exact_columns_overwritten": True,
        "primary_helper_discarded_exact_cap": 1,
        "max_ot_points": int(max_ot_points),
        "index_dtype": "little-endian int64",
        "records": observed_sampling,
    }
    return result, sampling_audit


def _single_value(frame: pd.DataFrame, column: str) -> Any:
    values = frame[column].drop_duplicates()
    if len(values) != 1:
        raise ContractError(
            f"{column} must be constant across projection repeats, found {len(values)} values"
        )
    return values.iloc[0]


def _validate_pairs(metrics: pd.DataFrame, *, projection_repeats: int) -> None:
    for (target, space, repeat), shared in metrics.groupby(
        ["target", "space", "projection_repeat"], sort=True, dropna=False
    ):
        if shared["projection_seed"].nunique() != 1:
            raise ContractError(
                f"projection seed is not shared by all methods/tracks for "
                f"t{target}/{space}/repeat{repeat}"
            )
        if shared["projection_sha256"].nunique() != 1:
            raise ContractError(
                f"projection basis is not shared by all methods/tracks for "
                f"t{target}/{space}/repeat{repeat}"
            )
        if shared["exact_ot_seed"].nunique() != 1:
            raise ContractError(
                f"exact-OT seed is not shared by all methods/tracks for "
                f"t{target}/{space}/repeat{repeat}"
            )
        for column in (
            "exact_ot_observed_seed",
            "exact_ot_observed_indices_key",
            "exact_ot_observed_indices_sha256",
            "exact_ot_observed_points",
        ):
            if shared[column].nunique() != 1:
                raise ContractError(
                    f"matched {column} is not shared by all methods/tracks for "
                    f"t{target}/{space}/repeat{repeat}"
                )
        if (
            not shared["exact_ot_matched"].astype(bool).all()
            or not shared["exact_ot_separate_rng"].astype(bool).all()
        ):
            raise ContractError(
                f"exact OT is not marked matched/separate-RNG for "
                f"t{target}/{space}/repeat{repeat}"
            )
    grouped = metrics.groupby(["method", "target", "space"], sort=True, dropna=False)
    for (method, target, space), pair in grouped:
        track_sets = set(pair["track"])
        if track_sets != set(TRACKS):
            raise ContractError(
                f"unpaired feature space for {method}/t{target}/{space}: {track_sets}"
            )
        by_track = {
            track: pair[pair["track"] == track].sort_values("projection_repeat")
            for track in TRACKS
        }
        for track, frame in by_track.items():
            repeats = frame["projection_repeat"].astype(int).tolist()
            if repeats != list(range(projection_repeats)):
                raise ContractError(
                    f"invalid repeats for {method}/{track}/t{target}/{space}: {repeats}"
                )
        for column in (
            "projection_seed",
            "projection_sha256",
            "exact_ot_seed",
            "exact_ot_observed_seed",
            "exact_ot_observed_indices_key",
            "exact_ot_observed_indices_sha256",
        ):
            left = by_track["loto"][column].tolist()
            right = by_track["full_data"][column].tolist()
            if left != right:
                raise ContractError(
                    f"paired {column} differs for {method}/t{target}/{space}"
                )
        if set(by_track["loto"]["output_scope"]) != set(
            by_track["full_data"]["output_scope"]
        ):
            raise ContractError(
                f"output scope differs between tracks for {method}/t{target}/{space}"
            )
        if set(by_track["loto"]["native_vs_adapter"]) != set(
            by_track["full_data"]["native_vs_adapter"]
        ):
            raise ContractError(
                f"adapter status differs between tracks for {method}/t{target}/{space}"
            )
        if set(pair["transform_sha256"]) != {str(pair["transform_sha256"].iloc[0])}:
            raise ContractError("matched rows do not share one transform")
        if set(pair["truth_bundle_sha256"]) != {
            str(pair["truth_bundle_sha256"].iloc[0])
        }:
            raise ContractError(
                f"truth differs between tracks for {method}/t{target}/{space}"
            )


def _collapse_track(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    columns_constant = (
        "source_time",
        "exact_w1",
        "exact_w2",
        "n_predicted",
        "n_observed",
        "exact_ot_predicted_points",
        "exact_ot_observed_points",
        "exact_ot_seed",
        "exact_ot_observed_indices_key",
        "exact_ot_observed_indices_sha256",
        "exact_ot_separate_rng",
        "exact_ot_matched",
        "tmv_available",
        "tmv",
        "tmv_absolute",
        "predicted_mass",
        "observed_mass_relative",
        "output_scope",
        "native_vs_adapter",
    )
    for (track, method, target, space), frame in metrics.groupby(
        ["track", "method", "target", "space"], sort=True, dropna=False
    ):
        row: dict[str, Any] = {
            "track": track,
            "method": method,
            "target": int(target),
            "space": space,
            "evaluation_scope": TRACK_SCOPES[str(track)],
            "is_in_sample": bool(track == "full_data"),
            "projection_repeats": int(frame["projection_repeat"].nunique()),
            "sliced_w2_mean": float(frame["sliced_w2"].mean()),
            "sliced_w2_std": float(frame["sliced_w2"].std(ddof=0)),
        }
        for column in columns_constant:
            row[column] = _single_value(frame, column)
        rows.append(row)
    return pd.DataFrame(rows)


def _paired_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    collapsed = _collapse_track(metrics)
    loto = collapsed[collapsed["track"] == "loto"].drop(columns="track")
    full = collapsed[collapsed["track"] == "full_data"].drop(columns="track")
    paired = loto.merge(
        full,
        on=["method", "target", "space"],
        how="outer",
        validate="one_to_one",
        suffixes=("_loto", "_full_data"),
        indicator=True,
    )
    if set(paired["_merge"]) != {"both"}:
        missing = paired.loc[
            paired["_merge"] != "both", ["method", "target", "space", "_merge"]
        ]
        raise ContractError(
            f"paired summary is incomplete: {missing.to_dict('records')}"
        )
    paired = paired.drop(columns="_merge")
    paired["full_data_is_in_sample"] = True
    paired["comparison_type"] = "descriptive_paired_gap"
    paired["comparison"] = "loto_held_out_minus_full_data_in_sample"
    paired["exact_comparison_type"] = "matched_shared_observed_indices_separate_rng"
    for metric in ("sliced_w2_mean", "exact_w1", "exact_w2"):
        paired[f"{metric}_loto_minus_full_data"] = (
            paired[f"{metric}_loto"] - paired[f"{metric}_full_data"]
        )
    paired["tmv_directly_comparable"] = (
        paired["tmv_available_loto"].astype(bool)
        & paired["tmv_available_full_data"].astype(bool)
        & (paired["source_time_loto"] == paired["source_time_full_data"])
        & np.isclose(
            paired["observed_mass_relative_loto"].astype(float),
            paired["observed_mass_relative_full_data"].astype(float),
            equal_nan=False,
        )
    )
    paired["tmv_loto_minus_full_data"] = np.where(
        paired["tmv_directly_comparable"],
        paired["tmv_loto"] - paired["tmv_full_data"],
        np.nan,
    )
    forbidden = [
        column
        for column in paired.columns
        if "rank" in column.lower() or "overall" in column.lower()
    ]
    if forbidden:
        raise ContractError(f"forbidden cross-space summary columns: {forbidden}")
    return paired.sort_values(["target", "space", "method"], kind="stable").reset_index(
        drop=True
    )


def evaluate_matched(
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    input_manifest = args.input_manifest.expanduser().resolve()
    root = primary._load_json(input_manifest)
    manifest_sha = primary.sha256_file(input_manifest)
    targets = _common_targets(root, args.targets)
    anchors = _anchor_times(args.anchor_times)
    _validate_requested_config(args)
    output_dir = args.output_dir.expanduser().resolve()
    code_dependencies = _code_dependencies()
    software_versions = _software_versions()
    run_contract_payload = _run_contract_payload(
        args=args,
        input_manifest=input_manifest,
        manifest_sha=manifest_sha,
        targets=targets,
        anchors=anchors,
        code_dependencies=code_dependencies,
        software_versions=software_versions,
    )
    run_contract_path, run_contract_sha = _bind_run_contract(
        output_dir, run_contract_payload
    )
    cases, transform, transform_path, anchor_audit = _load_cases(
        root=root,
        input_manifest=input_manifest,
        targets=targets,
        anchor_times=anchors,
        output_dir=output_dir,
    )

    requested_methods = None if args.methods is None else set(args.methods)
    predictions = {
        "loto": _scan_predictions(
            predictions_root=args.loto_predictions_root.expanduser().resolve(),
            track="loto",
            targets=targets,
            cases=cases,
            manifest_sha=manifest_sha,
            methods=requested_methods,
            include_nonprimary=args.include_nonprimary,
        ),
        "full_data": _scan_predictions(
            predictions_root=args.full_data_predictions_root.expanduser().resolve(),
            track="full_data",
            targets=targets,
            cases=cases,
            manifest_sha=manifest_sha,
            methods=requested_methods,
            include_nonprimary=args.include_nonprimary,
        ),
    }
    methods = _method_grid(predictions, targets, args.methods)
    prediction_inventory = _prediction_inventory(
        predictions=predictions,
        methods=methods,
        targets=targets,
    )
    (
        prediction_inventory_path,
        prediction_inventory_sha,
        bound_run_contract_path,
        bound_run_contract_sha,
    ) = _bind_prediction_inventory(
        output_dir=output_dir,
        inventory=prediction_inventory,
        base_run_contract_path=run_contract_path,
        base_run_contract_sha=run_contract_sha,
    )
    # Catch any external mutation that raced the byte snapshot/inventory bind
    # before spending time on metrics. Formal calculations below use only the
    # already parsed in-memory arrays and summary dictionaries.
    _verify_prediction_inventory_files(prediction_inventory)
    metrics, exact_sampling_audit = _evaluate_predictions(
        root=root,
        input_manifest=input_manifest,
        manifest_sha=manifest_sha,
        cases=cases,
        transform=transform,
        anchor_audit=anchor_audit,
        predictions=predictions,
        methods=methods,
        targets=targets,
        n_projections=args.n_projections,
        projection_repeats=args.projection_repeats,
        max_ot_points=args.max_ot_points,
    )
    _validate_pairs(metrics, projection_repeats=args.projection_repeats)
    paired = _paired_summary(metrics)

    exact_sampling_path = output_dir / "matched_exact_ot_sampling.json"
    _write_immutable_json(exact_sampling_path, exact_sampling_audit)
    metrics_path = output_dir / "matched_metrics_long.csv"
    paired_path = output_dir / "matched_paired_summary.csv"
    _write_immutable_csv(metrics_path, metrics)
    _write_immutable_csv(paired_path, paired)
    manifest = {
        "schema_version": "1.0.0",
        "status": "complete",
        "design": "matched_loto_vs_full_data",
        "input_manifest": str(input_manifest),
        "input_manifest_sha256": manifest_sha,
        "run_contract": str(run_contract_path),
        "run_contract_sha256": run_contract_sha,
        "prediction_inventory": str(prediction_inventory_path),
        "prediction_inventory_sha256": prediction_inventory_sha,
        "bound_run_contract": str(bound_run_contract_path),
        "bound_run_contract_sha256": bound_run_contract_sha,
        "dataset": _dataset_label(root),
        "targets": targets,
        "target_policy": "intersection of configured LOTO and full-data targets",
        "tracks": {
            "loto": {
                "evaluation_scope": "held_out",
                "predictions_root": str(
                    args.loto_predictions_root.expanduser().resolve()
                ),
            },
            "full_data": {
                "evaluation_scope": "in_sample",
                "is_in_sample": True,
                "predictions_root": str(
                    args.full_data_predictions_root.expanduser().resolve()
                ),
            },
        },
        "methods": methods,
        "spaces": sorted(metrics["space"].unique().tolist()),
        "anchor_times": list(anchors),
        "anchor_rows": int(anchor_audit["anchor_rows"]),
        "anchor_bundle_sha256": anchor_audit["anchor_bundle_sha256"],
        "anchor_concatenated_c_order_sha256": anchor_audit[
            "anchor_concatenated_c_order_sha256"
        ],
        "anchor_concatenated_field_order": anchor_audit[
            "anchor_concatenated_field_order"
        ],
        "anchor_component_signatures": anchor_audit["anchor_component_signatures"],
        "all_participating_training_splits_byte_identical": True,
        "all_targets_semantically_verified": True,
        "semantic_split_audit": anchor_audit["semantic_split_audit"],
        "semantic_split_audit_sha256": anchor_audit["semantic_split_audit_sha256"],
        "common_anchor_audit": anchor_audit["audit_path"],
        "common_anchor_audit_sha256": anchor_audit["audit_sha256"],
        "common_transform": str(transform_path),
        "common_transform_sha256": primary.sha256_file(transform_path),
        "transform_fit_policy": (
            "fit once on designated anchor rows after proving state, spatial, time, "
            "and row_id bytes are identical in every participating training split"
        ),
        "seed_policy": {
            "pairing_split_template": "matched_t{target}",
            "projection_key": "dataset + matched target + space + repeat",
            "exact_ot_subsampling_key": "dataset + matched target + space",
            "same_seed_verified_between_tracks": True,
            "exact_ot_predicted_and_observed_rng_are_separate": True,
            "exact_ot_observed_indices_shared_across_methods_and_tracks": True,
        },
        "matched_exact_ot_sampling": str(exact_sampling_path),
        "matched_exact_ot_sampling_sha256": primary.sha256_file(exact_sampling_path),
        "n_projections": int(args.n_projections),
        "projection_repeats": int(args.projection_repeats),
        "max_ot_points": int(args.max_ot_points),
        "metrics_long_csv": str(metrics_path),
        "metrics_long_csv_sha256": primary.sha256_file(metrics_path),
        "paired_summary_csv": str(paired_path),
        "paired_summary_csv_sha256": primary.sha256_file(paired_path),
        "n_metrics_rows": int(len(metrics)),
        "n_paired_rows": int(len(paired)),
        "reporting_policy": {
            "pairing_unit": "method + target + feature space",
            "comparison_type": "descriptive_paired_gap",
            "full_data_is_in_sample": True,
            "cross_space_aggregation": False,
            "overall_score": False,
            "ranking": False,
            "statistical_inference": False,
            "model_training_seed_replicates": 1,
            "exact_gap_is_matched": True,
        },
        "interpretation_limit": (
            "Each method/track/split prediction is one model-training-seed realization; "
            "paired gaps are descriptive and do not estimate between-fit uncertainty."
        ),
        "scientific_limitations": [
            (
                "Full-data values are in-sample reconstruction references, so their "
                "difference from held-out LOTO is not an unbiased generalization estimate."
            ),
            (
                "The common transform uses the explicitly supplied anchors, including a "
                "future anchor when t0+t4 is selected; this is a transductive frozen-space "
                "diagnostic rather than a causal raw-gene forecast."
            ),
            (
                "Projection repeats quantify random-projection variation only; one fitted "
                "prediction realization per method/track/split provides no training-seed "
                "uncertainty."
            ),
            (
                "Exact OT fixes one shared observed index set per target/space. Predicted "
                "indices can still differ when prediction counts or weights differ, even "
                "though they use the same independent predicted RNG seed."
            ),
            (
                "State-only methods remain state-only and receive no fabricated spatial or "
                "joint score."
            ),
        ],
        "tmv_policy": (
            "reported only when the prediction declares native unnormalised mass; "
            "a paired TMV delta is emitted only for equal source-time denominators"
        ),
        "code_dependencies": code_dependencies,
        "software_versions": software_versions,
        "evaluator": code_dependencies["matched_evaluator"]["path"],
        "evaluator_sha256": code_dependencies["matched_evaluator"]["sha256"],
    }
    _verify_bound_inventory_from_manifest(manifest)
    return metrics, paired, manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--loto-predictions-root", type=Path, required=True)
    parser.add_argument("--full-data-predictions-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--targets", type=int, nargs="+")
    parser.add_argument(
        "--anchor-times",
        type=float,
        nargs="+",
        required=True,
        help="Times present with byte-identical rows in every participating split.",
    )
    parser.add_argument("--methods", nargs="+")
    parser.add_argument(
        "--include-nonprimary",
        action="store_true",
        help="Include explicitly sensitivity-only predictions when present in both tracks.",
    )
    parser.add_argument("--n-projections", type=int, default=1024)
    parser.add_argument("--projection-repeats", type=int, default=5)
    parser.add_argument("--max-ot-points", type=int, default=800)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        _, _, manifest = evaluate_matched(args)
        output_dir = args.output_dir.expanduser().resolve()
        # Close the final-publish TOCTOU window: inputs external to the output
        # directory must still match the single byte snapshot bound above.
        _verify_bound_inventory_from_manifest(manifest)
        _write_final_manifest(output_dir / "matched_evaluation_manifest.json", manifest)
    except (ContractError, FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
