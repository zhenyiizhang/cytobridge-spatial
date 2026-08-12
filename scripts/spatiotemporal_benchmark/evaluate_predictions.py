#!/usr/bin/env python3
"""Evaluate audited spatiotemporal predictions in frozen comparable spaces.

This entry point deliberately knows nothing about how a method was trained.  It
only accepts predictions whose summaries point back to the exact benchmark input
manifest and split-specific training reference.  LOTO and full-data results are
written separately and cannot be pooled by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from CytoBridge.tl.downstream.benchmark import (  # noqa: E402
    FrozenBenchmarkTransform,
    evaluate_spatiotemporal_prediction,
    fit_frozen_benchmark_transform,
)


TRACK_ALIASES = {
    "loto": "loto",
    "full_data": "full_data",
    "full-data": "full_data",
    "noholdout": "full_data",
    "no-holdout": "full_data",
}
COMPLETED_STATUSES = {"complete", "completed", "success"}
NON_NUMERIC_STATUSES = {
    "timeout",
    "oom",
    "failed",
    "not_available",
    "not_applicable",
}
STATUS_ALIASES = {
    "out_of_memory": "oom",
    "unavailable": "not_available",
    "n/a": "not_applicable",
    "na": "not_applicable",
}


class ContractError(ValueError):
    """Raised when an artifact cannot be tied to the frozen benchmark."""


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ContractError(f"{path} must contain a JSON object")
    return payload


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _resolve_path(value: str | Path, base: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _artifact(
    value: Any,
    *,
    base: Path,
    label: str,
    verify_sha: bool = True,
) -> tuple[Path, str]:
    if isinstance(value, str):
        path = _resolve_path(value, base)
        expected = None
    elif isinstance(value, dict) and ("relative_path" in value or "path" in value):
        raw = value.get("relative_path", value.get("path"))
        path = _resolve_path(str(raw), base)
        expected = str(value.get("sha256", "")).lower() or None
    else:
        raise ContractError(f"{label} is not an artifact record")
    if not path.is_file():
        raise ContractError(f"{label} is missing: {path}")
    observed = sha256_file(path)
    if verify_sha and expected is not None and observed != expected:
        raise ContractError(
            f"{label} SHA-256 mismatch: expected {expected}, observed {observed}"
        )
    return path, observed


def _normalize_track(value: Any) -> str:
    key = str(value).strip().lower().replace(" ", "_")
    try:
        return TRACK_ALIASES[key]
    except KeyError as exc:
        raise ContractError(f"unknown benchmark track/regime {value!r}") from exc


def _normalize_status(value: Any) -> str:
    status = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    status = STATUS_ALIASES.get(status, status)
    if status in COMPLETED_STATUSES:
        return "completed"
    if status not in NON_NUMERIC_STATUSES:
        allowed = sorted({"completed", *NON_NUMERIC_STATUSES})
        raise ContractError(
            f"unknown execution status {value!r}; expected one of {allowed}"
        )
    return status


def _load_status_table(
    path: Path | None, track: str
) -> dict[tuple[str, int], dict[str, Any]]:
    if path is None:
        return {}
    path = path.expanduser().resolve()
    try:
        table = pd.read_csv(path)
    except (OSError, pd.errors.ParserError) as exc:
        raise ContractError(f"cannot read status table {path}: {exc}") from exc
    required = {"method", "target", "status"}
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ContractError(f"status table is missing columns {missing}")
    if "track" in table:
        normalized_tracks = table["track"].map(_normalize_track)
        table = table.loc[normalized_tracks.eq(track)].copy()
    reason_column = next(
        (name for name in ("reason", "failure_reason", "message") if name in table),
        None,
    )
    result: dict[tuple[str, int], dict[str, Any]] = {}
    for row in table.itertuples(index=False):
        method = str(getattr(row, "method")).strip()
        if not method:
            raise ContractError("status table contains an empty method")
        raw_target = getattr(row, "target")
        target = int(raw_target)
        if float(raw_target) != target:
            raise ContractError(
                f"status target must be an integer, found {raw_target!r}"
            )
        key = (method, target)
        if key in result:
            raise ContractError(f"duplicate status row for {method}/t{target}")
        reason_value = getattr(row, reason_column) if reason_column else ""
        reason = "" if pd.isna(reason_value) else str(reason_value).strip()
        result[key] = {
            "method": method,
            "track": track,
            "target": target,
            "status": _normalize_status(getattr(row, "status")),
            "reason": reason,
        }
    return result


def _method_target_status(
    *,
    methods: list[str],
    targets: list[int],
    completed: set[tuple[str, int]],
    declared: dict[tuple[str, int], dict[str, Any]],
    track: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for method in methods:
        for target in targets:
            key = (method, target)
            record = declared.get(key)
            if key in completed:
                if record is not None and record["status"] != "completed":
                    raise ContractError(
                        f"{method}/t{target} has a prediction but status={record['status']}"
                    )
                status = "completed"
                reason = "" if record is None else record["reason"]
            elif record is None:
                missing.append(f"{method}/t{target}")
                continue
            elif record["status"] == "completed":
                raise ContractError(
                    f"{method}/t{target} is marked completed but has no prediction"
                )
            else:
                status = record["status"]
                reason = record["reason"]
            rows.append(
                {
                    "track": track,
                    "target": target,
                    "method": method,
                    "status": status,
                    "reason": reason,
                }
            )
    if missing:
        raise ContractError(
            "missing predictions without an explicit failure status: "
            + ", ".join(missing)
        )
    return pd.DataFrame.from_records(rows)


def _targets(root: dict[str, Any], track: str) -> list[int]:
    key = "loto_targets" if track == "loto" else "full_data_targets"
    default = (1, 2, 3) if track == "loto" else (1, 2, 3, 4)
    raw = root.get(key, default)
    try:
        values = [int(value) for value in raw]
    except (TypeError, ValueError) as exc:
        raise ContractError(f"manifest {key} must be a sequence of integers") from exc
    if not values or len(set(values)) != len(values):
        raise ContractError(f"manifest {key} must be non-empty and unique")
    return values


def _split_record(
    root: dict[str, Any], track: str, target: int
) -> tuple[str, dict[str, Any]]:
    split_id = f"loto_t{target}" if track == "loto" else "full_data"
    splits = root.get("splits")
    if not isinstance(splits, dict) or not isinstance(splits.get(split_id), dict):
        raise ContractError(f"input manifest is missing split {split_id}")
    return split_id, splits[split_id]


def _training_artifact(split: dict[str, Any]) -> Any:
    train = split.get("train")
    if isinstance(train, dict):
        for key in ("training_reference_npz", "reference_npz"):
            if key in train:
                return train[key]
    for key in ("training_reference_npz", "reference_npz"):
        if key in split:
            return split[key]
    raise ContractError("split is missing training_reference_npz")


def _source_roster_artifact(split: dict[str, Any]) -> Any:
    train = split.get("train")
    if isinstance(train, dict) and "source_roster_npz" in train:
        return train["source_roster_npz"]
    if "source_roster_npz" in split:
        return split["source_roster_npz"]
    raise ContractError("split is missing source_roster_npz")


def _truth_artifact(split: dict[str, Any], target: int) -> Any:
    by_time = split.get("truth_by_time_npz")
    if not isinstance(by_time, dict) or str(target) not in by_time:
        raise ContractError(f"split is missing truth_by_time_npz[{target!r}]")
    return by_time[str(target)]


def _matrix(archive: Any, key: str, *, columns: int | None = None) -> np.ndarray:
    if key not in archive:
        raise ContractError(f"NPZ is missing {key!r}")
    value = np.asarray(archive[key], dtype=np.float64)
    if value.ndim != 2 or value.shape[0] == 0:
        raise ContractError(f"NPZ {key!r} must be a non-empty 2D array")
    if columns is not None and value.shape[1] != columns:
        raise ContractError(
            f"NPZ {key!r} must have {columns} columns, found {value.shape[1]}"
        )
    if not np.isfinite(value).all():
        raise ContractError(f"NPZ {key!r} contains non-finite values")
    return value


def _load_reference(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        state = _matrix(archive, "state")
        spatial = _matrix(archive, "spatial")
        if state.shape[0] != spatial.shape[0]:
            raise ContractError(f"{path}: state/spatial row counts differ")
        if "time" not in archive:
            raise ContractError(f"{path}: missing time")
        time = np.asarray(archive["time"], dtype=np.float64).reshape(-1)
        if time.shape != (state.shape[0],) or not np.isfinite(time).all():
            raise ContractError(f"{path}: invalid time array")
    return state, spatial, time


def _load_truth(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        state = _matrix(archive, "state")
        spatial = _matrix(archive, "spatial")
    if state.shape[0] != spatial.shape[0]:
        raise ContractError(f"{path}: truth state/spatial row counts differ")
    return state, spatial


def _find_summary(prediction_path: Path) -> tuple[Path, dict[str, Any]]:
    candidates = (
        prediction_path.with_suffix(".summary.json"),
        prediction_path.parent / "summary.json",
        prediction_path.parent / "run_manifest.json",
    )
    for path in candidates:
        if path.is_file():
            return path, _load_json(path)
    raise ContractError(f"prediction has no summary JSON: {prediction_path}")


def _summary_target(summary: dict[str, Any]) -> int:
    for key in ("target_time", "target", "holdout_time"):
        if key in summary and summary[key] is not None:
            try:
                return int(float(summary[key]))
            except (TypeError, ValueError) as exc:
                raise ContractError(
                    f"summary has invalid {key}={summary[key]!r}"
                ) from exc
    raise ContractError("summary has no target_time/target/holdout_time")


def _summary_track(summary: dict[str, Any]) -> str:
    for key in ("track", "regime", "evaluation_mode", "mode"):
        if key in summary and summary[key] is not None:
            return _normalize_track(summary[key])
    raise ContractError("summary has no track/regime")


def _summary_method(summary: dict[str, Any]) -> str:
    method = str(summary.get("method", "")).strip()
    if not method:
        raise ContractError("summary has no non-empty method")
    return method


def _summary_scope(summary: dict[str, Any], has_spatial: bool) -> str:
    scope = str(summary.get("output_scope", "")).strip().lower()
    if not scope:
        return "joint_unspecified" if has_spatial else "native_state"
    state_only = {
        "state",
        "state_only",
        "native_state",
        "native-state",
        "hybrid_state",
    }
    if scope in state_only and has_spatial:
        raise ContractError("state-only summary must not export spatial predictions")
    if scope not in state_only and not has_spatial:
        raise ContractError(f"output_scope={scope!r} requires a spatial prediction")
    return scope


def _summary_sha(summary: dict[str, Any], keys: Iterable[str]) -> str | None:
    for key in keys:
        value = summary.get(key)
        if value is not None:
            return str(value).lower()
    nested = summary.get("input")
    if isinstance(nested, dict):
        for key in keys:
            value = nested.get(key)
            if value is not None:
                return str(value).lower()
    return None


def _summary_prediction_sha(summary: dict[str, Any]) -> str | None:
    declared = _summary_sha(
        summary,
        ("prediction_npz_sha256", "prediction_sha256"),
    )
    if declared is not None:
        return declared
    prediction = summary.get("prediction")
    if isinstance(prediction, dict):
        value = prediction.get("sha256")
        if value is not None:
            return str(value).lower()
    return None


def _primary_eligible(summary: dict[str, Any]) -> bool:
    if "primary_benchmark_eligible" in summary:
        return bool(summary["primary_benchmark_eligible"])
    representation = str(summary.get("representation", "")).strip().lower()
    return representation != "native_gene_sensitivity"


def _verify_prediction_provenance(
    summary: dict[str, Any],
    *,
    manifest_sha: str,
    training_sha: str,
    source_roster_sha: str,
) -> None:
    status = str(summary.get("status", "")).strip().lower()
    if status not in {"complete", "completed", "success"}:
        raise ContractError(f"prediction summary status is not complete: {status!r}")
    declared_manifest = _summary_sha(
        summary,
        ("input_manifest_sha256", "manifest_sha256", "contract_sha256"),
    )
    if declared_manifest is None:
        raise ContractError("prediction summary does not record input manifest SHA-256")
    if declared_manifest != manifest_sha:
        raise ContractError("prediction summary input-manifest SHA-256 does not match")
    declared_training = _summary_sha(
        summary,
        (
            "training_reference_sha256",
            "training_reference_npz_sha256",
            "training_reference_actual_sha256",
        ),
    )
    if declared_training is None:
        raise ContractError(
            "prediction summary does not record training-reference SHA-256"
        )
    if declared_training != training_sha:
        raise ContractError(
            "prediction summary training-reference SHA-256 does not match"
        )
    declared_roster = _summary_sha(summary, ("source_roster_sha256",))
    if declared_roster is None:
        raise ContractError("prediction summary does not record source-roster SHA-256")
    if declared_roster != source_roster_sha:
        raise ContractError("prediction summary source-roster SHA-256 does not match")


def _prediction_arrays(
    path: Path,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    with np.load(path, allow_pickle=False) as archive:
        state = _matrix(archive, "state")
        spatial = _matrix(archive, "spatial") if "spatial" in archive else None
        if spatial is not None and spatial.shape[0] != state.shape[0]:
            raise ContractError(f"{path}: prediction state/spatial row counts differ")
        weights = None
        if "weights" in archive:
            weights = np.asarray(archive["weights"], dtype=np.float64).reshape(-1)
            if weights.shape != (state.shape[0],):
                raise ContractError(
                    f"{path}: weights length does not match predictions"
                )
            if (
                not np.isfinite(weights).all()
                or np.any(weights < 0)
                or weights.sum() <= 0
            ):
                raise ContractError(
                    f"{path}: weights must be finite, nonnegative and nonzero"
                )
    return state, spatial, weights


def _source_time(
    summary: dict[str, Any], track: str, target: int, times: np.ndarray
) -> int:
    if track == "full_data":
        return int(summary.get("source_time", np.min(times)))
    for key in ("source_time", "previous_anchor_time", "start_time"):
        if key in summary and summary[key] is not None:
            return int(float(summary[key]))
    previous = sorted(int(value) for value in np.unique(times) if value < target)
    if not previous:
        raise ContractError(f"cannot infer a source time before target t{target}")
    return previous[-1]


def _tmv_columns(
    summary: dict[str, Any],
    weights: np.ndarray | None,
    *,
    track: str,
    target: int,
    training_times: np.ndarray,
    target_count: int,
) -> dict[str, Any]:
    native_mass = bool(
        summary.get("native_mass", False) or summary.get("native_growth", False)
    )
    weights_are_unnormalised = bool(summary.get("weights_are_unnormalised", False))
    if not native_mass or not weights_are_unnormalised:
        return {
            "tmv_available": False,
            "tmv": np.nan,
            "tmv_absolute": np.nan,
            "predicted_mass": np.nan,
            "observed_mass_relative": np.nan,
        }
    if weights is None:
        raise ContractError("native-mass prediction must export unnormalised weights")
    source_time = _source_time(summary, track, target, training_times)
    source_count = int(np.count_nonzero(np.isclose(training_times, source_time)))
    if source_count <= 0:
        raise ContractError(
            f"source time t{source_time} is absent from training reference"
        )
    predicted_mass = float(weights.sum())
    observed_mass_relative = float(target_count / source_count)
    tmv_absolute = abs(predicted_mass - observed_mass_relative)
    return {
        "tmv_available": True,
        "tmv": float(tmv_absolute / observed_mass_relative),
        "tmv_absolute": float(tmv_absolute),
        "predicted_mass": predicted_mass,
        "observed_mass_relative": observed_mass_relative,
    }


def evaluate_track(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    input_manifest = args.input_manifest.expanduser().resolve()
    root = _load_json(input_manifest)
    manifest_sha = sha256_file(input_manifest)
    track = _normalize_track(args.track)
    allowed_targets = _targets(root, track)
    requested_targets = allowed_targets if args.targets is None else args.targets
    invalid = sorted(set(requested_targets).difference(allowed_targets))
    if invalid:
        raise ContractError(f"targets {invalid} are not valid for track {track}")
    declared_status = _load_status_table(args.status_table, track)

    candidates = sorted(
        args.predictions_root.expanduser().resolve().rglob("prediction.npz")
    )
    if not candidates and not declared_status:
        raise ContractError(f"no prediction.npz found below {args.predictions_root}")

    cases: dict[int, dict[str, Any]] = {}
    transform_dir = args.output_dir.expanduser().resolve() / "transforms"
    for target in requested_targets:
        split_id, split = _split_record(root, track, target)
        training_path, training_sha = _artifact(
            _training_artifact(split),
            base=input_manifest.parent,
            label=f"{split_id}/training_reference",
        )
        source_roster_path, source_roster_sha = _artifact(
            _source_roster_artifact(split),
            base=input_manifest.parent,
            label=f"{split_id}/source_roster",
        )
        truth_path, truth_sha = _artifact(
            _truth_artifact(split, target),
            base=input_manifest.parent,
            label=f"{split_id}/truth_t{target}",
        )
        train_state, train_spatial, train_time = _load_reference(training_path)
        truth_state, truth_spatial = _load_truth(truth_path)
        transform = fit_frozen_benchmark_transform(train_state, train_spatial)
        transform_path = transform_dir / f"{split_id}.json"
        transform_path.parent.mkdir(parents=True, exist_ok=True)
        if (
            transform_path.exists()
            and transform_path.read_text(encoding="utf-8").strip()
            != transform.to_json()
        ):
            raise ContractError(
                f"refusing to replace a different frozen transform: {transform_path}"
            )
        transform_path.write_text(transform.to_json() + "\n", encoding="utf-8")
        cases[target] = {
            "split_id": split_id,
            "training_path": training_path,
            "training_sha": training_sha,
            "source_roster_path": source_roster_path,
            "source_roster_sha": source_roster_sha,
            "truth_path": truth_path,
            "truth_sha": truth_sha,
            "train_time": train_time,
            "truth_state": truth_state,
            "truth_spatial": truth_spatial,
            "transform": transform,
            "transform_path": transform_path,
            "transform_sha": sha256_file(transform_path),
        }

    rows: list[pd.DataFrame] = []
    seen: set[tuple[str, int]] = set()
    method_filter = None if args.methods is None else set(args.methods)
    for prediction_path in candidates:
        summary_path, summary = _find_summary(prediction_path)
        prediction_track = _summary_track(summary)
        if prediction_track != track:
            continue
        target = _summary_target(summary)
        if target not in cases:
            continue
        method = _summary_method(summary)
        if method_filter is not None and method not in method_filter:
            continue
        if not args.include_nonprimary and not _primary_eligible(summary):
            continue
        key = (method, target)
        if key in seen:
            raise ContractError(f"duplicate prediction for {method}/{track}/t{target}")
        seen.add(key)
        case = cases[target]
        _verify_prediction_provenance(
            summary,
            manifest_sha=manifest_sha,
            training_sha=case["training_sha"],
            source_roster_sha=case["source_roster_sha"],
        )
        prediction_sha = sha256_file(prediction_path)
        declared_prediction_sha = _summary_prediction_sha(summary)
        if declared_prediction_sha is None:
            raise ContractError(
                f"prediction summary does not record prediction SHA-256: {summary_path}"
            )
        if declared_prediction_sha != prediction_sha:
            raise ContractError(
                f"prediction SHA-256 does not match its summary: {prediction_path}"
            )
        state, spatial, weights = _prediction_arrays(prediction_path)
        scope = _summary_scope(summary, spatial is not None)
        observed_spatial = case["truth_spatial"] if spatial is not None else None
        metrics = evaluate_spatiotemporal_prediction(
            transform=case["transform"],
            benchmark=str(
                root.get(
                    "dataset_id",
                    root.get("dataset", root.get("dataset_name", "spatiotemporal")),
                )
            ),
            split=case["split_id"],
            method=method,
            predicted_state=state,
            observed_state=case["truth_state"],
            predicted_spatial=spatial,
            observed_spatial=observed_spatial,
            predicted_weights=weights,
            n_projections=args.n_projections,
            projection_repeats=args.projection_repeats,
            max_ot_points=args.max_ot_points,
        )
        tmv = _tmv_columns(
            summary,
            weights,
            track=track,
            target=target,
            training_times=case["train_time"],
            target_count=case["truth_state"].shape[0],
        )
        metrics.insert(0, "track", track)
        metrics.insert(1, "target", target)
        metrics["output_scope"] = scope
        metrics["native_vs_adapter"] = str(
            summary.get("native_vs_adapter", summary.get("adapter_type", scope))
        )
        metrics["source_time"] = _source_time(
            summary, track, target, case["train_time"]
        )
        metrics["prediction_path"] = str(prediction_path)
        metrics["prediction_sha256"] = prediction_sha
        metrics["prediction_summary"] = str(summary_path)
        metrics["prediction_summary_sha256"] = sha256_file(summary_path)
        metrics["input_manifest"] = str(input_manifest)
        metrics["input_manifest_sha256"] = manifest_sha
        metrics["training_reference"] = str(case["training_path"])
        metrics["training_reference_sha256"] = case["training_sha"]
        metrics["source_roster"] = str(case["source_roster_path"])
        metrics["source_roster_sha256"] = case["source_roster_sha"]
        metrics["truth_reference"] = str(case["truth_path"])
        metrics["truth_reference_sha256"] = case["truth_sha"]
        metrics["transform_path"] = str(case["transform_path"])
        metrics["transform_sha256"] = case["transform_sha"]
        for column, value in tmv.items():
            metrics[column] = value
        rows.append(metrics)

    if args.methods is not None:
        methods = list(args.methods)
    elif declared_status:
        methods = sorted(
            {method for method, _ in declared_status} | {method for method, _ in seen}
        )
    else:
        methods = sorted({method for method, _ in seen})
    if not methods or len(methods) != len(set(methods)):
        raise ContractError("--methods must contain unique non-empty names")
    status = _method_target_status(
        methods=methods,
        targets=list(requested_targets),
        completed=seen,
        declared=declared_status,
        track=track,
    )
    if rows:
        result = pd.concat(rows, ignore_index=True)
        result = result.sort_values(
            ["target", "space", "method", "projection_repeat"], kind="stable"
        ).reset_index(drop=True)
    else:
        result = pd.DataFrame(
            columns=[
                "track",
                "target",
                "source_time",
                "method",
                "space",
                "output_scope",
                "native_vs_adapter",
                "projection_repeat",
                "sliced_w2",
                "exact_w1",
                "exact_w2",
                "tmv_available",
                "tmv",
                "tmv_absolute",
                "predicted_mass",
                "observed_mass_relative",
                "n_predicted",
                "n_observed",
            ]
        )
    return result, status


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--predictions-root", type=Path, required=True)
    parser.add_argument("--track", choices=sorted(TRACK_ALIASES), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--status-table",
        type=Path,
        help=(
            "CSV with method,target,status[,track,reason]. Missing predictions are "
            "reported as NA only when status is timeout, oom, failed, "
            "not_available, or not_applicable."
        ),
    )
    parser.add_argument("--targets", type=int, nargs="+")
    parser.add_argument("--methods", nargs="+")
    parser.add_argument(
        "--include-nonprimary",
        action="store_true",
        help="Include explicitly sensitivity-only predictions in a separate evaluation.",
    )
    parser.add_argument("--n-projections", type=int, default=1024)
    parser.add_argument("--projection-repeats", type=int, default=5)
    parser.add_argument("--max-ot-points", type=int, default=800)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        metrics, method_status = evaluate_track(args)
        track = _normalize_track(args.track)
        output_dir = args.output_dir.expanduser().resolve()
        output_csv = output_dir / f"{track}_metrics_long.csv"
        status_csv = output_dir / f"{track}_method_target_status.csv"
        _atomic_csv(output_csv, metrics)
        _atomic_csv(status_csv, method_status)
        manifest = {
            "schema_version": "1.0.0",
            "status": "complete",
            "track": track,
            "input_manifest": str(args.input_manifest.expanduser().resolve()),
            "input_manifest_sha256": sha256_file(
                args.input_manifest.expanduser().resolve()
            ),
            "predictions_root": str(args.predictions_root.expanduser().resolve()),
            "n_projections": int(args.n_projections),
            "projection_repeats": int(args.projection_repeats),
            "max_ot_points": int(args.max_ot_points),
            "methods": method_status["method"].drop_duplicates().tolist(),
            "completed_methods": sorted(metrics["method"].unique().tolist()),
            "targets": sorted(int(value) for value in method_status["target"].unique()),
            "spaces": sorted(metrics["space"].unique().tolist()),
            "method_target_status": method_status.to_dict(orient="records"),
            "method_target_status_csv": str(status_csv),
            "status_table_source": (
                None
                if args.status_table is None
                else str(args.status_table.expanduser().resolve())
            ),
            "metrics_long_csv": str(output_csv),
            "metrics_long_csv_sha256": sha256_file(output_csv),
            "n_rows": int(len(metrics)),
        }
        _atomic_json(output_dir / f"{track}_evaluation_manifest.json", manifest)
    except (ContractError, FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
