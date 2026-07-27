#!/usr/bin/env python3
"""Supplemental, method-independent shape analysis for a matched benchmark.

The primary benchmark deliberately mixes location and shape.  This opt-in
reviewer analysis reports the centroid error separately and then recomputes
distribution distances after centering each prediction and truth distribution
at its own weighted centroid.  It consumes a completed
``matched_evaluation_manifest.json`` and never edits or replaces primary
benchmark outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
for search_path in (SCRIPT_DIR, REPO_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

import evaluate_matched_tracks as matched  # noqa: E402
import evaluate_predictions as primary  # noqa: E402
from CytoBridge.tl.downstream import benchmark as benchmark_metrics  # noqa: E402
from CytoBridge.tl.downstream.benchmark import FrozenBenchmarkTransform  # noqa: E402


ContractError = primary.ContractError
TRACK_ORDER = ("loto", "full_data")
SPACE_ORDER = ("joint", "state", "spatial")
TRACK_METADATA = {
    "loto": {
        "evaluation_scope": "held_out",
        "comparison_role": "loto_transductive_interpolation",
        "is_in_sample": False,
        "is_oracle_control": False,
    },
    "full_data": {
        "evaluation_scope": "in_sample",
        "comparison_role": "full_data_in_sample_oracle_control",
        "is_in_sample": True,
        "is_oracle_control": True,
    },
}
SHAPE_METRICS = (
    "centroid_error",
    "centered_sliced_w2",
    "centered_exact_w1",
    "centered_exact_w2",
    "covariance_bures_distance",
)
METRIC_LABELS = {
    "centroid_error": "Centroid error (location)",
    "centered_sliced_w2": "Centered sliced W2 (shape)",
    "centered_exact_w1": "Centered exact W1 (shape)",
    "centered_exact_w2": "Centered exact W2 (shape)",
    "covariance_bures_distance": "Covariance Bures distance (shape)",
}


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ContractError(f"{name} must be a positive integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{name} must be a positive integer") from exc
    if parsed != value or parsed <= 0:
        raise ContractError(f"{name} must be a positive integer")
    return parsed


def _finite_matrix(values: np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ContractError(f"{name} must be a non-empty 2D matrix")
    if not np.isfinite(matrix).all():
        raise ContractError(f"{name} contains non-finite values")
    return matrix


def _collapse_weighted_support(
    values: np.ndarray,
    weights: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Collapse exact duplicate rows while preserving their total probability.

    Zero-mass rows are removed.  Positive duplicate rows are represented once
    and receive the sum of their normalized row masses.  This changes only the
    representation of the empirical measure, not the measure itself.
    """

    matrix = _finite_matrix(values, name="support")
    normalized, raw_sum = matched._normalized_weights(weights, matrix.shape[0])
    positive = normalized > 0
    positive_values = matrix[positive]
    positive_weights = normalized[positive]
    if positive_values.shape[0] == 0:
        raise ContractError("weighted support has no positive-mass rows")
    unique_values, inverse = np.unique(
        positive_values, axis=0, return_inverse=True
    )
    unique_weights = np.bincount(
        inverse,
        weights=positive_weights,
        minlength=unique_values.shape[0],
    ).astype(np.float64)
    unique_weights /= float(unique_weights.sum())
    details = {
        "n_input_rows": int(matrix.shape[0]),
        "n_positive_weight_rows": int(positive_values.shape[0]),
        "n_zero_weight_rows": int(matrix.shape[0] - positive_values.shape[0]),
        "n_unique_support": int(unique_values.shape[0]),
        "n_duplicate_positive_rows_collapsed": int(
            positive_values.shape[0] - unique_values.shape[0]
        ),
        "raw_weight_sum": float(raw_sum),
        "effective_support_size": float(1.0 / np.sum(unique_weights**2)),
    }
    return unique_values, unique_weights, details


def _weighted_centroid_and_covariance(
    values: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    matrix = _finite_matrix(values, name="weighted values")
    mass = np.asarray(weights, dtype=np.float64).reshape(-1)
    if mass.shape != (matrix.shape[0],):
        raise ContractError("weights do not match weighted values")
    if not np.isfinite(mass).all() or np.any(mass < 0) or mass.sum() <= 0:
        raise ContractError("weights must be finite, nonnegative and nonzero")
    mass = mass / float(mass.sum())
    centroid = mass @ matrix
    centered = matrix - centroid
    covariance = (centered * mass[:, None]).T @ centered
    covariance = 0.5 * (covariance + covariance.T)
    return centroid, covariance


def _psd_sqrt(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ContractError("PSD square root requires a square matrix")
    values = 0.5 * (values + values.T)
    eigenvalues, eigenvectors = np.linalg.eigh(values)
    scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    tolerance = 1e-10 * scale
    if float(np.min(eigenvalues)) < -tolerance:
        raise ContractError("covariance calculation produced a non-PSD matrix")
    clipped = np.clip(eigenvalues, 0.0, None)
    return (eigenvectors * np.sqrt(clipped)) @ eigenvectors.T


def covariance_bures_distance(
    predicted_covariance: np.ndarray,
    observed_covariance: np.ndarray,
) -> float:
    """Return the Bures distance between two population covariance matrices."""

    predicted = np.asarray(predicted_covariance, dtype=np.float64)
    observed = np.asarray(observed_covariance, dtype=np.float64)
    if predicted.shape != observed.shape:
        raise ContractError("Bures covariance matrices must have the same shape")
    # The eigendecomposition formula can lose several ulps when the
    # covariances are identical but rank deficient.  That cancellation may
    # otherwise turn the theoretical zero into a small negative number.
    scale = max(
        1.0,
        float(np.trace(predicted)),
        float(np.trace(observed)),
    )
    if np.allclose(predicted, observed, rtol=1e-12, atol=1e-12 * scale):
        return 0.0
    observed_sqrt = _psd_sqrt(observed)
    middle = observed_sqrt @ predicted @ observed_sqrt
    middle = 0.5 * (middle + middle.T)
    middle_eigenvalues = np.linalg.eigvalsh(middle)
    tolerance = 1e-9 * scale
    if float(np.min(middle_eigenvalues)) < -tolerance:
        raise ContractError("Bures middle matrix is not positive semidefinite")
    trace_middle_sqrt = float(
        np.sqrt(np.clip(middle_eigenvalues, 0.0, None)).sum()
    )
    squared = float(
        np.trace(predicted) + np.trace(observed) - 2.0 * trace_middle_sqrt
    )
    if squared < -tolerance:
        raise ContractError("Bures squared distance is materially negative")
    return float(np.sqrt(max(squared, 0.0)))


def _array_sha256(values: np.ndarray) -> str:
    matrix = np.ascontiguousarray(np.asarray(values, dtype="<f8"))
    shape = np.ascontiguousarray(np.asarray(matrix.shape, dtype="<i8"))
    digest = hashlib.sha256()
    digest.update(shape.tobytes(order="C"))
    digest.update(matrix.tobytes(order="C"))
    return digest.hexdigest()


def _space_metrics(
    *,
    predicted: np.ndarray,
    observed: np.ndarray,
    predicted_weights: np.ndarray | None,
    benchmark: str,
    seed_split: str,
    space: str,
    n_projections: int,
    projection_repeats: int,
    max_ot_points: int,
    observed_indices: np.ndarray,
) -> dict[str, Any]:
    """Compute location and centered-shape metrics in one frozen space."""

    predicted_values = _finite_matrix(predicted, name="predicted values")
    observed_values = _finite_matrix(observed, name="observed values")
    if predicted_values.shape[1] != observed_values.shape[1]:
        raise ContractError("predicted and observed dimensions differ")

    support, support_weights, support_audit = _collapse_weighted_support(
        predicted_values, predicted_weights
    )
    observed_weights = np.full(
        observed_values.shape[0],
        1.0 / observed_values.shape[0],
        dtype=np.float64,
    )
    predicted_centroid, predicted_covariance = _weighted_centroid_and_covariance(
        support, support_weights
    )
    observed_centroid, observed_covariance = _weighted_centroid_and_covariance(
        observed_values, observed_weights
    )
    centered_predicted = support - predicted_centroid
    centered_observed = observed_values - observed_centroid

    exact_seed = matched._exact_ot_seed(benchmark, seed_split, space)
    exact = matched._matched_exact_metrics(
        predicted=centered_predicted,
        observed=centered_observed,
        predicted_weights=support_weights,
        max_ot_points=max_ot_points,
        seed=exact_seed,
        observed_indices=observed_indices,
    )

    sliced_values: list[float] = []
    projection_seeds: list[int] = []
    projection_hashes: list[str] = []
    for repeat in range(projection_repeats):
        projection_seed = benchmark_metrics.benchmark_projection_seed(
            benchmark, seed_split, space, repeat
        )
        sliced, projection_hash = benchmark_metrics._sliced_w2(
            centered_predicted,
            centered_observed,
            predicted_weights=support_weights,
            n_projections=n_projections,
            seed=projection_seed,
        )
        sliced_values.append(float(sliced))
        projection_seeds.append(int(projection_seed))
        projection_hashes.append(str(projection_hash))

    observed_unique = np.unique(observed_values, axis=0).shape[0]
    result: dict[str, Any] = {
        "dimensions": int(predicted_values.shape[1]),
        "centroid_error": float(
            np.linalg.norm(predicted_centroid - observed_centroid)
        ),
        "centered_sliced_w2": float(np.mean(sliced_values)),
        "centered_sliced_w2_projection_sd": float(
            np.std(sliced_values, ddof=1) if len(sliced_values) > 1 else 0.0
        ),
        "centered_exact_w1": float(exact["exact_w1"]),
        "centered_exact_w2": float(exact["exact_w2"]),
        "covariance_bures_distance": covariance_bures_distance(
            predicted_covariance, observed_covariance
        ),
        "n_observed_rows": int(observed_values.shape[0]),
        "n_observed_unique_support": int(observed_unique),
        "n_observed_duplicate_rows": int(
            observed_values.shape[0] - observed_unique
        ),
        "n_projections": int(n_projections),
        "projection_repeats": int(projection_repeats),
        "projection_seeds_json": json.dumps(projection_seeds),
        "projection_hashes_json": json.dumps(projection_hashes),
        "exact_ot_seed": int(exact_seed),
        "exact_ot_max_points": int(max_ot_points),
        "exact_ot_predicted_points": int(exact["exact_ot_predicted_points"]),
        "exact_ot_observed_points": int(exact["exact_ot_observed_points"]),
        "exact_ot_predicted_indices_sha256": str(
            exact["exact_ot_predicted_indices_sha256"]
        ),
        "exact_ot_observed_indices_sha256": str(
            exact["exact_ot_observed_indices_sha256"]
        ),
        "exact_ot_is_full_support": bool(
            support.shape[0] <= max_ot_points
            and observed_values.shape[0] <= max_ot_points
        ),
        "metric_weight_normalized": True,
        "duplicate_support_policy": (
            "exact duplicate predicted rows collapsed per transformed space; "
            "normalized probability masses summed"
        ),
    }
    result.update(
        {
            "n_predicted_rows": support_audit["n_input_rows"],
            "n_predicted_positive_weight_rows": support_audit[
                "n_positive_weight_rows"
            ],
            "n_predicted_zero_weight_rows": support_audit["n_zero_weight_rows"],
            "n_predicted_unique_support": support_audit["n_unique_support"],
            "n_predicted_duplicate_rows_collapsed": support_audit[
                "n_duplicate_positive_rows_collapsed"
            ],
            "prediction_weight_sum_raw": support_audit["raw_weight_sum"],
            "predicted_effective_support_size": support_audit[
                "effective_support_size"
            ],
        }
    )
    return result


def _resolve_bound_path(
    payload: dict[str, Any],
    path_key: str,
    sha_key: str,
    *,
    base: Path,
) -> tuple[Path, str]:
    raw = Path(str(payload.get(path_key, ""))).expanduser()
    path = raw.resolve() if raw.is_absolute() else (base / raw).resolve()
    expected = str(payload.get(sha_key, "")).lower()
    if not expected:
        raise ContractError(f"manifest has no {sha_key}")
    if not path.is_file():
        raise ContractError(f"manifest-bound artifact is missing: {path}")
    observed = primary.sha256_file(path)
    if observed != expected:
        raise ContractError(
            f"{path_key} SHA-256 mismatch: expected {expected}, observed {observed}"
        )
    return path, observed


def _select_values(
    available: Iterable[Any],
    requested: list[Any] | None,
    *,
    label: str,
) -> list[Any]:
    available_values = list(available)
    if requested is None:
        return available_values
    if not requested or len(set(requested)) != len(requested):
        raise ContractError(f"--{label} must be non-empty and unique")
    invalid = [value for value in requested if value not in available_values]
    if invalid:
        raise ContractError(
            f"--{label} contains values absent from matched manifest: {invalid}"
        )
    return list(requested)


def _load_context(
    matched_manifest_path: Path,
    *,
    requested_methods: list[str] | None,
    requested_tracks: list[str] | None,
    requested_targets: list[int] | None,
) -> dict[str, Any]:
    manifest_path = matched_manifest_path.expanduser().resolve()
    manifest = primary._load_json(manifest_path)
    if (
        str(manifest.get("status", "")).casefold() != "complete"
        or manifest.get("design") != "matched_loto_vs_full_data"
    ):
        raise ContractError(
            "--matched-manifest must be a completed matched_loto_vs_full_data run"
        )
    matched._verify_bound_inventory_from_manifest(manifest)

    input_manifest, input_manifest_sha = _resolve_bound_path(
        manifest,
        "input_manifest",
        "input_manifest_sha256",
        base=manifest_path.parent,
    )
    inventory_path, inventory_sha = _resolve_bound_path(
        manifest,
        "prediction_inventory",
        "prediction_inventory_sha256",
        base=manifest_path.parent,
    )
    transform_path, transform_sha = _resolve_bound_path(
        manifest,
        "common_transform",
        "common_transform_sha256",
        base=manifest_path.parent,
    )
    root = primary._load_json(input_manifest)
    inventory = primary._load_json(inventory_path)
    matched._verify_prediction_inventory_files(inventory)
    transform = FrozenBenchmarkTransform.from_json(
        transform_path.read_text(encoding="utf-8")
    )

    available_methods = [str(value) for value in manifest.get("methods", [])]
    available_tracks = [
        track for track in TRACK_ORDER if track in manifest.get("tracks", {})
    ]
    available_targets = [int(value) for value in manifest.get("targets", [])]
    methods = _select_values(
        available_methods, requested_methods, label="methods"
    )
    tracks = _select_values(
        available_tracks, requested_tracks, label="tracks"
    )
    targets = _select_values(
        available_targets, requested_targets, label="targets"
    )
    selected = [
        record
        for record in inventory.get("records", [])
        if str(record.get("canonical_method")) in methods
        and str(record.get("track")) in tracks
        and int(record.get("target")) in targets
    ]
    expected = {
        (method, track, target)
        for method in methods
        for track in tracks
        for target in targets
    }
    observed = {
        (
            str(record.get("canonical_method")),
            str(record.get("track")),
            int(record.get("target")),
        )
        for record in selected
    }
    missing = sorted(expected.difference(observed))
    if missing:
        rendered = ", ".join(
            f"{method}/{track}/t{target}" for method, track, target in missing
        )
        raise ContractError(f"selected prediction inventory is incomplete: {rendered}")
    return {
        "matched_manifest_path": manifest_path,
        "matched_manifest_sha256": primary.sha256_file(manifest_path),
        "matched_manifest": manifest,
        "input_manifest_path": input_manifest,
        "input_manifest_sha256": input_manifest_sha,
        "input_manifest": root,
        "inventory_path": inventory_path,
        "inventory_sha256": inventory_sha,
        "inventory": inventory,
        "transform_path": transform_path,
        "transform_sha256": transform_sha,
        "transform": transform,
        "methods": methods,
        "tracks": tracks,
        "targets": targets,
        "records": selected,
    }


def _load_case(
    *,
    root: dict[str, Any],
    input_manifest: Path,
    manifest_sha: str,
    track: str,
    target: int,
) -> dict[str, Any]:
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
    truth_state, truth_spatial = primary._load_truth(truth_path)
    return {
        "split_id": split_id,
        "training_path": training_path,
        "training_sha256": training_sha,
        "source_roster_path": roster_path,
        "source_roster_sha256": roster_sha,
        "truth_path": truth_path,
        "truth_sha256": truth_sha,
        "truth_state": truth_state,
        "truth_spatial": truth_spatial,
        "input_manifest_sha256": manifest_sha,
    }


def _verified_prediction(
    record: dict[str, Any],
    *,
    case: dict[str, Any],
) -> dict[str, Any]:
    prediction_path = Path(str(record["prediction_path"])).expanduser().resolve()
    summary_path = Path(str(record["prediction_summary"])).expanduser().resolve()
    if primary.sha256_file(prediction_path) != str(record["prediction_sha256"]):
        raise ContractError(f"prediction changed after inventory bind: {prediction_path}")
    if primary.sha256_file(summary_path) != str(
        record["prediction_summary_sha256"]
    ):
        raise ContractError(f"summary changed after inventory bind: {summary_path}")
    summary = primary._load_json(summary_path)
    primary._verify_prediction_provenance(
        summary,
        manifest_sha=case["input_manifest_sha256"],
        training_sha=case["training_sha256"],
        source_roster_sha=case["source_roster_sha256"],
    )
    declared_prediction_sha = primary._summary_prediction_sha(summary)
    if declared_prediction_sha != str(record["prediction_sha256"]):
        raise ContractError(
            f"summary prediction SHA differs from inventory: {summary_path}"
        )
    state, spatial, weights = primary._prediction_arrays(prediction_path)
    actual_spaces = ["state"] if spatial is None else ["joint", "state", "spatial"]
    if set(actual_spaces) != set(record.get("actual_output_spaces", [])):
        raise ContractError(
            f"prediction spaces differ from inventory: {prediction_path}"
        )
    return {
        "path": prediction_path,
        "summary_path": summary_path,
        "summary": summary,
        "state": state,
        "spatial": spatial,
        "weights": weights,
        "actual_spaces": actual_spaces,
    }


def _assert_shared_truth(
    cache: dict[int, tuple[np.ndarray, np.ndarray]],
    *,
    target: int,
    state: np.ndarray,
    spatial: np.ndarray,
) -> None:
    if target not in cache:
        cache[target] = (state.copy(), spatial.copy())
        return
    expected_state, expected_spatial = cache[target]
    if not np.array_equal(state, expected_state) or not np.array_equal(
        spatial, expected_spatial
    ):
        raise ContractError(
            f"matched tracks do not expose identical truth arrays for t{target}"
        )


def _evaluate(context: dict[str, Any], args: argparse.Namespace) -> pd.DataFrame:
    root = context["input_manifest"]
    transform = context["transform"]
    benchmark = matched._dataset_label(root)
    cases: dict[tuple[str, int], dict[str, Any]] = {}
    shared_truth: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    observed_sampling: dict[tuple[int, str], dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []

    records = sorted(
        context["records"],
        key=lambda record: (
            context["targets"].index(int(record["target"])),
            context["methods"].index(str(record["canonical_method"])),
            TRACK_ORDER.index(str(record["track"])),
        ),
    )
    for record in records:
        track = str(record["track"])
        target = int(record["target"])
        case_key = (track, target)
        if case_key not in cases:
            cases[case_key] = _load_case(
                root=root,
                input_manifest=context["input_manifest_path"],
                manifest_sha=context["input_manifest_sha256"],
                track=track,
                target=target,
            )
            _assert_shared_truth(
                shared_truth,
                target=target,
                state=cases[case_key]["truth_state"],
                spatial=cases[case_key]["truth_spatial"],
            )
        case = cases[case_key]
        prediction = _verified_prediction(record, case=case)
        transformed = matched._transformed_spaces(
            transform=transform,
            predicted_state=prediction["state"],
            observed_state=case["truth_state"],
            predicted_spatial=prediction["spatial"],
            observed_spatial=(
                case["truth_spatial"]
                if prediction["spatial"] is not None
                else None
            ),
        )
        seed_split = f"matched_t{target}"
        for space in SPACE_ORDER:
            if space not in transformed:
                continue
            predicted_values, observed_values = transformed[space]
            sampling_key = (target, space)
            observed_sha = _array_sha256(observed_values)
            exact_seed = matched._exact_ot_seed(benchmark, seed_split, space)
            if sampling_key not in observed_sampling:
                indices = matched._observed_exact_indices(
                    observed_values.shape[0],
                    args.max_ot_points,
                    exact_seed,
                )
                observed_sampling[sampling_key] = {
                    "observed_sha256": observed_sha,
                    "indices": indices,
                }
            sampling = observed_sampling[sampling_key]
            if sampling["observed_sha256"] != observed_sha:
                raise ContractError(
                    f"observed transformed population differs for t{target}/{space}"
                )
            metrics = _space_metrics(
                predicted=predicted_values,
                observed=observed_values,
                predicted_weights=prediction["weights"],
                benchmark=benchmark,
                seed_split=seed_split,
                space=space,
                n_projections=args.n_projections,
                projection_repeats=args.projection_repeats,
                max_ot_points=args.max_ot_points,
                observed_indices=sampling["indices"],
            )
            role = TRACK_METADATA[track]
            rows.append(
                {
                    "benchmark": benchmark,
                    "analysis": "supplemental_centered_shape",
                    "track": track,
                    "evaluation_scope": role["evaluation_scope"],
                    "comparison_role": role["comparison_role"],
                    "is_in_sample": role["is_in_sample"],
                    "is_oracle_control": role["is_oracle_control"],
                    "target": target,
                    "fold": case["split_id"],
                    "seed_pairing_split": seed_split,
                    "method": str(record["canonical_method"]),
                    "canonical_method": str(record["canonical_method"]),
                    "raw_method": str(record["raw_method"]),
                    "method_display_name": str(record["method_display_name"]),
                    "space": space,
                    "output_scope": str(record["output_scope"]),
                    "native_vs_adapter": str(record["native_vs_adapter"]),
                    "scope_compatibility": str(record["scope_compatibility"]),
                    "prediction_weights_source": (
                        "prediction_npz_weights"
                        if prediction["weights"] is not None
                        else "uniform_over_prediction_rows"
                    ),
                    "native_mass_normalized_away": bool(
                        record.get("native_mass", False)
                        and record.get("weights_are_unnormalised", False)
                    ),
                    "prediction_path": str(prediction["path"]),
                    "prediction_sha256": str(record["prediction_sha256"]),
                    "prediction_summary": str(prediction["summary_path"]),
                    "prediction_summary_sha256": str(
                        record["prediction_summary_sha256"]
                    ),
                    "truth_reference": str(case["truth_path"]),
                    "truth_reference_sha256": case["truth_sha256"],
                    "training_reference": str(case["training_path"]),
                    "training_reference_sha256": case["training_sha256"],
                    "source_roster": str(case["source_roster_path"]),
                    "source_roster_sha256": case["source_roster_sha256"],
                    "transform_path": str(context["transform_path"]),
                    "transform_sha256": context["transform_sha256"],
                    **metrics,
                }
            )
    if not rows:
        raise ContractError("no prediction records remained after filtering")
    return pd.DataFrame.from_records(rows).sort_values(
        ["target", "space", "method", "track"], kind="stable"
    ).reset_index(drop=True)


def _summary(metrics: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "benchmark",
        "track",
        "evaluation_scope",
        "comparison_role",
        "is_in_sample",
        "is_oracle_control",
        "method",
        "canonical_method",
        "raw_method",
        "method_display_name",
        "space",
    ]
    rows: list[dict[str, Any]] = []
    for values, frame in metrics.groupby(keys, sort=False, dropna=False):
        row = dict(zip(keys, values))
        row["n_targets"] = int(frame["target"].nunique())
        for metric in SHAPE_METRICS:
            metric_values = frame[metric].to_numpy(dtype=np.float64)
            row[f"{metric}_mean"] = float(metric_values.mean())
            row[f"{metric}_target_sd"] = float(
                metric_values.std(ddof=1) if metric_values.size > 1 else 0.0
            )
            row[f"{metric}_min"] = float(metric_values.min())
            row[f"{metric}_max"] = float(metric_values.max())
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def _paired_gaps(metrics: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "benchmark",
        "method",
        "canonical_method",
        "method_display_name",
        "target",
        "space",
        "comparison",
    ]
    for metric in SHAPE_METRICS:
        columns.extend(
            [
                f"{metric}_loto_transductive",
                f"{metric}_full_data_in_sample",
                f"{metric}_loto_minus_full_data",
            ]
        )
    rows: list[dict[str, Any]] = []
    grouped = metrics.groupby(
        [
            "benchmark",
            "method",
            "canonical_method",
            "method_display_name",
            "target",
            "space",
        ],
        sort=False,
        dropna=False,
    )
    for values, frame in grouped:
        by_track = {str(row["track"]): row for _, row in frame.iterrows()}
        if set(by_track) != {"loto", "full_data"}:
            continue
        row = dict(
            zip(
                [
                    "benchmark",
                    "method",
                    "canonical_method",
                    "method_display_name",
                    "target",
                    "space",
                ],
                values,
            )
        )
        row["comparison"] = (
            "loto_transductive_interpolation_minus_"
            "full_data_in_sample_oracle_control"
        )
        for metric in SHAPE_METRICS:
            loto = float(by_track["loto"][metric])
            full = float(by_track["full_data"][metric])
            row[f"{metric}_loto_transductive"] = loto
            row[f"{metric}_full_data_in_sample"] = full
            row[f"{metric}_loto_minus_full_data"] = loto - full
        rows.append(row)
    return pd.DataFrame.from_records(rows, columns=columns)


def _safe_filename(value: str) -> str:
    rendered = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return rendered or "plot"


def _plot_metric(
    metrics: pd.DataFrame,
    *,
    metric: str,
    method_order: list[str],
    output_dir: Path,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    colors = {"full_data": "#9E9E9E", "loto": "#2C7FB8"}
    offsets = {"full_data": -0.18, "loto": 0.18}
    display = (
        metrics[["method", "method_display_name"]]
        .drop_duplicates("method")
        .set_index("method")["method_display_name"]
        .to_dict()
    )
    figure, axes = plt.subplots(1, 3, figsize=(max(12.5, len(method_order) * 1.4), 4.8))
    for axis, space in zip(axes, SPACE_ORDER):
        frame = metrics[metrics["space"] == space]
        x = np.arange(len(method_order), dtype=np.float64)
        for track in ("full_data", "loto"):
            means = (
                frame[frame["track"] == track]
                .groupby("method", sort=False)[metric]
                .mean()
            )
            heights = np.asarray(
                [float(means.get(method, np.nan)) for method in method_order]
            )
            valid = np.isfinite(heights)
            axis.bar(
                x[valid] + offsets[track],
                heights[valid],
                width=0.32,
                color=colors[track],
                alpha=0.72,
                label=(
                    "Full-data (in-sample/oracle control)"
                    if track == "full_data"
                    else "LOTO (held-out, transductive)"
                ),
                zorder=1,
            )
            for method_index, method in enumerate(method_order):
                points = frame[
                    (frame["track"] == track) & (frame["method"] == method)
                ]
                if points.empty:
                    continue
                axis.scatter(
                    np.full(len(points), x[method_index] + offsets[track]),
                    points[metric],
                    s=20,
                    color=colors[track],
                    edgecolor="white",
                    linewidth=0.45,
                    zorder=3,
                )
        for method_index, method in enumerate(method_order):
            method_frame = frame[frame["method"] == method]
            for _, target_frame in method_frame.groupby("target", sort=True):
                if set(target_frame["track"]) != {"loto", "full_data"}:
                    continue
                values = target_frame.set_index("track")[metric]
                axis.plot(
                    [
                        x[method_index] + offsets["full_data"],
                        x[method_index] + offsets["loto"],
                    ],
                    [values["full_data"], values["loto"]],
                    color="#666666",
                    alpha=0.35,
                    linewidth=0.7,
                    zorder=2,
                )
        axis.set_title(space.capitalize())
        axis.set_xticks(x)
        axis.set_xticklabels(
            [display.get(method, method) for method in method_order],
            rotation=40,
            ha="right",
        )
        axis.set_ylabel(METRIC_LABELS[metric] + " (lower is better)")
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.6, alpha=0.8)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        figure.legend(
            handles[:2],
            labels[:2],
            loc="upper center",
            ncol=2,
            frameon=False,
        )
    figure.suptitle(
        f"{METRIC_LABELS[metric]}: paired targets, no method-specific tuning",
        y=1.03,
    )
    figure.tight_layout()
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    stem = _safe_filename(metric)
    png = plot_dir / f"{stem}.png"
    pdf = plot_dir / f"{stem}.pdf"
    if png.exists() or pdf.exists():
        raise ContractError(f"refusing to overwrite shape plot {stem}")
    figure.savefig(
        png,
        dpi=180,
        bbox_inches="tight",
        metadata={"Software": "reviewer_shape_analysis.py"},
    )
    figure.savefig(
        pdf,
        bbox_inches="tight",
        metadata={"Creator": "reviewer_shape_analysis.py", "CreationDate": None},
    )
    plt.close(figure)
    return [png, pdf]


def _software_versions() -> dict[str, str]:
    import matplotlib
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
        "matplotlib": str(matplotlib.__version__),
    }


def _dependency_hashes() -> dict[str, dict[str, str]]:
    paths = {
        "shape_analysis": Path(__file__).resolve(),
        "matched_evaluator": Path(matched.__file__).resolve(),
        "primary_evaluator": Path(primary.__file__).resolve(),
        "benchmark_metrics": Path(benchmark_metrics.__file__).resolve(),
    }
    return {
        name: {"path": str(path), "sha256": primary.sha256_file(path)}
        for name, path in paths.items()
    }


def _prepare_output_dir(output_dir: Path) -> Path:
    resolved = output_dir.expanduser().resolve()
    if resolved.exists() and any(resolved.iterdir()):
        raise ContractError(
            f"shape-analysis output directory must be empty: {resolved}"
        )
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _write_outputs(
    *,
    context: dict[str, Any],
    args: argparse.Namespace,
    metrics: pd.DataFrame,
    summary: pd.DataFrame,
    paired: pd.DataFrame,
) -> dict[str, Any]:
    output_dir = _prepare_output_dir(args.output_dir)
    metrics_path = output_dir / "shape_metrics_long.csv"
    summary_path = output_dir / "shape_metrics_summary.csv"
    paired_path = output_dir / "shape_metrics_paired_gaps.csv"
    matched._write_immutable_csv(metrics_path, metrics)
    matched._write_immutable_csv(summary_path, summary)
    matched._write_immutable_csv(paired_path, paired)

    plot_paths: list[Path] = []
    if not args.no_plots:
        for metric in SHAPE_METRICS:
            plot_paths.extend(
                _plot_metric(
                    metrics,
                    metric=metric,
                    method_order=context["methods"],
                    output_dir=output_dir,
                )
            )
    artifacts = {
        "shape_metrics_long_csv": {
            "path": str(metrics_path),
            "sha256": primary.sha256_file(metrics_path),
        },
        "shape_metrics_summary_csv": {
            "path": str(summary_path),
            "sha256": primary.sha256_file(summary_path),
        },
        "shape_metrics_paired_gaps_csv": {
            "path": str(paired_path),
            "sha256": primary.sha256_file(paired_path),
        },
        "plots": [
            {"path": str(path), "sha256": primary.sha256_file(path)}
            for path in plot_paths
        ],
    }
    manifest = {
        "schema_version": "1.0.0",
        "status": "complete",
        "analysis": "supplemental_centered_shape",
        "primary_metrics_modified": False,
        "matched_evaluation_manifest": str(context["matched_manifest_path"]),
        "matched_evaluation_manifest_sha256": context[
            "matched_manifest_sha256"
        ],
        "input_manifest": str(context["input_manifest_path"]),
        "input_manifest_sha256": context["input_manifest_sha256"],
        "prediction_inventory": str(context["inventory_path"]),
        "prediction_inventory_sha256": context["inventory_sha256"],
        "common_transform": str(context["transform_path"]),
        "common_transform_sha256": context["transform_sha256"],
        "method_registry": context["matched_manifest"]["method_registry"],
        "dataset": matched._dataset_label(context["input_manifest"]),
        "methods": context["methods"],
        "tracks": {
            track: TRACK_METADATA[track] for track in context["tracks"]
        },
        "targets": context["targets"],
        "spaces": [
            space for space in SPACE_ORDER if space in set(metrics["space"])
        ],
        "metrics": list(SHAPE_METRICS),
        "metric_direction": "lower_is_better_for_every_reported_metric",
        "metric_definitions": {
            "centroid_error": (
                "Euclidean distance between weighted prediction and uniform-truth "
                "centroids in the matched frozen transformed space."
            ),
            "centered_sliced_w2": (
                "Mean sliced W2 after subtracting each distribution's own centroid; "
                "uses full empirical supports and shared projections."
            ),
            "centered_exact_w1": (
                "Exact EMD W1 on the centered evaluated supports; when a support "
                "exceeds max_ot_points this is exact only on the deterministic "
                "audited sample."
            ),
            "centered_exact_w2": (
                "Square root of exact squared-cost EMD on the centered evaluated "
                "supports; the same cap qualification as centered_exact_w1 applies."
            ),
            "covariance_bures_distance": (
                "Bures distance between full-support weighted population covariance "
                "matrices after centering (equivalently centered Gaussian W2)."
            ),
        },
        "weight_policy": {
            "prediction": (
                "Use prediction weights when present, otherwise uniform row mass; "
                "normalize to probability for all shape metrics."
            ),
            "truth": "uniform mass over observed truth rows",
            "native_mass": (
                "Total predicted mass is intentionally normalized away here; TMV "
                "remains a separate primary-benchmark quantity."
            ),
            "duplicate_support": (
                "Exact duplicate prediction rows are collapsed independently in "
                "each transformed space and their normalized masses are summed. "
                "This preserves the empirical measure and avoids treating bootstrap "
                "copies as new geometry."
            ),
        },
        "randomness_policy": {
            "seed_pairing_split": "matched_t{target}",
            "method_in_seed": False,
            "track_in_seed": False,
            "projection_basis_shared_across_methods_and_tracks": True,
            "observed_exact_ot_indices_shared_across_methods_and_tracks": True,
            "predicted_and_observed_exact_ot_rng_streams_separate": True,
        },
        "n_projections": int(args.n_projections),
        "projection_repeats": int(args.projection_repeats),
        "max_ot_points": int(args.max_ot_points),
        "n_metric_rows": int(len(metrics)),
        "n_summary_rows": int(len(summary)),
        "n_paired_rows": int(len(paired)),
        "reporting_policy": {
            "all_methods_same_metrics_and_parameters": True,
            "method_specific_tuning": False,
            "cross_space_aggregation": False,
            "overall_score": False,
            "ranking": False,
            "statistical_inference": False,
            "bars_are_unweighted_target_means": True,
            "dots_are_individual_targets": True,
            "paired_lines_connect_the_same_target": True,
        },
        "scientific_limitations": [
            (
                "Zebrafish LOTO is held out at the row level but remains a "
                "transductive frozen-representation interpolation benchmark: the "
                "shared PCA/spatial representations and the matched transform were "
                "defined using the audited common preprocessing/anchors."
            ),
            (
                "Full-data values are in-sample/oracle controls, not independent "
                "generalization estimates; they are never pooled with LOTO or used "
                "to rank it."
            ),
            (
                "Centering deliberately removes global translation. A method can "
                "have good centered shape but poor centroid placement, so centroid "
                "error and centered metrics must be read together."
            ),
            (
                "Covariance Bures distance measures only second-order shape; centered "
                "Wasserstein metrics retain higher-order distribution information."
            ),
            (
                "Each row represents one available fitted prediction. Projection "
                "repeats do not estimate model-training-seed uncertainty."
            ),
        ],
        "artifacts": artifacts,
        "code_dependencies": _dependency_hashes(),
        "software_versions": _software_versions(),
    }

    # Rehash external inputs immediately before final publication.
    matched._verify_bound_inventory_from_manifest(context["matched_manifest"])
    for path, expected, label in (
        (
            context["matched_manifest_path"],
            context["matched_manifest_sha256"],
            "matched manifest",
        ),
        (
            context["input_manifest_path"],
            context["input_manifest_sha256"],
            "input manifest",
        ),
        (
            context["transform_path"],
            context["transform_sha256"],
            "common transform",
        ),
    ):
        if primary.sha256_file(path) != expected:
            raise ContractError(f"{label} changed before final publication: {path}")
    matched._write_final_manifest(
        output_dir / "shape_analysis_manifest.json", manifest
    )
    return manifest


def run_analysis(
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    args.n_projections = _positive_int(
        args.n_projections, name="--n-projections"
    )
    args.projection_repeats = _positive_int(
        args.projection_repeats, name="--projection-repeats"
    )
    args.max_ot_points = _positive_int(
        args.max_ot_points, name="--max-ot-points"
    )
    context = _load_context(
        args.matched_manifest,
        requested_methods=args.methods,
        requested_tracks=args.tracks,
        requested_targets=args.targets,
    )
    metrics = _evaluate(context, args)
    summary = _summary(metrics)
    paired = _paired_gaps(metrics)
    manifest = _write_outputs(
        context=context,
        args=args,
        metrics=metrics,
        summary=summary,
        paired=paired,
    )
    return metrics, summary, paired, manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matched-manifest",
        type=Path,
        required=True,
        help="Completed matched_evaluation_manifest.json to verify and reuse.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--methods",
        nargs="+",
        help="Optional exact canonical method names from the matched manifest.",
    )
    parser.add_argument(
        "--tracks",
        nargs="+",
        choices=list(TRACK_ORDER),
        help="Defaults to both loto and full_data.",
    )
    parser.add_argument("--targets", type=int, nargs="+")
    parser.add_argument("--n-projections", type=int, default=1024)
    parser.add_argument("--projection-repeats", type=int, default=5)
    parser.add_argument("--max-ot-points", type=int, default=800)
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Write audited CSV/manifest outputs without PNG/PDF figures.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        _, _, _, manifest = run_analysis(args)
    except (ContractError, FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
