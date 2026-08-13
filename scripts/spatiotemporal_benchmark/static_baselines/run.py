"""Run leakage-audited static/coupling spatiotemporal benchmark adapters."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .coupling import (
    compose_row_plans,
    linear_centroid_loto,
    linear_centroid_trajectory,
    project_composed_joint,
    project_composed_state,
    project_loto_joint,
    project_loto_state,
    random_independent_plan,
    take_roster,
    validate_and_row_normalize,
)
from .data import (
    CONTRACT_UNS_KEY,
    InputKeys,
    TrajectoryInput,
    build_source_roster,
    ids_sha256,
    load_trajectory,
    sha256_file,
    stable_seed,
)
from .errors import BaselineError, DependencyUnavailable
from .methods import (
    fit_block_balance,
    representation_spec,
    resolve_parameters,
    run_official_coupling,
)
from .provenance import dependency_probe
from .registry import get_method_spec, list_method_specs, load_registry


METHODS = tuple(sorted(list_method_specs()))
OFFICIAL_METHODS = {"moscot", "wot", "paste", "spateo", "spatrack"}
REPRESENTATIONS = ("matched_state_spatial", "native_gene_sensitivity")
_ADAPTER_IMPLEMENTATION_FILES = (
    "scripts/spatiotemporal_benchmark/static_baselines/run.py",
    "scripts/spatiotemporal_benchmark/static_baselines/methods.py",
    "scripts/spatiotemporal_benchmark/static_baselines/coupling.py",
    "scripts/spatiotemporal_benchmark/static_baselines/data.py",
    "scripts/spatiotemporal_benchmark/static_baselines/provenance.py",
    "scripts/spatiotemporal_benchmark/static_baselines/registry.py",
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return "inf" if value > 0 else "-inf"
    return value


def _adapter_implementation() -> dict[str, Any]:
    """Hash the exact package-side static adapter implementation."""

    root = Path(__file__).resolve().parents[3]
    files = {name: sha256_file(root / name) for name in _ADAPTER_IMPLEMENTATION_FILES}
    aggregate = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": "1.0.0",
        "files": files,
        "aggregate_sha256": aggregate,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _artifact(path: Path, **extra: Any) -> dict[str, Any]:
    result = {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "size_bytes": int(path.stat().st_size),
    }
    result.update(extra)
    return result


def _time_token(value: float) -> str:
    return (
        str(int(value)) if float(value).is_integer() else str(value).replace(".", "p")
    )


def _load_parameter_overrides(raw: str | None) -> dict[str, Any]:
    if raw is None:
        return {}
    candidate = Path(raw)
    text = candidate.read_text(encoding="utf-8") if candidate.is_file() else raw
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("--params-json must encode a JSON object")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    registry_parser = commands.add_parser(
        "registry", help="Print the audited method registry"
    )
    registry_parser.add_argument("--method", choices=METHODS)

    run_parser = commands.add_parser("run", help="Run or dry-run one adapter")
    run_parser.add_argument("--method", choices=METHODS, required=True)
    run_parser.add_argument(
        "--representation", choices=REPRESENTATIONS, default="matched_state_spatial"
    )
    run_parser.add_argument("--input-h5ad", type=Path, required=True)
    run_parser.add_argument("--input-manifest", type=Path, default=None)
    run_parser.add_argument("--training-reference", type=Path, default=None)
    run_parser.add_argument(
        "--target-time",
        "--holdout-time",
        dest="target_time",
        type=float,
        default=None,
        help="Required for LOTO. No-holdout always writes every non-initial stage.",
    )
    run_parser.add_argument(
        "--evaluation-mode",
        choices=["loto", "no-holdout", "full-data"],
        required=True,
        help="full-data is accepted as a backward-compatible alias for no-holdout",
    )
    run_parser.add_argument("--output-dir", type=Path, required=True)
    run_parser.add_argument("--max-fit-n", type=int, default=800)
    run_parser.add_argument("--seed", type=int, default=20260718)
    run_parser.add_argument("--params-json", default=None)
    run_parser.add_argument("--source-root", type=Path, default=None)
    run_parser.add_argument("--expression-key", default=None)
    run_parser.add_argument("--spatial-key", default=None)
    run_parser.add_argument("--state-key", default=None)
    run_parser.add_argument("--time-key", default=None)
    run_parser.add_argument("--row-id-key", default=None)
    run_parser.add_argument("--contract-uns-key", default=CONTRACT_UNS_KEY)
    run_parser.add_argument("--allow-unverified-preprocessing", action="store_true")
    run_parser.add_argument("--dry-run", action="store_true")
    run_parser.add_argument("--require-dependency-in-dry-run", action="store_true")
    run_parser.add_argument("--save-coupling", action="store_true")
    run_parser.add_argument("--write-csv", action="store_true")
    return parser


def _manifest_split_candidates(mode: str, target_time: float | None) -> tuple[str, ...]:
    if mode == "no-holdout":
        return ("no_holdout", "no-holdout", "full_data", "full-data")
    if target_time is None:
        return ()
    token = _time_token(target_time)
    return (f"loto_t{token}", f"loto_{token}")


def _nested_sha(record: dict[str, Any]) -> str | None:
    candidates: tuple[Any, ...] = (
        record.get("train_h5ad_sha256"),
        record.get("h5ad_sha256"),
        record.get("train", {}).get("h5ad", {}).get("sha256")
        if isinstance(record.get("train"), dict)
        else None,
    )
    return next((str(value) for value in candidates if value), None)


def _nested_reference_sha(record: dict[str, Any]) -> str | None:
    train = record.get("train")
    nested = train.get("training_reference_npz") if isinstance(train, dict) else None
    candidates: tuple[Any, ...] = (
        record.get("training_reference_sha256"),
        nested.get("sha256") if isinstance(nested, dict) else None,
    )
    return next((str(value) for value in candidates if value), None)


def _verify_external_provenance(
    args: argparse.Namespace, input_sha: str
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "input_manifest": None,
        "input_manifest_sha256": None,
        "input_manifest_split": None,
        "input_manifest_h5ad_verified": False,
        "training_reference": None,
        "training_reference_sha256": None,
        "source_roster": None,
        "source_roster_sha256": None,
    }
    manifest_reference: Path | None = None
    manifest_roster: Path | None = None
    manifest_roster_sha: str | None = None
    if args.input_manifest is not None:
        path = args.input_manifest.resolve()
        if not path.is_file():
            raise ValueError(f"Input manifest does not exist: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Input manifest root must be a JSON object")
        splits = payload.get("splits")
        split_name = None
        split = None
        if isinstance(splits, dict):
            for candidate in _manifest_split_candidates(
                args.evaluation_mode, args.target_time
            ):
                if candidate in splits:
                    split_name, split = candidate, splits[candidate]
                    break
        if split is None:
            # A split-local manifest is also accepted.
            split = payload
            split_name = str(payload.get("split", "split-local"))
        if not isinstance(split, dict):
            raise ValueError("Input manifest split is invalid")
        expected = _nested_sha(split)
        if expected is None:
            if not args.allow_unverified_preprocessing:
                raise ValueError("Input manifest has no train H5AD SHA256")
        elif expected != input_sha:
            raise ValueError(
                f"Input H5AD SHA256 does not match manifest: {input_sha} != {expected}"
            )
        train_record = split.get("train")
        reference_record = (
            train_record.get("training_reference_npz")
            if isinstance(train_record, dict)
            else None
        )
        roster_record = (
            train_record.get("source_roster_npz")
            if isinstance(train_record, dict)
            else None
        )
        if isinstance(reference_record, dict) and (
            reference_record.get("relative_path") or reference_record.get("path")
        ):
            raw_reference = Path(
                str(reference_record.get("relative_path", reference_record.get("path")))
            ).expanduser()
            manifest_reference = (
                raw_reference.resolve()
                if raw_reference.is_absolute()
                else (path.parent / raw_reference).resolve()
            )
        if isinstance(roster_record, dict) and (
            roster_record.get("relative_path") or roster_record.get("path")
        ):
            raw_roster = Path(
                str(roster_record.get("relative_path", roster_record.get("path")))
            ).expanduser()
            manifest_roster = (
                raw_roster.resolve()
                if raw_roster.is_absolute()
                else (path.parent / raw_roster).resolve()
            )
            manifest_roster_sha = str(roster_record.get("sha256", "")).lower()
        result.update(
            {
                "input_manifest": str(path),
                "input_manifest_sha256": sha256_file(path),
                "input_manifest_split": split_name,
                "input_manifest_h5ad_verified": expected == input_sha,
                "input_manifest_training_reference_expected_sha256": _nested_reference_sha(
                    split
                ),
            }
        )
    if args.training_reference is not None or manifest_reference is not None:
        reference = (
            args.training_reference.resolve()
            if args.training_reference is not None
            else manifest_reference
        )
        assert reference is not None
        if not reference.is_file():
            raise ValueError(f"Training reference does not exist: {reference}")
        result["training_reference"] = str(reference)
        result["training_reference_sha256"] = sha256_file(reference)
        expected_reference = result.get(
            "input_manifest_training_reference_expected_sha256"
        )
        if (
            expected_reference
            and result["training_reference_sha256"] != expected_reference
        ):
            raise ValueError(
                "Training reference SHA256 does not match input manifest: "
                f"{result['training_reference_sha256']} != {expected_reference}"
            )
        result["input_manifest_training_reference_verified"] = bool(
            expected_reference
            and result["training_reference_sha256"] == expected_reference
        )
    if args.input_manifest is None:
        return result
    if manifest_roster is None or not manifest_roster.is_file():
        raise ValueError("Input manifest has no readable canonical source roster")
    observed_roster_sha = sha256_file(manifest_roster)
    if not manifest_roster_sha or observed_roster_sha != manifest_roster_sha:
        raise ValueError("Canonical source roster SHA256 does not match input manifest")
    result["source_roster"] = str(manifest_roster)
    result["source_roster_sha256"] = observed_roster_sha
    return result


def _save_anchor_selection(output_dir: Path, data: TrajectoryInput) -> dict[str, Any]:
    arrays: dict[str, np.ndarray] = {}
    for stage in data.stages:
        token = _time_token(stage.time)
        arrays[f"row_ids_t{token}"] = stage.row_ids.astype(str)
        arrays[f"source_indices_t{token}"] = stage.source_indices.astype(np.int64)
    path = output_dir / "anchor_selection.npz"
    np.savez_compressed(path, **arrays)
    return _artifact(
        path,
        stage_shapes={
            str(stage.time): [stage.n_obs, stage.state_pca.shape[1]]
            for stage in data.stages
        },
    )


def _canonical_source_roster(
    data: TrajectoryInput,
    source_stage: Any,
    provenance: Mapping[str, Any],
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    if not provenance.get("source_roster"):
        return build_source_roster(source_stage, data.prediction_n, args.seed)
    expected_support = int(data.contract.get("source_roster_support_n", -1))
    expected_seed = int(data.contract.get("source_roster_seed", -1))
    if expected_support != int(args.max_fit_n):
        raise ValueError(
            f"--max-fit-n must equal canonical source_roster_support_n={expected_support}"
        )
    if expected_seed != int(args.seed):
        raise ValueError(
            f"--seed must equal canonical source_roster_seed={expected_seed}"
        )
    roster_path = Path(str(provenance["source_roster"]))
    with np.load(roster_path, allow_pickle=False) as archive:
        required = {"row_id", "source_time", "spatial", "state"}
        missing = sorted(required.difference(archive.files))
        if missing:
            raise ValueError(f"Canonical source roster lacks keys {missing}")
        row_ids = np.asarray(archive["row_id"]).astype(str)
        source_time = float(np.asarray(archive["source_time"]).reshape(-1)[0])
        spatial = np.asarray(archive["spatial"], dtype=np.float32)
        state = np.asarray(archive["state"], dtype=np.float32)
    if row_ids.shape != (data.prediction_n,) or not np.isclose(
        source_time, source_stage.time
    ):
        raise ValueError("Canonical source roster has wrong row count or source time")
    lookup = {
        value: index for index, value in enumerate(source_stage.row_ids.astype(str))
    }
    try:
        indices = np.asarray([lookup[value] for value in row_ids], dtype=np.int64)
    except KeyError as exc:
        raise ValueError(
            "Canonical source roster row is absent from the method-independent fitted support"
        ) from exc
    if not np.allclose(source_stage.spatial[indices], spatial, rtol=1e-6, atol=1e-6):
        raise ValueError(
            "Canonical source roster spatial values differ from fitted support"
        )
    if not np.allclose(source_stage.state_pca[indices], state, rtol=1e-6, atol=1e-6):
        raise ValueError(
            "Canonical source roster state values differ from fitted support"
        )
    return indices, row_ids


def _base_manifest(
    args: argparse.Namespace,
    data: TrajectoryInput,
    method_spec: dict[str, Any],
    rep_spec: dict[str, Any],
    params: dict[str, Any],
    provenance: dict[str, Any],
    roster_indices: np.ndarray,
    roster_ids: np.ndarray,
) -> dict[str, Any]:
    state_dim = data.stages[0].state_pca.shape[1]
    spatial_dim = data.stages[0].spatial.shape[1]
    scope = rep_spec.get("output_scope", "N/A")
    return {
        "schema_version": "2.0.0",
        "status": "initialized",
        "dataset": data.contract.get(
            "dataset_id", data.contract.get("dataset", "unspecified")
        ),
        "method": args.method,
        "adapter_implementation": _adapter_implementation(),
        "representation": args.representation,
        "method_spec": method_spec,
        "representation_spec": rep_spec,
        "parameters": params,
        "seed": int(args.seed),
        "max_fit_n": int(args.max_fit_n),
        "input": {
            "train_h5ad": data.input_path,
            "train_h5ad_sha256": data.input_sha256,
            "contract_uns_key": args.contract_uns_key,
            "contract": data.contract,
            "resolved_keys": asdict(data.keys),
            **provenance,
        },
        "protocol": {
            "mode": data.mode,
            "requested_target_time": args.target_time,
            # In LOTO, ``data.time_values`` may retain the declared held target
            # for bracket semantics.  Provenance must describe only fitted
            # train anchors, which are exactly the loaded stages.
            "time_values": sorted(float(stage.time) for stage in data.stages),
            "loto_target": data.target_time,
            "prediction_n": int(data.prediction_n),
            "prediction_size_source": "train H5AD cytobridge_benchmark_contract.prediction_n",
            "truth_artifact_opened": False,
            "truth_cell_count_read": False,
            "target_n_used_for_prediction": False,
            "no_holdout_policy": (
                "fit every adjacent training coupling and compose P01...P(t-1,t) from one fixed t0 roster"
                if data.mode == "no-holdout"
                else None
            ),
            "loto_policy": (
                "physically target-removed nearest left/right bracket coupling with fractional alpha"
                if data.mode == "loto"
                else None
            ),
            "direct_previous_to_target_alpha_one_used": False,
            "comparable_to_strict_loto": data.mode == "loto",
            "no_holdout_is_in_sample": data.mode == "no-holdout",
        },
        "anchors": {
            "sampling_policy": "method-independent seed+row_id hash; sampled once per stage",
            "stage_fit_counts": {str(stage.time): stage.n_obs for stage in data.stages},
            "stage_row_ids_sha256": {
                str(stage.time): ids_sha256(stage.row_ids) for stage in data.stages
            },
            "source_stage": float(
                data.loto_pair().previous.time
                if data.mode == "loto"
                else data.time_values[0]
            ),
            "source_roster_policy": "fixed bootstrap from fitted source anchor using base seed + source row_id digest",
            "source_roster_n": int(len(roster_indices)),
            "source_roster_row_ids_sha256": ids_sha256(roster_ids),
            "source_roster_unique_rows": int(len(np.unique(roster_indices))),
        },
        "output_scope": {
            "scope": scope,
            "state_dimensions": 0 if scope == "N/A" else int(state_dim),
            "spatial_dimensions": (
                0
                if scope in {"N/A", "native_state", "hybrid_state"}
                else int(spatial_dim)
            ),
            "hybrid_adapter": bool(rep_spec.get("hybrid", False)),
            "native_output": rep_spec.get("native_output"),
            "benchmark_output": rep_spec.get("benchmark_output"),
            "weights_exported": False,
            "growth_or_total_mass_evaluated": False,
        },
        "primary_benchmark_eligible": bool(
            args.representation == "matched_state_spatial"
            and rep_spec.get("primary_ranking", False)
        ),
        "surrogate_attempted": False,
    }


def _write_roster(
    output_dir: Path,
    source_time: float,
    indices: np.ndarray,
    row_ids: np.ndarray,
) -> dict[str, Any]:
    path = output_dir / "source_roster.npz"
    np.savez_compressed(
        path,
        source_anchor_indices=np.asarray(indices, dtype=np.int64),
        source_row_ids=np.asarray(row_ids, dtype=str),
        source_time=np.float64(source_time),
        source_row_ids_sha256=np.asarray(ids_sha256(row_ids)),
    )
    return _artifact(path, rows=int(len(indices)), row_ids_sha256=ids_sha256(row_ids))


def _save_prediction(
    path: Path,
    prediction: np.ndarray,
    target_time: float,
    *,
    state_only: bool,
    spatial_dim: int,
    source_row_ids: np.ndarray,
) -> dict[str, Any]:
    prediction = np.asarray(prediction, dtype=np.float32)
    if prediction.ndim != 2 or not np.isfinite(prediction).all():
        raise ValueError(f"Prediction is invalid: shape={prediction.shape}")
    common = {
        "state": prediction if state_only else prediction[:, spatial_dim:],
        "state_pca": prediction if state_only else prediction[:, spatial_dim:],
        "time": np.full(len(prediction), float(target_time), dtype=np.float32),
        "source_row_id": np.asarray(source_row_ids, dtype=str),
    }
    if state_only:
        np.savez_compressed(path, **common)
        shape = {"state": list(prediction.shape), "spatial": None}
    else:
        np.savez_compressed(
            path,
            points=prediction,
            spatial=prediction[:, :spatial_dim],
            **common,
        )
        shape = {
            "points": list(prediction.shape),
            "state": list(prediction[:, spatial_dim:].shape),
            "spatial": list(prediction[:, :spatial_dim].shape),
        }
    return _artifact(path, rows=int(len(prediction)), arrays=shape)


def _write_prediction_csv(
    path: Path,
    prediction: np.ndarray,
    target_time: float,
    *,
    state_only: bool,
    spatial_dim: int,
) -> dict[str, Any]:
    import pandas as pd

    state_dim = prediction.shape[1] if state_only else prediction.shape[1] - spatial_dim
    columns = (
        [f"state_pc_{index + 1:03d}" for index in range(state_dim)]
        if state_only
        else [f"spatial_{index + 1}" for index in range(spatial_dim)]
        + [f"state_pc_{index + 1:03d}" for index in range(state_dim)]
    )
    frame = pd.DataFrame(prediction, columns=columns)
    frame.insert(0, "time", float(target_time))
    frame.to_csv(path, index=False)
    return _artifact(path, rows=int(len(frame)), columns=int(len(frame.columns)))


def _official_plans(
    args: argparse.Namespace,
    data: TrajectoryInput,
    pairs: Sequence[Any],
    parameter_overrides: dict[str, Any],
    params: dict[str, Any],
) -> tuple[list[np.ndarray], list[dict[str, Any]], list[dict[str, Any]]]:
    transform = None
    if args.method == "moscot":
        transform = fit_block_balance(
            data.stages,
            state_weight=float(params["state_block_weight"]),
            spatial_weight=float(params["spatial_block_weight"]),
        )
    plans: list[np.ndarray] = []
    diagnostics: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    official_overrides = dict(parameter_overrides)
    for pair in pairs:
        plan, diag, meta = run_official_coupling(
            args.method,
            pair,
            args.representation,
            parameter_overrides=official_overrides,
            source_root=args.source_root,
            block_transform=transform,
        )
        plans.append(plan)
        diagnostics.append(
            {"from": pair.previous.time, "to": pair.following.time, **asdict(diag)}
        )
        metadata.append({"from": pair.previous.time, "to": pair.following.time, **meta})
    return plans, diagnostics, metadata


def _random_plans(
    args: argparse.Namespace,
    data: TrajectoryInput,
    pairs: Sequence[Any],
) -> tuple[list[np.ndarray], list[dict[str, Any]], list[np.ndarray]]:
    plans: list[np.ndarray] = []
    diagnostics: list[dict[str, Any]] = []
    target_indices: list[np.ndarray] = []
    for pair in pairs:
        rng = np.random.default_rng(
            stable_seed(
                args.seed,
                data.input_sha256,
                "random_independent_pairs",
                pair.previous.time,
                pair.following.time,
            )
        )
        raw, selected = random_independent_plan(pair, rng)
        plan, diag = validate_and_row_normalize(
            raw, (pair.previous.n_obs, pair.following.n_obs)
        )
        plans.append(plan)
        target_indices.append(selected)
        diagnostics.append(
            {"from": pair.previous.time, "to": pair.following.time, **asdict(diag)}
        )
    return plans, diagnostics, target_indices


def _coupling_predictions(
    data: TrajectoryInput,
    plans: Sequence[np.ndarray],
    roster_indices: np.ndarray,
    *,
    state_only: bool,
) -> tuple[dict[float, np.ndarray], dict[str, Any]]:
    if data.mode == "loto":
        pair = data.loto_pair()
        if len(plans) != 1:
            raise ValueError("LOTO requires exactly one bracket coupling")
        fitted = (
            project_loto_state(pair, plans[0])
            if state_only
            else project_loto_joint(pair, plans[0])
        )
        return {float(pair.target_time): take_roster(fitted, roster_indices)}, {
            "composition": "single bracket coupling; no target rows present",
            "interpolation_alpha": float(pair.interpolation_alpha),
        }

    pairs = data.adjacent_pairs()
    if len(plans) != len(pairs):
        raise ValueError(
            "No-holdout requires one coupling for every adjacent stage pair"
        )
    outputs: dict[float, np.ndarray] = {}
    histories: dict[str, Any] = {}
    declared_targets = data.contract.get("full_data_targets")
    allowed_targets = (
        None
        if declared_targets is None
        else tuple(float(value) for value in declared_targets)
    )
    composed: np.ndarray | None = None
    cumulative_history: list[dict[str, float]] = []
    for index, pair in enumerate(pairs, start=1):
        if composed is None:
            composed, step_history = compose_row_plans([plans[0]])
        else:
            composed, step_history = compose_row_plans([composed, plans[index - 1]])
        step_record = dict(step_history[-1])
        step_record["step"] = float(index)
        cumulative_history.append(step_record)
        fitted = (
            project_composed_state(pair.following, composed)
            if state_only
            else project_composed_joint(pair.following, composed)
        )
        if allowed_targets is None or any(
            np.isclose(pair.following.time, value) for value in allowed_targets
        ):
            outputs[float(pair.following.time)] = take_roster(fitted, roster_indices)
        histories[str(pair.following.time)] = {
            "formula": " @ ".join(
                f"P{_time_token(value.previous.time)}{_time_token(value.following.time)}"
                for value in pairs[:index]
            ),
            "shape": list(composed.shape),
            "normalization_history": list(cumulative_history),
        }
    return outputs, {
        "composition": "all adjacent couplings composed from fixed initial-stage roster",
        "targets": histories,
    }


def _control_predictions(
    args: argparse.Namespace,
    data: TrajectoryInput,
    roster_indices: np.ndarray,
) -> tuple[dict[float, np.ndarray], dict[str, Any], list[np.ndarray]]:
    if args.method == "linear_centroid_shift":
        if data.mode == "loto":
            fitted, shift = linear_centroid_loto(data.loto_pair())
            return (
                {float(data.target_time): take_roster(fitted, roster_indices)},
                {
                    "control": "linear_centroid_shift",
                    "conditioning": "target-removed bracket centroid difference times alpha",
                    "shift_l2": float(np.linalg.norm(shift)),
                },
                [],
            )
        fitted_by_time, shifts = linear_centroid_trajectory(data.stages)
        declared_targets = data.contract.get("full_data_targets")
        if declared_targets is not None:
            fitted_by_time = {
                time: points
                for time, points in fitted_by_time.items()
                if any(np.isclose(time, float(value)) for value in declared_targets)
            }
        return (
            {
                time: take_roster(points, roster_indices)
                for time, points in fitted_by_time.items()
            },
            {
                "control": "linear_centroid_shift",
                "conditioning": "sequential adjacent centroid shifts applied to the same t0 roster",
                "shift_l2_by_transition": [
                    float(np.linalg.norm(shift)) for shift in shifts
                ],
            },
            shifts,
        )
    pairs = [data.loto_pair()] if data.mode == "loto" else list(data.adjacent_pairs())
    plans, diagnostics, selections = _random_plans(args, data, pairs)
    predictions, composition = _coupling_predictions(
        data, plans, roster_indices, state_only=False
    )
    return (
        predictions,
        {
            "control": "random_independent_pairs",
            "conditioning": "independent one-hot coupling per observed transition",
            "coupling_diagnostics": diagnostics,
            **composition,
        },
        plans,
    )


def _write_outputs(
    args: argparse.Namespace,
    data: TrajectoryInput,
    predictions: dict[float, np.ndarray],
    roster_ids: np.ndarray,
    *,
    state_only: bool,
    provenance: dict[str, Any],
    output_scope: str,
    hybrid_adapter: bool,
) -> dict[str, Any]:
    spatial_dim = data.stages[0].spatial.shape[1]
    per_time: dict[str, Any] = {}
    combined: list[np.ndarray] = []
    combined_times: list[np.ndarray] = []
    for target_time in sorted(predictions):
        prediction = predictions[target_time]
        if len(prediction) != data.prediction_n:
            raise ValueError(
                f"Prediction at {target_time} has {len(prediction)} rows; contract requires {data.prediction_n}"
            )
        token = _time_token(target_time)
        target_dir = (
            args.output_dir if data.mode == "loto" else args.output_dir / f"t{token}"
        )
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / "prediction.npz"
        record = _save_prediction(
            path,
            prediction,
            target_time,
            state_only=state_only,
            spatial_dim=spatial_dim,
            source_row_ids=roster_ids,
        )
        if args.write_csv:
            record["csv"] = _write_prediction_csv(
                target_dir / "prediction.csv",
                prediction,
                target_time,
                state_only=state_only,
                spatial_dim=spatial_dim,
            )
        source_time = (
            float(data.loto_pair().previous.time)
            if data.mode == "loto"
            else float(data.time_values[0])
        )
        target_summary = {
            "schema_version": "2.0.0",
            "status": "complete",
            "dataset": data.contract.get(
                "dataset_id", data.contract.get("dataset", "unspecified")
            ),
            "method": args.method,
            "representation": args.representation,
            "primary_benchmark_eligible": bool(
                args.representation == "matched_state_spatial"
                and args.method != "spatrack"
            ),
            "track": "loto" if data.mode == "loto" else "full_data",
            "regime": "loto" if data.mode == "loto" else "full_data",
            "target_time": float(target_time),
            "source_time": source_time,
            "prediction_n": int(len(prediction)),
            "input_manifest_sha256": provenance.get("input_manifest_sha256"),
            "training_reference_sha256": provenance.get("training_reference_sha256"),
            "source_roster_sha256": provenance.get("source_roster_sha256"),
            "train_h5ad_sha256": data.input_sha256,
            "output_scope": output_scope,
            "native_vs_adapter": _native_vs_adapter(
                output_scope=output_scope,
                hybrid_adapter=hybrid_adapter,
            ),
            "native_mass": False,
            "native_growth": False,
            "weights_are_unnormalised": False,
            "target_n_used_for_prediction": False,
            "truth_artifact_opened": False,
            "source_roster_row_ids_sha256": ids_sha256(roster_ids),
            "prediction": record,
        }
        summary_path = target_dir / "summary.json"
        _write_json(summary_path, target_summary)
        record["summary"] = _artifact(summary_path)
        per_time[str(target_time)] = record
        combined.append(prediction)
        combined_times.append(np.full(len(prediction), target_time, dtype=np.float32))

    points = np.vstack(combined).astype(np.float32)
    times = np.concatenate(combined_times)
    # A multi-time archive is useful for trajectory inspection, but is not named
    # prediction.npz so the single-target evaluator cannot mistake it for a case.
    # Never reuse the evaluator-facing single-target filename here.  In LOTO
    # there is only one target, but rewriting prediction.npz would invalidate
    # the SHA already frozen in that target's summary.
    combined_path = args.output_dir / "trajectory_prediction.npz"
    if state_only:
        np.savez_compressed(
            combined_path,
            state=points,
            state_pca=points,
            time=times,
            source_row_id=np.tile(roster_ids.astype(str), len(combined)),
        )
    else:
        np.savez_compressed(
            combined_path,
            points=points,
            spatial=points[:, :spatial_dim],
            state=points[:, spatial_dim:],
            state_pca=points[:, spatial_dim:],
            time=times,
            source_row_id=np.tile(roster_ids.astype(str), len(combined)),
        )
    return {
        "trajectory_prediction": _artifact(
            combined_path,
            rows=int(len(points)),
            target_times=sorted(float(value) for value in predictions),
        ),
        "prediction_by_time": per_time,
    }


def _native_vs_adapter(*, output_scope: str, hybrid_adapter: bool) -> str:
    if hybrid_adapter:
        return "hybrid_coupling_adapter"
    if output_scope in {"state", "state_only", "native_state", "native-state"}:
        return "native_state"
    return "explicit_control"


def _write_summary(args: argparse.Namespace, manifest: dict[str, Any]) -> None:
    outputs = manifest.get("outputs", {})
    prediction_by_time = outputs.get("prediction_by_time", {})
    target_values = sorted(float(value) for value in prediction_by_time)
    source_time = manifest.get("anchors", {}).get("source_stage")
    output_scope = str(manifest.get("output_scope", {}).get("scope", ""))
    hybrid_adapter = bool(manifest.get("output_scope", {}).get("hybrid_adapter", False))
    payload = {
        "schema_version": "2.0.0",
        "status": manifest["status"],
        "dataset": manifest.get("dataset"),
        "method": args.method,
        "representation": args.representation,
        "regime": manifest.get("protocol", {}).get("mode"),
        "target_times": target_values,
        "target_time": target_values[0] if len(target_values) == 1 else None,
        "source_time": source_time,
        "prediction_n_per_time": manifest.get("protocol", {}).get("prediction_n"),
        "source_roster_row_ids_sha256": manifest.get("anchors", {}).get(
            "source_roster_row_ids_sha256"
        ),
        "output_scope": output_scope,
        "native_vs_adapter": (
            "not_applicable"
            if manifest["status"] == "not_applicable"
            else _native_vs_adapter(
                output_scope=output_scope,
                hybrid_adapter=hybrid_adapter,
            )
        ),
        "primary_benchmark_eligible": manifest.get("primary_benchmark_eligible", False),
        "input_manifest_sha256": manifest.get("input", {}).get("input_manifest_sha256"),
        "training_reference_sha256": manifest.get("input", {}).get(
            "training_reference_sha256"
        ),
        "prediction": outputs.get("trajectory_prediction"),
        "truth_artifact_opened": False,
        "target_n_used_for_prediction": False,
        "native_mass": False,
        "native_growth": False,
    }
    if manifest["status"] == "not_applicable":
        payload["reason"] = manifest.get("not_applicable_reason")
    destination = (
        args.output_dir / "run_summary.json"
        if payload.get("regime") == "loto" and prediction_by_time
        else args.output_dir / "summary.json"
    )
    _write_json(destination, payload)


def execute_run(args: argparse.Namespace) -> dict[str, Any]:
    if args.max_fit_n <= 0:
        raise ValueError("--max-fit-n must be positive")
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.evaluation_mode = (
        "no-holdout" if args.evaluation_mode == "full-data" else args.evaluation_mode
    )
    if args.evaluation_mode == "loto" and args.target_time is None:
        raise ValueError("LOTO requires --target-time")

    method_spec = get_method_spec(args.method)
    rep_spec = representation_spec(args.method, args.representation)
    parameter_overrides = _load_parameter_overrides(args.params_json)
    params = resolve_parameters(args.method, args.representation, parameter_overrides)
    keys = InputKeys(
        expression=args.expression_key,
        spatial=args.spatial_key,
        state=args.state_key,
        time=args.time_key,
        row_id=args.row_id_key,
        contract_uns=args.contract_uns_key,
    )
    data = load_trajectory(
        args.input_h5ad,
        mode=args.evaluation_mode,
        target_time=args.target_time if args.evaluation_mode == "loto" else None,
        max_fit_n=args.max_fit_n,
        seed=args.seed,
        keys=keys,
        require_expression=args.representation == "native_gene_sensitivity",
        allow_unverified_preprocessing=args.allow_unverified_preprocessing,
    )
    provenance = _verify_external_provenance(args, data.input_sha256)
    source_stage = (
        data.loto_pair().previous
        if data.mode == "loto"
        else data.stage(data.time_values[0])
    )
    roster_indices, roster_ids = _canonical_source_roster(
        data, source_stage, provenance, args
    )
    manifest = _base_manifest(
        args,
        data,
        method_spec,
        rep_spec,
        params,
        provenance,
        roster_indices,
        roster_ids,
    )
    manifest_path = args.output_dir / "run_manifest.json"
    manifest["outputs"] = {
        "anchor_selection": _save_anchor_selection(args.output_dir, data),
        "source_roster": _write_roster(
            args.output_dir, source_stage.time, roster_indices, roster_ids
        ),
    }

    if not rep_spec.get("applicable", False):
        manifest["status"] = "not_applicable"
        manifest["not_applicable_reason"] = rep_spec.get("reason")
        manifest["prediction_written"] = False
        manifest["dry_run"] = bool(args.dry_run)
        _write_json(manifest_path, manifest)
        _write_summary(args, manifest)
        return manifest

    if args.dry_run:
        probe = dependency_probe(args.method, args.source_root)
        manifest["dependency"] = probe
        if not probe["available"] and args.require_dependency_in_dry_run:
            manifest["status"] = "dry_run_failed_dependency_missing"
            _write_json(manifest_path, manifest)
            _write_summary(args, manifest)
            raise DependencyUnavailable(
                f"Official dependency is unavailable; see {manifest_path}"
            )
        manifest["status"] = (
            "dry_run_ready" if probe["available"] else "dry_run_dependency_missing"
        )
        manifest["prediction_written"] = False
        manifest["dry_run"] = True
        _write_json(manifest_path, manifest)
        _write_summary(args, manifest)
        return manifest

    pairs = [data.loto_pair()] if data.mode == "loto" else list(data.adjacent_pairs())
    plans: list[np.ndarray] = []
    if args.method in OFFICIAL_METHODS:
        plans, diagnostics, official_runs = _official_plans(
            args, data, pairs, parameter_overrides, params
        )
        state_only = args.method == "wot"
        predictions, composition = _coupling_predictions(
            data, plans, roster_indices, state_only=state_only
        )
        manifest["official_runs"] = official_runs
        manifest["coupling_diagnostics"] = diagnostics
        manifest["composition"] = composition
    else:
        state_only = False
        predictions, control_meta, plans = _control_predictions(
            args, data, roster_indices
        )
        manifest["control_run"] = control_meta

    if args.save_coupling and plans:
        coupling_path = args.output_dir / "couplings.npz"
        arrays = {
            f"P_{_time_token(pair.previous.time)}_{_time_token(pair.following.time)}": plan
            for pair, plan in zip(pairs, plans)
        }
        np.savez_compressed(coupling_path, **arrays)
        manifest["outputs"]["couplings"] = _artifact(
            coupling_path,
            shapes={key: list(value.shape) for key, value in arrays.items()},
        )

    manifest["outputs"].update(
        _write_outputs(
            args,
            data,
            predictions,
            roster_ids,
            state_only=state_only,
            provenance=provenance,
            output_scope=manifest["output_scope"]["scope"],
            hybrid_adapter=manifest["output_scope"]["hybrid_adapter"],
        )
    )
    manifest["status"] = "complete"
    manifest["dry_run"] = False
    manifest["prediction_written"] = True
    _write_json(manifest_path, manifest)
    _write_summary(args, manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "registry":
        payload = get_method_spec(args.method) if args.method else load_registry()
        print(json.dumps(_jsonable(payload), indent=2, sort_keys=True))
        return 0
    try:
        manifest = execute_run(args)
    except (BaselineError, ValueError, json.JSONDecodeError) as exc:
        if getattr(args, "output_dir", None) is not None:
            failure = {
                "schema_version": "2.0.0",
                "status": "failed",
                "method": getattr(args, "method", None),
                "representation": getattr(args, "representation", None),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "surrogate_attempted": False,
                "truth_artifact_opened": False,
            }
            _write_json(
                Path(args.output_dir).resolve() / "failure_manifest.json", failure
            )
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(_jsonable(manifest), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
