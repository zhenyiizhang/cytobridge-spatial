#!/usr/bin/env python3
"""Run the release-grade zebrafish daughter-noise sensitivity analysis.

Every observed interval is an independent, one-sided forecast.  For interval
``[t, t + 1]``, the simulation starts from *all* real cells observed at ``t``
and produces the midpoint plus, optionally, a generated endpoint.  It is not
conditioned on the following observed endpoint, is not a global-t0 rollout,
uses no spatial warp, and is not lineage-continuous across intervals.

The scientific grid is intentionally frozen: daughter noise
``{0, 0.01, 0.03, 0.06}``, paired seeds ``42..46``, ``dt=resample_dt=0.05``,
continuous diffusion ``sigma=0.03``, growth multiplier ``1``, interaction
group size ``1024``, and a fail-fast particle ceiling of ``100000``.  Input,
checkpoint, score, classifier, and acceptance-report hashes are mandatory.
Interaction grouping uses a dedicated RNG seeded as ``paired_seed + 10000``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence
from urllib.parse import quote

import anndata as ad
import numpy as np
import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import CytoBridge as cb  # noqa: E402


SCHEMA_VERSION = 1
ANALYSIS_ID = "zebrafish_interval_local_daughter_noise_sensitivity"
OBSERVED_TIMES = (0.0, 1.0, 2.0, 3.0, 4.0)
INTERVALS = tuple(zip(OBSERVED_TIMES[:-1], OBSERVED_TIMES[1:]))
NOISE_VALUES = (0.0, 0.01, 0.03, 0.06)
PAIRED_SEEDS = (42, 43, 44, 45, 46)
DT = 0.05
CONTINUOUS_DIFFUSION_SIGMA = 0.03
GROWTH_ALPHA = 1.0
INTERACTION_M = 1024
INTERACTION_SEED_OFFSET = 10_000
RESAMPLE_DT = 0.05
MAX_PARTICLES = 100_000
CLASSIFIER_KNN_NEIGHBORS = 10
ZEBRAFISH_LATENT_DIM = 50
ZEBRAFISH_JOINT_DIM = 52
OT_MAX_POINTS = 1024
LINEAGE_TRANSITION_COLUMNS = (
    "daughter_noise_std",
    "seed",
    "interaction_seed",
    "interval_start",
    "interval_end",
    "forecast_time",
    "forecast_role",
    "state_source",
    "following_endpoint_conditioned",
    "source_lineage_id",
    "lineage_namespace",
    "source_obs_id",
    "source_celltype",
    "target_celltype",
    "descendant_count",
    "fraction_within_lineage",
)
TRAJECTORY_SCOPE = (
    "independent observed-anchored interval-local one-sided forecasts; each "
    "interval starts from all real cells at its left observed anchor; not "
    "conditioned on the following observed endpoint; not global-t0; not "
    "lineage-continuous across intervals; no spatial warp"
)


@dataclass(frozen=True)
class ForecastFrame:
    """One generated frame and its interval-local lineage roster."""

    interval_start: float
    interval_end: float
    forecast_time: float
    forecast_role: str
    daughter_noise_std: float
    seed: int
    points: np.ndarray
    lineage_ids: np.ndarray
    labels: np.ndarray
    n_source: int


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalise_digest(value: str, *, name: str) -> str:
    digest = str(value).strip().lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError(f"{name} must be a 64-character lowercase SHA-256 digest")
    return digest


def _verified_file(
    path: str | Path,
    expected_sha256: str,
    *,
    description: str,
) -> tuple[Path, str]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Missing {description}: {resolved}")
    expected = _normalise_digest(expected_sha256, name=f"{description} SHA-256")
    observed = _sha256(resolved)
    if observed != expected:
        raise RuntimeError(
            f"{description} SHA-256 mismatch: expected {expected}, observed {observed}"
        )
    return resolved, observed


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_ready(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _stable_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        _json_ready(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _git_state() -> dict[str, object]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=all"],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}


def _load_acceptance_report(
    path: str | Path,
    expected_sha256: str,
) -> tuple[Path, str, Mapping[str, Any]]:
    report_path, digest = _verified_file(
        path,
        expected_sha256,
        description="canonical four-dataset acceptance report",
    )
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot read acceptance JSON {report_path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise RuntimeError("Canonical acceptance report must contain a JSON object")
    datasets = payload.get("datasets")
    zebrafish = datasets.get("zebrafish") if isinstance(datasets, Mapping) else None
    if payload.get("status") != "PASS" or not isinstance(zebrafish, Mapping):
        raise RuntimeError(
            "Acceptance report must record overall PASS and datasets.zebrafish"
        )
    if zebrafish.get("status") != "PASS":
        raise RuntimeError("Acceptance report does not record zebrafish status PASS")
    return report_path, digest, payload


def _prepare_output_dir(path: str | Path) -> Path:
    output_dir = Path(path).expanduser().resolve()
    if output_dir.exists():
        if not output_dir.is_dir():
            raise FileExistsError(
                f"Output path exists and is not a directory: {output_dir}"
            )
        if any(output_dir.iterdir()):
            raise FileExistsError(
                "Output directory must be new or empty; refusing to mix or overwrite "
                f"a prior sensitivity run: {output_dir}"
            )
    else:
        output_dir.mkdir(parents=True)
    return output_dir


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(_json_ready(payload), indent=2, sort_keys=True, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _joint_features(
    adata: ad.AnnData,
    *,
    spatial_key: str,
    latent_key: str,
) -> np.ndarray:
    missing = [key for key in (spatial_key, latent_key) if key not in adata.obsm]
    if missing:
        raise KeyError(f"Aligned H5AD lacks required obsm entries: {missing}")
    spatial = np.asarray(adata.obsm[spatial_key], dtype=np.float32)
    latent = np.asarray(adata.obsm[latent_key], dtype=np.float32)
    if spatial.ndim != 2 or spatial.shape[1] != 2:
        raise ValueError(
            f"Expected N x 2 aligned spatial coordinates, got {spatial.shape}"
        )
    if latent.ndim != 2 or latent.shape[0] != spatial.shape[0] or latent.shape[1] == 0:
        raise ValueError(
            "Latent coordinates must be non-empty and row-aligned with spatial data"
        )
    if latent.shape[1] != ZEBRAFISH_LATENT_DIM:
        raise ValueError(
            "Final Zebrafish aligned input must contain exactly "
            f"{ZEBRAFISH_LATENT_DIM} latent PCs, got {latent.shape[1]}"
        )
    joint = np.hstack((spatial, latent)).astype(np.float32, copy=False)
    if joint.shape[1] != ZEBRAFISH_JOINT_DIM:
        raise RuntimeError(
            f"Expected the canonical {ZEBRAFISH_JOINT_DIM}-dimensional joint state"
        )
    if not np.isfinite(joint).all():
        raise ValueError("Aligned joint model state contains non-finite values")
    return joint


def _parsed_times(adata: ad.AnnData, *, time_key: str) -> np.ndarray:
    if time_key not in adata.obs:
        raise KeyError(f"Aligned H5AD lacks obs[{time_key!r}]")
    try:
        values = np.asarray(
            [cb.tl.parse_time_value(value) for value in adata.obs[time_key].values],
            dtype=np.float64,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Cannot parse every value in obs[{time_key!r}]") from exc
    if not np.isfinite(values).all():
        raise ValueError("Observed time values must all be finite")
    observed = tuple(float(value) for value in np.unique(values))
    if len(observed) != len(OBSERVED_TIMES) or not np.allclose(
        observed, OBSERVED_TIMES, rtol=0.0, atol=1e-9
    ):
        raise ValueError(
            f"Expected zebrafish observed times {list(OBSERVED_TIMES)}, got {list(observed)}"
        )
    return values


def _validate_aligned_input(
    adata: ad.AnnData,
    *,
    time_key: str,
    annotation_key: str,
    spatial_key: str,
    latent_key: str,
) -> tuple[np.ndarray, np.ndarray]:
    if adata.n_obs == 0 or adata.n_vars == 0:
        raise ValueError("Aligned H5AD must contain observations and variables")
    if not adata.obs_names.is_unique:
        raise ValueError("Aligned H5AD obs_names must be unique for lineage provenance")
    if any(not str(value) for value in adata.obs_names):
        raise ValueError("Aligned H5AD contains an empty obs_name")
    if annotation_key not in adata.obs:
        raise KeyError(f"Aligned H5AD lacks obs[{annotation_key!r}]")
    if adata.obs[annotation_key].isna().any():
        raise ValueError(
            f"Aligned H5AD obs[{annotation_key!r}] contains missing labels"
        )
    for provenance_key in ("preprocess_info", "interaction_graph"):
        if not isinstance(adata.uns.get(provenance_key), Mapping):
            raise ValueError(
                f"Aligned H5AD lacks mapping-valued uns[{provenance_key!r}] provenance"
            )
    joint = _joint_features(adata, spatial_key=spatial_key, latent_key=latent_key)
    times = _parsed_times(adata, time_key=time_key)
    for anchor, _ in INTERVALS:
        if not np.isclose(times, anchor, rtol=0.0, atol=1e-9).any():
            raise ValueError(f"No observed cells found at left anchor t={anchor:g}")
    return joint, times


def _classifier_source_fingerprint(
    adata: ad.AnnData,
    *,
    label_col: str,
    time_key: str,
    classifier_inputs: np.ndarray,
) -> str:
    """Reproduce the public classifier cache's source fingerprint exactly."""

    digest = hashlib.sha1()
    digest.update(
        f"{adata.n_obs}|{adata.n_vars}|{label_col}|{time_key}".encode("utf-8")
    )
    for values in (
        adata.obs_names.astype(str),
        adata.obs[label_col].astype(str).values,
        adata.obs[time_key].astype(str).values,
    ):
        digest.update("\x1f".join(map(str, values)).encode("utf-8"))
    inputs = np.ascontiguousarray(classifier_inputs, dtype=np.float32)
    digest.update(str(inputs.shape).encode("utf-8"))
    digest.update(inputs.tobytes())
    return digest.hexdigest()


def _validate_classifier_contract(
    cached: Any,
    adata: ad.AnnData,
    joint: np.ndarray,
    times: np.ndarray,
    *,
    time_key: str,
    annotation_key: str,
    spatial_key: str,
    latent_key: str,
) -> dict[str, Any]:
    if str(cached.label_col) != annotation_key:
        raise RuntimeError(
            f"Classifier label column mismatch: {cached.label_col!r} != {annotation_key!r}"
        )
    if not bool(cached.include_time_feature):
        raise RuntimeError("Zebrafish sensitivity classifier must include time")
    if int(joint.shape[1]) != ZEBRAFISH_JOINT_DIM:
        raise RuntimeError("Aligned model state does not match the canonical dimension")
    if int(cached.feature_dim) != int(joint.shape[1]):
        raise RuntimeError(
            "Classifier feature contract mismatch: the final classifier must use "
            f"all spatial2+latent50={joint.shape[1]} dimensions, got {cached.feature_dim}"
        )
    expected_feature_cols = ("samples",) + tuple(
        f"x{index}" for index in range(1, ZEBRAFISH_JOINT_DIM + 1)
    )
    if tuple(cached.feature_cols) != expected_feature_cols:
        raise RuntimeError(
            "Classifier feature columns must be exactly samples,x1..x52 in order"
        )
    metadata = cached.metadata
    if not isinstance(metadata, Mapping):
        raise RuntimeError("Classifier cache lacks mapping-valued metadata")
    expected_classifier_metadata = {
        "best_epoch_metric": "bacc",
        "train_on_full_data": False,
        "refit_on_full_data_after_selection": False,
        "stratify_split": True,
        "strict_stratification": True,
        "selection_scope": "held_out_validation_phase_a",
        "seed": 42,
    }
    metadata_mismatches = {
        key: (expected, metadata.get(key))
        for key, expected in expected_classifier_metadata.items()
        if metadata.get(key) != expected
    }
    if int(metadata.get("version", -1)) < 8 or metadata_mismatches:
        raise RuntimeError(
            "Classifier does not match the final held-out-selection contract: "
            f"version={metadata.get('version')!r}, mismatches={metadata_mismatches}"
        )
    class_split = metadata.get("class_split")
    if not isinstance(class_split, Mapping) or class_split.get("strategy") != (
        "held_out_train_validation"
    ):
        raise RuntimeError("Classifier lacks the held-out class-split provenance")
    per_class_counts = class_split.get("per_class_counts")
    if not isinstance(per_class_counts, Mapping):
        raise RuntimeError("Classifier lacks per-class split counts")
    source = metadata.get("source") if isinstance(metadata, Mapping) else None
    if not isinstance(source, Mapping):
        raise RuntimeError("Classifier cache lacks source provenance")
    expected_source = {
        "kind": "AnnData",
        "n_obs": int(adata.n_obs),
        "n_vars": int(adata.n_vars),
        "obsm_key": latent_key,
        "spatial_key": spatial_key,
        "concat_spatial": True,
    }
    mismatches = {
        key: (expected, source.get(key))
        for key, expected in expected_source.items()
        if source.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(f"Classifier source provenance mismatch: {mismatches}")
    feature_selection = metadata.get("feature_selection")
    if (
        not isinstance(feature_selection, Mapping)
        or feature_selection.get("kind") != "leading_joint_dimensions"
        or int(feature_selection.get("n_features", -1)) != ZEBRAFISH_JOINT_DIM
        or feature_selection.get("requested_n_features") is not None
    ):
        raise RuntimeError(
            "Final classifier must record the uncapped full joint feature contract "
            "(leading_joint_dimensions, n_features=52, requested_n_features=None)"
        )
    classifier_inputs = np.hstack(
        (
            times.astype(np.float32).reshape(-1, 1),
            joint,
        )
    ).astype(np.float32, copy=False)
    fingerprint = _classifier_source_fingerprint(
        adata,
        label_col=annotation_key,
        time_key=time_key,
        classifier_inputs=classifier_inputs,
    )
    if source.get("fingerprint") != fingerprint:
        raise RuntimeError(
            "Classifier source fingerprint does not match the aligned H5AD; "
            "refusing a classifier trained on another roster or feature matrix"
        )
    observed_classes = set(adata.obs[annotation_key].astype(str))
    classifier_classes = set(map(str, cached.label_encoder.classes_))
    if classifier_classes != observed_classes:
        raise RuntimeError(
            "Classifier class roster does not exactly match aligned annotations: "
            f"missing={sorted(observed_classes - classifier_classes)}, "
            f"extra={sorted(classifier_classes - observed_classes)}"
        )
    split_classes = set(map(str, per_class_counts))
    if split_classes != observed_classes:
        raise RuntimeError(
            "Classifier split-count roster does not exactly match aligned "
            f"annotations: missing={sorted(observed_classes - split_classes)}, "
            f"extra={sorted(split_classes - observed_classes)}"
        )
    observed_counts = adata.obs[annotation_key].astype(str).value_counts().to_dict()
    for label, expected_total in observed_counts.items():
        counts = per_class_counts.get(label)
        if not isinstance(counts, Mapping):
            raise RuntimeError(f"Classifier split lacks class {label!r}")
        total = int(counts.get("total", -1))
        train = int(counts.get("train", -1))
        validation = int(counts.get("validation", -1))
        if total != int(expected_total) or train + validation != total or train <= 0:
            raise RuntimeError(
                f"Classifier split counts disagree for class {label!r}: {counts}"
            )
        if total >= 2 and validation <= 0:
            raise RuntimeError(
                f"Classifier class {label!r} has no held-out validation support"
            )
    metrics: dict[str, float] = {}
    for name in ("accuracy", "balanced_accuracy"):
        value = getattr(cached, name)
        if value is None or not np.isfinite(value) or not 0.0 <= float(value) <= 1.0:
            raise RuntimeError(f"Classifier {name} must be a finite held-out metric")
        metrics[name] = float(value)
    return {
        "feature_dim": int(cached.feature_dim),
        "feature_cols": list(cached.feature_cols),
        "label_col": str(cached.label_col),
        "include_time_feature": True,
        "source_fingerprint": fingerprint,
        "cache_protocol_version": int(metadata["version"]),
        "selection_scope": str(metadata["selection_scope"]),
        "class_split_strategy": str(class_split["strategy"]),
        **metrics,
    }


def _validate_learned_model_contract(
    loaded: Any, *, expected_dim: int
) -> dict[str, Any]:
    config = loaded.config
    model_config = config.get("model") if isinstance(config, Mapping) else None
    if not isinstance(model_config, Mapping):
        raise RuntimeError("Model config lacks a mapping-valued model section")
    components = tuple(map(str, model_config.get("components", ())))
    required = {"velocity", "growth", "score", "interaction"}
    if not required.issubset(components):
        raise RuntimeError(
            f"Final sensitivity model lacks required components: {sorted(required - set(components))}"
        )
    if str(model_config.get("interaction_type", "")).lower() != "gnn":
        raise RuntimeError(
            "Final sensitivity model must use the learned GNN interaction"
        )
    if int(model_config.get("interaction_group_size", -1)) != INTERACTION_M:
        raise RuntimeError(
            "Final model interaction_group_size must match the frozen sensitivity "
            f"contract {INTERACTION_M}"
        )
    interaction = model_config.get("interaction_net")
    if not isinstance(interaction, Mapping):
        raise RuntimeError("Model config lacks interaction_net provenance")
    if str(interaction.get("edge_prior_mode", "")).lower() != "learned":
        raise RuntimeError("Final sensitivity model must use edge_prior_mode='learned'")
    threshold = interaction.get("edge_predictor_thre")
    cutoff = interaction.get("cutoff")
    if threshold is None or not np.isfinite(float(threshold)):
        raise RuntimeError(
            "Learned model lacks a finite frozen edge-predictor threshold"
        )
    if cutoff is None or not np.isfinite(float(cutoff)) or float(cutoff) <= 0:
        raise RuntimeError("Learned model lacks a finite positive spatial cutoff")
    if str(loaded.weight_stage) != "Finetune":
        raise RuntimeError(
            f"Expected final Finetune checkpoint, loaded stage {loaded.weight_stage!r}"
        )
    if loaded.score_path is None or loaded.score_stage is None:
        raise RuntimeError("Final sensitivity model must load a score checkpoint")
    model = loaded.model
    if int(getattr(model, "latent_dim", -1)) != int(expected_dim):
        raise RuntimeError(
            f"Loaded model dimension {getattr(model, 'latent_dim', None)} != {expected_dim}"
        )
    interaction_net = getattr(model, "interaction_net", None)
    if (
        interaction_net is None
        or str(getattr(interaction_net, "edge_prior_mode", "")).lower() != "learned"
    ):
        raise RuntimeError("Loaded model does not contain the learned interaction gate")
    if not hasattr(interaction_net, "link_predictor"):
        raise RuntimeError(
            "Loaded learned interaction gate has no embedded link predictor"
        )
    loaded_model_config = getattr(model, "config", None)
    loaded_interaction_config = (
        loaded_model_config.get("interaction_net")
        if isinstance(loaded_model_config, Mapping)
        else None
    )
    if (
        not isinstance(loaded_interaction_config, Mapping)
        or loaded_interaction_config.get("load_edge_predictor_from_path") is not False
    ):
        raise RuntimeError(
            "Loaded model does not prove that the learned edge predictor was "
            "embedded in the hash-bound Finetune checkpoint; refusing an "
            "untracked external predictor dependency"
        )
    if not np.isclose(
        float(getattr(interaction_net, "edge_predictor_thre", np.nan)),
        float(threshold),
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError("Loaded edge-predictor threshold differs from model config")
    if not np.isclose(
        float(getattr(interaction_net, "cutoff", np.nan)),
        float(cutoff),
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError("Loaded interaction cutoff differs from model config")
    if int(getattr(model, "interaction_group_size", -1)) != INTERACTION_M:
        raise RuntimeError("Loaded interaction group size differs from frozen contract")
    return {
        "components": list(components),
        "interaction_type": "gnn",
        "edge_prior_mode": "learned",
        "edge_predictor_source": "embedded_in_weight_checkpoint",
        "edge_predictor_threshold": float(threshold),
        "spatial_cutoff": float(cutoff),
        "interaction_group_size": INTERACTION_M,
        "weight_stage": str(loaded.weight_stage),
        "score_stage": str(loaded.score_stage),
    }


def _set_seed(seed: int) -> None:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _lineage_namespace(anchor_time: float, source_obs_id: str) -> str:
    encoded = quote(str(source_obs_id), safe="")
    return f"anchor_time={float(anchor_time):g}/source_obs_id={encoded}"


def _interaction_seed(seed: int) -> int:
    return int(seed) + INTERACTION_SEED_OFFSET


def _validate_generated_frame(
    points: np.ndarray,
    lineage_ids: np.ndarray,
    labels: np.ndarray,
    *,
    n_source: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float32)
    lineage_ids = np.asarray(lineage_ids)
    labels = np.asarray(labels).astype(str)
    if points.ndim != 2 or not np.isfinite(points).all():
        raise RuntimeError("Generated states must be a finite two-dimensional array")
    if lineage_ids.ndim != 1 or lineage_ids.shape[0] != points.shape[0]:
        raise RuntimeError("Generated lineage IDs must have one value per particle")
    if lineage_ids.dtype == object:
        if any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            for value in lineage_ids
        ):
            raise RuntimeError("Generated lineage IDs must be integers")
    elif not np.issubdtype(lineage_ids.dtype, np.integer):
        raise RuntimeError("Generated lineage IDs must be integers")
    lineage_ids = lineage_ids.astype(np.int64, copy=False)
    if labels.ndim != 1 or labels.shape[0] != points.shape[0]:
        raise RuntimeError("Generated labels must have one value per particle")
    if lineage_ids.size and (
        int(lineage_ids.min()) < 0 or int(lineage_ids.max()) >= int(n_source)
    ):
        raise RuntimeError("Generated lineage ID is outside the interval anchor roster")
    return points, lineage_ids, labels


def _simulate_interval(
    *,
    x0: np.ndarray,
    runtime: Any,
    classifier: Any,
    interval_start: float,
    interval_end: float,
    daughter_noise_std: float,
    seed: int,
    include_end: bool,
    device: str,
) -> list[ForecastFrame]:
    midpoint = float((interval_start + interval_end) / 2.0)
    time_points = [float(interval_start), midpoint]
    roles = ["midpoint_one_sided_forecast"]
    if include_end:
        time_points.append(float(interval_end))
        roles.append("endpoint_one_sided_forecast")
    lineage_roster = np.arange(x0.shape[0], dtype=np.int64)
    _set_seed(seed)
    points_by_time, lineage_by_time = cb.tl.simulate_sde_points_split_from_x0(
        x0=x0,
        f_net=runtime.f_net,
        score_net=runtime.score_net,
        ts_points=time_points,
        dt=DT,
        sigma=CONTINUOUS_DIFFUSION_SIGMA,
        sigma_by_dim=None,
        growth_alpha=GROWTH_ALPHA,
        interaction_m=INTERACTION_M,
        device=device,
        verbose=True,
        resample_dt=RESAMPLE_DT,
        max_particles=MAX_PARTICLES,
        daughter_noise_std=float(daughter_noise_std),
        interaction_seed=_interaction_seed(seed),
        initial_lineage_ids=lineage_roster,
        return_lineage_ids=True,
    )
    if len(points_by_time) != len(time_points) or len(lineage_by_time) != len(
        time_points
    ):
        raise RuntimeError("Split-SDE returned an unexpected number of interval frames")
    initial_points = np.asarray(points_by_time[0], dtype=np.float32)
    initial_lineages = np.asarray(lineage_by_time[0], dtype=np.int64)
    if initial_points.shape != x0.shape or not np.array_equal(
        initial_lineages, lineage_roster
    ):
        raise RuntimeError(
            "Split-SDE changed the observed left-anchor roster at t_start"
        )
    if not np.allclose(initial_points, x0, rtol=0.0, atol=0.0):
        raise RuntimeError(
            "Split-SDE did not preserve the exact observed left-anchor state"
        )

    frames: list[ForecastFrame] = []
    for index, (forecast_time, role) in enumerate(zip(time_points[1:], roles), start=1):
        points = np.asarray(points_by_time[index], dtype=np.float32)
        predicted = cb.tl.predict_labels_for_points(
            points=points,
            time_value=float(forecast_time),
            model=classifier.model,
            label_encoder=classifier.label_encoder,
            feature_dim=int(classifier.feature_dim),
            device=device,
            knn_neighbors=CLASSIFIER_KNN_NEIGHBORS,
            include_time_feature=True,
            spatial_indices=(0, 1),
        )
        points, lineage_ids, predicted = _validate_generated_frame(
            points,
            np.asarray(lineage_by_time[index]),
            np.asarray(predicted),
            n_source=int(x0.shape[0]),
        )
        frames.append(
            ForecastFrame(
                interval_start=float(interval_start),
                interval_end=float(interval_end),
                forecast_time=float(forecast_time),
                forecast_role=role,
                daughter_noise_std=float(daughter_noise_std),
                seed=int(seed),
                points=points,
                lineage_ids=lineage_ids,
                labels=predicted,
                n_source=int(x0.shape[0]),
            )
        )
    return frames


def _common_frame_fields(frame: ForecastFrame) -> dict[str, object]:
    return {
        "daughter_noise_std": float(frame.daughter_noise_std),
        "seed": int(frame.seed),
        "interaction_seed": _interaction_seed(frame.seed),
        "interval_start": float(frame.interval_start),
        "interval_end": float(frame.interval_end),
        "forecast_time": float(frame.forecast_time),
        "forecast_role": str(frame.forecast_role),
        "state_source": "generated_interval_local_one_sided",
        "following_endpoint_conditioned": False,
    }


def _composition_rows(frame: ForecastFrame) -> list[dict[str, object]]:
    common = _common_frame_fields(frame)
    values, counts = np.unique(frame.labels.astype(str), return_counts=True)
    total = int(counts.sum())
    if total == 0:
        return [
            {
                **common,
                "celltype": "__empty__",
                "count": 0,
                "fraction": 0.0,
                "n_particles": 0,
                "population_empty": True,
            }
        ]
    return [
        {
            **common,
            "celltype": str(value),
            "count": int(count),
            "fraction": float(count / total),
            "n_particles": total,
            "population_empty": False,
        }
        for value, count in zip(values, counts)
    ]


def _lineage_rows(
    frame: ForecastFrame,
    *,
    source_obs_ids: np.ndarray,
    source_labels: np.ndarray,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if len(source_obs_ids) != frame.n_source or len(source_labels) != frame.n_source:
        raise ValueError("Source lineage metadata does not match the interval roster")
    common = _common_frame_fields(frame)
    descendants = np.bincount(frame.lineage_ids, minlength=frame.n_source)
    descendant_rows = []
    for lineage_id in range(frame.n_source):
        source_obs_id = str(source_obs_ids[lineage_id])
        descendant_rows.append(
            {
                **common,
                "source_lineage_id": int(lineage_id),
                "lineage_namespace": _lineage_namespace(
                    frame.interval_start, source_obs_id
                ),
                "source_obs_id": source_obs_id,
                "source_celltype": str(source_labels[lineage_id]),
                "descendant_count": int(descendants[lineage_id]),
                "lineage_alive": bool(descendants[lineage_id] > 0),
            }
        )

    if frame.lineage_ids.size == 0:
        return descendant_rows, []
    transitions = pd.DataFrame(
        {
            "source_lineage_id": frame.lineage_ids,
            "target_celltype": frame.labels.astype(str),
        }
    )
    grouped = (
        transitions.groupby(
            ["source_lineage_id", "target_celltype"], observed=True, sort=True
        )
        .size()
        .rename("descendant_count")
        .reset_index()
    )
    transition_rows = []
    for row in grouped.itertuples(index=False):
        lineage_id = int(row.source_lineage_id)
        count = int(row.descendant_count)
        source_obs_id = str(source_obs_ids[lineage_id])
        transition_rows.append(
            {
                **common,
                "source_lineage_id": lineage_id,
                "lineage_namespace": _lineage_namespace(
                    frame.interval_start, source_obs_id
                ),
                "source_obs_id": source_obs_id,
                "source_celltype": str(source_labels[lineage_id]),
                "target_celltype": str(row.target_celltype),
                "descendant_count": count,
                "fraction_within_lineage": float(count / descendants[lineage_id]),
            }
        )
    return descendant_rows, transition_rows


def _label_counts(labels: np.ndarray) -> dict[str, int]:
    values, counts = np.unique(np.asarray(labels).astype(str), return_counts=True)
    return {str(value): int(count) for value, count in zip(values, counts)}


def _total_variation(base: Mapping[str, int], other: Mapping[str, int]) -> float:
    base_total = int(sum(base.values()))
    other_total = int(sum(other.values()))
    if base_total == 0 and other_total == 0:
        return 0.0
    if base_total == 0 or other_total == 0:
        return 1.0
    labels = set(base) | set(other)
    return float(
        0.5
        * sum(
            abs(base.get(label, 0) / base_total - other.get(label, 0) / other_total)
            for label in labels
        )
    )


def _ot_seed(base: ForecastFrame, other: ForecastFrame, *, space: str) -> int:
    payload = {
        "namespace": "zebrafish-interval-daughter-noise-paired-ot-v1",
        "space": str(space),
        "seed": int(base.seed),
        "interaction_seed": _interaction_seed(base.seed),
        "interval_start": float(base.interval_start),
        "interval_end": float(base.interval_end),
        "forecast_time": float(base.forecast_time),
        "forecast_role": str(base.forecast_role),
        "baseline_daughter_noise_std": float(base.daughter_noise_std),
        "daughter_noise_std": float(other.daughter_noise_std),
    }
    return int.from_bytes(
        hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).digest()[:4],
        "little",
    )


def _paired_wasserstein_metrics(
    base: ForecastFrame, other: ForecastFrame
) -> dict[str, object]:
    """Exact empirical W1/W2 on retained joint and spatial supports."""

    if (
        base.points.ndim != 2
        or other.points.ndim != 2
        or base.points.shape[1] != other.points.shape[1]
        or base.points.shape[1] < 2
    ):
        raise ValueError(
            "Paired Wasserstein frames must share a feature dimension of at least two"
        )
    output: dict[str, object] = {"ot_max_points": OT_MAX_POINTS}
    for space, columns in (
        ("joint", slice(0, base.points.shape[1])),
        ("spatial", slice(0, 2)),
    ):
        seed = _ot_seed(base, other, space=space)
        output[f"{space}_ot_random_seed"] = seed
        if base.points.shape[0] == 0 or other.points.shape[0] == 0:
            output.update(
                {
                    f"{space}_w1_from_noise0": None,
                    f"{space}_w2_from_noise0": None,
                    f"{space}_ot_noise0_points": 0,
                    f"{space}_ot_noise_points": 0,
                    f"{space}_ot_status": "not_computable_empty_population",
                }
            )
            continue
        metrics = cb.tl.compute_distribution_metrics(
            other.points[:, columns],
            base.points[:, columns],
            max_ot_points=OT_MAX_POINTS,
            random_seed=seed,
        )
        w1 = float(metrics["w1"])
        w2 = float(metrics["w2"])
        if not np.isfinite([w1, w2]).all():
            raise RuntimeError(f"Paired {space} Wasserstein metric is non-finite")
        output.update(
            {
                f"{space}_w1_from_noise0": w1,
                f"{space}_w2_from_noise0": w2,
                f"{space}_ot_noise0_points": int(metrics["ot_observed_points"]),
                f"{space}_ot_noise_points": int(metrics["ot_predicted_points"]),
                f"{space}_ot_status": "complete",
            }
        )
    return output


def _lineage_fate_tv(base: ForecastFrame, other: ForecastFrame) -> tuple[float, float]:
    """Compare interval-local per-source fates, treating extinction as a fate."""

    target_labels = sorted(set(base.labels.astype(str)) | set(other.labels.astype(str)))
    target_index = {label: index for index, label in enumerate(target_labels)}
    extinct_index = len(target_labels)

    def fate_matrix(frame: ForecastFrame) -> np.ndarray:
        matrix = np.zeros((frame.n_source, len(target_labels) + 1), dtype=np.float64)
        for lineage_id, label in zip(frame.lineage_ids, frame.labels):
            matrix[int(lineage_id), target_index[str(label)]] += 1.0
        totals = matrix[:, :extinct_index].sum(axis=1)
        alive = totals > 0
        matrix[alive, :extinct_index] /= totals[alive, None]
        matrix[~alive, extinct_index] = 1.0
        return matrix

    source_tv = 0.5 * np.abs(fate_matrix(other) - fate_matrix(base)).sum(axis=1)
    return float(source_tv.mean()), float(source_tv.max(initial=0.0))


def _paired_delta_row(base: ForecastFrame, other: ForecastFrame) -> dict[str, object]:
    identity_base = (
        base.seed,
        base.interval_start,
        base.interval_end,
        base.forecast_time,
        base.forecast_role,
        base.n_source,
    )
    identity_other = (
        other.seed,
        other.interval_start,
        other.interval_end,
        other.forecast_time,
        other.forecast_role,
        other.n_source,
    )
    if identity_base != identity_other or base.daughter_noise_std != 0.0:
        raise ValueError("Paired delta frames do not share the noise-0 run identity")
    baseline_descendants = np.bincount(
        base.lineage_ids, minlength=base.n_source
    ).astype(np.int64)
    other_descendants = np.bincount(other.lineage_ids, minlength=other.n_source).astype(
        np.int64
    )
    baseline_alive = baseline_descendants > 0
    other_alive = other_descendants > 0
    alive_union = int(np.logical_or(baseline_alive, other_alive).sum())
    alive_intersection = int(np.logical_and(baseline_alive, other_alive).sum())
    count_delta = int(other.points.shape[0] - base.points.shape[0])
    lineage_mean_tv, lineage_max_tv = _lineage_fate_tv(base, other)
    return {
        "baseline_daughter_noise_std": 0.0,
        "daughter_noise_std": float(other.daughter_noise_std),
        "seed": int(base.seed),
        "interaction_seed": _interaction_seed(base.seed),
        "interval_start": float(base.interval_start),
        "interval_end": float(base.interval_end),
        "forecast_time": float(base.forecast_time),
        "forecast_role": str(base.forecast_role),
        "n_source_lineages": int(base.n_source),
        "baseline_n_particles": int(base.points.shape[0]),
        "n_particles": int(other.points.shape[0]),
        "particle_count_delta": count_delta,
        "particle_count_relative_delta": (
            0.0
            if base.points.shape[0] == 0 and other.points.shape[0] == 0
            else (
                None
                if base.points.shape[0] == 0
                else float(count_delta / base.points.shape[0])
            )
        ),
        "composition_total_variation": _total_variation(
            _label_counts(base.labels), _label_counts(other.labels)
        ),
        "mean_absolute_lineage_descendant_count_delta": float(
            np.abs(other_descendants - baseline_descendants).mean()
        ),
        "max_absolute_lineage_descendant_count_delta": int(
            np.abs(other_descendants - baseline_descendants).max(initial=0)
        ),
        "fraction_lineages_same_descendant_count": float(
            np.mean(other_descendants == baseline_descendants)
        ),
        "lineage_alive_status_agreement": float(np.mean(other_alive == baseline_alive)),
        "lineage_survival_jaccard": (
            1.0 if alive_union == 0 else float(alive_intersection / alive_union)
        ),
        "lineage_fate_mean_total_variation_from_noise0": lineage_mean_tv,
        "lineage_fate_max_total_variation_from_noise0": lineage_max_tv,
        "paired_common_seed": True,
        **_paired_wasserstein_metrics(base, other),
    }


def _paired_deltas(frames: Sequence[ForecastFrame]) -> list[dict[str, object]]:
    indexed = {
        (
            frame.daughter_noise_std,
            frame.seed,
            frame.interval_start,
            frame.forecast_time,
            frame.forecast_role,
        ): frame
        for frame in frames
    }
    if len(indexed) != len(frames):
        raise RuntimeError(
            "Duplicate forecast-frame identity in paired sensitivity output"
        )
    rows = []
    for interval_start, interval_end in INTERVALS:
        midpoint = (interval_start + interval_end) / 2.0
        roles = [(midpoint, "midpoint_one_sided_forecast")]
        if any(
            frame.interval_start == interval_start
            and frame.forecast_role == "endpoint_one_sided_forecast"
            for frame in frames
        ):
            roles.append((interval_end, "endpoint_one_sided_forecast"))
        for seed in PAIRED_SEEDS:
            for time_value, role in roles:
                baseline_key = (0.0, seed, interval_start, time_value, role)
                if baseline_key not in indexed:
                    raise RuntimeError(
                        f"Missing noise-0 paired baseline {baseline_key}"
                    )
                for noise in NOISE_VALUES[1:]:
                    other_key = (noise, seed, interval_start, time_value, role)
                    if other_key not in indexed:
                        raise RuntimeError(
                            f"Missing paired daughter-noise frame {other_key}"
                        )
                    rows.append(
                        _paired_delta_row(indexed[baseline_key], indexed[other_key])
                    )
    return rows


def _time_tag(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def _save_raw_interval(
    output_dir: Path,
    *,
    frames: Sequence[ForecastFrame],
    source_obs_ids: np.ndarray,
    source_labels: np.ndarray,
) -> Path:
    first = frames[0]
    arrays: dict[str, np.ndarray] = {
        "schema_version": np.asarray(SCHEMA_VERSION, dtype=np.int64),
        "trajectory_scope": np.asarray(TRAJECTORY_SCOPE),
        "interval_start": np.asarray(first.interval_start, dtype=np.float64),
        "interval_end": np.asarray(first.interval_end, dtype=np.float64),
        "daughter_noise_std": np.asarray(first.daughter_noise_std, dtype=np.float64),
        "seed": np.asarray(first.seed, dtype=np.int64),
        "interaction_seed": np.asarray(_interaction_seed(first.seed), dtype=np.int64),
        "source_lineage_id": np.arange(first.n_source, dtype=np.int64),
        "source_obs_id": np.asarray(source_obs_ids).astype(str),
        "source_celltype": np.asarray(source_labels).astype(str),
        "lineage_namespace": np.asarray(
            [
                _lineage_namespace(first.interval_start, source_obs_id)
                for source_obs_id in source_obs_ids
            ]
        ),
    }
    for frame in frames:
        prefix = "midpoint" if frame.forecast_role.startswith("midpoint") else "end"
        arrays[f"{prefix}_time"] = np.asarray(frame.forecast_time, dtype=np.float64)
        arrays[f"{prefix}_points"] = frame.points.astype(np.float32, copy=False)
        arrays[f"{prefix}_lineage_ids"] = frame.lineage_ids.astype(np.int64, copy=False)
        arrays[f"{prefix}_labels"] = frame.labels.astype(str)
    path = (
        output_dir
        / "raw"
        / (
            f"anchor_{_time_tag(first.interval_start)}_noise_"
            f"{_time_tag(first.daughter_noise_std)}_seed_{first.seed}.npz"
        )
    )
    _atomic_npz(path, **arrays)
    return path


def _artifact(path: Path, *, root: Path) -> dict[str, object]:
    resolved = path.resolve()
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        raise RuntimeError(f"Missing or empty output artifact: {resolved}")
    return {
        "path": resolved.relative_to(root.resolve()).as_posix(),
        "size_bytes": int(resolved.stat().st_size),
        "sha256": _sha256(resolved),
    }


def _implementation_hashes() -> dict[str, str]:
    paths = (
        Path(__file__).resolve(),
        REPO_ROOT / "CytoBridge/tl/downstream/simulation.py",
        REPO_ROOT / "CytoBridge/tl/downstream/checkpoint.py",
        REPO_ROOT / "CytoBridge/tl/downstream/classification.py",
        REPO_ROOT / "CytoBridge/tl/downstream/evaluation.py",
        REPO_ROOT / "CytoBridge/tl/downstream/runtime.py",
        REPO_ROOT / "CytoBridge/tl/core/interaction.py",
        REPO_ROOT / "CytoBridge/tl/core/models.py",
        REPO_ROOT / "CytoBridge/tl/graph/spatial_gnn.py",
    )
    return {path.relative_to(REPO_ROOT).as_posix(): _sha256(path) for path in paths}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aligned-h5ad", required=True, type=Path)
    parser.add_argument("--expected-aligned-sha256", required=True)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--expected-model-config-sha256", required=True)
    parser.add_argument("--expected-weight-sha256", required=True)
    parser.add_argument("--expected-score-sha256", required=True)
    parser.add_argument("--classifier-cache", required=True, type=Path)
    parser.add_argument("--expected-classifier-sha256", required=True)
    parser.add_argument("--acceptance-report", required=True, type=Path)
    parser.add_argument("--expected-acceptance-sha256", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--time-key", default="time_point_processed")
    parser.add_argument("--annotation-key", default="Annotation")
    parser.add_argument("--spatial-key", default="spatial_aligned")
    parser.add_argument("--latent-key", default="X_latent")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--include-end",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Also emit the generated right endpoint. It remains one-sided and is "
            "never replaced by or conditioned on observed endpoint cells."
        ),
    )
    return parser


def run(args: argparse.Namespace) -> Path:
    aligned_path, aligned_sha = _verified_file(
        args.aligned_h5ad,
        args.expected_aligned_sha256,
        description="aligned zebrafish H5AD",
    )
    model_dir = Path(args.model_dir).expanduser().resolve()
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Missing model directory: {model_dir}")
    config_path, config_sha = _verified_file(
        model_dir / "config.yaml",
        args.expected_model_config_sha256,
        description="model config",
    )
    classifier_path, classifier_sha = _verified_file(
        args.classifier_cache,
        args.expected_classifier_sha256,
        description="trajectory classifier cache",
    )
    acceptance_path, acceptance_sha, acceptance = _load_acceptance_report(
        args.acceptance_report, args.expected_acceptance_sha256
    )

    adata = ad.read_h5ad(aligned_path)
    joint, times = _validate_aligned_input(
        adata,
        time_key=args.time_key,
        annotation_key=args.annotation_key,
        spatial_key=args.spatial_key,
        latent_key=args.latent_key,
    )
    loaded = cb.tl.load_dynamical_model_from_dir(
        model_dir, dim=int(joint.shape[1]), device=args.device, stage="Finetune"
    )
    weight_path, weight_sha = _verified_file(
        loaded.weight_path,
        args.expected_weight_sha256,
        description="final Finetune checkpoint",
    )
    if loaded.score_path is None:
        raise RuntimeError("Loaded final model has no score checkpoint")
    score_path, score_sha = _verified_file(
        loaded.score_path,
        args.expected_score_sha256,
        description="final score checkpoint",
    )
    model_contract = _validate_learned_model_contract(
        loaded, expected_dim=int(joint.shape[1])
    )
    runtime = cb.tl.build_dynamical_runtime(loaded)
    classifier = cb.tl.load_cached_mlp_classifier(
        str(classifier_path), device=args.device
    )
    classifier_contract = _validate_classifier_contract(
        classifier,
        adata,
        joint,
        times,
        time_key=args.time_key,
        annotation_key=args.annotation_key,
        spatial_key=args.spatial_key,
        latent_key=args.latent_key,
    )
    output_dir = _prepare_output_dir(args.output_dir)

    roster_records: list[dict[str, object]] = []
    composition_records: list[dict[str, object]] = []
    particle_records: list[dict[str, object]] = []
    descendant_records: list[dict[str, object]] = []
    transition_records: list[dict[str, object]] = []
    forecast_frames: list[ForecastFrame] = []
    raw_paths: list[Path] = []

    labels_all = adata.obs[args.annotation_key].astype(str).to_numpy()
    obs_ids_all = adata.obs_names.astype(str).to_numpy()
    for interval_start, interval_end in INTERVALS:
        anchor_mask = np.isclose(times, interval_start, rtol=0.0, atol=1e-9)
        x0 = joint[anchor_mask].copy()
        source_obs_ids = obs_ids_all[anchor_mask].copy()
        source_labels = labels_all[anchor_mask].copy()
        if x0.shape[0] > MAX_PARTICLES:
            raise RuntimeError(
                f"Observed anchor t={interval_start:g} has {x0.shape[0]} cells, "
                f"exceeding the fail-fast cap {MAX_PARTICLES}"
            )
        for lineage_id, (source_obs_id, source_label) in enumerate(
            zip(source_obs_ids, source_labels)
        ):
            roster_records.append(
                {
                    "interval_start": float(interval_start),
                    "interval_end": float(interval_end),
                    "source_lineage_id": int(lineage_id),
                    "lineage_namespace": _lineage_namespace(
                        interval_start, str(source_obs_id)
                    ),
                    "source_obs_id": str(source_obs_id),
                    "source_celltype": str(source_label),
                    "source_state": "observed_real_left_anchor",
                }
            )
        for seed in PAIRED_SEEDS:
            for noise in NOISE_VALUES:
                print(
                    "[daughter-noise] "
                    f"interval=[{interval_start:g},{interval_end:g}] "
                    f"noise={noise:g} seed={seed} n_source={x0.shape[0]}",
                    flush=True,
                )
                frames = _simulate_interval(
                    x0=x0,
                    runtime=runtime,
                    classifier=classifier,
                    interval_start=interval_start,
                    interval_end=interval_end,
                    daughter_noise_std=noise,
                    seed=seed,
                    include_end=bool(args.include_end),
                    device=args.device,
                )
                particle_records.append(
                    {
                        "daughter_noise_std": float(noise),
                        "seed": int(seed),
                        "interaction_seed": _interaction_seed(seed),
                        "interval_start": float(interval_start),
                        "interval_end": float(interval_end),
                        "time": float(interval_start),
                        "frame_role": "observed_left_anchor",
                        "state_source": "observed_real",
                        "n_particles": int(x0.shape[0]),
                    }
                )
                for frame in frames:
                    forecast_frames.append(frame)
                    composition_records.extend(_composition_rows(frame))
                    descendants, transitions = _lineage_rows(
                        frame,
                        source_obs_ids=source_obs_ids,
                        source_labels=source_labels,
                    )
                    descendant_records.extend(descendants)
                    transition_records.extend(transitions)
                    particle_records.append(
                        {
                            "daughter_noise_std": float(noise),
                            "seed": int(seed),
                            "interaction_seed": _interaction_seed(seed),
                            "interval_start": float(interval_start),
                            "interval_end": float(interval_end),
                            "time": float(frame.forecast_time),
                            "frame_role": str(frame.forecast_role),
                            "state_source": "generated_interval_local_one_sided",
                            "n_particles": int(frame.points.shape[0]),
                        }
                    )
                raw_paths.append(
                    _save_raw_interval(
                        output_dir,
                        frames=frames,
                        source_obs_ids=source_obs_ids,
                        source_labels=source_labels,
                    )
                )

    paired_delta_records = _paired_deltas(forecast_frames)
    tables = {
        "anchor_roster": (
            output_dir / "tables/anchor_roster.csv",
            pd.DataFrame(roster_records),
        ),
        "composition_long": (
            output_dir / "tables/composition_long.csv",
            pd.DataFrame(composition_records),
        ),
        "particle_counts": (
            output_dir / "tables/particle_counts.csv",
            pd.DataFrame(particle_records),
        ),
        "lineage_descendant_counts": (
            output_dir / "tables/lineage_descendant_counts.csv",
            pd.DataFrame(descendant_records),
        ),
        "lineage_transition_long": (
            output_dir / "tables/lineage_transition_long.csv",
            pd.DataFrame(transition_records, columns=LINEAGE_TRANSITION_COLUMNS),
        ),
        "noise0_paired_deltas": (
            output_dir / "tables/noise0_paired_deltas.csv",
            pd.DataFrame(paired_delta_records),
        ),
    }
    for path, frame in tables.values():
        _atomic_csv(frame, path)

    table_artifacts = {
        name: {
            **_artifact(path, root=output_dir),
            "row_count": int(len(frame)),
            "columns": list(frame.columns),
        }
        for name, (path, frame) in tables.items()
    }
    raw_artifacts = [_artifact(path, root=output_dir) for path in raw_paths]
    preprocess_provenance = {
        "preprocess_info_sha256": _stable_json_sha256(adata.uns["preprocess_info"]),
        "interaction_graph_sha256": _stable_json_sha256(adata.uns["interaction_graph"]),
    }
    covered_payload = {
        "schema_version": SCHEMA_VERSION,
        "analysis": ANALYSIS_ID,
        "trajectory_scope": TRAJECTORY_SCOPE,
        "claim_guardrails": {
            "following_endpoint_conditioned": False,
            "global_t0_rollout": False,
            "lineage_continuous_across_intervals": False,
            "spatial_warp_applied": False,
            "endpoint_is_observed_when_included": False,
            "lineage_join_contract": (
                "Lineage joins are valid only within one interval and require the "
                "full (anchor_time, source_obs_id) namespace. source_lineage_id "
                "alone must never be joined across intervals."
            ),
        },
        "inputs": {
            "aligned_h5ad": {
                "path": str(aligned_path),
                "sha256": aligned_sha,
                "n_obs": int(adata.n_obs),
                "n_vars": int(adata.n_vars),
                **preprocess_provenance,
            },
            "model_config": {"path": str(config_path), "sha256": config_sha},
            "weight_checkpoint": {"path": str(weight_path), "sha256": weight_sha},
            "score_checkpoint": {"path": str(score_path), "sha256": score_sha},
            "classifier_cache": {
                "path": str(classifier_path),
                "sha256": classifier_sha,
                **classifier_contract,
            },
            "canonical_acceptance_report": {
                "path": str(acceptance_path),
                "sha256": acceptance_sha,
                "required_exact": {
                    "status": "PASS",
                    "datasets": {"zebrafish": {"status": "PASS"}},
                },
                "observed_run_root": acceptance.get("run_root"),
            },
        },
        "model_contract": model_contract,
        "data_contract": {
            "time_key": str(args.time_key),
            "annotation_key": str(args.annotation_key),
            "spatial_key": str(args.spatial_key),
            "latent_key": str(args.latent_key),
            "joint_feature_dim": int(joint.shape[1]),
            "observed_times": list(OBSERVED_TIMES),
            "intervals": [list(interval) for interval in INTERVALS],
            "initial_roster": "all real observed cells at each interval's left anchor",
            "fresh_lineage_roster_per_interval": True,
            "lineage_namespace_fields": ["anchor_time", "source_obs_id"],
        },
        "simulation": {
            "daughter_noise_std": list(NOISE_VALUES),
            "paired_seeds": list(PAIRED_SEEDS),
            "paired_common_seed_with_noise0": True,
            "midpoint_forecast": True,
            "end_forecast_included": bool(args.include_end),
            "dt": DT,
            "resample_dt": RESAMPLE_DT,
            "continuous_diffusion_sigma": CONTINUOUS_DIFFUSION_SIGMA,
            "growth_alpha": GROWTH_ALPHA,
            "interaction_m": INTERACTION_M,
            "interaction_grouping_rng": {
                "stream": "dedicated_torch_generator",
                "paired_across_daughter_noise": True,
                "seed_formula": "paired_seed + interaction_seed_offset",
                "interaction_seed_offset": INTERACTION_SEED_OFFSET,
                "interaction_seed_by_paired_seed": {
                    str(seed): _interaction_seed(seed) for seed in PAIRED_SEEDS
                },
            },
            "max_particles": MAX_PARTICLES,
            "spatial_warp": False,
            "classifier_feature_contract": (
                "complete canonical joint state: aligned spatial2 + latent PCs 1..50"
            ),
            "classifier_feature_dim": ZEBRAFISH_JOINT_DIM,
            "classifier_knn_neighbors": CLASSIFIER_KNN_NEIGHBORS,
            "device": str(args.device),
        },
        "metric_contract": {
            "noise0_pairing": (
                "Every nonzero-noise frame is compared with daughter_noise_std=0 "
                "using the identical simulation seed, dedicated interaction-grouping "
                "seed, interval anchor, and forecast time."
            ),
            "distribution_spaces": {
                "joint": "canonical aligned spatial2 + all latent PCA coordinates",
                "spatial": "the leading two canonical aligned spatial coordinates",
            },
            "wasserstein": {
                "implementation": "cb.tl.compute_distribution_metrics",
                "metrics": ["W1", "W2"],
                "weights": "uniform empirical weights",
                "solver": "exact discrete OT on retained support",
                "max_points_per_cloud": OT_MAX_POINTS,
                "subsampling": (
                    "deterministic without replacement when a uniform cloud exceeds "
                    "the retained-support cap; the per-row seed and retained counts "
                    "are recorded"
                ),
                "empty_population_policy": (
                    "W1/W2 are blank and status is "
                    "not_computable_empty_population when either cloud is empty"
                ),
            },
            "composition": "total-variation distance from the paired noise-0 labels",
            "count": "absolute and relative particle-count delta from paired noise-0",
            "lineage": (
                "interval-local per-source fate total variation, including extinction "
                "as an explicit fate, plus descendant-count and survival summaries"
            ),
            "claim_scope": (
                "Sensitivity of one frozen learned checkpoint under inference-time "
                "daughter perturbation; not a hypothesis test or training-seed analysis."
            ),
        },
        "run_counts": {
            "independent_interval_noise_seed_runs": int(
                len(INTERVALS) * len(NOISE_VALUES) * len(PAIRED_SEEDS)
            ),
            "forecast_frames": int(len(forecast_frames)),
            "noise0_paired_delta_rows": int(len(paired_delta_records)),
            "raw_state_files": int(len(raw_paths)),
        },
        "outputs": {
            "tables": table_artifacts,
            "raw_states_saved": True,
            "raw_states": raw_artifacts,
        },
        "code": {
            "git": _git_state(),
            "implementation_sha256": _implementation_hashes(),
        },
    }
    signature_fields = list(covered_payload)
    manifest = {
        **covered_payload,
        "status": "complete",
        "completed_at": _utc_now(),
        "signature": {
            "algorithm": "sha256-canonical-json",
            "value": _stable_json_sha256(covered_payload),
            "covered_top_level_fields": signature_fields,
            "excludes": [
                "status",
                "completed_at",
                "signature",
                "run_manifest.json self hash",
            ],
            "coverage_note": (
                "Covers every scientific setting, exact input/code hash, and every "
                "recorded table/raw-state artifact hash. The adjacent SHA-256 sidecar "
                "covers the serialized manifest itself."
            ),
        },
    }
    manifest_path = output_dir / "run_manifest.json"
    _atomic_json(manifest, manifest_path)
    manifest_sha = _sha256(manifest_path)
    sidecar = output_dir / "run_manifest.sha256"
    sidecar.write_text(f"{manifest_sha}  run_manifest.json\n", encoding="utf-8")
    print(f"Saved {manifest_path} (sha256={manifest_sha})", flush=True)
    return manifest_path


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
