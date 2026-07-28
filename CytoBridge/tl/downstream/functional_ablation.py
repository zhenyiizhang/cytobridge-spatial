"""Frozen-checkpoint functional ablations for interaction models.

This module separates two questions that are often conflated:

* a *retrained architecture ablation* asks what model is learned when a
  component or prior is removed during fitting;
* a *frozen-checkpoint functional ablation* asks what an already fitted full
  model does when one inference-time contribution is disabled.

The runner below implements the second question.  Every condition uses the
same model object, starting cells, deterministic Euler integrator, and
interaction-grouping seed.  It never calls an optimizer or modifies model
parameters.

The ``lr_gate_off`` condition is therefore a *same-checkpoint all-spatial gate
counterfactual*, not "no-LR-prior training".  It deliberately increases the
admitted edge set to every within-cutoff candidate.  A matched-density gate
shuffle would answer a different question: it would keep the fitted gate's
edge count fixed while permuting which candidate pairs pass.  That additional
null is useful for separating LR identity from edge-density effects, but is
not silently substituted for the all-spatial counterfactual here.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch

from .checkpoint import LoadedModel
from .evaluation import DistributionEvaluationResult, evaluate_model_distributions
from .perturbation import FixedCohortRollout, deterministic_fixed_cohort_rollout

__all__ = [
    "ContinuousFrozenCheckpointAblationResult",
    "DEFAULT_FROZEN_ABLATION_CONDITIONS",
    "FrozenAblationCondition",
    "FrozenCheckpointAblationResult",
    "LRGateOverride",
    "resolve_frozen_ablation_condition",
    "run_continuous_frozen_checkpoint_ablations",
    "run_frozen_checkpoint_ablations",
    "save_continuous_frozen_checkpoint_ablation_result",
    "save_frozen_checkpoint_ablation_result",
    "temporary_lr_gate_mode",
]


@dataclass(frozen=True)
class FrozenAblationCondition:
    """One inference-only condition applied to a fitted full model."""

    name: str
    interaction_enabled: bool
    lr_gate_mode: str
    description: str


@dataclass(frozen=True)
class LRGateOverride:
    """Record of how the live interaction network was temporarily configured."""

    requested_mode: str
    implementation: str
    original_mode: str | None
    original_threshold: float | None


@dataclass(frozen=True)
class FrozenCheckpointAblationResult:
    """Matched deterministic rollouts and an audit manifest."""

    rollouts: Mapping[str, FixedCohortRollout]
    manifest: Mapping[str, Any]


@dataclass(frozen=True)
class ContinuousFrozenCheckpointAblationResult:
    """Continuous non-split weighted-SDE evaluations of one checkpoint.

    Each condition starts from the same sampled cells at the first observed
    time and is integrated once across the complete requested time grid.  The
    contained :class:`DistributionEvaluationResult` objects expose predicted
    points and native unnormalised growth weights at every observed endpoint.
    """

    evaluations: Mapping[str, DistributionEvaluationResult]
    manifest: Mapping[str, Any]


_CONDITIONS = {
    "full": FrozenAblationCondition(
        name="full",
        interaction_enabled=True,
        lr_gate_mode="trained",
        description="Fitted full model with its trained LR-informed edge gate.",
    ),
    "interaction_off": FrozenAblationCondition(
        name="interaction_off",
        interaction_enabled=False,
        lr_gate_mode="trained",
        description=(
            "Same fitted model, but the interaction velocity is not evaluated "
            "or added during rollout."
        ),
    ),
    "lr_gate_off": FrozenAblationCondition(
        name="lr_gate_off",
        interaction_enabled=True,
        lr_gate_mode="all_spatial",
        description=(
            "Same-checkpoint all-spatial gate counterfactual: retain the "
            "fitted interaction GNN and spatial cutoff, but admit every "
            "within-cutoff candidate edge at inference. This is not "
            "no-LR-prior training and does not match the fitted edge density."
        ),
    ),
}

DEFAULT_FROZEN_ABLATION_CONDITIONS = tuple(_CONDITIONS)


def _unwrap_model(model_or_loaded):
    return (
        model_or_loaded.model
        if isinstance(model_or_loaded, LoadedModel)
        else model_or_loaded
    )


def resolve_frozen_ablation_condition(name: str) -> FrozenAblationCondition:
    """Resolve a public condition name and fail on ambiguous aliases."""

    normalized = str(name).strip().lower().replace("-", "_")
    try:
        return _CONDITIONS[normalized]
    except KeyError as exc:
        raise ValueError(
            f"Unknown frozen ablation condition '{name}'. Expected one of "
            f"{list(DEFAULT_FROZEN_ABLATION_CONDITIONS)}."
        ) from exc


def _require_interaction_network(model):
    components = set(getattr(model, "components", ()))
    interaction_net = getattr(model, "interaction_net", None)
    if "interaction" not in components or interaction_net is None:
        raise TypeError(
            "Frozen interaction ablation requires a trained full model with "
            "an interaction component."
        )
    return interaction_net


@contextmanager
def temporary_lr_gate_mode(
    model_or_loaded,
    mode: str,
) -> Iterator[LRGateOverride]:
    """Temporarily select the fitted or all-spatial edge gate.

    ``all_spatial`` does not remove the interaction GNN.  It only bypasses the
    pretrained LR-informed link-predictor decision while preserving the
    spatial cutoff, RBF features, attention layers, readout, and all fitted
    weights.

    Current models expose ``edge_prior_mode`` directly.  Legacy released
    models do not; for those, a zero sigmoid threshold is the equivalent
    inference-only implementation because every within-cutoff candidate edge
    passes.  All live attributes are restored on exit, including when rollout
    raises.
    """

    requested = str(mode).strip().lower().replace("-", "_")
    if requested not in {"trained", "all_spatial"}:
        raise ValueError("mode must be 'trained' or 'all_spatial'.")

    model = _unwrap_model(model_or_loaded)
    interaction_net = _require_interaction_network(model)
    original_mode = (
        str(interaction_net.edge_prior_mode)
        if hasattr(interaction_net, "edge_prior_mode")
        else None
    )
    original_threshold = (
        float(interaction_net.edge_predictor_thre)
        if hasattr(interaction_net, "edge_predictor_thre")
        else None
    )

    if requested == "trained":
        if original_mode == "all_spatial":
            raise ValueError(
                "The supplied checkpoint is already configured with "
                "edge_prior_mode='all_spatial'; it cannot define a fitted-gate "
                "'full' control."
            )
        if not hasattr(interaction_net, "link_predictor"):
            raise TypeError(
                "The interaction network has no fitted link_predictor, so the "
                "trained LR-gate control is unavailable."
            )
        yield LRGateOverride(
            requested_mode=requested,
            implementation="unchanged_fitted_gate",
            original_mode=original_mode,
            original_threshold=original_threshold,
        )
        return

    if not hasattr(interaction_net, "link_predictor"):
        raise TypeError(
            "The interaction network has no fitted link_predictor. Use the "
            "trained full checkpoint, not a separately trained no-LR model, "
            "for a frozen-checkpoint LR-gate ablation."
        )

    if hasattr(interaction_net, "edge_prior_mode"):
        implementation = "edge_prior_mode=all_spatial"
        interaction_net.edge_prior_mode = "all_spatial"
    elif hasattr(interaction_net, "edge_predictor_thre"):
        implementation = "legacy_edge_predictor_threshold=0"
        interaction_net.edge_predictor_thre = 0.0
    else:
        raise TypeError(
            "Unsupported interaction network: no inference-time LR-gate hook "
            "was found."
        )

    try:
        yield LRGateOverride(
            requested_mode=requested,
            implementation=implementation,
            original_mode=original_mode,
            original_threshold=original_threshold,
        )
    finally:
        if original_mode is not None:
            interaction_net.edge_prior_mode = original_mode
        if original_threshold is not None:
            interaction_net.edge_predictor_thre = original_threshold


@contextmanager
def _preserve_random_state() -> Iterator[None]:
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_states = (
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    )
    try:
        yield
    finally:
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def _array_sha256(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("utf-8"))
    digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _file_sha256(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_manifest(model_or_loaded) -> dict[str, Any]:
    if not isinstance(model_or_loaded, LoadedModel):
        return {
            "loader": "in_memory_model",
            "weight_stage": None,
            "weight_path": None,
            "weight_sha256": None,
            "score_stage": None,
            "score_path": None,
            "score_sha256": None,
        }
    weight_path = Path(model_or_loaded.weight_path)
    score_path = (
        Path(model_or_loaded.score_path)
        if model_or_loaded.score_path is not None
        else None
    )
    return {
        "loader": "LoadedModel",
        "weight_stage": str(model_or_loaded.weight_stage),
        "weight_path": str(weight_path),
        "weight_sha256": _file_sha256(weight_path),
        "score_stage": (
            str(model_or_loaded.score_stage)
            if model_or_loaded.score_stage is not None
            else None
        ),
        "score_path": (
            str(score_path) if score_path is not None else None
        ),
        "score_sha256": _file_sha256(score_path),
    }


def run_frozen_checkpoint_ablations(
    points: np.ndarray,
    model_or_loaded,
    *,
    start_time: float,
    end_time: float,
    dt: float,
    interaction_m: int,
    grouping_seed: int,
    conditions: Sequence[str] = DEFAULT_FROZEN_ABLATION_CONDITIONS,
    device: str = "cpu",
    spatial_dim: int = 2,
) -> FrozenCheckpointAblationResult:
    """Run matched functional ablations of one fitted full checkpoint.

    This is deliberately a fixed-cohort, ``sigma=0`` experiment.  Growth,
    particle splitting, resampling, and spatial warping are absent, so a
    difference between conditions is attributable to the inference-time
    interaction term or LR gate rather than changing cell identities.
    """

    initial = np.asarray(points, dtype=np.float32)
    if initial.ndim != 2 or initial.shape[0] < 2:
        raise ValueError("points must be an N x D matrix with N >= 2.")
    if not np.isfinite(initial).all():
        raise ValueError("points contain non-finite values.")
    if int(spatial_dim) < 0 or int(spatial_dim) > initial.shape[1]:
        raise ValueError(
            f"spatial_dim must be in [0, {initial.shape[1]}], got {spatial_dim}."
        )

    model = _unwrap_model(model_or_loaded)
    _require_interaction_network(model)
    resolved = [resolve_frozen_ablation_condition(name) for name in conditions]
    names = [condition.name for condition in resolved]
    if len(set(names)) != len(names):
        raise ValueError(f"Duplicate frozen ablation conditions: {names}.")
    if not resolved:
        raise ValueError("At least one frozen ablation condition is required.")

    rollouts: dict[str, FixedCohortRollout] = {}
    condition_manifest: dict[str, Any] = {}
    with _preserve_random_state():
        for condition in resolved:
            with temporary_lr_gate_mode(
                model,
                condition.lr_gate_mode,
            ) as gate_override:
                rollout = deterministic_fixed_cohort_rollout(
                    initial.copy(),
                    model,
                    start_time=float(start_time),
                    end_time=float(end_time),
                    dt=float(dt),
                    interaction_m=int(interaction_m),
                    grouping_seed=int(grouping_seed),
                    device=str(device),
                    spatial_dim=int(spatial_dim),
                    interaction_enabled=condition.interaction_enabled,
                    sigma=0.0,
                )
            if not np.array_equal(rollout.points[0], initial):
                raise RuntimeError(
                    f"Condition '{condition.name}' did not preserve the exact "
                    "supplied initial cohort."
                )
            rollouts[condition.name] = rollout
            condition_manifest[condition.name] = {
                "formal_label": (
                    "same_checkpoint_all_spatial_gate_counterfactual"
                    if condition.name == "lr_gate_off"
                    else f"same_checkpoint_{condition.name}"
                ),
                "interaction_enabled": condition.interaction_enabled,
                "lr_gate_mode": condition.lr_gate_mode,
                "description": condition.description,
                "gate_implementation": gate_override.implementation,
                "weights_retrained": False,
            }

    reference_times = next(iter(rollouts.values())).times
    for name, rollout in rollouts.items():
        if not np.array_equal(rollout.times, reference_times):
            raise RuntimeError(f"Condition '{name}' used a different time grid.")

    manifest = {
        "analysis": "frozen_checkpoint_functional_ablation",
        "conditions": condition_manifest,
        "condition_order": names,
        "checkpoint": _checkpoint_manifest(model_or_loaded),
        "matched_controls": {
            "same_model_object": True,
            "same_checkpoint_weights": True,
            "same_initial_points": True,
            "same_initial_points_sha256": _array_sha256(initial),
            "same_grouping_seed": True,
            "grouping_seed": int(grouping_seed),
            "same_time_grid": True,
            "start_time": float(start_time),
            "end_time": float(end_time),
            "dt_max": float(dt),
            "interaction_group_size": int(interaction_m),
        },
        "rollout": {
            "integrator": "fixed_cohort_deterministic_euler",
            "sigma": 0.0,
            "growth_or_particle_resampling": False,
            "spatial_warp": False,
            "n_initial_cells": int(initial.shape[0]),
            "state_dim": int(initial.shape[1]),
            "spatial_dim": int(spatial_dim),
        },
        "estimand": (
            "Inference-time functional dependence of the fitted full model; "
            "this does not estimate the effect of retraining an architecture "
            "without the component or prior."
        ),
        "lr_gate_null_scope": {
            "implemented": "all within-cutoff spatial candidates are admitted",
            "not_implemented": (
                "matched-density LR-gate shuffle; that null would preserve "
                "the number of fitted-gate edges while permuting edge identity"
            ),
        },
    }
    return FrozenCheckpointAblationResult(
        rollouts=rollouts,
        manifest=manifest,
    )


def _resolve_complete_observed_time_grid(
    adata,
    *,
    time_points: Sequence[float] | None,
    time_key: str | None,
) -> tuple[str, tuple[float, ...]]:
    from .downstream_data import infer_time_key, parse_time_value

    resolved_time_key = infer_time_key(adata.obs, preferred=time_key)
    observed = np.asarray(
        [parse_time_value(value) for value in adata.obs[resolved_time_key]],
        dtype=np.float64,
    )
    observed_unique = tuple(sorted(float(value) for value in np.unique(observed)))
    if len(observed_unique) < 2:
        raise ValueError(
            "Continuous functional ablation requires at least two observed "
            "time points."
        )
    if time_points is None:
        requested = observed_unique
    else:
        requested = tuple(float(value) for value in time_points)
        if len(requested) < 2:
            raise ValueError("time_points must contain at least two values.")
        if not np.isfinite(np.asarray(requested, dtype=np.float64)).all():
            raise ValueError("time_points must be finite.")
        if any(right <= left for left, right in zip(requested, requested[1:])):
            raise ValueError("time_points must be strictly increasing.")
        missing = [
            value
            for value in requested
            if not np.any(np.isclose(observed, value, rtol=0.0, atol=1e-9))
        ]
        if missing:
            raise ValueError(
                "Every requested evaluation time must be observed; missing "
                f"{missing}."
            )
    if not np.isclose(
        requested[0],
        observed_unique[0],
        rtol=0.0,
        atol=1e-9,
    ):
        raise ValueError(
            "Continuous whole-trajectory ablation must start at the earliest "
            f"observed time ({observed_unique[0]}), got {requested[0]}. "
            "Intermediate observed states must not be used to restart it."
        )
    if time_points is not None and (
        len(requested) != len(observed_unique)
        or not np.allclose(
            np.asarray(requested, dtype=np.float64),
            np.asarray(observed_unique, dtype=np.float64),
            rtol=0.0,
            atol=1e-9,
        )
    ):
        raise ValueError(
            "Explicit time_points must equal the complete sorted observed "
            f"time grid {list(observed_unique)}. Intermediate observed "
            "targets cannot be omitted from whole-trajectory evaluation."
        )
    return resolved_time_key, requested


def run_continuous_frozen_checkpoint_ablations(
    adata,
    model_or_loaded,
    *,
    time_points: Sequence[float] | None = None,
    n_samples: int = 5000,
    dt: float = 0.01,
    sigma: float = 0.03,
    interaction_m: int = 1024,
    conditions: Sequence[str] = DEFAULT_FROZEN_ABLATION_CONDITIONS,
    max_ot_points: int | None = 1024,
    structure_max_points: int | None = 5000,
    device: str = "cuda",
    time_key: str | None = "time_point_processed",
    obsm_key: str = "X_latent",
    spatial_key: str = "spatial_aligned",
    concat_spatial: bool | None = True,
    random_seed: int = 42,
    verbose: bool = True,
) -> ContinuousFrozenCheckpointAblationResult:
    """Evaluate same-checkpoint ablations over one continuous whole trajectory.

    The earliest observed population is sampled once per matched condition
    using an identical seed, then integrated continuously to every later
    observed time with the official non-split weighted SDE.  Observed
    intermediate states are evaluation targets only; they never restart the
    simulation.  Drift, score, model-native growth, particle mass, SDE noise,
    and the fitted checkpoint remain enabled and matched.  Only the
    interaction contribution or fitted LR-informed edge gate changes.

    A dedicated Brownian-noise generator supplies common random increments
    across conditions.  This prevents the interaction grouping RNG from
    changing SDE noise merely because ``interaction_off`` skips grouping.
    """

    if int(n_samples) <= 0:
        raise ValueError("n_samples must be positive.")
    if not np.isfinite(float(dt)) or float(dt) <= 0:
        raise ValueError("dt must be finite and positive.")
    if not np.isfinite(float(sigma)) or float(sigma) < 0:
        raise ValueError("sigma must be finite and non-negative.")
    if int(interaction_m) < 2:
        raise ValueError("interaction_m must be at least 2.")

    model = _unwrap_model(model_or_loaded)
    _require_interaction_network(model)
    components = set(getattr(model, "components", ()))
    missing_components = {"velocity", "growth", "score"} - components
    if missing_components:
        raise TypeError(
            "Continuous weighted-SDE functional ablation requires a fitted "
            "full model with velocity, growth, score, and interaction "
            f"components; missing {sorted(missing_components)}."
        )

    resolved_time_key, requested_times = _resolve_complete_observed_time_grid(
        adata,
        time_points=time_points,
        time_key=time_key,
    )
    resolved = [resolve_frozen_ablation_condition(name) for name in conditions]
    names = [condition.name for condition in resolved]
    if not names:
        raise ValueError("At least one frozen ablation condition is required.")
    if len(set(names)) != len(names):
        raise ValueError(f"Duplicate frozen ablation conditions: {names}.")

    evaluations: dict[str, DistributionEvaluationResult] = {}
    condition_manifest: dict[str, Any] = {}
    with _preserve_random_state():
        for condition in resolved:
            # ``evaluate_model_distributions`` resets NumPy/Torch/CUDA seeds.
            # The explicit noise seed additionally isolates Brownian draws
            # from the condition-dependent interaction-grouping RNG stream.
            with temporary_lr_gate_mode(
                model,
                condition.lr_gate_mode,
            ) as gate_override:
                evaluation = evaluate_model_distributions(
                    adata,
                    model,
                    time_points=requested_times,
                    n_samples=int(n_samples),
                    dt=float(dt),
                    sigma=float(sigma),
                    include_score=True,
                    include_interaction=condition.interaction_enabled,
                    interaction_m=int(interaction_m),
                    max_ot_points=max_ot_points,
                    structure_max_points=structure_max_points,
                    device=str(device),
                    time_key=resolved_time_key,
                    obsm_key=obsm_key,
                    spatial_key=spatial_key,
                    concat_spatial=concat_spatial,
                    random_seed=int(random_seed),
                    noise_seed=int(random_seed),
                    include_initial_time=False,
                    verbose=bool(verbose),
                )
            evaluations[condition.name] = evaluation
            condition_manifest[condition.name] = {
                "formal_label": (
                    "same_checkpoint_all_spatial_gate_counterfactual"
                    if condition.name == "lr_gate_off"
                    else f"same_checkpoint_{condition.name}"
                ),
                "interaction_enabled": condition.interaction_enabled,
                "lr_gate_mode": condition.lr_gate_mode,
                "gate_implementation": gate_override.implementation,
                "description": condition.description,
                "weights_retrained": False,
            }

    initial_time = requested_times[0]
    reference = evaluations[names[0]]
    reference_initial_points = np.asarray(
        reference.predicted_points[initial_time],
        dtype=np.float32,
    )
    reference_initial_weights = np.asarray(
        reference.predicted_weights[initial_time],
        dtype=np.float64,
    )
    for name, evaluation in evaluations.items():
        if tuple(evaluation.time_points) != requested_times:
            raise RuntimeError(
                f"Condition '{name}' returned a different observed time grid."
            )
        if not np.array_equal(
            np.asarray(evaluation.predicted_points[initial_time]),
            reference_initial_points,
        ):
            raise RuntimeError(
                f"Condition '{name}' did not use the same sampled t0 cells."
            )
        if not np.array_equal(
            np.asarray(evaluation.predicted_weights[initial_time]),
            reference_initial_weights,
        ):
            raise RuntimeError(
                f"Condition '{name}' did not use the same initial particle mass."
            )

    manifest = {
        "analysis": "continuous_frozen_checkpoint_functional_ablation",
        "conditions": condition_manifest,
        "condition_order": names,
        "checkpoint": _checkpoint_manifest(model_or_loaded),
        "source_time": float(initial_time),
        "evaluation_times": [float(value) for value in requested_times[1:]],
        "observed_intermediate_restart": False,
        "matched_controls": {
            "same_model_object": True,
            "same_checkpoint_weights": True,
            "same_sampled_initial_cells": True,
            "same_initial_points_sha256": _array_sha256(
                reference_initial_points
            ),
            "same_initial_particle_weights": True,
            "same_random_seed": True,
            "random_seed": int(random_seed),
            "common_brownian_noise": True,
            "brownian_noise_seed": int(random_seed),
            "legacy_bitwise_identity": False,
            "legacy_comparison_note": (
                "The dedicated Brownian RNG makes paired conditions valid but "
                "can change the Full particle-level random stream relative to "
                "older runs that interleaved interaction grouping and SDE "
                "noise. Compare to legacy Full at distribution level or with "
                "tolerance, not by bitwise particle identity."
            ),
            "same_time_grid": True,
        },
        "rollout": {
            "official_api": (
                "CytoBridge.tl.downstream.simulation.simulate_sde_points"
            ),
            "simulation_mode": "continuous_non_split_weighted_sde",
            "integrator": "euler_maruyama",
            "dt": float(dt),
            "sigma": float(sigma),
            "include_score": True,
            "include_growth": True,
            "native_unnormalised_growth_mass": True,
            "interaction_group_size": int(interaction_m),
            "particle_splitting": False,
            "resampling": False,
            "spatial_warp": False,
            "n_initial_particles": int(reference_initial_points.shape[0]),
        },
        "evaluation": {
            "initial_time_in_metrics": False,
            "target_times": [float(value) for value in requested_times[1:]],
            "spaces": ["joint", "spatial", "pca"],
            "w1_w2_prediction_mass": "native weights normalized to probability",
            "tmv_prediction_mass": "native unnormalised growth mass",
            "max_ot_points": (
                None if max_ot_points is None else int(max_ot_points)
            ),
        },
        "estimand": (
            "Whole-trajectory inference-time functional dependence of one "
            "fitted full checkpoint. This is not a retrained architecture "
            "ablation."
        ),
        "lr_gate_null_scope": {
            "implemented": "all within-cutoff spatial candidates are admitted",
            "edge_density_matched": False,
            "warning": (
                "The all-spatial condition changes both LR-gate identity and "
                "edge density; it is not a pure LR-identity ablation."
            ),
        },
    }
    return ContinuousFrozenCheckpointAblationResult(
        evaluations=evaluations,
        manifest=manifest,
    )


def save_continuous_frozen_checkpoint_ablation_result(
    result: ContinuousFrozenCheckpointAblationResult,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Save continuous endpoints, native weights, metrics, and provenance."""

    import pandas as pd

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    metric_tables = []
    first_evaluation = next(iter(result.evaluations.values()))
    times = np.asarray(first_evaluation.time_points, dtype=np.float64)

    for name, evaluation in result.evaluations.items():
        points = np.stack(
            [
                np.asarray(evaluation.predicted_points[float(value)])
                for value in times
            ]
        ).astype(np.float32)
        weights = np.stack(
            [
                np.asarray(
                    evaluation.predicted_weights[float(value)],
                    dtype=np.float64,
                ).reshape(-1)
                for value in times
            ]
        )
        trajectory_path = root / f"{name}_continuous_trajectory.npz"
        np.savez_compressed(
            trajectory_path,
            times=times,
            points=points,
            weights=weights,
        )
        paths[f"{name}_trajectory"] = trajectory_path

        metrics = evaluation.metrics.copy()
        metrics.insert(0, "condition", name)
        metrics_path = root / f"{name}_metrics.csv"
        metrics.to_csv(metrics_path, index=False)
        paths[f"{name}_metrics"] = metrics_path
        metric_tables.append(metrics)

    observed_path = root / "observed_endpoints.npz"
    observed_payload = {"times": times}
    for index, value in enumerate(times):
        observed_payload[f"observed_{index}"] = np.asarray(
            first_evaluation.observed_points[float(value)],
            dtype=np.float32,
        )
    np.savez_compressed(observed_path, **observed_payload)
    paths["observed"] = observed_path

    combined_metrics_path = root / "all_conditions_metrics.csv"
    pd.concat(metric_tables, ignore_index=True).to_csv(
        combined_metrics_path,
        index=False,
    )
    paths["metrics"] = combined_metrics_path

    manifest_path = root / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(result.manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    paths["manifest"] = manifest_path
    return paths


def save_frozen_checkpoint_ablation_result(
    result: FrozenCheckpointAblationResult,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Save one NPZ per condition plus a JSON audit manifest."""

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, rollout in result.rollouts.items():
        path = root / f"{name}.npz"
        np.savez_compressed(
            path,
            times=np.asarray(rollout.times, dtype=np.float64),
            points=np.asarray(rollout.points, dtype=np.float32),
            interaction_enabled=np.asarray(
                [rollout.interaction_enabled], dtype=bool
            ),
            grouping_seed=np.asarray([rollout.grouping_seed], dtype=np.int64),
            sigma=np.asarray([rollout.sigma], dtype=np.float64),
        )
        paths[name] = path

    manifest_path = root / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(result.manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    paths["manifest"] = manifest_path
    return paths
