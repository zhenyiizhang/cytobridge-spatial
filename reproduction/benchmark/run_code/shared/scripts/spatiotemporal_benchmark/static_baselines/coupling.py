"""Coupling validation, composition, and explicit static controls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .data import AnchorPair, StageSlice
from .errors import OfficialAPIError


@dataclass(frozen=True)
class CouplingDiagnostics:
    shape: tuple[int, int]
    total_mass: float
    row_sum_min: float
    row_sum_max: float
    zero_rows: int


def validate_and_row_normalize(
    plan: object,
    expected_shape: tuple[int, int],
) -> tuple[np.ndarray, CouplingDiagnostics]:
    if hasattr(plan, "toarray"):
        plan = plan.toarray()
    array = np.asarray(plan, dtype=np.float64)
    if array.shape != expected_shape:
        raise OfficialAPIError(
            f"Official coupling has shape {array.shape}; expected source-by-target "
            f"{expected_shape}. Orientation is never guessed."
        )
    if not np.isfinite(array).all():
        raise OfficialAPIError("Official coupling contains non-finite values")
    if np.min(array, initial=0.0) < -1e-10:
        raise OfficialAPIError("Official coupling contains negative mass")
    array = np.maximum(array, 0.0)
    row_sum = array.sum(axis=1)
    zero_rows = int(np.count_nonzero(row_sum <= 0.0))
    if zero_rows:
        raise OfficialAPIError(
            f"Official coupling contains {zero_rows} zero-mass source rows; "
            "a uniform surrogate fill is forbidden"
        )
    diagnostics = CouplingDiagnostics(
        shape=(int(array.shape[0]), int(array.shape[1])),
        total_mass=float(array.sum()),
        row_sum_min=float(row_sum.min()),
        row_sum_max=float(row_sum.max()),
        zero_rows=zero_rows,
    )
    return array / row_sum[:, None], diagnostics


def compose_row_plans(plans: Sequence[np.ndarray]) -> tuple[np.ndarray, list[dict[str, float]]]:
    """Compose P01, P12, ... while retaining a row-stochastic P0t map."""
    if not plans:
        raise ValueError("At least one coupling is required")
    composed = np.asarray(plans[0], dtype=np.float64)
    history: list[dict[str, float]] = []
    for index, next_plan in enumerate(plans):
        if index:
            next_array = np.asarray(next_plan, dtype=np.float64)
            if composed.shape[1] != next_array.shape[0]:
                raise OfficialAPIError(
                    f"Cannot compose coupling shapes {composed.shape} and {next_array.shape}"
                )
            composed = composed @ next_array
        if not np.isfinite(composed).all() or np.min(composed, initial=0.0) < -1e-10:
            raise OfficialAPIError("Composed coupling became invalid")
        composed = np.maximum(composed, 0.0)
        row_sum = composed.sum(axis=1)
        if np.any(row_sum <= 0.0):
            raise OfficialAPIError("Composed coupling contains a zero-mass source row")
        composed /= row_sum[:, None]
        history.append(
            {
                "step": float(index + 1),
                "rows": float(composed.shape[0]),
                "columns": float(composed.shape[1]),
                "row_sum_min": float(composed.sum(axis=1).min()),
                "row_sum_max": float(composed.sum(axis=1).max()),
            }
        )
    return composed, history


def project_loto_joint(pair: AnchorPair, row_plan: np.ndarray) -> np.ndarray:
    """Barycentric bracket projection followed by fractional time interpolation."""
    source = pair.previous.joint.astype(np.float64)
    mapped = np.asarray(row_plan, dtype=np.float64) @ pair.following.joint.astype(np.float64)
    alpha = float(pair.interpolation_alpha)
    result = (1.0 - alpha) * source + alpha * mapped
    if not np.isfinite(result).all():
        raise OfficialAPIError("LOTO joint projection contains non-finite values")
    return result.astype(np.float32)


def project_loto_state(pair: AnchorPair, row_plan: np.ndarray) -> np.ndarray:
    source = pair.previous.state_pca.astype(np.float64)
    mapped = np.asarray(row_plan, dtype=np.float64) @ pair.following.state_pca.astype(np.float64)
    alpha = float(pair.interpolation_alpha)
    result = (1.0 - alpha) * source + alpha * mapped
    if not np.isfinite(result).all():
        raise OfficialAPIError("LOTO state projection contains non-finite values")
    return result.astype(np.float32)


def project_composed_joint(target: StageSlice, composed_plan: np.ndarray) -> np.ndarray:
    result = np.asarray(composed_plan, dtype=np.float64) @ target.joint.astype(np.float64)
    if not np.isfinite(result).all():
        raise OfficialAPIError("Composed joint projection contains non-finite values")
    return result.astype(np.float32)


def project_composed_state(target: StageSlice, composed_plan: np.ndarray) -> np.ndarray:
    result = np.asarray(composed_plan, dtype=np.float64) @ target.state_pca.astype(np.float64)
    if not np.isfinite(result).all():
        raise OfficialAPIError("Composed state projection contains non-finite values")
    return result.astype(np.float32)


def take_roster(points: np.ndarray, roster_indices: np.ndarray) -> np.ndarray:
    array = np.asarray(points, dtype=np.float32)
    indices = np.asarray(roster_indices, dtype=np.int64)
    if array.ndim != 2 or len(array) == 0:
        raise ValueError(f"Cannot apply roster to prediction with shape {array.shape}")
    if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= len(array)):
        raise ValueError("Source roster indices are invalid")
    result = array[indices]
    if not np.isfinite(result).all():
        raise ValueError("Roster prediction contains non-finite values")
    return result.astype(np.float32, copy=False)


def random_independent_plan(
    pair: AnchorPair,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """One independent target draw per fitted source row, represented as a plan."""
    target_indices = rng.integers(0, pair.following.n_obs, size=pair.previous.n_obs, endpoint=False)
    plan = np.zeros((pair.previous.n_obs, pair.following.n_obs), dtype=np.float64)
    plan[np.arange(pair.previous.n_obs), target_indices] = 1.0
    return plan, np.asarray(target_indices, dtype=np.int64)


def linear_centroid_loto(pair: AnchorPair) -> tuple[np.ndarray, np.ndarray]:
    source = pair.previous.joint.astype(np.float64)
    shift = float(pair.interpolation_alpha) * (
        pair.following.joint.astype(np.float64).mean(axis=0) - source.mean(axis=0)
    )
    result = source + shift
    return result.astype(np.float32), shift.astype(np.float32)


def linear_centroid_trajectory(
    stages: Sequence[StageSlice],
) -> tuple[dict[float, np.ndarray], list[np.ndarray]]:
    """Sequentially compose adjacent centroid shifts from the original t0 points."""
    if len(stages) < 2:
        raise ValueError("Centroid trajectory needs at least two stages")
    current = stages[0].joint.astype(np.float64).copy()
    outputs: dict[float, np.ndarray] = {}
    shifts: list[np.ndarray] = []
    for left, right in zip(stages[:-1], stages[1:]):
        shift = right.joint.astype(np.float64).mean(axis=0) - left.joint.astype(np.float64).mean(axis=0)
        current = current + shift
        outputs[float(right.time)] = current.astype(np.float32)
        shifts.append(shift.astype(np.float32))
    return outputs, shifts
