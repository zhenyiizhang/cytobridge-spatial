"""Reusable post-training diagnostics for one-shot metabolic RNA labeling.

The estimator implements the elementary one-shot kinetic model

``new = alpha / gamma * (1 - exp(-gamma * tau))``

and estimates gene-specific degradation rates from a baseline population
assumed to be at steady state.  It is intentionally independent of model
training and is suitable for held-out directional evaluation.  Inputs must be
linear, consistently size-factor-normalized total/new RNA abundances.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse


@dataclass(frozen=True)
class OneShotLabelingVelocity:
    """Gene-space one-shot velocity and its fitted baseline parameters."""

    velocity_linear: np.ndarray
    velocity_log1p: np.ndarray
    baseline_new_fraction: np.ndarray
    degradation_rate: np.ndarray
    labeling_time: float


def _dense_finite_nonnegative(matrix, *, name: str) -> np.ndarray:
    values = matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional matrix.")
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError(f"{name} must contain finite non-negative values.")
    return values


def estimate_one_shot_labeling_velocity(
    total,
    new,
    *,
    baseline_mask,
    labeling_time: float,
    pseudocount: float = 1.0e-8,
    minimum_new_fraction: float = 1.0e-4,
    maximum_new_fraction: float = 0.95,
) -> OneShotLabelingVelocity:
    """Estimate held-out RNA velocity under a one-shot labeling model.

    ``baseline_mask`` defines a reference population assumed to be at steady
    state before stimulation.  Baseline new/total fractions estimate
    gene-specific degradation rates, after which each cell's new RNA estimates
    synthesis.  The returned ``velocity_log1p`` is the chain-rule derivative
    of ``log1p(total)`` and can be projected with PCA loadings fitted on that
    same state representation.
    """

    total_values = _dense_finite_nonnegative(total, name="total")
    new_values = _dense_finite_nonnegative(new, name="new")
    if total_values.shape != new_values.shape:
        raise ValueError("total and new must have identical shapes.")
    mask = np.asarray(baseline_mask, dtype=bool).reshape(-1)
    if mask.shape[0] != total_values.shape[0] or not np.any(mask):
        raise ValueError("baseline_mask must select at least one input row.")
    tau = float(labeling_time)
    if not np.isfinite(tau) or tau <= 0:
        raise ValueError("labeling_time must be positive and finite.")
    eps = float(pseudocount)
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("pseudocount must be positive and finite.")
    lower = float(minimum_new_fraction)
    upper = float(maximum_new_fraction)
    if not 0 < lower < upper < 1:
        raise ValueError(
            "minimum_new_fraction and maximum_new_fraction must satisfy "
            "0 < minimum < maximum < 1."
        )

    baseline_total = total_values[mask].sum(axis=0)
    baseline_new = new_values[mask].sum(axis=0)
    raw_fraction = np.divide(
        baseline_new,
        baseline_total,
        out=np.zeros_like(baseline_new),
        where=baseline_total > eps,
    )
    fraction = np.clip(raw_fraction, lower, upper)
    degradation = -np.log1p(-fraction) / tau
    synthesis = new_values * (degradation / fraction)[None, :]
    velocity = synthesis - total_values * degradation[None, :]
    velocity_log1p = velocity / (1.0 + total_values)
    if not np.isfinite(velocity_log1p).all():
        raise FloatingPointError("One-shot velocity contains non-finite values.")
    return OneShotLabelingVelocity(
        velocity_linear=velocity,
        velocity_log1p=velocity_log1p,
        baseline_new_fraction=fraction,
        degradation_rate=degradation,
        labeling_time=tau,
    )


def project_log_velocity_to_pca(
    velocity_log1p: np.ndarray, pca_loadings: np.ndarray
) -> np.ndarray:
    """Project a log-expression derivative through fitted PCA loadings."""

    velocity = np.asarray(velocity_log1p, dtype=np.float64)
    loadings = np.asarray(pca_loadings, dtype=np.float64)
    if velocity.ndim != 2 or loadings.ndim != 2:
        raise ValueError("velocity_log1p and pca_loadings must be 2D matrices.")
    if velocity.shape[1] != loadings.shape[0]:
        raise ValueError(
            "Velocity gene dimension must equal the loading row dimension."
        )
    result = velocity @ loadings
    if not np.isfinite(result).all():
        raise FloatingPointError("Projected PCA velocity contains non-finite values.")
    return result


def row_cosine_similarity(
    left: np.ndarray, right: np.ndarray, *, minimum_norm: float = 1.0e-12
) -> np.ndarray:
    """Return row-wise cosine similarity, using NaN for near-zero vectors."""

    left_values = np.asarray(left, dtype=np.float64)
    right_values = np.asarray(right, dtype=np.float64)
    if left_values.shape != right_values.shape or left_values.ndim != 2:
        raise ValueError("left and right must be equal-shaped 2D matrices.")
    threshold = float(minimum_norm)
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("minimum_norm must be finite and non-negative.")
    left_norm = np.linalg.norm(left_values, axis=1)
    right_norm = np.linalg.norm(right_values, axis=1)
    valid = (left_norm > threshold) & (right_norm > threshold)
    output = np.full(left_values.shape[0], np.nan, dtype=np.float64)
    output[valid] = np.sum(left_values[valid] * right_values[valid], axis=1) / (
        left_norm[valid] * right_norm[valid]
    )
    return np.clip(output, -1.0, 1.0)


__all__ = [
    "OneShotLabelingVelocity",
    "estimate_one_shot_labeling_velocity",
    "project_log_velocity_to_pca",
    "row_cosine_similarity",
]
