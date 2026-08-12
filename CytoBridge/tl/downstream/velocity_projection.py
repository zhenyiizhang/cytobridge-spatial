"""Dependency-light projection of latent velocities into 2D embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import numpy as np

__all__ = [
    "VelocityEmbeddingProjectionResult",
    "project_velocity_to_embedding",
]


@dataclass(frozen=True)
class VelocityEmbeddingProjectionResult:
    """Projected vectors and the auditable neighborhood transition contract."""

    projected_velocity: np.ndarray
    neighbor_indices: np.ndarray
    transition_probabilities: np.ndarray
    transition_weights: np.ndarray
    cosine_similarities: np.ndarray
    diagnostics: Mapping[str, object]


def _validate_matrix(name: str, values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array.")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} contains non-finite values.")
    return matrix


def _knn_indices(coordinates: np.ndarray, n_neighbors: int) -> np.ndarray:
    from scipy.spatial import cKDTree

    n_observations = int(coordinates.shape[0])
    tree = cKDTree(coordinates)
    query_k = min(n_observations, int(n_neighbors) + 1)
    distances, indices = tree.query(coordinates, k=query_k, workers=1)
    if query_k == 1:
        distances = np.asarray(distances)[:, None]
        indices = np.asarray(indices)[:, None]
    result = np.empty((n_observations, int(n_neighbors)), dtype=np.int64)
    for row in range(n_observations):
        keep = indices[row] != row
        candidates = indices[row, keep].astype(np.int64, copy=False)
        candidate_distances = distances[row, keep]
        if candidates.size < int(n_neighbors):
            raise RuntimeError(
                "Nearest-neighbor search did not return enough non-self neighbors."
            )
        # Query every point tied at the kth boundary. cKDTree may otherwise
        # choose an arbitrary subset of equidistant candidates, which would
        # make the downstream velocity result depend on library internals.
        provisional_order = np.lexsort((candidates, candidate_distances))
        cutoff = float(candidate_distances[provisional_order[int(n_neighbors) - 1]])
        radius = np.nextafter(cutoff, np.inf)
        candidates = np.asarray(
            [index for index in tree.query_ball_point(coordinates[row], radius) if index != row],
            dtype=np.int64,
        )
        candidate_distances = np.linalg.norm(
            coordinates[candidates] - coordinates[row],
            axis=1,
        )
        # A stable index key makes equidistant-neighbor ties reproducible.
        order = np.lexsort((candidates, candidate_distances))
        candidates = candidates[order]
        if candidates.size < int(n_neighbors):
            raise RuntimeError(
                "Nearest-neighbor search did not return enough non-self neighbors."
            )
        result[row] = candidates[: int(n_neighbors)]
    return result


def _validate_neighbor_indices(
    neighbor_indices: np.ndarray | Sequence[Sequence[int]],
    *,
    n_observations: int,
) -> np.ndarray:
    raw = np.asarray(neighbor_indices)
    if raw.ndim != 2 or raw.shape[0] != int(n_observations) or raw.shape[1] < 1:
        raise ValueError(
            "neighbor_indices must have shape (n_observations, n_neighbors) "
            "with at least one neighbor per row."
        )
    if not np.issubdtype(raw.dtype, np.integer):
        if not np.issubdtype(raw.dtype, np.floating) or not np.isfinite(raw).all():
            raise ValueError("neighbor_indices must contain finite integer indices.")
        if not np.equal(raw, np.floor(raw)).all():
            raise ValueError("neighbor_indices must contain integer indices.")
    indices = raw.astype(np.int64, copy=False)
    if np.any(indices < 0) or np.any(indices >= int(n_observations)):
        raise ValueError("neighbor_indices contains an out-of-range cell index.")
    rows = np.arange(int(n_observations), dtype=np.int64)[:, None]
    if np.any(indices == rows):
        raise ValueError("neighbor_indices must not contain self-neighbors.")
    for row in indices:
        if np.unique(row).size != row.size:
            raise ValueError("neighbor_indices must not contain duplicate row entries.")
    return indices.copy()


def project_velocity_to_embedding(
    latent_coordinates: np.ndarray,
    latent_velocity: np.ndarray,
    embedding: np.ndarray,
    *,
    n_neighbors: int = 30,
    neighbor_indices: Optional[np.ndarray | Sequence[Sequence[int]]] = None,
    temperature: float = 1.0,
    normalize_embedding_displacements: bool = True,
    center_probabilities: bool = True,
    epsilon: float = 1e-12,
) -> VelocityEmbeddingProjectionResult:
    """Project a high-dimensional velocity through local transitions.

    A k-nearest-neighbor graph is built in ``latent_coordinates`` (never in
    the target embedding). For each cell, cosine similarity between its full
    latent velocity and every latent neighbor displacement is transformed by a
    row-wise softmax. The resulting transition weights are applied to neighbor
    displacements in ``embedding``. By default, embedding displacements are
    unit-normalized and probabilities are mean-centered, matching the central
    geometric choices of scVelo-style velocity embedding without requiring
    AnnData, Scanpy, or scVelo.

    Supplying ``neighbor_indices`` freezes the graph and makes separate
    component projections directly comparable. It must be a dense integer
    array with one non-self, duplicate-free neighbor list per observation.
    ``n_neighbors`` is used only when that array is omitted.
    """
    coordinates = _validate_matrix("latent_coordinates", latent_coordinates)
    velocity = _validate_matrix("latent_velocity", latent_velocity)
    target = _validate_matrix("embedding", embedding)
    if coordinates.shape != velocity.shape:
        raise ValueError(
            "latent_coordinates and latent_velocity must have identical shapes."
        )
    if target.shape[0] != coordinates.shape[0] or target.shape[1] != 2:
        raise ValueError("embedding must have shape (n_observations, 2).")
    n_observations = int(coordinates.shape[0])
    if n_observations < 2:
        raise ValueError("At least two observations are required.")
    if coordinates.shape[1] < 1:
        raise ValueError("latent_coordinates must contain at least one dimension.")
    temperature = float(temperature)
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive.")
    epsilon = float(epsilon)
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be finite and positive.")

    requested_neighbors = int(n_neighbors)
    if neighbor_indices is None:
        if requested_neighbors <= 0:
            raise ValueError("n_neighbors must be positive.")
        effective_neighbors = min(requested_neighbors, n_observations - 1)
        indices = _knn_indices(coordinates, effective_neighbors)
        neighbor_source = "knn_latent_coordinates"
    else:
        indices = _validate_neighbor_indices(
            neighbor_indices,
            n_observations=n_observations,
        )
        effective_neighbors = int(indices.shape[1])
        neighbor_source = "caller_supplied"

    latent_displacements = coordinates[indices] - coordinates[:, None, :]
    latent_displacement_norm = np.linalg.norm(latent_displacements, axis=2)
    velocity_norm = np.linalg.norm(velocity, axis=1)
    valid_displacements = latent_displacement_norm > epsilon
    valid_velocity = velocity_norm > epsilon
    denominator = velocity_norm[:, None] * latent_displacement_norm
    cosine = np.zeros_like(latent_displacement_norm, dtype=np.float64)
    valid_cosine = valid_displacements & valid_velocity[:, None]
    cosine[valid_cosine] = (
        np.einsum(
            "nd,nkd->nk",
            velocity,
            latent_displacements,
        )[valid_cosine]
        / denominator[valid_cosine]
    )
    cosine = np.clip(cosine, -1.0, 1.0)

    probabilities = np.zeros_like(cosine, dtype=np.float64)
    rows_with_neighbors = valid_displacements.any(axis=1)
    rows_with_transitions = rows_with_neighbors & valid_velocity
    if np.any(rows_with_transitions):
        logits = cosine[rows_with_transitions] / temperature
        valid_subset = valid_displacements[rows_with_transitions]
        logits = np.where(valid_subset, logits, -np.inf)
        row_max = np.max(logits, axis=1, keepdims=True)
        exponentials = np.where(valid_subset, np.exp(logits - row_max), 0.0)
        probabilities[rows_with_transitions] = exponentials / exponentials.sum(
            axis=1,
            keepdims=True,
        )

    if bool(center_probabilities):
        valid_count = valid_displacements.sum(axis=1, keepdims=True)
        baseline = np.divide(
            1.0,
            valid_count,
            out=np.zeros_like(valid_count, dtype=np.float64),
            where=valid_count > 0,
        )
        weights = np.where(
            valid_displacements & valid_velocity[:, None],
            probabilities - baseline,
            0.0,
        )
    else:
        weights = probabilities.copy()

    embedding_displacements = target[indices] - target[:, None, :]
    zero_embedding_displacements = np.zeros(n_observations, dtype=int)
    if bool(normalize_embedding_displacements):
        embedding_norm = np.linalg.norm(embedding_displacements, axis=2)
        valid_embedding = embedding_norm > epsilon
        zero_embedding_displacements = (~valid_embedding).sum(axis=1)
        embedding_displacements = np.divide(
            embedding_displacements,
            embedding_norm[:, :, None],
            out=np.zeros_like(embedding_displacements),
            where=valid_embedding[:, :, None],
        )
    projected = np.einsum("nk,nkd->nd", weights, embedding_displacements)
    # The velocity direction is undefined for zero vectors. Return an exact
    # zero rather than a graph-density artifact even when centering is disabled.
    projected[~valid_velocity] = 0.0

    diagnostics = {
        "algorithm": "latent_knn_cosine_softmax_embedding_displacement",
        "n_observations": n_observations,
        "latent_dimension": int(coordinates.shape[1]),
        "embedding_dimension": 2,
        "requested_n_neighbors": requested_neighbors,
        "effective_n_neighbors": effective_neighbors,
        "neighbor_source": neighbor_source,
        "knn_backend": (
            "scipy.spatial.cKDTree_exact_boundary_ties"
            if neighbor_indices is None
            else None
        ),
        "n_neighbors_parameter_used": neighbor_indices is None,
        "temperature": temperature,
        "normalize_embedding_displacements": bool(normalize_embedding_displacements),
        "center_probabilities": bool(center_probabilities),
        "n_zero_velocity": int((~valid_velocity).sum()),
        "n_rows_without_distinct_latent_neighbor": int((~rows_with_neighbors).sum()),
        "n_rows_without_transition": int((~rows_with_transitions).sum()),
        "n_zero_embedding_displacements": int(zero_embedding_displacements.sum()),
        "epsilon": epsilon,
    }
    return VelocityEmbeddingProjectionResult(
        projected_velocity=projected,
        neighbor_indices=indices,
        transition_probabilities=probabilities,
        transition_weights=weights,
        cosine_similarities=cosine,
        diagnostics=diagnostics,
    )
