"""Utilities for radius graphs in non-spatial latent state spaces."""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

import numpy as np
from scipy.spatial.distance import pdist, squareform


def estimate_state_space_radius(
    latent: np.ndarray,
    *,
    groups: Optional[Sequence[object]] = None,
    quantile: float = 0.99,
    max_points_per_group: int = 2048,
    interaction_group_size: int = 16,
    diagnostic_groups: int = 256,
    random_seed: int = 42,
) -> dict:
    """Estimate and diagnose a radius cutoff for a latent-state GNN.

    The estimator samples cells independently within each supplied group
    (normally an observed time point), computes its pair-distance quantile,
    and uses the maximum group-specific quantile as the shared cutoff.  Taking
    the maximum keeps graph density comparable across time while avoiding a
    tissue-specific spatial edge classifier.

    Parameters
    ----------
    latent
        ``N x D`` latent state matrix.
    groups
        Optional length-``N`` labels, for example processed time points.  If
        omitted, all rows form one group.
    quantile
        Pair-distance quantile used as the radius.  A value near one gives the
        mean-field-like, nearly complete small groups used in non-spatial
        CytoBridge experiments.
    max_points_per_group
        Maximum number of cells used to estimate distances in each group.
    interaction_group_size
        Particle group size used by ``cal_interaction_gnn``.  It is used only
        for the reported connectivity diagnostics.
    diagnostic_groups
        Number of random particle groups sampled per label for diagnostics.
    random_seed
        Seed for deterministic sampling.
    """

    values = np.asarray(latent, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("latent must be a two-dimensional array with at least 2 rows.")
    if not np.isfinite(values).all():
        raise ValueError("latent contains non-finite values.")
    if not 0.0 < float(quantile) < 1.0:
        raise ValueError("quantile must lie strictly between 0 and 1.")
    if int(max_points_per_group) < 2:
        raise ValueError("max_points_per_group must be at least 2.")
    if int(interaction_group_size) < 2:
        raise ValueError("interaction_group_size must be at least 2.")
    if int(diagnostic_groups) < 1:
        raise ValueError("diagnostic_groups must be positive.")

    if groups is None:
        labels = np.full(values.shape[0], "all", dtype=object)
    else:
        labels = np.asarray(groups, dtype=object).reshape(-1)
        if labels.shape[0] != values.shape[0]:
            raise ValueError("groups must have the same number of rows as latent.")

    rng = np.random.default_rng(int(random_seed))
    unique_labels = list(dict.fromkeys(labels.tolist()))
    per_group: dict[str, dict[str, float | int]] = {}
    sampled_by_label: dict[str, np.ndarray] = {}
    for raw_label in unique_labels:
        key = str(raw_label)
        indices = np.flatnonzero(labels == raw_label)
        if indices.size < 2:
            raise ValueError(f"Group {key!r} has fewer than two rows.")
        sample_size = min(indices.size, int(max_points_per_group))
        sampled_indices = rng.choice(indices, size=sample_size, replace=False)
        sampled = values[sampled_indices]
        distances = pdist(sampled, metric="euclidean")
        sampled_by_label[key] = sampled
        per_group[key] = {
            "n_total": int(indices.size),
            "n_distance_sample": int(sample_size),
            "pair_distance_q025": float(np.quantile(distances, 0.025)),
            "pair_distance_q50": float(np.quantile(distances, 0.5)),
            "pair_distance_q95": float(np.quantile(distances, 0.95)),
            "pair_distance_q99": float(np.quantile(distances, 0.99)),
            "selected_quantile_distance": float(np.quantile(distances, quantile)),
        }

    cutoff = max(
        float(stats["selected_quantile_distance"]) for stats in per_group.values()
    )

    # Diagnose the actual random groups used by the interaction API.  Sampling
    # with replacement across diagnostic groups is intentional; each particle
    # group itself contains distinct rows.
    for key, sampled in sampled_by_label.items():
        group_size = min(int(interaction_group_size), sampled.shape[0])
        degrees: list[np.ndarray] = []
        for _ in range(int(diagnostic_groups)):
            choice = rng.choice(sampled.shape[0], size=group_size, replace=False)
            distances = squareform(pdist(sampled[choice], metric="euclidean"))
            adjacency = (distances < cutoff) & (distances > 1e-6)
            degrees.append(adjacency.sum(axis=1).astype(np.float64))
        all_degrees = np.concatenate(degrees)
        per_group[key].update(
            {
                "diagnostic_particle_group_size": int(group_size),
                "diagnostic_mean_degree": float(all_degrees.mean()),
                "diagnostic_isolated_fraction": float(np.mean(all_degrees == 0)),
            }
        )

    return {
        "method": "max_within_group_pair_distance_quantile",
        "cutoff": float(cutoff),
        "quantile": float(quantile),
        "max_points_per_group": int(max_points_per_group),
        "interaction_group_size": int(interaction_group_size),
        "diagnostic_groups": int(diagnostic_groups),
        "random_seed": int(random_seed),
        "per_group": per_group,
    }


def state_space_fit_params(radius_estimate: Mapping[str, object]) -> dict:
    """Build the dataset-specific ``adata.uns['fit_params']`` payload."""

    if "cutoff" not in radius_estimate:
        raise KeyError("radius_estimate is missing 'cutoff'.")
    cutoff = float(radius_estimate["cutoff"])
    if not np.isfinite(cutoff) or cutoff <= 0:
        raise ValueError("radius_estimate['cutoff'] must be positive and finite.")
    return {"interaction_cutoff": cutoff}


__all__ = ["estimate_state_space_radius", "state_space_fit_params"]
