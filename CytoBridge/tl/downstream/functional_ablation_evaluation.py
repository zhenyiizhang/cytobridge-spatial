"""Distribution evaluation for saved frozen-checkpoint ablation endpoints.

The functions in this module keep two estimands separate:

* ``condition_vs_observed`` measures distributional agreement with an observed
  endpoint.  Generated and observed empirical clouds are sampled independently
  and passed to :func:`compute_distribution_metrics`.
* ``condition_vs_full`` measures the effect of an inference-time switch on the
  same fixed cohort.  A single set of row indices is therefore applied to both
  clouds before exact OT is calculated.  The observed row identity also gives
  an identity-coupling upper bound and paired cell-level displacement.

The distinction matters for fixed-cohort rollouts: independently subsampling
the two sides of a paired comparison can make the reported OT distance larger
than the cost of the known identity coupling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .evaluation import compute_distribution_metrics

__all__ = [
    "FrozenAblationDistributionEvaluation",
    "evaluate_frozen_ablation_distributions",
]


@dataclass(frozen=True)
class FrozenAblationDistributionEvaluation:
    """Distribution metrics for one set of matched frozen rollouts."""

    condition_vs_observed: pd.DataFrame
    condition_vs_full: pd.DataFrame
    sensitivity_summary: pd.DataFrame
    settings: Mapping[str, Any]


def _validate_cloud(
    values: np.ndarray,
    *,
    label: str,
    expected_dim: int | None = None,
) -> np.ndarray:
    cloud = np.asarray(values, dtype=np.float64)
    if cloud.ndim != 2 or cloud.shape[0] < 2:
        raise ValueError(f"{label} must be an N x D matrix with N >= 2.")
    if expected_dim is not None and cloud.shape[1] != expected_dim:
        raise ValueError(
            f"{label} has {cloud.shape[1]} columns, expected {expected_dim}."
        )
    if not np.isfinite(cloud).all():
        raise ValueError(f"{label} contains non-finite values.")
    return cloud


def _feature_spaces(dim: int, spatial_dim: int) -> dict[str, slice]:
    if spatial_dim <= 0 or spatial_dim >= dim:
        raise ValueError(
            f"spatial_dim must be between 1 and D-1; got {spatial_dim} for D={dim}."
        )
    return {
        "joint": slice(0, dim),
        "spatial": slice(0, spatial_dim),
        "state": slice(spatial_dim, dim),
    }


def _resolve_seeds(
    primary_seed: int,
    sensitivity_seeds: Sequence[int],
) -> tuple[int, ...]:
    primary = int(primary_seed)
    ordered: list[int] = []
    for seed in (primary, *sensitivity_seeds):
        value = int(seed)
        if value not in ordered:
            ordered.append(value)
    return tuple(ordered)


def _shared_cohort_indices(
    n_cells: int,
    *,
    max_ot_points: int | None,
    random_seed: int,
) -> np.ndarray:
    if max_ot_points is None or n_cells <= int(max_ot_points):
        return np.arange(n_cells, dtype=np.int64)
    cap = int(max_ot_points)
    if cap <= 0:
        raise ValueError("max_ot_points must be positive or None.")
    return np.sort(
        np.random.default_rng(int(random_seed)).choice(
            n_cells,
            size=cap,
            replace=False,
        )
    )


def _identity_coupling_metrics(
    candidate: np.ndarray,
    reference: np.ndarray,
) -> dict[str, float]:
    displacement = np.linalg.norm(candidate - reference, axis=1)
    return {
        "identity_coupling_w1_upper_bound": float(np.mean(displacement)),
        "identity_coupling_w2_upper_bound": float(
            np.sqrt(np.mean(np.square(displacement)))
        ),
        "paired_displacement_median": float(np.median(displacement)),
        "paired_displacement_q25": float(np.quantile(displacement, 0.25)),
        "paired_displacement_q75": float(np.quantile(displacement, 0.75)),
    }


def _add_full_deltas(table: pd.DataFrame, *, full_condition: str) -> pd.DataFrame:
    full = (
        table.loc[table["condition"] == full_condition]
        .set_index(["space", "sampling_seed"])[["w1", "w2"]]
        .rename(columns={"w1": "full_w1", "w2": "full_w2"})
    )
    indexed = table.set_index(["space", "sampling_seed"]).join(
        full,
        how="left",
        validate="many_to_one",
    )
    if indexed[["full_w1", "full_w2"]].isna().any().any():
        raise RuntimeError("Full-condition reference metrics are incomplete.")
    for metric in ("w1", "w2"):
        indexed[f"{metric}_delta_vs_full"] = (
            indexed[metric] - indexed[f"full_{metric}"]
        )
        denominator = indexed[f"full_{metric}"].to_numpy(dtype=float)
        numerator = indexed[f"{metric}_delta_vs_full"].to_numpy(dtype=float)
        indexed[f"{metric}_percent_change_vs_full"] = np.divide(
            100.0 * numerator,
            denominator,
            out=np.full_like(numerator, np.nan),
            where=np.abs(denominator) > np.finfo(float).eps,
        )
    return indexed.reset_index()


def _summarize_sensitivity(
    condition_vs_observed: pd.DataFrame,
    condition_vs_full: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for comparison, table in (
        ("condition_vs_observed", condition_vs_observed),
        ("condition_vs_full", condition_vs_full),
    ):
        for (condition, space), group in table.groupby(
            ["condition", "space"],
            sort=False,
            observed=True,
        ):
            for metric in ("w1", "w2"):
                values = group[metric].to_numpy(dtype=float)
                primary = group.loc[group["is_primary_seed"], metric]
                if len(primary) != 1:
                    raise RuntimeError(
                        f"Expected one primary {metric} row for "
                        f"{comparison}/{condition}/{space}."
                    )
                rows.append(
                    {
                        "comparison": comparison,
                        "condition": condition,
                        "space": space,
                        "metric": metric,
                        "primary_value": float(primary.iloc[0]),
                        "sensitivity_mean": float(np.mean(values)),
                        "sensitivity_sd": (
                            float(np.std(values, ddof=1))
                            if len(values) > 1
                            else 0.0
                        ),
                        "sensitivity_min": float(np.min(values)),
                        "sensitivity_max": float(np.max(values)),
                        "n_sampling_seeds": int(len(values)),
                    }
                )
    return pd.DataFrame(rows)


def evaluate_frozen_ablation_distributions(
    condition_endpoints: Mapping[str, np.ndarray],
    observed_endpoint: np.ndarray,
    *,
    spatial_dim: int = 2,
    full_condition: str = "full",
    max_ot_points: int | None = 1024,
    primary_seed: int = 42,
    sensitivity_seeds: Sequence[int] = (17, 23, 42, 101, 202),
) -> FrozenAblationDistributionEvaluation:
    """Evaluate frozen endpoints against observations and their full control.

    All distributions use uniform empirical mass.  For
    ``condition_vs_observed``, the generated and observed point clouds are
    independently sampled without replacement by
    :func:`compute_distribution_metrics`.  For ``condition_vs_full``, the same
    fixed cohort indices are applied to both sides before exact OT is run.

    Parameters
    ----------
    condition_endpoints:
        Mapping from condition name to an ``N x D`` endpoint matrix.  Every
        condition must retain the same rows in the same order.
    observed_endpoint:
        Observed target-stage point cloud with ``D`` columns.  Its number of
        rows need not match the fixed rollout cohort.
    spatial_dim:
        Number of leading columns representing aligned spatial coordinates.
        Remaining columns are labelled ``state``.
    full_condition:
        Name of the fitted full-model control.
    max_ot_points:
        Point cap for exact EMD.  The cap is applied independently for
        condition-versus-observed and as one shared row subset for
        condition-versus-full.
    primary_seed:
        Prespecified primary sampling seed.  It is always included and marked
        separately even if omitted from ``sensitivity_seeds``.
    sensitivity_seeds:
        Fixed seeds used to quantify sampling sensitivity.
    """

    if not condition_endpoints:
        raise ValueError("condition_endpoints must contain at least one condition.")
    if full_condition not in condition_endpoints:
        raise KeyError(
            f"Full condition '{full_condition}' is missing from condition_endpoints."
        )
    if max_ot_points is not None and int(max_ot_points) <= 0:
        raise ValueError("max_ot_points must be positive or None.")

    condition_order = list(condition_endpoints)
    full = _validate_cloud(
        condition_endpoints[full_condition],
        label=f"condition_endpoints[{full_condition!r}]",
    )
    observed = _validate_cloud(
        observed_endpoint,
        label="observed_endpoint",
        expected_dim=full.shape[1],
    )
    endpoints: dict[str, np.ndarray] = {}
    for name in condition_order:
        cloud = _validate_cloud(
            condition_endpoints[name],
            label=f"condition_endpoints[{name!r}]",
            expected_dim=full.shape[1],
        )
        if cloud.shape != full.shape:
            raise ValueError(
                "Frozen condition endpoints must have the same fixed-cohort "
                f"shape; {name!r} has {cloud.shape}, full has {full.shape}."
            )
        endpoints[name] = cloud

    spaces = _feature_spaces(full.shape[1], int(spatial_dim))
    seeds = _resolve_seeds(primary_seed, sensitivity_seeds)
    observed_rows: list[dict[str, object]] = []
    paired_rows: list[dict[str, object]] = []

    for seed in seeds:
        for condition in condition_order:
            candidate = endpoints[condition]
            for space, columns in spaces.items():
                metric = compute_distribution_metrics(
                    candidate[:, columns],
                    observed[:, columns],
                    predicted_weights=None,
                    max_ot_points=max_ot_points,
                    random_seed=seed,
                )
                observed_rows.append(
                    {
                        "comparison": "condition_vs_observed",
                        "condition": condition,
                        "reference": "observed",
                        "space": space,
                        "sampling_seed": int(seed),
                        "is_primary_seed": bool(seed == int(primary_seed)),
                        "w1": float(metric["w1"]),
                        "w2": float(metric["w2"]),
                        "ot_condition_points": int(
                            metric["ot_predicted_points"]
                        ),
                        "ot_reference_points": int(metric["ot_observed_points"]),
                        "mass_policy": "uniform_empirical",
                        "subsampling_policy": (
                            "independent_without_replacement_by_seed"
                        ),
                    }
                )

        shared_indices = _shared_cohort_indices(
            full.shape[0],
            max_ot_points=max_ot_points,
            random_seed=seed,
        )
        full_subset = full[shared_indices]
        for condition in condition_order:
            candidate_subset = endpoints[condition][shared_indices]
            for space, columns in spaces.items():
                if condition == full_condition:
                    metric = {
                        "w1": 0.0,
                        "w2": 0.0,
                        "ot_predicted_points": len(shared_indices),
                        "ot_observed_points": len(shared_indices),
                    }
                else:
                    metric = compute_distribution_metrics(
                        candidate_subset[:, columns],
                        full_subset[:, columns],
                        predicted_weights=None,
                        max_ot_points=None,
                        random_seed=seed,
                    )
                identity = _identity_coupling_metrics(
                    candidate_subset[:, columns],
                    full_subset[:, columns],
                )
                full_cohort_identity = _identity_coupling_metrics(
                    endpoints[condition][:, columns],
                    full[:, columns],
                )
                eps = np.finfo(float).eps
                paired_rows.append(
                    {
                        "comparison": "condition_vs_full",
                        "condition": condition,
                        "reference": full_condition,
                        "space": space,
                        "sampling_seed": int(seed),
                        "is_primary_seed": bool(seed == int(primary_seed)),
                        "w1": float(metric["w1"]),
                        "w2": float(metric["w2"]),
                        "ot_condition_points": int(
                            metric["ot_predicted_points"]
                        ),
                        "ot_reference_points": int(metric["ot_observed_points"]),
                        **identity,
                        **{
                            f"full_cohort_{name}": value
                            for name, value in full_cohort_identity.items()
                        },
                        "exact_w1_fraction_of_identity_upper_bound": (
                            float(metric["w1"])
                            / identity["identity_coupling_w1_upper_bound"]
                            if identity["identity_coupling_w1_upper_bound"] > eps
                            else np.nan
                        ),
                        "exact_w2_fraction_of_identity_upper_bound": (
                            float(metric["w2"])
                            / identity["identity_coupling_w2_upper_bound"]
                            if identity["identity_coupling_w2_upper_bound"] > eps
                            else np.nan
                        ),
                        "mass_policy": "uniform_empirical",
                        "subsampling_policy": (
                            "shared_fixed_cohort_indices_without_replacement"
                        ),
                    }
                )

    condition_vs_observed = _add_full_deltas(
        pd.DataFrame(observed_rows),
        full_condition=full_condition,
    )
    condition_vs_full = pd.DataFrame(paired_rows)
    sensitivity = _summarize_sensitivity(
        condition_vs_observed,
        condition_vs_full,
    )
    settings = {
        "condition_order": condition_order,
        "full_condition": full_condition,
        "n_fixed_cohort_cells": int(full.shape[0]),
        "n_observed_cells": int(observed.shape[0]),
        "dim": int(full.shape[1]),
        "spatial_dim": int(spatial_dim),
        "state_dim": int(full.shape[1] - spatial_dim),
        "max_ot_points": (
            int(max_ot_points) if max_ot_points is not None else None
        ),
        "primary_seed": int(primary_seed),
        "sampling_sensitivity_seeds": list(seeds),
        "mass_policy": "uniform_empirical",
        "condition_vs_observed_sampling": (
            "independent empirical supports sampled without replacement with "
            "the same deterministic seed"
        ),
        "condition_vs_full_sampling": (
            "one shared fixed-cohort row subset applied to condition and full"
        ),
        "condition_vs_full_identity_coupling": (
            "known same-cell row matching; unprefixed values use the same "
            "sampled support as exact OT and are upper bounds on exact W1/W2; "
            "full_cohort_* values summarize all retained cells"
        ),
    }
    return FrozenAblationDistributionEvaluation(
        condition_vs_observed=condition_vs_observed,
        condition_vs_full=condition_vs_full,
        sensitivity_summary=sensitivity,
        settings=settings,
    )
