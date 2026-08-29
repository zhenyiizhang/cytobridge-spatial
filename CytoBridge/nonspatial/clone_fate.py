"""Clone-level fate agreement metrics for lineage-tracing experiments.

The functions in this module are deliberately independent of model training and
simulation.  A caller only needs to assign a lineage ID and a predicted fate
label to every generated endpoint, and a lineage ID and observed fate label to
every target cell.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CloneFateEvaluation:
    """Clone-level predicted/observed fate distributions and agreement scores.

    Attributes
    ----------
    categories
        Fate categories in the column order used by both distribution tables.
    predicted_distributions
        One row per evaluated lineage. Rows sum to one after applying generated
        endpoint weights.
    observed_distributions
        Empirical target-cell fate distributions for the same lineages.
    per_lineage
        Counts, total generated weight, and agreement metrics for every lineage.
    summary
        Clone-macro averages and filtering/provenance counts. Every evaluated
        lineage contributes equally to a ``clone_macro_*`` value.
    """

    categories: tuple[Any, ...]
    predicted_distributions: pd.DataFrame
    observed_distributions: pd.DataFrame
    per_lineage: pd.DataFrame
    summary: Mapping[str, Any]


@dataclass(frozen=True)
class PairedCloneBootstrapResult:
    """Paired bootstrap estimate of ``condition_a - condition_b``."""

    metric: str
    n_pairs: int
    n_bootstrap: int
    confidence_level: float
    condition_a_mean: float
    condition_b_mean: float
    mean_difference: float
    ci_low: float
    ci_high: float
    probability_difference_positive: float
    two_sided_pvalue: float
    bootstrap_differences: np.ndarray


def _as_object_vector(values: Sequence[Any] | np.ndarray, *, name: str) -> np.ndarray:
    if isinstance(values, (str, bytes)):
        raise ValueError(
            f"{name} must be a one-dimensional sequence, not a scalar string."
        )
    array = np.asarray(values, dtype=object)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {array.shape}.")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty.")
    return array


def _is_missing_scalar(value: Any) -> bool:
    missing = pd.isna(value)
    if isinstance(missing, (bool, np.bool_)):
        return bool(missing)
    raise ValueError(f"Expected scalar lineage IDs and labels, got {value!r}.")


def _validate_hashable_nonmissing(values: np.ndarray, *, name: str) -> None:
    for value in values:
        if _is_missing_scalar(value):
            raise ValueError(f"{name} contains a missing value.")
        try:
            hash(value)
        except TypeError as exc:
            raise ValueError(
                f"{name} values must be hashable scalars, got {value!r}."
            ) from exc


def _validate_min_count(value: int, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be at least 1, got {value}.")
    return value


def _resolve_categories(
    generated_labels: np.ndarray,
    observed_labels: np.ndarray,
    categories: Iterable[Any] | None,
) -> tuple[tuple[Any, ...], dict[Any, int]]:
    if categories is None:
        # Observed categories come first because they define the biological
        # target universe; generated-only labels are retained rather than
        # silently discarded.
        ordered: list[Any] = []
        seen: set[Any] = set()
        for value in np.concatenate((observed_labels, generated_labels)):
            if value not in seen:
                seen.add(value)
                ordered.append(value)
    else:
        if isinstance(categories, (str, bytes)):
            raise ValueError("categories must be a sequence, not a scalar string.")
        ordered = list(categories)
        if not ordered:
            raise ValueError("categories must not be empty.")

    category_array = np.asarray(ordered, dtype=object)
    if category_array.ndim != 1:
        raise ValueError("categories must contain hashable scalar values.")
    _validate_hashable_nonmissing(category_array, name="categories")
    if len(set(ordered)) != len(ordered):
        raise ValueError("categories must be unique.")

    category_to_index = {value: index for index, value in enumerate(ordered)}
    unknown_generated = [
        value for value in generated_labels if value not in category_to_index
    ]
    unknown_observed = [
        value for value in observed_labels if value not in category_to_index
    ]
    if unknown_generated or unknown_observed:
        unknown = list(dict.fromkeys(unknown_generated + unknown_observed))
        raise ValueError(f"Labels outside the declared categories: {unknown!r}.")
    return tuple(ordered), category_to_index


def _ordered_count_distributions(
    lineage_ids: np.ndarray,
    labels: np.ndarray,
    category_to_index: Mapping[Any, int],
    *,
    weights: np.ndarray | None,
) -> tuple[list[Any], dict[Any, int], dict[Any, float], dict[Any, np.ndarray]]:
    lineages: list[Any] = []
    counts: dict[Any, int] = {}
    totals: dict[Any, float] = {}
    distributions: dict[Any, np.ndarray] = {}
    n_categories = len(category_to_index)

    if weights is None:
        weights = np.ones(lineage_ids.shape[0], dtype=float)

    for lineage, label, weight in zip(lineage_ids, labels, weights):
        if lineage not in counts:
            lineages.append(lineage)
            counts[lineage] = 0
            totals[lineage] = 0.0
            distributions[lineage] = np.zeros(n_categories, dtype=float)
        counts[lineage] += 1
        numeric_weight = float(weight)
        totals[lineage] += numeric_weight
        distributions[lineage][category_to_index[label]] += numeric_weight

    return lineages, counts, totals, distributions


def _normalized_js_divergence(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Jensen-Shannon divergence normalized to [0, 1] using log base e."""
    midpoint = 0.5 * (p + q)
    p_term = np.zeros_like(p, dtype=float)
    q_term = np.zeros_like(q, dtype=float)
    p_positive = p > 0
    q_positive = q > 0
    p_term[p_positive] = p[p_positive] * np.log(p[p_positive] / midpoint[p_positive])
    q_term[q_positive] = q[q_positive] * np.log(q[q_positive] / midpoint[q_positive])
    divergence = 0.5 * (p_term.sum(axis=1) + q_term.sum(axis=1))
    normalized = divergence / np.log(2.0)
    return np.clip(normalized, 0.0, 1.0)


def evaluate_clone_fate_agreement(
    source_lineage_ids: Sequence[Any] | np.ndarray,
    generated_endpoint_labels: Sequence[Any] | np.ndarray,
    observed_target_lineage_ids: Sequence[Any] | np.ndarray,
    observed_target_labels: Sequence[Any] | np.ndarray,
    *,
    generated_endpoint_weights: Sequence[float] | np.ndarray | None = None,
    categories: Iterable[Any] | None = None,
    min_source: int = 1,
    min_target: int = 1,
) -> CloneFateEvaluation:
    """Compare generated and observed fate distributions within each lineage.

    Parameters
    ----------
    source_lineage_ids
        Lineage ID for every generated endpoint. String lineage IDs are fully
        supported. Repeated IDs represent multiple endpoints from one lineage.
    generated_endpoint_labels
        Predicted fate category for every generated endpoint.
    observed_target_lineage_ids, observed_target_labels
        Lineage IDs and measured fate categories for observed target cells.
    generated_endpoint_weights
        Optional non-negative endpoint weights. If omitted, endpoints receive
        equal weight. Every source lineage must have positive total weight.
    categories
        Optional ordered category universe. Labels outside it are rejected.
        If omitted, observed categories followed by generated-only categories
        define the order.
    min_source, min_target
        Minimum generated-endpoint and observed-target-cell counts required for
        a lineage. Only lineages present on both sides are evaluated.

    Notes
    -----
    ``tv_agreement`` is the distribution overlap
    ``1 - 0.5 * sum(abs(predicted - observed))``. ``js_similarity`` is one
    minus Jensen-Shannon divergence normalized by ``log(2)``. Dominant-fate
    matching is tie-aware: it is true when the two sets of maximum-probability
    categories overlap.
    """
    source_ids = _as_object_vector(source_lineage_ids, name="source_lineage_ids")
    generated_labels = _as_object_vector(
        generated_endpoint_labels,
        name="generated_endpoint_labels",
    )
    target_ids = _as_object_vector(
        observed_target_lineage_ids,
        name="observed_target_lineage_ids",
    )
    target_labels = _as_object_vector(
        observed_target_labels, name="observed_target_labels"
    )

    if source_ids.shape[0] != generated_labels.shape[0]:
        raise ValueError(
            "source_lineage_ids and generated_endpoint_labels must have equal length, "
            f"got {source_ids.shape[0]} and {generated_labels.shape[0]}."
        )
    if target_ids.shape[0] != target_labels.shape[0]:
        raise ValueError(
            "observed_target_lineage_ids and observed_target_labels must have equal length, "
            f"got {target_ids.shape[0]} and {target_labels.shape[0]}."
        )

    _validate_hashable_nonmissing(source_ids, name="source_lineage_ids")
    _validate_hashable_nonmissing(target_ids, name="observed_target_lineage_ids")
    _validate_hashable_nonmissing(generated_labels, name="generated_endpoint_labels")
    _validate_hashable_nonmissing(target_labels, name="observed_target_labels")
    min_source = _validate_min_count(min_source, name="min_source")
    min_target = _validate_min_count(min_target, name="min_target")

    if generated_endpoint_weights is None:
        weights = np.ones(source_ids.shape[0], dtype=float)
    else:
        weights = np.asarray(generated_endpoint_weights, dtype=float)
        if weights.ndim != 1:
            raise ValueError(
                "generated_endpoint_weights must be one-dimensional, "
                f"got shape {weights.shape}."
            )
        if weights.shape[0] != source_ids.shape[0]:
            raise ValueError(
                "generated_endpoint_weights must match source_lineage_ids length, "
                f"got {weights.shape[0]} and {source_ids.shape[0]}."
            )
        if not np.all(np.isfinite(weights)):
            raise ValueError("generated_endpoint_weights must all be finite.")
        if np.any(weights < 0):
            raise ValueError("generated_endpoint_weights must be non-negative.")

    resolved_categories, category_to_index = _resolve_categories(
        generated_labels,
        target_labels,
        categories,
    )
    (
        source_order,
        source_counts,
        source_weight_totals,
        predicted_weight_counts,
    ) = _ordered_count_distributions(
        source_ids,
        generated_labels,
        category_to_index,
        weights=weights,
    )
    (
        _,
        target_counts,
        _,
        observed_counts,
    ) = _ordered_count_distributions(
        target_ids,
        target_labels,
        category_to_index,
        weights=None,
    )

    zero_weight_lineages = [
        lineage for lineage, total in source_weight_totals.items() if total <= 0.0
    ]
    if zero_weight_lineages:
        raise ValueError(
            "Every source lineage must have positive total generated weight; "
            f"zero-weight lineages: {zero_weight_lineages!r}."
        )

    common_lineages = [lineage for lineage in source_order if lineage in target_counts]
    eligible_lineages = [
        lineage
        for lineage in common_lineages
        if source_counts[lineage] >= min_source and target_counts[lineage] >= min_target
    ]
    if not eligible_lineages:
        raise ValueError(
            "No lineages remain after intersecting source/target IDs and applying "
            f"min_source={min_source}, min_target={min_target}."
        )

    predicted = np.stack(
        [
            predicted_weight_counts[lineage] / source_weight_totals[lineage]
            for lineage in eligible_lineages
        ]
    )
    observed = np.stack(
        [
            observed_counts[lineage] / float(target_counts[lineage])
            for lineage in eligible_lineages
        ]
    )

    tv_distance = 0.5 * np.abs(predicted - observed).sum(axis=1)
    tv_agreement = np.clip(1.0 - tv_distance, 0.0, 1.0)
    js_divergence = _normalized_js_divergence(predicted, observed)
    js_similarity = np.clip(1.0 - js_divergence, 0.0, 1.0)

    predicted_max = predicted.max(axis=1, keepdims=True)
    observed_max = observed.max(axis=1, keepdims=True)
    predicted_ties = np.isclose(predicted, predicted_max, rtol=1e-12, atol=1e-15)
    observed_ties = np.isclose(observed, observed_max, rtol=1e-12, atol=1e-15)
    dominant_match = np.logical_and(predicted_ties, observed_ties).any(axis=1)
    predicted_argmax = predicted.argmax(axis=1)
    observed_argmax = observed.argmax(axis=1)

    index = pd.Index(eligible_lineages, name="lineage_id")
    predicted_frame = pd.DataFrame(predicted, index=index, columns=resolved_categories)
    observed_frame = pd.DataFrame(observed, index=index, columns=resolved_categories)
    per_lineage = pd.DataFrame(
        {
            "n_source": [source_counts[lineage] for lineage in eligible_lineages],
            "n_target": [target_counts[lineage] for lineage in eligible_lineages],
            "generated_weight": [
                source_weight_totals[lineage] for lineage in eligible_lineages
            ],
            "tv_distance": tv_distance,
            "tv_agreement": tv_agreement,
            "js_divergence": js_divergence,
            "js_similarity": js_similarity,
            "predicted_dominant_fate": [
                resolved_categories[index_value] for index_value in predicted_argmax
            ],
            "observed_dominant_fate": [
                resolved_categories[index_value] for index_value in observed_argmax
            ],
            "dominant_fate_match": dominant_match,
        },
        index=index,
    )

    summary: dict[str, Any] = {
        "categories": resolved_categories,
        "min_source": min_source,
        "min_target": min_target,
        "n_generated_endpoints": int(source_ids.shape[0]),
        "n_observed_target_cells": int(target_ids.shape[0]),
        "n_source_lineages": int(len(source_counts)),
        "n_target_lineages": int(len(target_counts)),
        "n_common_lineages": int(len(common_lineages)),
        "n_evaluated_lineages": int(len(eligible_lineages)),
        "clone_macro_tv_distance": float(np.mean(tv_distance)),
        "clone_macro_tv_agreement": float(np.mean(tv_agreement)),
        "clone_macro_js_divergence": float(np.mean(js_divergence)),
        "clone_macro_js_similarity": float(np.mean(js_similarity)),
        "clone_macro_dominant_fate_match": float(np.mean(dominant_match)),
    }
    return CloneFateEvaluation(
        categories=resolved_categories,
        predicted_distributions=predicted_frame,
        observed_distributions=observed_frame,
        per_lineage=per_lineage,
        summary=summary,
    )


def _metric_series(
    result: CloneFateEvaluation | pd.DataFrame | pd.Series,
    *,
    metric: str,
    condition_name: str,
) -> pd.Series:
    if isinstance(result, CloneFateEvaluation):
        frame_or_series: pd.DataFrame | pd.Series = result.per_lineage
    else:
        frame_or_series = result

    if isinstance(frame_or_series, pd.DataFrame):
        if metric not in frame_or_series.columns:
            raise KeyError(f"Metric {metric!r} not found for {condition_name}.")
        series = frame_or_series[metric]
    elif isinstance(frame_or_series, pd.Series):
        series = frame_or_series
    else:
        raise TypeError(
            f"{condition_name} must be CloneFateEvaluation, DataFrame, or Series, "
            f"got {type(frame_or_series).__name__}."
        )

    if not series.index.is_unique:
        raise ValueError(f"{condition_name} lineage index must be unique.")
    numeric = pd.to_numeric(series, errors="coerce").astype(float)
    if numeric.isna().any() or not np.all(np.isfinite(numeric.to_numpy())):
        raise ValueError(
            f"{condition_name} metric {metric!r} must contain only finite values."
        )
    return numeric


def paired_bootstrap_clone_metric_difference(
    condition_a: CloneFateEvaluation | pd.DataFrame | pd.Series,
    condition_b: CloneFateEvaluation | pd.DataFrame | pd.Series,
    *,
    metric: str = "tv_agreement",
    n_bootstrap: int = 10_000,
    confidence_level: float = 0.95,
    seed: int | None = 0,
) -> PairedCloneBootstrapResult:
    """Bootstrap a clone-macro paired condition difference.

    The two inputs are aligned by lineage index, and only their common lineages
    are resampled. The reported effect is always ``condition_a - condition_b``.
    The helper accepts full :class:`CloneFateEvaluation` objects, per-lineage
    DataFrames, or metric Series.
    """
    if isinstance(n_bootstrap, (bool, np.bool_)) or not isinstance(
        n_bootstrap, (int, np.integer)
    ):
        raise ValueError("n_bootstrap must be a positive integer.")
    n_bootstrap = int(n_bootstrap)
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be at least 1.")
    confidence_level = float(confidence_level)
    if not np.isfinite(confidence_level) or not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be strictly between 0 and 1.")

    series_a = _metric_series(condition_a, metric=metric, condition_name="condition_a")
    series_b = _metric_series(condition_b, metric=metric, condition_name="condition_b")
    common = series_a.index.intersection(series_b.index, sort=False)
    if common.empty:
        raise ValueError("condition_a and condition_b have no common lineage IDs.")

    values_a = series_a.loc[common].to_numpy(dtype=float)
    values_b = series_b.loc[common].to_numpy(dtype=float)
    paired_differences = values_a - values_b
    rng = np.random.default_rng(seed)
    # Bound temporary index memory for experiments with many thousands of
    # lineages.  The result is invariant to this chunking for a fixed RNG seed.
    bootstrap_differences = np.empty(n_bootstrap, dtype=float)
    max_indices_per_chunk = 1_000_000
    chunk_size = max(
        1, min(n_bootstrap, max_indices_per_chunk // len(paired_differences))
    )
    for start in range(0, n_bootstrap, chunk_size):
        stop = min(start + chunk_size, n_bootstrap)
        sample_indices = rng.integers(
            0,
            paired_differences.shape[0],
            size=(stop - start, paired_differences.shape[0]),
        )
        bootstrap_differences[start:stop] = paired_differences[sample_indices].mean(
            axis=1
        )

    alpha = 1.0 - confidence_level
    ci_low, ci_high = np.quantile(
        bootstrap_differences,
        [alpha / 2.0, 1.0 - alpha / 2.0],
    )
    probability_positive = float(np.mean(bootstrap_differences > 0.0))
    probability_nonpositive = float(np.mean(bootstrap_differences <= 0.0))
    probability_nonnegative = float(np.mean(bootstrap_differences >= 0.0))
    two_sided_pvalue = min(
        1.0,
        2.0 * min(probability_nonpositive, probability_nonnegative),
    )

    return PairedCloneBootstrapResult(
        metric=str(metric),
        n_pairs=int(paired_differences.shape[0]),
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        condition_a_mean=float(np.mean(values_a)),
        condition_b_mean=float(np.mean(values_b)),
        mean_difference=float(np.mean(paired_differences)),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        probability_difference_positive=probability_positive,
        two_sided_pvalue=float(two_sided_pvalue),
        bootstrap_differences=bootstrap_differences,
    )


__all__ = [
    "CloneFateEvaluation",
    "PairedCloneBootstrapResult",
    "evaluate_clone_fate_agreement",
    "paired_bootstrap_clone_metric_difference",
]
