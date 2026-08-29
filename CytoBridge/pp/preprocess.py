import scanpy as sc
from anndata import AnnData
import pandas as pd
import numpy as np
import logging
import json
import re
from urllib.parse import quote
from collections.abc import Mapping
from typing import Optional, Dict, Sequence
from scipy import sparse
from scipy.spatial import cKDTree

_TIME_PATTERN = re.compile(r"-?\d+(?:\.\d+)?")


def _resolve_observation_names(
    adata: AnnData,
    identity_keys: Optional[Sequence[str]],
) -> Dict[str, object]:
    """Require stable cell IDs, optionally building them from named columns.

    AnnData permits duplicate index values, but joins and dictionaries do not.
    A dataset that reuses local cell IDs across sections must therefore declare
    the columns that form its stable identity (for example ``Batch`` and
    ``CellID``).  We intentionally do not invent row-order suffixes: those IDs
    change when the same input is reordered.
    """

    source_names = pd.Index(adata.obs_names.astype(str))
    duplicate_rows = int(source_names.duplicated(keep=False).sum())
    duplicate_values = int(source_names[source_names.duplicated()].nunique())
    resolved_keys = tuple(str(key).strip() for key in (identity_keys or ()))
    if any(not key for key in resolved_keys):
        raise ValueError("observation_id_keys cannot contain empty column names.")
    if len(set(resolved_keys)) != len(resolved_keys):
        raise ValueError("observation_id_keys must not contain duplicate columns.")

    if not resolved_keys and duplicate_rows == 0:
        return {
            "input_names_unique": True,
            "duplicate_rows": 0,
            "duplicate_values": 0,
            "strategy": "existing_index",
            "identity_keys": [],
            "original_name_column": "none",
        }

    if not resolved_keys:
        raise ValueError(
            "Input observation names are not unique. Declare stable "
            "preprocess.align.observation_id_keys (for example ['Batch', "
            "'CellID']) instead of relying on input row order."
        )

    missing_keys = [key for key in resolved_keys if key not in adata.obs]
    if missing_keys:
        raise KeyError(
            "observation_id_keys contains columns that are absent from adata.obs: "
            f"{missing_keys}. Available columns: {list(adata.obs.columns)}"
        )

    composite_names = []
    for row_index, values in enumerate(
        adata.obs.loc[:, list(resolved_keys)].itertuples(index=False, name=None)
    ):
        parts = []
        for key, value in zip(resolved_keys, values):
            if pd.isna(value):
                raise ValueError(
                    f"observation identity column {key!r} is missing at row "
                    f"{row_index}."
                )
            parts.append(f"{key}={quote(str(value), safe='-_.~')}")
        composite_names.append("|".join(parts))

    composite_index = pd.Index(composite_names)
    if not composite_index.is_unique:
        duplicate_examples = composite_index[composite_index.duplicated()].unique()[:5]
        raise ValueError(
            "observation_id_keys do not form a unique cell identity. Duplicate "
            f"examples: {duplicate_examples.tolist()}"
        )

    original_column = "original_obs_name"
    if original_column in adata.obs:
        recorded = adata.obs[original_column].astype(str).to_numpy()
        if not np.array_equal(recorded, source_names.to_numpy()):
            raise ValueError(
                "Duplicate observation names require preserving their source IDs in "
                f"obs[{original_column!r}], but that reserved column already contains "
                "different values. Rename the existing column before preprocessing."
            )
    else:
        adata.obs[original_column] = source_names.to_numpy()

    adata.obs_names = composite_index
    return {
        "input_names_unique": duplicate_rows == 0,
        "duplicate_rows": duplicate_rows,
        "duplicate_values": duplicate_values,
        "strategy": "composite_obs_columns",
        "identity_keys": list(resolved_keys),
        "original_name_column": original_column,
    }


def _sample_dense_rows(matrix, n_rows: int = 256) -> np.ndarray:
    """Return a bounded deterministic row sample without densifying a full matrix."""
    total_rows = int(matrix.shape[0])
    if total_rows == 0:
        return np.empty((0, int(matrix.shape[1])), dtype=np.float32)
    row_idx = np.unique(
        np.linspace(0, total_rows - 1, num=min(total_rows, n_rows), dtype=int)
    )
    sampled = matrix[row_idx]
    if sparse.issparse(sampled):
        return sampled.toarray()
    if hasattr(sampled, "to_memory"):
        sampled = sampled.to_memory()
        if sparse.issparse(sampled):
            return sampled.toarray()
    return np.asarray(sampled)


def _relation_stats(values: np.ndarray, target: np.ndarray) -> Dict[str, float | int]:
    close = np.isclose(values, target, rtol=1e-5, atol=1e-7, equal_nan=False)
    errors = np.abs(values - target)
    return {
        "n_compared": int(values.size),
        "n_close": int(np.count_nonzero(close)),
        "mismatch_count": int(values.size - np.count_nonzero(close)),
        "fraction_close": float(np.mean(close)) if close.size else 0.0,
        "max_abs_error": float(np.max(errors)) if errors.size else 0.0,
    }


def _detect_x_state_against_counts(
    adata: AnnData,
    counts_layer: str = "counts",
) -> Dict[str, object]:
    """Detect whether X is raw counts or a (near) log1p copy of a count layer."""
    result: Dict[str, object] = {
        "state": "unknown",
        "counts_layer": counts_layer,
        "comparison": "unavailable",
    }
    if counts_layer not in adata.layers:
        if "log1p" in adata.uns or bool(
            adata.uns.get("preprocess_info", {}).get("log1p", False)
        ):
            result["state"] = "transformed_from_metadata"
        return result

    x_matrix = adata.X
    count_matrix = adata.layers[counts_layer]
    values = None
    targets = None
    comparison = "sampled_rows"
    support_match_fraction = 0.0

    if sparse.issparse(x_matrix) and sparse.issparse(count_matrix):
        x_csr = x_matrix.tocsr(copy=False)
        count_csr = count_matrix.tocsr(copy=False)
        if (
            x_csr.shape == count_csr.shape
            and np.array_equal(x_csr.indptr, count_csr.indptr)
            and np.array_equal(x_csr.indices, count_csr.indices)
        ):
            values = np.asarray(x_csr.data, dtype=np.float64)
            targets = np.asarray(count_csr.data, dtype=np.float64)
            support_match_fraction = 1.0
            comparison = "full_sparse_nonzero_values"

    if values is None or targets is None:
        x_sample = _sample_dense_rows(x_matrix)
        count_sample = _sample_dense_rows(count_matrix)
        if x_sample.shape != count_sample.shape:
            return result
        support_x = x_sample != 0
        support_counts = count_sample != 0
        support_match_fraction = float(np.mean(support_x == support_counts))
        compared = support_x | support_counts
        values = np.asarray(x_sample[compared], dtype=np.float64)
        targets = np.asarray(count_sample[compared], dtype=np.float64)

    raw_stats = _relation_stats(values, targets)
    log_stats = _relation_stats(values, np.log1p(targets))
    result.update(
        {
            "comparison": comparison,
            "support_match_fraction": support_match_fraction,
            "raw_counts": raw_stats,
            "log1p_counts": log_stats,
        }
    )
    relation_threshold = 0.9999
    if (
        support_match_fraction >= relation_threshold
        and raw_stats["fraction_close"] >= relation_threshold
    ):
        result["state"] = "near_raw_counts"
    elif (
        support_match_fraction >= relation_threshold
        and log_stats["fraction_close"] >= relation_threshold
    ):
        result["state"] = "near_log1p_of_counts"
    elif "log1p" in adata.uns or bool(
        adata.uns.get("preprocess_info", {}).get("log1p", False)
    ):
        result["state"] = "transformed_from_metadata"
    return result


def _matrix_value_stats(matrix) -> Dict[str, object]:
    """Record bounded validation statistics for an expression source."""
    sampled = not sparse.issparse(matrix)
    if sparse.issparse(matrix):
        values = np.asarray(matrix.data, dtype=np.float64)
    else:
        values = np.asarray(_sample_dense_rows(matrix), dtype=np.float64).ravel()
    finite = np.isfinite(values)
    finite_values = values[finite]
    return {
        "sampled": bool(sampled),
        "n_values_checked": int(values.size),
        "all_finite": bool(np.all(finite)),
        "nonnegative": bool(np.all(finite_values >= 0)) if finite_values.size else True,
        "integer_like_fraction": (
            float(np.mean(np.isclose(finite_values, np.rint(finite_values), atol=1e-7)))
            if finite_values.size
            else 1.0
        ),
        "min": float(np.min(finite_values)) if finite_values.size else 0.0,
        "max": float(np.max(finite_values)) if finite_values.size else 0.0,
    }


def _raw_count_value_stats(
    matrix,
    *,
    integer_tolerance: float,
    chunk_rows: int = 4096,
) -> Dict[str, object]:
    """Validate a complete matrix without densifying a complete sparse input.

    Sparse implicit zeros satisfy the raw-count contract, so only explicitly
    stored values need to be visited. Dense/backed matrices are inspected in
    bounded row chunks. Unlike :func:`_matrix_value_stats`, this function does
    not sample rows.
    """
    integer_tolerance = float(integer_tolerance)
    if not np.isfinite(integer_tolerance) or integer_tolerance < 0:
        raise ValueError(
            "raw_count_integer_tolerance must be a finite non-negative number, "
            f"got {integer_tolerance}."
        )

    n_values = 0
    n_finite = 0
    n_nonnegative = 0
    n_integer_like = 0
    value_min = np.inf
    value_max = -np.inf

    if sparse.issparse(matrix):
        value_blocks = (np.asarray(matrix.data),)
        matrix_storage = "sparse_full_explicit_values"
    else:
        n_rows = int(matrix.shape[0])

        def _dense_blocks():
            for start in range(0, n_rows, int(chunk_rows)):
                block = matrix[start : start + int(chunk_rows)]
                if hasattr(block, "to_memory"):
                    block = block.to_memory()
                if sparse.issparse(block):
                    # A backed sparse block may materialize as sparse even when
                    # the original wrapper is not recognized by scipy.sparse.
                    block = block.toarray()
                yield np.asarray(block)

        value_blocks = _dense_blocks()
        matrix_storage = "dense_full_chunked"

    for block in value_blocks:
        values = np.asarray(block, dtype=np.float64).ravel()
        n_values += int(values.size)
        finite_mask = np.isfinite(values)
        finite_values = values[finite_mask]
        n_finite += int(finite_values.size)
        if finite_values.size == 0:
            continue
        n_nonnegative += int(np.count_nonzero(finite_values >= 0))
        integer_like = np.isclose(
            finite_values,
            np.rint(finite_values),
            rtol=0.0,
            atol=integer_tolerance,
        )
        n_integer_like += int(np.count_nonzero(integer_like))
        value_min = min(value_min, float(np.min(finite_values)))
        value_max = max(value_max, float(np.max(finite_values)))

    return {
        "validation_scope": matrix_storage,
        "integer_tolerance": integer_tolerance,
        "n_values_checked": int(n_values),
        "n_nonfinite": int(n_values - n_finite),
        "n_negative": int(n_finite - n_nonnegative),
        "n_noninteger_like": int(n_finite - n_integer_like),
        "all_finite": bool(n_values == n_finite),
        "nonnegative": bool(n_finite == n_nonnegative),
        "integer_like_fraction": (
            float(n_integer_like / n_finite) if n_finite else 1.0
        ),
        "min": float(value_min) if n_finite else 0.0,
        "max": float(value_max) if n_finite else 0.0,
    }


def _validate_raw_count_like(
    matrix,
    *,
    source: str,
    integer_tolerance: float,
) -> Dict[str, object]:
    """Require finite, non-negative, integer-like values in the full source."""
    stats = _raw_count_value_stats(
        matrix,
        integer_tolerance=integer_tolerance,
    )
    failures = []
    if not stats["all_finite"]:
        failures.append(f"{stats['n_nonfinite']} non-finite values")
    if not stats["nonnegative"]:
        failures.append(f"{stats['n_negative']} negative values")
    if stats["n_noninteger_like"]:
        failures.append(
            f"{stats['n_noninteger_like']} values outside integer tolerance "
            f"{stats['integer_tolerance']}"
        )
    if failures:
        raise ValueError(
            f"Expression source {source} failed strict raw-count-like validation: "
            + "; ".join(failures)
            + ". Select the actual raw-count layer, or set "
            "raw_count_validation='off' only for an explicitly documented "
            "non-count preprocessing contract."
        )
    return stats


def _resolved_time_mapping(
    unique_times,
    time_mapping: Mapping,
) -> Dict[object, float]:
    """Resolve a mapping against observed labels, tolerating JSON string keys.

    JSON objects necessarily stringify numeric keys. Exact key matches always
    win; otherwise a unique string or finite-numeric equivalent is accepted.
    Ambiguous equivalents are rejected rather than choosing by input order.
    """
    resolved: Dict[object, float] = {}
    missing = []
    mapping_items = list(time_mapping.items())
    for observed in unique_times:
        if observed in time_mapping:
            target = time_mapping[observed]
        else:
            equivalent_items = [
                (key, value)
                for key, value in mapping_items
                if str(key) == str(observed)
            ]
            if not isinstance(observed, (bool, np.bool_)):
                try:
                    observed_numeric = float(observed)
                except (TypeError, ValueError):
                    observed_numeric = None
                if observed_numeric is not None and np.isfinite(observed_numeric):
                    for key, value in mapping_items:
                        if any(
                            existing_key == key for existing_key, _ in equivalent_items
                        ):
                            continue
                        if isinstance(key, (bool, np.bool_)):
                            continue
                        try:
                            key_numeric = float(key)
                        except (TypeError, ValueError):
                            continue
                        if np.isfinite(key_numeric) and key_numeric == observed_numeric:
                            equivalent_items.append((key, value))
            if len(equivalent_items) == 1:
                target = equivalent_items[0][1]
            elif len(equivalent_items) > 1:
                equivalent_keys = [key for key, _ in equivalent_items]
                raise ValueError(
                    "time_mapping contains ambiguous equivalent keys for "
                    f"observed time {observed!r}: {equivalent_keys}."
                )
            else:
                missing.append(observed)
                continue
        try:
            numeric_target = float(target)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"time_mapping target for {observed!r} must be numeric, got {target!r}."
            ) from exc
        if not np.isfinite(numeric_target):
            raise ValueError(
                f"time_mapping target for {observed!r} must be finite, got {target!r}."
            )
        resolved[observed] = numeric_target
    if missing:
        raise ValueError(
            "The following time points are not present in the provided "
            f"time_mapping: {missing}"
        )
    return resolved


def _time_mapping_json(mapping: Mapping) -> str:
    """Return an H5AD-safe, type-explicit representation for provenance."""
    records = []
    for source, target in mapping.items():
        if isinstance(source, np.generic):
            source = source.item()
        records.append(
            {
                "source": str(source),
                "source_type": type(source).__name__,
                "target": float(target),
            }
        )
    return json.dumps(records, ensure_ascii=False, separators=(",", ":"))


def _auto_time_order(unique_times) -> list:
    """Robust auto ordering for mixed/string time labels."""
    unique_times = list(unique_times)
    series = pd.Series(unique_times)
    if pd.api.types.is_numeric_dtype(series):
        return sorted(unique_times, key=float)

    parsed = []
    for idx, value in enumerate(unique_times):
        match = _TIME_PATTERN.search(str(value))
        parsed.append((float(match.group()) if match else None, idx, value))

    # If every label has an embedded number (e.g. 24hpf, E10.5), sort by that.
    if all(item[0] is not None for item in parsed):
        parsed.sort(key=lambda x: (x[0], x[1]))
        return [item[2] for item in parsed]

    # Fallback to observed order to avoid lexicographic mis-ordering.
    return unique_times


def _filter_spatial_outliers(
    adata: AnnData,
    *,
    spatial_key: str,
    group_key: str,
    robust_nn_z_threshold: float,
) -> tuple[AnnData, Dict[str, object]]:
    """Remove label-blind spatially isolated observations within each group."""
    spatial_key = str(spatial_key).strip()
    group_key = str(group_key).strip()
    threshold = float(robust_nn_z_threshold)
    if not spatial_key:
        raise ValueError("spatial_outlier_key must be a non-empty obsm key.")
    if not group_key:
        raise ValueError("spatial_outlier_group_key must be a non-empty obs key.")
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError(
            "spatial_outlier_nn_mad_z_threshold must be positive and finite, "
            f"got {robust_nn_z_threshold!r}."
        )
    if spatial_key not in adata.obsm:
        raise KeyError(
            f"spatial_outlier_key {spatial_key!r} is absent from adata.obsm. "
            f"Available keys: {list(adata.obsm.keys())}"
        )
    if group_key not in adata.obs:
        raise KeyError(
            f"spatial_outlier_group_key {group_key!r} is absent from adata.obs. "
            f"Available columns: {list(adata.obs.columns)}"
        )

    coordinates = np.asarray(adata.obsm[spatial_key], dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[0] != adata.n_obs:
        raise ValueError(
            f"adata.obsm[{spatial_key!r}] must have one coordinate row per observation."
        )
    if coordinates.shape[1] < 2:
        raise ValueError(
            f"adata.obsm[{spatial_key!r}] must contain at least two coordinate columns."
        )
    if not np.isfinite(coordinates).all():
        raise ValueError(
            f"adata.obsm[{spatial_key!r}] contains non-finite coordinates."
        )
    if adata.obs[group_key].isna().any():
        raise ValueError(
            f"adata.obs[{group_key!r}] contains missing grouping values."
        )

    flagged = np.zeros(adata.n_obs, dtype=bool)
    robust_z_all = np.full(adata.n_obs, np.nan, dtype=np.float64)
    nearest_all = np.full(adata.n_obs, np.nan, dtype=np.float64)
    group_summaries: list[Dict[str, object]] = []
    group_values = adata.obs[group_key].to_numpy()
    for group_value in pd.unique(group_values):
        positions = np.flatnonzero(group_values == group_value)
        if positions.size < 3:
            raise ValueError(
                f"Spatial outlier group {group_value!r} has only {positions.size} "
                "observations; at least three are required."
            )
        nearest = cKDTree(coordinates[positions]).query(
            coordinates[positions], k=2
        )[0][:, 1]
        median = float(np.median(nearest))
        mad = float(np.median(np.abs(nearest - median)))
        robust_scale = 1.4826 * mad
        if not np.isfinite(robust_scale) or robust_scale <= 0:
            raise ValueError(
                f"Spatial outlier group {group_value!r} has zero/non-finite "
                "nearest-neighbor MAD; the robust filter is undefined."
            )
        robust_z = (nearest - median) / robust_scale
        group_flagged = robust_z > threshold
        flagged[positions] = group_flagged
        robust_z_all[positions] = robust_z
        nearest_all[positions] = nearest
        group_summaries.append(
            {
                "group": str(group_value),
                "n_input": int(positions.size),
                "n_removed": int(group_flagged.sum()),
                "nearest_neighbor_median": median,
                "nearest_neighbor_mad": mad,
            }
        )

    removed_positions = np.flatnonzero(flagged)
    removed = [
        {
            "obs_name": str(adata.obs_names[position]),
            "group": str(adata.obs.iloc[position][group_key]),
            "nearest_neighbor_distance": float(nearest_all[position]),
            "robust_nn_z": float(robust_z_all[position]),
        }
        for position in removed_positions
    ]
    info: Dict[str, object] = {
        "enabled": True,
        "method": "within-group robust 1-nearest-neighbor isolation",
        "label_blind": True,
        "spatial_key": spatial_key,
        "group_key": group_key,
        "robust_nn_z_threshold": threshold,
        "n_input": int(adata.n_obs),
        "n_removed": int(flagged.sum()),
        "n_retained": int((~flagged).sum()),
        "group_summaries": group_summaries,
        "removed_observations": removed,
    }
    if not flagged.any():
        return adata, info
    return adata[~flagged].copy(), info


def preprocess(
    adata: AnnData,
    time_key: str,
    n_top_genes: int = 2000,
    dim_reduction: str = "pca",
    n_pcs: int = 50,
    time_mapping: Optional[Dict[object, float]] = None,
    normalization: bool = True,
    normalization_target_sum: Optional[float] = 1e4,
    log1p: bool = True,
    select_hvg: bool = True,
    expression_layer: Optional[str] = None,
    allow_retransform_preprocessed_x: bool = False,
    counts_layer: str = "counts",
    raw_count_validation: str = "auto",
    raw_count_integer_tolerance: float = 1e-6,
    required_latent_features: Optional[Sequence[str]] = None,
    observation_id_keys: Optional[Sequence[str]] = None,
    hvg_batch_key: Optional[str] = None,
    hvg_selection_transform: str = "post_transform",
    normalization_reference: str = "all_features",
    latent_fit_obs_values: Optional[Sequence[object]] = None,
    spatial_outlier_filter: bool = False,
    spatial_outlier_key: str = "spatial",
    spatial_outlier_group_key: Optional[str] = None,
    spatial_outlier_nn_mad_z_threshold: float = 50.0,
) -> AnnData:
    """
    Preprocess step for dynamical optimal transport analysis.

    This function implements a standard preprocessing workflow. Key steps include:
    1. Mapping categorical time points to a numerical scale.
    2. Normalizing and log-transforming the data.
    3. Selecting highly variable genes (HVGs).
    4. Scaling the data and performing PCA.

    Parameters
    ----------
    adata
        The annotated data matrix.
    time_key
        The key in `adata.obs` that specifies the time point of each cell.
    n_top_genes
        Number of highly variable genes to select.
    dim_reduction
        The dimension reduction method to use.
    n_pcs
        Number of principal components to compute.
    time_mapping
        A dictionary to map non-numeric time points to numeric values.
        Example: {'Day0': 0, 'Day2': 2.0, 'Day7': 7.0}.
        If None (default), time points are automatically sorted and mapped to
        0, 1, 2, ...
    normalization
        If True, normalize the data.
    normalization_target_sum
        Target total count passed to :func:`scanpy.pp.normalize_total`. Use
        ``None`` to normalize to the median total count, matching Scanpy's
        historical default and the published ARISTA preprocessing notebook.
    log1p
        If True, apply log1p transformation to the data.
    select_hvg
        If True, select highly variable genes.
    expression_layer
        Optional layer to copy into ``adata.X`` before normalization and
        log-transformation. Use this when ``adata.X`` is already transformed
        but a raw-count layer (for example ``layers['counts']``) is available.
        ``None`` preserves the historical behavior of preprocessing the
        existing ``adata.X`` matrix.
    allow_retransform_preprocessed_x
        Permit normalization/log1p when the existing ``adata.X`` is detected
        as already transformed relative to ``layers['counts']``. This is off
        by default to prevent silent double transformation; enable it only for
        an explicitly labelled legacy reproduction.
    counts_layer
        Compatibility layer name used to preserve pre-transformation counts
        when ``expression_layer`` is not supplied. When an explicit
        ``expression_layer`` is selected, that selected layer becomes the
        canonical raw-expression source recorded for downstream graph
        construction; a stale pre-existing ``layers['counts']`` is not used in
        its place.
    raw_count_validation
        ``'strict'`` validates the complete selected expression source as
        finite, non-negative, and integer-like. ``'auto'`` (default) applies
        the same strict validation when an explicit expression layer is about
        to be normalized or log-transformed, while preserving the historical
        X-first behavior otherwise. ``'off'`` retains only basic finite and
        non-negative checks.
    raw_count_integer_tolerance
        Absolute tolerance used by strict integer-like validation.
    required_latent_features
        Optional feature names that must participate in the PCA fit in
        addition to the statistically selected HVGs.  This is intended for
        dataset adapters that need reconstructable marker or ligand/receptor
        genes. Missing names raise instead of being silently ignored.
    observation_id_keys
        Optional ``adata.obs`` columns that jointly define a stable cell ID.
        Required when input observation names are duplicated.
    hvg_batch_key
        Optional observation column for batch-aware highly-variable-gene
        selection. The complete input population still participates in the
        PCA fit before any later alignment-time subset is selected.
    hvg_selection_transform
        ``'post_transform'`` (default) preserves the standard workflow and
        ranks HVGs after the main normalization/log1p transform.
        ``'log1p_counts'`` ranks HVGs on an independent ``log1p`` view of the
        selected raw-count expression source, without total-count
        normalization. The latter separates feature ranking from the clean
        expression transform used for the latent state.
    normalization_reference
        ``'all_features'`` (default) computes per-cell size factors from every
        gene. ``'latent_features'`` computes size factors from the final HVG +
        required-feature mask, then applies those factors to the full gene
        matrix before log1p. This keeps full gene-space expression available
        while making the latent feature universe define its own scale.
    latent_fit_obs_values
        Optional values from ``adata.obs[time_key]`` to retain after HVG
        ranking but before normalization and PCA. This permits HVGs to be
        selected on a reference population while the latent transform is fit
        only on the observations used by the model.
    spatial_outlier_filter
        If True, remove spatially isolated observations after any independent
        HVG-reference ranking and latent observation subsetting, but before
        normalization and PCA. The filter is label-blind and disabled by
        default.
    spatial_outlier_key
        ``adata.obsm`` key used by the spatial-isolation filter.
    spatial_outlier_group_key
        Optional ``adata.obs`` column within which nearest-neighbor distances
        are calibrated independently. Defaults to ``time_key``.
    spatial_outlier_nn_mad_z_threshold
        Robust within-group 1-nearest-neighbor z-score threshold. The score is
        ``(distance - median(distance)) / (1.4826 * MAD(distance))``.
    Returns
    -------
    AnnData
        Processed AnnData used for training/downstream.
        `adata.X` is kept as gene-level expression matrix (after optional normalization/log1p/HVG).
        Reduced representation is stored in `adata.obsm['X_latent']`.
    """
    # --- Input Validation and Setup ---
    if time_key not in adata.obs.keys():
        raise KeyError(
            f"The specified time_key '{time_key}' was not found in adata.obs. "
            f"Available keys are: {list(adata.obs.keys())}"
        )
    print(f"Using '{time_key}' as the time point identifier.")
    observation_name_info = _resolve_observation_names(adata, observation_id_keys)
    if observation_name_info["strategy"] == "composite_obs_columns":
        print(
            "Built stable observation names from obs columns "
            f"{observation_name_info['identity_keys']} while preserving source IDs "
            "in obs['original_obs_name']."
        )

    dim_reduction_name = (
        "none" if dim_reduction is None else str(dim_reduction).strip().lower()
    )
    if dim_reduction_name not in {"pca", "umap", "none"}:
        raise ValueError(f"Invalid dimension reduction method: {dim_reduction}")

    counts_layer = str(counts_layer).strip()
    if not counts_layer:
        raise ValueError("counts_layer must be a non-empty layer name.")
    raw_count_validation = str(raw_count_validation).strip().lower()
    if raw_count_validation not in {"auto", "strict", "off"}:
        raise ValueError(
            "raw_count_validation must be one of {'auto', 'strict', 'off'}, "
            f"got {raw_count_validation!r}."
        )
    hvg_selection_transform = str(hvg_selection_transform).strip().lower()
    if hvg_selection_transform not in {"post_transform", "log1p_counts"}:
        raise ValueError(
            "hvg_selection_transform must be one of "
            "{'post_transform', 'log1p_counts'}, "
            f"got {hvg_selection_transform!r}."
        )
    normalization_reference = str(normalization_reference).strip().lower()
    if normalization_reference not in {"all_features", "latent_features"}:
        raise ValueError(
            "normalization_reference must be one of "
            "{'all_features', 'latent_features'}, "
            f"got {normalization_reference!r}."
        )
    if normalization_reference == "latent_features" and not select_hvg:
        raise ValueError(
            "normalization_reference='latent_features' requires select_hvg=True."
        )
    if latent_fit_obs_values is not None and hvg_selection_transform != "log1p_counts":
        raise ValueError(
            "latent_fit_obs_values requires hvg_selection_transform='log1p_counts' "
            "so HVGs can be ranked before observations are subset."
        )
    resolved_spatial_outlier_group_key = (
        time_key
        if spatial_outlier_group_key is None
        else str(spatial_outlier_group_key).strip()
    )
    if spatial_outlier_filter and not resolved_spatial_outlier_group_key:
        raise ValueError("spatial_outlier_group_key must not be empty.")

    # --- Time Point Mapping Logic ---
    unique_times = adata.obs[time_key].unique()

    # The processed time key is current hard-coded for convenience
    time_key_added = "time_point_processed"
    if time_mapping is not None:
        print(f"Using user-provided time mapping.")
        if not isinstance(time_mapping, Mapping):
            raise TypeError("time_mapping must be a mapping or None.")
        resolved_time_mapping = _resolved_time_mapping(unique_times, time_mapping)
        # Apply the user-defined mapping
        adata.obs[time_key_added] = (
            adata.obs[time_key].map(resolved_time_mapping).astype(float)
        )
        time_mapping_source = "user"

    else:
        print("No time mapping provided. Generating automatic mapping.")
        # Automatically map time points with robust numeric-aware ordering.
        sorted_times = _auto_time_order(unique_times)
        auto_mapping = {time_point: i for i, time_point in enumerate(sorted_times)}
        print(f"Automatically generated time mapping: {auto_mapping}")
        # Apply the automatic mapping
        adata.obs[time_key_added] = adata.obs[time_key].map(auto_mapping).astype(float)
        resolved_time_mapping = {
            key: float(value) for key, value in auto_mapping.items()
        }
        time_mapping_source = "automatic"

    print(f"Numerical time points stored in `adata.obs['{time_key_added}']`.")

    # --- Standard Preprocessing Steps ---
    input_x_state = _detect_x_state_against_counts(adata, counts_layer=counts_layer)
    preexisting_log1p_marker = "log1p" in adata.uns
    counts_layer_origin = (
        "existing"
        if counts_layer in adata.layers
        else "synthesized_from_selected_expression"
    )
    expression_source = "X"
    if expression_layer is not None:
        expression_layer = str(expression_layer).strip()
        if not expression_layer:
            raise ValueError("expression_layer must be a non-empty layer name or None.")
        if expression_layer not in adata.layers:
            raise KeyError(
                f"expression_layer '{expression_layer}' was not found in adata.layers. "
                f"Available layers are: {list(adata.layers.keys())}"
            )
        adata.X = adata.layers[expression_layer].copy()
        expression_source = f"layers['{expression_layer}']"
        # A Scanpy log1p marker describes the previous X matrix, not the layer
        # that was just promoted into X.
        adata.uns.pop("log1p", None)
        print(f"Using {expression_source} as the expression input for preprocessing.")
    elif (
        (normalization or log1p)
        and input_x_state["state"]
        in {"near_log1p_of_counts", "transformed_from_metadata"}
        and not allow_retransform_preprocessed_x
    ):
        raise ValueError(
            "adata.X appears to be already transformed while layers['counts'] is available; "
            "normalizing/log1p-transforming X again would double-transform expression. "
            f"Use expression_layer='{counts_layer}' for a clean run, disable normalization/log1p, "
            "or set allow_retransform_preprocessed_x=True only for a labelled legacy replay."
        )

    # Keep raw counts for downstream modules that need gene-space count values
    # (e.g., ligand-receptor interaction graph construction).
    if counts_layer not in adata.layers:
        adata.layers[counts_layer] = adata.X.copy()

    # The explicitly selected layer is authoritative for both preprocessing and
    # interaction-graph construction. This prevents a stale layers['counts']
    # from silently feeding the graph while another layer feeds the model.
    resolved_counts_layer = (
        expression_layer if expression_layer is not None else counts_layer
    )

    strict_raw_count_check = raw_count_validation == "strict" or (
        raw_count_validation == "auto"
        and expression_layer is not None
        and (normalization or log1p)
    )
    raw_count_stats = None
    if strict_raw_count_check:
        raw_count_stats = _validate_raw_count_like(
            adata.X,
            source=expression_source,
            integer_tolerance=raw_count_integer_tolerance,
        )

    selected_expression_stats = _matrix_value_stats(adata.X)
    if not selected_expression_stats["all_finite"]:
        raise ValueError(
            f"Expression source {expression_source} contains non-finite values."
        )
    if normalization and not selected_expression_stats["nonnegative"]:
        raise ValueError(
            f"Expression source {expression_source} contains negative values and cannot be "
            "used with normalize_total."
        )

    # Resolve the batch-aware HVG contract once so both selection modes use
    # identical validation and required-feature handling.
    hvg_mask = None
    statistical_hvg_mask = None
    hvg_fit_n_obs = None
    required_latent_features_requested = tuple(
        dict.fromkeys(str(name) for name in (required_latent_features or ()))
    )
    required_latent_features_added: list[str] = []
    resolved_hvg_batch_key = None
    if hvg_batch_key is not None:
        resolved_hvg_batch_key = str(hvg_batch_key).strip()
        if not resolved_hvg_batch_key:
            raise ValueError("hvg_batch_key must be a non-empty obs column name.")
        if resolved_hvg_batch_key not in adata.obs:
            raise KeyError(
                f"hvg_batch_key {resolved_hvg_batch_key!r} is absent from "
                f"adata.obs. Available columns: {list(adata.obs.columns)}"
            )

    def _select_hvgs(target: AnnData) -> None:
        print(f"Selecting top {n_top_genes} highly variable genes.")
        sc.pp.highly_variable_genes(
            target,
            n_top_genes=n_top_genes,
            batch_key=resolved_hvg_batch_key,
        )

    def _finalize_hvg_mask() -> None:
        nonlocal hvg_mask, statistical_hvg_mask, required_latent_features_added
        statistical_hvg_mask = adata.var["highly_variable"].copy()
        hvg_mask = statistical_hvg_mask.copy()
        if required_latent_features_requested:
            feature_index = pd.Index(adata.var_names.astype(str))
            missing_required = [
                name
                for name in required_latent_features_requested
                if name not in feature_index
            ]
            if missing_required:
                raise KeyError(
                    "required_latent_features contains names absent from adata.var_names; "
                    f"missing_count={len(missing_required)}, examples={missing_required[:10]}."
                )
            required_positions = feature_index.get_indexer(
                required_latent_features_requested
            )
            previous_mask = statistical_hvg_mask.to_numpy(dtype=bool, copy=True)
            final_mask = previous_mask.copy()
            final_mask[required_positions] = True
            adata.var["highly_variable"] = final_mask
            hvg_mask = adata.var["highly_variable"].copy()
            required_latent_features_added = [
                name
                for name, position in zip(
                    required_latent_features_requested, required_positions
                )
                if not previous_mask[int(position)]
            ]
        print(
            f"PCA feature mask: {int(np.sum(hvg_mask.values))} genes "
            f"({len(required_latent_features_added)} required features added; "
            "no subsetting of adata.X)."
        )

    if not select_hvg and required_latent_features_requested:
        raise ValueError(
            "required_latent_features is only meaningful when select_hvg=True."
        )

    # Paper-compatible clean mode: rank features on a separate log1p(counts)
    # view. The model expression matrix itself remains raw until the final HVG
    # mask and optional model-observation scope are frozen.
    if select_hvg and hvg_selection_transform == "log1p_counts":
        hvg_obs = pd.DataFrame(index=adata.obs_names.copy())
        if resolved_hvg_batch_key is not None:
            hvg_obs[resolved_hvg_batch_key] = adata.obs[
                resolved_hvg_batch_key
            ].copy()
        hvg_source = AnnData(
            X=adata.X.copy(),
            obs=hvg_obs,
            var=pd.DataFrame(index=adata.var_names.copy()),
        )
        sc.pp.log1p(hvg_source)
        _select_hvgs(hvg_source)
        for column in hvg_source.var.columns:
            adata.var[column] = hvg_source.var[column].to_numpy(copy=True)
        del hvg_source
        hvg_fit_n_obs = int(adata.n_obs)
        _finalize_hvg_mask()

    latent_fit_obs_values_resolved: list[str] = []
    latent_input_n_obs = int(adata.n_obs)
    if latent_fit_obs_values is not None:
        requested = list(dict.fromkeys(latent_fit_obs_values))
        if not requested:
            raise ValueError("latent_fit_obs_values must not be empty.")
        observed = pd.Index(pd.unique(adata.obs[time_key]))
        missing = [value for value in requested if value not in observed]
        if missing:
            raise ValueError(
                f"latent_fit_obs_values contains values absent from {time_key!r}: {missing}"
            )
        keep = adata.obs[time_key].isin(requested).to_numpy(dtype=bool)
        adata = adata[keep].copy()
        hvg_mask = adata.var["highly_variable"].copy() if select_hvg else None
        statistical_hvg_mask = (
            adata.var["highly_variable"].copy()
            if statistical_hvg_mask is None and select_hvg
            else statistical_hvg_mask
        )
        latent_fit_obs_values_resolved = [str(value) for value in requested]
        print(
            "Restricted latent normalization/PCA fit after HVG ranking: "
            f"{latent_input_n_obs} -> {adata.n_obs} observations."
        )

    spatial_outlier_filter_info: Dict[str, object] = {
        "enabled": False,
        "method": "not_applied",
        "n_input": int(adata.n_obs),
        "n_removed": 0,
        "n_retained": int(adata.n_obs),
    }
    if spatial_outlier_filter:
        before_filter = int(adata.n_obs)
        adata, spatial_outlier_filter_info = _filter_spatial_outliers(
            adata,
            spatial_key=spatial_outlier_key,
            group_key=resolved_spatial_outlier_group_key,
            robust_nn_z_threshold=spatial_outlier_nn_mad_z_threshold,
        )
        print(
            "Filtered label-blind spatial outliers before latent normalization/PCA: "
            f"{before_filter} -> {adata.n_obs} observations "
            f"(removed {spatial_outlier_filter_info['n_removed']})."
        )

    resolved_normalization_target_sum = None
    normalization_reference_feature_count = int(adata.n_vars)
    if normalization:
        if normalization_target_sum is not None:
            normalization_target_sum = float(normalization_target_sum)
            if (
                not np.isfinite(normalization_target_sum)
                or normalization_target_sum <= 0
            ):
                raise ValueError(
                    "normalization_target_sum must be a positive finite value or None, "
                    f"got {normalization_target_sum}."
                )
        reference_matrix = adata.X
        if normalization_reference == "latent_features":
            if hvg_mask is None:
                raise RuntimeError("Latent-feature normalization has no resolved HVG mask.")
            reference_mask = hvg_mask.to_numpy(dtype=bool)
            normalization_reference_feature_count = int(reference_mask.sum())
            reference_matrix = adata.X[:, reference_mask]
        if sparse.issparse(reference_matrix):
            library_sizes = np.asarray(reference_matrix.sum(axis=1)).reshape(-1)
        else:
            library_sizes = np.asarray(reference_matrix, dtype=np.float64).sum(axis=1)
        positive = np.isfinite(library_sizes) & (library_sizes > 0)
        if not np.all(positive):
            raise ValueError(
                "Cannot normalize because the selected normalization reference has "
                f"{int((~positive).sum())} non-positive or non-finite cell totals."
            )
        resolved_normalization_target_sum = (
            float(np.median(library_sizes))
            if normalization_target_sum is None
            else float(normalization_target_sum)
        )
        target_label = (
            "median library size"
            if normalization_target_sum is None
            else normalization_target_sum
        )
        print(
            f"Normalizing total counts to {target_label} using "
            f"{normalization_reference}."
        )
        if normalization_reference == "all_features":
            sc.pp.normalize_total(adata, target_sum=normalization_target_sum)
        else:
            factors = resolved_normalization_target_sum / library_sizes
            if sparse.issparse(adata.X):
                adata.X = sparse.diags(factors).dot(adata.X).tocsr()
            else:
                adata.X = np.asarray(adata.X, dtype=np.float64) * factors[:, None]

    if log1p:
        sc.pp.log1p(adata)

    # Standard mode ranks HVGs on the matrix after its main transform.
    if select_hvg and hvg_selection_transform == "post_transform":
        _select_hvgs(adata)
        hvg_fit_n_obs = int(adata.n_obs)
        _finalize_hvg_mask()

    # Persist the exact feature-wise center of the matrix used to fit PCA.
    # Spatial alignment may subsequently retain only a subset of time points;
    # recomputing the center from that subset would then make inverse PCA
    # systematically inconsistent with the fitted transform.
    if dim_reduction_name in {"pca", "umap"}:
        if sparse.issparse(adata.X):
            pca_center = np.asarray(adata.X.mean(axis=0)).reshape(-1)
        else:
            pca_center = np.asarray(adata.X, dtype=np.float64).mean(axis=0)
        if pca_center.shape[0] != adata.n_vars or not np.isfinite(pca_center).all():
            raise ValueError(
                "Could not persist a finite PCA fit center for every feature."
            )
        adata.var["pca_center"] = pca_center.astype(np.float32, copy=False)
        adata.uns["pca_center_info"] = {
            "var_key": "pca_center",
            "source": "adata.X immediately before PCA fit",
            "n_obs_fit": int(adata.n_obs),
            "n_vars_fit": int(adata.n_vars),
        }

    # --- Dimension Reduction ---
    # ---------- : PCA | UMAP | none ----------
    if dim_reduction_name == "pca":
        sc.pp.pca(
            adata,
            n_comps=n_pcs,
            svd_solver="arpack",
            use_highly_variable=bool(select_hvg),
        )
        adata.obsm["X_latent"] = np.asarray(adata.obsm["X_pca"], dtype=np.float32)

    elif dim_reduction_name == "umap":
        sc.pp.pca(
            adata,
            n_comps=n_pcs,
            svd_solver="arpack",
            use_highly_variable=bool(select_hvg),
        )
        sc.pp.neighbors(adata, n_pcs=n_pcs)
        sc.tl.umap(adata)
        adata.obsm["X_latent"] = np.asarray(adata.obsm["X_umap"], dtype=np.float32)

    elif dim_reduction_name == "none":
        # Convert to dense float array so downstream torch.tensor(...) is safe.
        if hasattr(adata.X, "toarray"):
            adata.obsm["X_latent"] = adata.X.toarray().astype(np.float32)
        else:
            adata.obsm["X_latent"] = np.asarray(adata.X, dtype=np.float32)
        print("Dimension reduction set to 'none'.")

    # Store preprocessing provenance in-place.
    preprocess_info = {
        "time_key": time_key,
        "time_key_added": time_key_added,
        "normalization": bool(normalization),
        "normalization_target_sum": (
            float(normalization_target_sum)
            if normalization_target_sum is not None
            else "median"
        ),
        "normalization_target_sum_resolved": (
            float(resolved_normalization_target_sum)
            if resolved_normalization_target_sum is not None
            else "not_applied"
        ),
        "normalization_reference": normalization_reference,
        "normalization_reference_feature_count": int(
            normalization_reference_feature_count
        ),
        "log1p": bool(log1p),
        "select_hvg": bool(select_hvg),
        "n_top_genes": int(n_top_genes),
        "dim_reduction": dim_reduction_name,
        "n_pcs": int(n_pcs),
        "hvg_for_latent_only": bool(select_hvg),
        "required_latent_features_requested": list(required_latent_features_requested),
        "required_latent_features_added": list(required_latent_features_added),
        "n_latent_fit_features": (
            int(np.sum(hvg_mask.values)) if hvg_mask is not None else int(adata.n_vars)
        ),
        "n_statistical_hvgs": (
            int(np.sum(statistical_hvg_mask.values))
            if statistical_hvg_mask is not None
            else 0
        ),
        "hvg_selection_transform": hvg_selection_transform,
        "hvg_fit_n_obs": int(hvg_fit_n_obs) if hvg_fit_n_obs is not None else 0,
        "latent_fit_input_n_obs": int(latent_input_n_obs),
        "latent_fit_n_obs": int(adata.n_obs),
        "latent_fit_obs_values": latent_fit_obs_values_resolved,
        "spatial_outlier_filter": bool(spatial_outlier_filter),
        "spatial_outlier_filter_json": json.dumps(
            spatial_outlier_filter_info, sort_keys=True
        ),
        "x_representation": "gene_expression",
        "expression_source": expression_source,
        "expression_layer": expression_layer
        if expression_layer is not None
        else "none",
        "input_x_state_detected": input_x_state,
        "counts_layer_origin": counts_layer_origin,
        "raw_counts_layer": resolved_counts_layer,
        "counts_compatibility_layer": counts_layer,
        "raw_count_validation_requested": raw_count_validation,
        "raw_count_validation_effective": "strict"
        if strict_raw_count_check
        else "basic",
        "raw_count_integer_tolerance": float(raw_count_integer_tolerance),
        "raw_count_validation_stats": raw_count_stats
        if raw_count_stats is not None
        else "not_run",
        "preexisting_log1p_marker": bool(preexisting_log1p_marker),
        "allow_retransform_preprocessed_x": bool(allow_retransform_preprocessed_x),
        "selected_expression_stats": selected_expression_stats,
        "transformation_sequence": [
            step
            for step, enabled in (("normalize_total", normalization), ("log1p", log1p))
            if enabled
        ],
        "latent_key": "X_latent",
        "counts_layer": resolved_counts_layer,
        "time_mapping_source": time_mapping_source,
        "time_mapping_json": _time_mapping_json(resolved_time_mapping),
        "observation_names": observation_name_info,
        "hvg_batch_key": (
            str(hvg_batch_key).strip() if hvg_batch_key is not None else "none"
        ),
    }
    adata.uns["preprocess_info"] = preprocess_info
    original_gene_info = {
        "var": adata.var.copy(deep=True),
        "var_names": adata.var_names.tolist(),
        "X_shape": np.array(adata.X.shape),
        "select_hvg": select_hvg,
    }
    if hvg_mask is not None:
        original_gene_info["hvg_mask"] = hvg_mask.values.astype(bool)
    if statistical_hvg_mask is not None:
        original_gene_info["statistical_hvg_mask"] = (
            statistical_hvg_mask.values.astype(bool)
        )
    adata.uns["original_gene_info"] = original_gene_info

    print(
        "Preprocess output: "
        f"X(gene) shape={adata.X.shape}, X_latent shape={adata.obsm['X_latent'].shape}"
    )

    print("Preprocessing recipe finished.")
    return adata
