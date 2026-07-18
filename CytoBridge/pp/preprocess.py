import scanpy as sc
from anndata import AnnData
import pandas as pd
import numpy as np
import logging
import json
import re
from collections.abc import Mapping
from typing import Optional, Dict
from scipy import sparse

_TIME_PATTERN = re.compile(r"-?\d+(?:\.\d+)?")


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
        if "log1p" in adata.uns or bool(adata.uns.get("preprocess_info", {}).get("log1p", False)):
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
    if support_match_fraction >= relation_threshold and raw_stats["fraction_close"] >= relation_threshold:
        result["state"] = "near_raw_counts"
    elif support_match_fraction >= relation_threshold and log_stats["fraction_close"] >= relation_threshold:
        result["state"] = "near_log1p_of_counts"
    elif "log1p" in adata.uns or bool(adata.uns.get("preprocess_info", {}).get("log1p", False)):
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
    win; a unique ``str(key)`` match is used only when no exact match exists.
    """
    resolved: Dict[object, float] = {}
    missing = []
    mapping_items = list(time_mapping.items())
    for observed in unique_times:
        if observed in time_mapping:
            target = time_mapping[observed]
        else:
            string_matches = [value for key, value in mapping_items if str(key) == str(observed)]
            if len(string_matches) == 1:
                target = string_matches[0]
            elif len(string_matches) > 1:
                raise ValueError(
                    "time_mapping contains ambiguous string-equivalent keys for "
                    f"observed time {observed!r}."
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


def preprocess(
    adata: AnnData,
    time_key: str,
    n_top_genes: int = 2000,
    dim_reduction: str = 'pca',
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

    # --- Time Point Mapping Logic ---
    unique_times = adata.obs[time_key].unique()

    # The processed time key is current hard-coded for convenience
    time_key_added = 'time_point_processed'
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
        resolved_time_mapping = {key: float(value) for key, value in auto_mapping.items()}
        time_mapping_source = "automatic"

    print(f"Numerical time points stored in `adata.obs['{time_key_added}']`.")

    # --- Standard Preprocessing Steps ---
    input_x_state = _detect_x_state_against_counts(adata, counts_layer=counts_layer)
    preexisting_log1p_marker = "log1p" in adata.uns
    counts_layer_origin = (
        "existing" if counts_layer in adata.layers else "synthesized_from_selected_expression"
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
        and input_x_state["state"] in {"near_log1p_of_counts", "transformed_from_metadata"}
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
    resolved_counts_layer = expression_layer if expression_layer is not None else counts_layer

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
        raise ValueError(f"Expression source {expression_source} contains non-finite values.")
    if normalization and not selected_expression_stats["nonnegative"]:
        raise ValueError(
            f"Expression source {expression_source} contains negative values and cannot be "
            "used with normalize_total."
        )

    if normalization:
        if normalization_target_sum is not None:
            normalization_target_sum = float(normalization_target_sum)
            if not np.isfinite(normalization_target_sum) or normalization_target_sum <= 0:
                raise ValueError(
                    "normalization_target_sum must be a positive finite value or None, "
                    f"got {normalization_target_sum}."
                )
        target_label = "median library size" if normalization_target_sum is None else normalization_target_sum
        print(f"Normalizing total counts to {target_label}.")
        sc.pp.normalize_total(adata, target_sum=normalization_target_sum)

    if log1p:
        sc.pp.log1p(adata)

    # Record the HVG mask (for later storage in original_gene_info)
    hvg_mask = None
    if select_hvg:
        print(f"Selecting top {n_top_genes} highly variable genes.")
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=n_top_genes,
        )
        hvg_mask = adata.var.highly_variable.copy()
        print(f"HVG marked: {int(np.sum(hvg_mask.values))} genes (no subsetting of adata.X).")

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
            raise ValueError("Could not persist a finite PCA fit center for every feature.")
        adata.var["pca_center"] = pca_center.astype(np.float32, copy=False)
        adata.uns["pca_center_info"] = {
            "var_key": "pca_center",
            "source": "adata.X immediately before PCA fit",
            "n_obs_fit": int(adata.n_obs),
            "n_vars_fit": int(adata.n_vars),
        }

    # --- Dimension Reduction ---
    # ---------- : PCA | UMAP | none ----------
    if dim_reduction_name == 'pca':
        sc.pp.pca(
            adata,
            n_comps=n_pcs,
            svd_solver='arpack',
            use_highly_variable=bool(select_hvg),
        )
        adata.obsm['X_latent'] = np.asarray(adata.obsm['X_pca'], dtype=np.float32)

    elif dim_reduction_name == 'umap':
        sc.pp.pca(
            adata,
            n_comps=n_pcs,
            svd_solver='arpack',
            use_highly_variable=bool(select_hvg),
        )
        sc.pp.neighbors(adata, n_pcs=n_pcs)
        sc.tl.umap(adata)
        adata.obsm['X_latent'] = np.asarray(adata.obsm['X_umap'], dtype=np.float32)

    elif dim_reduction_name == 'none':
        # Convert to dense float array so downstream torch.tensor(...) is safe.
        if hasattr(adata.X, "toarray"):
            adata.obsm['X_latent'] = adata.X.toarray().astype(np.float32)
        else:
            adata.obsm['X_latent'] = np.asarray(adata.X, dtype=np.float32)
        print("Dimension reduction set to 'none'.")

    # Store preprocessing provenance in-place.
    preprocess_info = {
        'time_key': time_key,
        'time_key_added': time_key_added,
        'normalization': bool(normalization),
        'normalization_target_sum': (
            float(normalization_target_sum) if normalization_target_sum is not None else 'median'
        ),
        'log1p': bool(log1p),
        'select_hvg': bool(select_hvg),
        'n_top_genes': int(n_top_genes),
        'dim_reduction': dim_reduction_name,
        'n_pcs': int(n_pcs),
        'hvg_for_latent_only': bool(select_hvg),
        'x_representation': 'gene_expression',
        'expression_source': expression_source,
        'expression_layer': expression_layer if expression_layer is not None else 'none',
        'input_x_state_detected': input_x_state,
        'counts_layer_origin': counts_layer_origin,
        'raw_counts_layer': resolved_counts_layer,
        'counts_compatibility_layer': counts_layer,
        'raw_count_validation_requested': raw_count_validation,
        'raw_count_validation_effective': 'strict' if strict_raw_count_check else 'basic',
        'raw_count_integer_tolerance': float(raw_count_integer_tolerance),
        'raw_count_validation_stats': raw_count_stats if raw_count_stats is not None else 'not_run',
        'preexisting_log1p_marker': bool(preexisting_log1p_marker),
        'allow_retransform_preprocessed_x': bool(allow_retransform_preprocessed_x),
        'selected_expression_stats': selected_expression_stats,
        'transformation_sequence': [
            step
            for step, enabled in (("normalize_total", normalization), ("log1p", log1p))
            if enabled
        ],
        'latent_key': 'X_latent',
        'counts_layer': resolved_counts_layer,
        'time_mapping_source': time_mapping_source,
        'time_mapping_json': _time_mapping_json(resolved_time_mapping),
    }
    adata.uns['preprocess_info'] = preprocess_info
    original_gene_info = {
        'var': adata.var.copy(deep=True),
        'var_names': adata.var_names.tolist(),
        'X_shape': np.array(adata.X.shape),
        'select_hvg': select_hvg,
    }
    if hvg_mask is not None:
        original_gene_info['hvg_mask'] = hvg_mask.values.astype(bool)
    adata.uns['original_gene_info'] = original_gene_info

    print(
        "Preprocess output: "
        f"X(gene) shape={adata.X.shape}, X_latent shape={adata.obsm['X_latent'].shape}"
    )

    print("Preprocessing recipe finished.")
    return adata
