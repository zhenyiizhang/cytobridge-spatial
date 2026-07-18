"""Generic temporal-profile analysis for simulated spatiotemporal states."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "DevelopmentalWaveResult",
    "PCAAnchorReconstructionQCResult",
    "PCAReconstructionSpec",
    "TemporalGenePatternResult",
    "TemporalProfileClusteringResult",
    "analyze_developmental_wave",
    "cluster_temporal_profiles",
    "evaluate_pca_anchor_reconstruction",
    "infer_pca_center",
    "inverse_pca_states",
    "load_pca_reconstruction_spec",
    "make_pca_reconstruction_spec",
    "pca_reconstruction_feature_coverage",
    "simplify_gene_names",
    "summarize_temporal_gene_patterns",
]


@dataclass(frozen=True)
class DevelopmentalWaveResult:
    """A variance-selected, peak-ordered temporal wave and its phases.

    ``selected_profiles`` retains variance rank, whereas ``ordered_profiles``
    is gene-wise standardized and ordered by peak time.  ``assignments`` uses
    the same row order as ``ordered_profiles`` and makes the dynamic-programming
    phase boundaries explicit.
    """

    selected_profiles: pd.DataFrame
    standardized_profiles: pd.DataFrame
    ordered_profiles: pd.DataFrame
    assignments: pd.DataFrame
    prototypes: pd.DataFrame
    diagnostics: pd.DataFrame
    settings: Mapping[str, object]


@dataclass(frozen=True)
class PCAAnchorReconstructionQCResult:
    """Chunked observed-versus-inverse-PCA anchor reconstruction metrics."""

    aggregate_metrics: pd.DataFrame
    per_feature_metrics: pd.DataFrame
    settings: Mapping[str, object]


@dataclass(frozen=True)
class PCAReconstructionSpec:
    """Explicit PCA inverse-transform contract.

    The feature names, loadings, and center are kept together so a historical
    PCA transform can be reproduced without embedding dataset-specific CSV
    parsing in a downstream workflow.
    """

    feature_names: tuple[str, ...]
    loadings: np.ndarray
    center: np.ndarray
    metadata: Mapping[str, object]


@dataclass(frozen=True)
class TemporalProfileClusteringResult:
    """Normalized profiles, assignments, prototypes, and cluster diagnostics."""

    normalized_profiles: pd.DataFrame
    assignments: pd.DataFrame
    prototypes: pd.DataFrame
    diagnostics: pd.DataFrame


@dataclass(frozen=True)
class TemporalGenePatternResult:
    """Mean reconstructed expression and clustered temporal gene programs."""

    expression: pd.DataFrame
    top_variable_genes: pd.DataFrame
    clustering: TemporalProfileClusteringResult
    gene_name_map: pd.DataFrame
    settings: Mapping[str, object]


def _matrix(adata, layer: Optional[str]):
    if layer is None:
        return adata.X
    if layer not in adata.layers:
        raise KeyError(f"adata.layers is missing '{layer}'.")
    return adata.layers[layer]


def infer_pca_center(
    adata,
    *,
    layer: Optional[str] = None,
    center_var_key: str = "pca_center",
) -> np.ndarray:
    """Return the feature mean used by a zero-centered PCA fit.

    Clean CytoBridge preprocessing persists the fit-time center in
    ``adata.var[center_var_key]``.  This is important when PCA is fitted on a
    reference population and the AnnData is later subset to training time
    points.  Historical objects without that field fall back to the column
    mean of the requested matrix. ``layer`` controls only that historical
    fallback; it never overrides a persisted fit-time center.
    """
    from scipy import sparse

    # The stored center and loadings form one fit-time PCA contract. A caller
    # selecting a fallback matrix layer must not silently replace that center
    # with a mean from a different population/representation.
    if center_var_key in adata.var:
        center = np.asarray(adata.var[center_var_key], dtype=np.float64).reshape(-1)
        if center.shape[0] != adata.n_vars:
            raise ValueError(
                f"adata.var['{center_var_key}'] has {center.shape[0]} values, "
                f"expected {adata.n_vars}."
            )
        if not np.isfinite(center).all():
            raise ValueError("The persisted PCA center contains non-finite values.")
        return center.astype(np.float32, copy=False)

    matrix = _matrix(adata, layer)
    if sparse.issparse(matrix):
        center = np.asarray(matrix.mean(axis=0)).reshape(-1)
    else:
        center = np.asarray(matrix, dtype=np.float64).mean(axis=0)
    if not np.isfinite(center).all():
        raise ValueError("The inferred PCA center contains non-finite values.")
    return center.astype(np.float32, copy=False)


def make_pca_reconstruction_spec(
    feature_names: Sequence[str],
    loadings: np.ndarray,
    center: np.ndarray,
    *,
    metadata: Optional[Mapping[str, object]] = None,
) -> PCAReconstructionSpec:
    """Validate and package an explicit PCA inverse-transform contract."""
    names = tuple(map(str, feature_names))
    loading_values = np.asarray(loadings, dtype=np.float32)
    center_values = np.asarray(center, dtype=np.float32).reshape(-1)
    if loading_values.ndim != 2:
        raise ValueError("loadings must be a two-dimensional array.")
    if not names:
        raise ValueError("feature_names must be non-empty.")
    if len(names) != loading_values.shape[0]:
        raise ValueError(
            f"feature_names has {len(names)} entries, expected "
            f"{loading_values.shape[0]} from loadings."
        )
    if center_values.shape[0] != loading_values.shape[0]:
        raise ValueError(
            f"center has {center_values.shape[0]} entries, expected "
            f"{loading_values.shape[0]} from loadings."
        )
    if not np.isfinite(loading_values).all() or not np.isfinite(center_values).all():
        raise ValueError("PCA reconstruction arrays contain non-finite values.")
    return PCAReconstructionSpec(
        feature_names=names,
        loadings=loading_values,
        center=center_values,
        metadata=dict(metadata or {}),
    )


def load_pca_reconstruction_spec(
    loadings_source: str | Path | pd.DataFrame,
    center_source: str | Path | pd.DataFrame,
    *,
    feature_column: Optional[str] = None,
    center_feature_column: Optional[str] = None,
    center_value_column: str = "mean",
    component_columns: Optional[Sequence[str]] = None,
    component_prefix: str = "PC",
) -> PCAReconstructionSpec:
    """Load a portable PCA inverse-transform contract from two tables.

    The loadings table contains one feature-name column and one column per
    component. The center table contains feature names and their PCA-fit mean.
    Features are aligned by name when the two tables are not already in the
    same order.
    """
    loadings_table = (
        pd.read_csv(loadings_source)
        if not isinstance(loadings_source, pd.DataFrame)
        else loadings_source.copy()
    )
    center_table = (
        pd.read_csv(center_source)
        if not isinstance(center_source, pd.DataFrame)
        else center_source.copy()
    )
    if loadings_table.empty or center_table.empty:
        raise ValueError("PCA loadings and center tables must be non-empty.")

    feature_column = feature_column or str(loadings_table.columns[0])
    if feature_column not in loadings_table:
        raise KeyError(f"Loadings table is missing feature column '{feature_column}'.")
    center_feature_column = center_feature_column or (
        feature_column
        if feature_column in center_table
        else str(center_table.columns[0])
    )
    if center_feature_column not in center_table:
        raise KeyError(
            f"Center table is missing feature column '{center_feature_column}'."
        )
    if center_value_column not in center_table:
        raise KeyError(f"Center table is missing value column '{center_value_column}'.")
    if component_columns is None:
        component_columns = [
            str(column)
            for column in loadings_table.columns
            if str(column).startswith(str(component_prefix))
        ]
    else:
        component_columns = list(map(str, component_columns))
    if not component_columns:
        raise ValueError(
            f"No PCA component columns matched prefix '{component_prefix}'."
        )
    missing_components = [
        column for column in component_columns if column not in loadings_table
    ]
    if missing_components:
        raise KeyError(f"Loadings table is missing columns {missing_components}.")

    features = pd.Index(loadings_table[feature_column].astype(str))
    center_features = pd.Index(center_table[center_feature_column].astype(str))
    if features.equals(center_features):
        center_values = center_table[center_value_column].to_numpy(dtype=np.float32)
    else:
        if not features.is_unique or not center_features.is_unique:
            raise ValueError(
                "PCA feature rows differ in order and cannot be name-aligned because "
                "at least one table contains duplicate feature names."
            )
        center_series = pd.Series(
            center_table[center_value_column].to_numpy(dtype=np.float32),
            index=center_features,
        )
        missing = features.difference(center_series.index)
        if len(missing):
            raise ValueError(
                f"PCA center table is missing {len(missing)} features; "
                f"examples={missing[:5].tolist()}."
            )
        center_values = center_series.loc[features].to_numpy(dtype=np.float32)

    def _source_label(source) -> str:
        return "<dataframe>" if isinstance(source, pd.DataFrame) else str(Path(source))

    return make_pca_reconstruction_spec(
        features.tolist(),
        loadings_table[component_columns].to_numpy(dtype=np.float32),
        center_values,
        metadata={
            "loadings_source": _source_label(loadings_source),
            "center_source": _source_label(center_source),
            "feature_column": feature_column,
            "center_feature_column": center_feature_column,
            "center_value_column": center_value_column,
            "component_columns": list(component_columns),
        },
    )


def pca_reconstruction_feature_coverage(
    feature_names: Sequence[str],
    loadings: np.ndarray,
    *,
    absolute_tolerance: float = 1e-12,
    relative_tolerance: float = 1e-7,
) -> pd.DataFrame:
    """Identify features that can vary under an inverse-PCA reconstruction.

    A feature whose retained-component loadings are all numerically zero is
    reconstructed as its PCA center at every simulated state.  Such a feature
    is *not* evidence for a generated gene trajectory and must not silently be
    used in gene- or ligand-receptor dynamics.  This helper makes that contract
    explicit and returns one row per feature with its maximum absolute loading
    and an ``active`` flag.

    The loading threshold is ``max(absolute_tolerance,
    relative_tolerance * max(abs(loadings)))``.  The conservative defaults
    distinguish stored exact/near-zero rows from ordinary PCA loadings while
    remaining stable across float32 and float64 serialization.
    """
    names = tuple(map(str, feature_names))
    values = np.asarray(loadings)
    if values.ndim != 2:
        raise ValueError("loadings must be a two-dimensional array.")
    if len(names) != values.shape[0]:
        raise ValueError(
            f"feature_names has {len(names)} entries, expected {values.shape[0]} "
            "from loadings."
        )
    if values.shape[1] == 0:
        raise ValueError("loadings must contain at least one PCA component.")
    if not np.isfinite(values).all():
        raise ValueError("PCA loadings contain non-finite values.")
    absolute_tolerance = float(absolute_tolerance)
    relative_tolerance = float(relative_tolerance)
    if not np.isfinite(absolute_tolerance) or absolute_tolerance < 0:
        raise ValueError("absolute_tolerance must be finite and non-negative.")
    if not np.isfinite(relative_tolerance) or relative_tolerance < 0:
        raise ValueError("relative_tolerance must be finite and non-negative.")

    max_abs_loading = np.max(np.abs(values.astype(np.float64, copy=False)), axis=1)
    global_scale = float(max_abs_loading.max(initial=0.0))
    loading_tolerance = max(
        absolute_tolerance,
        relative_tolerance * global_scale,
    )
    return pd.DataFrame(
        {
            "feature_name": names,
            "max_abs_loading": max_abs_loading,
            "active": max_abs_loading > loading_tolerance,
            "loading_tolerance": np.full(len(names), loading_tolerance),
        }
    )


def inverse_pca_states(
    adata,
    states: np.ndarray,
    *,
    spatial_dim: int = 2,
    loadings_key: str = "PCs",
    center: Optional[np.ndarray] = None,
    layer: Optional[str] = None,
    clip_min: Optional[float] = None,
    reconstruction: Optional[PCAReconstructionSpec] = None,
) -> np.ndarray:
    """Map joint spatial/PCA states back to the processed gene feature space.

    The returned matrix keeps every feature for positional compatibility.
    Features with all-zero retained loadings are center-only constants; use
    :func:`pca_reconstruction_feature_coverage` before interpreting their
    reconstructed values as generated expression dynamics.
    """
    values = np.asarray(states, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("states must be a two-dimensional array.")
    if reconstruction is None:
        if loadings_key not in adata.varm:
            raise KeyError(f"adata.varm is missing PCA loadings '{loadings_key}'.")
        loadings = np.asarray(adata.varm[loadings_key], dtype=np.float32)
    else:
        if center is not None:
            raise ValueError(
                "center cannot be supplied together with an explicit reconstruction."
            )
        loadings = np.asarray(reconstruction.loadings, dtype=np.float32)
        center = reconstruction.center
    n_pcs = int(loadings.shape[1])
    if values.shape[1] < int(spatial_dim) + n_pcs:
        raise ValueError(
            f"states has {values.shape[1]} columns, but spatial_dim={spatial_dim} "
            f"and {n_pcs} PCA columns are required."
        )
    if center is None:
        center = infer_pca_center(adata, layer=layer)
    center = np.asarray(center, dtype=np.float32).reshape(-1)
    if center.shape[0] != loadings.shape[0]:
        raise ValueError(
            f"PCA center has {center.shape[0]} genes, expected {loadings.shape[0]}."
        )
    reconstructed = values[:, int(spatial_dim) : int(spatial_dim) + n_pcs] @ loadings.T
    reconstructed = reconstructed + center[None, :]
    if clip_min is not None:
        reconstructed = np.maximum(reconstructed, float(clip_min))
    return reconstructed.astype(np.float32, copy=False)


def _gene_candidates(label: str) -> list[tuple[str, Optional[str]]]:
    candidates = []
    for raw in str(label).split("|"):
        token = raw.strip()
        if not token or token.lower() == "nan" or token.upper().startswith("AMEX"):
            continue
        match = re.match(r"^(.*?)(?:\[([^\]]+)\])?$", token)
        if match is None:
            continue
        symbol = match.group(1).strip()
        species = match.group(2)
        if symbol:
            candidates.append((symbol, species.lower() if species else None))
    return candidates


def simplify_gene_names(
    var_names: Sequence[str],
    *,
    preferred_species_tag: Optional[str] = None,
) -> pd.DataFrame:
    """Create stable display/LR symbols from compound cross-species gene names."""
    preferred = preferred_species_tag.lower() if preferred_species_tag else None
    rows = []
    used: dict[str, int] = {}
    for raw in map(str, var_names):
        candidates = _gene_candidates(raw)
        selected = None
        if preferred is not None:
            selected = next(
                (symbol for symbol, species in candidates if species == preferred),
                None,
            )
        if selected is None and candidates:
            selected = candidates[0][0]
        if selected is None:
            selected = str(raw).strip()
        used[selected] = used.get(selected, 0) + 1
        unique = selected if used[selected] == 1 else f"{selected}__{used[selected]}"
        rows.append(
            {
                "var_name": raw,
                "gene": unique,
                "gene_symbol": selected,
                "duplicate_index": int(used[selected]),
            }
        )
    return pd.DataFrame(rows)


def _row_normalize(values: np.ndarray, method: str) -> np.ndarray:
    if method == "zscore":
        center = values.mean(axis=1, keepdims=True)
        scale = values.std(axis=1, keepdims=True)
        return np.divide(values - center, np.maximum(scale, 1e-12))
    if method == "minmax":
        lo = values.min(axis=1, keepdims=True)
        hi = values.max(axis=1, keepdims=True)
        return np.divide(values - lo, np.maximum(hi - lo, 1e-12))
    if method == "none":
        return values.copy()
    raise ValueError("normalization must be 'zscore', 'minmax', or 'none'.")


def _feature_name_set_sha256(feature_names: Sequence[str]) -> str:
    """Hash a feature set independently of caller ordering."""
    payload = json.dumps(
        sorted(map(str, feature_names)),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def _ordered_feature_names_sha256(feature_names: Sequence[str]) -> str:
    """Hash feature names in positional order for PCA contracts."""
    payload = json.dumps(
        list(map(str, feature_names)),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def _numeric_array_sha256(values: np.ndarray) -> str:
    """Hash numeric values in a dtype-independent canonical float64 layout."""
    array = np.ascontiguousarray(np.asarray(values, dtype="<f8"))
    header = json.dumps(
        {"dtype": "float64_le", "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    digest = sha256()
    digest.update(header)
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _canonical_json_sha256(payload: Mapping[str, object]) -> str:
    """Hash a JSON-compatible mapping with stable key ordering."""
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _json_scalar(value):
    """Convert a time label to a deterministic JSON-compatible scalar."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def evaluate_pca_anchor_reconstruction(
    adata,
    *,
    latent_key: str,
    time_key: str,
    expression_layer: Optional[str] = None,
    loadings_key: str = "PCs",
    center_var_key: str = "pca_center",
    pca_reconstruction: Optional[PCAReconstructionSpec] = None,
    chunk_size: int = 512,
    expression_space: str = "log1p",
    require_active_features: bool = True,
    pca_active_absolute_tolerance: float = 1e-12,
    pca_active_relative_tolerance: float = 1e-7,
) -> PCAAnchorReconstructionQCResult:
    """Compare observed anchors with their exact-center PCA reconstruction.

    The comparison is evaluated in bounded cell chunks and never materializes
    the full cells-by-features reconstruction.  ``adata.obsm[latent_key]`` must
    contain only the retained PCA coordinates, in the same component order as
    the supplied loadings.  The observed matrix is ``adata.X`` unless
    ``expression_layer`` is supplied.

    The caller should first subset ``adata`` to the biological observations
    and active candidate features that define the analysis (for example,
    MOSTA Brain cells and the original 2,000 HVGs).  This keeps the computation
    bounded and prevents center-only genes outside the fitted PCA feature set
    from diluting the metrics.  By default, any feature with numerically zero
    retained-component loadings is rejected; set ``require_active_features``
    to ``False`` only for an explicitly documented diagnostic.

    Feature order is a strict contract.  An explicit
    :class:`PCAReconstructionSpec` must match ``adata.var_names`` exactly, not
    merely as a set.  Without an explicit spec, loadings and the persisted
    fit-time center are read from ``adata.varm[loadings_key]`` and
    ``adata.var[center_var_key]``; a current-data mean fallback is deliberately
    disallowed for this QC.

    Aggregate R2 is the variance-weighted multi-output coefficient:
    ``1 - total SSE / sum_g SST_g``, where each feature's SST is centered by
    its observed mean within that time point.  Negative reconstructed fraction
    is measured before any clipping.  This API currently accepts only
    ``expression_space='log1p'`` so direct and reconstructed values cannot be
    silently compared on different scales.
    """
    from scipy import sparse

    if str(expression_space) != "log1p":
        raise ValueError("expression_space must be 'log1p' for anchor QC.")
    chunk_size = int(chunk_size)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if time_key not in adata.obs:
        raise KeyError(f"adata.obs is missing time key '{time_key}'.")
    if latent_key not in adata.obsm:
        raise KeyError(f"adata.obsm is missing latent key '{latent_key}'.")

    observed_feature_names = tuple(map(str, adata.var_names))
    if not observed_feature_names:
        raise ValueError("adata must contain at least one feature.")
    if not pd.Index(observed_feature_names).is_unique:
        raise ValueError("adata.var_names must be unique for strict feature matching.")

    if pca_reconstruction is None:
        if loadings_key not in adata.varm:
            raise KeyError(f"adata.varm is missing PCA loadings '{loadings_key}'.")
        if center_var_key not in adata.var:
            raise KeyError(
                f"adata.var is missing persisted PCA center '{center_var_key}'; "
                "anchor QC does not infer a center from the observed matrix."
            )
        reconstruction_feature_names = observed_feature_names
        loadings = np.asarray(adata.varm[loadings_key], dtype=np.float64)
        center = np.asarray(
            adata.var[center_var_key],
            dtype=np.float64,
        ).reshape(-1)
        reconstruction_source: object = "reference_adata"
        center_source = f"adata.var['{center_var_key}']"
    else:
        reconstruction_feature_names = tuple(map(str, pca_reconstruction.feature_names))
        loadings = np.asarray(pca_reconstruction.loadings, dtype=np.float64)
        center = np.asarray(pca_reconstruction.center, dtype=np.float64).reshape(-1)
        reconstruction_source = dict(pca_reconstruction.metadata)
        center_source = "explicit_pca_reconstruction_spec"

    if reconstruction_feature_names != observed_feature_names:
        same_set = set(reconstruction_feature_names) == set(observed_feature_names)
        detail = (
            "the same names occur in a different order"
            if same_set
            else "the feature-name sets differ"
        )
        raise ValueError(
            "PCA reconstruction feature order must exactly match adata.var_names; "
            f"{detail}."
        )
    if loadings.ndim != 2 or loadings.shape[0] != adata.n_vars:
        raise ValueError(
            "PCA loadings must have shape (adata.n_vars, n_components); "
            f"found {loadings.shape}, expected first dimension {adata.n_vars}."
        )
    if loadings.shape[1] < 1:
        raise ValueError("PCA loadings must contain at least one component.")
    if center.shape[0] != adata.n_vars:
        raise ValueError(
            f"PCA center has {center.shape[0]} values, expected {adata.n_vars}."
        )
    if not np.isfinite(loadings).all() or not np.isfinite(center).all():
        raise ValueError("PCA loadings and center must contain only finite values.")

    feature_coverage = pca_reconstruction_feature_coverage(
        observed_feature_names,
        loadings,
        absolute_tolerance=pca_active_absolute_tolerance,
        relative_tolerance=pca_active_relative_tolerance,
    )
    active_mask = feature_coverage["active"].to_numpy(dtype=bool)
    n_active_features = int(active_mask.sum())
    n_inactive_features = int(active_mask.size - n_active_features)
    loading_tolerance = float(feature_coverage["loading_tolerance"].iloc[0])
    if bool(require_active_features) and n_inactive_features:
        examples = feature_coverage.loc[~active_mask, "feature_name"].head(5).tolist()
        raise ValueError(
            "PCA anchor reconstruction QC requires active PCA features, but "
            f"{n_inactive_features} of {active_mask.size} supplied features are "
            "center-only under the retained loadings. Subset the caller-supplied "
            "AnnData view to the intended active biological candidate features "
            "(for example, Brain cells plus the original 2,000 HVGs) before "
            f"calling this API; examples={examples}."
        )

    latent = adata.obsm[latent_key]
    if getattr(latent, "ndim", None) != 2:
        raise ValueError(f"adata.obsm['{latent_key}'] must be two-dimensional.")
    if latent.shape[0] != adata.n_obs:
        raise ValueError(
            f"adata.obsm['{latent_key}'] has {latent.shape[0]} rows, expected "
            f"{adata.n_obs}."
        )
    if latent.shape[1] != loadings.shape[1]:
        raise ValueError(
            f"adata.obsm['{latent_key}'] has {latent.shape[1]} components, but "
            f"the PCA loadings have {loadings.shape[1]}; exact equality is required."
        )

    observed_matrix = _matrix(adata, expression_layer)
    if observed_matrix.shape != adata.shape:
        raise ValueError(
            "The observed expression matrix must have shape adata.shape; "
            f"found {observed_matrix.shape}, expected {adata.shape}."
        )
    raw_time_values = np.asarray(adata.obs[time_key].astype(object), dtype=object)
    if pd.isna(raw_time_values).any():
        raise ValueError(f"adata.obs['{time_key}'] contains missing values.")
    unique_time_values = list(pd.unique(raw_time_values))
    if not unique_time_values:
        raise ValueError("No anchor time points were found.")

    def _time_sort_key(value) -> tuple[object, ...]:
        converted = _json_scalar(value)
        if isinstance(converted, (int, float)) and not isinstance(converted, bool):
            numeric = float(converted)
            if np.isfinite(numeric):
                return (0, numeric, type(converted).__name__)
        return (1, type(converted).__name__, str(converted))

    unique_time_values.sort(key=_time_sort_key)
    obs_name_order_sha256 = _ordered_feature_names_sha256(
        tuple(map(str, adata.obs_names))
    )
    feature_name_set_sha256 = _feature_name_set_sha256(observed_feature_names)
    time_assignment_sha256 = sha256(
        json.dumps(
            [_json_scalar(value) for value in raw_time_values],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    feature_order_sha256 = _ordered_feature_names_sha256(observed_feature_names)
    loadings_sha256 = _numeric_array_sha256(loadings)
    center_sha256 = _numeric_array_sha256(center)
    pca_hashes = {
        "feature_name_order_sha256": feature_order_sha256,
        "loadings_sha256": loadings_sha256,
        "center_sha256": center_sha256,
    }
    pca_contract_sha256 = _canonical_json_sha256(pca_hashes)
    analysis_input_sha256 = _canonical_json_sha256(
        {
            "obs_name_order_sha256": obs_name_order_sha256,
            "feature_name_set_sha256": feature_name_set_sha256,
            "feature_name_order_sha256": feature_order_sha256,
            "time_assignment_sha256": time_assignment_sha256,
            "pca_contract_sha256": pca_contract_sha256,
        }
    )
    audit_hashes = {
        **pca_hashes,
        "obs_name_order_sha256": obs_name_order_sha256,
        "feature_name_set_sha256": feature_name_set_sha256,
        "time_assignment_sha256": time_assignment_sha256,
        "analysis_input_sha256": analysis_input_sha256,
    }

    aggregate_rows: list[dict[str, object]] = []
    per_feature_tables: list[pd.DataFrame] = []
    total_chunks = 0
    n_features = int(adata.n_vars)
    for time_value in unique_time_values:
        cell_indices = np.flatnonzero(raw_time_values == time_value)
        n_cells = int(cell_indices.size)
        if n_cells == 0:
            continue
        feature_squared_error = np.zeros(n_features, dtype=np.float64)
        feature_absolute_error = np.zeros(n_features, dtype=np.float64)
        feature_observed_mean = np.zeros(n_features, dtype=np.float64)
        feature_observed_m2 = np.zeros(n_features, dtype=np.float64)
        feature_reconstructed_mean = np.zeros(n_features, dtype=np.float64)
        feature_reconstructed_m2 = np.zeros(n_features, dtype=np.float64)
        feature_observed_reconstructed_cross_m2 = np.zeros(n_features, dtype=np.float64)
        observed_count = 0
        negative_reconstructed_count = 0
        reconstructed_min = np.inf

        for start in range(0, n_cells, chunk_size):
            chunk_indices = cell_indices[start : start + chunk_size]
            latent_chunk = (
                latent.iloc[chunk_indices]
                if isinstance(latent, pd.DataFrame)
                else latent[chunk_indices]
            )
            if sparse.issparse(latent_chunk):
                latent_chunk = latent_chunk.toarray()
            latent_chunk = np.asarray(latent_chunk, dtype=np.float64)
            if not np.isfinite(latent_chunk).all():
                raise ValueError(
                    f"Non-finite latent values found at time {_json_scalar(time_value)!r}."
                )
            reconstructed = latent_chunk @ loadings.T
            reconstructed += center[None, :]

            observed_chunk = observed_matrix[chunk_indices]
            if sparse.issparse(observed_chunk):
                observed_chunk = observed_chunk.toarray()
            observed_chunk = np.asarray(observed_chunk, dtype=np.float64)
            if not np.isfinite(observed_chunk).all():
                raise ValueError(
                    "Non-finite observed expression values found at time "
                    f"{_json_scalar(time_value)!r}."
                )
            error = reconstructed - observed_chunk
            if not np.isfinite(error).all():
                raise ValueError(
                    f"Non-finite reconstruction errors found at time "
                    f"{_json_scalar(time_value)!r}."
                )
            feature_squared_error += np.square(error).sum(axis=0)
            feature_absolute_error += np.abs(error).sum(axis=0)
            # Merge each chunk's feature moments with vectorized Welford
            # updates, avoiding the cancellation of sum(x^2)-sum(x)^2/n.
            chunk_count = int(observed_chunk.shape[0])
            chunk_observed_mean = observed_chunk.mean(axis=0)
            chunk_reconstructed_mean = reconstructed.mean(axis=0)
            chunk_observed_centered = observed_chunk - chunk_observed_mean[None, :]
            chunk_reconstructed_centered = (
                reconstructed - chunk_reconstructed_mean[None, :]
            )
            chunk_observed_m2 = np.square(chunk_observed_centered).sum(axis=0)
            chunk_reconstructed_m2 = np.square(chunk_reconstructed_centered).sum(axis=0)
            chunk_cross_m2 = (
                chunk_observed_centered * chunk_reconstructed_centered
            ).sum(axis=0)
            combined_count = observed_count + chunk_count
            observed_mean_delta = chunk_observed_mean - feature_observed_mean
            reconstructed_mean_delta = (
                chunk_reconstructed_mean - feature_reconstructed_mean
            )
            merge_weight = float(observed_count * chunk_count) / float(combined_count)
            feature_observed_m2 += (
                chunk_observed_m2 + np.square(observed_mean_delta) * merge_weight
            )
            feature_reconstructed_m2 += (
                chunk_reconstructed_m2
                + np.square(reconstructed_mean_delta) * merge_weight
            )
            feature_observed_reconstructed_cross_m2 += (
                chunk_cross_m2
                + observed_mean_delta * reconstructed_mean_delta * merge_weight
            )
            feature_observed_mean += observed_mean_delta * (
                float(chunk_count) / float(combined_count)
            )
            feature_reconstructed_mean += reconstructed_mean_delta * (
                float(chunk_count) / float(combined_count)
            )
            observed_count = combined_count
            negative_reconstructed_count += int(np.count_nonzero(reconstructed < 0.0))
            reconstructed_min = min(reconstructed_min, float(reconstructed.min()))
            total_chunks += 1

        n_values = int(n_cells * n_features)
        total_squared_error = float(feature_squared_error.sum())
        if observed_count != n_cells:
            raise RuntimeError("Chunked anchor-QC observation count is inconsistent.")
        total_sst = float(np.maximum(feature_observed_m2, 0.0).sum())
        r2 = (
            float(1.0 - total_squared_error / total_sst)
            if total_sst > np.finfo(np.float64).eps
            else np.nan
        )
        json_time = _json_scalar(time_value)
        observed_variance = np.maximum(feature_observed_m2, 0.0) / float(n_cells)
        reconstructed_variance = np.maximum(feature_reconstructed_m2, 0.0) / float(
            n_cells
        )
        observed_std = np.sqrt(observed_variance)
        reconstructed_std = np.sqrt(reconstructed_variance)
        correlation_denominator = np.sqrt(
            np.maximum(feature_observed_m2, 0.0)
            * np.maximum(feature_reconstructed_m2, 0.0)
        )
        correlation = np.divide(
            feature_observed_reconstructed_cross_m2,
            correlation_denominator,
            out=np.full(n_features, np.nan, dtype=np.float64),
            where=correlation_denominator > np.finfo(np.float64).eps,
        )
        std_ratio = np.divide(
            reconstructed_std,
            observed_std,
            out=np.full(n_features, np.nan, dtype=np.float64),
            where=observed_std > np.finfo(np.float64).eps,
        )
        aggregate_rows.append(
            {
                "time": json_time,
                "n_cells_effective": n_cells,
                "n_features_effective": n_features,
                "n_values_effective": n_values,
                "rmse": float(np.sqrt(total_squared_error / n_values)),
                "mae": float(feature_absolute_error.sum() / n_values),
                "r2": r2,
                "negative_reconstructed_fraction": float(
                    negative_reconstructed_count / n_values
                ),
                "minimum_reconstructed_log1p": reconstructed_min,
                "scale": "log1p",
                "pca_contract_sha256": pca_contract_sha256,
            }
        )
        per_feature_tables.append(
            pd.DataFrame(
                {
                    "time": np.repeat(json_time, n_features),
                    "feature_index": np.arange(n_features, dtype=int),
                    "feature": observed_feature_names,
                    "n_cells_effective": np.full(n_features, n_cells, dtype=int),
                    "rmse": np.sqrt(feature_squared_error / float(n_cells)),
                    "mae": feature_absolute_error / float(n_cells),
                    "observed_mean": feature_observed_mean,
                    "reconstructed_mean": feature_reconstructed_mean,
                    "observed_std": observed_std,
                    "reconstructed_std": reconstructed_std,
                    "bias": feature_reconstructed_mean - feature_observed_mean,
                    "correlation": correlation,
                    "std_ratio": std_ratio,
                    "scale": np.repeat("log1p", n_features),
                }
            )
        )

    aggregate_metrics = pd.DataFrame(aggregate_rows)
    per_feature_metrics = pd.concat(per_feature_tables, ignore_index=True)
    settings = {
        "algorithm": "chunked_observed_vs_exact_center_inverse_pca",
        "scale": "log1p",
        "latent_key": str(latent_key),
        "time_key": str(time_key),
        "expression_source": (
            "adata.X"
            if expression_layer is None
            else f"adata.layers['{expression_layer}']"
        ),
        "expression_layer": expression_layer,
        "loadings_key": loadings_key if pca_reconstruction is None else None,
        "center_var_key": center_var_key if pca_reconstruction is None else None,
        "center_source": center_source,
        "pca_reconstruction": reconstruction_source,
        "feature_order_validation": "exact_order_and_name_match",
        "subset_policy": "caller_supplied_adata_view",
        "subset_expectation": (
            "biological_observations_and_intended_active_candidate_features"
        ),
        "active_feature_filter_applied_by_helper": False,
        "require_active_features": bool(require_active_features),
        "pca_active_absolute_tolerance": float(pca_active_absolute_tolerance),
        "pca_active_relative_tolerance": float(pca_active_relative_tolerance),
        "pca_active_loading_tolerance": loading_tolerance,
        "n_active_features": n_active_features,
        "n_inactive_features": n_inactive_features,
        "n_cells_effective": int(adata.n_obs),
        "n_features_effective": n_features,
        "n_components": int(loadings.shape[1]),
        "n_timepoints": int(len(aggregate_metrics)),
        "time_points": aggregate_metrics["time"].tolist(),
        "chunk_size": chunk_size,
        "n_chunks_evaluated": int(total_chunks),
        "aggregate_rmse_definition": "sqrt(sum_cell_feature_squared_error/n_values)",
        "aggregate_mae_definition": "sum_cell_feature_absolute_error/n_values",
        "aggregate_r2_definition": (
            "1-total_sse/sum_feature_observed_mean_centered_sst"
        ),
        "per_feature_std_definition": "population_std_sqrt(mean_squared_deviation)",
        "per_feature_bias_definition": "reconstructed_mean-observed_mean",
        "per_feature_correlation_definition": (
            "observed_reconstructed_centered_cross_sum/"
            "sqrt(observed_centered_sse*reconstructed_centered_sse); "
            "nan_if_either_variance_is_zero"
        ),
        "per_feature_std_ratio_definition": (
            "reconstructed_population_std/observed_population_std; "
            "nan_if_observed_std_is_zero"
        ),
        "negative_reconstructed_fraction_definition": (
            "fraction_of_unclipped_inverse_pca_log1p_values_below_zero"
        ),
        "hashes": audit_hashes,
        "pca_contract_sha256": pca_contract_sha256,
        "analysis_input_sha256": analysis_input_sha256,
        "hash_contracts": {
            "feature_name_order": "sha256_ordered_json_utf8",
            "feature_name_set": "sha256_sorted_json_utf8",
            "obs_name_order": "sha256_ordered_json_utf8",
            "time_assignment": "sha256_ordered_json_utf8",
            "numeric_array": "sha256_json_shape_float64_le_null_c_order_bytes",
            "pca_contract": "sha256_canonical_json_utf8",
            "analysis_input": "sha256_canonical_json_utf8",
        },
    }
    return PCAAnchorReconstructionQCResult(
        aggregate_metrics=aggregate_metrics,
        per_feature_metrics=per_feature_metrics,
        settings=settings,
    )


def _segment_costs(values: np.ndarray) -> np.ndarray:
    """Return half-open interval SSE costs for one ordered vector."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    prefix = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
    prefix_sq = np.concatenate(([0.0], np.cumsum(np.square(values), dtype=np.float64)))
    n_values = int(values.size)
    costs = np.full((n_values + 1, n_values + 1), np.inf, dtype=np.float64)
    for start in range(n_values):
        ends = np.arange(start + 1, n_values + 1, dtype=int)
        counts = ends - start
        totals = prefix[ends] - prefix[start]
        sum_squares = prefix_sq[ends] - prefix_sq[start]
        # Roundoff can produce a tiny negative number for a constant segment.
        costs[start, ends] = np.maximum(
            sum_squares - np.square(totals) / counts,
            0.0,
        )
    return costs


def _optimal_contiguous_segments(
    ordered_values: np.ndarray,
    *,
    n_segments: int,
    min_segment_size: int,
) -> tuple[list[tuple[int, int]], float]:
    """Exactly minimize within-segment SSE with deterministic DP tie breaks."""
    values = np.asarray(ordered_values, dtype=np.float64).reshape(-1)
    n_values = int(values.size)
    n_segments = int(n_segments)
    min_segment_size = int(min_segment_size)
    if n_segments <= 0:
        raise ValueError("n_phases must be positive.")
    if min_segment_size <= 0:
        raise ValueError("min_phase_size must be positive.")
    if n_values < n_segments * min_segment_size:
        raise ValueError(
            f"Cannot divide {n_values} profiles into {n_segments} phases with "
            f"min_phase_size={min_segment_size}; at least "
            f"{n_segments * min_segment_size} profiles are required."
        )

    costs = _segment_costs(values)
    objective = np.full((n_segments + 1, n_values + 1), np.inf, dtype=np.float64)
    previous = np.full((n_segments + 1, n_values + 1), -1, dtype=int)
    objective[0, 0] = 0.0
    for segment_count in range(1, n_segments + 1):
        smallest_end = segment_count * min_segment_size
        largest_end = n_values - (n_segments - segment_count) * min_segment_size
        for end in range(smallest_end, largest_end + 1):
            smallest_start = (segment_count - 1) * min_segment_size
            largest_start = end - min_segment_size
            candidates = np.arange(smallest_start, largest_start + 1, dtype=int)
            candidate_values = (
                objective[segment_count - 1, candidates] + costs[candidates, end]
            )
            best_offset = int(np.argmin(candidate_values))
            best_start = int(candidates[best_offset])
            objective[segment_count, end] = float(candidate_values[best_offset])
            previous[segment_count, end] = best_start

    total = float(objective[n_segments, n_values])
    if not np.isfinite(total):
        raise RuntimeError("Dynamic-programming phase segmentation did not converge.")
    boundaries: list[tuple[int, int]] = []
    end = n_values
    for segment_count in range(n_segments, 0, -1):
        start = int(previous[segment_count, end])
        if start < 0:
            raise RuntimeError("Dynamic-programming phase backtracking failed.")
        boundaries.append((start, end))
        end = start
    boundaries.reverse()
    return boundaries, total


def analyze_developmental_wave(
    profiles: pd.DataFrame,
    *,
    n_top_profiles: Optional[int] = 250,
    n_phases: int = 3,
    min_phase_size: int = 5,
    standardization: str = "zscore",
) -> DevelopmentalWaveResult:
    """Select and segment a temporal developmental wave.

    Rows are arbitrary named features (usually genes), and columns are numeric
    time points.  Features are selected by population temporal variance,
    standardized per row, ordered by their earliest maximum, and divided into
    contiguous phases by exact dynamic programming.  The objective is the sum
    of within-phase squared deviations of peak times.  The implementation has
    no dataset-, species-, or marker-specific assumptions.

    Parameters
    ----------
    profiles
        Feature-by-time matrix. Feature names must be unique and time columns
        must be finite, unique numeric values.
    n_top_profiles
        Number of highest-variance rows to retain. ``None`` retains all rows.
    n_phases
        Number of contiguous phases; defaults to three.
    min_phase_size
        Hard lower bound on profiles per phase.
    standardization
        Row normalization accepted by :func:`cluster_temporal_profiles`:
        ``"zscore"`` (default), ``"minmax"``, or ``"none"``.
    """
    table = pd.DataFrame(profiles).copy()
    if table.empty or table.shape[1] < 2:
        raise ValueError("profiles must contain at least one row and two time columns.")
    profile_names = pd.Index(table.index.map(str), name="profile")
    if not profile_names.is_unique:
        raise ValueError("profiles index must contain unique feature names.")
    try:
        time_values = np.asarray(table.columns, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("profiles columns must be numeric time values.") from exc
    if not np.isfinite(time_values).all():
        raise ValueError("profiles time columns must be finite.")
    if np.unique(time_values).size != time_values.size:
        raise ValueError("profiles time columns must be unique numeric values.")
    column_order = np.argsort(time_values, kind="mergesort")
    time_values = time_values[column_order]
    values = table.to_numpy(dtype=np.float64)[:, column_order]
    if not np.isfinite(values).all():
        raise ValueError("profiles contains non-finite values.")
    table = pd.DataFrame(values, index=profile_names, columns=time_values)
    canonical_input_order = np.argsort(
        profile_names.to_numpy(dtype=str), kind="mergesort"
    )
    input_profile_names_sha256 = _feature_name_set_sha256(profile_names.tolist())
    input_profile_matrix_sha256 = _numeric_array_sha256(values[canonical_input_order])
    time_points_sha256 = _numeric_array_sha256(time_values)

    n_input = int(table.shape[0])
    if n_top_profiles is None:
        n_selected = n_input
    else:
        n_top_profiles = int(n_top_profiles)
        if n_top_profiles <= 0:
            raise ValueError("n_top_profiles must be positive or None.")
        n_selected = min(n_input, n_top_profiles)
    variances = np.var(values, axis=1, ddof=0)
    # Primary key is descending variance; names make ties input-order invariant.
    variance_order = np.lexsort((profile_names.to_numpy(dtype=str), -variances))
    selected_positions = variance_order[:n_selected]
    selected = table.iloc[selected_positions].copy()
    selected_variances = variances[selected_positions]
    standardized_values = _row_normalize(
        selected.to_numpy(dtype=np.float64),
        standardization,
    )
    standardized = pd.DataFrame(
        standardized_values,
        index=selected.index.copy(),
        columns=selected.columns.copy(),
    )
    peak_indices = np.argmax(standardized_values, axis=1).astype(int)
    peak_times = time_values[peak_indices]
    # Primary peak time, secondary descending variance, tertiary feature name.
    wave_order = np.lexsort(
        (
            selected.index.to_numpy(dtype=str),
            -selected_variances,
            peak_times,
        )
    )
    ordered = standardized.iloc[wave_order].copy()
    ordered_names = selected.index.to_numpy(dtype=str)[wave_order]
    ordered_variances = selected_variances[wave_order]
    ordered_peak_indices = peak_indices[wave_order]
    ordered_peak_times = peak_times[wave_order]

    boundaries, total_objective = _optimal_contiguous_segments(
        ordered_peak_times,
        n_segments=int(n_phases),
        min_segment_size=int(min_phase_size),
    )
    selected_profile_names = selected.index.astype(str).tolist()
    selected_profile_set_sha256 = _feature_name_set_sha256(selected_profile_names)
    selected_profile_order_sha256 = _ordered_feature_names_sha256(
        selected_profile_names
    )
    selected_profile_matrix_sha256 = _numeric_array_sha256(
        selected.to_numpy(dtype=np.float64)
    )
    phases = np.empty(n_selected, dtype=int)
    diagnostic_rows: list[dict[str, object]] = []
    prototype_rows: list[dict[str, object]] = []
    for phase, (start, end) in enumerate(boundaries, start=1):
        phases[start:end] = phase
        phase_peaks = ordered_peak_times[start:end]
        phase_profiles = ordered.to_numpy(dtype=np.float64)[start:end]
        phase_sse = float(np.square(phase_peaks - phase_peaks.mean()).sum())
        diagnostic_rows.append(
            {
                "phase": phase,
                "start_rank": int(start),
                "end_rank_exclusive": int(end),
                "n_profiles": int(end - start),
                "peak_time_min": float(phase_peaks.min()),
                "peak_time_max": float(phase_peaks.max()),
                "peak_time_mean": float(phase_peaks.mean()),
                "within_peak_sse": phase_sse,
                "total_within_peak_sse": total_objective,
                "n_input_profiles": n_input,
                "n_selected_profiles": n_selected,
                "n_phases": int(n_phases),
                "min_phase_size": int(min_phase_size),
                "standardization": standardization,
                "n_constant_profiles": int(np.count_nonzero(selected_variances <= 0.0)),
            }
        )
        for column_index, time_value in enumerate(time_values):
            prototype_rows.append(
                {
                    "phase": phase,
                    "time": float(time_value),
                    "mean": float(phase_profiles[:, column_index].mean()),
                    "std": float(phase_profiles[:, column_index].std(ddof=0)),
                    "n_profiles": int(end - start),
                }
            )

    assignments = pd.DataFrame(
        {
            "profile": ordered_names,
            "wave_rank": np.arange(n_selected, dtype=int),
            "temporal_variance": ordered_variances,
            "peak_index": ordered_peak_indices,
            "peak_time": ordered_peak_times,
            "phase": phases,
        }
    )
    return DevelopmentalWaveResult(
        selected_profiles=selected,
        standardized_profiles=standardized,
        ordered_profiles=ordered,
        assignments=assignments,
        prototypes=pd.DataFrame(prototype_rows),
        diagnostics=pd.DataFrame(diagnostic_rows),
        settings={
            "selection": "population_temporal_variance",
            "peak_tie_policy": "earliest_time",
            "wave_tie_policy": "variance_desc_then_profile_name",
            "segmentation": "contiguous_dynamic_programming_peak_time_sse",
            "n_top_profiles": None if n_top_profiles is None else int(n_top_profiles),
            "n_phases": int(n_phases),
            "min_phase_size": int(min_phase_size),
            "standardization": standardization,
            "n_input_profiles": n_input,
            "n_selected_profiles": n_selected,
            "time_points": [float(value) for value in time_values],
            "phase_boundaries": [
                {"start": int(start), "end_exclusive": int(end)}
                for start, end in boundaries
            ],
            "total_within_peak_sse": float(total_objective),
            "hashes": {
                "input_profile_name_set_sha256": input_profile_names_sha256,
                "input_profile_matrix_sha256": input_profile_matrix_sha256,
                "time_points_sha256": time_points_sha256,
                "selected_profile_name_set_sha256": selected_profile_set_sha256,
                "selected_profile_name_order_sha256": selected_profile_order_sha256,
                "selected_profile_matrix_sha256": selected_profile_matrix_sha256,
                "name_set_contract": "sha256_canonical_sorted_json_utf8",
                "name_order_contract": "sha256_ordered_json_utf8",
                "numeric_array_contract": (
                    "sha256_json_shape_float64_le_null_c_order_bytes"
                ),
            },
        },
    )


def cluster_temporal_profiles(
    profiles: pd.DataFrame,
    *,
    n_clusters: int = 2,
    normalization: str = "zscore",
    method: str = "average",
    cluster_order: str = "peak_time",
) -> TemporalProfileClusteringResult:
    """Cluster rows of a time-indexed profile matrix and summarize prototypes."""
    from scipy.cluster.hierarchy import cut_tree, leaves_list, linkage
    from sklearn.metrics import silhouette_score

    table = pd.DataFrame(profiles).copy()
    if table.empty or table.shape[1] < 2:
        raise ValueError("profiles must contain at least one row and two time columns.")
    values = table.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("profiles contains non-finite values.")
    normalized = _row_normalize(values, normalization)
    normalized_df = pd.DataFrame(normalized, index=table.index, columns=table.columns)

    requested_k = int(n_clusters)
    if requested_k <= 0:
        raise ValueError("n_clusters must be positive.")
    if cluster_order not in {"peak_time", "dendrogram", "raw"}:
        raise ValueError("cluster_order must be 'peak_time', 'dendrogram', or 'raw'.")
    chosen_k = min(requested_k, int(table.shape[0]))
    hierarchy = None
    if chosen_k == 1:
        raw_labels = np.ones(table.shape[0], dtype=int)
        silhouette = np.nan
        n_zero_distance_merges = 0
    else:
        hierarchy = linkage(normalized, method=method, metric="euclidean")
        # fcluster(..., criterion="maxclust") treats tied merge distances as one
        # threshold and can silently return fewer than requested clusters.  That
        # is common for temporal profiles containing duplicate/zero-variance
        # rows.  cut_tree performs an exact tree cut by merge count instead, so
        # every non-trivial request deterministically yields chosen_k clusters.
        raw_labels = (
            cut_tree(hierarchy, n_clusters=[chosen_k]).reshape(-1).astype(int) + 1
        )
        found = len(np.unique(raw_labels))
        if found != chosen_k:
            raise RuntimeError(
                "Exact hierarchical tree cut failed to produce the requested "
                f"number of clusters: expected {chosen_k}, found {found}."
            )
        n_zero_distance_merges = int(
            np.count_nonzero(np.isclose(hierarchy[:, 2], 0.0, rtol=0.0, atol=1e-15))
        )
        silhouette = (
            float(silhouette_score(normalized, raw_labels))
            if 1 < found < table.shape[0]
            else np.nan
        )

    if cluster_order == "raw" or chosen_k == 1:
        labels = raw_labels.astype(int, copy=True)
    elif cluster_order == "dendrogram":
        leaf_order = leaves_list(hierarchy)
        ordered_raw_labels = []
        for row_idx in leaf_order:
            raw_label = int(raw_labels[int(row_idx)])
            if raw_label not in ordered_raw_labels:
                ordered_raw_labels.append(raw_label)
        remap = {raw: idx + 1 for idx, raw in enumerate(ordered_raw_labels)}
        labels = np.asarray([remap[int(raw)] for raw in raw_labels], dtype=int)
    else:
        # Hierarchical cluster IDs are arbitrary. Reorder by prototype peak time
        # so Pattern 1 is consistently the earlier program across datasets/runs.
        order = []
        for raw_label in sorted(np.unique(raw_labels)):
            prototype = normalized[raw_labels == raw_label].mean(axis=0)
            order.append((int(np.argmax(prototype)), int(raw_label)))
        remap = {raw: idx + 1 for idx, (_, raw) in enumerate(sorted(order))}
        labels = np.asarray([remap[int(raw)] for raw in raw_labels], dtype=int)

    assignment_data = {
        "profile": table.index.astype(str),
        "cluster": labels,
    }
    if cluster_order == "dendrogram":
        leaf_order = (
            np.arange(table.shape[0], dtype=int)
            if hierarchy is None
            else leaves_list(hierarchy).astype(int)
        )
        dendrogram_rank = np.empty(table.shape[0], dtype=int)
        dendrogram_rank[leaf_order] = np.arange(table.shape[0], dtype=int)
        assignment_data["dendrogram_rank"] = dendrogram_rank
    assignments = pd.DataFrame(assignment_data)
    prototype_rows = []
    for cluster in sorted(np.unique(labels)):
        subset = normalized[labels == cluster]
        for column_idx, time_value in enumerate(table.columns):
            prototype_rows.append(
                {
                    "cluster": int(cluster),
                    "time": float(time_value),
                    "mean": float(subset[:, column_idx].mean()),
                    "std": float(subset[:, column_idx].std()),
                    "n_profiles": int(subset.shape[0]),
                }
            )
    diagnostics = pd.DataFrame(
        [
            {
                "requested_clusters": requested_k,
                "chosen_clusters": chosen_k,
                "clusters_found": int(len(np.unique(labels))),
                "silhouette": silhouette,
                "normalization": normalization,
                "linkage_method": method,
                "cluster_order": cluster_order,
                "cut_strategy": "scipy_cut_tree_exact_n_clusters",
                "n_zero_distance_merges": n_zero_distance_merges,
            }
        ]
    )
    return TemporalProfileClusteringResult(
        normalized_profiles=normalized_df,
        assignments=assignments,
        prototypes=pd.DataFrame(prototype_rows),
        diagnostics=diagnostics,
    )


def summarize_temporal_gene_patterns(
    adata_dict: Mapping[str, object],
    reference_adata,
    *,
    time_points: Optional[Sequence[float]] = None,
    spatial_dim: int = 2,
    loadings_key: str = "PCs",
    reference_layer: Optional[str] = None,
    n_top_genes: int = 250,
    n_cluster_genes: Optional[int] = None,
    n_clusters: int = 2,
    candidate_features: Optional[Sequence[str]] = None,
    preferred_species_tag: Optional[str] = None,
    pca_reconstruction: Optional[PCAReconstructionSpec] = None,
    profile_normalization: str = "zscore",
    profile_linkage_method: str = "average",
    profile_cluster_order: str = "peak_time",
    active_features_only: bool = True,
    pca_active_absolute_tolerance: float = 1e-12,
    pca_active_relative_tolerance: float = 1e-7,
) -> TemporalGenePatternResult:
    """Reconstruct mean gene profiles from simulated PCA states and cluster them.

    By default, center-only features with numerically zero retained PCA
    loadings are excluded.  They cannot vary with a simulated state and would
    otherwise create gene programs determined solely by the fit-time center.

    ``candidate_features`` optionally freezes the universe eligible for
    temporal-variance ranking. Names are matched exactly against the original
    PCA reconstruction feature names, before display-symbol simplification.
    The requested names must be unique, present, and active under the PCA
    loading tolerance; violations are hard errors. This separates a biological
    feature-selection contract (for example, a fit-time HVG set) from extra
    features retained solely to support another downstream calculation.
    """
    if time_points is None:
        time_points = sorted(float(key) for key in adata_dict)
    else:
        time_points = [float(value) for value in time_points]
    if not time_points:
        raise ValueError("time_points must be non-empty.")
    if int(n_top_genes) <= 0:
        raise ValueError("n_top_genes must be positive.")
    if n_cluster_genes is not None and int(n_cluster_genes) <= 0:
        raise ValueError("n_cluster_genes must be positive when supplied.")

    center = (
        infer_pca_center(reference_adata, layer=reference_layer)
        if pca_reconstruction is None
        else None
    )
    feature_names = tuple(
        map(
            str,
            (
                reference_adata.var_names
                if pca_reconstruction is None
                else pca_reconstruction.feature_names
            ),
        )
    )
    loadings = np.asarray(
        (
            reference_adata.varm[loadings_key]
            if pca_reconstruction is None
            else pca_reconstruction.loadings
        )
    )
    effective_center = np.asarray(
        center if pca_reconstruction is None else pca_reconstruction.center,
        dtype=np.float64,
    ).reshape(-1)
    if pca_reconstruction is not None:
        center_source = "explicit_pca_reconstruction_spec"
    elif "pca_center" in reference_adata.var:
        center_source = "reference_adata.var['pca_center']"
    elif reference_layer is None:
        center_source = "reference_adata.X_column_mean_historical_fallback"
    else:
        center_source = (
            f"reference_adata.layers['{reference_layer}']_column_mean_"
            "historical_fallback"
        )
    feature_coverage = pca_reconstruction_feature_coverage(
        feature_names,
        loadings,
        absolute_tolerance=pca_active_absolute_tolerance,
        relative_tolerance=pca_active_relative_tolerance,
    )
    active_mask = feature_coverage["active"].to_numpy(dtype=bool)
    active_feature_names = (
        feature_coverage.loc[active_mask, "feature_name"].astype(str).tolist()
    )
    pca_hashes = {
        "feature_name_order_sha256": _ordered_feature_names_sha256(feature_names),
        "loadings_sha256": _numeric_array_sha256(loadings),
        "center_sha256": _numeric_array_sha256(effective_center),
        "active_feature_name_set_sha256": _feature_name_set_sha256(
            active_feature_names
        ),
        "active_feature_name_order_sha256": _ordered_feature_names_sha256(
            active_feature_names
        ),
    }
    pca_contract_sha256 = _canonical_json_sha256(pca_hashes)
    if bool(active_features_only) and not active_mask.any():
        raise ValueError(
            "No active PCA features remain at the requested loading tolerance."
        )
    selected_feature_mask = (
        active_mask
        if bool(active_features_only)
        else np.ones(len(feature_names), dtype=bool)
    )
    candidate_requested: Optional[list[str]] = None
    candidate_missing: list[str] = []
    candidate_inactive: list[str] = []
    if candidate_features is not None:
        if isinstance(candidate_features, (str, bytes)):
            raise TypeError(
                "candidate_features must be a sequence of feature names, not a "
                "single string."
            )
        candidate_requested = list(map(str, candidate_features))
        if not candidate_requested:
            raise ValueError("candidate_features must contain at least one name.")
        candidate_index = pd.Index(candidate_requested, dtype=object)
        if not candidate_index.is_unique:
            duplicates = sorted(
                candidate_index[candidate_index.duplicated(keep=False)]
                .unique()
                .tolist()
            )
            raise ValueError(
                "candidate_features must contain unique names; "
                f"duplicates={duplicates[:5]}."
            )
        feature_index = pd.Index(feature_names, dtype=object)
        if not feature_index.is_unique:
            duplicated_reference = sorted(
                feature_index[feature_index.duplicated(keep=False)].unique().tolist()
            )
            raise ValueError(
                "PCA reconstruction feature names must be unique for exact "
                "candidate_features matching; duplicate examples="
                f"{duplicated_reference[:5]}."
            )
        feature_position = {name: index for index, name in enumerate(feature_names)}
        candidate_missing = sorted(
            name for name in candidate_requested if name not in feature_position
        )
        if candidate_missing:
            raise ValueError(
                f"candidate_features contains {len(candidate_missing)} names absent "
                "from the PCA reconstruction feature space; examples="
                f"{candidate_missing[:5]}."
            )
        candidate_inactive = sorted(
            name
            for name in candidate_requested
            if not bool(active_mask[feature_position[name]])
        )
        if candidate_inactive:
            raise ValueError(
                f"candidate_features contains {len(candidate_inactive)} inactive "
                "center-only PCA features at the requested loading tolerance; "
                f"examples={candidate_inactive[:5]}."
            )
        candidate_set = set(candidate_requested)
        candidate_mask = np.fromiter(
            (name in candidate_set for name in feature_names),
            dtype=bool,
            count=len(feature_names),
        )
        # Candidate features are always required to be active, independently
        # of the compatibility switch that can retain inactive non-candidates.
        selected_feature_mask = selected_feature_mask & candidate_mask & active_mask
    means = []
    for time_value in time_points:
        key = str(float(time_value))
        if key not in adata_dict:
            raise KeyError(f"adata_dict is missing time key '{key}'.")
        states = np.asarray(adata_dict[key].X, dtype=np.float32)
        # PCA inversion is linear, so invert only the per-time mean state. This
        # avoids materializing a cells-by-genes matrix for every dense slice.
        reconstructed_mean = inverse_pca_states(
            reference_adata,
            states.mean(axis=0, keepdims=True),
            spatial_dim=spatial_dim,
            loadings_key=loadings_key,
            center=center,
            layer=reference_layer,
            reconstruction=pca_reconstruction,
        )[0]
        means.append(reconstructed_mean[selected_feature_mask])

    selected_coverage = feature_coverage.loc[selected_feature_mask].reset_index(
        drop=True
    )
    used_candidate_features = selected_coverage["feature_name"].astype(str).tolist()
    name_map = simplify_gene_names(
        selected_coverage["feature_name"].tolist(),
        preferred_species_tag=preferred_species_tag,
    )
    name_map["pca_max_abs_loading"] = selected_coverage["max_abs_loading"].to_numpy(
        dtype=float
    )
    name_map["pca_active"] = selected_coverage["active"].to_numpy(dtype=bool)
    expression = pd.DataFrame(
        np.stack(means, axis=1),
        index=name_map["gene"].to_numpy(),
        columns=[float(value) for value in time_points],
    )
    variance = expression.var(axis=1, ddof=0).sort_values(ascending=False)
    top_n = min(int(n_top_genes), int(expression.shape[0]))
    cluster_n = min(
        int(n_cluster_genes) if n_cluster_genes is not None else top_n,
        int(expression.shape[0]),
    )
    selected = variance.head(top_n).index
    selected_for_clustering = variance.head(cluster_n).index
    top_variable = pd.DataFrame(
        {
            "gene": selected,
            "variance": variance.loc[selected].to_numpy(dtype=float),
        }
    )
    clustering = cluster_temporal_profiles(
        expression.loc[selected_for_clustering],
        n_clusters=int(n_clusters),
        normalization=profile_normalization,
        method=profile_linkage_method,
        cluster_order=profile_cluster_order,
    )
    top_variable = top_variable.merge(
        clustering.assignments.rename(columns={"profile": "gene"}),
        on="gene",
        how="left",
        validate="1:1",
    )
    settings = {
        "time_points": [float(value) for value in time_points],
        "spatial_dim": int(spatial_dim),
        "loadings_key": loadings_key,
        "reference_layer": reference_layer,
        "n_top_genes": int(top_n),
        "n_cluster_genes": int(cluster_n),
        "n_clusters": int(n_clusters),
        "candidate_features": {
            "match_space": "exact_pca_reconstruction_feature_name",
            "policy": "strict" if candidate_requested is not None else "not_requested",
            "requested": candidate_requested,
            "used": used_candidate_features,
            "missing": candidate_missing,
            "inactive": candidate_inactive,
            "requested_count": (
                None if candidate_requested is None else len(candidate_requested)
            ),
            "used_count": len(used_candidate_features),
            "missing_count": len(candidate_missing),
            "inactive_count": len(candidate_inactive),
            "requested_sha256": (
                None
                if candidate_requested is None
                else _feature_name_set_sha256(candidate_requested)
            ),
            "used_sha256": _feature_name_set_sha256(used_candidate_features),
            "used_ordered_sha256": _ordered_feature_names_sha256(
                used_candidate_features
            ),
            "hash_contract": "sha256_canonical_sorted_json_utf8",
            "set_hash_contract": "sha256_canonical_sorted_json_utf8",
            "order_hash_contract": "sha256_ordered_json_utf8",
        },
        "preferred_species_tag": preferred_species_tag,
        "pca_reconstruction": (
            "reference_adata"
            if pca_reconstruction is None
            else dict(pca_reconstruction.metadata)
        ),
        "profile_normalization": profile_normalization,
        "profile_linkage_method": profile_linkage_method,
        "profile_cluster_order": profile_cluster_order,
        "active_features_only": bool(active_features_only),
        "pca_features_total": int(len(feature_coverage)),
        "pca_features_active": int(active_mask.sum()),
        "pca_features_inactive": int((~active_mask).sum()),
        "pca_active_loading_tolerance": float(
            feature_coverage["loading_tolerance"].iloc[0]
        ),
        "pca_active_absolute_tolerance": float(pca_active_absolute_tolerance),
        "pca_active_relative_tolerance": float(pca_active_relative_tolerance),
        "pca_contract": {
            "center_source": center_source,
            "feature_count": int(len(feature_names)),
            "component_count": int(loadings.shape[1]),
            "active_feature_count": int(active_mask.sum()),
            "hashes": pca_hashes,
            "contract_sha256": pca_contract_sha256,
            "contract_hash_contract": "sha256_canonical_json_utf8",
            "name_set_hash_contract": "sha256_canonical_sorted_json_utf8",
            "name_order_hash_contract": "sha256_ordered_json_utf8",
            "numeric_array_hash_contract": (
                "sha256_json_shape_float64_le_null_c_order_bytes"
            ),
        },
        "expression_contract": {
            "source_policy": "inverse_pca_all_timepoints",
            "output_space": "mean_log1p_expression",
            "aggregation_identity": (
                "inverse_pca(mean_latent_pca) == mean(inverse_pca(latent_pca))"
            ),
            "count_space_conversion": "not_applied",
        },
    }
    return TemporalGenePatternResult(
        expression=expression,
        top_variable_genes=top_variable,
        clustering=clustering,
        gene_name_map=name_map,
        settings=settings,
    )
