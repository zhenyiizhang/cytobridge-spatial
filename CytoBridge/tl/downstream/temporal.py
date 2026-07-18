"""Generic temporal-profile analysis for simulated spatiotemporal states."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "PCAReconstructionSpec",
    "TemporalGenePatternResult",
    "TemporalProfileClusteringResult",
    "cluster_temporal_profiles",
    "infer_pca_center",
    "inverse_pca_states",
    "load_pca_reconstruction_spec",
    "make_pca_reconstruction_spec",
    "simplify_gene_names",
    "summarize_temporal_gene_patterns",
]


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
    mean of the requested matrix.
    """
    from scipy import sparse

    if layer is None and center_var_key in adata.var:
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
        feature_column if feature_column in center_table else str(center_table.columns[0])
    )
    if center_feature_column not in center_table:
        raise KeyError(
            f"Center table is missing feature column '{center_feature_column}'."
        )
    if center_value_column not in center_table:
        raise KeyError(
            f"Center table is missing value column '{center_value_column}'."
        )
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
    """Map joint spatial/PCA states back to the processed gene feature space."""
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


def cluster_temporal_profiles(
    profiles: pd.DataFrame,
    *,
    n_clusters: int = 2,
    normalization: str = "zscore",
    method: str = "average",
    cluster_order: str = "peak_time",
) -> TemporalProfileClusteringResult:
    """Cluster rows of a time-indexed profile matrix and summarize prototypes."""
    from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
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
        raise ValueError(
            "cluster_order must be 'peak_time', 'dendrogram', or 'raw'."
        )
    chosen_k = min(requested_k, int(table.shape[0]))
    hierarchy = None
    if chosen_k == 1:
        raw_labels = np.ones(table.shape[0], dtype=int)
        silhouette = np.nan
    else:
        hierarchy = linkage(normalized, method=method, metric="euclidean")
        raw_labels = fcluster(hierarchy, chosen_k, criterion="maxclust").astype(int)
        found = len(np.unique(raw_labels))
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
                "clusters_found": int(len(np.unique(labels))),
                "silhouette": silhouette,
                "normalization": normalization,
                "linkage_method": method,
                "cluster_order": cluster_order,
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
    preferred_species_tag: Optional[str] = None,
    pca_reconstruction: Optional[PCAReconstructionSpec] = None,
    profile_normalization: str = "zscore",
    profile_linkage_method: str = "average",
    profile_cluster_order: str = "peak_time",
) -> TemporalGenePatternResult:
    """Reconstruct mean gene profiles from simulated PCA states and cluster them."""
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
        means.append(reconstructed_mean)

    name_map = simplify_gene_names(
        (
            reference_adata.var_names
            if pca_reconstruction is None
            else pca_reconstruction.feature_names
        ),
        preferred_species_tag=preferred_species_tag,
    )
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
        "preferred_species_tag": preferred_species_tag,
        "pca_reconstruction": (
            "reference_adata"
            if pca_reconstruction is None
            else dict(pca_reconstruction.metadata)
        ),
        "profile_normalization": profile_normalization,
        "profile_linkage_method": profile_linkage_method,
        "profile_cluster_order": profile_cluster_order,
    }
    return TemporalGenePatternResult(
        expression=expression,
        top_variable_genes=top_variable,
        clustering=clustering,
        gene_name_map=name_map,
        settings=settings,
    )
