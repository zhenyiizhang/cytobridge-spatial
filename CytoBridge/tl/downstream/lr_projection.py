"""Expression-aware ligand-receptor projection of communication matrices."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .downstream_data import infer_time_key, parse_time_value
from .temporal import (
    PCAReconstructionSpec,
    TemporalProfileClusteringResult,
    infer_pca_center,
    pca_reconstruction_feature_coverage,
    simplify_gene_names,
    cluster_temporal_profiles,
)

__all__ = [
    "LRTemporalProjectionResult",
    "load_ligand_receptor_database",
    "project_communication_to_lr_timecourses",
]


@dataclass(frozen=True)
class LRTemporalProjectionResult:
    """LR-pair time courses, cell-type contributions, patterns, and coverage."""

    pair_timecourse: pd.DataFrame
    celltype_timecourse: pd.DataFrame
    pattern_summary: pd.DataFrame
    clustering: TemporalProfileClusteringResult
    coverage: pd.DataFrame
    settings: Mapping[str, object]


def load_ligand_receptor_database(
    source: str | Path | pd.DataFrame,
) -> pd.DataFrame:
    """Read a ligand-receptor table while tolerating common column conventions."""
    table = (
        pd.read_csv(source) if not isinstance(source, pd.DataFrame) else source.copy()
    )
    if table.empty:
        raise ValueError("Ligand-receptor database is empty.")
    columns = list(map(str, table.columns))
    lower = {column.lower(): column for column in columns}
    ligand_candidates = ("ligand", "ligand_symbol", "source", "gene_a", "0")
    receptor_candidates = ("receptor", "receptor_symbol", "target", "gene_b", "1")
    ligand_col = next(
        (lower[name] for name in ligand_candidates if name in lower), None
    )
    receptor_col = next(
        (lower[name] for name in receptor_candidates if name in lower), None
    )
    if ligand_col is None or receptor_col is None:
        usable = [
            column for column in columns if not column.lower().startswith("unnamed")
        ]
        if len(usable) >= 2:
            ligand_col, receptor_col = usable[:2]
    if ligand_col is None or receptor_col is None:
        raise ValueError(f"Could not identify ligand/receptor columns from {columns}.")
    result = table[[ligand_col, receptor_col]].rename(
        columns={ligand_col: "ligand", receptor_col: "receptor"}
    )
    result["ligand"] = result["ligand"].astype(str).str.strip()
    result["receptor"] = result["receptor"].astype(str).str.strip()
    result = result.loc[
        result["ligand"].ne("")
        & result["receptor"].ne("")
        & result["ligand"].str.lower().ne("nan")
        & result["receptor"].str.lower().ne("nan")
    ]
    return result.drop_duplicates().reset_index(drop=True)


def _complex_tokens(value: str) -> list[str]:
    return [token.strip() for token in str(value).split("_") if token.strip()]


def _combine_complex(
    token: str,
    symbol_to_vector: Mapping[str, np.ndarray],
    *,
    mode: str,
    require_all_subunits: bool,
) -> tuple[Optional[np.ndarray], list[str]]:
    subunits = _complex_tokens(token)
    missing = [name for name in subunits if name not in symbol_to_vector]
    if require_all_subunits and missing:
        return None, missing
    vectors = [symbol_to_vector[name] for name in subunits if name in symbol_to_vector]
    if not vectors:
        return None, missing
    values = np.stack(vectors, axis=0)
    if mode == "min":
        return values.min(axis=0), missing
    if mode == "mean":
        return values.mean(axis=0), missing
    if mode == "product":
        return values.prod(axis=0), missing
    raise ValueError("complex_mode must be 'min', 'mean', or 'product'.")


def _communication_record(
    communications: Mapping[str, Mapping[str, object]],
    time_value: float,
) -> Mapping[str, object]:
    candidates = (str(float(time_value)), str(time_value))
    for key in candidates:
        if key in communications:
            return communications[key]
    raise KeyError(
        f"communications is missing time {time_value}; available={list(communications)}"
    )


def _mean_states_by_type(
    states: np.ndarray,
    labels: np.ndarray,
    cell_types: Sequence[str],
) -> tuple[np.ndarray, dict[str, int]]:
    means = np.zeros((len(cell_types), states.shape[1]), dtype=np.float32)
    counts = {}
    for idx, cell_type in enumerate(cell_types):
        mask = labels == str(cell_type)
        counts[str(cell_type)] = int(mask.sum())
        if mask.any():
            means[idx] = states[mask].mean(axis=0)
    return means, counts


def _expression_matrix(adata, layer: Optional[str]):
    if layer is None:
        return adata.X
    if layer not in adata.layers:
        raise KeyError(f"observed_adata.layers is missing '{layer}'.")
    return adata.layers[layer]


def _align_expression_features(
    adata,
    matrix,
    feature_names: Sequence[str],
):
    """Align an observed expression matrix to the PCA feature contract."""
    observed_names = pd.Index(map(str, adata.var_names))
    target_names = pd.Index(map(str, feature_names))
    if observed_names.equals(target_names):
        return matrix, "exact"
    if not observed_names.is_unique:
        raise ValueError(
            "observed_adata.var_names must be unique when their order differs "
            "from the PCA reconstruction feature order."
        )
    if not target_names.is_unique:
        raise ValueError(
            "PCA reconstruction feature names must be unique when observed "
            "genes require name-based reordering."
        )
    missing = target_names.difference(observed_names)
    if len(missing):
        raise ValueError(
            f"observed_adata is missing {len(missing)} PCA reconstruction "
            f"features; examples={missing[:5].tolist()}."
        )
    indexer = observed_names.get_indexer(target_names)
    return matrix[:, indexer], "var_name"


def _mean_expression_by_type(
    expression,
    labels: np.ndarray,
    cell_types: Sequence[str],
    *,
    source_space: str,
    target_space: str,
) -> tuple[np.ndarray, dict[str, int]]:
    from scipy import sparse

    means = np.zeros((len(cell_types), expression.shape[1]), dtype=np.float32)
    counts = {}
    for idx, cell_type in enumerate(cell_types):
        mask = labels == str(cell_type)
        counts[str(cell_type)] = int(mask.sum())
        if not mask.any():
            continue
        subset = expression[mask]
        subset = _convert_expression_matrix_space(
            subset,
            source=source_space,
            target=target_space,
        )
        if sparse.issparse(subset):
            mean = np.asarray(subset.mean(axis=0)).reshape(-1)
        else:
            mean = np.asarray(subset, dtype=np.float64).mean(axis=0)
        means[idx] = mean
    if not np.isfinite(means).all():
        raise ValueError("Observed mean expression contains non-finite values.")
    return means, counts


def _convert_expression_matrix_space(expression, *, source: str, target: str):
    """Convert each cell before aggregation, preserving sparse zero entries."""
    from scipy import sparse

    if source not in {"log1p", "count"}:
        raise ValueError("Expression source space must be 'log1p' or 'count'.")
    if target not in {"log1p", "count"}:
        raise ValueError("Expression target space must be 'log1p' or 'count'.")
    if sparse.issparse(expression):
        values = expression.astype(np.float64, copy=True)
        if source == "count":
            values.data = np.clip(values.data, 0.0, None)
            if target == "log1p":
                values.data = np.log1p(values.data)
        elif target == "count":
            values.data = np.clip(np.expm1(values.data), 0.0, None)
        values.eliminate_zeros()
        if not np.isfinite(values.data).all():
            raise ValueError("Expression-space conversion produced non-finite values.")
        return values
    return _convert_expression_space(
        np.asarray(expression),
        source=source,
        target=target,
    )


def _convert_expression_space(
    expression: np.ndarray,
    *,
    source: str,
    target: str,
) -> np.ndarray:
    if source not in {"log1p", "count"}:
        raise ValueError("Expression source space must be 'log1p' or 'count'.")
    if target not in {"log1p", "count"}:
        raise ValueError("Expression target space must be 'log1p' or 'count'.")
    values = np.asarray(expression, dtype=np.float64)
    if source == "count":
        values = np.clip(values, 0.0, None)
        if target == "log1p":
            values = np.log1p(values)
    elif target == "count":
        values = np.clip(np.expm1(values), 0.0, None)
    if not np.isfinite(values).all():
        raise ValueError("Expression-space conversion produced non-finite values.")
    return values


def _mean_inverse_pca_expression_by_type(
    states: np.ndarray,
    labels: np.ndarray,
    cell_types: Sequence[str],
    *,
    spatial_dim: int,
    loadings: np.ndarray,
    center: np.ndarray,
    target_space: str,
    batch_size: int = 4096,
) -> tuple[np.ndarray, dict[str, int]]:
    """Inverse PCA and convert each generated cell before cell-type averaging."""
    values = np.asarray(states, dtype=np.float32)
    loading_values = np.asarray(loadings, dtype=np.float32)
    center_values = np.asarray(center, dtype=np.float32).reshape(-1)
    n_pcs = int(loading_values.shape[1])
    if values.ndim != 2:
        raise ValueError("states must be a two-dimensional array.")
    if values.shape[1] < int(spatial_dim) + n_pcs:
        raise ValueError(
            f"states has {values.shape[1]} columns, but spatial_dim={spatial_dim} "
            f"and {n_pcs} PCA columns are required."
        )
    if center_values.shape[0] != loading_values.shape[0]:
        raise ValueError(
            "PCA center and active loadings have different feature counts."
        )
    if target_space not in {"log1p", "count"}:
        raise ValueError("target_space must be 'log1p' or 'count'.")

    pcs = values[:, int(spatial_dim) : int(spatial_dim) + n_pcs]
    means = np.zeros((len(cell_types), loading_values.shape[0]), dtype=np.float32)
    counts: dict[str, int] = {}
    for type_idx, cell_type in enumerate(cell_types):
        matching = np.flatnonzero(labels == str(cell_type))
        counts[str(cell_type)] = int(matching.size)
        if matching.size == 0:
            continue
        accumulator = np.zeros(loading_values.shape[0], dtype=np.float64)
        for start in range(0, int(matching.size), int(batch_size)):
            indices = matching[start : start + int(batch_size)]
            reconstructed = pcs[indices] @ loading_values.T
            reconstructed += center_values[None, :]
            if target_space == "count":
                reconstructed = np.clip(np.expm1(reconstructed), 0.0, None)
            if not np.isfinite(reconstructed).all():
                raise ValueError(
                    "Inverse-PCA expression conversion produced non-finite values."
                )
            accumulator += reconstructed.sum(axis=0)
        means[type_idx] = (accumulator / matching.size).astype(np.float32)
    return means, counts


def _match_time(
    time_value: float,
    candidates: Sequence[float],
    *,
    atol: float,
) -> Optional[float]:
    matches = [
        float(candidate)
        for candidate in candidates
        if np.isclose(float(time_value), float(candidate), rtol=0.0, atol=atol)
    ]
    if len(matches) > 1:
        raise ValueError(
            f"Time {time_value} ambiguously matches observed times {matches} "
            f"within atol={atol}."
        )
    return matches[0] if matches else None


def _symbol_vectors(
    expression_by_type: np.ndarray,
    gene_name_map: pd.DataFrame,
) -> dict[str, np.ndarray]:
    vectors = {}
    for symbol, rows in gene_name_map.groupby("gene_symbol", sort=False):
        indices = rows.index.to_numpy(dtype=int)
        vectors[str(symbol)] = expression_by_type[:, indices].mean(axis=1)
    return vectors


def _profile_summary(
    pair_timecourse: pd.DataFrame,
    clustering: TemporalProfileClusteringResult,
) -> pd.DataFrame:
    assignments = clustering.assignments.rename(columns={"profile": "pair"})
    rows = []
    for pair, subset in pair_timecourse.groupby("pair", sort=True):
        subset = subset.sort_values("time")
        times = subset["time"].to_numpy(dtype=float)
        scores = subset["score"].to_numpy(dtype=float)
        peak = int(np.argmax(scores))
        rows.append(
            {
                "pair": str(pair),
                "auc": float(np.trapz(scores, times)),
                "peak_time": float(times[peak]),
                "peak_score": float(scores[peak]),
                "start_score": float(scores[0]),
                "end_score": float(scores[-1]),
                "delta_end_start": float(scores[-1] - scores[0]),
                "score_range": float(scores.max() - scores.min()),
            }
        )
    return pd.DataFrame(rows).merge(
        assignments,
        on="pair",
        how="left",
        validate="1:1",
    )


def project_communication_to_lr_timecourses(
    adata_dict: Mapping[str, object],
    reference_adata,
    communications: Mapping[str, Mapping[str, object]],
    lr_database: str | Path | pd.DataFrame,
    *,
    time_points: Optional[Sequence[float]] = None,
    annotation_key: str = "Annotation",
    matrix_key: str = "M_per_source",
    spatial_dim: int = 2,
    loadings_key: str = "PCs",
    reference_layer: Optional[str] = None,
    expression_space: str = "count",
    complex_mode: str = "min",
    require_all_subunits: bool = False,
    duplicate_policy: str = "first",
    preferred_species_tag: Optional[str] = None,
    n_clusters: int = 2,
    pca_reconstruction: Optional[PCAReconstructionSpec] = None,
    profile_linkage_method: str = "average",
    profile_cluster_order: str = "peak_time",
    observed_adata=None,
    observed_time_key: Optional[str] = None,
    observed_time_points: Optional[Sequence[float]] = None,
    observed_annotation_key: Optional[str] = None,
    observed_layer: Optional[str] = None,
    observed_expression_space: str = "log1p",
    observed_missing_time_policy: str = "error",
    observed_time_atol: float = 1e-8,
    pca_active_absolute_tolerance: float = 1e-12,
    pca_active_relative_tolerance: float = 1e-7,
) -> LRTemporalProjectionResult:
    """Project cell-type communication into LR-pair trajectories.

    For sender type ``A`` and receiver type ``B``, the score is
    ``mean_A(ligand) * mean_B(receptor) * communication(A, B)``. Simulated
    PCA states are inverted with loadings and the center recovered from the
    reference AnnData, making the computation reusable across datasets.

    ``observed_adata`` enables a hybrid expression contract used by the
    historical zebrafish analysis: requested times listed in
    ``observed_time_points`` use real expression from the matching rows of
    that AnnData; all other times use inverse-PCA expression from
    ``adata_dict``. If ``observed_time_points`` is omitted, all times present
    in ``observed_adata.obs[observed_time_key]`` are considered observed.
    Observed cells carry their own annotation labels, so no positional row
    alignment with trajectory slices is assumed. Genes are aligned exactly by
    ``var_names`` to the PCA reconstruction feature order. The default
    ``observed_adata=None`` preserves the original all-inverse-PCA behavior.
    """
    if expression_space not in {"log1p", "count"}:
        raise ValueError("expression_space must be 'log1p' or 'count'.")
    if duplicate_policy not in {"first", "last", "sum", "max"}:
        raise ValueError("duplicate_policy must be first, last, sum, or max.")
    if observed_expression_space not in {"log1p", "count"}:
        raise ValueError("observed_expression_space must be 'log1p' or 'count'.")
    if observed_missing_time_policy not in {"error", "generated"}:
        raise ValueError("observed_missing_time_policy must be 'error' or 'generated'.")
    observed_time_atol = float(observed_time_atol)
    if not np.isfinite(observed_time_atol) or observed_time_atol < 0:
        raise ValueError("observed_time_atol must be finite and non-negative.")
    if time_points is None:
        time_points = sorted(float(key) for key in adata_dict)
    else:
        time_points = [float(value) for value in time_points]
    if not time_points:
        raise ValueError("time_points must be non-empty.")

    lr_table = load_ligand_receptor_database(lr_database)
    center = (
        infer_pca_center(reference_adata, layer=reference_layer)
        if pca_reconstruction is None
        else np.asarray(pca_reconstruction.center, dtype=np.float32)
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
    full_gene_name_map = simplify_gene_names(
        feature_names,
        preferred_species_tag=preferred_species_tag,
    )
    loadings = np.asarray(
        (
            reference_adata.varm[loadings_key]
            if pca_reconstruction is None
            else pca_reconstruction.loadings
        ),
        dtype=np.float32,
    )
    feature_coverage = pca_reconstruction_feature_coverage(
        feature_names,
        loadings,
        absolute_tolerance=pca_active_absolute_tolerance,
        relative_tolerance=pca_active_relative_tolerance,
    )
    active_mask = feature_coverage["active"].to_numpy(dtype=bool)
    if not active_mask.any():
        raise ValueError(
            "No active PCA features remain at the requested loading tolerance; "
            "inverse-generated LR expression cannot be scored."
        )
    all_feature_symbols = set(full_gene_name_map["gene_symbol"].astype(str))
    active_feature_symbols = set(
        full_gene_name_map.loc[active_mask, "gene_symbol"].astype(str)
    )
    requested_lr_symbols = {
        token
        for row in lr_table.itertuples(index=False)
        for token in _complex_tokens(str(row.ligand))
        + _complex_tokens(str(row.receptor))
    }
    lr_feature_mask = active_mask & full_gene_name_map["gene_symbol"].isin(
        requested_lr_symbols
    ).to_numpy(dtype=bool)
    active_indices = np.flatnonzero(lr_feature_mask)
    if not len(active_indices):
        raise ValueError("No active PCA features overlap the ligand-receptor database.")
    active_gene_name_map = full_gene_name_map.loc[lr_feature_mask].reset_index(
        drop=True
    )
    observed_lr_feature_mask = (
        full_gene_name_map["gene_symbol"]
        .isin(requested_lr_symbols)
        .to_numpy(dtype=bool)
    )
    observed_lr_indices = np.flatnonzero(observed_lr_feature_mask)
    observed_gene_name_map = full_gene_name_map.loc[
        observed_lr_feature_mask
    ].reset_index(drop=True)

    resolved_observed_time_key = None
    resolved_observed_annotation_key = observed_annotation_key or annotation_key
    observed_time_values = None
    expected_observed_times: list[float] = []
    observed_expression = None
    observed_gene_alignment = None
    if observed_adata is not None:
        resolved_observed_time_key = infer_time_key(
            observed_adata.obs,
            preferred=observed_time_key,
        )
        if resolved_observed_annotation_key not in observed_adata.obs:
            raise KeyError(
                "observed_adata.obs is missing annotation key "
                f"'{resolved_observed_annotation_key}'."
            )
        if observed_adata.obs[resolved_observed_annotation_key].isna().any():
            raise ValueError("observed_adata contains missing annotation labels.")
        observed_time_values = np.asarray(
            [
                parse_time_value(value)
                for value in observed_adata.obs[resolved_observed_time_key].to_numpy()
            ],
            dtype=np.float64,
        )
        if not np.isfinite(observed_time_values).all():
            raise ValueError("observed_adata contains non-finite time values.")
        expected_observed_times = (
            sorted(np.unique(observed_time_values).astype(float).tolist())
            if observed_time_points is None
            else [parse_time_value(value) for value in observed_time_points]
        )
        for idx, first in enumerate(expected_observed_times):
            for second in expected_observed_times[idx + 1 :]:
                if np.isclose(first, second, rtol=0.0, atol=observed_time_atol):
                    raise ValueError(
                        "observed_time_points contains duplicate/ambiguous values "
                        f"{first} and {second} within atol={observed_time_atol}."
                    )
        observed_expression, observed_gene_alignment = _align_expression_features(
            observed_adata,
            _expression_matrix(observed_adata, observed_layer),
            feature_names,
        )
        observed_expression = observed_expression[:, observed_lr_indices]
    elif observed_time_points is not None:
        raise ValueError("observed_time_points requires observed_adata.")

    pair_rows = []
    celltype_rows = []
    coverage_rows = []

    for time_value in time_points:
        key = str(float(time_value))
        if key not in adata_dict:
            raise KeyError(f"adata_dict is missing time key '{key}'.")
        adata_t = adata_dict[key]
        if annotation_key not in adata_t.obs:
            raise KeyError(f"adata_dict['{key}'].obs is missing '{annotation_key}'.")
        record = _communication_record(communications, time_value)
        if "types" not in record or matrix_key not in record:
            raise KeyError(
                f"Communication record {key} must contain 'types' and '{matrix_key}'."
            )
        cell_types = [str(value) for value in np.asarray(record["types"]).tolist()]
        communication = np.asarray(record[matrix_key], dtype=np.float64)
        if communication.shape != (len(cell_types), len(cell_types)):
            raise ValueError(
                f"Communication matrix at {key} has shape {communication.shape}, "
                f"expected {(len(cell_types), len(cell_types))}."
            )
        states = np.asarray(adata_t.X, dtype=np.float32)
        matched_observed_time = _match_time(
            time_value,
            expected_observed_times,
            atol=observed_time_atol,
        )
        expression_source = "inverse_pca"
        observed_missing_fallback = False
        n_expression_cells = int(states.shape[0])
        if matched_observed_time is not None:
            observed_mask = np.isclose(
                observed_time_values,
                matched_observed_time,
                rtol=0.0,
                atol=observed_time_atol,
            )
            if not observed_mask.any():
                if observed_missing_time_policy == "error":
                    raise ValueError(
                        f"No observed_adata rows matched expected observed time "
                        f"{matched_observed_time} within atol={observed_time_atol}."
                    )
                observed_missing_fallback = True
            else:
                observed_labels = (
                    observed_adata.obs.loc[
                        observed_mask, resolved_observed_annotation_key
                    ]
                    .astype(str)
                    .to_numpy()
                )
                expression, counts = _mean_expression_by_type(
                    observed_expression[observed_mask],
                    observed_labels,
                    cell_types,
                    source_space=observed_expression_space,
                    target_space=expression_space,
                )
                expression_source = "observed"
                n_expression_cells = int(observed_mask.sum())

        if expression_source == "inverse_pca":
            labels = adata_t.obs[annotation_key].astype(str).to_numpy()
            expression, counts = _mean_inverse_pca_expression_by_type(
                states,
                labels,
                cell_types,
                spatial_dim=spatial_dim,
                loadings=loadings[active_indices],
                center=np.asarray(center, dtype=np.float32)[active_indices],
                target_space=expression_space,
            )
            current_gene_name_map = active_gene_name_map
            available_symbols = active_feature_symbols
        else:
            current_gene_name_map = observed_gene_name_map
            available_symbols = set(observed_gene_name_map["gene_symbol"].astype(str))
        symbol_to_vector = _symbol_vectors(expression, current_gene_name_map)

        scored: dict[str, np.ndarray] = {}
        skipped_missing = 0
        partial_complexes = 0
        duplicates = 0
        pairs_with_missing_subunit = 0
        pairs_with_unreconstructable_subunit = 0
        missing_subunits_seen: set[str] = set()
        unreconstructable_subunits_seen: set[str] = set()
        for row in lr_table.itertuples(index=False):
            ligand = str(row.ligand)
            receptor = str(row.receptor)
            requested_subunits = _complex_tokens(ligand) + _complex_tokens(receptor)
            missing_subunits = {
                name for name in requested_subunits if name not in all_feature_symbols
            }
            unreconstructable_subunits = {
                name
                for name in requested_subunits
                if name in all_feature_symbols and name not in available_symbols
            }
            if missing_subunits:
                pairs_with_missing_subunit += 1
                missing_subunits_seen.update(missing_subunits)
            if unreconstructable_subunits:
                pairs_with_unreconstructable_subunit += 1
                unreconstructable_subunits_seen.update(unreconstructable_subunits)
            ligand_values, ligand_missing = _combine_complex(
                ligand,
                symbol_to_vector,
                mode=complex_mode,
                require_all_subunits=require_all_subunits,
            )
            receptor_values, receptor_missing = _combine_complex(
                receptor,
                symbol_to_vector,
                mode=complex_mode,
                require_all_subunits=require_all_subunits,
            )
            if ligand_values is None or receptor_values is None:
                skipped_missing += 1
                continue
            if ligand_missing or receptor_missing:
                partial_complexes += 1
            pair = f"{ligand}_{receptor}"
            score = np.outer(ligand_values, receptor_values) * communication
            if pair in scored:
                duplicates += 1
                if duplicate_policy == "first":
                    continue
                if duplicate_policy == "last":
                    scored[pair] = score
                elif duplicate_policy == "sum":
                    scored[pair] = scored[pair] + score
                else:
                    scored[pair] = np.maximum(scored[pair], score)
            else:
                scored[pair] = score

        for pair, matrix in scored.items():
            incoming = matrix.sum(axis=0)
            outgoing = matrix.sum(axis=1)
            pair_rows.append(
                {
                    "time": float(time_value),
                    "pair": pair,
                    "score": float(matrix.sum()),
                    "max_edge": float(matrix.max()) if matrix.size else 0.0,
                    "nonzero_edges": int((matrix > 0).sum()),
                    "n_cell_types": int(len(cell_types)),
                    "peak_sender": cell_types[int(np.argmax(outgoing))]
                    if cell_types
                    else None,
                    "peak_receiver": cell_types[int(np.argmax(incoming))]
                    if cell_types
                    else None,
                }
            )
            celltype_rows.extend(
                {
                    "time": float(time_value),
                    "pair": pair,
                    "cell_type": cell_type,
                    "incoming": float(in_value),
                    "outgoing": float(out_value),
                    "total": float(in_value + out_value),
                    "n_cells": int(counts[cell_type]),
                }
                for cell_type, in_value, out_value in zip(
                    cell_types, incoming, outgoing
                )
            )
        coverage_rows.append(
            {
                "time": float(time_value),
                "n_lr_pairs_database": int(len(lr_table)),
                "n_lr_pairs_scored": int(len(scored)),
                "n_skipped_missing_gene": int(skipped_missing),
                "n_partial_complexes": int(partial_complexes),
                "n_lr_pairs_with_missing_subunit": int(pairs_with_missing_subunit),
                "n_lr_pairs_with_unreconstructable_subunit": int(
                    pairs_with_unreconstructable_subunit
                ),
                "n_missing_subunits": int(len(missing_subunits_seen)),
                "n_unreconstructable_subunits": int(
                    len(unreconstructable_subunits_seen)
                ),
                "missing_subunits": ";".join(sorted(missing_subunits_seen)),
                "unreconstructable_subunits": ";".join(
                    sorted(unreconstructable_subunits_seen)
                ),
                "n_duplicate_pairs": int(duplicates),
                "n_cell_types": int(len(cell_types)),
                "n_cells": int(states.shape[0]),
                "n_expression_cells": int(n_expression_cells),
                "expression_source": expression_source,
                "observed_time_expected": matched_observed_time is not None,
                "observed_missing_fallback": bool(observed_missing_fallback),
                "n_pca_features_total": int(len(feature_coverage)),
                "n_pca_features_active": int(active_mask.sum()),
                "n_pca_features_inactive": int((~active_mask).sum()),
                "n_active_lr_features": int(lr_feature_mask.sum()),
                "n_observed_lr_features": int(observed_lr_feature_mask.sum()),
            }
        )

    pair_timecourse = pd.DataFrame(pair_rows)
    if pair_timecourse.empty:
        raise ValueError("No ligand-receptor pairs could be scored.")
    pair_timecourse = pair_timecourse.sort_values(["pair", "time"]).reset_index(
        drop=True
    )
    totals = pair_timecourse.groupby("pair")["score"].sum()
    nonzero_pairs = totals.index[totals > 0]
    pair_timecourse = pair_timecourse.loc[
        pair_timecourse["pair"].isin(nonzero_pairs)
    ].reset_index(drop=True)
    if pair_timecourse.empty:
        raise ValueError("All ligand-receptor trajectories are zero.")
    profile_matrix = pair_timecourse.pivot(
        index="pair", columns="time", values="score"
    ).fillna(0.0)
    clustering = cluster_temporal_profiles(
        profile_matrix,
        n_clusters=int(n_clusters),
        normalization="minmax",
        method=profile_linkage_method,
        cluster_order=profile_cluster_order,
    )
    pattern_summary = _profile_summary(pair_timecourse, clustering)
    celltype_timecourse = pd.DataFrame(celltype_rows)
    if not celltype_timecourse.empty:
        celltype_timecourse = celltype_timecourse.loc[
            celltype_timecourse["pair"].isin(nonzero_pairs)
        ].sort_values(["pair", "time", "total"], ascending=[True, True, False])
        celltype_timecourse = celltype_timecourse.reset_index(drop=True)

    settings = {
        "time_points": [float(value) for value in time_points],
        "annotation_key": annotation_key,
        "matrix_key": matrix_key,
        "spatial_dim": int(spatial_dim),
        "loadings_key": loadings_key,
        "reference_layer": reference_layer,
        "expression_space": expression_space,
        "complex_mode": complex_mode,
        "require_all_subunits": bool(require_all_subunits),
        "duplicate_policy": duplicate_policy,
        "preferred_species_tag": preferred_species_tag,
        "n_clusters": int(n_clusters),
        "pca_reconstruction": (
            "reference_adata"
            if pca_reconstruction is None
            else dict(pca_reconstruction.metadata)
        ),
        "profile_linkage_method": profile_linkage_method,
        "profile_cluster_order": profile_cluster_order,
        "generated_expression_source_space": "log1p",
        "celltype_expression_aggregation": (
            "per-cell source-to-target conversion followed by arithmetic "
            "cell-type mean"
        ),
        "count_space_caveat": (
            "Generated values are arithmetic means of per-cell "
            "expm1(inverse-PCA log1p estimates); they are model-derived "
            "pseudocounts, not observed raw-count means."
            if expression_space == "count"
            else None
        ),
        "generated_expression_feature_policy": (
            "active retained-PCA loadings only; center-only features excluded"
        ),
        "pca_feature_coverage": {
            "n_total": int(len(feature_coverage)),
            "n_active": int(active_mask.sum()),
            "n_inactive": int((~active_mask).sum()),
            "n_active_lr_features": int(lr_feature_mask.sum()),
            "n_observed_lr_features": int(observed_lr_feature_mask.sum()),
            "loading_tolerance": float(feature_coverage["loading_tolerance"].iloc[0]),
            "absolute_tolerance": float(pca_active_absolute_tolerance),
            "relative_tolerance": float(pca_active_relative_tolerance),
        },
        "pca_center_source": (
            "explicit_reconstruction"
            if pca_reconstruction is not None
            else (
                "reference_adata.var['pca_center']"
                if reference_layer is None and "pca_center" in reference_adata.var
                else "reference_matrix_mean"
            )
        ),
        "observed_expression": (
            None
            if observed_adata is None
            else {
                "time_key": resolved_observed_time_key,
                "time_points": [float(value) for value in expected_observed_times],
                "time_atol": observed_time_atol,
                "annotation_key": resolved_observed_annotation_key,
                "layer": observed_layer,
                "source_space": observed_expression_space,
                "missing_time_policy": observed_missing_time_policy,
                "gene_alignment": observed_gene_alignment,
                "cell_alignment": "within_observed_adata_by_time_and_annotation",
            }
        ),
    }
    return LRTemporalProjectionResult(
        pair_timecourse=pair_timecourse,
        celltype_timecourse=celltype_timecourse,
        pattern_summary=pattern_summary,
        clustering=clustering,
        coverage=pd.DataFrame(coverage_rows),
        settings=settings,
    )
