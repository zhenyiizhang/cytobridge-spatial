"""Expression-aware ligand-receptor projection of communication matrices."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .temporal import (
    PCAReconstructionSpec,
    TemporalProfileClusteringResult,
    infer_pca_center,
    inverse_pca_states,
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
    table = pd.read_csv(source) if not isinstance(source, pd.DataFrame) else source.copy()
    if table.empty:
        raise ValueError("Ligand-receptor database is empty.")
    columns = list(map(str, table.columns))
    lower = {column.lower(): column for column in columns}
    ligand_candidates = ("ligand", "ligand_symbol", "source", "gene_a", "0")
    receptor_candidates = ("receptor", "receptor_symbol", "target", "gene_b", "1")
    ligand_col = next((lower[name] for name in ligand_candidates if name in lower), None)
    receptor_col = next((lower[name] for name in receptor_candidates if name in lower), None)
    if ligand_col is None or receptor_col is None:
        usable = [column for column in columns if not column.lower().startswith("unnamed")]
        if len(usable) >= 2:
            ligand_col, receptor_col = usable[:2]
    if ligand_col is None or receptor_col is None:
        raise ValueError(
            f"Could not identify ligand/receptor columns from {columns}."
        )
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
) -> LRTemporalProjectionResult:
    """Project cell-type communication into LR-pair trajectories.

    For sender type ``A`` and receiver type ``B``, the score is
    ``mean_A(ligand) * mean_B(receptor) * communication(A, B)``. Simulated
    PCA states are inverted with loadings and the center recovered from the
    reference AnnData, making the computation reusable across datasets.
    """
    if expression_space not in {"log1p", "count"}:
        raise ValueError("expression_space must be 'log1p' or 'count'.")
    if duplicate_policy not in {"first", "last", "sum", "max"}:
        raise ValueError("duplicate_policy must be first, last, sum, or max.")
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
        else None
    )
    gene_name_map = simplify_gene_names(
        (
            reference_adata.var_names
            if pca_reconstruction is None
            else pca_reconstruction.feature_names
        ),
        preferred_species_tag=preferred_species_tag,
    )
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
        labels = adata_t.obs[annotation_key].astype(str).to_numpy()
        mean_states, counts = _mean_states_by_type(states, labels, cell_types)
        expression = inverse_pca_states(
            reference_adata,
            mean_states,
            spatial_dim=spatial_dim,
            loadings_key=loadings_key,
            center=center,
            layer=reference_layer,
            reconstruction=pca_reconstruction,
        ).astype(np.float64)
        if expression_space == "count":
            expression = np.clip(np.expm1(expression), 0.0, None)
        symbol_to_vector = _symbol_vectors(expression, gene_name_map)

        scored: dict[str, np.ndarray] = {}
        skipped_missing = 0
        partial_complexes = 0
        duplicates = 0
        for row in lr_table.itertuples(index=False):
            ligand = str(row.ligand)
            receptor = str(row.receptor)
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
                "n_duplicate_pairs": int(duplicates),
                "n_cell_types": int(len(cell_types)),
                "n_cells": int(states.shape[0]),
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
    }
    return LRTemporalProjectionResult(
        pair_timecourse=pair_timecourse,
        celltype_timecourse=celltype_timecourse,
        pattern_summary=pattern_summary,
        clustering=clustering,
        coverage=pd.DataFrame(coverage_rows),
        settings=settings,
    )
