#!/usr/bin/env python3
"""Prepare auditable zebrafish inputs for two NicheNet-v2 runs.

This module deliberately stops before running NicheNet.  It performs the
species-independent work once:

* validate raw counts and reconstruct the single-log expression matrix;
* require an exact, high-confidence Ensembl one-to-one zebrafish/mouse map;
* construct lagged receiver response gene sets for adjacent observed stages;
* summarize source-stage expression for sender/receiver candidate gates; and
* map a zebrafish LR table without changing NicheNet's signaling/GRN prior.

The companion ``run_nichenet_v2.R`` script consumes these files and calls the
official ``nichenetr::predict_ligand_activities`` implementation.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import re
import sys
from typing import Mapping, Sequence

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse, stats


SCHEMA_VERSION = 1
FORMAL_NORMALIZATION_TARGET_SUM = 1105.0
FORMAL_H5AD_SHA256 = "433b344b32300c9f58c7de4ac6b8f4ce808934be93b05c939ef24b9ea80fe1cd"
FORMAL_CUSTOM_LR_SHA256 = (
    "27fd0eb35da035a371ef68783d3e2dcf0729668fd58c2bb59f203173ea1b3f37"
)


ORTHOLOGY_COLUMN_ALIASES = {
    "zebrafish_symbol": (
        "zebrafish_symbol",
        "external_gene_name",
        "drerio_gene_symbol",
    ),
    "mouse_symbol": (
        "mouse_symbol",
        "mmusculus_homolog_associated_gene_name",
    ),
    "orthology_type": (
        "orthology_type",
        "mmusculus_homolog_orthology_type",
    ),
    "orthology_confidence": (
        "orthology_confidence",
        "mmusculus_homolog_orthology_confidence",
    ),
    "zebrafish_ensembl_gene": (
        "zebrafish_ensembl_gene",
        "ensembl_gene_id",
    ),
    "mouse_ensembl_gene": (
        "mouse_ensembl_gene",
        "mmusculus_homolog_ensembl_gene",
    ),
}


@dataclass(frozen=True)
class PrepareConfig:
    h5ad: Path
    orthology_csv: Path
    custom_lr_db: Path
    out_dir: Path
    formal_mode: bool = True
    expected_h5ad_sha256: str | None = None
    expected_custom_lr_sha256: str | None = None
    counts_layer: str = "counts"
    time_key: str = "time_point_processed"
    label_key: str = "Annotation"
    transitions: tuple[tuple[str, str], ...] = (
        ("0", "1"),
        ("1", "2"),
        ("2", "3"),
        ("3", "4"),
    )
    normalization_target_sum: float = FORMAL_NORMALIZATION_TARGET_SUM
    normalized_x_tolerance: float = 1e-8
    raw_count_integer_tolerance: float = 1e-8
    verify_x: bool = True
    min_cells_per_receiver_stage: int = 30
    min_expression_fraction: float = 0.05
    min_abs_log2fc: float = 0.25
    fdr_cutoff: float = 0.05
    min_target_genes: int = 20
    min_background_genes: int = 500
    de_chunk_size: int = 256
    stage_label_map: Mapping[str, str] | None = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, object]:
    path = path.resolve()
    return {
        "path": str(path),
        "size_bytes": int(path.stat().st_size),
        "sha256": _sha256(path),
        "md5": _md5(path),
    }


def _as_csr_float64(matrix) -> sparse.csr_matrix:
    if sparse.issparse(matrix):
        return matrix.tocsr().astype(np.float64, copy=True)
    return sparse.csr_matrix(np.asarray(matrix, dtype=np.float64))


def validate_raw_counts(
    counts: sparse.csr_matrix,
    *,
    integer_tolerance: float,
) -> dict[str, object]:
    data = np.asarray(counts.data, dtype=np.float64)
    finite = bool(np.isfinite(data).all())
    nonnegative = bool((data >= 0).all())
    integer_error = float(np.max(np.abs(data - np.rint(data)))) if data.size else 0.0
    audit = {
        "shape": [int(counts.shape[0]), int(counts.shape[1])],
        "nnz": int(counts.nnz),
        "all_finite": finite,
        "all_nonnegative": nonnegative,
        "max_integer_error": integer_error,
        "integer_tolerance": float(integer_tolerance),
        "integer_like": bool(integer_error <= integer_tolerance),
    }
    if not finite:
        raise ValueError("Raw counts contain non-finite values.")
    if not nonnegative:
        raise ValueError("Raw counts contain negative values.")
    if integer_error > integer_tolerance:
        raise ValueError(
            "Raw counts are not integer-like: "
            f"max error {integer_error:.6g} exceeds {integer_tolerance:.6g}."
        )
    libraries = np.asarray(counts.sum(axis=1)).ravel()
    if np.any(libraries <= 0):
        raise ValueError("At least one cell has a zero raw-count library size.")
    audit.update(
        {
            "library_min": float(libraries.min()),
            "library_median_retained_cells": float(np.median(libraries)),
            "library_max": float(libraries.max()),
        }
    )
    return audit


def single_log_from_counts(
    counts,
    *,
    target_sum: float,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix, np.ndarray]:
    """Return normalized linear values, log1p values, and library sizes.

    ``target_sum`` is intentionally explicit.  For the formal corrected
    zebrafish H5AD it is frozen at 1105 by the preprocessing audit; it is not
    recomputed from the retained-cell subset (whose median is different).
    """

    if not np.isfinite(target_sum) or target_sum <= 0:
        raise ValueError("target_sum must be finite and positive.")
    csr = _as_csr_float64(counts)
    libraries = np.asarray(csr.sum(axis=1)).ravel()
    if np.any(libraries <= 0):
        raise ValueError("Cannot normalize cells with zero library size.")
    normalized = sparse.diags(target_sum / libraries).dot(csr).tocsr()
    logged = normalized.copy()
    logged.data = np.log1p(logged.data)
    return normalized, logged, libraries


def validate_single_log_x(
    observed_x,
    expected_logged: sparse.csr_matrix,
    *,
    tolerance: float,
) -> dict[str, object]:
    observed = _as_csr_float64(observed_x)
    if observed.shape != expected_logged.shape:
        raise ValueError(
            f"X shape {observed.shape} does not match expected {expected_logged.shape}."
        )

    expected_support = expected_logged.copy()
    expected_support.data = np.ones_like(expected_support.data)
    observed_support = observed.copy()
    observed_support.eliminate_zeros()
    observed_support.data = np.ones_like(observed_support.data)
    support_mismatch = int((expected_support != observed_support).nnz)

    difference = (observed - expected_logged).tocsr()
    max_abs_error = (
        float(np.max(np.abs(difference.data))) if difference.data.size else 0.0
    )
    audit = {
        "shape": [int(observed.shape[0]), int(observed.shape[1])],
        "support_mismatch_count": support_mismatch,
        "max_abs_error": max_abs_error,
        "absolute_tolerance": float(tolerance),
        "passed": bool(support_mismatch == 0 and max_abs_error <= tolerance),
        "formula": "log1p(counts * frozen_target_sum / cell_library_size)",
    }
    if not audit["passed"]:
        raise ValueError(
            "H5AD X is not the expected single-log expression matrix: "
            f"support_mismatch={support_mismatch}, max_abs_error={max_abs_error:.6g}, "
            f"tolerance={tolerance:.6g}. Refusing to clip, relog, or continue."
        )
    return audit


def _resolve_column(
    frame: pd.DataFrame, logical_name: str, required: bool
) -> str | None:
    for candidate in ORTHOLOGY_COLUMN_ALIASES[logical_name]:
        if candidate in frame.columns:
            return candidate
    if required:
        raise ValueError(
            f"Orthology table lacks {logical_name}; accepted columns are "
            f"{ORTHOLOGY_COLUMN_ALIASES[logical_name]}."
        )
    return None


def load_strict_one_to_one_orthology(
    path: Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    raw = pd.read_csv(path)
    z_col = _resolve_column(raw, "zebrafish_symbol", True)
    m_col = _resolve_column(raw, "mouse_symbol", True)
    type_col = _resolve_column(raw, "orthology_type", True)
    confidence_col = _resolve_column(raw, "orthology_confidence", True)
    z_id_col = _resolve_column(raw, "zebrafish_ensembl_gene", False)
    m_id_col = _resolve_column(raw, "mouse_ensembl_gene", False)

    standardized = pd.DataFrame(
        {
            "zebrafish_symbol": raw[z_col].fillna("").astype(str).str.strip(),
            "mouse_symbol": raw[m_col].fillna("").astype(str).str.strip(),
            "orthology_type": raw[type_col].fillna("").astype(str).str.strip(),
            "orthology_confidence": pd.to_numeric(raw[confidence_col], errors="coerce"),
            "zebrafish_ensembl_gene": (
                raw[z_id_col].fillna("").astype(str).str.strip()
                if z_id_col is not None
                else ""
            ),
            "mouse_ensembl_gene": (
                raw[m_id_col].fillna("").astype(str).str.strip()
                if m_id_col is not None
                else ""
            ),
        }
    )
    nonempty = standardized[
        standardized["zebrafish_symbol"].ne("") & standardized["mouse_symbol"].ne("")
    ].copy()
    one_to_one = nonempty[
        nonempty["orthology_type"].str.casefold().eq("ortholog_one2one")
    ].copy()
    confident = one_to_one[one_to_one["orthology_confidence"].eq(1.0)].copy()
    confident["_z_key"] = confident["zebrafish_symbol"].str.casefold()
    confident["_m_key"] = confident["mouse_symbol"].str.casefold()
    confident = confident.drop_duplicates(["_z_key", "_m_key"])

    z_degree = confident.groupby("_z_key")["_m_key"].nunique()
    m_degree = confident.groupby("_m_key")["_z_key"].nunique()
    strict = confident[
        confident["_z_key"].map(z_degree).eq(1)
        & confident["_m_key"].map(m_degree).eq(1)
    ].copy()
    strict = strict.sort_values(["_z_key", "_m_key"]).drop_duplicates("_z_key")
    strict = strict.drop(columns=["_z_key", "_m_key"]).reset_index(drop=True)

    audit = {
        "input_rows": int(len(raw)),
        "rows_with_nonempty_symbols": int(len(nonempty)),
        "ensembl_ortholog_one2one_rows": int(len(one_to_one)),
        "high_confidence_one2one_rows": int(len(confident)),
        "strict_bijective_symbol_pairs": int(len(strict)),
        "filter": {
            "orthology_type": "ortholog_one2one",
            "orthology_confidence": 1,
            "require_nonempty_symbols": True,
            "require_symbol_level_bijection_after_casefold": True,
        },
    }
    if strict.empty:
        raise ValueError("No strict high-confidence one-to-one orthologues remain.")
    return strict, audit


def restrict_orthology_to_adata_genes(
    strict: pd.DataFrame,
    var_names: Sequence[str],
) -> tuple[pd.DataFrame, np.ndarray, dict[str, object]]:
    lookup: dict[str, list[int]] = {}
    for index, name in enumerate(map(str, var_names)):
        lookup.setdefault(name.casefold(), []).append(index)
    unique_lookup = {
        key: indices[0] for key, indices in lookup.items() if len(indices) == 1
    }
    duplicated_keys = {key for key, indices in lookup.items() if len(indices) > 1}

    rows = []
    indices = []
    for row in strict.itertuples(index=False):
        key = str(row.zebrafish_symbol).casefold()
        if key not in unique_lookup:
            continue
        rows.append(row._asdict())
        indices.append(unique_lookup[key])
    present = pd.DataFrame(rows, columns=strict.columns)
    audit = {
        "h5ad_var_count": int(len(var_names)),
        "h5ad_casefold_duplicate_symbol_count": int(len(duplicated_keys)),
        "strict_pairs_before_h5ad_intersection": int(len(strict)),
        "strict_pairs_present_in_h5ad": int(len(present)),
    }
    if present.empty:
        raise ValueError("No strict orthologues match the H5AD var names.")
    return present.reset_index(drop=True), np.asarray(indices, dtype=int), audit


def _bh_adjust(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    p = np.where(np.isfinite(p), p, 1.0)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 1.0
    n = len(p)
    for reverse_index in range(n - 1, -1, -1):
        rank = reverse_index + 1
        original_index = order[reverse_index]
        running = min(running, p[original_index] * n / rank)
        adjusted[original_index] = running
    return np.clip(adjusted, 0.0, 1.0)


def _wilcoxon_by_chunk(
    source: sparse.csr_matrix,
    target: sparse.csr_matrix,
    *,
    test_mask: np.ndarray,
    chunk_size: int,
) -> np.ndarray:
    p_values = np.ones(source.shape[1], dtype=float)
    indices = np.flatnonzero(test_mask)
    for start in range(0, len(indices), chunk_size):
        selected = indices[start : start + chunk_size]
        source_dense = source[:, selected].toarray()
        target_dense = target[:, selected].toarray()
        try:
            result = stats.mannwhitneyu(
                target_dense,
                source_dense,
                axis=0,
                alternative="two-sided",
                method="asymptotic",
                use_continuity=True,
            )
            p_values[selected] = np.asarray(result.pvalue, dtype=float)
        except Exception:
            for local_index, gene_index in enumerate(selected):
                try:
                    result = stats.mannwhitneyu(
                        target_dense[:, local_index],
                        source_dense[:, local_index],
                        alternative="two-sided",
                        method="asymptotic",
                        use_continuity=True,
                    )
                    p_values[gene_index] = float(result.pvalue)
                except Exception:
                    p_values[gene_index] = 1.0
    return p_values


def _stage_id(value: object) -> str:
    try:
        numeric = float(value)
        if np.isfinite(numeric):
            return format(numeric, ".15g")
    except (TypeError, ValueError):
        pass
    return str(value)


def _match_stage(requested: str, observed: Sequence[object]) -> object:
    matches = [value for value in observed if _stage_id(value) == _stage_id(requested)]
    if len(matches) != 1:
        raise ValueError(
            f"Transition stage {requested!r} matched {len(matches)} observed stages: {matches}."
        )
    return matches[0]


def _stage_label(value: object, mapping: Mapping[str, str] | None) -> str:
    if mapping is None:
        return str(value)
    key = _stage_id(value)
    candidates = [
        mapping_key for mapping_key in mapping if _stage_id(mapping_key) == key
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"stage_label_map must contain exactly one entry for stage {value!r}."
        )
    return str(mapping[candidates[0]])


def _slug(value: object, *, limit: int = 70) -> str:
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_")
    return (token or "unnamed")[:limit]


def expression_summary(
    *,
    counts: sparse.csr_matrix,
    normalized: sparse.csr_matrix,
    logged: sparse.csr_matrix,
    obs: pd.DataFrame,
    time_key: str,
    label_key: str,
    mouse_genes: Sequence[str],
    zebrafish_genes: Sequence[str],
    stage_label_map: Mapping[str, str] | None,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    group_frame = obs[[time_key, label_key]].copy()
    for (stage, cell_type), positions in group_frame.groupby(
        [time_key, label_key], observed=True, sort=True
    ).indices.items():
        positions = np.asarray(positions, dtype=int)
        group_counts = counts[positions]
        detected = np.asarray((group_counts > 0).sum(axis=0)).ravel() / len(positions)
        mean_linear = np.asarray(normalized[positions].mean(axis=0)).ravel()
        mean_log = np.asarray(logged[positions].mean(axis=0)).ravel()
        rows.append(
            pd.DataFrame(
                {
                    "stage_id": _stage_id(stage),
                    "stage_label": _stage_label(stage, stage_label_map),
                    "cell_type": str(cell_type),
                    "n_cells": int(len(positions)),
                    "gene_mouse": list(map(str, mouse_genes)),
                    "gene_zebrafish": list(map(str, zebrafish_genes)),
                    "pct_detected": detected,
                    "mean_normalized_linear": mean_linear,
                    "mean_log1p": mean_log,
                }
            )
        )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _standardize_lr_database(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    rename = {"0": "ligand", "1": "receptor", "2": "pathway", "3": "category"}
    raw = raw.rename(columns=rename)
    required = {"ligand", "receptor"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"LR database lacks columns {sorted(missing)}.")
    if "pathway" not in raw:
        raw["pathway"] = ""
    if "category" not in raw:
        raw["category"] = ""
    standardized = raw[["ligand", "receptor", "pathway", "category"]].copy()
    for column in standardized.columns:
        standardized[column] = standardized[column].fillna("").astype(str).str.strip()
    standardized.insert(0, "input_row_id", np.arange(len(standardized), dtype=int))
    standardized["is_exact_duplicate"] = standardized.duplicated(
        ["ligand", "receptor", "pathway", "category"]
    )
    return standardized


def _split_complex(value: str) -> list[str]:
    return [token.strip() for token in value.split("_") if token.strip()]


def map_custom_lr_database(
    path: Path,
    strict_present_orthology: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    raw = _standardize_lr_database(path)
    orthology_map = {
        str(row.zebrafish_symbol).casefold(): str(row.mouse_symbol)
        for row in strict_present_orthology.itertuples(index=False)
    }
    audit_rows: list[dict[str, object]] = []
    for row in raw.itertuples(index=False):
        ligand_parts = _split_complex(row.ligand)
        receptor_parts = _split_complex(row.receptor)
        ligand_mouse = [orthology_map.get(part.casefold()) for part in ligand_parts]
        receptor_mouse = [orthology_map.get(part.casefold()) for part in receptor_parts]

        reasons: list[str] = []
        if row.is_exact_duplicate:
            reasons.append("exact_duplicate")
        if not ligand_parts or not receptor_parts:
            reasons.append("empty_ligand_or_receptor")
        missing_ligands = [
            part for part, mapped in zip(ligand_parts, ligand_mouse) if mapped is None
        ]
        missing_receptors = [
            part
            for part, mapped in zip(receptor_parts, receptor_mouse)
            if mapped is None
        ]
        if missing_ligands:
            reasons.append("ligand_component_missing_strict_one2one")
        if missing_receptors:
            reasons.append("receptor_component_missing_strict_one2one")
        if len(ligand_parts) > 1:
            reasons.append("unsupported_multisubunit_ligand_prior_column")

        eligible = not reasons
        audit_rows.append(
            {
                "input_row_id": int(row.input_row_id),
                "ligand_zebrafish": row.ligand,
                "receptor_zebrafish": row.receptor,
                "pathway": row.pathway,
                "category": row.category,
                "ligand_components_zebrafish": ";".join(ligand_parts),
                "receptor_components_zebrafish": ";".join(receptor_parts),
                "ligand_mouse": (
                    ligand_mouse[0]
                    if len(ligand_mouse) == 1 and ligand_mouse[0]
                    else ""
                ),
                "receptor_mouse_components": ";".join(
                    mapped for mapped in receptor_mouse if mapped is not None
                ),
                "missing_ligand_components": ";".join(missing_ligands),
                "missing_receptor_components": ";".join(missing_receptors),
                "eligible_for_custom_nichenet_gate": bool(eligible),
                "exclusion_reason": ";".join(reasons),
            }
        )
    audit_frame = pd.DataFrame(audit_rows)
    eligible = audit_frame[audit_frame["eligible_for_custom_nichenet_gate"]].copy()

    if eligible.empty:
        mapped = pd.DataFrame(
            columns=[
                "ligand_mouse",
                "receptor_mouse_components",
                "ligand_zebrafish",
                "receptor_zebrafish",
                "pathways",
                "categories",
                "n_input_rows",
            ]
        )
    else:
        mapped_rows = []
        for (_, _), group in eligible.groupby(
            ["ligand_mouse", "receptor_mouse_components"], sort=True
        ):
            mapped_rows.append(
                {
                    "ligand_mouse": group["ligand_mouse"].iloc[0],
                    "receptor_mouse_components": group[
                        "receptor_mouse_components"
                    ].iloc[0],
                    "ligand_zebrafish": ";".join(
                        sorted(set(group["ligand_zebrafish"]))
                    ),
                    "receptor_zebrafish": ";".join(
                        sorted(set(group["receptor_zebrafish"]))
                    ),
                    "pathways": ";".join(
                        sorted(value for value in set(group["pathway"]) if value)
                    ),
                    "categories": ";".join(
                        sorted(value for value in set(group["category"]) if value)
                    ),
                    "n_input_rows": int(len(group)),
                }
            )
        mapped = pd.DataFrame(mapped_rows)

    reason_counts: dict[str, int] = {}
    for reason_string in audit_frame["exclusion_reason"]:
        for reason in filter(None, str(reason_string).split(";")):
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    summary = {
        "input_rows": int(len(raw)),
        "exact_duplicate_rows": int(raw["is_exact_duplicate"].sum()),
        "eligible_input_rows": int(
            audit_frame["eligible_for_custom_nichenet_gate"].sum()
        ),
        "eligible_unique_lr_pairs": int(len(mapped)),
        "excluded_rows": int((~audit_frame["eligible_for_custom_nichenet_gate"]).sum()),
        "exclusion_reason_counts": reason_counts,
        "complex_rule": {
            "receptor": "all components must have strict one-to-one mappings; expression AND gate is applied in R",
            "ligand": "multi-subunit ligands are excluded because the fixed NicheNet-v2 ligand-target matrix has no composite column",
        },
    }
    return mapped, audit_frame, summary


def _write_gene_list(
    path: Path, mouse: Sequence[str], zebrafish: Sequence[str]
) -> None:
    pd.DataFrame(
        {
            "gene_mouse": list(map(str, mouse)),
            "gene_zebrafish": list(map(str, zebrafish)),
        }
    ).to_csv(path, index=False)


def _receiver_de(
    *,
    source_indices: np.ndarray,
    target_indices: np.ndarray,
    counts: sparse.csr_matrix,
    normalized: sparse.csr_matrix,
    logged: sparse.csr_matrix,
    mouse_genes: np.ndarray,
    zebrafish_genes: np.ndarray,
    config: PrepareConfig,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    source_counts = counts[source_indices]
    target_counts = counts[target_indices]
    source_normalized = normalized[source_indices]
    target_normalized = normalized[target_indices]
    source_logged = logged[source_indices]
    target_logged = logged[target_indices]

    source_pct = np.asarray((source_counts > 0).mean(axis=0)).ravel()
    target_pct = np.asarray((target_counts > 0).mean(axis=0)).ravel()
    pooled_pct = np.asarray(
        (sparse.vstack([source_counts, target_counts]) > 0).mean(axis=0)
    ).ravel()
    source_mean_linear = np.asarray(source_normalized.mean(axis=0)).ravel()
    target_mean_linear = np.asarray(target_normalized.mean(axis=0)).ravel()
    source_mean_log = np.asarray(source_logged.mean(axis=0)).ravel()
    target_mean_log = np.asarray(target_logged.mean(axis=0)).ravel()
    epsilon = 1e-12
    log2fc = np.log2((target_mean_linear + epsilon) / (source_mean_linear + epsilon))

    test_mask = np.maximum(source_pct, target_pct) >= config.min_expression_fraction
    p_values = _wilcoxon_by_chunk(
        source_logged,
        target_logged,
        test_mask=test_mask,
        chunk_size=config.de_chunk_size,
    )
    q_values = _bh_adjust(p_values)
    gene_set_mask = (
        test_mask
        & (q_values <= config.fdr_cutoff)
        & (np.abs(log2fc) >= config.min_abs_log2fc)
    )
    background_mask = pooled_pct >= config.min_expression_fraction

    de = pd.DataFrame(
        {
            "gene_mouse": mouse_genes,
            "gene_zebrafish": zebrafish_genes,
            "pct_source": source_pct,
            "pct_target": target_pct,
            "pct_pooled": pooled_pct,
            "mean_normalized_linear_source": source_mean_linear,
            "mean_normalized_linear_target": target_mean_linear,
            "mean_log1p_source": source_mean_log,
            "mean_log1p_target": target_mean_log,
            "avg_log2fc_target_vs_source": log2fc,
            "p_value_cell_level_wilcoxon": p_values,
            "q_value_bh": q_values,
            "passes_min_expression": test_mask,
            "in_receiver_response_geneset": gene_set_mask,
            "in_receiver_background": background_mask,
        }
    ).sort_values(
        ["in_receiver_response_geneset", "q_value_bh", "avg_log2fc_target_vs_source"],
        ascending=[False, True, False],
    )
    return de, gene_set_mask, background_mask


def _validate_config(config: PrepareConfig) -> None:
    for path, label in (
        (config.h5ad, "h5ad"),
        (config.orthology_csv, "orthology_csv"),
        (config.custom_lr_db, "custom_lr_db"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    if config.formal_mode:
        if config.normalization_target_sum != FORMAL_NORMALIZATION_TARGET_SUM:
            raise ValueError(
                "Formal zebrafish mode requires normalization_target_sum exactly "
                f"{FORMAL_NORMALIZATION_TARGET_SUM:g}; use --nonformal only for tests "
                "or a separately documented new dataset."
            )
        if not config.verify_x:
            raise ValueError("Formal zebrafish mode requires X verification.")
        for supplied, frozen, label in (
            (config.expected_h5ad_sha256, FORMAL_H5AD_SHA256, "H5AD"),
            (
                config.expected_custom_lr_sha256,
                FORMAL_CUSTOM_LR_SHA256,
                "custom LR database",
            ),
        ):
            if supplied is not None and supplied.casefold() != frozen.casefold():
                raise ValueError(
                    f"Formal {label} expected SHA256 must be the frozen value {frozen}; "
                    f"received {supplied}."
                )

    expected_h5ad = (
        FORMAL_H5AD_SHA256 if config.formal_mode else config.expected_h5ad_sha256
    )
    expected_custom_lr = (
        FORMAL_CUSTOM_LR_SHA256
        if config.formal_mode
        else config.expected_custom_lr_sha256
    )
    for path, expected, label in (
        (config.h5ad, expected_h5ad, "H5AD"),
        (config.custom_lr_db, expected_custom_lr, "custom LR database"),
    ):
        if expected is None:
            continue
        observed = _sha256(path)
        if observed.casefold() != expected.casefold():
            raise ValueError(
                f"{label} SHA256 mismatch: expected {expected}, observed {observed}."
            )
    if config.out_dir.exists() and any(config.out_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is non-empty: {config.out_dir}. Use a new immutable run directory."
        )
    if not 0 < config.min_expression_fraction <= 1:
        raise ValueError("min_expression_fraction must be in (0, 1].")
    if not 0 < config.fdr_cutoff <= 1:
        raise ValueError("fdr_cutoff must be in (0, 1].")
    if config.normalization_target_sum <= 0:
        raise ValueError("normalization_target_sum must be positive.")


def prepare_shared_inputs(config: PrepareConfig) -> dict[str, object]:
    _validate_config(config)
    config.out_dir.mkdir(parents=True, exist_ok=True)
    units_dir = config.out_dir / "units"
    units_dir.mkdir()

    adata = ad.read_h5ad(config.h5ad)
    if config.counts_layer not in adata.layers:
        raise KeyError(f"H5AD lacks layers[{config.counts_layer!r}].")
    for key in (config.time_key, config.label_key):
        if key not in adata.obs:
            raise KeyError(f"H5AD obs lacks {key!r}.")

    full_counts = _as_csr_float64(adata.layers[config.counts_layer])
    raw_audit = validate_raw_counts(
        full_counts, integer_tolerance=config.raw_count_integer_tolerance
    )
    full_normalized, full_logged, _ = single_log_from_counts(
        full_counts, target_sum=config.normalization_target_sum
    )
    if config.verify_x:
        x_audit = validate_single_log_x(
            adata.X,
            full_logged,
            tolerance=config.normalized_x_tolerance,
        )
    else:
        x_audit = {
            "passed": None,
            "skipped": True,
            "reason": "--skip-x-verification was explicitly requested",
        }

    strict, orthology_audit = load_strict_one_to_one_orthology(config.orthology_csv)
    present, gene_indices, h5ad_orthology_audit = restrict_orthology_to_adata_genes(
        strict, adata.var_names.astype(str)
    )
    orthology_audit.update(h5ad_orthology_audit)
    present.to_csv(config.out_dir / "orthology_strict_one2one_present.csv", index=False)

    counts = full_counts[:, gene_indices].tocsr()
    normalized = full_normalized[:, gene_indices].tocsr()
    logged = full_logged[:, gene_indices].tocsr()
    mouse_genes = present["mouse_symbol"].astype(str).to_numpy()
    zebrafish_genes = present["zebrafish_symbol"].astype(str).to_numpy()

    summary = expression_summary(
        counts=counts,
        normalized=normalized,
        logged=logged,
        obs=adata.obs,
        time_key=config.time_key,
        label_key=config.label_key,
        mouse_genes=mouse_genes,
        zebrafish_genes=zebrafish_genes,
        stage_label_map=config.stage_label_map,
    )
    expression_path = config.out_dir / "expression_by_stage_celltype.csv.gz"
    summary.to_csv(
        expression_path, index=False, compression="gzip", float_format="%.10g"
    )

    mapped_lr, lr_audit_frame, lr_audit = map_custom_lr_database(
        config.custom_lr_db, present
    )
    mapped_lr.to_csv(
        config.out_dir / "custom_lr_strict_one2one_mapped.csv", index=False
    )
    lr_audit_frame.to_csv(config.out_dir / "custom_lr_mapping_audit.csv", index=False)

    observed_stages = list(pd.unique(adata.obs[config.time_key]))
    resolved_transitions = [
        (
            _match_stage(source, observed_stages),
            _match_stage(target, observed_stages),
        )
        for source, target in config.transitions
    ]

    unit_rows: list[dict[str, object]] = []
    unit_counter = 0
    labels = adata.obs[config.label_key].astype(str).to_numpy()
    times = adata.obs[config.time_key].to_numpy()
    for source_stage, target_stage in resolved_transitions:
        source_stage_mask = np.asarray(
            [_stage_id(value) == _stage_id(source_stage) for value in times]
        )
        target_stage_mask = np.asarray(
            [_stage_id(value) == _stage_id(target_stage) for value in times]
        )
        source_labels = set(labels[source_stage_mask])
        target_labels = set(labels[target_stage_mask])
        for receiver in sorted(source_labels & target_labels):
            source_indices = np.flatnonzero(source_stage_mask & (labels == receiver))
            target_indices = np.flatnonzero(target_stage_mask & (labels == receiver))
            if (
                len(source_indices) < config.min_cells_per_receiver_stage
                or len(target_indices) < config.min_cells_per_receiver_stage
            ):
                status = "ineligible_too_few_receiver_cells"
                n_targets = 0
                n_background = 0
                unit_relative = ""
            else:
                unit_id = (
                    f"u{unit_counter:04d}_"
                    f"{_slug(_stage_id(source_stage))}_to_{_slug(_stage_id(target_stage))}_"
                    f"{_slug(receiver)}"
                )
                unit_counter += 1
                unit_path = units_dir / unit_id
                unit_path.mkdir()
                de, target_mask, background_mask = _receiver_de(
                    source_indices=source_indices,
                    target_indices=target_indices,
                    counts=counts,
                    normalized=normalized,
                    logged=logged,
                    mouse_genes=mouse_genes,
                    zebrafish_genes=zebrafish_genes,
                    config=config,
                )
                de.to_csv(
                    unit_path / "receiver_de.csv.gz",
                    index=False,
                    compression="gzip",
                    float_format="%.10g",
                )
                _write_gene_list(
                    unit_path / "receiver_response_genes.csv",
                    mouse_genes[target_mask],
                    zebrafish_genes[target_mask],
                )
                _write_gene_list(
                    unit_path / "receiver_background_genes.csv",
                    mouse_genes[background_mask],
                    zebrafish_genes[background_mask],
                )
                n_targets = int(target_mask.sum())
                n_background = int(background_mask.sum())
                status = "eligible"
                if n_targets < config.min_target_genes:
                    status = "ineligible_too_few_response_genes"
                elif n_background < config.min_background_genes:
                    status = "ineligible_too_few_background_genes"
                unit_relative = str(unit_path.relative_to(config.out_dir))
                metadata = {
                    "unit_id": unit_id,
                    "source_stage_id": _stage_id(source_stage),
                    "target_stage_id": _stage_id(target_stage),
                    "source_stage_label": _stage_label(
                        source_stage, config.stage_label_map
                    ),
                    "target_stage_label": _stage_label(
                        target_stage, config.stage_label_map
                    ),
                    "receiver": receiver,
                    "n_receiver_source": int(len(source_indices)),
                    "n_receiver_target": int(len(target_indices)),
                    "n_response_genes_before_nichenet_prior_intersection": n_targets,
                    "n_background_genes_before_nichenet_prior_intersection": n_background,
                    "status": status,
                    "cell_level_q_value_caveat": (
                        "Cells are not biological replicates; q values select a descriptive "
                        "receiver response gene set and are not embryo-level inference."
                    ),
                }
                (unit_path / "unit_metadata.json").write_text(
                    json.dumps(metadata, indent=2), encoding="utf-8"
                )

            unit_rows.append(
                {
                    "unit_id": (
                        unit_id
                        if "unit_id" in locals() and unit_relative
                        else f"skipped_{_slug(source_stage)}_{_slug(target_stage)}_{_slug(receiver)}"
                    ),
                    "source_stage_id": _stage_id(source_stage),
                    "target_stage_id": _stage_id(target_stage),
                    "source_stage_label": _stage_label(
                        source_stage, config.stage_label_map
                    ),
                    "target_stage_label": _stage_label(
                        target_stage, config.stage_label_map
                    ),
                    "receiver": receiver,
                    "n_receiver_source": int(len(source_indices)),
                    "n_receiver_target": int(len(target_indices)),
                    "n_response_genes": n_targets,
                    "n_background_genes": n_background,
                    "status": status,
                    "unit_dir": unit_relative,
                }
            )
            if "unit_id" in locals():
                del unit_id

    units_manifest = pd.DataFrame(
        unit_rows,
        columns=[
            "unit_id",
            "source_stage_id",
            "target_stage_id",
            "source_stage_label",
            "target_stage_label",
            "receiver",
            "n_receiver_source",
            "n_receiver_target",
            "n_response_genes",
            "n_background_genes",
            "status",
            "unit_dir",
        ],
    )
    units_manifest.to_csv(config.out_dir / "units_manifest.csv", index=False)
    status_counts = {
        str(key): int(value)
        for key, value in units_manifest["status"].value_counts().items()
    }

    coverage_rows = [
        {
            "component": "expression",
            "metric": "h5ad_n_obs",
            "value": int(adata.n_obs),
        },
        {
            "component": "expression",
            "metric": "h5ad_n_vars",
            "value": int(adata.n_vars),
        },
        {
            "component": "orthology",
            "metric": "raw_input_rows",
            "value": int(orthology_audit["input_rows"]),
        },
        {
            "component": "orthology",
            "metric": "strict_one2one_pairs_present_in_h5ad",
            "value": int(orthology_audit["strict_pairs_present_in_h5ad"]),
        },
        {
            "component": "custom_lr",
            "metric": "input_rows",
            "value": int(lr_audit["input_rows"]),
        },
        {
            "component": "custom_lr",
            "metric": "eligible_unique_lr_pairs",
            "value": int(lr_audit["eligible_unique_lr_pairs"]),
        },
        {
            "component": "receiver_units",
            "metric": "total",
            "value": int(len(units_manifest)),
        },
    ]
    coverage_rows.extend(
        {
            "component": "receiver_units",
            "metric": f"status:{status}",
            "value": count,
        }
        for status, count in sorted(status_counts.items())
    )
    coverage_rows.extend(
        {
            "component": "custom_lr_exclusion",
            "metric": reason,
            "value": int(count),
        }
        for reason, count in sorted(lr_audit["exclusion_reason_counts"].items())
    )
    pd.DataFrame(coverage_rows).to_csv(
        config.out_dir / "coverage_summary.csv", index=False
    )

    output_files = []
    for path in sorted(config.out_dir.rglob("*")):
        if path.is_file() and path.name != "prepare_manifest.json":
            output_files.append(
                {
                    "path": str(path.relative_to(config.out_dir)),
                    "size_bytes": int(path.stat().st_size),
                    "sha256": _sha256(path),
                    "md5": _md5(path),
                }
            )

    manifest: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "workflow": "reviewer_zebrafish_nichenet_shared_input_preparation",
        "status": "complete",
        "formal_mode": bool(config.formal_mode),
        "created_at_utc": _utc_now(),
        "inputs": {
            "h5ad": _file_record(config.h5ad),
            "orthology_csv": _file_record(config.orthology_csv),
            "custom_lr_db": _file_record(config.custom_lr_db),
        },
        "h5ad": {
            "shape": [int(adata.n_obs), int(adata.n_vars)],
            "counts_layer": config.counts_layer,
            "time_key": config.time_key,
            "label_key": config.label_key,
            "observed_stage_counts": {
                _stage_id(key): int(value)
                for key, value in adata.obs[config.time_key].value_counts().items()
            },
        },
        "normalization": {
            "frozen_target_sum": float(config.normalization_target_sum),
            "target_sum_provenance": (
                "The formal zebrafish preprocessing audit resolved target_sum=1105 "
                "from the full source dataset. It must not be replaced by the retained-cell median."
                if config.formal_mode
                else "Nonformal/test run: target_sum was explicitly supplied and carries no formal zebrafish provenance claim."
            ),
            "raw_counts": raw_audit,
            "x_validation": x_audit,
        },
        "orthology": orthology_audit,
        "custom_lr_mapping": lr_audit,
        "receiver_units": {
            "n_rows": int(len(units_manifest)),
            "status_counts": status_counts,
            "transitions": [
                {
                    "source_stage_id": _stage_id(source),
                    "target_stage_id": _stage_id(target),
                }
                for source, target in resolved_transitions
            ],
        },
        "parameters": {
            **{
                key: str(value) if isinstance(value, Path) else value
                for key, value in asdict(config).items()
                if key not in {"stage_label_map", "transitions"}
            },
            "transitions": [list(pair) for pair in config.transitions],
            "stage_label_map": dict(config.stage_label_map or {}),
        },
        "output_files": output_files,
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "anndata": ad.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "interpretation_boundary": (
            "These files support cross-species NicheNet-v2 ligand-activity analysis. "
            "They do not turn NicheNet activity into a spatial or biochemical communication strength."
        ),
    }
    (config.out_dir / "prepare_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def _parse_transitions(value: str) -> tuple[tuple[str, str], ...]:
    pairs: list[tuple[str, str]] = []
    for token in value.split(","):
        fields = [field.strip() for field in token.split(":")]
        if len(fields) != 2 or not all(fields):
            raise argparse.ArgumentTypeError(
                "Transitions must be comma-separated SOURCE:TARGET pairs."
            )
        pairs.append((fields[0], fields[1]))
    if not pairs:
        raise argparse.ArgumentTypeError("At least one transition is required.")
    return tuple(pairs)


def _parse_stage_label_map(value: str | None) -> Mapping[str, str] | None:
    if value is None:
        return None
    candidate = Path(value)
    text = candidate.read_text(encoding="utf-8") if candidate.is_file() else value
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("stage-label-map must be a JSON object or JSON file.")
    return {str(key): str(label) for key, label in payload.items()}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5ad", type=Path, required=True)
    parser.add_argument("--orthology-csv", type=Path, required=True)
    parser.add_argument("--custom-lr-db", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--nonformal",
        action="store_true",
        help="Permit non-frozen inputs/normalization for tests or a separately documented new dataset.",
    )
    parser.add_argument("--expected-h5ad-sha256", default=None)
    parser.add_argument("--expected-custom-lr-sha256", default=None)
    parser.add_argument("--counts-layer", default="counts")
    parser.add_argument("--time-key", default="time_point_processed")
    parser.add_argument("--label-key", default="Annotation")
    parser.add_argument(
        "--transitions",
        type=_parse_transitions,
        default=_parse_transitions("0:1,1:2,2:3,3:4"),
    )
    parser.add_argument("--stage-label-map", default=None)
    parser.add_argument(
        "--normalization-target-sum",
        type=float,
        default=FORMAL_NORMALIZATION_TARGET_SUM,
    )
    parser.add_argument("--normalized-x-tolerance", type=float, default=1e-8)
    parser.add_argument("--raw-count-integer-tolerance", type=float, default=1e-8)
    parser.add_argument("--skip-x-verification", action="store_true")
    parser.add_argument("--min-cells-per-receiver-stage", type=int, default=30)
    parser.add_argument("--min-expression-fraction", type=float, default=0.05)
    parser.add_argument("--min-abs-log2fc", type=float, default=0.25)
    parser.add_argument("--fdr-cutoff", type=float, default=0.05)
    parser.add_argument("--min-target-genes", type=int, default=20)
    parser.add_argument("--min-background-genes", type=int, default=500)
    parser.add_argument("--de-chunk-size", type=int, default=256)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = PrepareConfig(
        h5ad=args.h5ad,
        orthology_csv=args.orthology_csv,
        custom_lr_db=args.custom_lr_db,
        out_dir=args.out_dir,
        formal_mode=not args.nonformal,
        expected_h5ad_sha256=args.expected_h5ad_sha256,
        expected_custom_lr_sha256=args.expected_custom_lr_sha256,
        counts_layer=args.counts_layer,
        time_key=args.time_key,
        label_key=args.label_key,
        transitions=args.transitions,
        normalization_target_sum=args.normalization_target_sum,
        normalized_x_tolerance=args.normalized_x_tolerance,
        raw_count_integer_tolerance=args.raw_count_integer_tolerance,
        verify_x=not args.skip_x_verification,
        min_cells_per_receiver_stage=args.min_cells_per_receiver_stage,
        min_expression_fraction=args.min_expression_fraction,
        min_abs_log2fc=args.min_abs_log2fc,
        fdr_cutoff=args.fdr_cutoff,
        min_target_genes=args.min_target_genes,
        min_background_genes=args.min_background_genes,
        de_chunk_size=args.de_chunk_size,
        stage_label_map=_parse_stage_label_map(args.stage_label_map),
    )
    manifest = prepare_shared_inputs(config)
    print(
        json.dumps(
            {"status": manifest["status"], "out_dir": str(config.out_dir)}, indent=2
        )
    )


if __name__ == "__main__":
    main()
