"""Shared input and output contracts for the zebrafish CCC comparison.

The important preprocessing decision lives here instead of in each method
runner: expression always comes from the declared raw-count layer, library
sizes are calculated over *all* measured genes, and the selected LR genes are
then transformed exactly once with the target frozen in the preprocessing
audit (1105 for the formal zebrafish artifact) plus ``log1p``.  The
reconstructed values are checked against the frozen H5AD ``X``.
"""

from __future__ import annotations

import hashlib
import json
import platform
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import anndata as ad
import numpy as np
import pandas as pd
import scipy
from scipy import sparse
from scipy.io import mmwrite


LR_COLUMNS = ["database_row", "ligand", "receptor", "pathway", "category"]
COMMON_SCORE_COLUMNS = [
    "method",
    "database_variant",
    "stage",
    "stage_time",
    "sender_type",
    "receiver_type",
    "ligand",
    "receptor",
    "pathway",
    "category",
    "interaction_id",
    "score",
    "p_value",
    "significant",
    "n_sender_cells",
    "n_receiver_cells",
    "score_semantics",
]


@dataclass(frozen=True)
class PreparedInputs:
    adata: ad.AnnData
    lr_database: pd.DataFrame
    stage_order: list[str]
    stage_times: dict[str, float | None]
    diagnostics: dict[str, object]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def json_dump(payload: object, path: Path) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def file_record(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def software_versions() -> dict[str, str]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "anndata": ad.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
    }


def json_ready(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def _find_column(frame: pd.DataFrame, candidates: Sequence[str]) -> str:
    by_string = {str(column).strip().lower(): column for column in frame.columns}
    for candidate in candidates:
        if candidate.lower() in by_string:
            return by_string[candidate.lower()]
    raise ValueError(f"None of columns {list(candidates)!r} is present in {list(frame.columns)!r}")


def read_lr_database(path: Path) -> pd.DataFrame:
    """Read a COMMOT-style CellChatDB flat table without silently deduplicating.

    The first unnamed/index column is retained as ``database_row``.  CellChat
    uses it to verify the exact official zebrafish database rows.  Exact flat
    duplicates are therefore intentional and remain distinguishable.
    """

    raw = pd.read_csv(path)
    ligand_col = _find_column(raw, ["ligand", "0"])
    receptor_col = _find_column(raw, ["receptor", "1"])
    pathway_col = _find_column(raw, ["pathway", "pathway_name", "2"])
    category_col = _find_column(raw, ["category", "annotation", "3"])
    index_candidates = [column for column in raw.columns if str(column).lower().startswith("unnamed")]
    if "database_row" in raw.columns:
        database_rows = pd.to_numeric(raw["database_row"], errors="raise").astype(int)
    elif index_candidates:
        database_rows = pd.to_numeric(raw[index_candidates[0]], errors="raise").astype(int)
    else:
        database_rows = pd.Series(np.arange(len(raw), dtype=int), index=raw.index)

    result = pd.DataFrame(
        {
            "database_row": database_rows,
            "ligand": raw[ligand_col].astype(str).str.strip().str.lower(),
            "receptor": raw[receptor_col].astype(str).str.strip().str.lower(),
            "pathway": raw[pathway_col].astype(str).str.strip(),
            "category": raw[category_col].astype(str).str.strip(),
        }
    )
    if result[LR_COLUMNS[1:]].replace("", np.nan).isna().any().any():
        raise ValueError("LR database contains empty ligand/receptor/pathway/category values")
    if result["database_row"].duplicated().any():
        raise ValueError("database_row must be unique")
    result["interaction_id"] = result.apply(
        lambda row: f"dbrow_{int(row.database_row)}:{row.ligand}->{row.receptor}", axis=1
    )
    result["lr_key"] = result["ligand"] + "|" + result["receptor"]
    result["flat_key"] = (
        result["lr_key"] + "|" + result["pathway"] + "|" + result["category"]
    )
    return result


def complex_subunits(token: str) -> tuple[str, ...]:
    """Return CellChatDB/COMMOT underscore-delimited complex subunits."""

    parts = tuple(part.strip().lower() for part in str(token).split("_") if part.strip())
    if not parts:
        raise ValueError(f"Invalid empty LR token: {token!r}")
    return parts


def filter_lr_database_by_features(
    database: pd.DataFrame, feature_names: Iterable[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply only a strict structural complex-availability filter.

    Expression prevalence is deliberately not used here: all methods receive
    the same global LR universe, including pairs that are zero in a stage.
    """

    features = {str(gene).strip().lower() for gene in feature_names}
    keep: list[bool] = []
    missing_values: list[str] = []
    for row in database.itertuples(index=False):
        required = (*complex_subunits(row.ligand), *complex_subunits(row.receptor))
        missing = sorted(set(required).difference(features))
        keep.append(not missing)
        missing_values.append(";".join(missing))
    audit = database[["database_row", "interaction_id", "ligand", "receptor"]].copy()
    audit["all_subunits_present"] = keep
    audit["missing_subunits"] = missing_values
    filtered = database.loc[np.asarray(keep, dtype=bool)].reset_index(drop=True)
    if filtered.empty:
        raise ValueError("No LR rows remain after strict complex feature filtering")
    return filtered, audit


def _matrix_data(matrix: object) -> np.ndarray:
    if sparse.issparse(matrix):
        return np.asarray(matrix.data)
    return np.asarray(matrix).ravel()


def _as_csr(matrix: object) -> sparse.csr_matrix:
    if sparse.issparse(matrix):
        return matrix.tocsr(copy=True)
    return sparse.csr_matrix(np.asarray(matrix))


def _validate_counts(matrix: object, *, integer_tolerance: float) -> dict[str, object]:
    data = _matrix_data(matrix)
    if data.size and not np.isfinite(data).all():
        raise ValueError("Counts layer contains non-finite values")
    minimum = float(data.min()) if data.size else 0.0
    if minimum < 0:
        raise ValueError(f"Counts layer contains negative values (minimum={minimum})")
    max_integer_residual = float(np.max(np.abs(data - np.rint(data)))) if data.size else 0.0
    if max_integer_residual > integer_tolerance:
        raise ValueError(
            "Counts layer is not integer-like; refusing to log-transform a possibly "
            f"already transformed matrix (max residual={max_integer_residual:.6g})"
        )
    return {
        "storage": "sparse" if sparse.issparse(matrix) else "dense",
        "n_nonzero": int(np.count_nonzero(data)),
        "minimum": minimum,
        "maximum": float(data.max()) if data.size else 0.0,
        "max_integer_residual": max_integer_residual,
    }


def _casefold_projection(
    source_names: Sequence[str], selected_names: Sequence[str]
) -> sparse.csr_matrix:
    """Map each selected symbol to exactly one source feature.

    Zebrafish symbols are normally lower-case, while the corrected artifact
    also contains a small number of upper-case features with the same
    case-folded spelling (for example ``tnc`` and ``TNC``).  Summing those
    columns before ``log1p`` is mathematically wrong and no longer reproduces
    the frozen expression matrix.  Prefer an exact spelling match and use a
    case-insensitive fallback only when it is unambiguous.
    """

    source_strings = [str(name).strip() for name in source_names]
    exact_lookup: dict[str, list[int]] = {}
    folded_lookup: dict[str, list[int]] = {}
    for source_idx, name in enumerate(source_strings):
        exact_lookup.setdefault(name, []).append(source_idx)
        folded_lookup.setdefault(name.lower(), []).append(source_idx)

    source_indices: list[int] = []
    target_indices: list[int] = []
    for target_idx, raw_selected in enumerate(selected_names):
        selected = str(raw_selected).strip()
        exact = exact_lookup.get(selected, [])
        if len(exact) == 1:
            source_idx = exact[0]
        elif len(exact) > 1:
            raise ValueError(
                f"Selected gene {selected!r} has duplicate exact source features"
            )
        else:
            folded = folded_lookup.get(selected.lower(), [])
            if len(folded) != 1:
                candidates = [source_strings[idx] for idx in folded]
                raise ValueError(
                    f"Selected gene {selected!r} has {len(folded)} case-insensitive "
                    f"source matches {candidates!r}; exact one-to-one mapping is required"
                )
            source_idx = folded[0]
        source_indices.append(source_idx)
        target_indices.append(target_idx)
    return sparse.csr_matrix(
        (
            np.ones(len(source_indices), dtype=np.float32),
            (np.asarray(source_indices), np.asarray(target_indices)),
        ),
        shape=(len(source_names), len(selected_names)),
    )


def _ordered_stages(
    obs: pd.DataFrame, stage_col: str, time_col: str | None
) -> tuple[list[str], dict[str, float | None]]:
    stages = obs[stage_col].astype(str)
    if stages.isna().any() or (stages == "nan").any():
        raise ValueError(f"obs[{stage_col!r}] contains missing values")
    unique = list(pd.unique(stages))
    if time_col is None:
        return sorted(unique), {stage: None for stage in unique}
    if time_col not in obs:
        raise KeyError(f"obs time column {time_col!r} is missing")
    numeric_time = pd.to_numeric(obs[time_col], errors="raise")
    stage_times = {
        stage: float(np.median(numeric_time.loc[stages == stage].to_numpy(dtype=float)))
        for stage in unique
    }
    order = sorted(unique, key=lambda stage: (stage_times[stage], stage))
    return order, stage_times


def prepare_inputs(
    h5ad_path: Path,
    lr_database_path: Path,
    *,
    preprocess_audit_path: Path | None = None,
    counts_layer: str = "counts",
    label_col: str = "Annotation",
    stage_col: str = "time_point_processed",
    time_col: str | None = "time",
    spatial_key: str = "spatial_aligned",
    target_sum: float | str = "audit",
    integer_tolerance: float = 1e-5,
    source_x_tolerance: float = 1e-10,
) -> PreparedInputs:
    """Build the exact shared method input from the corrected H5AD."""

    source = ad.read_h5ad(h5ad_path)
    for column in [label_col, stage_col]:
        if column not in source.obs:
            raise KeyError(f"obs column {column!r} is missing")
    if counts_layer not in source.layers:
        raise KeyError(
            f"Required raw-count layer {counts_layer!r} is missing; X is never used as a fallback"
        )
    if spatial_key not in source.obsm:
        raise KeyError(f"obsm spatial key {spatial_key!r} is missing")

    counts = source.layers[counts_layer]
    count_diagnostics = _validate_counts(counts, integer_tolerance=integer_tolerance)
    counts_csr = _as_csr(counts).astype(np.float64)
    library_size = np.asarray(counts_csr.sum(axis=1)).ravel()
    if np.any(~np.isfinite(library_size)) or np.any(library_size <= 0):
        bad = int(np.count_nonzero(~np.isfinite(library_size) | (library_size <= 0)))
        raise ValueError(f"Counts layer contains {bad} cells with invalid/zero library size")

    audit_record: dict[str, object] | None = None
    if isinstance(target_sum, str):
        if target_sum.lower() != "audit":
            raise ValueError("target_sum must be a positive number or 'audit'")
        if preprocess_audit_path is None:
            raise ValueError("preprocess_audit_path is required when target_sum='audit'")
        audit_record = json.loads(preprocess_audit_path.read_text(encoding="utf-8"))
        if audit_record.get("all_checks_passed") is not True:
            raise ValueError("Preprocessing audit does not declare all_checks_passed=true")
        try:
            target_sum_value = float(
                audit_record["normalization_and_log1p"]["resolved_target_sum"]
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "Preprocessing audit lacks normalization_and_log1p.resolved_target_sum"
            ) from error
        expected_h5ad_sha = audit_record.get("output_h5ad", {}).get("sha256")
        if not expected_h5ad_sha:
            raise ValueError("Preprocessing audit lacks output_h5ad.sha256")
        observed_h5ad_sha = sha256_file(h5ad_path)
        if observed_h5ad_sha != expected_h5ad_sha:
            raise ValueError("Input H5AD SHA256 does not match the preprocessing audit")
        expected_lr_sha = audit_record.get("inputs", {}).get("lr_database", {}).get("sha256")
        if expected_lr_sha and sha256_file(lr_database_path) != expected_lr_sha:
            raise ValueError("LR database SHA256 does not match the preprocessing audit")
        target_sum_rule = "frozen preprocessing-audit resolved target"
    else:
        target_sum_value = float(target_sum)
        target_sum_rule = "explicit numeric value"
    if not np.isfinite(target_sum_value) or target_sum_value <= 0:
        raise ValueError("Resolved target_sum must be positive and finite")

    database = read_lr_database(lr_database_path)
    filtered_database, feature_audit = filter_lr_database_by_features(database, source.var_names)
    selected_genes = sorted(
        {
            gene
            for row in filtered_database.itertuples(index=False)
            for token in (row.ligand, row.receptor)
            for gene in complex_subunits(token)
        }
    )
    projection = _casefold_projection(source.var_names, selected_genes)
    selected_counts = (counts_csr @ projection).tocsr()
    selected_counts = selected_counts.multiply((target_sum_value / library_size)[:, None]).tocsr()
    selected_counts.data = np.log1p(selected_counts.data)

    # The corrected zebrafish artifact already stores this exact single-log
    # matrix in X.  This check prevents a future runner from accidentally using
    # a 1e4 target, applying log1p twice, or reading a scaled layer.
    source_x = _as_csr(source.X).astype(np.float64)
    source_selected = (source_x @ projection).tocsr()
    residual = (selected_counts - source_selected).tocsr()
    source_x_max_abs_residual = (
        float(np.max(np.abs(residual.data))) if residual.nnz else 0.0
    )
    if source_x_max_abs_residual > source_x_tolerance:
        raise ValueError(
            "counts-derived normalize_total+log1p does not reproduce source X over "
            f"the LR-gene submatrix (max abs residual={source_x_max_abs_residual:.6g}, "
            f"tolerance={source_x_tolerance:.6g}, target_sum={target_sum_value:.6g})"
        )

    obs = source.obs.copy()
    obs["ccc_label"] = obs[label_col].astype(str)
    obs["ccc_stage"] = obs[stage_col].astype(str)
    if obs["ccc_label"].isna().any() or (obs["ccc_label"] == "nan").any():
        raise ValueError(f"obs[{label_col!r}] contains missing labels")
    stage_order, stage_times = _ordered_stages(source.obs, stage_col, time_col)
    obs["ccc_stage_time"] = obs["ccc_stage"].map(stage_times)

    coordinates = np.asarray(source.obsm[spatial_key], dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[0] != source.n_obs or coordinates.shape[1] < 2:
        raise ValueError(f"obsm[{spatial_key!r}] must be an n_cells x >=2 matrix")
    if not np.isfinite(coordinates).all():
        raise ValueError(f"obsm[{spatial_key!r}] contains non-finite values")

    prepared = ad.AnnData(
        X=selected_counts.astype(np.float32),
        obs=obs,
        var=pd.DataFrame(index=pd.Index(selected_genes, name="gene")),
    )
    prepared.obsm["spatial"] = coordinates.copy()
    prepared.obsm["spatial_aligned"] = coordinates.copy()
    interaction_graph = json_ready(source.uns.get("interaction_graph", {}))
    if not isinstance(interaction_graph, dict):
        raise ValueError("uns['interaction_graph'] must be a mapping when present")
    prepared.uns["ccc_preprocessing"] = {
        "source_h5ad": str(h5ad_path.resolve()),
        "source_lr_database": str(lr_database_path.resolve()),
        "preprocess_audit": (
            str(preprocess_audit_path.resolve()) if preprocess_audit_path is not None else None
        ),
        "source_expression": f"layers[{counts_layer!r}]",
        "normalization": "per-cell library-size normalization over all measured genes",
        "target_sum": target_sum_value,
        "target_sum_rule": target_sum_rule,
        "transform": "numpy.log1p exactly once after normalization",
        "source_x_verification": "every selected LR-gene value reconstructed from counts and compared with X",
        "source_x_max_abs_residual": source_x_max_abs_residual,
        "source_x_tolerance": float(source_x_tolerance),
        "gene_selection": "strict structurally available LR subunits after normalization denominator calculation",
        "spatial_source": f"obsm[{spatial_key!r}]",
        "label_source": f"obs[{label_col!r}]",
        "stage_source": f"obs[{stage_col!r}]",
        "interaction_graph": interaction_graph,
    }
    diagnostics: dict[str, object] = {
        "source_n_cells": int(source.n_obs),
        "source_n_genes": int(source.n_vars),
        "selected_n_genes": len(selected_genes),
        "database_rows_input": int(len(database)),
        "database_rows_structurally_available": int(len(filtered_database)),
        "database_unique_flat_rows_available": int(filtered_database["flat_key"].nunique()),
        "database_exact_flat_duplicates_available": int(
            len(filtered_database) - filtered_database["flat_key"].nunique()
        ),
        "count_layer": count_diagnostics,
        "library_size_min": float(library_size.min()),
        "library_size_median": float(np.median(library_size)),
        "library_size_max": float(library_size.max()),
        "resolved_target_sum": target_sum_value,
        "target_sum_rule": target_sum_rule,
        "source_x_max_abs_residual": source_x_max_abs_residual,
        "source_x_tolerance": float(source_x_tolerance),
        "preprocess_audit_all_checks_passed": (
            audit_record.get("all_checks_passed") if audit_record is not None else None
        ),
        "feature_filter_excluded_rows": int((~feature_audit["all_subunits_present"]).sum()),
    }
    return PreparedInputs(prepared, filtered_database, stage_order, stage_times, diagnostics)


def stratified_subsample_indices(
    labels: Sequence[str], max_cells: int, seed: int
) -> np.ndarray:
    """Return deterministic label-stratified indices, or every index for max=0."""

    labels_array = np.asarray(labels, dtype=str)
    n_cells = len(labels_array)
    if max_cells <= 0 or n_cells <= max_cells:
        return np.arange(n_cells, dtype=int)
    if max_cells < 1:
        raise ValueError("max_cells must be zero (all cells) or positive")
    rng = np.random.default_rng(seed)
    groups, counts = np.unique(labels_array, return_counts=True)
    if max_cells < len(groups):
        return np.sort(rng.choice(n_cells, size=max_cells, replace=False))

    ideal = max_cells * counts.astype(float) / n_cells
    quota = np.maximum(1, np.floor(ideal).astype(int))
    quota = np.minimum(quota, counts)
    while int(quota.sum()) > max_cells:
        candidates = np.where(quota > 1)[0]
        drop = candidates[np.argmax(quota[candidates] - ideal[candidates])]
        quota[drop] -= 1
    while int(quota.sum()) < max_cells:
        candidates = np.where(quota < counts)[0]
        add = candidates[np.argmax(ideal[candidates] - quota[candidates])]
        quota[add] += 1

    selected: list[int] = []
    for group, amount in zip(groups, quota, strict=True):
        candidates = np.flatnonzero(labels_array == group)
        selected.extend(rng.choice(candidates, size=int(amount), replace=False).tolist())
    return np.sort(np.asarray(selected, dtype=int))


def safe_stage_token(position: int, stage: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", str(stage)).strip("._") or "stage"
    return f"{position:03d}_{slug}"


def write_input_bundle(
    prepared: PreparedInputs,
    out_dir: Path,
    *,
    source_h5ad: Path,
    source_lr_database: Path,
    source_preprocess_audit: Path,
    max_cells_per_stage: int = 0,
    subsample_seed: int = 20260722,
) -> dict[str, object]:
    """Write method-neutral MatrixMarket stage inputs and a bound manifest."""

    out_dir.mkdir(parents=True, exist_ok=True)
    nonempty = [path for path in out_dir.iterdir()]
    if nonempty:
        raise FileExistsError(f"Output directory is not empty: {out_dir}")

    database_path = out_dir / "filtered_lr_database.csv"
    prepared.lr_database.to_csv(database_path, index=False)
    availability_path = out_dir / "normalized_lr_expression.h5ad"
    prepared.adata.write_h5ad(availability_path, compression="gzip")

    stages: list[dict[str, object]] = []
    artifact_paths: list[Path] = [database_path, availability_path]
    for position, stage in enumerate(prepared.stage_order):
        token = safe_stage_token(position, stage)
        stage_dir = out_dir / "stages" / token
        stage_dir.mkdir(parents=True, exist_ok=False)
        stage_mask = prepared.adata.obs["ccc_stage"].astype(str).to_numpy() == stage
        global_indices = np.flatnonzero(stage_mask)
        local_indices = stratified_subsample_indices(
            prepared.adata.obs.iloc[global_indices]["ccc_label"].astype(str).to_numpy(),
            max_cells_per_stage,
            subsample_seed + position,
        )
        selected_global = global_indices[local_indices]
        snapshot = prepared.adata[selected_global].copy()
        cell_ids = snapshot.obs_names.astype(str).tolist()
        if len(set(cell_ids)) != len(cell_ids):
            raise ValueError("Cell IDs must be unique")

        matrix_path = stage_dir / "expression_genes_by_cells.mtx"
        genes_path = stage_dir / "genes.txt"
        metadata_path = stage_dir / "metadata.csv"
        spatial_path = stage_dir / "spatial_aligned.csv"
        mmwrite(matrix_path, snapshot.X.T.tocoo())
        genes_path.write_text("\n".join(snapshot.var_names.astype(str)) + "\n", encoding="utf-8")
        metadata = pd.DataFrame(
            {
                "cell_id": cell_ids,
                "label": snapshot.obs["ccc_label"].astype(str).to_numpy(),
                "stage": stage,
                "stage_time": prepared.stage_times[stage],
                "source_obs_index": cell_ids,
            }
        )
        metadata.to_csv(metadata_path, index=False)
        coords = np.asarray(snapshot.obsm["spatial_aligned"], dtype=float)
        spatial = pd.DataFrame(coords, columns=[f"coord_{idx}" for idx in range(coords.shape[1])])
        spatial.insert(0, "cell_id", cell_ids)
        spatial.to_csv(spatial_path, index=False)
        stage_files = [matrix_path, genes_path, metadata_path, spatial_path]
        artifact_paths.extend(stage_files)
        stages.append(
            {
                "stage": stage,
                "stage_time": prepared.stage_times[stage],
                "token": token,
                "n_cells_before_subsampling": int(stage_mask.sum()),
                "n_cells": int(snapshot.n_obs),
                "n_cell_types": int(snapshot.obs["ccc_label"].nunique()),
                "cell_type_counts": {
                    str(key): int(value)
                    for key, value in snapshot.obs["ccc_label"].value_counts().sort_index().items()
                },
                "files": {path.name: file_record(path) for path in stage_files},
            }
        )

    manifest: dict[str, object] = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "purpose": "shared corrected-zebrafish input for external CCC methods",
        "sources": {
            "h5ad": file_record(source_h5ad),
            "lr_database": file_record(source_lr_database),
            "preprocess_audit": file_record(source_preprocess_audit),
        },
        "preprocessing": prepared.adata.uns["ccc_preprocessing"],
        "diagnostics": prepared.diagnostics,
        "selection": {
            "max_cells_per_stage": int(max_cells_per_stage),
            "subsampling": "all cells" if max_cells_per_stage <= 0 else "deterministic label-stratified",
            "seed": int(subsample_seed),
        },
        "database": {
            "path": str(database_path.resolve()),
            "sha256": sha256_file(database_path),
            "rows": int(len(prepared.lr_database)),
            "unique_flat_rows": int(prepared.lr_database["flat_key"].nunique()),
            "filter": "all underscore-delimited ligand and receptor subunits present in H5AD var_names; no stage prevalence filter",
        },
        "stages": stages,
        "software": software_versions(),
        "artifacts": {str(path.relative_to(out_dir)): file_record(path) for path in artifact_paths},
    }
    json_dump(manifest, out_dir / "input_manifest.json")
    return manifest


def ensure_common_score_schema(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in COMMON_SCORE_COLUMNS if column not in frame]
    if missing:
        raise ValueError(f"Common score table is missing columns: {missing}")
    prefix = COMMON_SCORE_COLUMNS
    suffix = [column for column in frame.columns if column not in prefix]
    return frame.loc[:, prefix + suffix]
