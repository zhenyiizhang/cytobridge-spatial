"""Prepare GSE149457 chicken-heart inputs for CytoBridge workflows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from scipy.spatial.distance import pdist

from ..graph_database import match_graph_database_features
from .chicken_heart import (
    CHICKEN_HEART_TIMEPOINTS,
    _remove_legacy_chicken_heart_validation_metadata,
    apply_chicken_heart_coordinate_validation,
    chicken_heart_anatomical_orientation_qc,
)
from .spatial_align import AlignConfig, preprocess_fixed_spatial


INPUT_SCHEMA_VERSION = 4
OT_INPUT_SCHEMA_VERSION = 2
TIMEPOINTS = CHICKEN_HEART_TIMEPOINTS
TIME_MAPPING = {"D4": 0.0, "D7": 1.0, "D10": 2.0, "D14": 3.0}
EXPECTED_COUNTS = {"D4": 147, "D7": 528, "D10": 908, "D14": 1967}
RAW_FILENAMES = {
    "D4": "GSM4502482_chicken_heart_spatial_RNAseq_D4_filtered_feature_bc_matrix.h5",
    "D7": "GSM4502483_chicken_heart_spatial_RNAseq_D7_filtered_feature_bc_matrix.h5",
    "D10": "GSM4502484_chicken_heart_spatial_RNAseq_D10_filtered_feature_bc_matrix.h5",
    "D14": "GSM4502485_chicken_heart_spatial_RNAseq_D14_filtered_feature_bc_matrix.h5",
}
REQUIRED_METADATA = ("timepoint", "region", "celltype_prediction")


def _timepoint_mask(adata: ad.AnnData, timepoint: str) -> np.ndarray:
    return adata.obs["timepoint"].astype(str).to_numpy() == timepoint


def _validate_reference_input(
    metadata: ad.AnnData,
    aligned: ad.AnnData,
) -> dict[str, Any]:
    for name, current in (("Metadata", metadata), ("Alignment", aligned)):
        missing = [column for column in REQUIRED_METADATA if column not in current.obs]
        if missing:
            raise ValueError(f"{name} H5AD lacks required obs columns: {missing}.")
    if not metadata.obs_names.is_unique or not aligned.obs_names.is_unique:
        raise ValueError("Metadata and alignment observation names must be unique.")
    if not aligned.obs_names.isin(metadata.obs_names).all():
        raise ValueError("Alignment observations are not an exact metadata subset.")
    if "spatial_aligned" not in aligned.obsm or "spatial" not in metadata.obsm:
        raise ValueError(
            "Expected aligned.obsm['spatial_aligned'] and metadata.obsm['spatial']."
        )
    coordinates = np.asarray(aligned.obsm["spatial_aligned"], dtype=np.float64)
    if coordinates.shape != (aligned.n_obs, 2) or not np.isfinite(coordinates).all():
        raise ValueError("spatial_aligned coordinates must be finite Nx2.")
    expected_total = sum(EXPECTED_COUNTS.values())
    if aligned.n_obs != expected_total:
        raise ValueError(
            f"Expected {expected_total} reference spots, found {aligned.n_obs}."
        )

    ordered_times = aligned.obs["timepoint"].astype(str).tolist()
    expected_order = [
        timepoint for timepoint in TIMEPOINTS for _ in range(EXPECTED_COUNTS[timepoint])
    ]
    if ordered_times != expected_order:
        raise ValueError(
            "Reference observations must be grouped in D4/D7/D10/D14 order."
        )
    for timepoint, expected in EXPECTED_COUNTS.items():
        observed = int(np.count_nonzero(_timepoint_mask(aligned, timepoint)))
        if observed != expected:
            raise ValueError(
                f"Expected {expected} reference {timepoint} spots, found {observed}."
            )

    metadata_subset = metadata.obs.loc[aligned.obs_names, REQUIRED_METADATA]
    aligned_subset = aligned.obs.loc[:, REQUIRED_METADATA]
    for column in REQUIRED_METADATA:
        if not np.array_equal(
            metadata_subset[column].astype(str).to_numpy(),
            aligned_subset[column].astype(str).to_numpy(),
        ):
            raise ValueError(f"Reference metadata column {column!r} does not match.")
    return {
        "n_obs": int(aligned.n_obs),
        "timepoint_counts": dict(EXPECTED_COUNTS),
        "coordinate_shape": [int(value) for value in coordinates.shape],
        "coordinate_source": "aligned_reference",
    }


def _read_raw_counts(raw_dir: Path) -> dict[str, ad.AnnData]:
    result: dict[str, ad.AnnData] = {}
    expected_vars: pd.Index | None = None
    for timepoint in TIMEPOINTS:
        path = raw_dir / RAW_FILENAMES[timepoint]
        if not path.is_file():
            raise FileNotFoundError(path)
        current = sc.read_10x_h5(path)
        current.var_names_make_unique()
        if not current.obs_names.is_unique or not current.var_names.is_unique:
            raise ValueError(f"Raw {timepoint} matrix has non-unique identifiers.")
        if expected_vars is None:
            expected_vars = current.var_names.copy()
        elif not np.array_equal(current.var_names, expected_vars):
            raise ValueError("Raw 10x matrices do not share exact feature order.")
        values = np.asarray(current.X.data if sparse.issparse(current.X) else current.X)
        if (
            not np.isfinite(values).all()
            or np.any(values < 0)
            or not np.allclose(values, np.rint(values), atol=1e-6)
        ):
            raise ValueError(
                f"Raw {timepoint} matrix is not nonnegative integer counts."
            )
        result[timepoint] = current
    return result


def assemble_chicken_heart_reference_counts(
    raw_by_timepoint: Mapping[str, ad.AnnData],
    metadata: ad.AnnData,
    aligned: ad.AnnData,
) -> ad.AnnData:
    """Assemble raw counts in the reference alignment row order."""

    _validate_reference_input(metadata, aligned)
    matrices = []
    reference_vars: pd.Index | None = None
    for timepoint in TIMEPOINTS:
        raw = raw_by_timepoint[timepoint]
        if reference_vars is None:
            reference_vars = raw.var_names.copy()
        elif not np.array_equal(raw.var_names, reference_vars):
            raise ValueError("Raw count feature order differs across timepoints.")
        names = aligned.obs_names[_timepoint_mask(aligned, timepoint)].astype(str)
        suffix = f"_{timepoint}"
        if any(not name.endswith(suffix) for name in names):
            raise ValueError(f"Reference {timepoint} observation suffix is invalid.")
        barcodes = pd.Index([name[: -len(suffix)] for name in names])
        if not barcodes.isin(raw.obs_names).all():
            missing = barcodes[~barcodes.isin(raw.obs_names)].tolist()[:5]
            raise ValueError(
                f"Raw {timepoint} matrix lacks reference barcodes: {missing}."
            )
        matrices.append(raw[barcodes].X.copy())

    counts = sparse.vstack(matrices, format="csr")
    assembled = ad.AnnData(
        X=counts.copy(),
        obs=metadata.obs.loc[aligned.obs_names].copy(),
        var=raw_by_timepoint[TIMEPOINTS[0]].var.copy(),
    )
    assembled.obs_names = aligned.obs_names.copy()
    assembled.layers["counts"] = counts
    assembled.obs["Annotation"] = (
        assembled.obs["celltype_prediction"].astype(str).to_numpy()
    )
    assembled.obsm["spatial_original"] = np.asarray(
        metadata[aligned.obs_names].obsm["spatial"], dtype=np.float64
    )
    assembled.obsm["spatial_aligned"] = np.asarray(
        aligned.obsm["spatial_aligned"], dtype=np.float64
    ).copy()
    assembled.obsm["spatial"] = assembled.obsm["spatial_aligned"].copy()
    return assembled


def _source_alignment_qc(adata: ad.AnnData) -> dict[str, Any]:
    records: dict[str, Any] = {}
    for timepoint in TIMEPOINTS:
        mask = _timepoint_mask(adata, timepoint)
        raw = np.asarray(adata.obsm["spatial_original"])[mask]
        fixed = np.asarray(adata.obsm["spatial_aligned"])[mask]
        design = np.column_stack((raw, np.ones(raw.shape[0])))
        coefficients, _, _, _ = np.linalg.lstsq(design, fixed, rcond=None)
        fitted = design @ coefficients
        determinant = float(np.linalg.det(coefficients[:2, :].T))
        records[timepoint] = {
            "n_spots": int(mask.sum()),
            "raw_to_reference_affine_determinant": determinant,
            "raw_to_reference_affine_orientation": (
                "reflected" if determinant < 0 else "preserved"
            ),
            "affine_rmse": float(np.sqrt(np.mean(np.square(fitted - fixed)))),
            "reference_bounds": {
                "min": fixed.min(axis=0).astype(float).tolist(),
                "max": fixed.max(axis=0).astype(float).tolist(),
            },
        }
    return records


def _clean_feature_coverage(coverage: dict[str, Any] | str) -> dict[str, Any] | str:
    if not isinstance(coverage, dict):
        return coverage
    result = dict(coverage)
    result.pop("database_path", None)
    return result


def prepare_chicken_heart_input(
    *,
    raw_dir: str | Path,
    metadata_h5ad: str | Path,
    aligned_reference_h5ad: str | Path,
    output_h5ad: str | Path,
    output_table: str | Path,
    manifest_path: str | Path,
    graph_database: str | Path | None = None,
    repair_legacy_d7_left_right: bool = False,
) -> dict[str, Any]:
    """Prepare counts and reference coordinates for the chicken-heart workflow."""

    raw_dir = Path(raw_dir).expanduser().resolve()
    metadata_h5ad = Path(metadata_h5ad).expanduser().resolve()
    aligned_reference_h5ad = Path(aligned_reference_h5ad).expanduser().resolve()
    output_h5ad = Path(output_h5ad).expanduser().resolve()
    output_table = Path(output_table).expanduser().resolve()
    manifest_path = Path(manifest_path).expanduser().resolve()
    graph_database_path = (
        None if graph_database is None else Path(graph_database).expanduser().resolve()
    )
    existing = [
        path.name
        for path in (output_h5ad, output_table, manifest_path)
        if path.exists()
    ]
    if existing:
        raise FileExistsError(f"Output files already exist: {existing}")

    metadata = ad.read_h5ad(metadata_h5ad)
    aligned = ad.read_h5ad(aligned_reference_h5ad)
    reference = _validate_reference_input(metadata, aligned)
    assembled = assemble_chicken_heart_reference_counts(
        _read_raw_counts(raw_dir), metadata, aligned
    )
    coordinate_adjustment = apply_chicken_heart_coordinate_validation(
        assembled,
        repair_legacy_d7_left_right=repair_legacy_d7_left_right,
    )

    required_features: tuple[str, ...] = ()
    coverage: dict[str, Any] | str = "not_requested"
    if graph_database_path is not None:
        required_features, coverage = match_graph_database_features(
            graph_database_path,
            assembled.var_names,
            preferred_species_tag="chicken",
        )
        coverage = _clean_feature_coverage(coverage)
        if not required_features:
            raise ValueError(
                "Graph database has no exact chicken-heart feature matches."
            )

    processed, table = preprocess_fixed_spatial(
        assembled,
        time_key="timepoint",
        spatial_key="spatial_aligned",
        cfg=AlignConfig(
            n_top_genes=2000,
            n_pcs=50,
            normalization_target_sum=None,
            spatial_dim=2,
            expression_layer="counts",
            counts_layer="counts",
            raw_count_validation="strict",
            time_mapping=TIME_MAPPING,
            hvg_batch_key="timepoint",
            required_latent_features=required_features,
        ),
    )
    if processed.obs_names.tolist() != aligned.obs_names.tolist():
        raise RuntimeError("Processed rows differ from the reference alignment.")
    if not np.array_equal(
        np.asarray(processed.obsm["spatial_aligned"]),
        np.asarray(assembled.obsm["spatial_aligned"]),
    ):
        raise RuntimeError("Processed coordinates differ from the prepared input.")

    anatomy = chicken_heart_anatomical_orientation_qc(processed)
    validation = {
        "schema_version": INPUT_SCHEMA_VERSION,
        "reference": reference,
        "downstream_annotation": {
            "key": "celltype_prediction",
            "compatibility_key": "Annotation",
            "source": metadata_h5ad.name,
            "n_classes": int(
                processed.obs["celltype_prediction"].astype(str).nunique()
            ),
        },
        "coordinate_adjustment": coordinate_adjustment,
        "anatomical_orientation": anatomy,
        "source_alignment": _source_alignment_qc(processed),
        "graph_feature_coverage": coverage,
    }
    _remove_legacy_chicken_heart_validation_metadata(processed)
    processed.uns["chicken_heart_input_validation_json"] = json.dumps(
        validation, sort_keys=True, separators=(",", ":")
    )

    output_h5ad.parent.mkdir(parents=True, exist_ok=True)
    output_table.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    processed.write_h5ad(output_h5ad)
    table.to_csv(output_table, index=False)

    manifest = {
        "schema_version": INPUT_SCHEMA_VERSION,
        "dataset": "chicken_heart",
        "source_accession": "GSE149457",
        "inputs": {
            "metadata_h5ad": metadata_h5ad.name,
            "aligned_reference_h5ad": aligned_reference_h5ad.name,
            "raw_10x": dict(RAW_FILENAMES),
            "graph_database": (
                graph_database_path.name if graph_database_path is not None else None
            ),
        },
        "reference": reference,
        "coordinate_adjustment": coordinate_adjustment,
        "anatomical_orientation": anatomy,
        "source_alignment": _source_alignment_qc(processed),
        "preprocessing": {
            "implementation": "CytoBridge.pp.preprocess_fixed_spatial",
            "time_mapping": TIME_MAPPING,
            "n_top_genes": 2000,
            "n_pcs": 50,
            "normalization_target_sum": "median",
            "raw_count_validation": "strict",
            "graph_feature_coverage": coverage,
        },
        "outputs": {
            "aligned_h5ad": output_h5ad.name,
            "model_input_table": output_table.name,
            "n_obs": int(processed.n_obs),
            "n_vars": int(processed.n_vars),
            "model_input_shape": [int(value) for value in table.shape],
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _validate_ot_source(adata: ad.AnnData) -> dict[str, Any]:
    missing_obs = [key for key in REQUIRED_METADATA if key not in adata.obs]
    if missing_obs:
        raise ValueError(f"Chicken-heart input lacks obs columns: {missing_obs}.")
    if "counts" not in adata.layers:
        raise ValueError("Chicken-heart input lacks layers['counts'].")
    if "spatial_original" not in adata.obsm:
        raise ValueError("Chicken-heart input lacks obsm['spatial_original'].")
    if not adata.obs_names.is_unique:
        raise ValueError("Chicken-heart spot identifiers must be unique.")

    times = adata.obs["timepoint"].astype(str).to_numpy()
    observed = {stage: int(np.count_nonzero(times == stage)) for stage in TIMEPOINTS}
    if observed != EXPECTED_COUNTS:
        raise ValueError(
            f"Chicken-heart stage counts are {observed}, expected {EXPECTED_COUNTS}."
        )
    expected_order = [
        stage for stage in TIMEPOINTS for _ in range(EXPECTED_COUNTS[stage])
    ]
    if times.tolist() != expected_order:
        raise ValueError("Chicken-heart rows must be grouped D4/D7/D10/D14.")

    raw = np.asarray(adata.obsm["spatial_original"], dtype=np.float64)
    if raw.shape != (adata.n_obs, 2) or not np.isfinite(raw).all():
        raise ValueError("spatial_original must be a finite n_obs x 2 matrix.")
    values = (
        np.asarray(adata.layers["counts"].data)
        if sparse.issparse(adata.layers["counts"])
        else np.asarray(adata.layers["counts"])
    )
    if (
        not np.isfinite(values).all()
        or np.any(values < 0)
        or not np.allclose(values, np.rint(values), atol=1e-6)
    ):
        raise ValueError("layers['counts'] must contain nonnegative integer counts.")
    for key in ("region", "celltype_prediction"):
        labels = adata.obs[key]
        if labels.isna().any() or labels.astype(str).str.strip().eq("").any():
            raise ValueError(f"obs[{key!r}] contains missing labels.")
    return {
        "n_obs": int(adata.n_obs),
        "n_vars": int(adata.n_vars),
        "timepoint_counts": observed,
        "raw_coordinate_shape": [int(value) for value in raw.shape],
    }


def prepare_chicken_heart_ot_adata(
    adata: ad.AnnData,
) -> tuple[ad.AnnData, dict[str, Any]]:
    """Return raw counts with the workflow's deterministic OT coordinates."""

    input_summary = _validate_ot_source(adata)
    prepared = adata.copy()
    raw = np.asarray(prepared.obsm["spatial_original"], dtype=np.float64).copy()
    times = prepared.obs["timepoint"].astype(str).to_numpy()
    d7_mask = times == "D7"
    d7_before = raw[d7_mask].copy()
    d7_center = d7_before.mean(axis=0)
    ot_input = raw.copy()
    ot_input[d7_mask] = (2.0 * d7_center) - d7_before

    if "spatial_reference" not in prepared.obsm and "spatial_aligned" in prepared.obsm:
        prepared.obsm["spatial_reference"] = np.asarray(
            prepared.obsm["spatial_aligned"], dtype=np.float64
        ).copy()
    prepared.obsm["spatial_ot_input"] = ot_input
    prepared.obsm["spatial"] = ot_input.copy()
    prepared.obsm.pop("spatial_aligned", None)
    prepared.obsm.pop("X_latent", None)
    prepared.obsm.pop("X_pca", None)
    prepared.varm.pop("PCs", None)
    prepared.X = prepared.layers["counts"].copy()
    prepared.obs["Annotation"] = (
        prepared.obs["celltype_prediction"].astype(str).to_numpy()
    )
    for key in (
        "highly_variable",
        "highly_variable_nbatches",
        "highly_variable_intersection",
        "means",
        "dispersions",
        "dispersions_norm",
        "pca_center",
    ):
        if key in prepared.var:
            del prepared.var[key]
    for key in (
        "hvg",
        "log1p",
        "pca",
        "pca_center_info",
        "preprocess_info",
        "spatial_alignment_info",
        "interaction_graph",
        "chicken_heart_input_validation_json",
    ):
        prepared.uns.pop(key, None)
    _remove_legacy_chicken_heart_validation_metadata(prepared)

    correction = {
        "timepoint": "D7",
        "operation": "xy_prime=2*stage_centroid-xy",
        "center": d7_center.astype(float).tolist(),
        "linear_matrix": [[-1.0, 0.0], [0.0, -1.0]],
        "translation": (2.0 * d7_center).astype(float).tolist(),
        "max_pairwise_distance_error": float(
            np.max(np.abs(pdist(d7_before) - pdist(ot_input[d7_mask])))
        ),
        "outside_d7_exactly_unchanged": bool(
            np.array_equal(raw[~d7_mask], ot_input[~d7_mask])
        ),
    }
    validation = {
        "schema_version": OT_INPUT_SCHEMA_VERSION,
        "coordinate_policy": "raw_coordinates_with_predefined_d7_180_rotation",
        "input": input_summary,
        "d7_orientation_correction": correction,
        "downstream_annotation": {
            "key": "celltype_prediction",
            "compatibility_key": "Annotation",
            "n_classes": int(prepared.obs["celltype_prediction"].astype(str).nunique()),
        },
    }
    prepared.uns["chicken_heart_ot_input_validation_json"] = json.dumps(
        validation, sort_keys=True, separators=(",", ":")
    )
    return prepared, validation


def validate_chicken_heart_ot_input(adata: ad.AnnData) -> dict[str, Any]:
    """Validate a chicken-heart input before package OT alignment."""

    input_summary = _validate_ot_source(adata)
    raw_validation = adata.uns.get("chicken_heart_ot_input_validation_json")
    if not isinstance(raw_validation, str):
        raise ValueError("Chicken-heart OT input lacks validation metadata.")
    try:
        validation = json.loads(raw_validation)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Chicken-heart OT input validation is not valid JSON."
        ) from exc
    if validation.get("schema_version") != OT_INPUT_SCHEMA_VERSION:
        raise ValueError(
            f"Chicken-heart OT input requires schema_version {OT_INPUT_SCHEMA_VERSION}."
        )
    if "spatial_ot_input" not in adata.obsm:
        raise ValueError("Chicken-heart OT input lacks obsm['spatial_ot_input'].")
    coordinates = np.asarray(adata.obsm["spatial_ot_input"], dtype=np.float64)
    if coordinates.shape != (adata.n_obs, 2) or not np.isfinite(coordinates).all():
        raise ValueError("spatial_ot_input must be a finite n_obs x 2 matrix.")
    if "spatial" not in adata.obsm or not np.array_equal(
        np.asarray(adata.obsm["spatial"], dtype=np.float64), coordinates
    ):
        raise ValueError("obsm['spatial'] must match obsm['spatial_ot_input'].")
    annotation = validation.get("downstream_annotation", {})
    if annotation.get("key") != "celltype_prediction":
        raise ValueError(
            "Chicken-heart OT input must use celltype_prediction downstream."
        )
    if annotation.get("n_classes") != int(
        adata.obs["celltype_prediction"].astype(str).nunique()
    ):
        raise ValueError("Chicken-heart cell-type class count does not match metadata.")
    return {
        "schema_version": OT_INPUT_SCHEMA_VERSION,
        "coordinate_policy": validation.get("coordinate_policy"),
        "n_obs": input_summary["n_obs"],
        "n_vars": input_summary["n_vars"],
        "timepoint_counts": input_summary["timepoint_counts"],
        "downstream_annotation_key": "celltype_prediction",
    }


def prepare_chicken_heart_ot_input(
    *,
    input_h5ad: str | Path,
    output_h5ad: str | Path,
    output_table: str | Path,
    manifest_path: str | Path,
) -> dict[str, Any]:
    """Write the raw-coordinate input used by the chicken-heart workflow."""

    input_h5ad = Path(input_h5ad).expanduser().resolve()
    output_h5ad = Path(output_h5ad).expanduser().resolve()
    output_table = Path(output_table).expanduser().resolve()
    manifest_path = Path(manifest_path).expanduser().resolve()
    existing = [
        path.name
        for path in (output_h5ad, output_table, manifest_path)
        if path.exists()
    ]
    if existing:
        raise FileExistsError(f"Output files already exist: {existing}")

    prepared, validation = prepare_chicken_heart_ot_adata(ad.read_h5ad(input_h5ad))
    output_h5ad.parent.mkdir(parents=True, exist_ok=True)
    output_table.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    prepared.write_h5ad(output_h5ad)
    coordinates = np.asarray(prepared.obsm["spatial_ot_input"], dtype=np.float64)
    table = pd.DataFrame(
        {
            "spot_id": prepared.obs_names.astype(str),
            "timepoint": prepared.obs["timepoint"].astype(str).to_numpy(),
            "x": coordinates[:, 0],
            "y": coordinates[:, 1],
        }
    )
    table.to_csv(output_table, index=False)

    manifest = {
        "schema_version": OT_INPUT_SCHEMA_VERSION,
        "dataset": "chicken_heart",
        "purpose": "raw_coordinate_input_for_package_ot_alignment",
        "input": input_h5ad.name,
        "validation": validation,
        "outputs": {
            "h5ad": output_h5ad.name,
            "table": output_table.name,
            "n_obs": int(prepared.n_obs),
            "n_vars": int(prepared.n_vars),
            "coordinate_table_shape": [int(value) for value in table.shape],
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


__all__ = [
    "EXPECTED_COUNTS",
    "RAW_FILENAMES",
    "TIME_MAPPING",
    "TIMEPOINTS",
    "assemble_chicken_heart_reference_counts",
    "prepare_chicken_heart_input",
    "prepare_chicken_heart_ot_adata",
    "prepare_chicken_heart_ot_input",
    "validate_chicken_heart_ot_input",
]
