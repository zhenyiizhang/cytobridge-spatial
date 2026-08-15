#!/usr/bin/env python3
"""Build the fixed-alignment chicken-heart input used by CytoBridge.

The public GSE149457 Visium slides contain several tissue sections per capture
area and are not a drop-in input for generic spatial registration.  This adapter
recovers raw 10x counts, selects the exact spots in a reviewed aligned H5AD,
copies those reviewed coordinates without modification, and then applies the
package's standard expression preprocessing and PCA contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

from CytoBridge.graph_database import match_graph_database_features
from CytoBridge.pp import (
    AlignConfig,
    ChickenHeartContractError,
    apply_chicken_heart_coordinate_contract,
    chicken_heart_anatomical_orientation_qc,
    preprocess_fixed_spatial,
)


SCHEMA_VERSION = 3
TIMEPOINTS = ("D4", "D7", "D10", "D14")
TIME_MAPPING = {"D4": 0.0, "D7": 1.0, "D10": 2.0, "D14": 3.0}
EXPECTED_COUNTS = {"D4": 147, "D7": 528, "D10": 908, "D14": 1967}
RAW_FILENAMES = {
    "D4": "GSM4502482_chicken_heart_spatial_RNAseq_D4_filtered_feature_bc_matrix.h5",
    "D7": "GSM4502483_chicken_heart_spatial_RNAseq_D7_filtered_feature_bc_matrix.h5",
    "D10": "GSM4502484_chicken_heart_spatial_RNAseq_D10_filtered_feature_bc_matrix.h5",
    "D14": "GSM4502485_chicken_heart_spatial_RNAseq_D14_filtered_feature_bc_matrix.h5",
}
REQUIRED_METADATA = ("timepoint", "region", "celltype_prediction")


ContractError = ChickenHeartContractError
_anatomical_orientation_qc = chicken_heart_anatomical_orientation_qc
_apply_anatomical_coordinate_contract = apply_chicken_heart_coordinate_contract


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "path": str(resolved),
        "size": int(resolved.stat().st_size),
        "sha256": _sha256(resolved),
    }


def _array_sha256(values: np.ndarray, *, dtype: str) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype=dtype))
    digest = hashlib.sha256()
    digest.update(str(tuple(int(value) for value in array.shape)).encode("ascii"))
    digest.update(f"|{array.dtype.str}|".encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _text_sha256(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def _timepoint_mask(adata: ad.AnnData, timepoint: str) -> np.ndarray:
    return adata.obs["timepoint"].astype(str).to_numpy() == timepoint


def _validate_reference(metadata: ad.AnnData, aligned: ad.AnnData) -> dict[str, Any]:
    missing = [column for column in REQUIRED_METADATA if column not in metadata.obs]
    if missing:
        raise ContractError(f"Metadata H5AD lacks required obs columns: {missing}.")
    missing = [column for column in REQUIRED_METADATA if column not in aligned.obs]
    if missing:
        raise ContractError(f"Aligned H5AD lacks required obs columns: {missing}.")
    if not metadata.obs_names.is_unique or not aligned.obs_names.is_unique:
        raise ContractError("Metadata and aligned observation names must be unique.")
    if not aligned.obs_names.isin(metadata.obs_names).all():
        raise ContractError("Aligned observations are not an exact metadata subset.")
    if "spatial_aligned" not in aligned.obsm or "spatial" not in metadata.obsm:
        raise ContractError(
            "Expected aligned.obsm['spatial_aligned'] and metadata.obsm['spatial']."
        )
    coordinates = np.asarray(aligned.obsm["spatial_aligned"], dtype=np.float64)
    if coordinates.shape != (aligned.n_obs, 2) or not np.isfinite(coordinates).all():
        raise ContractError("Reviewed spatial_aligned coordinates must be finite Nx2.")
    if aligned.n_obs != sum(EXPECTED_COUNTS.values()):
        raise ContractError(
            f"Expected {sum(EXPECTED_COUNTS.values())} reviewed spots, "
            f"found {aligned.n_obs}."
        )

    ordered_times = aligned.obs["timepoint"].astype(str).tolist()
    expected_order = [
        timepoint for timepoint in TIMEPOINTS for _ in range(EXPECTED_COUNTS[timepoint])
    ]
    if ordered_times != expected_order:
        raise ContractError(
            "Reviewed observations must be grouped in canonical D4/D7/D10/D14 order."
        )
    for timepoint, expected in EXPECTED_COUNTS.items():
        observed = int(np.count_nonzero(_timepoint_mask(aligned, timepoint)))
        if observed != expected:
            raise ContractError(
                f"Expected {expected} reviewed {timepoint} spots, found {observed}."
            )

    metadata_subset = metadata.obs.loc[aligned.obs_names, REQUIRED_METADATA]
    aligned_subset = aligned.obs.loc[:, REQUIRED_METADATA]
    for column in REQUIRED_METADATA:
        left = metadata_subset[column].astype(str).to_numpy()
        right = aligned_subset[column].astype(str).to_numpy()
        if not np.array_equal(left, right):
            raise ContractError(
                f"Reviewed reference changed metadata column {column!r}."
            )
    return {
        "n_obs": int(aligned.n_obs),
        "timepoint_counts": dict(EXPECTED_COUNTS),
        "observation_order_sha256": _text_sha256(
            aligned.obs_names.astype(str).tolist()
        ),
        "coordinate_sha256": _array_sha256(coordinates, dtype="<f8"),
        "coordinate_source": "reviewed_aligned_reference_exact_copy",
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
            raise ContractError(f"Raw {timepoint} matrix has non-unique identifiers.")
        if expected_vars is None:
            expected_vars = current.var_names.copy()
        elif not np.array_equal(current.var_names, expected_vars):
            raise ContractError("Raw 10x matrices do not share exact feature order.")
        values = np.asarray(current.X.data if sparse.issparse(current.X) else current.X)
        if (
            not np.isfinite(values).all()
            or np.any(values < 0)
            or not np.allclose(values, np.rint(values), atol=1e-6)
        ):
            raise ContractError(
                f"Raw {timepoint} matrix is not nonnegative integer counts."
            )
        result[timepoint] = current
    return result


def assemble_reviewed_counts(
    raw_by_timepoint: Mapping[str, ad.AnnData],
    metadata: ad.AnnData,
    aligned: ad.AnnData,
) -> ad.AnnData:
    """Assemble raw counts in the exact reviewed-reference row order."""

    _validate_reference(metadata, aligned)
    matrices = []
    reference_vars: pd.Index | None = None
    for timepoint in TIMEPOINTS:
        raw = raw_by_timepoint[timepoint]
        if reference_vars is None:
            reference_vars = raw.var_names.copy()
        elif not np.array_equal(raw.var_names, reference_vars):
            raise ContractError("Raw count feature order differs across timepoints.")
        names = aligned.obs_names[_timepoint_mask(aligned, timepoint)].astype(str)
        suffix = f"_{timepoint}"
        if any(not name.endswith(suffix) for name in names):
            raise ContractError(f"Reviewed {timepoint} observation suffix is invalid.")
        barcodes = pd.Index([name[: -len(suffix)] for name in names])
        if not barcodes.isin(raw.obs_names).all():
            missing = barcodes[~barcodes.isin(raw.obs_names)].tolist()[:5]
            raise ContractError(
                f"Raw {timepoint} matrix lacks reviewed barcodes: {missing}."
            )
        matrices.append(raw[barcodes].X.copy())

    counts = sparse.vstack(matrices, format="csr")
    obs = metadata.obs.loc[aligned.obs_names].copy()
    var = raw_by_timepoint[TIMEPOINTS[0]].var.copy()
    assembled = ad.AnnData(X=counts.copy(), obs=obs, var=var)
    assembled.obs_names = aligned.obs_names.copy()
    assembled.layers["counts"] = counts
    # Keep region for anatomy/orientation QC, but use the unsmoothed cell-type
    # calls as the downstream classification target.  The compatibility
    # Annotation column must never silently turn anatomical regions into
    # purported cell types.
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


def _orientation_qc(adata: ad.AnnData) -> dict[str, Any]:
    records = {}
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


def prepare(
    *,
    raw_dir: Path,
    metadata_h5ad: Path,
    aligned_reference_h5ad: Path,
    output_h5ad: Path,
    output_table: Path,
    manifest_path: Path,
    graph_database: Path | None,
    repair_legacy_d7_left_right: bool,
) -> dict[str, Any]:
    outputs = (output_h5ad, output_table, manifest_path)
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(
            f"Refusing to overwrite chicken-heart outputs: {existing}"
        )
    metadata = ad.read_h5ad(metadata_h5ad)
    aligned = ad.read_h5ad(aligned_reference_h5ad)
    reference_contract = _validate_reference(metadata, aligned)
    raw_by_timepoint = _read_raw_counts(raw_dir)
    assembled = assemble_reviewed_counts(raw_by_timepoint, metadata, aligned)
    coordinate_repair = _apply_anatomical_coordinate_contract(
        assembled,
        repair_legacy_d7_left_right=repair_legacy_d7_left_right,
    )

    required_features: tuple[str, ...] = ()
    coverage: dict[str, Any] | str = "not_requested"
    if graph_database is not None:
        required_features, coverage = match_graph_database_features(
            graph_database,
            assembled.var_names,
            preferred_species_tag="chicken",
        )
        if not required_features:
            raise ContractError(
                "Graph database has no exact chicken-heart feature matches."
            )

    cfg = AlignConfig(
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
    )
    processed, table = preprocess_fixed_spatial(
        assembled,
        time_key="timepoint",
        spatial_key="spatial_aligned",
        cfg=cfg,
    )
    if processed.obs_names.tolist() != aligned.obs_names.tolist():
        raise RuntimeError(
            "Processed chicken-heart rows drifted from reviewed reference."
        )
    if not np.array_equal(
        np.asarray(processed.obsm["spatial_aligned"]),
        np.asarray(assembled.obsm["spatial_aligned"]),
    ):
        raise RuntimeError(
            "Processed chicken-heart coordinates drifted from the contracted input."
        )

    processed.uns["chicken_heart_input_contract_json"] = json.dumps(
        {
            "schema_version": SCHEMA_VERSION,
            "reference": reference_contract,
            "downstream_annotation": {
                "key": "celltype_prediction",
                "compatibility_key": "Annotation",
                "source": "metadata_h5ad",
                "n_classes": int(
                    processed.obs["celltype_prediction"].astype(str).nunique()
                ),
                "ordered_label_sha256": _text_sha256(
                    processed.obs["celltype_prediction"].astype(str).tolist()
                ),
            },
            "coordinate_repair": coordinate_repair,
            "anatomical_orientation_qc": _anatomical_orientation_qc(processed),
            "orientation_qc": _orientation_qc(processed),
            "graph_feature_coverage": coverage,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    output_h5ad.parent.mkdir(parents=True, exist_ok=True)
    processed.write_h5ad(output_h5ad)
    table.to_csv(output_table, index=False)

    raw_identities = {
        timepoint: _file_identity(raw_dir / RAW_FILENAMES[timepoint])
        for timepoint in TIMEPOINTS
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "dataset": "chicken_heart",
        "source_accession": "GSE149457",
        "coordinate_policy": coordinate_repair["policy"],
        "inputs": {
            "metadata_h5ad": _file_identity(metadata_h5ad),
            "aligned_reference_h5ad": _file_identity(aligned_reference_h5ad),
            "raw_10x": raw_identities,
            "graph_database": (
                _file_identity(graph_database)
                if graph_database is not None
                else "not_requested"
            ),
        },
        "reference_contract": reference_contract,
        "coordinate_repair": coordinate_repair,
        "anatomical_orientation_qc": _anatomical_orientation_qc(processed),
        "orientation_qc": _orientation_qc(processed),
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
            "aligned_h5ad": _file_identity(output_h5ad),
            "model_input_table": _file_identity(output_table),
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--metadata-h5ad", type=Path, required=True)
    parser.add_argument("--aligned-reference-h5ad", type=Path, required=True)
    parser.add_argument("--output-h5ad", type=Path, required=True)
    parser.add_argument("--output-table", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--graph-database", type=Path)
    parser.add_argument(
        "--repair-legacy-d7-left-right",
        action="store_true",
        help=(
            "Explicitly reflect only D7 around its stage mean x when the reviewed "
            "reference fails solely by the known D7 RV/LV left-right mirror."
        ),
    )
    return parser


def main() -> None:
    options = _parser().parse_args()
    manifest = prepare(
        raw_dir=options.raw_dir.expanduser().resolve(),
        metadata_h5ad=options.metadata_h5ad.expanduser().resolve(),
        aligned_reference_h5ad=options.aligned_reference_h5ad.expanduser().resolve(),
        output_h5ad=options.output_h5ad.expanduser().resolve(),
        output_table=options.output_table.expanduser().resolve(),
        manifest_path=options.manifest.expanduser().resolve(),
        graph_database=(
            options.graph_database.expanduser().resolve()
            if options.graph_database is not None
            else None
        ),
        repair_legacy_d7_left_right=options.repair_legacy_d7_left_right,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
