#!/usr/bin/env python3
"""Prepare the raw-coordinate chicken-heart input for package OT alignment.

The GSE149457 adapter first recovers raw counts, reviewed spot identities, and
cell-type/anatomy metadata.  This second, deterministic step discards the old
fitted coordinate state, rotates only the known reversed D7 raw section by 180
degrees around its own centroid, and writes ``obsm['spatial_ot_input']``.  The
``chicken_heart`` workflow preset then fits spatial alignment, a fresh edge
predictor, and the dynamical model from that input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial.distance import pdist


SCHEMA_VERSION = 1
TIMEPOINTS = ("D4", "D7", "D10", "D14")
EXPECTED_COUNTS = {"D4": 147, "D7": 528, "D10": 908, "D14": 1967}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(values: np.ndarray, *, dtype: str = "<f8") -> str:
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


def _validate_input(adata: ad.AnnData) -> dict[str, Any]:
    required_obs = ("timepoint", "region", "celltype_prediction")
    missing_obs = [key for key in required_obs if key not in adata.obs]
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
    counts_values = (
        np.asarray(adata.layers["counts"].data)
        if sparse.issparse(adata.layers["counts"])
        else np.asarray(adata.layers["counts"])
    )
    if (
        not np.isfinite(counts_values).all()
        or np.any(counts_values < 0)
        or not np.allclose(counts_values, np.rint(counts_values), atol=1e-6)
    ):
        raise ValueError("layers['counts'] must contain nonnegative integer counts.")
    for key in ("region", "celltype_prediction"):
        values = adata.obs[key]
        if values.isna().any() or values.astype(str).str.strip().eq("").any():
            raise ValueError(f"obs[{key!r}] contains missing labels.")
    return {
        "n_obs": int(adata.n_obs),
        "n_vars": int(adata.n_vars),
        "timepoint_counts": observed,
        "observation_order_sha256": _text_sha256(
            adata.obs_names.astype(str).tolist()
        ),
        "raw_coordinate_sha256": _array_sha256(raw),
    }


def prepare_ot_input(adata: ad.AnnData) -> tuple[ad.AnnData, dict[str, Any]]:
    """Return a raw-count AnnData with the deterministic OT coordinate input."""

    input_contract = _validate_input(adata)
    prepared = adata.copy()
    raw = np.asarray(prepared.obsm["spatial_original"], dtype=np.float64).copy()
    times = prepared.obs["timepoint"].astype(str).to_numpy()
    d7_mask = times == "D7"
    d7_before = raw[d7_mask].copy()
    d7_center = d7_before.mean(axis=0)
    ot_input = raw.copy()
    ot_input[d7_mask] = (2.0 * d7_center) - d7_before

    if "spatial_reviewed_reference" not in prepared.obsm and "spatial_aligned" in prepared.obsm:
        prepared.obsm["spatial_reviewed_reference"] = np.asarray(
            prepared.obsm["spatial_aligned"], dtype=np.float64
        ).copy()
    prepared.obsm["spatial_ot_input"] = ot_input
    prepared.obsm["spatial"] = ot_input.copy()
    prepared.obsm.pop("spatial_aligned", None)
    prepared.obsm.pop("X_latent", None)
    prepared.obsm.pop("X_pca", None)
    prepared.varm.pop("PCs", None)

    counts = prepared.layers["counts"].copy()
    prepared.X = counts.copy()
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
        "chicken_heart_input_contract_json",
    ):
        prepared.uns.pop(key, None)

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
    output_contract = {
        "schema_version": SCHEMA_VERSION,
        "coordinate_policy": "raw_coordinates_with_predefined_d7_180_rotation",
        "input": input_contract,
        "d7_orientation_correction": correction,
        "spatial_ot_input_sha256": _array_sha256(ot_input),
        "downstream_annotation": {
            "key": "celltype_prediction",
            "compatibility_key": "Annotation",
            "n_classes": int(
                prepared.obs["celltype_prediction"].astype(str).nunique()
            ),
            "ordered_label_sha256": _text_sha256(
                prepared.obs["celltype_prediction"].astype(str).tolist()
            ),
        },
    }
    prepared.uns["chicken_heart_ot_input_contract_json"] = json.dumps(
        output_contract, sort_keys=True, separators=(",", ":")
    )
    return prepared, output_contract


def _file_identity(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    return {
        "path": str(resolved),
        "size": int(resolved.stat().st_size),
        "sha256": _sha256(resolved),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-h5ad", type=Path, required=True)
    parser.add_argument("--output-h5ad", type=Path, required=True)
    parser.add_argument("--output-table", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    input_h5ad = args.input_h5ad.expanduser().resolve()
    output_h5ad = args.output_h5ad.expanduser().resolve()
    output_table = args.output_table.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    existing = [
        str(path) for path in (output_h5ad, output_table, manifest_path) if path.exists()
    ]
    if existing:
        raise FileExistsError(f"Refusing to overwrite chicken-heart outputs: {existing}")

    prepared, contract = prepare_ot_input(ad.read_h5ad(input_h5ad))
    output_h5ad.parent.mkdir(parents=True, exist_ok=True)
    output_table.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    prepared.write_h5ad(output_h5ad)
    coords = np.asarray(prepared.obsm["spatial_ot_input"], dtype=np.float64)
    pd.DataFrame(
        {
            "spot_id": prepared.obs_names.astype(str),
            "timepoint": prepared.obs["timepoint"].astype(str).to_numpy(),
            "x": coords[:, 0],
            "y": coords[:, 1],
        }
    ).to_csv(output_table, index=False)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "dataset": "chicken_heart",
        "purpose": "raw_coordinate_input_for_package_ot_alignment",
        "input_h5ad": _file_identity(input_h5ad),
        "output_h5ad": _file_identity(output_h5ad),
        "output_table": _file_identity(output_table),
        "contract": contract,
        "next_command": (
            "cytobridge workflow --config chicken_heart --train "
            f"--input-h5ad {output_h5ad} --output-dir <run-dir> --device cuda"
        ),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
