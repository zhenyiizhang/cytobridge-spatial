from __future__ import annotations

import json

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

import CytoBridge.pp.chicken_heart_input as heart_input
from CytoBridge.pp import validate_chicken_heart_ot_input


def _fixture() -> ad.AnnData:
    stages = ["D4", "D4", "D7", "D7", "D10", "D10", "D14", "D14"]
    counts = sparse.csr_matrix(np.arange(1, 25).reshape(8, 3))
    obs = pd.DataFrame(
        {
            "timepoint": stages,
            "region": ["Atria", "Valves"] * 4,
            "celltype_prediction": ["type_a", "type_b"] * 4,
            "Annotation": ["old"] * 8,
        },
        index=[f"spot_{index}" for index in range(8)],
    )
    result = ad.AnnData(X=np.log1p(counts.toarray()), obs=obs)
    result.layers["counts"] = counts
    raw = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [10.0, 10.0],
            [12.0, 14.0],
            [20.0, 20.0],
            [21.0, 20.0],
            [30.0, 30.0],
            [31.0, 30.0],
        ]
    )
    result.obsm["spatial_original"] = raw
    result.obsm["spatial_aligned"] = raw / 10.0
    result.obsm["X_latent"] = np.ones((8, 50))
    result.obsm["X_pca"] = np.ones((8, 50))
    result.varm["PCs"] = np.ones((3, 2))
    result.uns["preprocess_info"] = {"stale": True}
    return result


def test_prepare_ot_input_rotates_only_d7_and_resets_fitted_state(monkeypatch):
    monkeypatch.setattr(
        heart_input,
        "EXPECTED_COUNTS",
        {"D4": 2, "D7": 2, "D10": 2, "D14": 2},
    )
    original = _fixture()
    prepared, validation = heart_input.prepare_chicken_heart_ot_adata(original)

    raw = np.asarray(original.obsm["spatial_original"])
    expected = raw.copy()
    expected[2:4] = 2.0 * raw[2:4].mean(axis=0) - raw[2:4]
    np.testing.assert_array_equal(prepared.obsm["spatial_ot_input"], expected)
    np.testing.assert_array_equal(prepared.obsm["spatial_reference"], raw / 10.0)
    assert "spatial_aligned" not in prepared.obsm
    assert "X_latent" not in prepared.obsm
    assert "X_pca" not in prepared.obsm
    assert "PCs" not in prepared.varm
    assert "preprocess_info" not in prepared.uns
    assert (
        prepared.obs["Annotation"].tolist()
        == prepared.obs["celltype_prediction"].tolist()
    )
    assert validation["d7_orientation_correction"]["max_pairwise_distance_error"] == 0.0
    validation_json = json.dumps(validation).lower()
    assert "sha256" not in validation_json
    assert "hashlib" not in validation_json
    assert validate_chicken_heart_ot_input(prepared)["timepoint_counts"] == {
        "D4": 2,
        "D7": 2,
        "D10": 2,
        "D14": 2,
    }


def test_ot_manifest_uses_names_shapes_and_counts(tmp_path, monkeypatch):
    monkeypatch.setattr(
        heart_input,
        "EXPECTED_COUNTS",
        {"D4": 2, "D7": 2, "D10": 2, "D14": 2},
    )
    input_h5ad = tmp_path / "input.h5ad"
    _fixture().write_h5ad(input_h5ad)
    manifest = heart_input.prepare_chicken_heart_ot_input(
        input_h5ad=input_h5ad,
        output_h5ad=tmp_path / "ot.h5ad",
        output_table=tmp_path / "coordinates.csv",
        manifest_path=tmp_path / "manifest.json",
    )

    assert manifest["input"] == "input.h5ad"
    assert manifest["outputs"]["h5ad"] == "ot.h5ad"
    assert manifest["outputs"]["coordinate_table_shape"] == [8, 4]
    public_json = json.dumps(manifest).lower()
    assert str(tmp_path).lower() not in public_json
    assert "sha256" not in public_json
    assert "hashlib" not in public_json
