from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


SCRIPT = Path(__file__).parents[1] / "scripts" / "prepare_chicken_heart_ot_input.py"
SPEC = importlib.util.spec_from_file_location("prepare_chicken_heart_ot_input", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _fixture() -> ad.AnnData:
    stages = ["D4", "D4", "D7", "D7", "D10", "D10", "D14", "D14"]
    counts = sparse.csr_matrix(np.arange(1, 25).reshape(8, 3))
    obs = pd.DataFrame(
        {
            "timepoint": stages,
            "region": ["Atria", "Valves"] * 4,
            "celltype_prediction": ["type_a", "type_b"] * 4,
            "Annotation": ["old_region"] * 8,
        },
        index=[f"spot_{index}" for index in range(8)],
    )
    adata = ad.AnnData(X=np.log1p(counts.toarray()), obs=obs)
    adata.layers["counts"] = counts
    raw = np.array(
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
    adata.obsm["spatial_original"] = raw
    adata.obsm["spatial_aligned"] = raw / 10.0
    adata.obsm["X_latent"] = np.ones((8, 50))
    adata.obsm["X_pca"] = np.ones((8, 50))
    adata.varm["PCs"] = np.ones((3, 2))
    adata.uns["preprocess_info"] = {"stale": True}
    adata.uns["interaction_graph"] = {"stale": True}
    return adata


def test_prepare_ot_input_rotates_only_d7_and_resets_fitted_state(monkeypatch):
    monkeypatch.setattr(
        MODULE,
        "EXPECTED_COUNTS",
        {"D4": 2, "D7": 2, "D10": 2, "D14": 2},
    )
    original = _fixture()
    prepared, contract = MODULE.prepare_ot_input(original)

    raw = np.asarray(original.obsm["spatial_original"])
    observed = np.asarray(prepared.obsm["spatial_ot_input"])
    expected = raw.copy()
    expected[2:4] = 2.0 * raw[2:4].mean(axis=0) - raw[2:4]

    np.testing.assert_array_equal(observed, expected)
    np.testing.assert_array_equal(observed[[0, 1, 4, 5, 6, 7]], raw[[0, 1, 4, 5, 6, 7]])
    np.testing.assert_array_equal(prepared.obsm["spatial"], expected)
    np.testing.assert_array_equal(prepared.obsm["spatial_reviewed_reference"], raw / 10.0)
    np.testing.assert_array_equal(prepared.X.toarray(), original.layers["counts"].toarray())
    assert "spatial_aligned" not in prepared.obsm
    assert "X_latent" not in prepared.obsm
    assert "X_pca" not in prepared.obsm
    assert "PCs" not in prepared.varm
    assert "preprocess_info" not in prepared.uns
    assert "interaction_graph" not in prepared.uns
    assert prepared.obs["Annotation"].tolist() == prepared.obs["celltype_prediction"].tolist()
    assert contract["coordinate_policy"] == "raw_coordinates_with_predefined_d7_180_rotation"
    assert contract["d7_orientation_correction"]["max_pairwise_distance_error"] == 0.0
    assert json.loads(prepared.uns["chicken_heart_ot_input_contract_json"])[
        "downstream_annotation"
    ]["key"] == "celltype_prediction"
