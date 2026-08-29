from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import read_h5ad

from CytoBridge.pp import legacy_model_input_csv_to_adata, write_legacy_model_input_h5ad


def _write_example(path) -> None:
    pd.DataFrame(
        {
            "samples": [0.0, 1.0, 1.0],
            "x1": [0.1, 0.2, 0.3],
            "x2": [1.1, 1.2, 1.3],
            "x3": [2.1, 2.2, 2.3],
            "x4": [3.1, 3.2, 3.3],
            "Annotation": ["a", "b", "b"],
        }
    ).to_csv(path, index=False)


def test_legacy_model_input_is_split_and_provenanced(tmp_path) -> None:
    source = tmp_path / "legacy.csv"
    _write_example(source)
    adata = legacy_model_input_csv_to_adata(
        source,
        interaction_cutoff=0.05,
        edge_predictor_threshold=0.45,
        edge_predictor_path="edge_classifier/example.pt",
    )

    np.testing.assert_allclose(
        adata.obsm["spatial_aligned"],
        [[0.1, 1.1], [0.2, 1.2], [0.3, 1.3]],
    )
    np.testing.assert_allclose(
        adata.obsm["X_latent"],
        [[2.1, 3.1], [2.2, 3.2], [2.3, 3.3]],
    )
    assert adata.var_names.tolist() == ["x3", "x4"]
    assert adata.obs["time_point_processed"].tolist() == [0.0, 1.0, 1.0]
    assert adata.obs["Annotation"].tolist() == ["a", "b", "b"]
    assert adata.uns["preprocess_info"]["gene_expression_available"] is False
    assert adata.uns["legacy_model_input"]["model_input_order"] == ["x1", "x2", "x3", "x4"]
    assert adata.uns["fit_params"] == {
        "interaction_cutoff": 0.05,
        "edge_predictor_threshold": 0.45,
        "edge_predictor_path": "edge_classifier/example.pt",
    }


def test_legacy_model_input_h5ad_round_trip(tmp_path) -> None:
    source = tmp_path / "legacy.csv"
    output = tmp_path / "legacy.h5ad"
    _write_example(source)
    written = write_legacy_model_input_h5ad(source, output)
    loaded = read_h5ad(output)

    assert output.is_file()
    assert written.n_obs == 3
    assert loaded.n_obs == 3
    assert loaded.uns["legacy_model_input"]["source_csv"] == str(source.resolve())


def test_legacy_model_input_accepts_table_without_annotation(tmp_path) -> None:
    source = tmp_path / "legacy.csv"
    _write_example(source)
    frame = pd.read_csv(source).drop(columns="Annotation")
    frame.to_csv(source, index=False)

    adata = legacy_model_input_csv_to_adata(source)

    assert "Annotation" not in adata.obs
    assert adata.uns["legacy_model_input"]["annotation_column"] == "absent"
    assert adata.obsm["X_latent"].shape == (3, 2)


def test_legacy_model_input_rejects_nonfinite_values(tmp_path) -> None:
    source = tmp_path / "legacy.csv"
    _write_example(source)
    frame = pd.read_csv(source)
    frame.loc[1, "x3"] = np.inf
    frame.to_csv(source, index=False)

    with pytest.raises(ValueError, match="must be finite"):
        legacy_model_input_csv_to_adata(source)
