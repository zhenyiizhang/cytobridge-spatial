from __future__ import annotations

import sys

import anndata as ad
import numpy as np

from scripts import preprocess_pipeline


def test_parse_time_mapping_supports_numeric_pair_list_and_json_file(tmp_path):
    assert preprocess_pipeline._parse_time_mapping_arg("[[1, 0.0], [2, 1.5]]") == {
        1: 0.0,
        2: 1.5,
    }

    mapping_path = tmp_path / "mapping.json"
    mapping_path.write_text('{"stage-a": 0, "stage-b": 2}', encoding="utf-8")
    assert preprocess_pipeline._parse_time_mapping_arg(str(mapping_path)) == {
        "stage-a": 0,
        "stage-b": 2,
    }


def test_cli_forwards_time_expression_and_spatial_schema(monkeypatch, tmp_path):
    captured = {}

    def fake_run_preprocessing_pipeline(**kwargs):
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(
        preprocess_pipeline,
        "run_preprocessing_pipeline",
        fake_run_preprocessing_pipeline,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "preprocess_pipeline.py",
            "--data-name",
            "new-data",
            "--h5ad-path",
            str(tmp_path / "input.h5ad"),
            "--time-key",
            "stage_label",
            "--time-mapping",
            '[[1, -1], [2, 0], [3, 1]]',
            "--expression-layer",
            "raw_counts",
            "--counts-layer",
            "legacy_counts",
            "--raw-count-validation",
            "strict",
            "--raw-count-integer-tolerance",
            "0.00001",
            "--input-spatial-key",
            "coordinates",
            "--spatial-obs-keys",
            "x_coord,y_coord",
            "--device",
            "cpu",
        ],
    )

    preprocess_pipeline.main()

    cfg = captured["align_config"]
    assert captured["time_key"] == "stage_label"
    assert cfg.time_mapping == {1: -1, 2: 0, 3: 1}
    assert cfg.expression_layer == "raw_counts"
    assert cfg.counts_layer == "legacy_counts"
    assert cfg.raw_count_validation == "strict"
    assert cfg.raw_count_integer_tolerance == 1e-5
    assert cfg.input_spatial_key == "coordinates"
    assert cfg.spatial_obs_keys == ("x_coord", "y_coord")


def test_interaction_graph_uses_preprocess_canonical_raw_layer():
    adata = ad.AnnData(X=np.ones((2, 2), dtype=np.float32))
    adata.layers["counts"] = np.full((2, 2), 99.0, dtype=np.float32)
    adata.layers["raw_counts"] = np.ones((2, 2), dtype=np.float32)
    adata.uns["preprocess_info"] = {
        "counts_layer": "raw_counts",
        "raw_counts_layer": "raw_counts",
    }

    assert preprocess_pipeline._interaction_expression_layer(adata) == "raw_counts"
