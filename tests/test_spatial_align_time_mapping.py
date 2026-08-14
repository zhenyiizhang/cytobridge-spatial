import anndata as ad
import json
import numpy as np
import pandas as pd

import CytoBridge.pp.spatial_align as spatial_align


def test_preprocess_and_align_forwards_explicit_time_mapping(monkeypatch):
    adata = ad.AnnData(
        X=np.ones((6, 3), dtype=np.float32),
        obs=pd.DataFrame(
            {
                "time": ["3.3hpf", "5.25hpf", "10hpf", "12hpf", "18hpf", "24hpf"],
                "spatial_x": np.arange(6, dtype=float),
                "spatial_y": np.arange(6, dtype=float),
            }
        ),
    )
    mapping = {
        "3.3hpf": -1.0,
        "5.25hpf": 0.0,
        "10hpf": 1.0,
        "12hpf": 2.0,
        "18hpf": 3.0,
        "24hpf": 4.0,
    }

    def fake_preprocess(*, adata, time_key, time_mapping, **kwargs):
        assert time_key == "time"
        assert time_mapping == mapping
        adata.obs["time_point_processed"] = (
            adata.obs[time_key].map(time_mapping).astype(float)
        )
        adata.obsm["X_latent"] = np.zeros((adata.n_obs, 2), dtype=np.float32)
        return adata

    def fake_align(*, adata, batch_indices, **kwargs):
        selected = adata[adata.obs["time"] != "3.3hpf"].copy()
        assert batch_indices == [1, 2, 3, 4, 5]
        assert selected.obs["time_point_processed"].tolist() == [
            0.0,
            1.0,
            2.0,
            3.0,
            4.0,
        ]
        return selected, pd.DataFrame({"samples": [0.0, 1.0, 2.0, 3.0, 4.0]})

    monkeypatch.setattr(spatial_align, "preprocess", fake_preprocess)
    monkeypatch.setattr(spatial_align, "_align_preprocessed_adata", fake_align)

    cfg = spatial_align.AlignConfig(time_mapping=mapping)
    result, _ = spatial_align.preprocess_and_align(
        adata,
        time_key="time",
        cfg=cfg,
        batch_indices=[1, 2, 3, 4, 5],
        device="cpu",
    )

    assert result.obs["time_point_processed"].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_fixed_spatial_preprocessing_preserves_external_reference(monkeypatch):
    coordinates = np.asarray([[-1.0, 0.5], [0.0, 1.5], [1.0, 2.5]], dtype=np.float64)
    adata = ad.AnnData(
        X=np.ones((3, 4), dtype=np.float32),
        obs=pd.DataFrame(
            {"stage": ["D4", "D7", "D14"]},
            index=["spot-a", "spot-b", "spot-c"],
        ),
    )
    adata.obsm["spatial_aligned"] = coordinates.copy()

    def fake_preprocess(*, adata, time_key, time_mapping, **kwargs):
        assert time_key == "stage"
        adata.obs["time_point_processed"] = (
            adata.obs[time_key].map(time_mapping).astype(float)
        )
        adata.obsm["X_latent"] = np.arange(6, dtype=float).reshape(3, 2)
        return adata

    monkeypatch.setattr(spatial_align, "preprocess", fake_preprocess)
    result, table = spatial_align.preprocess_fixed_spatial(
        adata,
        time_key="stage",
        cfg=spatial_align.AlignConfig(
            spatial_dim=2,
            time_mapping={"D4": 0.0, "D7": 1.0, "D14": 3.0},
        ),
    )

    np.testing.assert_array_equal(result.obsm["spatial_aligned"], coordinates)
    assert result.obs_names.tolist() == ["spot-a", "spot-b", "spot-c"]
    assert result.uns["spatial_alignment_info"]["mode"] == "fixed_external_reference"
    assert len(result.uns["spatial_alignment_info"]["coordinate_sha256"]) == 64
    np.testing.assert_array_equal(table["samples"], [0.0, 1.0, 3.0])
    np.testing.assert_array_equal(table[["x1", "x2"]], coordinates)


def test_fixed_spatial_preprocessing_fails_if_preprocess_changes_coordinates(
    monkeypatch,
):
    adata = ad.AnnData(
        X=np.ones((2, 2), dtype=np.float32),
        obs=pd.DataFrame({"stage": ["D4", "D7"]}, index=["a", "b"]),
    )
    adata.obsm["spatial_aligned"] = np.zeros((2, 2), dtype=float)

    def mutate_coordinates(*, adata, **kwargs):
        adata.obs["time_point_processed"] = [0.0, 1.0]
        adata.obsm["X_latent"] = np.zeros((2, 1), dtype=float)
        adata.obsm["spatial_aligned"][0, 0] = 1.0
        return adata

    monkeypatch.setattr(spatial_align, "preprocess", mutate_coordinates)
    with np.testing.assert_raises_regex(RuntimeError, "changed fixed spatial"):
        spatial_align.preprocess_fixed_spatial(
            adata,
            time_key="stage",
            cfg=spatial_align.AlignConfig(spatial_dim=2),
        )


def test_numeric_time_mapping_in_alignment_config_is_h5ad_safe(tmp_path):
    cfg = spatial_align.AlignConfig(
        time_mapping={1: 0.0, 2: 1.0},
        spatial_obs_keys=("x_coord", "y_coord"),
    )
    safe_config = spatial_align._align_config_for_uns(cfg)
    assert safe_config["time_mapping"] == {"1": 0.0, "2": 1.0}

    adata = ad.AnnData(X=np.ones((2, 2), dtype=np.float32))
    adata.uns["spatial_alignment_info"] = {"config": safe_config}
    output_path = tmp_path / "safe_alignment_config.h5ad"
    adata.write_h5ad(output_path)
    assert output_path.is_file()


def test_prepare_alignment_accepts_explicit_coordinate_schema():
    obs = pd.DataFrame(
        {
            "stage": ["a", "b"],
            "time_point_processed": [0.0, 1.0],
            "x_coord": [10.0, 20.0],
            "y_coord": [30.0, 40.0],
        },
        index=["cell-a", "cell-b"],
    )
    adata = ad.AnnData(X=np.ones((2, 2), dtype=np.float32), obs=obs)
    adata.obsm["X_latent"] = np.ones((2, 2), dtype=np.float32)
    cfg = spatial_align.AlignConfig(
        spatial_dim=2,
        input_spatial_key="missing_obsm_key",
        spatial_obs_keys=("x_coord", "y_coord"),
    )

    prepared, batches = spatial_align._prepare_adata_for_alignment(
        adata,
        time_key="stage",
        cfg=cfg,
    )

    np.testing.assert_array_equal(
        prepared.obsm["spatial"],
        np.asarray([[10.0, 30.0], [20.0, 40.0]], dtype=np.float32),
    )
    assert batches == ["a", "b"]


def test_prepare_alignment_selects_named_batches_independent_of_category_order():
    obs = pd.DataFrame(
        {
            "stage": pd.Categorical(
                ["late", "early", "middle"],
                categories=["middle", "late", "early"],
            ),
            "time_point_processed": [2.0, 0.0, 1.0],
        },
        index=["late-cell", "early-cell", "middle-cell"],
    )
    adata = ad.AnnData(X=np.ones((3, 2), dtype=np.float32), obs=obs)
    adata.obsm["X_latent"] = np.ones((3, 2), dtype=np.float32)
    adata.obsm["spatial"] = np.arange(6, dtype=np.float32).reshape(3, 2)

    prepared, batches = spatial_align._prepare_adata_for_alignment(
        adata,
        time_key="stage",
        cfg=spatial_align.AlignConfig(),
        batch_values=["early", "middle"],
    )

    assert batches == ["early", "middle"]
    assert prepared.obs_names.tolist() == ["early-cell", "middle-cell"]


def test_prepare_alignment_rejects_mixed_batch_selectors():
    obs = pd.DataFrame(
        {"stage": ["a"], "time_point_processed": [0.0]},
        index=["cell"],
    )
    adata = ad.AnnData(X=np.ones((1, 1), dtype=np.float32), obs=obs)
    adata.obsm["X_latent"] = np.ones((1, 1), dtype=np.float32)
    adata.obsm["spatial"] = np.ones((1, 2), dtype=np.float32)

    with np.testing.assert_raises_regex(ValueError, "not both"):
        spatial_align._prepare_adata_for_alignment(
            adata,
            time_key="stage",
            cfg=spatial_align.AlignConfig(),
            batch_indices=[0],
            batch_values=["a"],
        )


def test_prepare_alignment_rejects_unused_categorical_batch_value():
    obs = pd.DataFrame(
        {
            "stage": pd.Categorical(["a"], categories=["a", "empty"]),
            "time_point_processed": [0.0],
        },
        index=["cell"],
    )
    adata = ad.AnnData(X=np.ones((1, 1), dtype=np.float32), obs=obs)
    adata.obsm["X_latent"] = np.ones((1, 1), dtype=np.float32)
    adata.obsm["spatial"] = np.ones((1, 2), dtype=np.float32)

    with np.testing.assert_raises_regex(ValueError, "no observed cells"):
        spatial_align._prepare_adata_for_alignment(
            adata,
            time_key="stage",
            cfg=spatial_align.AlignConfig(),
            batch_values=["empty"],
        )


def test_preprocess_file_wrapper_drops_only_declared_raw_uns(monkeypatch, tmp_path):
    raw = ad.AnnData(
        X=np.ones((2, 2), dtype=np.float32),
        obs=pd.DataFrame({"time": [0, 1]}, index=["a", "b"]),
    )
    raw.uns["large_attachment"] = {"seg_cell": np.ones((4, 4), dtype=np.int32)}
    raw.uns["keep_colors"] = ["#000000", "#ffffff"]
    raw_path = tmp_path / "raw.h5ad"
    raw.write_h5ad(raw_path)

    def fake_preprocess_and_align(*, adata, **kwargs):
        assert "large_attachment" not in adata.uns
        assert list(adata.uns["keep_colors"]) == ["#000000", "#ffffff"]
        return adata, pd.DataFrame({"samples": [0.0, 1.0]})

    monkeypatch.setattr(
        spatial_align, "preprocess_and_align", fake_preprocess_and_align
    )
    result = spatial_align.preprocess_align_to_files(
        h5ad_path=str(raw_path),
        time_key="time",
        output_csv=str(tmp_path / "aligned.csv"),
        output_h5ad=None,
        drop_uns_keys=["large_attachment"],
    )

    record = json.loads(result.uns["cytobridge_removed_raw_uns_json"])
    assert record == {
        "reason": "dataset preset excludes non-model imaging attachments",
        "already_absent": [],
        "removed": [
            {
                "key": "large_attachment",
                "type": "dict",
                "child_keys": ["seg_cell"],
            }
        ],
    }
    assert "large_attachment" not in result.uns
    assert list(result.uns["keep_colors"]) == ["#000000", "#ffffff"]


def test_preprocess_file_wrapper_accepts_already_slim_input(monkeypatch, tmp_path):
    raw = ad.AnnData(X=np.ones((1, 1), dtype=np.float32))
    raw.obs["time"] = [0]
    raw_path = tmp_path / "raw.h5ad"
    raw.write_h5ad(raw_path)

    monkeypatch.setattr(
        spatial_align,
        "preprocess_and_align",
        lambda *, adata, **kwargs: (adata, pd.DataFrame({"samples": [0.0]})),
    )
    result = spatial_align.preprocess_align_to_files(
        h5ad_path=str(raw_path),
        time_key="time",
        output_csv=str(tmp_path / "aligned.csv"),
        output_h5ad=None,
        drop_uns_keys=["missing_attachment"],
    )

    record = json.loads(result.uns["cytobridge_removed_raw_uns_json"])
    assert record["removed"] == []
    assert record["already_absent"] == ["missing_attachment"]
