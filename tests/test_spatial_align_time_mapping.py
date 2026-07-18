import anndata as ad
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
        adata.obs["time_point_processed"] = adata.obs[time_key].map(time_mapping).astype(float)
        adata.obsm["X_latent"] = np.zeros((adata.n_obs, 2), dtype=np.float32)
        return adata

    def fake_align(*, adata, batch_indices, **kwargs):
        selected = adata[adata.obs["time"] != "3.3hpf"].copy()
        assert batch_indices == [1, 2, 3, 4, 5]
        assert selected.obs["time_point_processed"].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]
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
