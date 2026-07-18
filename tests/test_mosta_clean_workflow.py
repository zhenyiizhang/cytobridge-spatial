from __future__ import annotations

import json

import anndata as ad
import numpy as np
import pandas as pd

from CytoBridge.pp.preprocess import preprocess
from CytoBridge.pp.spatial_align import AlignConfig, _align_preprocessed_adata
from CytoBridge.utils.config import load_config
from scripts.run_mosta_end_to_end import (
    _build_parser,
    _lr_feature_contract,
    _paths,
    _profile_defaults,
    _resolved_training_config,
    _write_pca_contract,
)
from scripts.run_spatial_training import MOSTA_TIME_MAPPING, _preset_config


def test_mosta_preset_selects_singular_raw_count_layer_and_canonical_times():
    cfg, time_key, batch_indices, data_name = _preset_config("mosta")

    assert time_key == "timepoint"
    assert batch_indices == [3, 4, 5, 6]
    assert data_name == "Mouse_embryo_all_stage"
    assert cfg.expression_layer == "count"
    assert cfg.counts_layer == "count"
    assert cfg.raw_count_validation == "strict"
    assert cfg.normalization_target_sum == 1e4
    assert cfg.allow_retransform_preprocessed_x is False
    assert cfg.auto_scale_from_centered_x_max is False
    assert cfg.scale_x == 0.01
    assert cfg.scale_y == 0.01
    assert cfg.flip_y is True
    assert cfg.time_mapping == MOSTA_TIME_MAPPING
    assert [
        cfg.time_mapping[label] for label in ("E12.5", "E13.5", "E14.5", "E15.5")
    ] == [
        0.0,
        1.0,
        2.0,
        3.0,
    ]


def test_preprocess_forces_required_features_and_records_resolved_target():
    counts = np.asarray(
        [
            [1, 0, 2, 0, 1],
            [0, 3, 0, 1, 0],
            [2, 1, 0, 0, 4],
            [0, 1, 3, 2, 0],
            [5, 0, 0, 1, 1],
            [1, 4, 1, 0, 0],
            [0, 2, 5, 1, 0],
            [3, 0, 1, 4, 1],
        ],
        dtype=np.int64,
    )
    adata = ad.AnnData(
        X=np.log1p(counts.astype(np.float32)),
        obs=pd.DataFrame({"stage": ["a"] * 4 + ["b"] * 4}),
        var=pd.DataFrame(index=["A", "B", "C", "Fzd7", "Lrp6"]),
    )
    adata.layers["count"] = counts

    result = preprocess(
        adata,
        time_key="stage",
        n_top_genes=2,
        n_pcs=2,
        normalization_target_sum=100.0,
        expression_layer="count",
        counts_layer="count",
        raw_count_validation="strict",
        required_latent_features=("Fzd7", "Lrp6"),
    )

    assert result.var.loc["Fzd7", "highly_variable"]
    assert result.var.loc["Lrp6", "highly_variable"]
    info = result.uns["preprocess_info"]
    assert info["expression_source"] == "layers['count']"
    assert info["transformation_sequence"] == ["normalize_total", "log1p"]
    assert info["normalization_target_sum"] == 100.0
    assert info["normalization_target_sum_resolved"] == 100.0
    assert set(info["required_latent_features_requested"]) == {"Fzd7", "Lrp6"}
    assert info["n_latent_fit_features"] >= 2
    assert np.linalg.norm(result.varm["PCs"][result.var_names == "Lrp6"]) > 0


def test_alignment_fit_subsample_still_transforms_every_selected_cell():
    obs = pd.DataFrame(
        {
            "stage": ["a"] * 4 + ["b"] * 4,
            "time_point_processed": [0.0] * 4 + [1.0] * 4,
        },
        index=[f"cell-{index}" for index in range(8)],
    )
    adata = ad.AnnData(X=np.ones((8, 3), dtype=np.float32), obs=obs)
    adata.obsm["X_latent"] = np.arange(16, dtype=np.float32).reshape(8, 2) / 10
    adata.obsm["spatial"] = np.arange(16, dtype=np.float32).reshape(8, 2)
    cfg = AlignConfig(
        n_pcs=2,
        phase1_epochs=0,
        phase2_epochs=0,
        max_cells_per_timepoint=2,
        auto_scale_from_centered_x_max=False,
        random_seed=42,
    )

    aligned, table = _align_preprocessed_adata(
        adata,
        time_key="stage",
        cfg=cfg,
        device="cpu",
        verbose=False,
    )

    assert aligned.n_obs == 8
    assert aligned.obsm["spatial_aligned"].shape == (8, 2)
    assert table.shape == (8, 5)
    assert table["samples"].value_counts().sort_index().tolist() == [4, 4]


def test_mosta_lr_adapter_forces_present_subunits_and_records_missing(tmp_path):
    source = ad.AnnData(
        X=np.ones((2, 4), dtype=np.float32),
        var=pd.DataFrame(index=["Wnt3a", "Fzd7", "Lrp6", "Other"]),
    )
    source_path = tmp_path / "source.h5ad"
    source.write_h5ad(source_path)
    database_path = tmp_path / "mouse_lr.csv"
    pd.DataFrame({"0": ["Wnt3a", "AbsentLigand"], "1": ["Fzd7_Lrp6", "Fzd7"]}).to_csv(
        database_path, index=False
    )

    present, contract = _lr_feature_contract(
        source_path, database_path, tmp_path / "contract"
    )

    assert set(present) == {"Wnt3a", "Fzd7", "Lrp6"}
    assert contract["missing_subunits"] == ["AbsentLigand"]
    written = json.loads(
        (tmp_path / "contract" / "lr_feature_contract.json").read_text(encoding="utf-8")
    )
    assert written["required_focal_panel_subunits"] == ["Fzd7", "Lrp6", "Wnt3a"]


def test_pca_contract_records_required_but_inactive_smoke_features(tmp_path):
    adata = ad.AnnData(
        X=np.asarray([[1.0, 5.0], [2.0, 5.0]], dtype=np.float32),
        var=pd.DataFrame(index=["active", "rare_zero_variance"]),
    )
    adata.varm["PCs"] = np.asarray([[1.0], [0.0]], dtype=np.float32)
    adata.var["pca_center"] = np.asarray([1.5, 5.0], dtype=np.float32)
    adata.obsm["X_latent"] = np.asarray([[-0.5], [0.5]], dtype=np.float32)
    adata.uns["preprocess_info"] = {
        "required_latent_features_requested": ["active", "rare_zero_variance"]
    }
    path = tmp_path / "aligned.h5ad"
    adata.write_h5ad(path)

    report = _write_pca_contract(path, tmp_path / "pca_contract")

    assert report["n_required_latent_features"] == 2
    assert report["n_inactive_required_latent_features"] == 1
    assert report["inactive_required_latent_features"] == ["rare_zero_variance"]


def test_mosta_alpha0015_config_has_recovered_six_stage_schedule():
    config = load_config(
        "CytoBridge/configs/mosta_spatial_full_alpha_express_0015.yaml"
    )
    assert config["training"]["defaults"]["alpha_spatial"] == 10.0
    assert config["training"]["defaults"]["alpha_express"] == 0.015
    plan = config["training"]["plan"]
    assert [stage["name"] for stage in plan] == [
        "Pretrain",
        "Refine",
        "Init_interaction",
        "Train_Score",
        "Finetune",
        "Score_Refine",
    ]
    assert [stage["epochs"] for stage in plan] == [100, 100, 50, 2001, 1000, 2001]
    assert [stage["batch_size"] for stage in plan] == [1024, 1024, 1024, 512, 1024, 512]
    assert _profile_defaults("full")["edge_max_train_edges"] == 2_000_000


def test_mosta_training_alpha_cli_defaults_and_override(tmp_path):
    parser = _build_parser()
    required = [
        "--h5ad-path",
        str(tmp_path / "source.h5ad"),
        "--database-path",
        str(tmp_path / "lr.csv"),
        "--output-dir",
        str(tmp_path / "run"),
        "--profile",
        "full",
    ]

    default_args = parser.parse_args(required)
    default_config = _resolved_training_config(
        default_args, {"training_dir": tmp_path / "default_training"}
    )
    assert default_config["training"]["defaults"]["alpha_spatial"] == 10.0
    assert default_config["training"]["defaults"]["alpha_express"] == 0.015

    override_args = parser.parse_args(
        required + ["--alpha-spatial", "7.5", "--alpha-express", "0.05"]
    )
    override_config = _resolved_training_config(
        override_args, {"training_dir": tmp_path / "override_training"}
    )
    assert override_config["training"]["defaults"]["alpha_spatial"] == 7.5
    assert override_config["training"]["defaults"]["alpha_express"] == 0.05


def test_mosta_train_evaluate_can_share_read_only_preprocess(tmp_path):
    output_dir = tmp_path / "alpha005"
    shared_preprocess = tmp_path / "alpha0015" / "preprocess"
    paths = _paths(output_dir, reuse_preprocess_dir=shared_preprocess)

    assert paths["root"] == output_dir
    assert paths["training_dir"] == output_dir / "training"
    assert paths["evaluation_dir"] == output_dir / "evaluation"
    assert paths["preprocess_dir"] == shared_preprocess.resolve()
    assert paths["aligned_h5ad"] == shared_preprocess.resolve() / "mosta_aligned.h5ad"
    assert paths["edge_path"] == (
        shared_preprocess.resolve() / "edge_classifier" / "mosta_edge_model.pt"
    )

    args = _build_parser().parse_args(
        [
            "--h5ad-path",
            str(tmp_path / "source.h5ad"),
            "--database-path",
            str(tmp_path / "lr.csv"),
            "--output-dir",
            str(output_dir),
            "--stage",
            "train-evaluate",
            "--reuse-preprocess-dir",
            str(shared_preprocess),
            "--alpha-express",
            "0.05",
        ]
    )
    assert args.stage == "train-evaluate"
    assert args.reuse_preprocess_dir == shared_preprocess
    assert args.alpha_express == 0.05
