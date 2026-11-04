from __future__ import annotations

import gzip
import json
import pickle
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import yaml

import CytoBridge as cb
from CytoBridge.workflow import WorkflowOptions, run_workflow


def _write_raw_h5ad(path: Path) -> None:
    """Write three small spatial time slices with two co-active LR systems."""

    genes = [
        "LigA",
        "LigB",
        "LigC",
        "RecA",
        "RecB",
        "RecC",
        "House1",
        "House2",
        "State1",
        "State2",
        "Noise1",
        "Noise2",
    ]
    n_per_time = 18
    rows = []
    coordinates = []
    times = []
    annotations = []
    rng = np.random.default_rng(42)

    for time_index, time_label in enumerate(("E1", "E2", "E3")):
        for cell_index in range(n_per_time):
            counts = rng.integers(1, 5, size=len(genes)).astype(np.float32)
            role = cell_index % 3
            if role == 0:
                counts[:3] = 40
                annotations.append("sender")
            elif role == 1:
                counts[3:6] = 40
                annotations.append("receiver")
            else:
                counts[6:8] = 40
                annotations.append("background")
            counts[8] += 3 * time_index
            counts[9] += 2 * (2 - time_index)
            rows.append(counts)
            coordinates.append(
                [
                    0.10 * (cell_index % 6) + 0.01 * time_index,
                    0.10 * (cell_index // 6),
                ]
            )
            times.append(time_label)

    counts = np.asarray(rows, dtype=np.float32)
    obs = pd.DataFrame(
        {"Timepoint": times, "Annotation": annotations},
        index=[f"cell_{index}" for index in range(counts.shape[0])],
    )
    data = ad.AnnData(X=counts.copy(), obs=obs)
    data.var_names = genes
    data.layers["counts"] = counts.copy()
    data.obsm["spatial"] = np.asarray(coordinates, dtype=np.float32)
    data.write_h5ad(path)


def _write_lr_database(path: Path) -> None:
    pd.DataFrame(
        {
            "Ligand": ["LIGA_LIGB", "LIGC", "LIG"],
            "Receptor": ["RECA_RECB", "RECC", "RECA"],
            "Pathway": ["complex_path", "simple_path", "substring_control"],
            "Annotation": [
                "Secreted Signaling",
                "Secreted Signaling",
                "Secreted Signaling",
            ],
        }
    ).to_csv(path, index=False)


def _smoke_config() -> dict:
    return {
        "dataset": {
            "name": "scientific_smoke",
            "time_key": "time_point_processed",
            "obsm_key": "X_latent",
            "spatial_key": "spatial_aligned",
            "annotation_key": "Annotation",
            "concat_spatial": True,
        },
        "steps": {"default": ["preprocess"]},
        "scientific": {
            "seed": 42,
            "alpha_spatial": 10.0,
            "alpha_express": 0.015,
            "classifier_k": 10,
        },
        "preprocess": {
            "enabled": True,
            "time_key": "Timepoint",
            "batch_indices": None,
            "annotation_source": "Annotation",
            "align": {
                "n_top_genes": 12,
                "n_pcs": 4,
                "normalization_target_sum": 10_000.0,
                "spatial_dim": 2,
                "auto_scale_from_centered_x_max": False,
                "center_x": False,
                "center_y": False,
                "phase1_epochs": 1,
                "phase2_epochs": 1,
                "alpha": 1.0,
                "beta": 0.01,
                "lambda_local": 1.0,
                "lambda_ot": 1.0,
                "batch_size": 18,
                "distance_pairs": 64,
                "learning_rate": 0.001,
                "random_seed": 42,
                "expression_layer": "counts",
                "counts_layer": "counts",
                "raw_count_validation": "strict",
                "input_spatial_key": "spatial",
                "time_mapping": {"E1": 0.0, "E2": 1.0, "E3": 2.0},
            },
            "edge_predictor": {
                "epochs": 2,
                "batch_size": 64,
                "learning_rate": 0.01,
                "num_workers": 0,
                "spot_diameter": 0.05,
                "verbose": False,
                "use_tqdm": False,
            },
        },
        "train": {
            "config": "st_spatial_smoke.yaml",
            "requires_edge_predictor": True,
            "interaction_cutoff": 10.0,
            "evaluate_after_training": False,
        },
        "downstream": {"enabled": False},
    }


def _write_training_config(path: Path, *, edge_prior_mode: str = "learned") -> None:
    """Keep the real three-stage trainer but bound each stage to one epoch."""

    config = {
        "model": {
            "components": ["velocity", "growth", "score", "interaction"],
            "interaction_type": "gnn",
            "interaction_group_size": 64,
            "velocity_net": {
                "hidden_dim": 16,
                "n_layers": 1,
                "residual": False,
                "activation": "leaky_relu",
            },
            "growth_net": {
                "hidden_dim": 16,
                "n_layers": 1,
                "residual": False,
                "activation": "leaky_relu",
            },
            "score_net": {
                "hidden_dim": 16,
                "n_layers": 1,
                "activation": "leaky_relu",
            },
            "interaction_net": {
                "hidden_dim": 16,
                "num_heads": 2,
                "num_layers": 1,
                "activation": "leakyrelu",
                "num_rbf": 4,
                "cutoff": 10.0,
                "use_spatial": True,
                "edge_prior_mode": edge_prior_mode,
            },
        },
        "seed": 42,
        "training": {
            "defaults": {
                "lr": 0.0005,
                "lambda_ot": 1.0,
                "lambda_mass": 1.0,
                "lambda_energy": 0.01,
                "sigma": 0.03,
                "batch_size": 12,
                "alpha_spatial": 10.0,
                "alpha_express": 0.015,
            },
            "plan": [
                {
                    "name": "Pretrain",
                    "mode": "neural_ode",
                    "epochs": 1,
                    "OT_loss": "sinkhorn",
                    "train_strategy": "v+g",
                },
                {
                    "name": "Train_FM",
                    "mode": "flow_matching",
                    "epochs": 1,
                    "score_use": True,
                    "train_strategy": "s",
                    "flow_matching": {"lambda_penalty": 0},
                },
                {
                    "name": "Finetune",
                    "mode": "neural_ode",
                    "epochs": 1,
                    "OT_loss": "sinkhorn_detach",
                    "train_strategy": "v+g+i+s",
                },
            ],
        },
    }
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def test_real_raw_h5ad_to_model_and_velocity_components(tmp_path: Path):
    """Exercise the scientific path without mocking graph/model computation."""

    pytest.importorskip("qnorm", reason="requires the CytoBridge preprocess extra")
    raw_h5ad = tmp_path / "raw.h5ad"
    lr_database = tmp_path / "lr.csv"
    training_config = tmp_path / "training.yaml"
    output_dir = tmp_path / "run"
    _write_raw_h5ad(raw_h5ad)
    _write_lr_database(lr_database)
    _write_training_config(training_config)

    result = run_workflow(
        _smoke_config(),
        options=WorkflowOptions(
            input_h5ad=raw_h5ad,
            output_dir=output_dir,
            graph_database=lr_database,
            training_config=str(training_config),
            device="cpu",
            steps=("preprocess",),
            train=True,
        ),
    )

    assert result["completed"] == ["preprocess", "edge_predictor", "train"]
    aligned_path = Path(result["outputs"]["aligned_h5ad"])
    edge_output = result["outputs"]["edge_predictor"]
    edge_model_path = Path(edge_output["model_path"])
    edge_meta = json.loads(Path(edge_output["meta_path"]).read_text())
    model_dir = Path(result["outputs"]["model_dir"])

    aligned = ad.read_h5ad(aligned_path)
    assert aligned.shape == (54, 12)
    assert aligned.obsm["X_latent"].shape == (54, 4)
    assert aligned.obsm["spatial_aligned"].shape == (54, 2)
    np.testing.assert_array_equal(
        np.unique(aligned.obs["time_point_processed"]), [0.0, 1.0, 2.0]
    )
    assert aligned.uns["preprocess_info"]["raw_counts_layer"] == "counts"

    # Exact complex matching accepts the full complex and rejects substring LIG.
    for graph_slice in edge_output["graph_slices"]:
        stats = graph_slice["lr_database_stats"]
        assert stats["rows_total"] == 3
        assert stats["rows_matched"] == 2
        assert stats["rows_missing_subunit"] == 1
        assert stats["matched_complex_pairs"] == 1
        slice_name = graph_slice["data_name"]
        graph_path = (
            Path(edge_output["graph_input_dir"])
            / slice_name
            / f"{slice_name}_adjacency_records"
        )
        with gzip.open(graph_path, "rb") as handle:
            records = pickle.load(handle)
        assert {tuple(pair) for pair in records[2]} == {
            ("LIGA_LIGB", "RECA_RECB"),
            ("LIGC", "RECC"),
        }

    # Both LR systems support the same cell pairs, but binary edge training
    # sees each directed pair once and selects its threshold on validation.
    dedup = edge_meta["positive_edge_deduplication"]
    assert dedup["raw"] > dedup["unique"] > 0
    assert dedup["duplicates_removed"] == dedup["raw"] - dedup["unique"]
    assert edge_meta["selection_source"] == "validation"
    assert (
        edge_meta["edge_predictor_threshold"]
        == edge_meta["edge_predictor_threshold_selected"]
    )
    assert edge_meta["split"]["strategy"] == "node_disjoint_holdout"
    universe = edge_meta["candidate_universe"]
    assert universe["definition"] == (
        "all directed pairs with 1e-6 < distance < cutoff"
    )
    assert 0.0 < universe["validation_positive_fraction"] < 1.0
    assert universe["training_balanced_edges"] % 2 == 0
    assert edge_model_path.is_file()

    # The validation-selected predictor threshold is wired into main training.
    training_config = yaml.safe_load((model_dir / "config.yaml").read_text())
    interaction = training_config["model"]["interaction_net"]
    assert Path(interaction["edge_predictor_path"]) == edge_model_path.resolve()
    assert (
        interaction["edge_predictor_thre"]
        == edge_meta["edge_predictor_threshold_selected"]
    )
    assert training_config["training"]["defaults"]["alpha_express"] == 0.015

    # Exercise real public downstream APIs against the checkpoint produced above.
    loaded = cb.tl.load_dynamical_model_from_dir(
        model_dir,
        dim=6,
        device="cpu",
        edge_predictor_path=edge_model_path,
    )
    components = cb.tl.compute_velocity_components_from_adata(
        aligned,
        loaded.model,
        device="cpu",
        time_key="time_point_processed",
        obsm_key="X_latent",
        spatial_key="spatial_aligned",
        concat_spatial=True,
        write_to_adata=False,
    )
    assert set(components) == {
        "drift",
        "interaction",
        "score",
        "full",
        "times",
        "features",
    }
    assert components["full"].shape == (54, 6)
    assert np.isfinite(components["full"]).all()


def test_real_raw_h5ad_all_spatial_skips_edge_predictor_and_has_finite_velocity(
    tmp_path: Path,
):
    """Run the raw-data all-spatial scientific path without mocked compute."""

    pytest.importorskip("qnorm", reason="requires the CytoBridge preprocess extra")
    pytest.importorskip("torch_geometric", reason="requires the CytoBridge graph extra")
    raw_h5ad = tmp_path / "raw.h5ad"
    training_config_path = tmp_path / "training_all_spatial.yaml"
    output_dir = tmp_path / "run"
    _write_raw_h5ad(raw_h5ad)
    _write_training_config(training_config_path, edge_prior_mode="all_spatial")

    result = run_workflow(
        _smoke_config(),
        options=WorkflowOptions(
            input_h5ad=raw_h5ad,
            output_dir=output_dir,
            training_config=str(training_config_path),
            device="cpu",
            steps=("preprocess",),
            train=True,
        ),
    )

    assert result["completed"] == ["preprocess", "train"]
    assert "edge_predictor" not in result["outputs"]
    assert not (output_dir / "preprocess" / "edge_classifier").exists()
    assert not (output_dir / "preprocess" / "input_graph").exists()
    assert not list(output_dir.rglob("*_edge_model.pt"))
    assert not list(output_dir.rglob("*_edge_model.meta.json"))

    aligned = ad.read_h5ad(result["outputs"]["aligned_h5ad"])
    assert aligned.shape == (54, 12)
    assert "interaction_graph" not in aligned.uns

    model_dir = Path(result["outputs"]["model_dir"])
    resolved_config = yaml.safe_load((model_dir / "config.yaml").read_text())
    interaction_config = resolved_config["model"]["interaction_net"]
    assert interaction_config["edge_prior_mode"] == "all_spatial"
    assert "edge_predictor_path" not in interaction_config
    assert "edge_predictor_thre" not in interaction_config

    trained = ad.read_h5ad(model_dir / "adata.h5ad")
    assert "interaction_graph" not in trained.uns
    assert trained.uns["all_model"]["edge_prior_mode"] == "all_spatial"
    assert trained.uns["all_model"].get("edge_predictor_path") is None
    assert trained.uns["all_model"].get("edge_predictor_threshold") is None

    loaded = cb.tl.load_dynamical_model_from_dir(model_dir, dim=6, device="cpu")
    assert loaded.model.interaction_net.edge_prior_mode == "all_spatial"
    assert not hasattr(loaded.model.interaction_net, "link_predictor")
    components = cb.tl.compute_velocity_components_from_adata(
        aligned,
        loaded.model,
        device="cpu",
        time_key="time_point_processed",
        obsm_key="X_latent",
        spatial_key="spatial_aligned",
        concat_spatial=True,
        write_to_adata=False,
    )
    assert components["full"].shape == (54, 6)
    assert np.isfinite(components["full"]).all()
    assert np.isfinite(components["interaction"]).all()
