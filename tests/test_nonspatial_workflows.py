from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import yaml

from CytoBridge.nonspatial import (
    available_nonspatial_presets,
    nonspatial_plan,
    packaged_training_config,
    prepare_scnt_nonspatial,
    prepare_weinreb_nonspatial,
)
from CytoBridge.nonspatial import figures as nonspatial_figures
from CytoBridge.nonspatial import workflow as nonspatial_workflow


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_nonspatial_plan_is_dependency_free_from_source_checkout():
    completed = subprocess.run(
        [
            sys.executable,
            "-S",
            "-m",
            "CytoBridge.cli",
            "nonspatial",
            "plan",
            "--dataset",
            "scnt",
            "--json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    plan = json.loads(completed.stdout)
    assert plan["preset"]["name"] == "scnt_cortex"
    assert plan["steps"][3].endswith("from t=0")


def test_nonspatial_presets_and_configs_encode_corrected_single_factor_contract():
    from CytoBridge.tl.train.trainer import _normalize_matched_ablation

    assert available_nonspatial_presets() == ("scnt_cortex", "weinreb")
    for dataset in available_nonspatial_presets():
        full = yaml.safe_load(packaged_training_config(dataset, "full").read_text())
        no_interaction = yaml.safe_load(
            packaged_training_config(dataset, "no_interaction").read_text()
        )
        assert (
            full["matched_ablation"]["family"]
            == no_interaction["matched_ablation"]["family"]
        )
        assert full["matched_ablation"]["arm"] == "full"
        assert no_interaction["matched_ablation"]["arm"] == "no_interaction"
        for config in (full, no_interaction):
            assert config["seed"] == 42
            assert config["model"]["velocity_net"]["use_spatial"] is False
            assert config["training"]["defaults"]["sigma"] == 0.1
            assert (
                config["training"]["defaults"]["score_energy_objective"]
                == "velocity_score_cross_term"
            )
            assert [stage["epochs"] for stage in config["training"]["plan"]] == [
                100,
                100,
                50,
                2001,
                1000,
                2001,
            ]
        assert "interaction" in full["model"]["components"]
        assert "interaction" not in no_interaction["model"]["components"]
        assert all(
            stage.get("interaction_use") is False
            for stage in no_interaction["training"]["plan"]
            if stage["mode"] == "neural_ode"
        )
        _normalize_matched_ablation(full, "full_learned")
        _normalize_matched_ablation(no_interaction, "no_interaction")

        assert {
            key: value
            for key, value in full.items()
            if key not in {"model", "training", "ckpt_dir", "matched_ablation"}
        } == {
            key: value
            for key, value in no_interaction.items()
            if key not in {"model", "training", "ckpt_dir", "matched_ablation"}
        }
        for network in ("velocity_net", "growth_net", "score_net"):
            assert no_interaction["model"][network] == full["model"][network]
        assert no_interaction["training"]["defaults"] == full["training"]["defaults"]
        for full_stage, no_interaction_stage in zip(
            full["training"]["plan"],
            no_interaction["training"]["plan"],
            strict=True,
        ):
            ignored = {"name", "train_strategy", "interaction_use"}
            assert {
                key: value for key, value in full_stage.items() if key not in ignored
            } == {
                key: value
                for key, value in no_interaction_stage.items()
                if key not in ignored
            }
            assert no_interaction_stage["train_strategy"] == full_stage[
                "train_strategy"
            ].replace("+i", "")
        plan = nonspatial_plan(dataset)
        assert plan["preset"]["expected_latent_dim"] == 50
        assert plan["steps"][3].endswith("from t=0")


def test_preprocess_manifests_are_dataset_bound_and_blinded():
    weinreb = nonspatial_workflow.nonspatial_preset("weinreb")
    valid_weinreb = {
        "schema_version": 2,
        "operation": "prepare_weinreb_nonspatial",
        "shape_latent": [49_302, 50],
        "preprocessing": {
            "uses_spatial_coordinates": False,
            "uses_clone_or_annotation_for_preprocessing": False,
        },
    }
    nonspatial_workflow._validate_preprocessing_manifest(weinreb, valid_weinreb)
    with pytest.raises(ValueError, match="requires operation"):
        nonspatial_workflow._validate_preprocessing_manifest(
            weinreb,
            {
                **valid_weinreb,
                "operation": "prepare_scnt_nonspatial",
            },
        )

    scnt = nonspatial_workflow.nonspatial_preset("scnt_cortex")
    valid_scnt = {
        "schema_version": 2,
        "operation": "prepare_scnt_nonspatial",
        "model_shape": [20_547, 50],
        "preprocessing": {"uses_spatial_coordinates": False},
        "training_blinding": {
            "new_layer_present_in_training_h5ad": False,
            "old_layer_present_in_training_h5ad": False,
            "new_layer_present_in_lr_expression_h5ad": False,
            "old_layer_present_in_lr_expression_h5ad": False,
            "cell_type_used_for_latent_or_radius": False,
        },
    }
    nonspatial_workflow._validate_preprocessing_manifest(scnt, valid_scnt)
    tampered = {**valid_scnt, "training_blinding": {"new_layer": True}}
    with pytest.raises(ValueError, match="blinding"):
        nonspatial_workflow._validate_preprocessing_manifest(scnt, tampered)


def test_historical_figure_bundle_rejects_panel_data_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    bundle = tmp_path / "bundle"
    panel_data = bundle / "panel_data"
    panel_data.mkdir(parents=True)
    observed = panel_data / "observed_cells.csv.gz"
    copied = panel_data / "metric.csv"
    observed.write_bytes(b"observed")
    copied.write_bytes(b"metric")
    manifest = {
        "figure": "weinreb_nonspatial_interaction_a4",
        "sources": {
            "derived_observed_cells": {"sha256": _sha256(observed)},
            "metric_source": {"sha256": _sha256(copied)},
        },
    }
    (panel_data / "source_manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(
        nonspatial_figures,
        "_load_builder",
        lambda _dataset: (
            "weinreb",
            SimpleNamespace(COPIED_NAMES={"metric_source": "metric.csv"}),
        ),
    )
    result = nonspatial_figures.validate_historical_figure_bundle("weinreb", bundle)
    assert set(result["panel_data"]) == {"observed_cells.csv.gz", "metric.csv"}

    copied.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        nonspatial_figures.validate_historical_figure_bundle("weinreb", bundle)


def test_identity_simulation_lands_exactly_and_preserves_model_state():
    torch = pytest.importorskip("torch")
    from CytoBridge.nonspatial.weinreb_simulation import simulate_sde_from_x0

    class ConstantModel(torch.nn.Module):
        components = ("velocity", "growth")

        def __init__(self):
            super().__init__()
            self.marker = torch.nn.Parameter(torch.tensor(1.0))

        def predict_velocity(self, *, t, x):
            return torch.zeros_like(x) * self.marker

        def predict_growth(self, *, t, x):
            return torch.ones((x.shape[0], 1), device=x.device) * self.marker

    model = ConstantModel().train()
    x0 = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    points, weights, normalized = simulate_sde_from_x0(
        x0=x0,
        model=model,
        ts_points=[0.0, 0.25],
        dt=0.1,
        sigma=0.0,
        include_interaction=False,
        device="cpu",
        noise_seed=7,
        interaction_seed=10_042,
        verbose=False,
    )
    assert points.shape == (2, 2, 2)
    assert np.array_equal(points[0], x0)
    assert np.array_equal(points[1], x0)
    assert np.allclose(weights[0, :, 0], 0.5)
    assert np.allclose(weights[1, :, 0], 0.5 * np.exp(0.25), rtol=2e-6)
    assert np.allclose(normalized.sum(axis=1), 1.0)
    assert model.training is True
    assert model.marker.requires_grad is True


def test_prepare_weinreb_separates_lr_expression_from_50pc_state(tmp_path: Path):
    rng = np.random.default_rng(7)
    n_cells, n_genes = 18, 10
    obs = pd.DataFrame(
        {
            "Time point": np.repeat([2, 4, 6], 6),
            "Starting population": ["p1", "p2"] * 9,
            "Cell type annotation": ["a", "b", "c"] * 6,
            "clone": [f"c{index % 4}" for index in range(n_cells)],
            "SPRING-x": rng.normal(size=n_cells),
            "SPRING-y": rng.normal(size=n_cells),
        },
        index=[f"cell{index}" for index in range(n_cells)],
    )
    source = ad.AnnData(
        X=rng.gamma(2.0, 2.0, size=(n_cells, n_genes)).astype(np.float32),
        obs=obs,
        var=pd.DataFrame(
            {"gene": [f"g{index}" for index in range(n_genes)]},
            index=[f"id{index}" for index in range(n_genes)],
        ),
    )
    source_path = tmp_path / "weinreb.h5ad"
    source.write_h5ad(source_path)
    result = prepare_weinreb_nonspatial(
        source_path,
        tmp_path / "model.h5ad",
        expression_output_h5ad=tmp_path / "expression.h5ad",
        n_hvg=8,
        n_pcs=3,
        interaction_group_size=4,
    )
    model = ad.read_h5ad(result.model_h5ad)
    expression = ad.read_h5ad(result.expression_h5ad)
    manifest = json.loads(result.manifest.read_text())
    assert model.shape == (n_cells, 3)
    assert model.obsm["X_latent"].shape == (n_cells, 3)
    assert np.allclose(np.asarray(expression.X.sum(axis=1)).reshape(-1), 10_000)
    assert "SPRING-x" not in model.obsm
    assert bool(model.uns["preprocessing"]["uses_spring_coordinates"]) is False
    assert manifest["expression_output_sha256"]
    assert sorted(model.obs["time_point_processed"].unique()) == [0.0, 1.0, 2.0]


def test_prepare_scnt_seals_new_old_and_uses_total_only(tmp_path: Path):
    rng = np.random.default_rng(11)
    n_cells, n_genes = 20, 12
    new = rng.poisson(2, size=(n_cells, n_genes)).astype(np.float32)
    old = rng.poisson(3, size=(n_cells, n_genes)).astype(np.float32)
    total = new + old
    obs = pd.DataFrame(
        {
            "time_point_processed": np.repeat([0.0, 0.25, 0.5, 1.0], 5),
            "cell_type": ["Ex", "Inh"] * 10,
        },
        index=[f"cell{index}" for index in range(n_cells)],
    )
    source = ad.AnnData(
        X=total.copy(),
        obs=obs,
        var=pd.DataFrame(
            {"gene_short_name": [f"g{index}" for index in range(n_genes)]},
            index=[f"id{index}" for index in range(n_genes)],
        ),
        layers={"new": new, "old": old, "total": total},
    )
    source_path = tmp_path / "scnt.h5ad"
    source.write_h5ad(source_path)
    result = prepare_scnt_nonspatial(
        source_path,
        tmp_path / "model.h5ad",
        tmp_path / "expression.h5ad",
        n_hvg=10,
        n_pcs=3,
        interaction_group_size=4,
    )
    model = ad.read_h5ad(result.model_h5ad)
    expression = ad.read_h5ad(result.expression_h5ad)
    manifest = json.loads(result.manifest.read_text())
    assert model.shape == (n_cells, 3)
    assert not model.layers
    assert not expression.layers
    assert np.allclose(np.asarray(expression.X.sum(axis=1)).reshape(-1), 10_000)
    assert manifest["training_blinding"] == {
        "cell_type_used_for_latent_or_radius": False,
        "new_layer_present_in_lr_expression_h5ad": False,
        "new_layer_present_in_training_h5ad": False,
        "old_layer_present_in_lr_expression_h5ad": False,
        "old_layer_present_in_training_h5ad": False,
    }


def test_scnt_preprocessing_rejects_total_new_old_inconsistency(tmp_path: Path):
    source = ad.AnnData(
        X=np.ones((8, 6), dtype=np.float32),
        obs=pd.DataFrame(
            {
                "time_point_processed": [0.0] * 4 + [1.0] * 4,
                "cell_type": ["a", "b"] * 4,
            }
        ),
        layers={
            "new": np.ones((8, 6), dtype=np.float32),
            "old": np.ones((8, 6), dtype=np.float32),
            "total": np.ones((8, 6), dtype=np.float32),
        },
    )
    source_path = tmp_path / "bad.h5ad"
    source.write_h5ad(source_path)
    with pytest.raises(ValueError, match=r"total must equal new \+ old"):
        prepare_scnt_nonspatial(
            source_path,
            tmp_path / "model.h5ad",
            tmp_path / "expression.h5ad",
            n_hvg=4,
            n_pcs=2,
        )
