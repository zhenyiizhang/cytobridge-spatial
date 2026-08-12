from __future__ import annotations

import json
from pathlib import Path

import pytest
import CytoBridge as cb

from CytoBridge.cli import main
from CytoBridge.workflow import (
    WorkflowOptions,
    _run_edge_predictor,
    build_workflow_plan,
    load_workflow_config,
    run_workflow,
    _run_train,
)


@pytest.mark.parametrize(
    ("name", "classifier_k"),
    (("zebrafish", 10), ("mosta", 10), ("arista", 10), ("admouse", 1)),
)
def test_packaged_presets_share_formal_scientific_defaults(name, classifier_k):
    config, source = load_workflow_config(name)

    assert source == f"packaged preset: {name}"
    assert config["scientific"] == {
        "seed": 42,
        "alpha_spatial": 10.0,
        "alpha_express": 0.015,
        "classifier_k": classifier_k,
    }


def test_cli_exposes_explicit_complete_reference_pca_center_opt_in(capsys):
    assert (
        main(
            [
                "workflow",
                "--config",
                "admouse",
                "--dry-run",
                "--allow-complete-reference-pca-center-fallback",
            ]
        )
        == 0
    )
    assert "allow-complete-reference-pca-center-fallback" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("name", "graph_database", "cutoff", "edge_threshold"),
    (
        (
            "mosta",
            "CellChatDB.ligrec.mouse.csv",
            0.02400244047956264,
            0.44999998807907104,
        ),
        (
            "arista",
            "CellChatDB.ligrec.human.csv",
            0.03154105148551745,
            0.23999999463558197,
        ),
        (
            "zebrafish",
            "CellChatDB.ligrec.zebrafish.csv",
            0.09606367405591873,
            0.4999999701976776,
        ),
        (
            "admouse",
            "CellChatDB.ligrec.mouse.csv",
            0.012106042891492197,
            0.32999998331069946,
        ),
    ),
)
def test_packaged_presets_plan_the_formal_graph_contract(
    name,
    graph_database,
    cutoff,
    edge_threshold,
):
    config, source = load_workflow_config(name)
    plan = build_workflow_plan(
        config,
        source=source,
        options=WorkflowOptions(train=True),
    )
    training = next(step for step in plan["steps"] if step["name"] == "train")

    assert training["training_config"] == config["train"]["config"]
    assert config["train"]["graph_database"] == graph_database
    assert config["train"]["edge_predictor_threshold"] == edge_threshold
    assert training["interaction_cutoff"] == cutoff
    assert training["edge_predictor_threshold"] is None
    assert training["edge_predictor_threshold_source"] == (
        "validation-selected during preprocessing"
    )


def test_admouse_preset_supports_de_novo_and_historical_artifact_paths(tmp_path):
    config, source = load_workflow_config("admouse")
    de_novo = build_workflow_plan(
        config,
        source=source,
        options=WorkflowOptions(
            train=True,
            input_h5ad=tmp_path / "admouse_raw.h5ad",
            output_dir=tmp_path / "run",
        ),
    )
    preprocessing = next(
        step for step in de_novo["steps"] if step["name"] == "preprocess"
    )
    training = next(step for step in de_novo["steps"] if step["name"] == "train")
    downstream = next(step for step in de_novo["steps"] if step["name"] == "downstream")

    assert preprocessing["status"] == "ready"
    assert preprocessing["edge_predictor"]["status"] == (
        "will be trained automatically"
    )
    assert preprocessing["edge_predictor"]["graph_database"].endswith(
        "CellChatDB.ligrec.mouse.csv"
    )
    assert training["training_config"] == "admouse_spatial_full_alpha_express_0015.yaml"
    assert training["status"] == "ready"
    assert downstream["model_format"] == "current"

    align = config["preprocess"]["align"]
    assert config["steps"]["default"] == ["downstream"]
    assert config["preprocess"]["time_key"] == "Timepoint"
    assert config["preprocess"]["batch_indices"] is None
    assert align["time_mapping"] == {"1": 0.0, "2": 1.0, "3": 2.0}
    assert align["expression_layer"] == "counts"
    assert align["normalization_target_sum"] == 10_000.0
    assert align["n_top_genes"] == 2_000
    assert align["n_pcs"] == 50
    assert align["auto_scale_from_centered_x_max"] is True

    released_edge = tmp_path / "released_edge.pt"
    released_edge.touch()
    historical = build_workflow_plan(
        config,
        source=source,
        options=WorkflowOptions(
            train=True,
            steps=("downstream",),
            aligned_h5ad=tmp_path / "released_aligned.h5ad",
            edge_predictor_path=released_edge,
            output_dir=tmp_path / "historical",
        ),
    )
    historical_preprocess = next(
        step for step in historical["steps"] if step["name"] == "preprocess"
    )
    historical_training = next(
        step for step in historical["steps"] if step["name"] == "train"
    )
    assert historical_preprocess["status"] == "skipped"
    assert historical_training["status"] == "ready"
    assert historical_training["edge_predictor_source"] == (
        "explicit --edge-predictor-path"
    )


def test_training_passes_formal_graph_values_as_explicit_fit_arguments(
    monkeypatch,
    tmp_path,
):
    config, _ = load_workflow_config("admouse")
    edge_model = tmp_path / "admouse_edge_model.pt"
    captured = {}

    def fake_fit(aligned_h5ad, **kwargs):
        captured["aligned_h5ad"] = aligned_h5ad
        captured.update(kwargs)

    monkeypatch.setattr(cb.tl, "fit", fake_fit)
    _run_train(
        config,
        WorkflowOptions(edge_predictor_path=edge_model, device="cpu"),
        aligned_h5ad=tmp_path / "admouse_aligned.h5ad",
        model_dir=tmp_path / "training",
    )

    assert captured["interaction_cutoff"] == 0.012106042891492197
    assert captured["edge_predictor_threshold"] == 0.32999998331069946
    assert captured["edge_predictor_path"] == str(edge_model.resolve())
    resolved = captured["config"]
    assert resolved["training"]["defaults"]["alpha_spatial"] == 10.0
    assert resolved["training"]["defaults"]["alpha_express"] == 0.015
    assert resolved["model"]["interaction_net"]["cutoff"] == 0.012106042891492197
    assert captured["time_key"] == "time_point_processed"
    assert captured["obsm_key"] == "X_latent"
    assert captured["spatial_key"] == "spatial_aligned"
    assert captured["is_spatial"] is True


def test_training_uses_custom_dataset_schema(monkeypatch, tmp_path):
    config, _ = load_workflow_config("zebrafish")
    config["dataset"] = {
        **config["dataset"],
        "time_key": "stage",
        "obsm_key": "latent_custom",
        "spatial_key": "xy_custom",
        "concat_spatial": False,
    }
    captured = {}

    def fake_fit(_aligned_h5ad, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(cb.tl, "fit", fake_fit)
    _run_train(
        config,
        WorkflowOptions(edge_predictor_path=tmp_path / "edge.pt", device="cpu"),
        aligned_h5ad=tmp_path / "aligned.h5ad",
        model_dir=tmp_path / "training",
    )

    assert captured["time_key"] == "stage"
    assert captured["obsm_key"] == "latent_custom"
    assert captured["spatial_key"] == "xy_custom"
    assert captured["is_spatial"] is False


def test_formal_downstream_simulation_profiles_are_packaged():
    expected = {
        "zebrafish": (None, 9, 0.05),
        "mosta": (12000, 13, 0.05),
        "arista": (7668, 9, 0.01),
        "admouse": (None, 26, 0.01),
    }
    for dataset, (particles, n_times, split_dt) in expected.items():
        config, _ = load_workflow_config(dataset)
        downstream = config["downstream"]
        time_grid = set(downstream["observed"]) | set(downstream["interpolated"])
        assert downstream["sde_n_samples"] == particles
        assert len(time_grid) == n_times
        assert downstream["split_sde_dt"] == split_dt
        assert downstream["gene_dynamics_enabled"] is True
        assert downstream["lr_enabled"] is True


def test_arista_preprocessing_uses_raw_counts_layer():
    config, _ = load_workflow_config("arista")
    align = config["preprocess"]["align"]

    assert align["expression_layer"] == "counts"
    assert align["counts_layer"] == "counts"
    assert align["raw_count_validation"] == "strict"
    assert align["observation_id_keys"] == ["Batch", "CellID"]
    assert align["hvg_batch_key"] == "Batch"
    assert align["center_x"] is True
    assert align["center_y"] is True
    assert config["preprocess"]["time_key"] == "Batch"
    assert config["preprocess"]["batch_values"] == [
        "Injury_2DPI_rep1_SS200000147BL_D5",
        "Injury_5DPI_rep1_SS200000147BL_D2",
        "Injury_10DPI_rep1_SS200000147BL_B5",
        "Injury_15DPI_rep4_FP200000266TR_E4",
        "Injury_20DPI_rep2_SS200000147BL_B4",
    ]
    assert set(config["preprocess"]["drop_uns_keys"]) == {
        "Injury_2DPI_rep1_SS200000147BL_D5",
        "Injury_5DPI_rep1_SS200000147BL_D2",
        "Injury_10DPI_rep1_SS200000147BL_B5",
        "Injury_15DPI_rep4_FP200000266TR_E4",
        "Injury_20DPI_rep2_SS200000147BL_B4",
        "Injury_30DPI_rep2_FP200000264BL_A6",
        "Injury_60DPI_rep3_FP200000264BL_A6",
        "Injury_control_FP200000239BL_E3",
    }


def test_admouse_preprocessing_uses_stable_sample_cell_identity():
    config, _ = load_workflow_config("admouse")

    assert config["preprocess"]["align"]["observation_id_keys"] == [
        "sample",
        "cell_id",
    ]


def test_zebrafish_preprocessing_declares_observation_coordinate_columns():
    config, _ = load_workflow_config("zebrafish")

    assert config["preprocess"]["align"]["spatial_obs_keys"] == [
        "spatial_x",
        "spatial_y",
    ]


def test_training_is_skipped_until_explicitly_enabled():
    config, source = load_workflow_config("zebrafish")
    base = build_workflow_plan(
        config,
        source=source,
        options=WorkflowOptions(),
    )
    enabled = build_workflow_plan(
        config,
        source=source,
        options=WorkflowOptions(train=True),
    )

    base_training = next(step for step in base["steps"] if step["name"] == "train")
    base_preprocessing = next(
        step for step in base["steps"] if step["name"] == "preprocess"
    )
    enabled_training = next(
        step for step in enabled["steps"] if step["name"] == "train"
    )
    assert base_training["status"] == "skipped; add --train to run"
    assert base_preprocessing["status"] == "skipped"
    assert enabled_training["status"] == "missing input"
    assert "--edge-predictor-path" not in enabled_training["missing"]
    assert enabled_training["edge_predictor_source"] == "generated by preprocessing"


def test_preprocess_and_downstream_require_training_or_separate_commands():
    config, source = load_workflow_config("zebrafish")

    with pytest.raises(ValueError, match="cannot share one command without --train"):
        build_workflow_plan(
            config,
            source=source,
            options=WorkflowOptions(steps=("preprocess", "downstream")),
        )


def test_raw_training_plan_uses_bundled_formal_database_by_default(tmp_path):
    config, source = load_workflow_config("mosta")
    plan = build_workflow_plan(
        config,
        source=source,
        options=WorkflowOptions(
            train=True,
            input_h5ad=tmp_path / "raw.h5ad",
            output_dir=tmp_path / "run",
        ),
    )

    preprocessing = next(step for step in plan["steps"] if step["name"] == "preprocess")
    training = next(step for step in plan["steps"] if step["name"] == "train")
    edge = preprocessing["edge_predictor"]

    assert preprocessing["status"] == "ready"
    assert edge["status"] == "will be trained automatically"
    assert edge["graph_database"].endswith("CellChatDB.ligrec.mouse.csv")
    assert edge["database_source"] == "bundled formal CellChatDB resource"
    assert training["status"] == "ready"
    assert training["edge_predictor_source"] == "generated by preprocessing"
    assert training["edge_predictor_path"].endswith("mosta_edge_model.pt")
    assert training["edge_predictor_threshold"] is None
    assert training["edge_predictor_threshold_source"] == (
        "validation-selected during preprocessing"
    )


def test_raw_training_plan_reports_explicit_graph_overrides(tmp_path):
    config, source = load_workflow_config("arista")
    plan = build_workflow_plan(
        config,
        source=source,
        options=WorkflowOptions(
            train=True,
            input_h5ad=tmp_path / "raw.h5ad",
            output_dir=tmp_path / "run",
            interaction_cutoff=0.123,
            edge_predictor_threshold=0.42,
        ),
    )

    preprocessing = next(step for step in plan["steps"] if step["name"] == "preprocess")
    training = next(step for step in plan["steps"] if step["name"] == "train")
    edge = preprocessing["edge_predictor"]

    assert edge["interaction_cutoff"] == 0.123
    assert edge["decision_threshold"] == 0.42
    assert edge["decision_threshold_source"] == ("explicit --edge-predictor-threshold")
    assert training["interaction_cutoff"] == 0.123
    assert training["edge_predictor_threshold"] == 0.42
    assert training["edge_predictor_threshold_source"] == (
        "explicit --edge-predictor-threshold"
    )


def test_raw_preprocess_and_train_rejects_existing_edge_predictor(tmp_path):
    config, source = load_workflow_config("zebrafish")
    edge_predictor = tmp_path / "existing_edge.pt"
    edge_predictor.touch()
    with pytest.raises(ValueError, match="must fit a new edge predictor"):
        build_workflow_plan(
            config,
            source=source,
            options=WorkflowOptions(
                train=True,
                input_h5ad=tmp_path / "raw.h5ad",
                output_dir=tmp_path / "run",
                edge_predictor_path=edge_predictor,
            ),
        )


def test_admouse_historical_training_still_requires_released_edge_predictor(tmp_path):
    config, source = load_workflow_config("admouse")
    plan = build_workflow_plan(
        config,
        source=source,
        options=WorkflowOptions(
            train=True,
            steps=("downstream",),
            aligned_h5ad=tmp_path / "admouse_aligned.h5ad",
            output_dir=tmp_path / "run",
        ),
    )
    training = next(step for step in plan["steps"] if step["name"] == "train")

    assert training["status"] == "missing input"
    assert "--edge-predictor-path" in training["missing"]


def test_workflow_passes_automatic_edge_model_and_threshold_to_training(
    monkeypatch,
    tmp_path,
):
    import CytoBridge.workflow as workflow

    config, _ = load_workflow_config("arista")
    captured = {}

    def preprocess(_config, resolved_options, *, aligned_h5ad):
        captured["preprocess_species"] = resolved_options.preferred_species_tag
        return aligned_h5ad

    monkeypatch.setattr(workflow, "_run_preprocess", preprocess)

    def edge(_config, resolved_options, *, aligned_h5ad, edge_predictor_path):
        captured["edge_aligned"] = aligned_h5ad
        captured["edge_path"] = edge_predictor_path
        captured["edge_species"] = resolved_options.preferred_species_tag
        return {
            "model_path": str(edge_predictor_path),
            "edge_predictor_threshold": 0.24,
        }

    def train(
        _config,
        _options,
        *,
        aligned_h5ad,
        model_dir,
        edge_predictor_path,
        edge_predictor_threshold,
    ):
        captured["train_aligned"] = aligned_h5ad
        captured["train_edge_path"] = edge_predictor_path
        captured["train_edge_threshold"] = edge_predictor_threshold
        return model_dir

    monkeypatch.setattr(workflow, "_run_edge_predictor", edge)
    monkeypatch.setattr(workflow, "_run_train", train)

    result = run_workflow(
        config,
        options=WorkflowOptions(
            train=True,
            input_h5ad=tmp_path / "raw.h5ad",
            output_dir=tmp_path / "run",
            steps=("preprocess",),
            preferred_species_tag="nr",
        ),
    )

    expected_edge = (
        tmp_path / "run" / "preprocess" / "edge_classifier" / "arista_edge_model.pt"
    )
    assert result["completed"] == ["preprocess", "edge_predictor", "train"]
    assert captured["edge_path"] == expected_edge
    assert captured["preprocess_species"] == "nr"
    assert captured["edge_species"] == "nr"
    assert captured["train_edge_path"] == expected_edge
    assert captured["train_edge_threshold"] == 0.24


def test_edge_predictor_uses_formal_graph_and_training_parameters(
    monkeypatch,
    tmp_path,
):
    import CytoBridge.workflow as workflow

    config, _ = load_workflow_config("mosta")
    database_path = tmp_path / "CellChatDB.ligrec.mouse.csv"
    database_path.write_text("ligand,receptor\nA,B\n", encoding="utf-8")
    captured_graphs = []
    captured_training = {}

    class FakeAdata:
        obs = __import__("pandas").DataFrame(
            {"time_point_processed": [0.0, 0.0, 1.0, 1.0]}
        )
        uns = {"preprocess_info": {"raw_counts_layer": "count"}}
        layers = {"count": object()}

        def write_h5ad(self, path):
            captured_training["written_h5ad"] = Path(path)

    monkeypatch.setattr("anndata.read_h5ad", lambda _path: FakeAdata())
    monkeypatch.setattr(
        workflow,
        "resolve_graph_database",
        lambda _dataset, _path, *, bundled_filename: database_path,
    )

    def graph(**kwargs):
        captured_graphs.append(kwargs)
        return {"data_name": kwargs["data_name"]}

    def train_edge(**kwargs):
        captured_training.update(kwargs)
        return {
            "meta_path": str(tmp_path / "edge.pt.meta.json"),
            "edge_predictor_threshold": 0.47,
            "edge_predictor_threshold_selected": 0.47,
        }

    monkeypatch.setattr(cb.pp, "generate_interaction_graph", graph)
    monkeypatch.setattr(cb.pp, "train_edge_predictor", train_edge)
    monkeypatch.setattr(cb.pp, "sanitize_interaction_graph_uns", lambda _adata: None)

    output = _run_edge_predictor(
        config,
        WorkflowOptions(device="cpu", preferred_species_tag="nr"),
        aligned_h5ad=tmp_path / "mosta_aligned.h5ad",
        edge_predictor_path=tmp_path / "edge.pt",
    )

    assert [item["data_name"] for item in captured_graphs] == ["mosta_t0", "mosta_t1"]
    assert all(item["database_path"] == str(database_path) for item in captured_graphs)
    assert all(item["preferred_species_tag"] == "nr" for item in captured_graphs)
    assert all(
        item["neighborhood_threshold"] == 0.02400244047956264
        for item in captured_graphs
    )
    assert captured_training["distance_threshold"] == 0.02400244047956264
    assert captured_training["edge_predictor_threshold"] is None
    assert captured_training["max_train_edges_per_epoch"] == 2_000_000
    assert output["graph_database"] == str(database_path)
    assert output["edge_predictor_threshold"] == 0.47


def test_dry_run_is_read_only_and_reports_scientific_parameters(capsys):
    assert main(["workflow", "--config", "admouse", "--dry-run"]) == 0
    output = capsys.readouterr().out

    assert "alpha_express=0.015" in output
    assert "alpha_spatial=10" in output
    assert "seed=42" in output
    assert "classifier_k=1" in output
    assert "train: skipped; add --train to run" in output
    assert "gene dynamics: enabled" in output
    assert "strict ligand-receptor projection: enabled" in output
    assert "CellChatDB.ligrec.mouse.csv" in output
    assert "dry-run: no work executed" in output


def test_dry_run_json_is_machine_readable(capsys):
    assert main(["workflow", "--config", "mosta", "--dry-run", "--json"]) == 0
    plan = json.loads(capsys.readouterr().out)

    assert plan["dataset"]["name"] == "mosta"
    assert plan["scientific"]["classifier_k"] == 10
    assert plan["scientific"]["alpha_express"] == 0.015
    assert plan["scientific"]["alpha_spatial"] == 10.0
