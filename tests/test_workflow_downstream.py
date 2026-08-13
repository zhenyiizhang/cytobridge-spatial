from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from CytoBridge.workflow import (
    WorkflowOptions,
    _effective_downstream_analyses,
    _loaded_model_scientific_contract,
    _require_pca_reference,
    _run_downstream,
    _write_communication_outputs,
    _write_composition_outputs,
    _write_lr_outputs,
    _write_reconstruction_diagnostic,
    _write_standard_figures,
    _write_velocity_outputs,
    build_workflow_plan,
    load_workflow_config,
    plan_missing_inputs,
)


class _Reference:
    def __init__(self, *, with_pca: bool = True):
        self.varm = {"PCs": np.eye(2)} if with_pca else {}
        self.var = pd.DataFrame(
            {"pca_center": [0.0, 0.0]} if with_pca else {},
            index=["GeneA", "GeneB"],
        )


def test_plan_describes_core_and_explicit_optional_downstream(tmp_path: Path):
    config, source = load_workflow_config("zebrafish")
    missing_lr = tmp_path / "missing_lr.csv"
    plan = build_workflow_plan(
        config,
        source=source,
        options=WorkflowOptions(
            aligned_h5ad=tmp_path / "aligned.h5ad",
            model_dir=tmp_path / "model",
            output_dir=tmp_path / "output",
            gene_dynamics=True,
            lr_database=missing_lr,
            reconstruction_diagnostic=True,
            steps=("downstream",),
        ),
    )

    downstream = next(step for step in plan["steps"] if step["name"] == "downstream")
    analyses = {item["name"]: item for item in downstream["analyses"]}
    assert downstream["simulation"]["split_resample_dt"] == 0.05
    assert downstream["simulation"]["split_max_particles"] == 100_000
    assert analyses["time-slice velocity"]["status"] == "enabled"
    assert analyses["sparse communication"]["status"] == "enabled"
    assert analyses["gene dynamics"]["status"] == "enabled"
    assert analyses["gene dynamics"]["source"] == "explicit --gene-dynamics"
    assert analyses["strict ligand-receptor projection"]["status"] == "enabled"
    assert analyses["strict ligand-receptor projection"]["source"] == (
        "explicit --lr-database override"
    )
    assert (
        "not a training holdout"
        in analyses["fitted-model reconstruction diagnostic"]["note"]
    )
    assert plan_missing_inputs(plan) == [
        f"strict ligand-receptor projection: LR database file not found: {missing_lr}"
    ]


def test_packaged_downstream_defaults_share_the_graph_database():
    expected_species_tags = {
        "zebrafish": "zebrafish",
        "mosta": "mouse",
        "arista": "hs",
        "admouse": "mouse",
    }
    for name, species_tag in expected_species_tags.items():
        config, source = load_workflow_config(name)
        effective = _effective_downstream_analyses(config, WorkflowOptions())
        plan = build_workflow_plan(
            config,
            source=source,
            options=WorkflowOptions(
                aligned_h5ad=Path("aligned.h5ad"),
                model_dir=Path("model"),
                output_dir=Path("output"),
                steps=("downstream",),
            ),
        )
        downstream = next(
            step for step in plan["steps"] if step["name"] == "downstream"
        )
        analyses = {item["name"]: item for item in downstream["analyses"]}

        assert effective["gene_dynamics"] is True
        assert effective["lr_enabled"] is True
        assert effective["preferred_species_tag"] == species_tag
        assert Path(effective["lr_database"]).name == config["train"]["graph_database"]
        assert analyses["gene dynamics"]["status"] == "enabled"
        assert analyses["gene dynamics"]["source"] == "packaged preset default"
        assert analyses["strict ligand-receptor projection"]["status"] == "enabled"
        assert analyses["strict ligand-receptor projection"]["database"].endswith(
            config["train"]["graph_database"]
        )
        assert analyses["strict ligand-receptor projection"]["missing"] == []


def test_zebrafish_package_workflow_passes_the_formal_split_contract(
    monkeypatch,
    tmp_path,
):
    import anndata as ad
    import CytoBridge as cb
    import CytoBridge.workflow as workflow

    config, _ = load_workflow_config("zebrafish")
    adata = SimpleNamespace(
        obsm={
            "X_latent": np.zeros((2, 2), dtype=np.float32),
            "spatial_aligned": np.zeros((2, 2), dtype=np.float32),
        },
        obs=pd.DataFrame(
            {
                "Annotation": ["A", "A"],
                "time_point_processed": [0.0, 1.0],
            }
        ),
    )
    dataframe = pd.DataFrame(
        {
            "samples": [0.0, 1.0],
            "x1": [0.0, 0.0],
            "x2": [0.0, 0.0],
            "x3": [0.0, 0.0],
            "x4": [0.0, 0.0],
            "Annotation": ["A", "A"],
        }
    )
    captured = {}

    class StopAfterSimulationCall(Exception):
        pass

    def capture_workflow(**kwargs):
        captured.update(kwargs)
        raise StopAfterSimulationCall

    monkeypatch.setattr(ad, "read_h5ad", lambda _path: adata)
    monkeypatch.setattr(
        cb.tl,
        "adata_to_aligned_dataframe",
        lambda *_args, **_kwargs: (dataframe, "time_point_processed"),
    )
    monkeypatch.setattr(
        cb.tl,
        "infer_feature_columns",
        lambda *_args, **_kwargs: ["x1", "x2", "x3", "x4"],
    )
    monkeypatch.setattr(
        cb.tl,
        "load_dynamical_model_from_dir",
        lambda *_args, **_kwargs: SimpleNamespace(model=object()),
    )
    monkeypatch.setattr(cb.tl, "build_dynamical_runtime", lambda _loaded: object())
    monkeypatch.setattr(cb.tl, "run_interpolation_workflow", capture_workflow)
    monkeypatch.setattr(
        workflow,
        "_loaded_model_scientific_contract",
        lambda *_args, **_kwargs: {
            "status": "test",
            "edge_prior_mode": "learned",
            "interaction_group_size": 64,
        },
    )

    with pytest.raises(StopAfterSimulationCall):
        _run_downstream(
            config,
            WorkflowOptions(device="cpu"),
            aligned_h5ad=tmp_path / "aligned.h5ad",
            model_dir=tmp_path / "model",
            output_dir=tmp_path / "out",
        )

    assert captured["split_sde_dt"] == 0.05
    assert captured["split_resample_dt"] == 0.05
    assert captured["split_max_particles"] == 100_000
    assert captured["split_growth_alpha"] == 1.0
    assert captured["split_interaction_m"] == 64


def test_downstream_cli_database_and_species_overrides_take_precedence(tmp_path: Path):
    config, _ = load_workflow_config("arista")
    custom_database = tmp_path / "custom_lr.csv"
    custom_database.write_text("ligand,receptor\nL,R\n", encoding="utf-8")

    effective = _effective_downstream_analyses(
        config,
        WorkflowOptions(
            lr_database=custom_database,
            preferred_species_tag="nr",
        ),
    )

    assert effective["lr_database"] == custom_database
    assert effective["lr_database_source"] == "explicit --lr-database override"
    assert effective["preferred_species_tag"] == "nr"


def test_packaged_downstream_can_skip_pca_dependent_outputs_for_old_artifacts():
    config, _ = load_workflow_config("arista")
    effective = _effective_downstream_analyses(
        config,
        WorkflowOptions(skip_gene_dynamics=True, skip_lr=True),
    )

    assert effective["gene_dynamics"] is False
    assert effective["lr_enabled"] is False
    assert effective["lr_database"] is None


@pytest.mark.parametrize("name", ("zebrafish", "mosta", "arista", "admouse"))
def test_canonical_current_checkpoint_configs_match_requested_preset(name):
    config, _ = load_workflow_config(name)
    import CytoBridge.workflow as workflow

    expected_training = workflow._read_training_config(config["train"]["config"])
    expected_training["ckpt_dir"] = "/copied/canonical/model"
    expected_training["spatial_dim"] = 2
    expected_training["model"]["spatial_dim"] = 2
    if config["train"].get("requires_edge_predictor", False):
        expected_training["model"]["interaction_net"][
            "edge_predictor_path"
        ] = "/copied/canonical/edge.pt"
    loaded = SimpleNamespace(
        config=expected_training,
        weight_stage="Finetune",
        score_stage="Score_Refine",
    )
    contract = _loaded_model_scientific_contract(
        loaded,
        config=config,
        options=WorkflowOptions(),
    )

    assert contract["status"] == "matches requested preset"
    assert contract["alpha_express"] == 0.015
    expected_threshold = config["train"].get("edge_predictor_threshold")
    assert contract["edge_predictor_threshold"] == expected_threshold


def test_checkpoint_without_explicit_learned_mode_uses_historical_default():
    config, _ = load_workflow_config("zebrafish")
    import CytoBridge.workflow as workflow

    checkpoint = deepcopy(workflow._read_training_config(config["train"]["config"]))
    checkpoint["model"]["interaction_net"].pop("edge_prior_mode")
    checkpoint["ckpt_dir"] = "/copied/canonical/model"
    checkpoint["spatial_dim"] = 2
    checkpoint["model"]["spatial_dim"] = 2
    checkpoint["model"]["interaction_net"][
        "edge_predictor_path"
    ] = "/copied/canonical/edge.pt"
    loaded = SimpleNamespace(
        config=checkpoint,
        weight_stage="Finetune",
        score_stage="Score_Refine",
    )

    contract = _loaded_model_scientific_contract(
        loaded,
        config=config,
        options=WorkflowOptions(),
    )

    assert contract["edge_prior_mode"] == "learned"


def _zebrafish_checkpoint_config(*, threshold: float, edge_path: str) -> dict:
    import CytoBridge.workflow as workflow

    config, _ = load_workflow_config("zebrafish")
    checkpoint = deepcopy(workflow._read_training_config(config["train"]["config"]))
    checkpoint["ckpt_dir"] = "/runs/zebrafish/training"
    checkpoint["spatial_dim"] = 2
    checkpoint["model"]["spatial_dim"] = 2
    interaction = checkpoint["model"]["interaction_net"]
    interaction["edge_predictor_path"] = edge_path
    interaction["edge_predictor_thre"] = threshold
    return checkpoint


def test_standalone_downstream_accepts_validation_selected_checkpoint_threshold():
    config, _ = load_workflow_config("zebrafish")
    selected_threshold = 0.6063615679740906
    loaded = SimpleNamespace(
        config=_zebrafish_checkpoint_config(
            threshold=selected_threshold,
            edge_path="/runs/zebrafish/preprocess/edge_classifier/zebrafish_edge_model.pt",
        ),
        weight_stage="Finetune",
        score_stage="Score_Refine",
    )

    contract = _loaded_model_scientific_contract(
        loaded,
        config=config,
        options=WorkflowOptions(),
    )

    assert contract["edge_predictor_threshold"] == selected_threshold
    assert contract["edge_predictor_threshold_check"] == (
        "loaded checkpoint recorded effective threshold"
    )


def test_standalone_downstream_rejects_conflicting_explicit_threshold():
    config, _ = load_workflow_config("zebrafish")
    loaded = SimpleNamespace(
        config=_zebrafish_checkpoint_config(
            threshold=0.6063615679740906,
            edge_path="/runs/zebrafish/preprocess/edge_classifier/zebrafish_edge_model.pt",
        ),
        weight_stage="Finetune",
        score_stage="Score_Refine",
    )

    with pytest.raises(ValueError, match="edge_predictor_thre") as error:
        _loaded_model_scientific_contract(
            loaded,
            config=config,
            options=WorkflowOptions(edge_predictor_threshold=0.42),
        )

    message = str(error.value)
    assert (
        "loaded model.interaction_net.edge_predictor_thre=0.6063615679740906" in message
    )
    assert "expected model.interaction_net.edge_predictor_thre=0.42" in message


def test_standalone_downstream_accepts_historical_preset_threshold():
    config, _ = load_workflow_config("zebrafish")
    historical_threshold = config["train"]["edge_predictor_threshold"]
    loaded = SimpleNamespace(
        config=_zebrafish_checkpoint_config(
            threshold=historical_threshold,
            edge_path="edge_classifier/zebrafish.pt",
        ),
        weight_stage="Finetune",
        score_stage="Score_Refine",
    )

    contract = _loaded_model_scientific_contract(
        loaded,
        config=config,
        options=WorkflowOptions(),
    )

    assert contract["edge_predictor_threshold"] == historical_threshold


@pytest.mark.parametrize(
    ("path", "replacement"),
    (
        (("training", "defaults", "lambda_mass"), 999.0),
        (("training", "defaults", "lambda_energy"), 999.0),
        (("training", "defaults", "global_mass"), False),
        (("reverse",), False),
        (("model", "velocity_net", "hidden_dim"), 17),
        (("training", "plan", 0, "OT_loss"), "sinkhorn"),
        (("training", "plan", 0, "lambda_ot"), 999.0),
        (("training", "plan", 0, "lambda_mass"), 999.0),
        (("training", "plan", 0, "lambda_energy"), 999.0),
        (("training", "plan", 0, "reverse_mass_norm"), False),
        (("training", "plan", 3, "optimizer_type"), "sgd"),
        (("training", "plan", 3, "scheduler_type"), "steplr"),
        (("training", "plan", 0, "mode"), "score_matching"),
        (("training", "plan", 0, "epochs"), 17),
        (("training", "plan", 0, "batch_size"), 17),
        (("training", "plan", 0, "train_strategy"), "v"),
    ),
)
def test_artifact_reuse_rejects_readable_scientific_config_mismatch(
    path,
    replacement,
):
    config, _ = load_workflow_config("admouse")
    import CytoBridge.workflow as workflow

    altered = deepcopy(workflow._read_training_config(config["train"]["config"]))
    target = altered
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = replacement
    loaded = SimpleNamespace(
        config=altered,
        weight_stage="Finetune",
        score_stage="Score_Refine",
    )

    with pytest.raises(ValueError) as error:
        _loaded_model_scientific_contract(
            loaded,
            config=config,
            options=WorkflowOptions(),
        )

    readable_path = ""
    for key in path:
        readable_path += (
            f"[{key}]"
            if isinstance(key, int)
            else (("." if readable_path else "") + key)
        )
    message = str(error.value)
    assert readable_path in message
    assert f"loaded {readable_path}={replacement!r}" in message


def test_all_spatial_artifact_reuse_rejects_inert_predictor_settings():
    config, _ = load_workflow_config("admouse")
    import CytoBridge.workflow as workflow

    no_lr_config = "admouse_spatial_full_alpha_express_0015_no_lr_prior.yaml"
    altered = deepcopy(workflow._read_training_config(no_lr_config))
    altered["ckpt_dir"] = "/copied/model"
    altered["device"] = "cpu"
    altered["training"]["history_flush_every"] = 1
    loaded = SimpleNamespace(
        config=altered,
        weight_stage="Finetune",
        score_stage="Score_Refine",
    )

    contract = _loaded_model_scientific_contract(
        loaded,
        config=config,
        options=WorkflowOptions(training_config=no_lr_config),
    )
    assert contract["status"] == "matches requested preset"

    altered["model"]["interaction_net"]["edge_predictor_path"] = "/copied/edge.pt"
    loaded.config = altered
    with pytest.raises(ValueError, match="records predictor settings"):
        _loaded_model_scientific_contract(
            loaded,
            config=config,
            options=WorkflowOptions(training_config=no_lr_config),
        )


def test_artifact_reuse_validates_derived_nondefault_spatial_dimension(tmp_path):
    config, _ = load_workflow_config("admouse")
    config = deepcopy(config)
    config["preprocess"]["align"]["spatial_dim"] = 3
    import CytoBridge.workflow as workflow

    training = deepcopy(workflow._read_training_config(config["train"]["config"]))
    training["spatial_dim"] = 3
    training["model"]["spatial_dim"] = 3
    training_path = tmp_path / "three_dimensional.yaml"
    import yaml

    training_path.write_text(yaml.safe_dump(training), encoding="utf-8")
    options = WorkflowOptions(training_config=str(training_path))
    loaded = SimpleNamespace(
        config=deepcopy(training),
        weight_stage="Finetune",
        score_stage="Score_Refine",
    )
    assert (
        _loaded_model_scientific_contract(
            loaded,
            config=config,
            options=options,
        )["status"]
        == "matches requested preset"
    )

    loaded.config["spatial_dim"] = 2
    with pytest.raises(ValueError, match="conflicting derived spatial dimensions"):
        _loaded_model_scientific_contract(
            loaded,
            config=config,
            options=options,
        )


@pytest.mark.parametrize("expected_spatial_dim", (0, 3))
def test_historical_checkpoint_default_two_dimensional_semantics_are_not_reused_as_nondefault(
    tmp_path,
    expected_spatial_dim,
):
    config, _ = load_workflow_config("admouse")
    config = deepcopy(config)
    config["dataset"]["concat_spatial"] = bool(expected_spatial_dim)
    config["preprocess"]["align"]["spatial_dim"] = expected_spatial_dim
    import CytoBridge.workflow as workflow
    import yaml

    training = deepcopy(workflow._read_training_config(config["train"]["config"]))
    training_path = tmp_path / f"spatial_{expected_spatial_dim}.yaml"
    training_path.write_text(yaml.safe_dump(training), encoding="utf-8")
    loaded = SimpleNamespace(
        config=deepcopy(training),
        weight_stage="Finetune",
        score_stage="Score_Refine",
    )

    with pytest.raises(ValueError, match="derived spatial dimension"):
        _loaded_model_scientific_contract(
            loaded,
            config=config,
            options=WorkflowOptions(training_config=str(training_path)),
        )


@pytest.mark.parametrize("spatial_dim", (0, 3))
def test_package_downstream_fails_closed_before_non_two_dimensional_analysis(
    monkeypatch,
    tmp_path,
    spatial_dim,
):
    import anndata as ad

    adata = ad.AnnData(X=np.zeros((2, 1), dtype=np.float32))
    adata.obsm["X_latent"] = np.zeros((2, 1), dtype=np.float32)
    if spatial_dim:
        adata.obsm["spatial_aligned"] = np.zeros((2, spatial_dim), dtype=np.float32)
    adata.obs["time_point_processed"] = [0.0, 1.0]
    adata.obs["Annotation"] = ["A", "A"]
    config, _ = load_workflow_config("admouse")
    config = deepcopy(config)
    config["dataset"]["concat_spatial"] = bool(spatial_dim)
    monkeypatch.setattr("anndata.read_h5ad", lambda _path: adata)

    with pytest.raises(ValueError, match=f"got {spatial_dim}"):
        _run_downstream(
            config,
            WorkflowOptions(),
            aligned_h5ad=tmp_path / "aligned.h5ad",
            model_dir=tmp_path / "model",
            output_dir=tmp_path / "out",
        )


def test_loaded_checkpoint_requires_the_complete_six_stage_contract():
    config, _ = load_workflow_config("admouse")
    import CytoBridge.workflow as workflow

    incomplete = workflow._read_training_config(config["train"]["config"])
    incomplete["training"]["plan"] = incomplete["training"]["plan"][:-1]
    loaded = SimpleNamespace(
        config=incomplete,
        weight_stage="Finetune",
        score_stage=None,
    )
    with pytest.raises(ValueError, match="training stages do not match"):
        _loaded_model_scientific_contract(
            loaded,
            config=config,
            options=WorkflowOptions(),
        )


def _historical_pca_reference(*, consistent: bool = True):
    import anndata as ad

    expression = np.asarray(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        dtype=np.float32,
    )
    reference = ad.AnnData(X=expression)
    reference.var_names = ["GeneA", "GeneB"]
    reference.varm["PCs"] = np.eye(2, dtype=np.float32)
    reference.obsm["X_pca"] = expression - expression.mean(axis=0, keepdims=True)
    if not consistent:
        reference.obsm["X_pca"][0, 0] += 1.0
    return reference


def test_historical_reference_accepts_verified_complete_x_center():
    reference = _historical_pca_reference()
    _require_pca_reference(
        reference,
        allow_complete_reference_pca_center_fallback=True,
    )


def test_historical_reference_without_center_fails_closed_by_default():
    with pytest.raises(
        ValueError,
        match="allow_complete_reference_pca_center_fallback=True",
    ):
        _require_pca_reference(_historical_pca_reference())


def test_historical_reference_rejects_subset_or_inconsistent_center():
    complete = _historical_pca_reference()
    subset = complete[:2].copy()
    for reference in (subset, _historical_pca_reference(consistent=False)):
        with pytest.raises(ValueError, match="original complete reference H5AD"):
            _require_pca_reference(
                reference,
                allow_complete_reference_pca_center_fallback=True,
            )


def test_missing_center_nullspace_ambiguity_still_fails_closed_by_default():
    import anndata as ad

    # The third feature lies in the loading nullspace. Its subset mean can move
    # by 50 while reconstructed PCA scores remain identical, so score residuals
    # alone cannot prove that an arbitrary object is the complete fit reference.
    expression = np.asarray([[1.0, 2.0, 0.0], [3.0, 4.0, 100.0]], dtype=np.float32)
    reference = ad.AnnData(X=expression)
    reference.varm["PCs"] = np.asarray(
        [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], dtype=np.float32
    )
    center = expression.mean(axis=0, keepdims=True)
    reference.obsm["X_pca"] = (expression - center) @ reference.varm["PCs"]

    with pytest.raises(
        ValueError,
        match="allow_complete_reference_pca_center_fallback=True",
    ):
        _require_pca_reference(reference)


def test_velocity_is_recomputed_per_slice_and_exports_all_components(tmp_path: Path):
    calls = []
    components = {
        "times": np.asarray([0.0, 0.0, 1.0, 1.0]),
        "features": np.asarray(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 2.0],
                [0.0, 1.0, 3.0],
                [1.0, 1.0, 4.0],
            ]
        ),
        "drift": np.ones((4, 3)),
        "interaction": np.full((4, 3), 2.0),
        "score": np.full((4, 3), 3.0),
        "full": np.full((4, 3), 6.0),
    }

    def compute(_adata, _model, **kwargs):
        assert kwargs["reuse_if_present"] is False
        assert kwargs["write_to_adata"] is False
        assert kwargs["interaction_threshold"] == 0.25
        return components

    def plot(**kwargs):
        calls.append(kwargs)

    cb = SimpleNamespace(
        tl=SimpleNamespace(compute_velocity_components_from_adata=compute),
        pl=SimpleNamespace(plot_velocity_component=plot),
    )
    adata = SimpleNamespace(
        obs=pd.DataFrame({"Annotation": ["A", "B", "A", "B"]}),
        obsm={"spatial_aligned": np.zeros((4, 2))},
    )
    model = SimpleNamespace(interaction_net=SimpleNamespace(cutoff=0.25))
    result = _write_velocity_outputs(
        cb=cb,
        adata=adata,
        model=model,
        dataset={
            "time_key": "time",
            "obsm_key": "X_latent",
            "spatial_key": "spatial_aligned",
            "concat_spatial": True,
        },
        annotation_key="Annotation",
        label_to_color={"A": "#000000", "B": "#ffffff"},
        output_dir=tmp_path,
        device="cpu",
    )

    assert result["status"] == "completed"
    assert len(result["figures"]) == 6
    assert {call["title"].split(" (")[0] for call in calls} == {
        "Intrinsic velocity",
        "Interaction velocity",
        "Full velocity",
    }
    with np.load(tmp_path / "velocity_components.npz") as archive:
        np.testing.assert_allclose(archive["full"], components["full"])


def test_communication_uses_prewarp_states_and_writes_tidy_table(tmp_path: Path):
    prewarp = {"0.0": object(), "1.0": object()}
    communication = {
        key: {
            "types": np.asarray(["A", "B"]),
            "M_per_source": np.asarray([[0.0, 1.0], [2.0, 0.0]]),
        }
        for key in prewarp
    }

    def compute(**kwargs):
        assert kwargs["adata_dict"] is prewarp
        assert kwargs["save_dense_attention_matrix"] is False
        return communication

    cb = SimpleNamespace(tl=SimpleNamespace(compute_timepoint_communications=compute))
    result = SimpleNamespace(
        communication_adata_dict=prewarp,
        ts_points=[0.0, 1.0],
    )
    summary, returned = _write_communication_outputs(
        cb=cb,
        result=result,
        runtime=SimpleNamespace(f_net=object()),
        annotation_key="Annotation",
        output_dir=tmp_path,
        device="cpu",
        downstream={},
        seed=42,
    )

    assert returned is communication
    assert summary["representation"] == "sparse model-edge attention"
    table = pd.read_csv(tmp_path / "communication_by_celltype.csv")
    assert table.shape == (8, 4)
    assert set(table.columns) == {
        "time",
        "source",
        "target",
        "attention_per_source",
    }


def test_composition_uses_real_observed_and_prewarp_generated_labels(tmp_path: Path):
    captured = {}

    def summarize(labels_by_time, time_points):
        captured["labels"] = [np.asarray(values) for values in labels_by_time]
        captured["times"] = list(time_points)
        return pd.DataFrame(
            {
                "time": [0.0, 1.0],
                "celltype": ["real", "generated"],
                "fraction": [1.0, 1.0],
            }
        )

    cb = SimpleNamespace(
        tl=SimpleNamespace(summarize_label_composition=summarize),
        pl=SimpleNamespace(plot_celltype_composition=lambda *args, **kwargs: None),
    )
    result = SimpleNamespace(
        communication_adata_dict={
            "0.0": SimpleNamespace(obs=pd.DataFrame({"Annotation": ["real"]})),
            "1.0": SimpleNamespace(obs=pd.DataFrame({"Annotation": ["generated"]})),
        },
        time_keys=["0.0", "1.0"],
        ts_points=[0.0, 1.0],
        # This deliberately disagrees with the observed label and must not win.
        slice_labels_split=[np.asarray(["simulated"]), np.asarray(["generated"])],
    )

    output = _write_composition_outputs(
        cb=cb,
        result=result,
        annotation_key="Annotation",
        label_to_color={"real": "#000000", "generated": "#ffffff"},
        output_dir=tmp_path,
    )

    assert output["status"] == "completed"
    assert captured["labels"][0].tolist() == ["real"]
    assert captured["labels"][1].tolist() == ["generated"]
    assert captured["times"] == [0.0, 1.0]


def test_3d_figure_omits_lineage_when_fixed_particle_labels_are_absent(
    tmp_path: Path,
):
    captured = {}

    def plot_3d(**kwargs):
        captured.update(kwargs)

    def unexpected_lineage(**kwargs):
        raise AssertionError("lineage plot should not run")

    def save_snapshots(**kwargs):
        return None

    cb = SimpleNamespace(
        tl=SimpleNamespace(
            save_timepoint_snapshots=save_snapshots,
            plot_lineage_sankey=unexpected_lineage,
            plot_spatiotemporal_3d=plot_3d,
        )
    )
    result = SimpleNamespace(
        adata_dict={"0.0": object(), "1.0": object()},
        time_keys=["0.0", "1.0"],
        ts_points=[0.0, 1.0],
        plot_3d_time_keys=["0.0", "1.0"],
        plot_3d_ts_points=[0.0, 1.0],
        observed_time_points=[0.0, 1.0],
        interp_points=[],
        predicted_labels_list=None,
    )
    output = _write_standard_figures(
        cb=cb,
        result=result,
        communications={"0.0": {}, "1.0": {}},
        annotation_key="Annotation",
        label_to_color={"A": "#000000"},
        output_dir=tmp_path,
        lineage_enabled=False,
    )

    assert output["lineage"]["status"] == "not applicable"
    assert output["spatiotemporal_3d"]["lineage_ribbons"] == "omitted"
    assert captured["predicted_labels_list"] is None
    assert captured["ribbon_render_mode"] == "none"


def test_3d_renderer_accepts_no_lineage_when_ribbons_are_disabled(tmp_path: Path):
    from CytoBridge.pl.spatiotemporal_sankey import plot_3d_spatial_sankey_style

    class Slice:
        def __init__(self, x):
            self.obs = pd.DataFrame({"Annotation": ["A", "A"]})
            self.obsm = {"spatial": np.asarray([[x, 0.0], [x, 1.0]])}

    figure = plot_3d_spatial_sankey_style(
        adata_dict={"0.0": Slice(0.0), "1.0": Slice(1.0)},
        all_time_communications={
            "0.0": {"types": ["A"], "M_per_source": np.asarray([[0.0]])},
            "1.0": {"types": ["A"], "M_per_source": np.asarray([[0.0]])},
        },
        time_keys=["0.0", "1.0"],
        label_to_color={"A": "#336699"},
        predicted_labels_list=None,
        ribbon_render_mode="none",
        edge_render_mode="none",
        show_legend=False,
        show_title=False,
        out_html=None,
    )

    assert figure is not None


def test_lr_requires_exact_pca_and_freezes_strict_subunit_contract(tmp_path: Path):
    try:
        _require_pca_reference(_Reference(with_pca=False))
    except KeyError as error:
        assert "exact PCA loadings" in str(error)
    else:
        raise AssertionError("missing PCA metadata was accepted")

    captured = {}
    tables = {
        name: pd.DataFrame({"value": [1.0]})
        for name in (
            "pair_timecourse",
            "celltype_timecourse",
            "pattern_summary",
            "coverage",
            "trajectory_coverage",
            "dropped_trajectories",
        )
    }

    def project(*args, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(**tables)

    cb = SimpleNamespace(
        tl=SimpleNamespace(project_communication_to_lr_timecourses=project)
    )
    lr_database = tmp_path / "lr.csv"
    lr_database.write_text("ligand,receptor\nGeneA,GeneB\n", encoding="utf-8")
    communications = {"0.0": {"M_per_source": np.ones((1, 1))}}
    result = SimpleNamespace(
        communication_adata_dict={"0.0": object()},
        ts_points=[0.0],
        observed_time_points=[0.0],
    )
    output = _write_lr_outputs(
        cb=cb,
        result=result,
        reference_adata=_Reference(),
        communications=communications,
        lr_database=lr_database,
        lr_complex_mode="min",
        preferred_species_tag=None,
        annotation_key="Annotation",
        resolved_time_key="time",
        spatial_dim=2,
        output_dir=tmp_path / "out",
    )

    assert captured["complex_mode"] == "min"
    assert captured["expression_space"] == "log1p"
    assert captured["require_all_subunits"] is True
    assert captured["observed_time_points"] == [0.0]
    assert output["require_all_subunits"] is True
    assert all(Path(path).is_file() for path in output["tables"].values())


def test_reconstruction_output_is_explicitly_not_a_holdout(tmp_path: Path):
    captured = []

    def evaluate(**kwargs):
        captured.append(kwargs)
        return pd.DataFrame(
            {
                "projection_sha256": ["unused"],
                "space": ["joint"],
                "primary_value": [0.1],
            }
        )

    cb = SimpleNamespace(
        tl=SimpleNamespace(
            fit_frozen_benchmark_transform=lambda state, spatial: object(),
            evaluate_spatiotemporal_prediction=evaluate,
        )
    )
    frame = pd.DataFrame(
        {
            "x1": [0.0, 1.0, 0.2, 1.2],
            "x2": [0.0, 0.0, 0.1, 0.1],
            "x3": [2.0, 3.0, 2.1, 3.1],
            "samples": [0.0, 0.0, 1.0, 1.0],
        }
    )
    result = SimpleNamespace(
        observed_time_points=[0.0, 1.0],
        ts_points=[0.0, 1.0],
        sde_points_split=np.asarray(
            [
                np.asarray([[0.0, 0.0, 2.0], [1.0, 0.0, 3.0]]),
                np.asarray([[0.1, 0.1, 2.2], [1.1, 0.1, 3.2]]),
            ],
            dtype=object,
        ),
    )
    summary = _write_reconstruction_diagnostic(
        cb=cb,
        result=result,
        dataframe=frame,
        feature_columns=["x1", "x2", "x3"],
        spatial_dim=2,
        dataset_name="toy",
        output_dir=tmp_path,
        downstream={
            "reconstruction_diagnostic": {
                "n_projections": 8,
                "projection_repeats": 1,
                "max_ot_points": 16,
            }
        },
    )

    assert "not a training holdout" in summary["claim"]
    assert captured[0]["benchmark"] == "toy_fitted_reconstruction"
    assert captured[0]["method"] == "CytoBridge fitted model"
    table = pd.read_csv(summary["table"])
    assert "projection_sha256" not in table.columns
