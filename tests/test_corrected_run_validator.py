from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import runpy
import sys
import tempfile

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = runpy.run_path(str(ROOT / "scripts" / "validate_corrected_de_novo_run.py"))
DATASETS = VALIDATOR["DATASETS"]
ABLATION_PROFILES = VALIDATOR["ABLATION_PROFILES"]
VALIDATION_PROFILES = VALIDATOR["VALIDATION_PROFILES"]
NO_INTERACTION_STAGES = VALIDATOR["NO_INTERACTION_STAGES"]
required_files = VALIDATOR["required_files"]
classifier_split_contract = VALIDATOR["classifier_split_contract"]
complete_lr_pair_time_grid = VALIDATOR["complete_lr_pair_time_grid"]
communication_edge_selection_contract = VALIDATOR[
    "communication_edge_selection_contract"
]
downstream_scope_contract = VALIDATOR["downstream_scope_contract"]
downstream_model_contract = VALIDATOR["downstream_model_contract"]
downstream_analysis_status_contract = VALIDATOR["downstream_analysis_status_contract"]
edge_threshold_provenance_contract = VALIDATOR["edge_threshold_provenance_contract"]
lr_database_provenance_contract = VALIDATOR["lr_database_provenance_contract"]
parse_args = VALIDATOR["parse_args"]
retained_top_level_lr_pairs = VALIDATOR["retained_top_level_lr_pairs"]
slice_provenance_summary_contract = VALIDATOR["slice_provenance_summary_contract"]
trained_edge_prior_contract = VALIDATOR["trained_edge_prior_contract"]
valid_color_map = VALIDATOR["valid_color_map"]
validate_slice = VALIDATOR["validate_slice"]
zebrafish_split_sde_contract = VALIDATOR["zebrafish_split_sde_contract"]
legacy_zero_daughter_noise_provenance = VALIDATOR[
    "legacy_zero_daughter_noise_provenance"
]
validate_no_lr_ablation_artifact_cleanliness = VALIDATOR[
    "validate_no_lr_ablation_artifact_cleanliness"
]
no_interaction_downstream_artifact_cleanliness = VALIDATOR[
    "no_interaction_downstream_artifact_cleanliness"
]
velocity_component_contract = VALIDATOR["velocity_component_contract"]
canonical_matched_config_contract = VALIDATOR["canonical_matched_config_contract"]
edge_predictor_artifact_contract = VALIDATOR["edge_predictor_artifact_contract"]
matched_config_only_delta_contract = VALIDATOR["matched_config_only_delta_contract"]
retained_checkpoint_contract = VALIDATOR["retained_checkpoint_contract"]
training_summary_embedding_contract = VALIDATOR["training_summary_embedding_contract"]
validate_matched_family = VALIDATOR["validate_matched_family"]
Audit = VALIDATOR["Audit"]
main = VALIDATOR["main"]
_canonical_array_identity = VALIDATOR["_canonical_array_identity"]
_comparable_global_rng = VALIDATOR["_comparable_global_rng"]
_expected_matched_ablation = VALIDATOR["_expected_matched_ablation"]
_matched_environment_signature = VALIDATOR["_matched_environment_signature"]
_ordered_obs_names_identity = VALIDATOR["_ordered_obs_names_identity"]
_training_implementation_identity = VALIDATOR["_training_implementation_identity"]
_valid_global_rng_record = VALIDATOR["_valid_global_rng_record"]


def test_corrected_run_validator_requires_predictor_for_all_four_main_runs(
    tmp_path: Path,
) -> None:
    admouse_paths = required_files(tmp_path / "admouse", "admouse", DATASETS["admouse"])
    zebrafish_paths = required_files(
        tmp_path / "zebrafish", "zebrafish", DATASETS["zebrafish"]
    )

    assert "generated edge model" in admouse_paths
    assert "generated edge metadata" in admouse_paths
    assert "generated edge model" in zebrafish_paths
    assert "generated edge metadata" in zebrafish_paths


def test_legacy_zero_daughter_noise_exception_requires_exact_provenance(
    tmp_path: Path, monkeypatch
) -> None:
    acceptance_path = tmp_path / "acceptance-4d53ec9.json"
    acceptance_path.write_text(
        json.dumps(
            {
                "status": "PASS",
                "datasets": {"zebrafish": {"status": "PASS"}},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    acceptance_sha = hashlib.sha256(acceptance_path.read_bytes()).hexdigest()
    globals_dict = legacy_zero_daughter_noise_provenance.__globals__
    monkeypatch.setitem(
        globals_dict, "LEGACY_ZERO_DAUGHTER_NOISE_RELEASE", "test-release"
    )
    monkeypatch.setitem(
        globals_dict,
        "LEGACY_ZERO_DAUGHTER_NOISE_ACCEPTANCE_SHA256",
        acceptance_sha,
    )
    (tmp_path / "final-manifest-4d53ec9.json").write_text(
        json.dumps(
            {
                "release_commit": "test-release",
                "canonical_run_root": str(tmp_path),
                "acceptance_report": acceptance_path.name,
                "acceptance_report_sha256": acceptance_sha,
                "acceptance_status": "PASS",
            }
        ),
        encoding="utf-8",
    )
    manifest_sha = hashlib.sha256(
        (tmp_path / "final-manifest-4d53ec9.json").read_bytes()
    ).hexdigest()
    monkeypatch.setitem(
        globals_dict,
        "LEGACY_ZERO_DAUGHTER_NOISE_MANIFEST_SHA256",
        manifest_sha,
    )

    valid, _ = legacy_zero_daughter_noise_provenance(tmp_path)
    assert valid

    acceptance_path.write_text('{"status":"FAIL"}', encoding="utf-8")
    valid, _ = legacy_zero_daughter_noise_provenance(tmp_path)
    assert not valid


def test_corrected_run_validator_locks_all_four_validation_selected_thresholds() -> (
    None
):
    assert {
        dataset: spec["edge_predictor_threshold"] for dataset, spec in DATASETS.items()
    } == {
        "zebrafish": 0.6063615679740906,
        "mosta": 0.1192110925912857,
        "arista": 0.5884028673171997,
        "admouse": 0.9956824779510498,
    }


@pytest.mark.parametrize("dataset", tuple(DATASETS))
def test_each_production_threshold_requires_the_frozen_validation_selection(
    dataset: str,
) -> None:
    expected = DATASETS[dataset]["edge_predictor_threshold"]
    workflow_config = json.loads(
        (ROOT / "CytoBridge" / "workflow_configs" / f"{dataset}.json").read_text(
            encoding="utf-8"
        )
    )
    assert workflow_config["train"]["requires_edge_predictor"] is True
    assert workflow_config["train"]["edge_predictor_threshold"] == expected
    assert (
        workflow_config["train"]["graph_database"]
        == DATASETS[dataset]["lr_database_name"]
    )
    metadata = {
        "edge_predictor_threshold": expected,
        "edge_predictor_threshold_selected": expected,
        "selection_source": "validation",
    }
    valid, _ = edge_threshold_provenance_contract(
        metadata,
        expected_threshold=expected,
    )
    assert valid

    metadata["edge_predictor_threshold_selected"] = expected / 2.0
    valid, _ = edge_threshold_provenance_contract(
        metadata,
        expected_threshold=expected,
    )
    assert not valid


def test_admouse_no_lr_prior_is_an_explicit_reachable_cli_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = VALIDATION_PROFILES["admouse_no_lr_prior"]
    run_dir = tmp_path / "admouse_no_lr_prior"
    paths = required_files(run_dir, "admouse_no_lr_prior", spec)

    assert spec["artifact_dataset"] == "admouse"
    assert spec["edge_prior_mode"] == "all_spatial"
    assert spec["training_config"] == (
        "admouse_spatial_full_alpha_express_0015_no_lr_prior.yaml"
    )
    packaged_profile = yaml.safe_load(
        (ROOT / "CytoBridge" / "configs" / spec["training_config"]).read_text(
            encoding="utf-8"
        )
    )
    interaction = packaged_profile["model"]["interaction_net"]
    assert interaction["edge_prior_mode"] == "all_spatial"
    assert "edge_predictor_path" not in interaction
    assert "edge_predictor_thre" not in interaction
    assert paths["aligned H5AD"] == run_dir / "preprocess" / "admouse_aligned.h5ad"
    assert "generated edge model" not in paths
    assert "generated edge metadata" not in paths

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_corrected_de_novo_run.py",
            "--run-root",
            str(tmp_path),
            "--datasets",
            "admouse_no_lr_prior",
        ],
    )
    args = parse_args()
    assert args.datasets == ["admouse_no_lr_prior"]


@pytest.mark.parametrize("dataset", tuple(DATASETS))
@pytest.mark.parametrize("ablation", ("no_lr_prior", "no_interaction"))
def test_all_eight_matched_ablation_profiles_are_explicit_and_packaged(
    dataset: str,
    ablation: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile_name = f"{dataset}_{ablation}"
    spec = ABLATION_PROFILES[profile_name]
    config_path = ROOT / "CytoBridge" / "configs" / spec["training_config"]
    packaged = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model = packaged["model"]
    stages = tuple(stage["name"] for stage in packaged["training"]["plan"])
    paths = required_files(tmp_path / profile_name, profile_name, spec)

    assert spec["artifact_dataset"] == dataset
    assert spec["run_role"] == (
        "no-LR-prior ablation"
        if ablation == "no_lr_prior"
        else "no-interaction ablation"
    )
    assert config_path.is_file()
    assert "generated edge model" not in paths
    assert "generated edge metadata" not in paths
    assert paths["aligned H5AD"].name == f"{dataset}_aligned.h5ad"
    if ablation == "no_lr_prior":
        assert spec["interaction_component"] is True
        assert spec["edge_prior_mode"] == "all_spatial"
        assert stages == VALIDATOR["STAGES"]
        assert set(model["components"]) == {
            "velocity",
            "growth",
            "score",
            "interaction",
        }
        assert model["interaction_net"]["edge_prior_mode"] == "all_spatial"
        assert "edge_predictor_path" not in model["interaction_net"]
        assert "edge_predictor_thre" not in model["interaction_net"]
    else:
        assert spec["interaction_component"] is False
        assert spec["edge_prior_mode"] == "none"
        assert stages == NO_INTERACTION_STAGES
        assert set(model["components"]) == {"velocity", "growth", "score"}
        assert "interaction_net" not in model
        assert "interaction_type" not in model
        assert "interaction_group_size" not in model

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_corrected_de_novo_run.py",
            "--run-root",
            str(tmp_path),
            "--datasets",
            profile_name,
        ],
    )
    assert parse_args().datasets == [profile_name]


def test_main_validation_profiles_remain_the_default_and_learned() -> None:
    assert tuple(DATASETS) == ("zebrafish", "mosta", "arista", "admouse")
    assert tuple(VALIDATION_PROFILES)[:4] == tuple(DATASETS)
    assert all(DATASETS[name]["edge_prior_mode"] == "learned" for name in DATASETS)


def test_matched_family_cli_is_repeatable_and_does_not_implicitly_add_arms(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_corrected_de_novo_run.py",
            "--run-root",
            str(tmp_path),
            "--datasets",
            "zebrafish",
            "zebrafish_no_lr_prior",
            "zebrafish_no_interaction",
            "mosta",
            "mosta_no_lr_prior",
            "mosta_no_interaction",
            "--matched-family",
            "zebrafish",
            "--matched-family",
            "mosta",
        ],
    )
    args = parse_args()
    assert args.matched_family == ["zebrafish", "mosta"]
    assert args.datasets == [
        "zebrafish",
        "zebrafish_no_lr_prior",
        "zebrafish_no_interaction",
        "mosta",
        "mosta_no_lr_prior",
        "mosta_no_interaction",
    ]


def test_matched_family_report_section_affects_overall_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = tmp_path / "acceptance.json"
    profiles = ["zebrafish", "zebrafish_no_lr_prior", "zebrafish_no_interaction"]

    def passing_dataset(_run_root: Path, profile: str) -> Audit:
        return Audit(profile)

    def failing_family(*_args, **_kwargs) -> Audit:
        audit = Audit("zebrafish")
        audit.check(False, "formal family", "intentional test failure")
        return audit

    monkeypatch.setitem(main.__globals__, "validate_dataset", passing_dataset)
    monkeypatch.setitem(main.__globals__, "validate_matched_family", failing_family)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_corrected_de_novo_run.py",
            "--run-root",
            str(tmp_path),
            "--datasets",
            *profiles,
            "--matched-family",
            "zebrafish",
            "--report",
            str(report),
        ],
    )
    assert main() == 1
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["status"] == "FAIL"
    assert payload["matched_families"]["zebrafish"]["status"] == "FAIL"


def test_environment_and_rng_comparison_ignore_only_physical_gpu_index() -> None:
    left_environment = _matched_environment()
    left_environment.update(
        {
            "device": "cuda:0",
            "device_type": "cuda",
            "cuda_compiled_version": "12.4",
            "cuda_available": True,
            "cuda_device_name": "NVIDIA RTX 4090",
            "cuda_device_index": 0,
            "cudnn_version": 90100,
        }
    )
    left_environment["dependency_versions"]["cuda"] = {
        "version": "12.4",
        "status": "compiled",
    }
    left_environment["dependency_versions"]["cudnn"] = {
        "version": 90100,
        "status": "available",
    }
    right_environment = deepcopy(left_environment)
    right_environment["device"] = "cuda:6"
    right_environment["cuda_device_index"] = 6
    left_signature = _matched_environment_signature(left_environment)
    right_signature = _matched_environment_signature(right_environment)
    assert left_signature is not None
    assert left_signature == right_signature

    def cuda_rng(index: int) -> dict:
        record = _matched_global_rng(7)
        record["torch_cuda"] = {
            "available": True,
            "visible_device_count": 8,
            "selected_device": {"index": index, "name": "NVIDIA RTX 4090"},
            "state_sha256": _sha("same selected-device state"),
            "snapshot_scope": "selected_training_device_only",
        }
        record["aggregate_sha256"] = hashlib.sha256(
            json.dumps(
                {
                    key: value
                    for key, value in record.items()
                    if key != "aggregate_sha256"
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        return record

    left_rng = cuda_rng(0)
    right_rng = cuda_rng(6)
    assert _valid_global_rng_record(left_rng)
    assert _valid_global_rng_record(right_rng)
    assert _comparable_global_rng(left_rng) == _comparable_global_rng(right_rng)


def test_main_model_contract_keeps_canonical_pre_component_field_compatibility() -> (
    None
):
    spec = DATASETS["zebrafish"]
    valid, _ = downstream_model_contract(
        {
            "weight_stage": "Finetune",
            "score_stage": "Score_Refine",
            "scientific_contract": {
                "status": "matches requested preset",
                "interaction_cutoff": spec["cutoff"],
                "edge_prior_mode": "learned",
                "edge_predictor_threshold": spec["edge_predictor_threshold"],
                "weight_stage": "Finetune",
                "score_stage": "Score_Refine",
                "interaction_group_size": 1024,
                "interaction_group_max_size": 2047,
                "interaction_group_remainder_policy": (
                    "merge a nonzero remainder into the final base-size group"
                ),
            },
        },
        {"interaction_group_size": 1024},
        spec=spec,
        expected_threshold=spec["edge_predictor_threshold"],
    )
    assert valid


@pytest.mark.parametrize("dataset", tuple(DATASETS))
def test_each_production_database_provenance_is_exact_and_role_aware(
    dataset: str,
) -> None:
    spec = DATASETS[dataset]
    database = ROOT / "CytoBridge" / "workflow_databases" / spec["lr_database_name"]
    assert database.is_file()
    assert (
        hashlib.sha256(database.read_bytes()).hexdigest() == spec["lr_database_sha256"]
    )

    graph = {
        "lr_database_path": str(database),
        "lr_matching_rule": (
            "selected_symbol_exact_case_insensitive_all_complex_subunits"
        ),
        "lr_complex_expression_rule": "minimum",
        "preferred_species_tag": spec["species"],
        "lr_unique_resolved_pairs": spec.get("strict_lr_pairs", 11),
    }
    downstream = {
        "database": str(database),
        "complex_mode": "min",
        "require_all_subunits": True,
        "preferred_species_tag": spec["species"],
    }
    valid, _ = lr_database_provenance_contract(
        graph_metadata=graph,
        graph_metadata_present=True,
        downstream_analysis=downstream,
        spec=spec,
    )
    assert valid

    wrong_database = (
        ROOT
        / "CytoBridge"
        / "workflow_databases"
        / (
            "CellChatDB.ligrec.human.csv"
            if spec["lr_database_name"] != "CellChatDB.ligrec.human.csv"
            else "CellChatDB.ligrec.mouse.csv"
        )
    )
    wrong_downstream = dict(downstream, database=str(wrong_database))
    valid, _ = lr_database_provenance_contract(
        graph_metadata=graph,
        graph_metadata_present=True,
        downstream_analysis=wrong_downstream,
        spec=spec,
    )
    assert not valid


def test_all_spatial_database_is_downstream_only_and_allows_declared_ignored_input_graph() -> (
    None
):
    spec = VALIDATION_PROFILES["admouse_no_lr_prior"]
    database = ROOT / "CytoBridge" / "workflow_databases" / spec["lr_database_name"]
    downstream = {
        "database": str(database),
        "complex_mode": "min",
        "require_all_subunits": True,
        "preferred_species_tag": spec["species"],
    }

    valid, _ = lr_database_provenance_contract(
        graph_metadata=None,
        graph_metadata_present=False,
        downstream_analysis=downstream,
        spec=spec,
    )
    assert valid

    valid, _ = lr_database_provenance_contract(
        graph_metadata={"lr_database_path": str(database)},
        graph_metadata_present=True,
        downstream_analysis=downstream,
        spec=spec,
    )
    assert not valid

    valid, _ = lr_database_provenance_contract(
        graph_metadata={
            "lr_database_path": str(database),
            "lr_matching_rule": (
                "selected_symbol_exact_case_insensitive_all_complex_subunits"
            ),
            "lr_complex_expression_rule": "minimum",
            "preferred_species_tag": spec["species"],
            "lr_unique_resolved_pairs": 7,
        },
        graph_metadata_present=True,
        downstream_analysis=downstream,
        spec=spec,
        allow_ignored_input_graph=True,
    )
    assert valid

    valid, _ = lr_database_provenance_contract(
        graph_metadata={"lr_database_path": str(database)},
        graph_metadata_present=True,
        downstream_analysis=downstream,
        spec=spec,
        allow_ignored_input_graph=True,
    )
    assert not valid


@pytest.mark.parametrize(
    "relative", ("edge_classifier/model.pt", "input_graph/t0.csv", "metadata/t0.csv")
)
def test_all_spatial_ablation_rejects_any_learned_graph_artifact(
    tmp_path: Path, relative: str
) -> None:
    run_dir = tmp_path / "admouse_no_lr_prior"
    artifact = run_dir / "preprocess" / relative
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"stale learned artifact")

    valid, detail = validate_no_lr_ablation_artifact_cleanliness(run_dir)
    assert not valid
    assert relative in detail


def test_all_spatial_ablation_accepts_clean_preprocess_tree(tmp_path: Path) -> None:
    run_dir = tmp_path / "admouse_no_lr_prior"
    aligned = run_dir / "preprocess" / "admouse_aligned.h5ad"
    aligned.parent.mkdir(parents=True)
    aligned.write_bytes(b"aligned")

    valid, detail = validate_no_lr_ablation_artifact_cleanliness(run_dir)
    assert valid
    assert detail == "none"


@pytest.mark.parametrize("profile_name", tuple(ABLATION_PROFILES))
@pytest.mark.parametrize(
    "relative", ("edge_classifier/model.pt", "input_graph/t0.csv", "metadata/t0.csv")
)
def test_every_ablation_rejects_generated_predictor_or_graph_artifacts(
    tmp_path: Path,
    profile_name: str,
    relative: str,
) -> None:
    run_dir = tmp_path / profile_name
    artifact = run_dir / "preprocess" / relative
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"forbidden generated artifact")

    valid, detail = validate_no_lr_ablation_artifact_cleanliness(run_dir)

    assert not valid
    assert relative in detail


def test_no_interaction_database_contract_only_allows_explicitly_ignored_main_graph() -> (
    None
):
    spec = VALIDATION_PROFILES["admouse_no_interaction"]
    database = ROOT / "CytoBridge" / "workflow_databases" / spec["lr_database_name"]
    graph = {
        "lr_database_path": str(database),
        "lr_matching_rule": (
            "selected_symbol_exact_case_insensitive_all_complex_subunits"
        ),
        "lr_complex_expression_rule": "minimum",
        "preferred_species_tag": spec["species"],
        "lr_unique_resolved_pairs": 7,
    }
    not_applicable = {
        "status": "not applicable",
        "reason": (
            "model has no interaction component; LR projection requires "
            "model-derived sparse communication attention"
        ),
        "analysis_scope": None,
    }

    valid, _ = lr_database_provenance_contract(
        graph_metadata=graph,
        graph_metadata_present=True,
        downstream_analysis=not_applicable,
        spec=spec,
        allow_ignored_input_graph=True,
    )
    assert valid

    valid, _ = lr_database_provenance_contract(
        graph_metadata=graph,
        graph_metadata_present=True,
        downstream_analysis=not_applicable,
        spec=spec,
        allow_ignored_input_graph=False,
    )
    assert not valid

    stale_projection = dict(not_applicable, database=str(database))
    valid, _ = lr_database_provenance_contract(
        graph_metadata=graph,
        graph_metadata_present=True,
        downstream_analysis=stale_projection,
        spec=spec,
        allow_ignored_input_graph=True,
    )
    assert not valid


def test_trained_learned_edge_prior_must_match_predictor_sidecar_and_graph(
    tmp_path: Path,
) -> None:
    edge_path = tmp_path / "edge.pt"
    edge_path.write_bytes(b"weights")
    threshold = 0.37
    all_model = {
        "edge_prior_mode": "learned",
        "edge_predictor_path": str(edge_path),
        "edge_predictor_threshold": threshold,
    }
    interaction_graph = {
        "edge_predictor_path": str(edge_path),
        "edge_predictor_model_path": str(edge_path),
        "edge_predictor_threshold": threshold,
        "edge_predictor_threshold_selected": threshold,
    }
    sidecar = {
        "edge_predictor_threshold": threshold,
        "edge_predictor_threshold_selected": threshold,
    }

    valid, _ = trained_edge_prior_contract(
        all_model,
        interaction_graph,
        expected_mode="learned",
        expected_threshold=threshold,
        expected_edge_path=edge_path,
        predictor_metadata=sidecar,
        interaction_graph_present=True,
    )
    assert valid

    stale_sidecar = dict(sidecar, edge_predictor_threshold=0.51)
    valid, _ = trained_edge_prior_contract(
        all_model,
        interaction_graph,
        expected_mode="learned",
        expected_threshold=threshold,
        expected_edge_path=edge_path,
        predictor_metadata=stale_sidecar,
        interaction_graph_present=True,
    )
    assert not valid


def test_trained_all_spatial_edge_prior_rejects_inert_or_stale_predictor_metadata() -> (
    None
):
    clean = {
        "edge_prior_mode": "all_spatial",
        "edge_predictor_path": None,
        "edge_predictor_threshold": None,
    }
    valid, _ = trained_edge_prior_contract(
        clean,
        None,
        expected_mode="all_spatial",
        expected_threshold=None,
        expected_edge_path=None,
        interaction_graph_present=False,
    )
    assert valid

    valid, _ = trained_edge_prior_contract(
        clean,
        {},
        expected_mode="all_spatial",
        expected_threshold=None,
        expected_edge_path=None,
        interaction_graph_present=True,
    )
    assert not valid

    inert = dict(clean, edge_predictor_threshold=0.5)
    valid, _ = trained_edge_prior_contract(
        inert,
        None,
        expected_mode="all_spatial",
        expected_threshold=None,
        expected_edge_path=None,
        interaction_graph_present=False,
    )
    assert not valid
    for key, value in (
        ("edge_predictor_size_bytes", 123),
        ("edge_predictor_sha256", _sha("stale predictor")),
    ):
        valid, _ = trained_edge_prior_contract(
            dict(clean, **{key: value}),
            None,
            expected_mode="all_spatial",
            expected_threshold=None,
            expected_edge_path=None,
            interaction_graph_present=False,
        )
        assert not valid


@pytest.mark.parametrize(
    "mutation",
    (
        "interaction_component",
        "interaction_net",
        "interaction_type",
        "interaction_group_size",
        "finite_cutoff",
        "predictor_path",
        "predictor_threshold",
        "predictor_size",
        "predictor_sha256",
        "trained_graph",
    ),
)
def test_trained_no_interaction_contract_rejects_every_inert_interaction_field(
    mutation: str,
) -> None:
    clean = {
        "model_config": {"components": ["velocity", "growth", "score"]},
        "interaction_cutoff": np.nan,
        "edge_prior_mode": None,
        "edge_predictor_path": None,
        "edge_predictor_threshold": None,
    }
    graph = None
    graph_present = False
    if mutation == "interaction_component":
        clean["model_config"]["components"].append("interaction")
    elif mutation in {"interaction_net", "interaction_type", "interaction_group_size"}:
        clean["model_config"][mutation] = {} if mutation == "interaction_net" else 1
    elif mutation == "finite_cutoff":
        clean["interaction_cutoff"] = 0.1
    elif mutation == "predictor_path":
        clean["edge_predictor_path"] = "/stale/edge.pt"
    elif mutation == "predictor_threshold":
        clean["edge_predictor_threshold"] = 0.5
    elif mutation == "predictor_size":
        clean["edge_predictor_size_bytes"] = 123
    elif mutation == "predictor_sha256":
        clean["edge_predictor_sha256"] = _sha("stale predictor")
    elif mutation == "trained_graph":
        graph = {}
        graph_present = True

    valid, _ = trained_edge_prior_contract(
        clean,
        graph,
        expected_mode="none",
        expected_threshold=None,
        expected_edge_path=None,
        interaction_graph_present=graph_present,
    )

    assert not valid


def test_trained_no_interaction_contract_accepts_only_clean_retained_components() -> (
    None
):
    valid, _ = trained_edge_prior_contract(
        {
            "model_config": {"components": ["velocity", "growth", "score"]},
            "interaction_cutoff": np.nan,
            "edge_prior_mode": None,
            "edge_predictor_path": None,
            "edge_predictor_threshold": None,
        },
        None,
        expected_mode="none",
        expected_threshold=None,
        expected_edge_path=None,
        interaction_graph_present=False,
    )
    assert valid


def _write_velocity_archive(path: Path, interaction: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    drift = np.asarray([[0.0, 1.0], [1.0, 2.0], [2.0, 4.0]], dtype=np.float64)
    score = np.asarray([[0.2, 0.0], [0.4, 0.1], [0.8, 0.2]], dtype=np.float64)
    np.savez_compressed(
        path,
        drift=drift,
        interaction=interaction,
        score=score,
        full=drift + interaction + score,
        times=np.asarray([0.0, 1.0, 2.0]),
        features=np.asarray([[1.0, 2.0], [2.0, 3.0], [3.0, 5.0]]),
    )


def test_no_interaction_velocity_requires_an_exact_zero_sentinel(
    tmp_path: Path,
) -> None:
    path = tmp_path / "velocity_components.npz"
    _write_velocity_archive(path, np.zeros((3, 2), dtype=np.float64))
    valid, _ = velocity_component_contract(
        path,
        expected_shape=(3, 2),
        interaction_component=False,
    )
    assert valid

    almost_zero = np.zeros((3, 2), dtype=np.float64)
    almost_zero[0, 0] = np.nextafter(0.0, 1.0)
    _write_velocity_archive(path, almost_zero)
    valid, _ = velocity_component_contract(
        path,
        expected_shape=(3, 2),
        interaction_component=False,
    )
    assert not valid


@pytest.mark.parametrize(
    "relative",
    (
        "communication/communication_by_celltype.csv",
        "ligand_receptor/pair_timecourse.csv",
        "velocity/interaction_time_0.pdf",
        "figures/spatiotemporal_communication_3d.html",
    ),
)
def test_no_interaction_downstream_rejects_any_forbidden_artifact(
    tmp_path: Path,
    relative: str,
) -> None:
    downstream = tmp_path / "downstream"
    downstream.mkdir()
    valid, detail = no_interaction_downstream_artifact_cleanliness(downstream)
    assert valid
    assert detail == "none"

    artifact = downstream / relative
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(b"stale or fabricated")
    valid, detail = no_interaction_downstream_artifact_cleanliness(downstream)
    assert not valid
    assert relative in detail


def _valid_classifier_split() -> dict:
    return {
        "cache_protocol_version": 8,
        "per_class_counts": {
            "Common": {"total": 10, "train": 8, "validation": 2},
            "Rare": {"total": 2, "train": 1, "validation": 1},
            "Singleton": {"total": 1, "train": 1, "validation": 0},
        },
        "training_only_singleton_classes": ["Singleton"],
        "singleton_class_policy": (
            "A class represented by one row is retained in training and excluded "
            "from validation."
        ),
    }


def test_classifier_split_contract_requires_protocol_and_support_for_every_class() -> (
    None
):
    valid, _ = classifier_split_contract(_valid_classifier_split())
    assert valid

    wrong_protocol = dict(_valid_classifier_split(), cache_protocol_version=7)
    valid, _ = classifier_split_contract(wrong_protocol)
    assert not valid

    missing_singleton_listing = dict(
        _valid_classifier_split(), training_only_singleton_classes=[]
    )
    valid, _ = classifier_split_contract(missing_singleton_listing)
    assert not valid

    missing_validation = _valid_classifier_split()
    missing_validation["per_class_counts"] = dict(
        missing_validation["per_class_counts"],
        Rare={"total": 2, "train": 2, "validation": 0},
    )
    valid, _ = classifier_split_contract(missing_validation)
    assert not valid

    valid, _ = classifier_split_contract(
        _valid_classifier_split(),
        expected_class_counts={"Common": 10, "Rare": 2, "Singleton": 1},
    )
    assert valid
    valid, _ = classifier_split_contract(
        _valid_classifier_split(),
        expected_class_counts={"Common": 10, "Rare": 2, "Singleton": 1, "Lost": 4},
    )
    assert not valid


def test_zebrafish_split_sde_contract_requires_resampling_and_particle_ceiling() -> (
    None
):
    slice_counts = {
        0.0: 563,
        0.5: 735,
        1.0: 1036,
        1.5: 1470,
        2.0: 2081,
        2.5: 2619,
        3.0: 3048,
        3.5: 3962,
        4.0: 5271,
    }
    observed_counts = DATASETS["zebrafish"]["observed_counts"]
    generated_counts = {
        time: count
        for time, count in slice_counts.items()
        if time not in observed_counts
    }
    simulation = {
        "trajectory_mode": ("piecewise_observed_anchored_interval_forward_simulation"),
        "split_sde_piecewise": True,
        "piecewise_observed_sample_mode": "per_timepoint",
        "piecewise_include_end": False,
        "trajectory_scope": (
            "Each observed slice is an observed anchor for interval-local, one-sided "
            "forward simulation. A generated slice is not conditioned on the following "
            "observed endpoint. This is not global-t0 extrapolation and not a "
            "lineage-continuous trajectory."
        ),
        "initial_particles": 563,
        "configured_particle_cap": None,
        "initial_particle_cap": None,
        "split_dt": 0.05,
        "split_resample_dt": 0.05,
        "sigma": 0.03,
        "daughter_noise_std": 0.0,
        "growth_alpha": 1.0,
        "split_particle_ceiling": 100_000,
        "non_split_lineage_rollout": False,
        "particle_counts_by_time": {
            str(time): count for time, count in slice_counts.items()
        },
        "observed_particle_counts": {
            str(time): count for time, count in observed_counts.items()
        },
        "generated_particle_counts": {
            str(time): count for time, count in generated_counts.items()
        },
    }
    analyses = {
        "communication": {"trajectory_scope": simulation["trajectory_scope"]},
        "ligand_receptor": {"trajectory_scope": simulation["trajectory_scope"]},
        "reconstruction_diagnostic": {
            "status": "not requested",
            "claim": "not applicable as a global-t0 reconstruction diagnostic",
        },
    }
    valid, _ = zebrafish_split_sde_contract(
        simulation,
        slice_counts=slice_counts,
        expected_observed_counts=observed_counts,
        analyses=analyses,
    )
    assert valid

    implicit_zero = dict(simulation)
    implicit_zero.pop("daughter_noise_std")
    valid, _ = zebrafish_split_sde_contract(
        implicit_zero,
        slice_counts=slice_counts,
        expected_observed_counts=observed_counts,
        analyses=analyses,
    )
    assert not valid
    valid, _ = zebrafish_split_sde_contract(
        implicit_zero,
        slice_counts=slice_counts,
        expected_observed_counts=observed_counts,
        analyses=analyses,
        allow_legacy_implicit_zero_daughter_noise=True,
    )
    assert valid

    invalid_cases = [
        dict(simulation, piecewise_observed_sample_mode="t0_fixed"),
        dict(simulation, piecewise_include_end=True),
        dict(simulation, trajectory_mode="global_t0_extrapolation"),
        dict(simulation, split_particle_ceiling=None),
    ]
    for invalid in invalid_cases:
        valid, _ = zebrafish_split_sde_contract(
            invalid,
            slice_counts=slice_counts,
            expected_observed_counts=observed_counts,
            analyses=analyses,
        )
        assert not valid

    wrong_observed = dict(slice_counts, **{})
    wrong_observed[1.0] = 1035
    valid, _ = zebrafish_split_sde_contract(
        simulation,
        slice_counts=wrong_observed,
        expected_observed_counts=observed_counts,
        analyses=analyses,
    )
    assert not valid

    above_ceiling = dict(slice_counts)
    above_ceiling[0.5] = 100_001
    valid, _ = zebrafish_split_sde_contract(
        simulation,
        slice_counts=above_ceiling,
        expected_observed_counts=observed_counts,
        analyses=analyses,
    )
    assert not valid


@pytest.mark.parametrize(
    ("mutation", "value"),
    (
        ("split_dt", 0.01),
        ("split_resample_dt", 0.1),
        ("sigma", 0.04),
        ("daughter_noise_std", 0.01),
        ("growth_alpha", 0.5),
        ("configured_particle_cap", 563),
        ("initial_particle_cap", 563),
        ("initial_particles", 564),
        ("non_split_lineage_rollout", True),
    ),
)
def test_zebrafish_split_sde_contract_locks_all_simulation_constants(
    mutation: str,
    value: object,
) -> None:
    observed_counts = DATASETS["zebrafish"]["observed_counts"]
    slice_counts = {
        0.0: 563,
        0.5: 735,
        1.0: 1036,
        1.5: 1470,
        2.0: 2081,
        2.5: 2619,
        3.0: 3048,
        3.5: 3962,
        4.0: 5271,
    }
    scope = (
        "Each observed slice is an observed anchor for interval-local, one-sided "
        "forward simulation. A generated slice is not conditioned on the following "
        "observed endpoint. This is not global-t0 extrapolation and not a "
        "lineage-continuous trajectory."
    )
    simulation = {
        "trajectory_mode": ("piecewise_observed_anchored_interval_forward_simulation"),
        "split_sde_piecewise": True,
        "piecewise_observed_sample_mode": "per_timepoint",
        "piecewise_include_end": False,
        "trajectory_scope": scope,
        "initial_particles": 563,
        "configured_particle_cap": None,
        "initial_particle_cap": None,
        "split_dt": 0.05,
        "split_resample_dt": 0.05,
        "sigma": 0.03,
        "daughter_noise_std": 0.0,
        "growth_alpha": 1.0,
        "split_particle_ceiling": 100_000,
        "non_split_lineage_rollout": False,
        "particle_counts_by_time": {
            str(time): count for time, count in slice_counts.items()
        },
        "observed_particle_counts": {
            str(time): count for time, count in observed_counts.items()
        },
        "generated_particle_counts": {
            str(time): count
            for time, count in slice_counts.items()
            if time not in observed_counts
        },
    }
    simulation[mutation] = value
    analyses = {
        "communication": {"trajectory_scope": scope},
        "ligand_receptor": {"trajectory_scope": scope},
        "reconstruction_diagnostic": {
            "claim": "not applicable as a global-t0 reconstruction diagnostic"
        },
    }

    valid, _ = zebrafish_split_sde_contract(
        simulation,
        slice_counts=slice_counts,
        expected_observed_counts=observed_counts,
        analyses=analyses,
    )

    assert not valid


def _provenance_aligned_adata() -> ad.AnnData:
    aligned = ad.AnnData(
        X=np.zeros((4, 2), dtype=np.float32),
        obs=pd.DataFrame(
            {
                "time_point_processed": [0.0, 0.0, 1.0, 1.0],
                "Annotation": ["A", "B", "A", "B"],
            },
            index=["cell0", "cell1", "cell2", "cell3"],
        ),
    )
    aligned.obsm["spatial_aligned"] = np.asarray(
        [[0.0, 0.1], [0.2, 0.3], [1.0, 1.1], [1.2, 1.3]], dtype=np.float32
    )
    aligned.obsm["X_latent"] = np.asarray(
        [[2.0], [3.0], [4.0], [5.0]], dtype=np.float32
    )
    return aligned


def _observed_slice(aligned: ad.AnnData, *, time_value: float = 0.0) -> ad.AnnData:
    mask = np.isclose(
        aligned.obs["time_point_processed"].to_numpy(dtype=float), time_value
    )
    subset = aligned[mask]
    state = np.hstack((subset.obsm["spatial_aligned"], subset.obsm["X_latent"])).astype(
        np.float32
    )
    result = ad.AnnData(
        X=state,
        obs=pd.DataFrame(
            {
                "Annotation": subset.obs["Annotation"].astype(str).to_numpy(),
                "source_obs_id": subset.obs_names.astype(str).to_numpy(),
            },
        ),
    )
    result.obsm["spatial"] = state[:, :2]
    result.uns["slice_origin"] = "observed_real"
    result.uns["source_anchor_time"] = float(time_value)
    return result


@pytest.mark.parametrize(
    "mutation",
    ("source_order", "annotation", "state", "origin", "anchor"),
)
def test_observed_slice_provenance_rejects_adversarial_mutations(
    tmp_path: Path,
    mutation: str,
) -> None:
    aligned = _provenance_aligned_adata()
    slice_data = _observed_slice(aligned)
    if mutation == "source_order":
        slice_data.obs["source_obs_id"] = ["cell1", "cell0"]
    elif mutation == "annotation":
        slice_data.obs["Annotation"] = ["B", "A"]
    elif mutation == "state":
        slice_data.X[0, 2] += 1e-3
    elif mutation == "origin":
        slice_data.uns["slice_origin"] = "generated_interval_local"
    elif mutation == "anchor":
        slice_data.uns["source_anchor_time"] = 1.0
    path = tmp_path / f"observed_{mutation}.h5ad"
    slice_data.write_h5ad(path)
    audit = VALIDATOR["Audit"]("zebrafish")

    validate_slice(
        audit,
        path,
        "Annotation",
        3,
        time_value=0.0,
        observed_time=True,
        aligned=aligned,
    )

    assert any(
        check["name"] == "exact observed slice provenance t=0"
        and check["status"] == "FAIL"
        for check in audit.checks
    )


def test_generated_slice_rejects_source_ids_wrong_anchor_and_unknown_labels(
    tmp_path: Path,
) -> None:
    aligned = _provenance_aligned_adata()
    state = np.asarray([[0.4, 0.5, 3.5], [0.6, 0.7, 3.8]], dtype=np.float32)
    generated = ad.AnnData(
        X=state,
        obs=pd.DataFrame(
            {
                "Annotation": ["A", "unknown"],
                "source_obs_id": ["cell0", "cell1"],
            }
        ),
    )
    generated.obsm["spatial"] = state[:, :2]
    generated.uns["slice_origin"] = "generated_interval_local"
    generated.uns["source_anchor_time"] = 1.0
    path = tmp_path / "generated_bad.h5ad"
    generated.write_h5ad(path)
    audit = VALIDATOR["Audit"]("zebrafish")

    validate_slice(
        audit,
        path,
        "Annotation",
        3,
        time_value=0.5,
        observed_time=False,
        aligned=aligned,
    )

    assert any(
        check["name"] == "interval-local generated slice provenance t=0.5"
        and check["status"] == "FAIL"
        for check in audit.checks
    )


def test_slice_provenance_summary_must_exactly_match_emitted_slices() -> None:
    provenance = {
        0.0: {"origin": "observed_real", "anchor_time": 0.0},
        0.5: {"origin": "generated_interval_local", "anchor_time": 0.0},
        1.0: {"origin": "observed_real", "anchor_time": 1.0},
    }
    simulation = {
        "slice_origins_by_time": {
            "0.0": "observed_real",
            "0.5": "generated_interval_local",
            "1.0": "observed_real",
        },
        "source_anchor_times_by_time": {"0.0": 0.0, "0.5": 0.0, "1.0": 1.0},
    }
    valid, _ = slice_provenance_summary_contract(
        simulation,
        slice_provenance=provenance,
        expected_times=(0.0, 0.5, 1.0),
    )
    assert valid

    simulation["source_anchor_times_by_time"]["0.5"] = 1.0
    valid, _ = slice_provenance_summary_contract(
        simulation,
        slice_provenance=provenance,
        expected_times=(0.0, 0.5, 1.0),
    )
    assert not valid


def _ad_summary_scope() -> dict:
    return {
        "analyses": {
            "communication": {
                "edge_prior_mode": "learned",
                "interpretation": (
                    "Spatial-attention summaries over learned-predictor-gated, "
                    "within-cutoff candidates trained from seven strict pairs; do not "
                    "report these values as a global cell-cell communication screen."
                ),
                "attention_scope": "full time-slice radius candidate graph",
                "training_interaction_scope": (
                    "stochastic interaction groups with base size 1024; a nonzero "
                    "remainder is merged into the final group (at most 2047); radius "
                    "candidates are evaluated only within each group during model "
                    "training and dynamics"
                ),
            },
            "ligand_receptor": {
                "analysis_scope": {
                    "strict_supported_pair_count": 7,
                    "interpretation": (
                        "Strict complete-subunit projection is limited to the seven "
                        "CellChatDB pairs fully represented by the AD expression panel; "
                        "it is not a global CCI screen."
                    ),
                }
            },
        }
    }


def test_ad_downstream_scope_is_readable_and_matches_learned_model() -> None:
    valid, _ = downstream_scope_contract(
        _ad_summary_scope(),
        expected_edge_mode="learned",
        expected_ad_lr_pairs=7,
    )
    assert valid

    wrong = _ad_summary_scope()
    wrong["analyses"]["ligand_receptor"]["analysis_scope"][
        "strict_supported_pair_count"
    ] = 8
    valid, _ = downstream_scope_contract(
        wrong,
        expected_edge_mode="learned",
        expected_ad_lr_pairs=7,
    )
    assert not valid


def _ad_no_lr_prior_summary_scope() -> dict:
    summary = _ad_summary_scope()
    summary["analyses"]["communication"].update(
        {
            "edge_prior_mode": "all_spatial",
            "interpretation": (
                "Spatial-attention summaries over within-cutoff candidates without "
                "a learned ligand-receptor edge gate. This is a no-LR-prior "
                "ablation, not the production main model and not a global cell-cell "
                "communication screen."
            ),
        }
    )
    summary["analyses"]["ligand_receptor"]["analysis_scope"] = {
        "strict_supported_pair_count": 7,
        "interpretation": (
            "The downstream strict complete-subunit projection remains limited to "
            "the seven CellChatDB pairs represented by the AD expression panel; "
            "those labels did not gate the all-spatial ablation model and do not "
            "form a global CCI screen."
        ),
    }
    return summary


def test_ad_all_spatial_downstream_scope_separates_attention_from_lr_projection() -> (
    None
):
    valid, _ = downstream_scope_contract(
        _ad_no_lr_prior_summary_scope(),
        expected_edge_mode="all_spatial",
        expected_ad_lr_pairs=7,
    )
    assert valid

    misleading = _ad_no_lr_prior_summary_scope()
    misleading["analyses"]["communication"][
        "interpretation"
    ] = "Spatial attention trained from seven strict LR pairs within-cutoff."
    valid, _ = downstream_scope_contract(
        misleading,
        expected_edge_mode="all_spatial",
        expected_ad_lr_pairs=7,
    )
    assert not valid

    gated_projection = _ad_no_lr_prior_summary_scope()
    gated_projection["analyses"]["ligand_receptor"]["analysis_scope"][
        "interpretation"
    ] = (
        "Seven expression panel pairs define the all-spatial model and form a global "
        "CCI screen."
    )
    valid, _ = downstream_scope_contract(
        gated_projection,
        expected_edge_mode="all_spatial",
        expected_ad_lr_pairs=7,
    )
    assert not valid


def _no_interaction_analyses() -> dict:
    return {
        "velocity": {
            "status": "completed",
            "interaction_component": False,
            "interaction_cutoff": None,
            "interaction_vector_status": (
                "not applicable; zero sentinel retained in the component archive"
            ),
        },
        "growth": {"status": "completed"},
        "composition": {"status": "completed"},
        "figures": {
            "status": "completed",
            "spatiotemporal_3d": {
                "status": "not applicable",
                "reason": (
                    "model has no interaction component, so no communication graph "
                    "or communication ribbons are defined"
                ),
            },
        },
        "gene_dynamics": {"status": "completed"},
        "communication": {
            "status": "not applicable",
            "representation": None,
            "edge_prior_mode": "none",
            "interpretation": (
                "Not applicable: this matched ablation has no interaction component. "
                "No radius graph, learned-gate graph, zero communication matrix, or "
                "ligand-receptor projection is substituted."
            ),
            "reason": "model has no interaction component",
            "edge_selection_by_time": None,
            "table": None,
            "attention_directory": None,
        },
        "ligand_receptor": {
            "status": "not applicable",
            "reason": (
                "model has no interaction component; LR projection requires "
                "model-derived sparse communication attention"
            ),
            "analysis_scope": None,
        },
    }


def _no_interaction_summary_model() -> dict:
    return {
        "weight_stage": "Finetune_no_interaction",
        "score_stage": "Score_Refine",
        "scientific_contract": {
            "status": "matches requested preset",
            "components": ["growth", "score", "velocity"],
            "interaction_component": False,
            "edge_prior_mode": "none",
            "interaction_cutoff": None,
            "edge_predictor_threshold": None,
            "interaction_group_size": None,
            "interaction_group_max_size": None,
            "interaction_group_remainder_policy": None,
            "edge_predictor_threshold_check": None,
            "weight_stage": "Finetune_no_interaction",
            "score_stage": "Score_Refine",
        },
    }


@pytest.mark.parametrize("dataset", tuple(DATASETS))
def test_no_interaction_model_contract_requires_exact_none_fields(
    dataset: str,
) -> None:
    spec = VALIDATION_PROFILES[f"{dataset}_no_interaction"]
    valid, _ = downstream_model_contract(
        _no_interaction_summary_model(),
        {"interaction_group_size": None},
        spec=spec,
        expected_threshold=None,
    )
    assert valid


@pytest.mark.parametrize(
    ("target", "key", "value"),
    (
        ("contract", "interaction_component", True),
        ("contract", "edge_prior_mode", "all_spatial"),
        ("contract", "interaction_cutoff", 0.1),
        ("contract", "edge_predictor_threshold", 0.5),
        ("contract", "edge_predictor_threshold_check", "stale validation"),
        ("contract", "interaction_group_size", 1024),
        ("contract", "components", ["velocity", "growth", "score", "interaction"]),
        ("contract", "weight_stage", "Finetune"),
        ("contract", "score_stage", "Train_Score"),
        ("model", "weight_stage", "Finetune"),
        ("simulation", "interaction_group_size", 1024),
    ),
)
def test_no_interaction_model_contract_rejects_interaction_residue(
    target: str,
    key: str,
    value: object,
) -> None:
    spec = VALIDATION_PROFILES["zebrafish_no_interaction"]
    model = _no_interaction_summary_model()
    simulation = {"interaction_group_size": None}
    if target == "contract":
        model["scientific_contract"][key] = value
    elif target == "model":
        model[key] = value
    else:
        simulation[key] = value

    valid, _ = downstream_model_contract(
        model,
        simulation,
        spec=spec,
        expected_threshold=None,
    )

    assert not valid


def test_no_interaction_status_and_scope_keep_retained_analyses_productive() -> None:
    analyses = _no_interaction_analyses()
    valid, _ = downstream_analysis_status_contract(
        analyses,
        interaction_component=False,
    )
    assert valid
    valid, _ = downstream_scope_contract(
        {"analyses": analyses},
        expected_edge_mode="none",
    )
    assert valid

    for retained in ("velocity", "growth", "composition", "figures", "gene_dynamics"):
        mutated = _no_interaction_analyses()
        mutated[retained]["status"] = "not applicable"
        valid, _ = downstream_analysis_status_contract(
            mutated,
            interaction_component=False,
        )
        assert not valid


@pytest.mark.parametrize(
    ("target", "key", "value"),
    (
        ("velocity", "interaction_component", True),
        ("velocity", "interaction_cutoff", 0.1),
        ("velocity", "interaction_vector_status", "evaluated"),
        ("spatiotemporal_3d", "status", "completed"),
        ("spatiotemporal_3d", "reason", "communication rendered"),
    ),
)
def test_no_interaction_analysis_status_rejects_interaction_metadata(
    target: str,
    key: str,
    value: object,
) -> None:
    analyses = _no_interaction_analyses()
    record = (
        analyses["figures"]["spatiotemporal_3d"]
        if target == "spatiotemporal_3d"
        else analyses[target]
    )
    record[key] = value

    valid, _ = downstream_analysis_status_contract(
        analyses,
        interaction_component=False,
    )

    assert not valid


@pytest.mark.parametrize(
    ("analysis", "key", "value"),
    (
        ("communication", "status", "completed"),
        ("communication", "edge_prior_mode", "all_spatial"),
        ("communication", "table", "communication_by_celltype.csv"),
        ("communication", "edge_selection_by_time", {}),
        ("ligand_receptor", "status", "completed"),
        ("ligand_receptor", "analysis_scope", {}),
    ),
)
def test_no_interaction_scope_rejects_fabricated_zero_or_projection_outputs(
    analysis: str,
    key: str,
    value: object,
) -> None:
    analyses = _no_interaction_analyses()
    analyses[analysis][key] = value

    valid, _ = downstream_scope_contract(
        {"analyses": analyses},
        expected_edge_mode="none",
    )

    assert not valid


def _communication_edge_selection_fixture(
    tmp_path: Path,
) -> tuple[pd.DataFrame, dict, Path]:
    sparse_dir = tmp_path / "sparse_attention"
    sparse_dir.mkdir()
    selected_by_time = {0.0: 2, 0.5: 0, 1.0: 1}
    candidate_by_time = {0.0: 4, 0.5: 6, 1.0: 2}
    for time_value, selected_count in selected_by_time.items():
        np.save(
            sparse_dir / f"attn_mean_interp_t{time_value}.npy",
            np.linspace(0.1, 0.2, selected_count, dtype=np.float32),
        )
        np.save(
            sparse_dir / f"edge_index_interp_t{time_value}.npy",
            np.asarray([[0, 1], [1, 2]], dtype=np.int64)[:, :selected_count],
        )
    summary = {
        "structural_zero_interpretation": (
            "When candidate edges exist, a structural zero means no candidate edge "
            "passed the LR-informed learned edge-predictor gate at that time point; "
            "when the candidate "
            "count is zero, no edge was available within the spatial cutoff. Neither "
            "case establishes absence of all biological communication."
        ),
        "edge_selection_by_time": {
            f"{time_value:g}": {
                "candidate_count": candidate_by_time[time_value],
                "selected_count": selected_count,
                "selected_fraction": (selected_count / candidate_by_time[time_value]),
                "status": (
                    "selected_edges"
                    if selected_count > 0
                    else "no_edges_passed_learned_gate"
                ),
            }
            for time_value, selected_count in selected_by_time.items()
        },
    }
    communication_rows = []
    for time_value, selected_count in selected_by_time.items():
        for source in ("A", "B"):
            for target in ("A", "B"):
                communication_rows.append(
                    {
                        "time": time_value,
                        "source": source,
                        "target": target,
                        "attention_per_source": (
                            0.0
                            if selected_count == 0
                            else (0.2 if source != target else 0.0)
                        ),
                    }
                )
    communication = pd.DataFrame(communication_rows)
    return communication, summary, sparse_dir


def test_communication_contract_allows_canonical_per_time_structural_zero(
    tmp_path: Path,
) -> None:
    communication, summary, sparse_dir = _communication_edge_selection_fixture(tmp_path)

    valid, _ = communication_edge_selection_contract(
        communication,
        summary,
        sparse_dir,
        expected_times=(0.0, 0.5, 1.0),
        expected_node_counts={0.0: 3, 0.5: 3, 1.0: 3},
        expected_label_sets={0.0: {"A", "B"}, 0.5: {"A", "B"}, 1.0: {"A", "B"}},
        expected_edge_mode="learned",
    )

    assert valid


def test_communication_contract_allows_zero_candidates_when_run_is_nonzero(
    tmp_path: Path,
) -> None:
    communication, summary, sparse_dir = _communication_edge_selection_fixture(tmp_path)
    summary["edge_selection_by_time"]["0.5"]["candidate_count"] = 0
    summary["edge_selection_by_time"]["0.5"]["status"] = "no_edges_within_cutoff"

    valid, _ = communication_edge_selection_contract(
        communication,
        summary,
        sparse_dir,
        expected_times=(0.0, 0.5, 1.0),
        expected_node_counts={0.0: 3, 0.5: 3, 1.0: 3},
        expected_label_sets={0.0: {"A", "B"}, 0.5: {"A", "B"}, 1.0: {"A", "B"}},
        expected_edge_mode="learned",
    )

    assert valid


@pytest.mark.parametrize(
    "mutation",
    (
        "noncanonical_empty_attention",
        "noncanonical_empty_edge_index",
        "nonfinite_attention",
        "negative_attention",
        "float_edge_index",
        "negative_edge_index",
        "out_of_bounds_edge_index",
        "self_loop_edge_index",
        "duplicate_edge_index",
        "nonfloating_attention",
        "missing_time_file",
        "all_times_empty",
        "missing_summary_time",
        "extra_summary_time",
        "negative_candidate_count",
        "odd_candidate_count",
        "selected_count_mismatch",
        "selected_fraction_mismatch",
        "boolean_selected_fraction",
        "zero_status_mismatch",
        "positive_status_mismatch",
        "misleading_interpretation",
        "globally_zero_celltype_table",
        "structural_zero_nonzero_table",
        "missing_table_column",
        "duplicate_table_key",
        "wrong_table_time",
        "missing_table_pair",
        "missing_table_type",
        "unknown_table_types",
    ),
)
def test_communication_contract_rejects_incomplete_or_misleading_zero_evidence(
    tmp_path: Path,
    mutation: str,
) -> None:
    communication, summary, sparse_dir = _communication_edge_selection_fixture(tmp_path)
    if mutation == "noncanonical_empty_attention":
        np.save(
            sparse_dir / "attn_mean_interp_t0.5.npy",
            np.empty((0, 1), dtype=np.float32),
        )
    elif mutation == "noncanonical_empty_edge_index":
        np.save(
            sparse_dir / "edge_index_interp_t0.5.npy",
            np.empty((0, 2), dtype=np.int64),
        )
    elif mutation == "nonfinite_attention":
        np.save(
            sparse_dir / "attn_mean_interp_t0.0.npy",
            np.asarray([0.1, np.nan], dtype=np.float32),
        )
    elif mutation == "negative_attention":
        np.save(
            sparse_dir / "attn_mean_interp_t0.0.npy",
            np.asarray([0.1, -0.2], dtype=np.float32),
        )
    elif mutation == "float_edge_index":
        np.save(
            sparse_dir / "edge_index_interp_t0.0.npy",
            np.zeros((2, 2), dtype=np.float32),
        )
    elif mutation == "negative_edge_index":
        np.save(
            sparse_dir / "edge_index_interp_t0.0.npy",
            np.asarray([[0, -1], [1, 2]], dtype=np.int64),
        )
    elif mutation == "out_of_bounds_edge_index":
        np.save(
            sparse_dir / "edge_index_interp_t0.0.npy",
            np.asarray([[0, 3], [1, 2]], dtype=np.int64),
        )
    elif mutation == "self_loop_edge_index":
        np.save(
            sparse_dir / "edge_index_interp_t0.0.npy",
            np.asarray([[0, 1], [0, 2]], dtype=np.int64),
        )
    elif mutation == "duplicate_edge_index":
        np.save(
            sparse_dir / "edge_index_interp_t0.0.npy",
            np.asarray([[0, 0], [1, 1]], dtype=np.int64),
        )
    elif mutation == "nonfloating_attention":
        np.save(
            sparse_dir / "attn_mean_interp_t0.0.npy",
            np.asarray([1, 2], dtype=np.int64),
        )
    elif mutation == "missing_time_file":
        (sparse_dir / "attn_mean_interp_t0.5.npy").unlink()
    elif mutation == "all_times_empty":
        for time_value in (0.0, 0.5, 1.0):
            np.save(
                sparse_dir / f"attn_mean_interp_t{time_value}.npy",
                np.empty((0,), dtype=np.float32),
            )
            np.save(
                sparse_dir / f"edge_index_interp_t{time_value}.npy",
                np.empty((2, 0), dtype=np.int64),
            )
            record = summary["edge_selection_by_time"][f"{time_value:g}"]
            record["selected_count"] = 0
            record["selected_fraction"] = 0.0
            record["status"] = "no_edges_passed_learned_gate"
    elif mutation == "missing_summary_time":
        del summary["edge_selection_by_time"]["0.5"]
    elif mutation == "extra_summary_time":
        summary["edge_selection_by_time"]["1.5"] = dict(
            summary["edge_selection_by_time"]["1"]
        )
    elif mutation == "negative_candidate_count":
        summary["edge_selection_by_time"]["0.5"]["candidate_count"] = -1
    elif mutation == "odd_candidate_count":
        summary["edge_selection_by_time"]["0.5"]["candidate_count"] = 5
    elif mutation == "selected_count_mismatch":
        summary["edge_selection_by_time"]["0"]["selected_count"] = 1
    elif mutation == "selected_fraction_mismatch":
        summary["edge_selection_by_time"]["0"]["selected_fraction"] = 0.49
    elif mutation == "boolean_selected_fraction":
        summary["edge_selection_by_time"]["0.5"]["selected_fraction"] = False
    elif mutation == "zero_status_mismatch":
        summary["edge_selection_by_time"]["0.5"]["status"] = "selected_edges"
    elif mutation == "positive_status_mismatch":
        summary["edge_selection_by_time"]["0"][
            "status"
        ] = "no_edges_passed_learned_gate"
    elif mutation == "misleading_interpretation":
        summary[
            "structural_zero_interpretation"
        ] = "A structural zero proves that no biological communication exists."
    elif mutation == "globally_zero_celltype_table":
        communication["attention_per_source"] = 0.0
    elif mutation == "structural_zero_nonzero_table":
        communication.loc[communication["time"] == 0.5, "attention_per_source"] = 0.1
    elif mutation == "missing_table_column":
        communication = communication.drop(columns="source")
    elif mutation == "duplicate_table_key":
        communication = pd.concat([communication, communication.iloc[[0]]])
    elif mutation == "wrong_table_time":
        communication.loc[communication["time"] == 0.5, "time"] = 1.5
    elif mutation == "missing_table_pair":
        communication = communication.drop(index=communication.index[0])
    elif mutation == "missing_table_type":
        communication = communication.loc[
            ~(
                (communication["time"] == 0.5)
                & ((communication["source"] == "B") | (communication["target"] == "B"))
            )
        ]
    elif mutation == "unknown_table_types":
        time_mask = communication["time"] == 0.5
        communication.loc[time_mask, "source"] = communication.loc[
            time_mask, "source"
        ].map({"A": "X", "B": "Y"})
        communication.loc[time_mask, "target"] = communication.loc[
            time_mask, "target"
        ].map({"A": "X", "B": "Y"})

    valid, _ = communication_edge_selection_contract(
        communication,
        summary,
        sparse_dir,
        expected_times=(0.0, 0.5, 1.0),
        expected_node_counts={0.0: 3, 0.5: 3, 1.0: 3},
        expected_label_sets={0.0: {"A", "B"}, 0.5: {"A", "B"}, 1.0: {"A", "B"}},
        expected_edge_mode="learned",
    )

    assert not valid


def test_all_spatial_communication_contract_requires_every_candidate() -> None:
    communication = pd.DataFrame(
        {
            "time": [0.0, 0.0, 0.0, 0.0],
            "source": ["A", "A", "B", "B"],
            "target": ["A", "B", "A", "B"],
            "attention_per_source": [0.0, 0.2, 0.1, 0.0],
        }
    )
    with tempfile.TemporaryDirectory() as temp_dir:
        sparse_dir = Path(temp_dir)
        np.save(
            sparse_dir / "attn_mean_interp_t0.0.npy",
            np.asarray([0.1, 0.2], dtype=np.float32),
        )
        np.save(
            sparse_dir / "edge_index_interp_t0.0.npy",
            np.asarray([[0, 1], [1, 0]], dtype=np.int64),
        )
        summary = {
            "structural_zero_interpretation": (
                "A structural zero means no edge was available within the spatial "
                "cutoff at that time point; it does not establish absence of all "
                "biological communication."
            ),
            "edge_selection_by_time": {
                "0": {
                    "candidate_count": 2,
                    "selected_count": 2,
                    "selected_fraction": 1.0,
                    "status": "selected_edges",
                }
            },
        }
        valid, _ = communication_edge_selection_contract(
            communication,
            summary,
            sparse_dir,
            expected_times=(0.0,),
            expected_node_counts={0.0: 2},
            expected_label_sets={0.0: {"A", "B"}},
            expected_edge_mode="all_spatial",
        )
        assert valid

        summary["edge_selection_by_time"]["0"]["candidate_count"] = 3
        summary["edge_selection_by_time"]["0"]["selected_fraction"] = 2 / 3
        valid, _ = communication_edge_selection_contract(
            communication,
            summary,
            sparse_dir,
            expected_times=(0.0,),
            expected_node_counts={0.0: 2},
            expected_label_sets={0.0: {"A", "B"}},
            expected_edge_mode="all_spatial",
        )
        assert not valid

        summary["edge_selection_by_time"]["0"]["candidate_count"] = 2
        summary["edge_selection_by_time"]["0"]["selected_fraction"] = 1.0
        np.save(
            sparse_dir / "edge_index_interp_t0.0.npy",
            np.asarray([[0, 0], [1, 2]], dtype=np.int64),
        )
        valid, _ = communication_edge_selection_contract(
            communication,
            summary,
            sparse_dir,
            expected_times=(0.0,),
            expected_node_counts={0.0: 3},
            expected_label_sets={0.0: {"A", "B"}},
            expected_edge_mode="all_spatial",
        )
        assert not valid


def test_ad_retained_pair_count_excludes_pair_celltype_children() -> None:
    rows = [
        {"trajectory_kind": "pair", "retained": True, "pair_id": f"pair_{index}"}
        for index in range(7)
    ]
    rows.extend(
        {
            "trajectory_kind": "pair_celltype",
            "retained": True,
            "pair_id": f"pair_{index}",
        }
        for index in range(20)
    )
    count, _ = retained_top_level_lr_pairs(pd.DataFrame(rows))
    assert count == 7

    duplicated = pd.DataFrame(rows + [rows[0]])
    count, _ = retained_top_level_lr_pairs(duplicated)
    assert count is None


def test_ad_lr_pair_time_grid_requires_same_ids_and_every_time_once() -> None:
    coverage = pd.DataFrame(
        [
            {
                "trajectory_kind": "pair",
                "retained": True,
                "pair_id": pair_id,
            }
            for pair_id in ("pair_a", "pair_b")
        ]
    )
    complete = pd.DataFrame(
        [
            {"pair_id": pair_id, "time": time}
            for pair_id in ("pair_a", "pair_b")
            for time in (0.0, 0.5, 1.0)
        ]
    )
    valid, _ = complete_lr_pair_time_grid(
        coverage, complete, expected_times=(0.0, 0.5, 1.0)
    )
    assert valid

    incomplete = complete.loc[
        ~((complete["pair_id"] == "pair_b") & (complete["time"] == 0.5))
    ]
    valid, _ = complete_lr_pair_time_grid(
        coverage, incomplete, expected_times=(0.0, 0.5, 1.0)
    )
    assert not valid

    wrong_ids = complete.replace({"pair_b": "pair_c"})
    valid, _ = complete_lr_pair_time_grid(
        coverage, wrong_ids, expected_times=(0.0, 0.5, 1.0)
    )
    assert not valid


def test_color_map_must_cover_labels_with_distinct_valid_hex_colors() -> None:
    valid, _ = valid_color_map(
        {"Microglia": "#112233", "Astrocyte": "#aabbcc"},
        required_labels={"Microglia", "Astrocyte"},
    )
    assert valid

    same, _ = valid_color_map(
        {"Microglia": "#112233", "Astrocyte": "#112233"},
        required_labels={"Microglia", "Astrocyte"},
    )
    assert not same

    missing, _ = valid_color_map(
        {"Microglia": "#112233"},
        required_labels={"Microglia", "Astrocyte"},
    )
    assert not missing


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _matched_global_rng(boundary: int) -> dict:
    record = {
        "python_random_sha256": _sha(f"python-{boundary}"),
        "numpy_legacy_sha256": _sha(f"numpy-{boundary}"),
        "torch_cpu_sha256": _sha(f"torch-{boundary}"),
        "torch_cuda": {
            "available": False,
            "visible_device_count": 0,
            "selected_device": None,
            "state_sha256": None,
            "snapshot_scope": "selected_training_device_only",
        },
        "determinism": {
            "deterministic_algorithms_enabled": False,
            "deterministic_algorithms_warn_only_enabled": False,
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
            "bit_exact_cuda_determinism_claimed": False,
        },
    }
    record["aggregate_sha256"] = hashlib.sha256(
        json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return record


def _matched_environment() -> dict:
    package = lambda distribution, module: {
        "distribution": distribution,
        "module": module,
        "version": "test",
        "status": "available",
    }
    return {
        "device": "cpu",
        "device_type": "cpu",
        "torch_version": "test",
        "cuda_compiled_version": None,
        "cuda_available": False,
        "cuda_device_name": None,
        "cuda_device_index": None,
        "cudnn_version": None,
        "python_version": "3.test",
        "platform": "test-platform",
        "dependency_versions": {
            "numpy": package("numpy", "numpy"),
            "pot": package("POT", "ot"),
            "torchdiffeq": package("torchdiffeq", "torchdiffeq"),
            "torch_geometric": package("torch-geometric", "torch_geometric"),
            "torch": package("torch", "torch"),
            "cuda": {"version": None, "status": "not_compiled"},
            "cudnn": {"version": None, "status": "unavailable"},
            "python": {
                "version": "3.test",
                "implementation": "CPython",
                "status": "available",
            },
            "platform": {"value": "test-platform", "status": "available"},
        },
    }


def _matched_crn(config: dict, arm: str) -> dict:
    condition, mode = VALIDATOR["MATCHED_CONDITIONS"][arm]
    has_interaction = arm != "no_interaction"
    components = [str(value).lower() for value in config["model"]["components"]]
    return {
        "schema_version": 1,
        "protocol": "isolated-interaction-crn-v1",
        "strict_matched_entrypoint": True,
        "condition": condition,
        "interaction_mode": mode,
        "components": components,
        "formal_data_contract": {
            "matched_ablation_declared": True,
            "h5ad_and_exact_model_input_provenance_valid": True,
            "edge_predictor_provenance_valid": True,
        },
        "global_streams": {
            "base_seed": 42,
            "seed_application": "once_before_model_construction",
            "python_random": {"api": "random.seed", "seed": 42},
            "numpy_legacy": {"api": "numpy.random.seed", "seed": 42},
            "torch_cpu": {"api": "torch.manual_seed", "seed": 42},
            "torch_cuda": {
                "api": "torch.cuda.manual_seed_all",
                "seed": 42,
                "available": False,
            },
            "optional_interaction_advance_policy": "forbidden",
            "cuda_determinism_boundary": "selected-device state; no bit-exact claim",
        },
        "constructor_isolation": {
            "optional_interaction_component": {
                "mechanism": "torch.random.fork_rng(devices=[])",
                "construction_device": "cpu_before_model.to(device)",
                "restores_global_torch_cpu_state": True,
            },
            "frozen_edge_predictor": {
                "active": mode == "learned",
                "mechanism": (
                    "nested torch.random.fork_rng(devices=[])"
                    if mode == "learned"
                    else "not_applicable"
                ),
                "trainable_backbone_initialization_isolated": mode
                in {"learned", "all_spatial"},
            },
        },
        "interaction_grouping_stream": {
            "active": has_interaction,
            "generator": "private torch.Generator" if has_interaction else None,
            "device": "cpu" if has_interaction else None,
            "seed_offset": 10_000 if has_interaction else None,
            "seed": 10_042 if has_interaction else None,
            "reset_between_stages": False,
            "shared_seed_for_full_and_no_lr": mode in {"learned", "all_spatial"},
            "advances_global_torch_stream": False,
        },
        "inactive_interaction": {
            "forward_compute_skipped": True,
            "private_stream_not_advanced": True,
            "no_interaction_constructor_skipped": not has_interaction,
            "private_generator_created": has_interaction,
            "no_interaction_arm_skips_constructor_and_generator": (
                arm == "no_interaction"
            ),
        },
    }


def _write_embedded_training_summary(path: Path, summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(summary, sort_keys=True, allow_nan=False)
    artifact = ad.AnnData(X=np.zeros((1, 1), dtype=np.float32))
    metadata = {
        "schema_version": int(summary.get("schema_version", 1)),
        "file": "training_run_summary.json",
        "summary_json": encoded,
    }
    artifact.uns["training_run_summary"] = metadata
    artifact.uns["all_model"] = {"training_run_summary": metadata}
    artifact.write_h5ad(path)


def _rewrite_summary(run_root: Path, profile: str, summary: dict) -> None:
    training = run_root / profile / "training"
    (training / "training_run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_embedded_training_summary(training / "adata.h5ad", summary)


def _build_tiny_matched_family(tmp_path: Path) -> tuple[Path, dict[str, dict]]:
    dataset = "zebrafish"
    arms = {
        "full": dataset,
        "no_lr_prior": f"{dataset}_no_lr_prior",
        "no_interaction": f"{dataset}_no_interaction",
    }
    source = tmp_path / "shared.h5ad"
    input_artifact = ad.AnnData(
        X=np.zeros((3, 2), dtype=np.float32),
        obs=pd.DataFrame(
            {"time_point_processed": [0.0, 1.0, 2.0]},
            index=["cell-a", "cell-b", "cell-c"],
        ),
    )
    input_artifact.obsm["X_latent"] = np.asarray(
        [[1.0, 2.0], [2.0, 3.0], [3.0, 5.0]], dtype=np.float32
    )
    input_artifact.obsm["spatial_aligned"] = np.asarray(
        [[0.0, 0.0], [0.5, 0.25], [1.0, 0.5]], dtype=np.float32
    )
    input_artifact.write_h5ad(source)
    source_bytes = source.read_bytes()
    input_size = len(source_bytes)
    input_sha = hashlib.sha256(source_bytes).hexdigest()
    selection = {
        "time_key": "time_point_processed",
        "processed_time_key": "time_point_processed",
        "obsm_key": "X_latent",
        "resolved_latent_key": "X_latent",
        "spatial_key": "spatial_aligned",
        "is_spatial": True,
    }
    model_input = np.hstack(
        (input_artifact.obsm["spatial_aligned"], input_artifact.obsm["X_latent"])
    ).astype(np.float32)
    summaries: dict[str, dict] = {}
    configs: dict[str, dict] = {}
    predictor = (
        tmp_path
        / dataset
        / "preprocess"
        / "edge_classifier"
        / (f"{dataset}_edge_model.pt")
    )
    predictor.parent.mkdir(parents=True)
    predictor.write_bytes(b"exact learned predictor bytes")
    predictor_size = predictor.stat().st_size
    predictor_sha = hashlib.sha256(predictor.read_bytes()).hexdigest()

    private_boundaries = [_sha(f"private-{index}") for index in range(5)]
    for arm, profile in arms.items():
        run_dir = tmp_path / profile
        aligned = run_dir / "preprocess" / f"{dataset}_aligned.h5ad"
        aligned.parent.mkdir(parents=True, exist_ok=True)
        aligned.write_bytes(source_bytes)
        config_name = VALIDATOR["_matched_config_name"](dataset, arm)
        config = yaml.safe_load(
            (ROOT / "CytoBridge" / "configs" / config_name).read_text(encoding="utf-8")
        )
        training = run_dir / "training"
        config["ckpt_dir"] = str(training)
        config["spatial_dim"] = 2
        config["model"]["spatial_dim"] = 2
        if arm == "full":
            config["model"]["interaction_net"]["edge_predictor_path"] = str(
                predictor.resolve()
            )
        training.mkdir(parents=True, exist_ok=True)
        (training / "config.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )
        configs[arm] = config

        plan = config["training"]["plan"]
        stages = []
        private_cursor = 0
        for index, stage_config in enumerate(plan):
            interaction_active = arm != "no_interaction" and index >= 2
            if arm == "no_interaction":
                private_start = private_end = {
                    "active": False,
                    "seed": None,
                    "state_sha256": None,
                }
            else:
                private_start = {
                    "active": True,
                    "seed": 10_042,
                    "state_sha256": private_boundaries[private_cursor],
                }
                if interaction_active:
                    private_cursor += 1
                private_end = {
                    "active": True,
                    "seed": 10_042,
                    "state_sha256": private_boundaries[private_cursor],
                }
            stages.append(
                {
                    "stage_index": index,
                    "stage": stage_config["name"],
                    "mode": stage_config["mode"],
                    "configured_epochs": int(stage_config["epochs"]),
                    "recorded_epochs": int(stage_config["epochs"]),
                    "batch_size": int(stage_config.get("batch_size", 1)),
                    "optimizer_step_count": (
                        int(stage_config["epochs"]) * 4
                        if stage_config["mode"] == "neural_ode"
                        else int(stage_config["epochs"])
                    ),
                    "interaction_active": interaction_active,
                    "interaction_rng_action": (
                        "consume_private_interaction_generator"
                        if interaction_active
                        else "inactive_skip_without_rng_advance"
                    ),
                    "score_energy_objective": (
                        "velocity_score_cross_term"
                        if stage_config["mode"] == "neural_ode"
                        else None
                    ),
                    "rng_state_digests": {
                        "stage_start": {
                            "global": _matched_global_rng(index),
                            "private_interaction_grouping": private_start,
                        },
                        "stage_end": {
                            "global": _matched_global_rng(index + 1),
                            "private_interaction_grouping": private_end,
                        },
                    },
                }
            )
        expected_declaration = _expected_matched_ablation(dataset, arm)
        edge_identity = (
            {
                "applicable": True,
                "edge_prior_mode": "learned",
                "path": str(predictor.resolve()),
                "size_bytes": predictor_size,
                "sha256": predictor_sha,
                "not_applicable": False,
                "not_applicable_reason": None,
                "unchanged_during_training": True,
            }
            if arm == "full"
            else {
                "applicable": False,
                "edge_prior_mode": VALIDATOR["MATCHED_CONDITIONS"][arm][1],
                "path": None,
                "size_bytes": None,
                "sha256": None,
                "not_applicable": True,
                "not_applicable_reason": "condition has no learned edge gate",
                "unchanged_during_training": True,
            }
        )
        summary = {
            "schema_version": 1,
            "environment": _matched_environment(),
            "data": {
                "n_observations": 3,
                "n_timepoints": 3,
                "sample_counts_by_timepoint": [1, 1, 1],
                "input_h5ad": {
                    "source_kind": "h5ad_path",
                    "path": str(aligned.resolve()),
                    "size_bytes": input_size,
                    "sha256": input_sha,
                    "not_applicable": False,
                    "not_applicable_reason": None,
                },
                "input_selection": selection,
                "model_input": _canonical_array_identity(model_input),
                "processed_time": _canonical_array_identity(
                    input_artifact.obs["time_point_processed"].to_numpy()
                ),
                "obs_names": _ordered_obs_names_identity(input_artifact.obs_names),
                "edge_predictor": edge_identity,
            },
            "training": {
                "score_energy_objective_default": "velocity_score_cross_term",
                "common_random_numbers": _matched_crn(config, arm),
                "matched_ablation": {
                    "declared": True,
                    "config_declaration": expected_declaration,
                    "normalized": expected_declaration,
                    "actual_condition": VALIDATOR["MATCHED_CONDITIONS"][arm][0],
                },
                "implementation": _training_implementation_identity(),
            },
            "stages": stages,
        }
        summaries[arm] = summary
        _rewrite_summary(tmp_path, profile, summary)
        for stage in ("Pretrain", "Refine"):
            checkpoint = training / stage / "best_model.pth"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            state = {"velocity_net.weight": torch.arange(4, dtype=torch.float32)}
            if arm != "no_interaction":
                state["interaction_net.weight"] = torch.full(
                    (2,), 1.0 if arm == "full" else 2.0
                )
            torch.save(state, checkpoint)
    return tmp_path, summaries


def _validate_tiny_matched_family(run_root: Path):
    profiles = ["zebrafish", "zebrafish_no_lr_prior", "zebrafish_no_interaction"]
    audits = {profile: Audit(profile) for profile in profiles}
    return validate_matched_family(
        run_root,
        "zebrafish",
        requested_profiles=profiles,
        individual_audits=audits,
    )


def test_matched_family_requires_complete_three_arm_request(tmp_path: Path) -> None:
    audit = validate_matched_family(
        tmp_path,
        "zebrafish",
        requested_profiles=["zebrafish", "zebrafish_no_lr_prior"],
        individual_audits={},
    )
    assert audit.errors
    assert audit.checks[0]["name"] == "complete matched three-arm request"


def test_matched_family_accepts_exact_formal_three_arm_fixture(tmp_path: Path) -> None:
    run_root, _ = _build_tiny_matched_family(tmp_path)
    audit = _validate_tiny_matched_family(run_root)
    assert not audit.errors, audit.errors


@pytest.mark.parametrize(
    "mutation",
    (
        "old_summary",
        "input",
        "implementation",
        "environment",
        "rng",
        "embedded",
        "checkpoint",
        "canonical_config",
        "predictor",
        "short_plan",
        "non_list_plan",
    ),
)
def test_matched_family_fails_closed_on_provenance_mutation(
    tmp_path: Path,
    mutation: str,
) -> None:
    run_root, summaries = _build_tiny_matched_family(tmp_path)
    profile = "zebrafish_no_lr_prior"
    summary = deepcopy(summaries["no_lr_prior"])
    if mutation == "old_summary":
        summary["training"].pop("matched_ablation")
        _rewrite_summary(run_root, profile, summary)
    elif mutation == "input":
        summary["data"]["model_input"]["sha256"] = _sha("wrong model input")
        _rewrite_summary(run_root, profile, summary)
    elif mutation == "implementation":
        summary["training"]["implementation"]["aggregate_sha256"] = _sha("wrong code")
        _rewrite_summary(run_root, profile, summary)
    elif mutation == "environment":
        summary["environment"]["torch_version"] = "different"
        _rewrite_summary(run_root, profile, summary)
    elif mutation == "rng":
        global_rng = summary["stages"][0]["rng_state_digests"]["stage_start"]["global"]
        global_rng["python_random_sha256"] = _sha("different rng")
        global_rng["aggregate_sha256"] = hashlib.sha256(
            json.dumps(
                {
                    key: value
                    for key, value in global_rng.items()
                    if key != "aggregate_sha256"
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        _rewrite_summary(run_root, profile, summary)
    elif mutation == "embedded":
        summary["training"]["implementation"]["aggregate_sha256"] = _sha(
            "standalone-only mutation"
        )
        (run_root / profile / "training" / "training_run_summary.json").write_text(
            json.dumps(summary), encoding="utf-8"
        )
    elif mutation == "checkpoint":
        torch.save(
            {"velocity_net.weight": torch.full((4,), 99.0)},
            run_root / profile / "training" / "Refine" / "best_model.pth",
        )
    elif mutation == "canonical_config":
        path = run_root / profile / "training" / "config.yaml"
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        config["reverse"] = False
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    elif mutation == "predictor":
        predictor = (
            run_root
            / "zebrafish"
            / "preprocess"
            / "edge_classifier"
            / "zebrafish_edge_model.pt"
        )
        predictor.write_bytes(b"mutated predictor bytes")
    elif mutation == "short_plan":
        path = run_root / profile / "training" / "config.yaml"
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        config["training"]["plan"] = config["training"]["plan"][:1]
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    elif mutation == "non_list_plan":
        path = run_root / profile / "training" / "config.yaml"
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        config["training"]["plan"] = {"not": "a list"}
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    audit = _validate_tiny_matched_family(run_root)
    assert audit.errors, mutation


@pytest.mark.parametrize("bad_count", (None, 0, True))
def test_matched_family_rejects_synchronized_bogus_optimizer_step_counts(
    tmp_path: Path,
    bad_count: object,
) -> None:
    run_root, summaries = _build_tiny_matched_family(tmp_path)
    profiles = {
        "full": "zebrafish",
        "no_lr_prior": "zebrafish_no_lr_prior",
        "no_interaction": "zebrafish_no_interaction",
    }
    for arm, profile in profiles.items():
        summary = deepcopy(summaries[arm])
        summary["stages"][0]["optimizer_step_count"] = bad_count
        _rewrite_summary(run_root, profile, summary)
    audit = _validate_tiny_matched_family(run_root)
    assert audit.errors
    failed = {item["name"] for item in audit.errors}
    assert "matched stage budgets and optimizer-step counts" in failed


def test_matched_family_recomputes_timepoint_count_for_optimizer_budget(
    tmp_path: Path,
) -> None:
    run_root, summaries = _build_tiny_matched_family(tmp_path)
    profiles = {
        "full": "zebrafish",
        "no_lr_prior": "zebrafish_no_lr_prior",
        "no_interaction": "zebrafish_no_interaction",
    }
    for arm, profile in profiles.items():
        summary = deepcopy(summaries[arm])
        summary["data"]["n_timepoints"] = 2
        summary["data"]["sample_counts_by_timepoint"] = [1, 2]
        for stage in summary["stages"]:
            if stage["mode"] == "neural_ode":
                stage["optimizer_step_count"] = stage["configured_epochs"] * 2
        _rewrite_summary(run_root, profile, summary)
    audit = _validate_tiny_matched_family(run_root)
    failed = {item["name"] for item in audit.errors}
    assert "exact shared reconstructed training arrays and observation order" in failed
    assert "matched stage budgets and optimizer-step counts" in failed


def test_edge_predictor_bytes_contract_rejects_summary_or_anndata_mutation(
    tmp_path: Path,
) -> None:
    predictor = tmp_path / "edge.pt"
    predictor.write_bytes(b"predictor")
    digest = hashlib.sha256(predictor.read_bytes()).hexdigest()
    summary = {
        "applicable": True,
        "edge_prior_mode": "learned",
        "path": str(predictor.resolve()),
        "size_bytes": predictor.stat().st_size,
        "sha256": digest,
        "not_applicable": False,
        "not_applicable_reason": None,
        "unchanged_during_training": True,
    }
    all_model = {
        "edge_predictor_size_bytes": predictor.stat().st_size,
        "edge_predictor_sha256": digest,
    }
    valid, _ = edge_predictor_artifact_contract(
        edge_path=predictor,
        summary_provenance=summary,
        all_model=all_model,
    )
    assert valid
    mutated = dict(all_model, edge_predictor_sha256=_sha("wrong"))
    valid, _ = edge_predictor_artifact_contract(
        edge_path=predictor,
        summary_provenance=summary,
        all_model=mutated,
    )
    assert not valid
