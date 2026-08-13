from __future__ import annotations

import hashlib
import json
from pathlib import Path
import runpy
import sys

import pandas as pd
import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = runpy.run_path(str(ROOT / "scripts" / "validate_corrected_de_novo_run.py"))
DATASETS = VALIDATOR["DATASETS"]
VALIDATION_PROFILES = VALIDATOR["VALIDATION_PROFILES"]
required_files = VALIDATOR["required_files"]
classifier_split_contract = VALIDATOR["classifier_split_contract"]
complete_lr_pair_time_grid = VALIDATOR["complete_lr_pair_time_grid"]
downstream_scope_contract = VALIDATOR["downstream_scope_contract"]
edge_threshold_provenance_contract = VALIDATOR["edge_threshold_provenance_contract"]
lr_database_provenance_contract = VALIDATOR["lr_database_provenance_contract"]
parse_args = VALIDATOR["parse_args"]
retained_top_level_lr_pairs = VALIDATOR["retained_top_level_lr_pairs"]
trained_edge_prior_contract = VALIDATOR["trained_edge_prior_contract"]
valid_color_map = VALIDATOR["valid_color_map"]
zebrafish_split_sde_contract = VALIDATOR["zebrafish_split_sde_contract"]
validate_no_lr_ablation_artifact_cleanliness = VALIDATOR[
    "validate_no_lr_ablation_artifact_cleanliness"
]


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
    valid, _ = zebrafish_split_sde_contract(
        {"split_resample_dt": 0.05, "split_particle_ceiling": 100_000}
    )
    assert valid

    valid, _ = zebrafish_split_sde_contract(
        {"split_resample_dt": None, "split_particle_ceiling": None}
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
