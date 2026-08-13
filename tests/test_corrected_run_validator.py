from __future__ import annotations

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
import yaml


ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = runpy.run_path(str(ROOT / "scripts" / "validate_corrected_de_novo_run.py"))
DATASETS = VALIDATOR["DATASETS"]
VALIDATION_PROFILES = VALIDATOR["VALIDATION_PROFILES"]
required_files = VALIDATOR["required_files"]
classifier_split_contract = VALIDATOR["classifier_split_contract"]
complete_lr_pair_time_grid = VALIDATOR["complete_lr_pair_time_grid"]
communication_edge_selection_contract = VALIDATOR[
    "communication_edge_selection_contract"
]
downstream_scope_contract = VALIDATOR["downstream_scope_contract"]
edge_threshold_provenance_contract = VALIDATOR["edge_threshold_provenance_contract"]
lr_database_provenance_contract = VALIDATOR["lr_database_provenance_contract"]
parse_args = VALIDATOR["parse_args"]
retained_top_level_lr_pairs = VALIDATOR["retained_top_level_lr_pairs"]
slice_provenance_summary_contract = VALIDATOR["slice_provenance_summary_contract"]
trained_edge_prior_contract = VALIDATOR["trained_edge_prior_contract"]
valid_color_map = VALIDATOR["valid_color_map"]
validate_slice = VALIDATOR["validate_slice"]
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
