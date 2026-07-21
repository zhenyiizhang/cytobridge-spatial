from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from reviewer_zebrafish_ccc import compare_multimethod_ccc as comparison  # noqa: E402


PAIRS = [("A", "A"), ("A", "B"), ("B", "A"), ("B", "B")]
STAGES = [
    (0.0, "5.25hpf"),
    (1.0, "10hpf"),
    (2.0, "12hpf"),
    (3.0, "18hpf"),
    (4.0, "24hpf"),
]
STAGE_TIMES = {0.0: 5.25, 1.0: 10.0, 2.0: 12.0, 3.0: 18.0, 4.0: 24.0}


def _json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _prepare_record(path: Path, *, relative_to: Path) -> dict[str, object]:
    return {
        "path": str(path.relative_to(relative_to)),
        "size_bytes": path.stat().st_size,
        "sha256": comparison._sha256(path),
        "md5": comparison._md5(path),
    }


def _type_rows(score_name: str, values: list[float]) -> pd.DataFrame:
    rows = []
    for stage_index, (stage, stage_label) in enumerate(STAGES):
        for pair_index, (sender, receiver) in enumerate(PAIRS):
            rows.append(
                {
                    "stage": stage,
                    "stage_label": stage_label,
                    "sender_type": sender,
                    "receiver_type": receiver,
                    score_name: values[(pair_index + stage_index) % len(values)],
                }
            )
    return pd.DataFrame(rows)


def _write_control(directory: Path, *, attention: float, message: float) -> None:
    directory.mkdir(parents=True)
    permutation_rows = []
    for target, value in (
        ("log1p_attention", attention),
        ("log1p_edge_message_joint", message),
    ):
        permutation_rows.append(
            {
                "score": "lr_compatibility_forward",
                "strata": comparison.STRICT_PERMUTATION_STRATA,
                "min_stratum_size": 4,
                "n_edges_total": 100,
                "n_edges_retained": 80,
                "retained_fraction": 0.8,
                "n_strata": 10,
                "observed_spearman": value,
                "null_mean": 0.0,
                "null_sd": 0.01,
                "empirical_p_greater": 0.01,
                "n_permutations": 100,
                "target": target,
                "residual_definition": "test",
            }
        )
    permutation_path = directory / "conditional_permutation_tests.csv"
    pd.DataFrame(permutation_rows).to_csv(permutation_path, index=False)
    nested_path = directory / "nested_grouped_cv_metrics.csv"
    pd.DataFrame(
        [
            {
                "target": target,
                "model": "confounders_plus_forward_lr",
                "n_edges": 100,
                "n_receiver_groups": 20,
                "n_folds": 5,
                "out_of_fold_r2": 0.4,
                "out_of_fold_rmse": 0.2,
                "out_of_fold_spearman": 0.5,
                "delta_r2_vs_confounders": delta,
            }
            for target, delta in (
                ("log1p_attention", attention / 100),
                ("log1p_edge_message_joint", message / 100),
            )
        ]
    ).to_csv(nested_path, index=False)
    reciprocal_path = directory / "reciprocal_edge_direction_tests.csv"
    pd.DataFrame(
        [
            {
                "stage": "all",
                "learned_direction_delta": direction,
                "n_reciprocal_pairs": 40,
                "spearman_with_lr_direction_delta": value,
                "sign_flip_null_mean": 0.0,
                "sign_flip_null_sd": 0.02,
                "empirical_p_greater": pvalue,
                "n_permutations": 100,
            }
            for direction, value, pvalue in (
                ("attention_direction_delta", attention / 2, 0.15),
                ("message_direction_delta", -abs(message), 1.0),
            )
        ]
    ).to_csv(reciprocal_path, index=False)
    _json(
        directory / "run_manifest.json",
        {
            "schema_version": 1,
            "artifacts": {
                "conditional_permutations": comparison._file_record(permutation_path),
                "nested_grouped_cv": comparison._file_record(nested_path),
                "reciprocal_edge_direction_tests": comparison._file_record(
                    reciprocal_path
                ),
            },
        },
    )


def _build_formal_tree(root: Path) -> None:
    cytobridge = root / "01_cytobridge"
    cytobridge.mkdir(parents=True)
    _json(
        cytobridge / "run_manifest.json",
        {
            "method": "cytobridge_one_layer_spatial_attention_and_exact_message",
            "interpretation": {"probability_claim": False},
        },
    )
    cyto = _type_rows("G_AB_attention_mean_mean", [4.0, 3.0, 2.0, 1.0])
    cyto["D_AB_joint_mean"] = np.resize([1.0, 2.0, 3.0, 4.0], len(cyto))
    cyto_path = cytobridge / "type_pair_summary.csv"
    cyto.to_csv(cyto_path, index=False)
    cyto_manifest = json.loads(
        (cytobridge / "run_manifest.json").read_text(encoding="utf-8")
    )
    cyto_manifest["artifacts"] = {
        "type_pair_summary": comparison._file_record(cyto_path)
    }
    _json(cytobridge / "run_manifest.json", cyto_manifest)

    _write_control(root / "02_attention_controls", attention=0.0284, message=0.0136)
    _write_control(
        root / "02_attention_controls_init_interaction",
        attention=0.0332,
        message=0.0260,
    )
    _write_control(
        root / "02_attention_controls_random_seed17",
        attention=0.0059,
        message=0.0625,
    )

    external = root / "03_external_ccc"
    shared = external / "shared_inputs"
    shared.mkdir(parents=True)
    shared_manifest_path = shared / "input_manifest.json"
    _json(
        shared_manifest_path,
        {
            "schema_version": 1,
            "stages": [
                {
                    "stage": str(stage),
                    "stage_time": STAGE_TIMES[stage],
                    "token": f"stage_{int(stage)}",
                    "n_cells": 5,
                    "n_cell_types": 2,
                    "cell_type_counts": {"A": 3, "B": 2},
                }
                for stage, _ in STAGES
            ],
        },
    )
    for method, display in (("commot", "COMMOT"), ("cellchat", "CellChat")):
        directory = external / f"{method}_current_lr"
        directory.mkdir(parents=True)
        rows = []
        for stage_index, (stage, _) in enumerate(STAGES):
            for pair_index, (sender, receiver) in enumerate(PAIRS):
                score = (
                    0.0
                    if stage_index == 0 or pair_index == 3
                    else float(4 - ((pair_index + stage_index) % 3))
                )
                if score == 0:
                    continue
                rows.append(
                    {
                        "method": display,
                        "database_variant": "current_zebrafish_lr_database",
                        "stage": str(stage),
                        "stage_time": STAGE_TIMES[stage],
                        "sender_type": sender,
                        "receiver_type": receiver,
                        "abundance_controlled_score": score,
                    }
                )
        score_path = directory / f"{method}_type_pair_scores.csv.gz"
        pd.DataFrame(rows).to_csv(
            score_path,
            index=False,
            compression="gzip",
        )
        manifest = {
            "method": display,
            "database_variant": "current_zebrafish_lr_database",
            "input_manifest": comparison._file_record(shared_manifest_path),
            "design": {
                "long_table_zero_policy": (
                    "structural zeros omitted; outer-join to input universe and "
                    "fill zero for comparisons"
                )
            },
            "artifacts": {
                (
                    "type_pair_scores"
                    if method == "commot"
                    else "cellchat_type_pair_scores.csv.gz"
                ): comparison._file_record(score_path)
            },
        }
        if method == "cellchat":
            manifest["design"].update(
                {
                    "population_size": False,
                    "nboot": 100,
                    "mean_method": "triMean",
                    "raw_use": True,
                }
            )
            manifest["software"] = {
                "CellChat": "2.2.0.9001",
                "CellChat_load_mode": "pinned official core R source",
                "CellChat_source_commit": comparison.PINNED_CELLCHAT_COMMIT,
            }
            exclusion_path = directory / "excluded_lr_rows.csv"
            pd.DataFrame(
                [
                    {
                        "database_row": 2281,
                        "interaction_id": "il10->il10ra_il10rb",
                        "current_ligand": "il10",
                        "current_receptor": "il10ra_il10rb",
                        "cellchat_ligand_token": "IL10",
                        "cellchat_receptor_token": "IL10RA_IL10RB",
                        "eligible": False,
                        "exclusion_reason": (
                            "receptor:token_not_geneinfo_or_declared_complex"
                        ),
                    },
                    {
                        "database_row": 2292,
                        "interaction_id": "ifng1->ifngr1_ifngr2",
                        "current_ligand": "ifng1",
                        "current_receptor": "ifngr1_ifngr2",
                        "cellchat_ligand_token": "IFNG1",
                        "cellchat_receptor_token": "IFNGR1_IFNGR2",
                        "eligible": False,
                        "exclusion_reason": (
                            "receptor:token_not_geneinfo_or_declared_complex"
                        ),
                    },
                ]
            ).to_csv(exclusion_path, index=False)
            manifest["database_validation"] = {
                "rows_requested": 2268,
                "rows_eligible": 2266,
                "rows_excluded": 2,
                "excluded_rows_are_method_unavailable_not_biological_zero": True,
                "exclusion_table": comparison._file_record(exclusion_path),
            }
            manifest["design"].update(
                {
                    "method_unavailable_policy": (
                        "database rows listed in excluded_lr_rows.csv must be excluded "
                        "from CellChat cross-method universes, never zero-filled"
                    )
                }
            )
        _json(directory / "manifest.json", manifest)

    nichenet = root / "04_nichenet"
    nichenet_shared = nichenet / "01_shared_inputs"
    nichenet_shared.mkdir(parents=True)
    expression_path = nichenet_shared / "expression_by_stage_celltype.csv.gz"
    pd.DataFrame(
        [
            {
                "stage_id": stage,
                "stage_label": stage_label,
                "cell_type": cell_type,
                "n_cells": 3 if cell_type == "A" else 2,
                "gene_mouse": "Gene1",
                "pct_detected": 0.5,
                "mean_normalized_linear": 1.0,
                "mean_log1p": 0.5,
            }
            for stage, stage_label in STAGES
            for cell_type in ("A", "B")
        ]
    ).to_csv(expression_path, index=False, compression="gzip")
    prepare_manifest_path = nichenet_shared / "prepare_manifest.json"
    _json(
        prepare_manifest_path,
        {
            "schema_version": 2,
            "workflow": "reviewer_zebrafish_nichenet_shared_input_preparation",
            "status": "complete",
            "orthology_policy": "one2one_bijective_all_confidence",
            "analysis_tier": "sensitivity",
            "primary_claim_allowed": False,
            "output_files": [
                _prepare_record(expression_path, relative_to=nichenet_shared)
            ],
        },
    )
    for directory_name, mode in (
        ("02_default_mouse_v2", "default"),
        ("03_custom_zebrafish_lr", "custom"),
    ):
        directory = nichenet / directory_name
        directory.mkdir(parents=True)
        rows = []
        status_rows = []
        for stage_index, (stage, stage_label) in enumerate(STAGES):
            unit_id = f"unit_{stage_index}"
            status = "complete" if stage_index < 4 else "skipped_nichenet_ineligible"
            status_rows.append(
                {
                    "unit_id": unit_id,
                    "source_stage_id": str(stage),
                    "target_stage_id": str(stage + 1),
                    "source_stage_label": stage_label,
                    "target_stage_label": f"target_{stage_index}",
                    "receiver": "A",
                    "mode": mode,
                    "input_status": "eligible",
                    "status": status,
                    "detail": ""
                    if status == "complete"
                    else "too_few_potential_ligands",
                }
            )
            if status == "complete":
                rows.append(
                    {
                        "unit_id": unit_id,
                        "source_stage_id": str(stage),
                        "source_stage_label": stage_label,
                        "target_stage_id": str(stage + 1),
                        "receiver": "A",
                        "mode": mode,
                        "sender": "A",
                        "ligand": f"L{stage_index}",
                        "sender_ligand_pct_detected": 0.5,
                        "sender_ligand_mean_normalized_linear": 1.0,
                        "aupr_corrected": float(0.4 - 0.05 * stage_index),
                        "ligand_activity_rank": 1,
                        "activity_scope": (
                            "transition_receiver_ligand_not_sender_specific"
                        ),
                    }
                )
        sender_path = directory / "sender_ligand_activity.csv"
        unit_path = directory / "unit_status.csv"
        pd.DataFrame(rows).to_csv(sender_path, index=False)
        pd.DataFrame(status_rows).to_csv(unit_path, index=False)
        _json(
            directory / "run_manifest.json",
            {
                "workflow": "reviewer_zebrafish_cross_species_nichenet_v2",
                "status": "complete",
                "mode": mode,
                "activity_semantics": "not a direct sender-specific strength",
                "orthology_policy": "one2one_bijective_all_confidence",
                "analysis_tier": "sensitivity",
                "primary_claim_allowed": False,
                "method_label": (
                    "cross-species NicheNet-v2 "
                    "[orthology sensitivity: confidence unfiltered]"
                ),
                "official_prior": {
                    "md5_verified": True,
                    "expected_md5": {"ligand_target_matrix": "a" * 32},
                    "observed_md5": {"ligand_target_matrix": "a" * 32},
                },
                "software": {
                    "nichenetr": {
                        "mode": "pinned_core_source",
                        "version": "2.2.1.1",
                        "version_verified": True,
                        "git_commit": comparison.PINNED_NICHENETR_COMMIT,
                        "expected_git_commit": comparison.PINNED_NICHENETR_COMMIT,
                        "commit_verified": True,
                        "core_md5_verified": True,
                        "observed_core_md5": {"application_prediction.R": "b" * 32},
                    }
                },
                "shared_prepare_manifest": {
                    "path": str(prepare_manifest_path),
                    "md5": comparison._md5(prepare_manifest_path),
                },
                "counts": {"units_complete": 4},
                "output_files": {
                    "sender_ligand_activity": str(sender_path),
                    "unit_status": str(unit_path),
                },
                "output_md5": {
                    "sender_ligand_activity": comparison._md5(sender_path),
                    "unit_status": comparison._md5(unit_path),
                },
            },
        )

    cellagent = root / "05_cellagentchat"
    cellagent_shared = cellagent / "shared"
    cellagent_shared.mkdir(parents=True)
    cellagent_prepare = cellagent_shared / "manifest.json"
    cellagent_prepare.write_text("{}", encoding="utf-8")
    mapped_expression = cellagent_shared / "mapped_expression.h5ad"
    mapped_expression.write_bytes(b"fixture")
    source_file = cellagent_shared / "model_setup.py"
    source_file.write_text("# pinned fixture\n", encoding="utf-8")
    sample_plan_path = cellagent_shared / "shared_sampled_cells.csv.gz"
    pd.DataFrame(
        [
            {
                "sampling_seed": seed,
                "stage": stage,
                "stage_label": stage_label,
                "cell_type": cell_type,
                "obs_name": f"{seed}_{stage:g}_{cell_type}",
            }
            for seed in (101, 202, 303)
            for stage, stage_label in STAGES
            for cell_type in ("A", "B")
        ]
    ).to_csv(sample_plan_path, index=False, compression="gzip")
    for condition in (
        comparison.CELLAGENTCHAT_OFFICIAL,
        comparison.CELLAGENTCHAT_CUSTOM,
    ):
        directory = cellagent / condition
        directory.mkdir(parents=True)
        frame = _type_rows("cellagentchat_native_primary_mean", [4.0, 3.0, 2.0, 1.0])
        frame["n_sampling_seeds"] = 3
        type_pair_path = directory / "cellagentchat_type_pair_scores.csv"
        frame.to_csv(type_pair_path, index=False)
        _json(
            directory / "manifest.json",
            {
                "method": "official_cellagentchat_v0_2_0_spatial",
                "database_condition": condition,
                "design": {
                    "native_primary": (
                        "number of Bonferroni-significant LR pairs per directed cell-type pair"
                    ),
                    "stages": [stage for stage, _ in STAGES],
                    "sampling_seeds": [101, 202, 303],
                    "epochs": 50,
                    "permutation_score_target": 10_000,
                    "spatial": True,
                    "permutation_background_distance_scaled": True,
                },
                "source": {
                    "release": "v0.2.0",
                    "expected_commit": comparison.PINNED_CELLAGENTCHAT_COMMIT,
                    "observed_commit": comparison.PINNED_CELLAGENTCHAT_COMMIT,
                    "pinned_source_verified": True,
                    "files": {"model_setup.py": comparison._file_record(source_file)},
                },
                "shared_input": {
                    "preparation_manifest": comparison._file_record(cellagent_prepare),
                    "mapped_expression": comparison._file_record(mapped_expression),
                    "sample_plan": comparison._file_record(sample_plan_path),
                    "preparation_claims": {
                        "orthology_policy": "one2one_bijective_all_confidence",
                        "orthology_analysis_tier": "sensitivity",
                        "primary_claim_allowed": False,
                    },
                },
                "counts": {
                    "n_runs": 15,
                    "type_pair_rows_by_seed": len(frame) * 3,
                },
                "artifacts": {
                    type_pair_path.name: comparison._file_record(type_pair_path)
                },
            },
        )
    _json(
        cellagent / "manifest.json",
        {
            "workflow": "official_cellagentchat_spatial_dual_lr_database",
            "status": "complete",
            "conditions": [
                comparison.CELLAGENTCHAT_OFFICIAL,
                comparison.CELLAGENTCHAT_CUSTOM,
            ],
            "same_mapped_expression_and_sample_plan_verified": True,
            "same_preparation_manifest_and_orthology_claims_verified": True,
            "same_formal_design_and_pinned_source_verified": True,
            "exact_stage_seed_grid_verified": True,
            "formal_non_smoke_verified": True,
            "database_sha256_are_distinct": True,
            "formal_design": {
                "stages": [stage for stage, _ in STAGES],
                "sampling_seeds": [101, 202, 303],
                "epochs": 50,
                "permutation_score_target": 10_000,
                "source_commit": comparison.PINNED_CELLAGENTCHAT_COMMIT,
            },
            "condition_manifests": {
                condition: comparison._file_record(
                    cellagent / condition / "manifest.json"
                )
                for condition in (
                    comparison.CELLAGENTCHAT_OFFICIAL,
                    comparison.CELLAGENTCHAT_CUSTOM,
                )
            },
        },
    )


def _args(
    root: Path, output: Path, *, allow_partial: bool = False
) -> argparse.Namespace:
    return argparse.Namespace(
        run_root=root,
        output_dir=output,
        cytobridge_dir=None,
        controls_trained_dir=None,
        controls_init_dir=None,
        controls_random_dir=None,
        commot_dir=None,
        cellchat_dir=None,
        nichenet_default_dir=None,
        nichenet_custom_dir=None,
        cellagentchat_dir=None,
        top_k=2,
        allow_partial=allow_partial,
        overwrite=False,
    )


def test_formal_comparison_writes_all_rank_and_control_artifacts(
    tmp_path: Path,
) -> None:
    root = tmp_path / "run"
    _build_formal_tree(root)
    output = tmp_path / "comparison"
    manifest = comparison.run(_args(root, output))

    assert manifest["status"] == "complete"
    assert manifest["formal_reviewer_ready"] is True
    assert len(manifest["loaded_score_views"]) == 8
    canonical = pd.read_csv(output / "canonical_type_pair_scores.csv.gz")
    assert canonical["view_id"].nunique() == 8
    full_grid_ids = {
        "commot__project_lr",
        "cellchat__project_lr",
        "cellagentchat__official_mouse_default",
        "cellagentchat__project_lr",
    }
    full_grid = canonical.loc[canonical["view_id"].isin(full_grid_ids)]
    assert set(full_grid["stage"]) == {0.0, 1.0, 2.0, 3.0, 4.0}
    assert full_grid.groupby(["view_id", "stage"]).size().eq(4).all()
    for view_id in ("commot__project_lr", "cellchat__project_lr"):
        stage0 = canonical.loc[
            (canonical["view_id"] == view_id) & (canonical["stage"] == 0.0)
        ]
        assert len(stage0) == 4
        assert stage0["native_score"].eq(0).all()
        assert stage0["structural_zero_filled"].all()
    nichenet_completed = canonical.loc[
        canonical["view_id"].str.startswith("nichenet_v2__")
    ]
    assert set(nichenet_completed["stage"]) == {0.0, 1.0, 2.0, 3.0}
    assert set(nichenet_completed["receiver_type"]) == {"A"}
    assert nichenet_completed.groupby(["view_id", "stage"]).size().eq(2).all()
    assert nichenet_completed.loc[
        nichenet_completed["sender_type"] == "B", "structural_zero_filled"
    ].all()
    summary = pd.read_csv(output / "pairwise_consistency_summary.csv")
    assert not summary.empty
    assert summary["mean_stage_spearman"].between(-1, 1).all()
    compared_pairs = set(zip(summary["view_id_left"], summary["view_id_right"])) | set(
        zip(summary["view_id_right"], summary["view_id_left"])
    )
    assert ("cytobridge__trained__attention", "commot__project_lr") in compared_pairs
    controls = pd.read_csv(output / "cytobridge_control_metrics.csv")
    trained = controls.loc[
        (controls["control"] == "trained")
        & (controls["target"] == "attention")
        & (controls["metric"] == "conditional_residual_spearman_forward_lr")
    ]
    assert trained["estimate"].item() == pytest.approx(0.0284)
    assert manifest["contract"]["raw_cross_method_units_compared"] is False
    assert manifest["contract"]["cytobridge_attention_is_ccc_probability"] is False
    assert (
        manifest["contract"]["cellchat_method_unavailable_lr_rows_zero_filled"] is False
    )
    assert manifest["cellchat_method_unavailable_lr_rows"]["count"] == 2
    assert manifest["cellchat_method_unavailable_lr_rows"]["zero_filled"] is False
    assert manifest["six_condition_execution_complete"] is True
    assert manifest["reviewer_reporting_ready"] is True
    assert all(manifest["formal_readiness_checks"].values())
    assert (
        manifest["formal_readiness_checks"]["all_primary_score_artifacts_hash_verified"]
        is True
    )
    assert set(
        manifest["primary_score_artifact_hash_verification"]["score_views"]
    ) == set(manifest["loaded_score_views"])
    assert set(
        manifest["primary_score_artifact_hash_verification"]["cytobridge_controls"]
    ) == {
        "trained",
        "init_interaction",
        "randomized_interaction_seed17",
    }
    assert set(manifest["score_view_zero_completion"]) == set(
        manifest["loaded_score_views"]
    )
    zero_audit = pd.read_csv(output / "structural_zero_audit.csv")
    assert set(zero_audit["view_id"]) == set(manifest["loaded_score_views"])
    assert not zero_audit["unevaluated_units_zero_filled"].any()
    assert not zero_audit["method_unavailable_lr_rows_zero_filled"].any()
    unavailable = pd.read_csv(output / "method_unavailable_lr_rows.csv")
    assert unavailable["database_row"].tolist() == [2281, 2292]
    assert not unavailable["zero_filled"].any()
    nichenet = canonical.loc[canonical["method"] == "NicheNet-v2 (cross-species)"]
    assert set(nichenet["orthology_policy"]) == {"one2one_bijective_all_confidence"}
    assert set(nichenet["analysis_tier"]) == {"sensitivity"}
    assert (
        nichenet["display_label"]
        .str.contains("all-confidence orthology sensitivity", regex=False)
        .all()
    )
    cellagent = canonical.loc[canonical["method"] == "CellAgentChat (cross-species)"]
    assert set(cellagent["orthology_policy"]) == {"one2one_bijective_all_confidence"}
    assert set(cellagent["analysis_tier"]) == {"sensitivity"}
    assert (
        cellagent["display_label"]
        .str.contains("all-confidence orthology sensitivity", regex=False)
        .all()
    )
    for stem in (
        "rank_concordance",
        "top_edge_overlap",
        "condition_coverage",
        "directionality_concordance",
        "stage_stability",
        "cytobridge_control_panel",
    ):
        assert (output / f"{stem}.png").stat().st_size > 1000
        assert (output / f"{stem}.pdf").stat().st_size > 1000


@pytest.mark.parametrize(
    "relative_path",
    [
        "01_cytobridge/type_pair_summary.csv",
        "03_external_ccc/commot_current_lr/commot_type_pair_scores.csv.gz",
        "03_external_ccc/cellchat_current_lr/cellchat_type_pair_scores.csv.gz",
        *[
            f"{directory}/{filename}"
            for directory in (
                "02_attention_controls",
                "02_attention_controls_init_interaction",
                "02_attention_controls_random_seed17",
            )
            for filename in (
                "conditional_permutation_tests.csv",
                "nested_grouped_cv_metrics.csv",
                "reciprocal_edge_direction_tests.csv",
            )
        ],
    ],
)
def test_formal_comparison_fails_closed_on_tampered_primary_artifact(
    tmp_path: Path, relative_path: str
) -> None:
    root = tmp_path / "run"
    _build_formal_tree(root)
    path = root / relative_path
    payload = bytearray(path.read_bytes())
    assert payload
    payload[len(payload) // 2] ^= 1
    path.write_bytes(payload)

    with pytest.raises(ValueError, match="Manifest SHA256 does not match"):
        comparison.run(_args(root, tmp_path / "tampered"))


def test_formal_missing_condition_fails_but_partial_records_it(tmp_path: Path) -> None:
    root = tmp_path / "run"
    _build_formal_tree(root)
    missing = (
        root
        / "03_external_ccc"
        / "cellchat_current_lr"
        / "cellchat_type_pair_scores.csv.gz"
    )
    missing.unlink()
    with pytest.raises(FileNotFoundError):
        comparison.run(_args(root, tmp_path / "formal_failure"))

    output = tmp_path / "partial"
    manifest = comparison.run(_args(root, output, allow_partial=True))
    assert manifest["status"] == "partial_diagnostic"
    assert manifest["formal_reviewer_ready"] is False
    assert "cellchat__project_lr" not in manifest["loaded_score_views"]
    diagnostics = pd.read_csv(output / "input_diagnostics.csv")
    assert "cellchat__project_lr" in set(diagnostics["view_id"])
    coverage = pd.read_csv(output / "condition_coverage.csv")
    row = coverage.loc[coverage["view_id"] == "cellchat__project_lr"]
    assert set(row["status"]) == {"missing_or_invalid_method"}


def test_nichenet_manifest_policy_must_be_explicit_and_paired(tmp_path: Path) -> None:
    root = tmp_path / "run"
    _build_formal_tree(root)
    custom_manifest_path = (
        root / "04_nichenet" / "03_custom_zebrafish_lr" / "run_manifest.json"
    )
    custom_manifest = json.loads(custom_manifest_path.read_text())
    custom_manifest.update(
        {
            "orthology_policy": "strict_confidence1",
            "analysis_tier": "primary",
            "primary_claim_allowed": True,
            "method_label": "custom-LR-constrained cross-species mapped NicheNet-v2",
        }
    )
    _json(custom_manifest_path, custom_manifest)
    with pytest.raises(ValueError, match="orthology_policy"):
        comparison.run(_args(root, tmp_path / "mismatched"))


def test_external_stage_id_and_hpf_time_must_match_shared_manifest(
    tmp_path: Path,
) -> None:
    root = tmp_path / "run"
    _build_formal_tree(root)
    commot_path = (
        root
        / "03_external_ccc"
        / "commot_current_lr"
        / "commot_type_pair_scores.csv.gz"
    )
    commot = pd.read_csv(commot_path)
    assert set(commot["stage"].astype(str)) == {"1.0", "2.0", "3.0", "4.0"}
    assert set(commot["stage_time"]) == {10.0, 12.0, 18.0, 24.0}
    commot.loc[commot["stage"].astype(str) == "1.0", "stage_time"] = 10.5
    commot.to_csv(commot_path, index=False, compression="gzip")
    commot_manifest_path = commot_path.parent / "manifest.json"
    commot_manifest = json.loads(commot_manifest_path.read_text())
    commot_manifest["artifacts"]["type_pair_scores"] = comparison._file_record(
        commot_path
    )
    _json(commot_manifest_path, commot_manifest)
    with pytest.raises(
        ValueError, match="disagree with the verified shared input manifest"
    ):
        comparison.run(_args(root, tmp_path / "bad_stage_mapping"))


def test_cellagentchat_conditions_must_share_manifest_orthology_claims(
    tmp_path: Path,
) -> None:
    root = tmp_path / "run"
    _build_formal_tree(root)
    custom_manifest_path = (
        root / "05_cellagentchat" / comparison.CELLAGENTCHAT_CUSTOM / "manifest.json"
    )
    custom_manifest = json.loads(custom_manifest_path.read_text())
    custom_manifest["shared_input"]["preparation_claims"] = {
        "orthology_policy": "strict_confidence1",
        "orthology_analysis_tier": "primary",
        "primary_claim_allowed": True,
    }
    _json(custom_manifest_path, custom_manifest)
    parent_path = root / "05_cellagentchat" / "manifest.json"
    parent = json.loads(parent_path.read_text())
    parent["condition_manifests"][
        comparison.CELLAGENTCHAT_CUSTOM
    ] = comparison._file_record(custom_manifest_path)
    _json(parent_path, parent)
    with pytest.raises(ValueError, match="CellAgentChat conditions do not share"):
        comparison.run(_args(root, tmp_path / "mismatched_cellagent"))


def test_pairwise_top_k_is_computed_on_exact_shared_key_universe() -> None:
    rows = []
    for view_id, scores, keys in (
        ("left", [4.0, 3.0, 2.0], [("A", "B"), ("B", "A"), ("C", "A")]),
        ("right", [9.0, 8.0, 7.0], [("B", "A"), ("A", "B"), ("D", "A")]),
    ):
        for score, (sender, receiver) in zip(scores, keys):
            rows.append(
                {
                    "view_id": view_id,
                    "display_label": view_id,
                    "method": view_id,
                    "database_condition": "test",
                    "score_view": "test",
                    "stage": 0.0,
                    "stage_label": "t0",
                    "sender_type": sender,
                    "receiver_type": receiver,
                    "native_score": score,
                    "heterotypic": sender != receiver,
                }
            )
    frame = comparison._add_stage_ranks(pd.DataFrame(rows))
    by_stage, summary = comparison.pairwise_consistency(frame, top_k=2)
    row = by_stage.iloc[0]
    assert row["n_shared_directed_pairs"] == 2
    assert row["effective_top_k"] == 2
    assert row["top_k_jaccard"] == 1.0
    assert not row["top_k_informative"]
    assert summary["n_top_k_informative_stages"].item() == 0
    assert np.isnan(summary["mean_stage_top_k_jaccard_informative_only"].item())


def test_top_k_summary_separates_trivial_all_selected_stages() -> None:
    rows = []
    keys = [("A", "B"), ("B", "A"), ("C", "A")]
    for stage, n_keys in ((0.0, 2), (1.0, 3)):
        for view_id, values in (
            ("left", [3.0, 2.0, 1.0]),
            ("right", [3.0, 1.0, 2.0]),
        ):
            for score, (sender, receiver) in zip(values[:n_keys], keys[:n_keys]):
                rows.append(
                    {
                        "view_id": view_id,
                        "display_label": view_id,
                        "method": view_id,
                        "database_condition": "test",
                        "score_view": "test",
                        "stage": stage,
                        "stage_label": f"t{stage:g}",
                        "sender_type": sender,
                        "receiver_type": receiver,
                        "native_score": score,
                        "heterotypic": True,
                    }
                )
    frame = comparison._add_stage_ranks(pd.DataFrame(rows))
    by_stage, summary = comparison.pairwise_consistency(frame, top_k=2)
    stage0 = by_stage.loc[by_stage["stage"] == 0.0].iloc[0]
    stage1 = by_stage.loc[by_stage["stage"] == 1.0].iloc[0]
    assert not stage0["top_k_informative"]
    assert stage0["top_k_jaccard"] == 1.0
    assert stage1["top_k_informative"]
    assert stage1["top_k_jaccard"] == pytest.approx(1 / 3)
    result = summary.iloc[0]
    assert result["n_top_k_informative_stages"] == 1
    assert result["mean_stage_top_k_jaccard_all_stages"] == pytest.approx(2 / 3)
    assert result["mean_stage_top_k_jaccard_informative_only"] == pytest.approx(1 / 3)


def test_control_loader_selects_strict_stratum_not_row_position(tmp_path: Path) -> None:
    directory = tmp_path / "control"
    _write_control(directory, attention=0.02, message=0.03)
    path = directory / "conditional_permutation_tests.csv"
    frame = pd.read_csv(path)
    decoy = frame.iloc[[0]].copy()
    decoy["strata"] = "stage+distance_bin"
    decoy["observed_spearman"] = 0.99
    pd.concat([decoy, frame], ignore_index=True).to_csv(path, index=False)
    manifest_path = directory / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"]["conditional_permutations"] = comparison._file_record(path)
    _json(manifest_path, manifest)

    result = comparison.load_cytobridge_control(
        directory, control="trained", display_label="Trained"
    )
    selected = result.loc[
        (result["target"] == "attention")
        & (result["metric"] == "conditional_residual_spearman_forward_lr")
    ]
    assert selected["estimate"].item() == pytest.approx(0.02)


def _two_view_scores(left: list[float], right: list[float]) -> pd.DataFrame:
    keys = [("A", "A"), ("A", "B"), ("B", "A"), ("B", "B")]
    rows = []
    for view_id, values in (("left", left), ("right", right)):
        for score, (sender, receiver) in zip(values, keys):
            rows.append(
                {
                    "view_id": view_id,
                    "display_label": view_id,
                    "method": view_id,
                    "database_condition": "test",
                    "score_view": "test",
                    "stage": 0.0,
                    "stage_label": "t0",
                    "sender_type": sender,
                    "receiver_type": receiver,
                    "native_score": score,
                    "heterotypic": sender != receiver,
                    "structural_zero_filled": score == 0,
                }
            )
    return comparison._add_stage_ranks(pd.DataFrame(rows))


def test_top_k_all_zero_support_is_na_not_false_perfect_overlap() -> None:
    frame = _two_view_scores([0, 0, 0, 0], [0, 0, 0, 0])
    by_stage, summary = comparison.pairwise_consistency(frame, top_k=20)
    row = by_stage.iloc[0]
    assert row["n_positive_left"] == 0
    assert row["n_positive_right"] == 0
    assert row["effective_top_k"] == 0
    assert not row["top_k_informative"]
    assert np.isnan(row["top_k_jaccard"])
    assert summary["n_finite_spearman_stages"].item() == 0


def test_top_k_caps_at_positive_support_and_never_selects_zero_tail() -> None:
    frame = _two_view_scores([3, 0, 0, 0], [4, 2, 0, 0])
    by_stage, _ = comparison.pairwise_consistency(frame, top_k=20)
    row = by_stage.iloc[0]
    assert row["n_positive_left"] == 1
    assert row["n_positive_right"] == 2
    assert row["effective_top_k"] == 1
    assert row["top_k_left_realized_set_size"] == 1
    assert row["top_k_right_realized_set_size"] == 1
    assert row["top_k_jaccard"] == 1.0


def test_top_k_expands_kth_boundary_ties_without_label_or_order_truncation() -> None:
    frame = _two_view_scores([3, 2, 2, 0], [3, 2, 2, 0])
    frame = frame.sample(frac=1.0, random_state=19).reset_index(drop=True)
    by_stage, _ = comparison.pairwise_consistency(frame, top_k=2)
    row = by_stage.iloc[0]
    assert row["effective_top_k"] == 2
    assert row["top_k_left_realized_set_size"] == 3
    assert row["top_k_right_realized_set_size"] == 3
    assert row["top_k_left_boundary_tie_count"] == 2
    assert row["top_k_right_boundary_tie_count"] == 2
    assert row["top_k_left_boundary_tie_expanded"]
    assert row["top_k_right_boundary_tie_expanded"]
    assert row["top_k_jaccard"] == 1.0


def test_external_grid_gaps_require_explicit_positive_only_runner_policy(
    tmp_path: Path,
) -> None:
    root = tmp_path / "run"
    _build_formal_tree(root)
    manifest_path = root / "03_external_ccc" / "commot_current_lr" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["design"].pop("long_table_zero_policy")
    _json(manifest_path, manifest)
    with pytest.raises(ValueError, match="does not explicitly declare positive-only"):
        comparison.run(_args(root, tmp_path / "bad_zero_policy"))


def test_cellagentchat_missing_native_grid_row_fails_closed(tmp_path: Path) -> None:
    root = tmp_path / "run"
    _build_formal_tree(root)
    condition = comparison.CELLAGENTCHAT_OFFICIAL
    directory = root / "05_cellagentchat" / condition
    score_path = directory / "cellagentchat_type_pair_scores.csv"
    scores = pd.read_csv(score_path).iloc[:-1]
    scores.to_csv(score_path, index=False)
    condition_manifest_path = directory / "manifest.json"
    condition_manifest = json.loads(condition_manifest_path.read_text())
    condition_manifest["artifacts"][score_path.name] = comparison._file_record(
        score_path
    )
    _json(condition_manifest_path, condition_manifest)
    parent_path = root / "05_cellagentchat" / "manifest.json"
    parent = json.loads(parent_path.read_text())
    parent["condition_manifests"][condition] = comparison._file_record(
        condition_manifest_path
    )
    _json(parent_path, parent)
    with pytest.raises(ValueError, match="complete verified stage/type square"):
        comparison.run(_args(root, tmp_path / "missing_cag_grid"))


def test_cellagentchat_smoke_seed_design_cannot_be_reviewer_ready(
    tmp_path: Path,
) -> None:
    root = tmp_path / "run"
    _build_formal_tree(root)
    condition = comparison.CELLAGENTCHAT_OFFICIAL
    condition_manifest_path = root / "05_cellagentchat" / condition / "manifest.json"
    condition_manifest = json.loads(condition_manifest_path.read_text())
    condition_manifest["design"]["sampling_seeds"] = [101]
    _json(condition_manifest_path, condition_manifest)
    parent_path = root / "05_cellagentchat" / "manifest.json"
    parent = json.loads(parent_path.read_text())
    parent["condition_manifests"][condition] = comparison._file_record(
        condition_manifest_path
    )
    _json(parent_path, parent)
    with pytest.raises(ValueError, match="sampling seeds 101,202,303"):
        comparison.run(_args(root, tmp_path / "smoke_cag"))
