from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from reviewer_zebrafish_ccc import build_reviewer_bundle as bundle  # noqa: E402


def _json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(
        path,
        index=False,
        compression="gzip" if path.suffix == ".gz" else None,
    )


def _artifact_manifest(directory: Path, names: tuple[str, ...]) -> dict:
    return {name: bundle._file_record(directory / name) for name in names}


def _write_methods(root: Path) -> dict[str, Path]:
    commot = root / "commot"
    commot.mkdir(parents=True)
    _json(
        commot / "manifest.json",
        {
            "method": "COMMOT",
            "database_variant": "current_zebrafish_lr_database",
            "score_semantics": {"raw_cross_method_units_comparable": False},
        },
    )

    cellchat = root / "cellchat"
    cellchat.mkdir(parents=True)
    _json(
        cellchat / "manifest.json",
        {
            "method": "CellChat",
            "database_variant": "current_zebrafish_lr_database",
            "database_validation": {
                "rows_requested": 10,
                "rows_eligible": 9,
                "rows_excluded": 1,
                "excluded_rows_are_method_unavailable_not_biological_zero": True,
            },
            "design": {
                "method_unavailable_policy": "excluded from the universe; never zero-filled"
            },
        },
    )

    nichenet_default = root / "nichenet_default"
    nichenet_custom = root / "nichenet_custom"
    for directory, mode, activities in (
        (nichenet_default, "default", 8),
        (nichenet_custom, "custom", 6),
    ):
        directory.mkdir(parents=True)
        _json(
            directory / "run_manifest.json",
            {
                "workflow": "reviewer_zebrafish_cross_species_nichenet_v2",
                "status": "complete",
                "mode": mode,
                "orthology_policy": "one2one_bijective_all_confidence",
                "analysis_tier": "sensitivity",
                "primary_claim_allowed": False,
                "activity_semantics": (
                    "sender and receptor assignments do not make the activity a direct "
                    "sender-specific, receptor-specific, spatial strength"
                ),
                "counts": {
                    "units_complete": 2,
                    "sender_ligand_activity_rows": activities,
                },
            },
        )

    cellagentchat = root / "cellagentchat"
    claims = {
        "orthology_policy": "one2one_bijective_all_confidence",
        "orthology_analysis_tier": "sensitivity",
        "primary_claim_allowed": False,
    }
    condition_paths = {}
    for condition, significant in (
        (bundle.CELLAGENTCHAT_OFFICIAL, 5),
        (bundle.CELLAGENTCHAT_CUSTOM, 3),
    ):
        directory = cellagentchat / condition
        directory.mkdir(parents=True)
        path = directory / "manifest.json"
        _json(
            path,
            {
                "method": "official_cellagentchat_v0_2_0_spatial",
                "database_condition": condition,
                "shared_input": {"preparation_claims": claims},
                "design": {
                    "native_primary": (
                        "number of Bonferroni-significant LR pairs per directed cell-type pair"
                    )
                },
                "counts": {"n_runs": 2, "significant_lr_rows": significant},
            },
        )
        condition_paths[condition] = bundle._file_record(path)
    _json(
        cellagentchat / "manifest.json",
        {
            "workflow": "official_cellagentchat_spatial_dual_lr_database",
            "conditions": [
                bundle.CELLAGENTCHAT_OFFICIAL,
                bundle.CELLAGENTCHAT_CUSTOM,
            ],
            "same_mapped_expression_and_sample_plan_verified": True,
            "same_preparation_manifest_and_orthology_claims_verified": True,
            "condition_manifests": condition_paths,
        },
    )
    return {
        "commot": commot,
        "cellchat": cellchat,
        "nichenet_default": nichenet_default,
        "nichenet_custom": nichenet_custom,
        "cellagentchat": cellagentchat,
    }


def _write_comparison(root: Path, methods: dict[str, Path]) -> Path:
    directory = root / "comparison"
    directory.mkdir()
    view_specs = [
        {"view_id": view_id, "display_label": view_id}
        for view_id in bundle.EXPECTED_VIEW_IDS
    ]
    canonical_rows = [
        {
            "view_id": view_id,
            "display_label": view_id,
            "method": "test",
            "database_condition": "test",
            "score_view": "test",
            "stage": 0,
            "stage_label": "5.25hpf",
            "sender_type": "A",
            "receiver_type": "B",
            "native_score": 1,
        }
        for view_id in bundle.EXPECTED_VIEW_IDS
    ]
    _csv(directory / "canonical_type_pair_scores.csv.gz", canonical_rows)
    pairwise_rows = []
    for left in bundle.INTERNAL_VIEW_IDS:
        for right in bundle.EXTERNAL_VIEW_IDS:
            pairwise_rows.append(
                {
                    "view_id_left": left,
                    "display_label_left": left,
                    "view_id_right": right,
                    "display_label_right": right,
                    "n_stages_compared": 1,
                    "n_finite_spearman_stages": 1,
                    "n_shared_directed_pairs_total": 20,
                    "mean_stage_spearman": 0.25,
                    "median_stage_spearman": 0.25,
                    "n_top_k_informative_stages": 1,
                    "mean_stage_top_k_jaccard_all_stages": 0.2,
                    "median_stage_top_k_jaccard_all_stages": 0.2,
                    "mean_stage_top_k_jaccard_informative_only": 0.2,
                    "median_stage_top_k_jaccard_informative_only": 0.2,
                }
            )
    _csv(directory / "pairwise_consistency_summary.csv", pairwise_rows)
    _csv(
        directory / "pairwise_consistency_by_stage.csv",
        [
            {
                "view_id_left": left,
                "view_id_right": right,
                "stage": 0,
                "n_shared_directed_pairs": 20,
                "spearman_rank_concordance": 0.25,
                "n_positive_left": 5,
                "n_positive_right": 3,
                "effective_top_k": 2,
                "top_k_left_realized_set_size": 3,
                "top_k_right_realized_set_size": 2,
                "top_k_left_boundary_score": 0.5,
                "top_k_right_boundary_score": 0.4,
                "top_k_left_boundary_tie_count": 2,
                "top_k_right_boundary_tie_count": 1,
                "top_k_left_boundary_tie_expanded": True,
                "top_k_right_boundary_tie_expanded": False,
                "top_k_selection_rule": "positive_support_boundary_tie_inclusive",
                "top_k_jaccard": 0.2,
            }
            for left in bundle.INTERNAL_VIEW_IDS
            for right in bundle.EXTERNAL_VIEW_IDS
        ],
    )
    _csv(directory / "reciprocal_rank_asymmetry.csv.gz", [{"view_id": "x", "stage": 0}])
    _csv(directory / "directionality_concordance_by_stage.csv", [{"stage": 0, "x": 1}])
    _csv(directory / "directionality_concordance_summary.csv", [{"x": 1}])
    _csv(directory / "stage_stability.csv", [{"view_id": "x", "status": "complete"}])
    _csv(
        directory / "condition_coverage.csv",
        [
            {
                "view_id": view_id,
                "status": "complete",
                "stage": 0,
                "stage_label": "5.25hpf",
                "n_directed_pairs": 20,
            }
            for view_id in bundle.EXPECTED_VIEW_IDS
        ],
    )
    _csv(
        directory / "cytobridge_control_metrics.csv",
        [
            {
                "control_label": "Trained",
                "target": "attention",
                "metric": "conditional_residual_spearman_forward_lr",
                "estimate": 0.1,
                "p_value": 0.02,
                "n_observations": 100,
            }
        ],
    )
    zero_completion = {}
    structural_rows = []
    for view_id in bundle.EXPECTED_VIEW_IDS:
        internal = view_id in bundle.INTERNAL_VIEW_IDS
        full_grid = view_id in bundle.FULL_STAGE_TYPE_SQUARE_VIEW_IDS
        expected_rows = 1 if internal else 20
        native_rows = 1 if internal else 5
        filled_rows = 0 if internal else 15
        zero_completion[view_id] = {
            "universe_scope": (
                "native_cytobridge_type_pair_summary"
                if internal
                else "verified_evaluated_stage_type_grid"
            ),
            "universe_source": None if internal else {"path": "/test/manifest.json"},
            "runner_export_contract": "native complete"
            if internal
            else "positive-only",
            "full_stage_type_square_required": full_grid,
            "expected_stage_count": 5 if full_grid else None,
            "observed_stage_count": 5 if full_grid else 1,
            "expected_rows": expected_rows,
            "native_emitted_rows": native_rows,
            "structural_zero_filled_rows": filled_rows,
            "verified_complete_evaluated_universe": not internal,
            "unevaluated_units_zero_filled": False,
            "method_unavailable_lr_rows_zero_filled": False,
        }
        if view_id in bundle.NICHENET_VIEW_IDS:
            zero_completion[view_id]["completed_units"] = 2
            zero_completion[view_id]["skipped_or_ineligible_units"] = 1
        structural_rows.append(
            {
                "view_id": view_id,
                "display_label": view_id,
                "method": "test",
                "database_condition": "test",
                "universe_scope": zero_completion[view_id]["universe_scope"],
                "stage": 0,
                "stage_label": "5.25hpf",
                "receiver_unit": "",
                "n_cell_types": 2,
                "expected_directed_rows": expected_rows,
                "native_emitted_rows": native_rows,
                "native_emitted_positive_rows": native_rows,
                "native_emitted_zero_rows": 0,
                "structural_zero_filled_rows": filled_rows,
                "verified_complete_evaluated_universe": not internal,
                "unevaluated_units_zero_filled": False,
                "method_unavailable_lr_rows_zero_filled": False,
                "provenance_manifest_path": "/test/manifest.json",
            }
        )
    _csv(directory / "structural_zero_audit.csv", structural_rows)
    _csv(
        directory / "method_unavailable_lr_rows.csv",
        [
            {
                "database_row": 10,
                "interaction_id": "l->r_complex",
                "ligand": "l",
                "receptor": "r_complex",
                "reason": "method token unavailable",
                "status": "method_unavailable_excluded_not_biological_zero",
                "zero_filled": False,
            }
        ],
    )
    _csv(directory / "input_diagnostics.csv", [{"status": "method_unavailable"}])
    for filename in bundle.COMPARISON_FIGURES:
        (directory / filename).write_bytes(b"test-figure")
    (directory / "README.md").write_text("comparison note\n", encoding="utf-8")

    artifacts = _artifact_manifest(
        directory,
        bundle.COMPARISON_TABLES + bundle.COMPARISON_FIGURES + ("README.md",),
    )
    orthology = {
        "orthology_policy": "one2one_bijective_all_confidence",
        "analysis_tier": "sensitivity",
        "primary_claim_allowed": False,
    }
    _json(
        directory / "manifest.json",
        {
            "workflow": "reviewer_zebrafish_multimethod_directed_ccc_rank_comparison",
            "status": "complete",
            "formal_reviewer_ready": True,
            "reviewer_reporting_ready": True,
            "six_condition_execution_complete": True,
            "allow_partial": False,
            "expected_score_views": view_specs,
            "loaded_score_views": list(bundle.EXPECTED_VIEW_IDS),
            "contract": {
                "raw_cross_method_units_compared": False,
                "cytobridge_attention_is_ccc_probability": False,
                "nichenet_type_pair_score_is_native": False,
                "cellchat_method_unavailable_lr_rows_zero_filled": False,
                "structural_zero_policy": "verified evaluated universe only",
                "top_k_selection_rule": (
                    "positive scores only; include every kth-score boundary tie"
                ),
                "top_k_tie_break": "none; kth-score boundary ties are expanded",
                "formal_expected_full_grid_stages": [0, 1, 2, 3, 4],
            },
            "formal_readiness_checks": {
                key: True for key in bundle.EXPECTED_FORMAL_READINESS_CHECKS
            },
            "readiness_semantics": {
                "readiness_is_not_primary_claim_permission": True,
                "condition_level_primary_claim_allowed_remains_authoritative": True,
            },
            "inputs": {
                key: {"path": str(path), "exists": True}
                for key, path in methods.items()
            },
            "nichenet_orthology_conditions": {
                "nichenet_v2__official_mouse_lr": orthology,
                "nichenet_v2__project_lr_gate": orthology,
            },
            "cellagentchat_orthology_conditions": {
                "cellagentchat__official_mouse_default": orthology,
                "cellagentchat__project_lr": orthology,
            },
            "cellchat_method_unavailable_lr_rows": {
                "count": 1,
                "status": "excluded_from_method_universe_not_biological_zero",
                "zero_filled": False,
            },
            "score_view_zero_completion": zero_completion,
            "artifacts": artifacts,
        },
    )
    return directory


def _write_validation(root: Path) -> Path:
    directory = root / "validation"
    directory.mkdir()
    _csv(directory / "sender_receiver_contexts.csv", [{"stage": 0, "n_edges": 10}])
    _csv(
        directory / "context_enrichment_tests.csv",
        [
            {
                "target": "attention_confounder_residual_mean",
                "score": "lr_forward_mean",
                "high_minus_low_score": 0.01,
                "within_stage_rank_correlation": 0.02,
                "empirical_p_greater": 0.5,
                "bh_q_within_context_family": 0.8,
            }
        ],
    )
    _csv(
        directory / "degree_matched_conditional_tests.csv",
        [
            {
                "target": "attention_residual",
                "score": "lr_compatibility_forward",
                "conditional_rank_correlation": 0.08,
                "observed_minus_null_mean": 0.08,
                "empirical_p_greater": 0.001,
                "bh_q_within_degree_matched_family": 0.004,
                "n_edges_retained": 100,
            }
        ],
    )
    for filename in bundle.VALIDATION_TABLES[3:]:
        _csv(directory / filename, [{"x": 1}])
    _csv(
        directory / "virtual_ablation_summary.csv",
        [
            {
                "variant": "remove_EVL",
                "space": "spatial",
                "endpoint_normalized_centroid_shift": 0.2,
                "endpoint_normalized_shift_minus_t0": 0.08,
                "endpoint_composition_total_variation": 0.1,
            }
        ],
    )
    _csv(directory / "virtual_ablation_observed_stages.csv", [{"time": 0, "x": 1}])
    for filename in bundle.VALIDATION_FIGURES:
        (directory / filename).write_bytes(b"test-validation-figure")
    (directory / "reviewer_validation_summary.md").write_text(
        "validation note\n", encoding="utf-8"
    )
    names = (
        bundle.VALIDATION_TABLES
        + bundle.OPTIONAL_VALIDATION_TABLES
        + bundle.VALIDATION_FIGURES
        + ("reviewer_validation_summary.md",)
    )
    _json(
        directory / "run_manifest.json",
        {
            "method": "zebrafish_reviewer_ccc_internal_consistency_axes",
            "claims": {
                "attention_is_ccc_probability": False,
                "exact_message_is_biochemical_flux": False,
                "lr_database_agreement_is_independent_validation": False,
                "virtual_ablation_is_causal_perturbation": False,
                "known_axis_literature_claim": False,
            },
            "artifacts": _artifact_manifest(directory, names),
        },
    )
    return directory


def _fixture(tmp_path: Path) -> argparse.Namespace:
    methods = _write_methods(tmp_path / "methods")
    comparison = _write_comparison(tmp_path, methods)
    validation = _write_validation(tmp_path)
    return argparse.Namespace(
        comparison_dir=comparison,
        validation_dir=validation,
        output_dir=tmp_path / "bundle",
        commot_dir=None,
        cellchat_dir=None,
        nichenet_default_dir=None,
        nichenet_custom_dir=None,
        cellagentchat_dir=None,
        overwrite=False,
    )


def _write_positive_consistency(root: Path) -> Path:
    directory = root / "positive_consistency"
    directory.mkdir()
    for filename in bundle.POSITIVE_CONSISTENCY_TABLES:
        path = directory / filename
        if filename.endswith(".csv.gz"):
            pd.DataFrame([{"value": 1}]).to_csv(path, index=False, compression="gzip")
        else:
            pd.DataFrame([{"value": 1}]).to_csv(path, index=False)
    for filename in bundle.POSITIVE_CONSISTENCY_FIGURES:
        (directory / filename).write_bytes(b"positive-figure")
    for filename in ("README.md", "reviewer_response_draft.md", "汇报说明.md"):
        (directory / filename).write_text("positive note\n", encoding="utf-8")
    names = (
        bundle.POSITIVE_CONSISTENCY_TABLES
        + bundle.POSITIVE_CONSISTENCY_FIGURES
        + ("README.md", "reviewer_response_draft.md", "汇报说明.md")
    )
    _json(
        directory / "manifest.json",
        {
            "workflow": "zebrafish_paper_style_positive_communication_consistency",
            "primary_design": {
                "name": "external_only_native_primary",
                "cytobridge_excluded": True,
            },
            "supporting_design": {"self_inclusion_disclosed": True},
            "cellagentchat_ctps_correction": {
                "source_column": "cellagentchat_significant_score_sum_mean",
                "source_files_mutated": False,
            },
            "artifacts": _artifact_manifest(directory, names),
        },
    )
    return directory


def test_builds_six_condition_bundle_with_claim_guardrails(tmp_path: Path) -> None:
    args = _fixture(tmp_path)
    manifest = bundle.build_bundle(args)
    assert manifest["formal_reviewer_ready"] is True
    assert len(manifest["conditions"]) == 6
    assert all("sensitivity only" in row["tier"] for row in manifest["conditions"][2:])
    assert manifest["claims"]["cytobridge_attention_is_ccc_probability"] is False
    assert (args.output_dir / "figures" / "rank_concordance.png").is_file()
    assert (args.output_dir / "tables" / "virtual_ablation_summary.csv").is_file()

    readme = (args.output_dir / "README.md").read_text(encoding="utf-8")
    assert "six external conditions" in readme
    assert "not a calibrated cell-cell communication (CCC) probability" in readme
    assert "method-unavailable" in readme
    assert "never zero-filled" in readme
    assert "sensitivity only" in readme
    assert "mouse ligand-target prior unchanged" in readme
    assert "Structural-zero provenance" in readme
    assert "Positive-support and boundary-tie audit" in readme
    assert "finite-rho stage count" in readme

    readme_cn = (args.output_dir / "README_CN.md").read_text(encoding="utf-8")
    assert "六个外部条件" in readme_cn
    assert "structural zero 审计" in readme_cn
    assert "不是 CCC probability" in readme_cn


def test_optionally_bundles_hash_verified_positive_consistency(tmp_path: Path) -> None:
    args = _fixture(tmp_path)
    args.positive_consistency_dir = _write_positive_consistency(tmp_path)
    manifest = bundle.build_bundle(args)
    assert manifest["positive_consistency_included"] is True
    assert (
        args.output_dir / "figures" / "positive_consistency_overview.png"
    ).is_file()
    assert (args.output_dir / "tables" / "consensus_summary.csv").is_file()
    assert (args.output_dir / "notes" / "reviewer_response_draft.md").is_file()
    assert "Positive-consistency addendum" in (
        args.output_dir / "README.md"
    ).read_text(encoding="utf-8")


def test_rejects_nonformal_comparison_before_creating_bundle(tmp_path: Path) -> None:
    args = _fixture(tmp_path)
    manifest_path = args.comparison_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["formal_reviewer_ready"] = False
    _json(manifest_path, manifest)
    with pytest.raises(ValueError, match="formal_reviewer_ready"):
        bundle.build_bundle(args)
    assert not args.output_dir.exists()


def test_rejects_legacy_positive_only_comparison_without_zero_audit(
    tmp_path: Path,
) -> None:
    args = _fixture(tmp_path)
    manifest_path = args.comparison_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("score_view_zero_completion")
    _json(manifest_path, manifest)
    with pytest.raises(ValueError, match="score_view_zero_completion"):
        bundle.build_bundle(args)
    assert not args.output_dir.exists()


def test_rejects_comparison_without_primary_artifact_hash_readiness(
    tmp_path: Path,
) -> None:
    args = _fixture(tmp_path)
    manifest_path = args.comparison_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["formal_readiness_checks"].pop("all_primary_score_artifacts_hash_verified")
    _json(manifest_path, manifest)
    with pytest.raises(ValueError, match="all_primary_score_artifacts_hash_verified"):
        bundle.build_bundle(args)
    assert not args.output_dir.exists()
