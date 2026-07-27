from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "scripts"
    / "reviewer_zebrafish_ccc"
    / "plain_language_consistency_report.py"
)
SPEC = importlib.util.spec_from_file_location("plain_language_consistency_report", SCRIPT)
assert SPEC and SPEC.loader
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


def test_top_set_keeps_boundary_ties() -> None:
    frame = pd.DataFrame({"score": [5.0, 4.0, 4.0, 1.0, 0.0]})
    assert report.top_set(frame, "score", 2) == {0, 1, 2}
    assert report.top_set(pd.DataFrame({"score": [0.0, 0.0]}), "score", 1) == set()


def test_stage_table_reproduces_formal_top_overlap() -> None:
    scores = pd.DataFrame(
        {
            "stage": [0.0] * 5,
            "sender_type": [f"S{i}" for i in range(5)],
            "receiver_type": ["R"] * 5,
            "cytobridge_attention": [5, 4, 3, 2, 1],
            "cytobridge_attention_rank": [1.0, 0.8, 0.6, 0.4, 0.2],
            "external_native_consensus": [5, 3, 4, 2, 1],
        }
    )
    formal = pd.DataFrame(
        {
            "target": ["CytoBridge attention"],
            "reference": ["External native consensus"],
            "stage": [0.0],
            "target_set_size_after_boundary_ties": [1],
            "reference_set_size_after_boundary_ties": [1],
            "intersection": [1],
        }
    )
    result = report.prepare_stage_table(scores, formal)
    assert result["shared_top20"].sum() == 1
    assert result.loc[result["shared_top20"], "sender_type"].item() == "S0"


def test_figure_index_separates_evidence_from_audit() -> None:
    index = report.build_figure_index().set_index("figure")
    assert index.loc["02_direct_ccc_comparison", "recommended_role"] == "最主要直接证据"
    assert index.loc["01_reviewer_evidence_map", "recommended_role"] == "新主图"
    assert index.loc["condition_coverage", "recommended_role"] == "质量审计"
    assert index.loc["directionality_concordance", "recommended_role"] == "限制/审计"
    formal = report.build_figure_index(spatial_audit_available=True).set_index("figure")
    assert formal.loc["07_spatial_null_sensitivity", "recommended_role"] == "空间主审计"
    assert "不构成额外独立空间验证" in formal.loc[
        "07_spatial_null_sensitivity", "plain_conclusion"
    ]


def test_figure_index_uses_current_metrics_instead_of_hardcoded_values() -> None:
    index = report.build_figure_index(
        headline={
            "attention_commot": 0.1234,
            "exact_message_commot": 0.5678,
            "attention_external": 0.2,
            "exact_message_external": 0.3,
            "top_enrichment": 1.4,
        }
    ).set_index("figure")
    conclusion = index.loc["02_direct_ccc_comparison", "plain_conclusion"]
    assert "0.123" in conclusion
    assert "0.568" in conclusion
    assert "0.566" not in conclusion


def test_reviewer_reply_without_formal_spatial_audit_does_not_claim_strong_overlap() -> None:
    direct = pd.DataFrame(
        {
            "attention_vs_commot_rho": [0.4, 0.6],
            "exact_message_vs_commot_rho": [0.6, 0.8],
            "attention_vs_external_consensus_rho": [0.3, 0.5],
            "exact_message_vs_external_consensus_rho": [0.5, 0.7],
        }
    )
    top = pd.DataFrame(
        {
            "target": ["CytoBridge attention", "CytoBridge attention"],
            "reference": [
                "External native consensus",
                "External native consensus",
            ],
            "overlap_enrichment_over_random": [1.4, 1.8],
        }
    )
    text = report.reviewer_reply_text(direct, top)
    assert "descriptive only" in text
    assert "formal fixed-support null was not supplied" in text
    assert "did not exceed the audited fixed-support null" not in text
    assert "most high-ranked" not in text
    assert "rho = 0.500" in text
    assert "rho = 0.700" in text
    assert "1.60-fold" in text


def test_spatial_audit_loader_rejects_tampered_artifact(tmp_path: Path) -> None:
    primary = pd.DataFrame(
        {
            "example_id": ["a", "b", "c"],
            "stage_label": ["18hpf"] * 3,
            "ligand": ["l1", "l2", "l3"],
            "receptor": ["r1", "r2", "r3"],
            "top_fraction": [0.2] * 3,
            "scale_factor": [0.5] * 3,
            "field_overlap_ovl": [0.1] * 3,
            "hdr80_dice": [0.1] * 3,
            "spatial_match_f1": [0.1] * 3,
        }
    )
    primary.to_csv(tmp_path / "spatial_primary_metrics.csv", index=False)
    pd.DataFrame(
        {
            "example_id": ["a", "b", "c"],
            "top_fraction": [0.2] * 3,
            "scale_factor": [0.5] * 3,
            "metric": ["field_overlap_ovl"] * 3,
            "observed": [0.1] * 3,
            "null_mean": [0.2] * 3,
            "null_ci_low": [0.15] * 3,
            "null_ci_high": [0.25] * 3,
            "empirical_p_greater_equal": [1.0] * 3,
            "n_permutations": [20] * 3,
        }
    ).to_csv(tmp_path / "spatial_null_sensitivity.csv.gz", index=False)
    pd.DataFrame(
        {
            "example_id": ["a"],
            "component": ["attention_lr"],
            "field_overlap_ovl": [0.1],
            "observed_minus_null_mean": [-0.1],
            "empirical_p_greater_equal": [1.0],
            "delta_vs_lr_only": [-0.1],
        }
    ).to_csv(tmp_path / "spatial_component_control_metrics.csv", index=False)
    pd.DataFrame(
        {
            "example_id": ["a"],
            "direction": ["outgoing"],
            "cell_mass_overlap_ovl": [0.1],
            "spearman_active_union_cells": [0.0],
            "positive_cell_support_jaccard": [0.1],
            "top20_positive_cell_jaccard": [0.1],
        }
    ).to_csv(tmp_path / "spatial_sender_receiver_metrics.csv", index=False)
    pd.DataFrame(
        {
            "example_id": ["a"],
            "analysis": ["primary_score_null"],
            "method": ["cytobridge"],
            "coarsening_level": ["fine_type_covariate"],
            "fraction_edges": [1.0],
            "n_strata": [1],
            "min_realized_stratum_size": [10],
            "movable_edge_fraction_overall": [1.0],
            "assignment_sha256": ["0" * 64],
        }
    ).to_csv(tmp_path / "permutation_strata_diagnostics.csv", index=False)
    (tmp_path / "README_CN.md").write_text("ok", encoding="utf-8")
    for name in report.SPATIAL_AUDIT_FIGURES:
        for suffix in ("png", "pdf"):
            (tmp_path / f"{name}.{suffix}").write_bytes(b"figure")
    artifacts = [
        report.record(path, tmp_path)
        for path in sorted(tmp_path.iterdir())
        if path.is_file()
    ]
    manifest = {
        "workflow": "zebrafish_spatial_coordinate_consistency",
        "parameters": {"permutations": 20, "max_global_fallback_fraction": 0.05},
        "claims": {
            "spatial_consistency_not_ground_truth": True,
            "midpoint_overlap_not_direction_accuracy": True,
            "component_control_required_for_attention_increment": True,
            "selected_examples_not_all_lr_axes": True,
        },
        "artifacts": artifacts,
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    report.load_spatial_audit(tmp_path)
    (tmp_path / "README_CN.md").write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA256 mismatch|byte count mismatch"):
        report.load_spatial_audit(tmp_path)


def test_homotypic_type_pair_is_not_mislabeled_as_cell_self_loop() -> None:
    label = report.pair_label("Somite", "Somite")
    assert "homotypic type pair" in label
    assert "self-loop" not in label


def test_direct_comparison_reconstructs_stagewise_commot_correlations() -> None:
    score_rows = []
    pair_rows = []
    consensus_rows = []
    for stage in range(5):
        for index in range(4):
            score_rows.append(
                {
                    "stage": float(stage),
                    "cytobridge_attention": float(index),
                    "cytobridge_exact_message": float(3 - index),
                    "commot": float(index),
                }
            )
        for target, rho, intersection in (
            ("CytoBridge attention", 1.0, 2),
            ("CytoBridge exact message", -1.0, 1),
        ):
            pair_rows.append(
                {
                    "display_label_left": target,
                    "display_label_right": "COMMOT | project LR",
                    "stage": float(stage),
                    "spearman_rank_concordance": rho,
                    "top_k_intersection": intersection,
                    "effective_top_k": 10,
                }
            )
            consensus_rows.append(
                {
                    "design": "external_only_native_primary",
                    "target": target,
                    "stage": float(stage),
                    "spearman": rho / 2,
                }
            )
    result = report.direct_comparison_table(
        pd.DataFrame(pair_rows),
        pd.DataFrame(consensus_rows),
        pd.DataFrame(score_rows),
    )
    assert result["attention_vs_commot_rho"].eq(1.0).all()
    assert result["exact_message_vs_commot_rho"].eq(-1.0).all()
    assert result["attention_vs_external_consensus_rho"].eq(0.5).all()
