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
STAGES = [(0.0, "5.25hpf"), (1.0, "10hpf")]
STAGE_TIMES = {0.0: 5.25, 1.0: 10.0}


def _json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


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
    _json(directory / "run_manifest.json", {"schema_version": 1})
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
    pd.DataFrame(permutation_rows).to_csv(
        directory / "conditional_permutation_tests.csv", index=False
    )
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
    ).to_csv(directory / "nested_grouped_cv_metrics.csv", index=False)
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
    ).to_csv(directory / "reciprocal_edge_direction_tests.csv", index=False)


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
    cyto["D_AB_joint_mean"] = np.tile([1.0, 2.0, 3.0, 4.0], 2)
    cyto.to_csv(cytobridge / "type_pair_summary.csv", index=False)

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
                rows.append(
                    {
                        "method": display,
                        "database_variant": "current_zebrafish_lr_database",
                        "stage": str(stage),
                        "stage_time": STAGE_TIMES[stage],
                        "sender_type": sender,
                        "receiver_type": receiver,
                        "abundance_controlled_score": float(
                            4 - ((pair_index + stage_index) % 4)
                        ),
                    }
                )
        pd.DataFrame(rows).to_csv(
            directory / f"{method}_type_pair_scores.csv.gz",
            index=False,
            compression="gzip",
        )
        manifest = {
            "method": display,
            "database_variant": "current_zebrafish_lr_database",
            "input_manifest": comparison._file_record(shared_manifest_path),
        }
        if method == "cellchat":
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
            manifest["design"] = {
                "method_unavailable_policy": (
                    "database rows listed in excluded_lr_rows.csv must be excluded "
                    "from CellChat cross-method universes, never zero-filled"
                )
            }
        _json(directory / "manifest.json", manifest)

    nichenet = root / "04_nichenet"
    for directory_name, mode in (
        ("02_default_mouse_v2", "default"),
        ("03_custom_zebrafish_lr", "custom"),
    ):
        directory = nichenet / directory_name
        directory.mkdir(parents=True)
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
            },
        )
        rows = []
        for stage_index, (stage, stage_label) in enumerate(STAGES):
            for pair_index, (sender, receiver) in enumerate(PAIRS):
                rows.append(
                    {
                        "source_stage_id": str(stage),
                        "source_stage_label": stage_label,
                        "target_stage_id": str(stage + 1),
                        "receiver": receiver,
                        "mode": mode,
                        "sender": sender,
                        "ligand": f"L{pair_index}",
                        "sender_ligand_pct_detected": 0.5,
                        "sender_ligand_mean_normalized_linear": 1.0,
                        "aupr_corrected": float(
                            0.4 - 0.1 * ((pair_index + stage_index) % 4)
                        ),
                        "ligand_activity_rank": pair_index + 1,
                        "activity_scope": (
                            "transition_receiver_ligand_not_sender_specific"
                        ),
                    }
                )
        pd.DataFrame(rows).to_csv(directory / "sender_ligand_activity.csv", index=False)

    cellagent = root / "05_cellagentchat"
    for condition in (
        comparison.CELLAGENTCHAT_OFFICIAL,
        comparison.CELLAGENTCHAT_CUSTOM,
    ):
        directory = cellagent / condition
        directory.mkdir(parents=True)
        _json(
            directory / "manifest.json",
            {
                "method": "official_cellagentchat_v0_2_0_spatial",
                "database_condition": condition,
                "design": {
                    "native_primary": (
                        "number of Bonferroni-significant LR pairs per directed cell-type pair"
                    )
                },
                "shared_input": {
                    "preparation_claims": {
                        "orthology_policy": "one2one_bijective_all_confidence",
                        "orthology_analysis_tier": "sensitivity",
                        "primary_claim_allowed": False,
                    }
                },
            },
        )
        frame = _type_rows(
            "cellagentchat_native_primary_mean", [4.0, 3.0, 2.0, 1.0]
        )
        frame.to_csv(directory / "cellagentchat_type_pair_scores.csv", index=False)


def _args(root: Path, output: Path, *, allow_partial: bool = False) -> argparse.Namespace:
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


def test_formal_comparison_writes_all_rank_and_control_artifacts(tmp_path: Path) -> None:
    root = tmp_path / "run"
    _build_formal_tree(root)
    output = tmp_path / "comparison"
    manifest = comparison.run(_args(root, output))

    assert manifest["status"] == "complete"
    assert manifest["formal_reviewer_ready"] is True
    assert len(manifest["loaded_score_views"]) == 8
    canonical = pd.read_csv(output / "canonical_type_pair_scores.csv.gz")
    assert canonical["view_id"].nunique() == 8
    assert canonical.groupby(["view_id", "stage"])[
        ["sender_type", "receiver_type"]
    ].size().eq(4).all()
    summary = pd.read_csv(output / "pairwise_consistency_summary.csv")
    assert not summary.empty
    assert summary["mean_stage_spearman"].between(-1, 1).all()
    compared_pairs = set(
        zip(summary["view_id_left"], summary["view_id_right"])
    ) | set(zip(summary["view_id_right"], summary["view_id_left"]))
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
    assert manifest["contract"]["cellchat_method_unavailable_lr_rows_zero_filled"] is False
    assert manifest["cellchat_method_unavailable_lr_rows"]["count"] == 2
    assert manifest["cellchat_method_unavailable_lr_rows"]["zero_filled"] is False
    unavailable = pd.read_csv(output / "method_unavailable_lr_rows.csv")
    assert unavailable["database_row"].tolist() == [2281, 2292]
    assert not unavailable["zero_filled"].any()
    nichenet = canonical.loc[canonical["method"] == "NicheNet-v2 (cross-species)"]
    assert set(nichenet["orthology_policy"]) == {
        "one2one_bijective_all_confidence"
    }
    assert set(nichenet["analysis_tier"]) == {"sensitivity"}
    assert nichenet["display_label"].str.contains(
        "all-confidence orthology sensitivity", regex=False
    ).all()
    cellagent = canonical.loc[
        canonical["method"] == "CellAgentChat (cross-species)"
    ]
    assert set(cellagent["orthology_policy"]) == {
        "one2one_bijective_all_confidence"
    }
    assert set(cellagent["analysis_tier"]) == {"sensitivity"}
    assert cellagent["display_label"].str.contains(
        "all-confidence orthology sensitivity", regex=False
    ).all()
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
    with pytest.raises(ValueError, match="do not share the same orthology_policy"):
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
    assert set(commot["stage"].astype(str)) == {"0.0", "1.0"}
    assert set(commot["stage_time"]) == {5.25, 10.0}
    commot.loc[commot["stage"].astype(str) == "0.0", "stage_time"] = 5.5
    commot.to_csv(commot_path, index=False, compression="gzip")
    with pytest.raises(ValueError, match="disagree with the verified shared input manifest"):
        comparison.run(_args(root, tmp_path / "bad_stage_mapping"))


def test_cellagentchat_conditions_must_share_manifest_orthology_claims(
    tmp_path: Path,
) -> None:
    root = tmp_path / "run"
    _build_formal_tree(root)
    custom_manifest_path = (
        root
        / "05_cellagentchat"
        / comparison.CELLAGENTCHAT_CUSTOM
        / "manifest.json"
    )
    custom_manifest = json.loads(custom_manifest_path.read_text())
    custom_manifest["shared_input"]["preparation_claims"] = {
        "orthology_policy": "strict_confidence1",
        "orthology_analysis_tier": "primary",
        "primary_claim_allowed": True,
    }
    _json(custom_manifest_path, custom_manifest)
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

    result = comparison.load_cytobridge_control(
        directory, control="trained", display_label="Trained"
    )
    selected = result.loc[
        (result["target"] == "attention")
        & (result["metric"] == "conditional_residual_spearman_forward_lr")
    ]
    assert selected["estimate"].item() == pytest.approx(0.02)
