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
STAGES = [(0.0, "t0"), (1.0, "t1")]


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
    for method, display in (("commot", "COMMOT"), ("cellchat", "CellChat")):
        directory = external / method
        directory.mkdir(parents=True)
        _json(
            directory / "manifest.json",
            {
                "method": display,
                "database_variant": "current_zebrafish_lr_database",
            },
        )
        rows = []
        for stage_index, (stage_time, stage_label) in enumerate(STAGES):
            for pair_index, (sender, receiver) in enumerate(PAIRS):
                rows.append(
                    {
                        "method": display,
                        "database_variant": "current_zebrafish_lr_database",
                        "stage": stage_label,
                        "stage_time": stage_time,
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
    controls = pd.read_csv(output / "cytobridge_control_metrics.csv")
    trained = controls.loc[
        (controls["control"] == "trained")
        & (controls["target"] == "attention")
        & (controls["metric"] == "conditional_residual_spearman_forward_lr")
    ]
    assert trained["estimate"].item() == pytest.approx(0.0284)
    assert manifest["contract"]["raw_cross_method_units_compared"] is False
    assert manifest["contract"]["cytobridge_attention_is_ccc_probability"] is False
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
    missing = root / "03_external_ccc" / "cellchat" / "cellchat_type_pair_scores.csv.gz"
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
    by_stage, _ = comparison.pairwise_consistency(frame, top_k=2)
    row = by_stage.iloc[0]
    assert row["n_shared_directed_pairs"] == 2
    assert row["effective_top_k"] == 2
    assert row["top_k_jaccard"] == 1.0


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
