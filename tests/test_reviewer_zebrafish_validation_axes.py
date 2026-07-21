from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from reviewer_zebrafish_ccc.validate_reviewer_axes import (  # noqa: E402
    ATTENTION_RESIDUAL,
    MESSAGE_RESIDUAL,
    _benjamini_hochberg,
    _conditional_rank_permutation,
    _sha256,
    attach_known_axis_provenance,
    run_context_enrichment_tests,
    run_degree_matched_tests,
    score_identifiable_lr_axes,
    summarize_sender_receiver_contexts,
    summarize_virtual_ablation,
    validate_ablation_bundle,
)


def test_benjamini_hochberg_is_monotone_in_ranked_p_values():
    p_values = np.array([0.04, 0.001, 0.02, 0.9])
    adjusted = _benjamini_hochberg(p_values)
    order = np.argsort(p_values)
    assert np.all(np.diff(adjusted[order]) >= 0)
    assert np.all(adjusted >= p_values)
    assert adjusted[1] == pytest.approx(0.004)


def _edge_table() -> pd.DataFrame:
    rows = []
    for index in range(20):
        score = index / 19
        rows.append(
            {
                "stage": 0.0,
                "stage_label": "5hpf",
                "sender_type": "A",
                "receiver_type": "B",
                "source_index": index,
                "target_index": 100 + index,
                "attention_abs_mean": 1.0 + score,
                "edge_message_norm_joint": 2.0 + score,
                ATTENTION_RESIDUAL: score,
                MESSAGE_RESIDUAL: score,
                "lr_compatibility_forward": score,
                "lr_compatibility_reverse": 1.0 - score,
                "active_lr_count": int(score > 0),
                "spatial_distance": 0.05,
                "residual_non_lr_pca_cosine": 0.0,
                "source_outdegree": 2,
                "target_indegree": 3,
            }
        )
    return pd.DataFrame(rows)


def test_sender_receiver_context_summary_preserves_exact_and_residual_metrics():
    edges = _edge_table()
    result = summarize_sender_receiver_contexts(edges, min_edges=10)
    assert len(result) == 1
    row = result.iloc[0]
    assert row["n_edges"] == 20
    assert row["n_sender_cells"] == 20
    assert row["n_receiver_cells"] == 20
    assert row["attention_mean"] == pytest.approx(1.5)
    assert row["exact_message_mean"] == pytest.approx(2.5)
    assert row["attention_confounder_residual_mean"] == pytest.approx(0.5)
    assert row["lr_supported_edge_fraction"] == pytest.approx(19 / 20)
    assert bool(row["passes_min_edges"])


def test_context_enrichment_is_stage_stratified_and_reports_reverse_control():
    rows = []
    for stage in (0.0, 1.0):
        for index in range(10):
            value = float(index)
            rows.append(
                {
                    "stage": stage,
                    "stage_label": f"t{stage:g}",
                    "sender_type": f"S{index}",
                    "receiver_type": "R",
                    "n_edges": 12,
                    "passes_min_edges": True,
                    "attention_mean": value,
                    "exact_message_mean": value,
                    "attention_confounder_residual_mean": value,
                    "exact_message_confounder_residual_mean": value,
                    "lr_forward_mean": value,
                    "lr_reverse_mean": 9.0 - value,
                }
            )
    tests = run_context_enrichment_tests(
        pd.DataFrame(rows), quantile=0.8, n_permutations=99, random_state=11
    )
    assert len(tests) == 8
    forward = tests.query(
        "target == 'attention_confounder_residual_mean' and score == 'lr_forward_mean'"
    ).iloc[0]
    reverse = tests.query(
        "target == 'attention_confounder_residual_mean' and score == 'lr_reverse_mean'"
    ).iloc[0]
    assert forward["high_minus_low_score"] > 0
    assert forward["within_stage_rank_correlation"] == pytest.approx(1.0)
    assert reverse["high_minus_low_score"] < 0
    assert reverse["within_stage_rank_correlation"] == pytest.approx(-1.0)


def test_conditional_rank_permutation_centers_each_matching_stratum():
    frame = pd.DataFrame(
        {
            "stratum": np.repeat(["x", "y"], 10),
            "target": np.tile(np.arange(10), 2),
            "score": np.tile(np.arange(10), 2),
        }
    )
    result = _conditional_rank_permutation(
        frame,
        target="target",
        score="score",
        keys=["stratum"],
        min_stratum_size=4,
        n_permutations=99,
        random_state=17,
    )
    assert result["conditional_rank_correlation"] == pytest.approx(1.0)
    assert result["n_edges_retained"] == 20
    assert result["n_strata"] == 2
    assert result["empirical_p_greater"] <= 0.02


def test_degree_matched_test_explicitly_includes_distance_state_and_degree():
    tests, audit = run_degree_matched_tests(
        _edge_table(),
        matching_bins=5,
        degree_bins=3,
        min_stratum_size=4,
        n_permutations=49,
        random_state=5,
    )
    assert len(tests) == 4
    assert not audit.empty
    strata = tests["strata"].iloc[0]
    for token in (
        "stage",
        "sender_type",
        "receiver_type",
        "distance_match_bin",
        "state_match_bin",
        "source_degree_match_bin",
        "target_degree_match_bin",
    ):
        assert token in strata
    forward = tests.query(
        f"target == '{ATTENTION_RESIDUAL}' and score == 'lr_compatibility_forward'"
    ).iloc[0]
    reverse = tests.query(
        f"target == '{ATTENTION_RESIDUAL}' and score == 'lr_compatibility_reverse'"
    ).iloc[0]
    assert forward["conditional_rank_correlation"] == pytest.approx(1.0)
    assert reverse["conditional_rank_correlation"] == pytest.approx(-1.0)


def test_identifiable_axis_scores_use_expression_direction_and_exact_message():
    edges = pd.DataFrame(
        {
            "stage": [0.0, 0.0, 0.0, 0.0],
            "stage_label": ["t0"] * 4,
            "sender_type": ["A", "A", "C", "C"],
            "receiver_type": ["B", "B", "D", "D"],
            "source_index": [0, 1, 2, 3],
            "target_index": [1, 0, 3, 2],
            "attention_abs_mean": [2.0, 1.0, 1.0, 1.0],
            "edge_message_norm_joint": [3.0, 1.0, 1.0, 1.0],
        }
    )
    axes = pd.DataFrame(
        {
            "axis_id": ["l->r", "z->q"],
            "ligand": ["l", "z"],
            "receptor": ["r", "q"],
            "database_rows": ["1", "2"],
            "pathways": ["P", "Q"],
            "categories": ["secreted", "contact"],
        }
    )
    activities = {
        "l": np.array([1.0, 0.0, 0.0, 0.0]),
        "r": np.array([0.0, 1.0, 0.0, 0.0]),
        "z": np.zeros(4),
        "q": np.zeros(4),
    }
    scores, top = score_identifiable_lr_axes(edges, axes, activities, top_per_stage=1)
    supported = scores.query("axis_id == 'l->r'").iloc[0]
    absent = scores.query("axis_id == 'z->q'").iloc[0]
    assert supported["n_active_edges"] == 1
    assert supported["mean_exact_message_times_lr_activity"] == pytest.approx(0.75)
    assert supported["top_exact_message_sender_type"] == "A"
    assert supported["top_exact_message_receiver_type"] == "B"
    assert absent["top_exact_message_sender_type"] == ""
    assert set(top["axis_id"]) == {"l->r"}
    assert set(top["ranking_target"]) == {"attention", "exact_message"}


def test_known_axis_provenance_must_match_exact_database_pair(tmp_path: Path):
    provenance = tmp_path / "known.csv"
    pd.DataFrame(
        {
            "ligand": ["l"],
            "receptor": ["r"],
            "evidence_scope": ["gene-family developmental relevance"],
            "source_ids": ["PMID:1"],
            "source_urls": ["https://pubmed.ncbi.nlm.nih.gov/1/"],
            "claim_guardrail": ["no exact direction claim"],
        }
    ).to_csv(provenance, index=False)
    axes = pd.DataFrame(
        {
            "ligand": ["l"],
            "receptor": ["r"],
            "axis_id": ["l->r"],
            "database_rows": ["1"],
            "pathways": ["P"],
            "categories": ["C"],
        }
    )
    scores = axes.assign(
        stage=0.0,
        stage_label="t0",
        n_model_edges=2,
        n_active_edges=1,
        active_edge_fraction=0.5,
        mean_scaled_lr_activity=0.2,
    )
    audit, stage = attach_known_axis_provenance(provenance, axes, scores)
    assert bool(audit["database_present"].iloc[0])
    assert stage["axis_id"].tolist() == ["l->r"]
    changed = pd.read_csv(provenance)
    changed["receptor"] = "missing"
    changed.to_csv(provenance, index=False)
    with pytest.raises(ValueError, match="not tied"):
        attach_known_axis_provenance(provenance, axes, scores)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _fake_ablation_bundle(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    experiment = run_dir / "ablation" / "experiment"
    experiment.mkdir(parents=True)
    metrics = pd.DataFrame(
        {
            "variant": ["remove_A"] * 4,
            "time": [0.0, 1.0, 0.0, 1.0],
            "space": ["spatial", "spatial", "latent", "latent"],
            "n_baseline": [10, 20, 10, 20],
            "n_ablation": [8, 15, 8, 15],
            "count_ratio": [0.8, 0.75, 0.8, 0.75],
            "centroid_shift": [1.0, 3.0, 2.0, 2.0],
            "baseline_rms_radius": [2.0, 3.0, 4.0, 4.0],
        }
    )
    composition = pd.DataFrame(
        {
            "variant": ["remove_A", "remove_A"],
            "time": [0.0, 1.0],
            "label": ["A", "A"],
            "baseline_fraction": [1.0, 1.0],
            "ablation_fraction": [0.0, 0.5],
        }
    )
    metrics_path = experiment / "ablation_metrics.csv"
    composition_path = experiment / "label_composition.csv"
    manifest_path = experiment / "manifest.json"
    metrics.to_csv(metrics_path, index=False)
    composition.to_csv(composition_path, index=False)
    _write_json(
        manifest_path,
        {
            "settings": {
                "simulation": (
                    "continuous split-SDE; no re-anchoring; no spatial warp; "
                    "no replacement"
                ),
                "common_random_seed": True,
                "concat_spatial": True,
                "spatial_dim": 2,
                "start_time": 0.0,
                "random_stream_coupling": "same branch-level seed",
                "simulation_seeds": {"baseline": 42, "remove_A": 42},
                "ablations": {"remove_A": ["A"]},
                "n_initial": 10,
                "variant_initial_counts": {"remove_A": 8},
            }
        },
    )
    artifacts = []
    for path in (metrics_path, composition_path, manifest_path):
        artifacts.append(
            {"path": f"/old/experiment/{path.name}", "sha256": _sha256(path)}
        )
    _write_json(
        run_dir / "ablation" / "stage_manifest.json",
        {"output_artifacts": artifacts},
    )
    _write_json(
        run_dir / "run_manifest.json",
        {
            "profile": "full",
            "completed_stages": ["ablation"],
            "common": {
                "aligned_h5ad_sha256": "h5",
                "weight_sha256": "weight",
            },
        },
    )
    return run_dir


def test_full_ablation_bundle_is_hash_checked_and_summarized(tmp_path: Path):
    run_dir = _fake_ablation_bundle(tmp_path)
    metrics, composition, metadata = validate_ablation_bundle(
        run_dir,
        expected_h5ad_sha256="h5",
        expected_model_weight_sha256="weight",
    )
    summary, observed = summarize_virtual_ablation(metrics, composition)
    spatial = summary.query("space == 'spatial'").iloc[0]
    assert spatial["initial_normalized_centroid_shift"] == pytest.approx(0.5)
    assert spatial["endpoint_normalized_centroid_shift"] == pytest.approx(1.0)
    assert spatial["endpoint_normalized_shift_minus_t0"] == pytest.approx(0.5)
    assert spatial["endpoint_composition_total_variation"] == pytest.approx(0.25)
    assert set(observed["time"]) == {0.0, 1.0}
    assert metadata["single_seed_no_uncertainty"] is True

    run_manifest = json.loads((run_dir / "run_manifest.json").read_text())
    run_manifest["common"]["weight_sha256"] = "different"
    _write_json(run_dir / "run_manifest.json", run_manifest)
    with pytest.raises(ValueError, match="model-weight hash"):
        validate_ablation_bundle(
            run_dir,
            expected_h5ad_sha256="h5",
            expected_model_weight_sha256="weight",
        )
