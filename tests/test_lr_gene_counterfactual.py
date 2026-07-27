from __future__ import annotations

from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys

import anndata as ad
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SUPPORT = _load(
    "_lr_counterfactual_test_support",
    ROOT / "tests" / "test_downstream_perturbation.py",
)
RUNNER = _load(
    "lr_gene_counterfactual_under_test",
    ROOT / "scripts" / "reviewer_zebrafish_ccc" / "lr_gene_counterfactual.py",
)


def _sha(path: Path) -> str:
    digest = sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def test_cli_defaults_use_five_exact_message_technical_seeds():
    args = RUNNER._parser().parse_args(
        ["--h5ad", "input.h5ad", "--model-dir", "model", "--output-dir", "out"]
    )
    assert args.grouping_seeds == (101, 202, 303, 404, 505)


def _synthetic_adata() -> ad.AnnData:
    genes = ["cxcl12a", "cxcr4a", "sham_a", "sham_b", "sham_c", "sham_d"]
    stage0 = np.array(
        [
            [2.0, 0.0, 1.0, 0.5, 1.2, 0.2],
            [0.0, 2.0, 0.3, 1.4, 0.5, 1.0],
            [0.0, 1.0, 1.2, 0.4, 1.1, 0.8],
            [1.0, 0.0, 0.5, 1.1, 0.4, 1.3],
        ],
        dtype=np.float32,
    )
    stage1 = np.array(
        [
            [1.8, 0.0, 1.1, 0.6, 1.0, 0.3],
            [0.0, 2.1, 0.4, 1.2, 0.6, 1.1],
            [0.0, 1.2, 1.0, 0.5, 1.2, 0.7],
            [0.9, 0.0, 0.6, 1.0, 0.5, 1.2],
        ],
        dtype=np.float32,
    )
    expression = np.vstack((stage0, stage1))
    loadings = np.array(
        [
            [0.75, 0.10, 0.20],
            [0.15, 0.72, 0.12],
            [0.68, 0.14, 0.21],
            [0.61, 0.18, 0.27],
            [0.31, 0.58, 0.22],
            [0.25, 0.63, 0.18],
        ],
        dtype=np.float32,
    )
    center = expression.mean(axis=0)
    state = (expression - center) @ loadings
    spatial0 = np.array(
        [[0.0, 0.0], [1.0, 0.2], [0.3, 1.1], [1.2, 1.4]],
        dtype=np.float32,
    )
    spatial = np.vstack((spatial0, spatial0 + 0.05))
    data = ad.AnnData(
        X=expression,
        obs=pd.DataFrame(
            {
                "time_point_processed": [0.0] * 4 + [1.0] * 4,
                "Annotation": [
                    "sender",
                    "receiver",
                    "receiver",
                    "sender",
                ]
                * 2,
            },
            index=[f"cell_{index}" for index in range(8)],
        ),
        var=pd.DataFrame(index=genes),
    )
    data.obsm["X_latent"] = state.astype(np.float32)
    data.obsm["spatial_aligned"] = spatial
    data.varm["PCs"] = loadings
    data.var["highly_variable"] = True
    data.var["pca_center"] = center.astype(np.float32)
    return data


def test_synthetic_end_to_end_writes_strict_reviewer_bundle(tmp_path: Path):
    data = _synthetic_adata()
    model = SUPPORT.FakeDynamicalModel()
    output = tmp_path / "lr_counterfactual"
    with SUPPORT.loaded_perturbation_modules():
        manifest = RUNNER.run_analysis(
            data,
            model,
            output_dir=output,
            anchors=((0.0, 1.0),),
            fractions=(0.5, 1.0),
            n_shams=2,
            group_size=4,
            dt=1.0,
            grouping_seeds=(11, 17),
            metric_seed=23,
            max_ot_points=None,
            device="cpu",
        )

    assert manifest["status"] == "complete"
    disk_manifest = json.loads((output / "run_manifest.json").read_text())
    assert manifest == disk_manifest
    assert disk_manifest["axis"]["ligand_symbol"] == "cxcl12a"
    assert disk_manifest["axis"]["receptor_symbol"] == "cxcr4a"
    assert disk_manifest["design"]["full_rollout_grouping_seed"] == 11
    assert disk_manifest["design"]["exact_message_grouping_seeds"] == [11, 17]
    assert disk_manifest["evidence_contract"]["primary_outcome"] == (
        "counterfactual_D_target"
    )
    assert (
        disk_manifest["evidence_contract"][
            "counterfactual_missing_support_edge_message"
        ]
        == 0.0
    )
    assert disk_manifest["claim_bounds"]["experimental_causality"] is False
    assert disk_manifest["checks"]["attention_lr_expression_collapse_used"] is False
    assert disk_manifest["checks"]["every_sham_edits_at_least_one_primary_fixed_sender"]
    assert (
        disk_manifest["evidence_contract"][
            "primary_outcome_is_lr_specific_message_component"
        ]
        is False
    )
    assert disk_manifest["evidence_contract"]["joint_wasserstein_scale_dependent"]
    assert disk_manifest["design"][
        "ot_indices_common_across_conditions_doses_and_on_off"
    ]
    assert (
        disk_manifest["evidence_contract"]["message_alignment"][
            "same_grouping_plan_as_rollout_driver"
        ]
        is False
    )
    source = disk_manifest["reproducibility"]["analysis_script"]
    assert source["sha256"] == _sha(ROOT / source["path"])
    assert (
        len(
            disk_manifest["reproducibility"][
                "runtime_dependency_version_fingerprint_sha256"
            ]
        )
        == 64
    )
    git = disk_manifest["reproducibility"]["git"]
    assert git["available"]
    assert len(git["commit"]) == 40
    assert isinstance(git["dirty"], bool)
    assert isinstance(git["status_porcelain"], list)

    shams = pd.read_csv(output / "matched_hvg_shams.csv")
    assert len(shams) == 2
    assert shams["anchor_id"].eq("0_to_1").all()
    assert (
        shams["matching_compartment"]
        .eq("baseline_primary_ligand_positive_fixed_sender")
        .all()
    )
    target = pd.read_csv(output / "fixed_lr_target_message.csv")
    assert set(target["grouping_seed"]) == {11, 17}
    assert set(target["space"]) == {"joint", "spatial", "state"}
    assert target["complete_message_missing_edges_treated_as_zero"].all()
    assert not target["attention_missing_edges_treated_as_zero"].any()
    assert {
        "baseline_D_target",
        "counterfactual_D_target",
        "delta_D_target",
        "ratio_D_target",
    }.issubset(target.columns)
    monotonicity = pd.read_csv(output / "fixed_lr_target_monotonicity.csv")
    assert len(monotonicity) == 6  # one anchor x two seeds x three spaces

    metrics = pd.read_csv(output / "counterfactual_metrics.csv")
    fixed = metrics.loc[metrics["cohort"].eq("fixed_receptor_positive_ligand_negative")]
    assert {"w1", "w2", "centroid_shift"}.issubset(fixed.columns)
    assert set(fixed["interaction_enabled"].astype(str)) == {"True", "False"}
    common_ot = metrics.groupby(["anchor_id", "cohort"], sort=False)[
        ["ot_random_seed", "ot_support_index_sha256"]
    ].nunique()
    assert common_ot.eq(1).all().all()
    assert (
        metrics.loc[metrics["space"].eq("joint"), "wasserstein_scale_contract"]
        .str.contains("scale-dependent", regex=False)
        .all()
    )
    assert (output / "interaction_mediation.csv").is_file()
    diagnostic = pd.read_csv(output / "independent_message_alignment_diagnostic.csv")
    assert not diagnostic["same_grouping_plan_as_rollout_driver"].any()
    assert not diagnostic["eligible_for_primary_inference"].any()
    descriptive_shams = pd.read_csv(output / "matched_sham_descriptive_comparison.csv")
    assert descriptive_shams["formal_p_value_reported"].eq(False).all()
    assert (
        descriptive_shams["null_is_exchangeable_randomization_sample"].eq(False).all()
    )
    assert "empirical_upper_tail_p" not in descriptive_shams
    assert {
        "descriptive_sham_upper_tail_fraction",
        "descriptive_observed_relative_rank",
        "descriptive_pseudocount_tail_score_not_formal_pvalue",
    }.issubset(descriptive_shams)
    receiver_edits = pd.read_csv(output / "receiver_edit_audit.csv")
    sham_receiver_edits = receiver_edits.loc[
        receiver_edits["condition_role"].eq("matched_hvg_sham")
    ]
    assert sham_receiver_edits["n_directly_edited_fixed_receivers"].eq(0).all()
    assert sham_receiver_edits["max_fixed_receiver_projected_delta_norm"].eq(0).all()
    assert sham_receiver_edits["n_directly_edited_primary_fixed_senders"].gt(0).all()
    assert not sham_receiver_edits["fixed_receiver_intersects_selected_edit_mask"].any()
    assert (output / "README_CN.md").is_file()
    response = (output / "REVIEWER_RESPONSE.md").read_text()
    assert "trained CytoBridge model" in response
    assert "does not establish causal" in response

    for stem in (
        "dose_response_target_message",
        "receiver_wasserstein_dose_response",
        "matched_sham_interaction_mediation",
    ):
        assert (output / f"{stem}.png").stat().st_size > 1000
        assert (output / f"{stem}.pdf").stat().st_size > 1000

    checksum_lines = (output / "checksums.sha256").read_text().splitlines()
    checked = {}
    for line in checksum_lines:
        digest, relative = line.split("  ", 1)
        checked[relative] = digest
        assert _sha(output / relative) == digest
    assert "run_manifest.json" in checked
    assert "fixed_lr_target_message.csv" in checked
    assert "REVIEWER_RESPONSE.md" in checked
