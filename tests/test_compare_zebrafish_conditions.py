from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "compare_zebrafish_conditions.py"
)
SPEC = importlib.util.spec_from_file_location("compare_zebrafish_conditions", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _metric_table(*, distance_offset: float, local_good: bool) -> pd.DataFrame:
    rows = []
    for time in (1.0, 2.0):
        for space in ("joint", "pca", "spatial"):
            rows.append(
                {
                    "time": time,
                    "space": space,
                    "w1": 0.20 + distance_offset,
                    "w2": 0.25 + distance_offset,
                    "tmv": 0.10 + distance_offset,
                    "nn_dispersion_ratio": 1.0 if local_good else 0.10,
                    "support_recall_at_observed_q95": 0.95 if local_good else 0.20,
                    "support_precision_at_observed_q95": 0.92 if local_good else 0.15,
                    "clump_fraction_at_0_1_observed_nn": 0.01 if local_good else 0.50,
                }
            )
    return pd.DataFrame(rows)


def _write_condition(
    run_root: Path,
    condition: str,
    metrics: pd.DataFrame,
    *,
    alpha_express: float,
) -> None:
    downstream = run_root / "conditions" / condition / "downstream"
    evaluation = downstream / "distribution_evaluation"
    training = run_root / "conditions" / condition / "training"
    evaluation.mkdir(parents=True)
    training.mkdir(parents=True)
    metrics_path = evaluation / "distribution_metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    manifest = {
        "condition": condition,
        "alpha_spatial": 10.0,
        "alpha_express": alpha_express,
        "input_h5ad_sha256": "same-input",
        "aligned_h5ad_sha256": "same-aligned",
        "training_dir": str(training.resolve()),
        "weight_checkpoint_sha256": f"weight-{condition}",
        "score_checkpoint_sha256": f"score-{condition}",
        "distribution_evaluation": {
            "paths": {"metrics": str(metrics_path)},
            "settings": {"dt": 0.01, "sigma": 0.03, "random_seed": 42},
        },
    }
    (downstream / "run_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    (training / "config.yaml").write_text("seed: 42\n", encoding="utf-8")
    (training / "launch_manifest.json").write_text("{}\n", encoding="utf-8")

    paper_root = run_root / "conditions" / condition / "paper_downstream"
    paper_root.mkdir(parents=True)
    paper_common = {
        "aligned_h5ad_sha256": "same-aligned",
        "weight_sha256": f"weight-{condition}",
        "score_sha256": f"score-{condition}",
        "model_dir": str(training.resolve()),
    }
    stage_manifests = {}
    stage_signatures = {}
    for stage in MODULE.PAPER_STAGES:
        stage_dir = paper_root / stage
        stage_dir.mkdir()
        audit_output = stage_dir / f"{stage}_audit.txt"
        audit_output.write_text(f"{condition}:{stage}\n", encoding="utf-8")
        outputs = [audit_output]
        if stage == "s22":
            panel = stage_dir / f"panel_{condition}.svg"
            panel.write_text(f"<svg id='{condition}'/>\n", encoding="utf-8")
            outputs.append(panel)
        stage_manifest = stage_dir / "stage_manifest.json"
        stage_signature = MODULE._stable_hash(
            {"stage": stage, "common": paper_common, "settings": {}}
        )
        stage_manifest.write_text(
            json.dumps(
                {
                    "stage": stage,
                    "status": "complete",
                    "signature": stage_signature,
                    "settings": {},
                    "outputs": [str(path.resolve()) for path in outputs],
                    "output_artifacts": [
                        {
                            "path": str(path.resolve()),
                            "size_bytes": path.stat().st_size,
                            "sha256": MODULE._sha256(path),
                        }
                        for path in outputs
                    ],
                }
            ),
            encoding="utf-8",
        )
        stage_manifests[stage] = str(stage_manifest.resolve())
        stage_signatures[stage] = stage_signature
    (paper_root / "run_manifest.json").write_text(
        json.dumps(
            {
                "workflow": "zebrafish_native_paper_downstream",
                "profile": "full",
                "completed_stages": list(MODULE.PAPER_STAGES),
                "stage_manifests": stage_manifests,
                "stage_signatures": stage_signatures,
                "common": paper_common,
                "model": {
                    "alpha_spatial": 10.0,
                    "alpha_express": alpha_express,
                },
            }
        ),
        encoding="utf-8",
    )
    (paper_root / "s22" / "unrecorded_stale_panel.svg").write_text(
        "<svg id='stale'/>\n", encoding="utf-8"
    )
    (paper_root / "classifier" / "classifier_cache.pt").write_text(
        "not-a-panel\n",
        encoding="utf-8",
    )


def test_auto_score_does_not_select_lower_w1_collapsed_model(tmp_path) -> None:
    run_root = tmp_path / "run"
    # The collapsed model wins all three distance criteria by a small amount,
    # while the structurally sound model wins all four local diagnostics.
    _write_condition(
        run_root,
        "lower_distance_but_collapsed",
        _metric_table(distance_offset=-0.01, local_good=False),
        alpha_express=0.015,
    )
    _write_condition(
        run_root,
        "sound_local_structure",
        _metric_table(distance_offset=0.0, local_good=True),
        alpha_express=0.05,
    )

    manifest = MODULE.run_comparison(
        run_root=run_root,
        output_dir=tmp_path / "comparison",
        conditions=("lower_distance_but_collapsed", "sound_local_structure"),
        baseline="lower_distance_but_collapsed",
    )

    assert manifest["selection"]["selected_condition"] == "sound_local_structure"
    assert manifest["automatic_scoring"]["policy"] == "equal_criterion_rank_v1"
    output = tmp_path / "comparison"
    assert (output / "model_metrics_long.csv").is_file()
    assert (output / "model_metrics_paired_deltas.csv").is_file()
    assert (output / "condition_ranking.csv").is_file()
    assert (output / "selection_criteria_long.csv").is_file()
    assert (output / "selection_manifest.json").is_file()


def test_review_bundle_copies_both_conditions_and_complete_audit(tmp_path) -> None:
    run_root = tmp_path / "run"
    for condition, offset in (("a", 0.0), ("b", -0.01)):
        _write_condition(
            run_root,
            condition,
            _metric_table(distance_offset=offset, local_good=True),
            alpha_express=0.01 if condition == "a" else 0.02,
        )
    logs = run_root / "logs"
    logs.mkdir()
    canonical_a = logs / "canonical_a_source.log"
    canonical_b = logs / "canonical_b_source.log"
    canonical_a.write_text("a done\n", encoding="utf-8")
    canonical_b.write_text("b done\n", encoding="utf-8")
    (logs / "stale_failed.log").write_text("old failure\n", encoding="utf-8")

    bundle = tmp_path / "review_bundle"
    manifest = MODULE.run_comparison(
        run_root=run_root,
        output_dir=bundle,
        conditions=("a", "b"),
        baseline="a",
        winner="b",
        bundle_mode="copy",
        canonical_logs={
            "a_canonical.log": canonical_a,
            "b_canonical.log": canonical_b,
        },
    )

    assert manifest["selection"]["reason"] == "explicit_user_override"
    for condition in ("a", "b"):
        panel = (
            bundle
            / "02_condition_panels"
            / condition
            / "s22"
            / f"panel_{condition}.svg"
        )
        assert panel.is_file()
        assert not panel.is_symlink()
        assert not (
            bundle
            / "02_condition_panels"
            / condition
            / "s22"
            / "unrecorded_stale_panel.svg"
        ).exists()
        assert (
            bundle
            / "03_condition_inputs"
            / condition
            / "paper_downstream"
            / "run_manifest.json"
        ).is_file()
        for stage in MODULE.PAPER_STAGES:
            assert (
                bundle
                / "03_condition_inputs"
                / condition
                / "paper_downstream"
                / stage
                / "stage_manifest.json"
            ).is_file()

    assert (
        bundle / "02_selected_manuscript_panels" / "s22" / "panel_b.svg"
    ).is_file()
    assert not (
        bundle
        / "02_selected_manuscript_panels"
        / "classifier"
        / "classifier_cache.pt"
    ).exists()
    assert not (
        bundle
        / "02_selected_manuscript_panels"
        / "s22"
        / "unrecorded_stale_panel.svg"
    ).exists()
    assert (
        bundle / "03_condition_inputs" / "a" / "distribution_metrics.csv"
    ).is_file()
    assert (bundle / "04_logs" / "a_canonical.log").read_text() == "a done\n"
    assert (bundle / "04_logs" / "b_canonical.log").read_text() == "b done\n"
    assert not (bundle / "04_logs" / "stale_failed.log").exists()

    readme = (bundle / "README.md").read_text(encoding="utf-8")
    assert "`a`" in readme
    assert "`b`" in readme
    assert "Selected condition: `b`" in readme
    commands = (
        bundle / "05_provenance" / "reproduction_commands.txt"
    ).read_text(encoding="utf-8")
    assert "--bundle-mode copy" in commands
    assert "a_canonical.log=" in commands
    assert 'test ! -e "$NEW_BUNDLE_DIR"' in commands

    inventory = pd.read_csv(
        bundle / "05_provenance" / "artifact_inventory.csv"
    )
    assert list(inventory.columns) == ["relative_path", "size_bytes", "sha256"]
    inventory_paths = set(inventory["relative_path"])
    assert "01_condition_comparison/selection_manifest.json" in inventory_paths
    assert "02_condition_panels/a/s22/panel_a.svg" in inventory_paths
    assert "02_condition_panels/b/s22/panel_b.svg" in inventory_paths
    assert "README.md" in inventory_paths
    assert "05_provenance/reproduction_commands.txt" in inventory_paths
    assert "05_provenance/artifact_inventory.csv" not in inventory_paths
    assert inventory["sha256"].str.fullmatch(r"[0-9a-f]{64}").all()
    assert (inventory["size_bytes"] >= 0).all()
    selection_row = inventory.loc[
        inventory["relative_path"]
        == "01_condition_comparison/selection_manifest.json"
    ].iloc[0]
    assert selection_row["sha256"] == MODULE._sha256(
        bundle / "01_condition_comparison" / "selection_manifest.json"
    )

    assert set(manifest["canonical_log_records"]) == {
        "a_canonical.log",
        "b_canonical.log",
    }
    assert manifest["outputs"]["bundle"]["artifact_inventory"].endswith(
        "artifact_inventory.csv"
    )
    with pytest.raises(FileExistsError, match="new empty output directory"):
        MODULE.run_comparison(
            run_root=run_root,
            output_dir=bundle,
            conditions=("a", "b"),
            baseline="a",
            winner="b",
            bundle_mode="copy",
            canonical_logs={"a_canonical.log": canonical_a},
            overwrite=True,
        )


def test_review_bundle_uses_explicit_final_paper_roots_without_copying_states(
    tmp_path,
) -> None:
    run_root = tmp_path / "run"
    paper_overrides = {}
    for condition in ("a", "b"):
        _write_condition(
            run_root,
            condition,
            _metric_table(distance_offset=0.0, local_good=True),
            alpha_express=0.01,
        )
        old_root = run_root / "conditions" / condition / "paper_downstream"
        final_root = (
            run_root
            / "conditions"
            / condition
            / "paper_downstream_final_20260718"
        )
        old_root.rename(final_root)
        rewritten_stage_manifests = {}
        for stage in MODULE.PAPER_STAGES:
            stage_manifest = final_root / stage / "stage_manifest.json"
            payload = json.loads(stage_manifest.read_text(encoding="utf-8"))
            payload["outputs"] = [
                str(final_root / Path(path).relative_to(old_root))
                for path in payload["outputs"]
            ]
            for record in payload["output_artifacts"]:
                record["path"] = str(
                    final_root / Path(record["path"]).relative_to(old_root)
                )
            stage_manifest.write_text(json.dumps(payload), encoding="utf-8")
            rewritten_stage_manifests[stage] = str(stage_manifest.resolve())
        root_manifest = final_root / "run_manifest.json"
        root_payload = json.loads(root_manifest.read_text(encoding="utf-8"))
        root_payload["stage_manifests"] = rewritten_stage_manifests
        root_manifest.write_text(json.dumps(root_payload), encoding="utf-8")
        (final_root / "s22" / "canonical_states.npz").write_bytes(b"large-state")
        paper_overrides[condition] = final_root

    bundle = tmp_path / "review_bundle_final"
    MODULE.run_comparison(
        run_root=run_root,
        output_dir=bundle,
        conditions=("a", "b"),
        baseline="a",
        winner="a",
        bundle_mode="copy",
        panel_overrides=paper_overrides,
        paper_output_overrides=paper_overrides,
    )

    for condition in ("a", "b"):
        assert (
            bundle
            / "02_condition_panels"
            / condition
            / "s22"
            / f"panel_{condition}.svg"
        ).is_file()
        assert not (
            bundle
            / "02_condition_panels"
            / condition
            / "s22"
            / "canonical_states.npz"
        ).exists()
        copied_root = (
            bundle
            / "03_condition_inputs"
            / condition
            / "paper_downstream"
        )
        assert (copied_root / "run_manifest.json").is_file()
        assert all(
            (copied_root / stage / "stage_manifest.json").is_file()
            for stage in MODULE.PAPER_STAGES
        )

    commands = (
        bundle / "05_provenance" / "reproduction_commands.txt"
    ).read_text(encoding="utf-8")
    assert "--paper-output" in commands
    assert "paper_downstream_final_20260718" in commands


def test_review_bundle_rejects_tampered_manifest_recorded_panel(tmp_path) -> None:
    run_root = tmp_path / "run"
    for condition in ("a", "b"):
        _write_condition(
            run_root,
            condition,
            _metric_table(distance_offset=0.0, local_good=True),
            alpha_express=0.01,
        )
    tampered = (
        run_root
        / "conditions"
        / "b"
        / "paper_downstream"
        / "s22"
        / "panel_b.svg"
    )
    tampered.write_text("<svg id='tampered'/>\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Recorded size mismatch|SHA-256 mismatch"):
        MODULE.run_comparison(
            run_root=run_root,
            output_dir=tmp_path / "tampered_bundle",
            conditions=("a", "b"),
            baseline="a",
            bundle_mode="copy",
        )


@pytest.mark.parametrize("unsafe", [".", "..", "a/b", "/absolute"])
def test_condition_names_must_be_safe_path_components(tmp_path, unsafe) -> None:
    with pytest.raises(ValueError, match="single safe path component"):
        MODULE.run_comparison(
            run_root=tmp_path / "run",
            output_dir=tmp_path / "bundle",
            conditions=(unsafe, "safe"),
            baseline=unsafe,
        )


def test_review_bundle_rejects_swapped_condition_paper_roots(tmp_path) -> None:
    run_root = tmp_path / "run"
    for condition, alpha in (("a", 0.01), ("b", 0.02)):
        _write_condition(
            run_root,
            condition,
            _metric_table(distance_offset=0.0, local_good=True),
            alpha_express=alpha,
        )
    root_a = run_root / "conditions" / "a" / "paper_downstream"
    root_b = run_root / "conditions" / "b" / "paper_downstream"

    with pytest.raises(ValueError, match="mismatch"):
        MODULE.run_comparison(
            run_root=run_root,
            output_dir=tmp_path / "swapped_bundle",
            conditions=("a", "b"),
            baseline="a",
            bundle_mode="copy",
            panel_overrides={"a": root_b, "b": root_a},
            paper_output_overrides={"a": root_b, "b": root_a},
        )


def test_review_bundle_rejects_failed_or_signature_mismatched_stage(tmp_path) -> None:
    run_root = tmp_path / "run"
    for condition in ("a", "b"):
        _write_condition(
            run_root,
            condition,
            _metric_table(distance_offset=0.0, local_good=True),
            alpha_express=0.01,
        )
    failed_manifest = (
        run_root
        / "conditions"
        / "b"
        / "paper_downstream"
        / "growth"
        / "stage_manifest.json"
    )
    payload = json.loads(failed_manifest.read_text(encoding="utf-8"))
    payload["status"] = "failed"
    failed_manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="not complete"):
        MODULE.run_comparison(
            run_root=run_root,
            output_dir=tmp_path / "failed_bundle",
            conditions=("a", "b"),
            baseline="a",
            bundle_mode="copy",
        )
