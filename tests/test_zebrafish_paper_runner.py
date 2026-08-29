from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_zebrafish_paper_downstream.py"
)
SPEC = importlib.util.spec_from_file_location("zebrafish_paper_runner", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def test_time_grid_and_stage_parser_are_strict():
    assert runner._time_grid(0.0, 1.0, 0.25) == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert runner._parse_stages("s25,communication,s25") == ["s25", "communication"]
    assert runner._parse_stages("all") == list(runner.ALL_STAGES)


def test_production_classifier_defaults_use_balanced_accuracy_and_k10():
    args = runner._build_parser().parse_args(
        [
            "--aligned-h5ad",
            "aligned.h5ad",
            "--model-dir",
            "model",
            "--output-dir",
            "output",
        ]
    )
    assert args.classifier_epochs == 500
    assert args.ablation_classifier_epochs == 500
    assert args.s25_classifier_knn_neighbors == 10
    assert args.communication_classifier_knn_neighbors == 10
    assert args.lr_expression_time_policy == "all_inverse_pca"
    assert args.lr_complex_mode == "min"

    sensitivity_args = runner._build_parser().parse_args(
        [
            "--aligned-h5ad",
            "aligned.h5ad",
            "--model-dir",
            "model",
            "--output-dir",
            "output",
            "--lr-complex-mode",
            "geometric_mean",
        ]
    )
    assert sensitivity_args.lr_complex_mode == "geometric_mean"


def _matched_acceptance_fixture(tmp_path):
    run_root = tmp_path / "matched"
    aligned = run_root / "zebrafish" / "preprocess" / "zebrafish_aligned.h5ad"
    aligned.parent.mkdir(parents=True)
    aligned.write_bytes(b"aligned")
    model = run_root / "zebrafish" / "training"
    model.mkdir()
    report = run_root / "matched_ablation_acceptance.json"
    report.write_text(
        json.dumps(
            {
                "run_root": str(run_root.resolve()),
                "status": "PASS",
                "datasets": {"zebrafish": {"status": "PASS"}},
                "matched_families": {"zebrafish": {"status": "PASS"}},
            }
        ),
        encoding="utf-8",
    )
    return run_root, aligned, model, report, runner._sha256(report)


def _acceptance_args(tmp_path):
    run_root, aligned, model, report, digest = _matched_acceptance_fixture(tmp_path)
    args = runner._build_parser().parse_args(
        [
            "--aligned-h5ad",
            str(aligned),
            "--model-dir",
            str(model),
            "--output-dir",
            str(run_root / "zebrafish" / "paper_downstream"),
            "--acceptance-report",
            str(report),
            "--expected-acceptance-sha256",
            digest,
        ]
    )
    return args, run_root, aligned, model, report


def test_formal_profile_requires_both_acceptance_arguments(tmp_path):
    with pytest.raises(ValueError, match="acceptance-report.*required"):
        runner.main(
            [
                "--aligned-h5ad",
                str(tmp_path / "aligned.h5ad"),
                "--model-dir",
                str(tmp_path / "model"),
                "--output-dir",
                str(tmp_path / "output"),
            ]
        )


def test_matched_acceptance_rejects_tampered_report(tmp_path):
    args, _run_root, aligned, model, report = _acceptance_args(tmp_path)
    report.write_text(report.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        runner._build_matched_acceptance_binding(
            args,
            aligned_h5ad=aligned,
            model_dir=model,
        )


@pytest.mark.parametrize("wrong_input", ["aligned", "model"])
def test_matched_acceptance_rejects_input_from_wrong_root(tmp_path, wrong_input):
    args, _run_root, aligned, model, _report = _acceptance_args(tmp_path)
    other = tmp_path / "other" / "zebrafish"
    if wrong_input == "aligned":
        aligned = other / "preprocess" / "zebrafish_aligned.h5ad"
        aligned.parent.mkdir(parents=True)
        aligned.write_bytes(b"other")
    else:
        model = other / "training"
        model.mkdir(parents=True)
    with pytest.raises(RuntimeError, match="outside canonical matched run_root"):
        runner._build_matched_acceptance_binding(
            args,
            aligned_h5ad=aligned,
            model_dir=model,
        )


def test_communication_classifier_knn_must_be_positive(tmp_path):
    with pytest.raises(
        ValueError, match="--communication-classifier-knn-neighbors must be > 0"
    ):
        runner.main(
            [
                "--aligned-h5ad",
                str(tmp_path / "missing.h5ad"),
                "--model-dir",
                str(tmp_path / "missing-model"),
                "--output-dir",
                str(tmp_path / "output"),
                "--communication-classifier-knn-neighbors",
                "0",
            ]
        )


def test_manuscript_layout_and_plot_provenance_contracts():
    assert runner.S22_MOSAIC_COLUMNS == 3
    assert runner.S25_HEATMAP_COLUMNS == 2
    required = {
        "CytoBridge/pl/trajectory.py",
        "CytoBridge/pl/growth.py",
        "CytoBridge/pl/temporal.py",
        "CytoBridge/tl/downstream/visualization.py",
        "CytoBridge/tl/downstream/checkpoint.py",
        "CytoBridge/tl/downstream/runtime.py",
    }
    assert required.issubset(set(runner.IMPLEMENTATION_SOURCE_RELATIVE_PATHS))
    expected_all_python = {
        path.relative_to(runner.REPO_ROOT).as_posix()
        for path in (runner.REPO_ROOT / "CytoBridge").rglob("*.py")
    }
    assert set(runner.IMPLEMENTATION_SOURCE_RELATIVE_PATHS) == expected_all_python


def _write_current_stage_manifest(tmp_path, *, stage, common, settings=None):
    settings = dict(settings or {})
    stage_dir = tmp_path / stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    output = stage_dir / "output.txt"
    output.write_text("complete\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "stage": stage,
        "status": "complete",
        "signature": runner._stable_hash(
            {"stage": stage, "common": common, "settings": settings}
        ),
        "settings": settings,
        "outputs": [str(output.resolve())],
        "output_artifacts": [
            {
                "path": str(output.resolve()),
                "size_bytes": output.stat().st_size,
                "sha256": runner._sha256(output),
            }
        ],
    }
    path = stage_dir / "stage_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path, output


def test_current_stage_manifest_rejects_changed_common_or_output(tmp_path):
    common = {
        "model_config_sha256": "config-a",
        "profile": "full",
        "data_contract": {"annotation_key": "Annotation"},
    }
    _, output = _write_current_stage_manifest(tmp_path, stage="s22", common=common)
    context = SimpleNamespace(output_dir=tmp_path, common_signature=common)
    assert (
        runner._require_current_stage_manifest(context, "s22")["status"] == "complete"
    )

    changed_context = SimpleNamespace(
        output_dir=tmp_path,
        common_signature={**common, "model_config_sha256": "config-b"},
    )
    with pytest.raises(RuntimeError, match="different data/model/code"):
        runner._require_current_stage_manifest(changed_context, "s22")

    output.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="missing or modified outputs"):
        runner._require_current_stage_manifest(context, "s22")


def test_acceptance_binding_is_explicit_in_stage_and_final_manifests(tmp_path):
    args, _run_root, aligned, model, _report = _acceptance_args(tmp_path)
    binding = runner._build_matched_acceptance_binding(
        args,
        aligned_h5ad=aligned,
        model_dir=model,
    )
    common = {runner.MATCHED_ACCEPTANCE_KEY: binding, "model": "current"}
    output_dir = tmp_path / "paper"
    context = SimpleNamespace(
        args=SimpleNamespace(
            force=False, profile="full", device="cpu", time_key="time"
        ),
        output_dir=output_dir,
        common_signature=common,
        loaded=SimpleNamespace(config={}),
        shared_cache_dir=tmp_path / "cache",
        adata=ad.AnnData(
            X=np.ones((1, 1)),
            obs={"time": [0.0]},
        ),
    )

    def action(stage_dir):
        output = stage_dir / "result.txt"
        output.write_text("complete\n", encoding="utf-8")
        return [output], {}

    manifest = runner._execute_stage(context, "classifier", {}, action)
    assert manifest[runner.MATCHED_ACCEPTANCE_KEY] == binding
    assert manifest["signature"] == runner._stable_hash(
        {"stage": "classifier", "common": common, "settings": {}}
    )

    root_manifest = runner._write_root_manifest(
        context,
        ["classifier"],
        {"classifier": manifest},
    )
    payload = json.loads(root_manifest.read_text(encoding="utf-8"))
    assert payload[runner.MATCHED_ACCEPTANCE_KEY] == binding
    assert payload["common"][runner.MATCHED_ACCEPTANCE_KEY] == binding
    assert payload["signature"] == runner._stable_hash(
        {
            "workflow": "zebrafish_native_paper_downstream",
            "common": common,
            "stage_signatures": {"classifier": manifest["signature"]},
        }
    )


def test_s25_interval_local_manifest_semantics_reject_global_t0_bundle(tmp_path):
    manifest_path = tmp_path / "stage_manifest.json"
    interval_local = {
        "settings": {
            "trajectory_mode": (
                "piecewise_observed_anchored_interval_forward_simulation"
            ),
            "split_sde_piecewise": True,
            "piecewise_observed_sample_mode": "per_timepoint",
            "piecewise_include_end": False,
            "daughter_noise_std": 0.0,
            "display_warp": {"applied": False},
            "simulation": runner.S25_COMMUNICATION_TRAJECTORY_SCOPE,
        },
        "details": {
            "trajectory_scope": runner.S25_COMMUNICATION_TRAJECTORY_SCOPE,
            "display_warp_applied": False,
        },
    }
    runner._require_s25_interval_local_manifest_semantics(interval_local, manifest_path)

    global_t0 = json.loads(json.dumps(interval_local))
    global_t0["settings"]["trajectory_mode"] = runner.S22_TRAJECTORY_MODE
    global_t0["settings"]["split_sde_piecewise"] = False
    global_t0["settings"]["piecewise_observed_sample_mode"] = None
    global_t0["settings"]["piecewise_include_end"] = None
    global_t0["settings"]["simulation"] = runner.S22_TRAJECTORY_SCOPE
    global_t0["details"]["trajectory_scope"] = runner.S22_TRAJECTORY_SCOPE
    with pytest.raises(RuntimeError, match="refusing incompatible/global-t0"):
        runner._require_s25_interval_local_manifest_semantics(global_t0, manifest_path)


def test_s25_does_not_implicitly_consume_neighboring_s22_bundle(tmp_path, monkeypatch):
    canonical = tmp_path / "s22" / "canonical_prewarp_states"
    canonical.mkdir(parents=True)
    (canonical / "index.json").write_text("{}\n", encoding="utf-8")
    args = runner._build_parser().parse_args(
        [
            "--aligned-h5ad",
            str(tmp_path / "aligned.h5ad"),
            "--model-dir",
            str(tmp_path / "model"),
            "--output-dir",
            str(tmp_path),
        ]
    )
    context = SimpleNamespace(
        output_dir=tmp_path,
        shared_cache_dir=tmp_path / "cache",
        common_signature={"model_config_sha256": "current"},
        args=args,
    )
    captured = {}

    def fake_execute(_ctx, stage, settings, action):
        captured.update(settings)
        return {"stage": stage}

    monkeypatch.setattr(runner, "_execute_stage", fake_execute)
    assert runner._stage_s25(context) == {"stage": "s25"}
    assert captured["canonical_trajectory"] is None


def _external_s22_test_context(tmp_path, *, common):
    bundle = tmp_path / "external_s22" / "canonical_prewarp_states"
    args = runner._build_parser().parse_args(
        [
            "--aligned-h5ad",
            str(tmp_path / "aligned.h5ad"),
            "--model-dir",
            str(tmp_path / "model"),
            "--output-dir",
            str(tmp_path / "current_run"),
            "--s25-canonical-state-bundle",
            str(bundle),
        ]
    )
    return (
        SimpleNamespace(
            args=args,
            output_dir=tmp_path / "current_run",
            shared_cache_dir=tmp_path / "shared_cache",
            common_signature=common,
        ),
        bundle,
    )


def _write_external_s22_bundle(bundle, *, common, settings):
    bundle.mkdir(parents=True)
    index = bundle / "index.json"
    index.write_text('{"frames": []}\n', encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "stage": "s22",
        "status": "complete",
        "signature": runner._stable_hash(
            {"stage": "s22", "common": common, "settings": settings}
        ),
        "settings": settings,
        "details": {
            "trajectory_scope": runner.S25_COMMUNICATION_TRAJECTORY_SCOPE,
            "display_warp_applied": False,
        },
        "outputs": [str(index.resolve())],
        "output_artifacts": [
            {
                "path": str(index.resolve()),
                "size_bytes": index.stat().st_size,
                "sha256": runner._sha256(index),
            }
        ],
    }
    if runner.MATCHED_ACCEPTANCE_KEY in common:
        manifest[runner.MATCHED_ACCEPTANCE_KEY] = common[runner.MATCHED_ACCEPTANCE_KEY]
    manifest_path = bundle.parent / "stage_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def _canonical_s22_test_settings(context):
    settings = {
        "trajectory_mode": ("piecewise_observed_anchored_interval_forward_simulation"),
        "split_sde_piecewise": True,
        "piecewise_observed_sample_mode": "per_timepoint",
        "piecewise_include_end": False,
        "display_warp": {"applied": False},
        "simulation": runner.S25_COMMUNICATION_TRAJECTORY_SCOPE,
    }
    settings.update(runner._expected_s22_state_settings(context))
    return settings


def test_s25_external_s22_signature_must_bind_current_common(tmp_path):
    current_common = {"model_config_sha256": "current"}
    context, bundle = _external_s22_test_context(tmp_path, common=current_common)
    settings = _canonical_s22_test_settings(context)
    _write_external_s22_bundle(
        bundle,
        common={"model_config_sha256": "stale"},
        settings=settings,
    )

    with pytest.raises(RuntimeError, match="different data/model/code"):
        runner._stage_s25(context)


@pytest.mark.parametrize("external_binding", ["missing", "stale"])
def test_s25_external_s22_must_bind_same_matched_acceptance(tmp_path, external_binding):
    args, _run_root, aligned, model, _report = _acceptance_args(tmp_path)
    binding = runner._build_matched_acceptance_binding(
        args,
        aligned_h5ad=aligned,
        model_dir=model,
    )
    common = {
        "model_config_sha256": "current",
        runner.MATCHED_ACCEPTANCE_KEY: binding,
    }
    context, bundle = _external_s22_test_context(tmp_path, common=common)
    settings = _canonical_s22_test_settings(context)
    manifest_path = _write_external_s22_bundle(
        bundle,
        common=common,
        settings=settings,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if external_binding == "missing":
        manifest.pop(runner.MATCHED_ACCEPTANCE_KEY)
    else:
        manifest[runner.MATCHED_ACCEPTANCE_KEY] = {
            **binding,
            "sha256": "0" * 64,
        }
    # Preserve a current common signature to prove that the explicit external
    # acceptance record is independently enforced.
    manifest["signature"] = runner._stable_hash(
        {"stage": "s22", "common": common, "settings": settings}
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match="missing or.*stale.*acceptance"):
        runner._stage_s25(context)


@pytest.mark.parametrize(
    ("mismatch", "expected_path"),
    [("sigma", "sigma"), ("classifier", "classifier.epochs")],
)
def test_s25_refuses_s22_state_setting_mismatch(tmp_path, mismatch, expected_path):
    common = {"model_config_sha256": "current"}
    context, bundle = _external_s22_test_context(tmp_path, common=common)
    settings = _canonical_s22_test_settings(context)
    if mismatch == "sigma":
        settings["sigma"] = float(settings["sigma"]) + 0.01
    else:
        settings["classifier"]["epochs"] = int(settings["classifier"]["epochs"]) + 1
    _write_external_s22_bundle(bundle, common=common, settings=settings)

    with pytest.raises(
        RuntimeError,
        match=rf"state-affecting settings.*{expected_path}",
    ):
        runner._stage_s25(context)


def test_communication_refuses_stale_existing_s25_states(tmp_path):
    common = {"model_config_sha256": "old"}
    _write_current_stage_manifest(tmp_path, stage="s25", common=common)
    states = tmp_path / "s25" / "generated_states"
    states.mkdir()
    (states / "index.json").write_text("{}\n", encoding="utf-8")
    lr_database = tmp_path / "lr.csv"
    lr_database.write_text("ligand,receptor\na,b\n", encoding="utf-8")
    context = SimpleNamespace(
        output_dir=tmp_path,
        common_signature={"model_config_sha256": "new"},
        args=SimpleNamespace(
            lr_database=lr_database,
            profile="full",
            smoke_n_samples=8,
            communication_max_cells=None,
        ),
    )
    with pytest.raises(RuntimeError, match="different data/model/code"):
        runner._stage_communication(context)


def test_generated_state_bundle_round_trip(tmp_path):
    states = {}
    for time_value, n_cells in ((0.0, 3), (0.5, 2)):
        points = np.arange(n_cells * 5, dtype=np.float32).reshape(n_cells, 5)
        state = ad.AnnData(X=points)
        state.obs["Annotation"] = [f"type_{idx}" for idx in range(n_cells)]
        state.obsm["spatial"] = points[:, :2]
        states[str(time_value)] = state

    files = runner._write_state_bundle(
        states,
        [0.0, 0.5],
        tmp_path,
        annotation_key="Annotation",
        source_by_time={0.0: "observed", 0.5: "generated_prewarp"},
    )
    assert all(path.exists() for path in files)

    restored, times, sources = runner._read_state_bundle(
        tmp_path, annotation_key="Annotation"
    )
    assert times == [0.0, 0.5]
    assert sources == {0.0: "observed", 0.5: "generated_prewarp"}
    np.testing.assert_array_equal(restored["0.5"].X, states["0.5"].X)
    assert restored["0.5"].obs["Annotation"].tolist() == ["type_0", "type_1"]


def test_communication_relabels_generated_states_without_changing_points(monkeypatch):
    observed_points = np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.float32)
    generated_points = np.asarray([[6, 7, 8], [9, 10, 11]], dtype=np.float32)
    observed = {
        0.0: runner._minimal_state_adata(
            observed_points, ["actual_a", "actual_b"], annotation_key="Annotation"
        )
    }
    generated = {
        "0.0": runner._minimal_state_adata(
            observed_points, ["legacy", "legacy"], annotation_key="Annotation"
        ),
        "0.5": runner._minimal_state_adata(
            generated_points, ["legacy", "legacy"], annotation_key="Annotation"
        ),
    }
    calls = []

    def fake_predict(**kwargs):
        calls.append(kwargs)
        label = "direct" if kwargs["knn_neighbors"] == 1 else "legacy"
        return np.asarray([label] * len(kwargs["points"]))

    monkeypatch.setattr(runner.cb.tl, "predict_labels_for_points", fake_predict)
    cached = SimpleNamespace(
        model=object(), label_encoder=object(), include_time_feature=True
    )
    (
        hybrid,
        sources,
        assignments,
        summary,
        counts,
    ) = runner._build_explicitly_labeled_hybrid_states(
        generated,
        observed,
        time_points=(0.0, 0.5),
        annotation_key="Annotation",
        cached_classifier=cached,
        classifier_feature_dim=3,
        device="cpu",
        knn_neighbors=1,
    )
    assert len(calls) == 1
    assert calls[0]["time_value"] == 0.5
    assert calls[0]["knn_neighbors"] == 1
    np.testing.assert_array_equal(hybrid["0.0"].X, observed_points)
    np.testing.assert_array_equal(hybrid["0.5"].X, generated_points)
    assert hybrid["0.0"].obs["Annotation"].tolist() == ["actual_a", "actual_b"]
    assert hybrid["0.5"].obs["Annotation"].tolist() == ["direct", "direct"]
    assert sources == {
        0.0: "observed_actual_annotation",
        0.5: "generated_interval_local_classifier_knn_1",
    }
    generated_summary = summary.loc[summary["time"].eq(0.5)].iloc[0]
    assert generated_summary["n_labels_changed"] == 2
    assert generated_summary["fraction_labels_changed"] == 1.0
    assert bool(generated_summary["points_preserved_exactly"])
    assert assignments["changed"].all()
    assert set(counts["label_policy"]) == {
        "actual_annotation",
        "inherited_bundle",
        "classifier_knn_1",
    }

    (
        legacy_hybrid,
        _,
        legacy_assignments,
        _,
        _,
    ) = runner._build_explicitly_labeled_hybrid_states(
        generated,
        observed,
        time_points=(0.0, 0.5),
        annotation_key="Annotation",
        cached_classifier=cached,
        classifier_feature_dim=3,
        device="cpu",
        knn_neighbors=10,
    )
    assert legacy_hybrid["0.5"].obs["Annotation"].tolist() == ["legacy", "legacy"]
    assert not legacy_assignments["changed"].any()


def test_lr_measurement_diagnostic_isolates_observed_expression_operator():
    def result(pair_scores):
        pair_rows = []
        cell_rows = []
        for pair, values in pair_scores.items():
            for time_value, score in zip((0.0, 0.5, 1.0), values):
                pair_rows.append({"pair": pair, "time": time_value, "score": score})
                cell_rows.append(
                    {
                        "pair": pair,
                        "time": time_value,
                        "cell_type": "A",
                        "incoming": score * 0.4,
                        "outgoing": score * 0.6,
                        "total": score,
                        "n_cells": 5,
                    }
                )
        import pandas as pd

        return SimpleNamespace(
            pair_timecourse=pd.DataFrame(pair_rows),
            celltype_timecourse=pd.DataFrame(cell_rows),
        )

    hybrid = result({"l1_r1": (2.0, 3.0, 4.0), "l2_r2": (4.0, 2.0, 1.0)})
    inverse = result({"l1_r1": (1.0, 3.0, 5.0), "l2_r2": (3.0, 2.0, 2.0)})
    (
        pair,
        celltype,
        metrics,
        continuity,
        continuity_metrics,
    ) = runner._compare_lr_measurement_contracts(
        hybrid, inverse, observed_times=(0.0, 1.0)
    )
    generated = metrics.loc[metrics["time"].eq(0.5)].iloc[0]
    assert generated["max_abs_delta"] == 0.0
    assert generated["n_zero_mismatches"] == 0
    observed = metrics.loc[metrics["time"].eq(0.0)].iloc[0]
    assert observed["rmse"] == 1.0
    assert observed["measurement_source_hybrid"] == "exact_observed_expression"
    assert pair.loc[pair["time"].eq(0.0), "score_abs_delta"].eq(1.0).all()
    assert celltype["n_cells_hybrid"].equals(celltype["n_cells_all_inverse_pca"])
    assert set(continuity["neighbor_mode"]) == {
        "one_sided_right",
        "one_sided_left",
    }
    assert set(continuity_metrics["anchor_time"]) == {0.0, 1.0}


def test_lr_measurement_diagnostic_rejects_generated_common_pair_drift():
    import pandas as pd

    def result(scores):
        pair = pd.DataFrame(
            {
                "pair": ["l_r"] * 3,
                "time": [0.0, 0.5, 1.0],
                "score": scores,
            }
        )
        cell = pair.assign(
            cell_type="A",
            incoming=pair["score"] / 2,
            outgoing=pair["score"] / 2,
            total=pair["score"],
            n_cells=2,
        )
        return SimpleNamespace(pair_timecourse=pair, celltype_timecourse=cell)

    with pytest.raises(RuntimeError, match="disagree at generated times"):
        runner._compare_lr_measurement_contracts(
            result([1.0, 2.0, 3.0]),
            result([1.0, 2.1, 3.0]),
            observed_times=(0.0, 1.0),
        )


def test_lr_measurement_diagnostic_labels_one_sided_pairs_as_missing():
    import pandas as pd

    def result(extra_pair, extra_score):
        pair = pd.DataFrame(
            {
                "pair": ["common", "common", "common", extra_pair],
                "time": [0.0, 0.5, 1.0, 0.0],
                "score": [1.0, 2.0, 3.0, extra_score],
            }
        )
        cell = pair.assign(
            cell_type="A",
            incoming=pair["score"] / 2,
            outgoing=pair["score"] / 2,
            total=pair["score"],
            n_cells=2,
        )
        return SimpleNamespace(pair_timecourse=pair, celltype_timecourse=cell)

    pair, *_ = runner._compare_lr_measurement_contracts(
        result("hybrid_only", 4.0),
        result("inverse_only", 5.0),
        observed_times=(0.0, 1.0),
    )
    status = pair.loc[pair["time"].eq(0.0)].set_index("pair")["zero_status"]
    assert status["hybrid_only"] == "all_inverse_missing"
    assert status["inverse_only"] == "hybrid_missing"
    assert status["common"] == "both_nonzero"


def test_velocity_stage_emits_direct_and_latent_projection_contracts(
    tmp_path, monkeypatch
):
    data = ad.AnnData(X=np.zeros((6, 1), dtype=np.float32))
    data.obs["Annotation"] = ["A", "B"] * 3
    data.obs["time_point_processed"] = np.repeat([0.0, 2.0, 4.0], 2)
    data.obsm["spatial_aligned"] = np.arange(12, dtype=np.float32).reshape(6, 2)
    data.obsm["X_latent"] = np.arange(18, dtype=np.float32).reshape(6, 3)
    features = np.hstack((data.obsm["spatial_aligned"], data.obsm["X_latent"]))
    components = {
        "times": data.obs["time_point_processed"].to_numpy(dtype=float),
        "features": features,
        **{
            name: np.ones_like(features, dtype=np.float32)
            for name in ("drift", "interaction", "score", "full")
        },
    }
    components["full"] = (
        components["drift"] + components["interaction"] + components["score"]
    )
    monkeypatch.setattr(
        runner.cb.tl,
        "compute_velocity_components_from_adata",
        lambda *args, **kwargs: components,
    )
    calls = []

    def fake_plot(**kwargs):
        calls.append(kwargs)
        Path(kwargs["out_path"]).write_text("panel", encoding="utf-8")

    monkeypatch.setattr(runner.cb.pl, "plot_velocity_component", fake_plot)
    args = SimpleNamespace(
        interaction_m=32,
        velocity_neighbors=2,
        device="cpu",
        time_key="time_point_processed",
        latent_key="X_latent",
        spatial_key="spatial_aligned",
        annotation_key="Annotation",
        force=True,
    )
    loaded = SimpleNamespace(
        config={"model": {"interaction_net": {"cutoff": 0.1}}},
        model=object(),
    )
    context = runner.RunContext(
        args=args,
        adata=data,
        df=None,
        loaded=loaded,
        runtime=None,
        dim=5,
        spatial_dim=2,
        output_dir=tmp_path,
        shared_cache_dir=tmp_path / "cache",
        label_to_color={"A": "#111111", "B": "#eeeeee"},
        common_signature={"test": True},
    )
    runner._stage_velocity(context)

    assert len(calls) == 3 * 4 * 2
    direct = [call for call in calls if "spatial_direct" in call["out_path"]]
    latent = [call for call in calls if "latent_to_spatial" in call["out_path"]]
    assert all(call["feature_matrix"] is None for call in direct)
    assert all(call["velocity"].shape[1] == 2 for call in direct)
    assert all(call["feature_matrix"].shape[1] == 3 for call in latent)
    assert all(call["velocity"].shape[1] == 3 for call in latent)


def test_s25_missing_target_is_strict_outside_smoke(tmp_path, monkeypatch):
    data = ad.AnnData(X=np.zeros((10, 2), dtype=np.float32))
    data.obs["Annotation"] = ["A"] * 10
    data.obs["time_point_processed"] = np.repeat(runner.OBSERVED_TIMES, 2)
    data.obsm["spatial_aligned"] = np.zeros((10, 2), dtype=np.float32)
    data.obsm["X_latent"] = np.zeros((10, 3), dtype=np.float32)
    generated = ad.AnnData(X=np.zeros((3, 5), dtype=np.float32))
    generated.obs["Annotation"] = ["A", "A", "A"]
    states = {str(float(t)): generated.copy() for t in runner.HALF_TIMES}
    interpolation = SimpleNamespace(
        adata_dict=states,
        classifier_cache_path="cache",
        classifier_accuracy=1.0,
        classifier_balanced_accuracy=1.0,
        simulation_seeds={"split_population": 43},
    )
    monkeypatch.setattr(runner, "_run_interpolation", lambda *a, **k: interpolation)
    monkeypatch.setattr(runner, "_write_state_bundle", lambda *a, **k: [])
    monkeypatch.setattr(
        runner.cb.tl,
        "load_cached_mlp_classifier",
        lambda *a, **k: SimpleNamespace(
            model=object(), label_encoder=object(), include_time_feature=True
        ),
    )
    monkeypatch.setattr(
        runner.cb.tl,
        "predict_labels_for_points",
        lambda **kwargs: np.asarray(["A"] * len(kwargs["points"])),
    )
    args = SimpleNamespace(
        profile="full",
        s25_top_genes=250,
        s25_n_clusters=4,
        sde_dt=0.05,
        sde_sigma=0.03,
        growth_alpha=1.0,
        ysl_label="Yolk Syncytial Layer",
        annotation_key="Annotation",
        preferred_species_tag="zebrafish",
        classifier_epochs=500,
        random_seed=42,
        device="cpu",
        time_key="time_point_processed",
        latent_key="X_latent",
        spatial_key="spatial_aligned",
        interaction_m=1024,
        smoke_n_samples=64,
        sde_n_samples=None,
        sde_max_particles=100000,
        s25_classifier_knn_neighbors=1,
        force=True,
    )
    context = runner.RunContext(
        args=args,
        adata=data,
        df=None,
        loaded=None,
        runtime=None,
        dim=2,
        spatial_dim=2,
        output_dir=tmp_path,
        shared_cache_dir=tmp_path / "cache",
        label_to_color={"A": "#111111"},
        common_signature={"test": True},
    )
    with pytest.raises(ValueError, match="No predicted 'Yolk Syncytial Layer'"):
        runner._stage_s25(context)


def _global_t0_states_for_test():
    observed_t0 = np.asarray([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=np.float32)
    states = {}
    for time_value in runner.HALF_TIMES:
        points = observed_t0.copy()
        if float(time_value) > 0.0:
            points = points + np.float32(time_value)
        state = runner._minimal_state_adata(
            points, ["A", "B"], annotation_key="Annotation"
        )
        state.uns["slice_origin"] = "generated_global_t0"
        state.uns["source_anchor_time"] = 0.0
        states[str(float(time_value))] = state
    return states, observed_t0


def test_global_t0_state_guard_rejects_observed_integer_substitution():
    states, observed_t0 = _global_t0_states_for_test()
    runner._require_global_t0_generated_states(
        states, runner.HALF_TIMES, observed_t0_points=observed_t0
    )

    states["2.0"].uns["slice_origin"] = "observed_real"
    states["2.0"].uns["source_anchor_time"] = 2.0
    with pytest.raises(RuntimeError, match="observed-substituted or re-anchored"):
        runner._require_global_t0_generated_states(
            states, runner.HALF_TIMES, observed_t0_points=observed_t0
        )


def test_global_t0_state_guard_rejects_nonzero_anchor_and_foreign_t0():
    states, observed_t0 = _global_t0_states_for_test()
    states["3.5"].uns["source_anchor_time"] = 3.0
    with pytest.raises(RuntimeError, match="single t=0 anchor"):
        runner._require_global_t0_generated_states(
            states, runner.HALF_TIMES, observed_t0_points=observed_t0
        )

    states, observed_t0 = _global_t0_states_for_test()
    expanded = np.vstack((states["1.5"].X, states["1.5"].X[:1]))
    states["1.5"] = runner._minimal_state_adata(
        expanded, ["A", "B", "A"], annotation_key="Annotation"
    )
    states["1.5"].uns["slice_origin"] = "generated_global_t0"
    states["1.5"].uns["source_anchor_time"] = 0.0
    with pytest.raises(RuntimeError, match="fixed-population.*changed particle count"):
        runner._require_global_t0_generated_states(
            states, runner.HALF_TIMES, observed_t0_points=observed_t0
        )

    states, observed_t0 = _global_t0_states_for_test()
    states["0.0"].X[0, 0] = 99.0
    with pytest.raises(RuntimeError, match="exact sample of the real observed t=0"):
        runner._require_global_t0_generated_states(
            states, runner.HALF_TIMES, observed_t0_points=observed_t0
        )


def test_s22_stage_manifest_contract_is_global_t0_and_reference_only(
    tmp_path, monkeypatch
):
    args = runner._build_parser().parse_args(
        [
            "--aligned-h5ad",
            str(tmp_path / "aligned.h5ad"),
            "--model-dir",
            str(tmp_path / "model"),
            "--output-dir",
            str(tmp_path),
            "--profile",
            "smoke",
        ]
    )
    context = SimpleNamespace(
        args=args,
        output_dir=tmp_path,
        shared_cache_dir=tmp_path / "cache",
    )
    captured = {}

    def fake_execute(_ctx, stage, settings, action):
        captured.update(settings)
        return {"stage": stage}

    monkeypatch.setattr(runner, "_execute_stage", fake_execute)
    assert runner._stage_s22(context) == {"stage": "s22"}
    assert captured["trajectory_mode"] == runner.S22_TRAJECTORY_MODE
    assert captured["split_sde_piecewise"] is False
    assert captured["use_real_for_observed_trajectory_frames"] is False
    assert captured["trajectory_frames"] == (
        "generated_at_every_time_including_integer_times"
    )
    assert captured["observed_integer_frames"] == "separate_reference_only"
    assert captured["simulation_grid"] == list(runner.HALF_TIMES)
    assert captured["simulation_grid"][0] == 0.0
    assert captured["simulation_grid"][-1] == 4.0
    assert captured["sigma"] == pytest.approx(0.03)
    assert captured["downstream_state_contract"]["implicit_s22_reuse"] is False
    assert captured["population_mode"] == "fixed_population_state_transport"
    assert captured["growth_alpha"] == 0.0
    assert captured["trained_growth_head"]["applied_to_s22_transport"] is False
    assert "not an abundance forecast" in captured["scientific_claim"]


def test_s22_stage_keeps_generated_integer_frames_and_observed_references_separate(
    tmp_path, monkeypatch
):
    data = ad.AnnData(X=np.zeros((10, 1), dtype=np.float32))
    data.obs["Annotation"] = ["A", "B"] * 5
    data.obs["time_point_processed"] = np.repeat(runner.OBSERVED_TIMES, 2)
    data.obsm["spatial_aligned"] = np.column_stack(
        (
            data.obs["time_point_processed"].to_numpy(dtype=np.float32),
            np.tile([0, 1], 5),
        )
    ).astype(np.float32)
    data.obsm["X_latent"] = np.zeros((10, 1), dtype=np.float32)
    observed_t0 = runner._joint_features(
        data, latent_key="X_latent", spatial_key="spatial_aligned"
    )[:2]
    states = {}
    for time_value in runner.HALF_TIMES:
        points = observed_t0 + np.float32(time_value)
        state = runner._minimal_state_adata(
            points, ["A", "B"], annotation_key="Annotation"
        )
        state.uns["slice_origin"] = "generated_global_t0"
        state.uns["source_anchor_time"] = 0.0
        states[str(float(time_value))] = state
    interpolation = SimpleNamespace(
        adata_dict=states,
        ts_points=list(runner.HALF_TIMES),
        classifier_cache_path="classifier.pt",
        classifier_accuracy=1.0,
        classifier_balanced_accuracy=1.0,
        simulation_seeds={"split_population": 43},
    )

    def fake_interpolation(*args, **kwargs):
        assert kwargs["trajectory_mode"] == runner.S22_TRAJECTORY_MODE
        assert kwargs["split_growth_alpha"] == runner.S22_FIXED_GROWTH_ALPHA
        return interpolation

    monkeypatch.setattr(runner, "_run_interpolation", fake_interpolation)
    monkeypatch.setattr(
        runner.cb.tl, "save_timepoint_snapshots", lambda *args, **kwargs: None
    )

    import matplotlib.pyplot as plt

    def fake_grid(*args, **kwargs):
        Path(kwargs["out_path"]).write_bytes(b"%PDF-test")
        return plt.figure()

    def fake_animation(*args, **kwargs):
        Path(kwargs["out_path"]).write_bytes(b"GIF-test")

    monkeypatch.setattr(runner.cb.pl, "plot_trajectory_grid", fake_grid)
    monkeypatch.setattr(runner.cb.pl, "plot_trajectory_gif", fake_animation)
    args = runner._build_parser().parse_args(
        [
            "--aligned-h5ad",
            str(tmp_path / "aligned.h5ad"),
            "--model-dir",
            str(tmp_path / "model"),
            "--output-dir",
            str(tmp_path),
            "--profile",
            "smoke",
            "--video-formats",
            "gif",
        ]
    )
    context = runner.RunContext(
        args=args,
        adata=data,
        df=None,
        loaded=None,
        runtime=None,
        dim=3,
        spatial_dim=2,
        output_dir=tmp_path,
        shared_cache_dir=tmp_path / "cache",
        label_to_color={"A": "#111111", "B": "#eeeeee"},
        common_signature={"test": True},
    )
    manifest = runner._stage_s22(context)

    import pandas as pd

    sources = pd.read_csv(tmp_path / "s22" / "frame_sources.csv")
    assert set(sources["population_mode"]) == {"fixed_population_state_transport"}
    assert set(sources["growth_alpha"]) == {0.0}
    after_t0 = sources.loc[sources["time"].gt(0.0)]
    assert (
        after_t0["trajectory_display_source"]
        .eq("generated_global_t0_fixed_population_state_transport")
        .all()
    )
    integer_after_t0 = sources.loc[sources["time"].isin([1.0, 2.0, 3.0, 4.0])]
    assert integer_after_t0["observed_reference_available"].all()
    assert set(integer_after_t0["observed_reference_source"]) == {
        "observed_reference_only"
    }
    assert (
        tmp_path / "s22" / "global_t0_fixed_population_states" / "index.json"
    ).is_file()
    assert (tmp_path / "s22" / "observed_reference_states" / "index.json").is_file()
    assert (tmp_path / "s22" / "S22_trajectory_support_audit.csv").is_file()
    assert (tmp_path / "s22" / "S22_trajectory_support_audit.json").is_file()
    assert (tmp_path / "s22" / "S22_panel_caption.json").is_file()
    assert (
        tmp_path / "s22" / "S22_global_t0_fixed_population_state_transport_mosaic.pdf"
    ).is_file()
    assert manifest["details"]["trajectory_support_audit"]["n_frames"] == len(
        runner.HALF_TIMES
    )
    assert manifest["details"]["trajectory_support_audit_publication_blocking"] is False
    assert manifest["details"]["fixed_particle_count"] == 2
    assert manifest["details"]["particle_count_constant_across_all_frames"] is True
    assert manifest["details"]["growth_head_applied_to_transport"] is False
    assert "not a cell-abundance forecast" in manifest["details"]["panel_caption"]
    assert (
        manifest["details"]["observed_integer_frames_substituted_into_trajectory"]
        is False
    )


def test_interpolation_separates_global_t0_and_interval_local_contracts(
    tmp_path, monkeypatch
):
    captured = {}

    def fake_workflow(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(runner.cb.tl, "run_interpolation_workflow", fake_workflow)
    args = SimpleNamespace(
        profile="smoke",
        classifier_epochs=500,
        random_seed=42,
        annotation_key="Annotation",
        time_key="time_point_processed",
        latent_key="X_latent",
        spatial_key="spatial_aligned",
        device="cpu",
        smoke_n_samples=8,
        sde_dt=0.05,
        sde_sigma=0.03,
        growth_alpha=1.0,
        interaction_m=1024,
        sde_max_particles=100000,
    )
    context = runner.RunContext(
        args=args,
        adata=object(),
        df=object(),
        loaded=None,
        runtime=object(),
        dim=52,
        spatial_dim=2,
        output_dir=tmp_path,
        shared_cache_dir=tmp_path / "cache",
        label_to_color={},
        common_signature={"test": True},
    )
    runner._run_interpolation(
        context,
        output_dir=tmp_path / "workflow",
        time_points=(0.0, 0.5, 1.0),
        trajectory_mode=runner.S22_TRAJECTORY_MODE,
        split_growth_alpha=runner.S22_FIXED_GROWTH_ALPHA,
        display_piecewise_warp=False,
    )
    assert captured["classifier_cache_tag"] == runner.MAIN_CLASSIFIER_CACHE_TAG
    assert captured["use_real_for_observed"] is False
    assert captured["split_sde_piecewise"] is False
    assert captured["split_sde_piecewise_include_end"] is False
    assert captured["split_daughter_noise_std"] == 0.0
    assert captured["split_growth_alpha"] == 0.0
    assert captured["split_sigma_scalar"] == pytest.approx(0.03)
    assert captured["piecewise_observed_sample_mode"] == "t0_fixed"
    assert captured["spatial_warp_to_observed"] is False
    assert captured["spatial_warp_to_observed_piecewise"] is False
    assert captured["spatial_warp_visualization_only"] is False
    assert "initialized once" in runner.S22_TRAJECTORY_SCOPE
    assert "integer times after t=0" in runner.S22_TRAJECTORY_SCOPE
    assert "references only" in runner.S22_TRAJECTORY_SCOPE
    assert "not an abundance forecast" in runner.S22_TRAJECTORY_SCOPE

    captured.clear()
    runner._run_interpolation(
        context,
        output_dir=tmp_path / "workflow_interval_local",
        time_points=(0.0, 0.5, 1.0),
        trajectory_mode="interval_local_observed_anchored",
        split_growth_alpha=1.0,
        display_piecewise_warp=False,
    )
    assert captured["use_real_for_observed"] is True
    assert captured["split_sde_piecewise"] is True
    assert captured["piecewise_observed_sample_mode"] == "per_timepoint"
    assert captured["split_growth_alpha"] == 1.0
    assert "one-sided" in runner.S25_COMMUNICATION_TRAJECTORY_SCOPE
    assert "not global-t0" in runner.S25_COMMUNICATION_TRAJECTORY_SCOPE


def test_interpolation_rejects_legacy_endpoint_directed_display_warp(
    tmp_path, monkeypatch
):
    called = False

    def fake_workflow(**kwargs):
        nonlocal called
        called = True
        return object()

    monkeypatch.setattr(runner.cb.tl, "run_interpolation_workflow", fake_workflow)
    args = SimpleNamespace(
        profile="smoke",
        classifier_epochs=500,
        random_seed=42,
        annotation_key="Annotation",
        time_key="time_point_processed",
        latent_key="X_latent",
        spatial_key="spatial_aligned",
        device="cpu",
        smoke_n_samples=8,
        sde_dt=0.05,
        sde_sigma=0.03,
        growth_alpha=1.0,
        interaction_m=1024,
        sde_max_particles=100000,
    )
    context = runner.RunContext(
        args=args,
        adata=object(),
        df=object(),
        loaded=None,
        runtime=object(),
        dim=52,
        spatial_dim=2,
        output_dir=tmp_path,
        shared_cache_dir=tmp_path / "cache",
        label_to_color={},
        common_signature={"test": True},
    )
    with pytest.raises(ValueError, match="does not permit the legacy"):
        runner._run_interpolation(
            context,
            output_dir=tmp_path / "workflow",
            time_points=(0.0, 0.5, 1.0),
            trajectory_mode=runner.S22_TRAJECTORY_MODE,
            split_growth_alpha=runner.S22_FIXED_GROWTH_ALPHA,
            display_piecewise_warp=True,
        )
    assert not called


def test_trajectory_support_audit_is_pure_and_publication_guard_fails_ood():
    observed_latent = np.asarray([[1.0, 0.0], [2.0, 0.0], [0.0, 1.5]], dtype=np.float32)
    good_frame = np.asarray(
        [[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 1.5, 0.0]],
        dtype=np.float32,
    )
    good_audit, good_summary = runner._compute_trajectory_support_audit(
        observed_latent,
        {"matched": (good_frame, good_frame.copy())},
        (0.0, 1.0),
        spatial_dim=2,
    )
    assert good_summary["status"] == "PASS"
    assert good_summary["observed_reference"]["latent_norm_max"] == 2.0
    assert good_audit["passes_publication_support_gate"].all()
    # Inputs are untouched: the reusable helper has no file or array mutation.
    np.testing.assert_array_equal(
        good_frame,
        np.asarray(
            [[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 1.5, 0.0]],
            dtype=np.float32,
        ),
    )

    outlier_frame = np.repeat(good_frame[:1], 100, axis=0)
    outlier_frame[:2, 2:] = np.asarray([5.0, 0.0], dtype=np.float32)
    bad_audit, bad_summary = runner._compute_trajectory_support_audit(
        observed_latent,
        {"EVL_remove_EVL": (outlier_frame,)},
        (4.0,),
        spatial_dim=2,
    )
    assert bad_summary["status"] == "FAIL"
    row = bad_audit.iloc[0]
    assert row["fraction_outside_observed_max"] == pytest.approx(0.02)
    assert row["latent_norm_max"] == pytest.approx(5.0)
    assert "outside_observed_max_fraction" in row["failure_reasons"]
    assert "maximum_norm_multiplier" in row["failure_reasons"]
    with pytest.raises(RuntimeError, match="publication latent-support gate"):
        runner._require_trajectory_support_audit_pass(bad_summary, stage="S24")

    nonfinite_frame = good_frame.copy()
    nonfinite_frame[0, 2] = np.nan
    _, nonfinite_summary = runner._compute_trajectory_support_audit(
        observed_latent,
        {"YSL_remove_YSL": (nonfinite_frame,)},
        (1.0,),
        spatial_dim=2,
    )
    assert nonfinite_summary["status"] == "FAIL"
    assert (
        "nonfinite_generated_values"
        in nonfinite_summary["failed_frames"][0]["failure_reasons"]
    )


@pytest.mark.parametrize("support_gate_fails", [False, True])
def test_s24_stage_runs_preterminal_t3_sigma0_spatial_protocol(
    tmp_path, monkeypatch, support_gate_fails
):
    labels = [
        "Yolk Syncytial Layer",
        "EVL",
        "A",
        "A",
        "A",
        "A",
        "A",
        "A",
        "A",
        "A",
    ]
    times = [0.0] * 6 + [1.0, 2.0, 3.0, 4.0]
    data = ad.AnnData(X=np.zeros((10, 1), dtype=np.float32))
    data.obs["Annotation"] = labels
    data.obs["time_point_processed"] = times
    data.obsm["spatial_aligned"] = np.zeros((10, 2), dtype=np.float32)
    data.obsm["X_latent"] = np.asarray(
        [[10.0, 0.0]] + [[1.0, 0.0]] * 9, dtype=np.float32
    )
    args = runner._build_parser().parse_args(
        [
            "--aligned-h5ad",
            str(tmp_path / "aligned.h5ad"),
            "--model-dir",
            str(tmp_path / "model"),
            "--output-dir",
            str(tmp_path),
            "--profile",
            "smoke",
            "--sde-dt",
            "0.123",
            "--sde-sigma",
            "0.17",
            "--force",
        ]
    )
    context = runner.RunContext(
        args=args,
        adata=data,
        df=data.obs.copy(),
        loaded=None,
        runtime=object(),
        dim=4,
        spatial_dim=2,
        output_dir=tmp_path,
        shared_cache_dir=tmp_path / "cache",
        label_to_color={
            "Yolk Syncytial Layer": "#111111",
            "EVL": "#222222",
            "A": "#333333",
        },
        common_signature={"test": True},
    )

    def fake_classifier(_ctx, stage_dir):
        cache_path = stage_dir / "classifier.pt"
        pca_path = stage_dir / "classifier_pca.npz"
        cache_path.write_bytes(b"classifier")
        pca_path.write_bytes(b"pca")
        cached = SimpleNamespace(accuracy=0.9, balanced_accuracy=0.8)
        return cached, cache_path, pca_path, lambda *args, **kwargs: None

    calls = []

    def fake_ablation(_adata, _runtime, **kwargs):
        calls.append(kwargs)
        variant = next(iter(kwargs["ablations"]))
        n_points = 4
        frames = [
            np.column_stack(
                (
                    np.zeros((n_points, 2), dtype=np.float32),
                    np.full((n_points, 1), 1.0 + 0.01 * index, dtype=np.float32),
                    np.zeros((n_points, 1), dtype=np.float32),
                )
            ).astype(np.float32)
            for index in range(len(kwargs["time_points"]))
        ]
        if support_gate_fails and variant == "remove_EVL":
            frames[-1] = frames[-1].copy()
            frames[-1][0, 2] = 30.0
        frames = tuple(frames)
        labels_by_time = tuple(
            np.asarray(["A"] * n_points) for _ in kwargs["time_points"]
        )
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        artifact = output_dir / "mock_result.npz"
        artifact.write_bytes(b"result")
        return SimpleNamespace(
            initial_obs_names=tuple(f"cell_{index}" for index in range(n_points)),
            time_points=tuple(kwargs["time_points"]),
            baseline_points=frames,
            ablation_points={variant: tuple(frame.copy() for frame in frames)},
            baseline_labels=labels_by_time,
            ablation_labels={variant: labels_by_time},
            metrics=runner.pd.DataFrame.from_records(
                {
                    "variant": variant,
                    "time": float(time_value),
                    "space": space,
                    "w2": 0.0,
                }
                for time_value in kwargs["time_points"]
                for space in ("joint", "spatial")
            ),
            files=(artifact,),
            settings={
                "mass_control": kwargs["mass_control"],
                "dt": kwargs["dt"],
                "resample_dt": kwargs["resample_dt"],
                "sigma": kwargs["sigma"],
                "sigma_by_dim": None,
                "growth_alpha": kwargs["growth_alpha"],
                "interaction_m": kwargs["interaction_m"],
                "interaction_seed": kwargs["interaction_seed"],
                "variant_initial_counts": {variant: n_points},
                "simulation_seeds": {
                    "baseline": kwargs["random_seed"],
                    variant: kwargs["random_seed"],
                },
            },
        )

    import matplotlib as mpl
    import matplotlib.pyplot as plt

    rc_keys = (
        "font.family",
        "font.size",
        "axes.titlesize",
        "axes.labelsize",
        "xtick.labelsize",
        "ytick.labelsize",
        "legend.fontsize",
        "pdf.fonttype",
        "ps.fonttype",
    )

    def rc_snapshot():
        return {
            key: tuple(value) if isinstance(value, list) else value
            for key in rc_keys
            for value in (mpl.rcParams[key],)
        }

    rc_before = rc_snapshot()

    plot_calls = []
    plot_rc_snapshots = []

    def fake_grid(**kwargs):
        plot_calls.append(kwargs)
        plot_rc_snapshots.append(rc_snapshot())
        Path(kwargs["out_path"]).write_bytes(b"figure")
        return plt.figure()

    monkeypatch.setattr(runner, "_ablation_classifier", fake_classifier)
    monkeypatch.setattr(runner.cb.tl, "run_virtual_cell_type_ablation", fake_ablation)
    monkeypatch.setattr(runner.cb.pl, "plot_trajectory_comparison_grid", fake_grid)

    if support_gate_fails:
        with pytest.raises(RuntimeError, match="publication latent-support gate"):
            runner._stage_ablation(context)
        assert len(calls) == 2
        assert plot_calls == []
        assert rc_snapshot() == rc_before
        assert (
            tmp_path
            / "ablation"
            / "S24_preterminal_t3_sigma0_trajectory_support_audit.csv"
        ).is_file()
        assert not list((tmp_path / "ablation").glob("S24_*_grid.*"))
        return

    manifest = runner._stage_ablation(context)
    assert len(calls) == 2
    assert [call["ablations"] for call in calls] == [
        {"remove_YSL": ["Yolk Syncytial Layer"]},
        {"remove_EVL": ["EVL"]},
    ]
    for call in calls:
        assert call["time_points"] == [0.0, 1.0, 2.0, 3.0]
        assert 4.0 not in call["time_points"]
        assert call["mass_control"] is True
        assert call["growth_alpha"] == 0.0
        assert call["sigma"] == 0.0
        assert call["common_random_seed"] is True
        assert call["random_seed"] == 42
        assert call["interaction_seed"] == 10043
        assert call["interaction_m"] == 1024
        assert call["resample_dt"] == call["dt"] == 0.005
        assert call["save_snapshots"] is False

    settings = manifest["settings"]
    assert settings["publication_protocol"] == "preterminal_t3_sigma0"
    assert settings["time_points"] == [0.0, 1.0, 2.0, 3.0]
    assert settings["publication_snapshot_times"] == [0.0, 1.0, 2.0, 3.0]
    assert settings["dt"] == settings["split_resample_dt"] == 0.005
    assert settings["sigma"] == 0.0
    assert settings["s24_fixed_numerics"] == {
        "source": "hard-coded publication protocol; CLI SDE values do not apply",
        "cli_sde_dt": 0.123,
        "cli_sde_sigma": 0.17,
    }
    assert settings["mass_control"] is True
    assert settings["growth_alpha"] == 0.0
    assert settings["interaction_seed"] == 10043
    assert settings["superseded_legacy_result"]["status"] == (
        "superseded_diagnostic_only"
    )
    assert settings["superseded_legacy_result"]["reused"] is False
    assert settings["terminal_t4_scope"]["included"] is False
    assert settings["terminal_t4_scope"]["evaluated"] is False
    assert settings["terminal_t4_scope"]["claimed"] is False
    assert "t=4 is not evaluated or claimed" in settings["terminal_t4_scope"]["reason"]
    assert "binding_to_external_failed_run" not in settings["terminal_t4_scope"]
    assert settings["trajectory_support_audit"][
        "maximum_fraction_outside_observed_max"
    ] == pytest.approx(0.01)
    assert settings["trajectory_support_audit"][
        "maximum_generated_norm_multiplier"
    ] == pytest.approx(2.0)
    assert manifest["details"]["matched_initial_particle_counts"] == {
        "YSL": 4,
        "EVL": 4,
    }
    assert manifest["details"]["trajectory_support_audit"]["status"] == "PASS"
    assert manifest["details"]["trajectory_support_audit"]["n_frames"] == 16
    assert manifest["details"]["publication_protocol"] == "preterminal_t3_sigma0"
    assert manifest["details"]["end_time"] == 3.0
    assert manifest["details"]["sigma"] == 0.0

    expected = [
        tmp_path / "ablation" / f"S24_{target}_preterminal_t3_sigma0_grid.{ext}"
        for target in ("YSL", "EVL")
        for ext in ("pdf", "png")
    ]
    assert all(path.is_file() for path in expected)
    assert len(plot_calls) == 4
    assert all(
        call["selected_times"] == runner.S24_PUBLICATION_TIMES for call in plot_calls
    )
    assert all("through t=3" in call["title"] for call in plot_calls)
    expected_plot_rc = {
        "font.family": ("Arial",),
        "font.size": 9.0,
        "axes.titlesize": 9.0,
        "axes.labelsize": 9.0,
        "xtick.labelsize": 9.0,
        "ytick.labelsize": 9.0,
        "legend.fontsize": 9.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
    assert plot_rc_snapshots == [expected_plot_rc] * 4
    assert rc_snapshot() == rc_before
    assert not (tmp_path / "ablation" / "S24_virtual_ablation_grid.pdf").exists()
    captions = json.loads(
        (
            tmp_path / "ablation" / "S24_preterminal_t3_sigma0_panel_captions.json"
        ).read_text(encoding="utf-8")
    )
    assert all(
        "preterminal_t3_sigma0" in value
        and "not a stochastic forecast" in value
        and "causal knockout" in value
        and "full joint-state terminal result" in value
        and "Terminal t=4 is not evaluated or claimed" in value
        for value in captions.values()
    )
    for target in ("YSL", "EVL"):
        metrics = runner.pd.read_csv(
            tmp_path / "ablation" / f"S24_{target}_preterminal_t3_sigma0_metrics.csv"
        )
        assert set(metrics["space"]) == {"spatial"}
        assert metrics["time"].max() == 3.0
        assert 4.0 not in set(metrics["time"])


@pytest.mark.parametrize(
    ("protocol_drift", "error_match"),
    [
        ("sigma", "did not use sigma=0"),
        ("dt", "did not use dt=resample_dt"),
        ("resample_dt", "did not use dt=resample_dt"),
        ("end_time", "wrong output-time grid"),
        ("fixed_n", "changed particle count"),
    ],
)
def test_s24_result_validator_rejects_protocol_drift(protocol_drift, error_match):
    variant = "remove_EVL"
    n_points = 4
    frame = np.zeros((n_points, 4), dtype=np.float32)
    frames = tuple(frame.copy() for _ in runner.S24_PUBLICATION_TIMES)
    result = SimpleNamespace(
        initial_obs_names=tuple(f"cell_{index}" for index in range(n_points)),
        time_points=runner.S24_PUBLICATION_TIMES,
        baseline_points=frames,
        ablation_points={variant: tuple(value.copy() for value in frames)},
        settings={
            "mass_control": True,
            "dt": runner.S24_FIXED_DT,
            "resample_dt": runner.S24_FIXED_DT,
            "sigma": runner.S24_FIXED_SIGMA,
            "sigma_by_dim": None,
            "growth_alpha": runner.S24_FIXED_GROWTH_ALPHA,
            "interaction_m": runner.S24_INTERACTION_M,
            "interaction_seed": 10043,
            "variant_initial_counts": {variant: n_points},
            "simulation_seeds": {"baseline": 42, variant: 42},
        },
    )
    if protocol_drift == "sigma":
        result.settings["sigma"] = 0.03
    elif protocol_drift == "dt":
        result.settings["dt"] = 0.01
    elif protocol_drift == "resample_dt":
        result.settings["resample_dt"] = 0.01
    elif protocol_drift == "end_time":
        result.time_points = (0.0, 1.0, 2.0, 4.0)
    elif protocol_drift == "fixed_n":
        result.baseline_points = (*frames[:-1], frame[:-1].copy())

    with pytest.raises(RuntimeError, match=error_match):
        runner._require_s24_preterminal_t3_sigma0_result(
            result,
            variant=variant,
            time_points=runner.S24_PUBLICATION_TIMES,
            random_seed=42,
            interaction_seed=10043,
            interaction_m=runner.S24_INTERACTION_M,
        )
