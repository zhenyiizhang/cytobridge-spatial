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


def test_canonical_s22_manifest_semantics_reject_legacy_global_bundle(tmp_path):
    manifest_path = tmp_path / "stage_manifest.json"
    canonical = {
        "settings": {
            "trajectory_mode": (
                "piecewise_observed_anchored_interval_forward_simulation"
            ),
            "split_sde_piecewise": True,
            "piecewise_observed_sample_mode": "per_timepoint",
            "piecewise_include_end": False,
            "daughter_noise_std": 0.0,
            "display_warp": {"applied": False},
            "simulation": runner.CANONICAL_TRAJECTORY_SCOPE,
        },
        "details": {
            "trajectory_scope": runner.CANONICAL_TRAJECTORY_SCOPE,
            "display_warp_applied": False,
        },
    }
    runner._require_canonical_s22_manifest_semantics(canonical, manifest_path)

    legacy = json.loads(json.dumps(canonical))
    legacy["settings"]["trajectory_mode"] = "global_t0_continuous"
    legacy["settings"]["split_sde_piecewise"] = False
    legacy["settings"]["display_warp"]["applied"] = True
    with pytest.raises(RuntimeError, match="refusing legacy/global-t0"):
        runner._require_canonical_s22_manifest_semantics(legacy, manifest_path)


def test_s25_refuses_stale_existing_s22_trajectory(tmp_path):
    common = {"model_config_sha256": "old"}
    _write_current_stage_manifest(tmp_path, stage="s22", common=common)
    canonical = tmp_path / "s22" / "canonical_prewarp_states"
    canonical.mkdir()
    (canonical / "index.json").write_text("{}\n", encoding="utf-8")
    context = SimpleNamespace(
        output_dir=tmp_path,
        common_signature={"model_config_sha256": "new"},
        args=SimpleNamespace(profile="full", s25_top_genes=250),
    )
    with pytest.raises(RuntimeError, match="different data/model/code"):
        runner._stage_s25(context)


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
            "trajectory_scope": runner.CANONICAL_TRAJECTORY_SCOPE,
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
        "simulation": runner.CANONICAL_TRAJECTORY_SCOPE,
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


def test_interpolation_uses_the_classifier_stage_cache_tag(tmp_path, monkeypatch):
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
        use_real_for_observed=True,
        display_piecewise_warp=False,
    )
    assert captured["classifier_cache_tag"] == runner.MAIN_CLASSIFIER_CACHE_TAG
    assert captured["split_sde_piecewise"] is True
    assert captured["split_sde_piecewise_include_end"] is False
    assert captured["split_daughter_noise_std"] == 0.0
    assert captured["piecewise_observed_sample_mode"] == "per_timepoint"
    assert captured["spatial_warp_to_observed"] is False
    assert captured["spatial_warp_to_observed_piecewise"] is False
    assert captured["spatial_warp_visualization_only"] is False
    assert "one-sided" in runner.CANONICAL_TRAJECTORY_SCOPE
    assert "not conditioned on the following observed endpoint" in (
        runner.CANONICAL_TRAJECTORY_SCOPE
    )
    assert "not global-t0" in runner.CANONICAL_TRAJECTORY_SCOPE
    assert "not lineage-continuous" in runner.CANONICAL_TRAJECTORY_SCOPE


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
            use_real_for_observed=True,
            display_piecewise_warp=True,
        )
    assert not called
