from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_zebrafish_paper_downstream.py"
SPEC = importlib.util.spec_from_file_location("zebrafish_paper_runner", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def test_time_grid_and_stage_parser_are_strict():
    assert runner._time_grid(0.0, 1.0, 0.25) == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert runner._parse_stages("s25,communication,s25") == ["s25", "communication"]
    assert runner._parse_stages("all") == list(runner.ALL_STAGES)


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
    _, output = _write_current_stage_manifest(
        tmp_path, stage="s22", common=common
    )
    context = SimpleNamespace(output_dir=tmp_path, common_signature=common)
    assert runner._require_current_stage_manifest(context, "s22")["status"] == "complete"

    changed_context = SimpleNamespace(
        output_dir=tmp_path,
        common_signature={**common, "model_config_sha256": "config-b"},
    )
    with pytest.raises(RuntimeError, match="different data/model/code"):
        runner._require_current_stage_manifest(changed_context, "s22")

    output.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="missing or modified outputs"):
        runner._require_current_stage_manifest(context, "s22")


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


def test_velocity_stage_emits_direct_and_latent_projection_contracts(tmp_path, monkeypatch):
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
