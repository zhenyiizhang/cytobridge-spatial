from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "run_zebrafish_interval_daughter_noise_sensitivity.py"
)
SPEC = importlib.util.spec_from_file_location(
    "zebrafish_interval_daughter_noise_sensitivity", SCRIPT
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _frame(
    *,
    noise: float,
    points: np.ndarray,
    lineage_ids: np.ndarray,
    labels: np.ndarray,
    anchor: float = 0.0,
    seed: int = 42,
    n_source: int = 3,
) -> MODULE.ForecastFrame:
    return MODULE.ForecastFrame(
        interval_start=anchor,
        interval_end=anchor + 1.0,
        forecast_time=anchor + 0.5,
        forecast_role="midpoint_one_sided_forecast",
        daughter_noise_std=noise,
        seed=seed,
        points=np.asarray(points, dtype=np.float32),
        lineage_ids=np.asarray(lineage_ids, dtype=np.int64),
        labels=np.asarray(labels).astype(str),
        n_source=n_source,
    )


def test_scientific_grid_is_frozen_and_not_cli_overridable():
    assert MODULE.NOISE_VALUES == (0.0, 0.01, 0.03, 0.06)
    assert MODULE.PAIRED_SEEDS == (42, 43, 44, 45, 46)
    assert MODULE.INTERVALS == ((0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 4.0))
    assert MODULE.DT == MODULE.RESAMPLE_DT == 0.05
    assert MODULE.CONTINUOUS_DIFFUSION_SIGMA == 0.03
    assert MODULE.GROWTH_ALPHA == 1.0
    assert MODULE.INTERACTION_M == 1024
    assert MODULE.INTERACTION_SEED_OFFSET == 10_000
    assert MODULE.MAX_PARTICLES == 100_000
    assert MODULE.ZEBRAFISH_LATENT_DIM == 50
    assert MODULE.ZEBRAFISH_JOINT_DIM == 52

    destinations = {action.dest for action in MODULE._build_parser()._actions}
    for forbidden in (
        "noise_values",
        "seeds",
        "dt",
        "diffusion_sigma",
        "growth_alpha",
        "interaction_m",
        "resample_dt",
        "max_particles",
        "classifier_knn_neighbors",
        "save_states",
    ):
        assert forbidden not in destinations

    root = SCRIPT.parents[1]
    manifest = (root / "MANIFEST.in").read_text(encoding="utf-8")
    assert (
        "include scripts/run_zebrafish_interval_daughter_noise_sensitivity.py"
        in manifest
    )
    script_readme = (root / "scripts/README.md").read_text(encoding="utf-8")
    assert "interval-local `(anchor_time, source_obs_id)`" in script_readme


def test_acceptance_report_is_hash_bound_and_requires_nested_zebrafish_pass(
    tmp_path: Path,
):
    path = tmp_path / "acceptance.json"
    path.write_text(
        json.dumps(
            {
                "status": "PASS",
                "datasets": {"zebrafish": {"status": "PASS"}},
            }
        ),
        encoding="utf-8",
    )
    resolved, digest, payload = MODULE._load_acceptance_report(path, _sha256(path))
    assert resolved == path.resolve()
    assert digest == _sha256(path)
    assert payload["datasets"]["zebrafish"]["status"] == "PASS"

    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        MODULE._load_acceptance_report(path, "0" * 64)

    path.write_text(
        json.dumps(
            {
                "status": "PASS",
                "datasets": {"zebrafish": {"status": "FAIL"}},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="zebrafish status PASS"):
        MODULE._load_acceptance_report(path, _sha256(path))


def test_classifier_contract_recomputes_exact_aligned_source_fingerprint():
    obs = pd.DataFrame(
        {
            "time_point_processed": [0.0, 1.0, 2.0, 3.0, 4.0],
            "Annotation": ["A", "B", "A", "B", "A"],
        },
        index=[f"cell-{index}" for index in range(5)],
    )
    adata = ad.AnnData(X=np.ones((5, 2), dtype=np.float32), obs=obs)
    adata.obsm["spatial_aligned"] = np.arange(10, dtype=np.float32).reshape(5, 2)
    adata.obsm["X_latent"] = np.arange(250, dtype=np.float32).reshape(5, 50)
    joint = MODULE._joint_features(
        adata, spatial_key="spatial_aligned", latent_key="X_latent"
    )
    times = np.arange(5, dtype=np.float64)
    inputs = np.hstack((times.astype(np.float32)[:, None], joint))
    fingerprint = MODULE._classifier_source_fingerprint(
        adata,
        label_col="Annotation",
        time_key="time_point_processed",
        classifier_inputs=inputs,
    )
    cached = SimpleNamespace(
        label_col="Annotation",
        include_time_feature=True,
        feature_dim=52,
        feature_cols=("samples", *(f"x{index}" for index in range(1, 53))),
        metadata={
            "version": 8,
            "best_epoch_metric": "bacc",
            "train_on_full_data": False,
            "refit_on_full_data_after_selection": False,
            "stratify_split": True,
            "strict_stratification": True,
            "selection_scope": "held_out_validation_phase_a",
            "seed": 42,
            "class_split": {
                "strategy": "held_out_train_validation",
                "per_class_counts": {
                    "A": {"total": 3, "train": 2, "validation": 1},
                    "B": {"total": 2, "train": 1, "validation": 1},
                },
            },
            "feature_selection": {
                "kind": "leading_joint_dimensions",
                "n_features": 52,
                "requested_n_features": None,
            },
            "source": {
                "kind": "AnnData",
                "n_obs": 5,
                "n_vars": 2,
                "obsm_key": "X_latent",
                "spatial_key": "spatial_aligned",
                "concat_spatial": True,
                "fingerprint": fingerprint,
            },
        },
        label_encoder=SimpleNamespace(classes_=np.asarray(["A", "B"])),
        accuracy=0.8,
        balanced_accuracy=0.75,
    )
    contract = MODULE._validate_classifier_contract(
        cached,
        adata,
        joint,
        times,
        time_key="time_point_processed",
        annotation_key="Annotation",
        spatial_key="spatial_aligned",
        latent_key="X_latent",
    )
    assert contract["source_fingerprint"] == fingerprint
    assert contract["feature_dim"] == 52

    cached.metadata["source"]["fingerprint"] = "stale"
    with pytest.raises(RuntimeError, match="source fingerprint"):
        MODULE._validate_classifier_contract(
            cached,
            adata,
            joint,
            times,
            time_key="time_point_processed",
            annotation_key="Annotation",
            spatial_key="spatial_aligned",
            latent_key="X_latent",
        )


def test_model_contract_requires_actual_learned_gate_and_final_score():
    interaction_net = SimpleNamespace(
        edge_prior_mode="learned",
        link_predictor=object(),
        edge_predictor_thre=0.606,
        cutoff=0.096,
    )
    loaded = SimpleNamespace(
        config={
            "model": {
                "components": ["velocity", "growth", "score", "interaction"],
                "interaction_type": "gnn",
                "interaction_group_size": 1024,
                "interaction_net": {
                    "edge_prior_mode": "learned",
                    "edge_predictor_thre": 0.606,
                    "cutoff": 0.096,
                },
            }
        },
        weight_stage="Finetune",
        score_stage="Score_Refine",
        score_path=Path("score_model.pth"),
        model=SimpleNamespace(
            latent_dim=52,
            config={"interaction_net": {"load_edge_predictor_from_path": False}},
            interaction_net=interaction_net,
            interaction_group_size=1024,
        ),
    )
    contract = MODULE._validate_learned_model_contract(loaded, expected_dim=52)
    assert contract["edge_prior_mode"] == "learned"
    assert contract["edge_predictor_source"] == "embedded_in_weight_checkpoint"

    loaded.model.interaction_net.edge_prior_mode = "all_spatial"
    with pytest.raises(RuntimeError, match="learned interaction gate"):
        MODULE._validate_learned_model_contract(loaded, expected_dim=52)

    loaded.model.interaction_net.edge_prior_mode = "learned"
    loaded.model.config["interaction_net"]["load_edge_predictor_from_path"] = True
    with pytest.raises(RuntimeError, match="embedded in the hash-bound"):
        MODULE._validate_learned_model_contract(loaded, expected_dim=52)


def test_lineage_namespace_contains_anchor_and_source_id_and_never_cross_joins():
    frame0 = _frame(
        noise=0.01,
        points=np.zeros((3, 2)),
        lineage_ids=np.asarray([0, 0, 1]),
        labels=np.asarray(["A", "B", "B"]),
        anchor=0.0,
        n_source=3,
    )
    source_ids = np.asarray(["shared/id", "cell 1", "lost"])
    source_labels = np.asarray(["S0", "S1", "S2"])
    descendants, transitions = MODULE._lineage_rows(
        frame0, source_obs_ids=source_ids, source_labels=source_labels
    )
    assert descendants[0]["lineage_namespace"] == (
        "anchor_time=0/source_obs_id=shared%2Fid"
    )
    assert descendants[2]["descendant_count"] == 0
    assert descendants[2]["lineage_alive"] is False
    assert all("anchor_time" in row["lineage_namespace"] for row in transitions)
    assert all("source_obs_id" in row["lineage_namespace"] for row in transitions)

    namespace0 = MODULE._lineage_namespace(0.0, "same-cell")
    namespace1 = MODULE._lineage_namespace(1.0, "same-cell")
    assert namespace0 != namespace1


def test_simulation_is_interval_local_uses_all_rows_and_resets_lineage_roster(
    monkeypatch: pytest.MonkeyPatch,
):
    calls = []

    def fake_simulate(**kwargs):
        calls.append(kwargs)
        x0 = np.asarray(kwargs["x0"], dtype=np.float32)
        times = kwargs["ts_points"]
        points = [x0] + [x0 + index for index in range(1, len(times))]
        roster = np.asarray(kwargs["initial_lineage_ids"], dtype=np.int64)
        lineages = [roster.copy() for _ in times]
        point_frames = np.empty(len(points), dtype=object)
        point_frames[:] = points
        lineage_frames = np.empty(len(lineages), dtype=object)
        lineage_frames[:] = lineages
        return point_frames, lineage_frames

    def fake_predict(*, points, time_value, **_kwargs):
        return np.asarray([f"t{time_value:g}"] * len(points))

    monkeypatch.setattr(
        MODULE.cb.tl, "simulate_sde_points_split_from_x0", fake_simulate
    )
    monkeypatch.setattr(MODULE.cb.tl, "predict_labels_for_points", fake_predict)
    runtime = SimpleNamespace(f_net=object(), score_net=object())
    classifier = SimpleNamespace(model=object(), label_encoder=object(), feature_dim=2)
    x0 = np.asarray([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=np.float32)
    frames = MODULE._simulate_interval(
        x0=x0,
        runtime=runtime,
        classifier=classifier,
        interval_start=1.0,
        interval_end=2.0,
        daughter_noise_std=0.03,
        seed=44,
        include_end=True,
        device="cpu",
    )
    assert [frame.forecast_time for frame in frames] == [1.5, 2.0]
    assert [frame.forecast_role for frame in frames] == [
        "midpoint_one_sided_forecast",
        "endpoint_one_sided_forecast",
    ]
    call = calls[0]
    assert call["ts_points"] == [1.0, 1.5, 2.0]
    np.testing.assert_array_equal(call["x0"], x0)
    np.testing.assert_array_equal(call["initial_lineage_ids"], [0, 1, 2])
    assert call["dt"] == call["resample_dt"] == 0.05
    assert call["sigma"] == 0.03
    assert call["growth_alpha"] == 1.0
    assert call["interaction_m"] == 1024
    assert call["interaction_seed"] == 10_044
    assert call["max_particles"] == 100_000
    assert call["daughter_noise_std"] == 0.03
    assert "warp" not in " ".join(call)

    MODULE._simulate_interval(
        x0=x0[:1],
        runtime=runtime,
        classifier=classifier,
        interval_start=2.0,
        interval_end=3.0,
        daughter_noise_std=0.03,
        seed=44,
        include_end=False,
        device="cpu",
    )
    np.testing.assert_array_equal(calls[1]["initial_lineage_ids"], [0])
    assert calls[1]["ts_points"] == [2.0, 2.5]
    assert calls[1]["interaction_seed"] == 10_044


def test_noise0_pairing_reports_composition_count_and_lineage_deltas():
    base = _frame(
        noise=0.0,
        points=np.zeros((3, 2)),
        lineage_ids=np.asarray([0, 0, 1]),
        labels=np.asarray(["A", "A", "B"]),
    )
    other = _frame(
        noise=0.03,
        points=np.ones((4, 2)),
        lineage_ids=np.asarray([0, 1, 1, 2]),
        labels=np.asarray(["A", "B", "B", "B"]),
    )
    row = MODULE._paired_delta_row(base, other)
    assert row["particle_count_delta"] == 1
    assert row["particle_count_relative_delta"] == pytest.approx(1 / 3)
    assert row["composition_total_variation"] == pytest.approx(5 / 12)
    assert row["mean_absolute_lineage_descendant_count_delta"] == pytest.approx(1.0)
    assert row["lineage_survival_jaccard"] == pytest.approx(2 / 3)
    assert row["lineage_fate_mean_total_variation_from_noise0"] == pytest.approx(1 / 3)
    assert row["lineage_fate_max_total_variation_from_noise0"] == pytest.approx(1.0)
    assert row["joint_w2_from_noise0"] == pytest.approx(np.sqrt(2.0))
    assert row["spatial_w2_from_noise0"] == pytest.approx(np.sqrt(2.0))
    assert row["joint_ot_status"] == row["spatial_ot_status"] == "complete"
    assert row["joint_ot_noise0_points"] == 3
    assert row["joint_ot_noise_points"] == 4
    assert row["paired_common_seed"] is True


def test_raw_npz_has_non_object_arrays_and_anchor_scoped_lineage_keys(tmp_path: Path):
    frame = _frame(
        noise=0.01,
        points=np.ones((2, 3)),
        lineage_ids=np.asarray([0, 1]),
        labels=np.asarray(["A", "B"]),
        anchor=2.0,
        n_source=2,
    )
    path = MODULE._save_raw_interval(
        tmp_path,
        frames=[frame],
        source_obs_ids=np.asarray(["cell/0", "cell 1"]),
        source_labels=np.asarray(["S0", "S1"]),
    )
    with np.load(path, allow_pickle=False) as payload:
        assert payload["midpoint_points"].dtype == np.float32
        assert payload["midpoint_lineage_ids"].dtype == np.int64
        assert payload["interaction_seed"].item() == 10_042
        assert payload["lineage_namespace"].tolist() == [
            "anchor_time=2/source_obs_id=cell%2F0",
            "anchor_time=2/source_obs_id=cell%201",
        ]


def test_output_directory_refuses_to_mix_with_existing_results(tmp_path: Path):
    output = tmp_path / "result"
    output.mkdir()
    (output / "old.txt").write_text("old", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to mix or overwrite"):
        MODULE._prepare_output_dir(output)


def test_run_writes_hash_bound_interval_local_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    obs = pd.DataFrame(
        {
            "time_point_processed": np.arange(5, dtype=np.float64),
            "Annotation": ["A"] * 5,
        },
        index=[f"cell-{index}" for index in range(5)],
    )
    adata = ad.AnnData(X=np.ones((5, 1), dtype=np.float32), obs=obs)
    adata.obsm["spatial_aligned"] = np.arange(10, dtype=np.float32).reshape(5, 2)
    adata.obsm["X_latent"] = np.arange(250, dtype=np.float32).reshape(5, 50)
    adata.uns["preprocess_info"] = {"expression_layer": "counts"}
    adata.uns["interaction_graph"] = {"edge_prior_mode": "learned"}
    aligned = tmp_path / "aligned.h5ad"
    adata.write_h5ad(aligned)
    loaded_adata = ad.read_h5ad(aligned)
    joint = MODULE._joint_features(
        loaded_adata, spatial_key="spatial_aligned", latent_key="X_latent"
    )
    times = np.arange(5, dtype=np.float64)
    classifier_inputs = np.hstack((times.astype(np.float32)[:, None], joint))
    fingerprint = MODULE._classifier_source_fingerprint(
        loaded_adata,
        label_col="Annotation",
        time_key="time_point_processed",
        classifier_inputs=classifier_inputs,
    )

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    config = model_dir / "config.yaml"
    weight = model_dir / "best_model.pth"
    score = model_dir / "score_model.pth"
    config.write_text("model: fake\n", encoding="utf-8")
    weight.write_bytes(b"weight")
    score.write_bytes(b"score")
    classifier_path = tmp_path / "classifier.pt"
    classifier_path.write_bytes(b"classifier")
    acceptance = tmp_path / "acceptance.json"
    acceptance.write_text(
        json.dumps(
            {
                "run_root": "/canonical/final",
                "status": "PASS",
                "datasets": {"zebrafish": {"status": "PASS"}},
            }
        ),
        encoding="utf-8",
    )
    interaction_net = SimpleNamespace(
        edge_prior_mode="learned",
        link_predictor=object(),
        edge_predictor_thre=0.6,
        cutoff=0.1,
    )
    loaded = SimpleNamespace(
        config={
            "model": {
                "components": ["velocity", "growth", "score", "interaction"],
                "interaction_type": "gnn",
                "interaction_group_size": 1024,
                "interaction_net": {
                    "edge_prior_mode": "learned",
                    "edge_predictor_thre": 0.6,
                    "cutoff": 0.1,
                },
            }
        },
        weight_stage="Finetune",
        score_stage="Score_Refine",
        weight_path=weight,
        score_path=score,
        model=SimpleNamespace(
            latent_dim=52,
            config={"interaction_net": {"load_edge_predictor_from_path": False}},
            interaction_net=interaction_net,
            interaction_group_size=1024,
        ),
    )
    cached = SimpleNamespace(
        label_col="Annotation",
        include_time_feature=True,
        feature_dim=52,
        feature_cols=("samples", *(f"x{index}" for index in range(1, 53))),
        metadata={
            "version": 8,
            "best_epoch_metric": "bacc",
            "train_on_full_data": False,
            "refit_on_full_data_after_selection": False,
            "stratify_split": True,
            "strict_stratification": True,
            "selection_scope": "held_out_validation_phase_a",
            "seed": 42,
            "class_split": {
                "strategy": "held_out_train_validation",
                "per_class_counts": {"A": {"total": 5, "train": 4, "validation": 1}},
            },
            "feature_selection": {
                "kind": "leading_joint_dimensions",
                "n_features": 52,
                "requested_n_features": None,
            },
            "source": {
                "kind": "AnnData",
                "n_obs": 5,
                "n_vars": 1,
                "obsm_key": "X_latent",
                "spatial_key": "spatial_aligned",
                "concat_spatial": True,
                "fingerprint": fingerprint,
            },
        },
        label_encoder=SimpleNamespace(classes_=np.asarray(["A"])),
        accuracy=0.8,
        balanced_accuracy=0.8,
        model=object(),
    )
    monkeypatch.setattr(
        MODULE.cb.tl, "load_dynamical_model_from_dir", lambda *_args, **_kwargs: loaded
    )
    monkeypatch.setattr(
        MODULE.cb.tl,
        "build_dynamical_runtime",
        lambda _loaded: SimpleNamespace(f_net=object(), score_net=object()),
    )
    monkeypatch.setattr(
        MODULE.cb.tl, "load_cached_mlp_classifier", lambda *_args, **_kwargs: cached
    )

    def fake_simulate(**kwargs):
        x0 = np.asarray(kwargs["x0"], dtype=np.float32)
        roster = np.asarray(kwargs["initial_lineage_ids"], dtype=np.int64)
        point_frames = np.empty(len(kwargs["ts_points"]), dtype=object)
        point_frames[:] = [x0.copy() for _ in kwargs["ts_points"]]
        lineage_frames = np.empty(len(kwargs["ts_points"]), dtype=object)
        lineage_frames[:] = [roster.copy() for _ in kwargs["ts_points"]]
        return point_frames, lineage_frames

    monkeypatch.setattr(
        MODULE.cb.tl, "simulate_sde_points_split_from_x0", fake_simulate
    )
    monkeypatch.setattr(
        MODULE.cb.tl,
        "predict_labels_for_points",
        lambda *, points, **_kwargs: np.asarray(["A"] * len(points)),
    )
    output = tmp_path / "output"
    args = SimpleNamespace(
        aligned_h5ad=aligned,
        expected_aligned_sha256=_sha256(aligned),
        model_dir=model_dir,
        expected_model_config_sha256=_sha256(config),
        expected_weight_sha256=_sha256(weight),
        expected_score_sha256=_sha256(score),
        classifier_cache=classifier_path,
        expected_classifier_sha256=_sha256(classifier_path),
        acceptance_report=acceptance,
        expected_acceptance_sha256=_sha256(acceptance),
        output_dir=output,
        time_key="time_point_processed",
        annotation_key="Annotation",
        spatial_key="spatial_aligned",
        latent_key="X_latent",
        device="cpu",
        include_end=False,
    )
    manifest_path = MODULE.run(args)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["trajectory_scope"] == MODULE.TRAJECTORY_SCOPE
    assert manifest["simulation"]["daughter_noise_std"] == list(MODULE.NOISE_VALUES)
    assert manifest["simulation"]["paired_seeds"] == list(MODULE.PAIRED_SEEDS)
    assert manifest["simulation"]["spatial_warp"] is False
    assert manifest["simulation"]["interaction_grouping_rng"] == {
        "interaction_seed_by_paired_seed": {
            "42": 10042,
            "43": 10043,
            "44": 10044,
            "45": 10045,
            "46": 10046,
        },
        "interaction_seed_offset": 10000,
        "paired_across_daughter_noise": True,
        "seed_formula": "paired_seed + interaction_seed_offset",
        "stream": "dedicated_torch_generator",
    }
    assert manifest["claim_guardrails"]["following_endpoint_conditioned"] is False
    assert manifest["claim_guardrails"]["global_t0_rollout"] is False
    assert manifest["claim_guardrails"]["lineage_continuous_across_intervals"] is False
    assert manifest["run_counts"] == {
        "forecast_frames": 80,
        "independent_interval_noise_seed_runs": 80,
        "noise0_paired_delta_rows": 60,
        "raw_state_files": 80,
    }
    assert manifest["metric_contract"]["wasserstein"]["metrics"] == ["W1", "W2"]
    assert manifest["metric_contract"]["wasserstein"]["max_points_per_cloud"] == 1024
    assert manifest["model_contract"]["edge_predictor_source"] == (
        "embedded_in_weight_checkpoint"
    )
    assert (
        "CytoBridge/tl/downstream/evaluation.py"
        in manifest["code"]["implementation_sha256"]
    )
    assert (
        "CytoBridge/tl/core/interaction.py" in manifest["code"]["implementation_sha256"]
    )
    covered = {
        field: manifest[field]
        for field in manifest["signature"]["covered_top_level_fields"]
    }
    assert manifest["signature"]["value"] == MODULE._stable_json_sha256(covered)
    paired = pd.read_csv(output / "tables/noise0_paired_deltas.csv")
    assert paired.shape[0] == 60
    assert paired["joint_w2_from_noise0"].eq(0.0).all()
    assert paired["spatial_w2_from_noise0"].eq(0.0).all()
    assert paired["paired_common_seed"].eq(True).all()  # noqa: E712
    assert paired["interaction_seed"].isin(range(10042, 10047)).all()
    for artifact in manifest["outputs"]["tables"].values():
        path = output / artifact["path"]
        assert path.stat().st_size == artifact["size_bytes"]
        assert _sha256(path) == artifact["sha256"]
    assert len(manifest["outputs"]["raw_states"]) == 80
    for artifact in manifest["outputs"]["raw_states"]:
        path = output / artifact["path"]
        assert path.stat().st_size == artifact["size_bytes"]
        assert _sha256(path) == artifact["sha256"]
    recorded_manifest_hash = (output / "run_manifest.sha256").read_text().split()[0]
    assert recorded_manifest_hash == _sha256(manifest_path)
