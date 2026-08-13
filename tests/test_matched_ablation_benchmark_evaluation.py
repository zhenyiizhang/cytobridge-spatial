from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import sys
import types

import numpy as np
import pandas as pd
from PIL import Image
import pytest
import yaml

from scripts.spatiotemporal_benchmark.cytobridge import common as WRITER_COMMON


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts/run_matched_ablation_benchmark_evaluation.py"
SPEC = importlib.util.spec_from_file_location(
    "run_matched_ablation_benchmark_evaluation", SCRIPT_PATH
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
REAL_METRIC_RECOMPUTE = MODULE._recompute_metric_values
FAST_METRIC_ORACLE: dict[str, pd.DataFrame] = {}


TARGETS = {
    "zebrafish": [1, 2, 3, 4],
    "mosta": [1, 2, 3],
    "arista": [1, 2, 3, 4],
    "admouse": [1, 2],
}


@pytest.fixture(autouse=True)
def _fast_metric_recomputation(monkeypatch: pytest.MonkeyPatch) -> None:
    def oracle(**kwargs: object) -> pd.DataFrame:
        key = str(Path(kwargs["prediction_path"]).resolve())
        if key not in FAST_METRIC_ORACLE:
            return REAL_METRIC_RECOMPUTE(**kwargs)
        return FAST_METRIC_ORACLE[key].copy()

    monkeypatch.setattr(MODULE, "_recompute_metric_values", oracle)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _identity(path: Path) -> dict[str, object]:
    path = path.resolve()
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _artifact(path: Path, input_root: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "relative_path": path.relative_to(input_root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _model_config(arm: str, model_dir: Path) -> dict[str, object]:
    stages = []
    for index, (name, mode) in enumerate(
        (
            ("Pretrain", "neural_ode"),
            ("Refine", "neural_ode"),
            ("Init_interaction", "neural_ode"),
            ("Train_Score", "score_matching"),
            ("Finetune", "neural_ode"),
            ("Score_Refine", "score_matching"),
        )
    ):
        stages.append(
            {
                "name": name,
                "mode": mode,
                "epochs": 1,
                "batch_size": 2,
                "save_strategy": "best",
                "interaction_use": arm != "no_interaction" and index in {2, 4},
            }
        )
    model: dict[str, object] = {
        "components": (
            ["velocity", "growth", "score"]
            if arm == "no_interaction"
            else ["velocity", "growth", "score", "interaction"]
        )
    }
    if arm != "no_interaction":
        interaction = {
            "edge_prior_mode": MODULE.ARM_INTERACTION_MODE[arm],
            "cutoff": 0.1,
        }
        if arm == "full":
            interaction.update(
                {
                    "edge_predictor_path": "/immutable/predictor.pt",
                    "edge_predictor_thre": 0.6,
                }
            )
        model.update(
            {
                "interaction_type": "gnn",
                "interaction_group_size": MODULE.INTERACTION_M,
                "interaction_net": interaction,
            }
        )
    return {
        "seed": MODULE.MATCHED_SEED,
        "ckpt_dir": str(model_dir),
        "matched_ablation": {
            "protocol": MODULE.MATCHED_PROTOCOL,
            "condition": arm,
        },
        "model": model,
        "training": {"plan": stages},
    }


def _write_benchmark_input(
    root: Path, dataset: str, aligned_sha: str
) -> tuple[Path, str]:
    input_root = root / dataset / "inputs"
    full = input_root / "full_data"
    full.mkdir(parents=True)
    train_h5ad = full / "train.h5ad"
    training_reference = full / "training_reference.npz"
    source_roster = full / "source_roster.npz"
    train_h5ad.write_bytes(b"fake benchmark h5ad" + dataset.encode())
    times = np.concatenate(
        [
            np.zeros(7, dtype=np.int16),
            *[
                np.full(5 + target, target, dtype=np.int16)
                for target in TARGETS[dataset]
            ],
        ]
    )
    rows = np.arange(len(times), dtype=np.float64)
    state = np.column_stack(
        (rows * 0.17 + times * 0.2, np.sin(rows * 0.3) + times * 0.1)
    ).astype(np.float32)
    spatial = np.column_stack(
        (rows * 0.05 + times * 0.3, np.cos(rows * 0.2) - times * 0.15)
    ).astype(np.float32)
    row_id = np.asarray([f"{dataset}_row_{index}" for index in range(len(times))])
    np.savez_compressed(
        training_reference,
        state=state,
        spatial=spatial,
        time=times,
        row_id=row_id,
        annotation=np.asarray(["cell"] * len(times)),
    )
    source_indices = np.flatnonzero(times == 0)
    indices = np.resize(source_indices, MODULE.PREDICTION_N).astype(np.int64)
    np.savez_compressed(
        source_roster,
        indices=indices,
        row_id=row_id[indices],
        source_time=np.asarray([0], dtype=np.int16),
        state=state[indices],
        spatial=spatial[indices],
        support_row_id=row_id[source_indices],
        support_indices=source_indices.astype(np.int64),
    )
    truth: dict[str, dict[str, object]] = {}
    for target in TARGETS[dataset]:
        path = full / f"truth_t{target}.npz"
        mask = times == target
        np.savez_compressed(path, state=state[mask], spatial=spatial[mask])
        truth[str(target)] = _artifact(path, input_root)
    split_manifest = full / "manifest.json"
    _write_json(
        split_manifest,
        {
            "contract_version": MODULE.INPUT_CONTRACT,
            "dataset_id": dataset,
            "split": "full_data",
            "protocol": "full_data",
        },
    )
    split_sidecar = split_manifest.with_suffix(".json.sha256")
    split_sidecar.write_text(f"{_sha256(split_manifest)}  manifest.json\n")
    split_record = {
        "contract_version": MODULE.INPUT_CONTRACT,
        "dataset_id": dataset,
        "split": "full_data",
        "protocol": "full_data",
        "held_out_benchmark_time": None,
        "evaluation_targets": TARGETS[dataset],
        "prediction_n": MODULE.PREDICTION_N,
        "source_time": 0,
        "transductive_frozen_representation": True,
        "representation_refit_per_fold": False,
        "train": {
            "h5ad": _artifact(train_h5ad, input_root),
            "training_reference_npz": _artifact(training_reference, input_root),
            "source_roster_npz": _artifact(source_roster, input_root),
        },
        "truth_by_time_npz": truth,
        "manifest": _artifact(split_manifest, input_root),
        "manifest_sha256_sidecar": _artifact(split_sidecar, input_root),
    }
    manifest = input_root / "manifest.json"
    _write_json(
        manifest,
        {
            "contract_version": MODULE.INPUT_CONTRACT,
            "status": "complete",
            "dataset_id": dataset,
            "prediction_n": MODULE.PREDICTION_N,
            "full_data_targets": TARGETS[dataset],
            "source": {"h5ad_sha256": aligned_sha},
            "splits": {"full_data": split_record},
        },
    )
    sidecar = manifest.with_suffix(".json.sha256")
    sidecar.write_text(f"{_sha256(manifest)}  manifest.json\n")
    return manifest, _sha256(manifest)


def _write_fixture(tmp_path: Path) -> dict[str, object]:
    release = tmp_path / "release"
    launcher_root = tmp_path / "matched-training"
    benchmark_root = tmp_path / "benchmark"
    release.mkdir()
    launcher_root.mkdir()
    for relative in (*MODULE.ADAPTER_FILES, *MODULE.EVALUATOR_FILES):
        path = release / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"implementation:{relative}\n")
    for dataset in MODULE.DATASET_ORDER:
        path = release / "configs/unified_benchmark" / f"{dataset}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            yaml.safe_dump(
                {
                    "dataset_id": dataset,
                    "full_data_targets": TARGETS[dataset],
                    "prediction_n": MODULE.PREDICTION_N,
                },
                sort_keys=False,
            )
        )
    aligned: dict[str, dict[str, object]] = {}
    conditions: dict[str, dict[str, object]] = {}
    for dataset in MODULE.DATASET_ORDER:
        source = tmp_path / "accepted-inputs" / f"{dataset}.h5ad"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(f"aligned-{dataset}".encode())
        aligned[dataset] = _identity(source)
        for arm in MODULE.ARM_ORDER:
            profile = MODULE.profile_name(dataset, arm)
            profile_root = launcher_root / profile
            input_link = profile_root / "preprocess" / f"{dataset}_aligned.h5ad"
            input_link.parent.mkdir(parents=True)
            input_link.symlink_to(source)
            model_dir = profile_root / "training"
            model_dir.mkdir()
            config = _model_config(arm, model_dir)
            saved_config = model_dir / "config.yaml"
            saved_config.write_text(yaml.safe_dump(config, sort_keys=False))
            _write_json(model_dir / "training_run_summary.json", {"schema_version": 1})
            for stage in config["training"]["plan"]:
                filename = (
                    "score_model.pth"
                    if stage["mode"] == "score_matching"
                    else "best_model.pth"
                )
                checkpoint = model_dir / stage["name"] / filename
                checkpoint.parent.mkdir()
                checkpoint.write_bytes(f"{profile}-{stage['name']}".encode())
            source_config = release / "CytoBridge/configs" / f"{profile}.yaml"
            source_config.parent.mkdir(parents=True, exist_ok=True)
            source_config.write_text(yaml.safe_dump(config, sort_keys=False))
            conditions[profile] = {
                "dataset": dataset,
                "arm": arm,
                "protocol": MODULE.MATCHED_PROTOCOL,
                "shared_seed": MODULE.MATCHED_SEED,
                "gpu": len(conditions),
                "training_config": _identity(source_config),
                "paths": {
                    "condition_root": str(profile_root),
                    "aligned_h5ad": str(input_link),
                    "training": str(model_dir),
                },
            }
    for relative in (
        "scripts/run_matched_ablation_matrix.py",
        "scripts/validate_corrected_de_novo_run.py",
    ):
        path = release / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"fixture:{relative}\n")
    subprocess.run(["git", "init", "-q", str(release)], check=True)
    subprocess.run(["git", "-C", str(release), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(release),
            "-c",
            "user.name=CytoBridge Test",
            "-c",
            "user.email=cytobridge-test@example.invalid",
            "commit",
            "-q",
            "-m",
            "fixture",
        ],
        check=True,
    )
    commit = subprocess.run(
        ["git", "-C", str(release), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    python_identity = MODULE._python_executable_identity(
        Path(sys.executable).absolute()
    )
    launcher = {
        "schema_version": MODULE.LAUNCHER_SCHEMA_VERSION,
        "kind": MODULE.LAUNCHER_KIND,
        "run_root": str(launcher_root),
        "release": {
            "root": str(release),
            "commit": commit,
            "git": MODULE._git_release_identity(release.resolve(), commit),
            "training_code": MODULE._release_tree_identity(
                release.resolve(), python_only=True
            ),
            "package_payload": MODULE._release_tree_identity(
                release.resolve(), python_only=False
            ),
            "launcher": _identity(release / "scripts/run_matched_ablation_matrix.py"),
            "python_executable": python_identity,
        },
        "matrix": {
            "datasets": list(MODULE.DATASET_ORDER),
            "arms": list(MODULE.ARM_ORDER),
            "profiles": list(MODULE.PROFILE_ORDER),
            "fit_count": 12,
            "protocol": MODULE.MATCHED_PROTOCOL,
            "shared_seed": MODULE.MATCHED_SEED,
        },
        "sources": {
            dataset: {"aligned_h5ad": identity} for dataset, identity in aligned.items()
        },
        "conditions": conditions,
    }
    launcher_path = (
        launcher_root / "_matched_launcher/matched_ablation_matrix_manifest.json"
    )
    _write_json(launcher_path, launcher)
    launcher_sha = _sha256(launcher_path)
    launcher_path.with_suffix(".json.sha256").write_text(launcher_sha + "\n")
    acceptance = {
        "run_root": str(launcher_root),
        "status": "PASS",
        "datasets": {profile: {"status": "PASS"} for profile in MODULE.PROFILE_ORDER},
        "matched_families": {
            dataset: {"status": "PASS"} for dataset in MODULE.DATASET_ORDER
        },
    }
    acceptance_path = launcher_root / "matched_ablation_acceptance.json"
    _write_json(acceptance_path, acceptance)
    benchmark_inputs: dict[str, Path] = {}
    benchmark_hashes: dict[str, str] = {}
    for dataset in MODULE.DATASET_ORDER:
        path, digest = _write_benchmark_input(
            benchmark_root, dataset, str(aligned[dataset]["sha256"])
        )
        benchmark_inputs[dataset] = path
        benchmark_hashes[dataset] = digest
    return {
        "release": release,
        "launcher_root": launcher_root,
        "launcher_path": launcher_path,
        "launcher_sha": launcher_sha,
        "acceptance_path": acceptance_path,
        "acceptance_sha": _sha256(acceptance_path),
        "benchmark_inputs": benchmark_inputs,
        "benchmark_hashes": benchmark_hashes,
    }


def _prepare(tmp_path: Path) -> tuple[dict[str, object], Path, dict[str, object], str]:
    fixture = _write_fixture(tmp_path)
    root = tmp_path / "matched-evaluation"
    plan = MODULE.build_plan(
        evaluation_root=root,
        launcher_manifest=fixture["launcher_path"],
        expected_launcher_sha256=fixture["launcher_sha"],
        acceptance_report=fixture["acceptance_path"],
        expected_acceptance_sha256=fixture["acceptance_sha"],
        benchmark_inputs=fixture["benchmark_inputs"],
        expected_benchmark_sha256=fixture["benchmark_hashes"],
    )
    _, digest = MODULE.prepare_run_root(plan)
    return fixture, root, plan, digest


def _write_prediction_outputs(plan: dict[str, object], profile: str) -> None:
    record = plan["profiles"][profile]
    dataset = record["dataset"]
    arm = record["arm"]
    root = Path(record["paths"]["predictions"])
    targets = plan["benchmark_inputs"][dataset]["targets"]
    manifest_sha = plan["benchmark_inputs"][dataset]["manifest"]["sha256"]
    roster_sha = plan["benchmark_inputs"][dataset]["artifacts"][
        "train_source_roster_npz"
    ]["sha256"]
    input_record = plan["benchmark_inputs"][dataset]
    canonical_roster = Path(
        input_record["artifacts"]["train_source_roster_npz"]["path"]
    )
    output_roster = root / "source_roster.npz"
    root.mkdir(parents=True)
    output_roster.write_bytes(canonical_roster.read_bytes())
    training_path = Path(
        input_record["artifacts"]["train_training_reference_npz"]["path"]
    )
    with np.load(training_path, allow_pickle=False) as archive:
        training_state = np.asarray(archive["state"], dtype=np.float32)
        training_spatial = np.asarray(archive["spatial"], dtype=np.float32)
        training_time = np.asarray(archive["time"], dtype=np.float64)
        training_row_id = np.asarray(archive["row_id"]).astype(str)
    with np.load(output_roster, allow_pickle=False) as archive:
        roster_indices = np.asarray(archive["indices"], dtype=np.int64)
        roster_state = np.asarray(archive["state"], dtype=np.float32)
        roster_spatial = np.asarray(archive["spatial"], dtype=np.float32)
    roster_summary = {
        "source_roster": str(output_roster),
        "source_roster_sha256": _sha256(output_roster),
        "source_indices_sha256": MODULE._sha256_array(roster_indices),
        "source_row_id_sha256": MODULE._sha256_array(
            training_row_id[roster_indices].astype("U")
        ),
        "source_time": 0,
        "source_available_n": int(np.count_nonzero(training_time == 0)),
        "prediction_n": MODULE.PREDICTION_N,
        "sampled_with_replacement": True,
        "canonical_input_roster": str(canonical_roster),
        "canonical_input_roster_sha256": roster_sha,
        "shared_across_all_benchmark_families": True,
    }
    inventory = record["inventory"]
    checkpoint_sha = {
        stage: identity["sha256"]
        for stage, identity in inventory["checkpoints"].items()
    }
    adapter = plan["implementation"]["adapter"]
    adapter_summary = {
        "schema_version": adapter["schema_version"],
        "files": {
            name: identity["sha256"] for name, identity in adapter["files"].items()
        },
        "aggregate_sha256": adapter["aggregate_sha256"],
    }
    expected_simulation = inventory["inference_contract"]
    include_interaction = arm != "no_interaction"
    simulation = {
        "official_api": MODULE.OFFICIAL_SIMULATION_API,
        "official_api_signature": "(adata, model, dim, time_index, n_samples, ts_points, dt, sigma, include_score, interaction_m, device, time_key, obsm_key, spatial_key, concat_spatial, interaction_seed)",
        "simulation_mode": MODULE.OFFICIAL_SIMULATION_MODE,
        "weight_stage": expected_simulation["weight_stage"],
        "score_stage": expected_simulation["score_stage"],
        "interaction_mode": MODULE.ARM_INTERACTION_MODE[arm],
        "edge_prior_mode": MODULE.ARM_INTERACTION_MODE[arm],
        "include_interaction": include_interaction,
        "edge_predictor_used": arm == "full",
        "interaction_m": expected_simulation["interaction_m"],
        "loaded_model_interaction_group_size": expected_simulation["interaction_m"],
        "interaction_group_binding": (
            "exact_checkpoint_model_match"
            if include_interaction
            else "not_applicable_no_interaction_component"
        ),
        "interaction_grouping_seed": expected_simulation["interaction_grouping_seed"],
        "stochastic_stream_contract": MODULE.STOCHASTIC_STREAM_CONTRACT,
        "dynamics_components": expected_simulation["components"],
        "weights_semantics": MODULE.WEIGHTS_SEMANTICS,
        "torch_version": "fixture",
    }
    provenance = {
        "input_manifest": input_record["manifest"]["path"],
        "input_manifest_sha256": manifest_sha,
        "train_h5ad": input_record["artifacts"]["train_h5ad"]["path"],
        "train_h5ad_sha256": input_record["artifacts"]["train_h5ad"]["sha256"],
        "training_reference_npz": input_record["artifacts"][
            "train_training_reference_npz"
        ]["path"],
        "training_reference_sha256": input_record["artifacts"][
            "train_training_reference_npz"
        ]["sha256"],
        "source_roster_npz": canonical_roster.as_posix(),
        "source_roster_sha256": roster_sha,
        "truth_inputs_opened": False,
    }
    compact = []
    for target in targets:
        target_dir = root / f"t{target}"
        target_dir.mkdir(parents=True)
        prediction = target_dir / "prediction.npz"
        target_count = int(np.count_nonzero(training_time == target))
        source_count = int(np.count_nonzero(training_time == 0))
        base_tmv = 0.05 + target * 0.002
        multiplier = (1.0, 1.1, 1.2)[MODULE.ARM_ORDER.index(arm)]
        predicted_mass = (target_count / source_count) * (1.0 + base_tmv * multiplier)
        weights = np.full(
            MODULE.PREDICTION_N,
            predicted_mass / MODULE.PREDICTION_N,
            dtype=np.float64,
        )
        np.savez_compressed(
            prediction,
            spatial=roster_spatial + np.float32(target * 0.01),
            state=roster_state + np.float32(target * 0.02),
            weights=weights,
            source_time=np.asarray([0.0], dtype=np.float64),
            target_time=np.asarray([target], dtype=np.float64),
        )
        summary = {
            "status": "complete",
            "dataset": dataset,
            "method": MODULE.METHOD,
            "implementation": MODULE.EXPECTED_IMPLEMENTATION,
            "regime": "full_data",
            "track": "full_data",
            "split_id": "full_data",
            "target": target,
            "target_time": target,
            "source_time": 0,
            "source_policy": "fixed t0 bootstrap shared across all full-data targets; no intermediate reset",
            "output_scope": "native_joint",
            "primary_benchmark_eligible": True,
            "native_vs_adapter": "native_joint",
            "native_joint": True,
            "native_mass": True,
            "native_growth": True,
            "spatial_warp_applied": False,
            "spatial_warp": "none",
            "split_sde": False,
            "continuous_across_targets": True,
            "prediction_n": MODULE.PREDICTION_N,
            "prediction_n_policy": "fixed_from_train_contract_before_truth_access",
            "alpha_express": MODULE.ALPHA_EXPRESS,
            "alpha_spatial": MODULE.ALPHA_SPATIAL,
            "sigma": MODULE.INFERENCE_SIGMA,
            "dt": MODULE.INFERENCE_DT,
            "include_score": True,
            "seed": MODULE.MATCHED_SEED,
            "device": "cuda:0",
            "include_interaction": include_interaction,
            "interaction_mode": MODULE.ARM_INTERACTION_MODE[arm],
            "edge_prior_mode": MODULE.ARM_INTERACTION_MODE[arm],
            "edge_predictor_used": arm == "full",
            "interaction_m": expected_simulation["interaction_m"],
            "predicted_mass": predicted_mass,
            "weights_are_unnormalised": True,
            "prediction_npz": str(prediction),
            "prediction_npz_sha256": _sha256(prediction),
            "state_dim": training_state.shape[1],
            "spatial_dim": training_spatial.shape[1],
            "joint_dim": training_state.shape[1] + training_spatial.shape[1],
            "config_sha256": inventory["resolved_config"]["sha256"],
            "checkpoint_sha256": checkpoint_sha,
            "stage_complete": True,
            "stage_count": 6,
            "training_reference_match": {
                "proof": "saved_adata_exact_frozen_arrays",
                "row_identity_proof": "contracted_row_id_exact_order",
                "row_identity_key": "obs/row_id",
                "array_sha256": {
                    "state": MODULE._sha256_array(
                        np.asarray(training_state, dtype=np.float32)
                    ),
                    "spatial": MODULE._sha256_array(
                        np.asarray(training_spatial, dtype=np.float32)
                    ),
                    "time": MODULE._sha256_array(
                        np.asarray(training_time, dtype=np.float64)
                    ),
                    "row_identity": MODULE._sha256_array(training_row_id.astype("U")),
                },
            },
            "source_roster": roster_summary,
            "simulation": simulation,
            "repo": {
                "repo": plan["launcher"]["release_root"],
                "git_commit": plan["launcher"]["release_commit"],
                "git_dirty": False,
            },
            "adapter_implementation": adapter_summary,
            **provenance,
        }
        payload = json.dumps(summary, indent=2, sort_keys=True) + "\n"
        (target_dir / "summary.json").write_text(payload)
        (target_dir / "prediction.summary.json").write_text(payload)
        compact.append(
            {
                "target": target,
                "prediction_npz": str(prediction),
                "prediction_npz_sha256": _sha256(prediction),
                "predicted_mass": predicted_mass,
            }
        )
    _write_json(
        root / "run_summary.json",
        {
            "status": "complete",
            "method": MODULE.METHOD,
            "regime": "full_data",
            "split_id": "full_data",
            "source_time": 0,
            "targets": targets,
            "single_continuous_non_split_call": True,
            "intermediate_reset": False,
            "spatial_warp_applied": False,
            "prediction_n": MODULE.PREDICTION_N,
            "seed": MODULE.MATCHED_SEED,
            "source_roster": roster_summary,
            "prediction_summaries": compact,
            **provenance,
        },
    )


def _write_score_outputs(plan: dict[str, object], profile: str) -> None:
    record = plan["profiles"][profile]
    dataset = record["dataset"]
    arm = record["arm"]
    arm_index = MODULE.ARM_ORDER.index(arm)
    dataset_index = MODULE.DATASET_ORDER.index(dataset)
    root = Path(record["paths"]["scores"])
    transform = root / "transforms/full_data.json"
    transform.parent.mkdir(parents=True)
    input_record = plan["benchmark_inputs"][dataset]
    training_path = Path(
        input_record["artifacts"]["train_training_reference_npz"]["path"]
    )
    training = MODULE._training_reference_arrays(training_path)
    transform.write_bytes(MODULE._expected_transform_bytes(training))
    source_count = int(np.count_nonzero(np.isclose(training["time"], 0.0)))
    rows = []
    for target in input_record["targets"]:
        prediction = Path(record["paths"]["predictions"]) / f"t{target}/prediction.npz"
        summary = prediction.parent / "summary.json"
        with np.load(prediction, allow_pickle=False) as archive:
            predicted_mass = float(
                np.asarray(archive["weights"], dtype=np.float64).sum()
            )
        with np.load(
            input_record["artifacts"][f"truth_t{target}"]["path"],
            allow_pickle=False,
        ) as archive:
            observed_n = int(np.asarray(archive["state"]).shape[0])
        observed_mass = observed_n / source_count
        tmv_absolute = abs(predicted_mass - observed_mass)
        tmv = tmv_absolute / observed_mass
        for space_index, space in enumerate(MODULE.SPACE_ORDER):
            exact_base = 0.8 + dataset_index * 0.1 + target * 0.01 + space_index * 0.02
            for repeat in range(MODULE.PROJECTION_REPEATS):
                sliced_base = (
                    1.0
                    + dataset_index * 0.1
                    + target * 0.01
                    + space_index * 0.02
                    + repeat * 0.001
                )
                multiplier = (1.0, 1.1, 1.2)[arm_index]
                projection_seed = MODULE._expected_projection_seed(
                    dataset, space, repeat
                )
                dimension = {
                    "state": training["state"].shape[1],
                    "spatial": training["spatial"].shape[1],
                    "joint": training["state"].shape[1] + training["spatial"].shape[1],
                }[space]
                rows.append(
                    {
                        "track": "full_data",
                        "target": target,
                        "source_time": 0,
                        "benchmark": dataset,
                        "split": "full_data",
                        "method": MODULE.METHOD,
                        "space": space,
                        "projection_repeat": repeat,
                        "projection_seed": projection_seed,
                        "projection_sha256": MODULE._expected_projection_sha256(
                            int(dimension), projection_seed
                        ),
                        "n_projections": MODULE.N_PROJECTIONS,
                        "primary_metric": "sliced_w2",
                        "primary_value": sliced_base * multiplier,
                        "sliced_w2": sliced_base * multiplier,
                        "exact_w1": exact_base * multiplier,
                        "exact_w2": exact_base * 1.2 * multiplier,
                        "n_predicted": MODULE.PREDICTION_N,
                        "n_observed": observed_n,
                        "predicted_weight_sum": predicted_mass,
                        "exact_ot_predicted_points": MODULE.MAX_OT_POINTS,
                        "exact_ot_observed_points": min(
                            observed_n, MODULE.MAX_OT_POINTS
                        ),
                        "output_scope": "native_joint",
                        "native_vs_adapter": "native_joint",
                        "prediction_path": str(prediction),
                        "prediction_sha256": _sha256(prediction),
                        "prediction_summary": str(summary),
                        "prediction_summary_sha256": _sha256(summary),
                        "input_manifest": input_record["manifest"]["path"],
                        "input_manifest_sha256": input_record["manifest"]["sha256"],
                        "training_reference": input_record["artifacts"][
                            "train_training_reference_npz"
                        ]["path"],
                        "training_reference_sha256": input_record["artifacts"][
                            "train_training_reference_npz"
                        ]["sha256"],
                        "source_roster": input_record["artifacts"][
                            "train_source_roster_npz"
                        ]["path"],
                        "source_roster_sha256": input_record["artifacts"][
                            "train_source_roster_npz"
                        ]["sha256"],
                        "truth_reference": input_record["artifacts"][
                            f"truth_t{target}"
                        ]["path"],
                        "truth_reference_sha256": input_record["artifacts"][
                            f"truth_t{target}"
                        ]["sha256"],
                        "transform_path": str(transform),
                        "transform_sha256": _sha256(transform),
                        "tmv_available": True,
                        "tmv": tmv,
                        "tmv_absolute": tmv_absolute,
                        "predicted_mass": predicted_mass,
                        "observed_mass_relative": observed_mass,
                    }
                )
    metrics = pd.DataFrame(rows)
    for target in input_record["targets"]:
        prediction = Path(record["paths"]["predictions"]) / f"t{target}/prediction.npz"
        FAST_METRIC_ORACLE[str(prediction.resolve())] = metrics.loc[
            metrics["target"].eq(target),
            [
                "space",
                "projection_repeat",
                "sliced_w2",
                "exact_w1",
                "exact_w2",
            ],
        ].reset_index(drop=True)
    metrics_path = root / "full_data_metrics_long.csv"
    metrics.to_csv(metrics_path, index=False)
    status = pd.DataFrame(
        [
            {
                "track": "full_data",
                "target": target,
                "method": MODULE.METHOD,
                "status": "completed",
                "reason": "",
            }
            for target in input_record["targets"]
        ]
    )
    status_path = root / "full_data_method_target_status.csv"
    status.to_csv(status_path, index=False)
    _write_json(
        root / "full_data_evaluation_manifest.json",
        {
            "schema_version": "1.0.0",
            "status": "complete",
            "track": "full_data",
            "input_manifest": input_record["manifest"]["path"],
            "input_manifest_sha256": input_record["manifest"]["sha256"],
            "predictions_root": record["paths"]["predictions"],
            "n_projections": MODULE.N_PROJECTIONS,
            "projection_repeats": MODULE.PROJECTION_REPEATS,
            "max_ot_points": MODULE.MAX_OT_POINTS,
            "methods": [MODULE.METHOD],
            "completed_methods": [MODULE.METHOD],
            "targets": input_record["targets"],
            "spaces": sorted(MODULE.SPACE_ORDER),
            "method_target_status": status.to_dict(orient="records"),
            "method_target_status_csv": str(status_path),
            "status_table_source": None,
            "metrics_long_csv": str(metrics_path),
            "metrics_long_csv_sha256": _sha256(metrics_path),
            "n_rows": len(metrics),
        },
    )


def _write_all_outputs(plan: dict[str, object]) -> None:
    for profile in MODULE.PROFILE_ORDER:
        _write_prediction_outputs(plan, profile)
        _write_score_outputs(plan, profile)


def _rewrite_score_table(
    plan: dict[str, object], profile: str, frame: pd.DataFrame
) -> None:
    score_root = Path(plan["profiles"][profile]["paths"]["scores"])
    metrics_path = score_root / "full_data_metrics_long.csv"
    frame.to_csv(metrics_path, index=False)
    manifest_path = score_root / "full_data_evaluation_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["metrics_long_csv_sha256"] = _sha256(metrics_path)
    _write_json(manifest_path, manifest)


def _rewrite_prediction_summary(
    plan: dict[str, object], profile: str, target: int, payload: dict[str, object]
) -> None:
    prediction_root = Path(plan["profiles"][profile]["paths"]["predictions"])
    target_dir = prediction_root / f"t{target}"
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    (target_dir / "summary.json").write_text(encoded)
    (target_dir / "prediction.summary.json").write_text(encoded)


def test_prepare_binds_matrix_and_renders_only_official_full_data_commands(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fixture, root, plan, digest = _prepare(tmp_path)

    assert len(digest) == 64
    assert plan["matrix"]["track"] == "full_data"
    assert len(plan["profiles"]) == 12
    assert plan["matrix"]["target_prediction_count"] == 39
    assert {
        dataset: plan["benchmark_inputs"][dataset]["targets"]
        for dataset in MODULE.DATASET_ORDER
    } == TARGETS
    for profile in MODULE.PROFILE_ORDER:
        record = plan["profiles"][profile]
        assert len(record["inventory"]["checkpoints"]) == 6
        assert set(record["input_binding"]) == {
            "aligned_h5ad_sha256",
            "benchmark_manifest_sha256",
            "full_data_train_h5ad_sha256",
            "source_roster_sha256",
        }
        infer = record["commands"]["infer_full"]["argv"]
        score = record["commands"]["score_full_data"]["argv"]
        assert "scripts.spatiotemporal_benchmark.cytobridge.run_cytobridge" in infer
        assert "infer-full" in infer
        assert "--split" in infer and "full_data" in infer
        assert "scripts.spatiotemporal_benchmark.evaluate_predictions" in score
        assert score[score.index("--max-ot-points") + 1] == "800"
        assert "downstream" not in " ".join(infer + score).lower()
        for command in record["commands"].values():
            assert command["cwd"] == str(fixture["release"])
            assert command["shell"].startswith(
                f"cd {shlex.quote(str(fixture['release']))} && env "
            )
    assert MODULE.main(["render", "--run-root", str(root)]) == 0
    rendered = capsys.readouterr().out
    assert rendered.count("] infer-full") == 12
    assert rendered.count("] score-full-data") == 12
    assert "downstream" not in rendered.lower()


def test_prepared_plan_fails_closed_when_a_checkpoint_changes(tmp_path: Path) -> None:
    _, root, plan, _ = _prepare(tmp_path)
    checkpoint = Path(
        next(iter(plan["profiles"]["zebrafish"]["inventory"]["checkpoints"].values()))[
            "path"
        ]
    )
    checkpoint.write_bytes(b"changed checkpoint")

    with pytest.raises(MODULE.ContractError, match="changed after preparation"):
        MODULE.verify_prepared_plan(root)


def test_prepared_plan_rechecks_clean_release_payload(tmp_path: Path) -> None:
    fixture, root, _, _ = _prepare(tmp_path)
    release_file = Path(fixture["release"]) / "CytoBridge/tl/core/methods.py"
    release_file.write_text("tampered release payload\n")

    with pytest.raises(MODULE.ContractError, match="Git checkout"):
        MODULE.verify_prepared_plan(root)


def test_python_identity_requires_complete_lexical_runtime_record(
    tmp_path: Path,
) -> None:
    fixture = _write_fixture(tmp_path)
    launcher = json.loads(Path(fixture["launcher_path"]).read_text())
    python_identity = dict(launcher["release"]["python_executable"])
    python_identity.pop("symlink_chain")

    with pytest.raises(MODULE.ContractError, match="lexical/runtime"):
        MODULE._verify_python_binding(python_identity)


def test_no_lr_config_rejects_even_null_predictor_keys(tmp_path: Path) -> None:
    model_dir = tmp_path / "training"
    model_dir.mkdir()
    config = _model_config("no_lr_prior", model_dir)
    config["model"]["interaction_net"]["edge_predictor_path"] = None
    config["model"]["interaction_net"]["edge_predictor_thre"] = None
    config_path = model_dir / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    _write_json(model_dir / "training_run_summary.json", {"schema_version": 1})
    for stage in config["training"]["plan"]:
        filename = (
            "score_model.pth" if stage["mode"] == "score_matching" else "best_model.pth"
        )
        checkpoint = model_dir / stage["name"] / filename
        checkpoint.parent.mkdir()
        checkpoint.write_bytes(b"checkpoint")

    with pytest.raises(MODULE.ContractError, match="carries predictor fields"):
        MODULE._checkpoint_inventory(
            "zebrafish_no_lr_prior",
            "no_lr_prior",
            model_dir,
            _identity(config_path),
        )


def test_validation_rejects_compact_target_splicing(tmp_path: Path) -> None:
    _, _, plan, _ = _prepare(tmp_path)
    profile = "zebrafish"
    _write_prediction_outputs(plan, profile)
    root = Path(plan["profiles"][profile]["paths"]["predictions"])
    run_summary = json.loads((root / "run_summary.json").read_text())
    run_summary["prediction_summaries"][0]["prediction_npz"] = run_summary[
        "prediction_summaries"
    ][1]["prediction_npz"]
    _write_json(root / "run_summary.json", run_summary)

    with pytest.raises(MODULE.ContractError, match="compact prediction"):
        MODULE._validate_prediction_outputs(plan, profile, plan["profiles"][profile])


def test_validation_rejects_simulator_contract_drift(tmp_path: Path) -> None:
    _, _, plan, _ = _prepare(tmp_path)
    profile = "zebrafish_no_interaction"
    _write_prediction_outputs(plan, profile)
    root = Path(plan["profiles"][profile]["paths"]["predictions"])
    summary = json.loads((root / "t1/summary.json").read_text())
    summary["simulation"][
        "interaction_grouping_seed"
    ] = MODULE.INTERACTION_GROUPING_SEED
    _rewrite_prediction_summary(plan, profile, 1, summary)

    with pytest.raises(MODULE.ContractError, match="prediction summary"):
        MODULE._validate_prediction_outputs(plan, profile, plan["profiles"][profile])


def test_validation_rejects_score_provenance_not_bound_to_input_artifacts(
    tmp_path: Path,
) -> None:
    _, _, plan, _ = _prepare(tmp_path)
    _write_all_outputs(plan)
    profile = "zebrafish_no_lr_prior"
    score_root = Path(plan["profiles"][profile]["paths"]["scores"])
    metrics = pd.read_csv(score_root / "full_data_metrics_long.csv")
    metrics["source_roster_sha256"] = "0" * 64
    _rewrite_score_table(plan, profile, metrics)

    with pytest.raises(MODULE.ContractError, match="source roster SHA-256"):
        MODULE._validate_all_outputs(plan)


def test_validation_rejects_cross_arm_transform_drift(tmp_path: Path) -> None:
    _, _, plan, _ = _prepare(tmp_path)
    _write_all_outputs(plan)
    profile = "zebrafish_no_lr_prior"
    score_root = Path(plan["profiles"][profile]["paths"]["scores"])
    transform = score_root / "transforms/full_data.json"
    transform.write_text('{"tampered": true}\n')
    metrics = pd.read_csv(score_root / "full_data_metrics_long.csv")
    metrics["transform_sha256"] = _sha256(transform)
    _rewrite_score_table(plan, profile, metrics)

    with pytest.raises(
        MODULE.ContractError, match="recomputed from training reference"
    ):
        MODULE._validate_all_outputs(plan)


@pytest.mark.parametrize(
    ("column", "replacement", "message"),
    (
        ("sliced_w2", 999.0, "differs from NPZ recomputation"),
        ("exact_w1", 999.0, "differs from NPZ recomputation"),
        ("exact_w2", 999.0, "differs from NPZ recomputation"),
        ("projection_sha256", "0" * 64, "projection basis"),
        ("tmv", 999.0, "was not recomputed"),
        ("n_observed", 999, "was not recomputed"),
        ("exact_ot_observed_points", 799, "was not recomputed"),
    ),
)
def test_validation_recomputes_score_evidence_from_frozen_npz(
    tmp_path: Path, column: str, replacement: object, message: str
) -> None:
    _, _, plan, _ = _prepare(tmp_path)
    profile = "admouse"
    _write_prediction_outputs(plan, profile)
    _write_score_outputs(plan, profile)
    score_root = Path(plan["profiles"][profile]["paths"]["scores"])
    metrics = pd.read_csv(score_root / "full_data_metrics_long.csv")
    metrics.loc[metrics["target"].eq(1), column] = replacement
    if column == "sliced_w2":
        metrics.loc[metrics["target"].eq(1), "primary_value"] = replacement
    _rewrite_score_table(plan, profile, metrics)

    predictions, _ = MODULE._validate_prediction_outputs(
        plan, profile, plan["profiles"][profile]
    )
    with pytest.raises(MODULE.ContractError, match=message):
        MODULE._validate_score_outputs(
            plan, profile, plan["profiles"][profile], predictions
        )


def test_report_validates_paired_scores_and_writes_publication_bundle(
    tmp_path: Path,
) -> None:
    _, root, plan, plan_sha = _prepare(tmp_path)
    _write_all_outputs(plan)
    verified_root, loaded, loaded_sha = MODULE.verify_prepared_plan(root)
    combined, source_artifacts = MODULE._validate_all_outputs(loaded)

    assert verified_root == root
    assert loaded_sha == plan_sha
    assert len(combined) == 585
    report_dir = root / "report"
    manifest_path = MODULE.generate_report(
        root, loaded, loaded_sha, combined, source_artifacts, report_dir
    )

    paired = pd.read_csv(report_dir / "paired_target_deltas.csv")
    sliced = paired.loc[paired["metric"].eq("sliced_w2")]
    assert np.allclose(sliced["no_lr_prior_relative_to_full"], 0.1)
    assert np.allclose(sliced["no_interaction_relative_to_full"], 0.2)
    assert set(sliced["space"]) == set(MODULE.SPACE_ORDER)
    assert set(sliced["dataset"]) == set(MODULE.DATASET_ORDER)
    tmv = pd.read_csv(report_dir / "paired_tmv_deltas.csv")
    assert np.allclose(tmv["no_lr_prior_relative_to_full"], 0.1)
    assert np.allclose(tmv["no_interaction_relative_to_full"], 0.2)
    caption = (report_dir / "figure_caption.md").read_text()
    assert "in-sample full-data reconstruction" in caption
    assert "must not be interpreted" in caption
    provenance = (report_dir / "PROVENANCE.md").read_text()
    assert "## Source paths" in provenance
    assert "## Rebuild" in provenance
    pdf = report_dir / f"{MODULE.FIGURE_BASENAME}.pdf"
    png = report_dir / f"{MODULE.FIGURE_BASENAME}.png"
    assert pdf.read_bytes().startswith(b"%PDF-")
    with Image.open(png) as image:
        assert image.width >= 2600
        assert image.height >= 3700
        assert image.info["dpi"][0] == pytest.approx(320, abs=1)
    manifest = json.loads(manifest_path.read_text())
    assert manifest["status"] == "complete"
    assert manifest["evaluation_plan_sha256"] == plan_sha
    assert "run_summary" in manifest["source_output_artifacts"]["zebrafish"]
    assert "source_roster" in manifest["source_output_artifacts"]["zebrafish"]
    sidecar = report_dir / "report_manifest.sha256"
    assert sidecar.read_text().split()[0] == _sha256(manifest_path)


def test_entry_is_shipped_in_source_distribution() -> None:
    manifest = (REPO_ROOT / "MANIFEST.in").read_text()
    assert "include scripts/run_matched_ablation_benchmark_evaluation.py" in manifest


def test_frozen_summary_constants_match_the_real_writer() -> None:
    assert MODULE.METHOD == WRITER_COMMON.METHOD
    assert MODULE.MATCHED_SEED == WRITER_COMMON.SEED
    assert MODULE.PREDICTION_N == WRITER_COMMON.PREDICTION_N
    assert MODULE.INFERENCE_SIGMA == WRITER_COMMON.SIGMA
    assert MODULE.ALPHA_EXPRESS == WRITER_COMMON.ALPHA_EXPRESS
    assert MODULE.ALPHA_SPATIAL == WRITER_COMMON.ALPHA_SPATIAL
    assert MODULE.EXPECTED_IMPLEMENTATION == (
        f"CytoBridge alpha_spatial={WRITER_COMMON.ALPHA_SPATIAL:g}, "
        f"alpha_express={WRITER_COMMON.ALPHA_EXPRESS:g}"
    )


def test_metric_recomputation_matches_writer_core_on_real_npz(tmp_path: Path) -> None:
    package_name = "_matched_eval_writer_core"
    package = types.ModuleType(package_name)
    package.__path__ = [str(REPO_ROOT / "CytoBridge/tl/downstream")]
    sys.modules[package_name] = package
    for name in ("evaluation", "benchmark"):
        path = REPO_ROOT / "CytoBridge/tl/downstream" / f"{name}.py"
        spec = importlib.util.spec_from_file_location(f"{package_name}.{name}", path)
        assert spec is not None and spec.loader is not None
        loaded = importlib.util.module_from_spec(spec)
        sys.modules[f"{package_name}.{name}"] = loaded
        spec.loader.exec_module(loaded)
    writer = sys.modules[f"{package_name}.benchmark"]

    rng = np.random.default_rng(20260813)
    training = {
        "state": rng.normal(size=(31, 2)),
        "spatial": rng.normal(size=(31, 2)),
        "time": np.resize(np.arange(5), 31).astype(float),
        "row_id": np.asarray([f"r{index}" for index in range(31)]),
    }
    truth = {
        "state": rng.normal(size=(19, 2)),
        "spatial": rng.normal(size=(19, 2)),
    }
    prediction = tmp_path / "prediction.npz"
    predicted_state = rng.normal(size=(37, 2))
    predicted_spatial = rng.normal(size=(37, 2))
    weights = rng.uniform(0.01, 0.2, size=37)
    np.savez_compressed(
        prediction,
        state=predicted_state,
        spatial=predicted_spatial,
        weights=weights,
    )
    observed = REAL_METRIC_RECOMPUTE(
        dataset="zebrafish",
        prediction_path=prediction,
        training=training,
        truth=truth,
    ).sort_values(["space", "projection_repeat"])
    transform = writer.fit_frozen_benchmark_transform(
        training["state"], training["spatial"]
    )
    expected = writer.evaluate_spatiotemporal_prediction(
        transform=transform,
        benchmark="zebrafish",
        split="full_data",
        method=MODULE.METHOD,
        predicted_state=predicted_state,
        observed_state=truth["state"],
        predicted_spatial=predicted_spatial,
        observed_spatial=truth["spatial"],
        predicted_weights=weights,
        n_projections=MODULE.N_PROJECTIONS,
        projection_repeats=MODULE.PROJECTION_REPEATS,
        max_ot_points=MODULE.MAX_OT_POINTS,
    ).sort_values(["space", "projection_repeat"])
    assert (
        observed[["space", "projection_repeat"]]
        .reset_index(drop=True)
        .equals(expected[["space", "projection_repeat"]].reset_index(drop=True))
    )
    assert np.allclose(
        observed[["sliced_w2", "exact_w1", "exact_w2"]],
        expected[["sliced_w2", "exact_w1", "exact_w2"]],
        rtol=1e-12,
        atol=1e-13,
    )
