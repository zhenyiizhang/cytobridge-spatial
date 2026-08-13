from __future__ import annotations

import csv
import json
import multiprocessing
import shutil
import subprocess
from collections import Counter
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import yaml

from scripts.spatiotemporal_benchmark import build_inputs as input_builder
from scripts.spatiotemporal_benchmark import evaluate_predictions as evaluator
from scripts.spatiotemporal_benchmark import run_unified_benchmark as runner
from scripts.spatiotemporal_benchmark.static_baselines import run as static_runner


def _merge_one_status(path, method):
    runner.merge_status_rows(
        path,
        [
            {
                "track": "loto",
                "target": 1,
                "method": method,
                "status": "completed",
                "reason": "",
                "elapsed_seconds": 1.0,
            }
        ],
    )


def _write_input_manifest(root, cfg):
    manifest = root / "inputs/manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    config_path = (runner.CONFIG_DIR / f"{cfg['dataset_id']}.yaml").resolve()
    resolved_config = input_builder.resolve_config(
        input_builder.parse_args(["--config", str(config_path), "--validate-only"])
    )
    resolved_config_path = manifest.parent / "resolved_config.yaml"
    resolved_config_path.write_text(
        yaml.safe_dump(dict(resolved_config), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    split_targets = {
        "full_data": list(cfg["full_data_targets"]),
        **{f"loto_t{target}": [target] for target in cfg["loto_targets"]},
    }
    splits = {}
    benchmark_times = sorted(
        {0, *map(int, cfg["loto_targets"]), *map(int, cfg["full_data_targets"])}
    )
    for split_id, targets in split_targets.items():
        split_dir = manifest.parent / split_id
        split_dir.mkdir(parents=True, exist_ok=True)
        train = split_dir / "train.h5ad"
        reference = split_dir / "training_reference.npz"
        roster = split_dir / "source_roster.npz"
        if not train.is_file():
            train.write_bytes(f"{cfg['dataset_id']}:{split_id}:train".encode())
        if not reference.is_file():
            np.savez_compressed(reference, state=np.ones((1, cfg["state_dim"])))
        if not roster.is_file():
            np.savez_compressed(roster, indices=np.arange(2, dtype=np.int64))
        splits[split_id] = {
            "evaluation_targets": targets,
            "prediction_n": int(cfg["prediction_n"]),
            "source_time": 0 if split_id == "full_data" else int(targets[0]) - 1,
            "train_time_counts": {
                str(value): (
                    0
                    if split_id.startswith("loto_t") and value == int(targets[0])
                    else 3
                )
                for value in benchmark_times
            },
            "train": {
                "h5ad": {
                    "path": str(train.resolve()),
                    "sha256": runner.sha256_file(train),
                },
                "training_reference_npz": {
                    "path": str(reference.resolve()),
                    "sha256": runner.sha256_file(reference),
                },
                "source_roster_npz": {
                    "path": str(roster.resolve()),
                    "sha256": runner.sha256_file(roster),
                },
            },
        }
    manifest.write_text(
        json.dumps(
            {
                "status": "complete",
                "dataset_id": cfg["dataset_id"],
                "prediction_n": int(cfg["prediction_n"]),
                "config_source": {
                    "path": str(config_path),
                    "sha256": runner.sha256_file(config_path),
                },
                "resolved_config": {
                    "path": str(resolved_config_path.resolve()),
                    "relative_path": "resolved_config.yaml",
                    "sha256": runner.sha256_file(resolved_config_path),
                    "size_bytes": resolved_config_path.stat().st_size,
                },
                "splits": splits,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return manifest


def _install_current_test_config(monkeypatch, tmp_path, cfg):
    config_dir = tmp_path / "unified_benchmark_configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f"{cfg['dataset_id']}.yaml"
    config_path.write_text(
        yaml.safe_dump(dict(cfg), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    monkeypatch.setattr(runner, "CONFIG_DIR", config_dir)
    return config_path, runner.load_datasets([cfg["dataset_id"]])[cfg["dataset_id"]]


def _mutate_dataset_recipe(payload, field):
    if field == "source_roster_seed":
        payload[field] = int(payload[field]) + 1
    elif field in {"time_key", "state_key"}:
        payload[field] = f"{payload[field]}_changed"
    elif field == "expected_source_sha256":
        payload[field] = "f" * 64
    elif field == "preprocess_contract":
        payload[field] = deepcopy(payload[field])
        payload[field]["required_exact"] = dict(
            payload[field].get("required_exact", {}), n_pcs=51
        )
    else:
        raise AssertionError(field)


def _write_complete_prediction(
    root, method, track, target, cfg, *, interaction_mode="learned"
):
    manifest = _write_input_manifest(root, cfg)
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    split_id = "full_data" if track == "full_data" else f"loto_t{target}"
    split_train = manifest_payload["splits"][split_id]["train"]
    output = root / f"predictions/{track}/{method}/t{target}"
    output.mkdir(parents=True, exist_ok=True)
    prediction = output / "prediction.npz"
    arrays = {
        "state": np.ones(
            (int(cfg["prediction_n"]), int(cfg["state_dim"])), dtype=np.float32
        )
    }
    output_scope = "native_state"
    if method in {
        "cytobridge",
        "stvcr",
        "moscot",
        "paste",
        "spateo",
        "linear_centroid_shift",
        "random_independent_pairs",
    }:
        arrays["spatial"] = np.ones(
            (int(cfg["prediction_n"]), int(cfg["spatial_dim"])), dtype=np.float32
        )
        output_scope = (
            "hybrid_joint"
            if method in {"moscot", "paste", "spateo"}
            else "native_joint"
        )
    elif method == "wot":
        output_scope = "hybrid_state"
    if method in {"cytobridge", "stvcr"}:
        arrays["weights"] = (
            np.ones(int(cfg["prediction_n"]), dtype=np.float64)
            if method == "cytobridge"
            else np.full(
                int(cfg["prediction_n"]),
                1.0 / int(cfg["prediction_n"]),
                dtype=np.float64,
            )
        )
    np.savez_compressed(prediction, **arrays)
    summary = {
        "status": "complete",
        "dataset": cfg["dataset_id"],
        "method": runner.METHOD_NAME[method],
        "track": track,
        "regime": track,
        "split_id": split_id,
        "target_time": target,
        "source_time": manifest_payload["splits"][split_id]["source_time"],
        "prediction_n": int(cfg["prediction_n"]),
        "state_dim": int(cfg["state_dim"]),
        "spatial_dim": int(cfg["spatial_dim"]),
        "output_scope": output_scope,
        "native_vs_adapter": runner.NATIVE_VS_ADAPTER_BY_METHOD[method],
        "primary_benchmark_eligible": True,
        "input_manifest": str(manifest.resolve()),
        "input_manifest_sha256": runner.sha256_file(manifest),
        "train_h5ad_sha256": split_train["h5ad"]["sha256"],
        "training_reference_sha256": split_train["training_reference_npz"]["sha256"],
        "source_roster_sha256": split_train["source_roster_npz"]["sha256"],
        "prediction_npz": str(prediction.resolve()),
        "prediction_npz_sha256": runner.sha256_file(prediction),
    }
    if method == "cytobridge":
        include_interaction = interaction_mode != "none"
        interaction_m = 1024 if include_interaction else None
        summary.update(
            {
                "seed": 42,
                "interaction_mode": interaction_mode,
                "edge_prior_mode": interaction_mode,
                "include_interaction": include_interaction,
                "edge_predictor_used": interaction_mode == "learned",
                "interaction_m": interaction_m,
                "dt": 0.01,
                "sigma": 0.03,
                "alpha_express": 0.015,
                "alpha_spatial": 10.0,
                "include_score": True,
                "split_sde": False,
                "continuous_across_targets": track == "full_data",
                "native_mass": True,
                "native_growth": True,
                "weights_are_unnormalised": True,
                "predicted_mass": float(arrays["weights"].sum()),
                "truth_inputs_opened": False,
                "adapter_implementation": (
                    runner.current_cytobridge_adapter_implementation()
                ),
                "simulation": {
                    "interaction_mode": interaction_mode,
                    "edge_prior_mode": interaction_mode,
                    "include_interaction": include_interaction,
                    "edge_predictor_used": interaction_mode == "learned",
                    "interaction_m": interaction_m,
                    "loaded_model_interaction_group_size": interaction_m,
                    "interaction_group_binding": (
                        "exact_checkpoint_model_match"
                        if include_interaction
                        else "not_applicable_no_interaction_component"
                    ),
                    "interaction_grouping_seed": (
                        10_042 if include_interaction else None
                    ),
                    "stochastic_stream_contract": (
                        "interaction grouping uses an independent torch.Generator; "
                        "the global torch stream remains paired for Brownian diffusion"
                    ),
                    "dynamics_components": ["velocity", "growth", "score"]
                    + (["interaction"] if include_interaction else []),
                },
            }
        )
        roster_output = (
            root / "predictions" / track / method if track == "full_data" else output
        ) / "source_roster.npz"
        roster_output.parent.mkdir(parents=True, exist_ok=True)
        canonical_roster = Path(split_train["source_roster_npz"]["path"])
        shutil.copy2(canonical_roster, roster_output)
        summary["source_roster"] = {
            "source_roster": str(roster_output.resolve()),
            "source_roster_sha256": runner.sha256_file(roster_output),
            "canonical_input_roster": str(canonical_roster.resolve()),
            "canonical_input_roster_sha256": runner.sha256_file(canonical_roster),
            "source_time": summary["source_time"],
            "prediction_n": int(cfg["prediction_n"]),
        }
    if method == "stvcr":
        summary.update(
            {
                "native_output_n": int(cfg["prediction_n"]),
                "native_growth": True,
                "native_count_changed": False,
                "native_mass": True,
                "native_growth": True,
                "weights_are_unnormalised": True,
                "weight_semantics": (
                    "uniform initial-particle mass 1/5000; "
                    "sum(weights)=native_n/5000"
                ),
                "weight_sum": float(arrays["weights"].sum()),
                "growth_ratio": 1.0,
                "truth_inputs_opened": False,
            }
        )
    elif method in runner.DYNAMIC:
        summary.update(
            {
                "native_mass": False,
                "native_growth": False,
                "weights_are_unnormalised": False,
                "truth_inputs_opened": False,
            }
        )
    elif method != "cytobridge":
        summary.update(
            {
                "native_mass": False,
                "native_growth": False,
                "weights_are_unnormalised": False,
                "truth_artifact_opened": False,
                "target_n_used_for_prediction": False,
            }
        )
    if method in runner.DYNAMIC:
        fit = _write_complete_dynamic_fit(root, method, split_id, cfg)
        fit_manifest = fit / "fit_manifest.json"
        summary["fit_manifest"] = str(fit_manifest.resolve())
        summary["fit_manifest_sha256"] = runner.sha256_file(fit_manifest)
        summary["seed_base"] = runner.DYNAMIC_SEED
        summary["params"] = runner.DYNAMIC_DEFAULT_PARAMS[method]
        summary[
            "adapter_implementation"
        ] = runner.current_dynamic_adapter_implementation()
        summary["infer_seed"] = runner.stable_seed(
            runner.DYNAMIC_SEED,
            cfg["dataset_id"],
            split_id,
            method,
            "full_trajectory" if track == "full_data" else "loto_trajectory",
        )
        summary["shared_full_trajectory_seed_across_targets"] = track == "full_data"
    (output / "summary.json").write_text(
        json.dumps(summary),
        encoding="utf-8",
    )
    if method == "cytobridge":
        (output / "prediction.summary.json").write_text(
            json.dumps(summary), encoding="utf-8"
        )
        if track == "full_data":
            parent = root / "predictions/full_data/cytobridge"
            target_summaries = []
            for target_value in cfg["full_data_targets"]:
                target_prediction = parent / f"t{target_value}/prediction.npz"
                target_summary = parent / f"t{target_value}/summary.json"
                if target_prediction.is_file() and target_summary.is_file():
                    payload = json.loads(target_summary.read_text(encoding="utf-8"))
                    target_summaries.append(
                        {
                            "target": target_value,
                            "prediction_npz": str(target_prediction.resolve()),
                            "prediction_npz_sha256": runner.sha256_file(
                                target_prediction
                            ),
                            "predicted_mass": payload["predicted_mass"],
                        }
                    )
            # The production full-data command writes all targets atomically as
            # one continuous trajectory. Tests that create targets incrementally
            # use temporary placeholders so the parent contract remains exact.
            for target_value in cfg["full_data_targets"]:
                target_parent = parent / f"t{target_value}"
                target_prediction = target_parent / "prediction.npz"
                target_summary = target_parent / "summary.json"
                if not target_prediction.is_file():
                    target_parent.mkdir(parents=True, exist_ok=True)
                    arrays_for_target = {
                        key: np.asarray(value) for key, value in arrays.items()
                    }
                    np.savez_compressed(target_prediction, **arrays_for_target)
                    placeholder = dict(summary)
                    placeholder["target_time"] = target_value
                    placeholder["prediction_npz"] = str(target_prediction.resolve())
                    placeholder["prediction_npz_sha256"] = runner.sha256_file(
                        target_prediction
                    )
                    target_summary.write_text(json.dumps(placeholder), encoding="utf-8")
                    (target_parent / "prediction.summary.json").write_text(
                        json.dumps(placeholder), encoding="utf-8"
                    )
            target_summaries = []
            for target_value in cfg["full_data_targets"]:
                target_prediction = parent / f"t{target_value}/prediction.npz"
                payload = json.loads(
                    (target_prediction.parent / "summary.json").read_text(
                        encoding="utf-8"
                    )
                )
                target_summaries.append(
                    {
                        "target": target_value,
                        "prediction_npz": str(target_prediction.resolve()),
                        "prediction_npz_sha256": runner.sha256_file(target_prediction),
                        "predicted_mass": payload["predicted_mass"],
                    }
                )
            run_summary = {
                "status": "complete",
                "method": runner.METHOD_NAME["cytobridge"],
                "regime": "full_data",
                "split_id": "full_data",
                "source_time": summary["source_time"],
                "targets": cfg["full_data_targets"],
                "single_continuous_non_split_call": True,
                "intermediate_reset": False,
                "spatial_warp_applied": False,
                "prediction_n": int(cfg["prediction_n"]),
                "seed": 42,
                "source_roster": summary["source_roster"],
                "prediction_summaries": target_summaries,
                "input_manifest_sha256": summary["input_manifest_sha256"],
                "train_h5ad_sha256": summary["train_h5ad_sha256"],
                "training_reference_sha256": summary["training_reference_sha256"],
                "source_roster_sha256": summary["source_roster_sha256"],
            }
            (parent / "run_summary.json").write_text(
                json.dumps(run_summary), encoding="utf-8"
            )
    if method in runner.STATIC:
        _write_static_run_manifest(root, method, track, target, cfg)
    return output


def _write_static_run_manifest(root, method, track, target, cfg):
    registry = json.loads(runner.STATIC_REGISTRY.read_text(encoding="utf-8"))
    method_spec = registry["methods"][method]
    representation_spec = method_spec["representations"]["matched_state_spatial"]
    manifest_path = root / "inputs/manifest.json"
    input_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    split_id = "full_data" if track == "full_data" else f"loto_t{target}"
    split = input_payload["splits"][split_id]
    train = split["train"]
    source_time = float(split["source_time"])
    time_values = sorted(
        float(value)
        for value, count in split["train_time_counts"].items()
        if int(count) > 0
    )
    if track == "loto":
        run_dir = root / f"predictions/loto/{method}/t{target}"
        target_dirs = [run_dir]
        left = max(value for value in time_values if value < float(target))
        right = min(value for value in time_values if value > float(target))
        fitted_pairs = [(left, right)]
    else:
        run_dir = root / f"predictions/full_data/{method}"
        target_dirs = sorted(run_dir.glob("t*"))
        fitted_pairs = list(zip(time_values[:-1], time_values[1:]))

    predictions = {}
    for target_dir in target_dirs:
        prediction = target_dir / "prediction.npz"
        summary = target_dir / "summary.json"
        if not prediction.is_file() or not summary.is_file():
            continue
        target_payload = json.loads(summary.read_text(encoding="utf-8"))
        target_value = float(target_payload["target_time"])
        predictions[str(target_value)] = {
            "path": str(prediction.resolve()),
            "sha256": runner.sha256_file(prediction),
            "summary": {
                "path": str(summary.resolve()),
                "sha256": runner.sha256_file(summary),
            },
        }

    def input_record(name):
        record = train[name]
        return Path(record.get("relative_path", record.get("path"))).resolve()

    payload = {
        "schema_version": "2.0.0",
        "status": "complete",
        "dataset": cfg["dataset_id"],
        "method": method,
        "adapter_implementation": runner.current_static_adapter_implementation(),
        "representation": "matched_state_spatial",
        "method_spec": method_spec,
        "representation_spec": representation_spec,
        "parameters": representation_spec["default_parameters"],
        "seed": 20_260_718,
        "max_fit_n": 800,
        "dry_run": False,
        "prediction_written": True,
        "primary_benchmark_eligible": True,
        "surrogate_attempted": False,
        "input": {
            "train_h5ad": str(input_record("h5ad")),
            "train_h5ad_sha256": train["h5ad"]["sha256"],
            "input_manifest": str(manifest_path.resolve()),
            "input_manifest_sha256": runner.sha256_file(manifest_path),
            "input_manifest_split": split_id,
            "input_manifest_h5ad_verified": True,
            "training_reference": str(input_record("training_reference_npz")),
            "training_reference_sha256": train["training_reference_npz"]["sha256"],
            "input_manifest_training_reference_expected_sha256": train[
                "training_reference_npz"
            ]["sha256"],
            "input_manifest_training_reference_verified": True,
            "source_roster": str(input_record("source_roster_npz")),
            "source_roster_sha256": train["source_roster_npz"]["sha256"],
        },
        "protocol": {
            "mode": "loto" if track == "loto" else "no-holdout",
            "requested_target_time": float(target) if track == "loto" else None,
            "loto_target": float(target) if track == "loto" else None,
            "time_values": time_values,
            "prediction_n": int(cfg["prediction_n"]),
            "truth_artifact_opened": False,
            "truth_cell_count_read": False,
            "target_n_used_for_prediction": False,
            "comparable_to_strict_loto": track == "loto",
            "no_holdout_is_in_sample": track == "full_data",
        },
        "output_scope": {
            "scope": representation_spec["output_scope"],
            "state_dimensions": int(cfg["state_dim"]),
            "spatial_dimensions": 0 if method == "wot" else int(cfg["spatial_dim"]),
            "hybrid_adapter": bool(representation_spec["hybrid"]),
            "weights_exported": False,
            "growth_or_total_mass_evaluated": False,
        },
        "outputs": {"prediction_by_time": predictions},
    }
    if method in runner.EXTERNAL_STATIC:
        source_root = (root / "software" / method).resolve()
        dependency = {
            "available": True,
            "requested_source_root": str(source_root),
            "source_mode": True,
            "expected_git_commit": method_spec["reference_commit"],
            "git_commit": method_spec["reference_commit"],
            "git_dirty": False,
            "git_toplevel": str(source_root),
            "source_commit_verified": True,
            "source_import_roots": [str(source_root)],
            "distribution_metadata_required": False,
            "module_file": str(source_root / method / "__init__.py"),
            "module_from_requested_source": True,
            "compatibility_versions": method_spec.get("compatibility_versions", {}),
        }
        payload["official_runs"] = [
            {
                "from": left,
                "to": right,
                "parameters": representation_spec["default_parameters"],
                "dependency": dependency,
                "official_api": method_spec["official_api"],
                "representation": "matched_state_spatial",
            }
            for left, right in fitted_pairs
        ]
    else:
        payload["control_run"] = {"control": method}
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_manifest.json").write_text(
        json.dumps(payload, sort_keys=True), encoding="utf-8"
    )
    return run_dir / "run_manifest.json"


def _run_real_static_loto_producer(root, cfg):
    split_dir = root / "inputs/loto_t1"
    split_dir.mkdir(parents=True, exist_ok=True)
    train = split_dir / "train.h5ad"
    reference = split_dir / "training_reference.npz"
    canonical_roster = split_dir / "source_roster.npz"
    manifest = _write_input_manifest(root, cfg)

    times = np.repeat(np.asarray([0.0, 2.0]), 3)
    row_ids = np.asarray([f"row_{index}" for index in range(len(times))], dtype=str)
    state = np.arange(len(times) * int(cfg["state_dim"]), dtype=np.float32).reshape(
        len(times), int(cfg["state_dim"])
    )
    spatial = np.column_stack((times, np.arange(len(times), dtype=np.float32)))
    obs = pd.DataFrame(
        {"benchmark_time": times, "row_id": row_ids}, index=pd.Index(row_ids)
    )
    data = ad.AnnData(X=np.ones((len(times), 2), dtype=np.float32), obs=obs)
    data.obsm["benchmark_state"] = state
    data.obsm["benchmark_spatial"] = spatial.astype(np.float32)
    data.uns["cytobridge_benchmark_contract"] = {
        "dataset_id": cfg["dataset_id"],
        "state_key": "benchmark_state",
        "state_dim": int(cfg["state_dim"]),
        "spatial_key": "benchmark_spatial",
        "spatial_dim": int(cfg["spatial_dim"]),
        "time_key": "benchmark_time",
        "row_id_key": "row_id",
        "prediction_n": int(cfg["prediction_n"]),
        "source_roster_support_n": 800,
        "source_roster_seed": 20_260_718,
        "loto_targets": [1],
        "full_data_targets": list(cfg["full_data_targets"]),
        "target_removed": True,
        "held_out_benchmark_time": 1,
    }
    data.write_h5ad(train)
    np.savez_compressed(
        reference,
        state=state,
        spatial=spatial.astype(np.float32),
        time=times,
        row_id=row_ids,
    )
    roster_indices = np.resize(np.arange(3, dtype=np.int64), int(cfg["prediction_n"]))
    np.savez_compressed(
        canonical_roster,
        row_id=row_ids[roster_indices],
        source_time=np.asarray([0.0]),
        spatial=spatial.astype(np.float32)[roster_indices],
        state=state[roster_indices],
    )
    split_train = {
        "h5ad": {"path": str(train.resolve()), "sha256": runner.sha256_file(train)},
        "training_reference_npz": {
            "path": str(reference.resolve()),
            "sha256": runner.sha256_file(reference),
        },
        "source_roster_npz": {
            "path": str(canonical_roster.resolve()),
            "sha256": runner.sha256_file(canonical_roster),
        },
    }
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    manifest_payload["splits"] = {
        "loto_t1": {
            "evaluation_targets": [1],
            "prediction_n": int(cfg["prediction_n"]),
            "source_time": 0,
            "train_time_counts": {"0": 3, "1": 0, "2": 3},
            "train": split_train,
        }
    }
    manifest.write_text(json.dumps(manifest_payload, sort_keys=True), encoding="utf-8")
    output = root / "predictions/loto/linear_centroid_shift/t1"
    code = static_runner.main(
        [
            "run",
            "--method",
            "linear_centroid_shift",
            "--evaluation-mode",
            "loto",
            "--target-time",
            "1",
            "--input-h5ad",
            str(train),
            "--input-manifest",
            str(manifest),
            "--output-dir",
            str(output),
            "--max-fit-n",
            "800",
        ]
    )
    assert code == 0
    return output


def _write_complete_dynamic_fit(root, method, split, cfg):
    input_manifest = root / "inputs/manifest.json"
    if not input_manifest.is_file():
        _write_input_manifest(root, cfg)
    fit_dir = root / f"fits/{method}/{split}"
    fit_dir.mkdir(parents=True, exist_ok=True)
    roster = fit_dir / "source_roster.npz"
    input_payload = json.loads(input_manifest.read_text(encoding="utf-8"))
    canonical_roster = Path(
        input_payload["splits"][split]["train"]["source_roster_npz"]["path"]
    )
    split_train = input_payload["splits"][split]["train"]
    shutil.copy2(canonical_roster, roster)
    manifest_path, summary_path = (
        fit_dir / "fit_manifest.json",
        fit_dir / "summary.json",
    )
    payload = {
        "status": "complete",
        "method": method,
        "dataset": cfg["dataset_id"],
        "split_id": split,
        "regime": "full_data" if split == "full_data" else "loto",
        "seed_base": runner.DYNAMIC_SEED,
        "fit_seed": runner.stable_seed(
            runner.DYNAMIC_SEED, cfg["dataset_id"], split, method, "fit"
        ),
        "params": runner.DYNAMIC_DEFAULT_PARAMS[method],
        "adapter_implementation": runner.current_dynamic_adapter_implementation(),
        "method_pin_registry_sha256": runner.sha256_file(runner.DYNAMIC_PINS),
        "method_pin": json.loads(runner.DYNAMIC_PINS.read_text(encoding="utf-8"))[
            "methods"
        ][method],
        "source_expected_git_commit": json.loads(
            runner.DYNAMIC_PINS.read_text(encoding="utf-8")
        )["methods"][method]["commit"],
        "source_git_commit": json.loads(
            runner.DYNAMIC_PINS.read_text(encoding="utf-8")
        )["methods"][method]["commit"],
        "source_tracked_tree_clean": True,
        "source_worktree_clean": True,
        "input_manifest_sha256": runner.sha256_file(input_manifest),
        "train_h5ad_sha256": split_train["h5ad"]["sha256"],
        "training_reference_sha256": split_train["training_reference_npz"]["sha256"],
        "source_roster_sha256": split_train["source_roster_npz"]["sha256"],
        "fit_manifest": str(manifest_path.resolve()),
        "summary": str(summary_path.resolve()),
        "source_roster": {
            "canonical_input_roster": str(canonical_roster.resolve()),
            "canonical_input_roster_sha256": split_train["source_roster_npz"]["sha256"],
        },
        "artifacts": {
            "source_roster": {
                "path": str(roster.resolve()),
                "sha256": runner.sha256_file(roster),
            },
        },
    }
    required = runner.DYNAMIC_REQUIRED_ARTIFACTS[method] - {"source_roster"}
    for name in required:
        suffix = ".npz" if name == "state_transform" else ".bin"
        artifact = fit_dir / f"{name}{suffix}"
        artifact.write_bytes(f"{method}:{split}:{name}".encode())
        payload["artifacts"][name] = {
            "path": str(artifact.resolve()),
            "sha256": runner.sha256_file(artifact),
        }
    text = json.dumps(payload)
    manifest_path.write_text(text, encoding="utf-8")
    summary_path.write_text(text, encoding="utf-8")
    return fit_dir


def _write_formal_cytobridge_model(formal_dataset_root, root, cfg, split_id):
    model = (
        formal_dataset_root / "training"
        if split_id == "full_data"
        else root / "fits" / "cytobridge" / split_id
    )
    model.mkdir(parents=True, exist_ok=True)
    training_config = formal_dataset_root / cfg["benchmark"]["training_config"]
    training_config.parent.mkdir(parents=True, exist_ok=True)
    plan = {
        "training": {
            "plan": [
                {"name": "Finetune", "mode": "neural_ode"},
                {"name": "Score_Refine", "mode": "score_matching"},
            ]
        }
    }
    training_config.write_text(yaml.safe_dump(plan), encoding="utf-8")
    saved_config = model / "config.yaml"
    if saved_config != training_config:
        saved_config.write_text(yaml.safe_dump(plan), encoding="utf-8")
    for relative, payload in (
        ("Finetune/best_model.pth", b"finetune"),
        ("Score_Refine/score_model.pth", b"score"),
        ("adata.h5ad", b"frozen-adata"),
    ):
        destination = model / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
    contract = runner._current_cytobridge_model_contract(
        root,
        formal_dataset_root,
        cfg,
        "full_data" if split_id == "full_data" else "loto",
        split_id,
    )
    assert contract is not None
    return model, contract


def test_dataset_matrix_has_the_complete_target_plan():
    configs = runner.load_datasets(list(runner.DATASETS))
    assert all(
        "corrected-matched-ablation-20260813-3c87a3e-r1" in cfg["input_h5ad"]
        for cfg in configs.values()
    )
    expected_source_sha256 = {
        "zebrafish": "14753bbfdd05c9971b4ed5db4a7e70693479c7b7074ed1ef1d6f3187e1119811",
        "mosta": "8b9ca0ad3475040235036548d54b96272bf6c49f057f6c2a643152c11350ce25",
        "arista": "eb72988986af42aeb8853c253d07218a9cb6294615eff55178fc0b409823205d",
        "admouse": "26d9a68acde90afc09d11b9c17de38525e37b1ee6b2e0290ddbda3efbe9ab968",
    }
    acceptance_sha256 = (
        "c4f8e203e2da73fe78e28525516bbec192d3cbbd35d423dcd64080a0f83a10df"
    )
    for dataset, cfg in configs.items():
        assert cfg["expected_source_sha256"] == expected_source_sha256[dataset]
        audits = cfg["preprocess_contract"]["external_audits"]
        assert len(audits) == 1
        assert audits[0]["sha256"] == acceptance_sha256
        assert audits[0]["required_exact"] == {
            "status": "PASS",
            "datasets": {dataset: {"status": "PASS"}},
            "matched_families": {dataset: {"status": "PASS"}},
        }
    assert configs["zebrafish"]["loto_targets"] == [1, 2, 3]
    assert configs["zebrafish"]["full_data_targets"] == [1, 2, 3, 4]
    assert configs["mosta"]["loto_targets"] == [1, 2]
    assert configs["mosta"]["full_data_targets"] == [1, 2, 3]
    assert configs["arista"]["loto_targets"] == [1, 2, 3]
    assert configs["arista"]["full_data_targets"] == [1, 2, 3, 4]
    assert configs["admouse"]["loto_targets"] == [1]
    assert configs["admouse"]["full_data_targets"] == [1, 2]


def test_launcher_defaults_bind_the_accepted_final_root():
    assert runner.DEFAULT_FORMAL_ROOT.name == (
        "corrected-matched-ablation-20260813-3c87a3e-r1"
    )
    assert runner.DEFAULT_RUN_ROOT.name == (
        "corrected-benchmark-20260813-matched-3c87a3e-c4f8e203-r1"
    )


def test_cytobridge_loto_preparation_receives_the_training_profile(tmp_path):
    cfg = runner.load_datasets(["admouse"])["admouse"]
    commands = runner.cytobridge_commands(
        Path("python"),
        cfg,
        tmp_path / "formal/admouse",
        tmp_path / "benchmark/admouse/inputs/manifest.json",
        tmp_path / "benchmark/admouse",
        "loto_t1",
        [1],
        "cpu",
    )
    prepare = commands[0]
    self_config = tmp_path / "formal/admouse/training/config.yaml"
    assert "prepare-loto" in prepare
    assert prepare[prepare.index("--training-config") + 1] == str(self_config)
    assert "--database" in prepare

    learned_cfg = runner.load_datasets(["zebrafish"])["zebrafish"]
    learned = runner.cytobridge_commands(
        Path("python"),
        learned_cfg,
        tmp_path / "formal/zebrafish",
        tmp_path / "benchmark/zebrafish/inputs/manifest.json",
        tmp_path / "benchmark/zebrafish",
        "loto_t1",
        [1],
        "cpu",
    )[0]
    assert "--database" in learned

    radius_cfg = dict(cfg)
    radius_cfg["benchmark"] = dict(cfg["benchmark"], edge_prior_mode="all_spatial")
    radius = runner.cytobridge_commands(
        Path("python"),
        radius_cfg,
        tmp_path / "formal/admouse",
        tmp_path / "benchmark/admouse/inputs/manifest.json",
        tmp_path / "benchmark/admouse",
        "loto_t1",
        [1],
        "cpu",
    )[0]
    assert "--database" not in radius

    none_cfg = dict(cfg)
    none_cfg["benchmark"] = dict(
        cfg["benchmark"],
        edge_prior_mode="none",
        training_config="training/no_interaction/config.yaml",
    )
    none = runner.cytobridge_commands(
        Path("python"),
        none_cfg,
        tmp_path / "formal/admouse",
        tmp_path / "benchmark/admouse/inputs/manifest.json",
        tmp_path / "benchmark/admouse",
        "loto_t1",
        [1],
        "cpu",
    )
    expected_none_config = str(
        tmp_path / "formal/admouse/training/no_interaction/config.yaml"
    )
    assert len(none) == 3
    assert "--database" not in none[0]
    assert [item[item.index("--training-config") + 1] for item in none] == [
        expected_none_config,
        expected_none_config,
        expected_none_config,
    ]
    assert "prepare-loto" in none[0]
    assert "fit-loto" in none[1]
    assert "infer-loto" in none[2]

    none_full = runner.cytobridge_commands(
        Path("python"),
        none_cfg,
        tmp_path / "formal/admouse",
        tmp_path / "benchmark/admouse/inputs/manifest.json",
        tmp_path / "benchmark/admouse",
        "full_data",
        [1, 2],
        "cpu",
    )
    assert len(none_full) == 2
    assert "validate-model" in none_full[0]
    assert "infer-full" in none_full[1]
    assert [item[item.index("--training-config") + 1] for item in none_full] == [
        expected_none_config,
        expected_none_config,
    ]


def test_mosta_full_dry_run_includes_t3(tmp_path, capsys):
    runner.main(
        [
            "--datasets",
            "mosta",
            "--formal-root",
            str(tmp_path / "formal"),
            "--run-root",
            str(tmp_path / "benchmark"),
            "--dry-run",
            "run",
            "--methods",
            "stvcr",
            "--tracks",
            "full_data",
        ]
    )
    output = capsys.readouterr().out
    assert "--target-time 1" in output
    assert "--target-time 2" in output
    assert "--target-time 3" in output


def test_not_applicable_rows_are_written_without_running_a_job(tmp_path):
    runner.main(
        [
            "--datasets",
            "admouse",
            "--run-root",
            str(tmp_path),
            "run",
            "--methods",
            "spatrack",
            "--tracks",
            "loto",
            "full_data",
        ]
    )
    path = tmp_path / "admouse" / "status" / "method_target_status.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [(row["track"], row["target"]) for row in rows] == [
        ("full_data", "1"),
        ("full_data", "2"),
        ("loto", "1"),
    ]
    assert {row["status"] for row in rows} == {"not_applicable"}


def test_primary_evaluation_uses_the_shared_registry_and_excludes_sensitivity_only(
    tmp_path, capsys
):
    runner.main(
        [
            "--datasets",
            "admouse",
            "--run-root",
            str(tmp_path),
            "--dry-run",
            "evaluate",
            "--tracks",
            "loto",
        ]
    )
    output = capsys.readouterr().out
    assert "--method-registry" in output
    assert str(runner.METHOD_REGISTRY) in output
    assert "CytoBridge-0.015" in output
    assert "random_independent_pairs" in output
    assert "spatrack" not in output.lower()


def test_execute_maps_missing_timeout_oom_and_failure(tmp_path):
    missing = tmp_path / "missing"
    assert (
        runner.execute([], [missing], 1, tmp_path / "missing.log")[0] == "not_available"
    )

    completed = SimpleNamespace(returncode=0, stdout="ok")
    with mock.patch.object(runner.subprocess, "run", return_value=completed):
        assert runner.execute([["job"]], [], 1, tmp_path / "ok.log")[0] == "completed"

    oom = SimpleNamespace(returncode=1, stdout="CUDA out of memory")
    with mock.patch.object(runner.subprocess, "run", return_value=oom):
        assert runner.execute([["job"]], [], 1, tmp_path / "oom.log")[0] == "oom"

    failed = SimpleNamespace(returncode=2, stdout="bad input")
    with mock.patch.object(runner.subprocess, "run", return_value=failed):
        assert runner.execute([["job"]], [], 1, tmp_path / "failed.log")[0] == "failed"

    with mock.patch.object(
        runner.subprocess,
        "run",
        side_effect=subprocess.TimeoutExpired("job", 1, output="slow"),
    ):
        assert (
            runner.execute([["job"]], [], 1, tmp_path / "timeout.log")[0] == "timeout"
        )


def test_partial_full_run_keeps_completed_target_and_failed_later_target(
    tmp_path, monkeypatch
):
    cfg = runner.load_datasets(["admouse"])["admouse"]
    args = SimpleNamespace(
        run_root=tmp_path,
        formal_root=tmp_path / "formal",
        methods=["stories"],
        tracks=["full_data"],
        software_root=tmp_path / "software",
        device="cpu",
        timeout=1,
        dry_run=False,
    )
    root = tmp_path / "admouse"
    _write_complete_prediction(root, "stories", "full_data", 1, cfg)
    _write_complete_dynamic_fit(root, "stories", "full_data", cfg)
    observed = {}

    def execute(commands, *unused):
        observed["commands"] = commands
        return "timeout", "timeout after t1"

    monkeypatch.setattr(runner, "execute", execute)

    runner.run_dataset("admouse", cfg, args, {}, {"stories": tmp_path})

    assert len(observed["commands"]) == 1
    assert "infer" in observed["commands"][0]
    assert (
        observed["commands"][0][observed["commands"][0].index("--target-time") + 1]
        == "2"
    )

    rows = list(
        csv.DictReader(
            (tmp_path / "admouse/status/method_target_status.csv").open(
                newline="", encoding="utf-8"
            )
        )
    )
    assert [(row["target"], row["status"]) for row in rows] == [
        ("1", "completed"),
        ("2", "timeout"),
    ]


def test_complete_job_is_resumed_without_overwriting_outputs(tmp_path, monkeypatch):
    cfg = runner.load_datasets(["admouse"])["admouse"]
    args = SimpleNamespace(
        run_root=tmp_path,
        formal_root=tmp_path / "formal",
        methods=["stories"],
        tracks=["full_data"],
        software_root=tmp_path / "software",
        device="cpu",
        timeout=1,
        dry_run=False,
    )
    for target in (1, 2):
        _write_complete_prediction(
            tmp_path / "admouse", "stories", "full_data", target, cfg
        )

    monkeypatch.setattr(
        runner,
        "execute",
        lambda *unused: (_ for _ in ()).throw(AssertionError("must not rerun")),
    )
    runner.run_dataset("admouse", cfg, args, {}, {"stories": tmp_path})

    rows = list(
        csv.DictReader(
            (tmp_path / "admouse/status/method_target_status.csv").open(
                newline="", encoding="utf-8"
            )
        )
    )
    assert [(row["target"], row["status"]) for row in rows] == [
        ("1", "completed"),
        ("2", "completed"),
    ]


def test_run_dataset_hashes_immutable_inputs_and_cytobridge_model_once(
    tmp_path, monkeypatch
):
    cfg = runner.load_datasets(["admouse"])["admouse"]
    run_root = tmp_path / "benchmark"
    root = run_root / "admouse"
    formal_root = tmp_path / "formal"
    formal = formal_root / "admouse"
    for method in ("cytobridge", "stories"):
        for target in cfg["full_data_targets"]:
            _write_complete_prediction(root, method, "full_data", target, cfg)

    model, model_contract = _write_formal_cytobridge_model(
        formal, root, cfg, "full_data"
    )
    for target in cfg["full_data_targets"]:
        output = root / f"predictions/full_data/cytobridge/t{target}"
        summary_path = output / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary.update(
            {
                "config_sha256": model_contract["saved_config_sha256"],
                "checkpoint_sha256": model_contract["checkpoint_sha256"],
                "training_reference_match": {
                    "proof": "saved_adata_exact_frozen_arrays",
                    "path": str(model_contract["saved_adata_path"]),
                    "sha256": model_contract["saved_adata_sha256"],
                },
            }
        )
        payload = json.dumps(summary)
        summary_path.write_text(payload, encoding="utf-8")
        (output / "prediction.summary.json").write_text(payload, encoding="utf-8")

    args = SimpleNamespace(
        run_root=run_root,
        formal_root=formal_root,
        methods=["cytobridge", "stories"],
        tracks=["full_data"],
        software_root=tmp_path / "software",
        device="cpu",
        timeout=1,
        dry_run=False,
    )
    monkeypatch.setattr(
        runner,
        "execute",
        lambda *unused: (_ for _ in ()).throw(
            AssertionError("complete jobs must not rerun")
        ),
    )
    original_sha256_file = runner.sha256_file
    hash_counts = Counter()

    def counted_sha256_file(path):
        hash_counts[Path(path).expanduser().resolve()] += 1
        return original_sha256_file(path)

    monkeypatch.setattr(runner, "sha256_file", counted_sha256_file)
    runner.run_dataset("admouse", cfg, args, {}, {"stories": tmp_path})

    manifest = json.loads((root / "inputs/manifest.json").read_text(encoding="utf-8"))
    split_train = manifest["splits"]["full_data"]["train"]
    immutable_inputs = [
        Path(split_train[name]["path"]).resolve()
        for name in (
            "h5ad",
            "training_reference_npz",
            "source_roster_npz",
        )
    ]
    assert {path: hash_counts[path] for path in immutable_inputs} == {
        path: 1 for path in immutable_inputs
    }

    model_paths = {
        (formal / cfg["benchmark"]["training_config"]).resolve(),
        (model / "config.yaml").resolve(),
        (model / "Finetune/best_model.pth").resolve(),
        (model / "Score_Refine/score_model.pth").resolve(),
        (model / "adata.h5ad").resolve(),
    }
    assert {path: hash_counts[path] for path in model_paths} == {
        path: 1 for path in model_paths
    }

    # Mutable predictions and dynamic fit products are intentionally re-read
    # during the post-job status pass; they must never enter the immutable cache.
    for method in ("cytobridge", "stories"):
        for target in cfg["full_data_targets"]:
            prediction = (
                root / f"predictions/full_data/{method}/t{target}/prediction.npz"
            ).resolve()
            assert hash_counts[prediction] >= 2
    fit_manifest = (root / "fits/stories/full_data/fit_manifest.json").resolve()
    assert hash_counts[fit_manifest] >= 2
    fit_payload = json.loads(fit_manifest.read_text(encoding="utf-8"))
    for record in fit_payload["artifacts"].values():
        assert hash_counts[Path(record["path"]).resolve()] >= 2


def test_resume_cache_rehashes_and_rejects_changed_input_bytes(tmp_path):
    cfg = runner.load_datasets(["admouse"])["admouse"]
    root = tmp_path / "admouse"
    _write_complete_prediction(root, "stories", "full_data", 1, cfg)
    cache = runner.ResumeValidationCache()
    assert runner.target_output_is_complete(
        root, "stories", "full_data", 1, cfg, validation_cache=cache
    )

    manifest = json.loads((root / "inputs/manifest.json").read_text(encoding="utf-8"))
    train = Path(manifest["splits"]["full_data"]["train"]["h5ad"]["path"])
    original = train.read_bytes()
    train.write_bytes(original + b"tampered")
    assert not runner.target_output_is_complete(
        root, "stories", "full_data", 1, cfg, validation_cache=cache
    )
    train.write_bytes(original)
    assert runner.target_output_is_complete(
        root, "stories", "full_data", 1, cfg, validation_cache=cache
    )


def test_resume_rejects_corrupt_or_tampered_prediction_pairs(tmp_path):
    cfg = runner.load_datasets(["admouse"])["admouse"]
    root = tmp_path / "admouse"
    manifest = _write_input_manifest(root, cfg)
    target = root / "predictions/full_data/stories/t1"
    target.mkdir(parents=True)
    prediction = target / "prediction.npz"
    prediction.write_bytes(b"not-an-npz")
    (target / "summary.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "dataset": "admouse",
                "method": "stories",
                "track": "full_data",
                "prediction_n_contract": cfg["prediction_n"],
                "input_manifest": str(manifest.resolve()),
                "input_manifest_sha256": runner.sha256_file(manifest),
                "target_time": 1,
                "prediction_npz": str(prediction.resolve()),
                "prediction_npz_sha256": runner.sha256_file(prediction),
            }
        ),
        encoding="utf-8",
    )
    assert not runner.target_output_is_complete(root, "stories", "full_data", 1, cfg)

    np.savez_compressed(
        prediction,
        state=np.ones((cfg["prediction_n"], cfg["state_dim"]), dtype=np.float32),
    )
    assert not runner.target_output_is_complete(root, "stories", "full_data", 1, cfg)

    summary = json.loads((target / "summary.json").read_text(encoding="utf-8"))
    summary["prediction_npz_sha256"] = runner.sha256_file(prediction)
    summary["prediction_npz"] = str(tmp_path / "different.npz")
    (target / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    assert not runner.target_output_is_complete(root, "stories", "full_data", 1, cfg)


def test_resume_binds_current_manifest_method_track_count_and_dimensions(tmp_path):
    cfg = runner.load_datasets(["admouse"])["admouse"]
    root = tmp_path / "admouse"
    output = _write_complete_prediction(root, "stories", "full_data", 1, cfg)
    summary_path = output / "summary.json"
    original = json.loads(summary_path.read_text(encoding="utf-8"))
    assert runner.target_output_is_complete(root, "stories", "full_data", 1, cfg)

    mutations = {
        "method": "mioflow",
        "track": "loto",
        "input_manifest_sha256": "f" * 64,
        "prediction_n": cfg["prediction_n"] + 1,
        "state_dim": cfg["state_dim"] + 1,
        "spatial_dim": cfg["spatial_dim"] + 1,
    }
    for key, value in mutations.items():
        changed = dict(original)
        changed[key] = value
        summary_path.write_text(json.dumps(changed), encoding="utf-8")
        assert not runner.target_output_is_complete(
            root, "stories", "full_data", 1, cfg
        ), key
    summary_path.write_text(json.dumps(original), encoding="utf-8")

    manifest = root / "inputs/manifest.json"
    original_manifest = manifest.read_text(encoding="utf-8")
    manifest_payload = json.loads(original_manifest)
    manifest_payload["audit_nonce"] = "changed-current-manifest"
    manifest.write_text(json.dumps(manifest_payload), encoding="utf-8")
    assert not runner.target_output_is_complete(root, "stories", "full_data", 1, cfg)

    manifest.write_text(original_manifest, encoding="utf-8")
    original["input_manifest_sha256"] = runner.sha256_file(manifest)
    restored_manifest = json.loads(original_manifest)
    train_path = Path(restored_manifest["splits"]["full_data"]["train"]["h5ad"]["path"])
    original_train = train_path.read_bytes()
    train_path.write_bytes(original_train + b"tampered")
    summary_path.write_text(json.dumps(original), encoding="utf-8")
    assert not runner.target_output_is_complete(root, "stories", "full_data", 1, cfg)
    train_path.write_bytes(original_train)

    prediction = output / "prediction.npz"
    np.savez_compressed(
        prediction,
        state=np.ones((cfg["prediction_n"], cfg["state_dim"] + 1), dtype=np.float32),
    )
    original["prediction_npz_sha256"] = runner.sha256_file(prediction)
    summary_path.write_text(json.dumps(original), encoding="utf-8")
    assert not runner.target_output_is_complete(root, "stories", "full_data", 1, cfg)

    np.savez_compressed(
        prediction,
        state=np.ones((cfg["prediction_n"] - 1, cfg["state_dim"]), dtype=np.float32),
    )
    original["prediction_npz_sha256"] = runner.sha256_file(prediction)
    summary_path.write_text(json.dumps(original), encoding="utf-8")
    assert not runner.target_output_is_complete(root, "stories", "full_data", 1, cfg)


def test_resume_binds_current_dataset_yaml_and_runtime_config(tmp_path, monkeypatch):
    source_cfg = runner.load_datasets(["admouse"])["admouse"]
    config_path, cfg = _install_current_test_config(monkeypatch, tmp_path, source_cfg)
    root = tmp_path / "admouse"
    _write_complete_prediction(root, "stories", "full_data", 1, cfg)
    assert runner.target_output_is_complete(root, "stories", "full_data", 1, cfg)

    recipe_fields = (
        "source_roster_seed",
        "time_key",
        "state_key",
        "expected_source_sha256",
        "preprocess_contract",
    )
    for field in recipe_fields:
        changed_cfg = deepcopy(cfg)
        _mutate_dataset_recipe(changed_cfg, field)
        assert (
            runner._current_prediction_contract(root, changed_cfg, "full_data", 1)
            is None
        ), field

    original_source = config_path.read_bytes()
    for field in recipe_fields:
        changed_source = yaml.safe_load(original_source.decode("utf-8"))
        _mutate_dataset_recipe(changed_source, field)
        config_path.write_text(
            yaml.safe_dump(changed_source, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        # Neither a caller still holding the old config nor one that reloads
        # the newly changed YAML may resume artifacts built from the old bytes.
        assert (
            runner._current_prediction_contract(root, cfg, "full_data", 1) is None
        ), field
        reloaded = runner.load_datasets(["admouse"])["admouse"]
        assert (
            runner._current_prediction_contract(root, reloaded, "full_data", 1) is None
        ), field
        config_path.write_bytes(original_source)
        assert (
            runner._current_prediction_contract(root, cfg, "full_data", 1) is not None
        )


def test_resume_rejects_config_manifest_and_resolved_artifact_tampering(
    tmp_path, monkeypatch
):
    source_cfg = runner.load_datasets(["admouse"])["admouse"]
    config_path, cfg = _install_current_test_config(monkeypatch, tmp_path, source_cfg)
    root = tmp_path / "admouse"
    _write_complete_prediction(root, "stories", "full_data", 1, cfg)
    manifest_path = root / "inputs/manifest.json"
    resolved_path = root / "inputs/resolved_config.yaml"
    original_manifest = manifest_path.read_bytes()
    original_resolved = resolved_path.read_bytes()
    assert runner._current_prediction_contract(root, cfg, "full_data", 1) is not None

    alternate_source = config_path.parent / "admouse-alternate.yaml"
    alternate_source.write_bytes(config_path.read_bytes())
    alternate_resolved = resolved_path.parent / "resolved_config-alternate.yaml"
    alternate_resolved.write_bytes(original_resolved)
    record_mutations = (
        (
            "config_source.path",
            lambda value: value["config_source"].update(
                path=str(alternate_source.resolve())
            ),
        ),
        (
            "config_source.sha256",
            lambda value: value["config_source"].update(sha256="f" * 64),
        ),
        (
            "resolved_config.path",
            lambda value: value["resolved_config"].update(
                path=str(alternate_resolved.resolve())
            ),
        ),
        (
            "resolved_config.relative_path",
            lambda value: value["resolved_config"].update(
                relative_path=alternate_resolved.name
            ),
        ),
        (
            "resolved_config.sha256",
            lambda value: value["resolved_config"].update(sha256="f" * 64),
        ),
        (
            "resolved_config.size_bytes",
            lambda value: value["resolved_config"].update(
                size_bytes=int(value["resolved_config"]["size_bytes"]) + 1
            ),
        ),
    )
    for label, mutate in record_mutations:
        payload = json.loads(original_manifest)
        mutate(payload)
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")
        assert (
            runner._current_prediction_contract(root, cfg, "full_data", 1) is None
        ), label
        manifest_path.write_bytes(original_manifest)

    resolved_path.write_bytes(original_resolved + b"\n# tampered\n")
    assert runner._current_prediction_contract(root, cfg, "full_data", 1) is None
    resolved_path.write_bytes(original_resolved)

    # This is the old-input/CLI-override attack: both the artifact bytes and
    # its manifest digest are internally consistent, but its resolved recipe
    # is not the current checked-in dataset YAML.
    for field in (
        "source_roster_seed",
        "time_key",
        "state_key",
        "expected_source_sha256",
        "preprocess_contract",
    ):
        resolved_payload = yaml.safe_load(original_resolved.decode("utf-8"))
        _mutate_dataset_recipe(resolved_payload, field)
        resolved_path.write_text(
            yaml.safe_dump(resolved_payload, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        manifest_payload = json.loads(original_manifest)
        manifest_payload["resolved_config"].update(
            sha256=runner.sha256_file(resolved_path),
            size_bytes=resolved_path.stat().st_size,
        )
        manifest_path.write_text(json.dumps(manifest_payload), encoding="utf-8")
        assert (
            runner._current_prediction_contract(root, cfg, "full_data", 1) is None
        ), field
        resolved_path.write_bytes(original_resolved)
        manifest_path.write_bytes(original_manifest)


def test_resume_validates_no_interaction_has_no_group_claim(tmp_path, monkeypatch):
    cfg = runner.load_datasets(["admouse"])["admouse"]
    cfg = dict(cfg)
    cfg["benchmark"] = dict(cfg["benchmark"], edge_prior_mode="none")
    _, cfg = _install_current_test_config(monkeypatch, tmp_path, cfg)
    root = tmp_path / "admouse"
    output = _write_complete_prediction(
        root,
        "cytobridge",
        "full_data",
        1,
        cfg,
        interaction_mode="none",
    )
    assert runner.target_output_is_complete(root, "cytobridge", "full_data", 1, cfg)

    summary_path = output / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["interaction_m"] = 1024
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    assert not runner.target_output_is_complete(root, "cytobridge", "full_data", 1, cfg)

    summary["interaction_m"] = None
    summary["simulation"]["interaction_grouping_seed"] = 10_042
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    assert not runner.target_output_is_complete(root, "cytobridge", "full_data", 1, cfg)

    summary["simulation"]["interaction_grouping_seed"] = None
    prediction = output / "prediction.npz"
    np.savez_compressed(
        prediction,
        state=np.ones((cfg["prediction_n"], cfg["state_dim"]), dtype=np.float32),
        spatial=np.ones(
            (cfg["prediction_n"], cfg["spatial_dim"] + 1), dtype=np.float32
        ),
        weights=np.ones(cfg["prediction_n"], dtype=np.float64),
    )
    summary["prediction_npz_sha256"] = runner.sha256_file(prediction)
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    assert not runner.target_output_is_complete(root, "cytobridge", "full_data", 1, cfg)


def test_resume_accepts_static_nested_record_and_stvcr_native_growth(tmp_path):
    cfg = runner.load_datasets(["admouse"])["admouse"]
    root = tmp_path / "admouse"

    static_output = _write_complete_prediction(
        root, "linear_centroid_shift", "full_data", 1, cfg
    )
    static_summary_path = static_output / "summary.json"
    static_summary = json.loads(static_summary_path.read_text(encoding="utf-8"))
    prediction = static_output / "prediction.npz"
    static_summary.pop("prediction_npz")
    static_summary.pop("prediction_npz_sha256")
    static_summary["prediction"] = {
        "path": str(prediction.resolve()),
        "sha256": runner.sha256_file(prediction),
    }
    static_summary_path.write_text(json.dumps(static_summary), encoding="utf-8")
    _write_static_run_manifest(root, "linear_centroid_shift", "full_data", 1, cfg)
    assert runner.target_output_is_complete(
        root, "linear_centroid_shift", "full_data", 1, cfg
    )

    stvcr_output = _write_complete_prediction(root, "stvcr", "full_data", 2, cfg)
    stvcr_summary_path = stvcr_output / "summary.json"
    stvcr_summary = json.loads(stvcr_summary_path.read_text(encoding="utf-8"))
    stvcr_prediction = stvcr_output / "prediction.npz"
    native_n = cfg["prediction_n"] - 1
    np.savez_compressed(
        stvcr_prediction,
        state=np.ones((native_n, cfg["state_dim"]), dtype=np.float32),
        spatial=np.ones((native_n, cfg["spatial_dim"]), dtype=np.float32),
        weights=np.full(native_n, 1.0 / cfg["prediction_n"], dtype=np.float64),
    )
    stvcr_summary.update(
        {
            "native_output_n": native_n,
            "native_growth": True,
            "native_count_changed": True,
            "weight_sum": native_n / cfg["prediction_n"],
            "growth_ratio": native_n / cfg["prediction_n"],
            "prediction_npz_sha256": runner.sha256_file(stvcr_prediction),
        }
    )
    stvcr_summary_path.write_text(json.dumps(stvcr_summary), encoding="utf-8")
    assert runner.target_output_is_complete(root, "stvcr", "full_data", 2, cfg)

    stvcr_summary["native_count_changed"] = False
    stvcr_summary_path.write_text(json.dumps(stvcr_summary), encoding="utf-8")
    assert not runner.target_output_is_complete(root, "stvcr", "full_data", 2, cfg)


def test_resume_requires_spatial_for_every_joint_static_scope(tmp_path):
    cfg = runner.load_datasets(["admouse"])["admouse"]
    root = tmp_path / "admouse"
    for method in (
        "moscot",
        "paste",
        "spateo",
        "linear_centroid_shift",
        "random_independent_pairs",
    ):
        output = _write_complete_prediction(root, method, "full_data", 1, cfg)
        prediction = output / "prediction.npz"
        np.savez_compressed(
            prediction,
            state=np.ones((cfg["prediction_n"], cfg["state_dim"]), dtype=np.float32),
        )
        summary_path = output / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["output_scope"] = "hybrid_joint"
        summary["prediction_npz_sha256"] = runner.sha256_file(prediction)
        summary_path.write_text(json.dumps(summary), encoding="utf-8")
        assert not runner.target_output_is_complete(
            root, method, "full_data", 1, cfg
        ), method


def test_static_resume_accepts_current_manifest_for_all_primary_adapters(tmp_path):
    cfg = runner.load_datasets(["admouse"])["admouse"]
    root = tmp_path / "admouse"
    for method in runner.STATIC:
        for track in ("loto", "full_data"):
            _write_complete_prediction(root, method, track, 1, cfg)
            assert runner.target_output_is_complete(root, method, track, 1, cfg), (
                method,
                track,
            )


def test_strict_resume_accepts_real_static_loto_producer_bytes(tmp_path):
    cfg = runner.load_datasets(["admouse"])["admouse"]
    root = tmp_path / "admouse"
    output = _run_real_static_loto_producer(root, cfg)
    produced = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))

    assert produced["protocol"]["time_values"] == [0.0, 2.0]
    assert produced["protocol"]["loto_target"] == 1.0
    assert produced["anchors"]["source_stage"] == 0.0
    assert (
        json.loads((output / "summary.json").read_text(encoding="utf-8"))["source_time"]
        == 0.0
    )
    assert runner.target_output_is_complete(
        root, "linear_centroid_shift", "loto", 1, cfg
    )


def test_static_resume_rejects_run_manifest_recipe_and_output_mutations(tmp_path):
    cfg = runner.load_datasets(["admouse"])["admouse"]
    root = tmp_path / "admouse"
    output = _write_complete_prediction(root, "moscot", "full_data", 1, cfg)
    manifest_path = output.parent / "run_manifest.json"
    original = json.loads(manifest_path.read_text(encoding="utf-8"))

    mutations = (
        (("status",), "initialized"),
        (("representation",), "native_gene_sensitivity"),
        (("seed",), 42),
        (("max_fit_n",), 801),
        (("parameters",), {}),
        (("method_spec",), {}),
        (("representation_spec",), {}),
        (("protocol", "mode"), "loto"),
        (("protocol", "time_values"), [-100, 50, 999]),
        (("input", "input_manifest_sha256"), "f" * 64),
        (("input", "train_h5ad"), str(tmp_path / "other.h5ad")),
        (("outputs", "prediction_by_time", "1.0", "sha256"), "e" * 64),
        (
            ("outputs", "prediction_by_time", "1.0", "summary", "path"),
            str(tmp_path / "other-summary.json"),
        ),
        (("official_runs", 0, "dependency", "git_commit"), "0" * 40),
        (("official_runs", 0, "dependency", "source_commit_verified"), False),
        (("official_runs", 0, "from"), -1000),
        (("official_runs", 0, "to"), 1000),
        (("official_runs", 0, "parameters"), {}),
    )
    for path, value in mutations:
        changed = json.loads(json.dumps(original))
        cursor = changed
        for token in path[:-1]:
            cursor = cursor[token]
        cursor[path[-1]] = value
        manifest_path.write_text(json.dumps(changed), encoding="utf-8")
        assert not runner.target_output_is_complete(
            root, "moscot", "full_data", 1, cfg
        ), path
    manifest_path.write_text(json.dumps(original), encoding="utf-8")
    assert runner.target_output_is_complete(root, "moscot", "full_data", 1, cfg)


def test_static_control_resume_requires_target_local_manifest_and_control_recipe(
    tmp_path,
):
    cfg = runner.load_datasets(["admouse"])["admouse"]
    root = tmp_path / "admouse"
    output = _write_complete_prediction(root, "linear_centroid_shift", "loto", 1, cfg)
    manifest_path = output / "run_manifest.json"
    original = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert runner.target_output_is_complete(
        root, "linear_centroid_shift", "loto", 1, cfg
    )

    changed = json.loads(json.dumps(original))
    changed["control_run"]["control"] = "random_independent_pairs"
    manifest_path.write_text(json.dumps(changed), encoding="utf-8")
    assert not runner.target_output_is_complete(
        root, "linear_centroid_shift", "loto", 1, cfg
    )

    changed = json.loads(json.dumps(original))
    changed["official_runs"] = [{}]
    manifest_path.write_text(json.dumps(changed), encoding="utf-8")
    assert not runner.target_output_is_complete(
        root, "linear_centroid_shift", "loto", 1, cfg
    )

    changed = json.loads(json.dumps(original))
    changed["protocol"]["time_values"] = [-100, 50, 999]
    manifest_path.write_text(json.dumps(changed), encoding="utf-8")
    assert not runner.target_output_is_complete(
        root, "linear_centroid_shift", "loto", 1, cfg
    )

    manifest_path.unlink()
    wrong_location = output.parent / "run_manifest.json"
    wrong_location.write_text(json.dumps(original), encoding="utf-8")
    assert not runner.target_output_is_complete(
        root, "linear_centroid_shift", "loto", 1, cfg
    )


def test_static_loto_resume_binds_nearest_observed_bracket(tmp_path):
    cfg = runner.load_datasets(["admouse"])["admouse"]
    root = tmp_path / "admouse"
    output = _write_complete_prediction(root, "moscot", "loto", 1, cfg)
    manifest_path = output / "run_manifest.json"
    original = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert original["protocol"]["time_values"] == [0.0, 2.0]
    assert [(run["from"], run["to"]) for run in original["official_runs"]] == [
        (0.0, 2.0)
    ]
    assert runner.target_output_is_complete(root, "moscot", "loto", 1, cfg)

    for key, value in (("from", -1), ("to", 3)):
        changed = json.loads(json.dumps(original))
        changed["official_runs"][0][key] = value
        manifest_path.write_text(json.dumps(changed), encoding="utf-8")
        assert not runner.target_output_is_complete(root, "moscot", "loto", 1, cfg), key


def test_resume_rejects_wrong_fold_mode_and_stale_dynamic_fit(tmp_path):
    cfg = runner.load_datasets(["zebrafish"])["zebrafish"]
    root = tmp_path / "zebrafish"
    output = _write_complete_prediction(root, "cytobridge", "loto", 1, cfg)
    summary_path = output / "summary.json"
    original = json.loads(summary_path.read_text(encoding="utf-8"))
    assert runner.target_output_is_complete(root, "cytobridge", "loto", 1, cfg)

    wrong_fold = dict(original)
    wrong_fold["split_id"] = "loto_t2"
    summary_path.write_text(json.dumps(wrong_fold), encoding="utf-8")
    assert not runner.target_output_is_complete(root, "cytobridge", "loto", 1, cfg)

    wrong_mode = dict(original)
    wrong_mode["interaction_mode"] = "all_spatial"
    wrong_mode["edge_prior_mode"] = "all_spatial"
    wrong_mode["edge_predictor_used"] = False
    wrong_mode["simulation"] = dict(original["simulation"])
    wrong_mode["simulation"].update(
        {
            "interaction_mode": "all_spatial",
            "edge_prior_mode": "all_spatial",
            "edge_predictor_used": False,
        }
    )
    summary_path.write_text(json.dumps(wrong_mode), encoding="utf-8")
    assert not runner.target_output_is_complete(root, "cytobridge", "loto", 1, cfg)

    dynamic = _write_complete_prediction(root, "stories", "loto", 2, cfg)
    assert runner.target_output_is_complete(root, "stories", "loto", 2, cfg)
    fit_manifest = root / "fits/stories/loto_t2/fit_manifest.json"
    fit_manifest.write_text('{"status":"refit"}', encoding="utf-8")
    assert not runner.target_output_is_complete(root, "stories", "loto", 2, cfg)


def test_dynamic_fit_resume_binds_dataset_regime_seed_and_adapter_bytes(tmp_path):
    cfg = runner.load_datasets(["admouse"])["admouse"]
    root = tmp_path / "admouse"
    input_manifest = _write_input_manifest(root, cfg)
    fit_dir = _write_complete_dynamic_fit(root, "stories", "loto_t1", cfg)
    manifest_path = fit_dir / "fit_manifest.json"
    summary_path = fit_dir / "summary.json"
    original = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert runner.dynamic_fit_is_complete(root, "stories", "loto_t1", input_manifest)
    bad_values = {
        "dataset": "other-dataset",
        "regime": "full_data",
        "fit_seed": original["fit_seed"] + 1,
        "source_worktree_clean": False,
        "adapter_implementation": {
            "schema_version": "1.0.0",
            "files": {},
            "aggregate_sha256": "0" * 64,
        },
    }
    for field, bad_value in bad_values.items():
        for missing in (False, True):
            changed = json.loads(json.dumps(original))
            if missing:
                changed.pop(field)
            else:
                changed[field] = bad_value
            text = json.dumps(changed)
            manifest_path.write_text(text, encoding="utf-8")
            summary_path.write_text(text, encoding="utf-8")
            assert not runner.dynamic_fit_is_complete(
                root, "stories", "loto_t1", input_manifest
            ), (field, missing)

    text = json.dumps(original)
    manifest_path.write_text(text, encoding="utf-8")
    summary_path.write_text(text, encoding="utf-8")
    assert runner.dynamic_fit_is_complete(root, "stories", "loto_t1", input_manifest)


def test_dynamic_prediction_resume_binds_current_adapter_bytes(tmp_path):
    cfg = runner.load_datasets(["admouse"])["admouse"]
    root = tmp_path / "admouse"
    output = _write_complete_prediction(root, "stories", "loto", 1, cfg)
    assert runner.target_output_is_complete(root, "stories", "loto", 1, cfg)

    summary_path = output / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["adapter_implementation"]["aggregate_sha256"] = "0" * 64
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    assert not runner.target_output_is_complete(root, "stories", "loto", 1, cfg)


def test_static_prediction_resume_binds_current_adapter_bytes(tmp_path):
    cfg = runner.load_datasets(["admouse"])["admouse"]
    root = tmp_path / "admouse"
    output = _write_complete_prediction(root, "linear_centroid_shift", "loto", 1, cfg)
    assert runner.target_output_is_complete(
        root, "linear_centroid_shift", "loto", 1, cfg
    )

    manifest_path = output / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["adapter_implementation"]["aggregate_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert not runner.target_output_is_complete(
        root, "linear_centroid_shift", "loto", 1, cfg
    )


def test_cytobridge_prediction_resume_binds_current_adapter_bytes(tmp_path):
    cfg = runner.load_datasets(["admouse"])["admouse"]
    root = tmp_path / "admouse"
    output = _write_complete_prediction(root, "cytobridge", "loto", 1, cfg)
    assert runner.target_output_is_complete(root, "cytobridge", "loto", 1, cfg)

    summary_path = output / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["adapter_implementation"]["aggregate_sha256"] = "0" * 64
    payload = json.dumps(summary)
    summary_path.write_text(payload, encoding="utf-8")
    (output / "prediction.summary.json").write_text(payload, encoding="utf-8")
    assert not runner.target_output_is_complete(root, "cytobridge", "loto", 1, cfg)


def test_resume_binds_current_formal_cytobridge_model_bytes(tmp_path):
    cfg = runner.load_datasets(["admouse"])["admouse"]
    root = tmp_path / "benchmark/admouse"
    formal = tmp_path / "formal/admouse"
    output = _write_complete_prediction(root, "cytobridge", "full_data", 1, cfg)
    model, contract = _write_formal_cytobridge_model(formal, root, cfg, "full_data")
    summary_path = output / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update(
        {
            "config_sha256": contract["saved_config_sha256"],
            "checkpoint_sha256": contract["checkpoint_sha256"],
            "training_reference_match": {
                "proof": "saved_adata_exact_frozen_arrays",
                "path": str(contract["saved_adata_path"]),
                "sha256": contract["saved_adata_sha256"],
            },
        }
    )
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    (output / "prediction.summary.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )
    assert runner.target_output_is_complete(
        root, "cytobridge", "full_data", 1, cfg, formal
    )

    (model / "adata.h5ad").write_bytes(b"mutated-adata")
    assert not runner.target_output_is_complete(
        root, "cytobridge", "full_data", 1, cfg, formal
    )
    (model / "adata.h5ad").write_bytes(b"frozen-adata")
    assert runner.target_output_is_complete(
        root, "cytobridge", "full_data", 1, cfg, formal
    )
    (model / "Finetune/best_model.pth").write_bytes(b"mutated-checkpoint")
    assert not runner.target_output_is_complete(
        root, "cytobridge", "full_data", 1, cfg, formal
    )


def test_resume_rejects_evaluator_schema_and_science_mutations(tmp_path):
    cfg = runner.load_datasets(["admouse"])["admouse"]
    root = tmp_path / "admouse"
    output = _write_complete_prediction(
        root, "linear_centroid_shift", "full_data", 1, cfg
    )
    summary_path = output / "summary.json"
    original = json.loads(summary_path.read_text(encoding="utf-8"))
    assert runner.target_output_is_complete(
        root, "linear_centroid_shift", "full_data", 1, cfg
    )
    mutations = (
        {"primary_benchmark_eligible": False},
        {"source_time": 999},
        {"output_scope": "native_state"},
        {"native_vs_adapter": "native_joint"},
        {"truth_artifact_opened": True},
        {"target_n_used_for_prediction": True},
        {"native_mass": True},
        {"native_growth": True},
        {"weights_are_unnormalised": True},
    )
    for mutation in mutations:
        changed = dict(original)
        changed.update(mutation)
        summary_path.write_text(json.dumps(changed), encoding="utf-8")
        assert not runner.target_output_is_complete(
            root, "linear_centroid_shift", "full_data", 1, cfg
        ), mutation

    cytobridge = _write_complete_prediction(root, "cytobridge", "full_data", 2, cfg)
    cb_summary_path = cytobridge / "summary.json"
    cb_original = json.loads(cb_summary_path.read_text(encoding="utf-8"))
    for mutation in (
        {"dt": 0.02},
        {"sigma": 0.05},
        {"alpha_express": 0.02},
        {"alpha_spatial": 5.0},
        {"include_score": False},
        {"interaction_m": 1},
    ):
        changed = dict(cb_original)
        changed.update(mutation)
        cb_summary_path.write_text(json.dumps(changed), encoding="utf-8")
        assert not runner.target_output_is_complete(
            root, "cytobridge", "full_data", 2, cfg
        ), mutation


def test_partial_shared_full_jobs_fail_closed_without_overwriting(
    tmp_path, monkeypatch
):
    cfg = runner.load_datasets(["admouse"])["admouse"]
    args = SimpleNamespace(
        run_root=tmp_path,
        formal_root=tmp_path / "formal",
        methods=["cytobridge", "linear_centroid_shift"],
        tracks=["full_data"],
        software_root=tmp_path / "software",
        device="cpu",
        timeout=1,
        dry_run=False,
    )
    root = tmp_path / "admouse"
    for method in ("cytobridge", "linear_centroid_shift"):
        _write_complete_prediction(root, method, "full_data", 1, cfg)

    monkeypatch.setattr(
        runner,
        "execute",
        lambda *unused: (_ for _ in ()).throw(AssertionError("must fail closed")),
    )
    runner.run_dataset("admouse", cfg, args, {}, {})

    with (root / "status/method_target_status.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    failed = [row for row in rows if row["target"] == "2"]
    assert {row["method"] for row in failed} == {
        "CytoBridge-0.015",
        "linear_centroid_shift",
    }
    assert {row["status"] for row in failed} == {"failed"}
    assert all("new --run-root" in row["reason"] for row in failed)


def test_evaluate_revalidates_every_current_prediction_contract(tmp_path, monkeypatch):
    cfg = runner.load_datasets(["admouse"])["admouse"]
    args = SimpleNamespace(
        run_root=tmp_path,
        formal_root=tmp_path / "formal",
        tracks=["full_data"],
        dry_run=False,
    )
    calls = []

    def fake_complete(
        root,
        method,
        track,
        target,
        config,
        formal,
        validation_cache=None,
    ):
        del root, config, formal, validation_cache
        calls.append((method, track, target))
        return method != "stories"

    monkeypatch.setattr(runner, "target_output_is_complete", fake_complete)
    monkeypatch.setattr(
        runner,
        "run_or_print",
        lambda *unused: (_ for _ in ()).throw(
            AssertionError("evaluation command must not run")
        ),
    )

    with pytest.raises(RuntimeError, match="stories:t1"):
        runner.evaluate("admouse", cfg, args)
    assert len(calls) == len(runner.PRIMARY_METHODS) * len(cfg["full_data_targets"])


def test_evaluate_allows_only_explicit_non_numeric_status_for_missing_predictions(
    tmp_path, monkeypatch
):
    cfg = runner.load_datasets(["admouse"])["admouse"]
    args = SimpleNamespace(
        run_root=tmp_path,
        formal_root=tmp_path / "formal",
        tracks=["full_data"],
        dry_run=False,
    )
    status_path = tmp_path / "admouse/status/method_target_status.csv"
    runner.merge_status_rows(
        status_path,
        [
            {
                "track": "full_data",
                "target": target,
                "method": "stories",
                "status": "timeout",
                "reason": "declared test budget",
                "elapsed_seconds": 3600.0,
            }
            for target in cfg["full_data_targets"]
        ],
    )
    calls = []

    def fake_complete(
        root,
        method,
        track,
        target,
        config,
        formal,
        validation_cache=None,
    ):
        del root, track, target, config, formal, validation_cache
        return method != "stories"

    monkeypatch.setattr(runner, "target_output_is_complete", fake_complete)
    monkeypatch.setattr(
        runner,
        "run_or_print",
        lambda commands, dry_run: calls.append((commands, dry_run)),
    )

    runner.evaluate("admouse", cfg, args)

    assert len(calls) == 1
    commands, dry_run = calls[0]
    assert dry_run is False
    assert len(commands) == 2
    assert str(status_path) in commands[0]


def test_evaluate_omits_an_absent_optional_status_table(tmp_path, monkeypatch):
    cfg = runner.load_datasets(["admouse"])["admouse"]
    args = SimpleNamespace(
        run_root=tmp_path,
        formal_root=tmp_path / "formal",
        tracks=["full_data"],
        dry_run=False,
    )
    calls = []
    monkeypatch.setattr(
        runner, "target_output_is_complete", lambda *args, **kwargs: True
    )
    monkeypatch.setattr(
        runner,
        "run_or_print",
        lambda commands, dry_run: calls.append((commands, dry_run)),
    )

    runner.evaluate("admouse", cfg, args)

    assert len(calls) == 1
    assert "--status-table" not in calls[0][0][0]
    command = calls[0][0][0]
    prediction_root = Path(command[command.index("--predictions-root") + 1])
    assert prediction_root == tmp_path / "admouse/predictions/full_data"


def test_evaluate_scopes_prediction_scan_to_the_requested_track(tmp_path, monkeypatch):
    cfg = runner.load_datasets(["admouse"])["admouse"]
    args = SimpleNamespace(
        run_root=tmp_path,
        formal_root=tmp_path / "formal",
        tracks=["loto"],
        dry_run=False,
    )
    malformed_off_track = (
        tmp_path / "admouse/predictions/full_data/stories/t1/prediction.npz"
    )
    malformed_off_track.parent.mkdir(parents=True)
    malformed_off_track.write_bytes(b"partial off-track artifact")
    runner.merge_status_rows(
        tmp_path / "admouse/status/method_target_status.csv",
        [
            {
                "track": "loto",
                "target": target,
                "method": "stories",
                "status": "timeout",
                "reason": "declared test budget",
                "elapsed_seconds": 3600.0,
            }
            for target in cfg["loto_targets"]
        ],
    )
    calls = []

    def fake_complete(
        root,
        method,
        track,
        target,
        config,
        formal,
        validation_cache=None,
    ):
        del root, track, target, config, formal, validation_cache
        return method != "stories"

    monkeypatch.setattr(runner, "target_output_is_complete", fake_complete)
    monkeypatch.setattr(
        runner,
        "run_or_print",
        lambda commands, dry_run: calls.append((commands, dry_run)),
    )

    runner.evaluate("admouse", cfg, args)

    command = calls[0][0][0]
    prediction_root = Path(command[command.index("--predictions-root") + 1])
    assert prediction_root == tmp_path / "admouse/predictions/loto"
    assert malformed_off_track.is_file()


def test_evaluate_rejects_prediction_that_contradicts_failure_status(
    tmp_path, monkeypatch
):
    cfg = runner.load_datasets(["admouse"])["admouse"]
    args = SimpleNamespace(
        run_root=tmp_path,
        formal_root=tmp_path / "formal",
        tracks=["full_data"],
        dry_run=False,
    )
    runner.merge_status_rows(
        tmp_path / "admouse/status/method_target_status.csv",
        [
            {
                "track": "full_data",
                "target": cfg["full_data_targets"][0],
                "method": "stories",
                "status": "oom",
                "reason": "declared failure",
                "elapsed_seconds": 1.0,
            }
        ],
    )
    monkeypatch.setattr(
        runner, "target_output_is_complete", lambda *args, **kwargs: True
    )

    with pytest.raises(RuntimeError, match="contradict.*status=oom"):
        runner.evaluate("admouse", cfg, args)


@pytest.mark.parametrize(
    ("track_cell", "target_cell", "status_cell", "expected"),
    [
        ("full-data", "1.0", "out-of-memory", "oom"),
        ("no-holdout", "01", "unavailable", "not_available"),
        ("full data", "1", "not available", "not_available"),
        ("full_data", "1", "N/A", "not_applicable"),
    ],
)
def test_evaluation_status_preflight_matches_evaluator_aliases(
    tmp_path, track_cell, target_cell, status_cell, expected
):
    path = tmp_path / "status.csv"
    path.write_text(
        "track,target,method,status,reason\n"
        f"{track_cell},{target_cell},stories,{status_cell},declared\n",
        encoding="utf-8",
    )

    expected_rows = {("stories", 1): expected}
    assert runner._evaluation_status_rows(path, track="full_data") == expected_rows
    assert {
        key: row["status"]
        for key, row in evaluator._load_status_table(path, "full_data").items()
    } == expected_rows


def test_evaluation_status_preflight_accepts_optional_track_column(tmp_path):
    path = tmp_path / "status.csv"
    path.write_text(
        "target,method,status,reason\n1,stories,timeout,declared\n",
        encoding="utf-8",
    )

    expected_rows = {("stories", 1): "timeout"}
    assert runner._evaluation_status_rows(path, track="full_data") == expected_rows
    assert {
        key: row["status"]
        for key, row in evaluator._load_status_table(path, "full_data").items()
    } == expected_rows


def test_evaluation_status_parsers_both_reject_duplicate_headers(tmp_path):
    path = tmp_path / "status.csv"
    path.write_text(
        "track,target,method,status,status\n" "full_data,1,stories,timeout,completed\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="duplicate columns.*status"):
        runner._evaluation_status_rows(path, track="full_data")
    with pytest.raises(evaluator.ContractError, match="duplicate columns.*status"):
        evaluator._load_status_table(path, "full_data")


def test_evaluation_status_parsers_both_reject_ragged_rows(tmp_path):
    path = tmp_path / "status.csv"
    path.write_text(
        "track,target,method,status\n" "full_data,1,stories,timeout,EXTRA\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="different number of fields"):
        runner._evaluation_status_rows(path, track="full_data")
    with pytest.raises(evaluator.ContractError, match="different number of fields"):
        evaluator._load_status_table(path, "full_data")


def test_unverified_dynamic_fit_fails_closed(tmp_path, monkeypatch):
    cfg = runner.load_datasets(["admouse"])["admouse"]
    args = SimpleNamespace(
        run_root=tmp_path,
        formal_root=tmp_path / "formal",
        methods=["stories"],
        tracks=["full_data"],
        software_root=tmp_path / "software",
        device="cpu",
        timeout=1,
        dry_run=False,
    )
    fit = tmp_path / "admouse/fits/stories/full_data"
    fit.mkdir(parents=True)
    (fit / "partial.bin").write_bytes(b"partial")
    monkeypatch.setattr(
        runner,
        "execute",
        lambda *unused: (_ for _ in ()).throw(AssertionError("must fail closed")),
    )

    runner.run_dataset("admouse", cfg, args, {}, {"stories": tmp_path})

    with (tmp_path / "admouse/status/method_target_status.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert {row["status"] for row in rows} == {"failed"}
    assert all("unverified" in row["reason"] for row in rows)


def test_static_loto_resumes_later_targets_after_an_earlier_target(
    tmp_path, monkeypatch
):
    cfg = runner.load_datasets(["zebrafish"])["zebrafish"]
    args = SimpleNamespace(
        run_root=tmp_path,
        formal_root=tmp_path / "formal",
        methods=["linear_centroid_shift"],
        tracks=["loto"],
        software_root=tmp_path / "software",
        device="cpu",
        timeout=1,
        dry_run=False,
    )
    root = tmp_path / "zebrafish"
    _write_complete_prediction(root, "linear_centroid_shift", "loto", 1, cfg)
    seen_targets = []

    def execute(commands, *unused):
        command = commands[0]
        seen_targets.append(command[command.index("--target-time") + 1])
        return "timeout", "test timeout"

    monkeypatch.setattr(runner, "execute", execute)
    runner.run_dataset("zebrafish", cfg, args, {}, {})

    assert seen_targets == ["2", "3"]
    with (root / "status/method_target_status.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert [(row["target"], row["status"]) for row in rows] == [
        ("1", "completed"),
        ("2", "timeout"),
        ("3", "timeout"),
    ]


def test_cytobridge_loto_checks_only_the_current_split_artifacts(tmp_path):
    cfg = runner.load_datasets(["zebrafish"])["zebrafish"]
    args = SimpleNamespace(
        run_root=tmp_path,
        formal_root=tmp_path / "formal",
        methods=["cytobridge"],
        tracks=["loto"],
        software_root=tmp_path / "software",
        device="cpu",
    )
    root = tmp_path / "zebrafish"
    _write_complete_prediction(root, "cytobridge", "loto", 1, cfg)
    previous_graph = root / "graphs/loto_t1"
    previous_graph.mkdir(parents=True)
    (previous_graph / "complete.bin").write_bytes(b"previous fold")
    jobs = runner.jobs_for_dataset("zebrafish", cfg, args, {}, {})
    target_two = next(
        job
        for job in jobs
        if job[0] == "cytobridge" and job[1] == "loto" and job[2] == [2]
    )

    runnable, reason = runner.commands_for_safe_resume(
        root, target_two[0], target_two[1], target_two[2], target_two[3], cfg
    )

    assert reason is None
    assert runnable == target_two[3]


def test_status_updates_merge_instead_of_erasing_other_methods(tmp_path):
    path = tmp_path / "status.csv"
    runner.merge_status_rows(
        path,
        [
            {
                "track": "loto",
                "target": 1,
                "method": "stvcr",
                "status": "completed",
                "reason": "",
                "elapsed_seconds": 2.0,
            }
        ],
    )
    runner.merge_status_rows(
        path,
        [
            {
                "track": "loto",
                "target": 1,
                "method": "stories",
                "status": "timeout",
                "reason": "budget",
                "elapsed_seconds": 3600.0,
            }
        ],
    )
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert {(row["method"], row["status"]) for row in rows} == {
        ("stvcr", "completed"),
        ("stories", "timeout"),
    }


def test_parallel_status_updates_do_not_drop_methods(tmp_path):
    path = tmp_path / "status.csv"
    processes = [
        multiprocessing.Process(target=_merge_one_status, args=(path, method))
        for method in ("stvcr", "stories", "mioflow", "moscot")
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=10)
        assert process.exitcode == 0
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["method"] for row in rows} == {
        "stvcr",
        "stories",
        "mioflow",
        "moscot",
    }
