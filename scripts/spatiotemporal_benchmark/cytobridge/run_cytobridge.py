#!/usr/bin/env python3
"""Leakage-audited CytoBridge-0.015 benchmark adapter.

The CLI supports input/config preflight, training-only LOTO graph and edge-
classifier rebuilding, one exact six-stage fit per LOTO fold, validation of a
reused full-data checkpoint, and continuous non-split weighted SDE inference.
It never opens benchmark truth artifacts.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import inspect
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from common import (  # type: ignore
        ALPHA_EXPRESS,
        ALPHA_SPATIAL,
        METHOD,
        PREDICTION_N,
        SEED,
        SIGMA,
        ContractError,
        SplitInput,
        TrainingData,
        bootstrap_indices,
        checkpoint_inventory,
        checkpoint_training_match,
        environment_provenance,
        input_provenance,
        load_training_data,
        load_yaml,
        new_output_dir,
        plain,
        read_split_input,
        repo_identity,
        same_time,
        sha256_array,
        sha256_file,
        source_time,
        validate_training_config,
        write_json,
    )
else:
    from .common import (
        ALPHA_EXPRESS,
        ALPHA_SPATIAL,
        METHOD,
        PREDICTION_N,
        SEED,
        SIGMA,
        ContractError,
        SplitInput,
        TrainingData,
        bootstrap_indices,
        checkpoint_inventory,
        checkpoint_training_match,
        environment_provenance,
        input_provenance,
        load_training_data,
        load_yaml,
        new_output_dir,
        plain,
        read_split_input,
        repo_identity,
        same_time,
        sha256_array,
        sha256_file,
        source_time,
        validate_training_config,
        write_json,
    )


REPO_ROOT = Path(__file__).resolve().parents[3]


def _slug(value: str) -> str:
    result = re.sub(r"[^a-zA-Z0-9]+", "_", str(value)).strip("_").lower()
    if not result:
        raise ContractError("dataset_id cannot be converted to a graph-name slug")
    return result


def _time_suffix(value: float) -> str:
    rounded = int(round(value))
    return str(rounded) if same_time(value, rounded) else str(float(value))


def _target_json(value: float) -> int | float:
    rounded = int(round(value))
    return rounded if same_time(value, rounded) else float(value)


def _require_loto(split: SplitInput) -> None:
    if split.regime != "loto" or split.holdout_time is None:
        raise ContractError(f"command requires a LOTO split, found {split.split_id!r}")


def _require_full(split: SplitInput) -> None:
    if split.regime != "full_data":
        raise ContractError(f"command requires full_data, found {split.split_id!r}")


def inference_schedule(split: SplitInput) -> tuple[float, tuple[float, ...]]:
    """Return the auditable non-split integration schedule for a split."""
    source = source_time(split)
    if split.regime == "full_data":
        targets = tuple(sorted(float(value) for value in split.evaluation_targets))
        return source, (source, *targets)
    _require_loto(split)
    return source, (source, float(split.holdout_time))


def ordered_graph_plan(split: SplitInput) -> tuple[tuple[int, float, str], ...]:
    """Map fold-order graph names to actual observed benchmark times."""
    slug = _slug(split.dataset_id)
    return tuple(
        (order, float(actual), f"{slug}_t{order}")
        for order, actual in enumerate(split.observed_times)
    )


def _validate_runtime_constants(args: argparse.Namespace) -> None:
    if hasattr(args, "seed") and int(args.seed) != SEED:
        raise ContractError(f"locked CytoBridge benchmark seed must be {SEED}")
    if hasattr(args, "sigma") and not np.isclose(
        float(args.sigma), SIGMA, rtol=0.0, atol=1e-12
    ):
        raise ContractError(f"locked CytoBridge benchmark sigma must be {SIGMA}")
    if hasattr(args, "prediction_n") and int(args.prediction_n) != PREDICTION_N:
        raise ContractError(f"prediction_n must be {PREDICTION_N}")


def _input_report(split: SplitInput, data: TrainingData) -> dict[str, Any]:
    if data.spatial_dim != 2:
        raise ContractError(
            "the current CytoBridge GNN interaction implementation requires 2D spatial "
            f"coordinates, found {data.spatial_dim}D"
        )
    counts = {
        str(_target_json(value)): int(
            np.count_nonzero(np.isclose(data.time, value, rtol=0.0, atol=1e-8))
        )
        for value in split.observed_times
    }
    return {
        "dataset_id": split.dataset_id,
        "split_id": split.split_id,
        "regime": split.regime,
        "n_train": data.n_obs,
        "state_dim": data.state_dim,
        "spatial_dim": data.spatial_dim,
        "joint_dim": data.joint_dim,
        "observed_times": list(split.observed_times),
        "train_time_counts": counts,
        "evaluation_targets": list(split.evaluation_targets),
        "held_out_time": split.holdout_time,
        "target_physically_absent": split.regime != "loto" or not np.any(
            np.isclose(data.time, split.holdout_time, rtol=0.0, atol=1e-8)
        ),
        "prediction_n": split.prediction_n,
        "prediction_n_policy": "fixed_from_train_contract_before_truth_access",
        "transductive_frozen_representation": True,
        **input_provenance(split),
    }


def _probe_model_load(
    repo: Path, model_dir: Path, data: TrainingData, device: str
) -> dict[str, Any]:
    if str(repo.resolve()) not in sys.path:
        sys.path.insert(0, str(repo.resolve()))
    from CytoBridge.tl import load_dynamical_model_from_dir

    with _working_directory(repo.resolve()):
        loaded = load_dynamical_model_from_dir(
            model_dir, dim=data.joint_dim, device=str(device)
        )
    validate_training_config(loaded.config)
    if loaded.weight_stage != "Finetune" or loaded.score_stage != "Score_Refine":
        raise ContractError(
            "checkpoint loader must resolve Finetune weights and Score_Refine score; "
            f"found {loaded.weight_stage!r}/{loaded.score_stage!r}"
        )
    return {
        "loadable": True,
        "joint_dim": data.joint_dim,
        "weight_stage": loaded.weight_stage,
        "weight_path": str(loaded.weight_path),
        "weight_sha256": sha256_file(loaded.weight_path),
        "score_stage": loaded.score_stage,
        "score_path": str(loaded.score_path),
        "score_sha256": sha256_file(loaded.score_path),
        "device": str(device),
    }


def _model_report(
    model_dir: Path,
    split: SplitInput,
    data: TrainingData,
    *,
    repo: Path | None = None,
    device: str = "cpu",
    probe_load: bool = False,
) -> dict[str, Any]:
    inventory = checkpoint_inventory(model_dir)
    match = checkpoint_training_match(model_dir, split, data, inventory=inventory)
    report = {**inventory, "training_reference_match": match}
    if probe_load:
        if repo is None:
            raise ValueError("repo is required when probe_load=True")
        report["model_load_probe"] = _probe_model_load(repo, model_dir, data, device)
    return report


def command_preflight(args: argparse.Namespace) -> None:
    _validate_runtime_constants(args)
    split = read_split_input(args.input_manifest, args.split)
    data = load_training_data(split)
    config_path = args.training_config.expanduser().resolve()
    config = validate_training_config(load_yaml(config_path))
    report: dict[str, Any] = {
        "status": "ok",
        "phase": "preflight",
        "method": METHOD,
        "input": _input_report(split, data),
        "training_config_source": str(config_path),
        "training_config_source_sha256": sha256_file(config_path),
        "training_profile": config,
        "locked_parameters": {
            "alpha_express": ALPHA_EXPRESS,
            "alpha_spatial": ALPHA_SPATIAL,
            "sigma": SIGMA,
            "seed": SEED,
        },
        "repo": repo_identity(args.repo),
        "environment": environment_provenance(args.device),
    }
    if args.model_dir is not None:
        report["model"] = _model_report(
            args.model_dir,
            split,
            data,
            repo=args.repo,
            device=args.device,
            probe_load=True,
        )
    if args.output_json is not None:
        write_json(args.output_json, report)
    print(json.dumps(plain(report), indent=2, sort_keys=True))


def _resolve_interaction_cutoff(adata: Any, requested: float | None) -> tuple[float, str]:
    if requested is not None:
        value = float(requested)
        if not np.isfinite(value) or value <= 0:
            raise ContractError("--interaction-cutoff must be finite and positive")
        return value, "explicit_cli"
    graph = adata.uns.get("interaction_graph", {})
    if isinstance(graph, Mapping):
        value = graph.get("neighborhood_threshold")
        if value is not None and np.isfinite(float(value)) and float(value) > 0:
            return float(value), "train_h5ad.uns.interaction_graph.neighborhood_threshold"
    raise ContractError(
        "interaction cutoff is absent from preprocessing; pass --interaction-cutoff"
    )


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "prepare_graph_summary.json"
    }


def command_prepare_loto(args: argparse.Namespace) -> None:
    _validate_runtime_constants(args)
    split = read_split_input(args.input_manifest, args.split)
    _require_loto(split)
    data = load_training_data(split)
    input_report = _input_report(split, data)
    output = new_output_dir(args.output_dir)
    database = args.database.expanduser().resolve()
    if not database.is_file():
        raise FileNotFoundError(database)
    if str(args.repo.resolve()) not in sys.path:
        sys.path.insert(0, str(args.repo.resolve()))

    import scanpy as sc
    from CytoBridge.pp.edge_prediction import train_edge_predictor
    from CytoBridge.pp.interaction_graph import generate_interaction_graph

    adata = sc.read_h5ad(split.train_h5ad)
    try:
        cutoff, cutoff_source = _resolve_interaction_cutoff(
            adata, args.interaction_cutoff
        )
        if args.expression_layer is not None and args.expression_layer not in adata.layers:
            raise ContractError(
                f"LOTO graph requires layers[{args.expression_layer!r}], but it is absent"
            )
        spot_diameter = (
            float(args.spot_diameter)
            if args.spot_diameter is not None
            else float(cutoff / 4.0)
        )
        if not np.isfinite(spot_diameter) or spot_diameter <= 0:
            raise ContractError("spot diameter must be finite and positive")
        graph_root = output / "interaction_graphs"
        metadata_root = output / "metadata"
        slug = _slug(split.dataset_id)
        graph_results: list[dict[str, Any]] = []
        # Edge-classifier loading uses observed-stage order (t0, t1, ...).
        # Naming by actual time would collide with that lookup after a LOTO gap
        # (e.g. fold-order t2 could accidentally open actual-time t2).  Keep an
        # explicit actual-time mapping in provenance while writing ordered names.
        for order, actual_time, stage_name in ordered_graph_plan(split):
            result = generate_interaction_graph(
                data_name=stage_name,
                data_from=adata,
                data_to=str(graph_root / stage_name),
                metadata_to=str(metadata_root / stage_name),
                database_path=str(database),
                split=0,
                time_key=data.time_key,
                time_value=actual_time,
                neighborhood_threshold=cutoff,
                spot_diameter=spot_diameter,
                spatial_key=data.spatial_key,
                expression_layer=args.expression_layer,
                auto_neighborhood_threshold=False,
                save_metadata=False,
                verbose=not args.quiet,
                use_tqdm=not args.quiet,
            )
            graph_results.append(
                {
                    "ordered_fold_index": order,
                    "ordered_graph_name": stage_name,
                    "actual_benchmark_time": actual_time,
                    **plain(result),
                }
            )

        edge_path = output / "edge_classifier" / f"{slug}.pt"
        edge_result = train_edge_predictor(
            data_name=slug,
            adata_or_h5ad=adata,
            graph_input_dir=str(graph_root),
            output_model_path=str(edge_path),
            epochs=int(args.edge_epochs),
            batch_size=int(args.edge_batch_size),
            learning_rate=float(args.edge_learning_rate),
            spatial_dim=data.spatial_dim,
            distance_threshold=cutoff,
            device=str(args.device),
            time_key=data.time_key,
            latent_key=data.state_key,
            spatial_key=data.spatial_key,
            train_sample_ratio_per_epoch=float(args.edge_train_sample_ratio),
            max_train_edges_per_epoch=args.edge_max_train_edges,
            num_workers=int(args.edge_num_workers),
            random_seed=SEED,
            edge_predictor_threshold=args.edge_threshold,
        )
    finally:
        del adata

    edge_meta = Path(str(edge_result["meta_path"])).resolve()
    effective_threshold = float(edge_result["edge_predictor_threshold"])
    if not 0 < effective_threshold < 1:
        raise ContractError("trained edge classifier returned an invalid threshold")
    report = {
        "status": "complete",
        "phase": "prepare_loto_graph_and_edge_classifier",
        "method": METHOD,
        "split_id": split.split_id,
        "regime": "loto",
        "target": _target_json(float(split.holdout_time)),
        "input": input_report,
        **input_provenance(split),
        "training_only_recomputation": True,
        "held_out_graph_opened": False,
        "interaction_cutoff": cutoff,
        "interaction_cutoff_source": cutoff_source,
        "spot_diameter": spot_diameter,
        "expression_layer": args.expression_layer,
        "observed_stage_graphs": graph_results,
        "database": str(database),
        "database_sha256": sha256_file(database),
        "edge_epochs": int(args.edge_epochs),
        "edge_model": str(edge_path.resolve()),
        "edge_model_sha256": sha256_file(edge_path),
        "edge_meta": str(edge_meta),
        "edge_meta_sha256": sha256_file(edge_meta),
        "edge_threshold": effective_threshold,
        "edge_threshold_selected_on_validation": float(
            edge_result["edge_predictor_threshold_selected"]
        ),
        "edge_threshold_source": (
            "validation_selected" if args.edge_threshold is None else "explicit_cli"
        ),
        "seed": SEED,
        "device": str(args.device),
        "environment": environment_provenance(args.device),
        "repo": repo_identity(args.repo),
    }
    report["artifact_sha256"] = _tree_hashes(output)
    write_json(output / "prepare_graph_summary.json", report)
    print(json.dumps(plain(report), indent=2, sort_keys=True))


def _load_graph_summary(path: Path, split: SplitInput) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    summary_path = path / "prepare_graph_summary.json" if path.is_dir() else path
    try:
        report = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read graph summary {summary_path}: {exc}") from exc
    if str(report.get("status")) != "complete":
        raise ContractError("graph summary is not complete")
    if str(report.get("split_id")) != split.split_id:
        raise ContractError("graph summary split_id mismatch")
    if str(report.get("input_manifest_sha256")) != split.root_manifest_sha256:
        raise ContractError("graph summary input-manifest SHA mismatch")
    if str(report.get("training_reference_sha256")) != split.training_reference_sha256:
        raise ContractError("graph summary training-reference SHA mismatch")
    if report.get("training_only_recomputation") is not True:
        raise ContractError("graph summary does not prove training-only recomputation")
    declared_artifacts = report.get("artifact_sha256")
    if not isinstance(declared_artifacts, Mapping) or not declared_artifacts:
        raise ContractError("graph summary lacks immutable artifact hashes")
    for relative, expected in declared_artifacts.items():
        artifact = (summary_path.parent / str(relative)).resolve()
        if not artifact.is_file() or sha256_file(artifact) != str(expected):
            raise ContractError(f"graph/edge artifact changed or disappeared: {artifact}")
    edge = Path(str(report.get("edge_model", ""))).expanduser().resolve()
    meta = Path(str(report.get("edge_meta", ""))).expanduser().resolve()
    for artifact, key in ((edge, "edge_model_sha256"), (meta, "edge_meta_sha256")):
        if not artifact.is_file() or sha256_file(artifact) != str(report.get(key, "")):
            raise ContractError(f"graph artifact changed or disappeared: {artifact}")
    report["_summary_path"] = str(summary_path)
    report["_summary_sha256"] = sha256_file(summary_path)
    return report


def command_fit_loto(args: argparse.Namespace) -> None:
    _validate_runtime_constants(args)
    split = read_split_input(args.input_manifest, args.split)
    _require_loto(split)
    data = load_training_data(split)
    _input_report(split, data)
    config_source = args.training_config.expanduser().resolve()
    config = copy.deepcopy(load_yaml(config_source))
    config_profile = validate_training_config(config)
    graph = _load_graph_summary(args.graph_dir, split)
    output = new_output_dir(args.output_dir)
    edge_path = Path(str(graph["edge_model"])).resolve()
    cutoff = float(graph["interaction_cutoff"])
    threshold = float(graph["edge_threshold"])
    config["ckpt_dir"] = str(output)
    config["model"]["spatial_dim"] = data.spatial_dim
    config["model"]["interaction_net"]["cutoff"] = cutoff
    config["model"]["interaction_net"]["edge_predictor_path"] = str(edge_path)
    config["model"]["interaction_net"]["edge_predictor_thre"] = threshold
    validate_training_config(config)
    if str(args.repo.resolve()) not in sys.path:
        sys.path.insert(0, str(args.repo.resolve()))
    import CytoBridge as cb

    cb.tl.fit(
        str(split.train_h5ad),
        config=config,
        device=str(args.device),
        time_key=data.time_key,
        obsm_key=data.state_key,
        spatial_key=data.spatial_key,
        is_spatial=True,
        ckpt_dir=str(output),
        interaction_cutoff=cutoff,
        edge_predictor_path=str(edge_path),
        edge_predictor_threshold=threshold,
        sigma=SIGMA,
        evaluate_after_training=False,
    )
    inventory = checkpoint_inventory(output)
    # At this point the fit summary does not yet exist; verification therefore
    # uses the saved adata's exact frozen arrays.
    match = checkpoint_training_match(output, split, data, inventory=inventory)
    report = {
        "status": "complete",
        "phase": "fit_loto",
        "method": METHOD,
        "split_id": split.split_id,
        "regime": "loto",
        "target": _target_json(float(split.holdout_time)),
        "target_rows_physically_absent": True,
        "six_stage_fit_from_scratch": True,
        "alpha_express": ALPHA_EXPRESS,
        "alpha_spatial": ALPHA_SPATIAL,
        "sigma": SIGMA,
        "seed": SEED,
        "device": str(args.device),
        "training_profile": config_profile,
        "training_config_source": str(config_source),
        "training_config_source_sha256": sha256_file(config_source),
        "saved_config_sha256": inventory["config_sha256"],
        "stage_complete": inventory["stage_complete"],
        "checkpoint_sha256": {
            name: artifact["sha256"]
            for name, artifact in inventory["checkpoints"].items()
        },
        "checkpoint_inventory": inventory,
        "training_reference_match": match,
        "interaction_cutoff": cutoff,
        "edge_threshold": threshold,
        "edge_model": str(edge_path),
        "edge_model_sha256": sha256_file(edge_path),
        "prepare_graph_summary": graph["_summary_path"],
        "prepare_graph_summary_sha256": graph["_summary_sha256"],
        "environment": environment_provenance(args.device),
        "repo": repo_identity(args.repo),
        **input_provenance(split),
    }
    write_json(output / "benchmark_fit_summary.json", report)
    print(json.dumps(plain(report), indent=2, sort_keys=True))


def command_validate_model(args: argparse.Namespace) -> None:
    split = read_split_input(args.input_manifest, args.split)
    data = load_training_data(split)
    report = {
        "status": "complete",
        "phase": "validate_model",
        "method": METHOD,
        "split_id": split.split_id,
        "regime": split.regime,
        "input": _input_report(split, data),
        "model": _model_report(
            args.model_dir,
            split,
            data,
            repo=args.repo,
            device=args.device,
            probe_load=True,
        ),
        "repo": repo_identity(args.repo),
        "environment": environment_provenance(args.device),
        **input_provenance(split),
    }
    if args.output_json is not None:
        write_json(args.output_json, report)
    print(json.dumps(plain(report), indent=2, sort_keys=True))


def _source_roster(
    output: Path, split: SplitInput, data: TrainingData, source: float
) -> tuple[np.ndarray, dict[str, Any]]:
    if sha256_file(split.source_roster_npz) != split.source_roster_sha256:
        raise ContractError("canonical source roster SHA-256 differs from manifest")
    with np.load(split.source_roster_npz, allow_pickle=False) as archive:
        required = {"indices", "row_id", "source_time", "state", "spatial"}
        missing = sorted(required.difference(archive.files))
        if missing:
            raise ContractError(f"canonical source roster lacks keys {missing}")
        indices = np.asarray(archive["indices"], dtype=np.int64)
        row_id = np.asarray(archive["row_id"]).astype(str)
        roster_source = float(np.asarray(archive["source_time"]).reshape(-1)[0])
        state = np.asarray(archive["state"], dtype=np.float32)
        spatial = np.asarray(archive["spatial"], dtype=np.float32)
    if (
        indices.shape != (PREDICTION_N,)
        or row_id.shape != (PREDICTION_N,)
        or state.shape != (PREDICTION_N, data.state_dim)
        or spatial.shape != (PREDICTION_N, data.spatial_dim)
        or np.any(indices < 0)
        or np.any(indices >= data.n_obs)
        or not same_time(roster_source, source)
    ):
        raise ContractError("canonical source roster has invalid shape, indices, or time")
    if not np.array_equal(row_id, data.row_id[indices]):
        raise ContractError("canonical source roster row IDs differ from training rows")
    if not np.all(np.isclose(data.time[indices], source, rtol=0.0, atol=1e-8)):
        raise ContractError("canonical source roster includes rows outside source time")
    if not np.allclose(state, data.state[indices], rtol=1e-6, atol=1e-6):
        raise ContractError("canonical source roster state differs from training rows")
    if not np.allclose(spatial, data.spatial[indices], rtol=1e-6, atol=1e-6):
        raise ContractError("canonical source roster spatial differs from training rows")
    path = output / "source_roster.npz"
    shutil.copy2(split.source_roster_npz, path)
    available = int(
        np.count_nonzero(np.isclose(data.time, source, rtol=0.0, atol=1e-8))
    )
    return indices, {
        "source_roster": str(path),
        "source_roster_sha256": sha256_file(path),
        "source_indices_sha256": sha256_array(indices),
        "source_row_id_sha256": sha256_array(data.row_id[indices].astype("U")),
        "source_time": source,
        "source_available_n": available,
        "prediction_n": PREDICTION_N,
        "sampled_with_replacement": bool(len(np.unique(indices)) < len(indices)),
        "canonical_input_roster": str(split.source_roster_npz),
        "canonical_input_roster_sha256": split.source_roster_sha256,
        "shared_across_all_benchmark_families": True,
    }


def _bootstrap_adata(data: TrainingData, indices: np.ndarray, source: float):
    import anndata as ad
    import pandas as pd

    row_ids = np.asarray(
        [f"bootstrap_{index:05d}" for index in range(PREDICTION_N)], dtype=str
    )
    obs = pd.DataFrame(
        {
            data.time_key: np.full(PREDICTION_N, source, dtype=np.float32),
            data.row_id_key: row_ids,
        },
        index=pd.Index(row_ids, name=data.row_id_key),
    )
    result = ad.AnnData(
        X=data.state[indices].astype(np.float32),
        obs=obs,
    )
    result.obsm[data.state_key] = data.state[indices].astype(np.float32)
    result.obsm[data.spatial_key] = data.spatial[indices].astype(np.float32)
    return result


@contextlib.contextmanager
def _working_directory(path: Path):
    prior = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prior)


def _seed_runtime() -> None:
    import torch

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _simulate(
    *,
    repo: Path,
    model_dir: Path,
    data: TrainingData,
    bootstrap: Any,
    times: Sequence[float],
    device: str,
    dt: float,
    interaction_m: int,
) -> tuple[list[np.ndarray], list[np.ndarray], dict[str, Any]]:
    if str(repo.resolve()) not in sys.path:
        sys.path.insert(0, str(repo.resolve()))
    import torch
    from CytoBridge.tl import load_dynamical_model_from_dir
    from CytoBridge.tl.downstream import simulation as simulation_api

    signature = inspect.signature(simulation_api.simulate_sde_points)
    required = {
        "adata",
        "model",
        "dim",
        "time_index",
        "n_samples",
        "ts_points",
        "dt",
        "sigma",
        "include_score",
        "interaction_m",
        "device",
        "time_key",
        "obsm_key",
        "spatial_key",
        "concat_spatial",
    }
    missing = sorted(required - set(signature.parameters))
    if missing:
        raise ContractError(
            f"CytoBridge non-split simulator API lacks required arguments: {missing}"
        )
    _seed_runtime()
    with _working_directory(repo.resolve()):
        loaded = load_dynamical_model_from_dir(
            model_dir, dim=data.joint_dim, device=str(device)
        )
    validate_training_config(loaded.config)
    if loaded.weight_stage != "Finetune":
        raise ContractError(
            f"inference must load Finetune weights, found {loaded.weight_stage!r}"
        )
    if loaded.score_stage != "Score_Refine":
        raise ContractError(
            f"inference must load Score_Refine score, found {loaded.score_stage!r}"
        )
    # Model construction consumes torch RNG while initializing modules before
    # checkpoint loading. Reset after loading so SDE noise itself starts from
    # the declared benchmark seed, independent of constructor implementation.
    _seed_runtime()
    points, weights = simulation_api.simulate_sde_points(
        adata=bootstrap,
        model=loaded.model,
        dim=data.joint_dim,
        time_index=0,
        n_samples=PREDICTION_N,
        ts_points=[float(value) for value in times],
        dt=float(dt),
        sigma=SIGMA,
        include_score=True,
        interaction_m=int(interaction_m),
        device=str(device),
        time_key=data.time_key,
        obsm_key=data.state_key,
        spatial_key=data.spatial_key,
        concat_spatial=True,
        verbose=True,
    )
    point_list = [np.asarray(value, dtype=np.float32) for value in points]
    weight_list = [np.asarray(value, dtype=np.float64).reshape(-1) for value in weights]
    if len(point_list) != len(times) or len(weight_list) != len(times):
        raise ContractError("non-split simulator did not return every requested time")
    for index, (point, weight) in enumerate(zip(point_list, weight_list)):
        if point.shape != (PREDICTION_N, data.joint_dim):
            raise ContractError(
                f"simulation output {index} has shape {point.shape}, expected "
                f"{(PREDICTION_N, data.joint_dim)}"
            )
        if weight.shape != (PREDICTION_N,):
            raise ContractError(f"simulation weight {index} has shape {weight.shape}")
        if not np.isfinite(point).all():
            raise ContractError(f"simulation output {index} contains non-finite values")
        if not np.isfinite(weight).all() or np.any(weight < 0) or weight.sum() <= 0:
            raise ContractError(f"simulation weights {index} are invalid")
    return point_list, weight_list, {
        "official_api": "CytoBridge.tl.downstream.simulation.simulate_sde_points",
        "official_api_signature": str(signature),
        "simulation_mode": "continuous_non_split_weighted_sde",
        "weight_stage": loaded.weight_stage,
        "score_stage": loaded.score_stage,
        "weights_semantics": "native_unnormalised_growth_mass",
        "torch_version": torch.__version__,
    }


def _atomic_prediction(
    path: Path,
    point: np.ndarray,
    weight: np.ndarray,
    data: TrainingData,
    source: float,
    target: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            spatial=point[:, : data.spatial_dim].astype(np.float32),
            state=point[:, data.spatial_dim :].astype(np.float32),
            weights=weight.astype(np.float64),
            source_time=np.asarray([source], dtype=np.float64),
            target_time=np.asarray([target], dtype=np.float64),
        )
    os.replace(temporary, path)


def _prediction_summary(
    *,
    prediction_path: Path,
    split: SplitInput,
    data: TrainingData,
    target: float,
    source: float,
    weight: np.ndarray,
    model_report: Mapping[str, Any],
    roster: Mapping[str, Any],
    simulation: Mapping[str, Any],
    repo: Path,
    device: str,
    dt: float,
    interaction_m: int,
) -> dict[str, Any]:
    checkpoint_hashes = {
        name: record["sha256"]
        for name, record in model_report["checkpoints"].items()
    }
    return {
        "status": "complete",
        "method": METHOD,
        "implementation": (
            f"CytoBridge alpha_spatial={ALPHA_SPATIAL:g}, "
            f"alpha_express={ALPHA_EXPRESS:g}"
        ),
        "regime": split.regime,
        "track": split.regime,
        "split_id": split.split_id,
        "target": _target_json(target),
        "target_time": _target_json(target),
        "source_time": _target_json(source),
        "source_policy": (
            "fixed t0 bootstrap shared across all full-data targets; no intermediate reset"
            if split.regime == "full_data"
            else "nearest previous observed training stage to held-out target"
        ),
        "output_scope": "native_joint",
        "primary_benchmark_eligible": True,
        "native_vs_adapter": "native_joint",
        "native_joint": True,
        "native_mass": True,
        "native_growth": True,
        "spatial_warp_applied": False,
        "spatial_warp": "none",
        "split_sde": False,
        "continuous_across_targets": split.regime == "full_data",
        "prediction_n": PREDICTION_N,
        "prediction_n_policy": "fixed_from_train_contract_before_truth_access",
        "truth_inputs_opened": False,
        "alpha_express": ALPHA_EXPRESS,
        "alpha_spatial": ALPHA_SPATIAL,
        "sigma": SIGMA,
        "dt": float(dt),
        "include_score": True,
        "include_interaction": True,
        "interaction_m": int(interaction_m),
        "seed": SEED,
        "device": str(device),
        "predicted_mass": float(weight.sum()),
        "weights_are_unnormalised": True,
        "prediction_npz": str(prediction_path),
        "prediction_npz_sha256": sha256_file(prediction_path),
        "state_dim": data.state_dim,
        "spatial_dim": data.spatial_dim,
        "joint_dim": data.joint_dim,
        "config_sha256": model_report["config_sha256"],
        "checkpoint_sha256": checkpoint_hashes,
        "stage_complete": model_report["stage_complete"],
        "stage_count": model_report["stage_count"],
        "training_reference_match": model_report["training_reference_match"],
        "source_roster": dict(roster),
        "simulation": dict(simulation),
        "environment": environment_provenance(device),
        "repo": repo_identity(repo),
        **input_provenance(split),
    }


def _write_prediction_summary(path: Path, report: Mapping[str, Any]) -> None:
    write_json(path.with_suffix(".summary.json"), report)
    write_json(path.parent / "summary.json", report)


def command_infer_loto(args: argparse.Namespace) -> None:
    _validate_runtime_constants(args)
    split = read_split_input(args.input_manifest, args.split)
    _require_loto(split)
    data = load_training_data(split)
    _input_report(split, data)
    model_report = _model_report(args.model_dir, split, data)
    output = new_output_dir(args.output_dir)
    source, schedule = inference_schedule(split)
    target = schedule[-1]
    indices, roster = _source_roster(output, split, data, source)
    bootstrap = _bootstrap_adata(data, indices, source)
    points, weights, simulation = _simulate(
        repo=args.repo,
        model_dir=args.model_dir,
        data=data,
        bootstrap=bootstrap,
        times=schedule,
        device=args.device,
        dt=args.dt,
        interaction_m=args.interaction_m,
    )
    prediction = output / "prediction.npz"
    _atomic_prediction(prediction, points[-1], weights[-1], data, source, target)
    report = _prediction_summary(
        prediction_path=prediction,
        split=split,
        data=data,
        target=target,
        source=source,
        weight=weights[-1],
        model_report=model_report,
        roster=roster,
        simulation=simulation,
        repo=args.repo,
        device=args.device,
        dt=args.dt,
        interaction_m=args.interaction_m,
    )
    _write_prediction_summary(prediction, report)
    print(json.dumps(plain(report), indent=2, sort_keys=True))


def command_infer_full(args: argparse.Namespace) -> None:
    _validate_runtime_constants(args)
    split = read_split_input(args.input_manifest, args.split)
    _require_full(split)
    data = load_training_data(split)
    _input_report(split, data)
    model_report = _model_report(args.model_dir, split, data)
    output = new_output_dir(args.output_dir)
    source, times = inference_schedule(split)
    targets = times[1:]
    indices, roster = _source_roster(output, split, data, source)
    bootstrap = _bootstrap_adata(data, indices, source)
    # Exactly one simulator call: every target is sampled from the same
    # continuous trajectory originating at the fixed t0 roster.
    points, weights, simulation = _simulate(
        repo=args.repo,
        model_dir=args.model_dir,
        data=data,
        bootstrap=bootstrap,
        times=times,
        device=args.device,
        dt=args.dt,
        interaction_m=args.interaction_m,
    )
    summaries: list[dict[str, Any]] = []
    for index, target in enumerate(targets, start=1):
        target_dir = output / f"t{_time_suffix(target)}"
        prediction = target_dir / "prediction.npz"
        _atomic_prediction(
            prediction, points[index], weights[index], data, source, target
        )
        report = _prediction_summary(
            prediction_path=prediction,
            split=split,
            data=data,
            target=target,
            source=source,
            weight=weights[index],
            model_report=model_report,
            roster=roster,
            simulation=simulation,
            repo=args.repo,
            device=args.device,
            dt=args.dt,
            interaction_m=args.interaction_m,
        )
        _write_prediction_summary(prediction, report)
        summaries.append(report)
    run_report = {
        "status": "complete",
        "method": METHOD,
        "regime": "full_data",
        "split_id": split.split_id,
        "source_time": _target_json(source),
        "targets": [_target_json(value) for value in targets],
        "single_continuous_non_split_call": True,
        "intermediate_reset": False,
        "spatial_warp_applied": False,
        "prediction_n": PREDICTION_N,
        "seed": SEED,
        "source_roster": roster,
        "prediction_summaries": [
            {
                "target": report["target"],
                "prediction_npz": report["prediction_npz"],
                "prediction_npz_sha256": report["prediction_npz_sha256"],
                "predicted_mass": report["predicted_mass"],
            }
            for report in summaries
        ],
        **input_provenance(split),
    }
    write_json(output / "run_summary.json", run_report)
    print(json.dumps(plain(run_report), indent=2, sort_keys=True))


def _repo_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo", type=Path, default=REPO_ROOT)


def _input_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input-manifest", required=True, type=Path)
    parser.add_argument("--split", required=True)


def _inference_arguments(parser: argparse.ArgumentParser) -> None:
    _repo_argument(parser)
    _input_arguments(parser)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prediction-n", type=int, default=PREDICTION_N)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--sigma", type=float, default=SIGMA)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--interaction-m", type=int, default=1024)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    preflight = sub.add_parser("preflight", help="validate train contract/config/model")
    _repo_argument(preflight)
    _input_arguments(preflight)
    preflight.add_argument("--training-config", required=True, type=Path)
    preflight.add_argument("--model-dir", type=Path)
    preflight.add_argument("--output-json", type=Path)
    preflight.add_argument("--device", default="cuda")
    preflight.set_defaults(func=command_preflight)

    graph = sub.add_parser(
        "prepare-loto", help="rebuild LOTO graphs and edge classifier from train rows"
    )
    _repo_argument(graph)
    _input_arguments(graph)
    graph.add_argument("--database", required=True, type=Path)
    graph.add_argument("--output-dir", required=True, type=Path)
    graph.add_argument("--device", default="cuda")
    graph.add_argument("--expression-layer", default="counts")
    graph.add_argument("--interaction-cutoff", type=float)
    graph.add_argument("--spot-diameter", type=float)
    graph.add_argument("--edge-threshold", type=float)
    graph.add_argument("--edge-epochs", type=int, default=100)
    graph.add_argument("--edge-batch-size", type=int, default=1024)
    graph.add_argument("--edge-learning-rate", type=float, default=0.001)
    graph.add_argument("--edge-train-sample-ratio", type=float, default=1.0)
    graph.add_argument("--edge-max-train-edges", type=int)
    graph.add_argument("--edge-num-workers", type=int, default=4)
    graph.add_argument("--seed", type=int, default=SEED)
    graph.add_argument("--quiet", action="store_true")
    graph.set_defaults(func=command_prepare_loto)

    fit = sub.add_parser("fit-loto", help="run the exact six-stage fit for one LOTO fold")
    _repo_argument(fit)
    _input_arguments(fit)
    fit.add_argument("--training-config", required=True, type=Path)
    fit.add_argument("--graph-dir", required=True, type=Path)
    fit.add_argument("--output-dir", required=True, type=Path)
    fit.add_argument("--device", default="cuda")
    fit.add_argument("--seed", type=int, default=SEED)
    fit.add_argument("--sigma", type=float, default=SIGMA)
    fit.set_defaults(func=command_fit_loto)

    validate = sub.add_parser(
        "validate-model", help="validate six stages and exact training-reference linkage"
    )
    _repo_argument(validate)
    _input_arguments(validate)
    validate.add_argument("--model-dir", required=True, type=Path)
    validate.add_argument("--output-json", type=Path)
    validate.add_argument("--device", default="cpu")
    validate.set_defaults(func=command_validate_model)

    infer_loto = sub.add_parser(
        "infer-loto", help="predict held-out target from previous observed stage"
    )
    _inference_arguments(infer_loto)
    infer_loto.set_defaults(func=command_infer_loto)

    infer_full = sub.add_parser(
        "infer-full", help="one continuous t0-to-all-targets full-data simulation"
    )
    _inference_arguments(infer_full)
    infer_full.set_defaults(func=command_infer_full)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if hasattr(args, "dt") and (not np.isfinite(args.dt) or args.dt <= 0):
        raise ContractError("dt must be finite and positive")
    if hasattr(args, "interaction_m") and args.interaction_m <= 0:
        raise ContractError("interaction_m must be positive")
    args.repo = args.repo.expanduser().resolve()
    if not args.repo.is_dir():
        raise FileNotFoundError(args.repo)
    args.func(args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ContractError, FileExistsError, FileNotFoundError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
