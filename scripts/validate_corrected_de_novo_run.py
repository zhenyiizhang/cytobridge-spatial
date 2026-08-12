#!/usr/bin/env python3
"""Validate the four corrected CytoBridge de novo production runs.

This is an artifact-level scientific acceptance check.  It reads the outputs
created by ``cytobridge workflow --train`` and verifies the shared protocol,
the complete six-stage fit, the generated edge prior, and the numerical
downstream products.  It does not modify the run directory.

Example
-------
python scripts/validate_corrected_de_novo_run.py \
    --run-root /path/to/corrected-de-novo-20260813-r2 \
    --report /path/to/corrected-de-novo-acceptance.json
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import yaml


STAGES = (
    "Pretrain",
    "Refine",
    "Init_interaction",
    "Train_Score",
    "Finetune",
    "Score_Refine",
)


DATASETS = {
    "zebrafish": {
        "shape": (11999, 26628),
        "counts_layer": "counts",
        "annotation_key": "Annotation",
        "classifier_k": 10,
        "cutoff": 0.09606367405591873,
        "observed_counts": {0.0: 563, 1.0: 1036, 2.0: 2081, 3.0: 3048, 4.0: 5271},
        "interpolated": (0.5, 1.5, 2.5, 3.5),
        "score_epochs": 2001,
        "species": "zebrafish",
    },
    "mosta": {
        "shape": (344603, 23761),
        "counts_layer": "count",
        "annotation_key": "Annotation",
        "classifier_k": 10,
        "cutoff": 0.02400244047956264,
        "observed_counts": {0.0: 51365, 1.0: 77369, 2.0: 102519, 3.0: 113350},
        "interpolated": (0.25, 0.5, 0.75, 1.25, 1.5, 1.75, 2.25, 2.5, 2.75),
        "score_epochs": 2001,
        "species": "mouse",
    },
    "arista": {
        "shape": (46209, 16379),
        "counts_layer": "counts",
        "annotation_key": "Annotation",
        "classifier_k": 10,
        "cutoff": 0.03154105148551745,
        "observed_counts": {0.0: 7668, 1.0: 8106, 2.0: 9440, 3.0: 9676, 4.0: 11319},
        "interpolated": (0.5, 1.5, 2.5, 3.5),
        "score_epochs": 2001,
        "species": "hs",
    },
    "admouse": {
        "shape": (172092, 347),
        "counts_layer": "counts",
        "annotation_key": "major_annotation",
        "classifier_k": 1,
        "cutoff": 0.012106042891492197,
        "observed_counts": {0.0: 53615, 1.0: 58447, 2.0: 60030},
        "interpolated": tuple(
            round(value / 10, 1) for value in range(1, 26) if value not in (10, 20)
        ),
        "score_epochs": 3001,
        "species": "mouse",
    },
}


@dataclass
class Audit:
    dataset: str
    checks: list[dict[str, str]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def check(self, condition: bool, name: str, detail: str) -> None:
        self.checks.append(
            {"name": name, "status": "PASS" if condition else "FAIL", "detail": detail}
        )

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    @property
    def errors(self) -> list[dict[str, str]]:
        return [item for item in self.checks if item["status"] == "FAIL"]

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": "PASS" if not self.errors else "FAIL",
            "checks": self.checks,
            "warnings": self.warnings,
        }


def close_enough(actual: float, expected: float, atol: float = 1e-12) -> bool:
    return math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=atol)


def safe_time_name(value: float) -> str:
    return f"{float(value):g}".replace("-", "minus_").replace(".", "p")


def h5_shape(node: h5py.Dataset | h5py.Group) -> tuple[int, ...]:
    if isinstance(node, h5py.Dataset):
        return tuple(int(value) for value in node.shape)
    return tuple(int(value) for value in node.attrs["shape"])


def numeric_node(node: h5py.Dataset | h5py.Group) -> h5py.Dataset:
    """Return the stored numeric values for a dense or sparse AnnData matrix."""

    return node if isinstance(node, h5py.Dataset) else node["data"]


def numeric_stats(node: h5py.Dataset | h5py.Group) -> tuple[bool, int, float, float]:
    """Stream a numeric HDF5 array and return finite/count/min/max statistics."""

    values = numeric_node(node)
    count = 0
    minimum = math.inf
    maximum = -math.inf
    finite = True
    block_rows = 100_000
    if values.ndim == 0:
        blocks = (np.asarray(values[()]),)
    else:
        blocks = (
            np.asarray(values[start : start + block_rows])
            for start in range(0, values.shape[0], block_rows)
        )
    for block in blocks:
        if block.size == 0:
            continue
        finite = finite and bool(np.isfinite(block).all())
        count += int(block.size)
        minimum = min(minimum, float(np.nanmin(block)))
        maximum = max(maximum, float(np.nanmax(block)))
    return finite, count, minimum, maximum


def h5_sum_identity(
    handle: h5py.File,
    total_key: str,
    part_keys: tuple[str, ...],
    *,
    atol: float = 1e-5,
) -> bool:
    total = numeric_node(handle[total_key])
    parts = [numeric_node(handle[key]) for key in part_keys]
    if any(part.shape != total.shape for part in parts):
        return False
    for start in range(0, total.shape[0], 100_000):
        stop = start + 100_000
        expected = sum(np.asarray(part[start:stop]) for part in parts)
        if not np.allclose(
            np.asarray(total[start:stop]), expected, rtol=1e-5, atol=atol
        ):
            return False
    return True


def read_metadata(path: Path):
    data = ad.read_h5ad(path, backed="r")
    try:
        return data.shape, data.obs.copy(), data.obs_names.copy(), dict(data.uns)
    finally:
        data.file.close()


def expected_epochs(spec: dict[str, Any]) -> dict[str, int]:
    return {
        "Pretrain": 100,
        "Refine": 100,
        "Init_interaction": 50,
        "Train_Score": int(spec["score_epochs"]),
        "Finetune": 1000,
        "Score_Refine": int(spec["score_epochs"]),
    }


def required_files(run_dir: Path, dataset: str) -> dict[str, Path]:
    training = run_dir / "training"
    downstream = run_dir / "downstream"
    edge_model = run_dir / "preprocess" / "edge_classifier" / f"{dataset}_edge_model.pt"
    return {
        "aligned H5AD": run_dir / "preprocess" / f"{dataset}_aligned.h5ad",
        "generated edge model": edge_model,
        "generated edge metadata": edge_model.with_suffix(
            edge_model.suffix + ".meta.json"
        ),
        "resolved training config": training / "config.yaml",
        "training history": training / "training_history.csv",
        "training run summary": training / "training_run_summary.json",
        "trained AnnData": training / "adata.h5ad",
        "downstream summary": downstream / "summary.json",
    }


def validate_aligned(
    audit: Audit,
    paths: dict[str, Path],
    spec: dict[str, Any],
) -> tuple[int, int]:
    aligned_path = paths["aligned H5AD"]
    shape, obs, obs_names, uns = read_metadata(aligned_path)
    audit.check(
        tuple(shape) == tuple(spec["shape"]), "analyzed cohort", f"shape={shape}"
    )
    audit.check(
        bool(obs_names.is_unique),
        "stable observation identities",
        f"unique={obs_names.is_unique}",
    )

    time_values = pd.to_numeric(obs["time_point_processed"], errors="raise").astype(
        float
    )
    counts = {
        float(key): int(value)
        for key, value in time_values.value_counts().sort_index().items()
    }
    audit.check(
        counts == spec["observed_counts"],
        "observed slice membership",
        f"counts={counts}",
    )
    annotation = obs[spec["annotation_key"]]
    valid_annotation = bool(
        annotation.notna().all() and annotation.astype(str).str.len().gt(0).all()
    )
    audit.check(
        valid_annotation, "cell annotations", f"column={spec['annotation_key']!r}"
    )

    preprocess_info = uns.get("preprocess_info", {})
    raw_layer = str(preprocess_info.get("raw_counts_layer", ""))
    audit.check(
        raw_layer == spec["counts_layer"],
        "raw expression layer",
        f"recorded={raw_layer!r}, expected={spec['counts_layer']!r}",
    )

    with h5py.File(aligned_path, "r") as handle:
        latent = handle["obsm/X_latent"]
        spatial = handle["obsm/spatial_aligned"]
        pcs = handle["varm/PCs"]
        center = handle["var/pca_center"]
        counts_node = handle[f"layers/{spec['counts_layer']}"]
        latent_stats = numeric_stats(latent)
        spatial_stats = numeric_stats(spatial)
        pcs_stats = numeric_stats(pcs)
        center_stats = numeric_stats(center)
        counts_stats = numeric_stats(counts_node)
        audit.check(
            h5_shape(latent) == (shape[0], 50) and latent_stats[0],
            "aligned latent state",
            f"shape={h5_shape(latent)}, range=({latent_stats[2]:.6g}, {latent_stats[3]:.6g})",
        )
        audit.check(
            h5_shape(spatial) == (shape[0], 2) and spatial_stats[0],
            "aligned spatial state",
            f"shape={h5_shape(spatial)}, range=({spatial_stats[2]:.6g}, {spatial_stats[3]:.6g})",
        )
        audit.check(
            h5_shape(pcs) == (shape[1], 50) and pcs_stats[0] and center_stats[0],
            "retained PCA transform",
            f"PCs={h5_shape(pcs)}, center={h5_shape(center)}",
        )
        audit.check(
            counts_stats[0] and counts_stats[1] > 0 and counts_stats[2] >= 0,
            "nonnegative finite raw counts",
            f"stored_values={counts_stats[1]}, range=({counts_stats[2]:.6g}, {counts_stats[3]:.6g})",
        )
    return int(shape[0]), 52


def validate_edge_predictor(
    audit: Audit,
    run_dir: Path,
    paths: dict[str, Path],
    spec: dict[str, Any],
) -> float:
    meta = json.loads(paths["generated edge metadata"].read_text(encoding="utf-8"))
    threshold = float(meta["edge_predictor_threshold"])
    selected = float(meta["edge_predictor_threshold_selected"])
    audit.check(
        meta.get("selection_source") == "validation"
        and close_enough(threshold, selected),
        "validation-selected edge threshold",
        f"threshold={threshold:.9g}, source={meta.get('selection_source')!r}",
    )
    audit.check(
        0.0 <= threshold <= 1.0
        and close_enough(meta["distance_threshold"], spec["cutoff"]),
        "edge distance and decision thresholds",
        f"cutoff={float(meta['distance_threshold']):.17g}, decision={threshold:.9g}",
    )
    audit.check(
        int(meta.get("random_seed", -1)) == 42
        and meta.get("split", {}).get("strategy") == "node_disjoint_holdout",
        "edge predictor split and seed",
        f"split={meta.get('split', {}).get('strategy')!r}, seed={meta.get('random_seed')}",
    )
    universe = meta["candidate_universe"]
    candidate_ok = (
        universe.get("definition") == "all directed pairs with 1e-6 < distance < cutoff"
        and int(universe.get("positive_edges", 0)) > 0
        and int(universe.get("negative_edges", 0)) > 0
        and int(universe.get("training_balanced_edges", 0)) > 0
    )
    audit.check(
        candidate_ok,
        "edge candidate universe",
        f"positive={universe.get('positive_edges')}, negative={universe.get('negative_edges')}",
    )
    expected_times = tuple(spec["observed_counts"])
    meta_times = tuple(
        float(value) for value in meta["split"]["time_values_by_local_index"]
    )
    graph_files = sorted(
        (run_dir / "preprocess" / "input_graph").glob("*/*_adjacency_records")
    )
    audit.check(
        meta_times == expected_times and len(graph_files) == len(expected_times),
        "per-time LR interaction graphs",
        f"times={meta_times}, graph_files={len(graph_files)}",
    )
    validation = meta["validation_metrics_at_selected_threshold"]
    test = meta["test_metrics_at_validation_threshold"]
    quality = (
        f"validation AP={validation.get('average_precision')}, F1={validation.get('f1')}; "
        f"test AP={test.get('average_precision')}, F1={test.get('f1')}"
    )
    audit.check(
        int(validation.get("n_candidates", 0)) > 0
        and int(test.get("n_candidates", 0)) > 0,
        "natural-prevalence edge holdouts",
        quality,
    )
    if float(validation.get("f1", 0.0)) < 0.1:
        audit.warn(
            "Low edge-predictor validation F1 under natural prevalence: " + quality
        )
    return threshold


def validate_training(
    audit: Audit,
    paths: dict[str, Path],
    spec: dict[str, Any],
    threshold: float,
    model_dim: int,
) -> None:
    config = yaml.safe_load(
        paths["resolved training config"].read_text(encoding="utf-8")
    )
    defaults = config["training"]["defaults"]
    interaction = config["model"]["interaction_net"]
    audit.check(
        int(config["seed"]) == 42
        and close_enough(defaults["alpha_express"], 0.015)
        and close_enough(defaults["alpha_spatial"], 10.0),
        "shared training constants",
        f"seed={config['seed']}, alpha_express={defaults['alpha_express']}, alpha_spatial={defaults['alpha_spatial']}",
    )
    edge_path = paths["generated edge model"].resolve()
    configured_edge_path = Path(interaction["edge_predictor_path"]).resolve()
    audit.check(
        configured_edge_path == edge_path
        and close_enough(interaction["edge_predictor_thre"], threshold)
        and close_enough(interaction["cutoff"], spec["cutoff"]),
        "generated edge prior wired into training",
        f"path={configured_edge_path}, threshold={interaction['edge_predictor_thre']}",
    )

    plan = config["training"]["plan"]
    plan_names = tuple(str(stage["name"]) for stage in plan)
    epochs = expected_epochs(spec)
    audit.check(plan_names == STAGES, "six-stage training plan", f"stages={plan_names}")
    plan_epochs = {str(stage["name"]): int(stage["epochs"]) for stage in plan}
    audit.check(
        plan_epochs == epochs, "configured stage lengths", f"epochs={plan_epochs}"
    )

    history = pd.read_csv(paths["training history"])
    history_names = tuple(history["stage"].drop_duplicates().astype(str))
    complete = history_names == STAGES
    details = []
    for stage_name, stage_epochs in epochs.items():
        stage_rows = history.loc[history["stage"] == stage_name]
        epoch_values = stage_rows["epoch"].astype(int).to_numpy()
        selected = (
            stage_rows["is_selected_checkpoint"].astype(str).str.lower().eq("true")
        )
        stage_complete = (
            len(stage_rows) == stage_epochs
            and np.array_equal(epoch_values, np.arange(1, stage_epochs + 1))
            and int(selected.sum()) == 1
        )
        complete = complete and stage_complete
        details.append(
            f"{stage_name}:{len(stage_rows)}/{stage_epochs},selected={int(selected.sum())}"
        )
    core_numeric = history[["loss", "checkpoint_value", "learning_rate"]].to_numpy(
        dtype=float
    )
    complete = complete and bool(np.isfinite(core_numeric).all())
    audit.check(complete, "complete finite training history", "; ".join(details))

    for stage in plan:
        name = str(stage["name"])
        is_score = str(stage.get("mode", "")).lower() == "score_matching"
        strategy = str(
            stage.get("save_strategy", defaults.get("save_strategy", "best"))
        )
        filename = "score_model.pth" if is_score else f"{strategy}_model.pth"
        checkpoint = paths["resolved training config"].parent / name / filename
        audit.check(
            checkpoint.is_file() and checkpoint.stat().st_size > 0,
            f"{name} checkpoint",
            str(checkpoint),
        )

    run_summary = json.loads(paths["training run summary"].read_text(encoding="utf-8"))
    stage_summaries = run_summary["stages"]
    summary_complete = tuple(item["stage"] for item in stage_summaries) == STAGES
    summary_complete = summary_complete and all(
        int(item["recorded_epochs"]) == epochs[item["stage"]]
        and int(item["configured_epochs"]) == epochs[item["stage"]]
        and item["selected_checkpoint_epoch"] is not None
        for item in stage_summaries
    )
    audit.check(
        summary_complete and float(run_summary["timing"]["run_wall_time_seconds"]) > 0,
        "measured training run summary",
        f"wall_seconds={run_summary['timing']['run_wall_time_seconds']:.3f}",
    )

    trained_shape, _, _, _ = read_metadata(paths["trained AnnData"])
    audit.check(
        tuple(trained_shape) == tuple(spec["shape"]),
        "trained AnnData cohort",
        f"shape={trained_shape}",
    )
    with h5py.File(paths["trained AnnData"], "r") as handle:
        expected_vectors = (
            "obsm/velocity_model",
            "obsm/interaction_model",
            "obsm/score_gradient_model",
            "obsm/full_drift_model",
            "obsm/growth_rate",
        )
        vectors_ok = all(key in handle for key in expected_vectors)
        vector_details = []
        if vectors_ok:
            for key in expected_vectors:
                stats = numeric_stats(handle[key])
                vectors_ok = vectors_ok and stats[0] and stats[1] > 0
                vector_details.append(
                    f"{key.rsplit('/', 1)[-1]}={h5_shape(handle[key])}"
                )
            vectors_ok = vectors_ok and h5_shape(handle["obsm/full_drift_model"]) == (
                trained_shape[0],
                model_dim,
            )
            vectors_ok = vectors_ok and h5_sum_identity(
                handle,
                "obsm/full_drift_model",
                (
                    "obsm/velocity_model",
                    "obsm/interaction_model",
                    "obsm/score_gradient_model",
                ),
            )
        audit.check(
            vectors_ok, "finite fitted vector components", ", ".join(vector_details)
        )

    import CytoBridge as cb
    from CytoBridge.workflow import (
        WorkflowOptions,
        _loaded_model_scientific_contract,
        load_workflow_config,
    )

    loaded = cb.tl.load_dynamical_model_from_dir(
        paths["resolved training config"].parent,
        dim=model_dim,
        device="cpu",
        edge_predictor_path=paths["generated edge model"],
    )
    preset, _ = load_workflow_config(audit.dataset)
    contract = _loaded_model_scientific_contract(
        loaded,
        config=preset,
        options=WorkflowOptions(train=True),
    )
    parameters_finite = all(
        bool(np.isfinite(value.detach().cpu().numpy()).all())
        for value in loaded.model.state_dict().values()
    )
    audit.check(
        loaded.weight_stage == "Finetune"
        and loaded.score_stage == "Score_Refine"
        and parameters_finite
        and contract["status"] == "matches requested preset",
        "strict final checkpoint load",
        f"weight={loaded.weight_stage}, score={loaded.score_stage}, contract={contract['status']}",
    )


def validate_slice(
    audit: Audit,
    path: Path,
    annotation_key: str,
    model_dim: int,
) -> int:
    shape, obs, _, _ = read_metadata(path)
    annotations_ok = annotation_key in obs and obs[annotation_key].notna().all()
    with h5py.File(path, "r") as handle:
        state = handle["X"]
        spatial = handle["obsm/spatial"]
        state_stats = numeric_stats(state)
        spatial_stats = numeric_stats(spatial)
        consistent = h5_shape(state) == (shape[0], model_dim) and h5_shape(spatial) == (
            shape[0],
            2,
        )
        if consistent:
            state_values = numeric_node(state)
            spatial_values = numeric_node(spatial)
            for start in range(0, shape[0], 100_000):
                stop = start + 100_000
                if not np.allclose(
                    np.asarray(state_values[start:stop, :2]),
                    np.asarray(spatial_values[start:stop]),
                    rtol=0,
                    atol=0,
                ):
                    consistent = False
                    break
    audit.check(
        bool(
            shape[0] > 0
            and annotations_ok
            and state_stats[0]
            and spatial_stats[0]
            and consistent
        ),
        f"finite state {path.stem}",
        f"shape={shape}",
    )
    return int(shape[0])


def finite_numeric_frame(frame: pd.DataFrame, *, allow_na: bool = False) -> bool:
    numeric = frame.select_dtypes(include=[np.number])
    if numeric.empty:
        return False
    values = numeric.to_numpy(dtype=float)
    if allow_na:
        return bool(
            np.isfinite(values[~np.isnan(values)]).all() and np.isfinite(values).any()
        )
    return bool(np.isfinite(values).all())


def validate_downstream(
    audit: Audit,
    run_dir: Path,
    paths: dict[str, Path],
    spec: dict[str, Any],
    threshold: float,
    model_dim: int,
) -> None:
    downstream = run_dir / "downstream"
    summary = json.loads(paths["downstream summary"].read_text(encoding="utf-8"))
    observed = tuple(spec["observed_counts"])
    expected_times = tuple(sorted(set(observed + tuple(spec["interpolated"]))))
    actual_times = tuple(float(value) for value in summary["time_points"])
    audit.check(
        summary["dataset"] == audit.dataset
        and int(summary["seed"]) == 42
        and close_enough(summary["alpha_express"], 0.015)
        and int(summary["classifier_k"]) == int(spec["classifier_k"]),
        "downstream scientific constants",
        f"seed={summary['seed']}, alpha={summary['alpha_express']}, k={summary['classifier_k']}",
    )
    audit.check(
        actual_times == expected_times, "downstream time grid", f"times={actual_times}"
    )
    model_contract = summary["model"]["scientific_contract"]
    audit.check(
        model_contract["status"] == "matches requested preset"
        and close_enough(model_contract["interaction_cutoff"], spec["cutoff"])
        and close_enough(model_contract["edge_predictor_threshold"], threshold)
        and summary["model"]["weight_stage"] == "Finetune"
        and summary["model"]["score_stage"] == "Score_Refine",
        "downstream loaded corrected checkpoint",
        f"cutoff={model_contract['interaction_cutoff']}, threshold={model_contract['edge_predictor_threshold']}",
    )
    classifier_metrics = (
        float(summary["classifier_accuracy"]),
        float(summary["classifier_balanced_accuracy"]),
    )
    audit.check(
        all(
            math.isfinite(value) and 0.0 <= value <= 1.0 for value in classifier_metrics
        ),
        "classifier metrics",
        f"accuracy={classifier_metrics[0]:.6g}, balanced_accuracy={classifier_metrics[1]:.6g}",
    )

    expected_analyses = (
        "velocity",
        "growth",
        "composition",
        "communication",
        "figures",
        "gene_dynamics",
        "ligand_receptor",
    )
    analyses = summary["analyses"]
    analysis_ok = all(
        analyses.get(name, {}).get("status") == "completed"
        for name in expected_analyses
    )
    audit.check(
        analysis_ok,
        "complete standard downstream analyses",
        ", ".join(
            f"{name}={analyses.get(name, {}).get('status')}"
            for name in expected_analyses
        ),
    )

    slice_counts: dict[float, int] = {}
    for time_value in expected_times:
        path = downstream / "slice_data" / f"time_{safe_time_name(time_value)}.h5ad"
        if not path.is_file():
            audit.check(False, f"slice file t={time_value:g}", str(path))
            continue
        slice_counts[time_value] = validate_slice(
            audit, path, spec["annotation_key"], model_dim
        )
    observed_slice_counts = {time: slice_counts.get(time) for time in observed}
    audit.check(
        observed_slice_counts == spec["observed_counts"],
        "observed states retain real cells",
        f"counts={observed_slice_counts}",
    )

    velocity_path = downstream / "velocity" / "velocity_components.npz"
    velocity_ok = velocity_path.is_file()
    velocity_detail = "missing"
    if velocity_ok:
        with np.load(velocity_path) as velocity:
            expected_keys = {
                "drift",
                "interaction",
                "score",
                "full",
                "times",
                "features",
            }
            velocity_ok = set(velocity.files) == expected_keys
            arrays = {key: np.asarray(velocity[key]) for key in expected_keys}
            component_shape = arrays["full"].shape
            velocity_ok = velocity_ok and component_shape == (
                sum(spec["observed_counts"].values()),
                model_dim,
            )
            velocity_ok = velocity_ok and all(
                np.isfinite(value).all() for value in arrays.values()
            )
            velocity_ok = velocity_ok and np.allclose(
                arrays["full"],
                arrays["drift"] + arrays["interaction"] + arrays["score"],
                rtol=1e-5,
                atol=1e-5,
            )
            velocity_ok = velocity_ok and all(
                float(np.ptp(arrays[key])) > 0.0
                for key in ("drift", "interaction", "score", "full")
            )
            velocity_detail = (
                f"shape={component_shape}, times={tuple(np.unique(arrays['times']))}"
            )
    audit.check(
        velocity_ok, "finite nondegenerate velocity decomposition", velocity_detail
    )

    growth = pd.read_csv(downstream / "growth" / "growth_by_cell.csv")
    growth_ok = (
        len(growth) == sum(slice_counts.values())
        and finite_numeric_frame(growth)
        and float(growth["growth"].max() - growth["growth"].min()) > 0.0
    )
    audit.check(
        growth_ok,
        "finite nondegenerate growth",
        f"rows={len(growth)}, range=({growth['growth'].min():.6g}, {growth['growth'].max():.6g})",
    )

    composition = pd.read_csv(downstream / "composition" / "celltype_composition.csv")
    fraction_sums = composition.groupby("time")["fraction"].sum()
    composition_ok = (
        not composition.empty
        and finite_numeric_frame(composition)
        and bool((composition["count"] > 0).all())
        and bool(np.allclose(fraction_sums.to_numpy(), 1.0, atol=1e-8))
    )
    audit.check(composition_ok, "cell-type composition", f"rows={len(composition)}")

    communication_dir = downstream / "communication"
    communication = pd.read_csv(communication_dir / "communication_by_celltype.csv")
    communication_ok = not communication.empty and finite_numeric_frame(communication)
    communication_ok = communication_ok and bool(
        (communication["attention_per_source"] >= 0).all()
    )
    communication_ok = (
        communication_ok and float(communication["attention_per_source"].max()) > 0.0
    )
    sparse_details = []
    for time_value in expected_times:
        attention_path = (
            communication_dir
            / "sparse_attention"
            / f"attn_mean_interp_t{time_value}.npy"
        )
        edge_path = (
            communication_dir
            / "sparse_attention"
            / f"edge_index_interp_t{time_value}.npy"
        )
        if not attention_path.is_file() or not edge_path.is_file():
            communication_ok = False
            continue
        attention = np.load(attention_path, mmap_mode="r")
        edge_index = np.load(edge_path, mmap_mode="r")
        aligned = (
            edge_index.ndim == 2
            and edge_index.shape[0] == 2
            and edge_index.shape[1] == attention.shape[0]
        )
        valid = (
            aligned
            and attention.size > 0
            and np.isfinite(attention).all()
            and bool((attention >= 0).all())
        )
        communication_ok = communication_ok and valid
        sparse_details.append(f"t={time_value:g}:edges={attention.size}")
    dense_attention = list(
        (communication_dir / "sparse_attention").glob("attn_interp_t*.npy")
    )
    communication_ok = communication_ok and not dense_attention
    audit.check(
        communication_ok,
        "sparse nondegenerate communication",
        ", ".join(sparse_details),
    )

    gene_dir = downstream / "gene_dynamics"
    mean_expression = pd.read_csv(gene_dir / "mean_expression.csv", index_col=0)
    signed_expression = pd.read_csv(
        gene_dir / "signed_mean_expression.csv", index_col=0
    )
    diagnostics = pd.read_csv(gene_dir / "reconstruction_diagnostics.csv")
    top_genes = pd.read_csv(gene_dir / "top_variable_genes.csv")
    prototypes = pd.read_csv(gene_dir / "cluster_prototypes.csv", index_col=0)
    gene_ok = (
        not mean_expression.empty
        and mean_expression.shape[1] == len(expected_times)
        and finite_numeric_frame(mean_expression)
        and bool((mean_expression.to_numpy(dtype=float) >= 0).all())
        and finite_numeric_frame(signed_expression)
        and finite_numeric_frame(diagnostics)
        and not top_genes.empty
        and finite_numeric_frame(top_genes)
        and finite_numeric_frame(prototypes)
        and float(np.ptp(mean_expression.to_numpy(dtype=float))) > 0.0
    )
    audit.check(
        gene_ok,
        "finite nondegenerate gene dynamics",
        f"genes={mean_expression.shape[0]}, times={mean_expression.shape[1]}",
    )

    lr_dir = downstream / "ligand_receptor"
    lr_tables = {
        name: pd.read_csv(lr_dir / f"{name}.csv")
        for name in (
            "pair_timecourse",
            "celltype_timecourse",
            "pattern_summary",
            "coverage",
            "trajectory_coverage",
        )
    }
    lr_ok = all(not frame.empty for frame in lr_tables.values())
    lr_ok = lr_ok and all(
        finite_numeric_frame(frame, allow_na=True) for frame in lr_tables.values()
    )
    pair_times = tuple(
        sorted(
            pd.to_numeric(lr_tables["pair_timecourse"]["time"]).astype(float).unique()
        )
    )
    lr_ok = lr_ok and pair_times == expected_times
    retained = lr_tables["trajectory_coverage"].get("retained")
    lr_ok = (
        lr_ok
        and retained is not None
        and bool(retained.astype(str).str.lower().eq("true").any())
    )
    audit.check(
        lr_ok,
        "strict complete-trajectory LR dynamics",
        f"pair_rows={len(lr_tables['pair_timecourse'])}, celltype_rows={len(lr_tables['celltype_timecourse'])}",
    )

    expected_visuals = (
        downstream / "growth" / "growth_timepoint_grid.pdf",
        downstream / "composition" / "celltype_composition.pdf",
        downstream / "gene_dynamics" / "temporal_gene_programs.pdf",
        downstream / "figures" / "spatiotemporal_communication_3d.html",
    )
    visuals_ok = all(
        path.is_file() and path.stat().st_size > 0 for path in expected_visuals
    )
    audit.check(
        visuals_ok,
        "standard figure artifacts",
        ", ".join(path.name for path in expected_visuals),
    )


def validate_dataset(run_root: Path, dataset: str) -> Audit:
    audit = Audit(dataset)
    spec = DATASETS[dataset]
    run_dir = run_root / dataset
    paths = required_files(run_dir, dataset)
    missing = False
    for label, path in paths.items():
        present = path.is_file() and path.stat().st_size > 0
        audit.check(present, label, str(path))
        missing = missing or not present
    if missing:
        return audit

    n_cells, model_dim = validate_aligned(audit, paths, spec)
    audit.check(
        n_cells == sum(spec["observed_counts"].values()),
        "cohort count total",
        f"n_cells={n_cells}",
    )
    threshold = validate_edge_predictor(audit, run_dir, paths, spec)
    validate_training(audit, paths, spec, threshold, model_dim)
    validate_downstream(audit, run_dir, paths, spec, threshold, model_dim)
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=tuple(DATASETS),
        default=tuple(DATASETS),
        help="Datasets to validate (default: all four).",
    )
    parser.add_argument(
        "--report", type=Path, help="Optional human-readable JSON report."
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_root = args.run_root.expanduser().resolve()
    audits = []
    for dataset in args.datasets:
        print(f"\n=== {dataset} ===")
        try:
            audit = validate_dataset(run_root, dataset)
        except Exception as error:
            audit = Audit(dataset)
            audit.check(
                False, "acceptance execution", f"{type(error).__name__}: {error}"
            )
        audits.append(audit)
        for item in audit.checks:
            print(f"[{item['status']}] {item['name']}: {item['detail']}")
        for warning in audit.warnings:
            print(f"[WARN] {warning}")

    report = {
        "run_root": str(run_root),
        "status": "PASS" if all(not audit.errors for audit in audits) else "FAIL",
        "datasets": {audit.dataset: audit.as_dict() for audit in audits},
    }
    if args.report is not None:
        report_path = args.report.expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(f"\nReport: {report_path}")
    print(f"\nOverall: {report['status']}")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
