#!/usr/bin/env python3
"""Biology-first 24 hpf zebrafish Delta--Notch case study.

This workflow asks a deliberately narrow biological question: do model edges
between distinct cells in the ventral spinal-cord compartment that are
compatible with Delta (dla/dld) in the sender and notch1a/notch3 in the
receiver point towards a receiver state associated with the canonical her4
response program?

The analysis never calls attention ligand-receptor-specific.  Ligand and
receptor expression are an external, post-hoc filter on complete frozen
cell-pair messages.  No message is subtracted and no trajectory is rerun; this
is an ``LR-compatible exact-message direction audit``, not a perturbation or a
Delta/Notch knockout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Iterable, Sequence

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import sparse, stats
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold


LIGANDS = ("dla", "dld")
RECEPTORS = ("notch1a", "notch3")
RESPONSE_GENES = ("her4.1", "her4.3", "her4.4")
PROGENITOR_GENES = ("sox2", "pcna", "mki67")
PRONEURAL_GENES = ("neurog1", "ascl1a", "elavl3")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--h5ad", required=True, type=Path)
    result.add_argument("--attribution-dir", required=True, type=Path)
    result.add_argument(
        "--observed-cells",
        type=Path,
        help=(
            "Attribution observed_cells.csv.gz. Defaults to the file directly "
            "under --attribution-dir and is required for endpoint mapping."
        ),
    )
    result.add_argument("--circuit-screen", required=True, type=Path)
    result.add_argument("--output-dir", required=True, type=Path)
    result.add_argument("--stage", type=float, default=4.0)
    result.add_argument("--stage-label", default="24hpf")
    result.add_argument("--sender-type", default="Spinal Cord Ventral Region")
    result.add_argument("--receiver-type", default="Spinal Cord Ventral Region")
    result.add_argument("--spatial-key", default="spatial_aligned")
    result.add_argument("--state-key", default="X_latent")
    result.add_argument("--n-permutations", type=int, default=1000)
    result.add_argument("--seed", type=int, default=1731)
    result.add_argument("--overwrite", action="store_true")
    return result


def require(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    missing = [column for column in columns if column not in frame]
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")


def prepare_output(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(path)
        shutil.rmtree(path)
    path.mkdir(parents=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def integer_values(series: pd.Series, label: str) -> np.ndarray:
    values = pd.to_numeric(series, errors="raise").to_numpy(float)
    if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
        raise ValueError(f"{label} must contain finite integer values")
    return values.astype(int)


def resolve_observed_cells_path(
    attribution_dir: Path, supplied: Path | None
) -> Path:
    path = supplied if supplied is not None else attribution_dir / "observed_cells.csv.gz"
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            "A valid observed_cells.csv.gz is required to map attribution-global "
            f"edge indices into the supplied H5AD: {path}"
        )
    return path


def observed_global_to_h5ad(
    path: Path, data: ad.AnnData
) -> tuple[dict[int, int], dict[str, object]]:
    observed = pd.read_csv(path)
    require(
        observed,
        ["global_index", "obs_name", "stage", "stage_label", "cell_type"],
        "observed-cells table",
    )
    global_index = integer_values(observed["global_index"], "global_index")
    if len(np.unique(global_index)) != len(global_index):
        raise ValueError("observed-cells global_index is not unique")
    if set(global_index) != set(range(len(observed))):
        raise ValueError("observed-cells global_index must be a complete 0..N-1 grid")
    obs_names = observed["obs_name"].astype(str)
    if obs_names.duplicated().any():
        raise ValueError("observed-cells obs_name is not unique")
    if not data.obs_names.is_unique:
        raise ValueError("H5AD obs_names must be unique")
    if len(observed) != data.n_obs or set(obs_names) != set(data.obs_names.astype(str)):
        raise ValueError(
            "observed-cells and H5AD must contain exactly the same obs_name universe"
        )
    h5_lookup = {name: index for index, name in enumerate(data.obs_names.astype(str))}
    h5_index = obs_names.map(h5_lookup).to_numpy(int)
    observed_stage = pd.to_numeric(observed["stage"], errors="raise").to_numpy(float)
    h5_stage = pd.to_numeric(
        data.obs.iloc[h5_index]["time_point_processed"], errors="raise"
    ).to_numpy(float)
    if not np.isclose(observed_stage, h5_stage, rtol=0.0, atol=1e-12).all():
        raise ValueError("observed-cells stage disagrees with H5AD")
    observed_type = observed["cell_type"].astype(str).to_numpy()
    h5_type = data.obs.iloc[h5_index]["Annotation"].astype(str).to_numpy()
    if not np.array_equal(observed_type, h5_type):
        raise ValueError("observed-cells cell_type disagrees with H5AD Annotation")
    if "time" in data.obs:
        h5_label = data.obs.iloc[h5_index]["time"].astype(str).to_numpy()
        if not np.array_equal(observed["stage_label"].astype(str).to_numpy(), h5_label):
            raise ValueError("observed-cells stage_label disagrees with H5AD time")
    return dict(zip(global_index, h5_index)), {
        "mode": "observed_cells_obs_name_to_h5ad",
        "observed_cells": str(path),
        "observed_cells_sha256": sha256(path),
        "n_cells": int(len(observed)),
        "global_index_order_assumed_without_validation": False,
    }


def dense_columns(matrix, indices: Sequence[int]) -> np.ndarray:
    values = matrix[:, list(indices)]
    values = values.toarray() if sparse.issparse(values) else np.asarray(values)
    return np.asarray(values, dtype=np.float64)


def gene_matrix(data: ad.AnnData, genes: Iterable[str]) -> tuple[list[str], np.ndarray]:
    available = [gene for gene in genes if gene in data.var_names]
    if not available:
        raise ValueError(f"None of the requested genes are available: {tuple(genes)}")
    indices = [int(data.var_names.get_loc(gene)) for gene in available]
    return available, dense_columns(data.X, indices)


def q95_scale(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    positive = values[values > 0]
    scale = float(np.quantile(positive, 0.95)) if positive.size else 1.0
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    return np.clip(values / scale, 0.0, 1.0)


def module_score(data: ad.AnnData, genes: Sequence[str]) -> tuple[np.ndarray, list[str]]:
    available, matrix = gene_matrix(data, genes)
    scaled = np.column_stack([q95_scale(matrix[:, index]) for index in range(matrix.shape[1])])
    return scaled.mean(axis=1), available


def load_stage_edges(
    directory: Path,
    stage: float,
    stage_label: str,
    *,
    data: ad.AnnData,
    observed_cells: Path,
) -> tuple[pd.DataFrame, dict[int, np.ndarray], pd.DataFrame, dict[str, object]]:
    stage_dirs = sorted(directory.glob(f"stage_{stage:g}_*"))
    if len(stage_dirs) != 1:
        raise FileNotFoundError(
            f"Expected exactly one stage_{stage:g}_* directory under {directory}; "
            f"found {stage_dirs}"
        )
    stage_dir = stage_dirs[0]
    if stage_label not in stage_dir.name:
        raise ValueError(f"Stage label mismatch: expected {stage_label}, found {stage_dir.name}")
    frames: list[pd.DataFrame] = []
    arrays: dict[int, np.ndarray] = {}
    inventories: list[dict[str, object]] = []
    mapping, resolution = observed_global_to_h5ad(observed_cells, data)
    h5_stage = pd.to_numeric(
        data.obs["time_point_processed"], errors="raise"
    ).to_numpy(float)
    h5_type = data.obs["Annotation"].astype(str).to_numpy()
    for edge_path in sorted(stage_dir.glob("edges_seed_*.csv.gz")):
        frame = pd.read_csv(edge_path)
        require(
            frame,
            [
                "grouping_seed",
                "stage",
                "stage_label",
                "source_index",
                "target_index",
                "sender_type",
                "receiver_type",
                "attention_abs_mean",
                "edge_message_norm_joint",
                "spatial_distance",
                "edge_predictor_probability",
                "source_mass_fraction",
            ],
            str(edge_path),
        )
        seed_values = frame["grouping_seed"].drop_duplicates().tolist()
        if len(seed_values) != 1:
            raise ValueError(f"Ambiguous grouping seed in {edge_path}: {seed_values}")
        seed = int(seed_values[0])
        if seed in arrays:
            raise ValueError(f"Duplicate grouping-seed edge table: {seed}")
        if not np.isclose(
            pd.to_numeric(frame["stage"], errors="raise").to_numpy(float),
            float(stage),
            rtol=0.0,
            atol=1e-12,
        ).all():
            raise ValueError(f"Unexpected stage in {edge_path}")
        if set(frame["stage_label"].astype(str)) != {str(stage_label)}:
            raise ValueError(f"Unexpected stage label in {edge_path}")
        array_path = stage_dir / f"exact_arrays_seed_{seed}.npz"
        with np.load(array_path) as payload:
            edge_output = np.asarray(payload["edge_output"], dtype=np.float64)
            array_global = (
                np.asarray(payload["global_indices"], dtype=int)
                if "global_indices" in payload
                else None
            )
        if edge_output.shape[0] != len(frame):
            raise ValueError(
                f"Edge/array row mismatch for seed {seed}: {len(frame)} vs "
                f"{edge_output.shape[0]}"
            )
        if frame["source_index"].eq(frame["target_index"]).any():
            raise ValueError(
                f"Self edge found in {edge_path}; distinct-cell semantics failed"
            )
        source_attribution = integer_values(frame["source_index"], "source_index")
        target_attribution = integer_values(frame["target_index"], "target_index")
        missing = sorted(
            (set(source_attribution) | set(target_attribution)).difference(mapping)
        )
        if missing:
            raise ValueError(
                f"Edge endpoints are absent from observed-cells: {missing[:5]}"
            )
        source = np.asarray([mapping[value] for value in source_attribution], dtype=int)
        target = np.asarray([mapping[value] for value in target_attribution], dtype=int)
        if not np.isclose(h5_stage[source], float(stage), rtol=0.0, atol=1e-12).all() or not np.isclose(
            h5_stage[target], float(stage), rtol=0.0, atol=1e-12
        ).all():
            raise ValueError(f"Mapped edge endpoint is outside stage {stage:g}: {edge_path}")
        if not np.array_equal(frame["sender_type"].astype(str).to_numpy(), h5_type[source]):
            raise ValueError(f"sender_type disagrees with mapped H5AD cells: {edge_path}")
        if not np.array_equal(frame["receiver_type"].astype(str).to_numpy(), h5_type[target]):
            raise ValueError(f"receiver_type disagrees with mapped H5AD cells: {edge_path}")
        if array_global is not None:
            global_to_stage = {int(value): index for index, value in enumerate(array_global)}
            for role, attribution in (
                ("source", source_attribution),
                ("target", target_attribution),
            ):
                missing_array = sorted(set(attribution).difference(global_to_stage))
                if missing_array:
                    raise ValueError(
                        f"{role} endpoints absent from exact-array global_indices: "
                        f"{missing_array[:5]}"
                    )
                local_column = f"{role}_index_stage"
                if local_column in frame:
                    expected_local = np.asarray(
                        [global_to_stage[value] for value in attribution], dtype=int
                    )
                    if not np.array_equal(
                        integer_values(frame[local_column], local_column), expected_local
                    ):
                        raise ValueError(
                            f"{local_column} disagrees with exact-array global_indices"
                        )
        frame = frame.copy()
        frame["source_index_attribution"] = source_attribution
        frame["target_index_attribution"] = target_attribution
        frame["source_index"] = source
        frame["target_index"] = target
        frame["edge_row_within_seed"] = np.arange(len(frame), dtype=int)
        frames.append(frame)
        arrays[seed] = edge_output
        inventories.append(
            {
                "grouping_seed": seed,
                "n_rows": int(len(frame)),
                "edge_path": str(edge_path.resolve()),
                "edge_sha256": sha256(edge_path),
                "exact_array_path": str(array_path.resolve()),
                "exact_array_sha256": sha256(array_path),
                "index_mapping": resolution["mode"],
            }
        )
    if not frames:
        raise FileNotFoundError(f"No edge tables in {stage_dir}")
    return (
        pd.concat(frames, ignore_index=True),
        arrays,
        pd.DataFrame(inventories),
        resolution,
    )


def crossfit_response_direction(
    state: np.ndarray,
    response: np.ndarray,
    receiver_indices: np.ndarray,
    *,
    seed: int,
    n_splits: int = 5,
) -> tuple[pd.DataFrame, dict[int, tuple[np.ndarray, float]]]:
    """Cross-fit a latent direction associated with the observed response module.

    Returned coefficients are in the original PCA coordinate system, so a
    complete state-space edge message can be projected with a dot product.
    """
    x = np.asarray(state[receiver_indices], dtype=np.float64)
    y = np.asarray(response[receiver_indices], dtype=np.float64)
    fold_rows: list[pd.DataFrame] = []
    directions: dict[int, tuple[np.ndarray, float]] = {}
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=int(seed))
    for fold, (train, test) in enumerate(splitter.split(x)):
        mean = x[train].mean(axis=0)
        scale = x[train].std(axis=0, ddof=0)
        scale[~np.isfinite(scale) | (scale <= 1e-12)] = 1.0
        x_train = (x[train] - mean) / scale
        model = RidgeCV(alphas=np.asarray([0.1, 1.0, 10.0, 100.0]))
        model.fit(x_train, y[train])
        coefficient = np.asarray(model.coef_, dtype=float) / scale
        intercept = float(model.intercept_ - np.dot(mean, coefficient))
        prediction = x[test] @ coefficient + intercept
        directions[fold] = (coefficient, intercept)
        fold_rows.append(
            pd.DataFrame(
                {
                    "global_index": receiver_indices[test],
                    "fold": fold,
                    "observed_response": y[test],
                    "predicted_response": prediction,
                    "ridge_alpha": float(model.alpha_),
                }
            )
        )
    result = pd.concat(fold_rows, ignore_index=True).sort_values("global_index")
    return result, directions


def annotate_edges(
    edges: pd.DataFrame,
    arrays: dict[int, np.ndarray],
    *,
    ligand_activity: np.ndarray,
    receptor_activity: np.ndarray,
    receiver_fold: dict[int, int],
    directions: dict[int, tuple[np.ndarray, float]],
    spatial_dim: int = 2,
) -> pd.DataFrame:
    result = edges.copy()
    source = result["source_index"].to_numpy(int)
    target = result["target_index"].to_numpy(int)
    result["delta_activity"] = ligand_activity[source]
    result["notch_receptor_activity"] = receptor_activity[target]
    result["lr_activity"] = (
        result["delta_activity"] * result["notch_receptor_activity"]
    )
    result["attention_lr"] = (
        result["attention_abs_mean"].to_numpy(float)
        * result["lr_activity"].to_numpy(float)
    )
    result["message_lr"] = (
        result["edge_message_norm_joint"].to_numpy(float)
        * result["lr_activity"].to_numpy(float)
    )
    result["receiver_fold"] = [receiver_fold.get(int(index), -1) for index in target]
    projection = np.full(len(result), np.nan, dtype=float)
    for seed, positions in result.groupby("grouping_seed", sort=False).groups.items():
        positions = np.asarray(list(positions), dtype=int)
        rows = result.loc[positions, "edge_row_within_seed"].to_numpy(int)
        message = arrays[int(seed)][rows, spatial_dim:]
        folds = result.loc[positions, "receiver_fold"].to_numpy(int)
        for fold in np.unique(folds[folds >= 0]):
            local = folds == fold
            coefficient, _ = directions[int(fold)]
            projection[positions[local]] = message[local] @ coefficient
    result["response_direction_projection"] = projection
    result["lr_weighted_response_projection"] = (
        result["lr_activity"].to_numpy(float) * projection
    )
    return result


def aggregate_receiver_scores(
    annotated: pd.DataFrame,
    receiver_indices: np.ndarray,
    *,
    sender_type: str,
    receiver_type: str,
) -> pd.DataFrame:
    local = annotated.loc[
        annotated["sender_type"].eq(sender_type)
        & annotated["receiver_type"].eq(receiver_type)
    ].copy()
    grouped = (
        local.groupby(["grouping_seed", "target_index"], observed=True)
        .agg(
            attention_lr=("attention_lr", "sum"),
            lr_only=("lr_activity", "sum"),
            attention_only=("attention_abs_mean", "sum"),
            exact_message_lr=("message_lr", "sum"),
            response_projection=("lr_weighted_response_projection", "sum"),
            compatible_edge_count=("lr_activity", lambda value: int(np.sum(value > 0))),
            incoming_edge_count=("source_index", "size"),
        )
        .reset_index()
    )
    seeds = sorted(annotated["grouping_seed"].astype(int).unique())
    complete = pd.MultiIndex.from_product(
        [seeds, receiver_indices], names=["grouping_seed", "target_index"]
    ).to_frame(index=False)
    grouped = complete.merge(
        grouped, on=["grouping_seed", "target_index"], how="left", validate="one_to_one"
    ).fillna(0.0)
    result = (
        grouped.groupby("target_index", observed=True, as_index=False)
        .agg(
            attention_lr=("attention_lr", "mean"),
            lr_only=("lr_only", "mean"),
            attention_only=("attention_only", "mean"),
            exact_message_lr=("exact_message_lr", "mean"),
            response_projection=("response_projection", "mean"),
            compatible_edge_count=("compatible_edge_count", "mean"),
            incoming_edge_count=("incoming_edge_count", "mean"),
        )
        .rename(columns={"target_index": "global_index"})
    )
    return result


def partial_rank_correlation(
    x: np.ndarray, y: np.ndarray, covariates: np.ndarray
) -> tuple[float, float]:
    values = np.column_stack((x, y, covariates)).astype(float)
    mask = np.isfinite(values).all(axis=1)
    values = values[mask]
    ranked = np.column_stack(
        [stats.rankdata(values[:, index]) for index in range(values.shape[1])]
    )
    design = np.column_stack((np.ones(len(ranked)), ranked[:, 2:]))
    residuals: list[np.ndarray] = []
    for index in (0, 1):
        coefficient, *_ = np.linalg.lstsq(design, ranked[:, index], rcond=None)
        residuals.append(ranked[:, index] - design @ coefficient)
    correlation, p_value = stats.pearsonr(residuals[0], residuals[1])
    return float(correlation), float(p_value)


def matched_projection_null(
    annotated: pd.DataFrame,
    *,
    sender_type: str,
    receiver_type: str,
    n_permutations: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, float], pd.DataFrame]:
    """Exploratory receiver-clustered contrast for complete message direction.

    Technical grouping-seed repeats are first collapsed to one directed
    source--target edge.  Each LR-compatible edge is matched to
    non-LR-compatible edges reaching the same receiver, and resampled control
    means are aggregated first within receiver and then equally across
    receivers.  The returned tail fraction is an exploratory matched-resampling
    diagnostic, not a confirmatory P value: the case was selected post hoc and
    receiver cells are not biological replicates.
    """
    if int(n_permutations) < 2:
        raise ValueError("n_permutations must be at least 2")
    local = annotated.loc[
        annotated["sender_type"].eq(sender_type)
        & annotated["receiver_type"].eq(receiver_type)
        & annotated["response_direction_projection"].notna()
    ].copy()
    require(
        local,
        [
            "source_index",
            "target_index",
            "grouping_seed",
            "lr_activity",
            "response_direction_projection",
        ],
        "annotated exact edges",
    )
    covariates = [
        "spatial_distance",
        "edge_predictor_probability",
        "source_mass_fraction",
        "edge_message_norm_joint",
    ]
    edge_key = ["source_index", "target_index", "sender_type", "receiver_type"]
    unique_edges = (
        local.groupby(edge_key, observed=True, as_index=False)
        .agg(
            technical_seed_support=("grouping_seed", "nunique"),
            n_technical_occurrences=("grouping_seed", "size"),
            lr_activity=("lr_activity", "mean"),
            response_direction_projection=("response_direction_projection", "mean"),
            spatial_distance=("spatial_distance", "mean"),
            edge_predictor_probability=("edge_predictor_probability", "mean"),
            source_mass_fraction=("source_mass_fraction", "mean"),
            edge_message_norm_joint=("edge_message_norm_joint", "mean"),
        )
    )
    lr_state_counts = local.assign(_lr_positive=local["lr_activity"].gt(0)).groupby(
        edge_key, observed=True
    )["_lr_positive"].nunique()
    if (lr_state_counts > 1).any():
        raise ValueError("LR compatibility changed across technical grouping seeds")
    selected = unique_edges.loc[unique_edges["lr_activity"].gt(0)].copy()
    if selected.empty:
        raise ValueError("No LR-compatible exact edges for the requested circuit")
    receiver_pools: dict[int, list[np.ndarray]] = {}
    receiver_selected: dict[int, list[float]] = {}
    dropped = 0
    for target_index, selected_group in selected.groupby("target_index", observed=True):
        universe = unique_edges.loc[
            unique_edges["target_index"].eq(target_index)
            & unique_edges["lr_activity"].le(0)
        ].copy()
        if universe.empty:
            dropped += int(len(selected_group))
            continue
        pool_covariates = universe[covariates].to_numpy(float)
        scale = np.nanstd(
            np.vstack((selected_group[covariates].to_numpy(float), pool_covariates)),
            axis=0,
        )
        scale[~np.isfinite(scale) | (scale <= 1e-12)] = 1.0
        pool_covariates = pool_covariates / scale
        pool_projection = universe["response_direction_projection"].to_numpy(float)
        for row in selected_group.itertuples(index=False):
            query = np.asarray([getattr(row, column) for column in covariates]) / scale
            distance = np.sum(np.square(pool_covariates - query), axis=1)
            nearest = np.argsort(distance, kind="stable")[: min(10, len(distance))]
            receiver_pools.setdefault(int(target_index), []).append(
                pool_projection[nearest]
            )
            receiver_selected.setdefault(int(target_index), []).append(
                float(row.response_direction_projection)
            )
    if not receiver_pools:
        raise ValueError("No receiver-matched non-LR-compatible control edges")
    receivers = sorted(receiver_pools)
    observed_by_receiver = np.asarray(
        [np.mean(receiver_selected[index]) for index in receivers], dtype=float
    )
    expected_control_by_receiver = np.asarray(
        [
            np.mean([np.mean(pool) for pool in receiver_pools[index]])
            for index in receivers
        ],
        dtype=float,
    )
    observed = float(np.mean(observed_by_receiver))
    rng = np.random.default_rng(int(seed))
    null = np.empty(int(n_permutations), dtype=float)
    for iteration in range(int(n_permutations)):
        null[iteration] = float(
            np.mean(
                [
                    np.mean(
                        [
                            rng.choice(pool, size=1, replace=True)[0]
                            for pool in receiver_pools[index]
                        ]
                    )
                    for index in receivers
                ]
            )
        )
    tail_upper = float((1 + np.sum(null >= observed)) / (len(null) + 1))
    tail_lower = float((1 + np.sum(null <= observed)) / (len(null) + 1))
    matched_unique = int(sum(len(values) for values in receiver_selected.values()))
    selected_occurrences = int(local["lr_activity"].gt(0).sum())
    receiver_summary = pd.DataFrame(
        {
            "target_index": receivers,
            "n_matched_lr_compatible_unique_edges": [
                len(receiver_selected[index]) for index in receivers
            ],
            "selected_mean_projection": observed_by_receiver,
            "mean_of_matched_control_pool_means": expected_control_by_receiver,
        }
    )
    receiver_summary["selected_minus_matched_pool_mean"] = (
        receiver_summary["selected_mean_projection"]
        - receiver_summary["mean_of_matched_control_pool_means"]
    )
    summary = {
        "observed_mean_projection": observed,
        "null_mean": float(null.mean()),
        "null_sd": float(null.std(ddof=1)),
        "exploratory_tail_fraction_greater": tail_upper,
        "exploratory_tail_fraction_less": tail_lower,
        "n_lr_compatible_edge_occurrences": selected_occurrences,
        "n_lr_compatible_unique_edges": int(len(selected)),
        "n_lr_compatible_unique_edges_matched": matched_unique,
        "n_lr_compatible_unique_edges_dropped_no_control": int(dropped),
        "n_matched_receiver_clusters": int(len(receivers)),
        "n_all_same_circuit_edge_occurrences": int(len(local)),
        "n_all_same_circuit_unique_edges": int(len(unique_edges)),
        "observed_minus_null_mean": observed - float(null.mean()),
    }
    return pd.DataFrame({"null_mean_projection": null}), summary, receiver_summary


def circuit_rank_table(
    path: Path, *, stage: float, sender_type: str, receiver_type: str
) -> pd.DataFrame:
    screen = pd.read_csv(path)
    require(
        screen,
        [
            "stage",
            "axis_id",
            "sender_type",
            "receiver_type",
            "attention_context_percentile",
            "exact_message_context_percentile",
            "commot_context_percentile",
        ],
        "circuit screen",
    )
    axes = [f"{ligand}->{receptor}" for ligand in LIGANDS for receptor in RECEPTORS]
    result = screen.loc[
        np.isclose(screen["stage"].to_numpy(float), float(stage))
        & screen["axis_id"].isin(axes)
        & screen["sender_type"].eq(sender_type)
        & screen["receiver_type"].eq(receiver_type)
    ].copy()
    if len(result) != len(axes):
        raise ValueError(f"Expected {len(axes)} exact circuit rows, found {len(result)}")
    return result.sort_values("axis_id").reset_index(drop=True)


def gene_group_summary(
    data: ad.AnnData, receiver_indices: np.ndarray, scores: pd.DataFrame
) -> pd.DataFrame:
    genes = [*RESPONSE_GENES, *PROGENITOR_GENES, *PRONEURAL_GENES]
    available, matrix = gene_matrix(data, genes)
    local_scores = scores.set_index("global_index").loc[receiver_indices, "attention_lr"]
    incoming = local_scores.to_numpy(float)
    order = np.argsort(incoming, kind="stable")
    quartile_size = max(1, int(math.floor(len(order) * 0.25)))
    group = np.full(len(order), "Middle", dtype=object)
    group[order[:quartile_size]] = "Low incoming"
    group[order[-quartile_size:]] = "High incoming"
    rows: list[dict[str, object]] = []
    for index, gene in enumerate(available):
        values = matrix[receiver_indices, index]
        for label in ("High incoming", "Low incoming"):
            subset = values[group == label]
            rows.append(
                {
                    "gene": gene,
                    "program": (
                        "Notch response" if gene in RESPONSE_GENES else
                        "Progenitor" if gene in PROGENITOR_GENES else
                        "Proneural / neuronal"
                    ),
                    "group": label,
                    "n_cells": int(len(subset)),
                    "mean_expression": float(np.mean(subset)),
                    "positive_fraction": float(np.mean(subset > 0)),
                }
            )
    return pd.DataFrame(rows)


def _rank_percent(value: float) -> str:
    return f"top {(1.0 - value) * 100:.1f}%"


def plot_story(
    data: ad.AnnData,
    stage_mask: np.ndarray,
    receiver_indices: np.ndarray,
    cell_scores: pd.DataFrame,
    gene_summary: pd.DataFrame,
    rank_table: pd.DataFrame,
    null: pd.DataFrame,
    null_summary: dict[str, float],
    statistics: dict[str, float],
    *,
    spatial_key: str,
    output: Path,
) -> None:
    sns.set_theme(style="white", context="talk")
    coordinates = np.asarray(data.obsm[spatial_key], dtype=float)
    stage_indices = np.flatnonzero(stage_mask)
    receiver = cell_scores.set_index("global_index").loc[receiver_indices]
    response = receiver["notch_response"].to_numpy(float)
    incoming = receiver["attention_lr"].to_numpy(float)

    figure = plt.figure(figsize=(17, 11), constrained_layout=True)
    grid = figure.add_gridspec(2, 3, width_ratios=[1.05, 1.05, 1.0])
    axes = [figure.add_subplot(grid[row, column]) for row in range(2) for column in range(3)]

    ax = axes[0]
    ax.scatter(coordinates[stage_indices, 0], coordinates[stage_indices, 1], s=3, c="#d7d7d7", alpha=0.45, rasterized=True)
    ax.scatter(coordinates[receiver_indices, 0], coordinates[receiver_indices, 1], s=11, c="#2a6fbb", alpha=0.9, rasterized=True)
    ax.set_title("A  Anatomical context")
    ax.text(0.02, 0.02, f"24 hpf observed cells: {len(stage_indices):,}\nventral spinal cells: {len(receiver_indices):,}", transform=ax.transAxes, fontsize=11, va="bottom")
    ax.set_aspect("equal")
    ax.axis("off")

    ax = axes[1]
    ax.scatter(coordinates[stage_indices, 0], coordinates[stage_indices, 1], s=2, c="#ececec", alpha=0.28, rasterized=True)
    points = ax.scatter(coordinates[receiver_indices, 0], coordinates[receiver_indices, 1], s=16, c=incoming, cmap="magma", rasterized=True)
    figure.colorbar(points, ax=ax, shrink=0.72, label="incoming attention × Delta/Notch compatibility")
    ax.set_title("B  Where the candidate circuit is received")
    ax.set_aspect("equal")
    ax.axis("off")

    ax = axes[2]
    ax.scatter(coordinates[stage_indices, 0], coordinates[stage_indices, 1], s=2, c="#ececec", alpha=0.28, rasterized=True)
    points = ax.scatter(coordinates[receiver_indices, 0], coordinates[receiver_indices, 1], s=16, c=response, cmap="viridis", rasterized=True)
    figure.colorbar(points, ax=ax, shrink=0.72, label="her4 response module")
    ax.set_title("C  Same cells, observed receiver response")
    ax.set_aspect("equal")
    ax.axis("off")

    ax = axes[3]
    ax.scatter(incoming, response, s=22, c="#335c81", alpha=0.55, edgecolor="none", rasterized=True)
    if np.ptp(incoming) > 0:
        slope, intercept = np.polyfit(incoming, response, 1)
        line = np.linspace(incoming.min(), incoming.max(), 100)
        ax.plot(line, slope * line + intercept, color="#d1495b", linewidth=2.5)
    ax.set_xlabel("incoming attention × LR compatibility")
    ax.set_ylabel("observed her4 module")
    ax.set_title("D  Cell-resolved downstream coherence")
    ax.text(0.03, 0.97, f"Spearman ρ={statistics['attention_response_spearman_rho']:.2f}\npartial ρ={statistics['attention_response_partial_rho']:.2f}", transform=ax.transAxes, va="top", fontsize=12)

    ax = axes[4]
    plot_frame = gene_summary.pivot(index=["gene", "program"], columns="group", values="mean_expression").reset_index()
    plot_frame["difference"] = plot_frame["High incoming"] - plot_frame["Low incoming"]
    colors = plot_frame["program"].map({"Notch response": "#7b2cbf", "Progenitor": "#2a9d8f", "Proneural / neuronal": "#e76f51"})
    ax.barh(plot_frame["gene"], plot_frame["difference"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("mean expression: high − low incoming quartile")
    ax.set_title("E  Which biological genes differ?")
    ax.invert_yaxis()

    ax = axes[5]
    sns.histplot(null["null_mean_projection"], bins=35, color="#b8b8b8", ax=ax)
    observed = null_summary["observed_mean_projection"]
    ax.axvline(observed, color="#d1495b", linewidth=3, label="Delta/Notch-compatible edges")
    ax.axvline(null_summary["null_mean"], color="#4d4d4d", linestyle="--", linewidth=2, label="receiver-clustered matched reference")
    ax.set_xlabel("message projection toward her4-paralog latent direction")
    ax.set_title("F  Exploratory exact-message direction contrast")
    ax.legend(frameon=False, fontsize=10)

    rank_lines = []
    for row in rank_table.itertuples(index=False):
        rank_lines.append(
            f"{row.axis_id}: CB {_rank_percent(row.attention_context_percentile)}, "
            f"COMMOT {_rank_percent(row.commot_context_percentile)}"
        )
    figure.suptitle(
        "24 hpf zebrafish ventral spinal Delta–Notch: from exact cell circuit to receiver program\n"
        + " | ".join(rank_lines),
        fontsize=19,
        fontweight="bold",
    )
    figure.savefig(output, dpi=240, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parser().parse_args()
    prepare_output(args.output_dir, args.overwrite)
    tables = args.output_dir / "tables"
    figures = args.output_dir / "figures"
    tables.mkdir()
    figures.mkdir()

    data = ad.read_h5ad(args.h5ad)
    for key in ("time_point_processed", "Annotation"):
        if key not in data.obs:
            raise KeyError(f"Missing adata.obs[{key!r}]")
    for key in (args.spatial_key, args.state_key):
        if key not in data.obsm:
            raise KeyError(f"Missing adata.obsm[{key!r}]")
    observed_cells = resolve_observed_cells_path(
        args.attribution_dir, args.observed_cells
    )
    stage_values = pd.to_numeric(data.obs["time_point_processed"], errors="raise").to_numpy(float)
    labels = data.obs["Annotation"].astype(str).to_numpy()
    stage_mask = np.isclose(stage_values, args.stage, rtol=0.0, atol=1e-12)
    receiver_indices = np.flatnonzero(stage_mask & (labels == args.receiver_type))
    if receiver_indices.size < 20:
        raise ValueError(f"Too few receiver cells: {receiver_indices.size}")

    ligand_genes, ligand_matrix = gene_matrix(data, LIGANDS)
    ligand_activity = np.max(
        np.column_stack([q95_scale(ligand_matrix[:, index]) for index in range(ligand_matrix.shape[1])]),
        axis=1,
    )
    receptor_genes, receptor_matrix = gene_matrix(data, RECEPTORS)
    receptor_activity = np.max(
        np.column_stack(
            [
                q95_scale(receptor_matrix[:, index])
                for index in range(receptor_matrix.shape[1])
            ]
        ),
        axis=1,
    )
    response, response_genes = module_score(data, RESPONSE_GENES)
    progenitor, progenitor_genes = module_score(data, PROGENITOR_GENES)
    proneural, proneural_genes = module_score(data, PRONEURAL_GENES)

    state = np.asarray(data.obsm[args.state_key], dtype=np.float64)
    crossfit, directions = crossfit_response_direction(
        state, response, receiver_indices, seed=args.seed
    )
    receiver_fold = crossfit.set_index("global_index")["fold"].astype(int).to_dict()
    edges, arrays, edge_inventory, index_resolution = load_stage_edges(
        args.attribution_dir,
        args.stage,
        args.stage_label,
        data=data,
        observed_cells=observed_cells,
    )
    annotated = annotate_edges(
        edges,
        arrays,
        ligand_activity=ligand_activity,
        receptor_activity=receptor_activity,
        receiver_fold=receiver_fold,
        directions=directions,
    )
    cell_scores = aggregate_receiver_scores(
        annotated,
        receiver_indices,
        sender_type=args.sender_type,
        receiver_type=args.receiver_type,
    )
    cell_scores["notch_response"] = response[cell_scores["global_index"].to_numpy(int)]
    cell_scores["progenitor_program"] = progenitor[cell_scores["global_index"].to_numpy(int)]
    cell_scores["proneural_program"] = proneural[cell_scores["global_index"].to_numpy(int)]
    cell_scores["notch_receptor_scaled_expression"] = receptor_activity[
        cell_scores["global_index"].to_numpy(int)
    ]
    cell_scores = cell_scores.merge(crossfit, on="global_index", how="left", validate="one_to_one")

    rho_attention, p_attention = stats.spearmanr(cell_scores["attention_lr"], cell_scores["notch_response"])
    rho_lr, p_lr = stats.spearmanr(cell_scores["lr_only"], cell_scores["notch_response"])
    rho_crossfit, p_crossfit = stats.spearmanr(cell_scores["predicted_response"], cell_scores["observed_response"])
    partial_rho, partial_p = partial_rank_correlation(
        cell_scores["attention_lr"].to_numpy(float),
        cell_scores["notch_response"].to_numpy(float),
        cell_scores[
            ["lr_only", "notch_receptor_scaled_expression", "incoming_edge_count"]
        ].to_numpy(float),
    )
    statistics = {
        "attention_response_spearman_rho": float(rho_attention),
        "attention_response_spearman_p": float(p_attention),
        "lr_only_response_spearman_rho": float(rho_lr),
        "lr_only_response_spearman_p": float(p_lr),
        "attention_response_partial_rho": partial_rho,
        "attention_response_partial_p": partial_p,
        "crossfit_latent_response_spearman_rho": float(rho_crossfit),
        "crossfit_latent_response_spearman_p": float(p_crossfit),
    }
    null, null_summary, receiver_null_summary = matched_projection_null(
        annotated,
        sender_type=args.sender_type,
        receiver_type=args.receiver_type,
        n_permutations=args.n_permutations,
        seed=args.seed + 1,
    )
    rank_table = circuit_rank_table(
        args.circuit_screen,
        stage=args.stage,
        sender_type=args.sender_type,
        receiver_type=args.receiver_type,
    )
    gene_summary = gene_group_summary(data, receiver_indices, cell_scores)

    cell_scores.to_csv(tables / "receiver_cell_scores.csv", index=False)
    crossfit.to_csv(tables / "crossfit_response_direction.csv", index=False)
    rank_table.to_csv(tables / "exact_circuit_method_support.csv", index=False)
    gene_summary.to_csv(tables / "high_vs_low_incoming_gene_summary.csv", index=False)
    null.to_csv(tables / "receiver_matched_projection_null.csv.gz", index=False)
    receiver_null_summary.to_csv(
        tables / "receiver_cluster_matched_projection_summary.csv", index=False
    )
    edge_inventory.to_csv(tables / "edge_exact_input_inventory.csv", index=False)
    pd.DataFrame([statistics | null_summary]).to_csv(tables / "case_study_statistics.csv", index=False)

    plot_story(
        data,
        stage_mask,
        receiver_indices,
        cell_scores,
        gene_summary,
        rank_table,
        null,
        null_summary,
        statistics,
        spatial_key=args.spatial_key,
        output=figures / "delta_notch_biology_story.png",
    )

    positive_projection = null_summary["observed_mean_projection"] > null_summary["null_mean"]
    report = f"""# 24 hpf zebrafish Delta–Notch biology-first case study

## Biological question

Do complete CytoBridge messages on distinct cell pairs within the ventral spinal-cord compartment, selected only because the sender expresses `dla` or `dld` and the receiver expresses `notch1a` or `notch3`, align with an observed `her4` receiver-state program?

## What the figure shows

- Panel A locates the {len(receiver_indices):,} ventral spinal-cord cells in the observed 24 hpf spatial coordinates.
- Panel B shows the post-hoc incoming score `|attention| × max(dla,dld) × max(notch1a,notch3)` on those cells.
- Panel C shows the observed response module from {', '.join(response_genes)} in the same cells.
- Panel D compares the two cell by cell. Spearman rho is {rho_attention:.3f}; after rank-residualizing the LR-only score, receptor expression and incoming edge count, partial rho is {partial_rho:.3f}.
- Panel E shows gene expression differences between the top and bottom incoming-score quartiles. The plotted modules were fixed before looking at the result: Notch response ({', '.join(response_genes)}), neural progenitor ({', '.join(progenitor_genes)}), and proneural/neuronal ({', '.join(proneural_genes)}).
- Panel F projects each complete exact state-space edge message onto a cross-fitted latent direction associated with a separate `her4.1`/`her4.3`/`her4.4` module. Technical grouping-seed repeats are collapsed to unique directed edges before matching, and resampled controls are averaged within receiver before receivers are weighted equally. The compatible-edge receiver mean is {null_summary['observed_mean_projection']:.6g}, versus {null_summary['null_mean']:.6g} for the matched resampling reference (exploratory upper-tail fraction={null_summary['exploratory_tail_fraction_greater']:.4g}). {null_summary['n_lr_compatible_unique_edges_matched']:,} of {null_summary['n_lr_compatible_unique_edges']:,} compatible unique edges across {null_summary['n_matched_receiver_clusters']:,} receiver cells had an eligible non-compatible control. The observed direction is {'more positive than' if positive_projection else 'not more positive than'} the matched reference.

## What can and cannot be concluded

This is a biology-focused consistency chain: a literature-defined developmental context, an exact directed cell compartment, four Delta–Notch family axes, an observed receiver response module, and a matched exact-message direction audit. It is stronger than a pathway-level correlation alone. However, the direct attention-weighted incoming score does not explain the observed `her4` module better than the LR-only expression score in this compartment; that negative result is retained rather than converted into an attention-specific claim.

It is **not** a ligand-specific model knockout or intervention. The model attention and exact message are generic cell-pair quantities, and ligand/receptor expression is applied post hoc. No message was deleted and no trajectory was rerun. The {edges['grouping_seed'].nunique()} grouping seeds measure technical grouping sensitivity, not independent biological replication. The matched-resampling tail fraction is exploratory because this biological case was selected after inspecting the same dataset and neither unique edges nor receiver cells are biological replicates. Existing zebrafish perturbation experiments provide orthogonal biological plausibility; this dataset itself remains observational.
"""
    (args.output_dir / "README.md").write_text(report, encoding="utf-8")
    manifest = {
        "schema_version": 2,
        "analysis": "zebrafish_24hpf_delta_notch_biology_first_case_study",
        "status": "complete",
        "inputs": {
            "h5ad": str(args.h5ad.resolve()),
            "attribution_dir": str(args.attribution_dir.resolve()),
            "observed_cells": str(observed_cells),
            "circuit_screen": str(args.circuit_screen.resolve()),
        },
        "parameters": {
            "spatial_key": str(args.spatial_key),
            "state_key": str(args.state_key),
            "random_seed": int(args.seed),
            "matched_resampling_seed": int(args.seed + 1),
            "n_permutations": int(args.n_permutations),
            "crossfit_folds": 5,
        },
        "index_resolution": index_resolution,
        "edge_exact_input_inventory": edge_inventory.to_dict("records"),
        "case": {
            "stage": args.stage,
            "stage_label": args.stage_label,
            "sender_type": args.sender_type,
            "receiver_type": args.receiver_type,
            "ligands": ligand_genes,
            "receptors": receptor_genes,
            "response_genes": response_genes,
        },
        "response_modules": {
            "exact_message_direction_module": response_genes,
            "family_audit_compact_module_is_separate": True,
            "family_audit_compact_module": ["her4.1", "her6"],
        },
        "guardrails": {
            "attention_is_lr_specific": False,
            "exact_message_is_lr_specific": False,
            "analysis_is_a_ligand_knockout": False,
            "analysis_is_a_perturbation": False,
            "messages_were_deleted": False,
            "grouping_seeds_are_biological_replicates": False,
            "unique_directed_edges_are_biological_replicates": False,
            "receiver_cells_are_biological_replicates": False,
            "matched_resampling_is_confirmatory_p_value": False,
            "case_selection_was_preregistered": False,
            "post_selection_adjustment_was_applied": False,
            "global_h5ad_row_order_was_silently_assumed": False,
            "crossfit_response_direction_is_external_validation": False,
        },
        "statistics": statistics | null_summary,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest["statistics"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
