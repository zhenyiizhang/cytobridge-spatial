#!/usr/bin/env python3
"""Make reader-facing Delta--Notch biology and control figures.

The main 2 x 3 figure follows a biological story: anatomical location,
ligand/receptor expression, real recurrent cell-pair edges, exact-circuit
agreement with COMMOT, and an explicit claim ladder.  A separate controls
figure shows what is driven by LR abundance, what attention adds beyond that
baseline, and the strict matched exact-message null.

All quantities are observational.  Ligand/receptor expression is a post-hoc
compatibility filter on generic CytoBridge cell-pair quantities; neither figure
represents an LR-specific knockout, message deletion, or trajectory rerun.

Example
-------
python scripts/reviewer_zebrafish_ccc/make_delta_notch_biology_figure.py \
  --h5ad /path/to/zebrafish_aligned.h5ad \
  --stage4-edge-dir /path/to/01_cytobridge/stage_4_24hpf \
  --delta-notch-family-audit /path/to/formal_family_audit \
  --four-axis-circuit-screen /path/to/known_axis_exact_circuit_screen.csv.gz \
  --delta-notch-case-study /path/to/delta_notch_case_study \
  --output-dir /path/to/delta_notch_reader_figure
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import anndata as ad
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np
import pandas as pd
from scipy import sparse


PAIR_FAMILY = (
    ("dla", "notch1a"),
    ("dla", "notch3"),
    ("dld", "notch1a"),
    ("dld", "notch3"),
)
PAIR_ORDER = tuple(f"{ligand}->{receptor}" for ligand, receptor in PAIR_FAMILY)
LIGANDS = ("dla", "dld")
RECEPTORS = ("notch1a", "notch3")
VENTRAL_SPINAL = "Spinal Cord Ventral Region"

CB_BLUE = "#3568A8"
COMMOT_ORANGE = "#D9794A"
EXACT_PURPLE = "#7656A5"
LR_GREY = "#68707A"
PASS_GREEN = "#25856D"
FAIL_RED = "#BE4B44"
LIGHT_GREY = "#D9DDE2"
MID_GREY = "#737B84"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--h5ad",
        type=Path,
        help="Aligned H5AD (required unless --reuse-figure-data-dir is used).",
    )
    result.add_argument(
        "--stage4-edge-dir",
        "--edge-dir",
        dest="stage4_edge_dir",
        type=Path,
        help=(
            "Stage-4 directory, or its attribution parent, with edges_seed_*.csv.gz "
            "(required unless --reuse-figure-data-dir is used)."
        ),
    )
    result.add_argument(
        "--observed-cells",
        type=Path,
        help="Optional attribution observed_cells.csv.gz for index cross-checking.",
    )
    result.add_argument(
        "--delta-notch-family-audit", required=True, type=Path
    )
    result.add_argument(
        "--four-axis-circuit-screen", required=True, type=Path
    )
    result.add_argument(
        "--delta-notch-case-study", required=True, type=Path
    )
    result.add_argument("--output-dir", required=True, type=Path)
    result.add_argument(
        "--reuse-figure-data-dir",
        type=Path,
        help=(
            "Re-render from a previous formal figure_data directory. Spatial cells and "
            "the complete supported-edge table are reused, then direct her4.1 values, "
            "context ranks, controls, and the claim ladder are rebuilt from the formal "
            "family/case inputs. This mode does not require H5AD or raw edge tables."
        ),
    )
    result.add_argument("--stage", type=float, default=4.0)
    result.add_argument("--stage-label", default="24hpf")
    result.add_argument("--time-key", default="time_point_processed")
    result.add_argument("--annotation-key", default="Annotation")
    result.add_argument("--spatial-key", default="spatial_aligned")
    result.add_argument("--sender-type", default=VENTRAL_SPINAL)
    result.add_argument("--receiver-type", default=VENTRAL_SPINAL)
    result.add_argument("--expected-grouping-seeds", type=int, default=5)
    result.add_argument("--min-seed-support", type=int, default=2)
    result.add_argument("--max-display-edges", type=int, default=35)
    result.add_argument(
        "--cytobridge-rank-score",
        choices=("attention", "exact_message"),
        default="attention",
        help="CytoBridge score shown beside COMMOT in panel E.",
    )
    result.add_argument(
        "--material-delta-r2",
        type=float,
        default=0.005,
        help="Predeclared display threshold for a material gain beyond LR-only.",
    )
    result.add_argument("--dpi", type=int, default=300)
    result.add_argument("--overwrite", action="store_true")
    return result


def require_columns(
    frame: pd.DataFrame, columns: Sequence[str], label: str
) -> None:
    missing = [column for column in columns if column not in frame]
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")


def prepare_output(path: Path, overwrite: bool) -> None:
    if path.exists() and not path.is_dir():
        raise NotADirectoryError(path)
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory is non-empty; pass --overwrite: {path}"
            )
        shutil.rmtree(path)
    elif path.exists() and overwrite:
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, object]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "path": str(resolved),
        "bytes": int(resolved.stat().st_size),
        "sha256": sha256(resolved),
    }


def formal_table(directory: Path, filename: str) -> Path:
    """Resolve only the formal ``tables/`` layout, rejecting flat prototypes."""
    path = directory.expanduser().resolve() / "tables" / filename
    if not path.is_file():
        raise FileNotFoundError(
            f"Formal table is missing: {path}. Flat prototype outputs are not accepted."
        )
    return path


def resolve_circuit_screen(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.is_file():
        return path
    candidates = (
        path / "tables" / "known_axis_exact_circuit_screen.csv.gz",
        path / "known_axis_exact_circuit_screen.csv.gz",
        path / "tables" / "exact_circuit_method_support.csv",
    )
    existing = [candidate for candidate in candidates if candidate.is_file()]
    if len(existing) != 1:
        raise FileNotFoundError(
            f"Expected one four-axis circuit screen under {path}; found {existing}"
        )
    return existing[0]


def _require_false(
    mapping: Mapping[str, object], keys: Iterable[str], label: str
) -> None:
    bad = [key for key in keys if mapping.get(key) is not False]
    if bad:
        raise ValueError(f"{label} must explicitly set these guardrails false: {bad}")


def load_family_audit(
    directory: Path,
) -> tuple[dict[str, pd.DataFrame], dict[str, object], dict[str, Path]]:
    root = directory.expanduser().resolve()
    manifest_path = root / "manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("schema_version") not in {1, 2}:
        raise ValueError("Unsupported Delta--Notch family-audit schema")
    if manifest.get("status") != "complete":
        raise ValueError("Delta--Notch family audit is not complete")
    if manifest.get("analysis") != "zebrafish_delta_notch_family_observational_audit":
        raise ValueError("Unexpected Delta--Notch family-audit analysis contract")
    reported_pairs = manifest.get("pair_family")
    expected_pairs = [list(pair) for pair in PAIR_FAMILY]
    if reported_pairs != expected_pairs:
        raise ValueError(
            "Family audit must contain exactly dla/dld x notch1a/notch3"
        )
    resolution = manifest.get("index_resolution", {})
    if not isinstance(resolution, Mapping) or (
        resolution.get("global_index_order_assumed_without_validation") is not False
    ):
        raise ValueError(
            "Family audit must explicitly reject an unvalidated global-row-order assumption"
        )
    guardrails = manifest.get("guardrails", {})
    if not isinstance(guardrails, Mapping):
        raise ValueError("Family audit lacks guardrails")
    _require_false(
        guardrails,
        (
            "attention_is_lr_specific",
            "exact_message_times_lr_is_biochemical_flux",
            "analysis_is_a_causal_or_lr_specific_perturbation",
            "global_h5ad_row_order_was_silently_assumed",
            "cell_rows_or_grouping_seeds_are_biological_replicates",
        ),
        "family-audit guardrails",
    )

    filenames = {
        "mass": "delta_notch_family_context_mass.csv",
        "ranks": "delta_notch_family_ranks.csv",
        "enrichment": "delta_notch_pathway_enrichment.csv",
        "detection": "delta_notch_expression_detection.csv",
        "cells": "delta_notch_receiver_cell_scores.csv.gz",
        "correlations": "delta_notch_downstream_correlations.csv",
        "residuals": "delta_notch_downstream_residual_audit.csv",
        "inventory": "edge_input_inventory.csv",
    }
    paths = {name: formal_table(root, filename) for name, filename in filenames.items()}
    paths["manifest"] = manifest_path
    tables = {name: pd.read_csv(path) for name, path in paths.items() if name != "manifest"}
    return tables, manifest, paths


def load_case_study(
    directory: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object], dict[str, Path]]:
    root = directory.expanduser().resolve()
    manifest_path = root / "manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("schema_version") not in {1, 2}:
        raise ValueError("Unsupported Delta--Notch case-study schema")
    if manifest.get("status") != "complete":
        raise ValueError("Delta--Notch case study is not complete")
    if manifest.get("analysis") != "zebrafish_24hpf_delta_notch_biology_first_case_study":
        raise ValueError("Unexpected Delta--Notch case-study analysis contract")
    guardrails = manifest.get("guardrails", {})
    if not isinstance(guardrails, Mapping):
        raise ValueError("Case-study manifest lacks guardrails")
    _require_false(
        guardrails,
        (
            "attention_is_lr_specific",
            "exact_message_is_lr_specific",
            "analysis_is_a_ligand_knockout",
            "analysis_is_a_perturbation",
            "messages_were_deleted",
            "grouping_seeds_are_biological_replicates",
        ),
        "case-study guardrails",
    )
    if int(manifest.get("schema_version", 0)) >= 2:
        _require_false(
            guardrails,
            (
                "matched_resampling_is_confirmatory_p_value",
                "unique_directed_edges_are_biological_replicates",
                "receiver_cells_are_biological_replicates",
                "case_selection_was_preregistered",
                "post_selection_adjustment_was_applied",
            ),
            "case-study exploratory-resampling guardrails",
        )
    statistics_path = formal_table(root, "case_study_statistics.csv")
    null_path = formal_table(root, "receiver_matched_projection_null.csv.gz")
    statistics = pd.read_csv(statistics_path)
    null = pd.read_csv(null_path)
    if len(statistics) != 1:
        raise ValueError("Case-study statistics must contain exactly one row")
    require_columns(null, ["null_mean_projection"], "matched projection null")
    return statistics, null, manifest, {
        "manifest": manifest_path,
        "statistics": statistics_path,
        "null": null_path,
    }


def dense_gene_column(data: ad.AnnData, gene: str) -> np.ndarray:
    if gene not in data.var_names:
        raise ValueError(f"H5AD is missing required gene: {gene}")
    values = data.X[:, int(data.var_names.get_loc(gene))]
    values = values.toarray() if sparse.issparse(values) else np.asarray(values)
    return np.asarray(values, dtype=np.float64).reshape(-1)


def global_q95_activity(values: np.ndarray) -> np.ndarray:
    """Match the reviewer workflow's full-H5AD positive-q95 scaling."""
    values = np.asarray(values, dtype=np.float64)
    positive = values[values > 0]
    scale = float(np.quantile(positive, 0.95)) if positive.size else 1.0
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    return np.clip(values / scale, 0.0, 1.0)


def build_spatial_cell_table(
    data: ad.AnnData,
    audit_cells: pd.DataFrame,
    *,
    stage: float,
    time_key: str,
    annotation_key: str,
    spatial_key: str,
    receiver_type: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    require_columns(data.obs, [time_key, annotation_key], "H5AD obs")
    if spatial_key not in data.obsm:
        raise KeyError(f"Missing adata.obsm[{spatial_key!r}]")
    stage_values = pd.to_numeric(data.obs[time_key], errors="raise").to_numpy(float)
    stage_mask = np.isclose(stage_values, float(stage), rtol=0.0, atol=1e-12)
    stage_global = np.flatnonzero(stage_mask)
    if stage_global.size == 0:
        raise ValueError(f"H5AD has no cells for stage={stage:g}")

    require_columns(
        audit_cells,
        ["stage_index", "cell_type", "her4.1", "her46"],
        "family-audit receiver-cell table",
    )
    audit = audit_cells.copy()
    audit["stage_index"] = pd.to_numeric(
        audit["stage_index"], errors="raise"
    ).astype(int)
    if audit["stage_index"].duplicated().any():
        raise ValueError("Family-audit stage_index is not unique")
    audit = audit.sort_values("stage_index").reset_index(drop=True)
    expected = np.arange(stage_global.size, dtype=int)
    if not np.array_equal(audit["stage_index"].to_numpy(int), expected):
        raise ValueError("Family-audit receiver cells do not cover the complete stage grid")

    annotations = data.obs.iloc[stage_global][annotation_key].astype(str).to_numpy()
    if not np.array_equal(audit["cell_type"].astype(str).to_numpy(), annotations):
        raise ValueError("Family-audit cell types disagree with the H5AD stage order")
    coordinates = np.asarray(data.obsm[spatial_key], dtype=np.float64)[stage_global, :2]

    ligand_activity = np.maximum.reduce(
        [global_q95_activity(dense_gene_column(data, gene)) for gene in LIGANDS]
    )[stage_mask]
    receptor_activity = np.maximum.reduce(
        [global_q95_activity(dense_gene_column(data, gene)) for gene in RECEPTORS]
    )[stage_mask]
    result = pd.DataFrame(
        {
            "stage_index": expected,
            "global_index": stage_global,
            "obs_name": data.obs_names[stage_global].astype(str),
            "x": coordinates[:, 0],
            "y": coordinates[:, 1],
            "cell_type": annotations,
            "is_ventral_spinal": annotations == str(receiver_type),
            "delta_sender_activity": ligand_activity,
            "notch_receptor_activity": receptor_activity,
            "her4.1_activity": pd.to_numeric(
                audit["her4.1"], errors="raise"
            ).to_numpy(float),
            "her46_response": pd.to_numeric(audit["her46"], errors="raise").to_numpy(float),
        }
    )
    if not np.isfinite(
        result[
            [
                "x",
                "y",
                "delta_sender_activity",
                "notch_receptor_activity",
                "her4.1_activity",
                "her46_response",
            ]
        ].to_numpy(float)
    ).all():
        raise ValueError("Spatial figure table contains non-finite values")
    return result, stage_mask


def resolve_stage_edge_dir(path: Path, stage: float, stage_label: str) -> Path:
    root = path.expanduser().resolve()
    if list(root.glob("edges_seed_*.csv.gz")):
        return root
    candidates = sorted(root.glob(f"stage_{stage:g}_*"))
    candidates = [candidate for candidate in candidates if stage_label in candidate.name]
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected one stage_{stage:g}_* edge directory under {root}; found {candidates}"
        )
    return candidates[0]


def _integer_array(series: pd.Series, label: str) -> np.ndarray:
    values = pd.to_numeric(series, errors="raise").to_numpy(float)
    if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
        raise ValueError(f"{label} must contain finite integers")
    return values.astype(int)


def _observed_global_to_stage(
    observed_path: Path,
    data: ad.AnnData,
    stage_mask: np.ndarray,
) -> dict[int, int]:
    observed = pd.read_csv(observed_path)
    require_columns(observed, ["global_index", "obs_name"], "observed cells")
    global_index = _integer_array(observed["global_index"], "observed global_index")
    if len(np.unique(global_index)) != len(global_index):
        raise ValueError("observed global_index is not unique")
    obs_names = observed["obs_name"].astype(str)
    if obs_names.duplicated().any():
        raise ValueError("observed obs_name is not unique")
    if not data.obs_names.is_unique:
        raise ValueError("H5AD obs_names are not unique")
    if len(observed) != data.n_obs or set(obs_names) != set(data.obs_names.astype(str)):
        raise ValueError(
            "observed cells and H5AD must contain exactly the same obs_name universe"
        )
    h5_lookup = {name: index for index, name in enumerate(data.obs_names.astype(str))}
    h5_index = obs_names.map(h5_lookup)
    if h5_index.isna().any():
        raise ValueError("observed cells contain obs_name values absent from H5AD")
    stage_global = np.flatnonzero(stage_mask)
    h5_to_stage = np.full(data.n_obs, -1, dtype=int)
    h5_to_stage[stage_global] = np.arange(stage_global.size, dtype=int)
    stage_index = h5_to_stage[h5_index.to_numpy(int)]
    return {
        int(global_id): int(local_id)
        for global_id, local_id in zip(global_index, stage_index)
        if local_id >= 0
    }


def load_stage_edge_occurrences(
    edge_dir: Path,
    data: ad.AnnData,
    stage_mask: np.ndarray,
    *,
    stage: float,
    stage_label: str,
    annotation_key: str,
    observed_cells_path: Path | None,
    expected_seeds: int,
) -> tuple[pd.DataFrame, pd.DataFrame, Path | None]:
    stage_dir = resolve_stage_edge_dir(edge_dir, stage, stage_label)
    paths = sorted(stage_dir.glob("edges_seed_*.csv.gz"))
    if not paths:
        raise FileNotFoundError(f"No edge tables under {stage_dir}")

    observed = observed_cells_path
    if observed is None:
        candidate = stage_dir.parent / "observed_cells.csv.gz"
        observed = candidate
    if not observed.is_file():
        raise FileNotFoundError(
            "A valid observed_cells.csv.gz is required for edge endpoint verification: "
            f"{observed}"
        )
    observed_map = _observed_global_to_stage(observed, data, stage_mask)
    annotations = data.obs.loc[stage_mask, annotation_key].astype(str).to_numpy()
    n_stage = int(len(annotations))
    frames: list[pd.DataFrame] = []
    inventory: list[dict[str, object]] = []
    for path in paths:
        frame = pd.read_csv(path)
        require_columns(
            frame,
            [
                "grouping_seed",
                "source_index",
                "target_index",
                "sender_type",
                "receiver_type",
                "attention_abs_mean",
                "edge_message_norm_joint",
            ],
            str(path),
        )
        seeds = np.unique(_integer_array(frame["grouping_seed"], "grouping_seed"))
        if seeds.size != 1:
            raise ValueError(f"Ambiguous grouping seed in {path}")
        if "stage" in frame and not np.isclose(
            pd.to_numeric(frame["stage"], errors="raise").to_numpy(float),
            float(stage),
            rtol=0.0,
            atol=1e-12,
        ).all():
            raise ValueError(f"Unexpected stage in {path}")
        if "stage_label" in frame and set(frame["stage_label"].astype(str)) != {
            str(stage_label)
        }:
            raise ValueError(f"Unexpected stage label in {path}")

        has_source_local = "source_index_stage" in frame
        has_target_local = "target_index_stage" in frame
        if has_source_local != has_target_local:
            raise ValueError(
                "Edge table must contain both source_index_stage and "
                f"target_index_stage, or neither: {path}"
            )
        has_local = has_source_local
        if has_local:
            source = _integer_array(frame["source_index_stage"], "source_index_stage")
            target = _integer_array(frame["target_index_stage"], "target_index_stage")
            source_cross = frame["source_index"].map(observed_map)
            target_cross = frame["target_index"].map(observed_map)
            if source_cross.isna().any() or target_cross.isna().any():
                raise ValueError("observed-cells mapping does not cover all edge endpoints")
            if not np.array_equal(source, source_cross.to_numpy(int)) or not np.array_equal(
                target, target_cross.to_numpy(int)
            ):
                raise ValueError(
                    "Stage-local edge indices disagree with observed-cells mapping"
                )
            mapping_mode = "stage_local_columns"
        else:
            source_series = frame["source_index"].map(observed_map)
            target_series = frame["target_index"].map(observed_map)
            if source_series.isna().any() or target_series.isna().any():
                raise ValueError("observed-cells mapping does not cover all edge endpoints")
            source = source_series.to_numpy(int)
            target = target_series.to_numpy(int)
            mapping_mode = "observed_obs_name"
        if (
            (source < 0).any()
            or (target < 0).any()
            or (source >= n_stage).any()
            or (target >= n_stage).any()
        ):
            raise IndexError(f"Resolved stage edge index is out of range in {path}")
        if np.any(source == target):
            raise ValueError(f"Self edge found in {path}; panel D requires distinct cells")
        if not np.array_equal(frame["sender_type"].astype(str), annotations[source]):
            raise ValueError(f"sender_type disagrees with H5AD stage annotations: {path}")
        if not np.array_equal(frame["receiver_type"].astype(str), annotations[target]):
            raise ValueError(f"receiver_type disagrees with H5AD stage annotations: {path}")

        local = frame.copy()
        local["_source_stage_index"] = source
        local["_target_stage_index"] = target
        duplicate = local.duplicated(
            ["grouping_seed", "_source_stage_index", "_target_stage_index"]
        )
        if duplicate.any():
            raise ValueError(f"Duplicate directed cell pair within seed: {path}")
        frames.append(local)
        inventory.append(
            {
                "path": str(path.resolve()),
                "sha256": sha256(path),
                "grouping_seed": int(seeds[0]),
                "n_rows": int(len(local)),
                "mapping_mode": mapping_mode,
            }
        )

    raw = pd.concat(frames, ignore_index=True)
    seed_values = sorted(raw["grouping_seed"].astype(int).unique())
    if len(seed_values) != int(expected_seeds):
        raise ValueError(
            f"Expected {expected_seeds} grouping seeds, found {seed_values}"
        )
    return raw, pd.DataFrame(inventory), observed


def validate_edge_inventory(
    actual: pd.DataFrame, family_inventory: pd.DataFrame
) -> None:
    require_columns(
        family_inventory,
        ["grouping_seed", "n_rows", "sha256"],
        "family edge inventory",
    )
    left = actual[["grouping_seed", "n_rows", "sha256"]].sort_values(
        "grouping_seed"
    ).reset_index(drop=True)
    right = family_inventory[["grouping_seed", "n_rows", "sha256"]].copy()
    right["grouping_seed"] = pd.to_numeric(right["grouping_seed"], errors="raise").astype(int)
    right["n_rows"] = pd.to_numeric(right["n_rows"], errors="raise").astype(int)
    right = right.sort_values("grouping_seed").reset_index(drop=True)
    if not left.equals(right):
        raise ValueError("Stage-4 edge inputs disagree with the formal family-audit inventory")


def collapse_display_edges(
    raw_edges: pd.DataFrame,
    spatial_cells: pd.DataFrame,
    *,
    sender_type: str,
    receiver_type: str,
    min_seed_support: int,
    max_display_edges: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    require_columns(
        raw_edges,
        [
            "grouping_seed",
            "sender_type",
            "receiver_type",
            "attention_abs_mean",
            "edge_message_norm_joint",
            "_source_stage_index",
            "_target_stage_index",
        ],
        "stage edge occurrences",
    )
    local = raw_edges.loc[
        raw_edges["sender_type"].astype(str).eq(sender_type)
        & raw_edges["receiver_type"].astype(str).eq(receiver_type)
    ].copy()
    if local.empty:
        raise ValueError("No edge occurrences for the requested exact cell-type circuit")
    source = local["_source_stage_index"].to_numpy(int)
    target = local["_target_stage_index"].to_numpy(int)
    delta = spatial_cells["delta_sender_activity"].to_numpy(float)
    receptor = spatial_cells["notch_receptor_activity"].to_numpy(float)
    local["family_lr_activity"] = delta[source] * receptor[target]
    local["attention_lr"] = (
        local["family_lr_activity"].to_numpy(float)
        * local["attention_abs_mean"].to_numpy(float)
    )
    local["exact_message_lr"] = (
        local["family_lr_activity"].to_numpy(float)
        * local["edge_message_norm_joint"].to_numpy(float)
    )
    if "spatial_distance" not in local:
        xy = spatial_cells[["x", "y"]].to_numpy(float)
        local["spatial_distance"] = np.linalg.norm(xy[source] - xy[target], axis=1)
    n_seeds = int(raw_edges["grouping_seed"].nunique())
    collapsed = (
        local.groupby(
            [
                "_source_stage_index",
                "_target_stage_index",
                "sender_type",
                "receiver_type",
            ],
            observed=True,
            as_index=False,
        )
        .agg(
            seed_support=("grouping_seed", "nunique"),
            seed_list=(
                "grouping_seed",
                lambda values: ";".join(
                    str(value) for value in sorted(set(map(int, values)))
                ),
            ),
            family_lr_activity=("family_lr_activity", "mean"),
            mean_attention_abs=("attention_abs_mean", "mean"),
            mean_edge_message_norm=("edge_message_norm_joint", "mean"),
            mean_attention_lr=("attention_lr", "mean"),
            mean_exact_message_lr=("exact_message_lr", "mean"),
            mean_spatial_distance=("spatial_distance", "mean"),
        )
    )
    collapsed["n_total_grouping_seeds"] = n_seeds
    collapsed["seed_support_fraction"] = collapsed["seed_support"] / n_seeds
    collapsed["display_score"] = (
        collapsed["mean_attention_lr"] * collapsed["seed_support_fraction"]
    )
    xy = spatial_cells.set_index("stage_index")[["x", "y"]]
    source_index = collapsed["_source_stage_index"].to_numpy(int)
    target_index = collapsed["_target_stage_index"].to_numpy(int)
    collapsed[["source_x", "source_y"]] = xy.loc[source_index].to_numpy(float)
    collapsed[["target_x", "target_y"]] = xy.loc[target_index].to_numpy(float)
    supported = collapsed.loc[
        collapsed["family_lr_activity"].gt(0)
        & collapsed["seed_support"].ge(int(min_seed_support))
    ].sort_values(
        ["display_score", "seed_support", "_source_stage_index", "_target_stage_index"],
        ascending=[False, False, True, True],
        kind="stable",
    )
    if supported.empty:
        raise ValueError("No LR-compatible distinct-cell edge passes the seed-support rule")
    display = supported.head(int(max_display_edges)).copy().reset_index(drop=True)
    display["display_rank"] = np.arange(1, len(display) + 1, dtype=int)
    return supported.reset_index(drop=True), display


def rerank_reused_supported_edges(
    supported: pd.DataFrame,
    *,
    min_seed_support: int,
    max_display_edges: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rebuild the reader-facing edge selection from a formal support table.

    Older reader figures ranked arrows by exact-message magnitude.  The main
    biology figure now uses the same quantity as panel E: generic model edge
    weight multiplied by post-hoc Delta/Notch expression compatibility.  The
    grouping-run support fraction is used only to stabilize display ranking.
    """
    require_columns(
        supported,
        [
            "_source_stage_index",
            "_target_stage_index",
            "family_lr_activity",
            "mean_attention_lr",
            "seed_support",
            "seed_support_fraction",
            "n_total_grouping_seeds",
            "source_x",
            "source_y",
            "target_x",
            "target_y",
        ],
        "reused supported-edge table",
    )
    result = supported.copy()
    result["display_score"] = (
        pd.to_numeric(result["mean_attention_lr"], errors="raise")
        * pd.to_numeric(result["seed_support_fraction"], errors="raise")
    )
    result = result.loc[
        pd.to_numeric(result["family_lr_activity"], errors="raise").gt(0)
        & pd.to_numeric(result["seed_support"], errors="raise").ge(
            int(min_seed_support)
        )
    ].sort_values(
        ["display_score", "seed_support", "_source_stage_index", "_target_stage_index"],
        ascending=[False, False, True, True],
        kind="stable",
    )
    if result.empty:
        raise ValueError("Reused support table has no edge passing the display rule")
    display = result.head(int(max_display_edges)).copy().reset_index(drop=True)
    display["display_rank"] = np.arange(1, len(display) + 1, dtype=int)
    return result.reset_index(drop=True), display


def prepare_circuit_percentiles(
    screen: pd.DataFrame,
    *,
    stage: float,
    sender_type: str,
    receiver_type: str,
    cytobridge_rank_score: str,
) -> pd.DataFrame:
    require_columns(
        screen,
        [
            "stage",
            "axis_id",
            "sender_type",
            "receiver_type",
            "passes_context_support_filter",
            "is_evaluated_context",
            "n_evaluated_contexts",
            "attention_context_percentile",
            "attention_context_rank_from_top",
            "exact_message_context_percentile",
            "exact_message_context_rank_from_top",
            "commot_context_percentile",
            "commot_context_rank_from_top",
        ],
        "four-axis circuit screen",
    )
    stage_mask = np.isclose(
        pd.to_numeric(screen["stage"], errors="raise").to_numpy(float),
        float(stage),
        rtol=0.0,
        atol=1e-12,
    )
    stage_screen = screen.loc[
        stage_mask & screen["axis_id"].astype(str).isin(PAIR_ORDER)
    ].copy()
    for axis_id in PAIR_ORDER:
        axis = stage_screen.loc[stage_screen["axis_id"].astype(str).eq(axis_id)]
        evaluated = axis.loc[
            axis["passes_context_support_filter"].eq(True)
            & axis["is_evaluated_context"].eq(True)
        ]
        if evaluated.empty:
            raise ValueError(f"Circuit screen has no evaluated contexts for {axis_id}")
        reported_counts = pd.to_numeric(
            evaluated["n_evaluated_contexts"], errors="raise"
        ).astype(int)
        if reported_counts.nunique() != 1 or int(reported_counts.iloc[0]) != len(
            evaluated
        ):
            raise ValueError(
                f"Support-qualified context count disagrees for {axis_id}: "
                f"reported={sorted(reported_counts.unique().tolist())}, "
                f"observed={len(evaluated)}"
            )
    selected = stage_screen.loc[
        stage_screen["sender_type"].astype(str).eq(sender_type)
        & stage_screen["receiver_type"].astype(str).eq(receiver_type)
    ].copy()
    if selected["axis_id"].duplicated().any() or set(selected["axis_id"]) != set(
        PAIR_ORDER
    ):
        raise ValueError("Circuit screen must contain exactly one row for each family pair")
    if not selected["passes_context_support_filter"].eq(True).all() or not selected[
        "is_evaluated_context"
    ].eq(True).all():
        raise ValueError(
            "Selected Delta--Notch contexts must pass the formal support filter and "
            "belong to the evaluated rank universe"
        )
    cb_column = (
        "attention_context_percentile"
        if cytobridge_rank_score == "attention"
        else "exact_message_context_percentile"
    )
    cb_label = (
        "CytoBridge edge weight × LR"
        if cytobridge_rank_score == "attention"
        else "CytoBridge message magnitude × LR"
    )
    cb_rank_column = (
        "attention_context_rank_from_top"
        if cytobridge_rank_score == "attention"
        else "exact_message_context_rank_from_top"
    )
    rows: list[dict[str, object]] = []
    selected = selected.set_index("axis_id").loc[list(PAIR_ORDER)].reset_index()
    for row in selected.itertuples(index=False):
        for method, column, rank_column in (
            (cb_label, cb_column, cb_rank_column),
            ("COMMOT", "commot_context_percentile", "commot_context_rank_from_top"),
        ):
            percentile = float(getattr(row, column))
            if not np.isfinite(percentile) or not 0 <= percentile <= 1:
                raise ValueError(f"Invalid context percentile for {row.axis_id}: {percentile}")
            n_contexts = int(getattr(row, "n_evaluated_contexts"))
            rank_from_top = float(getattr(row, rank_column))
            if np.isclose(rank_from_top, round(rank_from_top), atol=1e-8):
                rank_from_top = float(round(rank_from_top))
            if not 1 <= rank_from_top <= n_contexts:
                raise ValueError(
                    f"Invalid rank for {row.axis_id}: {rank_from_top}/{n_contexts}"
                )
            expected_percentile = 1.0 - (rank_from_top - 1.0) / n_contexts
            if not np.isclose(
                percentile, expected_percentile, rtol=0.0, atol=1e-8
            ):
                raise ValueError(
                    f"Formal rank and percentile disagree for {row.axis_id} {method}: "
                    f"rank={rank_from_top}/{n_contexts}, percentile={percentile}"
                )
            rank_text = (
                str(int(rank_from_top))
                if float(rank_from_top).is_integer()
                else f"{rank_from_top:.1f}"
            )
            rows.append(
                {
                    "axis_id": str(row.axis_id),
                    "method": method,
                    "context_percentile": percentile,
                    "top_percent": 100.0 * (1.0 - percentile),
                    "rank_from_top": rank_from_top,
                    "n_contexts": n_contexts,
                    "rank_label": f"{rank_text}/{n_contexts}",
                    "sender_type": sender_type,
                    "receiver_type": receiver_type,
                    "stage": float(stage),
                }
            )
    return pd.DataFrame(rows)


def prepare_control_tables(
    correlations: pd.DataFrame, residuals: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    require_columns(
        correlations,
        ["subset", "module", "score", "n_cells", "spearman_rho"],
        "downstream correlations",
    )
    require_columns(
        residuals,
        ["module", "baseline", "score", "n_cells", "delta_r2"],
        "downstream residual audit",
    )
    score_labels = {
        "lr_only": "LR expression only",
        "attention_lr": "Edge weight × LR",
        "exact_message_lr": "Message magnitude × LR",
    }
    module_labels = {
        "her46": "Compact Her-associated state\n(her4.1 + her6)",
        "her_broad": "Broad neural Her/Hey module",
    }
    corr = correlations.loc[
        correlations["subset"].astype(str).eq("core_neural")
        & correlations["module"].isin(module_labels)
        & correlations["score"].isin(score_labels)
    ].copy()
    expected_corr = pd.MultiIndex.from_product(
        [module_labels, score_labels], names=["module", "score"]
    )
    actual_corr = pd.MultiIndex.from_frame(corr[["module", "score"]])
    if actual_corr.has_duplicates or set(actual_corr) != set(expected_corr):
        raise ValueError("Correlation table lacks the complete 2-module x 3-score grid")
    corr["score_label"] = corr["score"].map(score_labels)
    corr["module_label"] = corr["module"].map(module_labels)

    baseline = "cell_type+notch1a+notch3+lr_only"
    delta = residuals.loc[
        residuals["baseline"].astype(str).eq(baseline)
        & residuals["module"].isin(module_labels)
        & residuals["score"].isin(["attention_lr", "exact_message_lr"])
    ].copy()
    expected_delta = pd.MultiIndex.from_product(
        [module_labels, ["attention_lr", "exact_message_lr"]],
        names=["module", "score"],
    )
    actual_delta = pd.MultiIndex.from_frame(delta[["module", "score"]])
    if actual_delta.has_duplicates or set(actual_delta) != set(expected_delta):
        raise ValueError("Residual table lacks the strict LR-adjusted 2 x 2 grid")
    delta["score_label"] = delta["score"].map(score_labels)
    delta["module_label"] = delta["module"].map(module_labels)
    return corr.reset_index(drop=True), delta.reset_index(drop=True)


def build_claim_ladder(
    circuit_percentiles: pd.DataFrame,
    detection: pd.DataFrame,
    delta_r2: pd.DataFrame,
    *,
    receiver_type: str,
    material_delta_r2: float,
    n_supported_edges: int,
    n_display_edges: int,
    min_seed_support: int,
    n_grouping_seeds: int,
) -> pd.DataFrame:
    require_columns(
        detection,
        ["cell_type", "n_cells", "gene", "detected_fraction_x_gt_zero"],
        "expression detection",
    )
    require_columns(
        circuit_percentiles,
        ["method", "rank_from_top", "n_contexts"],
        "exact-context ranks",
    )
    max_rank = float(circuit_percentiles["rank_from_top"].max())
    min_contexts = int(circuit_percentiles["n_contexts"].min())
    max_contexts = int(circuit_percentiles["n_contexts"].max())
    circuit_pass = max_rank <= 0.25 * min_contexts
    target_detection = detection.loc[
        detection["cell_type"].astype(str).eq(receiver_type)
        & detection["gene"].astype(str).isin([*LIGANDS, *RECEPTORS, "her4.1"])
    ]
    if set(target_detection["gene"].astype(str)) != {
        *LIGANDS,
        *RECEPTORS,
        "her4.1",
    }:
        raise ValueError("Ventral-spinal expression audit lacks required genes")
    detection_lookup = target_detection.set_index("gene")[
        "detected_fraction_x_gt_zero"
    ].astype(float)
    min_detection = float(detection_lookup.min())
    spatial_pass = min_detection > 0
    target_n = int(target_detection["n_cells"].iloc[0])
    if not (target_detection["n_cells"].astype(int) == target_n).all():
        raise ValueError("Ventral-spinal detection rows disagree on the cell count")
    attention_delta = delta_r2.loc[
        delta_r2["score"].astype(str).eq("attention_lr"), "delta_r2"
    ].astype(float)
    max_attention_delta = float(attention_delta.max())
    attention_pass = max_attention_delta >= float(material_delta_r2)

    rows = [
        {
            "order": 1,
            "status": True,
            "claim": "Matches known 24–42 hpf spinal biology",
            "evidence": "Spinal Notch activity; her4 responds to Notch",
            "threshold": "supported by cited zebrafish gain/loss-of-function studies",
        },
        {
            "order": 2,
            "status": bool(spatial_pass),
            "claim": "Delta, Notch and her4.1 are present",
            "evidence": (
                f"ventral spinal n={target_n:,}: Delta "
                f"{min(detection_lookup['dla'], detection_lookup['dld']):.0%}–"
                f"{max(detection_lookup['dla'], detection_lookup['dld']):.0%}; Notch "
                f"{min(detection_lookup['notch1a'], detection_lookup['notch3']):.0%}–"
                f"{max(detection_lookup['notch1a'], detection_lookup['notch3']):.0%}; "
                f"her4.1 {detection_lookup['her4.1']:.0%}"
            ),
            "threshold": "all required genes detected",
        },
        {
            "order": 3,
            "status": bool(circuit_pass),
            "claim": "CytoBridge and COMMOT agree on all four circuits",
            "evidence": (
                f"ranks 1–{max_rank:g} of "
                f"{min_contexts}"
                + (f"–{max_contexts}" if min_contexts != max_contexts else "")
                + " sender→receiver contexts"
            ),
            "threshold": "both methods within top quartile",
        },
        {
            "order": 4,
            "status": int(n_supported_edges) > 0,
            "claim": "Edges persist across technical grouping runs",
            "evidence": (
                f"{int(n_supported_edges):,} edges pass ≥{int(min_seed_support)}/"
                f"{int(n_grouping_seeds)} technical runs; {int(n_display_edges)} shown"
            ),
            "threshold": "technical sensitivity only; not biological replication",
        },
        {
            "order": 5,
            "status": bool(attention_pass),
            "claim": "Edge weight adds information beyond LR expression",
            "evidence": (
                f"additional variance ≤{100 * max_attention_delta:.3g}% "
                f"(material threshold {100 * material_delta_r2:g}%)"
            ),
            "threshold": f"ΔR² >= {material_delta_r2:g}",
        },
        {
            "order": 6,
            "status": False,
            "claim": "Causal Delta–Notch perturbation tested",
            "evidence": "No perturbation, message deletion, or trajectory rerun",
            "threshold": "requires an experimental or model intervention",
        },
    ]
    result = pd.DataFrame(rows)
    result["mark"] = np.where(result["status"], "check", "cross")
    return result


def _panel_title(ax, letter: str, title: str) -> None:
    ax.set_title(f"{letter}  {title}", loc="left", fontsize=14.5, fontweight="bold", pad=9)


def _robust_region_box(coordinates: np.ndarray) -> tuple[float, float, float, float]:
    """Return a robust box that links whole-embryo and zoomed spatial panels."""
    coordinates = np.asarray(coordinates, dtype=float)
    lower = np.quantile(coordinates, 0.01, axis=0)
    upper = np.quantile(coordinates, 0.99, axis=0)
    pad = 0.035 * np.maximum(upper - lower, 1e-6)
    return (
        float(lower[0] - pad[0]),
        float(lower[1] - pad[1]),
        float(upper[0] - lower[0] + 2 * pad[0]),
        float(upper[1] - lower[1] + 2 * pad[1]),
    )


def _draw_region_box(ax, bounds: tuple[float, float, float, float]) -> None:
    x, y, width, height = bounds
    ax.add_patch(
        Rectangle(
            (x, y),
            width,
            height,
            fill=False,
            edgecolor=CB_BLUE,
            linewidth=2.0,
            linestyle=(0, (4, 2)),
            zorder=8,
        )
    )


def _spatial_limits(ax, coordinates: np.ndarray, pad_fraction: float = 0.05) -> None:
    x_min, y_min = np.nanmin(coordinates, axis=0)
    x_max, y_max = np.nanmax(coordinates, axis=0)
    x_pad = max((x_max - x_min) * pad_fraction, 1e-6)
    y_pad = max((y_max - y_min) * pad_fraction, 1e-6)
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.set_aspect("equal")
    ax.axis("off")


def _activity_scatter(
    ax,
    coordinates: np.ndarray,
    values: np.ndarray,
    *,
    cmap: str,
    norm,
    background_size: float = 3.0,
    active_size: float = 11.0,
):
    ax.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        s=background_size,
        c=LIGHT_GREY,
        alpha=0.42,
        linewidths=0,
        rasterized=True,
    )
    active = np.asarray(values) != 0
    points = ax.scatter(
        coordinates[active, 0],
        coordinates[active, 1],
        s=active_size,
        c=np.asarray(values)[active],
        cmap=cmap,
        norm=norm,
        alpha=0.9,
        linewidths=0,
        rasterized=True,
    )
    _spatial_limits(ax, coordinates)
    return points


def _edge_widths(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 0 or np.ptp(values) <= 0:
        return np.full(values.shape, 2.4)
    low, high = np.quantile(values, [0.1, 0.9])
    if high <= low:
        low, high = float(values.min()), float(values.max())
    scaled = np.clip((values - low) / max(high - low, 1e-12), 0.0, 1.0)
    return 1.5 + 3.5 * scaled


def plot_main_figure(
    spatial_cells: pd.DataFrame,
    display_edges: pd.DataFrame,
    circuit_percentiles: pd.DataFrame,
    claim_ladder: pd.DataFrame,
    *,
    stage_label: str,
    receiver_type: str,
    n_supported_edges: int,
    output_png: Path,
    dpi: int,
) -> tuple[Path, Path]:
    require_columns(
        spatial_cells,
        [
            "stage_index",
            "x",
            "y",
            "is_ventral_spinal",
            "delta_sender_activity",
            "notch_receptor_activity",
            "her4.1_activity",
        ],
        "spatial figure cells",
    )
    style = {
        "font.family": "DejaVu Sans",
        "font.size": 14,
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "legend.fontsize": 11,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
    with plt.rc_context(style):
        figure, axes = plt.subplots(2, 3, figsize=(13.5, 8.6))
        figure.subplots_adjust(
            left=0.035, right=0.985, bottom=0.07, top=0.82, wspace=0.27, hspace=0.38
        )
        xy = spatial_cells[["x", "y"]].to_numpy(float)
        ventral = spatial_cells["is_ventral_spinal"].to_numpy(bool)
        ventral_xy = xy[ventral]
        region_box = _robust_region_box(ventral_xy)

        ax = axes[0, 0]
        ax.scatter(
            xy[:, 0], xy[:, 1], s=4, c=LIGHT_GREY, alpha=0.42, linewidths=0, rasterized=True
        )
        ax.scatter(
            ventral_xy[:, 0],
            ventral_xy[:, 1],
            s=13,
            c=CB_BLUE,
            alpha=0.9,
            linewidths=0,
            rasterized=True,
            label=f"Ventral spinal cells (n={ventral.sum():,})",
        )
        _draw_region_box(ax, region_box)
        _spatial_limits(ax, xy)
        _panel_title(ax, "A", "Target tissue:\nventral spinal cord")
        ax.legend(loc="lower left", frameon=False, markerscale=1.6)

        ax = axes[0, 1]
        delta = spatial_cells["delta_sender_activity"].to_numpy(float)
        delta_norm = Normalize(vmin=0.0, vmax=max(1.0, float(delta.max())))
        points = _activity_scatter(
            ax, xy, delta, cmap="YlOrRd", norm=delta_norm, active_size=12
        )
        _draw_region_box(ax, region_box)
        _panel_title(ax, "B", "Observed Delta\nligand expression")
        ax.text(
            0.02,
            0.02,
            "Dashed box = ventral-spinal zoom in C/D",
            transform=ax.transAxes,
            fontsize=10.5,
            color=CB_BLUE,
            fontweight="bold",
        )
        colorbar = figure.colorbar(
            points, ax=ax, orientation="horizontal", fraction=0.05, pad=0.018, aspect=26
        )
        colorbar.set_label(
            "Observed expression: max(dla, dld), scaled 0–1", fontsize=11
        )
        colorbar.ax.tick_params(labelsize=10)

        host = axes[0, 2]
        host.axis("off")
        _panel_title(host, "C", "Ventral-spinal zoom:\nreceptor and her4.1")
        left = host.inset_axes([0.00, 0.14, 0.48, 0.75])
        right = host.inset_axes([0.52, 0.14, 0.48, 0.75])
        receptor = spatial_cells.loc[ventral, "notch_receptor_activity"].to_numpy(float)
        receptor_norm = Normalize(vmin=0.0, vmax=max(1.0, float(receptor.max())))
        receptor_points = _activity_scatter(
            left, ventral_xy, receptor, cmap="Blues", norm=receptor_norm, active_size=16
        )
        left.set_title("Notch receptors\nnotch1a / notch3", fontsize=10.5, pad=4)
        response = spatial_cells.loc[ventral, "her4.1_activity"].to_numpy(float)
        response_norm = Normalize(vmin=0.0, vmax=max(1.0, float(response.max())))
        response_points = _activity_scatter(
            right, ventral_xy, response, cmap="Reds", norm=response_norm, active_size=16
        )
        right.set_title("Notch response gene\nher4.1", fontsize=10.5, pad=4)
        cax_left = host.inset_axes([0.07, 0.035, 0.34, 0.030])
        cb_left = figure.colorbar(receptor_points, cax=cax_left, orientation="horizontal")
        cb_left.set_ticks([0.0, 1.0])
        cb_left.ax.tick_params(labelsize=9, pad=1)
        cax_right = host.inset_axes([0.59, 0.035, 0.34, 0.030])
        cb_right = figure.colorbar(response_points, cax=cax_right, orientation="horizontal")
        cb_right.set_ticks([0.0, 1.0])
        cb_right.ax.tick_params(labelsize=9, pad=1)

        ax = axes[1, 0]
        ax.scatter(
            ventral_xy[:, 0],
            ventral_xy[:, 1],
            s=7,
            c="#C9CED5",
            alpha=0.5,
            linewidths=0,
            rasterized=True,
        )
        supports = sorted(display_edges["seed_support"].astype(int).unique())
        n_total = int(display_edges["n_total_grouping_seeds"].iloc[0])
        support_palette = {2: "#2A9D8F", 3: "#3A86C8", 4: "#7B2CBF", 5: "#D1495B"}
        support_colors = {support: support_palette.get(support, "#4C566A") for support in supports}
        widths = _edge_widths(display_edges["display_score"].to_numpy(float))
        for row, width in zip(
            display_edges.sort_values("display_score").itertuples(index=False),
            widths[np.argsort(display_edges["display_score"].to_numpy(float))],
        ):
            arrow = FancyArrowPatch(
                (row.source_x, row.source_y),
                (row.target_x, row.target_y),
                arrowstyle="-|>",
                mutation_scale=11.0 + 1.5 * int(row.seed_support),
                linewidth=float(width),
                color=support_colors[int(row.seed_support)],
                alpha=0.78,
                shrinkA=3.0,
                shrinkB=3.0,
                zorder=3,
            )
            ax.add_patch(arrow)

        source_xy = display_edges[["source_x", "source_y"]].drop_duplicates().to_numpy(float)
        target_frame = display_edges[
            ["_target_stage_index", "target_x", "target_y"]
        ].drop_duplicates("_target_stage_index")
        ax.scatter(
            source_xy[:, 0], source_xy[:, 1], s=30, c="#E38B2C", edgecolors="white",
            linewidths=0.6, zorder=5,
        )
        ax.scatter(
            target_frame["target_x"], target_frame["target_y"], s=30, c=CB_BLUE,
            edgecolors="white", linewidths=0.6, zorder=5,
        )
        her_lookup = spatial_cells.set_index("stage_index")["her4.1_activity"]
        her_positive = target_frame["_target_stage_index"].map(her_lookup).to_numpy(float) > 0
        if her_positive.any():
            ax.scatter(
                target_frame.loc[her_positive, "target_x"],
                target_frame.loc[her_positive, "target_y"],
                s=52,
                facecolors="none",
                edgecolors=FAIL_RED,
                linewidths=1.5,
                zorder=6,
            )
        _spatial_limits(ax, ventral_xy, pad_fraction=0.08)
        _panel_title(ax, "D", "Expression-compatible edges\nacross technical grouping runs")
        support_handles = [
            Line2D(
                [0], [0], color=support_colors[value], lw=3.0,
                label=f"{value}/{n_total} grouping runs",
            )
            for value in supports
        ]
        support_legend = ax.legend(
            handles=support_handles,
            loc="lower left",
            bbox_to_anchor=(0.0, 0.06),
            frameon=False,
            title="Technical support",
            title_fontsize=10.5,
            ncol=min(2, len(support_handles)),
        )
        ax.add_artist(support_legend)
        endpoint_handles = [
            Line2D([0], [0], marker="o", color="none", markerfacecolor="#E38B2C",
                   markeredgecolor="white", markersize=7, label="Delta+ source"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=CB_BLUE,
                   markeredgecolor="white", markersize=7, label="Notch+ target"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor="none",
                   markeredgecolor=FAIL_RED, markersize=8, label="her4.1+ target"),
        ]
        ax.legend(
            handles=endpoint_handles,
            loc="upper left",
            bbox_to_anchor=(0.0, 0.76),
            frameon=False,
            fontsize=9.0,
        )
        ax.text(
            0.02,
            0.90,
            f"Top {len(display_edges)}/{int(n_supported_edges):,} qualifying edges\n"
            "width = edge weight × LR × technical support",
            transform=ax.transAxes,
            va="top",
            fontsize=9.2,
            color="#31363B",
        )
        ax.text(
            0.02,
            0.015,
            "Grouping runs are not biological replicates",
            transform=ax.transAxes,
            va="bottom",
            fontsize=9.5,
            color=FAIL_RED,
            fontweight="bold",
        )

        ax = axes[1, 1]
        method_order = [
            method
            for method in (
                "CytoBridge edge weight × LR",
                "CytoBridge message magnitude × LR",
                "COMMOT",
            )
            if method in set(circuit_percentiles["method"])
        ]
        method_style = {
            "CytoBridge edge weight × LR": (CB_BLUE, "o"),
            "CytoBridge message magnitude × LR": (EXACT_PURPLE, "D"),
            "COMMOT": (COMMOT_ORANGE, "s"),
        }
        method_offset = {method_order[0]: 0.10, "COMMOT": -0.10}
        y_lookup = {axis_id: len(PAIR_ORDER) - 1 - index for index, axis_id in enumerate(PAIR_ORDER)}
        for axis_id in PAIR_ORDER:
            local = circuit_percentiles.loc[circuit_percentiles["axis_id"].eq(axis_id)]
            ax.plot(
                local["rank_from_top"],
                [y_lookup[axis_id]] * len(local),
                color="#BFC4CA",
                linewidth=1.8,
                zorder=1,
            )
        for method in method_order:
            local = circuit_percentiles.loc[circuit_percentiles["method"].eq(method)]
            color, marker = method_style[method]
            ys = [y_lookup[value] + method_offset.get(method, 0.0) for value in local["axis_id"]]
            ax.scatter(
                local["rank_from_top"],
                ys,
                s=82,
                c=color,
                marker=marker,
                edgecolors="white",
                linewidths=0.7,
                label=method,
                zorder=3,
            )
            for x, y, label in zip(local["rank_from_top"], ys, local["rank_label"]):
                ax.annotate(
                    str(label), (float(x), float(y)), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=9.2, color=color,
                )
        max_rank = float(circuit_percentiles["rank_from_top"].max())
        x_max = max(5.25, float(np.ceil(max_rank + 1.0)))
        ax.set_xlim(0.5, x_max)
        ax.set_xticks(np.arange(1, int(np.floor(x_max)) + 1))
        ax.set_yticks([y_lookup[value] for value in PAIR_ORDER])
        ax.set_yticklabels([value.replace("->", " → ") for value in PAIR_ORDER])
        context_counts = sorted(circuit_percentiles["n_contexts"].astype(int).unique())
        context_text = (
            str(context_counts[0])
            if len(context_counts) == 1
            else f"{context_counts[0]}–{context_counts[-1]}"
        )
        ax.set_xlabel(
            f"Rank among {context_text} cell-type sender→receiver contexts (1 = best)\n"
            f"zoom: ranks 1–{int(np.floor(x_max))}; full range 1–{context_text}"
        )
        _panel_title(ax, "E", "Independent ranking of\nthe same four circuits")
        ax.grid(axis="x", color="#E4E7EA", linewidth=0.8)
        ax.legend(frameon=False, loc="center right", fontsize=10)

        ax = axes[1, 2]
        ax.axis("off")
        _panel_title(ax, "F", "Biological evidence\nand limits")
        ladder = claim_ladder.sort_values("order")
        y_positions = np.linspace(0.88, 0.10, len(ladder))
        for y, row in zip(y_positions, ladder.itertuples(index=False)):
            color = PASS_GREEN if bool(row.status) else FAIL_RED
            symbol = "✓" if bool(row.status) else "×"
            ax.text(0.01, y, symbol, color=color, fontsize=22, fontweight="bold", va="center")
            ax.text(
                0.095, y + 0.022, row.claim, fontsize=10.4,
                fontweight="bold", va="center",
            )
            ax.text(
                0.095, y - 0.034, row.evidence, fontsize=8.8,
                color=MID_GREY, va="center",
            )

        figure.suptitle(
            f"{stage_label} ventral-spinal Delta–Notch audit",
            fontsize=22,
            fontweight="bold",
            y=0.985,
        )
        figure.text(
            0.5,
            0.925,
            "Observed expression • generic model edges filtered post hoc for LR compatibility • no perturbation",
            ha="center",
            fontsize=12.5,
            color=MID_GREY,
        )
        output_png.parent.mkdir(parents=True, exist_ok=True)
        output_pdf = output_png.with_suffix(".pdf")
        figure.savefig(output_png, dpi=int(dpi), bbox_inches="tight", facecolor="white")
        figure.savefig(output_pdf, bbox_inches="tight", facecolor="white")
        plt.close(figure)
    return output_png, output_pdf


def plot_controls_figure(
    correlations: pd.DataFrame,
    delta_r2: pd.DataFrame,
    null: pd.DataFrame,
    case_statistics: pd.DataFrame,
    *,
    material_delta_r2: float,
    output_png: Path,
    dpi: int,
) -> tuple[Path, Path]:
    style = {
        "font.family": "DejaVu Sans",
        "font.size": 13,
        "axes.labelsize": 13,
        "axes.titlesize": 16,
        "legend.fontsize": 10.5,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
    }
    score_order = [
        "LR expression only",
        "Edge weight × LR",
        "Message magnitude × LR",
    ]
    score_style = {
        "LR expression only": (LR_GREY, "o"),
        "Edge weight × LR": (CB_BLUE, "s"),
        "Message magnitude × LR": (EXACT_PURPLE, "D"),
    }
    module_order = [
        "Compact Her-associated state\n(her4.1 + her6)",
        "Broad neural Her/Hey module",
    ]
    with plt.rc_context(style):
        figure, axes = plt.subplots(1, 3, figsize=(13.5, 5.8))
        figure.subplots_adjust(left=0.07, right=0.985, bottom=0.17, top=0.70, wspace=0.38)

        ax = axes[0]
        offsets = {
            "LR expression only": -0.17,
            "Edge weight × LR": 0.0,
            "Message magnitude × LR": 0.17,
        }
        y_lookup = {module: len(module_order) - 1 - index for index, module in enumerate(module_order)}
        for module in module_order:
            local = correlations.loc[correlations["module_label"].eq(module)]
            ax.plot(
                [local["spearman_rho"].min(), local["spearman_rho"].max()],
                [y_lookup[module], y_lookup[module]],
                color="#C3C8CE",
                linewidth=2,
                zorder=1,
            )
        for score in score_order:
            local = correlations.loc[correlations["score_label"].eq(score)]
            color, marker = score_style[score]
            ax.scatter(
                local["spearman_rho"],
                [y_lookup[value] + offsets[score] for value in local["module_label"]],
                s=70,
                color=color,
                marker=marker,
                edgecolor="white",
                linewidth=0.6,
                label=score,
                zorder=3,
            )
        values = correlations["spearman_rho"].to_numpy(float)
        pad = max(np.ptp(values) * 0.8, 0.02)
        ax.set_xlim(min(0.0, float(values.min()) - pad), float(values.max()) + pad)
        ax.axvline(0, color="#B8BDC3", linewidth=0.9)
        ax.set_yticks([y_lookup[value] for value in module_order])
        ax.set_yticklabels(module_order)
        ax.set_xlabel("Spearman ρ with observed receiver-state module")
        _panel_title(ax, "A", "Weak and similar\nreceiver-state associations")
        ax.legend(frameon=False, loc="lower right")
        ax.grid(axis="x", color="#E4E7EA", linewidth=0.8)

        ax = axes[1]
        delta_order = [
            (module, score)
            for module in module_order
            for score in ("Edge weight × LR", "Message magnitude × LR")
        ]
        y_positions = np.arange(len(delta_order))[::-1]
        labels = []
        for y, (module, score) in zip(y_positions, delta_order):
            row = delta_r2.loc[
                delta_r2["module_label"].eq(module) & delta_r2["score_label"].eq(score)
            ].iloc[0]
            value = 100.0 * float(row["delta_r2"])
            color, marker = score_style[score]
            ax.plot([0, value], [y, y], color=color, linewidth=2.5, alpha=0.75)
            ax.scatter([value], [y], color=color, marker=marker, s=72, zorder=3)
            ax.text(
                value + 0.012,
                y,
                f"{value:.3g}%",
                color="#30353A",
                fontsize=10,
                va="center",
                ha="left",
            )
            module_short = (
                "Compact state" if module.startswith("Compact") else "Broad module"
            )
            score_short = (
                "Edge weight" if score.startswith("Edge") else "Message magnitude"
            )
            labels.append(f"{module_short}\n{score_short}")
        ax.axvline(0, color="#9FA6AD", linewidth=0.9)
        threshold_percent = 100.0 * float(material_delta_r2)
        ax.axvline(
            threshold_percent,
            color=FAIL_RED,
            linestyle="--",
            linewidth=2.0,
        )
        ax.text(
            threshold_percent,
            float(np.max(y_positions)) - 0.10,
            f"material threshold\n{threshold_percent:g}%",
            color=FAIL_RED,
            fontsize=9.5,
            ha="right",
            va="top",
        )
        ax.set_yticks(y_positions)
        ax.set_yticklabels(labels, fontsize=9.3)
        ax.set_xlim(0.0, max(0.55, 1.08 * threshold_percent))
        ax.set_xlabel(
            "Additional variance explained (%)\n"
            "after cell type + receptor expression + LR expression"
        )
        _panel_title(ax, "B", "<0.02% added variance\nbeyond LR expression")
        ax.grid(axis="x", color="#E4E7EA", linewidth=0.8)

        ax = axes[2]
        null_values = pd.to_numeric(null["null_mean_projection"], errors="raise").to_numpy(float)
        stats_row = case_statistics.iloc[0]
        observed = float(stats_row["observed_mean_projection"])
        null_mean = float(stats_row["null_mean"])
        ax.hist(null_values, bins=38, color="#C9CED5", edgecolor="white", linewidth=0.4)
        ax.axvline(
            null_mean,
            color=LR_GREY,
            linestyle="--",
            linewidth=2,
            label="Receiver-clustered reference",
        )
        ax.axvline(
            observed,
            color=FAIL_RED,
            linewidth=2.6,
            label="LR-compatible unique edges",
        )
        projection_min = min(float(np.min(null_values)), observed)
        projection_max = max(float(np.max(null_values)), observed)
        projection_pad = 0.05 * max(projection_max - projection_min, 1e-12)
        ax.set_xlim(projection_min - projection_pad, projection_max + projection_pad)
        ax.set_xlabel("Projection onto cross-fitted Her direction")
        ax.set_ylabel("Permutations")
        _panel_title(
            ax,
            "C",
            "Exploratory unique-edge contrast\nclustered by receiver",
        )
        require_columns(
            case_statistics,
            [
                "n_lr_compatible_unique_edges",
                "n_lr_compatible_unique_edges_matched",
                "n_lr_compatible_unique_edges_dropped_no_control",
                "n_matched_receiver_clusters",
                "exploratory_tail_fraction_greater",
            ],
            "receiver-clustered case statistics",
        )
        matched = int(stats_row["n_lr_compatible_unique_edges_matched"])
        total = int(stats_row["n_lr_compatible_unique_edges"])
        dropped = int(stats_row["n_lr_compatible_unique_edges_dropped_no_control"])
        n_receivers = int(stats_row["n_matched_receiver_clusters"])
        crossfit_rho = float(stats_row["crossfit_latent_response_spearman_rho"])
        ax.text(
            0.03,
            0.96,
            (
                f"matched unique edges = {matched:,}/{total:,}\n"
                f"{dropped:,} unmatched unique edges excluded\n"
                f"receiver clusters = {n_receivers:,}\n"
                f"cross-fit direction validity: ρ={crossfit_rho:.2f}\n"
                "exploratory upper-tail fraction="
                f"{float(stats_row['exploratory_tail_fraction_greater']):.3g}"
            ),
            transform=ax.transAxes,
            va="top",
            fontsize=9.6,
        )
        ax.legend(frameon=False, loc="lower right")

        figure.text(
            0.5,
            0.95,
            "OBSERVATIONAL AUDIT  •  NO PERTURBATION  •  NO LR-SPECIFIC KNOCKOUT",
            ha="center",
            va="center",
            fontsize=15,
            fontweight="bold",
            color=FAIL_RED,
            bbox={
                "boxstyle": "round,pad=0.45",
                "facecolor": "#FFF2F0",
                "edgecolor": "#E2A29D",
                "linewidth": 1.0,
            },
        )
        figure.text(
            0.5,
            0.805,
            "LR-compatible organization is visible, but model weighting adds no material receiver-state information beyond expression.",
            ha="center",
            fontsize=12,
            color=MID_GREY,
        )
        output_png.parent.mkdir(parents=True, exist_ok=True)
        output_pdf = output_png.with_suffix(".pdf")
        figure.savefig(output_png, dpi=int(dpi), bbox_inches="tight", facecolor="white")
        figure.savefig(output_pdf, bbox_inches="tight", facecolor="white")
        plt.close(figure)
    return output_png, output_pdf


def load_reused_spatial_and_edges(
    directory: Path,
    audit_cells: pd.DataFrame,
    *,
    expected_grouping_seeds: int,
    min_seed_support: int,
    max_display_edges: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Path]]:
    """Load and validate local formal figure data for a render-only refresh."""
    root = directory.expanduser().resolve()
    if (root / "figure_data").is_dir():
        root = root / "figure_data"
    spatial_path = root / "panel_abc_spatial_cells.csv.gz"
    supported_path = root / "panel_d_supported_edges.csv.gz"
    if not spatial_path.is_file() or not supported_path.is_file():
        raise FileNotFoundError(
            "Reuse mode requires panel_abc_spatial_cells.csv.gz and "
            f"panel_d_supported_edges.csv.gz under {root}"
        )
    spatial = pd.read_csv(spatial_path)
    supported = pd.read_csv(supported_path)
    require_columns(
        spatial,
        ["stage_index", "x", "y", "cell_type", "is_ventral_spinal"],
        "reused spatial-cell table",
    )
    require_columns(
        audit_cells,
        ["stage_index", "cell_type", "her4.1"],
        "family-audit receiver cells",
    )
    spatial["stage_index"] = pd.to_numeric(
        spatial["stage_index"], errors="raise"
    ).astype(int)
    if spatial["stage_index"].duplicated().any():
        raise ValueError("Reused spatial stage_index is not unique")
    audit = audit_cells[["stage_index", "cell_type", "her4.1"]].copy()
    audit["stage_index"] = pd.to_numeric(audit["stage_index"], errors="raise").astype(int)
    audit = audit.rename(columns={"cell_type": "audit_cell_type", "her4.1": "audit_her4.1"})
    spatial = spatial.merge(audit, on="stage_index", how="left", validate="one_to_one")
    if spatial[["audit_cell_type", "audit_her4.1"]].isna().any().any():
        raise ValueError("Family audit does not cover every reused spatial cell")
    if not np.array_equal(
        spatial["cell_type"].astype(str).to_numpy(),
        spatial["audit_cell_type"].astype(str).to_numpy(),
    ):
        raise ValueError("Reused spatial cell types disagree with the family audit")
    if "her4.1_activity" in spatial:
        existing = pd.to_numeric(spatial["her4.1_activity"], errors="raise").to_numpy(float)
        if not np.allclose(
            existing,
            pd.to_numeric(spatial["audit_her4.1"], errors="raise").to_numpy(float),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("Reused her4.1 activity disagrees with the family audit")
    spatial["her4.1_activity"] = pd.to_numeric(
        spatial.pop("audit_her4.1"), errors="raise"
    ).to_numpy(float)
    spatial = spatial.drop(columns=["audit_cell_type"])

    total_seeds = pd.to_numeric(
        supported["n_total_grouping_seeds"], errors="raise"
    ).astype(int)
    if set(total_seeds) != {int(expected_grouping_seeds)}:
        raise ValueError(
            "Reused supported edges disagree with expected grouping-run count"
        )
    supported, display = rerank_reused_supported_edges(
        supported,
        min_seed_support=min_seed_support,
        max_display_edges=max_display_edges,
    )

    xy = spatial.set_index("stage_index")[["x", "y"]]
    for role in ("source", "target"):
        indices = pd.to_numeric(
            supported[f"_{role}_stage_index"], errors="raise"
        ).astype(int)
        if not indices.isin(xy.index).all():
            raise ValueError(f"Reused {role} edge indices are outside the spatial table")
        expected_xy = xy.loc[indices].to_numpy(float)
        reported_xy = supported[[f"{role}_x", f"{role}_y"]].to_numpy(float)
        if not np.allclose(expected_xy, reported_xy, rtol=0.0, atol=1e-10):
            raise ValueError(f"Reused {role} coordinates disagree with spatial cells")
    return spatial, supported, display, {
        "reused_spatial_cells": spatial_path,
        "reused_supported_edges": supported_path,
    }


def write_figure_data(
    directory: Path,
    *,
    spatial_cells: pd.DataFrame,
    supported_edges: pd.DataFrame,
    display_edges: pd.DataFrame,
    circuit_percentiles: pd.DataFrame,
    claim_ladder: pd.DataFrame,
    correlations: pd.DataFrame,
    delta_r2: pd.DataFrame,
    null: pd.DataFrame,
    mass: pd.DataFrame,
    detection: pd.DataFrame,
    case_statistics: pd.DataFrame,
) -> dict[str, Path]:
    directory.mkdir(parents=True, exist_ok=False)
    paths = {
        "panel_abc_spatial_cells": directory / "panel_abc_spatial_cells.csv.gz",
        "panel_d_supported_edges": directory / "panel_d_supported_edges.csv.gz",
        "panel_d_display_edges": directory / "panel_d_display_edges.csv",
        "panel_e_exact_context_percentiles": directory / "panel_e_exact_context_percentiles.csv",
        "panel_f_claim_ladder": directory / "panel_f_claim_ladder.csv",
        "controls_correlations": directory / "controls_correlations.csv",
        "controls_delta_r2": directory / "controls_delta_r2.csv",
        "controls_matched_projection_null": directory / "controls_matched_projection_null.csv.gz",
        "family_context_mass": directory / "family_context_mass.csv",
        "expression_detection": directory / "expression_detection.csv",
        "case_statistics": directory / "case_statistics.csv",
    }
    spatial_cells.to_csv(paths["panel_abc_spatial_cells"], index=False)
    supported_edges.to_csv(paths["panel_d_supported_edges"], index=False)
    display_edges.to_csv(paths["panel_d_display_edges"], index=False)
    circuit_percentiles.to_csv(paths["panel_e_exact_context_percentiles"], index=False)
    claim_ladder.to_csv(paths["panel_f_claim_ladder"], index=False)
    correlations.to_csv(paths["controls_correlations"], index=False)
    delta_r2.to_csv(paths["controls_delta_r2"], index=False)
    null.to_csv(paths["controls_matched_projection_null"], index=False)
    mass.to_csv(paths["family_context_mass"], index=False)
    detection.to_csv(paths["expression_detection"], index=False)
    case_statistics.to_csv(paths["case_statistics"], index=False)
    return paths


def main() -> None:
    args = parser().parse_args()
    output = args.output_dir.expanduser().resolve()
    prepare_output(output, args.overwrite)
    figures = output / "figures"
    figures.mkdir()

    family_tables, family_manifest, family_paths = load_family_audit(
        args.delta_notch_family_audit
    )
    case_statistics, null, case_manifest, case_paths = load_case_study(
        args.delta_notch_case_study
    )
    if not np.isclose(float(family_manifest.get("stage", np.nan)), args.stage):
        raise ValueError("Family audit stage disagrees with requested figure stage")
    family_label = str(family_manifest.get("stage_label", ""))
    if family_label != str(args.stage_label):
        raise ValueError("Family audit stage label disagrees with requested figure")
    case = case_manifest.get("case", {})
    if not isinstance(case, Mapping):
        raise ValueError("Case-study manifest lacks a case contract")
    if (
        not np.isclose(float(case.get("stage", np.nan)), args.stage)
        or str(case.get("stage_label", "")) != str(args.stage_label)
        or str(case.get("sender_type", "")) != str(args.sender_type)
        or str(case.get("receiver_type", "")) != str(args.receiver_type)
    ):
        raise ValueError("Case-study manifest disagrees with the requested exact circuit")

    observed_path: Path | None = None
    reused_source_paths: dict[str, Path] = {}
    if args.reuse_figure_data_dir is not None:
        spatial_cells, supported_edges, display_edges, reused_source_paths = (
            load_reused_spatial_and_edges(
                args.reuse_figure_data_dir,
                family_tables["cells"],
                expected_grouping_seeds=args.expected_grouping_seeds,
                min_seed_support=args.min_seed_support,
                max_display_edges=args.max_display_edges,
            )
        )
        input_mode = "validated_local_figure_data_reuse"
    else:
        if args.h5ad is None or args.stage4_edge_dir is None:
            raise ValueError(
                "--h5ad and --stage4-edge-dir are required unless "
                "--reuse-figure-data-dir is used"
            )
        data = ad.read_h5ad(args.h5ad.expanduser().resolve())
        spatial_cells, stage_mask = build_spatial_cell_table(
            data,
            family_tables["cells"],
            stage=args.stage,
            time_key=args.time_key,
            annotation_key=args.annotation_key,
            spatial_key=args.spatial_key,
            receiver_type=args.receiver_type,
        )
        raw_edges, edge_inventory, observed_path = load_stage_edge_occurrences(
            args.stage4_edge_dir,
            data,
            stage_mask,
            stage=args.stage,
            stage_label=args.stage_label,
            annotation_key=args.annotation_key,
            observed_cells_path=args.observed_cells,
            expected_seeds=args.expected_grouping_seeds,
        )
        validate_edge_inventory(edge_inventory, family_tables["inventory"])
        supported_edges, display_edges = collapse_display_edges(
            raw_edges,
            spatial_cells,
            sender_type=args.sender_type,
            receiver_type=args.receiver_type,
            min_seed_support=args.min_seed_support,
            max_display_edges=args.max_display_edges,
        )
        input_mode = "raw_h5ad_and_edge_tables"

    circuit_path = resolve_circuit_screen(args.four_axis_circuit_screen)
    circuit_percentiles = prepare_circuit_percentiles(
        pd.read_csv(circuit_path),
        stage=args.stage,
        sender_type=args.sender_type,
        receiver_type=args.receiver_type,
        cytobridge_rank_score=args.cytobridge_rank_score,
    )
    correlations, delta_r2 = prepare_control_tables(
        family_tables["correlations"], family_tables["residuals"]
    )
    claims = build_claim_ladder(
        circuit_percentiles,
        family_tables["detection"],
        delta_r2,
        receiver_type=args.receiver_type,
        material_delta_r2=args.material_delta_r2,
        n_supported_edges=len(supported_edges),
        n_display_edges=len(display_edges),
        min_seed_support=args.min_seed_support,
        n_grouping_seeds=args.expected_grouping_seeds,
    )

    figure_data_paths = write_figure_data(
        output / "figure_data",
        spatial_cells=spatial_cells,
        supported_edges=supported_edges,
        display_edges=display_edges,
        circuit_percentiles=circuit_percentiles,
        claim_ladder=claims,
        correlations=correlations,
        delta_r2=delta_r2,
        null=null,
        mass=family_tables["mass"],
        detection=family_tables["detection"],
        case_statistics=case_statistics,
    )
    main_png, main_pdf = plot_main_figure(
        spatial_cells,
        display_edges,
        circuit_percentiles,
        claims,
        stage_label=args.stage_label,
        receiver_type=args.receiver_type,
        n_supported_edges=len(supported_edges),
        output_png=figures / "delta_notch_biology_main.png",
        dpi=args.dpi,
    )
    controls_png, controls_pdf = plot_controls_figure(
        correlations,
        delta_r2,
        null,
        case_statistics,
        material_delta_r2=args.material_delta_r2,
        output_png=figures / "delta_notch_biology_controls.png",
        dpi=args.dpi,
    )

    source_paths = {
        "circuit_screen": circuit_path,
        **{f"family_{key}": value for key, value in family_paths.items()},
        **{f"case_{key}": value for key, value in case_paths.items()},
        **reused_source_paths,
    }
    if args.h5ad is not None:
        source_paths["h5ad"] = args.h5ad.expanduser().resolve()
    if observed_path is not None:
        source_paths["observed_cells"] = observed_path.expanduser().resolve()
    output_paths = {
        "main_png": main_png,
        "main_pdf": main_pdf,
        "controls_png": controls_png,
        "controls_pdf": controls_pdf,
        **figure_data_paths,
    }
    manifest = {
        "schema_version": 2,
        "analysis": "zebrafish_delta_notch_reader_facing_biology_figure",
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_mode": input_mode,
        "stage": float(args.stage),
        "stage_label": str(args.stage_label),
        "exact_circuit": {
            "sender_type": str(args.sender_type),
            "receiver_type": str(args.receiver_type),
            "pairs": list(PAIR_ORDER),
        },
        "edge_display": {
            "n_grouping_seeds": int(args.expected_grouping_seeds),
            "minimum_seed_support": int(args.min_seed_support),
            "maximum_display_edges": int(args.max_display_edges),
            "n_supported_edges": int(len(supported_edges)),
            "n_display_edges": int(len(display_edges)),
            "collapse": "mean over seeds in which the edge occurs; missing seeds are not zero-imputed",
            "ranking": "mean model edge weight x family LR activity x grouping-run support fraction",
        },
        "panel_e": {
            "cytobridge_rank_score": str(args.cytobridge_rank_score),
            "unit": "rank from best within stage and ligand-receptor axis",
        },
        "claim_thresholds": {
            "high_exact_context": "top quartile in both displayed methods",
            "material_attention_delta_r2": float(args.material_delta_r2),
        },
        "guardrails": {
            "observational_no_perturbation": True,
            "attention_is_lr_specific": False,
            "exact_message_is_lr_specific": False,
            "messages_were_deleted": False,
            "trajectory_was_rerun": False,
            "grouping_seeds_are_biological_replicates": False,
            "edge_index_mapping_was_silently_assumed": False,
        },
        "inputs": {name: file_record(path) for name, path in source_paths.items()},
        "outputs": {name: file_record(path) for name, path in output_paths.items()},
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "main_figure": str(main_png),
                "controls_figure": str(controls_png),
                "manifest": str(manifest_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
