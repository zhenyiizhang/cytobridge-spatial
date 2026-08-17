#!/usr/bin/env python3
"""Build an auditable 18 hpf zebrafish Somite JAM biology-first case study.

The script deliberately separates four observational score constructions on the
same CytoBridge edge scaffold: raw attention magnitude, LR activity alone,
``|attention| x LR``, and exact edge-message norm x LR.  It also joins sparse
external CCI results under an explicit axis-availability contract and computes
cross-sectional expression/spatial checks for the Jam2a--Jam3b myocyte-fusion
case.  None of these post-hoc quantities is an LR-specific model mechanism,
causal perturbation, or proof of direct molecular contact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse, stats
from scipy.spatial import cKDTree


CASE_AXES = (("jam2a", "jam3b"), ("jam3b", "jam2a"))
SCORE_COLUMNS = (
    "cytobridge_raw_attention_magnitude_density",
    "cytobridge_lr_only_density",
    "cytobridge_attention_lr_density",
    "cytobridge_exact_message_lr_density",
)
BIOLOGY_18_GENES = ("jam2a", "jam3b", "myog", "mymk")
BIOLOGY_24_GENES = (
    "jam2a",
    "jam3b",
    "myog",
    "mymk",
    "mylpfa",
    "acta1a",
    "tnnt3a",
)


@dataclass(frozen=True)
class ExternalSpec:
    method: str
    table: Path
    availability: Path
    score_column: str
    score_mode: str

    @property
    def slug(self) -> str:
        value = re.sub(r"[^a-z0-9]+", "_", self.method.casefold()).strip("_")
        if not value:
            raise ValueError(f"External method has no usable slug: {self.method!r}")
        return value


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--h5ad", required=True, type=Path)
    result.add_argument(
        "--edge-dir",
        required=True,
        type=Path,
        help="18 hpf attribution directory containing exactly five edges_seed_*.csv.gz files.",
    )
    result.add_argument(
        "--observed-cells",
        required=True,
        type=Path,
        help="Attribution observed_cells.csv.gz; obs_name mapping is mandatory.",
    )
    result.add_argument(
        "--provenance",
        type=Path,
        default=Path(__file__).with_name("jam_myocyte_axis_provenance.csv"),
    )
    result.add_argument(
        "--external-spec",
        action="append",
        nargs=5,
        default=[],
        metavar=("METHOD", "TABLE", "AVAILABILITY", "SCORE_COLUMN", "SCORE_MODE"),
        help=(
            "Repeatable external method specification. SCORE_MODE is one of "
            "distinct_density, distinct_mass, or native_rank_only. An explicit "
            "stage x LR availability table is required for every supplied "
            "external method. Omit this option for a CytoBridge-only biology audit."
        ),
    )
    result.add_argument("--output-dir", required=True, type=Path)
    result.add_argument("--stage", type=float, default=3.0)
    result.add_argument("--stage-label", default="18hpf")
    result.add_argument(
        "--biological-time-hpf",
        type=float,
        default=18.0,
        help="Fallback external stage_time target only when numeric internal stage is absent.",
    )
    result.add_argument("--sender-type", default="Somite")
    result.add_argument("--receiver-type", default="Somite")
    result.add_argument("--maturity-stage", type=float, default=4.0)
    result.add_argument("--maturity-stage-label", default="24hpf")
    result.add_argument("--maturity-cell-type", default="Fast Muscle Cell")
    result.add_argument("--time-key", default="time_point_processed")
    result.add_argument("--time-label-key", default="time")
    result.add_argument("--annotation-key", default="Annotation")
    result.add_argument("--spatial-key", default="spatial")
    result.add_argument("--spatial-cutoff", type=float, default=0.096063674)
    result.add_argument(
        "--spatial-cutoff-source",
        default="frozen zebrafish preprocessing graph cutoff",
    )
    result.add_argument("--preprocess-manifest", type=Path)
    result.add_argument(
        "--trained-pre-interaction-random-control",
        "--trained-init-random-control",
        dest="trained_pre_interaction_random_control",
        type=Path,
        help=(
            "Optional audited trained/pre-interaction/random CytoBridge control "
            "metrics table. It is copied with a source hash; this script does not "
            "reconstruct controls. The trained-init spelling is deprecated."
        ),
    )
    result.add_argument("--permutation-seed", type=int, default=20260722)
    result.add_argument("--n-permutations", type=int, default=10_000)
    result.add_argument("--min-active-edges", type=int, default=5)
    result.add_argument("--min-cells-per-side", type=int, default=10)
    result.add_argument(
        "--expected-grouping-seeds",
        default="101,202,303,404,505",
        help="Exactly five comma-separated technical grouping seeds.",
    )
    result.add_argument("--overwrite", action="store_true")
    return result


def require(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    missing = [column for column in columns if column not in frame]
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "bytes": int(path.stat().st_size),
    }


def prepare_output(path: Path, overwrite: bool) -> None:
    if path.exists() and not path.is_dir():
        raise NotADirectoryError(path)
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory is non-empty; pass --overwrite explicitly: {path}"
            )
        shutil.rmtree(path)
    elif path.exists() and overwrite:
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=False)


def integer_values(series: pd.Series, label: str) -> np.ndarray:
    numeric = pd.to_numeric(series, errors="raise").to_numpy(float)
    if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
        raise ValueError(f"{label} must contain finite integer values")
    return numeric.astype(int)


def parse_seeds(value: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise ValueError("--expected-grouping-seeds must be comma-separated integers") from error
    if len(seeds) != 5 or len(set(seeds)) != 5:
        raise ValueError("Exactly five unique grouping seeds are required")
    return seeds


def parse_external_specs(values: Sequence[Sequence[str]]) -> list[ExternalSpec]:
    valid_modes = {"distinct_density", "distinct_mass", "native_rank_only"}
    result: list[ExternalSpec] = []
    for method, table, availability, score_column, score_mode in values:
        if score_mode not in valid_modes:
            raise ValueError(
                f"Invalid external score mode {score_mode!r}; expected {sorted(valid_modes)}"
            )
        spec = ExternalSpec(
            method=method,
            table=Path(table),
            availability=Path(availability),
            score_column=score_column,
            score_mode=score_mode,
        )
        result.append(spec)
    slugs = [item.slug for item in result]
    if len(slugs) != len(set(slugs)):
        raise ValueError(f"External method slugs must be unique: {slugs}")
    return result


def load_provenance(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = [
        "ligand",
        "receptor",
        "evidence_scope",
        "claim_guardrail",
        "source_ids",
        "source_urls",
    ]
    require(frame, required, "JAM literature provenance")
    frame = frame.copy()
    frame["ligand"] = frame["ligand"].astype(str).str.casefold()
    frame["receptor"] = frame["receptor"].astype(str).str.casefold()
    frame["axis_id"] = frame["ligand"] + "->" + frame["receptor"]
    if frame.duplicated(["ligand", "receptor"]).any():
        raise ValueError("JAM provenance contains duplicate ligand-receptor axes")
    expected = {f"{ligand}->{receptor}" for ligand, receptor in CASE_AXES}
    observed = set(frame["axis_id"])
    if observed != expected:
        raise ValueError(
            f"JAM provenance must contain exactly both orientations: expected={sorted(expected)}, "
            f"observed={sorted(observed)}"
        )
    for column in ("evidence_scope", "claim_guardrail", "source_ids", "source_urls"):
        if frame[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"JAM provenance has an empty {column}")
    if not frame["source_urls"].astype(str).str.contains("https://", regex=False).all():
        raise ValueError("JAM provenance source_urls must contain auditable HTTPS links")
    return frame


def map_observed_cells(
    data: ad.AnnData,
    observed: pd.DataFrame,
    *,
    stage: float,
    stage_label: str,
    time_key: str,
    time_label_key: str,
    annotation_key: str,
) -> tuple[pd.DataFrame, dict[int, int], dict[int, int]]:
    """Map attribution global indices through obs_name, never through row order."""
    require(
        observed,
        ["global_index", "obs_name", "stage", "stage_label", "cell_type"],
        "observed-cells table",
    )
    for key in (time_key, time_label_key, annotation_key):
        if key not in data.obs:
            raise KeyError(f"H5AD is missing obs[{key!r}]")
    table = observed.copy()
    table["global_index"] = integer_values(table["global_index"], "global_index")
    expected_global = np.arange(data.n_obs, dtype=int)
    if len(table) != data.n_obs or not np.array_equal(
        np.sort(table["global_index"].to_numpy(int)), expected_global
    ):
        raise ValueError("observed-cells global_index must be exactly 0..n_obs-1")
    table["obs_name"] = table["obs_name"].astype(str)
    if table["global_index"].duplicated().any() or table["obs_name"].duplicated().any():
        raise ValueError("observed-cells global_index and obs_name must both be unique")
    if not data.obs_names.is_unique:
        raise ValueError("H5AD obs_names must be unique")
    h5_names = data.obs_names.astype(str)
    if len(table) != data.n_obs or set(table["obs_name"]) != set(h5_names):
        raise ValueError(
            "observed-cells and H5AD must contain exactly the same obs_name universe"
        )
    h5_lookup = {name: index for index, name in enumerate(h5_names)}
    table["h5ad_index"] = table["obs_name"].map(h5_lookup).astype(int)
    h5_index = table["h5ad_index"].to_numpy(int)
    table["stage"] = pd.to_numeric(table["stage"], errors="raise").astype(float)
    h5_stage = pd.to_numeric(data.obs.iloc[h5_index][time_key], errors="raise").to_numpy(float)
    if not np.isclose(table["stage"], h5_stage, rtol=0.0, atol=1e-12).all():
        raise ValueError("observed-cells stage disagrees with H5AD after obs_name mapping")
    if not np.array_equal(
        table["stage_label"].astype(str).to_numpy(),
        data.obs.iloc[h5_index][time_label_key].astype(str).to_numpy(),
    ):
        raise ValueError("observed-cells stage_label disagrees with H5AD after obs_name mapping")
    if not np.array_equal(
        table["cell_type"].astype(str).to_numpy(),
        data.obs.iloc[h5_index][annotation_key].astype(str).to_numpy(),
    ):
        raise ValueError("observed-cells cell_type disagrees with H5AD after obs_name mapping")

    selected = table.loc[
        np.isclose(table["stage"], float(stage), rtol=0.0, atol=1e-12)
    ].copy()
    if selected.empty:
        raise ValueError(f"observed-cells has no rows for stage={stage:g}")
    if set(selected["stage_label"].astype(str)) != {str(stage_label)}:
        raise ValueError("Requested stage label disagrees with observed-cells")
    selected = selected.sort_values("global_index").copy()
    selected["attribution_stage_index"] = np.arange(len(selected), dtype=int)
    global_to_h5 = dict(zip(table["global_index"], table["h5ad_index"]))
    global_to_stage = dict(
        zip(selected["global_index"], selected["attribution_stage_index"])
    )
    return table, global_to_h5, global_to_stage


def resolve_edge_endpoints(
    edges: pd.DataFrame,
    data: ad.AnnData,
    *,
    observed_table: pd.DataFrame,
    global_to_h5: Mapping[int, int],
    global_to_stage: Mapping[int, int],
    stage: float,
    stage_label: str,
    time_key: str,
    annotation_key: str,
) -> pd.DataFrame:
    """Resolve endpoints by observed-cells obs_name and cross-check local indices."""
    require(
        edges,
        ["source_index", "target_index", "sender_type", "receiver_type"],
        "CytoBridge edge table",
    )
    result = edges.copy()
    source_global = integer_values(result["source_index"], "source_index")
    target_global = integer_values(result["target_index"], "target_index")
    all_indices = set(source_global).union(target_global)
    missing = sorted(all_indices.difference(global_to_h5))
    if missing:
        raise ValueError(f"Edge global indices are absent from observed-cells: {missing[:5]}")
    outside = sorted(all_indices.difference(global_to_stage))
    if outside:
        raise ValueError(f"Edge endpoints map outside requested stage: {outside[:5]}")

    source_h5 = np.asarray([global_to_h5[value] for value in source_global], dtype=int)
    target_h5 = np.asarray([global_to_h5[value] for value in target_global], dtype=int)
    source_stage = np.asarray([global_to_stage[value] for value in source_global], dtype=int)
    target_stage = np.asarray([global_to_stage[value] for value in target_global], dtype=int)

    has_source_local = "source_index_stage" in result
    has_target_local = "target_index_stage" in result
    if has_source_local != has_target_local:
        raise ValueError(
            "Edge table must contain both source_index_stage and target_index_stage, or neither"
        )
    if has_source_local:
        supplied_source = integer_values(result["source_index_stage"], "source_index_stage")
        supplied_target = integer_values(result["target_index_stage"], "target_index_stage")
        if not np.array_equal(supplied_source, source_stage) or not np.array_equal(
            supplied_target, target_stage
        ):
            raise ValueError(
                "Stage-local edge indices disagree with the obs_name-verified attribution mapping"
            )

    h5_stage = pd.to_numeric(data.obs[time_key], errors="raise").to_numpy(float)
    if not np.isclose(h5_stage[source_h5], stage, rtol=0.0, atol=1e-12).all() or not np.isclose(
        h5_stage[target_h5], stage, rtol=0.0, atol=1e-12
    ).all():
        raise ValueError("Mapped edge endpoint is outside the requested H5AD stage")
    if "stage" in result:
        edge_stage = pd.to_numeric(result["stage"], errors="raise").to_numpy(float)
        if not np.isclose(edge_stage, stage, rtol=0.0, atol=1e-12).all():
            raise ValueError("Edge table contains an unexpected stage")
    if "stage_label" in result and set(result["stage_label"].astype(str)) != {
        str(stage_label)
    }:
        raise ValueError("Edge table contains an unexpected stage label")
    annotation = data.obs[annotation_key].astype(str).to_numpy()
    if not np.array_equal(result["sender_type"].astype(str), annotation[source_h5]):
        raise ValueError("Edge sender_type disagrees with obs_name-mapped H5AD annotation")
    if not np.array_equal(result["receiver_type"].astype(str), annotation[target_h5]):
        raise ValueError("Edge receiver_type disagrees with obs_name-mapped H5AD annotation")
    result["_source_h5ad_index"] = source_h5
    result["_target_h5ad_index"] = target_h5
    result["_source_attribution_stage_index"] = source_stage
    result["_target_attribution_stage_index"] = target_stage
    return result


def load_edge_tables(
    edge_dir: Path,
    data: ad.AnnData,
    observed_path: Path,
    *,
    expected_seeds: Sequence[int],
    stage: float,
    stage_label: str,
    time_key: str,
    time_label_key: str,
    annotation_key: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    if not observed_path.is_file():
        raise FileNotFoundError(observed_path)
    observed = pd.read_csv(observed_path)
    observed_table, global_to_h5, global_to_stage = map_observed_cells(
        data,
        observed,
        stage=stage,
        stage_label=stage_label,
        time_key=time_key,
        time_label_key=time_label_key,
        annotation_key=annotation_key,
    )
    paths = sorted(edge_dir.glob("edges_seed_*.csv.gz"))
    if not paths:
        raise FileNotFoundError(f"No edges_seed_*.csv.gz files under {edge_dir}")
    frames: list[pd.DataFrame] = []
    inventory: list[dict[str, object]] = []
    observed_seeds: list[int] = []
    required = [
        "grouping_seed",
        "source_index",
        "target_index",
        "sender_type",
        "receiver_type",
        "attention_abs_mean",
        "edge_message_norm_joint",
    ]
    for path in paths:
        frame = pd.read_csv(path)
        require(frame, required, str(path))
        if frame.empty:
            raise ValueError(f"Edge seed table is empty: {path}")
        seeds = integer_values(frame["grouping_seed"], "grouping_seed")
        unique = np.unique(seeds)
        if unique.size != 1:
            raise ValueError(f"Edge table contains multiple grouping seeds: {path}")
        seed = int(unique[0])
        filename_match = re.search(r"edges_seed_(-?\d+)\.csv\.gz$", path.name)
        if filename_match is None or int(filename_match.group(1)) != seed:
            raise ValueError(f"Edge filename seed disagrees with table content: {path}")
        resolved = resolve_edge_endpoints(
            frame,
            data,
            observed_table=observed_table,
            global_to_h5=global_to_h5,
            global_to_stage=global_to_stage,
            stage=stage,
            stage_label=stage_label,
            time_key=time_key,
            annotation_key=annotation_key,
        )
        duplicate_key = ["grouping_seed", "_source_h5ad_index", "_target_h5ad_index"]
        if resolved.duplicated(duplicate_key).any():
            raise ValueError(f"Duplicate directed source-target edge within seed: {path}")
        for column in ("attention_abs_mean", "edge_message_norm_joint"):
            values = pd.to_numeric(resolved[column], errors="raise").to_numpy(float)
            if not np.isfinite(values).all() or (values < 0).any():
                raise ValueError(f"{column} must contain finite nonnegative values: {path}")
        n_self = int(
            resolved["_source_h5ad_index"].eq(resolved["_target_h5ad_index"]).sum()
        )
        frames.append(resolved)
        observed_seeds.append(seed)
        inventory.append(
            {
                "path": str(path.resolve()),
                "sha256": sha256(path),
                "grouping_seed": seed,
                "n_edge_rows": int(len(resolved)),
                "n_self_edge_rows_excluded_downstream": n_self,
            }
        )
    if len(observed_seeds) != len(set(observed_seeds)):
        raise ValueError("Multiple edge files represent the same grouping seed")
    if set(observed_seeds) != set(map(int, expected_seeds)):
        raise ValueError(
            f"Expected exactly grouping seeds {list(expected_seeds)}, observed {sorted(observed_seeds)}"
        )
    resolution = {
        "mode": "observed_cells.obs_name_to_h5ad",
        "observed_cells": artifact(observed_path),
        "global_index_order_assumed": False,
        "stage_local_columns_used_for_expression_lookup": False,
        "stage_local_columns_crosschecked_when_present": True,
        "n_observed_cells": int(len(observed_table)),
    }
    return pd.concat(frames, ignore_index=True), pd.DataFrame(inventory), resolution


def casefold_gene_lookup(var_names: Iterable[str]) -> dict[str, list[int]]:
    """Index every case-folded gene without rejecting unrelated duplicates."""
    grouped: dict[str, list[int]] = {}
    for index, gene in enumerate(var_names):
        grouped.setdefault(str(gene).casefold(), []).append(index)
    return grouped


def dense_column(matrix, index: int) -> np.ndarray:
    values = matrix[:, int(index)]
    values = values.toarray() if sparse.issparse(values) else np.asarray(values)
    return np.asarray(values, dtype=np.float64).reshape(-1)


def load_gene_values(
    data: ad.AnnData, genes: Iterable[str]
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    lookup = casefold_gene_lookup(data.var_names.astype(str))
    values: dict[str, np.ndarray] = {}
    audit: list[dict[str, object]] = []
    for gene in sorted(set(str(item).casefold() for item in genes)):
        matches = lookup.get(gene, [])
        if not matches:
            audit.append(
                {
                    "gene": gene,
                    "available": False,
                    "positive_cells": 0,
                    "positive_q95_scale": np.nan,
                }
            )
            continue
        if len(matches) > 1:
            matched_names = [str(data.var_names[index]) for index in matches]
            raise ValueError(
                f"Required gene {gene!r} has {len(matches)} case-insensitive H5AD "
                f"matches: {matched_names}"
            )
        raw = dense_column(data.X, matches[0])
        if not np.isfinite(raw).all() or (raw < 0).any():
            raise ValueError(f"Gene {gene!r} contains non-finite or negative expression")
        positive = raw[raw > 0]
        scale = float(np.quantile(positive, 0.95)) if positive.size else 1.0
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0
        values[gene] = raw
        audit.append(
            {
                "gene": gene,
                "available": True,
                "positive_cells": int(positive.size),
                "positive_q95_scale": scale,
            }
        )
    missing = sorted(set(str(item).casefold() for item in genes).difference(values))
    if missing:
        raise ValueError(f"Required biology genes are absent from H5AD: {missing}")
    return values, pd.DataFrame(audit)


def q95_activities(
    values: Mapping[str, np.ndarray], gene_audit: pd.DataFrame
) -> dict[str, np.ndarray]:
    scale = gene_audit.set_index("gene")["positive_q95_scale"].to_dict()
    return {
        gene: np.clip(np.asarray(raw, dtype=float) / float(scale[gene]), 0.0, 1.0)
        for gene, raw in values.items()
    }


def build_spatial_panel_cells(
    data: ad.AnnData,
    gene_values: Mapping[str, np.ndarray],
    *,
    stage: float,
    cell_type: str,
    time_key: str,
    annotation_key: str,
    spatial_key: str,
) -> pd.DataFrame:
    """Freeze all selected-stage cells needed to redraw the JAM spatial panel."""
    if spatial_key not in data.obsm:
        raise KeyError(f"Missing H5AD obsm[{spatial_key!r}]")
    spatial = np.asarray(data.obsm[spatial_key], dtype=float)
    if spatial.shape != (data.n_obs, 2) or not np.isfinite(spatial).all():
        raise ValueError(f"{spatial_key} must be a finite N x 2 matrix")
    stage_values = pd.to_numeric(data.obs[time_key], errors="raise").to_numpy(float)
    selected = np.flatnonzero(
        np.isclose(stage_values, float(stage), rtol=0.0, atol=1e-12)
    )
    if selected.size == 0:
        raise ValueError(f"H5AD has no cells for stage={stage:g}")
    labels = data.obs.iloc[selected][annotation_key].astype(str).to_numpy()
    result = pd.DataFrame(
        {
            "h5ad_index": selected,
            "obs_name": data.obs_names[selected].astype(str),
            "cell_type": labels,
            "is_somite": labels == str(cell_type),
            "x": spatial[selected, 0],
            "y": spatial[selected, 1],
        }
    )
    for gene in ("jam2a", "jam3b", "myog"):
        values = np.asarray(gene_values[gene], dtype=float)[selected]
        result[gene] = values
        result[f"{gene}_positive"] = values > 0
    return result


def build_trained_jam_display_edges(
    edges: pd.DataFrame,
    spatial_cells: pd.DataFrame,
    *,
    cell_type: str,
    minimum_seed_support: int = 3,
    maximum_display_edges: int = 15,
) -> pd.DataFrame:
    """Select a deterministic, model-first set of JAM-compatible display edges."""
    if minimum_seed_support < 1 or maximum_display_edges < 1:
        raise ValueError("Display-edge support/count thresholds must be positive")
    required = [
        "grouping_seed",
        "sender_type",
        "receiver_type",
        "attention_abs_mean",
        "_source_h5ad_index",
        "_target_h5ad_index",
    ]
    require(edges, required, "trained attribution edges")
    require(
        spatial_cells,
        [
            "h5ad_index",
            "obs_name",
            "x",
            "y",
            "jam2a",
            "jam3b",
            "myog",
            "jam2a_positive",
            "jam3b_positive",
        ],
        "spatial panel cells",
    )
    cells = spatial_cells.set_index("h5ad_index", verify_integrity=True)
    local = edges.loc[
        edges["sender_type"].astype(str).eq(str(cell_type))
        & edges["receiver_type"].astype(str).eq(str(cell_type))
        & edges["_source_h5ad_index"].ne(edges["_target_h5ad_index"])
    ].copy()
    source = local["_source_h5ad_index"].to_numpy(int)
    target = local["_target_h5ad_index"].to_numpy(int)
    if not set(source).issubset(cells.index) or not set(target).issubset(cells.index):
        raise ValueError("Somite attribution endpoints are absent from spatial panel cells")
    source_jam2 = cells.loc[source, "jam2a_positive"].to_numpy(bool)
    source_jam3 = cells.loc[source, "jam3b_positive"].to_numpy(bool)
    target_jam2 = cells.loc[target, "jam2a_positive"].to_numpy(bool)
    target_jam3 = cells.loc[target, "jam3b_positive"].to_numpy(bool)
    forward = source_jam2 & target_jam3
    reverse = source_jam3 & target_jam2
    local["jam_compatible"] = forward | reverse
    local["jam_compatible_orientation"] = np.select(
        [forward & reverse, forward, reverse],
        ["both", "source_jam2a_target_jam3b", "source_jam3b_target_jam2a"],
        default="none",
    )
    local = local.loc[local["jam_compatible"]].copy()
    if local.empty:
        raise ValueError("No JAM-compatible trained Somite edges")
    n_total_seeds = int(edges["grouping_seed"].nunique())
    collapsed = (
        local.groupby(
            ["_source_h5ad_index", "_target_h5ad_index"],
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
            mean_attention=("attention_abs_mean", "mean"),
            median_attention=("attention_abs_mean", "median"),
            jam_compatible_orientation=(
                "jam_compatible_orientation",
                lambda values: ";".join(sorted(set(map(str, values)))),
            ),
        )
    )
    collapsed["n_total_grouping_seeds"] = n_total_seeds
    collapsed["seed_support_fraction"] = collapsed["seed_support"] / n_total_seeds
    collapsed["mean_attention_percentile_within_collapsed_jam_compatible_somite_edges"] = (
        collapsed["mean_attention"].rank(method="average", pct=True)
    )
    collapsed["trained_attention_percentile"] = collapsed[
        "mean_attention_percentile_within_collapsed_jam_compatible_somite_edges"
    ]
    collapsed["display_score"] = (
        collapsed["mean_attention"] * collapsed["seed_support_fraction"]
    )
    stable = collapsed.loc[
        collapsed["seed_support"].ge(int(minimum_seed_support))
    ].sort_values(
        [
            "display_score",
            "seed_support",
            "_source_h5ad_index",
            "_target_h5ad_index",
        ],
        ascending=[False, False, True, True],
        kind="stable",
    )
    if stable.empty:
        raise ValueError("No JAM-compatible edge passes the seed-support threshold")
    display = stable.head(int(maximum_display_edges)).copy().reset_index(drop=True)
    display.insert(0, "display_rank", np.arange(1, len(display) + 1, dtype=int))
    display["selection_rule"] = (
        "JAM-compatible Somite->Somite; seed support >= "
        f"{int(minimum_seed_support)}/{n_total_seeds}; descending mean attention "
        "x seed-support fraction; deterministic endpoint tie-break"
    )
    for role in ("source", "target"):
        index = display[f"_{role}_h5ad_index"].to_numpy(int)
        display[f"{role}_obs_name"] = cells.loc[index, "obs_name"].to_numpy(str)
        for column in ("x", "y", "jam2a", "jam3b", "myog"):
            display[f"{role}_{column}"] = cells.loc[index, column].to_numpy()
    return display


def distinct_pair_count(n_sender: int, n_receiver: int, n_intersection: int) -> int:
    n_sender = int(n_sender)
    n_receiver = int(n_receiver)
    n_intersection = int(n_intersection)
    if min(n_sender, n_receiver, n_intersection) < 0:
        raise ValueError("Cell counts cannot be negative")
    if n_intersection > min(n_sender, n_receiver):
        raise ValueError("Cell-set intersection exceeds a marginal cell count")
    return n_sender * n_receiver - n_intersection


def context_universe(
    data: ad.AnnData,
    *,
    stage: float,
    stage_label: str,
    time_key: str,
    annotation_key: str,
    axes: pd.DataFrame,
) -> pd.DataFrame:
    stage_values = pd.to_numeric(data.obs[time_key], errors="raise").to_numpy(float)
    mask = np.isclose(stage_values, stage, rtol=0.0, atol=1e-12)
    labels = data.obs.loc[mask, annotation_key].astype(str)
    counts = labels.value_counts().sort_index()
    if counts.empty:
        raise ValueError(f"H5AD has no context cells for stage={stage:g}")
    rows: list[dict[str, object]] = []
    for axis in axes.itertuples(index=False):
        for sender, receiver in product(counts.index, repeat=2):
            n_sender = int(counts[sender])
            n_receiver = int(counts[receiver])
            intersection = n_sender if sender == receiver else 0
            rows.append(
                {
                    "stage": float(stage),
                    "stage_label": str(stage_label),
                    "axis_id": axis.axis_id,
                    "ligand": axis.ligand,
                    "receptor": axis.receptor,
                    "sender_type": sender,
                    "receiver_type": receiver,
                    "n_sender_cells": n_sender,
                    "n_receiver_cells": n_receiver,
                    "n_shared_sender_receiver_cells": intersection,
                    "n_possible_distinct_cell_pairs": distinct_pair_count(
                        n_sender, n_receiver, intersection
                    ),
                }
            )
    return pd.DataFrame(rows)


def score_cytobridge_contexts(
    edges: pd.DataFrame,
    universe: pd.DataFrame,
    activities: Mapping[str, np.ndarray],
    *,
    expected_seeds: Sequence[int],
) -> pd.DataFrame:
    """Score exact contexts; absent context/seed contributions remain zero."""
    n_seeds = len(expected_seeds)
    if n_seeds != 5:
        raise ValueError("This reviewer case requires exactly five grouping seeds")
    distinct = edges.loc[
        edges["_source_h5ad_index"].ne(edges["_target_h5ad_index"])
    ].copy()
    source = distinct["_source_h5ad_index"].to_numpy(int)
    target = distinct["_target_h5ad_index"].to_numpy(int)
    parts: list[pd.DataFrame] = []
    for ligand, receptor in CASE_AXES:
        lr = activities[ligand][source] * activities[receptor][target]
        local = distinct[
            [
                "grouping_seed",
                "_source_h5ad_index",
                "_target_h5ad_index",
                "sender_type",
                "receiver_type",
                "attention_abs_mean",
                "edge_message_norm_joint",
            ]
        ].copy()
        local["axis_id"] = f"{ligand}->{receptor}"
        local["ligand"] = ligand
        local["receptor"] = receptor
        local["lr_activity"] = lr
        local["attention_lr"] = (
            local["attention_abs_mean"].to_numpy(float) * lr
        )
        local["exact_message_lr"] = (
            local["edge_message_norm_joint"].to_numpy(float) * lr
        )
        local["lr_active"] = lr > 0
        keys = ["axis_id", "ligand", "receptor", "sender_type", "receiver_type"]
        grouped = (
            local.groupby(keys, observed=True, as_index=False)
            .agg(
                raw_attention_sum_occurrences=("attention_abs_mean", "sum"),
                lr_activity_sum_occurrences=("lr_activity", "sum"),
                attention_lr_sum_occurrences=("attention_lr", "sum"),
                exact_message_lr_sum_occurrences=("exact_message_lr", "sum"),
                n_edge_occurrences=("_source_h5ad_index", "size"),
                n_lr_active_occurrences=("lr_active", "sum"),
                n_seeds_with_any_edge=("grouping_seed", "nunique"),
            )
        )
        active = local.loc[local["lr_active"]]
        active_unique = (
            active.drop_duplicates(
                [
                    "axis_id",
                    "sender_type",
                    "receiver_type",
                    "_source_h5ad_index",
                    "_target_h5ad_index",
                ]
            )
            .groupby(keys, observed=True)
            .size()
            .rename("n_active_unique_edges")
            .reset_index()
        )
        active_seeds = (
            active.groupby(keys, observed=True)["grouping_seed"]
            .nunique()
            .rename("n_seeds_with_lr_active_edge")
            .reset_index()
        )
        grouped = grouped.merge(active_unique, on=keys, how="left", validate="one_to_one")
        grouped = grouped.merge(active_seeds, on=keys, how="left", validate="one_to_one")
        parts.append(grouped)
    aggregated = pd.concat(parts, ignore_index=True)
    keys = ["axis_id", "ligand", "receptor", "sender_type", "receiver_type"]
    result = universe.merge(aggregated, on=keys, how="left", validate="one_to_one")
    zero_columns = [
        "raw_attention_sum_occurrences",
        "lr_activity_sum_occurrences",
        "attention_lr_sum_occurrences",
        "exact_message_lr_sum_occurrences",
        "n_edge_occurrences",
        "n_lr_active_occurrences",
        "n_seeds_with_any_edge",
        "n_active_unique_edges",
        "n_seeds_with_lr_active_edge",
    ]
    for column in zero_columns:
        result[column] = pd.to_numeric(result[column], errors="coerce").fillna(0)
    integer_columns = [column for column in zero_columns if column.startswith("n_")]
    result[integer_columns] = result[integer_columns].astype(int)
    result["n_grouping_seeds"] = n_seeds
    result["n_context_seed_zeros"] = n_seeds - result["n_seeds_with_any_edge"]
    result["n_lr_active_context_seed_zeros"] = (
        n_seeds - result["n_seeds_with_lr_active_edge"]
    )
    denominator = (
        n_seeds * result["n_possible_distinct_cell_pairs"].to_numpy(float)
    )
    valid = denominator > 0
    score_map = {
        "raw_attention_sum_occurrences": "cytobridge_raw_attention_magnitude_density",
        "lr_activity_sum_occurrences": "cytobridge_lr_only_density",
        "attention_lr_sum_occurrences": "cytobridge_attention_lr_density",
        "exact_message_lr_sum_occurrences": "cytobridge_exact_message_lr_density",
    }
    for numerator, output in score_map.items():
        result[output] = np.nan
        result.loc[valid, output] = (
            result.loc[valid, numerator].to_numpy(float) / denominator[valid]
        )
    return result


def score_raw_type_pair_universe(
    edges: pd.DataFrame,
    universe: pd.DataFrame,
    *,
    expected_seeds: Sequence[int],
) -> pd.DataFrame:
    """Rank raw attention on the full type-pair square, independent of JAM support.

    Every observed directed annotation pair is retained, including a zero row
    when it has no model edge in one or all of the five grouping seeds. The
    primary raw score is the zero-completed attention sum divided by five and
    by the number of possible distinct ordered cell pairs.
    """
    if len(expected_seeds) != 5:
        raise ValueError("Raw type-pair audit requires exactly five grouping seeds")
    base_columns = [
        "stage",
        "stage_label",
        "sender_type",
        "receiver_type",
        "n_sender_cells",
        "n_receiver_cells",
        "n_shared_sender_receiver_cells",
        "n_possible_distinct_cell_pairs",
    ]
    base = universe[base_columns].drop_duplicates(
        ["stage", "sender_type", "receiver_type"]
    )
    distinct = edges.loc[
        edges["_source_h5ad_index"].ne(edges["_target_h5ad_index"])
    ].copy()
    grouped = (
        distinct.groupby(["sender_type", "receiver_type"], observed=True, as_index=False)
        .agg(
            raw_attention_sum_occurrences=("attention_abs_mean", "sum"),
            n_raw_edge_occurrences=("_source_h5ad_index", "size"),
            n_seeds_with_any_raw_edge=("grouping_seed", "nunique"),
        )
    )
    unique_edges = (
        distinct.drop_duplicates(
            [
                "sender_type",
                "receiver_type",
                "_source_h5ad_index",
                "_target_h5ad_index",
            ]
        )
        .groupby(["sender_type", "receiver_type"], observed=True)
        .size()
        .rename("n_unique_distinct_raw_edges")
        .reset_index()
    )
    result = base.merge(
        grouped,
        on=["sender_type", "receiver_type"],
        how="left",
        validate="one_to_one",
    ).merge(
        unique_edges,
        on=["sender_type", "receiver_type"],
        how="left",
        validate="one_to_one",
    )
    for column in (
        "raw_attention_sum_occurrences",
        "n_raw_edge_occurrences",
        "n_seeds_with_any_raw_edge",
        "n_unique_distinct_raw_edges",
    ):
        result[column] = pd.to_numeric(result[column], errors="coerce").fillna(0)
    for column in (
        "n_raw_edge_occurrences",
        "n_seeds_with_any_raw_edge",
        "n_unique_distinct_raw_edges",
    ):
        result[column] = result[column].astype(int)
    result["n_grouping_seeds"] = len(expected_seeds)
    result["n_raw_context_seed_zeros"] = (
        len(expected_seeds) - result["n_seeds_with_any_raw_edge"]
    )
    result["raw_attention_mean_on_present_edge_occurrences"] = np.where(
        result["n_raw_edge_occurrences"].gt(0),
        result["raw_attention_sum_occurrences"] / result["n_raw_edge_occurrences"],
        0.0,
    )
    denominator = (
        len(expected_seeds)
        * result["n_possible_distinct_cell_pairs"].to_numpy(float)
    )
    result["raw_attention_full_type_pair_density"] = np.nan
    valid = denominator > 0
    result.loc[valid, "raw_attention_full_type_pair_density"] = (
        result.loc[valid, "raw_attention_sum_occurrences"].to_numpy(float)
        / denominator[valid]
    )
    result["passes_raw_full_type_pair_rank_universe"] = result[
        "n_possible_distinct_cell_pairs"
    ].gt(0)
    result["axis_id"] = "__raw_attention_full_type_pair_universe__"
    result = attach_ranks(
        result,
        {"raw_attention_full_type_pair_density": "raw_attention_full_type_pair"},
        support_column="passes_raw_full_type_pair_rank_universe",
    )
    return result


def parse_boolean(series: pd.Series, label: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    mapped = series.astype(str).str.strip().str.casefold().map(
        {"true": True, "false": False, "1": True, "0": False}
    )
    if mapped.isna().any():
        raise ValueError(f"{label} must contain only true/false values")
    return mapped.astype(bool)


def normalized_stage(
    frame: pd.DataFrame,
    label: str,
    *,
    internal_stage: float,
    biological_time_hpf: float,
) -> tuple[pd.Series, str]:
    """Return a stage-selection mask and its audited coordinate basis.

    Internal ``stage`` wins whenever it is fully numeric.  ``stage_time`` is a
    biological-time fallback only; a table containing stage=3 and
    stage_time=18 must therefore join to the requested internal stage 3.
    """
    if "stage" in frame:
        numeric_stage = pd.to_numeric(frame["stage"], errors="coerce")
        if numeric_stage.notna().all():
            return (
                pd.Series(
                    np.isclose(
                        numeric_stage.to_numpy(float),
                        float(internal_stage),
                        rtol=0.0,
                        atol=1e-12,
                    ),
                    index=frame.index,
                ),
                "internal_stage",
            )
        if numeric_stage.notna().any():
            raise ValueError(f"{label} stage mixes numeric and nonnumeric values")
    if "stage_time" not in frame:
        raise ValueError(
            f"{label} has no numeric internal stage and no stage_time fallback"
        )
    stage_time = pd.to_numeric(frame["stage_time"], errors="coerce")
    if stage_time.isna().any():
        raise ValueError(f"{label} stage_time must be fully numeric for fallback")
    return (
        pd.Series(
            np.isclose(
                stage_time.to_numpy(float),
                float(biological_time_hpf),
                rtol=0.0,
                atol=1e-12,
            ),
            index=frame.index,
        ),
        "biological_time_hpf_fallback",
    )


def collapse_external_table(
    spec: ExternalSpec,
    *,
    stage: float,
    biological_time_hpf: float,
    axes: pd.DataFrame,
) -> pd.DataFrame:
    frame = pd.read_csv(spec.table)
    require(
        frame,
        ["ligand", "receptor", "sender_type", "receiver_type", spec.score_column],
        f"{spec.method} external score table",
    )
    frame = frame.copy()
    stage_mask, stage_basis = normalized_stage(
        frame,
        f"{spec.method} score table",
        internal_stage=stage,
        biological_time_hpf=biological_time_hpf,
    )
    frame["ligand"] = frame["ligand"].astype(str).str.casefold()
    frame["receptor"] = frame["receptor"].astype(str).str.casefold()
    frame["axis_id"] = frame["ligand"] + "->" + frame["receptor"]
    wanted = set(axes["axis_id"])
    frame = frame.loc[
        stage_mask & frame["axis_id"].isin(wanted)
    ].copy()
    frame["external_stage_coordinate_basis"] = stage_basis
    if "method" in frame and not frame.empty:
        observed_methods = set(frame["method"].astype(str).str.casefold())
        if observed_methods != {spec.method.casefold()}:
            raise ValueError(
                f"{spec.method} table method column disagrees: {sorted(observed_methods)}"
            )
    frame["_external_raw_score"] = pd.to_numeric(
        frame[spec.score_column], errors="raise"
    ).astype(float)
    if not np.isfinite(frame["_external_raw_score"]).all() or (
        frame["_external_raw_score"] < 0
    ).any():
        raise ValueError(f"{spec.method} external score must be finite and nonnegative")
    key = ["axis_id", "ligand", "receptor", "sender_type", "receiver_type"]
    rows: list[dict[str, object]] = []
    for values, group in frame.groupby(key, observed=True, sort=False):
        score_values = group["_external_raw_score"].to_numpy(float)
        if not np.isclose(score_values, score_values[0], rtol=1e-10, atol=1e-12).all():
            raise ValueError(
                f"{spec.method} duplicate provenance rows disagree for context {values}"
            )
        row = dict(zip(key, values))
        row["external_input_score"] = float(score_values[0])
        row["external_duplicate_provenance_rows_collapsed"] = int(len(group))
        if "n_possible_distinct_cell_pairs" in group:
            denominators = pd.to_numeric(
                group["n_possible_distinct_cell_pairs"], errors="raise"
            ).to_numpy(float)
            if not np.isfinite(denominators).all() or not np.equal(
                denominators, denominators[0]
            ).all():
                raise ValueError(
                    f"{spec.method} duplicate rows disagree on distinct-cell denominator"
                )
            row["external_input_distinct_denominator"] = float(denominators[0])
        else:
            row["external_input_distinct_denominator"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows, columns=key + [
        "external_input_score",
        "external_duplicate_provenance_rows_collapsed",
        "external_input_distinct_denominator",
    ])


def load_external_availability(
    spec: ExternalSpec,
    *,
    stage: float,
    biological_time_hpf: float,
    axes: pd.DataFrame,
) -> pd.DataFrame:
    frame = pd.read_csv(spec.availability)
    require(frame, ["ligand", "receptor"], f"{spec.method} availability table")
    available_column = (
        "method_available" if "method_available" in frame else "available"
    )
    require(frame, [available_column], f"{spec.method} availability table")
    frame = frame.copy()
    stage_mask, stage_basis = normalized_stage(
        frame,
        f"{spec.method} availability",
        internal_stage=stage,
        biological_time_hpf=biological_time_hpf,
    )
    frame["ligand"] = frame["ligand"].astype(str).str.casefold()
    frame["receptor"] = frame["receptor"].astype(str).str.casefold()
    frame["axis_id"] = frame["ligand"] + "->" + frame["receptor"]
    frame["external_axis_available"] = parse_boolean(
        frame[available_column], f"{spec.method} {available_column}"
    )
    wanted = set(axes["axis_id"])
    frame = frame.loc[
        stage_mask & frame["axis_id"].isin(wanted)
    ].copy()
    frame["external_stage_coordinate_basis"] = stage_basis
    key = ["axis_id", "ligand", "receptor"]
    rows: list[dict[str, object]] = []
    for values, group in frame.groupby(key, observed=True, sort=False):
        if group["external_axis_available"].nunique() != 1:
            raise ValueError(f"{spec.method} availability disagrees across provenance rows")
        row = dict(zip(key, values))
        row["external_axis_available"] = bool(group["external_axis_available"].iloc[0])
        row["external_availability_provenance_rows"] = int(len(group))
        row["external_stage_coordinate_basis"] = stage_basis
        if "matrix_key" in group:
            row["external_matrix_keys"] = ";".join(
                sorted(set(group["matrix_key"].astype(str)))
            )
        else:
            row["external_matrix_keys"] = ""
        rows.append(row)
    result = pd.DataFrame(rows)
    observed = set(result["axis_id"]) if not result.empty else set()
    missing = sorted(wanted.difference(observed))
    if missing:
        raise ValueError(
            f"{spec.method} availability table lacks explicit stage rows for axes: {missing}"
        )
    return result


def join_external_sparse(
    contexts: pd.DataFrame,
    scores: pd.DataFrame,
    availability: pd.DataFrame,
    *,
    spec: ExternalSpec,
) -> pd.DataFrame:
    """Join sparse external rows: valid missing context=0; unavailable axis=NA."""
    key = ["axis_id", "ligand", "receptor", "sender_type", "receiver_type"]
    axis_key = ["axis_id", "ligand", "receptor"]
    result = contexts.merge(scores, on=key, how="left", validate="one_to_one")
    result = result.merge(availability, on=axis_key, how="left", validate="many_to_one")
    if result["external_axis_available"].isna().any():
        raise ValueError(f"{spec.method} has context rows without explicit axis availability")
    unavailable_with_rows = (
        ~result["external_axis_available"].astype(bool)
        & result["external_input_score"].notna()
    )
    if unavailable_with_rows.any():
        raise ValueError(f"{spec.method} has scores for an explicitly unavailable axis")
    supplied_denominator = result["external_input_distinct_denominator"].notna()
    mismatch = supplied_denominator & ~np.isclose(
        result["external_input_distinct_denominator"],
        result["n_possible_distinct_cell_pairs"],
        rtol=0.0,
        atol=0.0,
    )
    if mismatch.any():
        raise ValueError(
            f"{spec.method} distinct-cell denominator disagrees with n_s*n_r-|intersection|"
        )
    if spec.score_mode in {"distinct_density", "distinct_mass"}:
        present = result["external_input_score"].notna()
        if (present & ~supplied_denominator).any():
            raise ValueError(
                f"{spec.method} {spec.score_mode} rows require n_possible_distinct_cell_pairs"
            )
    structural_zero = (
        result["external_axis_available"].astype(bool)
        & result["external_input_score"].isna()
    )
    result["external_sparse_context_completed_as_zero"] = structural_zero
    result.loc[structural_zero, "external_input_score"] = 0.0
    result.loc[
        structural_zero, "external_input_distinct_denominator"
    ] = result.loc[structural_zero, "n_possible_distinct_cell_pairs"].to_numpy(float)
    result["external_score"] = np.nan
    available = result["external_axis_available"].astype(bool)
    if spec.score_mode == "distinct_mass":
        valid = available & result["n_possible_distinct_cell_pairs"].gt(0)
        result.loc[valid, "external_score"] = (
            result.loc[valid, "external_input_score"].to_numpy(float)
            / result.loc[valid, "n_possible_distinct_cell_pairs"].to_numpy(float)
        )
    else:
        result.loc[available, "external_score"] = result.loc[
            available, "external_input_score"
        ].to_numpy(float)
    prefix = spec.slug
    rename = {
        "external_axis_available": f"{prefix}_axis_available",
        "external_stage_coordinate_basis": f"{prefix}_stage_coordinate_basis",
        "external_availability_provenance_rows": f"{prefix}_availability_provenance_rows",
        "external_matrix_keys": f"{prefix}_matrix_keys",
        "external_input_score": f"{prefix}_input_score",
        "external_input_distinct_denominator": f"{prefix}_input_distinct_denominator",
        "external_duplicate_provenance_rows_collapsed": f"{prefix}_duplicate_provenance_rows_collapsed",
        "external_sparse_context_completed_as_zero": f"{prefix}_sparse_context_completed_as_zero",
        "external_score": f"{prefix}_score",
    }
    result = result.rename(columns=rename)
    result[f"{prefix}_score_mode"] = spec.score_mode
    result[f"{prefix}_source_score_column"] = spec.score_column
    result[f"{prefix}_raw_units_directly_comparable_to_cytobridge"] = False
    return result


def descending_competition_rank(values: pd.Series) -> pd.DataFrame:
    numeric = pd.to_numeric(values, errors="raise").astype(float)
    if numeric.empty or not np.isfinite(numeric).all():
        raise ValueError("Ranking universe must be nonempty and finite")
    n = int(len(numeric))
    rank = numeric.rank(method="min", ascending=False).astype(int)
    tie = numeric.groupby(numeric, dropna=False).transform("size").astype(int)
    return pd.DataFrame(
        {
            "rank_from_top": rank,
            "n_ranked_contexts": n,
            "tie_count": tie,
            "top_tail_fraction": (rank - 1) / n,
            "top_tail_percent": 100.0 * (rank - 1) / n,
            "rank_over_n": rank.astype(str) + "/" + str(n),
        },
        index=values.index,
    )


def attach_ranks(
    frame: pd.DataFrame,
    score_columns: Mapping[str, str],
    *,
    support_column: str = "passes_context_support_filter",
) -> pd.DataFrame:
    """Apply support first, then descending competition/min rank including zeros."""
    result = frame.copy()
    for score, prefix in score_columns.items():
        for suffix in (
            "rank_from_top",
            "n_ranked_contexts",
            "tie_count",
            "top_tail_fraction",
            "top_tail_percent",
            "rank_over_n",
        ):
            result[f"{prefix}_{suffix}"] = np.nan if suffix != "rank_over_n" else pd.NA
        available = result[support_column].astype(bool) & result[score].notna()
        for _, indices in result.loc[available].groupby(
            ["stage", "axis_id"], observed=True
        ).groups.items():
            ranked = descending_competition_rank(result.loc[indices, score])
            for suffix in ranked:
                result.loc[indices, f"{prefix}_{suffix}"] = ranked[suffix]
    return result


def expression_detection_table(
    data: ad.AnnData,
    values: Mapping[str, np.ndarray],
    *,
    stage: float,
    stage_label: str,
    cell_type: str,
    genes: Sequence[str],
    time_key: str,
    annotation_key: str,
) -> pd.DataFrame:
    stage_values = pd.to_numeric(data.obs[time_key], errors="raise").to_numpy(float)
    annotation = data.obs[annotation_key].astype(str).to_numpy()
    mask = np.isclose(stage_values, stage, rtol=0.0, atol=1e-12) & (
        annotation == str(cell_type)
    )
    n = int(mask.sum())
    if n == 0:
        raise ValueError(f"No cells for {stage_label} {cell_type}")
    rows = []
    for gene in genes:
        detected = np.asarray(values[gene]) > 0
        positive = int(detected[mask].sum())
        rows.append(
            {
                "stage": float(stage),
                "stage_label": stage_label,
                "cell_type": cell_type,
                "gene": gene,
                "n_cells": n,
                "n_detected": positive,
                "detected_fraction": positive / n,
                "detected_percent": 100.0 * positive / n,
                "detection_definition": "H5AD X value > 0",
            }
        )
    return pd.DataFrame(rows)


def somite_gene_associations(
    data: ad.AnnData,
    values: Mapping[str, np.ndarray],
    *,
    stage: float,
    stage_label: str,
    cell_type: str,
    time_key: str,
    annotation_key: str,
) -> pd.DataFrame:
    stage_values = pd.to_numeric(data.obs[time_key], errors="raise").to_numpy(float)
    labels = data.obs[annotation_key].astype(str).to_numpy()
    mask = np.isclose(stage_values, stage, rtol=0.0, atol=1e-12) & (labels == cell_type)
    rows: list[dict[str, object]] = []
    for jam_gene in ("jam3b", "jam2a"):
        jam = np.asarray(values[jam_gene], dtype=float)[mask]
        myog = np.asarray(values["myog"], dtype=float)[mask]
        jam_detected = jam > 0
        myog_detected = myog > 0
        table = np.asarray(
            [
                [np.sum(jam_detected & myog_detected), np.sum(jam_detected & ~myog_detected)],
                [np.sum(~jam_detected & myog_detected), np.sum(~jam_detected & ~myog_detected)],
            ],
            dtype=int,
        )
        odds_ratio, fisher_p = stats.fisher_exact(table, alternative="two-sided")
        spearman = stats.spearmanr(jam, myog, nan_policy="raise")
        rows.append(
            {
                "stage": float(stage),
                "stage_label": stage_label,
                "cell_type": cell_type,
                "gene_a": jam_gene,
                "gene_b": "myog",
                "n_cells": int(mask.sum()),
                "both_detected": int(table[0, 0]),
                "gene_a_only": int(table[0, 1]),
                "gene_b_only": int(table[1, 0]),
                "neither_detected": int(table[1, 1]),
                "fisher_odds_ratio": float(odds_ratio),
                "fisher_two_sided_p": float(fisher_p),
                "spearman_rho_expression": float(spearman.statistic),
                "spearman_two_sided_p": float(spearman.pvalue),
                "literature_direction_context": (
                    "cross-sectional consistency with reported myog-mutant reduction of jam3b"
                    if jam_gene == "jam3b"
                    else "weaker contrast is directionally consistent with reported lack of jam2a reduction in myog mutant"
                ),
                "claim_guardrail": (
                    "Cross-sectional co-detection/co-expression consistency only; not evidence that myog regulates JAM signaling in this dataset."
                ),
            }
        )
    return pd.DataFrame(rows)


def compatible_neighbor_count(
    pairs: np.ndarray, ligand_detected: np.ndarray, receptor_detected: np.ndarray
) -> int:
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("pairs must have shape (n_pairs, 2)")
    first = pairs[:, 0]
    second = pairs[:, 1]
    compatible = (
        ligand_detected[first] & receptor_detected[second]
    ) | (receptor_detected[first] & ligand_detected[second])
    return int(compatible.sum())


def somite_spatial_permutation(
    data: ad.AnnData,
    values: Mapping[str, np.ndarray],
    *,
    stage: float,
    stage_label: str,
    cell_type: str,
    time_key: str,
    annotation_key: str,
    spatial_key: str,
    cutoff: float,
    n_permutations: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if cutoff <= 0 or not np.isfinite(cutoff):
        raise ValueError("Spatial cutoff must be finite and positive")
    if n_permutations < 1:
        raise ValueError("n_permutations must be positive")
    if spatial_key not in data.obsm:
        raise KeyError(f"H5AD is missing obsm[{spatial_key!r}]")
    stage_values = pd.to_numeric(data.obs[time_key], errors="raise").to_numpy(float)
    labels = data.obs[annotation_key].astype(str).to_numpy()
    mask = np.isclose(stage_values, stage, rtol=0.0, atol=1e-12) & (labels == cell_type)
    coordinates = np.asarray(data.obsm[spatial_key], dtype=float)[mask]
    if coordinates.ndim != 2 or coordinates.shape[1] < 2 or not np.isfinite(coordinates).all():
        raise ValueError("Spatial coordinates must be a finite N x D array")
    pairs_set = cKDTree(coordinates).query_pairs(r=float(cutoff), output_type="ndarray")
    pairs = np.asarray(pairs_set, dtype=int).reshape(-1, 2)
    if pairs.size == 0:
        raise ValueError("No Somite neighbor pairs found at the frozen cutoff")
    jam2a = np.asarray(values["jam2a"])[mask] > 0
    jam3b = np.asarray(values["jam3b"])[mask] > 0
    observed = compatible_neighbor_count(pairs, jam2a, jam3b)
    rng = np.random.default_rng(int(seed))
    null = np.empty(int(n_permutations), dtype=int)
    for iteration in range(int(n_permutations)):
        permuted_ligand = jam2a[rng.permutation(len(jam2a))]
        permuted_receptor = jam3b[rng.permutation(len(jam3b))]
        null[iteration] = compatible_neighbor_count(
            pairs, permuted_ligand, permuted_receptor
        )
    exceedances = int(np.sum(null >= observed))
    mc_p = (exceedances + 1) / (int(n_permutations) + 1)
    summary = pd.DataFrame(
        [
            {
                "stage": float(stage),
                "stage_label": stage_label,
                "cell_type": cell_type,
                "n_cells": int(mask.sum()),
                "spatial_cutoff": float(cutoff),
                "n_distinct_undirected_neighbor_pairs": int(len(pairs)),
                "observed_jam2a_jam3b_orientation_compatible_pairs": observed,
                "n_permutations": int(n_permutations),
                "permutation_seed": int(seed),
                "null_mean": float(np.mean(null)),
                "null_std_ddof1": float(np.std(null, ddof=1)),
                "null_q025": float(np.quantile(null, 0.025)),
                "null_q50": float(np.quantile(null, 0.5)),
                "null_q975": float(np.quantile(null, 0.975)),
                "observed_over_null_mean": (
                    float(observed / np.mean(null)) if np.mean(null) > 0 else np.nan
                ),
                "n_null_at_least_observed": exceedances,
                "monte_carlo_upper_tail_p_plus1": float(mc_p),
                "neighbor_definition": "distinct undirected Somite cell pairs with Euclidean distance <= cutoff",
                "compatible_definition": "jam2a detected in one endpoint and jam3b detected in the other, either orientation; each undirected pair counted once",
                "null_definition": "independently permute jam2a and jam3b detection labels within 18 hpf Somite cells, preserving each marginal prevalence and the spatial graph",
                "claim_guardrail": "Spatial enrichment may reflect a shared regional cell state; it is not proof of direct physical contact or causal signaling.",
            }
        ]
    )
    distribution = pd.DataFrame(
        {
            "iteration": np.arange(1, int(n_permutations) + 1, dtype=int),
            "permutation_seed": int(seed),
            "orientation_compatible_pair_count": null,
            "at_least_observed": null >= observed,
        }
    )
    return summary, distribution


def write_readme(path: Path, external_specs: Sequence[ExternalSpec]) -> None:
    methods = ", ".join(spec.method for spec in external_specs)
    external_text = (
        f"External methods: {methods}. A missing sparse context row becomes zero only "
        "when its explicit stage x LR availability record says the axis was computed. "
        "An unavailable axis remains NA. Native scores from different methods are not "
        "asserted to share units.\n\n"
        if methods
        else (
            "External methods were intentionally omitted for this CytoBridge-only "
            "biology audit; no external score or availability value was fabricated.\n\n"
        )
    )
    text = (
        "# 18 hpf Somite Jam2a--Jam3b audit\n\n"
        "This bundle evaluates both JAM orientations in the exact directed cell-type context "
        "universe. Support filtering is applied before competition/min ranking; every rank is "
        "reported as rank/N with tie count and top-tail percentage.\n\n"
        "CytoBridge columns are deliberately separate: raw attention magnitude, LR-only on the "
        "same edge scaffold, |attention| x LR, and exact message-norm x LR. Five grouping seeds "
        "are technical regroupings; a missing context edge in a valid seed contributes zero. "
        "Cell self-edges are excluded and the denominator is n_sender*n_receiver minus the "
        "sender/receiver cell-set intersection.\n\n"
        + external_text
        + "The expression and spatial artifacts are cross-sectional consistency checks. The "
        "literature validates the Jam2a/Jam3b heterophilic pair and somite-stage myocyte-fusion "
        "context, but not a polarized sender-to-receiver direction, CytoBridge score magnitude, "
        "direct molecular contact in these cells, or causality. The 24 hpf Fast Muscle table is "
        "a cross-sectional maturation-state comparison, not lineage tracing. If a trained/"
        "pre-interaction/random control table is supplied, it is copied as a hash-verified "
        "precomputed artifact; "
        "the present script does not reconstruct those model controls.\n"
    )
    path.write_text(text, encoding="utf-8")


def load_control_artifact(path: Path | None) -> pd.DataFrame:
    """Expose trained/pre-interaction/random controls without inventing results."""
    expected = ("trained", "pre_interaction", "randomized_interaction_seed17")
    if path is None:
        return pd.DataFrame(
            {
                "control": expected,
                "control_metrics_available": False,
                "unavailable_reason": (
                    "--trained-pre-interaction-random-control was not supplied"
                ),
            }
        )
    frame = pd.read_csv(path)
    identifier = "control" if "control" in frame else "condition" if "condition" in frame else None
    if identifier is None:
        raise ValueError("Control metrics table requires a control or condition column")
    normalized = frame[identifier].astype(str).str.casefold().str.replace(
        r"[^a-z0-9]+", "_", regex=True
    ).str.strip("_")
    aliases = {
        "trained": "trained",
        "init": "pre_interaction",
        "init_interaction": "pre_interaction",
        "initial_interaction": "pre_interaction",
        "pre_interaction": "pre_interaction",
        "preinteraction": "pre_interaction",
        "random": "randomized_interaction_seed17",
        "randomized_interaction": "randomized_interaction_seed17",
        "randomized_interaction_seed17": "randomized_interaction_seed17",
        "random_interaction_seed17": "randomized_interaction_seed17",
    }
    canonical = normalized.map(aliases)
    observed = set(canonical.dropna())
    missing = sorted(set(expected).difference(observed))
    if missing:
        raise ValueError(f"Control metrics table lacks required conditions: {missing}")
    result = frame.loc[canonical.notna()].copy()
    if identifier == "control":
        result["control"] = canonical.loc[canonical.notna()].to_numpy()
        ordered = ["control", *[column for column in result if column != "control"]]
        result = result[ordered]
    else:
        result.insert(0, "control", canonical.loc[canonical.notna()].to_numpy())
    result["control_metrics_available"] = True
    result["unavailable_reason"] = ""
    return result


def main() -> None:
    args = parser().parse_args()
    if args.min_active_edges < 0 or args.min_cells_per_side < 1:
        raise ValueError("Support thresholds must be nonnegative/positive")
    expected_seeds = parse_seeds(args.expected_grouping_seeds)
    external_specs = parse_external_specs(args.external_spec)
    for path in [args.h5ad, args.observed_cells, args.provenance]:
        if not path.is_file():
            raise FileNotFoundError(path)
    for spec in external_specs:
        for path in (spec.table, spec.availability):
            if not path.is_file():
                raise FileNotFoundError(path)
    if args.preprocess_manifest is not None and not args.preprocess_manifest.is_file():
        raise FileNotFoundError(args.preprocess_manifest)
    if (
        args.trained_pre_interaction_random_control is not None
        and not args.trained_pre_interaction_random_control.is_file()
    ):
        raise FileNotFoundError(args.trained_pre_interaction_random_control)
    prepare_output(args.output_dir, bool(args.overwrite))
    tables_dir = args.output_dir / "tables"
    tables_dir.mkdir()

    data = ad.read_h5ad(args.h5ad)
    provenance = load_provenance(args.provenance)
    raw_edges, edge_inventory, mapping_resolution = load_edge_tables(
        args.edge_dir,
        data,
        args.observed_cells,
        expected_seeds=expected_seeds,
        stage=args.stage,
        stage_label=args.stage_label,
        time_key=args.time_key,
        time_label_key=args.time_label_key,
        annotation_key=args.annotation_key,
    )
    required_genes = set(BIOLOGY_18_GENES).union(BIOLOGY_24_GENES)
    gene_values, gene_audit = load_gene_values(data, required_genes)
    activities = q95_activities(gene_values, gene_audit)
    universe = context_universe(
        data,
        stage=args.stage,
        stage_label=args.stage_label,
        time_key=args.time_key,
        annotation_key=args.annotation_key,
        axes=provenance,
    )
    contexts = score_cytobridge_contexts(
        raw_edges,
        universe,
        activities,
        expected_seeds=expected_seeds,
    )
    contexts["passes_context_support_filter"] = (
        contexts["n_possible_distinct_cell_pairs"].gt(0)
        & contexts["n_active_unique_edges"].ge(int(args.min_active_edges))
        & contexts["n_sender_cells"].ge(int(args.min_cells_per_side))
        & contexts["n_receiver_cells"].ge(int(args.min_cells_per_side))
    )
    rank_map = {
        score: score.removeprefix("cytobridge_").removesuffix("_density")
        for score in SCORE_COLUMNS
    }
    contexts = attach_ranks(contexts, rank_map)
    raw_type_pair_ranks = score_raw_type_pair_universe(
        raw_edges,
        universe,
        expected_seeds=expected_seeds,
    )

    external_audits: list[pd.DataFrame] = []
    external_inventory: list[dict[str, object]] = []
    for spec in external_specs:
        scores = collapse_external_table(
            spec,
            stage=args.stage,
            biological_time_hpf=args.biological_time_hpf,
            axes=provenance,
        )
        availability = load_external_availability(
            spec,
            stage=args.stage,
            biological_time_hpf=args.biological_time_hpf,
            axes=provenance,
        )
        external_audit = availability.copy()
        external_audit.insert(0, "method", spec.method)
        external_audit["score_column"] = spec.score_column
        external_audit["score_mode"] = spec.score_mode
        external_audits.append(external_audit)
        contexts = join_external_sparse(
            contexts, scores, availability, spec=spec
        )
        contexts = attach_ranks(
            contexts,
            {f"{spec.slug}_score": spec.slug},
        )
        external_inventory.append(
            {
                "method": spec.method,
                "slug": spec.slug,
                "score_table": artifact(spec.table),
                "availability_table": artifact(spec.availability),
                "score_column": spec.score_column,
                "score_mode": spec.score_mode,
                "raw_units_directly_comparable_to_cytobridge": False,
            }
        )

    selected = contexts.loc[
        contexts["sender_type"].eq(args.sender_type)
        & contexts["receiver_type"].eq(args.receiver_type)
    ].copy()
    if set(selected["axis_id"]) != set(provenance["axis_id"]):
        raise ValueError("Selected Somite context is missing one or both JAM orientations")
    selected = selected.merge(
        provenance,
        on=["axis_id", "ligand", "receptor"],
        how="left",
        validate="one_to_one",
    )
    selected_raw = raw_type_pair_ranks.loc[
        raw_type_pair_ranks["sender_type"].eq(args.sender_type)
        & raw_type_pair_ranks["receiver_type"].eq(args.receiver_type),
        [
            "stage",
            "sender_type",
            "receiver_type",
            "raw_attention_full_type_pair_density",
            "raw_attention_full_type_pair_rank_from_top",
            "raw_attention_full_type_pair_n_ranked_contexts",
            "raw_attention_full_type_pair_tie_count",
            "raw_attention_full_type_pair_top_tail_fraction",
            "raw_attention_full_type_pair_top_tail_percent",
            "raw_attention_full_type_pair_rank_over_n",
        ],
    ]
    if len(selected_raw) != 1:
        raise ValueError("Raw full type-pair universe lacks a unique selected Somite context")
    selected = selected.merge(
        selected_raw,
        on=["stage", "sender_type", "receiver_type"],
        how="left",
        validate="many_to_one",
    )
    selected["biological_interpretation_scope"] = (
        "observational Somite-context consistency with a literature-supported heterophilic JAM pair"
    )
    selected["causal_or_lr_specific_claim_allowed"] = False

    detection_18 = expression_detection_table(
        data,
        gene_values,
        stage=args.stage,
        stage_label=args.stage_label,
        cell_type=args.sender_type,
        genes=BIOLOGY_18_GENES,
        time_key=args.time_key,
        annotation_key=args.annotation_key,
    )
    associations_18 = somite_gene_associations(
        data,
        gene_values,
        stage=args.stage,
        stage_label=args.stage_label,
        cell_type=args.sender_type,
        time_key=args.time_key,
        annotation_key=args.annotation_key,
    )
    spatial_summary, spatial_null = somite_spatial_permutation(
        data,
        gene_values,
        stage=args.stage,
        stage_label=args.stage_label,
        cell_type=args.sender_type,
        time_key=args.time_key,
        annotation_key=args.annotation_key,
        spatial_key=args.spatial_key,
        cutoff=float(args.spatial_cutoff),
        n_permutations=int(args.n_permutations),
        seed=int(args.permutation_seed),
    )
    detection_24 = expression_detection_table(
        data,
        gene_values,
        stage=args.maturity_stage,
        stage_label=args.maturity_stage_label,
        cell_type=args.maturity_cell_type,
        genes=BIOLOGY_24_GENES,
        time_key=args.time_key,
        annotation_key=args.annotation_key,
    )
    detection_24["claim_guardrail"] = (
        "Cross-sectional maturation-state comparison only; not lineage tracing."
    )
    expression_detection = pd.concat([detection_18, detection_24], ignore_index=True)
    controls = load_control_artifact(args.trained_pre_interaction_random_control)
    spatial_panel_cells = build_spatial_panel_cells(
        data,
        gene_values,
        stage=args.stage,
        cell_type=args.sender_type,
        time_key=args.time_key,
        annotation_key=args.annotation_key,
        spatial_key=args.spatial_key,
    )
    trained_display_edges = build_trained_jam_display_edges(
        raw_edges,
        spatial_panel_cells,
        cell_type=args.sender_type,
        minimum_seed_support=3,
        maximum_display_edges=15,
    )

    paths = {
        "context_scores_and_ranks": tables_dir / "jam_context_scores_and_ranks.csv.gz",
        "somite_case_summary": tables_dir / "somite_jam_case_summary.csv",
        "edge_inventory": tables_dir / "cytobridge_edge_input_inventory.csv",
        "external_availability": tables_dir / "external_axis_availability_audit.csv",
        "gene_activity": tables_dir / "gene_activity_audit.csv",
        "literature_provenance": tables_dir / "jam_literature_provenance.csv",
        "somite_detection": tables_dir / "somite_18hpf_gene_detection.csv",
        "somite_association": tables_dir / "somite_18hpf_gene_association.csv",
        "somite_spatial_summary": tables_dir / "somite_18hpf_spatial_null_summary.csv",
        "somite_spatial_null": tables_dir / "somite_18hpf_spatial_null_iterations.csv.gz",
        "fast_muscle_detection": tables_dir / "fast_muscle_24hpf_gene_detection.csv",
        "expression_detection_by_stage_type": tables_dir / "expression_detection_by_stage_type.csv",
        "myog_association": tables_dir / "myog_association.csv",
        "spatial_neighbor_enrichment": tables_dir / "spatial_neighbor_enrichment.csv",
        "raw_type_pair_ranks": tables_dir / "raw_type_pair_ranks.csv.gz",
        "spatial_cells": tables_dir / "somite_18hpf_spatial_cells.csv.gz",
        "trained_display_edges": tables_dir / "trained_jam_display_edges.csv",
        # Preserve the historical transport filename/key for archived loaders;
        # the table itself uses the canonical pre_interaction condition label.
        "trained_init_random_control": tables_dir / "trained_init_random_control.csv",
    }
    contexts.to_csv(paths["context_scores_and_ranks"], index=False, compression="gzip")
    selected.to_csv(paths["somite_case_summary"], index=False)
    edge_inventory.to_csv(paths["edge_inventory"], index=False)
    external_availability = (
        pd.concat(external_audits, ignore_index=True)
        if external_audits
        else pd.DataFrame(
            columns=[
                "method",
                "axis_id",
                "ligand",
                "receptor",
                "external_axis_available",
                "external_availability_provenance_rows",
                "external_stage_coordinate_basis",
                "external_matrix_keys",
                "score_column",
                "score_mode",
            ]
        )
    )
    external_availability.to_csv(paths["external_availability"], index=False)
    gene_audit.to_csv(paths["gene_activity"], index=False)
    provenance.to_csv(paths["literature_provenance"], index=False)
    detection_18.to_csv(paths["somite_detection"], index=False)
    associations_18.to_csv(paths["somite_association"], index=False)
    spatial_summary.to_csv(paths["somite_spatial_summary"], index=False)
    spatial_null.to_csv(paths["somite_spatial_null"], index=False, compression="gzip")
    detection_24.to_csv(paths["fast_muscle_detection"], index=False)
    expression_detection.to_csv(paths["expression_detection_by_stage_type"], index=False)
    associations_18.to_csv(paths["myog_association"], index=False)
    spatial_summary.to_csv(paths["spatial_neighbor_enrichment"], index=False)
    raw_type_pair_ranks.to_csv(paths["raw_type_pair_ranks"], index=False, compression="gzip")
    spatial_panel_cells.to_csv(paths["spatial_cells"], index=False, compression="gzip")
    trained_display_edges.to_csv(paths["trained_display_edges"], index=False)
    controls.to_csv(paths["trained_init_random_control"], index=False)
    readme_path = args.output_dir / "README.md"
    write_readme(readme_path, external_specs)

    manifest = {
        "analysis": "18 hpf Somite Jam2a-Jam3b biology-first reviewer audit",
        "inputs": {
            "h5ad": artifact(args.h5ad),
            "observed_cells": artifact(args.observed_cells),
            "edge_dir": str(args.edge_dir.resolve()),
            "literature_provenance": artifact(args.provenance),
            "preprocess_manifest": (
                artifact(args.preprocess_manifest)
                if args.preprocess_manifest is not None
                else None
            ),
            "trained_pre_interaction_random_control": (
                artifact(args.trained_pre_interaction_random_control)
                if args.trained_pre_interaction_random_control is not None
                else None
            ),
            "external_methods": external_inventory,
        },
        "index_resolution": mapping_resolution,
        "design": {
            "stage": args.stage,
            "stage_label": args.stage_label,
            "case_context": f"{args.sender_type}->{args.receiver_type}",
            "axes": list(provenance["axis_id"]),
            "grouping_seeds": list(expected_seeds),
            "missing_context_edge_in_valid_seed": "zero contribution",
            "grouping_seeds_are_biological_replicates": False,
            "self_edges": "excluded from all CytoBridge score numerators",
            "distinct_denominator": "n_sender*n_receiver-|sender_cell_set intersection receiver_cell_set|",
            "support_filter_applied_before_rank": True,
            "rank_rule": "descending competition/min rank; zeros retained; rank/N, tie_count, and top-tail reported",
            "raw_attention_full_type_pair_rank_universe": "all observed directed 18 hpf annotation pairs with positive distinct-cell denominator; independent of JAM expression support",
            "min_active_unique_edges": int(args.min_active_edges),
            "min_cells_per_side": int(args.min_cells_per_side),
            "external_sparse_zero_rule": "missing context row is zero only for an explicitly available stage x LR axis; unavailable axis is NA",
            "spatial_cutoff": float(args.spatial_cutoff),
            "spatial_cutoff_source": args.spatial_cutoff_source,
            "spatial_permutation_seed": int(args.permutation_seed),
            "spatial_n_permutations": int(args.n_permutations),
            "spatial_null": "independently permute jam2a and jam3b detection labels within 18 hpf Somite cells while preserving marginals and graph",
            "spatial_panel_cells_scope": "all selected-stage cells, with an explicit is_somite flag; no spatial crop or zoom",
            "trained_display_edge_selection": "JAM-compatible Somite->Somite endpoints ranked by trained mean attention times grouping-seed support; no external method or response outcome enters selection",
            "trained_display_edge_minimum_grouping_seed_support": 3,
            "trained_display_edge_maximum_count": 15,
            "trained_pre_interaction_random_control_reconstructed_by_this_script": False,
            "trained_pre_interaction_random_control_handling": (
                "optional precomputed table copied with source hash; absent input "
                "is explicit unavailable/NA"
            ),
        },
        "score_semantics": {
            "cytobridge_raw_attention_magnitude_density": "sum attention_abs_mean on distinct-cell model edges divided by five and by distinct possible pairs; not LR-specific",
            "raw_attention_full_type_pair_density": "same raw attention density ranked over the full directed type-pair square, including zero-edge contexts and without JAM support filtering",
            "cytobridge_lr_only_density": "sum q95-scaled ligand(source)*receptor(target) on the same distinct-cell edge scaffold divided by five and by distinct possible pairs",
            "cytobridge_attention_lr_density": "sum attention_abs_mean*LR activity on the same scaffold divided by five and by distinct possible pairs; post-hoc compatibility score",
            "cytobridge_exact_message_lr_density": "sum exact edge-message joint norm*LR activity on the same scaffold divided by five and by distinct possible pairs; post-hoc compatibility score",
            "external_raw_units_directly_comparable_across_methods": False,
        },
        "claim_guardrails": {
            "attention_is_lr_specific": False,
            "attention_is_communication_probability": False,
            "attention_lr_is_native_model_output": False,
            "message_norm_lr_is_biochemical_flux": False,
            "directed_model_orientation_is_polarized_jam_biochemistry": False,
            "cross_sectional_association_is_regulatory_proof": False,
            "spatial_enrichment_is_direct_contact_proof": False,
            "analysis_is_causal_or_perturbational": False,
            "fast_muscle_24hpf_comparison_is_lineage_tracing": False,
            "same_dataset_external_method_is_independent_validation": False,
            "literature_scope": "supports the heterophilic Jam2a/Jam3b pair and somite-stage fusion context, not model direction or magnitude",
        },
        "panel_data_counts": {
            "n_selected_stage_cells": int(len(spatial_panel_cells)),
            "n_selected_stage_somite_cells": int(spatial_panel_cells["is_somite"].sum()),
            "n_trained_jam_display_edges": int(len(trained_display_edges)),
        },
        "artifacts": {name: artifact(path) for name, path in paths.items()},
        "readme": artifact(readme_path),
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote JAM Somite biology-first audit to {args.output_dir}")


if __name__ == "__main__":
    main()
