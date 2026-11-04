#!/usr/bin/env python3
"""Audit a support-aware 24 hpf zebrafish Delta--Notch family case.

The workflow combines four post-hoc LR-compatible axes (dla/dld x
notch1a/notch3), quantifies where their CytoBridge mass is located, and checks
whether incoming scores align with observed ``her`` response modules.  It is
an observational audit, not an LR-specific model intervention.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Mapping, Sequence

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse, stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


PAIR_FAMILY = (
    ("dla", "notch1a"),
    ("dla", "notch3"),
    ("dld", "notch1a"),
    ("dld", "notch3"),
)
CORE_NEURAL = (
    "Nervous System",
    "Spinal Cord Dorsal Region",
    "Spinal Cord Ventral Region",
    "Spinal Cord Anterior Region",
)
COMPACT_MODULE = ("her4.1", "her6")
BROAD_MODULE = (
    "her4.1",
    "her6",
    "hey1",
    "her9",
    "her12",
    "her15.1",
    "her15.2",
)
SCORE_COLUMNS = ("lr_only", "attention_lr", "exact_message_lr")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--h5ad", required=True, type=Path)
    result.add_argument(
        "--edge-dir",
        required=True,
        type=Path,
        help="One frozen attribution stage directory containing edges_seed_*.csv.gz.",
    )
    result.add_argument(
        "--observed-cells",
        type=Path,
        help=(
            "Attribution observed_cells.csv.gz. Defaults to the parent of "
            "--edge-dir. It is always required to verify edge endpoint identity."
        ),
    )
    result.add_argument("--top-axes", required=True, type=Path)
    result.add_argument("--pathway-enrichment", required=True, type=Path)
    result.add_argument("--output-dir", required=True, type=Path)
    result.add_argument("--stage", type=float, default=4.0)
    result.add_argument("--stage-label", default="24hpf")
    result.add_argument("--time-key", default="time_point_processed")
    result.add_argument("--time-label-key", default="time")
    result.add_argument("--annotation-key", default="Annotation")
    result.add_argument("--overwrite", action="store_true")
    return result


def require(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    missing = [column for column in columns if column not in frame]
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")


def prepare_output(path: Path, overwrite: bool) -> None:
    """Create an output directory without silently mixing old and new files."""
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
    if not path.exists():
        path.mkdir(parents=True, exist_ok=False)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dense_column(matrix, index: int) -> np.ndarray:
    value = matrix[:, int(index)]
    value = value.toarray() if sparse.issparse(value) else np.asarray(value)
    return np.asarray(value, dtype=np.float64).reshape(-1)


def global_q95_activity(values: np.ndarray) -> np.ndarray:
    """Match the reviewer bundle's global positive-q95 activity definition."""
    values = np.asarray(values, dtype=np.float64)
    positive = values[values > 0]
    scale = float(np.quantile(positive, 0.95)) if positive.size else 1.0
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    return np.clip(values / scale, 0.0, 1.0)


def zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    scale = float(np.std(values, ddof=0))
    return (values - float(np.mean(values))) / (scale if scale > 0 else 1.0)


def _integer_values(series: pd.Series, label: str) -> np.ndarray:
    numeric = pd.to_numeric(series, errors="raise").to_numpy(float)
    if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
        raise ValueError(f"{label} must contain finite integer values")
    return numeric.astype(int)


def _observed_to_h5ad_mapping(
    observed: pd.DataFrame,
    data: ad.AnnData,
    *,
    stage: float,
    stage_label: str,
    time_key: str,
    annotation_key: str,
) -> tuple[dict[int, int], pd.DataFrame]:
    require(
        observed,
        ["global_index", "obs_name", "stage", "stage_label", "cell_type"],
        "observed-cells table",
    )
    table = observed.copy()
    table["global_index"] = _integer_values(table["global_index"], "global_index")
    if table["global_index"].duplicated().any():
        raise ValueError("observed-cells global_index is not unique")
    table["obs_name"] = table["obs_name"].astype(str)
    if table["obs_name"].duplicated().any():
        raise ValueError("observed-cells obs_name is not unique")
    if not data.obs_names.is_unique:
        raise ValueError("H5AD obs_names must be unique for observed-cell mapping")
    h5_lookup = {name: index for index, name in enumerate(data.obs_names.astype(str))}
    missing = sorted(set(table["obs_name"]).difference(h5_lookup))
    if missing:
        preview = missing[:5]
        raise ValueError(
            f"observed-cells contains {len(missing)} obs_name values absent from H5AD; "
            f"examples={preview}"
        )
    if len(table) != data.n_obs or set(table["obs_name"]) != set(
        data.obs_names.astype(str)
    ):
        raise ValueError(
            "observed-cells and H5AD must contain exactly the same obs_name universe"
        )
    table["h5ad_index"] = table["obs_name"].map(h5_lookup).astype(int)
    table["stage"] = pd.to_numeric(table["stage"], errors="raise").astype(float)
    h5_stage = pd.to_numeric(
        data.obs.iloc[table["h5ad_index"].to_numpy(int)][time_key],
        errors="raise",
    ).to_numpy(float)
    if not np.isclose(table["stage"].to_numpy(float), h5_stage, rtol=0.0, atol=1e-12).all():
        raise ValueError("observed-cells stage disagrees with H5AD")
    h5_type = data.obs.iloc[table["h5ad_index"].to_numpy(int)][annotation_key].astype(str)
    if not np.array_equal(table["cell_type"].astype(str).to_numpy(), h5_type.to_numpy()):
        raise ValueError(
            f"observed-cells cell_type disagrees with H5AD {annotation_key}"
        )
    selected = table.loc[
        np.isclose(table["stage"], float(stage), rtol=0.0, atol=1e-12)
    ].copy()
    if selected.empty:
        raise ValueError(f"observed-cells has no rows for stage={stage:g}")
    labels = set(selected["stage_label"].astype(str))
    if labels != {str(stage_label)}:
        raise ValueError(
            f"observed-cells stage-label mismatch for stage={stage:g}: {sorted(labels)}"
        )
    mapping = dict(zip(table["global_index"], table["h5ad_index"]))
    return mapping, selected


def resolve_edge_stage_indices(
    edges: pd.DataFrame,
    data: ad.AnnData,
    stage_mask: np.ndarray,
    *,
    stage: float,
    stage_label: str,
    time_key: str,
    annotation_key: str,
    observed_cells: pd.DataFrame | None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Resolve edge endpoints to positions in the stage-specific H5AD slice.

    ``source_index_stage``/``target_index_stage`` are authoritative when both
    are present.  If they are absent, attribution-global indices are mapped by
    ``observed_cells.obs_name`` into the supplied H5AD.  We never silently
    treat attribution-global indices as H5AD row positions.
    """
    require(
        edges,
        ["source_index", "target_index", "sender_type", "receiver_type"],
        "edge table",
    )
    if "stage" in edges:
        values = pd.to_numeric(edges["stage"], errors="raise").to_numpy(float)
        if not np.isclose(values, float(stage), rtol=0.0, atol=1e-12).all():
            raise ValueError("Edge table contains rows from an unexpected stage")
    if "stage_label" in edges:
        labels = set(edges["stage_label"].astype(str))
        if labels != {str(stage_label)}:
            raise ValueError(f"Edge stage-label mismatch: {sorted(labels)}")

    stage_global = np.flatnonzero(np.asarray(stage_mask, dtype=bool))
    if stage_global.size == 0:
        raise ValueError(f"H5AD has no cells for stage={stage:g}")
    global_to_local = np.full(data.n_obs, -1, dtype=int)
    global_to_local[stage_global] = np.arange(stage_global.size, dtype=int)
    annotations = data.obs.iloc[stage_global][annotation_key].astype(str).to_numpy()

    has_source_local = "source_index_stage" in edges
    has_target_local = "target_index_stage" in edges
    if has_source_local != has_target_local:
        raise ValueError(
            "Edge table must contain both source_index_stage and "
            "target_index_stage, or neither"
        )

    result = edges.copy()
    mapping_mode: str
    if observed_cells is None:
        raise ValueError(
            "A valid observed-cells mapping is required even when stage-local "
            "edge indices are present"
        )
    observed_map, observed_selected = _observed_to_h5ad_mapping(
        observed_cells,
        data,
        stage=stage,
        stage_label=stage_label,
        time_key=time_key,
        annotation_key=annotation_key,
    )

    if has_source_local:
        source_local = _integer_values(
            result["source_index_stage"], "source_index_stage"
        )
        target_local = _integer_values(
            result["target_index_stage"], "target_index_stage"
        )
        for label, values in (
            ("source_index_stage", source_local),
            ("target_index_stage", target_local),
        ):
            if values.min() < 0 or values.max() >= stage_global.size:
                raise IndexError(
                    f"{label} is outside the {stage_global.size}-cell stage slice"
                )
        mapping_mode = "explicit_stage_local_columns"

        # If attribution-global columns are also present, verify rather than
        # silently trusting a particular H5AD global row order.
        for role, local in (("source", source_local), ("target", target_local)):
            global_column = f"{role}_index"
            if global_column not in result:
                continue
            attribution_global = _integer_values(result[global_column], global_column)
            missing = sorted(set(attribution_global).difference(observed_map))
            if missing:
                raise ValueError(
                    f"{global_column} contains indices absent from observed-cells: "
                    f"{missing[:5]}"
                )
            h5_global = np.asarray(
                [observed_map[index] for index in attribution_global], dtype=int
            )
            mapped_local = global_to_local[h5_global]
            if np.any(mapped_local < 0) or not np.array_equal(mapped_local, local):
                raise ValueError(
                    f"{role} stage-local indices disagree with the verified global mapping"
                )
    else:
        require(result, ["source_index", "target_index"], "edge table")
        if observed_selected is None:  # pragma: no cover - guarded above
            raise AssertionError("observed-cells mapping was not initialized")
        resolved: dict[str, np.ndarray] = {}
        for role in ("source", "target"):
            column = f"{role}_index"
            attribution_global = _integer_values(result[column], column)
            missing = sorted(set(attribution_global).difference(observed_map))
            if missing:
                raise ValueError(
                    f"{column} contains indices absent from observed-cells: {missing[:5]}"
                )
            h5_global = np.asarray(
                [observed_map[index] for index in attribution_global], dtype=int
            )
            local = global_to_local[h5_global]
            if np.any(local < 0):
                raise ValueError(f"{column} maps outside requested stage={stage:g}")
            resolved[role] = local
        source_local = resolved["source"]
        target_local = resolved["target"]
        mapping_mode = "observed_cells_obs_name_to_h5ad"

    sender = result["sender_type"].astype(str).to_numpy()
    receiver = result["receiver_type"].astype(str).to_numpy()
    if not np.array_equal(sender, annotations[source_local]):
        raise ValueError("sender_type disagrees with resolved H5AD source annotations")
    if not np.array_equal(receiver, annotations[target_local]):
        raise ValueError("receiver_type disagrees with resolved H5AD target annotations")

    result["_source_stage_index"] = source_local
    result["_target_stage_index"] = target_local
    return result, {
        "mode": mapping_mode,
        "n_stage_cells": int(stage_global.size),
        "observed_cells_used": True,
        "stage_local_columns_present": bool(has_source_local),
        "global_index_order_assumed_without_validation": False,
    }


def load_edge_occurrences(
    edge_dir: Path,
    data: ad.AnnData,
    stage_mask: np.ndarray,
    *,
    stage: float,
    stage_label: str,
    time_key: str,
    annotation_key: str,
    observed_cells_path: Path | None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, object]]]:
    paths = sorted(edge_dir.glob("edges_seed_*.csv.gz"))
    if not paths:
        raise FileNotFoundError(f"No edges_seed_*.csv.gz under {edge_dir}")
    observed = None
    if observed_cells_path is None or not observed_cells_path.is_file():
        raise FileNotFoundError(
            "A valid observed_cells.csv.gz is required for edge endpoint verification: "
            f"{observed_cells_path}"
        )
    observed = pd.read_csv(observed_cells_path)
    inventories: list[dict[str, object]] = []
    resolved_frames: list[pd.DataFrame] = []
    resolutions: list[dict[str, object]] = []
    required = [
        "grouping_seed",
        "sender_type",
        "receiver_type",
        "attention_abs_mean",
        "edge_message_norm_joint",
    ]
    for path in paths:
        frame = pd.read_csv(path)
        require(frame, required, str(path))
        seeds = _integer_values(frame["grouping_seed"], "grouping_seed")
        if np.unique(seeds).size != 1:
            raise ValueError(f"Ambiguous grouping seed in {path}")
        resolved, metadata = resolve_edge_stage_indices(
            frame,
            data,
            stage_mask,
            stage=stage,
            stage_label=stage_label,
            time_key=time_key,
            annotation_key=annotation_key,
            observed_cells=observed,
        )
        duplicate_key = [
            "grouping_seed",
            "_source_stage_index",
            "_target_stage_index",
        ]
        if resolved.duplicated(duplicate_key).any():
            raise ValueError(f"Duplicate source-target edge within a seed: {path}")
        resolved_frames.append(resolved)
        inventories.append(
            {
                "path": str(path.resolve()),
                "sha256": sha256(path),
                "grouping_seed": int(seeds[0]),
                "n_rows": int(len(frame)),
                "mapping_mode": metadata["mode"],
            }
        )
        resolutions.append(metadata)
    modes = {item["mode"] for item in resolutions}
    if len(modes) != 1:
        raise ValueError(f"Inconsistent edge index resolution modes: {sorted(modes)}")
    return (
        pd.concat(resolved_frames, ignore_index=True),
        pd.DataFrame(inventories),
        resolutions,
    )


def build_pair_edges(
    raw_edges: pd.DataFrame,
    activities: Mapping[str, np.ndarray],
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    source = raw_edges["_source_stage_index"].to_numpy(int)
    target = raw_edges["_target_stage_index"].to_numpy(int)
    for ligand, receptor in PAIR_FAMILY:
        frame = raw_edges.copy()
        lr = activities[ligand][source] * activities[receptor][target]
        frame["pair"] = f"{ligand}->{receptor}"
        frame["lr_only"] = lr
        frame["attention_lr"] = lr * frame["attention_abs_mean"].to_numpy(float)
        frame["exact_message_lr"] = (
            lr * frame["edge_message_norm_joint"].to_numpy(float)
        )
        parts.append(
            frame.groupby(
                [
                    "pair",
                    "_source_stage_index",
                    "_target_stage_index",
                    "sender_type",
                    "receiver_type",
                ],
                observed=True,
                as_index=False,
            )[list(SCORE_COLUMNS)].mean()
        )
    return pd.concat(parts, ignore_index=True)


def context_mass_table(edges: pd.DataFrame) -> pd.DataFrame:
    contexts = {
        "core_neural_to_core_neural": (
            edges["sender_type"].isin(CORE_NEURAL)
            & edges["receiver_type"].isin(CORE_NEURAL)
        ),
        "same_annotation": edges["sender_type"].eq(edges["receiver_type"]),
        "ventral_spinal_self": (
            edges["sender_type"].eq("Spinal Cord Ventral Region")
            & edges["receiver_type"].eq("Spinal Cord Ventral Region")
        ),
        "nervous_system_self": (
            edges["sender_type"].eq("Nervous System")
            & edges["receiver_type"].eq("Nervous System")
        ),
    }
    rows: list[dict[str, object]] = []
    groups = list(edges.groupby("pair", observed=True)) + [("four_pair_family", edges)]
    for pair, frame in groups:
        for context, mask in contexts.items():
            local_mask = mask.loc[frame.index]
            for score in SCORE_COLUMNS:
                denominator = float(frame[score].sum())
                if not np.isfinite(denominator) or denominator <= 0:
                    raise ValueError(f"Non-positive family mass for pair={pair}, score={score}")
                numerator = float(frame.loc[local_mask, score].sum())
                rows.append(
                    {
                        "pair": pair,
                        "context": context,
                        "score": score,
                        "numerator": numerator,
                        "denominator": denominator,
                        "mass_fraction": numerator / denominator,
                    }
                )
    return pd.DataFrame(rows)


def residual_audit(
    frame: pd.DataFrame,
    *,
    module: str,
    baseline_name: str,
    baseline: np.ndarray,
    score: str,
) -> dict[str, object]:
    y = frame[module].to_numpy(float)
    base_fit = LinearRegression(fit_intercept=False).fit(baseline, y)
    base_prediction = base_fit.predict(baseline)
    y_residual = y - base_prediction
    base_r2 = float(r2_score(y, base_prediction))
    x = np.log1p(frame[score].to_numpy(float))[:, None]
    x_fit = LinearRegression(fit_intercept=False).fit(baseline, x)
    x_residual = x[:, 0] - x_fit.predict(baseline)[:, 0]
    if np.std(x_residual) <= 0 or np.std(y_residual) <= 0:
        partial_r = np.nan
    else:
        partial_r = float(stats.pearsonr(x_residual, y_residual).statistic)
    extended = np.column_stack([baseline, x])
    extended_fit = LinearRegression(fit_intercept=False).fit(extended, y)
    extended_r2 = float(r2_score(y, extended_fit.predict(extended)))
    return {
        "module": module,
        "baseline": baseline_name,
        "score": score,
        "n_cells": int(len(frame)),
        "partial_pearson_r": partial_r,
        "base_r2": base_r2,
        "extended_r2": extended_r2,
        "delta_r2": extended_r2 - base_r2,
    }


def downstream_tables(
    edges: pd.DataFrame,
    expression: Mapping[str, np.ndarray],
    annotations: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n_cells = int(len(annotations))
    incoming = (
        edges.groupby("_target_stage_index", observed=True)[list(SCORE_COLUMNS)]
        .sum()
        .reindex(np.arange(n_cells))
        .fillna(0.0)
    )
    cells = pd.DataFrame(index=np.arange(n_cells))
    cells[list(SCORE_COLUMNS)] = incoming.to_numpy()
    cells["cell_type"] = np.asarray(annotations, dtype=str)
    for gene, values in expression.items():
        cells[gene] = np.asarray(values, dtype=float)
    cells["her46"] = np.mean(
        [zscore(cells[gene]) for gene in COMPACT_MODULE], axis=0
    )
    cells["her_broad"] = np.mean(
        [zscore(cells[gene]) for gene in BROAD_MODULE], axis=0
    )
    core = cells.loc[cells["cell_type"].isin(CORE_NEURAL)].copy()
    if len(core) < 20:
        raise ValueError(f"Too few core-neural cells for downstream audit: {len(core)}")

    correlations: list[dict[str, object]] = []
    for module in ("her46", "her_broad"):
        for score in SCORE_COLUMNS:
            result = stats.spearmanr(core[score], core[module])
            correlations.append(
                {
                    "subset": "core_neural",
                    "module": module,
                    "score": score,
                    "n_cells": int(len(core)),
                    "spearman_rho": float(result.statistic),
                    "p_value_descriptive_only": float(result.pvalue),
                }
            )

    cell_type_matrix = pd.get_dummies(
        core["cell_type"], drop_first=False, dtype=float
    ).to_numpy()
    baseline_receptors = np.column_stack(
        [
            np.ones(len(core)),
            cell_type_matrix,
            np.log1p(core["notch1a"].to_numpy(float)),
            np.log1p(core["notch3"].to_numpy(float)),
        ]
    )
    baseline_plus_lr = np.column_stack(
        [baseline_receptors, np.log1p(core["lr_only"].to_numpy(float))]
    )
    residuals: list[dict[str, object]] = []
    for module in ("her46", "her_broad"):
        for score in ("attention_lr", "exact_message_lr"):
            residuals.append(
                residual_audit(
                    core,
                    module=module,
                    baseline_name="cell_type+notch1a+notch3",
                    baseline=baseline_receptors,
                    score=score,
                )
            )
            residuals.append(
                residual_audit(
                    core,
                    module=module,
                    baseline_name="cell_type+notch1a+notch3+lr_only",
                    baseline=baseline_plus_lr,
                    score=score,
                )
            )

    detection: list[dict[str, object]] = []
    for cell_type in CORE_NEURAL:
        subset = cells.loc[cells["cell_type"].eq(cell_type)]
        for gene in ("dla", "dld", "notch1a", "notch3", "her4.1", "her6"):
            detection.append(
                {
                    "cell_type": cell_type,
                    "n_cells": int(len(subset)),
                    "gene": gene,
                    "detected_fraction_x_gt_zero": float((subset[gene] > 0).mean()),
                    "mean_x": float(subset[gene].mean()),
                }
            )
    return (
        cells.reset_index(names="stage_index"),
        pd.DataFrame(correlations),
        pd.DataFrame(residuals),
        pd.DataFrame(detection),
    )


def rank_and_enrichment_tables(
    top_axes_path: Path,
    enrichment_path: Path,
    *,
    stage_label: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    top_axes = pd.read_csv(top_axes_path)
    require(
        top_axes,
        ["stage_label", "ranking_target", "ligand", "receptor", "rank"],
        "top-axis table",
    )
    family = {f"{ligand}->{receptor}" for ligand, receptor in PAIR_FAMILY}
    axis_id = top_axes["ligand"].astype(str) + "->" + top_axes["receptor"].astype(str)
    selected = top_axes.loc[
        top_axes["stage_label"].astype(str).eq(stage_label)
        & axis_id.isin(family)
        & top_axes["ranking_target"].isin(["attention", "exact_message"])
    ].copy()
    selected["axis_id"] = (
        selected["ligand"].astype(str) + "->" + selected["receptor"].astype(str)
    )
    expected = pd.MultiIndex.from_product(
        [["attention", "exact_message"], sorted(family)],
        names=["ranking_target", "axis_id"],
    )
    actual = pd.MultiIndex.from_frame(selected[["ranking_target", "axis_id"]])
    if actual.has_duplicates or set(actual) != set(expected):
        raise ValueError(
            "Top-axis table does not contain exactly the expected four Delta--Notch "
            "pairs for both ranking targets"
        )

    enrichment = pd.read_csv(enrichment_path)
    require(enrichment, ["pathway", "fold_enrichment", "bh_q"], "enrichment table")
    notch = enrichment.loc[enrichment["pathway"].astype(str).eq("NOTCH")].copy()
    if len(notch) != 1:
        raise ValueError(f"Expected one NOTCH enrichment row, found {len(notch)}")
    return selected.sort_values(["ranking_target", "rank"]), notch


def main() -> None:
    args = parser().parse_args()
    prepare_output(args.output_dir, args.overwrite)
    tables = args.output_dir / "tables"
    tables.mkdir()

    data = ad.read_h5ad(args.h5ad)
    require(
        data.obs,
        [args.time_key, args.time_label_key, args.annotation_key],
        "H5AD obs",
    )
    stage_values = pd.to_numeric(data.obs[args.time_key], errors="raise").to_numpy(float)
    stage_mask = np.isclose(stage_values, float(args.stage), rtol=0.0, atol=1e-12)
    if not stage_mask.any():
        raise ValueError(f"H5AD has no cells for stage={args.stage:g}")
    labels = set(data.obs.loc[stage_mask, args.time_label_key].astype(str))
    if labels != {str(args.stage_label)}:
        raise ValueError(
            f"H5AD stage-label mismatch for stage={args.stage:g}: {sorted(labels)}"
        )
    stage_annotations = (
        data.obs.loc[stage_mask, args.annotation_key].astype(str).to_numpy()
    )

    genes = tuple(
        dict.fromkeys(
            [gene for pair in PAIR_FAMILY for gene in pair] + list(BROAD_MODULE)
        )
    )
    missing = sorted(set(genes).difference(data.var_names.astype(str)))
    if missing:
        raise ValueError(f"H5AD is missing required genes: {missing}")
    expression_all = {
        gene: dense_column(data.X, int(data.var_names.get_loc(gene))) for gene in genes
    }
    expression_stage = {
        gene: values[stage_mask] for gene, values in expression_all.items()
    }
    family_gene_set = {gene for pair in PAIR_FAMILY for gene in pair}
    activities_stage = {
        gene: global_q95_activity(values)[stage_mask]
        for gene, values in expression_all.items()
        if gene in family_gene_set
    }

    observed_path = args.observed_cells
    if observed_path is None:
        candidate = args.edge_dir.parent / "observed_cells.csv.gz"
        observed_path = candidate
    if not observed_path.is_file():
        raise FileNotFoundError(
            "A valid observed_cells.csv.gz is required for edge endpoint verification: "
            f"{observed_path}"
        )
    raw_edges, inventory, resolutions = load_edge_occurrences(
        args.edge_dir,
        data,
        stage_mask,
        stage=args.stage,
        stage_label=args.stage_label,
        time_key=args.time_key,
        annotation_key=args.annotation_key,
        observed_cells_path=observed_path,
    )
    pair_edges = build_pair_edges(raw_edges, activities_stage)
    mass = context_mass_table(pair_edges)
    cells, correlations, residuals, detection = downstream_tables(
        pair_edges, expression_stage, stage_annotations
    )
    ranks, enrichment = rank_and_enrichment_tables(
        args.top_axes, args.pathway_enrichment, stage_label=args.stage_label
    )

    inventory.to_csv(tables / "edge_input_inventory.csv", index=False)
    mass.to_csv(tables / "delta_notch_family_context_mass.csv", index=False)
    cells.to_csv(tables / "delta_notch_receiver_cell_scores.csv.gz", index=False)
    correlations.to_csv(tables / "delta_notch_downstream_correlations.csv", index=False)
    residuals.to_csv(tables / "delta_notch_downstream_residual_audit.csv", index=False)
    detection.to_csv(tables / "delta_notch_expression_detection.csv", index=False)
    ranks.to_csv(tables / "delta_notch_family_ranks.csv", index=False)
    enrichment.to_csv(tables / "delta_notch_pathway_enrichment.csv", index=False)

    resolution = resolutions[0]
    manifest = {
        "schema_version": 2,
        "analysis": "zebrafish_delta_notch_family_observational_audit",
        "status": "complete",
        "stage": float(args.stage),
        "stage_label": str(args.stage_label),
        "pair_family": [list(pair) for pair in PAIR_FAMILY],
        "core_neural": list(CORE_NEURAL),
        "lr_activity": (
            "full-H5AD positive q95 per gene, clip [0,1], then "
            "ligand[source] x receptor[target]"
        ),
        "edge_seed_collapse": (
            "mean over grouping seeds in which a source-target edge occurs; "
            "missing edge/seed combinations are not zero-imputed"
        ),
        "incoming_score": (
            "sum over all four family pairs and all source cells for each target cell"
        ),
        "compact_module": list(COMPACT_MODULE),
        "broad_module": list(BROAD_MODULE),
        "module_scaling": (
            "each gene z-scored over all cells at the selected stage before an "
            "unweighted module mean"
        ),
        "index_resolution": resolution,
        "guardrails": {
            "attention_is_lr_specific": False,
            "attention_times_lr_is_model_native_probability": False,
            "exact_message_times_lr_is_biochemical_flux": False,
            "downstream_association_is_independent_validation": False,
            "cell_rows_or_grouping_seeds_are_biological_replicates": False,
            "literature_or_external_method_support_proves_inferred_direction": False,
            "analysis_is_a_causal_or_lr_specific_perturbation": False,
            "global_h5ad_row_order_was_silently_assumed": False,
        },
        "inputs": {
            "h5ad": str(args.h5ad.resolve()),
            "edge_dir": str(args.edge_dir.resolve()),
            "observed_cells": (
                None if observed_path is None else str(observed_path.resolve())
            ),
            "top_axes": str(args.top_axes.resolve()),
            "pathway_enrichment": str(args.pathway_enrichment.resolve()),
        },
        "outputs": {
            "n_stage_cells": int(stage_mask.sum()),
            "n_edge_occurrences": int(len(raw_edges)),
            "n_pair_edges_after_seed_collapse": int(len(pair_edges)),
            "n_core_neural_cells": int(
                np.isin(stage_annotations, np.asarray(CORE_NEURAL)).sum()
            ),
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest["outputs"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
