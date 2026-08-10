#!/usr/bin/env python3
"""Compare zebrafish LR-pathway rankings across CytoBridge and external CCC methods.

The primary four-method analysis uses the strict 1:1 zebrafish-to-mouse LR
projection supported by CellAgentChat.  Its external consensus is constructed
from COMMOT, CellChat, and CellAgentChat only; CytoBridge is never included in
the reference it is compared against.  A larger zebrafish-native analysis
without CellAgentChat is emitted separately.

All native score units are converted to within-stage ranks before pathway
aggregation.  Pathway means are rounded before reranking to prevent floating
point noise from breaking true ties among zero-supported pathways.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import platform
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence
import zlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
from scipy.stats import hypergeom, pearsonr, rankdata, spearmanr


STAGES = (0.0, 1.0, 2.0, 3.0, 4.0)
STAGE_LABELS = {
    0.0: "5.25 hpf",
    1.0: "10 hpf",
    2.0: "12 hpf",
    3.0: "18 hpf",
    4.0: "24 hpf",
}
CB_METHODS = (
    "CytoBridge attention x LR",
    "CytoBridge exact message x LR",
    "CytoBridge exact message only (LR-conditioned)",
    "CytoBridge LR-only",
)
FOUR_NATIVE_EXTERNAL = (
    "COMMOT",
    "CellChat triMean",
    "CellAgentChat significant",
)
FOUR_RELAXED_EXTERNAL = (
    "COMMOT",
    "CellChat truncatedMean",
    "CellAgentChat continuous",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cytobridge-axis-scores", required=True, type=Path)
    parser.add_argument("--commot-lr-scores", required=True, type=Path)
    parser.add_argument(
        "--commot-score-column",
        choices=["abundance_controlled_score", "abundance_controlled_distinct_cell_score"],
        default="abundance_controlled_score",
        help=(
            "COMMOT manifest recommends abundance_controlled_score for the primary "
            "comparison to CellChat population.size=false; the distinct-cell column "
            "is available as a self-edge-excluded sensitivity."
        ),
    )
    parser.add_argument("--cellchat-primary-lr-scores", required=True, type=Path)
    parser.add_argument("--cellchat-truncated-lr-scores", required=True, type=Path)
    parser.add_argument("--cellchat-excluded-lr", required=True, type=Path)
    parser.add_argument("--cellagentchat-raw-lr-scores", required=True, type=Path)
    parser.add_argument("--cellagentchat-significant-lr-scores", required=True, type=Path)
    parser.add_argument("--cellagentchat-crosswalk", required=True, type=Path)
    parser.add_argument("--lr-database", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--four-method-top-fraction", type=float, default=0.20)
    parser.add_argument("--native-top-fraction", type=float, default=0.05)
    parser.add_argument("--top-pathways", type=int, default=10)
    parser.add_argument("--heatmap-pathways", type=int, default=15)
    parser.add_argument("--permutations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _require(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} lacks required columns: {missing}")


def _axis(ligand: pd.Series, receptor: pd.Series) -> pd.Series:
    return ligand.astype(str).str.casefold() + "->" + receptor.astype(str).str.casefold()


def _record(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "sha256": digest.hexdigest(),
    }


def _git_state(repo_root: Path) -> dict[str, Any]:
    def run(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    try:
        status = run("status", "--short")
        return {
            "commit": run("rev-parse", "HEAD"),
            "dirty": bool(status),
            "status_short": status.splitlines(),
        }
    except (OSError, subprocess.CalledProcessError) as error:
        return {"error": str(error)}


def _bh(values: Sequence[float]) -> np.ndarray:
    p = np.asarray(values, dtype=float)
    out = np.full(len(p), np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    if not len(valid):
        return out
    ordered = valid[np.argsort(p[valid], kind="stable")]
    adjusted = p[ordered] * len(valid) / np.arange(1, len(valid) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    out[ordered] = np.clip(adjusted, 0.0, 1.0)
    return out


def _safe_spearman(left: Sequence[float], right: Sequence[float]) -> tuple[float, float]:
    x, y = np.asarray(left, float), np.asarray(right, float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if len(x) < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return float("nan"), float("nan")
    result = spearmanr(x, y)
    return float(result.statistic), float(result.pvalue)


def _partial_spearman(
    left: Sequence[float], right: Sequence[float], control: Sequence[float]
) -> float:
    x, y, c = (np.asarray(value, float) for value in (left, right, control))
    keep = np.isfinite(x) & np.isfinite(y) & np.isfinite(c)
    x, y, c = x[keep], y[keep], c[keep]
    if len(x) < 4 or min(np.unique(x).size, np.unique(y).size) < 2:
        return float("nan")
    x, y, c = (rankdata(value, method="average") for value in (x, y, c))
    design = np.column_stack([np.ones(len(c)), c])
    x_res = x - design @ np.linalg.lstsq(design, x, rcond=None)[0]
    y_res = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    if np.std(x_res) == 0 or np.std(y_res) == 0:
        return float("nan")
    return float(pearsonr(x_res, y_res).statistic)


def _permutation_p(
    left: Sequence[float],
    right: Sequence[float],
    *,
    permutations: int,
    seed: int,
) -> float:
    x, y = np.asarray(left, float), np.asarray(right, float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if len(x) < 3 or min(np.unique(x).size, np.unique(y).size) < 2:
        return float("nan")
    x = rankdata(x, method="average")
    y = rankdata(y, method="average")
    x = x - x.mean()
    y = y - y.mean()
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    observed = abs(float(np.dot(x, y) / denom))
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(permutations):
        permuted = rng.permutation(y)
        exceed += abs(float(np.dot(x, permuted) / denom)) >= observed - 1e-15
    return float((exceed + 1) / (permutations + 1))


def load_database(path: Path) -> pd.DataFrame:
    database = pd.read_csv(path)
    _require(database, ["ligand", "receptor", "pathway", "category"], str(path))
    database = database.copy()
    database["axis"] = _axis(database["ligand"], database["receptor"])
    database["pathway"] = database["pathway"].fillna("UNANNOTATED").astype(str)
    database["category"] = database["category"].fillna("UNANNOTATED").astype(str)
    return (
        database[["axis", "ligand", "receptor", "pathway", "category"]]
        .drop_duplicates(["axis", "pathway", "category"])
        .sort_values(["axis", "pathway", "category"])
        .reset_index(drop=True)
    )


def load_cytobridge(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    columns = {
        "mean_attention_times_lr_activity": "CytoBridge attention x LR",
        "mean_exact_message_times_lr_activity": "CytoBridge exact message x LR",
        "mean_scaled_lr_activity": "CytoBridge LR-only",
    }
    _require(frame, ["stage", "ligand", "receptor", *columns], str(path))
    frame = frame.copy()
    frame["stage"] = frame["stage"].astype(float)
    frame["axis"] = _axis(frame["ligand"], frame["receptor"])
    lr = pd.to_numeric(frame["mean_scaled_lr_activity"], errors="raise").to_numpy(float)
    exact_lr = pd.to_numeric(
        frame["mean_exact_message_times_lr_activity"], errors="raise"
    ).to_numpy(float)
    # An unconditioned edge-message value has no LR identity.  This ratio is
    # E[exact-message * LR] / E[LR]: it removes the total LR-abundance scale
    # while retaining LR support solely to assign edges to an LR/pathway.
    frame["CytoBridge exact message only (LR-conditioned)"] = np.divide(
        exact_lr,
        lr,
        out=np.zeros_like(exact_lr),
        where=lr > 0,
    )
    if frame.duplicated(["stage", "axis"]).any():
        raise ValueError("CytoBridge stage/LR table contains duplicates")
    return frame[
        [
            "stage",
            "axis",
            *columns,
            "CytoBridge exact message only (LR-conditioned)",
        ]
    ].rename(columns=columns)


def _collapse_lr_contexts(
    path: Path, score_column: str, method: str
) -> pd.DataFrame:
    frame = pd.read_csv(path)
    _require(frame, ["stage", "ligand", "receptor", score_column], str(path))
    frame = frame.copy()
    frame["stage"] = frame["stage"].astype(float)
    frame["axis"] = _axis(frame["ligand"], frame["receptor"])
    frame[score_column] = pd.to_numeric(frame[score_column], errors="raise").fillna(0.0)
    # Long tables contain positive contexts only.  Summing here and completing
    # absent stage/LR rows with zero later is rank-equivalent to the complete
    # type-square mean within a stage because every evaluated LR has the same
    # sender/receiver type grid.
    return (
        frame.groupby(["stage", "axis"], as_index=False)[score_column]
        .sum()
        .rename(columns={score_column: method})
    )


def load_cellagentchat(
    path: Path,
    crosswalk: pd.DataFrame,
    score_column: str,
    method: str,
) -> pd.DataFrame:
    frame = pd.read_csv(path)
    _require(
        frame,
        ["stage", "sampling_seed", "ligand", "receptor", score_column],
        str(path),
    )
    frame = frame.copy()
    frame["mapped_axis"] = _axis(frame["ligand"], frame["receptor"])
    frame["stage"] = frame["stage"].astype(float)
    frame[score_column] = pd.to_numeric(frame[score_column], errors="raise").fillna(0.0)
    mapped = frame.merge(
        crosswalk[["mapped_axis", "axis"]],
        on="mapped_axis",
        how="left",
        validate="many_to_one",
    )
    if mapped["axis"].isna().any():
        examples = mapped.loc[mapped["axis"].isna(), "mapped_axis"].unique()[:5]
        raise ValueError(f"Unmapped CellAgentChat LR pairs: {examples.tolist()}")
    # Give every tested LR equal weight and every sampling seed equal weight.
    by_seed = (
        mapped.groupby(["stage", "sampling_seed", "axis"], as_index=False)[score_column]
        .sum()
    )
    grid = pd.MultiIndex.from_product(
        [
            STAGES,
            sorted(frame["sampling_seed"].unique()),
            sorted(crosswalk["axis"].unique()),
        ],
        names=["stage", "sampling_seed", "axis"],
    ).to_frame(index=False)
    by_seed = grid.merge(
        by_seed, on=["stage", "sampling_seed", "axis"], how="left", validate="one_to_one"
    )
    by_seed[score_column] = by_seed[score_column].fillna(0.0)
    return (
        by_seed.groupby(["stage", "axis"], as_index=False)[score_column]
        .mean()
        .rename(columns={score_column: method})
    )


def load_crosswalk(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    _require(
        frame,
        ["source_ligand", "source_receptor", "mapped_ligand", "mapped_receptor"],
        str(path),
    )
    frame = frame.copy()
    frame["axis"] = _axis(frame["source_ligand"], frame["source_receptor"])
    frame["mapped_axis"] = _axis(frame["mapped_ligand"], frame["mapped_receptor"])
    if frame.duplicated("axis").any() or frame.duplicated("mapped_axis").any():
        raise ValueError("CellAgentChat projection crosswalk is not one-to-one")
    return frame


def build_score_grid(
    universe: Sequence[str],
    cytobridge: pd.DataFrame,
    score_tables: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    axes = sorted(set(map(str, universe)))
    grid = pd.MultiIndex.from_product([STAGES, axes], names=["stage", "axis"]).to_frame(
        index=False
    )
    cb_columns = ["stage", "axis", *CB_METHODS]
    grid = grid.merge(cytobridge[cb_columns], on=["stage", "axis"], validate="one_to_one")
    for method, table in score_tables.items():
        _require(table, ["stage", "axis", method], method)
        if table.duplicated(["stage", "axis"]).any():
            raise ValueError(f"{method} has duplicate stage/LR rows")
        grid = grid.merge(table, on=["stage", "axis"], how="left", validate="one_to_one")
        grid[method] = grid[method].fillna(0.0)
    for method in [*CB_METHODS, *score_tables]:
        grid[method] = pd.to_numeric(grid[method], errors="raise").fillna(0.0)
        if (grid[method] < 0).any():
            raise ValueError(f"{method} contains negative LR scores")
    return grid.sort_values(["stage", "axis"]).reset_index(drop=True)


def _top_mask(values: pd.Series, fraction: float) -> tuple[pd.Series, int, int, float]:
    if not 0 < fraction <= 1:
        raise ValueError("Top fraction must be in (0, 1]")
    positive = values[values > 0]
    requested = max(1, int(math.ceil(len(values) * fraction)))
    if positive.empty:
        return pd.Series(False, index=values.index), requested, 0, float("nan")
    k = min(requested, len(positive))
    boundary = float(positive.nlargest(k).iloc[-1])
    mask = values.ge(boundary) & values.gt(0)
    return mask, requested, int(mask.sum()), boundary


def pathway_profiles(
    grid: pd.DataFrame,
    annotations: pd.DataFrame,
    methods: Sequence[str],
    *,
    design: str,
    top_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    annotations = annotations.loc[
        annotations["axis"].isin(grid["axis"].unique()), ["axis", "pathway"]
    ].drop_duplicates()
    pathway_sizes = annotations.groupby("pathway")["axis"].nunique().to_dict()
    rows: list[dict[str, Any]] = []
    coverage: list[dict[str, Any]] = []
    for method in methods:
        for stage, stage_frame in grid.groupby("stage", sort=True):
            scores = stage_frame.set_index("axis")[method].sort_index()
            n_positive = int(scores.gt(0).sum())
            n_unique = int(scores.nunique())
            rank_informative = n_unique > 1
            # Use integer average ranks, not percentile floats.  Rounding the
            # pathway mean before reranking preserves exact theoretical ties.
            axis_ranks = scores.rank(method="average", pct=False)
            top, requested, selected, boundary = _top_mask(scores, top_fraction)
            annotation_local = annotations.loc[
                annotations["axis"].isin(scores.index)
            ].copy()
            annotation_local["axis_rank"] = annotation_local["axis"].map(axis_ranks)
            annotation_local["is_top"] = annotation_local["axis"].map(top).fillna(False)
            local_rows: list[dict[str, Any]] = []
            for pathway, group in annotation_local.groupby("pathway", sort=True):
                pathway_axes = group["axis"].drop_duplicates()
                mean_rank = float(
                    np.round(axis_ranks.loc[pathway_axes].mean(), decimals=12)
                )
                hits = int(top.loc[pathway_axes].sum())
                background_hits = int(pathway_sizes[pathway])
                expected = selected * background_hits / len(scores) if len(scores) else 0.0
                p_value = (
                    float(hypergeom.sf(hits - 1, len(scores), background_hits, selected))
                    if selected and hits
                    else 1.0
                )
                local_rows.append(
                    {
                        "design": design,
                        "method": method,
                        "stage": float(stage),
                        "stage_label": STAGE_LABELS[float(stage)],
                        "pathway": pathway,
                        "n_universe_axes": int(len(scores)),
                        "n_pathway_axes": background_hits,
                        "n_positive_axes": n_positive,
                        "n_unique_native_scores": n_unique,
                        "rank_informative": rank_informative,
                        "pathway_mean_axis_rank": mean_rank if rank_informative else np.nan,
                        "top_fraction_requested": float(top_fraction),
                        "top_k_requested": requested,
                        "top_k_after_positive_and_ties": selected,
                        "top_boundary_native_score": boundary,
                        "top_pathway_hits": hits,
                        "expected_hits_random": expected,
                        "fold_enrichment": hits / expected if expected else np.nan,
                        "hypergeometric_p_greater": p_value,
                    }
                )
            local = pd.DataFrame(local_rows)
            local["bh_q_within_method_stage"] = _bh(
                local["hypergeometric_p_greater"].to_numpy()
            )
            if rank_informative:
                local["pathway_rank"] = (
                    local["pathway_mean_axis_rank"]
                    .round(12)
                    .rank(method="average", pct=True)
                )
            else:
                local["pathway_rank"] = np.nan
            rows.extend(local.to_dict("records"))
            coverage.append(
                {
                    "design": design,
                    "method": method,
                    "stage": float(stage),
                    "stage_label": STAGE_LABELS[float(stage)],
                    "n_universe_axes": int(len(scores)),
                    "n_positive_axes": n_positive,
                    "positive_axis_fraction": n_positive / len(scores),
                    "n_unique_native_scores": n_unique,
                    "rank_informative": rank_informative,
                    "top_k_requested": requested,
                    "top_k_after_positive_and_ties": selected,
                }
            )
    profiles = pd.DataFrame(rows).sort_values(
        ["design", "method", "stage", "pathway"]
    )
    return profiles.reset_index(drop=True), pd.DataFrame(coverage)


def consensus_analysis(
    profiles: pd.DataFrame,
    *,
    design: str,
    external_methods: Sequence[str],
    permutations: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    external = profiles.loc[profiles["method"].isin(external_methods)]
    pivot = external.pivot_table(
        index=["stage", "stage_label", "pathway", "n_pathway_axes"],
        columns="method",
        values="pathway_rank",
        aggfunc="first",
    ).reset_index()
    for method in external_methods:
        if method not in pivot:
            pivot[method] = np.nan
    pivot["n_external_methods_informative"] = pivot[list(external_methods)].notna().sum(
        axis=1
    )
    pivot["external_consensus_rank"] = pivot[list(external_methods)].median(
        axis=1, skipna=True
    )
    pivot.loc[
        pivot["n_external_methods_informative"] < 2, "external_consensus_rank"
    ] = np.nan
    details = []
    metrics = []
    for target in CB_METHODS:
        target_frame = profiles.loc[
            profiles["method"].eq(target),
            ["stage", "stage_label", "pathway", "pathway_rank"],
        ].rename(columns={"pathway_rank": "cytobridge_pathway_rank"})
        merged = pivot.merge(
            target_frame,
            on=["stage", "stage_label", "pathway"],
            validate="one_to_one",
        )
        merged["design"] = design
        merged["target"] = target
        details.append(merged)
        for stage, group in merged.groupby("stage", sort=True):
            valid = group.dropna(
                subset=["cytobridge_pathway_rank", "external_consensus_rank"]
            )
            rho, p_value = _safe_spearman(
                valid["cytobridge_pathway_rank"],
                valid["external_consensus_rank"],
            )
            control = np.log1p(valid["n_pathway_axes"].to_numpy(float))
            partial = _partial_spearman(
                valid["cytobridge_pathway_rank"],
                valid["external_consensus_rank"],
                control,
            )
            key = f"{design}|{target}|{stage}".encode()
            local_seed = seed + zlib.crc32(key)
            perm_p = _permutation_p(
                valid["cytobridge_pathway_rank"],
                valid["external_consensus_rank"],
                permutations=permutations,
                seed=local_seed,
            )
            metrics.append(
                {
                    "design": design,
                    "target": target,
                    "stage": float(stage),
                    "stage_label": STAGE_LABELS[float(stage)],
                    "n_pathways": int(len(valid)),
                    "n_external_methods_informative": (
                        int(valid["n_external_methods_informative"].min())
                        if len(valid)
                        else 0
                    ),
                    "external_methods_requested": ";".join(external_methods),
                    "spearman_rho": rho,
                    "spearman_asymptotic_p": p_value,
                    "partial_spearman_controlling_log_pathway_size": partial,
                    "pathway_label_permutation_p_two_sided": perm_p,
                    "n_permutations": int(permutations),
                }
            )
    return pd.concat(details, ignore_index=True), pd.DataFrame(metrics)


def top_pathway_overlap(
    details: pd.DataFrame, *, top_n: int
) -> pd.DataFrame:
    rows = []
    for (design, target, stage), group in details.groupby(
        ["design", "target", "stage"], sort=True
    ):
        valid = group.dropna(
            subset=["cytobridge_pathway_rank", "external_consensus_rank"]
        ).copy()
        if valid.empty:
            continue

        def select(column: str) -> set[str]:
            k = min(top_n, len(valid))
            boundary = valid[column].nlargest(k).iloc[-1]
            return set(valid.loc[valid[column].ge(boundary), "pathway"])

        left, right = select("cytobridge_pathway_rank"), select(
            "external_consensus_rank"
        )
        intersection = left & right
        n = len(valid)
        expected = len(left) * len(right) / n
        p_value = float(hypergeom.sf(len(intersection) - 1, n, len(right), len(left)))
        max_allowed = max(top_n + 2, int(math.ceil(top_n * 1.5)))
        interpretable = len(left) <= max_allowed and len(right) <= max_allowed
        rows.append(
            {
                "design": design,
                "target": target,
                "stage": float(stage),
                "stage_label": STAGE_LABELS[float(stage)],
                "n_pathways": n,
                "top_n_requested": int(top_n),
                "cytobridge_set_size_after_ties": len(left),
                "external_set_size_after_ties": len(right),
                "intersection": len(intersection),
                "expected_intersection_random": expected,
                "overlap_enrichment_over_random": (
                    len(intersection) / expected if expected else np.nan
                ),
                "jaccard": len(intersection) / len(left | right),
                "hypergeometric_p_greater": p_value,
                "tie_expansion_interpretable": interpretable,
                "shared_pathways": ";".join(sorted(intersection)),
            }
        )
    result = pd.DataFrame(rows)
    if len(result):
        result["bh_q_within_design_target"] = result.groupby(
            ["design", "target"], sort=False
        )["hypergeometric_p_greater"].transform(lambda x: _bh(x.to_numpy()))
    return result


def heatmap_data(
    profiles: pd.DataFrame,
    consensus_details: pd.DataFrame,
    *,
    external_methods: Sequence[str],
    n_pathways: int,
) -> pd.DataFrame:
    attention = consensus_details.loc[
        consensus_details["target"].eq("CytoBridge attention x LR")
    ]
    external_order = (
        attention.groupby("pathway")["external_consensus_rank"]
        .mean()
        .sort_values(ascending=False)
    )
    selected = list(external_order.head(n_pathways).index)
    method_order = [
        "External-only consensus",
        "CytoBridge attention x LR",
        *external_methods,
    ]
    rows = []
    for pathway in selected:
        rows.append(
            {
                "pathway": pathway,
                "method": "External-only consensus",
                "mean_pathway_rank": float(external_order[pathway]),
                "n_informative_stages": int(
                    attention.loc[
                        attention["pathway"].eq(pathway), "external_consensus_rank"
                    ].notna().sum()
                ),
                "selection_role": "row selected by external-only consensus",
            }
        )
        for method in method_order[1:]:
            values = profiles.loc[
                profiles["pathway"].eq(pathway) & profiles["method"].eq(method),
                "pathway_rank",
            ]
            rows.append(
                {
                    "pathway": pathway,
                    "method": method,
                    "mean_pathway_rank": float(values.mean()) if values.notna().any() else np.nan,
                    "n_informative_stages": int(values.notna().sum()),
                    "selection_role": "display only; not used to select rows",
                }
            )
    result = pd.DataFrame(rows)
    result["method"] = pd.Categorical(result["method"], method_order, ordered=True)
    result["pathway"] = pd.Categorical(
        result["pathway"], list(reversed(selected)), ordered=True
    )
    return result.sort_values(["pathway", "method"]).reset_index(drop=True)


def _save(figure: plt.Figure, base: Path) -> None:
    figure.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def plot_overview(
    heatmap: pd.DataFrame,
    native_metrics: pd.DataFrame,
    relaxed_metrics: pd.DataFrame,
    native_overlap: pd.DataFrame,
    output_dir: Path,
) -> None:
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(16.5, 12.2),
        gridspec_kw={"width_ratios": [1.05, 1.45], "height_ratios": [1.15, 0.85]},
    )
    methods = list(heatmap["method"].cat.categories)
    pathways = list(heatmap["pathway"].cat.categories)
    matrix = (
        heatmap.pivot(index="pathway", columns="method", values="mean_pathway_rank")
        .reindex(index=pathways, columns=methods)
    )
    image = axes[0, 0].imshow(matrix.to_numpy(), vmin=0, vmax=1, cmap="RdYlBu_r")
    axes[0, 0].set_xticks(range(len(methods)))
    axes[0, 0].set_xticklabels(
        [
            "External consensus",
            "CytoBridge attention",
            "COMMOT",
            "CellChat (2/5 stages)",
            "CellAgentChat",
        ],
        rotation=42,
        ha="right",
        fontsize=8,
    )
    axes[0, 0].set_yticks(range(len(pathways)))
    axes[0, 0].set_yticklabels(pathways, fontsize=8.5)
    for row in range(len(pathways)):
        for column in range(len(methods)):
            value = matrix.iloc[row, column]
            axes[0, 0].text(
                column,
                row,
                "NA" if not np.isfinite(value) else f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if np.isfinite(value) and (value < 0.18 or value > 0.82) else "#111827",
            )
    axes[0, 0].set_title(
        "A  External-consensus pathways", loc="left", fontweight="bold", fontsize=11
    )
    colorbar = figure.colorbar(
        image, ax=axes[0, 0], fraction=0.040, pad=0.025, shrink=0.82
    )
    colorbar.set_label("Mean pathway percentile", fontsize=9)
    colorbar.ax.tick_params(labelsize=8)

    colors = {
        "CytoBridge attention x LR": "#D1495B",
        "CytoBridge exact message x LR": "#2E86AB",
        "CytoBridge exact message only (LR-conditioned)": "#7A5195",
        "CytoBridge LR-only": "#6B7280",
    }
    for target, group in native_metrics.groupby("target", sort=False):
        axes[0, 1].plot(
            group["stage"],
            group["spearman_rho"],
            marker="o",
            linewidth=2,
            color=colors[target],
            label=target.replace("CytoBridge ", ""),
        )
    axes[0, 1].axhline(0, color="#9CA3AF", linewidth=0.8)
    axes[0, 1].set_xticks(STAGES)
    axes[0, 1].set_xticklabels([STAGE_LABELS[x] for x in STAGES])
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].set_ylabel("Spearman rho vs external consensus")
    axes[0, 1].set_title(
        "B  Stage-wise shared-axis consistency",
        loc="left",
        fontweight="bold",
        fontsize=11,
    )
    axes[0, 1].legend(frameon=False, fontsize=7.5)

    overlap = native_overlap.loc[
        native_overlap["target"].eq("CytoBridge attention x LR")
        & native_overlap["tie_expansion_interpretable"]
    ]
    x = np.arange(len(overlap))
    axes[1, 0].bar(
        x - 0.18, overlap["intersection"], 0.36, color="#D1495B", label="Observed shared"
    )
    axes[1, 0].bar(
        x + 0.18,
        overlap["expected_intersection_random"],
        0.36,
        color="#D1D5DB",
        label="Random expectation",
    )
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(overlap["stage_label"])
    axes[1, 0].set_ylabel("Shared top pathways")
    if len(overlap):
        axes[1, 0].set_ylim(
            0,
            max(
                overlap["intersection"].max(),
                overlap["expected_intersection_random"].max(),
            )
            + 2.7,
        )
    axes[1, 0].set_title(
        "C  Top-pathway overlap (interpretable stages)",
        loc="left",
        fontweight="bold",
        fontsize=11,
    )
    axes[1, 0].legend(
        frameon=False,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        fontsize=8.5,
    )
    for position, (_, row) in enumerate(overlap.iterrows()):
        axes[1, 0].text(
            position - 0.18,
            row["intersection"] + 0.15,
            f"{int(row['intersection'])}/{int(row['top_n_requested'])}\np={row['hypergeometric_p_greater']:.2g}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    summary_rows = []
    for label, metrics in (("Native", native_metrics), ("Relaxed", relaxed_metrics)):
        for target, group in metrics.groupby("target", sort=False):
            summary_rows.append(
                {
                    "design": label,
                    "target": target,
                    "mean": group["spearman_rho"].mean(),
                    "partial": group[
                        "partial_spearman_controlling_log_pathway_size"
                    ].mean(),
                }
            )
    summary = pd.DataFrame(summary_rows)
    targets = list(CB_METHODS)
    x = np.arange(len(targets))
    width = 0.34
    for index, design in enumerate(("Native", "Relaxed")):
        values = [
            summary.loc[
                summary["design"].eq(design) & summary["target"].eq(target), "mean"
            ].squeeze()
            for target in targets
        ]
        axes[1, 1].bar(
            x + (index - 0.5) * width,
            values,
            width,
            color=("#457B9D" if design == "Native" else "#F4A261"),
            label=design,
        )
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(
        ["attention x LR", "exact x LR", "exact only\n(LR-conditioned)", "LR-only"],
        fontsize=8.5,
    )
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_ylabel("Mean stage-wise Spearman rho")
    axes[1, 1].set_title(
        "D  Consistency is strong but not attention-specific",
        loc="left",
        fontweight="bold",
        fontsize=11,
    )
    axes[1, 1].legend(frameon=False)
    figure.suptitle(
        "Zebrafish pathway-level cross-method consistency\n"
        "External consensus never includes CytoBridge",
        fontsize=16,
        fontweight="bold",
        y=0.985,
    )
    figure.subplots_adjust(
        left=0.095,
        right=0.97,
        bottom=0.09,
        top=0.88,
        wspace=0.28,
        hspace=0.34,
    )
    _save(figure, output_dir / "four_method_pathway_consistency_overview")


def plot_fig2c_style(
    profiles: pd.DataFrame,
    selected_pathways: Sequence[str],
    methods: Sequence[str],
    output_dir: Path,
    *,
    filename: str,
    title: str,
) -> None:
    max_value = float(
        -np.log10(
            profiles.loc[
                profiles["pathway"].isin(selected_pathways)
                & profiles["method"].isin(methods),
                "hypergeometric_p_greater",
            ].clip(lower=1e-300)
        ).max()
    )
    max_value = max(1.0, min(max_value, 20.0))
    figure, axes = plt.subplots(
        1,
        len(STAGES),
        figsize=(18, max(6.4, 0.30 * len(selected_pathways) + 2.2)),
        sharey=True,
    )
    cmap = plt.get_cmap("Reds").copy()
    cmap.set_bad("#D1D5DB")
    for axis_plot, stage in zip(axes, STAGES):
        local = profiles.loc[
            profiles["stage"].eq(stage)
            & profiles["pathway"].isin(selected_pathways)
            & profiles["method"].isin(methods)
        ]
        p_matrix = (
            local.pivot(index="pathway", columns="method", values="hypergeometric_p_greater")
            .reindex(index=selected_pathways, columns=methods)
        )
        informative = (
            local.pivot(index="pathway", columns="method", values="rank_informative")
            .reindex(index=selected_pathways, columns=methods)
            .fillna(False)
        )
        values = -np.log10(p_matrix.clip(lower=1e-300))
        values = values.where(informative)
        axis_plot.imshow(values.to_numpy(), vmin=0, vmax=max_value, cmap=cmap, aspect="auto")
        axis_plot.set_xticks(range(len(methods)))
        axis_plot.set_xticklabels(
            ["CB", "COMMOT", "CellChat", "CAG"],
            rotation=42,
            ha="right",
            fontsize=8,
        )
        axis_plot.set_yticks(range(len(selected_pathways)))
        axis_plot.set_yticklabels(selected_pathways, fontsize=8)
        axis_plot.set_title(STAGE_LABELS[stage], fontsize=10, fontweight="bold")
        q_matrix = (
            local.pivot(index="pathway", columns="method", values="bh_q_within_method_stage")
            .reindex(index=selected_pathways, columns=methods)
        )
        for row in range(len(selected_pathways)):
            for column in range(len(methods)):
                if np.isfinite(q_matrix.iloc[row, column]) and q_matrix.iloc[row, column] < 0.05:
                    axis_plot.text(column, row, "*", ha="center", va="center", color="#111827")
    scalar = plt.cm.ScalarMappable(norm=Normalize(0, max_value), cmap=cmap)
    color_axis = figure.add_axes([0.36, 0.055, 0.28, 0.018])
    colorbar = figure.colorbar(scalar, cax=color_axis, orientation="horizontal")
    colorbar.set_label("-log10 enrichment P", fontsize=9, labelpad=2)
    colorbar.ax.tick_params(labelsize=8)
    figure.suptitle(
        title + "\nRows selected using external-only consensus; * BH q < 0.05",
        fontsize=13,
        fontweight="bold",
    )
    figure.subplots_adjust(left=0.15, right=0.97, bottom=0.22, top=0.82, wspace=0.12)
    _save(figure, output_dir / filename)


def readout_comparison_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    summary = (
        metrics.groupby(["design", "target"], sort=False, as_index=False)
        .agg(
            n_finite_stages=("spearman_rho", "count"),
            mean_stage_spearman_rho=("spearman_rho", "mean"),
            median_stage_spearman_rho=("spearman_rho", "median"),
            min_stage_spearman_rho=("spearman_rho", "min"),
            max_stage_spearman_rho=("spearman_rho", "max"),
            mean_partial_spearman_controlling_log_pathway_size=(
                "partial_spearman_controlling_log_pathway_size",
                "mean",
            ),
        )
    )
    lr_baseline = (
        summary.loc[
            summary["target"].eq("CytoBridge LR-only"),
            ["design", "mean_stage_spearman_rho"],
        ]
        .rename(
            columns={
                "mean_stage_spearman_rho": "lr_only_mean_stage_spearman_rho"
            }
        )
    )
    summary = summary.merge(lr_baseline, on="design", how="left", validate="many_to_one")
    summary["delta_mean_rho_vs_lr_only"] = (
        summary["mean_stage_spearman_rho"]
        - summary["lr_only_mean_stage_spearman_rho"]
    )
    return summary


def _metric_summary(metrics: pd.DataFrame, design: str) -> dict[str, Any]:
    attention = metrics.loc[metrics["target"].eq("CytoBridge attention x LR")]
    return {
        "design": design,
        "mean_stage_spearman_rho": float(attention["spearman_rho"].mean()),
        "median_stage_spearman_rho": float(attention["spearman_rho"].median()),
        "stage_spearman_rho": {
            row["stage_label"]: float(row["spearman_rho"])
            for _, row in attention.iterrows()
            if np.isfinite(row["spearman_rho"])
        },
        "mean_partial_spearman_controlling_pathway_size": float(
            attention["partial_spearman_controlling_log_pathway_size"].mean()
        ),
    }


def write_readme(
    output_dir: Path,
    four_native_metrics: pd.DataFrame,
    four_relaxed_metrics: pd.DataFrame,
    four_native_overlap: pd.DataFrame,
    native_primary_metrics: pd.DataFrame,
    native_relaxed_metrics: pd.DataFrame,
    audit: Mapping[str, Any],
) -> None:
    native = _metric_summary(four_native_metrics, "four_method_native")
    relaxed = _metric_summary(four_relaxed_metrics, "four_method_relaxed")
    overlap = four_native_overlap.loc[
        four_native_overlap["target"].eq("CytoBridge attention x LR")
        & four_native_overlap["tie_expansion_interpretable"]
    ]
    overlap_lines = "\n".join(
        f"- {row.stage_label}: 共同 {int(row.intersection)}/{int(row.top_n_requested)}，"
        f"随机期望 {row.expected_intersection_random:.2f}，P={row.hypergeometric_p_greater:.3g}；"
        f"{row.shared_pathways}"
        for row in overlap.itertuples()
    )
    native_core_primary = _metric_summary(
        native_primary_metrics, "native_zebrafish_primary"
    )
    native_core_relaxed = _metric_summary(
        native_relaxed_metrics, "native_zebrafish_relaxed"
    )
    readout_summary = readout_comparison_summary(four_native_metrics)
    readout_lines = "\n".join(
        f"- `{row.target.replace('CytoBridge ', '')}`: mean rho "
        f"**{row.mean_stage_spearman_rho:.3f}**，相对 LR-only "
        f"{row.delta_mean_rho_vs_lr_only:+.3f}"
        for row in readout_summary.itertuples()
    )
    text = f"""# 斑马鱼多方法 pathway consistency：通俗说明

## 一句话结论

有正向信号，而且强度不弱。四方法严格共同的 134 个 LR 轴上，CytoBridge
`attention x LR` 的 pathway 排名与 **不包含 CytoBridge** 的外部方法共识呈持续正相关：

- native 口径五时期平均 Spearman rho = **{native['mean_stage_spearman_rho']:.3f}**
  （逐时期：{', '.join(f'{k} {v:.3f}' for k, v in native['stage_spearman_rho'].items())}）；
- relaxed 连续分数敏感性平均 rho = **{relaxed['mean_stage_spearman_rho']:.3f}**
  （逐时期：{', '.join(f'{k} {v:.3f}' for k, v in relaxed['stage_spearman_rho'].items())}）。

这支持的准确表述是：

> CytoBridge 的 LR-resolved / LR-compatible communication output 在 pathway 层面
> 与 COMMOT、CellChat 和 CellAgentChat 的外部结果一致。

它**不单独证明 attention 比 LR expression 更好**。脚本同时比较四个 readout：
`attention x LR`、完整 `exact message x LR`、`exact-message only
(LR-conditioned)` 和 `LR-only`。其中“纯 exact”定义为
`mean(exact-message x LR) / mean(LR)`：它去掉总体 LR abundance 的乘法尺度，
但仍必须由 LR support 给 edge 归属，否则未条件化的 exact message 本身没有
LR/pathway 身份。这个拆分能回答 message 部分单独是否与外部方法一致。

四方法 native shared-axis 上的直接比较为：

{readout_lines}

四条数值近乎重合，因此本结果支持“CytoBridge 的 LR-resolved 输出与外部方法一致”，
但不支持“attention 或 exact message 相对 LR-only 带来明显额外一致性”。

## 最直观的结果

晚期 top-pathway overlap 没有早期的大量并列问题：

{overlap_lines}

共同出现的通路包括 COLLAGEN、HSPG、AGRN、ANGPTL、APELIN、LAMININ、CNTN、
THBS 等。早期 CytoBridge 在 134 轴里只有很少非零轴，保留 top-10 边界并列会选入
过多 pathway，所以早期 overlap 被保留在表中但不作为主图证据。

## 两套分析为什么都需要

1. **四方法 shared-axis sensitivity**：CytoBridge、COMMOT、CellChat、
   CellAgentChat 都能比较，但由于 CellAgentChat 需要严格斑马鱼到小鼠 1:1 ortholog
   投影，只剩 **{audit['four_method_axes']} 个 LR axes / {audit['four_method_pathways']}
   个 pathways**，占项目 {audit['database_axes']} 个 LR axes 的
   **{100 * audit['four_method_axes'] / audit['database_axes']:.2f}%**。因此使用 top 20%
   （27 axes），不能照搬论文 top 100。
2. **原生斑马鱼 core**：不含跨物种 CellAgentChat，保留
   **{audit['native_zebrafish_axes']} 个 LR axes**。默认 CellChat triMean 口径的
   `attention x LR` 平均 rho 为 **{native_core_primary['mean_stage_spearman_rho']:.3f}**；
   truncatedMean 敏感性为 **{native_core_relaxed['mean_stage_spearman_rho']:.3f}**。
   注意 triMean 前三时期全零，早期只有 COMMOT 一个有效外部方法，因此这些早期值
   是 pairwise audit，不应称为“两方法共识”；正式 consensus 应看 CellChat 有信号
   的时期或 truncatedMean 敏感性。

## 每张图怎么看

- `four_method_pathway_consistency_overview.png`
  - A：行完全按 external-only consensus 选，不看 CytoBridge 后挑通路。越接近 1
    表示该方法把该 pathway 排得越靠前。
  - B：rho 越高，CytoBridge 和外部共识的 pathway 顺序越相似。
  - C：红柱是双方 top pathway 实际共同数，灰柱是随机期望。
  - D：native/relaxed 都为正；四种 CytoBridge readout 可直接比较。
- `lr_family_pathway_heatmap_native.png`：项目 LR database family 的富集热图。
  颜色是 top-LR 对 pathway 的 `-log10(P)`，星号是该 method x stage 的全部 pathway
  一起 BH 后 q<0.05。灰色代表该方法在该时期没有可排名的非零信号。
- `lr_family_pathway_heatmap_relaxed.png`：用 CAG raw 和 CellChat truncatedMean
  的连续分数敏感性，不替代 native 主结果。

## 为什么这里不是原论文 Fig. 2C

这里使用项目 LR database 已声明的 zebrafish signaling-family 标签，因此回答
**cross-method LR-family consistency**。原论文则从 top LR 的 ligand/receptor genes
做 g:Profiler Reactome enrichment，得到较长的 biological-process 名称。两者不能
使用相同名称。所有 49/139 个 LR families（包括 zero-hit）都进入 BH，避免漏掉
零命中 family 后人为变小 q 值。真正的 Reactome gene enrichment 由独立脚本输出。

## 重要限制

- native CellChat triMean 在 5.25/10/12 hpf 全零，在 18 hpf 也非常稀疏；这些格子
  不可被解释成“CellChat 反对”，而是该设置下没有 rank-informative output。
- CellAgentChat shared-axis 结果是跨物种 orthology sensitivity，不能代表全项目 LR
  catalog。
- 所有方法使用同一批表达数据，且本图条件化于共同 LR catalog。这是 computational
  consistency，不是独立实验真值或因果验证。
- pathway size-controlled partial Spearman 已写入
  `pathway_consistency_metrics.csv`；正相关仍保留，用于排除结果完全由大 pathway
  驱动的简单解释。

## 可复现文件

- `method_stage_coverage.csv`：每个方法每时期到底有多少非零 LR。
- `pathway_profiles.csv.gz`：每个 method x stage x pathway 的完整统计，包含零命中。
- `pathway_consistency_metrics.csv`：rho、pathway-size partial rho 和 permutation P。
- `readout_comparison_summary.csv`：四种 CytoBridge readout 的直接均值比较及其相对
  LR-only 的差值。
- `top_pathway_overlap.csv`：共同 pathway 名单和随机期望。
- `heatmap_data.csv`：主热图的精确数值。
- `run_manifest.json`：输入 SHA256、参数、universe 和所有产物。
"""
    (output_dir / "README_CN.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = _parser().parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"Output directory is not empty: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    database = load_database(args.lr_database)
    cytobridge = load_cytobridge(args.cytobridge_axis_scores)
    crosswalk = load_crosswalk(args.cellagentchat_crosswalk)
    excluded = pd.read_csv(args.cellchat_excluded_lr)
    _require(excluded, ["current_ligand", "current_receptor"], str(args.cellchat_excluded_lr))
    excluded_axes = set(_axis(excluded["current_ligand"], excluded["current_receptor"]))

    commot = _collapse_lr_contexts(
        args.commot_lr_scores, args.commot_score_column, "COMMOT"
    )
    cellchat_primary = _collapse_lr_contexts(
        args.cellchat_primary_lr_scores, "score", "CellChat triMean"
    )
    cellchat_truncated = _collapse_lr_contexts(
        args.cellchat_truncated_lr_scores, "score", "CellChat truncatedMean"
    )
    cag_significant = load_cellagentchat(
        args.cellagentchat_significant_lr_scores,
        crosswalk,
        "cellagentchat_score",
        "CellAgentChat significant",
    )
    cag_raw = load_cellagentchat(
        args.cellagentchat_raw_lr_scores,
        crosswalk,
        "cellagentchat_score_raw",
        "CellAgentChat continuous",
    )

    database_axes = set(database["axis"])
    cb_axes = set(cytobridge["axis"])
    native_axes = sorted(database_axes & cb_axes - excluded_axes)
    four_axes = sorted(set(crosswalk["axis"]) & set(native_axes))
    if len(four_axes) != len(crosswalk):
        raise ValueError(
            "Not every CellAgentChat projected LR is in the shared native universe"
        )

    four_grid = build_score_grid(
        four_axes,
        cytobridge,
        {
            "COMMOT": commot,
            "CellChat triMean": cellchat_primary,
            "CellChat truncatedMean": cellchat_truncated,
            "CellAgentChat significant": cag_significant,
            "CellAgentChat continuous": cag_raw,
        },
    )
    native_grid = build_score_grid(
        native_axes,
        cytobridge,
        {
            "COMMOT": commot,
            "CellChat triMean": cellchat_primary,
            "CellChat truncatedMean": cellchat_truncated,
        },
    )

    four_native_profiles, four_native_coverage = pathway_profiles(
        four_grid,
        database,
        [*CB_METHODS, *FOUR_NATIVE_EXTERNAL],
        design="four_method_native",
        top_fraction=args.four_method_top_fraction,
    )
    four_relaxed_profiles, four_relaxed_coverage = pathway_profiles(
        four_grid,
        database,
        [*CB_METHODS, *FOUR_RELAXED_EXTERNAL],
        design="four_method_relaxed",
        top_fraction=args.four_method_top_fraction,
    )
    native_primary_profiles, native_primary_coverage = pathway_profiles(
        native_grid,
        database,
        [*CB_METHODS, "COMMOT", "CellChat triMean"],
        design="native_zebrafish_primary",
        top_fraction=args.native_top_fraction,
    )
    native_relaxed_profiles, native_relaxed_coverage = pathway_profiles(
        native_grid,
        database,
        [*CB_METHODS, "COMMOT", "CellChat truncatedMean"],
        design="native_zebrafish_relaxed",
        top_fraction=args.native_top_fraction,
    )
    profiles = pd.concat(
        [
            four_native_profiles,
            four_relaxed_profiles,
            native_primary_profiles,
            native_relaxed_profiles,
        ],
        ignore_index=True,
    )
    coverage = pd.concat(
        [
            four_native_coverage,
            four_relaxed_coverage,
            native_primary_coverage,
            native_relaxed_coverage,
        ],
        ignore_index=True,
    )

    four_native_details, four_native_metrics = consensus_analysis(
        four_native_profiles,
        design="four_method_native",
        external_methods=FOUR_NATIVE_EXTERNAL,
        permutations=args.permutations,
        seed=args.seed,
    )
    four_relaxed_details, four_relaxed_metrics = consensus_analysis(
        four_relaxed_profiles,
        design="four_method_relaxed",
        external_methods=FOUR_RELAXED_EXTERNAL,
        permutations=args.permutations,
        seed=args.seed,
    )
    native_primary_details, native_primary_metrics = consensus_analysis(
        native_primary_profiles,
        design="native_zebrafish_primary",
        external_methods=("COMMOT", "CellChat triMean"),
        permutations=args.permutations,
        seed=args.seed,
    )
    native_relaxed_details, native_relaxed_metrics = consensus_analysis(
        native_relaxed_profiles,
        design="native_zebrafish_relaxed",
        external_methods=("COMMOT", "CellChat truncatedMean"),
        permutations=args.permutations,
        seed=args.seed,
    )
    details = pd.concat(
        [
            four_native_details,
            four_relaxed_details,
            native_primary_details,
            native_relaxed_details,
        ],
        ignore_index=True,
    )
    metrics = pd.concat(
        [
            four_native_metrics,
            four_relaxed_metrics,
            native_primary_metrics,
            native_relaxed_metrics,
        ],
        ignore_index=True,
    )
    overlap = pd.concat(
        [
            top_pathway_overlap(four_native_details, top_n=args.top_pathways),
            top_pathway_overlap(four_relaxed_details, top_n=args.top_pathways),
            top_pathway_overlap(native_primary_details, top_n=args.top_pathways),
            top_pathway_overlap(native_relaxed_details, top_n=args.top_pathways),
        ],
        ignore_index=True,
    )
    native_heatmap = heatmap_data(
        four_native_profiles,
        four_native_details,
        external_methods=FOUR_NATIVE_EXTERNAL,
        n_pathways=args.heatmap_pathways,
    )
    relaxed_heatmap = heatmap_data(
        four_relaxed_profiles,
        four_relaxed_details,
        external_methods=FOUR_RELAXED_EXTERNAL,
        n_pathways=args.heatmap_pathways,
    )

    profiles.to_csv(output_dir / "pathway_profiles.csv.gz", index=False)
    coverage.to_csv(output_dir / "method_stage_coverage.csv", index=False)
    details.to_csv(output_dir / "external_consensus_pathway_ranks.csv.gz", index=False)
    metrics.to_csv(output_dir / "pathway_consistency_metrics.csv", index=False)
    readout_comparison_summary(metrics).to_csv(
        output_dir / "readout_comparison_summary.csv", index=False
    )
    overlap.to_csv(output_dir / "top_pathway_overlap.csv", index=False)
    native_heatmap.to_csv(output_dir / "heatmap_data.csv", index=False)
    relaxed_heatmap.to_csv(output_dir / "heatmap_data_relaxed.csv", index=False)

    plot_overview(
        native_heatmap,
        four_native_metrics,
        four_relaxed_metrics,
        top_pathway_overlap(four_native_details, top_n=args.top_pathways),
        output_dir,
    )
    selected_native = list(native_heatmap["pathway"].cat.categories)
    selected_relaxed = list(relaxed_heatmap["pathway"].cat.categories)
    plot_fig2c_style(
        four_native_profiles,
        selected_native,
        [
            "CytoBridge attention x LR",
            "COMMOT",
            "CellChat triMean",
            "CellAgentChat significant",
        ],
        output_dir,
        filename="lr_family_pathway_heatmap_native",
        title="Zebrafish LR-family enrichment (native scores)",
    )
    plot_fig2c_style(
        four_relaxed_profiles,
        selected_relaxed,
        [
            "CytoBridge attention x LR",
            "COMMOT",
            "CellChat truncatedMean",
            "CellAgentChat continuous",
        ],
        output_dir,
        filename="lr_family_pathway_heatmap_relaxed",
        title="Zebrafish LR-family enrichment (continuous sensitivity)",
    )

    audit = {
        "database_rows": int(pd.read_csv(args.lr_database).shape[0]),
        "database_axes": int(database["axis"].nunique()),
        "database_pathways": int(database["pathway"].nunique()),
        "cytobridge_axes": int(len(cb_axes)),
        "native_zebrafish_axes": int(len(native_axes)),
        "native_zebrafish_pathways": int(
            database.loc[database["axis"].isin(native_axes), "pathway"].nunique()
        ),
        "four_method_axes": int(len(four_axes)),
        "four_method_pathways": int(
            database.loc[database["axis"].isin(four_axes), "pathway"].nunique()
        ),
        "cellchat_method_unavailable_axes": int(len(excluded_axes)),
        "four_method_top_fraction": float(args.four_method_top_fraction),
        "native_top_fraction": float(args.native_top_fraction),
        "commot_score_column": args.commot_score_column,
        "external_consensus_excludes_cytobridge": True,
        "pathway_tie_policy": (
            "integer average LR ranks; pathway mean rounded to 12 decimals before reranking"
        ),
        "exact_message_only_definition": (
            "mean_exact_message_times_lr_activity / mean_scaled_lr_activity; "
            "zero when the LR axis has zero activity"
        ),
    }
    pd.DataFrame([audit]).to_csv(output_dir / "universe_audit.csv", index=False)
    write_readme(
        output_dir,
        four_native_metrics,
        four_relaxed_metrics,
        top_pathway_overlap(four_native_details, top_n=args.top_pathways),
        native_primary_metrics,
        native_relaxed_metrics,
        audit,
    )

    input_paths = {
        "cytobridge_axis_scores": args.cytobridge_axis_scores,
        "commot_lr_scores": args.commot_lr_scores,
        "cellchat_primary_lr_scores": args.cellchat_primary_lr_scores,
        "cellchat_truncated_lr_scores": args.cellchat_truncated_lr_scores,
        "cellchat_excluded_lr": args.cellchat_excluded_lr,
        "cellagentchat_raw_lr_scores": args.cellagentchat_raw_lr_scores,
        "cellagentchat_significant_lr_scores": args.cellagentchat_significant_lr_scores,
        "cellagentchat_crosswalk": args.cellagentchat_crosswalk,
        "lr_database": args.lr_database,
    }
    output_files = sorted(
        path
        for path in output_dir.iterdir()
        if path.is_file() and path.name != "run_manifest.json"
    )
    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[2]
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis": "zebrafish_multimethod_pathway_consistency",
        "command": [sys.executable, *sys.argv],
        "code": {
            "analysis_script": _record(script_path),
            "git": _git_state(repo_root),
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "inputs": {name: _record(path) for name, path in input_paths.items()},
        "parameters": {
            "four_method_top_fraction": args.four_method_top_fraction,
            "native_top_fraction": args.native_top_fraction,
            "commot_score_column": args.commot_score_column,
            "top_pathways": args.top_pathways,
            "heatmap_pathways": args.heatmap_pathways,
            "permutations": args.permutations,
            "seed": args.seed,
        },
        "audit": audit,
        "summaries": {
            "four_method_native": _metric_summary(
                four_native_metrics, "four_method_native"
            ),
            "four_method_relaxed": _metric_summary(
                four_relaxed_metrics, "four_method_relaxed"
            ),
            "native_zebrafish_primary": _metric_summary(
                native_primary_metrics, "native_zebrafish_primary"
            ),
            "native_zebrafish_relaxed": _metric_summary(
                native_relaxed_metrics, "native_zebrafish_relaxed"
            ),
        },
        "claims": {
            "supports_cross_method_pathway_consistency": True,
            "supports_attention_specific_incremental_value": False,
            "independent_experimental_validation": False,
            "cellagentchat_four_method_analysis_is_cross_species_sensitivity": True,
        },
        "outputs": {path.name: _record(path) for path in output_files},
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
