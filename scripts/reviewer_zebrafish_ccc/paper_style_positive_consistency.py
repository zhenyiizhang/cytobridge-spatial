#!/usr/bin/env python3
"""Build CellAgentChat-Fig.-2-style positive-consistency analyses.

The primary consensus excludes CytoBridge.  A self-included all-method
ensemble is emitted only to mirror the CellAgentChat Fig. 2A benchmark design.
All cross-method aggregation uses within-stage percentile ranks; native score
units are never averaged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import hypergeom, pearsonr, spearmanr


KEYS = ["stage", "sender_type", "receiver_type"]
STAGE_LABELS = {0.0: "5.25 hpf", 1.0: "10 hpf", 2.0: "12 hpf", 3.0: "18 hpf", 4.0: "24 hpf"}
NATIVE_METHODS = {
    "CytoBridge attention": "cytobridge_attention",
    "COMMOT": "commot",
    "CellAgentChat CTPS": "cellagentchat_ctps",
    "CellChat triMean": "cellchat_trimean",
}
RELAXED_METHODS = {
    "CytoBridge attention": "cytobridge_attention",
    "COMMOT": "commot",
    "CellAgentChat continuous": "cellagentchat_continuous",
    "CellChat truncatedMean": "cellchat_truncatedmean",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-dir", required=True, type=Path)
    parser.add_argument("--cytobridge-dir", required=True, type=Path)
    parser.add_argument("--cellagentchat-project-dir", required=True, type=Path)
    parser.add_argument("--cellchat-truncated-dir", required=True, type=Path)
    parser.add_argument("--validation-dir", required=True, type=Path)
    parser.add_argument("--nichenet-custom-dir", required=True, type=Path)
    parser.add_argument("--shared-input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--top-fraction", type=float, default=0.20)
    parser.add_argument("--top-axis-rank", type=int, default=20)
    parser.add_argument("--nichenet-top-ligands", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "bytes": int(path.stat().st_size), "sha256": _sha256(path)}


def _prepare_output(path: Path, overwrite: bool) -> Path:
    path = path.expanduser().resolve()
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory is not empty: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _require(frame: pd.DataFrame, columns: Iterable[str], path: Path) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{path} lacks required columns: {missing}")


def _correlation(left: Sequence[float], right: Sequence[float], kind: str) -> float:
    x, y = np.asarray(left, float), np.asarray(right, float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if len(x) < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return float("nan")
    return float((spearmanr if kind == "spearman" else pearsonr)(x, y).statistic)


def _bh(values: Sequence[float]) -> np.ndarray:
    p = np.asarray(values, float)
    result = np.full(len(p), np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    if not len(valid):
        return result
    order = valid[np.argsort(p[valid])]
    adjusted = p[order] * len(valid) / np.arange(1, len(valid) + 1)
    result[order] = np.clip(np.minimum.accumulate(adjusted[::-1])[::-1], 0, 1)
    return result


def _canonical_view(frame: pd.DataFrame, view_id: str, name: str) -> pd.DataFrame:
    selected = frame.loc[frame["view_id"].eq(view_id), KEYS + ["native_score"]].copy()
    if selected.empty or selected.duplicated(KEYS).any():
        raise ValueError(f"Canonical view {view_id!r} is missing or duplicated")
    return selected.rename(columns={"native_score": name})


def load_scores(
    comparison_dir: Path,
    cellagentchat_project_dir: Path,
    cellchat_truncated_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    canonical_path = comparison_dir / "canonical_type_pair_scores.csv.gz"
    canonical = pd.read_csv(canonical_path)
    _require(canonical, ["view_id", "native_score", *KEYS], canonical_path)
    views = [
        ("cytobridge__trained__attention", "cytobridge_attention"),
        ("cytobridge__trained__exact_message", "cytobridge_exact_message"),
        ("commot__project_lr", "commot"),
        ("cellchat__project_lr", "cellchat_trimean"),
    ]
    scores = _canonical_view(canonical, *views[0])
    for view_id, name in views[1:]:
        scores = scores.merge(
            _canonical_view(canonical, view_id, name), on=KEYS, validate="one_to_one"
        )

    cag_path = cellagentchat_project_dir / "cellagentchat_type_pair_scores.csv"
    cag = pd.read_csv(cag_path)
    cag_columns = [
        *KEYS,
        "cellagentchat_significant_score_sum_mean",
        "cellagentchat_raw_score_sum_mean",
        "cellagentchat_native_primary_mean",
    ]
    _require(cag, cag_columns, cag_path)
    cag = cag[cag_columns].rename(
        columns={
            "cellagentchat_significant_score_sum_mean": "cellagentchat_ctps",
            "cellagentchat_raw_score_sum_mean": "cellagentchat_continuous",
            "cellagentchat_native_primary_mean": "legacy_significant_lr_count",
        }
    )
    scores = scores.merge(cag, on=KEYS, validate="one_to_one")

    cellchat_path = cellchat_truncated_dir / "cellchat_type_pair_scores.csv.gz"
    cellchat = pd.read_csv(cellchat_path)
    _require(cellchat, [*KEYS, "score"], cellchat_path)
    scores = scores.merge(
        cellchat[KEYS + ["score"]].rename(columns={"score": "cellchat_truncatedmean"}),
        on=KEYS,
        validate="one_to_one",
    )
    if len(scores) != 776 or scores.duplicated(KEYS).any():
        raise ValueError("Harmonized scores are not the expected five-stage type-pair grid")
    columns = [
        "cytobridge_attention",
        "cytobridge_exact_message",
        "commot",
        "cellagentchat_ctps",
        "cellagentchat_continuous",
        "cellchat_trimean",
        "cellchat_truncatedmean",
    ]
    for column in columns:
        scores[column] = pd.to_numeric(scores[column], errors="raise")
        scores[f"{column}_rank"] = scores.groupby("stage")[column].rank(
            method="average", pct=True
        )
    scores["external_native_consensus"] = scores[
        ["commot_rank", "cellagentchat_ctps_rank", "cellchat_trimean_rank"]
    ].mean(axis=1)
    scores["external_threshold_relaxed_consensus"] = scores[
        ["commot_rank", "cellagentchat_continuous_rank", "cellchat_truncatedmean_rank"]
    ].mean(axis=1)
    provenance = {
        "canonical_scores": _record(canonical_path),
        "cellagentchat_scores": _record(cag_path),
        "cellagentchat_manifest": _record(cellagentchat_project_dir / "manifest.json"),
        "cellchat_truncated_scores": _record(cellchat_path),
        "cellchat_truncated_manifest": _record(cellchat_truncated_dir / "manifest.json"),
    }
    return scores.sort_values(KEYS).reset_index(drop=True), provenance


def consensus_metrics(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []

    def add(
        design: str,
        target: str,
        column: str,
        consensus: pd.Series,
        group: pd.DataFrame,
        includes_target: bool,
        components: Sequence[str],
    ) -> None:
        local = consensus.loc[group.index]
        rows.append(
            {
                "design": design,
                "target": target,
                "stage": float(group["stage"].iloc[0]),
                "stage_label": STAGE_LABELS[float(group["stage"].iloc[0])],
                "n_directed_pairs": int(len(group)),
                "spearman": _correlation(group[column], local, "spearman"),
                "pearson_on_percentile_ranks": _correlation(
                    group[f"{column}_rank"], local, "pearson"
                ),
                "consensus_includes_target": bool(includes_target),
                "consensus_components": ";".join(components),
            }
        )

    for _, group in scores.groupby("stage", sort=True):
        for target, column in (
            ("CytoBridge attention", "cytobridge_attention"),
            ("CytoBridge exact message", "cytobridge_exact_message"),
        ):
            add(
                "external_only_native_primary",
                target,
                column,
                scores["external_native_consensus"],
                group,
                False,
                ("COMMOT", "CellAgentChat CTPS", "CellChat triMean"),
            )
            add(
                "external_only_threshold_relaxed_sensitivity",
                target,
                column,
                scores["external_threshold_relaxed_consensus"],
                group,
                False,
                ("COMMOT", "CellAgentChat continuous", "CellChat truncatedMean"),
            )
        for design, methods in (
            ("article_style_all_method_native_primary", NATIVE_METHODS),
            ("article_style_all_method_threshold_relaxed_sensitivity", RELAXED_METHODS),
        ):
            ensemble = scores[[f"{column}_rank" for column in methods.values()]].mean(axis=1)
            for target, column in methods.items():
                add(design, target, column, ensemble, group, True, tuple(methods))
                others = {name: value for name, value in methods.items() if name != target}
                loo = scores[[f"{value}_rank" for value in others.values()]].mean(axis=1)
                add(
                    design.replace("article_style_all_method", "leave_one_method_out"),
                    target,
                    column,
                    loo,
                    group,
                    False,
                    tuple(others),
                )
    by_stage = pd.DataFrame(rows)
    summary = (
        by_stage.groupby(
            ["design", "target", "consensus_includes_target", "consensus_components"],
            sort=False,
            dropna=False,
        )
        .agg(
            n_stages=("stage", "nunique"),
            n_finite_stages=("spearman", lambda x: int(pd.Series(x).notna().sum())),
            mean_stage_spearman=("spearman", "mean"),
            median_stage_spearman=("spearman", "median"),
            mean_stage_pearson_on_percentile_ranks=("pearson_on_percentile_ranks", "mean"),
        )
        .reset_index()
    )
    return by_stage, summary


def pairwise_metrics(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    references = {
        "COMMOT": "commot",
        "CellAgentChat CTPS": "cellagentchat_ctps",
        "CellAgentChat continuous sensitivity": "cellagentchat_continuous",
        "CellChat triMean": "cellchat_trimean",
        "CellChat truncatedMean sensitivity": "cellchat_truncatedmean",
    }
    rows = []
    for stage, group in scores.groupby("stage", sort=True):
        for target, column in (
            ("CytoBridge attention", "cytobridge_attention"),
            ("CytoBridge exact message", "cytobridge_exact_message"),
        ):
            for reference, reference_column in references.items():
                rows.append(
                    {
                        "target": target,
                        "reference": reference,
                        "stage": float(stage),
                        "stage_label": STAGE_LABELS[float(stage)],
                        "n_directed_pairs": int(len(group)),
                        "spearman": _correlation(group[column], group[reference_column], "spearman"),
                    }
                )
    by_stage = pd.DataFrame(rows)
    summary = (
        by_stage.groupby(["target", "reference"], sort=False)
        .agg(
            n_stages=("stage", "nunique"),
            n_finite_stages=("spearman", lambda x: int(pd.Series(x).notna().sum())),
            mean_stage_spearman=("spearman", "mean"),
            median_stage_spearman=("spearman", "median"),
        )
        .reset_index()
    )
    return by_stage, summary


def _top_set(frame: pd.DataFrame, column: str, requested: int) -> set[int]:
    values = pd.to_numeric(frame[column], errors="coerce")
    finite = values[np.isfinite(values)]
    if finite.empty or requested <= 0:
        return set()
    boundary = finite.nlargest(min(requested, len(finite))).iloc[-1]
    return set(frame.index[np.isfinite(values) & (values >= boundary)])


def top_overlap(
    scores: pd.DataFrame, top_fraction: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    references = {
        "External native consensus": "external_native_consensus",
        "External threshold-relaxed consensus": "external_threshold_relaxed_consensus",
        "COMMOT": "commot",
        "CellAgentChat CTPS": "cellagentchat_ctps",
        "CellAgentChat continuous": "cellagentchat_continuous",
        "CellChat triMean": "cellchat_trimean",
        "CellChat truncatedMean": "cellchat_truncatedmean",
    }
    rows = []
    for stage, group in scores.groupby("stage", sort=True):
        n = int(len(group))
        requested = max(1, int(round(n * top_fraction)))
        for target, column in (
            ("CytoBridge attention", "cytobridge_attention"),
            ("CytoBridge exact message", "cytobridge_exact_message"),
        ):
            left = _top_set(group, column, requested)
            for reference, reference_column in references.items():
                right = _top_set(group, reference_column, requested)
                intersection, union = len(left & right), len(left | right)
                expected = len(left) * len(right) / n if n else float("nan")
                rows.append(
                    {
                        "target": target,
                        "reference": reference,
                        "stage": float(stage),
                        "stage_label": STAGE_LABELS[float(stage)],
                        "n_directed_pairs": n,
                        "top_fraction_requested": float(top_fraction),
                        "top_k_requested": requested,
                        "target_set_size_after_boundary_ties": len(left),
                        "reference_set_size_after_boundary_ties": len(right),
                        "intersection": intersection,
                        "jaccard": intersection / union if union else float("nan"),
                        "overlap_fraction_of_smaller_set": (
                            intersection / min(len(left), len(right))
                            if left and right
                            else float("nan")
                        ),
                        "expected_intersection_under_random_sets": expected,
                        "overlap_enrichment_over_random": (
                            intersection / expected if expected else float("nan")
                        ),
                        "hypergeometric_p_greater": (
                            float(hypergeom.sf(intersection - 1, n, len(right), len(left)))
                            if left and right
                            else float("nan")
                        ),
                        "tie_policy": "include_all_scores_at_kth_boundary",
                    }
                )
    by_stage = pd.DataFrame(rows)
    by_stage["bh_q_within_target_reference_family"] = by_stage.groupby(
        ["target", "reference"], sort=False
    )["hypergeometric_p_greater"].transform(lambda x: _bh(x.to_numpy()))
    summary = (
        by_stage.groupby(["target", "reference"], sort=False)
        .agg(
            n_stages=("stage", "nunique"),
            mean_overlap_fraction=("overlap_fraction_of_smaller_set", "mean"),
            mean_jaccard=("jaccard", "mean"),
            mean_overlap_enrichment=("overlap_enrichment_over_random", "mean"),
            n_stages_p_lt_0p05=(
                "hypergeometric_p_greater",
                lambda x: int((pd.Series(x) < 0.05).sum()),
            ),
            n_stages_bh_q_lt_0p05=(
                "bh_q_within_target_reference_family",
                lambda x: int((pd.Series(x) < 0.05).sum()),
            ),
        )
        .reset_index()
    )
    return by_stage, summary


def spatial_metrics(
    scores: pd.DataFrame, cytobridge_dir: Path, validation_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    distance_path = cytobridge_dir / "type_pair_summary.csv"
    distance = pd.read_csv(distance_path)
    _require(distance, [*KEYS, "spatial_distance_mean_mean"], distance_path)
    merged = scores.merge(
        distance[KEYS + ["spatial_distance_mean_mean"]],
        on=KEYS,
        how="left",
        validate="one_to_one",
    )
    methods = {
        "CytoBridge attention": "cytobridge_attention",
        "CytoBridge exact message": "cytobridge_exact_message",
        "COMMOT": "commot",
        "CellAgentChat CTPS": "cellagentchat_ctps",
        "CellAgentChat continuous": "cellagentchat_continuous",
        "CellChat triMean": "cellchat_trimean",
        "CellChat truncatedMean": "cellchat_truncatedmean",
    }
    rows = []
    for stage, group in merged.groupby("stage", sort=True):
        for method, column in methods.items():
            complete = group[[column, "spatial_distance_mean_mean"]].dropna()
            rows.append(
                {
                    "method": method,
                    "stage": float(stage),
                    "stage_label": STAGE_LABELS[float(stage)],
                    "n_type_pairs_with_observed_graph_distance": int(len(complete)),
                    "spearman_score_vs_inverse_mean_spatial_distance": _correlation(
                        complete[column], -complete["spatial_distance_mean_mean"], "spearman"
                    ),
                    "interpretation": (
                        "paper-style proximity localization only; not LR-specificity evidence"
                    ),
                }
            )
    by_stage = pd.DataFrame(rows)
    summary = (
        by_stage.groupby("method", sort=False)
        .agg(
            n_finite_stages=(
                "spearman_score_vs_inverse_mean_spatial_distance",
                lambda x: int(pd.Series(x).notna().sum()),
            ),
            mean_stage_spearman_vs_inverse_distance=(
                "spearman_score_vs_inverse_mean_spatial_distance",
                "mean",
            ),
            median_stage_spearman_vs_inverse_distance=(
                "spearman_score_vs_inverse_mean_spatial_distance",
                "median",
            ),
        )
        .reset_index()
    )
    conditional_path = validation_dir / "degree_matched_conditional_tests.csv"
    conditional = pd.read_csv(conditional_path)
    _require(
        conditional,
        [
            "target",
            "score",
            "conditional_rank_correlation",
            "observed_minus_null_mean",
            "empirical_p_greater",
            "bh_q_within_degree_matched_family",
        ],
        conditional_path,
    )
    conditional = conditional.copy()
    conditional["interpretation"] = (
        "LR consistency after matching stage, type pair, distance, state, and graph degree"
    )
    return by_stage, summary, conditional, {
        "type_pair_distance_summary": _record(distance_path),
        "degree_matched_conditional_tests": _record(conditional_path),
    }


def pathway_enrichment(
    validation_dir: Path, shared_input_dir: Path, top_axis_rank: int
) -> tuple[pd.DataFrame, dict[str, Any]]:
    axes_path = validation_dir / "top_identifiable_lr_axes.csv"
    database_path = shared_input_dir / "filtered_lr_database.csv"
    axes, database = pd.read_csv(axes_path), pd.read_csv(database_path)
    _require(axes, ["ranking_target", "rank", "ligand", "receptor"], axes_path)
    _require(database, ["ligand", "receptor", "pathway"], database_path)
    top = axes.loc[
        axes["ranking_target"].eq("attention")
        & (pd.to_numeric(axes["rank"], errors="coerce") <= top_axis_rank)
    ].copy()
    for frame in (top, database):
        frame["axis"] = (
            frame["ligand"].astype(str).str.casefold()
            + "->"
            + frame["receptor"].astype(str).str.casefold()
        )
    background_axes, top_axes = set(database["axis"]), set(top["axis"])
    top_axes &= background_axes
    rows = []
    for pathway, group in database.groupby("pathway", dropna=False, sort=True):
        pathway_axes = set(group["axis"])
        hit = len(top_axes & pathway_axes)
        if not hit:
            continue
        population, successes, draws = len(background_axes), len(pathway_axes), len(top_axes)
        expected = draws * successes / population
        rows.append(
            {
                "pathway": str(pathway),
                "top_axis_hits": hit,
                "background_pathway_axes": successes,
                "n_unique_top_axes": draws,
                "n_unique_background_axes": population,
                "fold_enrichment": hit / expected,
                "hypergeometric_p_greater": float(
                    hypergeom.sf(hit - 1, population, successes, draws)
                ),
                "top_definition": (
                    f"rank <= {top_axis_rank} attention×LR axes per stage, pooled unique"
                ),
            }
        )
    result = pd.DataFrame(rows).sort_values(
        ["hypergeometric_p_greater", "fold_enrichment"], ascending=[True, False]
    )
    result["bh_q"] = _bh(result["hypergeometric_p_greater"].to_numpy())
    return result, {
        "top_identifiable_lr_axes": _record(axes_path),
        "lr_database_background": _record(database_path),
    }


def nichenet_consistency(
    validation_dir: Path,
    nichenet_custom_dir: Path,
    top_axis_rank: int,
    nichenet_top_ligands: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    axes_path = validation_dir / "top_identifiable_lr_axes.csv"
    activity_path = nichenet_custom_dir / "ligand_activity.csv"
    links_path = nichenet_custom_dir / "ligand_target_links.csv"
    axes, activity, links = (
        pd.read_csv(axes_path),
        pd.read_csv(activity_path),
        pd.read_csv(links_path),
    )
    _require(
        activity,
        ["unit_id", "source_stage_id", "target_stage_id", "receiver", "rank", "test_ligand", "aupr_corrected"],
        activity_path,
    )
    _require(links, ["unit_id", "ligand", "target"], links_path)
    top_axes = axes.loc[
        axes["ranking_target"].eq("attention")
        & (pd.to_numeric(axes["rank"], errors="coerce") <= top_axis_rank)
    ].copy()
    top_axes["ligand_key"] = top_axes["ligand"].astype(str).str.casefold()
    best_rank = top_axes.groupby(["stage", "ligand_key"])["rank"].min().to_dict()
    detail = activity.loc[
        pd.to_numeric(activity["rank"], errors="coerce") <= nichenet_top_ligands
    ].copy()
    detail["ligand_key"] = detail["test_ligand"].astype(str).str.casefold()
    detail["best_attention_axis_rank_at_source_stage"] = [
        best_rank.get((float(stage), ligand), np.nan)
        for stage, ligand in zip(detail["source_stage_id"], detail["ligand_key"])
    ]
    detail["attention_top_axis_supported"] = detail[
        "best_attention_axis_rank_at_source_stage"
    ].notna()
    links = links.copy()
    links["ligand_key"] = links["ligand"].astype(str).str.casefold()
    link_summary = (
        links.dropna(subset=["target"])
        .groupby(["unit_id", "ligand_key"])
        .agg(
            n_reported_nichenet_targets=("target", "nunique"),
            reported_nichenet_targets=("target", lambda x: ";".join(sorted(set(map(str, x))))),
        )
        .reset_index()
    )
    detail = detail.merge(link_summary, on=["unit_id", "ligand_key"], how="left")
    detail["n_reported_nichenet_targets"] = detail["n_reported_nichenet_targets"].fillna(0).astype(int)
    detail["reported_nichenet_targets"] = detail["reported_nichenet_targets"].fillna("")
    detail["analysis_role"] = (
        "downstream ligand-target interpretation; not direct spatial CCC strength"
    )

    def overlap_names(group: pd.DataFrame) -> str:
        return ";".join(
            sorted(group.loc[group["attention_top_axis_supported"], "test_ligand"].astype(str))
        )

    summary_rows = []
    group_keys = ["unit_id", "source_stage_id", "target_stage_id", "receiver"]
    for values, group in detail.groupby(group_keys, sort=False):
        summary_rows.append(
            {
                **dict(zip(group_keys, values)),
                "n_top_nichenet_ligands": int(len(group)),
                "n_attention_top_axis_supported": int(group["attention_top_axis_supported"].sum()),
                "attention_supported_fraction": float(group["attention_top_axis_supported"].mean()),
                "overlapping_ligands": overlap_names(group),
            }
        )
    return detail, pd.DataFrame(summary_rows), {
        "nichenet_ligand_activity": _record(activity_path),
        "nichenet_ligand_target_links": _record(links_path),
        "nichenet_run_manifest": _record(nichenet_custom_dir / "run_manifest.json"),
    }


def _save_figure(figure: plt.Figure, base: Path) -> None:
    figure.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def plot_overview(
    consensus_by_stage: pd.DataFrame,
    consensus_summary: pd.DataFrame,
    overlap_by_stage: pd.DataFrame,
    spatial_summary: pd.DataFrame,
    conditional: pd.DataFrame,
    output: Path,
) -> None:
    colors = {
        "CytoBridge attention": "#D1495B",
        "CytoBridge exact message": "#2E86AB",
        "COMMOT": "#2A9D8F",
        "CellAgentChat CTPS": "#F4A261",
        "CellChat triMean": "#7A5195",
    }
    figure, axes = plt.subplots(2, 2, figsize=(13.8, 10.2))
    panel = consensus_by_stage.loc[
        consensus_by_stage["design"].eq("external_only_native_primary")
    ]
    for target, group in panel.groupby("target", sort=False):
        axes[0, 0].plot(
            group["stage"], group["spearman"], marker="o", linewidth=2.2,
            label=target, color=colors[target]
        )
    axes[0, 0].axhline(0, color="#9CA3AF", linewidth=0.8)
    axes[0, 0].set_xticks(sorted(panel["stage"].unique()))
    axes[0, 0].set_xticklabels([STAGE_LABELS[x] for x in sorted(panel["stage"].unique())])
    axes[0, 0].set_ylim(-0.05, 0.9)
    axes[0, 0].set_ylabel("Spearman ρ vs external-only consensus")
    axes[0, 0].set_title("A  External-only method consensus (primary)", loc="left")
    axes[0, 0].legend(frameon=False, fontsize=9)

    article = consensus_summary.loc[
        consensus_summary["design"].isin(
            ["article_style_all_method_native_primary", "leave_one_method_out_native_primary"]
        )
    ]
    targets = list(NATIVE_METHODS)
    x, width = np.arange(len(targets)), 0.36
    designs = (
        ("article_style_all_method_native_primary", "All-method (self included)", "#8ECAE6"),
        ("leave_one_method_out_native_primary", "Leave-one-method-out", "#FB8500"),
    )
    for offset, (design, label, color) in enumerate(designs):
        values = [
            article.loc[
                article["design"].eq(design) & article["target"].eq(target),
                "mean_stage_spearman",
            ].squeeze()
            for target in targets
        ]
        axes[0, 1].bar(x + (offset - 0.5) * width, values, width, label=label, color=color)
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(
        ["CytoBridge\nattention", "COMMOT", "CellAgentChat\nCTPS", "CellChat\ntriMean"],
        fontsize=9,
    )
    axes[0, 1].set_ylim(-0.1, 1.0)
    axes[0, 1].set_ylabel("Mean stage-wise Spearman ρ")
    axes[0, 1].set_title("B  Article-style vs non-circular ensemble", loc="left")
    axes[0, 1].legend(frameon=False, fontsize=8)

    overlap = overlap_by_stage.loc[
        overlap_by_stage["target"].eq("CytoBridge attention")
        & overlap_by_stage["reference"].isin(
            ["External native consensus", "COMMOT", "CellAgentChat CTPS", "CellChat triMean"]
        )
    ]
    overlap_colors = {
        "External native consensus": "#D1495B",
        "COMMOT": "#2A9D8F",
        "CellAgentChat CTPS": "#F4A261",
        "CellChat triMean": "#7A5195",
    }
    for reference, group in overlap.groupby("reference", sort=False):
        axes[1, 0].plot(
            group["stage"], group["overlap_enrichment_over_random"], marker="o",
            linewidth=2, label=reference, color=overlap_colors[reference]
        )
    axes[1, 0].axhline(1, color="#374151", linestyle="--", linewidth=1)
    axes[1, 0].set_xticks(sorted(overlap["stage"].unique()))
    axes[1, 0].set_xticklabels([STAGE_LABELS[x] for x in sorted(overlap["stage"].unique())])
    axes[1, 0].set_ylabel("Top-20% overlap / random expectation")
    axes[1, 0].set_title("C  Top-signal overlap enrichment", loc="left")
    axes[1, 0].legend(frameon=False, fontsize=8, ncol=2)

    spatial_order = [
        "CytoBridge attention", "CytoBridge exact message", "COMMOT",
        "CellAgentChat CTPS", "CellChat triMean",
    ]
    values = [
        spatial_summary.loc[
            spatial_summary["method"].eq(method),
            "mean_stage_spearman_vs_inverse_distance",
        ].squeeze()
        for method in spatial_order
    ]
    axes[1, 1].bar(
        np.arange(len(values)), values,
        color=[colors.get(method, "#6B7280") for method in spatial_order]
    )
    axes[1, 1].axhline(0, color="#9CA3AF", linewidth=0.8)
    axes[1, 1].set_xticks(np.arange(len(values)))
    axes[1, 1].set_xticklabels(
        ["CytoBridge\nattention", "Exact\nmessage", "COMMOT", "CellAgentChat\nCTPS", "CellChat\ntriMean"],
        fontsize=8.5,
    )
    axes[1, 1].set_ylabel("Mean ρ(score, inverse mean distance)")
    axes[1, 1].set_title("D  Paper-style proximity localization", loc="left")
    matched = conditional.loc[
        conditional["target"].str.contains("attention", case=False, na=False)
        & conditional["score"].eq("lr_compatibility_forward")
    ]
    if len(matched) == 1:
        row = matched.iloc[0]
        axes[1, 1].text(
            0.02, 0.97,
            "Beyond proximity: attention residual vs forward LR\n"
            f"conditional ρ={row['conditional_rank_correlation']:.3f}, "
            f"q={row['bh_q_within_degree_matched_family']:.4g}",
            transform=axes[1, 1].transAxes, va="top", fontsize=8.5,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85},
        )
    for axis in axes.flat:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#E5E7EB", linewidth=0.7, alpha=0.8)
    figure.suptitle(
        "Zebrafish communication consistency: independent consensus and paper-style views",
        fontsize=15, fontweight="bold",
    )
    figure.text(
        0.5, 0.015,
        "Primary consensus excludes CytoBridge. The all-method ensemble includes the scored method and is supporting only.",
        ha="center", fontsize=8.5, color="#4B5563",
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.95))
    _save_figure(figure, output / "positive_consistency_overview")


def plot_biology(pathway: pd.DataFrame, niche: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13.8, 5.6))
    selected = pathway.head(10).sort_values("fold_enrichment")
    y = np.arange(len(selected))
    axes[0].barh(y, selected["fold_enrichment"], color="#2A9D8F")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(selected["pathway"])
    axes[0].axvline(1, color="#374151", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Fold enrichment vs LR-database background")
    axes[0].set_title("A  Pathways among top attention×LR axes", loc="left")
    for index, (_, row) in enumerate(selected.iterrows()):
        axes[0].text(row["fold_enrichment"] + 0.3, index, f"q={row['bh_q']:.2g}", va="center", fontsize=8)

    labels = [
        f"{int(row.source_stage_id)}→{int(row.target_stage_id)} | {row.receiver}"
        for row in niche.itertuples(index=False)
    ]
    y = np.arange(len(niche))
    axes[1].barh(y, niche["attention_supported_fraction"], color="#F4A261")
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(labels, fontsize=8)
    axes[1].set_xlim(0, 1.03)
    axes[1].set_xlabel("Fraction of top NicheNet ligands also in\ntop attention-supported LR axes")
    axes[1].set_title("B  NicheNet downstream ligand consistency", loc="left")
    for index, (_, row) in enumerate(niche.iterrows()):
        axes[1].text(
            min(float(row["attention_supported_fraction"]) + 0.02, 0.82), index,
            str(row["overlapping_ligands"]) or "none", va="center", fontsize=7.5,
        )
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="x", color="#E5E7EB", linewidth=0.7, alpha=0.8)
    figure.suptitle(
        "Top-signal biological interpretation (descriptive, not causal validation)",
        fontsize=14, fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    _save_figure(figure, output / "top_signal_biology")


def _summary_value(frame: pd.DataFrame, design: str, target: str) -> float:
    value = frame.loc[
        frame["design"].eq(design) & frame["target"].eq(target),
        "mean_stage_spearman",
    ]
    if len(value) != 1:
        raise ValueError(f"Missing unique summary row for {design} / {target}")
    return float(value.iloc[0])


def write_reports(
    output: Path,
    consensus_summary: pd.DataFrame,
    pairwise_summary: pd.DataFrame,
    overlap_summary: pd.DataFrame,
    pathway: pd.DataFrame,
    conditional: pd.DataFrame,
    niche: pd.DataFrame,
) -> None:
    external = _summary_value(consensus_summary, "external_only_native_primary", "CytoBridge attention")
    exact = _summary_value(consensus_summary, "external_only_native_primary", "CytoBridge exact message")
    article = _summary_value(consensus_summary, "article_style_all_method_native_primary", "CytoBridge attention")
    relaxed = _summary_value(
        consensus_summary, "external_only_threshold_relaxed_sensitivity", "CytoBridge attention"
    )

    def pair(reference: str) -> float:
        row = pairwise_summary.loc[
            pairwise_summary["target"].eq("CytoBridge attention")
            & pairwise_summary["reference"].eq(reference),
            "mean_stage_spearman",
        ]
        return float(row.iloc[0])

    commot, ctps = pair("COMMOT"), pair("CellAgentChat CTPS")
    continuous = pair("CellAgentChat continuous sensitivity")
    overlap = overlap_summary.loc[
        overlap_summary["target"].eq("CytoBridge attention")
        & overlap_summary["reference"].eq("External native consensus")
    ].iloc[0]
    degree = conditional.loc[
        conditional["target"].str.contains("attention", case=False, na=False)
        & conditional["score"].eq("lr_compatibility_forward")
    ].iloc[0]
    pathways = pathway.loc[pathway["bh_q"] < 0.05, "pathway"].tolist()
    ligands = sorted(
        {item for text in niche["overlapping_ligands"].astype(str) for item in text.split(";") if item}
    )
    readme = f"""# Zebrafish positive communication-consistency addendum

The primary external-only consensus excludes CytoBridge and combines within-stage
percentile ranks from COMMOT, CellAgentChat CTPS, and CellChat triMean.

- Attention vs external-only consensus: **mean stage Spearman rho = {external:.3f}**.
- Exact message vs external-only consensus: **{exact:.3f}**.
- Attention vs COMMOT directly: **{commot:.3f}**.
- Article-style all-method ensemble, which includes CytoBridge: **{article:.3f}**.
  This self-included result is supporting rather than primary evidence.
- Top-20% attention interactions overlap the external consensus by
  **{overlap['mean_overlap_fraction']:.1%}** on average, or
  **{overlap['mean_overlap_enrichment']:.2f}x** random expectation.

CellAgentChat Methods Eq. 8 defines CTPS as the sum of significant interaction
scores, not the number of significant LR pairs. The immutable run already stored
the correct score-sum column. Direct attention-vs-CTPS rho is **{ctps:.3f}**;
the unthresholded CellAgentChat sensitivity is **{continuous:.3f}**.

CellChat triMean is all-zero in the first three sparse stages. A truncatedMean
sensitivity is nonzero in all five stages; the threshold-relaxed external
consensus remains positive (**{relaxed:.3f}**). CellChat is still non-spatial in
this run.

After matching stage, type pair, distance, state similarity, and graph degree,
attention residuals retain positive forward-LR consistency (conditional rho =
**{degree['conditional_rank_correlation']:.3f}**, BH q =
**{degree['bh_q_within_degree_matched_family']:.4g}**), so the signal is not
explained solely by proximity.

Top attention-supported LR axes are enriched for {', '.join(pathways[:10])}.
NicheNet is treated as downstream ligand-to-target interpretation; overlapping
ligands include {', '.join(ligands[:12])}.

Reference: Raghavan et al., Genome Research (2025), CellAgentChat:
https://genome.cshlp.org/content/early/2025/04/29/gr279771124
"""
    (output / "README.md").write_text(readme, encoding="utf-8")

    reviewer = f"""## Draft reviewer response

We added a multi-method validation modeled on the CellAgentChat benchmarking
framework. To avoid circular inflation, our primary analysis formed a
within-stage rank consensus from COMMOT, CellAgentChat, and CellChat while
excluding CytoBridge. CytoBridge attention agreed positively with this
external-only consensus across all five stages (mean stage-wise Spearman rho =
{external:.3f}); the exactly reconstructed spatial-GNN message was even more
concordant (rho = {exact:.3f}). Direct agreement with COMMOT was also strong
(rho = {commot:.3f}).

For comparison with the published CellAgentChat design, we also report a
self-included all-method ensemble (rho = {article:.3f}), but do not use it as
the primary independent result. Top-20% attention interactions overlapped the
external consensus by {overlap['mean_overlap_fraction']:.1%} on average
({overlap['mean_overlap_enrichment']:.2f}-fold over random expectation). After
matching developmental stage, sender/receiver identity, distance, state
similarity, and graph degree, attention residuals retained positive forward-LR
consistency (conditional rho = {degree['conditional_rank_correlation']:.3f},
BH q = {degree['bh_q_within_degree_matched_family']:.4g}).

We corrected CellAgentChat to its published CTPS definition (the sum of
significant interaction scores, Methods Eq. 8) and added unthresholded and
CellChat truncatedMean sensitivity analyses. NicheNet is now used for
downstream ligand-to-target interpretation rather than as an interchangeable
spatial CCC-strength score. These analyses support the bounded conclusion that
CytoBridge attention contains a reproducible communication-related interaction
signal, without claiming that attention is itself biochemical flux or a causal
communication probability.
"""
    (output / "reviewer_response_draft.md").write_text(reviewer, encoding="utf-8")

    chinese = f"""# 斑马鱼 communication 正向一致性结果说明

最严格的 external-only consensus 完全不包含 CytoBridge，由 COMMOT、
CellAgentChat CTPS 和 CellChat 组成。Attention 与它在五个 stage 的平均
Spearman 相关为 **{external:.3f}**，exact message 为 **{exact:.3f}**；
attention 与 COMMOT 的直接相关为 **{commot:.3f}**。这可以支持审稿人回复：
CytoBridge attention 含有可重复的 communication-related interaction signal。

仿照 CellAgentChat Fig. 2A 把我们也放进 all-method ensemble 后，attention
相关为 **{article:.3f}**。这个结果可以展示，但必须注明 ensemble 含有被
评价方法自身，因此是 supporting result，主结论仍用 external-only。

Top 20% signal 与 external consensus 的平均 overlap 为
**{overlap['mean_overlap_fraction']:.1%}**，是随机期望的
**{overlap['mean_overlap_enrichment']:.2f} 倍**。在匹配时间、type pair、
距离、状态和图 degree 后，attention residual 与 forward LR 仍显著正相关
（rho = **{degree['conditional_rank_correlation']:.3f}**, BH q =
**{degree['bh_q_within_degree_matched_family']:.4g}**），不能只解释为近邻效应。

需要修正两点：CellAgentChat Eq. 8 的 CTPS 是显著 interaction score 之和，
不是显著 LR 数量；已有输出包含正确列，不需重训。严格 CTPS 与 attention
的相关为 **{ctps:.3f}**，未阈值连续分数为 **{continuous:.3f}**。CellChat
默认 triMean 在前三阶段全零，truncatedMean 后五阶段均非零，但本次
CellChat 仍是非空间 type-level 分析。

NicheNet 改为 downstream ligand-to-target consistency，不再和 spatial CCC
strength 强行同单位比较。重合 ligand 包括 {', '.join(ligands[:12])}。
Top LR 通路富集包括 {', '.join(pathways[:10])}。
"""
    (output / "汇报说明.md").write_text(chinese, encoding="utf-8")


def main() -> None:
    args = _parser().parse_args()
    if not 0 < args.top_fraction < 1:
        raise ValueError("--top-fraction must be between 0 and 1")
    if min(args.top_axis_rank, args.nichenet_top_ligands) <= 0:
        raise ValueError("Top-rank arguments must be positive")
    output = _prepare_output(args.output_dir, args.overwrite)
    scores, provenance = load_scores(
        args.comparison_dir.resolve(),
        args.cellagentchat_project_dir.resolve(),
        args.cellchat_truncated_dir.resolve(),
    )
    consensus_stage, consensus_summary = consensus_metrics(scores)
    pairwise_stage, pairwise_summary = pairwise_metrics(scores)
    overlap_stage, overlap_summary = top_overlap(scores, args.top_fraction)
    spatial_stage, spatial_summary, conditional, spatial_provenance = spatial_metrics(
        scores, args.cytobridge_dir.resolve(), args.validation_dir.resolve()
    )
    pathway, pathway_provenance = pathway_enrichment(
        args.validation_dir.resolve(), args.shared_input_dir.resolve(), args.top_axis_rank
    )
    niche_detail, niche_summary, niche_provenance = nichenet_consistency(
        args.validation_dir.resolve(), args.nichenet_custom_dir.resolve(),
        args.top_axis_rank, args.nichenet_top_ligands,
    )

    artifacts: list[Path] = []
    tables = {
        "harmonized_type_pair_scores.csv.gz": (scores, {"compression": "gzip"}),
        "consensus_by_stage.csv": (consensus_stage, {}),
        "consensus_summary.csv": (consensus_summary, {}),
        "pairwise_sensitivity_by_stage.csv": (pairwise_stage, {}),
        "pairwise_sensitivity_summary.csv": (pairwise_summary, {}),
        "top_signal_overlap_by_stage.csv": (overlap_stage, {}),
        "top_signal_overlap_summary.csv": (overlap_summary, {}),
        "spatial_proximity_by_stage.csv": (spatial_stage, {}),
        "spatial_proximity_summary.csv": (spatial_summary, {}),
        "spatial_beyond_proximity_conditional_tests.csv": (conditional, {}),
        "pathway_enrichment.csv": (pathway, {}),
        "nichenet_downstream_ligand_detail.csv": (niche_detail, {}),
        "nichenet_downstream_consistency_summary.csv": (niche_summary, {}),
    }
    for name, (frame, options) in tables.items():
        path = output / name
        frame.to_csv(path, index=False, **options)
        artifacts.append(path)
    plot_overview(
        consensus_stage, consensus_summary, overlap_stage, spatial_summary,
        conditional, output,
    )
    plot_biology(pathway, niche_summary, output)
    artifacts.extend(
        output / name for name in (
            "positive_consistency_overview.png", "positive_consistency_overview.pdf",
            "top_signal_biology.png", "top_signal_biology.pdf",
        )
    )
    write_reports(
        output, consensus_summary, pairwise_summary, overlap_summary, pathway,
        conditional, niche_summary,
    )
    artifacts.extend(output / name for name in ("README.md", "reviewer_response_draft.md", "汇报说明.md"))
    manifest = {
        "schema_version": 1,
        "workflow": "zebrafish_paper_style_positive_communication_consistency",
        "primary_design": {
            "name": "external_only_native_primary",
            "components": ["COMMOT", "CellAgentChat CTPS", "CellChat triMean"],
            "cytobridge_excluded": True,
            "normalization": "within-stage percentile rank",
        },
        "supporting_design": {
            "name": "article_style_all_method_native_primary",
            "self_inclusion_disclosed": True,
        },
        "cellagentchat_ctps_correction": {
            "definition": "sum of Bonferroni-significant interaction scores (Methods Eq. 8)",
            "source_column": "cellagentchat_significant_score_sum_mean",
            "source_files_mutated": False,
        },
        "top_fraction": args.top_fraction,
        "top_axis_rank": args.top_axis_rank,
        "provenance": {**provenance, **spatial_provenance, **pathway_provenance, **niche_provenance},
        "artifacts": {path.name: _record(path) for path in sorted(artifacts)},
        "guardrails": {
            "attention_is_ccc_probability": False,
            "all_method_ensemble_is_independent_validation": False,
            "nichenet_is_direct_spatial_ccc_strength": False,
            "pathway_enrichment_is_causal_validation": False,
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote positive-consistency analysis to {output}")


if __name__ == "__main__":
    main()
