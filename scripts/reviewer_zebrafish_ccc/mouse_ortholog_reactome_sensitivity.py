#!/usr/bin/env python3
"""Run a post-hoc mouse-ortholog Reactome sensitivity analysis.

This entry point consumes the frozen zebrafish top-LR query manifest emitted by
``reactome_pathway_consistency.py`` and projects its ligand/receptor genes
through the CellAgentChat project-LR crosswalk.  It is deliberately a
cross-species sensitivity analysis, not a native zebrafish primary analysis:
the frozen crosswalk was built from Ensembl one-to-one orthologues without a
confidence filter.

The live analysis is pinned to the g:Profiler e113 archive and mouse organism.
Unit tests never access the network.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import shutil
import sys
import textwrap
from typing import Any, Callable, Iterable, Mapping, Sequence
from urllib.parse import urlencode

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from reactome_pathway_consistency import (
    ALPHA,
    ALL_SCORE_METHODS,
    ARCHIVE_BASE,
    CB_METHODS,
    EXPECTED_GPROFILER_VERSION,
    NATIVE_EXTERNAL,
    RELAXED_EXTERNAL,
    STAGES,
    STAGE_LABELS,
    _record,
    _git_state,
    _request_json,
    _write_json,
    parse_profile_response,
    readout_consistency_metrics,
    select_external_pathways,
)


ORGANISM = "mmusculus"
MOUSE_ENSEMBL_PATTERN = re.compile(r"ENSMUSG\d+")
ANALYSIS_TIER = "post-hoc cross-species sensitivity"
ORTHOLOGY_POLICY = "one2one_bijective_all_confidence"

NATIVE_DISPLAY_METHODS = (
    "CytoBridge attention x LR",
    "COMMOT",
    "CellChat triMean",
    "CellAgentChat significant",
)
RELAXED_DISPLAY_METHODS = (
    "CytoBridge attention x LR",
    "COMMOT",
    "CellChat truncatedMean",
    "CellAgentChat continuous",
)
METHOD_LABELS = {
    "CytoBridge attention x LR": "CytoBridge attention x LR",
    "CytoBridge exact message x LR": "CytoBridge exact x LR",
    "CytoBridge exact message only (LR-conditioned)": (
        "CytoBridge exact only (LR-conditioned)"
    ),
    "CytoBridge LR-only": "CytoBridge LR-only",
    "COMMOT": "COMMOT",
    "CellChat triMean": "CellChat triMean",
    "CellChat truncatedMean": "CellChat truncatedMean",
    "CellAgentChat significant": "CellAgentChat significant",
    "CellAgentChat continuous": "CellAgentChat continuous",
}
PROFILE_SPECS = (
    {
        "profile_id": "mouse_annotated_gscs_ligand_receptor",
        "domain_scope": "annotated",
        "correction": "g_SCS",
        "background": None,
        "profile_label": "annotated/default background + g:SCS",
    },
    {
        "profile_id": "mouse_custom_fdr_ligand_receptor",
        "domain_scope": "custom",
        "correction": "fdr",
        "background": "strict_background",
        "profile_label": "shared mouse-ortholog background + FDR",
    },
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-manifest", required=True, type=Path)
    parser.add_argument("--top-lr-pairs", type=Path)
    parser.add_argument("--cellagentchat-crosswalk", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--min-external-methods", type=int, default=2)
    parser.add_argument("--max-pathways", type=int, default=24)
    parser.add_argument("--api-timeout-seconds", type=int, default=180)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} lacks required columns: {missing}")


def _source_components(value: str) -> list[str]:
    return sorted(
        {part.strip().casefold() for part in str(value).split("_") if part.strip()}
    )


def _mapped_components(value: str) -> list[str]:
    seen: dict[str, str] = {}
    for part in str(value).split("_"):
        stripped = part.strip()
        if stripped:
            seen.setdefault(stripped.casefold(), stripped)
    return [seen[key] for key in sorted(seen)]


def build_strict_unique_gene_map(
    crosswalk: pd.DataFrame,
    *,
    require_complete: bool = True,
) -> tuple[dict[str, str], pd.DataFrame]:
    """Build a symbol-bijective zebrafish-to-mouse map from LR columns.

    Complex fields are rejected because a row-level LR projection does not
    establish component-level correspondence.  The formal 134-axis crosswalk
    contains singleton genes, so this is a guardrail rather than a data loss
    operation.
    """

    required = (
        "source_ligand",
        "source_receptor",
        "mapped_ligand",
        "mapped_receptor",
    )
    _require_columns(crosswalk, required, "CellAgentChat crosswalk")
    candidates: list[dict[str, str]] = []
    for row_number, row in enumerate(crosswalk.itertuples(index=False)):
        for role in ("ligand", "receptor"):
            source = _source_components(getattr(row, f"source_{role}"))
            mapped = _mapped_components(getattr(row, f"mapped_{role}"))
            if len(source) != 1 or len(mapped) != 1:
                raise ValueError(
                    "Cannot infer component-level orthology from a complex "
                    f"crosswalk field at row {row_number}, role {role}: "
                    f"{source!r} -> {mapped!r}"
                )
            candidates.append(
                {
                    "source_symbol": source[0],
                    "mapped_symbol": mapped[0],
                    "mapped_key": mapped[0].casefold(),
                    "role": role,
                }
            )
    candidate_frame = pd.DataFrame(candidates).drop_duplicates()
    source_targets = (
        candidate_frame.groupby("source_symbol")["mapped_key"]
        .agg(lambda values: sorted(set(values)))
        .to_dict()
    )
    target_sources = (
        candidate_frame.groupby("mapped_key")["source_symbol"]
        .agg(lambda values: sorted(set(values)))
        .to_dict()
    )
    mapped_spelling = (
        candidate_frame.sort_values(["mapped_key", "mapped_symbol"])
        .drop_duplicates("mapped_key")
        .set_index("mapped_key")["mapped_symbol"]
        .to_dict()
    )
    mapping: dict[str, str] = {}
    audit_rows: list[dict[str, Any]] = []
    for source_symbol in sorted(source_targets):
        target_keys = source_targets[source_symbol]
        if len(target_keys) != 1:
            status = "ambiguous_source"
        elif len(target_sources[target_keys[0]]) != 1:
            status = "ambiguous_target"
        else:
            status = "one_to_one"
            mapping[source_symbol] = mapped_spelling[target_keys[0]]
        audit_rows.append(
            {
                "source_symbol_zebrafish": source_symbol,
                "mapped_symbols_mouse": ";".join(
                    mapped_spelling[key] for key in target_keys
                ),
                "n_mapped_symbols": int(len(target_keys)),
                "n_source_symbols_for_target": (
                    int(len(target_sources[target_keys[0]]))
                    if len(target_keys) == 1
                    else np.nan
                ),
                "status": status,
            }
        )
    audit = pd.DataFrame(audit_rows)
    if require_complete and not audit["status"].eq("one_to_one").all():
        examples = audit.loc[
            ~audit["status"].eq("one_to_one"),
            ["source_symbol_zebrafish", "mapped_symbols_mouse", "status"],
        ].head(8)
        raise ValueError(
            "Crosswalk does not define a complete symbol-bijective gene map: "
            + examples.to_dict("records").__repr__()
        )
    if len(set(value.casefold() for value in mapping.values())) != len(mapping):
        raise AssertionError("Internal error: mapped mouse symbols are not unique")
    return mapping, audit


def map_query_manifest(
    query_manifest: pd.DataFrame,
    gene_map: Mapping[str, str],
) -> pd.DataFrame:
    required = (
        "query_id",
        "gene_mode",
        "method",
        "stage",
        "stage_label",
        "query_gene_symbols",
    )
    _require_columns(query_manifest, required, "zebrafish query manifest")
    result = query_manifest.loc[
        query_manifest["gene_mode"].eq("ligand_receptor")
    ].copy()
    if result.empty:
        raise ValueError("Query manifest has no ligand_receptor rows")
    result["stage"] = result["stage"].astype(float)
    if result.duplicated("query_id").any():
        raise ValueError("Ligand+receptor query_id values are not unique")
    unknown_methods = sorted(set(result["method"]) - set(ALL_SCORE_METHODS))
    if unknown_methods:
        raise ValueError(
            f"Query manifest contains unsupported methods: {unknown_methods}"
        )
    for column in (
        "query_genes_ensembl",
        "excluded_ambiguous_or_failed_symbols",
        "n_query_genes_ensembl_1to1",
        "api_queried",
    ):
        if column in result:
            result = result.drop(columns=column)
    source_values: list[str] = []
    mouse_values: list[str] = []
    missing_values: list[str] = []
    for row in result.itertuples(index=False):
        raw_symbols = row.query_gene_symbols
        symbol_values = (
            []
            if pd.isna(raw_symbols)
            else [value for value in str(raw_symbols).split(";") if value.strip()]
        )
        source = sorted({value.strip().casefold() for value in symbol_values})
        missing = sorted(set(source) - set(gene_map))
        if missing:
            raise ValueError(
                f"Query {row.query_id} has genes absent from the strict crosswalk: "
                f"{missing[:8]}"
            )
        mouse = sorted(
            {gene_map[value] for value in source},
            key=lambda value: value.casefold(),
        )
        source_values.append(";".join(source))
        mouse_values.append(";".join(mouse))
        missing_values.append("")
    result = result.rename(
        columns={"query_gene_symbols": "query_gene_symbols_input_zebrafish"}
    )
    result["query_gene_symbols_zebrafish"] = source_values
    result["query_gene_symbols_mouse"] = mouse_values
    result["n_query_gene_symbols_mouse"] = [
        len([value for value in values.split(";") if value]) for values in mouse_values
    ]
    result["excluded_missing_crosswalk_symbols"] = missing_values
    return result.sort_values(["method", "stage", "query_id"]).reset_index(drop=True)


def validate_top_lr_pairs(
    top_lr_pairs: pd.DataFrame,
    query_manifest_mouse: pd.DataFrame,
    crosswalk: pd.DataFrame,
) -> pd.DataFrame:
    _require_columns(
        top_lr_pairs,
        ("method", "stage", "ligand", "receptor"),
        "top LR pairs",
    )
    _require_columns(
        crosswalk,
        ("source_ligand", "source_receptor"),
        "CellAgentChat crosswalk",
    )
    valid_axes = {
        f"{str(row.source_ligand).casefold()}->{str(row.source_receptor).casefold()}"
        for row in crosswalk.itertuples(index=False)
    }
    local = top_lr_pairs.copy()
    local["stage"] = local["stage"].astype(float)
    local["axis_from_columns"] = (
        local["ligand"].astype(str).str.casefold()
        + "->"
        + local["receptor"].astype(str).str.casefold()
    )
    invalid_axes = sorted(set(local["axis_from_columns"]) - valid_axes)
    if invalid_axes:
        raise ValueError(
            "Top LR pairs contain axes absent from the crosswalk: "
            f"{invalid_axes[:8]}"
        )
    rows: list[dict[str, Any]] = []
    for query in query_manifest_mouse.itertuples(index=False):
        group = local.loc[
            local["method"].eq(query.method) & local["stage"].eq(float(query.stage))
        ]
        reconstructed = sorted(
            {
                gene
                for value in pd.concat(
                    [group["ligand"].astype(str), group["receptor"].astype(str)]
                )
                for gene in _source_components(value)
            }
        )
        manifest_genes = sorted(
            value
            for value in str(query.query_gene_symbols_zebrafish).split(";")
            if value
        )
        matches = reconstructed == manifest_genes
        rows.append(
            {
                "query_id": query.query_id,
                "method": query.method,
                "stage": float(query.stage),
                "stage_label": query.stage_label,
                "n_top_lr_pairs": int(len(group)),
                "n_reconstructed_source_genes": int(len(reconstructed)),
                "n_manifest_source_genes": int(len(manifest_genes)),
                "source_gene_sets_match": bool(matches),
                "reconstructed_source_genes": ";".join(reconstructed),
                "manifest_source_genes": ";".join(manifest_genes),
            }
        )
    audit = pd.DataFrame(rows)
    if not audit["source_gene_sets_match"].all():
        examples = audit.loc[
            ~audit["source_gene_sets_match"],
            ["query_id", "method", "stage_label"],
        ].head(8)
        raise ValueError(
            "Top LR pairs disagree with query_manifest gene sets: "
            + examples.to_dict("records").__repr__()
        )
    return audit


def strict_mouse_one_to_one_conversion(
    response: Mapping[str, Any],
    expected_symbols: Iterable[str],
) -> tuple[dict[str, str], pd.DataFrame]:
    """Keep only symbol-bijective ``ENSMUSG`` conversion records."""

    result = pd.DataFrame(response.get("result", []))
    if not result.empty:
        _require_columns(result, ("incoming", "converted"), "g:Profiler conversion")
        result = result.copy()
        result["incoming_key"] = result["incoming"].astype(str).str.casefold()
    expected = sorted(set(map(str, expected_symbols)), key=str.casefold)
    candidates: dict[str, list[str]] = {}
    for symbol in expected:
        if result.empty:
            values: list[str] = []
        else:
            values = sorted(
                {
                    value
                    for value in (
                        result.loc[
                            result["incoming_key"].eq(symbol.casefold()), "converted"
                        ]
                        .dropna()
                        .astype(str)
                    )
                    if MOUSE_ENSEMBL_PATTERN.fullmatch(value)
                }
            )
        candidates[symbol] = values
    target_sources: dict[str, list[str]] = {}
    for symbol, values in candidates.items():
        if len(values) == 1:
            target_sources.setdefault(values[0], []).append(symbol)
    conversion: dict[str, str] = {}
    audit_rows: list[dict[str, Any]] = []
    for symbol in expected:
        values = candidates[symbol]
        if not values:
            status = "failed"
        elif len(values) > 1:
            status = "ambiguous_source"
        elif len(target_sources[values[0]]) > 1:
            status = "ambiguous_target"
        else:
            status = "one_to_one"
            conversion[symbol] = values[0]
        audit_rows.append(
            {
                "mouse_gene_symbol": symbol,
                "status": status,
                "n_valid_ensmusg_ids": int(len(values)),
                "converted_ensembl_ids": ";".join(values),
                "n_symbols_for_ensembl_id": (
                    int(len(target_sources[values[0]])) if len(values) == 1 else np.nan
                ),
            }
        )
    return conversion, pd.DataFrame(audit_rows)


def add_mouse_ensembl_queries(
    manifest: pd.DataFrame,
    conversion: Mapping[str, str],
) -> pd.DataFrame:
    result = manifest.copy()
    conversion_by_key = {
        symbol.casefold(): ensembl for symbol, ensembl in conversion.items()
    }
    query_ensembl: list[str] = []
    excluded: list[str] = []
    counts: list[int] = []
    for row in result.itertuples(index=False):
        symbols = [
            value for value in str(row.query_gene_symbols_mouse).split(";") if value
        ]
        ensembl = sorted(
            {
                conversion_by_key[value.casefold()]
                for value in symbols
                if value.casefold() in conversion_by_key
            }
        )
        missing = sorted(
            {value for value in symbols if value.casefold() not in conversion_by_key},
            key=str.casefold,
        )
        query_ensembl.append(";".join(ensembl))
        excluded.append(";".join(missing))
        counts.append(len(ensembl))
    result["query_genes_ensembl"] = query_ensembl
    result["excluded_mouse_symbols_conversion"] = excluded
    result["n_query_genes_ensembl_1to1"] = counts
    result["api_queried"] = result["n_query_genes_ensembl_1to1"].gt(0)
    return result


def build_mouse_profile_payload(
    manifest: pd.DataFrame,
    *,
    domain_scope: str,
    correction: str,
    background: Sequence[str] | None,
) -> dict[str, Any]:
    local = manifest.loc[manifest["api_queried"]]
    queries = {
        row.query_id: [
            value for value in str(row.query_genes_ensembl).split(";") if value
        ]
        for row in local.itertuples(index=False)
    }
    payload: dict[str, Any] = {
        "organism": ORGANISM,
        "query": queries,
        "sources": ["REAC"],
        "user_threshold": 1.0,
        "all_results": True,
        "ordered": False,
        "combined": False,
        "domain_scope": domain_scope,
        "significance_threshold_method": correction,
        "no_evidences": False,
    }
    if background is not None:
        payload["background"] = list(background)
    return payload


def _selected_pathways(selection: pd.DataFrame) -> pd.DataFrame:
    selected = (
        selection.loc[selection["selected_for_heatmap"]]
        .drop_duplicates(["reactome_id", "reactome_name"])
        .sort_values(
            [
                "n_stages_passing_external_only",
                "n_external_stage_method_significant",
                "mean_external_minus_log10_p",
                "reactome_name",
            ],
            ascending=[False, False, False, True],
        )
    )
    return selected


def wrap_pathway_name(value: str, width: int = 40) -> str:
    return textwrap.fill(
        str(value),
        width=width,
        break_long_words=False,
        break_on_hyphens=False,
    )


def _figure_subtitle(
    external_methods: Sequence[str],
    display_methods: Sequence[str],
    *,
    min_external_methods: int,
) -> str:
    discovery = ", ".join(METHOD_LABELS[method] for method in external_methods)
    display = ", ".join(METHOD_LABELS[method] for method in display_methods)
    return (
        f"Discovery rows: {discovery} only; CytoBridge excluded "
        f"(>={min_external_methods} external methods, adjusted P < {ALPHA:.2f}).\n"
        f"Display columns: {display}."
    )


def _empty_diagnostic(
    enrichment: pd.DataFrame,
    selection: pd.DataFrame,
    *,
    profile_id: str,
    external_methods: Sequence[str],
    display_methods: Sequence[str],
    min_external_methods: int,
    title: str,
    output_path: Path,
) -> None:
    local = enrichment.loc[
        enrichment["profile_id"].eq(profile_id) & ~enrichment["is_reactome_root"]
    ]
    x = np.arange(len(STAGES), dtype=float)
    figure, axis = plt.subplots(figsize=(12, 5.2))
    width = 0.20
    colors = ("#4C78A8", "#59A14F", "#F28E2B")
    for method_index, method in enumerate(external_methods):
        counts = [
            int(
                local.loc[
                    local["stage"].eq(stage)
                    & local["method"].eq(method)
                    & local["adjusted_p_value"].lt(ALPHA),
                    "reactome_id",
                ].nunique()
            )
            for stage in STAGES
        ]
        offset = (method_index - (len(external_methods) - 1) / 2) * width
        axis.bar(
            x + offset,
            counts,
            width=width,
            color=colors[method_index % len(colors)],
            label=METHOD_LABELS[method],
        )
    overlap_counts = [
        int(
            selection.loc[
                selection["stage"].eq(stage) & selection["passes_external_only_rule"],
                "reactome_id",
            ].nunique()
        )
        for stage in STAGES
    ]
    axis.plot(
        x,
        overlap_counts,
        color="#111827",
        linewidth=2.0,
        marker="o",
        markersize=5,
        label=f"Pathways supported by >={min_external_methods} external methods",
    )
    axis.set_xticks(x)
    axis.set_xticklabels([STAGE_LABELS[stage] for stage in STAGES])
    axis.set_ylabel("Reactome pathways with adjusted P < 0.01")
    axis.set_xlabel("Developmental stage")
    axis.set_ylim(bottom=0)
    axis.grid(axis="y", alpha=0.25, linewidth=0.7)
    axis.legend(
        loc="upper left",
        frameon=False,
        fontsize=8,
        ncol=2,
    )
    axis.set_title(
        _figure_subtitle(
            external_methods,
            display_methods,
            min_external_methods=min_external_methods,
        ),
        fontsize=9.5,
        pad=12,
    )
    axis.text(
        0.99,
        0.96,
        "No pathway passed external-only discovery;\n"
        "there are no all-method heatmap rows to display.",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        fontweight="bold",
        color="#7F1D1D",
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "#FEF2F2",
            "edgecolor": "#FCA5A5",
        },
    )
    figure.suptitle(title, fontsize=14, fontweight="bold", y=0.995)
    figure.subplots_adjust(left=0.08, right=0.98, bottom=0.15, top=0.78)
    figure.savefig(output_path.with_suffix(".png"), dpi=300)
    figure.savefig(output_path.with_suffix(".pdf"))
    plt.close(figure)


def plot_sensitivity_heatmap(
    enrichment: pd.DataFrame,
    query_manifest: pd.DataFrame,
    selection: pd.DataFrame,
    *,
    profile_id: str,
    external_methods: Sequence[str],
    display_methods: Sequence[str],
    min_external_methods: int,
    title: str,
    output_path: Path,
) -> None:
    selected = _selected_pathways(selection)
    if selected.empty:
        _empty_diagnostic(
            enrichment,
            selection,
            profile_id=profile_id,
            external_methods=external_methods,
            display_methods=display_methods,
            min_external_methods=min_external_methods,
            title=title,
            output_path=output_path,
        )
        return
    pathway_ids = selected["reactome_id"].astype(str).tolist()
    pathway_names = selected["reactome_name"].astype(str).tolist()
    columns = [(float(stage), method) for stage in STAGES for method in display_methods]
    matrix = np.zeros((len(pathway_ids), len(columns)), dtype=float)
    significance = np.zeros_like(matrix, dtype=bool)
    local = enrichment.loc[enrichment["profile_id"].eq(profile_id)]
    lookup = {
        (str(row.reactome_id), float(row.stage), str(row.method)): float(
            row.adjusted_p_value
        )
        for row in local.itertuples(index=False)
    }
    query_available = {
        (float(row.stage), str(row.method)): bool(row.api_queried)
        for row in query_manifest.itertuples(index=False)
    }
    for row_index, pathway_id in enumerate(pathway_ids):
        for column_index, (stage, method) in enumerate(columns):
            if not query_available.get((stage, method), False):
                matrix[row_index, column_index] = np.nan
                continue
            p_value = lookup.get((pathway_id, stage, method), 1.0)
            matrix[row_index, column_index] = -math.log10(max(p_value, 1e-300))
            significance[row_index, column_index] = p_value < ALPHA
    wrapped_names = [wrap_pathway_name(value) for value in pathway_names]
    n_label_lines = sum(value.count("\n") + 1 for value in wrapped_names)
    height = max(7.6, 0.30 * n_label_lines + 3.8)
    figure, axes = plt.subplots(
        1,
        len(STAGES),
        figsize=(18, height),
        sharey=True,
    )
    cmap = plt.get_cmap("Reds").copy()
    cmap.set_bad("#D1D5DB")
    finite = matrix[np.isfinite(matrix)]
    vmax = min(30.0, max(2.0, float(np.quantile(finite, 0.98)))) if len(finite) else 2.0
    short_labels = {
        "CytoBridge attention x LR": "CytoBridge",
        "CytoBridge exact message x LR": "Exact x LR",
        "CytoBridge exact message only (LR-conditioned)": "Exact only",
        "CytoBridge LR-only": "LR-only",
        "COMMOT": "COMMOT",
        "CellChat triMean": "CellChat",
        "CellChat truncatedMean": "CellChat",
        "CellAgentChat significant": "CellAgentChat",
        "CellAgentChat continuous": "CellAgentChat",
    }
    image = None
    n_methods = len(display_methods)
    for stage_index, (axis, stage) in enumerate(zip(axes, STAGES)):
        start = stage_index * n_methods
        stop = start + n_methods
        local_matrix = matrix[:, start:stop]
        local_significance = significance[:, start:stop]
        image = axis.imshow(
            local_matrix,
            cmap=cmap,
            vmin=0,
            vmax=vmax,
            aspect="auto",
        )
        axis.set_xticks(range(n_methods))
        axis.set_xticklabels(
            [short_labels[method] for method in display_methods],
            rotation=42,
            ha="right",
            fontsize=8,
        )
        axis.set_yticks(range(len(wrapped_names)))
        axis.set_yticklabels(
            wrapped_names,
            fontsize=8.4,
            linespacing=1.10,
        )
        axis.set_title(
            STAGE_LABELS[stage],
            fontsize=10.5,
            fontweight="bold",
            pad=8,
        )
        for row_index in range(local_matrix.shape[0]):
            for column_index in range(local_matrix.shape[1]):
                if local_significance[row_index, column_index]:
                    axis.text(
                        column_index,
                        row_index,
                        "*",
                        ha="center",
                        va="center",
                        color="#111827",
                        fontsize=8.5,
                    )
    assert image is not None
    subtitle = _figure_subtitle(
        external_methods,
        display_methods,
        min_external_methods=min_external_methods,
    )
    figure.suptitle(title, fontsize=14, fontweight="bold", y=0.995)
    figure.text(
        0.5,
        0.915,
        subtitle
        + "\nColor: -log10(adjusted P); * adjusted P < 0.01; "
        "grey = no positive-LR query.",
        ha="center",
        va="top",
        fontsize=9.2,
    )
    color_axis = figure.add_axes([0.39, 0.045, 0.28, 0.018])
    colorbar = figure.colorbar(
        image,
        cax=color_axis,
        orientation="horizontal",
    )
    colorbar.set_label("-log10 adjusted P", fontsize=9.5)
    colorbar.ax.tick_params(labelsize=8)
    figure.subplots_adjust(
        left=0.25,
        right=0.985,
        bottom=0.18,
        top=0.80,
        wspace=0.10,
    )
    figure.savefig(
        output_path.with_suffix(".png"),
        dpi=300,
        bbox_inches="tight",
    )
    figure.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def _profile_specs_with_background(
    background: Sequence[str],
) -> tuple[dict[str, Any], ...]:
    specs: list[dict[str, Any]] = []
    for spec in PROFILE_SPECS:
        local = dict(spec)
        if local["background"] == "strict_background":
            local["background"] = list(background)
        specs.append(local)
    return tuple(specs)


def _write_readme(
    output_dir: Path,
    selections: pd.DataFrame,
    metrics: pd.DataFrame,
    crosswalk_audit: pd.DataFrame,
    conversion_audit: pd.DataFrame,
    audit: Mapping[str, Any],
) -> None:
    selection_counts = (
        selections.loc[selections["selected_for_heatmap"]]
        .groupby("analysis_id")["reactome_id"]
        .nunique()
        .to_dict()
    )
    metric_rows: list[str] = []
    for row in metrics.itertuples(index=False):
        statistic = row.spearman_rho_vs_external_median_minus_log10_p
        rho_label = "未定义" if pd.isna(statistic) else f"{float(statistic):.3f}"
        metric_rows.append(
            f"- `{row.analysis_id}` / "
            f"`{str(row.target).replace('CytoBridge ', '')}`："
            f"{int(row.n_external_selected_stage_pathway_cells)} 个 external-selected "
            f"stage-pathway cells；rho={rho_label}"
        )
    metric_lines = "\n".join(metric_rows)
    text = f"""# Mouse-ortholog Reactome sensitivity：通俗说明

## 定位

这是 **{ANALYSIS_TIER}**，不是 native zebrafish primary result。

现有 CellAgentChat crosswalk 来自 Ensembl one-to-one、symbol-bijective mapping，
但 **orthology confidence 未过滤**（`{ORTHOLOGY_POLICY}`）。因此即使结果为正，
也只能作为跨物种敏感性和生物学 plausibility 支撑，不能替代 Danio rerio 主分析。

## 分析流程

1. 读取冻结的 zebrafish `query_manifest.csv`；可选的 `top_lr_pairs.csv`
   只用于核对 foreground gene set，不重新排名。
2. 从 `source_ligand/source_receptor -> mapped_ligand/mapped_receptor`
   构建严格 symbol-bijective gene map。
3. 固定使用 g:Profiler `{EXPECTED_GPROFILER_VERSION}`、`organism={ORGANISM}`；
   convert 后只有严格一对一 `ENSMUSG` ID 进入 foreground/background。
4. 同时输出：
   - annotated/default background + g:SCS；
   - shared mouse-ortholog custom background + FDR。
5. Native selection 使用 COMMOT、CellChat triMean、CellAgentChat significant；
   relaxed selection 使用 COMMOT、CellChat truncatedMean、CellAgentChat continuous。
   同一 stage 至少 {audit['min_external_methods']} 个外部方法 adjusted P<0.01
   才能选行；CytoBridge 完全不参与 discovery，但在 heatmap 中与外部方法一起展示。

## 映射审计

- Crosswalk source genes：{audit['crosswalk_source_genes']}
- 严格 source->mouse symbol map：{audit['strict_mouse_symbol_map']}
- Mouse symbols 经 e113 convert 后的严格 ENSMUSG：{audit['background_ensmusg']}
- Crosswalk 非一一状态：{
        int((~crosswalk_audit['status'].eq('one_to_one')).sum())
    }
- Convert ambiguous/failed：{
        int((~conversion_audit['status'].eq('one_to_one')).sum())
    }

## External-only 结果

- custom+FDR native：{
        selection_counts.get('mouse_custom_fdr_native', 0)
    } 个 pathway
- custom+FDR relaxed：{
        selection_counts.get('mouse_custom_fdr_relaxed', 0)
    } 个 pathway
- annotated+g:SCS native：{
        selection_counts.get('mouse_annotated_gscs_native', 0)
    } 个 pathway
- annotated+g:SCS relaxed：{
        selection_counts.get('mouse_annotated_gscs_relaxed', 0)
    } 个 pathway

若没有 pathway 入选，图不会生成大白板，而会显示每个 stage 各外部方法显著
pathway 数量和“至少两个外部方法共同支持”的数量。

## CytoBridge readout 对照

{metric_lines if metric_lines else '- 没有可计算的 external-selected cells。'}

没有 selected rows 时 rho 是“未定义”（机器表中记为 NaN），不是程序失败。

## 主要文件

- `query_manifest_mouse.csv`：zebrafish、mouse symbol 与 ENSMUSG foreground。
- `crosswalk_gene_map_audit.csv`、`mouse_conversion_audit.csv`：逐基因映射审计。
- `mouse_reactome_enrichment.csv.gz`：完整 Reactome 结果。
- `external_only_pathway_selection.csv`：external-only 选行明细。
- `cytobridge_readout_reactome_metrics.csv`：四种 CytoBridge readout 对照。
- `api/`：所有固定版本 request、raw response 和 HTTP metadata。
- `run_manifest.json`：输入、代码、参数、API profile 及全部输出 SHA256。

## 解释边界

Reactome ORA 丢掉 sender->receiver 方向、细胞类型和 LR pairing。所有方法仍共享表达
数据与 LR catalog。结果只能说明 top LR genes 在跨物种 Reactome pathway 层面的
计算一致性，不能证明独立实验真值或 attention-specific causal gain。
"""
    (output_dir / "README_CN.md").write_text(text, encoding="utf-8")


def run_analysis(
    args: argparse.Namespace,
    *,
    request_json: Callable[..., tuple[dict[str, Any] | list[Any], dict[str, Any]]] = (
        _request_json
    ),
) -> None:
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"Output directory is not empty: {output_dir}")
        shutil.rmtree(output_dir)
    api_dir = output_dir / "api"
    api_dir.mkdir(parents=True, exist_ok=True)

    query_input = pd.read_csv(args.query_manifest)
    crosswalk = pd.read_csv(args.cellagentchat_crosswalk)
    gene_map, crosswalk_audit = build_strict_unique_gene_map(crosswalk)
    query_manifest = map_query_manifest(query_input, gene_map)
    crosswalk_audit.to_csv(output_dir / "crosswalk_gene_map_audit.csv", index=False)
    if args.top_lr_pairs is not None:
        top_pairs = pd.read_csv(args.top_lr_pairs)
        top_pair_audit = validate_top_lr_pairs(top_pairs, query_manifest, crosswalk)
        top_pair_audit.to_csv(output_dir / "top_lr_pairs_validation.csv", index=False)

    background_mouse_symbols = sorted(set(gene_map.values()), key=str.casefold)
    conversion_request = {
        "organism": ORGANISM,
        "query": background_mouse_symbols,
        "target": "ENSG",
    }
    _write_json(api_dir / "convert_request.json", conversion_request)
    conversion_response, conversion_http = request_json(
        f"{ARCHIVE_BASE}/api/convert/convert/",
        payload=conversion_request,
        timeout=args.api_timeout_seconds,
    )
    _write_json(api_dir / "convert_response.json", conversion_response)
    _write_json(api_dir / "convert_http_metadata.json", conversion_http)
    conversion, conversion_audit = strict_mouse_one_to_one_conversion(
        conversion_response, background_mouse_symbols
    )
    if not conversion:
        raise ValueError("No strict ENSMUSG mappings survived e113 conversion")
    conversion_audit.to_csv(output_dir / "mouse_conversion_audit.csv", index=False)
    pd.DataFrame(
        sorted(conversion.items(), key=lambda item: item[0].casefold()),
        columns=["mouse_gene_symbol", "ensembl_mouse_gene_id"],
    ).to_csv(output_dir / "mouse_symbol_to_ensembl_1to1.csv", index=False)
    query_manifest = add_mouse_ensembl_queries(query_manifest, conversion)
    query_manifest.to_csv(output_dir / "query_manifest_mouse.csv", index=False)
    background_ensembl = sorted(set(conversion.values()))
    pd.DataFrame(
        [
            {
                "source_symbol_zebrafish": source,
                "mouse_gene_symbol": mapped,
                "ensembl_mouse_gene_id": conversion.get(mapped, ""),
                "included_strict_ensmusg": mapped in conversion,
            }
            for source, mapped in sorted(gene_map.items())
        ]
    ).to_csv(output_dir / "mouse_background_genes.csv", index=False)

    versions_url = f"{ARCHIVE_BASE}/api/util/data_versions?" + urlencode(
        {"organism": ORGANISM}
    )
    organism_url = f"{ARCHIVE_BASE}/api/util/organisms_list?" + urlencode(
        {"organism": ORGANISM, "extra_data": "True"}
    )
    versions, versions_http = request_json(
        versions_url, timeout=args.api_timeout_seconds
    )
    organism_metadata, organism_http = request_json(
        organism_url, timeout=args.api_timeout_seconds
    )
    _write_json(api_dir / "data_versions.json", versions)
    _write_json(api_dir / "data_versions_http_metadata.json", versions_http)
    _write_json(api_dir / "organism_metadata.json", organism_metadata)
    _write_json(api_dir / "organism_http_metadata.json", organism_http)

    enrichment_frames: list[pd.DataFrame] = []
    api_profiles: dict[str, Any] = {}
    ensembl_to_symbol = {value: key for key, value in conversion.items()}
    for spec in _profile_specs_with_background(background_ensembl):
        profile_id = str(spec["profile_id"])
        payload = build_mouse_profile_payload(
            query_manifest,
            domain_scope=str(spec["domain_scope"]),
            correction=str(spec["correction"]),
            background=spec["background"],
        )
        _write_json(api_dir / f"{profile_id}_request.json", payload)
        response, http_metadata = request_json(
            f"{ARCHIVE_BASE}/api/gost/profile/",
            payload=payload,
            timeout=args.api_timeout_seconds,
        )
        _write_json(api_dir / f"{profile_id}_response.json", response)
        _write_json(api_dir / f"{profile_id}_http_metadata.json", http_metadata)
        parsed = parse_profile_response(
            response,
            query_manifest,
            profile_id=profile_id,
            ensembl_to_symbol=ensembl_to_symbol,
            expected_background_size=(
                len(spec["background"]) if spec["background"] is not None else None
            ),
        )
        enrichment_frames.append(parsed)
        api_profiles[profile_id] = {
            "organism": ORGANISM,
            "profile_label": spec["profile_label"],
            "domain_scope": spec["domain_scope"],
            "correction": spec["correction"],
            "background_size": (
                len(spec["background"]) if spec["background"] is not None else None
            ),
            "response_version": response.get("meta", {}).get("version"),
            "response_timestamp": response.get("meta", {}).get("timestamp"),
            "n_result_rows": int(len(parsed)),
        }
    enrichment = pd.concat(enrichment_frames, ignore_index=True)
    enrichment.to_csv(output_dir / "mouse_reactome_enrichment.csv.gz", index=False)

    selection_specs = (
        (
            "mouse_custom_fdr_native",
            "mouse_custom_fdr_ligand_receptor",
            NATIVE_EXTERNAL,
            NATIVE_DISPLAY_METHODS,
        ),
        (
            "mouse_custom_fdr_relaxed",
            "mouse_custom_fdr_ligand_receptor",
            RELAXED_EXTERNAL,
            RELAXED_DISPLAY_METHODS,
        ),
        (
            "mouse_annotated_gscs_native",
            "mouse_annotated_gscs_ligand_receptor",
            NATIVE_EXTERNAL,
            NATIVE_DISPLAY_METHODS,
        ),
        (
            "mouse_annotated_gscs_relaxed",
            "mouse_annotated_gscs_ligand_receptor",
            RELAXED_EXTERNAL,
            RELAXED_DISPLAY_METHODS,
        ),
    )
    selection_frames: list[pd.DataFrame] = []
    selection_by_id: dict[str, pd.DataFrame] = {}
    for analysis_id, profile_id, external_methods, display_methods in selection_specs:
        selection = select_external_pathways(
            enrichment,
            profile_id=profile_id,
            external_methods=external_methods,
            all_methods=display_methods,
            min_external_methods=args.min_external_methods,
            max_pathways=args.max_pathways,
            analysis_id=analysis_id,
        )
        selection_frames.append(selection)
        selection_by_id[analysis_id] = selection
    selections = pd.concat(selection_frames, ignore_index=True)
    selections.to_csv(output_dir / "external_only_pathway_selection.csv", index=False)

    metric_frames: list[pd.DataFrame] = []
    for analysis_id, profile_id, external_methods, _ in selection_specs:
        metric = readout_consistency_metrics(
            enrichment,
            selection_by_id[analysis_id],
            profile_id=profile_id,
            external_methods=external_methods,
        )
        metric.insert(0, "analysis_id", analysis_id)
        metric_frames.append(metric)
    metrics = pd.concat(metric_frames, ignore_index=True)
    metrics.to_csv(output_dir / "cytobridge_readout_reactome_metrics.csv", index=False)

    figure_specs = (
        (
            "mouse_custom_fdr_native",
            "mouse_custom_fdr_ligand_receptor",
            NATIVE_EXTERNAL,
            NATIVE_DISPLAY_METHODS,
            "mouse_reactome_custom_fdr_native",
            "Post-hoc mouse-ortholog Reactome sensitivity - custom background + FDR",
        ),
        (
            "mouse_custom_fdr_relaxed",
            "mouse_custom_fdr_ligand_receptor",
            RELAXED_EXTERNAL,
            RELAXED_DISPLAY_METHODS,
            "mouse_reactome_custom_fdr_relaxed",
            "Post-hoc mouse-ortholog Reactome sensitivity - custom background + FDR (relaxed)",
        ),
        (
            "mouse_annotated_gscs_native",
            "mouse_annotated_gscs_ligand_receptor",
            NATIVE_EXTERNAL,
            NATIVE_DISPLAY_METHODS,
            "mouse_reactome_annotated_gscs_native",
            "Post-hoc mouse-ortholog Reactome sensitivity - annotated background + g:SCS",
        ),
        (
            "mouse_annotated_gscs_relaxed",
            "mouse_annotated_gscs_ligand_receptor",
            RELAXED_EXTERNAL,
            RELAXED_DISPLAY_METHODS,
            "mouse_reactome_annotated_gscs_relaxed",
            "Post-hoc mouse-ortholog Reactome sensitivity - annotated background + g:SCS (relaxed)",
        ),
    )
    for (
        analysis_id,
        profile_id,
        external_methods,
        display_methods,
        filename,
        title,
    ) in figure_specs:
        plot_sensitivity_heatmap(
            enrichment,
            query_manifest,
            selection_by_id[analysis_id],
            profile_id=profile_id,
            external_methods=external_methods,
            display_methods=display_methods,
            min_external_methods=args.min_external_methods,
            title=title,
            output_path=output_dir / filename,
        )
    plot_sensitivity_heatmap(
        enrichment,
        query_manifest,
        selection_by_id["mouse_custom_fdr_native"],
        profile_id="mouse_custom_fdr_ligand_receptor",
        external_methods=NATIVE_EXTERNAL,
        display_methods=CB_METHODS,
        min_external_methods=args.min_external_methods,
        title=(
            "Post-hoc mouse-ortholog Reactome sensitivity - "
            "CytoBridge readout ablation on external-only rows"
        ),
        output_path=output_dir / "mouse_reactome_cytobridge_readout_ablation",
    )

    audit = {
        "analysis_tier": ANALYSIS_TIER,
        "native_primary": False,
        "organism": ORGANISM,
        "gprofiler_archive_base": ARCHIVE_BASE,
        "gprofiler_expected_version": EXPECTED_GPROFILER_VERSION,
        "orthology_policy": ORTHOLOGY_POLICY,
        "orthology_confidence_filtered": False,
        "crosswalk_source_genes": int(len(crosswalk_audit)),
        "strict_mouse_symbol_map": int(len(gene_map)),
        "background_ensmusg": int(len(background_ensembl)),
        "external_pathway_selection_excludes_cytobridge": True,
        "min_external_methods": int(args.min_external_methods),
        "alpha_adjusted_p": ALPHA,
        "gene_mode": "ligand_receptor",
    }
    pd.DataFrame([audit]).to_csv(
        output_dir / "mouse_reactome_run_audit.csv", index=False
    )
    _write_readme(
        output_dir,
        selections,
        metrics,
        crosswalk_audit,
        conversion_audit,
        audit,
    )

    input_paths: dict[str, Path] = {
        "zebrafish_query_manifest": args.query_manifest,
        "cellagentchat_crosswalk": args.cellagentchat_crosswalk,
    }
    if args.top_lr_pairs is not None:
        input_paths["top_lr_pairs"] = args.top_lr_pairs
    output_files = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.name != "run_manifest.json"
    )
    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[2]
    dependency_script = script_path.with_name("reactome_pathway_consistency.py")
    indirect_dependency_script = script_path.with_name(
        "multimethod_pathway_consistency.py"
    )
    run_manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis": "zebrafish_mouse_ortholog_reactome_sensitivity",
        "analysis_tier": ANALYSIS_TIER,
        "inputs": {name: _record(path) for name, path in input_paths.items()},
        "code": {
            "entry_point": _record(script_path),
            "shared_reactome_helpers": _record(dependency_script),
            "indirect_multimethod_helpers": _record(
                indirect_dependency_script
            ),
            "git": _git_state(repo_root),
        },
        "command": [sys.executable, *sys.argv],
        "runtime": {
            "python": sys.version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "parameters": {
            "organism": ORGANISM,
            "archive_base": ARCHIVE_BASE,
            "expected_gprofiler_version": EXPECTED_GPROFILER_VERSION,
            "min_external_methods": args.min_external_methods,
            "max_pathways": args.max_pathways,
            "alpha_adjusted_p": ALPHA,
        },
        "audit": audit,
        "api_profiles": api_profiles,
        "claims": {
            "post_hoc_cross_species_sensitivity": True,
            "native_primary_result": False,
            "crosswalk_orthology_confidence_unfiltered": True,
            "external_pathway_discovery_excludes_cytobridge": True,
            "heatmap_display_includes_cytobridge_and_external_methods": True,
            "independent_experimental_validation": False,
            "attention_specific_incremental_value_not_assumed": True,
        },
        "outputs": {
            str(path.relative_to(output_dir)): _record(path) for path in output_files
        },
    }
    _write_json(output_dir / "run_manifest.json", run_manifest)


def main() -> None:
    run_analysis(_parser().parse_args())


if __name__ == "__main__":
    main()
