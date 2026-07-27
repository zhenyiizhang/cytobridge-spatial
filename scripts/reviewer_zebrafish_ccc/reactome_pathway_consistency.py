#!/usr/bin/env python3
"""Run a frozen g:Profiler Reactome comparison on zebrafish top LR genes.

This analysis follows the biological unit used by Raghavan et al. Fig. 2C:
rank LR pairs, split them into ligand/receptor genes, and test Reactome
pathway enrichment.  It differs deliberately in two ways:

1. pathway rows are selected using external methods only, so CytoBridge is
   never used to choose the pathways against which it is evaluated;
2. the primary analysis uses a shared LR-gene background and a frozen
   g:Profiler archive, making the statistical universe explicit and
   reproducible.

The paper's main text specifies ligand plus receptor genes, whereas a
supplementary caption says receptor-only.  Both are emitted, with
ligand-plus-receptor pre-specified as primary.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version as package_version
import json
import math
import platform
from pathlib import Path
import re
import shutil
import subprocess
import sys
import textwrap
import time
from typing import Any, Iterable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from multimethod_pathway_consistency import (
    CB_METHODS,
    STAGES,
    STAGE_LABELS,
    _axis,
    _collapse_lr_contexts,
    _require,
    _top_mask,
    build_score_grid,
    load_cellagentchat,
    load_crosswalk,
    load_cytobridge,
    load_database,
)


ARCHIVE_BASE = "https://biit.cs.ut.ee/gprofiler_archive3/e113_eg59_p19"
EXPECTED_GPROFILER_VERSION = "e113_eg59_p19_903d2221"
ORGANISM = "drerio"
ALPHA = 0.01

NATIVE_EXTERNAL = (
    "COMMOT",
    "CellChat triMean",
    "CellAgentChat significant",
)
RELAXED_EXTERNAL = (
    "COMMOT",
    "CellChat truncatedMean",
    "CellAgentChat continuous",
)
ALL_SCORE_METHODS = (
    *CB_METHODS,
    "COMMOT",
    "CellChat triMean",
    "CellChat truncatedMean",
    "CellAgentChat significant",
    "CellAgentChat continuous",
)
METHOD_CODES = {
    "CytoBridge attention x LR": "cb_attention",
    "CytoBridge exact message x LR": "cb_exact_lr",
    "CytoBridge exact message only (LR-conditioned)": "cb_exact_only",
    "CytoBridge LR-only": "cb_lr_only",
    "COMMOT": "commot",
    "CellChat triMean": "cellchat_primary",
    "CellChat truncatedMean": "cellchat_relaxed",
    "CellAgentChat significant": "cag_significant",
    "CellAgentChat continuous": "cag_continuous",
}
DISPLAY_NAMES = {
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cytobridge-axis-scores", required=True, type=Path)
    parser.add_argument("--commot-lr-scores", required=True, type=Path)
    parser.add_argument(
        "--commot-score-column",
        default="abundance_controlled_score",
        choices=[
            "abundance_controlled_score",
            "abundance_controlled_distinct_cell_score",
        ],
    )
    parser.add_argument("--cellchat-primary-lr-scores", required=True, type=Path)
    parser.add_argument("--cellchat-truncated-lr-scores", required=True, type=Path)
    parser.add_argument("--cellchat-excluded-lr", required=True, type=Path)
    parser.add_argument("--cellagentchat-raw-lr-scores", required=True, type=Path)
    parser.add_argument("--cellagentchat-significant-lr-scores", required=True, type=Path)
    parser.add_argument("--cellagentchat-crosswalk", required=True, type=Path)
    parser.add_argument("--lr-database", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--top-fraction", type=float, default=0.20)
    parser.add_argument("--min-external-methods", type=int, default=2)
    parser.add_argument("--max-pathways", type=int, default=24)
    parser.add_argument("--api-timeout-seconds", type=int, default=180)
    parser.add_argument("--overwrite", action="store_true")
    return parser


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


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _request_json(
    url: str,
    *,
    payload: Mapping[str, Any] | None = None,
    timeout: int = 180,
    attempts: int = 3,
) -> tuple[dict[str, Any] | list[Any], dict[str, Any]]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {
        "Accept": "application/json",
        "User-Agent": "CytoBridge-reviewer-analysis/1.0",
    }
    if body is not None:
        headers["Content-Type"] = "application/json"
    last_error: Exception | None = None
    for attempt in range(attempts):
        started = datetime.now(timezone.utc)
        request = Request(url, data=body, headers=headers, method="POST" if body else "GET")
        try:
            with urlopen(request, timeout=timeout) as response:
                raw = response.read()
                metadata = {
                    "url": url,
                    "http_status": int(response.status),
                    "requested_at_utc": started.isoformat(),
                    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                    "response_bytes": int(len(raw)),
                    "response_sha256": hashlib.sha256(raw).hexdigest(),
                }
                return json.loads(raw), metadata
        except (HTTPError, URLError, TimeoutError) as error:
            last_error = error
            if attempt + 1 < attempts:
                time.sleep(2**attempt)
    raise RuntimeError(f"g:Profiler request failed after {attempts} attempts: {last_error}")


def _components(value: str) -> list[str]:
    return sorted({part.strip().casefold() for part in str(value).split("_") if part.strip()})


def _axis_definitions(database: pd.DataFrame, universe: Sequence[str]) -> pd.DataFrame:
    local = (
        database.loc[
            database["axis"].isin(universe), ["axis", "ligand", "receptor"]
        ]
        .drop_duplicates()
        .sort_values(["axis", "ligand", "receptor"])
    )
    counts = local.groupby("axis")[["ligand", "receptor"]].nunique()
    if (counts > 1).any().any():
        raise ValueError("An LR axis has conflicting ligand/receptor definitions")
    return local.drop_duplicates("axis").reset_index(drop=True)


def build_query_manifest(
    grid: pd.DataFrame,
    axis_definitions: pd.DataFrame,
    *,
    methods: Sequence[str],
    top_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    definition = axis_definitions.set_index("axis")
    query_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    query_number = 0
    for method in methods:
        for stage in STAGES:
            scores = (
                grid.loc[grid["stage"].eq(stage), ["axis", method]]
                .set_index("axis")[method]
                .sort_index()
            )
            top, requested, selected, boundary = _top_mask(scores, top_fraction)
            selected_axes = scores.index[top].tolist()
            selected_frame = definition.loc[selected_axes] if selected_axes else definition.iloc[0:0]
            for axis_name in selected_axes:
                pair_rows.append(
                    {
                        "method": method,
                        "stage": float(stage),
                        "stage_label": STAGE_LABELS[stage],
                        "axis": axis_name,
                        "ligand": definition.loc[axis_name, "ligand"],
                        "receptor": definition.loc[axis_name, "receptor"],
                        "native_score": float(scores.loc[axis_name]),
                        "top_boundary_native_score": boundary,
                    }
                )
            ligand_genes = sorted(
                {
                    gene
                    for value in selected_frame["ligand"].astype(str)
                    for gene in _components(value)
                }
            )
            receptor_genes = sorted(
                {
                    gene
                    for value in selected_frame["receptor"].astype(str)
                    for gene in _components(value)
                }
            )
            for gene_mode, genes in (
                ("ligand_receptor", sorted(set(ligand_genes) | set(receptor_genes))),
                ("receptor_only", receptor_genes),
            ):
                query_number += 1
                query_rows.append(
                    {
                        "query_id": f"q{query_number:03d}",
                        "gene_mode": gene_mode,
                        "method": method,
                        "method_code": METHOD_CODES[method],
                        "stage": float(stage),
                        "stage_label": STAGE_LABELS[stage],
                        "n_universe_axes": int(len(scores)),
                        "n_positive_axes": int(scores.gt(0).sum()),
                        "top_k_requested": int(requested),
                        "top_k_after_positive_and_ties": int(selected),
                        "top_boundary_native_score": boundary,
                        "n_ligand_genes": int(len(ligand_genes)),
                        "n_receptor_genes": int(len(receptor_genes)),
                        "n_query_genes_symbols": int(len(genes)),
                        "query_gene_symbols": ";".join(genes),
                    }
                )
    return pd.DataFrame(query_rows), pd.DataFrame(pair_rows)


def strict_one_to_one_conversion(
    response: Mapping[str, Any],
    expected_symbols: Iterable[str],
    *,
    ensembl_pattern: str = r"ENSDARG\d+",
) -> tuple[dict[str, str], pd.DataFrame]:
    rows = pd.DataFrame(response.get("result", []))
    expected = sorted(set(map(str, expected_symbols)))
    converted: dict[str, str] = {}
    audit_rows: list[dict[str, Any]] = []
    for symbol in expected:
        if rows.empty:
            values: list[str] = []
        else:
            values = sorted(
                {
                    value
                    for value in (
                        rows.loc[
                            rows["incoming"].astype(str).eq(symbol), "converted"
                        ]
                        .dropna()
                        .astype(str)
                    )
                    if re.fullmatch(ensembl_pattern, value)
                }
            )
        status = "one_to_one" if len(values) == 1 else ("failed" if not values else "ambiguous")
        if status == "one_to_one":
            converted[symbol] = values[0]
        audit_rows.append(
            {
                "gene_symbol": symbol,
                "status": status,
                "n_converted_ids": int(len(values)),
                "converted_ensembl_ids": ";".join(values),
            }
        )
    return converted, pd.DataFrame(audit_rows)


def add_ensembl_queries(
    manifest: pd.DataFrame, conversion: Mapping[str, str]
) -> pd.DataFrame:
    result = manifest.copy()
    ensembl_values: list[str] = []
    excluded_values: list[str] = []
    n_values: list[int] = []
    for row in result.itertuples():
        symbols = [value for value in row.query_gene_symbols.split(";") if value]
        ensembl = sorted({conversion[value] for value in symbols if value in conversion})
        excluded = sorted(set(symbols) - set(conversion))
        ensembl_values.append(";".join(ensembl))
        excluded_values.append(";".join(excluded))
        n_values.append(len(ensembl))
    result["query_genes_ensembl"] = ensembl_values
    result["excluded_ambiguous_or_failed_symbols"] = excluded_values
    result["n_query_genes_ensembl_1to1"] = n_values
    result["api_queried"] = result["n_query_genes_ensembl_1to1"].gt(0)
    return result


def _background_symbols(
    axis_definitions: pd.DataFrame, gene_mode: str
) -> list[str]:
    receptors = {
        gene
        for value in axis_definitions["receptor"].astype(str)
        for gene in _components(value)
    }
    if gene_mode == "receptor_only":
        return sorted(receptors)
    ligands = {
        gene
        for value in axis_definitions["ligand"].astype(str)
        for gene in _components(value)
    }
    return sorted(ligands | receptors)


def build_profile_payload(
    manifest: pd.DataFrame,
    *,
    gene_mode: str,
    domain_scope: str,
    correction: str,
    background: Sequence[str] | None,
    organism: str = ORGANISM,
) -> dict[str, Any]:
    local = manifest.loc[
        manifest["gene_mode"].eq(gene_mode) & manifest["api_queried"]
    ]
    queries = {
        row.query_id: [value for value in row.query_genes_ensembl.split(";") if value]
        for row in local.itertuples()
    }
    payload: dict[str, Any] = {
        "organism": organism,
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


def _meta_has_mapping_failures(response: Mapping[str, Any]) -> bool:
    metadata = response.get("meta", {}).get("genes_metadata", {})
    failed = metadata.get("failed", [])
    ambiguous = metadata.get("ambiguous", {})
    return bool(failed) or bool(ambiguous)


def parse_profile_response(
    response: Mapping[str, Any],
    manifest: pd.DataFrame,
    *,
    profile_id: str,
    ensembl_to_symbol: Mapping[str, str],
    expected_background_size: int | None,
) -> pd.DataFrame:
    version = response.get("meta", {}).get("version")
    if version != EXPECTED_GPROFILER_VERSION:
        raise ValueError(
            f"Unexpected g:Profiler version {version!r}; expected "
            f"{EXPECTED_GPROFILER_VERSION!r}"
        )
    if _meta_has_mapping_failures(response):
        raise ValueError("Ensembl-ID g:Profiler query unexpectedly failed or was ambiguous")
    query_meta = (
        response.get("meta", {})
        .get("genes_metadata", {})
        .get("query", {})
    )
    manifest_index = manifest.set_index("query_id")
    rows: list[dict[str, Any]] = []
    for result in response.get("result", []):
        query_id = str(result["query"])
        if query_id not in manifest_index.index:
            raise ValueError(f"Unknown query label in g:Profiler response: {query_id}")
        query_row = manifest_index.loc[query_id]
        ensgs = query_meta.get(query_id, {}).get("ensgs", [])
        intersections = result.get("intersections", [])
        hit_ids = [
            ensg
            for ensg, marker in zip(ensgs, intersections)
            if marker
        ]
        effective_domain_size = int(result["effective_domain_size"])
        if (
            expected_background_size is not None
            and effective_domain_size != expected_background_size
        ):
            raise ValueError(
                f"Custom background mismatch for {query_id}: API used "
                f"{effective_domain_size}, submitted {expected_background_size}"
            )
        adjusted_p = float(result["p_value"])
        rows.append(
            {
                "profile_id": profile_id,
                "query_id": query_id,
                "gene_mode": query_row["gene_mode"],
                "method": query_row["method"],
                "stage": float(query_row["stage"]),
                "stage_label": query_row["stage_label"],
                "reactome_id": str(result["native"]),
                "reactome_name": str(result["name"]),
                "adjusted_p_value": adjusted_p,
                "minus_log10_adjusted_p": float(
                    -math.log10(max(adjusted_p, 1e-300))
                ),
                "significant_adjusted_p_lt_0_01": adjusted_p < ALPHA,
                "effective_domain_size": effective_domain_size,
                "query_size": int(result["query_size"]),
                "term_size": int(result["term_size"]),
                "intersection_size": int(result["intersection_size"]),
                "precision": float(result["precision"]),
                "recall": float(result["recall"]),
                "intersection_ensembl_ids": ";".join(hit_ids),
                "intersection_gene_symbols": ";".join(
                    sorted(
                        {
                            ensembl_to_symbol.get(identifier, identifier)
                            for identifier in hit_ids
                        }
                    )
                ),
                "is_reactome_root": str(result["native"]).endswith("0000000"),
            }
        )
    return pd.DataFrame(rows)


def select_external_pathways(
    enrichment: pd.DataFrame,
    *,
    profile_id: str,
    external_methods: Sequence[str],
    all_methods: Sequence[str],
    min_external_methods: int,
    max_pathways: int,
    analysis_id: str,
) -> pd.DataFrame:
    local = enrichment.loc[
        enrichment["profile_id"].eq(profile_id) & ~enrichment["is_reactome_root"]
    ]
    all_terms = local[["reactome_id", "reactome_name"]].drop_duplicates()
    rows: list[dict[str, Any]] = []
    for stage in STAGES:
        stage_local = local.loc[local["stage"].eq(stage)]
        for term in all_terms.itertuples(index=False):
            term_local = stage_local.loc[stage_local["reactome_id"].eq(term.reactome_id)]
            p_by_method = (
                term_local.set_index("method")["adjusted_p_value"].to_dict()
            )
            external_p = np.asarray(
                [p_by_method.get(method, 1.0) for method in external_methods],
                dtype=float,
            )
            all_p = np.asarray(
                [p_by_method.get(method, 1.0) for method in all_methods],
                dtype=float,
            )
            rows.append(
                {
                    "analysis_id": analysis_id,
                    "profile_id": profile_id,
                    "stage": float(stage),
                    "stage_label": STAGE_LABELS[stage],
                    "reactome_id": term.reactome_id,
                    "reactome_name": term.reactome_name,
                    "n_external_methods_significant": int((external_p < ALPHA).sum()),
                    "n_all_methods_significant": int((all_p < ALPHA).sum()),
                    "external_median_minus_log10_p": float(
                        np.median(-np.log10(np.clip(external_p, 1e-300, 1.0)))
                    ),
                    "external_mean_minus_log10_p": float(
                        np.mean(-np.log10(np.clip(external_p, 1e-300, 1.0)))
                    ),
                    "passes_external_only_rule": int((external_p < ALPHA).sum())
                    >= min_external_methods,
                }
            )
    candidates = pd.DataFrame(rows)
    global_summary = (
        candidates.groupby(["reactome_id", "reactome_name"], as_index=False)
        .agg(
            n_stages_passing_external_only=("passes_external_only_rule", "sum"),
            n_external_stage_method_significant=(
                "n_external_methods_significant",
                "sum",
            ),
            mean_external_minus_log10_p=(
                "external_mean_minus_log10_p",
                "mean",
            ),
            max_external_median_minus_log10_p=(
                "external_median_minus_log10_p",
                "max",
            ),
        )
    )
    global_summary = global_summary.loc[
        global_summary["n_stages_passing_external_only"].gt(0)
    ].sort_values(
        [
            "n_stages_passing_external_only",
            "n_external_stage_method_significant",
            "mean_external_minus_log10_p",
            "reactome_name",
        ],
        ascending=[False, False, False, True],
    )
    selected_ids = set(global_summary.head(max_pathways)["reactome_id"])
    candidates["selected_for_heatmap"] = candidates["reactome_id"].isin(selected_ids)
    candidates = candidates.merge(
        global_summary,
        on=["reactome_id", "reactome_name"],
        how="left",
        validate="many_to_one",
    )
    return candidates


def _matrix_for_figure(
    enrichment: pd.DataFrame,
    manifest: pd.DataFrame,
    selection: pd.DataFrame,
    *,
    profile_id: str,
    methods: Sequence[str],
) -> tuple[np.ndarray, list[str], list[str], list[str], np.ndarray]:
    selected = (
        selection.loc[selection["selected_for_heatmap"]]
        .drop_duplicates(["reactome_id", "reactome_name"])
        .sort_values(
            [
                "n_stages_passing_external_only",
                "n_external_stage_method_significant",
                "mean_external_minus_log10_p",
            ],
            ascending=False,
        )
    )
    pathway_ids = selected["reactome_id"].tolist()
    pathway_names = selected["reactome_name"].tolist()
    columns: list[tuple[float, str]] = [
        (stage, method) for stage in STAGES for method in methods
    ]
    matrix = np.zeros((len(pathway_ids), len(columns)), dtype=float)
    significance = np.zeros_like(matrix, dtype=bool)
    availability = np.ones_like(matrix, dtype=bool)
    local = enrichment.loc[enrichment["profile_id"].eq(profile_id)]
    lookup = {
        (row.reactome_id, float(row.stage), row.method): float(row.adjusted_p_value)
        for row in local.itertuples()
    }
    gene_mode = (
        "receptor_only" if "receptor_only" in profile_id else "ligand_receptor"
    )
    query_available = {
        (float(row.stage), row.method): bool(row.api_queried)
        for row in manifest.itertuples()
        if row.gene_mode == gene_mode
    }
    for row_index, pathway_id in enumerate(pathway_ids):
        for column_index, (stage, method) in enumerate(columns):
            if not query_available.get((stage, method), False):
                availability[row_index, column_index] = False
                matrix[row_index, column_index] = np.nan
                continue
            p_value = lookup.get((pathway_id, stage, method), 1.0)
            matrix[row_index, column_index] = -math.log10(max(p_value, 1e-300))
            significance[row_index, column_index] = p_value < ALPHA
    labels = [
        f"{STAGE_LABELS[stage]}\n{DISPLAY_NAMES[method]}" for stage, method in columns
    ]
    return matrix, pathway_names, labels, pathway_ids, significance


def plot_heatmap(
    enrichment: pd.DataFrame,
    manifest: pd.DataFrame,
    selection: pd.DataFrame,
    *,
    profile_id: str,
    methods: Sequence[str],
    output_path: Path,
    title: str,
    external_methods: Sequence[str] | None = None,
) -> None:
    matrix, names, labels, _, significance = _matrix_for_figure(
        enrichment,
        manifest,
        selection,
        profile_id=profile_id,
        methods=methods,
    )
    discovery_methods = tuple(
        external_methods
        if external_methods is not None
        else [method for method in methods if method not in CB_METHODS]
    )
    discovery_label = ", ".join(DISPLAY_NAMES.get(method, method) for method in discovery_methods)
    if not names:
        figure, axis = plt.subplots(figsize=(11.5, 4.8))
        profile_rows = enrichment.loc[
            enrichment["profile_id"].eq(profile_id)
            & ~enrichment["is_reactome_root"]
        ]
        x_positions = np.arange(len(STAGES), dtype=float)
        series: list[tuple[str, list[int]]] = []
        for method in discovery_methods:
            counts = [
                int(
                    profile_rows.loc[
                        profile_rows["stage"].eq(stage)
                        & profile_rows["method"].eq(method)
                        & profile_rows["adjusted_p_value"].lt(ALPHA),
                        "reactome_id",
                    ].nunique()
                )
                for stage in STAGES
            ]
            series.append((DISPLAY_NAMES.get(method, method), counts))
        overlap_counts = [
            int(
                selection.loc[
                    selection["stage"].eq(stage)
                    & selection["passes_external_only_rule"],
                    "reactome_id",
                ].nunique()
            )
            for stage in STAGES
        ]
        series.append((">=2 external overlap", overlap_counts))
        width = 0.8 / max(1, len(series))
        colors = ["#7F8C8D", "#4C78A8", "#F28E2B", "#B2182B"]
        maximum = max([count for _, counts in series for count in counts] + [0])
        for index, (label, counts) in enumerate(series):
            offsets = x_positions - 0.4 + width / 2 + index * width
            bars = axis.bar(
                offsets,
                counts,
                width=width,
                label=label,
                color=colors[index % len(colors)],
            )
            axis.bar_label(bars, padding=2, fontsize=8)
        axis.set_xticks(x_positions)
        axis.set_xticklabels([STAGE_LABELS[stage] for stage in STAGES])
        axis.set_ylabel(f"Reactome terms with adjusted P < {ALPHA:g}")
        axis.set_ylim(0, max(1.0, maximum * 1.28 + 0.2))
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#E5E7EB", linewidth=0.8)
        axis.set_axisbelow(True)
        axis.legend(frameon=False, ncol=min(4, len(series)), loc="upper left")
        axis.set_title(
            "Negative strict control: no term was significant in >=2 external methods\n"
            f"Discovery methods only: {discovery_label or 'external consensus supplied by the selection table'}; "
            "CytoBridge was excluded from row discovery",
            fontsize=10,
            pad=12,
        )
        figure.suptitle(title, fontsize=15, fontweight="bold", y=0.99)
        figure.subplots_adjust(left=0.09, right=0.98, bottom=0.15, top=0.76)
    else:
        height = max(8.5, 0.43 * len(names) + 4.0)
        figure, axis = plt.subplots(figsize=(20, height))
        cmap = plt.get_cmap("Reds").copy()
        cmap.set_bad("#D1D5DB")
        finite = matrix[np.isfinite(matrix)]
        vmax = min(30.0, max(2.0, float(np.quantile(finite, 0.98)))) if len(finite) else 2.0
        image = axis.imshow(matrix, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
        axis.set_yticks(range(len(names)))
        axis.set_yticklabels(
            [textwrap.fill(name, width=43) for name in names],
            fontsize=9,
        )
        axis.set_xticks(range(len(labels)))
        axis.set_xticklabels(labels, rotation=52, ha="right", fontsize=8)
        for stage_index in range(1, len(STAGES)):
            axis.axvline(stage_index * len(methods) - 0.5, color="#6B7280", linewidth=0.8)
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                if significance[row, column]:
                    axis.text(
                        column,
                        row,
                        "*",
                        ha="center",
                        va="center",
                        color="#111827",
                        fontsize=9,
                    )
        axis.set_title(
            f"Row discovery: {discovery_label} only; CytoBridge excluded from discovery "
            "but included in display\n"
            f"Selection: >=2 external methods with adjusted P < {ALPHA:g}; "
            f"* adjusted P < {ALPHA:g}; grey = no positive LR query",
            fontsize=10,
            pad=12,
        )
        colorbar = figure.colorbar(image, ax=axis, fraction=0.025, pad=0.015)
        colorbar.set_label("-log10 adjusted P", fontsize=10)
        figure.suptitle(title, fontsize=15, fontweight="bold", y=0.995)
        figure.subplots_adjust(left=0.30, right=0.94, bottom=0.19, top=0.90)
    figure.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def readout_consistency_metrics(
    enrichment: pd.DataFrame,
    selection: pd.DataFrame,
    *,
    profile_id: str,
    external_methods: Sequence[str],
) -> pd.DataFrame:
    selected = selection.loc[selection["selected_for_heatmap"]]
    selected_keys = set(
        zip(selected["stage"].astype(float), selected["reactome_id"].astype(str))
    )
    local = enrichment.loc[
        enrichment["profile_id"].eq(profile_id)
        & ~enrichment["is_reactome_root"]
    ]
    lookup = {
        (float(row.stage), row.reactome_id, row.method): float(row.adjusted_p_value)
        for row in local.itertuples()
    }
    rows: list[dict[str, Any]] = []
    for target in CB_METHODS:
        x: list[float] = []
        y: list[float] = []
        n_target_significant = 0
        for stage, pathway_id in sorted(selected_keys):
            target_p = lookup.get((stage, pathway_id, target), 1.0)
            external_p = np.asarray(
                [
                    lookup.get((stage, pathway_id, method), 1.0)
                    for method in external_methods
                ],
                dtype=float,
            )
            x.append(-math.log10(max(target_p, 1e-300)))
            y.append(float(np.median(-np.log10(np.clip(external_p, 1e-300, 1.0)))))
            n_target_significant += target_p < ALPHA
        if len(x) >= 3 and np.unique(x).size > 1 and np.unique(y).size > 1:
            statistic = float(spearmanr(x, y).statistic)
        else:
            statistic = float("nan")
        rows.append(
            {
                "profile_id": profile_id,
                "target": target,
                "n_external_selected_stage_pathway_cells": int(len(x)),
                "n_target_significant_stage_pathway_cells": int(n_target_significant),
                "fraction_target_significant": (
                    float(n_target_significant / len(x)) if x else float("nan")
                ),
                "spearman_rho_vs_external_median_minus_log10_p": statistic,
            }
        )
    return pd.DataFrame(rows)


def write_readme(
    output_dir: Path,
    selections: pd.DataFrame,
    metrics: pd.DataFrame,
    conversion_audit: pd.DataFrame,
    audit: Mapping[str, Any],
) -> None:
    primary_selection = selections.loc[
        selections["analysis_id"].eq("custom_fdr_ligand_receptor_native")
        & selections["selected_for_heatmap"]
    ]
    selected_terms = (
        primary_selection.drop_duplicates(["reactome_id", "reactome_name"])
        .sort_values(
            [
                "n_stages_passing_external_only",
                "n_external_stage_method_significant",
                "mean_external_minus_log10_p",
            ],
            ascending=False,
        )
    )
    term_lines = "\n".join(
        f"- {row.reactome_name} (`{row.reactome_id}`): "
        f"{int(row.n_stages_passing_external_only)} stage(s) passed the "
        "external-only rule"
        for row in selected_terms.head(12).itertuples()
    )
    metric_lines_list: list[str] = []
    for row in metrics.loc[
        metrics["profile_id"].eq("custom_fdr_ligand_receptor")
    ].itertuples():
        denominator = int(row.n_external_selected_stage_pathway_cells)
        if denominator == 0:
            metric_lines_list.append(
                f"- `{row.target.replace('CytoBridge ', '')}`: 未定义；"
                "external-only 没有选出 pathway 行（0/0），不是 rho=0。"
            )
        else:
            metric_lines_list.append(
                f"- `{row.target.replace('CytoBridge ', '')}`: rho "
                f"{row.spearman_rho_vs_external_median_minus_log10_p:.3f}, "
                f"{int(row.n_target_significant_stage_pathway_cells)}/"
                f"{denominator} selected stage-pathway cells significant"
            )
    metric_lines = "\n".join(metric_lines_list)
    status_counts = conversion_audit["status"].value_counts().to_dict()
    text = f"""# 斑马鱼 Reactome pathway consistency：通俗说明

## 这次真正换成了什么

这不是 EPHB、THBS、COLLAGEN 这类 LR-database family 标签。现在严格按照论文
Fig. 2C 的核心流程：

1. 每个 method x stage 在共同的 {audit['shared_lr_axes']} 个 LR axes 中取 top
   {100 * audit['top_fraction']:.0f}%（目标 {audit['top_k_requested']} 个；只保留
   正分数并保留边界并列）；
2. 把 LR pair 拆成去重后的 ligand 和 receptor genes；
3. 用固定 g:Profiler `{EXPECTED_GPROFILER_VERSION}` 做 Danio rerio Reactome
   over-representation analysis；
4. pathway 行完全由外部方法选择：同一 stage 至少两个外部方法 adjusted P<0.01，
   不使用 CytoBridge 挑行；
5. 热图颜色是 `-log10(adjusted P)`。

原文 top 100 是在更大的 CellTalkDB universe 中使用。我们的四方法共同 universe
只有 {audit['shared_lr_axes']} 个 LR，若强行 top 100 会选走 74.6% 的背景，所以主分析
使用预先固定的 top 20%（{audit['top_k_requested']} 个）。这是保持鉴别力所必需的
适配，不声称逐参数复刻原文。

## 严格主结果

固定版本 + 共同 LR-gene custom background + FDR 的 ligand+receptor 主分析共选择
**{selected_terms['reactome_id'].nunique()} 个 Reactome pathways**：

{term_lines if term_lines else '- 没有 pathway 通过预设的 external-only 规则。'}

四种 CytoBridge readout 与外部 Reactome profile 的直接比较：

{metric_lines}

没有入选行表示这一严格检验在当前共同 universe 中没有足够证据；它不是程序失败，
也不能反过来证明各方法在生物学上“不一致”。

## 五套输出分别是什么

- `reactome_custom_background_ligand_receptor_native.png`：主结果。共同 LR-gene
  background，ligand+receptor，CellChat triMean 和 CAG significant。
- `reactome_custom_background_ligand_receptor_relaxed.png`：CellChat truncatedMean
  和 CAG continuous 的 sensitivity。
- `reactome_custom_background_receptor_only_native.png`：解决论文内部口径矛盾；
  主文写 ligand+receptor，但 Supplementary Fig. 7 caption 写 receptor-only。
- `reactome_paper_default_background_native.png`：更接近论文未报告 custom background
  时的 g:Profiler default-background 结果。注意它同时把 `custom + FDR` 改成了
  `annotated + g:SCS`，因此不能把两套 P 值差异单独归因于 background。
- `reactome_cytobridge_readout_ablation.png`：在完全相同的 external-only pathways
  上比较 attention x LR、exact x LR、LR-conditioned exact-only 和 LR-only。

## 基因标识和复现

没有把模糊 zebrafish symbols 暗中映射到第一个 Ensembl gene。全部
{audit['background_gene_symbols']} 个共同背景 symbols 先用同一固定 g:Profiler
版本转换，只保留严格一对一映射：

- one-to-one: {status_counts.get('one_to_one', 0)}
- ambiguous: {status_counts.get('ambiguous', 0)}
- failed: {status_counts.get('failed', 0)}

被排除的 ambiguous/failed symbols 在 `gene_conversion_audit.csv` 中逐个列出；
foreground 和 background 使用同一映射表。所有 API request、raw response、
版本 metadata 和 SHA256 均已保存。

## 能和不能支持什么

这能检验不同方法的 top LR genes 是否指向相似的、名称更完整的 Reactome biological
processes。它比 LR-family 标签更接近原论文，也更利于生物学解释。

本次严格 native-zebrafish 结果没有形成跨外部方法的显著 Reactome 共识，因此不能把
这五张图当成正向 consistency 主证据。可将其保留为透明的 negative/robustness
control；正向一致性应主要依据预设的 LR-level rank、score 和 top-signal overlap。

但 Reactome ORA 会丢掉 sender->receiver 方向、细胞类型和 LR pair 配对关系；所有方法
还共享表达数据与 LR catalog。因此它是 pathway-level computational consistency /
biological plausibility，不是独立实验验证，也不能单独证明 attention-specific gain。

## 关键明细

- `top_lr_pairs.csv`：每个 method x stage 实际进入 top 集合的 LR。
- `query_manifest.csv`：每次 enrichment 的 gene set、正分数数目和映射损失。
- `reactome_enrichment.csv.gz`：所有 g:Profiler Reactome 结果和命中 genes。
- `external_only_pathway_selection.csv`：哪些 pathway 为什么被选入。
- `cytobridge_readout_reactome_metrics.csv`：四种 readout 的直接比较。
- `api/`：conversion/profile request、raw response、版本信息和 HTTP provenance。
- `run_manifest.json`：输入、参数、API 和全部输出文件的 SHA256。
"""
    (output_dir / "README_CN.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = _parser().parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"Output directory is not empty: {output_dir}")
        shutil.rmtree(output_dir)
    api_dir = output_dir / "api"
    api_dir.mkdir(parents=True, exist_ok=True)

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
    cellchat_relaxed = _collapse_lr_contexts(
        args.cellchat_truncated_lr_scores, "score", "CellChat truncatedMean"
    )
    cag_significant = load_cellagentchat(
        args.cellagentchat_significant_lr_scores,
        crosswalk,
        "cellagentchat_score",
        "CellAgentChat significant",
    )
    cag_continuous = load_cellagentchat(
        args.cellagentchat_raw_lr_scores,
        crosswalk,
        "cellagentchat_score_raw",
        "CellAgentChat continuous",
    )
    native_axes = (
        set(database["axis"]) & set(cytobridge["axis"])
    ) - excluded_axes
    shared_axes = sorted(native_axes & set(crosswalk["axis"]))
    if len(shared_axes) != len(crosswalk):
        raise ValueError("CellAgentChat crosswalk is not fully represented in shared axes")
    grid = build_score_grid(
        shared_axes,
        cytobridge,
        {
            "COMMOT": commot,
            "CellChat triMean": cellchat_primary,
            "CellChat truncatedMean": cellchat_relaxed,
            "CellAgentChat significant": cag_significant,
            "CellAgentChat continuous": cag_continuous,
        },
    )
    axis_definitions = _axis_definitions(database, shared_axes)
    manifest, top_pairs = build_query_manifest(
        grid,
        axis_definitions,
        methods=ALL_SCORE_METHODS,
        top_fraction=args.top_fraction,
    )

    all_background_symbols = _background_symbols(
        axis_definitions, "ligand_receptor"
    )
    conversion_request = {
        "organism": ORGANISM,
        "query": all_background_symbols,
        "target": "ENSG",
    }
    _write_json(api_dir / "convert_request.json", conversion_request)
    conversion_response, conversion_http = _request_json(
        f"{ARCHIVE_BASE}/api/convert/convert/",
        payload=conversion_request,
        timeout=args.api_timeout_seconds,
    )
    _write_json(api_dir / "convert_response.json", conversion_response)
    _write_json(api_dir / "convert_http_metadata.json", conversion_http)
    conversion, conversion_audit = strict_one_to_one_conversion(
        conversion_response, all_background_symbols
    )
    conversion_audit.to_csv(output_dir / "gene_conversion_audit.csv", index=False)
    pd.DataFrame(
        sorted(conversion.items()), columns=["gene_symbol", "ensembl_gene_id"]
    ).to_csv(output_dir / "symbol_to_ensembl_1to1.csv", index=False)

    manifest = add_ensembl_queries(manifest, conversion)
    manifest.to_csv(output_dir / "query_manifest.csv", index=False)
    top_pairs.to_csv(output_dir / "top_lr_pairs.csv", index=False)
    background_rows: list[dict[str, Any]] = []
    backgrounds: dict[str, list[str]] = {}
    for gene_mode in ("ligand_receptor", "receptor_only"):
        symbols = _background_symbols(axis_definitions, gene_mode)
        ensembl = sorted({conversion[symbol] for symbol in symbols if symbol in conversion})
        backgrounds[gene_mode] = ensembl
        for symbol in symbols:
            background_rows.append(
                {
                    "gene_mode": gene_mode,
                    "gene_symbol": symbol,
                    "ensembl_gene_id": conversion.get(symbol, ""),
                    "included_strict_one_to_one": symbol in conversion,
                }
            )
    pd.DataFrame(background_rows).to_csv(
        output_dir / "background_genes.csv", index=False
    )

    versions_url = (
        f"{ARCHIVE_BASE}/api/util/data_versions?"
        + urlencode({"organism": ORGANISM})
    )
    organism_url = (
        f"{ARCHIVE_BASE}/api/util/organisms_list?"
        + urlencode({"organism": ORGANISM, "extra_data": "True"})
    )
    versions, versions_http = _request_json(
        versions_url, timeout=args.api_timeout_seconds
    )
    organism_metadata, organism_http = _request_json(
        organism_url, timeout=args.api_timeout_seconds
    )
    _write_json(api_dir / "data_versions.json", versions)
    _write_json(api_dir / "data_versions_http_metadata.json", versions_http)
    _write_json(api_dir / "organism_metadata.json", organism_metadata)
    _write_json(api_dir / "organism_http_metadata.json", organism_http)

    profile_specs = (
        {
            "profile_id": "custom_fdr_ligand_receptor",
            "gene_mode": "ligand_receptor",
            "domain_scope": "custom",
            "correction": "fdr",
            "background": backgrounds["ligand_receptor"],
        },
        {
            "profile_id": "custom_fdr_receptor_only",
            "gene_mode": "receptor_only",
            "domain_scope": "custom",
            "correction": "fdr",
            "background": backgrounds["receptor_only"],
        },
        {
            "profile_id": "annotated_gscs_ligand_receptor",
            "gene_mode": "ligand_receptor",
            "domain_scope": "annotated",
            "correction": "g_SCS",
            "background": None,
        },
    )
    ensembl_to_symbol = {value: key for key, value in conversion.items()}
    enrichment_frames: list[pd.DataFrame] = []
    api_provenance: dict[str, Any] = {}
    for spec in profile_specs:
        payload = build_profile_payload(
            manifest,
            gene_mode=spec["gene_mode"],
            domain_scope=spec["domain_scope"],
            correction=spec["correction"],
            background=spec["background"],
        )
        profile_id = str(spec["profile_id"])
        _write_json(api_dir / f"{profile_id}_request.json", payload)
        response, http_metadata = _request_json(
            f"{ARCHIVE_BASE}/api/gost/profile/",
            payload=payload,
            timeout=args.api_timeout_seconds,
        )
        _write_json(api_dir / f"{profile_id}_response.json", response)
        _write_json(api_dir / f"{profile_id}_http_metadata.json", http_metadata)
        parsed = parse_profile_response(
            response,
            manifest,
            profile_id=profile_id,
            ensembl_to_symbol=ensembl_to_symbol,
            expected_background_size=(
                len(spec["background"]) if spec["background"] is not None else None
            ),
        )
        enrichment_frames.append(parsed)
        api_provenance[profile_id] = {
            "gene_mode": spec["gene_mode"],
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
    enrichment.to_csv(output_dir / "reactome_enrichment.csv.gz", index=False)

    selection_specs = (
        (
            "custom_fdr_ligand_receptor_native",
            "custom_fdr_ligand_receptor",
            NATIVE_EXTERNAL,
            (
                "CytoBridge attention x LR",
                "COMMOT",
                "CellChat triMean",
                "CellAgentChat significant",
            ),
        ),
        (
            "custom_fdr_ligand_receptor_relaxed",
            "custom_fdr_ligand_receptor",
            RELAXED_EXTERNAL,
            (
                "CytoBridge attention x LR",
                "COMMOT",
                "CellChat truncatedMean",
                "CellAgentChat continuous",
            ),
        ),
        (
            "custom_fdr_receptor_only_native",
            "custom_fdr_receptor_only",
            NATIVE_EXTERNAL,
            (
                "CytoBridge attention x LR",
                "COMMOT",
                "CellChat triMean",
                "CellAgentChat significant",
            ),
        ),
        (
            "annotated_gscs_ligand_receptor_native",
            "annotated_gscs_ligand_receptor",
            NATIVE_EXTERNAL,
            (
                "CytoBridge attention x LR",
                "COMMOT",
                "CellChat triMean",
                "CellAgentChat significant",
            ),
        ),
    )
    selection_frames: list[pd.DataFrame] = []
    selection_by_id: dict[str, pd.DataFrame] = {}
    for analysis_id, profile_id, external_methods, all_methods in selection_specs:
        selection = select_external_pathways(
            enrichment,
            profile_id=profile_id,
            external_methods=external_methods,
            all_methods=all_methods,
            min_external_methods=args.min_external_methods,
            max_pathways=args.max_pathways,
            analysis_id=analysis_id,
        )
        selection_frames.append(selection)
        selection_by_id[analysis_id] = selection
    selections = pd.concat(selection_frames, ignore_index=True)
    selections.to_csv(
        output_dir / "external_only_pathway_selection.csv", index=False
    )

    figure_specs = (
        (
            "custom_fdr_ligand_receptor_native",
            "custom_fdr_ligand_receptor",
            (
                "CytoBridge attention x LR",
                "COMMOT",
                "CellChat triMean",
                "CellAgentChat significant",
            ),
            "reactome_custom_background_ligand_receptor_native",
            "Reactome consistency - ligand + receptor genes, shared LR background",
        ),
        (
            "custom_fdr_ligand_receptor_relaxed",
            "custom_fdr_ligand_receptor",
            (
                "CytoBridge attention x LR",
                "COMMOT",
                "CellChat truncatedMean",
                "CellAgentChat continuous",
            ),
            "reactome_custom_background_ligand_receptor_relaxed",
            "Reactome consistency - continuous-score sensitivity",
        ),
        (
            "custom_fdr_receptor_only_native",
            "custom_fdr_receptor_only",
            (
                "CytoBridge attention x LR",
                "COMMOT",
                "CellChat triMean",
                "CellAgentChat significant",
            ),
            "reactome_custom_background_receptor_only_native",
            "Reactome consistency - receptor-only sensitivity",
        ),
        (
            "annotated_gscs_ligand_receptor_native",
            "annotated_gscs_ligand_receptor",
            (
                "CytoBridge attention x LR",
                "COMMOT",
                "CellChat triMean",
                "CellAgentChat significant",
            ),
            "reactome_paper_default_background_native",
            "Reactome consistency - paper-default-background sensitivity",
        ),
    )
    for analysis_id, profile_id, methods, filename, title in figure_specs:
        plot_heatmap(
            enrichment,
            manifest,
            selection_by_id[analysis_id],
            profile_id=profile_id,
            methods=methods,
            output_path=output_dir / filename,
            title=title,
            external_methods=tuple(
                method for method in methods if method not in CB_METHODS
            ),
        )
    plot_heatmap(
        enrichment,
        manifest,
        selection_by_id["custom_fdr_ligand_receptor_native"],
        profile_id="custom_fdr_ligand_receptor",
        methods=CB_METHODS,
        output_path=output_dir / "reactome_cytobridge_readout_ablation",
        title="CytoBridge readout ablation on external-only Reactome pathways",
        external_methods=NATIVE_EXTERNAL,
    )

    metrics = pd.concat(
        [
            readout_consistency_metrics(
                enrichment,
                selection_by_id["custom_fdr_ligand_receptor_native"],
                profile_id="custom_fdr_ligand_receptor",
                external_methods=NATIVE_EXTERNAL,
            ),
            readout_consistency_metrics(
                enrichment,
                selection_by_id["custom_fdr_ligand_receptor_relaxed"],
                profile_id="custom_fdr_ligand_receptor",
                external_methods=RELAXED_EXTERNAL,
            ).assign(profile_id="custom_fdr_ligand_receptor_relaxed_consensus"),
        ],
        ignore_index=True,
    )
    metrics.to_csv(
        output_dir / "cytobridge_readout_reactome_metrics.csv", index=False
    )

    top_k_requested = int(math.ceil(len(shared_axes) * args.top_fraction))
    audit = {
        "shared_lr_axes": int(len(shared_axes)),
        "top_fraction": float(args.top_fraction),
        "top_k_requested": top_k_requested,
        "background_gene_symbols": int(len(all_background_symbols)),
        "background_ligand_receptor_ensembl_1to1": int(
            len(backgrounds["ligand_receptor"])
        ),
        "background_receptor_only_ensembl_1to1": int(
            len(backgrounds["receptor_only"])
        ),
        "external_consensus_excludes_cytobridge": True,
        "primary_gene_mode": "ligand_receptor",
        "paper_caption_sensitivity_gene_mode": "receptor_only",
        "alpha_adjusted_p": ALPHA,
        "gprofiler_archive_base": ARCHIVE_BASE,
        "gprofiler_expected_version": EXPECTED_GPROFILER_VERSION,
        "organism": ORGANISM,
        "cellagentchat_projection_is_cross_species_sensitivity": True,
    }
    pd.DataFrame([audit]).to_csv(output_dir / "reactome_run_audit.csv", index=False)
    write_readme(output_dir, selections, metrics, conversion_audit, audit)

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
        for path in output_dir.rglob("*")
        if path.is_file() and path.name != "run_manifest.json"
    )
    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[2]
    imported_script_path = script_path.with_name(
        "multimethod_pathway_consistency.py"
    )
    manifest_json = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis": "zebrafish_gprofiler_reactome_pathway_consistency",
        "command": [sys.executable, *sys.argv],
        "code": {
            "analysis_script": _record(script_path),
            "imported_analysis_script": _record(imported_script_path),
            "git": _git_state(repo_root),
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
            "scipy": package_version("scipy"),
        },
        "paper_reference": {
            "doi": "10.1101/gr.279771.124",
            "fig2c_core": (
                "top LR pairs -> ligand/receptor genes -> g:Profiler Reactome"
            ),
            "reported_ambiguity": (
                "main text says ligand+receptor; Supplementary Fig. 7 caption "
                "says receptor-only"
            ),
        },
        "inputs": {name: _record(path) for name, path in input_paths.items()},
        "parameters": {
            "top_fraction": args.top_fraction,
            "min_external_methods": args.min_external_methods,
            "max_pathways": args.max_pathways,
            "commot_score_column": args.commot_score_column,
            "alpha_adjusted_p": ALPHA,
        },
        "audit": audit,
        "api_profiles": api_provenance,
        "claims": {
            "is_true_reactome_gene_enrichment": True,
            "external_pathway_selection_excludes_cytobridge": True,
            "independent_experimental_validation": False,
            "attention_specific_incremental_value_not_assumed": True,
        },
        "outputs": {
            str(path.relative_to(output_dir)): _record(path) for path in output_files
        },
    }
    _write_json(output_dir / "run_manifest.json", manifest_json)


if __name__ == "__main__":
    main()
