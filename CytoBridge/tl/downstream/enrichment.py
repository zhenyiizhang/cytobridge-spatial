"""Dataset-agnostic offline gene-set over-representation analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "GeneSetLibrary",
    "load_gmt_gene_sets",
    "make_gene_set_library",
    "overrepresentation_analysis",
]


@dataclass(frozen=True)
class GeneSetLibrary:
    """Normalized term-to-gene mapping plus source provenance."""

    gene_sets: Mapping[str, frozenset[str]]
    descriptions: Mapping[str, str]
    metadata: Mapping[str, object]


def _normalize_gene(value: object, *, uppercase: bool) -> str:
    text = str(value).strip()
    return text.upper() if uppercase else text


def make_gene_set_library(
    gene_sets: Mapping[str, Sequence[str]],
    *,
    descriptions: Optional[Mapping[str, str]] = None,
    metadata: Optional[Mapping[str, object]] = None,
    uppercase: bool = True,
) -> GeneSetLibrary:
    """Validate a caller-supplied mapping as a portable gene-set library."""
    normalized: dict[str, frozenset[str]] = {}
    normalized_descriptions: dict[str, str] = {}
    descriptions = descriptions or {}
    for raw_term, raw_genes in gene_sets.items():
        term = str(raw_term).strip()
        if not term:
            raise ValueError("Gene-set terms must be non-empty strings.")
        genes = frozenset(
            gene
            for gene in (
                _normalize_gene(value, uppercase=uppercase) for value in raw_genes
            )
            if gene
        )
        if not genes:
            continue
        if term in normalized:
            raise ValueError(f"Duplicate gene-set term '{term}'.")
        normalized[term] = genes
        normalized_descriptions[term] = str(descriptions.get(raw_term, "")).strip()
    if not normalized:
        raise ValueError("The gene-set library contains no non-empty terms.")
    library_metadata = dict(metadata or {})
    library_metadata.update(
        {
            "n_terms": int(len(normalized)),
            "uppercase": bool(uppercase),
        }
    )
    return GeneSetLibrary(
        gene_sets=normalized,
        descriptions=normalized_descriptions,
        metadata=library_metadata,
    )


def load_gmt_gene_sets(
    path: str | Path,
    *,
    uppercase: bool = True,
) -> GeneSetLibrary:
    """Load a GMT file without requiring an online enrichment service."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"GMT gene-set library not found: {source}")
    gene_sets: dict[str, list[str]] = {}
    descriptions: dict[str, str] = {}
    with source.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.rstrip("\r\n")
            if not line:
                continue
            fields = line.split("\t")
            if len(fields) < 3:
                raise ValueError(
                    f"GMT line {line_number} has {len(fields)} fields; expected at "
                    "least term, description, and one gene."
                )
            term = fields[0].strip()
            if not term:
                raise ValueError(f"GMT line {line_number} has an empty term.")
            if term in gene_sets:
                raise ValueError(f"GMT contains duplicate term '{term}'.")
            gene_sets[term] = fields[2:]
            descriptions[term] = fields[1].strip()
    return make_gene_set_library(
        gene_sets,
        descriptions=descriptions,
        metadata={
            "format": "gmt",
            "source": str(source),
        },
        uppercase=uppercase,
    )


def _benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1:
        raise ValueError("p_values must be one-dimensional.")
    if not np.isfinite(values).all() or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("p_values must be finite values between zero and one.")
    if values.size == 0:
        return values.copy()
    order = np.argsort(values, kind="mergesort")
    ranked = values[order]
    adjusted_ranked = ranked * float(values.size) / np.arange(1, values.size + 1)
    adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = np.minimum(adjusted_ranked, 1.0)
    return adjusted


def _term_parts(term: str) -> tuple[str, str]:
    match = re.search(r"\((GO:\d+)\)\s*$", str(term), flags=re.IGNORECASE)
    if match is None:
        return str(term), ""
    name = str(term)[: match.start()].strip()
    return name, match.group(1).upper()


def overrepresentation_analysis(
    query_genes: Sequence[str],
    gene_sets: GeneSetLibrary | Mapping[str, Sequence[str]],
    *,
    background_genes: Optional[Sequence[str]] = None,
    min_set_size: int = 5,
    max_set_size: Optional[int] = 5000,
    min_overlap: int = 1,
    alpha: float = 0.05,
    uppercase: bool = True,
    multiple_testing_scope: str = "reported",
) -> pd.DataFrame:
    """Run one-sided hypergeometric ORA with Benjamini-Hochberg correction.

    The tested universe is explicit. If ``background_genes`` is omitted, the
    union of the supplied library is used. Query genes outside that universe
    are excluded and their counts are reported through the returned columns.

    ``multiple_testing_scope='reported'`` preserves the historical contract:
    terms below ``min_overlap`` are omitted and BH correction covers only the
    returned terms. ``multiple_testing_scope='all_eligible'`` instead returns
    and corrects across every term passing the explicit set-size gates. Terms
    with zero overlap then have p-value 1, fold enrichment 0, and remain in the
    output so a library-wide multiple-testing family is fully auditable.
    """
    from scipy.stats import hypergeom

    if int(min_set_size) <= 0:
        raise ValueError("min_set_size must be positive.")
    if max_set_size is not None and int(max_set_size) < int(min_set_size):
        raise ValueError("max_set_size must be at least min_set_size.")
    if int(min_overlap) <= 0:
        raise ValueError("min_overlap must be positive.")
    if not 0.0 < float(alpha) <= 1.0:
        raise ValueError("alpha must be in (0, 1].")
    multiple_testing_scope = str(multiple_testing_scope).strip().lower()
    if multiple_testing_scope not in {"reported", "all_eligible"}:
        raise ValueError(
            "multiple_testing_scope must be 'reported' or 'all_eligible'."
        )
    library = (
        gene_sets
        if isinstance(gene_sets, GeneSetLibrary)
        else make_gene_set_library(gene_sets, uppercase=uppercase)
    )
    library_union = set().union(*library.gene_sets.values())
    if background_genes is None:
        background = set(library_union)
    else:
        background = {
            gene
            for gene in (
                _normalize_gene(value, uppercase=uppercase)
                for value in background_genes
            )
            if gene
        }
    background &= library_union
    if not background:
        raise ValueError("No background genes overlap the gene-set library.")
    raw_query = {
        gene
        for gene in (
            _normalize_gene(value, uppercase=uppercase) for value in query_genes
        )
        if gene
    }
    query = raw_query & background
    if not query:
        raise ValueError("No query genes overlap the tested background.")

    background_size = int(len(background))
    query_size = int(len(query))
    rows: list[dict[str, object]] = []
    for term, members in library.gene_sets.items():
        tested_members = set(members) & background
        set_size = int(len(tested_members))
        if set_size < int(min_set_size):
            continue
        if max_set_size is not None and set_size > int(max_set_size):
            continue
        overlap = sorted(query & tested_members)
        overlap_count = int(len(overlap))
        expected = float(query_size * set_size / background_size)
        p_value = float(
            hypergeom.sf(
                overlap_count - 1,
                background_size,
                set_size,
                query_size,
            )
        )
        a = float(overlap_count)
        b = float(query_size - overlap_count)
        c = float(set_size - overlap_count)
        d = float(background_size - query_size - set_size + overlap_count)
        denominator = b * c
        if overlap_count == 0:
            odds_ratio = 0.0
        else:
            odds_ratio = (
                float((a * d) / denominator) if denominator > 0.0 else np.inf
            )
        term_name, term_id = _term_parts(term)
        rows.append(
            {
                "term": str(term),
                "term_name": term_name,
                "term_id": term_id,
                "description": str(library.descriptions.get(term, "")),
                "query_size": query_size,
                "query_input_size": int(len(raw_query)),
                "background_size": background_size,
                "set_size": set_size,
                "overlap_count": overlap_count,
                "expected_overlap": expected,
                "gene_ratio": float(overlap_count / query_size),
                "background_ratio": float(set_size / background_size),
                "fold_enrichment": float(overlap_count / expected),
                "odds_ratio": odds_ratio,
                "p_value": p_value,
                "passes_min_overlap": bool(overlap_count >= int(min_overlap)),
                "overlap_genes": ";".join(overlap),
            }
        )
    columns = [
        "term",
        "term_name",
        "term_id",
        "description",
        "query_size",
        "query_input_size",
        "background_size",
        "set_size",
        "overlap_count",
        "expected_overlap",
        "gene_ratio",
        "background_ratio",
        "fold_enrichment",
        "odds_ratio",
        "p_value",
        "adjusted_p_value",
        "significant",
        "passes_min_overlap",
        "eligible_test_count",
        "multiple_testing_test_count",
        "multiple_testing_scope",
        "overlap_genes",
    ]
    if not rows:
        result = pd.DataFrame(columns=columns)
        result.attrs.update(
            {
                "eligible_test_count": 0,
                "multiple_testing_test_count": 0,
                "multiple_testing_scope": multiple_testing_scope,
            }
        )
        return result
    eligible = pd.DataFrame(rows)
    eligible_test_count = int(len(eligible))
    if multiple_testing_scope == "reported":
        result = eligible.loc[eligible["passes_min_overlap"]].copy()
    else:
        result = eligible.copy()
    multiple_testing_test_count = int(len(result))
    if result.empty:
        result = pd.DataFrame(columns=columns)
        result.attrs.update(
            {
                "eligible_test_count": eligible_test_count,
                "multiple_testing_test_count": 0,
                "multiple_testing_scope": multiple_testing_scope,
            }
        )
        return result
    result["adjusted_p_value"] = _benjamini_hochberg(
        result["p_value"].to_numpy(dtype=float)
    )
    result["significant"] = result["adjusted_p_value"] <= float(alpha)
    result["eligible_test_count"] = eligible_test_count
    result["multiple_testing_test_count"] = multiple_testing_test_count
    result["multiple_testing_scope"] = multiple_testing_scope
    result = result.sort_values(
        ["adjusted_p_value", "p_value", "fold_enrichment", "term"],
        ascending=[True, True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    result = result.loc[:, columns]
    result.attrs.update(
        {
            "eligible_test_count": eligible_test_count,
            "multiple_testing_test_count": multiple_testing_test_count,
            "multiple_testing_scope": multiple_testing_scope,
        }
    )
    return result
