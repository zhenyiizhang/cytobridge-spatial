from __future__ import annotations

import numpy as np

from CytoBridge.tl.downstream.enrichment import (
    load_gmt_gene_sets,
    make_gene_set_library,
    overrepresentation_analysis,
)


def test_offline_overrepresentation_uses_explicit_background() -> None:
    library = make_gene_set_library(
        {
            "Early response (GO:0000001)": ["A", "B", "C", "D"],
            "Unrelated (GO:0000002)": ["G", "H", "I"],
        }
    )
    result = overrepresentation_analysis(
        ["a", "b", "c", "outside"],
        library,
        background_genes=list("ABCDEFGHIJ"),
        min_set_size=2,
    )
    assert result.loc[0, "term_id"] == "GO:0000001"
    assert result.loc[0, "term_name"] == "Early response"
    assert result.loc[0, "overlap_count"] == 3
    assert result.loc[0, "query_input_size"] == 4
    assert result.loc[0, "query_size"] == 3
    assert result.loc[0, "background_size"] == 7
    assert result.loc[0, "overlap_genes"] == "A;B;C"
    assert np.isfinite(result.loc[0, "p_value"])
    assert np.isfinite(result.loc[0, "adjusted_p_value"])


def test_load_gmt_records_hash_and_deduplicates_genes(tmp_path) -> None:
    path = tmp_path / "library.gmt"
    path.write_text(
        "Term A (GO:0000001)\tdescription\tA\tB\tB\n"
        "Term B\t\tC\tD\n",
        encoding="utf-8",
    )
    library = load_gmt_gene_sets(path)
    assert library.gene_sets["Term A (GO:0000001)"] == frozenset({"A", "B"})
    assert library.descriptions["Term A (GO:0000001)"] == "description"
    assert library.metadata["format"] == "gmt"
    assert len(str(library.metadata["sha256"])) == 64


def test_empty_overlap_returns_declared_schema() -> None:
    library = make_gene_set_library({"Term": ["A", "B", "C"]})
    result = overrepresentation_analysis(
        ["A"],
        library,
        background_genes=["A", "B", "C"],
        min_set_size=2,
        min_overlap=2,
    )
    assert result.empty
    assert "adjusted_p_value" in result.columns
