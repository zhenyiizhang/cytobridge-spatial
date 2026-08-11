from __future__ import annotations

import numpy as np
import pytest

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
    assert result.loc[0, "eligible_test_count"] == 2
    assert result.loc[0, "multiple_testing_test_count"] == 1
    assert result.loc[0, "multiple_testing_scope"] == "reported"


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
    assert result.attrs["eligible_test_count"] == 1
    assert result.attrs["multiple_testing_test_count"] == 0


def test_all_eligible_scope_keeps_zero_overlap_terms_in_one_bh_family() -> None:
    library = make_gene_set_library(
        {
            "enriched": ["A", "B", "C"],
            "partial": ["A", "D", "E"],
            "zero": ["D", "E", "F"],
            "too_small": ["A"],
        }
    )
    default = overrepresentation_analysis(
        ["A", "B"],
        library,
        background_genes=list("ABCDEF"),
        min_set_size=2,
        min_overlap=2,
    )
    assert default["term"].tolist() == ["enriched"]
    assert default.loc[0, "adjusted_p_value"] == pytest.approx(0.2)
    assert default.loc[0, "eligible_test_count"] == 3
    assert default.loc[0, "multiple_testing_test_count"] == 1

    all_eligible = overrepresentation_analysis(
        ["A", "B"],
        library,
        background_genes=list("ABCDEF"),
        min_set_size=2,
        min_overlap=2,
        multiple_testing_scope="all_eligible",
    )
    assert set(all_eligible["term"]) == {"enriched", "partial", "zero"}
    assert (all_eligible["eligible_test_count"] == 3).all()
    assert (all_eligible["multiple_testing_test_count"] == 3).all()
    assert (all_eligible["multiple_testing_scope"] == "all_eligible").all()
    enriched = all_eligible.set_index("term").loc["enriched"]
    assert enriched["p_value"] == pytest.approx(0.2)
    assert enriched["adjusted_p_value"] == pytest.approx(0.6)
    zero = all_eligible.set_index("term").loc["zero"]
    assert zero["overlap_count"] == 0
    assert zero["p_value"] == pytest.approx(1.0)
    assert zero["adjusted_p_value"] == pytest.approx(1.0)
    assert zero["fold_enrichment"] == pytest.approx(0.0)
    assert zero["odds_ratio"] == pytest.approx(0.0)
    assert zero["overlap_genes"] == ""
    assert not bool(zero["passes_min_overlap"])
    assert not bool(zero["significant"])
    assert all_eligible.attrs == {
        "eligible_test_count": 3,
        "multiple_testing_test_count": 3,
        "multiple_testing_scope": "all_eligible",
    }


def test_multiple_testing_scope_validation() -> None:
    library = make_gene_set_library({"term": ["A", "B"]})
    with pytest.raises(ValueError, match="multiple_testing_scope"):
        overrepresentation_analysis(
            ["A"],
            library,
            min_set_size=1,
            multiple_testing_scope="per_cluster_magic",
        )
