from __future__ import annotations

from pathlib import Path

import pytest

from CytoBridge.graph_database import (
    FORMAL_GRAPH_DATABASES,
    bundled_graph_database_path,
    match_graph_database_features,
    resolve_graph_database,
)


def test_all_formal_presets_resolve_species_matched_bundled_databases():
    expected = {
        "zebrafish": "CellChatDB.ligrec.zebrafish.csv",
        "mosta": "CellChatDB.ligrec.mouse.csv",
        "arista": "CellChatDB.ligrec.human.csv",
        "admouse": "CellChatDB.ligrec.mouse.csv",
    }

    assert FORMAL_GRAPH_DATABASES == expected
    for dataset, filename in expected.items():
        path = bundled_graph_database_path(dataset)
        assert path.is_file()
        assert path.name == filename
        assert path.stat().st_size > 50_000


def test_explicit_graph_database_overrides_bundled_resource(tmp_path: Path):
    explicit = tmp_path / "custom_lr.csv"
    explicit.write_text("ligand,receptor\nA,B\n", encoding="utf-8")

    assert resolve_graph_database("mosta", explicit) == explicit.resolve()


def test_lr_subunits_match_exactly_case_insensitively_and_report_coverage(
    tmp_path: Path,
):
    database = tmp_path / "lr.csv"
    database.write_text(
        "ligand,receptor,pathway,annotation\n"
        "Tgfb1,Tgfbr1_Tgfbr2,TGFb,Secreted Signaling\n"
        "missing,AMBIG,Other,Secreted Signaling\n",
        encoding="utf-8",
    )

    matched, coverage = match_graph_database_features(
        database,
        ["tgfb1", "TGFBR1", "Tgfbr2", "Ambig", "AMBIG", "Tgfb1-extra"],
    )

    assert matched == ("tgfb1", "TGFBR1", "Tgfbr2")
    assert coverage["matching_policy"] == (
        "selected_symbol_exact_case_insensitive_unique"
    )
    assert coverage["n_unique_database_subunits"] == 5
    assert coverage["n_matched_features"] == 3
    assert coverage["n_missing_database_subunits"] == 1
    assert coverage["missing_database_subunits"] == ["missing"]
    assert coverage["n_ambiguous_database_subunits"] == 1
    assert coverage["ambiguous_database_subunits"] == ["AMBIG"]
    assert coverage["coverage_fraction"] == pytest.approx(3 / 5)


def test_duplicate_input_feature_is_ambiguous_not_silently_selected(tmp_path: Path):
    database = tmp_path / "lr.csv"
    database.write_text("0,1,2,3\nA,B,p,s\n", encoding="utf-8")

    matched, coverage = match_graph_database_features(database, ["A", "A", "B"])

    assert matched == ("B",)
    assert coverage["ambiguous_database_subunits"] == ["A"]


def test_compound_feature_uses_requested_species_before_exact_matching(tmp_path: Path):
    database = tmp_path / "human_lr.csv"
    database.write_text("0,1,2,3\nZNF268,GP1BB,p,s\n", encoding="utf-8")
    features = [
        "LOC115474470[nr]|ZNF268[hs] | AMEX60DD000058",
        "GP1BB[nr] | AMEX60DD000145",
    ]

    human, coverage = match_graph_database_features(
        database,
        features,
        preferred_species_tag="hs",
    )

    assert human == (features[0],)
    assert coverage["preferred_species_tag"] == "hs"
    assert coverage["missing_database_subunits"] == ["GP1BB"]
