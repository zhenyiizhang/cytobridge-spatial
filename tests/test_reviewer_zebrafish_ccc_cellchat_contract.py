from pathlib import Path


def test_cellchat_runner_enforces_current_database_and_discloses_nonspatial_use() -> None:
    script = (
        Path(__file__).parents[1]
        / "scripts"
        / "reviewer_zebrafish_ccc"
        / "run_cellchat.R"
    ).read_text(encoding="utf-8")
    assert "database_row" in script
    assert "ligand_expanded" in script
    assert "receptor_expanded" in script
    assert "all_structural_rows_match = TRUE" in script
    assert "pair_lr$pathway_name <- flat_database$pathway" in script
    assert "official_pathway_mismatch_count" in script
    assert "spatial_coordinates_used_by_cellchat = FALSE" in script
    assert "expression_retransformed_in_runner = FALSE" in script
    assert "abundance_controlled_score" in script
    assert 'args[["cellchat-source"]]' in script
    assert '"CellChatDB.zebrafish.rda"' in script
    assert 'CellChat_source_commit = cellchat_commit' in script
