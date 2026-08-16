from pathlib import Path


def test_cellchat_runner_enforces_current_database_and_discloses_nonspatial_use() -> (
    None
):
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
    assert "resolve_official_index" in script
    assert "rows_resolved_by_structural_key" in script
    assert "length(candidates) != 1L" in script
    assert "pair_lr$pathway_name <- flat_database$pathway" in script
    assert "official_pathway_mismatch_count" in script
    assert 'c("agonist", "antagonist", "co_A_receptor", "co_I_receptor")' in script
    assert 'pair_lr[[column]] <- ""' in script
    assert "cellchat_token_eligibility" in script
    assert 'reason = "token_not_geneinfo_or_declared_complex"' in script
    assert "database_eligibility_audit.csv" in script
    assert "excluded_lr_rows.csv" in script
    assert "pair_lr_requested[database_eligibility$eligible" in script
    assert "excluded_rows_are_method_unavailable_not_biological_zero = TRUE" in script
    assert (
        "must be excluded from CellChat cross-method universes, never zero-filled"
        in script
    )
    assert "spatial_coordinates_used_by_cellchat = FALSE" in script
    assert "expression_retransformed_in_runner = FALSE" in script
    assert "abundance_controlled_score" in script
    assert 'args[["cellchat-source"]]' in script
    assert 'database_filename <- paste0(database_object_name, ".rda")' in script
    assert 'c("zebrafish", "mouse", "human")' in script
    assert 'official_database <- get(database_object_name)' in script
    assert "CellChat_source_commit = cellchat_commit" in script
    assert (
        'pinned_cellchat_commit <- "75253cd0c9e68410e6e721a6d3a0419a1d7e358f"' in script
    )
    assert (
        '"sum of unthresholded official CellChat LR probabilities over the prepared database",\n    FALSE'
        in script
    )
    assert "complete_directed_stage_type_square = TRUE" in script
    assert 'universe_source = "input_manifest.stages[].cell_type_counts"' in script
    assert "matrix_context(\n      score_matrix" in script
    assert "positive_only\n    )" in script
    assert "excluded_rows_are_method_unavailable_not_biological_zero = TRUE" in script
