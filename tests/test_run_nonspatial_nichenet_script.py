from pathlib import Path


RUNNER = Path(__file__).parents[1] / "scripts" / "run_nonspatial_nichenet.R"


def test_nichenet_runner_logs_unrepresented_receiver_and_continues() -> None:
    source = RUNNER.read_text(encoding="utf-8")

    assert 'status = "skipped_no_potential_ligands"' in source
    assert "n_potential_ligands = 0L" in source
    assert "next" in source
    assert 'file.path(output_dir, "receiver_status.csv")' in source
    assert "no receiver has a candidate ligand represented" in source


def test_nichenet_runner_keeps_fail_closed_input_contracts() -> None:
    source = RUNNER.read_text(encoding="utf-8")

    assert "has fewer than ten response genes" in source
    assert "has fewer than twenty background genes" in source
    assert "candidate and receiver gene-set datasets must match exactly" in source
