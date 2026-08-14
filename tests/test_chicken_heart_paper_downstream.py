from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_chicken_heart_paper_downstream.py"
SPEC = importlib.util.spec_from_file_location("chicken_heart_paper", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _complete_summary() -> dict[str, object]:
    required = (
        "velocity",
        "growth",
        "composition",
        "communication",
        "figures",
        "gene_dynamics",
        "ligand_receptor",
    )
    return {
        "dataset": "chicken_heart",
        "analyses": {name: {"status": "completed"} for name in required},
    }


def test_formal_perturbation_contract_is_global_d4_and_fixed_population():
    assert MODULE.TIME_POINTS == (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
    assert MODULE.DISPLAY_TIMES == (0.0, 1.0, 2.0, 3.0)
    assert MODULE.CELLTYPE_ABLATIONS == {
        "remove_endocardial": "Endocardial cells",
        "remove_valve": "Valve cells",
        "remove_immature_myocardial": "Immature myocardial cells",
    }


def test_standard_downstream_requires_every_formal_analysis():
    MODULE._validate_standard_downstream_summary(_complete_summary())
    bad = _complete_summary()
    bad["analyses"]["ligand_receptor"]["status"] = "failed"
    with pytest.raises(RuntimeError, match="ligand_receptor"):
        MODULE._validate_standard_downstream_summary(bad)


def test_interaction_composition_rows_sum_to_one():
    frames = {
        "interaction_on": tuple(
            np.asarray(["A", "A", "B"]) for _ in MODULE.TIME_POINTS
        ),
        "interaction_off": tuple(
            np.asarray(["A", "B", "B"]) for _ in MODULE.TIME_POINTS
        ),
    }
    table = MODULE._composition_rows(frames)
    sums = table.groupby(["condition", "time"])["fraction"].sum()
    assert np.allclose(sums.to_numpy(), 1.0)
    assert set(table.columns) == {
        "condition",
        "time_index",
        "time",
        "celltype",
        "count",
        "fraction",
    }


def test_output_root_must_be_new_or_empty(tmp_path):
    fresh = MODULE._require_empty_output(tmp_path / "fresh")
    assert fresh.is_dir()
    (fresh / "evidence.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError, match="non-empty"):
        MODULE._require_empty_output(fresh)


def test_cli_rejects_nonpositive_classifier_epochs(tmp_path):
    required = [
        "--run-root",
        str(tmp_path / "run"),
        "--input-h5ad",
        str(tmp_path / "input.h5ad"),
        "--model-dir",
        str(tmp_path / "model"),
        "--standard-downstream",
        str(tmp_path / "downstream"),
        "--output-dir",
        str(tmp_path / "output"),
        "--classifier-epochs",
        "0",
    ]
    with pytest.raises(SystemExit):
        MODULE.parse_args(required)
