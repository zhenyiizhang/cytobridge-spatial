from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pandas as pd
import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "reviewer_zebrafish_response"
    / "compare_lr_complex_aggregation.py"
)
SPEC = importlib.util.spec_from_file_location(
    "compare_lr_complex_aggregation", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def _write_inputs(tmp_path):
    minimum = pd.DataFrame(
        {
            "time": [0.0, 0.0, 1.0, 1.0],
            "pair": ["L1_R1", "L2_L3_R2", "L1_R1", "L2_L3_R2"],
            "score": [1.0, 2.0, 2.0, 4.0],
        }
    )
    geometric = minimum.copy()
    geometric["score"] = [1.0, 3.0, 2.0, 6.0]
    database = pd.DataFrame(
        {
            "ligand": ["L1", "L2_L3"],
            "receptor": ["R1", "R2"],
        }
    )
    paths = (
        tmp_path / "min.csv",
        tmp_path / "geometric.csv",
        tmp_path / "lr.csv",
    )
    minimum.to_csv(paths[0], index=False)
    geometric.to_csv(paths[1], index=False)
    database.to_csv(paths[2], index=False)
    return paths


def test_compare_lr_complex_aggregation_identifies_multisubunit_sensitivity(
    tmp_path,
):
    min_path, geometric_path, database_path = _write_inputs(tmp_path)
    merged, per_time, per_pair, summary = module.compare_tables(
        min_path,
        geometric_path,
        database_path,
        top_fraction=1.0,
        top_k=10,
    )

    assert summary["n_scored_pairs"] == 2
    assert summary["n_multisubunit_pairs"] == 1
    assert summary["primary_result_is_mathematically_invariant"] is False
    simple = merged.loc[merged["pair"] == "L1_R1"]
    assert simple["absolute_difference"].eq(0).all()
    multi = per_pair.set_index("pair").loc["L2_L3_R2"]
    assert bool(multi["is_multisubunit"]) is True
    assert multi["max_symmetric_relative_difference"] == pytest.approx(1 / 3)
    assert set(per_time["scope"]) == {
        "all_scored_pairs",
        "multisubunit_pairs",
    }


def test_lr_complex_aggregation_cli_writes_auditable_bundle(tmp_path):
    min_path, geometric_path, database_path = _write_inputs(tmp_path)
    output_dir = tmp_path / "out"
    assert (
        module.main(
            [
                "--min-table",
                str(min_path),
                "--geometric-table",
                str(geometric_path),
                "--lr-database",
                str(database_path),
                "--output-dir",
                str(output_dir),
            ]
        )
        == 0
    )
    manifest = json.loads(
        (output_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["schema_version"] == 1
    assert len(manifest["inputs"]) == 3
    assert (output_dir / "lr_complex_aggregation_sensitivity.png").exists()


def test_lr_complex_aggregation_rejects_mismatched_pair_universe(tmp_path):
    min_path, geometric_path, database_path = _write_inputs(tmp_path)
    geometric = pd.read_csv(geometric_path).iloc[:-1]
    geometric.to_csv(geometric_path, index=False)
    with pytest.raises(ValueError, match="same time/pair universe"):
        module.compare_tables(min_path, geometric_path, database_path)
