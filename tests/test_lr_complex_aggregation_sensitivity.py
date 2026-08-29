from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pandas as pd
import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_lr_complex_aggregation_sensitivity.py"
)
SPEC = importlib.util.spec_from_file_location("lr_complex_sensitivity", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def test_communication_table_round_trip_preserves_matrix_order(tmp_path: Path):
    table = tmp_path / "communication.csv"
    pd.DataFrame(
        [
            {"time": 0.0, "source": "B", "target": "B", "attention_per_source": 1},
            {"time": 0.0, "source": "B", "target": "A", "attention_per_source": 2},
            {"time": 0.0, "source": "A", "target": "B", "attention_per_source": 3},
            {"time": 0.0, "source": "A", "target": "A", "attention_per_source": 4},
        ]
    ).to_csv(table, index=False)

    record = runner._communications_from_table(table, [0.0])["0.0"]
    assert record["types"].tolist() == ["B", "A"]
    assert record["M_per_source"].tolist() == [[1.0, 2.0], [3.0, 4.0]]


def test_primary_reproduction_requires_same_universe_and_scores(tmp_path: Path):
    primary = tmp_path / "primary.csv"
    pd.DataFrame(
        {"time": [0.0, 1.0], "pair": ["A_B", "A_B"], "score": [2.0, 4.0]}
    ).to_csv(primary, index=False)
    recomputed = pd.DataFrame(
        {
            "time": [0.0, 1.0],
            "pair": ["A_B", "A_B"],
            "score": [2.0 + 1e-14, 4.0],
        }
    )
    audit = runner._verify_primary_reproduction(primary, recomputed)
    assert audit["n_rows"] == 2
    assert audit["max_absolute_score_difference"] < 1e-12

    recomputed.loc[1, "score"] = 5.0
    with pytest.raises(ValueError, match="do not reproduce"):
        runner._verify_primary_reproduction(primary, recomputed)


def test_observed_times_are_derived_from_saved_slice_origins():
    assert runner._observed_times(
        {
            "simulation": {
                "slice_origins_by_time": {
                    "0.0": "observed_real",
                    "0.5": "generated_interval_local",
                    "1.0": "observed_real",
                }
            }
        }
    ) == [0.0, 1.0]
