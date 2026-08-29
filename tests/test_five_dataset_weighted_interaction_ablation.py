from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "run_five_dataset_weighted_interaction_ablation.py"
)
SPEC = importlib.util.spec_from_file_location(
    "run_five_dataset_weighted_interaction_ablation", SCRIPT
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_projection_seed_and_basis_are_stable():
    seed = MODULE._projection_seed("zebrafish", "joint", 0)
    assert seed == MODULE._projection_seed("zebrafish", "joint", 0)
    assert seed != MODULE._projection_seed("zebrafish", "joint", 1)
    assert len(MODULE._projection_sha256(52, seed)) == 64


def test_relative_tables_define_positive_as_off_worse():
    rows = []
    for arm, value in (("interaction_on", 2.0), ("interaction_off", 3.0)):
        for repeat in range(5):
            rows.append(
                {
                    "dataset": "zebrafish",
                    "target": 1.0,
                    "space": "joint",
                    "arm": arm,
                    "sliced_w2": value,
                    "tmv": value / 10.0,
                }
            )
    target, summary, tmv = MODULE.relative_tables(pd.DataFrame(rows))
    assert target.iloc[0]["off_relative_to_on"] == 0.5
    assert summary.iloc[0]["mean_relative_change"] == 0.5
    assert np.isclose(tmv.iloc[0]["off_relative_to_on"], 0.5)


def test_cli_accepts_benchmark_and_aligned_input_modes(tmp_path):
    common = [
        "run",
        "--dataset",
        "zebrafish",
        "--model-dir",
        str(tmp_path / "model"),
        "--expected-training-summary-sha256",
        "0" * 64,
        "--output-dir",
        str(tmp_path / "out"),
    ]
    benchmark = MODULE.parse_args(
        common
        + [
            "--benchmark-input-manifest",
            str(tmp_path / "manifest.json"),
            "--expected-benchmark-input-sha256",
            "1" * 64,
        ]
    )
    assert benchmark.benchmark_input_manifest.name == "manifest.json"
    aligned = MODULE.parse_args(
        common
        + [
            "--aligned-h5ad",
            str(tmp_path / "aligned.h5ad"),
            "--expected-aligned-sha256",
            "2" * 64,
        ]
    )
    assert aligned.aligned_h5ad.name == "aligned.h5ad"
