from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pandas as pd


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "reviewer_zebrafish_ccc"
    / "run_cytobridge_spatial_attribution.py"
)
SPEC = importlib.util.spec_from_file_location("reviewer_zebrafish_cb", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_stage_label_requires_one_label():
    assert MODULE._stage_label(pd.Series(["10hpf", "10hpf"])) == "10hpf"


def test_summary_averages_grouping_only_after_complete_tables():
    tables = []
    for seed, value in ((101, 2.0), (202, 4.0)):
        tables.append(
            pd.DataFrame(
                {
                    "stage": [1.0],
                    "stage_label": ["10hpf"],
                    "grouping_seed": [seed],
                    "sender_type": ["A"],
                    "receiver_type": ["B"],
                    "edge_count": [3],
                    "G_AB_attention_mean": [value],
                    "D_AB_joint": [value / 2],
                }
            )
        )
    summary = MODULE._summarize_type_pairs(tables)
    assert summary.loc[0, "G_AB_attention_mean_mean"] == 3.0
    assert summary.loc[0, "D_AB_joint_mean"] == 1.5
    assert summary.loc[0, "n_grouping_seeds"] == 2
