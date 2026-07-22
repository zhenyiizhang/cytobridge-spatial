from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "scripts"
    / "reviewer_zebrafish_ccc"
    / "plain_language_consistency_report.py"
)
SPEC = importlib.util.spec_from_file_location("plain_language_consistency_report", SCRIPT)
assert SPEC and SPEC.loader
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


def test_top_set_keeps_boundary_ties() -> None:
    frame = pd.DataFrame({"score": [5.0, 4.0, 4.0, 1.0, 0.0]})
    assert report.top_set(frame, "score", 2) == {0, 1, 2}


def test_stage_table_reproduces_formal_top_overlap() -> None:
    scores = pd.DataFrame(
        {
            "stage": [0.0] * 5,
            "sender_type": [f"S{i}" for i in range(5)],
            "receiver_type": ["R"] * 5,
            "cytobridge_attention": [5, 4, 3, 2, 1],
            "cytobridge_attention_rank": [1.0, 0.8, 0.6, 0.4, 0.2],
            "external_native_consensus": [5, 3, 4, 2, 1],
        }
    )
    formal = pd.DataFrame(
        {
            "target": ["CytoBridge attention"],
            "reference": ["External native consensus"],
            "stage": [0.0],
            "target_set_size_after_boundary_ties": [1],
            "reference_set_size_after_boundary_ties": [1],
            "intersection": [1],
        }
    )
    result = report.prepare_stage_table(scores, formal)
    assert result["shared_top20"].sum() == 1
    assert result.loc[result["shared_top20"], "sender_type"].item() == "S0"


def test_figure_index_separates_evidence_from_audit() -> None:
    index = report.build_figure_index().set_index("figure")
    assert index.loc["01_reviewer_evidence_map", "recommended_role"] == "新主图"
    assert index.loc["condition_coverage", "recommended_role"] == "质量审计"
    assert index.loc["directionality_concordance", "recommended_role"] == "限制/审计"
