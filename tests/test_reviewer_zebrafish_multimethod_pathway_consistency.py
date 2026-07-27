from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "scripts"
    / "reviewer_zebrafish_ccc"
    / "multimethod_pathway_consistency.py"
)
SPEC = importlib.util.spec_from_file_location("multimethod_pathway_consistency", SCRIPT)
assert SPEC and SPEC.loader
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def test_cytobridge_loader_separates_exact_message_and_lr(tmp_path: Path) -> None:
    path = tmp_path / "axis.csv"
    pd.DataFrame(
        {
            "stage": [0.0, 0.0],
            "ligand": ["L1", "L2"],
            "receptor": ["R1", "R2"],
            "mean_attention_times_lr_activity": [2.0, 0.0],
            "mean_exact_message_times_lr_activity": [1.5, 0.0],
            "mean_scaled_lr_activity": [0.5, 0.0],
        }
    ).to_csv(path, index=False)

    result = analysis.load_cytobridge(path)

    assert result["CytoBridge LR-only"].tolist() == [0.5, 0.0]
    assert result["CytoBridge exact message x LR"].tolist() == [1.5, 0.0]
    assert result[
        "CytoBridge exact message only (LR-conditioned)"
    ].tolist() == [3.0, 0.0]


def test_top_mask_keeps_positive_boundary_ties_but_never_zero() -> None:
    values = pd.Series([5.0, 4.0, 4.0, 0.0, 0.0])
    mask, requested, selected, boundary = analysis._top_mask(values, 0.40)
    assert requested == 2
    assert selected == 3
    assert boundary == 4.0
    assert mask.tolist() == [True, True, True, False, False]


def test_pathway_ranking_preserves_theoretical_zero_ties() -> None:
    rows = []
    for stage in analysis.STAGES:
        for axis, score in (("a", 1.0), ("b", 0.0), ("c", 0.0), ("d", 0.0)):
            rows.append({"stage": stage, "axis": axis, "method": score})
    grid = pd.DataFrame(rows)
    annotations = pd.DataFrame(
        {
            "axis": ["a", "b", "c", "c", "d"],
            "pathway": ["signal", "zero_1", "zero_1", "zero_2", "zero_2"],
        }
    )

    profiles, coverage = analysis.pathway_profiles(
        grid,
        annotations,
        ["method"],
        design="test",
        top_fraction=0.25,
    )

    for _, stage in profiles.groupby("stage"):
        zero = stage.loc[stage["pathway"].str.startswith("zero")]
        assert zero["pathway_mean_axis_rank"].nunique() == 1
        assert zero["pathway_rank"].nunique() == 1
    assert coverage["rank_informative"].all()


def test_all_zero_method_is_not_rank_informative() -> None:
    grid = pd.DataFrame(
        [
            {"stage": stage, "axis": axis, "method": 0.0}
            for stage in analysis.STAGES
            for axis in ("a", "b")
        ]
    )
    annotations = pd.DataFrame(
        {"axis": ["a", "b"], "pathway": ["p1", "p2"]}
    )
    profiles, coverage = analysis.pathway_profiles(
        grid,
        annotations,
        ["method"],
        design="zero",
        top_fraction=0.5,
    )
    assert not coverage["rank_informative"].any()
    assert profiles["pathway_rank"].isna().all()
    assert profiles["top_pathway_hits"].eq(0).all()
    assert profiles["hypergeometric_p_greater"].eq(1.0).all()


def test_bh_includes_zero_hit_pathways() -> None:
    p = np.asarray([0.01, 1.0, 1.0])
    adjusted = analysis._bh(p)
    assert adjusted[0] == pytest.approx(0.03)
    assert adjusted[1] == pytest.approx(1.0)
    assert adjusted[2] == pytest.approx(1.0)
