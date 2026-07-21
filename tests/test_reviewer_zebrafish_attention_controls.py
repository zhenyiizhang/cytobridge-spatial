from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "reviewer_zebrafish_ccc"
    / "analyze_attention_confound_controls.py"
)
SPEC = importlib.util.spec_from_file_location("reviewer_zebrafish_controls", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_pathway_balancing_gives_equal_total_pathway_weight():
    database = pd.DataFrame(
        {
            "pathway": ["A", "A", "B"],
            "ligand": ["l1", "l2", "l3"],
            "receptor": ["r1", "r2", "r3"],
        }
    )
    weights = MODULE._pathway_balanced_weights(database)
    np.testing.assert_allclose(weights, [0.25, 0.25, 0.5])


def test_complex_activity_requires_all_subunits_by_minimum():
    matrix = np.array([[2.0, 5.0], [0.0, 4.0], [3.0, 1.0]])
    np.testing.assert_allclose(MODULE._complex_activity(matrix, [0, 1]), [2, 0, 1])


def test_forward_and_reverse_lr_scores_preserve_direction():
    database = pd.DataFrame(
        {"pathway": ["P"], "ligand": ["L"], "receptor": ["R"]}
    )
    activities = {
        "L": np.array([1.0, 0.0, 0.2]),
        "R": np.array([0.0, 1.0, 0.5]),
    }
    scores = MODULE._edge_lr_compatibility(
        np.array([0, 1]), np.array([1, 0]), database, activities
    )
    np.testing.assert_allclose(scores["lr_compatibility_forward"], [1.0, 0.0])
    np.testing.assert_allclose(scores["lr_compatibility_reverse"], [0.0, 1.0])
    np.testing.assert_array_equal(scores["active_lr_count"], [1, 0])


def test_conditional_permutation_uses_plus_one_correction():
    edges = pd.DataFrame(
        {
            "stage": [0] * 8,
            "distance_bin": [0] * 8,
            "state_bin": [0] * 8,
            "score": np.arange(8, dtype=float),
        }
    )
    result = MODULE._conditional_permutation(
        edges,
        residual=np.arange(8, dtype=float),
        score_column="score",
        keys=["stage", "distance_bin", "state_bin"],
        min_stratum_size=4,
        n_permutations=19,
        random_state=7,
    )
    assert 1 / 20 <= result["empirical_p_greater"] <= 1
    assert result["n_edges_retained"] == 8
