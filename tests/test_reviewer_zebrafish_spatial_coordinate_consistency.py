from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "scripts"
    / "reviewer_zebrafish_ccc"
    / "spatial_coordinate_consistency.py"
)
SPEC = importlib.util.spec_from_file_location("spatial_coordinate_consistency", SCRIPT)
assert SPEC and SPEC.loader
analysis = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = analysis
SPEC.loader.exec_module(analysis)


def _coordinates() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cell_index_global": np.arange(6),
            "cell_id": [f"C{i}" for i in range(6)],
            "stage": [3.0] * 6,
            "stage_label": ["18hpf"] * 6,
            "cell_type": ["A", "A", "A", "B", "B", "B"],
            "x": [0.0, 0.1, 0.2, 0.0, 0.1, 0.2],
            "y": [0.0, 0.0, 0.0, 0.2, 0.2, 0.2],
        }
    )


def test_spatial_field_is_unit_mass_and_hdr_contains_requested_mass() -> None:
    coordinates = _coordinates()
    grid = analysis.make_grid(
        coordinates,
        cutoff=0.15,
        step_factor=0.25,
        mask_radius_factor=1.0,
    )
    points = np.array([[0.05, 0.05], [0.15, 0.05], [0.10, 0.18]])
    field = analysis.spatial_field(points, grid, bandwidth=0.075)
    assert field.sum() == pytest.approx(1.0)
    assert np.all(field[~grid.tissue_mask] == 0)
    mask, threshold = analysis.hdr_mask(field, 0.8)
    assert np.isfinite(threshold)
    assert field[mask].sum() >= 0.8


def test_one_to_one_matching_does_not_reuse_dense_reference_point() -> None:
    left = np.array([[0.0, 0.0], [0.01, 0.0]])
    right = np.array([[0.005, 0.0]])
    result = analysis.one_to_one_match_f1(left, right, radius=0.02)
    assert result["one_to_one_matches"] == 1
    assert result["spatial_match_f1"] == pytest.approx(2 / 3)


def test_permutation_stays_within_strata_and_is_reproducible() -> None:
    values = np.array([1.0, 2.0, 10.0, 20.0])
    strata = np.array(["a", "a", "b", "b"])
    first = analysis.permute_within_strata(
        values, strata, np.random.default_rng(7)
    )
    second = analysis.permute_within_strata(
        values, strata, np.random.default_rng(7)
    )
    assert np.array_equal(first, second)
    assert set(first[:2]) == {1.0, 2.0}
    assert set(first[2:]) == {10.0, 20.0}


def test_adaptive_strata_are_auditable_and_do_not_silently_create_singletons() -> None:
    n = 40
    frame = pd.DataFrame(
        {
            "sender_type": ["rare"] * 7 + ["common"] * (n - 7),
            "receiver_type": ["target"] * n,
            "source_x": np.linspace(0.0, 1.0, n),
            "source_y": np.zeros(n),
            "target_x": np.linspace(0.01, 1.01, n),
            "target_y": np.linspace(0.0, 0.2, n),
            "mean_scaled_lr_activity": np.linspace(0.01, 1.0, n),
        }
    )
    assignment = analysis.permutation_strata_assignment(
        frame, method="cytobridge", min_size=5, bins=3
    )
    sizes = assignment["permutation_stratum"].value_counts()
    assert sizes.min() >= 5
    assert assignment["permutation_level"].isin(
        ["fine_type_covariate", "pooled_covariate", "pooled_distance", "global"]
    ).all()
    assert assignment["edge_distance"].notna().all()


def test_attach_coordinates_maps_global_indices_and_cell_ids() -> None:
    coordinates = _coordinates()
    cb = pd.DataFrame(
        {
            "stage": [3.0],
            "source_index": [1],
            "target_index": [4],
        }
    )
    commot = pd.DataFrame(
        {
            "stage": [3.0],
            "source_cell_id": ["C1"],
            "target_cell_id": ["C4"],
        }
    )
    mapped_cb, mapped_commot = analysis.attach_coordinates(cb, commot, coordinates)
    assert mapped_cb.loc[0, ["source_x", "source_y"]].tolist() == [0.1, 0.0]
    assert mapped_cb.loc[0, ["target_x", "target_y"]].tolist() == [0.1, 0.2]
    assert mapped_commot.loc[0, ["source_x", "source_y"]].tolist() == [0.1, 0.0]
    assert mapped_commot.loc[0, ["target_x", "target_y"]].tolist() == [0.1, 0.2]


def test_field_overlap_is_one_for_identical_fields_and_zero_for_disjoint() -> None:
    left = np.array([[0.5, 0.5], [0.0, 0.0]])
    identical = analysis.field_metrics(left, left)
    assert identical["field_overlap_ovl"] == pytest.approx(1.0)
    assert identical["hdr80_dice"] == pytest.approx(1.0)
    right = np.array([[0.0, 0.0], [0.5, 0.5]])
    disjoint = analysis.field_metrics(left, right)
    assert disjoint["field_overlap_ovl"] == pytest.approx(0.0)
    assert disjoint["hdr80_dice"] == pytest.approx(0.0)
