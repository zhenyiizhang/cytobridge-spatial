from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from CytoBridge.tl.downstream.lr_panels import (
    _combine_subunit_vectors,
    _load_lr_db,
    _parse_lr_tokens,
)


def test_panel_complex_geometric_mean_aliases_are_equivalent() -> None:
    vectors = [
        np.asarray([1.0, 4.0, 0.0]),
        np.asarray([4.0, 9.0, 5.0]),
    ]
    expected = np.asarray([2.0, 6.0, 0.0])
    for mode in ("geometric_mean", "geometric-mean", "geomean", "geometric"):
        np.testing.assert_allclose(_combine_subunit_vectors(vectors, mode), expected)


def test_panel_complex_aggregation_rejects_nonfinite_and_negative_geomean() -> None:
    with pytest.raises(ValueError, match="finite"):
        _combine_subunit_vectors([np.asarray([1.0]), np.asarray([np.nan])], "min")
    with pytest.raises(ValueError, match="non-negative"):
        _combine_subunit_vectors(
            [np.asarray([1.0]), np.asarray([-1.0])],
            "geometric_mean",
        )


def test_panel_database_does_not_collapse_ambiguous_legacy_pair_labels(tmp_path) -> None:
    database_path = tmp_path / "lr.csv"
    pd.DataFrame(
        {
            "ligand": ["A_B", "A"],
            "receptor": ["C", "B_C"],
        }
    ).to_csv(database_path, index=False)

    database = _load_lr_db(database_path)
    assert len(database) == 2
    assert database["pair_id"].nunique() == 2
    assert database["lr_pair"].nunique() == 1
    with pytest.raises(ValueError, match="Ambiguous legacy LR pair label"):
        _parse_lr_tokens("A_B_C", database)
    for row in database.itertuples(index=False):
        assert _parse_lr_tokens(row.pair_id, database) == (row.ligand, row.receptor)
