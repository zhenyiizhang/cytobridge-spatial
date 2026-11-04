from __future__ import annotations

import numpy as np

from CytoBridge.pl.celltype import plot_celltype_composition
from CytoBridge.tl.downstream.celltype import summarize_label_composition


def test_label_composition_summary_and_plot(tmp_path) -> None:
    summary = summarize_label_composition(
        [np.asarray(["A", "A", "B"]), np.asarray(["A", "B", "B", "B"])],
        [0.0, 0.5],
    )
    assert summary.groupby("time_index")["count"].sum().to_dict() == {0: 3, 1: 4}
    np.testing.assert_allclose(
        summary.groupby("time_index")["fraction"].sum().to_numpy(),
        [1.0, 1.0],
    )
    path = plot_celltype_composition(
        summary,
        out_path=tmp_path / "composition.svg",
        label_to_color={"A": "#111111", "B": "#cccccc"},
    )
    assert path.is_file()
    assert path.stat().st_size > 0
