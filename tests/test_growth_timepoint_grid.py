from __future__ import annotations

import anndata as ad
import numpy as np
import torch

from CytoBridge.pl.growth import plot_growth_timepoint_grid
from CytoBridge.tl.downstream.celltype import evaluate_growth_by_timepoint


class _GrowthModel(torch.nn.Module):
    components = ["growth"]

    def predict_growth(self, t, x):
        return x[:, :1] + t


def test_growth_timepoint_evaluation_and_grid(tmp_path) -> None:
    slices = {}
    for time in [0.0, 0.5]:
        values = np.asarray([[0.0, 0.0, 1.0], [1.0, 1.0, 2.0]], dtype=np.float32)
        current = ad.AnnData(X=values)
        current.obsm["spatial"] = values[:, :2]
        current.obs["Annotation"] = ["A", "B"]
        slices[str(time)] = current
    table = evaluate_growth_by_timepoint(
        slices,
        _GrowthModel(),
        time_points=[0.0, 0.5],
        device="cpu",
    )
    np.testing.assert_allclose(table["growth"], [0.0, 1.0, 0.5, 1.5])
    path = plot_growth_timepoint_grid(
        slices,
        time_points=[0.0, 0.5],
        out_path=tmp_path / "growth.svg",
    )
    assert path.is_file()
    assert path.stat().st_size > 0
