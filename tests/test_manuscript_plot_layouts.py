from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import anndata as ad
from CytoBridge.pl.growth import plot_growth_timepoint_grid
from CytoBridge.pl.temporal import plot_temporal_gene_heatmap
from CytoBridge.pl.trajectory import (
    plot_trajectory_comparison_grid,
    plot_trajectory_grid,
)


def test_temporal_heatmap_splits_one_global_gene_order(monkeypatch, tmp_path):
    genes = [f"g{index}" for index in range(6)]
    expression = pd.DataFrame(
        np.arange(24, dtype=float).reshape(6, 4),
        index=genes,
        columns=[0.0, 0.5, 1.0, 1.5],
    )
    table = pd.DataFrame(
        {
            "gene": genes,
            "variance": np.arange(6, dtype=float),
            "cluster": [1, 1, 2, 2, 3, 3],
            "dendrogram_rank": [5, 0, 4, 1, 3, 2],
        }
    )
    captured = []

    def fake_heatmap(values, **kwargs):
        captured.append(values.copy())
        return kwargs["ax"]

    monkeypatch.setattr("seaborn.heatmap", fake_heatmap)
    path = tmp_path / "heatmap.png"
    plot_temporal_gene_heatmap(
        expression,
        table,
        out_path=path,
        top_n=5,
        panel_columns=2,
    )

    assert path.exists() and path.stat().st_size > 0
    assert [len(block) for block in captured] == [3, 2]
    expected = table.sort_values("dendrogram_rank")["gene"].tolist()[:5]
    assert [gene for block in captured for gene in block.index] == expected
    assert all(block.columns.tolist() == expression.columns.tolist() for block in captured)

    with pytest.raises(ValueError, match="panel_columns must be positive"):
        plot_temporal_gene_heatmap(
            expression,
            table,
            out_path=tmp_path / "invalid.png",
            panel_columns=0,
        )


def test_trajectory_grid_wraps_to_three_by_three_with_one_legend(tmp_path):
    times = [0.5 * index for index in range(9)]
    points = np.empty(9, dtype=object)
    labels = []
    for index in range(9):
        points[index] = np.asarray([[index, 0.0], [index, 1.0]], dtype=float)
        labels.append(np.asarray(["A", "B"]))
    path = tmp_path / "trajectory.png"
    figure = plot_trajectory_grid(
        points,
        times,
        labels_list=labels,
        label_to_color={"A": "#ff0000", "B": "#0000ff"},
        out_path=str(path),
        n_cols=3,
        show_axes=False,
        show_legend=True,
        equal_aspect=True,
    )

    assert path.exists() and path.stat().st_size > 0
    assert len(figure.axes) == 9
    assert [axis.get_title() for axis in figure.axes] == [f"t={value}" for value in times]
    assert all(len(axis.get_xticks()) == 0 and len(axis.get_yticks()) == 0 for axis in figure.axes)
    assert len(figure.legends) == 1
    assert [text.get_text() for text in figure.legends[0].get_texts()] == ["A", "B"]
    plt.close(figure)


def test_trajectory_comparison_grid_selects_times_and_shares_limits(tmp_path):
    times = [0.0, 1.0, 2.0, 3.0, 4.0]
    trajectories = {}
    labels = {}
    for condition_index, condition in enumerate(("baseline", "remove_A", "remove_B")):
        frames = np.empty(len(times), dtype=object)
        label_frames = []
        for time_index, time_value in enumerate(times):
            n_points = condition_index + time_index + 2
            frames[time_index] = np.column_stack(
                (
                    np.linspace(condition_index, condition_index + 1, n_points),
                    np.linspace(time_value, time_value + 1, n_points),
                )
            )
            label_frames.append(np.repeat("A", n_points))
        trajectories[condition] = frames
        labels[condition] = label_frames

    path = tmp_path / "comparison.png"
    figure = plot_trajectory_comparison_grid(
        trajectories,
        times,
        out_path=str(path),
        labels_by_condition=labels,
        label_to_color={"A": "#123456"},
        selected_times=[0.0, 2.0, 4.0],
        condition_titles={"baseline": "Baseline"},
    )

    assert path.exists() and path.stat().st_size > 0
    assert len(figure.axes) == 9
    x_limits = [axis.get_xlim() for axis in figure.axes]
    y_limits = [axis.get_ylim() for axis in figure.axes]
    assert all(limit == pytest.approx(x_limits[0]) for limit in x_limits[1:])
    assert all(limit == pytest.approx(y_limits[0]) for limit in y_limits[1:])
    assert [figure.axes[index].get_title() for index in range(3)] == [
        "Baseline",
        "remove_A",
        "remove_B",
    ]
    plt.close(figure)


def test_trajectory_color_mapping_normalizes_non_string_label_keys(tmp_path):
    points = np.empty(1, dtype=object)
    points[0] = np.asarray([[0.0, 0.0], [1.0, 1.0]])
    figure = plot_trajectory_grid(
        points,
        [0.0],
        labels_list=[np.asarray([1, 2])],
        label_to_color={1: "#ff0000", 2: "#0000ff"},
        out_path=str(tmp_path / "numeric_labels.png"),
        show_legend=True,
    )
    colors = figure.axes[0].collections[0].get_facecolors()[:, :3]
    np.testing.assert_allclose(colors[0], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(colors[1], [0.0, 0.0, 1.0])
    assert [text.get_text() for text in figure.legends[0].get_texts()] == ["1", "2"]
    plt.close(figure)


def test_growth_grid_robust_scales_each_time_and_uses_shared_colorbar(
    monkeypatch, tmp_path
):
    states = {}
    for index, time_value in enumerate((0.0, 1.0, 2.0, 3.0, 4.0)):
        state = ad.AnnData(X=np.zeros((5, 3), dtype=np.float32))
        state.obsm["spatial"] = np.column_stack(
            (
                np.arange(5, dtype=float),
                np.repeat(time_value, 5),
                np.repeat(99.0, 5),
            )
        )
        state.obs["growth_rate"] = np.arange(5, dtype=float) + index
        states[str(time_value)] = state

    from matplotlib.axes import Axes

    original_scatter = Axes.scatter
    captured = []

    def wrapped_scatter(self, *args, **kwargs):
        captured.append(
            {
                "values": np.asarray(kwargs["c"], dtype=float),
                "vmin": kwargs["vmin"],
                "vmax": kwargs["vmax"],
            }
        )
        return original_scatter(self, *args, **kwargs)

    monkeypatch.setattr(Axes, "scatter", wrapped_scatter)
    path = tmp_path / "growth.png"
    plot_growth_timepoint_grid(
        states,
        time_points=[0.0, 1.0, 2.0, 3.0, 4.0],
        out_path=str(path),
        spatial_key="spatial",
        n_cols=2,
        cmap="RdYlBu_r",
        scale_mode="per_time_0_1",
        shared_colorbar=True,
        colorbar_label="g (scaled 5-95%)",
    )

    assert path.exists() and path.stat().st_size > 0
    assert len(captured) == 5
    assert all(record["vmin"] == 0.0 and record["vmax"] == 1.0 for record in captured)
    assert all(np.isfinite(record["values"]).all() for record in captured)
    assert all(record["values"].min() == 0.0 for record in captured)
    assert all(record["values"].max() == 1.0 for record in captured)
