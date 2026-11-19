import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from CytoBridge.pl.growth_summary import plot_growth_heatmap, plot_growth_size_maps


def test_heatmap_uses_supplied_values_order_and_missing_cells():
    matrix = pd.DataFrame([[1, np.nan, 3], [4, 5, 6]], index=["B", "A"], columns=["D4", "D5", "D7"])
    ax = plot_growth_heatmap(matrix, observed_times=["D4", "D7"])
    np.testing.assert_allclose(ax.images[0].get_array().filled(np.nan), matrix.values, equal_nan=True)
    assert [label.get_text() for label in ax.get_yticklabels()] == ["B", "A"]
    assert len(ax.texts) == 5
    assert len(ax.patches) == 1
    plt.close(ax.figure)


def example():
    return pd.DataFrame(dict(time_key=["D4", "D4", "D5", "D5"],
                             celltype=["A", "B", "A", "B"],
                             growth=[0., 1., 1., 2.], x=[0., 1., 2., 3.], y=[0., 0., 1., 1.]))


def test_maps_share_growth_and_coordinate_scales_without_changing_input():
    frame = example()
    original = frame.copy(deep=True)
    fig, axes = plot_growth_size_maps(frame, palette={"A": "red", "B": "blue"},
                                     time_order=["D4", "D5"], observed_times=["D4"])
    pd.testing.assert_frame_equal(frame, original)
    # Identical values at different stages must have identical marker areas.
    np.testing.assert_allclose(axes[0].collections[1].get_sizes(), axes[1].collections[0].get_sizes())
    np.testing.assert_allclose(axes[1].collections[1].get_offsets(), [[1, 1/3]])
    assert not axes[0].patches and len(axes[1].patches) == 1
    assert sum(len(c.get_offsets()) for a in axes for c in a.collections) == len(frame)
    plt.close(fig)


@pytest.mark.parametrize("change", ["missing_colour", "omit_time", "duplicate_time", "nan_growth"])
def test_invalid_inputs_are_not_silently_dropped(change):
    frame = example()
    palette = {"A": "red", "B": "blue"}
    times = ["D4", "D5"]
    if change == "missing_colour":
        palette.pop("B")
    elif change == "omit_time":
        times = ["D4"]
    elif change == "duplicate_time":
        times.append("D4")
    else:
        frame.loc[0, "growth"] = np.nan
    with pytest.raises(ValueError):
        plot_growth_size_maps(frame, palette=palette, time_order=times)
