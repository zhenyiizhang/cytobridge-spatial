import numpy as np
import pandas as pd

from CytoBridge.tl.downstream.evaluation import (
    DistributionEvaluationResult,
    compute_generated_vs_observed_plot_limits,
    plot_generated_vs_observed,
)


def _result(predicted, observed, weights=None):
    times = tuple(float(value) for value in predicted)
    if weights is None:
        weights = {
            time: np.full(len(values), 1.0 / len(values))
            for time, values in predicted.items()
        }
    return DistributionEvaluationResult(
        time_points=times,
        spatial_dim=2,
        predicted_points=predicted,
        predicted_weights=weights,
        observed_points=observed,
        metrics=pd.DataFrame(),
        settings={},
    )


def test_shared_plot_limits_pool_runs_at_each_time():
    first = _result(
        {0.0: np.array([[0, 0, 1, 2]]), 1.0: np.array([[2, 2, 4, 5]])},
        {0.0: np.array([[1, 1, 2, 3]]), 1.0: np.array([[3, 3, 5, 6]])},
    )
    second = _result(
        {0.0: np.array([[-2, -3, 0, 1]]), 1.0: np.array([[5, 4, 7, 8]])},
        {0.0: np.array([[4, 5, 3, 4]]), 1.0: np.array([[6, 7, 8, 9]])},
    )

    spatial = compute_generated_vs_observed_plot_limits(
        [first, second], space="spatial", pad_fraction=0
    )
    pca = compute_generated_vs_observed_plot_limits(
        [first, second], space="pca", pad_fraction=0
    )

    assert spatial[0.0] == ((-2.0, 4.0), (-3.0, 5.0))
    assert spatial[1.0] == ((2.0, 6.0), (2.0, 7.0))
    assert pca[0.0] == ((0.0, 3.0), (1.0, 4.0))
    assert pca[1.0] == ((4.0, 8.0), (5.0, 9.0))


def test_relative_weight_mode_maps_particle_mass_to_marker_area(tmp_path, monkeypatch):
    import matplotlib.axes

    result = _result(
        {0.0: np.array([[0, 0, 1, 2], [1, 1, 2, 3]])},
        {0.0: np.array([[0, 0, 1, 2], [1, 1, 2, 3]])},
        weights={0.0: np.array([1.0, 3.0])},
    )
    scatter_sizes = []
    original_scatter = matplotlib.axes.Axes.scatter

    def recording_scatter(self, *args, **kwargs):
        scatter_sizes.append(np.asarray(kwargs["s"]))
        return original_scatter(self, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "scatter", recording_scatter)
    out_path = tmp_path / "weighted.svg"
    plot_generated_vs_observed(
        result,
        space="spatial",
        out_path=out_path,
        point_size=2.0,
        generated_point_size_mode="relative_weight",
    )

    assert out_path.is_file()
    assert scatter_sizes[0] == np.asarray(2.0)
    np.testing.assert_allclose(scatter_sizes[1], np.array([1.0, 3.0]))
