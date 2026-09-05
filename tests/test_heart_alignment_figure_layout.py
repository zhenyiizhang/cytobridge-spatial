"""Display order and legend placement for Supplementary Figures S7 and S8."""

import importlib.util
from pathlib import Path

import pytest

mpl = pytest.importorskip("matplotlib")
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


@pytest.fixture
def plotter(monkeypatch):
    script = (
        Path(__file__).resolve().parents[1]
        / "release_artifacts/chicken_heart_alignment_sensitivity_20260831"
        / "figure_code/plot_heart_alignment_sensitivity.py"
    )
    monkeypatch.syspath_prepend(str(script.parent))
    spec = importlib.util.spec_from_file_location("heart_alignment_plotter", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_early_sections_drawn_last_without_changing_coordinates(plotter):
    stages = np.array(["D4", "D7", "D10", "D14"])
    xy = np.arange(8, dtype=float).reshape(4, 2)
    fig, ax = plt.subplots()
    try:
        plotter.scatter_stages(ax, xy, stages, size=3.3)
        assert plotter.DRAW_ORDER == ("D14", "D10", "D7", "D4")
        assert plotter.TIME_ORDER == ("D4", "D7", "D10", "D14")
        assert [c.get_zorder() for c in ax.collections] == [1, 2, 3, 4]
        for collection, stage in zip(ax.collections, plotter.DRAW_ORDER):
            np.testing.assert_array_equal(collection.get_offsets(), xy[stages == stage])
            assert not collection.get_rasterized()
            assert collection.get_alpha() == 0.80
    finally:
        plt.close(fig)


def test_s7_legend_has_separate_row(plotter, monkeypatch):
    def check_layout(fig, *args, **kwargs):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        legends = [ax.get_legend() for ax in fig.axes if ax.get_legend() is not None]
        assert len(legends) == 1
        legend = legends[0]
        assert [text.get_text() for text in legend.get_texts()] == list(plotter.TIME_ORDER)
        legend_box = legend.get_window_extent(renderer)
        section_axes = [ax for ax in fig.axes if ax.get_title() in ("Input sections", "Aligned sections")]
        assert len(section_axes) == 2
        for ax in section_axes:
            assert legend_box.y1 < ax.get_window_extent(renderer).y0

    monkeypatch.setattr(plotter, "save_figure", check_layout)
    plotter.plot_s7(plotter.load_plot_inputs())
