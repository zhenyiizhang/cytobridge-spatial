"""The calculation shown on the page must supply the figure that follows it."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_s36_plots_the_given_table_and_responds_to_a_changed_value(tmp_path):
    from CytoBridge.results._zebrafish_loss import make_figure, draw
    table = pd.read_csv(ROOT / "CytoBridge/results/data/zebrafish_si/s32_loss_weight_metrics.csv")
    mask = table.condition.eq("formal_alpha_control") & table.space.eq("joint") & table.time.eq(1)
    expected = table.loc[mask, "w1"].item()
    original = make_figure(table)
    changed = table.copy()
    changed.loc[mask, "w1"] = expected + .25
    revised = make_figure(changed)
    try:
        assert original.axes[0].patches[0].get_height() == pytest.approx(expected)
        assert revised.axes[0].patches[0].get_height() == pytest.approx(expected + .25)
        # A caller edit changes only the corresponding bar.
        np.testing.assert_array_equal([p.get_height() for p in original.axes[0].patches[1:]],
                                      [p.get_height() for p in revised.axes[0].patches[1:]])
    finally:
        plt.close(original)
        plt.close(revised)
    paths = draw(changed, tmp_path)
    assert all(p.is_file() for p in paths)
    saved = pd.read_csv(tmp_path / "loss_weight_metrics.csv")
    pd.testing.assert_frame_equal(saved, changed.loc[:, saved.columns], check_exact=False)
    summary = pd.read_csv(tmp_path / "mean_w1.csv")
    value = summary.query("condition == 'formal_alpha_control' and space == 'joint'").mean_w1.item()
    assert value == pytest.approx(changed.query("condition == 'formal_alpha_control' and space == 'joint'").w1.mean())


def test_numeric_evaluation_export_does_not_call_plotter(tmp_path, monkeypatch):
    import CytoBridge.tl.downstream.evaluation as module
    def unexpected(*args, **kwargs):
        raise AssertionError("Numerical export should not draw a diagnostic figure")
    monkeypatch.setattr(module, "plot_generated_vs_observed", unexpected)
    points = {1.: np.ones((3, 4), dtype=np.float32)}
    result = SimpleNamespace(time_points=(1.,), spatial_dim=2,
        predicted_points=points, observed_points=points, predicted_weights={1.: np.ones(3)},
        metrics=pd.DataFrame({"time": [1.], "space": ["joint"], "w1": [.2]}))
    paths = module.save_distribution_evaluation(result, tmp_path, save_figures=False)
    assert set(paths) == {"metrics", "samples"}
    assert sorted(p.suffix for p in tmp_path.iterdir()) == [".csv", ".npz"]


def test_loss_table_requires_complete_nonduplicated_measurements():
    from CytoBridge.results._zebrafish_loss import validate_metrics
    table = pd.read_csv(ROOT / "CytoBridge/results/data/zebrafish_si/s32_loss_weight_metrics.csv")
    for incomplete in (table.iloc[:-1], pd.concat([table, table.iloc[[0]]])):
        with pytest.raises(ValueError, match="exactly once"):
            validate_metrics(incomplete)


def test_tutorial_uses_new_results_and_a_consistent_model_directory():
    book = json.loads((ROOT / "docs/tutorials/paper_figures/zebrafish_si_s31_s38.ipynb").read_text())
    cells = {cell["id"]: "".join(cell["source"]) for cell in book["cells"]}
    for cell in book["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
    assert "destination=project" in cells["setup"]
    assert "draw_supplementary" not in "\n".join(cells.values())
    for name in ("removal-calculate", "ysl-calculate-plot", "daughter-calculate"):
        assert "model_dir=model_dir" in cells[name]
    for name in ("population-calculate", "reconstruction-calculate"):
        assert '"--model-dir", model_dir' in cells[name]
        assert '"--calculate-only"' in cells[name]
    assert 'plot_loss_weights(loss_metrics,' in cells["loss-plot"]
    assert 'plot_daughter_results(daughter_tables,' in cells["daughter-plot"]
    assert 'plot_gene_reconstruction(model_results,' in cells["reconstruction-plot"]


def test_s34_uses_current_paper_style_without_changing_centroid_values(tmp_path, monkeypatch):
    from CytoBridge.results import _zebrafish_si_plot as renderer
    from CytoBridge.results.zebrafish_si import load_zebrafish_si_results, calculate_zebrafish_si_panels
    from matplotlib.colors import to_hex

    results = load_zebrafish_si_results()
    panels = calculate_zebrafish_si_panels(results)
    figures = []
    monkeypatch.setattr(renderer, "_save", lambda figure, *args: figures.append(figure))
    renderer._render_virtual_removal_quantitative(results, panels, tmp_path)
    figure = figures[0]
    try:
        centroid = figure.axes[-1]
        assert [to_hex(p.get_facecolor()) for p in centroid.patches] == ["#2166ac", "#b2182b"]
        expected = panels.ablation_centroid_summary.set_index("variant").loc[
            ["remove_YSL", "remove_EVL"], "mean"].to_numpy()
        np.testing.assert_allclose([p.get_height() for p in centroid.patches], expected)
        assert not any(t.get_bbox_patch() for t in centroid.texts)
        assert not any(line.get_visible() for line in centroid.get_xgridlines())
        assert "Centroid shift" in [t.get_text() for t in centroid.texts]
    finally:
        plt.close(figure)
