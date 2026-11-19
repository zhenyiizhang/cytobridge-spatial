"""Check the shared Figure 6f renderer and its display-only changes."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def load_drawer():
    spec = importlib.util.spec_from_file_location(
        'ad_drawer', ROOT / 'reproduction/admouse/draw_figures.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_package_entry_calls_shared_renderer(tmp_path, monkeypatch):
    drawer = load_drawer()
    source = drawer.source_module('admouse_trem2_plot')
    panel = drawer.source_module('ad_main_figuref')
    called = []
    monkeypatch.setattr(panel, 'draw_figure', lambda src, out: called.append((src, out)))
    monkeypatch.setattr(drawer, 'source_module',
                        lambda name: source if name == 'admouse_trem2_plot' else panel)
    drawer.draw_trem2_spatial(tmp_path / 'data', tmp_path / 'out')
    assert called == [(source, tmp_path / 'out')]
    assert source.SOURCE_DIR == tmp_path / 'data/trem2/figure_source'


def test_focus_preserves_composition_and_rotates_only_display(tmp_path, monkeypatch):
    drawer = load_drawer()
    panel = drawer.source_module('ad_main_figuref')
    source = panel.load_source_module()
    labels = np.array(['Microglia', 'Excitatory neurons', 'Astrocytes'])
    xy = np.array([[1., 2.], [3., 4.], [5., 6.]])
    case = dict(xy=xy, state=xy, labels=labels,
                edge_index=np.empty((2, 0), dtype=int), attention=np.empty(0))
    monkeypatch.setattr(source, 'load_case', lambda *args: case)
    captured = {}
    def capture(src, cases, composition, window, vmax, **kwargs):
        captured.update(kwargs)
        captured['cases'] = cases
        captured['window'] = window
    monkeypatch.setattr(panel, 'draw_panel_a', capture)
    panel.draw_figure(source, tmp_path)
    assert source.CELLTYPE_ORDER[-1] == 'Microglia'
    assert source.CELLTYPE_COLORS['Astrocytes'] == '#d9d9d9'
    assert captured['composition_colors']['Astrocytes'] == '#1f77b4'
    assert captured['focus_other_alpha'] == 0.25
    table = pd.read_csv(tmp_path / 'tables/whole_tissue_composition_endpoint.csv')
    np.testing.assert_allclose(table.groupby('condition').fraction.sum(), 1)
    assert table.query("celltype == 'Microglia'")['count'].tolist() == [1, 1, 1]
    rotated, _ = panel.rotate_cases_counterclockwise(
        captured['cases'], captured['window'], source)
    np.testing.assert_array_equal(rotated['baseline']['xy'], [[-2, 1], [-4, 3], [-6, 5]])
    np.testing.assert_array_equal(xy, [[1, 2], [3, 4], [5, 6]])
    np.testing.assert_allclose(source.ROI_CENTER, [-.65, .55])
