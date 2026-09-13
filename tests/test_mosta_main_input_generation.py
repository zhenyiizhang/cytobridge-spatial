"""Figure 4 calculates its derived inputs and plots those same results."""
import ast
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pytest

from reproduction.mosta.calculations import cartilage_lineage_inputs, save_population_states

ROOT = Path(__file__).resolve().parents[1]


def test_lineage_selects_the_same_particles_and_responds_to_new_labels():
    states = [np.arange(20).reshape(5, 4), np.arange(20, 40).reshape(5, 4)]
    labels = [np.array(['other', 'Cartilage primordium', 'other', 'Cartilage primordium', 'other']),
              np.array(['other', 'Cartilage', 'other', 'Connective tissue', 'other'])]
    reference = ad.AnnData(np.zeros((3, 4)))
    reference.obs['time_point_processed'] = [2.5, 3., 3.]
    reference.obsm['spatial_aligned'] = np.arange(6).reshape(3, 2)
    result = cartilage_lineage_inputs(states, labels, [2.5, 3.], reference)
    np.testing.assert_array_equal(result['selected_lineage_id'], [1, 3])
    np.testing.assert_array_equal(result['target_spatial'], states[1][[1, 3], :2])
    labels[1][1] = 'other'
    changed = cartilage_lineage_inputs(states, labels, [2.5, 3.], reference)
    assert changed['target_labels'][0] == 'other'
    with pytest.raises(ValueError, match='same particles'):
        cartilage_lineage_inputs([states[0], states[1][:2]], labels, [2.5, 3.], reference)


def test_population_export_writes_input_consumed_by_plot(tmp_path):
    cells = ad.AnnData(np.array([[1., 2., 3.]]))
    cells.obs['Annotation'] = ['Brain']
    cells.obsm['spatial'] = cells.X[:, :2].copy()
    save_population_states({'0.5': cells}, tmp_path)
    actual = ad.read_h5ad(tmp_path / 'time_0p5.h5ad')
    np.testing.assert_array_equal(actual.X, cells.X)


def test_notebook_computes_model_inputs_instead_of_loading_derived_tables():
    notebook = json.loads((ROOT / 'docs/tutorials/paper_figures/main_figure_4.ipynb').read_text())
    source = '\n'.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code')
    tree = ast.parse(source)
    calls = {ast.unparse(n.func): n for n in ast.walk(tree) if isinstance(n, ast.Call)}
    for name in ('run_interpolation_workflow', 'compute_timepoint_communications',
                 'compute_focal_lr_type_hotspots', 'simulate_sde_points', 'predict_labels_for_trajectories'):
        assert f'cb.tl.{name}' in calls
    assert 'pd.read_csv' not in calls
    assert 'np.load' not in calls
    assert ast.unparse(next(k.value for k in calls['draw_cartilage'].keywords if k.arg == 'arrays')) == 'lineage'
    assert ast.unparse(next(k.value for k in calls['draw_interaction_maps'].keywords if k.arg == 'mapping')) == 'mapping'
    assert ast.unparse(calls['map_interaction_scores'].args[1]) == 'lr.type_scores'


def test_cartilage_plot_accepts_new_arrays_without_reopening_archive(monkeypatch, tmp_path):
    from reproduction.mosta import main_figure, cartilage
    supplied = {'target_labels': np.array(['Cartilage', 'Other']),
                'selected_lineage_id': np.array([4, 8]),
                'target_spatial': np.array([[1., 2.], [3., 4.]])}
    monkeypatch.setattr(main_figure.np, 'load', lambda *a, **k: pytest.fail('Reopened saved lineage'))
    received = []
    def scatter(arrays, transitions, palette, path):
        received.append(arrays)
        return {}, None
    monkeypatch.setattr(cartilage, 'create_scatter_layer', scatter)
    monkeypatch.setattr(cartilage, 'assemble_panel', lambda *a, **kw: None)
    main_figure.draw_cartilage(tmp_path, {}, arrays=supplied)
    assert received == [supplied]


def test_figure4_generator_matches_the_published_calculation_cells():
    pytest.importorskip('nbformat')
    import importlib.util
    spec = importlib.util.spec_from_file_location('figure4_builder', ROOT / 'scripts/build_mosta_main_tutorial.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    actual = json.loads((ROOT / 'docs/tutorials/paper_figures/main_figure_4.ipynb').read_text())
    cells = lambda nb: [(c['cell_type'], ''.join(c['source'])) for c in nb['cells']]
    assert cells(module.build_notebook()) == cells(actual)
