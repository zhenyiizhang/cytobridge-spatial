"""Reader calculations must supply the values consumed by the next plot."""
import importlib.util
import json
from pathlib import Path
import pickle
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]


def notebook(name):
    return json.loads((ROOT / f'docs/tutorials/paper_figures/{name}.ipynb').read_text())


def cell_source(book, marker):
    return next(''.join(cell['source']) for cell in book['cells']
                if cell['cell_type'] == 'code' and marker in ''.join(cell['source']))


def test_figure5_notebook_passes_calculated_velocity_and_growth(monkeypatch, tmp_path):
    ad = pytest.importorskip('anndata')
    from reproduction.arista import main_figure, plotting

    book = notebook('main_figure_5')
    scope = dict(np=np, pd=pd, ad=ad, SOURCE=main_figure.SOURCE, output=tmp_path,
                 plotting=plotting, ROI=main_figure.ROI, FOCUS=main_figure.FOCUS,
                 palette={}, display=lambda value: None, show=lambda value: None)
    exec(cell_source(book, 'with np.load(SOURCE'), scope)
    # A reader's changed calculation must survive the following plotting cell.
    scope['velocity_table']['cosine_full_vs_interaction'] *= -1
    expected = scope['velocity_table']['cosine_full_vs_interaction'].copy()

    def velocity_plot(population, table, roi, focus, palette, output):
        assert table is scope['velocity_table']
        assert population is scope['velocity_population']
        pd.testing.assert_series_equal(table.cosine_full_vs_interaction, expected)
        return {'Figure5c_spatial_and_roi': []}

    monkeypatch.setattr(plotting, 'plot_figure5c', velocity_plot)
    exec(cell_source(book, 'velocity_table.to_csv('), scope)
    np.testing.assert_allclose(pd.read_csv(tmp_path / 'Figure5c_spatial_velocity.csv')
                               .cosine_full_vs_interaction, expected)

    exec(cell_source(book, 'values = pd.read_csv('), scope)
    assert len(scope['values']) == 82306 and len(scope['means']) == 177
    scope['means']['growth_mean'] += .125
    expected_means = scope['means'].copy()

    def growth_plot(table, stem):
        assert table is scope['means']
        pd.testing.assert_frame_equal(table, expected_means)
        return []

    scope['plot_growth_interaction'] = growth_plot
    exec(cell_source(book, 'means.to_csv('), scope)
    pd.testing.assert_frame_equal(pd.read_csv(tmp_path / 'Figure5e_growth_interaction.csv'),
                                  expected_means, check_dtype=False)


@pytest.mark.parametrize('name,data_variable,numbers', [
    ('agist_figures', 'data', [2, 3]),
    ('nonspatial_figures', 'results', [4, 5]),
    ('arista_local_domains', 'data', [25]),
])
def test_notebook_transports_its_calculated_objects(monkeypatch, tmp_path, name, data_variable, numbers):
    from reproduction import paper_figures

    data = SimpleNamespace(source_dir=tmp_path / 'inputs', changed=np.array([7., 9.]))
    panels = SimpleNamespace(changed=pd.DataFrame({'value': [123., 456.]}))
    transported = []

    def plot_process(command, **kwargs):
        path = Path(command[command.index('--calculated-inputs') + 1])
        with path.open('rb') as handle:
            received_data, received_panels = pickle.load(handle)
        np.testing.assert_array_equal(received_data.changed, data.changed)
        pd.testing.assert_frame_equal(received_panels.changed, panels.changed)
        transported.append(path)
        # The test checks transport, not rendering. Empty fixture files satisfy
        # the wrapper's existence check without claiming to be figures.
        output = Path(command[command.index('--output-dir') + 1])
        for number in numbers:
            for suffix in ('pdf', 'png'):
                (output / f'S{number}.{suffix}').touch()
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(paper_figures.subprocess, 'run', plot_process)
    scope = {data_variable: data, 'panels': panels, 'output_dir': tmp_path / 'figures',
             'write_arista_local_domain_tables': lambda *args: {},
             'Image': lambda **kwargs: None, 'display': lambda value: None}
    exec(cell_source(notebook(name), 'from reproduction.paper_figures import'), scope)
    assert len(transported) == 1
    assert not transported[0].exists()


def test_s2_plot_uses_changed_cell_scores_and_precalculated_medians(tmp_path):
    from CytoBridge.results import load_agist_figures, calculate_agist_figure_panels
    from reproduction.paper_figures import draw_supplementary

    data = load_agist_figures()
    data.velocity_per_cell['physical_cosine'] -= .025
    panels = calculate_agist_figure_panels(data)
    output = tmp_path / 'figures'
    draw_supplementary([2], output, data=data, panels=panels)
    plotted = pd.read_csv(output / 'tables/S2_medians.csv')
    expected = panels.velocity_by_time.query("velocity_space == 'physical'").sort_values('time')
    actual = plotted.query("space == 'physical' and partition == 'time'").sort_values('category')
    np.testing.assert_allclose(actual['median'], expected['median'], atol=1e-12)
    assert (output / 'S2.pdf').stat().st_size > 1000
    assert (output / 'S2.png').stat().st_size > 1000


@pytest.mark.parametrize('api_name,loader,calculator,figure_numbers', [
    ('agist_figures', 'load_agist_figures', 'calculate_agist_figure_panels', (3,)),
    ('nonspatial_figures', 'load_nonspatial_figures', 'calculate_nonspatial_panels', (4, 5)),
    ('arista_local_domains', 'load_arista_local_domains', 'calculate_arista_local_domain_panels', (25,)),
])
def test_accepted_renderers_use_supplied_objects_without_reloading(
        monkeypatch, tmp_path, api_name, loader, calculator, figure_numbers):
    import importlib
    monkeypatch.syspath_prepend(str(ROOT / 'reproduction/supplementary_figures'))
    render = importlib.import_module('plot_panels')
    domains = importlib.import_module('plot_domains')
    summaries = importlib.import_module('plot_summaries')
    api = importlib.import_module('CytoBridge.results.' + api_name)
    data = getattr(api, loader)()
    calculated = getattr(api, calculator)(data)

    def forbid_reload(*args, **kwargs):
        raise AssertionError('Supplied objects must not be reloaded or recalculated')

    monkeypatch.setattr(api, loader, forbid_reload)
    monkeypatch.setattr(api, calculator, forbid_reload)
    tables = tmp_path / 'tables'
    tables.mkdir()
    for module in (render, domains, summaries):
        monkeypatch.setattr(module, 'OUT', tmp_path)
        monkeypatch.setattr(module, 'TABLES', tables, raising=False)
    monkeypatch.setattr(domains, 'save', summaries.save)
    render.defaults()
    if figure_numbers == (3,):
        render.agist(data=data, panel_values=calculated)
    elif figure_numbers == (4, 5):
        render.nonspatial(clone_values=True, data=data, panel_values=calculated)
    else:
        domains.s25(data=data, panel_values=calculated)
    for number in figure_numbers:
        assert (tmp_path / f'S{number}.pdf').stat().st_size > 1000
        assert (tmp_path / f'S{number}.png').stat().st_size > 1000


def test_arista_generator_keeps_current_population_and_calculation_routes(monkeypatch, tmp_path):
    # Supply a tiny nbformat stand-in to inspect generator cells without
    # rebuilding or overwriting any notebook or its recorded outputs.
    import sys
    module_name = 'build_numerical_paper_tutorials_continuity_test'
    fake_nbf = SimpleNamespace(v4=SimpleNamespace(
        new_markdown_cell=lambda text: SimpleNamespace(cell_type='markdown', source=text),
        new_code_cell=lambda text: SimpleNamespace(cell_type='code', source=text)))
    monkeypatch.setitem(sys.modules, 'nbformat', fake_nbf)
    spec = importlib.util.spec_from_file_location(module_name, ROOT / 'scripts/build_numerical_paper_tutorials.py')
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    generated = {}
    monkeypatch.setattr(generator, 'write', lambda name, cells: generated.update({name: cells}))
    generator.arista_main()
    generator.arista_supplementary()
    main = generated['main_figure_5.ipynb']
    source = '\n'.join(cell.source for cell in main)
    assert '7,780' in source and '7,798' not in source
    assert 'spatially anchored display coordinates' not in source
    for marker in ('velocity_table.to_csv(', 'means.to_csv('):
        emitted = next(cell.source for cell in main if marker in cell.source)
        assert emitted == cell_source(notebook('main_figure_5'), marker)
    si_source = '\n'.join(cell.source for cell in generated['arista_figures.ipynb'])
    assert 'draw_supplementary(populations, output, figures=[19])' in si_source


def test_calculated_panel_contract_rejects_incomplete_or_mismatched_inputs(tmp_path):
    from reproduction.paper_figures import draw_supplementary

    data = SimpleNamespace(source_dir=tmp_path / 'input')
    with pytest.raises(ValueError, match='both data and panels'):
        draw_supplementary([2], tmp_path / 'output', data=data)
    with pytest.raises(ValueError, match='must match'):
        draw_supplementary([2], tmp_path / 'output', data=data, panels=object(),
                           results_dir=tmp_path / 'other')
    with pytest.raises(ValueError, match='outside the input'):
        draw_supplementary([2], data.source_dir, data=data, panels=object())


def test_arista_simulation_exports_same_states_and_calculates_communication(monkeypatch, tmp_path):
    ad = pytest.importorskip('anndata')
    import yaml
    from reproduction.arista import main_figure, simulate_paper_populations as producer

    times = list(np.arange(0., 4.01, .5))
    selected, communication = {}, {}
    for time in times:
        cells = ad.AnnData(np.array([[time, 1., 10.], [time, 2., 20.]], dtype=np.float32))
        cells.obs['Annotation'] = ['first', 'second']
        cells.obsm['spatial'] = np.asarray(cells.X)[:, :2].copy()
        selected[str(time)] = cells
        communication[str(time)] = cells.copy()
    result = SimpleNamespace(
        ts_points=times, adata_dict=selected, communication_adata_dict=communication,
        sde_points_split=[selected[str(time)].X.copy() for time in times],
        slice_labels_split=[['first', 'second'] for time in times],
        predicted_labels_list=[['first', 'second'] for time in times])
    data = tmp_path / 'inputs'
    data.mkdir()
    selected['0.0'].write_h5ad(data / 'aligned.h5ad')
    monkeypatch.setattr(producer.cb.tl, 'adata_to_aligned_dataframe', lambda *a, **k: (pd.DataFrame(), 'time'))
    monkeypatch.setattr(producer.cb.tl, 'load_dynamical_model_from_dir', lambda *a, **k: object())
    runtime = SimpleNamespace(f_net=object())
    monkeypatch.setattr(producer.cb.tl, 'build_dynamical_runtime', lambda *a, **k: runtime)
    captured = {}

    def simulation(**kwargs):
        captured['simulation'] = kwargs
        return result

    def compute_communication(**kwargs):
        captured['communication'] = kwargs
        with Path(kwargs['save_pickle_path']).open('wb') as handle:
            pickle.dump({'computed_for': list(kwargs['adata_dict'])}, handle)

    monkeypatch.setattr(producer.cb.tl, 'run_interpolation_workflow', simulation)
    monkeypatch.setattr(producer.cb.tl, 'compute_timepoint_communications', compute_communication)
    output = producer.generate(data, tmp_path / 'new_run', data / 'selected_classifier.pt', device='cpu')
    assert captured['simulation']['classifier_cache_path'] == str(data / 'selected_classifier.pt')
    assert captured['simulation']['spatial_warp_to_observed'] is False
    assert captured['communication']['adata_dict'] is communication
    assert captured['communication']['f_net'] is runtime.f_net
    config = yaml.safe_load((ROOT / 'CytoBridge/configs/arista_downstream.yaml').read_text())['communication']
    for key, value in config.items():
        assert captured['communication'][key] == value
    assert json.loads((output / 'communication_settings.json').read_text()) == config
    assert (output / 'all_time_communications.pkl').is_file()
    loaded = main_figure.load_populations(output)
    for key, cells in loaded.items():
        np.testing.assert_array_equal(cells.X, selected[key].X)
        np.testing.assert_array_equal(cells.obsm['spatial'], selected[key].X[:, :2])
