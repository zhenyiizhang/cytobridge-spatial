"""Small regression tests for the actual ARISTA producer/plot contracts."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


def test_spatial_projection_uses_physical_neighbors_but_52d_directions(monkeypatch):
    from reproduction.arista import spatial_velocity as module
    captured = {}
    def neighbors(population, **kwargs):
        captured.update(kwargs)
        assert population.X.shape == (40, 52)
        assert population.layers['velocity'].shape == (40, 52)
        assert population.obsm['X_spatial'].shape == (40, 2)
    def embedding(population, **kwargs):
        population.obsm['velocity_spatial'] = np.ones((40, 2))
    monkeypatch.setattr(module.sc.pp, 'neighbors', neighbors)
    monkeypatch.setattr(module.scv.tl, 'velocity_graph', lambda *a, **k: None)
    monkeypatch.setattr(module.scv.tl, 'velocity_embedding', embedding)
    result = module.project_velocity(np.zeros((40, 52)), np.zeros((40, 2)), np.ones((40, 52)))
    assert captured['use_rep'] == 'X_spatial' and captured['n_neighbors'] == 30
    np.testing.assert_array_equal(result, np.ones((40, 2)))


def test_observed_fields_restore_simulation_rng_before_calculation(monkeypatch, tmp_path):
    import torch
    import anndata as ad
    from reproduction.arista import model_fields as module
    ad.AnnData(np.zeros((4, 52), dtype=np.float32)).write_h5ad(tmp_path / 'aligned.h5ad')
    torch.manual_seed(713)
    rng = torch.get_rng_state().numpy()
    np.savez(tmp_path / 'post_simulation_rng.npz', cpu=rng)
    expected = torch.rand(4)
    monkeypatch.setattr(module.cb.tl, 'load_dynamical_model_from_dir', lambda *a, **k:
                        SimpleNamespace(model=SimpleNamespace(interaction_net=SimpleNamespace(cutoff=.031541))))
    def compute(reference, model, **kwargs):
        torch.testing.assert_close(torch.rand(4), expected)
        assert kwargs['interaction_m'] == 1024 and kwargs['reuse_if_present'] is False
        assert kwargs['time_key'] == 'time_point_processed'
        fields = {k: np.ones((4, 52), np.float32) for k in ['features', 'full', 'interaction', 'drift', 'score']}
        fields['times'] = np.array([0., 0., 1., 1.])
        return fields
    monkeypatch.setattr(module.cb.tl, 'compute_velocity_components_from_adata', compute)
    output = module.calculate_observed_fields(tmp_path, tmp_path, tmp_path / 'fields', device='cpu')
    with np.load(output / 'velocity_time_1.npz') as values:
        assert values['features'].shape == (2, 52)
    with pytest.raises(FileExistsError):
        module.calculate_observed_fields(tmp_path, tmp_path, output, device='cpu')


def test_lr_calculation_matches_archived_public_algorithm():
    from CytoBridge.results import load_arista_supplementary_figures, calculate_arista_ligand_receptor_panels
    from reproduction.arista.analysis import calculate_lr_panels
    data = load_arista_supplementary_figures()
    expected = calculate_arista_ligand_receptor_panels(data)
    actual = calculate_lr_panels(data.tables['ligand_receptor_all_pair_timecourse'])
    for name in ['prototypes', 'assignments', 'normalized_profiles', 'k_selection',
                 'display_roster', 'display_timecourse']:
        pd.testing.assert_frame_equal(getattr(actual, name), getattr(expected, name), check_exact=False)


def test_supplementary_passes_explicit_gene_tables_and_lr_panels(monkeypatch, tmp_path):
    from reproduction.arista import supplementary as module
    from CytoBridge.results import _arista_supplementary_figures_plot as renderer
    tables, panels = tmp_path / 'new_gene_tables', object()
    def genes(output, tables_dir):
        assert tables_dir == tables
        return []
    def lr(received, output):
        assert received is panels
        return {'S23': (), 'S24': ()}
    monkeypatch.setattr(module, 'draw_gene_programs', genes)
    monkeypatch.setattr(renderer, 'render_arista_ligand_receptor_figures', lr)
    module.draw_supplementary(tmp_path / 'states', tmp_path / 'figures',
                              figures=[22, 23, 24], tables_dir=tables, lr_panels=panels)


def test_figure5e_comparison_does_not_modify_or_replace_new_values(tmp_path):
    from reproduction.arista.figure5e_check import compare_evaluations
    source = Path(__file__).resolve().parents[1] / 'reproduction/arista/data/figure5e_growth_interaction_by_cell.csv'
    paper = pd.read_csv(source)
    actual = paper.copy()
    actual['interaction'] *= 1.25
    expected = actual.copy()
    result = compare_evaluations(actual, paper, tmp_path / 'comparison')
    pd.testing.assert_frame_equal(actual, expected)
    np.testing.assert_allclose(result['groups'].interaction_mean_new,
                               result['groups'].interaction_mean_paper * 1.25)
    assert result['rankings'].same_top3_set.all()
    # Floating-point averaging can break exact ties after scaling.
    assert result['statistics'].iloc[0].interaction_spearman > .999


def test_supplementary_cli_passes_explicit_calculated_inputs(monkeypatch, tmp_path):
    from reproduction.arista import supplementary as module, analysis
    table = pd.DataFrame({'pair_name': ['new-pair'], 'time': [0.], 'score': [71.]})
    source = tmp_path / 'pair_timecourse.csv'
    table.to_csv(source, index=False)
    panels = object()
    def calculate(received):
        pd.testing.assert_frame_equal(received, table)
        return panels
    def draw(data, output, figures, **kwargs):
        assert figures == [22, 23, 24]
        assert kwargs == {'tables_dir': tmp_path / 'genes', 'lr_panels': panels}
        return {}
    monkeypatch.setattr(analysis, 'calculate_lr_panels', calculate)
    monkeypatch.setattr(module, 'draw_supplementary', draw)
    module.main(['--data-dir', str(tmp_path / 'states'), '--output-dir', str(tmp_path / 'figures'),
                 '--figures', '22', '23', '24', '--tables-dir', str(tmp_path / 'genes'),
                 '--lr-input', str(source)])


def test_supplementary_cli_missing_explicit_lr_does_not_reload_defaults(tmp_path):
    from reproduction.arista.supplementary import main
    with pytest.raises(FileNotFoundError):
        main(['--output-dir', str(tmp_path / 'figures'), '--figures', '23',
              '--lr-input', str(tmp_path / 'missing.csv')])


def test_optional_training_selects_new_model_without_overwriting_download(monkeypatch, tmp_path):
    import json
    import CytoBridge as cb
    root = Path(__file__).resolve().parents[1]
    book = json.loads((root / 'docs/tutorials/dataset_workflows/arista.ipynb').read_text())
    source = next(''.join(c['source']) for c in book['cells']
                  if c['cell_type'] == 'code' and 'TRAIN_MODEL = False' in ''.join(c['source']))
    scope = dict(Path=Path, cb=cb, data=tmp_path / 'data', analysis=tmp_path / 'new_run', device='cpu')
    calls = []
    monkeypatch.setattr(cb.tl, 'fit', lambda *a, **k: calls.append((a, k)))
    exec(source, scope)
    assert calls == [] and scope['model_dir'] == tmp_path / 'data/model'
    exec(source.replace('TRAIN_MODEL = False', 'TRAIN_MODEL = True'), scope)
    assert scope['model_dir'] == tmp_path / 'new_run/trained_model'
    args, kwargs = calls.pop()
    assert args == (tmp_path / 'data/aligned.h5ad',)
    assert kwargs['ckpt_dir'] == scope['model_dir']
    assert kwargs['spatial_key'] == 'spatial_aligned' and kwargs['obsm_key'] == 'X_latent'
    assert kwargs['evaluate_after_training'] is False


def test_growth_loads_the_explicitly_selected_model(monkeypatch, tmp_path):
    from reproduction.arista import analysis
    selected = tmp_path / 'new_model'
    captured = []
    monkeypatch.setattr(analysis.cb.tl, 'load_dynamical_model_from_dir',
                        lambda path, **k: captured.append(path) or SimpleNamespace(model=object()))
    monkeypatch.setattr(analysis, 'load_states', lambda path: {})
    monkeypatch.setattr(analysis.cb.tl, 'evaluate_growth_by_timepoint', lambda *a, **k: pd.DataFrame({'growth': [2.]}))
    analysis.calculate_growth(tmp_path, tmp_path, tmp_path / 'growth', device='cpu', model_dir=selected)
    assert captured == [selected]
