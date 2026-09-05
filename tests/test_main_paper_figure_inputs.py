"""Regression checks for the selected Figure 4/5 numerical inputs."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
ARISTA = ROOT / 'reproduction/arista/data'


def test_figure5d_is_intrinsic_not_previous_full_field():
    with np.load(ARISTA / 'figure5d_intrinsic_gene_velocity_state.npz', allow_pickle=False) as chosen:
        assert str(chosen['velocity_component']) == 'drift'
        assert len(chosen['labels']) == 46199
        assert chosen['display_mask'].all()
        velocity = chosen['embedded_gene_velocity_pca']
    with np.load(ARISTA / 'figure5d_corrected_gene_velocity_state.npz', allow_pickle=False) as previous:
        assert not np.allclose(velocity, previous['embedded_gene_velocity_pca'])


def test_figure5a_reads_original_unwarped_population_files(monkeypatch):
    pytest.importorskip('matplotlib')
    ad = pytest.importorskip('anndata')
    from reproduction.arista import main_figure
    paths = []
    def read(path):
        paths.append(Path(path))
        result = ad.AnnData(np.array([[1., 2., 3.], [4., 5., 6.]]))
        result.obsm['spatial'] = np.zeros((2, 2))
        return result
    monkeypatch.setattr(main_figure.ad, 'read_h5ad', read)
    result = main_figure.load_populations(Path('data/arista/paper'))
    assert len(paths) == 5
    assert all(path.parent.name == 'slice_data' for path in paths)
    for cells in result.values():
        np.testing.assert_array_equal(cells.obsm['spatial'], cells.X[:, :2])


def test_figure5e_original_populations_and_groups():
    values = pd.read_csv(ARISTA / 'figure5e_growth_interaction_by_cell.csv')
    assert values.groupby('time').size().tolist() == [7668, 7780, 8106, 8608, 9436, 9684, 9673, 10035, 11316]
    assert len(values) == 82306
    assert values.groupby(['time', 'celltype']).ngroups == 177


def test_figure4_lr_scores_recalculate_archived_values():
    pytest.importorskip('matplotlib')
    pytest.importorskip('fitz')
    from reproduction.mosta.calculations import interaction_scores
    from reproduction.mosta.main_figure import PANELS
    edges = pd.read_csv(PANELS / 'fig4b/evidence/type_matrix.csv')
    expected = pd.read_csv(PANELS / 'fig4b/evidence/type_scores.csv')
    actual = interaction_scores(edges).set_index(['time', 'cell_type'])
    expected = expected.set_index(['time', 'cell_type'])
    for key in ('incoming', 'outgoing', 'total'):
        np.testing.assert_allclose(actual.loc[expected.index, key], expected[key], atol=1e-12)


def test_model_download_in_figure4_uses_an_existing_archive():
    notebook = json.loads((ROOT / 'docs/tutorials/paper_figures/main_figure_4.ipynb').read_text())
    source = '\n'.join(''.join(cell['source']) for cell in notebook['cells'])
    assert 'kind="mosta_model.zip"' in source
    assert 'numeric_d = calculate_velocity_panel' in source
    assert 'numeric_e = calculate_velocity_panel' in source
    assert 'numeric_path=numeric_e' in source


def test_arista_guide_does_not_call_full_velocity_intrinsic():
    notebook = json.loads((ROOT / 'docs/tutorials/paper_figures/main_figure_5.ipynb').read_text())
    source = '\n'.join(''.join(cell['source']) for cell in notebook['cells'])
    assert 'Intrinsic-context gene velocity' in source
    assert 'Interpolate the full gene-velocity' not in source
    assert 'arista_model_fields.md' in source
    manifest = json.loads((ROOT / 'CytoBridge/results/data/downloads/manifest.json').read_text())
    record = next(a for a in manifest['archives'] if a['archive'] == 'arista_growth_model_states.zip')
    assert record['parts'][0]['asset_id'] > 0
