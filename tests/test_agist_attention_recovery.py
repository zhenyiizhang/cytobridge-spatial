from pathlib import Path

import nbformat
import numpy as np
import pytest
from scipy.stats import spearmanr

from reproduction.agist.main_figure import attention_strength_correlation


def edges(matrix):
    rows, cols = np.where(matrix > 0)
    return dict(source=rows, target=cols, attention=matrix[rows, cols])


def test_strength_spearman_uses_paired_nonzero_rows():
    truth = np.array([[0., 2, 0, 0], [0, 0, 3, 0], [4, 0, 0, 0], [0, 0, 0, 0]])
    prediction = truth.copy()
    prediction[0, 1], prediction[1, 2] = 5, 1
    metrics, pred, ref, mask = attention_strength_correlation(edges(prediction), truth)
    np.testing.assert_array_equal(pred, prediction.mean(axis=1))
    np.testing.assert_array_equal(ref, truth.mean(axis=1))
    assert mask.tolist() == [True, True, True, False]
    assert metrics['n_compared'] == 3
    assert metrics['spearman'] == spearmanr(pred[:3], ref[:3]).statistic


def test_equal_nonzero_counts_do_not_allow_mismatched_cell_ids():
    truth = np.array([[0., 1, 0], [0, 0, 2], [0, 0, 0]])
    pred = np.array([[0., 1, 0], [0, 0, 0], [2, 0, 0]])
    with pytest.raises(ValueError, match='different nonzero cells'):
        attention_strength_correlation(edges(pred), truth)


def test_undefined_correlation_is_rejected():
    truth = np.array([[0., 1], [1, 0]])
    with pytest.raises(ValueError, match='nonconstant'):
        attention_strength_correlation(edges(truth), truth)


def test_figure2_notebook_uses_one_final_model_for_every_panel():
    root = Path(__file__).resolve().parents[1]
    book = nbformat.read(root / 'docs/tutorials/paper_figures/main_figure_2.ipynb', as_version=4)
    source = '\n'.join(c.source for c in book.cells if c.cell_type == 'code')
    assert 'kind="agist_final.zip"' in source
    assert 'model_1214' not in source
    assert 'calculate("evaluate.py", "truth"' in source
    assert 'calculate("evaluate.py", "predicted"' in source
    assert 'calculate("supplementary.py", "attention"' in source
    assert 'calculate("layout/layout.py"' in source
    assert 'CYTOBRIDGE_FIGURE2_TEACHER' in source
    evaluate=(root/'reproduction/agist/experiment/evaluate.py').read_text()
    assert "RUN/'cytobridge/model'" in evaluate
    recovery = nbformat.read(root / 'docs/tutorials/paper_figures/agist_attention_recovery.ipynb', as_version=4)
    assert 'calculation/evaluation/attention_by_time.csv' in '\n'.join(c.source for c in recovery.cells)



def test_generator_keeps_both_notebooks_in_sync(monkeypatch):
    from scripts import build_simulation_nonspatial_tutorials as generator
    generated = {}
    monkeypatch.setattr(generator, 'write', lambda name, cells: generated.update({name: cells}))
    generator.main2()
    generator.attention_recovery()
    for name, cells in generated.items():
        book = nbformat.read(generator.DEST / name, as_version=4)
        assert [c.source for c in cells] == [c.source for c in book.cells]


def test_figure2_inputs_are_a_separate_download(tmp_path, monkeypatch):
    from CytoBridge import datasets
    calls = []
    monkeypatch.setattr(datasets, '_download_archive', lambda record, path: calls.append(record['archive']))
    datasets.download('agist', tmp_path, kind='agist_figure2_inputs.zip')
    assert calls == ['agist_figure2_inputs.zip']


def test_original_score_download_is_available(tmp_path, monkeypatch):
    from CytoBridge import datasets
    calls = []
    monkeypatch.setattr(datasets, '_download_archive', lambda record, path: calls.append(record['archive']))
    datasets.download('agist', tmp_path, kind='agist_figure2_score_model.zip')
    assert calls == ['agist_figure2_score_model.zip']


def test_figure2_setup_downloads_only_the_final_experiment(tmp_path, monkeypatch):
    from CytoBridge import datasets
    root = Path(__file__).resolve().parents[1]
    book = nbformat.read(root / 'docs/tutorials/paper_figures/main_figure_2.ipynb', as_version=4)
    setup = next(c.source for c in book.cells if c.cell_type == 'code')
    calls = []
    monkeypatch.setenv('CYTOBRIDGE_PROJECT_DIR', str(tmp_path))
    def fake_download(dataset, destination, kind):
        calls.append(kind)
        folder=Path(destination)/'data/agist_final'
        folder.mkdir(parents=True,exist_ok=True)
        (folder/'original_schematic_source.pdf').write_bytes(b'artwork')
    monkeypatch.setattr(datasets, 'download', fake_download)
    scope = {}
    exec(compile(setup, '<figure2-setup>', 'exec'), scope)
    assert calls == ['agist_final.zip']
    assert scope['source'] == tmp_path / 'data/agist_final'



def test_distance_panels_keep_physical_axis_labels(tmp_path):
    fitz = pytest.importorskip('fitz')
    from CytoBridge.results.main_figure_2 import load_main_figure_2
    from reproduction.agist.main_figure import draw_distance_panels
    pdf, _ = draw_distance_panels(load_main_figure_2(), tmp_path)
    with fitz.open(pdf) as document:
        text = document[0].get_text()
    assert '0.1' in text and '0.2' in text
