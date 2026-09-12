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


def test_figure2_notebook_separates_original_attention_and_revised_growth():
    root = Path(__file__).resolve().parents[1]
    book = nbformat.read(root / 'docs/tutorials/paper_figures/main_figure_2.ipynb', as_version=4)
    source = '\n'.join(c.source for c in book.cells if c.cell_type == 'code')
    assert 'kind="agist_figure2_inputs.zip"' in source
    assert 'attention_model = figure2_root / "model_1214"' in source
    assert 'output / "original_attention"' in source
    assert 'output / "released_growth"' in source
    assert 'Not drawn:' not in source
    assert 'recorded_predictions' not in source
    recovery = nbformat.read(root / 'docs/tutorials/paper_figures/agist_attention_recovery.ipynb', as_version=4)
    code = '\n'.join(c.source for c in recovery.cells if c.cell_type == 'code')
    assert 'evaluate_attention_recovery(' in code
    assert 'recorded_predictions' not in code
    compile(code, '<notebook>', 'exec')


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
