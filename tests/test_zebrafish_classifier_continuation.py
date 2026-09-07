"""The classifier selected in the notebook also labels perturbation results."""
import inspect
from pathlib import Path
from types import SimpleNamespace

from reproduction.zebrafish import classifier


def test_explicit_classifier_is_loaded(monkeypatch, tmp_path):
    selected = tmp_path / "my_classifier.pt"
    called = []
    expected = SimpleNamespace(feature_cols=("samples", *(f"x{i}" for i in range(1, 53))),
                               include_time_feature=True)
    def load(path, device):
        called.append((path, device))
        return expected
    monkeypatch.setattr(classifier.cb.tl, "load_cached_mlp_classifier", load)
    assert classifier.load_classifier(tmp_path, "cpu", selected) is expected
    assert called == [(str(selected), "cpu")]


def test_notebook_passes_classifier_to_both_perturbation_analyses():
    import nbformat
    root = Path(__file__).resolve().parents[1]
    notebook = nbformat.read(root / "docs/tutorials/paper_figures/zebrafish_si_s31_s38.ipynb", 4)
    for function in ("plot_removal(", "simulate_daughter_perturbations("):
        cells = [cell.source for cell in notebook.cells if cell.cell_type == "code" and function in cell.source]
        assert len(cells) == 1
        assert "classifier_path=classifier_path" in cells[0]
    for filename in ("daughter_noise.py", "plot_virtual_removal.py"):
        source = (root / "reproduction/zebrafish" / filename).read_text()
        assert "load_classifier(data, device, classifier_path=classifier_path)" in source
