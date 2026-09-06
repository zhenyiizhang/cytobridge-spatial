"""Use the same classifier and feature order in the zebrafish analyses."""
from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
import pytest


def test_predictions_use_all_original_state_features(monkeypatch):
    from reproduction.zebrafish import classifier as module
    points = np.arange(156, dtype=np.float32).reshape(3, 52)
    calls = []
    monkeypatch.setattr(module.cb.tl, "predict_labels_for_points",
                        lambda **kw: calls.append(kw) or ["A", "B", "A"])
    cached = SimpleNamespace(model=object(), label_encoder=object())
    labels = module.assign_cell_types(points, 2, cached)
    np.testing.assert_array_equal(calls[0]["points"], points)
    assert calls[0]["feature_dim"] == 52
    assert calls[0]["knn_neighbors"] == 10
    assert calls[0]["include_time_feature"]
    assert labels.tolist() == ["A", "B", "A"]
    with pytest.raises(ValueError, match="50 expression PCs"):
        module.assign_cell_types(points[:, :12], 2, cached)


def test_checkpoint_feature_order_is_checked(tmp_path, monkeypatch):
    from reproduction.zebrafish import classifier as module
    cached = SimpleNamespace(feature_cols=("samples", *(f"x{i}" for i in range(1, 53))), include_time_feature=True)
    calls = []
    monkeypatch.setattr(module.cb.tl, "load_cached_mlp_classifier",
                        lambda path, **kw: calls.append(path) or cached)
    assert module.load_classifier(tmp_path) is cached
    assert calls == [str(tmp_path / module.CLASSIFIER_FILE)]
    cached.feature_cols = ("samples", *(f"x{i}" for i in range(1, 13)))
    with pytest.raises(ValueError, match="50 expression PCs"):
        module.load_classifier(tmp_path)


def test_tutorial_passes_one_supplied_checkpoint_to_both_stages():
    root = Path(__file__).resolve().parents[1]
    book = json.loads((root / "docs/tutorials/paper_figures/zebrafish_si_s31_s38.ipynb").read_text())
    cells = {cell["id"]: "".join(cell["source"]) for cell in book["cells"]}
    assert "25f65c49dc60ea4c" in cells["setup"]
    assert "0adc1c3a0170a81e" not in "".join(cells.values())
    for name in ("population-calculate", "reconstruction-calculate"):
        assert '"--classifier-cache", classifier_path' in cells[name]


def test_supplied_classifier_is_loaded_without_training(monkeypatch):
    from scripts import run_zebrafish_paper_downstream as module
    cached = SimpleNamespace(feature_cols=("samples", *(f"x{i}" for i in range(1, 53))), include_time_feature=True)
    ctx = SimpleNamespace(args=SimpleNamespace(profile="paper", classifier_epochs=500,
        classifier_cache=Path("classifier.pt"), random_seed=42, device="cpu"), shared_cache_dir=Path("cache"))
    monkeypatch.setattr(module.cb.tl, "load_cached_mlp_classifier", lambda *a, **k: cached)
    def no_training(*args, **kwargs):
        raise AssertionError("The supplied classifier must not be retrained")
    monkeypatch.setattr(module.cb.tl, "train_cached_mlp_classifier_from_adata", no_training)
    loaded, path = module._train_main_classifier(ctx)
    assert loaded is cached
    assert path == Path("classifier.pt").resolve()
    assert module._main_classifier_settings(ctx)["n_joint_features"] == 52
    ctx.args.classifier_cache = None
    assert module._main_classifier_settings(ctx)["n_joint_features"] == 52
    assert module._main_classifier_settings(ctx)["train_on_full_data"] is False
