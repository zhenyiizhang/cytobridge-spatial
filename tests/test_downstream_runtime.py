import torch
import pytest

from CytoBridge.tl.downstream.runtime import build_dynamical_runtime


class _DummyModel:
    def __init__(self):
        self.components = ["velocity", "growth", "score", "interaction"]
        self.interaction_net = object()
        self.score_create_graph = None

    def predict_velocity(self, t, x):
        return torch.zeros_like(x)

    def predict_growth(self, t, x):
        return torch.zeros((x.shape[0], 1), dtype=x.dtype, device=x.device)

    def compute_score(self, t, x, create_graph=True):
        self.score_create_graph = create_graph
        return x.sum(dim=1, keepdim=True), torch.ones_like(x)


def test_score_runtime_does_not_retain_training_graph():
    model = _DummyModel()
    runtime = build_dynamical_runtime(model)

    gradient = runtime.score_net.compute_gradient(
        torch.zeros((2, 1)), torch.zeros((2, 3))
    )

    assert model.score_create_graph is False
    assert gradient.shape == (2, 3)


def test_runtime_accepts_a_clean_model_without_interaction():
    model = _DummyModel()
    model.components = ["velocity", "growth", "score"]
    del model.interaction_net

    runtime = build_dynamical_runtime(model)

    assert runtime.f_net.interaction_net is None


def test_runtime_rejects_a_stale_interaction_net_on_no_interaction_model():
    model = _DummyModel()
    model.components = ["velocity", "growth", "score"]

    with pytest.raises(TypeError, match="still exposes interaction_net"):
        build_dynamical_runtime(model)
