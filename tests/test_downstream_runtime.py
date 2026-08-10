import torch

from CytoBridge.tl.downstream.runtime import build_dynamical_runtime


class _DummyModel:
    def __init__(self):
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
