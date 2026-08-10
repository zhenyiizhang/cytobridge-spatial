"""Runtime adapters for downstream dynamical-model inference."""

from __future__ import annotations

from dataclasses import dataclass

from .checkpoint import LoadedModel

__all__ = [
    "DynamicalRuntime",
    "build_dynamical_runtime",
]


class _VelocityGrowthAdapter:
    def __init__(self, model):
        self.model = model
        self.interaction_net = getattr(model, "interaction_net", None)

    def v_net(self, t, z):
        return self.model.predict_velocity(t=t, x=z)

    def g_net(self, t, z):
        return self.model.predict_growth(t=t, x=z)


class _ScoreGradientAdapter:
    def __init__(self, model):
        self.model = model

    def compute_gradient(self, t, z):
        # Downstream integration never backpropagates through the score field.
        # Keeping the higher-order graph here retains every Euler step and can
        # exhaust GPU memory during full split-SDE trajectories.
        _, grad = self.model.compute_score(t=t, x=z, create_graph=False)
        return grad


@dataclass(frozen=True)
class DynamicalRuntime:
    """Unified downstream runtime used by simulation and attention helpers."""

    model: object
    f_net: object
    score_net: object


def build_dynamical_runtime(model_or_loaded) -> DynamicalRuntime:
    """Build a unified runtime view over current or legacy dynamical models."""
    model = model_or_loaded.model if isinstance(model_or_loaded, LoadedModel) else model_or_loaded

    required = ("predict_velocity", "predict_growth", "compute_score")
    missing = [name for name in required if not hasattr(model, name)]
    if missing:
        raise TypeError(
            "Model is missing downstream inference methods: "
            f"{missing}. Expected a CytoBridge dynamical model or LoadedModel."
        )
    if getattr(model, "interaction_net", None) is None:
        raise TypeError("Model is missing interaction_net, which is required for downstream attention analysis.")

    return DynamicalRuntime(
        model=model,
        f_net=_VelocityGrowthAdapter(model),
        score_net=_ScoreGradientAdapter(model),
    )
