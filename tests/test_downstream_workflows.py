import pytest
import pandas as pd
from types import SimpleNamespace

from CytoBridge.tl.downstream.workflows import run_interpolation_workflow


class _UntouchableRuntime:
    @property
    def f_net(self):
        raise AssertionError("runtime must not be accessed before argument validation")


def test_supplied_classifier_is_read_only_even_with_training_data(tmp_path, monkeypatch):
    from CytoBridge.tl.downstream import workflows

    cache = tmp_path / "classifier.pt"
    cache.write_bytes(b"selected validation weights")
    loaded = SimpleNamespace(
        model=None, label_encoder=None, feature_dim=2,
        feature_cols=("samples", "x1", "x2"), accuracy=.8,
        balanced_accuracy=.7, metadata={"version": 6}, evaluation={},
    )
    calls = []
    monkeypatch.setattr(workflows, "load_cached_mlp_classifier",
                        lambda path, **kw: calls.append(path) or loaded)
    def unexpected_training(*args, **kwargs):
        raise AssertionError("an explicitly selected classifier must not be retrained")
    monkeypatch.setattr(workflows, "train_cached_mlp_classifier_from_adata", unexpected_training)
    class SimulationReached(Exception):
        pass
    def stop_after_classifier(*args, **kwargs):
        raise SimulationReached
    monkeypatch.setattr(workflows, "simulate_sde_points", stop_after_classifier)
    frame = pd.DataFrame({"samples": [0., 1.], "x1": [0., 1.],
                          "x2": [1., 0.], "Annotation": ["A", "B"]})
    with pytest.raises(SimulationReached):
        run_interpolation_workflow(
            df=frame, dim=2, annotation_key="Annotation",
            runtime=SimpleNamespace(f_net=None, score_net=None), device="cpu",
            output_dir=str(tmp_path), interp_time_points=[.5],
            classifier_cache_path=str(cache), classifier_adata=object())
    assert calls == [str(cache)]
    assert cache.read_bytes() == b"selected validation weights"


@pytest.mark.parametrize('separate, expected', [(True, 10043), (False, None)])
def test_interaction_random_stream_can_reproduce_earlier_analyses(tmp_path, separate, expected):
    frame = pd.DataFrame({'samples': [0., 1.], 'x1': [0., 1.], 'x2': [1., 0.],
                          'Annotation': ['A', 'B']})
    result = run_interpolation_workflow(
        df=frame, dim=2, annotation_key='Annotation',
        runtime=SimpleNamespace(f_net=None, score_net=None),
        device='cpu', output_dir=str(tmp_path), no_interp=True,
        random_seed=42, separate_interaction_random_stream=separate)
    assert result.simulation_seeds['split_population'] == 43
    assert result.simulation_seeds['split_interaction_grouping'] == expected
    assert result.simulation_seeds['separate_interaction_random_stream'] is separate


@pytest.mark.parametrize(
    "invalid_mode",
    ["per-timepoint", "T0_fixed", "", None],
)
def test_piecewise_observed_sample_mode_fails_closed_before_side_effects(
    invalid_mode,
):
    with pytest.raises(
        ValueError,
        match="piecewise_observed_sample_mode must be exactly one of",
    ):
        run_interpolation_workflow(
            df=object(),
            dim=3,
            annotation_key="cell_type",
            runtime=_UntouchableRuntime(),
            device="cpu",
            output_dir="unused",
            split_sde_piecewise=True,
            piecewise_observed_sample_mode=invalid_mode,
        )
