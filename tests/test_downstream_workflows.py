import pytest
import pandas as pd
from types import SimpleNamespace

from CytoBridge.tl.downstream.workflows import run_interpolation_workflow


class _UntouchableRuntime:
    @property
    def f_net(self):
        raise AssertionError("runtime must not be accessed before argument validation")


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
