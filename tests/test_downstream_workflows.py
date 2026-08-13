import pytest

from CytoBridge.tl.downstream.workflows import run_interpolation_workflow


class _UntouchableRuntime:
    @property
    def f_net(self):
        raise AssertionError("runtime must not be accessed before argument validation")


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
