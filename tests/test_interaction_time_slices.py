from __future__ import annotations

import numpy as np
import torch

from CytoBridge.tl.core.interaction import cal_interaction
from CytoBridge.tl.train.fit import _compute_interaction_by_time


class _SliceMeanInteraction(torch.nn.Module):
    requires_time = True

    def forward(self, z, lnw, t):
        return z.mean(dim=0, keepdim=True).expand_as(z) + t.reshape(-1)[0]


def test_interaction_export_is_computed_within_each_time_slice():
    data = torch.tensor([[0.0], [2.0], [10.0], [14.0]])
    times = torch.tensor([[0.0], [0.0], [1.0], [1.0]])
    interaction_net = _SliceMeanInteraction()

    actual = _compute_interaction_by_time(
        data,
        times,
        interaction_net,
        group_size=16,
        cutoff=1000.0,
        use_mass=True,
    )

    manual = torch.zeros_like(data)
    for time_value in (0.0, 1.0):
        mask = times[:, 0] == time_value
        slice_data = data[mask]
        slice_lnw = torch.full(
            (slice_data.shape[0], 1), -float(np.log(slice_data.shape[0]))
        )
        manual[mask] = cal_interaction(
            slice_data,
            slice_lnw,
            interaction_net,
            m=16,
            t=torch.tensor([time_value]),
        )

    mixed_lnw = torch.full((data.shape[0], 1), -float(np.log(data.shape[0])))
    old_mixed_result = cal_interaction(
        data,
        mixed_lnw,
        interaction_net,
        m=16,
        t=torch.tensor([0.0]),
    )

    torch.testing.assert_close(actual, manual)
    torch.testing.assert_close(actual[:, 0], torch.tensor([1.0, 1.0, 13.0, 13.0]))
    assert not torch.allclose(actual, old_mixed_result)

