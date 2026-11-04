"""Deterministic particle grouping helpers shared by downstream analyses."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def runtime_style_random_groups(
    indices: Sequence[int], *, group_size: int, seed: int
) -> list[np.ndarray]:
    """Partition cells with the same group-size/remainder rule as GNN runtime.

    A final partial group is merged with the preceding full group, matching
    :func:`CytoBridge.tl.core.interaction.cal_interaction_gnn` and preventing
    isolated one-particle remainders.  The NumPy seed makes post-training
    diagnostics exactly reproducible without mutating PyTorch's global RNG.
    """

    indices = np.asarray(indices, dtype=np.int64)
    if len(indices) < 2:
        raise ValueError("At least two cells are required for interaction grouping.")
    if int(group_size) < 2:
        raise ValueError("group_size must be at least two.")
    permutation = np.random.default_rng(int(seed)).permutation(indices)
    n = len(permutation)
    if n % int(group_size) == 0:
        n_full = n // int(group_size)
    elif n < int(group_size):
        n_full = 0
    else:
        n_full = n // int(group_size) - 1
    groups = [
        permutation[start : start + int(group_size)]
        for start in range(0, n_full * int(group_size), int(group_size))
    ]
    remainder = permutation[n_full * int(group_size) :]
    if len(remainder):
        groups.append(remainder)
    if any(len(group) < 2 for group in groups):
        raise RuntimeError("Runtime-style grouping produced an isolated cell.")
    return groups


__all__ = ["runtime_style_random_groups"]
