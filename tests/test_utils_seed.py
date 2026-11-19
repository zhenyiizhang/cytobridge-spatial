from __future__ import annotations

import os
import random

import numpy as np
import pytest
import torch

from CytoBridge.utils.utils import set_seed


def _draw_rng_values() -> tuple[float, float, torch.Tensor]:
    return random.random(), float(np.random.random()), torch.rand(4)


def test_set_seed_replays_python_numpy_and_torch_cpu_rngs() -> None:
    set_seed(123)
    first = _draw_rng_values()

    set_seed(123)
    second = _draw_rng_values()

    assert first[:2] == pytest.approx(second[:2])
    torch.testing.assert_close(first[2], second[2], rtol=0, atol=0)
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False


def test_set_seed_preserves_cuda_rng_seeding_when_cuda_is_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, int]] = []
    monkeypatch.setattr(random, "seed", lambda seed: calls.append(("python", seed)))
    monkeypatch.setattr(np.random, "seed", lambda seed: calls.append(("numpy", seed)))
    monkeypatch.setattr(
        torch, "manual_seed", lambda seed: calls.append(("torch", seed))
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "manual_seed",
        lambda seed: calls.append(("cuda", seed)),
    )
    monkeypatch.setattr(
        torch.cuda,
        "manual_seed_all",
        lambda seed: calls.append(("cuda_all", seed)),
    )

    set_seed(789)

    assert calls == [
        ("python", 789),
        ("numpy", 789),
        ("torch", 789),
        ("cuda", 789),
        ("cuda_all", 789),
    ]


@pytest.mark.parametrize("preexisting", (None, "launch-time-seed"))
def test_set_seed_does_not_claim_to_reseed_python_hashes(
    monkeypatch: pytest.MonkeyPatch,
    preexisting: str | None,
) -> None:
    if preexisting is None:
        monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    else:
        monkeypatch.setenv("PYTHONHASHSEED", preexisting)

    set_seed(456)

    assert os.environ.get("PYTHONHASHSEED") == preexisting
