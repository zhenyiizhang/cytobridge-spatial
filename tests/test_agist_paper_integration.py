import ast
from pathlib import Path

import torch

from reproduction.agist.integration import cal_interaction, euler_sdeint_split


def test_agist_grouping_restores_original_cell_order():
    state = torch.arange(15, dtype=torch.float32).reshape(5, 3)
    weights = torch.full((5, 1), -torch.log(torch.tensor(5.)))
    groups = []

    def interaction(values, log_weights, time):
        groups.append(len(values))
        return values * 2

    torch.manual_seed(1)
    result = cal_interaction(state, weights, interaction, torch.tensor([0.]), m=4)
    torch.testing.assert_close(result, 2 * state)
    assert groups == [5]


def test_agist_archived_time_convention_is_preserved():
    class LinearDynamics:
        def f(self, time, state):
            return torch.ones_like(state[0]), torch.zeros_like(state[1])

        def g(self, time, state):
            return torch.zeros_like(state)

    state = torch.zeros((4, 2))
    weights = torch.full((4, 1), -torch.log(torch.tensor(4.)))
    positions, log_weights = euler_sdeint_split(
        LinearDynamics(), (state, weights), dt=0.1,
        ts=torch.tensor([0., 0.2, 0.4]), noise_std=0.,
    )
    # The paper helper records the first Euler step in its first output slot.
    for value, expected in zip(positions, (0.1, 0.2, 0.4)):
        torch.testing.assert_close(value, torch.full_like(value, expected))
    assert all(value.shape == (4, 1) for value in log_weights)


def test_agist_script_imports_only_distributed_modules():
    source = Path(__file__).resolve().parents[1] / "scripts/run_agist_split_sde_replicates.py"
    modules = [node.module for node in ast.walk(ast.parse(source.read_text())) if isinstance(node, ast.ImportFrom)]
    assert not any(module and module.startswith("DeepRUOT") for module in modules)
    assert "reproduction.agist.integration" in modules
