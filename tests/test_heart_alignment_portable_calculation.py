import importlib.util
from pathlib import Path
import math
import json
import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "reproduction/chicken_heart/alignment_sensitivity_20260906/calculate.py"


def module():
    spec = importlib.util.spec_from_file_location("heart_portable", SCRIPT)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_perturbation_magnitudes_are_the_current_integer_design():
    code = module()
    specs = code.perturbations(code.source("prepare_and_run"))
    assert len(specs) == 6
    for name, stages in specs.items():
        distance, angle = (1, 1) if name.endswith("low") else (2, 3)
        for spec in stages.values():
            expected_distance = 0 if name.startswith("rotate_") else distance
            expected_angle = 0 if name.startswith("translate_") and not name.startswith("translate_rotate_") else angle
            np.testing.assert_allclose(math.hypot(spec.translate_x_nn, spec.translate_y_nn), expected_distance)
            assert abs(spec.rotation_deg) == expected_angle


def test_train_uses_selected_inputs_and_output_root(tmp_path, monkeypatch):
    code = module()
    (tmp_path / "experiment.json").write_text(json.dumps({"input": str(tmp_path / "source.h5ad")}))
    commands = []
    monkeypatch.setattr(code.subprocess, "run", lambda command, **kwargs: commands.append((command, kwargs)))
    code.train(tmp_path, ["baseline_repeat", "rotate_low"], "cuda:3")
    assert len(commands) == 2
    for (command, kwargs), name in zip(commands, ["baseline_repeat", "rotate_low"]):
        assert command[command.index("--output-dir") + 1] == str(tmp_path / "runs" / name)
        assert command[-1] == "cuda:3"
        assert kwargs["env"]["PYTHONHASHSEED"] == "0"
    assert commands[0][0][commands[0][0].index("--input-h5ad") + 1] == str(tmp_path / "source.h5ad")
    assert commands[1][0][commands[1][0].index("--input-h5ad") + 1] == str(tmp_path / "inputs/rotate_low.h5ad")
