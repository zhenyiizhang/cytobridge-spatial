"""Use one selected dataset configuration throughout benchmark generation and reuse."""
import argparse
from pathlib import Path

import yaml

from scripts.spatiotemporal_benchmark import run_unified_benchmark as runner


def test_selected_config_directory_reaches_prepare_and_resume(tmp_path, monkeypatch):
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    cfg = runner.load_datasets(["chicken_heart"])["chicken_heart"]
    cfg["input_h5ad"] = str(tmp_path / "downloaded" / "aligned.h5ad")
    cfg["benchmark"]["formal_run_root"] = str(tmp_path / "downloaded" / "model_run")
    config = config_dir / "chicken_heart.yaml"
    config.write_text(yaml.safe_dump(cfg))
    commands = []
    monkeypatch.setattr(runner, "CONFIG_DIR", runner.CONFIG_DIR)
    monkeypatch.setattr(runner, "run_or_print", lambda items, _: commands.extend(items))
    runner.main(["--datasets", "chicken_heart", "--config-dir", str(config_dir),
                 "--run-root", str(tmp_path / "benchmark"), "prepare"])
    assert str(config) in commands[0]
    assert cfg["input_h5ad"] in commands[0]
    assert runner._current_dataset_config(cfg)[0] == config
    assert runner.formal_dataset_root("chicken_heart", cfg, argparse.Namespace(
        formal_root=Path("unused"))) == Path(cfg["benchmark"]["formal_run_root"])

