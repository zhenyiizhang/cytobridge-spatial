"""Prepare, train and compare the seven chicken-heart alignment conditions."""
from __future__ import annotations
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
VARIANTS = ("baseline_repeat", "translate_low", "translate_moderate",
            "rotate_low", "rotate_moderate", "translate_rotate_low", "translate_rotate_moderate")


def source(name):
    spec = importlib.util.spec_from_file_location("heart_alignment_" + name, HERE / "server_code" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def perturbations(base):
    directions = base._translation("moderate")
    signs = {"D4": 1, "D7": -1, "D10": 1, "D14": -1}
    result = {}
    for level, distance, angle in (("low", 1, 1), ("moderate", 2, 3)):
        for kind in ("translate", "rotate", "translate_rotate"):
            stages = {}
            for stage in base.TIME_ORDER:
                direction = directions[stage]
                norm = math.hypot(direction.translate_x_nn, direction.translate_y_nn)
                stages[stage] = base.RigidPerturbation(
                    signs[stage] * angle if kind != "translate" else 0,
                    distance * direction.translate_x_nn / norm if kind != "rotate" else 0,
                    distance * direction.translate_y_nn / norm if kind != "rotate" else 0)
            result[f"{kind}_{level}"] = stages
    return result


def prepare(input_h5ad, output_dir):
    input_h5ad, output = Path(input_h5ad).resolve(), Path(output_dir).resolve()
    if not input_h5ad.is_file():
        raise FileNotFoundError(input_h5ad)
    output.mkdir(parents=True, exist_ok=False)
    base = source("prepare_and_run")
    base.SOURCE_INPUT = input_h5ad
    base.AUDIT_ROOT = output
    base.INPUT_DIR = output / "inputs"
    base.VARIANT_SPECS = perturbations(base)
    base.prepare_inputs(compression="gzip")
    (output / "experiment.json").write_text(json.dumps({
        "input": str(input_h5ad), "translations_median_nn_units": [1, 2],
        "rotation_degrees": [1, 3], "variants": VARIANTS,
        "seed": 42, "alignment_seed": 42}, indent=2) + "\n")
    return output


def train(output_dir, variants=VARIANTS, device="cuda:0"):
    output = Path(output_dir).resolve()
    record = json.loads((output / "experiment.json").read_text())
    config = REPO / "CytoBridge/workflow_configs/chicken_heart.json"
    source("prepare_and_run")._validate_template(json.loads(config.read_text()))
    env = {**os.environ, "PYTHONHASHSEED": "0"}
    for variant in variants:
        if variant not in VARIANTS:
            raise ValueError(f"Unknown condition: {variant}")
        input_h5ad = Path(record["input"]) if variant == "baseline_repeat" else output / "inputs" / (variant + ".h5ad")
        run = output / "runs" / variant
        if run.exists():
            raise FileExistsError(run)
        command = [sys.executable, "-m", "CytoBridge.cli", "workflow", "--config", str(config),
                   "--train", "--input-h5ad", str(input_h5ad), "--output-dir", str(run), "--device", device]
        subprocess.run(command, cwd=REPO, env=env, check=True)


def compare(output_dir, reference_run):
    output, reference = Path(output_dir).resolve(), Path(reference_run).resolve()
    record = json.loads((output / "experiment.json").read_text())
    comparison = source("compare_runs")
    comparison.ACCEPTED_RUN = reference
    comparison.RUNS_DIR = output / "runs"
    comparison.SUMMARY_DIR = output / "summary"
    comparison.VARIANTS = VARIANTS
    comparison.compare()
    export = source("export_plot_inputs")
    export.SOURCE_INPUT = Path(record["input"])
    export.ACCEPTED_RUN = reference
    export.AUDIT_ROOT = output
    export.RUNS_DIR = output / "runs"
    export.OUTPUT = output / "summary/plot_inputs.npz"
    export.MANIFEST = output / "summary/plot_inputs_manifest.json"
    export.DISPLAY_VARIANTS = VARIANTS
    export.main()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=["prepare", "train", "compare"])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--input-h5ad", type=Path)
    parser.add_argument("--reference-run", type=Path)
    parser.add_argument("--variant", choices=VARIANTS, action="append")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.step == "prepare":
        if args.input_h5ad is None:
            parser.error("prepare needs --input-h5ad")
        prepare(args.input_h5ad, args.output_dir)
    elif args.step == "train":
        train(args.output_dir, args.variant or VARIANTS, args.device)
    else:
        if args.reference_run is None:
            parser.error("compare needs --reference-run")
        compare(args.output_dir, args.reference_run)


if __name__ == "__main__":
    main()
