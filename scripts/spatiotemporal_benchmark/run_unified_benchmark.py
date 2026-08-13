#!/usr/bin/env python3
"""Prepare, run, and evaluate the four-dataset benchmark with one small CLI."""

from __future__ import annotations

import argparse
import csv
import fcntl
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

import yaml

# fmt: off
REPO = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO / "configs" / "unified_benchmark"
DATASETS = ("zebrafish", "mosta", "arista", "admouse")
DYNAMIC = ("stvcr", "stories", "mioflow")
STATIC = ("moscot", "wot", "paste", "spateo", "linear_centroid_shift", "random_independent_pairs")
PRIMARY_METHODS = ("cytobridge", *DYNAMIC, *STATIC)
METHODS = (*PRIMARY_METHODS, "spatrack")
METHOD_NAME = {"cytobridge": "CytoBridge-0.015", **{name: name for name in METHODS[1:]}}
EXTERNAL_STATIC = {"moscot", "wot", "paste", "spateo"}
METHOD_REGISTRY = REPO / "scripts" / "spatiotemporal_benchmark" / "method_registry.json"


def load_datasets(names):
    return {name: yaml.safe_load((CONFIG_DIR / f"{name}.yaml").read_text(encoding="utf-8")) for name in names}


def assignments(values):
    result = {}
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or name not in METHODS:
            raise ValueError(f"expected METHOD=PATH, found {value!r}")
        result[name] = Path(path).expanduser().resolve()
    return result


def command(module, python, *args):
    return [str(python), "-m", module, *(str(value) for value in args)]


def dynamic_commands(method, python, source, manifest, root, split, targets):
    module = "scripts.spatiotemporal_benchmark.dynamic.run_dynamic"
    fit = root / "fits" / method / split
    shared = ("--method", method, "--input-manifest", manifest, "--split-id", split, "--source-root", source)
    commands = [command(module, python, "fit", *shared, "--output-dir", fit)]
    track = "loto" if split.startswith("loto") else "full_data"
    for target in targets:
        output = root / "predictions" / track / method / f"t{target}"
        commands.append(command(module, python, "infer", *shared, "--fit-dir", fit,
                                "--target-time", target, "--output-dir", output))
    return commands


def static_commands(method, python, source, manifest, root, split, targets):
    module = "scripts.spatiotemporal_benchmark.static_baselines.run"
    track = "loto" if split.startswith("loto") else "full_data"
    output = root / "predictions" / track / method
    if track == "loto":
        output /= f"t{targets[0]}"
    args = ["run", "--method", method, "--evaluation-mode", "loto" if track == "loto" else "no-holdout",
            "--input-h5ad", root / "inputs" / split / "train.h5ad", "--input-manifest", manifest,
            "--output-dir", output, "--max-fit-n", 800]
    if track == "loto":
        args += ["--target-time", targets[0]]
    if source:
        args += ["--source-root", source]
    return [command(module, python, *args)]


def cytobridge_commands(python, cfg, formal, manifest, root, split, targets, device):
    module = "scripts.spatiotemporal_benchmark.cytobridge.run_cytobridge"
    model = formal / "training"
    training_config = formal / cfg["benchmark"]["training_config"]
    shared = ("--repo", REPO, "--input-manifest", manifest, "--split", split)
    if split == "full_data":
        output = root / "predictions" / "full_data" / "cytobridge"
        return [command(module, python, "validate-model", *shared, "--model-dir", model,
                        "--training-config", training_config),
                command(module, python, "infer-full", *shared, "--model-dir", model,
                        "--training-config", training_config, "--output-dir", output, "--device", device)]
    target, graph = targets[0], root / "graphs" / split
    loto_model = root / "fits" / "cytobridge" / split
    output = root / "predictions" / "loto" / "cytobridge" / f"t{target}"
    prepare_args = ["--training-config", training_config]
    if cfg["benchmark"].get("edge_prior_mode", "learned") == "learned":
        prepare_args += ["--database", REPO / cfg["benchmark"]["graph_database"]]
    return [command(module, python, "prepare-loto", *shared, *prepare_args,
                    "--expression-layer", cfg["benchmark"]["expression_layer"],
                    "--output-dir", graph, "--device", device),
            command(module, python, "fit-loto", *shared, "--training-config", training_config,
                    "--graph-dir", graph, "--output-dir", loto_model, "--device", device),
            command(module, python, "infer-loto", *shared, "--model-dir", loto_model,
                    "--training-config", training_config, "--output-dir", output, "--device", device)]


def jobs_for_dataset(name, cfg, args, pythons, sources):
    root, formal = args.run_root / name, args.formal_root / name
    manifest, jobs = root / "inputs" / "manifest.json", []
    for method in args.methods:
        for track in args.tracks:
            targets = cfg["loto_targets"] if track == "loto" else cfg["full_data_targets"]
            splits = [(f"loto_t{target}", [target]) for target in targets] if track == "loto" else [("full_data", targets)]
            for split, selected in splits:
                if method == "spatrack":
                    jobs.append((method, track, selected, [], [], "not_applicable")); continue
                python, source = pythons.get(method, Path(sys.executable)), sources.get(method, args.software_root / method)
                required = [python, manifest]
                if method == "cytobridge":
                    commands = cytobridge_commands(python, cfg, formal, manifest, root, split, selected, args.device)
                    required += [formal / "training", formal / cfg["benchmark"]["training_config"]]
                elif method in DYNAMIC:
                    commands = dynamic_commands(method, python, source, manifest, root, split, selected); required.append(source)
                else:
                    official_source = source if method in EXTERNAL_STATIC else None
                    commands = static_commands(method, python, official_source, manifest, root, split, selected)
                    if official_source: required.append(official_source)
                jobs.append((method, track, selected, commands, required, None))
    return jobs


def execute(commands, required, timeout, log_path):
    missing = [str(path) for path in required if not Path(path).exists()]
    if missing:
        return "not_available", "missing: " + ", ".join(missing)
    deadline, output = time.monotonic() + timeout, []
    try:
        for item in commands:
            run = subprocess.run(item, cwd=REPO, text=True, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, timeout=max(1, deadline - time.monotonic()))
            output.append(run.stdout)
            if run.returncode:
                status = "oom" if "out of memory" in "\n".join(output).lower() else "failed"
                break
        else:
            status = "completed"
    except subprocess.TimeoutExpired as error:
        fragment = error.stdout or ""
        output.append(fragment.decode() if isinstance(fragment, bytes) else fragment)
        status = "timeout"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("\n".join(output), encoding="utf-8")
    reason = "" if status == "completed" else f"{status}; see {log_path}"
    return status, reason


def target_output_is_complete(root, method, track, target):
    """Return whether one target has both prediction data and its readable summary."""

    target_dir = root / "predictions" / track / method / f"t{target}"
    return (target_dir / "prediction.npz").is_file() and (
        target_dir / "summary.json"
    ).is_file()


def merge_status_rows(path, rows):
    """Update method/track/target rows without erasing separately run methods."""

    columns = ("track", "target", "method", "status", "reason", "elapsed_seconds")
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        merged = {}
        if path.is_file():
            with path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    merged[(row["track"], int(row["target"]), row["method"])] = row
        for row in rows:
            merged[(row["track"], int(row["target"]), row["method"])] = row
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            with temporary.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=columns)
                writer.writeheader()
                writer.writerows(
                    sorted(
                        merged.values(),
                        key=lambda row: (
                            row["track"],
                            int(row["target"]),
                            row["method"],
                        ),
                    )
                )
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()


def run_or_print(commands, dry_run):
    for item in commands:
        print(shlex.join(item))
        if not dry_run:
            subprocess.run(item, cwd=REPO, check=True)


def prepare(name, _cfg, args):
    root = args.run_root / name
    build = command("scripts.spatiotemporal_benchmark.build_inputs", sys.executable,
                    "--config", CONFIG_DIR / f"{name}.yaml", "--h5ad",
                    args.formal_root / name / "preprocess" / f"{name}_aligned.h5ad", "--output-dir", root)
    if args.overwrite: build.append("--overwrite")
    verify = command("scripts.spatiotemporal_benchmark.verify_inputs", sys.executable, "--output-dir", root)
    run_or_print([build, verify], args.dry_run)


def run_dataset(name, cfg, args, pythons, sources):
    root = args.run_root / name
    rows = []
    for method, track, targets, commands, required, fixed in jobs_for_dataset(name, cfg, args, pythons, sources):
        run_or_print(commands, True)
        if args.dry_run: continue
        started = time.monotonic()
        log = args.run_root / name / "logs" / f"{track}_{method}_{targets[0]}.log"
        status, reason = ((fixed, "matched signed-PC benchmark is not applicable") if fixed
                          else execute(commands, required, args.timeout, log))
        elapsed = round(time.monotonic() - started, 3)
        for target in targets:
            target_complete = target_output_is_complete(root, method, track, target)
            target_status = "completed" if target_complete else status
            target_reason = "" if target_complete else reason
            if target_status == "completed" and not target_complete:
                target_status = "failed"
                target_reason = "job exited without prediction.npz and summary.json"
            rows.append({"track": track, "target": target, "method": METHOD_NAME[method], "status": target_status,
                         "reason": target_reason, "elapsed_seconds": elapsed})
    if not args.dry_run:
        path = args.run_root / name / "status" / "method_target_status.csv"
        merge_status_rows(path, rows)


def evaluate(name, cfg, args):
    root = args.run_root / name
    for track in args.tracks:
        targets = cfg["loto_targets"] if track == "loto" else cfg["full_data_targets"]
        output = root / "evaluation" / track
        score = command("scripts.spatiotemporal_benchmark.evaluate_predictions", sys.executable,
                        "--input-manifest", root / "inputs" / "manifest.json", "--predictions-root", root / "predictions",
                        "--status-table", root / "status" / "method_target_status.csv", "--track", track,
                        "--targets", *targets, "--methods", *(METHOD_NAME[method] for method in PRIMARY_METHODS),
                        "--output-dir", output)
        report = command("scripts.spatiotemporal_benchmark.summarize_results", sys.executable,
                         "--metrics-long", output / f"{track}_metrics_long.csv",
                         "--evaluation-manifest", output / f"{track}_evaluation_manifest.json",
                         "--method-registry", METHOD_REGISTRY,
                         "--output-dir", root / "reports" / track)
        run_or_print([score, report], args.dry_run)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--formal-root", type=Path, default=Path("/data/cytobridge/projects/CytoBridge-ST-1104/runs/corrected-de-novo-20260813-r2"))
    parser.add_argument("--run-root", type=Path, default=Path("/data/cytobridge/projects/CytoBridge-ST-1104/runs/corrected-benchmark-20260813-r1"))
    parser.add_argument("--software-root", type=Path, default=REPO / "software")
    parser.add_argument("--dry-run", action="store_true")
    sub = parser.add_subparsers(dest="action", required=True)
    prep = sub.add_parser("prepare"); prep.add_argument("--overwrite", action="store_true")
    run = sub.add_parser("run"); run.add_argument("--methods", nargs="+", choices=METHODS, default=list(PRIMARY_METHODS))
    run.add_argument("--tracks", nargs="+", choices=("loto", "full_data"), default=["loto", "full_data"])
    run.add_argument("--timeout", type=int, default=3600); run.add_argument("--device", default="cuda")
    run.add_argument("--python", action="append", default=[]); run.add_argument("--source", action="append", default=[])
    report = sub.add_parser("evaluate")
    report.add_argument("--tracks", nargs="+", choices=("loto", "full_data"), default=["loto", "full_data"])
    args = parser.parse_args(argv); configs = load_datasets(args.datasets)
    if args.action == "prepare":
        for name, cfg in configs.items(): prepare(name, cfg, args)
    elif args.action == "run":
        pythons, sources = assignments(args.python), assignments(args.source)
        for name, cfg in configs.items(): run_dataset(name, cfg, args, pythons, sources)
    else:
        for name, cfg in configs.items(): evaluate(name, cfg, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
# fmt: on
