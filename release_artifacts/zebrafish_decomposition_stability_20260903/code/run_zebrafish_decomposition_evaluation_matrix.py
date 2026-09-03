#!/usr/bin/env python3
"""Evaluate the complete Zebrafish stability matrix on the available GPUs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--training-status", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--evaluator", type=Path, required=True)
    parser.add_argument("--aligned-h5ad", type=Path, required=True)
    parser.add_argument("--edge-predictor", type=Path, required=True)
    parser.add_argument("--gpu-count", type=int, default=8)
    return parser.parse_args()


def condition_matrix(run_root: Path) -> list[dict]:
    plan_path = run_root / "experiment_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    return [
        {
            "condition": job["condition"],
            "training_dir": Path(job["training_dir"]),
            "analysis_group": job["analysis_group"],
        }
        for job in plan["jobs"]
    ]


def main() -> int:
    args = arguments()
    run_root = args.run_root.expanduser().resolve(strict=True)
    training_status = args.training_status.expanduser().resolve(strict=True)
    training_state = json.loads(training_status.read_text(encoding="utf-8"))
    if training_state.get("state") != "complete":
        raise RuntimeError(
            "Evaluation starts after all fits finish successfully. "
            f"Training state is {training_state.get('state')!r}."
        )
    failed_training = {
        record["condition"]
        for record in training_state.get("completed", [])
        if int(record.get("return_code", 1)) != 0
    }
    if failed_training:
        raise RuntimeError(f"Failed training conditions: {sorted(failed_training)}")
    jobs = condition_matrix(run_root)
    for job in jobs:
        if not (job["training_dir"] / "training_run_summary.json").is_file():
            raise FileNotFoundError(job["training_dir"] / "training_run_summary.json")
        if not (job["training_dir"] / "config.yaml").is_file():
            raise FileNotFoundError(job["training_dir"] / "config.yaml")

    evaluation_root = run_root / "evaluation_edge_centered"
    logs_dir = run_root / "evaluation_logs"
    status_dir = run_root / "status"
    evaluation_root.mkdir(exist_ok=True)
    logs_dir.mkdir(exist_ok=True)
    state_path = status_dir / "evaluation_matrix_status.json"
    if state_path.exists():
        raise FileExistsError(state_path)
    pending = []
    running: dict[int, dict] = {}
    completed = []
    for job in jobs:
        existing_manifest = evaluation_root / job["condition"] / "evaluation_manifest.json"
        if existing_manifest.is_file():
            completed.append(
                {
                    "condition": job["condition"],
                    "analysis_group": job["analysis_group"],
                    "return_code": 0,
                    "reused_completed_evaluation": True,
                    "manifest": str(existing_manifest),
                }
            )
        else:
            pending.append(job)
    gpu_ids = list(range(int(args.gpu_count)))

    def write_state(state: str) -> None:
        payload = {
            "schema_version": 1,
            "state": state,
            "updated_at_utc": utc_now(),
            "pending": [job["condition"] for job in pending],
            "running": [
                {
                    "condition": record["job"]["condition"],
                    "gpu": gpu,
                    "pid": record["process"].pid,
                }
                for gpu, record in running.items()
            ],
            "completed": completed,
        }
        state_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    write_state("starting")
    while pending or running:
        for gpu in gpu_ids:
            if gpu in running or not pending:
                continue
            ready_index = next(
                (
                    index
                    for index, candidate in enumerate(pending)
                    if (candidate["training_dir"] / "training_run_summary.json").is_file()
                ),
                None,
            )
            if ready_index is None:
                continue
            job = pending.pop(ready_index)
            condition = job["condition"]
            output_dir = evaluation_root / condition
            if output_dir.exists() and any(output_dir.iterdir()):
                raise FileExistsError(output_dir)
            log_path = logs_dir / f"{condition}.log"
            log_handle = log_path.open("xb")
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
            environment["OMP_NUM_THREADS"] = "8"
            environment["MKL_NUM_THREADS"] = "8"
            environment["PYTHONPATH"] = str(args.release)
            command = [
                str(args.python),
                str(args.evaluator),
                "--condition",
                condition,
                "--aligned-h5ad",
                str(args.aligned_h5ad),
                "--training-dir",
                str(job["training_dir"]),
                "--edge-predictor",
                str(args.edge_predictor),
                "--output-dir",
                str(output_dir),
                "--device",
                "cuda:0",
            ]
            process = subprocess.Popen(
                command,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                env=environment,
                start_new_session=True,
            )
            running[gpu] = {
                "job": job,
                "process": process,
                "log_handle": log_handle,
                "log": str(log_path),
                "command": command,
            }
        write_state("running")
        time.sleep(5)
        for gpu, record in list(running.items()):
            return_code = record["process"].poll()
            if return_code is None:
                continue
            record["log_handle"].close()
            completed.append(
                {
                    "condition": record["job"]["condition"],
                    "analysis_group": record["job"]["analysis_group"],
                    "gpu": gpu,
                    "return_code": int(return_code),
                    "log": record["log"],
                    "command": record["command"],
                }
            )
            del running[gpu]
            write_state("running")
    failed = [record for record in completed if record["return_code"] != 0]
    write_state("failed" if failed else "complete")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
