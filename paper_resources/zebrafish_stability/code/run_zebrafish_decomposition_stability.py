#!/usr/bin/env python3
"""Run the prepared Zebrafish stability fits on a fixed pool of GPUs."""

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
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--trainer", type=Path, required=True)
    parser.add_argument("--aligned-h5ad", type=Path, required=True)
    parser.add_argument("--edge-predictor", type=Path, required=True)
    parser.add_argument("--gpu-count", type=int, default=8)
    parser.add_argument("--status-name", default="training_matrix_status.json")
    return parser.parse_args()


def main() -> int:
    args = arguments()
    plan_path = args.plan.expanduser().resolve(strict=True)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    run_root = plan_path.parent
    logs_dir = run_root / "logs"
    status_dir = run_root / "status"
    logs_dir.mkdir(parents=True, exist_ok=True)
    status_dir.mkdir(parents=True, exist_ok=True)
    state_path = status_dir / str(args.status_name)
    if state_path.exists():
        raise FileExistsError(f"Status file already exists: {state_path}")

    pending = list(plan["jobs"])
    running: dict[int, dict] = {}
    completed = []
    gpu_ids = list(range(int(args.gpu_count)))
    if not gpu_ids:
        raise ValueError("gpu-count must be positive")

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
                    "started_at_utc": record["started_at_utc"],
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
            job = pending.pop(0)
            condition = job["condition"]
            log_path = logs_dir / f"{condition}.train.log"
            log_handle = log_path.open("xb")
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
            environment["PYTHONHASHSEED"] = str(job["training_seed"])
            environment["OMP_NUM_THREADS"] = "8"
            environment["MKL_NUM_THREADS"] = "8"
            existing_pythonpath = environment.get("PYTHONPATH", "")
            environment["PYTHONPATH"] = (
                str(args.release)
                if not existing_pythonpath
                else f"{args.release}{os.pathsep}{existing_pythonpath}"
            )
            command = [
                str(args.python),
                str(args.trainer),
                "--aligned-h5ad",
                str(args.aligned_h5ad),
                "--training-config",
                str(job["config"]),
                "--training-dir",
                str(job["training_dir"]),
                "--edge-predictor-path",
                str(args.edge_predictor),
                "--edge-predictor-threshold",
                str(plan["edge_predictor_threshold"]),
                "--interaction-cutoff",
                str(job["interaction_cutoff"]),
                "--expected-alpha-express",
                str(job["expected_alpha_express"]),
                "--device",
                "cuda:0",
            ]
            started = utc_now()
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
                "log_path": str(log_path),
                "started_at_utc": started,
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
                    "gpu": gpu,
                    "return_code": int(return_code),
                    "started_at_utc": record["started_at_utc"],
                    "finished_at_utc": utc_now(),
                    "log": record["log_path"],
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
