#!/usr/bin/env python3
"""Draw zebrafish loss-weight sensitivity from the evaluation tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--alpha-metrics",
        required=True,
        type=Path,
        help="CSV with columns model, time, space, and w1 for the two expression weights",
    )
    parser.add_argument(
        "--evaluation-root",
        required=True,
        type=Path,
        help="directory containing reference, ot_mass_10_to_1, and ot_mass_1_to_10 evaluations",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def load_results(alpha_metrics: Path, evaluation_root: Path) -> pd.DataFrame:
    alpha = pd.read_csv(alpha_metrics)
    required = {"model", "time", "space", "w1"}
    if not required.issubset(alpha.columns):
        raise ValueError(f"{alpha_metrics} must contain {sorted(required)}")
    alpha = alpha.loc[:, ["time", "space", "w1", "model"]].rename(
        columns={"model": "condition"}
    )
    alpha["condition"] = alpha["condition"].replace(
        {"alpha_express_0015": "reference_alpha"}
    )

    frames = [alpha]
    for folder, condition in (
        ("reference", "reference_ratio"),
        ("ot_mass_10_to_1", "ot_mass_10_to_1"),
        ("ot_mass_1_to_10", "ot_mass_1_to_10"),
    ):
        path = evaluation_root / folder / "distribution_metrics.csv"
        frame = pd.read_csv(path)
        if not {"time", "space", "w1"}.issubset(frame.columns):
            raise ValueError(f"{path} must contain time, space, and w1")
        frame = frame.loc[:, ["time", "space", "w1"]].copy()
        frame["condition"] = condition
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    if not np.isfinite(result["w1"]).all() or (result["w1"] < 0).any():
        raise ValueError("W1 values must be finite and non-negative")
    return result


def draw(frame: pd.DataFrame, output_dir: Path) -> tuple[Path, Path]:
    from CytoBridge.results._zebrafish_loss import draw as draw_current_s36
    return draw_current_s36(frame, output_dir)


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    frame = load_results(
        args.alpha_metrics.expanduser().resolve(strict=True),
        args.evaluation_root.expanduser().resolve(strict=True),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "loss_weight_metrics.csv", index=False)
    # The combined SI notebook uses these two names for the same control model.
    si_input = frame.assign(condition=frame["condition"].replace({
        "reference_alpha": "formal_alpha_control", "reference_ratio": "formal",
    }))
    si_input.to_csv(output_dir / "s32_loss_weight_metrics.csv", index=False)
    summary = frame.groupby(["condition", "space"], sort=False)["w1"].mean().reset_index(name="mean_w1")
    summary.to_csv(output_dir / "summary_statistics.csv", index=False)
    for path in draw(frame, output_dir):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
