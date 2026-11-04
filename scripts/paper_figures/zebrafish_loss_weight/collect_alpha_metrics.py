"""Combine two evaluated expression-weight settings for the sensitivity plot."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def collect(reference: Path, alternative: Path) -> pd.DataFrame:
    frames = []
    keys = None
    for path, name in ((reference, "alpha_express_0015"), (alternative, "alpha_expr_005")):
        frame = pd.read_csv(path)
        required = ["time", "space", "w1"]
        if not set(required).issubset(frame):
            raise ValueError(f"{path} must contain {required}")
        frame = frame[required].copy()
        if frame.duplicated(["time", "space"]).any():
            raise ValueError(f"Duplicate time/space values in {path}")
        values = frame["w1"].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"W1 values must be finite and non-negative: {path}")
        current = set(zip(frame["time"], frame["space"]))
        if not current or (keys is not None and current != keys):
            raise ValueError("Both evaluations must cover the same times and spaces.")
        keys = current
        frame["model"] = name
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--alternative", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = collect(args.reference, args.alternative)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
