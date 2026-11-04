#!/usr/bin/env python3
"""
Collect manual MOSTA ablation run outputs into a single overview CSV.

Expected per-run files under runs-root/<run_dir>/:
- summary.json
- morphology_delta_summary.json (preferred)
- morphology_delta_by_time.csv (fallback to derive metrics)
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def _safe_read_json(path: str) -> Optional[Dict]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None
    return None


def _derive_morph_from_csv(path: str) -> Dict[str, float]:
    if not os.path.exists(path):
        return {
            "mean_morph_delta": float("nan"),
            "max_morph_delta": float("nan"),
            "auc_morph_delta": float("nan"),
        }
    df = pd.read_csv(path)
    if ("time" not in df.columns) or ("morph_delta_t" not in df.columns):
        return {
            "mean_morph_delta": float("nan"),
            "max_morph_delta": float("nan"),
            "auc_morph_delta": float("nan"),
        }
    d = df.sort_values("time")
    t = d["time"].to_numpy(dtype=float)
    y = d["morph_delta_t"].to_numpy(dtype=float)
    auc = float(np.trapz(y, t)) if len(y) > 1 else 0.0
    return {
        "mean_morph_delta": float(np.mean(y)) if len(y) > 0 else float("nan"),
        "max_morph_delta": float(np.max(y)) if len(y) > 0 else float("nan"),
        "auc_morph_delta": auc,
    }


def _collect_one_run(run_dir: str) -> Optional[Dict]:
    summary_path = os.path.join(run_dir, "summary.json")
    if not os.path.exists(summary_path):
        return None

    summary = _safe_read_json(summary_path)
    if not summary:
        return None

    morph_summary_path = os.path.join(run_dir, "morphology_delta_summary.json")
    morph_csv_path = os.path.join(run_dir, "morphology_delta_by_time.csv")
    morph = _safe_read_json(morph_summary_path)

    if morph and isinstance(morph.get("metrics"), dict):
        metrics = morph["metrics"]
        mean_morph = float(metrics.get("mean_morph_delta", float("nan")))
        max_morph = float(metrics.get("max_morph_delta", float("nan")))
        auc_morph = float(metrics.get("auc_morph_delta", float("nan")))
    else:
        d = _derive_morph_from_csv(morph_csv_path)
        mean_morph = d["mean_morph_delta"]
        max_morph = d["max_morph_delta"]
        auc_morph = d["auc_morph_delta"]

    row = {
        "run_dir": run_dir,
        "run_name": os.path.basename(run_dir.rstrip("/")),
        "target_label": summary.get("target_label"),
        "gif_path": summary.get("gif_path"),
        "n_frames": summary.get("n_frames"),
        "time_start": summary.get("time_start"),
        "time_end": summary.get("time_end"),
        "time_step": summary.get("time_step"),
        "n_init_baseline": summary.get("n_init_baseline"),
        "n_init_ablation": summary.get("n_init_ablation"),
        "n_target_removed_t0": summary.get("n_target_t0_removed"),
        "mean_morph_delta": mean_morph,
        "max_morph_delta": max_morph,
        "auc_morph_delta": auc_morph,
        "summary_json": summary_path,
        "morphology_delta_summary_json": morph_summary_path if os.path.exists(morph_summary_path) else "",
        "morphology_delta_by_time_csv": morph_csv_path if os.path.exists(morph_csv_path) else "",
    }
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect manual MOSTA ablation run summaries")
    parser.add_argument("--runs-root", default="results/mosta_high_impact_annotation_ablation")
    parser.add_argument("--out-csv", default="manual_batch_overview.csv")
    args = parser.parse_args()

    runs_root = args.runs_root
    if not os.path.isdir(runs_root):
        raise FileNotFoundError(f"--runs-root not found: {runs_root}")

    run_dirs: List[str] = []
    for name in sorted(os.listdir(runs_root)):
        p = os.path.join(runs_root, name)
        if os.path.isdir(p):
            run_dirs.append(p)

    rows: List[Dict] = []
    for rd in run_dirs:
        row = _collect_one_run(rd)
        if row is not None:
            rows.append(row)

    if not rows:
        raise ValueError(f"No valid run summaries found under: {runs_root}")

    out_df = pd.DataFrame(rows)
    # Sort by morphology impact descending; NaNs sink to bottom.
    out_df["__sort_auc__"] = out_df["auc_morph_delta"].fillna(-1e9)
    out_df = out_df.sort_values("__sort_auc__", ascending=False).drop(columns=["__sort_auc__"]).reset_index(drop=True)
    out_df["rank_by_auc_morph_delta"] = np.arange(1, len(out_df) + 1)

    out_csv = args.out_csv
    if not os.path.isabs(out_csv):
        out_csv = os.path.join(runs_root, out_csv)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    print("Saved:", out_csv)
    print(out_df[["rank_by_auc_morph_delta", "target_label", "auc_morph_delta", "mean_morph_delta", "max_morph_delta"]].to_string(index=False))


if __name__ == "__main__":
    main()

