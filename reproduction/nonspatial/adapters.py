"""Convert explicit analysis outputs to the S4/S5 table schemas.

These conversions do not load the included figure results. Inference repeats
are averaged within the selected fitted arm, never treated as training seeds.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd


def distribution_inputs(paired_metrics, output_dir):
    """Pivot public/paper-wrapper long metrics into S4 and S5 input tables."""
    table = pd.read_csv(paired_metrics)
    keys, metrics = ["time", "space"], ["w1", "w2", "tmv_absolute"]
    if set(table.condition) != {"full", "no_interaction"}:
        raise ValueError("Expected exactly full and no_interaction conditions")
    table = table[table.space == "pca"]
    if table.empty or not np.isfinite(table[metrics].to_numpy(float)).all():
        raise ValueError("Expected finite PCA distribution metrics")
    means = table.groupby(["condition", *keys], as_index=False)[metrics].mean()
    arms = {arm: means.loc[means.condition == arm, keys + metrics].copy()
            for arm in ("full", "no_interaction")}
    wide = arms["full"].merge(arms["no_interaction"], on=keys,
        suffixes=("_full", "_no_interaction"), validate="one_to_one", how="outer")
    if wide.isna().any().any():
        raise ValueError("Both arms must have matching endpoint times")
    for metric in metrics:
        if (wide[f"{metric}_full"] == 0).any():
            raise ValueError(f"Cannot calculate relative change with zero Full {metric}")
        name = "tmv" if metric == "tmv_absolute" else metric
        wide[f"{name}_relative_change"] = (wide[f"{metric}_no_interaction"] - wide[f"{metric}_full"]) / wide[f"{metric}_full"]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    paths = {"distribution": output / "paired_distribution.csv"}
    wide.to_csv(paths["distribution"], index=False)
    for arm, frame in arms.items():
        paths[f"{arm}_distribution"] = output / f"{arm}_distribution.csv"
        frame.to_csv(paths[f"{arm}_distribution"], index=False)
    return paths


def clone_fate_input(analysis_dir, output_csv):
    """Read actual full/summary.json and no_interaction/summary.json outputs."""
    root = Path(analysis_dir)
    summaries = {arm: json.loads((root / arm / "summary.json").read_text())
                 for arm in ("full", "no_interaction")}
    rows = []
    for metric in ("tv_agreement", "js_similarity", "dominant_fate_match"):
        full, no = [float(summaries[arm][f"clone_macro_{metric}"])
                    for arm in ("full", "no_interaction")]
        if not np.isfinite([full, no]).all() or full == 0:
            raise ValueError(f"Invalid clone-fate summary for {metric}")
        rows.append({"metric": metric, "full": full, "no_interaction": no,
            "delta_no_interaction_minus_full": no - full,
            "relative_change": (no - full) / full, "higher_is_better": True})
    path = Path(output_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def weinreb_cellchat_input(exact_message_summary, cellchat_reference, output_csv, *, time_mapping=None):
    """Join newly computed D_AB to the independent observed CellChat scores.

    Only CellChat scores and the measured-cell eligibility are reused from the
    reference. Its old CytoBridge message scores are never retained.
    ``time_mapping={0: 2, 1: 4, 2: 6}`` converts a new run's model times to days.
    """
    keys = ["time", "sender_type", "receiver_type"]
    drift = pd.read_csv(exact_message_summary)[keys + ["D_AB_mean"]]
    if time_mapping is not None:
        drift["time"] = drift.time.map(time_mapping)
        if drift.time.isna().any():
            raise ValueError("time_mapping must cover every attribution time")
    reference = pd.read_csv(cellchat_reference)[keys + ["heterotypic", "min10_eligible", "cellchat_native_raw"]]
    joined = reference.merge(drift.rename(columns={"D_AB_mean": "message_D_AB_raw"}),
        on=keys, how="left", validate="one_to_one")
    if joined.message_D_AB_raw.isna().any():
        raise ValueError("New attribution is missing CellChat comparison pairs")
    path = Path(output_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    joined.to_csv(path, index=False)
    return path
