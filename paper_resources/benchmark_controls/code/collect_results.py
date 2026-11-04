"""Collect the five completed SpaTrack runs and compare matched Sliced-W2 values."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    targets = {"zebrafish": [1, 2, 3], "mosta": [1, 2], "arista": [1, 2, 3],
               "admouse": [1], "chicken_heart": [1, 2]}
    frames, fit_rows = [], []
    for dataset, expected in targets.items():
        folder = args.results_root / dataset
        run = json.loads((folder / "run.json").read_text())
        assert run["status"] == "complete"
        values = pd.read_csv(folder / "metrics_long.csv")
        assert len(values) == len(expected) * 3 * 5
        assert set(values["target"]) == set(expected)
        assert set(values["space"]) == {"joint", "spatial", "state"}
        assert not values.duplicated(["target", "space", "projection_repeat"]).any()
        assert values["n_predicted"].eq(5000).all()
        assert np.isfinite(values[["sliced_w2", "exact_w1", "exact_w2"]]).all().all()
        reference = pd.read_csv(args.reference_root / dataset / "loto_metrics_long.csv")
        reference = reference[reference["method"] == "CytoBridge-0.015"]
        keys = ["target", "space", "projection_repeat", "projection_seed", "n_projections", "n_observed"]
        assert len(values.merge(reference[keys], on=keys)) == len(values)
        # Projection hashes may differ across BLAS implementations at floating
        # point precision. The dataset, dimensions, seed, and repeat must match.
        for fit in run["fits"]:
            assert not fit["warnings"], fit["warnings"]
            assert fit["coupling_checks"]["zero_rows"] == 0
            assert fit["marginal_l1_error_left"] < 1e-6
            assert fit["marginal_l1_error_right"] < 1e-6
            fit_rows.append({"dataset": dataset, **fit})
        frames.append(values)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(output / "spatrack_metrics.csv", index=False)
    keys = ["dataset", "target", "space"]
    means = combined.groupby(keys, as_index=False)["sliced_w2"].mean()
    reference = pd.read_csv(output / "loto_target_stage_means.csv")
    reference = reference[reference["method"] == "CytoBridge-0.015"]
    matched = means.merge(reference[keys + ["sliced_w2"]], on=keys, suffixes=("_spatrack", "_cytobridge"))
    assert len(matched) == 33
    matched["relative_error_pct"] = 100 * (matched["sliced_w2_spatrack"] / matched["sliced_w2_cytobridge"] - 1)
    matched["cytobridge_lower"] = matched["sliced_w2_cytobridge"] < matched["sliced_w2_spatrack"]
    matched.to_csv(output / "spatrack_paired_comparison.csv", index=False)
    summary = matched.groupby("dataset", as_index=False).agg(
        comparisons=("target", "size"), cytobridge_lower=("cytobridge_lower", "sum"),
        mean_relative_error_pct=("relative_error_pct", "mean"),
    )
    summary.to_csv(output / "spatrack_dataset_summary.csv", index=False)
    (output / "spatrack_fit_summaries.json").write_text(json.dumps(fit_rows, indent=2) + "\n")
    print(summary.to_string(index=False))
    print("CytoBridge lower:", int(matched["cytobridge_lower"].sum()), "/", len(matched))


if __name__ == "__main__":
    main()
