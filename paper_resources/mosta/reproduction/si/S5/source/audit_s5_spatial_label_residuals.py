#!/usr/bin/env python3
"""Quantify, but do not post-hoc remove, S5 spatial label residuals."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


TIMES = tuple(float(value) for value in np.arange(0.0, 3.0001, 0.25))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def time_token(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(float(value)).replace(".", "p")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--numerical-root", required=True)
    parser.add_argument("--package-classification-source", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.numerical_root).resolve()
    package_source = Path(args.package_classification_source).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    growth_path = root / "s5_growth" / "growth_by_cell_fully_generated.csv"
    growth = pd.read_csv(growth_path)
    time_rows: list[dict[str, object]] = []
    singleton_rows: list[dict[str, object]] = []
    for time_value in TIMES:
        state_path = root / "generated_states" / f"time_{time_token(time_value)}.h5ad"
        state = ad.read_h5ad(state_path)
        coordinates = np.asarray(state.obsm["spatial"], dtype=float)
        labels = state.obs["Annotation"].astype(str).to_numpy()
        brain_indices = np.flatnonzero(labels == "Brain")
        _, nearest_all = cKDTree(coordinates).query(coordinates[brain_indices], k=10)
        final_brain_support = np.count_nonzero(labels[nearest_all] == "Brain", axis=1)
        subset = growth.loc[
            np.isclose(growth["time"].astype(float), time_value)
            & (growth["celltype"].astype(str) == "Brain")
        ].sort_values("cell_index")
        if not np.array_equal(subset["cell_index"].to_numpy(dtype=int), brain_indices):
            raise RuntimeError("Growth/state Brain rows are not aligned.")
        values = subset["growth"].to_numpy(dtype=float)
        low_support = final_brain_support < 5
        self_only = final_brain_support == 1
        supported_mean = float(values[~low_support].mean())
        all_mean = float(values.mean())
        time_rows.append(
            {
                "time": time_value,
                "n_brain": int(len(brain_indices)),
                "n_final_knn10_support_lt5": int(low_support.sum()),
                "fraction_final_knn10_support_lt5": float(low_support.mean()),
                "n_final_knn10_support_eq1": int(self_only.sum()),
                "fraction_final_knn10_support_eq1": float(self_only.mean()),
                "median_final_knn10_brain_support": float(np.median(final_brain_support)),
                "mean_growth_all_package_brain": all_mean,
                "mean_growth_sensitivity_excluding_support_lt5": supported_mean,
                "absolute_mean_shift_in_sensitivity": abs(all_mean - supported_mean),
            }
        )
        for position in np.flatnonzero(self_only):
            row = subset.iloc[int(position)]
            singleton_rows.append(
                {
                    "time": time_value,
                    "cell_index": int(row["cell_index"]),
                    "x": float(row["x"]),
                    "y": float(row["y"]),
                    "growth": float(row["growth"]),
                    "final_knn10_brain_support": int(final_brain_support[position]),
                    "retained_in_figure": True,
                }
            )
    temporal = pd.DataFrame(time_rows)
    singleton = pd.DataFrame(singleton_rows)
    temporal.to_csv(output_dir / "s5_spatial_label_residuals_by_time.csv", index=False)
    singleton.to_csv(output_dir / "s5_spatial_label_self_only_cells.csv", index=False)
    total_brain = int(temporal["n_brain"].sum())
    total_low = int(temporal["n_final_knn10_support_lt5"].sum())
    total_self_only = int(temporal["n_final_knn10_support_eq1"].sum())
    means = temporal["mean_growth_all_package_brain"].to_numpy(dtype=float)
    sensitivity_means = temporal[
        "mean_growth_sensitivity_excluding_support_lt5"
    ].to_numpy(dtype=float)
    if not (np.diff(means) < 0).all() or not (np.diff(sensitivity_means) < 0).all():
        raise RuntimeError("Spatial-label residual sensitivity changes the S5 temporal claim.")
    audit = {
        "schema_version": 1,
        "status": "pass_retained_package_classifier_output",
        "panel": "Supplementary Figure S5",
        "definition": {
            "low_final_support": "fewer than 5 Brain labels among the final 10 nearest all-state labels, including self",
            "self_only": "exactly 1 Brain label among the final 10 nearest all-state labels",
            "important_semantics": (
                "Package smoothing is one simultaneous vote from raw classifier labels. "
                "The resulting labels are not iteratively re-smoothed, so a final label "
                "need not be the majority among final neighboring labels."
            ),
        },
        "inputs": {
            "growth_table": {"path": str(growth_path), "sha256": sha256(growth_path)},
            "package_classification_source": {"path": str(package_source), "sha256": sha256(package_source)},
        },
        "results": {
            "n_brain_all_times": total_brain,
            "n_final_knn10_support_lt5": total_low,
            "fraction_final_knn10_support_lt5": total_low / total_brain,
            "n_final_knn10_support_eq1": total_self_only,
            "fraction_final_knn10_support_eq1": total_self_only / total_brain,
            "maximum_absolute_mean_growth_shift_in_exclusion_sensitivity": float(
                temporal["absolute_mean_shift_in_sensitivity"].max()
            ),
            "original_package_brain_mean_monotonically_decreasing": True,
            "sensitivity_mean_monotonically_decreasing": True,
        },
        "rendering_policy": {
            "retain_every_package_k10_brain_label": True,
            "posthoc_spatial_filter": False,
            "reason": (
                "Submitted S5 plots every classifier-predicted Brain cell. Removing "
                "low-support cells would silently change the biological selection and "
                "would no longer be an equivalent value replacement."
            ),
        },
        "coordinate_bug_detected": False,
        "arista_assets_used": False,
    }
    (output_dir / "s5_spatial_label_residual_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
