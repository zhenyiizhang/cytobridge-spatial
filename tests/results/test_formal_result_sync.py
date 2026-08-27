from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from CytoBridge.results.compute_cost import load_full_model_compute_cost  # noqa: E402
from CytoBridge.results.lr_complex_aggregation import (  # noqa: E402
    DATASET_LABELS,
    load_lr_complex_aggregation_results,
)


FORMAL_DATA = REPOSITORY_ROOT / "docs" / "data"

COMPUTE_ROW_KEYS = {
    "admouse": ("AD", "matched full"),
    "arista": ("ARISTA", "matched full"),
    "chicken_heart": ("Chicken heart", "full learned single profile"),
    "mosta": ("MOSTA", "matched full"),
    "zebrafish": ("Zebrafish", "matched full"),
}


def test_formal_full_model_compute_cost_matches_packaged_measurements() -> None:
    formal = pd.read_csv(FORMAL_DATA / "formal_training_compute_cost.csv")
    packaged = load_full_model_compute_cost().measurements.set_index("dataset")

    for dataset, (label, run_role) in COMPUTE_ROW_KEYS.items():
        rows = formal.loc[
            formal["dataset"].eq(label) & formal["run_role"].eq(run_role)
        ]
        assert len(rows) == 1
        row = rows.iloc[0]
        canonical = packaged.loc[dataset]
        assert int(row["cells"]) == int(canonical["observed_cells_or_spots"])
        np.testing.assert_array_equal(
            [
                row["wall_time_seconds"],
                row["peak_cpu_rss_mib"],
                row["peak_cuda_allocated_mib"],
            ],
            [
                float(f"{canonical['training_time_seconds']:.6f}"),
                float(f"{canonical['peak_host_memory_mib']:.6f}"),
                float(f"{canonical['peak_gpu_allocation_mib']:.6f}"),
            ],
        )


def test_formal_lr_sensitivity_matches_packaged_paired_scores() -> None:
    formal = pd.read_csv(
        FORMAL_DATA / "formal_lr_complex_aggregation_sensitivity.csv",
        float_precision="round_trip",
    ).set_index("dataset")
    results = load_lr_complex_aggregation_results()
    summary = results.dataset_summary.set_index("dataset")

    pd.testing.assert_series_equal(
        formal.loc[summary.index, "n_scored_pairs"].astype(int),
        summary["n_scored_pairs"].astype(int),
        check_names=False,
    )
    pd.testing.assert_series_equal(
        formal.loc[summary.index, "n_multisubunit_pairs"].astype(int),
        summary["n_multisubunit_pairs"].astype(int),
        check_names=False,
    )
    np.testing.assert_allclose(
        formal.loc[summary.index, "global_spearman"],
        summary["pooled_spearman"],
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        formal.loc[summary.index, "min_per_time_spearman"],
        summary["min_per_time_spearman"],
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        formal.loc[summary.index, "min_top10_jaccard"],
        summary["min_top10_jaccard"],
        rtol=0.0,
        atol=1e-12,
    )

    pearson_by_dataset = {}
    max_relative_difference_by_dataset = {}
    for dataset, paired in results.paired_scores.groupby("dataset", sort=False):
        minimum = paired["score_min"].to_numpy(dtype=float)
        geometric = paired["score_geometric_mean"].to_numpy(dtype=float)
        denominator = np.maximum(
            np.maximum(np.abs(minimum), np.abs(geometric)),
            np.finfo(float).eps,
        )
        label = DATASET_LABELS[dataset]
        pearson_by_dataset[label] = float(pearsonr(minimum, geometric).statistic)
        max_relative_difference_by_dataset[label] = float(
            np.max(np.abs(geometric - minimum) / denominator)
        )

    np.testing.assert_allclose(
        formal.loc[summary.index, "global_pearson"],
        pd.Series(pearson_by_dataset).loc[summary.index],
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        formal.loc[summary.index, "max_symmetric_relative_difference"],
        pd.Series(max_relative_difference_by_dataset).loc[summary.index],
        rtol=0.0,
        atol=1e-12,
    )

    # This audit needs the full primary/recomputed tables, which are intentionally
    # absent from the compact package. Keep its documented acceptance gate explicit.
    reproduction_error = formal.loc[
        summary.index, "primary_reproduction_max_abs_error"
    ].to_numpy(dtype=float)
    assert np.isfinite(reproduction_error).all()
    assert (reproduction_error <= 1e-8).all()
