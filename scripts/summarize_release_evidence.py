#!/usr/bin/env python3
"""Summarize the compact scientific evidence shipped with the documentation."""

from __future__ import annotations

from collections import Counter
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs" / "data"


def read_rows(name: str) -> list[dict[str, str]]:
    with (DATA / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    winners = read_rows("unified_w2_winners.csv")
    k_policy = read_rows("formal_k_policy.csv")
    compute = read_rows("formal_training_compute_cost.csv")
    hyperparameters = read_rows("formal_hyperparameter_settings.csv")

    winner_counts = Counter(row["method"] for row in winners)
    spatial_cytobridge = sum(
        row["space"] == "spatial" and row["method"] == "CytoBridge"
        for row in winners
    )
    spatial_total = sum(row["space"] == "spatial" for row in winners)

    print("Unified W2 winners:")
    for method, count in winner_counts.most_common():
        print(f"  {method}: {count}")
    print(f"CytoBridge spatial wins: {spatial_cytobridge}/{spatial_total}")
    print(
        "Formal label policy: "
        + ", ".join(
            f"{row['dataset']} k={row['formal_analysis_k']}" for row in k_policy
        )
    )
    print(f"Formal training rows: {len(compute)}")
    print(f"Formal hyperparameter rows: {len(hyperparameters)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
