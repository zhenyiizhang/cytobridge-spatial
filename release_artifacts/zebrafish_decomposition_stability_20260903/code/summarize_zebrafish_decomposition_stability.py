#!/usr/bin/env python3
"""Summarize the completed Zebrafish decomposition-stability evaluations."""

from __future__ import annotations

import argparse
from itertools import combinations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr


REFERENCE = "formal_seed42_cutoff1p0"
SEED_CONDITIONS = [f"formal_seed{seed}_cutoff1p0" for seed in (42, 43, 44, 46, 47)]
SETTING_COMPARISONS = [
    *[
        (f"formal_seed{seed}_cutoff0p8", f"formal_seed{seed}_cutoff1p0", "Neighborhood 0.8×", seed)
        for seed in (42, 43, 44)
    ],
    *[
        (f"formal_seed{seed}_cutoff1p2", f"formal_seed{seed}_cutoff1p0", "Neighborhood 1.2×", seed)
        for seed in (42, 43, 44)
    ],
    ("alpha_expr_005_seed42_cutoff1p0", REFERENCE, "Expression loss weight 0.05", 42),
    ("ot_mass_10_to_1_seed42_cutoff1p0", REFERENCE, "Transport:mass weight 10:1", 42),
    ("ot_mass_1_to_10_seed42_cutoff1p0", REFERENCE, "Transport:mass weight 1:10", 42),
]
COMPONENTS = ("total", "intrinsic", "interaction")
SPACES = {"joint": slice(None), "spatial": slice(0, 2), "state": slice(2, None)}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def condition_label(condition: str) -> str:
    if condition.startswith("formal_seed") and "cutoff1p0" in condition:
        return f"Seed {condition.split('seed', 1)[1].split('_', 1)[0]}"
    if "cutoff0p8" in condition:
        seed = condition.split("seed", 1)[1].split("_", 1)[0]
        return f"0.8×, seed {seed}"
    if "cutoff1p2" in condition:
        seed = condition.split("seed", 1)[1].split("_", 1)[0]
        return f"1.2×, seed {seed}"
    return {
        "alpha_expr_005_seed42_cutoff1p0": "Expression 0.05",
        "ot_mass_10_to_1_seed42_cutoff1p0": "Transport:mass 10:1",
        "ot_mass_1_to_10_seed42_cutoff1p0": "Transport:mass 1:10",
    }.get(condition, condition)


def load_arrays(root: Path, condition: str) -> dict[str, np.ndarray]:
    path = root / condition / "observed_cell_components.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=True) as archive:
        return {key: archive[key] for key in archive.files}


def cellwise_cosine(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, float]:
    left_norm = np.linalg.norm(left, axis=1)
    right_norm = np.linalg.norm(right, axis=1)
    valid = (left_norm > 1e-10) & (right_norm > 1e-10)
    values = np.sum(left[valid] * right[valid], axis=1) / (left_norm[valid] * right_norm[valid])
    return np.clip(values, -1.0, 1.0), float(valid.mean())


def component_agreement(
    arrays: dict[str, dict[str, np.ndarray]],
    left_name: str,
    right_name: str,
) -> list[dict]:
    left = arrays[left_name]
    right = arrays[right_name]
    if not np.array_equal(left["obs_names"], right["obs_names"]):
        raise ValueError(f"Cell order differs: {left_name} vs {right_name}")
    rows = []
    for component in COMPONENTS:
        for space, columns in SPACES.items():
            values, valid_fraction = cellwise_cosine(
                np.asarray(left[component])[:, columns],
                np.asarray(right[component])[:, columns],
            )
            rows.append(
                {
                    "condition_a": left_name,
                    "condition_b": right_name,
                    "component": component,
                    "space": space,
                    "n_valid_cells": int(values.size),
                    "valid_cell_fraction": valid_fraction,
                    "cosine_median": float(np.median(values)),
                    "cosine_q25": float(np.quantile(values, 0.25)),
                    "cosine_q75": float(np.quantile(values, 0.75)),
                    "cosine_mean": float(np.mean(values)),
                    "flattened_cosine": float(
                        np.dot(
                            np.asarray(left[component])[:, columns].ravel(),
                            np.asarray(right[component])[:, columns].ravel(),
                        )
                        /
                        (
                            np.linalg.norm(np.asarray(left[component])[:, columns])
                            * np.linalg.norm(np.asarray(right[component])[:, columns])
                        )
                    ),
                }
            )
    rho = float(spearmanr(left["growth"], right["growth"]).statistic)
    rows.append(
        {
            "condition_a": left_name,
            "condition_b": right_name,
            "component": "growth",
            "space": "scalar",
            "n_valid_cells": int(left["growth"].size),
            "valid_cell_fraction": 1.0,
            "cosine_median": rho,
            "cosine_q25": np.nan,
            "cosine_q75": np.nan,
            "cosine_mean": rho,
            "flattened_cosine": rho,
        }
    )
    return rows


def pair_vector(root: Path, condition: str) -> pd.Series:
    table = pd.read_csv(root / condition / "directed_celltype_pair_attribution.csv")
    keys = ["time", "sender_type", "receiver_type"]
    return table.groupby(keys, sort=True)["D_AB_state"].mean()


def aligned_pair_vectors(left: pd.Series, right: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    index = left.index.union(right.index)
    left_values = left.reindex(index, fill_value=0.0).to_numpy(float)
    right_values = right.reindex(index, fill_value=0.0).to_numpy(float)
    active = (left_values > 1e-12) | (right_values > 1e-12)
    return left_values[active], right_values[active]


def weighted_top_jaccard(left: np.ndarray, right: np.ndarray, fraction: float = 0.20) -> float:
    count = max(1, int(np.ceil(left.size * fraction)))
    left_top = np.argpartition(left, -count)[-count:]
    right_top = np.argpartition(right, -count)[-count:]
    union = np.union1d(left_top, right_top)
    denominator = float(np.maximum(left[union], right[union]).sum())
    if denominator <= 0:
        return np.nan
    return float(np.minimum(left[union], right[union]).sum() / denominator)


def pair_agreement(left: pd.Series, right: pd.Series) -> dict[str, float]:
    left_values, right_values = aligned_pair_vectors(left, right)
    rho = float(spearmanr(left_values, right_values).statistic)
    count = max(1, int(np.ceil(left_values.size * 0.20)))
    left_top = set(np.argpartition(left_values, -count)[-count:].tolist())
    right_top = set(np.argpartition(right_values, -count)[-count:].tolist())
    return {
        "directed_pair_spearman": rho,
        "top20_set_jaccard": float(len(left_top & right_top) / len(left_top | right_top)),
        "top20_weighted_jaccard": weighted_top_jaccard(left_values, right_values),
        "n_time_pair_entries": int(left_values.size),
    }


def interaction_fraction_rows(condition: str, arrays: dict[str, np.ndarray]) -> list[dict]:
    times = np.asarray(arrays["times"], dtype=float)
    interaction_norm = np.linalg.norm(np.asarray(arrays["interaction"])[:, 2:], axis=1)
    intrinsic_norm = np.linalg.norm(np.asarray(arrays["intrinsic"])[:, 2:], axis=1)
    denominator = interaction_norm + intrinsic_norm
    fraction = np.divide(
        interaction_norm,
        denominator,
        out=np.zeros_like(interaction_norm),
        where=denominator > 1e-10,
    )
    rows = []
    for time_value in np.unique(times):
        values = fraction[np.isclose(times, time_value)]
        rows.append(
            {
                "condition": condition,
                "condition_label": condition_label(condition),
                "time": float(time_value),
                "n_cells": int(values.size),
                "interaction_fraction_median": float(np.median(values)),
                "interaction_fraction_q25": float(np.quantile(values, 0.25)),
                "interaction_fraction_q75": float(np.quantile(values, 0.75)),
            }
        )
    return rows


def main() -> int:
    args = arguments()
    root = args.evaluation_root.expanduser().resolve(strict=True)
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    required = sorted(
        set(SEED_CONDITIONS)
        | {name for comparison in SETTING_COMPARISONS for name in comparison[:2]}
    )
    arrays = {condition: load_arrays(root, condition) for condition in required}
    pair_vectors = {condition: pair_vector(root, condition) for condition in required}

    seed_rows = []
    seed_pair_rows = []
    for left_name, right_name in combinations(SEED_CONDITIONS, 2):
        comparison_id = f"{condition_label(left_name)} vs {condition_label(right_name)}"
        for row in component_agreement(arrays, left_name, right_name):
            row["comparison"] = comparison_id
            seed_rows.append(row)
        pair_row = {
            "condition_a": left_name,
            "condition_b": right_name,
            "comparison": comparison_id,
            **pair_agreement(pair_vectors[left_name], pair_vectors[right_name]),
        }
        seed_pair_rows.append(pair_row)
    pd.DataFrame(seed_rows).to_csv(output / "training_seed_component_agreement.csv", index=False)
    pd.DataFrame(seed_pair_rows).to_csv(output / "training_seed_directed_pair_agreement.csv", index=False)

    setting_rows = []
    setting_pair_rows = []
    for condition, matching_default, setting, replicate in SETTING_COMPARISONS:
        for row in component_agreement(arrays, condition, matching_default):
            row["setting"] = setting
            row["replicate"] = replicate
            setting_rows.append(row)
        setting_pair_rows.append(
            {
                "condition": condition,
                "matching_default": matching_default,
                "setting": setting,
                "replicate": replicate,
                **pair_agreement(pair_vectors[condition], pair_vectors[matching_default]),
            }
        )
    pd.DataFrame(setting_rows).to_csv(output / "model_setting_component_agreement.csv", index=False)
    pd.DataFrame(setting_pair_rows).to_csv(output / "model_setting_directed_pair_agreement.csv", index=False)

    magnitude_rows = []
    for condition in required:
        magnitude_rows.extend(interaction_fraction_rows(condition, arrays[condition]))
    pd.DataFrame(magnitude_rows).to_csv(output / "interaction_fraction_by_time.csv", index=False)

    reference_pair_table = pd.read_csv(root / REFERENCE / "directed_celltype_pair_attribution.csv")
    reference_mean = reference_pair_table.groupby(["sender_type", "receiver_type"])["D_AB_state"].mean()
    top_pairs = reference_mean.sort_values(ascending=False).head(12).index.tolist()
    rank_rows = []
    display_order = [
        *SEED_CONDITIONS,
        *[f"formal_seed{seed}_cutoff0p8" for seed in (42, 43, 44)],
        *[f"formal_seed{seed}_cutoff1p2" for seed in (42, 43, 44)],
        "alpha_expr_005_seed42_cutoff1p0",
        "ot_mass_10_to_1_seed42_cutoff1p0",
        "ot_mass_1_to_10_seed42_cutoff1p0",
    ]
    for condition_index, condition in enumerate(display_order):
        table = pd.read_csv(root / condition / "directed_celltype_pair_attribution.csv")
        means = table.groupby(["sender_type", "receiver_type"])["D_AB_state"].mean()
        percentile = pd.Series(
            100.0 * rankdata(means.to_numpy(float), method="average") / len(means),
            index=means.index,
        )
        descending_rank = pd.Series(
            rankdata(-means.to_numpy(float), method="average"),
            index=means.index,
        )
        for pair_index, pair in enumerate(top_pairs):
            rank_rows.append(
                {
                    "condition": condition,
                    "condition_label": condition_label(condition),
                    "condition_order": condition_index,
                    "sender_type": pair[0],
                    "receiver_type": pair[1],
                    "pair_label": f"{pair[0]} → {pair[1]}",
                    "pair_order": pair_index,
                    "reference_mean_D_AB_state": float(reference_mean.loc[pair]),
                    "condition_mean_D_AB_state": float(means.get(pair, 0.0)),
                    "condition_percentile": float(percentile.get(pair, 0.0)),
                    "condition_rank_descending": float(
                        descending_rank.get(pair, len(means))
                    ),
                    "n_directed_pairs": int(len(means)),
                }
            )
    pd.DataFrame(rank_rows).to_csv(output / "reference_top_directed_pair_percentiles.csv", index=False)

    w1_rows = []
    for condition in display_order:
        metrics = pd.read_csv(root / condition / "distribution_evaluation" / "distribution_metrics.csv")
        for space, group in metrics.groupby("space", sort=False):
            w1_rows.append(
                {
                    "condition": condition,
                    "condition_label": condition_label(condition),
                    "space": "state" if space == "pca" else str(space),
                    "n_target_times": int(group.shape[0]),
                    "w1_mean": float(group["w1"].mean()),
                    "w1_median": float(group["w1"].median()),
                }
            )
    w1 = pd.DataFrame(w1_rows)
    reference_w1 = w1.loc[w1["condition"].eq(REFERENCE)].set_index("space")["w1_mean"]
    w1["w1_relative_to_seed42"] = [
        value / float(reference_w1.loc[space])
        for value, space in zip(w1["w1_mean"], w1["space"])
    ]
    w1.to_csv(output / "distribution_w1_summary.csv", index=False)

    summary = {
        "schema_version": 1,
        "status": "complete",
        "dataset": "Zebrafish",
        "n_observed_cells": int(arrays[REFERENCE]["times"].size),
        "n_observed_time_points": int(np.unique(arrays[REFERENCE]["times"]).size),
        "training_seeds": [42, 43, 44, 46, 47],
        "excluded_training_attempt": (
            "Seed 45 was not analyzed because two independent attempts terminated "
            "with a non-finite score-matching loss at epoch 320, before Finetune."
        ),
        "n_pairwise_seed_comparisons": 10,
        "neighborhood_factors": [0.8, 1.0, 1.2],
        "primary_interaction_metric": "D_AB_state exact directed-edge message attribution",
        "primary_vector_space": "50-dimensional aligned state space; spatial coordinates excluded",
        "comparison_rule": (
            "Cutoff variants are compared with the default fit of the same training seed. "
            "Other one-factor settings are compared with the accepted seed-42 default."
        ),
    }
    (output / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
