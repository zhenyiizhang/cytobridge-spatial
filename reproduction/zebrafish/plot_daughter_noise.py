"""Collect daughter-noise runs and draw Supplementary Figure S37."""
from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import matplotlib as mpl
import numpy as np
import pandas as pd

from CytoBridge.results.zebrafish_si import _complete_composition
from CytoBridge.results._zebrafish_si_plot import _RC, _render_daughter_noise

PAIR_FILE = Path(__file__).resolve().parents[2] / "CytoBridge/results/data/zebrafish_si/s33_lineage_pairs.csv"


def collect(run_dirs, output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    directories = [Path(p) for p in run_dirs]
    composition = pd.concat([pd.read_csv(p / "composition_long.csv") for p in directories], ignore_index=True)
    transitions = pd.concat([pd.read_csv(p / "lineage_transition_long.csv") for p in directories], ignore_index=True)
    particles = pd.concat([pd.read_csv(p / "particle_counts.csv") for p in directories], ignore_index=True)
    observed = pd.read_csv(directories[0] / "observed_composition.csv")
    keys = ["daughter_noise_std", "seed", "time", "celltype"]
    if composition.duplicated(keys).any():
        raise ValueError("Each seed must be included exactly once")
    if not np.allclose(composition.groupby(keys[:-1]).fraction.sum(), 1):
        raise ValueError("Cell-type fractions must sum to one in each population")
    for table in (composition, transitions, particles, observed):
        if not np.isfinite(table.select_dtypes(include=np.number).to_numpy()).all():
            raise ValueError("A numerical input contains missing or non-finite values")

    # Keep the six source/target pairs displayed in the paper.
    pairs = pd.read_csv(PAIR_FILE)
    endpoint = transitions[np.isclose(transitions.time, transitions.time.max())]
    selected_rows = []
    for (noise, seed), part in endpoint.groupby(["daughter_noise_std", "seed"]):
        values = pairs[["source_celltype", "target_celltype"]].merge(
            part[["source_celltype", "target_celltype", "fraction_within_source"]],
            how="left", on=["source_celltype", "target_celltype"], validate="one_to_one")
        values = values.rename(columns={"fraction_within_source": "fraction"}).fillna({"fraction": 0.})
        selected_rows.append(values.assign(daughter_noise_std=noise, seed=seed))
    lineage = pd.concat(selected_rows, ignore_index=True)
    sensitivity_rows = []
    for (seed, time), part in composition.groupby(["seed", "time"]):
        frequencies = part.pivot(index="celltype", columns="daughter_noise_std", values="fraction").fillna(0)
        if 0. not in frequencies:
            raise ValueError(f"Seed {seed} lacks the zero-noise comparison")
        lineage_at_time = transitions[(transitions.seed == seed) & np.isclose(transitions.time, time)]
        lineage_frequencies = lineage_at_time.pivot(
            index=["source_celltype", "target_celltype"], columns="daughter_noise_std",
            values="fraction_within_source").fillna(0)
        weights = lineage_at_time.groupby("source_celltype").n_source_initial.first()
        weights = weights / weights.sum()
        for noise in frequencies:
            difference = (frequencies[noise] - frequencies[0.]).abs()
            source_tv = (lineage_frequencies[noise] - lineage_frequencies[0.]).abs().groupby(level=0).sum() * .5
            sensitivity_rows.append(dict(
                seed=seed, time=time, daughter_noise_std=noise,
                composition_tv_from_reference=float(difference.sum()*.5),
                composition_max_abs_fraction_change=float(difference.max()),
                lineage_weighted_tv_from_reference=float((source_tv * weights).sum()),
                lineage_max_source_tv_from_reference=float(source_tv.max()),
            ))
    tables = dict(composition=composition, observed_composition=observed,
                  lineage_values=lineage, lineage_pairs=pairs,
                  sensitivity=pd.DataFrame(sensitivity_rows), particle_counts=particles)
    for name, table in tables.items():
        table.to_csv(output / f"s33_{name}.csv", index=False)
    return tables


def draw(tables, output_dir):
    def summarize(table, keys):
        result = table.groupby(keys).fraction.agg(mean="mean", std="std", count="count").reset_index()
        result["sem"] = result["std"].fillna(0) / np.sqrt(result["count"])
        return result

    composition = tables["composition"]
    results = SimpleNamespace(
        daughter_composition=composition,
        daughter_observed_composition=tables["observed_composition"],
        daughter_lineage_values=tables["lineage_values"],
        daughter_lineage_pairs=tables["lineage_pairs"],
        daughter_sensitivity=tables["sensitivity"],
        daughter_particle_counts=tables["particle_counts"],
    )
    panels = SimpleNamespace(
        daughter_composition_summary=summarize(_complete_composition(composition),
                                               ["daughter_noise_std", "time", "celltype"]),
        daughter_lineage_summary=summarize(tables["lineage_values"],
                                          ["daughter_noise_std", "source_celltype", "target_celltype"]),
        daughter_top_celltypes=tuple(composition[composition.time > 0].groupby("celltype").fraction.mean()
                                    .sort_values(ascending=False).head(6).index),
    )
    with mpl.rc_context(_RC):
        return _render_daughter_noise(results, panels, Path(output_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    tables = collect(args.run_dir, args.output_dir)
    print(draw(tables, args.output_dir))
