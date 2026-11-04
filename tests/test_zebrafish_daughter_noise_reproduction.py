from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from reproduction.zebrafish.daughter_noise import ObservedSupportLinkPredictor
from reproduction.zebrafish import plot_daughter_noise as plotting


def test_support_filter_uses_both_expression_endpoints():
    class Predictor(torch.nn.Module):
        def forward(self, pairs):
            return torch.ones((len(pairs), 1)) * 3

    predictor = ObservedSupportLinkPredictor(Predictor(), np.array([[-1., -1.], [1., 1.]], dtype=np.float32))
    pairs = torch.zeros((3, 8))
    pairs[1, 2] = 2.  # sender outside expression support
    pairs[2, 6] = 2.  # receiver outside expression support
    values = predictor(pairs).ravel()
    assert values[0] == 3
    assert (values[1:] < -1e8).all()


def make_run(root: Path):
    root.mkdir()
    composition, transitions, particles = [], [], []
    for noise, fractions in [(0., [.5, .5]), (.01, [.25, .75])]:
        for label, fraction in zip(["A", "B"], fractions):
            common = dict(daughter_noise_std=noise, seed=42, time=4.)
            composition.append(dict(**common, celltype=label, count=int(100*fraction),
                                    fraction=fraction, n_particles=100))
            transitions.append(dict(**common, source_celltype="A", target_celltype=label,
                                    count=int(100*fraction), fraction_within_source=fraction,
                                    n_source_initial=100, n_source_descendants=100))
        particles.append(dict(daughter_noise_std=noise, seed=42, time=4., n_particles=100))
    pd.DataFrame(composition).to_csv(root / "composition_long.csv", index=False)
    pd.DataFrame(transitions).to_csv(root / "lineage_transition_long.csv", index=False)
    pd.DataFrame(particles).to_csv(root / "particle_counts.csv", index=False)
    pd.DataFrame(dict(time=[4.], celltype=["A"], count=[100], fraction=[1.], n_cells=[100])).to_csv(
        root / "observed_composition.csv", index=False)


def test_collector_calculates_total_variation_and_rejects_duplicate_seed(tmp_path, monkeypatch):
    run = tmp_path / "seed_42"
    make_run(run)
    pair_file = tmp_path / "pairs.csv"
    pd.DataFrame(dict(source_celltype=["A"], target_celltype=["B"], global_flow=[.5])).to_csv(pair_file, index=False)
    monkeypatch.setattr(plotting, "PAIR_FILE", pair_file)
    tables = plotting.collect([run], tmp_path / "figure")
    row = tables["sensitivity"].query("daughter_noise_std > 0").iloc[0]
    assert row.composition_tv_from_reference == pytest.approx(.25)
    assert row.lineage_weighted_tv_from_reference == pytest.approx(.25)
    assert row.lineage_max_source_tv_from_reference == pytest.approx(.25)
    assert tables["lineage_values"].query("daughter_noise_std > 0").fraction.iloc[0] == pytest.approx(.75)
    with pytest.raises(ValueError, match="exactly once"):
        plotting.collect([run, run], tmp_path / "duplicate")
