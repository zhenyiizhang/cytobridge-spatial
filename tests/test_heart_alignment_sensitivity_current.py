"""Checks for the numerical inputs of the current S7/S8 figures."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]/'reproduction/chicken_heart/alignment_sensitivity_20260906'


def test_actual_integer_perturbation_amplitudes():
    manifest = json.loads((ROOT/'input_manifest.json').read_text())
    assert len(manifest['variants']) == 6
    for record in manifest['variants']:
        name = record['variant']
        stages = record['stage_records']
        assert {s['timepoint'] for s in stages} == {'D4','D7','D10','D14'}
        shifts = [np.hypot(s['translate_x_nn'],s['translate_y_nn']) for s in stages]
        angles = [abs(s['rotation_deg']) for s in stages]
        distance = 0 if name.startswith('rotate_') else (1 if name.endswith('_low') else 2)
        angle = 0 if name.startswith('translate_') and not name.startswith('translate_rotate_') else (1 if name.endswith('_low') else 3)
        np.testing.assert_allclose(shifts,distance,rtol=0,atol=1e-12)
        np.testing.assert_allclose(angles,angle,rtol=0,atol=1e-12)


def test_comparison_references_and_reported_bounds():
    coordinates = pd.read_csv(ROOT/'summary/coordinate_metrics.csv')
    velocity = pd.read_csv(ROOT/'summary/velocity_metrics_pooled.csv')
    interaction = pd.read_csv(ROOT/'summary/interaction_metrics.csv')
    for table in (coordinates,velocity,interaction):
        assert table.variant.nunique() == 7
        assert set(table[table.variant=='baseline_repeat'].reference) == {'accepted_baseline'}
        assert set(table[table.variant!='baseline_repeat'].reference) == {'baseline_repeat'}
    assert 100*coordinates.rigid_adjusted_rmsd_fraction_of_baseline_radius.max() <= 1.55
    perturbed = velocity[velocity.variant!='baseline_repeat']
    assert perturbed[perturbed.velocity_key=='full'].median_cosine_rigid_adjusted.min() >= .975
    assert perturbed[perturbed.velocity_key=='interaction'].median_cosine_rigid_adjusted.min() >= .877
    assert interaction[interaction.variant!='baseline_repeat'].attention_weighted_jaccard_union.min() >= .838


def test_complete_observation_set():
    table = pd.read_csv(ROOT/'summary/observation_order_check.csv')
    assert len(table) == 7 and table.same_obs_order.all()
    assert (table.n_obs == 3550).all()
    with np.load(ROOT/'summary/plot_inputs.npz',allow_pickle=False) as arrays:
        assert len(arrays['obs_names']) == 3550
        assert len(set(arrays['obs_names'])) == 3550
        for name in arrays.files:
            if name.endswith('_xy'):
                assert arrays[name].shape == (3550,2)
                assert np.isfinite(arrays[name]).all()
