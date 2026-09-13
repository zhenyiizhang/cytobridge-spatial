"""The main-panel and supplementary lineage inputs share classifier labels."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUTS = ROOT / 'release_artifacts/lineage_classifier_predictions_20260913/mosta'

def test_cartilage_cohort_and_all_destinations():
    with np.load(INPUTS / 'cartilage_lineage.npz') as data:
        labels = data['target_labels'].astype(str)
    counts = pd.Series(labels).value_counts()
    assert len(labels) == 1464
    assert counts['Cartilage primordium'] == 941
    assert counts['Cartilage'] == 407
    assert counts['Connective tissue'] == 44
    assert counts.sum() == 1464
    reported = pd.read_csv(INPUTS / 'Figure4c_lineage_fractions.csv').set_index('cell_type')
    for label, count in counts.items():
        assert reported.loc[label, 'cells'] == count
        assert np.isclose(reported.loc[label, 'fraction'], count / 1464)

def test_s14_uses_the_same_source_cohort():
    labels = pd.read_csv(INPUTS / 'fixed_particle_labels.csv.gz')
    source = labels.loc[labels.time.eq(2.5) & labels.celltype.eq('Cartilage primordium')]
    assert len(source) == 1464
    target = labels.loc[labels.time.eq(3) & labels.particle_id.isin(source.particle_id)]
    expected = pd.read_csv(INPUTS / 'Figure4c_lineage_fractions.csv').set_index('cell_type')['cells']
    actual = target.celltype.value_counts()
    assert actual.to_dict() == expected.to_dict()
