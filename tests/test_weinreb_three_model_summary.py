from pathlib import Path
import numpy as np
import pandas as pd
from reproduction.nonspatial.weinreb_matched import summarize


def test_raw_simulations_and_clone_probabilities_reproduce_s4(tmp_path):
    root=Path(__file__).resolve().parents[1]
    raw=root/'reproduction/nonspatial/larry_evaluation/raw'
    paths=summarize(raw,tmp_path/'summary')
    expected=root/'CytoBridge/results/data/nonspatial_figures'
    for name,path in paths.items():
        result=pd.read_csv(path)
        reference=pd.read_csv(expected/f'weinreb_{name}.csv')
        pd.testing.assert_frame_equal(result[reference.columns],reference,atol=1e-12,rtol=1e-12)
    per_seed=pd.read_csv(tmp_path/'summary/clone_metrics_per_training_seed.csv')
    assert set(per_seed.training_seed)=={42,43,44} and len(per_seed)==6
    # The reported score is calculated after averaging probabilities, not scores.
    averaged_scores=per_seed.query("model == 'LR Full'").clone_macro_tv_agreement.mean()
    displayed=pd.read_csv(paths['clone_fate']).set_index('metric').loc['tv_agreement','full']
    assert not np.isclose(averaged_scores,displayed)
