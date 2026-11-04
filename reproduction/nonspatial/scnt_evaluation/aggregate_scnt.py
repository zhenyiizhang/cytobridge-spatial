"""Summarize all prespecified models; never select runs by their results."""
from pathlib import Path
import json
import os
import tempfile
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SEEDS = [42, 43, 44]
ARMS = ['full', 'no_interaction']

def main():
    distribution, direction, manifests = [], [], []
    for arm in ARMS:
        for seed in SEEDS:
            folder = ROOT/f'{arm}_seed{seed}'
            state = json.loads((folder/'status.json').read_text())
            if state['phase'] != 'complete':
                raise RuntimeError(f'{arm} seed {seed} has not completed evaluation')
            ev = folder/'evaluation'
            manifest = json.loads((ev/'manifest.json').read_text())
            assert manifest['training_seed'] == seed and manifest['condition'] == arm
            assert manifest['new_rna_used_for_training'] is False
            manifests.append(manifest)
            dist = pd.read_csv(ev/'distribution.csv')
            dirs = pd.read_csv(ev/'direction_timewise.csv')
            assert dist.time.tolist() == [.25,.5,1.,2.]
            assert dirs.time_hours.tolist() == [.25,.5,1.,2.]
            assert dist.training_seed.eq(seed).all() and dirs.training_seed.eq(seed).all()
            assert dist.condition.eq(arm).all() and dirs.condition.eq(arm).all()
            assert np.isfinite(dist[['w1','w2']]).all().all()
            assert np.isfinite(dirs[['cell_cosine_mean','cell_cosine_median']]).all().all()
            distribution.append(dist)
            direction.append(dirs)
    assert len({m['prepared_sha256'] for m in manifests}) == 1
    assert len({m['weight_sha256'] for m in manifests}) == 6
    target = ROOT/'summary'
    temp = Path(tempfile.mkdtemp(prefix='.summary-', dir=ROOT))
    d = pd.concat(distribution, ignore_index=True)
    v = pd.concat(direction, ignore_index=True)
    d.to_csv(temp/'distribution_all_models.csv', index=False)
    v.to_csv(temp/'direction_all_models.csv', index=False)

    rows = []
    for table, timecol, metrics in [(d,'time',['w1','w2']),
            (v,'time_hours',['cell_cosine_mean','cell_cosine_median'])]:
        for (arm,t), part in table.groupby(['condition',timecol], sort=True):
            assert sorted(part.training_seed) == SEEDS
            for metric in metrics:
                rows.append(dict(condition=arm, time_hours=t, metric=metric,
                    mean=part[metric].mean(), sd=part[metric].std(ddof=1),
                    n_training_runs=3))
    means = pd.DataFrame(rows)
    means.to_csv(temp/'timewise_mean_and_sd.csv', index=False)
    comparisons = means.pivot(index=['time_hours','metric'], columns='condition', values='mean')
    comparisons['no_interaction_minus_full'] = comparisons.no_interaction-comparisons.full
    comparisons.to_csv(temp/'timewise_comparisons.csv')
    global_direction = v.groupby(['condition','training_seed'])[
        ['cell_cosine_mean','cell_cosine_median']].mean().reset_index()
    global_direction.to_csv(temp/'direction_equal_endpoint_weight_per_model.csv', index=False)
    summary = dict(status='complete', training_seeds=SEEDS, models_per_condition=3,
        aggregation='equal weight per independently trained model; equal endpoint weight for overall direction',
        inference_repeats_per_model=1, evaluation_seed=42,
        full_lower_mean_w1_endpoints=int((comparisons.xs('w1',level='metric').no_interaction_minus_full>0).sum()),
        full_lower_mean_w2_endpoints=int((comparisons.xs('w2',level='metric').no_interaction_minus_full>0).sum()),
        full_higher_direction_mean_endpoints=int((comparisons.xs('cell_cosine_mean',level='metric').no_interaction_minus_full<0).sum()),
        full_higher_direction_median_endpoints=int((comparisons.xs('cell_cosine_median',level='metric').no_interaction_minus_full<0).sum()),
        manifests=manifests)
    (temp/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    if target.exists():
        raise FileExistsError('Existing summary is preserved. Review before replacing.')
    temp.rename(target)
    print(json.dumps({k:v for k,v in summary.items() if k!='manifests'},indent=2))

if __name__ == '__main__':
    main()
