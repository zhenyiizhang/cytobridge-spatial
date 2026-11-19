"""Update S4 plot inputs from all three LR Full / No-interaction models.

Run from the repository root:
python reproduction/nonspatial/update_weinreb_s4.py
The input tables retain all predefined training and inference seeds.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def update(summary, bundle):
    summary, bundle = Path(summary), Path(bundle)
    raw = pd.read_csv(summary / 'distribution_all_repeats.csv')
    assert set(raw.training_seed) == {42, 43, 44}
    assert set(raw.model) == {'LR Full', 'No-interaction'}
    assert not raw.duplicated(['training_seed','inference_seed','model','time','metric']).any()
    for _, block in raw.groupby(['training_seed','model','time','metric']):
        assert len(block) == 50 and set(block.inference_seed) == set(range(10000,10050))
    means = raw.groupby(['training_seed','model','time','metric']).value.mean()
    rows = []
    for time in (1., 2.):
        row = {'time': time, 'space': 'pca'}
        for metric in ('w1','w2'):
            for name, arm in [('LR Full','full'),('No-interaction','no_interaction')]:
                values = [float(means.loc[seed,name,time,metric]) for seed in (42,43,44)]
                row[f'{metric}_{arm}'] = np.mean(values)
                row.update({f'{metric}_{arm}_seed{seed}': value for seed,value in zip((42,43,44),values)})
            row[f'{metric}_relative_change'] = row[f'{metric}_no_interaction']/row[f'{metric}_full']-1
        rows.append(row)
    distribution = pd.DataFrame(rows)
    distribution.to_csv(bundle/'weinreb_distribution.csv', index=False)
    ensemble = pd.read_csv(summary/'clone_metrics_probability_ensemble.csv').set_index('model')
    rows = []
    for short in ('tv_agreement','js_similarity','dominant_fate_match'):
        full, no = [float(ensemble.loc[name,'clone_macro_'+short]) for name in ('LR Full','No-interaction')]
        rows.append(dict(metric=short, full=full, no_interaction=no,
                         delta_no_interaction_minus_full=no-full, relative_change=no/full-1, higher_is_better=True))
    pd.DataFrame(rows).to_csv(bundle/'weinreb_clone_fate.csv', index=False)
    path = bundle/'weinreb_metrics.json'
    metrics = json.loads(path.read_text())
    metrics['distribution'] = dict(continuous_from_t0=True, training_seeds=[42,43,44],
        simulations_per_model=50, aggregation='mean within model, then equal mean across models',
        equal_weight_W1_W2_error_increase_after_removal_pct=float(100*np.mean(
            distribution[['w1_relative_change','w2_relative_change']].to_numpy())))
    metrics['clone_fate'] = dict(all_source_cells=4638, evaluated_lineages=1237,
        training_seeds=[42,43,44], simulation_seeds=list(range(10)),
        aggregation='average clone probabilities across models before scoring',
        metrics={r['metric']:{k:r[k] for k in ('full','no_interaction','relative_change')} for r in rows})
    metrics['model_field']['training_seeds'] = [42,43,44]
    path.write_text(json.dumps(metrics,indent=2)+'\n')
    path = bundle/'manifest.json'
    manifest = json.loads(path.read_text())
    old = manifest.pop('conditions',None)
    manifest['conditions_by_dataset'] = {
        'weinreb':dict(training_seeds=[42,43,44], with_interaction='LR-informed Full',
                       without_interaction='retrained without interaction using the Full energy expression'),
        'scnt': old if old else manifest['conditions_by_dataset']['scnt']}
    for name, record in manifest['files'].items():
        record['bytes'] = (bundle/name).stat().st_size
    path.write_text(json.dumps(manifest,indent=2)+'\n')
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary-dir',type=Path,default=ROOT/'reproduction/nonspatial/larry_evaluation/summary')
    parser.add_argument('--bundle-dir',type=Path,default=ROOT/'CytoBridge/results/data/nonspatial_figures')
    args = parser.parse_args()
    print(json.dumps(update(args.summary_dir,args.bundle_dir)['distribution'],indent=2))
