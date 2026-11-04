"""Evaluate the three scNT model pairs used in current Supplementary Figure S6."""
from pathlib import Path
import json
import numpy as np
import pandas as pd


def evaluate(models_dir, prepared, raw_counts, pca, output, *, device='cuda:0'):
    from reproduction.nonspatial import analyze
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    distributions=[];directions=[]
    for seed in (42,43,44):
        model=Path(models_dir)
        full=model/f'full_seed{seed}/model'; no=model/f'no_interaction_seed{seed}/model'
        folder=output/f'seed{seed}';folder.mkdir(exist_ok=True)
        distribution=analyze.distribution(prepared,full,no,folder/'distribution',device=device,time_points=[0.,.25,.5,1.,2.])
        direction=analyze.direction(raw_counts,prepared,pca,full,no,folder/'direction',device=device)
        # The one-shot reference and fixed evaluation seed are identical across model pairs.
        d=pd.read_csv(distribution)
        d=d[d.time.isin([.25,.5,1.,2.])].copy()
        d['training_seed']=seed;distributions.append(d)
        v=pd.read_csv(direction);v['training_seed']=seed
        for arm,label in [('full','full_interaction_noise'),('no_interaction','no_interaction_noise')]:
            cells=pd.read_csv(direction.parent/f'{arm}_cellwise.csv.gz')
            fractions=cells.groupby('time_hours').cosine_inference_drift_vs_scnt.apply(lambda x:100*np.mean(x.dropna()>0))
            keep=v.condition.eq(label)
            v.loc[keep,'percentage_angle_lt_90']=v.loc[keep,'time_hours'].map(fractions)
        directions.append(v)
    pd.concat(distributions,ignore_index=True).to_csv(output/'distribution_all_models.csv',index=False)
    pd.concat(directions,ignore_index=True).to_csv(output/'direction_all_models.csv',index=False)
    return output


def update_inputs(summary, bundle):
    """Pass newly evaluated c/d values to the immediately following plot."""
    summary=Path(summary);bundle=Path(bundle)
    d=pd.read_csv(summary/'distribution_all_models.csv');v=pd.read_csv(summary/'direction_all_models.csv')
    v['condition']=v.condition.replace({'full':'full_interaction_noise','no_interaction':'no_interaction_noise'})
    if 'percentage_angle_lt_90' not in v:
        pct=pd.read_csv(summary/'scNT_timepoint_direction_percentages.csv')
        pct=pct[pct.cell_type.eq('ALL')].copy();pct['condition']=pct.condition.replace({'full':'full_interaction_noise','no_interaction':'no_interaction_noise'})
        v=v.merge(pct[['condition','training_seed','time_hours','percentage_angle_lt_90']],on=['condition','training_seed','time_hours'],validate='one_to_one')
    assert set(v.training_seed)=={42,43,44} and len(v)==24
    v.to_csv(bundle/'scnt_direction.csv',index=False)
    means=d.groupby(['time','condition'])[['w1','w2','tmv_absolute']].mean().unstack()
    result=pd.DataFrame({'time':means.index})
    for arm in ['full','no_interaction']:
        for metric in ['w1','w2','tmv_absolute']:result[f'{metric}_{arm}']=means[(metric,arm)].to_numpy()
    result.to_csv(bundle/'scnt_distribution.csv',index=False)
    m=json.loads((bundle/'scnt_metrics.json').read_text())
    m['conditions']={'full':'LR-informed Full','no_interaction':'No-interaction','training_seeds':[42,43,44]}
    a=v.groupby(['condition','training_seed'])[['cell_cosine_mean','cell_cosine_median']].mean().groupby('condition').mean()
    e=m['direction_evaluation'];e['training_seeds']=[42,43,44];e.pop('training_seed',None);e.pop('inferential_scope',None)
    e['equal_endpoint_average']={key:float(a.loc[arm,metric]) for key,arm,metric in [('mean_cellwise_cosine_full','full_interaction_noise','cell_cosine_mean'),('mean_cellwise_cosine_no_interaction','no_interaction_noise','cell_cosine_mean'),('median_cellwise_cosine_full','full_interaction_noise','cell_cosine_median'),('median_cellwise_cosine_no_interaction','no_interaction_noise','cell_cosine_median')]}
    b=v.groupby(['condition','time_hours'])[['cell_cosine_mean','cell_cosine_median']].mean()
    e['endpoint_wins_full']={'mean':int((b.loc['full_interaction_noise'].cell_cosine_mean>b.loc['no_interaction_noise'].cell_cosine_mean).sum()),'median':int((b.loc['full_interaction_noise'].cell_cosine_median>b.loc['no_interaction_noise'].cell_cosine_median).sum()),'out_of':4}
    (bundle/'scnt_metrics.json').write_text(json.dumps(m,indent=2)+'\n')
    manifest=json.loads((bundle/'manifest.json').read_text())
    for name,record in manifest['files'].items():record['bytes']=(bundle/name).stat().st_size
    manifest['conditions_by_dataset']['scnt']={'training_seeds':[42,43,44]}
    (bundle/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return bundle
