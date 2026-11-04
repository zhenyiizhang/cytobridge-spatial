"""Summarize the three-model LARRY comparison."""
from pathlib import Path
import argparse
import importlib.util
import json
import sys
import time

ROOT=Path(__file__).resolve().parent
LR=Path('/data/cytobridge/projects/CytoBridge-ST-1104/runs/weinreb-cellchat-lr/20260718')
REVIEW=LR/'reviews/20260719/reviewer_three_condition_corrected_v3'

def summarize():
    import numpy as np
    import pandas as pd
    script=ROOT/'runtime/scripts/visualize_weinreb_clone_fate_probabilities.py'
    spec=importlib.util.spec_from_file_location('clone_probability_metrics',script)
    helpers=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=helpers
    spec.loader.exec_module(helpers)
    out=ROOT/'summary';out.mkdir(exist_ok=True)
    metrics=[];distributions=[];probabilities={'LR Full':[],'No-interaction':[]};observed=None
    for seed in [42,43,44]:
        fullroot=REVIEW/f'clone_fate_runs/clone_fate_seed{seed}_state_v2_10/full_interaction_noise/all_fates'
        noiroot=ROOT/f'seed{seed}/clone_fate'
        for name,root in [('LR Full',fullroot),('No-interaction',noiroot)]:
            truth=pd.read_csv(root/'observed_fate_distributions.csv',index_col=0)
            pred=pd.read_csv(root/'predicted_fate_distributions.csv',index_col=0)
            if observed is None: observed=truth
            assert truth.index.equals(observed.index) and truth.columns.equals(observed.columns)
            np.testing.assert_allclose(truth.to_numpy(),observed.to_numpy(),rtol=0,atol=1e-14)
            pred=pred.loc[observed.index,observed.columns]
            helpers._assert_probability_frame(pred,label=f'{name} seed{seed}')
            _,values=helpers._distribution_metrics(pred,observed)
            metrics.append({'training_seed':seed,'model':name,**values})
            probabilities[name].append(pred)
        full=pd.read_csv(REVIEW/f'paired_inference_repeats_10000_10049/seed{seed}/per_repeat_metrics_long.csv')
        full=full[full.condition.eq('lr_predictor_noise') & full.metric.isin(['w1','w2'])].copy()
        full['model']='LR Full'
        noi=pd.read_csv(ROOT/f'seed{seed}/distribution_per_repeat.csv').melt(
            id_vars=['training_seed','inference_seed','time','space'],value_vars=['w1','w2'],var_name='metric',value_name='value')
        noi['model']='No-interaction'
        columns=['training_seed','inference_seed','time','space','metric','model','value']
        assert len(full)==len(noi)==200
        assert set(full.inference_seed)==set(noi.inference_seed)==set(range(10000,10050))
        distributions.extend([full[columns],noi[columns]])
    pd.DataFrame(metrics).to_csv(out/'clone_metrics_per_training_seed.csv',index=False)
    ensemble=[]
    for name,frames in probabilities.items():
        avg=sum(frames)/3
        avg.to_csv(out/f'{name.replace(" ","_")}_ensemble_probabilities.csv')
        _,values=helpers._distribution_metrics(avg,observed)
        ensemble.append({'model':name,**values})
    pd.DataFrame(ensemble).to_csv(out/'clone_metrics_probability_ensemble.csv',index=False)
    all_dist=pd.concat(distributions,ignore_index=True)
    all_dist.to_csv(out/'distribution_all_repeats.csv',index=False)
    per_seed=all_dist.groupby(['training_seed','model','time','metric'],as_index=False).value.mean()
    per_seed.to_csv(out/'distribution_means_per_training_seed.csv',index=False)
    across=per_seed.groupby(['model','time','metric']).value.agg(['mean','std','count']).reset_index()
    across.to_csv(out/'distribution_across_training_seeds.csv',index=False)
    selected=['clone_macro_tv_agreement','clone_macro_js_similarity','clone_macro_dominant_fate_match']
    text='S4 LR-informed Full versus matched No-interaction\nTraining seeds: 42, 43 and 44.\n\n'
    text+='Clone metrics per training seed (higher is better):\n'+pd.DataFrame(metrics)[['training_seed','model']+selected].to_string(index=False)+'\n\n'
    text+='Probability ensemble: average each clone probability vector across the same three models in each arm, then score.\n'
    text+=pd.DataFrame(ensemble)[['model']+selected].to_string(index=False)+'\n\n'
    text+='Distribution metrics: mean of 50 inference repeats within each seed, equal weight across training seeds (lower is better).\n'+across.to_string(index=False)+'\n'
    (out/'RESULTS.txt').write_text(text)
    (ROOT/'summary_status.json').write_text(json.dumps({'status':'complete','time':time.time(),'result':str(out/'RESULTS.txt')},indent=2))
    print(text)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--wait',action='store_true');args=parser.parse_args()
    deadline=time.time()+12*3600
    try:
        while True:
            states=[json.loads((ROOT/f'seed{s}/status.json').read_text()) for s in [42,43,44]]
            if any(s['phase']=='failed' for s in states):raise RuntimeError(str(states))
            if all(s['phase']=='complete' for s in states):break
            if not args.wait or time.time()>deadline:raise RuntimeError('Training/evaluation not complete')
            time.sleep(30)
        summarize()
    except BaseException as error:
        (ROOT/'summary_status.json').write_text(json.dumps({'status':'failed','time':time.time(),'error':repr(error)},indent=2))
        raise
