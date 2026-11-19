"""Evaluate generator fields, inferred fields and all newly fitted baselines."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
import pandas as pd
import torch

from run_cytobridge import RUN, PACKAGE, TEACHER, SEEDS, load_model, seed_all, write_json
sys.path.insert(0,str(PACKAGE))


def fields(role, device):
    from CytoBridge.tl.downstream.simulation import compute_velocity_components
    from reproduction.agist.main_figure import _attention_edges
    frame = pd.read_csv(RUN/'data/mouse_brain_simulation.csv')
    values = frame.iloc[:,1:].to_numpy(np.float32)
    times = frame.samples.to_numpy(np.float32)
    loaded = load_model(TEACHER/'training' if role=='truth' else RUN/'cytobridge/model', device)
    model = loaded.model
    result={name:np.empty_like(values) for name in ['base','interaction','score','full']}
    growth=np.empty(len(values),dtype=np.float32)
    for t in [0,1,2,3]:
        rows=np.flatnonzero(times==t)
        seed_all(42+t)
        components=compute_velocity_components(values[rows],float(t),model,
            interaction_m=1024,device=device)
        for key in result:
            result[key][rows]=components['drift' if key=='base' else key]
        with torch.no_grad():
            growth[rows]=model.predict_growth(torch.tensor([float(t)],device=device),
                torch.tensor(values[rows],device=device)).cpu().numpy().ravel()
    seed_all(42)
    attention=_attention_edges(model,torch.tensor(values[times==0],device=device),0.,device)
    out=RUN/'evaluation'/role
    out.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(out/'fields.npz',**result,growth=growth,time=times)
    np.savez_compressed(out/'attention.npz',**attention)
    write_json(out/'complete.json',dict(model=str(loaded.weight_path),score=str(loaded.score_path),
        field_grouping_seed='42 + time',interaction_m=1024,
        attention_graph='all t0 cells',n_cells=len(frame)))


def draw():
    from scipy.stats import pearsonr,spearmanr
    from reproduction.agist.main_figure import (draw_observed_snapshots,
        calculate_velocity_display,draw_velocity_display,calculate_attention_display,
        draw_attention_display,draw_growth_correlation,normalize_growth)
    frame=pd.read_csv(RUN/'data/mouse_brain_simulation.csv')
    truth=dict(np.load(RUN/'evaluation/truth/fields.npz'))
    pred=dict(np.load(RUN/'evaluation/predicted/fields.npz'))
    out=RUN/'figures';out.mkdir(exist_ok=True)
    reference=RUN/'evaluation/generator_fields.npz'
    np.savez_compressed(reference,**{f'truth_{k}':truth[k] for k in ['base','interaction','score']},time=truth['time'])
    draw_observed_snapshots(frame,out)
    display=calculate_velocity_display(frame,pred,reference,RUN/'evaluation/velocity_display',n_jobs=8)
    for name,adata in display.items():
        adata.write_h5ad(RUN/f'evaluation/velocity_display/{name}.h5ad')
    draw_velocity_display(display,out)
    edges={r:dict(np.load(RUN/f'evaluation/{r}/attention.npz')) for r in ['truth','predicted']}
    xy=frame.loc[frame.samples==0,['x1','x2']].to_numpy()
    flow={r:calculate_attention_display(e,xy) for r,e in edges.items()}
    draw_attention_display(xy,flow['predicted'],flow['truth'],out)
    draw_growth_correlation(pred['growth'],truth['growth'],out)
    metrics={}
    for space,sl in [('physical',slice(0,2)),('gene',slice(2,52))]:
        a,b=pred['full'][:,sl],truth['full'][:,sl]
        denom=np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1)
        valid=denom>0
        cosine=np.divide(np.sum(a*b,axis=1),denom,out=np.full(len(a),np.nan),where=valid)
        metrics[f'{space}_cosine']=float(np.nanmean(cosine))
        metrics[f'{space}_undefined_cosine_count']=int((~valid).sum())
        frame[f'{space}_cosine']=cosine
    metrics['growth_pearson_raw']=float(pearsonr(pred['growth'],truth['growth']).statistic)
    metrics['growth_pearson_display']=float(pearsonr(normalize_growth(pred['growth']),normalize_growth(truth['growth'])).statistic)
    a,b=flow['predicted']['strength'],flow['truth']['strength']
    mask=(a!=0)|(b!=0)
    metrics['attention_outgoing_strength_spearman']=float(spearmanr(a[mask],b[mask]).statistic)
    metrics['attention_cells_nonzero_union']=int(mask.sum())
    frame[['samples','physical_cosine','gene_cosine']].to_csv(RUN/'evaluation/velocity_per_cell.csv',index=False)
    write_json(RUN/'evaluation/panel_metrics.json',metrics)
    print(json.dumps(metrics),flush=True)


def one_distance(task):
    from scripts.evaluate_and_plot_agist_w2_replicates import exact_w2
    method,seed,t,space=task
    truth=np.load(RUN/'data/generated_populations.npz')[f't{t}']
    path=(RUN/f'cytobridge/predictions/seed_{seed}.npz' if method=='CytoBridge'
        else RUN/f'stvcr/seed_{seed}.npz' if method=='stVCR' else RUN/'stories/prediction.npz')
    pred=np.load(path)[f't{t}']
    sl=slice(0,2) if space=='physical' else slice(2,52)
    ref=truth[:,sl]
    p=pred if method=='STORIES' else pred[:,sl]
    value=exact_w2(p,ref,np.ones(len(p)))
    return dict(method=method,seed=seed,time=t,space=space,w2=value,
        n_predicted=len(p),n_truth=len(ref))


def benchmark():
    from concurrent.futures import ProcessPoolExecutor
    for path in ['cytobridge/predictions/complete.json','stvcr/complete.json','stories/complete.json']:
        if not (RUN/path).exists():raise FileNotFoundError(RUN/path)
    tasks=[(m,s,t,space) for m in ['CytoBridge','stVCR'] for s in SEEDS for t in [1,2,3]
        for space in ['gene','physical']]+[('STORIES',42,t,'gene') for t in [1,2,3]]
    out=RUN/'benchmark';out.mkdir(exist_ok=True)
    rows=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for row in pool.map(one_distance,tasks):
            rows.append(row);pd.DataFrame(rows).to_csv(out/'w2_replicates.csv',index=False)
            print(row,flush=True)
    table=pd.DataFrame(rows)
    summary=table.groupby(['method','space','time']).w2.agg(['mean','std','count']).reset_index()
    summary.to_csv(out/'summary.csv',index=False)
    write_json(out/'complete.json',dict(n_distances=len(rows),
        protocol='Full-data reconstruction from exact same 5000 t0 cells',
        metric='sqrt of exact uniform empirical squared-Euclidean OT cost',
        stochastic_seeds=SEEDS,training_seed=42,stories='gene output only',
        old_baseline_values_used=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['truth','predicted','draw','benchmark']);p.add_argument('--device',default='cuda:3')
    a=p.parse_args()
    if a.stage in ['truth','predicted']:fields(a.stage,a.device)
    else:globals()[a.stage]()
