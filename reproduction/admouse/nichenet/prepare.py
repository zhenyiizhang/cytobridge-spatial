#!/usr/bin/env python3
"""Prepare CytoBridge-derived expression inputs for the official NicheNet R run.

This script does NOT calculate NicheNet scores. It only inverse-projects each
of the 51 model states into the same processed log1p gene space used for model
training and writes model-derived gene sets/backgrounds for official R calls.
"""
from pathlib import Path
import json
import argparse
import numpy as np
import pandas as pd
import anndata as ad

parser = argparse.ArgumentParser(description='Reconstruct expression and prepare the 50 NicheNet windows.')
parser.add_argument('--data-dir', type=Path, default=Path('data/admouse'))
parser.add_argument('--states', type=Path, help='Override the saved NicheNet slice_data directory.')
parser.add_argument('--output-dir', type=Path, required=True)
args = parser.parse_args()
OUT = args.output_dir.resolve()
REF = args.data_dir / 'aligned.h5ad'
STATES = args.states or args.data_dir / 'nichenet/slice_data'
if OUT == args.data_dir.resolve() or args.data_dir.resolve() in OUT.parents or STATES.resolve() in OUT.parents:
    parser.error('Choose an output directory outside the input data.')
for folder in ['data', 'results', 'provenance']:
    (OUT / folder).mkdir(parents=True, exist_ok=True)
TIMES=np.round(np.arange(0,2.5001,.05),2)
PCT=.10; EPS=1e-4

def tok(t): return f't{t:.2f}'.replace('.','p')
def age(t):
    if t<=1: return 2.5+3.2*t
    if t<=2: return 5.7+12.2*(t-1)
    return np.nan

ref=ad.read_h5ad(REF,backed='r')
pcs=np.asarray(ref.varm['PCs'],dtype=np.float32)
center=np.asarray(ref.var['pca_center'],dtype=np.float32)
genes=np.asarray(ref.var_names)
ref.file.close()
all_summary=[]; windows=[]
prev_micro=None; prev_time=None
for t in TIMES:
    s=ad.read_h5ad(STATES/f'time_{tok(t)[1:]}.h5ad')
    z=np.asarray(s.X[:,2:52],dtype=np.float32)
    expr=np.clip(z@pcs.T+center,0,None)
    ct=s.obs['major_annotation'].astype(str).to_numpy()
    for c in sorted(np.unique(ct)):
        x=expr[ct==c]
        all_summary.append(pd.DataFrame({'model_time':t,'age_months':age(t),'celltype':c,'gene':genes,
          'mean_log1p':x.mean(0),'pct_positive':(x>0).mean(0),'n_cells':x.shape[0]}))
    micro=expr[ct=='Microglia']
    if prev_micro is not None:
        fc=np.log2((np.expm1(micro.mean(0))+EPS)/(np.expm1(prev_micro.mean(0))+EPS))
        windows.append(pd.DataFrame({'window':f'{tok(prev_time)}_to_{tok(t)}','baseline_model_time':prev_time,
          'response_model_time':t,'baseline_age_months':age(prev_time),'response_age_months':age(t),
          'gene':genes,'log2fc_response_vs_baseline':fc}))
    prev_micro,prev_time=micro,t
summary=pd.concat(all_summary,ignore_index=True)
summary.to_csv(OUT/'data/model_expression_summary_51_states.csv',index=False)
effects=pd.concat(windows,ignore_index=True)
effects.to_csv(OUT/'data/microglia_adjacent_effect_sizes_50_windows.csv',index=False)

# Inputs retain every dynamic state-specific sender/receiver expression call.
# `pct_positive >= 0.10` is exactly the documented default threshold of
# NicheNet get_expressed_genes.default; the expression projection is external.
inputs=[]
for w,g in effects.groupby('window',sort=False):
    t=float(g.baseline_model_time.iloc[0]); base=summary[summary.model_time.eq(t)]
    receiver=set(base[(base.celltype=='Microglia')&(base.pct_positive>=PCT)].gene)
    send=base[(base.celltype!='Microglia')&(base.pct_positive>=PCT)][['celltype','gene']]
    up=g[g.log2fc_response_vs_baseline>0].nlargest(50,'log2fc_response_vs_baseline')
    inputs.append({'window':w,'baseline_model_time':t,'response_model_time':float(g.response_model_time.iloc[0]),
      'background_genes':';'.join(sorted(receiver)),'sender_genes':';'.join(sorted(set(send.gene))),
      'target_genes':';'.join(up.gene),'n_background':len(receiver),'n_targets':len(up)})
pd.DataFrame(inputs).to_csv(OUT/'data/official_nichenet_window_inputs.csv',index=False)
json.dump({'expression_space':'inverse-PCA reconstructed processed log1p','states':51,'windows':50,
 'threshold':'pct_positive >= 0.10 (official NicheNet default)','targets':'top 50 positive model-derived Microglia effect sizes per adjacent window',
 'note':'NicheNet prediction itself is performed only by official R functions in script 02.'},open(OUT/'provenance/input_contract.json','w'),indent=2)
