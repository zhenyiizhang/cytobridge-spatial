"""Validate all formal folds before publishing a comparison table."""
from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
R=Path(__file__).resolve().parent
DATA={'zebrafish':[1,2,3],'mosta':[1,2],'arista':[1,2,3],'admouse':[1],'chicken_heart':[1,2]}
rows=[];problems=[];audits=[]
expected=pd.read_csv(R/'reference_projection_records.csv')
for method,spaces in [('paste2',{'joint','state','spatial'}),('mioflow2',{'state'})]:
 for dataset,targets in DATA.items():
  for target in targets:
   key=f'{method}/{dataset}/t{target}';p=R/'results'/key
   try:
    run=json.loads((p/'run.json').read_text());assert run['status']=='complete',run['status']
    pred=np.load(p/'prediction.npz',allow_pickle=False)
    assert pred['state'].shape==(5000,50)
    assert np.isfinite(pred['state']).all()
    if method=='mioflow2':
     assert 'spatial' not in pred
     w=pred['weights'];assert np.isfinite(w).all() and np.all(w>=0) and np.isclose(w.sum(),1)
     loss=json.loads((p/'losses.json').read_text())
     def check_loss(x):
      if isinstance(x,dict):
       for y in x.values():check_loss(y)
      elif isinstance(x,list):
       for y in x:check_loss(y)
      elif isinstance(x,(float,int)):assert np.isfinite(x),'nonfinite training loss'
     check_loss(loss)
     assert run['settings']['epochs']==200 and run['settings']['ae_epochs']==100 and run['settings']['growth_epochs']==50
    else:
     assert pred['spatial'].shape==(5000,2) and np.isfinite(pred['spatial']).all()
    df=pd.read_csv(p/'metrics.csv');assert len(df)==5*len(spaces) and set(df.space)==spaces
    assert np.isfinite(df.sliced_w2).all() and (df.sliced_w2>=0).all()
    for space in spaces:
     d=df[df.space==space];assert set(d.projection_repeat)==set(range(5))
     ref=expected[(expected.dataset==dataset)&(expected.target==target)&(expected.space==space)]
     merged=d.merge(ref,on='projection_repeat',suffixes=('','_ref'))
     assert len(merged)==5 and (merged.projection_sha256==merged.projection_sha256_ref).all(),'projection mismatch'
    rows.append(df);audits.append({'run':key,'verified':True,'details':run['details']})
   except Exception as ex:problems.append({'run':key,'problem':str(ex)})
report={'expected_runs':22,'verified_runs':len(audits),'problems':problems,'runs':audits}
(R/'validation.json').write_text(json.dumps(report,indent=2))
if problems:
 print(json.dumps({'verified':len(audits),'problems':problems},indent=2));raise SystemExit(1)
allrows=pd.concat(rows,ignore_index=True);allrows.to_csv(R/'metrics_all.csv',index=False)
means=allrows.groupby(['dataset','target','method','space']).sliced_w2.agg(sliced_w2='mean',projection_sd='std',n_projection_repeats='count').reset_index()
means['display_name']=means.method.replace({'PASTE2 (PCA)':'PASTE2','MIOFlow 2.0 (spatial SDE)':'MIOFlow 2.0'})
means.to_csv(R/'new_target_means.csv',index=False)
old=pd.read_csv(R/'previous_target_means.csv')
combined=pd.concat([old,means],ignore_index=True);combined.to_csv(R/'all_target_means.csv',index=False)
cb=combined[combined.display_name=='CytoBridge'][['dataset','target','space','sliced_w2']].rename(columns={'sliced_w2':'cytobridge_sliced_w2'})
paired=means.merge(cb,on=['dataset','target','space'],validate='many_to_one')
paired['cytobridge_lower']=paired.cytobridge_sliced_w2<paired.sliced_w2
paired.to_csv(R/'new_paired_comparisons.csv',index=False)
winners=combined.loc[combined.groupby(['dataset','target','space']).sliced_w2.idxmin()]
winners.to_csv(R/'first_place_all_methods.csv',index=False)
print(paired.groupby('display_name').cytobridge_lower.agg(['sum','count']).to_string())
print('All 22 folds verified. Existing projection directions match.')
