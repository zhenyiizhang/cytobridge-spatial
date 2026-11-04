"""Validate all formal folds before publishing a comparison table."""
from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
R=Path(__file__).resolve().parent
DATA={'zebrafish':[1,2,3],'mosta':[1,2],'arista':[1,2,3],'admouse':[1],'chicken_heart':[1,2]}
rows=[];problems=[];audits=[]
expected=pd.read_csv(R/'reference_projection_records.csv')
for method,spaces in [('paste2',{'joint','state','spatial'})]:
 for dataset,targets in DATA.items():
  for target in targets:
   key=f'{method}/{dataset}/t{target}';p=R/'results'/key
   try:
    run=json.loads((p/'run.json').read_text());assert run['status']=='complete',run['status']
    assert run['details']['selected_overlap']==1.0 and run['details']['unmapped_roster_count']==0
    cp=np.load(p/'coupling.npz',allow_pickle=False)['plan'];np.testing.assert_allclose(cp.sum(1),np.ones(cp.shape[0])/cp.shape[0],atol=1e-8,rtol=0);np.testing.assert_allclose(cp.sum(0),np.ones(cp.shape[1])/cp.shape[1],atol=1e-8,rtol=0)
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
report={'expected_runs':11,'verified_runs':len(audits),'problems':problems,'runs':audits}
(R/'validation.json').write_text(json.dumps(report,indent=2))
if problems:
 print(json.dumps({'verified':len(audits),'problems':problems},indent=2));raise SystemExit(1)
allrows=pd.concat(rows,ignore_index=True);allrows.to_csv(R/'metrics_all.csv',index=False)
means=allrows.groupby(['dataset','target','method','space']).sliced_w2.agg(sliced_w2='mean',projection_sd='std',n_projection_repeats='count').reset_index()
means['display_name']=means.method.replace({'PASTE2 (full matching)':'PASTE2 (full matching)','MIOFlow 2.0 (spatial SDE)':'MIOFlow 2.0'})
means.to_csv(R/'new_target_means.csv',index=False)
print('All 11 full-matching folds verified. No automatic replacement.')
