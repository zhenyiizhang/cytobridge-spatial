"""PASTE2 and MIOFlow 2.0 comparisons on the existing five-dataset held-out inputs."""
from pathlib import Path
import os,sys,json,time,argparse,hashlib,random,traceback
ROOT=Path(__file__).resolve().parent
PROJECT=Path('/data/cytobridge/projects/CytoBridge-ST-1104')
CODE=Path(__file__).resolve().parents[4]
sys.path[:0]=[str(ROOT.parent/'official_sources/MIOFlow'),str(ROOT.parent/'official_sources/paste2/src'),str(CODE)]
import numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
from scripts.spatiotemporal_benchmark.static_baselines.data import _ranked_indices

def save(p,x):
 tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,default=str));tmp.replace(p)
def read(p):
 with np.load(p,allow_pickle=False) as f:return {k:f[k] for k in f.files}
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def arrays(args):
 base=PROJECT/'runs'/('rev03-arista-heart-all-method-loto-20260826-r1' if args.dataset in ['arista','chicken_heart'] else 'corrected-benchmark-20260813-matched-3c87a3e-c4f8e203-ff55c7a-r4')
 split=(args.input_root/args.dataset/f'loto_t{args.target}') if args.input_root else base/args.dataset/'inputs'/f'loto_t{args.target}'
 train=read(split/'training_reference.npz');roster=read(split/'source_roster.npz');manifest=json.loads((split/'manifest.json').read_text())
 times=sorted(np.unique(train['time']).tolist());assert args.target not in times
 support={}
 for t in times:
  ix=np.flatnonzero(train['time']==t);support[t]=ix[_ranked_indices(train['row_id'][ix],800,int(manifest['source_roster_seed']),t)]
 left=max(t for t in times if t<args.target);right=min(t for t in times if t>args.target)
 lookup={k:i for i,k in enumerate(train['row_id'])};positions=np.array([lookup[x] for x in roster['row_id']]);np.testing.assert_array_equal(train['state'][positions],roster['state'])
 assert len(positions)==5000
 return split,train,roster,times,support,left,right,positions

def paste(args,out,train,roster,times,support,left,right,positions):
 import anndata as ad
 from scipy.spatial.distance import cdist
 from paste2.PASTE2 import partial_pairwise_align_given_cost_matrix
 from paste2.model_selection import calculate_convex_hull_edge_inconsistency
 # Record the official solver's existing loss history without changing its numerical settings.
 import importlib
 solver_module=importlib.import_module('paste2.PASTE2')
 original_solver=solver_module.partial_fused_gromov_wasserstein
 solver_records=[]
 def audited_solver(*pos,**kw):
  kw['numItermax']=200
  result=original_solver(*pos,**kw)
  assert kw.get('log') is True
  plan,info=result
  losses=info.get('loss',[])
  record={'overlap':float(kw['m']),'iterations':len(losses),'iteration_limit':kw.get('numItermax',1000),'final_objective':float(info['partial_fgw_cost'])}
  if len(losses)>1:
   delta=float(abs(losses[-1]-losses[-2]));record['last_objective_change']=delta
   record['objective_tolerance_met']=bool(delta<1e-9 or (abs(losses[-1])>0 and delta/abs(losses[-1])<1e-9))
  record['reached_iteration_cap']=bool(len(losses)>=200)
  if record['reached_iteration_cap']:
   audit_kw=dict(kw);audit_kw['numItermax']=1000
   check_plan,check_info=original_solver(*pos,**audit_kw)
   record['long_budget_iterations']=len(check_info.get('loss',[]))
   record['long_budget_objective']=float(check_info['partial_fgw_cost'])
   record['long_budget_plan_max_difference']=float(np.max(np.abs(check_plan-plan)))
   np.savez_compressed(out/f'convergence_check_{len(solver_records):02d}.npz',budget200=plan,budget1000=check_plan)
  solver_records.append(record);save(out/'solver_iterations.json',solver_records)
  return result
 solver_module.partial_fused_gromov_wasserstein=audited_solver
 idx1,idx2=support[left],support[right]
 slices=[]
 for ix in (idx1,idx2):
  a=ad.AnnData(train['state'][ix].astype(float));a.obsm['spatial']=train['spatial'][ix].astype(float);slices.append(a)
 a,b=slices
 import ot
 M=ot.dist(a.X,b.X,metric='sqeuclidean')
 # Prespecified full-overlap sensitivity analysis. No overlap search.
 selected=1.0
 # Respect floating-point sums of the uniform marginals in the upstream solver.
 feasible_selected=min(selected,float((np.ones(len(a))/len(a)).sum()),float((np.ones(len(b))/len(b)).sum()))
 plan=partial_pairwise_align_given_cost_matrix(a,b,M=M,s=feasible_selected,alpha=.1,norm=False,G_init=np.outer(np.ones(len(a))/len(a),np.ones(len(b))/len(b))*feasible_selected)
 assert np.isfinite(plan).all() and plan.min()>-1e-10
 np.testing.assert_allclose(plan.sum(1),np.ones(len(a))/len(a),atol=1e-8,rtol=0)
 np.testing.assert_allclose(plan.sum(0),np.ones(len(b))/len(b),atol=1e-8,rtol=0)
 save(out/'overlap_selection.json',{'policy':'fixed full matching before comparison','requested_overlap':1.0,'represented_overlap':feasible_selected})
 rows=plan.sum(1);assert np.isclose(plan.sum(),selected,atol=1e-5),(plan.sum(),selected)
 joint=np.concatenate([train['spatial'],train['state']],1).astype(float);prediction=joint[idx1].copy();active=rows>1e-12
 bary=plan[active]@joint[idx2]/rows[active,None]
 fraction=(args.target-left)/(right-left);prediction[active]=(1-fraction)*joint[idx1][active]+fraction*bary
 # Unmapped source cells have no target assignment, so retain their source state.
 look={k:i for i,k in enumerate(train['row_id'][idx1])};local=np.array([look[k] for k in roster['row_id']]);prediction=prediction[local]
 d=train['spatial'].shape[1];np.savez_compressed(out/'prediction.npz',state=prediction[:,d:],spatial=prediction[:,:d],row_id=roster['row_id'])
 np.savez_compressed(out/'coupling.npz',plan=plan,left_row_id=train['row_id'][idx1],right_row_id=train['row_id'][idx2])
 return {'expression_cost':'squared Euclidean, matching PASTE implementation','solver_iteration_limit':200,'distance_normalization':False,'max_observed_solver_iterations':max(x['iterations'] for x in solver_records),'comparison_role':'full matching sensitivity; retain partial matching results','selected_overlap':selected,'unmapped_support_count':int((~active).sum()),'unmapped_roster_count':int((~active)[local].sum()),'overlap_rule':'fixed s=1, no overlap selection','partial_initial_plan':'feasible total mass s','unmapped_prediction':'retain source state','interpolation_fraction':fraction}

def mio(args,out,train,roster,times,support,left,right,positions):
 import torch,torchsde,anndata as ad,pickle
 from mioflow import compute_spatial_features,fit_joint_gaga,MIOFlow,GrowthRateModel
 from mioflow.core.datasets import TimeSeriesDataset
 torch.set_num_threads(4);torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
 torch.cuda.manual_seed_all(args.seed)
 a=ad.AnnData(train['state'].astype(np.float32));a.obs['stage']=train['time'];a.obs['celltype']=train['annotation'].astype(str);a.obsm['spatial']=train['spatial'];a.obsm['X_pca']=train['state']
 print('Computing spatial neighborhood features on training stages',flush=True)
 compute_spatial_features(a,k=5,n_hops=3,d_max=None,celltype_key='celltype',batch_key='stage',n_pca_niche=50,n_spatial_pca=50)
 selection=np.concatenate([support[t] for t in times]);subset=a[selection].copy()
 gaga=fit_joint_gaga(subset,gene_key='X_pca',spatial_scale=.2,latent_dim=4,hidden_dims=[64,32],batch_size=512,encoder_epochs=args.ae_epochs,decoder_epochs=args.ae_epochs,device='cuda')
 gaga.cpu().eval()
 gs,ss=gaga.joint_scalers
 a.obsm['X_joint']=np.hstack([gs.transform(a.obsm['X_pca']),.2*ss.transform(a.obsm['X_spatial'])]).astype(np.float32)
 growth=GrowthRateModel(subset,gaga_model=gaga,gaga_input_key='X_joint',obs_time_key='stage',use_cuda=True)
 # The package normally reindexes observed stages 0,1,... . Preserve benchmark
 # time gaps when a stage is held out, for both flow and growth.
 growth.dataset=TimeSeriesDataset([(x,float(t)) for (x,_),t in zip(growth.dataset.time_series_data,times)])
 print('Pretraining growth model',flush=True);growth.pretrain(num_epochs=args.growth_epochs)
 gaga.cpu().eval()
 mf=MIOFlow(subset,gaga_model=gaga,gaga_input_key='X_joint',obs_time_key='stage',use_cuda=True,use_sde=True,growth_rate_model=growth,n_epochs=args.epochs,use_density_loss=True,sample_size=512,n_trajectories=100,n_bins=20,exp_dir=str(out/'training'),energy_time_steps=20,sde_dt=.1)
 mf.dataset=TimeSeriesDataset([(x,float(t)) for (x,_),t in zip(mf.dataset.time_series_data,times)])
 # Skip the package's unused post-fit preview (it moves time tensors to CPU).
 # The benchmark prediction below uses the trained model and CUDA time tensors.
 mf._generate_trajectories = lambda dataset: None
 print('Training MIOFlow SDE',flush=True);mf.fit();save(out/'losses.json',mf.losses)
 gaga.cpu().eval()
 # Encode the existing starting roster, integrate only from its observed anchor.
 source=gaga.input_scaler.transform(a.obsm['X_joint'][positions]).astype(np.float32)
 with torch.no_grad():z=gaga.encode(torch.from_numpy(source)).numpy()
 z=((z-mf.mean_vals)/mf.std_vals).astype(np.float32)
 start=torch.from_numpy(z).cuda();mf.ode_model.eval();mf.ode_model.reset_momentum()
 with torch.no_grad():
  pred=torchsde.sdeint(mf.ode_model,start,torch.tensor([float(left),float(args.target)],device='cuda'),dt=.1,method='euler')[-1]
  masses=growth(start,torch.tensor(float(left),device='cuda')).cpu().numpy()
  latent=pred.cpu().numpy()*mf.std_vals+mf.mean_vals
  decoded=gaga.decode(torch.tensor(latent,dtype=torch.float32)).numpy()
 joint=gaga.input_scaler.inverse_transform(decoded);state=gs.inverse_transform(joint[:,:train['state'].shape[1]])
 weights=np.maximum(masses,1e-12)**((args.target-left)/(right-left));weights/=weights.sum()
 assert np.isfinite(state).all() and np.isfinite(weights).all()
 np.savez_compressed(out/'prediction.npz',state=state,row_id=roster['row_id'],weights=weights)
 torch.save({'flow':mf.ode_model.state_dict(),'growth':growth.state_dict(),'gaga':gaga.state_dict(),'mean':mf.mean_vals,'std':mf.std_vals},out/'checkpoint.pt')
 with (out/'transforms.pkl').open('wb') as f:pickle.dump({'joint_scalers':gaga.joint_scalers,'gaga_scaler':gaga.input_scaler,'fit_row_id':train['row_id'][selection]},f)
 np.savez_compressed(out/'training_features.npz',X_joint=subset.obsm['X_joint'],row_id=train['row_id'][selection],time=train['time'][selection])
 return {'features':'shared PCA; local expression niche; neighborhood cell-type composition','lr_features':False,'spatial_output':False,'use_sde':True,'growth':'official UOT-pretrained model used by official flow loss','latent_dim':4,'gaga_cells':len(selection),'gaga_epochs_per_phase':args.ae_epochs,'growth_epochs':args.growth_epochs,'flow_epochs':args.epochs,'all_train_cells_for_spatial_features':len(a),'time_coordinates':times,'mass_interpolation':'source-anchor UOT relative mass raised to fractional anchor interval, then normalized','gene_output':'inverse GAGA decoder and training-only scalers'}

def evaluate(args,out,split,train):
 from CytoBridge.tl.downstream.benchmark import fit_frozen_benchmark_transform,evaluate_spatiotemporal_prediction
 pred=read(out/'prediction.npz');truth=read(split/f'truth_t{args.target}.npz');transform=fit_frozen_benchmark_transform(train['state'],train['spatial']);(out/'evaluation_transform.json').write_text(transform.to_json())
 kw={'predicted_spatial':pred['spatial'],'observed_spatial':truth['spatial']} if 'spatial' in pred else {}
 if 'weights' in pred:kw['predicted_weights']=pred['weights']
 metrics=evaluate_spatiotemporal_prediction(transform=transform,benchmark=args.dataset,split=f'loto_t{args.target}',method={'paste2':'PASTE2 (full matching)','mioflow2':'MIOFlow 2.0 (spatial SDE)'}[args.method],predicted_state=pred['state'],observed_state=truth['state'],n_projections=1024,projection_repeats=5,max_ot_points=800,**kw)
 metrics['dataset']=args.dataset;metrics['target']=args.target;metrics.to_csv(out/'metrics.csv',index=False)

def main():
 p=argparse.ArgumentParser();p.add_argument('--input-root',type=Path,required=True);p.add_argument('--method',choices=['paste2','mioflow2'],required=True);p.add_argument('--dataset',required=True);p.add_argument('--target',type=int,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,default=20261031);p.add_argument('--epochs',type=int,default=200);p.add_argument('--ae-epochs',type=int,default=100);p.add_argument('--growth-epochs',type=int,default=50);p.add_argument('--skip-evaluation',action='store_true');args=p.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=False)
 status={'method':args.method,'dataset':args.dataset,'target':args.target,'status':'preparing','started':time.time(),'settings':vars(args),'source_versions':{'paste2':'bundled official source','mioflow2':'bundled official source'}}
 save(out/'run.json',status)
 try:
  with threadpool_limits(limits=4):
   split,train,roster,times,support,left,right,positions=arrays(args);status.update(input_root=str(split),input_sha256=digest(split/'training_reference.npz'),status='fitting');save(out/'run.json',status)
   details=(paste if args.method=='paste2' else mio)(args,out,train,roster,times,support,left,right,positions)
   status.update(details=details,status='prediction_saved');save(out/'run.json',status)
   if not args.skip_evaluation:evaluate(args,out,split,train)
  status.update(status='complete',finished=time.time());save(out/'run.json',status)
 except BaseException as ex:
  status.update(status='failed',error=str(ex),traceback=traceback.format_exc());save(out/'run.json',status);raise
if __name__=='__main__':main()
