"""Train matched LARRY No-interaction models and evaluate the LR Full models."""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
BASE = Path('/data/cytobridge/projects/CytoBridge-ST-1104')
LR = BASE / 'runs/weinreb-cellchat-lr/20260718'
DATA = BASE / 'runs/weinreb-gnn-review/20260718/data/Weinreb_prepared_50pc.h5ad'
OLD_EVAL = BASE / 'runs/nonspatial-corrected-matched-e80c18a-20260815-r1'
PYTHON = BASE / 'envs/arista-api/bin/python'

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def write_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + '\n')
    tmp.replace(path)

def verify_energy():
    import torch
    from CytoBridge.tl.core.methods import ODEFunc
    class Dummy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.v = torch.nn.Parameter(torch.tensor([[.2, -.3], [.5, .6]]))
            self.q = torch.nn.Parameter(torch.tensor([[.4, .7], [-.3, .8]]))
            self.g = torch.nn.Parameter(torch.tensor([[.02], [-.04]]))
            self.interaction_net = type('Graph', (), {'cutoff': 1.0})()
        def predict_components(self, **kw):
            d = {'velocity': self.v, 'growth': self.g}
            if kw['include_score']:
                d.update(score=self.g * 2, score_gradient=self.q)
            if kw['include_interaction']:
                d['interaction'] = torch.zeros_like(self.v)
            return d
    model = Dummy()
    state = (torch.zeros(2,2), torch.tensor([[.1],[-.2]]), torch.zeros(2,1))
    for use_score in [False, True]:
        full = ODEFunc(model, use_mass=True, score_use=use_score, interaction_use=True)
        noi = ODEFunc(model, use_mass=True, score_use=use_score, interaction_use=False)
        ef = full(0.,state)[2]; en = noi(0.,state)[2]
        torch.testing.assert_close(ef, en)
        gf = torch.autograd.grad(ef.sum(), tuple(model.parameters()), allow_unused=True, retain_graph=True)
        gn = torch.autograd.grad(en.sum(), tuple(model.parameters()), allow_unused=True)
        for a,b in zip(gf,gn):
            if a is None or b is None:
                assert a is None and b is None
            else:
                torch.testing.assert_close(a,b)
    return {'energy_value_and_gradient_match': True, 'score_on_and_off_tested': True}

def worker(seed):
    import yaml
    import anndata as ad
    import numpy as np
    import pandas as pd
    import torch
    import CytoBridge as cb
    from CytoBridge.tl.downstream import load_dynamical_model_from_dir, evaluate_model_distributions
    torch.set_num_threads(4)
    out = ROOT / f'seed{seed}'
    out.mkdir(exist_ok=False)
    start = time.time()
    state = {'seed':seed, 'started_epoch':start, 'phase':'preflight'}
    def status(phase, **extras):
        state.update(phase=phase, updated_epoch=time.time(), **extras)
        write_json(out/'status.json',state)
    try:
        assert Path(cb.__file__).resolve().is_relative_to(ROOT/'runtime')
        reference = LR/f'runs/lr_seed{seed}_full'
        cfg = yaml.safe_load((reference/'resolved_config.yaml').read_text())
        assert cfg['seed']==seed and cfg['model']['interaction_net']['edge_mode']=='predictor'
        assert sha(DATA)=='b8f1f6f5394bdda9d4fbeb3d7beab13a4bae40c949b6f00f2f3f546e60d1784d'
        source_manifest = json.loads((reference/'run_manifest.json').read_text())
        for rel in ['CytoBridge/tl/core/models.py','CytoBridge/tl/graph/spatial_gnn.py']:
            assert sha(ROOT/'runtime'/rel)==source_manifest['training_implementation_files_sha256'][rel]
        checks = verify_energy()
        config = copy.deepcopy(cfg)
        config['ckpt_dir'] = str(out/'model')
        config['model']['components'].remove('interaction')
        config['model'].pop('interaction_net')
        config['model'].pop('interaction_type',None)
        config['model'].pop('interaction_group_size',None)
        for stage in config['training']['plan']:
            stage['train_strategy']='+'.join(x for x in stage['train_strategy'].split('+') if x!='i')
            if stage['name']=='Init_interaction':stage['name']='Matched_stage_3_no_interaction'
            if stage['name']=='Finetune':stage['name']='Finetune_no_interaction'
        assert [p['epochs'] for p in config['training']['plan']]==[100,100,50,2001,1000,2001]
        (out/'resolved_config.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
        manifest={'training_seed':seed,'full_model':str(reference/'model'),'prepared_h5ad':str(DATA),
          'prepared_sha256':sha(DATA),'reference_config_sha256':sha(reference/'resolved_config.yaml'),
          'energy':'historical LR Full velocity-score cross term, retained without interaction',
          'config_changes':['output path','remove interaction module','remove i training token','stage names'],
          'preflight':checks,'python':sys.executable,'torch':torch.__version__,
          'runtime_source_sha256':{str(p.relative_to(ROOT/'runtime')):sha(p) for p in sorted((ROOT/'runtime/CytoBridge').rglob('*.py'))},
          'distribution_seeds':list(range(10000,10050)),'clone_seeds':list(range(10)),
          'selection':'All training seeds 42,43,44 will be included regardless of results.'}
        write_json(out/'run_manifest.json',manifest)
        status('training')
        data=ad.read_h5ad(DATA)
        cb.tl.fit(data,config=config,device='cuda:0',time_key='time_point_processed',obsm_key='X_latent',
          is_spatial=False,ckpt_dir=out/'model',sigma=.1,evaluate_after_training=False)
        status('distribution_evaluation',training_seconds=time.time()-start)
        loaded=load_dynamical_model_from_dir(out/'model',dim=50,device='cuda:0',stage='Finetune_no_interaction')
        records=[]
        for index, inference_seed in enumerate(range(10000,10050)):
            result=evaluate_model_distributions(data,loaded.model,time_points=[0.,1.,2.],n_samples=2048,
              dt=.05,sigma=.1,include_score=True,interaction_m=16,max_ot_points=1024,
              structure_max_points=2048,device='cuda:0',time_key='time_point_processed',
              obsm_key='X_latent',concat_spatial=False,random_seed=inference_seed,noise_seed=inference_seed,
              include_initial_time=False,verbose=False)
            frame=result.metrics.copy();frame['training_seed']=seed;frame['inference_seed']=inference_seed
            records.append(frame)
            pd.concat(records,ignore_index=True).to_csv(out/'distribution_per_repeat.csv',index=False)
            status('distribution_evaluation',distribution_repeats_completed=index+1)
        del loaded
        torch.cuda.empty_cache()
        status('clone_evaluation')
        frozen=LR/f'reviews/20260719/reviewer_three_condition_corrected_v3/clone_fate_runs/clone_fate_seed{seed}_state_v2_10'
        subprocess.run([sys.executable,str(ROOT/'evaluate_clone.py'),'--prepared-h5ad',str(DATA),
          '--no-interaction-model-dir',str(out/'model'),
          '--historical-evaluator',str(ROOT/'runtime/scripts/evaluate_weinreb_clone_fate.py'),
          '--frozen-full-root',str(frozen),'--output-dir',str(out/'clone_fate'),'--device','cuda:0'],check=True)
        status('complete',elapsed_seconds=time.time()-start)
    except BaseException as error:
        status('failed',error=repr(error))
        raise

def launch():
    processes=[]
    for seed,gpu in zip([42,43,44],[0,1,2]):
        if (ROOT/f'seed{seed}').exists():
            raise FileExistsError(f'seed{seed} already exists. Do not start duplicate training.')
    for seed,gpu in zip([42,43,44],[0,1,2]):
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH=str(ROOT/'runtime'),
          PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',PYTHONHASHSEED=str(seed),
          OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
          MPLCONFIGDIR=str(ROOT/f'cache/seed{seed}/mpl'),NUMBA_CACHE_DIR=str(ROOT/f'cache/seed{seed}/numba'))
        log=(ROOT/f'seed{seed}.log').open('x')
        p=subprocess.Popen([str(PYTHON),'-u',str(Path(__file__)),'--seed',str(seed)],
          cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        log.close();processes.append({'seed':seed,'gpu':gpu,'pid':p.pid})
    write_json(ROOT/'launch.json',{'launched_epoch':time.time(),'jobs':processes})
    print(json.dumps(processes))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,choices=[42,43,44])
    parser.add_argument('--launch',action='store_true');args=parser.parse_args()
    if args.launch:launch()
    elif args.seed is not None:worker(args.seed)
    else:parser.error('Choose --launch or --seed')
