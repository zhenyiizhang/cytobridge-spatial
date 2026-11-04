"""Fit the official stVCR/STORIES implementations on the new Figure 2 data."""
import argparse
import json
import os
import pickle
import random
import subprocess
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

BASE = Path('/data/cytobridge/projects/CytoBridge-ST-1104')
RUN = Path(os.environ.get('CYTOBRIDGE_FIGURE2_RUN', str(Path(__file__).resolve().parent)))
SEEDS = [1,4,8,32,256,512,1024,2048,4096,8192]
STVCR = Path(os.environ.get('STVCR_SOURCE', str(BASE / 'runs/zebrafish-benchmark/corrected-alpha0015-loto-full-v1/software/clean-snapshots/stvcr-26aa79a-independent-upload')))
STORIES = Path(os.environ.get('STORIES_SOURCE', str(BASE / 'runs/arista-benchmark/corrected-alpha0015-v1/software/stories')))
sys.path.insert(0, os.environ.get('CYTOBRIDGE_CODE', str(RUN / 'code')))


def data():
    frame = pd.read_csv(RUN / 'data/mouse_brain_simulation.csv')
    values = frame[[f'x{i}' for i in range(1,53)]].to_numpy(np.float32)
    a = ad.AnnData(values[:,2:])
    a.obs['time'] = frame.samples.to_numpy(np.float32)
    a.obs['time_input'] = frame.samples.to_numpy(np.float32)
    a.obsm['spatial'] = values[:,:2]
    a.obsm['X_spatial_input'] = values[:,:2]
    a.obsm['X_gene_input'] = values[:,2:]
    a.obsm['X_stories'] = values[:,2:]
    return a


def source_record(path):
    return dict(path=str(path), commit=subprocess.check_output(
        ['git','-C',str(path),'rev-parse','HEAD'], text=True).strip())


def stvcr(device, predict_only=False, seeds=None):
    if predict_only and seeds is None:
        from concurrent.futures import ThreadPoolExecutor
        groups = [SEEDS[i::4] for i in range(4)]
        def group_prediction(group):
            subprocess.run([sys.executable, '-u', str(Path(__file__).resolve()),
                'stvcr', '--predict-only', '--seeds', *map(str, group)], check=True)
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(group_prediction, groups))
        record = json.loads((RUN/'stvcr/training_complete.json').read_text())
        record['prediction_seeds'] = SEEDS
        (RUN/'stvcr/complete.json').write_text(json.dumps(record, indent=2))
        return
    import torch
    from scripts.spatiotemporal_benchmark.dynamic.run_dynamic import _import_stvcr
    _, defaults, train, predict = _import_stvcr(STVCR)
    output = RUN / 'stvcr'
    output.mkdir(exist_ok=predict_only)
    a = data()
    random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    config = dict(defaults)
    config.update(n_epochs=2000, num_samples=500)
    # Match the existing formal five-dataset baseline training budget.
    record = dict(method='stVCR', source=source_record(STVCR), epochs=2000,
        num_samples=500, training_seed=42, growth=True, alignment=False,
        prediction_dt=.01, input=str(RUN/'data/mouse_brain_simulation.csv'))
    if not predict_only:
        (output/'started.json').write_text(json.dumps(record,indent=2))
        train(a, model_path=str(output/'model.pt'),
            rigid_transformation_path=str(output/'rigid_transform.pt'), config=config,
            use_alignment=False, use_growth=True, device=torch.device(device))
        a.write_h5ad(output/'fitted_input.h5ad')
        (output/'training_complete.json').write_text(json.dumps(record,indent=2))
    # The official simulator creates CPU tensors and calls .numpy() directly.
    model = torch.load(output/'model.pt', map_location='cpu', weights_only=False).eval()
    torch.set_num_threads(8)
    initial = np.load(RUN/'data/initial_states.npy')
    # One continuous official trajectory supplies all three target stages.
    for seed in (SEEDS if seeds is None else seeds):
        if (output/f'seed_{seed}.npz').exists():
            continue
        result = {'t0':initial}
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        with torch.no_grad():
            output_series = predict(torch.from_numpy(initial), model, 0., 3.,
                spatial_dim=2, delta_t=.01, other_time_points=[1.,2.,3.])
        for t in [1,2,3]:
            index=int(np.flatnonzero(np.isclose(output_series[4],float(t)))[0])
            def array(v):
                return v.detach().cpu().numpy() if hasattr(v,'detach') else np.asarray(v)
            x = np.concatenate([array(output_series[0][index]), array(output_series[1][index])],axis=1).astype(np.float32)
            assert len(x) and x.shape[1] == 52 and np.isfinite(x).all()
            result[f't{t}'] = x
        np.savez_compressed(output/f'seed_{seed}.npz', **result)
        print('stVCR trajectory saved',seed,flush=True)
        del output_series
    record['prediction_seeds'] = SEEDS if seeds is None else seeds
    marker = 'complete.json' if seeds is None else f'complete_seeds_{seeds[0]}.json'
    (output/marker).write_text(json.dumps(record,indent=2))


def stories(device):
    sys.path.insert(0,str(STORIES))
    import stories as module
    import jax
    a = data()
    output = RUN / 'stories'
    output.mkdir(exist_ok=False)
    model = module.SpaceTime(proximal_step=module.steps.ExplicitStep())
    record = dict(method='STORIES', source=source_record(STORIES), max_iter=100,
        batch_size=128, training_seed=42, output_space='gene',
        input=str(RUN/'data/mouse_brain_simulation.csv'), devices=[str(d) for d in jax.devices()])
    (output/'started.json').write_text(json.dumps(record,indent=2))
    model.fit(a, time_key='time', omics_key='X_stories', space_key='spatial',
        batch_size=128, max_iter=100, checkpoint_manager=str(output/'checkpoints'),
        key=jax.random.PRNGKey(42), restore=False)
    with (output/'model.pkl').open('wb') as f:
        pickle.dump(model,f)
    pd.DataFrame({'loss':getattr(model,'train_losses',[])}).to_csv(output/'train_loss.csv',index=False)
    pd.DataFrame({'loss':getattr(model,'val_losses',[])}).to_csv(output/'val_loss.csv',index=False)
    (output/'training_complete.json').write_text(json.dumps(record,indent=2))
    initial=a[a.obs.time==0].copy()
    result={f't{t}':np.asarray(model.transform(initial, omics_key='X_stories',tau=float(t),
        batch_size=128, key=jax.random.PRNGKey(43)),dtype=np.float32) for t in [1,2,3]}
    assert all(x.shape==(5000,50) and np.isfinite(x).all() for x in result.values())
    np.savez_compressed(output/'prediction.npz',**result)
    (output/'complete.json').write_text(json.dumps(record,indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('method',choices=['stvcr','stories']);p.add_argument('--device',default='cuda:0');p.add_argument('--predict-only',action='store_true');p.add_argument('--seeds',nargs='+',type=int)
    args=p.parse_args()
    if args.method=='stvcr':stvcr(args.device,args.predict_only,args.seeds)
    else:stories(args.device)
