"""Train and evaluate scNT models with and without interactions."""
from pathlib import Path
import argparse
import copy
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
BASE = Path('/data/cytobridge/projects/CytoBridge-ST-1104')
OLD = BASE / 'runs/neuron-full-cortex/20260811'
NO = BASE / 'runs/nonspatial-corrected-matched-e80c18a-20260815-r1/scnt_cortex/no_interaction'
FULL = OLD / 'runs/lr_seed42_full_hvg2000'
DATA = OLD / 'processed/hvg2000/scnt_cortical_full_hvg2000_latent.h5ad'
PYTHON = BASE / 'envs/arista-api/bin/python'
TIMES = [0., .25, .5, 1., 2.]
GROUPS = [101, 202, 303, 404, 505]

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''): h.update(b)
    return h.hexdigest()

def dump(path, data):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2) + '\n')
    tmp.replace(path)

def status(out, phase, **extra):
    dump(out/'status.json', dict(phase=phase, updated_epoch=time.time(), **extra))

def evaluate(seed, arm, model_dir, out):
    import anndata as ad
    import numpy as np
    import torch
    from CytoBridge.tl.downstream import load_dynamical_model_from_dir, evaluate_model_distributions
    torch.set_num_threads(4)
    ev = out/'evaluation'
    ev.mkdir(exist_ok=True)
    data = ad.read_h5ad(DATA)
    stage = 'Finetune' if arm == 'full' else 'Finetune_no_interaction'
    loaded = load_dynamical_model_from_dir(model_dir, dim=50, device='cuda:0', stage=stage)
    assert loaded.weight_stage == stage
    status(out, 'distribution_evaluation', seed=seed, arm=arm)
    result = evaluate_model_distributions(data, loaded.model, time_points=TIMES,
        n_samples=2048, dt=.05, sigma=.1, include_score=True, interaction_m=16,
        max_ot_points=1024, structure_max_points=2048, device='cuda:0',
        time_key='time_point_processed', obsm_key='X_latent', concat_spatial=False,
        random_seed=42, noise_seed=42, include_initial_time=False, verbose=False)
    df = result.metrics.copy()
    df['training_seed'] = seed
    df['condition'] = arm
    df.to_csv(ev/'distribution.csv', index=False)
    status(out, 'direction_evaluation', seed=seed, arm=arm)
    sys.path.insert(0, str(ROOT/'source'))
    spec = importlib.util.spec_from_file_location('direction_reference', ROOT/'source/evaluate_full_cortex_scnt_direction.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    prepared, times, reference, reference_audit = module.build_direction_reference(
        OLD/'processed/scnt_cortical_full_20547_raw_counts.h5ad', DATA,
        OLD/'processed/hvg2000/scnt_cortical_full_hvg2000_pca_artifacts.npz',
        target_sum=10000., labeling_time=2.)
    field = module.model_field(loaded.model, np.asarray(prepared.obsm['X_latent']), times,
        include_interaction=arm == 'full', grouping_seeds=GROUPS,
        batch_size=512, device=torch.device('cuda:0'))
    cells, timewise = module.alignment_tables(seed=seed, condition=arm,
        obs_names=prepared.obs_names.astype(str), times=times, reference=reference, field=field)
    cells.to_csv(ev/'direction_cellwise.csv.gz', index=False, compression='gzip')
    timewise.to_csv(ev/'direction_timewise.csv', index=False)
    dump(ev/'manifest.json', dict(training_seed=seed, condition=arm,
        weight_path=str(loaded.weight_path), weight_sha256=sha(loaded.weight_path),
        score_path=str(loaded.score_path), score_sha256=sha(loaded.score_path),
        prepared_sha256=sha(DATA), distribution_seed=42, grouping_seeds=GROUPS,
        distribution_output_times_hours=TIMES,
        distribution_sampling='direct observed-time outputs, not dense-grid subsampling',
        reference_audit=reference_audit, runtime=str(ROOT/'runtime_full'),
        new_rna_used_for_training=False))
    status(out, 'complete', seed=seed, arm=arm)
    # The final model automatically writes the complete, unselected summary.
    paths = [ROOT/f'{a}_seed{s}/status.json' for a in ['full','no_interaction'] for s in [42,43,44]]
    if all(p.exists() and json.loads(p.read_text())['phase'] == 'complete' for p in paths):
        import fcntl
        with (ROOT/'summary.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not (ROOT/'summary').exists():
                subprocess.run([str(PYTHON), str(ROOT/'aggregate_scnt.py')], check=True)

def worker(seed, arm, evaluation_only=False):
    import yaml
    import anndata as ad
    import torch
    import CytoBridge as cb
    out = ROOT/f'{arm}_seed{seed}'
    out.mkdir(exist_ok=False)
    try:
        status(out, 'preflight', seed=seed, arm=arm)
        assert sha(DATA) == 'f59348767ea61cf6d59522158edfcc803dfb80236356b4dee1baf8c4725c97fc'
        ref = FULL if arm == 'full' else NO
        if evaluation_only:
            assert seed == 42
            model_dir = ref/'model'
        else:
            assert seed in (43, 44)
            expected_runtime = ROOT/('runtime_full' if arm == 'full' else 'runtime_no_interaction')
            assert Path(cb.__file__).resolve().is_relative_to(expected_runtime)
            cfg = yaml.safe_load((ref/'model/config.yaml').read_text())
            cfg['seed'] = seed
            if 'matched_ablation' in cfg: cfg['matched_ablation']['shared_seed'] = seed
            model_dir = out/'model'
            cfg['ckpt_dir'] = str(model_dir)
            assert [x['epochs'] for x in cfg['training']['plan']] == [100,100,50,2001,1000,2001]
            (out/'resolved_config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
            dump(out/'run_manifest.json', dict(condition=arm, seed=seed, smoke=False,
                prepared_sha256=sha(DATA), reference_config=str(ref/'model/config.yaml'),
                reference_config_sha256=sha(ref/'model/config.yaml'),
                uses_metabolic_velocity_for_training=False,
                changed_fields=['seed','matched_ablation.shared_seed when present','ckpt_dir'],
                runtime_files={str(p.relative_to(expected_runtime)):sha(p)
                    for p in sorted((expected_runtime/'CytoBridge').rglob('*.py'))}))
            torch.set_num_threads(4)
            data = ad.read_h5ad(DATA)
            assert not data.layers
            status(out, 'training', seed=seed, arm=arm)
            start = time.time()
            # The matched-ablation runtime checks provenance from the H5AD path.
            fit_input = DATA if arm == 'no_interaction' else data
            cb.tl.fit(fit_input, config=cfg, device='cuda:0', time_key='time_point_processed',
                obsm_key='X_latent', is_spatial=False, ckpt_dir=model_dir, sigma=.1,
                interaction_cutoff=(23.652481535751416 if arm == 'full' else None),
                evaluate_after_training=False)
            dump(out/'training_complete.json', dict(seconds=time.time()-start))
        env = os.environ.copy()
        env['PYTHONPATH'] = str(ROOT/'runtime_full')
        subprocess.run([str(PYTHON), '-u', str(Path(__file__)), '--evaluate',
            '--seed', str(seed), '--arm', arm, '--model-dir', str(model_dir)],
            env=env, check=True)
    except BaseException as exc:
        status(out, 'failed', seed=seed, arm=arm, error=repr(exc))
        raise

def launch(jobs=None):
    if jobs is None:
        jobs = [(43,'full',0),(43,'no_interaction',1),(44,'full',2),(44,'no_interaction',3),
                (42,'full',4),(42,'no_interaction',5)]
    for seed,arm,_ in jobs:
        if (ROOT/f'{arm}_seed{seed}').exists(): raise FileExistsError(f'{arm}_seed{seed}')
    records = []
    for seed,arm,gpu in jobs:
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=str(gpu), PYTHONHASHSEED=str(seed),
            PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='4',
            MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
            PYTHONPATH=str(ROOT/('runtime_full' if arm=='full' else 'runtime_no_interaction')),
            MPLCONFIGDIR=str(ROOT/f'cache/{arm}_{seed}/mpl'),
            NUMBA_CACHE_DIR=str(ROOT/f'cache/{arm}_{seed}/numba'))
        cmd = [str(PYTHON), '-u', str(Path(__file__)), '--seed', str(seed), '--arm', arm]
        if seed==42: cmd.append('--evaluation-only')
        with (ROOT/f'{arm}_seed{seed}.log').open('x') as log:
            p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True)
        records.append(dict(seed=seed, arm=arm, gpu=gpu, pid=p.pid))
    launch_file = ROOT/'launch.json'
    prior = json.loads(launch_file.read_text()) if launch_file.exists() else []
    dump(launch_file, prior + records)
    print(json.dumps(records))

def retry_no_interaction_preflight():
    jobs = [(43, 'no_interaction', 1), (44, 'no_interaction', 3)]
    for seed, arm, _ in jobs:
        out = ROOT/f'{arm}_seed{seed}'
        state = json.loads((out/'status.json').read_text())
        assert state['phase'] == 'failed'
        assert ('requires fit(...) to receive an H5AD file path' in state['error']
                or 'shared_seed must equal 42' in state['error'])
        assert not list(out.rglob('*.pth'))
    attempts = ROOT/'attempts'
    attempts.mkdir(exist_ok=True)
    for seed, arm, _ in jobs:
        name = f'{arm}_seed{seed}'
        suffix = f'preflight_{time.time_ns()}'
        (ROOT/name).rename(attempts/f'{name}_{suffix}')
        (ROOT/f'{name}.log').rename(attempts/f'{name}_{suffix}.log')
    launch(jobs)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--launch', action='store_true')
    p.add_argument('--retry-no-interaction-preflight', action='store_true')
    p.add_argument('--evaluate', action='store_true')
    p.add_argument('--evaluation-only', action='store_true')
    p.add_argument('--seed', type=int, choices=[42,43,44])
    p.add_argument('--arm', choices=['full','no_interaction'])
    p.add_argument('--model-dir', type=Path)
    a = p.parse_args()
    if a.retry_no_interaction_preflight: retry_no_interaction_preflight()
    elif a.launch: launch()
    elif a.evaluate: evaluate(a.seed, a.arm, a.model_dir, ROOT/f'{a.arm}_seed{a.seed}')
    else: worker(a.seed, a.arm, a.evaluation_only)
