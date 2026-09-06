"""Chicken-heart alignment sensitivity: translations 1/2 and rotations 1/3 degrees."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import argparse
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import tarfile
import time

ROOT = Path('/home/ubuntu/cytobridge_runs/heart_alignment_small_20260906')
SCRATCH = Path('/dev/shm/cytobridge-heart-alignment-small-20260906-r1')
SOFTWARE = Path('/home/ubuntu/cytobridge_runs/heart_alignment_precision_20260906/software')
ARCHIVES = Path('/data/cytobridge/projects/CytoBridge-ST-1104/runs/heart-alignment-small-20260906-r1')
VARIANTS = ('baseline_repeat','translate_low','translate_moderate',
            'rotate_low','rotate_moderate','translate_rotate_low','translate_rotate_moderate')


def load(name):
    spec = importlib.util.spec_from_file_location(name,ROOT/'code'/f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def configure():
    base = load('prepare_and_run')
    directions = base._translation('moderate')
    signs = {'D4':1,'D7':-1,'D10':1,'D14':-1}
    specs = {}
    for level,distance,angle in (('low',1,1),('moderate',2,3)):
        for kind in ('translate','rotate','translate_rotate'):
            values = {}
            for stage in base.TIME_ORDER:
                old = directions[stage]
                norm = math.hypot(old.translate_x_nn,old.translate_y_nn)
                dx = distance*old.translate_x_nn/norm if kind!='rotate' else 0
                dy = distance*old.translate_y_nn/norm if kind!='rotate' else 0
                values[stage] = base.RigidPerturbation(
                    signs[stage]*angle if kind!='translate' else 0,dx,dy)
            specs[f'{kind}_{level}'] = values
    base.VARIANT_SPECS = specs
    base.PACKAGE_ROOT = SOFTWARE
    base.CONFIG_TEMPLATE = SOFTWARE/'CytoBridge/workflow_configs/chicken_heart.json'
    base.AUDIT_ROOT = SCRATCH
    base.INPUT_DIR = SCRATCH/'inputs'
    base.RUNS_DIR = SCRATCH/'runs'
    return base


def prepare():
    if SCRATCH.exists() or ARCHIVES.exists():
        raise FileExistsError('An experiment already exists at the selected path')
    if shutil.disk_usage(ARCHIVES.parent).free < 2*1024**3:
        raise RuntimeError('Need 2 GiB free for permanent archives')
    SCRATCH.mkdir()
    for folder in ('logs','summary','alignment_summary'):
        (ROOT/folder).mkdir(exist_ok=True)
    base = configure()
    base.prepare_inputs()
    shutil.copy2(SCRATCH/'input_manifest.json',ROOT/'input_manifest.json')
    shutil.copy2(SOFTWARE.parent/'precision_change.json',ROOT/'precision_change.json')
    design = {'translations_median_nn_units':[1,2],'rotation_degrees':[1,3],
              'variants':list(VARIANTS),'seed':42,'alignment_seed':42,
              'software':str(SOFTWARE),'input_manifest':str(ROOT/'input_manifest.json'),
              'scratch':str(SCRATCH),'run_archives':str(ARCHIVES),
              'scope':'Small rigid perturbations. The earlier 6-degree experiment is retained separately.',
              'earlier_experiment':'/home/ubuntu/cytobridge_runs/heart_alignment_integer_20260906',
              'precision_change':json.loads((ROOT/'precision_change.json').read_text()),
              'started':time.strftime('%Y-%m-%d %H:%M:%S')}
    (ROOT/'experiment.json').write_text(json.dumps(design,indent=2))


def one(step,variant):
    base = configure()
    if step=='full':
        base.run_variant(variant,'cuda')
        shutil.copy2(ROOT/'precision_change.json',base.RUNS_DIR/variant/'precision_change.json')
        return
    output = SCRATCH/'alignment'/variant
    output.mkdir(parents=True,exist_ok=False)
    config_path = output/'workflow_config.json'
    shutil.copy2(base.CONFIG_TEMPLATE,config_path)
    command = [sys.executable,'-m','CytoBridge.cli','workflow','--config',str(config_path),
               '--step','preprocess','--input-h5ad',str(base._input_for_variant(variant)),
               '--output-dir',str(output),'--device','cuda']
    subprocess.run(command,cwd=SOFTWARE,env=os.environ.copy(),check=True)


def launch(step,gpu,variant):
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(gpu),PYTHONHASHSEED='0',
               OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    with (ROOT/'logs'/f'{step}_{variant}.log').open('w') as log:
        subprocess.run([sys.executable,str(Path(__file__).resolve()),'one',step,variant],
                       env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    result = {'variant':variant,'completed':time.time()}
    if step=='full':
        archive = ARCHIVES/f'{variant}.tar.gz'
        partial = archive.with_suffix('.gz.partial')
        if archive.exists() or partial.exists():
            raise FileExistsError(archive)
        with tarfile.open(partial,'w:gz',compresslevel=5) as bundle:
            bundle.add(SCRATCH/'runs'/variant,arcname=variant)
        partial.rename(archive)
        result.update(archive=str(archive),archive_bytes=archive.stat().st_size)
    return result


def run(step):
    if step=='full':
        check = json.loads((ROOT/'alignment_summary/check.json').read_text())
        if not check['no_relative_reflections']:
            raise RuntimeError('An alignment is reflected relative to the unperturbed run')
        ARCHIVES.mkdir(exist_ok=False)
    completed = []
    with ThreadPoolExecutor(max_workers=7) as pool:
        futures = [pool.submit(launch,step,gpu,variant) for gpu,variant in enumerate(VARIANTS)]
        for future in as_completed(futures):
            completed.append(future.result())
            (ROOT/f'{step}_status.json').write_text(json.dumps({'status':'running','completed':completed},indent=2))
    (ROOT/f'{step}_status.json').write_text(json.dumps({'status':'completed','completed':completed},indent=2))


def check_alignment():
    import anndata as ad
    import numpy as np
    import pandas as pd
    from scipy.linalg import orthogonal_procrustes
    arrays = {}
    for variant in VARIANTS:
        data = ad.read_h5ad(SCRATCH/'alignment'/variant/'preprocess/chicken_heart_aligned.h5ad',backed='r')
        if variant=='baseline_repeat':
            names = data.obs_names
            stages = data.obs.timepoint.astype(str).to_numpy()
        positions = data.obs_names.get_indexer(names)
        assert (positions>=0).all() and len(data)==len(names)
        arrays[variant] = np.asarray(data.obsm['spatial_aligned'],dtype=float)[positions]
        data.file.close()
    rows = []
    for variant in VARIANTS[1:]:
        for stage in ('D4','D7','D10','D14'):
            mask = stages==stage
            ref = arrays['baseline_repeat'][mask]
            current = arrays[variant][mask]
            ref = ref-ref.mean(axis=0)
            current = current-current.mean(axis=0)
            best,_ = orthogonal_procrustes(current,ref)
            determinant = float(np.linalg.det(best))
            if determinant<0:
                u,_,vt = np.linalg.svd(current.T@ref)
                u[:,-1]*=-1
                best=u@vt
            residual = 100*np.sqrt(np.mean((current@best-ref)**2))/np.sqrt(np.mean(np.sum(ref**2,axis=1)))
            rows.append({'variant':variant,'stage':stage,'orthogonal_determinant':determinant,
                         'proper_rotation_residual_percent':float(residual)})
    table = pd.DataFrame(rows)
    table.to_csv(ROOT/'alignment_summary/residuals.csv',index=False)
    check = {'no_relative_reflections':bool((table.orthogonal_determinant>0).all()),
             'largest_coordinate_residual_percent':float(table.proper_rotation_residual_percent.max())}
    (ROOT/'alignment_summary/check.json').write_text(json.dumps(check,indent=2))
    np.savez_compressed(ROOT/'alignment_summary/coordinates.npz',timepoint=stages.astype(str),**arrays)
    print(table.to_string(index=False),flush=True)
    print(json.dumps(check,indent=2),flush=True)
    if not check['no_relative_reflections']:
        raise RuntimeError('Check reflected sections before training')


def summarize():
    comparison = load('compare_runs')
    comparison.RUNS_DIR = SCRATCH/'runs'
    comparison.SUMMARY_DIR = ROOT/'summary'
    comparison.VARIANTS = VARIANTS
    comparison.compare()
    export = load('export_plot_inputs')
    export.AUDIT_ROOT = SCRATCH
    export.RUNS_DIR = SCRATCH/'runs'
    export.OUTPUT = ROOT/'summary/plot_inputs.npz'
    export.MANIFEST = ROOT/'summary/plot_inputs_manifest.json'
    export.DISPLAY_VARIANTS = VARIANTS
    export.main()
    for filename in ('experiment.json','input_manifest.json','precision_change.json'):
        shutil.copy2(ROOT/filename,ARCHIVES/filename)
    for folder in ('code','summary','alignment_summary'):
        shutil.copytree(ROOT/folder,ARCHIVES/folder)
    shutil.copytree(SOFTWARE,ARCHIVES/'software')
    (ROOT/'status.json').write_text(json.dumps({'status':'completed','archive':str(ARCHIVES)},indent=2))


if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='one':
        one(sys.argv[2],sys.argv[3])
    else:
        try:
            prepare()
            run('alignment')
            check_alignment()
            run('full')
            summarize()
        except Exception as error:
            (ROOT/'status.json').write_text(json.dumps({'status':'failed','error':repr(error)},indent=2))
            raise
