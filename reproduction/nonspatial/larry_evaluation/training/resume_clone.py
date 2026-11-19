"""Retry evaluation only after fixing boolean difference reporting."""
from run_matched import ROOT, LR, DATA, PYTHON, write_json
import os
import json
import subprocess
import sys
import time

def worker(seed):
    out = ROOT / f'seed{seed}'
    state = json.loads((out/'status.json').read_text())
    assert state['distribution_repeats_completed'] == 50
    assert (out/'model/Score_Refine/score_model.pth').exists()
    (out/'clone_fate').rename(out/'clone_fate_before_boolean_report_fix')
    state['previous_error'] = state.pop('error')
    state.update(phase='clone_evaluation', updated_epoch=time.time())
    write_json(out/'status.json',state)
    try:
        frozen=LR/f'reviews/20260719/reviewer_three_condition_corrected_v3/clone_fate_runs/clone_fate_seed{seed}_state_v2_10'
        subprocess.run([str(PYTHON),str(ROOT/'evaluate_clone.py'),'--prepared-h5ad',str(DATA),
            '--no-interaction-model-dir',str(out/'model'), '--historical-evaluator',str(ROOT/'runtime/scripts/evaluate_weinreb_clone_fate.py'),
            '--frozen-full-root',str(frozen),'--output-dir',str(out/'clone_fate'),'--device','cuda:0'],check=True)
        state.update(phase='complete',updated_epoch=time.time(),elapsed_seconds=time.time()-state['started_epoch'])
    except BaseException as error:
        state.update(phase='failed',error=repr(error),updated_epoch=time.time())
        raise
    finally:
        write_json(out/'status.json',state)

if __name__=='__main__':
    if len(sys.argv)>1:
        worker(int(sys.argv[1]))
    else:
        for seed,gpu in zip([42,43,44],[0,1,2]):
            env=os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH=str(ROOT/'runtime'),PYTHONNOUSERSITE='1',
                OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
            with (ROOT/f'seed{seed}_clone_retry.log').open('x') as log:
                p=subprocess.Popen([str(PYTHON),str(ROOT/'resume_clone.py'),str(seed)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                print(seed,p.pid)
