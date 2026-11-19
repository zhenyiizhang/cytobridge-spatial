"""Regression checks for the author's tutorial review corrections."""
import ast
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_ad_training_selects_the_model_used_by_analysis(tmp_path):
    nb = json.loads((ROOT/'docs/tutorials/dataset_workflows/admouse.ipynb').read_text())
    code = [''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code']
    training = next(c for c in code if c.startswith('if train_model:'))
    calls = []
    scope = dict(train_model=True, PROJECT_DIR=tmp_path, DATA_DIR=tmp_path/'data/admouse',
                 MODEL_DIR=tmp_path/'data/admouse/model', EDGE_MODEL=tmp_path/'edge.pt',
                 DEVICE='cuda', cb=SimpleNamespace(tl=SimpleNamespace(fit=lambda *a,**k:calls.append((a,k)))))
    exec(compile(training,'ad_training','exec'),scope)
    assert len(calls)==1
    args, kwargs=calls[0]
    assert args==(str(tmp_path/'data/admouse/aligned.h5ad'),)
    assert kwargs['ckpt_dir']==scope['MODEL_DIR']==tmp_path/'outputs/admouse_training/model'
    assert kwargs['config']==str(tmp_path/'data/admouse/model/config.yaml')
    assert kwargs['edge_predictor_path']==str(tmp_path/'edge.pt')
    assert 'MODEL_DIR, dim=reference.n_vars' in '\n'.join(code)
    prose='\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='markdown')
    assert 'expression weight of 0.015' not in prose
    assert 'train_model = True' in prose and 'train_model = False' in prose


def test_agist_rejects_existing_metrics_instead_of_silently_reusing_them(tmp_path):
    spec=importlib.util.spec_from_file_location('agist_evaluation',ROOT/'scripts/evaluate_and_plot_agist_w2_replicates.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    csv=tmp_path/'metrics.csv'
    csv.write_text('seed,time,space,w2\n42,1.0,gene,1.0\n')
    with pytest.raises(FileExistsError,match='Choose a new output directory'):
        module.compute_metrics(tmp_path/'changed_trajectory',tmp_path/'changed_truth.csv',csv)
    assert csv.read_text().endswith('42,1.0,gene,1.0\n')


def test_mosta_wave_accepts_new_phase_sizes(tmp_path):
    from reproduction.mosta.gene_enrichment import render_wave_axes
    folder=tmp_path/'s10_developmental_wave'
    folder.mkdir()
    names=[f'gene_{i}' for i in range(1000)]
    pd.DataFrame(np.zeros((1000,13)),index=names,columns=np.arange(13)/4).to_csv(
        folder/'s10_top1000_peak_ordered_profiles.csv')
    pd.DataFrame({'profile':names,'phase':np.repeat([1,2,3],[400,300,300]),
                  'peak_time':np.zeros(1000)}).to_csv(folder/'s10_top1000_dp3_assignments.csv',index=False)
    fig=plt.figure()
    try:
        result=render_wave_axes(fig,tmp_path)
        assert result['phase_sizes']=={'1':400,'2':300,'3':300}
    finally:
        plt.close(fig)


def test_ad_go_emits_only_the_four_paper_plot_types():
    script=(ROOT/'reproduction/admouse/go/enrich_program.R').read_text()
    assert '"_bar.pdf"' not in script
    for name in ('dot','cnet','upset','emap'):
        assert f'"_{name}.pdf"' in script
