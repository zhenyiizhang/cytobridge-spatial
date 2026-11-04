"""Generate the corrected Figure 2 observations, then fit and evaluate CytoBridge.

Run on the GPU server. Input expression coordinates are used as supplied, without
any additional normalization, logarithm or PCA fitting.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse
import json
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
import yaml

BASE = Path(os.environ.get('CYTOBRIDGE_PROJECT_DIR', '.')).resolve()
RUN = Path(os.environ.get('CYTOBRIDGE_FIGURE2_RUN', str(Path(__file__).resolve().parent)))
PACKAGE = Path(os.environ.get('CYTOBRIDGE_CODE', str(RUN / 'code')))
TEACHER = Path(os.environ.get('CYTOBRIDGE_FIGURE2_TEACHER', str(RUN / 'teacher')))
sys.path.insert(0, str(PACKAGE))
SEEDS = [1, 4, 8, 32, 256, 512, 1024, 2048, 4096, 8192]


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(8)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str) + '\n')


def load_model(directory, device):
    from CytoBridge.tl.downstream.checkpoint import load_dynamical_model_from_dir
    result = load_dynamical_model_from_dir(directory, dim=52, device=device,
        stage='Finetune', score_stage_prefer=['Score_Refine'])
    if result.score_stage != 'Score_Refine':
        raise RuntimeError('The final score model is required')
    for p in result.model.parameters():
        p.requires_grad_(False)
    return result


class Score:
    def __init__(self, model):
        self.model = model

    def compute_gradient(self, t, x):
        with torch.enable_grad():
            return self.model.compute_score(t, x.detach().requires_grad_(True),
                create_graph=False)[1].detach()


def simulate(model, x0, *, device, seed, dt=0.01):
    from CytoBridge.tl.downstream.simulation import simulate_sde_points_split_from_x0
    seed_all(seed)
    adapter = SimpleNamespace(v_net=model.predict_velocity, g_net=model.predict_growth,
        interaction_net=model.interaction_net)
    frames, lineages = simulate_sde_points_split_from_x0(
        x0=x0, f_net=adapter, score_net=Score(model), ts_points=[0., 1., 2., 3.],
        dt=dt, sigma=0.03, sigma_by_dim=None, growth_alpha=1., interaction_m=1024,
        device=device, daughter_noise_std=0., max_particles=100000,
        initial_lineage_ids=np.arange(len(x0)), return_lineage_ids=True,
        interaction_seed=seed + 10000)
    if not np.array_equal(frames[0], x0):
        raise AssertionError('The first frame must equal the exact initial condition')
    if any(len(x) == 0 or not np.isfinite(x).all() for x in frames):
        raise ValueError('Empty or non-finite generated population')
    return list(frames), list(lineages)


def generate(device, initial_states=None):
    import anndata as ad
    directory = RUN / 'data'
    directory.mkdir(exist_ok=False)
    if initial_states is None:
        a = ad.read_h5ad(TEACHER / 'preprocess/mosta_aligned.h5ad', backed='r')
        info = a.uns['preprocess_info']
        assert info['expression_source'] == "layers['count']"
        assert list(info['transformation_sequence']) == ['normalize_total', 'log1p']
        assert not info['allow_retransform_preprocessed_x']
        selected = np.flatnonzero((a.obs['time_point_processed'].to_numpy() == 0.) &
            (a.obs['annotation'].astype(str).to_numpy() == 'Brain'))
        chosen = np.random.default_rng(20260914).choice(selected, 5000, replace=False)
        x0 = np.concatenate([np.asarray(a.obsm['spatial_aligned'])[chosen],
            np.asarray(a.obsm['X_latent'])[chosen]], axis=1).astype(np.float32)
        pd.DataFrame({'source_row': chosen, 'cell_name': a.obs_names[chosen]}).to_csv(
            directory / 'initial_cells.csv', index=False)
        a.file.close()
    else:
        x0 = np.load(initial_states, allow_pickle=False).astype(np.float32)
        info = {'expression_source': "layers['count']"}
        cells = Path(initial_states).with_name('initial_cells.csv')
        if cells.exists():
            pd.read_csv(cells).to_csv(directory / 'initial_cells.csv', index=False)
    assert x0.shape == (5000, 52) and np.isfinite(x0).all()
    np.save(directory / 'initial_states.npy', x0)
    source = load_model(TEACHER / 'training', device)
    record = dict(source_h5ad=str(TEACHER / 'preprocess/mosta_aligned.h5ad'),
        teacher_checkpoint=str(source.weight_path), teacher_score=str(source.score_path),
        expression_source=info['expression_source'], transformations=['normalize_total', 'log1p'],
        initialization='5000 Brain cells from E12.5, without replacement', seed=20260914,
        times=[0,1,2,3], dt=0.01, sigma=0.03, daughter_noise=0.,
        interaction_group_size=1024, interaction_seed=20270914,
        split_times=[1,2,3], initial_time_is_exact=True, spatial_postselection=False,
        population_ceiling='Abort at 100000, never downsample',
        teacher_retrained_in_this_run=False, initial_states_input=initial_states)
    write_json(directory / 'generation_started.json', record)
    frames, lineage = simulate(source.model, x0, device=device, seed=20260914)
    table = pd.concat([pd.DataFrame(x, columns=[f'x{i}' for i in range(1,53)]).assign(
        samples=float(t)) for t,x in enumerate(frames)], ignore_index=True)
    table = table[['samples'] + [f'x{i}' for i in range(1,53)]]
    table.to_csv(directory / 'mouse_brain_simulation.csv', index=False)
    np.savez_compressed(directory / 'generated_populations.npz',
        **{f't{t}':x for t,x in enumerate(frames)},
        **{f'lineage_t{t}':x for t,x in enumerate(lineage)})
    record['counts'] = [len(x) for x in frames]
    record['csv'] = str(directory / 'mouse_brain_simulation.csv')
    write_json(directory / 'generation_complete.json', record)
    print(json.dumps(record), flush=True)


def train(device):
    import anndata as ad
    from CytoBridge.tl.train import fit
    config = yaml.safe_load((PACKAGE / 'CytoBridge/configs/simulation_config.yaml').read_text())
    teacher_config = yaml.safe_load((TEACHER / 'training/config.yaml').read_text())
    config['model']['interaction_net'] = teacher_config['model']['interaction_net'].copy()
    downloaded_classifier = TEACHER / 'training/edge_classifier.pt'
    if downloaded_classifier.exists():
        config['model']['interaction_net']['edge_predictor_path'] = str(downloaded_classifier)
    config['ckpt_dir'] = str(RUN / 'cytobridge/model')
    config['seed'] = 42
    # Only the corrected graph prior is reused. Drift, growth, score and GNN
    # trainable weights are freshly initialized by fit, not copied from teacher.
    out = RUN / 'cytobridge'
    out.mkdir(exist_ok=False)
    (out / 'training_config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    table = pd.read_csv(RUN / 'data/mouse_brain_simulation.csv')
    values = table[[f'x{i}' for i in range(1,53)]].to_numpy(np.float32)
    observed = ad.AnnData(X=values[:,2:], obs=pd.DataFrame({
        'samples':table.samples.to_numpy()}, index=[f'cell_{i}' for i in range(len(table))]))
    observed.obsm['X_latent'] = values[:,2:]
    observed.obsm['spatial_aligned'] = values[:,:2]
    observed.write_h5ad(out / 'training_input.h5ad')
    fit(str(out / 'training_input.h5ad'), config=config, device=device,
        time_key='samples', is_spatial=True,
        evaluate_after_training=False)
    write_json(out / 'training_complete.json', dict(config=str(out/'training_config.yaml'),
        seed=42, epochs=[s['epochs'] for s in config['training']['plan']],
        input=str(RUN/'data/mouse_brain_simulation.csv'), teacher_weights_used=False))


def predict(device, seeds=None):
    loaded = load_model(RUN / 'cytobridge/model', device)
    x0 = np.load(RUN / 'data/initial_states.npy')
    out = RUN / 'cytobridge/predictions'
    out.mkdir(exist_ok=True)
    seeds = SEEDS if seeds is None else seeds
    for seed in seeds:
        if (out / f'seed_{seed}.npz').exists():
            raise FileExistsError(out / f'seed_{seed}.npz')
        frames, lineage = simulate(loaded.model, x0, device=device, seed=seed)
        np.savez_compressed(out / f'seed_{seed}.npz', **{f't{t}':x for t,x in enumerate(frames)})
    write_json(out / f'complete_seeds_{seeds[0]}.json', dict(seeds=seeds, dt=.01, times=[0,1,2,3]))
    if all((out / f'seed_{seed}.npz').exists() for seed in SEEDS):
        write_json(out / 'complete.json', dict(seeds=SEEDS, dt=.01, times=[0,1,2,3]))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['generate', 'train', 'predict'])
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--seeds', nargs='+', type=int)
    p.add_argument('--initial-states', type=Path,
        help='Optional released 5000-cell initial_states.npy instead of the full MOSTA H5AD')
    args = p.parse_args()
    seed_all(42)
    if args.stage == 'predict':
        predict(args.device, args.seeds)
    elif args.stage == 'generate':
        generate(args.device, args.initial_states)
    else:
        globals()[args.stage](args.device)
