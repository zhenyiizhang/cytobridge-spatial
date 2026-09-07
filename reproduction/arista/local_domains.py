"""Generate S25 domain assignments and matched nulls from model-derived inputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
import CytoBridge as cb

SCRIPTS = Path(__file__).resolve().parents[2] / 'scripts/reviewer_arista_20260824'
FILES = {
    'roi_assignments.csv': 'roi_two_niche_assignments.csv',
    'domain_metadata.csv': 'two_niche_metadata.csv',
    'celltype_edges.csv': 'two_niche_t1_celltype_edges.csv',
    'attention_null.csv': 'two_niche_attention_matched_null.csv',
    'pathway_null.csv': 'two_niche_lr_pathway_matched_null.csv',
}


def generate(data_dir, population_dir, velocity_table, pair_timecourse, output_dir):
    """Run the original segmentation and 9,999/1,999-permutation calculations.

    The six-file export changes filenames only; it does not create or adjust
    scientific values. A changed field can fail the original two-domain or
    significant-pathway contracts and is not forced to match the paper.
    """
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f'Choose a new output directory: {output}')
    output.mkdir(parents=True)
    velocity = pd.read_csv(velocity_table)
    roi = velocity.loc[velocity.in_roi.astype(str).str.lower().eq('true')].copy()
    if roi.empty:
        raise ValueError('The calculated Figure 5c table contains no ROI cells.')
    assignments = output / 'input_roi.csv'
    roi.to_csv(assignments, index=False)
    population = Path(population_dir).resolve()
    aligned = Path(data_dir).resolve() / 'aligned.h5ad'
    database = Path(cb.__file__).parent / 'workflow_databases/CellChatDB.ligrec.human.csv'
    common = ['--aligned-h5ad', str(aligned), '--lr-database', str(database),
              '--pair-timecourse', str(Path(pair_timecourse).resolve())]
    subprocess.run([sys.executable, str(SCRIPTS / 'server_analyze_figure5c_two_niche_timecourse.py'),
                    *common, '--assignments', str(assignments),
                    '--attention-dir', str(population / 'attention'),
                    '--slice-dir', str(population / 'model_states'),
                    '--output-dir', str(output / 'timecourse'),
                    '--upper-quantile', '.75', '--minimum-size', '20',
                    '--interaction-cutoff', '0.03154105148551745',
                    '--n-attention-permutations', '9999', '--n-lr-permutations', '1999',
                    '--seed', '42'], check=True)
    tables = output / 'timecourse/tables'
    subprocess.run([sys.executable, str(SCRIPTS / 'server_analyze_figure5c_two_niche_lr_axes.py'),
                    *common, '--assignments', str(tables / FILES['roi_assignments.csv']),
                    '--edge-index', str(population / 'attention/edge_index_interp_t1.0.npy'),
                    '--attention', str(population / 'attention/attn_mean_interp_t1.0.npy'),
                    '--output-dir', str(output / 'lr_axes'), '--time', '1',
                    '--n-permutations', '1999', '--seed', '260824'], check=True)
    compact = output / 'inputs'
    compact.mkdir()
    for name, source in FILES.items():
        pd.read_csv(tables / source).to_csv(compact / name, index=False)
    pd.read_csv(output / 'lr_axes/two_niche_lr_pair_matched_null.csv.gz').to_csv(
        compact / 'lr_pair_null.csv.gz', index=False)
    manifest = dict(analysis='arista_local_domains_calculated',
                    source='original two-niche segmentation and matched-null producers',
                    velocity_table=str(Path(velocity_table).resolve()),
                    pair_timecourse=str(Path(pair_timecourse).resolve()),
                    population_dir=str(population),
                    attention_permutations=9999, lr_permutations=1999)
    (compact / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return load_calculated_domains(compact)


def load_calculated_domains(input_dir):
    """Load a new six-file calculation without asserting archived cell counts."""
    from CytoBridge.results.arista_local_domains import AristaLocalDomainData
    source = Path(input_dir).resolve()
    manifest = json.loads((source / 'manifest.json').read_text())
    if manifest.get('analysis') != 'arista_local_domains_calculated':
        raise ValueError('Use this loader only for the output of local_domains.generate.')
    return AristaLocalDomainData(
        source, manifest, pd.read_csv(source / 'roi_assignments.csv'),
        pd.read_csv(source / 'domain_metadata.csv'), pd.read_csv(source / 'celltype_edges.csv'),
        pd.read_csv(source / 'attention_null.csv'), pd.read_csv(source / 'pathway_null.csv'),
        pd.read_csv(source / 'lr_pair_null.csv.gz'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data-dir', 'population-dir', 'velocity-table', 'pair-timecourse', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    result = generate(args.data_dir, args.population_dir, args.velocity_table,
                      args.pair_timecourse, args.output_dir)
    print(result.domain_metadata)
