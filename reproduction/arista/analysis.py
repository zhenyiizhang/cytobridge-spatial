"""Calculate ARISTA S20/S22--S24 inputs from unwarped model populations.

No packaged result table or finished figure is read by these producers.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import CytoBridge as cb

TIMES = np.arange(0., 4.01, .5)
GO_LIBRARY = Path(__file__).parent / 'data/GO_Biological_Process_2023.gmt'


def load_states(population_dir):
    """Read the exact communication/model states exported by ``generate``."""
    return {str(t): ad.read_h5ad(Path(population_dir) / 'model_states' /
                               f"time_{f'{t:g}'.replace('.', 'p')}.h5ad")
            for t in TIMES}


def calculate_growth(data_dir, population_dir, output_dir, device='cuda', *, model_dir=None):
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f'Choose a new output directory: {output}')
    selected_model = Path(data_dir) / 'model' if model_dir is None else Path(model_dir)
    loaded = cb.tl.load_dynamical_model_from_dir(selected_model, dim=52, device=device)
    growth = cb.tl.evaluate_growth_by_timepoint(
        load_states(population_dir), loaded.model, time_points=TIMES,
        annotation_key='Annotation', device=device)
    output.mkdir(parents=True)
    growth.to_csv(output / 'growth_by_cell.csv.gz', index=False)
    return growth


def calculate_gene_programs(data_dir, population_dir, output_dir, gene_set_gmt=GO_LIBRARY):
    """Persisted-center inverse PCA, top-2000 clustering, expression-background ORA."""
    from scripts.reviewer_arista_20260824.build_s15_s17_strict_legacy_style import (
        _build_corrected_gene_programs, _load_gmt, _gene_symbol,
        _unique_display_map, _ora_expression_background)
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f'Choose a new output directory: {output}')
    reference = ad.read_h5ad(Path(data_dir) / 'aligned.h5ad')
    result = cb.tl.summarize_temporal_gene_patterns(
        load_states(population_dir), reference, time_points=TIMES, spatial_dim=2,
        n_top_genes=250, n_clusters=2, preferred_species_tag='hs', clip_min=0.,
        allow_complete_reference_pca_center_fallback=False)
    expression = result.expression
    ranking, assignments, normalized, prototypes = _build_corrected_gene_programs(expression)
    roster = ranking.head(18).copy()
    roster['display_gene'] = roster.raw_gene.map(_unique_display_map(roster.raw_gene))
    roster['gene_symbol'] = roster.raw_gene.map(_gene_symbol)
    library = _load_gmt(Path(gene_set_gmt))
    background = {s for s in map(_gene_symbol, expression.index) if s}
    output.mkdir(parents=True)
    expression.to_csv(output / 'gene_trajectories.csv')
    result.signed_expression.to_csv(output / 'signed_mean_expression.csv')
    result.reconstruction_diagnostics.to_csv(output / 'reconstruction_diagnostics.csv', index=False)
    roster.to_csv(output / 'gene_display_roster.csv', index=False)
    assignments.to_csv(output / 'gene_program_assignments.csv', index=False)
    normalized.to_csv(output / 'gene_program_normalized_profiles.csv')
    prototypes.to_csv(output / 'gene_program_prototypes.csv', index=False)
    for pattern, subset in assignments.groupby('pattern', sort=True):
        enriched = _ora_expression_background(
            [s for s in subset.gene_symbol if isinstance(s, str)], library, background,
            alpha=.05, min_set_size=5, max_set_size=5000, min_overlap=2)
        enriched.insert(0, 'pattern', int(pattern))
        enriched.to_csv(output / f'gene_program_{pattern}_GO_terms.csv', index=False)
    return prototypes


def calculate_lr_timecourses(data_dir, population_dir, output_dir):
    """Strict-min complexes; real observed expression and reconstructed intermediates."""
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f'Choose a new output directory: {output}')
    reference = ad.read_h5ad(Path(data_dir) / 'aligned.h5ad')
    # This pickle is produced by the preceding trusted local calculation.
    with (Path(population_dir) / 'all_time_communications.pkl').open('rb') as handle:
        communications = pickle.load(handle)
    database = Path(cb.__file__).parent / 'workflow_databases/CellChatDB.ligrec.human.csv'
    result = cb.tl.project_communication_to_lr_timecourses(
        load_states(population_dir), reference, communications, database,
        time_points=TIMES, annotation_key='Annotation', spatial_dim=2,
        complex_mode='min', expression_space='log1p', require_all_subunits=True,
        preferred_species_tag='hs', allow_complete_reference_pca_center_fallback=False,
        observed_adata=reference, observed_time_key='time_point_processed',
        observed_time_points=[0., 1., 2., 3., 4.],
        observed_annotation_key='Annotation', observed_expression_space='log1p')
    output.mkdir(parents=True)
    for name in ('pair_timecourse', 'celltype_timecourse', 'pattern_summary', 'coverage',
                 'trajectory_coverage', 'dropped_trajectories'):
        getattr(result, name).to_csv(output / f'{name}.csv', index=False)
    return result.pair_timecourse


def calculate_lr_panels(timecourse):
    """Apply the paper k=2/min-max clustering and 25-per-pattern display selection.

    Unlike the archived-result validator, this accepts the actual cluster sizes
    from a new run. The k=2--8 diagnostic is reported, never used to silently
    change the paper's fixed two-program analysis.
    """
    from CytoBridge.results.arista_supplementary_figures import AristaLigandReceptorPanels
    profiles = timecourse.pivot(index='pair', columns='time', values='score').sort_index()
    if not np.array_equal(profiles.columns.to_numpy(float), TIMES) or profiles.isna().any().any():
        raise ValueError('LR profiles must contain every pair at all nine ARISTA times.')
    runs = {k: cb.tl.cluster_temporal_profiles(
        profiles, n_clusters=k, normalization='minmax', method='kmeans',
        cluster_order='peak_time') for k in range(2, 9)}
    clustering = runs[2]
    assignments = clustering.assignments.rename(columns={'profile': 'pair'})
    prototypes = clustering.prototypes.rename(columns={
        'mean': 'mean_normalized_score', 'std': 'std_normalized_score', 'n_profiles': 'n_pairs'})
    k_rows = []
    for k, run in runs.items():
        counts = run.assignments.groupby('cluster').size()
        k_rows.append(dict(k=k, silhouette=float(run.diagnostics.iloc[0].silhouette),
                           minimum_cluster_size=int(counts.min()), maximum_cluster_size=int(counts.max()),
                           cluster_counts=';'.join(f'{i}:{n}' for i, n in counts.items())))
    normalized = clustering.normalized_profiles.copy()
    normalized.index.name = 'pair'
    normalized.columns = [f'time_{t:.1f}' for t in TIMES]
    normalized = normalized.reset_index()
    joined = normalized.merge(assignments, on='pair', validate='one_to_one')
    columns = [f'time_{t:.1f}' for t in TIMES]
    blocks = []
    for cluster, block in joined.groupby('cluster', sort=True):
        block = block.copy()
        block['distance_to_pattern_prototype'] = np.linalg.norm(
            block[columns].to_numpy() - block[columns].mean().to_numpy(), axis=1)
        block = block.sort_values(['distance_to_pattern_prototype', 'pair'], kind='mergesort')
        block['representativeness_rank_within_pattern'] = np.arange(1, len(block)+1)
        if len(block) < 25:
            raise ValueError(f'Program {cluster} has fewer than 25 pairs.')
        blocks.append(block.head(25))
    roster = pd.concat(blocks, ignore_index=True)
    roster['display_order'] = np.arange(1, len(roster)+1)
    keys = ['pair', 'cluster', 'representativeness_rank_within_pattern',
            'distance_to_pattern_prototype', 'display_order']
    displayed = timecourse.merge(roster[keys], on='pair', validate='many_to_one').sort_values(
        ['display_order', 'time'], kind='mergesort').reset_index(drop=True)
    return AristaLigandReceptorPanels(
        prototypes.reset_index(drop=True), assignments.sort_values('pair').reset_index(drop=True),
        normalized, pd.DataFrame(k_rows), clustering.diagnostics.copy(), roster, displayed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('analysis', choices=['growth', 'genes', 'lr'])
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--population-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--model-dir', type=Path, help='Selected model for the growth calculation.')
    args = parser.parse_args()
    functions = {'growth': calculate_growth, 'genes': calculate_gene_programs, 'lr': calculate_lr_timecourses}
    kwargs = {'device': args.device, 'model_dir': args.model_dir} if args.analysis == 'growth' else {}
    result = functions[args.analysis](args.data_dir, args.population_dir, args.output_dir, **kwargs)
    print(result.head())
