"""Quantify a new Figure 5e field evaluation against the retained paper table.

This is a comparison, not a numerical compatibility correction. It never
modifies either input or substitutes archived values into a model calculation.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def summarize(values):
    return values.groupby(['time', 'celltype'], as_index=False).agg(
        growth_mean=('growth', 'mean'), interaction_mean=('interaction', 'mean'),
        n=('growth', 'size')).sort_values(['time', 'celltype']).reset_index(drop=True)


def compare_evaluations(calculated_values, paper_values, output_dir=None):
    """Return group, weighted-time and interpretation-sensitive comparisons."""
    actual, reference = summarize(calculated_values), summarize(paper_values)
    keys = ['time', 'celltype']
    if not actual[keys + ['n']].equals(reference[keys + ['n']]):
        raise ValueError('Comparison requires the same time/cell-type groups and cell counts.')
    groups = reference.merge(actual, on=keys + ['n'], suffixes=('_paper', '_new'), validate='one_to_one')
    for value in ['growth_mean', 'interaction_mean']:
        groups[value + '_delta'] = groups[value + '_new'] - groups[value + '_paper']

    def positions(table):
        # Same min/max mapping and physical axes extent as growth_plot.py.
        v = table[['interaction_mean', 'growth_mean']].to_numpy()
        r = np.ptp(v, axis=0)
        return (v - v.min(axis=0) + .05*r) / (1.1*r) * [220.0726, 149.75158]

    groups['circle_displacement_pt'] = np.linalg.norm(positions(actual)-positions(reference), axis=1)
    times = paper_values.groupby('time')[['growth', 'interaction']].mean().join(
        calculated_values.groupby('time')[['growth', 'interaction']].mean(),
        lsuffix='_paper', rsuffix='_new').reset_index()
    rankings = []
    for time, block in groups.groupby('time'):
        paper_top = block.nlargest(3, 'interaction_mean_paper').celltype.tolist()
        new_top = block.nlargest(3, 'interaction_mean_new').celltype.tolist()
        rankings.append(dict(time=time, paper_top3=';'.join(paper_top), new_top3=';'.join(new_top),
                             same_top3_set=set(paper_top) == set(new_top),
                             spearman=float(spearmanr(block.interaction_mean_paper,
                                                      block.interaction_mean_new).statistic)))
    interpretation = []
    for name, table, values in [('paper', reference, paper_values), ('new', actual, calculated_values)]:
        indexed = table.set_index(keys)
        delta = (indexed.xs(4.) - indexed.xs(0.)).dropna()
        interpretation.append(dict(
            evaluation=name, matched_t0_t4_celltypes=len(delta),
            growth_increased=int((delta.growth_mean > 0).sum()),
            interaction_increased=int((delta.interaction_mean > 0).sum()),
            interaction_decreased=int((delta.interaction_mean < 0).sum()),
            early_weighted_interaction=values.loc[values.time.le(1), 'interaction'].mean(),
            late_weighted_interaction=values.loc[values.time.ge(3), 'interaction'].mean()))
    statistics = pd.DataFrame([dict(
        groups=len(groups), interaction_pearson=pearsonr(reference.interaction_mean, actual.interaction_mean).statistic,
        interaction_spearman=spearmanr(reference.interaction_mean, actual.interaction_mean).statistic,
        max_group_interaction_delta=groups.interaction_mean_delta.abs().max(),
        max_circle_displacement_pt=groups.circle_displacement_pt.max(),
        median_circle_displacement_pt=groups.circle_displacement_pt.median(),
        p95_circle_displacement_pt=groups.circle_displacement_pt.quantile(.95))])
    result = dict(groups=groups, time_means=times, rankings=pd.DataFrame(rankings),
                  interpretation=pd.DataFrame(interpretation), statistics=statistics)
    if output_dir is not None:
        output = Path(output_dir)
        if output.exists():
            raise FileExistsError(f'Choose a new output directory: {output}')
        output.mkdir(parents=True)
        for name, table in result.items():
            table.to_csv(output / f'{name}.csv', index=False)
    return result


def run_seed_audit(model_dir, state_dir, paper_table, output_dir, device='cuda'):
    """Evaluate all prespecified seeds 0--4; do not select a best seed."""
    from .model_fields import calculate_fields
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f'Choose a new output directory: {output}')
    output.mkdir(parents=True)
    paper = pd.read_csv(paper_table)
    reference = summarize(paper).set_index(['time', 'celltype'])
    groups, time_means, interpretations = [], [], []
    for seed in range(5):
        target = output / f'seed_{seed}'
        calculate_fields(model_dir, state_dir, target, device=device, seed=seed, save_components=False)
        actual = pd.read_csv(target / 'figure5e_growth_interaction_by_cell.csv')
        comparison = compare_evaluations(actual, paper)
        groups.append(summarize(actual).assign(seed=seed))
        time_means.append(comparison['time_means'].assign(seed=seed))
        interpretations.append(comparison['interpretation'].query("evaluation == 'new'").assign(seed=seed))
    group_runs = pd.concat(groups, ignore_index=True)
    summary = group_runs.groupby(['time', 'celltype']).interaction_mean.agg(
        seed_min='min', seed_max='max', seed_mean='mean', seed_sd='std')
    summary['paper_mean'] = reference.interaction_mean
    summary['paper_within_five_seed_range'] = summary.paper_mean.between(summary.seed_min, summary.seed_max)
    summary['paper_minus_seed_mean_over_seed_sd'] = (
        summary.paper_mean - summary.seed_mean) / summary.seed_sd.replace(0, np.nan)
    group_runs.to_csv(output / 'all_seed_group_means.csv', index=False)
    summary.reset_index().to_csv(output / 'paper_vs_five_seed_groups.csv', index=False)
    pd.concat(time_means, ignore_index=True).to_csv(output / 'all_seed_weighted_time_means.csv', index=False)
    pd.concat(interpretations, ignore_index=True).to_csv(output / 'all_seed_interpretation.csv', index=False)
    (output / 'protocol.json').write_text(json.dumps({
        'seeds_prespecified': list(range(5)), 'best_seed_selection': False,
        'state_dir': str(Path(state_dir).resolve()), 'model_dir': str(Path(model_dir).resolve()),
        'paper_table': str(Path(paper_table).resolve()),
        'interpretation': 'Five grouping draws are a sensitivity check, not a calibrated confidence interval.'}, indent=2) + '\n')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['model-dir', 'state-dir', 'paper-table', 'output-dir']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    print(run_seed_audit(args.model_dir, args.state_dir, args.paper_table, args.output_dir, args.device))
