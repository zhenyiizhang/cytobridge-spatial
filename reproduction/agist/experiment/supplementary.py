"""Calculate later attention graphs and redraw the two AGIST SI figures."""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from run_cytobridge import RUN, TEACHER, SEEDS, load_model, seed_all


def attention(role, device):
    from reproduction.agist.main_figure import _attention_edges
    frame = pd.read_csv(RUN / 'data/mouse_brain_simulation.csv')
    model = load_model(TEACHER / 'training' if role == 'truth' else RUN / 'cytobridge/model', device).model
    for t in (1, 2, 3):
        destination = RUN / f'evaluation/{role}/attention_time{t}.npz'
        if destination.exists():
            raise FileExistsError(destination)
        x = frame.loc[frame.samples == t].iloc[:, 1:].to_numpy(np.float32)
        seed_all(42 + t)
        edges = _attention_edges(model, torch.tensor(x, device=device), float(t), device)
        np.savez_compressed(destination, **edges)
        print(role, t, len(x), len(edges['attention']), flush=True)


def style():
    import matplotlib as mpl
    mpl.rcParams.update({'font.family': 'Arial', 'font.size': 9,
        'text.color': 'black', 'axes.labelcolor': 'black', 'axes.edgecolor': 'black',
        'xtick.color': 'black', 'ytick.color': 'black', 'axes.titlesize': 9,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.linewidth': .7})


def save(fig, name):
    from matplotlib.text import Text
    for item in fig.findobj(Text):
        item.set_fontfamily('Arial')
        item.set_color('black')
    fig.savefig(RUN / f'figures/{name}.pdf')
    fig.savefig(RUN / f'figures/{name}.png', dpi=320)


def spatial_view(ax, xy):
    """Enlarge the main tissue region and retain all cells in an overview inset.

    This changes display bounds only. No cell is removed from any calculation.
    The inset rectangle marks the main panel's bounds.
    """
    from matplotlib.patches import Rectangle
    lower, upper = np.quantile(xy, [.005, .995], axis=0)
    padding = (upper - lower) * .06
    lower, upper = lower - padding, upper + padding
    outside = np.any((xy < lower) | (xy > upper), axis=1)
    if not outside.any():
        return
    ax.set_xlim(lower[0], upper[0]); ax.set_ylim(lower[1], upper[1])
    inset = ax.inset_axes([.80, .70, .19, .29])
    inset.scatter(xy[:,0], xy[:,1], color='#777777', s=.15, linewidths=0)
    inset.add_patch(Rectangle(lower, *(upper-lower), fill=False, edgecolor='black', linewidth=.4))
    inset.set_xticks([]); inset.set_yticks([]); inset.set_aspect('equal')
    for spine in inset.spines.values(): spine.set_linewidth(.35)


def s2():
    import matplotlib.pyplot as plt
    style()
    scores = pd.read_csv(RUN / 'evaluation/velocity_per_cell.csv')
    clusters = pd.read_csv(RUN / 'evaluation/state_clusters/agist_state_cluster_assignments.csv')
    assert np.array_equal(clusters.row_index, np.arange(len(scores)))
    np.testing.assert_array_equal(scores.samples, clusters.time)
    scores['state_cluster'] = clusters.state_cluster
    rows = []
    fig, axs = plt.subplots(2, 2, figsize=(7.8, 4.9))
    fig.subplots_adjust(left=.105, right=.975, bottom=.125, top=.90, hspace=.64, wspace=.29)
    for i, ax in enumerate(axs.flat):
        space = 'physical' if i % 2 == 0 else 'gene'
        key = 'samples' if i < 2 else 'state_cluster'
        stats = scores.groupby(key)[space + '_cosine'].agg(['median', 'count'])
        x, y = np.arange(len(stats)), stats['median'].to_numpy()
        if i < 2:
            ax.plot(x, y, color='#2166AC', marker='o', ms=3.5, lw=1.05)
            # Include every current result. Never clip new values to old limits.
            lower = min(.75, np.floor(np.nanmin(y) * 10) / 10)
            ax.set_ylim(lower, 1.015)
            ax.set_xlim(-.2, len(x) - .8)
        else:
            ax.bar(x, y, width=.34, color='#2166AC', linewidth=0)
            ax.set_ylim(min(0, np.nanmin(y) - .05), 1.04)
            ax.set_yticks([0, .25, .5, .75, 1.])
            ax.set_xlim(-.5, len(x) - .5)
        ax.set_xticks(x, [f'{v:g}' if isinstance(v, (float, int)) else str(v) for v in stats.index])
        ax.set_xlabel('Time' if i < 2 else 'State-space cluster')
        if i % 2 == 0:
            ax.set_ylabel('Median cosine similarity')
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis='y', color='#dddddd', linewidth=.5)
        b = ax.get_position()
        fig.text(b.x0 - .045, b.y1 + .045, 'abcd'[i], fontweight='bold', fontsize=12)
        fig.text(b.x0, b.y1 + .045, 'Spatial velocity' if space == 'physical' else 'Gene-state velocity')
        rows.extend(dict(panel='abcd'[i], space=space, partition=key, category=k,
            median=float(r['median']), n_cells=int(r['count'])) for k, r in stats.iterrows())
    pd.DataFrame(rows).to_csv(RUN / 'evaluation/S2_medians.csv', index=False)
    save(fig, 'S2_corrected')
    plt.close(fig)


def statistics():
    from scipy.stats import spearmanr
    from reproduction.agist.main_figure import calculate_attention_display
    frame = pd.read_csv(RUN / 'data/mouse_brain_simulation.csv')
    correlations = []
    for t in (0, 1, 2, 3):
        xy = frame.loc[frame.samples == t, ['x1', 'x2']].to_numpy()
        suffix = 'attention.npz' if t == 0 else f'attention_time{t}.npz'
        strengths = []
        for role in ('truth', 'predicted'):
            edges = dict(np.load(RUN / f'evaluation/{role}/{suffix}'))
            strengths.append(calculate_attention_display(edges, xy)['strength'])
        mask = (strengths[0] != 0) | (strengths[1] != 0)
        correlations.append(dict(time=t, n_cells=len(xy), n_compared=int(mask.sum()),
            spearman=float(spearmanr(strengths[0][mask], strengths[1][mask]).statistic)))
    pd.DataFrame(correlations).to_csv(RUN / 'evaluation/attention_by_time.csv', index=False)
    table = pd.read_csv(RUN / 'benchmark/w2_replicates.csv')
    tests = []
    for space in ('gene', 'physical'):
        for t in (1, 2, 3):
            selection = table[(table.space == space) & (table.time == t)]
            cb = selection[selection.method == 'CytoBridge'].set_index('seed').loc[SEEDS, 'w2'].to_numpy()
            for method in (('stVCR', 'STORIES') if space == 'gene' else ('stVCR',)):
                other = selection[selection.method == method].set_index('seed')['w2']
                difference = cb - (other.loc[SEEDS].to_numpy() if method == 'stVCR' else other.iloc[0])
                signs = np.array(list(itertools.product((-1, 1), repeat=len(difference))))
                null = np.abs((signs * difference).mean(axis=1))
                p = float(np.mean(null >= abs(difference.mean()) - 1e-14))
                tests.append(dict(space=space, time=t, comparison=method,
                    mean_cb_minus_baseline=float(difference.mean()), p_two_sided=p))
    p = np.array([r['p_two_sided'] for r in tests]); order = np.argsort(p)
    adjusted = np.minimum(1, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    for j, value in zip(order, adjusted):
        tests[j]['p_holm'] = float(value)
    pd.DataFrame(tests).to_csv(RUN / 'benchmark/sign_flip_tests.csv', index=False)
    # These tests concern stochastic inference conditional on one training fit.
    rows = []
    truth = np.load(RUN / 'data/generated_populations.npz')
    for method in ('CytoBridge', 'stVCR', 'STORIES'):
        seeds = SEEDS if method != 'STORIES' else [42]
        for seed in seeds:
            filename = (RUN / f'cytobridge/predictions/seed_{seed}.npz' if method == 'CytoBridge'
                else RUN / f'stvcr/seed_{seed}.npz' if method == 'stVCR'
                else RUN / 'stories/prediction.npz')
            pred = np.load(filename)
            for t in (1, 2, 3):
                true_mass = len(truth[f't{t}']) / 5000
                predicted_mass = len(pred[f't{t}']) / 5000
                rows.append(dict(method=method, seed=seed, time=t, true_mass=true_mass,
                    predicted_mass=predicted_mass, relative_mass_error=abs(predicted_mass-true_mass)/true_mass))
    pd.DataFrame(rows).to_csv(RUN / 'benchmark/mass_replicates.csv', index=False)


def s1():
    import matplotlib.pyplot as plt
    from reproduction.agist.main_figure import calculate_attention_display
    style()
    frame = pd.read_csv(RUN / 'data/mouse_brain_simulation.csv')
    fields = {r: dict(np.load(RUN / f'evaluation/{r}/fields.npz')) for r in ('truth', 'predicted')}
    # Both models are evaluated at the same observed cell states.
    limits = np.percentile(np.concatenate([v['growth'] for v in fields.values()]), [1, 99])
    fig = plt.figure(figsize=(8.27, 8.3))
    for t in range(4):
        for j, role in enumerate(('predicted', 'truth')):
            ax = fig.add_axes([.04 + (t % 2) * .48 + j * .225, .745 - (t // 2) * .22, .205, .175])
            mask = frame.samples == t
            xy = frame.loc[mask, ['x1', 'x2']].to_numpy()
            points = ax.scatter(xy[:, 0], xy[:, 1], c=fields[role]['growth'][mask],
                cmap='coolwarm', vmin=limits[0], vmax=limits[1], s=.6, linewidths=0)
            ax.set_aspect('equal'); ax.set_axis_off()
            spatial_view(ax, xy)
            ax.set_title(('CytoBridge' if role == 'predicted' else 'Ground truth') + f', T={t}', fontsize=9)
    fig.text(.02, .958, 'a', fontweight='bold', fontsize=13)
    fig.text(.12, .958, 'Growth rate')
    cax = fig.add_axes([.96, .62, .011, .23]); cb = fig.colorbar(points, cax=cax)
    cb.ax.tick_params(labelsize=7)
    cb.ax.yaxis.set_ticks_position('right')
    attention_max = []
    attention_axes = []
    for row, t in enumerate((2, 3)):
        xy = frame.loc[frame.samples == t, ['x1', 'x2']].to_numpy()
        for col, role in enumerate(('predicted', 'truth')):
            edges = dict(np.load(RUN / f'evaluation/{role}/attention_time{t}.npz'))
            flow = calculate_attention_display(edges, xy)
            ax = fig.add_axes([.04 + col * .285, .265 - row * .225, .25, .185])
            attention_axes.append((ax, flow, xy))
            attention_max.append(np.percentile(flow['strength'], 95))
            ax.set_title(('CytoBridge' if role == 'predicted' else 'Ground truth') + f', T={t}', fontsize=9)
    for ax, flow, xy in attention_axes:
        points = ax.scatter(xy[:, 0], xy[:, 1], c=flow['strength'], cmap='Reds',
            vmin=0, vmax=max(attention_max), s=.6, linewidths=0)
        ax.streamplot(flow['X'], flow['Y'], flow['U'], flow['V'], color='black', density=1., linewidth=.55, arrowsize=.6)
        ax.set_aspect('equal'); ax.set_axis_off()
        spatial_view(ax, xy)
    fig.text(.02, .49, 'b', fontweight='bold', fontsize=13)
    cax = fig.add_axes([.598, .115, .01, .27]); cb = fig.colorbar(points, cax=cax)
    cb.set_ticks([]); cb.set_label('Attention strength', fontsize=8)
    table = pd.read_csv(RUN / 'benchmark/w2_replicates.csv')
    mass = pd.read_csv(RUN / 'benchmark/mass_replicates.csv')
    palette = {'STORIES': '#E1CCBB', 'stVCR': '#C18071', 'CytoBridge': '#8B3842'}
    for i, (data, value, title) in enumerate(((table[table.space == 'physical'], 'w2', 'Spatial Wasserstein-2'),
            (mass, 'relative_mass_error', 'Relative mass error'))):
        ax = fig.add_axes([.72, .295 - i * .235, .25, .16])
        methods = ['stVCR', 'CytoBridge'] if i == 0 else list(palette)
        width = .26
        for j, method in enumerate(methods):
            stats = data[data.method == method].groupby('time')[value].agg(['mean', 'std'])
            ax.bar(np.arange(1, 4) + (j - (len(methods)-1)/2)*width, stats['mean'], width,
                yerr=stats['std'].fillna(0), color=palette[method], edgecolor='black', linewidth=.35,
                capsize=1.4, error_kw={'elinewidth': .6}, label=method)
        ax.set_xticks([1, 2, 3], ['T=1', 'T=2', 'T=3']); ax.set_title(title, fontsize=9)
        ax.spines[['top', 'right']].set_visible(False)
        if i == 1: ax.legend(frameon=False, fontsize=6.5)
    fig.text(.665, .49, 'c', fontweight='bold', fontsize=13)
    save(fig, 'S1_corrected'); plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['attention', 's1', 's2', 'statistics'])
    p.add_argument('--role', choices=['truth', 'predicted'])
    p.add_argument('--device', default='cuda:2')
    args = p.parse_args()
    if args.stage == 'attention': attention(args.role, args.device)
    else: globals()[args.stage]()
