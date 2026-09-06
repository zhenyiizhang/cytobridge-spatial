"""Keep tutorial renderers in step with the accepted supplementary figures."""
import ast
from pathlib import Path
import json
import shutil
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]

def notebook_source(name):
    notebook = json.loads((ROOT / 'docs/tutorials' / f'{name}.ipynb').read_text())
    return '\n'.join(''.join(c['source']) for c in notebook['cells'])

@pytest.mark.parametrize('name,numbers', [
    ('agist_figures','[2, 3]'), ('nonspatial_figures','[4, 5]'),
    ('classifier_smoothing','[6]'), ('arista_local_domains','[25]'),
    ('zebrafish_attention','[39]'), ('lr_complex_aggregation','[41]'),
    ('interaction_ablation','[42]'), ('spatial_communication','[43]'),
    ('loto_benchmark_summary','[44]'), ('loto_benchmark','[45]'),
    ('training_histories','[46]'),
])
def test_tutorial_uses_current_numerical_renderer(name,numbers):
    source = notebook_source('paper_figures/'+name)
    assert 'from reproduction.paper_figures import' in source
    assert f'draw_supplementary({numbers},' in source

def test_lr_tutorial_uses_current_ranking_cutoff():
    source = notebook_source('paper_figures/lr_complex_aggregation')
    assert 'top 100 LR pairs' in source
    assert 'min(10,' not in source
    assert 'top_n=100' in source
    assert 'results_dir=results.source_dir' in source

def test_current_lr_summaries_use_requested_cutoff():
    from CytoBridge.results.lr_complex_aggregation import load_lr_complex_aggregation_results
    results = load_lr_complex_aggregation_results(top_n=100)
    assert 'min_top100_jaccard' in results.dataset_summary
    assert 'min_top10_jaccard' not in results.dataset_summary
    table = results.per_time_summary.query("scope == 'all_scored_pairs'")
    finite = table.loc[table.top_jaccard.notna()]
    assert finite.top_n.max() == 100
    assert finite.top_n.le(100).all()

def test_s42_current_renderer_uses_new_inference_results(tmp_path):
    from CytoBridge.results.interaction_ablation import load_interaction_ablation_results
    from reproduction.paper_figures import draw_supplementary
    source = load_interaction_ablation_results().source_dir
    inputs = tmp_path/'inputs'
    shutil.copytree(source, inputs)
    path = inputs/'inference_metrics.csv'
    raw = pd.read_csv(path)
    raw.loc[raw.arm.eq('interaction_off'), 'sliced_w2'] *= 1.25
    raw.to_csv(path, index=False)
    expected = load_interaction_ablation_results(inputs).interaction
    output = tmp_path/'figures'
    figures = draw_supplementary([42], output, results_dir=inputs)
    plotted = pd.read_csv(output/'tables/S42_interaction_off.csv')
    np.testing.assert_allclose(plotted.off_relative_to_on, expected.off_relative_to_on)
    np.testing.assert_allclose(plotted.change, 100*expected.off_relative_to_on)
    assert all(path.is_file() for path in figures['s42'])

def test_training_continues_at_real_generic_analysis_section():
    training = (ROOT/'docs/training.md').read_text()
    source = notebook_source('model_analysis')
    assert '(tutorials/model_analysis.ipynb)' in training
    assert '## Calculate velocity components' in source
    assert 'cb.tl.compute_velocity_components(' in source
    assert 'cb.tl.evaluate_growth_by_timepoint(' in source
    assert 'velocity_components.npz' in source
    assert 'growth_by_cell.csv' in source
    assert 'cb.pl.plot_velocity_component(' not in source
    assert 'cb.pl.plot_growth_timepoint_grid(' not in source
    assert 'paper_figures/chicken_heart_daily.ipynb' in source
    assert 'Image(filename=' not in source

def test_current_renderer_rejects_source_overwrite(tmp_path):
    from reproduction.paper_figures import draw_supplementary
    with pytest.raises(ValueError):
        draw_supplementary([41], ROOT/'reproduction/temporary')
    with pytest.raises(ValueError):
        draw_supplementary([99], tmp_path)

def test_s12_uses_accepted_layout_without_changing_growth_values():
    pytest.importorskip('matplotlib')
    import matplotlib.pyplot as plt
    from reproduction.mosta.growth import DISPLAY_TIMES, plot_brain_growth
    cells = pd.DataFrame([
        {'time': t, 'x': float(i), 'y': float(i%2), 'growth': (i-1)*.2}
        for t in np.arange(0,3.01,.25) for i in range(4)
    ])
    original = cells.copy()
    fig, summary = plot_brain_growth(cells, vmin=-.1, vmax=.5)
    try:
        assert len(fig.axes)==13
        assert tuple(summary.time)==DISPLAY_TIMES
        assert summary.cells.eq(4).all()
        pd.testing.assert_frame_equal(cells,original)
        for ax,time in zip(fig.axes[:12],DISPLAY_TIMES):
            assert ax.get_title()==f't = {time:g}'
            assert ax.collections[0].get_cmap().name=='viridis'
            assert not ax.axison
            assert not ax.collections[0].get_rasterized()
        assert not fig.axes[-1].collections[-1].get_rasterized()
    finally:
        plt.close(fig)
