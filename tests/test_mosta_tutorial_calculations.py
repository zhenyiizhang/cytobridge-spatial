"""The MOSTA tutorial calculates values before passing them to a renderer."""
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd

from reproduction.mosta.program_genes import select_representative_genes


ROOT = Path(__file__).resolve().parents[1]


def test_representative_genes_follow_variance_and_correlation_ranks():
    profiles = pd.DataFrame([[0., 1., 2.], [0., 2., 4.], [2., 1., 0.]],
                            index=['a', 'b', 'c'])
    normalized = profiles.sub(profiles.mean(axis=1), axis=0).div(
        profiles.std(axis=1, ddof=0), axis=0)
    assignments = pd.DataFrame({'profile': ['a', 'b', 'c'], 'cluster': [1, 1, 2]})
    selected = select_representative_genes(profiles, normalized, assignments, n_per_program=1)
    assert selected.gene.tolist() == ['b', 'c']
    np.testing.assert_allclose(selected.prototype_correlation, [1, 1])
    profiles.loc['a'] *= 3
    selected = select_representative_genes(profiles, normalized, assignments, n_per_program=1)
    assert selected.gene.tolist() == ['a', 'c']


def test_mosta_notebook_exposes_calculation_calls_and_uses_their_results():
    notebook = json.loads((ROOT / 'docs/tutorials/paper_figures/mosta_figures.ipynb').read_text())
    source = '\n'.join(''.join(cell['source']) for cell in notebook['cells']
                       if cell['cell_type'] == 'code')
    tree = ast.parse(source)
    calls = {ast.unparse(node.func): node for node in ast.walk(tree) if isinstance(node, ast.Call)}
    for function in ('evaluate_growth_by_timepoint', 'summarize_label_composition',
                     'summarize_temporal_gene_patterns', 'analyze_developmental_wave',
                     'compute_timepoint_communications', 'project_communication_to_lr_timecourses'):
        assert f'cb.tl.{function}' in calls
    assert 'draw_supplementary' not in calls
    assert ast.unparse(calls['plot_brain_growth'].args[0]) == 'brain'
    assert ast.unparse(calls['plot_composition'].args[0]) == 'composition'
    assert ast.unparse(calls['plot_lr_profiles'].args[0]) == 'lr.pair_timecourse'
    assert ast.unparse(calls['cb.tl.analyze_developmental_wave'].args[0]) == 'gene.expression'
    assert 'subprocess.run' in calls
    assert 'enrich_go.R' in source
    assert 'gene_programs.load_inputs' not in calls
    for name in ('cb.tl.compute_timepoint_communications',
                 'cb.tl.project_communication_to_lr_timecourses'):
        time_argument = next(keyword.value for keyword in calls[name].keywords
                             if keyword.arg == 'time_points')
        assert ast.unparse(time_argument) == 'lr_times'
    lr_grid = next(node.value for node in ast.walk(tree) if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == 'lr_times'
                           for target in node.targets))
    np.testing.assert_array_equal(
        eval(compile(ast.Expression(lr_grid), '<time-grid-test>', 'eval'), {'np': np}),
        [0, .5, 1, 1.5, 2, 2.5, 3],
    )
    for path in ('growth_by_cell_fully_generated.csv', 'brain_hvg_mean_log1p_by_time.csv',
                 'numerical_truth/seed42_M_sum/lr_pair_timecourse.csv'):
        assert path not in source


def test_lineage_export_selects_half_steps_from_the_dense_grid():
    notebook = json.loads((ROOT / 'docs/tutorials/dataset_workflows/mosta.ipynb').read_text())
    source = '\n'.join(''.join(cell['source']) for cell in notebook['cells']
                       if cell['cell_type'] == 'code')
    tree = ast.parse(source)
    assignment = next(node for node in ast.walk(tree)
                      if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == 'fixed_labels'
                              for target in node.targets))
    from types import SimpleNamespace
    times = tuple(np.arange(0, 3.001, .25))
    result = SimpleNamespace(predicted_labels_list=list(range(13)))
    selected = eval(compile(ast.Expression(assignment.value), '<lineage-test>', 'eval'),
                    {'TIMES': times, 'result': result, 'fixed_times': np.arange(0, 3.001, .5)})
    assert selected == [0, 2, 4, 6, 8, 10, 12]
