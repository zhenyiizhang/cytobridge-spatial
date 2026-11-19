"""Check the accepted video settings and portable drawing entry points."""
import ast
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from reproduction.videos import _mosta_style, render, support, zebrafish

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('module', ['simulate', 'render', 'zebrafish'])
def test_video_commands_have_help(module):
    result = subprocess.run([sys.executable, '-m', f'reproduction.videos.{module}', '--help'],
                            cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_simulation_disables_both_warp_options():
    tree = ast.parse((ROOT / 'reproduction/videos/simulate.py').read_text())
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and n.func.attr == 'run_interpolation_workflow']
    assert len(calls) == 1
    keywords = {kw.arg: kw.value for kw in calls[0].keywords}
    for key in ('spatial_warp_to_observed', 'spatial_warp_to_observed_piecewise',
                'use_real_for_observed'):
        assert ast.literal_eval(keywords[key]) is False


def test_support_filter_preserves_scores_and_never_promotes_a_pair(tmp_path):
    labels = np.array(['Brain', 'A', 'B', 'B', 'C', 'C', 'D'])
    matrix = np.zeros((5, 5))
    matrix[0] = [0, 4, 3, 2, 1]
    comm = {'0': dict(types=['Brain', 'A', 'B', 'C', 'D'], M_per_source=matrix.copy())}
    attention = tmp_path / 'attention'
    attention.mkdir()
    np.save(attention / 'edge_index_interp_t0.0.npy', np.array([[0, 0, 0, 0, 0, 0], [1, 2, 3, 4, 5, 6]]))
    np.save(attention / 'attn_mean_interp_t0.0.npy', np.ones(6))
    rows = support.attach_mosta_support(tmp_path, [dict(time=0., labels=labels)], comm)
    assert [r['cell_edges'] for r in rows[0]['top3']] == [1, 2, 2]
    assert _mosta_style._get_focus_edges(comm['0'], focus_type='Brain', top_k=3) == [
        (3., 'Brain', 'B'), (2., 'Brain', 'C')]
    np.testing.assert_array_equal(comm['0']['M_per_source'], matrix)


def test_reader_preserves_coordinates_and_rejects_warped_input(tmp_path):
    states = tmp_path / 'states'
    states.mkdir()
    points = np.array([[0.1, 0.2, 3.], [0.3, 0.4, 5.]])
    np.savez(states / 'frame_000.npz', points=points, labels=['A', 'B'], time=0.)
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps(dict(time_points=[0.], spatial_warp=False)))
    frames = render.load_frames(tmp_path)
    np.testing.assert_array_equal(frames[0]['xy'], points[:, :2])
    manifest.write_text(json.dumps(dict(time_points=[0.], spatial_warp=True)))
    with pytest.raises(ValueError, match='unwarped'):
        render.load_frames(tmp_path)


def test_evl_r2_keeps_final_position(monkeypatch):
    box = dict(x0=4., y0=3., w=1., h=1., score=1.)
    monkeypatch.setattr(zebrafish, 'stable_diff_boxes', lambda *a, **kw: [box.copy()])
    frames = [np.zeros((1, 52))] * 81
    schedule = zebrafish.roi_schedule('EVL', frames, frames, (0., 10.), (0., 10.))
    r2 = schedule[-1]['boxes'][1]
    assert r2['x0'] == pytest.approx(2.9)
    assert r2['w'] == box['w']
    assert r2['y0'] == box['y0']
    assert schedule[-1]['t_min'] == pytest.approx(3.36)


def test_style_contains_only_drawing_parameters():
    settings = json.loads((ROOT / 'reproduction/videos/mosta_style.json').read_text())
    assert all(k.startswith('focus_') or k == 'draw_focus_interactions' for k in settings)


def test_documented_panel_b_input_matches_renderer():
    page = (ROOT / 'docs/reference/figure_sources/main-figure-5-reference.md').read_text()
    assert 'slice_data/time_0p5.h5ad' in page
    assert 'display_states/time_0p5.h5ad' not in page
