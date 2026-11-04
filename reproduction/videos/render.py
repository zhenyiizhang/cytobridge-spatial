"""Draw Supplementary Videos 1–3 from reproduction.videos.simulate outputs.

MOSTA and ARISTA reuse the submitted video drawing code. AD uses the original
manuscript's spatial colors with the submitted single-panel layout and orientation.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse
import json
import subprocess
import os
import shutil

os.environ.setdefault('MPLBACKEND', 'Agg')
import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from . import _mosta_style as mosta
from .support import attach_mosta_support

CODE = Path(__file__).resolve().parent
NAMES = {
    'mosta': 'Supplementary_Video_1_Figure4_Mouse_Organogenesis.mp4',
    'arista': 'Supplementary_Video_2_Figure5_Axolotl_Regeneration.mp4',
    'admouse': 'Supplementary_Video_3_Figure6_Alzheimer_Progression.mp4',
}
AD_COLORS = dict(zip(
    ['Astrocytes', 'Excitatory neurons', 'Fibroblast', 'Inhibitory neurons',
     'Microglia', 'OPC', 'Oligodendrocytes', 'Pericytes/Endothelial'],
    ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
     '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']))


def load_frames(root):
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest.get('spatial_warp') is not False:
        raise ValueError('Use the unwarped output of reproduction.videos.simulate.')
    frames = []
    for path in sorted((root / 'states').glob('frame_*.npz')):
        with np.load(path) as data:
            frames.append(dict(time=float(data['time']),
                               xy=data['points'][:, :2].copy(),
                               labels=data['labels'].copy()))
    expected = len(manifest['time_points'])
    if not frames or len(frames) != expected:
        raise ValueError(f'Incomplete trajectory: {len(frames)} of {expected} frames')
    return frames


def check_colors(frames, colors):
    labels = set(np.concatenate([f['labels'] for f in frames]))
    missing = labels - set(colors)
    if missing:
        raise ValueError(f'Cell types missing from original palette: {sorted(missing)}')


def draw_mosta(frames, output, communications, limits=None):
    colors = json.loads((CODE / 'mosta_colors.json').read_text())
    check_colors(frames, colors)
    config = SimpleNamespace(**json.loads((CODE / 'mosta_style.json').read_text()))
    # One camera for the entire trajectory, including all simulated cells.
    xlim, ylim = limits if limits is not None else mosta._compute_xy_limits([frame['xy'] for frame in frames], pad_frac=.03)
    for i, frame in enumerate(frames):
        xy, labels = frame['xy'], frame['labels']
        mosta._render_frame_single_2d(
            xy=xy, labels=labels, time_value=frame['time'],
            label_to_color=colors, out_png=output / f'frame_{i:03d}.png',
            xlim=xlim, ylim=ylim, point_size=2.5, alpha=.9,
            dpi=300, panel_size=4.2, title='', show_titles=False,
            comm_result=communications[str(i)], config=config)


def draw_arista(frames, output, communications):
    colors = json.loads((CODE / 'arista_colors.json').read_text())
    check_colors(frames, colors)
    populations = {}
    times = [f['time'] for f in frames]
    for frame in frames:
        a = ad.AnnData(X=frame['xy'])
        a.obsm['spatial'] = frame['xy']
        a.obs['Annotation'] = frame['labels']
        populations[str(frame['time'])] = a
    namespace = dict(
        np=np, plt=plt, os=os, Path=Path,
        adata_dict=populations, time_keys=list(populations), ts_points=times,
        all_time_communications={str(t): communications[str(i)] for i,t in enumerate(times)},
        frames_dir=str(output), label_to_color=colors, rng_seed=42,
        focus_celltype='reaEGC', focus_mode='both', focus_top_k=3,
        focus_min_weight=0., focus_edge_quantile=None, focus_neighbor_pct=.1,
        focus_edge_width=(1.,2.6), focus_edge_alpha=(.25,.85),
        focus_edge_curve=.18, focus_edge_curve_bi=.35,
        focus_halo_scale=3., focus_other_desaturate=.3,
        focus_show_endpoints=True, focus_endpoint_size=12, focus_endpoint_alpha=.9,
        focus_zoom_enabled=True, focus_zoom_mode='box_inset',
        focus_zoom_padding_frac=.06, focus_zoom_position='right_upper',
        focus_zoom_size='40%', focus_zoom_outside=True,
        focus_zoom_outside_pad=.01, focus_zoom_fixed=True, focus_zoom_fixed_mode='global',
        point_size=3.5, point_alpha=.7, frame_point_subsample=None,
        background_color='white', draw_comm_arrows=False, draw_focus_interactions=True)
    exec(compile((CODE / '_arista_style.py').read_text(),
                 str(CODE / '_arista_style.py'), 'exec'), namespace)


def draw_ad(frames, output):
    check_colors(frames, AD_COLORS)
    # The submitted video uses the horizontal aligned coordinates. The static
    # manuscript script additionally rotates the tissue, which is not used here.
    coords = [f['xy'] for f in frames]
    xlim, ylim = mosta._compute_xy_limits(coords, pad_frac=.025)
    # Use the main-figure cell-type colors without an extra saturation change.
    colors = AD_COLORS
    for i, (frame, xy) in enumerate(zip(frames, coords)):
        fig = plt.figure(figsize=(7.2, 6.16), dpi=200)
        ax = fig.add_axes([.025, .025, .95, .87])
        ax.scatter(xy[:, 0], xy[:, 1], s=3.,
                   c=[colors[label] for label in frame['labels']],
                   alpha=.75, linewidths=0, edgecolors='none')
        ax.set(xlim=xlim, ylim=ylim, aspect='equal')
        ax.axis('off')
        age = np.interp(frame['time'], [0., 1., 2.], [2.5, 5.7, 17.9])
        fig.text(.5, .98, f'{age:.1f} months', ha='center', va='top', fontsize=22)
        fig.savefig(output / f'frame_{i:03d}.png', dpi=200)
        plt.close(fig)


def crop_box(paths):
    boxes = []
    for path in paths:
        im = np.asarray(Image.open(path).convert('RGB'))
        y, x = np.where(np.any(im < 245, axis=2))
        boxes.append((x.min(), y.min(), x.max()+1, y.max()+1))
    boxes = np.asarray(boxes)
    x, y = np.maximum(0, boxes[:, :2].min(axis=0)-12)
    end = np.minimum([im.shape[1], im.shape[0]], boxes[:, 2:].max(axis=0)+12)
    return int(x), int(y), int(end[0]-x), int(end[1]-y)


def encode(dataset, frames, output, crf=20):
    width, height, input_fps, output_fps = {
        'mosta': (1260,1260,10,20),
        'arista': (1904,1288,8,8),
        'admouse': (1440,1232,10,10),
    }[dataset]
    filters = []
    if dataset == 'arista':
        x, y, w, h = crop_box(sorted(frames.glob('frame_*.png')))
        filters.append(f'crop={w}:{h}:{x}:{y}')
    filters += [f'scale={width}:{height}:force_original_aspect_ratio=decrease',
                f'pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:white',
                f'fps={output_fps}', 'setsar=1']
    path = output / NAMES[dataset]
    if path.exists():
        raise FileExistsError(path)
    import imageio_ffmpeg
    ffmpeg = shutil.which('ffmpeg') or imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([ffmpeg,'-v','error','-framerate',str(input_fps),
                    '-i',str(frames/'frame_%03d.png'),'-vf',','.join(filters),
                    '-c:v','libx264','-preset','slow','-crf',str(crf),'-threads','4',
                    '-pix_fmt','yuv420p','-movflags','+faststart',str(path)], check=True)
    (output / 'encoding.json').write_text(json.dumps({
        'video': path.name, 'width': width, 'height': height,
        'input_fps': input_fps, 'output_fps': output_fps,
        'filters': filters, 'codec': 'H.264', 'crf': crf,
        'pixel_format': 'yuv420p', 'faststart': True,
    }, indent=2)+'\n')
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', choices=NAMES)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--encode-only', action='store_true')
    parser.add_argument('--crf', type=int,
                        help='Encoding quality: default 20 for MOSTA, 24 for ARISTA and AD.')
    args = parser.parse_args()
    if args.crf is None:
        args.crf = 20 if args.dataset == 'mosta' else 24
    if args.encode_only:
        print(encode(args.dataset, args.output / 'frames', args.output, args.crf), flush=True)
        return
    frames = load_frames(args.input)
    args.output.mkdir(parents=True, exist_ok=False)
    directory = args.output / 'frames'
    directory.mkdir()
    communications = json.loads((args.input/'communication.json').read_text())
    # JSON arrays are restored for the unchanged drawing functions.
    for comm in communications.values():
        comm['M_per_source'] = np.asarray(comm['M_per_source'])
    if args.dataset == 'mosta':
        records = attach_mosta_support(args.input, frames, communications)
        (args.output / 'edge_support.json').write_text(json.dumps(records, indent=2) + '\n')
        draw_mosta(frames, directory, communications)
    elif args.dataset == 'arista': draw_arista(frames, directory, communications)
    else: draw_ad(frames, directory)
    path = encode(args.dataset, directory, args.output, args.crf)
    (args.output/'render_manifest.json').write_text(json.dumps({
        'input': str(args.input.resolve()), 'video': str(path.resolve()),
        'frame_count_before_encoding': len(frames),
        'style': 'Submitted layout and cell-type colors',
        'spatial_warp': False,
        'mosta_arrow_selection': 'Top three Brain-related pairs, omitting pairs with fewer than two distinct cell edges without replacement.',
        'mosta_camera': 'Global x/y limits across all frames, no random display subsampling, no trajectory smoothing or coordinate warp.',
        'original_mosta_code': 'downstream_helpers/mosta_baseline_video.py',
        'original_arista_code': 'evaluation/arista_code/arista_spatiotemporal_animation.ipynb cell 10',
        'ad_spatial_code': 'reproduction/admouse/final_figures/main/scripts/ad_main_figureb.py',
    }, indent=2)+'\n')
    print(path, flush=True)


if __name__ == '__main__': main()
