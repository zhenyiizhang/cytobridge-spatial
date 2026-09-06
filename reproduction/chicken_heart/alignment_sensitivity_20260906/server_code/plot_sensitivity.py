"""Draw S7/S8 using the measured results and actual perturbation amplitudes."""
from pathlib import Path
import argparse
import json
import math
import sys

import numpy as np
from matplotlib.ticker import MaxNLocator

HERE = Path(__file__).resolve().parent
PLOTTING = HERE.parents[1]/'worktrees/package-final-audit-20260829/reproduction/supplementary_figures'
sys.path.insert(0,str(PLOTTING))
import plot_panels as panels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir',type=Path,default=HERE/'summary')
    parser.add_argument('--manifest',type=Path,default=HERE/'input_manifest.json')
    parser.add_argument('--output-dir',type=Path,default=HERE/'figures')
    parser.add_argument('--figures',nargs='+',type=int,choices=(7,8),default=[7,8])
    args = parser.parse_args()
    model = panels.load('heart')
    model.RESULTS_DIR = args.results_dir.resolve()
    model.FIGURE_DIR = args.output_dir.resolve()
    model.FIGURE_DIR.mkdir(parents=True,exist_ok=True)
    manifest = json.loads(args.manifest.read_text())
    shifts,angles = {},{}
    labels = {'baseline_repeat':'Unperturbed'}
    for record in manifest['variants']:
        name = record['variant']
        stages = record['stage_records']
        assert {s['timepoint'] for s in stages} == {'D4','D7','D10','D14'}
        shift = np.array([math.hypot(s['translate_x_nn'],s['translate_y_nn']) for s in stages])
        angle = np.array([abs(s['rotation_deg']) for s in stages])
        expected_shift = 0 if name.startswith('rotate_') else (1 if name.endswith('_low') else 2)
        expected_angle = 0 if name.startswith('translate_') and not name.startswith('translate_rotate_') else (1 if name.endswith('_low') else 3)
        np.testing.assert_allclose(shift,expected_shift,rtol=0,atol=1e-12)
        np.testing.assert_allclose(angle,expected_angle,rtol=0,atol=1e-12)
        shifts[name],angles[name] = expected_shift,expected_angle
        if expected_shift and expected_angle:
            labels[name] = f'Combined\n{expected_shift}×, {expected_angle}°'
        elif expected_shift:
            labels[name] = f'Translation {expected_shift}×'
        else:
            labels[name] = f'Rotation {expected_angle}°'
    model.DISPLAY_VARIANTS = tuple((name,labels[name]) for name in (
        'baseline_repeat','translate_low','translate_moderate','rotate_low',
        'rotate_moderate','translate_rotate_low','translate_rotate_moderate'))
    model.perturbation_sizes = lambda:(shifts,angles)
    original_bars = model.perturbation_bars

    def integer_bars(ax,values,**kwargs):
        kwargs['digits'] = 0
        kwargs['ylabel'] = 'Shift (median-neighbor units)' if kwargs['title']=='Translation' else 'Rotation (°)'
        original_bars(ax,values,**kwargs)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    model.perturbation_bars = integer_bars
    model.TRANSLATION_COLOR,model.ROTATION_COLOR = panels.BLUE,panels.RED
    model.configure_style()
    original_heatmap = model.heatmap

    def full_range(ax,table,value,title,**kwargs):
        values = model.value_matrix(table,value)
        if not np.isfinite(values).all():
            raise ValueError(f'Non-finite values in {title}')
        if kwargs.get('percent'):
            kwargs['vmax'] = max(kwargs['vmax'],math.ceil(float(values.max())*20)/20)
        else:
            kwargs['vmin'] = min(kwargs['vmin'],math.floor(float(values.min())*10)/10)
            kwargs['vmax'] = max(kwargs['vmax'],float(values.max()))
        original_heatmap(ax,table,value,title,**kwargs)

    model.heatmap = full_range

    def format_panels(fig):
        for ax in fig.axes:
            if ax.get_title() in ('Translation','Rotation'):
                for patch in ax.patches:
                    center = patch.get_x()+patch.get_width()/2
                    patch.set_width(.38)
                    patch.set_x(center-.19)
            for text in ax.texts:
                if text.get_text().startswith('* Values are rounded'):
                    text.set_text('* Rounded to two decimal places. A value marked 1.00* is below one.')

    arrays = model.load_plot_inputs()
    # The overview and sensitivity calculations use the same new control.
    arrays['accepted_aligned_xy'] = arrays['baseline_repeat__aligned_xy']
    with panels.edit_before_save(format_panels):
        for number in args.figures:
            getattr(model,f'plot_s{number}')(arrays)
    import fitz
    combined = fitz.open()
    for number in args.figures:
        with fitz.open(model.FIGURE_DIR/f'heart_alignment_sensitivity_S{number}_final.pdf') as page:
            combined.insert_pdf(page)
    preview = args.output_dir.parent/('_'.join(f'S{number}' for number in args.figures)+'_small_perturbations_preview.pdf')
    combined.save(preview)
    combined.close()
    print(preview)


if __name__=='__main__':
    main()
