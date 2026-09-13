"""Arrange the four calculated brain fields in the Figure 4e layout."""
from importlib.util import module_from_spec, spec_from_file_location
import json

import fitz
from matplotlib import font_manager

from .figures import SOURCE
from .brain_arrows import ARROWS, draw_arrow


def source_module(name):
    path = SOURCE / 'main_fig4_panels/fig4e/source' / f'{name}.py'
    spec = spec_from_file_location(name, path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save_brain_field(fig, ax, name, output):
    """Use the original axes crop and vector-stroke export settings."""
    render = source_module('render_fig4e_exact_notebook_sources')
    for directory in ('fields', 'axes', 'layout_fields'):
        (output / directory).mkdir(parents=True, exist_ok=True)
    render.save_full_and_axes_crop(
        fig, ax, stem=f'Figure4e_{name}', raw_dir=output / 'fields',
        crop_dir=output / 'axes', ai_ready_dir=output / 'layout_fields',
        ai_ready_rect=render.AI_READY_RECTS[name],
    )
    return output / 'layout_fields' / f'Figure4e_{name}__AI_ready_stroke_parity.pdf'


def assemble_brain_fields(fields, output):
    """Place newly plotted fields, with the paper's shared labels and arrows."""
    layout = source_module('assemble_fig4e_exact_ai_layout')
    panel_root = SOURCE / 'main_fig4_panels'
    annotations = json.loads((panel_root / 'fig4e/evidence/annotation_semantics_audit.json').read_text())
    with fitz.open(panel_root / 'style_authority/Figure_mouse1.ai') as original, fitz.open() as document:
        page = document.new_page(width=original[0].rect.width, height=original[0].rect.height)
        page.show_pdf_page(page.rect, original, 0)
        # Cover the old field rectangle without removing neighboring layout objects.
        page.draw_rect(layout.NUMERICAL_FIELD_KNOCKOUT, color=(1, 1, 1), fill=(1, 1, 1), width=0)
        for name, path in fields.items():
            with fitz.open(path) as field:
                rectangle = layout.fit_inside(layout.PANEL_RECTS[name], field[0].rect)
                page.show_pdf_page(rectangle, field, 0)
        for key, weight in (('ArialFigure', 'normal'), ('ArialFigureBold', 'bold')):
            font = font_manager.findfont(font_manager.FontProperties(family='Arial', weight=weight),
                                         fallback_to_default=False)
            page.insert_font(fontname=key, fontfile=font)
        for label, box, size, bold in layout.AI_LABELS:
            box = fitz.Rect(box.x0 - 3, box.y0 - 2, box.x1 + 3, box.y1 + 3)
            result = page.insert_textbox(box, label, fontsize=size,
                fontname='ArialFigureBold' if bold else 'ArialFigure',
                color=(0, 0, 0), align=fitz.TEXT_ALIGN_CENTER)
            if result < 0:
                raise ValueError(f'Cannot place Figure 4e label: {label}')
        arrows = {item['id']: item for group in annotations['final_arrow_geometry'].values()
                  for item in group}
        for name, glyph in layout.ARROW_GLYPHS.items():
            if name in ('pi_3', 'pi_4'):
                continue
            if name in ARROWS:
                draw_arrow(page, glyph, *ARROWS[name])
            else:
                layout.draw_annotation_arrow(page, glyph, arrows[name]['tail'], arrows[name]['tip'])
        for label, point in layout.TISSUE_LABELS:
            page.insert_text(point, label, fontname='ArialFigure', fontsize=12, color=(0, 0, 0))
        for label, rectangle, origin, size, color in layout.CALLOUTS:
            page.draw_rect(rectangle, color=color, fill=color, width=0)
            page.insert_text(origin, label, fontname='ArialFigureBold', fontsize=size,
                             color=layout.CALLOUT_TEXT)
        page.draw_rect(layout.OUTER_BORDER, color=(35/255, 24/255, 21/255), width=2)
        from .main_figure import save_page
        return save_page(document, layout.AI_CROP, output, 'Figure4e_brain_velocity')
