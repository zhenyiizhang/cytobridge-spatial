"""Annotation arrows for the gene and physical interaction fields in Figure 4e."""
import fitz
import numpy as np

# Endpoints in PDF points on the original figure canvas (top-left origin).
ARROWS = {
    'gi_1': ((490., 558.), (510., 530.)),
    'gi_2': ((481., 580.), (468., 551.)),
    'gi_3': ((467., 601.), (462., 570.)),
    'pi_1': ((483.54551733, 682.5), (498.26794919, 691.)),
    'pi_2': ((516.45448267, 701.5), (501.73205081, 693.)),
}

def draw_arrow(page, glyph, tail, tip):
    """Rotate and scale the original filled glyph to the specified endpoints."""
    tail, tip = np.asarray(tail), np.asarray(tip)
    old_tail, old_tip = np.asarray(glyph['tail']), np.asarray(glyph['tip'])
    old, new = old_tip - old_tail, tip - tail
    angle = np.arctan2(new[1], new[0]) - np.arctan2(old[1], old[0])
    scale = np.linalg.norm(new) / np.linalg.norm(old)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    def transform(p):
        return fitz.Point(*(tail + scale * rotation @ (np.asarray(p) - old_tail)))
    kind, points = glyph['path']
    points = [transform(p) for p in points]
    shape = page.new_shape()
    if kind == 'line':
        shape.draw_line(*points)
    else:
        shape.draw_bezier(*points)
    shape.finish(color=glyph['color'], width=5. * scale, lineCap=0, closePath=False)
    shape.commit()
    head = page.new_shape()
    points = [transform(p) for p in glyph['head']]
    head.draw_polyline(points + [points[0]])
    head.finish(color=glyph['color'], fill=glyph['color'], width=0)
    head.commit()
