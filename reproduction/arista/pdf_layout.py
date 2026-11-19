"""Place the Figure 5a PDF without resampling its 3D image layer."""
from pathlib import Path
from io import BytesIO

import numpy as np
from pypdf import PdfReader, PdfWriter, PageObject, Transformation


def draw_injury_arrow(page):
    """Point to the dashed injury outline on the observed 2-DPI section."""
    import fitz
    tail, tip = np.array([48., 348.]), np.array([96., 366.])
    direction = (tip - tail) / np.linalg.norm(tip - tail)
    normal = np.array([-direction[1], direction[0]])
    shape = page.new_shape()
    shape.draw_line(fitz.Point(*tail), fitz.Point(*tip))
    shape.draw_polyline([fitz.Point(*(tip - 6 * direction + 3 * normal)),
                         fitz.Point(*tip), fitz.Point(*(tip - 6 * direction - 3 * normal))])
    shape.finish(color=(.16, .13, .52), width=1.7, closePath=False)
    shape.commit()


def place_stack(source_pdf, labels_pdf, output_pdf, output_png=None):
    source = PdfReader(source_pdf).pages[0]
    labels = PdfReader(labels_pdf).pages[0]
    width, height = float(source.mediabox.width), float(source.mediabox.height)
    page_width, page_height = float(labels.mediabox.width), float(labels.mediabox.height)
    # Convert the exported PDF to the original layout's coordinate system.
    to_canvas = np.array([[7014 / width, 0, 0], [0, -4962 / height, 4962], [0, 0, 1]])
    placement = np.array([[1.16519507, -.00272857794, -1326.42311],
                          [-.00257634855, 1.17106427, -26.494962], [0, 0, 1]])
    x0, y0, x1, y1 = 23.7819900513, 14.293762207, 441.495361328, 434.780700684
    to_page = np.array([[(x1-x0)/5723, 0, x0],
                        [0, -(y1-y0)/5761, page_height-y0], [0, 0, 1]])
    matrix = to_page @ placement @ to_canvas
    transform = Transformation((matrix[0, 0], matrix[1, 0], matrix[0, 1],
                                matrix[1, 1], matrix[0, 2], matrix[1, 2]))
    page = PageObject.create_blank_page(width=page_width, height=page_height)
    page.merge_transformed_page(source, transform)
    page.merge_page(labels)
    writer = PdfWriter()
    writer.add_page(page)
    import fitz
    buffer = BytesIO()
    writer.write(buffer)
    with fitz.open(stream=buffer.getvalue(), filetype='pdf') as document:
        draw_injury_arrow(document[0])
        document.save(output_pdf, garbage=4, deflate=True)
    if output_png is not None:
        import fitz
        with fitz.open(output_pdf) as document:
            document[0].get_pixmap(dpi=300, alpha=False).save(output_png)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-pdf', type=Path, required=True)
    parser.add_argument('--labels-pdf', type=Path, default=Path(__file__).parent / 'data/figure5a_labels.pdf')
    parser.add_argument('--output-pdf', type=Path, required=True)
    args = parser.parse_args()
    place_stack(args.source_pdf, args.labels_pdf, args.output_pdf, args.output_pdf.with_suffix('.png'))
