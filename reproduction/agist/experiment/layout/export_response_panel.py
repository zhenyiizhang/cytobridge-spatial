"""Export panel e from the approved Figure 2 PDF without redrawing its contents."""
import argparse
from pathlib import Path
import fitz

p=argparse.ArgumentParser()
p.add_argument('source',type=Path)
p.add_argument('output',type=Path)
a=p.parse_args()
with fitz.open(a.source) as source:
    clip=fitz.Rect(208,625,595.276,824)
    result=fitz.open()
    page=result.new_page(width=clip.width,height=clip.height)
    page.show_pdf_page(page.rect,source,0,clip=clip)
    result.save(a.output,deflate=True)
    page.get_pixmap(matrix=fitz.Matrix(300/72,300/72)).save(a.output.with_suffix('.png'))
