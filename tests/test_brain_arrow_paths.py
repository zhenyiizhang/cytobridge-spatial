"""The manuscript annotations remain editable vector paths."""
import json
from pathlib import Path
import pytest
fitz = pytest.importorskip('fitz')
from reproduction.mosta.brain_arrows import draw_author_arrows


def test_author_arrows_render_at_panel_coordinates():
    path = Path(__file__).parents[1] / 'reproduction/mosta/brain_arrow_paths.json'
    record = json.loads(path.read_text())
    assert len(record['paths']) == 20  # shaft and head for each of ten arrows
    with fitz.open() as doc:
        page = doc.new_page(width=record['page_size'][0], height=record['page_size'][1])
        draw_author_arrows(page, (0, 0))
        stream = b''.join(doc.xref_stream(x) for x in page.get_contents())
        assert b'{lineJoin}' not in stream
        assert len(page.get_drawings()) == 20
        assert not page.get_images()
        pix = page.get_pixmap()
        assert min(pix.samples) < 240
