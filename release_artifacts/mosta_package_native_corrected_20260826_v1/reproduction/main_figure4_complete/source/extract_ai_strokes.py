#!/usr/bin/env python3
"""Audit straight vector strokes in the Illustrator-compatible Figure 4 PDF.

This is read-only with respect to the style authority.  It resolves PDF graphics
state transforms so that candidate stroke endpoints are reported in final A4
page coordinates (PDF origin: lower left).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from pypdf import PdfReader
from pypdf.generic import ContentStream


ROOT = Path(__file__).resolve().parents[3]
AI = (
    ROOT
    / "output/mosta_main_fig4_completion_20260825_v1/archive_v1"
    / "style_authority/Figure_mouse1.ai"
)
OUT = Path(__file__).resolve().parent / "ai_straight_strokes.json"


def mul(left, right):
    """Multiply affine matrices represented as (a,b,c,d,e,f)."""
    a, b, c, d, e, f = left
    g, h, i, j, k, l = right
    return (
        a * g + c * h,
        b * g + d * h,
        a * i + c * j,
        b * i + d * j,
        a * k + c * l + e,
        b * k + d * l + f,
    )


def point(matrix, xy):
    a, b, c, d, e, f = matrix
    x, y = map(float, xy)
    return (a * x + c * y + e, b * x + d * y + f)


reader = PdfReader(str(AI))
stream = ContentStream(reader.pages[0].get_contents(), reader)
identity = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
state = {"ctm": identity, "width": 1.0, "stroke": (0.0, 0.0, 0.0), "cap": 0, "join": 0}
stack = []
path = []
strokes = []

for index, (args, op_raw) in enumerate(stream.operations):
    op = op_raw.decode("ascii")
    if op == "q":
        stack.append(state.copy())
    elif op == "Q":
        state = stack.pop()
        path = []
    elif op == "cm":
        incoming = tuple(map(float, args))
        # PDF cm pre-multiplies the current transformation matrix.
        state["ctm"] = mul(incoming, state["ctm"])
    elif op == "w":
        state["width"] = float(args[0])
    elif op == "RG":
        state["stroke"] = tuple(map(float, args))
    elif op == "G":
        grey = float(args[0])
        state["stroke"] = (grey, grey, grey)
    elif op == "J":
        state["cap"] = int(args[0])
    elif op == "j":
        state["join"] = int(args[0])
    elif op == "m":
        path = [("m", point(state["ctm"], args))]
    elif op == "l":
        path.append(("l", point(state["ctm"], args)))
    elif op in {"c", "v", "y", "re", "h"}:
        path.append((op, None))
    elif op in {"S", "s", "B", "B*", "b", "b*"}:
        points = [p for kind, p in path if kind in {"m", "l"}]
        curved = any(kind not in {"m", "l"} for kind, _ in path)
        if len(points) == 2 and not curved:
            p0, p1 = points
            length = math.dist(p0, p1)
            if length >= 80:
                strokes.append(
                    {
                        "index": index,
                        "p0": [round(v, 6) for v in p0],
                        "p1": [round(v, 6) for v in p1],
                        "length": round(length, 6),
                        "width": round(state["width"], 6),
                        "stroke_rgb": [round(v, 6) for v in state["stroke"]],
                        "cap": state["cap"],
                        "join": state["join"],
                    }
                )
        path = []
    elif op in {"n", "f", "f*", "F"}:
        path = []

OUT.write_text(json.dumps(strokes, indent=2) + "\n")
print(f"wrote {len(strokes)} long straight strokes to {OUT}")
for row in strokes:
    if (abs(row["p0"][0] - 286) < 15 or abs(row["p1"][0] - 286) < 15):
        print(row)
