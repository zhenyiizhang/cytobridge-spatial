#!/usr/bin/env python3
"""Create a local Figure 5a HTML preview with biological foreground lifted in z.

This utility is deliberately renderer-only.  It starts from the accepted
server-generated Plotly HTML and changes only the z arrays of the five cell
point clouds, 25 communication lines, 25 communication arrowheads, six
lineage paths, and one endpoint-marker trace.  Mesh3d slice planes and the
five slice-border traces remain byte-for-byte equivalent at the JSON-value
level; x/y coordinates, scientific values, layout, and all style properties
are unchanged.
"""

from __future__ import annotations

import argparse
import base64
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any, Sequence

import numpy as np


EXPECTED_INPUT_SHA256 = "de6c804f279de3a4e1d7bc09b27b0bfd96cd1ef0afcb557bdc75285187b55986"
TIME_KEYS = {"0.0", "0.5", "1.0", "1.5", "2.0"}
EXPECTED_COUNTS = {
    "cell_points": 5,
    "communication_lines": 25,
    "communication_arrows": 25,
    "lineage_paths": 6,
    "endpoint_markers": 1,
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-html", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--foreground-z-offset", required=True, type=float)
    parser.add_argument("--expected-input-sha256", default=EXPECTED_INPUT_SHA256)
    parser.add_argument(
        "--dynamic-trace-counts",
        action="store_true",
        help=(
            "Allow corrected runs to have a different positive number of "
            "communication/lineage traces while retaining all structural checks."
        ),
    )
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _plotly_dtype(code: str) -> np.dtype:
    aliases = {
        "f4": np.dtype("<f4"),
        "f8": np.dtype("<f8"),
        "i1": np.dtype("i1"),
        "i2": np.dtype("<i2"),
        "i4": np.dtype("<i4"),
        "u1": np.dtype("u1"),
        "u2": np.dtype("<u2"),
        "u4": np.dtype("<u4"),
    }
    try:
        return aliases[str(code)]
    except KeyError as exc:
        raise ValueError(f"Unsupported Plotly typed-array dtype: {code!r}") from exc


def _decode_typed_array(value: dict[str, Any]) -> np.ndarray:
    dtype = _plotly_dtype(str(value["dtype"]))
    raw = base64.b64decode(str(value["bdata"]))
    array = np.frombuffer(raw, dtype=dtype).copy()
    shape = value.get("shape")
    if shape:
        if isinstance(shape, str):
            dims = tuple(int(part.strip()) for part in shape.split(",") if part.strip())
        else:
            dims = tuple(int(part) for part in shape)
        array = array.reshape(dims)
    return array


def _coordinate_length(value: Any) -> int:
    if isinstance(value, list):
        return len(value)
    if isinstance(value, dict) and {"dtype", "bdata"}.issubset(value):
        return int(_decode_typed_array(value).shape[0])
    raise TypeError(f"Unsupported Plotly coordinate payload: {type(value).__name__}")


def _finite_unique(value: Any) -> list[float]:
    if isinstance(value, list):
        raw = [item for item in value if item is not None]
        array = np.asarray(raw, dtype=float)
    elif isinstance(value, dict) and {"dtype", "bdata"}.issubset(value):
        array = np.asarray(_decode_typed_array(value), dtype=float).reshape(-1)
    else:
        raise TypeError(f"Unsupported z coordinate payload: {type(value).__name__}")
    return sorted({round(float(item), 12) for item in array if np.isfinite(item)})


def _shift_coordinate(value: Any, offset: float) -> Any:
    if isinstance(value, list):
        return [
            None
            if item is None
            else float(item) + offset
            if np.isfinite(float(item))
            else float(item)
            for item in value
        ]
    if isinstance(value, dict) and {"dtype", "bdata"}.issubset(value):
        shifted = copy.deepcopy(value)
        array = _decode_typed_array(value)
        if array.dtype.kind != "f":
            array = array.astype(np.dtype("<f8"))
            shifted["dtype"] = "f8"
        finite = np.isfinite(array)
        array[finite] += offset
        shifted["bdata"] = base64.b64encode(array.tobytes(order="C")).decode("ascii")
        return shifted
    raise TypeError(f"Unsupported z coordinate payload: {type(value).__name__}")


def _trace_kind(trace: dict[str, Any]) -> str | None:
    if str(trace.get("type", "")) != "scatter3d":
        return None
    mode = str(trace.get("mode", ""))
    hoverinfo = str(trace.get("hoverinfo", ""))
    x_length = _coordinate_length(trace["x"])
    hover = trace.get("hovertext")

    if mode == "lines" and hoverinfo == "skip" and x_length == 5:
        return None
    if str(trace.get("name", "")) in TIME_KEYS and mode == "markers":
        return "cell_points"
    if isinstance(hover, str) and hover.startswith("Comm:"):
        return "communication_lines"
    if mode == "lines" and hoverinfo == "skip" and x_length == 6:
        return "communication_arrows"
    if isinstance(hover, str) and "Fate:" in hover:
        return "lineage_paths"
    if isinstance(hover, list) and hover:
        labels = [str(value) for value in hover]
        if all(value.endswith(")") and " (" in value for value in labels):
            return "endpoint_markers"
    return "unexpected_scatter3d"


def _parse_plotly_call(html: str) -> tuple[int, int, list[dict[str, Any]], dict[str, Any]]:
    marker = "Plotly.newPlot("
    call_start = html.rfind(marker)
    if call_start < 0:
        raise ValueError("No Plotly.newPlot call found in input HTML.")
    cursor = call_start + len(marker)
    decoder = json.JSONDecoder()
    while cursor < len(html) and html[cursor].isspace():
        cursor += 1
    _, div_end = decoder.raw_decode(html, cursor)
    cursor = div_end
    while cursor < len(html) and (html[cursor].isspace() or html[cursor] == ","):
        cursor += 1
    data_start = cursor
    data, data_end = decoder.raw_decode(html, data_start)
    cursor = data_end
    while cursor < len(html) and (html[cursor].isspace() or html[cursor] == ","):
        cursor += 1
    layout, _ = decoder.raw_decode(html, cursor)
    if not isinstance(data, list) or not isinstance(layout, dict):
        raise TypeError("Unexpected Plotly.newPlot data/layout payload.")
    return data_start, data_end, data, layout


def _without_z(data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = copy.deepcopy(data)
    for trace in result:
        trace.pop("z", None)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    offset = float(args.foreground_z_offset)
    if not np.isfinite(offset) or offset <= 0.0:
        raise ValueError("--foreground-z-offset must be a finite positive value.")

    input_html = Path(args.input_html).expanduser().resolve()
    if not input_html.is_file():
        raise FileNotFoundError(input_html)
    input_sha = _sha256(input_html)
    if input_sha.lower() != str(args.expected_input_sha256).lower():
        raise ValueError(
            f"Input HTML SHA-256 mismatch: expected={args.expected_input_sha256}, actual={input_sha}"
        )

    html = input_html.read_text(encoding="utf-8")
    data_start, data_end, data, layout = _parse_plotly_call(html)
    no_z_sha_before = _json_sha256(_without_z(data))
    layout_sha_before = _json_sha256(layout)
    counts: dict[str, int] = {}
    foreground_before: set[float] = set()
    foreground_after: set[float] = set()
    plane_before = [copy.deepcopy(trace.get("z")) for trace in data if trace.get("type") == "mesh3d"]
    border_before = [
        copy.deepcopy(trace.get("z"))
        for trace in data
        if trace.get("type") == "scatter3d"
        and trace.get("mode") == "lines"
        and trace.get("hoverinfo") == "skip"
        and _coordinate_length(trace["x"]) == 5
    ]

    for trace in data:
        kind = _trace_kind(trace)
        if kind is None:
            continue
        if kind == "unexpected_scatter3d":
            raise AssertionError(
                "Refusing unclassified Scatter3d trace: "
                f"name={trace.get('name')!r}, mode={trace.get('mode')!r}, "
                f"hoverinfo={trace.get('hoverinfo')!r}."
            )
        counts[kind] = counts.get(kind, 0) + 1
        foreground_before.update(_finite_unique(trace["z"]))
        trace["z"] = _shift_coordinate(trace["z"], offset)
        foreground_after.update(_finite_unique(trace["z"]))

    if args.dynamic_trace_counts:
        required_fixed = {"cell_points": 5, "endpoint_markers": 1}
        for key, expected in required_fixed.items():
            if counts.get(key) != expected:
                raise AssertionError(
                    f"Foreground trace count changed for {key}: expected={expected}, actual={counts.get(key)}"
                )
        if counts.get("communication_lines", 0) <= 0:
            raise AssertionError("No communication lines found")
        if counts.get("communication_lines") != counts.get("communication_arrows"):
            raise AssertionError("Communication line/arrow counts differ")
        if counts.get("lineage_paths", 0) <= 0:
            raise AssertionError("No lineage paths found")
        expected_count_contract: Any = {
            "mode": "dynamic corrected payload",
            "fixed": required_fixed,
            "communication_lines_equal_arrows": True,
            "positive_lineage_paths": True,
        }
    else:
        if counts != EXPECTED_COUNTS:
            raise AssertionError(
                f"Foreground trace counts changed: expected={EXPECTED_COUNTS}, actual={counts}"
            )
        expected_count_contract = EXPECTED_COUNTS
    if plane_before != [trace.get("z") for trace in data if trace.get("type") == "mesh3d"]:
        raise AssertionError("Slice-plane z values changed.")
    border_after = [
        trace.get("z")
        for trace in data
        if trace.get("type") == "scatter3d"
        and trace.get("mode") == "lines"
        and trace.get("hoverinfo") == "skip"
        and _coordinate_length(trace["x"]) == 5
    ]
    if border_before != border_after:
        raise AssertionError("Slice-border z values changed.")
    if _json_sha256(_without_z(data)) != no_z_sha_before:
        raise AssertionError("A non-z trace property changed.")
    if _json_sha256(layout) != layout_sha_before:
        raise AssertionError("Plotly layout changed.")

    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.stage-", dir=output_dir.parent))
    try:
        output_html = stage / "spatiotemporal_3d.html"
        replacement = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
        output_html.write_text(html[:data_start] + replacement + html[data_end:], encoding="utf-8")
        manifest = {
            "schema": "cytobridge.arista.fig5a.foreground-z-preview.v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "input_html": str(input_html),
            "input_html_sha256": input_sha,
            "output_html": output_html.name,
            "output_html_sha256": _sha256(output_html),
            "foreground_z_offset": offset,
            "offset_fraction_of_historical_z_spacing": offset / 3.8,
            "trace_counts": counts,
            "trace_count_contract": expected_count_contract,
            "foreground_levels_before": sorted(foreground_before),
            "foreground_levels_after": sorted(foreground_after),
            "unchanged_contract": {
                "non_z_trace_properties_sha256": no_z_sha_before,
                "layout_sha256": layout_sha_before,
                "slice_plane_z": plane_before,
                "slice_border_z": border_before,
                "scientific_recomputation": False,
            },
        }
        (stage / "preview_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        stage.rename(output_dir)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    print((output_dir / "preview_manifest.json").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
