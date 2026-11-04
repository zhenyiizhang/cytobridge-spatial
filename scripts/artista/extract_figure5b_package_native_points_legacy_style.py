#!/usr/bin/env python3
"""Extract the fresh ARISTA t=0.5 generated markers for old-style Figure 5b.

The input snapshot has already undergone semantic palette restoration.  This
step removes only axes/title/background nodes; it does not replot, resample,
filter, or move a single generated cell.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree as ET


FILL_RE = re.compile(r"(?:^|;)\s*fill:\s*(#[0-9a-fA-F]{6})")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def geometry(root: ET.Element) -> dict[str, str | None]:
    return {
        "width": root.get("width"),
        "height": root.get("height"),
        "viewBox": root.get("viewBox"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-svg", required=True, type=Path)
    parser.add_argument("--palette-json", required=True, type=Path)
    parser.add_argument("--palette-restoration-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    snapshot = args.snapshot_svg.expanduser().resolve()
    palette_path = args.palette_json.expanduser().resolve()
    restoration_manifest = args.palette_restoration_manifest.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)

    palette = json.loads(palette_path.read_text(encoding="utf-8"))
    valid_colors = {str(color).lower() for color in palette.values()}
    ET.register_namespace("", "http://www.w3.org/2000/svg")
    ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")
    tree = ET.parse(snapshot)
    source_root = tree.getroot()
    collection = next(
        node
        for node in source_root.iter()
        if node.get("id", "").startswith("PathCollection_")
    )
    uses = [node for node in collection.iter() if node.tag.endswith("use")]
    if not uses:
        raise ValueError("No generated markers found")
    colors: list[str] = []
    for node in uses:
        match = FILL_RE.search(node.get("style", ""))
        if match is None:
            raise ValueError("Generated marker without fill")
        color = match.group(1).lower()
        if color not in valid_colors:
            raise ValueError(f"Generated marker uses noncanonical color {color}")
        colors.append(color)

    output_root = ET.Element(source_root.tag, dict(source_root.attrib))
    for node in source_root:
        if node.tag.endswith("defs") and any(child.tag.endswith("clipPath") for child in node):
            output_root.append(copy.deepcopy(node))
    output_root.append(copy.deepcopy(collection))
    points_svg = output_dir / "time_0.5_points_only.svg"
    ET.ElementTree(output_root).write(points_svg, encoding="utf-8", xml_declaration=True)

    manifest = {
        "schema": "cytobridge.arista.fig5b.package-native-oldstyle-payload.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "workflow": "semantic palette restoration followed by node-only SVG extraction; no replotting",
        "scientific_state": {
            "model_time": 0.5,
            "biological_time": "3.5 DPI",
            "source": "fresh package-native generated split-SDE state with visualization-only k=1 warp",
            "n_full_corrected": len(uses),
            "n_displayed": len(uses),
            "display_mask_indices": [],
            "spatial_warp_k": 1,
            "spatial_warp_affects_computation": False,
        },
        "style_contract": {
            "palette_json": str(palette_path),
            "palette_sha256": sha256(palette_path),
            "marker_area_pt2": 2.5,
            "marker_alpha": 0.9,
            "marker_linewidth": 0.0,
            "aspect": "equal",
            "background": "transparent points-only payload",
            "source_svg_geometry": geometry(source_root),
            "points_svg_geometry": geometry(output_root),
            "coordinates_and_marker_order_unchanged": True,
        },
        "inputs": {
            "snapshot_svg": {"path": str(snapshot), "sha256": sha256(snapshot)},
            "palette_restoration_manifest": {
                "path": str(restoration_manifest),
                "sha256": sha256(restoration_manifest),
            },
        },
        "qa": {
            "passed": True,
            "marker_count": len(uses),
            "unique_canonical_colors_used": sorted(set(colors)),
            "all_marker_colors_in_submitted_palette": True,
            "display_filter_applied": False,
        },
        "outputs": {
            "points_only_svg": {
                "path": points_svg.name,
                "sha256": sha256(points_svg),
            }
        },
    }
    (output_dir / "render_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(output_dir), "marker_count": len(uses)}, indent=2))


if __name__ == "__main__":
    main()
