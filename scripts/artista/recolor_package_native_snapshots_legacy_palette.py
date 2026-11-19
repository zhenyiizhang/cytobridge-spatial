#!/usr/bin/env python3
"""Recolor package-native ARISTA snapshot SVGs with the submitted palette.

The package output and the submitted figures use the same 27 hex colors, but
the package assigned those colors to alphabetically ordered labels.  This
script performs a semantic label-preserving conversion:

    package color -> cell-type label -> submitted color

Coordinates, marker count, marker opacity, and SVG geometry are unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path


FILL_RE = re.compile(r"(?P<prefix>fill:\s*)(?P<color>#[0-9a-fA-F]{6})")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--package-palette", required=True, type=Path)
    parser.add_argument("--legacy-palette", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    package_palette_path = args.package_palette.expanduser().resolve()
    legacy_palette_path = args.legacy_palette.expanduser().resolve()
    package_palette = json.loads(package_palette_path.read_text(encoding="utf-8"))
    legacy_palette = json.loads(legacy_palette_path.read_text(encoding="utf-8"))

    if set(package_palette) != set(legacy_palette):
        raise ValueError("Package and submitted palettes do not contain the same labels")
    package_color_to_label = {
        str(color).lower(): str(label) for label, color in package_palette.items()
    }
    if len(package_color_to_label) != len(package_palette):
        raise ValueError("Package palette colors are not one-to-one")
    color_conversion = {
        color: str(legacy_palette[label]).lower()
        for color, label in package_color_to_label.items()
    }

    output_dir.mkdir(parents=True, exist_ok=False)
    snapshots_out = output_dir / "snapshots"
    snapshots_out.mkdir()
    audits: list[dict[str, object]] = []
    for source in sorted((input_dir / "snapshots").glob("*.svg")):
        text = source.read_text(encoding="utf-8")
        converted = 0
        semantic_changes = 0

        def replace(match: re.Match[str]) -> str:
            nonlocal converted, semantic_changes
            current = match.group("color").lower()
            target = color_conversion.get(current)
            if target is None:
                return match.group(0)
            converted += 1
            semantic_changes += int(current != target)
            return f"{match.group('prefix')}{target}"

        recolored = FILL_RE.sub(replace, text)
        target = snapshots_out / source.name
        target.write_text(recolored, encoding="utf-8")
        audits.append(
            {
                "file": source.name,
                "input_sha256": sha256(source),
                "output_sha256": sha256(target),
                "recognized_fill_occurrences": converted,
                "changed_fill_occurrences": semantic_changes,
                "geometry_tokens_unchanged": bool(
                    re.sub(r"fill:\s*#[0-9a-fA-F]{6}", "fill:#COLOR", text)
                    == re.sub(r"fill:\s*#[0-9a-fA-F]{6}", "fill:#COLOR", recolored)
                ),
            }
        )

    # Preserve the numerical and lineage inputs beside the recolored snapshots.
    for name in ["all_time_communications.pkl", "fixed_particle_lineage_labels.npz"]:
        source = input_dir / name
        if source.is_file():
            shutil.copy2(source, output_dir / name)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "operation": "semantic palette restoration only",
        "scientific_values_changed": False,
        "geometry_changed": False,
        "marker_count_changed": False,
        "input_dir": str(input_dir),
        "package_palette": {
            "path": str(package_palette_path),
            "sha256": sha256(package_palette_path),
        },
        "legacy_palette": {
            "path": str(legacy_palette_path),
            "sha256": sha256(legacy_palette_path),
        },
        "files": audits,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(output_dir), "files": len(audits)}, indent=2))


if __name__ == "__main__":
    main()
