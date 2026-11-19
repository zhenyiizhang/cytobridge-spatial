#!/usr/bin/env python3
"""Render corrected S7 with the frozen old Plotly helper and pinned Arial files."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time

import kaleido
import pandas as pd
import plotly


EXPECTED_LABELS_SHA256 = "667d19659ffb7cab18caed14e543e5d03dc57c0697f48557728fe0bc9af003cb"
EXPECTED_HELPER_SHA256 = "128be4a718d8ca43575e7b944a6f06c59fd9cd29cfc0b668a2ad5c19c9ff10d6"
EXPECTED_PALETTE_SHA256 = "7e95e868e0a6ecd4a2ed13b57e6a8223e77e2302a0f9634ca30f41390c040b71"
EXPECTED_ARIAL = {
    "Arial.ttf": "525979822591a3447cfc49d943d6f7683508e25543407871c0ed8fed05fd2bd9",
    "Arial Bold.ttf": "d72db21f9242aedd6b917d8549ad5921766b24d5f8d0becfda2ff4c620b3c2e0",
    "Arial Italic.ttf": "ce1d2f1ab89db45f9796100eee960f5702a40e84c225c2b48c3ec3e81d153f98",
    "Arial Bold Italic.ttf": "374b0190a9844343110d8f8ed1818117a4591803d022bbb2bd189d63a681e731",
}
TIMES = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "size_bytes": int(path.stat().st_size),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True)
    parser.add_argument("--helper", required=True)
    parser.add_argument("--palette", required=True)
    parser.add_argument("--font-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def configure_arial(font_dir: Path, provenance: Path) -> tuple[Path, str]:
    font_identities: dict[str, dict[str, object]] = {}
    for name, expected_sha in EXPECTED_ARIAL.items():
        path = font_dir / name
        actual_sha = sha256(path)
        if actual_sha != expected_sha:
            raise RuntimeError(f"Arial SHA mismatch for {path}: {actual_sha} != {expected_sha}")
        font_identities[name] = identity(path)

    cache_dir = Path(tempfile.mkdtemp(prefix="mosta_s7_fontconfig_"))
    fontconfig_path = provenance / "fonts.conf"
    fontconfig_path.write_text(
        "<?xml version=\"1.0\"?>\n"
        "<!DOCTYPE fontconfig SYSTEM \"urn:fontconfig:fonts.dtd\">\n"
        "<fontconfig>\n"
        "  <include ignore_missing=\"yes\">/etc/fonts/fonts.conf</include>\n"
        f"  <dir>{font_dir}</dir>\n"
        f"  <cachedir>{cache_dir}</cachedir>\n"
        "</fontconfig>\n",
        encoding="utf-8",
    )
    os.environ["FONTCONFIG_FILE"] = str(fontconfig_path)
    os.environ["FONTCONFIG_PATH"] = str(provenance)
    match = subprocess.run(
        ["fc-match", "--format", "%{family}|%{style}|%{file}", "Arial"],
        check=True,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    ).stdout.strip()
    if not match.startswith("Arial|") or str(font_dir) not in match:
        raise RuntimeError(f"Arial fontconfig pin failed: {match}")
    (provenance / "arial_font_identities.json").write_text(
        json.dumps(font_identities, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (provenance / "fontconfig_match.txt").write_text(match + "\n", encoding="utf-8")
    return fontconfig_path, match


def main() -> None:
    args = parse_args()
    labels_path = Path(args.labels).resolve()
    helper_path = Path(args.helper).resolve()
    palette_path = Path(args.palette).resolve()
    font_dir = Path(args.font_dir).resolve()
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"immutable output already exists: {output}")
    for path, expected in (
        (labels_path, EXPECTED_LABELS_SHA256),
        (helper_path, EXPECTED_HELPER_SHA256),
        (palette_path, EXPECTED_PALETTE_SHA256),
    ):
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"SHA mismatch for {path}: {actual} != {expected}")

    started = time.time()
    output.mkdir(parents=True, exist_ok=False)
    figures = output / "figures"
    provenance = output / "provenance"
    figures.mkdir()
    provenance.mkdir()
    shutil.copy2(Path(__file__).resolve(), provenance / Path(__file__).name)
    shutil.copy2(helper_path, provenance / helper_path.name)
    shutil.copy2(palette_path, provenance / palette_path.name)
    fontconfig_path, fontconfig_match = configure_arial(font_dir, provenance)

    labels = pd.read_csv(labels_path)
    if len(labels) != 350000:
        raise RuntimeError(f"expected 350000 fixed-particle rows, found {len(labels)}")
    labels_list: list[list[str]] = []
    for time_value in TIMES:
        subset = labels[labels["time"].eq(time_value)].sort_values("particle_id", kind="stable")
        if len(subset) != 50000 or subset["particle_id"].to_list() != list(range(50000)):
            raise RuntimeError(f"invalid persistent particle set at t={time_value:g}")
        labels_list.append(subset["celltype"].astype(str).to_list())

    palette = json.loads(palette_path.read_text(encoding="utf-8"))
    spec = importlib.util.spec_from_file_location("s7_old_style_helper", helper_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load frozen old helper")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)

    stem = "Figure_S7_MOSTA_latest_package_global_t0_fixed_particle_k10_exact_old_plotly_style_Arial"
    html_path = figures / f"{stem}.html"
    figure = helper.plot_sankey(
        predicted_labels_list=labels_list,
        out_html=str(html_path),
        start_index=0,
        time_keys=[f"{value:.1f}" for value in TIMES],
        show_time_axis=True,
        min_flow=None,
        keep_source_cumfrac=0.8,
        normalize_mode=None,
        label_to_color=palette,
        lineage_anchor_mode=False,
        style="nature-methods",
        title="Cell Fate Transitions",
        width=None,
        height=None,
    )
    paths = {
        "html": html_path,
        "svg": figures / f"{stem}.svg",
        "pdf": figures / f"{stem}.pdf",
        "png": figures / f"{stem}.png",
    }
    figure.write_image(str(paths["svg"]), format="svg", scale=1)
    figure.write_image(str(paths["pdf"]), format="pdf", scale=1)
    figure.write_image(str(paths["png"]), format="png", scale=2)

    manifest = {
        "schema_version": 1,
        "status": "complete",
        "panel": "Supplementary Figure S7",
        "dataset": "MOSTA",
        "numerical_input": identity(labels_path),
        "style_helper": identity(helper_path),
        "palette": identity(palette_path),
        "fontconfig": {
            "path": str(fontconfig_path),
            "match": fontconfig_match,
            "files": {name: identity(font_dir / name) for name in EXPECTED_ARIAL},
        },
        "parameters": {
            "trajectory": "global_t0_non_split_fixed_particle",
            "n_particles": 50000,
            "time_points": list(TIMES),
            "classifier_k": 10,
            "spatial_warp": False,
            "restart_from_observed_anchor": False,
            "keep_source_cumfrac": 0.8,
            "normalize_mode": None,
            "min_flow": None,
            "style": "nature-methods",
            "title": "Cell Fate Transitions",
            "show_time_axis": True,
            "width": 1600,
            "height": 1000,
        },
        "runtime": {
            "python_executable": sys.executable,
            "python": sys.version,
            "platform": platform.platform(),
            "plotly": plotly.__version__,
            "kaleido": getattr(kaleido, "__version__", "unknown"),
        },
        "outputs": {kind: identity(path) for kind, path in paths.items()},
        "wall_seconds": time.time() - started,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    checksum_paths = [*sorted(figures.iterdir()), manifest_path, *sorted(provenance.iterdir())]
    checksum_path = output / "SHA256SUMS.txt"
    checksum_path.write_text(
        "".join(f"{sha256(path)}  {path.relative_to(output)}\n" for path in checksum_paths),
        encoding="utf-8",
    )
    (output / "COMPLETE").write_text("complete\n", encoding="utf-8")
    for path in output.rglob("*"):
        os.chmod(path, 0o444 if path.is_file() else 0o555)
    os.chmod(output, 0o555)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
