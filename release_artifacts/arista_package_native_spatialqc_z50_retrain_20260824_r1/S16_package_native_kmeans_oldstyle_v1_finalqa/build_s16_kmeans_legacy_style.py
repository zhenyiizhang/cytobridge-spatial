#!/usr/bin/env python3
"""Render corrected ARISTA S16 k-means prototypes in the submitted style."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import fitz
import matplotlib as mpl
import pandas as pd

import build_s15_s17_strict_legacy_style as legacy


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recluster-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    source = args.recluster_dir.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite immutable output: {output}")

    prototype_path = source / "S16_lr_kmeans_prototypes.csv"
    diagnostic_path = source / "S16_lr_kmeans_diagnostics.csv"
    k_selection_path = source / "S16_lr_k_selection.csv"
    noise_path = source / "S16_lr_noise_stability_summary.csv"
    leave_path = source / "S16_lr_leave_one_time_stability.csv"
    prototypes = pd.read_csv(prototype_path)
    diagnostics = pd.read_csv(diagnostic_path)
    if sorted(prototypes["cluster"].unique().tolist()) != [1, 2]:
        raise ValueError("S16 requires exactly two clusters")
    counts = prototypes.groupby("cluster")["n_pairs"].first().astype(int)
    if counts.sum() != 531 or counts.min() < 20:
        raise ValueError(f"Invalid corrected S16 cluster sizes: {counts.to_dict()}")

    legacy._configure_legacy_style()
    mpl.rcParams.update({
        "font.family": "Arial",
        "font.sans-serif": ["Arial"],
        "svg.fonttype": "none",
        "svg.hashsalt": "arista-s16-kmeans-legacy-style-v1",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.stage-", dir=output.parent))
    try:
        figures = stage / "figures"
        tables = stage / "tables"
        figures.mkdir(parents=True)
        tables.mkdir(parents=True)
        stem = "FigureS16_ARISTA_package_native_kmeans_legacy_style"
        svg_path = figures / f"{stem}.svg"
        pdf_path = figures / f"{stem}.pdf"
        png_path = figures / f"{stem}.png"
        legacy._plot_s16(prototypes, svg_path)
        legacy._svg_to_exact_pdf(svg_path, pdf_path, legacy.S16_PAGE_PT)
        rendered_size = legacy._render_pdf(pdf_path, png_path, 300)
        page_size = legacy._pdf_page_size(pdf_path)

        for path in [prototype_path, diagnostic_path, k_selection_path, noise_path, leave_path]:
            shutil.copy2(path, tables / path.name)
        shutil.copy2(Path(__file__).resolve(), stage / Path(__file__).name)
        k_selection = pd.read_csv(k_selection_path)
        noise = pd.read_csv(noise_path)
        leave_time = pd.read_csv(leave_path)
        manifest = {
            "schema": "cytobridge.arista.figureS16.strict-lr-kmeans.legacy-style.v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "scientific_contract": {
                "source": "package-native strict all-subunit LR scores",
                "clustering": "deterministic package kmeans on row-minmax nine-time-point profiles",
                "cluster_counts": {str(k): int(v) for k, v in counts.items()},
                "silhouette": float(diagnostics.loc[0, "silhouette"]),
                "k2_best_silhouette_among_k2_to_k8": bool(
                    int(k_selection.sort_values(["silhouette", "k"], ascending=[False, True]).iloc[0]["k"]) == 2
                ),
                "median_noise_ari_sigma_0p05": float(
                    noise.loc[noise["noise_sigma"].eq(0.05), "median_adjusted_rand_index"].iloc[0]
                ),
                "minimum_leave_one_time_ari": float(leave_time["adjusted_rand_index"].min()),
            },
            "display_contract": {
                "style": "submitted S16 line/marker/fill/legend grammar",
                "palette": "legacy seaborn Set2",
                "font": "Arial",
                "page_pt": list(legacy.S16_PAGE_PT),
            },
            "inputs": {str(path): sha256(path) for path in [prototype_path, diagnostic_path, k_selection_path, noise_path, leave_path]},
            "outputs": {},
            "qa": {
                "passed": bool(
                    counts.min() >= 20
                    and len(counts) == 2
                    and all(abs(a - b) < 1e-3 for a, b in zip(page_size, legacy.S16_PAGE_PT))
                ),
                "rendered_png_px": list(rendered_size),
                "page_pt": list(page_size),
                "singleton_cluster_absent": bool(counts.min() > 1),
            },
        }
        if not manifest["qa"]["passed"]:
            raise AssertionError(f"S16 QA failed: {manifest['qa']}")
        manifest["outputs"] = {
            str(path.relative_to(stage)): {"sha256": sha256(path), "size_bytes": path.stat().st_size}
            for path in sorted(stage.rglob("*")) if path.is_file()
        }
        (stage / "MANIFEST.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        stage.rename(output)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    print(json.dumps({"output_dir": str(output), "qa": manifest["qa"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
