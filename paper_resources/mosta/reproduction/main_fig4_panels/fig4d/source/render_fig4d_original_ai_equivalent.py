#!/usr/bin/env python3
"""Render corrected MOSTA Fig. 4d with the original Illustrator visual grammar.

The script is render-only.  It consumes the immutable current-package E15.5
numeric export and never loads a model, recomputes a velocity, communication
matrix, or projection.  The legacy notebook controls streamline construction;
Figure_mouse1.ai controls physical panel geometry, typography, colors, nodes,
arrows, labels, and the telencephalon ROI.  Coordinate values are never rotated,
rescaled, or warped.  The historical notebook's ordinary Matplotlib ``auto``
axes aspect is retained as presentation style instead of applying a post-hoc
transform to the stored coordinates.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import stat
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


EXPECTED_PACKAGE_COMMIT = "2b3c79eff3face7c4dd33de24d45384b9dbd8a84"
EXPECTED_REFERENCE_SHA256 = "8b9ca0ad3475040235036548d54b96272bf6c49f057f6c2a643152c11350ce25"
EXPECTED_PALETTE_SHA256 = "7e95e868e0a6ecd4a2ed13b57e6a8223e77e2302a0f9634ca30f41390c040b71"
EXPECTED_AI_SHA256 = "340a5ed88dc911d6923bc6b21cf1ceb39fdbef16edf2e325822a5b422045cbc2"
EXPECTED_TIME = 3.0
EXPECTED_STAGE = "E15.5"
EXPECTED_COMPUTE_CELLS = 8000
EXPECTED_BACKGROUND_CELLS = 113350
PANEL_WIDTH_PT = 290.0
PANEL_HEIGHT_PT = 378.0
PANEL_PAGE_TOP_PT = 464.0
AXES_RECT = (9.5512 / PANEL_WIDTH_PT, 28.078 / PANEL_HEIGHT_PT, 250.124 / PANEL_WIDTH_PT, 310.301 / PANEL_HEIGHT_PT)
ROI_BOUNDS = (-1.3, -0.5, 3.3, 4.2)
EDGE_IDENTITIES = (
    ("Choroid plexus", "Brain"),
    ("Meninges", "Brain"),
    ("Brain", "Meninges"),
    ("Brain", "Brain"),
)
LABEL_SPECS: Mapping[str, Mapping[str, Any]] = {
    "Choroid plexus": {
        "text": "Choroid Plexus",
        "offset": (2.2, -16.3),
        "ha": "center",
        "va": "top",
    },
    "Brain": {
        "text": "Brain",
        "offset": (24.7, 8.9),
        "ha": "left",
        "va": "center",
    },
    "Meninges": {
        "text": "Meninges",
        "offset": (12.0, -7.0),
        "ha": "left",
        "va": "center",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def file_record(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": sha256(path), "size_bytes": path.stat().st_size}


def freeze_tree(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_file():
            path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        elif path.is_dir():
            path.chmod(stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
    root.chmod(stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("utf-8"))
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def validate_ai_geometry(ai_path: Path) -> dict[str, Any]:
    import fitz

    document = fitz.open(ai_path)
    if len(document) != 1:
        raise RuntimeError("The original Illustrator style authority must have one PDF-compatible page.")
    page = document[0]
    images = {item[0]: item for item in page.get_images(full=True)}
    if 187 not in images:
        raise RuntimeError("Original Fig. 4d observed-background xref 187 is missing.")
    xref = images[187]
    rects = page.get_image_rects(187)
    if len(rects) != 1:
        raise RuntimeError("Original Fig. 4d xref 187 placement is ambiguous.")
    texts: dict[str, dict[str, Any]] = {}
    wanted = {"d", "Interaction-induced gene velocity", "E15.5", "Brain", "Meninges", "Choroid Plexus"}
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = str(span["text"])
                if text in wanted and float(span["bbox"][1]) >= 460.0:
                    texts.setdefault(
                        text,
                        {
                            "bbox": [float(x) for x in span["bbox"]],
                            "font": str(span["font"]),
                            "size_pt": float(span["size"]),
                        },
                    )
    missing = sorted(wanted - set(texts))
    if missing:
        raise RuntimeError(f"Original Fig. 4d typography anchors are missing: {missing}")
    return {
        "page_points": [float(page.rect.width), float(page.rect.height)],
        "observed_background_xref": 187,
        "observed_background_pixels": [int(xref[2]), int(xref[3])],
        "observed_background_bbox": [float(x) for x in rects[0]],
        "typography": texts,
    }


def select_edges(table: pd.DataFrame) -> pd.DataFrame:
    required = {"source", "target", "is_same_type", "weight_per_source"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise KeyError(f"Communication table is missing columns: {missing}")
    rows = []
    for draw_order, (source, target) in enumerate(EDGE_IDENTITIES, start=1):
        match = table.loc[(table["source"].astype(str) == source) & (table["target"].astype(str) == target)]
        if len(match) != 1:
            raise RuntimeError(f"Expected one communication row for {source} -> {target}; found {len(match)}.")
        row = match.iloc[0].copy()
        weight = float(row["weight_per_source"])
        if not np.isfinite(weight) or weight <= 0.0:
            raise RuntimeError(f"Displayed communication edge is not finite positive: {source} -> {target}.")
        row["draw_order"] = draw_order
        row["legacy_display_role"] = "focus_same_type_loop" if source == target else "original_AI_cross_type_identity"
        rows.append(row)
    result = pd.DataFrame(rows).reset_index(drop=True)
    result["is_self_edge"] = result["is_same_type"].astype(bool)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--numeric", type=Path, required=True)
    parser.add_argument("--server-manifest", type=Path, required=True)
    parser.add_argument("--calculation-gate", type=Path, required=True)
    parser.add_argument("--communication", type=Path, required=True)
    parser.add_argument("--palette", type=Path, required=True)
    parser.add_argument("--original-ai", type=Path, required=True)
    parser.add_argument("--legacy-helper", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    inputs = {
        "numeric": args.numeric.resolve(),
        "server_manifest": args.server_manifest.resolve(),
        "calculation_gate": args.calculation_gate.resolve(),
        "communication": args.communication.resolve(),
        "palette": args.palette.resolve(),
        "original_ai": args.original_ai.resolve(),
        "legacy_helper": args.legacy_helper.resolve(),
        "renderer": Path(__file__).resolve(),
    }
    for path in inputs.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    if sha256(inputs["palette"]) != EXPECTED_PALETTE_SHA256:
        raise RuntimeError("MOSTA original-article palette hash differs from the accepted contract.")
    if sha256(inputs["original_ai"]) != EXPECTED_AI_SHA256:
        raise RuntimeError("Original Illustrator style authority hash differs from the accepted contract.")

    server_manifest = json.loads(inputs["server_manifest"].read_text(encoding="utf-8"))
    gate = json.loads(inputs["calculation_gate"].read_text(encoding="utf-8"))
    if server_manifest["package"]["commit"] != EXPECTED_PACKAGE_COMMIT:
        raise RuntimeError("Server numeric package commit differs from the accepted contract.")
    if server_manifest["inputs"]["reference_h5ad"]["sha256"] != EXPECTED_REFERENCE_SHA256:
        raise RuntimeError("Server reference H5AD differs from the accepted contract.")
    if gate.get("status") != "PASS" or gate["velocity"]["fig4d_component"] != "interaction expression-state derivative":
        raise RuntimeError("Server calculation gate does not approve the Fig. 4d component.")
    numeric_rel = inputs["numeric"].name
    numeric_record = server_manifest["outputs"].get(numeric_rel)
    if numeric_record is None or numeric_record["sha256"] != sha256(inputs["numeric"]):
        raise RuntimeError("Numeric display package does not match the server manifest.")

    with np.load(inputs["numeric"], allow_pickle=False) as archive:
        full_xy = np.asarray(archive["full_background_spatial"], dtype=np.float32)
        full_labels = np.asarray(archive["full_background_labels"]).astype(str)
        compute_xy = np.asarray(archive["compute_spatial"], dtype=np.float32)
        compute_labels = np.asarray(archive["compute_labels"]).astype(str)
        velocity = np.asarray(archive["gene_interaction_projected_spatial"], dtype=np.float32)
        times = np.asarray(archive["time"], dtype=float)
    if full_xy.shape != (EXPECTED_BACKGROUND_CELLS, 2) or compute_xy.shape != (EXPECTED_COMPUTE_CELLS, 2):
        raise RuntimeError("Unexpected Fig. 4d background or quantitative-cohort dimensions.")
    if velocity.shape != compute_xy.shape or len(compute_labels) != len(compute_xy):
        raise RuntimeError("Projected interaction velocity does not align with the quantitative cohort.")
    if not all(np.isfinite(value).all() for value in (full_xy, compute_xy, velocity, times)):
        raise RuntimeError("Fig. 4d render arrays contain non-finite values.")
    if not np.allclose(times, EXPECTED_TIME, rtol=0.0, atol=1e-8):
        raise RuntimeError("Fig. 4d quantitative cells are not all observed E15.5.")

    palette = json.loads(inputs["palette"].read_text(encoding="utf-8"))
    missing_palette = sorted(set(full_labels) - set(palette))
    if missing_palette:
        raise RuntimeError(f"MOSTA palette is missing displayed labels: {missing_palette}")
    communication = pd.read_csv(inputs["communication"])
    edges = select_edges(communication)
    ai_geometry = validate_ai_geometry(inputs["original_ai"])

    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Immutable output already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    provenance_dir = output_dir / "provenance"
    provenance_dir.mkdir()

    os.environ.setdefault("MPLBACKEND", "Agg")
    import anndata as ad
    import matplotlib as mpl
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    import scvelo as scv

    helper_root = inputs["legacy_helper"].parent.parent
    sys.path.insert(0, str(helper_root))
    from downstream_helpers.mosta_fig4d_legacy_style import compute_legacy_communication_centroids

    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.unicode_minus": False,
        }
    )

    # Historical notebook Y-IQR gate, referenced to the complete observed state.
    q1, q3 = np.percentile(full_xy[:, 1], [25.0, 75.0])
    iqr = float(q3 - q1)
    y_bounds = (float(q1 - 1.5 * iqr), float(q3 + 1.5 * iqr))
    finite_velocity = np.isfinite(compute_xy).all(axis=1) & np.isfinite(velocity).all(axis=1) & (np.linalg.norm(velocity, axis=1) > 1e-12)
    in_y = (compute_xy[:, 1] >= y_bounds[0]) & (compute_xy[:, 1] <= y_bounds[1])
    keep = finite_velocity & in_y
    if int(keep.sum()) < 3:
        raise RuntimeError("Legacy Y-IQR display gate removed the velocity cohort.")
    velocity_adata = ad.AnnData(X=np.zeros((int(keep.sum()), 1), dtype=np.float32))
    velocity_adata.obsm["X_spatial"] = compute_xy[keep]
    velocity_adata.obsm["velocity_interaction_spatial"] = velocity[keep]
    categories = sorted(set(full_labels))
    velocity_adata.obs["Annotation"] = pd.Categorical(compute_labels[keep], categories=categories)

    centroids, centroid_table = compute_legacy_communication_centroids(
        full_xy,
        full_labels,
        ["Brain", "Choroid plexus", "Meninges"],
        top_n_y=200,
        top_n_y_exclusions=("Brain", "Meninges", "Choroid plexus"),
    )

    fig = plt.figure(figsize=(PANEL_WIDTH_PT / 72.0, PANEL_HEIGHT_PT / 72.0), facecolor="white")
    ax = fig.add_axes(AXES_RECT, facecolor="white")
    for label in categories:
        mask = full_labels == label
        ax.scatter(
            full_xy[mask, 0],
            full_xy[mask, 1],
            c=palette[label],
            s=2.35,
            alpha=0.35,
            linewidths=0,
            rasterized=True,
            zorder=0,
        )
    scv.pl.velocity_embedding_stream(
        velocity_adata,
        basis="spatial",
        vkey="velocity_interaction",
        color="Annotation",
        palette=[palette[label] for label in categories],
        ax=ax,
        show=False,
        density=2.0,
        smooth=0.8,
        min_mass=1.0,
        cutoff_perc=3.0,
        linewidth=0.55,
        arrow_size=0.26,
        n_neighbors=30,
        alpha=0.35,
        size=0.01,
        legend_loc="none",
        title="",
        frameon=False,
    )

    max_weight = float(edges["weight_per_source"].max())
    arrow_records = []
    drawn_nodes: set[str] = set()
    for row in edges.itertuples(index=False):
        source, target = str(row.source), str(row.target)
        p1, p2 = centroids[source], centroids[target]
        weight = float(row.weight_per_source)
        width = 1.0 + 10.0 * np.sqrt(weight / max_weight)
        if source == target:
            start = (float(p1[0] - 0.03), float(p1[1] + 0.05))
            end = (float(p1[0] + 0.06), float(p1[1] + 0.02))
            connectionstyle, mutation_scale = "arc3,rad=-10.0", 1.75
        else:
            start = (float(p1[0]), float(p1[1]))
            end = (float(p2[0]), float(p2[1]))
            connectionstyle, mutation_scale = "arc3,rad=0.15", 4.35
        ax.add_patch(
            mpatches.FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>,head_length=0.8,head_width=0.5",
                mutation_scale=mutation_scale,
                connectionstyle=connectionstyle,
                color=palette[source],
                linewidth=width,
                alpha=1.0,
                zorder=30,
            )
        )
        arrow_records.append(
            {
                "source": source,
                "target": target,
                "weight_per_source": weight,
                "linewidth_pt": float(width),
                "same_type_loop": source == target,
            }
        )
        for node, position in ((source, p1), (target, p2)):
            if node not in drawn_nodes:
                ax.scatter(position[0], position[1], s=162.0, c=palette[node], edgecolors="white", linewidth=2.5, zorder=31)
                drawn_nodes.add(node)

    ax.add_patch(
        mpatches.Rectangle(
            (ROI_BOUNDS[0], ROI_BOUNDS[2]),
            ROI_BOUNDS[1] - ROI_BOUNDS[0],
            ROI_BOUNDS[3] - ROI_BOUNDS[2],
            fill=False,
            edgecolor="black",
            linewidth=1.6,
            zorder=36,
        )
    )
    for node, spec in LABEL_SPECS.items():
        p = centroids[node]
        ax.annotate(
            str(spec["text"]),
            xy=(float(p[0]), float(p[1])),
            xytext=spec["offset"],
            textcoords="offset points",
            ha=str(spec["ha"]),
            va=str(spec["va"]),
            fontsize=9.14134,
            fontweight="bold",
            color="black",
            bbox={"boxstyle": "square,pad=0.12", "facecolor": "white", "edgecolor": "none", "alpha": 1.0},
            zorder=40,
            clip_on=False,
        )

    # The archived notebook uses plt.subplots(figsize=(16, 20)) and never
    # overrides Matplotlib's default aspect.  Preserve that original panel
    # grammar while keeping every stored coordinate value byte-identical.
    ax.set_aspect("auto")
    ax.set_axis_off()
    fig.text(3.167236 / PANEL_WIDTH_PT, 1.0 - (469.115 - PANEL_PAGE_TOP_PT) / PANEL_HEIGHT_PT, "d", ha="left", va="top", fontsize=14.0, fontweight="bold")
    fig.text(13.6812 / PANEL_WIDTH_PT, 1.0 - (468.835 - PANEL_PAGE_TOP_PT) / PANEL_HEIGHT_PT, "Interaction-induced gene velocity", ha="left", va="top", fontsize=14.0, fontweight="bold")
    fig.text(32.8916 / PANEL_WIDTH_PT, 1.0 - (491.79 - PANEL_PAGE_TOP_PT) / PANEL_HEIGHT_PT, EXPECTED_STAGE, ha="left", va="top", fontsize=12.0, fontweight="bold")

    stem = "Fig4d_MOSTA_interaction_gene_velocity_communication_E15p5_latest52D_original_AI_equivalent_v5"
    outputs = {suffix: output_dir / f"{stem}.{suffix}" for suffix in ("pdf", "svg", "png")}
    fig.savefig(outputs["pdf"], format="pdf", dpi=400, facecolor="white")
    fig.savefig(outputs["svg"], format="svg", dpi=400, facecolor="white")
    fig.savefig(outputs["png"], format="png", dpi=600, facecolor="white")
    plt.close(fig)

    edges.to_csv(output_dir / "communication_displayed_original_AI_identities.csv", index=False)
    centroid_table.to_csv(output_dir / "communication_centroids.csv", index=False)
    for name, path in inputs.items():
        shutil.copy2(path, provenance_dir / f"{name}__{path.name}")
    provenance_text = f"""# MOSTA Figure 4d provenance

## Source paths

- Numerical truth: `{inputs['numeric']}` from current package `{EXPECTED_PACKAGE_COMMIT}`, accepted observed E15.5 state, 8,000-cell same-state quantitative cohort.
- Velocity: saved interaction 50D expression derivative projected package-natively to observed `spatial_aligned`; no recomputation in this renderer.
- Communication: `{inputs['communication']}`; original AI identities are retained only because all four are finite positive in the corrected result.
- Style truth: `{inputs['original_ai']}` SHA-256 `{EXPECTED_AI_SHA256}` and `{inputs['legacy_helper']}`.

## Rendering contract

- Geometry: original 290 x 378 pt panel, Arial typography, original MOSTA palette, legacy density/smoothing/cutoff/arrow grammar.
- Scientific constraint: stored coordinates are unchanged. No coordinate rotation, rescaling, warp, or generated state is used. The axes retain the archived notebook's default `auto` aspect.
- Raster policy: complete 113,350-cell observed background is the only intentional raster layer; streamlines, communication, ROI, labels, and titles remain vector in PDF/SVG.

## Rebuild

Run the archived renderer copied into this bundle with the input paths recorded in `manifest.json`; choose a new non-existing output directory.
"""
    (provenance_dir / "figure-provenance.md").write_text(provenance_text, encoding="utf-8")

    manifest = {
        "schema_version": 1,
        "status": "COMPLETE",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "python": sys.executable,
        "dataset": "mosta",
        "panel": "Fig4d",
        "stage": EXPECTED_STAGE,
        "numeric_contract": {
            "package_commit": EXPECTED_PACKAGE_COMMIT,
            "reference_h5ad_sha256": EXPECTED_REFERENCE_SHA256,
            "state": "observed_real_E15.5",
            "velocity_component": "interaction expression-state derivative projected package-natively to observed spatial_aligned",
            "quantitative_cells": EXPECTED_COMPUTE_CELLS,
            "full_background_cells": EXPECTED_BACKGROUND_CELLS,
            "same_state_velocity_and_communication": True,
        },
        "style_contract": {
            "original_ai_sha256": EXPECTED_AI_SHA256,
            "palette_sha256": EXPECTED_PALETTE_SHA256,
            "ai_geometry": ai_geometry,
            "panel_points": [PANEL_WIDTH_PT, PANEL_HEIGHT_PT],
            "axes_rect_fraction": list(AXES_RECT),
            "legacy_stream": {"density": 2.0, "smooth": 0.8, "min_mass": 1.0, "cutoff_perc": 3.0, "linewidth_pt": 0.55, "arrow_size": 0.26},
            "background_points": {"size_pt2": 2.35, "alpha": 0.35},
            "rasterized_background_dpi": 400,
            "communication_arrow_alpha": 1.0,
            "axes_aspect": "auto (verbatim historical notebook default)",
            "font": "Arial",
            "vector_outputs": ["pdf", "svg"],
            "intentional_raster_layer": "complete observed E15.5 cell-type scatter",
        },
        "spatial_integrity": {
            "coordinate_values_modified": False,
            "rotation_applied": False,
            "coordinate_rescaling_applied": False,
            "warp_applied": False,
            "axes_aspect": "auto",
            "axes_aspect_source": "archived plot_single_velocity_field: plt.subplots(figsize=(16,20)); no set_aspect call",
        },
        "display": {
            "velocity_cells_rendered": int(keep.sum()),
            "velocity_cells_input": int(len(compute_xy)),
            "full_background_cells_rendered": int(len(full_xy)),
            "velocity_y_iqr_bounds": list(y_bounds),
            "communication_arrows": arrow_records,
            "communication_nodes": sorted(drawn_nodes),
            "roi_bounds": list(ROI_BOUNDS),
        },
        "array_hashes": {
            "full_background_spatial": array_sha256(full_xy),
            "full_background_labels": array_sha256(full_labels.astype("U")),
            "compute_spatial": array_sha256(compute_xy),
            "compute_labels": array_sha256(compute_labels.astype("U")),
            "gene_interaction_projected_spatial": array_sha256(velocity),
        },
        "inputs": {name: file_record(path) for name, path in inputs.items()},
        "outputs": {name: file_record(path) for name, path in outputs.items()},
    }
    write_json(output_dir / "manifest.json", manifest)
    checksum_paths = sorted(path for path in output_dir.rglob("*") if path.is_file() and path.name not in {"SHA256SUMS.json", "COMPLETE"})
    write_json(output_dir / "SHA256SUMS.json", {str(path.relative_to(output_dir)): sha256(path) for path in checksum_paths})
    (output_dir / "COMPLETE").write_text("COMPLETE\n", encoding="utf-8")
    freeze_tree(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
