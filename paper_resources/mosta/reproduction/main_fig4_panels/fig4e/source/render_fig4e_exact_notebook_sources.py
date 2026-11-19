#!/usr/bin/env python3
"""Render MOSTA Fig. 4e source panels by calling the archived plotter directly.

Only numerical arrays are replaced.  Figure construction and all visible
stream/scatter parameters come from the historical MOSTA notebook's imported
``evaluation.arista_code.mosta_ported.velocity.plot_single_velocity_field``.
Despite the module's historical location, no ARISTA data, labels, model, or
analysis code is loaded or used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np


EXPECTED_NUMERIC_SHA256 = "f0e66c42cde757186d6b5ab11f2bb2fca851157045a36f4274001bc60d3d0ef4"
EXPECTED_GATE_SHA256 = "115c14d1931d65534441d2c92e54cd6685ca1141650d144e910fbe64ae345e3a"
EXPECTED_NOTEBOOK_SHA256 = "ab1ee780f57ac6868804a642e70004d238b36c50a79efed74dbb3df666f6423b"
EXPECTED_PLOTTER_SHA256 = "0c3a78bd663c056dc66ceeb781ab3d9073441c3d993c512ddd945322b2105532"

ROI = (-1.3, -0.5, 3.3, 4.2)
CATEGORY_ORDER = (
    "Apical Progenitors (RG)",
    "Basal Progenitors (IP)",
    "Choroid Plexus",
    "Excitatory Neurons",
    "Glioblasts",
    "Inhibitory Neurons",
    "Other",
)
PALETTE = {
    "Apical Progenitors (RG)": "#1f77b4",
    "Basal Progenitors (IP)": "#aec7e8",
    "Choroid Plexus": "#7f7f7f",
    "Excitatory Neurons": "#ffbb78",
    "Glioblasts": "#8c564b",
    "Inhibitory Neurons": "#9467bd",
    "Other": "#d9d9d9",
}
PANEL_SPECS = {
    "gene_full": ("Gene Space", "full", "gene_full_projected_spatial"),
    "gene_interaction": (
        "Gene Space",
        "interaction",
        "gene_interaction_projected_spatial",
    ),
    "physical_full": ("Physical Space", "full", "physical_full"),
    "physical_interaction": (
        "Physical Space",
        "interaction",
        "physical_interaction",
    ),
}

# Uniformly fitted rectangles recovered from the final Illustrator panel
# xrefs.  The AI-ready export uses these page sizes directly so stream strokes
# keep their notebook point widths while the scatter markers receive only the
# exact uniform geometric scale applied to the original raster scatter layer.
AI_READY_RECTS = {
    "gene_full": (293.2336730957031, 521.8057385408634, 424.50286865234375, 614.9645251310116),
    "gene_interaction": (450.05767822265625, 517.11820009743, 586.3214111328125, 613.8214971681952),
    "physical_full": (289.7364807128906, 655.7266585548849, 429.0517272949219, 754.5955460349587),
    "physical_interaction": (451.6594543457031, 655.8231945850408, 583.97119140625, 749.7218493602718),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def freeze_tree(root: Path) -> None:
    for path in root.rglob("*"):
        path.chmod(0o444 if path.is_file() else 0o555)
    root.chmod(0o555)


def sanitize_export_only(fig) -> dict[str, int]:
    """Make old-style artists PDF-safe without changing visible geometry."""
    from matplotlib.collections import LineCollection, PathCollection

    audit = {"scatter_collections_rasterized": 0, "nonfinite_linewidths_repaired": 0}
    for ax in fig.axes:
        for collection in ax.collections:
            if isinstance(collection, PathCollection):
                collection.set_rasterized(True)
                audit["scatter_collections_rasterized"] += 1
            if isinstance(collection, LineCollection):
                widths = np.asarray(collection.get_linewidths(), dtype=float)
                if widths.size and not np.isfinite(widths).all():
                    finite = widths[np.isfinite(widths)]
                    replacement = float(np.median(finite)) if finite.size else 1.5
                    audit["nonfinite_linewidths_repaired"] += int(
                        np.count_nonzero(~np.isfinite(widths))
                    )
                    collection.set_linewidths(
                        np.where(np.isfinite(widths), widths, replacement)
                    )
    return audit


def save_full_and_axes_crop(
    fig,
    ax,
    *,
    stem: str,
    raw_dir: Path,
    crop_dir: Path,
    ai_ready_dir: Path,
    ai_ready_rect: tuple[float, float, float, float],
) -> dict:
    """Save the exact notebook figure plus its axes-only source for AI assembly."""
    fig.canvas.draw()
    export_audit = sanitize_export_only(fig)
    raw_outputs = {}
    for suffix in ("pdf", "svg", "png"):
        out = raw_dir / f"{stem}.{suffix}"
        if suffix == "png":
            fig.savefig(out, dpi=300, bbox_inches="tight")
        else:
            fig.savefig(out, format=suffix, bbox_inches="tight", dpi=300)
        raw_outputs[out.name] = {"sha256": sha256(out), "size_bytes": out.stat().st_size}

    # Illustrator uses only the plot field; titles and the shared legend are
    # supplied by Figure_mouse1.ai.  Cropping the exact axes is a composition
    # step, not a second plot implementation.
    title_artist = ax.title
    title_visible = title_artist.get_visible()
    title_artist.set_visible(False)
    legend = ax.get_legend()
    legend_visible = legend.get_visible() if legend is not None else None
    if legend is not None:
        legend.set_visible(False)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    extent = ax.get_window_extent(renderer).transformed(fig.dpi_scale_trans.inverted())
    crop_outputs = {}
    for suffix in ("pdf", "svg", "png"):
        out = crop_dir / f"{stem}__axes_only.{suffix}"
        if suffix == "png":
            fig.savefig(out, dpi=300, bbox_inches=extent, pad_inches=0)
        else:
            fig.savefig(out, format=suffix, dpi=300, bbox_inches=extent, pad_inches=0)
        crop_outputs[out.name] = {"sha256": sha256(out), "size_bytes": out.stat().st_size}

    # Illustrator retained vector stroke/effect widths when the imported
    # stream layer was resized, while its rasterized scatter layer scaled
    # geometrically.  Reproduce that deterministic composition behavior:
    # shrink scatter marker areas by scale^2, keep stream/arrow point widths,
    # and make the axes page itself equal to the fitted AI rectangle so the
    # assembler applies no further scale.
    from matplotlib.collections import PathCollection
    from matplotlib.patches import FancyArrowPatch

    x0, y0, x1, y1 = ai_ready_rect
    target_width_points = float(x1 - x0)
    target_height_points = float(y1 - y0)
    source_width_points = float(extent.width * 72.0)
    source_height_points = float(extent.height * 72.0)
    scale_x = target_width_points / source_width_points
    scale_y = target_height_points / source_height_points
    if not np.isclose(scale_x, scale_y, rtol=0, atol=1e-6):
        raise RuntimeError(f"AI-ready source would require anisotropic scaling: {scale_x}, {scale_y}")
    marker_scale = float(scale_x)
    scatter_collection_count = 0
    for collection in ax.collections:
        if isinstance(collection, PathCollection):
            collection.set_sizes(np.asarray(collection.get_sizes(), dtype=float) * marker_scale**2)
            scatter_collection_count += 1
    streamline_arrow_patch_count = 0
    for patch in ax.patches:
        if isinstance(patch, FancyArrowPatch):
            patch.set_mutation_scale(float(patch.get_mutation_scale()) * marker_scale)
            streamline_arrow_patch_count += 1
    fig.set_size_inches(target_width_points / 72.0, target_height_points / 72.0, forward=True)
    ax.set_position([0.0, 0.0, 1.0, 1.0])
    fig.canvas.draw()
    ai_ready_outputs = {}
    for suffix in ("pdf", "svg", "png"):
        out = ai_ready_dir / f"{stem}__AI_ready_stroke_parity.{suffix}"
        if suffix == "png":
            fig.savefig(out, dpi=600, pad_inches=0)
        else:
            fig.savefig(out, format=suffix, dpi=600, pad_inches=0)
        ai_ready_outputs[out.name] = {"sha256": sha256(out), "size_bytes": out.stat().st_size}
    title_artist.set_visible(title_visible)
    if legend is not None and legend_visible is not None:
        legend.set_visible(legend_visible)
    return {
        "export_audit": export_audit,
        "raw_outputs": raw_outputs,
        "axes_crop_outputs": crop_outputs,
        "axes_crop_inches": [float(extent.width), float(extent.height)],
        "AI_ready_stroke_parity": {
            "target_page_points": [target_width_points, target_height_points],
            "source_axes_points": [source_width_points, source_height_points],
            "uniform_marker_geometry_scale": marker_scale,
            "scatter_marker_area_scale": marker_scale**2,
            "scatter_collections_scaled": scatter_collection_count,
            "stream_linewidth_points_retained": 1.5,
            "stream_arrow_glyph_geometry_scale": marker_scale,
            "stream_arrow_patches_scaled": streamline_arrow_patch_count,
            "stream_arrow_style_parameter_source": 1.2,
            "outputs": ai_ready_outputs,
            "rotation": False,
            "anisotropic_stretch": False,
            "warp": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--numeric", type=Path, required=True)
    parser.add_argument("--calculation-gate", type=Path, required=True)
    parser.add_argument("--style-notebook", type=Path, required=True)
    parser.add_argument("--plotter", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()

    numeric = args.numeric.expanduser().resolve()
    gate_path = args.calculation_gate.expanduser().resolve()
    notebook = args.style_notebook.expanduser().resolve()
    plotter_path = args.plotter.expanduser().resolve()
    identities = {
        "numeric": sha256(numeric),
        "calculation_gate": sha256(gate_path),
        "style_notebook": sha256(notebook),
        "plotter": sha256(plotter_path),
    }
    expected = {
        "numeric": EXPECTED_NUMERIC_SHA256,
        "calculation_gate": EXPECTED_GATE_SHA256,
        "style_notebook": EXPECTED_NOTEBOOK_SHA256,
        "plotter": EXPECTED_PLOTTER_SHA256,
    }
    if identities != expected:
        raise RuntimeError(f"Input identity contract failed: {identities} vs {expected}")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if (
        gate.get("status") != "PASS"
        or gate.get("cohort", {}).get("compute_n_cells") != 17071
        or gate.get("velocity", {}).get("interaction_m") != 1024
    ):
        raise RuntimeError("Calculation gate does not pass the m=1024 Brain contract.")

    with np.load(numeric, allow_pickle=False) as archive:
        values = {key: np.asarray(archive[key]) for key in archive.files}
    coords = np.asarray(values["compute_spatial"], dtype=np.float32)
    labels = values["telencephalon_notebook_labels"].astype(str).copy()
    labels[labels == "Immature Neurons"] = "Other"
    labels[~np.isin(labels, CATEGORY_ORDER)] = "Other"
    if coords.shape != (17071, 2) or set(labels) != set(CATEGORY_ORDER):
        raise RuntimeError("Complete-Brain coordinates or seven-class labels failed.")

    out_dir = args.output_dir.expanduser().resolve()
    if args.freeze and out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite immutable output: {out_dir}")
    raw_dir = out_dir / "raw_notebook_style"
    crop_dir = out_dir / "axes_only_for_AI_assembly"
    ai_ready_dir = out_dir / "AI_ready_stroke_parity_sources"
    provenance_dir = out_dir / "provenance"
    raw_dir.mkdir(parents=True)
    crop_dir.mkdir(parents=True)
    ai_ready_dir.mkdir(parents=True)
    provenance_dir.mkdir(parents=True)

    cache = Path(tempfile.mkdtemp(prefix="mosta_fig4e_exact_style_"))
    os.environ.setdefault("NUMBA_CACHE_DIR", str(cache / "numba"))
    os.environ.setdefault("MPLCONFIGDIR", str(cache / "matplotlib"))
    os.environ.setdefault("MPLBACKEND", "Agg")

    import anndata as ad
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import pandas as pd
    sys.path.insert(0, str(plotter_path.parents[3]))
    from evaluation.arista_code.mosta_ported import velocity as velmod

    if Path(velmod.__file__).resolve() != plotter_path:
        raise RuntimeError(f"Imported unexpected plotter: {velmod.__file__}")
    adata = ad.AnnData(X=np.asarray(values["features"], dtype=np.float32))
    adata.obsm["X_spatial"] = coords
    adata.obs["telencephalon"] = pd.Categorical(
        labels, categories=CATEGORY_ORDER, ordered=True
    )

    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    panels = {}
    for panel, (space_title, component, array_key) in PANEL_SPECS.items():
        adata_panel = adata.copy()
        adata_panel.obsm[f"velocity_{component}_spatial"] = np.asarray(
            values[array_key], dtype=np.float32
        )
        fig, ax = velmod.plot_single_velocity_field(
            adata_panel,
            velocity_key=f"velocity_{component}",
            density=1.0,
            figsize=(7, 5),
            flip_y=False,
            flip_x=False,
            title=f"{space_title} - {component.capitalize()} Velocity (stream)",
            color_key="telencephalon",
            mode="default",
            remove_outliers=True,
            timepoint_str="E15.5",
            plot_region=ROI,
            palette=PALETTE,
        )
        panels[panel] = save_full_and_axes_crop(
            fig,
            ax,
            stem=f"Fig4e_{panel}_latest52D_m1024_exact_old_notebook",
            raw_dir=raw_dir,
            crop_dir=crop_dir,
            ai_ready_dir=ai_ready_dir,
            ai_ready_rect=AI_READY_RECTS[panel],
        )
        plt.close(fig)
    shutil.rmtree(cache, ignore_errors=True)

    for source, destination in (
        (numeric, provenance_dir / "numeric__fig4e_complete_brain_numeric_inputs.npz"),
        (gate_path, provenance_dir / "calculation_gate__calculation_gate.json"),
        (notebook, provenance_dir / notebook.name),
        (plotter_path, provenance_dir / "plotter__mosta_ported_velocity.py"),
        (Path(__file__).resolve(), provenance_dir / Path(__file__).name),
    ):
        shutil.copy2(source, destination)

    audit = {
        "schema_version": 1,
        "status": "PASS",
        "dataset": "MOSTA",
        "panel": "Fig4e",
        "calculation": {
            "accepted_latest_model": True,
            "complete_E15p5_Brain_n": 17071,
            "interaction_m": 1024,
            "generated_state": False,
            "ARISTA_data_labels_model_analysis_used": False,
        },
        "style": {
            "implementation": "direct call to archived plot_single_velocity_field",
            "figsize_inches": [7, 5],
            "density": 1.0,
            "smooth": 0.8,
            "min_mass": 1,
            "cutoff_perc": 3,
            "linewidth": 1.5,
            "arrow_size": 1.2,
            "point_alpha": 0.25,
            "point_size": 60,
            "remove_outliers": True,
            "flip_x": False,
            "flip_y": False,
            "plot_region": list(ROI),
            "legend_loc": "right margin",
            "titles": "historical notebook raw; omitted only from axes crop used by AI composition",
            "no_manual_reimplementation": True,
            "AI_composition_behavior": "raster scatter markers receive exact uniform AI scale; vector stream strokes and arrow effects retain historical point widths",
        },
        "inputs": {
            key: {"path": str(path), "sha256": identities[key]}
            for key, path in {
                "numeric": numeric,
                "calculation_gate": gate_path,
                "style_notebook": notebook,
                "plotter": plotter_path,
            }.items()
        },
        "panels": panels,
    }
    audit_path = out_dir / "render_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    checksum_paths = sorted(
        path for path in out_dir.rglob("*")
        if path.is_file() and path.name not in {"SHA256SUMS.json", "COMPLETE"}
    )
    checks = {str(path.relative_to(out_dir)): sha256(path) for path in checksum_paths}
    (out_dir / "SHA256SUMS.json").write_text(
        json.dumps(checks, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (out_dir / "COMPLETE").write_text("COMPLETE\n", encoding="utf-8")
    if args.freeze:
        freeze_tree(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
