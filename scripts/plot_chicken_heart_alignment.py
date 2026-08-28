#!/usr/bin/env python3
"""Plot raw, package-OT, and reviewed chicken-heart spatial coordinates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import anndata as ad
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


STAGES = ["D4", "D7", "D10", "D14"]
COLORS = {
    "D4": "#4477AA",
    "D7": "#66AEB4",
    "D10": "#EE8866",
    "D14": "#E6B84A",
}
PANELS = [
    ("spatial_original", "Raw spatial coordinates"),
    ("spatial_aligned", "Package OT alignment"),
    ("spatial_reviewed_reference", "Reviewed alignment"),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 9,
            "text.color": "black",
            "axes.labelcolor": "black",
            "axes.edgecolor": "black",
            "axes.titlecolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def padded_limits(coords: np.ndarray) -> tuple[tuple[float, float], tuple[float, float]]:
    xmin, ymin = np.nanmin(coords, axis=0)
    xmax, ymax = np.nanmax(coords, axis=0)
    xpad = max((xmax - xmin) * 0.045, 1e-6)
    ypad = max((ymax - ymin) * 0.045, 1e-6)
    return (xmin - xpad, xmax + xpad), (ymin - ypad, ymax + ypad)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-h5ad", type=Path, required=True)
    parser.add_argument(
        "--alignment-record",
        type=Path,
        help="Alignment JSON; defaults to alignment_manifest.json beside the H5AD",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    input_path = args.input_h5ad.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_stem = output_dir / "chicken_heart_raw_ot_reviewed_alignment"
    alignment_record = (
        args.alignment_record.expanduser().resolve()
        if args.alignment_record is not None
        else input_path.parent / "alignment_manifest.json"
    )

    set_style()
    adata = ad.read_h5ad(input_path, backed="r")
    timepoint = adata.obs["timepoint"].astype(str).to_numpy()

    fig, axes = plt.subplots(1, 3, figsize=(7.35, 3.05), constrained_layout=False)
    source_rows: list[pd.DataFrame] = []
    panel_letters = ["a", "b", "c"]

    for ax, letter, (key, title) in zip(axes, panel_letters, PANELS):
        coords = np.asarray(adata.obsm[key])[:, :2]
        for stage in STAGES:
            mask = timepoint == stage
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                s=5.0,
                c=COLORS[stage],
                alpha=0.72,
                linewidths=0,
                label=stage,
            )
        (xmin, xmax), (ymin, ymax) = padded_limits(coords)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(title, fontsize=10.2, fontweight="normal", pad=7)
        ax.set_xlabel("X coordinate", fontsize=9)
        if ax is axes[0]:
            ax.set_ylabel("Y coordinate", fontsize=9)
        else:
            ax.set_ylabel("")
        ax.tick_params(labelsize=8, length=2.8, width=0.7)
        ax.text(
            -0.14,
            1.08,
            letter,
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            va="top",
            ha="left",
            color="black",
        )
        source_rows.append(
            pd.DataFrame(
                {
                    "spot_id": adata.obs_names.astype(str),
                    "timepoint": timepoint,
                    "coordinate_source": key,
                    "x": coords[:, 0],
                    "y": coords[:, 1],
                }
            )
        )

    handles = [
        mpl.lines.Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markersize=5.0,
            markerfacecolor=COLORS[stage],
            markeredgewidth=0,
            label=stage,
        )
        for stage in STAGES
    ]
    fig.legend(
        handles=handles,
        labels=STAGES,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=4,
        frameon=False,
        fontsize=9,
        handletextpad=0.35,
        columnspacing=1.4,
    )
    fig.subplots_adjust(left=0.075, right=0.995, top=0.88, bottom=0.21, wspace=0.24)

    fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(
        out_stem.with_suffix(".png"),
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)

    source_csv = out_stem.with_name(out_stem.name + "_source.csv")
    pd.concat(source_rows, ignore_index=True).to_csv(source_csv, index=False)

    caption = (
        "Chicken heart spatial-coordinate alignment audit. (a) Raw coordinates from the "
        "four observed stages. (b) Coordinates produced by the CytoBridge package OT "
        "alignment after a predefined 180-degree correction of the D7 raw section. "
        "(c) Previously reviewed aligned coordinates shown only as an external visual "
        "reference; they were not used to fit the OT alignment."
    )
    out_stem.with_name(out_stem.name + "_caption.txt").write_text(caption + "\n")

    manifest = json.loads(alignment_record.read_text())
    provenance = {
        "figure": out_stem.name,
        "created_by": str(Path(__file__).resolve()),
        "input_h5ad": str(input_path),
        "input_h5ad_sha256": sha256(input_path),
        "source_data": str(source_csv.resolve()),
        "coordinate_panels": {letter: key for letter, (key, _) in zip(panel_letters, PANELS)},
        "stage_order": STAGES,
        "stage_colors": COLORS,
        "package_commit": "f5550e1",
        "fit_note": (
            "spatial_reviewed_reference was retained for post-hoc display only and was "
            "not passed to align_spatial."
        ),
        "d7_preorientation": manifest["d7_raw_orientation_correction"],
        "alignment_config": manifest["alignment_config"],
        "alignment_qc": {
            "anatomical_orientation": manifest["anatomical_orientation_qc"]["status"],
            "distance_preservation": manifest["distance_preservation"],
        },
    }
    out_stem.with_name(out_stem.name + "_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )

    print(out_stem.with_suffix(".pdf"))
    print(out_stem.with_suffix(".png"))


if __name__ == "__main__":
    main()
