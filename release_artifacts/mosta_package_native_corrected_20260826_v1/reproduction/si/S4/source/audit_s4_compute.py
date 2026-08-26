#!/usr/bin/env python3
"""Audit native-coordinate inputs for corrected MOSTA Supplementary Figure S4."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


DISPLAY = (
    ("0.0 (observed)", "s4/observed_t0.h5ad", "observed_real", 0.0),
    ("0.0 (generated)", "generated_states/time_0.h5ad", "generated_global_t0", 0.0),
    ("0.5", "generated_states/time_0p5.h5ad", "generated_global_t0", 0.5),
    ("1.0 (generated)", "generated_states/time_1.h5ad", "generated_global_t0", 1.0),
    ("1.5", "generated_states/time_1p5.h5ad", "generated_global_t0", 1.5),
    ("2.0 (generated)", "generated_states/time_2.h5ad", "generated_global_t0", 2.0),
    ("2.5", "generated_states/time_2p5.h5ad", "generated_global_t0", 2.5),
    ("3.0 (generated)", "generated_states/time_3.h5ad", "generated_global_t0", 3.0),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shared-root", required=True)
    parser.add_argument("--palette", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.shared_root).resolve()
    palette_path = Path(args.palette).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if not (root / "COMPLETE").is_file():
        raise RuntimeError("Shared compute is not sealed COMPLETE")
    palette = json.loads(palette_path.read_text(encoding="utf-8"))

    rows = []
    label_rows = []
    coords_by_title: dict[str, np.ndarray] = {}
    for order, (title, relative, expected_origin, time_value) in enumerate(DISPLAY, start=1):
        path = root / relative
        data = ad.read_h5ad(path, backed="r")
        state = np.asarray(data.X[:, :2], dtype=np.float64)
        spatial = np.asarray(data.obsm["spatial"], dtype=np.float64)
        if state.shape != spatial.shape or not np.allclose(state, spatial, rtol=0.0, atol=1e-6):
            raise RuntimeError(f"Stored spatial coordinates differ from native state columns: {path}")
        if not np.isfinite(spatial).all():
            raise RuntimeError(f"Non-finite coordinates: {path}")
        labels = data.obs["Annotation"].astype(str).to_numpy()
        unknown = sorted(set(labels).difference(palette))
        if unknown:
            raise RuntimeError(f"Palette lacks labels at {title}: {unknown}")
        origin = str(data.uns.get("slice_origin"))
        source_anchor = float(data.uns.get("source_anchor_time"))
        warp = bool(data.uns.get("spatial_warp"))
        if origin != expected_origin or source_anchor != 0.0 or warp:
            raise RuntimeError(
                f"Origin contract failed at {title}: origin={origin}, source_anchor={source_anchor}, warp={warp}"
            )
        quantiles = np.quantile(spatial, [0.001, 0.01, 0.5, 0.99, 0.999], axis=0)
        robust_span = quantiles[3] - quantiles[1]
        central_span = np.quantile(spatial, 0.95, axis=0) - np.quantile(spatial, 0.05, axis=0)
        tree = cKDTree(spatial)
        nearest = tree.query(spatial, k=2, workers=-1)[0][:, 1]
        row = {
            "display_order": order,
            "title": title,
            "time": time_value,
            "origin": origin,
            "source_anchor_time": source_anchor,
            "spatial_warp": warp,
            "n_cells": int(spatial.shape[0]),
            "n_labels": int(pd.Series(labels).nunique()),
            "x_q001": float(quantiles[0, 0]),
            "x_q01": float(quantiles[1, 0]),
            "x_median": float(quantiles[2, 0]),
            "x_q99": float(quantiles[3, 0]),
            "x_q999": float(quantiles[4, 0]),
            "y_q001": float(quantiles[0, 1]),
            "y_q01": float(quantiles[1, 1]),
            "y_median": float(quantiles[2, 1]),
            "y_q99": float(quantiles[3, 1]),
            "y_q999": float(quantiles[4, 1]),
            "robust_x_span_q01_q99": float(robust_span[0]),
            "robust_y_span_q01_q99": float(robust_span[1]),
            "robust_area_proxy": float(np.prod(robust_span)),
            "robust_aspect_y_over_x": float(robust_span[1] / robust_span[0]),
            "central_x_span_q05_q95": float(central_span[0]),
            "central_y_span_q05_q95": float(central_span[1]),
            "nearest_neighbor_q50": float(np.quantile(nearest, 0.5)),
            "nearest_neighbor_q99": float(np.quantile(nearest, 0.99)),
            "nearest_neighbor_q999": float(np.quantile(nearest, 0.999)),
            "nearest_neighbor_max": float(nearest.max()),
            "path": str(path),
            "sha256": sha256(path),
        }
        rows.append(row)
        counts = pd.Series(labels).value_counts(sort=False)
        for label, count in counts.items():
            label_rows.append(
                {
                    "display_order": order,
                    "title": title,
                    "time": time_value,
                    "origin": origin,
                    "celltype": str(label),
                    "count": int(count),
                    "fraction": float(count / spatial.shape[0]),
                    "color": palette[str(label)],
                }
            )
        coords_by_title[title] = spatial
        data.file.close()

    metrics = pd.DataFrame(rows)
    metrics.to_csv(output / "s4_native_geometry_metrics.csv", index=False)
    pd.DataFrame(label_rows).to_csv(output / "s4_label_composition.csv", index=False)

    observed = metrics.iloc[0]
    generated_t0 = metrics.iloc[1]
    t0_center_delta = np.asarray(
        [generated_t0["x_median"] - observed["x_median"], generated_t0["y_median"] - observed["y_median"]]
    )
    t0_span_ratio = np.asarray(
        [
            generated_t0["robust_x_span_q01_q99"] / observed["robust_x_span_q01_q99"],
            generated_t0["robust_y_span_q01_q99"] / observed["robust_y_span_q01_q99"],
        ]
    )
    generated_metrics = metrics.iloc[1:].reset_index(drop=True)
    area = generated_metrics["robust_area_proxy"].to_numpy(dtype=float)
    aspect = generated_metrics["robust_aspect_y_over_x"].to_numpy(dtype=float)
    adjacent_area_ratio = area[1:] / area[:-1]
    adjacent_aspect_ratio = aspect[1:] / aspect[:-1]
    audit = {
        "schema_version": 1,
        "status": "PASS",
        "panel": "Supplementary Figure S4",
        "shared_complete": True,
        "display_titles": [title for title, _, _, _ in DISPLAY],
        "origin_contract": {
            "observed_panels": 1,
            "generated_global_t0_panels": 7,
            "all_generated_source_anchor_time": 0.0,
            "spatial_warp": False,
        },
        "t0_observed_vs_generated": {
            "median_coordinate_delta": t0_center_delta.tolist(),
            "robust_span_ratio_generated_over_observed": t0_span_ratio.tolist(),
            "sampling_interpretation": "generated t0 is the 50,000-particle global-t0 initial sample",
        },
        "generated_geometry": {
            "robust_area_proxy_by_time": dict(
                zip(generated_metrics["time"].astype(str), generated_metrics["robust_area_proxy"].astype(float))
            ),
            "adjacent_area_ratios": adjacent_area_ratio.tolist(),
            "adjacent_aspect_ratios": adjacent_aspect_ratio.tolist(),
            "minimum_adjacent_area_ratio": float(adjacent_area_ratio.min()),
            "maximum_adjacent_area_ratio": float(adjacent_area_ratio.max()),
            "minimum_adjacent_aspect_ratio": float(adjacent_aspect_ratio.min()),
            "maximum_adjacent_aspect_ratio": float(adjacent_aspect_ratio.max()),
        },
        "palette": {"path": str(palette_path), "sha256": sha256(palette_path), "unknown_labels": []},
        "checks": {
            "native_state_coordinates_equal_saved_spatial": True,
            "finite_coordinates": True,
            "no_warp_rotation_or_stretch": True,
            "palette_complete": True,
            "t0_span_ratio_within_two_percent": bool(np.all((t0_span_ratio > 0.98) & (t0_span_ratio < 1.02))),
            "no_generated_adjacent_area_collapse_below_0p75": bool(np.all(adjacent_area_ratio >= 0.75)),
            "no_generated_adjacent_area_explosion_above_1p5": bool(np.all(adjacent_area_ratio <= 1.5)),
        },
    }
    if not all(audit["checks"].values()):
        audit["status"] = "REVIEW_REQUIRED"
    (output / "s4_numerical_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0 if audit["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
