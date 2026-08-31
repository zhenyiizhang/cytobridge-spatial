from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.linalg import orthogonal_procrustes


PROJECT_ROOT = Path("/data/cytobridge/projects/CytoBridge-ST-1104")
ACCEPTED = PROJECT_ROOT / "runs/chicken-heart-full-ot-20260823-r2/preprocess/chicken_heart_aligned.h5ad"
AUDIT = PROJECT_ROOT / "runs/chicken-heart-alignment-sensitivity-audit-20260831-r1"
VARIANTS = (
    "baseline_repeat",
    "translate_low",
    "translate_moderate",
    "translate_strong",
    "rotate_low",
    "rotate_moderate",
    "rotate_strong",
    "translate_rotate_low",
    "translate_rotate_moderate",
    "translate_rotate_strong",
)
TIME_ORDER = ("D4", "D7", "D10", "D14")


def _fit(source: np.ndarray, target: np.ndarray, allow_reflection: bool) -> tuple[np.ndarray, float, float]:
    rotation, _ = orthogonal_procrustes(source, target)
    original_det = float(np.linalg.det(rotation))
    if not allow_reflection and original_det < 0:
        u, _, vt = np.linalg.svd(source.T @ target)
        u[:, -1] *= -1
        rotation = u @ vt
    residual = float(np.sqrt(np.mean((source @ rotation - target) ** 2)))
    return rotation, original_det, residual


def main() -> None:
    rows = []
    for variant in VARIANTS:
        reference = ACCEPTED if variant == "baseline_repeat" else (
            AUDIT / "runs/baseline_repeat/preprocess/chicken_heart_aligned.h5ad"
        )
        baseline = sc.read_h5ad(reference)
        baseline_ids = pd.Index(baseline.obs_names.astype(str))
        path = AUDIT / "runs" / variant / "preprocess/chicken_heart_aligned.h5ad"
        current = sc.read_h5ad(path)
        positions = pd.Index(current.obs_names.astype(str)).get_indexer(baseline_ids)
        for timepoint in TIME_ORDER:
            mask = np.asarray(baseline.obs["timepoint"].astype(str) == timepoint)
            x = np.asarray(current.obsm["spatial_aligned"], dtype=float)[positions][mask]
            y = np.asarray(baseline.obsm["spatial_aligned"], dtype=float)[mask]
            x = x - x.mean(axis=0, keepdims=True)
            y = y - y.mean(axis=0, keepdims=True)
            proper_r, unconstrained_det, proper_residual = _fit(x, y, allow_reflection=False)
            _, _, unconstrained_residual = _fit(x, y, allow_reflection=True)
            scale = float(np.sum((x @ proper_r) * y) / np.sum(x * x))
            scaled_residual = float(np.sqrt(np.mean((scale * x @ proper_r - y) ** 2)))
            radius = float(np.sqrt(np.mean(np.sum(y * y, axis=1))))
            rows.append(
                {
                    "variant": variant,
                    "reference": "accepted_baseline" if variant == "baseline_repeat" else "baseline_repeat",
                    "timepoint": timepoint,
                    "unconstrained_det": unconstrained_det,
                    "proper_residual_fraction_radius": proper_residual / radius,
                    "unconstrained_residual_fraction_radius": unconstrained_residual / radius,
                    "scaled_proper_residual_fraction_radius": scaled_residual / radius,
                    "best_scale": scale,
                    "proper_rotation_deg": float(
                        np.degrees(np.arctan2(proper_r[0, 1], proper_r[0, 0]))
                    ),
                }
            )
    output = AUDIT / "summary" / "coordinate_diagnostic.csv"
    pd.DataFrame(rows).to_csv(output, index=False)


if __name__ == "__main__":
    main()
