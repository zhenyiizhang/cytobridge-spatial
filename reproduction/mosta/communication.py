"""Spatial anchors for the communication arrows in Figure 4d."""
from __future__ import annotations
from typing import Any, Sequence
import numpy as np
import pandas as pd

def compute_legacy_communication_centroids(
    coordinates: np.ndarray,
    labels: Sequence[object],
    node_labels: Sequence[object],
    *,
    top_n_y: int = 200,
    top_n_y_exclusions: Sequence[str] = ("Brain", "Meninges", "Choroid plexus"),
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    """Compute notebook centroids, including its optional top-Y placement."""

    coords = np.asarray(coordinates, dtype=float)
    values = np.asarray(labels).astype(str)
    if coords.ndim != 2 or coords.shape[1] != 2 or len(coords) != len(values):
        raise ValueError("Coordinates and labels must align with shape (n, 2).")
    exclusions = set(map(str, top_n_y_exclusions))
    centroids: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    for label in sorted(set(map(str, node_labels))):
        subset = coords[values == label]
        subset = subset[np.isfinite(subset).all(axis=1)]
        if len(subset) == 0:
            raise ValueError(
                f"No finite observed coordinates exist for node {label!r}."
            )
        use_top_y = label not in exclusions and int(top_n_y) > 0
        if use_top_y and len(subset) > int(top_n_y):
            order = np.argsort(subset[:, 1], kind="stable")[-int(top_n_y) :]
            centroid_rows = subset[order]
        else:
            centroid_rows = subset
        centroid = centroid_rows.mean(axis=0)
        centroids[label] = centroid
        rows.append(
            {
                "node": label,
                "n_available": int(len(subset)),
                "n_used": int(len(centroid_rows)),
                "top_n_y_rule_applied": bool(use_top_y and len(subset) > int(top_n_y)),
                "centroid_x": float(centroid[0]),
                "centroid_y": float(centroid[1]),
            }
        )
    return centroids, pd.DataFrame(rows)
