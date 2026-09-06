"""Cell-type assignment for the zebrafish paper analyses."""
from pathlib import Path

import numpy as np

import CytoBridge as cb

CLASSIFIER_FILE = "classifier_cache/classifier_resmlp_25f65c49dc60ea4c.pt"


def load_classifier(data_dir, device="cpu"):
    path = Path(data_dir) / CLASSIFIER_FILE
    classifier = cb.tl.load_cached_mlp_classifier(str(path), device=device)
    expected = ("samples", *(f"x{i}" for i in range(1, 53)))
    if tuple(classifier.feature_cols) != expected or not classifier.include_time_feature:
        raise ValueError("The zebrafish classifier requires time, two spatial coordinates and 50 expression PCs")
    return classifier


def assign_cell_types(points, time, classifier, device="cpu"):
    """Predict from the original 52-dimensional state, then use 10-neighbor voting."""
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 52 or not np.isfinite(points).all():
        raise ValueError("Expected finite states with two spatial coordinates followed by 50 expression PCs")
    return np.asarray(cb.tl.predict_labels_for_points(
        points=points, time_value=float(time), model=classifier.model,
        label_encoder=classifier.label_encoder, feature_dim=52,
        device=device, knn_neighbors=10, include_time_feature=True,
    )).astype(str)
