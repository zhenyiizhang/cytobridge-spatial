"""Compatibility import for the single supported edge-predictor implementation.

Edge-predictor training belongs to preprocessing.  The implementation lives in
``CytoBridge.pp.edge_prediction``; this module remains only so older imports do
not silently execute a second, scientifically different training algorithm.
"""

from CytoBridge.pp.edge_prediction import (
    LinkPredictionDataset,
    LinkPredictorMLP,
    train_edge_predictor,
    vectorized_negative_sampling,
)

__all__ = [
    "LinkPredictionDataset",
    "LinkPredictorMLP",
    "train_edge_predictor",
    "vectorized_negative_sampling",
]
