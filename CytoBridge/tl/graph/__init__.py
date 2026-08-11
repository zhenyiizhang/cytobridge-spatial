"""Graph interaction and edge prediction utilities."""

from .edge_predictor import train_edge_predictor
from .spatial_gnn import GNNInteraction

__all__ = [
    "GNNInteraction",
    "train_edge_predictor",
]
