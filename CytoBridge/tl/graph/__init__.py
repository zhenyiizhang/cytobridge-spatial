"""Graph interaction and edge prediction utilities."""

from .edge_predictor import train_edge_predictor

try:
    from .spatial_gnn import GNNInteraction
except (ImportError, ModuleNotFoundError, TypeError) as exc:
    if isinstance(exc, TypeError) and "NoneType takes no arguments" not in str(exc):
        raise
    GNNInteraction = None

__all__ = [
    "GNNInteraction",
    "train_edge_predictor",
]
