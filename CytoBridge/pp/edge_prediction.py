import os
import gzip
import pickle
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import scanpy as sc
from scipy.spatial import cKDTree
from torch.utils.data import Dataset
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm import tqdm
from typing import Dict, List, Tuple

# --- Helper Classes ---


class LinkPredictorMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super(LinkPredictorMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        return self.network(x)


class LinkPredictionDataset(Dataset):
    def __init__(self, edges, labels, node_features, time_indices):
        self.edges = edges
        self.labels = labels
        self.node_features = node_features
        self.time_indices = time_indices

    def __len__(self):
        return len(self.edges)

    def __getitem__(self, idx):
        edge = self.edges[idx]
        label = self.labels[idx]
        time_idx = self.time_indices[idx]

        u_idx, v_idx = edge[0], edge[1]
        u_features = self.node_features[time_idx][u_idx]
        v_features = self.node_features[time_idx][v_idx]

        combined_features = torch.cat([u_features, v_features])

        return combined_features, torch.tensor(label, dtype=torch.float32)


# --- Core Functionality ---


def vectorized_negative_sampling(
    positive_edges,
    num_nodes,
    num_neg_samples: int | None,
    spatial_coords,
    distance_threshold: float,
):
    """
    Efficient negative sampling via spatial candidate pool. Candidate pairs use
    the interaction model's strict ``1e-6 < distance < distance_threshold``
    contract before known positive pairs are removed.
    """
    positive_edges = np.asarray(positive_edges, dtype=np.int64)
    target = None if num_neg_samples is None else int(num_neg_samples)
    if (target is not None and target <= 0) or int(num_nodes) <= 1:
        return np.empty((0, 2), dtype=np.int32)

    coords = np.asarray(spatial_coords, dtype=np.float64)
    tree = cKDTree(coords)
    neighbors = tree.query_ball_point(coords, r=float(distance_threshold))

    candidate_count = int(sum(max(0, len(vs) - 1) for vs in neighbors))
    if candidate_count <= 0:
        return np.empty((0, 2), dtype=np.int32)

    cand_u = np.empty(candidate_count, dtype=np.int32)
    cand_v = np.empty(candidate_count, dtype=np.int32)
    cursor = 0
    for u, vs in enumerate(neighbors):
        if len(vs) <= 1:
            continue
        vs_arr = np.asarray(vs, dtype=np.int32)
        vs_arr = vs_arr[vs_arr != u]
        if vs_arr.size:
            deltas = coords[vs_arr] - coords[u]
            distances = np.sqrt(np.einsum("ij,ij->i", deltas, deltas))
            vs_arr = vs_arr[
                (distances > 1e-6) & (distances < float(distance_threshold))
            ]
        if vs_arr.size == 0:
            continue
        n = int(vs_arr.size)
        cand_u[cursor : cursor + n] = u
        cand_v[cursor : cursor + n] = vs_arr
        cursor += n

    if cursor == 0:
        return np.empty((0, 2), dtype=np.int32)
    if cursor < candidate_count:
        cand_u = cand_u[:cursor]
        cand_v = cand_v[:cursor]

    cand_ids = cand_u.astype(np.int64) * int(num_nodes) + cand_v.astype(np.int64)
    if positive_edges.size > 0:
        pos_ids = positive_edges[:, 0] * int(num_nodes) + positive_edges[:, 1]
        pos_ids = np.unique(pos_ids)
        pos_ids.sort()
        idx = np.searchsorted(pos_ids, cand_ids)
        valid = idx < pos_ids.size
        is_pos = np.zeros(cand_ids.shape[0], dtype=bool)
        if np.any(valid):
            is_pos[valid] = pos_ids[idx[valid]] == cand_ids[valid]
        neg_mask = ~is_pos
    else:
        neg_mask = np.ones(cand_ids.shape[0], dtype=bool)

    neg_u = cand_u[neg_mask]
    neg_v = cand_v[neg_mask]
    if neg_u.size == 0:
        return np.empty((0, 2), dtype=np.int32)

    sample_n = int(neg_u.size) if target is None else min(target, int(neg_u.size))
    if sample_n == int(neg_u.size):
        return np.stack([neg_u, neg_v], axis=1).astype(np.int32, copy=False)
    chosen = np.random.choice(int(neg_u.size), size=sample_n, replace=False)
    sampled = np.stack([neg_u[chosen], neg_v[chosen]], axis=1)
    return sampled.astype(np.int32, copy=False)


def _resolve_spatial_key_from_adata(
    adata, preferred_key: str = "spatial_aligned"
) -> str:
    if preferred_key in adata.obsm:
        return preferred_key
    if "spatial_aligned" in adata.obsm:
        return "spatial_aligned"
    if "spatial" in adata.obsm:
        return "spatial"
    if "spatial_x" in adata.obs and "spatial_y" in adata.obs:
        adata.obsm["spatial"] = np.column_stack(
            (adata.obs["spatial_x"], adata.obs["spatial_y"])
        )
        return "spatial"
    raise KeyError(
        "No spatial coordinates found in adata.obsm['spatial_aligned'] or "
        "adata.obsm['spatial'] or obs['spatial_x/y']."
    )


def _load_features_from_adata(
    adata_or_h5ad,
    *,
    time_key: str,
    latent_key: str,
    spatial_key: str = "spatial_aligned",
) -> Tuple[List[float], Dict[float, torch.Tensor]]:
    if isinstance(adata_or_h5ad, str):
        adata = sc.read_h5ad(adata_or_h5ad)
    else:
        adata = adata_or_h5ad

    if time_key not in adata.obs:
        raise KeyError(f"time_key '{time_key}' not found in adata.obs")
    if latent_key not in adata.obsm:
        raise KeyError(f"latent_key '{latent_key}' not found in adata.obsm")
    spatial_key_used = _resolve_spatial_key_from_adata(adata, preferred_key=spatial_key)

    time_vals_raw = pd.to_numeric(adata.obs[time_key], errors="coerce")
    if time_vals_raw.isna().any():
        raise ValueError(
            f"time_key '{time_key}' contains non-numeric values. "
            "Please use preprocess-generated numeric time (e.g., time_point_processed)."
        )
    time_points = sorted(time_vals_raw.unique())
    spatial_all = np.asarray(adata.obsm[spatial_key_used], dtype=np.float32)
    latent_all = np.asarray(adata.obsm[latent_key], dtype=np.float32)
    if spatial_all.shape[0] != latent_all.shape[0]:
        raise ValueError(
            f"Row mismatch between spatial '{spatial_key_used}' ({spatial_all.shape[0]}) "
            f"and latent '{latent_key}' ({latent_all.shape[0]})."
        )

    node_features_by_time: Dict[float, torch.Tensor] = {}
    for t_val in time_points:
        mask = (time_vals_raw == t_val).to_numpy()
        spatial = spatial_all[mask]
        latent = latent_all[mask]
        features = np.concatenate((spatial, latent), axis=1)
        node_features_by_time[float(t_val)] = torch.from_numpy(features).float()
    return [float(t) for t in time_points], node_features_by_time


def _load_features_from_csv(
    feature_csv_path: str,
) -> Tuple[List[float], Dict[float, torch.Tensor]]:
    df = pd.read_csv(feature_csv_path)
    if "samples" not in df.columns:
        raise KeyError("CSV must include a 'samples' column.")
    time_points = sorted(pd.to_numeric(df["samples"], errors="raise").unique())
    node_features_by_time: Dict[float, torch.Tensor] = {}
    for t_val in time_points:
        data_i = df[df["samples"] == t_val].iloc[:, 1:].values
        node_features_by_time[float(t_val)] = torch.tensor(data_i).float()
    return [float(t) for t in time_points], node_features_by_time


def _resolve_distance_threshold(distance_threshold, adata_or_h5ad) -> float:
    if distance_threshold is not None and float(distance_threshold) > 0:
        return float(distance_threshold)
    if adata_or_h5ad is None:
        raise ValueError(
            "distance_threshold is not set and adata_or_h5ad is None. "
            "Provide distance_threshold explicitly or pass AnnData/h5ad with "
            "uns['interaction_graph']['neighborhood_threshold']."
        )
    if isinstance(adata_or_h5ad, str):
        adata = sc.read_h5ad(adata_or_h5ad)
    else:
        adata = adata_or_h5ad
    ig = adata.uns.get("interaction_graph", {})
    threshold = ig.get("neighborhood_threshold", None)
    if threshold is None:
        raise KeyError(
            "distance_threshold is not set and adata.uns['interaction_graph']['neighborhood_threshold'] is missing."
        )
    return float(threshold)


def _resolve_graph_path_candidates(
    data_name: str,
    graph_input_dir: str,
    time_idx: int,
    t_val: float,
) -> list[str]:
    candidates: list[str] = []
    # Canonical naming: use time order index (t0, t1, ...)
    slice_name_idx = f"{data_name}_t{time_idx}"
    candidates.append(
        os.path.join(
            graph_input_dir, slice_name_idx, f"{slice_name_idx}_adjacency_records"
        )
    )

    # Backward compatibility: integer-like time value naming.
    t_float = float(t_val)
    t_int = int(round(t_float))
    if np.isclose(t_float, t_int):
        slice_name_int = f"{data_name}_t{t_int}"
        path_int = os.path.join(
            graph_input_dir, slice_name_int, f"{slice_name_int}_adjacency_records"
        )
        if path_int not in candidates:
            candidates.append(path_int)

    # Backward compatibility: raw float/string value naming.
    t_str = str(t_val)
    slice_name_raw = f"{data_name}_t{t_str}"
    path_raw = os.path.join(
        graph_input_dir, slice_name_raw, f"{slice_name_raw}_adjacency_records"
    )
    if path_raw not in candidates:
        candidates.append(path_raw)
    return candidates


def _validate_edges_with_features(
    edges: np.ndarray,
    num_nodes_features: int,
) -> tuple[bool, str]:
    if edges.size == 0:
        return True, "empty_edges"
    if edges.ndim != 2 or edges.shape[1] != 2:
        return False, f"invalid edge shape {edges.shape}, expected [E,2]"
    edge_min = int(edges.min())
    edge_max = int(edges.max())
    if edge_min < 0:
        return False, f"negative edge index found: min={edge_min}"
    if edge_max >= int(num_nodes_features):
        return False, (
            f"edge index out of bounds: max_edge={edge_max}, "
            f"num_nodes_features={num_nodes_features}"
        )
    return True, f"ok(max_edge={edge_max}, n={num_nodes_features})"


def _unique_directed_edges(edges: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    """Collapse LR multiedges to one positive example per directed cell pair."""
    edges = np.asarray(edges, dtype=np.int32)
    if edges.size == 0:
        unique = np.empty((0, 2), dtype=np.int32)
    else:
        unique = np.unique(edges.reshape(-1, 2), axis=0).astype(np.int32, copy=False)
    raw_count = int(edges.shape[0]) if edges.ndim == 2 else 0
    unique_count = int(unique.shape[0])
    return unique, {
        "raw": raw_count,
        "unique": unique_count,
        "duplicates_removed": raw_count - unique_count,
    }


def _validate_positive_spatial_contract(
    edges: np.ndarray,
    spatial_coords: np.ndarray,
    distance_threshold: float,
    *,
    time_value: float,
) -> None:
    """Reject graph positives outside the predictor's candidate universe."""

    if edges.size == 0:
        return
    coordinates = np.asarray(spatial_coords, dtype=np.float64)
    deltas = coordinates[edges[:, 0]] - coordinates[edges[:, 1]]
    distances = np.sqrt(np.einsum("ij,ij->i", deltas, deltas))
    valid = (distances > 1e-6) & (distances < float(distance_threshold))
    if np.all(valid):
        return
    bad = np.flatnonzero(~valid)
    examples = [
        {
            "edge": [int(edges[index, 0]), int(edges[index, 1])],
            "distance": float(distances[index]),
        }
        for index in bad[:3]
    ]
    raise ValueError(
        "Positive interaction edges must belong to the same spatial candidate "
        "universe used for edge-predictor deployment "
        f"(1e-6 < distance < {float(distance_threshold):g}). "
        f"time={float(time_value):g}, invalid={bad.size}, examples={examples}"
    )


def _load_adjacency_with_compatibility(
    candidate_paths: list[str],
    num_nodes_features: int,
) -> tuple[np.ndarray, str]:
    failure_reasons: list[str] = []
    for path in candidate_paths:
        if not os.path.exists(path):
            failure_reasons.append(f"{path} [missing]")
            continue
        with gzip.open(path, "rb") as f:
            adjacency_records = pickle.load(f)
        edges = np.asarray(adjacency_records[0], dtype=np.int32)
        # ``np.asarray([])`` has shape ``(0,)`` whereas non-empty edge lists
        # have shape ``(E, 2)``.  Keep an empty slice in the canonical edge
        # shape so it can be safely stacked with non-empty time slices.
        if edges.size == 0:
            edges = np.empty((0, 2), dtype=np.int32)
        ok, reason = _validate_edges_with_features(edges, num_nodes_features)
        if ok:
            return edges, path
        failure_reasons.append(f"{path} [{reason}]")

    raise ValueError(
        "No compatible adjacency_records file found for this time slice. "
        "Candidates checked:\n- " + "\n- ".join(failure_reasons)
    )


def _find_best_threshold(
    labels: np.ndarray,
    probs: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> tuple[float, float, float]:
    labels = np.asarray(labels).reshape(-1)
    probs = np.asarray(probs).reshape(-1)
    if labels.shape[0] == 0:
        return 0.5, 0.0, 0.0

    if thresholds is None:
        # Evaluate every distinct prediction boundary in O(N log N).  This is
        # both more precise and substantially cheaper than repeatedly scanning
        # a large deployment candidate set at an arbitrary threshold grid.
        order = np.argsort(-probs, kind="stable")
        sorted_probs = probs[order]
        sorted_labels = labels[order].astype(np.int64, copy=False)
        true_positives = np.cumsum(sorted_labels)
        false_positives = np.cumsum(1 - sorted_labels)
        group_ends = np.flatnonzero(np.r_[sorted_probs[:-1] != sorted_probs[1:], True])
        tp = true_positives[group_ends]
        fp = false_positives[group_ends]
        total_positive = int(np.sum(sorted_labels))
        false_negatives = total_positive - tp
        denominator = 2 * tp + fp + false_negatives
        f1_values = np.divide(
            2 * tp,
            denominator,
            out=np.zeros_like(tp, dtype=np.float64),
            where=denominator > 0,
        )
        true_negatives = (labels.size - total_positive) - fp
        accuracy_values = (tp + true_negatives) / max(1, labels.size)
        best_f1 = float(np.max(f1_values))
        tied = np.flatnonzero(np.isclose(f1_values, best_f1))
        best_group = int(tied[np.argmax(accuracy_values[tied])])
        best_end = int(group_ends[best_group])
        return (
            float(sorted_probs[best_end]),
            best_f1,
            float(accuracy_values[best_group]),
        )

    best_threshold = 0.5
    best_f1 = -1.0
    best_acc = 0.0
    for th in thresholds:
        pred = (probs >= float(th)).astype(int)
        f1 = f1_score(labels, pred, zero_division=0)
        acc = accuracy_score(labels, pred)
        if (f1 > best_f1) or (np.isclose(f1, best_f1) and acc > best_acc):
            best_f1 = float(f1)
            best_acc = float(acc)
            best_threshold = float(th)
    return best_threshold, best_f1, best_acc


def _binary_metrics(
    labels: np.ndarray,
    probs: np.ndarray,
    threshold: float,
) -> dict[str, float | int | None]:
    """Return interpretable metrics at the deployment decision threshold."""

    labels = np.asarray(labels).reshape(-1).astype(np.int64, copy=False)
    probs = np.asarray(probs).reshape(-1)
    if labels.size == 0:
        return {
            "n_candidates": 0,
            "positive_fraction": None,
            "average_precision": None,
            "roc_auc": None,
            "f1": None,
            "precision": None,
            "recall": None,
            "accuracy": None,
            "predicted_edge_fraction": None,
        }
    predictions = (probs >= float(threshold)).astype(np.int64)
    has_positive = bool(np.any(labels == 1))
    has_both_classes = np.unique(labels).size == 2
    return {
        "n_candidates": int(labels.size),
        "positive_fraction": float(np.mean(labels)),
        "average_precision": (
            float(average_precision_score(labels, probs)) if has_positive else None
        ),
        "roc_auc": (float(roc_auc_score(labels, probs)) if has_both_classes else None),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "predicted_edge_fraction": float(np.mean(predictions)),
    }


def _metrics_by_time(
    labels: np.ndarray,
    probs: np.ndarray,
    time_indices: np.ndarray,
    time_values: list[float],
    threshold: float,
) -> list[dict]:
    rows = []
    for local_index in sorted(np.unique(time_indices).astype(int).tolist()):
        mask = time_indices == local_index
        rows.append(
            {
                "time_index": int(local_index),
                "time_value": float(time_values[local_index]),
                **_binary_metrics(labels[mask], probs[mask], threshold),
            }
        )
    return rows


def _sample_train_indices_for_epoch(
    train_pool_idx: np.ndarray,
    *,
    train_sample_ratio_per_epoch: float,
    max_train_edges_per_epoch: int | None,
) -> np.ndarray:
    total = int(train_pool_idx.shape[0])
    if total == 0:
        return train_pool_idx

    ratio = float(train_sample_ratio_per_epoch)
    if ratio <= 0 or ratio > 1:
        raise ValueError("train_sample_ratio_per_epoch must be in (0, 1].")

    target = total
    if ratio < 1.0:
        target = min(target, max(1, int(round(total * ratio))))
    if max_train_edges_per_epoch is not None and int(max_train_edges_per_epoch) > 0:
        target = min(target, int(max_train_edges_per_epoch))

    if target >= total:
        return train_pool_idx

    chosen = np.random.choice(total, size=target, replace=False)
    return train_pool_idx[chosen]


def _balanced_training_pool(
    split_indices: np.ndarray,
    labels: np.ndarray,
    *,
    random_seed: int,
) -> np.ndarray:
    """Balance only the optimization pool, leaving validation prevalence intact."""

    split_indices = np.asarray(split_indices, dtype=np.int64)
    positive = split_indices[labels[split_indices] == 1]
    negative = split_indices[labels[split_indices] == 0]
    if positive.size == 0 or negative.size == 0:
        raise ValueError("Training partition must contain positive and negative edges.")
    sample_size = min(int(positive.size), int(negative.size))
    rng = np.random.default_rng(int(random_seed))
    if positive.size > sample_size:
        positive = rng.choice(positive, size=sample_size, replace=False)
    if negative.size > sample_size:
        negative = rng.choice(negative, size=sample_size, replace=False)
    balanced = np.concatenate([positive, negative]).astype(np.int64, copy=False)
    return balanced[rng.permutation(balanced.size)]


def _build_combined_edge_features(
    edges: np.ndarray,
    time_indices: np.ndarray,
    node_features: list[torch.Tensor],
) -> torch.Tensor:
    """Vectorized edge feature materialization to avoid per-sample __getitem__ overhead."""
    if edges.shape[0] == 0:
        feat_dim = int(node_features[0].shape[1]) * 2
        return torch.empty((0, feat_dim), dtype=torch.float32)

    feat_dim = int(node_features[0].shape[1])
    out = torch.empty((edges.shape[0], feat_dim * 2), dtype=torch.float32)
    unique_t = np.unique(time_indices)
    for t in unique_t:
        mask = time_indices == t
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            continue
        e = edges[idx]
        u_idx = torch.from_numpy(e[:, 0].astype(np.int64, copy=False))
        v_idx = torch.from_numpy(e[:, 1].astype(np.int64, copy=False))
        node_t = node_features[int(t)]
        out[idx] = torch.cat((node_t[u_idx], node_t[v_idx]), dim=1)
    return out


def _iter_edge_batches(
    edges: np.ndarray,
    labels: np.ndarray,
    time_indices: np.ndarray,
    node_features: list[torch.Tensor],
    *,
    batch_size: int,
    shuffle: bool,
    random_seed: int,
):
    """Build only one feature batch at a time to bound formal-run memory."""

    order = np.arange(edges.shape[0], dtype=np.int64)
    if shuffle and order.size:
        order = np.random.default_rng(int(random_seed)).permutation(order)
    for start in range(0, order.size, int(batch_size)):
        batch_indices = order[start : start + int(batch_size)]
        yield (
            _build_combined_edge_features(
                edges[batch_indices], time_indices[batch_indices], node_features
            ),
            torch.from_numpy(labels[batch_indices].astype(np.float32, copy=False)),
        )


def _predict_edge_probabilities(
    model: nn.Module,
    edges: np.ndarray,
    labels: np.ndarray,
    time_indices: np.ndarray,
    node_features: list[torch.Tensor],
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    probability_chunks = []
    label_chunks = []
    with torch.no_grad():
        for features, batch_labels in _iter_edge_batches(
            edges,
            labels,
            time_indices,
            node_features,
            batch_size=batch_size,
            shuffle=False,
            random_seed=0,
        ):
            probabilities = torch.sigmoid(model(features.to(device))).cpu()
            probability_chunks.append(probabilities.numpy().reshape(-1))
            label_chunks.append(batch_labels.numpy().reshape(-1))
    return np.concatenate(probability_chunks), np.concatenate(label_chunks)


def _indices_for_time_groups(
    time_indices: np.ndarray,
    labels: np.ndarray,
    *,
    random_seed: int,
    validation_ratio: float,
    test_ratio: float,
) -> tuple[dict[str, np.ndarray], dict] | None:
    """Randomly hold out observed time slices for a sensitivity experiment.

    This is not a future-time or forecasting split: assigning observed slices
    at random does not preserve temporal order. Production presets use the
    node-disjoint strategy so every observed time contributes training data.
    """
    all_times = [int(time_idx) for time_idx in np.unique(time_indices)]
    informative = [
        int(time_idx)
        for time_idx in all_times
        if np.unique(labels[time_indices == time_idx]).size == 2
    ]
    if len(informative) < 2:
        return None

    rng = np.random.default_rng(int(random_seed))
    shuffled = list(np.asarray(informative)[rng.permutation(len(informative))])
    if len(shuffled) == 2:
        groups = {"train": shuffled[:1], "validation": shuffled[1:], "test": []}
    else:
        n_test = (
            max(1, int(round(len(shuffled) * float(test_ratio))))
            if float(test_ratio) > 0
            else 0
        )
        n_validation = max(1, int(round(len(shuffled) * float(validation_ratio))))
        while n_test + n_validation >= len(shuffled):
            if n_test >= n_validation and n_test > 1:
                n_test -= 1
            elif n_validation > 1:
                n_validation -= 1
            else:
                return None
        groups = {
            "test": shuffled[:n_test],
            "validation": shuffled[n_test : n_test + n_validation],
            "train": shuffled[n_test + n_validation :],
        }

    split_indices = {
        name: np.flatnonzero(np.isin(time_indices, values)).astype(np.int64)
        for name, values in groups.items()
    }
    if any(
        split_indices[name].size == 0
        or np.unique(labels[split_indices[name]]).size != 2
        for name in ("train", "validation")
    ):
        return None

    metadata = {
        "strategy": "time_group_holdout",
        "interpretation": (
            "random observed-time holdout sensitivity; not future-time "
            "generalization"
        ),
        "train_time_indices": [int(value) for value in groups["train"]],
        "validation_time_indices": [int(value) for value in groups["validation"]],
        "test_time_indices": [int(value) for value in groups["test"]],
        "excluded_time_indices_without_both_classes": sorted(
            set(all_times).difference(informative)
        ),
        "test_available": bool(split_indices["test"].size),
    }
    return split_indices, metadata


def _node_disjoint_indices(
    edges: np.ndarray,
    labels: np.ndarray,
    time_indices: np.ndarray,
    node_counts: list[int],
    *,
    random_seed: int,
    validation_ratio: float,
    test_ratio: float,
) -> tuple[dict[str, np.ndarray], dict]:
    """Split by nodes, then retain only edges internal to one split.

    This is used when complete time-slice holdout is impossible. It guarantees
    that a node appearing in validation or test never appears in training.
    """
    if sum(count >= 6 for count in node_counts) == 0:
        raise ValueError(
            "A leakage-free edge-predictor validation split needs at least one "
            "time slice with six nodes."
        )

    use_test = float(test_ratio) > 0 and any(count >= 9 for count in node_counts)
    split_names = (
        ("train", "validation", "test")
        if use_test
        else (
            "train",
            "validation",
        )
    )

    for attempt in range(100):
        rng = np.random.default_rng(int(random_seed) + attempt)
        assignments: list[np.ndarray] = []
        node_counts_by_split = {name: 0 for name in split_names}
        for node_count in node_counts:
            assignment = np.full(node_count, "train", dtype=object)
            if node_count >= 6:
                permutation = rng.permutation(node_count)
                if use_test and node_count >= 9:
                    n_test = max(3, int(round(node_count * float(test_ratio))))
                    n_validation = max(
                        3, int(round(node_count * float(validation_ratio)))
                    )
                    if n_test + n_validation > node_count - 3:
                        excess = n_test + n_validation - (node_count - 3)
                        reduce_test = min(excess, max(0, n_test - 3))
                        n_test -= reduce_test
                        excess -= reduce_test
                        n_validation -= min(excess, max(0, n_validation - 3))
                    assignment[permutation[:n_test]] = "test"
                    assignment[
                        permutation[n_test : n_test + n_validation]
                    ] = "validation"
                else:
                    n_validation = max(3, int(round(node_count * 0.5)))
                    n_validation = min(n_validation, node_count - 3)
                    assignment[permutation[:n_validation]] = "validation"
            assignments.append(assignment)
            for name in split_names:
                node_counts_by_split[name] += int(np.sum(assignment == name))

        masks: dict[str, np.ndarray] = {}
        for name in split_names:
            mask = np.zeros(edges.shape[0], dtype=bool)
            for time_idx, assignment in enumerate(assignments):
                at_time = time_indices == time_idx
                if not np.any(at_time):
                    continue
                local_edges = edges[at_time]
                mask[at_time] = (assignment[local_edges[:, 0]] == name) & (
                    assignment[local_edges[:, 1]] == name
                )
            masks[name] = mask

        if all(
            np.any(masks[name]) and np.unique(labels[masks[name]]).size == 2
            for name in split_names
        ):
            split_indices = {
                name: np.flatnonzero(masks[name]).astype(np.int64)
                for name in split_names
            }
            if not use_test:
                split_indices["test"] = np.empty(0, dtype=np.int64)
                node_counts_by_split["test"] = 0
            kept = sum(indices.size for indices in split_indices.values())
            metadata = {
                "strategy": "node_disjoint_holdout",
                "test_available": bool(split_indices["test"].size),
                "node_counts": {
                    name: int(node_counts_by_split[name])
                    for name in ("train", "validation", "test")
                },
                "cross_split_edges_excluded": int(edges.shape[0] - kept),
            }
            return split_indices, metadata

    raise ValueError(
        "Could not form leakage-free train/validation partitions containing "
        "both positive and negative edges. Add more cells/edges or provide more "
        "time slices."
    )


def _build_leakage_free_split(
    edges: np.ndarray,
    labels: np.ndarray,
    time_indices: np.ndarray,
    node_counts: list[int],
    *,
    split_strategy: str,
    random_seed: int,
    validation_ratio: float,
    test_ratio: float,
) -> tuple[dict[str, np.ndarray], dict]:
    if split_strategy not in {"group_or_node", "time_group", "node_disjoint"}:
        raise ValueError(
            "split_strategy must be 'group_or_node', 'time_group', or "
            "'node_disjoint'."
        )

    if split_strategy in {"group_or_node", "time_group"}:
        grouped = _indices_for_time_groups(
            time_indices,
            labels,
            random_seed=random_seed,
            validation_ratio=validation_ratio,
            test_ratio=test_ratio,
        )
        if grouped is not None:
            return grouped
        if split_strategy == "time_group":
            raise ValueError(
                "time_group split requires at least two time slices, each with "
                "both positive and negative edges."
            )

    return _node_disjoint_indices(
        edges,
        labels,
        time_indices,
        node_counts,
        random_seed=random_seed,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
    )


def train_edge_predictor(
    data_name: str,
    adata_or_h5ad=None,
    feature_csv_path: str | None = None,
    graph_input_dir: str = "input_graph/",
    output_model_path: str = "edge_classifier/model.pt",
    epochs: int = 100,
    batch_size: int = 1024,
    learning_rate: float = 0.001,
    spatial_dim: int = 2,
    distance_threshold: float | None = None,
    device: str = "cuda",
    time_key: str = "time_point_processed",
    latent_key: str = "X_latent",
    spatial_key: str = "spatial_aligned",
    train_sample_ratio_per_epoch: float = 1.0,
    max_train_edges_per_epoch: int | None = None,
    num_workers: int = 4,
    pin_memory: bool | None = None,
    random_seed: int = 42,
    edge_predictor_threshold: float | None = None,
    split_strategy: str = "node_disjoint",
    validation_ratio: float = 0.1,
    test_ratio: float = 0.2,
) -> dict:
    """
    Train an edge predictor MLP using interaction graph + node features.

    Preferred input is AnnData/h5ad (`adata_or_h5ad`) so feature construction is
    consistent with preprocessing/alignment outputs.
    CSV remains as backward-compatible fallback.

    Parameters
    ----------
    train_sample_ratio_per_epoch
        Fraction of training edges randomly sampled each epoch. Default 1.0
        means full training set every epoch.
    max_train_edges_per_epoch
        Optional hard cap on number of training edges per epoch. If set,
        effective epoch training size = min(ratio-based size, this cap).
    num_workers
        Number of DataLoader worker processes.
    pin_memory
        Whether to pin CPU memory in DataLoader. If None, auto-enable on CUDA.
    random_seed
        Seed for negative sampling, train-epoch sampling, model initialization,
        and DataLoader shuffling.
    edge_predictor_threshold
        Optional fixed decision threshold for a published or externally
        calibrated analysis. The validation-optimal threshold is still
        calculated and recorded for comparison.
    split_strategy
        ``"node_disjoint"`` (default) uses every observed time slice while
        keeping cell nodes disjoint across train, validation, and test. This is
        the production split for the dataset-specific edge prior. Explicit
        ``"time_group"`` remains available as a random observed-time holdout
        sensitivity, not a future-time or temporal-generalization experiment;
        ``"group_or_node"`` requests that holdout when possible and otherwise
        falls back to node-disjoint partitions. No cell node is shared between
        training and reported validation/test.
    """
    np.random.seed(int(random_seed))
    torch.manual_seed(int(random_seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(random_seed))
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    if pin_memory is None:
        pin_memory = device.type == "cuda"
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    if edge_predictor_threshold is not None:
        edge_predictor_threshold = float(edge_predictor_threshold)
        if (
            not np.isfinite(edge_predictor_threshold)
            or not 0 < edge_predictor_threshold < 1
        ):
            raise ValueError(
                "edge_predictor_threshold must lie strictly between 0 and 1 or be None, "
                f"got {edge_predictor_threshold}."
            )
    if not 0 < float(validation_ratio) < 1:
        raise ValueError("validation_ratio must lie strictly between 0 and 1.")
    if not 0 <= float(test_ratio) < 1:
        raise ValueError("test_ratio must lie in [0, 1).")
    if float(validation_ratio) + float(test_ratio) >= 1:
        raise ValueError("validation_ratio + test_ratio must be less than 1.")
    print(f"--- Using device: {device} ---")

    # 1. Load Data
    if adata_or_h5ad is not None:
        print("Loading features from AnnData/h5ad")
        time_points, node_features_by_time = _load_features_from_adata(
            adata_or_h5ad,
            time_key=time_key,
            latent_key=latent_key,
            spatial_key=spatial_key,
        )
    elif feature_csv_path is not None:
        print(f"Loading features from CSV: {feature_csv_path}")
        time_points, node_features_by_time = _load_features_from_csv(feature_csv_path)
    else:
        raise ValueError("Either adata_or_h5ad or feature_csv_path must be provided.")

    distance_threshold = _resolve_distance_threshold(distance_threshold, adata_or_h5ad)
    print(f"Using distance_threshold={distance_threshold:.6f}")

    all_node_features = []
    all_positive_edges = []
    all_time_indices = []
    loaded_time_values = []
    positive_edge_counts_by_time = []

    print("Loading graphs and preparing features...")
    for time_idx, t_val in enumerate(time_points):
        data_i = node_features_by_time[float(t_val)]
        num_nodes_features = int(data_i.shape[0])
        candidate_paths = _resolve_graph_path_candidates(
            data_name=data_name,
            graph_input_dir=graph_input_dir,
            time_idx=time_idx,
            t_val=float(t_val),
        )
        try:
            edges, adj_path = _load_adjacency_with_compatibility(
                candidate_paths=candidate_paths,
                num_nodes_features=num_nodes_features,
            )
        except ValueError as exc:
            raise ValueError(
                "Every observed nonempty time slice needs a matching interaction "
                "graph before edge-predictor training. "
                f"time_idx={time_idx}, time_value={float(t_val):g}, "
                f"n_nodes={num_nodes_features}. {exc}"
            ) from exc

        edges, edge_counts = _unique_directed_edges(edges)
        _validate_positive_spatial_contract(
            edges,
            data_i[:, :spatial_dim].numpy(),
            distance_threshold,
            time_value=float(t_val),
        )
        local_time_idx = len(all_node_features)
        all_node_features.append(data_i)
        all_positive_edges.append(edges)
        loaded_time_values.append(float(t_val))
        positive_edge_counts_by_time.append(
            {
                "time_value": float(t_val),
                **edge_counts,
            }
        )
        all_time_indices.extend([local_time_idx] * len(edges))
        print(
            f"  loaded slice time_idx={time_idx}, local_idx={local_time_idx}, "
            f"time_value={t_val}, n_nodes={num_nodes_features}, "
            f"positive_edges={edge_counts['unique']} unique "
            f"({edge_counts['duplicates_removed']} LR duplicates removed) "
            f"from {adj_path}"
        )

    if not all_node_features:
        raise ValueError("No data loaded. Check input paths and graph files.")

    num_features = all_node_features[0].shape[1]

    # 2. Complete candidate universe and leakage-free partitions. Training is
    # balanced later, but validation/test retain their natural positive rate so
    # the selected decision threshold matches deployment over all radius pairs.
    print("\n--- Preparing complete spatial candidate universe ---")
    positive_edges = np.vstack(all_positive_edges)
    positive_labels = np.ones(len(positive_edges), dtype=np.float32)

    all_negative_edges = []
    all_negative_time_indices = []

    for time_idx in range(len(all_node_features)):
        spatial_coords = all_node_features[time_idx][:, :spatial_dim].numpy()

        negative_edges = vectorized_negative_sampling(
            all_positive_edges[time_idx],
            all_node_features[time_idx].shape[0],
            None,
            spatial_coords,
            distance_threshold=distance_threshold,
        )
        all_negative_edges.append(negative_edges)
        all_negative_time_indices.extend([time_idx] * len(negative_edges))

    negative_edges = np.vstack(all_negative_edges)
    negative_labels = np.zeros(len(negative_edges), dtype=np.float32)

    all_edges = np.vstack([positive_edges, negative_edges])
    all_labels = np.hstack([positive_labels, negative_labels])
    full_time_indices = np.array(
        all_time_indices + all_negative_time_indices
    )  # local variable renamed to avoid conflict

    split_indices, split_metadata = _build_leakage_free_split(
        all_edges,
        all_labels,
        full_time_indices,
        [int(features.shape[0]) for features in all_node_features],
        split_strategy=split_strategy,
        random_seed=int(random_seed),
        validation_ratio=float(validation_ratio),
        test_ratio=float(test_ratio),
    )
    train_idx = split_indices["train"]
    val_idx = split_indices["validation"]
    test_idx = split_indices["test"]
    split_metadata["edge_counts"] = {
        "train": int(train_idx.size),
        "validation": int(val_idx.size),
        "test": int(test_idx.size),
    }
    if split_metadata["strategy"] == "time_group_holdout":
        for name in ("train", "validation", "test"):
            split_metadata[f"{name}_time_values"] = [
                float(loaded_time_values[local_index])
                for local_index in split_metadata[f"{name}_time_indices"]
            ]
        split_metadata["excluded_time_values_without_both_classes"] = [
            float(loaded_time_values[local_index])
            for local_index in split_metadata[
                "excluded_time_indices_without_both_classes"
            ]
        ]
    print(
        "Leakage-free split: "
        f"{split_metadata['strategy']} | "
        f"train={train_idx.size}, validation={val_idx.size}, test={test_idx.size}"
    )

    train_pool_idx = _balanced_training_pool(
        train_idx,
        all_labels,
        random_seed=int(random_seed),
    )
    # 3. Model & Training
    print("\n--- Training Model ---")
    model = LinkPredictorMLP(input_dim=num_features * 2).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_ap = -float("inf")
    best_val_auc = -float("inf")
    patience = 5
    epochs_no_improve = 0
    model_dir = os.path.dirname(output_model_path)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)

    best_threshold = 0.5
    best_val_f1 = 0.0
    best_val_acc = 0.0

    epoch_iter = tqdm(range(epochs), desc="Edge predictor training", dynamic_ncols=True)
    for epoch in epoch_iter:
        epoch_train_idx = _sample_train_indices_for_epoch(
            train_pool_idx,
            train_sample_ratio_per_epoch=train_sample_ratio_per_epoch,
            max_train_edges_per_epoch=max_train_edges_per_epoch,
        )
        train_batches = _iter_edge_batches(
            all_edges[epoch_train_idx],
            all_labels[epoch_train_idx],
            full_time_indices[epoch_train_idx],
            all_node_features,
            batch_size=batch_size,
            shuffle=True,
            random_seed=int(random_seed) + epoch,
        )

        model.train()
        total_loss = 0
        train_batch_count = 0
        for features, labels in train_batches:
            features = features.to(device)
            labels = labels.to(device).unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            train_batch_count += 1

        # Validation
        model.eval()
        val_preds, val_labels_flat = _predict_edge_probabilities(
            model,
            all_edges[val_idx],
            all_labels[val_idx],
            full_time_indices[val_idx],
            all_node_features,
            batch_size=batch_size,
            device=device,
        )
        val_best_threshold, val_best_f1, val_best_acc = _find_best_threshold(
            val_labels_flat, val_preds
        )

        val_ap = float(average_precision_score(val_labels_flat, val_preds))

        try:
            val_auc = roc_auc_score(val_labels_flat, val_preds)
        except ValueError:
            val_auc = 0.5

        avg_train_loss = total_loss / max(1, train_batch_count)
        epoch_iter.set_postfix(
            loss=f"{avg_train_loss:.4f}",
            n_train=f"{len(epoch_train_idx)}",
            val_ap=f"{val_ap:.4f}",
            val_auc=f"{val_auc:.4f}",
            val_f1=f"{val_best_f1:.4f}",
            th=f"{val_best_threshold:.2f}",
        )
        print(
            f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, "
            f"TrainEdges={len(epoch_train_idx)}, "
            f"Val AP={val_ap:.4f}, Val AUC={val_auc:.4f}, "
            f"Val F1@best_th={val_best_f1:.4f}, "
            f"Val Acc@best_th={val_best_acc:.4f}, BestTh={val_best_threshold:.2f}"
        )

        if val_ap > best_val_ap + 1e-4:
            best_val_ap = val_ap
            best_val_auc = val_auc
            epochs_no_improve = 0
            torch.save(model.state_dict(), output_model_path)
            best_threshold = float(val_best_threshold)
            best_val_f1 = float(val_best_f1)
            best_val_acc = float(val_best_acc)
            print(f"  New best model saved to {output_model_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break

    model.load_state_dict(torch.load(output_model_path, map_location=device))
    model.eval()
    val_preds, val_labels_flat = _predict_edge_probabilities(
        model,
        all_edges[val_idx],
        all_labels[val_idx],
        full_time_indices[val_idx],
        all_node_features,
        batch_size=batch_size,
        device=device,
    )
    validation_metrics = _binary_metrics(
        val_labels_flat, val_preds, float(best_threshold)
    )
    validation_metrics_by_time = _metrics_by_time(
        val_labels_flat,
        val_preds,
        full_time_indices[val_idx],
        loaded_time_values,
        float(best_threshold),
    )
    test_metrics = None
    test_metrics_by_time = []
    if test_idx.size:
        test_probs, test_labels = _predict_edge_probabilities(
            model,
            all_edges[test_idx],
            all_labels[test_idx],
            full_time_indices[test_idx],
            all_node_features,
            batch_size=batch_size,
            device=device,
        )
        test_metrics = _binary_metrics(test_labels, test_probs, float(best_threshold))
        test_metrics_by_time = _metrics_by_time(
            test_labels,
            test_probs,
            full_time_indices[test_idx],
            loaded_time_values,
            float(best_threshold),
        )

    selected_threshold = float(best_threshold)
    effective_threshold = (
        selected_threshold
        if edge_predictor_threshold is None
        else float(edge_predictor_threshold)
    )
    meta = {
        "edge_predictor_threshold": effective_threshold,
        "edge_predictor_threshold_selected": selected_threshold,
        "selection_source": (
            "validation" if edge_predictor_threshold is None else "user_override"
        ),
        "checkpoint_selection_metric": "average_precision",
        "threshold_selection_metric": "f1",
        "best_val_average_precision": float(best_val_ap),
        "best_val_auc": float(best_val_auc),
        "best_val_f1": float(best_val_f1),
        "best_val_acc": float(best_val_acc),
        "validation_metrics_at_selected_threshold": validation_metrics,
        "validation_metrics_by_time": validation_metrics_by_time,
        "test_metrics_at_validation_threshold": test_metrics,
        "test_metrics_by_time": test_metrics_by_time,
        "distance_threshold": float(distance_threshold),
        "time_key": time_key,
        "latent_key": latent_key,
        "spatial_key": spatial_key,
        "train_sample_ratio_per_epoch": float(train_sample_ratio_per_epoch),
        "max_train_edges_per_epoch": (
            int(max_train_edges_per_epoch)
            if max_train_edges_per_epoch is not None
            else None
        ),
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
        "random_seed": int(random_seed),
        "candidate_universe": {
            "definition": "all directed pairs with 1e-6 < distance < cutoff",
            "positive_edges": int(positive_edges.shape[0]),
            "negative_edges": int(negative_edges.shape[0]),
            "positive_fraction": float(
                positive_edges.shape[0] / max(1, all_edges.shape[0])
            ),
            "validation_positive_fraction": float(np.mean(all_labels[val_idx])),
            "test_positive_fraction": (
                None if not test_idx.size else float(np.mean(all_labels[test_idx]))
            ),
            "training_balanced_edges": int(train_pool_idx.size),
        },
        "positive_edge_deduplication": {
            "unit": "directed_cell_pair_per_time_slice",
            "raw": int(sum(item["raw"] for item in positive_edge_counts_by_time)),
            "unique": int(sum(item["unique"] for item in positive_edge_counts_by_time)),
            "duplicates_removed": int(
                sum(item["duplicates_removed"] for item in positive_edge_counts_by_time)
            ),
            "by_time": positive_edge_counts_by_time,
        },
        "split": {
            **split_metadata,
            "requested_strategy": split_strategy,
            "validation_ratio": float(validation_ratio),
            "test_ratio": float(test_ratio),
            "time_values_by_local_index": [
                float(value) for value in loaded_time_values
            ],
        },
    }
    meta_path = f"{output_model_path}.meta.json"
    with open(meta_path, "w", encoding="utf-8") as fp:
        json.dump(meta, fp, ensure_ascii=False, indent=2)

    if adata_or_h5ad is not None and not isinstance(adata_or_h5ad, str):
        adata_or_h5ad.uns.setdefault("interaction_graph", {})
        adata_or_h5ad.uns["interaction_graph"][
            "edge_predictor_threshold"
        ] = effective_threshold
        adata_or_h5ad.uns["interaction_graph"][
            "edge_predictor_threshold_selected"
        ] = selected_threshold
        adata_or_h5ad.uns["interaction_graph"]["edge_predictor_path"] = os.path.abspath(
            output_model_path
        )
        adata_or_h5ad.uns["interaction_graph"][
            "edge_predictor_model_path"
        ] = os.path.abspath(output_model_path)
        adata_or_h5ad.uns["interaction_graph"][
            "edge_predictor_meta_path"
        ] = os.path.abspath(meta_path)
        adata_or_h5ad.uns["interaction_graph"]["edge_predictor_best_val_auc"] = float(
            best_val_auc
        )
        adata_or_h5ad.uns["interaction_graph"]["edge_predictor_best_val_f1"] = float(
            best_val_f1
        )
        adata_or_h5ad.uns["interaction_graph"][
            "edge_predictor_split_strategy"
        ] = split_metadata["strategy"]

    print(
        f"Training complete. Best Val AP={best_val_ap:.4f}, "
        f"AUC={best_val_auc:.4f}, "
        f"validation threshold={selected_threshold:.2f}, "
        f"effective threshold={effective_threshold:.2f} (saved: {meta_path})"
    )
    return {
        "model_path": output_model_path,
        "meta_path": meta_path,
        "edge_predictor_threshold": effective_threshold,
        "edge_predictor_threshold_selected": selected_threshold,
        "best_val_average_precision": float(best_val_ap),
        "best_val_auc": float(best_val_auc),
        "best_val_f1": float(best_val_f1),
        "best_val_acc": float(best_val_acc),
        "validation_metrics_at_selected_threshold": validation_metrics,
        "test_metrics_at_validation_threshold": test_metrics,
        "positive_edge_deduplication": meta["positive_edge_deduplication"],
        "split": meta["split"],
    }
