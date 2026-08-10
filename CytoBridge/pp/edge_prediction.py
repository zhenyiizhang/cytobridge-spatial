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
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
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
    num_neg_samples,
    spatial_coords,
    distance_threshold: float,
):
    """
    Efficient negative sampling via spatial candidate pool:
    build radius-neighbor directed candidates with cKDTree, then remove positives.
    """
    positive_edges = np.asarray(positive_edges, dtype=np.int64)
    target = int(num_neg_samples)
    if target <= 0 or int(num_nodes) <= 1:
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

    sample_n = min(target, int(neg_u.size))
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
    if thresholds is None:
        thresholds = np.arange(0.01, 0.99, 0.01, dtype=np.float32)
    labels = np.asarray(labels).reshape(-1)
    probs = np.asarray(probs).reshape(-1)
    if labels.shape[0] == 0:
        return 0.5, 0.0, 0.0

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
            print(
                "Warning: skip time slice due to graph-feature mismatch. "
                f"time_idx={time_idx}, time_value={t_val}\n{exc}"
            )
            continue

        local_time_idx = len(all_node_features)
        all_node_features.append(data_i)
        all_positive_edges.append(edges)
        all_time_indices.extend([local_time_idx] * len(edges))
        print(
            f"  loaded slice time_idx={time_idx}, local_idx={local_time_idx}, "
            f"time_value={t_val}, n_nodes={num_nodes_features}, n_edges={len(edges)} "
            f"from {adj_path}"
        )

    if not all_node_features:
        raise ValueError("No data loaded. Check input paths and graph files.")

    num_features = all_node_features[0].shape[1]

    # 2. Negative Sampling & Dataset Construction
    print("\n--- Preparing Dataset (Negative Sampling) ---")
    positive_edges = np.vstack(all_positive_edges)
    positive_labels = np.ones(len(positive_edges), dtype=np.float32)

    all_negative_edges = []
    all_negative_time_indices = []

    for time_idx in range(len(all_node_features)):
        num_pos_edges = len(all_positive_edges[time_idx])
        spatial_coords = all_node_features[time_idx][:, :spatial_dim].numpy()

        negative_edges = vectorized_negative_sampling(
            all_positive_edges[time_idx],
            all_node_features[time_idx].shape[0],
            num_pos_edges,
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

    # Split Data
    train_idx, test_idx = train_test_split(
        np.arange(len(all_edges)), test_size=0.2, random_state=42, stratify=all_labels
    )
    train_idx, val_idx = train_test_split(
        train_idx,
        test_size=0.125,
        random_state=42,
        stratify=all_labels[train_idx],  # 0.1 / (1 - 0.2) = 0.125
    )

    train_pool_idx = np.asarray(train_idx, dtype=np.int64)
    val_edges = all_edges[val_idx]
    val_time_idx = full_time_indices[val_idx]
    val_features = _build_combined_edge_features(
        val_edges, val_time_idx, all_node_features
    )
    val_labels_tensor = torch.from_numpy(
        all_labels[val_idx].astype(np.float32, copy=False)
    )
    val_dataset = TensorDataset(val_features, val_labels_tensor)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(0, int(num_workers)),
        pin_memory=bool(pin_memory),
        persistent_workers=bool(num_workers and int(num_workers) > 0),
    )

    # 3. Model & Training
    print("\n--- Training Model ---")
    model = LinkPredictorMLP(input_dim=num_features * 2).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

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
        train_edges = all_edges[epoch_train_idx]
        train_time_idx = full_time_indices[epoch_train_idx]
        train_features = _build_combined_edge_features(
            train_edges, train_time_idx, all_node_features
        )
        train_labels_tensor = torch.from_numpy(
            all_labels[epoch_train_idx].astype(np.float32, copy=False)
        )
        train_dataset = TensorDataset(train_features, train_labels_tensor)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=max(0, int(num_workers)),
            pin_memory=bool(pin_memory),
            persistent_workers=False,
        )

        model.train()
        total_loss = 0
        for features, labels in tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{epochs} [train]", leave=False
        ):
            features = features.to(device)
            labels = labels.to(device).unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        all_preds = []
        all_labels_val = []
        with torch.no_grad():
            for features, labels in tqdm(
                val_loader, desc=f"Epoch {epoch+1}/{epochs} [val]", leave=False
            ):
                features = features.to(device)
                outputs = model(features)
                preds = torch.sigmoid(outputs).cpu()
                all_preds.append(preds)
                all_labels_val.append(labels)

        val_preds = torch.cat(all_preds).numpy()
        val_labels_flat = torch.cat(all_labels_val).numpy()
        val_best_threshold, val_best_f1, val_best_acc = _find_best_threshold(
            val_labels_flat, val_preds
        )

        try:
            val_auc = roc_auc_score(val_labels_flat, val_preds)
        except ValueError:
            val_auc = 0.5

        avg_train_loss = total_loss / max(1, len(train_loader))
        epoch_iter.set_postfix(
            loss=f"{avg_train_loss:.4f}",
            n_train=f"{len(epoch_train_idx)}",
            val_auc=f"{val_auc:.4f}",
            val_f1=f"{val_best_f1:.4f}",
            th=f"{val_best_threshold:.2f}",
        )
        print(
            f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, "
            f"TrainEdges={len(epoch_train_idx)}, "
            f"Val AUC={val_auc:.4f}, Val F1@best_th={val_best_f1:.4f}, "
            f"Val Acc@best_th={val_best_acc:.4f}, BestTh={val_best_threshold:.2f}"
        )

        if val_auc > best_val_auc + 1e-4:
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
        "selection_metric": "f1",
        "best_val_auc": float(best_val_auc),
        "best_val_f1": float(best_val_f1),
        "best_val_acc": float(best_val_acc),
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

    print(
        f"Training complete. Best Val AUC={best_val_auc:.4f}, "
        f"validation threshold={selected_threshold:.2f}, "
        f"effective threshold={effective_threshold:.2f} (saved: {meta_path})"
    )
    return {
        "model_path": output_model_path,
        "meta_path": meta_path,
        "edge_predictor_threshold": effective_threshold,
        "edge_predictor_threshold_selected": selected_threshold,
        "best_val_auc": float(best_val_auc),
        "best_val_f1": float(best_val_f1),
        "best_val_acc": float(best_val_acc),
    }
