import argparse
import gzip
import os
import pickle
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def _vectorized_negative_sampling(positive_edges, num_nodes, num_neg_samples, spatial_coords, distance_threshold):
    pos_edge_ids = set(positive_edges[:, 0] * num_nodes + positive_edges[:, 1])
    num_candidates = int(num_neg_samples * 5000)
    candidate_u = np.random.randint(0, num_nodes, size=num_candidates)
    candidate_v = np.random.randint(0, num_nodes, size=num_candidates)
    non_self_loop_mask = candidate_u != candidate_v
    candidate_u = candidate_u[non_self_loop_mask]
    candidate_v = candidate_v[non_self_loop_mask]
    u_coords = spatial_coords[candidate_u]
    v_coords = spatial_coords[candidate_v]
    spatial_distances = np.sqrt(np.sum((u_coords - v_coords) ** 2, axis=1))
    distance_mask = spatial_distances < distance_threshold
    candidate_u = candidate_u[distance_mask]
    candidate_v = candidate_v[distance_mask]
    candidate_ids = candidate_u * num_nodes + candidate_v
    is_negative_mask = ~np.isin(candidate_ids, list(pos_edge_ids))
    neg_u = candidate_u[is_negative_mask][:num_neg_samples]
    neg_v = candidate_v[is_negative_mask][:num_neg_samples]
    return np.stack([neg_u, neg_v], axis=1)


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


class LinkPredictorMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        return self.network(x)


def train_edge_predictor(
    data_csv: str,
    input_graph_dir: str,
    output_path: str,
    data_name_prefix: str,
    time_indices: Sequence[int],
    samples_col: str = "samples",
    spatial_feature_cols: Sequence[int] = (0, 1),
    distance_threshold: float = 0.12,
    learning_rate: float = 0.001,
    epochs: int = 100,
    batch_size: int = 1024,
    test_ratio: float = 0.2,
    validation_ratio: float = 0.1,
) -> None:
    df = pd.read_csv(data_csv)
    time_points = sorted(df[samples_col].unique())

    all_node_features = {}
    all_positive_edges = {}
    all_time_indices = []

    for time_idx in time_indices:
        data_i = df[df[samples_col] == time_points[time_idx]].iloc[:, 1:].values
        data_name = f"{data_name_prefix}_t{time_idx}"
        graph_path = os.path.join(input_graph_dir, data_name, f"{data_name}_adjacency_records")
        with gzip.open(graph_path, "rb") as f:
            adjacency_records = pickle.load(f)
        all_node_features[time_idx] = torch.tensor(data_i).float()
        all_positive_edges[time_idx] = np.array(adjacency_records[0])
        all_time_indices.extend([time_idx] * len(adjacency_records[0]))

    last_key = time_indices[-1]
    num_nodes = all_node_features[last_key].shape[0]
    num_features = all_node_features[last_key].shape[1]

    positive_edges = np.vstack([all_positive_edges[idx] for idx in time_indices])
    positive_labels = np.ones(len(positive_edges))

    all_negative_edges = []
    all_negative_time_indices = []
    for time_idx in time_indices:
        num_pos_edges = len(all_positive_edges[time_idx])
        spatial_coords = all_node_features[time_idx][:, spatial_feature_cols].numpy()
        negative_edges = _vectorized_negative_sampling(
            all_positive_edges[time_idx],
            all_node_features[time_idx].shape[0],
            num_pos_edges,
            spatial_coords,
            distance_threshold,
        )
        all_negative_edges.append(negative_edges)
        all_negative_time_indices.extend([time_idx] * len(negative_edges))

    negative_edges = np.vstack(all_negative_edges)
    negative_labels = np.zeros(len(negative_edges))

    all_edges = np.vstack([positive_edges, negative_edges])
    all_labels = np.hstack([positive_labels, negative_labels])
    all_time_indices = np.array(all_time_indices + all_negative_time_indices)

    train_idx, test_idx = train_test_split(
        np.arange(len(all_edges)),
        test_size=test_ratio,
        random_state=42,
        stratify=all_labels,
    )
    val_split_ratio = validation_ratio / (1 - test_ratio)
    train_idx, val_idx = train_test_split(
        train_idx,
        test_size=val_split_ratio,
        random_state=42,
        stratify=all_labels[train_idx],
    )

    train_dataset = LinkPredictionDataset(all_edges[train_idx], all_labels[train_idx], all_node_features, all_time_indices[train_idx])
    val_dataset = LinkPredictionDataset(all_edges[val_idx], all_labels[val_idx], all_node_features, all_time_indices[val_idx])
    test_dataset = LinkPredictionDataset(all_edges[test_idx], all_labels[test_idx], all_node_features, all_time_indices[test_idx])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LinkPredictorMLP(input_dim=num_features * 2).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    patience = 5
    min_delta = 1e-4
    best_metric = -float("inf")
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        for features, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [train]"):
            features = features.to(device)
            labels = labels.to(device).unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)

        model.eval()
        all_val_preds = []
        with torch.no_grad():
            for features, labels in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} [val]"):
                features = features.to(device)
                outputs = model(features)
                preds = torch.sigmoid(outputs)
                all_val_preds.append(preds.cpu())

        val_preds = torch.cat(all_val_preds).numpy().ravel()
        val_labels = val_dataset.labels
        try:
            val_auc = roc_auc_score(val_labels, val_preds)
        except ValueError:
            val_auc = float("nan")
        val_preds_binary = (val_preds >= 0.5).astype(int)
        val_acc = accuracy_score(val_labels, val_preds_binary)
        print(
            f"Epoch {epoch + 1}/{epochs} | Train Loss: {avg_train_loss:.4f} | "
            f"Val AUC: {val_auc:.4f} | Val Acc: {val_acc:.4f}"
        )

        metric = val_auc
        if metric > best_metric + min_delta:
            best_metric = metric
            epochs_no_improve = 0
            torch.save(model.state_dict(), output_path)
            print(f"Saved new best model (Val AUC={val_auc:.4f}) -> {output_path}")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve}/{patience} epochs (best={best_metric:.4f})")
        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break

    if os.path.exists(output_path):
        model.load_state_dict(torch.load(output_path, map_location=device))
        model.eval()

    all_test_preds = []
    all_test_labels = []
    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            outputs = model(features)
            preds = torch.sigmoid(outputs).detach().cpu().numpy().ravel()
            all_test_preds.append(preds)
            all_test_labels.append(labels.numpy().ravel())
    all_test_preds = np.concatenate(all_test_preds)
    all_test_labels = np.concatenate(all_test_labels)
    try:
        final_auc = roc_auc_score(all_test_labels, all_test_preds)
    except ValueError:
        final_auc = float("nan")
    print(f"Test ROC-AUC: {final_auc:.4f}")


def _parse_indices(value: str) -> Iterable[int]:
    return [int(x) for x in value.split(",") if x.strip() != ""]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_csv", required=True)
    parser.add_argument("--input_graph_dir", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--data_name_prefix", required=True)
    parser.add_argument("--time_indices", required=True, help="Comma-separated time indices, e.g. 0,1,2,3")
    parser.add_argument("--samples_col", default="samples")
    parser.add_argument("--distance_threshold", type=float, default=0.12)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--validation_ratio", type=float, default=0.1)
    args = parser.parse_args()
    train_edge_predictor(
        data_csv=args.data_csv,
        input_graph_dir=args.input_graph_dir,
        output_path=args.output_path,
        data_name_prefix=args.data_name_prefix,
        time_indices=_parse_indices(args.time_indices),
        samples_col=args.samples_col,
        distance_threshold=args.distance_threshold,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        batch_size=args.batch_size,
        test_ratio=args.test_ratio,
        validation_ratio=args.validation_ratio,
    )


if __name__ == "__main__":
    main()
