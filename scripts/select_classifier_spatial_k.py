#!/usr/bin/env python3
"""Select spatial label-smoothing k on observed held-out cells.

This trains only the downstream cell-type classifier. CytoBridge dynamics and
generated trajectories are never used to choose k.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import anndata as ad
import numpy as np
import pandas as pd
import torch
from scipy.stats import mode
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from CytoBridge.tl.downstream.classification import (
    ResidualMLP,
    _stable_spatial_neighbors,
)
from CytoBridge.tl.downstream.downstream_data import infer_time_key, parse_time_value


K_VALUES = (1, 5, 10, 20, 50)
SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--h5ad", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--annotation-key", default="Annotation")
    parser.add_argument("--time-key")
    parser.add_argument("--latent-key", default="X_latent")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def observed_arrays(
    adata: ad.AnnData,
    annotation_key: str,
    requested_time_key: str | None,
    latent_key: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    time_key = infer_time_key(adata.obs, preferred=requested_time_key)
    times = np.asarray(
        [parse_time_value(value) for value in adata.obs[time_key]],
        dtype=np.float32,
    )
    spatial = np.asarray(adata.obsm["spatial_aligned"], dtype=np.float32)
    latent = np.asarray(adata.obsm[latent_key], dtype=np.float32)
    states = np.column_stack((spatial, latent)).astype(np.float32)
    features = np.column_stack((times, states)).astype(np.float32)
    labels = adata.obs[annotation_key].astype(str).to_numpy()
    return features, states, spatial, labels, time_key


def fixed_split(
    labels: np.ndarray,
) -> tuple[LabelEncoder, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    encoder = LabelEncoder()
    encoded = encoder.fit_transform(labels)
    indices = np.arange(len(labels))
    counts = np.bincount(encoded, minlength=len(encoder.classes_))
    rare_class_indices = np.flatnonzero(counts < 2)
    train_only = indices[np.isin(encoded, rare_class_indices)]
    splittable = indices[~np.isin(encoded, rare_class_indices)]
    train, heldout = train_test_split(
        splittable,
        test_size=0.1,
        random_state=SEED,
        stratify=encoded[splittable],
    )
    train = np.concatenate((np.asarray(train), train_only))
    train_only_classes = encoder.classes_[rare_class_indices].astype(str).tolist()
    return encoder, encoded, train, np.asarray(heldout), train_only_classes


def train_classifier(
    features: np.ndarray,
    encoded_labels: np.ndarray,
    train_indices: np.ndarray,
    heldout_indices: np.ndarray,
    n_classes: int,
    device: torch.device,
) -> tuple[ResidualMLP, dict]:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)

    x_train = torch.as_tensor(features[train_indices], device=device)
    y_train = torch.as_tensor(encoded_labels[train_indices], device=device)
    x_heldout = torch.as_tensor(features[heldout_indices], device=device)
    y_heldout = encoded_labels[heldout_indices]

    model = ResidualMLP(features.shape[1], hidden_size=128, num_classes=n_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=500, eta_min=1e-5
    )
    loss_function = torch.nn.CrossEntropyLoss()

    best_epoch = 0
    best_balanced_accuracy = -np.inf
    best_weights = copy.deepcopy(model.state_dict())
    for epoch in range(1, 501):
        model.train()
        optimizer.zero_grad()
        loss = loss_function(model(x_train), y_train)
        loss.backward()
        optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            prediction = model(x_heldout).argmax(dim=1).cpu().numpy()
        score = balanced_accuracy_score(y_heldout, prediction)
        if score > best_balanced_accuracy:
            best_epoch = epoch
            best_balanced_accuracy = float(score)
            best_weights = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_weights)
    model.eval()
    return model, {
        "epochs_run": 500,
        "best_epoch": best_epoch,
        "best_heldout_balanced_accuracy_before_smoothing": best_balanced_accuracy,
        "n_train": int(len(train_indices)),
        "n_heldout": int(len(heldout_indices)),
    }


def predict_in_batches(
    model: ResidualMLP,
    features: np.ndarray,
    device: torch.device,
    batch_size: int = 65_536,
) -> np.ndarray:
    predictions = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            batch = torch.as_tensor(features[start : start + batch_size], device=device)
            predictions.append(model(batch).argmax(dim=1).cpu().numpy())
    return np.concatenate(predictions)


def predict_probabilities_in_batches(
    model: ResidualMLP,
    features: np.ndarray,
    device: torch.device,
    batch_size: int = 65_536,
) -> np.ndarray:
    probabilities = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            batch = torch.as_tensor(features[start : start + batch_size], device=device)
            probabilities.append(torch.softmax(model(batch), dim=1).cpu().numpy())
    return np.concatenate(probabilities)


def smooth_all_k(raw_encoded: np.ndarray, spatial: np.ndarray) -> dict[int, np.ndarray]:
    """Reuse one exact 50-neighbor query for all candidate vote sizes."""
    maximum_k = min(max(K_VALUES), len(raw_encoded))
    _, neighbors = _stable_spatial_neighbors(
        spatial,
        k=maximum_k,
        include_self=True,
    )
    neighbor_labels = raw_encoded[neighbors]
    results = {1: raw_encoded.copy()}
    for requested_k in K_VALUES[1:]:
        effective_k = min(requested_k, maximum_k)
        results[requested_k] = mode(
            neighbor_labels[:, :effective_k], axis=1, keepdims=False
        ).mode.astype(np.int64)
    return results


def observed_k_metrics(
    raw_predictions: np.ndarray,
    truth: np.ndarray,
    times: np.ndarray,
    spatial: np.ndarray,
    heldout_indices: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    heldout_mask = np.zeros(len(truth), dtype=bool)
    heldout_mask[heldout_indices] = True
    pooled_predictions = {k: np.empty(len(truth), dtype=np.int64) for k in K_VALUES}
    per_time_rows = []

    for time_value in np.unique(times):
        slice_indices = np.flatnonzero(times == time_value)
        smoothed = smooth_all_k(raw_predictions[slice_indices], spatial[slice_indices])
        slice_heldout = heldout_mask[slice_indices]
        slice_truth = truth[slice_indices][slice_heldout]
        for k, prediction in smoothed.items():
            pooled_predictions[k][slice_indices] = prediction
            evaluated = prediction[slice_heldout]
            per_time_rows.append(
                {
                    "time": float(time_value),
                    "k": k,
                    "n_heldout": int(len(slice_truth)),
                    "balanced_accuracy": balanced_accuracy_score(slice_truth, evaluated),
                    "macro_f1": f1_score(slice_truth, evaluated, average="macro"),
                    "accuracy": accuracy_score(slice_truth, evaluated),
                }
            )

    pooled_rows = []
    heldout_truth = truth[heldout_indices]
    for k in K_VALUES:
        evaluated = pooled_predictions[k][heldout_indices]
        pooled_rows.append(
            {
                "k": k,
                "n_heldout": int(len(heldout_indices)),
                "balanced_accuracy": balanced_accuracy_score(heldout_truth, evaluated),
                "macro_f1": f1_score(heldout_truth, evaluated, average="macro"),
                "accuracy": accuracy_score(heldout_truth, evaluated),
                "changed_from_raw_fraction": float(
                    np.mean(pooled_predictions[k] != raw_predictions)
                ),
            }
        )
    return pd.DataFrame(pooled_rows), pd.DataFrame(per_time_rows)


def choose_k(metrics: pd.DataFrame) -> tuple[int, str]:
    ordered = metrics.sort_values(
        ["balanced_accuracy", "macro_f1", "k"],
        ascending=[False, False, True],
        kind="mergesort",
    )
    return int(ordered.iloc[0]["k"]), (
        "highest held-out balanced accuracy; exact ties use macro-F1 then smaller k"
    )


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )

    adata = ad.read_h5ad(args.h5ad, backed="r")
    features, observed_states, spatial, labels, time_key = observed_arrays(
        adata, args.annotation_key, args.time_key, args.latent_key
    )
    times = features[:, 0]
    (
        encoder,
        encoded_labels,
        train_indices,
        heldout_indices,
        train_only_classes,
    ) = fixed_split(labels)
    model, training = train_classifier(
        features,
        encoded_labels,
        train_indices,
        heldout_indices,
        len(encoder.classes_),
        device,
    )
    raw_predictions = predict_in_batches(model, features, device)
    np.savez_compressed(args.output / "heldout_inputs.npz", train_indices=train_indices,
                        heldout_indices=heldout_indices, predicted_labels=raw_predictions,
                        true_labels=encoded_labels, time_points=times, spatial_coords=spatial,
                        classes=np.asarray(encoder.classes_, dtype=str))
    metrics, per_time = observed_k_metrics(
        raw_predictions,
        encoded_labels,
        times,
        spatial,
        heldout_indices,
    )
    selected_k, reason = choose_k(metrics)

    metrics.insert(0, "dataset", args.dataset)
    per_time.insert(0, "dataset", args.dataset)
    metrics.to_csv(args.output / "k_metrics.csv", index=False)
    per_time.to_csv(args.output / "per_time_metrics.csv", index=False)

    summary = {
        "input_h5ad": str(args.h5ad.resolve()),
        "dataset": args.dataset,
        "selected_k": selected_k,
        "selection_reason": reason,
        "candidate_k": list(K_VALUES),
        "selection_rule": (
            "maximum balanced accuracy; exact tie: maximum macro-F1; "
            "exact tie again: smaller k"
        ),
        "seed": SEED,
        "time_key": time_key,
        "annotation_key": args.annotation_key,
        "n_observed": int(adata.n_obs),
        "n_classes": int(len(encoder.classes_)),
        "feature_contract": (
            f"time + spatial_aligned[2] + {args.latent_key}"
            f"[{observed_states.shape[1] - 2}]"
        ),
        "validation_scope": "fixed stratified 10% model-selection rows",
        "classes_assigned_to_training_only": train_only_classes,
        "training": training,
    }

    torch.save(
        {
            "state_dict": {key: value.cpu() for key, value in model.state_dict().items()},
            "classes": encoder.classes_.tolist(),
            "input_size": int(features.shape[1]),
            "hidden_size": 128,
            "best_epoch": training["best_epoch"],
            "seed": SEED,
        },
        args.output / "classifier.pt",
    )
    (args.output / "selection.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

