"""Cell-type classification for downstream trajectory labeling (AnnData-first)."""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
from hashlib import sha1
import json
import os
from pathlib import Path
import re
import time
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

__all__ = [
    "build_cached_classifier_inputs_from_adata",
    "LoadedClassifierCache",
    "train_mlp_classifier",
    "train_mlp_classifier_from_adata",
    "train_cached_mlp_classifier_from_adata",
    "load_cached_mlp_classifier",
    "predict_cached_mlp_classifier_from_adata",
    "predict_labels_for_points",
    "predict_labels_for_trajectories",
    "MLP",
]


class ResidualBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LeakyReLU(0.2),
        )
        self.skip = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x) + self.skip(x)


class ResidualMLP(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_classes: int):
        super().__init__()
        hidden_size = int(hidden_size)
        if hidden_size <= 0:
            raise ValueError("hidden_size must be > 0.")
        wide_size = 4 * hidden_size
        middle_size = 2 * hidden_size
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, wide_size),
            nn.LeakyReLU(0.2),
        )
        self.res1 = ResidualBlock(wide_size, wide_size)
        self.res2 = ResidualBlock(wide_size, middle_size)
        self.res3 = ResidualBlock(middle_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.input_proj(x)
        out = self.res1(out)
        out = self.res2(out)
        out = self.res3(out)
        return self.fc_out(out)


MLP = ResidualMLP


@contextmanager
def _classifier_cache_lock(path: Path):
    """Serialize writers of one classifier cache on POSIX filesystems."""

    lock_path = path.with_name(f".{path.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+b")
    try:
        try:
            import fcntl
        except ImportError:  # pragma: no cover - non-POSIX fallback
            fcntl = None
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def _atomic_torch_save(payload: dict, path: Path) -> None:
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )
    try:
        torch.save(payload, str(temporary))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class LoadedClassifierCache:
    model: MLP
    label_encoder: object
    feature_cols: tuple[str, ...]
    label_col: str
    accuracy: Optional[float]
    balanced_accuracy: Optional[float]
    metadata: dict
    evaluation: dict

    @property
    def include_time_feature(self) -> bool:
        return bool(self.feature_cols) and str(self.feature_cols[0]) == "samples"

    @property
    def feature_dim(self) -> int:
        if self.include_time_feature:
            return max(0, len(self.feature_cols) - 1)
        return len(self.feature_cols)


def load_cached_mlp_classifier(
    cache_path: str,
    *,
    device: str = "cpu",
) -> LoadedClassifierCache:
    """Load an MLP classifier checkpoint saved by the legacy downstream cache logic.

    The old MOSTA/ARISTA review scripts store classifier checkpoints as a dict with:
    - ``state_dict``
    - ``meta`` containing ``feature_cols`` / ``classes`` / ``input_size``
    - optional ``acc`` / ``bacc``
    """
    from sklearn.preprocessing import LabelEncoder

    try:
        payload = torch.load(str(cache_path), map_location="cpu")
    except Exception as exc:
        if "Weights only load failed" not in str(exc):
            raise
        payload = torch.load(str(cache_path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported classifier cache payload type: {type(payload)}")

    meta = payload.get("meta", {})
    if not isinstance(meta, dict):
        raise TypeError("Classifier cache is missing a valid `meta` dictionary.")

    feature_cols = tuple(str(c) for c in meta.get("feature_cols", []))
    classes = tuple(str(c) for c in meta.get("classes", payload.get("classes", [])))
    if not feature_cols:
        raise KeyError("Classifier cache meta is missing `feature_cols`.")
    if not classes:
        raise KeyError("Classifier cache meta is missing `classes`.")
    if "state_dict" not in payload:
        raise KeyError("Classifier cache payload is missing `state_dict`.")

    input_size = int(meta.get("input_size", payload.get("input_size", len(feature_cols))))
    hidden_size = int(meta.get("hidden_size", payload.get("hidden_size", 128)))
    num_classes = int(payload.get("num_classes", len(classes)))
    model = ResidualMLP(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes)
    model.load_state_dict(payload["state_dict"], strict=True)

    dev = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
    model = model.to(dev)
    model.eval()

    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.asarray(classes)

    return LoadedClassifierCache(
        model=model,
        label_encoder=label_encoder,
        feature_cols=feature_cols,
        label_col=str(meta.get("label_col", "Annotation")),
        accuracy=float(payload["acc"]) if payload.get("acc") is not None else None,
        balanced_accuracy=float(payload["bacc"]) if payload.get("bacc") is not None else None,
        metadata=dict(meta),
        evaluation=dict(payload.get("evaluation", {})),
    )


def _resolve_cached_latent_indices(feature_cols: Sequence[str], latent_dim: int) -> np.ndarray:
    if not feature_cols:
        return np.zeros((0,), dtype=int)

    indices = []
    for col in feature_cols:
        match = re.fullmatch(r"x(\d+)", str(col))
        if match is None:
            raise ValueError(
                "Unsupported cached classifier feature columns. "
                "Expected latent feature names like x1, x2, ..., xN; "
                f"got {tuple(feature_cols)}."
            )
        idx = int(match.group(1)) - 1
        if idx < 0 or idx >= int(latent_dim):
            raise IndexError(
                f"Cached classifier requests latent dimension {idx + 1}, "
                f"but adata.obsm latent matrix only has {latent_dim} columns."
            )
        indices.append(idx)
    return np.asarray(indices, dtype=int)


def build_cached_classifier_inputs_from_adata(
    adata,
    cached: LoadedClassifierCache,
    *,
    latent_key: str = "latent_x",
    samples_column: str = "samples",
) -> np.ndarray:
    """Build an input matrix compatible with a legacy cached classifier.

    Legacy downstream caches store feature names such as ``samples`` and ``x1``..``x10``.
    This helper maps those feature names onto an AnnData object with latent coordinates in
    ``adata.obsm[latent_key]``.
    """
    if not (hasattr(adata, "obs") and hasattr(adata, "obsm")):
        raise TypeError(
            "Cached-classifier utilities require AnnData input. "
            f"Got: {type(adata)}"
        )
    if latent_key not in adata.obsm:
        raise KeyError(f"adata.obsm['{latent_key}'] is required to rebuild cached classifier inputs.")

    latent = np.asarray(adata.obsm[latent_key], dtype=np.float32)
    blocks = []
    latent_feature_cols = cached.feature_cols

    if cached.include_time_feature:
        if samples_column not in adata.obs.columns:
            raise KeyError(f"adata.obs['{samples_column}'] is required when the cached classifier uses `samples`.")
        from CytoBridge.tl.downstream.downstream_data import parse_time_value

        samples = np.asarray(
            [parse_time_value(v) for v in adata.obs[samples_column].values],
            dtype=np.float32,
        ).reshape(-1, 1)
        blocks.append(samples)
        latent_feature_cols = cached.feature_cols[1:]

    latent_indices = _resolve_cached_latent_indices(latent_feature_cols, latent.shape[1])
    blocks.append(latent[:, latent_indices].astype(np.float32))

    X = np.hstack(blocks).astype(np.float32)
    expected_dim = int(cached.metadata.get("input_size", X.shape[1]))
    if X.shape[1] != expected_dim:
        raise ValueError(
            f"Rebuilt cached classifier input has shape {X.shape[1]}, "
            f"but checkpoint metadata expects {expected_dim}."
        )
    return X


def predict_cached_mlp_classifier_from_adata(
    adata,
    cached: LoadedClassifierCache,
    *,
    latent_key: str = "latent_x",
    samples_column: str = "samples",
    device: Optional[str] = None,
) -> np.ndarray:
    """Predict labels for an AnnData object using a loaded legacy classifier cache."""
    X = build_cached_classifier_inputs_from_adata(
        adata,
        cached,
        latent_key=latent_key,
        samples_column=samples_column,
    )

    if device is None:
        dev = next(cached.model.parameters()).device
        model = cached.model
    else:
        dev = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
        model = cached.model.to(dev)

    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32, device=dev))
        pred_idx = torch.argmax(logits, dim=1).detach().cpu().numpy()
    return cached.label_encoder.inverse_transform(pred_idx)


def predict_labels_for_points(
    *,
    points: np.ndarray,
    time_value: float,
    model: MLP,
    label_encoder,
    feature_dim: int,
    device: str = "cuda",
    knn_neighbors: int = 50,
    include_time_feature: bool = True,
) -> np.ndarray:
    from sklearn.neighbors import KNeighborsClassifier

    pts = np.asarray(points, dtype=np.float32)
    n = int(pts.shape[0])
    if n == 0:
        return np.asarray([], dtype=str)

    dev = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
    model.eval()
    model.to(dev)

    feature_t = torch.tensor(pts[:, :feature_dim], dtype=torch.float32)
    if include_time_feature:
        samples_t = torch.full((n, 1), fill_value=float(time_value), dtype=torch.float32)
        input_t = torch.cat((samples_t, feature_t), dim=1)
        spatial_start = 1
    else:
        input_t = feature_t
        spatial_start = 0

    with torch.no_grad():
        outputs = model(input_t.float().to(dev))
        _, predicted = torch.max(outputs, 1)
        predicted_labels = label_encoder.inverse_transform(predicted.detach().cpu().numpy())

    coords = input_t[:, spatial_start:spatial_start + 2].cpu().numpy()
    k = int(min(knn_neighbors, max(1, len(coords))))
    if k <= 1:
        return np.asarray(predicted_labels).astype(str)

    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(coords, predicted_labels)
    refined_labels = knn.predict(coords)
    return np.asarray(refined_labels).astype(str)


def _prepare_classifier_arrays(
    adata,
    *,
    label_col: str,
    time_key: Optional[str],
    obsm_key: str,
    spatial_key: str,
    concat_spatial: Optional[bool],
    samples_column: str,
    include_time_feature: bool,
    n_features: Optional[int] = None,
):
    if not (hasattr(adata, "obs") and hasattr(adata, "obsm")):
        raise TypeError(
            "Downstream classification APIs require AnnData input. "
            f"Got: {type(adata)}"
        )

    from CytoBridge.tl.downstream.downstream_data import infer_time_key, parse_time_value

    if label_col not in adata.obs.columns:
        raise KeyError(
            f"Label column '{label_col}' not found in adata.obs."
        )

    resolved_time_key = infer_time_key(adata.obs, preferred=time_key)
    raw_times = adata.obs[resolved_time_key].values
    samples = np.asarray([parse_time_value(v) for v in raw_times], dtype=np.float32).reshape(-1, 1)

    if obsm_key in adata.obsm:
        latent = np.asarray(adata.obsm[obsm_key], dtype=np.float32)
    else:
        latent = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
        latent = np.asarray(latent, dtype=np.float32)

    use_spatial = bool(concat_spatial) if concat_spatial is not None else (spatial_key in adata.obsm)
    if use_spatial:
        if spatial_key not in adata.obsm:
            raise KeyError(f"concat_spatial=True but adata.obsm['{spatial_key}'] is missing.")
        spatial = np.asarray(adata.obsm[spatial_key], dtype=np.float32)
        if spatial.shape[0] != latent.shape[0]:
            raise ValueError(
                f"Row mismatch between '{spatial_key}' ({spatial.shape[0]}) and "
                f"'{obsm_key}' ({latent.shape[0]})."
            )
        features = np.hstack((spatial, latent)).astype(np.float32)
    else:
        features = latent.astype(np.float32)

    if n_features is not None:
        n_features = int(n_features)
        if n_features <= 0:
            raise ValueError("n_features must be positive when supplied.")
        if n_features > features.shape[1]:
            raise ValueError(
                f"n_features={n_features} exceeds the available joint feature "
                f"dimension {features.shape[1]}."
            )
        features = features[:, :n_features]

    X = np.hstack((samples, features)).astype(np.float32) if include_time_feature else features.astype(np.float32)
    y = adata.obs[label_col].astype(str).values
    return X, y


def _train_mlp_classifier_arrays(
    X: np.ndarray,
    y: Sequence[str],
    hidden_size: int = 128,
    epochs: int = 50,
    lr: float = 1e-3,
    test_size: float = 0.1,
    seed: int = 42,
    device: str = "cuda",
) -> Tuple[MLP, object, float]:
    model, label_encoder, accuracy, _, _ = _train_mlp_classifier_arrays_detailed(
        X=X,
        y=y,
        hidden_size=hidden_size,
        epochs=epochs,
        lr=lr,
        test_size=test_size,
        seed=seed,
        device=device,
        best_epoch_metric="accuracy",
        train_on_full_data=False,
    )
    return model, label_encoder, accuracy


def _train_mlp_classifier_arrays_detailed(
    X: np.ndarray,
    y: Sequence[str],
    hidden_size: int = 128,
    epochs: int = 50,
    lr: float = 1e-3,
    test_size: float = 0.1,
    seed: int = 42,
    device: str = "cuda",
    best_epoch_metric: str = "accuracy",
    train_on_full_data: bool = False,
    stratify_split: bool = True,
) -> Tuple[MLP, object, float, float, dict]:
    import copy

    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        classification_report,
        confusion_matrix,
    )
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    metric = str(best_epoch_metric).strip().lower()
    if metric not in {"accuracy", "bacc"}:
        raise ValueError("best_epoch_metric must be one of {'accuracy', 'bacc'}.")
    if int(epochs) <= 0:
        raise ValueError("epochs must be > 0.")
    if not 0.0 < float(test_size) < 1.0:
        raise ValueError("test_size must be in (0, 1).")

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    all_indices = np.arange(len(y_encoded), dtype=np.int64)
    stratify_used = False
    if train_on_full_data:
        train_indices = all_indices
        test_indices = all_indices
        X_train, y_train = X, y_encoded
        X_test, y_test = X, y_encoded
    else:
        class_counts = np.bincount(y_encoded, minlength=len(label_encoder.classes_))
        n_test = int(np.ceil(float(test_size) * len(y_encoded)))
        n_train = len(y_encoded) - n_test
        can_stratify = (
            bool(stratify_split)
            and bool(np.all(class_counts >= 2))
            and n_test >= len(label_encoder.classes_)
            and n_train >= len(label_encoder.classes_)
        )
        train_indices, test_indices = train_test_split(
            all_indices,
            test_size=test_size,
            random_state=seed,
            stratify=y_encoded if can_stratify else None,
        )
        stratify_used = bool(can_stratify)
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y_encoded[train_indices], y_encoded[test_indices]

    dev = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=dev)
    X_test_t = torch.tensor(X_test, dtype=torch.float32, device=dev)
    y_train_t = torch.tensor(y_train, dtype=torch.long, device=dev)

    input_size = X_train_t.shape[1]
    num_classes = int(len(label_encoder.classes_))
    model = ResidualMLP(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes).to(dev)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(epochs), eta_min=1e-5)

    best_score = float("-inf")
    best_epoch = 0
    best_model_wts = copy.deepcopy(model.state_dict())
    for epoch in range(int(epochs)):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_test_t)
            _, val_preds = torch.max(val_outputs, 1)
            val_acc = accuracy_score(y_test, val_preds.detach().cpu().numpy())
            val_bacc = balanced_accuracy_score(y_test, val_preds.detach().cpu().numpy())
            score = float(val_acc if metric == "accuracy" else val_bacc)
            if score > best_score:
                best_score = score
                best_epoch = int(epoch + 1)
                best_model_wts = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_wts)
    model.eval()
    with torch.no_grad():
        test_preds = torch.argmax(model(X_test_t), dim=1).detach().cpu().numpy()
        train_preds = torch.argmax(model(X_train_t), dim=1).detach().cpu().numpy()
    accuracy = accuracy_score(y_test, test_preds)
    balanced_accuracy = balanced_accuracy_score(y_test, test_preds)
    train_accuracy = accuracy_score(y_train, train_preds)
    train_balanced_accuracy = balanced_accuracy_score(y_train, train_preds)
    labels = np.arange(len(label_encoder.classes_), dtype=int)
    per_class = classification_report(
        y_test,
        test_preds,
        labels=labels,
        target_names=label_encoder.classes_.astype(str),
        output_dict=True,
        zero_division=0,
    )
    split_digest = sha1()
    split_digest.update(np.asarray(train_indices, dtype=np.int64).tobytes())
    split_digest.update(np.asarray(test_indices, dtype=np.int64).tobytes())
    evaluation = {
        "metric_scope": "training data" if train_on_full_data else "held-out validation split",
        "validation_is_independent_test": False,
        "best_epoch": int(best_epoch),
        "best_epoch_metric": metric,
        "best_epoch_score": float(best_score),
        "train_accuracy": float(train_accuracy),
        "train_balanced_accuracy": float(train_balanced_accuracy),
        "validation_accuracy": float(accuracy),
        "validation_balanced_accuracy": float(balanced_accuracy),
        "stratify_requested": bool(stratify_split),
        "stratify_used": bool(stratify_used),
        "split_indices_sha1": split_digest.hexdigest(),
        "n_train": int(len(train_indices)),
        "n_validation": int(len(test_indices)),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(y_test, test_preds, labels=labels).tolist(),
    }
    return model, label_encoder, float(accuracy), float(balanced_accuracy), evaluation


def _classifier_cache_fingerprint(
    adata,
    *,
    label_col: str,
    time_key: Optional[str],
    classifier_inputs: Optional[np.ndarray] = None,
) -> str:
    """Build a stable fingerprint over identities, labels, times, and classifier inputs."""
    from CytoBridge.tl.downstream.downstream_data import infer_time_key

    resolved_time_key = infer_time_key(adata.obs, preferred=time_key)
    digest = sha1()
    digest.update(f"{adata.n_obs}|{adata.n_vars}|{label_col}|{resolved_time_key}".encode("utf-8"))
    for values in (
        adata.obs_names.astype(str),
        adata.obs[label_col].astype(str).values,
        adata.obs[resolved_time_key].astype(str).values,
    ):
        digest.update("\x1f".join(map(str, values)).encode("utf-8"))
    if classifier_inputs is not None:
        inputs = np.ascontiguousarray(classifier_inputs, dtype=np.float32)
        digest.update(str(inputs.shape).encode("utf-8"))
        digest.update(inputs.tobytes())
    return digest.hexdigest()


def train_cached_mlp_classifier_from_adata(
    adata,
    *,
    cache_path: Optional[str | Path] = None,
    cache_dir: Optional[str | Path] = None,
    cache_tag: Optional[str] = None,
    reuse_if_compatible: bool = True,
    label_col: str = "Annotation",
    time_key: Optional[str] = None,
    obsm_key: str = "X_latent",
    spatial_key: str = "spatial_aligned",
    concat_spatial: Optional[bool] = None,
    samples_column: str = "samples",
    hidden_size: int = 128,
    epochs: int = 500,
    lr: float = 1e-3,
    test_size: float = 0.1,
    seed: int = 42,
    device: str = "cuda",
    include_time_feature: bool = True,
    n_features: Optional[int] = None,
    best_epoch_metric: str = "bacc",
    train_on_full_data: bool = False,
    stratify_split: bool = True,
) -> tuple[LoadedClassifierCache, Path]:
    """Train, persist, and reload a trajectory-label classifier from AnnData.

    The cache format is intentionally compatible with historical ARISTA/MOSTA
    ``classifier_resmlp_*.pt`` files, so current workflows can either reuse an old
    cache or create one through this public API. Feature names describe the joint
    aligned state (``samples``, then ``x1..xD``), independent of dataset-specific
    AnnData key names. ``n_features`` selects the leading joint dimensions after
    optional spatial concatenation; it exists for compatibility with historical
    classifiers whose caches used ``x1..xN``.
    """
    if cache_path is None and cache_dir is None:
        raise ValueError("Provide cache_path or cache_dir.")

    X, y = _prepare_classifier_arrays(
        adata,
        label_col=label_col,
        time_key=time_key,
        obsm_key=obsm_key,
        spatial_key=spatial_key,
        concat_spatial=concat_spatial,
        samples_column=samples_column,
        include_time_feature=include_time_feature,
        n_features=n_features,
    )
    feature_dim = int(X.shape[1] - (1 if include_time_feature else 0))
    feature_cols = ([samples_column] if include_time_feature else []) + [
        f"x{i + 1}" for i in range(feature_dim)
    ]
    classes = sorted({str(v) for v in y})
    metadata = {
        "version": 5,
        "cache_tag": str(cache_tag or ""),
        "feature_cols": feature_cols,
        "label_col": str(label_col),
        "hidden_size": int(hidden_size),
        "epochs": int(epochs),
        "lr": float(lr),
        "test_size": float(test_size),
        "seed": int(seed),
        "input_size": int(X.shape[1]),
        "classes": classes,
        "best_epoch_metric": str(best_epoch_metric).strip().lower(),
        "train_on_full_data": bool(train_on_full_data),
        "stratify_split": bool(stratify_split),
        "include_time_feature": bool(include_time_feature),
        "feature_selection": {
            "kind": "leading_joint_dimensions",
            "n_features": int(feature_dim),
            "requested_n_features": (
                None if n_features is None else int(n_features)
            ),
        },
        "source": {
            "kind": "AnnData",
            "n_obs": int(adata.n_obs),
            "n_vars": int(adata.n_vars),
            "fingerprint": _classifier_cache_fingerprint(
                adata,
                label_col=label_col,
                time_key=time_key,
                classifier_inputs=X,
            ),
            "obsm_key": str(obsm_key),
            "spatial_key": str(spatial_key),
            "concat_spatial": concat_spatial,
        },
    }

    if cache_path is None:
        cache_root = Path(cache_dir).expanduser().resolve()
        cache_root.mkdir(parents=True, exist_ok=True)
        key = sha1(
            json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:16]
        resolved_path = cache_root / f"classifier_resmlp_{key}.pt"
    else:
        resolved_path = Path(cache_path).expanduser().resolve()
        resolved_path.parent.mkdir(parents=True, exist_ok=True)

    with _classifier_cache_lock(resolved_path):
        # Recheck after acquiring the lock: another process may have completed
        # the same deterministic cache while this process was waiting.
        if reuse_if_compatible and resolved_path.exists():
            try:
                cached = load_cached_mlp_classifier(
                    str(resolved_path), device=device
                )
            except Exception:
                cached = None
            if cached is not None and cached.metadata == metadata:
                return cached, resolved_path

        model, label_encoder, accuracy, balanced_accuracy, evaluation = (
            _train_mlp_classifier_arrays_detailed(
                X=X,
                y=y,
                hidden_size=hidden_size,
                epochs=epochs,
                lr=lr,
                test_size=test_size,
                seed=seed,
                device=device,
                best_epoch_metric=best_epoch_metric,
                train_on_full_data=train_on_full_data,
                stratify_split=stratify_split,
            )
        )
        payload = {
            "meta": metadata,
            "state_dict": {
                key: value.detach().cpu()
                for key, value in model.state_dict().items()
            },
            "acc": float(accuracy),
            "bacc": float(balanced_accuracy),
            "evaluation": evaluation,
            "num_classes": int(len(label_encoder.classes_)),
            "saved_at": float(time.time()),
        }
        _atomic_torch_save(payload, resolved_path)
        return (
            load_cached_mlp_classifier(str(resolved_path), device=device),
            resolved_path,
        )


def train_mlp_classifier(
    adata,
    *,
    label_col: str = "Annotation",
    time_key: Optional[str] = None,
    obsm_key: str = "X_latent",
    spatial_key: str = "spatial_aligned",
    concat_spatial: Optional[bool] = None,
    samples_column: str = "samples",
    hidden_size: int = 128,
    epochs: int = 50,
    lr: float = 1e-3,
    test_size: float = 0.1,
    seed: int = 42,
    device: str = "cuda",
    include_time_feature: bool = True,
    n_features: Optional[int] = None,
) -> Tuple[MLP, object, float]:
    """Train downstream MLP classifier from AnnData."""
    X, y = _prepare_classifier_arrays(
        adata,
        label_col=label_col,
        time_key=time_key,
        obsm_key=obsm_key,
        spatial_key=spatial_key,
        concat_spatial=concat_spatial,
        samples_column=samples_column,
        include_time_feature=include_time_feature,
        n_features=n_features,
    )
    return _train_mlp_classifier_arrays(
        X=X,
        y=y,
        hidden_size=hidden_size,
        epochs=epochs,
        lr=lr,
        test_size=test_size,
        seed=seed,
        device=device,
    )


def train_mlp_classifier_from_adata(
    adata,
    *,
    label_col: str = "Annotation",
    time_key: Optional[str] = None,
    obsm_key: str = "X_latent",
    spatial_key: str = "spatial_aligned",
    concat_spatial: Optional[bool] = None,
    samples_column: str = "samples",
    hidden_size: int = 128,
    epochs: int = 50,
    lr: float = 1e-3,
    test_size: float = 0.1,
    seed: int = 42,
    device: str = "cuda",
    include_time_feature: bool = True,
    n_features: Optional[int] = None,
) -> Tuple[MLP, object, float]:
    return train_mlp_classifier(
        adata,
        label_col=label_col,
        time_key=time_key,
        obsm_key=obsm_key,
        spatial_key=spatial_key,
        concat_spatial=concat_spatial,
        samples_column=samples_column,
        hidden_size=hidden_size,
        epochs=epochs,
        lr=lr,
        test_size=test_size,
        seed=seed,
        device=device,
        include_time_feature=include_time_feature,
        n_features=n_features,
    )


def predict_labels_for_trajectories(
    sde_points: np.ndarray,
    ts_points: Sequence[float],
    model: MLP,
    label_encoder,
    feature_dim: int,
    device: str = "cuda",
    knn_neighbors: int = 50,
    include_time_feature: bool = True,
) -> Sequence[np.ndarray]:
    from sklearn.neighbors import KNeighborsClassifier

    dev = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
    model.eval()
    model.to(dev)

    predicted_labels_list = []
    for i, t in enumerate(ts_points):
        traj_t = np.asarray(sde_points[i], dtype=float)
        if traj_t.ndim == 1:
            traj_t = traj_t.reshape(1, -1)
        traj_t_tensor = torch.tensor(traj_t, dtype=torch.float32)
        n_samples = int(traj_t_tensor.shape[0])

        feature_t = traj_t_tensor[:, :feature_dim]
        if include_time_feature:
            samples_t = torch.full((n_samples, 1), fill_value=float(t), dtype=torch.float32)
            input_t = torch.cat((samples_t, feature_t), dim=1)
            spatial_start = 1
        else:
            input_t = feature_t
            spatial_start = 0

        with torch.no_grad():
            outputs = model(input_t.float().to(dev))
            _, predicted = torch.max(outputs, 1)
            predicted_labels = label_encoder.inverse_transform(predicted.detach().cpu().numpy())

        coords = input_t[:, spatial_start:spatial_start + 2].cpu().numpy()
        k = int(min(knn_neighbors, max(1, len(coords))))
        if k <= 1:
            refined_labels = predicted_labels
        else:
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(coords, predicted_labels)
            refined_labels = knn.predict(coords)

        predicted_labels_list.append(refined_labels)

    return predicted_labels_list
