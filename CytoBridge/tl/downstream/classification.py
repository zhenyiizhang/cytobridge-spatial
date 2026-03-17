"""Cell-type classification for downstream trajectory labeling (AnnData-first)."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

__all__ = [
    "train_mlp_classifier",
    "train_mlp_classifier_from_adata",
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
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.LeakyReLU(0.2),
        )
        self.res1 = ResidualBlock(512, 512)
        self.res2 = ResidualBlock(512, 256)
        self.res3 = ResidualBlock(256, 128)
        self.fc_out = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.input_proj(x)
        out = self.res1(out)
        out = self.res2(out)
        out = self.res3(out)
        return self.fc_out(out)


MLP = ResidualMLP


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
    import copy

    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    torch.manual_seed(seed)
    np.random.seed(seed)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=test_size, random_state=seed)

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

    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    for _ in range(int(epochs)):
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
            if val_acc > best_acc:
                best_acc = float(val_acc)
                best_model_wts = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_wts)
    model.eval()
    with torch.no_grad():
        preds = torch.argmax(model(X_test_t), dim=1)
    accuracy = accuracy_score(y_test, preds.detach().cpu().numpy())
    return model, label_encoder, float(accuracy)


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
