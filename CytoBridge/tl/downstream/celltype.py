"""Reusable cell-type summaries for fitted spatiotemporal models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .downstream_data import adata_to_aligned_dataframe, infer_feature_columns
from .simulation import compute_velocity_components

__all__ = [
    "GrowthInteractionSummary",
    "summarize_growth_interaction_by_celltype",
    "evaluate_growth_by_timepoint",
    "summarize_label_composition",
]


@dataclass(frozen=True)
class GrowthInteractionSummary:
    """Per-cell values and grouped cell-type/time summaries."""

    raw: pd.DataFrame
    grouped: pd.DataFrame
    time_key: str
    annotation_key: str


def summarize_label_composition(
    labels_by_time,
    time_points,
) -> pd.DataFrame:
    """Return long-form cell-type counts and fractions across timepoints.

    Parameters are deliberately array-like rather than ARISTA-specific so the
    same summary can be used for observed labels, simulated trajectories, or
    classifier outputs from any dataset.
    """
    label_arrays = [np.asarray(labels).astype(str).reshape(-1) for labels in labels_by_time]
    times = list(time_points)
    if len(label_arrays) != len(times):
        raise ValueError(
            "labels_by_time and time_points must have the same length: "
            f"{len(label_arrays)} != {len(times)}."
        )
    rows = []
    for index, (time_value, labels) in enumerate(zip(times, label_arrays)):
        if labels.size == 0:
            raise ValueError(f"No labels were supplied for timepoint {time_value!r}.")
        values, counts = np.unique(labels, return_counts=True)
        total = int(counts.sum())
        for label, count in zip(values, counts):
            rows.append(
                {
                    "time_index": int(index),
                    "time": time_value,
                    "celltype": str(label),
                    "count": int(count),
                    "fraction": float(count / total),
                    "total": total,
                }
            )
    return pd.DataFrame(rows)


def evaluate_growth_by_timepoint(
    adata_dict,
    model,
    *,
    time_points,
    time_keys=None,
    annotation_key: str | None = "Annotation",
    spatial_key: str = "spatial",
    value_key: str = "growth_rate",
    device: str = "cuda",
) -> pd.DataFrame:
    """Evaluate model growth on a dictionary of observed/generated slices.

    Each slice is expected to store the complete model state in ``adata.X``.
    Growth values are written to ``adata.obs[value_key]`` and returned in a
    long-form table with spatial coordinates for reproducible plotting.
    """
    import torch

    if "growth" not in set(getattr(model, "components", [])):
        raise ValueError("Model does not contain a growth component.")
    times = list(time_points)
    keys = [str(value) for value in times] if time_keys is None else list(time_keys)
    if len(times) != len(keys):
        raise ValueError("time_points and time_keys must have the same length.")
    if hasattr(model, "to"):
        model.to(device)
    if hasattr(model, "eval"):
        model.eval()

    rows = []
    for time_value, key in zip(times, keys):
        if key not in adata_dict:
            raise KeyError(f"Missing timepoint key in adata_dict: {key!r}.")
        adata_t = adata_dict[key]
        values = adata_t.X.toarray() if hasattr(adata_t.X, "toarray") else np.asarray(adata_t.X)
        values = np.asarray(values, dtype=np.float32)
        if values.ndim != 2 or values.shape[0] == 0:
            raise ValueError(f"Slice {key!r} has invalid model-state shape {values.shape}.")
        tensor = torch.as_tensor(values, dtype=torch.float32, device=device)
        time_tensor = torch.full(
            (values.shape[0], 1),
            float(time_value),
            dtype=torch.float32,
            device=device,
        )
        with torch.no_grad():
            growth = (
                model.predict_growth(t=time_tensor, x=tensor)
                .detach()
                .cpu()
                .numpy()
                .reshape(-1)
            )
        if not np.isfinite(growth).all():
            raise FloatingPointError(f"Non-finite growth values at timepoint {time_value}.")
        adata_t.obs[value_key] = growth
        coords = (
            np.asarray(adata_t.obsm[spatial_key], dtype=np.float32)
            if spatial_key in adata_t.obsm
            else values[:, :2]
        )
        labels = (
            adata_t.obs[annotation_key].astype(str).to_numpy()
            if annotation_key is not None and annotation_key in adata_t.obs
            else np.repeat("", values.shape[0])
        )
        rows.append(
            pd.DataFrame(
                {
                    "time": float(time_value),
                    "time_key": str(key),
                    "cell_index": np.arange(values.shape[0], dtype=int),
                    "x": coords[:, 0],
                    "y": coords[:, 1],
                    "growth": growth,
                    "celltype": labels,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def summarize_growth_interaction_by_celltype(
    adata,
    model,
    *,
    annotation_key: str = "Annotation",
    time_key: Optional[str] = None,
    obsm_key: str = "X_latent",
    spatial_key: str = "spatial_aligned",
    concat_spatial: Optional[bool] = True,
    interaction_m: int = 1024,
    interaction_threshold: Optional[float] = None,
    max_cells_per_timepoint: Optional[int] = None,
    random_seed: int = 42,
    device: str = "cuda",
) -> GrowthInteractionSummary:
    """Summarize growth and interaction magnitude for every cell type and time.

    This is dataset-agnostic: AnnData supplies time, annotation, latent, and
    optional spatial keys; model evaluation and interaction calculation use the
    same public component API as the velocity workflow.
    """
    import torch

    if annotation_key not in adata.obs.columns:
        raise KeyError(f"adata.obs is missing annotation column '{annotation_key}'.")
    if max_cells_per_timepoint is not None and int(max_cells_per_timepoint) <= 0:
        raise ValueError("max_cells_per_timepoint must be positive or None.")

    frame, resolved_time_key = adata_to_aligned_dataframe(
        adata,
        time_key=time_key,
        obsm_key=obsm_key,
        spatial_key=spatial_key,
        concat_spatial=concat_spatial,
        annotation_key=annotation_key,
    )
    feature_columns = list(
        infer_feature_columns(frame, annotation_column=annotation_key)
    )
    if not feature_columns:
        raise ValueError("No model feature columns were resolved from AnnData.")
    use_spatial = (
        bool(concat_spatial)
        if concat_spatial is not None
        else spatial_key in adata.obsm
    )
    spatial_dim = (
        int(np.asarray(adata.obsm[spatial_key]).shape[1])
        if use_spatial and spatial_key in adata.obsm
        else 0
    )
    if interaction_threshold is None:
        interaction_threshold = float(
            getattr(getattr(model, "interaction_net", None), "cutoff", 1000.0)
        )

    rng = np.random.default_rng(int(random_seed))
    rows = []
    if hasattr(model, "to"):
        model.to(device)
    if hasattr(model, "eval"):
        model.eval()

    for time_value in sorted(frame["samples"].astype(float).unique()):
        positions = np.flatnonzero(
            np.isclose(frame["samples"].to_numpy(dtype=float), float(time_value))
        )
        if (
            max_cells_per_timepoint is not None
            and positions.size > int(max_cells_per_timepoint)
        ):
            positions = np.sort(
                rng.choice(
                    positions,
                    size=int(max_cells_per_timepoint),
                    replace=False,
                )
            )
        values = frame.iloc[positions][feature_columns].to_numpy(dtype=np.float32)
        components = compute_velocity_components(
            data=values,
            time_value=float(time_value),
            model=model,
            interaction_m=int(interaction_m),
            interaction_threshold=float(interaction_threshold),
            device=device,
            spatial_dim=spatial_dim,
        )
        if "growth" in set(getattr(model, "components", [])):
            values_tensor = torch.as_tensor(values, dtype=torch.float32, device=device)
            time_tensor = torch.full(
                (values.shape[0], 1),
                float(time_value),
                dtype=torch.float32,
                device=device,
            )
            with torch.no_grad():
                growth = (
                    model.predict_growth(t=time_tensor, x=values_tensor)
                    .detach()
                    .cpu()
                    .numpy()
                    .reshape(-1)
                )
        else:
            growth = np.zeros(values.shape[0], dtype=np.float32)

        rows.append(
            pd.DataFrame(
                {
                    "time": float(time_value),
                    "celltype": frame.iloc[positions][annotation_key]
                    .astype(str)
                    .to_numpy(),
                    "growth": growth,
                    "interaction": np.linalg.norm(
                        np.asarray(components["interaction"], dtype=np.float32),
                        axis=1,
                    ),
                }
            )
        )

    if not rows:
        raise ValueError("No timepoint rows were available for summarization.")
    raw = pd.concat(rows, ignore_index=True)
    grouped = (
        raw.groupby(["time", "celltype"], observed=True, sort=True)
        .agg(
            growth_mean=("growth", "mean"),
            interaction_mean=("interaction", "mean"),
            n=("growth", "size"),
        )
        .reset_index()
    )
    return GrowthInteractionSummary(
        raw=raw,
        grouped=grouped,
        time_key=resolved_time_key,
        annotation_key=annotation_key,
    )
