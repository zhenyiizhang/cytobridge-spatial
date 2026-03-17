"""Shared data-loading helpers for downstream analysis.

This module keeps downstream I/O and dataframe adaptation logic in one place
so workflow scripts and notebooks can reuse the same behavior.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd

_TIME_PATTERN = re.compile(r"-?\d+(?:\.\d+)?")


def parse_time_value(v) -> float:
    """Convert heterogeneous time labels to float.

    Supports numeric values and strings containing numeric fragments
    (e.g. ``"24hpf"``, ``"E10.5"``).
    """
    if isinstance(v, (int, float, np.number)):
        return float(v)
    s = str(v).strip().lower()
    try:
        return float(s)
    except ValueError:
        match = _TIME_PATTERN.search(s)
        if match:
            return float(match.group())
    raise ValueError(
        f"Cannot parse time value '{v}'. Please provide numeric times "
        "or add a numeric 'time_point_processed' column."
    )


def infer_time_key(obs: pd.DataFrame, preferred: Optional[str] = None) -> str:
    """Infer time column from adata.obs-like dataframe."""
    if preferred is not None:
        if preferred not in obs.columns:
            raise KeyError(f"Preferred time key '{preferred}' not found in obs columns.")
        return preferred
    for key in ("time_point_processed", "samples", "time"):
        if key in obs.columns:
            return key
    raise KeyError(
        "Cannot infer time key; expected one of {'time_point_processed','samples','time'} in obs."
    )


def adata_to_aligned_dataframe(
    adata,
    *,
    time_key: Optional[str] = None,
    obsm_key: str = "X_latent",
    spatial_key: Optional[str] = "spatial_aligned",
    concat_spatial: Optional[bool] = None,
    annotation_key: str = "Annotation",
    samples_column: str = "samples",
    cell_id_column: str = "cell_id",
) -> Tuple[pd.DataFrame, str]:
    """Convert aligned AnnData into downstream dataframe schema.

    Returns a dataframe with:
    - `samples_column` (float)
    - `x1..xD` features
    - `cell_id_column`
    - optional `annotation_key`
    """
    obs = adata.obs.copy()
    resolved_time_key = infer_time_key(obs, preferred=time_key)

    if obsm_key in adata.obsm:
        latent = np.asarray(adata.obsm[obsm_key])
    else:
        latent = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    latent = np.asarray(latent, dtype=np.float32)

    use_spatial = bool(concat_spatial) if concat_spatial is not None else (
        spatial_key is not None and spatial_key in adata.obsm
    )
    if use_spatial:
        if spatial_key is None or spatial_key not in adata.obsm:
            raise KeyError(
                f"concat_spatial=True but adata.obsm['{spatial_key}'] is missing."
            )
        spatial = np.asarray(adata.obsm[spatial_key], dtype=np.float32)
        if spatial.shape[0] != latent.shape[0]:
            raise ValueError(
                f"Row mismatch between '{spatial_key}' ({spatial.shape[0]}) and "
                f"'{obsm_key}' ({latent.shape[0]})."
            )
        X = np.hstack((spatial, latent)).astype(np.float32)
    else:
        X = latent

    dim = int(X.shape[1])
    x_cols = [f"x{i+1}" for i in range(dim)]
    df = pd.DataFrame(X, columns=x_cols, index=adata.obs_names)
    df.insert(0, samples_column, obs[resolved_time_key].values)
    df[samples_column] = [parse_time_value(v) for v in df[samples_column].values]
    df[cell_id_column] = adata.obs_names.astype(str)

    if annotation_key in obs.columns:
        df[annotation_key] = obs[annotation_key].astype(str).values

    return df, resolved_time_key


def infer_feature_columns(
    df: pd.DataFrame,
    *,
    samples_column: str = "samples",
    annotation_column: str = "Annotation",
    cell_id_column: str = "cell_id",
) -> Sequence[str]:
    """Infer ordered feature columns x1..xD, with robust fallback."""
    x_cols = [c for c in df.columns if c.startswith("x")]
    if x_cols:
        return sorted(x_cols, key=lambda s: int(s[1:]) if s[1:].isdigit() else s)
    return [
        c
        for c in df.columns
        if c not in {samples_column, annotation_column, cell_id_column}
    ]


def merge_annotation(
    df: pd.DataFrame,
    annotation_csv: str | Path,
    *,
    annotation_key: str = "Annotation",
    cell_id_column: str = "cell_id",
) -> pd.DataFrame:
    """Merge annotation column into dataframe.

    If both dataframes contain `cell_id_column`, merge by id (recommended).
    Otherwise, fallback to strict row-order merge with equal length.
    """
    anno_df = pd.read_csv(annotation_csv)
    if annotation_key not in anno_df.columns:
        raise KeyError(f"annotation csv missing required column '{annotation_key}'")

    out = df.copy()
    if cell_id_column in out.columns and cell_id_column in anno_df.columns:
        merged = out.merge(
            anno_df[[cell_id_column, annotation_key]],
            how="left",
            on=cell_id_column,
            validate="1:1",
        )
        if merged[annotation_key].isna().any():
            raise ValueError("Annotation merge by cell_id produced missing labels.")
        return merged

    if len(anno_df) != len(out):
        raise ValueError(
            "Annotation row count mismatch. Provide matching `cell_id` in both files "
            "or identical row order/length."
        )
    out[annotation_key] = anno_df[annotation_key].astype(str).values
    return out


def build_time_grid(
    *,
    df: Optional[pd.DataFrame] = None,
    adata=None,
    samples_column: str = "samples",
    time_key: Optional[str] = None,
    include_midpoints: bool = True,
    subdivisions: Optional[int] = None,
) -> Tuple[Sequence[float], Sequence[float]]:
    """Return (observed_times, integration_times).

    Supports both dataframe-first and AnnData-first usage:
    - dataframe-first: pass ``df=...`` and read ``samples_column``
    - adata-first: pass ``adata=...`` and read ``adata.obs[time_key]`` (auto-infer if omitted)
    """
    if adata is not None:
        if not hasattr(adata, "obs"):
            raise TypeError(f"`adata` must have `.obs`, got {type(adata)}")
        resolved_time_key = infer_time_key(adata.obs, preferred=time_key)
        raw = adata.obs[resolved_time_key].values
    elif df is not None:
        if samples_column not in df.columns:
            raise KeyError(f"'{samples_column}' not found in dataframe columns.")
        raw = df[samples_column].values
    else:
        raise ValueError("build_time_grid requires either `adata` or `df`.")

    observed = sorted(pd.unique(raw).tolist())
    observed = [parse_time_value(x) for x in observed]
    if len(observed) < 2:
        return observed, observed

    # Backward-compatible behavior:
    # - include_midpoints=True -> subdivisions=2
    # - include_midpoints=False -> subdivisions=1
    if subdivisions is None:
        subdivisions = 2 if include_midpoints else 1
    subdivisions = int(subdivisions)
    if subdivisions < 1:
        raise ValueError(f"subdivisions must be >= 1, got {subdivisions}.")
    if subdivisions == 1:
        return observed, observed

    ts: list[float] = [observed[0]]
    for t0, t1 in zip(observed[:-1], observed[1:]):
        delta = (t1 - t0) / float(subdivisions)
        for k in range(1, subdivisions + 1):
            ts.append(float(t0 + delta * k))
    return observed, ts
