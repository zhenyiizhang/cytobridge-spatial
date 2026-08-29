"""Import legacy CytoBridge model-input tables without pretending to preprocess genes.

Legacy releases often stored the already prepared model state as a CSV with one
time column followed by spatial and latent dimensions. This module migrates that
representation into the AnnData contract used by the current APIs. It does not
reconstruct unavailable gene counts.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from anndata import AnnData


_TIME_PATTERN = re.compile(r"-?\d+(?:\.\d+)?")


def _map_time_values(
    values: pd.Series,
    time_mapping: Mapping[object, float] | None,
) -> tuple[np.ndarray, dict[str, float] | None]:
    if time_mapping is not None:
        missing = [value for value in pd.unique(values) if value not in time_mapping]
        if missing:
            raise ValueError(f"time_mapping is missing values: {missing}")
        mapped = values.map(time_mapping).to_numpy(dtype=np.float64)
        mapping_used = {str(key): float(value) for key, value in time_mapping.items()}
        return mapped, mapping_used

    if pd.api.types.is_numeric_dtype(values):
        return values.to_numpy(dtype=np.float64), None

    unique = list(pd.unique(values))
    parsed: list[tuple[float | None, int, object]] = []
    for index, value in enumerate(unique):
        match = _TIME_PATTERN.search(str(value))
        parsed.append((float(match.group()) if match else None, index, value))
    if all(value[0] is not None for value in parsed):
        parsed.sort(key=lambda item: (item[0], item[1]))
        ordered = [item[2] for item in parsed]
    else:
        ordered = unique
    mapping = {value: float(index) for index, value in enumerate(ordered)}
    mapped = values.map(mapping).to_numpy(dtype=np.float64)
    return mapped, {str(key): value for key, value in mapping.items()}


def legacy_model_input_csv_to_adata(
    csv_path: str | Path,
    *,
    time_column: str = "samples",
    annotation_column: str | None = "Annotation",
    spatial_columns: Sequence[str] = ("x1", "x2"),
    latent_columns: Sequence[str] | None = None,
    time_mapping: Mapping[object, float] | None = None,
    interaction_cutoff: float | None = None,
    edge_predictor_threshold: float | None = None,
    edge_predictor_path: str | Path | None = None,
) -> AnnData:
    """Convert an already prepared model-input CSV into current AnnData keys.

    ``X`` and ``obsm['X_latent']`` contain the latent dimensions, while
    ``obsm['spatial_aligned']`` contains the spatial dimensions. Provenance
    explicitly records that gene expression is unavailable. Optional
    interaction arguments are stored so :func:`CytoBridge.tl.fit` resolves them
    through its normal dataset-override mechanism.
    """
    source = Path(csv_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    frame = pd.read_csv(source)
    if frame.empty:
        raise ValueError(f"Legacy model-input CSV is empty: {source}")
    if not frame.columns.is_unique:
        duplicates = frame.columns[frame.columns.duplicated()].tolist()
        raise ValueError(f"Duplicate CSV columns are not supported: {duplicates}")
    if time_column not in frame:
        raise KeyError(f"Missing time column {time_column!r} in {source}")

    spatial_columns = tuple(spatial_columns)
    if not spatial_columns:
        raise ValueError("spatial_columns must contain at least one column.")
    missing_spatial = [column for column in spatial_columns if column not in frame]
    if missing_spatial:
        raise KeyError(f"Missing spatial columns in {source}: {missing_spatial}")

    if annotation_column is not None and annotation_column not in frame:
        annotation_column = None

    if latent_columns is None:
        excluded = {time_column, *spatial_columns}
        if annotation_column is not None:
            excluded.add(annotation_column)
        latent_columns = tuple(column for column in frame.columns if column not in excluded)
    else:
        latent_columns = tuple(latent_columns)
    if not latent_columns:
        raise ValueError("No latent columns were selected.")
    missing_latent = [column for column in latent_columns if column not in frame]
    if missing_latent:
        raise KeyError(f"Missing latent columns in {source}: {missing_latent}")
    overlap = sorted(set(spatial_columns).intersection(latent_columns))
    if overlap:
        raise ValueError(f"Spatial and latent columns overlap: {overlap}")

    selected_columns = [*spatial_columns, *latent_columns]
    non_numeric = [
        column for column in selected_columns if not pd.api.types.is_numeric_dtype(frame[column])
    ]
    if non_numeric:
        raise TypeError(f"Model-input columns must be numeric: {non_numeric}")
    model_values = frame.loc[:, selected_columns].to_numpy(dtype=np.float64)
    if not np.isfinite(model_values).all():
        bad_rows, bad_columns = np.where(~np.isfinite(model_values))
        examples = [
            f"row={int(row)}, column={selected_columns[int(column)]}"
            for row, column in zip(bad_rows[:5], bad_columns[:5])
        ]
        raise ValueError("Model-input values must be finite; examples: " + "; ".join(examples))

    processed_time, mapping_used = _map_time_values(frame[time_column], time_mapping)
    if not np.isfinite(processed_time).all():
        raise ValueError("Processed time values must be finite.")

    obs = pd.DataFrame(index=pd.Index([f"legacy_row_{index:08d}" for index in range(len(frame))]))
    obs[time_column] = frame[time_column].to_numpy()
    obs["time_point_processed"] = processed_time
    if annotation_column is not None:
        obs[annotation_column] = frame[annotation_column].astype(str).to_numpy()

    latent = frame.loc[:, latent_columns].to_numpy(dtype=np.float32)
    spatial = frame.loc[:, spatial_columns].to_numpy(dtype=np.float32)
    var = pd.DataFrame(index=pd.Index(list(latent_columns), name="legacy_latent_dimension"))
    adata = AnnData(X=latent.copy(), obs=obs, var=var)
    adata.obsm["X_latent"] = latent
    adata.obsm["spatial_aligned"] = spatial

    fit_params: dict[str, object] = {}
    interaction_graph: dict[str, object] = {
        "spatial_key": "spatial_aligned",
        "source": "legacy_model_input_import",
    }
    if interaction_cutoff is not None:
        cutoff = float(interaction_cutoff)
        if not np.isfinite(cutoff) or cutoff <= 0:
            raise ValueError("interaction_cutoff must be a positive finite number.")
        fit_params["interaction_cutoff"] = cutoff
        interaction_graph["neighborhood_threshold"] = cutoff
        interaction_graph["threshold_source"] = "legacy_import_argument"
    if edge_predictor_threshold is not None:
        threshold = float(edge_predictor_threshold)
        if not np.isfinite(threshold) or not 0 < threshold < 1:
            raise ValueError("edge_predictor_threshold must be between 0 and 1.")
        fit_params["edge_predictor_threshold"] = threshold
        interaction_graph["edge_predictor_threshold"] = threshold
    if edge_predictor_path is not None:
        edge_path = str(Path(edge_predictor_path).expanduser())
        fit_params["edge_predictor_path"] = edge_path
        interaction_graph["edge_predictor_path"] = edge_path
    if fit_params:
        adata.uns["fit_params"] = fit_params
    adata.uns["interaction_graph"] = interaction_graph
    adata.uns["preprocess_info"] = {
        "method": "legacy_model_input_import",
        "gene_expression_available": False,
        "x_representation": "legacy_latent_model_state",
        "latent_key": "X_latent",
        "spatial_key": "spatial_aligned",
    }
    adata.uns["legacy_model_input"] = {
        "source_csv": str(source),
        "n_rows": int(len(frame)),
        "time_column": time_column,
        "time_mapping": mapping_used if mapping_used is not None else "numeric_values_preserved",
        "annotation_column": annotation_column if annotation_column is not None else "absent",
        "spatial_columns": list(spatial_columns),
        "latent_columns": list(latent_columns),
        "model_input_order": [*spatial_columns, *latent_columns],
        "gene_expression_available": False,
    }
    return adata


def write_legacy_model_input_h5ad(
    csv_path: str | Path,
    output_path: str | Path,
    **kwargs,
) -> AnnData:
    """Convert a legacy model-input CSV, write H5AD, and return the AnnData."""
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    adata = legacy_model_input_csv_to_adata(csv_path, **kwargs)
    adata.write_h5ad(output)
    return adata
