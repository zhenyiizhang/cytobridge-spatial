"""Audited preprocessing for non-spatial temporal single-cell datasets.

The modeled state contains expression PCs only.  Display embeddings, clone and
cell-type annotations, and scNT new/old RNA are retained only for post-training
analyses and never enter feature selection or radius estimation.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class PreparedNonSpatialData:
    """Paths and immutable manifest returned by one preprocessing run."""

    model_h5ad: Path
    pca_artifacts: Path
    manifest: Path
    expression_h5ad: Path | None = None


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_new_outputs(paths: tuple[Path, ...], *, overwrite: bool) -> None:
    if len(set(paths)) != len(paths):
        raise ValueError("Every non-spatial output path must be distinct.")
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing non-spatial outputs: "
            + ", ".join(str(path) for path in existing)
        )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _matrix_values(matrix):
    from scipy import sparse

    return matrix.data if sparse.issparse(matrix) else np.asarray(matrix).reshape(-1)


def _difference_max_abs(left, right) -> float:
    from scipy import sparse

    difference = left.astype(np.float64) - right.astype(np.float64)
    if sparse.issparse(difference):
        difference.eliminate_zeros()
    values = _matrix_values(difference)
    return float(np.max(np.abs(values))) if values.size else 0.0


def prepare_weinreb_nonspatial(
    input_h5ad: str | Path,
    output_h5ad: str | Path,
    *,
    expression_output_h5ad: str | Path | None = None,
    artifacts_npz: str | Path | None = None,
    manifest_json: str | Path | None = None,
    n_hvg: int = 2000,
    n_pcs: int = 50,
    target_sum: float = 1.0e4,
    radius_quantile: float = 0.99,
    radius_sample_per_time: int = 2048,
    interaction_group_size: int = 16,
    seed: int = 42,
    expected_cells: int | None = None,
    overwrite: bool = False,
) -> PreparedNonSpatialData:
    """Prepare the 49k-cell Weinreb lineage dataset exactly as the accepted run.

    The supplied ``X`` is already library-normalized linear expression.  It is
    globally rescaled to ``target_sum`` and log-transformed exactly once.
    SPRING coordinates, clone identities, and annotations are not used to fit
    the PCA representation or the interaction radius.
    """

    import anndata as ad
    import scanpy as sc

    from CytoBridge.pp.state_space import (
        estimate_state_space_radius,
        state_space_fit_params,
    )

    input_path = Path(input_h5ad).expanduser().resolve()
    output_path = Path(output_h5ad).expanduser().resolve()
    artifact_path = (
        Path(artifacts_npz).expanduser().resolve()
        if artifacts_npz is not None
        else output_path.with_name(output_path.stem + "_pca_artifacts.npz")
    )
    manifest_path = (
        Path(manifest_json).expanduser().resolve()
        if manifest_json is not None
        else output_path.with_name(output_path.stem + "_manifest.json")
    )
    expression_path = (
        Path(expression_output_h5ad).expanduser().resolve()
        if expression_output_h5ad is not None
        else None
    )
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    outputs = (output_path, artifact_path, manifest_path) + (
        (expression_path,) if expression_path is not None else ()
    )
    if input_path in outputs:
        raise ValueError("The Weinreb source H5AD cannot also be an output.")
    _require_new_outputs(outputs, overwrite=overwrite)

    adata = sc.read_h5ad(input_path)
    if expected_cells is not None and adata.n_obs != int(expected_cells):
        raise ValueError(
            f"Expected {int(expected_cells)} Weinreb cells, found {adata.n_obs}."
        )
    if not adata.obs_names.is_unique or not adata.var_names.is_unique:
        raise ValueError("Weinreb cell and source-gene identifiers must be unique.")
    if not 2 <= int(n_hvg) <= adata.n_vars:
        raise ValueError("Weinreb n_hvg is outside the valid gene range.")
    if not 2 <= int(n_pcs) < min(adata.n_obs, int(n_hvg)):
        raise ValueError("Weinreb n_pcs is invalid for the requested HVG state.")
    required_obs = {
        "Time point",
        "Starting population",
        "Cell type annotation",
        "clone",
    }
    missing = sorted(required_obs.difference(adata.obs.columns))
    if missing:
        raise KeyError(f"Missing required Weinreb obs columns: {missing}")
    if "gene" not in adata.var:
        raise KeyError("Expected Weinreb gene symbols in adata.var['gene'].")
    gene_names = adata.var["gene"].astype(str).to_numpy()
    if len(set(gene_names.tolist())) != len(gene_names):
        raise ValueError("Weinreb gene symbols must be unique.")
    adata.var_names = gene_names

    original_sums = np.asarray(adata.X.sum(axis=1)).reshape(-1)
    if not np.isfinite(original_sums).all() or np.any(original_sums <= 0):
        raise ValueError("Weinreb expression has invalid library sizes.")
    time_mapping = {2: 0.0, 4: 1.0, 6: 2.0}
    observed = set(np.asarray(adata.obs["Time point"]).astype(int).tolist())
    if observed != set(time_mapping):
        raise ValueError(
            f"Expected Weinreb days {sorted(time_mapping)}, got {sorted(observed)}."
        )
    adata.obs["time_point_processed"] = (
        adata.obs["Time point"].astype(int).map(time_mapping).astype(float)
    )
    adata.obs["lineage_id"] = (
        adata.obs["Starting population"].astype(str)
        + "::"
        + adata.obs["clone"].astype(str)
    )

    sc.pp.normalize_total(adata, target_sum=float(target_sum))
    expression = adata.copy() if expression_path is not None else None
    if expression is not None:
        expression.uns["expression_semantics"] = {
            "X": "library-normalized linear expression",
            "target_sum": float(target_sum),
            "contains_clone_or_annotation_features": False,
            "use": "LR-prior construction only",
        }
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=int(n_hvg))
    sc.pp.pca(
        adata,
        n_comps=int(n_pcs),
        svd_solver="arpack",
        use_highly_variable=True,
        random_state=int(seed),
    )
    latent = np.asarray(adata.obsm["X_pca"], dtype=np.float32)
    radius = estimate_state_space_radius(
        latent,
        groups=adata.obs["time_point_processed"].to_numpy(),
        quantile=float(radius_quantile),
        max_points_per_group=int(radius_sample_per_time),
        interaction_group_size=int(interaction_group_size),
        random_seed=int(seed),
    )
    slim = ad.AnnData(X=latent.copy(), obs=adata.obs.copy())
    slim.var_names = [f"PC{index + 1}" for index in range(latent.shape[1])]
    slim.obsm["X_latent"] = latent.copy()
    slim.uns["fit_params"] = state_space_fit_params(radius)
    slim.uns["state_space_radius"] = radius
    slim.uns["preprocessing"] = {
        "dataset": "Weinreb lineage tracing",
        "input_x_semantics": "library-normalized linear expression (not raw counts)",
        "target_sum": float(target_sum),
        "log1p_applications": 1,
        "n_hvg": int(n_hvg),
        "n_pcs": int(n_pcs),
        "pca_solver": "arpack",
        "pca_random_state": int(seed),
        "time_mapping": {str(key): value for key, value in time_mapping.items()},
        "uses_spatial_coordinates": False,
        "uses_spring_coordinates": False,
        "uses_clone_or_annotation_for_preprocessing": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    if expression_path is not None:
        expression_path.parent.mkdir(parents=True, exist_ok=True)
        assert expression is not None
        expression.write_h5ad(expression_path, compression="gzip")
    slim.write_h5ad(output_path, compression="gzip")
    np.savez_compressed(
        artifact_path,
        gene_names=gene_names,
        highly_variable=np.asarray(adata.var["highly_variable"], dtype=bool),
        pca_loadings=np.asarray(adata.varm["PCs"], dtype=np.float32),
        pca_variance=np.asarray(adata.uns["pca"]["variance"], dtype=np.float64),
        pca_variance_ratio=np.asarray(
            adata.uns["pca"]["variance_ratio"], dtype=np.float64
        ),
    )
    manifest = {
        "schema_version": 2,
        "operation": "prepare_weinreb_nonspatial",
        "input_h5ad": str(input_path),
        "input_sha256": _sha256(input_path),
        "output_h5ad": str(output_path),
        "output_sha256": _sha256(output_path),
        "artifacts_npz": str(artifact_path),
        "artifacts_sha256": _sha256(artifact_path),
        "expression_output_h5ad": (
            str(expression_path) if expression_path is not None else None
        ),
        "expression_output_sha256": (
            _sha256(expression_path) if expression_path is not None else None
        ),
        "shape_original": [int(adata.n_obs), int(adata.n_vars)],
        "shape_latent": [int(latent.shape[0]), int(latent.shape[1])],
        "time_counts": {
            str(key): int(value)
            for key, value in adata.obs["Time point"]
            .value_counts()
            .sort_index()
            .items()
        },
        "state_space_radius": radius,
        "preprocessing": dict(slim.uns["preprocessing"]),
    }
    _write_json(manifest_path, manifest)
    return PreparedNonSpatialData(
        output_path,
        artifact_path,
        manifest_path,
        expression_h5ad=expression_path,
    )


def prepare_scnt_nonspatial(
    input_h5ad: str | Path,
    output_h5ad: str | Path,
    expression_output_h5ad: str | Path,
    *,
    artifacts_npz: str | Path | None = None,
    manifest_json: str | Path | None = None,
    time_key: str = "time_point_processed",
    cell_type_key: str = "cell_type",
    n_hvg: int = 2000,
    n_pcs: int = 50,
    target_sum: float = 1.0e4,
    radius_quantile: float = 0.99,
    radius_sample_per_time: int = 2048,
    interaction_group_size: int = 16,
    seed: int = 42,
    expected_cells: int | None = None,
    overwrite: bool = False,
) -> PreparedNonSpatialData:
    """Prepare scNT total RNA while sealing new/old RNA from model training."""

    import anndata as ad
    import pandas as pd
    import scanpy as sc

    from CytoBridge.pp.state_space import (
        estimate_state_space_radius,
        state_space_fit_params,
    )

    input_path = Path(input_h5ad).expanduser().resolve()
    output_path = Path(output_h5ad).expanduser().resolve()
    expression_path = Path(expression_output_h5ad).expanduser().resolve()
    artifact_path = (
        Path(artifacts_npz).expanduser().resolve()
        if artifacts_npz is not None
        else output_path.with_name(output_path.stem + "_pca_artifacts.npz")
    )
    manifest_path = (
        Path(manifest_json).expanduser().resolve()
        if manifest_json is not None
        else output_path.with_name(output_path.stem + "_manifest.json")
    )
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    outputs = (output_path, expression_path, artifact_path, manifest_path)
    if input_path in outputs:
        raise ValueError("The scNT source H5AD cannot also be an output.")
    _require_new_outputs(outputs, overwrite=overwrite)

    source = ad.read_h5ad(input_path)
    if expected_cells is not None and source.n_obs != int(expected_cells):
        raise ValueError(
            f"Expected {int(expected_cells)} scNT cells, found {source.n_obs}."
        )
    if not source.obs_names.is_unique or not source.var_names.is_unique:
        raise ValueError("scNT cell and source-gene identifiers must be unique.")
    missing_obs = {time_key, cell_type_key}.difference(source.obs.columns)
    if missing_obs:
        raise KeyError(f"Missing scNT obs columns: {sorted(missing_obs)}")
    missing_layers = {"new", "old", "total"}.difference(source.layers.keys())
    if missing_layers:
        raise KeyError(f"Missing scNT count layers: {sorted(missing_layers)}")
    for key in ("X", "new", "old", "total"):
        matrix = source.X if key == "X" else source.layers[key]
        values = _matrix_values(matrix)
        if not np.isfinite(values).all() or (values.size and np.min(values) < 0):
            raise ValueError(f"scNT {key} must contain finite non-negative counts.")
    if _difference_max_abs(source.X, source.layers["total"]) != 0.0:
        raise ValueError("scNT source X must equal its total layer.")
    if (
        _difference_max_abs(
            source.layers["total"], source.layers["new"] + source.layers["old"]
        )
        != 0.0
    ):
        raise ValueError("scNT total must equal new + old exactly.")
    time_values = np.asarray(source.obs[time_key], dtype=float)
    if not np.isfinite(time_values).all() or np.unique(time_values).size < 2:
        raise ValueError("scNT time values must contain at least two finite stages.")

    raw_sums = np.asarray(source.layers["total"].sum(axis=1)).reshape(-1)
    if np.any(raw_sums <= 0):
        raise ValueError("Every scNT cell must have a positive total-RNA library.")
    gene_symbols = (
        source.var["gene_short_name"].astype(str).to_numpy()
        if "gene_short_name" in source.var
        else source.var_names.astype(str).to_numpy()
    )
    if len(set(gene_symbols.tolist())) != source.n_vars:
        raise ValueError("scNT gene symbols must be unique.")

    normalized = ad.AnnData(
        X=source.layers["total"].astype(np.float32).copy(),
        obs=source.obs.copy(),
        var=source.var.copy(),
    )
    normalized.var_names = gene_symbols
    sc.pp.normalize_total(normalized, target_sum=float(target_sum))
    normalized_sums = np.asarray(normalized.X.sum(axis=1)).reshape(-1)
    if not np.allclose(normalized_sums, float(target_sum), rtol=0.0, atol=0.02):
        raise AssertionError("scNT library normalization failed.")

    expression = ad.AnnData(
        X=normalized.X.copy(), obs=normalized.obs.copy(), var=normalized.var.copy()
    )
    expression.var["gene_symbol"] = gene_symbols
    expression.uns["expression_semantics"] = {
        "X": "library-normalized linear total RNA",
        "target_sum": float(target_sum),
        "contains_new_or_old_layers": False,
        "use": "LR-prior construction only",
    }
    state = normalized.copy()
    sc.pp.log1p(state)
    if not 2 <= int(n_hvg) <= state.n_vars:
        raise ValueError("scNT n_hvg is outside the valid gene range.")
    sc.pp.highly_variable_genes(state, n_top_genes=int(n_hvg))
    feature_mask = np.asarray(state.var["highly_variable"], dtype=bool)
    pca_input = state[:, feature_mask].copy()
    if not 2 <= int(n_pcs) < min(pca_input.n_obs, pca_input.n_vars):
        raise ValueError("scNT n_pcs is invalid for the selected HVG state.")
    pca_mean = np.asarray(pca_input.X.mean(axis=0), dtype=np.float64).reshape(-1)
    sc.pp.pca(
        pca_input,
        n_comps=int(n_pcs),
        svd_solver="arpack",
        random_state=int(seed),
    )
    latent = np.asarray(pca_input.obsm["X_pca"], dtype=np.float32)
    radius = estimate_state_space_radius(
        latent,
        groups=time_values,
        quantile=float(radius_quantile),
        max_points_per_group=int(radius_sample_per_time),
        interaction_group_size=int(interaction_group_size),
        random_seed=int(seed),
    )
    model = ad.AnnData(
        X=latent.copy(),
        obs=source.obs.copy(),
        var=pd.DataFrame(index=[f"PC{index + 1}" for index in range(int(n_pcs))]),
    )
    model.obsm["X_latent"] = latent.copy()
    model.uns["fit_params"] = state_space_fit_params(radius)
    model.uns["state_space_radius"] = radius
    model.uns["preprocessing"] = {
        "method": "scNT total-RNA non-spatial PCA",
        "input_x_semantics": "raw total UMI counts; X == new + old",
        "target_sum": float(target_sum),
        "log1p_applications": 1,
        "feature_selection": {
            "mode": "Seurat highly variable genes on log1p total RNA",
            "n_hvg_requested": int(n_hvg),
            "n_features": int(feature_mask.sum()),
        },
        "n_pcs": int(n_pcs),
        "pca_solver": "arpack",
        "pca_random_state": int(seed),
        "time_key": str(time_key),
        "cell_type_key_retained_for_downstream_only": str(cell_type_key),
        "cell_type_used_for_feature_selection_or_radius": False,
        "uses_new_or_old_rna_for_training": False,
        "uses_spatial_coordinates": False,
    }
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
    expression.write_h5ad(expression_path, compression="gzip")
    model.write_h5ad(output_path, compression="gzip")
    np.savez_compressed(
        artifact_path,
        feature_gene_names=np.asarray(gene_symbols[feature_mask], dtype=np.str_),
        feature_mask=feature_mask,
        pca_loadings=np.asarray(pca_input.varm["PCs"], dtype=np.float32),
        pca_mean=np.asarray(pca_mean, dtype=np.float32),
        pca_variance=np.asarray(pca_input.uns["pca"]["variance"], dtype=np.float64),
        pca_variance_ratio=np.asarray(
            pca_input.uns["pca"]["variance_ratio"], dtype=np.float64
        ),
    )
    manifest = {
        "schema_version": 2,
        "operation": "prepare_scnt_nonspatial",
        "input_h5ad": str(input_path),
        "input_sha256": _sha256(input_path),
        "output_h5ad": str(output_path),
        "output_sha256": _sha256(output_path),
        "expression_output_h5ad": str(expression_path),
        "expression_output_sha256": _sha256(expression_path),
        "artifacts_npz": str(artifact_path),
        "artifacts_sha256": _sha256(artifact_path),
        "source_shape": [int(source.n_obs), int(source.n_vars)],
        "model_shape": [int(model.n_obs), int(model.n_vars)],
        "time_counts": {
            str(key): int(value)
            for key, value in source.obs[time_key].value_counts().sort_index().items()
        },
        "cell_type_counts": {
            str(key): int(value)
            for key, value in source.obs[cell_type_key]
            .astype(str)
            .value_counts()
            .sort_index()
            .items()
        },
        "state_space_radius": radius,
        "preprocessing": dict(model.uns["preprocessing"]),
        "training_blinding": {
            "new_layer_present_in_training_h5ad": False,
            "old_layer_present_in_training_h5ad": False,
            "new_layer_present_in_lr_expression_h5ad": False,
            "old_layer_present_in_lr_expression_h5ad": False,
            "cell_type_used_for_latent_or_radius": False,
        },
    }
    _write_json(manifest_path, manifest)
    return PreparedNonSpatialData(
        output_path, artifact_path, manifest_path, expression_h5ad=expression_path
    )
