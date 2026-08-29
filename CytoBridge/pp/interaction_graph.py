import argparse
import gc
import gzip
import os
import pickle
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from scipy.spatial import cKDTree

from CytoBridge.graph_database import selected_feature_symbol

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


def _ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)


def _resolve_output_dir(base_dir: str | None, data_name: str, default_root: str) -> str:
    """Resolve output directory with optional auto path generation."""
    if base_dir is None or str(base_dir).strip() == "":
        return os.path.join(default_root, data_name)

    base_dir_str = str(base_dir)
    normalized = base_dir_str.rstrip("/").rstrip(os.sep)
    if normalized == "":
        return os.path.join(default_root, data_name)

    # If user gives a root with trailing slash, auto-append data_name.
    if base_dir_str.endswith("/") or base_dir_str.endswith(os.sep):
        if os.path.basename(normalized) != data_name:
            return os.path.join(normalized, data_name)
    return normalized


def _resolve_lr_columns(df: pd.DataFrame) -> tuple[str, str, str, str]:
    cols = list(df.columns)
    if len(cols) < 4:
        raise ValueError("Ligand-receptor database must have at least 4 columns.")
    if "Ligand" in cols and "Receptor" in cols:
        lig_col = "Ligand"
        rec_col = "Receptor"
        pathway_col = "Pathway" if "Pathway" in cols else cols[2]
        annotation_col = "Annotation" if "Annotation" in cols else cols[3]
        return lig_col, rec_col, pathway_col, annotation_col
    # Common legacy format: ['Unnamed: 0', '0', '1', '2', '3'] where
    # '0','1','2','3' correspond to ligand/receptor/pathway/annotation.
    str_cols = [str(c) for c in cols]
    if all(k in str_cols for k in ["0", "1", "2", "3"]):
        idx = {str(c): c for c in cols}
        return idx["0"], idx["1"], idx["2"], idx["3"]
    # If the first column is an index-like unnamed column, skip it.
    if len(cols) >= 5 and str(cols[0]).lower().startswith("unnamed"):
        return cols[1], cols[2], cols[3], cols[4]
    return cols[0], cols[1], cols[2], cols[3]


def _to_dense_array(x: Any) -> np.ndarray:
    if sparse.issparse(x):
        return x.toarray()
    return np.asarray(x)


def _log(message: str, verbose: bool) -> None:
    if verbose:
        print(message)


def _progress(iterable, *, desc: str, total: int | None = None, enable: bool = True):
    if not enable or tqdm is None:
        return iterable
    return tqdm(iterable, desc=desc, total=total, leave=False)


def _complex_subunits(value: Any) -> tuple[str, ...]:
    """Parse a CellChat/COMMOT underscore-delimited LR complex."""
    parts = tuple(part.strip() for part in str(value).split("_") if part.strip())
    if not parts:
        raise ValueError(f"Empty ligand/receptor identifier: {value!r}")
    return parts


def _gene_name_lookup(
    gene_ids: list[str],
    *,
    preferred_species_tag: str | None = None,
) -> dict[str, tuple[str, ...]]:
    """Index selected gene symbols without allowing substring matches."""
    grouped: dict[str, list[str]] = defaultdict(list)
    for gene in gene_ids:
        symbol = selected_feature_symbol(
            gene,
            preferred_species_tag=preferred_species_tag,
        )
        if symbol is None:
            continue
        grouped[symbol.casefold()].append(str(gene))
    return {key: tuple(values) for key, values in grouped.items()}


def _resolve_complex_subunits(
    value: Any,
    gene_lookup: dict[str, tuple[str, ...]],
) -> tuple[tuple[str, ...] | None, str | None]:
    """Resolve every LR subunit by exact, species-tolerant gene-name equality.

    Gene-symbol capitalization differs across human, mouse and zebrafish
    resources, so equality is case-insensitive. A case-insensitive collision is
    treated as ambiguous instead of selecting an arbitrary feature.
    """
    resolved: list[str] = []
    for subunit in _complex_subunits(value):
        matches = gene_lookup.get(subunit.casefold(), ())
        if not matches:
            return None, f"missing:{subunit}"
        if len(matches) != 1:
            return None, f"ambiguous:{subunit}"
        resolved.append(matches[0])
    return tuple(resolved), None


def _read_h5ad(data_from: str | sc.AnnData) -> sc.AnnData:
    if isinstance(data_from, sc.AnnData):
        return data_from
    return sc.read_h5ad(data_from)


def _resolve_time_mask(
    adata: sc.AnnData,
    time_key: str | None,
    time_value: str | int | float | None,
) -> np.ndarray:
    if time_key is not None and time_value is not None:
        if time_key not in adata.obs:
            raise KeyError(f"time_key '{time_key}' not found in adata.obs")
        return (adata.obs[time_key] == time_value).to_numpy()
    return np.ones(adata.n_obs, dtype=bool)


def _resolve_spatial_key(adata: sc.AnnData, preferred_key: str) -> str:
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
    raise ValueError(
        "No spatial coordinates found in adata.obsm['spatial_aligned'] or adata.obsm['spatial'] "
        "or obs['spatial_x/y']."
    )


def nn1_distances(xy: np.ndarray) -> np.ndarray:
    xy = np.asarray(xy, dtype=np.float64)
    if xy.ndim != 2:
        raise ValueError(f"Expected 2D coordinates, got shape {xy.shape}.")
    if xy.shape[0] < 2:
        return np.zeros(xy.shape[0], dtype=np.float64)
    tree = cKDTree(xy)
    dist, _ = tree.query(xy, k=2)
    return dist[:, 1]


def _radius_neighbors(
    coordinates: np.ndarray,
    neighborhood_threshold: float,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Return directed positive-distance neighbors and cutoff-normalized weights.

    The same radius-graph definition is used at every dataset size.  Edges use
    the same strict spatial contract as the interaction model,
    ``1e-6 < distance < neighborhood_threshold``, and the distance weight
    ``1 - distance / neighborhood_threshold``.
    """
    coordinates = np.asarray(coordinates, dtype=np.float64)
    threshold = float(neighborhood_threshold)
    tree = cKDTree(coordinates)
    raw_neighbors = tree.query_ball_point(coordinates, r=threshold)
    inv_threshold = 1.0 / max(threshold, 1e-8)

    neighbors: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    distances: list[np.ndarray] = []
    for source, candidates in enumerate(raw_neighbors):
        targets = np.asarray(candidates, dtype=np.int32)
        targets = targets[targets != source]
        if targets.size == 0:
            neighbors.append(np.empty(0, dtype=np.int32))
            weights.append(np.empty(0, dtype=np.float32))
            distances.append(np.empty(0, dtype=np.float32))
            continue

        deltas = coordinates[targets] - coordinates[source]
        edge_distances = np.sqrt(np.einsum("ij,ij->i", deltas, deltas))
        keep = (edge_distances > 1e-6) & (edge_distances < threshold)
        targets = targets[keep]
        edge_distances = edge_distances[keep].astype(np.float32, copy=False)
        edge_weights = (1.0 - edge_distances * inv_threshold).astype(
            np.float32, copy=False
        )
        neighbors.append(targets.astype(np.int32, copy=False))
        weights.append(edge_weights)
        distances.append(edge_distances)

    return neighbors, weights, distances


def sanitize_interaction_graph_uns(adata: sc.AnnData) -> None:
    """Make `adata.uns['interaction_graph']` h5ad-write safe in-place."""
    ig = adata.uns.get("interaction_graph")
    if not isinstance(ig, dict):
        return
    # Keep uns compact: remove bulky diagnostics; use returned DataFrame / CSV instead.
    ig.pop("nn1_stats", None)
    ig.pop("nn1_stats_format", None)
    adata.uns["interaction_graph"] = ig


def estimate_neighborhood_threshold_from_aligned_spatial(
    adata: sc.AnnData,
    *,
    time_key: str | None,
    spatial_key: str = "spatial_aligned",
    recommended_spot_scale: float = 1.2,
    neighborhood_factor: float = 4.0,
    store_nn1_in_obs: bool = True,
    store_in_uns: bool = True,
    verbose: bool = False,
) -> tuple[float, float, pd.DataFrame, str]:
    """Estimate a global neighborhood threshold from per-timepoint aligned NN distances.

    Per timepoint:
    - compute nearest-neighbor distance distribution on `spatial_key`
    - recommended spot diameter = `recommended_spot_scale * median(nn1)`

    Global threshold:
    - `neighborhood_threshold = neighborhood_factor * mean(recommended_spot_diameter)`
    """
    if adata.n_obs == 0:
        raise ValueError("Cannot estimate neighborhood threshold on empty AnnData.")

    spatial_key_used = _resolve_spatial_key(adata, spatial_key)
    coords = np.asarray(adata.obsm[spatial_key_used], dtype=np.float64)

    if time_key is not None and time_key in adata.obs:
        batch_names = list(pd.unique(adata.obs[time_key]))
    else:
        batch_names = ["all"]

    nn_col = (
        "nn1_dist_aligned"
        if spatial_key_used == "spatial_aligned"
        else f"nn1_dist_{spatial_key_used}"
    )
    nn_values = np.full(adata.n_obs, np.nan, dtype=np.float64)
    rows: list[dict[str, Any]] = []
    recommended_list: list[float] = []

    batch_iter = _progress(
        batch_names,
        desc="NN1 distance stats",
        total=len(batch_names),
        enable=verbose,
    )
    for batch in batch_iter:
        if batch == "all":
            mask = np.ones(adata.n_obs, dtype=bool)
        else:
            mask = (adata.obs[time_key] == batch).to_numpy()
        xy = coords[mask]
        if xy.shape[0] == 0:
            continue

        nn = nn1_distances(xy)
        nn_values[mask] = nn

        med = float(np.median(nn)) if nn.size > 0 else 0.0
        rec = float(recommended_spot_scale * med)
        recommended_list.append(rec)

        row = {
            "timepoint": str(batch),
            "n": int(nn.size),
            "min": float(np.min(nn)) if nn.size else 0.0,
            "p05": float(np.percentile(nn, 5)) if nn.size else 0.0,
            "median": med,
            "mean": float(np.mean(nn)) if nn.size else 0.0,
            "p95": float(np.percentile(nn, 95)) if nn.size else 0.0,
            "max": float(np.max(nn)) if nn.size else 0.0,
            "recommended_spot_diameter": rec,
        }
        rows.append(row)

    if not recommended_list:
        raise ValueError(
            "Failed to estimate neighborhood threshold: no valid batches found."
        )

    mean_recommended_spot = float(np.mean(recommended_list))
    neighborhood_threshold = float(neighborhood_factor * mean_recommended_spot)

    if store_nn1_in_obs:
        adata.obs[nn_col] = nn_values

    stats_df = pd.DataFrame(rows)

    if store_in_uns:
        ig = dict(adata.uns.get("interaction_graph", {}))
        ig.update(
            {
                "spatial_key": spatial_key_used,
                "nn1_column": nn_col,
                "recommended_spot_scale": float(recommended_spot_scale),
                "neighborhood_factor": float(neighborhood_factor),
                "recommended_spot_diameter": mean_recommended_spot,
                "neighborhood_threshold": neighborhood_threshold,
                "threshold_formula": "neighborhood_factor * mean(recommended_spot_scale * median(nn1_dist))",
            }
        )
        adata.uns["interaction_graph"] = ig
        sanitize_interaction_graph_uns(adata)

    return neighborhood_threshold, mean_recommended_spot, stats_df, spatial_key_used


def generate_interaction_graph(
    data_name: str,
    data_from: str | sc.AnnData,
    data_to: str | None = None,
    metadata_to: str | None = None,
    filter_min_cell: int = 1,
    threshold_gene_exp: float = 98,
    tissue_position_file: str | None = None,
    spot_diameter: float = 0.0,
    split: int = 0,
    neighborhood_threshold: float = 0.0,
    database_path: str = "database/CellNEST_database.csv",
    time_key: str | None = "time_point_processed",
    time_value: str | int | float | None = None,
    spatial_key: str = "spatial_aligned",
    expression_layer: str | None = "counts",
    auto_neighborhood_threshold: bool = True,
    recommended_spot_scale: float = 1.2,
    neighborhood_factor: float = 4.0,
    save_metadata: bool = False,
    save_quantile_matrix: bool = False,
    verbose: bool = True,
    use_tqdm: bool = True,
    preferred_species_tag: str | None = None,
) -> dict[str, Any]:
    """Build cell-cell interaction graph.

    Preferred path is aligned AnnData (`obsm['spatial_aligned']`).
    When `neighborhood_threshold <= 0` and `auto_neighborhood_threshold=True`,
    threshold is estimated from aligned NN distance statistics.
    """
    try:
        import qnorm
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "generate_interaction_graph requires the optional 'qnorm' package. "
            "Install it with: pip install 'CytoBridge[preprocess]'"
        ) from exc

    data_to = _resolve_output_dir(data_to, data_name, default_root="input_graph")
    metadata_to = _resolve_output_dir(metadata_to, data_name, default_root="metadata")

    _ensure_dir(data_to)
    if save_metadata:
        _ensure_dir(metadata_to)
    elif metadata_to:
        _log(
            "metadata_to is provided but save_metadata=False, metadata files will be skipped.",
            verbose,
        )

    progress_enabled = bool(verbose and use_tqdm)

    threshold_source = "manual"
    spot_source = "manual"
    nn_stats_df: pd.DataFrame | None = None
    spatial_key_used = spatial_key

    if tissue_position_file is None and (
        isinstance(data_from, sc.AnnData)
        or (isinstance(data_from, str) and data_from.endswith(".h5ad"))
    ):
        adata_full = _read_h5ad(data_from)

        should_auto_threshold = auto_neighborhood_threshold and (
            neighborhood_threshold is None or float(neighborhood_threshold) <= 0
        )
        should_auto_spot = spot_diameter is None or float(spot_diameter) <= 0

        if should_auto_threshold or should_auto_spot:
            (
                est_threshold,
                est_spot,
                nn_stats_df,
                spatial_key_used,
            ) = estimate_neighborhood_threshold_from_aligned_spatial(
                adata_full,
                time_key=time_key,
                spatial_key=spatial_key,
                recommended_spot_scale=recommended_spot_scale,
                neighborhood_factor=neighborhood_factor,
                store_nn1_in_obs=True,
                store_in_uns=True,
                verbose=verbose,
            )
            if should_auto_threshold:
                neighborhood_threshold = est_threshold
                threshold_source = "auto_from_aligned_nn"
            if should_auto_spot:
                spot_diameter = est_spot
                spot_source = "auto_from_aligned_nn"
        else:
            spatial_key_used = _resolve_spatial_key(adata_full, spatial_key)

        time_mask = _resolve_time_mask(adata_full, time_key, time_value)
        if not np.any(time_mask):
            raise ValueError(
                "No cells selected by time filter. Check time_key/time_value."
            )

        expression_source = "X"
        if expression_layer is not None and expression_layer in adata_full.layers:
            expr_matrix = adata_full.layers[expression_layer][time_mask]
            expression_source = f"layers['{expression_layer}']"
        else:
            expr_matrix = adata_full.X[time_mask]
        if expression_layer is not None and expression_layer not in adata_full.layers:
            _log(
                f"Warning: expression_layer '{expression_layer}' not found; "
                "falling back to adata.X.",
                verbose,
            )

        gene_count_before = int(adata_full.n_vars)
        if filter_min_cell > 1:
            if sparse.issparse(expr_matrix):
                gene_nnz = np.asarray(expr_matrix.getnnz(axis=0)).ravel()
            else:
                gene_nnz = np.count_nonzero(np.asarray(expr_matrix), axis=0)
            gene_mask = gene_nnz >= int(filter_min_cell)
        else:
            gene_mask = np.ones(gene_count_before, dtype=bool)
        gene_count_after = int(np.sum(gene_mask))
        if gene_count_after == 0:
            raise ValueError(
                "No genes left after filtering. "
                f"filter_min_cell={filter_min_cell} is too strict for selected data."
            )
        _log(
            "Gene filtering done. "
            f"Number of genes reduced from {gene_count_before} to {gene_count_after} "
            f"(expression source: {expression_source})",
            verbose,
        )
        gene_ids = np.asarray(adata_full.var_names)[gene_mask].tolist()

        spatial_key_used = _resolve_spatial_key(adata_full, spatial_key_used)
        coordinates = np.asarray(adata_full.obsm[spatial_key_used], dtype=np.float64)[
            time_mask
        ]
        cell_barcode = np.asarray(adata_full.obs.index)[time_mask]
        _log(f"Number of barcodes: {cell_barcode.shape[0]}", verbose)
        _log("Applying quantile normalization", verbose)
        dense_expr = _to_dense_array(expr_matrix[:, gene_mask])
        temp = qnorm.quantile_normalize(np.transpose(dense_expr))
        cell_vs_gene = np.transpose(temp).astype(np.float32, copy=False)
        del temp
        del dense_expr
        del expr_matrix
    else:
        temp = sc.read_10x_mtx(data_from)
        _log("*.mtx file read done", verbose)
        gene_count_before = len(list(temp.var_names))
        sc.pp.filter_genes(temp, min_cells=filter_min_cell)
        gene_count_after = len(list(temp.var_names))
        _log(
            f"Gene filtering done. Number of genes reduced from {gene_count_before} to {gene_count_after}",
            verbose,
        )
        gene_ids = list(temp.var_names)
        cell_barcode = np.array(temp.obs.index)
        _log(f"Number of barcodes: {cell_barcode.shape[0]}", verbose)
        _log("Applying quantile normalization", verbose)
        dense_expr = _to_dense_array(temp.X)
        temp = qnorm.quantile_normalize(np.transpose(dense_expr))
        cell_vs_gene = np.transpose(temp).astype(np.float32, copy=False)
        del temp
        del dense_expr
        if tissue_position_file is None:
            raise ValueError(
                "tissue_position_file is required when data_from is not .h5ad"
            )

        df_pos = pd.read_csv(tissue_position_file, sep=",", header=None)
        tissue_position = df_pos.values
        barcode_vs_xy = {}
        for i in range(tissue_position.shape[0]):
            barcode_vs_xy[tissue_position[i][0]] = [
                tissue_position[i][4],
                tissue_position[i][5],
            ]

        coordinates = np.zeros((cell_barcode.shape[0], 2))
        for i in range(cell_barcode.shape[0]):
            coordinates[i, :] = barcode_vs_xy[cell_barcode[i]]

        if spot_diameter is None or float(spot_diameter) <= 0:
            nn_all = nn1_distances(coordinates)
            spot_diameter = float(recommended_spot_scale * np.median(nn_all))
            spot_source = "auto_from_input_coords"
        if neighborhood_threshold is None or float(neighborhood_threshold) <= 0:
            neighborhood_threshold = float(neighborhood_factor * float(spot_diameter))
            threshold_source = "auto_from_input_coords"

    neighborhood_threshold = float(neighborhood_threshold)
    spot_diameter = float(spot_diameter)
    _log(
        "Interaction graph thresholds: "
        f"spot_diameter={spot_diameter:.6f} ({spot_source}), "
        f"neighborhood_threshold={neighborhood_threshold:.6f} ({threshold_source})",
        verbose,
    )

    barcode_info: list[list[Any]] = []
    if save_metadata:
        for i, cell_code in enumerate(cell_barcode):
            entry = [cell_code] + coordinates[i, :].tolist() + [0]
            barcode_info.append(entry)

    gene_index = {gene: i for i, gene in enumerate(gene_ids)}

    if split > 0 and save_metadata:
        node_id_sorted_xy = []
        for i, _ in enumerate(cell_barcode):
            node_id_sorted_xy.append([i, coordinates[i, 0], coordinates[i, 1]])
        node_id_sorted_xy = sorted(node_id_sorted_xy, key=lambda x: (x[1], x[2]))
        with gzip.open(
            os.path.join(metadata_to, f"{data_name}_node_id_sorted_xy"), "wb"
        ) as fp:
            pickle.dump(node_id_sorted_xy, fp)
    elif split > 0 and not save_metadata:
        _log(
            "split>0 but save_metadata=False, node sorting metadata will not be saved.",
            verbose,
        )

    _log("Ligand-receptor database reading.", verbose)
    df = pd.read_csv(database_path, sep=",")
    lig_col, rec_col, pathway_col, annotation_col = _resolve_lr_columns(df)
    gene_lookup = _gene_name_lookup(
        gene_ids,
        preferred_species_tag=preferred_species_tag,
    )
    interaction_specs: dict[
        tuple[tuple[str, ...], tuple[str, ...]], dict[str, Any]
    ] = {}
    database_stats = {
        "rows_total": int(df.shape[0]),
        "rows_matched": 0,
        "rows_missing_subunit": 0,
        "rows_ambiguous_gene": 0,
        "matched_simple_pairs": 0,
        "matched_complex_pairs": 0,
        "unique_resolved_pairs": 0,
    }
    lr_iter = _progress(
        range(df.shape[0]),
        desc="Matching ligand-receptor pairs",
        total=df.shape[0],
        enable=progress_enabled,
    )
    for i in lr_iter:
        ligand_token = str(df.loc[i, lig_col]).strip()
        receptor_token = str(df.loc[i, rec_col]).strip()
        pathway = str(df.loc[i, pathway_col]).strip()
        annotation = str(df.loc[i, annotation_col]).strip()
        ligand_subunits, ligand_error = _resolve_complex_subunits(
            ligand_token, gene_lookup
        )
        receptor_subunits, receptor_error = _resolve_complex_subunits(
            receptor_token, gene_lookup
        )
        errors = tuple(
            error for error in (ligand_error, receptor_error) if error is not None
        )
        if errors:
            if any(error.startswith("ambiguous:") for error in errors):
                database_stats["rows_ambiguous_gene"] += 1
            else:
                database_stats["rows_missing_subunit"] += 1
            continue

        assert ligand_subunits is not None and receptor_subunits is not None
        database_stats["rows_matched"] += 1
        if len(ligand_subunits) > 1 or len(receptor_subunits) > 1:
            database_stats["matched_complex_pairs"] += 1
        else:
            database_stats["matched_simple_pairs"] += 1

        pair_key = (ligand_subunits, receptor_subunits)
        if pair_key not in interaction_specs:
            interaction_specs[pair_key] = {
                "ligand_token": ligand_token,
                "receptor_token": receptor_token,
                "ligand_subunits": ligand_subunits,
                "receptor_subunits": receptor_subunits,
                "pathways": set(),
                "cell_cell_contact": False,
            }
        interaction_specs[pair_key]["pathways"].add(pathway)
        interaction_specs[pair_key]["cell_cell_contact"] = bool(
            interaction_specs[pair_key]["cell_cell_contact"]
            or annotation.casefold() == "cell-cell contact".casefold()
        )

    database_stats["unique_resolved_pairs"] = int(len(interaction_specs))
    _log("Ligand-receptor database reading done.", verbose)
    _log(
        "LR exact-complex matching: "
        f"{database_stats['rows_matched']}/{database_stats['rows_total']} rows, "
        f"{database_stats['unique_resolved_pairs']} unique pairs, "
        f"{database_stats['matched_complex_pairs']} complex rows",
        verbose,
    )
    if not interaction_specs:
        raise ValueError(
            "No ligand-receptor database row could be represented by the input genes. "
            "LR identifiers are matched by exact gene name (case-insensitive), and "
            "underscore-delimited complexes require every subunit. Check that the "
            "database species and H5AD gene identifiers agree."
        )

    included_gene = sorted(
        {
            gene
            for pair in interaction_specs.values()
            for gene in (
                *pair["ligand_subunits"],
                *pair["receptor_subunits"],
            )
        },
        key=lambda gene: gene_index[gene],
    )
    _log(
        f"Total genes in this dataset: {len(gene_ids)}, "
        f"number of genes working as ligand and/or receptor: {len(included_gene)}",
        verbose,
    )

    cell_percentile = np.percentile(cell_vs_gene, threshold_gene_exp, axis=1).astype(
        np.float32, copy=False
    )
    row_min = np.min(cell_vs_gene, axis=1)
    row_max = np.max(cell_vs_gene, axis=1)
    fallback_mask = cell_percentile == row_min
    cell_percentile[fallback_mask] = row_max[fallback_mask]

    required_gene_names = included_gene

    expr_by_gene: dict[str, np.ndarray] = {}
    for gene_name in required_gene_names:
        g_idx = gene_index[gene_name]
        gene_expr = np.asarray(cell_vs_gene[:, g_idx], dtype=np.float32).copy()
        expr_by_gene[gene_name] = gene_expr

    quantile_matrix_to_save = cell_vs_gene if save_quantile_matrix else None
    del cell_vs_gene
    gc.collect()

    for relation_id, spec in enumerate(interaction_specs.values()):
        spec["relation_id"] = int(relation_id)
        spec["pathway"] = ";".join(sorted(spec.pop("pathways")))

    complex_expression_cache: dict[tuple[str, ...], np.ndarray] = {}

    def _complex_expression(subunits: tuple[str, ...]) -> np.ndarray:
        cached = complex_expression_cache.get(subunits)
        if cached is not None:
            return cached
        values = np.vstack([expr_by_gene[gene] for gene in subunits])
        # A signaling complex is available only to the extent that every
        # required subunit is available. This is the same strict minimum rule
        # used by the downstream LR analyses.
        activity = np.min(values, axis=0).astype(np.float32, copy=False)
        complex_expression_cache[subunits] = activity
        return activity

    # One radius-graph definition is used for every dataset size.  This avoids
    # the former size-dependent distance normalization and O(N^2) matrix.
    n_cells = coordinates.shape[0]
    _log(
        f"Building unified cKDTree radius graph (n_cells={n_cells}, "
        f"distance < {neighborhood_threshold:.6f})",
        verbose,
    )
    (
        valid_neighbors,
        valid_neighbor_weights,
        valid_neighbor_distances,
    ) = _radius_neighbors(coordinates, neighborhood_threshold)

    row_col = []
    edge_weight = []
    lig_rec = []
    edge_pathway = []
    self_loop_found = defaultdict(dict) if save_metadata else None

    interaction_list = list(interaction_specs.values())
    ligand_iter = _progress(
        interaction_list,
        desc="Scoring ligand-receptor edges",
        total=len(interaction_list),
        enable=progress_enabled,
    )
    for interaction in ligand_iter:
        ligand_expr = _complex_expression(interaction["ligand_subunits"])
        receptor_expr = _complex_expression(interaction["receptor_subunits"])
        ligand_active = ligand_expr >= cell_percentile
        receptor_active = receptor_expr >= cell_percentile
        source_cells = np.where(ligand_active)[0]
        for i in source_cells:
            neighbors_i = valid_neighbors[i]
            if neighbors_i.size == 0:
                continue
            neighbor_weights_i = valid_neighbor_weights[i]
            neighbor_distances_i = valid_neighbor_distances[i]
            ligand_expr_i = float(ligand_expr[i])
            recv_mask = receptor_active[neighbors_i]
            recv_cells = neighbors_i[recv_mask]
            recv_dist_weights = neighbor_weights_i[recv_mask]
            recv_distances = neighbor_distances_i[recv_mask]
            if recv_cells.size == 0:
                continue
            if interaction["cell_cell_contact"]:
                contact_mask = recv_distances <= spot_diameter
                recv_cells = recv_cells[contact_mask]
                recv_dist_weights = recv_dist_weights[contact_mask]
                if recv_cells.size == 0:
                    continue
            communication_scores = ligand_expr_i * receptor_expr[recv_cells]
            positive_mask = communication_scores > 0
            recv_cells = recv_cells[positive_mask]
            communication_scores = communication_scores[positive_mask]
            recv_dist_weights = recv_dist_weights[positive_mask]
            if recv_cells.size == 0:
                continue
            for j, score, dist_w in zip(
                recv_cells, communication_scores, recv_dist_weights
            ):
                row_col.append([int(i), int(j)])
                edge_weight.append(
                    [
                        float(dist_w),
                        float(score),
                        int(interaction["relation_id"]),
                    ]
                )
                lig_rec.append(
                    [interaction["ligand_token"], interaction["receptor_token"]]
                )
                edge_pathway.append(interaction["pathway"] or "NA")
                if save_metadata and i == j:
                    self_loop_found[int(i)][int(j)] = ""

    total_num_cell = int(coordinates.shape[0])
    _log(
        f"total number of nodes is {total_num_cell}, and edges is {len(row_col)} in the input graph",
        verbose,
    )
    _log("preprocess done.", verbose)
    _log("writing data ...", verbose)

    write_tasks: list[tuple[str, Any]] = []

    def _write_adjacency_records() -> None:
        with gzip.open(
            os.path.join(data_to, f"{data_name}_adjacency_records"), "wb"
        ) as fp:
            pickle.dump(
                [row_col, edge_weight, lig_rec, total_num_cell, edge_pathway], fp
            )

    write_tasks.append(("adjacency_records", _write_adjacency_records))

    if save_metadata:

        def _write_self_loop() -> None:
            with gzip.open(
                os.path.join(metadata_to, f"{data_name}_self_loop_record"), "wb"
            ) as fp:
                pickle.dump(self_loop_found, fp)

        def _write_barcode_info() -> None:
            with gzip.open(
                os.path.join(metadata_to, f"{data_name}_barcode_info"), "wb"
            ) as fp:
                pickle.dump(barcode_info, fp)

        def _write_gene_ids() -> None:
            pd.DataFrame(gene_ids).to_csv(
                os.path.join(metadata_to, f"gene_ids_{data_name}.csv"),
                index=False,
                header=False,
            )

        def _write_cell_barcodes() -> None:
            pd.DataFrame(cell_barcode).to_csv(
                os.path.join(metadata_to, f"cell_barcode_{data_name}.csv"),
                index=False,
                header=False,
            )

        def _write_coordinates() -> None:
            pd.DataFrame(coordinates).to_csv(
                os.path.join(metadata_to, f"coordinates_{data_name}.csv"),
                index=False,
                header=False,
            )

        def _write_graph_params() -> None:
            params_df = pd.DataFrame(
                [
                    {
                        "data_name": data_name,
                        "spatial_key": spatial_key_used,
                        "spot_diameter": spot_diameter,
                        "neighborhood_threshold": neighborhood_threshold,
                        "threshold_gene_exp": threshold_gene_exp,
                        "expression_layer": expression_layer
                        if expression_layer is not None
                        else "X",
                        "threshold_source": threshold_source,
                        "spot_source": spot_source,
                        "recommended_spot_scale": recommended_spot_scale,
                        "neighborhood_factor": neighborhood_factor,
                        "lr_database_path": os.path.abspath(database_path),
                        "lr_matching_rule": (
                            "selected_symbol_exact_case_insensitive_"
                            "all_complex_subunits"
                        ),
                        "preferred_species_tag": preferred_species_tag,
                        "lr_complex_expression_rule": "minimum",
                        **database_stats,
                    }
                ]
            )
            params_df.to_csv(
                os.path.join(metadata_to, f"{data_name}_graph_params.csv"), index=False
            )

        write_tasks.extend(
            [
                ("self_loop_record", _write_self_loop),
                ("barcode_info", _write_barcode_info),
                ("gene_ids_csv", _write_gene_ids),
                ("cell_barcode_csv", _write_cell_barcodes),
                ("coordinates_csv", _write_coordinates),
                ("graph_params_csv", _write_graph_params),
            ]
        )
        if nn_stats_df is not None:
            write_tasks.append(
                (
                    "nn1_stats_csv",
                    lambda: nn_stats_df.to_csv(
                        os.path.join(metadata_to, f"{data_name}_nn1_stats.csv"),
                        index=False,
                    ),
                )
            )

    if save_quantile_matrix:

        def _write_quantile_matrix() -> None:
            with gzip.open(
                os.path.join(data_to, f"{data_name}_cell_vs_gene_quantile_transformed"),
                "wb",
            ) as fp:
                pickle.dump(quantile_matrix_to_save, fp)

        write_tasks.append(
            (
                "cell_vs_gene_quantile_transformed",
                _write_quantile_matrix,
            )
        )

    write_iter = _progress(
        write_tasks,
        desc="Writing graph files",
        total=len(write_tasks),
        enable=progress_enabled,
    )
    for _, write_fn in write_iter:
        write_fn()

    _log("write data done", verbose)

    if "adata_full" in locals():
        graph_metadata = dict(adata_full.uns.get("interaction_graph", {}))
        graph_metadata.update(
            {
                "lr_database_path": os.path.abspath(database_path),
                "lr_matching_rule": (
                    "selected_symbol_exact_case_insensitive_all_complex_subunits"
                ),
                "preferred_species_tag": preferred_species_tag,
                "lr_complex_expression_rule": "minimum",
                "lr_database_rows_total": database_stats["rows_total"],
                "lr_database_rows_matched": database_stats["rows_matched"],
                "lr_unique_resolved_pairs": database_stats["unique_resolved_pairs"],
                "lr_matched_complex_rows": database_stats["matched_complex_pairs"],
            }
        )
        adata_full.uns["interaction_graph"] = graph_metadata
        sanitize_interaction_graph_uns(adata_full)

    return {
        "data_name": data_name,
        "spatial_key": spatial_key_used,
        "spot_diameter": float(spot_diameter),
        "neighborhood_threshold": float(neighborhood_threshold),
        "threshold_source": threshold_source,
        "spot_source": spot_source,
        "num_cells": int(total_num_cell),
        "num_edges": int(len(row_col)),
        "lr_database_path": os.path.abspath(database_path),
        "lr_matching_rule": (
            "selected_symbol_exact_case_insensitive_all_complex_subunits"
        ),
        "preferred_species_tag": preferred_species_tag,
        "lr_complex_expression_rule": "minimum",
        "lr_database_stats": database_stats,
        "saved_outputs": [name for name, _ in write_tasks],
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, required=True)
    parser.add_argument("--data_from", type=str, required=True)
    parser.add_argument(
        "--data_to",
        type=str,
        default=None,
        help="Output directory for graph files. If omitted, uses input_graph/<data_name>/",
    )
    parser.add_argument(
        "--metadata_to",
        type=str,
        default=None,
        help="Output directory for metadata files. If omitted, uses metadata/<data_name>/",
    )
    parser.add_argument("--filter_min_cell", type=int, default=1)
    parser.add_argument("--threshold_gene_exp", type=float, default=98)
    parser.add_argument("--tissue_position_file", type=str, default=None)
    parser.add_argument("--spot_diameter", type=float, default=0.0)
    parser.add_argument("--split", type=int, default=0)
    parser.add_argument("--neighborhood_threshold", type=float, default=0.0)
    parser.add_argument(
        "--database_path", type=str, default="database/CellNEST_database.csv"
    )
    parser.add_argument("--time_key", type=str, default="time_point_processed")
    parser.add_argument("--time_value", type=str, default=None)
    parser.add_argument("--spatial_key", type=str, default="spatial_aligned")
    parser.add_argument(
        "--expression_layer",
        type=str,
        default="counts",
        help="Expression layer used for LR graph (default: counts; fallback to X if absent).",
    )
    parser.add_argument(
        "--auto_neighborhood_threshold", type=int, choices=[0, 1], default=1
    )
    parser.add_argument("--recommended_spot_scale", type=float, default=1.2)
    parser.add_argument("--neighborhood_factor", type=float, default=4.0)
    parser.add_argument(
        "--save_metadata",
        type=int,
        choices=[0, 1],
        default=0,
        help="Whether to write metadata files (not needed by edge predictor).",
    )
    parser.add_argument(
        "--save_quantile_matrix",
        type=int,
        choices=[0, 1],
        default=0,
        help="Whether to persist cell-vs-gene quantile matrix (large file).",
    )
    parser.add_argument("--verbose", type=int, choices=[0, 1], default=1)
    parser.add_argument("--use_tqdm", type=int, choices=[0, 1], default=1)
    parser.add_argument(
        "--preferred_species_tag",
        type=str,
        default=None,
        help="select this [species] symbol before exact LR matching",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    generate_interaction_graph(
        data_name=args.data_name,
        data_from=args.data_from,
        data_to=args.data_to,
        metadata_to=args.metadata_to,
        filter_min_cell=args.filter_min_cell,
        threshold_gene_exp=args.threshold_gene_exp,
        tissue_position_file=args.tissue_position_file,
        spot_diameter=args.spot_diameter,
        split=args.split,
        neighborhood_threshold=args.neighborhood_threshold,
        database_path=args.database_path,
        time_key=args.time_key,
        time_value=args.time_value,
        spatial_key=args.spatial_key,
        expression_layer=args.expression_layer,
        auto_neighborhood_threshold=bool(args.auto_neighborhood_threshold),
        recommended_spot_scale=args.recommended_spot_scale,
        neighborhood_factor=args.neighborhood_factor,
        save_metadata=bool(args.save_metadata),
        save_quantile_matrix=bool(args.save_quantile_matrix),
        verbose=bool(args.verbose),
        use_tqdm=bool(args.use_tqdm),
        preferred_species_tag=args.preferred_species_tag,
    )


if __name__ == "__main__":
    main()
