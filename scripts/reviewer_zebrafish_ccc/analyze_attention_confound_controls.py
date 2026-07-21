#!/usr/bin/env python3
"""Test whether learned spatial attention contains LR structure beyond confounders.

This analysis does not treat attention as a communication probability.  It
tests whether variation among already selected model edges is associated with
curated ligand-to-receptor compatibility after accounting for spatial distance,
transcriptional similarity, cell identity, degree, library size, and the frozen
edge-classifier probability.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import shutil
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5ad", required=True, type=Path)
    parser.add_argument("--lr-database", required=True, type=Path)
    parser.add_argument("--attribution-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--grouping-seed", type=int, default=101)
    parser.add_argument("--cell-type-key", default="Annotation")
    parser.add_argument("--time-key", default="time_point_processed")
    parser.add_argument("--state-key", default="X_latent")
    parser.add_argument("--counts-layer", default="counts")
    parser.add_argument("--non-lr-pcs", type=int, default=20)
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--random-state", type=int, default=20260722)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": int(path.stat().st_size),
        "sha256": _sha256(path),
    }


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(_json_value(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _prepare_output(path: Path, overwrite: bool) -> Path:
    path = path.expanduser().resolve()
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory is not empty: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _resolve_column(frame: pd.DataFrame, names: Sequence[str]) -> str:
    lookup = {str(column).casefold(): str(column) for column in frame.columns}
    for name in names:
        if str(name).casefold() in lookup:
            return lookup[str(name).casefold()]
    raise KeyError(f"Could not resolve any of {list(names)} from {list(frame.columns)}.")


def _load_lr_database(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    ligand = _resolve_column(raw, ("ligand", "0"))
    receptor = _resolve_column(raw, ("receptor", "1"))
    pathway = _resolve_column(raw, ("pathway", "2"))
    category = _resolve_column(raw, ("category", "annotation", "3"))
    result = pd.DataFrame(
        {
            "database_row": pd.to_numeric(
                raw.get("Unnamed: 0", pd.Series(np.arange(len(raw)))), errors="raise"
            ).astype(int),
            "ligand": raw[ligand].astype(str).str.strip(),
            "receptor": raw[receptor].astype(str).str.strip(),
            "pathway": raw[pathway].astype(str).str.strip(),
            "category": raw[category].astype(str).str.strip(),
        }
    )
    if (result[["ligand", "receptor", "pathway", "category"]] == "").any().any():
        raise ValueError("LR database contains empty identifiers.")
    return result.drop_duplicates(
        ["ligand", "receptor", "pathway", "category"]
    ).reset_index(drop=True)


def _subunits(token: str) -> tuple[str, ...]:
    values = tuple(part.strip() for part in str(token).split("_") if part.strip())
    if not values:
        raise ValueError(f"Empty complex token: {token!r}.")
    return values


def _casefold_gene_index(var_names: Iterable[object]) -> tuple[dict[str, int], set[str]]:
    candidates: dict[str, list[int]] = {}
    for index, gene in enumerate(var_names):
        candidates.setdefault(str(gene).casefold(), []).append(index)
    ambiguous = {key for key, values in candidates.items() if len(values) != 1}
    unique = {key: values[0] for key, values in candidates.items() if len(values) == 1}
    return unique, ambiguous


def _strict_lr_filter(
    database: pd.DataFrame,
    gene_index: Mapping[str, int],
    ambiguous: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    keep: list[bool] = []
    for row in database.itertuples(index=False):
        required = (*_subunits(row.ligand), *_subunits(row.receptor))
        folded = [gene.casefold() for gene in required]
        missing = [gene for gene, key in zip(required, folded) if key not in gene_index]
        ambiguous_genes = [gene for gene, key in zip(required, folded) if key in ambiguous]
        available = not missing and not ambiguous_genes
        keep.append(available)
        rows.append(
            {
                "database_row": int(row.database_row),
                "ligand": row.ligand,
                "receptor": row.receptor,
                "pathway": row.pathway,
                "available": bool(available),
                "missing_subunits": ";".join(missing),
                "ambiguous_subunits": ";".join(ambiguous_genes),
            }
        )
    filtered = database.loc[keep].reset_index(drop=True)
    if filtered.empty:
        raise ValueError("No strict-complex LR rows are available in the H5AD.")
    return filtered, pd.DataFrame(rows)


def _complex_activity(matrix, indices: Sequence[int]) -> np.ndarray:
    values = matrix[:, list(indices)]
    values = values.toarray() if hasattr(values, "toarray") else np.asarray(values)
    values = np.asarray(values, dtype=np.float32)
    return np.min(values, axis=1)


def _scaled_complex_activities(
    matrix,
    database: pd.DataFrame,
    gene_index: Mapping[str, int],
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    tokens = sorted(set(database["ligand"]) | set(database["receptor"]))
    activities: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    for token in tokens:
        indices = [gene_index[gene.casefold()] for gene in _subunits(token)]
        raw = _complex_activity(matrix, indices)
        positive = raw[raw > 0]
        scale = float(np.quantile(positive, 0.95)) if positive.size else 1.0
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0
        activities[token] = np.clip(raw / scale, 0.0, 1.0).astype(np.float32)
        rows.append(
            {
                "complex": token,
                "n_subunits": len(indices),
                "detection_fraction": float(np.mean(raw > 0)),
                "positive_q95_scale": scale,
            }
        )
    return activities, pd.DataFrame(rows)


def _pathway_balanced_weights(database: pd.DataFrame) -> np.ndarray:
    counts = database.groupby("pathway")["pathway"].transform("size").to_numpy(float)
    n_pathways = int(database["pathway"].nunique())
    weights = 1.0 / (float(n_pathways) * counts)
    if not np.isclose(weights.sum(), 1.0, atol=1e-12):
        raise RuntimeError("Pathway-balanced LR weights do not sum to one.")
    return weights


def _edge_lr_compatibility(
    source: np.ndarray,
    target: np.ndarray,
    database: pd.DataFrame,
    activities: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    weights = _pathway_balanced_weights(database)
    forward = np.zeros(source.size, dtype=np.float64)
    reverse = np.zeros(source.size, dtype=np.float64)
    active_count = np.zeros(source.size, dtype=np.int32)
    for weight, row in zip(weights, database.itertuples(index=False)):
        ligand = activities[row.ligand]
        receptor = activities[row.receptor]
        product = ligand[source] * receptor[target]
        forward += float(weight) * product
        reverse += float(weight) * ligand[target] * receptor[source]
        active_count += product > 0
    return {
        "lr_compatibility_forward": forward,
        "lr_compatibility_reverse": reverse,
        "active_lr_count": active_count,
    }


def _cosine_and_l2(
    values: np.ndarray, source: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    left = np.asarray(values[source], dtype=np.float64)
    right = np.asarray(values[target], dtype=np.float64)
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    cosine = np.divide(
        np.einsum("ij,ij->i", left, right),
        denominator,
        out=np.zeros(left.shape[0], dtype=float),
        where=denominator > 0,
    )
    l2 = np.linalg.norm(left - right, axis=1)
    return cosine, l2


def _fit_non_lr_pca(
    data,
    lr_genes: set[str],
    *,
    n_components: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    from sklearn.decomposition import PCA

    if "highly_variable" in data.var:
        candidate = data.var["highly_variable"].astype(bool).to_numpy()
    else:
        candidate = np.ones(data.n_vars, dtype=bool)
    non_lr = np.array(
        [str(gene).casefold() not in lr_genes for gene in data.var_names], dtype=bool
    )
    selected = candidate & non_lr
    if selected.sum() <= n_components:
        selected = non_lr
    matrix = data.X[:, selected]
    matrix = matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)
    matrix = np.asarray(matrix, dtype=np.float32)
    n_components = min(int(n_components), matrix.shape[0] - 1, matrix.shape[1])
    if n_components < 2:
        raise ValueError("Insufficient non-LR genes/cells for PCA control.")
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=random_state)
    scores = pca.fit_transform(matrix).astype(np.float32)

    stage = pd.to_numeric(data.obs["time_point_processed"], errors="raise").to_numpy(float)
    labels = data.obs["Annotation"].astype(str).to_numpy()
    residual = scores.copy()
    frame = pd.DataFrame({"stage": stage, "label": labels})
    for indices in frame.groupby(["stage", "label"], sort=False).indices.values():
        residual[indices] -= residual[indices].mean(axis=0, keepdims=True)
    metadata = {
        "n_selected_genes": int(selected.sum()),
        "n_components": int(n_components),
        "explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
        "selection": "HVG excluding every strict LR subunit",
        "residualization": "subtract stage-by-cell-type PC centroid",
    }
    return scores, residual, metadata


def _load_primary_edges(attribution_dir: Path, seed: int) -> pd.DataFrame:
    paths = sorted(attribution_dir.glob(f"stage_*/edges_seed_{seed}.csv.gz"))
    if not paths:
        raise FileNotFoundError(
            f"No stage_*/edges_seed_{seed}.csv.gz under {attribution_dir}."
        )
    frames = [pd.read_csv(path) for path in paths]
    result = pd.concat(frames, ignore_index=True)
    if result.empty:
        raise ValueError("Primary attribution edge table is empty.")
    return result


def _add_edge_covariates(
    edges: pd.DataFrame,
    data,
    non_lr_pca: np.ndarray,
    residual_non_lr_pca: np.ndarray,
    database: pd.DataFrame,
    activities: Mapping[str, np.ndarray],
    *,
    state_key: str,
    counts_layer: str,
) -> pd.DataFrame:
    result = edges.copy()
    source = result["source_index"].to_numpy(int)
    target = result["target_index"].to_numpy(int)
    if source.min() < 0 or target.min() < 0 or source.max() >= data.n_obs or target.max() >= data.n_obs:
        raise IndexError("Attribution edge indices do not align to the H5AD.")
    model_state = np.asarray(data.obsm[state_key], dtype=np.float32)
    result["model_state_cosine"], result["model_state_l2"] = _cosine_and_l2(
        model_state, source, target
    )
    result["non_lr_pca_cosine"], result["non_lr_pca_l2"] = _cosine_and_l2(
        non_lr_pca, source, target
    )
    result["residual_non_lr_pca_cosine"], result["residual_non_lr_pca_l2"] = (
        _cosine_and_l2(residual_non_lr_pca, source, target)
    )
    for name, values in _edge_lr_compatibility(
        source, target, database, activities
    ).items():
        result[name] = values
    counts = data.layers[counts_layer]
    library = np.asarray(counts.sum(axis=1)).reshape(-1).astype(float)
    result["log1p_source_library"] = np.log1p(library[source])
    result["log1p_target_library"] = np.log1p(library[target])
    result["source_outdegree"] = result.groupby(
        ["stage", "source_index"], sort=False
    )["target_index"].transform("size")
    result["target_indegree"] = result.groupby(
        ["stage", "target_index"], sort=False
    )["source_index"].transform("size")
    result["log1p_source_outdegree"] = np.log1p(result["source_outdegree"])
    result["log1p_target_indegree"] = np.log1p(result["target_indegree"])
    result["log_spatial_distance"] = np.log(
        np.maximum(result["spatial_distance"].to_numpy(float), 1e-8)
    )
    result["log_spatial_distance_sq"] = result["log_spatial_distance"] ** 2
    result["log1p_attention"] = np.log1p(result["attention_abs_mean"].to_numpy(float))
    result["log1p_edge_message_joint"] = np.log1p(
        result["edge_message_norm_joint"].to_numpy(float)
    )
    return result


def _cross_validated_nested_models(
    edges: pd.DataFrame,
    *,
    target_column: str,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    from scipy.stats import spearmanr
    from sklearn.compose import ColumnTransformer
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.model_selection import GroupKFold, cross_val_predict
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.linear_model import Ridge

    base_numeric = [
        "log_spatial_distance",
        "log_spatial_distance_sq",
        "model_state_cosine",
        "model_state_l2",
        "non_lr_pca_cosine",
        "non_lr_pca_l2",
        "residual_non_lr_pca_cosine",
        "residual_non_lr_pca_l2",
        "edge_predictor_probability",
        "log1p_source_library",
        "log1p_target_library",
        "log1p_source_outdegree",
        "log1p_target_indegree",
    ]
    categoricals = ["stage_label", "sender_type", "receiver_type"]
    models = {
        "confounders_only": base_numeric,
        "confounders_plus_forward_lr": base_numeric + ["lr_compatibility_forward"],
        "confounders_plus_reverse_lr": base_numeric + ["lr_compatibility_reverse"],
    }
    y = edges[target_column].to_numpy(float)
    groups = (
        edges["stage"].astype(str) + ":" + edges["target_index"].astype(str)
    ).to_numpy()
    n_splits = min(5, len(np.unique(groups)))
    if n_splits < 2:
        raise ValueError("Insufficient receiver groups for cross-validation.")
    cv = GroupKFold(n_splits=n_splits)
    predictions: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    for model_name, numeric in models.items():
        try:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
        except TypeError:  # scikit-learn < 1.2
            encoder = OneHotEncoder(handle_unknown="ignore", sparse=True)
        preprocessor = ColumnTransformer(
            [
                ("numeric", StandardScaler(), numeric),
                ("categorical", encoder, categoricals),
            ]
        )
        pipeline = Pipeline(
            [("features", preprocessor), ("ridge", Ridge(alpha=1.0, solver="lsqr"))]
        )
        columns = numeric + categoricals
        prediction = cross_val_predict(
            pipeline,
            edges[columns],
            y,
            groups=groups,
            cv=cv,
            method="predict",
            n_jobs=1,
        )
        predictions[model_name] = prediction
        rows.append(
            {
                "target": target_column,
                "model": model_name,
                "n_edges": int(len(edges)),
                "n_receiver_groups": int(len(np.unique(groups))),
                "n_folds": int(n_splits),
                "out_of_fold_r2": float(r2_score(y, prediction)),
                "out_of_fold_rmse": float(mean_squared_error(y, prediction) ** 0.5),
                "out_of_fold_spearman": float(spearmanr(y, prediction).statistic),
            }
        )
    table = pd.DataFrame(rows)
    base_r2 = float(
        table.loc[table["model"] == "confounders_only", "out_of_fold_r2"].iloc[0]
    )
    table["delta_r2_vs_confounders"] = table["out_of_fold_r2"] - base_r2
    return table, predictions


def _rank_correlation(left: np.ndarray, right: np.ndarray) -> float:
    from scipy.stats import rankdata

    x = rankdata(np.asarray(left, dtype=float), method="average")
    y = rankdata(np.asarray(right, dtype=float), method="average")
    x -= x.mean()
    y -= y.mean()
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    return float(np.dot(x, y) / denominator) if denominator else float("nan")


def _conditional_permutation(
    edges: pd.DataFrame,
    residual: np.ndarray,
    score_column: str,
    *,
    keys: Sequence[str],
    min_stratum_size: int,
    n_permutations: int,
    random_state: int,
) -> dict[str, Any]:
    from scipy.stats import rankdata

    counts = edges.groupby(list(keys), sort=False, dropna=False)[score_column].transform("size")
    mask = counts.to_numpy(int) >= int(min_stratum_size)
    subset = edges.loc[mask].copy()
    residual = np.asarray(residual)[mask]
    score = subset[score_column].to_numpy(float)
    groups = [
        np.asarray(indices, dtype=int)
        for indices in subset.groupby(list(keys), sort=False, dropna=False).indices.values()
    ]
    residual_rank = rankdata(residual, method="average").astype(float)
    score_rank = rankdata(score, method="average").astype(float)
    residual_rank -= residual_rank.mean()
    score_rank -= score_rank.mean()
    denominator = np.linalg.norm(residual_rank) * np.linalg.norm(score_rank)
    observed = float(np.dot(residual_rank, score_rank) / denominator) if denominator else np.nan
    rng = np.random.default_rng(int(random_state))
    null = np.empty(int(n_permutations), dtype=float)
    permuted = score_rank.copy()
    for permutation in range(int(n_permutations)):
        for indices in groups:
            permuted[indices] = score_rank[rng.permutation(indices)]
        null[permutation] = (
            float(np.dot(residual_rank, permuted) / denominator)
            if denominator
            else np.nan
        )
    p_greater = float((1 + np.sum(null >= observed)) / (len(null) + 1))
    return {
        "score": score_column,
        "strata": "+".join(keys),
        "min_stratum_size": int(min_stratum_size),
        "n_edges_total": int(len(edges)),
        "n_edges_retained": int(mask.sum()),
        "retained_fraction": float(mask.mean()),
        "n_strata": int(len(groups)),
        "observed_spearman": observed,
        "null_mean": float(np.nanmean(null)),
        "null_sd": float(np.nanstd(null, ddof=1)),
        "empirical_p_greater": p_greater,
        "n_permutations": int(n_permutations),
    }


def _add_strata_bins(edges: pd.DataFrame, bins: int = 5) -> pd.DataFrame:
    result = edges.copy()
    for source, target in (
        ("spatial_distance", "distance_bin"),
        ("residual_non_lr_pca_cosine", "state_bin"),
    ):
        result[target] = result.groupby("stage", sort=False)[source].transform(
            lambda values: pd.qcut(
                values.rank(method="first"), q=min(bins, len(values)), labels=False, duplicates="drop"
            )
        )
    return result


def _matched_top_low(
    edges: pd.DataFrame,
    *,
    target: str,
    score: str,
    keys: Sequence[str],
    min_stratum_size: int = 10,
) -> dict[str, Any]:
    from scipy.stats import wilcoxon

    differences: list[float] = []
    top_values: list[float] = []
    low_values: list[float] = []
    n_top = 0
    n_low = 0
    for _, frame in edges.groupby(list(keys), sort=False, dropna=False):
        if len(frame) < min_stratum_size:
            continue
        q_low, q_high = frame[target].quantile([0.2, 0.8])
        low = frame.loc[frame[target] <= q_low, score].to_numpy(float)
        top = frame.loc[frame[target] >= q_high, score].to_numpy(float)
        if not len(low) or not len(top):
            continue
        low_mean = float(np.mean(low))
        top_mean = float(np.mean(top))
        low_values.append(low_mean)
        top_values.append(top_mean)
        differences.append(top_mean - low_mean)
        n_low += len(low)
        n_top += len(top)
    if differences and not np.allclose(differences, 0):
        pvalue = float(wilcoxon(differences, alternative="greater").pvalue)
    else:
        pvalue = 1.0
    return {
        "target": target,
        "score": score,
        "strata": "+".join(keys),
        "n_matched_strata": int(len(differences)),
        "n_top_edges": int(n_top),
        "n_low_edges": int(n_low),
        "mean_top_score_across_strata": float(np.mean(top_values)) if top_values else np.nan,
        "mean_low_score_across_strata": float(np.mean(low_values)) if low_values else np.nan,
        "median_paired_difference": float(np.median(differences)) if differences else np.nan,
        "wilcoxon_greater_p": pvalue,
    }


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    import anndata as ad
    from scipy.stats import spearmanr

    h5ad_path = args.h5ad.expanduser().resolve()
    lr_path = args.lr_database.expanduser().resolve()
    attribution_dir = args.attribution_dir.expanduser().resolve()
    for path in (h5ad_path, lr_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not attribution_dir.is_dir():
        raise FileNotFoundError(attribution_dir)
    output = _prepare_output(args.output_dir, bool(args.overwrite))
    data = ad.read_h5ad(h5ad_path)
    for key in (args.cell_type_key, args.time_key):
        if key not in data.obs:
            raise KeyError(f"Missing adata.obs[{key!r}].")
    if args.state_key not in data.obsm:
        raise KeyError(f"Missing adata.obsm[{args.state_key!r}].")
    if args.counts_layer not in data.layers:
        raise KeyError(f"Missing adata.layers[{args.counts_layer!r}].")

    database_raw = _load_lr_database(lr_path)
    gene_index, ambiguous = _casefold_gene_index(data.var_names)
    database, database_audit = _strict_lr_filter(
        database_raw, gene_index, ambiguous
    )
    activities, activity_audit = _scaled_complex_activities(
        data.X, database, gene_index
    )
    lr_genes = {
        gene.casefold()
        for token in set(database["ligand"]) | set(database["receptor"])
        for gene in _subunits(token)
    }
    # These column names are fixed by the corrected zebrafish data contract.
    if args.time_key != "time_point_processed" or args.cell_type_key != "Annotation":
        renamed = data.copy()
        renamed.obs["time_point_processed"] = data.obs[args.time_key].to_numpy()
        renamed.obs["Annotation"] = data.obs[args.cell_type_key].astype(str).to_numpy()
        pca_data = renamed
    else:
        pca_data = data
    non_lr_pca, residual_non_lr_pca, pca_metadata = _fit_non_lr_pca(
        pca_data,
        lr_genes,
        n_components=int(args.non_lr_pcs),
        random_state=int(args.random_state),
    )
    edges = _load_primary_edges(attribution_dir, int(args.grouping_seed))
    edges = _add_edge_covariates(
        edges,
        data,
        non_lr_pca,
        residual_non_lr_pca,
        database,
        activities,
        state_key=args.state_key,
        counts_layer=args.counts_layer,
    )
    edges = _add_strata_bins(edges)

    edge_path = output / f"edge_controls_seed_{args.grouping_seed}.csv.gz"
    edges.to_csv(edge_path, index=False, compression="gzip")
    database_audit_path = output / "lr_database_availability_audit.csv"
    database_audit.to_csv(database_audit_path, index=False)
    activity_audit_path = output / "lr_complex_activity_audit.csv"
    activity_audit.to_csv(activity_audit_path, index=False)

    correlation_rows: list[dict[str, Any]] = []
    targets = ["attention_abs_mean", "edge_message_norm_joint"]
    covariates = [
        "spatial_distance",
        "model_state_cosine",
        "non_lr_pca_cosine",
        "residual_non_lr_pca_cosine",
        "edge_predictor_probability",
        "lr_compatibility_forward",
        "lr_compatibility_reverse",
        "active_lr_count",
    ]
    for stage_value in ["all", *sorted(edges["stage"].unique())]:
        frame = edges if stage_value == "all" else edges.loc[edges["stage"] == stage_value]
        for target in targets:
            for covariate in covariates:
                statistic = spearmanr(frame[target], frame[covariate])
                correlation_rows.append(
                    {
                        "stage": stage_value,
                        "target": target,
                        "covariate": covariate,
                        "n_edges": int(len(frame)),
                        "spearman": float(statistic.statistic),
                        "p_value_naive_edges_not_independent": float(statistic.pvalue),
                    }
                )
    correlations = pd.DataFrame(correlation_rows)
    correlation_path = output / "descriptive_edge_correlations.csv"
    correlations.to_csv(correlation_path, index=False)

    nested_tables: list[pd.DataFrame] = []
    prediction_map: dict[str, np.ndarray] = {}
    for target in ("log1p_attention", "log1p_edge_message_joint"):
        table, predictions = _cross_validated_nested_models(
            edges, target_column=target
        )
        nested_tables.append(table)
        for name, values in predictions.items():
            prediction_map[f"{target}:{name}"] = values
            edges[f"oof_{target}_{name}"] = values
        edges[f"oof_{target}_confounder_residual"] = (
            edges[target].to_numpy(float) - predictions["confounders_only"]
        )
    nested = pd.concat(nested_tables, ignore_index=True)
    nested_path = output / "nested_grouped_cv_metrics.csv"
    nested.to_csv(nested_path, index=False)

    permutation_rows: list[dict[str, Any]] = []
    strict_keys = [
        "stage",
        "sender_type",
        "receiver_type",
        "distance_bin",
        "state_bin",
    ]
    coarse_keys = ["stage", "distance_bin", "state_bin"]
    for target in ("log1p_attention", "log1p_edge_message_joint"):
        residual = edges[f"oof_{target}_confounder_residual"].to_numpy(float)
        for score in ("lr_compatibility_forward", "lr_compatibility_reverse"):
            for keys, minimum in ((strict_keys, 4), (coarse_keys, 8)):
                row = _conditional_permutation(
                    edges,
                    residual,
                    score,
                    keys=keys,
                    min_stratum_size=minimum,
                    n_permutations=int(args.permutations),
                    random_state=int(args.random_state),
                )
                row["target"] = target
                row["residual_definition"] = "out-of-fold target minus confounders-only prediction"
                permutation_rows.append(row)
    permutation_table = pd.DataFrame(permutation_rows)
    permutation_path = output / "conditional_permutation_tests.csv"
    permutation_table.to_csv(permutation_path, index=False)

    matched_rows: list[dict[str, Any]] = []
    for target in ("attention_abs_mean", "edge_message_norm_joint"):
        for score in (
            "lr_compatibility_forward",
            "lr_compatibility_reverse",
            "active_lr_count",
        ):
            for keys in (strict_keys, coarse_keys):
                matched_rows.append(
                    _matched_top_low(
                        edges,
                        target=target,
                        score=score,
                        keys=keys,
                    )
                )
    matched = pd.DataFrame(matched_rows)
    matched_path = output / "matched_top_vs_low_enrichment.csv"
    matched.to_csv(matched_path, index=False)
    # Rewrite with OOF residual columns included for downstream diagnostics.
    edges.to_csv(edge_path, index=False, compression="gzip")

    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "cytobridge_attention_exact_message_lr_confound_controls",
        "claims": {
            "attention_is_probability": False,
            "cross_method_or_lr_agreement_is_independent_experimental_validation": False,
            "purpose": (
                "test residual CCC-related structure while explicitly conditioning "
                "on proximity, state similarity, cell identity, degree, library size, "
                "and the frozen edge classifier"
            ),
        },
        "input": {
            "h5ad": _artifact(h5ad_path),
            "lr_database": _artifact(lr_path),
            "attribution_manifest": _artifact(attribution_dir / "run_manifest.json"),
            "grouping_seed": int(args.grouping_seed),
        },
        "lr_scoring": {
            "database_rows_input_deduplicated": int(len(database_raw)),
            "database_rows_strict_available": int(len(database)),
            "complex_rule": "minimum across all underscore-delimited subunits",
            "activity_scale": "global positive q95 then clip to [0,1]",
            "row_weights": "equal pathways; equal rows within pathway",
            "reverse_control": "same LR rows with sender and receiver cells swapped",
            "circularity_warning": (
                "the frozen edge classifier was itself trained from an LR-informed "
                "graph, so curated LR enrichment is a consistency test, not an "
                "independent validation"
            ),
        },
        "non_lr_pca": pca_metadata,
        "statistics": {
            "cv_group": "receiver cell within stage",
            "cv_folds": 5,
            "conditional_permutations": int(args.permutations),
            "empirical_p_correction": "(b+1)/(B+1)",
            "strict_strata": strict_keys,
            "coarse_strata": coarse_keys,
            "naive_edge_p_values_in_correlation_table_are_descriptive_only": True,
        },
        "artifacts": {
            "edge_controls": _artifact(edge_path),
            "lr_database_audit": _artifact(database_audit_path),
            "lr_complex_activity_audit": _artifact(activity_audit_path),
            "descriptive_correlations": _artifact(correlation_path),
            "nested_grouped_cv": _artifact(nested_path),
            "conditional_permutations": _artifact(permutation_path),
            "matched_top_vs_low": _artifact(matched_path),
        },
    }
    manifest_path = output / "run_manifest.json"
    _write_json(manifest_path, manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = run(args)
    print(json.dumps({"status": "ok", "artifacts": manifest["artifacts"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
