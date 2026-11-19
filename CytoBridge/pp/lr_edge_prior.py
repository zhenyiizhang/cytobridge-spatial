"""Expression-guided ligand--receptor edge priors for non-spatial data.

This module turns a curated ligand--receptor (LR) database and observed
expression into a frozen, directed pair classifier that is compatible with
``GNNInteraction(edge_mode="predictor")``.  It deliberately does *not* run a
spatial communication method: physical distances are unavailable in this
setting, and neither PCA nor a visualization embedding is treated as space.

The default score is fixed before any downstream model is evaluated:

1. Starting from non-negative, linear expression, library-normalize every
   cell to 10,000 and apply ``log1p`` exactly once.
2. Retain CellChatDB ``Secreted Signaling`` rows for which every subunit is
   present and detected in at least 50 cells globally.  A heteromeric complex
   has activity equal to the minimum expression of all its subunits, matching
   the strict complex semantics commonly used by COMMOT.
3. For LR row ``p``, divide ligand and receptor activities by their respective
   95th percentiles among *positive training-cell activities*, then clip each
   to ``[0, 1]``.  Positive-only quantiles retain rare signals whose global
   95th percentile would otherwise be zero.
4. For sender ``i`` and receiver ``j``, compute

   ``S(i,j) = mean_h mean_{p in pathway h} L_tilde[i,p] R_tilde[j,p]``.

   Thus pathways, rather than database row counts, receive equal total weight.
5. Within each time point, label sampled training pairs above the training
   score's 80th percentile as edges.  With random GNN groups of 16 cells, a
   20% directed edge rate gives an expected in-degree near three and avoids
   the substantial isolation induced by a top-10% rule.

The global ``min_cells`` filter is an unsupervised expression-detection filter
and is intentionally applied before splitting; it cannot access outcomes or
annotations.  Cells are then split, within time point, into disjoint
train/validation/test sets before any activity scaling, thresholding, or
predictor fitting.  Pair sampling is restricted to the same latent-space
radius used by the GNN.  Only the time column is read from ``obs``; clone,
fate, cell type, starting population, and visualization coordinates are never
used.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import random
from typing import Any, Iterable, Mapping, Optional, Sequence

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import cKDTree
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from CytoBridge.tl.graph.spatial_gnn import LinkPredictorMLP


@dataclass(frozen=True)
class LREdgePriorConfig:
    """Configuration for :func:`build_lr_edge_prior`.

    The defaults are the preregistered non-spatial Weinreb policy.  They are
    exposed so another dataset can change them explicitly and have the exact
    values recorded in the output manifest.
    """

    time_key: str = "Time point"
    gene_symbol_key: str = "gene"
    latent_key: Optional[str] = "X_latent"
    annotation_filter: tuple[str, ...] = ("Secreted Signaling",)
    min_cells: int = 50
    target_sum: float = 1.0e4
    activity_scale_quantile: float = 0.95
    edge_score_quantile: float = 0.80
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    train_pairs_per_time: int = 200_000
    validation_pairs_per_time: int = 75_000
    test_pairs_per_time: int = 75_000
    pair_score_batch_size: int = 8_192
    candidate_radius: Optional[float] = None
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-5
    epochs: int = 50
    batch_size: int = 4_096
    patience: int = 8
    min_delta: float = 1.0e-5
    decision_threshold_mode: str = "density_match"
    calibration_bins: int = 10
    seed: int = 42
    num_workers: int = 0
    expected_latent_dim: Optional[int] = 50
    diagnostic_group_size: int = 16
    diagnostic_groups_per_time: int = 256
    database_source: Optional[str] = None
    database_version: Optional[str] = None
    database_commit: Optional[str] = None


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(values: np.ndarray) -> str:
    values = np.ascontiguousarray(values)
    digest = sha256()
    digest.update(str(values.dtype).encode("utf-8"))
    digest.update(json.dumps(list(values.shape)).encode("utf-8"))
    digest.update(values.tobytes())
    return digest.hexdigest()


def _sha256_strings(values: Iterable[Any]) -> str:
    digest = sha256()
    for value in values:
        encoded = str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"Non-finite value cannot be serialized: {value!r}")
        return value
    return str(value)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _resolve_database_column(
    frame: pd.DataFrame,
    candidates: Sequence[str],
    *,
    fallback_index: int,
) -> str:
    lower = {str(column).strip().lower(): str(column) for column in frame.columns}
    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    usable = [
        str(column)
        for column in frame.columns
        if not str(column).strip().lower().startswith("unnamed")
    ]
    if fallback_index < len(usable):
        return usable[fallback_index]
    raise ValueError(
        f"Could not resolve database column {candidates!r} from {list(frame.columns)!r}."
    )


def load_lr_database(path: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load an LR table and preserve ligand, receptor, pathway and annotation.

    Numeric CellChatDB CSV columns (``0``, ``1``, ``2``, ``3``) and descriptive
    column names are both supported.  Exact duplicate standardized rows are
    removed deterministically while their count is retained in the metadata.
    """

    path = Path(path).expanduser().resolve()
    table = pd.read_csv(path)
    ligand_col = _resolve_database_column(
        table, ("ligand", "ligand_symbol", "source", "gene_a", "0"), fallback_index=0
    )
    receptor_col = _resolve_database_column(
        table,
        ("receptor", "receptor_symbol", "target", "gene_b", "1"),
        fallback_index=1,
    )
    pathway_col = _resolve_database_column(
        table, ("pathway", "pathway_name", "signaling", "2"), fallback_index=2
    )
    annotation_col = _resolve_database_column(
        table,
        ("annotation", "category", "interaction_type", "3"),
        fallback_index=3,
    )
    standardized = pd.DataFrame(
        {
            "database_row": np.arange(len(table), dtype=np.int64),
            "ligand": table[ligand_col].astype(str).str.strip(),
            "receptor": table[receptor_col].astype(str).str.strip(),
            "pathway": table[pathway_col].astype(str).str.strip(),
            "annotation": table[annotation_col].astype(str).str.strip(),
        }
    )
    valid = np.ones(len(standardized), dtype=bool)
    for column in ("ligand", "receptor", "pathway", "annotation"):
        values = standardized[column]
        valid &= values.ne("") & values.str.lower().ne("nan")
    invalid_rows = int((~valid).sum())
    standardized = standardized.loc[valid].copy()
    before = len(standardized)
    standardized = standardized.drop_duplicates(
        subset=["ligand", "receptor", "pathway", "annotation"], keep="first"
    ).reset_index(drop=True)
    metadata = {
        "path": str(path),
        "sha256": _sha256_file(path),
        "columns": {
            "ligand": ligand_col,
            "receptor": receptor_col,
            "pathway": pathway_col,
            "annotation": annotation_col,
        },
        "rows_input": int(len(table)),
        "rows_invalid": invalid_rows,
        "rows_exact_duplicates": int(before - len(standardized)),
        "rows_standardized": int(len(standardized)),
    }
    return standardized, metadata


def complex_subunits(token: str) -> tuple[str, ...]:
    """Return strict underscore-delimited subunits for a CellChatDB token."""

    subunits = tuple(part.strip() for part in str(token).split("_") if part.strip())
    if not subunits:
        raise ValueError(f"Invalid empty ligand/receptor token: {token!r}")
    return subunits


def _validate_config(config: LREdgePriorConfig) -> None:
    if config.min_cells < 1:
        raise ValueError("min_cells must be at least one.")
    if not np.isfinite(config.target_sum) or config.target_sum <= 0:
        raise ValueError("target_sum must be positive and finite.")
    for name in ("activity_scale_quantile", "edge_score_quantile"):
        value = float(getattr(config, name))
        if not 0.0 < value < 1.0:
            raise ValueError(f"{name} must be strictly between zero and one.")
    fractions = np.asarray(
        [config.train_fraction, config.validation_fraction, config.test_fraction],
        dtype=float,
    )
    if np.any(fractions <= 0) or not np.isclose(fractions.sum(), 1.0, atol=1e-12):
        raise ValueError(
            "train/validation/test fractions must be positive and sum to one."
        )
    for name in (
        "train_pairs_per_time",
        "validation_pairs_per_time",
        "test_pairs_per_time",
        "pair_score_batch_size",
        "epochs",
        "batch_size",
        "patience",
        "calibration_bins",
        "diagnostic_group_size",
        "diagnostic_groups_per_time",
    ):
        if int(getattr(config, name)) < 1:
            raise ValueError(f"{name} must be at least one.")
    if config.decision_threshold_mode not in {"density_match", "f1"}:
        raise ValueError("decision_threshold_mode must be 'density_match' or 'f1'.")
    if config.candidate_radius is not None and (
        not np.isfinite(config.candidate_radius) or config.candidate_radius <= 0
    ):
        raise ValueError("candidate_radius must be positive and finite when provided.")


def _matrix_min_and_finite(x: Any) -> tuple[float, bool]:
    if sparse.issparse(x):
        data = np.asarray(x.data)
        return (float(data.min()) if data.size else 0.0, bool(np.isfinite(data).all()))
    values = np.asarray(x)
    return float(values.min()), bool(np.isfinite(values).all())


def _normalized_lr_expression(
    adata: ad.AnnData,
    *,
    gene_symbol_key: str,
    genes: Sequence[str],
    target_sum: float,
) -> tuple[np.ndarray, dict[str, int], dict[str, Any]]:
    """Extract LR genes after one normalize-total + log1p transform.

    Total-library scaling uses every input feature, not only LR genes.  The
    returned dense matrix has cells by the subset of requested genes present
    in ``adata.var[gene_symbol_key]``.
    """

    if gene_symbol_key not in adata.var:
        raise KeyError(f"Expected gene symbols in adata.var[{gene_symbol_key!r}].")
    symbols = adata.var[gene_symbol_key].astype(str).to_numpy()
    duplicated = pd.Series(symbols).duplicated(keep=False)
    if duplicated.any():
        examples = sorted(set(symbols[duplicated].tolist()))[:5]
        raise ValueError(
            f"adata.var[{gene_symbol_key!r}] must be unique; duplicates include {examples}."
        )
    minimum, finite = _matrix_min_and_finite(adata.X)
    if not finite or minimum < 0:
        raise ValueError("Input X must contain finite, non-negative linear expression.")
    row_sums = np.asarray(adata.X.sum(axis=1), dtype=np.float64).reshape(-1)
    if not np.isfinite(row_sums).all() or np.any(row_sums <= 0):
        raise ValueError("Every input cell must have a positive finite library size.")

    symbol_to_input = {symbol: index for index, symbol in enumerate(symbols)}
    present_genes = sorted(set(map(str, genes)).intersection(symbol_to_input))
    input_indices = np.asarray(
        [symbol_to_input[gene] for gene in present_genes], dtype=np.int64
    )
    subset = adata.X[:, input_indices]
    if sparse.issparse(subset):
        subset = subset.tocsr().astype(np.float32, copy=True)
        subset.eliminate_zeros()
        detected = np.asarray(subset.getnnz(axis=0), dtype=np.int64).reshape(-1)
        scale = (float(target_sum) / row_sums).astype(np.float32)
        subset = sparse.diags(scale).dot(subset).tocsr()
        np.log1p(subset.data, out=subset.data)
        expression = subset.toarray().astype(np.float32, copy=False)
    else:
        subset = np.asarray(subset, dtype=np.float32).copy()
        detected = np.count_nonzero(subset > 0, axis=0).astype(np.int64)
        subset *= (float(target_sum) / row_sums).astype(np.float32)[:, None]
        np.log1p(subset, out=subset)
        expression = subset
    if not np.isfinite(expression).all() or np.any(expression < 0):
        raise RuntimeError("Normalized LR expression contains invalid values.")
    gene_to_column = {gene: index for index, gene in enumerate(present_genes)}
    detection = {gene: int(detected[index]) for gene, index in gene_to_column.items()}
    metadata = {
        "input_semantics_required": "non-negative linear expression",
        "normalization": "normalize_total_then_log1p",
        "target_sum": float(target_sum),
        "normalize_total_applications": 1,
        "log1p_applications": 1,
        "library_sum_before": {
            "min": float(row_sums.min()),
            "median": float(np.median(row_sums)),
            "max": float(row_sums.max()),
        },
        "n_requested_lr_genes": int(len(set(map(str, genes)))),
        "n_present_lr_genes": int(len(present_genes)),
        "present_genes": present_genes,
    }
    return expression, detection, metadata


def _ordered_time_values(values: np.ndarray) -> list[Any]:
    unique = list(pd.unique(values))
    if any(pd.isna(value) for value in unique):
        raise ValueError("time_key contains missing values.")
    try:
        return sorted(unique)
    except TypeError:
        return sorted(unique, key=lambda value: str(value))


def stratified_cell_splits(
    times: Sequence[Any],
    *,
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[list[Any], dict[int, dict[str, np.ndarray]]]:
    """Make deterministic, within-time, cell-disjoint global-index splits."""

    fractions = np.asarray(
        [train_fraction, validation_fraction, test_fraction], dtype=float
    )
    if np.any(fractions <= 0) or not np.isclose(fractions.sum(), 1.0, atol=1e-12):
        raise ValueError("Split fractions must be positive and sum to one.")
    values = np.asarray(times)
    ordered = _ordered_time_values(values)
    rng = np.random.default_rng(int(seed))
    result: dict[int, dict[str, np.ndarray]] = {}
    for time_code, time_value in enumerate(ordered):
        indices = np.flatnonzero(values == time_value).astype(np.int64)
        permuted = rng.permutation(indices)
        n_train = int(math.floor(len(indices) * float(train_fraction)))
        n_val = int(math.floor(len(indices) * float(validation_fraction)))
        n_test = int(len(indices) - n_train - n_val)
        if min(n_train, n_val, n_test) < 2:
            raise ValueError(
                f"Time {time_value!r} needs at least two cells in every split; "
                f"got train/validation/test={n_train}/{n_val}/{n_test}."
            )
        result[time_code] = {
            "train": np.sort(permuted[:n_train]),
            "validation": np.sort(permuted[n_train : n_train + n_val]),
            "test": np.sort(permuted[n_train + n_val :]),
        }
    return ordered, result


def _sample_directed_pairs(
    cells: np.ndarray,
    n_requested: int,
    rng: np.random.Generator,
    *,
    latent: np.ndarray,
    candidate_radius: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Uniformly sample directed, non-self pairs inside the GNN radius.

    Small pools are enumerated exactly with a KD-tree.  Large pools use
    uniform rejection sampling over all directed non-self pairs; conditioning
    a uniform draw on the radius event remains uniform over valid candidates.
    The large-pool path fails closed if the candidate rate is too low for the
    bounded proposal budget instead of silently returning a biased sample.
    """

    cells = np.asarray(cells, dtype=np.int64)
    n_cells = int(len(cells))
    universe = n_cells * (n_cells - 1)
    if universe <= 0:
        raise ValueError("At least two cells are required for directed pair sampling.")

    if n_cells <= 2_000:
        coordinates = latent[cells]
        neighbor_lists = cKDTree(coordinates).query_ball_point(
            coordinates, r=float(candidate_radius)
        )
        sources: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        distances: list[np.ndarray] = []
        for local_source, neighbors in enumerate(neighbor_lists):
            local_targets = np.asarray(neighbors, dtype=np.int64)
            local_targets = local_targets[local_targets != local_source]
            if not len(local_targets):
                continue
            delta = coordinates[local_targets] - coordinates[local_source]
            local_distances = np.linalg.norm(delta, axis=1)
            valid = (local_distances < float(candidate_radius)) & (
                local_distances > 1.0e-6
            )
            if not valid.any():
                continue
            sources.append(np.full(int(valid.sum()), local_source, dtype=np.int64))
            targets.append(local_targets[valid])
            distances.append(local_distances[valid].astype(np.float32))
        if not sources:
            raise ValueError("No directed cell pairs fall inside candidate_radius.")
        local_source_all = np.concatenate(sources)
        local_target_all = np.concatenate(targets)
        distance_all = np.concatenate(distances)
        candidate_count = int(len(local_source_all))
        n_pairs = min(int(n_requested), candidate_count)
        chosen = rng.choice(candidate_count, size=n_pairs, replace=False, shuffle=False)
        source = cells[local_source_all[chosen]]
        target = cells[local_target_all[chosen]]
        distance = distance_all[chosen]
        metadata = {
            "sampler": "exact_kdtree_candidates_then_uniform_without_replacement",
            "directed_nonself_universe": int(universe),
            "valid_candidate_count": candidate_count,
            "requested": int(n_requested),
            "returned": int(n_pairs),
            "candidate_fraction": float(candidate_count / universe),
        }
        return source, target, distance, metadata

    target_count = min(int(n_requested), universe)
    accepted: dict[int, float] = {}
    proposed_unique: set[int] = set()
    max_proposals = min(universe, max(1_000_000, target_count * 100))
    while len(accepted) < target_count and len(proposed_unique) < max_proposals:
        remaining = target_count - len(accepted)
        batch_size = min(
            max(10_000, remaining * 3), max_proposals - len(proposed_unique)
        )
        if batch_size <= 0:
            break
        proposed = rng.integers(0, universe, size=batch_size, dtype=np.int64)
        proposed = np.unique(proposed)
        if proposed_unique:
            proposed = np.asarray(
                [value for value in proposed.tolist() if value not in proposed_unique],
                dtype=np.int64,
            )
        if not len(proposed):
            continue
        rng.shuffle(proposed)
        proposed_unique.update(proposed.tolist())
        local_source = proposed // (n_cells - 1)
        remainder = proposed % (n_cells - 1)
        local_target = remainder + (remainder >= local_source)
        delta = latent[cells[local_source]] - latent[cells[local_target]]
        distance = np.linalg.norm(delta, axis=1)
        valid = (distance < float(candidate_radius)) & (distance > 1.0e-6)
        for pair_id, pair_distance in zip(proposed[valid], distance[valid]):
            accepted[int(pair_id)] = float(pair_distance)
            if len(accepted) >= target_count:
                break
    if len(accepted) < target_count:
        raise RuntimeError(
            "Candidate-radius rejection sampling exhausted its unbiased proposal "
            f"budget: requested={target_count}, accepted={len(accepted)}, "
            f"unique_proposals={len(proposed_unique)}, universe={universe}. "
            "Reduce pairs_per_time or use a less restrictive GNN radius."
        )
    accepted_ids = np.fromiter(accepted.keys(), dtype=np.int64, count=target_count)
    accepted_distances = np.fromiter(
        accepted.values(), dtype=np.float32, count=target_count
    )
    local_source = accepted_ids // (n_cells - 1)
    remainder = accepted_ids % (n_cells - 1)
    local_target = remainder + (remainder >= local_source)
    source = cells[local_source]
    target = cells[local_target]
    metadata = {
        "sampler": "uniform_directed_rejection_without_replacement",
        "directed_nonself_universe": int(universe),
        "valid_candidate_count": None,
        "requested": int(n_requested),
        "returned": int(target_count),
        "unique_proposals": int(len(proposed_unique)),
        "observed_candidate_fraction": float(target_count / len(proposed_unique)),
    }
    return source, target, accepted_distances, metadata


def _complex_activity(
    token: str,
    *,
    expression: np.ndarray,
    gene_to_column: Mapping[str, int],
) -> np.ndarray:
    columns = [gene_to_column[gene] for gene in complex_subunits(token)]
    if len(columns) == 1:
        return expression[:, columns[0]]
    return np.min(expression[:, columns], axis=1)


def _score_pairs(
    source: np.ndarray,
    target: np.ndarray,
    ligand_activity: np.ndarray,
    receptor_activity: np.ndarray,
    pair_weights: np.ndarray,
    *,
    batch_size: int,
) -> np.ndarray:
    scores = np.empty(len(source), dtype=np.float32)
    weights = np.asarray(pair_weights, dtype=np.float32)
    for start in range(0, len(source), int(batch_size)):
        stop = min(start + int(batch_size), len(source))
        products = (
            ligand_activity[source[start:stop]] * receptor_activity[target[start:stop]]
        )
        scores[start:stop] = products @ weights
    return scores


def _safe_auc(labels: np.ndarray, probabilities: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        raise ValueError("Metric split contains only one edge class.")
    return float(roc_auc_score(labels, probabilities))


def _calibration_summary(
    labels: np.ndarray, probabilities: np.ndarray, *, bins: int
) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int8)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    records: list[dict[str, Any]] = []
    ece = 0.0
    for index in range(int(bins)):
        if index == int(bins) - 1:
            mask = (probabilities >= edges[index]) & (probabilities <= edges[index + 1])
        else:
            mask = (probabilities >= edges[index]) & (probabilities < edges[index + 1])
        count = int(mask.sum())
        mean_probability = float(probabilities[mask].mean()) if count else None
        positive_rate = float(labels[mask].mean()) if count else None
        if count:
            ece += (count / len(labels)) * abs(mean_probability - positive_rate)
        records.append(
            {
                "lower": float(edges[index]),
                "upper": float(edges[index + 1]),
                "count": count,
                "mean_probability": mean_probability,
                "positive_rate": positive_rate,
            }
        )
    return {
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "expected_calibration_error": float(ece),
        "bins": records,
    }


def _f1_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    if not len(thresholds):
        return 0.5
    denominator = precision[:-1] + recall[:-1]
    f1 = np.divide(
        2.0 * precision[:-1] * recall[:-1],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    return float(thresholds[int(np.argmax(f1))])


def _density_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    prevalence = float(np.asarray(labels).mean())
    try:
        return float(np.quantile(probabilities, 1.0 - prevalence, method="higher"))
    except TypeError:  # NumPy < 1.22
        return float(
            np.quantile(probabilities, 1.0 - prevalence, interpolation="higher")
        )


def _classification_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    threshold: float,
    calibration_bins: int,
) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int8)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predicted = (probabilities >= float(threshold)).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    precision = float(tp / (tp + fp)) if tp + fp else 0.0
    recall = float(tp / (tp + fn)) if tp + fn else 0.0
    f1 = (
        float(2 * precision * recall / (precision + recall))
        if precision + recall
        else 0.0
    )
    return {
        "n_pairs": int(len(labels)),
        "roc_auc": _safe_auc(labels, probabilities),
        "average_precision": float(average_precision_score(labels, probabilities)),
        "decision_threshold": float(threshold),
        "true_edge_rate": float(labels.mean()),
        "predicted_edge_rate": float(predicted.mean()),
        "accuracy": float((predicted == labels).mean()),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "calibration": _calibration_summary(
            labels, probabilities, bins=int(calibration_bins)
        ),
    }


def _predict_probabilities(
    model: nn.Module,
    features: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    result: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(features), int(batch_size)):
            stop = min(start + int(batch_size), len(features))
            values = torch.from_numpy(features[start:stop]).to(device)
            probabilities = torch.sigmoid(model(values)).reshape(-1)
            result.append(probabilities.cpu().numpy())
    return np.concatenate(result).astype(np.float32, copy=False)


def _group_graph_diagnostics(
    model: nn.Module,
    latent: np.ndarray,
    *,
    test_cells_by_time: Mapping[int, np.ndarray],
    time_values: Sequence[Any],
    candidate_radius: float,
    decision_threshold: float,
    group_size: int,
    groups_per_time: int,
    seed: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    """Measure realized predictor graphs in held-out random GNN-sized groups."""

    by_time: dict[str, Any] = {}
    aggregate = {
        "groups": 0,
        "nodes": 0,
        "candidate_edges": 0,
        "predicted_edges": 0,
        "zero_indegree_nodes": 0,
        "isolated_nodes": 0,
        "empty_groups": 0,
    }
    for time_code, time_value in enumerate(time_values):
        test_cells = np.asarray(test_cells_by_time[time_code], dtype=np.int64)
        if len(test_cells) < int(group_size):
            raise ValueError(
                f"Time {time_value!r} has {len(test_cells)} test cells, fewer than "
                f"diagnostic_group_size={group_size}."
            )
        grouping_seed = int(seed) + 10_000 + int(time_code)
        rng = np.random.default_rng(grouping_seed)
        group_records: list[tuple[np.ndarray, np.ndarray]] = []
        source_all: list[np.ndarray] = []
        target_all: list[np.ndarray] = []
        for _ in range(int(groups_per_time)):
            group = rng.choice(test_cells, size=int(group_size), replace=False)
            local_source, local_target = np.where(~np.eye(int(group_size), dtype=bool))
            source = group[local_source]
            target = group[local_target]
            distances = np.linalg.norm(latent[source] - latent[target], axis=1)
            valid = (distances < float(candidate_radius)) & (distances > 1.0e-6)
            source = source[valid]
            target = target[valid]
            group_records.append((source, target))
            if len(source):
                source_all.append(source)
                target_all.append(target)
        if source_all:
            concatenated_source = np.concatenate(source_all)
            concatenated_target = np.concatenate(target_all)
            candidate_features = _pair_features(
                latent, concatenated_source, concatenated_target
            )
            predicted_probabilities = _predict_probabilities(
                model,
                candidate_features,
                batch_size=int(batch_size),
                device=device,
            )
        else:
            predicted_probabilities = np.empty(0, dtype=np.float32)

        cursor = 0
        predicted_edges = 0
        candidate_edges = 0
        zero_indegree_nodes = 0
        isolated_nodes = 0
        empty_groups = 0
        for source, target in group_records:
            count = int(len(source))
            probabilities = predicted_probabilities[cursor : cursor + count]
            cursor += count
            connected = probabilities >= float(decision_threshold)
            connected_source = source[connected]
            connected_target = target[connected]
            # Convert the global cell IDs in this group to compact positions.
            group_nodes = (
                np.unique(np.concatenate((source, target)))
                if count
                else np.array([], dtype=np.int64)
            )
            if len(group_nodes) < int(group_size):
                # Candidate-radius isolation can remove all appearances of a
                # node. Recover the original group from the valid-pair record
                # is impossible, so count these missing nodes explicitly as
                # isolated/zero-indegree below.
                missing_nodes = int(group_size) - int(len(group_nodes))
            else:
                missing_nodes = 0
            if len(group_nodes):
                node_position = {
                    int(node): index for index, node in enumerate(group_nodes)
                }
                source_position = np.asarray(
                    [node_position[int(node)] for node in connected_source],
                    dtype=np.int64,
                )
                target_position = np.asarray(
                    [node_position[int(node)] for node in connected_target],
                    dtype=np.int64,
                )
                indegree = np.bincount(target_position, minlength=len(group_nodes))
                outdegree = np.bincount(source_position, minlength=len(group_nodes))
                zero_indegree_nodes += int((indegree == 0).sum()) + missing_nodes
                isolated_nodes += (
                    int(((indegree + outdegree) == 0).sum()) + missing_nodes
                )
            else:
                zero_indegree_nodes += int(group_size)
                isolated_nodes += int(group_size)
            n_connected = int(connected.sum())
            candidate_edges += count
            predicted_edges += n_connected
            empty_groups += int(n_connected == 0)
        n_groups = int(groups_per_time)
        n_nodes = n_groups * int(group_size)
        record = {
            "time_value": _json_scalar(time_value),
            "grouping_seed": grouping_seed,
            "n_groups": n_groups,
            "group_size": int(group_size),
            "candidate_radius": float(candidate_radius),
            "decision_threshold": float(decision_threshold),
            "candidate_edges": int(candidate_edges),
            "predicted_edges": int(predicted_edges),
            "mean_candidate_indegree": float(candidate_edges / n_nodes),
            "mean_indegree": float(predicted_edges / n_nodes),
            "zero_indegree_node_fraction": float(zero_indegree_nodes / n_nodes),
            "isolated_node_fraction": float(isolated_nodes / n_nodes),
            "empty_group_fraction": float(empty_groups / n_groups),
        }
        by_time[str(time_code)] = record
        aggregate["groups"] += n_groups
        aggregate["nodes"] += n_nodes
        aggregate["candidate_edges"] += candidate_edges
        aggregate["predicted_edges"] += predicted_edges
        aggregate["zero_indegree_nodes"] += zero_indegree_nodes
        aggregate["isolated_nodes"] += isolated_nodes
        aggregate["empty_groups"] += empty_groups
    return {
        "population": "held_out_test_cells_within_each_time",
        "directed": True,
        "self_edges": False,
        "sampling": "fixed-seed random groups without replacement within group",
        "by_time": by_time,
        "overall": {
            "n_groups": int(aggregate["groups"]),
            "group_size": int(group_size),
            "candidate_edges": int(aggregate["candidate_edges"]),
            "predicted_edges": int(aggregate["predicted_edges"]),
            "mean_candidate_indegree": float(
                aggregate["candidate_edges"] / aggregate["nodes"]
            ),
            "mean_indegree": float(aggregate["predicted_edges"] / aggregate["nodes"]),
            "zero_indegree_node_fraction": float(
                aggregate["zero_indegree_nodes"] / aggregate["nodes"]
            ),
            "isolated_node_fraction": float(
                aggregate["isolated_nodes"] / aggregate["nodes"]
            ),
            "empty_group_fraction": float(
                aggregate["empty_groups"] / aggregate["groups"]
            ),
        },
    }


def _pair_features(
    latent: np.ndarray, source: np.ndarray, target: np.ndarray
) -> np.ndarray:
    return np.concatenate((latent[source], latent[target]), axis=1).astype(
        np.float32, copy=False
    )


def _select_device(value: str | torch.device) -> torch.device:
    if isinstance(value, torch.device):
        return value
    requested = str(value).strip().lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {value}")
    return device


def build_lr_edge_prior(
    expression_h5ad: str | Path,
    latent_h5ad: str | Path,
    lr_database_csv: str | Path,
    output_dir: str | Path,
    *,
    config: Optional[LREdgePriorConfig] = None,
    device: str | torch.device = "auto",
    overwrite: bool = False,
    implementation_paths: Sequence[str | Path] = (),
) -> dict[str, Any]:
    """Build and train a provenance-complete directed LR edge prior.

    The saved ``link_predictor.pt`` is a raw ``LinkPredictorMLP.state_dict``
    and can therefore be supplied directly as ``edge_predictor_path`` to the
    existing GNN.  The manifest's ``recommended_edge_predictor_threshold`` is
    the corresponding validation-selected ``edge_predictor_thre`` value.
    """

    config = config or LREdgePriorConfig()
    _validate_config(config)
    expression_path = Path(expression_h5ad).expanduser().resolve()
    latent_path = Path(latent_h5ad).expanduser().resolve()
    database_path = Path(lr_database_csv).expanduser().resolve()
    output_path = Path(output_dir).expanduser().resolve()
    for path in (expression_path, latent_path, database_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    output_path.mkdir(parents=True, exist_ok=True)
    artifact_names = (
        "link_predictor.pt",
        "cell_splits.npz",
        "pair_samples.npz",
        "lr_pair_metadata.csv",
        "training_history.csv",
        "manifest.json",
    )
    existing = [
        output_path / name for name in artifact_names if (output_path / name).exists()
    ]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing LR-prior artifacts: "
            + ", ".join(str(path) for path in existing)
        )

    random.seed(int(config.seed))
    np.random.seed(int(config.seed))
    torch.manual_seed(int(config.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(config.seed))

    source = ad.read_h5ad(expression_path)
    latent_adata = ad.read_h5ad(latent_path)
    if source.n_obs != latent_adata.n_obs or not np.array_equal(
        source.obs_names.astype(str).to_numpy(),
        latent_adata.obs_names.astype(str).to_numpy(),
    ):
        raise ValueError(
            "Expression and latent AnnData rows/obs_names must align exactly."
        )
    if config.time_key not in source.obs:
        raise KeyError(f"time_key {config.time_key!r} is missing from expression obs.")
    if config.latent_key is None:
        latent = np.asarray(latent_adata.X, dtype=np.float32)
        latent_source = "X"
    else:
        if config.latent_key not in latent_adata.obsm:
            raise KeyError(
                f"latent_key {config.latent_key!r} is missing from latent obsm."
            )
        latent = np.asarray(latent_adata.obsm[config.latent_key], dtype=np.float32)
        latent_source = f"obsm[{config.latent_key!r}]"
    if latent.ndim != 2 or latent.shape[0] != source.n_obs:
        raise ValueError(f"Invalid latent shape: {latent.shape!r}.")
    if not np.isfinite(latent).all():
        raise ValueError("Latent representation contains non-finite values.")
    if config.expected_latent_dim is not None and latent.shape[1] != int(
        config.expected_latent_dim
    ):
        raise ValueError(
            f"Expected {config.expected_latent_dim} latent dimensions, got {latent.shape[1]}."
        )
    if config.candidate_radius is None:
        fit_params = latent_adata.uns.get("fit_params")
        if (
            not isinstance(fit_params, Mapping)
            or "interaction_cutoff" not in fit_params
        ):
            raise KeyError(
                "candidate_radius='auto' requires "
                "latent_adata.uns['fit_params']['interaction_cutoff']."
            )
        candidate_radius = float(fit_params["interaction_cutoff"])
        candidate_radius_source = "latent_adata.uns['fit_params']['interaction_cutoff']"
    else:
        candidate_radius = float(config.candidate_radius)
        candidate_radius_source = "explicit_config"
    if not np.isfinite(candidate_radius) or candidate_radius <= 0:
        raise ValueError(
            f"Resolved candidate radius must be positive and finite, got {candidate_radius!r}."
        )

    lr_table, database_metadata = load_lr_database(database_path)
    database_metadata["declared_provenance"] = {
        "source": config.database_source,
        "version": config.database_version,
        "commit": config.database_commit,
        "caller_supplied": True,
    }
    annotation_values = tuple(str(value) for value in config.annotation_filter)
    if annotation_values:
        annotation_mask = lr_table["annotation"].isin(annotation_values)
        rows_before_annotation = int(len(lr_table))
        lr_table = lr_table.loc[annotation_mask].reset_index(drop=True)
    else:
        rows_before_annotation = int(len(lr_table))
    if lr_table.empty:
        raise ValueError(
            f"No LR database rows remain after exact annotation filter {annotation_values!r}."
        )

    all_lr_genes = sorted(
        {
            gene
            for token in pd.concat((lr_table["ligand"], lr_table["receptor"]))
            for gene in complex_subunits(token)
        }
    )
    expression, detection, normalization_metadata = _normalized_lr_expression(
        source,
        gene_symbol_key=config.gene_symbol_key,
        genes=all_lr_genes,
        target_sum=float(config.target_sum),
    )
    gene_to_column = {
        gene: index
        for index, gene in enumerate(normalization_metadata["present_genes"])
    }

    filter_reason_counts: dict[str, int] = {
        "missing_subunit": 0,
        "subunit_below_min_cells": 0,
        "no_positive_train_complex_activity": 0,
    }
    prelim_rows: list[dict[str, Any]] = []
    for row in lr_table.itertuples(index=False):
        ligand_subunits = complex_subunits(row.ligand)
        receptor_subunits = complex_subunits(row.receptor)
        subunits = ligand_subunits + receptor_subunits
        if any(gene not in gene_to_column for gene in subunits):
            filter_reason_counts["missing_subunit"] += 1
            continue
        if any(detection[gene] < int(config.min_cells) for gene in subunits):
            filter_reason_counts["subunit_below_min_cells"] += 1
            continue
        prelim_rows.append(
            {
                "database_row": int(row.database_row),
                "ligand": str(row.ligand),
                "receptor": str(row.receptor),
                "pathway": str(row.pathway),
                "annotation": str(row.annotation),
                "ligand_subunits": "|".join(ligand_subunits),
                "receptor_subunits": "|".join(receptor_subunits),
                "min_subunit_detected_cells": int(min(detection[g] for g in subunits)),
            }
        )
    if not prelim_rows:
        raise ValueError("No LR pairs remain after strict global subunit filtering.")

    times = source.obs[config.time_key].to_numpy()
    time_values, cell_splits = stratified_cell_splits(
        times,
        train_fraction=float(config.train_fraction),
        validation_fraction=float(config.validation_fraction),
        test_fraction=float(config.test_fraction),
        seed=int(config.seed),
    )
    train_cells = np.concatenate(
        [cell_splits[code]["train"] for code in range(len(time_values))]
    )

    complex_cache: dict[str, np.ndarray] = {}
    ligand_columns: list[np.ndarray] = []
    receptor_columns: list[np.ndarray] = []
    final_rows: list[dict[str, Any]] = []
    for row in prelim_rows:
        ligand_token = str(row["ligand"])
        receptor_token = str(row["receptor"])
        if ligand_token not in complex_cache:
            complex_cache[ligand_token] = _complex_activity(
                ligand_token, expression=expression, gene_to_column=gene_to_column
            )
        if receptor_token not in complex_cache:
            complex_cache[receptor_token] = _complex_activity(
                receptor_token, expression=expression, gene_to_column=gene_to_column
            )
        ligand_raw = complex_cache[ligand_token]
        receptor_raw = complex_cache[receptor_token]
        ligand_positive = ligand_raw[train_cells]
        ligand_positive = ligand_positive[ligand_positive > 0]
        receptor_positive = receptor_raw[train_cells]
        receptor_positive = receptor_positive[receptor_positive > 0]
        if not len(ligand_positive) or not len(receptor_positive):
            filter_reason_counts["no_positive_train_complex_activity"] += 1
            continue
        ligand_scale = float(
            np.quantile(ligand_positive, float(config.activity_scale_quantile))
        )
        receptor_scale = float(
            np.quantile(receptor_positive, float(config.activity_scale_quantile))
        )
        if ligand_scale <= 0 or receptor_scale <= 0:
            filter_reason_counts["no_positive_train_complex_activity"] += 1
            continue
        ligand_columns.append(np.clip(ligand_raw / ligand_scale, 0.0, 1.0))
        receptor_columns.append(np.clip(receptor_raw / receptor_scale, 0.0, 1.0))
        final_row = dict(row)
        final_row.update(
            {
                "ligand_positive_train_cells": int(len(ligand_positive)),
                "receptor_positive_train_cells": int(len(receptor_positive)),
                "activity_scale_quantile": float(config.activity_scale_quantile),
                "ligand_train_positive_activity_scale": ligand_scale,
                "receptor_train_positive_activity_scale": receptor_scale,
            }
        )
        final_rows.append(final_row)
    if not final_rows:
        raise ValueError(
            "No LR pairs have positive ligand and receptor activity in training cells."
        )
    ligand_activity = np.column_stack(ligand_columns).astype(np.float32, copy=False)
    receptor_activity = np.column_stack(receptor_columns).astype(np.float32, copy=False)
    final_table = pd.DataFrame(final_rows)
    pathway_counts = final_table["pathway"].value_counts().to_dict()
    n_pathways = int(len(pathway_counts))
    final_table["pathway_pair_count"] = (
        final_table["pathway"].map(pathway_counts).astype(int)
    )
    final_table["pair_weight"] = final_table["pathway"].map(
        lambda pathway: 1.0 / (n_pathways * pathway_counts[pathway])
    )
    pair_weights = final_table["pair_weight"].to_numpy(dtype=np.float32)
    if not np.isclose(pair_weights.sum(), 1.0, rtol=1e-6, atol=1e-7):
        raise RuntimeError("Pathway-balanced LR weights do not sum to one.")

    split_code_map = {"train": 0, "validation": 1, "test": 2}
    pair_arrays: dict[str, list[np.ndarray]] = {
        "source": [],
        "target": [],
        "time_code": [],
        "split_code": [],
        "distance": [],
        "score": [],
    }
    score_thresholds: dict[int, float] = {}
    candidate_sampling_metadata: dict[str, dict[str, Any]] = {}
    rng = np.random.default_rng(int(config.seed) + 1)
    requested_by_split = {
        "train": int(config.train_pairs_per_time),
        "validation": int(config.validation_pairs_per_time),
        "test": int(config.test_pairs_per_time),
    }
    for time_code in range(len(time_values)):
        time_pairs: dict[
            str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ] = {}
        candidate_sampling_metadata[str(time_code)] = {}
        for split_name in ("train", "validation", "test"):
            (
                source_indices,
                target_indices,
                distances,
                sampling_metadata,
            ) = _sample_directed_pairs(
                cell_splits[time_code][split_name],
                requested_by_split[split_name],
                rng,
                latent=latent,
                candidate_radius=candidate_radius,
            )
            scores = _score_pairs(
                source_indices,
                target_indices,
                ligand_activity,
                receptor_activity,
                pair_weights,
                batch_size=int(config.pair_score_batch_size),
            )
            time_pairs[split_name] = (
                source_indices,
                target_indices,
                distances,
                scores,
            )
            candidate_sampling_metadata[str(time_code)][split_name] = sampling_metadata
        threshold = float(
            np.quantile(time_pairs["train"][3], float(config.edge_score_quantile))
        )
        train_labels = time_pairs["train"][3] > threshold
        if not train_labels.any() or train_labels.all():
            raise ValueError(
                f"Training LR scores at time {time_values[time_code]!r} do not "
                "yield both classes under the configured strict-quantile rule."
            )
        score_thresholds[time_code] = threshold
        for split_name, (
            source_indices,
            target_indices,
            distances,
            scores,
        ) in time_pairs.items():
            pair_arrays["source"].append(source_indices)
            pair_arrays["target"].append(target_indices)
            pair_arrays["time_code"].append(
                np.full(len(scores), time_code, dtype=np.int16)
            )
            pair_arrays["split_code"].append(
                np.full(len(scores), split_code_map[split_name], dtype=np.int8)
            )
            pair_arrays["distance"].append(distances.astype(np.float32, copy=False))
            pair_arrays["score"].append(scores)

    packed = {key: np.concatenate(values) for key, values in pair_arrays.items()}
    thresholds_array = np.asarray(
        [score_thresholds[int(code)] for code in packed["time_code"]],
        dtype=np.float32,
    )
    packed["label"] = (packed["score"] > thresholds_array).astype(np.int8)
    for split_code, split_name in enumerate(("train", "validation", "test")):
        labels = packed["label"][packed["split_code"] == split_code]
        if len(np.unique(labels)) < 2:
            raise ValueError(
                f"The combined {split_name} pair sample has only one class."
            )

    cell_split_path = output_path / "cell_splits.npz"
    cell_split_payload = {
        f"time_{time_code}_{split_name}": indices
        for time_code, splits in cell_splits.items()
        for split_name, indices in splits.items()
    }
    np.savez_compressed(cell_split_path, **cell_split_payload)
    pair_sample_path = output_path / "pair_samples.npz"
    np.savez_compressed(pair_sample_path, **packed)
    lr_metadata_path = output_path / "lr_pair_metadata.csv"
    final_table.to_csv(lr_metadata_path, index=False)

    # LR expression is no longer needed once immutable scores and labels are saved.
    del expression, complex_cache, ligand_columns, receptor_columns
    del ligand_activity, receptor_activity

    selected_device = _select_device(device)
    split_masks = {
        name: packed["split_code"] == code for name, code in split_code_map.items()
    }
    feature_arrays = {
        name: _pair_features(latent, packed["source"][mask], packed["target"][mask])
        for name, mask in split_masks.items()
    }
    label_arrays = {
        name: packed["label"][mask].astype(np.float32, copy=False)
        for name, mask in split_masks.items()
    }
    train_dataset = TensorDataset(
        torch.from_numpy(feature_arrays["train"]),
        torch.from_numpy(label_arrays["train"]),
    )
    loader_generator = torch.Generator()
    loader_generator.manual_seed(int(config.seed))
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config.batch_size),
        shuffle=True,
        num_workers=int(config.num_workers),
        pin_memory=selected_device.type == "cuda",
        generator=loader_generator,
    )
    model = LinkPredictorMLP(input_dim=int(latent.shape[1]) * 2).to(selected_device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
    )
    criterion = nn.BCEWithLogitsLoss()
    best_ap = -np.inf
    best_epoch = 0
    best_state: Optional[dict[str, torch.Tensor]] = None
    no_improvement = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, int(config.epochs) + 1):
        model.train()
        total_loss = 0.0
        n_seen = 0
        for features_batch, labels_batch in train_loader:
            features_batch = features_batch.to(selected_device, non_blocking=True)
            labels_batch = labels_batch.to(
                selected_device, non_blocking=True
            ).unsqueeze(1)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features_batch)
            loss = criterion(logits, labels_batch)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(labels_batch)
            n_seen += int(len(labels_batch))
        val_probabilities = _predict_probabilities(
            model,
            feature_arrays["validation"],
            batch_size=int(config.batch_size),
            device=selected_device,
        )
        val_auc = _safe_auc(label_arrays["validation"], val_probabilities)
        val_ap = float(
            average_precision_score(label_arrays["validation"], val_probabilities)
        )
        val_brier = float(
            brier_score_loss(label_arrays["validation"], val_probabilities)
        )
        history.append(
            {
                "epoch": epoch,
                "train_bce": total_loss / max(n_seen, 1),
                "validation_roc_auc": val_auc,
                "validation_average_precision": val_ap,
                "validation_brier_score": val_brier,
            }
        )
        if val_ap > best_ap + float(config.min_delta):
            best_ap = val_ap
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            no_improvement = 0
        else:
            no_improvement += 1
            if no_improvement >= int(config.patience):
                break
    if best_state is None:
        raise RuntimeError(
            "Edge predictor training did not produce a valid checkpoint."
        )
    model.load_state_dict(best_state)
    model.to(selected_device)
    model_path = output_path / "link_predictor.pt"
    torch.save(best_state, model_path)
    history_path = output_path / "training_history.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)

    probabilities = {
        name: _predict_probabilities(
            model,
            feature_arrays[name],
            batch_size=int(config.batch_size),
            device=selected_device,
        )
        for name in ("train", "validation", "test")
    }
    validation_f1_threshold = _f1_threshold(
        label_arrays["validation"], probabilities["validation"]
    )
    validation_density_threshold = _density_threshold(
        label_arrays["validation"], probabilities["validation"]
    )
    if config.decision_threshold_mode == "density_match":
        selected_threshold = validation_density_threshold
    else:
        selected_threshold = validation_f1_threshold
    metrics = {
        name: _classification_metrics(
            label_arrays[name],
            probabilities[name],
            threshold=selected_threshold,
            calibration_bins=int(config.calibration_bins),
        )
        for name in ("train", "validation", "test")
    }
    metrics["test_by_time"] = {}
    test_mask = split_masks["test"]
    test_time_codes = packed["time_code"][test_mask]
    for time_code, time_value in enumerate(time_values):
        local_mask = test_time_codes == time_code
        metrics["test_by_time"][str(time_code)] = {
            "time_value": _json_scalar(time_value),
            **_classification_metrics(
                label_arrays["test"][local_mask],
                probabilities["test"][local_mask],
                threshold=selected_threshold,
                calibration_bins=int(config.calibration_bins),
            ),
        }
    group_graph_diagnostics = _group_graph_diagnostics(
        model,
        latent,
        test_cells_by_time={
            time_code: cell_splits[time_code]["test"]
            for time_code in range(len(time_values))
        },
        time_values=time_values,
        candidate_radius=candidate_radius,
        decision_threshold=selected_threshold,
        group_size=int(config.diagnostic_group_size),
        groups_per_time=int(config.diagnostic_groups_per_time),
        seed=int(config.seed),
        batch_size=int(config.batch_size),
        device=selected_device,
    )

    cell_split_manifest: dict[str, Any] = {}
    for time_code, time_value in enumerate(time_values):
        cell_split_manifest[str(time_code)] = {
            "time_value": _json_scalar(time_value),
            "splits": {
                split_name: {
                    "n_cells": int(len(indices)),
                    "global_indices_sha256": _sha256_array(indices),
                }
                for split_name, indices in cell_splits[time_code].items()
            },
        }
    pair_split_manifest: dict[str, Any] = {}
    for split_name, split_code in split_code_map.items():
        mask = packed["split_code"] == split_code
        pair_split_manifest[split_name] = {
            "n_pairs": int(mask.sum()),
            "source_indices_sha256": _sha256_array(packed["source"][mask]),
            "target_indices_sha256": _sha256_array(packed["target"][mask]),
            "distances_sha256": _sha256_array(packed["distance"][mask]),
            "scores_sha256": _sha256_array(packed["score"][mask]),
            "labels_sha256": _sha256_array(packed["label"][mask]),
            "edge_rate": float(packed["label"][mask].mean()),
            "distance_min": float(packed["distance"][mask].min()),
            "distance_max": float(packed["distance"][mask].max()),
        }

    implementation_files = {
        str(Path(__file__).resolve()): _sha256_file(Path(__file__).resolve()),
        str(
            Path(
                __import__(LinkPredictorMLP.__module__, fromlist=["x"]).__file__
            ).resolve()
        ): _sha256_file(
            Path(
                __import__(LinkPredictorMLP.__module__, fromlist=["x"]).__file__
            ).resolve()
        ),
    }
    for path_value in implementation_paths:
        implementation_path = Path(path_value).expanduser().resolve()
        implementation_files[str(implementation_path)] = _sha256_file(
            implementation_path
        )

    artifacts = {}
    for artifact_path in (
        model_path,
        cell_split_path,
        pair_sample_path,
        lr_metadata_path,
        history_path,
    ):
        artifacts[artifact_path.name] = {
            "path": str(artifact_path),
            "sha256": _sha256_file(artifact_path),
            "bytes": int(artifact_path.stat().st_size),
        }
    manifest = {
        "schema_version": 1,
        "method": "nonspatial_cellchatdb_expression_guided_directed_lr_edge_prior",
        "spatial_method_claimed": False,
        "formula": {
            "complex_activity": (
                "min(log1p(normalize_total(X, "
                f"target_sum={float(config.target_sum):g}))) over all strict subunits"
            ),
            "activity_scaling": (
                "clip(activity / "
                f"q{100 * float(config.activity_scale_quantile):g}"
                "(activity>0 on training cells), 0, 1)"
            ),
            "pair_score": "mean_over_pathways(mean_over_lr_rows_in_pathway(scaled_ligand_sender * scaled_receptor_receiver))",
            "edge_label": (
                "score > within-time sampled-training score "
                f"q{100 * float(config.edge_score_quantile):g}"
            ),
            "candidate_pair": "0 < Euclidean PCA distance < resolved GNN interaction_cutoff",
            "rationale": (
                "Secreted signaling avoids treating ECM/contact entries as observed physical "
                "contacts in non-spatial data; the global unsupervised min_cells filter retains "
                "rare signals while removing unsupported subunits; pathway balancing prevents "
                "large database pathways from dominating; the configured upper-quantile edge "
                "rate is chosen for useful connectivity in the configured GNN group size."
            ),
        },
        "configuration": asdict(config),
        "random_seed": int(config.seed),
        "device": str(selected_device),
        "inputs": {
            "expression_h5ad": {
                "path": str(expression_path),
                "sha256": _sha256_file(expression_path),
                "shape": [int(source.n_obs), int(source.n_vars)],
                "obs_names_sha256": _sha256_strings(source.obs_names.astype(str)),
            },
            "latent_h5ad": {
                "path": str(latent_path),
                "sha256": _sha256_file(latent_path),
                "shape": [int(latent.shape[0]), int(latent.shape[1])],
                "source": latent_source,
                "obs_names_sha256": _sha256_strings(latent_adata.obs_names.astype(str)),
            },
            "lr_database": database_metadata,
        },
        "data_usage": {
            "obs_keys_used": [config.time_key],
            "gene_symbol_source": f"var[{config.gene_symbol_key!r}]",
            "uses_clone": False,
            "uses_fate_or_cell_type": False,
            "uses_starting_population": False,
            "uses_spatial_or_visualization_coordinates": False,
        },
        "normalization": normalization_metadata,
        "database_filtering": {
            "annotation_match": "exact",
            "annotation_filter": list(annotation_values),
            "rows_before_annotation_filter": rows_before_annotation,
            "rows_after_annotation_filter": int(len(lr_table)),
            "strict_all_subunits": True,
            "complex_delimiter": "_",
            "complex_activity": "minimum",
            "min_cells_global_per_subunit": int(config.min_cells),
            "min_cells_scope": "all cells before cell split",
            "min_cells_filter_unsupervised": True,
            "min_cells_filter_uses_only_nonzero_expression_detection": True,
            "min_cells_filter_uses_outcomes_or_annotations": False,
            "filter_reason_counts": filter_reason_counts,
            "rows_after_global_subunit_filter": int(len(prelim_rows)),
            "rows_final": int(len(final_table)),
            "pathways_final": n_pathways,
            "pathway_counts": {
                str(key): int(value) for key, value in sorted(pathway_counts.items())
            },
        },
        "cell_splits": cell_split_manifest,
        "pair_sampling": {
            "directed": True,
            "self_edges": False,
            "candidate_radius": float(candidate_radius),
            "candidate_radius_source": candidate_radius_source,
            "candidate_rule": "distance < radius and distance > 1e-6",
            "candidate_feature_space": latent_source,
            "uniform_without_replacement_within_split_and_time": True,
            "split_code_map": split_code_map,
            "requested_pairs_per_time": requested_by_split,
            "candidate_sampling_by_time_and_split": candidate_sampling_metadata,
            "splits": pair_split_manifest,
        },
        "lr_score_thresholds": {
            str(time_code): {
                "time_value": _json_scalar(time_values[time_code]),
                "training_quantile": float(config.edge_score_quantile),
                "strict_greater_than": True,
                "threshold": float(score_thresholds[time_code]),
            }
            for time_code in range(len(time_values))
        },
        "predictor": {
            "architecture": "LinkPredictorMLP",
            "implementation_module": LinkPredictorMLP.__module__,
            "input_order": (
                f"concatenate(sender_{latent.shape[1]}d_pca, "
                f"receiver_{latent.shape[1]}d_pca)"
            ),
            "input_dim": int(latent.shape[1]) * 2,
            "hidden_dim": 256,
            "state_dict_compatible_with_gnn_interaction": True,
            "loss": "unweighted BCEWithLogitsLoss",
            "early_stopping_metric": "validation_average_precision",
            "best_epoch": int(best_epoch),
            "best_validation_average_precision": float(best_ap),
            "decision_threshold_mode": config.decision_threshold_mode,
            "validation_f1_threshold": float(validation_f1_threshold),
            "validation_density_matched_threshold": float(validation_density_threshold),
            "recommended_edge_predictor_threshold": float(selected_threshold),
        },
        "metrics": metrics,
        "group_graph_diagnostics": group_graph_diagnostics,
        "implementation_files_sha256": implementation_files,
        "versions": {
            "anndata": ad.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
        },
        "artifacts": artifacts,
    }
    manifest_path = output_path / "manifest.json"
    _atomic_write_json(manifest_path, manifest)
    return manifest


__all__ = [
    "LREdgePriorConfig",
    "build_lr_edge_prior",
    "complex_subunits",
    "load_lr_database",
    "stratified_cell_splits",
]
