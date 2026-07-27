"""Axis-specific, fixed-cohort computational perturbations.

This module provides the reusable mechanics used by the zebrafish
``cxcl12a -> cxcr4a`` reviewer analysis.  A gene knockdown is represented in
the fitted PCA state exactly as it would be by the original PCA transform:

``delta_state = delta_expression @ pca_loadings``.

The rollout is intentionally deterministic and identity preserving.  It uses
the public velocity-component evaluator, does not apply growth/resampling, and
requires ``sigma=0``.  Baseline and counterfactual runs can therefore use the
same cells and the same spatial-GNN grouping seed.  Interaction mediation is
estimated by comparing interaction-on and interaction-off rollouts of the same
trained model; no model is retrained or replaced.

Attention, link-predictor probabilities, and exact complete edge messages are
kept as separate quantities.  In particular, this module never constructs an
``attention * ligand/receptor expression`` score.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .ablation import compute_virtual_ablation_metrics
from .simulation import compute_velocity_components
from .spatial_interaction_attribution import (
    decompose_spatial_gnn_group,
    make_interaction_groups,
)

__all__ = [
    "CompleteMessageAudit",
    "CounterfactualMetricResult",
    "FixedCohortRollout",
    "GeneCounterfactualResult",
    "ProjectedGeneEdit",
    "apply_projected_gene_knockdowns",
    "audit_spatial_complete_messages",
    "compute_counterfactual_metrics",
    "compute_fixed_lr_target_message_metrics",
    "compute_interaction_mediation_metrics",
    "deterministic_fixed_cohort_rollout",
    "match_hvg_sham_genes",
    "run_gene_counterfactual",
    "select_fixed_receiver_cohort",
    "validate_pca_model_visibility",
]


@dataclass(frozen=True)
class ProjectedGeneEdit:
    """A gene-space knockdown projected into the retained PCA state."""

    points: np.ndarray
    delta_state: np.ndarray
    gene_table: pd.DataFrame


@dataclass(frozen=True)
class FixedCohortRollout:
    """A deterministic trajectory whose row identities never change."""

    times: np.ndarray
    points: np.ndarray
    interaction_enabled: bool
    grouping_seed: int
    sigma: float


@dataclass(frozen=True)
class CompleteMessageAudit:
    """Exact one-layer spatial-GNN output and its non-collapsed diagnostics."""

    output: np.ndarray
    baseline: np.ndarray
    edge_output: np.ndarray
    attention_signed: np.ndarray
    edge_table: pd.DataFrame
    reconstruction_table: pd.DataFrame
    groups: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class CounterfactualMetricResult:
    """Distributional and message-alignment summaries for one comparison."""

    distribution: pd.DataFrame
    alignment: pd.DataFrame


@dataclass(frozen=True)
class GeneCounterfactualResult:
    """All matched outputs for one gene perturbation and dose."""

    edit: ProjectedGeneEdit
    baseline_audit: CompleteMessageAudit
    counterfactual_audit: CompleteMessageAudit
    baseline_on: FixedCohortRollout
    counterfactual_on: FixedCohortRollout
    baseline_off: FixedCohortRollout
    counterfactual_off: FixedCohortRollout
    metrics_on: CounterfactualMetricResult
    metrics_off: CounterfactualMetricResult
    mediation: pd.DataFrame


def _as_feature_names(feature_names: Sequence[object]) -> tuple[str, ...]:
    names = tuple(str(value) for value in feature_names)
    if not names:
        raise ValueError("feature_names must be non-empty.")
    folded = [name.casefold() for name in names]
    duplicates = sorted(name for name, count in Counter(folded).items() if count > 1)
    if duplicates:
        raise ValueError(
            "feature_names must be unique under case-insensitive matching; "
            f"duplicates={duplicates[:5]}."
        )
    return names


def _feature_index(feature_names: Sequence[object], gene: str) -> int:
    names = _as_feature_names(feature_names)
    target = str(gene).casefold()
    matches = [index for index, name in enumerate(names) if name.casefold() == target]
    if len(matches) != 1:
        raise KeyError(
            f"Gene {gene!r} must resolve to exactly one feature; observed {len(matches)}."
        )
    return int(matches[0])


def _expression_column(expression, index: int) -> np.ndarray:
    column = expression[:, int(index)]
    if hasattr(column, "toarray"):
        column = column.toarray()
    values = np.asarray(column, dtype=np.float64).reshape(-1)
    if not np.isfinite(values).all():
        raise ValueError(f"Expression column {index} contains non-finite values.")
    return values


def _expression_shape(expression) -> tuple[int, int]:
    shape = getattr(expression, "shape", None)
    if shape is None or len(shape) != 2:
        raise ValueError("expression must be a two-dimensional matrix.")
    return int(shape[0]), int(shape[1])


def _spaces(feature_dim: int, spatial_dim: int) -> dict[str, slice]:
    feature_dim = int(feature_dim)
    spatial_dim = int(spatial_dim)
    if spatial_dim < 0 or spatial_dim > feature_dim:
        raise ValueError(
            f"spatial_dim must be in [0, {feature_dim}], got {spatial_dim}."
        )
    result = {"joint": slice(0, feature_dim)}
    if spatial_dim:
        result["spatial"] = slice(0, spatial_dim)
    if spatial_dim < feature_dim:
        result["state"] = slice(spatial_dim, feature_dim)
    return result


def _cosine_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    numerator = np.einsum("ij,ij->i", left, right)
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    result = np.full(left.shape[0], np.nan, dtype=np.float64)
    valid = denominator > 1e-12
    result[valid] = numerator[valid] / denominator[valid]
    return result


def _cosine_vectors(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1e-12:
        return float("nan")
    return float(np.dot(left, right) / denominator)


def validate_pca_model_visibility(
    feature_names: Sequence[object],
    loadings: np.ndarray,
    genes: Sequence[str],
    *,
    highly_variable: Optional[Sequence[bool]] = None,
    loading_tolerance: float = 1e-10,
) -> pd.DataFrame:
    """Fail closed unless every requested gene is an active model input.

    A gene is model-visible when it is an HVG (when the HVG mask is supplied)
    and has a non-zero loading in the retained PCA components.
    """

    names = _as_feature_names(feature_names)
    values = np.asarray(loadings, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != len(names) or values.shape[1] == 0:
        raise ValueError(
            "loadings must have shape (len(feature_names), n_components), "
            "with at least one component."
        )
    if not np.isfinite(values).all():
        raise ValueError("PCA loadings contain non-finite values.")
    hvg = None
    if highly_variable is not None:
        hvg = np.asarray(highly_variable, dtype=bool).reshape(-1)
        if hvg.shape != (len(names),):
            raise ValueError(
                f"highly_variable must have shape ({len(names)},), got {hvg.shape}."
            )
    tolerance = float(loading_tolerance)
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("loading_tolerance must be finite and non-negative.")

    rows: list[dict[str, object]] = []
    for gene in genes:
        index = _feature_index(names, gene)
        norm = float(np.linalg.norm(values[index]))
        is_hvg = bool(hvg[index]) if hvg is not None else None
        active = norm > tolerance
        if hvg is not None and not is_hvg:
            raise ValueError(f"Gene {gene!r} is not marked highly variable.")
        if not active:
            raise ValueError(
                f"Gene {gene!r} has PCA-loading norm {norm:.6g}, which is not "
                f"above tolerance {tolerance:.6g}."
            )
        rows.append(
            {
                "gene": names[index],
                "feature_index": index,
                "highly_variable": is_hvg,
                "pca_loading_norm": norm,
                "loading_tolerance": tolerance,
                "model_visible": True,
            }
        )
    return pd.DataFrame(rows)


def apply_projected_gene_knockdowns(
    points: np.ndarray,
    expression,
    feature_names: Sequence[object],
    loadings: np.ndarray,
    genes: Sequence[str],
    fractions: float | Mapping[str, float],
    *,
    spatial_dim: int = 2,
    cell_mask: Optional[Sequence[bool]] = None,
) -> ProjectedGeneEdit:
    """Scale selected genes and project the edit into retained PCA coordinates.

    ``expression`` must be in the same feature scale on which PCA was fitted.
    Spatial coordinates are copied unchanged.  A fraction of one is a complete
    in-silico knockdown; a fraction of zero is rejected because it is not a
    perturbation.
    """

    baseline = np.asarray(points, dtype=np.float64)
    if baseline.ndim != 2 or baseline.shape[0] == 0:
        raise ValueError("points must be a non-empty two-dimensional matrix.")
    if not np.isfinite(baseline).all():
        raise ValueError("points contain non-finite values.")
    n_cells, n_features = _expression_shape(expression)
    if n_cells != baseline.shape[0]:
        raise ValueError(
            f"expression has {n_cells} rows, expected {baseline.shape[0]}."
        )
    names = _as_feature_names(feature_names)
    if n_features != len(names):
        raise ValueError(f"expression has {n_features} columns, expected {len(names)}.")
    loading_values = np.asarray(loadings, dtype=np.float64)
    if loading_values.ndim != 2 or loading_values.shape[0] != len(names):
        raise ValueError("loadings must have shape (len(feature_names), n_components).")
    n_components = int(loading_values.shape[1])
    if baseline.shape[1] != int(spatial_dim) + n_components:
        raise ValueError(
            "points must contain exactly spatial_dim coordinates followed by "
            f"{n_components} PCA components; got {baseline.shape[1]} columns."
        )
    if not np.isfinite(loading_values).all():
        raise ValueError("loadings contain non-finite values.")
    requested = tuple(str(gene) for gene in genes)
    if not requested or len({gene.casefold() for gene in requested}) != len(requested):
        raise ValueError("genes must contain unique gene names.")

    if cell_mask is None:
        selected = np.ones(n_cells, dtype=bool)
    else:
        selected = np.asarray(cell_mask, dtype=bool).reshape(-1)
        if selected.shape != (n_cells,):
            raise ValueError(
                f"cell_mask must have shape ({n_cells},), got {selected.shape}."
            )
    if isinstance(fractions, Mapping):
        fraction_map = {
            str(key).casefold(): float(value) for key, value in fractions.items()
        }
    else:
        fraction_map = {gene.casefold(): float(fractions) for gene in requested}

    delta_state = np.zeros((n_cells, n_components), dtype=np.float64)
    rows: list[dict[str, object]] = []
    for gene in requested:
        folded = gene.casefold()
        if folded not in fraction_map:
            raise KeyError(f"No knockdown fraction was supplied for {gene!r}.")
        fraction = float(fraction_map[folded])
        if not np.isfinite(fraction) or fraction <= 0 or fraction > 1:
            raise ValueError(
                f"Knockdown fraction for {gene!r} must be in (0, 1], got {fraction}."
            )
        index = _feature_index(names, gene)
        gene_expression = _expression_column(expression, index)
        expression_delta = np.zeros(n_cells, dtype=np.float64)
        expression_delta[selected] = -fraction * gene_expression[selected]
        gene_delta = expression_delta[:, None] * loading_values[index][None, :]
        delta_state += gene_delta
        rows.append(
            {
                "gene": names[index],
                "feature_index": index,
                "knockdown_fraction": fraction,
                "n_selected_cells": int(selected.sum()),
                "n_expression_positive_selected": int(
                    np.sum(selected & (gene_expression > 0))
                ),
                "baseline_expression_mean_selected": (
                    float(np.mean(gene_expression[selected]))
                    if np.any(selected)
                    else float("nan")
                ),
                "pca_loading_norm": float(np.linalg.norm(loading_values[index])),
                "projected_delta_frobenius_norm": float(np.linalg.norm(gene_delta)),
            }
        )

    perturbed = baseline.copy()
    perturbed[:, int(spatial_dim) :] += delta_state
    return ProjectedGeneEdit(
        points=perturbed.astype(np.float32),
        delta_state=delta_state.astype(np.float32),
        gene_table=pd.DataFrame(rows),
    )


def select_fixed_receiver_cohort(
    expression,
    feature_names: Sequence[object],
    *,
    ligand: str,
    receptor: str,
    positive_threshold: float = 0.0,
) -> np.ndarray:
    """Select the baseline receptor-positive, ligand-negative receiver cohort."""

    ligand_values = _expression_column(
        expression, _feature_index(feature_names, ligand)
    )
    receptor_values = _expression_column(
        expression, _feature_index(feature_names, receptor)
    )
    threshold = float(positive_threshold)
    if not np.isfinite(threshold):
        raise ValueError("positive_threshold must be finite.")
    mask = (receptor_values > threshold) & (ligand_values <= threshold)
    if not np.any(mask):
        raise ValueError(
            "The fixed receptor-positive/ligand-negative receiver cohort is empty."
        )
    return mask


def match_hvg_sham_genes(
    expression,
    feature_names: Sequence[object],
    loadings: np.ndarray,
    highly_variable: Sequence[bool],
    *,
    target_gene: str,
    n_shams: int = 100,
    exclude_genes: Sequence[str] = (),
    detection_threshold: float = 0.0,
    loading_tolerance: float = 1e-10,
    cell_mask: Optional[Sequence[bool]] = None,
) -> pd.DataFrame:
    """Choose deterministic nearest HVG shams.

    Matching uses detection fraction, mean expression, and retained-PCA
    loading norm after standardizing each covariate over eligible HVGs.
    ``cell_mask`` can bind expression covariates to a predeclared anchor and
    cell compartment; PCA-loading covariates remain properties of the fitted
    model.  The full loading-vector cosine is reported as a diagnostic but is
    not used to manufacture a directionally aligned null.
    """

    names = _as_feature_names(feature_names)
    n_cells, n_features = _expression_shape(expression)
    if n_cells < 1 or n_features != len(names):
        raise ValueError("expression shape is not aligned to feature_names.")
    values = np.asarray(loadings, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != len(names):
        raise ValueError("loadings are not aligned to feature_names.")
    hvg = np.asarray(highly_variable, dtype=bool).reshape(-1)
    if hvg.shape != (len(names),):
        raise ValueError("highly_variable is not aligned to feature_names.")
    n_shams = int(n_shams)
    if n_shams < 1:
        raise ValueError("n_shams must be positive.")
    if cell_mask is None:
        matching_expression = expression
        n_matching_cells = n_cells
    else:
        selected = np.asarray(cell_mask, dtype=bool).reshape(-1)
        if selected.shape != (n_cells,):
            raise ValueError(
                f"cell_mask must have shape ({n_cells},), got {selected.shape}."
            )
        if not np.any(selected):
            raise ValueError("cell_mask must select at least one matching cell.")
        matching_expression = expression[selected]
        n_matching_cells = int(selected.sum())

    threshold = float(detection_threshold)
    if hasattr(matching_expression, "toarray"):
        detected = np.asarray((matching_expression > threshold).mean(axis=0)).reshape(
            -1
        )
        means = np.asarray(matching_expression.mean(axis=0)).reshape(-1)
    else:
        dense = np.asarray(matching_expression, dtype=np.float64)
        detected = np.mean(dense > threshold, axis=0)
        means = np.mean(dense, axis=0)
    loading_norm = np.linalg.norm(values, axis=1)
    finite = np.isfinite(detected) & np.isfinite(means) & np.isfinite(loading_norm)
    eligible = hvg & finite & (loading_norm > float(loading_tolerance))

    target_index = _feature_index(names, target_gene)
    if not eligible[target_index]:
        raise ValueError(
            f"Target gene {target_gene!r} is not an eligible model-visible HVG."
        )
    excluded = {str(gene).casefold() for gene in exclude_genes}
    excluded.add(str(target_gene).casefold())
    for index, name in enumerate(names):
        if name.casefold() in excluded:
            eligible[index] = False
    candidates = np.flatnonzero(eligible)
    if candidates.size < n_shams:
        raise ValueError(
            f"Requested {n_shams} shams but only {candidates.size} eligible HVGs remain."
        )

    covariates = np.column_stack(
        (
            detected,
            np.log1p(np.maximum(means, 0.0)),
            np.log(np.maximum(loading_norm, float(loading_tolerance))),
        )
    )
    reference = np.flatnonzero(hvg & finite & (loading_norm > float(loading_tolerance)))
    scale = np.std(covariates[reference], axis=0, ddof=0)
    scale[~np.isfinite(scale) | (scale <= 1e-12)] = 1.0
    delta = (covariates[candidates] - covariates[target_index]) / scale
    distance = np.linalg.norm(delta, axis=1)
    order = np.lexsort(
        (np.asarray([names[index].casefold() for index in candidates]), distance)
    )
    chosen = candidates[order[:n_shams]]
    chosen_distance = distance[order[:n_shams]]

    target_loading = values[target_index]
    target_norm = loading_norm[target_index]
    cosine = (
        values[chosen]
        @ target_loading
        / np.maximum(loading_norm[chosen] * target_norm, 1e-12)
    )
    return pd.DataFrame(
        {
            "sham_rank": np.arange(1, n_shams + 1, dtype=int),
            "gene": [names[index] for index in chosen],
            "feature_index": chosen.astype(int),
            "match_distance": chosen_distance,
            "detection_fraction": detected[chosen],
            "target_detection_fraction": detected[target_index],
            "mean_expression": means[chosen],
            "target_mean_expression": means[target_index],
            "pca_loading_norm": loading_norm[chosen],
            "target_pca_loading_norm": target_norm,
            "loading_cosine_to_target": cosine,
            "n_matching_cells": n_matching_cells,
            "matched_covariates": "detection_fraction;mean_expression;pca_loading_norm",
        }
    )


def deterministic_fixed_cohort_rollout(
    points: np.ndarray,
    model,
    *,
    start_time: float,
    end_time: float,
    dt: float,
    interaction_m: int,
    grouping_seed: int,
    device: str = "cpu",
    spatial_dim: int = 2,
    interaction_enabled: bool = True,
    sigma: float = 0.0,
) -> FixedCohortRollout:
    """Euler-roll a fixed cohort with deterministic diffusion ``sigma=0``."""

    import torch

    initial = np.asarray(points, dtype=np.float32)
    if initial.ndim != 2 or initial.shape[0] < 2:
        raise ValueError("points must be an N x D matrix with N >= 2.")
    if not np.isfinite(initial).all():
        raise ValueError("points contain non-finite values.")
    start = float(start_time)
    end = float(end_time)
    step_limit = float(dt)
    if not np.isfinite(start) or not np.isfinite(end) or end <= start:
        raise ValueError("end_time must be finite and greater than start_time.")
    if not np.isfinite(step_limit) or step_limit <= 0:
        raise ValueError("dt must be positive and finite.")
    if float(sigma) != 0.0:
        raise ValueError(
            "Fixed-cohort counterfactual rollout requires sigma=0 exactly."
        )
    n_steps = int(math.ceil((end - start) / step_limit))
    step = (end - start) / float(n_steps)

    np.random.seed(int(grouping_seed))
    torch.manual_seed(int(grouping_seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(grouping_seed))
    was_training = bool(getattr(model, "training", False))
    if hasattr(model, "to"):
        model.to(device)
    if hasattr(model, "eval"):
        model.eval()

    frames = [initial.copy()]
    times = [start]
    current = initial.copy()
    try:
        for step_index in range(n_steps):
            time_value = start + step_index * step
            components = compute_velocity_components(
                current,
                time_value,
                model,
                interaction_m=int(interaction_m),
                device=str(device),
                spatial_dim=int(spatial_dim),
                include_interaction=bool(interaction_enabled),
            )
            velocity = (
                np.asarray(components["full"], dtype=np.float32)
                if interaction_enabled
                else (
                    np.asarray(components["drift"], dtype=np.float32)
                    + np.asarray(components["score"], dtype=np.float32)
                )
            )
            if velocity.shape != current.shape or not np.isfinite(velocity).all():
                raise RuntimeError(
                    "Model velocity is non-finite or not aligned to the fixed cohort."
                )
            current = current + np.float32(step) * velocity
            frames.append(current.copy())
            times.append(start + (step_index + 1) * step)
    finally:
        if hasattr(model, "train"):
            model.train(was_training)

    return FixedCohortRollout(
        times=np.asarray(times, dtype=np.float64),
        points=np.stack(frames).astype(np.float32, copy=False),
        interaction_enabled=bool(interaction_enabled),
        grouping_seed=int(grouping_seed),
        sigma=0.0,
    )


def audit_spatial_complete_messages(
    interaction_net,
    points: np.ndarray,
    *,
    time_value: float,
    group_size: int,
    grouping_seed: int,
    device: str = "cpu",
    spatial_dim: int = 2,
) -> CompleteMessageAudit:
    """Recompute predictor gates, attention, and exact complete messages.

    The edge table reports these as separate diagnostics.  Exact messages are
    the only edge quantities that add up to the official interaction output.
    """

    import torch

    values = np.asarray(points, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("points must be an N x D matrix with N >= 2.")
    if not np.isfinite(values).all():
        raise ValueError("points contain non-finite values.")
    groups = make_interaction_groups(
        values.shape[0], int(group_size), random_state=int(grouping_seed)
    )
    if hasattr(interaction_net, "to"):
        interaction_net.to(device)
    tensor = torch.as_tensor(values, device=device, dtype=torch.float32)
    output = np.zeros_like(values, dtype=np.float32)
    receiver_baseline = np.zeros_like(values, dtype=np.float32)
    edge_outputs: list[np.ndarray] = []
    attentions: list[np.ndarray] = []
    edge_frames: list[pd.DataFrame] = []
    reconstruction_rows: list[dict[str, object]] = []

    for group_index, indices in enumerate(groups):
        index_tensor = torch.as_tensor(indices, device=device, dtype=torch.long)
        group_lnw = torch.full(
            (indices.size, 1),
            -math.log(float(indices.size)),
            device=device,
            dtype=torch.float32,
        )
        result = decompose_spatial_gnn_group(
            interaction_net,
            tensor[index_tensor],
            group_lnw,
            torch.tensor(float(time_value), device=device, dtype=torch.float32),
        )
        group_output = result.output.detach().cpu().numpy().astype(np.float32)
        group_baseline = result.baseline.detach().cpu().numpy().astype(np.float32)
        output[indices] = group_output
        receiver_baseline[indices] = group_baseline

        local_edges = result.edge_index.detach().cpu().numpy()
        source = indices[local_edges[0]]
        target = indices[local_edges[1]]
        edge_output = result.edge_output.detach().cpu().numpy().astype(np.float32)
        attention = result.attention_signed.detach().cpu().numpy().astype(np.float32)
        predictor = (
            result.edge_predictor_probability.detach().cpu().numpy().astype(float)
        )
        attention_abs = result.attention_abs_mean.detach().cpu().numpy().astype(float)
        distance = result.edge_distance.detach().cpu().numpy().astype(float)
        mass_fraction = result.source_mass_fraction.detach().cpu().numpy().astype(float)
        edge_frames.append(
            pd.DataFrame(
                {
                    "grouping_seed": int(grouping_seed),
                    "group_index": int(group_index),
                    "source_index": source.astype(int),
                    "target_index": target.astype(int),
                    "edge_predictor_probability": predictor,
                    "attention_abs_mean": attention_abs,
                    "source_mass_fraction": mass_fraction,
                    "spatial_distance": distance,
                    "complete_message_norm_joint": np.linalg.norm(edge_output, axis=1),
                    "complete_message_norm_spatial": np.linalg.norm(
                        edge_output[:, : int(spatial_dim)], axis=1
                    ),
                    "complete_message_norm_state": np.linalg.norm(
                        edge_output[:, int(spatial_dim) :], axis=1
                    ),
                }
            )
        )
        edge_outputs.append(edge_output)
        attentions.append(attention)
        reconstruction_rows.append(
            {
                "grouping_seed": int(grouping_seed),
                "group_index": int(group_index),
                "n_cells": int(indices.size),
                "n_edges": int(local_edges.shape[1]),
                "max_abs_residual": float(result.max_abs_residual),
                "relative_l2_residual": float(result.relative_l2_residual),
            }
        )

    n_heads = int(getattr(interaction_net.gnn_layers[0], "num_heads"))
    return CompleteMessageAudit(
        output=output,
        baseline=receiver_baseline,
        edge_output=(
            np.concatenate(edge_outputs, axis=0)
            if edge_outputs
            else np.empty((0, values.shape[1]), dtype=np.float32)
        ),
        attention_signed=(
            np.concatenate(attentions, axis=0)
            if attentions
            else np.empty((0, n_heads), dtype=np.float32)
        ),
        edge_table=(
            pd.concat(edge_frames, ignore_index=True) if edge_frames else pd.DataFrame()
        ),
        reconstruction_table=pd.DataFrame(reconstruction_rows),
        groups=groups,
    )


def _validate_rollout_pair(
    baseline: FixedCohortRollout,
    counterfactual: FixedCohortRollout,
) -> None:
    if baseline.points.shape != counterfactual.points.shape:
        raise ValueError(
            "Baseline and counterfactual trajectories must have identical shapes."
        )
    if not np.array_equal(baseline.times, counterfactual.times):
        raise ValueError(
            "Baseline and counterfactual trajectories must use identical times."
        )
    if baseline.grouping_seed != counterfactual.grouping_seed:
        raise ValueError(
            "Baseline and counterfactual trajectories must use the same grouping seed."
        )
    if baseline.interaction_enabled != counterfactual.interaction_enabled:
        raise ValueError("A direct comparison must use the same interaction setting.")


def compute_fixed_lr_target_message_metrics(
    baseline: CompleteMessageAudit,
    counterfactual: CompleteMessageAudit,
    *,
    ligand_positive_mask: Sequence[bool],
    receiver_mask: Sequence[bool],
    spatial_dim: int = 2,
    counterfactual_points: Optional[np.ndarray] = None,
    interaction_net=None,
    device: str = "cpu",
) -> pd.DataFrame:
    """Measure generic complete GNN messages on LR-conditioned fixed support.

    The fixed support contains baseline graph edges whose sender is
    ligand-positive and whose receiver belongs to the baseline
    receptor-positive/ligand-negative cohort.  For each fixed receiver, exact
    complete messages are summed over that support and then normed.  ``D_target``
    is the mean receiver norm, including zero for every fixed receiver without
    a retained edge.  A support edge absent after perturbation contributes a
    zero message; newly appearing edges are recorded but excluded.  The
    ligand/receptor expressions define support only: the complete GNN message
    itself is generic and is not an LR-specific model-output component.

    Predictor probabilities and attention gates are diagnostics, not factors
    in ``D_target``.  Counterfactual predictor probabilities can be recomputed
    for every fixed support pair, including gated-out pairs, when
    ``counterfactual_points`` and ``interaction_net`` are supplied.  Attention
    is undefined for a gated-out edge and is therefore summarized only over
    retained fixed-support edges.
    """

    n_cells, feature_dim = baseline.output.shape
    if counterfactual.output.shape != (n_cells, feature_dim):
        raise ValueError("Baseline and counterfactual audits are not cell aligned.")
    if len(baseline.groups) != len(counterfactual.groups) or any(
        not np.array_equal(left, right)
        for left, right in zip(baseline.groups, counterfactual.groups)
    ):
        raise ValueError(
            "Baseline and counterfactual audits must use identical fixed groups."
        )
    ligand = np.asarray(ligand_positive_mask, dtype=bool).reshape(-1)
    receiver = np.asarray(receiver_mask, dtype=bool).reshape(-1)
    if ligand.shape != (n_cells,) or receiver.shape != (n_cells,):
        raise ValueError("Ligand and receiver masks must align to audited cells.")
    if not np.any(ligand) or not np.any(receiver):
        raise ValueError("Fixed LR target support requires senders and receivers.")
    if len(baseline.edge_table) != baseline.edge_output.shape[0]:
        raise ValueError("Baseline edge table and exact edge outputs are not aligned.")
    if len(counterfactual.edge_table) != counterfactual.edge_output.shape[0]:
        raise ValueError(
            "Counterfactual edge table and exact edge outputs are not aligned."
        )

    keys = ["group_index", "source_index", "target_index"]
    baseline_edges = baseline.edge_table.reset_index(drop=True).copy()
    counterfactual_edges = counterfactual.edge_table.reset_index(drop=True).copy()
    support_mask = (
        ligand[baseline_edges["source_index"].to_numpy(dtype=int)]
        & receiver[baseline_edges["target_index"].to_numpy(dtype=int)]
    )
    support_positions = np.flatnonzero(support_mask)
    support = baseline_edges.loc[support_mask].copy()
    if support.empty:
        raise ValueError(
            "No baseline graph edge connects a ligand-positive sender to the "
            "fixed receiver cohort."
        )
    support_index = pd.MultiIndex.from_frame(support[keys])
    counterfactual_index = pd.MultiIndex.from_frame(counterfactual_edges[keys])
    lookup = pd.Series(
        np.arange(len(counterfactual_edges), dtype=int),
        index=counterfactual_index,
    )
    retained = support_index.isin(counterfactual_index)
    retained_positions = np.full(len(support), -1, dtype=int)
    if np.any(retained):
        retained_positions[retained] = lookup.loc[support_index[retained]].to_numpy(
            dtype=int
        )

    baseline_messages = baseline.edge_output[support_positions].astype(
        np.float64, copy=False
    )
    counterfactual_messages = np.zeros_like(baseline_messages)
    counterfactual_messages[retained] = counterfactual.edge_output[
        retained_positions[retained]
    ]
    targets = support["target_index"].to_numpy(dtype=int)
    receiver_indices = np.flatnonzero(receiver)
    receiver_lookup = np.full(n_cells, -1, dtype=int)
    receiver_lookup[receiver_indices] = np.arange(receiver_indices.size)
    target_local = receiver_lookup[targets]

    baseline_predictor = support["edge_predictor_probability"].to_numpy(dtype=float)
    counterfactual_predictor = None
    if (counterfactual_points is None) != (interaction_net is None):
        raise ValueError(
            "counterfactual_points and interaction_net must be supplied together."
        )
    if counterfactual_points is not None:
        import torch

        points = np.asarray(counterfactual_points, dtype=np.float32)
        if points.shape != (n_cells, feature_dim):
            raise ValueError("counterfactual_points are not aligned to the audit.")
        source = support["source_index"].to_numpy(dtype=int)
        target = support["target_index"].to_numpy(dtype=int)
        tensor = torch.as_tensor(points, device=device, dtype=torch.float32)
        pair = torch.cat((tensor[source], tensor[target]), dim=1)
        was_training = bool(getattr(interaction_net, "training", False))
        if hasattr(interaction_net, "to"):
            interaction_net.to(device)
        if hasattr(interaction_net, "eval"):
            interaction_net.eval()
        try:
            with torch.no_grad():
                counterfactual_predictor = (
                    torch.sigmoid(interaction_net.link_predictor(pair))
                    .reshape(-1)
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(float)
                )
        finally:
            if hasattr(interaction_net, "train"):
                interaction_net.train(was_training)
    elif np.any(retained):
        counterfactual_predictor = np.full(len(support), np.nan, dtype=float)
        counterfactual_predictor[retained] = counterfactual_edges.iloc[
            retained_positions[retained]
        ]["edge_predictor_probability"].to_numpy(dtype=float)

    counterfactual_target_mask = (
        ligand[counterfactual_edges["source_index"].to_numpy(dtype=int)]
        & receiver[counterfactual_edges["target_index"].to_numpy(dtype=int)]
    )
    counterfactual_target_index = pd.MultiIndex.from_frame(
        counterfactual_edges.loc[counterfactual_target_mask, keys]
    )
    added = counterfactual_target_index.difference(support_index)
    baseline_attention = support["attention_abs_mean"].to_numpy(dtype=float)
    retained_attention = (
        counterfactual_edges.iloc[retained_positions[retained]][
            "attention_abs_mean"
        ].to_numpy(dtype=float)
        if np.any(retained)
        else np.empty(0, dtype=float)
    )

    rows: list[dict[str, object]] = []
    for space, columns in _spaces(feature_dim, int(spatial_dim)).items():
        baseline_sum = np.zeros(
            (receiver_indices.size, baseline_messages[:, columns].shape[1]),
            dtype=np.float64,
        )
        counterfactual_sum = np.zeros_like(baseline_sum)
        np.add.at(baseline_sum, target_local, baseline_messages[:, columns])
        np.add.at(
            counterfactual_sum,
            target_local,
            counterfactual_messages[:, columns],
        )
        baseline_norm = np.linalg.norm(baseline_sum, axis=1)
        counterfactual_norm = np.linalg.norm(counterfactual_sum, axis=1)
        baseline_d = float(np.mean(baseline_norm))
        counterfactual_d = float(np.mean(counterfactual_norm))
        rows.append(
            {
                "space": space,
                "n_fixed_receivers": int(receiver_indices.size),
                "n_ligand_positive_senders": int(ligand.sum()),
                "n_baseline_fixed_support_edges": int(len(support)),
                "n_counterfactual_retained_support_edges": int(retained.sum()),
                "n_counterfactual_missing_support_edges_zero_filled": int(
                    (~retained).sum()
                ),
                "n_counterfactual_added_target_edges_excluded": int(len(added)),
                "fixed_support_retained_fraction": float(np.mean(retained)),
                "baseline_D_target": baseline_d,
                "counterfactual_D_target": counterfactual_d,
                "delta_D_target": counterfactual_d - baseline_d,
                "absolute_delta_D_target": abs(counterfactual_d - baseline_d),
                "ratio_D_target": (
                    counterfactual_d / baseline_d
                    if baseline_d > 1e-12
                    else float("nan")
                ),
                "baseline_receiver_target_norm_mean": baseline_d,
                "counterfactual_receiver_target_norm_mean": counterfactual_d,
                "baseline_receiver_target_norm_median": float(np.median(baseline_norm)),
                "counterfactual_receiver_target_norm_median": float(
                    np.median(counterfactual_norm)
                ),
                "baseline_edge_predictor_probability_mean": float(
                    np.mean(baseline_predictor)
                ),
                "counterfactual_fixed_support_predictor_probability_mean": (
                    float(np.nanmean(counterfactual_predictor))
                    if counterfactual_predictor is not None
                    and np.any(np.isfinite(counterfactual_predictor))
                    else float("nan")
                ),
                "baseline_attention_abs_mean": float(np.mean(baseline_attention)),
                "counterfactual_attention_abs_mean_retained_edges": (
                    float(np.mean(retained_attention))
                    if retained_attention.size
                    else float("nan")
                ),
                "attention_missing_edges_treated_as_zero": False,
                "complete_message_missing_edges_treated_as_zero": True,
                "support_policy": (
                    "baseline ligand-positive sender to fixed "
                    "receptor-positive/ligand-negative receiver edges"
                ),
                "message_semantics": (
                    "generic complete GNN message on expression-conditioned "
                    "fixed support; not an LR-specific message component"
                ),
            }
        )
    return pd.DataFrame(rows)


def compute_counterfactual_metrics(
    baseline: FixedCohortRollout,
    counterfactual: FixedCohortRollout,
    *,
    receiver_mask: Sequence[bool],
    message_delta: Optional[np.ndarray] = None,
    spatial_dim: int = 2,
    max_ot_points: Optional[int] = 1024,
    random_seed: int = 42,
) -> CounterfactualMetricResult:
    """Compute endpoint W1/W2/centroid shift and optional message alignment.

    OT capping is identity paired because fixed-cohort rows retain cell
    identity.  ``message_delta`` yields a diagnostic geometric alignment only;
    callers must separately establish whether its interaction grouping plan is
    identical to the rollout's grouping plan before treating it as a
    rollout-driving field.
    """

    _validate_rollout_pair(baseline, counterfactual)
    n_cells = int(baseline.points.shape[1])
    mask = np.asarray(receiver_mask, dtype=bool).reshape(-1)
    if mask.shape != (n_cells,) or not np.any(mask):
        raise ValueError("receiver_mask must select at least one aligned cell.")
    endpoint_time = float(baseline.times[-1])
    baseline_end = np.asarray(baseline.points[-1], dtype=np.float64)
    counterfactual_end = np.asarray(counterfactual.points[-1], dtype=np.float64)

    distribution_frames: list[pd.DataFrame] = []
    cohort_masks = {
        "all_cells": np.ones(n_cells, dtype=bool),
        "fixed_receptor_positive_ligand_negative": mask,
    }
    for cohort, cohort_mask in cohort_masks.items():
        table = compute_virtual_ablation_metrics(
            np.asarray([baseline_end[cohort_mask]], dtype=object),
            {
                "counterfactual": np.asarray(
                    [counterfactual_end[cohort_mask]], dtype=object
                )
            },
            [endpoint_time],
            spatial_dim=int(spatial_dim),
            max_ot_points=max_ot_points,
            random_seed=int(random_seed),
            paired_ot_support=True,
        )
        table["space"] = table["space"].replace({"latent": "state"})
        table["wasserstein_scale_contract"] = table["space"].map(
            {
                "joint": (
                    "scale-dependent mixture of spatial coordinates and PCA "
                    "state; descriptive, not cross-space comparable"
                ),
                "spatial": "spatial-coordinate scale; primary space",
                "state": "PCA-state scale; primary space",
            }
        )
        table.insert(0, "cohort", cohort)
        table.insert(1, "interaction_enabled", baseline.interaction_enabled)
        distribution_frames.append(table)

    alignment_rows: list[dict[str, object]] = []
    if message_delta is not None:
        message = np.asarray(message_delta, dtype=np.float64)
        if message.shape != baseline_end.shape or not np.isfinite(message).all():
            raise ValueError(
                "message_delta must be finite and aligned to endpoint points."
            )
        endpoint_delta = counterfactual_end - baseline_end
        for cohort, cohort_mask in cohort_masks.items():
            for space, columns in _spaces(
                baseline_end.shape[1], int(spatial_dim)
            ).items():
                endpoint_values = endpoint_delta[cohort_mask, columns]
                message_values = message[cohort_mask, columns]
                cosine = _cosine_rows(endpoint_values, message_values)
                valid = np.isfinite(cosine)
                alignment_rows.append(
                    {
                        "cohort": cohort,
                        "interaction_enabled": baseline.interaction_enabled,
                        "space": space,
                        "n_cells": int(cohort_mask.sum()),
                        "n_nonzero_pairs": int(valid.sum()),
                        "mean_cellwise_cosine": (
                            float(np.mean(cosine[valid]))
                            if np.any(valid)
                            else float("nan")
                        ),
                        "median_cellwise_cosine": (
                            float(np.median(cosine[valid]))
                            if np.any(valid)
                            else float("nan")
                        ),
                        "fraction_positive_cosine": (
                            float(np.mean(cosine[valid] > 0))
                            if np.any(valid)
                            else float("nan")
                        ),
                        "centroid_cosine": _cosine_vectors(
                            np.mean(endpoint_values, axis=0),
                            np.mean(message_values, axis=0),
                        ),
                        "endpoint_delta_centroid_norm": float(
                            np.linalg.norm(np.mean(endpoint_values, axis=0))
                        ),
                        "anchor_message_delta_centroid_norm": float(
                            np.linalg.norm(np.mean(message_values, axis=0))
                        ),
                    }
                )

    return CounterfactualMetricResult(
        distribution=pd.concat(distribution_frames, ignore_index=True),
        alignment=pd.DataFrame(alignment_rows),
    )


def compute_interaction_mediation_metrics(
    baseline_on: FixedCohortRollout,
    counterfactual_on: FixedCohortRollout,
    baseline_off: FixedCohortRollout,
    counterfactual_off: FixedCohortRollout,
    *,
    receiver_mask: Sequence[bool],
    spatial_dim: int = 2,
) -> pd.DataFrame:
    """Difference-in-differences mediation control for the same trained model."""

    _validate_rollout_pair(baseline_on, counterfactual_on)
    _validate_rollout_pair(baseline_off, counterfactual_off)
    if not baseline_on.interaction_enabled or baseline_off.interaction_enabled:
        raise ValueError("Expected an interaction-on pair and an interaction-off pair.")
    if baseline_on.points.shape != baseline_off.points.shape:
        raise ValueError(
            "Interaction-on/off runs must retain the same cells and times."
        )
    if not np.array_equal(baseline_on.times, baseline_off.times):
        raise ValueError("Interaction-on/off runs must use identical time grids.")

    n_cells = int(baseline_on.points.shape[1])
    receiver = np.asarray(receiver_mask, dtype=bool).reshape(-1)
    if receiver.shape != (n_cells,) or not np.any(receiver):
        raise ValueError("receiver_mask must select at least one aligned cell.")
    effect_on = np.asarray(counterfactual_on.points[-1], dtype=np.float64) - np.asarray(
        baseline_on.points[-1], dtype=np.float64
    )
    effect_off = np.asarray(
        counterfactual_off.points[-1], dtype=np.float64
    ) - np.asarray(baseline_off.points[-1], dtype=np.float64)
    mediated = effect_on - effect_off
    rows: list[dict[str, object]] = []
    for cohort, mask in {
        "all_cells": np.ones(n_cells, dtype=bool),
        "fixed_receptor_positive_ligand_negative": receiver,
    }.items():
        for space, columns in _spaces(effect_on.shape[1], int(spatial_dim)).items():
            on_values = effect_on[mask, columns]
            off_values = effect_off[mask, columns]
            mediated_values = mediated[mask, columns]
            on_centroid = np.mean(on_values, axis=0)
            off_centroid = np.mean(off_values, axis=0)
            mediated_centroid = np.mean(mediated_values, axis=0)
            on_norm = float(np.linalg.norm(on_centroid))
            mediated_norm = float(np.linalg.norm(mediated_centroid))
            rows.append(
                {
                    "cohort": cohort,
                    "space": space,
                    "n_cells": int(mask.sum()),
                    "interaction_on_effect_centroid_norm": on_norm,
                    "interaction_off_effect_centroid_norm": float(
                        np.linalg.norm(off_centroid)
                    ),
                    "interaction_mediated_centroid_norm": mediated_norm,
                    "mediated_to_on_norm_ratio": (
                        mediated_norm / on_norm if on_norm > 1e-12 else float("nan")
                    ),
                    "mediated_to_on_centroid_cosine": _cosine_vectors(
                        mediated_centroid, on_centroid
                    ),
                    "mean_cellwise_mediated_norm": float(
                        np.mean(np.linalg.norm(mediated_values, axis=1))
                    ),
                    "mean_cellwise_on_effect_norm": float(
                        np.mean(np.linalg.norm(on_values, axis=1))
                    ),
                }
            )
    return pd.DataFrame(rows)


def run_gene_counterfactual(
    points: np.ndarray,
    expression,
    feature_names: Sequence[object],
    loadings: np.ndarray,
    model,
    *,
    genes: Sequence[str],
    fraction: float,
    receiver_mask: Sequence[bool],
    start_time: float,
    end_time: float,
    dt: float,
    interaction_m: int,
    grouping_seed: int,
    device: str = "cpu",
    spatial_dim: int = 2,
    max_ot_points: Optional[int] = 1024,
    metric_seed: int = 42,
    cell_mask: Optional[Sequence[bool]] = None,
    baseline_audit: Optional[CompleteMessageAudit] = None,
    baseline_on: Optional[FixedCohortRollout] = None,
    baseline_off: Optional[FixedCohortRollout] = None,
) -> GeneCounterfactualResult:
    """Run one matched gene/dose counterfactual with mediation controls."""

    edit = apply_projected_gene_knockdowns(
        points,
        expression,
        feature_names,
        loadings,
        genes,
        fraction,
        spatial_dim=int(spatial_dim),
        cell_mask=cell_mask,
    )
    interaction_net = getattr(model, "interaction_net", None)
    if interaction_net is None:
        raise ValueError("model must expose interaction_net.")
    if baseline_audit is None:
        baseline_audit = audit_spatial_complete_messages(
            interaction_net,
            points,
            time_value=float(start_time),
            group_size=int(interaction_m),
            grouping_seed=int(grouping_seed),
            device=str(device),
            spatial_dim=int(spatial_dim),
        )
    counterfactual_audit = audit_spatial_complete_messages(
        interaction_net,
        edit.points,
        time_value=float(start_time),
        group_size=int(interaction_m),
        grouping_seed=int(grouping_seed),
        device=str(device),
        spatial_dim=int(spatial_dim),
    )
    rollout_kwargs = {
        "model": model,
        "start_time": float(start_time),
        "end_time": float(end_time),
        "dt": float(dt),
        "interaction_m": int(interaction_m),
        "grouping_seed": int(grouping_seed),
        "device": str(device),
        "spatial_dim": int(spatial_dim),
        "sigma": 0.0,
    }
    if baseline_on is None:
        baseline_on = deterministic_fixed_cohort_rollout(
            points, interaction_enabled=True, **rollout_kwargs
        )
    if baseline_off is None:
        baseline_off = deterministic_fixed_cohort_rollout(
            points, interaction_enabled=False, **rollout_kwargs
        )
    counterfactual_on = deterministic_fixed_cohort_rollout(
        edit.points, interaction_enabled=True, **rollout_kwargs
    )
    counterfactual_off = deterministic_fixed_cohort_rollout(
        edit.points, interaction_enabled=False, **rollout_kwargs
    )
    message_delta = counterfactual_audit.output - baseline_audit.output
    metrics_on = compute_counterfactual_metrics(
        baseline_on,
        counterfactual_on,
        receiver_mask=receiver_mask,
        message_delta=message_delta,
        spatial_dim=int(spatial_dim),
        max_ot_points=max_ot_points,
        random_seed=int(metric_seed),
    )
    metrics_off = compute_counterfactual_metrics(
        baseline_off,
        counterfactual_off,
        receiver_mask=receiver_mask,
        message_delta=None,
        spatial_dim=int(spatial_dim),
        max_ot_points=max_ot_points,
        random_seed=int(metric_seed),
    )
    mediation = compute_interaction_mediation_metrics(
        baseline_on,
        counterfactual_on,
        baseline_off,
        counterfactual_off,
        receiver_mask=receiver_mask,
        spatial_dim=int(spatial_dim),
    )
    return GeneCounterfactualResult(
        edit=edit,
        baseline_audit=baseline_audit,
        counterfactual_audit=counterfactual_audit,
        baseline_on=baseline_on,
        counterfactual_on=counterfactual_on,
        baseline_off=baseline_off,
        counterfactual_off=counterfactual_off,
        metrics_on=metrics_on,
        metrics_off=metrics_off,
        mediation=mediation,
    )
