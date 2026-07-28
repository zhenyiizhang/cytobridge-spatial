"""Build reader-facing spatial LOTO comparisons for zebrafish.

This report deliberately complements, rather than replaces, the formal
Wasserstein benchmark.  For every physically held-out stage it places the
observed tissue beside:

* the bracket centroid-shift control, which uses the observed left and right
  training anchors and translates the complete left-anchor cloud by one vector;
* the CytoBridge LOTO prediction trained without the target stage.

The report also shows train-only kNN cell-type and gene-program readouts.  These
readouts make the spatial result interpretable to a biological reader, but they
are not direct gene-expression simulations and are labelled accordingly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
import textwrap
from typing import Any, Mapping, Sequence

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.ndimage import gaussian_filter
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor


matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

SCHEMA_VERSION = "zebrafish-loto-spatial-reader-report-v2"
METHODS = (
    ("Observed held-out", "#222222"),
    ("Bracket centroid-shift", "#D97706"),
    ("CytoBridge", "#1769AA"),
)


class ReportError(RuntimeError):
    """Raised when the comparison contract is not satisfied."""


@dataclass(frozen=True)
class Prediction:
    spatial: np.ndarray
    state: np.ndarray
    weights: np.ndarray
    path: Path
    sha256: str


@dataclass(frozen=True)
class StageComparison:
    target: float
    observed_spatial: np.ndarray
    observed_state: np.ndarray
    observed_labels: np.ndarray
    linear: Prediction
    cytobridge: Prediction
    train_state: np.ndarray
    train_labels: np.ndarray


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _time_token(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value).replace(".", "p")


def _as_dense(values: Any) -> np.ndarray:
    if hasattr(values, "to_memory"):
        values = values.to_memory()
    if sparse.issparse(values):
        values = values.toarray()
    return np.asarray(values)


def _normalised_weights(values: np.ndarray | None, n_obs: int) -> np.ndarray:
    if values is None:
        return np.full(n_obs, 1.0 / n_obs, dtype=np.float64)
    weights = np.asarray(values, dtype=np.float64).reshape(-1)
    if weights.shape != (n_obs,):
        raise ReportError(f"Prediction weights have shape {weights.shape}, expected {(n_obs,)}")
    if not np.isfinite(weights).all() or np.any(weights < 0.0) or weights.sum() <= 0.0:
        raise ReportError("Prediction weights must be finite, nonnegative and have positive mass")
    return weights / weights.sum()


def load_prediction(path: Path, *, expected_spatial_dim: int = 2) -> Prediction:
    path = path.resolve()
    if not path.is_file():
        raise ReportError(f"Missing prediction: {path}")
    with np.load(path, allow_pickle=False) as archive:
        if "spatial" not in archive or "state" not in archive:
            raise ReportError(f"{path} must contain spatial and state arrays")
        spatial_values = np.asarray(archive["spatial"], dtype=np.float64)
        state_values = np.asarray(archive["state"], dtype=np.float64)
        raw_weights = np.asarray(archive["weights"]) if "weights" in archive else None
    if spatial_values.ndim != 2 or spatial_values.shape[1] != expected_spatial_dim:
        raise ReportError(
            f"{path} spatial array has shape {spatial_values.shape}; "
            f"expected (n, {expected_spatial_dim})"
        )
    if state_values.ndim != 2 or state_values.shape[0] != spatial_values.shape[0]:
        raise ReportError(
            f"{path} state array has shape {state_values.shape}, incompatible with spatial"
        )
    if not np.isfinite(spatial_values).all() or not np.isfinite(state_values).all():
        raise ReportError(f"{path} contains non-finite prediction values")
    weights = _normalised_weights(raw_weights, len(spatial_values))
    return Prediction(
        spatial=spatial_values,
        state=state_values,
        weights=weights,
        path=path,
        sha256=sha256_file(path),
    )


def choose_cell_types(labels: Sequence[object], maximum: int) -> list[str]:
    values = pd.Series(np.asarray(labels).astype(str))
    counts = (
        values.value_counts()
        .rename_axis("label")
        .reset_index(name="count")
        .sort_values(["count", "label"], ascending=[False, True], kind="stable")
    )
    return counts["label"].head(maximum).tolist()


def choose_shared_cell_types(
    labels_by_method: Sequence[Sequence[object]],
    maximum: int,
    *,
    minimum_count: int,
    minimum_fraction: float,
) -> tuple[list[str], dict[str, dict[str, int]], list[int]]:
    """Choose visible compartments without changing any method output.

    Candidates are ranked only by measured held-out abundance.  A candidate is
    retained only when the measured target and both train-only prediction
    readouts contain enough cells to draw a stable spatial territory.
    """

    if len(labels_by_method) != len(METHODS):
        raise ReportError(
            f"Expected labels for {len(METHODS)} methods, got {len(labels_by_method)}"
        )
    if maximum <= 0 or minimum_count <= 0:
        raise ReportError("Cell-type selection limits must be positive")
    if not 0.0 <= minimum_fraction <= 1.0:
        raise ReportError("--min-cell-type-fraction must lie in [0, 1]")
    arrays = [np.asarray(values).astype(str) for values in labels_by_method]
    thresholds = [
        max(int(minimum_count), int(np.ceil(minimum_fraction * len(values))))
        for values in arrays
    ]
    observed_counts = pd.Series(arrays[0]).value_counts()
    ordered_candidates = sorted(
        observed_counts.index.astype(str),
        key=lambda label: (-int(observed_counts[label]), label),
    )
    selected: list[str] = []
    counts_by_label: dict[str, dict[str, int]] = {}
    method_names = [name for name, _ in METHODS]
    for label in ordered_candidates:
        counts = [int(np.sum(values == label)) for values in arrays]
        if all(count >= threshold for count, threshold in zip(counts, thresholds)):
            selected.append(label)
            counts_by_label[label] = dict(zip(method_names, counts))
        if len(selected) >= maximum:
            break
    return selected, counts_by_label, thresholds


def spatial_extent(arrays: Sequence[np.ndarray], margin_fraction: float = 0.04) -> tuple[float, ...]:
    points = np.concatenate([np.asarray(values, dtype=np.float64) for values in arrays], axis=0)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ReportError("Spatial coordinates must be finite n-by-2 arrays")
    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)
    x_margin = max((x_max - x_min) * margin_fraction, np.finfo(float).eps)
    y_margin = max((y_max - y_min) * margin_fraction, np.finfo(float).eps)
    return (
        float(x_min - x_margin),
        float(x_max + x_margin),
        float(y_min - y_margin),
        float(y_max + y_margin),
    )


def density_grid(
    spatial_values: np.ndarray,
    weights: np.ndarray,
    extent: Sequence[float],
    *,
    bins: int,
    smooth_sigma: float,
) -> np.ndarray:
    points = np.asarray(spatial_values, dtype=np.float64)
    weights = _normalised_weights(np.asarray(weights), len(points))
    x_min, x_max, y_min, y_max = map(float, extent)
    grid, _, _ = np.histogram2d(
        points[:, 1],
        points[:, 0],
        bins=(bins, bins),
        range=((y_min, y_max), (x_min, x_max)),
        weights=weights,
    )
    grid = gaussian_filter(grid, sigma=float(smooth_sigma), mode="constant")
    total = float(grid.sum())
    if total <= 0.0 or not np.isfinite(grid).all():
        raise ReportError("Density grid is empty or non-finite")
    return grid / total


def enclosed_mass_threshold(grid: np.ndarray, fraction: float) -> float:
    if not (0.0 < fraction < 1.0):
        raise ValueError("Enclosed-mass fraction must lie strictly between zero and one")
    values = np.asarray(grid, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values) & (values > 0.0)]
    if len(values) == 0:
        raise ReportError("Cannot find a density contour for an empty grid")
    ordered = np.sort(values)[::-1]
    cumulative = np.cumsum(ordered)
    index = int(np.searchsorted(cumulative, fraction * cumulative[-1], side="left"))
    return float(ordered[min(index, len(ordered) - 1)])


def _weighted_signal_grid(
    spatial_values: np.ndarray,
    signal: np.ndarray,
    weights: np.ndarray,
    extent: Sequence[float],
    *,
    bins: int,
    smooth_sigma: float,
) -> np.ndarray:
    points = np.asarray(spatial_values, dtype=np.float64)
    signal = np.asarray(signal, dtype=np.float64).reshape(-1)
    weights = _normalised_weights(np.asarray(weights), len(points))
    if signal.shape != (len(points),) or not np.isfinite(signal).all():
        raise ReportError("Spatial signal is invalid")
    x_min, x_max, y_min, y_max = map(float, extent)
    common = {
        "bins": (bins, bins),
        "range": ((y_min, y_max), (x_min, x_max)),
    }
    numerator, _, _ = np.histogram2d(
        points[:, 1], points[:, 0], weights=weights * signal, **common
    )
    denominator, _, _ = np.histogram2d(
        points[:, 1], points[:, 0], weights=weights, **common
    )
    numerator = gaussian_filter(numerator, sigma=float(smooth_sigma), mode="constant")
    denominator = gaussian_filter(denominator, sigma=float(smooth_sigma), mode="constant")
    result = np.full_like(numerator, np.nan)
    support = denominator >= max(float(denominator.max()) * 0.015, np.finfo(float).eps)
    result[support] = numerator[support] / denominator[support]
    return result


def _image_cmap(color: str) -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list("method_density", ["#FFFFFF", color])


def _format_axis(ax: plt.Axes, extent: Sequence[float]) -> None:
    x_min, x_max, y_min, y_max = map(float, extent)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _save_figure(figure: plt.Figure, stem: Path) -> list[dict[str, Any]]:
    outputs = []
    for extension, kwargs in (
        (".png", {"dpi": 240}),
        (".pdf", {}),
    ):
        path = stem.with_suffix(extension)
        figure.savefig(path, bbox_inches="tight", facecolor="white", **kwargs)
        outputs.append(
            {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "size_bytes": int(path.stat().st_size),
            }
        )
    plt.close(figure)
    return outputs


def _method_arrays(stage: StageComparison) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    observed_weights = np.full(
        len(stage.observed_spatial), 1.0 / len(stage.observed_spatial), dtype=np.float64
    )
    return (
        (stage.observed_spatial, observed_weights),
        (stage.linear.spatial, stage.linear.weights),
        (stage.cytobridge.spatial, stage.cytobridge.weights),
    )


def _predict_labels(
    train_state: np.ndarray,
    train_labels: np.ndarray,
    state_by_method: Sequence[np.ndarray],
    *,
    neighbors: int,
) -> list[np.ndarray]:
    n_neighbors = min(int(neighbors), len(train_state))
    if n_neighbors < 1:
        raise ReportError("At least one training cell is required for label readout")
    classifier = KNeighborsClassifier(n_neighbors=n_neighbors, weights="distance")
    classifier.fit(train_state, train_labels)
    return [classifier.predict(values).astype(str) for values in state_by_method]


def _resolve_genes(var_names: Sequence[object], genes: Sequence[str]) -> tuple[list[int], list[str]]:
    observed: dict[str, int] = {}
    for index, value in enumerate(np.asarray(var_names).astype(str)):
        observed.setdefault(value.casefold(), index)
    indices = []
    resolved = []
    for gene in genes:
        index = observed.get(str(gene).casefold())
        if index is not None:
            indices.append(index)
            resolved.append(str(np.asarray(var_names).astype(str)[index]))
    return indices, resolved


def _expression_columns(adata: ad.AnnData, row_mask: np.ndarray, columns: Sequence[int]) -> np.ndarray:
    values = adata[row_mask, list(columns)].X
    return _as_dense(values).astype(np.float64, copy=False)


def _program_scores(
    adata: ad.AnnData,
    row_mask: np.ndarray,
    genes: Sequence[str],
    *,
    center: np.ndarray,
    scale: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    indices, resolved = _resolve_genes(adata.var_names, genes)
    if not indices:
        raise ReportError(f"None of the requested genes are present: {list(genes)}")
    expression = _expression_columns(adata, row_mask, indices)
    selected_center = center[np.asarray(indices, dtype=int)]
    selected_scale = scale[np.asarray(indices, dtype=int)]
    score = np.mean((expression - selected_center) / selected_scale, axis=1)
    return np.asarray(score, dtype=np.float64), resolved


def _training_gene_scale(adata: ad.AnnData) -> tuple[np.ndarray, np.ndarray]:
    values = adata.X
    if hasattr(values, "to_memory"):
        values = values.to_memory()
    if sparse.issparse(values):
        center = np.asarray(values.mean(axis=0)).reshape(-1)
        squared = np.asarray(values.multiply(values).mean(axis=0)).reshape(-1)
        variance = np.maximum(squared - center**2, 0.0)
    else:
        array = np.asarray(values, dtype=np.float64)
        center = array.mean(axis=0)
        variance = array.var(axis=0)
    scale = np.sqrt(variance)
    scale[scale < 1e-8] = 1.0
    return center.astype(np.float64), scale.astype(np.float64)


def _fit_program_readout(
    train_state: np.ndarray,
    train_score: np.ndarray,
    prediction_states: Sequence[np.ndarray],
    *,
    neighbors: int,
) -> list[np.ndarray]:
    n_neighbors = min(int(neighbors), len(train_state))
    readout = KNeighborsRegressor(n_neighbors=n_neighbors, weights="distance")
    readout.fit(train_state, train_score)
    return [np.asarray(readout.predict(state), dtype=np.float64) for state in prediction_states]


def plot_morphology(
    stage: StageComparison,
    output_dir: Path,
    *,
    bins: int,
    smooth_sigma: float,
) -> tuple[list[dict[str, Any]], list[np.ndarray], tuple[float, ...]]:
    method_values = _method_arrays(stage)
    extent = spatial_extent([values[0] for values in method_values])
    grids = [
        density_grid(points, weights, extent, bins=bins, smooth_sigma=smooth_sigma)
        for points, weights in method_values
    ]
    vmax = max(float(np.quantile(grid[grid > 0.0], 0.995)) for grid in grids)
    figure, axes = plt.subplots(1, 3, figsize=(10.8, 3.6), constrained_layout=True)
    for ax, (name, color), grid in zip(axes, METHODS, grids):
        ax.imshow(
            grid,
            extent=extent,
            origin="lower",
            interpolation="bilinear",
            cmap=_image_cmap(color),
            vmin=0.0,
            vmax=vmax,
        )
        ax.set_title(name, fontsize=11, color=color, fontweight="bold")
        _format_axis(ax, extent)
    figure.suptitle(
        f"Held-out stage t{_time_token(stage.target)}: tissue-density reconstruction",
        fontsize=13,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.01,
        "Linear is bracketed: it uses both flanking observed stages and translates "
        "the left cloud by one global centroid vector.",
        ha="center",
        fontsize=8.5,
        color="#444444",
    )
    outputs = _save_figure(
        figure, output_dir / f"t{_time_token(stage.target)}_01_tissue_density"
    )
    return outputs, grids, extent


def plot_boundaries(
    stage: StageComparison,
    method_grids: Sequence[np.ndarray],
    labels_by_method: Sequence[np.ndarray],
    selected_cell_types: Sequence[str],
    extent: Sequence[float],
    output_dir: Path,
    *,
    bins: int,
    smooth_sigma: float,
    enclosed_fraction: float,
) -> list[dict[str, Any]]:
    method_values = _method_arrays(stage)
    titles = ["Whole tissue", *selected_cell_types]
    figure, axes = plt.subplots(
        1,
        len(titles),
        figsize=(3.6 * len(titles), 5.0),
        squeeze=False,
    )
    for column, (ax, title) in enumerate(zip(axes[0], titles)):
        for method_index, ((name, color), (points, weights)) in enumerate(
            zip(METHODS, method_values)
        ):
            if column == 0:
                grid = method_grids[method_index]
            else:
                keep = np.asarray(labels_by_method[method_index]).astype(str) == title
                if not np.any(keep):
                    continue
                grid = density_grid(
                    points[keep],
                    weights[keep],
                    extent,
                    bins=bins,
                    smooth_sigma=smooth_sigma,
                )
            level = enclosed_mass_threshold(grid, enclosed_fraction)
            ax.contour(
                grid,
                levels=[level],
                colors=[color],
                linewidths=2.0,
                extent=extent,
                origin="lower",
            )
        ax.set_title(
            "\n".join(textwrap.wrap(title, width=24)),
            fontsize=9.5,
            fontweight="bold",
            pad=8,
        )
        _format_axis(ax, extent)
    handles = [
        plt.Line2D([0], [0], color=color, linewidth=2.3, label=name)
        for name, color in METHODS
    ]
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.88),
        ncol=3,
        frameon=False,
        fontsize=9.5,
    )
    figure.suptitle(
        f"Do predicted anatomical boundaries follow the held-out tissue? "
        f"(t{_time_token(stage.target)}, {enclosed_fraction:.0%} mass contours)",
        y=0.98,
        fontsize=12.0,
        fontweight="bold",
    )
    figure.subplots_adjust(
        top=0.70,
        bottom=0.05,
        left=0.02,
        right=0.99,
        wspace=0.10,
    )
    return _save_figure(
        figure, output_dir / f"t{_time_token(stage.target)}_02_boundary_overlay"
    )


def plot_cell_type_density(
    stage: StageComparison,
    labels_by_method: Sequence[np.ndarray],
    selected_cell_types: Sequence[str],
    extent: Sequence[float],
    output_dir: Path,
    *,
    bins: int,
    smooth_sigma: float,
) -> list[dict[str, Any]]:
    method_values = _method_arrays(stage)
    figure, axes = plt.subplots(
        len(selected_cell_types),
        3,
        figsize=(10.6, 2.7 * len(selected_cell_types)),
        constrained_layout=True,
        squeeze=False,
    )
    for row, label in enumerate(selected_cell_types):
        grids = []
        for (points, weights), method_labels in zip(method_values, labels_by_method):
            keep = np.asarray(method_labels).astype(str) == label
            if np.any(keep):
                grids.append(
                    density_grid(
                        points[keep],
                        weights[keep],
                        extent,
                        bins=bins,
                        smooth_sigma=smooth_sigma,
                    )
                )
            else:
                grids.append(np.zeros((bins, bins), dtype=float))
        positive = np.concatenate([grid[grid > 0.0] for grid in grids])
        vmax = float(np.quantile(positive, 0.995)) if len(positive) else 1.0
        for column, (ax, (name, color), grid) in enumerate(zip(axes[row], METHODS, grids)):
            ax.imshow(
                grid,
                extent=extent,
                origin="lower",
                interpolation="bilinear",
                cmap=_image_cmap(color),
                vmin=0.0,
                vmax=vmax,
            )
            if row == 0:
                ax.set_title(name, fontsize=10.5, color=color, fontweight="bold")
            if column == 0:
                ax.text(
                    -0.04,
                    0.5,
                    label,
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="right",
                    fontsize=10,
                    fontweight="bold",
                )
            _format_axis(ax, extent)
    figure.suptitle(
        f"Cell-type territories at held-out t{_time_token(stage.target)}",
        fontsize=13,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.005,
        "Prediction labels come from the same distance-weighted kNN trained only on "
        "the target-removed fold; observed labels are measured annotations.",
        ha="center",
        fontsize=8.5,
        color="#444444",
    )
    return _save_figure(
        figure, output_dir / f"t{_time_token(stage.target)}_03_cell_type_territories"
    )


def plot_gene_programs(
    stage: StageComparison,
    source: ad.AnnData,
    train: ad.AnnData,
    observed_mask: np.ndarray,
    programs: Mapping[str, Sequence[str]],
    extent: Sequence[float],
    output_dir: Path,
    *,
    bins: int,
    smooth_sigma: float,
    neighbors: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not programs:
        return [], []
    train_center, train_scale = _training_gene_scale(train)
    method_values = _method_arrays(stage)
    outputs: list[dict[str, Any]] = []
    program_records: list[dict[str, Any]] = []
    figure, axes = plt.subplots(
        len(programs),
        3,
        figsize=(10.8, 2.8 * len(programs)),
        constrained_layout=True,
        squeeze=False,
    )
    for row, (program, genes) in enumerate(programs.items()):
        train_mask = np.ones(train.n_obs, dtype=bool)
        train_score, resolved_train = _program_scores(
            train,
            train_mask,
            genes,
            center=train_center,
            scale=train_scale,
        )
        source_indices, resolved_source = _resolve_genes(source.var_names, genes)
        if [gene.casefold() for gene in resolved_train] != [
            gene.casefold() for gene in resolved_source
        ]:
            raise ReportError(
                f"Program {program!r} resolves differently in source and train H5AD files"
            )
        source_expression = _expression_columns(source, observed_mask, source_indices)
        train_indices, _ = _resolve_genes(train.var_names, resolved_train)
        observed_score = np.mean(
            (
                source_expression
                - train_center[np.asarray(train_indices, dtype=int)]
            )
            / train_scale[np.asarray(train_indices, dtype=int)],
            axis=1,
        )
        predicted_scores = _fit_program_readout(
            stage.train_state,
            train_score,
            [stage.linear.state, stage.cytobridge.state],
            neighbors=neighbors,
        )
        method_scores = [observed_score, *predicted_scores]
        grids = [
            _weighted_signal_grid(
                points,
                score,
                weights,
                extent,
                bins=bins,
                smooth_sigma=smooth_sigma,
            )
            for (points, weights), score in zip(method_values, method_scores)
        ]
        finite = np.concatenate([grid[np.isfinite(grid)] for grid in grids])
        vmin, vmax = np.quantile(finite, [0.02, 0.98])
        if np.isclose(vmin, vmax):
            vmin, vmax = float(vmin - 1.0), float(vmax + 1.0)
        for column, (ax, (name, _), grid) in enumerate(zip(axes[row], METHODS, grids)):
            image = ax.imshow(
                grid,
                extent=extent,
                origin="lower",
                interpolation="bilinear",
                cmap="coolwarm",
                vmin=float(vmin),
                vmax=float(vmax),
            )
            if row == 0:
                ax.set_title(name, fontsize=10.5, fontweight="bold")
            if column == 0:
                ax.text(
                    -0.04,
                    0.5,
                    program,
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="right",
                    fontsize=9.5,
                    fontweight="bold",
                )
            _format_axis(ax, extent)
            if column == 2:
                figure.colorbar(image, ax=axes[row], shrink=0.72, pad=0.01)
        program_records.append(
            {
                "program": program,
                "requested_genes": list(genes),
                "resolved_genes": resolved_train,
                "readout": (
                    "distance-weighted kNN from target-removed training PCA state "
                    "to standardized measured program score"
                ),
            }
        )
    figure.suptitle(
        f"Measured versus state-imputed gene programs at held-out "
        f"t{_time_token(stage.target)}",
        fontsize=13,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.002,
        "Observed = measured normalized expression. Linear/CytoBridge = the same "
        "train-only PCA-state readout, not direct gene-expression simulation.",
        ha="center",
        fontsize=8.3,
        color="#444444",
    )
    outputs.extend(
        _save_figure(
            figure, output_dir / f"t{_time_token(stage.target)}_04_gene_program_readout"
        )
    )
    return outputs, program_records


def _obs_values(adata: ad.AnnData, key: str) -> np.ndarray:
    if key not in adata.obs:
        raise ReportError(f"adata.obs is missing {key!r}")
    return adata.obs[key].to_numpy()


def _obsm_values(adata: ad.AnnData, key: str) -> np.ndarray:
    if key not in adata.obsm:
        raise ReportError(f"adata.obsm is missing {key!r}")
    result = np.asarray(adata.obsm[key], dtype=np.float64)
    if result.ndim != 2 or not np.isfinite(result).all():
        raise ReportError(f"adata.obsm[{key!r}] is invalid")
    return result


def load_stage(
    source: ad.AnnData,
    train: ad.AnnData,
    prediction_root: Path,
    target: float,
    *,
    source_time_key: str,
    source_spatial_key: str,
    source_state_key: str,
    source_annotation_key: str,
    train_state_key: str,
    train_annotation_key: str,
    linear_method: str,
    cytobridge_method: str,
) -> tuple[StageComparison, np.ndarray]:
    source_time = np.asarray(_obs_values(source, source_time_key), dtype=float)
    observed_mask = np.isclose(source_time, target)
    if not np.any(observed_mask):
        raise ReportError(f"Source H5AD has no observed cells at target {target}")
    token = _time_token(target)
    linear = load_prediction(
        prediction_root / linear_method / f"t{token}" / "prediction.npz"
    )
    cytobridge = load_prediction(
        prediction_root / cytobridge_method / f"t{token}" / "prediction.npz"
    )
    observed_state = _obsm_values(source, source_state_key)[observed_mask]
    state_dim = observed_state.shape[1]
    if linear.state.shape[1] != state_dim or cytobridge.state.shape[1] != state_dim:
        raise ReportError(
            f"State dimensions disagree at t{token}: observed={state_dim}, "
            f"linear={linear.state.shape[1]}, CytoBridge={cytobridge.state.shape[1]}"
        )
    train_times = np.asarray(_obs_values(train, "benchmark_time"), dtype=float)
    if np.any(np.isclose(train_times, target)):
        raise ReportError(
            f"Training fold for target t{token} still contains target rows; "
            "this is not a physical LOTO split"
        )
    stage = StageComparison(
        target=float(target),
        observed_spatial=_obsm_values(source, source_spatial_key)[observed_mask],
        observed_state=observed_state,
        observed_labels=np.asarray(
            _obs_values(source, source_annotation_key)[observed_mask]
        ).astype(str),
        linear=linear,
        cytobridge=cytobridge,
        train_state=_obsm_values(train, train_state_key),
        train_labels=np.asarray(_obs_values(train, train_annotation_key)).astype(str),
    )
    return stage, observed_mask


def _load_programs(path: Path | None) -> dict[str, list[str]]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload:
        raise ReportError("--programs-json must contain a non-empty JSON object")
    result: dict[str, list[str]] = {}
    for name, genes in payload.items():
        if not isinstance(genes, list) or not genes or not all(
            isinstance(gene, str) and gene for gene in genes
        ):
            raise ReportError(f"Program {name!r} must be a non-empty string list")
        result[str(name)] = genes
    return result


def _reader_guide(
    targets: Sequence[float],
    programs: Mapping[str, Sequence[str]],
    *,
    linear_method: str,
    cytobridge_method: str,
    selected_by_target: Mapping[str, Sequence[str]],
) -> str:
    target_text = ", ".join(f"t{_time_token(value)}" for value in targets)
    program_text = (
        "\n".join(f"- {name}: `{', '.join(genes)}`" for name, genes in programs.items())
        if programs
        else "- No gene-program panel was requested."
    )
    cell_text = "\n".join(
        f"- t{target}: {', '.join(labels)}" for target, labels in selected_by_target.items()
    )
    return f"""# Zebrafish held-out spatial comparison

## What is being compared

Targets {target_text} are physically absent from each method's training fold.
The observed target is opened only by this reporting step.

- **Observed held-out**: measured cells at the omitted stage.
- **Bracket centroid-shift** (`{linear_method}`): starts from the nearest left
  observed stage and adds one global vector,
  `alpha * (right centroid - left centroid)`, to every source cell.  It therefore
  uses both flanking stages.  It is a strong **transductive interpolation**, not
  a forward-only prediction.
- **CytoBridge** (`{cytobridge_method}`): the target-removed LOTO prediction.

The comparison is intentionally not tuned to make either method win.  The same
coordinates, density grid, contour mass, classifier and program readout are used
for both predictions.

## How to read the figures

1. `01_tissue_density`: compare the measured tissue silhouette with the two
   generated silhouettes.  A centroid-shift control can translate a cloud but
   cannot create a local bend, expansion, contraction or cell-type-specific
   rearrangement.
2. `02_boundary_overlay`: black is observed, orange is bracket interpolation and
   blue is CytoBridge.  The most direct visual question is whether blue follows
   the black anatomical boundary more closely than orange, and where it does not.
3. `03_cell_type_territories`: rows are ranked by measured held-out abundance,
   then retained only when observed, Linear and CytoBridge each have enough
   cells for a readable territory.  This display-only rule is fixed across
   methods and does not change predictions or numerical metrics.  Observed
   labels are measured; both predictions are labelled by the same train-only kNN.
4. `04_gene_program_readout`: observed maps use measured expression.  Prediction
   maps use the same train-only kNN readout from 50D PCA state to a standardized
   gene-program score.  These are interpretable state readouts, **not direct gene
   simulations** and not perturbation evidence.

## Cell types with shared display support

{cell_text}

## Gene signatures

{program_text}

## Appropriate conclusion

These panels can support a statement about recovery of held-out spatial
morphology, cell-type territories and PCA-state-associated gene programs.  They
must be shown together with the complete numerical benchmark.  They cannot turn
an interpolation experiment into future forecasting, and they do not by
themselves establish a causal signaling mechanism.
"""


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ReportError(f"Refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    programs = _load_programs(args.programs_json)
    source_path = args.source_h5ad.expanduser().resolve()
    input_root = args.loto_input_root.expanduser().resolve()
    prediction_root = args.prediction_root.expanduser().resolve()
    source = ad.read_h5ad(source_path)
    stage_records: list[dict[str, Any]] = []
    selected_by_target: dict[str, list[str]] = {}
    try:
        for target in args.targets:
            token = _time_token(target)
            train_path = input_root / f"loto_t{token}" / "train.h5ad"
            if not train_path.is_file():
                raise ReportError(f"Missing target-removed train H5AD: {train_path}")
            train = ad.read_h5ad(train_path)
            try:
                stage, observed_mask = load_stage(
                    source,
                    train,
                    prediction_root,
                    target,
                    source_time_key=args.source_time_key,
                    source_spatial_key=args.source_spatial_key,
                    source_state_key=args.source_state_key,
                    source_annotation_key=args.source_annotation_key,
                    train_state_key=args.train_state_key,
                    train_annotation_key=args.train_annotation_key,
                    linear_method=args.linear_method,
                    cytobridge_method=args.cytobridge_method,
                )
                predicted_labels = _predict_labels(
                    stage.train_state,
                    stage.train_labels,
                    [stage.linear.state, stage.cytobridge.state],
                    neighbors=args.knn_neighbors,
                )
                labels_by_method = [
                    stage.observed_labels,
                    predicted_labels[0],
                    predicted_labels[1],
                ]
                if args.cell_types:
                    selected_types = list(args.cell_types)
                    selection_counts = {
                        label: {
                            name: int(np.sum(np.asarray(labels).astype(str) == label))
                            for (name, _), labels in zip(METHODS, labels_by_method)
                        }
                        for label in selected_types
                    }
                    selection_thresholds = [0] * len(METHODS)
                else:
                    (
                        selected_types,
                        selection_counts,
                        selection_thresholds,
                    ) = choose_shared_cell_types(
                        labels_by_method,
                        args.max_cell_types,
                        minimum_count=args.min_cell_type_count,
                        minimum_fraction=args.min_cell_type_fraction,
                    )
                if not selected_types:
                    raise ReportError(
                        f"No cell type at t{token} met the shared display-support "
                        "thresholds; lower --min-cell-type-count or "
                        "--min-cell-type-fraction explicitly."
                    )
                selected_by_target[token] = selected_types
                morphology_outputs, grids, extent = plot_morphology(
                    stage,
                    output_dir,
                    bins=args.grid_bins,
                    smooth_sigma=args.smooth_sigma,
                )
                boundary_outputs = plot_boundaries(
                    stage,
                    grids,
                    labels_by_method,
                    selected_types,
                    extent,
                    output_dir,
                    bins=args.grid_bins,
                    smooth_sigma=args.smooth_sigma,
                    enclosed_fraction=args.enclosed_fraction,
                )
                cell_outputs = plot_cell_type_density(
                    stage,
                    labels_by_method,
                    selected_types,
                    extent,
                    output_dir,
                    bins=args.grid_bins,
                    smooth_sigma=args.smooth_sigma,
                )
                gene_outputs, program_records = plot_gene_programs(
                    stage,
                    source,
                    train,
                    observed_mask,
                    programs,
                    extent,
                    output_dir,
                    bins=args.grid_bins,
                    smooth_sigma=args.smooth_sigma,
                    neighbors=args.knn_neighbors,
                )
                stage_records.append(
                    {
                        "target": float(target),
                        "observed_n": int(observed_mask.sum()),
                        "train_n": int(train.n_obs),
                        "target_physically_absent_from_train": True,
                        "selected_cell_types_observed_ranked_shared_support": selected_types,
                        "selected_cell_type_counts": selection_counts,
                        "selected_cell_type_minimum_counts": {
                            name: int(threshold)
                            for (name, _), threshold in zip(METHODS, selection_thresholds)
                        },
                        "linear_prediction": {
                            "path": str(stage.linear.path),
                            "sha256": stage.linear.sha256,
                            "n": int(len(stage.linear.spatial)),
                            "weights": "uniform (prediction archive has no native weights)",
                        },
                        "cytobridge_prediction": {
                            "path": str(stage.cytobridge.path),
                            "sha256": stage.cytobridge.sha256,
                            "n": int(len(stage.cytobridge.spatial)),
                            "weights": "native weights normalised to unit display mass",
                        },
                        "programs": program_records,
                        "figures": [
                            *morphology_outputs,
                            *boundary_outputs,
                            *cell_outputs,
                            *gene_outputs,
                        ],
                    }
                )
            finally:
                train.file.close() if train.isbacked else None
    finally:
        source.file.close() if source.isbacked else None
    guide_path = output_dir / "START_HERE.md"
    guide_path.write_text(
        _reader_guide(
            args.targets,
            programs,
            linear_method=args.linear_method,
            cytobridge_method=args.cytobridge_method,
            selected_by_target=selected_by_target,
        ),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "analysis": "reader_facing_zebrafish_loto_spatial_comparison",
        "scientific_scope": (
            "transductive LOTO in frozen aligned spatial and PCA spaces; "
            "not forward-only forecasting"
        ),
        "source_h5ad": {
            "path": str(source_path),
            "sha256": sha256_file(source_path),
        },
        "loto_input_root": str(input_root),
        "prediction_root": str(prediction_root),
        "methods": {
            "linear": {
                "id": args.linear_method,
                "display_name": "Bracket centroid-shift interpolation",
                "uses_left_and_right_anchors": True,
                "changes_shape_beyond_global_translation": False,
            },
            "cytobridge": {"id": args.cytobridge_method},
        },
        "reporting": {
            "grid_bins": int(args.grid_bins),
            "smooth_sigma_grid_cells": float(args.smooth_sigma),
            "enclosed_mass_fraction": float(args.enclosed_fraction),
            "knn_neighbors": int(args.knn_neighbors),
            "cell_type_selection": (
                "explicit CLI list"
                if args.cell_types
                else (
                    f"top {args.max_cell_types} by measured held-out abundance after "
                    "requiring shared display support in observed, Linear and "
                    "CytoBridge; used only for reader-facing plots"
                )
            ),
            "minimum_cell_type_count": int(args.min_cell_type_count),
            "minimum_cell_type_fraction": float(args.min_cell_type_fraction),
            "gene_program_readout": (
                "measured at observed target; distance-weighted kNN from target-removed "
                "training PCA state for both predictions"
            ),
            "metrics_replaced": False,
            "posthoc_method_tuning": False,
        },
        "stages": stage_records,
        "reader_guide": {
            "path": str(guide_path),
            "sha256": sha256_file(guide_path),
        },
    }
    _write_json(output_dir / "run_manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-h5ad", required=True, type=Path)
    parser.add_argument(
        "--loto-input-root",
        required=True,
        type=Path,
        help="Root containing loto_t1/train.h5ad, loto_t2/train.h5ad, ...",
    )
    parser.add_argument(
        "--prediction-root",
        required=True,
        type=Path,
        help="Root containing METHOD/t1/prediction.npz, METHOD/t2/..., ...",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--targets", nargs="+", type=float, default=[1.0, 2.0, 3.0])
    parser.add_argument("--linear-method", default="linear_centroid_shift")
    parser.add_argument("--cytobridge-method", default="CytoBridge-0.015")
    parser.add_argument("--source-time-key", default="time_point_processed")
    parser.add_argument("--source-spatial-key", default="spatial_aligned")
    parser.add_argument("--source-state-key", default="X_latent")
    parser.add_argument("--source-annotation-key", default="Annotation")
    parser.add_argument("--train-state-key", default="benchmark_state")
    parser.add_argument("--train-annotation-key", default="benchmark_annotation")
    parser.add_argument("--cell-types", nargs="*", default=None)
    parser.add_argument("--max-cell-types", type=int, default=3)
    parser.add_argument("--min-cell-type-count", type=int, default=100)
    parser.add_argument("--min-cell-type-fraction", type=float, default=0.005)
    parser.add_argument("--programs-json", type=Path, default=None)
    parser.add_argument("--knn-neighbors", type=int, default=10)
    parser.add_argument("--grid-bins", type=int, default=180)
    parser.add_argument("--smooth-sigma", type=float, default=2.2)
    parser.add_argument("--enclosed-fraction", type=float, default=0.80)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest = build_report(args)
    print(json.dumps(_jsonable(manifest), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
