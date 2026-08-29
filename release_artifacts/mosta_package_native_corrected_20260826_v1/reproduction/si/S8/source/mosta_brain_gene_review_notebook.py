"""Render-only reproduction of ``mosta-brain-gene-review.ipynb``.

The frozen notebook is the calculation and visual-style oracle.  This module
reads a current S8 mean-log1p gene-by-time table, recomputes the notebook's
sample-standardized profiles, average-linkage k=2 programs, and contiguous
k=3 developmental-wave phases, then renders the five notebook figure groups.
Saved formal Ward assignments, prototypes, and historical figure values are
never used as numerical inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


NOTEBOOK_ORACLE = "output/jupyter-notebook/mosta-brain-gene-review.ipynb"
NOTEBOOK_ORACLE_SHA256 = (
    "0a922803eb70662418f1ec65163dda30167fa2d9d2f60e07972e35a916bdef9f"
)
NOTEBOOK_ORACLE_CELLS = {
    "global_style_and_io": 1,
    "input_loading_and_sample_zscore": 3,
    "calculation_constants": 4,
    "temporal_k2_clustering": 6,
    "top_variable_heatmap": 9,
    "top_variable_trajectories": 11,
    "temporal_programs": 13,
    "wave_k3_preparation": 15,
    "developmental_wave_map": 17,
    "pattern_level_curves": 19,
}

TEMPORAL_N_CLUSTERS = 2
WAVE_N_PHASES = 3
HEATMAP_TOP_N = 60
TRAJECTORY_TOP_N = 25
WAVE_TOP_N = 1000
PATTERN_GENES_PER_CLUSTER = 12
REPRESENTATIVE_GENES_PER_PATTERN = 5
PATTERN_LABEL_TOP_N = 3
DISPLAY_SMOOTH_SIGMA = 0.0
PATTERN_VALUE_CLIP = 1.8
PATTERN_CLUSTER_COLORS = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#17becf",
    "#e377c2",
    "#bcbd22",
)
PATTERN_CMAP_NAME = "viridis"
PANEL_BG = "#ffffff"
GRID_COLOR = "#d9d4cb"
TEXT_DARK = "#2f2b28"
TEXT_MID = "#625b56"

FIGURE_STEMS = {
    "heatmap": "brain_top_variable_genes_heatmap",
    "top_variable_trajectories": "brain_top_variable_gene_trajectories",
    "temporal_programs": "brain_temporal_programs",
    "developmental_wave_map": "brain_developmental_wave_map",
    "pattern_level_curves": "brain_pattern_level_temporal_curves",
}


@dataclass(frozen=True)
class BrainGeneNotebookInputs:
    expression: pd.DataFrame
    s8_dir: Path
    input_identities: Mapping[str, Mapping[str, Any] | None]
    validation: Mapping[str, Any]


@dataclass(frozen=True)
class BrainGeneNotebookState:
    expression: pd.DataFrame
    zscore: pd.DataFrame
    variance_rank: pd.DataFrame
    assignments: pd.DataFrame
    curves: pd.DataFrame
    representatives: pd.DataFrame
    cluster_summary: pd.DataFrame
    wave_matrix: pd.DataFrame
    wave_assignments: pd.DataFrame
    wave_metrics: pd.DataFrame
    wave_summary: pd.DataFrame


def _sha256(path: str | Path) -> str:
    source = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_identity(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    stat = source.stat()
    return {
        "path": str(source),
        "size_bytes": int(stat.st_size),
        "sha256": _sha256(source),
    }


def _optional_file_identity(path: Path) -> dict[str, Any] | None:
    return _file_identity(path) if path.is_file() else None


def _array_sha256(values: np.ndarray) -> str:
    array = np.asarray(values)
    if array.dtype.kind in "fiu":
        array = np.asarray(array, dtype="<f8")
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _ordered_strings_sha256(values: Sequence[object]) -> str:
    payload = json.dumps(
        [str(value) for value in values], separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _profile_slug(value: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError("analysis_profile must be a non-empty string.")
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._-")
    if not slug:
        raise ValueError("analysis_profile has no usable filename token.")
    return slug


def _load_expression(path: Path) -> pd.DataFrame:
    expression = pd.read_csv(path, index_col=0)
    if expression.shape[0] < 3 or expression.shape[1] < 2:
        raise ValueError("S8 expression must contain at least 3 genes and 2 times.")
    genes = expression.index.astype(str)
    if genes.has_duplicates:
        raise ValueError("S8 expression gene names must be unique.")
    if any(not value.strip() for value in genes):
        raise ValueError("S8 expression contains an empty gene name.")
    try:
        times = np.asarray([float(value) for value in expression.columns], dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("S8 expression columns must be numeric model times.") from exc
    if not bool(np.isfinite(times).all()) or not bool(np.all(np.diff(times) > 0)):
        raise ValueError(
            "S8 model-time columns must be finite and strictly increasing."
        )
    values = expression.apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    if not bool(np.isfinite(values).all()):
        raise ValueError("S8 expression contains non-finite values.")
    return pd.DataFrame(values, index=genes, columns=times)


def _validate_formal_companions(
    s8_dir: Path,
    expression: pd.DataFrame,
) -> tuple[dict[str, dict[str, Any] | None], dict[str, Any]]:
    paths = {
        "mean_log1p": s8_dir / "brain_hvg_mean_log1p_by_time.csv",
        "variance_rank": s8_dir / "brain_hvg_temporal_variance_rank.csv",
        "formal_population_zscore": s8_dir / "brain_hvg_gene_wise_zscore.csv",
        "gene_name_map": s8_dir / "brain_hvg_gene_name_map.csv",
        "formal_ward_assignments_ignored": (
            s8_dir / "brain_hvg_ward_k2_assignments.csv"
        ),
        "formal_ward_prototypes_ignored": (
            s8_dir / "brain_hvg_ward_k2_prototypes.csv"
        ),
        "settings": s8_dir / "s8_gene_program_settings.json",
    }
    identities = {key: _optional_file_identity(path) for key, path in paths.items()}
    notebook_variance = expression.var(axis=1).sort_values(ascending=False)
    formal_population_variance = expression.var(axis=1, ddof=0)
    sample_zscore = (
        expression.sub(expression.mean(axis=1), axis=0)
        .div(expression.std(axis=1).replace(0, np.nan), axis=0)
        .fillna(0.0)
    )
    population_zscore = (
        expression.sub(expression.mean(axis=1), axis=0)
        .div(expression.std(axis=1, ddof=0).replace(0, np.nan), axis=0)
        .fillna(0.0)
    )
    validation: dict[str, Any] = {
        "expression_shape": [int(expression.shape[0]), int(expression.shape[1])],
        "current_formal_2000_by_13_shape": bool(expression.shape == (2000, 13)),
        "time_points": [float(value) for value in expression.columns],
        "gene_order_sha256": _ordered_strings_sha256(expression.index),
        "time_order_sha256": _array_sha256(expression.columns.to_numpy(float)),
        "mean_log1p_matrix_sha256": _array_sha256(expression.to_numpy(float)),
        "notebook_sample_zscore_matrix_sha256": _array_sha256(
            sample_zscore.to_numpy(float)
        ),
        "formal_population_zscore_is_compute_source": False,
        "formal_ward_assignments_are_compute_source": False,
        "formal_ward_prototypes_are_compute_source": False,
        "historical_notebook_tables_are_compute_source": False,
    }

    rank_path = paths["variance_rank"]
    if rank_path.is_file():
        rank = pd.read_csv(rank_path)
        required = {"gene", "variance"}
        missing = sorted(required.difference(rank.columns))
        if missing:
            raise KeyError(f"Formal variance table is missing columns: {missing}.")
        rank_genes = rank["gene"].astype(str)
        if rank_genes.tolist() != notebook_variance.index.astype(str).tolist():
            raise ValueError(
                "Formal variance rank order differs from recomputed current expression."
            )
        saved = pd.to_numeric(rank["variance"], errors="raise").to_numpy(float)
        expected = formal_population_variance.loc[rank_genes].to_numpy(float)
        if not np.allclose(saved, expected, rtol=1e-5, atol=1e-9):
            raise ValueError(
                "Formal variance values do not match population variance of "
                "the current mean-log1p table."
            )
        validation["formal_variance_rank"] = {
            "status": "exact_gene_order_and_population_variance_match",
            "saved_ddof": 0,
            "notebook_recomputed_ddof": 1,
            "used_as_compute_source": False,
        }
    else:
        validation["formal_variance_rank"] = {"status": "not_supplied"}

    zscore_path = paths["formal_population_zscore"]
    if zscore_path.is_file():
        saved_zscore = pd.read_csv(zscore_path, index_col=0)
        saved_zscore.columns = [float(value) for value in saved_zscore.columns]
        if set(saved_zscore.index.astype(str)) != set(expression.index.astype(str)):
            raise ValueError("Formal z-score gene set differs from current expression.")
        if saved_zscore.columns.tolist() != expression.columns.tolist():
            raise ValueError("Formal z-score times differ from current expression.")
        aligned = saved_zscore.loc[expression.index].to_numpy(float)
        expected = population_zscore.to_numpy(float)
        if not np.allclose(aligned, expected, rtol=1e-5, atol=1e-5):
            raise ValueError(
                "Formal z-score does not match population-standardized expression."
            )
        validation["formal_zscore"] = {
            "status": "population_zscore_match",
            "saved_ddof": 0,
            "notebook_recomputed_ddof": 1,
            "used_as_compute_source": False,
        }
    else:
        validation["formal_zscore"] = {"status": "not_supplied"}

    names_path = paths["gene_name_map"]
    if names_path.is_file():
        names = pd.read_csv(names_path)
        if "gene" not in names.columns:
            raise KeyError("Formal gene-name map is missing the gene column.")
        if set(names["gene"].astype(str)) != set(expression.index.astype(str)):
            raise ValueError("Formal gene-name map differs from expression genes.")
        validation["gene_name_map"] = {
            "status": "exact_gene_set_match",
            "used_as_compute_source": False,
        }
    else:
        validation["gene_name_map"] = {"status": "not_supplied"}

    settings_path = paths["settings"]
    if settings_path.is_file():
        settings = json.loads(settings_path.read_text(encoding="utf-8"))
        if str(settings.get("roi", "")).strip().lower() != "brain":
            raise ValueError("S8 settings ROI is not Brain.")
        configured_times = (
            settings.get("api_settings", {}).get("time_points", [])
        )
        if not np.allclose(
            np.asarray(configured_times, dtype=float),
            expression.columns.to_numpy(float),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("S8 settings times differ from expression columns.")
        validation["settings"] = {
            "status": "brain_roi_and_times_match",
            "formal_linkage": settings.get("api_settings", {}).get(
                "profile_linkage_method"
            ),
            "formal_normalization": settings.get("api_settings", {}).get(
                "profile_normalization"
            ),
            "used_as_compute_source": False,
        }
    else:
        validation["settings"] = {"status": "not_supplied"}
    return identities, validation


def load_brain_gene_notebook_inputs(
    s8_dir: str | Path,
) -> BrainGeneNotebookInputs:
    """Load the corrected formal S8 matrix and validate companion provenance."""

    source_dir = Path(s8_dir).expanduser().resolve()
    if not source_dir.is_dir():
        raise NotADirectoryError(source_dir)
    expression_path = source_dir / "brain_hvg_mean_log1p_by_time.csv"
    if not expression_path.is_file():
        raise FileNotFoundError(expression_path)
    expression = _load_expression(expression_path)
    identities, validation = _validate_formal_companions(source_dir, expression)
    return BrainGeneNotebookInputs(
        expression=expression,
        s8_dir=source_dir,
        input_identities=identities,
        validation=validation,
    )


def _build_pattern_curves(
    zscore: pd.DataFrame,
    assignments: pd.DataFrame,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for cluster_id, genes in assignments.groupby("cluster"):
        sub = zscore.loc[genes["gene"].tolist()]
        mean_curve = sub.mean(axis=0)
        std_curve = sub.std(axis=0)
        for time, mean, std in zip(zscore.columns, mean_curve, std_curve):
            records.append(
                {
                    "cluster": int(cluster_id),
                    "time": float(time),
                    "mean": float(mean),
                    "std": float(std),
                    "n_genes": int(sub.shape[0]),
                }
            )
    return pd.DataFrame(records)


def _rank_pattern_genes(
    zscore: pd.DataFrame,
    assignments: pd.DataFrame,
    *,
    representative_top_n: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    representative_rows: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    for cluster_id, cluster_assignments in assignments.groupby("cluster"):
        genes = cluster_assignments["gene"].tolist()
        sub = zscore.loc[genes]
        prototype = sub.mean(axis=0)
        concordance = sub.T.corrwith(prototype, axis=0).fillna(0.0)
        dynamic = sub.var(axis=1).fillna(0.0)
        score = 0.7 * concordance.rank(pct=True) + 0.3 * dynamic.rank(pct=True)
        ranked = pd.DataFrame(
            {
                "gene": genes,
                "cluster": int(cluster_id),
                "score": score.reindex(genes).fillna(0.0).to_numpy(),
                "variance": dynamic.reindex(genes).fillna(0.0).to_numpy(),
                "corr_with_pattern": (
                    concordance.reindex(genes).fillna(0.0).to_numpy()
                ),
                "peak_time": (
                    sub.idxmax(axis=1).reindex(genes).astype(float).to_numpy()
                ),
            }
        ).sort_values(
            ["score", "variance", "gene"],
            ascending=[False, False, True],
            kind="stable",
        )
        representative_rows.append(ranked.head(int(representative_top_n)))
        summary_rows.append(
            {
                "cluster": int(cluster_id),
                "n_genes_total": int(len(genes)),
                "n_genes_displayed": int(
                    min(len(genes), PATTERN_GENES_PER_CLUSTER)
                ),
                "headline_genes": ", ".join(
                    ranked.head(PATTERN_LABEL_TOP_N)["gene"].tolist()
                ),
                "prototype_peak_time": float(prototype.idxmax()),
            }
        )
    representatives = pd.concat(representative_rows, ignore_index=True)
    return representatives, pd.DataFrame(summary_rows)


def _choose_contiguous_phase_labels(
    ordered: pd.DataFrame,
    *,
    n_phases: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Exact fixed-k dynamic program from notebook cell 4."""

    n_rows = int(ordered.shape[0])
    k = int(n_phases)
    if k < 1 or n_rows < k:
        raise ValueError("Wave phase count must be positive and no larger than genes.")
    values = ordered.to_numpy(dtype=float)
    prefix = np.vstack([np.zeros((1, values.shape[1])), np.cumsum(values, axis=0)])
    prefix_sq = np.vstack(
        [np.zeros((1, values.shape[1])), np.cumsum(values**2, axis=0)]
    )

    def segment_cost(start: int, end: int) -> float:
        length = end - start
        segment_sum = prefix[end] - prefix[start]
        segment_sq = prefix_sq[end] - prefix_sq[start]
        mean = segment_sum / max(length, 1)
        return float(
            np.sum(
                segment_sq
                - 2.0 * mean * segment_sum
                + length * (mean**2)
            )
        )

    dp = np.full((k + 1, n_rows + 1), np.inf)
    previous = np.full((k + 1, n_rows + 1), -1, dtype=int)
    dp[0, 0] = 0.0
    min_size = max(1, n_rows // (k * 4) if n_rows >= 20 else 1)
    for phase in range(1, k + 1):
        for end in range(phase * min_size, n_rows + 1):
            start_min = (phase - 1) * min_size
            start_max = end - min_size
            for start in range(start_min, start_max + 1):
                candidate = dp[phase - 1, start] + segment_cost(start, end)
                if candidate < dp[phase, end]:
                    dp[phase, end] = candidate
                    previous[phase, end] = start
    if not np.isfinite(dp[k, n_rows]):
        raise RuntimeError("Notebook wave phase dynamic program found no solution.")
    labels = np.zeros(n_rows, dtype=int)
    end = n_rows
    boundaries: list[dict[str, int]] = []
    for phase in range(k, 0, -1):
        start = int(previous[phase, end])
        if start < 0:
            raise RuntimeError("Notebook wave phase traceback failed.")
        labels[start:end] = phase
        boundaries.append(
            {"phase": int(phase), "start": start, "end_exclusive": int(end)}
        )
        end = start
    boundaries.reverse()
    metrics = pd.DataFrame(
        [
            {
                "k": k,
                "objective": float(dp[k, n_rows]),
                "mean_segment_size": float(n_rows / k),
                "minimum_segment_size": int(min_size),
                "phase_boundaries_json": json.dumps(
                    boundaries, separators=(",", ":")
                ),
            }
        ]
    )
    return labels, metrics


def compute_brain_gene_notebook_state(
    expression: pd.DataFrame,
) -> BrainGeneNotebookState:
    """Recompute the fixed notebook k=2 programs and k=3 wave phases."""

    from scipy.cluster.hierarchy import fcluster, linkage

    table = pd.DataFrame(expression).copy()
    zscore = (
        table.sub(table.mean(axis=1), axis=0)
        .div(table.std(axis=1).replace(0, np.nan), axis=0)
        .fillna(0.0)
    )
    gene_variance = table.var(axis=1).sort_values(ascending=False)
    variance_rank = pd.DataFrame(
        {
            "rank": np.arange(1, len(gene_variance) + 1, dtype=np.int64),
            "gene": gene_variance.index.astype(str),
            "variance": gene_variance.to_numpy(float),
        }
    )
    hierarchy = linkage(
        zscore.to_numpy(dtype=float), method="average", metric="euclidean"
    )
    labels = fcluster(
        hierarchy, TEMPORAL_N_CLUSTERS, criterion="maxclust"
    ).astype(int)
    found = sorted(np.unique(labels).tolist())
    if len(found) != TEMPORAL_N_CLUSTERS:
        raise RuntimeError(
            "Notebook average-linkage fcluster did not produce exactly k=2."
        )
    assignments = pd.DataFrame(
        {"gene": zscore.index.astype(str), "cluster": labels}
    ).sort_values(["cluster", "gene"], kind="stable")
    assignments = assignments.reset_index(drop=True)
    curves = _build_pattern_curves(zscore, assignments)
    representatives, cluster_summary = _rank_pattern_genes(
        zscore,
        assignments,
        representative_top_n=REPRESENTATIVE_GENES_PER_PATTERN,
    )

    wave_genes = gene_variance.head(WAVE_TOP_N).index.tolist()
    wave_base = zscore.loc[wave_genes]
    wave_peak_meta = pd.DataFrame(
        {
            "gene": wave_base.index.astype(str),
            "peak_time": wave_base.idxmax(axis=1).astype(float).to_numpy(),
            "amplitude": (
                wave_base.max(axis=1) - wave_base.min(axis=1)
            ).to_numpy(float),
        }
    ).sort_values(
        ["peak_time", "amplitude", "gene"],
        ascending=[True, False, True],
        kind="stable",
    )
    wave_peak_meta = wave_peak_meta.reset_index(drop=True)
    wave_ordered = wave_base.loc[wave_peak_meta["gene"].tolist()]
    wave_labels, wave_metrics = _choose_contiguous_phase_labels(
        wave_ordered, n_phases=WAVE_N_PHASES
    )
    wave_assignments = wave_peak_meta.copy()
    wave_assignments["cluster"] = wave_labels
    wave_assignments.insert(
        0, "wave_order", np.arange(len(wave_assignments), dtype=np.int64)
    )
    wave_matrix = wave_ordered.loc[wave_assignments["gene"].tolist()]
    wave_summary = pd.DataFrame(
        [
            {
                "cluster": int(cluster_id),
                "n_genes": int(len(group)),
                "top_preview": ", ".join(group["gene"].head(8).tolist()),
                "peak_time_min": float(group["peak_time"].min()),
                "peak_time_max": float(group["peak_time"].max()),
            }
            for cluster_id, group in wave_assignments.groupby("cluster")
        ]
    ).sort_values("cluster")
    wave_summary = wave_summary.reset_index(drop=True)
    if wave_summary["cluster"].tolist() != [1, 2, 3]:
        raise RuntimeError("Notebook wave segmentation did not produce phases 1, 2, 3.")
    return BrainGeneNotebookState(
        expression=table,
        zscore=zscore,
        variance_rank=variance_rank,
        assignments=assignments,
        curves=curves,
        representatives=representatives,
        cluster_summary=cluster_summary,
        wave_matrix=wave_matrix,
        wave_assignments=wave_assignments,
        wave_metrics=wave_metrics,
        wave_summary=wave_summary,
    )


def _set_sparse_time_ticks(
    ax,
    time_values: Sequence[float],
    *,
    max_labels: int = 7,
    rotation: int = 35,
) -> np.ndarray:
    from matplotlib.ticker import FixedLocator

    times = np.asarray(time_values, dtype=float)
    if len(times) <= int(max_labels):
        positions = np.arange(len(times), dtype=int)
    else:
        positions = np.unique(
            np.linspace(0, len(times) - 1, int(max_labels), dtype=int)
        )
    ax.xaxis.set_major_locator(FixedLocator(positions))
    ax.set_xticklabels(
        [f"{times[index]:.2f}" for index in positions],
        rotation=int(rotation),
        ha="right" if rotation else "center",
        fontsize=8,
    )
    return positions


def _save_figure_pair(
    fig,
    *,
    output_dir: Path,
    stem: str,
) -> dict[str, Path]:
    paths = {
        "png": output_dir / f"{stem}.png",
        "svg": output_dir / f"{stem}.svg",
    }
    fig.savefig(
        paths["png"],
        format="png",
        dpi=300,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    fig.savefig(
        paths["svg"],
        format="svg",
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    return paths


def _render_heatmap(
    state: BrainGeneNotebookState,
    *,
    figure_dir: Path,
) -> dict[str, Path]:
    import matplotlib.pyplot as plt
    import seaborn as sns

    top_genes = state.variance_rank.head(HEATMAP_TOP_N)["gene"].tolist()
    compact = state.expression.loc[top_genes]
    compact_z = (
        compact.sub(compact.mean(axis=1), axis=0)
        .div(compact.std(axis=1).replace(0, np.nan), axis=0)
        .fillna(0.0)
    )
    fig, ax = plt.subplots(figsize=(8.5, 9.0))
    sns.heatmap(
        compact_z,
        cmap="RdBu_r",
        center=0,
        vmin=-2,
        vmax=2,
        ax=ax,
        cbar_kws={"label": "Gene-wise z-score", "shrink": 0.4},
    )
    ax.set_title(f"Brain top {HEATMAP_TOP_N} variable genes")
    ax.set_xlabel("Time")
    ax.set_ylabel("Gene")
    plt.tight_layout()
    outputs = _save_figure_pair(
        fig, output_dir=figure_dir, stem=FIGURE_STEMS["heatmap"]
    )
    plt.close(fig)
    return outputs


def _render_top_variable_trajectories(
    state: BrainGeneNotebookState,
    *,
    figure_dir: Path,
) -> dict[str, Path]:
    import matplotlib.pyplot as plt
    import seaborn as sns

    genes = state.variance_rank.head(TRAJECTORY_TOP_N)["gene"].tolist()
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    palette = sns.color_palette("tab10", n_colors=len(genes))
    for color, gene in zip(palette, genes):
        ax.plot(
            state.expression.columns,
            state.expression.loc[gene],
            marker="o",
            linewidth=1.8,
            color=color,
            label=gene,
        )
    ax.set_title("Top variable gene trajectories")
    ax.set_xlabel("Time")
    ax.set_ylabel("Mean reconstructed log1p")
    ax.grid(True, alpha=0.2, linestyle="--")
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
    )
    plt.tight_layout()
    outputs = _save_figure_pair(
        fig,
        output_dir=figure_dir,
        stem=FIGURE_STEMS["top_variable_trajectories"],
    )
    plt.close(fig)
    return outputs


def _render_temporal_programs(
    state: BrainGeneNotebookState,
    *,
    figure_dir: Path,
) -> dict[str, Path]:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    import seaborn as sns

    times = state.expression.columns.to_numpy(dtype=float)
    pattern_ids = sorted(state.assignments["cluster"].astype(int).unique().tolist())
    fig = plt.figure(
        figsize=(3.35 * len(pattern_ids), 5.6),
        facecolor=PANEL_BG,
    )
    outer = GridSpec(
        2,
        len(pattern_ids),
        figure=fig,
        height_ratios=[1.0, 2.2],
        hspace=0.28,
        wspace=0.24,
    )
    heat_axes = []
    for panel_index, cluster_id in enumerate(pattern_ids):
        color = PATTERN_CLUSTER_COLORS[
            (int(cluster_id) - 1) % len(PATTERN_CLUSTER_COLORS)
        ]
        curve = state.curves.loc[
            state.curves["cluster"] == cluster_id
        ].sort_values("time")
        mean = curve["mean"].to_numpy(dtype=float)
        std = curve["std"].fillna(0.0).to_numpy(dtype=float)
        ax_top = fig.add_subplot(outer[0, panel_index])
        ax_top.plot(
            curve["time"],
            mean,
            color=color,
            linewidth=2.4,
            solid_capstyle="round",
        )
        ax_top.fill_between(
            curve["time"],
            mean - std,
            mean + std,
            color=color,
            alpha=0.14,
        )
        ax_top.scatter(
            curve["time"],
            curve["mean"],
            color=color,
            s=18,
            zorder=3,
            edgecolor=PANEL_BG,
            linewidth=0.45,
        )
        ax_top.axhline(0, color=TEXT_MID, linewidth=0.8, alpha=0.32)
        ax_top.set_title(
            f"Pattern {cluster_id}",
            loc="left",
            fontsize=12,
            color=TEXT_DARK,
        )
        ax_top.set_xlabel("")
        ax_top.set_ylabel("Program z")
        ax_top.set_facecolor(PANEL_BG)
        ax_top.grid(
            True,
            axis="y",
            alpha=0.22,
            linestyle="-",
            color=GRID_COLOR,
        )
        if panel_index > 0:
            ax_top.set_ylabel("")
            ax_top.set_yticklabels([])
        ax_top.tick_params(axis="x", labelbottom=False)
        ax_top.tick_params(axis="y", labelsize=8, colors=TEXT_MID)
        summary = state.cluster_summary.loc[
            state.cluster_summary["cluster"] == cluster_id
        ].iloc[0]
        ax_top.text(
            0.01,
            0.05,
            (
                f"peak {summary['prototype_peak_time']:.2f}\n"
                f"{int(summary['n_genes_total'])} genes"
            ),
            transform=ax_top.transAxes,
            ha="left",
            va="bottom",
            fontsize=8,
            color=TEXT_MID,
        )

        representative_genes = state.representatives.loc[
            state.representatives["cluster"] == cluster_id, "gene"
        ].tolist()
        representative_heat = state.zscore.loc[representative_genes]
        representative_heat = (
            representative_heat.assign(
                _peak=representative_heat.idxmax(axis=1).astype(float)
            )
            .sort_values(["_peak"], kind="stable")
            .drop(columns="_peak")
        )
        ax_bottom = fig.add_subplot(outer[1, panel_index])
        sns.heatmap(
            representative_heat,
            ax=ax_bottom,
            cmap=PATTERN_CMAP_NAME,
            center=0,
            vmin=-PATTERN_VALUE_CLIP,
            vmax=PATTERN_VALUE_CLIP,
            cbar=False,
            linewidths=0.5,
            linecolor="#f1ece4",
        )
        ax_bottom.set_title(
            ", ".join(representative_heat.index[:3]),
            fontsize=8.5,
            loc="left",
            pad=8,
            color=TEXT_MID,
        )
        ax_bottom.set_xlabel("Time")
        ax_bottom.set_ylabel("")
        ax_bottom.set_facecolor(PANEL_BG)
        ax_bottom.set_yticklabels(
            ax_bottom.get_yticklabels(),
            rotation=0,
            fontsize=8,
            fontstyle="italic",
            color=TEXT_DARK,
        )
        _set_sparse_time_ticks(
            ax_bottom,
            times,
            max_labels=6,
            rotation=35,
        )
        ax_bottom.tick_params(axis="x", colors=TEXT_MID)
        heat_axes.append(ax_bottom)

    scalar = mpl.cm.ScalarMappable(
        norm=mpl.colors.Normalize(
            vmin=-PATTERN_VALUE_CLIP,
            vmax=PATTERN_VALUE_CLIP,
        ),
        cmap=PATTERN_CMAP_NAME,
    )
    colorbar = fig.colorbar(
        scalar,
        ax=heat_axes,
        fraction=0.015,
        pad=0.02,
        aspect=25,
    )
    colorbar.set_label("Gene-wise z-score")
    colorbar.ax.tick_params(labelsize=8, colors=TEXT_MID)
    fig.suptitle(
        f"Brain temporal programs (k={TEMPORAL_N_CLUSTERS})",
        x=0.02,
        y=1.02,
        ha="left",
        fontsize=13.5,
        color=TEXT_DARK,
    )
    outputs = _save_figure_pair(
        fig,
        output_dir=figure_dir,
        stem=FIGURE_STEMS["temporal_programs"],
    )
    plt.close(fig)
    return outputs


def _render_developmental_wave_map(
    state: BrainGeneNotebookState,
    *,
    figure_dir: Path,
) -> dict[str, Path]:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.gridspec import GridSpec
    import seaborn as sns

    cluster_strip = (
        state.wave_assignments["cluster"].to_numpy(dtype=int) - 1
    ).reshape(-1, 1)
    phase_ids = state.wave_assignments["cluster"].unique().tolist()
    phase_colors = [
        PATTERN_CLUSTER_COLORS[(int(cluster_id) - 1) % len(PATTERN_CLUSTER_COLORS)]
        for cluster_id in phase_ids
    ]
    fig = plt.figure(figsize=(10.8, 6.1), facecolor=PANEL_BG)
    grid = GridSpec(
        1,
        2,
        figure=fig,
        width_ratios=[0.28, 10.0],
        wspace=0.04,
    )
    ax_strip = fig.add_subplot(grid[0, 0])
    ax = fig.add_subplot(grid[0, 1])
    ax_strip.imshow(
        cluster_strip,
        aspect="auto",
        cmap=ListedColormap(phase_colors),
    )
    ax_strip.set_xticks([])
    ax_strip.set_yticks([])
    ax_strip.set_title("Phase", fontsize=9, color=TEXT_MID, pad=6)
    ax_strip.set_facecolor(PANEL_BG)
    sns.heatmap(
        state.wave_matrix,
        ax=ax,
        cmap=PATTERN_CMAP_NAME,
        center=0,
        vmin=-PATTERN_VALUE_CLIP,
        vmax=PATTERN_VALUE_CLIP,
        cbar_kws={
            "shrink": 0.34,
            "pad": 0.02,
            "label": "Gene-wise z-score",
        },
        yticklabels=False,
        linewidths=0.0,
    )
    ax.set_title(
        (
            "Developmental wave map ordered by peak time with phase "
            f"annotation (k={WAVE_N_PHASES})"
        ),
        loc="left",
        fontsize=13,
        color=TEXT_DARK,
    )
    ax.set_xlabel("Time")
    ax.set_ylabel("")
    ax.set_facecolor(PANEL_BG)
    fig.text(
        0.032,
        0.47,
        f"Top {WAVE_TOP_N} dynamic genes",
        rotation=90,
        va="center",
        ha="center",
        fontsize=10,
        color=TEXT_MID,
    )
    times = state.expression.columns.to_numpy(dtype=float)
    _set_sparse_time_ticks(ax, times, max_labels=7, rotation=35)
    ax.tick_params(axis="x", colors=TEXT_MID)
    for peak_time in sorted(state.wave_assignments["peak_time"].unique()):
        # The notebook deliberately draws at heatmap cell boundaries, not +0.5.
        ax.axvline(
            np.searchsorted(times, peak_time),
            color="#efe9df",
            linewidth=0.7,
            alpha=0.8,
        )
    phase_starts = [
        int(group.index.min())
        for _, group in state.wave_assignments.groupby("cluster", sort=True)
    ]
    for boundary in phase_starts[1:]:
        ax.axhline(boundary, color="#ece6dd", linewidth=1.0)
        ax_strip.axhline(boundary, color="#ece6dd", linewidth=1.0)
    ax.collections[0].colorbar.ax.tick_params(labelsize=8, colors=TEXT_MID)
    plt.tight_layout(rect=(0.05, 0.0, 1.0, 1.0))
    outputs = _save_figure_pair(
        fig,
        output_dir=figure_dir,
        stem=FIGURE_STEMS["developmental_wave_map"],
    )
    plt.close(fig)
    return outputs


def _render_pattern_level_curves(
    state: BrainGeneNotebookState,
    *,
    figure_dir: Path,
) -> dict[str, Path]:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.0, 4.1), facecolor=PANEL_BG)
    for cluster_id, subset in state.curves.groupby("cluster"):
        color = PATTERN_CLUSTER_COLORS[
            (int(cluster_id) - 1) % len(PATTERN_CLUSTER_COLORS)
        ]
        ax.plot(
            subset["time"],
            subset["mean"],
            marker="o",
            markersize=4.2,
            linewidth=2.2,
            color=color,
            label=f"Pattern {cluster_id}",
        )
        ax.fill_between(
            subset["time"],
            subset["mean"] - subset["std"],
            subset["mean"] + subset["std"],
            color=color,
            alpha=0.15,
        )
    ax.set_title("Pattern-level temporal curves", loc="left", color=TEXT_DARK)
    ax.set_xlabel("Time")
    ax.set_ylabel("Mean z-score")
    ax.axhline(0, color=TEXT_MID, linewidth=0.8, alpha=0.4)
    ax.set_facecolor(PANEL_BG)
    _set_sparse_time_ticks(
        ax,
        np.asarray(sorted(state.curves["time"].unique()), dtype=float),
        max_labels=7,
        rotation=35,
    )
    ax.tick_params(axis="x", colors=TEXT_MID)
    ax.tick_params(axis="y", colors=TEXT_MID)
    ax.grid(True, axis="y", alpha=0.2, linestyle="-", color=GRID_COLOR)
    ax.legend(frameon=False, labelcolor=TEXT_DARK)
    plt.tight_layout()
    outputs = _save_figure_pair(
        fig,
        output_dir=figure_dir,
        stem=FIGURE_STEMS["pattern_level_curves"],
    )
    plt.close(fig)
    return outputs


def _write_tables(
    state: BrainGeneNotebookState,
    *,
    table_dir: Path,
) -> dict[str, Path]:
    tables = {
        "notebook_sample_zscore": table_dir / "brain_notebook_sample_zscore.csv",
        "notebook_variance_rank": (
            table_dir / "brain_notebook_temporal_variance_rank.csv"
        ),
        "notebook_average_k2_assignments": (
            table_dir / "brain_notebook_average_k2_assignments.csv"
        ),
        "notebook_average_k2_curves": (
            table_dir / "brain_notebook_average_k2_curves.csv"
        ),
        "notebook_average_k2_representatives": (
            table_dir / "brain_notebook_average_k2_representatives_top5.csv"
        ),
        "notebook_average_k2_summary": (
            table_dir / "brain_notebook_average_k2_summary.csv"
        ),
        "notebook_wave_k3_assignments": (
            table_dir / "brain_notebook_wave_k3_assignments.csv"
        ),
        "notebook_wave_k3_metrics": (
            table_dir / "brain_notebook_wave_k3_metrics.csv"
        ),
        "notebook_wave_k3_summary": (
            table_dir / "brain_notebook_wave_k3_summary.csv"
        ),
    }
    state.zscore.to_csv(tables["notebook_sample_zscore"])
    state.variance_rank.to_csv(tables["notebook_variance_rank"], index=False)
    state.assignments.to_csv(
        tables["notebook_average_k2_assignments"], index=False
    )
    state.curves.to_csv(tables["notebook_average_k2_curves"], index=False)
    state.representatives.to_csv(
        tables["notebook_average_k2_representatives"], index=False
    )
    state.cluster_summary.to_csv(
        tables["notebook_average_k2_summary"], index=False
    )
    state.wave_assignments.to_csv(
        tables["notebook_wave_k3_assignments"], index=False
    )
    state.wave_metrics.to_csv(tables["notebook_wave_k3_metrics"], index=False)
    state.wave_summary.to_csv(tables["notebook_wave_k3_summary"], index=False)
    return tables


def _verify_notebook_oracle(
    notebook_oracle: str | Path | None,
) -> dict[str, Any]:
    audit = {
        "relative_path": NOTEBOOK_ORACLE,
        "expected_sha256": NOTEBOOK_ORACLE_SHA256,
        "cells": NOTEBOOK_ORACLE_CELLS,
        "runtime_file_supplied": notebook_oracle is not None,
        "runtime_sha256_verified": False,
    }
    if notebook_oracle is None:
        return audit
    path = Path(notebook_oracle).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    identity = _file_identity(path)
    if identity["sha256"] != NOTEBOOK_ORACLE_SHA256:
        raise ValueError(
            "Notebook oracle SHA-256 differs from the frozen reviewed notebook."
        )
    audit["runtime_file"] = identity
    audit["runtime_sha256_verified"] = True
    return audit


def render_brain_gene_review_notebook_style(
    *,
    s8_dir: str | Path,
    output_dir: str | Path,
    analysis_profile: str,
    notebook_oracle: str | Path | None = None,
) -> dict[str, Path]:
    """Recompute current S8 values and render all five notebook-style groups."""

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import scipy
    import seaborn as sns

    profile = str(analysis_profile).strip()
    profile_slug = _profile_slug(profile)
    oracle_audit = _verify_notebook_oracle(notebook_oracle)
    inputs = load_brain_gene_notebook_inputs(s8_dir)
    state = compute_brain_gene_notebook_state(inputs.expression)
    out_dir = Path(output_dir).expanduser().resolve()
    figure_dir = out_dir / "figures"
    table_dir = out_dir / "tables"
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    with mpl.rc_context():
        sns.set_theme(style="white", context="paper")
        plt.rcParams["figure.dpi"] = 160
        plt.rcParams["savefig.dpi"] = 300
        plt.rcParams["axes.spines.top"] = False
        plt.rcParams["axes.spines.right"] = False
        plt.rcParams["axes.linewidth"] = 0.8
        plt.rcParams["xtick.major.width"] = 0.8
        plt.rcParams["ytick.major.width"] = 0.8
        plt.rcParams["font.size"] = 10
        figures = {
            "heatmap": _render_heatmap(state, figure_dir=figure_dir),
            "top_variable_trajectories": _render_top_variable_trajectories(
                state, figure_dir=figure_dir
            ),
            "temporal_programs": _render_temporal_programs(
                state, figure_dir=figure_dir
            ),
            "developmental_wave_map": _render_developmental_wave_map(
                state, figure_dir=figure_dir
            ),
            "pattern_level_curves": _render_pattern_level_curves(
                state, figure_dir=figure_dir
            ),
        }
    tables = _write_tables(state, table_dir=table_dir)

    cluster_sizes = {
        str(int(cluster)): int(count)
        for cluster, count in state.assignments.groupby("cluster").size().items()
    }
    cluster_peaks = {
        str(int(row.cluster)): float(row.prototype_peak_time)
        for row in state.cluster_summary.itertuples(index=False)
    }
    wave_sizes = {
        str(int(cluster)): int(count)
        for cluster, count in state.wave_assignments.groupby("cluster").size().items()
    }
    figure_manifest = {
        group: {
            suffix: {
                "relative_path": str(path.relative_to(out_dir)),
                "sha256": _sha256(path),
            }
            for suffix, path in paths.items()
        }
        for group, paths in figures.items()
    }
    table_manifest = {
        name: {
            "relative_path": str(path.relative_to(out_dir)),
            "sha256": _sha256(path),
        }
        for name, path in tables.items()
    }
    audit_path = out_dir / "brain_gene_review_notebook_style_audit.json"
    manifest_path = out_dir / "brain_gene_review_notebook_style_manifest.json"
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "workflow": "mosta_brain_gene_review_notebook_style_render_only",
        "analysis_profile": profile,
        "analysis_profile_slug": profile_slug,
        "source_s8_dir": str(inputs.s8_dir),
        "source_mean_log1p": inputs.input_identities["mean_log1p"],
        "source_companions": {
            key: identity
            for key, identity in inputs.input_identities.items()
            if key != "mean_log1p"
        },
        "notebook_oracle": {
            "relative_path": NOTEBOOK_ORACLE,
            "expected_sha256": NOTEBOOK_ORACLE_SHA256,
            "runtime_sha256_verified": oracle_audit["runtime_sha256_verified"],
            "cells": NOTEBOOK_ORACLE_CELLS,
        },
        "render_contract": {
            "historical_figure_values_loaded": False,
            "historical_assignment_tables_loaded": False,
            "formal_ward_assignments_used": False,
            "formal_population_zscore_used": False,
            "result_cluster_and_phase_sizes_hardcoded": False,
            "result_size_source": (
                "groupby counts of freshly recomputed average-k2 and wave-k3 "
                "assignments"
            ),
        },
        "calculation_profile": {
            "zscore": "pandas gene-wise sample standardization ddof=1",
            "variance": "pandas temporal sample variance ddof=1",
            "temporal_programs": (
                "SciPy average-linkage Euclidean fcluster maxclust k=2; "
                "raw cluster IDs retained exactly as notebook"
            ),
            "developmental_wave": (
                "top-1000 variance genes; peak/amplitude/gene stable order; "
                "contiguous within-segment SSE dynamic program k=3"
            ),
        },
        "result_summary": {
            "temporal_cluster_sizes_recomputed": cluster_sizes,
            "temporal_cluster_prototype_peak_times_recomputed": cluster_peaks,
            "wave_phase_sizes_recomputed": wave_sizes,
            "hardcoded": False,
        },
        "figures": figure_manifest,
        "tables": table_manifest,
        "audit": str(audit_path.relative_to(out_dir)),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    audit = {
        "schema_version": 1,
        "status": "complete",
        "analysis_profile": profile,
        "analysis_profile_slug": profile_slug,
        "render_contract": {
            "mode": "render_only_from_current_corrected_formal_s8_tables",
            "model_loaded": False,
            "simulation_run": False,
            "classifier_run": False,
            "historical_figure_values_loaded": False,
            "historical_assignment_tables_loaded": False,
            "formal_ward_assignments_used": False,
            "formal_population_zscore_used": False,
            "mean_log1p_expression_is_numeric_source": True,
            "result_cluster_and_phase_sizes_hardcoded": False,
            "result_size_source": (
                "groupby counts of freshly recomputed average-k2 and wave-k3 "
                "assignments"
            ),
        },
        "notebook_oracle": oracle_audit,
        "calculation_contract": {
            "input_shape": [
                int(state.expression.shape[0]),
                int(state.expression.shape[1]),
            ],
            "time_points": [float(value) for value in state.expression.columns],
            "sample_zscore_ddof": 1,
            "temporal_variance_ddof": 1,
            "temporal_linkage": "average",
            "temporal_distance": "euclidean",
            "temporal_cut": "scipy_fcluster_maxclust",
            "temporal_requested_k": TEMPORAL_N_CLUSTERS,
            "temporal_raw_cluster_ids_retained": True,
            "temporal_cluster_sizes": cluster_sizes,
            "temporal_cluster_prototype_peak_times": cluster_peaks,
            "representative_score": (
                "0.7 * percentile_rank(Pearson correlation to prototype) + "
                "0.3 * percentile_rank(temporal variance)"
            ),
            "pattern_genes_per_cluster_computed_by_notebook": (
                PATTERN_GENES_PER_CLUSTER
            ),
            "pattern_genes_per_cluster_role": (
                "computed notebook summary only; not the representative "
                "heatmap rows"
            ),
            "representatives_per_pattern": REPRESENTATIVE_GENES_PER_PATTERN,
            "wave_top_genes": int(len(state.wave_matrix)),
            "wave_requested_top_genes": WAVE_TOP_N,
            "wave_order": "peak_time asc, amplitude desc, gene asc; stable",
            "wave_requested_k": WAVE_N_PHASES,
            "wave_phase_sizes": wave_sizes,
            "wave_dynamic_program": state.wave_metrics.iloc[0].to_dict(),
            "display_smoothing_sigma": DISPLAY_SMOOTH_SIGMA,
        },
        "style_contract": {
            "seaborn_theme": {"style": "white", "context": "paper"},
            "figure_dpi": 160,
            "savefig_dpi": 300,
            "top_and_right_spines": False,
            "axes_and_tick_width": 0.8,
            "font_size": 10,
            "heatmap": {
                "top_n": HEATMAP_TOP_N,
                "figsize": [8.5, 9.0],
                "cmap": "RdBu_r",
                "limits": [-2.0, 2.0],
            },
            "top_variable_trajectories": {
                "top_n": TRAJECTORY_TOP_N,
                "figsize": [8.0, 4.5],
                "palette": "seaborn tab10 with n_colors=25",
            },
            "temporal_programs": {
                "figsize": [6.7, 5.6],
                "cmap": PATTERN_CMAP_NAME,
                "limits": [-PATTERN_VALUE_CLIP, PATTERN_VALUE_CLIP],
                "cluster_colors": list(PATTERN_CLUSTER_COLORS[:2]),
            },
            "developmental_wave_map": {
                "figsize": [10.8, 6.1],
                "cmap": PATTERN_CMAP_NAME,
                "limits": [-PATTERN_VALUE_CLIP, PATTERN_VALUE_CLIP],
                "vertical_line_coordinate": (
                    "numpy.searchsorted(time); notebook heatmap-cell boundary"
                ),
            },
            "pattern_level_curves": {
                "figsize": [8.0, 4.1],
                "cluster_colors": list(PATTERN_CLUSTER_COLORS[:2]),
            },
        },
        "input_validation": inputs.validation,
        "inputs": inputs.input_identities,
        "software": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": mpl.__version__,
            "seaborn": sns.__version__,
            "scipy": scipy.__version__,
        },
        "outputs": {
            "figures": {
                group: {
                    suffix: _file_identity(path)
                    for suffix, path in paths.items()
                }
                for group, paths in figures.items()
            },
            "tables": {
                name: _file_identity(path) for name, path in tables.items()
            },
            "manifest": _file_identity(manifest_path),
        },
    }
    audit_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    outputs: dict[str, Path] = {
        "manifest": manifest_path,
        "audit": audit_path,
    }
    for group, paths in figures.items():
        for suffix, path in paths.items():
            outputs[f"{group}_{suffix}"] = path
    for name, path in tables.items():
        outputs[f"table_{name}"] = path
    return outputs


__all__ = [
    "BrainGeneNotebookInputs",
    "BrainGeneNotebookState",
    "FIGURE_STEMS",
    "NOTEBOOK_ORACLE",
    "NOTEBOOK_ORACLE_SHA256",
    "PATTERN_CLUSTER_COLORS",
    "TEMPORAL_N_CLUSTERS",
    "WAVE_N_PHASES",
    "compute_brain_gene_notebook_state",
    "load_brain_gene_notebook_inputs",
    "render_brain_gene_review_notebook_style",
]
