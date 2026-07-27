#!/usr/bin/env python3
"""Run the Reviewer #4 axis-specific LR gene counterfactual.

The fixed primary axis is ``cxcl12a -> cxcr4a``.  The default design uses
observed anchors 0->1 and 3->4, ligand knockdown fractions 0.25/0.5/1.0,
receptor and dual sensitivity runs, and 100 model-visible HVG shams matched on
detection fraction, mean expression, and PCA-loading norm within each anchor's
baseline ligand-positive fixed-sender compartment.

This is a trained-model sensitivity analysis.  It does not estimate
experimental or biological causality.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
from importlib import metadata
import json
import math
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd


DEFAULT_ANCHORS = ((0.0, 1.0), (3.0, 4.0))
DEFAULT_FRACTIONS = (0.25, 0.5, 1.0)
DEFAULT_GROUPING_SEEDS = (101, 202, 303, 404, 505)


def _csv_floats(value: str) -> tuple[float, ...]:
    parsed = tuple(float(token.strip()) for token in value.split(",") if token.strip())
    if not parsed or len(set(parsed)) != len(parsed):
        raise argparse.ArgumentTypeError("Expected unique comma-separated floats.")
    return parsed


def _csv_ints(value: str) -> tuple[int, ...]:
    parsed = tuple(int(token.strip()) for token in value.split(",") if token.strip())
    if not parsed or len(set(parsed)) != len(parsed):
        raise argparse.ArgumentTypeError("Expected unique comma-separated integers.")
    return parsed


def _anchors(value: str) -> tuple[tuple[float, float], ...]:
    parsed: list[tuple[float, float]] = []
    for token in value.split(","):
        fields = token.strip().split(":")
        if len(fields) != 2:
            raise argparse.ArgumentTypeError(
                "Anchors must be comma-separated start:end pairs."
            )
        start, end = map(float, fields)
        if not end > start:
            raise argparse.ArgumentTypeError("Each anchor end must exceed its start.")
        parsed.append((start, end))
    if not parsed or len(set(parsed)) != len(parsed):
        raise argparse.ArgumentTypeError("Anchors must be non-empty and unique.")
    return tuple(parsed)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5ad", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--ligand", default="cxcl12a")
    parser.add_argument("--receptor", default="cxcr4a")
    parser.add_argument("--anchors", type=_anchors, default=DEFAULT_ANCHORS)
    parser.add_argument("--fractions", type=_csv_floats, default=DEFAULT_FRACTIONS)
    parser.add_argument("--n-shams", type=int, default=100)
    parser.add_argument("--time-key", default="time_point_processed")
    parser.add_argument("--cell-type-key", default="Annotation")
    parser.add_argument("--spatial-key", default="spatial_aligned")
    parser.add_argument("--state-key", default="X_latent")
    parser.add_argument("--loadings-key", default="PCs")
    parser.add_argument("--hvg-key", default="highly_variable")
    parser.add_argument("--pca-center-key", default="pca_center")
    parser.add_argument("--group-size", type=int, default=1024)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument(
        "--grouping-seeds",
        type=_csv_ints,
        default=DEFAULT_GROUPING_SEEDS,
        help=(
            "Technical spatial-GNN grouping seeds. The first seed is used for "
            "full deterministic rollouts; all seeds are used for the exact "
            "generic complete-message audit on fixed LR-conditioned support. "
            "Extra seeds do not repeat trajectory, Wasserstein, or mediation "
            "calculations."
        ),
    )
    parser.add_argument("--metric-seed", type=int, default=1701)
    parser.add_argument("--max-ot-points", type=int, default=1024)
    parser.add_argument("--receiver-threshold", type=float, default=0.0)
    parser.add_argument("--pca-contract-atol", type=float, default=5e-4)
    parser.add_argument("--pca-contract-rtol", type=float, default=5e-4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint-stage", default="Finetune")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    resolved = path.expanduser().resolve()
    if resolved.exists() and any(resolved.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory is not empty: {resolved}")
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _artifact(path: Path, *, root: Optional[Path] = None) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    label = str(resolved.relative_to(root)) if root is not None else str(resolved)
    return {
        "path": label,
        "bytes": int(resolved.stat().st_size),
        "sha256": _sha256(resolved),
    }


def _reproducibility_provenance() -> dict[str, Any]:
    script = Path(__file__).resolve()
    repository = script.parents[2]
    source_paths = (
        script,
        repository / "CytoBridge" / "tl" / "downstream" / "perturbation.py",
        repository / "CytoBridge" / "tl" / "downstream" / "ablation.py",
        repository / "CytoBridge" / "tl" / "downstream" / "evaluation.py",
        repository / "CytoBridge" / "tl" / "downstream" / "simulation.py",
        repository
        / "CytoBridge"
        / "tl"
        / "downstream"
        / "spatial_interaction_attribution.py",
    )
    source_files = {
        str(path.relative_to(repository)): _artifact(path, root=repository)
        for path in source_paths
    }
    source_fingerprint_material = "\n".join(
        f"{name}\0{record['sha256']}" for name, record in sorted(source_files.items())
    )
    dependency_versions: dict[str, str] = {}
    for distribution in (
        "anndata",
        "matplotlib",
        "numpy",
        "pandas",
        "POT",
        "scipy",
        "torch",
        "torch-geometric",
    ):
        try:
            dependency_versions[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            dependency_versions[distribution] = "not-installed"
    dependency_payload = {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "packages": dependency_versions,
    }
    dependency_material = json.dumps(
        dependency_payload,
        sort_keys=True,
        separators=(",", ":"),
    )
    git: dict[str, Any] = {
        "available": False,
        "commit": None,
        "dirty": None,
        "status_porcelain": None,
        "error": None,
    }
    try:
        commit = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        status = subprocess.run(
            [
                "git",
                "-C",
                str(repository),
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.splitlines()
        git.update(
            {
                "available": True,
                "commit": commit,
                "dirty": bool(status),
                "status_porcelain": status,
            }
        )
    except (OSError, subprocess.SubprocessError) as exc:
        git["error"] = f"{type(exc).__name__}: {exc}"
    return {
        "analysis_script": source_files[str(script.relative_to(repository))],
        "local_source_dependencies": source_files,
        "local_source_bundle_sha256": sha256(
            source_fingerprint_material.encode("utf-8")
        ).hexdigest(),
        "runtime_dependency_versions": dependency_payload,
        "runtime_dependency_version_fingerprint_sha256": sha256(
            dependency_material.encode("utf-8")
        ).hexdigest(),
        "python_executable": str(Path(sys.executable).resolve()),
        "git": git,
    }


def _model_digest(model) -> str:
    digest = sha256()
    state = getattr(model, "state_dict", lambda: {})()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _dense_column(matrix, index: int) -> np.ndarray:
    values = matrix[:, int(index)]
    if hasattr(values, "toarray"):
        values = values.toarray()
    return np.asarray(values, dtype=np.float64).reshape(-1)


def _resolve_gene(var_names: Sequence[object], requested: str) -> str:
    from CytoBridge.tl.downstream.temporal import simplify_gene_names

    names = tuple(str(name) for name in var_names)
    requested = str(requested)
    exact = [name for name in names if name.casefold() == requested.casefold()]
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        raise KeyError(
            f"Axis gene {requested!r} is ambiguous under case-insensitive matching; "
            f"observed {len(exact)} var_names."
        )
    simplified = simplify_gene_names(names)
    matches = simplified.loc[
        simplified["gene_symbol"].astype(str).str.casefold() == requested.casefold(),
        "var_name",
    ].astype(str)
    if len(matches) != 1:
        raise KeyError(
            f"Axis gene {requested!r} must resolve to exactly one var_name; "
            f"observed {len(matches)}."
        )
    return str(matches.iloc[0])


def _time_values(data, key: str) -> np.ndarray:
    from CytoBridge.tl.downstream.downstream_data import parse_time_value

    if key not in data.obs:
        raise KeyError(f"adata.obs is missing {key!r}.")
    values = np.asarray(
        [parse_time_value(value) for value in data.obs[key]], dtype=np.float64
    )
    if not np.isfinite(values).all():
        raise ValueError("Parsed time values contain non-finite values.")
    return values


def _validate_pca_contract(
    expression,
    state: np.ndarray,
    loadings: np.ndarray,
    center: np.ndarray,
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    center = np.asarray(center, dtype=np.float64).reshape(-1)
    loadings = np.asarray(loadings, dtype=np.float64)
    state = np.asarray(state, dtype=np.float64)
    if loadings.shape[0] != center.shape[0]:
        raise ValueError("PCA center and loadings are not feature aligned.")
    projected = np.asarray(expression @ loadings, dtype=np.float64)
    projected -= center @ loadings
    if projected.shape != state.shape:
        raise ValueError(
            f"PCA projection has shape {projected.shape}, state has {state.shape}."
        )
    residual = projected - state
    max_abs = float(np.max(np.abs(residual)))
    relative_l2 = float(np.linalg.norm(residual) / max(np.linalg.norm(state), 1e-12))
    passed = bool(np.allclose(projected, state, atol=float(atol), rtol=float(rtol)))
    if not passed:
        raise ValueError(
            "adata.obsm state is not the PCA transform bound by adata.X, "
            f"varm loadings, and the persisted center: max_abs={max_abs:.6g}, "
            f"relative_l2={relative_l2:.6g}."
        )
    return {
        "passed": True,
        "atol": float(atol),
        "rtol": float(rtol),
        "max_abs_residual": max_abs,
        "relative_l2_residual": relative_l2,
        "formula": "(expression - pca_center) @ loadings",
    }


def _annotate(
    table: pd.DataFrame,
    *,
    anchor_id: str,
    start: float,
    end: float,
    condition: str,
    condition_role: str,
    targets: Sequence[str],
    fraction: float,
    is_sham: bool,
    sham_rank: Optional[int],
    grouping_seed: int,
    edit_cell_mask_policy: str,
) -> pd.DataFrame:
    result = table.copy()
    prefix = {
        "anchor_id": anchor_id,
        "anchor_start": float(start),
        "anchor_end": float(end),
        "condition": condition,
        "condition_role": condition_role,
        "target_genes": ";".join(map(str, targets)),
        "knockdown_fraction": float(fraction),
        "is_sham": bool(is_sham),
        "sham_rank": sham_rank,
        "grouping_seed": int(grouping_seed),
        "edit_cell_mask_policy": str(edit_cell_mask_policy),
    }
    for offset, (name, value) in enumerate(prefix.items()):
        if name in result:
            result[name] = value
        else:
            result.insert(offset, name, value)
    return result


def _edge_set_summary(baseline, counterfactual) -> dict[str, Any]:
    keys = ["group_index", "source_index", "target_index"]
    left = baseline.edge_table.set_index(keys)
    right = counterfactual.edge_table.set_index(keys)
    common = left.index.intersection(right.index)
    added = right.index.difference(left.index)
    removed = left.index.difference(right.index)
    summary: dict[str, Any] = {
        "baseline_n_edges": int(len(left)),
        "counterfactual_n_edges": int(len(right)),
        "n_common_edges": int(len(common)),
        "n_added_edges": int(len(added)),
        "n_removed_edges": int(len(removed)),
        "counterfactual_edge_predictor_probability_mean": (
            float(right["edge_predictor_probability"].mean())
            if len(right)
            else float("nan")
        ),
        "counterfactual_attention_abs_mean": (
            float(right["attention_abs_mean"].mean()) if len(right) else float("nan")
        ),
        "counterfactual_complete_message_norm_joint_mean": (
            float(right["complete_message_norm_joint"].mean())
            if len(right)
            else float("nan")
        ),
    }
    for column in (
        "edge_predictor_probability",
        "attention_abs_mean",
        "complete_message_norm_joint",
        "complete_message_norm_spatial",
        "complete_message_norm_state",
    ):
        summary[f"common_edge_mean_delta_{column}"] = (
            float((right.loc[common, column] - left.loc[common, column]).mean())
            if len(common)
            else float("nan")
        )
    return summary


def _message_delta_summary(
    delta: np.ndarray,
    receiver_mask: np.ndarray,
    *,
    spatial_dim: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    spaces = {
        "joint": slice(0, delta.shape[1]),
        "spatial": slice(0, int(spatial_dim)),
        "state": slice(int(spatial_dim), delta.shape[1]),
    }
    for cohort, mask in {
        "all_cells": np.ones(delta.shape[0], dtype=bool),
        "fixed_receptor_positive_ligand_negative": receiver_mask,
    }.items():
        for space, columns in spaces.items():
            values = np.asarray(delta[mask, columns], dtype=np.float64)
            rows.append(
                {
                    "cohort": cohort,
                    "space": space,
                    "n_cells": int(mask.sum()),
                    "mean_complete_message_delta_norm": float(
                        np.mean(np.linalg.norm(values, axis=1))
                    ),
                    "centroid_complete_message_delta_norm": float(
                        np.linalg.norm(np.mean(values, axis=0))
                    ),
                }
            )
    return pd.DataFrame(rows)


def _descriptive_sham_comparison(
    distribution: pd.DataFrame,
    mediation: pd.DataFrame,
    target_message: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add_rows(
        table: pd.DataFrame,
        *,
        metrics: Sequence[str],
        group_columns: Sequence[str],
        family: str,
    ) -> None:
        primary = table.loc[table["condition_role"].eq("ligand_primary")]
        shams = table.loc[table["condition_role"].eq("matched_hvg_sham")]
        for _, observed in primary.iterrows():
            selector = np.ones(len(shams), dtype=bool)
            for column in group_columns:
                selector &= shams[column].eq(observed[column]).to_numpy()
            matched = shams.loc[selector]
            for metric in metrics:
                null = (
                    pd.to_numeric(matched[metric], errors="coerce").dropna().to_numpy()
                )
                value = float(observed[metric])
                if null.size == 0 or not np.isfinite(value):
                    continue
                rows.append(
                    {
                        "analysis_family": family,
                        **{column: observed[column] for column in group_columns},
                        "metric": metric,
                        "observed": value,
                        "n_matched_shams": int(null.size),
                        "sham_mean": float(np.mean(null)),
                        "sham_std": float(np.std(null, ddof=1))
                        if null.size > 1
                        else float("nan"),
                        "descriptive_sham_upper_tail_fraction": float(
                            np.mean(null >= value)
                        ),
                        "descriptive_sham_fraction_at_or_below_observed": float(
                            np.mean(null <= value)
                        ),
                        "descriptive_observed_relative_rank": float(
                            (1 + np.sum(null <= value)) / (null.size + 1)
                        ),
                        "descriptive_pseudocount_tail_score_not_formal_pvalue": float(
                            (1 + np.sum(null >= value)) / (null.size + 1)
                        ),
                        "null_selection": "deterministic_nearest_matched_HVGs",
                        "null_is_exchangeable_randomization_sample": False,
                        "formal_p_value_reported": False,
                    }
                )

    add_rows(
        distribution,
        metrics=("w1", "w2", "centroid_shift"),
        group_columns=(
            "anchor_id",
            "knockdown_fraction",
            "cohort",
            "interaction_enabled",
            "space",
        ),
        family="endpoint_distribution",
    )
    add_rows(
        mediation,
        metrics=("interaction_mediated_centroid_norm",),
        group_columns=("anchor_id", "knockdown_fraction", "cohort", "space"),
        family="interaction_difference_in_differences",
    )
    add_rows(
        target_message,
        metrics=("absolute_delta_D_target",),
        group_columns=(
            "anchor_id",
            "knockdown_fraction",
            "grouping_seed",
            "space",
        ),
        family="fixed_support_generic_complete_gnn_message",
    )
    return pd.DataFrame(rows)


def _target_monotonicity(target_message: pd.DataFrame) -> pd.DataFrame:
    primary = target_message.loc[
        target_message["condition_role"].eq("ligand_primary")
    ].copy()
    rows: list[dict[str, Any]] = []
    keys = ["anchor_id", "grouping_seed", "space"]
    for key_values, group in primary.groupby(keys, sort=True, dropna=False):
        group = group.sort_values("knockdown_fraction")
        dose = group["knockdown_fraction"].to_numpy(dtype=float)
        outcome = group["counterfactual_D_target"].to_numpy(dtype=float)
        delta = np.diff(outcome)
        tolerance = 1e-10 * max(1.0, float(np.max(np.abs(outcome))))
        nondecreasing = bool(np.all(delta >= -tolerance))
        nonincreasing = bool(np.all(delta <= tolerance))
        if nondecreasing and nonincreasing:
            direction = "flat"
        elif nondecreasing:
            direction = "nondecreasing"
        elif nonincreasing:
            direction = "nonincreasing"
        else:
            direction = "nonmonotonic"
        spearman = (
            float(pd.Series(dose).corr(pd.Series(outcome), method="spearman"))
            if len(dose) > 1
            else float("nan")
        )
        rows.append(
            {
                **dict(zip(keys, key_values)),
                "n_doses": int(len(dose)),
                "dose_min": float(np.min(dose)),
                "dose_max": float(np.max(dose)),
                "outcome": "counterfactual_D_target",
                "monotonic_direction": direction,
                "monotonic_non_decreasing": nondecreasing,
                "monotonic_non_increasing": nonincreasing,
                "spearman_dose_outcome": spearman,
                "largest_adjacent_increase": (
                    float(np.max(delta)) if delta.size else float("nan")
                ),
                "largest_adjacent_decrease": (
                    float(np.min(delta)) if delta.size else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


def _save_figure(fig, output: Path, stem: str) -> tuple[Path, Path]:
    png = output / f"{stem}.png"
    pdf = output / f"{stem}.pdf"
    fig.savefig(png, dpi=240, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    return png, pdf


def _make_reviewer_plots(
    output: Path,
    *,
    target_message: pd.DataFrame,
    distribution: pd.DataFrame,
    mediation: pd.DataFrame,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    paths: list[Path] = []
    colors = plt.get_cmap("tab10")

    primary_target = target_message.loc[
        target_message["condition_role"].eq("ligand_primary")
    ]
    sham_target = target_message.loc[
        target_message["condition_role"].eq("matched_hvg_sham")
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.2), sharex=True)
    for axis, space in zip(axes, ("joint", "spatial", "state")):
        subset = primary_target.loc[primary_target["space"].eq(space)]
        for color_index, (anchor, group) in enumerate(
            subset.groupby("anchor_id", sort=True)
        ):
            summary = (
                group.groupby("knockdown_fraction")["counterfactual_D_target"]
                .agg(["mean", "std"])
                .reset_index()
                .sort_values("knockdown_fraction")
            )
            baseline = float(group["baseline_D_target"].mean())
            axis.plot(
                summary["knockdown_fraction"],
                summary["mean"],
                marker="o",
                linewidth=2,
                color=colors(color_index),
                label=f"{anchor} · primary",
            )
            anchor_shams = sham_target.loc[
                sham_target["space"].eq(space) & sham_target["anchor_id"].eq(anchor)
            ]
            if not anchor_shams.empty:
                null_summary = (
                    anchor_shams.groupby("knockdown_fraction")[
                        "counterfactual_D_target"
                    ]
                    .agg(
                        sham_median="median",
                        sham_q025=lambda values: values.quantile(0.025),
                        sham_q975=lambda values: values.quantile(0.975),
                    )
                    .reset_index()
                    .sort_values("knockdown_fraction")
                )
                axis.fill_between(
                    null_summary["knockdown_fraction"],
                    null_summary["sham_q025"],
                    null_summary["sham_q975"],
                    color=colors(color_index),
                    alpha=0.18,
                    label=f"{anchor} · sham central 95% range",
                )
                axis.plot(
                    null_summary["knockdown_fraction"],
                    null_summary["sham_median"],
                    color=colors(color_index),
                    linestyle=":",
                    linewidth=1.3,
                )
            axis.axhline(
                baseline,
                color=colors(color_index),
                linestyle="--",
                linewidth=1,
                alpha=0.55,
            )
        title = {
            "joint": "Joint generic message\n(scale-dependent; descriptive)",
            "spatial": "Spatial generic message\n(primary space)",
            "state": "State generic message\n(primary space)",
        }[space]
        axis.set_title(title)
        axis.set_xlabel("Ligand knockdown fraction")
        axis.set_ylabel(r"$D_{\mathrm{target}}$ (generic GNN message norm)")
        axis.grid(alpha=0.22)
    axes[0].legend(title="Anchor / series", frameon=False, fontsize=8)
    fig.suptitle(
        "Generic complete GNN message on fixed ligand+→receptor+ support\n"
        "(primary points; descriptive matched-HVG central 95% ranges)",
        fontsize=13,
    )
    paths.extend(_save_figure(fig, output, "dose_response_target_message"))
    plt.close(fig)

    receiver = distribution.loc[
        distribution["condition_role"].eq("ligand_primary")
        & distribution["cohort"].eq("fixed_receptor_positive_ligand_negative")
        & distribution["interaction_enabled"].eq(True)
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), sharex=True)
    for axis, metric in zip(axes, ("w1", "w2")):
        for color_index, ((anchor, space), group) in enumerate(
            receiver.groupby(["anchor_id", "space"], sort=True)
        ):
            group = group.sort_values("knockdown_fraction")
            axis.plot(
                group["knockdown_fraction"],
                group[metric],
                marker="o",
                linewidth=1.8,
                color=colors(color_index),
                label=(
                    f"{anchor} · joint (scale-dependent)"
                    if space == "joint"
                    else f"{anchor} · {space} (primary)"
                ),
            )
        axis.set_title(metric.upper())
        axis.set_xlabel("Ligand knockdown fraction")
        axis.set_ylabel("Endpoint distance")
        axis.grid(alpha=0.22)
    axes[1].legend(frameon=False, fontsize=8, bbox_to_anchor=(1.02, 1))
    fig.suptitle(
        "Fixed receiver-cohort endpoint distribution shift\n"
        "(joint W mixes coordinate scales; do not compare it across spaces)",
        fontsize=13,
    )
    paths.extend(_save_figure(fig, output, "receiver_wasserstein_dose_response"))
    plt.close(fig)

    fixed_state = mediation.loc[
        mediation["cohort"].eq("fixed_receptor_positive_ligand_negative")
        & mediation["space"].eq("state")
    ]
    primary = fixed_state.loc[fixed_state["condition_role"].eq("ligand_primary")]
    sham = fixed_state.loc[fixed_state["condition_role"].eq("matched_hvg_sham")]
    off_on = distribution.loc[
        distribution["condition_role"].eq("ligand_primary")
        & distribution["cohort"].eq("fixed_receptor_positive_ligand_negative")
        & distribution["space"].eq("state")
    ]
    anchors = sorted(primary["anchor_id"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.5))
    positions: list[float] = []
    labels: list[str] = []
    position = 1.0
    for anchor in anchors:
        for dose in sorted(primary["knockdown_fraction"].unique()):
            null = sham.loc[
                sham["anchor_id"].eq(anchor) & sham["knockdown_fraction"].eq(dose),
                "interaction_mediated_centroid_norm",
            ].dropna()
            if len(null):
                axes[0].boxplot(
                    [null.to_numpy()],
                    positions=[position],
                    widths=0.55,
                    showfliers=False,
                    patch_artist=True,
                    boxprops={"facecolor": "#d9e6f2", "edgecolor": "#4c78a8"},
                    medianprops={"color": "#1f3552"},
                )
            observed = primary.loc[
                primary["anchor_id"].eq(anchor)
                & primary["knockdown_fraction"].eq(dose),
                "interaction_mediated_centroid_norm",
            ]
            if len(observed):
                axes[0].scatter(
                    [position],
                    [float(observed.iloc[0])],
                    color="#c43c39",
                    marker="D",
                    s=42,
                    zorder=4,
                )
            positions.append(position)
            labels.append(f"{anchor}\n{dose:g}")
            position += 1.0
        position += 0.5
    axes[0].set_xticks(positions, labels, rotation=35, ha="right")
    axes[0].set_ylabel("State mediated centroid norm")
    axes[0].set_title("Primary ligand (red) vs matched-HVG shams")
    axes[0].grid(axis="y", alpha=0.22)

    for color_index, ((anchor, enabled), group) in enumerate(
        off_on.groupby(["anchor_id", "interaction_enabled"], sort=True)
    ):
        group = group.sort_values("knockdown_fraction")
        axes[1].plot(
            group["knockdown_fraction"],
            group["w2"],
            marker="o",
            linewidth=2,
            linestyle="-" if enabled else "--",
            color=colors(color_index // 2),
            label=f"{anchor} · interaction {'on' if enabled else 'off'}",
        )
    axes[1].set_xlabel("Ligand knockdown fraction")
    axes[1].set_ylabel("Receiver state W2")
    axes[1].set_title("Same-trained-model mediation control")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(alpha=0.22)
    fig.suptitle("Matched-sham and interaction-off controls", fontsize=13)
    paths.extend(_save_figure(fig, output, "matched_sham_interaction_mediation"))
    plt.close(fig)
    return paths


def _documentation(
    output: Path,
    *,
    ligand: str,
    receptor: str,
    anchors: Sequence[tuple[float, float]],
    fractions: Sequence[float],
    n_shams: int,
    grouping_seeds: Sequence[int],
) -> tuple[Path, Path]:
    readme = output / "README_CN.md"
    readme.write_text(
        f"""# {ligand} → {receptor} 轴向计算扰动

本目录是 Reviewer #4 所要求的、预先固定设计的模型敏感性分析。锚点为
{", ".join(f"{a:g}→{b:g}" for a, b in anchors)}；敲低比例为
{", ".join(f"{value:g}" for value in fractions)}。主分析敲低配体 `{ligand}`，
并分别提供受体 `{receptor}` 及双敲低敏感性分析。

核心约束：

- 两个基因都必须是模型可见的 HVG，并在保留的 PCA 分量中具有非零载荷。
- 在 PCA 训练所用的表达尺度上做基因敲低，再用原始 PCA 载荷投影回模型状态。
- 每个锚点固定同一批细胞；扩散系数严格为 0，不做生长、分裂或重采样。
- 接收细胞群在锚点时一次性定义为“受体阳性、配体阴性”，之后不重新筛选。
- 基线与反事实使用同一分组随机种子。完整轨迹使用种子
  `{grouping_seeds[0]}`；固定 LR-expression-conditioned 边支持上的 generic
  complete GNN message 精确审计使用
  `{", ".join(map(str, grouping_seeds))}`。额外种子只重复精确 message 审计，
  不重复 trajectory、Wasserstein 或 mediation；它们是技术分组敏感性，不是独立训练重复。
- 每次输入改变后重新计算 link predictor、attention 及能够精确重构官方输出的
  完整 edge message。主要结果 `D_target` 是基线配体阳性发送细胞→固定受体阳性
  接收细胞支持上的 **generic complete GNN message**，不是
  {ligand}→{receptor} 特异的 message 分量；反事实中消失的边按零完整 message
  计入，新增边不进入固定支持。主图同时显示 deterministic matched-sham 的
  描述性 central 95% 范围和 primary 点。
- attention、link-predictor probability 和完整 message 分开报告；不把
  attention×配体/受体表达当作证据。
- interaction-on 与同一训练模型的 interaction-off 做差中差控制。
- 每个 anchor 内的 `{n_shams}` 个 HVG 假靶点仅在基线配体阳性 fixed-sender
  compartment 中按检出率、平均表达和 PCA 载荷范数匹配，也只在同一 sender
  mask 中编辑；`receiver_edit_audit.csv` 验证 fixed receiver 未被 sham 直接编辑。
- sham 是确定性最近匹配集合，并非可交换的随机抽样；
  `matched_sham_descriptive_comparison.csv` 只报告描述性 tail fraction、observed
  relative rank 和明确标注的 pseudocount score，不提供 randomization p-value，
  也不作显著性结论。
- spatial/state Wasserstein 是主空间结果；joint Wasserstein 混合空间坐标与 PCA
  状态尺度，具有 scale dependence，仅作描述，不能跨空间比较。
- OT 截断在每个 anchor 内固定同一个 cell-ID index hash；所有 condition、dose
  以及 interaction-on/off 共用这组 indices，避免 matched-sham 或 mediation
  对比混入不同抽样支持。
- `independent_message_alignment_diagnostic.csv` 使用独立的 NumPy audit grouping，
  而 rollout 内部使用 torch grouping；它不是 rollout 驱动 message 场，也不进入
  primary inference。

主要表格为 `fixed_lr_target_message.csv`（固定支持完整 message 剂量反应）、
`counterfactual_metrics.csv`（W1/W2/质心位移）、
`independent_message_alignment_diagnostic.csv`、`interaction_mediation.csv`、
`matched_hvg_shams.csv` 和 `matched_sham_descriptive_comparison.csv`。完整运行参数、输入绑定、
模型摘要及文件哈希见 `run_manifest.json` 与 `checksums.sha256`。三套
reviewer 图均提供 PNG 与 PDF。

结论边界：这些结果只说明已训练 CytoBridge 模型对计算输入扰动的敏感性；
不能解释为真实基因敲低实验、机制证明或生物学因果效应。
""",
        encoding="utf-8",
    )
    response = output / "REVIEWER_RESPONSE.md"
    response.write_text(
        f"""## Reviewer #4: axis-specific computational perturbation

We added a pre-specified `{ligand} -> {receptor}` counterfactual at anchors
{", ".join(f"{a:g}->{b:g}" for a, b in anchors)}. Both genes are required to be
model-visible HVGs. Ligand knockdown is evaluated at
{", ".join(f"{100 * value:g}%" for value in fractions)}, with receptor-only and
dual perturbations as sensitivity analyses. The baseline receptor-positive,
ligand-negative receiver cohort, cell identities, time grid, and grouping seed
are held fixed.

The analysis recomputes the frozen link predictor, signed attention gates, and
the exact complete one-layer GNN messages after every input edit. The primary
message outcome is `D_target`: the mean, over all fixed receivers, of the norm
of summed generic complete GNN messages on the baseline
ligand-positive-sender to fixed receptor-positive receiver edge support. It is
not a `{ligand}->{receptor}`-specific message component. A support edge that
disappears contributes zero; new edges are excluded. Predictor and attention
diagnostics remain separate, and no attention-by-LR-expression collapse is
used. Exact message audits use technical grouping seeds
{", ".join(map(str, grouping_seeds))}; only the first seed drives trajectories,
Wasserstein metrics, and mediation, while extra seeds repeat only the exact
message audit. They are not independent trained models. Endpoint effects are
summarized by W1, W2, and centroid displacement. Spatial and state results are
primary; joint Wasserstein mixes coordinate scales and is scale-dependent.
Within each anchor, one identity-paired OT index set is reused across every
condition, dose, and interaction-on/off comparison.
The separately labeled alignment table compares an independently grouped
NumPy audit with a torch-grouped rollout endpoint, so it is a technical
diagnostic rather than the rollout-driving message field. An interaction-on
versus same-trained-model interaction-off
difference-in-differences control assesses whether the modeled response is
interaction-dependent. Within each anchor, a matched reference uses {n_shams} HVGs
selected on detection fraction, mean expression, and PCA-loading norm inside
the baseline ligand-positive fixed-sender compartment. Sham edits are confined
to that same sender mask, and a receiver-edit audit verifies that fixed
receivers are never directly edited by a sham. These are deterministic nearest
matches, not exchangeable random draws. We therefore report descriptive sham
tail fractions and observed relative ranks, not a randomization p-value or a
significance claim; the optional +1 score is explicitly labeled as a
non-formal pseudocount tail score.

We deliberately bound the claim: this is evidence of sensitivity within the
trained CytoBridge model. It is not an experimental knockdown, does not identify
a biological mechanism, and does not establish causal ligand-receptor action.
""",
        encoding="utf-8",
    )
    return readme, response


def run_analysis(
    data,
    model,
    *,
    output_dir: str | Path,
    ligand: str = "cxcl12a",
    receptor: str = "cxcr4a",
    anchors: Sequence[tuple[float, float]] = DEFAULT_ANCHORS,
    fractions: Sequence[float] = DEFAULT_FRACTIONS,
    n_shams: int = 100,
    time_key: str = "time_point_processed",
    cell_type_key: str = "Annotation",
    spatial_key: str = "spatial_aligned",
    state_key: str = "X_latent",
    loadings_key: str = "PCs",
    hvg_key: str = "highly_variable",
    pca_center_key: str = "pca_center",
    group_size: int = 1024,
    dt: float = 0.05,
    grouping_seeds: Sequence[int] = DEFAULT_GROUPING_SEEDS,
    metric_seed: int = 1701,
    max_ot_points: Optional[int] = 1024,
    receiver_threshold: float = 0.0,
    pca_contract_atol: float = 5e-4,
    pca_contract_rtol: float = 5e-4,
    device: str = "cpu",
    overwrite: bool = False,
    input_provenance: Optional[Mapping[str, Any]] = None,
    checkpoint_provenance: Optional[Mapping[str, Any]] = None,
) -> Mapping[str, Any]:
    """Run the complete bundle from an already loaded AnnData and model."""

    from CytoBridge.tl.downstream.perturbation import (
        audit_spatial_complete_messages,
        apply_projected_gene_knockdowns,
        compute_fixed_lr_target_message_metrics,
        deterministic_fixed_cohort_rollout,
        match_hvg_sham_genes,
        run_gene_counterfactual,
        select_fixed_receiver_cohort,
        validate_pca_model_visibility,
    )
    from CytoBridge.tl.downstream.spatial_interaction_attribution import (
        validate_spatial_exact_decomposition_model,
    )

    output = _prepare_output(Path(output_dir), bool(overwrite))
    if not hasattr(data, "obs") or not hasattr(data, "obsm"):
        raise TypeError("data must be AnnData-like.")
    for key in (spatial_key, state_key):
        if key not in data.obsm:
            raise KeyError(f"adata.obsm is missing {key!r}.")
    for key in (loadings_key,):
        if key not in data.varm:
            raise KeyError(f"adata.varm is missing {key!r}.")
    for key in (hvg_key, pca_center_key):
        if key not in data.var:
            raise KeyError(f"adata.var is missing {key!r}.")
    if cell_type_key not in data.obs:
        raise KeyError(f"adata.obs is missing {cell_type_key!r}.")

    fractions = tuple(float(value) for value in fractions)
    if (
        not fractions
        or len(set(fractions)) != len(fractions)
        or any(not np.isfinite(value) or value <= 0 or value > 1 for value in fractions)
    ):
        raise ValueError("fractions must be unique values in (0, 1].")
    anchors = tuple((float(start), float(end)) for start, end in anchors)
    if not anchors or any(end <= start for start, end in anchors):
        raise ValueError("anchors must contain increasing start/end pairs.")
    if int(group_size) < 2:
        raise ValueError("group_size must be at least two.")
    if int(n_shams) < 1:
        raise ValueError("n_shams must be positive.")
    grouping_seeds = tuple(int(value) for value in grouping_seeds)
    if not grouping_seeds or len(set(grouping_seeds)) != len(grouping_seeds):
        raise ValueError("grouping_seeds must be non-empty and unique.")

    feature_names = tuple(str(name) for name in data.var_names)
    ligand_feature = _resolve_gene(feature_names, ligand)
    receptor_feature = _resolve_gene(feature_names, receptor)
    if ligand_feature == receptor_feature:
        raise ValueError("Ligand and receptor must resolve to distinct features.")
    loadings = np.asarray(data.varm[loadings_key], dtype=np.float32)
    hvg = np.asarray(data.var[hvg_key], dtype=bool)
    center = np.asarray(data.var[pca_center_key], dtype=np.float32)
    state = np.asarray(data.obsm[state_key], dtype=np.float32)
    spatial = np.asarray(data.obsm[spatial_key], dtype=np.float32)
    if spatial.shape != (data.n_obs, 2):
        raise ValueError(f"{spatial_key} must have shape (N, 2), got {spatial.shape}.")
    if state.shape != (data.n_obs, loadings.shape[1]):
        raise ValueError(
            f"{state_key} must have shape ({data.n_obs}, {loadings.shape[1]})."
        )
    if not np.isfinite(spatial).all() or not np.isfinite(state).all():
        raise ValueError("Spatial/PCA model input contains non-finite values.")
    expression = data.X
    if expression.shape != (data.n_obs, data.n_vars):
        raise ValueError("adata.X shape is inconsistent with AnnData dimensions.")

    visibility = validate_pca_model_visibility(
        feature_names,
        loadings,
        (ligand_feature, receptor_feature),
        highly_variable=hvg,
    )
    pca_contract = _validate_pca_contract(
        expression,
        state,
        loadings,
        center,
        atol=float(pca_contract_atol),
        rtol=float(pca_contract_rtol),
    )
    times = _time_values(data, time_key)
    observed_stages = set(np.unique(times).tolist())
    for start, end in anchors:
        if start not in observed_stages or end not in observed_stages:
            raise ValueError(
                f"Anchor {start:g}->{end:g} is not fully represented in observed stages "
                f"{sorted(observed_stages)}."
            )

    interaction_net = getattr(model, "interaction_net", None)
    if interaction_net is None:
        raise ValueError("Loaded model does not expose interaction_net.")
    validate_spatial_exact_decomposition_model(interaction_net)
    configured_group_size = int(getattr(model, "interaction_group_size", group_size))
    if configured_group_size != int(group_size):
        raise ValueError(
            "Requested group size does not match the trained model: "
            f"requested={group_size}, configured={configured_group_size}."
        )

    base_condition_specs: list[dict[str, Any]] = [
        {
            "condition": "ligand",
            "condition_role": "ligand_primary",
            "genes": (ligand_feature,),
            "is_sham": False,
            "sham_rank": None,
            "edit_cell_mask_policy": (
                "baseline_primary_ligand_positive_fixed_sender_only"
            ),
        },
        {
            "condition": "receptor",
            "condition_role": "receptor_sensitivity",
            "genes": (receptor_feature,),
            "is_sham": False,
            "sham_rank": None,
            "edit_cell_mask_policy": "all_anchor_cells_expression_scaled",
        },
        {
            "condition": "dual",
            "condition_role": "dual_sensitivity",
            "genes": (ligand_feature, receptor_feature),
            "is_sham": False,
            "sham_rank": None,
            "edit_cell_mask_policy": "all_anchor_cells_expression_scaled",
        },
    ]

    sham_frames: list[pd.DataFrame] = []
    cells_frames: list[pd.DataFrame] = []
    distribution_frames: list[pd.DataFrame] = []
    alignment_frames: list[pd.DataFrame] = []
    mediation_frames: list[pd.DataFrame] = []
    edit_frames: list[pd.DataFrame] = []
    exact_summary_rows: list[dict[str, Any]] = []
    reconstruction_frames: list[pd.DataFrame] = []
    message_frames: list[pd.DataFrame] = []
    target_message_frames: list[pd.DataFrame] = []
    primary_edge_frames: list[pd.DataFrame] = []
    receiver_edit_rows: list[dict[str, Any]] = []
    array_payload: dict[str, np.ndarray] = {}
    receiver_counts: dict[str, int] = {}
    ot_sampling_seeds: dict[str, int] = {}

    ligand_index = feature_names.index(ligand_feature)
    receptor_index = feature_names.index(receptor_feature)
    for anchor_index, (start, end) in enumerate(anchors):
        anchor_id = f"{start:g}_to_{end:g}"
        anchor_metric_seed = int(metric_seed) + anchor_index * 1_000_000
        ot_sampling_seeds[anchor_id] = anchor_metric_seed
        start_mask = np.isclose(times, start, rtol=0.0, atol=1e-12)
        indices = np.flatnonzero(start_mask)
        if indices.size < 2:
            raise ValueError(f"Anchor {anchor_id} has fewer than two initial cells.")
        points = np.hstack((spatial[indices], state[indices])).astype(np.float32)
        anchor_expression = expression[indices]
        receiver_mask = select_fixed_receiver_cohort(
            anchor_expression,
            feature_names,
            ligand=ligand_feature,
            receptor=receptor_feature,
            positive_threshold=float(receiver_threshold),
        )
        ligand_positive_mask = _dense_column(anchor_expression, ligand_index) > float(
            receiver_threshold
        )
        primary_sender_mask = ligand_positive_mask & ~receiver_mask
        if not np.any(primary_sender_mask):
            raise ValueError(
                f"Anchor {anchor_id} has no baseline ligand-positive fixed senders."
            )
        if np.any(primary_sender_mask & receiver_mask):
            raise RuntimeError(
                "Primary fixed sender and fixed receiver masks must be disjoint."
            )
        receiver_counts[anchor_id] = int(receiver_mask.sum())
        cell_table = pd.DataFrame(
            {
                "anchor_id": anchor_id,
                "anchor_start": start,
                "anchor_end": end,
                "local_index": np.arange(indices.size, dtype=int),
                "global_index": indices,
                "obs_name": data.obs_names[indices].astype(str),
                "cell_type": data.obs.iloc[indices][cell_type_key]
                .astype(str)
                .to_numpy(),
                "ligand_expression": _dense_column(anchor_expression, ligand_index),
                "receptor_expression": _dense_column(anchor_expression, receptor_index),
                "fixed_primary_ligand_positive_sender": primary_sender_mask,
                "fixed_receiver": receiver_mask,
            }
        )
        cells_frames.append(cell_table)

        anchor_shams = match_hvg_sham_genes(
            anchor_expression,
            feature_names,
            loadings,
            hvg,
            target_gene=ligand_feature,
            n_shams=int(n_shams),
            exclude_genes=(ligand_feature, receptor_feature),
            cell_mask=primary_sender_mask,
        )
        anchor_shams.insert(0, "anchor_id", anchor_id)
        anchor_shams.insert(1, "anchor_start", float(start))
        anchor_shams.insert(2, "anchor_end", float(end))
        anchor_shams.insert(3, "target_axis_symbol", str(ligand))
        anchor_shams.insert(
            4,
            "matching_compartment",
            "baseline_primary_ligand_positive_fixed_sender",
        )
        sham_frames.append(anchor_shams)
        condition_specs = [dict(spec) for spec in base_condition_specs]
        condition_specs[0]["cell_mask"] = primary_sender_mask
        condition_specs[1]["cell_mask"] = None
        condition_specs[2]["cell_mask"] = None
        condition_specs.extend(
            {
                "condition": f"sham_{int(row.sham_rank):03d}_{row.gene}",
                "condition_role": "matched_hvg_sham",
                "genes": (str(row.gene),),
                "is_sham": True,
                "sham_rank": int(row.sham_rank),
                "cell_mask": primary_sender_mask,
                "edit_cell_mask_policy": (
                    "baseline_primary_ligand_positive_fixed_sender_only"
                ),
            }
            for row in anchor_shams.itertuples(index=False)
        )

        anchor_grouping_seed = int(grouping_seeds[0])
        baseline_audit = audit_spatial_complete_messages(
            interaction_net,
            points,
            time_value=start,
            group_size=int(group_size),
            grouping_seed=anchor_grouping_seed,
            device=str(device),
            spatial_dim=2,
        )
        baseline_on = deterministic_fixed_cohort_rollout(
            points,
            model,
            start_time=start,
            end_time=end,
            dt=float(dt),
            interaction_m=int(group_size),
            grouping_seed=anchor_grouping_seed,
            device=str(device),
            spatial_dim=2,
            interaction_enabled=True,
            sigma=0.0,
        )
        baseline_off = deterministic_fixed_cohort_rollout(
            points,
            model,
            start_time=start,
            end_time=end,
            dt=float(dt),
            interaction_m=int(group_size),
            grouping_seed=anchor_grouping_seed,
            device=str(device),
            spatial_dim=2,
            interaction_enabled=False,
            sigma=0.0,
        )
        seed_prefix = f"{anchor_id}__seed_{anchor_grouping_seed}"
        array_payload[f"{seed_prefix}__baseline_on_endpoint"] = baseline_on.points[-1]
        array_payload[f"{seed_prefix}__baseline_off_endpoint"] = baseline_off.points[-1]
        array_payload[
            f"{seed_prefix}__baseline_complete_output"
        ] = baseline_audit.output
        baseline_reconstruction = baseline_audit.reconstruction_table.copy()
        baseline_reconstruction.insert(0, "anchor_id", anchor_id)
        baseline_reconstruction.insert(1, "condition", "baseline")
        baseline_reconstruction.insert(2, "knockdown_fraction", 0.0)
        reconstruction_frames.append(baseline_reconstruction)
        baseline_edges = baseline_audit.edge_table.copy()
        baseline_edges.insert(0, "anchor_id", anchor_id)
        baseline_edges.insert(1, "condition", "baseline")
        baseline_edges.insert(2, "knockdown_fraction", 0.0)
        for head in range(baseline_audit.attention_signed.shape[1]):
            baseline_edges[
                f"attention_signed_head_{head}"
            ] = baseline_audit.attention_signed[:, head]
        primary_edge_frames.append(baseline_edges)

        for spec in condition_specs:
            for fraction in fractions:
                condition = str(spec["condition"])
                result = run_gene_counterfactual(
                    points,
                    anchor_expression,
                    feature_names,
                    loadings,
                    model,
                    genes=spec["genes"],
                    fraction=float(fraction),
                    receiver_mask=receiver_mask,
                    start_time=start,
                    end_time=end,
                    dt=float(dt),
                    interaction_m=int(group_size),
                    grouping_seed=anchor_grouping_seed,
                    device=str(device),
                    spatial_dim=2,
                    max_ot_points=max_ot_points,
                    metric_seed=anchor_metric_seed,
                    cell_mask=spec["cell_mask"],
                    baseline_audit=baseline_audit,
                    baseline_on=baseline_on,
                    baseline_off=baseline_off,
                )
                annotation = {
                    "anchor_id": anchor_id,
                    "start": start,
                    "end": end,
                    "condition": condition,
                    "condition_role": str(spec["condition_role"]),
                    "targets": spec["genes"],
                    "fraction": fraction,
                    "is_sham": bool(spec["is_sham"]),
                    "sham_rank": spec["sham_rank"],
                    "grouping_seed": anchor_grouping_seed,
                    "edit_cell_mask_policy": str(spec["edit_cell_mask_policy"]),
                }
                receiver_delta_norm = np.linalg.norm(
                    np.asarray(
                        result.edit.delta_state[receiver_mask], dtype=np.float64
                    ),
                    axis=1,
                )
                sender_delta_norm = np.linalg.norm(
                    np.asarray(
                        result.edit.delta_state[primary_sender_mask],
                        dtype=np.float64,
                    ),
                    axis=1,
                )
                direct_edit_tolerance = 1e-12
                receiver_edit_rows.append(
                    {
                        "anchor_id": anchor_id,
                        "anchor_start": float(start),
                        "anchor_end": float(end),
                        "condition": condition,
                        "condition_role": str(spec["condition_role"]),
                        "target_genes": ";".join(map(str, spec["genes"])),
                        "knockdown_fraction": float(fraction),
                        "is_sham": bool(spec["is_sham"]),
                        "sham_rank": spec["sham_rank"],
                        "grouping_seed": anchor_grouping_seed,
                        "edit_cell_mask_policy": str(spec["edit_cell_mask_policy"]),
                        "n_fixed_receivers": int(receiver_mask.sum()),
                        "n_primary_fixed_senders": int(primary_sender_mask.sum()),
                        "n_directly_edited_fixed_receivers": int(
                            np.sum(receiver_delta_norm > direct_edit_tolerance)
                        ),
                        "max_fixed_receiver_projected_delta_norm": float(
                            np.max(receiver_delta_norm)
                        ),
                        "mean_fixed_receiver_projected_delta_norm": float(
                            np.mean(receiver_delta_norm)
                        ),
                        "n_directly_edited_primary_fixed_senders": int(
                            np.sum(sender_delta_norm > direct_edit_tolerance)
                        ),
                        "max_primary_fixed_sender_projected_delta_norm": float(
                            np.max(sender_delta_norm)
                        ),
                        "fixed_receiver_intersects_selected_edit_mask": bool(
                            np.any(
                                receiver_mask
                                & (
                                    np.ones(indices.size, dtype=bool)
                                    if spec["cell_mask"] is None
                                    else np.asarray(spec["cell_mask"], dtype=bool)
                                )
                            )
                        ),
                    }
                )
                target_message_frames.append(
                    _annotate(
                        compute_fixed_lr_target_message_metrics(
                            result.baseline_audit,
                            result.counterfactual_audit,
                            ligand_positive_mask=ligand_positive_mask,
                            receiver_mask=receiver_mask,
                            spatial_dim=2,
                            counterfactual_points=result.edit.points,
                            interaction_net=interaction_net,
                            device=str(device),
                        ),
                        **annotation,
                    )
                )
                distribution_frames.extend(
                    [
                        _annotate(result.metrics_on.distribution, **annotation),
                        _annotate(result.metrics_off.distribution, **annotation),
                    ]
                )
                alignment_diagnostic = result.metrics_on.alignment.copy()
                alignment_diagnostic["same_grouping_plan_as_rollout_driver"] = False
                alignment_diagnostic["eligible_for_primary_inference"] = False
                alignment_diagnostic["diagnostic_scope"] = (
                    "independent_numpy_grouping_anchor_audit_vs_"
                    "torch_grouped_rollout_endpoint"
                )
                alignment_frames.append(_annotate(alignment_diagnostic, **annotation))
                mediation_frames.append(_annotate(result.mediation, **annotation))
                edit_frames.append(_annotate(result.edit.gene_table, **annotation))
                message_frames.append(
                    _annotate(
                        _message_delta_summary(
                            result.counterfactual_audit.output
                            - result.baseline_audit.output,
                            receiver_mask,
                            spatial_dim=2,
                        ),
                        **annotation,
                    )
                )

                summary = _edge_set_summary(
                    result.baseline_audit, result.counterfactual_audit
                )
                exact_summary_rows.append(
                    {
                        **{
                            key: value
                            for key, value in _annotate(
                                pd.DataFrame([{}]), **annotation
                            )
                            .iloc[0]
                            .items()
                        },
                        **summary,
                        "max_abs_reconstruction_residual": float(
                            result.counterfactual_audit.reconstruction_table[
                                "max_abs_residual"
                            ].max()
                        ),
                        "max_relative_l2_reconstruction_residual": float(
                            result.counterfactual_audit.reconstruction_table[
                                "relative_l2_residual"
                            ].max()
                        ),
                    }
                )
                reconstruction_frames.append(
                    _annotate(
                        result.counterfactual_audit.reconstruction_table,
                        **annotation,
                    )
                )

                if not bool(spec["is_sham"]):
                    key = re.sub(
                        r"[^A-Za-z0-9_]+",
                        "_",
                        f"{seed_prefix}__{condition}__kd_{fraction:g}",
                    )
                    array_payload[
                        f"{key}__on_endpoint"
                    ] = result.counterfactual_on.points[-1]
                    array_payload[
                        f"{key}__off_endpoint"
                    ] = result.counterfactual_off.points[-1]
                    array_payload[
                        f"{key}__complete_output"
                    ] = result.counterfactual_audit.output
                    array_payload[
                        f"{key}__edge_output"
                    ] = result.counterfactual_audit.edge_output
                    array_payload[
                        f"{key}__attention_signed"
                    ] = result.counterfactual_audit.attention_signed
                    edges = result.counterfactual_audit.edge_table.copy()
                    edges = _annotate(edges, **annotation)
                    for head in range(
                        result.counterfactual_audit.attention_signed.shape[1]
                    ):
                        edges[
                            f"attention_signed_head_{head}"
                        ] = result.counterfactual_audit.attention_signed[:, head]
                    primary_edge_frames.append(edges)

        # Additional grouping seeds repeat the exact fixed-support anchor audit
        # only.  Full trajectory/OT/mediation rollouts use the first seed above
        # to keep the 100-sham design computationally tractable.
        for extra_grouping_seed in grouping_seeds[1:]:
            extra_baseline = audit_spatial_complete_messages(
                interaction_net,
                points,
                time_value=start,
                group_size=int(group_size),
                grouping_seed=int(extra_grouping_seed),
                device=str(device),
                spatial_dim=2,
            )
            extra_baseline_reconstruction = extra_baseline.reconstruction_table.copy()
            extra_baseline_reconstruction.insert(0, "anchor_id", anchor_id)
            extra_baseline_reconstruction.insert(1, "condition", "baseline")
            extra_baseline_reconstruction.insert(2, "knockdown_fraction", 0.0)
            reconstruction_frames.append(extra_baseline_reconstruction)
            for spec in condition_specs:
                for fraction in fractions:
                    condition = str(spec["condition"])
                    edit = apply_projected_gene_knockdowns(
                        points,
                        anchor_expression,
                        feature_names,
                        loadings,
                        spec["genes"],
                        float(fraction),
                        spatial_dim=2,
                        cell_mask=spec["cell_mask"],
                    )
                    extra_counterfactual = audit_spatial_complete_messages(
                        interaction_net,
                        edit.points,
                        time_value=start,
                        group_size=int(group_size),
                        grouping_seed=int(extra_grouping_seed),
                        device=str(device),
                        spatial_dim=2,
                    )
                    annotation = {
                        "anchor_id": anchor_id,
                        "start": start,
                        "end": end,
                        "condition": condition,
                        "condition_role": str(spec["condition_role"]),
                        "targets": spec["genes"],
                        "fraction": fraction,
                        "is_sham": bool(spec["is_sham"]),
                        "sham_rank": spec["sham_rank"],
                        "grouping_seed": int(extra_grouping_seed),
                        "edit_cell_mask_policy": str(spec["edit_cell_mask_policy"]),
                    }
                    target_message_frames.append(
                        _annotate(
                            compute_fixed_lr_target_message_metrics(
                                extra_baseline,
                                extra_counterfactual,
                                ligand_positive_mask=ligand_positive_mask,
                                receiver_mask=receiver_mask,
                                spatial_dim=2,
                                counterfactual_points=edit.points,
                                interaction_net=interaction_net,
                                device=str(device),
                            ),
                            **annotation,
                        )
                    )
                    reconstruction_frames.append(
                        _annotate(
                            extra_counterfactual.reconstruction_table,
                            **annotation,
                        )
                    )
                    summary = _edge_set_summary(extra_baseline, extra_counterfactual)
                    exact_summary_rows.append(
                        {
                            **{
                                key: value
                                for key, value in _annotate(
                                    pd.DataFrame([{}]), **annotation
                                )
                                .iloc[0]
                                .items()
                            },
                            **summary,
                            "max_abs_reconstruction_residual": float(
                                extra_counterfactual.reconstruction_table[
                                    "max_abs_residual"
                                ].max()
                            ),
                            "max_relative_l2_reconstruction_residual": float(
                                extra_counterfactual.reconstruction_table[
                                    "relative_l2_residual"
                                ].max()
                            ),
                            "scope": "exact_message_grouping_sensitivity_only",
                        }
                    )

    shams = pd.concat(sham_frames, ignore_index=True)
    cells = pd.concat(cells_frames, ignore_index=True)
    distribution = pd.concat(distribution_frames, ignore_index=True)
    alignment = pd.concat(alignment_frames, ignore_index=True)
    mediation = pd.concat(mediation_frames, ignore_index=True)
    edits = pd.concat(edit_frames, ignore_index=True)
    exact_summary = pd.DataFrame(exact_summary_rows)
    reconstruction = pd.concat(
        [
            frame.dropna(axis=1, how="all")
            for frame in reconstruction_frames
            if not frame.empty
        ],
        ignore_index=True,
    )
    message_summary = pd.concat(message_frames, ignore_index=True)
    target_message = pd.concat(target_message_frames, ignore_index=True)
    target_monotonicity = _target_monotonicity(target_message)
    primary_edges = pd.concat(primary_edge_frames, ignore_index=True)
    receiver_edit_audit = pd.DataFrame(receiver_edit_rows)
    descriptive_shams = _descriptive_sham_comparison(
        distribution,
        mediation,
        target_message,
    )

    artifact_paths: list[Path] = []

    def write_csv(name: str, table: pd.DataFrame, *, gzip: bool = False) -> Path:
        path = output / name
        table.to_csv(
            path,
            index=False,
            compression="gzip" if gzip else None,
        )
        artifact_paths.append(path)
        return path

    write_csv("cohort_cells.csv.gz", cells, gzip=True)
    write_csv("matched_hvg_shams.csv", shams)
    write_csv("receiver_edit_audit.csv", receiver_edit_audit)
    write_csv("fixed_lr_target_message.csv", target_message)
    write_csv("fixed_lr_target_monotonicity.csv", target_monotonicity)
    write_csv("counterfactual_metrics.csv", distribution)
    write_csv("independent_message_alignment_diagnostic.csv", alignment)
    write_csv("interaction_mediation.csv", mediation)
    write_csv("projected_gene_edits.csv", edits)
    write_csv("exact_message_summary.csv", exact_summary)
    write_csv("complete_message_delta_summary.csv", message_summary)
    write_csv("exact_reconstruction_diagnostics.csv", reconstruction)
    write_csv("primary_edge_diagnostics.csv.gz", primary_edges, gzip=True)
    write_csv("matched_sham_descriptive_comparison.csv", descriptive_shams)
    arrays_path = output / "primary_counterfactual_arrays.npz"
    np.savez_compressed(arrays_path, **array_payload)
    artifact_paths.append(arrays_path)
    visibility_path = write_csv("axis_model_visibility.csv", visibility)
    plot_paths = _make_reviewer_plots(
        output,
        target_message=target_message,
        distribution=distribution,
        mediation=mediation,
    )
    artifact_paths.extend(plot_paths)
    readme, response = _documentation(
        output,
        ligand=str(ligand),
        receptor=str(receptor),
        anchors=anchors,
        fractions=fractions,
        n_shams=int(n_shams),
        grouping_seeds=grouping_seeds,
    )
    artifact_paths.extend((readme, response))

    checks = {
        "axis_genes_model_visible": bool(visibility["model_visible"].all()),
        "fixed_receiver_nonempty_every_anchor": all(
            count > 0 for count in receiver_counts.values()
        ),
        "matched_sham_count_exact_per_anchor": bool(
            set(shams["anchor_id"].astype(str)) == set(receiver_counts)
            and shams.groupby("anchor_id", sort=False).size().eq(int(n_shams)).all()
        ),
        "shams_matched_in_primary_sender_compartment": bool(
            shams["matching_compartment"]
            .eq("baseline_primary_ligand_positive_fixed_sender")
            .all()
        ),
        "sham_fixed_receivers_never_directly_edited": bool(
            receiver_edit_audit.loc[
                receiver_edit_audit["condition_role"].eq("matched_hvg_sham"),
                "n_directly_edited_fixed_receivers",
            ]
            .eq(0)
            .all()
            and receiver_edit_audit.loc[
                receiver_edit_audit["condition_role"].eq("matched_hvg_sham"),
                "max_fixed_receiver_projected_delta_norm",
            ]
            .eq(0.0)
            .all()
        ),
        "every_sham_edits_at_least_one_primary_fixed_sender": bool(
            receiver_edit_audit.loc[
                receiver_edit_audit["condition_role"].eq("matched_hvg_sham"),
                "n_directly_edited_primary_fixed_senders",
            ]
            .gt(0)
            .all()
            and receiver_edit_audit.loc[
                receiver_edit_audit["condition_role"].eq("matched_hvg_sham"),
                "max_primary_fixed_sender_projected_delta_norm",
            ]
            .gt(0.0)
            .all()
        ),
        "primary_ligand_fixed_receivers_never_directly_edited": bool(
            receiver_edit_audit.loc[
                receiver_edit_audit["condition_role"].eq("ligand_primary"),
                "n_directly_edited_fixed_receivers",
            ]
            .eq(0)
            .all()
        ),
        "counterfactual_ot_support_identity_paired": bool(
            distribution["ot_support_is_identity_paired"].all()
            and distribution["ot_sampling_policy"]
            .eq("identity_paired_shared_indices")
            .all()
        ),
        "ot_indices_common_across_conditions_doses_and_on_off": bool(
            distribution.groupby(["anchor_id", "cohort"], sort=False)[
                ["ot_random_seed", "ot_support_index_sha256"]
            ]
            .nunique(dropna=False)
            .eq(1)
            .all()
            .all()
        ),
        "sigma_exactly_zero": True,
        "fixed_cell_identity": True,
        "growth_split_resampling_disabled": True,
        "same_seed_baseline_counterfactual": True,
        "interaction_off_uses_same_trained_model": True,
        "exact_message_reconstruction_passed": bool(
            reconstruction["max_abs_residual"].max() <= 2e-5
            and reconstruction["relative_l2_residual"].max() <= 2e-5
        ),
        "fixed_lr_target_support_predeclared": bool(
            target_message["support_policy"].nunique() == 1
        ),
        "target_message_is_generic_not_lr_specific": bool(
            target_message["message_semantics"]
            .eq(
                "generic complete GNN message on expression-conditioned "
                "fixed support; not an LR-specific message component"
            )
            .all()
        ),
        "missing_target_edges_zero_filled": bool(
            target_message["complete_message_missing_edges_treated_as_zero"].all()
        ),
        "target_message_grouping_seed_coverage": set(
            target_message["grouping_seed"].astype(int)
        )
        == set(grouping_seeds),
        "reviewer_png_pdf_plots_written": len(plot_paths) == 6
        and all(path.stat().st_size > 0 for path in plot_paths),
        "message_alignment_is_independent_diagnostic_only": bool(
            (~alignment["same_grouping_plan_as_rollout_driver"]).all()
            and (~alignment["eligible_for_primary_inference"]).all()
        ),
        "joint_wasserstein_marked_scale_dependent": bool(
            distribution.loc[
                distribution["space"].eq("joint"),
                "wasserstein_scale_contract",
            ]
            .str.contains("scale-dependent", regex=False)
            .all()
        ),
        "matched_sham_statistics_descriptive_not_formal_pvalues": bool(
            "empirical_upper_tail_p" not in descriptive_shams
            and descriptive_shams["formal_p_value_reported"].eq(False).all()
            and descriptive_shams["null_is_exchangeable_randomization_sample"]
            .eq(False)
            .all()
        ),
        "attention_lr_expression_collapse_used": False,
        "pca_contract_passed": bool(pca_contract["passed"]),
    }
    if (
        not all(
            value is True
            for key, value in checks.items()
            if key != "attention_lr_expression_collapse_used"
        )
        or checks["attention_lr_expression_collapse_used"]
    ):
        raise RuntimeError(f"Final scientific contract checks failed: {checks}")

    artifacts = {path.name: _artifact(path, root=output) for path in artifact_paths}
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis": "zebrafish_axis_specific_lr_gene_counterfactual",
        "axis": {
            "ligand_symbol": str(ligand),
            "receptor_symbol": str(receptor),
            "ligand_feature": ligand_feature,
            "receptor_feature": receptor_feature,
            "primary_condition": "ligand",
            "sensitivity_conditions": ["receptor", "dual"],
        },
        "design": {
            "anchors": [list(pair) for pair in anchors],
            "knockdown_fractions": list(fractions),
            "receiver_cohort": (
                "baseline receptor-positive and ligand-negative; fixed thereafter"
            ),
            "receiver_positive_threshold": float(receiver_threshold),
            "receiver_counts": receiver_counts,
            "n_matched_hvg_shams_per_anchor": int(n_shams),
            "sham_matching": [
                "detection_fraction",
                "mean_expression",
                "pca_loading_norm",
            ],
            "sham_matching_compartment": (
                "baseline primary-ligand-positive fixed senders, separately "
                "within each anchor"
            ),
            "primary_ligand_and_sham_edit_scope": (
                "baseline primary-ligand-positive fixed-sender mask"
            ),
            "fixed_receiver_direct_sham_edit_allowed": False,
            "rollout": "fixed-cohort deterministic Euler",
            "sigma": 0.0,
            "growth_or_split_resampling": False,
            "dt": float(dt),
            "interaction_group_size": int(group_size),
            "full_rollout_grouping_seed": int(grouping_seeds[0]),
            "trajectory_wasserstein_mediation_grouping_seeds": [int(grouping_seeds[0])],
            "exact_message_grouping_seeds": list(grouping_seeds),
            "grouping_seed_scope": (
                "The first seed drives full on/off trajectories; every seed "
                "drives the exact generic-message audit on the fixed "
                "LR-conditioned support. Seeds are technical partitions, not "
                "independent trained models."
            ),
            "metric_seed_base": int(metric_seed),
            "ot_sampling_seed_by_anchor": ot_sampling_seeds,
            "max_ot_points": max_ot_points,
            "ot_cap_policy": (
                "identity-preserving shared cell indices for each aligned "
                "baseline/counterfactual cohort"
            ),
            "ot_indices_common_across_conditions_doses_and_on_off": True,
        },
        "pca_contract": {
            **pca_contract,
            "state_key": state_key,
            "loadings_key": loadings_key,
            "center_key": pca_center_key,
            "hvg_key": hvg_key,
            "expression_source": "adata.X",
            "edit_formula": (
                "delta_pca = (-knockdown_fraction * baseline_gene_expression) "
                "* gene_loading_vector"
            ),
        },
        "model": {
            "state_dict_sha256": _model_digest(model),
            "components": list(getattr(model, "components", [])),
            "interaction_group_size": configured_group_size,
            "interaction_cutoff": float(interaction_net.cutoff),
            "edge_predictor_threshold": float(interaction_net.edge_predictor_thre),
            "num_layers": int(len(interaction_net.gnn_layers)),
            "num_heads": int(interaction_net.gnn_layers[0].num_heads),
            "checkpoint": dict(checkpoint_provenance or {}),
        },
        "input": {
            "shape": [int(data.n_obs), int(data.n_vars)],
            "time_key": time_key,
            "cell_type_key": cell_type_key,
            "spatial_key": spatial_key,
            "state_key": state_key,
            "provenance": dict(input_provenance or {}),
        },
        "evidence_contract": {
            "link_predictor_recomputed_after_every_edit": True,
            "attention_recomputed_after_every_edit": True,
            "exact_complete_message_recomputed_after_every_edit": True,
            "complete_message_reconstructs_official_interaction_output": True,
            "primary_outcome": "counterfactual_D_target",
            "primary_outcome_definition": (
                "mean over every fixed receiver of the norm of summed exact "
                "generic complete GNN messages from baseline ligand-positive "
                "senders on the fixed receptor-positive receiver support"
            ),
            "primary_outcome_is_lr_specific_message_component": False,
            "support_is_lr_expression_conditioned": True,
            "fixed_target_support": True,
            "counterfactual_missing_support_edge_message": 0.0,
            "counterfactual_added_target_edges_in_primary_outcome": False,
            "dose_monotonicity_reported_without_enforcement": True,
            "attention_reported_separately_from_link_predictor": True,
            "attention_times_lr_expression_used_as_evidence": False,
            "interaction_mediation_control": (
                "interaction-on minus same-trained-model interaction-off "
                "difference in differences"
            ),
            "endpoint_metrics": ["W1", "W2", "centroid_shift"],
            "primary_endpoint_spaces": ["spatial", "state"],
            "joint_wasserstein_scale_dependent": True,
            "joint_wasserstein_cross_space_comparison_allowed": False,
            "message_alignment": {
                "status": "independent technical diagnostic only",
                "same_grouping_plan_as_rollout_driver": False,
                "eligible_for_primary_inference": False,
                "reason": (
                    "anchor exact-message audit uses NumPy grouping while the "
                    "rollout interaction uses torch.randperm grouping"
                ),
            },
            "primary_plot_includes_descriptive_matched_sham_central_95_percent_range": True,
            "matched_sham_inference": {
                "selection": "deterministic nearest matched HVGs",
                "exchangeable_random_draws": False,
                "formal_randomization_p_value_reported": False,
                "significance_claim_allowed": False,
                "reported_statistics": [
                    "descriptive_sham_upper_tail_fraction",
                    "descriptive_observed_relative_rank",
                    "descriptive_pseudocount_tail_score_not_formal_pvalue",
                ],
            },
        },
        "claim_bounds": {
            "trained_model_sensitivity": True,
            "experimental_perturbation": False,
            "experimental_causality": False,
            "biological_mechanism_proven": False,
            "language": (
                "Results quantify sensitivity of the trained CytoBridge model "
                "to projected gene-input edits."
            ),
        },
        "reproducibility": _reproducibility_provenance(),
        "checks": checks,
        "artifacts": artifacts,
        "manifest_path": "run_manifest.json",
        "checksums_path": "checksums.sha256",
        "axis_visibility_path": str(visibility_path.relative_to(output)),
        "checksum_contract": (
            "checksums.sha256 binds every artifact above plus run_manifest.json; "
            "the checksum file omits its own recursive hash"
        ),
    }
    manifest_path = output / "run_manifest.json"
    _write_json(manifest_path, manifest)

    checksum_paths = [*artifact_paths, manifest_path]
    checksum_path = output / "checksums.sha256"
    checksum_path.write_text(
        "".join(
            f"{_sha256(path)}  {path.relative_to(output)}\n"
            for path in sorted(checksum_paths, key=lambda item: str(item))
        ),
        encoding="utf-8",
    )
    return manifest


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    import anndata as ad
    import torch

    from CytoBridge.tl.downstream.checkpoint import load_dynamical_model_from_dir

    h5ad_path = args.h5ad.expanduser().resolve()
    model_dir = args.model_dir.expanduser().resolve()
    if not h5ad_path.is_file():
        raise FileNotFoundError(h5ad_path)
    if not model_dir.is_dir():
        raise FileNotFoundError(model_dir)
    data = ad.read_h5ad(h5ad_path)
    feature_dim = int(
        np.asarray(data.obsm[args.spatial_key]).shape[1]
        + np.asarray(data.obsm[args.state_key]).shape[1]
    )
    loaded = load_dynamical_model_from_dir(
        model_dir,
        dim=feature_dim,
        device=torch.device(args.device),
        stage=str(args.checkpoint_stage),
    )
    checkpoint = {
        "model_dir": str(model_dir),
        "requested_stage": str(args.checkpoint_stage),
        "weight_stage": loaded.weight_stage,
        "weight": _artifact(Path(loaded.weight_path)),
        "score_stage": loaded.score_stage,
        "score": (
            _artifact(Path(loaded.score_path))
            if loaded.score_path is not None
            else None
        ),
    }
    return run_analysis(
        data,
        loaded.model,
        output_dir=args.output_dir,
        ligand=args.ligand,
        receptor=args.receptor,
        anchors=args.anchors,
        fractions=args.fractions,
        n_shams=args.n_shams,
        time_key=args.time_key,
        cell_type_key=args.cell_type_key,
        spatial_key=args.spatial_key,
        state_key=args.state_key,
        loadings_key=args.loadings_key,
        hvg_key=args.hvg_key,
        pca_center_key=args.pca_center_key,
        group_size=args.group_size,
        dt=args.dt,
        grouping_seeds=args.grouping_seeds,
        metric_seed=args.metric_seed,
        max_ot_points=args.max_ot_points,
        receiver_threshold=args.receiver_threshold,
        pca_contract_atol=args.pca_contract_atol,
        pca_contract_rtol=args.pca_contract_rtol,
        device=args.device,
        overwrite=args.overwrite,
        input_provenance={"h5ad": _artifact(h5ad_path)},
        checkpoint_provenance=checkpoint,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = run(args)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "analysis": manifest["analysis"],
                "output": str(args.output_dir.expanduser().resolve()),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
