#!/usr/bin/env python3
"""Audit trained/pre-interaction/random attention on one frozen edge scaffold.

This is a deliberately narrow technical control for the zebrafish Jam2a--Jam3b
case study.  The pre-interaction condition must be attributed from
``Refine/best_model.pth``, before any interaction-module training; the
post-50-epoch ``Init_interaction/best_model.pth`` checkpoint is not a valid
pre-interaction control. Raw attention is ranked only within each model
condition because its numerical scale is not calibrated across trained,
pre-interaction, and random models. Cross-condition changes are therefore
computed only from within-condition edge percentiles on the exact same directed
Somite-to-Somite edge scaffold.

Jam compatibility is a post-hoc expression label: source ``jam2a > 0`` and
target ``jam3b > 0``, or the reverse orientation.  It is not a ligand-specific
attention head, intervention, biochemical flux, or causal test.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import fisher_exact, mannwhitneyu


CONDITIONS = ("trained", "pre_interaction", "random")
EDGE_REQUIRED_COLUMNS = (
    "stage",
    "stage_label",
    "grouping_seed",
    "source_index",
    "target_index",
    "sender_type",
    "receiver_type",
    "attention_abs_mean",
)
SCAFFOLD_COLUMNS = ("source_global_index", "target_global_index")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--h5ad", required=True, type=Path)
    result.add_argument("--observed-cells", required=True, type=Path)
    result.add_argument("--trained-edges", required=True, type=Path)
    result.add_argument(
        "--pre-interaction-edges",
        type=Path,
        help=(
            "Canonical pre-interaction attribution edges generated from "
            "Refine/best_model.pth."
        ),
    )
    result.add_argument(
        "--init-edges",
        type=Path,
        help=(
            "Deprecated CLI alias for --pre-interaction-edges. The supplied data "
            "must still come from Refine/best_model.pth, never "
            "Init_interaction/best_model.pth."
        ),
    )
    result.add_argument("--random-edges", required=True, type=Path)
    result.add_argument("--output-dir", required=True, type=Path)
    result.add_argument("--stage", type=float, default=3.0)
    result.add_argument("--stage-label", default="18hpf")
    result.add_argument("--grouping-seed", type=int, default=101)
    result.add_argument("--sender-type", default="Somite")
    result.add_argument("--receiver-type", default="Somite")
    result.add_argument("--time-key", default="time_point_processed")
    result.add_argument("--time-label-key", default="time")
    result.add_argument("--annotation-key", default="Annotation")
    result.add_argument("--jam2a-gene", default="jam2a")
    result.add_argument("--jam3b-gene", default="jam3b")
    result.add_argument("--overwrite", action="store_true")
    return result


def resolve_pre_interaction_edges(
    args: argparse.Namespace,
) -> tuple[Path, dict[str, object]]:
    """Resolve the canonical input while preserving the old CLI spelling."""
    canonical = args.pre_interaction_edges
    deprecated = args.init_edges
    if canonical is None and deprecated is None:
        raise ValueError(
            "One of --pre-interaction-edges or deprecated --init-edges is required."
        )
    deprecated_provided = deprecated is not None
    deprecated_used = canonical is None and deprecated_provided
    deprecated_ignored = canonical is not None and deprecated_provided
    selected = canonical if canonical is not None else deprecated
    if selected is None:  # pragma: no cover - guarded above
        raise RuntimeError("Failed to resolve pre-interaction edge input.")
    metadata = {
        "canonical_argument": "--pre-interaction-edges",
        "deprecated_alias": "--init-edges",
        "deprecated_alias_provided": deprecated_provided,
        "deprecated_alias_used": deprecated_used,
        "deprecated_alias_ignored_because_canonical_was_provided": (deprecated_ignored),
        "resolved_argument": (
            "--init-edges" if deprecated_used else "--pre-interaction-edges"
        ),
        "canonical_condition": "pre_interaction",
        "required_checkpoint": "Refine/best_model.pth",
        "forbidden_checkpoint_for_this_control": ("Init_interaction/best_model.pth"),
        "checkpoint_semantics_verified_from_edge_table": False,
        "checkpoint_semantics_responsibility": (
            "The attribution run supplying this table must record and verify "
            "checkpoint_stage=Refine."
        ),
    }
    return Path(selected), metadata


def require(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    missing = [column for column in columns if column not in frame]
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")


def integer_values(series: pd.Series, label: str) -> np.ndarray:
    values = pd.to_numeric(series, errors="raise").to_numpy(float)
    if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
        raise ValueError(f"{label} must contain finite integer values")
    return values.astype(int)


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, object]:
    resolved = path.expanduser().resolve()
    return {
        "path": str(resolved),
        "bytes": int(resolved.stat().st_size),
        "sha256": file_sha256(resolved),
    }


def json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def prepare_output(path: Path, overwrite: bool) -> Path:
    output = path.expanduser().resolve()
    if output.exists() and not output.is_dir():
        raise NotADirectoryError(output)
    if output.exists() and any(output.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory is not empty; pass --overwrite: {output}"
            )
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    return output


def observed_cell_mapping(
    data: ad.AnnData,
    path: Path,
    *,
    time_key: str,
    time_label_key: str,
    annotation_key: str,
) -> tuple[dict[int, int], pd.DataFrame, dict[str, object]]:
    """Resolve attribution-global IDs by obs_name and verify the full universe."""
    require(data.obs, [time_key, time_label_key, annotation_key], "H5AD obs")
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    observed = pd.read_csv(resolved)
    require(
        observed,
        ["global_index", "obs_name", "stage", "stage_label", "cell_type"],
        "observed-cells table",
    )
    observed = observed.copy()
    observed["global_index"] = integer_values(
        observed["global_index"], "observed-cells global_index"
    )
    if observed["global_index"].duplicated().any():
        raise ValueError("observed-cells global_index is not unique")
    if set(observed["global_index"]) != set(range(len(observed))):
        raise ValueError("observed-cells global_index must be a complete 0..N-1 grid")
    observed["obs_name"] = observed["obs_name"].astype(str)
    if observed["obs_name"].duplicated().any():
        raise ValueError("observed-cells obs_name is not unique")
    if not data.obs_names.is_unique:
        raise ValueError("H5AD obs_names must be unique")
    h5_names = data.obs_names.astype(str)
    if len(observed) != data.n_obs or set(observed["obs_name"]) != set(h5_names):
        raise ValueError(
            "observed-cells and H5AD must contain exactly the same obs_name universe"
        )
    h5_lookup = {name: index for index, name in enumerate(h5_names)}
    h5_index = observed["obs_name"].map(h5_lookup).to_numpy(int)
    observed_stage = pd.to_numeric(observed["stage"], errors="raise").to_numpy(float)
    h5_stage = pd.to_numeric(
        data.obs.iloc[h5_index][time_key], errors="raise"
    ).to_numpy(float)
    if not np.isclose(observed_stage, h5_stage, rtol=0.0, atol=1e-12).all():
        raise ValueError("observed-cells stage disagrees with H5AD")
    observed_label = observed["stage_label"].astype(str).to_numpy()
    h5_label = data.obs.iloc[h5_index][time_label_key].astype(str).to_numpy()
    if not np.array_equal(observed_label, h5_label):
        raise ValueError("observed-cells stage_label disagrees with H5AD")
    observed_type = observed["cell_type"].astype(str).to_numpy()
    h5_type = data.obs.iloc[h5_index][annotation_key].astype(str).to_numpy()
    if not np.array_equal(observed_type, h5_type):
        raise ValueError("observed-cells cell_type disagrees with H5AD")
    observed["h5ad_index"] = h5_index
    mapping = dict(zip(observed["global_index"], observed["h5ad_index"]))
    metadata = {
        "mode": "observed_cells_obs_name_to_h5ad",
        "observed_cells": artifact(resolved),
        "n_cells": int(len(observed)),
        "global_h5ad_row_order_assumed_without_validation": False,
    }
    return mapping, observed, metadata


def load_condition_edges(
    path: Path,
    condition: str,
    data: ad.AnnData,
    global_to_h5ad: Mapping[int, int],
    *,
    stage: float,
    stage_label: str,
    grouping_seed: int,
    time_key: str,
    annotation_key: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Load one control table and map its selected-stage endpoints by obs_name."""
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition: {condition}")
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    raw = pd.read_csv(resolved)
    require(raw, EDGE_REQUIRED_COLUMNS, f"{condition} edge controls")
    if raw.empty:
        raise ValueError(f"{condition} edge controls are empty")
    seeds = integer_values(raw["grouping_seed"], f"{condition} grouping_seed")
    if set(seeds) != {int(grouping_seed)}:
        raise ValueError(
            f"{condition} must contain only grouping_seed={grouping_seed}; "
            f"found {sorted(set(seeds))}"
        )
    stage_values = pd.to_numeric(raw["stage"], errors="raise").to_numpy(float)
    selected = raw.loc[
        np.isclose(stage_values, float(stage), rtol=0.0, atol=1e-12)
    ].copy()
    if selected.empty:
        raise ValueError(f"{condition} has no edges for stage={stage:g}")
    labels = set(selected["stage_label"].astype(str))
    if labels != {str(stage_label)}:
        raise ValueError(
            f"{condition} stage-label mismatch for stage={stage:g}: {sorted(labels)}"
        )
    source_global = integer_values(
        selected["source_index"], f"{condition} source_index"
    )
    target_global = integer_values(
        selected["target_index"], f"{condition} target_index"
    )
    missing = sorted(
        (set(source_global) | set(target_global)).difference(global_to_h5ad)
    )
    if missing:
        raise ValueError(
            f"{condition} edge endpoints are absent from observed-cells: {missing[:5]}"
        )
    source_h5 = np.asarray(
        [global_to_h5ad[index] for index in source_global], dtype=int
    )
    target_h5 = np.asarray(
        [global_to_h5ad[index] for index in target_global], dtype=int
    )
    if np.any(source_h5 == target_h5):
        raise ValueError(f"{condition} contains cell self-edges")
    h5_stage = pd.to_numeric(data.obs[time_key], errors="raise").to_numpy(float)
    if (
        not np.isclose(h5_stage[source_h5], stage, rtol=0.0, atol=1e-12).all()
        or not np.isclose(h5_stage[target_h5], stage, rtol=0.0, atol=1e-12).all()
    ):
        raise ValueError(f"{condition} endpoints map outside stage={stage:g}")
    h5_type = data.obs[annotation_key].astype(str).to_numpy()
    if not np.array_equal(
        selected["sender_type"].astype(str).to_numpy(), h5_type[source_h5]
    ):
        raise ValueError(f"{condition} sender_type disagrees with mapped H5AD cells")
    if not np.array_equal(
        selected["receiver_type"].astype(str).to_numpy(), h5_type[target_h5]
    ):
        raise ValueError(f"{condition} receiver_type disagrees with mapped H5AD cells")
    attention = pd.to_numeric(selected["attention_abs_mean"], errors="raise").to_numpy(
        float
    )
    if not np.isfinite(attention).all() or np.any(attention < 0):
        raise ValueError(
            f"{condition} attention_abs_mean must be finite and non-negative"
        )
    selected["condition"] = condition
    selected["source_global_index"] = source_global
    selected["target_global_index"] = target_global
    selected["source_h5ad_index"] = source_h5
    selected["target_h5ad_index"] = target_h5
    selected["source_obs_name"] = data.obs_names[source_h5].astype(str)
    selected["target_obs_name"] = data.obs_names[target_h5].astype(str)
    selected["attention_abs_mean"] = attention
    if selected.duplicated(list(SCAFFOLD_COLUMNS)).any():
        raise ValueError(
            f"{condition} contains duplicate directed edges at stage={stage:g}"
        )
    selected = selected.sort_values(list(SCAFFOLD_COLUMNS)).reset_index(drop=True)
    inventory = {
        "condition": condition,
        "input": artifact(resolved),
        "grouping_seed": int(grouping_seed),
        "stage": float(stage),
        "stage_label": str(stage_label),
        "n_stage_edges": int(len(selected)),
    }
    return selected, inventory


def validate_same_scaffold(frames: Mapping[str, pd.DataFrame]) -> None:
    """Require identical directed selected-stage edge keys in all conditions."""
    if set(frames) != set(CONDITIONS):
        raise ValueError(f"Expected conditions {CONDITIONS}; found {sorted(frames)}")
    reference = frames["trained"].set_index(list(SCAFFOLD_COLUMNS)).sort_index()
    reference_index = reference.index
    for condition in ("pre_interaction", "random"):
        candidate = frames[condition].set_index(list(SCAFFOLD_COLUMNS)).sort_index()
        if not candidate.index.equals(reference_index):
            missing = reference_index.difference(candidate.index)
            extra = candidate.index.difference(reference_index)
            raise ValueError(
                f"Directed edge scaffold mismatch for {condition}: "
                f"missing={len(missing)}, extra={len(extra)}"
            )
        for column in (
            "source_h5ad_index",
            "target_h5ad_index",
            "source_obs_name",
            "target_obs_name",
            "sender_type",
            "receiver_type",
        ):
            if not np.array_equal(
                candidate[column].to_numpy(), reference[column].to_numpy()
            ):
                raise ValueError(
                    f"Directed edge scaffold metadata mismatch for {condition}: {column}"
                )


def positive_gene(data: ad.AnnData, gene: str) -> np.ndarray:
    matches = np.flatnonzero(
        np.asarray(
            [str(name).casefold() == str(gene).casefold() for name in data.var_names]
        )
    )
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one case-insensitive H5AD gene match for {gene!r}; "
            f"found {len(matches)}"
        )
    values = data.X[:, int(matches[0])]
    values = values.toarray() if sparse.issparse(values) else np.asarray(values)
    values = np.asarray(values, dtype=float).reshape(-1)
    if not np.isfinite(values).all():
        raise ValueError(f"H5AD expression for {gene!r} contains non-finite values")
    return values > 0


def jam_compatibility(
    source: np.ndarray,
    target: np.ndarray,
    jam2a_positive: np.ndarray,
    jam3b_positive: np.ndarray,
) -> pd.DataFrame:
    """Label either orientation of the heterophilic Jam2a--Jam3b pair."""
    source = np.asarray(source, dtype=int)
    target = np.asarray(target, dtype=int)
    forward = jam2a_positive[source] & jam3b_positive[target]
    reverse = jam3b_positive[source] & jam2a_positive[target]
    orientation = np.select(
        [forward & reverse, forward, reverse],
        ["both", "source_jam2a_target_jam3b", "source_jam3b_target_jam2a"],
        default="none",
    )
    return pd.DataFrame(
        {
            "jam2a_to_jam3b_compatible": forward,
            "jam3b_to_jam2a_compatible": reverse,
            "jam_compatible": forward | reverse,
            "jam_compatible_orientation": orientation,
        }
    )


def complete_type_pair_ranks(
    edges: pd.DataFrame,
    annotation_levels: Sequence[str],
    *,
    condition: str,
) -> pd.DataFrame:
    """Rank raw mean attention within one condition on a complete directed grid."""
    levels = sorted(set(map(str, annotation_levels)))
    if not levels:
        raise ValueError("Cannot construct a type-pair grid without annotations")
    summary = edges.groupby(
        ["sender_type", "receiver_type"], observed=True, as_index=False
    ).agg(
        raw_attention_mean=("attention_abs_mean", "mean"),
        n_directed_edges=("attention_abs_mean", "size"),
    )
    grid = pd.MultiIndex.from_product(
        [levels, levels], names=["sender_type", "receiver_type"]
    ).to_frame(index=False)
    result = grid.merge(
        summary,
        on=["sender_type", "receiver_type"],
        how="left",
        validate="one_to_one",
    )
    result["zero_completed_no_edge"] = result["n_directed_edges"].isna()
    result["raw_attention_mean"] = result["raw_attention_mean"].fillna(0.0)
    result["n_directed_edges"] = result["n_directed_edges"].fillna(0).astype(int)
    result["condition"] = condition
    n_pairs = int(len(result))
    result["rank_from_top"] = (
        result["raw_attention_mean"].rank(method="min", ascending=False).astype(int)
    )
    result["rank_tie_count"] = (
        result.groupby("raw_attention_mean", dropna=False)["raw_attention_mean"]
        .transform("size")
        .astype(int)
    )
    result["n_complete_directed_type_pairs"] = n_pairs
    result["rank_over_n"] = result["rank_from_top"].astype(str) + "/" + str(n_pairs)
    result["rank_fraction"] = result["rank_from_top"] / n_pairs
    return result[
        [
            "condition",
            "sender_type",
            "receiver_type",
            "raw_attention_mean",
            "n_directed_edges",
            "zero_completed_no_edge",
            "rank_from_top",
            "rank_tie_count",
            "n_complete_directed_type_pairs",
            "rank_over_n",
            "rank_fraction",
        ]
    ]


def within_condition_percentiles(edges: pd.DataFrame) -> pd.DataFrame:
    """Add [0,1] average-rank percentiles within one edge scaffold."""
    result = edges.copy()
    n_edges = int(len(result))
    if n_edges == 0:
        raise ValueError("Cannot rank an empty edge scaffold")
    if n_edges == 1:
        percentile = np.asarray([0.5], dtype=float)
    else:
        rank = (
            result["attention_abs_mean"]
            .rank(method="average", ascending=True)
            .to_numpy(float)
        )
        percentile = (rank - 1.0) / (n_edges - 1.0)
    result["attention_percentile_within_condition_somite_scaffold"] = percentile
    result["n_somite_scaffold_edges"] = n_edges
    return result


def compatibility_percentile_summary(edges: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    condition = str(edges["condition"].iloc[0])
    column = "attention_percentile_within_condition_somite_scaffold"
    for compatible, label in ((True, "Jam-compatible"), (False, "non-compatible")):
        values = edges.loc[edges["jam_compatible"].eq(compatible), column].to_numpy(
            float
        )
        rows.append(
            {
                "condition": condition,
                "compatibility_class": label,
                "n_directed_edges": int(len(values)),
                "mean_attention_percentile": (
                    float(np.mean(values)) if len(values) else np.nan
                ),
                "median_attention_percentile": (
                    float(np.median(values)) if len(values) else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def quartile_compatibility(edges: pd.DataFrame) -> pd.DataFrame:
    """Compare compatibility in tie-preserving top and bottom rank quartiles."""
    percentile = edges[
        "attention_percentile_within_condition_somite_scaffold"
    ].to_numpy(float)
    top = percentile >= 0.75
    bottom = percentile <= 0.25
    compatible = edges["jam_compatible"].to_numpy(bool)
    a = int(np.sum(top & compatible))
    b = int(np.sum(top & ~compatible))
    c = int(np.sum(bottom & compatible))
    d = int(np.sum(bottom & ~compatible))
    top_n, bottom_n = int(np.sum(top)), int(np.sum(bottom))
    if top_n and bottom_n:
        fisher = fisher_exact([[a, b], [c, d]], alternative="two-sided")
        odds_ratio = float(fisher.statistic)
        fisher_p = float(fisher.pvalue)
    else:
        odds_ratio, fisher_p = np.nan, np.nan
    corrected_odds = float((a + 0.5) * (d + 0.5) / ((b + 0.5) * (c + 0.5)))
    return pd.DataFrame(
        [
            {
                "condition": str(edges["condition"].iloc[0]),
                "top_threshold_percentile_inclusive": 0.75,
                "bottom_threshold_percentile_inclusive": 0.25,
                "top_n_edges_after_boundary_ties": top_n,
                "top_n_jam_compatible": a,
                "top_n_non_compatible": b,
                "top_compatibility_rate": a / top_n if top_n else np.nan,
                "bottom_n_edges_after_boundary_ties": bottom_n,
                "bottom_n_jam_compatible": c,
                "bottom_n_non_compatible": d,
                "bottom_compatibility_rate": c / bottom_n if bottom_n else np.nan,
                "top_vs_bottom_odds_ratio": odds_ratio,
                "top_vs_bottom_haldane_anscombe_odds_ratio": corrected_odds,
                "fisher_exact_two_sided_p_descriptive_technical": fisher_p,
            }
        ]
    )


def trained_pre_interaction_percentile_delta(
    frames: Mapping[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare trained and pre-interaction by same-edge percentile change."""
    percentile = "attention_percentile_within_condition_somite_scaffold"
    identifier = [
        "source_global_index",
        "target_global_index",
        "source_obs_name",
        "target_obs_name",
        "sender_type",
        "receiver_type",
        "jam_compatible",
        "jam_compatible_orientation",
    ]
    trained = frames["trained"][identifier + [percentile]].rename(
        columns={percentile: "trained_attention_percentile"}
    )
    pre_interaction = frames["pre_interaction"][identifier + [percentile]].rename(
        columns={percentile: "pre_interaction_attention_percentile"}
    )
    delta = trained.merge(
        pre_interaction,
        on=identifier,
        how="inner",
        validate="one_to_one",
    )
    if len(delta) != len(trained) or len(delta) != len(pre_interaction):
        raise ValueError(
            "Trained/pre-interaction percentile tables do not share the exact "
            "scaffold"
        )
    delta["trained_minus_pre_interaction_attention_percentile"] = (
        delta["trained_attention_percentile"]
        - delta["pre_interaction_attention_percentile"]
    )
    compatible = delta.loc[
        delta["jam_compatible"],
        "trained_minus_pre_interaction_attention_percentile",
    ].to_numpy(float)
    non_compatible = delta.loc[
        ~delta["jam_compatible"],
        "trained_minus_pre_interaction_attention_percentile",
    ].to_numpy(float)
    available = bool(len(compatible) and len(non_compatible))
    if available:
        two_sided = mannwhitneyu(
            compatible, non_compatible, alternative="two-sided", method="auto"
        )
        greater = mannwhitneyu(
            compatible, non_compatible, alternative="greater", method="auto"
        )
        statistic = float(two_sided.statistic)
        p_two_sided = float(two_sided.pvalue)
        p_greater = float(greater.pvalue)
        reason = ""
    else:
        statistic = p_two_sided = p_greater = np.nan
        reason = "both Jam-compatible and non-compatible edge groups are required"
    test = pd.DataFrame(
        [
            {
                "comparison": (
                    "trained_minus_pre_interaction_within_condition_percentile_delta"
                ),
                "available": available,
                "unavailable_reason": reason,
                "n_jam_compatible_edges": int(len(compatible)),
                "n_non_compatible_edges": int(len(non_compatible)),
                "jam_compatible_mean_delta": (
                    float(np.mean(compatible)) if len(compatible) else np.nan
                ),
                "jam_compatible_median_delta": (
                    float(np.median(compatible)) if len(compatible) else np.nan
                ),
                "non_compatible_mean_delta": (
                    float(np.mean(non_compatible)) if len(non_compatible) else np.nan
                ),
                "non_compatible_median_delta": (
                    float(np.median(non_compatible)) if len(non_compatible) else np.nan
                ),
                "mann_whitney_u": statistic,
                "mann_whitney_two_sided_p_descriptive_technical": p_two_sided,
                "mann_whitney_greater_p_descriptive_technical": p_greater,
            }
        ]
    )
    return delta, test


def write_readme(path: Path) -> None:
    path.write_text(
        """# 18 hpf Somite Jam2a--Jam3b trained/pre-interaction/random control

This control uses one frozen grouping seed and requires trained,
pre-interaction, and random models to contain exactly the same directed 18 hpf
edge scaffold. Every attribution index is resolved through
`observed_cells.obs_name`; H5AD row order is never assumed.

The `pre_interaction` attribution **must** come from `Refine/best_model.pth`,
the checkpoint immediately before interaction-module training. It must not
come from `Init_interaction/best_model.pth`: despite its historical name, that
checkpoint is saved after the 50-epoch interaction-initialization stage and is
therefore not an untrained/pre-interaction control. `--init-edges` is retained
only as a deprecated spelling of `--pre-interaction-edges`; it does not change
this checkpoint requirement.

`type_pair_raw_attention_ranks.csv` reports the Somite-to-Somite raw-attention
mean rank, together with every other ordered 18 hpf annotation pair. Missing
type pairs are explicit structural zeros. Raw attention is ranked separately
inside each model condition and must not be compared numerically across models.

For the Somite-to-Somite edges, Jam-compatible means source `jam2a > 0` and
target `jam3b > 0`, or source `jam3b > 0` and target `jam2a > 0`, in the supplied
H5AD expression matrix. The compatibility tables compare within-condition
attention percentiles. Top and bottom quartiles use percentile thresholds of
0.75 and 0.25 and retain boundary ties. The
trained-minus-pre-interaction table subtracts only within-condition
percentiles on the same edge, never raw attention.

The odds ratios, Fisher test, and Mann--Whitney test are descriptive technical
controls. One grouping seed, model initializations, directed edges, and cells
are not biological replicates. These results do not establish Jam-specific
attention, direct molecular binding on an edge, intervention effects, or
causality.
""",
        encoding="utf-8",
    )


def main() -> None:
    args = parser().parse_args()
    pre_interaction_edges, cli_compatibility = resolve_pre_interaction_edges(args)
    output = prepare_output(args.output_dir, args.overwrite)
    tables = output / "tables"
    tables.mkdir()

    data = ad.read_h5ad(args.h5ad.expanduser().resolve())
    mapping, _observed, index_resolution = observed_cell_mapping(
        data,
        args.observed_cells,
        time_key=args.time_key,
        time_label_key=args.time_label_key,
        annotation_key=args.annotation_key,
    )
    stage_values = pd.to_numeric(data.obs[args.time_key], errors="raise").to_numpy(
        float
    )
    stage_mask = np.isclose(stage_values, args.stage, rtol=0.0, atol=1e-12)
    if not stage_mask.any():
        raise ValueError(f"H5AD has no cells for stage={args.stage:g}")
    stage_labels = set(data.obs.loc[stage_mask, args.time_label_key].astype(str))
    if stage_labels != {str(args.stage_label)}:
        raise ValueError(
            f"H5AD stage-label mismatch for stage={args.stage:g}: {sorted(stage_labels)}"
        )
    annotation_levels = sorted(
        data.obs.loc[stage_mask, args.annotation_key].astype(str).unique()
    )
    if (
        args.sender_type not in annotation_levels
        or args.receiver_type not in annotation_levels
    ):
        raise ValueError(
            f"Requested {args.sender_type}->{args.receiver_type} is absent at "
            f"stage={args.stage:g}"
        )

    paths = {
        "trained": args.trained_edges,
        "pre_interaction": pre_interaction_edges,
        "random": args.random_edges,
    }
    frames: dict[str, pd.DataFrame] = {}
    inventories: list[dict[str, object]] = []
    for condition in CONDITIONS:
        frame, inventory = load_condition_edges(
            paths[condition],
            condition,
            data,
            mapping,
            stage=args.stage,
            stage_label=args.stage_label,
            grouping_seed=args.grouping_seed,
            time_key=args.time_key,
            annotation_key=args.annotation_key,
        )
        frames[condition] = frame
        inventories.append(inventory)
    validate_same_scaffold(frames)
    n_stage_scaffold_edges = int(len(frames["trained"]))

    jam2a_positive = positive_gene(data, args.jam2a_gene)
    jam3b_positive = positive_gene(data, args.jam3b_gene)
    type_pair_tables: list[pd.DataFrame] = []
    somite_tables: list[pd.DataFrame] = []
    percentile_summaries: list[pd.DataFrame] = []
    quartile_summaries: list[pd.DataFrame] = []
    for condition in CONDITIONS:
        frame = frames[condition]
        type_pair_tables.append(
            complete_type_pair_ranks(frame, annotation_levels, condition=condition)
        )
        somite = frame.loc[
            frame["sender_type"].eq(args.sender_type)
            & frame["receiver_type"].eq(args.receiver_type)
        ].copy()
        if somite.empty:
            raise ValueError(
                f"{condition} has no {args.sender_type}->{args.receiver_type} edges"
            )
        compatibility = jam_compatibility(
            somite["source_h5ad_index"].to_numpy(int),
            somite["target_h5ad_index"].to_numpy(int),
            jam2a_positive,
            jam3b_positive,
        )
        for column in compatibility:
            somite[column] = compatibility[column].to_numpy()
        somite = within_condition_percentiles(somite)
        somite_tables.append(somite)
        percentile_summaries.append(compatibility_percentile_summary(somite))
        quartile_summaries.append(quartile_compatibility(somite))
        frames[condition] = somite

    reference_compatibility = frames["trained"][
        [*SCAFFOLD_COLUMNS, "jam_compatible", "jam_compatible_orientation"]
    ]
    for condition in ("pre_interaction", "random"):
        candidate = frames[condition][
            [*SCAFFOLD_COLUMNS, "jam_compatible", "jam_compatible_orientation"]
        ]
        if not candidate.equals(reference_compatibility):
            raise ValueError(
                f"Jam compatibility changed across the shared scaffold: {condition}"
            )

    delta, mann_whitney = trained_pre_interaction_percentile_delta(frames)
    type_pair_ranks = pd.concat(type_pair_tables, ignore_index=True)
    somite_percentiles = pd.concat(somite_tables, ignore_index=True)
    percentile_summary = pd.concat(percentile_summaries, ignore_index=True)
    quartile_summary = pd.concat(quartile_summaries, ignore_index=True)
    inventory_table = pd.DataFrame(inventories)

    output_paths = {
        "edge_input_inventory": tables / "edge_input_inventory.csv",
        "type_pair_raw_attention_ranks": tables / "type_pair_raw_attention_ranks.csv",
        "somite_edge_percentiles": tables / "somite_edge_percentiles.csv.gz",
        "compatibility_percentile_summary": (
            tables / "jam_compatibility_percentile_summary.csv"
        ),
        "quartile_compatibility": tables / "jam_quartile_compatibility.csv",
        "trained_pre_interaction_edge_percentile_delta": (
            tables / "trained_pre_interaction_edge_percentile_delta.csv.gz"
        ),
        "trained_pre_interaction_mann_whitney": (
            tables / "trained_pre_interaction_mann_whitney.csv"
        ),
    }
    inventory_table.to_csv(output_paths["edge_input_inventory"], index=False)
    type_pair_ranks.to_csv(output_paths["type_pair_raw_attention_ranks"], index=False)
    somite_percentiles.to_csv(
        output_paths["somite_edge_percentiles"], index=False, compression="gzip"
    )
    percentile_summary.to_csv(
        output_paths["compatibility_percentile_summary"], index=False
    )
    quartile_summary.to_csv(output_paths["quartile_compatibility"], index=False)
    delta.to_csv(
        output_paths["trained_pre_interaction_edge_percentile_delta"],
        index=False,
        compression="gzip",
    )
    mann_whitney.to_csv(
        output_paths["trained_pre_interaction_mann_whitney"], index=False
    )
    write_readme(output / "README.md")

    somite_rank_rows = type_pair_ranks.loc[
        type_pair_ranks["sender_type"].eq(args.sender_type)
        & type_pair_ranks["receiver_type"].eq(args.receiver_type)
    ]
    manifest = {
        "schema_version": 1,
        "analysis": (
            "zebrafish_18hpf_somite_jam_trained_pre_interaction_random_control"
        ),
        "status": "complete",
        "cli_compatibility": cli_compatibility,
        "inputs": {
            "h5ad": artifact(args.h5ad),
            "observed_cells": artifact(args.observed_cells),
            "edge_controls": {
                condition: artifact(paths[condition]) for condition in CONDITIONS
            },
        },
        "parameters": {
            "stage": float(args.stage),
            "stage_label": str(args.stage_label),
            "grouping_seed": int(args.grouping_seed),
            "sender_type": str(args.sender_type),
            "receiver_type": str(args.receiver_type),
            "jam2a_gene": str(args.jam2a_gene),
            "jam3b_gene": str(args.jam3b_gene),
        },
        "index_resolution": index_resolution,
        "definitions": {
            "jam_compatible": (
                "(source jam2a > 0 and target jam3b > 0) or "
                "(source jam3b > 0 and target jam2a > 0), using supplied H5AD X"
            ),
            "type_pair_rank": (
                "descending competition min-rank of raw mean attention within each "
                "condition on the complete ordered 18 hpf annotation square; "
                "type pairs without scaffold edges are zero"
            ),
            "edge_percentile": (
                "average-rank [0,1] percentile within one condition on the exact "
                "18 hpf Somite-to-Somite directed scaffold"
            ),
            "quartiles": (
                "top percentile >= 0.75 and bottom percentile <= 0.25; "
                "boundary ties retained"
            ),
            "trained_pre_interaction_delta": (
                "trained within-condition percentile minus pre_interaction "
                "within-condition percentile for the same directed edge; no "
                "raw-score subtraction"
            ),
        },
        "guardrails": {
            "same_directed_18hpf_scaffold_required": True,
            "pre_interaction_required_checkpoint": "Refine/best_model.pth",
            "init_interaction_checkpoint_is_valid_pre_interaction_control": False,
            "raw_attention_scale_compared_across_models": False,
            "single_grouping_seed_is_biological_replication": False,
            "control_models_are_biological_replicates": False,
            "directed_edges_or_cells_are_biological_replicates": False,
            "mann_whitney_or_fisher_is_confirmatory_inference": False,
            "statistics_are_technical_descriptive": True,
            "attention_is_jam_specific": False,
            "analysis_is_an_intervention_or_causal_test": False,
            "global_h5ad_row_order_was_silently_assumed": False,
        },
        "counts": {
            "n_stage_cells": int(stage_mask.sum()),
            "n_stage_annotations": int(len(annotation_levels)),
            "n_complete_directed_type_pairs_per_condition": int(
                len(annotation_levels) ** 2
            ),
            "n_stage_scaffold_edges": n_stage_scaffold_edges,
            "n_somite_somite_scaffold_edges": int(len(frames["trained"])),
            "n_jam_compatible_somite_edges": int(
                frames["trained"]["jam_compatible"].sum()
            ),
        },
        "somite_raw_attention_rank_over_n": {
            str(row.condition): str(row.rank_over_n)
            for row in somite_rank_rows.itertuples(index=False)
        },
        "outputs": {name: artifact(path) for name, path in output_paths.items()},
    }
    (output / "manifest.json").write_text(
        json.dumps(json_value(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "somite_raw_attention_rank_over_n": manifest[
                    "somite_raw_attention_rank_over_n"
                ],
                "n_somite_somite_scaffold_edges": manifest["counts"][
                    "n_somite_somite_scaffold_edges"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
