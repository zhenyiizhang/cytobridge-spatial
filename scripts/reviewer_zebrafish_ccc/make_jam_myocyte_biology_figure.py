#!/usr/bin/env python3
"""Build a biology-first zebrafish Jam2a/Jam3b myocyte-fusion figure.

The figure keeps three evidence layers visually and semantically separate:

1. observed atlas anatomy, gene detection, and spatial adjacency;
2. generic CytoBridge edge quantities filtered *post hoc* for Jam2a/Jam3b
   expression compatibility, plus independently supplied method ranks; and
3. the published experimental mechanism from Powell & Wright (2011) and
   Luo et al. (2022).

Neither spatial adjacency nor a high attention rank is called biochemical
communication.  The directed model arrows do not assign a polarized Jam2a to
Jam3b biochemical direction because both proteins are co-expressed by muscle
precursors.  Grouping seeds are technical sensitivity runs, not biological
replicates.  The 18 hpf to 24 hpf comparison is cross-sectional, not lineage.

Formal case-output schema
-------------------------
The preferred input is the unmodified output from
``jam_myocyte_case_study.py``.  Its manifest and these six tables are verified
by hash before rendering: ``somite_jam_case_summary.csv``,
``spatial_neighbor_enrichment.csv``, ``myog_association.csv``,
``expression_detection_by_stage_type.csv``, ``raw_type_pair_ranks.csv.gz``,
and ``trained_init_random_control.csv``.  An independently generated
``jam_trained_init_random_control.py`` bundle can be supplied when the formal
case copied an unavailable control placeholder.  The older one-row
``case_statistics.csv`` remains a legacy compatibility path.

Panel C deliberately uses the native ``type_pair_summary.csv`` raw-attention
mean, not the case study's specialized five-seed full-density rank.  When a
formal COMMOT output directory is supplied, it ranks
``abundance_controlled_distinct_cell_score`` from the complete
``commot_type_pair_scores.csv.gz`` directed type-pair square at both 18 and
12 hpf.  No rank is hard-coded.  CellAgentChat is optional; if displayed, only
the significant-score-sum construction is admitted and it is labelled as a
sensitivity analysis.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial.distance import pdist, squareform


JAM_AXES = (("jam2a", "jam3b"), ("jam3b", "jam2a"))
JAM_GENES = ("jam2a", "jam3b")
MARKER_GENES = ("jam2a", "jam3b", "myog", "mymk", "mylpfa", "acta1a", "tnnt3a")

SOMITE = "Somite"
FAST_MUSCLE = "Fast Muscle Cell"

FORMAL_CASE_FILES = {
    "somite_case_summary": "somite_jam_case_summary.csv",
    "spatial_neighbor_enrichment": "spatial_neighbor_enrichment.csv",
    "myog_association": "myog_association.csv",
    "expression_detection_by_stage_type": "expression_detection_by_stage_type.csv",
    "raw_type_pair_ranks": "raw_type_pair_ranks.csv.gz",
    "trained_init_random_control": "trained_init_random_control.csv",
}
FORMAL_SPATIAL_NULL_FILE = "somite_18hpf_spatial_null_iterations.csv.gz"

INK = "#252A30"
MUTED = "#747D87"
LIGHT = "#DDE1E6"
SOMITE_PURPLE = "#6A51A3"
JAM2_ORANGE = "#E58A2B"
JAM3_BLUE = "#3573B9"
COMPAT_TEAL = "#218C83"
PASS_GREEN = "#24856D"
FAIL_RED = "#BE4B44"
COMMOT_ORANGE = "#D9794A"
CELLAGENT_PURPLE = "#8055A5"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--h5ad", required=True, type=Path)
    result.add_argument("--attribution-dir", required=True, type=Path)
    result.add_argument(
        "--type-pair-summary",
        type=Path,
        help=(
            "Explicit native CytoBridge type_pair_summary.csv containing "
            "G_AB_attention_mean_mean. Defaults to attribution-dir discovery."
        ),
    )
    result.add_argument(
        "--observed-cells",
        type=Path,
        help="observed_cells.csv.gz; defaults to a file under attribution-dir.",
    )
    result.add_argument(
        "--commot-distinct-cell-output",
        type=Path,
        help=(
            "Formal COMMOT run directory containing manifest.json and "
            "commot_type_pair_scores.csv.gz."
        ),
    )
    result.add_argument(
        "--cellagentchat-output",
        type=Path,
        help=(
            "Optional ready CellAgentChat significant-score-sum rank table; "
            "rendered explicitly as a sensitivity analysis."
        ),
    )
    result.add_argument("--jam-case-output", required=True, type=Path)
    result.add_argument(
        "--trained-init-random-control-output",
        type=Path,
        help=(
            "Optional independent jam_trained_init_random_control.py bundle or "
            "type_pair_raw_attention_ranks.csv. Required when the formal JAM "
            "case contains only the unavailable-control placeholder."
        ),
    )
    result.add_argument("--provenance-csv", required=True, type=Path)
    result.add_argument("--output-dir", required=True, type=Path)
    result.add_argument("--stage", type=float, default=3.0)
    result.add_argument("--stage-label", default="18hpf")
    result.add_argument("--comparison-stage", type=float, default=2.0)
    result.add_argument("--comparison-stage-label", default="12hpf")
    result.add_argument("--later-stage", type=float, default=4.0)
    result.add_argument("--later-stage-label", default="24hpf")
    result.add_argument("--somite-label", default=SOMITE)
    result.add_argument("--later-cell-type", default=FAST_MUSCLE)
    result.add_argument("--time-key", default="time_point_processed")
    result.add_argument("--annotation-key", default="Annotation")
    result.add_argument("--spatial-key", default="spatial_aligned")
    result.add_argument("--min-edge-seed-support", type=int, default=3)
    result.add_argument("--max-display-edges", type=int, default=15)
    result.add_argument("--dpi", type=int, default=300)
    result.add_argument("--overwrite", action="store_true")
    return result


def require_columns(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    missing = [column for column in columns if column not in frame]
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")


def prepare_output(path: Path, overwrite: bool) -> None:
    path = path.expanduser().resolve()
    if path.exists():
        if not overwrite:
            raise FileExistsError(path)
        shutil.rmtree(path)
    path.mkdir(parents=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, object]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "bytes": int(path.stat().st_size), "sha256": sha256(path)}


def _candidate_file(root: Path, names: Iterable[str], *, required: bool) -> Path | None:
    root = root.expanduser().resolve()
    if root.is_file():
        return root
    candidates = []
    for name in names:
        candidates.extend((root / name, root / "tables" / name))
    existing = [path for path in candidates if path.is_file()]
    if len(existing) > 1:
        unique = {path.resolve() for path in existing}
        if len(unique) > 1:
            raise ValueError(f"Ambiguous table under {root}: {sorted(map(str, unique))}")
    if existing:
        return existing[0]
    if required:
        raise FileNotFoundError(f"None of {list(names)} found under {root}")
    return None


def _stat(row: pd.Series, *names: str, required: bool = True):
    for name in names:
        if name in row.index and pd.notna(row[name]):
            return row[name]
    if required:
        raise ValueError(f"Case statistics lack all aliases: {names}")
    return np.nan


def _as_bool(value) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"false", "0", "no"}:
        return False
    if text in {"true", "1", "yes"}:
        return True
    raise ValueError(f"Cannot parse boolean value: {value!r}")


def _bundle_root(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.name == "tables" and (resolved.parent / "manifest.json").is_file():
        return resolved.parent
    return resolved


def _manifest_artifacts(manifest: Mapping[str, object]) -> Mapping[str, object]:
    for name in ("artifacts", "outputs"):
        value = manifest.get(name)
        if isinstance(value, Mapping):
            return value
    raise ValueError("Bundle manifest lacks an artifacts/outputs mapping")


def _verify_manifest_artifacts(
    root: Path,
    paths: Mapping[str, Path],
    *,
    label: str,
) -> Path:
    bundle = _bundle_root(root)
    manifest_path = bundle / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"{label} requires manifest.json: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts = _manifest_artifacts(manifest)
    for logical_name, path in paths.items():
        record = artifacts.get(logical_name)
        if not isinstance(record, Mapping):
            matches = [
                value
                for value in artifacts.values()
                if isinstance(value, Mapping)
                and Path(str(value.get("path", ""))).name == path.name
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"{label} manifest has no unique artifact for {logical_name}: {path.name}"
                )
            record = matches[0]
        expected_hash = str(record.get("sha256", "")).strip()
        if not expected_hash or sha256(path) != expected_hash:
            raise ValueError(f"{label} hash mismatch for {logical_name}: {path}")
        if "bytes" in record and int(record["bytes"]) != int(path.stat().st_size):
            raise ValueError(f"{label} byte-count mismatch for {logical_name}: {path}")
    return manifest_path


def _requested_row(
    frame: pd.DataFrame,
    *,
    stage: float,
    cell_type: str,
    label: str,
) -> pd.Series:
    require_columns(frame, ["stage", "cell_type"], label)
    selected = frame.loc[
        np.isclose(
            pd.to_numeric(frame["stage"], errors="raise").to_numpy(float),
            float(stage),
            rtol=0.0,
            atol=1e-12,
        )
        & frame["cell_type"].astype(str).eq(cell_type)
    ]
    if len(selected) != 1:
        raise ValueError(
            f"{label} requires exactly one stage={stage:g}, cell_type={cell_type!r} row; "
            f"found {len(selected)}"
        )
    return selected.iloc[0]


def _control_rank_rows(
    path: Path,
    *,
    stage: float,
    somite_label: str,
) -> dict[str, tuple[int, int]]:
    frame = pd.read_csv(path)
    if "control_metrics_available" in frame:
        available = frame["control_metrics_available"].map(_as_bool)
        if not available.any():
            raise ValueError(
                "trained_init_random_control.csv is an unavailable placeholder; "
                "supply --trained-init-random-control-output"
            )
        frame = frame.loc[available].copy()
    condition_column = (
        "condition" if "condition" in frame else "control" if "control" in frame else None
    )
    if condition_column is None:
        raise ValueError("Control rank table requires condition/control")
    aliases = {
        "trained": "trained",
        "init": "init",
        "init_interaction": "init",
        "initial_interaction": "init",
        "random": "random",
        "randomized_interaction_seed17": "random",
        "random_interaction_seed17": "random",
    }
    canonical = (
        frame[condition_column]
        .astype(str)
        .str.casefold()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
        .map(aliases)
    )
    frame = frame.loc[canonical.notna()].copy()
    frame["_condition"] = canonical.loc[canonical.notna()].to_numpy()
    if "stage" in frame:
        frame = frame.loc[
            np.isclose(
                pd.to_numeric(frame["stage"], errors="raise").to_numpy(float),
                float(stage),
                rtol=0.0,
                atol=1e-12,
            )
        ]
    if "sender_type" in frame:
        frame = frame.loc[frame["sender_type"].astype(str).eq(somite_label)]
    if "receiver_type" in frame:
        frame = frame.loc[frame["receiver_type"].astype(str).eq(somite_label)]
    rank_column = next(
        (
            column
            for column in (
                "rank_from_top",
                "raw_attention_rank_from_top",
                "trained_somite_attention_rank",
            )
            if column in frame
        ),
        None,
    )
    n_column = next(
        (
            column
            for column in (
                "n_complete_directed_type_pairs",
                "n_ranked_contexts",
                "raw_attention_n_ranked_contexts",
                "trained_somite_attention_n_contexts",
            )
            if column in frame
        ),
        None,
    )
    if rank_column is None or n_column is None:
        raise ValueError("Control rank table lacks rank-from-top and rank-universe columns")
    result: dict[str, tuple[int, int]] = {}
    for condition in ("trained", "init", "random"):
        selected = frame.loc[frame["_condition"].eq(condition)]
        if len(selected) != 1:
            raise ValueError(f"Control rank table requires one {condition} Somite row")
        rank = int(pd.to_numeric(selected.iloc[0][rank_column], errors="raise"))
        n_contexts = int(pd.to_numeric(selected.iloc[0][n_column], errors="raise"))
        if not 1 <= rank <= n_contexts:
            raise ValueError(f"Invalid control rank for {condition}: {rank}/{n_contexts}")
        result[condition] = (rank, n_contexts)
    return result


def _resolve_control_rank_source(
    case_control_path: Path,
    external_control: Path | None,
    *,
    stage: float,
    somite_label: str,
) -> tuple[dict[str, tuple[int, int]], dict[str, Path]]:
    if external_control is None:
        return _control_rank_rows(
            case_control_path,
            stage=stage,
            somite_label=somite_label,
        ), {"trained_init_random_control": case_control_path}
    root = external_control.expanduser().resolve()
    table = _candidate_file(
        root,
        ("type_pair_raw_attention_ranks.csv", "trained_init_random_control.csv"),
        required=True,
    )
    assert table is not None
    sources = {"trained_init_random_control": table}
    if root.is_dir():
        manifest_path = _verify_manifest_artifacts(
            root,
            {"type_pair_raw_attention_ranks": table},
            label="independent trained/init/random control",
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        guardrails = manifest.get("guardrails")
        if not isinstance(guardrails, Mapping):
            raise ValueError("Control manifest lacks guardrails")
        for key in (
            "raw_attention_scale_compared_across_models",
            "single_grouping_seed_is_biological_replication",
            "control_models_are_biological_replicates",
            "attention_is_jam_specific",
            "analysis_is_an_intervention_or_causal_test",
        ):
            if key not in guardrails or _as_bool(guardrails[key]):
                raise ValueError(f"Control manifest requires guardrails.{key}=false")
        sources["trained_init_random_control_manifest"] = manifest_path
    return _control_rank_rows(table, stage=stage, somite_label=somite_label), sources


def load_case_statistics(
    root: Path,
    *,
    trained_init_random_control_output: Path | None,
    stage: float,
    stage_label: str,
    somite_label: str,
) -> tuple[pd.Series, dict[str, Path], str]:
    legacy = _candidate_file(
        root,
        ("case_statistics.csv", "jam_case_statistics.csv"),
        required=False,
    )
    required_fields = (
        "spatial_cutoff",
        "n_somite_neighbor_pairs",
        "n_jam_compatible_neighbor_pairs",
        "neighbor_shuffle_null_mean",
        "neighbor_fold_enrichment",
        "neighbor_monte_carlo_tail_fraction",
        "jam3b_myog_fisher_or",
        "jam3b_myog_fisher_p",
        "trained_somite_attention_rank",
        "trained_somite_attention_n_contexts",
        "init_somite_attention_rank",
        "init_somite_attention_n_contexts",
        "random_somite_attention_rank",
        "random_somite_attention_n_contexts",
        "jam_edge_enrichment_training_specific",
    )
    if legacy is not None:
        frame = pd.read_csv(legacy)
        if len(frame) != 1:
            raise ValueError("Legacy JAM case statistics must contain exactly one row")
        row = frame.iloc[0]
        missing = [field for field in required_fields if field not in row.index]
        if missing:
            raise ValueError(f"Legacy JAM case statistics are missing fields: {missing}")
        if _as_bool(row["jam_edge_enrichment_training_specific"]):
            raise ValueError(
                "Current audit requires jam_edge_enrichment_training_specific=false"
            )
        return row, {"legacy_case_statistics": legacy}, "legacy_case_statistics_v1"

    formal_paths: dict[str, Path] = {}
    for logical_name, filename in FORMAL_CASE_FILES.items():
        resolved = _candidate_file(root, (filename,), required=True)
        assert resolved is not None
        formal_paths[logical_name] = resolved
    spatial_null_path = _candidate_file(
        root,
        (FORMAL_SPATIAL_NULL_FILE,),
        required=True,
    )
    assert spatial_null_path is not None
    formal_paths["somite_spatial_null"] = spatial_null_path
    manifest_path = _verify_manifest_artifacts(
        root,
        formal_paths,
        label="formal JAM case",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    guardrails = manifest.get("claim_guardrails")
    if not isinstance(guardrails, Mapping):
        raise ValueError("Formal JAM manifest lacks claim_guardrails")
    for key in (
        "attention_is_lr_specific",
        "attention_is_communication_probability",
        "attention_lr_is_native_model_output",
        "directed_model_orientation_is_polarized_jam_biochemistry",
        "cross_sectional_association_is_regulatory_proof",
        "spatial_enrichment_is_direct_contact_proof",
        "analysis_is_causal_or_perturbational",
        "fast_muscle_24hpf_comparison_is_lineage_tracing",
    ):
        if key not in guardrails or _as_bool(guardrails[key]):
            raise ValueError(f"Formal JAM manifest requires claim_guardrails.{key}=false")

    summary = pd.read_csv(formal_paths["somite_case_summary"])
    require_columns(
        summary,
        [
            "stage",
            "sender_type",
            "receiver_type",
            "ligand",
            "receptor",
            "causal_or_lr_specific_claim_allowed",
        ],
        "formal Somite JAM summary",
    )
    summary = summary.loc[
        np.isclose(pd.to_numeric(summary["stage"], errors="raise"), float(stage))
        & summary["sender_type"].astype(str).eq(somite_label)
        & summary["receiver_type"].astype(str).eq(somite_label)
    ].copy()
    observed_axes = set(zip(summary["ligand"].astype(str), summary["receptor"].astype(str)))
    if len(summary) != 2 or summary.duplicated(["ligand", "receptor"]).any():
        raise ValueError("Formal Somite JAM summary requires exactly two unique reciprocal rows")
    if observed_axes != set(JAM_AXES):
        raise ValueError(f"Formal Somite JAM summary lacks reciprocal axes: {observed_axes}")
    if summary["causal_or_lr_specific_claim_allowed"].map(_as_bool).any():
        raise ValueError("Formal Somite JAM summary must disallow LR-specific causality")
    require_columns(
        summary,
        [
            "lr_only_rank_from_top",
            "lr_only_n_ranked_contexts",
            "attention_lr_rank_from_top",
            "attention_lr_n_ranked_contexts",
            "exact_message_lr_rank_from_top",
            "exact_message_lr_n_ranked_contexts",
        ],
        "formal Somite JAM summary",
    )
    for row in summary.itertuples(index=False):
        ranks = {
            int(row.lr_only_rank_from_top),
            int(row.attention_lr_rank_from_top),
            int(row.exact_message_lr_rank_from_top),
        }
        universes = {
            int(row.lr_only_n_ranked_contexts),
            int(row.attention_lr_n_ranked_contexts),
            int(row.exact_message_lr_n_ranked_contexts),
        }
        if len(ranks) != 1 or len(universes) != 1:
            raise ValueError(
                "Formal reciprocal-JAM ordering no longer matches LR-only; "
                "the no-attention-specific-gain interpretation must be revisited"
            )

    spatial = pd.read_csv(formal_paths["spatial_neighbor_enrichment"])
    spatial_row = _requested_row(
        spatial,
        stage=stage,
        cell_type=somite_label,
        label="formal spatial neighbor enrichment",
    )
    require_columns(
        spatial,
        [
            "spatial_cutoff",
            "n_distinct_undirected_neighbor_pairs",
            "observed_jam2a_jam3b_orientation_compatible_pairs",
            "n_permutations",
            "permutation_seed",
            "null_mean",
            "observed_over_null_mean",
            "n_null_at_least_observed",
            "monte_carlo_upper_tail_p_plus1",
        ],
        "formal spatial neighbor enrichment",
    )
    spatial_null = pd.read_csv(formal_paths["somite_spatial_null"])
    require_columns(
        spatial_null,
        [
            "iteration",
            "permutation_seed",
            "orientation_compatible_pair_count",
            "at_least_observed",
        ],
        "formal spatial null iterations",
    )
    n_permutations = int(spatial_row["n_permutations"])
    if len(spatial_null) != n_permutations:
        raise ValueError(
            f"Spatial null length disagrees with summary: {len(spatial_null)} vs {n_permutations}"
        )
    iterations = pd.to_numeric(spatial_null["iteration"], errors="raise").astype(int)
    if not np.array_equal(iterations.to_numpy(), np.arange(1, n_permutations + 1)):
        raise ValueError("Spatial null iteration IDs must be the complete 1..N sequence")
    null_seed = pd.to_numeric(spatial_null["permutation_seed"], errors="raise").astype(int)
    if set(null_seed) != {int(spatial_row["permutation_seed"])}:
        raise ValueError("Spatial null permutation seed disagrees with summary")
    null_values = pd.to_numeric(
        spatial_null["orientation_compatible_pair_count"], errors="raise"
    ).to_numpy(float)
    if not np.isfinite(null_values).all() or not np.isclose(
        np.mean(null_values), float(spatial_row["null_mean"]), rtol=0.0, atol=1e-12
    ):
        raise ValueError("Spatial null mean disagrees with summary")
    at_least = spatial_null["at_least_observed"].map(_as_bool).to_numpy(bool)
    observed_count = int(spatial_row["observed_jam2a_jam3b_orientation_compatible_pairs"])
    if not np.array_equal(at_least, null_values >= observed_count):
        raise ValueError("Spatial null at_least_observed flags disagree with counts")
    if int(at_least.sum()) != int(spatial_row["n_null_at_least_observed"]):
        raise ValueError("Spatial null exceedance count disagrees with summary")
    expected_tail = (int(at_least.sum()) + 1) / (n_permutations + 1)
    if not np.isclose(
        expected_tail,
        float(spatial_row["monte_carlo_upper_tail_p_plus1"]),
        rtol=0.0,
        atol=1e-15,
    ):
        raise ValueError("Spatial Monte Carlo +1 tail fraction disagrees with null iterations")
    expected_fold = observed_count / float(spatial_row["null_mean"])
    if not np.isclose(
        expected_fold,
        float(spatial_row["observed_over_null_mean"]),
        rtol=1e-12,
        atol=1e-12,
    ):
        raise ValueError("Spatial observed/null fold enrichment disagrees with summary")
    association = pd.read_csv(formal_paths["myog_association"])
    require_columns(
        association,
        ["stage", "cell_type", "gene_a", "gene_b", "fisher_odds_ratio", "fisher_two_sided_p"],
        "formal myog association",
    )
    selected_association = association.loc[
        np.isclose(pd.to_numeric(association["stage"], errors="raise"), float(stage))
        & association["cell_type"].astype(str).eq(somite_label)
        & association["gene_a"].astype(str).str.casefold().eq("jam3b")
        & association["gene_b"].astype(str).str.casefold().eq("myog")
    ]
    if len(selected_association) != 1:
        raise ValueError("Formal myog association requires one jam3b–myog Somite row")
    association_row = selected_association.iloc[0]

    specialized = pd.read_csv(formal_paths["raw_type_pair_ranks"])
    require_columns(
        specialized,
        [
            "stage",
            "sender_type",
            "receiver_type",
            "raw_attention_full_type_pair_rank_from_top",
            "raw_attention_full_type_pair_n_ranked_contexts",
        ],
        "formal specialized raw-density ranks",
    )
    specialized_row = _requested_row(
        specialized,
        stage=stage,
        cell_type=somite_label,
        label="formal specialized raw-density ranks",
    ) if "cell_type" in specialized else None
    if specialized_row is None:
        selected_specialized = specialized.loc[
            np.isclose(pd.to_numeric(specialized["stage"], errors="raise"), float(stage))
            & specialized["sender_type"].astype(str).eq(somite_label)
            & specialized["receiver_type"].astype(str).eq(somite_label)
        ]
        if len(selected_specialized) != 1:
            raise ValueError("Formal specialized raw-density table requires one Somite→Somite row")
        specialized_row = selected_specialized.iloc[0]

    controls, control_sources = _resolve_control_rank_source(
        formal_paths["trained_init_random_control"],
        trained_init_random_control_output,
        stage=stage,
        somite_label=somite_label,
    )
    if controls["trained"][0] != controls["init"][0]:
        raise ValueError(
            "Current audited JAM claim requires the trained and initialization control ranks "
            "to match; a changed result needs a new interpretation rather than silent reuse"
        )
    row = pd.Series(
        {
            "spatial_cutoff": float(spatial_row["spatial_cutoff"]),
            "n_somite_neighbor_pairs": int(spatial_row["n_distinct_undirected_neighbor_pairs"]),
            "n_jam_compatible_neighbor_pairs": int(
                spatial_row["observed_jam2a_jam3b_orientation_compatible_pairs"]
            ),
            "neighbor_shuffle_null_mean": float(spatial_row["null_mean"]),
            "neighbor_fold_enrichment": float(spatial_row["observed_over_null_mean"]),
            "neighbor_monte_carlo_tail_fraction": float(
                spatial_row["monte_carlo_upper_tail_p_plus1"]
            ),
            "neighbor_n_permutations": n_permutations,
            "neighbor_permutation_seed": int(spatial_row["permutation_seed"]),
            "jam3b_myog_fisher_or": float(association_row["fisher_odds_ratio"]),
            "jam3b_myog_fisher_p": float(association_row["fisher_two_sided_p"]),
            "trained_somite_attention_rank": controls["trained"][0],
            "trained_somite_attention_n_contexts": controls["trained"][1],
            "init_somite_attention_rank": controls["init"][0],
            "init_somite_attention_n_contexts": controls["init"][1],
            "random_somite_attention_rank": controls["random"][0],
            "random_somite_attention_n_contexts": controls["random"][1],
            "jam_edge_enrichment_training_specific": False,
            "training_specificity_supported": False,
            "jam_rank_order_matches_lr_only": True,
            "attention_specific_jam_rank_gain_supported": False,
            "specialized_full_density_rank_not_used_in_panel_c": int(
                specialized_row["raw_attention_full_type_pair_rank_from_top"]
            ),
            "specialized_full_density_n_not_used_in_panel_c": int(
                specialized_row["raw_attention_full_type_pair_n_ranked_contexts"]
            ),
            "case_stage_label": stage_label,
        }
    )
    sources = {
        **{f"formal_case_{name}": path for name, path in formal_paths.items()},
        **control_sources,
        "formal_jam_case_manifest": manifest_path,
    }
    return row, sources, "formal_jam_case_bundle_v1"


def dense_gene(data: ad.AnnData, gene: str) -> np.ndarray:
    if gene not in data.var_names:
        raise KeyError(gene)
    values = data.X[:, int(data.var_names.get_loc(gene))]
    values = values.toarray() if sparse.issparse(values) else np.asarray(values)
    return np.asarray(values, dtype=float).reshape(-1)


def q95_scale(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    positive = values[values > 0]
    scale = float(np.quantile(positive, 0.95)) if positive.size else 1.0
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    return np.clip(values / scale, 0.0, 1.0)


def load_marker_detection(
    case_root: Path,
    data: ad.AnnData,
    *,
    time_key: str,
    annotation_key: str,
    stage: float,
    stage_label: str,
    cell_type: str,
    comparison_stage: float,
    comparison_stage_label: str,
    comparison_cell_type: str,
    later_stage: float,
    later_stage_label: str,
    later_cell_type: str,
) -> tuple[pd.DataFrame, Path | None]:
    path = _candidate_file(
        case_root,
        (
            "expression_detection_by_stage_type.csv",
            "stage_marker_detection.csv",
            "marker_detection.csv",
        ),
        required=False,
    )
    missing = [gene for gene in MARKER_GENES if gene not in data.var_names]
    if missing:
        raise ValueError(f"H5AD lacks required marker genes: {missing}")
    require_columns(data.obs, [time_key, annotation_key], "H5AD obs")
    stage_values = pd.to_numeric(data.obs[time_key], errors="raise").to_numpy(float)
    annotations = data.obs[annotation_key].astype(str).to_numpy()
    rows = []
    for stage_value, label, group in (
        (comparison_stage, comparison_stage_label, comparison_cell_type),
        (stage, stage_label, cell_type),
        (later_stage, later_stage_label, later_cell_type),
    ):
        mask = np.isclose(stage_values, stage_value, rtol=0.0, atol=1e-12) & (
            annotations == group
        )
        if int(mask.sum()) == 0:
            raise ValueError(f"No cells for {label} / {group}")
        for gene in MARKER_GENES:
            values = dense_gene(data, gene)[mask]
            n_detected = int(np.sum(values > 0))
            rows.append(
                {
                    "stage": float(stage_value),
                    "stage_label": label,
                    "cell_type": group,
                    "n_cells": int(mask.sum()),
                    "gene": gene,
                    "n_detected": n_detected,
                    "detected_fraction": float(n_detected / mask.sum()),
                }
            )
    frame = pd.DataFrame(rows)
    frame["formal_case_table_value_available"] = False
    frame["value_source"] = "computed from supplied H5AD X > 0"

    if path is not None:
        supplied = pd.read_csv(path)
        rename = {}
        for source in ("detected_fraction_x_gt_zero", "positive_fraction", "detection_fraction"):
            if source in supplied and "detected_fraction" not in supplied:
                rename[source] = "detected_fraction"
        supplied = supplied.rename(columns=rename)
        require_columns(
            supplied,
            ["stage", "stage_label", "cell_type", "n_cells", "gene", "detected_fraction"],
            "JAM marker detection",
        )
        supplied = supplied.loc[supplied["gene"].astype(str).isin(MARKER_GENES)].copy()
        supplied["stage"] = pd.to_numeric(supplied["stage"], errors="raise").astype(float)
        wanted = {
            (float(comparison_stage), comparison_cell_type),
            (float(stage), cell_type),
            (float(later_stage), later_cell_type),
        }
        supplied = supplied.loc[
            [
                (float(row.stage), str(row.cell_type)) in wanted
                for row in supplied.itertuples(index=False)
            ]
        ].copy()
        if supplied.duplicated(["stage", "cell_type", "gene"]).any():
            raise ValueError("JAM marker table has duplicate stage/cell-type/gene rows")
        required_formal_keys = {
            *{
                (float(stage), cell_type, gene)
                for gene in ("jam2a", "jam3b", "myog", "mymk")
            },
            *{
                (float(later_stage), later_cell_type, gene)
                for gene in MARKER_GENES
            },
        }
        observed_formal_keys = set(
            supplied[["stage", "cell_type", "gene"]].itertuples(index=False, name=None)
        )
        missing_formal_keys = sorted(required_formal_keys.difference(observed_formal_keys))
        if missing_formal_keys:
            raise ValueError(
                "Formal marker table lacks required 18 hpf fusion-window or 24 hpf "
                f"maturation rows: {missing_formal_keys}"
            )
        fraction = pd.to_numeric(supplied["detected_fraction"], errors="raise")
        if fraction.max() > 1.0:
            fraction = fraction / 100.0
        supplied["detected_fraction"] = fraction
        if not supplied["detected_fraction"].between(0.0, 1.0).all():
            raise ValueError("Supplied marker fractions must be in [0, 1]")
        lookup = frame.set_index(["stage", "cell_type", "gene"])
        for row in supplied.itertuples(index=False):
            key = (float(row.stage), str(row.cell_type), str(row.gene))
            if key not in lookup.index:
                raise ValueError(f"Unexpected formal marker key: {key}")
            expected = lookup.loc[key]
            if int(row.n_cells) != int(expected["n_cells"]):
                raise ValueError(f"Formal marker n_cells disagrees with H5AD for {key}")
            if not np.isclose(
                float(row.detected_fraction),
                float(expected["detected_fraction"]),
                rtol=0.0,
                atol=1e-12,
            ):
                raise ValueError(f"Formal marker fraction disagrees with H5AD for {key}")
            selected = (
                np.isclose(frame["stage"], key[0])
                & frame["cell_type"].astype(str).eq(key[1])
                & frame["gene"].astype(str).eq(key[2])
            )
            frame.loc[selected, "formal_case_table_value_available"] = True
            frame.loc[selected, "value_source"] = "hash-verified formal case table"
    if not frame["detected_fraction"].between(0.0, 1.0).all():
        raise ValueError("Marker detection fractions must be in [0, 1]")
    return frame.reset_index(drop=True), path


def resolve_observed_cells(attribution_dir: Path, supplied: Path | None) -> Path:
    if supplied is not None:
        path = supplied.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    root = attribution_dir.expanduser().resolve()
    candidates = [root / "observed_cells.csv.gz", root.parent / "observed_cells.csv.gz"]
    existing = [path for path in candidates if path.is_file()]
    if len(existing) != 1:
        raise FileNotFoundError(f"Expected one observed_cells.csv.gz near {root}: {existing}")
    return existing[0]


def observed_mapping(
    path: Path,
    data: ad.AnnData,
    *,
    time_key: str,
    annotation_key: str,
) -> dict[int, int]:
    frame = pd.read_csv(path)
    require_columns(frame, ["global_index", "obs_name", "stage", "cell_type"], "observed cells")
    global_index = pd.to_numeric(frame["global_index"], errors="raise").astype(int)
    if global_index.duplicated().any() or frame["obs_name"].astype(str).duplicated().any():
        raise ValueError("observed-cells indices and names must be unique")
    lookup = {name: index for index, name in enumerate(data.obs_names.astype(str))}
    h5_index = frame["obs_name"].astype(str).map(lookup)
    if h5_index.isna().any():
        raise ValueError("observed-cells contains names absent from H5AD")
    h5_index = h5_index.astype(int)
    require_columns(data.obs, [time_key, annotation_key], "H5AD obs")
    h5_stage = pd.to_numeric(data.obs.iloc[h5_index][time_key], errors="raise")
    if not np.isclose(
        pd.to_numeric(frame["stage"], errors="raise"), h5_stage, rtol=0.0, atol=1e-12
    ).all():
        raise ValueError("observed-cells stage disagrees with H5AD")
    if not np.array_equal(
        frame["cell_type"].astype(str).to_numpy(),
        data.obs.iloc[h5_index][annotation_key].astype(str).to_numpy(),
    ):
        raise ValueError("observed-cells cell_type disagrees with H5AD")
    return dict(zip(global_index.astype(int), h5_index.astype(int)))


def resolve_edge_paths(root: Path, stage: float, stage_label: str) -> list[Path]:
    root = root.expanduser().resolve()
    stage_dirs = [path for path in root.glob(f"stage_{stage:g}_*") if path.is_dir()]
    if stage_dirs:
        matching = [path for path in stage_dirs if stage_label in path.name]
        if len(matching) != 1:
            raise FileNotFoundError(f"Ambiguous stage directories: {stage_dirs}")
        paths = sorted(matching[0].glob("edges_seed_*.csv.gz"))
    else:
        paths = sorted(root.glob("edges_seed_*.csv.gz"))
    if not paths:
        raise FileNotFoundError(f"No edges_seed_*.csv.gz under {root}")
    return paths


def load_stage_edges(
    root: Path,
    data: ad.AnnData,
    mapping: Mapping[int, int],
    *,
    stage: float,
    stage_label: str,
    time_key: str,
    annotation_key: str,
) -> tuple[pd.DataFrame, list[Path]]:
    frames = []
    paths = resolve_edge_paths(root, stage, stage_label)
    stage_values = pd.to_numeric(data.obs[time_key], errors="raise").to_numpy(float)
    annotations = data.obs[annotation_key].astype(str).to_numpy()
    for path in paths:
        frame = pd.read_csv(path)
        require_columns(
            frame,
            [
                "stage",
                "stage_label",
                "grouping_seed",
                "source_index",
                "target_index",
                "sender_type",
                "receiver_type",
                "attention_abs_mean",
                "spatial_distance",
            ],
            str(path),
        )
        frame = frame.loc[
            np.isclose(
                pd.to_numeric(frame["stage"], errors="raise").to_numpy(float),
                float(stage),
                rtol=0.0,
                atol=1e-12,
            )
            & frame["stage_label"].astype(str).eq(stage_label)
        ].copy()
        if frame.empty:
            continue
        source = pd.to_numeric(frame["source_index"], errors="raise").astype(int).map(mapping)
        target = pd.to_numeric(frame["target_index"], errors="raise").astype(int).map(mapping)
        if source.isna().any() or target.isna().any():
            raise ValueError(f"Unmapped edge endpoints in {path}")
        frame["_source_h5"] = source.astype(int).to_numpy()
        frame["_target_h5"] = target.astype(int).to_numpy()
        if frame["_source_h5"].eq(frame["_target_h5"]).any():
            raise ValueError("Self edge found; JAM panel requires distinct cells")
        if not np.isclose(stage_values[frame["_source_h5"]], stage).all() or not np.isclose(
            stage_values[frame["_target_h5"]], stage
        ).all():
            raise ValueError("Mapped edge endpoint is outside the requested stage")
        if not np.array_equal(frame["sender_type"].astype(str), annotations[frame["_source_h5"]]):
            raise ValueError("sender_type disagrees with mapped H5AD cells")
        if not np.array_equal(frame["receiver_type"].astype(str), annotations[frame["_target_h5"]]):
            raise ValueError("receiver_type disagrees with mapped H5AD cells")
        frames.append(frame)
    if not frames:
        raise ValueError(f"No edge rows for stage {stage_label}")
    result = pd.concat(frames, ignore_index=True)
    if result.duplicated(["grouping_seed", "_source_h5", "_target_h5"]).any():
        raise ValueError("Duplicate directed edge within a grouping seed")
    return result, paths


def spatial_cell_table(
    data: ad.AnnData,
    *,
    stage: float,
    time_key: str,
    annotation_key: str,
    spatial_key: str,
    somite_label: str,
) -> pd.DataFrame:
    require_columns(data.obs, [time_key, annotation_key], "H5AD obs")
    if spatial_key not in data.obsm:
        raise KeyError(f"Missing adata.obsm[{spatial_key!r}]")
    stage_values = pd.to_numeric(data.obs[time_key], errors="raise").to_numpy(float)
    mask = np.isclose(stage_values, stage, rtol=0.0, atol=1e-12)
    global_index = np.flatnonzero(mask)
    coordinates = np.asarray(data.obsm[spatial_key], dtype=float)[global_index, :2]
    annotations = data.obs.iloc[global_index][annotation_key].astype(str).to_numpy()
    jam2 = q95_scale(dense_gene(data, "jam2a"))[global_index]
    jam3 = q95_scale(dense_gene(data, "jam3b"))[global_index]
    return pd.DataFrame(
        {
            "h5_index": global_index,
            "obs_name": data.obs_names[global_index].astype(str),
            "x": coordinates[:, 0],
            "y": coordinates[:, 1],
            "cell_type": annotations,
            "is_somite": annotations == somite_label,
            "jam2a": jam2,
            "jam3b": jam3,
            "jam2a_positive": jam2 > 0,
            "jam3b_positive": jam3 > 0,
        }
    )


def build_somite_adjacency(
    spatial_cells: pd.DataFrame,
    *,
    cutoff: float,
) -> pd.DataFrame:
    somite = spatial_cells.loc[spatial_cells["is_somite"]].reset_index(drop=True)
    if len(somite) < 2:
        raise ValueError("Too few Somite cells")
    xy = somite[["x", "y"]].to_numpy(float)
    distances = squareform(pdist(xy))
    source_local, target_local = np.where(np.triu((distances <= cutoff) & (distances > 0), 1))
    source = somite.iloc[source_local]
    target = somite.iloc[target_local]
    compatible = (
        source["jam2a_positive"].to_numpy(bool)
        & target["jam3b_positive"].to_numpy(bool)
    ) | (
        source["jam3b_positive"].to_numpy(bool)
        & target["jam2a_positive"].to_numpy(bool)
    )
    result = pd.DataFrame(
        {
            "source_h5": source["h5_index"].to_numpy(int),
            "target_h5": target["h5_index"].to_numpy(int),
            "source_x": source["x"].to_numpy(float),
            "source_y": source["y"].to_numpy(float),
            "target_x": target["x"].to_numpy(float),
            "target_y": target["y"].to_numpy(float),
            "distance": distances[source_local, target_local],
            "jam_compatible": compatible,
        }
    )
    if result.duplicated(["source_h5", "target_h5"]).any():
        raise AssertionError("Undirected adjacency contains duplicate pairs")
    return result


def collapse_jam_model_edges(
    edges: pd.DataFrame,
    spatial_cells: pd.DataFrame,
    *,
    somite_label: str,
    min_seed_support: int,
    max_display_edges: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    local = edges.loc[
        edges["sender_type"].astype(str).eq(somite_label)
        & edges["receiver_type"].astype(str).eq(somite_label)
    ].copy()
    cells = spatial_cells.set_index("h5_index")
    source = local["_source_h5"].to_numpy(int)
    target = local["_target_h5"].to_numpy(int)
    if not set(source).issubset(cells.index) or not set(target).issubset(cells.index):
        raise ValueError("Somite model edges are not covered by the spatial table")
    j2_source = cells.loc[source, "jam2a"].to_numpy(float)
    j3_source = cells.loc[source, "jam3b"].to_numpy(float)
    j2_target = cells.loc[target, "jam2a"].to_numpy(float)
    j3_target = cells.loc[target, "jam3b"].to_numpy(float)
    forward = j2_source * j3_target
    reverse = j3_source * j2_target
    local["jam_lr_activity"] = np.maximum(forward, reverse)
    local["jam_orientation"] = np.where(
        forward >= reverse, "jam2a-compatible → jam3b-compatible", "jam3b-compatible → jam2a-compatible"
    )
    local = local.loc[local["jam_lr_activity"].gt(0)].copy()
    if local.empty:
        raise ValueError("No post-hoc Jam2a/Jam3b-compatible Somite model edges")
    n_seeds = int(edges["grouping_seed"].nunique())
    collapsed = (
        local.groupby(["_source_h5", "_target_h5"], observed=True, as_index=False)
        .agg(
            seed_support=("grouping_seed", "nunique"),
            seed_list=(
                "grouping_seed",
                lambda values: ";".join(str(value) for value in sorted(set(map(int, values)))),
            ),
            mean_attention=("attention_abs_mean", "mean"),
            mean_jam_lr_activity=("jam_lr_activity", "mean"),
            mean_spatial_distance=("spatial_distance", "mean"),
            jam_orientation=("jam_orientation", lambda values: ";".join(sorted(set(map(str, values))))),
        )
    )
    collapsed["n_total_grouping_seeds"] = n_seeds
    collapsed["seed_support_fraction"] = collapsed["seed_support"] / n_seeds
    collapsed["display_score"] = (
        collapsed["mean_attention"]
        * collapsed["mean_jam_lr_activity"]
        * collapsed["seed_support_fraction"]
    )
    for role in ("source", "target"):
        index = collapsed[f"_{role}_h5"].to_numpy(int)
        collapsed[[f"{role}_x", f"{role}_y"]] = cells.loc[index, ["x", "y"]].to_numpy(float)
        collapsed[f"{role}_jam2a"] = cells.loc[index, "jam2a"].to_numpy(float)
        collapsed[f"{role}_jam3b"] = cells.loc[index, "jam3b"].to_numpy(float)
    stable = collapsed.loc[collapsed["seed_support"].ge(int(min_seed_support))].sort_values(
        ["display_score", "seed_support", "_source_h5", "_target_h5"],
        ascending=[False, False, True, True],
        kind="stable",
    )
    if stable.empty:
        raise ValueError("No JAM-compatible edge passes the grouping-seed support rule")
    display = stable.head(int(max_display_edges)).copy().reset_index(drop=True)
    display["display_rank"] = np.arange(1, len(display) + 1)
    return stable.reset_index(drop=True), display


def resolve_type_pair_summary(
    attribution_dir: Path,
    case_root: Path,
    supplied: Path | None,
) -> Path:
    if supplied is not None:
        resolved = supplied.expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        return resolved
    candidates = (
        attribution_dir / "type_pair_summary.csv",
        attribution_dir / "tables" / "type_pair_summary.csv",
        case_root / "type_pair_summary.csv",
        case_root / "tables" / "type_pair_summary.csv",
        case_root / "tables" / "raw_attention_type_pair_summary.csv",
    )
    existing = [path.expanduser().resolve() for path in candidates if path.is_file()]
    if len(existing) != 1:
        raise FileNotFoundError(f"Expected one type_pair_summary.csv: {existing}")
    return existing[0]


def raw_attention_rank_table(
    path: Path,
    *,
    stage: float,
    stage_label: str,
    comparison_stage: float,
    comparison_stage_label: str,
    somite_label: str,
) -> pd.DataFrame:
    frame = pd.read_csv(path)
    score_column = "G_AB_attention_mean_mean"
    require_columns(
        frame,
        ["stage", "stage_label", "sender_type", "receiver_type", score_column],
        "raw attention type-pair summary",
    )
    rows = []
    for stage_value, label in ((stage, stage_label), (comparison_stage, comparison_stage_label)):
        local = frame.loc[
            np.isclose(pd.to_numeric(frame["stage"], errors="raise"), stage_value)
            & frame["stage_label"].astype(str).eq(label)
        ].copy()
        if local.empty:
            raise ValueError(f"No raw-attention contexts for {label}")
        if local.duplicated(["sender_type", "receiver_type"]).any():
            raise ValueError(f"Duplicate native raw-attention type pair for {label}")
        levels = sorted(
            set(local["sender_type"].astype(str)).union(
                set(local["receiver_type"].astype(str))
            )
        )
        expected_pairs = {(sender, receiver) for sender in levels for receiver in levels}
        observed_pairs = set(
            local[["sender_type", "receiver_type"]].astype(str).itertuples(
                index=False, name=None
            )
        )
        if observed_pairs != expected_pairs:
            raise ValueError(
                f"Native type_pair_summary is not the complete directed square for {label}: "
                f"observed={len(observed_pairs)}, expected={len(expected_pairs)}"
            )
        local["rank"] = pd.to_numeric(local[score_column], errors="raise").rank(
            method="min", ascending=False
        )
        selected = local.loc[
            local["sender_type"].astype(str).eq(somite_label)
            & local["receiver_type"].astype(str).eq(somite_label)
        ]
        if len(selected) != 1:
            raise ValueError(f"Expected one Somite→Somite raw-attention row for {label}")
        selected_row = selected.iloc[0]
        rows.append(
            {
                "method": "CytoBridge native raw attention",
                "display_label": f"CytoBridge native raw attention ({label})",
                "stage": float(stage_value),
                "stage_label": label,
                "rank": float(selected_row["rank"]),
                "n_contexts": int(len(local)),
                "score": float(selected_row[score_column]),
                "score_column": score_column,
                "rank_universe": "complete native directed cell-type-pair square",
                "sensitivity_analysis": False,
                "source": str(path.resolve()),
            }
        )
    return pd.DataFrame(rows)


def _stage_subset(
    frame: pd.DataFrame,
    *,
    stage: float,
    stage_label: str,
    label: str,
) -> pd.DataFrame:
    require_columns(frame, ["stage"], label)
    numeric_stage = pd.to_numeric(frame["stage"], errors="coerce")
    if numeric_stage.notna().all() and np.isclose(
        numeric_stage.to_numpy(float), float(stage), rtol=0.0, atol=1e-12
    ).any():
        selected = frame.loc[
            np.isclose(
                numeric_stage.to_numpy(float), float(stage), rtol=0.0, atol=1e-12
            )
        ].copy()
    elif frame["stage"].astype(str).eq(stage_label).any():
        selected = frame.loc[frame["stage"].astype(str).eq(stage_label)].copy()
    else:
        match = pd.Series(False, index=frame.index)
        if "stage_time" in frame:
            parsed = pd.to_numeric(
                pd.Series([stage_label]).str.extract(
                    r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*hpf\s*$",
                    expand=False,
                ),
                errors="coerce",
            ).iloc[0]
            if pd.notna(parsed):
                match = pd.Series(
                    np.isclose(
                        pd.to_numeric(frame["stage_time"], errors="raise").to_numpy(float),
                        float(parsed),
                        rtol=0.0,
                        atol=1e-12,
                    ),
                    index=frame.index,
                )
        selected = frame.loc[match].copy()
    if selected.empty:
        raise ValueError(f"{label} has no rows for stage={stage:g} / {stage_label}")
    return selected


def commot_distinct_type_pair_rank_table(
    root: Path,
    *,
    stage: float,
    stage_label: str,
    comparison_stage: float,
    comparison_stage_label: str,
    somite_label: str,
) -> tuple[pd.DataFrame, dict[str, Path]]:
    bundle = root.expanduser().resolve()
    if not bundle.is_dir():
        raise NotADirectoryError(
            "Formal COMMOT distinct-cell input must be a run directory with manifest.json"
        )
    score_path = _candidate_file(
        bundle,
        ("commot_type_pair_scores.csv.gz",),
        required=True,
    )
    assert score_path is not None
    manifest_path = _verify_manifest_artifacts(
        bundle,
        {"type_pair_scores": score_path},
        label="formal COMMOT distinct-cell run",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if str(manifest.get("method", "")).casefold() != "commot":
        raise ValueError("COMMOT manifest method must be COMMOT")
    design = manifest.get("design")
    if not isinstance(design, Mapping):
        raise ValueError("COMMOT manifest lacks design")
    grid_export = design.get("type_pair_grid_export")
    if not isinstance(grid_export, Mapping) or not _as_bool(
        grid_export.get("complete_directed_stage_type_square")
    ):
        raise ValueError("COMMOT type-pair table must be a complete directed square")
    semantics = manifest.get("score_semantics")
    if not isinstance(semantics, Mapping) or "abundance_controlled_distinct_cell_score" not in semantics:
        raise ValueError("COMMOT manifest lacks distinct-cell score semantics")

    frame = pd.read_csv(score_path)
    score_column = "abundance_controlled_distinct_cell_score"
    require_columns(
        frame,
        [
            "method",
            "stage",
            "stage_time",
            "sender_type",
            "receiver_type",
            score_column,
            "score_mean_possible_distinct_cell_pairs",
            "n_possible_distinct_cell_pairs",
            "interaction_id",
            "matrix_key",
        ],
        "COMMOT complete type-pair scores",
    )
    if set(frame["method"].astype(str).str.casefold()) != {"commot"}:
        raise ValueError("COMMOT type-pair table has a conflicting method column")
    rows: list[dict[str, object]] = []
    for stage_value, label in (
        (stage, stage_label),
        (comparison_stage, comparison_stage_label),
    ):
        local = _stage_subset(
            frame,
            stage=stage_value,
            stage_label=label,
            label="COMMOT complete type-pair scores",
        )
        if local.duplicated(["sender_type", "receiver_type"]).any():
            raise ValueError(f"Duplicate COMMOT type pair for {label}")
        if set(local["interaction_id"].astype(str)) != {"total"}:
            raise ValueError(f"COMMOT type-pair rows must be interaction_id=total at {label}")
        if not local["matrix_key"].astype(str).str.endswith("-total-total").all():
            raise ValueError(f"COMMOT type-pair rows must use the total-total matrix at {label}")
        levels = sorted(
            set(local["sender_type"].astype(str)).union(
                set(local["receiver_type"].astype(str))
            )
        )
        expected_pairs = {(sender, receiver) for sender in levels for receiver in levels}
        observed_pairs = set(
            local[["sender_type", "receiver_type"]].astype(str).itertuples(
                index=False, name=None
            )
        )
        if observed_pairs != expected_pairs:
            raise ValueError(
                f"COMMOT stage {label} is not a complete directed square: "
                f"observed={len(observed_pairs)}, expected={len(expected_pairs)}"
            )
        score = pd.to_numeric(local[score_column], errors="raise").to_numpy(float)
        canonical_score = pd.to_numeric(
            local["score_mean_possible_distinct_cell_pairs"], errors="raise"
        ).to_numpy(float)
        if not np.allclose(score, canonical_score, rtol=0.0, atol=1e-15):
            raise ValueError(
                f"COMMOT abundance-controlled distinct score disagrees with its canonical "
                f"distinct-cell mean at {label}"
            )
        denominator = pd.to_numeric(
            local["n_possible_distinct_cell_pairs"], errors="raise"
        ).to_numpy(float)
        if not np.isfinite(score).all() or (score < 0).any():
            raise ValueError(f"COMMOT distinct-cell scores must be finite/nonnegative at {label}")
        if not np.isfinite(denominator).all() or (denominator <= 0).any():
            raise ValueError(f"COMMOT distinct-cell denominators must be positive at {label}")
        local["rank"] = pd.Series(score, index=local.index).rank(
            method="min", ascending=False
        )
        selected = local.loc[
            local["sender_type"].astype(str).eq(somite_label)
            & local["receiver_type"].astype(str).eq(somite_label)
        ]
        if len(selected) != 1:
            raise ValueError(f"COMMOT lacks one {somite_label}→{somite_label} row for {label}")
        selected_row = selected.iloc[0]
        rows.append(
            {
                "method": "COMMOT distinct-cell",
                "display_label": f"COMMOT distinct-cell total ({label})",
                "stage": float(stage_value),
                "stage_label": label,
                "rank": float(selected_row["rank"]),
                "n_contexts": int(len(local)),
                "score": float(selected_row[score_column]),
                "score_column": score_column,
                "rank_universe": "complete COMMOT directed cell-type-pair square",
                "sensitivity_analysis": False,
                "source": str(score_path),
            }
        )
    return pd.DataFrame(rows), {
        "commot_type_pair_scores": score_path,
        "commot_manifest": manifest_path,
    }


def commot_reciprocal_jam_rank_table(
    root: Path,
    *,
    stage: float,
    stage_label: str,
    comparison_stage: float,
    comparison_stage_label: str,
    somite_label: str,
) -> tuple[pd.DataFrame, dict[str, Path]]:
    bundle = root.expanduser().resolve()
    if not bundle.is_dir():
        raise NotADirectoryError(bundle)
    paths = {
        "type_pair_scores": _candidate_file(
            bundle, ("commot_type_pair_scores.csv.gz",), required=True
        ),
        "lr_scores": _candidate_file(
            bundle, ("commot_lr_scores.csv.gz",), required=True
        ),
        "lr_axis_stage_availability": _candidate_file(
            bundle, ("commot_lr_axis_stage_availability.csv.gz",), required=True
        ),
    }
    resolved_paths = {name: path for name, path in paths.items() if path is not None}
    if len(resolved_paths) != 3:
        raise FileNotFoundError("Incomplete COMMOT reciprocal-JAM inputs")
    manifest_path = _verify_manifest_artifacts(
        bundle,
        resolved_paths,
        label="formal COMMOT reciprocal-JAM audit",
    )
    total = pd.read_csv(resolved_paths["type_pair_scores"])
    lr = pd.read_csv(resolved_paths["lr_scores"])
    availability = pd.read_csv(resolved_paths["lr_axis_stage_availability"])
    score_column = "abundance_controlled_distinct_cell_score"
    require_columns(
        lr,
        ["stage", "ligand", "receptor", "sender_type", "receiver_type", score_column],
        "COMMOT LR scores",
    )
    require_columns(
        availability,
        ["stage", "ligand", "receptor", "method_available"],
        "COMMOT LR availability",
    )
    require_columns(total, ["stage", "sender_type", "receiver_type"], "COMMOT total grid")
    rows: list[dict[str, object]] = []
    wanted_axes = set(JAM_AXES)
    for stage_value, label in (
        (stage, stage_label),
        (comparison_stage, comparison_stage_label),
    ):
        grid = _stage_subset(
            total,
            stage=stage_value,
            stage_label=label,
            label="COMMOT total grid",
        )[["sender_type", "receiver_type"]].drop_duplicates()
        axis_availability = _stage_subset(
            availability,
            stage=stage_value,
            stage_label=label,
            label="COMMOT LR availability",
        )
        axis_availability = axis_availability.loc[
            [
                (str(ligand), str(receptor)) in wanted_axes
                for ligand, receptor in zip(
                    axis_availability["ligand"], axis_availability["receptor"]
                )
            ]
        ].copy()
        if len(axis_availability) != 2 or axis_availability.duplicated(
            ["ligand", "receptor"]
        ).any():
            raise ValueError(f"COMMOT requires exactly two reciprocal JAM availability rows at {label}")
        if set(
            zip(
                axis_availability["ligand"].astype(str),
                axis_availability["receptor"].astype(str),
            )
        ) != wanted_axes or not axis_availability["method_available"].map(_as_bool).all():
            raise ValueError(f"Both reciprocal JAM axes must be explicitly available at {label}")
        local_lr = _stage_subset(
            lr,
            stage=stage_value,
            stage_label=label,
            label="COMMOT LR scores",
        )
        local_lr = local_lr.loc[
            [
                (str(ligand), str(receptor)) in wanted_axes
                for ligand, receptor in zip(local_lr["ligand"], local_lr["receptor"])
            ]
        ].copy()
        axis_columns = []
        for ligand, receptor in JAM_AXES:
            axis = local_lr.loc[
                local_lr["ligand"].astype(str).eq(ligand)
                & local_lr["receptor"].astype(str).eq(receptor)
            ].copy()
            axis[score_column] = pd.to_numeric(axis[score_column], errors="raise")
            collapsed_rows = []
            for (sender, receiver), group in axis.groupby(
                ["sender_type", "receiver_type"], observed=True, sort=False
            ):
                values = group[score_column].to_numpy(float)
                if not np.isfinite(values).all() or (values < 0).any():
                    raise ValueError(f"Invalid COMMOT reciprocal-JAM score at {label}")
                if not np.isclose(values, values[0], rtol=1e-10, atol=1e-15).all():
                    raise ValueError(
                        f"Duplicate COMMOT provenance rows disagree for {ligand}->{receptor}"
                    )
                collapsed_rows.append(
                    {
                        "sender_type": str(sender),
                        "receiver_type": str(receiver),
                        f"{ligand}_to_{receptor}": float(values[0]),
                    }
                )
            column = f"{ligand}_to_{receptor}"
            axis_columns.append(column)
            collapsed = pd.DataFrame(
                collapsed_rows,
                columns=["sender_type", "receiver_type", column],
            )
            grid = grid.merge(
                collapsed,
                on=["sender_type", "receiver_type"],
                how="left",
                validate="one_to_one",
            )
            grid[column] = grid[column].fillna(0.0)
        grid["reciprocal_jam_score"] = grid[axis_columns].max(axis=1)
        grid["rank"] = grid["reciprocal_jam_score"].rank(
            method="min", ascending=False
        )
        grid["tie_count"] = grid.groupby("reciprocal_jam_score")[
            "reciprocal_jam_score"
        ].transform("size")
        selected = grid.loc[
            grid["sender_type"].astype(str).eq(somite_label)
            & grid["receiver_type"].astype(str).eq(somite_label)
        ]
        if len(selected) != 1:
            raise ValueError(f"COMMOT reciprocal-JAM lacks one Somite row at {label}")
        row = selected.iloc[0]
        score = float(row["reciprocal_jam_score"])
        rows.append(
            {
                "method": "COMMOT reciprocal-JAM",
                "stage": float(stage_value),
                "stage_label": label,
                "sender_type": somite_label,
                "receiver_type": somite_label,
                "score": score,
                "rank": int(row["rank"]),
                "n_contexts": int(len(grid)),
                "tie_count": int(row["tie_count"]),
                "status": "positive" if score > 0 else "not detected",
                "zero_completion_rule": (
                    "missing sparse context rows become zero only because both stage x JAM "
                    "axes are explicitly available"
                ),
                "biochemical_polarity_assigned": False,
            }
        )
    return pd.DataFrame(rows), {
        "commot_lr_scores": resolved_paths["lr_scores"],
        "commot_lr_axis_stage_availability": resolved_paths[
            "lr_axis_stage_availability"
        ],
        "commot_reciprocal_jam_manifest": manifest_path,
    }


def _resolve_method_table(path: Path, method: str) -> Path:
    path = path.expanduser().resolve()
    if path.is_file():
        return path
    preferred = (
        "lr_scores.csv",
        "commot_lr_scores.csv",
        "distinct_cell_lr_scores.csv",
        "exact_circuit_scores.csv",
        "method_ranks.csv",
        "rank_table.csv",
    )
    candidates = []
    for name in preferred:
        candidates.extend((path / name, path / "tables" / name))
    existing = [candidate for candidate in candidates if candidate.is_file()]
    if len(existing) == 1:
        return existing[0]
    if len(existing) > 1:
        raise ValueError(f"Ambiguous {method} tables: {existing}")
    csvs = sorted(path.rglob("*.csv")) + sorted(path.rglob("*.csv.gz"))
    if len(csvs) == 1:
        return csvs[0]
    raise FileNotFoundError(f"Cannot resolve one {method} table under {path}")


def external_method_rank(
    path: Path | None,
    *,
    method: str,
    stage: float,
    stage_label: str,
    somite_label: str,
) -> tuple[pd.DataFrame, Path | None]:
    if path is None:
        return pd.DataFrame(), None
    table_path = _resolve_method_table(path, method)
    frame = pd.read_csv(table_path)
    is_cellagentchat = "cellagentchat" in method.casefold()
    if {"method", "rank", "n_contexts"}.issubset(frame.columns):
        local = frame.copy()
        if "stage" in local:
            local = local.loc[np.isclose(pd.to_numeric(local["stage"], errors="raise"), stage)]
        if "stage_label" in local:
            local = local.loc[local["stage_label"].astype(str).eq(stage_label)]
        if "sender_type" in local:
            local = local.loc[local["sender_type"].astype(str).eq(somite_label)]
        if "receiver_type" in local:
            local = local.loc[local["receiver_type"].astype(str).eq(somite_label)]
        if local.empty:
            raise ValueError(f"Ready {method} rank table lacks the requested JAM context")
        row = local.iloc[0]
        score_definition = str(
            row.get("score_column", row.get("score_definition", ""))
        ).casefold()
        if is_cellagentchat and "significant_score_sum" not in score_definition:
            raise ValueError(
                "CellAgentChat may enter this figure only as the explicitly labelled "
                "significant-score-sum sensitivity"
            )
        result = pd.DataFrame(
            [
                {
                    "method": method,
                    "display_label": (
                        f"CellAgentChat score-sum sensitivity ({stage_label})"
                        if is_cellagentchat
                        else str(row.get("display_label", method))
                    ),
                    "stage": stage,
                    "stage_label": stage_label,
                    "rank": float(row["rank"]),
                    "n_contexts": int(row["n_contexts"]),
                    "score": float(row.get("score", np.nan)),
                    "score_column": score_definition,
                    "rank_universe": str(
                        row.get("rank_universe", "method-supplied directed context universe")
                    ),
                    "sensitivity_analysis": bool(is_cellagentchat),
                    "source": str(table_path),
                }
            ]
        )
        return result, table_path

    require_columns(frame, ["stage", "sender_type", "receiver_type"], f"{method} output")
    if "ligand" not in frame or "receptor" not in frame:
        if "axis_id" not in frame:
            raise ValueError(f"{method} output needs ligand/receptor or axis_id")
        split = frame["axis_id"].astype(str).str.replace("→", "->", regex=False).str.split("->", n=1, expand=True)
        frame["ligand"], frame["receptor"] = split[0], split[1]
    score_candidates = (
        (
            "cellagentchat_significant_score_sum_mean",
            "cellagentchat_significant_score_sum",
        )
        if is_cellagentchat
        else (
            "abundance_controlled_distinct_cell_score",
            "score_mean_possible_distinct_cell_pairs",
            "abundance_controlled_score",
            "score",
            "mean_score",
        )
    )
    score_column = next((column for column in score_candidates if column in frame), None)
    if score_column is None:
        raise ValueError(f"No supported score column in {method} output")
    available = np.ones(len(frame), dtype=bool)
    for column in ("axis_available", "structurally_available", "available"):
        if column in frame:
            available &= frame[column].map(_as_bool).to_numpy(bool)
    local = frame.loc[
        available
        & np.isclose(pd.to_numeric(frame["stage"], errors="raise"), stage)
        & [
            (str(ligand), str(receptor)) in set(JAM_AXES)
            for ligand, receptor in zip(frame["ligand"], frame["receptor"])
        ]
    ].copy()
    if local.empty:
        raise ValueError(f"{method} has no available JAM axis at {stage_label}")
    local[score_column] = pd.to_numeric(local[score_column], errors="coerce")
    local = local.dropna(subset=[score_column])
    # Jam2a/Jam3b is heterophilic and not assigned a biochemical polarity here;
    # take the stronger reciprocal-axis score for each cell-type context.
    contexts = (
        local.groupby(["sender_type", "receiver_type"], observed=True, as_index=False)[score_column]
        .max()
        .rename(columns={score_column: "score"})
    )
    contexts["rank"] = contexts["score"].rank(method="min", ascending=False)
    selected = contexts.loc[
        contexts["sender_type"].astype(str).eq(somite_label)
        & contexts["receiver_type"].astype(str).eq(somite_label)
    ]
    if len(selected) != 1:
        raise ValueError(f"{method} lacks one Somite→Somite reciprocal-JAM context")
    row = selected.iloc[0]
    result = pd.DataFrame(
        [
            {
                "method": method,
                "display_label": f"{method} ({stage_label})",
                "stage": float(stage),
                "stage_label": stage_label,
                "rank": float(row["rank"]),
                "n_contexts": int(len(contexts)),
                "score": float(row["score"]),
                "score_column": score_column,
                "rank_universe": "available reciprocal-JAM cell-type contexts",
                "sensitivity_analysis": bool(is_cellagentchat),
                "source": str(table_path),
            }
        ]
    )
    return result, table_path


def load_provenance(path: Path) -> pd.DataFrame:
    path = path.expanduser().resolve()
    frame = pd.read_csv(path)
    require_columns(
        frame,
        ["ligand", "receptor", "evidence_scope", "claim_guardrail", "source_ids", "source_urls"],
        "JAM provenance",
    )
    pairs = set(zip(frame["ligand"].astype(str), frame["receptor"].astype(str)))
    if pairs != set(JAM_AXES):
        raise ValueError(f"Provenance must contain exactly both reciprocal JAM axes: {pairs}")
    return frame


def build_claim_ladder(
    case: pd.Series,
    marker_detection: pd.DataFrame,
    ranks: pd.DataFrame,
    provenance: pd.DataFrame,
    *,
    stage: float,
    somite_label: str,
) -> pd.DataFrame:
    stage18 = marker_detection.loc[
        np.isclose(marker_detection["stage"], stage)
        & marker_detection["cell_type"].astype(str).eq(somite_label)
    ].set_index("gene")
    if stage18.empty:
        # Custom labels/stages remain supported; fall back to the first stage.
        first_stage = marker_detection["stage"].min()
        stage18 = marker_detection.loc[np.isclose(marker_detection["stage"], first_stage)].set_index("gene")
    trained = int(case["trained_somite_attention_rank"])
    initialized = int(case["init_somite_attention_rank"])
    random_rank = int(case["random_somite_attention_rank"])
    n_neighbors = int(case["n_somite_neighbor_pairs"])
    n_compatible = int(case["n_jam_compatible_neighbor_pairs"])
    null_mean = float(case["neighbor_shuffle_null_mean"])
    fold = float(case["neighbor_fold_enrichment"])
    rows = [
        {
            "group": "supported",
            "status": True,
            "claim": "Published experiments validate the fusion program",
            "evidence": "Jam2a/Jam3b apposition and F-actin-focused myocyte fusion (Powell & Wright 2011; Luo et al. 2022)",
        },
        {
            "group": "supported",
            "status": True,
            "claim": "The atlas places JAM genes in the fusion-window Somite",
            "evidence": (
                f"18 hpf detection: jam2a {stage18.loc['jam2a', 'detected_fraction']:.0%}, "
                f"jam3b {stage18.loc['jam3b', 'detected_fraction']:.0%}, "
                f"myog {stage18.loc['myog', 'detected_fraction']:.0%}"
            ),
        },
        {
            "group": "supported",
            "status": True,
            "claim": "JAM-positive Somite neighbors are spatially enriched",
            "evidence": (
                f"{n_compatible:,}/{n_neighbors:,} neighbor pairs; null mean {null_mean:.1f}; "
                f"{fold:.2f}× (co-regional state may contribute)"
            ),
        },
        {
            "group": "not_supported",
            "status": False,
            "claim": "Attention is a biochemical communication strength",
            "evidence": "Raw attention is generic; JAM identity is applied post hoc",
        },
        {
            "group": "not_supported",
            "status": False,
            "claim": "The JAM pattern is training-specific or LR-causal",
            "evidence": (
                f"trained/init ranks {trained}/{initialized}; random {random_rank}; "
                "no JAM-specific intervention or trajectory rerun"
            ),
        },
    ]
    if provenance["claim_guardrail"].astype(str).str.len().eq(0).any():
        raise ValueError("JAM provenance claim guardrails must be non-empty")
    return pd.DataFrame(rows)


def validate_render_contract(
    spatial_cells: pd.DataFrame,
    adjacency: pd.DataFrame,
    display_edges: pd.DataFrame,
    ranks: pd.DataFrame,
    marker_detection: pd.DataFrame,
    claims: pd.DataFrame,
) -> None:
    require_columns(spatial_cells, ["h5_index", "x", "y", "is_somite", "jam2a", "jam3b"], "spatial cells")
    require_columns(adjacency, ["source_h5", "target_h5", "distance", "jam_compatible"], "adjacency")
    if adjacency["source_h5"].eq(adjacency["target_h5"]).any():
        raise ValueError("Adjacency must contain distinct cells only")
    if adjacency.duplicated(["source_h5", "target_h5"]).any():
        raise ValueError("Adjacency contains duplicate undirected pairs")
    require_columns(
        display_edges,
        ["_source_h5", "_target_h5", "seed_support", "display_score", "source_x", "target_x"],
        "display edges",
    )
    if display_edges["_source_h5"].eq(display_edges["_target_h5"]).any():
        raise ValueError("Display model edges must be distinct-cell edges")
    require_columns(ranks, ["display_label", "rank", "n_contexts"], "method ranks")
    if (ranks["rank"] < 1).any() or (ranks["rank"] > ranks["n_contexts"]).any():
        raise ValueError("Invalid method rank/N")
    require_columns(marker_detection, ["stage_label", "cell_type", "gene", "detected_fraction"], "markers")
    if not set(MARKER_GENES).issubset(set(marker_detection["gene"].astype(str))):
        raise ValueError("Marker detection lacks the complete JAM/myogenesis panel")
    require_columns(claims, ["group", "status", "claim", "evidence"], "claim ladder")
    if not claims.loc[claims["group"].eq("not_supported"), "status"].eq(False).all():
        raise ValueError("Not-supported claims must be explicit false rows")
    if not {"Attention is a biochemical communication strength", "The JAM pattern is training-specific or LR-causal"}.issubset(
        set(claims["claim"])
    ):
        raise ValueError("Claim ladder lacks required attention/causal limitations")


def _panel_title(ax, letter: str, title: str) -> None:
    ax.set_title(f"{letter}  {title}", loc="left", fontsize=14.5, fontweight="bold", pad=8)


def _spatial_limits(ax, coordinates: np.ndarray, pad: float = 0.05) -> None:
    coordinates = np.asarray(coordinates, dtype=float)
    low = np.nanmin(coordinates, axis=0)
    high = np.nanmax(coordinates, axis=0)
    span = np.maximum(high - low, 1e-6)
    ax.set_xlim(low[0] - pad * span[0], high[0] + pad * span[0])
    ax.set_ylim(low[1] - pad * span[1], high[1] + pad * span[1])
    ax.set_aspect("equal")
    ax.axis("off")


def _rank_color(label: str) -> str:
    if label.startswith("CytoBridge"):
        return JAM3_BLUE
    if "COMMOT" in label.upper():
        return COMMOT_ORANGE
    return CELLAGENT_PURPLE


def _edge_widths(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) == 0 or np.ptp(values) <= 0:
        return np.full(values.shape, 2.2)
    low, high = np.quantile(values, [0.1, 0.9])
    scaled = np.clip((values - low) / max(high - low, 1e-12), 0, 1)
    return 1.4 + 3.0 * scaled


def plot_main_figure(
    spatial_cells: pd.DataFrame,
    adjacency: pd.DataFrame,
    stable_edges: pd.DataFrame,
    display_edges: pd.DataFrame,
    ranks: pd.DataFrame,
    reciprocal_jam: pd.DataFrame | None,
    marker_detection: pd.DataFrame,
    claims: pd.DataFrame,
    case: pd.Series,
    provenance: pd.DataFrame,
    *,
    stage_label: str,
    somite_label: str,
    comparison_stage_label: str,
    comparison_cell_type: str,
    later_stage_label: str,
    later_cell_type: str,
    min_edge_seed_support: int,
    missing_external_methods: Sequence[str],
    output_png: Path,
    dpi: int,
) -> tuple[Path, Path]:
    validate_render_contract(spatial_cells, adjacency, display_edges, ranks, marker_detection, claims)
    style = {
        "font.family": "DejaVu Sans",
        "font.size": 12.5,
        "axes.labelsize": 12.5,
        "axes.titlesize": 14.5,
        "legend.fontsize": 9.5,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
    with plt.rc_context(style):
        figure = plt.figure(figsize=(15.6, 13.2))
        grid = figure.add_gridspec(
            3,
            3,
            height_ratios=(1.0, 1.0, 0.50),
            left=0.045,
            right=0.985,
            bottom=0.045,
            top=0.89,
            wspace=0.34,
            hspace=0.50,
        )
        axes = np.asarray(
            [
                [figure.add_subplot(grid[0, index]) for index in range(3)],
                [figure.add_subplot(grid[1, index]) for index in range(3)],
            ]
        )
        claim_ax = figure.add_subplot(grid[2, :])

        xy = spatial_cells[["x", "y"]].to_numpy(float)
        somite = spatial_cells["is_somite"].to_numpy(bool)
        somite_frame = spatial_cells.loc[somite].copy()
        somite_xy = somite_frame[["x", "y"]].to_numpy(float)

        # A: whole-embryo anatomy.
        ax = axes[0, 0]
        ax.scatter(xy[:, 0], xy[:, 1], s=5, c=LIGHT, alpha=0.55, linewidths=0, rasterized=True)
        ax.scatter(
            somite_xy[:, 0], somite_xy[:, 1], s=17, c=SOMITE_PURPLE, alpha=0.92,
            linewidths=0, rasterized=True, label=f"{somite_label} (n={somite.sum():,})",
        )
        _spatial_limits(ax, xy)
        _panel_title(ax, "A", f"{stage_label} embryo:\n{somite_label} compartment")
        ax.legend(loc="lower left", frameon=False, markerscale=1.5)

        # B: adjacency is explicitly not communication.
        ax = axes[0, 1]
        all_segments = adjacency[["source_x", "source_y", "target_x", "target_y"]].to_numpy(float).reshape(-1, 2, 2)
        compatible = adjacency["jam_compatible"].to_numpy(bool)
        ax.add_collection(
            LineCollection(all_segments, colors="#C9CED5", linewidths=0.35, alpha=0.20, rasterized=True)
        )
        if compatible.any():
            ax.add_collection(
                LineCollection(
                    all_segments[compatible], colors=COMPAT_TEAL, linewidths=1.15,
                    alpha=0.78, rasterized=True, zorder=6,
                )
            )
        neither = ~(
            somite_frame["jam2a_positive"].to_numpy(bool)
            | somite_frame["jam3b_positive"].to_numpy(bool)
        )
        both = (
            somite_frame["jam2a_positive"].to_numpy(bool)
            & somite_frame["jam3b_positive"].to_numpy(bool)
        )
        j2_only = somite_frame["jam2a_positive"].to_numpy(bool) & ~both
        j3_only = somite_frame["jam3b_positive"].to_numpy(bool) & ~both
        ax.scatter(somite_xy[neither, 0], somite_xy[neither, 1], s=10, c="#D7DBE0", linewidths=0, zorder=2)
        ax.scatter(somite_xy[j2_only, 0], somite_xy[j2_only, 1], s=26, c=JAM2_ORANGE, edgecolors="white", linewidths=0.4, zorder=4)
        ax.scatter(somite_xy[j3_only, 0], somite_xy[j3_only, 1], s=26, c=JAM3_BLUE, edgecolors="white", linewidths=0.4, zorder=4)
        ax.scatter(somite_xy[both, 0], somite_xy[both, 1], s=34, c=SOMITE_PURPLE, edgecolors="white", linewidths=0.5, zorder=5)
        _spatial_limits(ax, somite_xy, pad=0.07)
        _panel_title(ax, "B", f"{somite_label} zoom:\nJAM-positive neighboring cells")
        handles = [
            Line2D([0], [0], marker="o", color="none", markerfacecolor=JAM2_ORANGE, markersize=7, label="jam2a+"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=JAM3_BLUE, markersize=7, label="jam3b+"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=SOMITE_PURPLE, markersize=7, label="both+"),
            Line2D([0], [0], color=COMPAT_TEAL, lw=2, label="compatible adjacency"),
        ]
        ax.legend(handles=handles, loc="lower left", frameon=False, ncol=1)
        ax.text(
            0.02, 0.99,
            f"{int(case['n_jam_compatible_neighbor_pairs']):,} / "
            f"{int(case['n_somite_neighbor_pairs']):,} distinct neighbor pairs\n"
            f"vs label-shuffle mean {float(case['neighbor_shuffle_null_mean']):.1f} "
            f"({float(case['neighbor_fold_enrichment']):.2f}×)\n"
            f"cutoff={float(case['spatial_cutoff']):.4g}; adjacency, not communication",
            transform=ax.transAxes, va="top", fontsize=9.3, color=INK,
        )

        # C: ranks are read/calculated from tables; no external result is hard-coded.
        ax = axes[0, 2]
        rank_plot = ranks.copy().reset_index(drop=True)
        rank_plot["top_fraction_percent"] = 100.0 * rank_plot["rank"] / rank_plot["n_contexts"]
        rank_plot["compact_label"] = [
            (
                f"{row.stage_label} G_AB edge mean"
                if str(row.method) == "CytoBridge native raw attention"
                else f"{row.stage_label} COMMOT total"
                if str(row.method) == "COMMOT distinct-cell"
                else f"{row.stage_label} {row.method}"
            )
            for row in rank_plot.itertuples(index=False)
        ]
        # Reserve the bottom row for the control audit so it cannot conceal a
        # method point (especially the 12 hpf comparison when only two rows exist).
        y = np.arange(len(rank_plot), 0, -1) + 1
        for y_value, row in zip(y, rank_plot.itertuples(index=False)):
            color = _rank_color(str(row.display_label))
            ax.plot([0, row.top_fraction_percent], [y_value, y_value], color=color, linewidth=3, alpha=0.55)
            ax.scatter([row.top_fraction_percent], [y_value], s=82, color=color, edgecolor="white", linewidth=0.7, zorder=3)
            rank_text = int(row.rank) if float(row.rank).is_integer() else f"{row.rank:.1f}"
            ax.text(row.top_fraction_percent + 0.25, y_value, f"{rank_text}/{int(row.n_contexts)}", va="center", fontsize=10, color=color, fontweight="bold")
        ax.set_yticks(y)
        ax.set_yticklabels(rank_plot["compact_label"])
        ax.set_ylim(0.0, len(rank_plot) + 1.35)
        x_max = max(8.0, float(rank_plot["top_fraction_percent"].max()) + 2.0)
        ax.set_xlim(0, x_max)
        ax.set_xlabel("Rank fraction among cell-type contexts (%)\nlower = stronger")
        ax.grid(axis="x", color="#E5E8EB", linewidth=0.8)
        _panel_title(ax, "C", f"{somite_label}→{somite_label} context ranks\nwithin each stage")
        trained = int(case["trained_somite_attention_rank"])
        trained_n = int(case["trained_somite_attention_n_contexts"])
        initialized = int(case["init_somite_attention_rank"])
        initialized_n = int(case["init_somite_attention_n_contexts"])
        random_rank = int(case["random_somite_attention_rank"])
        random_n = int(case["random_somite_attention_n_contexts"])
        note_lines = [
            "Scores: native G_AB edge mean; COMMOT distinct-cell total.\n"
        ]
        if reciprocal_jam is not None:
            require_columns(
                reciprocal_jam,
                ["stage_label", "score", "rank", "n_contexts", "tie_count", "status"],
                "COMMOT reciprocal-JAM temporal audit",
            )
            temporal = reciprocal_jam.set_index("stage_label")
            if stage_label not in temporal.index or comparison_stage_label not in temporal.index:
                raise ValueError("Reciprocal-JAM audit lacks both displayed stages")
            early = temporal.loc[comparison_stage_label]
            fusion = temporal.loc[stage_label]
            early_text = (
                f"not detected (score 0; {int(early['tie_count'])}-way zero tie)"
                if float(early["score"]) == 0
                else f"positive, {int(early['rank'])}/{int(early['n_contexts'])}"
            )
            fusion_text = (
                f"not detected (score 0; {int(fusion['tie_count'])}-way zero tie)"
                if float(fusion["score"]) == 0
                else f"positive, {int(fusion['rank'])}/{int(fusion['n_contexts'])}"
            )
            note_lines.append(
                f"Reciprocal-JAM COMMOT: {comparison_stage_label} {early_text}; "
                f"{stage_label} {fusion_text}.\n"
            )
        note_lines.extend(
            [
                "CytoBridge JAM ordering matches LR-only; no attention-specific gain.\n",
                f"Single-seed control: trained {trained}/{trained_n}; initialization "
                f"{initialized}/{initialized_n}; random {random_rank}/{random_n}; "
                "training specificity not demonstrated.",
            ]
        )
        note = "".join(note_lines)
        if missing_external_methods:
            note += "\nPending/not supplied: " + ", ".join(missing_external_methods) + "."
        ax.text(
            0.02, 0.02, note, transform=ax.transAxes, va="bottom", fontsize=7.9,
            color=FAIL_RED, bbox={"boxstyle": "round,pad=0.35", "facecolor": "#FFF3F1", "edgecolor": "#E7B0AA"},
        )

        # D: generic model edges, JAM identity assigned post hoc.
        ax = axes[1, 0]
        ax.scatter(somite_xy[:, 0], somite_xy[:, 1], s=8, c="#CFD4D9", alpha=0.55, linewidths=0, rasterized=True)
        support_colors = {3: "#2A9D8F", 4: "#4878B8", 5: "#7B2CBF"}
        ordered = display_edges.sort_values("display_score")
        widths = _edge_widths(ordered["display_score"].to_numpy(float))
        for row, width in zip(ordered.itertuples(index=False), widths):
            arrow = FancyArrowPatch(
                (row.source_x, row.source_y), (row.target_x, row.target_y),
                arrowstyle="-|>", mutation_scale=12 + 1.4 * int(row.seed_support),
                linewidth=float(width), color=support_colors.get(int(row.seed_support), INK),
                alpha=0.78, shrinkA=3, shrinkB=3, zorder=4,
            )
            ax.add_patch(arrow)
        _spatial_limits(ax, somite_xy, pad=0.07)
        _panel_title(ax, "D", "Post-hoc JAM-compatible\nmodel edges")
        ax.text(
            0.02, 0.99,
            f"Top {len(display_edges)} of {len(stable_edges)} stable directed edges shown\n"
            f"threshold ≥{int(min_edge_seed_support)}/"
            f"{int(display_edges['n_total_grouping_seeds'].iloc[0])} technical seeds\n"
            "generic model arrows; JAM filter applied post hoc",
            transform=ax.transAxes, va="top", fontsize=8.8, color=INK,
        )
        support_values = sorted(display_edges["seed_support"].astype(int).unique())
        ax.legend(
            handles=[
                Line2D([0], [0], color=support_colors.get(value, INK), lw=3, label=f"{value}/5 seeds")
                for value in support_values
            ],
            loc="lower right",
            bbox_to_anchor=(1.0, 0.24),
            frameon=False,
            title="Technical recurrence",
            title_fontsize=9.5,
        )
        ax.text(
            0.02, 0.02,
            "JAM arrow direction is not biochemical polarity\n"
            "No LR deletion or trajectory rerun",
            transform=ax.transAxes, fontsize=8.5, color=FAIL_RED, fontweight="bold",
        )

        # E: cross-sectional marker transition.
        ax = axes[1, 1]
        stage_order = [
            f"{comparison_stage_label} {comparison_cell_type}",
            f"{stage_label} {somite_label}",
            f"{later_stage_label} {later_cell_type}",
        ]
        stage_lookup = {}
        for row in marker_detection.itertuples(index=False):
            if (
                str(row.stage_label) == comparison_stage_label
                and str(row.cell_type) == comparison_cell_type
            ):
                label = stage_order[0]
            elif str(row.stage_label) == stage_label and str(row.cell_type) == somite_label:
                label = stage_order[1]
            elif str(row.stage_label) == later_stage_label and str(row.cell_type) == later_cell_type:
                label = stage_order[2]
            else:
                continue
            stage_lookup[(label, str(row.gene))] = float(row.detected_fraction)
        matrix = np.asarray(
            [[stage_lookup.get((label, gene), np.nan) for gene in MARKER_GENES] for label in stage_order]
        )
        masked = np.ma.masked_invalid(matrix)
        cmap = plt.get_cmap("YlGnBu").copy()
        cmap.set_bad("#EEEEEE")
        ax.imshow(masked, vmin=0, vmax=1, cmap=cmap, aspect="auto")
        ax.set_xticks(np.arange(len(MARKER_GENES)))
        ax.set_xticklabels(MARKER_GENES, rotation=40, ha="right")
        stage_counts = []
        for label in stage_order:
            local = marker_detection.loc[
                marker_detection.apply(
                    lambda row: f"{row['stage_label']} {row['cell_type']}" == label,
                    axis=1,
                )
            ]
            stage_counts.append(int(local["n_cells"].iloc[0]))
        ax.set_yticks(np.arange(len(stage_order)))
        ax.set_yticklabels(
            [f"{label}\n(n={count:,})" for label, count in zip(stage_order, stage_counts)]
        )
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if not np.isfinite(matrix[i, j]):
                    text = "n/a"
                elif matrix[i, j] == 0:
                    text = "0%"
                elif matrix[i, j] < 0.02:
                    text = f"{100 * matrix[i, j]:.1f}%"
                else:
                    text = f"{100 * matrix[i, j]:.0f}%"
                color = "white" if np.isfinite(matrix[i, j]) and matrix[i, j] > 0.55 else INK
                ax.text(j, i, text, ha="center", va="center", fontsize=9.2, color=color, fontweight="bold")
        _panel_title(
            ax,
            "E",
            "Marker-state transition\n(cross-sectional—not lineage)",
        )
        ax.text(
            0.0, -0.37,
            f"18 hpf jam3b–myog co-detection: Fisher OR={float(case['jam3b_myog_fisher_or']):.2f}, "
            f"P={float(case['jam3b_myog_fisher_p']):.3g}; association, not regulation.\n"
            "Different stage samples—not the same cells and not a lineage trajectory.",
            transform=ax.transAxes, fontsize=8.8, color=FAIL_RED, va="top",
        )

        # F: published experimental mechanism, intentionally separate from model panels.
        ax = axes[1, 2]
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        _panel_title(ax, "F", "Published experimental mechanism\n—not inferred by CytoBridge")
        left_cell = Ellipse((0.29, 0.57), 0.47, 0.57, facecolor="#F9D7A8", edgecolor=JAM2_ORANGE, linewidth=2)
        right_cell = Ellipse((0.71, 0.57), 0.47, 0.57, facecolor="#CFE0F4", edgecolor=JAM3_BLUE, linewidth=2)
        ax.add_patch(left_cell)
        ax.add_patch(right_cell)
        ax.plot([0.48, 0.48], [0.37, 0.77], color=JAM2_ORANGE, lw=3)
        ax.plot([0.52, 0.52], [0.37, 0.77], color=JAM3_BLUE, lw=3)
        for y_value in (0.48, 0.62, 0.70):
            ax.scatter([0.48], [y_value], s=45, c=JAM2_ORANGE, edgecolors="white", zorder=4)
            ax.scatter([0.52], [y_value], s=45, c=JAM3_BLUE, edgecolors="white", zorder=4)
            ax.plot([0.485, 0.515], [y_value, y_value], color=INK, lw=1.2)
        rng = np.random.default_rng(8)
        actin_x = 0.50 + rng.normal(0, 0.025, 24)
        actin_y = 0.39 + rng.normal(0, 0.025, 24)
        ax.scatter(actin_x, actin_y, s=18, c=FAIL_RED, alpha=0.85, linewidths=0, zorder=5)
        ax.text(0.22, 0.57, "myocyte", ha="center", va="center", fontweight="bold")
        ax.text(0.78, 0.57, "myocyte", ha="center", va="center", fontweight="bold")
        ax.text(0.50, 0.82, "apposed membranes", ha="center", fontsize=10, fontweight="bold")
        ax.text(0.50, 0.73, "Jam2a ↔ Jam3b", ha="center", fontsize=10, color=INK)
        ax.annotate(
            "localized F-actin focus", xy=(0.50, 0.39), xytext=(0.70, 0.25),
            arrowprops={"arrowstyle": "->", "color": FAIL_RED, "lw": 1.5},
            fontsize=9.5, color=FAIL_RED, ha="center",
        )
        ax.add_patch(FancyArrowPatch((0.50, 0.26), (0.50, 0.12), arrowstyle="-|>", mutation_scale=16, color=PASS_GREEN, lw=2))
        ax.text(0.50, 0.07, "membrane fusion", ha="center", color=PASS_GREEN, fontweight="bold")
        ax.text(
            0.5, 0.94, "Powell & Wright 2011 • Luo et al. 2022",
            ha="center", fontsize=10.2, color=PASS_GREEN, fontweight="bold",
        )
        ax.text(
            0.5,
            0.01,
            "Published perturbation/live-imaging evidence; sources in provenance table",
            ha="center",
            fontsize=8.3,
            color=MUTED,
        )

        # Full-width claim ladder.
        claim_ax.set_xlim(0, 1)
        claim_ax.set_ylim(0, 1)
        claim_ax.axis("off")
        claim_ax.text(0.0, 0.98, "Claim ladder: what is supported—and what is not", fontsize=15, fontweight="bold", va="top")
        claim_ax.add_patch(
            FancyBboxPatch((0.0, 0.04), 0.53, 0.80, boxstyle="round,pad=0.012", facecolor="#EFF8F4", edgecolor="#A8D5C5")
        )
        claim_ax.add_patch(
            FancyBboxPatch((0.55, 0.04), 0.45, 0.80, boxstyle="round,pad=0.012", facecolor="#FFF2F0", edgecolor="#E6B1AB")
        )
        claim_ax.text(0.018, 0.78, "SUPPORTED", color=PASS_GREEN, fontweight="bold", fontsize=11.5)
        claim_ax.text(0.568, 0.78, "NOT SUPPORTED", color=FAIL_RED, fontweight="bold", fontsize=11.5)
        supported_claims = claims.loc[claims["group"].eq("supported")]
        for y_value, row in zip((0.58, 0.35, 0.12), supported_claims.itertuples(index=False)):
            claim_ax.text(0.018, y_value + 0.06, "✓", color=PASS_GREEN, fontsize=18, fontweight="bold", va="center")
            claim_ax.text(0.052, y_value + 0.065, row.claim, fontsize=10.0, fontweight="bold", va="center")
            claim_ax.text(0.052, y_value - 0.005, row.evidence, fontsize=8.4, color=MUTED, va="center")
        negative_claims = claims.loc[claims["group"].eq("not_supported")]
        for y_value, row in zip((0.54, 0.20), negative_claims.itertuples(index=False)):
            claim_ax.text(0.568, y_value + 0.06, "×", color=FAIL_RED, fontsize=20, fontweight="bold", va="center")
            claim_ax.text(0.606, y_value + 0.065, row.claim, fontsize=10.0, fontweight="bold", va="center")
            claim_ax.text(0.606, y_value - 0.015, row.evidence, fontsize=8.5, color=MUTED, va="center")

        figure.suptitle(
            f"{stage_label} {somite_label} Jam2a–Jam3b myocyte-fusion case",
            fontsize=22,
            fontweight="bold",
            y=0.985,
        )
        figure.text(
            0.5,
            0.945,
            "Observed spatial program • generic model organization • published mechanism kept separate",
            ha="center",
            fontsize=12.5,
            color=MUTED,
        )
        output_png.parent.mkdir(parents=True, exist_ok=True)
        output_pdf = output_png.with_suffix(".pdf")
        figure.savefig(output_png, dpi=int(dpi), bbox_inches="tight", facecolor="white")
        figure.savefig(output_pdf, bbox_inches="tight", facecolor="white")
        plt.close(figure)
    return output_png, output_pdf


def plot_spatial_evidence_figure(
    spatial_cells: pd.DataFrame,
    adjacency: pd.DataFrame,
    spatial_null: pd.DataFrame,
    case: pd.Series,
    *,
    stage_label: str,
    somite_label: str,
    output_png: Path,
    dpi: int,
) -> tuple[Path, Path]:
    require_columns(
        spatial_null,
        ["orientation_compatible_pair_count", "at_least_observed"],
        "spatial null iterations",
    )
    null_values = pd.to_numeric(
        spatial_null["orientation_compatible_pair_count"], errors="raise"
    ).to_numpy(float)
    observed = int(case["n_jam_compatible_neighbor_pairs"])
    null_mean = float(case["neighbor_shuffle_null_mean"])
    fold = float(case["neighbor_fold_enrichment"])
    tail = float(case["neighbor_monte_carlo_tail_fraction"])
    if not np.isclose(np.mean(null_values), null_mean, rtol=0.0, atol=1e-12):
        raise ValueError("Standalone spatial figure null mean disagrees with case summary")

    style = {
        "font.family": "DejaVu Sans",
        "font.size": 12.0,
        "axes.titlesize": 14.0,
        "axes.labelsize": 11.5,
        "legend.fontsize": 9.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
    with plt.rc_context(style):
        figure, axes = plt.subplots(
            1,
            3,
            figsize=(16.0, 4.9),
            gridspec_kw={"width_ratios": (1.02, 1.02, 1.16), "wspace": 0.40},
        )
        figure.subplots_adjust(top=0.76, bottom=0.16)
        xy = spatial_cells[["x", "y"]].to_numpy(float)
        somite_frame = spatial_cells.loc[spatial_cells["is_somite"]].copy()
        somite_xy = somite_frame[["x", "y"]].to_numpy(float)

        ax = axes[0]
        ax.scatter(xy[:, 0], xy[:, 1], s=5, c=LIGHT, alpha=0.55, linewidths=0, rasterized=True)
        ax.scatter(
            somite_xy[:, 0],
            somite_xy[:, 1],
            s=18,
            c=SOMITE_PURPLE,
            alpha=0.92,
            linewidths=0,
            rasterized=True,
        )
        _spatial_limits(ax, xy)
        _panel_title(ax, "A", f"Whole embryo\n{stage_label} {somite_label} location")
        ax.legend(
            handles=[
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="none",
                    markerfacecolor=SOMITE_PURPLE,
                    markersize=7,
                    label=f"{somite_label} (n={len(somite_frame):,})",
                )
            ],
            loc="lower left",
            frameon=False,
        )

        ax = axes[1]
        compatible = adjacency["jam_compatible"].to_numpy(bool)
        compatible_segments = adjacency.loc[
            compatible,
            ["source_x", "source_y", "target_x", "target_y"],
        ].to_numpy(float).reshape(-1, 2, 2)
        ax.add_collection(
            LineCollection(
                compatible_segments,
                colors=COMPAT_TEAL,
                linewidths=1.25,
                alpha=0.86,
                rasterized=True,
                zorder=6,
            )
        )
        both = (
            somite_frame["jam2a_positive"].to_numpy(bool)
            & somite_frame["jam3b_positive"].to_numpy(bool)
        )
        j2_only = somite_frame["jam2a_positive"].to_numpy(bool) & ~both
        j3_only = somite_frame["jam3b_positive"].to_numpy(bool) & ~both
        ax.scatter(
            somite_xy[j2_only, 0],
            somite_xy[j2_only, 1],
            s=28,
            c=JAM2_ORANGE,
            edgecolors="white",
            linewidths=0.55,
            zorder=3,
        )
        ax.scatter(
            somite_xy[j3_only, 0],
            somite_xy[j3_only, 1],
            s=28,
            c=JAM3_BLUE,
            edgecolors="white",
            linewidths=0.55,
            zorder=3,
        )
        ax.scatter(
            somite_xy[both, 0],
            somite_xy[both, 1],
            s=34,
            c=SOMITE_PURPLE,
            edgecolors="white",
            linewidths=0.65,
            zorder=4,
        )
        _spatial_limits(ax, somite_xy, pad=0.07)
        _panel_title(ax, "B", "Somite close-up\nJAM-positive adjacency")
        ax.text(
            0.02,
            0.98,
            f"{observed:,} compatible adjacencies\n"
            f"among {int(case['n_somite_neighbor_pairs']):,} distinct neighbor pairs",
            transform=ax.transAxes,
            va="top",
            fontsize=9.8,
            color=INK,
        )
        ax.legend(
            handles=[
                Line2D([0], [0], marker="o", color="none", markerfacecolor=JAM2_ORANGE, markersize=7, label="jam2a+"),
                Line2D([0], [0], marker="o", color="none", markerfacecolor=JAM3_BLUE, markersize=7, label="jam3b+"),
                Line2D([0], [0], marker="o", color="none", markerfacecolor=SOMITE_PURPLE, markersize=7, label="both+"),
                Line2D([0], [0], color=COMPAT_TEAL, lw=2, label="compatible adjacency"),
            ],
            loc="lower left",
            frameon=False,
            ncol=2,
        )

        ax = axes[2]
        ax.hist(
            null_values,
            bins=38,
            color="#AEB8C2",
            edgecolor="white",
            linewidth=0.45,
            alpha=0.95,
        )
        ax.axvline(null_mean, color=MUTED, linestyle="--", linewidth=2.2)
        ax.axvline(observed, color=COMPAT_TEAL, linewidth=3.0)
        ymax = ax.get_ylim()[1]
        ax.text(
            null_mean,
            ymax * 0.92,
            f" label-shuffle mean\n {null_mean:.1f}",
            color=MUTED,
            va="top",
            ha="left",
            fontsize=10.0,
        )
        ax.text(
            observed,
            ymax * 0.92,
            f" observed\n {observed:,}",
            color=COMPAT_TEAL,
            va="top",
            ha="left",
            fontsize=10.0,
            fontweight="bold",
        )
        ax.set_xlabel("JAM-compatible neighbor pairs after shuffling gene labels")
        ax.set_ylabel("Number of shuffled datasets")
        _panel_title(ax, "C", "Label-shuffle null\nObserved versus expected")
        ax.text(
            0.98,
            0.04,
            f"Observed / null mean = {fold:.2f}×\n"
            f"Monte Carlo upper tail P = {tail:.4g}\n"
            f"{int(case['neighbor_n_permutations']):,} within-Somite shuffles",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=10.2,
            color=COMPAT_TEAL,
            fontweight="bold",
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "#EFF8F4",
                "edgecolor": "#A8D5C5",
            },
        )

        figure.suptitle(
            "Spatial evidence for the Jam2a–Jam3b fusion-window program",
            fontsize=19,
            fontweight="bold",
            y=0.97,
        )
        figure.text(
            0.5,
            -0.015,
            "Adjacency means within the frozen spatial-neighbor cutoff; it is not measured direct contact or biochemical signaling.",
            ha="center",
            fontsize=10.5,
            color=FAIL_RED,
            fontweight="bold",
        )
        output_png.parent.mkdir(parents=True, exist_ok=True)
        output_pdf = output_png.with_suffix(".pdf")
        figure.savefig(output_png, dpi=int(dpi), bbox_inches="tight", facecolor="white")
        figure.savefig(output_pdf, bbox_inches="tight", facecolor="white")
        plt.close(figure)
    return output_png, output_pdf


def write_tables(
    root: Path,
    *,
    spatial_cells: pd.DataFrame,
    adjacency: pd.DataFrame,
    stable_edges: pd.DataFrame,
    display_edges: pd.DataFrame,
    ranks: pd.DataFrame,
    marker_detection: pd.DataFrame,
    claims: pd.DataFrame,
    spatial_null: pd.DataFrame | None = None,
    reciprocal_jam: pd.DataFrame | None = None,
) -> dict[str, Path]:
    root.mkdir(parents=True, exist_ok=False)
    paths = {
        "spatial_cells": root / "panel_ab_spatial_cells.csv.gz",
        "somite_adjacency": root / "panel_b_somite_adjacency.csv.gz",
        "stable_model_edges": root / "panel_d_stable_jam_model_edges.csv.gz",
        "display_model_edges": root / "panel_d_display_jam_model_edges.csv",
        "method_ranks": root / "panel_c_method_ranks.csv",
        "marker_detection": root / "panel_e_marker_detection.csv",
        "claim_ladder": root / "claim_ladder.csv",
    }
    spatial_cells.to_csv(paths["spatial_cells"], index=False)
    adjacency.to_csv(paths["somite_adjacency"], index=False)
    stable_edges.to_csv(paths["stable_model_edges"], index=False)
    display_edges.to_csv(paths["display_model_edges"], index=False)
    ranks.to_csv(paths["method_ranks"], index=False)
    marker_detection.to_csv(paths["marker_detection"], index=False)
    claims.to_csv(paths["claim_ladder"], index=False)
    if spatial_null is not None:
        paths["spatial_null_iterations"] = root / "spatial_null_iterations.csv.gz"
        spatial_null.to_csv(
            paths["spatial_null_iterations"], index=False, compression="gzip"
        )
    if reciprocal_jam is not None:
        paths["reciprocal_jam_temporal_audit"] = (
            root / "panel_c_commot_reciprocal_jam_temporal_audit.csv"
        )
        reciprocal_jam.to_csv(paths["reciprocal_jam_temporal_audit"], index=False)
    return paths


def main() -> None:
    args = parser().parse_args()
    output = args.output_dir.expanduser().resolve()
    prepare_output(output, args.overwrite)
    figures = output / "figures"
    figures.mkdir()

    case, case_sources, case_schema = load_case_statistics(
        args.jam_case_output,
        trained_init_random_control_output=args.trained_init_random_control_output,
        stage=args.stage,
        stage_label=args.stage_label,
        somite_label=args.somite_label,
    )
    spatial_null = None
    spatial_null_source = case_sources.get("formal_case_somite_spatial_null")
    if spatial_null_source is not None:
        spatial_null = pd.read_csv(spatial_null_source)
    cutoff = float(case["spatial_cutoff"])
    if not np.isfinite(cutoff) or cutoff <= 0:
        raise ValueError("spatial_cutoff must be positive")
    provenance = load_provenance(args.provenance_csv)
    data = ad.read_h5ad(args.h5ad.expanduser().resolve())
    marker_detection, marker_source = load_marker_detection(
        args.jam_case_output,
        data,
        time_key=args.time_key,
        annotation_key=args.annotation_key,
        stage=args.stage,
        stage_label=args.stage_label,
        cell_type=args.somite_label,
        comparison_stage=args.comparison_stage,
        comparison_stage_label=args.comparison_stage_label,
        comparison_cell_type=args.somite_label,
        later_stage=args.later_stage,
        later_stage_label=args.later_stage_label,
        later_cell_type=args.later_cell_type,
    )
    observed_path = resolve_observed_cells(args.attribution_dir, args.observed_cells)
    mapping = observed_mapping(
        observed_path,
        data,
        time_key=args.time_key,
        annotation_key=args.annotation_key,
    )
    edges, edge_paths = load_stage_edges(
        args.attribution_dir,
        data,
        mapping,
        stage=args.stage,
        stage_label=args.stage_label,
        time_key=args.time_key,
        annotation_key=args.annotation_key,
    )
    spatial_cells = spatial_cell_table(
        data,
        stage=args.stage,
        time_key=args.time_key,
        annotation_key=args.annotation_key,
        spatial_key=args.spatial_key,
        somite_label=args.somite_label,
    )
    adjacency = build_somite_adjacency(spatial_cells, cutoff=cutoff)
    if len(adjacency) != int(case["n_somite_neighbor_pairs"]):
        raise ValueError(
            f"Somite neighbor count disagrees with case table: {len(adjacency)} vs "
            f"{int(case['n_somite_neighbor_pairs'])}"
        )
    compatible_count = int(adjacency["jam_compatible"].sum())
    if compatible_count != int(case["n_jam_compatible_neighbor_pairs"]):
        raise ValueError(
            "JAM-compatible neighbor count disagrees with case table: "
            f"{compatible_count} vs {int(case['n_jam_compatible_neighbor_pairs'])}"
        )
    stable_edges, display_edges = collapse_jam_model_edges(
        edges,
        spatial_cells,
        somite_label=args.somite_label,
        min_seed_support=args.min_edge_seed_support,
        max_display_edges=args.max_display_edges,
    )

    type_pair_path = resolve_type_pair_summary(
        args.attribution_dir,
        args.jam_case_output,
        args.type_pair_summary,
    )
    rank_frames = [
        raw_attention_rank_table(
            type_pair_path,
            stage=args.stage,
            stage_label=args.stage_label,
            comparison_stage=args.comparison_stage,
            comparison_stage_label=args.comparison_stage_label,
            somite_label=args.somite_label,
        )
    ]
    missing_external = []
    method_sources: dict[str, Path] = {}
    reciprocal_jam = None
    if args.commot_distinct_cell_output is None:
        missing_external.append("COMMOT distinct-cell")
    else:
        commot_ranks, commot_sources = commot_distinct_type_pair_rank_table(
            args.commot_distinct_cell_output,
            stage=args.stage,
            stage_label=args.stage_label,
            comparison_stage=args.comparison_stage,
            comparison_stage_label=args.comparison_stage_label,
            somite_label=args.somite_label,
        )
        rank_frames.append(commot_ranks)
        method_sources.update(commot_sources)
        reciprocal_jam, reciprocal_sources = commot_reciprocal_jam_rank_table(
            args.commot_distinct_cell_output,
            stage=args.stage,
            stage_label=args.stage_label,
            comparison_stage=args.comparison_stage,
            comparison_stage_label=args.comparison_stage_label,
            somite_label=args.somite_label,
        )
        method_sources.update(reciprocal_sources)
    if args.cellagentchat_output is not None:
        result, source = external_method_rank(
            args.cellagentchat_output,
            method="CellAgentChat score-sum sensitivity",
            stage=args.stage,
            stage_label=args.stage_label,
            somite_label=args.somite_label,
        )
        if result.empty or source is None:
            raise ValueError("Supplied CellAgentChat sensitivity yielded no rank")
        rank_frames.append(result)
        method_sources["cellagentchat_score_sum_sensitivity"] = source
    ranks = pd.concat(rank_frames, ignore_index=True)
    claims = build_claim_ladder(
        case,
        marker_detection,
        ranks,
        provenance,
        stage=args.stage,
        somite_label=args.somite_label,
    )
    validate_render_contract(spatial_cells, adjacency, display_edges, ranks, marker_detection, claims)

    table_paths = write_tables(
        output / "figure_data",
        spatial_cells=spatial_cells,
        adjacency=adjacency,
        stable_edges=stable_edges,
        display_edges=display_edges,
        ranks=ranks,
        marker_detection=marker_detection,
        claims=claims,
        spatial_null=spatial_null,
        reciprocal_jam=reciprocal_jam,
    )
    main_png, main_pdf = plot_main_figure(
        spatial_cells,
        adjacency,
        stable_edges,
        display_edges,
        ranks,
        reciprocal_jam,
        marker_detection,
        claims,
        case,
        provenance,
        stage_label=args.stage_label,
        somite_label=args.somite_label,
        comparison_stage_label=args.comparison_stage_label,
        comparison_cell_type=args.somite_label,
        later_stage_label=args.later_stage_label,
        later_cell_type=args.later_cell_type,
        min_edge_seed_support=args.min_edge_seed_support,
        missing_external_methods=missing_external,
        output_png=figures / "jam_myocyte_biology_main.png",
        dpi=args.dpi,
    )
    spatial_outputs: dict[str, Path] = {}
    if spatial_null is not None:
        spatial_png, spatial_pdf = plot_spatial_evidence_figure(
            spatial_cells,
            adjacency,
            spatial_null,
            case,
            stage_label=args.stage_label,
            somite_label=args.somite_label,
            output_png=figures / "jam_myocyte_spatial_evidence.png",
            dpi=args.dpi,
        )
        spatial_outputs = {
            "spatial_evidence_png": spatial_png,
            "spatial_evidence_pdf": spatial_pdf,
        }

    source_paths = {
        "h5ad": args.h5ad.expanduser().resolve(),
        "observed_cells": observed_path,
        "provenance": args.provenance_csv.expanduser().resolve(),
        "cytobridge_native_type_pair_summary": type_pair_path,
        **case_sources,
        **{f"edge_seed_{index}": path for index, path in enumerate(edge_paths, start=1)},
        **{f"method_{name}": path for name, path in method_sources.items()},
    }
    if marker_source is not None:
        source_paths["marker_detection"] = marker_source
    output_paths = {
        "main_png": main_png,
        "main_pdf": main_pdf,
        **spatial_outputs,
        **table_paths,
    }
    manifest = {
        "schema_version": 2,
        "analysis": "zebrafish_18hpf_somite_jam_myocyte_biology_figure",
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "case": {
            "input_schema": case_schema,
            "stage": float(args.stage),
            "stage_label": args.stage_label,
            "cell_type": args.somite_label,
            "axes": [list(pair) for pair in JAM_AXES],
            "later_stage": float(args.later_stage),
            "later_stage_label": args.later_stage_label,
            "later_cell_type": args.later_cell_type,
        },
        "edge_display": {
            "minimum_grouping_seed_support": int(args.min_edge_seed_support),
            "maximum_display_edges": int(args.max_display_edges),
            "n_stable_edges": int(len(stable_edges)),
            "n_display_edges": int(len(display_edges)),
            "ranking": "mean raw attention x reciprocal JAM expression compatibility x technical-seed support fraction",
        },
        "method_coverage": {
            "displayed": ranks["method"].astype(str).tolist(),
            "not_supplied": missing_external,
            "cellagentchat_optional_score_sum_sensitivity_supplied": bool(
                args.cellagentchat_output is not None
            ),
            "external_ranks_hard_coded": False,
        },
        "panel_c_rank_contract": {
            "cytobridge_method": "CytoBridge native edge-mean raw attention",
            "cytobridge_source_column": "G_AB_attention_mean_mean",
            "cytobridge_not_the_specialized_case_density_rank": True,
            "specialized_case_density_rank_over_n_not_plotted": (
                None
                if "specialized_full_density_rank_not_used_in_panel_c" not in case
                else (
                    f"{int(case['specialized_full_density_rank_not_used_in_panel_c'])}/"
                    f"{int(case['specialized_full_density_n_not_used_in_panel_c'])}"
                )
            ),
            "commot_method": "COMMOT total distinct-cell abundance-controlled type-pair score",
            "commot_source_column": "abundance_controlled_distinct_cell_score",
            "rank_rule": "within-stage descending competition/min rank on complete directed cell-type square",
            "raw_cross_method_scores_directly_comparable": False,
            "rank_over_n_only": True,
            "reciprocal_jam_commot_temporal_disclosure": bool(
                reciprocal_jam is not None
            ),
            "reciprocal_jam_zero_is_reported_as_not_detected_not_as_a_top_rank": True,
            "cytobridge_jam_order_matches_lr_only": bool(
                case.get("jam_rank_order_matches_lr_only", False)
            ),
            "attention_specific_jam_rank_gain_supported": False,
        },
        "guardrails": {
            "observational_no_perturbation": True,
            "spatial_adjacency_is_communication": False,
            "attention_is_biochemical_communication_strength": False,
            "jam_identity_is_native_to_model_edge": False,
            "directed_model_edge_is_jam_biochemical_polarity": False,
            "grouping_seeds_are_biological_replicates": False,
            "label_shuffle_is_confirmatory_p_value": False,
            "cross_sectional_stage_comparison_is_lineage": False,
            "training_specificity_supported": False,
            "lr_specific_causality_supported": False,
            "published_experiment_is_separated_from_atlas_and_model": True,
        },
        "inputs": {name: file_record(path) for name, path in source_paths.items()},
        "outputs": {name: file_record(path) for name, path in output_paths.items()},
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "complete",
                "figure": str(main_png),
                "manifest": str(manifest_path),
                "missing_external_methods": missing_external,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
