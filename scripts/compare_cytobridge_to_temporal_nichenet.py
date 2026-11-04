#!/usr/bin/env python3
"""Compare CytoBridge interaction results with temporal NicheNet.

The script compares NicheNet ligand activity and sender support with the
matching CytoBridge sender-to-receiver results.  When a PCA file and temporal
input directory are supplied, it also compares the two methods in gene space.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import shutil
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import hypergeom, kendalltau, rankdata, spearmanr


SCRIPT_SCHEMA_VERSION = 1
EARLY_TIME_RE = re.compile(
    r"^(?P<early>m?\d+(?:p\d+)?)_to_(?P<late>m?\d+(?:p\d+)?)$"
)
LEARNED_REQUIRED = {
    "manifest": "manifest.json",
    "heatmap": "sender_receiver_heatmap_table.csv",
    "drift_seed": "drift_by_training_seed.csv",
    "drift_summary": "sender_receiver_drift_summary.csv",
    "lr": "lr_pair_scores.csv",
    "pathway": "pathway_scores.csv",
}
NICHENET_REQUIRED = {
    "manifest": "run_manifest.json",
    "activities": "nichenet_ligand_activities.csv",
    "links": "nichenet_top_ligand_target_links.csv",
    "support": "nichenet_custom_sender_support_scores.csv",
    "components": "nichenet_custom_sender_support_components.csv",
    "candidates": "nichenet_candidate_lr_edges.csv",
    "coverage": "nichenet_context_coverage.csv",
    "prior_coverage": "nichenet_prior_coverage.csv",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--learned-dir", required=True, type=Path)
    parser.add_argument("--nichenet-default-dir", required=True, type=Path)
    parser.add_argument("--nichenet-matched-dir", type=Path)
    parser.add_argument(
        "--nichenet-official-overlap-lr",
        type=Path,
        help=(
            "Optional ligand/receptor table containing matched LR rows that are "
            "present in the official NicheNet LR network. Required for the "
            "official-overlap Q sensitivity."
        ),
    )
    parser.add_argument(
        "--temporal-input-dir",
        type=Path,
        help="Directory from prepare_temporal_nichenet_inputs.py; required for DE agreement.",
    )
    parser.add_argument("--pca-artifacts", type=Path)
    parser.add_argument("--pca-manifest", type=Path)
    parser.add_argument("--expected-pca-sha256", help=argparse.SUPPRESS)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--top-k", default="5,10,20,30")
    parser.add_argument("--permutations", type=int, default=2000)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--random-seed", type=int, default=1729)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-figures", action="store_true")
    return parser


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
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


def _parse_top_k(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(sorted({int(item.strip()) for item in value.split(",") if item.strip()}))
    except ValueError as error:
        raise argparse.ArgumentTypeError("--top-k must be comma-separated positive integers.") from error
    if not parsed or parsed[0] <= 0:
        raise argparse.ArgumentTypeError("--top-k must contain positive integers.")
    return parsed


def _token_to_float(value: str) -> float:
    negative = value.startswith("m")
    if negative:
        value = value[1:]
    result = float(value.replace("p", "."))
    return -result if negative else result


def transition_early_time(value: str) -> float:
    match = EARLY_TIME_RE.fullmatch(str(value).strip())
    if match is None:
        raise ValueError(
            f"Transition {value!r} is not parseable; expected tokens such as '2_to_4'."
        )
    early = _token_to_float(match.group("early"))
    late = _token_to_float(match.group("late"))
    if not (math.isfinite(early) and math.isfinite(late) and late > early):
        raise ValueError(f"Invalid ordered transition: {value!r}.")
    return early


def _require_columns(df: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}.")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except Exception as error:
        raise ValueError(f"Could not parse JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def _resolve_files(directory: Path, required: Mapping[str, str], label: str) -> dict[str, Path]:
    directory = directory.expanduser().resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"{label} directory does not exist: {directory}")
    paths = {key: directory / filename for key, filename in required.items()}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{label} is incomplete; missing: {missing}")
    return paths


def _finite_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _safe_spearman(x: Sequence[float], y: Sequence[float]) -> float:
    x_array = np.asarray(x, dtype=float)
    y_array = np.asarray(y, dtype=float)
    valid = np.isfinite(x_array) & np.isfinite(y_array)
    x_array, y_array = x_array[valid], y_array[valid]
    if len(x_array) < 3 or np.ptp(x_array) == 0 or np.ptp(y_array) == 0:
        return math.nan
    return float(spearmanr(x_array, y_array).statistic)


def _safe_kendall(x: Sequence[float], y: Sequence[float]) -> float:
    x_array = np.asarray(x, dtype=float)
    y_array = np.asarray(y, dtype=float)
    valid = np.isfinite(x_array) & np.isfinite(y_array)
    x_array, y_array = x_array[valid], y_array[valid]
    if len(x_array) < 3 or np.ptp(x_array) == 0 or np.ptp(y_array) == 0:
        return math.nan
    return float(kendalltau(x_array, y_array, variant="b").statistic)


def _rank_auc(scores: Sequence[float], labels: Sequence[bool]) -> float:
    scores_array = np.asarray(scores, dtype=float)
    labels_array = np.asarray(labels, dtype=bool)
    valid = np.isfinite(scores_array)
    scores_array, labels_array = scores_array[valid], labels_array[valid]
    n_positive = int(labels_array.sum())
    n_negative = int((~labels_array).sum())
    if n_positive == 0 or n_negative == 0:
        return math.nan
    ranks = rankdata(scores_array, method="average")
    rank_sum = float(ranks[labels_array].sum())
    return (rank_sum - n_positive * (n_positive + 1) / 2) / (n_positive * n_negative)


def _as_boolean(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.astype(str).str.lower().isin(["true", "1", "t", "yes", "y"])


def build_target_selection_audit(receiver_de: pd.DataFrame) -> pd.DataFrame:
    """Audit whether each receiver program is FDR-derived or fallback-ranked."""

    _require_columns(
        receiver_de,
        ["transition", "receiver", "gene", "effect", "selected_target", "target_selection_mode"],
        "receiver_de_genes.csv",
    )
    frame = receiver_de.copy()
    frame["selected_target"] = _as_boolean(frame["selected_target"])
    frame["effect"] = _finite_numeric(frame["effect"])
    if "q_value" in frame:
        frame["q_value"] = _finite_numeric(frame["q_value"])
        frame["fdr_positive"] = (
            (frame["q_value"] <= 0.05) & (frame["effect"] > 0)
        )
    else:
        frame["fdr_positive"] = False
    rows: list[dict[str, Any]] = []
    for (transition, receiver), group in frame.groupby(
        ["transition", "receiver"], observed=True, sort=True
    ):
        selected_mask = group["selected_target"]
        modes = sorted(
            set(
                group.loc[selected_mask, "target_selection_mode"]
                .dropna()
                .astype(str)
            )
            - {"not_selected"}
        )
        if len(modes) != 1:
            raise ValueError(
                f"Receiver DE context {transition}/{receiver} has non-unique target_selection_mode: {modes}."
            )
        mode = modes[0]
        if mode == "fdr_positive":
            evidence_class = "fdr_positive_only"
        elif mode == "fallback_ranked_positive":
            evidence_class = "fallback_ranked_positive"
        else:
            evidence_class = f"other:{mode}"
        selected = selected_mask
        rows.append(
            {
                "transition": str(transition),
                "receiver": str(receiver),
                "target_selection_mode": mode,
                "program_evidence_class": evidence_class,
                "n_de_genes": int(len(group)),
                "n_selected_targets": int(selected.sum()),
                "n_fdr_positive_genes": int(group["fdr_positive"].sum()),
                "n_selected_targets_not_fdr_positive": int(
                    (selected & ~group["fdr_positive"]).sum()
                ),
                "uses_fallback": bool(mode == "fallback_ranked_positive"),
                "program_label": (
                    "FDR-positive temporal response program"
                    if mode == "fdr_positive"
                    else "ranked positive-effect fallback program (not an FDR-significant gene set)"
                ),
            }
        )
    return pd.DataFrame(rows)


def _bootstrap_mean_ci(
    values: Sequence[float], *, replicates: int, rng: np.random.Generator
) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return math.nan, math.nan
    if len(array) == 1 or replicates <= 0:
        return float(array[0]), float(array[0])
    indices = rng.integers(0, len(array), size=(replicates, len(array)))
    means = array[indices].mean(axis=1)
    return tuple(float(value) for value in np.quantile(means, [0.025, 0.975]))


def _prepare_output(path: Path, overwrite: bool) -> Path:
    path = path.expanduser().resolve()
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory exists; pass --overwrite: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True)
    (path / "figures").mkdir()
    return path


def load_learned(directory: Path) -> tuple[dict[str, Any], dict[str, pd.DataFrame], dict[str, Path]]:
    paths = _resolve_files(directory, LEARNED_REQUIRED, "CytoBridge learned-interaction")
    manifest = _read_json(paths["manifest"])
    if manifest.get("method") != "exact_one_layer_state_gnn_sender_decomposition_with_posthoc_lr_annotation":
        raise ValueError("Unexpected learned-interaction method; refusing to reinterpret artifacts.")
    if not bool(manifest.get("exact_reconstruction", {}).get("passed", False)):
        raise ValueError("Formal CytoBridge exact-message reconstruction did not pass.")
    tables = {key: pd.read_csv(path) for key, path in paths.items() if key != "manifest"}
    _require_columns(
        tables["heatmap"],
        [
            "time", "sender_type", "receiver_type", "G_AB_abs_mean_per_edge_mean",
            "D_AB_mean", "Q_AB_total", "n_directed_cell_pairs",
        ],
        "sender_receiver_heatmap_table.csv",
    )
    _require_columns(
        tables["drift_seed"],
        ["training_seed", "time", "sender_type", "receiver_type", "D_AB",
         "G_AB_abs_mean_per_edge", "drift_pc_1"],
        "drift_by_training_seed.csv",
    )
    _require_columns(
        tables["drift_summary"],
        ["time", "sender_type", "receiver_type", "drift_pc_1_mean", "D_AB_mean"],
        "sender_receiver_drift_summary.csv",
    )
    _require_columns(
        tables["lr"],
        ["time", "sender_type", "receiver_type", "ligand", "receptor", "pathway",
         "Q_AB_lr_pair", "S_AB_lr_pair"],
        "lr_pair_scores.csv",
    )
    _require_columns(
        tables["pathway"],
        ["time", "sender_type", "receiver_type", "pathway", "Q_AB_pathway",
         "S_AB_pathway"],
        "pathway_scores.csv",
    )
    return manifest, tables, paths


def load_nichenet_run(
    directory: Path, expected_mode: str
) -> tuple[dict[str, Any], dict[str, pd.DataFrame], dict[str, Path]]:
    paths = _resolve_files(directory, NICHENET_REQUIRED, f"NicheNet {expected_mode}")
    manifest = _read_json(paths["manifest"])
    mode = str(
        manifest.get(
            "prior_mode",
            manifest.get("method", {}).get(
                "prior_mode", manifest.get("parameters", {}).get("prior_mode", "")
            ),
        )
    ).lower()
    if mode and mode != expected_mode:
        raise ValueError(
            f"NicheNet directory {directory} reports prior_mode={mode!r}, expected {expected_mode!r}."
        )
    tables = {key: pd.read_csv(path) for key, path in paths.items() if key != "manifest"}
    schemas = {
        "activities": ["prior_mode", "transition", "receiver", "ligand", "aupr_corrected", "pearson"],
        "links": ["prior_mode", "transition", "receiver", "ligand", "target", "regulatory_potential"],
        "support": ["prior_mode", "transition", "sender", "receiver", "sender_support_score",
                    "sender_support_fraction", "is_native_nichenet_edge_strength"],
        "components": ["prior_mode", "transition", "sender", "receiver", "ligand",
                       "positive_ligand_activity", "sender_expression_share",
                       "sender_support_component"],
        "candidates": ["prior_mode", "transition", "sender", "receiver", "ligand", "receptor",
                       "sender_expression_signal", "receiver_receptor_support"],
        "coverage": ["prior_mode", "transition", "receiver", "status"],
        "prior_coverage": ["prior_mode", "n_active_lr_edges"],
    }
    for key, columns in schemas.items():
        _require_columns(tables[key], columns, paths[key].name)
        if "prior_mode" in tables[key] and len(tables[key]):
            observed = {str(item).lower() for item in tables[key]["prior_mode"].dropna().unique()}
            if observed and observed != {expected_mode}:
                raise ValueError(f"{paths[key]} contains unexpected prior modes: {observed}.")
    if len(tables["support"]):
        native = tables["support"]["is_native_nichenet_edge_strength"].astype(str).str.lower()
        if native.isin(["true", "1", "t", "yes"]).any():
            raise ValueError("NicheNet custom sender support is incorrectly labeled as native.")
    return manifest, tables, paths


def audit_pca_artifacts(
    path: Path,
    *,
    expected_sha256: str | None,
    expected_pc_count: int,
    pca_manifest_path: Path | None = None,
    learned_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Read and check the PCA linear map used for message projection."""

    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"PCA artifact does not exist: {path}")
    observed_hash = _sha256_file(path)
    if expected_sha256 and observed_hash.lower() != expected_sha256.strip().lower():
        raise ValueError(
            f"PCA artifact SHA-256 mismatch: expected {expected_sha256}, observed {observed_hash}."
        )
    # The frozen producer stores gene_names as an object string array.  Loading
    # pickle is therefore necessary; only use a provenance-checked local artifact.
    with np.load(path, allow_pickle=True) as archive:
        required = {"gene_names", "highly_variable", "pca_loadings"}
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(f"PCA artifact is missing arrays: {missing}.")
        genes = np.asarray(archive["gene_names"]).astype(str)
        highly_variable = np.asarray(archive["highly_variable"], dtype=bool)
        loadings = np.asarray(archive["pca_loadings"], dtype=np.float64)
    if genes.ndim != 1 or highly_variable.shape != genes.shape:
        raise ValueError("PCA gene_names/highly_variable arrays are not aligned one-dimensional arrays.")
    if len(set(genes.tolist())) != len(genes):
        raise ValueError("PCA artifact gene names are not unique.")
    if loadings.shape != (len(genes), expected_pc_count):
        raise ValueError(
            f"PCA loadings shape {loadings.shape} is incompatible with "
            f"{len(genes)} genes and {expected_pc_count} PCs."
        )
    if not np.isfinite(loadings).all() or not highly_variable.any():
        raise ValueError("PCA loadings are non-finite or no highly variable genes are marked.")
    non_hvg_max_abs = float(np.abs(loadings[~highly_variable]).max(initial=0.0))
    if non_hvg_max_abs > 1e-7:
        raise ValueError(
            "Non-HVG PCA loading rows are not zero; subspace gene interpretation is ambiguous."
        )

    pca_manifest: dict[str, Any] | None = None
    manifest_checks: dict[str, Any] = {}
    if pca_manifest_path is not None:
        pca_manifest_path = pca_manifest_path.expanduser().resolve()
        pca_manifest = _read_json(pca_manifest_path)
        declared = str(pca_manifest.get("artifacts_sha256", "")).lower()
        if declared and declared != observed_hash.lower():
            raise ValueError("PCA manifest artifact checksum does not match PCA NPZ.")
        if learned_manifest is not None:
            learned_inputs = learned_manifest.get("inputs", {})
            formal_latent = str(learned_inputs.get("latent_h5ad", {}).get("sha256", "")).lower()
            formal_expression = str(learned_inputs.get("expression_h5ad", {}).get("sha256", "")).lower()
            pca_latent = str(pca_manifest.get("output_sha256", "")).lower()
            pca_expression = str(pca_manifest.get("input_sha256", "")).lower()
            manifest_checks = {
                "formal_latent_matches_pca_manifest": bool(formal_latent and formal_latent == pca_latent),
                "formal_expression_matches_pca_manifest": bool(
                    formal_expression and formal_expression == pca_expression
                ),
            }
            if not all(manifest_checks.values()):
                raise ValueError(
                    "PCA manifest does not provenance-match the formal learned interaction inputs."
                )
    return {
        "path": str(path),
        "sha256": observed_hash,
        "genes": genes,
        "highly_variable": highly_variable,
        "loadings": loadings,
        "n_genes": int(len(genes)),
        "n_hvg": int(highly_variable.sum()),
        "n_pcs": int(expected_pc_count),
        "non_hvg_max_abs_loading": non_hvg_max_abs,
        "pca_manifest_path": str(pca_manifest_path) if pca_manifest_path else None,
        "pca_manifest_sha256": _sha256_file(pca_manifest_path) if pca_manifest_path else None,
        "manifest_checks": manifest_checks,
        "interpretation": (
            "Exact linear projection of the exported signed message vector into the retained "
            "50-PC log1p-normalized HVG subspace; not recovery of PCA-discarded components and "
            "not a causal ligand-target decomposition."
        ),
    }


def project_pc_drift(pc_values: np.ndarray, pca_loadings: np.ndarray) -> np.ndarray:
    pc_values = np.asarray(pc_values, dtype=np.float64)
    pca_loadings = np.asarray(pca_loadings, dtype=np.float64)
    if pc_values.ndim != 2 or pca_loadings.ndim != 2:
        raise ValueError("PC values and loadings must be two-dimensional.")
    if pc_values.shape[1] != pca_loadings.shape[1]:
        raise ValueError("PC vector dimension does not match PCA loading dimension.")
    if not np.isfinite(pc_values).all() or not np.isfinite(pca_loadings).all():
        raise ValueError("PC values and loadings must be finite.")
    return pc_values @ pca_loadings.T


def attach_transition_time(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["early_time"] = [transition_early_time(value) for value in result["transition"]]
    return result


def build_sender_support_join(
    support: pd.DataFrame, learned_heatmap: pd.DataFrame
) -> pd.DataFrame:
    support = attach_transition_time(support)
    support = support.rename(columns={"sender": "sender_type", "receiver": "receiver_type"})
    learned = learned_heatmap.copy()
    keep = [
        "time", "sender_type", "receiver_type", "G_AB_abs_mean_per_edge_mean",
        "D_AB_mean", "Q_AB_total", "connected_edge_count_mean", "n_directed_cell_pairs",
    ]
    available = [column for column in keep if column in learned.columns]
    learned = learned[available].copy()
    joined = support.merge(
        learned,
        left_on=["early_time", "sender_type", "receiver_type"],
        right_on=["time", "sender_type", "receiver_type"],
        how="left",
        validate="many_to_one",
    )
    missing = joined["D_AB_mean"].isna()
    if missing.any():
        examples = joined.loc[missing, ["transition", "sender_type", "receiver_type"]].head(5)
        raise ValueError(
            "NicheNet sender labels/times do not fully match CytoBridge artifacts; examples: "
            + examples.to_dict(orient="records").__repr__()
        )
    joined["comparison_semantics"] = (
        "custom NicheNet sender support (not native NicheNet CCC edge strength) versus "
        "CytoBridge sender-to-receiver diagnostics within the same receiver/transition"
    )
    return joined


def aggregate_singleton_prior_q(
    learned_lr: pd.DataFrame,
    official_overlap_lr: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Aggregate Q over supplied singleton and audited official-overlap subsets.

    The supplied singleton table and the official NicheNet LR network are not the
    same universe.  Keeping their Q sums in separate columns prevents the former
    from being mislabeled as an exact official-network control.
    """

    required = [
        "time", "sender_type", "receiver_type", "database_row", "ligand", "receptor",
        "Q_AB_lr_pair",
    ]
    _require_columns(learned_lr, required, "lr_pair_scores.csv")
    metadata = learned_lr[["database_row", "ligand", "receptor"]].drop_duplicates()
    singleton_rows = metadata[
        ~metadata["ligand"].astype(str).str.contains("_", regex=False)
        & ~metadata["receptor"].astype(str).str.contains("_", regex=False)
    ]
    singleton_ids = set(singleton_rows["database_row"])
    supplied = learned_lr[learned_lr["database_row"].isin(singleton_ids)]
    aggregated = (
        supplied.groupby(
            ["time", "sender_type", "receiver_type"], observed=True, as_index=False
        )["Q_AB_lr_pair"]
        .sum()
        .rename(columns={"Q_AB_lr_pair": "Q_AB_supplied_singleton_prior"})
    )
    official_overlap_rows: int | None = None
    if official_overlap_lr is not None:
        _require_columns(official_overlap_lr, ["ligand", "receptor"], "official overlap LR table")
        overlap = official_overlap_lr[["ligand", "receptor"]].copy()
        overlap["ligand"] = overlap["ligand"].astype(str).str.strip()
        overlap["receptor"] = overlap["receptor"].astype(str).str.strip()
        overlap = overlap.drop_duplicates()
        official_overlap_rows = int(len(overlap))
        supplied_pairs = singleton_rows[["ligand", "receptor"]].drop_duplicates()
        membership = overlap.merge(
            supplied_pairs,
            on=["ligand", "receptor"],
            how="left",
            indicator=True,
            validate="one_to_one",
        )
        if (membership["_merge"] != "both").any():
            examples = membership.loc[
                membership["_merge"] != "both", ["ligand", "receptor"]
            ].head(5)
            raise ValueError(
                "Official-overlap LR table contains rows absent from the supplied singleton "
                f"CytoBridge prior; examples: {examples.to_dict(orient='records')!r}"
            )
        overlap_selected = supplied.merge(
            overlap,
            on=["ligand", "receptor"],
            how="inner",
            validate="many_to_one",
        )
        official_q = (
            overlap_selected.groupby(
                ["time", "sender_type", "receiver_type"], observed=True, as_index=False
            )["Q_AB_lr_pair"]
            .sum()
            .rename(columns={"Q_AB_lr_pair": "Q_AB_official_overlap_singleton_prior"})
        )
        aggregated = aggregated.merge(
            official_q,
            on=["time", "sender_type", "receiver_type"],
            how="left",
            validate="one_to_one",
        )
        if aggregated["Q_AB_official_overlap_singleton_prior"].isna().any():
            raise ValueError("Could not aggregate official-overlap Q for every CytoBridge context.")
    else:
        aggregated["Q_AB_official_overlap_singleton_prior"] = math.nan
    audit = {
        "full_model_prior_rows": int(metadata["database_row"].nunique()),
        "supplied_singleton_prior_rows": int(len(singleton_ids)),
        "official_overlap_singleton_prior_rows": official_overlap_rows,
        "excluded_complex_rows": int(metadata["database_row"].nunique() - len(singleton_ids)),
        "singleton_rule": "ligand and receptor labels contain no '_' complex delimiter",
        "q_columns": {
            "supplied_singleton_prior": "Q_AB_supplied_singleton_prior",
            "official_overlap_singleton_prior": "Q_AB_official_overlap_singleton_prior",
        },
        "supplied_singleton_is_exact_official_nichenet_prior": False,
        "official_overlap_status": (
            "audited_table_loaded" if official_overlap_lr is not None else "not_evaluated_missing_table"
        ),
    }
    return aggregated, audit


def aggregate_context_candidate_prior_q(
    candidates: pd.DataFrame,
    learned_lr: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sum CytoBridge Q on the exact LR candidates exported for each NicheNet sender context.

    For a matched run this is the effective, expression-filtered candidate subset of
    the supplied LR table.  For a default run it is only the intersection between
    official NicheNet context candidates and the CytoBridge model prior, and the
    exported coverage columns make that incompleteness explicit.
    """

    _require_columns(
        candidates,
        ["prior_mode", "transition", "sender", "receiver", "ligand", "receptor"],
        "nichenet_candidate_lr_edges.csv",
    )
    _require_columns(
        learned_lr,
        ["time", "sender_type", "receiver_type", "ligand", "receptor", "Q_AB_lr_pair"],
        "lr_pair_scores.csv",
    )
    candidate = attach_transition_time(candidates).rename(
        columns={"sender": "sender_type", "receiver": "receiver_type"}
    )
    key = [
        "prior_mode", "transition", "early_time", "sender_type", "receiver_type",
        "ligand", "receptor",
    ]
    candidate = candidate[key].drop_duplicates().copy()
    learned = learned_lr[
        ["time", "sender_type", "receiver_type", "ligand", "receptor", "Q_AB_lr_pair"]
    ].copy()
    duplicate = learned.duplicated(
        ["time", "sender_type", "receiver_type", "ligand", "receptor"], keep=False
    )
    if duplicate.any():
        raise ValueError(
            "CytoBridge LR table has duplicate exact ligand/receptor rows within a sender context."
        )
    joined = candidate.merge(
        learned,
        left_on=["early_time", "sender_type", "receiver_type", "ligand", "receptor"],
        right_on=["time", "sender_type", "receiver_type", "ligand", "receptor"],
        how="left",
        validate="many_to_one",
    )
    joined["candidate_in_cytobridge_model_prior"] = joined["Q_AB_lr_pair"].notna()
    joined["Q_AB_lr_pair"] = _finite_numeric(joined["Q_AB_lr_pair"]).fillna(0.0)
    group_columns = [
        "prior_mode", "transition", "early_time", "sender_type", "receiver_type"
    ]
    summary = (
        joined.groupby(group_columns, observed=True, as_index=False)
        .agg(
            n_nichenet_context_candidate_lr=("ligand", "size"),
            n_context_candidates_in_cytobridge_prior=(
                "candidate_in_cytobridge_model_prior", "sum"
            ),
            Q_AB_nichenet_context_candidate_intersection=("Q_AB_lr_pair", "sum"),
        )
        .rename(columns={"early_time": "time"})
    )
    summary["fraction_context_candidates_in_cytobridge_prior"] = (
        summary["n_context_candidates_in_cytobridge_prior"]
        / summary["n_nichenet_context_candidate_lr"]
    )
    audit_rows: list[dict[str, Any]] = []
    for mode, frame in joined.groupby("prior_mode", observed=True, sort=True):
        unique_candidates = frame[["ligand", "receptor"]].drop_duplicates()
        unique_intersection = frame.loc[
            frame["candidate_in_cytobridge_model_prior"], ["ligand", "receptor"]
        ].drop_duplicates()
        audit_rows.append(
            {
                "prior_mode": mode,
                "n_context_candidate_rows": int(len(frame)),
                "n_context_candidate_rows_in_cytobridge_prior": int(
                    frame["candidate_in_cytobridge_model_prior"].sum()
                ),
                "fraction_context_candidate_rows_in_cytobridge_prior": float(
                    frame["candidate_in_cytobridge_model_prior"].mean()
                ),
                "n_unique_context_candidate_lr": int(len(unique_candidates)),
                "n_unique_context_candidate_lr_in_cytobridge_prior": int(
                    len(unique_intersection)
                ),
                "q_interpretation": (
                    "exact expression-filtered matched candidate subset"
                    if str(mode) == "matched"
                    and bool(frame["candidate_in_cytobridge_model_prior"].all())
                    else "intersection with CytoBridge model prior; not complete NicheNet candidate Q"
                ),
            }
        )
    return summary, pd.DataFrame(audit_rows)


SUPPORT_METRICS = {
    "G": "G_AB_abs_mean_per_edge_mean",
    "D": "D_AB_mean",
    "Q_full_model_prior": "Q_AB_total",
    "Q_supplied_singleton_prior": "Q_AB_supplied_singleton_prior",
    "Q_official_overlap_prior": "Q_AB_official_overlap_singleton_prior",
    "Q_context_candidate_intersection": "Q_AB_nichenet_context_candidate_intersection",
}


def sender_support_context_correlations(joined: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_columns = ["prior_mode", "transition", "receiver_type"]
    for key, group in joined.groupby(group_columns, observed=True, sort=True):
        support = _finite_numeric(group["sender_support_fraction"]).to_numpy()
        available_metrics = {
            label: column
            for label, column in SUPPORT_METRICS.items()
            if column in group and _finite_numeric(group[column]).notna().any()
        }
        for label, column in available_metrics.items():
            values = _finite_numeric(group[column]).to_numpy()
            valid = np.isfinite(support) & np.isfinite(values)
            if valid.sum() < 3:
                undefined_reason = "fewer_than_3_finite_senders"
            elif np.ptp(support[valid]) == 0:
                undefined_reason = "all_zero_nichenet_sender_support" if np.all(
                    support[valid] == 0
                ) else "constant_nichenet_sender_support"
            elif np.ptp(values[valid]) == 0:
                undefined_reason = "constant_cytobridge_metric"
            else:
                undefined_reason = ""
            extras = {}
            for extra in ("target_selection_mode", "program_evidence_class"):
                if extra in group:
                    observed = sorted(set(group[extra].dropna().astype(str)))
                    extras[extra] = observed[0] if len(observed) == 1 else "mixed"
            rows.append(
                {
                    "prior_mode": key[0],
                    "transition": key[1],
                    "receiver": key[2],
                    "cytobridge_metric": label,
                    "cytobridge_column": column,
                    "n_senders": int(valid.sum()),
                    "spearman_rho": _safe_spearman(support[valid], values[valid]),
                    "kendall_tau_b": _safe_kendall(support[valid], values[valid]),
                    "nichenet_measure": "custom_sender_support_fraction",
                    "is_native_nichenet_edge_strength": False,
                    "undefined_reason": undefined_reason,
                    **extras,
                }
            )
    return pd.DataFrame(rows)


def _partial_rank_components(
    response: np.ndarray, covariates: Sequence[np.ndarray]
) -> tuple[np.ndarray | None, str]:
    response = np.asarray(response, dtype=float)
    if len(response) < 4:
        return None, "fewer_than_4_finite_senders"
    if np.ptp(response) == 0:
        return None, "constant_response"
    ranked_covariates: list[np.ndarray] = []
    for covariate in covariates:
        covariate = np.asarray(covariate, dtype=float)
        if np.ptp(covariate) == 0:
            return None, "constant_adjustment_covariate"
        ranked_covariates.append(rankdata(covariate, method="average"))
    design = np.column_stack([np.ones(len(response)), *ranked_covariates])
    if np.linalg.matrix_rank(design) < design.shape[1]:
        return None, "rank_deficient_adjustment_design"
    response_rank = rankdata(response, method="average")
    residual = response_rank - design @ np.linalg.lstsq(design, response_rank, rcond=None)[0]
    if np.linalg.norm(residual) <= 1e-12:
        return None, "zero_residual_variance_after_adjustment"
    return residual, ""


def sender_support_partial_correlations(
    joined: pd.DataFrame,
    *,
    permutations: int,
    random_seed: int,
) -> tuple[pd.DataFrame, dict[tuple[str, str, str, str, str], np.ndarray]]:
    """Partial Spearman sensitivity controlling Q and abundance/type propensity."""

    rng = np.random.default_rng(random_seed)
    rows: list[dict[str, Any]] = []
    nulls: dict[tuple[str, str, str, str, str], np.ndarray] = {}
    adjustments = {
        "full_model_prior_Q": ["Q_AB_total"],
        "full_model_prior_Q_plus_abundance": ["Q_AB_total", "n_directed_cell_pairs"],
        "supplied_singleton_prior_Q": ["Q_AB_supplied_singleton_prior"],
        "supplied_singleton_prior_Q_plus_abundance": [
            "Q_AB_supplied_singleton_prior", "n_directed_cell_pairs"
        ],
        "official_overlap_prior_Q": ["Q_AB_official_overlap_singleton_prior"],
        "official_overlap_prior_Q_plus_abundance": [
            "Q_AB_official_overlap_singleton_prior", "n_directed_cell_pairs"
        ],
        "context_candidate_intersection_Q": [
            "Q_AB_nichenet_context_candidate_intersection"
        ],
        "context_candidate_intersection_Q_plus_abundance": [
            "Q_AB_nichenet_context_candidate_intersection", "n_directed_cell_pairs"
        ],
    }
    adjustments = {
        label: columns
        for label, columns in adjustments.items()
        if all(
            column in joined and _finite_numeric(joined[column]).notna().any()
            for column in columns
        )
    }
    for (mode, transition, receiver), group in joined.groupby(
        ["prior_mode", "transition", "receiver_type"], observed=True, sort=True
    ):
        for metric, metric_column in {"G": SUPPORT_METRICS["G"], "D": SUPPORT_METRICS["D"]}.items():
            for adjustment, covariate_columns in adjustments.items():
                needed = ["sender_support_fraction", metric_column, *covariate_columns]
                values = group[needed].apply(_finite_numeric)
                valid = np.isfinite(values.to_numpy(dtype=float)).all(axis=1)
                values = values.loc[valid]
                support = values["sender_support_fraction"].to_numpy(dtype=float)
                metric_values = values[metric_column].to_numpy(dtype=float)
                covariates = [values[column].to_numpy(dtype=float) for column in covariate_columns]
                if len(values) < 4:
                    reason = "fewer_than_4_finite_senders"
                    rho = math.nan
                    null = np.full(max(permutations, 0), np.nan)
                elif np.ptp(support) == 0:
                    reason = (
                        "all_zero_nichenet_sender_support"
                        if np.all(support == 0)
                        else "constant_nichenet_sender_support"
                    )
                    rho = math.nan
                    null = np.full(max(permutations, 0), np.nan)
                elif np.ptp(metric_values) == 0:
                    reason = "constant_cytobridge_metric"
                    rho = math.nan
                    null = np.full(max(permutations, 0), np.nan)
                else:
                    support_residual, support_reason = _partial_rank_components(support, covariates)
                    metric_residual, metric_reason = _partial_rank_components(metric_values, covariates)
                    reason = support_reason or metric_reason
                    if reason:
                        rho = math.nan
                        null = np.full(max(permutations, 0), np.nan)
                    else:
                        assert support_residual is not None and metric_residual is not None
                        rho = float(
                            np.dot(support_residual, metric_residual)
                            / (np.linalg.norm(support_residual) * np.linalg.norm(metric_residual))
                        )
                        design = np.column_stack(
                            [
                                np.ones(len(support)),
                                *[rankdata(item, method="average") for item in covariates],
                            ]
                        )
                        residual_maker = np.eye(len(support)) - design @ np.linalg.pinv(design)
                        support_rank = rankdata(support, method="average")
                        if permutations:
                            permuted = np.vstack(
                                [rng.permutation(support_rank) for _ in range(permutations)]
                            )
                            permuted_residual = permuted @ residual_maker
                            denominators = np.linalg.norm(permuted_residual, axis=1) * np.linalg.norm(
                                metric_residual
                            )
                            null = np.divide(
                                permuted_residual @ metric_residual,
                                denominators,
                                out=np.full(permutations, np.nan),
                                where=denominators > 1e-12,
                            )
                        else:
                            null = np.asarray([], dtype=float)
                finite_null = null[np.isfinite(null)]
                p_value = (
                    float((1 + np.sum(finite_null >= rho)) / (1 + len(finite_null)))
                    if math.isfinite(rho) and len(finite_null)
                    else math.nan
                )
                record = {
                    "prior_mode": mode,
                    "transition": transition,
                    "receiver": receiver,
                    "cytobridge_metric": metric,
                    "adjustment": adjustment,
                    "adjustment_columns": ";".join(covariate_columns),
                    "n_senders": int(len(values)),
                    "partial_spearman_rho": rho,
                    "permutation_p_one_sided": p_value,
                    "n_valid_permutations": int(len(finite_null)),
                    "undefined_reason": reason,
                    "is_native_nichenet_edge_strength": False,
                    "abundance_proxy": (
                        "directed observed cell-pair count; within receiver this tracks sender abundance"
                        if adjustment.endswith("plus_abundance")
                        else "not included"
                    ),
                }
                for extra in ("target_selection_mode", "program_evidence_class"):
                    if extra in group:
                        observed = sorted(set(group[extra].dropna().astype(str)))
                        record[extra] = observed[0] if len(observed) == 1 else "mixed"
                rows.append(record)
                nulls[(str(mode), str(transition), str(receiver), metric, adjustment)] = null
    return pd.DataFrame(rows), nulls


def partial_correlation_macro_statistics(
    context: pd.DataFrame,
    nulls: Mapping[tuple[str, str, str, str, str], np.ndarray],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (mode, metric, adjustment), group in context.groupby(
        ["prior_mode", "cytobridge_metric", "adjustment"], observed=True, sort=True
    ):
        scopes = list(_scope_groups(group.rename(columns={"receiver": "receiver"})))
        for scope, scope_value, scoped in scopes:
            valid = scoped[_finite_numeric(scoped["partial_spearman_rho"]).notna()]
            observed = _finite_numeric(valid["partial_spearman_rho"]).to_numpy(dtype=float)
            keys = [
                (str(mode), str(row.transition), str(row.receiver), str(metric), str(adjustment))
                for row in valid.itertuples(index=False)
            ]
            arrays = [nulls[key] for key in keys if key in nulls]
            if arrays:
                null_macro = np.nanmean(np.column_stack(arrays), axis=1)
                finite_null = null_macro[np.isfinite(null_macro)]
            else:
                finite_null = np.asarray([], dtype=float)
            observed_macro = float(observed.mean()) if len(observed) else math.nan
            p_value = (
                float((1 + np.sum(finite_null >= observed_macro)) / (1 + len(finite_null)))
                if len(finite_null) and math.isfinite(observed_macro)
                else math.nan
            )
            rows.append(
                {
                    "prior_mode": mode,
                    "cytobridge_metric": metric,
                    "adjustment": adjustment,
                    "scope": scope,
                    "scope_value": scope_value,
                    "n_receiver_transition_contexts": int(len(observed)),
                    "macro_partial_spearman_rho": observed_macro,
                    "permutation_p_one_sided": p_value,
                    "n_valid_permutations": int(len(finite_null)),
                    "null_mean": float(finite_null.mean()) if len(finite_null) else math.nan,
                    "null_sd": float(finite_null.std(ddof=1)) if len(finite_null) > 1 else math.nan,
                }
            )
    return pd.DataFrame(rows)


def _scope_groups(context: pd.DataFrame) -> Iterable[tuple[str, str, pd.DataFrame]]:
    yield "overall", "all", context
    for transition, frame in context.groupby("transition", observed=True, sort=True):
        yield "transition", str(transition), frame
    for receiver, frame in context.groupby("receiver", observed=True, sort=True):
        yield "receiver", str(receiver), frame
    if "target_selection_mode" in context:
        for mode, frame in context.groupby(
            "target_selection_mode", observed=True, sort=True, dropna=True
        ):
            yield "target_selection_mode", str(mode), frame


def sender_support_macro_statistics(
    joined: pd.DataFrame,
    context: pd.DataFrame,
    *,
    permutations: int,
    bootstrap_replicates: int,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Macro-average receiver-context correlations and sender-label permutation nulls."""

    rng = np.random.default_rng(random_seed)
    rows: list[dict[str, Any]] = []
    null_rows: list[dict[str, Any]] = []
    for (mode, metric), metric_context in context.groupby(
        ["prior_mode", "cytobridge_metric"], observed=True, sort=True
    ):
        value_column = SUPPORT_METRICS[str(metric)]
        mode_joined = joined[joined["prior_mode"] == mode]
        context_nulls: dict[tuple[str, str], np.ndarray] = {}
        for (transition, receiver), group in mode_joined.groupby(
            ["transition", "receiver_type"], observed=True, sort=True
        ):
            x = _finite_numeric(group["sender_support_fraction"]).to_numpy()
            y = _finite_numeric(group[value_column]).to_numpy()
            valid = np.isfinite(x) & np.isfinite(y)
            x, y = x[valid], y[valid]
            null = np.full(max(permutations, 0), np.nan, dtype=float)
            if len(x) >= 3 and np.ptp(x) > 0 and np.ptp(y) > 0 and len(null):
                x_rank = rankdata(x, method="average")
                y_rank = rankdata(y, method="average")
                x_rank -= x_rank.mean()
                y_rank -= y_rank.mean()
                denominator = float(np.linalg.norm(x_rank) * np.linalg.norm(y_rank))
                if denominator > 0:
                    permuted = np.vstack([rng.permutation(x_rank) for _ in range(len(null))])
                    null = (permuted @ y_rank) / denominator
            context_nulls[(str(transition), str(receiver))] = null
        for scope, scope_value, scoped_context in _scope_groups(metric_context):
            valid_observed = scoped_context["spearman_rho"].dropna().to_numpy(dtype=float)
            observed = float(valid_observed.mean()) if len(valid_observed) else math.nan
            ci_low, ci_high = _bootstrap_mean_ci(
                valid_observed, replicates=bootstrap_replicates, rng=rng
            )
            scoped_keys = [
                (str(row.transition), str(row.receiver))
                for row in scoped_context.itertuples(index=False)
                if (str(row.transition), str(row.receiver)) in context_nulls
                and math.isfinite(float(row.spearman_rho))
            ]
            if scoped_keys and permutations:
                matrix = np.column_stack([context_nulls[key] for key in scoped_keys])
                null_values = np.nanmean(matrix, axis=1)
            else:
                null_values = np.full(max(permutations, 0), np.nan, dtype=float)
            finite_null = null_values[np.isfinite(null_values)]
            p_value = (
                float((1 + np.sum(finite_null >= observed)) / (1 + len(finite_null)))
                if math.isfinite(observed) and len(finite_null)
                else math.nan
            )
            rows.append(
                {
                    "prior_mode": mode,
                    "cytobridge_metric": metric,
                    "scope": scope,
                    "scope_value": scope_value,
                    "n_receiver_transition_contexts": int(len(valid_observed)),
                    "macro_spearman_rho": observed,
                    "bootstrap_ci_low": ci_low,
                    "bootstrap_ci_high": ci_high,
                    "permutation_p_one_sided": p_value,
                    "n_valid_permutations": int(len(finite_null)),
                    "null_mean": float(finite_null.mean()) if len(finite_null) else math.nan,
                    "null_sd": float(finite_null.std(ddof=1)) if len(finite_null) > 1 else math.nan,
                    "is_native_nichenet_edge_strength": False,
                }
            )
            for index, value in enumerate(null_values):
                null_rows.append(
                    {
                        "prior_mode": mode,
                        "cytobridge_metric": metric,
                        "scope": scope,
                        "scope_value": scope_value,
                        "permutation": index,
                        "macro_spearman_rho_null": value,
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(null_rows)


def sender_support_training_seed_stability(
    support: pd.DataFrame, drift_seed: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    support = attach_transition_time(support).rename(
        columns={"sender": "sender_type", "receiver": "receiver_type"}
    )
    keep = [
        "training_seed", "time", "sender_type", "receiver_type", "D_AB",
        "G_AB_abs_mean_per_edge",
    ]
    joined = support.merge(
        drift_seed[keep],
        left_on=["early_time", "sender_type", "receiver_type"],
        right_on=["time", "sender_type", "receiver_type"],
        how="inner",
        validate="many_to_many",
    )
    rows: list[dict[str, Any]] = []
    columns = {"G": "G_AB_abs_mean_per_edge", "D": "D_AB"}
    for mode, seed in (
        joined[["prior_mode", "training_seed"]]
        .drop_duplicates()
        .sort_values(["prior_mode", "training_seed"])
        .itertuples(index=False, name=None)
    ):
        subset = joined[(joined["prior_mode"] == mode) & (joined["training_seed"] == seed)]
        for metric_label, metric_column in columns.items():
            context_values: list[float] = []
            for (transition, receiver), group in subset.groupby(
                ["transition", "receiver_type"], observed=True, sort=True
            ):
                rho = _safe_spearman(
                    _finite_numeric(group["sender_support_fraction"]),
                    _finite_numeric(group[metric_column]),
                )
                rows.append(
                    {
                        "prior_mode": mode,
                        "training_seed": int(seed),
                        "transition": transition,
                        "receiver": receiver,
                        "cytobridge_metric": metric_label,
                        "spearman_rho": rho,
                        "is_macro": False,
                        "is_native_nichenet_edge_strength": False,
                    }
                )
                if math.isfinite(rho):
                    context_values.append(rho)
            rows.append(
                {
                    "prior_mode": mode,
                    "training_seed": int(seed),
                    "transition": "all",
                    "receiver": "MACRO",
                    "cytobridge_metric": metric_label,
                    "spearman_rho": float(np.mean(context_values)) if context_values else math.nan,
                    "is_macro": True,
                    "is_native_nichenet_edge_strength": False,
                }
            )
    per_seed = pd.DataFrame(rows)
    if per_seed.empty:
        return per_seed, pd.DataFrame(
            columns=[
                "prior_mode", "cytobridge_metric", "n_training_seeds",
                "macro_spearman_mean", "macro_spearman_sd", "macro_spearman_min",
                "macro_spearman_max", "is_native_nichenet_edge_strength",
            ]
        )
    macro = per_seed[per_seed["is_macro"]].copy()
    summary = (
        macro.groupby(["prior_mode", "cytobridge_metric"], observed=True, sort=True)
        .agg(
            n_training_seeds=("training_seed", "nunique"),
            macro_spearman_mean=("spearman_rho", "mean"),
            macro_spearman_sd=("spearman_rho", "std"),
            macro_spearman_min=("spearman_rho", "min"),
            macro_spearman_max=("spearman_rho", "max"),
        )
        .reset_index()
    )
    summary["is_native_nichenet_edge_strength"] = False
    return per_seed, summary


def _aggregate_feature_scores(frame: pd.DataFrame, feature: str, score: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=[feature, score])
    ranked = frame[[feature, score]].copy()
    ranked[score] = _finite_numeric(ranked[score])
    ranked = ranked.dropna(subset=[feature, score])
    ranked[feature] = ranked[feature].astype(str)
    ranked = (
        ranked.groupby(feature, observed=True, as_index=False)[score]
        .sum()
        .sort_values([score, feature], ascending=[False, True], kind="mergesort")
    )
    return ranked


def _top_set(frame: pd.DataFrame, feature: str, score: str, k: int) -> set[str]:
    """Positive-score top-k with tie-inclusive boundary (never lexical tie breaking)."""

    ranked = _aggregate_feature_scores(frame, feature, score)
    ranked = ranked[_finite_numeric(ranked[score]) > 0]
    if ranked.empty or k <= 0:
        return set()
    if len(ranked) <= k:
        return set(ranked[feature].astype(str))
    cutoff = float(ranked.iloc[k - 1][score])
    return set(ranked.loc[_finite_numeric(ranked[score]) >= cutoff, feature].astype(str))


def ranked_set_overlap(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    feature: str,
    left_score: str,
    right_score: str,
    k: int,
    universe: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Compare top-k sets on a declared common feature universe."""

    left_features = set(left[feature].dropna().astype(str)) if feature in left else set()
    right_features = set(right[feature].dropna().astype(str)) if feature in right else set()
    common = left_features & right_features
    if universe is not None:
        common &= {str(item) for item in universe}
    left_common = left[left[feature].astype(str).isin(common)].copy() if len(left) else left.copy()
    right_common = right[right[feature].astype(str).isin(common)].copy() if len(right) else right.copy()
    left_aggregated = _aggregate_feature_scores(left_common, feature, left_score)
    right_aggregated = _aggregate_feature_scores(right_common, feature, right_score)
    left_positive = set(
        left_aggregated.loc[_finite_numeric(left_aggregated[left_score]) > 0, feature].astype(str)
    )
    right_positive = set(
        right_aggregated.loc[_finite_numeric(right_aggregated[right_score]) > 0, feature].astype(str)
    )
    effective_k = min(int(k), len(common))
    left_top = _top_set(left_common, feature, left_score, effective_k)
    right_top = _top_set(right_common, feature, right_score, effective_k)
    overlap = left_top & right_top
    union = left_top | right_top
    n_universe = len(common)
    n_left, n_right = len(left_top), len(right_top)
    hypergeom_p = (
        float(hypergeom.sf(len(overlap) - 1, n_universe, n_left, n_right))
        if n_universe and n_left and n_right
        else math.nan
    )
    return {
        "requested_k": int(k),
        "effective_k": int(effective_k),
        "common_universe_size": int(n_universe),
        "n_left_positive_in_common_universe": int(len(left_positive)),
        "n_right_positive_in_common_universe": int(len(right_positive)),
        "n_both_positive_in_common_universe": int(len(left_positive & right_positive)),
        "fraction_common_universe_left_positive": (
            float(len(left_positive) / n_universe) if n_universe else math.nan
        ),
        "fraction_common_universe_right_positive": (
            float(len(right_positive) / n_universe) if n_universe else math.nan
        ),
        "n_left_top": int(n_left),
        "n_right_top": int(n_right),
        "n_overlap": int(len(overlap)),
        "jaccard": float(len(overlap) / len(union)) if union else math.nan,
        "overlap_coefficient": (
            float(len(overlap) / min(n_left, n_right)) if min(n_left, n_right) else math.nan
        ),
        "hypergeometric_p": hypergeom_p,
        "overlap_features": ";".join(sorted(overlap)),
        "left_top_features": ";".join(sorted(left_top)),
        "right_top_features": ";".join(sorted(right_top)),
        "tie_policy": "positive scores only; all features tied at kth score are included",
        "undefined_reason": (
            "no_positive_score_common_universe"
            if not left_positive or not right_positive
            else ""
        ),
    }


def build_native_ligand_summary(
    activities: pd.DataFrame, links: pd.DataFrame, *, top_n: int = 10
) -> pd.DataFrame:
    """Return native receiver-context NicheNet ligand activity summaries."""

    rows: list[pd.DataFrame] = []
    for _, group in activities.groupby(
        ["prior_mode", "transition", "receiver"], observed=True, sort=True
    ):
        group = group.copy()
        group["aupr_corrected"] = _finite_numeric(group["aupr_corrected"])
        group["pearson"] = _finite_numeric(group["pearson"])
        group = group.sort_values(
            ["aupr_corrected", "pearson", "ligand"],
            ascending=[False, False, True],
            kind="mergesort",
        ).head(top_n)
        group["summary_rank"] = np.arange(1, len(group) + 1)
        rows.append(group)
    summary = pd.concat(rows, ignore_index=True) if rows else activities.head(0).copy()
    if len(summary):
        target_counts = (
            links.groupby(["prior_mode", "transition", "receiver", "ligand"], observed=True)
            .agg(
                n_exported_targets=("target", "nunique"),
                max_regulatory_potential=("regulatory_potential", "max"),
                sum_regulatory_potential=("regulatory_potential", "sum"),
            )
            .reset_index()
        )
        summary = summary.merge(
            target_counts,
            on=["prior_mode", "transition", "receiver", "ligand"],
            how="left",
            validate="one_to_one",
        )
    summary["native_nichenet_semantics"] = (
        "receiver-context ligand activity for predicting the temporal target program; "
        "not sender-to-receiver CCC edge strength"
    )
    return summary


def cross_prior_native_activity_audit(activities: pd.DataFrame) -> pd.DataFrame:
    """Verify that common ligands retain identical native activity across prior modes."""

    modes = set(activities["prior_mode"].astype(str))
    if not {"default", "matched"}.issubset(modes):
        return pd.DataFrame()
    columns = ["transition", "receiver", "ligand", "auroc", "aupr", "aupr_corrected", "pearson"]
    default = activities[activities["prior_mode"] == "default"][columns].copy()
    matched = activities[activities["prior_mode"] == "matched"][columns].copy()
    joined = default.merge(
        matched,
        on=["transition", "receiver", "ligand"],
        how="inner",
        suffixes=("_default", "_matched"),
        validate="one_to_one",
    )
    for metric in ("auroc", "aupr", "aupr_corrected", "pearson"):
        joined[f"{metric}_absolute_difference"] = np.abs(
            _finite_numeric(joined[f"{metric}_default"])
            - _finite_numeric(joined[f"{metric}_matched"])
        )
    joined["interpretation"] = (
        "same official ligand-target matrix and receiver target/background; matched mode only "
        "restricts which potential ligands/LR pairs enter the candidate universe"
    )
    return joined


def _custom_lr_scores(
    activities: pd.DataFrame,
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    """Construct explicitly custom sender-specific LR support for rank overlap."""

    if candidates.empty:
        return pd.DataFrame(
            columns=[
                "prior_mode", "transition", "sender", "receiver", "ligand", "receptor",
                "custom_nichenet_lr_support",
            ]
        )
    candidate = candidates.copy()
    activity = activities[
        ["prior_mode", "transition", "receiver", "ligand", "aupr_corrected"]
    ].copy()
    candidate = candidate.merge(
        activity,
        on=["prior_mode", "transition", "receiver", "ligand"],
        how="left",
        validate="many_to_one",
    )
    unique_sender_ligand = candidate[
        [
            "prior_mode", "transition", "sender", "receiver", "ligand",
            "sender_expression_signal",
        ]
    ].drop_duplicates()
    denominator = (
        unique_sender_ligand.groupby(
            ["prior_mode", "transition", "receiver", "ligand"], observed=True
        )["sender_expression_signal"]
        .sum()
        .rename("ligand_sender_signal_total")
        .reset_index()
    )
    candidate = candidate.merge(
        denominator,
        on=["prior_mode", "transition", "receiver", "ligand"],
        how="left",
        validate="many_to_one",
    )
    numerator = _finite_numeric(candidate["sender_expression_signal"]).fillna(0).clip(lower=0)
    denominator_values = _finite_numeric(candidate["ligand_sender_signal_total"])
    candidate["sender_expression_share"] = np.divide(
        numerator,
        denominator_values,
        out=np.zeros(len(candidate), dtype=float),
        where=denominator_values.to_numpy(dtype=float) > 0,
    )
    candidate["positive_ligand_activity"] = (
        _finite_numeric(candidate["aupr_corrected"]).fillna(0).clip(lower=0)
    )
    candidate["custom_nichenet_lr_support"] = (
        candidate["positive_ligand_activity"]
        * candidate["sender_expression_share"]
        * _finite_numeric(candidate["receiver_receptor_support"]).fillna(0).clip(lower=0)
    )
    result = (
        candidate.groupby(
            ["prior_mode", "transition", "sender", "receiver", "ligand", "receptor"],
            observed=True,
            as_index=False,
        )["custom_nichenet_lr_support"]
        .max()
    )
    result["is_native_nichenet_edge_strength"] = False
    return result


def compute_ranked_overlaps(
    nichenet_tables: Mapping[str, pd.DataFrame],
    learned_lr: pd.DataFrame,
    learned_pathway: pd.DataFrame,
    *,
    top_k: Sequence[int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compare ligand, singleton LR, and post-hoc CellChat-pathway rankings."""

    activities = nichenet_tables["activities"].copy()
    candidates = nichenet_tables["candidates"].copy()
    custom_lr = _custom_lr_scores(activities, candidates)
    learned_lr = learned_lr.copy()
    learned_lr["early_time"] = _finite_numeric(learned_lr["time"])
    learned_pathway = learned_pathway.copy()
    learned_pathway["early_time"] = _finite_numeric(learned_pathway["time"])
    learned_lr_by_receiver = {
        (float(time), str(receiver)): frame
        for (time, receiver), frame in learned_lr.groupby(
            ["early_time", "receiver_type"], observed=True, sort=False
        )
    }
    learned_lr_by_edge = {
        (float(time), str(sender), str(receiver)): frame
        for (time, sender, receiver), frame in learned_lr.groupby(
            ["early_time", "sender_type", "receiver_type"], observed=True, sort=False
        )
    }
    learned_pathway_by_edge = {
        (float(time), str(sender), str(receiver)): frame
        for (time, sender, receiver), frame in learned_pathway.groupby(
            ["early_time", "sender_type", "receiver_type"], observed=True, sort=False
        )
    }

    rows: list[dict[str, Any]] = []
    membership_rows: list[dict[str, Any]] = []
    pathway_score_rows: list[pd.DataFrame] = []

    # Receiver-level ligand activity versus CytoBridge LR-annotated scores pooled
    # over senders.  Native NicheNet activity remains receiver-centric.
    for (mode, transition, receiver), activity_context in activities.groupby(
        ["prior_mode", "transition", "receiver"], observed=True, sort=True
    ):
        early = transition_early_time(transition)
        lr_context = learned_lr_by_receiver.get(
            (float(early), str(receiver)), learned_lr.head(0)
        )
        candidate_context = candidates[
            (candidates["prior_mode"] == mode)
            & (candidates["transition"] == transition)
            & (candidates["receiver"] == receiver)
        ]
        common_universe = set(candidate_context["ligand"].astype(str)) & set(
            lr_context["ligand"].astype(str)
        )
        for learned_score in ("S_AB_lr_pair", "Q_AB_lr_pair"):
            for k in top_k:
                result = ranked_set_overlap(
                    activity_context,
                    lr_context,
                    feature="ligand",
                    left_score="aupr_corrected",
                    right_score=learned_score,
                    k=k,
                    universe=common_universe,
                )
                rows.append(
                    {
                        "prior_mode": mode,
                        "transition": transition,
                        "sender": "ALL_SENDERS",
                        "receiver": receiver,
                        "feature_level": "ligand",
                        "nichenet_score": "native_aupr_corrected",
                        "cytobridge_score": learned_score,
                        "pathway_mapping_semantics": "not_applicable",
                        **result,
                    }
                )

    # Sender-specific custom LR score versus exact singleton rows in CytoBridge.
    for (mode, transition, sender, receiver), nn_context in custom_lr.groupby(
        ["prior_mode", "transition", "sender", "receiver"], observed=True, sort=True
    ):
        early = transition_early_time(transition)
        cb_context = learned_lr_by_edge.get(
            (float(early), str(sender), str(receiver)), learned_lr.head(0)
        ).copy()
        nn_context = nn_context.copy()
        nn_context["lr_pair"] = nn_context["ligand"].astype(str) + "→" + nn_context["receptor"].astype(str)
        cb_context["lr_pair"] = cb_context["ligand"].astype(str) + "→" + cb_context["receptor"].astype(str)
        common_pairs = set(nn_context["lr_pair"]) & set(cb_context["lr_pair"])
        nn_pair = nn_context.rename(columns={"lr_pair": "feature"})
        cb_pair = cb_context.rename(columns={"lr_pair": "feature"})
        for learned_score in ("S_AB_lr_pair", "Q_AB_lr_pair"):
            for k in top_k:
                result = ranked_set_overlap(
                    nn_pair,
                    cb_pair,
                    feature="feature",
                    left_score="custom_nichenet_lr_support",
                    right_score=learned_score,
                    k=k,
                    universe=common_pairs,
                )
                rows.append(
                    {
                        "prior_mode": mode,
                        "transition": transition,
                        "sender": sender,
                        "receiver": receiver,
                        "feature_level": "ligand_receptor",
                        "nichenet_score": "custom_sender_specific_lr_support",
                        "cytobridge_score": learned_score,
                        "pathway_mapping_semantics": "not_applicable",
                        **result,
                    }
                )

        # Map only exact singleton LR pairs into the CellChat pathway labels used
        # by the CytoBridge prior.  This is explicitly post-hoc, not NicheNet-native.
        mapping = cb_context[["ligand", "receptor", "pathway"]].drop_duplicates()
        mapped = nn_context.merge(
            mapping,
            on=["ligand", "receptor"],
            how="inner",
            validate="many_to_many",
        )
        nn_pathway = (
            mapped.groupby("pathway", observed=True, as_index=False)["custom_nichenet_lr_support"]
            .sum()
        )
        nn_pathway["prior_mode"] = mode
        nn_pathway["transition"] = transition
        nn_pathway["sender"] = sender
        nn_pathway["receiver"] = receiver
        nn_pathway["mapping"] = "exact_singleton_LR_to_CytoBridge_CellChat_pathway"
        pathway_score_rows.append(nn_pathway)
        cb_pathway = learned_pathway_by_edge.get(
            (float(early), str(sender), str(receiver)), learned_pathway.head(0)
        )
        common_pathways = set(nn_pathway["pathway"].astype(str)) & set(
            cb_pathway["pathway"].astype(str)
        )
        for learned_score in ("S_AB_pathway", "Q_AB_pathway"):
            for k in top_k:
                result = ranked_set_overlap(
                    nn_pathway,
                    cb_pathway,
                    feature="pathway",
                    left_score="custom_nichenet_lr_support",
                    right_score=learned_score,
                    k=k,
                    universe=common_pathways,
                )
                rows.append(
                    {
                        "prior_mode": mode,
                        "transition": transition,
                        "sender": sender,
                        "receiver": receiver,
                        "feature_level": "pathway",
                        "nichenet_score": "custom_LR_support_mapped_posthoc_to_CellChat_pathway",
                        "cytobridge_score": learned_score,
                        "pathway_mapping_semantics": (
                            "posthoc exact-singleton LR mapping to CytoBridge CellChat pathways; "
                            "not a native NicheNet pathway call"
                        ),
                        **result,
                    }
                )

    overlap = pd.DataFrame(rows)
    for _, row in overlap.iterrows():
        for side in ("left", "right", "overlap"):
            values = str(row.get(f"{side}_top_features" if side != "overlap" else "overlap_features", ""))
            if not values:
                continue
            for feature_value in values.split(";"):
                membership_rows.append(
                    {
                        "prior_mode": row["prior_mode"],
                        "transition": row["transition"],
                        "sender": row["sender"],
                        "receiver": row["receiver"],
                        "feature_level": row["feature_level"],
                        "cytobridge_score": row["cytobridge_score"],
                        "requested_k": row["requested_k"],
                        "membership": side,
                        "feature": feature_value,
                    }
                )
    pathway_scores = (
        pd.concat(pathway_score_rows, ignore_index=True)
        if pathway_score_rows
        else pd.DataFrame(
            columns=[
                "pathway", "custom_nichenet_lr_support", "prior_mode", "transition",
                "sender", "receiver", "mapping",
            ]
        )
    )
    return overlap, pd.DataFrame(membership_rows), pathway_scores


def summarize_database_coverage(
    runs: Sequence[Mapping[str, pd.DataFrame]], learned_lr: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    learned_pairs = set(
        learned_lr["ligand"].astype(str) + "\r" + learned_lr["receptor"].astype(str)
    )
    learned_ligands = set(learned_lr["ligand"].astype(str))
    rows: list[dict[str, Any]] = []
    contexts: list[pd.DataFrame] = []
    for tables in runs:
        prior = tables["prior_coverage"].copy()
        candidate = tables["candidates"].copy()
        mode = str(prior["prior_mode"].iloc[0]) if len(prior) else (
            str(candidate["prior_mode"].iloc[0]) if len(candidate) else "unknown"
        )
        candidate_pairs = set(
            candidate["ligand"].astype(str) + "\r" + candidate["receptor"].astype(str)
        )
        candidate_ligands = set(candidate["ligand"].astype(str))
        record = prior.iloc[0].to_dict() if len(prior) else {"prior_mode": mode}
        record.update(
            {
                "cytobridge_unique_lr_pairs": len(learned_pairs),
                "cytobridge_unique_ligands": len(learned_ligands),
                "contextual_nichenet_unique_lr_pairs": len(candidate_pairs),
                "contextual_nichenet_unique_ligands": len(candidate_ligands),
                "exact_lr_pair_intersection": len(candidate_pairs & learned_pairs),
                "ligand_intersection": len(candidate_ligands & learned_ligands),
                "fraction_cytobridge_lr_pairs_contextually_covered": (
                    len(candidate_pairs & learned_pairs) / len(learned_pairs) if learned_pairs else math.nan
                ),
                "fraction_contextual_nichenet_lr_pairs_in_cytobridge": (
                    len(candidate_pairs & learned_pairs) / len(candidate_pairs) if candidate_pairs else math.nan
                ),
            }
        )
        rows.append(record)
        context = tables["coverage"].copy()
        contexts.append(context)
    return pd.DataFrame(rows), pd.concat(contexts, ignore_index=True)


def summarize_ranked_overlaps(overlap: pd.DataFrame) -> pd.DataFrame:
    """Macro overlap only where each side has at least k positive candidates."""

    if overlap.empty:
        return pd.DataFrame()
    frame = overlap.copy()
    frame["eligible_positive_top_k"] = (
        (frame["n_left_positive_in_common_universe"] >= frame["requested_k"])
        & (frame["n_right_positive_in_common_universe"] >= frame["requested_k"])
    )
    rows: list[dict[str, Any]] = []
    group_columns = [
        "prior_mode", "feature_level", "nichenet_score", "cytobridge_score", "requested_k"
    ]
    for key, group in frame.groupby(group_columns, observed=True, sort=True):
        for scope, scoped in [("overall", group)] + [
            (f"transition:{transition}", subset)
            for transition, subset in group.groupby("transition", observed=True, sort=True)
        ]:
            eligible = scoped[scoped["eligible_positive_top_k"] & scoped["jaccard"].notna()]
            rows.append(
                {
                    **dict(zip(group_columns, key)),
                    "scope": scope,
                    "n_contexts_total": int(len(scoped)),
                    "n_contexts_eligible_positive_top_k": int(len(eligible)),
                    "eligible_fraction": float(len(eligible) / len(scoped)) if len(scoped) else math.nan,
                    "macro_jaccard": float(eligible["jaccard"].mean()) if len(eligible) else math.nan,
                    "macro_overlap_coefficient": (
                        float(eligible["overlap_coefficient"].mean()) if len(eligible) else math.nan
                    ),
                    "fraction_hypergeometric_p_lt_0_05": (
                        float((eligible["hypergeometric_p"] < 0.05).mean())
                        if len(eligible)
                        else math.nan
                    ),
                    "eligibility_rule": (
                        "both methods have at least requested_k strictly positive features in "
                        "the common covered universe; kth-score ties are included"
                    ),
                }
            )
    return pd.DataFrame(rows)


def _pc_columns(frame: pd.DataFrame, suffix: str = "") -> list[str]:
    pattern = re.compile(rf"^drift_pc_(\d+){re.escape(suffix)}$")
    pairs: list[tuple[int, str]] = []
    for column in frame.columns:
        match = pattern.fullmatch(column)
        if match:
            pairs.append((int(match.group(1)), column))
    pairs.sort()
    if pairs and [index for index, _ in pairs] != list(range(1, len(pairs) + 1)):
        raise ValueError(f"PC columns with suffix {suffix!r} are not contiguous from 1.")
    return [column for _, column in pairs]


def project_drift_table(
    drift: pd.DataFrame,
    *,
    pc_columns: Sequence[str],
    pca_info: Mapping[str, Any],
    value_column: str = "projected_gene_drift",
) -> pd.DataFrame:
    """Project rows to long-format HVG scores without ever inferring from D."""

    if not pc_columns:
        raise ValueError("No signed drift PC vector columns were supplied.")
    loadings = np.asarray(pca_info["loadings"], dtype=float)
    hvg = np.asarray(pca_info["highly_variable"], dtype=bool)
    genes = np.asarray(pca_info["genes"]).astype(str)
    if len(pc_columns) != loadings.shape[1]:
        raise ValueError(
            f"Drift table has {len(pc_columns)} PCs but PCA artifact has {loadings.shape[1]}."
        )
    projected = project_pc_drift(
        drift[list(pc_columns)].to_numpy(dtype=float), loadings[hvg]
    )
    identifiers = [
        column
        for column in [
            "prior_mode", "training_seed", "time", "transition", "sender_type",
            "receiver_type",
        ]
        if column in drift.columns
    ]
    pieces: list[pd.DataFrame] = []
    hvg_genes = genes[hvg]
    for row_index in range(len(drift)):
        piece = pd.DataFrame({"gene": hvg_genes, value_column: projected[row_index]})
        for column in identifiers:
            piece[column] = drift.iloc[row_index][column]
        pieces.append(piece)
    columns = identifiers + ["gene", value_column]
    return pd.concat(pieces, ignore_index=True)[columns] if pieces else pd.DataFrame(columns=columns)


def build_nichenet_target_scores(
    activities: pd.DataFrame,
    links: pd.DataFrame,
    components: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build receiver-level and custom sender-level ligand-weighted target scores."""

    activity = activities[
        ["prior_mode", "transition", "receiver", "ligand", "aupr_corrected"]
    ].copy()
    activity["positive_ligand_activity"] = (
        _finite_numeric(activity["aupr_corrected"]).fillna(0).clip(lower=0)
    )
    links_i = links.merge(
        activity,
        on=["prior_mode", "transition", "receiver", "ligand"],
        how="left",
        validate="many_to_one",
    )
    links_i["receiver_target_component"] = (
        _finite_numeric(links_i["regulatory_potential"]).fillna(0)
        * links_i["positive_ligand_activity"]
    )
    receiver = (
        links_i.groupby(
            ["prior_mode", "transition", "receiver", "target"],
            observed=True,
            as_index=False,
        )["receiver_target_component"]
        .sum()
        .rename(columns={"target": "gene", "receiver_target_component": "nichenet_target_score"})
    )
    receiver["target_score_semantics"] = (
        "derived sum of positive native ligand activity times official regulatory potential"
    )

    component = components[
        ["prior_mode", "transition", "sender", "receiver", "ligand", "sender_support_component"]
    ].copy()
    sender_links = links.merge(
        component,
        on=["prior_mode", "transition", "receiver", "ligand"],
        how="inner",
        validate="many_to_many",
    )
    sender_links["sender_target_component"] = (
        _finite_numeric(sender_links["regulatory_potential"]).fillna(0)
        * _finite_numeric(sender_links["sender_support_component"]).fillna(0)
    )
    sender = (
        sender_links.groupby(
            ["prior_mode", "transition", "sender", "receiver", "target"],
            observed=True,
            as_index=False,
        )["sender_target_component"]
        .sum()
        .rename(columns={"target": "gene", "sender_target_component": "custom_sender_target_score"})
    )
    sender["target_score_semantics"] = (
        "custom sender-support component times official regulatory potential; not native edge strength"
    )
    return receiver, sender


def _gene_permutation_p(
    x: np.ndarray,
    y: np.ndarray,
    *,
    observed: float,
    permutations: int,
    rng: np.random.Generator,
) -> tuple[float, float, float, int, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return math.nan, math.nan, math.nan, 0, np.full(max(permutations, 0), np.nan)
    # Spearman is Pearson on ranks.  Rank once and permute the centered rank
    # vector, avoiding thousands of repeated O(n log n) sorts per context.
    x_rank = rankdata(x, method="average")
    y_rank = rankdata(y, method="average")
    x_rank = x_rank - x_rank.mean()
    y_rank = y_rank - y_rank.mean()
    denominator = float(np.linalg.norm(x_rank) * np.linalg.norm(y_rank))
    null = np.full(max(permutations, 0), np.nan, dtype=float)
    if denominator > 0:
        for start in range(0, len(null), 128):
            stop = min(start + 128, len(null))
            permuted = np.vstack([rng.permutation(x_rank) for _ in range(stop - start)])
            null[start:stop] = (permuted @ y_rank) / denominator
    finite = null[np.isfinite(null)]
    p = (
        float((1 + np.sum(finite >= observed)) / (1 + len(finite)))
        if math.isfinite(observed) and len(finite)
        else math.nan
    )
    return (
        p,
        float(finite.mean()) if len(finite) else math.nan,
        float(finite.std(ddof=1)) if len(finite) > 1 else math.nan,
        int(len(finite)),
        null,
    )


def compute_gene_space_analyses(
    *,
    learned_summary: pd.DataFrame,
    learned_by_seed: pd.DataFrame,
    nichenet_tables: Mapping[str, pd.DataFrame],
    receiver_de: pd.DataFrame,
    pca_info: Mapping[str, Any],
    permutations: int,
    random_seed: int,
) -> dict[str, pd.DataFrame]:
    """Compare exact subspace message projections to temporal response programs."""

    rng = np.random.default_rng(random_seed)
    summary_pc = _pc_columns(learned_summary, suffix="_mean")
    seed_pc = _pc_columns(learned_by_seed)
    if len(summary_pc) != int(pca_info["n_pcs"]) or len(seed_pc) != int(pca_info["n_pcs"]):
        raise ValueError("Formal signed message-vector PC columns do not match PCA artifact.")

    transitions = sorted(nichenet_tables["activities"]["transition"].astype(str).unique())
    time_map = {transition: transition_early_time(transition) for transition in transitions}
    summary_rows: list[pd.DataFrame] = []
    seed_rows: list[pd.DataFrame] = []
    for transition, early in time_map.items():
        selected = learned_summary[np.isclose(_finite_numeric(learned_summary["time"]), early)].copy()
        selected["transition"] = transition
        summary_rows.append(selected)
        selected_seed = learned_by_seed[
            np.isclose(_finite_numeric(learned_by_seed["time"]), early)
        ].copy()
        selected_seed["transition"] = transition
        seed_rows.append(selected_seed)
    summary_selected = pd.concat(summary_rows, ignore_index=True) if summary_rows else learned_summary.head(0)
    seed_selected = pd.concat(seed_rows, ignore_index=True) if seed_rows else learned_by_seed.head(0)
    analyzed_contexts = {
        (str(transition), str(receiver))
        for transition, receiver in nichenet_tables["activities"][
            ["transition", "receiver"]
        ].drop_duplicates().itertuples(index=False, name=None)
    }
    summary_selected = summary_selected[
        [
            (str(transition), str(receiver)) in analyzed_contexts
            for transition, receiver in summary_selected[
                ["transition", "receiver_type"]
            ].itertuples(index=False, name=None)
        ]
    ].copy()
    seed_selected = seed_selected[
        [
            (str(transition), str(receiver)) in analyzed_contexts
            for transition, receiver in seed_selected[
                ["transition", "receiver_type"]
            ].itertuples(index=False, name=None)
        ]
    ].copy()

    # Projection is linear.  Summing sender vectors in PC space first gives the
    # receiver-level total interaction message with no scalar-D approximation.
    receiver_pc = (
        summary_selected.groupby(["transition", "receiver_type"], observed=True, as_index=False)[summary_pc]
        .sum()
    )
    receiver_gene = project_drift_table(
        receiver_pc,
        pc_columns=summary_pc,
        pca_info=pca_info,
        value_column="cytobridge_total_interaction_gene_drift",
    )
    sender_gene = project_drift_table(
        summary_selected,
        pc_columns=summary_pc,
        pca_info=pca_info,
        value_column="cytobridge_sender_interaction_gene_drift",
    )

    receiver_target, sender_target = build_nichenet_target_scores(
        nichenet_tables["activities"],
        nichenet_tables["links"],
        nichenet_tables["components"],
    )
    de = receiver_de.copy()
    _require_columns(
        de,
        [
            "transition", "receiver", "gene", "effect", "selected_target",
            "target_selection_mode",
        ],
        "receiver_de_genes.csv",
    )
    de["selected_target"] = _as_boolean(de["selected_target"])
    de["effect"] = _finite_numeric(de["effect"])

    agreement_rows: list[dict[str, Any]] = []
    receiver_join_rows: list[pd.DataFrame] = []
    receiver_nulls: dict[tuple[str, str, str, str], np.ndarray] = {}
    for (mode, transition, receiver), target_context in receiver_target.groupby(
        ["prior_mode", "transition", "receiver"], observed=True, sort=True
    ):
        drift_context = receiver_gene[
            (receiver_gene["transition"] == transition)
            & (receiver_gene["receiver_type"].astype(str) == str(receiver))
        ][["gene", "cytobridge_total_interaction_gene_drift"]]
        de_context = de[
            (de["transition"].astype(str) == str(transition))
            & (de["receiver"].astype(str) == str(receiver))
        ][["gene", "effect", "selected_target", "target_selection_mode"]]
        selection_modes = sorted(
            set(
                de_context.loc[de_context["selected_target"], "target_selection_mode"]
                .dropna()
                .astype(str)
            )
            - {"not_selected"}
        )
        if len(selection_modes) != 1:
            raise ValueError(
                f"Non-unique target selection mode for {transition}/{receiver}: {selection_modes}."
            )
        selection_mode = selection_modes[0]
        evidence_class = (
            "fdr_positive_only"
            if selection_mode == "fdr_positive"
            else "fallback_ranked_positive"
        )
        joined = drift_context.merge(de_context, on="gene", how="inner", validate="one_to_one")
        joined = joined.merge(
            target_context[["gene", "nichenet_target_score"]],
            on="gene",
            how="left",
            validate="one_to_one",
        )
        joined["nichenet_target_score"] = joined["nichenet_target_score"].fillna(0)
        joined["prior_mode"] = mode
        joined["transition"] = transition
        joined["receiver"] = receiver
        joined["program_evidence_class"] = evidence_class
        receiver_join_rows.append(joined)
        drift_values = joined["cytobridge_total_interaction_gene_drift"].to_numpy(dtype=float)
        de_values = joined["effect"].to_numpy(dtype=float)
        target_values = joined["nichenet_target_score"].to_numpy(dtype=float)
        rho_de = _safe_spearman(drift_values, de_values)
        rho_target = _safe_spearman(drift_values, target_values)
        p_de, null_de, null_de_sd, n_de_perm, null_de_values = _gene_permutation_p(
            drift_values,
            de_values,
            observed=rho_de,
            permutations=permutations,
            rng=rng,
        )
        p_target, null_target, null_target_sd, n_target_perm, null_target_values = _gene_permutation_p(
            drift_values,
            target_values,
            observed=rho_target,
            permutations=permutations,
            rng=rng,
        )
        nn_de_observed = _safe_spearman(target_values, de_values)
        p_nn_de, null_nn_de, null_nn_de_sd, n_nn_de_perm, null_nn_de_values = _gene_permutation_p(
            target_values,
            de_values,
            observed=nn_de_observed,
            permutations=permutations,
            rng=rng,
        )
        receiver_nulls[(str(mode), str(transition), str(receiver), "spearman_cytobridge_drift_vs_temporal_de")] = null_de_values
        receiver_nulls[(str(mode), str(transition), str(receiver), "spearman_cytobridge_drift_vs_nichenet_target_score")] = null_target_values
        receiver_nulls[(str(mode), str(transition), str(receiver), "spearman_nichenet_target_score_vs_temporal_de")] = null_nn_de_values
        agreement_rows.append(
            {
                "prior_mode": mode,
                "transition": transition,
                "receiver": receiver,
                "target_selection_mode": selection_mode,
                "program_evidence_class": evidence_class,
                "n_hvg_with_temporal_de": int(len(joined)),
                "n_selected_up_targets": int(joined["selected_target"].sum()),
                "n_nichenet_linked_targets": int((target_values != 0).sum()),
                "spearman_cytobridge_drift_vs_temporal_de": rho_de,
                "auc_cytobridge_drift_for_selected_up_targets": _rank_auc(
                    drift_values, joined["selected_target"]
                ),
                "spearman_cytobridge_drift_vs_nichenet_target_score": rho_target,
                "spearman_nichenet_target_score_vs_temporal_de": nn_de_observed,
                "permutation_p_drift_vs_de_one_sided": p_de,
                "permutation_null_mean_drift_vs_de": null_de,
                "permutation_null_sd_drift_vs_de": null_de_sd,
                "n_valid_permutations_drift_vs_de": n_de_perm,
                "permutation_p_drift_vs_nichenet_one_sided": p_target,
                "permutation_null_mean_drift_vs_nichenet": null_target,
                "permutation_null_sd_drift_vs_nichenet": null_target_sd,
                "n_valid_permutations_drift_vs_nichenet": n_target_perm,
                "permutation_p_nichenet_vs_de_one_sided": p_nn_de,
                "permutation_null_mean_nichenet_vs_de": null_nn_de,
                "permutation_null_sd_nichenet_vs_de": null_nn_de_sd,
                "n_valid_permutations_nichenet_vs_de": n_nn_de_perm,
                "projection_semantics": pca_info["interpretation"],
            }
        )

    sender_agreement_rows: list[dict[str, Any]] = []
    sender_join_rows: list[pd.DataFrame] = []
    sender_gene_i = sender_gene.rename(
        columns={"sender_type": "sender", "receiver_type": "receiver"}
    )
    for (mode, transition, sender, receiver), target_context in sender_target.groupby(
        ["prior_mode", "transition", "sender", "receiver"], observed=True, sort=True
    ):
        drift_context = sender_gene_i[
            (sender_gene_i["transition"] == transition)
            & (sender_gene_i["sender"].astype(str) == str(sender))
            & (sender_gene_i["receiver"].astype(str) == str(receiver))
        ][["gene", "cytobridge_sender_interaction_gene_drift"]]
        joined = drift_context.merge(
            target_context[["gene", "custom_sender_target_score"]],
            on="gene",
            how="left",
            validate="one_to_one",
        )
        joined["custom_sender_target_score"] = joined["custom_sender_target_score"].fillna(0)
        joined["prior_mode"] = mode
        joined["transition"] = transition
        joined["sender"] = sender
        joined["receiver"] = receiver
        sender_join_rows.append(joined)
        x = joined["cytobridge_sender_interaction_gene_drift"].to_numpy(dtype=float)
        y = joined["custom_sender_target_score"].to_numpy(dtype=float)
        rho = _safe_spearman(x, y)
        sender_agreement_rows.append(
            {
                "prior_mode": mode,
                "transition": transition,
                "sender": sender,
                "receiver": receiver,
                "n_hvg_or_linked_targets": int(len(joined)),
                "n_nonzero_nichenet_targets": int((y != 0).sum()),
                "spearman_cytobridge_drift_vs_custom_sender_target_score": rho,
                "auc_cytobridge_drift_for_linked_targets": _rank_auc(x, y != 0),
                "permutation_p_one_sided": math.nan,
                "permutation_null_mean": math.nan,
                "permutation_null_sd": math.nan,
                "n_valid_permutations": 0,
                "permutation_note": (
                    "not run per sender edge to avoid pseudoreplicating thousands of genes; "
                    "formal nulls are receiver-program and sender-ranking tests"
                ),
                "nichenet_measure_is_native_edge_strength": False,
                "projection_semantics": pca_info["interpretation"],
            }
        )

    # Pairwise training-seed stability of receiver-level total message vectors.
    receiver_seed_pc = (
        seed_selected.groupby(
            ["training_seed", "transition", "receiver_type"], observed=True, as_index=False
        )[seed_pc]
        .sum()
    )
    receiver_seed_gene = project_drift_table(
        receiver_seed_pc,
        pc_columns=seed_pc,
        pca_info=pca_info,
        value_column="cytobridge_total_interaction_gene_drift",
    )
    seed_stability_rows: list[dict[str, Any]] = []
    for (transition, receiver), group in receiver_seed_gene.groupby(
        ["transition", "receiver_type"], observed=True, sort=True
    ):
        seeds = sorted(group["training_seed"].unique())
        by_seed = {
            int(seed): group[group["training_seed"] == seed].set_index("gene")
            ["cytobridge_total_interaction_gene_drift"]
            for seed in seeds
        }
        for index, seed_a in enumerate(seeds):
            for seed_b in seeds[index + 1 :]:
                aligned = pd.concat([by_seed[int(seed_a)], by_seed[int(seed_b)]], axis=1).dropna()
                seed_stability_rows.append(
                    {
                        "transition": transition,
                        "receiver": receiver,
                        "training_seed_a": int(seed_a),
                        "training_seed_b": int(seed_b),
                        "n_hvg": int(len(aligned)),
                        "spearman_gene_drift": _safe_spearman(
                            aligned.iloc[:, 0], aligned.iloc[:, 1]
                        ),
                    }
                )

    receiver_agreement = pd.DataFrame(agreement_rows)
    if len(receiver_agreement):
        macro_rows: list[dict[str, Any]] = []
        metrics = [
            "spearman_cytobridge_drift_vs_temporal_de",
            "auc_cytobridge_drift_for_selected_up_targets",
            "spearman_cytobridge_drift_vs_nichenet_target_score",
            "spearman_nichenet_target_score_vs_temporal_de",
        ]
        for mode, mode_group in receiver_agreement.groupby(
            "prior_mode", observed=True, sort=True
        ):
            scope_groups: list[tuple[str, str, pd.DataFrame]] = [("overall", "all", mode_group)]
            scope_groups.extend(
                ("transition", str(value), frame)
                for value, frame in mode_group.groupby("transition", observed=True, sort=True)
            )
            scope_groups.extend(
                ("target_selection_mode", str(value), frame)
                for value, frame in mode_group.groupby(
                    "target_selection_mode", observed=True, sort=True
                )
            )
            scope_groups.extend(
                (
                    "transition_x_target_selection_mode",
                    f"{transition}|{selection}",
                    frame,
                )
                for (transition, selection), frame in mode_group.groupby(
                    ["transition", "target_selection_mode"], observed=True, sort=True
                )
            )
            for scope, scope_value, group in scope_groups:
                for metric in metrics:
                    values = _finite_numeric(group[metric]).dropna()
                    null_arrays = [
                        receiver_nulls.get(
                            (str(mode), str(row.transition), str(row.receiver), metric)
                        )
                        for row in group.loc[
                            _finite_numeric(group[metric]).notna()
                        ].itertuples(index=False)
                    ]
                    null_arrays = [array for array in null_arrays if array is not None]
                    if null_arrays and metric != "auc_cytobridge_drift_for_selected_up_targets":
                        null_macro = np.nanmean(np.column_stack(null_arrays), axis=1)
                        finite_null = null_macro[np.isfinite(null_macro)]
                        observed_macro = float(values.mean()) if len(values) else math.nan
                        macro_p = (
                            float(
                                (1 + np.sum(finite_null >= observed_macro))
                                / (1 + len(finite_null))
                            )
                            if len(finite_null) and math.isfinite(observed_macro)
                            else math.nan
                        )
                    else:
                        finite_null = np.asarray([], dtype=float)
                        macro_p = math.nan
                    selection_values = sorted(
                        set(group["target_selection_mode"].dropna().astype(str))
                    )
                    macro_rows.append(
                        {
                            "prior_mode": mode,
                            "scope": scope,
                            "scope_value": scope_value,
                            "target_selection_modes_in_scope": ";".join(selection_values),
                            "metric": metric,
                            "n_receivers": int(len(values)),
                            "receiver_macro_mean": (
                                float(values.mean()) if len(values) else math.nan
                            ),
                            "receiver_macro_sd": (
                                float(values.std(ddof=1)) if len(values) > 1 else math.nan
                            ),
                            "permutation_p_one_sided": macro_p,
                            "permutation_null_mean": (
                                float(finite_null.mean()) if len(finite_null) else math.nan
                            ),
                            "permutation_null_sd": (
                                float(finite_null.std(ddof=1))
                                if len(finite_null) > 1
                                else math.nan
                            ),
                            "n_valid_permutations": int(len(finite_null)),
                        }
                    )
        receiver_macro = pd.DataFrame(macro_rows)
    else:
        receiver_macro = pd.DataFrame()

    return {
        "receiver_gene_scores": (
            pd.concat(receiver_join_rows, ignore_index=True) if receiver_join_rows else pd.DataFrame()
        ),
        "sender_gene_scores": (
            pd.concat(sender_join_rows, ignore_index=True) if sender_join_rows else pd.DataFrame()
        ),
        "receiver_agreement": receiver_agreement,
        "receiver_macro": receiver_macro,
        "sender_agreement": pd.DataFrame(sender_agreement_rows),
        "training_seed_gene_stability": pd.DataFrame(seed_stability_rows),
        "receiver_target_scores": receiver_target,
        "sender_target_scores": sender_target,
    }


def _save_figure(fig: Any, output_stem: Path) -> list[Path]:
    png = output_stem.with_suffix(".png")
    svg = output_stem.with_suffix(".svg")
    fig.savefig(png, dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(svg, bbox_inches="tight", facecolor="white")
    return [png, svg]


def plot_static_figures(
    *,
    output_dir: Path,
    native_ligands: pd.DataFrame,
    support_context: pd.DataFrame,
    support_macro: pd.DataFrame,
    partial_macro: pd.DataFrame,
    overlap: pd.DataFrame,
    gene_results: Mapping[str, pd.DataFrame] | None,
    q_prior_audit: Mapping[str, Any],
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "savefig.dpi": 220,
        }
    )
    colors = {"default": "#0072B2", "matched": "#D55E00"}
    markers = {"default": "o", "matched": "s"}
    outputs: list[Path] = []
    full_q_rows = q_prior_audit.get("full_model_prior_rows", "?")
    supplied_q_rows = q_prior_audit.get("supplied_singleton_prior_rows", "?")
    official_q_rows = q_prior_audit.get("official_overlap_singleton_prior_rows")
    official_q_label = (
        f"Q (official overlap {official_q_rows})"
        if official_q_rows is not None
        else "Q (official overlap)"
    )

    overall = support_macro[support_macro["scope"] == "overall"].copy()
    if len(overall):
        metric_order = (
            "G", "D", "Q_full_model_prior", "Q_supplied_singleton_prior",
            "Q_official_overlap_prior", "Q_context_candidate_intersection",
        )
        metrics = [item for item in metric_order if item in set(overall["cytobridge_metric"])]
        metric_labels = {
            "G": "G",
            "D": "D",
            "Q_full_model_prior": f"Q full\n(n={full_q_rows})",
            "Q_supplied_singleton_prior": f"Q supplied\n(n={supplied_q_rows})",
            "Q_official_overlap_prior": (
                f"Q official overlap\n(n={official_q_rows})"
                if official_q_rows is not None
                else "Q official overlap"
            ),
            "Q_context_candidate_intersection": "Q context\ncandidates",
        }
        modes = sorted(overall["prior_mode"].astype(str).unique())
        fig, ax = plt.subplots(figsize=(9.6, 3.8))
        x = np.arange(len(metrics), dtype=float)
        offsets = np.linspace(-0.14, 0.14, max(len(modes), 1))
        for offset, mode in zip(offsets, modes):
            subset = overall.set_index(["prior_mode", "cytobridge_metric"])
            values, lows, highs = [], [], []
            for metric in metrics:
                row = subset.loc[(mode, metric)] if (mode, metric) in subset.index else None
                value = float(row["macro_spearman_rho"]) if row is not None else math.nan
                values.append(value)
                lows.append(value - float(row["bootstrap_ci_low"]) if row is not None else math.nan)
                highs.append(float(row["bootstrap_ci_high"]) - value if row is not None else math.nan)
            ax.errorbar(
                x + offset,
                values,
                yerr=np.asarray([lows, highs]),
                fmt=markers.get(mode, "o"),
                color=colors.get(mode, "#555555"),
                capsize=3,
                label=mode,
            )
        ax.axhline(0, color="#777777", linewidth=0.8)
        ax.set_xticks(x, [metric_labels[item] for item in metrics])
        ax.set_ylabel("Macro Spearman correlation")
        ax.set_xlabel("CytoBridge diagnostic")
        ax.set_title("Custom NicheNet sender support vs CytoBridge (receiver-context macro)")
        ax.legend(frameon=False, title="NicheNet prior")
        ax.text(
            0.01,
            -0.24,
            "G = learned attention gate; D = exact message magnitude; Q = expression LR compatibility.\n"
            f"Q(full {full_q_rows}), supplied singleton {supplied_q_rows}, official overlap "
            f"{official_q_rows if official_q_rows is not None else 'not evaluated'}, and "
            "context-candidate intersection (exact for matched; overlap-only for default) "
            "are separate controls.\n"
            "NicheNet sender support is a custom decomposition, not a native NicheNet CCC edge.",
            transform=ax.transAxes,
            va="top",
            fontsize=8,
        )
        outputs.extend(_save_figure(fig, output_dir / "figures" / "sender_support_macro"))
        plt.close(fig)

    if len(support_context):
        modes = sorted(support_context["prior_mode"].astype(str).unique())
        metric_order = (
            "G", "D", "Q_full_model_prior", "Q_official_overlap_prior",
            "Q_context_candidate_intersection",
        )
        metrics = [item for item in metric_order if item in set(support_context["cytobridge_metric"])]
        metric_labels = {
            "G": "G",
            "D": "D",
            "Q_full_model_prior": f"Q (full {full_q_rows})",
            "Q_official_overlap_prior": official_q_label,
            "Q_context_candidate_intersection": "Q (context candidates)",
        }
        fig, axes = plt.subplots(
            1, len(metrics), figsize=(4.0 * len(metrics), 3.3), sharey=True, squeeze=False
        )
        for axis, metric in zip(axes[0], metrics):
            for mode_index, mode in enumerate(modes):
                values = support_context[
                    (support_context["cytobridge_metric"] == metric)
                    & (support_context["prior_mode"] == mode)
                ]["spearman_rho"].dropna().to_numpy(dtype=float)
                x = np.full(len(values), mode_index, dtype=float)
                jitter = np.linspace(-0.08, 0.08, len(values)) if len(values) > 1 else np.zeros(len(values))
                axis.scatter(
                    x + jitter,
                    values,
                    s=18,
                    alpha=0.7,
                    color=colors.get(mode, "#555555"),
                    marker=markers.get(mode, "o"),
                    label=mode if metric == metrics[0] else None,
                )
                if len(values):
                    axis.plot(
                        [mode_index - 0.16, mode_index + 0.16],
                        [np.mean(values), np.mean(values)],
                        color="#222222",
                        linewidth=1.5,
                    )
            axis.axhline(0, color="#999999", linewidth=0.7)
            axis.set_xticks(range(len(modes)), modes, rotation=20)
            axis.set_title(metric_labels[metric])
            axis.set_xlabel("NicheNet prior")
        axes[0, 0].set_ylabel("Within-receiver Spearman correlation")
        fig.suptitle("Receiver × transition sender-ranking concordance")
        outputs.extend(_save_figure(fig, output_dir / "figures" / "sender_support_contexts"))
        plt.close(fig)

    partial_overall = partial_macro[partial_macro["scope"] == "overall"].copy()
    if len(partial_overall):
        adjustment_order = [
            "full_model_prior_Q",
            "full_model_prior_Q_plus_abundance",
            "supplied_singleton_prior_Q",
            "supplied_singleton_prior_Q_plus_abundance",
            "official_overlap_prior_Q",
            "official_overlap_prior_Q_plus_abundance",
            "context_candidate_intersection_Q",
            "context_candidate_intersection_Q_plus_abundance",
        ]
        adjustment_labels_map = {
            "full_model_prior_Q": "Full Q",
            "full_model_prior_Q_plus_abundance": "Full Q + N",
            "supplied_singleton_prior_Q": "Supplied Q",
            "supplied_singleton_prior_Q_plus_abundance": "Supplied Q + N",
            "official_overlap_prior_Q": "Official Q",
            "official_overlap_prior_Q_plus_abundance": "Official Q + N",
            "context_candidate_intersection_Q": "Context Q",
            "context_candidate_intersection_Q_plus_abundance": "Context Q + N",
        }
        available_adjustments = set(partial_overall["adjustment"].astype(str))
        adjustment_order = [
            item for item in adjustment_order if item in available_adjustments
        ]
        adjustment_labels = [adjustment_labels_map[item] for item in adjustment_order]
        fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.2), sharey=True)
        for axis, metric in zip(axes, ["G", "D"]):
            subset = partial_overall[partial_overall["cytobridge_metric"] == metric]
            modes = sorted(subset["prior_mode"].astype(str).unique())
            x = np.arange(len(adjustment_order), dtype=float)
            offsets = np.linspace(-0.12, 0.12, max(len(modes), 1))
            for offset, mode in zip(offsets, modes):
                indexed = subset[subset["prior_mode"] == mode].set_index("adjustment")
                values = [
                    float(indexed.loc[item, "macro_partial_spearman_rho"])
                    if item in indexed.index
                    else math.nan
                    for item in adjustment_order
                ]
                axis.scatter(
                    x + offset,
                    values,
                    s=34,
                    marker=markers.get(mode, "o"),
                    color=colors.get(mode, "#555555"),
                    label=mode,
                )
            axis.axhline(0, color="#999999", linewidth=0.7)
            axis.set_xticks(x, adjustment_labels, rotation=25, ha="right")
            axis.set_title(metric)
            axis.set_xlabel("Rank-residual adjustment (N = directed cell-pair count)")
        axes[0].set_ylabel("Macro partial Spearman correlation")
        axes[1].legend(frameon=False, title="NicheNet prior")
        fig.suptitle("Custom sender support vs CytoBridge after expression/abundance control")
        outputs.extend(_save_figure(fig, output_dir / "figures" / "sender_support_partial_Q"))
        plt.close(fig)

    if len(overlap):
        plot_data = overlap[
            overlap["cytobridge_score"].isin(["S_AB_lr_pair", "S_AB_pathway"])
        ].copy()
        plot_data = plot_data[
            (plot_data["n_left_positive_in_common_universe"] >= plot_data["requested_k"])
            & (plot_data["n_right_positive_in_common_universe"] >= plot_data["requested_k"])
        ]
        levels = [item for item in ("ligand", "ligand_receptor", "pathway") if item in set(plot_data["feature_level"])]
        if levels:
            fig, axes = plt.subplots(1, len(levels), figsize=(4.2 * len(levels), 3.4), sharey=True, squeeze=False)
            for axis, level in zip(axes[0], levels):
                subset = plot_data[plot_data["feature_level"] == level]
                macro = (
                    subset.groupby(["prior_mode", "requested_k"], observed=True)["jaccard"]
                    .agg(["mean", "sem", "count"])
                    .reset_index()
                )
                for mode, group in macro.groupby("prior_mode", observed=True, sort=True):
                    axis.errorbar(
                        group["requested_k"], group["mean"], yerr=group["sem"].fillna(0),
                        marker=markers.get(str(mode), "o"), color=colors.get(str(mode), "#555555"),
                        capsize=2, label=str(mode),
                    )
                    for row in group.itertuples(index=False):
                        if int(row.count) <= 7:
                            if float(row.mean) > 0.85:
                                label_offset = -15 if str(mode) == "default" else -27
                            else:
                                label_offset = 8 if str(mode) == "default" else 15
                            axis.annotate(
                                f"n={int(row.count)}",
                                (float(row.requested_k), float(row.mean)),
                                xytext=(0, label_offset),
                                textcoords="offset points",
                                ha="center",
                                fontsize=7,
                                color=colors.get(str(mode), "#555555"),
                            )
                axis.set_title(level.replace("_", " "))
                axis.set_xlabel("Top k")
            axes[0, 0].set_ylim(0, 1.05)
            axes[0, 0].set_ylabel("Macro Jaccard overlap")
            axes[0, -1].legend(frameon=False, title="NicheNet prior")
            fig.suptitle(
                "Ranked overlap with CytoBridge LR-annotated message score\n"
                "Positive-eligible contexts; sparse points with n ≤ 7 annotated",
                fontsize=11,
                y=0.94,
            )
            fig.subplots_adjust(top=0.67, wspace=0.20)
            outputs.extend(_save_figure(fig, output_dir / "figures" / "ranked_feature_overlap"))
            plt.close(fig)

    if len(native_ligands):
        modes = sorted(native_ligands["prior_mode"].astype(str).unique())
        fig, axes = plt.subplots(
            1, len(modes), figsize=(5.0 * len(modes), 4.5), squeeze=False
        )
        for axis, mode in zip(axes[0], modes):
            subset = native_ligands[native_ligands["prior_mode"] == mode].copy()
            summary = (
                subset.groupby("ligand", observed=True)
                .agg(mean_aupr_corrected=("aupr_corrected", "mean"), n_contexts=("receiver", "size"))
                .reset_index()
                .sort_values(["mean_aupr_corrected", "ligand"], ascending=[False, True])
                .head(15)
                .sort_values("mean_aupr_corrected")
            )
            axis.barh(
                summary["ligand"], summary["mean_aupr_corrected"],
                color=colors.get(mode, "#555555"), alpha=0.85,
            )
            axis.set_title(f"{mode} prior")
            axis.set_xlabel("Mean corrected AUPR across represented contexts")
        fig.suptitle("Native NicheNet receiver-program ligand activity")
        outputs.extend(_save_figure(fig, output_dir / "figures" / "native_ligand_activity"))
        plt.close(fig)

    if gene_results is not None and len(gene_results.get("receiver_agreement", pd.DataFrame())):
        data = gene_results["receiver_agreement"]
        fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.5), sharey=True)
        modes = sorted(data["prior_mode"].astype(str).unique())
        selection_markers = {
            "fdr_positive": "o",
            "fallback_ranked_positive": "x",
        }
        # CytoBridge-vs-DE is prior-independent; do not duplicate identical dots
        # under default/matched labels.
        first = data.drop_duplicates(["transition", "receiver"])
        values = first["spearman_cytobridge_drift_vs_temporal_de"].dropna().to_numpy(dtype=float)
        for selection_index, (selection, selection_frame) in enumerate(
            first.groupby("target_selection_mode", observed=True, sort=True)
        ):
            selection_values = selection_frame[
                "spearman_cytobridge_drift_vs_temporal_de"
            ].dropna().to_numpy(dtype=float)
            jitter = (
                np.linspace(-0.08, 0.08, len(selection_values))
                if len(selection_values) > 1
                else np.zeros(len(selection_values))
            )
            axes[0].scatter(
                jitter,
                selection_values,
                s=24,
                alpha=0.8,
                color="#009E73",
                marker=selection_markers.get(str(selection), "D"),
                label=(
                    "FDR-positive program"
                    if selection == "fdr_positive"
                    else "fallback-ranked program"
                ),
            )
        if len(values):
            axes[0].plot([-0.16, 0.16], [values.mean(), values.mean()], color="#222222", linewidth=1.5)
        axes[0].set_xticks([0], ["CytoBridge"])
        axes[0].set_title("CytoBridge drift vs DE")

        metrics = [
            ("spearman_cytobridge_drift_vs_nichenet_target_score", "CytoBridge drift vs NicheNet targets"),
            ("spearman_nichenet_target_score_vs_temporal_de", "NicheNet targets vs DE*"),
        ]
        for axis, (metric, title) in zip(axes[1:], metrics):
            for mode_index, mode in enumerate(modes):
                mode_frame = data[data["prior_mode"] == mode]
                values = mode_frame[metric].dropna().to_numpy(dtype=float)
                for selection, selection_frame in mode_frame.groupby(
                    "target_selection_mode", observed=True, sort=True
                ):
                    selection_values = selection_frame[metric].dropna().to_numpy(dtype=float)
                    jitter = (
                        np.linspace(-0.08, 0.08, len(selection_values))
                        if len(selection_values) > 1
                        else np.zeros(len(selection_values))
                    )
                    axis.scatter(
                        np.full(len(selection_values), mode_index) + jitter,
                        selection_values,
                        s=22,
                        alpha=0.75,
                        color=colors.get(mode, "#555555"),
                        marker=selection_markers.get(str(selection), "D"),
                    )
                if len(values):
                    axis.plot(
                        [mode_index - 0.16, mode_index + 0.16],
                        [values.mean(), values.mean()], color="#222222", linewidth=1.5,
                    )
            axis.axhline(0, color="#999999", linewidth=0.7)
            axis.set_xticks(range(len(modes)), modes, rotation=20)
            axis.set_title(title)
            axis.set_xlabel("NicheNet prior")
        axes[0].axhline(0, color="#999999", linewidth=0.7)
        axes[0].set_xlabel("Method")
        axes[0].set_ylabel("Receiver-context Spearman correlation")
        axes[0].legend(frameon=False, fontsize=8, loc="upper right")
        fig.suptitle("Temporal response agreement in the retained 50-PC gene subspace")
        fig.text(
            0.5,
            -0.02,
            "*NicheNet ligand activities/target links are fitted to the DE-defined receiver program; "
            "this panel is descriptive, not independent validation.",
            ha="center",
            fontsize=8,
        )
        outputs.extend(_save_figure(fig, output_dir / "figures" / "temporal_gene_program_agreement"))
        plt.close(fig)

    return outputs


def _write_methods_note(
    path: Path,
    *,
    gene_status: Mapping[str, Any],
    prior_modes: Sequence[str],
    permutations: int,
    bootstrap_replicates: int,
    target_selection_audit: pd.DataFrame,
    q_prior_audit: Mapping[str, Any],
    context_candidate_q_audit: pd.DataFrame,
) -> None:
    projection = gene_status.get("interpretation", gene_status.get("reason", "not evaluated"))
    if len(target_selection_audit):
        selection_counts = target_selection_audit["target_selection_mode"].value_counts()
        selection_summary = ", ".join(
            f"{mode}: {int(count)} contexts" for mode, count in selection_counts.items()
        )
    else:
        selection_summary = "selection-mode audit unavailable"
    full_q_rows = int(q_prior_audit.get("full_model_prior_rows", 0))
    singleton_q_rows = int(q_prior_audit.get("supplied_singleton_prior_rows", 0))
    official_q_value = q_prior_audit.get("official_overlap_singleton_prior_rows")
    official_q_rows = int(official_q_value) if official_q_value is not None else None
    complex_q_rows = int(q_prior_audit.get("excluded_complex_rows", 0))
    if len(context_candidate_q_audit):
        context_candidate_summary = "; ".join(
            (
                f"{row.prior_mode}: {int(row.n_unique_context_candidate_lr_in_cytobridge_prior)}"
                f"/{int(row.n_unique_context_candidate_lr)} unique context candidates map to "
                "the CytoBridge prior"
            )
            for row in context_candidate_q_audit.itertuples(index=False)
        )
    else:
        context_candidate_summary = "context-candidate audit unavailable"
    official_overlap_summary = (
        f"an audited {official_q_rows}-row intersection with the frozen official NicheNet LR network"
        if official_q_rows is not None
        else "not evaluated because no audited official-overlap LR table was supplied"
    )
    text = f"""# CytoBridge–temporal NicheNet method comparison and internal concordance

## What is compared

NicheNet's native output is a receiver-context ligand activity: it asks which ligands best
predict an observed temporal receiver target program. It is **not** a native global
sender-to-receiver CCC edge strength. Native ligand activities and official ligand-to-target
links are therefore summarized on their own terms.

The sender-level comparison uses the runner's explicitly custom sender-support decomposition:
positive corrected ligand AUPR × sender expression share × receiver receptor support. It is
compared with CytoBridge G (learned edge attention gate), D (magnitude of the exact signed
interaction message), and Q (post-hoc LR expression compatibility) only within the same
receiver and temporal transition. Receiver-context Spearman correlations are macro-averaged;
the null shuffles sender labels within each receiver/transition ({permutations} replicates).
Context bootstrap confidence intervals use {bootstrap_replicates} replicates.

Because custom sender support contains expression/receptor support, sensitivity tables also
report partial Spearman correlations after rank-residualizing both support and G/D on Q, and
after additionally controlling the directed observed cell-pair count (an abundance/type-
propensity proxy within receiver). Four Q definitions are kept separate. `full_model_prior_Q`
sums the complete {full_q_rows}-row model prior (including {complex_q_rows} complex rows).
`supplied_singleton_prior_Q` sums all {singleton_q_rows} singleton rows supplied to the matched
runner; it is a supplied-database sensitivity and is **not** called an exact effective NicheNet
prior. `official_overlap_prior_Q` uses {official_overlap_summary}.
`context_candidate_intersection_Q` sums Q only for LR pairs exported as candidates for the same
transition/sender/receiver ({context_candidate_summary}). This last control is exact for matched
contexts only when every exported candidate maps to the model prior; for default NicheNet it is
explicitly an intersection, not a complete official-prior Q. Constant all-zero contexts are
undefined, not zero-filled.

## Database modes

Analyzed NicheNet prior modes: {', '.join(prior_modes)}. The default run uses the official
mouse v2 NicheNet LR network. The matched run, when present, retains the same official
ligand-target matrix but supplies the frozen singleton CellChat LR table as its candidate prior.
Both are reported; neither is selected after seeing the result.

The official ligand-target matrix and receiver target/background are unchanged between modes.
Consequently, native activity values are mathematically identical for ligands present in both
runs; matched mode changes only the potential-ligand/LR candidate universe. The cross-prior
audit exports this equality explicitly, so stronger matched sender concordance is not presented
as an independent second ligand-activity fit.

Ligand overlap uses native receiver-context activity. LR overlap uses the explicitly custom
sender-specific NicheNet LR support. Pathway overlap maps exact singleton LR pairs post hoc to
the CellChat pathway labels carried by the CytoBridge prior; it is not a native NicheNet
pathway result.

Derived target-score tables are truncated summaries of the runner export (at most the configured
top 30 ligands and top 250 targets per ligand), not the full ligand-target matrix. Gene-label
permutation p-values are calibration references under an exchangeability null; correlated genes
make them unsuitable as standalone biological significance claims.

Top-k overlap uses strictly positive scores and includes every feature tied at the kth score;
lexical tie-breaking and zero/negative padding are prohibited. Macro top-k summaries include
only contexts with at least k positive features on both sides and report the eligible
denominator.

## Temporal target-program audit

{selection_summary}. `fdr_positive` contexts are FDR-supported positive response programs.
`fallback_ranked_positive` contexts were used only because too few FDR-positive genes were
available; they are ranked positive-effect sensitivity programs and are never described as
significant DE gene sets. All receiver and sender comparisons are exported both overall and
stratified by this selection mode. NicheNet target-vs-DE agreement is descriptive because the
DE-defined program is the NicheNet input, not an independent validation.

## Signed gene-space analysis

Status: **{gene_status.get('status', 'unknown')}**.

{projection}

Scalar D is never converted into gene scores. Gene-level results are emitted only when the
formal signed `drift_pc_1...drift_pc_50` vectors and a provenance-compatible PCA loading
artifact are both present. PCA inverse projection is exact for the exported vector inside the
retained 50-PC log1p-normalized HVG subspace, but PCA-discarded expression components cannot
be recovered. Agreement with observed temporal DE is associative, not causal evidence that a
specific ligand produced the response.
"""
    path.write_text(text)


def _artifact_index(output_dir: Path, excluded: set[Path] | None = None) -> pd.DataFrame:
    excluded = {path.resolve() for path in (excluded or set())}
    rows = []
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or path.resolve() in excluded:
            continue
        rows.append(
            {
                "relative_path": str(path.relative_to(output_dir)),
                "bytes": int(path.stat().st_size),
                "sha256": _sha256_file(path),
            }
        )
    return pd.DataFrame(rows)


def run_analysis(args: argparse.Namespace) -> dict[str, Any]:
    if args.permutations < 0 or args.bootstrap_replicates < 0:
        raise ValueError("Permutation and bootstrap replicate counts must be non-negative.")
    top_k = _parse_top_k(args.top_k)
    output_dir = _prepare_output(args.output_dir, args.overwrite)

    learned_manifest, learned, learned_paths = load_learned(args.learned_dir)
    default_manifest, default_tables, default_paths = load_nichenet_run(
        args.nichenet_default_dir, "default"
    )
    run_manifests = {"default": default_manifest}
    run_paths = {"default": default_paths}
    runs = [default_tables]
    if args.nichenet_matched_dir is not None:
        matched_manifest, matched_tables, matched_paths = load_nichenet_run(
            args.nichenet_matched_dir, "matched"
        )
        run_manifests["matched"] = matched_manifest
        run_paths["matched"] = matched_paths
        runs.append(matched_tables)
    modes = [str(tables["support"]["prior_mode"].iloc[0]) for tables in runs if len(tables["support"])]
    if not modes:
        modes = list(run_manifests)
    combined: dict[str, pd.DataFrame] = {}
    for key in NICHENET_REQUIRED:
        if key == "manifest":
            continue
        combined[key] = pd.concat([tables[key] for tables in runs], ignore_index=True)

    receiver_de: pd.DataFrame | None = None
    target_selection_audit = pd.DataFrame()
    if args.temporal_input_dir is not None:
        de_path = args.temporal_input_dir.expanduser().resolve() / "receiver_de_genes.csv"
        if not de_path.is_file():
            raise FileNotFoundError(f"Temporal input is missing receiver_de_genes.csv: {de_path}")
        receiver_de = pd.read_csv(de_path)
        target_selection_audit = build_target_selection_audit(receiver_de)

    official_overlap_path = getattr(args, "nichenet_official_overlap_lr", None)
    official_overlap_lr: pd.DataFrame | None = None
    if official_overlap_path is not None:
        official_overlap_path = official_overlap_path.expanduser().resolve()
        if not official_overlap_path.is_file():
            raise FileNotFoundError(
                f"Audited NicheNet official-overlap LR table does not exist: {official_overlap_path}"
            )
        official_overlap_lr = pd.read_csv(official_overlap_path)

    support_join = build_sender_support_join(combined["support"], learned["heatmap"])
    singleton_q, singleton_q_audit = aggregate_singleton_prior_q(
        learned["lr"], official_overlap_lr
    )
    if args.nichenet_matched_dir is not None:
        matched_prior = runs[-1]["prior_coverage"]
        if "n_active_edges_in_official_lr" in matched_prior and len(matched_prior):
            reported_official_overlap = int(
                _finite_numeric(matched_prior["n_active_edges_in_official_lr"]).iloc[0]
            )
            singleton_q_audit["nichenet_reported_official_overlap_rows"] = (
                reported_official_overlap
            )
            observed_official_overlap = singleton_q_audit.get(
                "official_overlap_singleton_prior_rows"
            )
            singleton_q_audit["official_overlap_count_matches_nichenet_report"] = (
                observed_official_overlap is not None
                and int(observed_official_overlap) == reported_official_overlap
            )
            if (
                observed_official_overlap is not None
                and int(observed_official_overlap) != reported_official_overlap
            ):
                raise ValueError(
                    "Audited official-overlap LR table count does not match the formal "
                    f"NicheNet prior coverage ({observed_official_overlap} vs "
                    f"{reported_official_overlap})."
                )
    if official_overlap_path is not None:
        matched_input_hashes = (
            run_manifests.get("matched", {}).get("inputs", {}).get("sha256", {})
        )
        singleton_q_audit["official_overlap_table_provenance"] = {
            "path": str(official_overlap_path),
            "sha256": _sha256_file(official_overlap_path),
            "is_original_r_runner_output": False,
            "provenance_class": "post-R-formal deterministic crosswalk",
            "derivation": (
                "Apply the R runner normalize_lr semantics (trim character ligand/receptor, "
                "drop missing, exact case-sensitive pair deduplication) to the supplied matched "
                "LR table and frozen official NicheNet LR RDS; retain their exact "
                "(ligand,receptor) inner join; sort by ligand/receptor; assert row count equals "
                "n_active_edges_in_official_lr from formal nichenet_prior_coverage.csv."
            ),
            "matched_lr_sha256": matched_input_hashes.get("matched_lr_sha256"),
            "official_lr_network_sha256": matched_input_hashes.get(
                "official_lr_network_sha256"
            ),
            "asserted_rows": singleton_q_audit.get(
                "official_overlap_singleton_prior_rows"
            ),
            "formal_reported_rows": singleton_q_audit.get(
                "nichenet_reported_official_overlap_rows"
            ),
            "formal_count_assertion_passed": singleton_q_audit.get(
                "official_overlap_count_matches_nichenet_report"
            ),
            "original_r_output_file_sha256_includes_this_table": False,
        }
    context_candidate_q, context_candidate_q_audit = aggregate_context_candidate_prior_q(
        combined["candidates"], learned["lr"]
    )
    support_join = support_join.merge(
        singleton_q,
        on=["time", "sender_type", "receiver_type"],
        how="left",
        validate="many_to_one",
    )
    support_join = support_join.merge(
        context_candidate_q,
        on=[
            "prior_mode", "transition", "time", "sender_type", "receiver_type"
        ],
        how="left",
        validate="one_to_one",
    )
    if support_join["Q_AB_supplied_singleton_prior"].isna().any():
        raise ValueError("Could not align supplied-singleton Q with NicheNet sender contexts.")
    if support_join["Q_AB_nichenet_context_candidate_intersection"].isna().any():
        raise ValueError("Could not align context-candidate Q with NicheNet sender contexts.")
    if len(target_selection_audit):
        selection_map = target_selection_audit[
            [
                "transition", "receiver", "target_selection_mode",
                "program_evidence_class",
            ]
        ].rename(columns={"receiver": "receiver_type"})
        support_join = support_join.merge(
            selection_map,
            on=["transition", "receiver_type"],
            how="left",
            validate="many_to_one",
        )
        if support_join["target_selection_mode"].isna().any():
            raise ValueError("NicheNet sender-support contexts are missing DE selection-mode audit rows.")
    support_context = sender_support_context_correlations(support_join)
    support_macro, support_null = sender_support_macro_statistics(
        support_join,
        support_context,
        permutations=args.permutations,
        bootstrap_replicates=args.bootstrap_replicates,
        random_seed=args.random_seed,
    )
    partial_context, partial_nulls = sender_support_partial_correlations(
        support_join,
        permutations=args.permutations,
        random_seed=args.random_seed + 17,
    )
    partial_macro = partial_correlation_macro_statistics(partial_context, partial_nulls)
    support_seed, support_seed_summary = sender_support_training_seed_stability(
        combined["support"], learned["drift_seed"]
    )
    native_ligands = build_native_ligand_summary(
        combined["activities"], combined["links"], top_n=max(top_k)
    )
    native_cross_prior = cross_prior_native_activity_audit(combined["activities"])
    receiver_targets, sender_targets = build_nichenet_target_scores(
        combined["activities"], combined["links"], combined["components"]
    )
    overlap, overlap_membership, mapped_pathways = compute_ranked_overlaps(
        combined,
        learned["lr"],
        learned["pathway"],
        top_k=top_k,
    )
    overlap_macro = summarize_ranked_overlaps(overlap)
    database_coverage, context_coverage = summarize_database_coverage(runs, learned["lr"])

    tables_to_write: dict[str, pd.DataFrame] = {
        "nichenet_native_ligand_activity_summary.csv": native_ligands,
        "nichenet_native_activity_cross_prior_audit.csv": native_cross_prior,
        "nichenet_derived_receiver_target_scores.csv": receiver_targets,
        "nichenet_custom_sender_target_scores.csv": sender_targets,
        "custom_sender_support_joined_G_D_Q.csv": support_join,
        "custom_sender_support_context_correlations.csv": support_context,
        "custom_sender_support_macro_statistics.csv": support_macro,
        "custom_sender_support_permutation_null.csv": support_null,
        "custom_sender_support_partial_Q_context.csv": partial_context,
        "custom_sender_support_partial_Q_macro.csv": partial_macro,
        "custom_sender_support_by_training_seed.csv": support_seed,
        "custom_sender_support_training_seed_summary.csv": support_seed_summary,
        "ranked_ligand_lr_pathway_overlap.csv": overlap,
        "ranked_ligand_lr_pathway_overlap_macro.csv": overlap_macro,
        "ranked_overlap_membership.csv": overlap_membership,
        "nichenet_custom_support_mapped_to_cellchat_pathways.csv": mapped_pathways,
        "database_coverage.csv": database_coverage,
        "nichenet_context_coverage.csv": context_coverage,
        "temporal_target_selection_audit.csv": target_selection_audit,
        "nichenet_context_candidate_Q_audit.csv": context_candidate_q_audit,
        "q_prior_sensitivity_universe.csv": pd.DataFrame(
            [
                {
                    "q_universe": "full_model_prior",
                    "n_lr_rows": singleton_q_audit.get("full_model_prior_rows"),
                    "interpretation": "complete CytoBridge post-hoc LR prior",
                },
                {
                    "q_universe": "supplied_singleton_prior",
                    "n_lr_rows": singleton_q_audit.get("supplied_singleton_prior_rows"),
                    "interpretation": (
                        "all singleton rows supplied to matched NicheNet; not the exact "
                        "effective context-candidate set"
                    ),
                },
                {
                    "q_universe": "official_overlap_singleton_prior",
                    "n_lr_rows": singleton_q_audit.get(
                        "official_overlap_singleton_prior_rows"
                    ),
                    "interpretation": (
                        "audited supplied-singleton intersection with frozen official "
                        "NicheNet LR network"
                    ),
                },
            ]
        ),
    }
    if official_overlap_lr is not None:
        tables_to_write["audited_nichenet_matched_official_overlap_lr.csv"] = (
            official_overlap_lr[["ligand", "receptor"]]
            .drop_duplicates()
            .sort_values(["ligand", "receptor"], kind="mergesort")
            .reset_index(drop=True)
        )

    gene_status: dict[str, Any]
    gene_results: dict[str, pd.DataFrame] | None = None
    if args.pca_artifacts is None:
        gene_status = {
            "status": "skipped",
            "reason": (
                "No PCA artifact supplied. Although exact signed 50-PC vectors are present, "
                "gene coordinates cannot be reconstructed from scalar D or PC labels alone."
            ),
            "scalar_D_expanded_to_gene_space": False,
        }
    elif args.temporal_input_dir is None:
        gene_status = {
            "status": "skipped",
            "reason": (
                "A PCA artifact was supplied, but --temporal-input-dir was omitted; receiver DE "
                "and target-background provenance are required for temporal gene-program tests."
            ),
            "scalar_D_expanded_to_gene_space": False,
        }
    else:
        summary_pc = _pc_columns(learned["drift_summary"], suffix="_mean")
        if not summary_pc:
            raise ValueError(
                "Formal artifacts contain no signed drift_pc_*_mean vectors; refusing to infer "
                "gene direction from scalar D."
            )
        pca_info = audit_pca_artifacts(
            args.pca_artifacts,
            expected_sha256=args.expected_pca_sha256,
            expected_pc_count=len(summary_pc),
            pca_manifest_path=args.pca_manifest,
            learned_manifest=learned_manifest,
        )
        if receiver_de is None:
            raise RuntimeError("Internal error: receiver DE was not loaded.")
        gene_results = compute_gene_space_analyses(
            learned_summary=learned["drift_summary"],
            learned_by_seed=learned["drift_seed"],
            nichenet_tables=combined,
            receiver_de=receiver_de,
            pca_info=pca_info,
            permutations=args.permutations,
            random_seed=args.random_seed + 101,
        )
        mapping = {
            "gene_receiver_scores.csv": "receiver_gene_scores",
            "gene_sender_scores.csv": "sender_gene_scores",
            "gene_receiver_temporal_agreement.csv": "receiver_agreement",
            "gene_receiver_macro_agreement.csv": "receiver_macro",
            "gene_sender_nichenet_agreement.csv": "sender_agreement",
            "gene_drift_training_seed_stability.csv": "training_seed_gene_stability",
        }
        tables_to_write.update({filename: gene_results[key] for filename, key in mapping.items()})
        gene_status = {
            "status": "completed",
            "scalar_D_expanded_to_gene_space": False,
            "signed_message_pc_vectors_used": True,
            "interpretation": pca_info["interpretation"],
            "pca_artifact": {
                key: _json_value(value)
                for key, value in pca_info.items()
                if key not in {"genes", "highly_variable", "loadings"}
            },
        }

    for filename, table in tables_to_write.items():
        table.to_csv(output_dir / filename, index=False)
    (output_dir / "gene_space_status.json").write_text(
        json.dumps(_json_value(gene_status), indent=2, sort_keys=True) + "\n"
    )
    _write_methods_note(
        output_dir / "METHODS_AND_INTERPRETATION.md",
        gene_status=gene_status,
        prior_modes=modes,
        permutations=args.permutations,
        bootstrap_replicates=args.bootstrap_replicates,
        target_selection_audit=target_selection_audit,
        q_prior_audit=singleton_q_audit,
        context_candidate_q_audit=context_candidate_q_audit,
    )
    figure_paths: list[Path] = []
    if not args.skip_figures:
        figure_paths = plot_static_figures(
            output_dir=output_dir,
            native_ligands=native_ligands,
            support_context=support_context,
            support_macro=support_macro,
            partial_macro=partial_macro,
            overlap=overlap,
            gene_results=gene_results,
            q_prior_audit=singleton_q_audit,
        )

    input_files = list(learned_paths.values())
    for paths in run_paths.values():
        input_files.extend(paths.values())
    if args.temporal_input_dir is not None:
        temporal_manifest_candidates = [
            args.temporal_input_dir / "input_manifest.json",
            args.temporal_input_dir / "manifest.json",
        ]
        input_files.extend(path for path in temporal_manifest_candidates if path.is_file())
        de_path = args.temporal_input_dir / "receiver_de_genes.csv"
        if de_path.is_file():
            input_files.append(de_path)
    if args.pca_artifacts is not None:
        input_files.append(args.pca_artifacts)
    if args.pca_manifest is not None:
        input_files.append(args.pca_manifest)
    if official_overlap_path is not None:
        input_files.append(official_overlap_path)
    input_hashes = {
        str(path.expanduser().resolve()): _sha256_file(path.expanduser().resolve())
        for path in sorted(set(input_files), key=lambda item: str(item))
    }

    manifest_path = output_dir / "analysis_manifest.json"
    artifact_index_path = output_dir / "artifact_sha256.csv"
    artifact_index = _artifact_index(output_dir, excluded={manifest_path, artifact_index_path})
    artifact_index.to_csv(artifact_index_path, index=False)
    selection_mode_counts = (
        target_selection_audit["target_selection_mode"].value_counts().to_dict()
        if len(target_selection_audit)
        else {}
    )
    native_difference_columns = [
        column for column in native_cross_prior if column.endswith("_absolute_difference")
    ]
    native_cross_prior_max_difference = {
        column: float(_finite_numeric(native_cross_prior[column]).max())
        for column in native_difference_columns
    }
    partial_reason_counts = (
        partial_context["undefined_reason"]
        .replace("", "defined")
        .fillna("undefined_unspecified")
        .value_counts()
        .to_dict()
    )
    manifest = {
        "schema_version": SCRIPT_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "temporal_nichenet_method_comparison_with_cytobridge_interaction_messages",
        "prior_modes": modes,
        "parameters": {
            "top_k": list(top_k),
            "permutations": int(args.permutations),
            "bootstrap_replicates": int(args.bootstrap_replicates),
            "random_seed": int(args.random_seed),
            "figures_generated": not bool(args.skip_figures),
        },
        "scientific_guardrails": {
            "nichenet_activity_is_global_ccc_edge_strength": False,
            "custom_sender_support_is_native_nichenet_edge_strength": False,
            "comparisons_are_within_same_receiver_and_transition": True,
            "spring_coordinates_used": False,
            "clone_or_fate_outcomes_used": False,
            "scalar_D_expanded_to_gene_space": False,
            "full_model_Q_is_exact_matched_singleton_Q": False,
            "supplied_singleton_Q_is_exact_official_overlap_Q": False,
            "supplied_singleton_Q_is_exact_context_candidate_Q": False,
            "q_prior_audit": singleton_q_audit,
            "context_candidate_q_audit": context_candidate_q_audit.to_dict(
                orient="records"
            ),
            "partial_spearman_adjustments": sorted(
                partial_context["adjustment"].dropna().astype(str).unique().tolist()
            ),
            "all_zero_sender_support_is_undefined": True,
            "partial_spearman_status_counts": partial_reason_counts,
            "top_k_policy": (
                "strictly positive scores only; kth-score ties included; macro denominator "
                "requires >=k positive features on both sides"
            ),
            "gene_permutation_interpretation": (
                "exchangeability calibration reference, not standalone biological significance"
            ),
            "derived_target_score_truncation": (
                "runner-exported top ligands and targets only; formal configuration top30 x top250"
            ),
            "target_selection_mode_counts": selection_mode_counts,
            "fallback_programs_are_significant_DE_sets": False,
            "native_common_ligand_activity_max_absolute_difference": (
                native_cross_prior_max_difference
            ),
            "matched_prior_changes": (
                "potential ligand/LR candidate universe only; official ligand-target matrix and "
                "receiver target/background unchanged"
            ),
            "pathway_mapping": (
                "posthoc exact singleton LR mapping into CytoBridge CellChat pathway labels"
            ),
        },
        "gene_space": gene_status,
        "counts": {
            "sender_support_rows": int(len(support_join)),
            "receiver_transition_contexts": int(
                support_context[["prior_mode", "transition", "receiver"]].drop_duplicates().shape[0]
            ),
            "native_ligand_summary_rows": int(len(native_ligands)),
            "ranked_overlap_rows": int(len(overlap)),
            "ranked_overlap_macro_rows": int(len(overlap_macro)),
            "partial_Q_context_rows": int(len(partial_context)),
            "target_selection_contexts": int(len(target_selection_audit)),
            "native_cross_prior_common_ligands": int(len(native_cross_prior)),
            "figures": int(len(figure_paths)),
        },
        "input_sha256": input_hashes,
        "upstream_manifests": {
            "cytobridge": learned_manifest,
            "nichenet": run_manifests,
        },
        "artifact_index": str(artifact_index_path),
        "artifact_index_sha256": _sha256_file(artifact_index_path),
    }
    manifest_path.write_text(
        json.dumps(_json_value(manifest), indent=2, sort_keys=True) + "\n"
    )
    return manifest


def main() -> None:
    args = build_parser().parse_args()
    manifest = run_analysis(args)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir.expanduser().resolve()),
                "prior_modes": manifest["prior_modes"],
                "gene_space_status": manifest["gene_space"]["status"],
                "sender_support_rows": manifest["counts"]["sender_support_rows"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
