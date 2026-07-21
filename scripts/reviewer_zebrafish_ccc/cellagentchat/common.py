"""Shared, dependency-light contracts for the CellAgentChat workflow."""

from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import shutil
import subprocess
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import sparse


PINNED_CELLAGENTCHAT_COMMIT = "310cfc03df91c5ec917f110801e0c2ae4ab57800"
OFFICIAL_DATABASE_LABEL = "official_mouse_default_celltalkdb"
CUSTOM_DATABASE_LABEL = "cytobridge_zebrafish_lr_projected_singletons"
CONDITION_LABELS = (OFFICIAL_DATABASE_LABEL, CUSTOM_DATABASE_LABEL)


def csv_ints(value: str) -> tuple[int, ...]:
    parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not parsed or len(parsed) != len(set(parsed)):
        raise argparse.ArgumentTypeError("Expected unique comma-separated integers.")
    return parsed


def csv_strings(value: str) -> tuple[str, ...]:
    parsed = tuple(item.strip() for item in value.split(",") if item.strip())
    if not parsed or len(parsed) != len(set(parsed)):
        raise argparse.ArgumentTypeError("Expected unique comma-separated values.")
    return parsed


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
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


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(json_value(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def prepare_output(path: Path, overwrite: bool) -> Path:
    path = path.expanduser().resolve()
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory is not empty: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_table(path: Path, separator: str | None = None) -> pd.DataFrame:
    if separator in (None, "", "auto"):
        separator = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    return pd.read_csv(path, sep=separator)


def git_head(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _function_parameters(tree: ast.Module) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            positional = [*node.args.posonlyargs, *node.args.args]
            result[node.name] = {argument.arg for argument in positional}
    return result


def inspect_official_source_api(source: Path) -> dict[str, Any]:
    """Validate the small official API surface used by the adapter via AST.

    AST inspection keeps preflight independent of Mesa, torch-sparse, and the
    other optional runtime dependencies.  Runtime import is performed only
    after this contract passes.
    """

    source = source.expanduser().resolve()
    src = source / "src"
    required_functions: dict[str, dict[str, set[str]]] = {
        "model_setup.py": {
            "load_db": {"adata", "file", "sep"},
            "load_tf_db": {"species", "adata", "rec_uni"},
            "train": {"adata", "lig_uni", "rec_uni", "tf_uni", "rec_tf_uni", "lr_pairs"},
            "load_model": {"path", "device"},
            "feature_selection": {"model", "mat", "C", "rec_uni"},
            "add_rates": {"conversion_rates", "rec_uni"},
        },
        "permutations.py": {
            "permutation_test": {"threshold", "N", "adata", "lig_uni", "rec_uni", "rates", "dist"},
        },
        "bckground_distribution.py": {
            "get_distribution": {"fin", "dist", "scaled"},
            "get_significant_lr_pairs": {"lr1", "fin", "cutoff"},
        },
        "preprocessor.py": {
            "setup_adata": {"adata_or_path", "coordinates_key", "cell_type_label"},
        },
    }
    records: dict[str, Any] = {}
    for filename, expected in required_functions.items():
        path = src / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing CellAgentChat source file: {path}")
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        functions = _function_parameters(tree)
        for function_name, required_parameters in expected.items():
            observed = functions.get(function_name)
            if observed is None:
                raise RuntimeError(f"{filename} lacks required function {function_name}.")
            missing = sorted(required_parameters.difference(observed))
            if missing:
                raise RuntimeError(
                    f"{filename}:{function_name} lacks required parameters: {missing}."
                )
        records[filename] = artifact(path)

    abm_path = src / "abm.py"
    if not abm_path.is_file():
        raise FileNotFoundError(abm_path)
    abm_tree = ast.parse(abm_path.read_text(encoding="utf-8"), filename=str(abm_path))
    cell_model = next(
        (
            node
            for node in abm_tree.body
            if isinstance(node, ast.ClassDef) and node.name == "CellModel"
        ),
        None,
    )
    if cell_model is None:
        raise RuntimeError("abm.py lacks required CellModel class.")
    methods = {
        node.name
        for node in cell_model.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    if not {"__init__", "step"}.issubset(methods):
        raise RuntimeError("CellModel must provide __init__ and step.")
    records["abm.py"] = artifact(abm_path)

    database_paths = {
        "mouse_lr_pair.tsv": src / "cellagentchat_data" / "mouse_lr_pair.tsv",
        "TF_TG_mouse.csv": src / "cellagentchat_data" / "databases" / "TF_TG_mouse.csv",
        "KEGG_mouse.csv": src / "cellagentchat_data" / "databases" / "KEGG_mouse.csv",
        "REACTOME_mouse.csv": src / "cellagentchat_data" / "databases" / "REACTOME_mouse.csv",
    }
    for label, path in database_paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        records[label] = artifact(path)
    return {"source": str(source), "files": records}


def validate_official_source(
    source: Path,
    *,
    allow_unpinned: bool = False,
) -> dict[str, Any]:
    source = source.expanduser().resolve()
    if not (source / ".git").exists():
        raise FileNotFoundError(f"CellAgentChat checkout is not a git repository: {source}")
    observed = git_head(source)
    if observed != PINNED_CELLAGENTCHAT_COMMIT and not allow_unpinned:
        raise RuntimeError(
            "CellAgentChat source mismatch: "
            f"expected {PINNED_CELLAGENTCHAT_COMMIT}, observed {observed}."
        )
    record = inspect_official_source_api(source)
    record.update(
        {
            "repository": "https://github.com/mcgilldinglab/CellAgentChat",
            "release": "v0.2.0",
            "expected_commit": PINNED_CELLAGENTCHAT_COMMIT,
            "observed_commit": observed,
            "pinned_source_verified": observed == PINNED_CELLAGENTCHAT_COMMIT,
        }
    )
    return record


def _string_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        raise KeyError(f"Missing required column {column!r}.")
    result = frame[column].fillna("").astype(str).str.strip()
    return result.mask(result.str.lower().eq("nan"), "")


def select_orthology_mapping(
    frame: pd.DataFrame,
    *,
    source_column: str,
    target_column: str,
    mapping_policy: str = "strict_one_to_one",
    orthology_type_column: str | None = "orthology_type",
    allowed_orthology_types: Iterable[str] = ("ortholog_one2one",),
    confidence_column: str | None = "orthology_confidence",
    minimum_confidence: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if mapping_policy not in {"strict_one_to_one", "many_to_one_sum"}:
        raise ValueError(f"Unsupported mapping policy: {mapping_policy}")
    work = frame.copy().reset_index(drop=False).rename(columns={"index": "source_row"})
    work["source_gene"] = _string_series(work, source_column)
    work["target_gene"] = _string_series(work, target_column)
    reasons = pd.Series("", index=work.index, dtype=object)

    def mark(mask: pd.Series | np.ndarray, reason: str) -> None:
        nonlocal reasons
        mask = pd.Series(mask, index=work.index).astype(bool)
        reasons.loc[mask & reasons.eq("")] = reason

    mark(work["source_gene"].eq(""), "missing_source_gene")
    mark(work["target_gene"].eq(""), "missing_target_gene")
    if orthology_type_column:
        allowed = {str(value) for value in allowed_orthology_types}
        observed_type = _string_series(work, orthology_type_column)
        mark(~observed_type.isin(allowed), "orthology_type_not_allowed")
    if confidence_column:
        if confidence_column not in work:
            raise KeyError(f"Missing required column {confidence_column!r}.")
        confidence = pd.to_numeric(work[confidence_column], errors="coerce")
        mark(confidence.isna() | confidence.lt(float(minimum_confidence)), "low_or_missing_confidence")

    candidate = reasons.eq("")
    duplicate = work.loc[candidate].duplicated(["source_gene", "target_gene"], keep="first")
    mark(duplicate.reindex(work.index, fill_value=False), "duplicate_mapping_row")

    candidate = reasons.eq("")
    source_target_counts = (
        work.loc[candidate].groupby("source_gene", sort=False)["target_gene"].nunique()
    )
    ambiguous_sources = set(source_target_counts[source_target_counts.ne(1)].index)
    mark(candidate & work["source_gene"].isin(ambiguous_sources), "source_maps_to_multiple_targets")

    if mapping_policy == "strict_one_to_one":
        candidate = reasons.eq("")
        target_source_counts = (
            work.loc[candidate].groupby("target_gene", sort=False)["source_gene"].nunique()
        )
        nonunique_targets = set(target_source_counts[target_source_counts.ne(1)].index)
        mark(candidate & work["target_gene"].isin(nonunique_targets), "target_has_multiple_sources")

    work["exclusion_reason"] = reasons
    used = work.loc[reasons.eq("")].copy()
    excluded = work.loc[~reasons.eq("")].copy()
    if used["source_gene"].duplicated().any():
        raise RuntimeError("Selected orthology map does not have unique source genes.")
    if mapping_policy == "strict_one_to_one" and used["target_gene"].duplicated().any():
        raise RuntimeError("Strict orthology map does not have unique target genes.")
    used = used.sort_values(["target_gene", "source_gene"], kind="mergesort").reset_index(drop=True)
    excluded = excluded.sort_values("source_row", kind="mergesort").reset_index(drop=True)
    counts = {
        "input_rows": int(len(work)),
        "selected_rows": int(len(used)),
        "selected_source_genes": int(used["source_gene"].nunique()),
        "selected_target_genes": int(used["target_gene"].nunique()),
        "excluded_rows": int(len(excluded)),
        "exclusions_by_reason": {
            str(key): int(value)
            for key, value in excluded["exclusion_reason"].value_counts().items()
        },
        "mapping_policy": mapping_policy,
    }
    return used, excluded, counts


def _validate_counts(matrix: Any) -> sparse.csr_matrix:
    result = sparse.csr_matrix(matrix, dtype=np.float64)
    values = result.data
    if values.size and (
        not np.isfinite(values).all()
        or float(values.min()) < 0
        or not np.allclose(values, np.rint(values), rtol=0.0, atol=1e-6)
    ):
        raise ValueError("Counts layer must contain finite, nonnegative integer values.")
    return result


def project_expression_matrices(
    expression: Any,
    counts: Any,
    var_names: Sequence[str],
    mapping: pd.DataFrame,
    *,
    mode: str = "strict_log1p_rename",
    normalization_target_sum: float = 1105.0,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Project zebrafish expression into the mouse-symbol feature space.

    ``strict_log1p_rename`` is the formal primary contract: reciprocal 1:1
    genes are subset and renamed, so the already single-log normalized values
    remain exactly unchanged. ``counts_sum_then_log1p`` is an explicit
    secondary adapter for many-to-one mappings and fixes the historical
    preprocessing target sum at 1105 rather than recomputing a filtered median.
    """

    if mode not in {"strict_log1p_rename", "counts_sum_then_log1p"}:
        raise ValueError(f"Unsupported expression projection mode: {mode}")
    names = pd.Index([str(value) for value in var_names])
    if names.has_duplicates:
        raise ValueError("Input gene names must be unique.")
    if getattr(expression, "shape", None) != getattr(counts, "shape", None):
        raise ValueError("Expression and counts matrices must have the same shape.")
    if expression.shape[1] != len(names):
        raise ValueError("Expression columns do not match the supplied gene names.")
    position = {gene: index for index, gene in enumerate(names)}
    present = mapping.loc[mapping["source_gene"].isin(position)].copy()
    absent = mapping.loc[~mapping["source_gene"].isin(position)].copy()
    if present.empty:
        raise ValueError("No selected orthology genes are present in the expression matrix.")
    present = present.sort_values(["target_gene", "source_gene"], kind="mergesort")
    target_names = sorted(present["target_gene"].unique())
    target_index = {gene: index for index, gene in enumerate(target_names)}
    source_indices = np.array([position[gene] for gene in present["source_gene"]], dtype=int)
    counts_csr = _validate_counts(counts)

    if mode == "strict_log1p_rename":
        if present["target_gene"].duplicated().any():
            raise ValueError(
                "strict_log1p_rename requires a one-source-to-one-target mapping; "
                "use counts_sum_then_log1p for many-to-one aggregation."
            )
        source_order = [
            position[
                present.loc[present["target_gene"].eq(target), "source_gene"].iloc[0]
            ]
            for target in target_names
        ]
        projected_expression = sparse.csr_matrix(expression[:, source_order], dtype=np.float32)
        expression_values = projected_expression.data
        if expression_values.size and (
            not np.isfinite(expression_values).all()
            or float(expression_values.min()) < 0
        ):
            raise ValueError(
                "strict_log1p_rename requires finite, nonnegative single-log expression."
            )
        projected_counts = _validate_counts(counts_csr[:, source_order]).astype(np.int32)
        identity_max_abs_error = 0.0
        resolved_target_sum = None
        transformation = "subset original single-log X and rename reciprocal 1:1 genes"
    else:
        if not np.isfinite(normalization_target_sum) or normalization_target_sum <= 0:
            raise ValueError("normalization_target_sum must be positive and finite.")
        projection = sparse.coo_matrix(
            (
                np.ones(len(present), dtype=np.int32),
                (
                    np.arange(len(present), dtype=int),
                    np.array([target_index[value] for value in present["target_gene"]], dtype=int),
                ),
            ),
            shape=(len(present), len(target_names)),
        ).tocsr()
        projected_counts = (counts_csr[:, source_indices] @ projection).tocsr()
        projected_counts = _validate_counts(projected_counts).astype(np.int32)
        library_sums = np.asarray(projected_counts.sum(axis=1)).reshape(-1).astype(float)
        if np.any(~np.isfinite(library_sums)) or np.any(library_sums <= 0):
            raise ValueError("Every cell must retain a positive mapped-count library.")
        normalized = sparse.diags(float(normalization_target_sum) / library_sums).dot(
            projected_counts.astype(np.float64)
        )
        normalized = normalized.tocsr()
        np.log1p(normalized.data, out=normalized.data)
        projected_expression = normalized.astype(np.float32)
        identity_max_abs_error = None
        resolved_target_sum = float(normalization_target_sum)
        transformation = "sum mapped raw counts, fixed-library normalization, then log1p"

    source_groups = present.groupby("target_gene", sort=True)["source_gene"].agg(list)
    var = pd.DataFrame(index=pd.Index(target_names, name="mouse_gene"))
    var["gene"] = var.index.astype(str)
    var["source_gene_count"] = [len(source_groups[target]) for target in target_names]
    var["source_genes"] = [";".join(source_groups[target]) for target in target_names]
    record = {
        "mode": mode,
        "transformation": transformation,
        "normalization_target_sum": resolved_target_sum,
        "selected_space_identity_max_abs_error": identity_max_abs_error,
        "full_matrix_elementwise_comparison_applicable": False,
        "full_matrix_comparison_reason": (
            "orthology projection drops unmapped genes, renames features, and changes the feature dimension"
            if mode == "strict_log1p_rename"
            else "orthology aggregation changes the feature universe and expression is re-normalized from raw counts"
        ),
        "n_input_genes": int(len(names)),
        "n_output_genes": int(len(target_names)),
        "n_mapping_rows_present": int(len(present)),
        "n_mapping_rows_absent": int(len(absent)),
        "mapped_count_library_median": float(
            np.median(np.asarray(projected_counts.sum(axis=1)).reshape(-1))
        ),
    }
    return projected_expression, projected_counts, var, present, record


def build_sampling_plan(
    obs: pd.DataFrame,
    obs_names: Sequence[str],
    *,
    cell_type_key: str,
    time_key: str,
    time_label_key: str,
    seeds: Sequence[int],
    max_cells_per_type: int,
    minimum_cells_per_type: int = 1,
) -> pd.DataFrame:
    for key in (cell_type_key, time_key, time_label_key):
        if key not in obs:
            raise KeyError(f"Missing adata.obs[{key!r}].")
    names = pd.Index([str(value) for value in obs_names])
    if names.has_duplicates:
        raise ValueError("Observation names must be unique.")
    if max_cells_per_type < 1 or minimum_cells_per_type < 1:
        raise ValueError("Sampling limits must be positive.")
    stages = pd.to_numeric(obs[time_key], errors="raise").to_numpy(float)
    labels = obs[cell_type_key].astype(str).to_numpy()
    time_labels = obs[time_label_key].astype(str).to_numpy()
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        rng = np.random.default_rng(int(seed))
        for stage in np.sort(np.unique(stages)):
            stage_indices = np.flatnonzero(np.isclose(stages, stage, rtol=0.0, atol=1e-12))
            observed_stage_labels = sorted(set(time_labels[stage_indices]))
            if len(observed_stage_labels) != 1:
                raise ValueError(
                    f"Stage {stage:g} maps to multiple labels: {observed_stage_labels}."
                )
            stage_label = observed_stage_labels[0]
            for cell_type in sorted(set(labels[stage_indices])):
                available = stage_indices[labels[stage_indices] == cell_type]
                if len(available) < minimum_cells_per_type:
                    continue
                count = min(int(max_cells_per_type), len(available))
                selected = np.sort(rng.choice(available, size=count, replace=False))
                for sample_order, index in enumerate(selected):
                    rows.append(
                        {
                            "sampling_seed": int(seed),
                            "stage": float(stage),
                            "stage_label": stage_label,
                            "cell_type": str(cell_type),
                            "obs_name": names[index],
                            "original_index": int(index),
                            "within_type_sample_order": int(sample_order),
                            "n_type_cells_available": int(len(available)),
                            "n_type_cells_sampled": int(count),
                        }
                    )
    plan = pd.DataFrame(rows)
    if plan.empty:
        raise ValueError("Sampling plan is empty.")
    duplicate_keys = ["sampling_seed", "stage", "obs_name"]
    if plan.duplicated(duplicate_keys).any():
        raise RuntimeError("Sampling plan contains duplicated cells within a stage/seed.")
    return plan.sort_values(
        ["sampling_seed", "stage", "cell_type", "original_index"], kind="mergesort"
    ).reset_index(drop=True)


def build_lr_databases(
    *,
    official_database: Path,
    custom_database: pd.DataFrame,
    mapping: pd.DataFrame,
    output_dir: Path,
    ligand_column: str = "0",
    receptor_column: str = "1",
    pathway_column: str | None = "2",
    category_column: str | None = "3",
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    official = pd.read_csv(official_database, sep="\t")
    official_required = {"lr_pair", "ligand_gene_symbol", "receptor_gene_symbol"}
    missing = sorted(official_required.difference(official.columns))
    if missing:
        raise ValueError(f"Official CellAgentChat database lacks columns: {missing}")
    if official["lr_pair"].duplicated().any():
        raise ValueError("Official CellAgentChat LR identifiers are not unique.")
    official_out = output_dir / "official_mouse_default.tsv"
    shutil.copy2(official_database, official_out)

    custom = custom_database.copy().reset_index(drop=False).rename(
        columns={"index": "source_lr_row"}
    )
    custom["source_ligand"] = _string_series(custom, ligand_column)
    custom["source_receptor"] = _string_series(custom, receptor_column)
    source_to_target = dict(zip(mapping["source_gene"], mapping["target_gene"]))
    reasons: list[str] = []
    mapped_ligands: list[str] = []
    mapped_receptors: list[str] = []
    for ligand, receptor in zip(custom["source_ligand"], custom["source_receptor"]):
        row_reasons: list[str] = []
        if not ligand:
            row_reasons.append("missing_ligand")
        if not receptor:
            row_reasons.append("missing_receptor")
        if "_" in ligand:
            row_reasons.append("ligand_complex_unrepresentable")
        if "_" in receptor:
            row_reasons.append("receptor_complex_unrepresentable")
        mapped_ligand = source_to_target.get(ligand, "")
        mapped_receptor = source_to_target.get(receptor, "")
        if ligand and "_" not in ligand and not mapped_ligand:
            row_reasons.append("ligand_not_in_selected_orthology")
        if receptor and "_" not in receptor and not mapped_receptor:
            row_reasons.append("receptor_not_in_selected_orthology")
        if "_" in mapped_ligand:
            row_reasons.append("mapped_ligand_contains_underscore")
        if "_" in mapped_receptor:
            row_reasons.append("mapped_receptor_contains_underscore")
        reasons.append(";".join(row_reasons))
        mapped_ligands.append(mapped_ligand)
        mapped_receptors.append(mapped_receptor)
    custom["mapped_ligand"] = mapped_ligands
    custom["mapped_receptor"] = mapped_receptors
    custom["exclusion_reason"] = reasons
    eligible = custom.loc[custom["exclusion_reason"].eq("")].copy()
    eligible["mapped_lr_pair"] = (
        eligible["mapped_ligand"] + "_" + eligible["mapped_receptor"]
    )
    eligible["source_lr_pair"] = (
        eligible["source_ligand"] + "_" + eligible["source_receptor"]
    )
    eligible["mapped_db_row"] = eligible.groupby(
        ["mapped_ligand", "mapped_receptor"], sort=True
    ).ngroup()
    eligible["mapped_pair_multiplicity"] = eligible.groupby(
        ["mapped_ligand", "mapped_receptor"], sort=False
    )["source_lr_row"].transform("size")

    custom_lr = (
        eligible[["mapped_ligand", "mapped_receptor"]]
        .drop_duplicates()
        .sort_values(["mapped_ligand", "mapped_receptor"], kind="mergesort")
        .rename(
            columns={
                "mapped_ligand": "ligand_gene_symbol",
                "mapped_receptor": "receptor_gene_symbol",
            }
        )
        .reset_index(drop=True)
    )
    custom_lr.insert(
        0,
        "lr_pair",
        custom_lr["ligand_gene_symbol"] + "_" + custom_lr["receptor_gene_symbol"],
    )
    if custom_lr.empty:
        raise ValueError("No custom LR pairs survive the CellAgentChat projection.")
    if custom_lr["lr_pair"].duplicated().any():
        raise RuntimeError("Projected custom LR identifiers are not unique.")
    custom_out = output_dir / "cytobridge_zebrafish_lr_projected_singletons.tsv"
    custom_lr.to_csv(custom_out, sep="\t", index=False)

    crosswalk_out = output_dir / "custom_lr_projection_crosswalk.csv"
    eligible.to_csv(crosswalk_out, index=False)
    excluded = custom.loc[~custom["exclusion_reason"].eq("")].copy()
    excluded_out = output_dir / "custom_lr_excluded_rows.csv"
    excluded.to_csv(excluded_out, index=False)

    coverage_rows: list[dict[str, Any]] = []
    if category_column and category_column in custom:
        categories = custom[category_column].fillna("<missing>").astype(str)
    else:
        categories = pd.Series("<not_provided>", index=custom.index)
    singleton_mask = ~custom["source_ligand"].str.contains("_", regex=False) & ~custom[
        "source_receptor"
    ].str.contains("_", regex=False)
    mapped_mask = custom["exclusion_reason"].eq("")
    for category in sorted(set(categories)):
        mask = categories.eq(category)
        total = int(mask.sum())
        singleton = int((mask & singleton_mask).sum())
        mapped = int((mask & mapped_mask).sum())
        coverage_rows.append(
            {
                "category": category,
                "total_source_rows": total,
                "singleton_representable_rows": singleton,
                "mapped_rows": mapped,
                "singleton_fraction": singleton / total if total else np.nan,
                "mapped_fraction": mapped / total if total else np.nan,
            }
        )
    category_coverage = pd.DataFrame(coverage_rows)
    category_out = output_dir / "custom_lr_category_coverage.csv"
    category_coverage.to_csv(category_out, index=False)

    pathway_out = output_dir / "custom_lr_pathway_coverage.csv"
    if pathway_column and pathway_column in custom:
        pathways = custom[pathway_column].fillna("<missing>").astype(str)
        pathway_rows = []
        for pathway in sorted(set(pathways)):
            mask = pathways.eq(pathway)
            pathway_rows.append(
                {
                    "pathway": pathway,
                    "total_source_rows": int(mask.sum()),
                    "singleton_representable_rows": int((mask & singleton_mask).sum()),
                    "mapped_rows": int((mask & mapped_mask).sum()),
                }
            )
        pd.DataFrame(pathway_rows).to_csv(pathway_out, index=False)
    else:
        pd.DataFrame(
            columns=["pathway", "total_source_rows", "singleton_representable_rows", "mapped_rows"]
        ).to_csv(pathway_out, index=False)

    counts = {
        "official_mouse_rows": int(len(official)),
        "custom_source_rows": int(len(custom)),
        "custom_singleton_representable_rows": int(singleton_mask.sum()),
        "custom_mapped_source_rows": int(len(eligible)),
        "custom_mapped_source_pairs": int(
            eligible[["source_ligand", "source_receptor"]].drop_duplicates().shape[0]
        ),
        "custom_mapped_target_pairs": int(len(custom_lr)),
        "custom_excluded_rows": int(len(excluded)),
        "custom_exclusions_by_reason": {
            str(key): int(value)
            for key, value in excluded["exclusion_reason"].value_counts().items()
        },
    }
    return {
        "counts": counts,
        "databases": {
            OFFICIAL_DATABASE_LABEL: artifact(official_out),
            CUSTOM_DATABASE_LABEL: artifact(custom_out),
        },
        "artifacts": {
            "custom_lr_projection_crosswalk.csv": artifact(crosswalk_out),
            "custom_lr_excluded_rows.csv": artifact(excluded_out),
            "custom_lr_category_coverage.csv": artifact(category_out),
            "custom_lr_pathway_coverage.csv": artifact(pathway_out),
        },
    }


def verify_artifact(record: Mapping[str, Any]) -> Path:
    path = Path(str(record["path"])).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    expected = str(record.get("sha256", ""))
    observed = sha256_file(path)
    if expected and observed != expected:
        raise RuntimeError(
            f"Artifact SHA256 mismatch for {path}: expected {expected}, observed {observed}."
        )
    return path
