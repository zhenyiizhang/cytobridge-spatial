#!/usr/bin/env python3
"""Run one pinned CellAgentChat spatial condition on frozen shared cells.

This adapter follows the CellAgentChat v0.2.0 spatial tutorial contract:
spatial distance is enabled for the observed model, the permutation background
is distance-scaled, and the native sender-to-receiver score is the number of
Bonferroni-significant LR pairs.  No UMAP or other embedding is treated as
physical space.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import importlib
import json
import os
from pathlib import Path
import random
import sys
import types
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

try:
    from .common import (
        CONDITION_LABELS,
        artifact,
        csv_ints,
        json_value,
        prepare_output,
        utc_now,
        validate_official_source,
        verify_artifact,
        write_json,
    )
except ImportError:  # pragma: no cover - direct CLI execution
    from common import (  # type: ignore
        CONDITION_LABELS,
        artifact,
        csv_ints,
        json_value,
        prepare_output,
        utc_now,
        validate_official_source,
        verify_artifact,
        write_json,
    )


def _csv_floats(value: str) -> tuple[float, ...]:
    parsed = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not parsed or len(parsed) != len(set(parsed)):
        raise argparse.ArgumentTypeError("Expected unique comma-separated numbers.")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation-dir", required=True, type=Path)
    parser.add_argument("--cellagentchat-source", required=True, type=Path)
    parser.add_argument("--database-label", required=True, choices=CONDITION_LABELS)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--sampling-seeds", type=csv_ints)
    parser.add_argument("--stages", type=_csv_floats)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--feature-shuffles", type=int, default=1)
    parser.add_argument("--permutation-score-target", type=int, default=10_000)
    parser.add_argument("--bonferroni-threshold", type=float, default=0.05)
    parser.add_argument("--tau", type=float, default=2.0)
    parser.add_argument("--delta", type=float, default=1.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--allow-unpinned-source", action="store_true")
    parser.add_argument(
        "--allow-nonprimary-preparation",
        action="store_true",
        help="Explicitly allow a preparation labelled as a sensitivity rather than formal primary.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _read_preparation(
    preparation_dir: Path,
    database_label: str,
) -> tuple[dict[str, Any], Path, Path, Path]:
    preparation_dir = preparation_dir.expanduser().resolve()
    manifest_path = preparation_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("workflow") != "zebrafish_cellagentchat_dual_database_shared_input":
        raise RuntimeError("Preparation manifest has an unexpected workflow identifier.")
    if database_label not in manifest.get("lr_databases", {}):
        raise KeyError(f"Preparation manifest lacks database {database_label!r}.")
    mapped_h5ad = verify_artifact(manifest["artifacts"]["mapped_expression"])
    sample_plan = verify_artifact(manifest["artifacts"]["shared_sampled_cells"])
    database = verify_artifact(manifest["lr_databases"][database_label])
    return manifest, mapped_h5ad, sample_plan, database


@contextmanager
def _official_imports(source: Path):
    source_dir = source / "src"
    names = (
        "output_paths",
        "preprocessor",
        "model_setup",
        "abm",
        "permutations",
        "bckground_distribution",
    )
    previous = {name: sys.modules.get(name) for name in names}
    for name in names:
        sys.modules.pop(name, None)
    sys.path.insert(0, str(source_dir))
    try:
        modules = {name: importlib.import_module(name) for name in names}
        yield modules
    finally:
        if sys.path and sys.path[0] == str(source_dir):
            sys.path.pop(0)
        for name in names:
            sys.modules.pop(name, None)
            if previous[name] is not None:
                sys.modules[name] = previous[name]


def _set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _ensure_torch_sparse_backend() -> str:
    """Provide the two ``torch_sparse`` operations used by sparselinear.

    CellAgentChat v0.2.0 depends on sparselinear 0.0.5.  When a compiled
    torch-sparse wheel is unavailable for the installed PyTorch/CUDA build,
    the same COO coalescing and sparse matrix multiplication are implemented
    with PyTorch's native differentiable sparse operators.  This is a runtime
    compatibility backend, not a change to the CellAgentChat model, and the
    selected backend is sealed in the output manifest.
    """

    try:
        import torch_sparse  # noqa: F401

        return "compiled_torch_sparse"
    except ModuleNotFoundError as error:
        if error.name != "torch_sparse":
            raise

    import torch

    compatibility = types.ModuleType("torch_sparse")

    def spmm(
        index: "torch.Tensor",
        value: "torch.Tensor",
        m: int,
        n: int,
        matrix: "torch.Tensor",
    ) -> "torch.Tensor":
        sparse_matrix = torch.sparse_coo_tensor(
            index,
            value,
            size=(int(m), int(n)),
            dtype=value.dtype,
            device=value.device,
        ).coalesce()
        return torch.sparse.mm(sparse_matrix, matrix)

    def coalesce(
        index: "torch.Tensor",
        value: "torch.Tensor",
        m: int,
        n: int,
        op: str = "add",
    ) -> tuple["torch.Tensor", "torch.Tensor"]:
        if op != "add":
            raise ValueError("The compatibility backend supports only coalesce(op='add').")
        sparse_matrix = torch.sparse_coo_tensor(
            index,
            value,
            size=(int(m), int(n)),
            dtype=value.dtype,
            device=value.device,
        ).coalesce()
        return sparse_matrix.indices(), sparse_matrix.values()

    compatibility.spmm = spmm
    compatibility.coalesce = coalesce
    sys.modules["torch_sparse"] = compatibility
    return "torch_native_sparse_compat_v1"


def _tokenize(labels: Iterable[str]) -> tuple[dict[str, str], dict[str, str]]:
    ordered = sorted(set(str(label) for label in labels))
    forward = {label: f"CT{index:03d}" for index, label in enumerate(ordered)}
    return forward, {token: label for label, token in forward.items()}


def _decode_pair(key: str, tokens: Sequence[str]) -> tuple[str, str]:
    matches = [
        (sender, receiver)
        for sender in tokens
        for receiver in tokens
        if key == f"{sender}_{receiver}"
    ]
    if len(matches) != 1:
        raise ValueError(f"Cannot uniquely decode CellAgentChat cluster pair {key!r}.")
    return matches[0]


def _score_float(value: Any) -> float:
    values = np.asarray(value, dtype=float)
    if values.size == 0:
        return 0.0
    result = float(values.sum())
    if not np.isfinite(result):
        raise ValueError("CellAgentChat returned a non-finite LR score.")
    return result


def _lr_key_map(
    ligand_universe: Mapping[str, Sequence[str]],
) -> dict[str, tuple[str, str]]:
    """Decode CellAgentChat's hyphen-delimited output without guessing.

    CellAgentChat emits ``ligand-receptor`` strings, while valid gene symbols
    may themselves contain hyphens.  Reconstructing keys from the exact loaded
    LR universe avoids an unsafe ``split('-', 1)``.
    """

    decoded: dict[str, tuple[str, str]] = {}
    for ligand, receptors in ligand_universe.items():
        for receptor in receptors:
            key = f"{ligand}-{receptor}"
            pair = (str(ligand), str(receptor))
            previous = decoded.get(key)
            if previous is not None and previous != pair:
                raise ValueError(
                    "CellAgentChat LR output key is ambiguous after hyphen joining: "
                    f"{key!r} represents both {previous!r} and {pair!r}."
                )
            decoded[key] = pair
    return decoded


def _decode_lr_key(
    key: str,
    lr_keys: Mapping[str, tuple[str, str]] | None,
) -> tuple[str, str]:
    if lr_keys is not None:
        try:
            return lr_keys[key]
        except KeyError as error:
            raise ValueError(
                f"CellAgentChat returned LR key {key!r} outside the loaded database."
            ) from error
    parts = key.split("-")
    if len(parts) != 2:
        raise ValueError(
            "An explicit LR-key map is required when ligand or receptor names "
            f"contain hyphens: {key!r}."
        )
    return parts[0], parts[1]


def _flatten_raw_results(
    results: Mapping[str, Mapping[str, Any]],
    token_to_label: Mapping[str, str],
    *,
    stage: float,
    stage_label: str,
    sampling_seed: int,
    lr_keys: Mapping[str, tuple[str, str]] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    tokens = sorted(token_to_label)
    for pair_key, lr_scores in results.items():
        sender_token, receiver_token = _decode_pair(str(pair_key), tokens)
        for lr_pair, score in lr_scores.items():
            ligand, receptor = _decode_lr_key(str(lr_pair), lr_keys)
            rows.append(
                {
                    "stage": float(stage),
                    "stage_label": stage_label,
                    "sampling_seed": int(sampling_seed),
                    "sender_type": token_to_label[sender_token],
                    "receiver_type": token_to_label[receiver_token],
                    "ligand": ligand,
                    "receptor": receptor,
                    "lr_pair": f"{ligand}_{receptor}",
                    "cellagentchat_score_raw": _score_float(score),
                }
            )
    columns = [
        "stage",
        "stage_label",
        "sampling_seed",
        "sender_type",
        "receiver_type",
        "ligand",
        "receptor",
        "lr_pair",
        "cellagentchat_score_raw",
    ]
    return pd.DataFrame(rows, columns=columns)


def _flatten_significant_results(
    significant: Mapping[str, Mapping[str, Any]],
    adjusted_pvalues: Mapping[str, Mapping[str, Any]],
    token_to_label: Mapping[str, str],
    *,
    stage: float,
    stage_label: str,
    sampling_seed: int,
    lr_keys: Mapping[str, tuple[str, str]] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    tokens = sorted(token_to_label)
    for pair_key, lr_scores in significant.items():
        sender_token, receiver_token = _decode_pair(str(pair_key), tokens)
        for lr_pair, score in lr_scores.items():
            ligand, receptor = _decode_lr_key(str(lr_pair), lr_keys)
            pvalue = adjusted_pvalues.get(pair_key, {}).get(lr_pair, np.nan)
            rows.append(
                {
                    "stage": float(stage),
                    "stage_label": stage_label,
                    "sampling_seed": int(sampling_seed),
                    "sender_type": token_to_label[sender_token],
                    "receiver_type": token_to_label[receiver_token],
                    "ligand": ligand,
                    "receptor": receptor,
                    "lr_pair": f"{ligand}_{receptor}",
                    "cellagentchat_score": _score_float(score),
                    "cellagentchat_bonferroni_p": float(pvalue),
                }
            )
    columns = [
        "stage",
        "stage_label",
        "sampling_seed",
        "sender_type",
        "receiver_type",
        "ligand",
        "receptor",
        "lr_pair",
        "cellagentchat_score",
        "cellagentchat_bonferroni_p",
    ]
    return pd.DataFrame(rows, columns=columns)


def _complete_type_pairs(
    raw: pd.DataFrame,
    significant: pd.DataFrame,
    labels: Sequence[str],
    sampled_counts: Mapping[str, int],
    *,
    stage: float,
    stage_label: str,
    sampling_seed: int,
    n_lr_pairs_tested: int,
) -> pd.DataFrame:
    raw_groups = {
        (str(sender), str(receiver)): frame
        for (sender, receiver), frame in raw.groupby(
            ["sender_type", "receiver_type"], sort=False
        )
    }
    sig_groups = {
        (str(sender), str(receiver)): frame
        for (sender, receiver), frame in significant.groupby(
            ["sender_type", "receiver_type"], sort=False
        )
    }
    rows = []
    for sender in sorted(labels):
        for receiver in sorted(labels):
            raw_frame = raw_groups.get((sender, receiver))
            sig_frame = sig_groups.get((sender, receiver))
            significant_count = 0 if sig_frame is None else int(len(sig_frame))
            rows.append(
                {
                    "stage": float(stage),
                    "stage_label": stage_label,
                    "sampling_seed": int(sampling_seed),
                    "sender_type": sender,
                    "receiver_type": receiver,
                    "n_sender_cells_sampled": int(sampled_counts[sender]),
                    "n_receiver_cells_sampled": int(sampled_counts[receiver]),
                    "n_lr_pairs_tested": int(n_lr_pairs_tested),
                    "cellagentchat_significant_lr_count": significant_count,
                    "cellagentchat_significant_lr_fraction": (
                        significant_count / n_lr_pairs_tested if n_lr_pairs_tested else np.nan
                    ),
                    "cellagentchat_significant_score_sum": (
                        0.0
                        if sig_frame is None
                        else float(sig_frame["cellagentchat_score"].sum())
                    ),
                    "cellagentchat_raw_score_sum": (
                        0.0
                        if raw_frame is None
                        else float(raw_frame["cellagentchat_score_raw"].sum())
                    ),
                    "cellagentchat_native_primary": significant_count,
                    "heterotypic": bool(sender != receiver),
                }
            )
    return pd.DataFrame(rows)


def _summarize_type_pairs(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["stage", "stage_label", "sender_type", "receiver_type"]
    numeric = [
        "cellagentchat_significant_lr_count",
        "cellagentchat_significant_lr_fraction",
        "cellagentchat_significant_score_sum",
        "cellagentchat_raw_score_sum",
        "cellagentchat_native_primary",
    ]
    grouped = frame.groupby(keys, sort=True, dropna=False)
    result = grouped[numeric].agg(["mean", "std", "min", "max"])
    result.columns = [
        "_".join(str(item) for item in column if str(item))
        for column in result.columns.to_flat_index()
    ]
    result = result.reset_index()
    result["n_sampling_seeds"] = grouped.size().to_numpy()
    return result


def _run_one(
    *,
    data: Any,
    plan: pd.DataFrame,
    stage: float,
    stage_label: str,
    sampling_seed: int,
    database: Path,
    run_dir: Path,
    keys: Mapping[str, str],
    args: argparse.Namespace,
    modules: Mapping[str, Any],
) -> dict[str, Any]:
    import scipy.sparse as sp

    selected_names = plan["obs_name"].astype(str).tolist()
    subset = data[selected_names].copy()
    observed_stage = pd.to_numeric(subset.obs[keys["time"]], errors="raise").to_numpy(float)
    if not np.allclose(observed_stage, stage, rtol=0.0, atol=1e-12):
        raise RuntimeError("Frozen sample plan does not match the mapped H5AD stage.")
    observed_labels = subset.obs[keys["cell_type"]].astype(str).to_numpy()
    if observed_labels.tolist() != plan["cell_type"].astype(str).tolist():
        raise RuntimeError("Frozen sample plan cell types do not match the mapped H5AD.")
    values = subset.X.data if sp.issparse(subset.X) else np.asarray(subset.X)
    values = np.asarray(values)
    if values.size and (not np.isfinite(values).all() or float(values.min()) < 0):
        raise ValueError("Mapped single-log expression must be finite and nonnegative.")
    if subset.var_names.has_duplicates:
        raise ValueError("CellAgentChat requires unique target gene symbols.")

    preprocessor = modules["preprocessor"]
    subset = preprocessor.setup_adata(
        subset,
        coordinates_key=keys["spatial"],
        batch_label=None,
        cell_type_label=keys["cell_type"],
        scale_for_mesa=True,
        copy=False,
    )
    label_to_token, token_to_label = _tokenize(observed_labels)
    subset.obs["cell_type"] = [label_to_token[label] for label in observed_labels]
    subset.obs["Batch"] = f"stage_{stage:g}"
    sampled_counts = {
        label: int(value)
        for label, value in pd.Series(observed_labels).value_counts().items()
    }

    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "original_label": list(label_to_token),
            "cellagentchat_token": list(label_to_token.values()),
        }
    ).to_csv(run_dir / "cell_type_token_map.csv", index=False)
    sampled_export = plan.copy()
    sampled_export["cellagentchat_token"] = [
        label_to_token[label] for label in sampled_export["cell_type"].astype(str)
    ]
    sampled_export.to_csv(run_dir / "sampled_cells.csv", index=False)

    model_setup = modules["model_setup"]
    permutations = modules["permutations"]
    background = modules["bckground_distribution"]
    abm_module = modules["abm"]
    _set_seed(int(sampling_seed))
    lig_uni, rec_uni, lr_pairs = model_setup.load_db(
        subset, file=str(database), sep="\t"
    )
    tf_uni, rec_tf_uni = model_setup.load_tf_db("mouse", subset, rec_uni)
    if not lig_uni or not rec_uni or not lr_pairs:
        raise RuntimeError("CellAgentChat LR universe is empty after input filtering.")
    lr_keys = _lr_key_map(lig_uni)
    if len(lr_keys) != len(lr_pairs):
        raise RuntimeError(
            "The loaded CellAgentChat LR universe contains duplicate or ambiguous pairs."
        )
    unsafe_lr_genes = sorted(
        gene for gene in {*lig_uni, *rec_uni} if "_" in str(gene)
    )
    if unsafe_lr_genes:
        raise ValueError(
            "CellAgentChat v0.2.0 cannot parse underscores in LR gene symbols: "
            f"{unsafe_lr_genes[:10]}."
        )
    if not tf_uni:
        raise RuntimeError("CellAgentChat mouse TF-target universe is empty.")

    model_path = run_dir / "conversion_model.pt"
    inputs, all_genes = model_setup.train(
        subset,
        lig_uni,
        rec_uni,
        tf_uni,
        rec_tf_uni,
        lr_pairs,
        path=str(model_path),
        epochs=int(args.epochs),
        lr=float(args.learning_rate),
        batch=int(args.batch_size),
        device=args.device,
    )
    fitted = model_setup.load_model(str(model_path), device=args.device)
    conversion_values = model_setup.feature_selection(
        model=fitted,
        mat=inputs,
        C=all_genes,
        rec_uni=rec_uni,
        perc=100,
        n_shuffles=int(args.feature_shuffles),
        batch=int(args.batch_size),
        seed=int(sampling_seed),
        device=args.device,
    )
    rates = model_setup.add_rates(conversion_values, rec_uni)
    rate_values = np.asarray(list(rates.values()), dtype=float)
    if (
        len(rates) != len(rec_uni)
        or rate_values.size != len(rec_uni)
        or not np.isfinite(rate_values).all()
        or np.any(rate_values < 0)
    ):
        raise RuntimeError("CellAgentChat returned invalid receptor conversion rates.")
    pd.DataFrame(
        {"receptor": list(rates), "conversion_rate": list(rates.values())}
    ).to_csv(run_dir / "conversion_rates.csv", index=False)

    permutation_scores, _, average_distance = permutations.permutation_test(
        threshold=int(args.permutation_score_target),
        N=subset.n_obs,
        adata=subset,
        lig_uni=lig_uni,
        rec_uni=rec_uni,
        rates=rates,
        dist=True,
        tau=float(args.tau),
        seed=int(sampling_seed),
    )
    if not np.isfinite(average_distance) or float(average_distance) < 0:
        raise RuntimeError("CellAgentChat returned an invalid spatial distance scale.")
    distribution = background.get_distribution(
        permutation_scores,
        dist=float(average_distance),
        scaled=True,
    )
    distribution_table = pd.DataFrame(
        [
            {
                "lr_pair": key.replace("-", "_"),
                "gamma_shape": float(value[0]),
                "gamma_location": float(value[1]),
                "gamma_scale": float(value[2]),
            }
            for key, value in sorted(distribution.items())
        ]
    )
    distribution_table.to_csv(run_dir / "background_distribution.csv", index=False)

    model = abm_module.CellModel(
        N=subset.n_obs,
        adata=subset,
        lig_uni=lig_uni,
        rec_uni=rec_uni,
        rates=rates,
        max_steps=1,
        dist=True,
        delta=float(args.delta),
        tau=float(args.tau),
        permutations=False,
        net=str(model_path),
    )
    model.step(show_agent_progress=True)
    raw = _flatten_raw_results(
        model.results2,
        token_to_label,
        stage=stage,
        stage_label=stage_label,
        sampling_seed=sampling_seed,
        lr_keys=lr_keys,
    )
    significant_dict, adjusted_pvalues, _ = background.get_significant_lr_pairs(
        model.results2,
        distribution,
        cutoff=float(args.bonferroni_threshold),
    )
    significant = _flatten_significant_results(
        significant_dict,
        adjusted_pvalues,
        token_to_label,
        stage=stage,
        stage_label=stage_label,
        sampling_seed=sampling_seed,
        lr_keys=lr_keys,
    )
    type_pairs = _complete_type_pairs(
        raw,
        significant,
        sorted(label_to_token),
        sampled_counts,
        stage=stage,
        stage_label=stage_label,
        sampling_seed=sampling_seed,
        n_lr_pairs_tested=len(lr_pairs),
    )
    raw.to_csv(run_dir / "cellagentchat_lr_scores_raw.csv.gz", index=False, compression="gzip")
    significant.to_csv(
        run_dir / "cellagentchat_lr_scores_significant.csv", index=False
    )
    type_pairs.to_csv(run_dir / "cellagentchat_type_pair_scores.csv", index=False)

    receiving = pd.DataFrame(
        {
            "stage": float(stage),
            "stage_label": stage_label,
            "sampling_seed": int(sampling_seed),
            "obs_name": selected_names,
            "cell_type": observed_labels,
            "cellagentchat_receiving_score": [
                float(agent.num_r) for agent in model.schedule.agents
            ],
        }
    )
    receiving.to_csv(run_dir / "cellagentchat_cell_receiving_scores.csv", index=False)
    run_manifest = {
        "stage": float(stage),
        "stage_label": stage_label,
        "sampling_seed": int(sampling_seed),
        "n_cells": int(subset.n_obs),
        "n_cell_types": int(len(label_to_token)),
        "n_genes": int(subset.n_vars),
        "n_ligands": int(len(lig_uni)),
        "n_receptors": int(len(rec_uni)),
        "n_lr_pairs_tested": int(len(lr_pairs)),
        "n_tfs": int(len(tf_uni)),
        "n_receptors_with_prior_tf": int(len(rec_tf_uni)),
        "n_raw_lr_rows": int(len(raw)),
        "n_significant_lr_rows": int(len(significant)),
        "average_normalized_spatial_distance": float(average_distance),
        "spatial_distance_used": True,
        "distance_scaled_permutation_background": True,
        "embedding_used_as_spatial": False,
        "artifacts": {
            path.name: artifact(path)
            for path in sorted(run_dir.iterdir())
            if path.is_file() and path.name != "run_manifest.json"
        },
    }
    write_json(run_dir / "run_manifest.json", run_manifest)
    return run_manifest


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    import anndata as ad
    import torch

    positive_ints = (
        "epochs",
        "batch_size",
        "feature_shuffles",
        "permutation_score_target",
    )
    for name in positive_ints:
        if int(getattr(args, name)) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive and finite.")
    if not 0 < float(args.bonferroni_threshold) < 1:
        raise ValueError("--bonferroni-threshold must lie between zero and one.")
    if args.tau <= 0 or args.delta <= 0:
        raise ValueError("--tau and --delta must be positive.")
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {args.device}")

    preparation, mapped_h5ad, plan_path, database = _read_preparation(
        args.preparation_dir, args.database_label
    )
    if (
        preparation.get("formal_primary") is not True
        and not bool(args.allow_nonprimary_preparation)
    ):
        raise RuntimeError(
            "Preparation is not labelled formal_primary=true. Pass "
            "--allow-nonprimary-preparation only for an explicitly labelled sensitivity run."
        )
    source = args.cellagentchat_source.expanduser().resolve()
    source_record = validate_official_source(
        source, allow_unpinned=bool(args.allow_unpinned_source)
    )
    output = prepare_output(args.output_dir, bool(args.overwrite))
    data = ad.read_h5ad(mapped_h5ad)
    plan = pd.read_csv(plan_path)
    keys = preparation["keys"]
    for key in (keys["cell_type"], keys["time"], keys["time_label"]):
        if key not in data.obs:
            raise KeyError(f"Mapped H5AD lacks adata.obs[{key!r}].")
    if keys["spatial"] not in data.obsm:
        raise KeyError(f"Mapped H5AD lacks adata.obsm[{keys['spatial']!r}].")
    required_plan = {
        "sampling_seed",
        "stage",
        "stage_label",
        "cell_type",
        "obs_name",
        "original_index",
    }
    missing_plan = sorted(required_plan.difference(plan.columns))
    if missing_plan:
        raise ValueError(f"Sample plan lacks columns: {missing_plan}")
    available_seeds = tuple(sorted(pd.to_numeric(plan["sampling_seed"]).astype(int).unique()))
    seeds = tuple(args.sampling_seeds) if args.sampling_seeds else available_seeds
    unknown_seeds = sorted(set(seeds).difference(available_seeds))
    if unknown_seeds:
        raise ValueError(f"Sampling seeds absent from frozen plan: {unknown_seeds}")
    available_stages = tuple(sorted(pd.to_numeric(plan["stage"]).astype(float).unique()))
    stages = tuple(args.stages) if args.stages else available_stages
    unknown_stages = [
        stage
        for stage in stages
        if not any(np.isclose(stage, observed, rtol=0.0, atol=1e-12) for observed in available_stages)
    ]
    if unknown_stages:
        raise ValueError(f"Stages absent from frozen plan: {unknown_stages}")

    torch_sparse_backend = _ensure_torch_sparse_backend()
    run_records: list[dict[str, Any]] = []
    with _official_imports(source) as modules:
        for stage in stages:
            stage_rows = plan.loc[
                np.isclose(pd.to_numeric(plan["stage"]), stage, rtol=0.0, atol=1e-12)
            ]
            stage_labels = sorted(stage_rows["stage_label"].astype(str).unique())
            if len(stage_labels) != 1:
                raise ValueError(f"Stage {stage:g} has ambiguous labels: {stage_labels}")
            stage_label = stage_labels[0]
            for seed in seeds:
                run_plan = stage_rows.loc[
                    pd.to_numeric(stage_rows["sampling_seed"]).astype(int).eq(int(seed))
                ].copy()
                run_plan = run_plan.sort_values(
                    ["cell_type", "original_index"], kind="mergesort"
                ).reset_index(drop=True)
                if run_plan.empty:
                    raise ValueError(f"Frozen sample plan is empty for stage={stage}, seed={seed}.")
                run_dir = output / f"stage_{stage:g}_{stage_label}" / f"seed_{seed}"
                run_records.append(
                    _run_one(
                        data=data,
                        plan=run_plan,
                        stage=float(stage),
                        stage_label=stage_label,
                        sampling_seed=int(seed),
                        database=database,
                        run_dir=run_dir,
                        keys=keys,
                        args=args,
                        modules=modules,
                    )
                )

    type_paths = sorted(output.glob("stage_*/seed_*/cellagentchat_type_pair_scores.csv"))
    raw_paths = sorted(output.glob("stage_*/seed_*/cellagentchat_lr_scores_raw.csv.gz"))
    significant_paths = sorted(
        output.glob("stage_*/seed_*/cellagentchat_lr_scores_significant.csv")
    )
    receiving_paths = sorted(
        output.glob("stage_*/seed_*/cellagentchat_cell_receiving_scores.csv")
    )
    type_pairs = pd.concat((pd.read_csv(path) for path in type_paths), ignore_index=True)
    raw = pd.concat((pd.read_csv(path) for path in raw_paths), ignore_index=True)
    significant_frames = [
        pd.read_csv(path) for path in significant_paths if path.stat().st_size > 1
    ]
    significant = (
        pd.concat(significant_frames, ignore_index=True)
        if significant_frames
        else pd.DataFrame()
    )
    receiving = pd.concat(
        (pd.read_csv(path) for path in receiving_paths), ignore_index=True
    )
    type_pairs_path = output / "cellagentchat_type_pair_scores_by_seed.csv.gz"
    type_pairs.to_csv(type_pairs_path, index=False, compression="gzip")
    type_summary_path = output / "cellagentchat_type_pair_scores.csv"
    _summarize_type_pairs(type_pairs).to_csv(type_summary_path, index=False)
    raw_path = output / "cellagentchat_lr_scores_raw_by_seed.csv.gz"
    raw.to_csv(raw_path, index=False, compression="gzip")
    significant_path = output / "cellagentchat_lr_scores_significant_by_seed.csv.gz"
    significant.to_csv(significant_path, index=False, compression="gzip")
    receiving_path = output / "cellagentchat_cell_receiving_scores_by_seed.csv.gz"
    receiving.to_csv(receiving_path, index=False, compression="gzip")

    top_artifacts = {
        path.name: artifact(path)
        for path in (
            type_pairs_path,
            type_summary_path,
            raw_path,
            significant_path,
            receiving_path,
        )
    }
    manifest = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "method": "official_cellagentchat_v0_2_0_spatial",
        "database_condition": args.database_label,
        "source": source_record,
        "shared_input": {
            "preparation_manifest": artifact(
                args.preparation_dir.expanduser().resolve() / "manifest.json"
            ),
            "mapped_expression": artifact(mapped_h5ad),
            "sample_plan": artifact(plan_path),
            "database": artifact(database),
            "same_sample_plan_required_for_both_database_conditions": True,
            "preparation_formal_primary": bool(preparation.get("formal_primary")),
            "projection": preparation.get("projection", {}),
        },
        "design": {
            "species_prior": "mouse",
            "cross_species_interpretation": "zebrafish expression projected into mouse ortholog space",
            "sampling_seeds": list(seeds),
            "stages": list(stages),
            "epochs": int(args.epochs),
            "learning_rate": float(args.learning_rate),
            "batch_size": int(args.batch_size),
            "feature_shuffles": int(args.feature_shuffles),
            "permutation_score_target": int(args.permutation_score_target),
            "multiple_testing": "CellAgentChat v0.2.0 Bonferroni",
            "bonferroni_threshold": float(args.bonferroni_threshold),
            "spatial": True,
            "spatial_key": keys["spatial"],
            "permutation_background_distance_scaled": True,
            "tau": float(args.tau),
            "delta": float(args.delta),
            "native_primary": "number of Bonferroni-significant LR pairs per directed cell-type pair",
            "raw_score_sum_is_secondary": True,
            "device": args.device,
            "torch_sparse_backend": torch_sparse_backend,
            "torch_sparse_backend_semantics": (
                "compiled dependency when available; otherwise differentiable "
                "torch.sparse COO coalesce/mm compatibility for sparselinear 0.0.5"
            ),
        },
        "runs": run_records,
        "counts": {
            "n_runs": int(len(run_records)),
            "type_pair_rows_by_seed": int(len(type_pairs)),
            "raw_lr_rows": int(len(raw)),
            "significant_lr_rows": int(len(significant)),
            "cell_receiving_rows": int(len(receiving)),
        },
        "artifacts": top_artifacts,
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = run(args)
    print(json.dumps(json_value({"status": "ok", "counts": manifest["counts"]}), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
