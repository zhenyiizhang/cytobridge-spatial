#!/usr/bin/env python3
"""Run and visualize non-spatial communication consistency analyses.

CellChat and CytoBridge inputs are accepted frozen tables.  CellAgentChat is
run through its official non-spatial path (``dist=False``).  NicheNet uses its
official mouse ligand-target matrix and ligand-activity implementation, with
fixed-size terminal-versus-previous receiver response-gene sets.  Cross-method comparisons
are rank-based on complete directed cell-type-pair grids.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import subprocess
import sys
import types
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.lines import Line2D
from scipy.stats import hypergeom

try:
    from CytoBridge.nonspatial.communication_consistency import (
        METHODS,
        complete_directed_pairs,
        encode_cellagentchat_labels,
        pairwise_rank_metrics,
        prepare_shared_lr_database,
        prepare_nichenet_tables,
        rank_percentile,
        sha256_file,
        stratified_sample_indices,
        summarize_cellagentchat_pair_matrices,
    )
except ImportError:  # Standalone archived analysis copy.
    from communication_consistency import (  # type: ignore[no-redef]
        METHODS,
        complete_directed_pairs,
        encode_cellagentchat_labels,
        pairwise_rank_metrics,
        prepare_shared_lr_database,
        prepare_nichenet_tables,
        rank_percentile,
        sha256_file,
        stratified_sample_indices,
        summarize_cellagentchat_pair_matrices,
    )


SEED = 20260815
TOP_FRACTION = 0.20
PALETTE = {
    "CytoBridge": "#5B4B8A",
    "CellChat": "#2A9D8F",
    "CellAgentChat": "#E9C46A",
    "NicheNet": "#E76F51",
}


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _source_identity(path: Path) -> dict[str, object]:
    commit = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(path), "status", "--porcelain", "--untracked-files=normal"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status != "":
        raise ValueError(f"official source checkout is dirty: {path}")
    return {"path": str(path.resolve()), "commit": commit, "clean": True}


def _install_torch_sparse_compatibility() -> None:
    """Provide only torch_sparse.coalesce/spmm via native PyTorch operations.

    CellAgentChat depends on sparselinear, whose static-connectivity path uses
    these two operations.  The shim preserves the exact sparse matrix
    multiplication and autograd math without changing CellAgentChat source.
    """

    module = types.ModuleType("torch_sparse")

    def coalesce(indices, values, m, n):
        tensor = torch.sparse_coo_tensor(indices, values, (m, n)).coalesce()
        return tensor.indices(), tensor.values()

    def spmm(indices, values, m, n, matrix):
        tensor = torch.sparse_coo_tensor(indices, values, (m, n)).coalesce()
        return torch.sparse.mm(tensor, matrix)

    module.coalesce = coalesce
    module.spmm = spmm
    sys.modules["torch_sparse"] = module


def _stratified_terminal_adata(
    input_h5ad: Path,
    *,
    cell_type_key: str,
    time_key: str,
    terminal_time: float,
    sample_n: int,
) -> tuple[ad.AnnData, pd.DataFrame, pd.DataFrame]:
    data = ad.read_h5ad(input_h5ad)
    times = pd.to_numeric(data.obs[time_key], errors="raise").to_numpy(float)
    terminal = np.flatnonzero(np.isclose(times, terminal_time, rtol=0, atol=1e-8))
    if len(terminal) == 0:
        raise ValueError("terminal time contains no cells")
    within = stratified_sample_indices(
        data.obs.iloc[terminal][cell_type_key].astype(str), total=sample_n, seed=SEED
    )
    selected = terminal[within]
    sample = data[selected].copy()
    original_cell_types = sample.obs[cell_type_key].astype(str).to_numpy()
    encoded_cell_types, label_map = encode_cellagentchat_labels(original_cell_types)
    sample.obs["source_cell_type"] = original_cell_types
    sample.obs["cell_type"] = encoded_cell_types
    sample.obs["Batch"] = "0"
    values = sample.X.data if hasattr(sample.X, "tocsr") else np.asarray(sample.X)
    if np.any(~np.isfinite(values)):
        raise ValueError("CellAgentChat sample contains non-finite expression")
    roster = pd.DataFrame(
        {
            "sample_row": np.arange(len(selected), dtype=int),
            "source_row": selected,
            "obs_name": data.obs_names[selected].astype(str),
            "cell_type": original_cell_types,
            "cellagentchat_label": encoded_cell_types,
        }
    )
    return sample, roster, label_map


def run_cellagentchat(args: argparse.Namespace) -> None:
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"output directory exists: {output}")
    output.mkdir(parents=True)
    source = Path(args.cellagentchat_source).resolve()
    source_identity = _source_identity(source)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(source / "src"))
    _install_torch_sparse_compatibility()
    model_setup = importlib.import_module("model_setup")
    permutations = importlib.import_module("permutations")
    background = importlib.import_module("bckground_distribution")
    communication = importlib.import_module("Communication")
    abm = importlib.import_module("abm")

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    sample, roster, label_map = _stratified_terminal_adata(
        Path(args.input_h5ad),
        cell_type_key=args.cell_type_key,
        time_key=args.time_key,
        terminal_time=args.terminal_time,
        sample_n=args.sample_n,
    )
    sample.write_h5ad(output / "terminal_stratified_sample.h5ad")
    roster.to_csv(output / "sample_roster.csv", index=False)
    label_map.to_csv(output / "cell_type_label_map.csv", index=False)
    lr_database = (
        Path(args.lr_database).expanduser().resolve()
        if args.lr_database
        else source / "src/cellagentchat_data/mouse_lr_pair.tsv"
    )
    ligands, receptors, lr_pairs = model_setup.load_db(
        sample, file=str(lr_database), sep="\t"
    )
    tf_universe, receptor_tf = model_setup.load_tf_db("mouse", sample, receptors)
    model_path = output / "cellagentchat_model.pt"
    matrix, targets = model_setup.train(
        sample,
        ligands,
        receptors,
        tf_universe,
        receptor_tf,
        lr_pairs,
        path=str(model_path),
        epochs=args.epochs,
        batch=args.batch_size,
        lr=args.learning_rate,
        device=args.device,
    )
    model = model_setup.load_model(str(model_path), device=args.device)
    rates = model_setup.feature_selection(
        model,
        matrix,
        targets,
        receptors,
        perc=100,
        n_shuffles=1,
        batch=args.batch_size,
        seed=SEED,
        device=args.device,
    )
    model_setup.save_conversion_rate(rates, str(output / "conversion_rates.txt"))
    rate_map = model_setup.add_rates(rates, receptors)
    permutation_scores, _, _ = permutations.permutation_test(
        threshold=args.permutation_threshold,
        N=sample.n_obs,
        adata=sample,
        lig_uni=ligands,
        rec_uni=receptors,
        rates=rate_map,
        dist=False,
        seed=SEED,
    )
    distribution = background.get_distribution(
        permutation_scores, dist=0, scaled=False, pseudotime=False
    )
    background.save_distribution(
        distribution, str(output / "background_distribution.csv")
    )
    clusters = abm.get_cluster_choices(sample)
    cci_dir = output / "cci"
    communication.CCI(
        N=sample.n_obs,
        adata=sample,
        lig_uni=ligands,
        rec_uni=receptors,
        rates=rate_map,
        distribution=distribution,
        clusters=clusters,
        dist=False,
        max_steps=1,
        threshold=0.05,
        net=str(model_path),
        path=str(cci_dir),
        plot=False,
    )
    raw_scores = pd.read_csv(cci_dir / "1_sig_lr_pairs.csv", index_col=0)
    significant_scores = pd.read_csv(cci_dir / "1_new_sig_lr_pairs.csv", index_col=0)
    scores = summarize_cellagentchat_pair_matrices(
        raw_scores, significant_scores, label_map
    )
    scores.insert(0, "dataset", args.dataset)
    scores.to_csv(output / "directed_pair_scores.csv", index=False)
    manifest = {
        "schema_version": 2,
        "dataset": args.dataset,
        "method": "CellAgentChat",
        "official_nonspatial_mode": {"dist": False, "coordinates_used": False},
        "source": source_identity,
        "lr_database": {
            "path": str(lr_database),
            "sha256": sha256_file(lr_database),
            "mode": "explicit shared database"
            if args.lr_database
            else "CellAgentChat bundled mouse database",
        },
        "input_h5ad": {
            "path": str(Path(args.input_h5ad).resolve()),
            "sha256": sha256_file(args.input_h5ad),
        },
        "terminal_time": float(args.terminal_time),
        "sample_n": int(sample.n_obs),
        "cell_type_label_encoding": {
            "reason": "official CellAgentChat uses underscore-delimited directed-pair keys",
            "mapping_file": "cell_type_label_map.csv",
            "reversible": True,
        },
        "seed": SEED,
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "feature_selection_shuffles": 1,
            "permutation_threshold": args.permutation_threshold,
        },
        "directed_pair_scores": {
            "primary": "cellagentchat_native_ctps",
            "primary_definition": "sum of significant LR interaction scores (CellAgentChat CTPS)",
            "sensitivity": "cellagentchat_continuous_score",
            "diagnostic": "cellagentchat_significant_lr_count",
        },
        "sparse_linear_compatibility": {
            "torch_sparse_operations": ["coalesce", "spmm"],
            "implementation": "native torch.sparse_coo_tensor/coalesce/mm",
            "cellagentchat_source_modified": False,
        },
        "outputs": {},
    }
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            manifest["outputs"][str(path.relative_to(output))] = {
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
    _write_json(output / "manifest.json", manifest)


def prepare_shared_lr(args: argparse.Namespace) -> None:
    manifest = prepare_shared_lr_database(args.database, args.output_dir)
    _write_json(Path(args.output_dir) / "manifest.json", manifest)


def summarize_cellagentchat(args: argparse.Namespace) -> None:
    """Derive audited CTPS/continuous pair scores from one official run."""

    source = Path(args.cellagentchat_dir).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"output directory exists: {output}")
    required = {
        "source_manifest": source / "manifest.json",
        "label_map": source / "cell_type_label_map.csv",
        "continuous_lr_scores": source / "cci/1_sig_lr_pairs.csv",
        "significant_lr_scores": source / "cci/1_new_sig_lr_pairs.csv",
    }
    for name, path in required.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing CellAgentChat {name}: {path}")
    source_manifest = json.loads(required["source_manifest"].read_text())
    if source_manifest.get("method") != "CellAgentChat":
        raise ValueError("source manifest is not a CellAgentChat run")
    label_map = pd.read_csv(required["label_map"])
    scores = summarize_cellagentchat_pair_matrices(
        pd.read_csv(required["continuous_lr_scores"], index_col=0),
        pd.read_csv(required["significant_lr_scores"], index_col=0),
        label_map,
    )
    scores.insert(0, "dataset", source_manifest["dataset"])
    output.mkdir(parents=True)
    score_path = output / "directed_pair_scores.csv"
    scores.to_csv(score_path, index=False)
    manifest = {
        "schema_version": 1,
        "method": "CellAgentChat",
        "dataset": source_manifest["dataset"],
        "derivation": "audited directed-pair summary of frozen official CellAgentChat outputs",
        "primary_score": {
            "column": "cellagentchat_native_ctps",
            "definition": "sum of significant LR interaction scores (CellAgentChat CTPS)",
        },
        "sensitivity_score": {
            "column": "cellagentchat_continuous_score",
            "definition": "sum of threshold-free raw LR interaction scores",
        },
        "sources": {
            name: {
                "path": str(path),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for name, path in required.items()
        },
        "outputs": {
            score_path.name: {
                "sha256": sha256_file(score_path),
                "size_bytes": score_path.stat().st_size,
            }
        },
    }
    _write_json(output / "manifest.json", manifest)


def prepare_nichenet(args: argparse.Namespace) -> None:
    source = Path(args.nichenetr_source).resolve()
    identity = _source_identity(source)
    data = ad.read_h5ad(args.input_h5ad)
    network_csv = Path(args.lr_network_csv)
    network = pd.read_csv(network_csv)
    manifest = prepare_nichenet_tables(
        data,
        dataset=args.dataset,
        cell_type_key=args.cell_type_key,
        time_key=args.time_key,
        terminal_time=args.terminal_time,
        previous_time=args.previous_time,
        lr_network=network,
        output_dir=args.output_dir,
    )
    path = Path(args.output_dir) / "manifest.json"
    manifest["nichenetr_source"] = identity
    manifest["input_h5ad"] = {
        "path": str(Path(args.input_h5ad).resolve()),
        "sha256": sha256_file(args.input_h5ad),
    }
    manifest["lr_network"] = {
        "path": str(network_csv.resolve()),
        "sha256": sha256_file(network_csv),
    }
    _write_json(path, manifest)


def _nichenet_pair_scores(directory: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates = pd.read_csv(directory / "sender_receiver_lr_candidates.csv")
    official = directory / "official"
    activities = pd.read_csv(official / "ligand_activities.csv")
    targets = pd.read_csv(official / "ligand_target_links.csv")
    activities["activity_rank"] = activities.groupby("receiver", sort=False)[
        "aupr_corrected"
    ].transform(rank_percentile)
    evidence = candidates.merge(
        activities[["receiver", "ligand", "aupr_corrected", "activity_rank"]],
        on=["receiver", "ligand"],
        how="inner",
        validate="many_to_one",
    )
    evidence["lr_evidence"] = evidence["activity_rank"] * np.sqrt(
        evidence["sender_fraction"] * evidence["receiver_fraction"]
    )
    top = (
        evidence.sort_values("lr_evidence", ascending=False, kind="mergesort")
        .groupby(["dataset", "sender", "receiver"], sort=False)
        .head(5)
    )
    pair_scores = (
        top.groupby(["dataset", "sender", "receiver"], as_index=False)
        .agg(nichenet_support_score=("lr_evidence", "mean"))
        .rename(columns={"sender": "sender_type", "receiver": "receiver_type"})
    )
    detailed = evidence.merge(
        targets,
        on=["dataset", "receiver", "ligand"],
        how="inner",
        validate="many_to_many",
    )
    detailed["ligand_target_evidence"] = detailed["lr_evidence"] * detailed["weight"]
    return pair_scores, detailed


def aggregate(args: argparse.Namespace) -> None:
    config_path = Path(args.config).expanduser().resolve()
    config = json.loads(config_path.read_text())
    output = Path(config["output_dir"]).resolve()
    if output.exists():
        raise FileExistsError(f"output directory exists: {output}")
    output.mkdir(parents=True)
    score_tables = []
    sensitivity_score_tables = []
    biological_tables = []
    source_records = {}
    for dataset, spec in sorted(config["datasets"].items()):
        cb = pd.read_csv(spec["cytobridge_csv"])
        cb = cb[np.isclose(pd.to_numeric(cb["time"]), spec["terminal_time"])]
        cb = cb.rename(columns={spec["cytobridge_score"]: "score"})
        cell_types = sorted(
            set(cb["sender_type"].astype(str)) | set(cb["receiver_type"].astype(str))
        )
        cc = pd.read_csv(spec["cellchat_csv"])
        cc = cc[np.isclose(pd.to_numeric(cc["time"]), spec["terminal_time"])]
        cc = cc.rename(columns={spec["cellchat_score"]: "score"})
        cag_score = spec.get("cellagentchat_score", "cellagentchat_native_ctps")
        cag = pd.read_csv(Path(spec["cellagentchat_dir"]) / "directed_pair_scores.csv")
        if cag_score not in cag.columns:
            raise ValueError(
                f"CellAgentChat score column {cag_score!r} is missing for {dataset}"
            )
        cag = cag.rename(columns={cag_score: "score"})
        nn, detail = _nichenet_pair_scores(Path(spec["nichenet_dir"]))
        nn = nn.rename(columns={"nichenet_support_score": "score"})
        biological_tables.append(detail.assign(dataset=dataset))
        for method, frame in (
            ("CytoBridge", cb),
            ("CellChat", cc),
            ("CellAgentChat", cag),
            ("NicheNet", nn),
        ):
            completed = complete_directed_pairs(
                frame, score_column="score", cell_types=cell_types
            )
            completed.insert(0, "dataset", dataset)
            completed["method"] = method
            completed["rank_percentile"] = rank_percentile(completed["score"])
            score_tables.append(completed)
        if "cellagentchat_continuous_score" in cag.columns:
            cag_continuous = cag.rename(
                columns={"cellagentchat_continuous_score": "continuous_score"}
            )
            for method, frame, score_column in (
                ("CytoBridge", cb, "score"),
                ("CellChat", cc, "score"),
                ("CellAgentChat", cag_continuous, "continuous_score"),
                ("NicheNet", nn, "score"),
            ):
                completed = complete_directed_pairs(
                    frame, score_column=score_column, cell_types=cell_types
                ).rename(columns={score_column: "score"})
                completed.insert(0, "dataset", dataset)
                completed["method"] = method
                completed["rank_percentile"] = rank_percentile(completed["score"])
                sensitivity_score_tables.append(completed)
        source_files = {
            "cytobridge_csv": spec["cytobridge_csv"],
            "cellchat_csv": spec["cellchat_csv"],
            "cellagentchat_manifest": str(
                Path(spec["cellagentchat_dir"]) / "manifest.json"
            ),
            "nichenet_manifest": str(Path(spec["nichenet_dir"]) / "manifest.json"),
            "nichenet_ligand_target_matrix": spec["nichenet_ligand_target_matrix"],
            "nichenet_ligand_activities": str(
                Path(spec["nichenet_dir"]) / "official/ligand_activities.csv"
            ),
            "nichenet_ligand_target_links": str(
                Path(spec["nichenet_dir"]) / "official/ligand_target_links.csv"
            ),
            "nichenet_r_session": str(
                Path(spec["nichenet_dir"]) / "official/R_sessionInfo.txt"
            ),
            "nichenet_r_runner": spec["nichenet_r_runner"],
        }
        if spec.get("shared_lr_database_manifest"):
            source_files["shared_lr_database_manifest"] = spec[
                "shared_lr_database_manifest"
            ]
        source_records[dataset] = {
            key: {"path": str(Path(value).resolve()), "sha256": sha256_file(value)}
            for key, value in source_files.items()
        }
        source_records[dataset]["cellagentchat_score"] = cag_score
    scores = pd.concat(score_tables, ignore_index=True)
    metrics = pairwise_rank_metrics(scores)
    sensitivity_scores = (
        pd.concat(sensitivity_score_tables, ignore_index=True)
        if sensitivity_score_tables
        else pd.DataFrame()
    )
    sensitivity_metrics = (
        pairwise_rank_metrics(sensitivity_scores)
        if not sensitivity_scores.empty
        else pd.DataFrame()
    )
    wide = scores.pivot(
        index=["dataset", "sender_type", "receiver_type"],
        columns="method",
        values="rank_percentile",
    ).reset_index()
    wide["consensus_rank"] = wide[list(METHODS)].mean(axis=1)
    for method in METHODS:
        wide[f"{method}_top"] = wide.groupby("dataset")[method].transform(
            lambda values: values >= values.quantile(1 - TOP_FRACTION)
        )
    wide["external_top_support"] = wide[
        [f"{method}_top" for method in METHODS if method != "CytoBridge"]
    ].sum(axis=1)
    biology = pd.concat(biological_tables, ignore_index=True)
    scores.to_csv(output / "directed_pair_method_scores.csv", index=False)
    metrics.to_csv(output / "pairwise_rank_metrics.csv", index=False)
    if not sensitivity_scores.empty:
        sensitivity_scores.to_csv(
            output / "directed_pair_method_scores_cellagentchat_continuous.csv",
            index=False,
        )
        sensitivity_metrics.to_csv(
            output / "pairwise_rank_metrics_cellagentchat_continuous.csv", index=False
        )
    wide.sort_values(
        ["dataset", "consensus_rank"], ascending=[True, False], kind="mergesort"
    ).to_csv(output / "directed_pair_consensus.csv", index=False)
    biology.sort_values(
        ["dataset", "ligand_target_evidence"], ascending=[True, False], kind="mergesort"
    ).to_csv(output / "nichenet_ligand_target_evidence.csv", index=False)
    manifest = {
        "schema_version": 1,
        "claim_scope": "shared-input cross-method computational consistency; not causal or independent validation",
        "comparison_unit": "complete directed sender/receiver cell-type pair grid",
        "comparison_scale": "within-method rank percentile",
        "top_fraction": TOP_FRACTION,
        "database_contract": config.get("database_contract", "method-native"),
        "cellagentchat_primary": "native CTPS (sum of significant LR interaction scores)",
        "cellagentchat_sensitivity": "threshold-free sum of raw LR interaction scores",
        "aggregate_config": {
            "path": str(config_path),
            "sha256": sha256_file(config_path),
        },
        "sources": source_records,
        "outputs": {},
    }
    for path in sorted(output.glob("*.csv")):
        manifest["outputs"][path.name] = {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    _write_json(output / "manifest.json", manifest)


DATASET_LABELS = {"weinreb": "Weinreb", "scnt_cortex": "scNT cortex"}
DATASET_STYLE = {
    "weinreb": ("#59616A", "o"),
    "scnt_cortex": ("#CC6677", "s"),
}
EXTERNAL_METHODS = ("CellChat", "CellAgentChat", "NicheNet")
SELECTED_INTERACTIONS = (
    ("weinreb", "Baso", "Monocyte", "CSF1–CSF1R"),
    ("weinreb", "Monocyte", "Neutrophil", "TNF–TNFRSF1A"),
    ("scnt_cortex", "Ex", "EX-NP2", "BDNF–NTRK2"),
    ("scnt_cortex", "EX-NP2", "EX-NP1", "DLL1/JAG1–NOTCH1"),
)
BIOLOGICAL_PROGRAMS = (
    {
        "program_id": "weinreb_csf1",
        "dataset": "weinreb",
        "sender": "Baso",
        "receiver": "Monocyte",
        "ligands": ("Csf1",),
        "receptor": "Csf1r",
        "targets": ("Gpnmb", "Ctsb", "Dab2", "Ctsd"),
        "label": "Baso → Monocyte\nCSF1–CSF1R · myeloid maturation",
        "color": "#07838B",
    },
    {
        "program_id": "weinreb_tnf",
        "dataset": "weinreb",
        "sender": "Monocyte",
        "receiver": "Neutrophil",
        "ligands": ("Tnf",),
        "receptor": "Tnfrsf1a",
        "targets": ("Nfkbia", "Plaur", "Noct"),
        "label": "Monocyte → Neutrophil\nTNF–TNFRSF1A · inflammatory remodeling",
        "color": "#D55E62",
    },
    {
        "program_id": "scnt_bdnf",
        "dataset": "scnt_cortex",
        "sender": "Ex",
        "receiver": "EX-NP2",
        "ligands": ("Bdnf",),
        "receptor": "Ntrk2",
        "targets": ("Egr1", "Gadd45g", "Trib2", "Coro1c"),
        "label": "Ex → EX-NP2\nBDNF–NTRK2 · activity-response program",
        "color": "#07838B",
    },
    {
        "program_id": "scnt_notch",
        "dataset": "scnt_cortex",
        "sender": "EX-NP2",
        "receiver": "EX-NP1",
        "ligands": ("Dll1", "Jag1"),
        "receptor": "Notch1",
        "targets": ("Hes5",),
        "label": "EX-NP2 → EX-NP1\nDLL1/JAG1–NOTCH1 · progenitor-state program",
        "color": "#D55E62",
    },
)


def _panel_heading(axis, figure_style, label: str, title: str) -> None:
    figure_style.panel_heading(axis, label, title)
    for text in axis.texts:
        text.set_color("black")


def _agreement_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    result = metrics[
        (metrics.left_method == "CytoBridge")
        & metrics.right_method.isin(EXTERNAL_METHODS)
    ].copy()
    if len(result) != 6:
        raise ValueError("expected six CytoBridge-to-external metric rows")
    result["expected_top_intersection"] = (
        result["top_k"] ** 2 / result["n_directed_pairs"]
    )
    result["top_overlap_enrichment"] = (
        result["top_k_intersection"] / result["expected_top_intersection"]
    )
    result["top_overlap_pvalue"] = [
        float(
            hypergeom.sf(
                int(row.top_k_intersection) - 1,
                int(row.n_directed_pairs),
                int(row.top_k),
                int(row.top_k),
            )
        )
        for row in result.itertuples()
    ]
    result["dataset_label"] = result.dataset.map(DATASET_LABELS)
    return result.sort_values(["right_method", "dataset"], kind="mergesort")


def _harmonization_summary(
    native_metrics: pd.DataFrame, shared_metrics: pd.DataFrame
) -> pd.DataFrame:
    keys = ["dataset", "left_method", "right_method"]
    methods = ("CellAgentChat", "NicheNet")
    native = native_metrics[
        (native_metrics.left_method == "CytoBridge")
        & native_metrics.right_method.isin(methods)
    ][keys + ["spearman_rho", "top_k_jaccard"]].rename(
        columns={
            "spearman_rho": "native_spearman",
            "top_k_jaccard": "native_jaccard",
        }
    )
    shared = shared_metrics[
        (shared_metrics.left_method == "CytoBridge")
        & shared_metrics.right_method.isin(methods)
    ][keys + ["spearman_rho", "top_k_jaccard"]].rename(
        columns={
            "spearman_rho": "shared_spearman",
            "top_k_jaccard": "shared_jaccard",
        }
    )
    result = native.merge(shared, on=keys, validate="one_to_one")
    if len(result) != 4:
        raise ValueError("expected four database-harmonization rows")
    result["spearman_change"] = result.shared_spearman - result.native_spearman
    result["jaccard_change"] = result.shared_jaccard - result.native_jaccard
    result["dataset_label"] = result.dataset.map(DATASET_LABELS)
    return result.sort_values(["dataset", "right_method"], kind="mergesort")


def _selected_interaction_ranks(consensus: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dataset, sender, receiver, program in SELECTED_INTERACTIONS:
        selected = consensus[
            (consensus.dataset == dataset)
            & (consensus.sender_type == sender)
            & (consensus.receiver_type == receiver)
        ]
        if len(selected) != 1:
            raise ValueError(
                f"missing unique selected interaction {dataset}/{sender}->{receiver}"
            )
        source = selected.iloc[0]
        for method in METHODS:
            rows.append(
                {
                    "dataset": dataset,
                    "dataset_label": DATASET_LABELS[dataset],
                    "sender_type": sender,
                    "receiver_type": receiver,
                    "program": program,
                    "method": method,
                    "rank_percentile": float(source[method]),
                    "external_top_support": int(source.external_top_support),
                }
            )
    return pd.DataFrame(rows)


def _selected_biological_programs(biology: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for program_order, spec in enumerate(BIOLOGICAL_PROGRAMS):
        selected = biology[
            (biology.dataset == spec["dataset"])
            & (biology.sender == spec["sender"])
            & (biology.receiver == spec["receiver"])
            & biology.ligand.isin(spec["ligands"])
            & (biology.receptor == spec["receptor"])
            & biology.target.isin(spec["targets"])
        ].copy()
        selected = selected.sort_values(
            "ligand_target_evidence", ascending=False, kind="mergesort"
        ).drop_duplicates(["ligand", "target"])
        expected = {
            (ligand, target)
            for ligand in spec["ligands"]
            for target in spec["targets"]
            if not (len(spec["ligands"]) > 1 and target != "Hes5")
        }
        observed = set(zip(selected.ligand, selected.target, strict=False))
        if not expected.issubset(observed):
            raise ValueError(
                f"biological program {spec['program_id']} lacks {sorted(expected - observed)}"
            )
        selected["program_id"] = spec["program_id"]
        selected["program_label"] = spec["label"]
        selected["program_color"] = spec["color"]
        selected["program_order"] = program_order
        target_order = {target: index for index, target in enumerate(spec["targets"])}
        ligand_order = {ligand: index for index, ligand in enumerate(spec["ligands"])}
        selected["target_order"] = selected.target.map(target_order)
        selected["ligand_order"] = selected.ligand.map(ligand_order)
        selected["display_target"] = np.where(
            len(spec["ligands"]) > 1,
            selected.ligand.astype(str) + "→" + selected.target.astype(str),
            selected.target.astype(str),
        )
        rows.append(selected)
    return pd.concat(rows, ignore_index=True).sort_values(
        ["program_order", "target_order", "ligand_order"], kind="mergesort"
    )


def _plot_global_agreement(axis_rho, axis_overlap, summary, figure_style) -> None:
    y_base = {method: value for method, value in zip(EXTERNAL_METHODS, (2, 1, 0))}
    offsets = {"weinreb": 0.13, "scnt_cortex": -0.13}
    for dataset, (color, marker) in DATASET_STYLE.items():
        subset = summary[summary.dataset == dataset].set_index("right_method")
        for method in EXTERNAL_METHODS:
            row = subset.loc[method]
            y = y_base[method] + offsets[dataset]
            axis_rho.scatter(
                row.spearman_rho,
                y,
                s=42,
                marker=marker,
                color=color,
                edgecolor="white",
                linewidth=0.5,
                zorder=3,
            )
            axis_rho.text(
                row.spearman_rho + 0.035,
                y,
                f"{row.spearman_rho:.2f}",
                va="center",
                ha="left",
                fontsize=7.2,
            )
            significant = row.top_overlap_pvalue < 0.05
            axis_overlap.scatter(
                row.top_overlap_enrichment,
                y,
                s=42,
                marker=marker,
                facecolor=color if significant else "white",
                edgecolor=color,
                linewidth=1.1,
                zorder=3,
            )
            axis_overlap.text(
                row.top_overlap_enrichment + 0.09,
                y,
                f"{row.top_overlap_enrichment:.1f}×",
                va="center",
                ha="left",
                fontsize=7.2,
            )
    for axis in (axis_rho, axis_overlap):
        axis.set_yticks([2, 1, 0], ["CellChat", "CellAgentChat", "NicheNet"])
        axis.set_ylim(-0.45, 2.45)
        figure_style.clean_axis(axis, grid=True)
    axis_rho.axvline(0, color="#8A949C", linewidth=0.8, zorder=1)
    axis_rho.set_xlim(-0.57, 0.90)
    axis_rho.set_xlabel("Spearman rank correlation (ρ)")
    axis_rho.set_title("Complete directed-pair ranks", color="black", pad=4)
    axis_overlap.axvline(1, color="#8A949C", linewidth=0.8, zorder=1)
    axis_overlap.set_xlim(0, 3.85)
    axis_overlap.set_xlabel("Top-20% overlap / random expectation")
    axis_overlap.set_title(
        "CytoBridge top-edge enrichment\nfilled: exact P < 0.05",
        color="black",
        pad=4,
    )
    axis_rho.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker=DATASET_STYLE[dataset][1],
                color="none",
                markerfacecolor=DATASET_STYLE[dataset][0],
                markeredgecolor="white",
                markersize=6,
                label=DATASET_LABELS[dataset],
            )
            for dataset in ("weinreb", "scnt_cortex")
        ],
        loc="upper left",
        frameon=False,
    )


def _plot_harmonization(axis, summary, figure_style) -> None:
    display_order = [
        ("weinreb", "CellAgentChat"),
        ("weinreb", "NicheNet"),
        ("scnt_cortex", "CellAgentChat"),
        ("scnt_cortex", "NicheNet"),
    ]
    labels = []
    for y, (dataset, method) in enumerate(reversed(display_order)):
        row = summary[
            (summary.dataset == dataset) & (summary.right_method == method)
        ].iloc[0]
        color = PALETTE[method]
        axis.annotate(
            "",
            xy=(row.shared_spearman, y),
            xytext=(row.native_spearman, y),
            arrowprops={
                "arrowstyle": "-|>",
                "color": color,
                "linewidth": 1.8,
                "mutation_scale": 10,
            },
        )
        axis.scatter(
            row.native_spearman,
            y,
            s=38,
            facecolor="white",
            edgecolor=color,
            linewidth=1.2,
            zorder=3,
        )
        axis.scatter(
            row.shared_spearman,
            y,
            s=38,
            facecolor=color,
            edgecolor="white",
            linewidth=0.5,
            zorder=4,
        )
        axis.text(
            max(row.native_spearman, row.shared_spearman) + 0.04,
            y,
            f"Δρ {row.spearman_change:+.2f}",
            va="center",
            fontsize=7.5,
        )
        labels.append(f"{DATASET_LABELS[dataset]}\n{method}")
    axis.set_yticks(range(len(labels)), labels)
    axis.axvline(0, color="#8A949C", linewidth=0.8)
    axis.set_xlim(-0.58, 0.72)
    axis.set_xlabel("CytoBridge Spearman correlation (ρ)")
    axis.text(
        0.01,
        1.04,
        "open: method-native database     filled: shared CellChatDB",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.4,
        color="#59616A",
    )
    figure_style.clean_axis(axis, grid=True)


def _plot_selected_interactions(axis, selected, figure_style) -> None:
    method_order = list(METHODS)
    row_keys = list(SELECTED_INTERACTIONS)
    labels = []
    cmap = plt.get_cmap("GnBu")
    for row_index, (dataset, sender, receiver, program) in enumerate(row_keys):
        labels.append(f"{DATASET_LABELS[dataset]}  ·  {sender} → {receiver}\n{program}")
        subset = selected[
            (selected.dataset == dataset)
            & (selected.sender_type == sender)
            & (selected.receiver_type == receiver)
        ].set_index("method")
        for method_index, method in enumerate(method_order):
            value = float(subset.loc[method, "rank_percentile"])
            axis.scatter(
                method_index,
                row_index,
                s=120 + 260 * value,
                color=cmap(0.15 + 0.8 * value),
                edgecolor="white",
                linewidth=0.8,
                zorder=3,
            )
            axis.text(
                method_index,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=7.1,
                color="white" if value >= 0.65 else "#24313A",
                fontweight="bold" if value >= 0.8 else "normal",
                zorder=4,
            )
    axis.set_xticks(range(len(method_order)), method_order)
    axis.xaxis.tick_top()
    axis.tick_params(axis="x", pad=3)
    axis.set_yticks(range(len(labels)), labels, fontsize=7.7)
    axis.set_ylim(len(labels) - 0.5, -0.5)
    axis.set_xlim(-0.55, len(method_order) - 0.45)
    axis.axhline(1.5, color="#B7C0C7", linewidth=0.8)
    axis.set_xlabel("Within-method directed-pair rank percentile")
    axis.xaxis.set_label_position("bottom")
    axis.tick_params(axis="x", bottom=False, labelbottom=False, top=True, labeltop=True)
    axis.grid(axis="x", color=figure_style.GRID_COLOR, linewidth=0.45, alpha=0.6)
    for side in axis.spines.values():
        side.set_visible(False)


def _plot_biological_programs(axis, selected, dataset, figure_style) -> None:
    subset = selected[selected.dataset == dataset].copy()
    program_ids = subset.sort_values("program_order").program_id.drop_duplicates()
    y_positions = []
    y_labels = []
    cursor = 0.0
    max_value = float(subset.ligand_target_evidence.max())
    for program_id in program_ids:
        program = subset[subset.program_id == program_id].sort_values(
            ["target_order", "ligand_order"], kind="mergesort"
        )
        header_y = cursor
        axis.text(
            0,
            header_y,
            str(program.program_label.iloc[0]),
            color=str(program.program_color.iloc[0]),
            fontsize=7.8,
            fontweight="bold",
            ha="left",
            va="center",
        )
        cursor += 0.90
        for row in program.itertuples():
            y_positions.append(cursor)
            y_labels.append(str(row.display_target))
            axis.hlines(
                cursor,
                0,
                row.ligand_target_evidence,
                color=row.program_color,
                linewidth=2.2,
                alpha=0.52,
                zorder=1,
            )
            axis.scatter(
                row.ligand_target_evidence,
                cursor,
                s=34,
                color=row.program_color,
                edgecolor="white",
                linewidth=0.5,
                zorder=3,
            )
            cursor += 0.62
        cursor += 0.38
    axis.set_yticks(y_positions, y_labels, fontsize=7.4)
    axis.set_ylim(cursor - 0.15, -0.4)
    axis.set_xlim(0, max_value * 1.14)
    axis.set_xlabel("NicheNet ligand–target evidence")
    axis.set_title(DATASET_LABELS[dataset], color="black", pad=4)
    figure_style.clean_axis(axis, grid=True)


def plot_figure(args: argparse.Namespace) -> None:
    from CytoBridge.nonspatial import scnt_figure_style as figure_style

    source = Path(args.source_dir).resolve()
    native_source = Path(args.native_source_dir).resolve()
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"output directory exists: {output}")
    output.mkdir(parents=True)
    metrics = pd.read_csv(source / "pairwise_rank_metrics.csv")
    native_metrics = pd.read_csv(native_source / "pairwise_rank_metrics.csv")
    consensus = pd.read_csv(source / "directed_pair_consensus.csv")
    biology = pd.read_csv(source / "nichenet_ligand_target_evidence.csv")
    agreement = _agreement_summary(metrics)
    harmonization = _harmonization_summary(native_metrics, metrics)
    selected_ranks = _selected_interaction_ranks(consensus)
    selected_biology = _selected_biological_programs(biology)
    figure_style.apply_style()
    fig = plt.figure(figsize=figure_style.A4_PORTRAIT, constrained_layout=False)
    outer = fig.add_gridspec(
        8,
        1,
        height_ratios=[0.24, 1.42, 0.24, 1.10, 0.24, 1.75, 0.24, 2.20],
        left=0.22,
        right=0.96,
        top=0.975,
        bottom=0.06,
        hspace=0.46,
    )

    ax_ah = fig.add_subplot(outer[0])
    _panel_heading(
        ax_ah, figure_style, "a", "Global agreement is method- and dataset-dependent"
    )
    grid_a = outer[1].subgridspec(1, 2, wspace=0.42)
    ax_a1 = fig.add_subplot(grid_a[0, 0])
    ax_a2 = fig.add_subplot(grid_a[0, 1])
    _plot_global_agreement(ax_a1, ax_a2, agreement, figure_style)

    ax_bh = fig.add_subplot(outer[2])
    _panel_heading(
        ax_bh, figure_style, "b", "A shared LR universe improves external concordance"
    )
    ax_b = fig.add_subplot(outer[3])
    _plot_harmonization(ax_b, harmonization, figure_style)

    ax_ch = fig.add_subplot(outer[4])
    _panel_heading(
        ax_ch,
        figure_style,
        "c",
        "High-confidence communication programs recur across methods",
    )
    ax_c = fig.add_subplot(outer[5])
    _plot_selected_interactions(ax_c, selected_ranks, figure_style)

    ax_dh = fig.add_subplot(outer[6])
    _panel_heading(
        ax_dh,
        figure_style,
        "d",
        "Lineage-relevant ligand–target programs support selected interactions",
    )
    grid_d = outer[7].subgridspec(1, 2, wspace=0.48)
    ax_d1 = fig.add_subplot(grid_d[0, 0])
    ax_d2 = fig.add_subplot(grid_d[0, 1])
    _plot_biological_programs(ax_d1, selected_biology, "weinreb", figure_style)
    _plot_biological_programs(ax_d2, selected_biology, "scnt_cortex", figure_style)

    pdf = output / "nonspatial_communication_consistency_a4.pdf"
    png = output / "nonspatial_communication_consistency_a4.png"
    figure_style.save_figure(fig, pdf, png, dpi=320)
    plt.close(fig)

    agreement.to_csv(output / "global_agreement_summary.csv", index=False)
    harmonization.to_csv(output / "database_harmonization_summary.csv", index=False)
    selected_ranks.to_csv(output / "selected_program_method_ranks.csv", index=False)
    selected_biology.to_csv(output / "selected_biological_programs.csv", index=False)

    def _metric(dataset: str, method: str, field: str) -> float:
        row = metrics[
            (metrics.dataset == dataset)
            & (metrics.left_method == "CytoBridge")
            & (metrics.right_method == method)
        ]
        if len(row) != 1:
            raise ValueError(f"missing unique metric for {dataset}/{method}")
        return float(row.iloc[0][field])

    caption = (
        "**Non-spatial communication support under a shared CellChatDB ligand–receptor universe.** "
        "(a) CytoBridge agreement with CellChat, official non-spatial CellAgentChat CTPS, and sender-focused NicheNet on "
        "complete directed cell-type-pair ranks (left) and enrichment of top-20% overlap over the random-set expectation "
        "(right). Filled overlap symbols pass the exact hypergeometric P<0.05 set-enrichment threshold. "
        "(b) Spearman agreement before and after placing CellAgentChat and NicheNet on the same package-bundled mouse "
        "CellChatDB LR universe used by CytoBridge and CellChat. Database harmonization improves all four comparisons but "
        "does not eliminate the weak Weinreb CellAgentChat rank agreement. "
        "(c) Within-method rank percentiles for four preselected, externally supported communication programs. "
        "(d) NicheNet-supported ligand-target responses for the same cell-type programs: CSF1-CSF1R and TNF-TNFRSF1A "
        "in Weinreb myeloid differentiation, and BDNF-NTRK2 plus DLL1/JAG1-NOTCH1 in the scNT cortical progenitor response. "
        "Evidence combines within-receiver ligand-activity rank, sender ligand expression, receiver receptor expression, "
        "and the NicheNet ligand-target weight. "
        "CytoBridge–CellChat Spearman rho was "
        f"{_metric('weinreb', 'CellChat', 'spearman_rho'):.3f} in Weinreb and "
        f"{_metric('scnt_cortex', 'CellChat', 'spearman_rho'):.3f} in scNT; corresponding top-20% Jaccard values were "
        f"{_metric('weinreb', 'CellChat', 'top_k_jaccard'):.3f} and "
        f"{_metric('scnt_cortex', 'CellChat', 'top_k_jaccard'):.3f}. "
        "All four methods used the same package-bundled mouse CellChatDB source; raw method scores were never pooled, and "
        "the comparison is descriptive rather than causal or independent experimental validation."
    )
    (output / "caption.md").write_text(caption + "\n")
    provenance_note = (
        "# Non-spatial communication consistency figure provenance\n\n"
        "## Source paths\n\n"
        f"- Aggregate manifest: `{(source / 'manifest.json').resolve()}`\n"
        f"- Native-database manifest: `{(native_source / 'manifest.json').resolve()}`\n"
        f"- Directed-pair scores: `{(source / 'directed_pair_method_scores.csv').resolve()}`\n"
        f"- Pairwise metrics: `{(source / 'pairwise_rank_metrics.csv').resolve()}`\n"
        f"- NicheNet evidence: `{(source / 'nichenet_ligand_target_evidence.csv').resolve()}`\n\n"
        "## Rebuild\n\n"
        "```bash\n"
        "python scripts/run_nonspatial_communication_consistency.py plot \\\n"
        f"  --source-dir {source} \\\n"
        f"  --native-source-dir {native_source} \\\n"
        "  --output-dir <new-empty-output-directory>\n"
        "```\n\n"
        "## SHA-256\n\n"
        f"- Aggregate manifest: `{sha256_file(source / 'manifest.json')}`\n"
        f"- Native-database manifest: `{sha256_file(native_source / 'manifest.json')}`\n"
        f"- PDF: `{sha256_file(pdf)}`\n"
        f"- PNG: `{sha256_file(png)}`\n"
        f"- Caption: `{sha256_file(output / 'caption.md')}`\n"
    )
    (output / "provenance.md").write_text(provenance_note)
    provenance = {
        "schema_version": 2,
        "source_manifest": {
            "path": str((source / "manifest.json").resolve()),
            "sha256": sha256_file(source / "manifest.json"),
        },
        "native_source_manifest": {
            "path": str((native_source / "manifest.json").resolve()),
            "sha256": sha256_file(native_source / "manifest.json"),
        },
        "figure": {
            pdf.name: {"sha256": sha256_file(pdf), "size_bytes": pdf.stat().st_size},
            png.name: {"sha256": sha256_file(png), "size_bytes": png.stat().st_size},
        },
        "caption_sha256": sha256_file(output / "caption.md"),
        "provenance_note_sha256": sha256_file(output / "provenance.md"),
        "claim_scope": "shared-input computational consistency; descriptive, non-causal",
    }
    _write_json(output / "figure_manifest.json", provenance)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    cag = sub.add_parser("cellagentchat")
    cag.add_argument("--dataset", required=True)
    cag.add_argument("--input-h5ad", required=True)
    cag.add_argument("--cell-type-key", required=True)
    cag.add_argument("--time-key", required=True)
    cag.add_argument("--terminal-time", type=float, required=True)
    cag.add_argument("--cellagentchat-source", required=True)
    cag.add_argument("--lr-database")
    cag.add_argument("--output-dir", required=True)
    cag.add_argument("--sample-n", type=int, default=3000)
    cag.add_argument("--epochs", type=int, default=50)
    cag.add_argument("--batch-size", type=int, default=256)
    cag.add_argument("--learning-rate", type=float, default=0.1)
    cag.add_argument("--permutation-threshold", type=float, default=1000)
    cag.add_argument("--device", default="cuda:0")
    cag.set_defaults(function=run_cellagentchat)

    shared_lr = sub.add_parser("prepare-shared-lr")
    shared_lr.add_argument("--database", required=True)
    shared_lr.add_argument("--output-dir", required=True)
    shared_lr.set_defaults(function=prepare_shared_lr)

    summarize_cag = sub.add_parser("summarize-cellagentchat")
    summarize_cag.add_argument("--cellagentchat-dir", required=True)
    summarize_cag.add_argument("--output-dir", required=True)
    summarize_cag.set_defaults(function=summarize_cellagentchat)

    prepare = sub.add_parser("prepare-nichenet")
    prepare.add_argument("--dataset", required=True)
    prepare.add_argument("--input-h5ad", required=True)
    prepare.add_argument("--cell-type-key", required=True)
    prepare.add_argument("--time-key", required=True)
    prepare.add_argument("--terminal-time", type=float, required=True)
    prepare.add_argument("--previous-time", type=float, required=True)
    prepare.add_argument("--lr-network-csv", required=True)
    prepare.add_argument("--nichenetr-source", required=True)
    prepare.add_argument("--output-dir", required=True)
    prepare.set_defaults(function=prepare_nichenet)

    aggregate_parser = sub.add_parser("aggregate")
    aggregate_parser.add_argument("--config", required=True)
    aggregate_parser.set_defaults(function=aggregate)

    plot_parser = sub.add_parser("plot")
    plot_parser.add_argument("--source-dir", required=True)
    plot_parser.add_argument("--native-source-dir", required=True)
    plot_parser.add_argument("--output-dir", required=True)
    plot_parser.set_defaults(function=plot_figure)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
