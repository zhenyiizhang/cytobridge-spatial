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
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib.lines import Line2D

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


def _correlation_heatmap(axis, metrics: pd.DataFrame, dataset: str) -> None:
    matrix = pd.DataFrame(np.eye(len(METHODS)), index=METHODS, columns=METHODS)
    for row in metrics[metrics.dataset == dataset].itertuples():
        matrix.loc[row.left_method, row.right_method] = row.spearman_rho
        matrix.loc[row.right_method, row.left_method] = row.spearman_rho
    sns.heatmap(
        matrix,
        ax=axis,
        cmap="vlag",
        center=0,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        annot_kws={"fontsize": 8.0},
        cbar=False,
        square=True,
        linewidths=0.4,
        linecolor="white",
    )
    axis.set_title(
        dataset.replace("weinreb", "Weinreb").replace("scnt_cortex", "scNT cortex"),
        pad=4,
        color="black",
    )
    axis.set_xticklabels(
        ["Cyto-\nBridge", "CellChat", "CellAgent\nChat", "NicheNet"],
        rotation=30,
        ha="right",
        rotation_mode="anchor",
        fontsize=7.2,
    )
    axis.tick_params(axis="x", pad=1)
    axis.tick_params(axis="y", rotation=0)
    axis.set_xlabel("")
    axis.set_ylabel("")


def _network(axis, table: pd.DataFrame, dataset: str, figure_style) -> None:
    subset = table[table.dataset == dataset].copy()
    subset = subset[subset.sender_type != subset.receiver_type]
    subset = subset[subset.external_top_support >= 1]
    subset = subset.sort_values(
        ["external_top_support", "consensus_rank"], ascending=False, kind="mergesort"
    ).head(11)
    graph = nx.DiGraph()
    for row in subset.itertuples():
        graph.add_edge(
            row.sender_type,
            row.receiver_type,
            width=0.8 + 2.6 * row.CytoBridge,
            support=int(row.external_top_support),
        )
    nodes = sorted(graph.nodes())
    positions = nx.circular_layout(graph)
    nx.draw_networkx_nodes(
        graph,
        positions,
        node_size=620,
        node_color="#E5F1F2",
        edgecolors=figure_style.CYTOBRIDGE_COLOR,
        linewidths=0.7,
        ax=axis,
    )
    nx.draw_networkx_labels(graph, positions, font_size=7.2, ax=axis)
    for support, color in (
        (1, "#E9C46A"),
        (2, "#F4A261"),
        (3, "#D1495B"),
    ):
        edges = [
            (u, v) for u, v, d in graph.edges(data=True) if d["support"] == support
        ]
        if edges:
            nx.draw_networkx_edges(
                graph,
                positions,
                edgelist=edges,
                width=[graph[u][v]["width"] for u, v in edges],
                edge_color=color,
                alpha=0.82,
                arrows=True,
                arrowsize=9,
                connectionstyle="arc3,rad=0.08",
                ax=axis,
            )
    axis.set_title(
        dataset.replace("weinreb", "Weinreb").replace("scnt_cortex", "scNT cortex"),
        color="black",
    )
    axis.axis("off")


def _top_edge_heatmap(axis, consensus: pd.DataFrame) -> None:
    selected = (
        consensus.sort_values(
            ["dataset", "CytoBridge"], ascending=[True, False], kind="mergesort"
        )
        .groupby("dataset", sort=False)
        .head(5)
    )
    labels = [
        f"{'S' if row.dataset == 'scnt_cortex' else 'W'}  {row.sender_type}→{row.receiver_type}"
        for row in selected.itertuples()
    ]
    sns.heatmap(
        selected[list(METHODS)].to_numpy(float),
        ax=axis,
        cmap="mako",
        vmin=0,
        vmax=1,
        yticklabels=labels,
        xticklabels=["Cyto-\nBridge", "CellChat", "CellAgent\nChat", "NicheNet"],
        cbar_kws={"label": "Rank percentile", "shrink": 0.72},
        linewidths=0.35,
        linecolor="white",
    )
    axis.tick_params(axis="x", rotation=0, pad=1, labelsize=7.6)
    axis.tick_params(axis="y", rotation=0, labelsize=8)
    axis.set_title("CytoBridge-leading interactions", color="black")


def _biological_support(
    axis, biology: pd.DataFrame, consensus: pd.DataFrame, dataset: str, figure_style
) -> None:
    top_pairs = (
        consensus[consensus.dataset == dataset]
        .sort_values(
            ["external_top_support", "consensus_rank"],
            ascending=False,
            kind="mergesort",
        )
        .head(12)
    )
    detail = biology[biology.dataset == dataset].merge(
        top_pairs[["sender_type", "receiver_type"]],
        left_on=["sender", "receiver"],
        right_on=["sender_type", "receiver_type"],
        how="inner",
    )
    detail = (
        detail.sort_values("ligand_target_evidence", ascending=False, kind="mergesort")
        .drop_duplicates(["sender", "receiver", "ligand", "target"])
        .head(7)
        .sort_values("ligand_target_evidence", ascending=True, kind="mergesort")
    )
    abbreviations = {
        "Monocyte": "Mono",
        "Neutrophil": "Neut.",
        "Undifferentiated": "Undiff.",
    }
    labels = [
        f"{row.ligand}→{row.target}\n"
        f"{abbreviations.get(row.sender, row.sender)}→"
        f"{abbreviations.get(row.receiver, row.receiver)}"
        for row in detail.itertuples()
    ]
    positions = np.arange(len(detail))
    axis.hlines(
        positions,
        0,
        detail.ligand_target_evidence,
        color="#9CC8CB",
        linewidth=2.2,
        zorder=1,
    )
    axis.scatter(
        detail.ligand_target_evidence,
        positions,
        s=34,
        color=figure_style.CYTOBRIDGE_COLOR,
        zorder=2,
    )
    axis.set_yticks(positions, labels, fontsize=7.2)
    axis.set_xlim(left=0)
    axis.set_xlabel("NicheNet ligand–target evidence")
    axis.set_title(
        dataset.replace("weinreb", "Weinreb").replace("scnt_cortex", "scNT cortex"),
        color="black",
    )
    figure_style.clean_axis(axis, grid=True)


def plot_figure(args: argparse.Namespace) -> None:
    from CytoBridge.nonspatial import scnt_figure_style as figure_style

    source = Path(args.source_dir).resolve()
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"output directory exists: {output}")
    output.mkdir(parents=True)
    scores = pd.read_csv(source / "directed_pair_method_scores.csv")
    metrics = pd.read_csv(source / "pairwise_rank_metrics.csv")
    consensus = pd.read_csv(source / "directed_pair_consensus.csv")
    biology = pd.read_csv(source / "nichenet_ligand_target_evidence.csv")
    figure_style.apply_style()
    fig = plt.figure(figsize=figure_style.A4_PORTRAIT, constrained_layout=False)
    outer = fig.add_gridspec(
        8,
        1,
        height_ratios=[0.24, 1.58, 0.42, 1.46, 0.42, 1.78, 0.34, 2.12],
        left=0.12,
        right=0.96,
        top=0.975,
        bottom=0.055,
        hspace=0.42,
    )

    ax_ah = fig.add_subplot(outer[0])
    figure_style.panel_heading(ax_ah, "a", "Directed-pair rank concordance")
    for text in ax_ah.texts:
        text.set_color("black")
    grid_a = outer[1].subgridspec(1, 2, wspace=0.55)
    ax_a1 = fig.add_subplot(grid_a[0, 0])
    ax_a2 = fig.add_subplot(grid_a[0, 1])
    _correlation_heatmap(ax_a1, metrics, "weinreb")
    _correlation_heatmap(ax_a2, metrics, "scnt_cortex")

    ax_bh = fig.add_subplot(outer[2])
    figure_style.panel_heading(ax_bh, "b", "CytoBridge top-edge agreement")
    for text in ax_bh.texts:
        text.set_color("black")
    grid_b = outer[3].subgridspec(1, 2, width_ratios=[0.82, 1.38], wspace=0.62)
    ax_b1 = fig.add_subplot(grid_b[0, 0])
    ax_b2 = fig.add_subplot(grid_b[0, 1])
    cb_metrics = metrics[metrics.left_method == "CytoBridge"].copy()
    cb_metrics["dataset_label"] = cb_metrics.dataset.replace(
        {"weinreb": "Weinreb", "scnt_cortex": "scNT cortex"}
    )
    method_order = ["CellChat", "CellAgentChat", "NicheNet"]
    dataset_style = {
        "Weinreb": ("#59616A", "o"),
        "scNT cortex": ("#CC6677", "s"),
    }
    x_positions = np.arange(len(method_order), dtype=float)
    for label, (color, marker) in dataset_style.items():
        subset = cb_metrics[cb_metrics.dataset_label == label].set_index("right_method")
        values = [subset.loc[method, "top_k_jaccard"] for method in method_order]
        ax_b1.plot(
            x_positions,
            values,
            marker=marker,
            markersize=5.5,
            color=color,
            linewidth=1.3,
            label=label,
        )
    ax_b1.set_xticks(x_positions, ["CellChat", "CellAgent\nChat", "NicheNet"])
    ax_b1.set_xlim(-0.18, 2.18)
    ax_b1.set_ylabel("Top-20% Jaccard")
    ax_b1.set_ylim(0, 1)
    figure_style.clean_axis(ax_b1, grid=True)
    ax_b1.legend(loc="upper right", handlelength=1.6)
    _top_edge_heatmap(ax_b2, consensus)

    ax_ch = fig.add_subplot(outer[4])
    figure_style.panel_heading(ax_ch, "c", "Shared directed interactions")
    for text in ax_ch.texts:
        text.set_color("black")
    support_colors = {1: "#E9C46A", 2: "#F4A261", 3: "#D1495B"}
    ax_ch.legend(
        handles=[
            Line2D([0], [0], color=color, lw=2.4, label=f"{support}/3 methods")
            for support, color in support_colors.items()
        ],
        loc="center right",
        ncol=3,
        handlelength=1.4,
        columnspacing=1.2,
    )
    grid_c = outer[5].subgridspec(1, 2, wspace=0.25)
    ax_c1 = fig.add_subplot(grid_c[0, 0])
    ax_c2 = fig.add_subplot(grid_c[0, 1])
    _network(ax_c1, consensus, "weinreb", figure_style)
    _network(ax_c2, consensus, "scnt_cortex", figure_style)

    ax_dh = fig.add_subplot(outer[6])
    figure_style.panel_heading(
        ax_dh, "d", "Ligand–target programs within shared interactions"
    )
    for text in ax_dh.texts:
        text.set_color("black")
    grid_d = outer[7].subgridspec(1, 2, wspace=0.82)
    ax_d1 = fig.add_subplot(grid_d[0, 0])
    ax_d2 = fig.add_subplot(grid_d[0, 1])
    _biological_support(ax_d1, biology, consensus, "weinreb", figure_style)
    _biological_support(ax_d2, biology, consensus, "scnt_cortex", figure_style)

    pdf = output / "nonspatial_communication_consistency_a4.pdf"
    png = output / "nonspatial_communication_consistency_a4.png"
    figure_style.save_figure(fig, pdf, png, dpi=320)
    plt.close(fig)

    biological = []
    for dataset in sorted(consensus.dataset.unique()):
        top_edges = (
            consensus[consensus.dataset == dataset]
            .sort_values(
                ["external_top_support", "consensus_rank"],
                ascending=False,
                kind="mergesort",
            )
            .head(12)
        )
        detail = biology[biology.dataset == dataset].merge(
            top_edges[
                [
                    "sender_type",
                    "receiver_type",
                    "consensus_rank",
                    "external_top_support",
                ]
            ],
            left_on=["sender", "receiver"],
            right_on=["sender_type", "receiver_type"],
            how="inner",
        )
        biological.append(detail.nlargest(12, "ligand_target_evidence"))
    biological = pd.concat(biological, ignore_index=True)
    biological.to_csv(output / "top_consensus_ligand_target_support.csv", index=False)

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
        "(a) Spearman correlations of complete directed cell-type-pair ranks for CytoBridge, CellChat, official "
        "non-spatial CellAgentChat, and sender-focused NicheNet in Weinreb and scNT cortex. "
        "(b) Jaccard overlap of each external method with the top 20% of CytoBridge interactions and rank profiles of "
        "the five leading CytoBridge interactions per dataset; W and S denote Weinreb and scNT. "
        "(c) Shared directed interactions. Edge width encodes the CytoBridge rank and color gives the number of external "
        "methods that also place the interaction in their top 20%. "
        "(d) Leading NicheNet ligand–target programs among shared cell-type interactions. Evidence combines within-receiver "
        "ligand-activity rank, sender ligand expression, receiver receptor expression, and the NicheNet ligand–target weight. "
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
        f"- Directed-pair scores: `{(source / 'directed_pair_method_scores.csv').resolve()}`\n"
        f"- Pairwise metrics: `{(source / 'pairwise_rank_metrics.csv').resolve()}`\n"
        f"- NicheNet evidence: `{(source / 'nichenet_ligand_target_evidence.csv').resolve()}`\n\n"
        "## Rebuild\n\n"
        "```bash\n"
        "python scripts/run_nonspatial_communication_consistency.py plot \\\n"
        f"  --source-dir {source} \\\n"
        "  --output-dir <new-empty-output-directory>\n"
        "```\n\n"
        "## SHA-256\n\n"
        f"- Aggregate manifest: `{sha256_file(source / 'manifest.json')}`\n"
        f"- PDF: `{sha256_file(pdf)}`\n"
        f"- PNG: `{sha256_file(png)}`\n"
        f"- Caption: `{sha256_file(output / 'caption.md')}`\n"
    )
    (output / "provenance.md").write_text(provenance_note)
    provenance = {
        "schema_version": 1,
        "source_manifest": {
            "path": str((source / "manifest.json").resolve()),
            "sha256": sha256_file(source / "manifest.json"),
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
    plot_parser.add_argument("--output-dir", required=True)
    plot_parser.set_defaults(function=plot_figure)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
