#!/usr/bin/env python3
"""Export reviewer-grade CytoBridge attention and exact spatial messages.

The script operates on observed, pre-warp states only.  Each observed stage is
partitioned with the same remainder-size contract as ``cal_interaction_gnn``;
technical grouping seeds are exported separately and averaged only after each
seed has produced a complete sender/receiver table.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


def _csv_ints(value: str) -> tuple[int, ...]:
    parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not parsed or len(set(parsed)) != len(parsed):
        raise argparse.ArgumentTypeError("Expected unique comma-separated integers.")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5ad", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--cell-type-key", default="Annotation")
    parser.add_argument("--time-key", default="time_point_processed")
    parser.add_argument("--time-label-key", default="time_label")
    parser.add_argument("--spatial-key", default="spatial_aligned")
    parser.add_argument("--state-key", default="X_latent")
    parser.add_argument("--group-size", type=int, default=1024)
    parser.add_argument(
        "--grouping-seeds", type=_csv_ints, default=(101, 202, 303, 404, 505)
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint-stage", default="Finetune")
    parser.add_argument(
        "--randomize-interaction-seed",
        type=int,
        help=(
            "Reset only the trainable spatial-interaction modules after loading "
            "the requested checkpoint. The frozen LR-informed edge predictor "
            "and fixed RBF basis remain unchanged."
        ),
    )
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
    path = path.expanduser().resolve()
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory is not empty: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": int(path.stat().st_size),
        "sha256": _sha256(path),
    }


def _stage_label(values: pd.Series) -> str:
    unique = values.astype(str).drop_duplicates().tolist()
    if len(unique) != 1:
        raise ValueError(f"A numeric time stage maps to multiple labels: {unique}.")
    return unique[0]


def _summarize_type_pairs(tables: Sequence[pd.DataFrame]) -> pd.DataFrame:
    combined = pd.concat(tables, ignore_index=True)
    keys = ["stage", "stage_label", "sender_type", "receiver_type"]
    numeric = [
        column
        for column in combined.columns
        if column not in {*keys, "grouping_seed"}
        and pd.api.types.is_numeric_dtype(combined[column])
    ]
    grouped = combined.groupby(keys, sort=True, dropna=False)
    summary = grouped[numeric].agg(["mean", "std", "min", "max"])
    summary.columns = [
        "_".join(str(item) for item in column if str(item))
        for column in summary.columns.to_flat_index()
    ]
    summary = summary.reset_index()
    summary["n_grouping_seeds"] = grouped.size().to_numpy()
    return summary


def _randomize_interaction_modules(interaction_net, *, seed: int) -> None:
    """Reset learned message modules while preserving graph construction."""
    import torch

    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    def reset(module) -> None:
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()

    for name in ("gene_embed", "distance_projection", "gnn_layers", "gene_readout"):
        getattr(interaction_net, name).apply(reset)


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    import anndata as ad
    import torch

    from CytoBridge.tl.downstream.checkpoint import load_dynamical_model_from_dir
    from CytoBridge.tl.downstream.spatial_interaction_attribution import (
        analyze_spatial_gnn_by_celltype,
        validate_spatial_exact_decomposition_model,
    )

    h5ad_path = args.h5ad.expanduser().resolve()
    model_dir = args.model_dir.expanduser().resolve()
    if not h5ad_path.is_file():
        raise FileNotFoundError(h5ad_path)
    if not model_dir.is_dir():
        raise FileNotFoundError(model_dir)
    if int(args.group_size) < 2:
        raise ValueError("--group-size must be at least two.")
    output = _prepare_output(args.output_dir, bool(args.overwrite))

    data = ad.read_h5ad(h5ad_path)
    for key in (args.cell_type_key, args.time_key, args.time_label_key):
        if key not in data.obs:
            raise KeyError(f"Missing adata.obs[{key!r}].")
    for key in (args.spatial_key, args.state_key):
        if key not in data.obsm:
            raise KeyError(f"Missing adata.obsm[{key!r}].")
    spatial = np.asarray(data.obsm[args.spatial_key], dtype=np.float32)
    state = np.asarray(data.obsm[args.state_key], dtype=np.float32)
    if spatial.shape != (data.n_obs, 2):
        raise ValueError(f"{args.spatial_key} must have shape (N, 2), got {spatial.shape}.")
    if state.ndim != 2 or state.shape[0] != data.n_obs:
        raise ValueError(f"{args.state_key} must be an N x D matrix.")
    if not np.isfinite(spatial).all() or not np.isfinite(state).all():
        raise ValueError("Model input contains non-finite values.")

    device = torch.device(args.device)
    feature_dim = int(2 + state.shape[1])
    loaded = load_dynamical_model_from_dir(
        model_dir,
        dim=feature_dim,
        device=device,
        stage=str(args.checkpoint_stage),
    )
    interaction_net = loaded.model.interaction_net
    validate_spatial_exact_decomposition_model(interaction_net)
    if args.randomize_interaction_seed is not None:
        _randomize_interaction_modules(
            interaction_net, seed=int(args.randomize_interaction_seed)
        )
    configured_group_size = int(
        getattr(loaded.model, "interaction_group_size", args.group_size)
    )
    if configured_group_size != int(args.group_size):
        raise ValueError(
            "Requested group size does not match checkpoint config: "
            f"requested={args.group_size}, configured={configured_group_size}."
        )

    cell_table = pd.DataFrame(
        {
            "global_index": np.arange(data.n_obs, dtype=int),
            "obs_name": data.obs_names.astype(str),
            "stage": pd.to_numeric(data.obs[args.time_key], errors="raise"),
            "stage_label": data.obs[args.time_label_key].astype(str).to_numpy(),
            "cell_type": data.obs[args.cell_type_key].astype(str).to_numpy(),
        }
    )
    cell_table_path = output / "observed_cells.csv.gz"
    cell_table.to_csv(cell_table_path, index=False, compression="gzip")

    type_tables: list[pd.DataFrame] = []
    reconstruction_tables: list[pd.DataFrame] = []
    run_records: list[dict[str, Any]] = []
    stages = np.sort(cell_table["stage"].unique().astype(float))
    for stage in stages:
        stage_mask = np.isclose(
            cell_table["stage"].to_numpy(float), stage, rtol=0.0, atol=1e-12
        )
        global_indices = np.flatnonzero(stage_mask)
        stage_label = _stage_label(cell_table.loc[stage_mask, "stage_label"])
        stage_x = np.hstack((spatial[global_indices], state[global_indices])).astype(
            np.float32
        )
        stage_labels = cell_table.loc[stage_mask, "cell_type"].to_numpy(str)
        tensor_x = torch.as_tensor(stage_x, device=device, dtype=torch.float32)
        lnw = torch.full(
            (global_indices.size, 1),
            -math.log(float(global_indices.size)),
            device=device,
            dtype=torch.float32,
        )
        stage_dir = output / f"stage_{stage:g}_{stage_label}"
        stage_dir.mkdir(parents=True, exist_ok=True)
        for seed in args.grouping_seeds:
            result = analyze_spatial_gnn_by_celltype(
                interaction_net,
                tensor_x,
                lnw,
                torch.tensor(float(stage), device=device),
                stage_labels,
                group_size=int(args.group_size),
                grouping_seed=int(seed),
            )
            type_pair = result.type_pair_table.copy()
            type_pair.insert(0, "stage_label", stage_label)
            type_pair.insert(0, "stage", float(stage))
            type_path = stage_dir / f"type_pair_seed_{seed}.csv"
            type_pair.to_csv(type_path, index=False)
            type_tables.append(type_pair)

            edge = result.edge_table.copy()
            edge["source_index_stage"] = edge["source_index"].astype(int)
            edge["target_index_stage"] = edge["target_index"].astype(int)
            edge["source_index"] = global_indices[edge["source_index_stage"]]
            edge["target_index"] = global_indices[edge["target_index_stage"]]
            edge.insert(0, "stage_label", stage_label)
            edge.insert(0, "stage", float(stage))
            edge_path = stage_dir / f"edges_seed_{seed}.csv.gz"
            edge.to_csv(edge_path, index=False, compression="gzip")

            group_id = np.full(global_indices.size, -1, dtype=np.int32)
            for group_index, group in enumerate(result.groups):
                group_id[group] = group_index
            array_path = stage_dir / f"exact_arrays_seed_{seed}.npz"
            np.savez_compressed(
                array_path,
                edge_output=result.edge_output,
                attention_signed=result.attention_signed,
                group_id_stage=group_id,
                global_indices=global_indices,
            )

            reconstruction = result.reconstruction_table.copy()
            reconstruction.insert(0, "stage_label", stage_label)
            reconstruction.insert(0, "stage", float(stage))
            reconstruction_tables.append(reconstruction)
            run_records.append(
                {
                    "stage": float(stage),
                    "stage_label": stage_label,
                    "grouping_seed": int(seed),
                    "n_cells": int(global_indices.size),
                    "n_edges": int(len(edge)),
                    "n_groups": int(len(result.groups)),
                    "max_abs_reconstruction_residual": float(
                        reconstruction["max_abs_residual"].max()
                    ),
                    "max_relative_l2_reconstruction_residual": float(
                        reconstruction["relative_l2_residual"].max()
                    ),
                    "artifacts": {
                        "type_pair": _artifact(type_path),
                        "edges": _artifact(edge_path),
                        "exact_arrays": _artifact(array_path),
                    },
                }
            )

    all_type_pairs = pd.concat(type_tables, ignore_index=True)
    all_type_pairs_path = output / "type_pair_by_grouping_seed.csv.gz"
    all_type_pairs.to_csv(all_type_pairs_path, index=False, compression="gzip")
    summary = _summarize_type_pairs(type_tables)
    summary_path = output / "type_pair_summary.csv"
    summary.to_csv(summary_path, index=False)
    reconstruction = pd.concat(reconstruction_tables, ignore_index=True)
    reconstruction_path = output / "exact_reconstruction_diagnostics.csv"
    reconstruction.to_csv(reconstruction_path, index=False)

    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "cytobridge_one_layer_spatial_attention_and_exact_message",
        "interpretation": {
            "attention": "absolute mean of signed, non-softmax gates across heads",
            "D_AB": (
                "mean over all receiver cells of the norm of the summed complete "
                "sender-type edge contributions; zero receivers remain in denominator"
            ),
            "bias": (
                "gene-readout bias is an explicit receiver baseline and is excluded "
                "from every sender contribution"
            ),
            "probability_claim": False,
        },
        "input": {
            "h5ad": _artifact(h5ad_path),
            "shape": [int(data.n_obs), int(data.n_vars)],
            "cell_type_key": args.cell_type_key,
            "time_key": args.time_key,
            "time_label_key": args.time_label_key,
            "spatial_key": args.spatial_key,
            "state_key": args.state_key,
            "feature_dim": feature_dim,
            "coordinate_contract": "observed_pre_warp_spatial_aligned",
        },
        "checkpoint": {
            "model_dir": str(model_dir),
            "weight_stage": loaded.weight_stage,
            "weight": _artifact(Path(loaded.weight_path)),
            "score_stage": loaded.score_stage,
            "score": (
                _artifact(Path(loaded.score_path))
                if loaded.score_path is not None
                else None
            ),
            "interaction_group_size": configured_group_size,
            "spatial_cutoff": float(interaction_net.cutoff),
            "edge_predictor_threshold": float(interaction_net.edge_predictor_thre),
            "num_heads": int(interaction_net.num_heads),
            "num_layers": int(interaction_net.num_layers),
            "requested_stage": str(args.checkpoint_stage),
            "interaction_randomization": (
                {
                    "seed": int(args.randomize_interaction_seed),
                    "reset_modules": [
                        "gene_embed",
                        "distance_projection",
                        "gnn_layers",
                        "gene_readout",
                    ],
                    "preserved_modules": ["link_predictor", "rbf_expansion"],
                }
                if args.randomize_interaction_seed is not None
                else None
            ),
        },
        "grouping": {
            "seeds": list(args.grouping_seeds),
            "technical_not_independent_training_replicates": True,
            "summary_order": "complete table per seed, then arithmetic mean/std",
        },
        "runs": run_records,
        "artifacts": {
            "observed_cells": _artifact(cell_table_path),
            "type_pair_by_grouping_seed": _artifact(all_type_pairs_path),
            "type_pair_summary": _artifact(summary_path),
            "exact_reconstruction_diagnostics": _artifact(reconstruction_path),
        },
    }
    manifest_path = output / "run_manifest.json"
    _write_json(manifest_path, manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = run(args)
    print(
        json.dumps(
            {
                "status": "ok",
                "n_runs": len(manifest["runs"]),
                "output": str(args.output_dir.expanduser().resolve()),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
