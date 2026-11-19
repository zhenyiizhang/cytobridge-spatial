#!/usr/bin/env python3
"""Build Figure 5d gene-velocity state from a newly trained ARISTA package run.

The numerical algorithm is the historical Figure 5d algorithm: fit a fresh
two-component PCA on the complete 50-dimensional gene state, build a 30-NN
graph in that state, construct the scVelo transition graph from a selected
complete 50-dimensional gene-velocity component, and project it into the fresh
PCA.  ``full`` remains the publication default; ``drift`` provides a matched
intrinsic-only diagnostic.  The input cells are the already QC-filtered package
cohort, so no second display mask or manuscript-coordinate mapping is applied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scvelo as scv


SCHEMA = "cytobridge.arista.fig5d.corrected-gene-velocity-state.v1"
FULL_DIM = 52


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def composite_key(source_obs_id: str) -> str:
    value = re.sub(r"^Batch=", "", str(source_obs_id))
    return value.replace("|CellID=", "|")


def time_token(time: float) -> str:
    return str(int(time)) if float(time).is_integer() else str(time).replace(".", "p")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--velocity-npz", required=True, type=Path)
    parser.add_argument("--slice-dir", required=True, type=Path)
    parser.add_argument("--identity-csv", required=True, type=Path)
    parser.add_argument("--palette-json", required=True, type=Path)
    parser.add_argument(
        "--velocity-component",
        choices=("full", "drift"),
        default="full",
        help=(
            "Component used by the scVelo transition graph. The publication "
            "default full equals drift + interaction + score; drift is the "
            "intrinsic-only diagnostic."
        ),
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    velocity_path = args.velocity_npz.expanduser().resolve()
    slice_dir = args.slice_dir.expanduser().resolve()
    identity_path = args.identity_csv.expanduser().resolve()
    palette_path = args.palette_json.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    for path in (velocity_path, identity_path, palette_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not slice_dir.is_dir():
        raise FileNotFoundError(slice_dir)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite immutable output: {output_dir}")

    with np.load(velocity_path, allow_pickle=False) as archive:
        required = {"features", "full", "drift", "interaction", "score", "times"}
        missing = sorted(required - set(archive.files))
        if missing:
            raise KeyError(f"Velocity archive is missing arrays: {missing}")
        arrays = {name: np.asarray(archive[name]).copy() for name in required}

    features = np.asarray(arrays["features"], dtype=np.float32)
    full = np.asarray(arrays["full"], dtype=np.float32)
    drift = np.asarray(arrays["drift"], dtype=np.float32)
    interaction = np.asarray(arrays["interaction"], dtype=np.float32)
    score = np.asarray(arrays["score"], dtype=np.float32)
    archived_times = np.asarray(arrays["times"], dtype=np.float64)
    n_cells = int(features.shape[0])
    if features.shape != (n_cells, FULL_DIM):
        raise ValueError(f"Expected a {FULL_DIM}D state, found {features.shape}")
    for name, value in {
        "full": full,
        "drift": drift,
        "interaction": interaction,
        "score": score,
    }.items():
        if value.shape != features.shape or not np.isfinite(value).all():
            raise ValueError(f"Invalid {name} component: {value.shape}")
    if not np.isfinite(features).all() or not np.isfinite(archived_times).all():
        raise ValueError("Feature or time archive contains non-finite values")
    identity_error = float(np.max(np.abs(full - (drift + interaction + score))))
    if identity_error > 3e-6:
        raise ValueError(f"Full velocity identity failed: {identity_error}")

    observed_times = sorted(float(value) for value in np.unique(archived_times))
    feature_rows: list[np.ndarray] = []
    label_rows: list[np.ndarray] = []
    time_rows: list[np.ndarray] = []
    slice_hashes: dict[str, str] = {}
    for time in observed_times:
        path = slice_dir / f"time_{time_token(time)}.h5ad"
        if not path.is_file():
            raise FileNotFoundError(path)
        state = ad.read_h5ad(path)
        if state.n_vars != FULL_DIM:
            raise ValueError(f"Unexpected state width at t={time:g}: {state.shape}")
        if "Annotation" not in state.obs:
            raise KeyError(f"{path} is missing Annotation")
        feature_rows.append(np.asarray(state.X, dtype=np.float32))
        label_rows.append(state.obs["Annotation"].astype(str).to_numpy())
        time_rows.append(np.full(state.n_obs, time, dtype=np.float64))
        slice_hashes[str(time)] = sha256(path)

    observed_features = np.vstack(feature_rows)
    labels = np.concatenate(label_rows)
    times = np.concatenate(time_rows)
    identity = pd.read_csv(identity_path)
    required_identity = {"source_obs_id", "time_point_processed", "Annotation"}
    if not required_identity.issubset(identity.columns):
        raise KeyError(
            f"Identity CSV is missing {sorted(required_identity - set(identity.columns))}"
        )
    source_ids = identity["source_obs_id"].astype(str).to_numpy()
    identity_times = identity["time_point_processed"].to_numpy(dtype=np.float64)
    identity_labels = identity["Annotation"].astype(str).to_numpy()
    if not np.array_equal(observed_features, features):
        raise ValueError("Observed slices are not row-identical to velocity features")
    if not np.array_equal(times, archived_times):
        raise ValueError("Observed slices are not row-identical to velocity times")
    if not np.array_equal(times, identity_times):
        raise ValueError("Identity CSV is not row-identical to observed slice times")
    if not np.array_equal(labels.astype(str), identity_labels):
        raise ValueError("Identity CSV is not row-identical to observed slice labels")
    if np.unique(source_ids).size != n_cells:
        raise ValueError("source_obs_id values must be unique")

    gene_state = features[:, 2:].copy()
    selected_velocity = {"full": full, "drift": drift}[args.velocity_component]
    gene_velocity = selected_velocity[:, 2:].copy()
    plot = ad.AnnData(X=gene_state.copy())
    sc.tl.pca(plot, n_comps=2, svd_solver="arpack", random_state=0)
    coordinates = np.asarray(plot.obsm["X_pca"], dtype=np.float32).copy()
    plot.layers["spliced"] = gene_state.copy()
    plot.layers["Ms"] = gene_state.copy()
    plot.layers["velocity"] = gene_velocity.copy()
    plot.obsm["X_pca"] = coordinates
    plot.obs["cell_type"] = pd.Categorical(labels.astype(str))
    plot.obs["time"] = times
    sc.pp.neighbors(plot, n_neighbors=30, use_rep="X", random_state=0)
    scv.tl.velocity_graph(
        plot,
        vkey="velocity",
        xkey="Ms",
        n_jobs=1,
        show_progress_bar=False,
    )
    scv.tl.velocity_embedding(plot, basis="pca", vkey="velocity")
    embedded_velocity = np.asarray(plot.obsm["velocity_pca"], dtype=np.float32).copy()
    if coordinates.shape != (n_cells, 2) or embedded_velocity.shape != (n_cells, 2):
        raise ValueError("Invalid Figure 5d PCA or velocity embedding shape")
    if not np.isfinite(coordinates).all() or not np.isfinite(embedded_velocity).all():
        raise ValueError("Figure 5d PCA or embedded velocity contains non-finite values")

    composite_keys = np.asarray([composite_key(value) for value in source_ids])
    display_mask = np.ones(n_cells, dtype=bool)
    counts_by_time = {
        str(float(time)): int(count)
        for time, count in zip(*np.unique(times, return_counts=True))
    }
    pca_hash = array_sha256(coordinates)
    embedded_hash = array_sha256(embedded_velocity)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.state-", dir=output_dir.parent))
    try:
        state_path = stage / "figure5d_corrected_gene_velocity_state.npz"
        table_path = stage / "figure5d_corrected_gene_velocity_embedding.csv"
        manifest_path = stage / "state_manifest.json"
        np.savez_compressed(
            state_path,
            corrected_raw_pca=coordinates,
            embedded_gene_velocity_pca=embedded_velocity,
            labels=labels.astype("U"),
            times=times,
            source_obs_ids=source_ids.astype("U"),
            composite_keys=composite_keys.astype("U"),
            display_mask=display_mask,
        )
        pd.DataFrame(
            {
                "cell_index": np.arange(n_cells, dtype=int),
                "source_obs_id": source_ids,
                "composite_key": composite_keys,
                "time": times,
                "cell_type": labels,
                "display_visible": display_mask,
                "corrected_raw_gene_pc1": coordinates[:, 0],
                "corrected_raw_gene_pc2": coordinates[:, 1],
                "embedded_gene_velocity_pc1": embedded_velocity[:, 0],
                "embedded_gene_velocity_pc2": embedded_velocity[:, 1],
            }
        ).to_csv(table_path, index=False)
        manifest = {
            "schema": SCHEMA,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "algorithm_contract": {
                "gene_state": "package velocity features[:, 2:]",
                "gene_velocity": (
                    "package full[:, 2:] where full=drift+interaction+score"
                    if args.velocity_component == "full"
                    else "package drift[:, 2:] (intrinsic only)"
                ),
                "velocity_component": args.velocity_component,
                "pca": "sc.tl.pca on complete 50D gene states, n_comps=2, svd_solver=arpack, random_state=0",
                "neighbors": "30-NN on complete 50D gene states, use_rep=X, random_state=0",
                "velocity_graph": "scv.tl.velocity_graph(vkey=velocity, xkey=Ms, n_jobs=1, show_progress_bar=False)",
                "velocity_embedding": "scv.tl.velocity_embedding into the fresh package-native gene PCA",
                "display_orientation": "none; raw corrected fresh PCA coordinates are used directly",
                "explicit_non_use": [
                    "no historical coordinate projection",
                    "no Procrustes/similarity/rotation/reflection/scale mapping",
                    "no second cell filtering after package preprocessing",
                ],
            },
            "scientific_state": {
                "n_cells_compute": n_cells,
                "n_cells_visible": n_cells,
                "n_graph_only_hidden_scatter": 0,
                "observed_counts_by_time": counts_by_time,
                "gene_state_shape": list(gene_state.shape),
                "gene_velocity_shape": list(gene_velocity.shape),
                "velocity_component": args.velocity_component,
                "full_component_identity_max_abs_error": identity_error,
                "corrected_raw_pca_sha256": pca_hash,
                "embedded_velocity_raw_sha256": embedded_hash,
                "embedded_velocity_mean_l2_norm": float(
                    np.linalg.norm(embedded_velocity, axis=1).mean()
                ),
            },
            "display_alignment": {
                "mapping": "none",
                "coordinates": "raw corrected fresh PCA",
                "manuscript_coordinate_reuse": False,
            },
            "inputs": {
                "velocity_npz": {"path": str(velocity_path), "sha256": sha256(velocity_path)},
                "slice_dir": {"path": str(slice_dir), "observed_slice_sha256": slice_hashes},
                "identity_csv": {"path": str(identity_path), "sha256": sha256(identity_path)},
                "palette_json": {"path": str(palette_path), "sha256": sha256(palette_path)},
            },
            "outputs": {
                state_path.name: {"sha256": sha256(state_path), "size_bytes": state_path.stat().st_size},
                table_path.name: {"sha256": sha256(table_path), "size_bytes": table_path.stat().st_size},
            },
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        shutil.move(str(stage), str(output_dir))
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "n_cells": n_cells,
                "velocity_component": args.velocity_component,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
