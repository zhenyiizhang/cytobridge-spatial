#!/usr/bin/env python3
"""Recompute the observed-slice velocity bank with the current package.

This narrowly scoped entry point is for accepted model runs whose historical
downstream bank predates the direct spatial-vector plotting contract. It loads
the accepted aligned H5AD and checkpoint directory, recomputes all model
velocity components, and writes a fresh immutable output directory. Spatial
arrows are the model's first two dimensions and are never projected through
scVelo. Gene-velocity panels use the remaining 50 expression-state dimensions
to build a scVelo transition graph and project that derivative onto the same
observed spatial coordinates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 2


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: Path, *, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    return resolved


def require_sha256(path: Path, expected: str, *, label: str) -> str:
    expected = str(expected).strip().lower()
    if len(expected) != 64 or any(ch not in "0123456789abcdef" for ch in expected):
        raise ValueError(
            f"{label} expected SHA-256 must be 64 lowercase hex characters."
        )
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(
            f"{label} SHA-256 mismatch: expected {expected}, observed {observed}."
        )
    return observed


def prepare_output_dir(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.exists() and any(resolved.iterdir()):
        raise FileExistsError(
            f"Velocity refresh output must be new or empty: {resolved}"
        )
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def scientific_contract() -> dict[str, Any]:
    return {
        "scope": "observed-slice fitted-vector-field evaluation",
        "spatial_velocity": {
            "coordinates": "spatial_aligned[:, :2]",
            "vectors": "first two fitted model dimensions",
            "projection": "direct; no scVelo projection",
        },
        "expression_velocity": {
            "state": "fitted 50-dimensional expression representation",
            "vectors": "fitted 50-dimensional expression-state derivative",
            "display_projection": (
                "scVelo transition graph in expression state projected onto "
                "observed spatial_aligned[:, :2] coordinates"
            ),
            "rendered": True,
        },
        "simulation": False,
        "observed_slice_reanchoring": False,
    }


def _artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "size": path.stat().st_size,
    }


def run(args: argparse.Namespace) -> Path:
    import anndata as ad
    import numpy as np
    import CytoBridge as cb
    import CytoBridge.workflow as workflow

    aligned_h5ad = require_file(args.aligned_h5ad, label="aligned H5AD")
    require_sha256(
        aligned_h5ad,
        args.expected_aligned_sha256,
        label="aligned H5AD",
    )
    model_dir = args.model_dir.expanduser().resolve()
    if not model_dir.is_dir():
        raise FileNotFoundError(f"model directory does not exist: {model_dir}")
    training_summary = require_file(
        model_dir / "training_run_summary.json", label="training run summary"
    )
    require_sha256(
        training_summary,
        args.expected_training_summary_sha256,
        label="training run summary",
    )
    output_dir = prepare_output_dir(args.output_dir)

    config, config_source = workflow.load_workflow_config(args.config)
    dataset = config["dataset"]
    adata = ad.read_h5ad(aligned_h5ad)
    annotation_key = str(dataset.get("annotation_key", "Annotation"))
    dataframe, _ = cb.tl.adata_to_aligned_dataframe(
        adata,
        time_key=dataset.get("time_key"),
        obsm_key=str(dataset.get("obsm_key", "X_latent")),
        spatial_key=str(dataset.get("spatial_key", "spatial_aligned")),
        concat_spatial=dataset.get("concat_spatial", True),
        annotation_key=annotation_key,
    )
    feature_columns = cb.tl.infer_feature_columns(
        dataframe, annotation_column=annotation_key
    )
    loaded = cb.tl.load_dynamical_model_from_dir(
        model_dir,
        dim=len(feature_columns),
        device=args.device,
    )
    labels = adata.obs[annotation_key].astype(str).to_numpy()
    label_to_color = cb.tl.load_label_to_color(
        labels,
        color_h5ad=str(aligned_h5ad),
        annotation_key=annotation_key,
    )
    velocity = workflow._write_velocity_outputs(
        cb=cb,
        adata=adata,
        model=loaded.model,
        dataset=dataset,
        annotation_key=annotation_key,
        label_to_color=label_to_color,
        output_dir=output_dir / "velocity",
        device=args.device,
    )

    archive = Path(velocity["component_archive"]).resolve()
    with np.load(archive, allow_pickle=False) as payload:
        required = {"times", "drift", "full"}
        missing = sorted(required.difference(payload.files))
        if missing:
            raise RuntimeError(f"velocity archive is missing arrays: {missing}")
        arrays = {key: np.asarray(payload[key]) for key in payload.files}
    nonfinite = [key for key, value in arrays.items() if not np.isfinite(value).all()]
    if nonfinite:
        raise RuntimeError(f"velocity archive contains non-finite arrays: {nonfinite}")
    if int(velocity["expression_dimensions"]) != 50:
        raise RuntimeError(
            "Formal velocity refresh requires exactly 50 expression-state "
            f"dimensions; observed {velocity['expression_dimensions']}."
        )

    spatial_figures = [_artifact(Path(path)) for path in velocity["spatial_figures"]]
    gene_figures = [_artifact(Path(path)) for path in velocity["gene_figures"]]
    if not spatial_figures or not gene_figures:
        raise RuntimeError(
            "Velocity refresh must render both spatial- and gene-velocity panels."
        )
    figures = [*spatial_figures, *gene_figures]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset["name"]),
        "config_source": config_source,
        "aligned_h5ad": _artifact(aligned_h5ad),
        "model_dir": str(model_dir),
        "training_run_summary": _artifact(training_summary),
        "scientific_contract": scientific_contract(),
        "package": {
            "version": str(getattr(cb, "__version__", "unknown")),
            "workflow_module": _artifact(Path(workflow.__file__).resolve()),
            "velocity_refresh_script": _artifact(Path(__file__).resolve()),
        },
        "velocity": {
            **velocity,
            "component_archive": _artifact(archive),
            "array_shapes": {key: list(value.shape) for key, value in arrays.items()},
            "all_finite": True,
            "spatial_figures": spatial_figures,
            "gene_figures": gene_figures,
            "figures": figures,
        },
    }
    manifest_path = output_dir / "velocity_refresh_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (output_dir / "velocity_refresh_manifest.json.sha256").write_text(
        f"{sha256_file(manifest_path)}  {manifest_path.name}\n", encoding="utf-8"
    )
    print(manifest_path)
    return manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--aligned-h5ad", type=Path, required=True)
    parser.add_argument("--expected-aligned-sha256", required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--expected-training-summary-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> int:
    run(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
