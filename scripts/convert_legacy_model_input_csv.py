#!/usr/bin/env python3
"""Convert an already prepared legacy model-input CSV to current AnnData keys."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from CytoBridge.pp import write_legacy_model_input_h5ad  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-h5ad", type=Path, required=True)
    parser.add_argument("--time-column", default="samples")
    parser.add_argument("--annotation-column", default="Annotation")
    parser.add_argument("--spatial-columns", nargs="+", default=["x1", "x2"])
    parser.add_argument("--latent-columns", nargs="+", default=None)
    parser.add_argument("--interaction-cutoff", type=float, default=None)
    parser.add_argument("--edge-predictor-threshold", type=float, default=None)
    parser.add_argument("--edge-predictor-path", type=Path, default=None)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    adata = write_legacy_model_input_h5ad(
        args.input_csv,
        args.output_h5ad,
        time_column=args.time_column,
        annotation_column=args.annotation_column or None,
        spatial_columns=args.spatial_columns,
        latent_columns=args.latent_columns,
        interaction_cutoff=args.interaction_cutoff,
        edge_predictor_threshold=args.edge_predictor_threshold,
        edge_predictor_path=args.edge_predictor_path,
    )
    payload = {
        "output_h5ad": str(args.output_h5ad.expanduser().resolve()),
        "n_obs": int(adata.n_obs),
        "spatial_dim": int(adata.obsm["spatial_aligned"].shape[1]),
        "latent_dim": int(adata.obsm["X_latent"].shape[1]),
        "time_counts": {
            str(key): int(value)
            for key, value in adata.obs["time_point_processed"].value_counts().sort_index().items()
        },
        "provenance": dict(adata.uns["legacy_model_input"]),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
