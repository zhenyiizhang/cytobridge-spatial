#!/usr/bin/env python3
"""Run inference-only interaction ablations from one full checkpoint.

The three outputs use the exact same sampled starting cells:

1. ``full``: fitted model and fitted LR-informed edge gate;
2. ``interaction_off``: interaction velocity omitted at inference;
3. ``lr_gate_off``: a same-checkpoint all-spatial gate counterfactual.  The
   fitted interaction GNN is retained, but every spatial candidate edge
   within its cutoff is admitted.

No training, growth/resampling, spatial warping, or stochastic diffusion is
performed by this script.  In particular, ``lr_gate_off`` is not
"no-LR-prior training" and is not a matched-density edge shuffle.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import scanpy as sc


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import CytoBridge as cb  # noqa: E402
from CytoBridge.tl.downstream.functional_ablation import (  # noqa: E402
    FrozenCheckpointAblationResult,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adata", required=True, help="Aligned AnnData input.")
    parser.add_argument(
        "--model-dir",
        required=True,
        help="Full-model result directory containing config.yaml/checkpoints.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--stage", default="Finetune")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--time-key", default=None)
    parser.add_argument("--obsm-key", default="X_latent")
    parser.add_argument("--spatial-key", default="spatial_aligned")
    parser.add_argument("--annotation-key", default="Annotation")
    parser.add_argument(
        "--no-concat-spatial",
        action="store_true",
        help="Use latent state only; default concatenates aligned spatial coordinates.",
    )
    parser.add_argument(
        "--time-index",
        type=int,
        default=0,
        help="Observed stage used as the fixed initial cohort.",
    )
    parser.add_argument(
        "--end-time",
        type=float,
        default=None,
        help="Rollout endpoint; default is the next observed stage.",
    )
    parser.add_argument("--n-cells", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--interaction-m", type=int, default=1024)
    parser.add_argument("--spatial-dim", type=int, default=2)
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=list(cb.tl.DEFAULT_FROZEN_ABLATION_CONDITIONS),
        choices=list(cb.tl.DEFAULT_FROZEN_ABLATION_CONDITIONS),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.n_cells < 2:
        raise ValueError("--n-cells must be at least 2.")

    adata_path = Path(args.adata).expanduser().resolve()
    model_dir = Path(args.model_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    adata = sc.read_h5ad(adata_path)
    frame, resolved_time_key = cb.tl.adata_to_aligned_dataframe(
        adata,
        time_key=args.time_key,
        obsm_key=args.obsm_key,
        spatial_key=args.spatial_key,
        concat_spatial=not args.no_concat_spatial,
        annotation_key=args.annotation_key,
    )
    feature_cols = list(cb.tl.infer_feature_columns(frame))
    observed_times = sorted(float(value) for value in frame["samples"].unique())
    if args.time_index < 0 or args.time_index >= len(observed_times):
        raise IndexError(
            f"--time-index {args.time_index} is outside the observed stage "
            f"range [0, {len(observed_times) - 1}]."
        )
    start_time = observed_times[args.time_index]
    if args.end_time is None:
        if args.time_index + 1 >= len(observed_times):
            raise ValueError(
                "--end-time is required when --time-index selects the last "
                "observed stage."
            )
        end_time = observed_times[args.time_index + 1]
    else:
        end_time = float(args.end_time)
    if end_time <= start_time:
        raise ValueError("--end-time must be greater than the selected start stage.")

    start_frame = frame.loc[
        np.isclose(frame["samples"].to_numpy(dtype=float), start_time)
    ].copy()
    if len(start_frame) < 2:
        raise ValueError(f"Only {len(start_frame)} cells are available at {start_time}.")
    rng = np.random.default_rng(args.seed)
    if len(start_frame) > args.n_cells:
        selected = np.sort(
            rng.choice(len(start_frame), size=args.n_cells, replace=False)
        )
        start_frame = start_frame.iloc[selected].copy()
    points = start_frame[feature_cols].to_numpy(dtype=np.float32)

    loaded = cb.tl.load_dynamical_model_from_dir(
        model_dir=model_dir,
        dim=len(feature_cols),
        device=args.device,
        stage=args.stage,
    )
    result = cb.tl.run_frozen_checkpoint_ablations(
        points,
        loaded,
        start_time=start_time,
        end_time=end_time,
        dt=args.dt,
        interaction_m=args.interaction_m,
        grouping_seed=args.seed,
        conditions=args.conditions,
        device=args.device,
        spatial_dim=args.spatial_dim,
    )
    manifest = dict(result.manifest)
    manifest["input"] = {
        "adata": str(adata_path),
        "adata_sha256": _sha256(adata_path),
        "model_dir": str(model_dir),
        "resolved_time_key": str(resolved_time_key),
        "obsm_key": str(args.obsm_key),
        "spatial_key": str(args.spatial_key),
        "annotation_key": str(args.annotation_key),
        "concat_spatial": not args.no_concat_spatial,
        "selected_observed_time_index": int(args.time_index),
        "initial_cohort_linkage": (
            "initial_cohort.csv row_index_in_adata indexes both the source "
            "AnnData spatial and latent matrices without duplicating them"
        ),
    }
    result = FrozenCheckpointAblationResult(
        rollouts=result.rollouts,
        manifest=manifest,
    )
    paths = cb.tl.save_frozen_checkpoint_ablation_result(result, output_dir)
    cohort_path = output_dir / "initial_cohort.csv"
    cohort = {
        "cohort_row": np.arange(len(start_frame), dtype=int),
        "row_index_in_adata": adata.obs_names.get_indexer(start_frame.index),
        "source_obs_name": start_frame["cell_id"].astype(str).to_numpy(),
        "start_time": np.full(len(start_frame), start_time, dtype=float),
    }
    if args.annotation_key in start_frame.columns:
        cohort["cell_type"] = (
            start_frame[args.annotation_key].astype(str).to_numpy()
        )
    else:
        cohort["cell_type"] = np.full(len(start_frame), "", dtype=object)
    pd.DataFrame(cohort).to_csv(cohort_path, index=False)

    print(f"Saved frozen-checkpoint ablations to {output_dir}")
    print(f"Manifest: {paths['manifest']}")
    print(f"Initial cohort: {cohort_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
