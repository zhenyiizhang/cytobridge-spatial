import argparse
import os

import sys

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import CytoBridge as cb

from CytoBridge.utils.config import load_config
from CytoBridge.pp.spatial_align import AlignConfig, preprocess_align_to_files


def _preset_config(name: str) -> tuple[AlignConfig, str, list[int], str]:
    name = name.lower()
    if name == "zebrafish":
        cfg = AlignConfig(
            center_x=True,
            center_y=False,
            scale_x=1.0,
            scale_y=1.0,
            flip_y=False,
            n_pcs=50,
        )
        return cfg, "time", [1, 2, 3, 4, 5], "spatial_sixtime_slice_stereoseq"
    if name == "mosta":
        cfg = AlignConfig(
            center_x=True,
            center_y=False,
            scale_x=1.0 / 100,
            scale_y=1.0 / 100,
            flip_y=True,
            n_pcs=50,
        )
        return cfg, "timepoint", [3, 4, 5, 6], "Mouse_embryo_all_stage"
    raise ValueError(f"Unknown preset: {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=["zebrafish", "mosta"], required=True)
    parser.add_argument("--h5ad_path", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--output_h5ad", default=None)
    parser.add_argument("--train_config", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--align_device", default=None)
    parser.add_argument("--phase1_epochs", type=int, default=None)
    parser.add_argument("--phase2_epochs", type=int, default=None)
    parser.add_argument("--batch_indices", default=None, help="Comma-separated batch indices override")
    parser.add_argument("--center_x", type=int, choices=[0, 1], default=None)
    parser.add_argument("--center_y", type=int, choices=[0, 1], default=None)
    parser.add_argument("--scale_x", type=float, default=None)
    parser.add_argument("--scale_y", type=float, default=None)
    parser.add_argument("--scale_z", type=float, default=None)
    parser.add_argument("--spatial_dim", type=int, default=2)
    parser.add_argument("--center_z", type=int, choices=[0, 1], default=None)
    parser.add_argument("--flip_y", type=int, choices=[0, 1], default=None)
    parser.add_argument("--n_top_genes", type=int, default=None)
    parser.add_argument("--n_pcs", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--lambda_local", type=float, default=None)
    parser.add_argument("--lambda_ot", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--distance_pairs", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--max_cells_per_timepoint", type=int, default=None)
    parser.add_argument("--output_chunk_size", type=int, default=None)
    parser.add_argument("--skip_train", action="store_true")
    args = parser.parse_args()

    cfg, time_key, batch_indices, _ = _preset_config(args.preset)
    cfg.spatial_dim = args.spatial_dim

    align_device = args.align_device or args.device
    if args.phase1_epochs is not None:
        cfg.phase1_epochs = args.phase1_epochs
    if args.phase2_epochs is not None:
        cfg.phase2_epochs = args.phase2_epochs
    if args.batch_indices is not None:
        batch_indices = [int(x) for x in args.batch_indices.split(",") if x.strip() != ""]
    if args.center_x is not None:
        cfg.center_x = bool(args.center_x)
    if args.center_y is not None:
        cfg.center_y = bool(args.center_y)
    if args.scale_x is not None:
        cfg.scale_x = args.scale_x
    if args.scale_y is not None:
        cfg.scale_y = args.scale_y
    if args.scale_z is not None:
        cfg.scale_z = args.scale_z
    if args.center_z is not None:
        cfg.center_z = bool(args.center_z)
    if args.flip_y is not None:
        cfg.flip_y = bool(args.flip_y)
    if args.n_top_genes is not None:
        cfg.n_top_genes = args.n_top_genes
    if args.n_pcs is not None:
        cfg.n_pcs = args.n_pcs
    if args.alpha is not None:
        cfg.alpha = args.alpha
    if args.beta is not None:
        cfg.beta = args.beta
    if args.lambda_local is not None:
        cfg.lambda_local = args.lambda_local
    if args.lambda_ot is not None:
        cfg.lambda_ot = args.lambda_ot
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.distance_pairs is not None:
        cfg.distance_pairs = args.distance_pairs
    if args.learning_rate is not None:
        cfg.learning_rate = args.learning_rate
    if args.max_cells_per_timepoint is not None:
        cfg.max_cells_per_timepoint = args.max_cells_per_timepoint
    if args.output_chunk_size is not None:
        cfg.output_chunk_size = args.output_chunk_size

    preprocess_align_to_files(
        h5ad_path=args.h5ad_path,
        time_key=time_key,
        output_csv=args.output_csv,
        output_h5ad=args.output_h5ad,
        cfg=cfg,
        batch_indices=batch_indices,
        device=align_device,
    )

    if not args.skip_train:
        # Load and update training config with spatial_dim
        train_config = load_config(args.train_config)
        train_config['model']['spatial_dim'] = cfg.spatial_dim
        
        cb.tl.fit(
            args.output_h5ad if args.output_h5ad else args.output_csv,
            config=train_config,
            device=args.device,
        )


if __name__ == "__main__":
    main()
