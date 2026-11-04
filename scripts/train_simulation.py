import argparse
import copy
import sys
from pathlib import Path

import torch
import yaml

pkg_path = "/lustre/home/2501111653/CytoBridge-ST-package/CytoBridge"
if pkg_path not in sys.path:
    sys.path.insert(0, pkg_path)

import CytoBridge as cb


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _prepare_quick_config(config: dict, quick_epochs: int, ckpt_dir: str | None) -> dict:
    cfg = copy.deepcopy(config)
    training = cfg.setdefault("training", {})
    plan = training.get("plan", [])
    if isinstance(plan, list):
        for stage in plan:
            if isinstance(stage, dict):
                stage["epochs"] = int(quick_epochs)
    if ckpt_dir is not None:
        cfg["ckpt_dir"] = str(Path(ckpt_dir).expanduser())
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Train simulation model with current CytoBridge fit API.")
    parser.add_argument(
        "--data-csv",
        type=str,
        default="/lustre/home/2501111653/CytoBridge-ST-1104/data/mouse_brain_simulation.csv",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="/lustre/home/2501111653/CytoBridge-ST-package/CytoBridge/CytoBridge/configs/simulation_config.yaml",
    )
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--quick", action="store_true", help="Use quick mode: set every stage to few epochs.")
    parser.add_argument("--quick-epochs", type=int, default=1)
    parser.add_argument(
        "--ckpt-dir",
        type=str,
        default=None,
        help="Optional runtime checkpoint dir override.",
    )
    parser.add_argument("--interaction-cutoff", type=float, default=None)
    parser.add_argument("--edge-predictor-path", type=str, default=None)
    parser.add_argument("--edge-predictor-threshold", type=float, default=None)
    parser.add_argument("--sigma", type=float, default=None)
    args = parser.parse_args()

    device = _resolve_device(args.device)
    print(f"[train_simulation] device={device}")
    print(f"[train_simulation] data={args.data_csv}")
    print(f"[train_simulation] config={args.config}")

    config_obj: dict | str = args.config
    if args.quick:
        base_cfg = _load_config(args.config)
        quick_default_ckpt = "/lustre/home/2501111653/CytoBridge-ST-package/CytoBridge/results/simulation_test_quick"
        config_obj = _prepare_quick_config(
            base_cfg,
            quick_epochs=args.quick_epochs,
            ckpt_dir=args.ckpt_dir or quick_default_ckpt,
        )
        print(
            f"[train_simulation] quick mode enabled: epochs_per_stage={args.quick_epochs}, "
            f"ckpt_dir={config_obj.get('ckpt_dir')}"
        )

    adata = cb.tl.fit(
        adata=args.data_csv,
        config=config_obj,
        batch_size=args.batch_size,
        device=device,
        samples_key="samples",
        time_key="samples",
        is_spatial=True,
        interaction_cutoff=args.interaction_cutoff,
        edge_predictor_path=args.edge_predictor_path,
        edge_predictor_threshold=args.edge_predictor_threshold,
        ckpt_dir=args.ckpt_dir if not args.quick else None,
        sigma=args.sigma,
    )
    print(
        "[train_simulation] training complete. output ckpt_dir="
        f"{adata.uns.get('all_model', {}).get('ckpt_dir', 'unknown')}"
    )


if __name__ == "__main__":
    main()
