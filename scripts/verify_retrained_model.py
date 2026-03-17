import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import pearsonr

# Add package path
pkg_path = "/lustre/home/2501111653/CytoBridge-ST-package/CytoBridge"
if pkg_path not in sys.path:
    sys.path.insert(0, pkg_path)

from CytoBridge.tl.core.models import DynamicalModel


def _safe_corr(a: np.ndarray, b: np.ndarray, label: str) -> float:
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    n = min(a.shape[0], b.shape[0])
    if n == 0:
        print(f"[verify] {label}: empty arrays, correlation=nan")
        return float("nan")
    a = a[:n]
    b = b[:n]
    if np.allclose(a.std(), 0.0) or np.allclose(b.std(), 0.0):
        print(f"[verify] {label}: one vector is constant, correlation=nan")
        return float("nan")
    corr, _ = pearsonr(a, b)
    return float(corr)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify retrained model against reference outputs.")
    parser.add_argument(
        "--config",
        type=str,
        default="/lustre/home/2501111653/CytoBridge-ST-package/CytoBridge/CytoBridge/configs/simulation_config.yaml",
    )
    parser.add_argument(
        "--data-csv",
        type=str,
        default="/lustre/home/2501111653/CytoBridge-ST-1104/data/mouse_brain_simulation.csv",
    )
    parser.add_argument(
        "--ckpt-path",
        type=str,
        default=None,
        help="Direct checkpoint path. If omitted, use <ckpt-dir>/<stage>/last_model.pth",
    )
    parser.add_argument(
        "--ckpt-dir",
        type=str,
        default="/lustre/home/2501111653/CytoBridge-ST-package/CytoBridge/results/simulation_test",
    )
    parser.add_argument("--stage", type=str, default="Finetune")
    parser.add_argument(
        "--gt-dir",
        type=str,
        default="/lustre/home/2501111653/CytoBridge-ST-1104/results/mosta_interaction_1017_tiaocan",
    )
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def verify_retrained_model(args: argparse.Namespace) -> tuple[float, float]:
    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device
    if args.device == "auto" and not torch.cuda.is_available():
        device = "cpu"
    print(f"[verify] Using device: {device}")

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print(f"[verify] Loading data from {args.data_csv}")
    df = pd.read_csv(args.data_csv)
    if "samples" not in df.columns:
        raise KeyError(f"'samples' column missing in {args.data_csv}")
    feature_cols = [c for c in df.columns if c != "samples"]
    if not feature_cols:
        raise ValueError("No feature columns found in CSV.")

    dim = len(feature_cols)
    time_points = sorted(df["samples"].unique())
    print(f"[verify] Time points: {time_points}, inferred dim={dim}")

    model = DynamicalModel(dim, config["model"]).to(device)

    ckpt_path = args.ckpt_path
    if ckpt_path is None:
        ckpt_path = str(Path(args.ckpt_dir) / args.stage / "last_model.pth")
    print(f"[verify] Loading checkpoint: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    gt_dir = args.gt_dir
    all_v = []
    all_g = []

    print("\n--- Verifying Attention (Per Time Point) ---")
    for t in time_points:
        data_t = df[df["samples"] == t][feature_cols].to_numpy()
        tensor_x = torch.tensor(data_t, dtype=torch.float32, device=device)
        tensor_t = torch.full((tensor_x.shape[0], 1), float(t), dtype=torch.float32, device=device)
        lnw = torch.zeros(tensor_x.shape[0], 1, device=device)

        with torch.enable_grad():
            tensor_x.requires_grad_(True)
            tensor_t.requires_grad_(True)
            model.interaction_net(tensor_x, lnw, t=tensor_t, return_attn=True)

        if not hasattr(model.interaction_net.gnn_layers[0], "attn"):
            print(f"[verify] Time {t}: attention not found on gnn layer, skip")
            continue
        attn = model.interaction_net.gnn_layers[0].attn
        attn_mean = attn.mean(dim=1).detach().cpu().numpy()

        gt_attn_path = os.path.join(gt_dir, f"attn_mean_time{int(t)}.npy")
        if not os.path.exists(gt_attn_path):
            print(f"[verify] Time {t}: GT attention file missing -> {gt_attn_path}")
            continue
        gt_attn = np.load(gt_attn_path)
        corr = _safe_corr(attn_mean, gt_attn, f"attention_t{t}")
        print(f"[verify] Time {t}: Attention Correlation = {corr:.4f}")

    print("\n--- Verifying Velocity and Growth (Global) ---")
    for t in time_points:
        data_t = df[df["samples"] == t][feature_cols].to_numpy()
        tensor_x = torch.tensor(data_t, dtype=torch.float32, device=device)
        tensor_x.requires_grad_(True)
        tensor_t = torch.full((tensor_x.shape[0], 1), float(t), dtype=torch.float32, device=device)
        tensor_t.requires_grad_(True)
        lnw = torch.zeros(tensor_x.shape[0], 1, dtype=torch.float32, device=device)
        lnw.requires_grad_(True)

        outputs = model(tensor_t, tensor_x, lnw)
        v = outputs["velocity"].detach().cpu().numpy()
        g = outputs["growth"].detach().cpu().numpy()
        all_v.append(v)
        all_g.append(g)

    all_v = np.concatenate(all_v, axis=0)
    all_g = np.concatenate(all_g, axis=0)
    print(f"[verify] Prediction shapes: V={all_v.shape}, G={all_g.shape}")

    gt_g_path = os.path.join(gt_dir, "g_values.npy")
    gt_v_path = os.path.join(gt_dir, "simulation_gradients_np_gt.npy")

    corr_g = float("nan")
    corr_v = float("nan")

    if os.path.exists(gt_g_path):
        gt_g = np.load(gt_g_path)
        if gt_g.ndim == 1:
            gt_g = gt_g.reshape(-1, 1)
        corr_g = _safe_corr(all_g, gt_g, "growth")
        print(f"[verify] GLOBAL Growth Correlation: {corr_g:.4f}")
    else:
        print(f"[verify] GT growth file missing: {gt_g_path}")

    if os.path.exists(gt_v_path):
        gt_v = np.load(gt_v_path)
        corr_v = _safe_corr(all_v, gt_v, "velocity")
        print(f"[verify] GLOBAL Velocity Correlation: {corr_v:.4f}")
    else:
        print(f"[verify] GT velocity file missing: {gt_v_path}")

    print(f"FINAL_GROWTH_CORRELATION={corr_g:.6f}")
    print(f"FINAL_VELOCITY_CORRELATION={corr_v:.6f}")
    # Keep a single canonical final value for automation/parsing.
    print(f"FINAL_CORRELATION={corr_v:.6f}")
    return corr_g, corr_v


if __name__ == "__main__":
    parsed = parse_args()
    verify_retrained_model(parsed)
