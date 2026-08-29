import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import pearsonr

pkg_path = "/lustre/home/2501111653/CytoBridge-ST-package/CytoBridge"
if pkg_path not in sys.path:
    sys.path.insert(0, pkg_path)

from CytoBridge.tl.core.models import DynamicalModel


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    n = min(a.shape[0], b.shape[0])
    if n == 0:
        return float("nan")
    a = a[:n]
    b = b[:n]
    if np.allclose(a.std(), 0.0) or np.allclose(b.std(), 0.0):
        return float("nan")
    return float(pearsonr(a, b)[0])


def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_key_map(
    new_state: dict,
    old_state: dict,
    old_score_state: dict,
) -> dict:
    key_map = {}

    # velocity: v_net.* -> velocity_net.*
    for key in old_state:
        if key.startswith("v_net."):
            new_key = key.replace("v_net.", "velocity_net.")
            if new_key in new_state:
                key_map[new_key] = old_state[key]

    # growth: g_net.net.* -> growth_net.*
    growth_mapping = {
        "g_net.net.0.weight": "growth_net.input_layer.0.weight",
        "g_net.net.0.bias": "growth_net.input_layer.0.bias",
        "g_net.net.2.weight": "growth_net.hidden_layers.0.0.weight",
        "g_net.net.2.bias": "growth_net.hidden_layers.0.0.bias",
        "g_net.net.4.weight": "growth_net.hidden_layers.1.0.weight",
        "g_net.net.4.bias": "growth_net.hidden_layers.1.0.bias",
        "g_net.net.6.weight": "growth_net.output_layer.weight",
        "g_net.net.6.bias": "growth_net.output_layer.bias",
    }
    for old_key, new_key in growth_mapping.items():
        if old_key in old_state and new_key in new_state:
            key_map[new_key] = old_state[old_key]

    # score: legacy score_model net.* -> score_net.*
    score_mapping = {
        "net.0.weight": "score_net.input_layer.0.weight",
        "net.0.bias": "score_net.input_layer.0.bias",
        "net.2.weight": "score_net.hidden_layers.0.0.weight",
        "net.2.bias": "score_net.hidden_layers.0.0.bias",
        "net.4.weight": "score_net.hidden_layers.1.0.weight",
        "net.4.bias": "score_net.hidden_layers.1.0.bias",
        "net.6.weight": "score_net.output_layer.weight",
        "net.6.bias": "score_net.output_layer.bias",
    }
    for old_key, new_key in score_mapping.items():
        if old_key in old_score_state and new_key in new_state:
            key_map[new_key] = old_score_state[old_key]

    # interaction: interaction_net.* -> interaction_net.*
    for key in old_state:
        if key.startswith("interaction_net.") and key in new_state:
            key_map[key] = old_state[key]

    return key_map


def _verify_against_gt(
    model: DynamicalModel,
    data_csv: str,
    gt_dir: str,
) -> dict:
    df = pd.read_csv(data_csv)
    if "samples" not in df.columns:
        raise KeyError(f"'samples' not found in {data_csv}")
    feature_cols = [c for c in df.columns if c != "samples"]
    time_points = sorted(df["samples"].unique())

    all_v = []
    all_g = []
    attn_corrs = []

    for t in time_points:
        sub = df[df["samples"] == t]
        x = torch.tensor(sub[feature_cols].to_numpy(), dtype=torch.float32)
        tt = torch.full((x.shape[0], 1), float(t), dtype=torch.float32)
        lnw = torch.zeros(x.shape[0], 1, dtype=torch.float32)

        x.requires_grad_(True)
        tt.requires_grad_(True)
        lnw.requires_grad_(True)
        out = model(tt, x, lnw)
        all_v.append(out["velocity"].detach().numpy())
        all_g.append(out["growth"].detach().numpy())

        model.interaction_net(x, lnw.detach(), t=tt, return_attn=True)
        attn = model.interaction_net.gnn_layers[0].attn
        attn_mean = attn.mean(dim=1).detach().numpy()
        gt_attn = np.load(os.path.join(gt_dir, f"attn_mean_time{int(t)}.npy"))
        attn_corrs.append(_safe_corr(attn_mean, gt_attn))

    all_v = np.concatenate(all_v, axis=0)
    all_g = np.concatenate(all_g, axis=0)
    gt_v = np.load(os.path.join(gt_dir, "simulation_gradients_np_gt.npy"))
    gt_g = np.load(os.path.join(gt_dir, "g_values.npy"))
    if gt_g.ndim == 1:
        gt_g = gt_g[:, None]

    return {
        "velocity_correlation": _safe_corr(all_v, gt_v),
        "growth_correlation": _safe_corr(all_g, gt_g),
        "attention_correlations": attn_corrs,
        "attention_mean_correlation": float(np.mean(attn_corrs)) if len(attn_corrs) else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert legacy (ST-1104) checkpoints to new CytoBridge ckpt format.")
    parser.add_argument(
        "--config",
        type=str,
        default="/lustre/home/2501111653/CytoBridge-ST-package/CytoBridge/CytoBridge/configs/simulation_config.yaml",
    )
    parser.add_argument(
        "--old-model-ckpt",
        type=str,
        default="/lustre/home/2501111653/CytoBridge-ST-1104/results/mosta_interaction_1017_tiaocan/model_final",
    )
    parser.add_argument(
        "--old-score-ckpt",
        type=str,
        default="/lustre/home/2501111653/CytoBridge-ST-1104/results/mosta_interaction_1017_tiaocan/score_model",
    )
    parser.add_argument("--latent-dim", type=int, default=52)
    parser.add_argument(
        "--out-ckpt",
        type=str,
        default="/lustre/home/2501111653/CytoBridge-ST-package/CytoBridge/results/simulation_from_old_mapped/Finetune/last_model.pth",
    )
    parser.add_argument("--verify", action="store_true")
    parser.add_argument(
        "--data-csv",
        type=str,
        default="/lustre/home/2501111653/CytoBridge-ST-1104/data/mouse_brain_simulation.csv",
    )
    parser.add_argument(
        "--gt-dir",
        type=str,
        default="/lustre/home/2501111653/CytoBridge-ST-1104/results/mosta_interaction_1017_tiaocan",
    )
    args = parser.parse_args()

    cfg = _load_yaml(args.config)
    model = DynamicalModel(args.latent_dim, cfg["model"], use_growth_in_ode_inter=True)

    new_state = model.state_dict()
    old_state = torch.load(args.old_model_ckpt, map_location="cpu")
    old_score_state = torch.load(args.old_score_ckpt, map_location="cpu")

    key_map = _build_key_map(new_state, old_state, old_score_state)
    missing = [k for k in new_state.keys() if k not in key_map]
    coverage = len(key_map) / max(1, len(new_state))
    print(f"[convert] mapped keys: {len(key_map)}/{len(new_state)} (coverage={coverage:.4f})")
    if missing:
        print(f"[convert] missing keys (first 20): {missing[:20]}")

    model.load_state_dict(key_map, strict=False)

    out_path = Path(args.out_ckpt)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(out_path))
    print(f"[convert] saved converted checkpoint: {out_path}")

    meta = {
        "config": args.config,
        "old_model_ckpt": args.old_model_ckpt,
        "old_score_ckpt": args.old_score_ckpt,
        "out_ckpt": str(out_path),
        "mapped_keys": len(key_map),
        "total_new_keys": len(new_state),
        "coverage": coverage,
        "missing_keys": missing,
    }

    if args.verify:
        print("[convert] running verification against GT ...")
        model.eval()
        verify_res = _verify_against_gt(model, args.data_csv, args.gt_dir)
        meta["verification"] = verify_res
        print(
            "[convert] verify correlations: "
            f"velocity={verify_res['velocity_correlation']:.6f}, "
            f"growth={verify_res['growth_correlation']:.6f}, "
            f"attn_mean={verify_res['attention_mean_correlation']:.6f}"
        )

    meta_path = out_path.with_suffix(out_path.suffix + ".meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"[convert] wrote metadata: {meta_path}")


if __name__ == "__main__":
    main()
