import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

import scanpy as sc

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from CytoBridge.utils.utils import load_model_from_adata
from CytoBridge.tl.core.interaction import cal_interaction as cal_interaction_ours


def _load_baseline_fnet(config_path: str, results_dir: str):
    sys.path.append(os.path.abspath("/lustre/home/2501111653/CytoBridge-ST-1104"))
    from DeepRUOT.utils import load_and_merge_config
    from DeepRUOT.models import FNet_interaction

    config = load_and_merge_config(config_path)
    model_cfg = config["model"]
    f_net = FNet_interaction(
        in_out_dim=model_cfg["in_out_dim"],
        hidden_dim=model_cfg["hidden_dim"],
        n_hiddens=model_cfg["n_hiddens"],
        activation=model_cfg["activation"],
        use_spatial=True,
        num_heads=8,
        thre=model_cfg["thre"],
        num_layers=1,
        edge_predictor_path=model_cfg["edge_predictor_path"],
        edge_predictor_thre=model_cfg["edge_predictor_thre"],
    )
    f_net.load_state_dict(torch.load(os.path.join(results_dir, "model_final"), map_location="cpu"))
    f_net.eval()
    return f_net


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape or a.size == 0:
        return float("nan")
    return float(np.corrcoef(a.flatten(), b.flatten())[0, 1])


def _mae(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape or a.size == 0:
        return float("nan")
    return float(np.mean(np.abs(a - b)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_csv", required=True)
    parser.add_argument("--baseline_config", required=True)
    parser.add_argument("--baseline_results", required=True)
    parser.add_argument("--ours_adata", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device(args.device)
    df = pd.read_csv(args.data_csv)
    df = df.iloc[:, :53]
    timepoints = sorted(df["samples"].unique())
    dim = df.shape[1] - 1

    base_fnet = _load_baseline_fnet(args.baseline_config, args.baseline_results).to(device)

    adata = sc.read(args.ours_adata)
    ours_model = load_model_from_adata(adata).to(device)
    ours_model.eval()

    sys.path.append(os.path.abspath("/lustre/home/2501111653/CytoBridge-ST-1104"))
    from DeepRUOT.interaction import cal_interaction as cal_interaction_base

    rows = []
    base_g_all = []
    ours_g_all = []
    base_inter_all = []
    ours_inter_all = []

    for t in timepoints:
        subset = df[df["samples"] == t]
        data = torch.tensor(subset.iloc[:, 1:dim + 1].values, dtype=torch.float32).to(device)
        t_tensor = torch.tensor([t], dtype=torch.float32).to(device)
        t_expand = t_tensor.expand(data.shape[0], 1)

        with torch.no_grad():
            _, g_base, _, _ = base_fnet(t_tensor, data)
        base_g = g_base.detach().cpu().numpy()
        with torch.no_grad():
            g_ours = ours_model.growth_net(torch.cat([data, t_expand], dim=1)).detach().cpu().numpy()

        base_g_all.append(base_g)
        ours_g_all.append(g_ours)

        lnw = torch.log(torch.ones(data.shape[0], 1, device=device) / data.shape[0])
        torch.manual_seed(args.seed)
        with torch.no_grad():
            inter_base = cal_interaction_base(
                data,
                lnw,
                base_fnet.interaction_net,
                torch.tensor([t], dtype=torch.float32, device=device),
                m=1024,
                threshold=1000,
            ).detach().cpu().numpy()
        torch.manual_seed(args.seed)
        with torch.no_grad():
            inter_ours = cal_interaction_ours(
                data,
                lnw,
                ours_model.interaction_net,
                m=1024,
                t=torch.tensor([t], dtype=torch.float32, device=device),
            ).detach().cpu().numpy()

        base_inter_all.append(inter_base)
        ours_inter_all.append(inter_ours)

        rows.append(
            {
                "metric": "growth_corr",
                "time": float(t),
                "value": _pearson(base_g, g_ours),
            }
        )
        rows.append(
            {
                "metric": "growth_mae",
                "time": float(t),
                "value": _mae(base_g, g_ours),
            }
        )

        rows.append(
            {
                "metric": "interaction_corr_xy",
                "time": float(t),
                "value": _pearson(inter_base[:, :2], inter_ours[:, :2]),
            }
        )
        rows.append(
            {
                "metric": "interaction_mae_xy",
                "time": float(t),
                "value": _mae(inter_base[:, :2], inter_ours[:, :2]),
            }
        )

    base_g_all = np.concatenate(base_g_all, axis=0)
    ours_g_all = np.concatenate(ours_g_all, axis=0)
    base_inter_all = np.concatenate(base_inter_all, axis=0)
    ours_inter_all = np.concatenate(ours_inter_all, axis=0)

    rows.append(
        {
            "metric": "growth_corr_all",
            "time": "all",
            "value": _pearson(base_g_all, ours_g_all),
        }
    )
    rows.append(
        {
            "metric": "growth_mae_all",
            "time": "all",
            "value": _mae(base_g_all, ours_g_all),
        }
    )
    rows.append(
        {
            "metric": "interaction_corr_xy_all",
            "time": "all",
            "value": _pearson(base_inter_all[:, :2], ours_inter_all[:, :2]),
        }
    )
    rows.append(
        {
            "metric": "interaction_mae_xy_all",
            "time": "all",
            "value": _mae(base_inter_all[:, :2], ours_inter_all[:, :2]),
        }
    )

    out_dir = os.path.dirname(args.out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out_csv, index=False)
    print(f"Saved zebrafish growth/interaction comparison -> {args.out_csv}")


if __name__ == "__main__":
    main()
