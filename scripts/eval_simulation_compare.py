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


def _load_baseline_fnet(config_path, results_dir):
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


def _collect_growth_values(df, timepoints, f_net, model, device):
    baseline_vals = []
    ours_vals = []
    for t in timepoints:
        subset = df[df["samples"] == t]
        data = torch.tensor(subset.iloc[:, 1:].values, dtype=torch.float32).to(device)
        t_tensor = torch.tensor([t], dtype=torch.float32).to(device)
        with torch.no_grad():
            _, g_base, _, _ = f_net(t_tensor, data)
        baseline_vals.append(g_base.detach().cpu().numpy())
        t_expand = t_tensor.expand(data.shape[0], 1)
        net_input = torch.cat([data, t_expand], dim=1)
        with torch.no_grad():
            g_ours = model.growth_net(net_input).detach().cpu().numpy()
        ours_vals.append(g_ours)
    baseline_vals = np.concatenate(baseline_vals, axis=0)
    ours_vals = np.concatenate(ours_vals, axis=0)
    return baseline_vals, ours_vals


def _collect_attn_means(df, timepoints, f_net, model, device):
    attn_rows = []
    for t in timepoints:
        subset = df[df["samples"] == t]
        data = torch.tensor(subset.iloc[:, 1:].values, dtype=torch.float32).to(device)
        lnw = torch.log(torch.ones(data.shape[0], 1) / data.shape[0]).to(device)
        t_tensor = torch.tensor([t], dtype=torch.float32).to(device)
        with torch.no_grad():
            _ = f_net.interaction_net(data, lnw, t_tensor, return_attn=True)
            base_attn = torch.abs(f_net.interaction_net.gnn_layers[0].attn)
            base_attn_mean = base_attn.mean(dim=1).detach().cpu().numpy()
        with torch.no_grad():
            _ = model.interaction_net(data, lnw, t_tensor, return_attn=True)
            ours_attn = torch.abs(model.interaction_net.gnn_layers[0].attn)
            ours_attn_mean = ours_attn.mean(dim=1).detach().cpu().numpy()
        attn_rows.append((t, base_attn_mean, ours_attn_mean))
    return attn_rows


def _collect_attn_means_gt(df, timepoints, model, device, gt_dir):
    attn_rows = []
    for t in timepoints:
        subset = df[df["samples"] == t]
        data = torch.tensor(subset.iloc[:, 1:].values, dtype=torch.float32).to(device)
        lnw = torch.log(torch.ones(data.shape[0], 1) / data.shape[0]).to(device)
        t_tensor = torch.tensor([t], dtype=torch.float32).to(device)
        with torch.no_grad():
            _ = model.interaction_net(data, lnw, t_tensor, return_attn=True)
            ours_attn = torch.abs(model.interaction_net.gnn_layers[0].attn)
            ours_attn_mean = ours_attn.mean(dim=1).detach().cpu().numpy()
        gt_path = os.path.join(gt_dir, f"attn_mean_time{int(t)}.npy")
        gt_attn_mean = np.load(gt_path)
        attn_rows.append((t, gt_attn_mean, ours_attn_mean))
    return attn_rows


def _collect_growth_values_gt(df, timepoints, model, device, gt_dir):
    ours_vals = []
    for t in timepoints:
        subset = df[df["samples"] == t]
        data = torch.tensor(subset.iloc[:, 1:].values, dtype=torch.float32).to(device)
        t_tensor = torch.tensor([t], dtype=torch.float32).to(device)
        t_expand = t_tensor.expand(data.shape[0], 1)
        net_input = torch.cat([data, t_expand], dim=1)
        with torch.no_grad():
            g_ours = model.growth_net(net_input).detach().cpu().numpy()
        ours_vals.append(g_ours)
    ours_vals = np.concatenate(ours_vals, axis=0)
    gt_vals = np.load(os.path.join(gt_dir, "g_values.npy"))
    return gt_vals, ours_vals


def _corr(a, b):
    if a.shape != b.shape:
        return np.nan
    if a.size == 0:
        return np.nan
    return float(np.corrcoef(a.flatten(), b.flatten())[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_csv", required=True)
    parser.add_argument("--baseline_config")
    parser.add_argument("--baseline_results")
    parser.add_argument("--ours_adata", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--gt_dir")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    df = pd.read_csv(args.data_csv)
    df = df.iloc[:, :53]
    timepoints = sorted(df["samples"].unique())

    adata = sc.read(args.ours_adata)
    model = load_model_from_adata(adata).to(device)

    if args.gt_dir:
        base_g, ours_g = _collect_growth_values_gt(df, timepoints, model, device, args.gt_dir)
    else:
        if not args.baseline_config or not args.baseline_results:
            raise ValueError("baseline_config and baseline_results are required when gt_dir is not set.")
        f_net = _load_baseline_fnet(args.baseline_config, args.baseline_results).to(device)
        base_g, ours_g = _collect_growth_values(df, timepoints, f_net, model, device)
    g_corr = _corr(base_g, ours_g)
    g_mae = float(np.mean(np.abs(base_g - ours_g)))

    if args.gt_dir:
        attn_rows = _collect_attn_means_gt(df, timepoints, model, device, args.gt_dir)
    else:
        attn_rows = _collect_attn_means(df, timepoints, f_net, model, device)
    attn_stats = []
    for t, base_attn, ours_attn in attn_rows:
        attn_corr = _corr(base_attn, ours_attn)
        attn_mae = float(np.mean(np.abs(base_attn - ours_attn))) if base_attn.shape == ours_attn.shape else np.nan
        attn_stats.append((t, attn_corr, attn_mae, base_attn.shape[0], ours_attn.shape[0]))

    rows = [
        {
            "metric": "growth_corr",
            "value": g_corr,
        },
        {
            "metric": "growth_mae",
            "value": g_mae,
        },
    ]
    for t, corr, mae, base_n, ours_n in attn_stats:
        rows.append(
            {
                "metric": f"attn_corr_t{int(t)}",
                "value": corr,
                "base_edges": base_n,
                "ours_edges": ours_n,
            }
        )
        rows.append(
            {
                "metric": f"attn_mae_t{int(t)}",
                "value": mae,
                "base_edges": base_n,
                "ours_edges": ours_n,
            }
        )

    out_dir = os.path.dirname(args.out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out_csv, index=False)
    print(f"Saved comparison metrics -> {args.out_csv}")


if __name__ == "__main__":
    main()
