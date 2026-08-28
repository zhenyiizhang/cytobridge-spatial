#!/usr/bin/env python3
"""Run independently seeded AGIST split-SDE simulations and evaluate Fig. 2e W2.

The fitted model and the initial AGIST cells are held fixed.  Each replicate
changes the PyTorch/NumPy random seed, which controls Brownian increments,
random interaction batches, and the split/extinction draws.  W2 is evaluated
in the original physical (x1--x2) and gene-state (x3--x52) coordinates.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml


DEFAULT_SEEDS = (1, 4, 8, 32, 256, 512, 1024, 2048, 4096, 8192)
SPACES = {
    "gene": np.arange(2, 52),
    "physical": np.arange(0, 2),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def load_models(project_root: Path, config_path: Path, checkpoint_dir: Path, device: str):
    sys.path.insert(0, str(project_root))
    from DeepRUOT.models import FNet_interaction, scoreNet2

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model_config = config["model"]
    f_net = FNet_interaction(
        in_out_dim=int(model_config["in_out_dim"]),
        hidden_dim=int(model_config["hidden_dim"]),
        n_hiddens=int(model_config["n_hiddens"]),
        activation=str(model_config["activation"]),
        use_spatial=bool(model_config.get("use_spatial", True)),
        num_heads=8,
        thre=float(model_config["thre"]),
        num_layers=1,
        edge_predictor_path=str(model_config["edge_predictor_path"]),
        edge_predictor_thre=float(model_config.get("edge_predictor_thre", 0.45)),
    ).to(device)
    score_net = scoreNet2(
        in_out_dim=int(model_config["in_out_dim"]),
        hidden_dim=int(model_config["score_hidden_dim"]),
        activation=str(model_config["activation"]),
    ).float().to(device)

    model_path = checkpoint_dir / "model_final"
    score_path = checkpoint_dir / "score_model"
    f_net.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    score_net.load_state_dict(torch.load(score_path, map_location=device, weights_only=False))
    f_net.eval()
    score_net.eval()
    for parameter in f_net.parameters():
        parameter.requires_grad_(False)
    for parameter in score_net.parameters():
        parameter.requires_grad_(False)
    return config, f_net, score_net, model_path, score_path


def simulate_split(
    x0: np.ndarray,
    f_net,
    score_net,
    *,
    device: str,
    dt: float,
    sigma: float,
    interaction_m: int,
    split_noise_std: float,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    from DeepRUOT.interaction import cal_interaction, euler_sdeint_split

    x0_tensor = torch.tensor(x0, dtype=torch.float32, device=device)
    lnw0 = torch.log(torch.ones(x0_tensor.shape[0], 1, device=device) / x0_tensor.shape[0])

    class SDE(torch.nn.Module):
        noise_type = "diagonal"
        sde_type = "ito"

        def f(self, t, state):
            z, lnw = state
            with torch.no_grad():
                drift = f_net.v_net(t, z)
                dlnw = f_net.g_net(t, z)
                interaction = cal_interaction(
                    z, lnw, f_net.interaction_net, t, m=int(interaction_m)
                )
            # Evaluate the score gradient on a detached leaf.  This is
            # numerically the same field evaluation, while preventing an
            # inference-only autograd graph from accumulating across Euler
            # steps.
            score_input = z.detach().requires_grad_(True)
            expanded_t = t.expand(z.shape[0], 1)
            score = score_net.compute_gradient(expanded_t, score_input).detach()
            return drift + interaction + score, dlnw

        def g(self, t, z):
            return torch.ones_like(z) * float(sigma)

    times = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float32, device=device)
    points, log_weights = euler_sdeint_split(
        SDE(),
        (x0_tensor, lnw0),
        dt=float(dt),
        ts=times,
        noise_std=float(split_noise_std),
    )
    point_arrays = [value.detach().cpu().numpy().astype(np.float32) for value in points]
    weight_arrays = [
        np.exp(value.detach().cpu().numpy().astype(np.float64)).reshape(-1)
        for value in log_weights
    ]
    return point_arrays, weight_arrays


def exact_w2(pred: np.ndarray, truth: np.ndarray, pred_weight: np.ndarray) -> float:
    import ot

    pred = np.asarray(pred, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    pred_weight = np.asarray(pred_weight, dtype=np.float64).reshape(-1)
    pred_weight /= pred_weight.sum()
    truth_weight = np.full(truth.shape[0], 1.0 / truth.shape[0], dtype=np.float64)
    cost = ot.dist(truth, pred, metric="sqeuclidean")
    value = ot.emd2(truth_weight, pred_weight, cost, numItermax=10_000_000)
    del cost
    return float(math.sqrt(max(0.0, value)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--data-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--sigma", type=float, default=0.03)
    parser.add_argument("--interaction-m", type=int, default=1024)
    parser.add_argument("--split-noise-std", type=float, default=0.0)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--simulate-only", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trajectory_dir = args.output_dir / "trajectories"
    trajectory_dir.mkdir(exist_ok=True)

    config, f_net, score_net, model_path, score_path = load_models(
        args.project_root.resolve(), args.config.resolve(), args.checkpoint_dir.resolve(), args.device
    )
    data = pd.read_csv(args.data_csv)
    feature_cols = [f"x{i}" for i in range(1, 53)]
    x0 = data.loc[data["samples"] == 0, feature_cols].to_numpy(np.float32)
    truth_by_time = {
        float(time): frame[feature_cols].to_numpy(np.float64)
        for time, frame in data.groupby("samples", sort=True)
    }

    rows: list[dict[str, object]] = []
    replicate_meta: list[dict[str, object]] = []
    for seed in args.seeds:
        seed = int(seed)
        set_seed(seed)
        points, weights = simulate_split(
            x0,
            f_net,
            score_net,
            device=args.device,
            dt=args.dt,
            sigma=args.sigma,
            interaction_m=args.interaction_m,
            split_noise_std=args.split_noise_std,
        )
        np.savez_compressed(
            trajectory_dir / f"split_sde_seed_{seed}.npz",
            points=np.asarray(points, dtype=object),
            weights=np.asarray(weights, dtype=object),
            times=np.asarray([0.0, 1.0, 2.0, 3.0]),
        )
        replicate_meta.append(
            {
                "seed": seed,
                "n_particles": [int(value.shape[0]) for value in points],
            }
        )
        if args.simulate_only:
            continue
        for time_index, time in enumerate((1.0, 2.0, 3.0), start=1):
            pred_all = points[time_index]
            pred_weights = weights[time_index]
            for space, indices in SPACES.items():
                value = exact_w2(
                    pred_all[:, indices],
                    truth_by_time[time][:, indices],
                    pred_weights,
                )
                rows.append(
                    {
                        "method": "CytoBridge",
                        "seed": seed,
                        "time": time,
                        "space": space,
                        "w2": value,
                        "n_initial": int(x0.shape[0]),
                        "n_predicted": int(pred_all.shape[0]),
                        "n_truth": int(truth_by_time[time].shape[0]),
                    }
                )
                pd.DataFrame(rows).to_csv(args.output_dir / "w2_replicates_long.csv", index=False)

    manifest = {
        "analysis": "AGIST Fig. 2e stochastic split-SDE repeats",
        "replicate_definition": (
            "independent PyTorch/NumPy seeds with fixed trained checkpoint and fixed initial cells; "
            "seed changes Brownian increments, interaction batches, and split/extinction draws"
        ),
        "seeds": [int(value) for value in args.seeds],
        "dt": float(args.dt),
        "sigma": float(args.sigma),
        "interaction_m": int(args.interaction_m),
        "split_noise_std": float(args.split_noise_std),
        "initial_cell_policy": "all AGIST t=0 cells in their archived order",
        "metric": "exact empirical Wasserstein-2 using POT network simplex",
        "spaces": {name: [int(value) for value in indices] for name, indices in SPACES.items()},
        "config": str(args.config.resolve()),
        "config_sha256": sha256_file(args.config.resolve()),
        "data_csv": str(args.data_csv.resolve()),
        "data_sha256": sha256_file(args.data_csv.resolve()),
        "model": str(model_path.resolve()),
        "model_sha256": sha256_file(model_path.resolve()),
        "score_model": str(score_path.resolve()),
        "score_model_sha256": sha256_file(score_path.resolve()),
        "model_config": config["model"],
        "replicates": replicate_meta,
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
