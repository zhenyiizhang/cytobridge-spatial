import os
import sys
import time
from dataclasses import dataclass
from hashlib import sha1
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union


def add_project_root() -> str:
    """Ensure the CytoBridge-ST-1104 project root is on sys.path."""
    here = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(here, "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    return project_root


def load_config(config_path: str) -> Dict:
    add_project_root()
    from DeepRUOT.utils import load_and_merge_config

    return load_and_merge_config(config_path)


def load_arista_df(config: Dict):
    add_project_root()
    import pandas as pd
    from DeepRUOT.constants import DATA_DIR

    csv_path = os.path.join(DATA_DIR, config["data"]["file_path"])
    df = pd.read_csv(csv_path)
    df = df.iloc[:, : config["data"]["dim"] + 1]
    return df, csv_path


def load_models(
    config: Dict,
    exp_name: Optional[str] = None,
    device: Optional[str] = None,
    model_tag: str = "model_result",
    score_tag: str = "score_model",
):
    add_project_root()
    import torch
    from DeepRUOT.constants import RES_DIR
    from DeepRUOT.exp import setup_exp
    from DeepRUOT.models import FNet_interaction, scoreNet2

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    exp_dir, _ = setup_exp(RES_DIR, config, exp_name or config["exp"]["name"])
    model_config = config["model"]

    edge_predictor_path = model_config.get("edge_predictor_path")
    edge_predictor_thre = model_config.get("edge_predictor_thre", 0.5)

    f_net = FNet_interaction(
        in_out_dim=model_config["in_out_dim"],
        hidden_dim=model_config["hidden_dim"],
        n_hiddens=model_config["n_hiddens"],
        activation=model_config["activation"],
        use_spatial=True,
        num_heads=8,
        thre=model_config["thre"],
        num_layers=1,
        edge_predictor_path=edge_predictor_path,
        edge_predictor_thre=edge_predictor_thre,
    ).to(device)

    score_net = scoreNet2(
        in_out_dim=model_config["in_out_dim"],
        hidden_dim=model_config["score_hidden_dim"],
        activation=model_config["activation"],
    ).float().to(device)

    f_net.load_state_dict(
        torch.load(os.path.join(exp_dir, model_tag), map_location=torch.device(device))
    )
    score_net.load_state_dict(
        torch.load(os.path.join(exp_dir, score_tag), map_location=torch.device(device))
    )

    return f_net, score_net, exp_dir, device


def plot_g_values(
    df,
    dim: int,
    f_net,
    exp_dir: str,
    time_index: int = 0,
    dim_reducer=None,
    device: str = "cpu",
    out_name: str = "g_values_plot.pdf",
):
    import numpy as np
    import matplotlib.pyplot as plt
    import torch

    time_points = df["samples"].unique()
    if time_index < 0 or time_index >= len(time_points):
        raise ValueError(f"time_index={time_index} out of range [0, {len(time_points)-1}]")

    data_by_time = {}
    for time in [time_points[time_index]]:
        subset = df[df["samples"] == time]
        column_names = [f"x{i}" for i in range(1, dim + 1)]
        tensors = [torch.tensor(subset[col].values, dtype=torch.float32).to(device) for col in column_names]
        data = torch.stack(tensors, dim=1)
        with torch.no_grad():
            t = torch.tensor([time], dtype=torch.float32).to(device)
            _, g, _, _ = f_net(t, data)
        data_by_time[time] = {"data": subset, "g_values": g.detach().cpu().numpy()}

    all_g_values = np.concatenate([content["g_values"] for content in data_by_time.values()])
    vmax_value = np.percentile(all_g_values, 95)
    norm = plt.Normalize(vmin=0, vmax=vmax_value, clip=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    for time, content in data_by_time.items():
        subset = content["data"]
        g_values = content["g_values"]
        column_names = [f"x{i}" for i in range(1, dim + 1)]
        new_data = subset[column_names]
        if dim_reducer is not None:
            data_reduced = dim_reducer.transform(new_data)
        else:
            data_reduced = new_data.iloc[:, :2].values
        x = data_reduced[:, 0]
        y = data_reduced[:, 1]
        colors = plt.cm.rainbow(norm(g_values))
        ax.scatter(x, y, c=colors, label=f"Time {time}", s=0.5, alpha=0.7, marker="o")

    ax.set_xlabel("Gene X1")
    ax.set_ylabel("Gene X2")
    ax.legend()

    sm = plt.cm.ScalarMappable(cmap="rainbow", norm=norm)
    sm.set_array(all_g_values)
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Normalized predicted growth rate")
    cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2f}"))

    out_path = os.path.join(exp_dir, out_name)
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    return out_path


def simulate_sde_points(
    df,
    dim: int,
    f_net,
    score_net,
    time_index: int = 0,
    n_samples: int = 5000,
    ts_points: Optional[Sequence[float]] = None,
    dt: float = 0.1,
    sigma: float = 0.0,
    include_score: bool = False,
    interaction_m: int = 512,
    device: str = "cuda",
    verbose: bool = True,
):
    """Simulate SDE and return (sde_points, weights)."""
    add_project_root()
    import numpy as np
    import torch
    from DeepRUOT.interaction import cal_interaction
    from DeepRUOT.utils import euler_sdeint

    if ts_points is None:
        ts_points = [0, 1, 2, 3, 4]

    time_points = df["samples"].unique()
    if time_index < 0 or time_index >= len(time_points):
        raise ValueError(f"time_index={time_index} out of range [0, {len(time_points)-1}]")

    t0 = time_points[time_index]
    numeric_cols = ["samples"] + [f"x{i}" for i in range(1, dim + 1)]
    data = torch.tensor(df[df["samples"] == t0][numeric_cols].values, dtype=torch.float32)
    x0 = data[:, 1:].requires_grad_().to(device)

    if x0.shape[0] > n_samples:
        indices = torch.randperm(x0.shape[0])[:n_samples]
        x0 = x0[indices]

    lnw0 = torch.log(torch.ones(x0.shape[0], 1, device=device) / x0.shape[0])
    initial_state = (x0, lnw0)

    class SDE(torch.nn.Module):
        noise_type = "diagonal"
        sde_type = "ito"

        def __init__(self, ode_drift, g, score, interaction, sigma):
            super().__init__()
            self.drift = ode_drift
            self.score = score
            self.interaction = interaction
            self.g_net = g
            self.sigma = sigma

        def f(self, t, y):
            z, lnw = y
            with torch.no_grad():
                drift = self.drift(t, z)
                dlnw = self.g_net(t, z)
                net_forces = cal_interaction(z, lnw, self.interaction, t, m=interaction_m)
            if include_score:
                t_expand = t.expand(z.shape[0], 1)
                with torch.enable_grad():
                    z_req = z.detach().requires_grad_(True)
                    drift = drift + self.score.compute_gradient(t_expand, z_req)
            return (drift + net_forces, dlnw)

        def g(self, t, y):
            return torch.ones_like(y) * self.sigma

    if verbose:
        try:
            t_min = float(min(ts_points))
            t_max = float(max(ts_points))
            est_steps = int(round((t_max - t_min) / dt)) if dt > 0 else None
        except Exception:
            t_min, t_max, est_steps = None, None, None
        print(
            "[simulate_sde_points] start | "
            f"n_init={x0.shape[0]}, ts_points={len(ts_points)}, "
            f"dt={dt}, sigma={sigma}, include_score={include_score}, "
            f"t_range=({t_min},{t_max}), est_steps={est_steps}"
        )

    sde = SDE(f_net.v_net, f_net.g_net, score_net, f_net.interaction_net, sigma=sigma)
    ts_tensor = torch.tensor(ts_points, dtype=torch.float32, device=device)
    sde_point, traj_lnw = euler_sdeint(sde, initial_state, dt=dt, ts=ts_tensor)

    weight = torch.exp(traj_lnw)
    weight_normed = weight / weight.sum(dim=1, keepdim=True)

    sde_point_np = [p.detach().cpu().numpy() for p in sde_point]
    if verbose:
        print(
            "[simulate_sde_points] done | "
            f"timepoints={len(sde_point_np)}, "
            f"shape0={sde_point_np[0].shape if sde_point_np else None}"
        )
    return np.array(sde_point_np, dtype=object), weight_normed.detach().cpu().numpy()


def simulate_sde_points_split(
    df,
    dim: int,
    f_net,
    score_net,
    time_index: int = 0,
    n_samples: int = 5000,
    ts_points: Optional[Sequence[float]] = None,
    dt: float = 0.01,
    sigma: float = 0.03,
    sigma_by_dim: Optional[Sequence[float]] = None,
    growth_alpha: float = 0.5,
    interaction_m: int = 1024,
    device: str = "cuda",
    verbose: bool = True,
):
    """Simulate SDE with split (cell division) and return sde_points."""
    add_project_root()
    import numpy as np
    import torch
    from DeepRUOT.interaction import cal_interaction, euler_sdeint_split

    if ts_points is None:
        ts_points = [0, 1, 2, 3, 4]

    time_points = df["samples"].unique()
    if time_index < 0 or time_index >= len(time_points):
        raise ValueError(f"time_index={time_index} out of range [0, {len(time_points)-1}]")

    t0 = time_points[time_index]
    numeric_cols = ["samples"] + [f"x{i}" for i in range(1, dim + 1)]
    data = torch.tensor(df[df["samples"] == t0][numeric_cols].values, dtype=torch.float32)
    x0 = data[:, 1:].to(device)

    if x0.shape[0] > n_samples:
        indices = torch.randperm(x0.shape[0])[:n_samples]
        x0 = x0[indices]

    lnw0 = torch.log(torch.ones(x0.shape[0], 1, device=device) / x0.shape[0])
    initial_state = (x0, lnw0)

    class SDE(torch.nn.Module):
        noise_type = "diagonal"
        sde_type = "ito"

        def __init__(self, ode_drift, g, score, interaction, sigma, sigma_by_dim):
            super().__init__()
            self.drift = ode_drift
            self.score = score
            self.interaction = interaction
            self.g_net = g
            if sigma_by_dim is None:
                self.register_buffer("sigma_vec", None)
                self.sigma = float(sigma)
            else:
                sigma_arr = np.asarray(list(sigma_by_dim), dtype=np.float32).reshape(-1)
                if sigma_arr.shape[0] != dim:
                    raise ValueError(
                        f"sigma_by_dim must have length {dim}, got {sigma_arr.shape[0]}"
                    )
                self.register_buffer("sigma_vec", torch.tensor(sigma_arr, dtype=torch.float32))
                self.sigma = None

        def f(self, t, y):
            z, lnw = y
            with torch.no_grad():
                drift = self.drift(t, z)
                dlnw = self.g_net(t, z) * growth_alpha
                net_forces = cal_interaction(z, lnw, self.interaction, t, m=interaction_m)
            t_expand = t.expand(z.shape[0], 1)
            score_grad = self.score.compute_gradient(t_expand, z)
            return (drift + score_grad + net_forces, dlnw)

        def g(self, t, y):
            if self.sigma_vec is None:
                return torch.ones_like(y) * self.sigma
            return self.sigma_vec.to(device=y.device, dtype=y.dtype).unsqueeze(0).expand_as(y)

    if verbose:
        try:
            t_min = float(min(ts_points))
            t_max = float(max(ts_points))
            est_steps = int(round((t_max - t_min) / dt)) if dt > 0 else None
        except Exception:
            t_min, t_max, est_steps = None, None, None
        print(
            "[simulate_sde_points_split] start | "
            f"n_init={x0.shape[0]}, ts_points={len(ts_points)}, "
            f"dt={dt}, sigma={'vector' if sigma_by_dim is not None else sigma}, "
            f"growth_alpha={growth_alpha}, "
            f"t_range=({t_min},{t_max}), est_steps={est_steps}"
        )

    sde = SDE(
        f_net.v_net,
        f_net.g_net,
        score_net,
        f_net.interaction_net,
        sigma=sigma,
        sigma_by_dim=sigma_by_dim,
    )
    ts_tensor = torch.tensor(ts_points, dtype=torch.float32, device=device)
    sde_points, _ = euler_sdeint_split(sde, initial_state, dt=dt, ts=ts_tensor, noise_std=0.0)
    sde_point_np = [p.detach().cpu().numpy() for p in sde_points]
    if verbose:
        print(
            "[simulate_sde_points_split] done | "
            f"timepoints={len(sde_point_np)}, "
            f"shape0={sde_point_np[0].shape if sde_point_np else None}"
        )
    return np.array(sde_point_np, dtype=object)


def plot_sde_vs_real(
    df,
    sde_points,
    time_values: Sequence[float],
    dim_pairs: Sequence[Tuple[int, int]] = ((0, 1),),
    annotation_key: Optional[str] = None,
    out_prefix: Optional[str] = None,
):
    import matplotlib.pyplot as plt
    import numpy as np

    for t_idx, t_val in enumerate(time_values):
        real = df[df["samples"] == t_val]
        if real.empty:
            continue
        sim = np.asarray(sde_points[t_idx], dtype=float)

        for d1, d2 in dim_pairs:
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.scatter(real.iloc[:, 1 + d1], real.iloc[:, 1 + d2], c="red", s=3, alpha=0.6, label="real")
            ax.scatter(sim[:, d1], sim[:, d2], c="blue", s=3, alpha=0.6, label="sim")
            ax.set_title(f"SDE vs real (t={t_val}) dims {d1+1},{d2+1}")
            ax.legend(loc="best")
            if out_prefix:
                out_path = f"{out_prefix}_t{t_val}_d{d1+1}_{d2+1}.png"
                fig.savefig(out_path, dpi=300, bbox_inches="tight")
            plt.show()


def compute_velocity_components(
    data: "np.ndarray",
    time_value: float,
    f_net,
    score_net,
    interaction_m: int = 1024,
    interaction_threshold: int = 1000,
    device: str = "cuda",
):
    """Return velocity components (drift, interaction, score) for a given time."""
    add_project_root()
    import numpy as np
    import torch
    from DeepRUOT.interaction import cal_interaction

    data = np.asarray(data, dtype=np.float32)
    n_cells = data.shape[0]

    data_tensor = torch.tensor(data, device=device, requires_grad=True)
    t_tensor = torch.full((n_cells, 1), float(time_value), device=device)

    with torch.no_grad():
        drift = f_net.v_net(t_tensor, data_tensor)
    drift_np = drift.detach().cpu().numpy()

    lnw = torch.log(torch.ones(n_cells, 1, device=device) / n_cells)
    with torch.no_grad():
        interaction_t = cal_interaction(
            data_tensor.detach(),
            lnw,
            f_net.interaction_net,
            torch.tensor([float(time_value)], dtype=torch.float32, device=device),
            m=interaction_m,
            threshold=interaction_threshold,
        )
    interaction_np = interaction_t.detach().cpu().numpy()

    logp = score_net(t_tensor, data_tensor)
    ones = torch.ones_like(logp)
    score_grad = torch.autograd.grad(
        outputs=logp,
        inputs=data_tensor,
        grad_outputs=ones,
        retain_graph=False,
        create_graph=False,
        only_inputs=True,
    )[0]
    score_np = score_grad.detach().cpu().numpy()

    return {
        "drift": drift_np,
        "interaction": interaction_np,
        "score": score_np,
        "full": drift_np + interaction_np + score_np,
    }


def compute_umap_embedding(data: "np.ndarray", n_neighbors: int = 30, min_dist: float = 0.3, seed: int = 0):
    """Compute a 2D UMAP embedding for gene space visualization."""
    import numpy as np
    try:
        import umap
    except Exception as exc:
        raise ImportError("umap-learn is required for UMAP embedding. Install it in the DeepRUOTv2 env.") from exc

    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=seed)
    embedding = reducer.fit_transform(np.asarray(data, dtype=float))
    return embedding, reducer


def plot_velocity_component(
    coords: "np.ndarray",
    velocity: "np.ndarray",
    labels: Optional[Sequence[str]] = None,
    label_to_color: Optional[Dict[str, str]] = None,
    title: str = "",
    out_path: Optional[str] = None,
    density: float = 2.0,
    basis: str = "spatial",
    show_legend: bool = False,
):
    import numpy as np
    import matplotlib.pyplot as plt
    import anndata as ad
    import scanpy as sc
    import scvelo as scv

    adata = ad.AnnData(X=coords)
    adata.obsm["X_spatial"] = coords
    adata.layers["Ms"] = coords
    adata.layers["velocity"] = velocity

    palette_list = None
    if labels is not None:
        labels_arr = np.asarray(labels).astype(str)
        if label_to_color is None:
            import matplotlib.pyplot as plt

            uniq = sorted(set(labels_arr))
            cmap = plt.get_cmap("tab20", len(uniq))
            label_to_color = {u: cmap(i) for i, u in enumerate(uniq)}
            label_to_color = {k: "#{:02x}{:02x}{:02x}".format(int(v[0] * 255), int(v[1] * 255), int(v[2] * 255)) for k, v in label_to_color.items()}

        categories = [c for c in label_to_color.keys() if c in set(labels_arr)]
        if not categories:
            categories = sorted(set(labels_arr))
        adata.obs["Annotation"] = labels_arr
        adata.obs["Annotation"] = adata.obs["Annotation"].astype("category")
        adata.obs["Annotation"] = adata.obs["Annotation"].cat.reorder_categories(categories, ordered=True)
        palette_list = [label_to_color.get(cat, "#888888") for cat in categories]
        adata.uns["Annotation_colors"] = palette_list

    sc.pp.neighbors(adata, n_neighbors=30, use_rep="X")
    scv.tl.velocity_graph(adata, vkey="velocity")
    scv.tl.velocity_embedding(adata, basis=basis, vkey="velocity")
    scv.settings.set_figure_params("scvelo")

    fig, ax = plt.subplots(figsize=(6, 6))
    legend_loc = "right margin" if show_legend else "none"
    scv.pl.velocity_embedding_stream(
        adata,
        basis=basis,
        color="Annotation" if labels is not None else None,
        palette=palette_list,
        density=density,
        ax=ax,
        show=False,
        title=title,
        legend_loc=legend_loc,
    )
    if out_path:
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()
    return adata


def save_interpolated_attention(
    adata,
    time_value: float,
    f_net,
    device: str = "cpu",
    out_dir: Optional[str] = None,
    save_files: bool = True,
    save_dense_matrix: bool = True,
):
    import numpy as np
    import os
    import torch

    f_net = f_net.to(device)
    data = torch.tensor(adata.X, dtype=torch.float32).to(device)
    n_particles = data.shape[0]
    lnw0 = torch.log(torch.ones(n_particles, 1, device=device) / n_particles)
    time_tensor = torch.tensor(time_value, dtype=torch.float32).to(device)

    with torch.no_grad():
        _ = f_net.interaction_net(data, lnw0, time_tensor, return_attn=True)
        attn = f_net.interaction_net.gnn_layers[0].attn

    attn = torch.abs(attn)
    attn_mean = attn.mean(dim=1).cpu().numpy()
    edge_index = f_net.interaction_net.edge_index.cpu().numpy()
    edge_index = edge_index[:, edge_index[0] != edge_index[1]]

    out = {"attn_mean": attn_mean, "edge_index": edge_index}

    attn_matrix = None
    if save_dense_matrix:
        # Warning: this is O(N^2) memory. For large slices, consider save_dense_matrix=False
        # and use edge_index + attn_mean directly.
        attn_matrix = np.zeros((n_particles, n_particles), dtype=float)
        attn_matrix[edge_index[0], edge_index[1]] = attn_mean
        out["attn_matrix"] = attn_matrix

    if save_files:
        if out_dir is None:
            out_dir = os.getcwd()
        os.makedirs(out_dir, exist_ok=True)

        if save_dense_matrix and attn_matrix is not None:
            np.save(os.path.join(out_dir, f"attn_interp_t{time_value}.npy"), attn_matrix)
        np.save(os.path.join(out_dir, f"attn_mean_interp_t{time_value}.npy"), attn_mean)
        np.save(os.path.join(out_dir, f"edge_index_interp_t{time_value}.npy"), edge_index)

    return out


def analyze_attention_by_celltype(
    edge_index,
    attn,
    labels,
    spatial_coord=None,
    time_title=None,
    remove_self_loop=True,
    winsor_quantile=0.995,
    distance_bins="quartile",
    n_permutations: int = 0,
    random_state: int = 0,
    plot: bool = True,
):
    """
    Aggregate attention weights at the cell-type level with optional distance stratification
    and permutation testing (ported from MOSTA multilayer communication notebook).
    """
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from scipy.sparse import coo_matrix

    edge_index = np.asarray(edge_index)
    attn = np.asarray(attn).astype(float)
    labels = np.asarray(labels)

    if edge_index.shape[0] != 2:
        raise ValueError("edge_index should have shape (2, E)")
    if attn.shape[0] != edge_index.shape[1]:
        raise ValueError("attn length must match number of edges")
    if spatial_coord is not None:
        spatial_coord = np.asarray(spatial_coord)
        if spatial_coord.shape[0] != labels.shape[0] or spatial_coord.shape[1] != 2:
            raise ValueError("spatial_coord must be N×2 and align with labels order")

    send = edge_index[0].copy()
    recv = edge_index[1].copy()
    w = attn.copy()

    if remove_self_loop:
        m = send != recv
        send, recv, w = send[m], recv[m], w[m]

    # Handle the degenerate case: no edges after filtering.
    # This can happen for very small slices (e.g. smoke tests) or strict graph thresholds.
    if w.size == 0:
        types, type_id = np.unique(labels, return_inverse=True)
        T = len(types)
        n_per_type = np.bincount(type_id, minlength=T).astype(float)
        n_per_type[n_per_type == 0] = 1.0

        M_sum = np.zeros((T, T), dtype=float)
        M_per_source = np.zeros((T, T), dtype=float)
        M_row = np.zeros((T, T), dtype=float)
        M_mean = np.zeros((T, T), dtype=float)
        asym = np.zeros((T, T), dtype=float)
        edge_counts = np.zeros((T, T), dtype=float)

        type_stats = pd.DataFrame(
            {
                "type": types,
                "out_strength": np.zeros(T, dtype=float),
                "in_strength": np.zeros(T, dtype=float),
                "net_out_minus_in": np.zeros(T, dtype=float),
            }
        ).sort_values("net_out_minus_in", ascending=False)

        result = {
            "types": types,
            "M_sum": M_sum,
            "M_per_source": M_per_source,
            "M_row": M_row,
            "M_mean": M_mean,
            "asym": asym,
            "type_stats": type_stats,
            "edge_counts": edge_counts,
        }
        return result

    if winsor_quantile is not None and 0.9 < winsor_quantile < 1.0:
        hi = np.quantile(w, winsor_quantile)
        w = np.minimum(w, hi)

    types, type_id = np.unique(labels, return_inverse=True)
    T = len(types)
    n_per_type = np.bincount(type_id, minlength=T).astype(float)
    n_per_type[n_per_type == 0] = 1.0

    M_sum = coo_matrix((w, (type_id[send], type_id[recv])), shape=(T, T)).toarray()
    M_per_source = M_sum / n_per_type[:, None]
    row_sums = M_sum.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    M_row = M_sum / row_sums

    edge_counts = coo_matrix((np.ones_like(w), (type_id[send], type_id[recv])), shape=(T, T)).toarray()
    edge_counts_safe = edge_counts.copy()
    edge_counts_safe[edge_counts_safe == 0] = 1.0
    M_mean = M_sum / edge_counts_safe

    out_strength = M_sum.sum(axis=1)
    in_strength = M_sum.sum(axis=0)
    type_stats = (
        pd.DataFrame(
            {
                "type": types,
                "out_strength": out_strength,
                "in_strength": in_strength,
                "net_out_minus_in": out_strength - in_strength,
            }
        ).sort_values("net_out_minus_in", ascending=False)
    )

    asym = M_per_source - M_per_source.T

    fdr = None
    sig_mask = None
    if n_permutations and n_permutations > 0:
        rng = np.random.default_rng(random_state)
        null = np.zeros((T, T, n_permutations))
        for b in range(n_permutations):
            shuf = rng.permutation(type_id)
            Mb = coo_matrix((w, (shuf[send], shuf[recv])), shape=(T, T)).toarray()
            null[..., b] = Mb / n_per_type[:, None]
        obs = M_per_source
        p = (null >= obs[..., None]).mean(axis=2)
        p_flat = p.ravel()
        order = np.argsort(p_flat)
        rank = np.empty_like(order)
        rank[order] = np.arange(1, p_flat.size + 1)
        fdr_flat = p_flat * p_flat.size / np.maximum(rank, 1)
        fdr = fdr_flat.reshape(T, T)
        sig_mask = fdr < 0.05

    distance_panels = None
    if spatial_coord is not None:
        pair_dist = np.linalg.norm(spatial_coord[recv] - spatial_coord[send], axis=1)
        if distance_bins == "quartile":
            bins = np.quantile(pair_dist, [0, 0.25, 0.5, 0.75, 1.0])
        elif isinstance(distance_bins, (list, tuple, np.ndarray)):
            bins = np.asarray(distance_bins, dtype=float)
            if not np.all(np.diff(bins) > 0):
                raise ValueError("distance_bins must be strictly increasing")
        else:
            bins = None

        if bins is not None:
            bin_id = np.digitize(pair_dist, bins, right=True)
            Mps_list = []
            for b in range(1, len(bins) + 1):
                m = bin_id == b
                Ms = coo_matrix((w[m], (type_id[send[m]], type_id[recv[m]])), shape=(T, T)).toarray()
                Mps = Ms / n_per_type[:, None]
                Mps_list.append(Mps)
            distance_panels = {"bins": bins, "M_per_source_bybin": Mps_list}

    if plot:
        sns.set_theme(
            style="whitegrid",
            font_scale=1.1,
            rc={"axes.facecolor": "white", "figure.facecolor": "white", "axes.labelcolor": "black", "text.color": "black", "font.family": "Arial"},
        )

        cmap_main = sns.color_palette("PuBu", as_cmap=True)
        cmap_asym = sns.diverging_palette(250, 10, as_cmap=True)
        title_prefix = f" (time={time_title})" if time_title is not None else ""

        plt.figure(figsize=(6.8, 5.6))
        ax = sns.heatmap(
            M_per_source,
            xticklabels=types,
            yticklabels=types,
            cmap=cmap_main,
            square=True,
            cbar_kws={"label": "Attention (A→B)"},
            linewidths=0.4,
            linecolor="white",
        )
        plt.title(f"Per-source-cell attention A→B{title_prefix}")
        plt.xlabel("Receiver type (B)")
        plt.ylabel("Sender type (A)")
        if sig_mask is not None:
            yy, xx = np.where(sig_mask)
            for i, j in zip(yy, xx):
                ax.text(j + 0.5, i + 0.5, "•", ha="center", va="center", fontsize=9, color="k")
            plt.suptitle("• : FDR < 0.05", y=1.02, fontsize=10)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(6.8, 5.6))
        sns.heatmap(
            asym,
            xticklabels=types,
            yticklabels=types,
            cmap=cmap_asym,
            center=0,
            square=True,
            linewidths=0.4,
            linecolor="white",
            cbar_kws={"label": "Asymmetry (A→B minus B→A)"},
        )
        plt.title(f"Directional asymmetry (per-source normalized){title_prefix}")
        plt.xlabel("B")
        plt.ylabel("A")
        plt.tight_layout()
        plt.show()

        if distance_panels is not None:
            bins = distance_panels["bins"]
            Mps_list = distance_panels["M_per_source_bybin"]
            k = len(Mps_list)
            cols = min(3, k)
            rows = int(np.ceil(k / cols))
            fig, axes = plt.subplots(rows, cols, figsize=(6.2 * cols, 5.2 * rows))
            axes = np.array(axes).reshape(rows, cols)
            for idx, Mps in enumerate(Mps_list):
                r, c = divmod(idx, cols)
                ax_bin = axes[r, c]
                sns.heatmap(Mps, ax=ax_bin, xticklabels=types, yticklabels=types, cmap="YlOrRd")
                lo = bins[idx - 1] if idx > 0 else bins[idx]
                hi = bins[idx] if idx < len(bins) else bins[-1]
                ax_bin.set_title(f"Per-source (dist in [{lo:.2f}, {hi:.2f}))")
                ax_bin.set_xlabel("B")
                ax_bin.set_ylabel("A")
            for j in range(k, rows * cols):
                r, c = divmod(j, cols)
                fig.delaxes(axes[r, c])
            fig.suptitle(f"Distance-stratified attention{title_prefix}", y=0.99, fontsize=12)
            fig.tight_layout()
            plt.show()

    result = {
        "types": types,
        "M_sum": M_sum,
        "M_per_source": M_per_source,
        "M_row": M_row,
        "M_mean": M_mean,
        "asym": asym,
        "type_stats": type_stats,
        "edge_counts": edge_counts,
    }
    if fdr is not None:
        result["fdr"] = fdr
        result["sig_mask"] = sig_mask
    if distance_panels is not None:
        result["distance_bins"] = bins
        result["M_per_source_bybin"] = Mps_list

    return result


def plot_3d_spatial_sankey_style(
    adata_dict,
    all_time_communications,
    time_keys,
    label_to_color,
    predicted_labels_list,
    spatial_key="spatial",
    intra_threshold=1.0,
    ribbon_resolution=20,
    ribbon_width_scale=0.01,
    z_spacing=3.0,
    focus_celltype=None,
    edge_focus_celltype=None,
    edge_top_k=None,
    edge_top_k_focus_label: Optional[str] = None,
    edge_weight_quantile=None,
    edge_width_scale=1.0,
    edge_global_top_k=None,
    edge_render_mode="line",
    edge_line_width_base=1.0,
    edge_line_width_scale=1.0,
    edge_color: Optional[str] = None,
    ribbon_top_k=None,
    ribbon_count_quantile=None,
    ribbon_min_count: Optional[float] = None,
    ribbon_keep_source_cumfrac: Optional[float] = None,
    ribbon_focus_celltype=None,
    ribbon_focus_target_only=False,
    ribbon_render_mode="line",
    ribbon_line_width_base=1.0,
    ribbon_line_width_scale=1.0,
    ribbon_line_alpha: float = 0.6,
    ribbon_line_curve: float = 0.0,
    ribbon_line_points: int = 16,
    point_size=1.0,
    point_subsample: Optional[Union[int, float]] = None,
    observed_point_subsample: Optional[Union[int, float]] = None,
    generated_point_subsample: Optional[Union[int, float]] = None,
    point_alpha: float = 0.6,
    observed_point_alpha: Optional[float] = None,
    generated_point_alpha: Optional[float] = None,
    point_line_width: float = 0.0,
    point_line_color: Optional[str] = None,
    observed_point_line_width: Optional[float] = None,
    observed_point_line_color: Optional[str] = None,
    generated_point_line_width: Optional[float] = None,
    generated_point_line_color: Optional[str] = None,
    show_centroid_nodes=False,
    centroid_node_size=6.0,
    centroid_node_opacity=0.9,
    highlight_endpoints=False,
    endpoint_size=8.0,
    endpoint_opacity=0.95,
    background_color="white",
    font_color="black",
    reverse_time_order=False,
    anchor_mode: str = "centroid",
    anchor_subsample: Optional[int] = 1000,
    slices_only: bool = False,
    show_time_axis: bool = True,
    show_legend: bool = True,
    show_title: bool = True,
    width: int = 1400,
    height: int = 1000,
    include_self_loops=False,
    self_loop_radius_scale=0.6,
    self_loop_line_width_base=1.0,
    self_loop_line_width_scale=1.0,
    self_loop_points=36,
    flow_normalize_mode: Optional[str] = None,
    lineage_anchor_mode: bool = False,
    anchor_time_index: int = 0,
    ribbon_focus_source_only: bool = False,
    out_html=None,
    show_slice_border: bool = False,
    slice_border_color: str = "black",
    slice_border_width: float = 2.0,
    slice_border_color_observed: Optional[str] = None,
    slice_border_color_generated: Optional[str] = None,
    slice_fill_color_observed: Optional[str] = None,
    slice_fill_color_generated: Optional[str] = None,
    slice_fill_opacity: float = 0.0,
    observed_time_points: Optional[Sequence] = None,
    generated_time_points: Optional[Sequence] = None,
):
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go

    def to_valid_color(color_str, default_alpha=1.0):
        if not isinstance(color_str, str):
            return "rgba(136,136,136,1)"
        color_str = color_str.strip()
        if color_str.startswith("#") and len(color_str) == 9:
            r = int(color_str[1:3], 16)
            g = int(color_str[3:5], 16)
            b = int(color_str[5:7], 16)
            a = int(color_str[7:9], 16) / 255.0
            return f"rgba({r},{g},{b},{a:.3f})"
        if color_str.startswith("#") and len(color_str) == 7:
            if default_alpha < 1.0:
                r = int(color_str[1:3], 16)
                g = int(color_str[3:5], 16)
                b = int(color_str[5:7], 16)
                return f"rgba({r},{g},{b},{default_alpha})"
            return color_str
        return color_str

    def get_rgb_tuple(color_str):
        valid_c = to_valid_color(color_str)
        if valid_c.startswith("#"):
            return (int(valid_c[1:3], 16), int(valid_c[3:5], 16), int(valid_c[5:7], 16))
        if valid_c.startswith("rgba") or valid_c.startswith("rgb"):
            import re

            nums = [float(x) for x in re.findall(r"[\\d\\.]+", valid_c)]
            return (int(nums[0]), int(nums[1]), int(nums[2]))
        return (136, 136, 136)

    def cylinder_between(p1, p2, radius=0.3, n=16):
        p1, p2 = np.array(p1), np.array(p2)
        v = p2 - p1
        length = np.linalg.norm(v)
        if length < 1e-6:
            return None
        v = v / length
        not_v = np.array([1, 0, 0]) if abs(v[0]) < 0.9 else np.array([0, 1, 0])
        n1 = np.cross(v, not_v)
        n1 /= np.linalg.norm(n1)
        n2 = np.cross(v, n1)
        theta = np.linspace(0, 2 * np.pi, n)
        circle = radius * (np.outer(np.cos(theta), n1) + np.outer(np.sin(theta), n2))
        p1_ring, p2_ring = circle + p1, circle + p2
        x = np.hstack([p1_ring[:, 0], p2_ring[:, 0]])
        y = np.hstack([p1_ring[:, 1], p2_ring[:, 1]])
        z = np.hstack([p1_ring[:, 2], p2_ring[:, 2]])
        i, j, k = [], [], []
        N = len(theta)
        for t in range(N - 1):
            base = t
            next_t = t + 1
            i.extend([base, next_t, base + N])
            j.extend([next_t, next_t + N, next_t + N])
            k.extend([base + N, base + N, base])
        i.extend([N - 1, 0, 2 * N - 1])
        j.extend([0, N, N])
        k.extend([2 * N - 1, 2 * N - 1, N - 1])
        return x, y, z, i, j, k

    def create_ribbon_surface(c1, c2, count, color_from, color_to, n_points=20):
        width = ribbon_width_scale * count
        width = min(width, 10.0)
        t = np.linspace(0, 1, n_points)
        x_path = c1["x"] + t * (c2["x"] - c1["x"])
        y_path = c1["y"] + t * (c2["y"] - c1["y"])
        z_path = c1["z"] + t * (c2["z"] - c1["z"])
        bend = 0.2 * np.sin(np.pi * t) * (abs(c2["x"] - c1["x"]) + abs(c2["y"] - c1["y"]))
        x_path = x_path + bend * 0.1
        dx = np.gradient(x_path)
        dy = np.gradient(y_path)
        norm = np.sqrt(dx**2 + dy**2) + 1e-6
        perp_x = -dy / norm
        perp_y = dx / norm
        x_left, x_right = x_path + perp_x * width, x_path - perp_x * width
        y_left, y_right = y_path + perp_y * width, y_path - perp_y * width
        x_mesh = np.column_stack([x_left, x_right]).flatten()
        y_mesh = np.column_stack([y_left, y_right]).flatten()
        z_mesh = np.column_stack([z_path, z_path]).flatten()
        i_idx, j_idx, k_idx = [], [], []
        for idx in range(n_points - 1):
            base = idx * 2
            i_idx.extend([base, base + 1])
            j_idx.extend([base + 1, base + 3])
            k_idx.extend([base + 2, base + 2])
        rgb_from = get_rgb_tuple(color_from)
        rgb_to = get_rgb_tuple(color_to)
        vertex_colors = []
        for idx in range(n_points):
            blend = t[idx]
            r, g, b = [int(s * (1 - blend) + e * blend) for s, e in zip(rgb_from, rgb_to)]
            c = f"rgb({r},{g},{b})"
            vertex_colors.extend([c, c])
        return {"x": x_mesh, "y": y_mesh, "z": z_mesh, "i": i_idx, "j": j_idx, "k": k_idx, "colors": vertex_colors}

    if anchor_mode not in ("centroid", "nearest"):
        raise ValueError("anchor_mode must be 'centroid' or 'nearest'")
    if slices_only:
        show_centroid_nodes = False
        highlight_endpoints = False

    all_types = set()
    for tk in time_keys:
        ad = adata_dict[tk]
        ann_col = "Annotation" if "Annotation" in ad.obs.columns else "annotation"
        all_types.update(ad.obs[ann_col].unique())
    all_types = sorted(all_types)
    if reverse_time_order:
        z_values = [(len(time_keys) - 1 - i) * z_spacing for i in range(len(time_keys))]
    else:
        z_values = [i * z_spacing for i in range(len(time_keys))]

    fig = go.Figure()
    centroids = {}
    type_coords_by_time = {}
    endpoint_nodes = {}
    anchor_cache_same: Dict[tuple, Optional[tuple]] = {}
    anchor_cache_cross: Dict[tuple, Optional[tuple]] = {}
    rng = np.random.default_rng(0)

    def _subsample_coords(coords, max_n):
        if max_n is None or coords.shape[0] <= max_n:
            return coords
        idx = rng.choice(coords.shape[0], size=int(max_n), replace=False)
        return coords[idx]

    def _normalize_focus_labels(value):
        if value is None:
            return None
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            labels = [x.strip() for x in text.split(",")] if "," in text else [text]
            labels = [x for x in labels if x]
            return set(labels) if labels else None
        if isinstance(value, (list, tuple, set, np.ndarray, pd.Index)):
            labels = [str(x).strip() for x in value if str(x).strip()]
            return set(labels) if labels else None
        text = str(value).strip()
        return {text} if text else None

    def _filter_transitions_keep_source_cumfrac(df: "pd.DataFrame", cumfrac: float) -> "pd.DataFrame":
        cumfrac = float(cumfrac)
        if not (0.0 < cumfrac <= 1.0):
            raise ValueError("ribbon_keep_source_cumfrac must be in (0, 1].")
        if df.empty:
            return df

        keep_indices = []
        for _, g in df.groupby("source", sort=False):
            g = g.sort_values("count", ascending=False)
            total = float(g["count"].sum())
            if total <= 0:
                continue
            cum = (g["count"].cumsum() / total).to_numpy()
            keep_n = int(np.searchsorted(cum, cumfrac, side="left")) + 1
            keep_n = max(1, min(len(g), keep_n))
            keep_indices.extend(g.head(keep_n).index.tolist())

        if not keep_indices:
            return df.iloc[0:0]
        return df.loc[keep_indices]

    def _canonical_time_value(value):
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                return value
        try:
            return float(value)
        except (TypeError, ValueError):
            return str(value)

    observed_set = None
    generated_set = None
    if observed_time_points:
        observed_set = {_canonical_time_value(v) for v in observed_time_points}
    if generated_time_points:
        generated_set = {_canonical_time_value(v) for v in generated_time_points}

    def _is_generated(time_key):
        if generated_set is not None:
            return time_key in generated_set
        if observed_set is not None:
            return time_key not in observed_set
        return False

    def _resolve_point_alpha(is_generated):
        if is_generated and generated_point_alpha is not None:
            return generated_point_alpha
        if (not is_generated) and observed_point_alpha is not None:
            return observed_point_alpha
        return point_alpha

    def _resolve_point_line_width(is_generated):
        if is_generated and generated_point_line_width is not None:
            return generated_point_line_width
        if (not is_generated) and observed_point_line_width is not None:
            return observed_point_line_width
        return point_line_width

    def _resolve_point_line_color(is_generated):
        if is_generated and generated_point_line_color:
            return generated_point_line_color
        if (not is_generated) and observed_point_line_color:
            return observed_point_line_color
        return point_line_color

    def _resolve_subsample_indices(n, spec):
        if spec is None:
            return None
        try:
            val = float(spec)
        except (TypeError, ValueError):
            return None
        if val <= 0:
            return None
        if 0 < val < 1.0:
            size = int(round(n * val))
        else:
            size = int(round(val))
        size = max(1, min(n, size))
        if size >= n:
            return None
        return rng.choice(n, size=size, replace=False)

    def _resolve_subsample_spec(is_generated):
        if is_generated:
            if generated_point_subsample is not None:
                return generated_point_subsample
        else:
            if observed_point_subsample is not None:
                return observed_point_subsample
        return point_subsample

    def _resolve_slice_border_color(is_generated):
        if is_generated and slice_border_color_generated:
            return slice_border_color_generated
        if (not is_generated) and slice_border_color_observed:
            return slice_border_color_observed
        return slice_border_color

    def _resolve_slice_fill_color(is_generated):
        if is_generated and slice_fill_color_generated:
            return slice_fill_color_generated
        if (not is_generated) and slice_fill_color_observed:
            return slice_fill_color_observed
        return None

    def _nearest_pair(coords_a, coords_b):
        if coords_a.size == 0 or coords_b.size == 0:
            return None
        coords_a = _subsample_coords(coords_a, anchor_subsample)
        coords_b = _subsample_coords(coords_b, anchor_subsample)
        if coords_a.size == 0 or coords_b.size == 0:
            return None
        try:
            from scipy.spatial import cKDTree

            tree = cKDTree(coords_b)
            dists, idx = tree.query(coords_a, k=1)
            min_idx = int(np.argmin(dists))
            return coords_a[min_idx], coords_b[int(idx[min_idx])]
        except Exception:
            dmat = np.linalg.norm(coords_a[:, None, :] - coords_b[None, :, :], axis=2)
            min_idx = np.unravel_index(np.argmin(dmat), dmat.shape)
            return coords_a[min_idx[0]], coords_b[min_idx[1]]

    def _get_anchor_same_layer(tk, type_from, type_to, z_val):
        if anchor_mode != "nearest":
            layer_cents = centroids.get(tk, {})
            if type_from not in layer_cents or type_to not in layer_cents:
                return None
            return layer_cents[type_from], layer_cents[type_to]
        key = (tk, type_from, type_to)
        if key in anchor_cache_same:
            cached = anchor_cache_same[key]
            if cached is None:
                return None
            p1, p2 = cached
            return {"x": float(p1[0]), "y": float(p1[1]), "z": z_val}, {"x": float(p2[0]), "y": float(p2[1]), "z": z_val}
        coords_map = type_coords_by_time.get(tk, {})
        if type_from not in coords_map or type_to not in coords_map:
            anchor_cache_same[key] = None
            return None
        pair = _nearest_pair(coords_map[type_from], coords_map[type_to])
        if pair is None:
            anchor_cache_same[key] = None
            return None
        p1, p2 = pair
        anchor_cache_same[key] = (p1, p2)
        return {"x": float(p1[0]), "y": float(p1[1]), "z": z_val}, {"x": float(p2[0]), "y": float(p2[1]), "z": z_val}

    def _get_anchor_cross_layer(t1_key, t2_key, src, tgt, z1, z2):
        if anchor_mode != "nearest":
            if src not in centroids.get(t1_key, {}) or tgt not in centroids.get(t2_key, {}):
                return None
            return centroids[t1_key][src], centroids[t2_key][tgt]
        key = (t1_key, t2_key, src, tgt)
        if key in anchor_cache_cross:
            cached = anchor_cache_cross[key]
            if cached is None:
                return None
            p1, p2 = cached
            return {"x": float(p1[0]), "y": float(p1[1]), "z": z1}, {"x": float(p2[0]), "y": float(p2[1]), "z": z2}
        coords_map_1 = type_coords_by_time.get(t1_key, {})
        coords_map_2 = type_coords_by_time.get(t2_key, {})
        if src not in coords_map_1 or tgt not in coords_map_2:
            anchor_cache_cross[key] = None
            return None
        pair = _nearest_pair(coords_map_1[src], coords_map_2[tgt])
        if pair is None:
            anchor_cache_cross[key] = None
            return None
        p1, p2 = pair
        anchor_cache_cross[key] = (p1, p2)
        return {"x": float(p1[0]), "y": float(p1[1]), "z": z1}, {"x": float(p2[0]), "y": float(p2[1]), "z": z2}

    def _align_labels_for_flow(lbl_list: Sequence[Sequence[str]], anchor_idx: int = 0):
        arrs = [np.asarray(list(x)).astype(str) for x in lbl_list]
        if not arrs:
            return [], None
        min_len = min(len(a) for a in arrs)
        arrs = [a[:min_len] for a in arrs]
        if anchor_idx < 0 or anchor_idx >= len(arrs):
            raise ValueError("anchor_time_index out of range.")
        return arrs, arrs[anchor_idx]

    labels_flow: List[np.ndarray] = []
    anchor_labels: Optional[np.ndarray] = None
    if predicted_labels_list:
        labels_flow, anchor_labels = _align_labels_for_flow(predicted_labels_list, anchor_time_index)
    focus_labels = _normalize_focus_labels(ribbon_focus_celltype if ribbon_focus_celltype is not None else focus_celltype)

    for layer_idx, (tk, z) in enumerate(zip(time_keys, z_values)):
        ad = adata_dict[tk]
        ann_col = "Annotation" if "Annotation" in ad.obs.columns else "annotation"
        if spatial_key not in ad.obsm:
            continue
        coords = np.asarray(ad.obsm[spatial_key])
        labels = ad.obs[ann_col].values
        time_key = _canonical_time_value(tk)
        is_generated = _is_generated(time_key)
        layer_centroids = {}
        layer_type_coords = {}
        for ct in all_types:
            mask = labels == ct
            if np.any(mask):
                layer_type_coords[ct] = coords[mask]
                centroid = coords[mask].mean(axis=0)
                layer_centroids[ct] = {"x": float(centroid[0]), "y": float(centroid[1]), "z": z}
        centroids[tk] = layer_centroids
        type_coords_by_time[tk] = layer_type_coords

        sample_spec = _resolve_subsample_spec(is_generated)
        idx = _resolve_subsample_indices(coords.shape[0], sample_spec)
        if idx is not None:
            coords_plot = coords[idx]
            labels_plot = labels[idx]
        else:
            coords_plot = coords
            labels_plot = labels

        point_alpha_val = _resolve_point_alpha(is_generated)
        point_colors = [
            to_valid_color(label_to_color.get(l, "#888888"), default_alpha=point_alpha_val)
            for l in labels_plot
        ]
        line_width = _resolve_point_line_width(is_generated)
        line_color = _resolve_point_line_color(is_generated)
        marker_kwargs = dict(size=point_size, color=point_colors)
        if line_width and line_width > 0:
            if not line_color:
                line_color = "#333333"
            marker_kwargs["line"] = dict(
                width=float(line_width),
                color=to_valid_color(line_color, default_alpha=1.0),
            )

        if (show_slice_border or slice_fill_opacity > 0.0) and coords.size:
            x_min, y_min = coords[:, 0].min(), coords[:, 1].min()
            x_max, y_max = coords[:, 0].max(), coords[:, 1].max()
            span = max(x_max - x_min, y_max - y_min)
            pad = span * 0.02 if span > 0 else 1.0
            x_min -= pad
            x_max += pad
            y_min -= pad
            y_max += pad

            if slice_fill_opacity > 0.0:
                fill_color = _resolve_slice_fill_color(is_generated)
                if fill_color is not None:
                    fig.add_trace(
                        go.Mesh3d(
                            x=[x_min, x_max, x_max, x_min],
                            y=[y_min, y_min, y_max, y_max],
                            z=[z, z, z, z],
                            i=[0, 0],
                            j=[1, 2],
                            k=[2, 3],
                            color=to_valid_color(fill_color, default_alpha=1.0),
                            opacity=slice_fill_opacity,
                            flatshading=True,
                            showscale=False,
                            hoverinfo="skip",
                        )
                    )

        fig.add_trace(
            go.Scatter3d(
                x=coords_plot[:, 0],
                y=coords_plot[:, 1],
                z=[z] * len(coords_plot),
                mode="markers",
                marker=marker_kwargs,
                name=tk,
                showlegend=False,
                hoverinfo="skip",
            )
        )
        if show_slice_border and coords.size:
            dash_style = "dash" if is_generated else "solid"
            border_color = to_valid_color(_resolve_slice_border_color(is_generated), default_alpha=1.0)
            fig.add_trace(
                go.Scatter3d(
                    x=[x_min, x_max, x_max, x_min, x_min],
                    y=[y_min, y_min, y_max, y_max, y_min],
                    z=[z, z, z, z, z],
                    mode="lines",
                    line=dict(color=border_color, width=slice_border_width, dash=dash_style),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
        if show_centroid_nodes and layer_centroids:
            cent_x = []
            cent_y = []
            cent_z = []
            cent_c = []
            cent_t = []
            for ct, vals in layer_centroids.items():
                cent_x.append(vals["x"])
                cent_y.append(vals["y"])
                cent_z.append(vals["z"])
                cent_c.append(to_valid_color(label_to_color.get(ct, "#888888"), default_alpha=centroid_node_opacity))
                cent_t.append(f"{ct} ({tk})")
            fig.add_trace(
                go.Scatter3d(
                    x=cent_x,
                    y=cent_y,
                    z=cent_z,
                    mode="markers",
                    marker=dict(size=centroid_node_size, color=cent_c, line=dict(width=0.5, color="white")),
                    showlegend=False,
                    hoverinfo="text",
                    hovertext=cent_t,
                )
            )

    def _add_endpoint_node(tk, ct):
        if not highlight_endpoints:
            return
        if tk not in centroids:
            return
        if ct not in centroids[tk]:
            return
        key = (tk, ct)
        if key in endpoint_nodes:
            return
        vals = centroids[tk][ct]
        endpoint_nodes[key] = {
            "x": vals["x"],
            "y": vals["y"],
            "z": vals["z"],
            "color": to_valid_color(label_to_color.get(ct, "#888888"), default_alpha=endpoint_opacity),
            "label": f"{ct} ({tk})",
        }

    if not slices_only:
        edges_by_time = {}
        edges_all = []
        self_loops_by_time = {}
        for layer_idx, (tk, z_val) in enumerate(zip(time_keys, z_values)):
            if tk not in all_time_communications:
                continue
            ad = adata_dict[tk]
            ann_col = "Annotation" if "Annotation" in ad.obs.columns else "annotation"
            curr_coords = np.asarray(ad.obsm[spatial_key])
            curr_labels = ad.obs[ann_col].values
            comm_data = all_time_communications[tk]
            M_comm, types = comm_data["M_per_source"], comm_data["types"]
            type_to_idx = {t: i for i, t in enumerate(types)}
            layer_cents = centroids.get(tk, {})
            edges_this_tk = []
            self_loops_this_tk = []
            for type_from in types:
                for type_to in types:
                    if type_from == type_to:
                        if include_self_loops:
                            i = type_to_idx.get(type_from)
                            if i is None:
                                continue
                            if type_from not in layer_cents:
                                continue
                            weight = float(M_comm[i, i])
                            if weight > intra_threshold:
                                self_loops_this_tk.append({"weight": weight, "type_from": type_from})
                        continue
                    focus_edge = edge_focus_celltype if edge_focus_celltype is not None else focus_celltype
                    if focus_edge and type_from != focus_edge and type_to != focus_edge:
                        continue
                    i, j = type_to_idx.get(type_from), type_to_idx.get(type_to)
                    if i is None or j is None:
                        continue
                    weight = float(M_comm[i, j])
                    if weight <= intra_threshold:
                        continue
                    edges_this_tk.append({"weight": weight, "type_from": type_from, "type_to": type_to})

            if edges_this_tk:
                weights = np.array([e["weight"] for e in edges_this_tk], dtype=float)
                thresh = intra_threshold
                if edge_weight_quantile is not None and len(weights) > 1:
                    try:
                        quant_cut = np.quantile(weights, edge_weight_quantile)
                        thresh = max(thresh, float(quant_cut))
                    except Exception:
                        pass
                edges_this_tk = [e for e in edges_this_tk if e["weight"] >= thresh]
                edges_this_tk = sorted(edges_this_tk, key=lambda x: x["weight"], reverse=True)
                if edge_top_k_focus_label:
                    edges_this_tk = [
                        e
                        for e in edges_this_tk
                        if e["type_from"] == edge_top_k_focus_label or e["type_to"] == edge_top_k_focus_label
                    ]
                if edge_top_k is not None and edge_global_top_k is None:
                    edges_this_tk = edges_this_tk[: int(edge_top_k)]

            edges_by_time[tk] = edges_this_tk
            self_loops_by_time[tk] = self_loops_this_tk
            for e in edges_this_tk:
                edges_all.append({"tk": tk, "z": z_val, **e})

        if edge_global_top_k is not None and edges_all:
            edges_all = sorted(edges_all, key=lambda x: x["weight"], reverse=True)
            selected = edges_all[: int(edge_global_top_k)]
            edges_by_time = {tk: [] for tk in time_keys}
            for e in selected:
                edges_by_time[e["tk"]].append(e)

        for tk, z_val in zip(time_keys, z_values):
            if tk not in all_time_communications:
                continue
            layer_cents = centroids.get(tk, {})
            for e in edges_by_time.get(tk, []):
                w = e["weight"]
                t_f, t_t = e["type_from"], e["type_to"]
                anchor_pair = _get_anchor_same_layer(tk, t_f, t_t, z_val)
                if anchor_pair is None:
                    continue
                a1, a2 = anchor_pair
                p1 = [a1["x"], a1["y"], a1["z"]]
                p2 = [a2["x"], a2["y"], a2["z"]]
                _add_endpoint_node(tk, t_f)
                _add_endpoint_node(tk, t_t)
                if edge_render_mode == "line":
                    line_w = edge_line_width_base + np.log1p(w * 10) * edge_line_width_scale
                    line_w = max(0.5, float(line_w))
                    if edge_color is not None:
                        line_color = to_valid_color(edge_color, default_alpha=0.85)
                    else:
                        line_color = to_valid_color(label_to_color.get(t_f, "#cccccc"), default_alpha=0.85)
                    fig.add_trace(
                        go.Scatter3d(
                            x=[p1[0], p2[0]],
                            y=[p1[1], p2[1]],
                            z=[p1[2], p2[2]],
                            mode="lines",
                            line=dict(width=line_w, color=line_color),
                            showlegend=False,
                            hoverinfo="text",
                            hovertext=f"Comm: {t_f} -> {t_t}<br>{tk}<br>Val: {w:.3f}",
                        )
                    )
                else:
                    line_w = 0.8 + np.log1p(w * 10) * edge_width_scale
                    rad = min(0.04 * line_w, 0.35)
                    cyl = cylinder_between(p1, p2, radius=rad, n=20)
                    if cyl:
                        mesh_color = (
                            to_valid_color(edge_color, default_alpha=0.9)
                            if edge_color is not None
                            else to_valid_color(label_to_color.get(t_f, "#cccccc"), default_alpha=0.9)
                        )
                        fig.add_trace(
                            go.Mesh3d(
                                x=cyl[0],
                                y=cyl[1],
                                z=cyl[2],
                                i=cyl[3],
                                j=cyl[4],
                                k=cyl[5],
                                color=mesh_color,
                                opacity=1.0,
                                showscale=False,
                                hoverinfo="text",
                                hovertext=f"Comm: {t_f} -> {t_t}<br>{tk}<br>Val: {w:.3f}",
                            )
                        )

            if include_self_loops:
                for loop in self_loops_by_time.get(tk, []):
                    t_f = loop["type_from"]
                    w = loop["weight"]
                    if t_f not in layer_cents:
                        continue
                    center = layer_cents[t_f]
                    radius = self_loop_radius_scale * (0.6 + np.log1p(w))
                    theta = np.linspace(0, 2 * np.pi, self_loop_points)
                    x_loop = center["x"] + radius * np.cos(theta)
                    y_loop = center["y"] + radius * np.sin(theta)
                    z_loop = [center["z"]] * len(theta)
                    line_w = self_loop_line_width_base + np.log1p(w) * self_loop_line_width_scale
                    if edge_color is not None:
                        loop_color = to_valid_color(edge_color, default_alpha=0.9)
                    else:
                        loop_color = to_valid_color(label_to_color.get(t_f, "#cccccc"), default_alpha=0.9)
                    fig.add_trace(
                        go.Scatter3d(
                            x=x_loop,
                            y=y_loop,
                            z=z_loop,
                            mode="lines",
                            line=dict(width=line_w, color=loop_color),
                            showlegend=False,
                            hoverinfo="text",
                            hovertext=f"Comm: {t_f} -> {t_f}<br>{tk}<br>Val: {w:.3f}",
                        )
                    )

        if flow_normalize_mode not in (None, "source", "global"):
            raise ValueError("flow_normalize_mode must be None, 'source', or 'global'")

        for t_idx in range(len(time_keys) - 1):
            t1_key = time_keys[t_idx]
            t2_key = time_keys[t_idx + 1]
            labels_t = labels_flow[t_idx] if labels_flow else predicted_labels_list[t_idx]
            labels_next = labels_flow[t_idx + 1] if labels_flow else predicted_labels_list[t_idx + 1]
            min_len = min(len(labels_t), len(labels_next))
            if min_len == 0:
                continue
            if lineage_anchor_mode and anchor_labels is not None:
                anc = anchor_labels[:min_len]
                df_flow = pd.DataFrame({"ancestor": anc, "source": labels_t[:min_len], "target": labels_next[:min_len]})
                transitions = df_flow.groupby(["ancestor", "source", "target"]).size().reset_index(name="count")
                if flow_normalize_mode == "source":
                    transitions["value"] = transitions["count"] / transitions.groupby(["ancestor", "source"])["count"].transform("sum")
                elif flow_normalize_mode == "global":
                    total = transitions["count"].sum()
                    transitions["value"] = transitions["count"] / total if total > 0 else transitions["count"]
                else:
                    transitions["value"] = transitions["count"]
            else:
                df_flow = pd.DataFrame({"source": labels_t[:min_len], "target": labels_next[:min_len]})
                transitions = df_flow.groupby(["source", "target"]).size().reset_index(name="count")
                if flow_normalize_mode == "source":
                    transitions["value"] = transitions["count"] / transitions.groupby("source")["count"].transform("sum")
                elif flow_normalize_mode == "global":
                    total = transitions["count"].sum()
                    transitions["value"] = transitions["count"] / total if total > 0 else transitions["count"]
                else:
                    transitions["value"] = transitions["count"]
            if transitions.empty:
                continue
            if ribbon_min_count is not None:
                transitions = transitions[transitions["count"] >= ribbon_min_count]
                if transitions.empty:
                    continue
            if ribbon_keep_source_cumfrac is not None:
                transitions = _filter_transitions_keep_source_cumfrac(transitions, ribbon_keep_source_cumfrac)
                if transitions.empty:
                    continue
            if ribbon_count_quantile is not None and len(transitions) > 1:
                try:
                    ribbon_cut = float(transitions["count"].quantile(ribbon_count_quantile))
                except Exception:
                    ribbon_cut = None
            else:
                ribbon_cut = None
            if ribbon_top_k is not None:
                transitions = transitions.sort_values("count", ascending=False).head(int(ribbon_top_k))
            if ribbon_render_mode == "none":
                continue
            for _, row in transitions.iterrows():
                anc = row["ancestor"] if (lineage_anchor_mode and "ancestor" in row) else None
                src, tgt, count = row["source"], row["target"], row["count"]
                flow_val = row["value"]
                if ribbon_cut is not None and count < ribbon_cut:
                    continue
                if focus_labels:
                    if ribbon_focus_source_only:
                        if src not in focus_labels:
                            continue
                    elif ribbon_focus_target_only:
                        if tgt not in focus_labels:
                            continue
                    elif src not in focus_labels and tgt not in focus_labels:
                        continue
                anchor_pair = _get_anchor_cross_layer(t1_key, t2_key, src, tgt, z_values[t_idx], z_values[t_idx + 1])
                if anchor_pair is None:
                    continue
                c1, c2 = anchor_pair
                _add_endpoint_node(t1_key, src)
                _add_endpoint_node(t2_key, tgt)
                base_color_key = anc if anc is not None else src
                c_src = label_to_color.get(base_color_key, "#888")
                c_tgt = label_to_color.get(tgt, "#888")
                if ribbon_render_mode == "line":
                    line_w = ribbon_line_width_base + np.log1p(float(flow_val)) * ribbon_line_width_scale
                    line_w = max(0.5, float(line_w))
                    line_color = to_valid_color(c_src, default_alpha=ribbon_line_alpha)
                    if ribbon_line_curve and ribbon_line_curve > 0:
                        n_pts = max(2, int(ribbon_line_points))
                        t_vals = np.linspace(0.0, 1.0, n_pts)
                        x_path = c1["x"] + t_vals * (c2["x"] - c1["x"])
                        y_path = c1["y"] + t_vals * (c2["y"] - c1["y"])
                        z_path = c1["z"] + t_vals * (c2["z"] - c1["z"])
                        dx = c2["x"] - c1["x"]
                        dy = c2["y"] - c1["y"]
                        norm = np.hypot(dx, dy) + 1e-6
                        px, py = -dy / norm, dx / norm
                        bend = np.sin(np.pi * t_vals) * ribbon_line_curve * norm
                        x_path = x_path + px * bend
                        y_path = y_path + py * bend
                        fig.add_trace(
                            go.Scatter3d(
                                x=x_path,
                                y=y_path,
                                z=z_path,
                                mode="lines",
                                line=dict(width=line_w, color=line_color),
                                showlegend=False,
                                hoverinfo="text",
                                hovertext=(
                                    f"Ancestor: {anc if anc is not None else 'NA'}<br>"
                                    f"Fate: {src} -> {tgt}<br>Flow: {flow_val:.4f}"
                                ),
                            )
                        )
                    else:
                        fig.add_trace(
                            go.Scatter3d(
                                x=[c1["x"], c2["x"]],
                                y=[c1["y"], c2["y"]],
                                z=[c1["z"], c2["z"]],
                                mode="lines",
                                line=dict(width=line_w, color=line_color),
                                showlegend=False,
                                hoverinfo="text",
                                hovertext=f"Ancestor: {anc if anc is not None else 'NA'}<br>Fate: {src} -> {tgt}<br>Flow: {flow_val:.4f}",
                            )
                        )
                else:
                    ribbon = create_ribbon_surface(c1, c2, flow_val, c_src, c_tgt, n_points=ribbon_resolution)
                    fig.add_trace(
                        go.Mesh3d(
                            x=ribbon["x"],
                            y=ribbon["y"],
                            z=ribbon["z"],
                            i=ribbon["i"],
                            j=ribbon["j"],
                            k=ribbon["k"],
                            vertexcolor=ribbon["colors"],
                            opacity=0.85,
                            showlegend=False,
                            hoverinfo="text",
                            hovertext=f"Ancestor: {anc if anc is not None else 'NA'}<br>Fate: {src} -> {tgt}<br>Flow: {flow_val:.4f}",
                        )
                    )

    if highlight_endpoints and endpoint_nodes:
        pts = list(endpoint_nodes.values())
        fig.add_trace(
            go.Scatter3d(
                x=[p["x"] for p in pts],
                y=[p["y"] for p in pts],
                z=[p["z"] for p in pts],
                mode="markers",
                marker=dict(size=endpoint_size, color=[p["color"] for p in pts], line=dict(width=0.6, color="white")),
                showlegend=False,
                hoverinfo="text",
                hovertext=[p["label"] for p in pts],
            )
        )

    if show_legend:
        for ct in all_types:
            raw_c = label_to_color.get(ct, "#888888")
            clean_c = to_valid_color(raw_c, default_alpha=1.0)
            fig.add_trace(
                go.Scatter3d(
                    x=[None],
                    y=[None],
                    z=[None],
                    mode="markers",
                    marker=dict(size=10, color=clean_c),
                    name=ct,
                    showlegend=True,
                )
            )

    zaxis_cfg = dict(
        showbackground=False,
        showgrid=False,
        showline=False,
        tickvals=z_values if show_time_axis else [],
        ticktext=time_keys if show_time_axis else [],
        showticklabels=show_time_axis,
        title=dict(text="Time", font=dict(color=font_color)) if show_time_axis else dict(text=""),
        tickfont=dict(color=font_color),
    )
    fig.update_layout(
        scene_camera={"projection": {"type": "orthographic"}},
        paper_bgcolor=background_color,
        plot_bgcolor=background_color,
        font=dict(color=font_color),
        title=dict(text="3D Spatial Fate & Communication", x=0.5, y=0.9) if show_title else None,
        showlegend=show_legend,
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=zaxis_cfg,
            bgcolor=background_color,
            camera=dict(eye=dict(x=1.8, y=1.2, z=1.0)),
            aspectmode="manual",
            aspectratio=dict(x=1.5, y=1, z=1.2),
        ),
        width=width,
        height=height,
    )

    if out_html:
        fig.write_html(out_html)
    return fig


def compute_drift(df, dim: int, f_net, score_net, interaction_m: int = 1024, device: str = "cuda"):
    add_project_root()
    import numpy as np
    import torch
    from DeepRUOT.interaction import cal_interaction

    all_times = df["samples"].values
    all_data = df[[f"x{i}" for i in range(1, dim + 1)]].values

    t_tensor = torch.tensor(all_times, dtype=torch.float32).unsqueeze(1).to(device)
    data_tensor = torch.tensor(all_data, dtype=torch.float32).to(device)

    with torch.no_grad():
        gradients = f_net.v_net(t_tensor, data_tensor)
    gradients_np = gradients.detach().cpu().numpy()

    time_points = df["samples"].unique()
    all_gradients = []
    for time in time_points:
        subset = df[df["samples"] == time]
        data = torch.tensor(subset.iloc[:, 1 : dim + 1].values, dtype=torch.float32).to(device)
        lnw = torch.log(torch.ones(data.shape[0], 1, device=device) / data.shape[0])
        with torch.no_grad():
            gradients_i = cal_interaction(
                data,
                lnw,
                f_net.interaction_net,
                torch.tensor([time], dtype=torch.float32, device=device),
                m=interaction_m,
                threshold=1000,
            )
        all_gradients.append(gradients_i.detach().cpu().numpy())

    gradients_np_retain = np.concatenate(all_gradients, axis=0)

    data_tensor = torch.tensor(all_data, dtype=torch.float32, requires_grad=True, device=device)
    log_density_values = score_net(t_tensor, data_tensor)
    ones = torch.ones_like(log_density_values)
    gradients_score = torch.autograd.grad(
        outputs=log_density_values,
        inputs=data_tensor,
        grad_outputs=ones,
        retain_graph=False,
        create_graph=False,
        only_inputs=True,
    )[0]

    drift = gradients_np + gradients_np_retain + gradients_score.detach().cpu().numpy()
    return drift


def plot_velocity_stream(
    df,
    drift,
    dim: int,
    color_key: Optional[str] = None,
    basis: str = "umap",
    n_neighbors: int = 30,
    subsample_frac: float = 1.0,
    out_path: Optional[str] = None,
):
    import numpy as np
    import scanpy as sc
    import scvelo as scv
    import anndata as ad
    import matplotlib.pyplot as plt

    all_times = df["samples"].values
    all_data = df[[f"x{i}" for i in range(1, dim + 1)]].values

    adata = ad.AnnData(X=all_data[:, :2])
    adata.layers["Ms"] = all_data[:, :2]
    adata.layers["velocity"] = drift[:, :2]
    adata.obsm["X_umap"] = all_data[:, :2]
    adata.obs["time"] = all_times

    if color_key is not None and color_key in df.columns:
        adata.obs[color_key] = df[color_key].values

    rng = np.random.default_rng(0)
    if subsample_frac < 1.0:
        idx = rng.choice(adata.n_obs, size=int(adata.n_obs * subsample_frac), replace=False)
        adata = adata[idx].copy()

    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep="X")
    scv.tl.velocity_graph(adata, vkey="velocity")
    scv.tl.velocity_embedding(adata, basis=basis, vkey="velocity")
    scv.settings.set_figure_params("scvelo")

    scv.pl.velocity_embedding_stream(
        adata,
        basis=basis,
        color=color_key,
        figsize=(6, 6),
        density=3,
        title="Velocity Stream Plot",
        legend_loc="right",
        palette="tab20",
        show=False,
    )

    if out_path:
        plt.savefig(out_path, bbox_inches="tight", dpi=300)
    return adata


def prepare_annotations(
    h5ad_path: str,
    input_csv: str,
    output_csv: str,
    batch_key: str = "Batch",
    annotation_key: str = "Annotation",
    batch_indices: Optional[Sequence[int]] = None,
    label_color_json: Optional[str] = None,
):
    import json
    import numpy as np
    import pandas as pd
    import scanpy as sc

    adata = sc.read(h5ad_path)
    batch_series = adata.obs[batch_key]
    if hasattr(batch_series, "cat"):
        batch_names = list(batch_series.cat.categories)
    else:
        batch_names = sorted(batch_series.unique().tolist())

    if batch_indices is None:
        batch_indices = list(range(len(batch_names)))

    labels_list = []
    for idx in batch_indices:
        batch = batch_names[idx]
        adata_batch = adata[adata.obs[batch_key] == batch].copy()
        labels_list.append(adata_batch.obs[annotation_key].values)

    all_labels = np.concatenate(labels_list)
    df = pd.read_csv(input_csv)
    if len(all_labels) != len(df):
        raise ValueError(
            f"Label count ({len(all_labels)}) does not match CSV rows ({len(df)}). "
            "Check batch_indices or input CSV." 
        )
    df[annotation_key] = all_labels
    df.to_csv(output_csv, index=False)

    if label_color_json and annotation_key in adata.uns:
        colors = adata.uns.get(f"{annotation_key}_colors")
        if colors is not None:
            categories = (
                adata.obs[annotation_key].cat.categories
                if hasattr(adata.obs[annotation_key], "cat")
                else sorted(adata.obs[annotation_key].unique())
            )
            label_to_color = dict(zip(categories, colors))
            with open(label_color_json, "w", encoding="utf-8") as f:
                json.dump(label_to_color, f, indent=2)

    return output_csv


def prepare_annotations_by_sample_counts(
    h5ad_path: str,
    input_csv: str,
    output_csv: str,
    sample_column: str = "samples",
    batch_key: str = "Batch",
    annotation_key: str = "Annotation",
    batch_order: Optional[Sequence[str]] = None,
    seed: int = 0,
):
    """Create an annotation CSV by downsampling each batch to match per-sample counts."""
    import numpy as np
    import pandas as pd
    import scanpy as sc

    df = pd.read_csv(input_csv)
    sample_counts = df[sample_column].value_counts().sort_index()

    adata = sc.read(h5ad_path)
    if batch_order is None:
        if hasattr(adata.obs[batch_key], "cat"):
            batch_order = list(adata.obs[batch_key].cat.categories)
        else:
            batch_order = sorted(adata.obs[batch_key].unique().tolist())

    if len(batch_order) != len(sample_counts):
        raise ValueError(
            f"batch_order length ({len(batch_order)}) does not match sample count ({len(sample_counts)})."
        )

    rng = np.random.default_rng(seed)
    labels = []
    for batch_name, (_, target_n) in zip(batch_order, sample_counts.items()):
        adata_batch = adata[adata.obs[batch_key] == batch_name].copy()
        if target_n > adata_batch.n_obs:
            raise ValueError(
                f"Target count {target_n} exceeds available cells {adata_batch.n_obs} for batch {batch_name}."
            )
        idx = rng.choice(adata_batch.n_obs, size=target_n, replace=False)
        labels.append(adata_batch.obs[annotation_key].values[idx])

    all_labels = np.concatenate(labels)
    if len(all_labels) != len(df):
        raise ValueError("Generated label count does not match CSV rows.")

    df[annotation_key] = all_labels
    df.to_csv(output_csv, index=False)
    return output_csv


def train_mlp_classifier(
    df,
    feature_cols: Sequence[str],
    label_col: str,
    hidden_size: int = 128,
    epochs: int = 50,
    lr: float = 1e-3,
    test_size: float = 0.1,
    seed: int = 42,
    *,
    cache_path: Optional[str] = None,
    cache_dir: Optional[str] = None,
    cache_tag: Optional[str] = None,
    df_source_path: Optional[str] = None,
    reuse_if_possible: bool = True,
    progress: bool = True,
    device: Optional[str] = None,
    best_epoch_metric: str = "accuracy",
    train_on_full_data: bool = False,
    checkpoint_dir: Optional[str] = None,
    save_best_acc: bool = False,
    save_best_bacc: bool = False,
    save_last_k_epochs: int = 0,
):
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import copy
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import accuracy_score, balanced_accuracy_score

    metric_key = str(best_epoch_metric).strip().lower()
    if metric_key not in ("accuracy", "bacc"):
        raise ValueError("best_epoch_metric must be one of: 'accuracy', 'bacc'")
    if int(save_last_k_epochs) < 0:
        raise ValueError("save_last_k_epochs must be >= 0")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_t = torch.device(device)

    # -------------------------
    # Classifier definition
    # -------------------------

    class ResidualBlock(nn.Module):
        """Residual block with optional projection for dimension mismatch."""

        def __init__(self, in_dim: int, out_dim: int):
            super().__init__()
            self.block = nn.Sequential(nn.Linear(in_dim, out_dim), nn.LeakyReLU(0.2))
            self.skip = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

        def forward(self, x):
            return self.block(x) + self.skip(x)

    class ResidualMLP(nn.Module):
        """
        Residual MLP used in the original notebook port.

        Note: `hidden_size` is kept for API compatibility but the architecture is fixed
        (512 -> 512 -> 256 -> 128) to avoid behavior drift.
        """

        def __init__(self, input_size: int, hidden_size: int, num_classes: int):
            super().__init__()
            self.input_proj = nn.Sequential(nn.Linear(input_size, 512), nn.LeakyReLU(0.2))
            self.res1 = ResidualBlock(512, 512)
            self.res2 = ResidualBlock(512, 256)
            self.res3 = ResidualBlock(256, 128)
            self.fc_out = nn.Linear(128, num_classes)

        def forward(self, x):
            out = self.input_proj(x)
            out = self.res1(out)
            out = self.res2(out)
            out = self.res3(out)
            out = self.fc_out(out)
            return out

    @dataclass(frozen=True)
    class _ClassifierCacheMeta:
        version: int
        feature_cols: Tuple[str, ...]
        label_col: str
        hidden_size: int
        epochs: int
        lr: float
        test_size: float
        seed: int
        input_size: int
        classes: Tuple[str, ...]
        best_epoch_metric: str
        train_on_full_data: bool
        df_fingerprint: Dict[str, Union[str, int, float]]

        def to_dict(self) -> Dict:
            return {
                "version": int(self.version),
                "feature_cols": list(self.feature_cols),
                "label_col": str(self.label_col),
                "hidden_size": int(self.hidden_size),
                "epochs": int(self.epochs),
                "lr": float(self.lr),
                "test_size": float(self.test_size),
                "seed": int(self.seed),
                "input_size": int(self.input_size),
                "classes": list(self.classes),
                "best_epoch_metric": str(self.best_epoch_metric),
                "train_on_full_data": bool(self.train_on_full_data),
                "df_fingerprint": dict(self.df_fingerprint),
            }

        @staticmethod
        def from_dict(d: Dict) -> "_ClassifierCacheMeta":
            return _ClassifierCacheMeta(
                version=int(d.get("version", 0)),
                feature_cols=tuple(d.get("feature_cols", [])),
                label_col=str(d.get("label_col", "")),
                hidden_size=int(d.get("hidden_size", 0)),
                epochs=int(d.get("epochs", 0)),
                lr=float(d.get("lr", 0.0)),
                test_size=float(d.get("test_size", 0.0)),
                seed=int(d.get("seed", 0)),
                input_size=int(d.get("input_size", 0)),
                classes=tuple(d.get("classes", [])),
                best_epoch_metric=str(d.get("best_epoch_metric", "accuracy")),
                train_on_full_data=bool(d.get("train_on_full_data", False)),
                df_fingerprint=dict(d.get("df_fingerprint", {})),
            )

    def _df_fingerprint(df_source_path_in: Optional[str], df_obj) -> Dict[str, Union[str, int, float]]:
        fp: Dict[str, Union[str, int, float]] = {
            "n_rows": int(len(df_obj)),
        }
        if df_source_path_in:
            try:
                st = os.stat(df_source_path_in)
                fp.update(
                    {
                        "path": os.path.abspath(df_source_path_in),
                        "size": int(st.st_size),
                        "mtime": float(st.st_mtime),
                    }
                )
            except Exception:
                fp.update({"path": os.path.abspath(df_source_path_in)})
        return fp

    def _cache_key(meta_dict: Dict, tag: Optional[str]) -> str:
        payload = dict(meta_dict)
        payload["tag"] = str(tag) if tag else ""
        blob = (json_dumps_sorted(payload)).encode("utf-8")
        return sha1(blob).hexdigest()[:16]

    def json_dumps_sorted(obj: Dict) -> str:
        import json

        return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    def _resolve_cache_path(meta_dict: Dict) -> Optional[str]:
        if cache_path:
            return cache_path
        if not cache_dir:
            return None
        os.makedirs(cache_dir, exist_ok=True)
        key = _cache_key(meta_dict, cache_tag)
        fname = f"classifier_resmlp_{key}.pt"
        return os.path.join(cache_dir, fname)

    def _try_load_cached(meta_expected: _ClassifierCacheMeta, cache_path_in: str):
        if not (reuse_if_possible and cache_path_in and os.path.exists(cache_path_in)):
            return None
        try:
            payload = torch.load(cache_path_in, map_location="cpu")
            meta_loaded = _ClassifierCacheMeta.from_dict(payload.get("meta", {}))
            if meta_loaded.to_dict() != meta_expected.to_dict():
                return None

            model_loaded = ResidualMLP(
                input_size=meta_loaded.input_size,
                hidden_size=meta_loaded.hidden_size,
                num_classes=len(meta_loaded.classes),
            )
            model_loaded.load_state_dict(payload["state_dict"])
            model_loaded.eval()

            label_encoder_loaded = LabelEncoder()
            label_encoder_loaded.classes_ = np.asarray(list(meta_loaded.classes))
            acc_loaded = payload.get("acc", None)
            if progress:
                try:
                    print(f"[train_mlp_classifier] loaded cache: {cache_path_in}")
                except Exception:
                    pass
            return model_loaded, label_encoder_loaded, acc_loaded
        except Exception:
            return None

    X = df[feature_cols].values
    y = df[label_col].values

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Cache meta (independent of train/test split)
    df_fp = _df_fingerprint(df_source_path, df)
    meta = _ClassifierCacheMeta(
        version=1,
        feature_cols=tuple(str(c) for c in feature_cols),
        label_col=str(label_col),
        hidden_size=int(hidden_size),
        epochs=int(epochs),
        lr=float(lr),
        test_size=float(test_size),
        seed=int(seed),
        input_size=int(len(feature_cols)),
        classes=tuple(str(c) for c in label_encoder.classes_),
        best_epoch_metric=metric_key,
        train_on_full_data=bool(train_on_full_data),
        df_fingerprint=df_fp,
    )
    resolved_cache_path = _resolve_cache_path(meta.to_dict())
    cached = _try_load_cached(meta, resolved_cache_path) if resolved_cache_path else None
    if cached is not None:
        model_cached, label_encoder_cached, acc_cached = cached
        return model_cached.to(device_t), label_encoder_cached, acc_cached if acc_cached is not None else float("nan")

    if train_on_full_data:
        X_train_np, y_train_np = X, y_encoded
        X_eval_np, y_eval_np = X, y_encoded
        eval_split_name = "train_full"
    else:
        X_train_np, X_eval_np, y_train_np, y_eval_np = train_test_split(
            X, y_encoded, test_size=test_size, random_state=seed
        )
        eval_split_name = "val"

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    X_train = torch.tensor(X_train_np, dtype=torch.float32, device=device_t)
    X_eval = torch.tensor(X_eval_np, dtype=torch.float32, device=device_t)
    y_train = torch.tensor(y_train_np, dtype=torch.long, device=device_t)
    y_eval = torch.tensor(y_eval_np, dtype=torch.long, device=device_t)

    if checkpoint_dir is not None and (
        bool(save_best_acc) or bool(save_best_bacc) or int(save_last_k_epochs) > 0
    ):
        os.makedirs(checkpoint_dir, exist_ok=True)

    def _save_epoch_ckpt(
        path: str,
        *,
        epoch_1based: int,
        loss_value: float,
        train_acc_value: float,
        train_bacc_value: float,
        eval_acc_value: float,
        eval_bacc_value: float,
    ) -> None:
        payload = {
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "epoch": int(epoch_1based),
            "metrics": {
                "loss": float(loss_value),
                "train_acc": float(train_acc_value),
                "train_bacc": float(train_bacc_value),
                f"{eval_split_name}_acc": float(eval_acc_value),
                f"{eval_split_name}_bacc": float(eval_bacc_value),
            },
            "meta": {
                "feature_cols": [str(c) for c in feature_cols],
                "label_col": str(label_col),
                "classes": [str(c) for c in label_encoder.classes_],
                "best_epoch_metric": str(metric_key),
                "train_on_full_data": bool(train_on_full_data),
                "saved_at": time.time(),
            },
        }
        torch.save(payload, path)

    model = ResidualMLP(
        input_size=X_train.shape[1],
        hidden_size=hidden_size,
        num_classes=len(label_encoder.classes_)
    )
    model.to(device_t)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_score = float("-inf")
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = float("-inf")
    best_bacc = float("-inf")
    best_acc_epoch = -1
    best_bacc_epoch = -1

    t0 = time.perf_counter()
    if progress:
        try:
            from tqdm.auto import trange

            epoch_iter = trange(epochs, desc="train_mlp_classifier", unit="epoch")
        except Exception:
            epoch_iter = range(epochs)
    else:
        epoch_iter = range(epochs)

    for epoch in epoch_iter:
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            train_outputs = model(X_train)
            _, train_preds = torch.max(train_outputs, 1)
            y_train_np_cpu = y_train.detach().cpu().numpy()
            train_preds_np = train_preds.detach().cpu().numpy()
            train_acc = accuracy_score(y_train_np_cpu, train_preds_np)
            train_bacc = balanced_accuracy_score(y_train_np_cpu, train_preds_np)

            eval_outputs = model(X_eval)
            _, eval_preds = torch.max(eval_outputs, 1)
            y_eval_np_cpu = y_eval.detach().cpu().numpy()
            eval_preds_np = eval_preds.detach().cpu().numpy()
            eval_acc = accuracy_score(y_eval_np_cpu, eval_preds_np)
            eval_bacc = balanced_accuracy_score(y_eval_np_cpu, eval_preds_np)
            monitor_acc = train_acc if train_on_full_data else eval_acc
            monitor_bacc = train_bacc if train_on_full_data else eval_bacc
            monitor_score = monitor_acc if metric_key == "accuracy" else monitor_bacc

            if monitor_score > best_score:
                best_score = monitor_score
                best_model_wts = copy.deepcopy(model.state_dict())

            epoch_1based = int(epoch) + 1
            if monitor_acc > best_acc:
                best_acc = monitor_acc
                best_acc_epoch = epoch_1based
                if checkpoint_dir and save_best_acc:
                    _save_epoch_ckpt(
                        os.path.join(checkpoint_dir, "best_acc.pt"),
                        epoch_1based=epoch_1based,
                        loss_value=float(loss.detach().cpu().item()),
                        train_acc_value=train_acc,
                        train_bacc_value=train_bacc,
                        eval_acc_value=eval_acc,
                        eval_bacc_value=eval_bacc,
                    )
            if monitor_bacc > best_bacc:
                best_bacc = monitor_bacc
                best_bacc_epoch = epoch_1based
                if checkpoint_dir and save_best_bacc:
                    _save_epoch_ckpt(
                        os.path.join(checkpoint_dir, "best_bacc.pt"),
                        epoch_1based=epoch_1based,
                        loss_value=float(loss.detach().cpu().item()),
                        train_acc_value=train_acc,
                        train_bacc_value=train_bacc,
                        eval_acc_value=eval_acc,
                        eval_bacc_value=eval_bacc,
                    )

            if checkpoint_dir and int(save_last_k_epochs) > 0 and epoch_1based > (epochs - int(save_last_k_epochs)):
                _save_epoch_ckpt(
                    os.path.join(checkpoint_dir, f"epoch_{epoch_1based:04d}.pt"),
                    epoch_1based=epoch_1based,
                    loss_value=float(loss.detach().cpu().item()),
                    train_acc_value=train_acc,
                    train_bacc_value=train_bacc,
                    eval_acc_value=eval_acc,
                    eval_bacc_value=eval_bacc,
                )
        if progress and hasattr(epoch_iter, "set_postfix"):
            try:
                if train_on_full_data:
                    epoch_iter.set_postfix(
                        loss=float(loss.detach().cpu().item()),
                        train_acc=float(train_acc),
                        train_bacc=float(train_bacc),
                        monitor=metric_key,
                        train_monitor=float(monitor_score),
                    )
                else:
                    epoch_iter.set_postfix(
                        loss=float(loss.detach().cpu().item()),
                        train_acc=float(train_acc),
                        train_bacc=float(train_bacc),
                        eval_acc=float(eval_acc),
                        eval_bacc=float(eval_bacc),
                        monitor=metric_key,
                        eval_monitor=float(monitor_score),
                    )
            except Exception:
                pass

    model.load_state_dict(best_model_wts)

    model.eval()
    with torch.no_grad():
        train_preds_final = torch.argmax(model(X_train), dim=1)
        eval_preds_final = torch.argmax(model(X_eval), dim=1)
    train_acc_final = accuracy_score(y_train.detach().cpu().numpy(), train_preds_final.detach().cpu().numpy())
    train_bacc_final = balanced_accuracy_score(y_train.detach().cpu().numpy(), train_preds_final.detach().cpu().numpy())
    eval_acc_final = accuracy_score(y_eval.detach().cpu().numpy(), eval_preds_final.detach().cpu().numpy())
    eval_bacc_final = balanced_accuracy_score(y_eval.detach().cpu().numpy(), eval_preds_final.detach().cpu().numpy())
    acc = float(train_acc_final if train_on_full_data else eval_acc_final)
    bacc = float(train_bacc_final if train_on_full_data else eval_bacc_final)

    if progress:
        elapsed = time.perf_counter() - t0
        if train_on_full_data:
            print(
                "[train_mlp_classifier] "
                f"done in {elapsed:.1f}s | "
                f"train_acc={train_acc_final:.4f}, train_bacc={train_bacc_final:.4f} | "
                f"select_by={metric_key}"
            )
        else:
            print(
                "[train_mlp_classifier] "
                f"done in {elapsed:.1f}s | "
                f"train_acc={train_acc_final:.4f}, train_bacc={train_bacc_final:.4f} | "
                f"{eval_split_name}_acc={eval_acc_final:.4f}, {eval_split_name}_bacc={eval_bacc_final:.4f} | "
                f"select_by={metric_key}"
            )
        if best_acc_epoch > 0 or best_bacc_epoch > 0:
            print(
                "[train_mlp_classifier] "
                f"best_acc_epoch={best_acc_epoch}, best_bacc_epoch={best_bacc_epoch}"
            )
        if checkpoint_dir and (save_best_acc or save_best_bacc or int(save_last_k_epochs) > 0):
            print(f"[train_mlp_classifier] checkpoints: {checkpoint_dir}")

    # Save cache
    if resolved_cache_path and (cache_dir or cache_path):
        try:
            payload = {
                "meta": meta.to_dict(),
                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "acc": float(acc),
                "bacc": float(bacc),
                "saved_at": time.time(),
            }
            torch.save(payload, resolved_cache_path)
            if progress:
                try:
                    print(f"[train_mlp_classifier] saved cache: {resolved_cache_path}")
                except Exception:
                    pass
        except Exception:
            pass

    return model, label_encoder, acc


def predict_labels_for_trajectories(
    sde_points,
    ts_points: Sequence[float],
    model,
    label_encoder,
    feature_dim: int,
    device: str = "cuda",
    knn_neighbors: int = 10,
):
    import numpy as np
    import torch
    from sklearn.neighbors import KNeighborsClassifier

    model.eval()
    model.to(device)

    predicted_labels_list = []
    for i, t in enumerate(ts_points):
        traj_t = np.array(sde_points[i], dtype=float)
        traj_t_tensor = torch.tensor(traj_t, dtype=torch.float32)
        n_samples = traj_t_tensor.shape[0]

        samples_t = torch.full((n_samples, 1), fill_value=float(t))
        input_t = torch.cat((samples_t, traj_t_tensor[:, :feature_dim]), dim=1)

        with torch.no_grad():
            outputs = model(input_t.float().to(device))
            _, predicted = torch.max(outputs, 1)
            predicted_labels = label_encoder.inverse_transform(predicted.detach().cpu().numpy())

        coords = input_t[:, 1:3].cpu().numpy()
        k = min(int(knn_neighbors), int(coords.shape[0]))
        if k <= 1:
            refined_labels = predicted_labels
        else:
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(coords, predicted_labels)
            refined_labels = knn.predict(coords)

        predicted_labels_list.append(refined_labels)

    return predicted_labels_list


def plot_sankey(
    predicted_labels_list: Sequence[Sequence[str]],
    out_html: Optional[str] = None,
    start_index: int = 0,
    focus_source_label: Optional[str] = None,
    focus_target_label: Optional[str] = None,
    include_labels: Optional[Sequence[str]] = None,
    time_keys: Optional[Sequence[str]] = None,
    show_time_axis: bool = False,
    time_axis_y: float = -0.08,
    normalize_mode: Optional[str] = None,
    min_flow: Optional[float] = None,
    keep_source_cumfrac: Optional[float] = None,
    label_to_color: Optional[Dict[str, str]] = None,
    lineage_anchor_mode: bool = False,
    anchor_time_index: int = 0,
    style: str = "default",
    title: Optional[str] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
):
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go

    if start_index < 0:
        raise ValueError("start_index must be >= 0")

    labels_list = [list(x) for x in predicted_labels_list]
    time_keys_list = list(time_keys) if time_keys is not None else None
    if time_keys_list is not None:
        min_len = min(len(time_keys_list), len(labels_list))
        time_keys_list = time_keys_list[:min_len]
        labels_list = labels_list[:min_len]

    if normalize_mode not in (None, "source", "global"):
        raise ValueError("normalize_mode must be None, 'source', or 'global'")
    if min_flow is not None and min_flow < 0:
        raise ValueError("min_flow must be >= 0")
    if keep_source_cumfrac is not None:
        keep_source_cumfrac = float(keep_source_cumfrac)
        if not (0.0 < keep_source_cumfrac <= 1.0):
            raise ValueError("keep_source_cumfrac must be in (0, 1].")
    if style not in ("default", "nature-methods"):
        raise ValueError("style must be 'default' or 'nature-methods'")

    def _align_for_anchor(lbl_list: Sequence[Sequence[str]], anchor_idx: int):
        arrs = [pd.Series(list(x)).astype(str).values for x in lbl_list]
        if not arrs:
            return [], None
        min_len = min(len(a) for a in arrs)
        arrs = [a[:min_len] for a in arrs]
        if anchor_idx < 0 or anchor_idx >= len(arrs):
            raise ValueError("anchor_time_index out of range.")
        return arrs, arrs[anchor_idx]

    links = []
    num_timepoints = len(labels_list)
    if start_index >= num_timepoints - 1:
        raise ValueError("start_index must be <= len(timepoints) - 2")

    anchor_labels = None
    labels_aligned = labels_list
    if lineage_anchor_mode:
        labels_aligned, anchor_labels = _align_for_anchor(labels_list, anchor_time_index)
        if anchor_labels is None or len(anchor_labels) == 0:
            raise ValueError("No labels available for ancestor-coupled Sankey.")
    else:
        labels_aligned = [list(x) for x in labels_list]

    def _filter_keep_source_cumfrac(df: "pd.DataFrame", cumfrac: float, group_cols: Sequence[str]) -> "pd.DataFrame":
        if df.empty:
            return df
        keep_indices = []
        for _, g in df.groupby(list(group_cols), sort=False):
            g = g.sort_values("value", ascending=False)
            total = float(g["value"].sum())
            if total <= 0:
                continue
            cum = (g["value"].cumsum() / total).to_numpy()
            # Keep the minimal prefix whose cumulative fraction reaches/exceeds cumfrac.
            # Using searchsorted avoids an off-by-one when cum contains an exact threshold hit.
            keep_n = int(np.searchsorted(cum, cumfrac, side="left")) + 1
            keep_n = max(1, min(len(g), keep_n))
            keep_indices.extend(g.head(keep_n).index.tolist())
        if not keep_indices:
            return df.iloc[0:0]
        return df.loc[keep_indices]

    if focus_source_label is not None:
        min_len = min(len(labels_aligned[t]) for t in range(start_index, num_timepoints))
        if min_len == 0:
            raise ValueError("No labels available for focus_source_label filtering.")
        base = pd.Series(labels_aligned[start_index][:min_len])
        mask = base == focus_source_label
        if not mask.any():
            raise ValueError(f"No entries found for focus_source_label='{focus_source_label}'.")
        for t in range(start_index, num_timepoints):
            labels_t = pd.Series(labels_aligned[t][:min_len])
            labels_aligned[t] = labels_t[mask].tolist()
        if lineage_anchor_mode and anchor_labels is not None:
            anchor_labels = anchor_labels[:min_len][mask.values]

    for t in range(start_index, num_timepoints - 1):
        src = list(labels_aligned[t])
        tgt = list(labels_aligned[t + 1])
        min_len = min(len(src), len(tgt))
        if min_len == 0:
            continue
        if lineage_anchor_mode:
            anc = anchor_labels[:min_len] if anchor_labels is not None else ["anc"] * min_len
            df = pd.DataFrame({"ancestor": anc, "source": src[:min_len], "target": tgt[:min_len]})
            if focus_target_label is not None:
                df = df[df["target"] == focus_target_label]
            if include_labels:
                df = df[df["source"].isin(include_labels) | df["target"].isin(include_labels)]
            if df.empty:
                continue
            group_cols = ["ancestor", "source", "target"]
            counts = df.groupby(group_cols).size().reset_index(name="value")
            if normalize_mode == "source":
                counts["value"] = counts["value"] / counts.groupby(["ancestor", "source"])["value"].transform("sum")
            elif normalize_mode == "global":
                total = counts["value"].sum()
                if total > 0:
                    counts["value"] = counts["value"] / total
            if keep_source_cumfrac is not None:
                counts = _filter_keep_source_cumfrac(counts, keep_source_cumfrac, group_cols=["ancestor", "source"])
                if counts.empty:
                    continue
            if min_flow is not None:
                counts = counts[counts["value"] >= min_flow]
                if counts.empty:
                    continue
            counts["source"] = counts.apply(
                lambda r: f"{r['ancestor']}->{r['source']}__T{t + 1}", axis=1
            )
            counts["target"] = counts.apply(
                lambda r: f"{r['ancestor']}->{r['target']}__T{t + 2}", axis=1
            )
        else:
            df = pd.DataFrame({"source": src[:min_len], "target": tgt[:min_len]})
            if focus_target_label is not None:
                df = df[df["target"] == focus_target_label]
            if include_labels:
                df = df[df["source"].isin(include_labels) | df["target"].isin(include_labels)]
            if df.empty:
                continue
            counts = df.groupby(["source", "target"]).size().reset_index(name="value")
            if normalize_mode == "source":
                counts["value"] = counts["value"] / counts.groupby("source")["value"].transform("sum")
            elif normalize_mode == "global":
                total = counts["value"].sum()
                if total > 0:
                    counts["value"] = counts["value"] / total
            if keep_source_cumfrac is not None:
                counts = _filter_keep_source_cumfrac(counts, keep_source_cumfrac, group_cols=["source"])
                if counts.empty:
                    continue
            if min_flow is not None:
                counts = counts[counts["value"] >= min_flow]
                if counts.empty:
                    continue
            counts["source"] = counts["source"].astype(str) + f"__T{t + 1}"
            counts["target"] = counts["target"].astype(str) + f"__T{t + 2}"
        links.append(counts)

    if not links:
        raise ValueError("No valid transitions to plot: all timepoint label arrays are empty.")

    all_links_df = pd.concat(links, axis=0)
    all_nodes = pd.unique(all_links_df[["source", "target"]].values.ravel("K"))
    node_indices = {node: i for i, node in enumerate(all_nodes)}

    all_links_df["source_idx"] = all_links_df["source"].map(node_indices)
    all_links_df["target_idx"] = all_links_df["target"].map(node_indices)

    def _split_node(node_id: str) -> str:
        return node_id.rsplit("__T", 1)[0] if "__T" in node_id else node_id

    def _parse_node(node_id: str):
        core = _split_node(node_id)
        if "->" in core:
            anc, curr = core.split("->", 1)
        else:
            anc, curr = core, core
        return anc, curr

    base_types = sorted({_split_node(node) for node in all_nodes})
    if lineage_anchor_mode:
        base_types = sorted({_parse_node(node)[0] for node in all_nodes})
    color_palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    color_map = {base_type: color_palette[i % len(color_palette)] for i, base_type in enumerate(base_types)}

    def _base_color(node_label: str) -> str:
        anc, _ = _parse_node(node_label)
        base = anc if lineage_anchor_mode else _split_node(node_label)
        if label_to_color and base in label_to_color:
            return label_to_color[base]
        return color_map.get(base, "#888888")

    def _sanitize_color(color: str, alpha: float) -> str:
        if not isinstance(color, str):
            return f"rgba(136,136,136,{alpha})"
        c = color.strip()
        if c.startswith("#") and len(c) == 9:
            r = int(c[1:3], 16)
            g = int(c[3:5], 16)
            b = int(c[5:7], 16)
            a = int(c[7:9], 16) / 255.0
            if alpha < 1.0:
                a = alpha
            return f"rgba({r},{g},{b},{a})"
        if c.startswith("#") and len(c) == 7:
            if alpha >= 1.0:
                return c
            r = int(c[1:3], 16)
            g = int(c[3:5], 16)
            b = int(c[5:7], 16)
            return f"rgba({r},{g},{b},{alpha})"
        if c.startswith("rgb"):
            import re

            nums = [float(x) for x in re.findall(r"[0-9\\.]+", c)]
            if len(nums) >= 3:
                return f"rgba({int(nums[0])},{int(nums[1])},{int(nums[2])},{alpha})"
        return c

    node_labels = []
    for node in all_nodes:
        if lineage_anchor_mode:
            anc, curr = _parse_node(node)
            node_labels.append(f"{curr} | anc {anc}")
        else:
            node_labels.append(_split_node(node))
    node_colors = [_sanitize_color(_base_color(node), alpha=1.0) for node in all_nodes]

    # Spread y positions within each time slice so later layers don't collapse.
    time_for_node: Dict[str, int] = {}
    for node in all_nodes:
        if "__T" in node:
            time_for_node[node] = int(node.rsplit("__T", 1)[1])
        else:
            time_for_node[node] = start_index + 1
    time_groups: Dict[int, List[str]] = {}
    for node, t_idx in time_for_node.items():
        time_groups.setdefault(t_idx, []).append(node)
    node_y = []
    for node in all_nodes:
        t_idx = time_for_node[node]
        group = time_groups.get(t_idx, [node])
        pos = group.index(node)
        if len(group) == 1:
            node_y.append(0.5)
        else:
            node_y.append(pos / (len(group) - 1))

    link_colors = []
    link_alpha = 0.4 if style == "nature-methods" else 0.5
    for src_idx in all_links_df["source_idx"]:
        source_node_label = all_nodes[src_idx]
        base_color = _base_color(source_node_label)
        link_colors.append(_sanitize_color(base_color, alpha=link_alpha))

    displayed_steps = num_timepoints - start_index
    if displayed_steps < 2:
        displayed_steps = 2

    def _node_x(node_id: str) -> float:
        if "__T" in node_id:
            time_idx = int(node_id.rsplit("__T", 1)[1]) - (start_index + 1)
        else:
            time_idx = 0
        if style == "nature-methods":
            return 0.05 + (time_idx / (displayed_steps - 1)) * 0.9
        return time_idx / (displayed_steps - 1)

    fig = go.Figure(
        data=[
            go.Sankey(
                arrangement="snap" if style == "nature-methods" else "fixed",
                node=dict(
                    pad=15 if style == "nature-methods" else 20,
                    thickness=20 if style == "nature-methods" else 25,
                    line=dict(color="black", width=0.5),
                    label=node_labels,
                    color=node_colors,
                    x=[_node_x(node) for node in all_nodes],
                    **({} if style == "nature-methods" else {"y": node_y}),
                    hovertemplate="Type: %{label}<br>Count: %{value}<extra></extra>",
                ),
                link=dict(
                    source=all_links_df["source_idx"],
                    target=all_links_df["target_idx"],
                    value=all_links_df["value"],
                    color=link_colors,
                    hovertemplate="%{source.label} → %{target.label}<br>%{value}<extra></extra>",
                ),
            )
        ]
    )

    if title is None:
        title = "Cell Fate Transitions" if style == "nature-methods" else "Cell Lineage Sankey"

    layout_kwargs = dict(
        font_family="Arial",
        font_size=12,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    if style == "nature-methods":
        layout_kwargs["title"] = dict(text=title, x=0.5, xanchor="center")
        layout_kwargs["width"] = 1600 if width is None else int(width)
        layout_kwargs["height"] = 1000 if height is None else int(height)
    else:
        layout_kwargs["title_text"] = title
        if width is not None:
            layout_kwargs["width"] = int(width)
        if height is not None:
            layout_kwargs["height"] = int(height)
    fig.update_layout(**layout_kwargs)

    if show_time_axis:
        if time_keys_list is None:
            time_keys_list = [f"T{i + 1}" for i in range(num_timepoints)]
        shown_keys = time_keys_list[start_index:]
        if len(shown_keys) >= 1:
            if style == "nature-methods":
                for idx, tk in enumerate(shown_keys):
                    x_pos = idx / max(1, len(shown_keys) - 1)
                    fig.add_annotation(
                        x=x_pos,
                        y=-0.10 if time_axis_y is None else float(time_axis_y),
                        xref="paper",
                        yref="paper",
                        text=str(tk),
                        showarrow=False,
                        font=dict(size=14, color="black"),
                    )
                fig.update_layout(margin=dict(b=120))
            else:
                for idx, tk in enumerate(shown_keys):
                    x_pos = idx / max(1, len(shown_keys) - 1)
                    fig.add_annotation(
                        x=x_pos,
                        y=time_axis_y,
                        xref="paper",
                        yref="paper",
                        text=str(tk),
                        showarrow=False,
                        font=dict(size=12, color="black"),
                    )
                fig.add_shape(
                    type="line",
                    x0=0,
                    x1=1,
                    y0=time_axis_y + 0.02,
                    y1=time_axis_y + 0.02,
                    xref="paper",
                    yref="paper",
                    line=dict(color="black", width=1),
                )
                fig.update_layout(margin=dict(b=80))

    if out_html:
        fig.write_html(out_html)
    return fig
