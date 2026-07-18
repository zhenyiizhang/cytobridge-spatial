"""SDE simulation and velocity decomposition for downstream analysis.

Downstream simulation APIs are AnnData-first and do not require a dataframe
intermediate.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn

if TYPE_CHECKING:
    import anndata as ad

__all__ = [
    "simulate_sde_points",
    "simulate_sde_points_split",
    "sample_observed_x0",
    "simulate_sde_points_split_from_x0",
    "apply_spatial_warp_to_segments",
    "simulate_piecewise_spatially_warped_split",
    "compute_velocity_components",
    "compute_velocity_components_from_adata",
    "compute_drift",
    "compute_drift_from_adata",
    "compute_umap_embedding",
]


def _sorted_unique(values) -> list[float]:
    uniq = list(dict.fromkeys(list(values)))
    try:
        return sorted(float(x) for x in uniq)
    except Exception:
        return [float(x) for x in uniq]


def _coerce_feature_matrix_from_adata(
    adata,
    *,
    obsm_key: str,
    spatial_key: str,
    concat_spatial: Optional[bool],
) -> tuple[np.ndarray, int]:
    if obsm_key in adata.obsm:
        latent = np.asarray(adata.obsm[obsm_key], dtype=np.float32)
    else:
        latent = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
        latent = np.asarray(latent, dtype=np.float32)

    use_spatial = bool(concat_spatial) if concat_spatial is not None else (spatial_key in adata.obsm)
    if not use_spatial:
        return latent, 0

    if spatial_key not in adata.obsm:
        raise KeyError(f"concat_spatial=True but adata.obsm['{spatial_key}'] is missing.")
    spatial = np.asarray(adata.obsm[spatial_key], dtype=np.float32)
    if spatial.shape[0] != latent.shape[0]:
        raise ValueError(
            f"Row mismatch between '{spatial_key}' ({spatial.shape[0]}) and "
            f"'{obsm_key}' ({latent.shape[0]})."
        )
    return np.hstack((spatial, latent)).astype(np.float32), int(spatial.shape[1])


def _prepare_adata_arrays(
    adata,
    dim: Optional[int],
    *,
    time_key: Optional[str],
    obsm_key: str,
    spatial_key: str,
    concat_spatial: Optional[bool],
) -> tuple[np.ndarray, np.ndarray, list[float], int]:
    if not (hasattr(adata, "obs") and hasattr(adata, "obsm")):
        raise TypeError(
            "Downstream simulation APIs require AnnData input. "
            f"Got: {type(adata)}"
        )

    from CytoBridge.tl.downstream.downstream_data import infer_time_key, parse_time_value

    resolved_time_key = infer_time_key(adata.obs, preferred=time_key)
    raw_times = adata.obs[resolved_time_key].values
    times = np.asarray([parse_time_value(v) for v in raw_times], dtype=np.float64)

    X, _ = _coerce_feature_matrix_from_adata(
        adata,
        obsm_key=obsm_key,
        spatial_key=spatial_key,
        concat_spatial=concat_spatial,
    )
    if dim is None:
        dim = int(X.shape[1])
    else:
        dim = int(dim)
        if dim > X.shape[1]:
            raise ValueError(f"Requested dim={dim}, but feature matrix has dim={X.shape[1]}.")
    X = X[:, :dim]
    unique_times = _sorted_unique(times)
    return X, times, unique_times, dim


def _time_mask(times: np.ndarray, t: float) -> np.ndarray:
    return np.isclose(times, float(t), rtol=0.0, atol=1e-9)


def _build_perturbation_keep_mask(
    adata,
    *,
    exclude_indices: Optional[Sequence[object]],
    exclude_cell_types: Optional[Sequence[str]],
    annotation_key: str,
) -> np.ndarray:
    n_obs = int(adata.n_obs)
    keep = np.ones(n_obs, dtype=bool)
    if not exclude_indices and not exclude_cell_types:
        return keep

    if exclude_indices:
        obs_names = np.asarray(adata.obs_names.astype(str))
        name_to_pos = {name: idx for idx, name in enumerate(obs_names)}
        for raw in exclude_indices:
            if raw is None:
                continue
            if isinstance(raw, (int, np.integer)):
                idx = int(raw)
                if idx < 0 or idx >= n_obs:
                    raise IndexError(f"exclude index {idx} out of range [0, {n_obs-1}]")
                keep[idx] = False
                continue
            token = str(raw).strip()
            if token == "":
                continue
            if token in name_to_pos:
                keep[name_to_pos[token]] = False
                continue
            try:
                idx = int(token)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid exclude index token '{token}'. Use integer row index or valid obs_name."
                ) from exc
            if idx < 0 or idx >= n_obs:
                raise IndexError(f"exclude index {idx} out of range [0, {n_obs-1}]")
            keep[idx] = False

    if exclude_cell_types:
        if annotation_key not in adata.obs.columns:
            raise KeyError(
                f"exclude_cell_types was provided but adata.obs['{annotation_key}'] is missing."
            )
        exclude_set = {str(x) for x in exclude_cell_types if str(x).strip() != ""}
        if exclude_set:
            labels = adata.obs[annotation_key].astype(str).values
            keep &= ~np.isin(labels, list(exclude_set))

    return keep


def _freeze_model_for_inference(model) -> Optional[list[tuple[torch.nn.Parameter, bool]]]:
    if not isinstance(model, nn.Module):
        return None
    state: list[tuple[torch.nn.Parameter, bool]] = []
    for param in model.parameters():
        state.append((param, bool(param.requires_grad)))
        if param.requires_grad:
            param.requires_grad_(False)
    return state


def _restore_model_after_inference(state: Optional[list[tuple[torch.nn.Parameter, bool]]]) -> None:
    if not state:
        return
    for param, requires_grad in state:
        param.requires_grad_(requires_grad)


def _euler_sdeint(sde: nn.Module, initial_state, dt: float, ts: torch.Tensor):
    import math

    device = initial_state[0].device
    t0 = float(ts[0].item())
    tf = float(ts[-1].item())
    current_state = initial_state
    current_time = t0

    output_states = []
    ts_list = [float(x) for x in ts.tolist()]
    next_output_idx = 0

    while current_time <= tf + 1e-8:
        if current_time >= ts_list[next_output_idx] - 1e-8:
            output_states.append(current_state)
            next_output_idx += 1
            if next_output_idx >= len(ts_list):
                break

        t_tensor = torch.tensor([current_time], device=device, dtype=torch.float32)
        f_z, f_lnw = sde.f(t_tensor, current_state)
        noise_z = torch.randn_like(current_state[0]) * math.sqrt(dt)
        g_z = sde.g(t_tensor, current_state[0])
        new_z = current_state[0] + f_z * dt + g_z * noise_z
        new_lnw = current_state[1] + f_lnw * dt
        current_state = (new_z, new_lnw)
        current_time += dt

    while len(output_states) < len(ts_list):
        output_states.append(current_state)

    traj_z = torch.stack([state[0] for state in output_states], dim=0)
    traj_lnw = torch.stack([state[1] for state in output_states], dim=0)
    return traj_z, traj_lnw


def _apply_split_event(
    current_state,
    *,
    previous_weights: torch.Tensor,
    initial_count: int,
    noise_std: float,
    max_particles: Optional[int] = None,
):
    new_z, new_lnw = current_state
    device = new_z.device
    if new_z.shape[0] == 0:
        return current_state, torch.exp(new_lnw)

    r = (torch.exp(new_lnw) / previous_weights).reshape(-1)
    mask_split = r >= 1
    mask_extinct = ~mask_split

    repeated_source_z = None
    repeated_source_lnw = None
    repeat_counts = None
    if mask_split.any():
        r_split = r[mask_split]
        r_floor = torch.floor(r_split)
        r_frac = r_split - r_floor
        m_j = r_floor.to(torch.int64) + (
            torch.rand_like(r_frac) < r_frac
        ).to(torch.int64)
        valid_mask = m_j > 0
        if valid_mask.any():
            repeated_source_z = new_z[mask_split][valid_mask]
            repeated_source_lnw = new_lnw[mask_split][valid_mask]
            repeat_counts = m_j[valid_mask]

    keep_mask = None
    if mask_extinct.any():
        r_extinct = r[mask_extinct]
        keep_mask = torch.rand_like(r_extinct) < r_extinct

    n_split = 0 if repeat_counts is None else int(repeat_counts.sum().item())
    n_extinct = 0 if keep_mask is None else int(keep_mask.sum().item())
    n_after = n_split + n_extinct
    if max_particles is not None and n_after > int(max_particles):
        raise RuntimeError(
            "Split-SDE particle limit exceeded before allocation: "
            f"requested={n_after}, max_particles={int(max_particles)}. "
            "Reduce the time horizon/growth multiplier or raise the explicit limit."
        )

    if repeat_counts is not None:
        repeated_z = torch.repeat_interleave(
            repeated_source_z, repeat_counts, dim=0
        )
        repeated_lnw = torch.repeat_interleave(
            repeated_source_lnw, repeat_counts, dim=0
        )
        noise = torch.normal(
            0,
            float(noise_std),
            size=repeated_z.shape,
            device=device,
        )
        split_z = repeated_z + noise
        split_lnw = repeated_lnw
    else:
        split_z = torch.empty(0, new_z.shape[1], device=device)
        split_lnw = torch.empty(0, 1, device=device)

    if keep_mask is not None:
        extinct_z = new_z[mask_extinct][keep_mask]
        extinct_lnw = new_lnw[mask_extinct][keep_mask]
    else:
        extinct_z = torch.empty(0, new_z.shape[1], device=device)
        extinct_lnw = torch.empty(0, 1, device=device)

    if split_z.shape[0] > 0 or extinct_z.shape[0] > 0:
        new_z = torch.cat([split_z, extinct_z], dim=0)
        new_lnw = torch.cat([split_lnw, extinct_lnw], dim=0)
        new_lnw = torch.log(
            torch.ones(new_z.shape[0], 1, device=device) / int(initial_count)
        )
    else:
        new_z = torch.empty(0, current_state[0].shape[1], device=device)
        new_lnw = torch.empty(0, 1, device=device)
    return (new_z, new_lnw), torch.exp(new_lnw)


def _euler_sdeint_split(
    sde: nn.Module,
    initial_state,
    dt: float,
    ts: torch.Tensor,
    noise_std: float = 0.01,
    resample_dt: Optional[float] = None,
    max_particles: Optional[int] = None,
):
    import math

    dt_value = float(dt)
    if not math.isfinite(dt_value) or dt_value <= 0:
        raise ValueError("dt must be finite and > 0.")
    if int(ts.numel()) == 0:
        raise ValueError("ts must contain at least one output time.")
    if max_particles is not None and int(max_particles) <= 0:
        raise ValueError("max_particles must be > 0 when provided.")
    device = initial_state[0].device
    ts_list = [float(x) for x in ts.tolist()]
    if not all(math.isfinite(value) for value in ts_list):
        raise ValueError("ts values must all be finite.")
    if any(b <= a for a, b in zip(ts_list[:-1], ts_list[1:])):
        raise ValueError("ts must be strictly increasing.")
    t0 = ts_list[0]
    current_state = initial_state
    current_time = t0

    if resample_dt is not None:
        event_dt = float(resample_dt)
        if not math.isfinite(event_dt) or event_dt <= 0:
            raise ValueError("resample_dt must be finite and > 0 when provided.")
        t0 = ts_list[0]
        aligned_steps = [round((value - t0) / event_dt) for value in ts_list]
        misaligned = [
            value
            for value, step in zip(ts_list, aligned_steps)
            if not math.isclose(
                value,
                t0 + step * event_dt,
                rel_tol=0.0,
                abs_tol=1e-6,
            )
        ]
        if misaligned:
            raise ValueError(
                "Every requested output time must lie on the fixed split-event "
                f"grid resample_dt={event_dt}; misaligned={misaligned}."
            )

        output_states = [initial_state]
        current_state = initial_state
        current_time = t0
        initial_count = int(initial_state[0].shape[0])
        previous_weights = torch.exp(initial_state[1])
        output_by_step = {
            int(step): index for index, step in enumerate(aligned_steps)
        }
        final_step = int(aligned_steps[-1])
        for event_step in range(1, final_step + 1):
            event_time = t0 + event_step * event_dt
            while current_time < event_time - 1e-8:
                if current_state[0].shape[0] == 0:
                    current_time = event_time
                    break
                step_dt = min(float(dt), event_time - current_time)
                t_tensor = torch.tensor(
                    [current_time], device=device, dtype=torch.float32
                )
                f_z, f_lnw = sde.f(t_tensor, current_state)
                noise_z = torch.randn_like(current_state[0]) * math.sqrt(step_dt)
                g_z = sde.g(t_tensor, current_state[0])
                current_state = (
                    current_state[0] + f_z * step_dt + g_z * noise_z,
                    current_state[1] + f_lnw * step_dt,
                )
                current_time += step_dt
            current_state, previous_weights = _apply_split_event(
                current_state,
                previous_weights=previous_weights,
                initial_count=initial_count,
                noise_std=noise_std,
                max_particles=max_particles,
            )
            if event_step in output_by_step:
                output_states.append(current_state)

        if len(output_states) != len(ts_list):
            raise RuntimeError(
                "Fixed split-event integrator recorded "
                f"{len(output_states)} states for {len(ts_list)} outputs."
            )
        return [state[0] for state in output_states], [
            state[1] for state in output_states
        ]

    # The state at ts[0] is the supplied initial condition. Historical code
    # integrated one dt step before recording it, shifting every nominal
    # segment start and causing adjacent piecewise segments to overwrite a
    # boundary with t_start + dt.
    output_states = [current_state]
    next_output_idx = 1
    w_prev = torch.exp(current_state[1])

    while next_output_idx < len(ts_list):
        target_time = ts_list[next_output_idx]
        while current_time < target_time - 1e-8:
            if current_state[0].shape[0] == 0:
                current_time = target_time
                break
            step_dt = min(float(dt), target_time - current_time)
            t_tensor = torch.tensor([current_time], device=device, dtype=torch.float32)
            f_z, f_lnw = sde.f(t_tensor, current_state)
            noise_z = torch.randn_like(current_state[0]) * math.sqrt(step_dt)
            g_z = sde.g(t_tensor, current_state[0])
            new_z = current_state[0] + f_z * step_dt + g_z * noise_z
            new_lnw = current_state[1] + f_lnw * step_dt
            current_state = (new_z, new_lnw)
            current_time += step_dt

        current_state, w_prev = _apply_split_event(
            current_state,
            previous_weights=w_prev,
            initial_count=int(initial_state[0].shape[0]),
            noise_std=noise_std,
            max_particles=max_particles,
        )
        output_states.append(current_state)
        next_output_idx += 1

    traj_z = [state[0] for state in output_states]
    traj_lnw = [state[1] for state in output_states]
    return traj_z, traj_lnw


def sample_observed_x0(
    df,
    *,
    time_value: float,
    feature_cols: Sequence[str],
    label_col: str,
    n_samples_cap: Optional[int],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    subset = df[df["samples"] == float(time_value)]
    X = subset[list(feature_cols)].values.astype(np.float32)
    labels = subset[label_col].astype(str).values
    if n_samples_cap is None:
        return X, labels
    cap = int(n_samples_cap)
    if cap <= 0:
        raise ValueError("n_samples_cap must be > 0 when provided")
    if X.shape[0] <= cap:
        return X, labels
    idx = rng.choice(X.shape[0], size=cap, replace=False)
    return X[idx], labels[idx]


def _compute_spatial_warp_displacements(
    query_xy: np.ndarray,
    anchor_source_xy: np.ndarray,
    anchor_target_xy: np.ndarray,
    *,
    k: int,
    eps: float,
) -> np.ndarray:
    from sklearn.neighbors import NearestNeighbors

    query_xy = np.asarray(query_xy, dtype=np.float32)
    anchor_source_xy = np.asarray(anchor_source_xy, dtype=np.float32)
    anchor_target_xy = np.asarray(anchor_target_xy, dtype=np.float32)

    if query_xy.ndim != 2 or query_xy.shape[1] < 2:
        raise ValueError("query_xy must be a 2D array with at least 2 columns")
    if anchor_source_xy.ndim != 2 or anchor_source_xy.shape[1] != 2:
        raise ValueError("anchor_source_xy must have shape (n, 2)")
    if anchor_target_xy.ndim != 2 or anchor_target_xy.shape[1] != 2:
        raise ValueError("anchor_target_xy must have shape (m, 2)")
    if anchor_source_xy.shape[0] == 0 or anchor_target_xy.shape[0] == 0:
        return np.zeros((query_xy.shape[0], 2), dtype=np.float32)

    target_nn = NearestNeighbors(n_neighbors=1)
    target_nn.fit(anchor_target_xy)
    _, target_idx = target_nn.kneighbors(anchor_source_xy)
    anchor_disp = anchor_target_xy[target_idx[:, 0]] - anchor_source_xy

    k_eff = max(1, min(int(k), anchor_source_xy.shape[0]))
    source_nn = NearestNeighbors(n_neighbors=k_eff)
    source_nn.fit(anchor_source_xy)
    dists, src_idx = source_nn.kneighbors(query_xy[:, :2])

    weights = 1.0 / np.maximum(dists, float(eps))
    weights /= weights.sum(axis=1, keepdims=True)
    disp = (anchor_disp[src_idx] * weights[..., None]).sum(axis=1)
    return disp.astype(np.float32, copy=False)


def simulate_sde_points_split_from_x0(
    *,
    x0: np.ndarray,
    f_net,
    score_net,
    ts_points: Sequence[float],
    dt: float,
    sigma: float,
    sigma_by_dim: Optional[Sequence[float]],
    growth_alpha: float,
    interaction_m: int,
    device: str,
    verbose: bool = True,
    resample_dt: Optional[float] = None,
    max_particles: Optional[int] = None,
) -> np.ndarray:
    from CytoBridge.tl.core.interaction import cal_interaction

    if len(ts_points) < 1:
        raise ValueError("ts_points must be non-empty")
    x0_array = np.asarray(x0, dtype=np.float32)
    if x0_array.ndim != 2 or x0_array.shape[0] == 0:
        raise ValueError("x0 must be a non-empty two-dimensional array")
    if not np.isfinite(x0_array).all():
        raise ValueError("x0 must contain only finite values")

    x0_t = torch.tensor(x0_array, device=device)
    lnw0 = torch.log(torch.ones(x0_t.shape[0], 1, device=device) / x0_t.shape[0])
    initial_state = (x0_t, lnw0)

    class SDE(nn.Module):
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
                if sigma_arr.shape[0] != x0_t.shape[1]:
                    raise ValueError(
                        f"sigma_by_dim must have length {x0_t.shape[1]}, got {sigma_arr.shape[0]}"
                    )
                self.register_buffer("sigma_vec", torch.tensor(sigma_arr, dtype=torch.float32))
                self.sigma = None

        def f(self, t, y):
            z, lnw = y
            with torch.no_grad():
                drift = self.drift(t, z)
                dlnw = self.g_net(t, z) * growth_alpha
                net_forces = cal_interaction(
                    z=z,
                    lnw=lnw,
                    interaction_potential=self.interaction,
                    m=interaction_m,
                    t=t,
                )
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
            "[piecewise split-SDE] start | "
            f"n_init={x0_t.shape[0]}, ts_points={len(ts_points)}, "
            f"dt={dt}, sigma={'vector' if sigma_by_dim is not None else sigma}, "
            f"growth_alpha={growth_alpha}, resample_dt={resample_dt}, "
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
    ts_tensor = torch.tensor(list(ts_points), dtype=torch.float32, device=device)
    sde_points, _ = _euler_sdeint_split(
        sde,
        initial_state,
        dt=dt,
        ts=ts_tensor,
        noise_std=0.0,
        resample_dt=resample_dt,
        max_particles=max_particles,
    )
    sde_point_np = [p.detach().cpu().numpy() for p in sde_points]

    if verbose:
        print(
            "[piecewise split-SDE] done | "
            f"timepoints={len(sde_point_np)}, "
            f"shape0={sde_point_np[0].shape if sde_point_np else None}"
        )
    return np.array(sde_point_np, dtype=object)


def apply_spatial_warp_to_segments(
    *,
    sde_points_split: np.ndarray,
    ts_points: Sequence[float],
    observed_time_points: Sequence[float],
    df,
    feature_cols_full: Sequence[str],
    label_col: str,
    rng: np.random.Generator,
    piecewise: bool,
    piecewise_include_end: bool,
    piecewise_endpoint_by_observed: Optional[Dict[float, np.ndarray]],
    use_real_for_observed: bool,
    k: int,
    eps: float,
    blend_observed_boundary_displacements: bool = False,
) -> np.ndarray:
    if len(ts_points) == 0 or len(observed_time_points) < 2:
        return sde_points_split

    sde_points_out = np.array(
        [np.asarray(p, dtype=np.float32).copy() for p in sde_points_split],
        dtype=object,
    )
    ts_index = {float(t): i for i, t in enumerate(ts_points)}

    if blend_observed_boundary_displacements:
        boundary_anchors: Dict[float, tuple[np.ndarray, np.ndarray]] = {}
        for time_value in observed_time_points:
            time_value = float(time_value)
            index = ts_index.get(time_value)
            if index is None:
                continue
            source_xy = np.asarray(
                sde_points_split[index], dtype=np.float32
            )[:, :2]
            if source_xy.shape[0] == 0:
                continue
            target, _ = sample_observed_x0(
                df,
                time_value=time_value,
                feature_cols=feature_cols_full,
                label_col=label_col,
                n_samples_cap=int(source_xy.shape[0]),
                rng=rng,
            )
            target_xy = np.asarray(target, dtype=np.float32)[:, :2]
            if target_xy.shape[0] > 0:
                boundary_anchors[time_value] = (source_xy, target_xy)

        for t_start, t_end in zip(
            observed_time_points[:-1], observed_time_points[1:]
        ):
            t_start = float(t_start)
            t_end = float(t_end)
            if t_start not in boundary_anchors or t_end not in boundary_anchors:
                continue
            source_start, target_start = boundary_anchors[t_start]
            source_end, target_end = boundary_anchors[t_end]
            segment_times = [
                float(value)
                for value in ts_points
                if t_start < float(value) < t_end
            ]
            if not use_real_for_observed:
                segment_times.extend([t_start, t_end])
            segment_times = sorted(set(segment_times))
            for time_value in segment_times:
                index = ts_index.get(time_value)
                if index is None:
                    continue
                raw = np.asarray(sde_points_split[index], dtype=np.float32)
                if raw.shape[0] == 0:
                    continue
                alpha = (time_value - t_start) / max(t_end - t_start, float(eps))
                start_disp = _compute_spatial_warp_displacements(
                    raw[:, :2], source_start, target_start, k=k, eps=eps
                )
                end_disp = _compute_spatial_warp_displacements(
                    raw[:, :2], source_end, target_end, k=k, eps=eps
                )
                displayed = raw.copy()
                displayed[:, :2] = (
                    raw[:, :2]
                    + (1.0 - float(alpha)) * start_disp
                    + float(alpha) * end_disp
                )
                sde_points_out[index] = displayed
            print(
                f"[spatial-warp continuous-display] segment {t_start}->{t_end} | "
                f"targets={segment_times}"
            )
        return sde_points_out

    for t_start, t_end in zip(observed_time_points[:-1], observed_time_points[1:]):
        t_start = float(t_start)
        t_end = float(t_end)
        interior_ts = sorted([float(t) for t in ts_points if t_start < float(t) < t_end])

        if piecewise:
            if not piecewise_include_end or not piecewise_endpoint_by_observed:
                print(
                    f"[spatial-warp] skip segment {t_start}->{t_end}: "
                    "--split-sde-piecewise requires --split-sde-piecewise-include-end for warp anchors"
                )
                continue
            source_endpoint = piecewise_endpoint_by_observed.get(t_end)
            if source_endpoint is None:
                print(f"[spatial-warp] skip segment {t_start}->{t_end}: missing simulated endpoint cache")
                continue
            source_endpoint_xy = np.asarray(source_endpoint, dtype=np.float32)[:, :2]
        else:
            idx_end = ts_index.get(t_end)
            if idx_end is None:
                continue
            source_endpoint_xy = np.asarray(sde_points_out[idx_end], dtype=np.float32)[:, :2]

        if source_endpoint_xy.shape[0] == 0:
            continue

        X_target, _ = sample_observed_x0(
            df,
            time_value=t_end,
            feature_cols=feature_cols_full,
            label_col=label_col,
            n_samples_cap=int(source_endpoint_xy.shape[0]),
            rng=rng,
        )
        target_endpoint_xy = np.asarray(X_target, dtype=np.float32)[:, :2]
        if target_endpoint_xy.shape[0] == 0:
            continue

        segment_apply_ts = list(interior_ts)
        if (not use_real_for_observed) and (t_end in ts_index):
            segment_apply_ts.append(t_end)
        if len(segment_apply_ts) == 0:
            continue

        for t_val in segment_apply_ts:
            idx = ts_index.get(float(t_val))
            if idx is None:
                continue
            alpha = (float(t_val) - t_start) / max(t_end - t_start, float(eps))
            pts = np.asarray(sde_points_out[idx], dtype=np.float32)
            if pts.shape[0] == 0:
                continue
            disp = _compute_spatial_warp_displacements(
                pts[:, :2],
                source_endpoint_xy,
                target_endpoint_xy,
                k=k,
                eps=eps,
            )
            pts[:, :2] = pts[:, :2] + float(alpha) * disp
            sde_points_out[idx] = pts

        print(
            f"[spatial-warp] segment {t_start}->{t_end} | "
            f"anchors_sim={source_endpoint_xy.shape[0]} anchors_real={target_endpoint_xy.shape[0]} "
            f"targets={segment_apply_ts}"
        )

    return sde_points_out


def simulate_piecewise_spatially_warped_split(
    *,
    x0: np.ndarray,
    f_net,
    score_net,
    observed_time_points: Sequence[float],
    ts_points: Sequence[float],
    df,
    feature_cols_full: Sequence[str],
    label_col: str,
    dt: float,
    sigma: float,
    sigma_by_dim: Optional[Sequence[float]],
    growth_alpha: float,
    interaction_m: int,
    device: str,
    rng: np.random.Generator,
    k: int,
    eps: float,
    return_prewarp: bool = False,
    warp_visualization_only: bool = False,
    use_real_for_observed: bool = True,
    resample_dt: Optional[float] = None,
    max_particles: Optional[int] = None,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    ts_sorted = [float(t) for t in ts_points]
    observed_sorted = [float(t) for t in observed_time_points if ts_sorted[0] <= float(t) <= ts_sorted[-1]]
    if len(observed_sorted) < 2:
        fallback = simulate_sde_points_split_from_x0(
            x0=x0,
            f_net=f_net,
            score_net=score_net,
            ts_points=ts_sorted,
            dt=dt,
            sigma=sigma,
            sigma_by_dim=sigma_by_dim,
            growth_alpha=growth_alpha,
            interaction_m=interaction_m,
            device=device,
            verbose=True,
            resample_dt=resample_dt,
            max_particles=max_particles,
        )
        if return_prewarp:
            return fallback, np.array(
                [np.asarray(points, dtype=np.float32).copy() for points in fallback],
                dtype=object,
            )
        return fallback

    if warp_visualization_only:
        # The warp is a rendering transform, so it must not segment or restart
        # the model trajectory.  Simulate x *and* log-mass once over the full
        # time grid, retain that state for labels/downstream analysis, then
        # bend only a display copy toward each observed endpoint.
        prewarp = simulate_sde_points_split_from_x0(
            x0=x0,
            f_net=f_net,
            score_net=score_net,
            ts_points=ts_sorted,
            dt=dt,
            sigma=sigma,
            sigma_by_dim=sigma_by_dim,
            growth_alpha=growth_alpha,
            interaction_m=interaction_m,
            device=device,
            verbose=True,
            resample_dt=resample_dt,
            max_particles=max_particles,
        )
        warped = apply_spatial_warp_to_segments(
            sde_points_split=prewarp,
            ts_points=ts_sorted,
            observed_time_points=observed_sorted,
            df=df,
            feature_cols_full=feature_cols_full,
            label_col=label_col,
            rng=rng,
            piecewise=False,
            piecewise_include_end=False,
            piecewise_endpoint_by_observed=None,
            use_real_for_observed=bool(use_real_for_observed),
            k=k,
            eps=eps,
            blend_observed_boundary_displacements=True,
        )
        if not return_prewarp:
            return warped
        return warped, np.array(
            [np.asarray(points, dtype=np.float32).copy() for points in prewarp],
            dtype=object,
        )

    current_x0 = np.asarray(x0, dtype=np.float32)
    points_by_time: Dict[float, np.ndarray] = {}
    prewarp_points_by_time: Dict[float, np.ndarray] = {}

    for t_start, t_end in zip(observed_sorted[:-1], observed_sorted[1:]):
        seg_ts = [float(t) for t in ts_sorted if float(t_start) <= float(t) <= float(t_end)]
        if len(seg_ts) == 0:
            continue
        if len(seg_ts) == 1:
            points_by_time[float(seg_ts[0])] = np.asarray(current_x0, dtype=np.float32).copy()
            prewarp_points_by_time[float(seg_ts[0])] = np.asarray(
                current_x0, dtype=np.float32
            ).copy()
            continue

        print(f"[spatial-warp piecewise] segment {t_start}->{t_end} | targets={seg_ts}")
        seg_points = simulate_sde_points_split_from_x0(
            x0=current_x0,
            f_net=f_net,
            score_net=score_net,
            ts_points=seg_ts,
            dt=dt,
            sigma=sigma,
            sigma_by_dim=sigma_by_dim,
            growth_alpha=growth_alpha,
            interaction_m=interaction_m,
            device=device,
            verbose=True,
            resample_dt=resample_dt,
            max_particles=max_particles,
        )

        source_endpoint_xy = np.asarray(seg_points[-1], dtype=np.float32)[:, :2]
        X_target, _ = sample_observed_x0(
            df,
            time_value=float(t_end),
            feature_cols=feature_cols_full,
            label_col=label_col,
            n_samples_cap=min(int(source_endpoint_xy.shape[0]), int((df["samples"] == float(t_end)).sum())),
            rng=rng,
        )
        target_endpoint_xy = np.asarray(X_target, dtype=np.float32)[:, :2]

        raw_start_xy = np.asarray(seg_points[0], dtype=np.float32)[:, :2]
        previous_display_boundary = points_by_time.get(float(t_start))
        start_display_xy = (
            np.asarray(previous_display_boundary, dtype=np.float32)[:, :2]
            if warp_visualization_only and previous_display_boundary is not None
            else None
        )

        for t_val, pts_raw in zip(seg_ts, seg_points):
            pts = np.asarray(pts_raw, dtype=np.float32).copy()
            # Preserve the dynamical state before this segment's display warp.
            # At a shared endpoint, keep the preceding segment's unwarped
            # endpoint rather than overwriting it with the next segment start.
            prewarp_points_by_time.setdefault(
                float(t_val), np.asarray(pts_raw, dtype=np.float32).copy()
            )
            alpha = (float(t_val) - float(t_start)) / max(float(t_end - t_start), float(eps))
            if warp_visualization_only and start_display_xy is not None:
                start_disp = _compute_spatial_warp_displacements(
                    pts[:, :2],
                    raw_start_xy,
                    start_display_xy,
                    k=k,
                    eps=eps,
                )
                end_disp = _compute_spatial_warp_displacements(
                    pts[:, :2],
                    source_endpoint_xy,
                    target_endpoint_xy,
                    k=k,
                    eps=eps,
                )
                pts[:, :2] = (
                    pts[:, :2]
                    + (1.0 - float(alpha)) * start_disp
                    + float(alpha) * end_disp
                )
                if float(t_val) == float(t_start):
                    # The shared boundary has the same particle ordering as
                    # the next model start, so preserve it bit-for-bit.
                    pts = np.asarray(previous_display_boundary, dtype=np.float32).copy()
            elif alpha > 0.0:
                end_disp = _compute_spatial_warp_displacements(
                    pts[:, :2],
                    source_endpoint_xy,
                    target_endpoint_xy,
                    k=k,
                    eps=eps,
                )
                pts[:, :2] = pts[:, :2] + float(alpha) * end_disp
            points_by_time[float(t_val)] = pts

        current_x0 = np.asarray(points_by_time[float(t_end)], dtype=np.float32).copy()

    missing = [float(t) for t in ts_sorted if float(t) not in points_by_time]
    if missing:
        raise ValueError(f"Piecewise spatial-warp split-SDE missing timepoints: {missing}")

    warped = np.array([points_by_time[float(t)] for t in ts_sorted], dtype=object)
    if not return_prewarp:
        return warped
    missing_prewarp = [
        float(t) for t in ts_sorted if float(t) not in prewarp_points_by_time
    ]
    if missing_prewarp:
        raise ValueError(
            "Piecewise spatial-warp split-SDE missing prewarp timepoints: "
            f"{missing_prewarp}"
        )
    prewarp = np.array(
        [prewarp_points_by_time[float(t)] for t in ts_sorted], dtype=object
    )
    return warped, prewarp


def simulate_sde_points(
    adata: Optional["ad.AnnData"] = None,
    model=None,
    dim: Optional[int] = None,
    time_index: int = 0,
    n_samples: int = 5000,
    ts_points: Optional[Sequence[float]] = None,
    dt: float = 0.1,
    sigma: float = 0.0,
    include_score: bool = False,
    interaction_m: int = 512,
    device: str = "cuda",
    time_key: Optional[str] = None,
    obsm_key: str = "X_latent",
    spatial_key: str = "spatial_aligned",
    concat_spatial: Optional[bool] = None,
    exclude_indices: Optional[Sequence[object]] = None,
    exclude_cell_types: Optional[Sequence[str]] = None,
    annotation_key: str = "Annotation",
    df=None,
    f_net=None,
    score_net=None,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    from CytoBridge.tl.core.interaction import cal_interaction

    if df is not None:
        if f_net is None or score_net is None:
            raise ValueError("Legacy dataframe mode requires both f_net and score_net.")
        if dim is None:
            raise ValueError("Legacy dataframe mode requires dim.")
        if ts_points is None:
            ts_points = [0, 1, 2, 3, 4]

        time_points = df["samples"].unique()
        if time_index < 0 or time_index >= len(time_points):
            raise ValueError(f"time_index={time_index} out of range [0, {len(time_points)-1}]")

        t0 = time_points[time_index]
        numeric_cols = ["samples"] + [f"x{i}" for i in range(1, int(dim) + 1)]
        data = torch.tensor(df[df["samples"] == t0][numeric_cols].values, dtype=torch.float32)
        x0 = data[:, 1:].requires_grad_().to(device)

        if x0.shape[0] > n_samples:
            indices = torch.randperm(x0.shape[0])[:n_samples]
            x0 = x0[indices]

        lnw0 = torch.log(torch.ones(x0.shape[0], 1, device=device) / x0.shape[0])
        initial_state = (x0, lnw0)

        class SDE(nn.Module):
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
                    net_forces = cal_interaction(
                        z=z,
                        lnw=lnw,
                        interaction_potential=self.interaction,
                        m=interaction_m,
                        t=t,
                    )
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
        sde_point, traj_lnw = _euler_sdeint(sde, initial_state, dt=dt, ts=ts_tensor)
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

    X, times, time_points, dim = _prepare_adata_arrays(
        adata,
        dim,
        time_key=time_key,
        obsm_key=obsm_key,
        spatial_key=spatial_key,
        concat_spatial=concat_spatial,
    )
    if ts_points is None:
        ts_points = [0, 1, 2, 3, 4]
    if time_index < 0 or time_index >= len(time_points):
        raise ValueError(f"time_index={time_index} out of range [0, {len(time_points)-1}]")

    t0 = time_points[time_index]
    mask = _time_mask(times, t0)
    keep_mask = _build_perturbation_keep_mask(
        adata,
        exclude_indices=exclude_indices,
        exclude_cell_types=exclude_cell_types,
        annotation_key=annotation_key,
    )
    mask = mask & keep_mask
    if not np.any(mask):
        raise ValueError(f"No rows found for time point {t0}.")
    data = torch.tensor(X[mask], dtype=torch.float32)
    x0 = data.to(device)
    if x0.shape[0] > n_samples:
        indices = torch.randperm(x0.shape[0])[:n_samples]
        x0 = x0[indices]

    lnw0 = torch.log(torch.ones(x0.shape[0], 1, device=device) / x0.shape[0])
    initial_state = (x0, lnw0)

    interaction_net = getattr(model, "interaction_net", None)
    components = set(getattr(model, "components", []))
    use_mass = bool(getattr(model, "use_growth_in_ode_inter", True))

    class SDE(nn.Module):
        noise_type = "diagonal"
        sde_type = "ito"

        def __init__(self, sigma_val: float):
            super().__init__()
            self.sigma = sigma_val

        def f(self, t, y):
            z, lnw = y
            t_expand = t.expand(z.shape[0], 1).to(dtype=z.dtype)
            with torch.no_grad():
                if "velocity" not in components:
                    raise ValueError("Model missing velocity_net.")
                drift = model.predict_velocity(t=t_expand, x=z)
                if "growth" in components:
                    dlnw = model.predict_growth(t=t_expand, x=z)
                else:
                    dlnw = torch.zeros_like(lnw)

            net_forces = torch.zeros_like(z)
            if "interaction" in components and interaction_net is not None:
                if getattr(interaction_net, "requires_time", False):
                    with torch.no_grad():
                        net_forces = cal_interaction(
                            z=z,
                            lnw=lnw,
                            interaction_potential=interaction_net,
                            m=interaction_m,
                            cutoff=1000,
                            use_mass=use_mass,
                            t=t,
                        ).float()
                else:
                    net_forces = cal_interaction(
                        z=z,
                        lnw=lnw,
                        interaction_potential=interaction_net,
                        m=interaction_m,
                        cutoff=1000,
                        use_mass=use_mass,
                        t=t,
                    ).float()

            if include_score and "score" in components:
                z_req = z.detach().requires_grad_(True)
                _, score_grad = model.compute_score(
                    t=t_expand.detach(),
                    x=z_req,
                    create_graph=False,
                )
                drift = drift + score_grad
            return (drift + net_forces, dlnw)

        def g(self, t, z):
            return torch.ones_like(z) * self.sigma

    freeze_state = _freeze_model_for_inference(model)
    try:
        sde = SDE(sigma_val=sigma)
        ts_tensor = torch.tensor(ts_points, dtype=torch.float32, device=device)
        sde_point, traj_lnw = _euler_sdeint(sde, initial_state, dt=dt, ts=ts_tensor)
        weight = torch.exp(traj_lnw)
        sde_point_np = [p.detach().cpu().numpy() for p in sde_point]
        return np.array(sde_point_np, dtype=object), weight.detach().cpu().numpy()
    finally:
        _restore_model_after_inference(freeze_state)


def simulate_sde_points_split(
    adata: Optional["ad.AnnData"] = None,
    model=None,
    dim: Optional[int] = None,
    time_index: int = 0,
    n_samples: int = 5000,
    ts_points: Optional[Sequence[float]] = None,
    dt: float = 0.01,
    sigma: float = 0.03,
    interaction_m: int = 1024,
    device: str = "cuda",
    time_key: Optional[str] = None,
    obsm_key: str = "X_latent",
    spatial_key: str = "spatial_aligned",
    concat_spatial: Optional[bool] = None,
    exclude_indices: Optional[Sequence[object]] = None,
    exclude_cell_types: Optional[Sequence[str]] = None,
    annotation_key: str = "Annotation",
    df=None,
    f_net=None,
    score_net=None,
    sigma_by_dim: Optional[Sequence[float]] = None,
    growth_alpha: float = 0.5,
    verbose: bool = True,
    resample_dt: Optional[float] = None,
    max_particles: Optional[int] = None,
) -> np.ndarray:
    from CytoBridge.tl.core.interaction import cal_interaction

    if df is not None:
        if f_net is None or score_net is None:
            raise ValueError("Legacy dataframe mode requires both f_net and score_net.")
        if dim is None:
            raise ValueError("Legacy dataframe mode requires dim.")
        if ts_points is None:
            ts_points = [0, 1, 2, 3, 4]

        time_points = df["samples"].unique()
        if time_index < 0 or time_index >= len(time_points):
            raise ValueError(f"time_index={time_index} out of range [0, {len(time_points)-1}]")

        t0 = time_points[time_index]
        numeric_cols = ["samples"] + [f"x{i}" for i in range(1, int(dim) + 1)]
        data = torch.tensor(df[df["samples"] == t0][numeric_cols].values, dtype=torch.float32)
        x0 = data[:, 1:].to(device)

        if x0.shape[0] > n_samples:
            indices = torch.randperm(x0.shape[0])[:n_samples]
            x0 = x0[indices]

        return simulate_sde_points_split_from_x0(
            x0=x0.detach().cpu().numpy(),
            f_net=f_net,
            score_net=score_net,
            ts_points=ts_points,
            dt=dt,
            sigma=sigma,
            sigma_by_dim=sigma_by_dim,
            growth_alpha=growth_alpha,
            interaction_m=interaction_m,
            device=device,
            verbose=verbose,
            resample_dt=resample_dt,
            max_particles=max_particles,
        )

    X, times, time_points, dim = _prepare_adata_arrays(
        adata,
        dim,
        time_key=time_key,
        obsm_key=obsm_key,
        spatial_key=spatial_key,
        concat_spatial=concat_spatial,
    )
    if ts_points is None:
        ts_points = [0, 1, 2, 3, 4]
    if time_index < 0 or time_index >= len(time_points):
        raise ValueError(f"time_index={time_index} out of range [0, {len(time_points)-1}]")

    t0 = time_points[time_index]
    mask = _time_mask(times, t0)
    keep_mask = _build_perturbation_keep_mask(
        adata,
        exclude_indices=exclude_indices,
        exclude_cell_types=exclude_cell_types,
        annotation_key=annotation_key,
    )
    mask = mask & keep_mask
    if not np.any(mask):
        raise ValueError(f"No rows found for time point {t0}.")
    data = torch.tensor(X[mask], dtype=torch.float32)
    x0 = data.to(device)
    if x0.shape[0] > n_samples:
        indices = torch.randperm(x0.shape[0])[:n_samples]
        x0 = x0[indices]

    lnw0 = torch.log(torch.ones(x0.shape[0], 1, device=device) / x0.shape[0])
    initial_state = (x0, lnw0)

    interaction_net = getattr(model, "interaction_net", None)
    components = set(getattr(model, "components", []))
    use_mass = bool(getattr(model, "use_growth_in_ode_inter", True))

    class SDE(nn.Module):
        noise_type = "diagonal"
        sde_type = "ito"

        def __init__(self, sigma_val: float):
            super().__init__()
            self.sigma = sigma_val

        def f(self, t, y):
            z, lnw = y
            t_expand = t.expand(z.shape[0], 1).to(dtype=z.dtype)
            with torch.no_grad():
                if "velocity" not in components:
                    raise ValueError("Model missing velocity_net.")
                drift = model.predict_velocity(t=t_expand, x=z)
                if "growth" in components:
                    dlnw = model.predict_growth(t=t_expand, x=z)
                else:
                    dlnw = torch.zeros_like(lnw)

            net_forces = torch.zeros_like(z)
            if "interaction" in components and interaction_net is not None:
                if getattr(interaction_net, "requires_time", False):
                    with torch.no_grad():
                        net_forces = cal_interaction(
                            z=z,
                            lnw=lnw,
                            interaction_potential=interaction_net,
                            m=interaction_m,
                            cutoff=1000,
                            use_mass=use_mass,
                            t=t,
                        ).float()
                else:
                    net_forces = cal_interaction(
                        z=z,
                        lnw=lnw,
                        interaction_potential=interaction_net,
                        m=interaction_m,
                        cutoff=1000,
                        use_mass=use_mass,
                        t=t,
                    ).float()

            if "score" in components:
                z_req = z.detach().requires_grad_(True)
                _, score_grad = model.compute_score(
                    t=t_expand.detach(),
                    x=z_req,
                    create_graph=False,
                )
                drift = drift + score_grad
            return (drift + net_forces, dlnw)

        def g(self, t, z):
            return torch.ones_like(z) * self.sigma

    freeze_state = _freeze_model_for_inference(model)
    try:
        sde = SDE(sigma_val=sigma)
        ts_tensor = torch.tensor(ts_points, dtype=torch.float32, device=device)
        sde_points, _ = _euler_sdeint_split(
            sde,
            initial_state,
            dt=dt,
            ts=ts_tensor,
            noise_std=0.0,
            resample_dt=resample_dt,
            max_particles=max_particles,
        )
        sde_point_np = [p.detach().cpu().numpy() for p in sde_points]
        return np.array(sde_point_np, dtype=object)
    finally:
        _restore_model_after_inference(freeze_state)


def compute_velocity_components(
    data: np.ndarray,
    time_value: float,
    model,
    interaction_m: int = 1024,
    interaction_threshold: int = 1000,
    device: str = "cuda",
    spatial_dim: int = 2,
) -> Dict[str, np.ndarray]:
    from CytoBridge.tl.core.interaction import cal_interaction

    data = np.asarray(data, dtype=np.float32)
    n_cells = data.shape[0]
    data_tensor = torch.tensor(data, device=device, dtype=torch.float32)
    t_tensor = torch.full((n_cells, 1), float(time_value), device=device, dtype=torch.float32)
    interaction_net = getattr(model, "interaction_net", None)
    components = set(getattr(model, "components", []))
    use_mass = bool(getattr(model, "use_growth_in_ode_inter", True))
    freeze_state = _freeze_model_for_inference(model)

    try:
        with torch.no_grad():
            if "velocity" not in components:
                raise ValueError("Model missing velocity_net.")
            drift = model.predict_velocity(t=t_tensor, x=data_tensor)
        drift_np = drift.detach().cpu().numpy()

        lnw = torch.log(torch.ones(n_cells, 1, device=device, dtype=torch.float32) / float(n_cells))
        interaction_np = np.zeros_like(drift_np)
        if "interaction" in components and interaction_net is not None:
            t_scalar = torch.tensor([float(time_value)], dtype=torch.float32, device=device)
            if getattr(interaction_net, "requires_time", False):
                with torch.no_grad():
                    interaction_t = cal_interaction(
                        z=data_tensor.detach(),
                        lnw=lnw,
                        interaction_potential=interaction_net,
                        m=interaction_m,
                        cutoff=float(interaction_threshold),
                        use_mass=use_mass,
                        t=t_scalar,
                    )
            else:
                interaction_t = cal_interaction(
                    z=data_tensor.detach(),
                    lnw=lnw,
                    interaction_potential=interaction_net,
                    m=interaction_m,
                    cutoff=float(interaction_threshold),
                    use_mass=use_mass,
                    t=t_scalar,
                )
            interaction_np = interaction_t.detach().cpu().numpy()

        score_np = np.zeros_like(drift_np)
        if "score" in components:
            _, score_grad = model.compute_score(
                t=t_tensor,
                x=data_tensor,
                create_graph=False,
            )
            score_np = score_grad.detach().cpu().numpy()

        return {
            "drift": drift_np,
            "interaction": interaction_np,
            "score": score_np,
            "full": drift_np + interaction_np + score_np,
        }
    finally:
        _restore_model_after_inference(freeze_state)


def _store_vector_component(
    adata: "ad.AnnData",
    *,
    name: str,
    values: np.ndarray,
    spatial_dim: int,
) -> None:
    adata.obsm[f"{name}_model"] = values
    if spatial_dim > 0:
        split = min(spatial_dim, values.shape[1])
        adata.obsm[f"{name}_spatial"] = values[:, :split]
        adata.obsm[f"{name}_latent"] = values[:, split:]
    else:
        adata.obsm[f"{name}_latent"] = values


def compute_velocity_components_from_adata(
    adata: "ad.AnnData",
    model,
    *,
    dim: Optional[int] = None,
    interaction_m: int = 1024,
    interaction_threshold: Optional[float] = None,
    device: str = "cuda",
    time_key: Optional[str] = None,
    obsm_key: str = "X_latent",
    spatial_key: str = "spatial_aligned",
    concat_spatial: Optional[bool] = None,
    write_to_adata: bool = True,
    reuse_if_present: bool = True,
) -> Dict[str, np.ndarray]:
    """Compute per-cell velocity decomposition from AnnData and optionally write back.

    Output keys follow training-time naming for consistency:
    - ``velocity_model``, ``interaction_model``, ``score_gradient_model``, ``full_drift_model``
    plus their ``*_spatial``/``*_latent`` splits when spatial features are present.
    """
    X, times, time_points, dim = _prepare_adata_arrays(
        adata,
        dim,
        time_key=time_key,
        obsm_key=obsm_key,
        spatial_key=spatial_key,
        concat_spatial=concat_spatial,
    )

    use_spatial = bool(concat_spatial) if concat_spatial is not None else (spatial_key in adata.obsm)
    spatial_dim = int(adata.obsm[spatial_key].shape[1]) if use_spatial and spatial_key in adata.obsm else 0

    components = set(getattr(model, "components", []))
    has_interaction = "interaction" in components
    has_score = "score" in components
    if interaction_threshold is None:
        interaction_threshold = float(getattr(getattr(model, "interaction_net", None), "cutoff", 1000.0))

    if write_to_adata and reuse_if_present:
        required = ["velocity_model", "full_drift_model"]
        if has_interaction:
            required.append("interaction_model")
        if has_score:
            required.append("score_gradient_model")
        if all(k in adata.obsm for k in required):
            out_cached = {
                "drift": np.asarray(adata.obsm["velocity_model"], dtype=np.float32),
                "interaction": (
                    np.asarray(adata.obsm["interaction_model"], dtype=np.float32)
                    if "interaction_model" in adata.obsm
                    else np.zeros((adata.n_obs, dim), dtype=np.float32)
                ),
                "score": (
                    np.asarray(adata.obsm["score_gradient_model"], dtype=np.float32)
                    if "score_gradient_model" in adata.obsm
                    else np.zeros((adata.n_obs, dim), dtype=np.float32)
                ),
                "full": np.asarray(adata.obsm["full_drift_model"], dtype=np.float32),
                "times": np.asarray(times, dtype=np.float64),
                "features": np.asarray(X, dtype=np.float32),
            }
            return out_cached

    n = X.shape[0]
    drift_all = np.zeros((n, dim), dtype=np.float32)
    interaction_all = np.zeros((n, dim), dtype=np.float32)
    score_all = np.zeros((n, dim), dtype=np.float32)
    full_all = np.zeros((n, dim), dtype=np.float32)

    for t_val in time_points:
        mask = _time_mask(times, t_val)
        if not np.any(mask):
            continue
        comp = compute_velocity_components(
            data=X[mask],
            time_value=float(t_val),
            model=model,
            interaction_m=interaction_m,
            interaction_threshold=float(interaction_threshold),
            device=device,
            spatial_dim=spatial_dim,
        )
        drift_all[mask] = comp["drift"]
        interaction_all[mask] = comp["interaction"]
        score_all[mask] = comp["score"]
        full_all[mask] = comp["full"]

    if write_to_adata:
        _store_vector_component(adata, name="velocity", values=drift_all, spatial_dim=spatial_dim)
        _store_vector_component(adata, name="interaction", values=interaction_all, spatial_dim=spatial_dim)
        _store_vector_component(adata, name="score_gradient", values=score_all, spatial_dim=spatial_dim)
        _store_vector_component(adata, name="full_drift", values=full_all, spatial_dim=spatial_dim)

    return {
        "drift": drift_all,
        "interaction": interaction_all,
        "score": score_all,
        "full": full_all,
        "times": np.asarray(times, dtype=np.float64),
        "features": np.asarray(X, dtype=np.float32),
    }


def compute_drift(
    adata: "ad.AnnData",
    model,
    dim: Optional[int] = None,
    interaction_m: int = 1024,
    device: str = "cuda",
    time_key: Optional[str] = None,
    obsm_key: str = "X_latent",
    spatial_key: str = "spatial_aligned",
    concat_spatial: Optional[bool] = None,
) -> Dict[str, np.ndarray]:
    X, times, time_points, dim = _prepare_adata_arrays(
        adata,
        dim,
        time_key=time_key,
        obsm_key=obsm_key,
        spatial_key=spatial_key,
        concat_spatial=concat_spatial,
    )
    result = {}
    for t_val in time_points:
        mask = _time_mask(times, t_val)
        if not np.any(mask):
            continue
        data = X[mask]
        components = compute_velocity_components(
            data=data,
            time_value=float(t_val),
            model=model,
            interaction_m=interaction_m,
            device=device,
        )
        result[t_val] = components["full"]
    return result


def compute_drift_from_adata(
    adata: "ad.AnnData",
    model,
    *,
    dim: Optional[int] = None,
    interaction_m: int = 1024,
    device: str = "cuda",
    time_key: Optional[str] = None,
    obsm_key: str = "X_latent",
    spatial_key: str = "spatial_aligned",
    concat_spatial: Optional[bool] = None,
) -> Dict[str, np.ndarray]:
    return compute_drift(
        adata=adata,
        model=model,
        dim=dim,
        interaction_m=interaction_m,
        device=device,
        time_key=time_key,
        obsm_key=obsm_key,
        spatial_key=spatial_key,
        concat_spatial=concat_spatial,
    )


def compute_umap_embedding(
    data: np.ndarray,
    n_neighbors: int = 30,
    min_dist: float = 0.3,
    seed: int = 0,
) -> tuple:
    try:
        import umap
    except ImportError as exc:
        raise ImportError(
            "umap-learn is required for UMAP embedding. Install with: pip install umap-learn"
        ) from exc

    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=seed)
    embedding = reducer.fit_transform(np.asarray(data, dtype=float))
    return embedding, reducer
