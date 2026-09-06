"""Numerical routines retained from the AGIST paper simulation.

The sampling order and output-time convention are unchanged from the archived
DeepRUOT/interaction.py used for Figure 2e.
"""

import math
import torch

def cal_interaction(z, lnw, interaction_potential, t, m=16, threshold=1000, use_mass=True, mass_detach=False):
    """Evaluate the paper model in randomly shuffled cell groups."""
    if m < 2:
        raise ValueError('Number of particles per group m must be greater than or equal to 2.')
    lnw = lnw.detach()
    w = torch.exp(lnw)
    n = w.shape[0]
    w = w * n
    batch_size, embed_dim = z.shape
    device = z.device
    if batch_size < 2:
        raise ValueError('Total number of particles must be at least 2.')
    perm = torch.randperm(batch_size, device=device)
    z_shuffled = z[perm]
    if batch_size % m == 0:
        num_full_groups = batch_size // m
        remainder = batch_size % m
    elif batch_size < m:
        num_full_groups = 0
        remainder = batch_size
    else:
        num_full_groups = batch_size // m - 1
        remainder = batch_size % m + m
    if remainder == 1:
        raise ValueError('Remainder group contains only 1 particle, isolated particles not allowed!')
    net_force = torch.zeros_like(z)
    for i in range(num_full_groups):
        groups_z = z_shuffled[i * m:(i + 1) * m]
        groups_w = w[perm[i * m:(i + 1) * m]]
        groups_lnw = torch.log(groups_w / groups_z.shape[0])
        force_groups = interaction_potential(groups_z, groups_lnw, t)
        net_force[perm[i * m:(i + 1) * m]] = force_groups
    if remainder > 0:
        groups_z = z_shuffled[num_full_groups * m:]
        groups_w = w[perm[num_full_groups * m:]]
        groups_lnw = torch.log(groups_w / groups_z.shape[0])
        force_groups = interaction_potential(groups_z, groups_lnw, t)
        net_force[perm[num_full_groups * m:]] = force_groups
    return net_force

def euler_sdeint_split(sde, initial_state, dt, ts, noise_std=0.01):
    """Use the archived AGIST sampling order and output-time convention."""
    device = initial_state[0].device
    t0 = ts[0].item()
    tf = ts[-1].item()
    current_state = initial_state
    current_time = t0
    output_states = []
    ts_list = ts.tolist()
    next_output_idx = 0
    w_prev = torch.exp(current_state[1])
    while current_time <= tf + 1e-08:
        t_tensor = torch.tensor([current_time], device=device)
        f_z, f_lnw = sde.f(t_tensor, current_state)
        noise_z = torch.randn_like(current_state[0]) * math.sqrt(dt)
        g_z = sde.g(t_tensor, current_state[0])
        new_z = current_state[0] + f_z * dt + g_z * noise_z
        new_lnw = current_state[1] + f_lnw * dt
        current_time += dt
        if current_time >= ts_list[next_output_idx] - 1e-08:
            w_next = torch.exp(new_lnw)
            r = w_next / w_prev
            mask_split = (r >= 1).squeeze()
            mask_extinct = ~mask_split
            if mask_split.any():
                r_split = r[mask_split]
                new_z_split = new_z[mask_split]
                new_lnw_split = new_lnw[mask_split]
                r_floor = torch.floor(r_split)
                r_frac = r_split - r_floor
                rand_frac = torch.rand_like(r_frac)
                m_j = r_floor.int().squeeze() + (rand_frac < r_frac).int().squeeze()
                valid_mask = m_j > 0
                m_j = m_j[valid_mask]
                if m_j.numel() > 0:
                    repeated_z = torch.repeat_interleave(new_z_split[valid_mask], m_j, dim=0)
                    repeated_lnw = torch.repeat_interleave(new_lnw_split[valid_mask], m_j, dim=0)
                    noise = torch.normal(0, noise_std, size=repeated_z.shape, device=device)
                    split_z = repeated_z + noise
                    split_lnw = repeated_lnw
                else:
                    split_z = torch.empty(0, new_z.shape[1], device=device)
                    split_lnw = torch.empty(0, 1, device=device)
            else:
                split_z = torch.empty(0, new_z.shape[1], device=device)
                split_lnw = torch.empty(0, 1, device=device)
            if mask_extinct.any():
                r_extinct = r[mask_extinct]
                new_z_extinct = new_z[mask_extinct]
                new_lnw_extinct = new_lnw[mask_extinct]
                rand_keep = torch.rand_like(r_extinct)
                keep_mask = rand_keep < r_extinct
                if keep_mask.dim() > 1 and keep_mask.shape[-1] == 1:
                    keep_mask = keep_mask.squeeze(-1)
                if keep_mask.any():
                    extinct_z = new_z_extinct[keep_mask]
                    extinct_lnw = new_lnw_extinct[keep_mask]
                else:
                    extinct_z = torch.empty(0, new_z.shape[1], device=device)
                    extinct_lnw = torch.empty(0, 1, device=device)
            else:
                extinct_z = torch.empty(0, new_z.shape[1], device=device)
                extinct_lnw = torch.empty(0, 1, device=device)
            if split_z.shape[0] > 0 or extinct_z.shape[0] > 0:
                new_z = torch.cat([split_z, extinct_z], dim=0)
                new_lnw = torch.cat([split_lnw, extinct_lnw], dim=0)
                new_lnw = torch.log(torch.ones(new_z.shape[0], 1, device=device) / initial_state[0].shape[0])
            else:
                new_z = torch.empty(0, current_state[0].shape[1], device=device)
                new_lnw = torch.empty(0, 1, device=device)
            current_state = (new_z, new_lnw)
            output_states.append(current_state)
            next_output_idx += 1
            w_prev = torch.exp(new_lnw)
            if next_output_idx >= len(ts_list):
                break
        else:
            current_state = (new_z, new_lnw)
    while len(output_states) < len(ts_list):
        output_states.append(current_state)
    traj_z = [state[0] for state in output_states]
    traj_lnw = [state[1] for state in output_states]
    return (traj_z, traj_lnw)
