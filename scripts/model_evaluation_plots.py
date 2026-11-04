#!/usr/bin/env python3
"""
Model Evaluation Plots - CytoBridge Zebrafish
==============================================
Generates the critical model evaluation visualizations:
1. SDE/ODE generated vs real point comparison
2. Velocity flow/streamlines
3. Interpolated trajectories
4. Growth rate spatial maps
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '/lustre/home/2501111653/CytoBridge-ST-package/CytoBridge')

import numpy as np
import pandas as pd
import torch
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm
import scanpy as sc
from scipy.ndimage import gaussian_filter

# Paths
MODEL_DIR = '/lustre/home/2501111653/CytoBridge-ST-package/results/zebrafish_training'
OUTPUT_DIR = f'{MODEL_DIR}/plots'
H5AD_PATH = '/lustre/home/2501111653/CytoBridge-ST-1104/spatial_data/spatial_sixtime_slice_stereoseq.h5ad'
ALIGNED_CSV = f'{MODEL_DIR}/zebrafish_aligned.csv'
TRAINED_ADATA = f'{MODEL_DIR}/adata.h5ad'

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*70)
print("Model Evaluation Plots - CytoBridge Zebrafish")
print("="*70)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

# ============================================================
# Load Data and Model
# ============================================================
print("\n[1] Loading data and model...")

df_aligned = pd.read_csv(ALIGNED_CSV)
adata_trained = sc.read_h5ad(TRAINED_ADATA)
adata_orig = sc.read_h5ad(H5AD_PATH)

def parse_time(t):
    if isinstance(t, str) and 'hpf' in t:
        return float(t.replace('hpf', ''))
    return float(t)

time_points = sorted(adata_orig.obs['time'].unique(), key=parse_time)
unique_times = sorted(df_aligned['samples'].unique())
print(f"  - Time points: {time_points}")

# Load model
from CytoBridge.tl.core.models import DynamicalModel
from CytoBridge.utils.config import load_config

config = load_config(f'{MODEL_DIR}/config.yaml')
dim = df_aligned.shape[1] - 1

model = DynamicalModel(dim, config['model'])
ckpt = torch.load(f'{MODEL_DIR}/Finetune/last_model.pth', map_location=device)
model.load_state_dict(ckpt)
model = model.to(device)
model.eval()
print(f"  ✓ Model loaded (components: {model.components})")

# ============================================================
# Plot 1: SDE/ODE Generated vs Real Point Comparison
# ============================================================
print("\n[2] Generating SDE simulation vs real data comparison...")

try:
    # Get initial data (first time point)
    t0_idx = 0
    t0_data = df_aligned[df_aligned['samples'] == t0_idx]
    feature_cols = [c for c in t0_data.columns if c != 'samples']
    
    # Sample initial points
    n_samples = min(500, len(t0_data))
    sample_idx = np.random.choice(len(t0_data), n_samples, replace=False)
    x0 = t0_data[feature_cols].values[sample_idx].astype(np.float32)
    
    x0_t = torch.tensor(x0, device=device, dtype=torch.float32).requires_grad_(True)
    lnw0 = torch.log(torch.ones(n_samples, 1, device=device) / n_samples)
    
    # Define time points for simulation
    ts_list = [t / (len(unique_times) - 1) for t in unique_times]
    
    # SDE integration using the model
    sigma = 0.03
    dt = 0.02
    
    # Simple SDE class inline
    class SimpleSDE:
        def __init__(self, model, sigma):
            self.model = model
            self.sigma = sigma
            
        def f(self, t, y):
            """Drift term"""
            z, lnw = y
            t_expand = t.expand(z.shape[0], 1).to(dtype=z.dtype)
            vel_in = torch.cat([z, t_expand], dim=1)
            
            # Velocity
            drift_z = self.model.velocity_net(vel_in)
            
            # Score correction if available
            if hasattr(self.model, 'score_net') and self.model.score_net is not None:
                try:
                    _, gradients = self.model.compute_score(t, z)
                    drift_z = drift_z + gradients
                except:
                    pass
            
            # Growth
            if hasattr(self.model, 'growth_net') and self.model.growth_net is not None:
                drift_lnw = self.model.growth_net(vel_in)
            else:
                drift_lnw = torch.zeros_like(lnw)
            
            return (drift_z, drift_lnw)
        
        def g(self, t, y):
            """Diffusion term"""
            z, lnw = y
            return (torch.ones_like(z) * self.sigma, torch.zeros_like(lnw))
    
    sde = SimpleSDE(model, sigma)
    
    # Euler-Maruyama integration
    def euler_sde(sde, y0, dt, ts_list):
        y = y0
        t = 0.0
        out = []
        idx = 0
        
        while t <= ts_list[-1] + 1e-8:
            if t >= ts_list[idx] - 1e-8:
                out.append((y[0].clone(), y[1].clone()))
                idx += 1
                if idx >= len(ts_list):
                    break
            
            t_tensor = torch.tensor([t], dtype=torch.float32, device=y[0].device)
            f_z, f_lnw = sde.f(t_tensor, y)
            g_z, g_lnw = sde.g(t_tensor, y)
            
            noise_z = torch.randn_like(y[0]) * math.sqrt(dt)
            noise_lnw = torch.randn_like(y[1]) * math.sqrt(dt)
            
            y = (y[0] + f_z * dt + g_z * noise_z,
                 y[1] + f_lnw * dt + g_lnw * noise_lnw)
            t += dt
        
        while len(out) < len(ts_list):
            out.append(out[-1])
        
        return [o[0] for o in out], [o[1] for o in out]
    
    # Run SDE simulation
    with torch.no_grad():
        sde_points, sde_weights = euler_sde(sde, (x0_t, lnw0), dt, ts_list)
    
    # Convert to numpy
    sde_points_np = [p.cpu().numpy() for p in sde_points]
    
    # Plot: Generated vs Real
    n_plots = min(6, len(unique_times))
    fig, axes = plt.subplots(2, n_plots, figsize=(n_plots*5, 10))
    
    for idx, t_val in enumerate(unique_times[:n_plots]):
        # Real data
        real_data = df_aligned[df_aligned['samples'] == t_val]
        
        # Top row: Real data
        ax = axes[0, idx]
        ax.scatter(real_data['x1'], real_data['x2'], s=0.5, alpha=0.5, c='blue')
        ax.set_title(f'Real: {time_points[int(t_val)]}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_aspect('equal', adjustable='datalim')
        
        # Bottom row: Generated (SDE)
        ax = axes[1, idx]
        gen_data = sde_points_np[idx]
        ax.scatter(gen_data[:, 0], gen_data[:, 1], s=1, alpha=0.7, c='red')
        ax.set_title(f'SDE Generated (n={len(gen_data)})')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_aspect('equal', adjustable='datalim')
        
        # Match axis limits
        xlim = (min(real_data['x1'].min(), gen_data[:, 0].min()), 
                max(real_data['x1'].max(), gen_data[:, 0].max()))
        ylim = (min(real_data['x2'].min(), gen_data[:, 1].min()), 
                max(real_data['x2'].max(), gen_data[:, 1].max()))
        axes[0, idx].set_xlim(xlim)
        axes[0, idx].set_ylim(ylim)
        axes[1, idx].set_xlim(xlim)
        axes[1, idx].set_ylim(ylim)
    
    plt.suptitle('SDE Generated Points vs Real Data', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/eval_01_sde_vs_real.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: eval_01_sde_vs_real.png")
    
except Exception as e:
    print(f"  ✗ SDE simulation error: {e}")
    import traceback
    traceback.print_exc()

# ============================================================
# Plot 2: Velocity Streamlines/Flow
# ============================================================
print("\n[3] Generating velocity streamline plots...")

try:
    for t_idx, t_val in enumerate(unique_times[:4]):
        # Get data for this time point
        subset = df_aligned[df_aligned['samples'] == t_val]
        feature_cols = [c for c in subset.columns if c != 'samples']
        data = subset[feature_cols].values.astype(np.float32)
        
        if len(data) > 2000:
            sample_idx = np.random.choice(len(data), 2000, replace=False)
            data = data[sample_idx]
        
        # Compute velocity
        t_norm = t_val / (len(unique_times) - 1)
        with torch.no_grad():
            x_t = torch.tensor(data, device=device, dtype=torch.float32)
            t_expand = torch.ones(len(data), 1, device=device) * t_norm
            vel_in = torch.cat([x_t, t_expand], dim=1)
            velocity = model.velocity_net(vel_in).cpu().numpy()
        
        # Get spatial velocity (first 2 dimensions)
        vel_spatial = velocity[:, :2]
        
        # Create figure with quiver and streamline
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # Left: Quiver plot
        ax = axes[0]
        ax.scatter(data[:, 0], data[:, 1], s=0.5, alpha=0.3, c='gray')
        
        # Subsample for quiver
        n_quiver = min(400, len(data))
        q_idx = np.random.choice(len(data), n_quiver, replace=False)
        
        vel_mag = np.linalg.norm(vel_spatial[q_idx], axis=1)
        colors = cm.viridis(Normalize()(vel_mag))
        
        ax.quiver(data[q_idx, 0], data[q_idx, 1], 
                 vel_spatial[q_idx, 0], vel_spatial[q_idx, 1],
                 color=colors, scale=np.percentile(vel_mag, 95)*15, 
                 scale_units='xy', alpha=0.8, width=0.004)
        
        ax.set_title(f'{time_points[int(t_val)]} - Velocity Arrows')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_aspect('equal', adjustable='datalim')
        
        # Right: Streamline-like visualization (grid-based)
        ax = axes[1]
        
        # Create a grid for streamlines
        x_min, x_max = data[:, 0].min(), data[:, 0].max()
        y_min, y_max = data[:, 1].min(), data[:, 1].max()
        
        grid_size = 30
        x_grid = np.linspace(x_min, x_max, grid_size)
        y_grid = np.linspace(y_min, y_max, grid_size)
        X, Y = np.meshgrid(x_grid, y_grid)
        
        # Interpolate velocity to grid
        from scipy.interpolate import griddata
        U = griddata((data[:, 0], data[:, 1]), vel_spatial[:, 0], (X, Y), method='linear', fill_value=0)
        V = griddata((data[:, 0], data[:, 1]), vel_spatial[:, 1], (X, Y), method='linear', fill_value=0)
        
        # Smooth the velocity field
        U = gaussian_filter(U, sigma=1)
        V = gaussian_filter(V, sigma=1)
        
        # Background scatter
        ax.scatter(data[:, 0], data[:, 1], s=0.3, alpha=0.2, c='gray')
        
        # Streamlines
        speed = np.sqrt(U**2 + V**2)
        lw = 2 * speed / speed.max()
        strm = ax.streamplot(X, Y, U, V, color=speed, cmap='plasma', 
                            density=1.5, linewidth=lw, arrowsize=1)
        plt.colorbar(strm.lines, ax=ax, label='Speed')
        
        ax.set_title(f'{time_points[int(t_val)]} - Velocity Streamlines')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_aspect('equal', adjustable='datalim')
        
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/eval_02_velocity_stream_t{int(t_val)}.png', dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: eval_02_velocity_stream_t{int(t_val)}.png")

except Exception as e:
    print(f"  ✗ Velocity streamline error: {e}")
    import traceback
    traceback.print_exc()

# ============================================================
# Plot 3: Interpolated Trajectories
# ============================================================
print("\n[4] Generating interpolated trajectory plots...")

try:
    # Sample trajectories from initial time point
    t0_data = df_aligned[df_aligned['samples'] == 0]
    feature_cols = [c for c in t0_data.columns if c != 'samples']
    
    n_traj = 50
    sample_idx = np.random.choice(len(t0_data), n_traj, replace=False)
    x0 = t0_data[feature_cols].values[sample_idx].astype(np.float32)
    
    x0_t = torch.tensor(x0, device=device, dtype=torch.float32).requires_grad_(True)
    lnw0 = torch.log(torch.ones(n_traj, 1, device=device) / n_traj)
    
    # Fine time points for smooth trajectories
    n_interp = 100
    ts_fine = np.linspace(0, 1, n_interp)
    
    # ODE integration (deterministic)
    def euler_ode(model, y0, ts_list, dt=0.01):
        y = y0
        t = 0.0
        out = []
        idx = 0
        
        while t <= ts_list[-1] + 1e-8:
            if t >= ts_list[idx] - 1e-8:
                out.append(y[0].clone())
                idx += 1
                if idx >= len(ts_list):
                    break
            
            # Compute velocity at current state
            t_expand = torch.ones(y[0].shape[0], 1, device=device) * t
            vel_in = torch.cat([y[0], t_expand], dim=1)
            
            with torch.no_grad():
                f_z = model.velocity_net(vel_in)
                
                # Growth
                if hasattr(model, 'growth_net') and model.growth_net is not None:
                    f_lnw = model.growth_net(vel_in)
                else:
                    f_lnw = torch.zeros_like(y[1])
            
            y = (y[0] + f_z * dt, y[1] + f_lnw * dt)
            t += dt
        
        while len(out) < len(ts_list):
            out.append(out[-1])
        
        return torch.stack(out)  # [T, N, D]
    
    # Run ODE
    traj = euler_ode(model, (x0_t, lnw0), ts_fine)
    traj_np = traj.cpu().numpy()  # [T, N, D]
    
    # Plot trajectories
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Left: Trajectories in spatial space
    ax = axes[0]
    colors = cm.viridis(np.linspace(0, 1, n_traj))
    for i in range(n_traj):
        ax.plot(traj_np[:, i, 0], traj_np[:, i, 1], '-', color=colors[i], alpha=0.7, linewidth=0.8)
        ax.scatter(traj_np[0, i, 0], traj_np[0, i, 1], s=20, c='green', zorder=5, marker='o')
        ax.scatter(traj_np[-1, i, 0], traj_np[-1, i, 1], s=20, c='red', zorder=5, marker='x')
    
    ax.set_title('ODE Interpolated Trajectories (Spatial)')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_aspect('equal', adjustable='datalim')
    
    # Middle: Trajectories overlaid on real data
    ax = axes[1]
    # Background: all real data
    for t_val in unique_times:
        real = df_aligned[df_aligned['samples'] == t_val]
        ax.scatter(real['x1'], real['x2'], s=0.3, alpha=0.2, c='gray')
    
    # Trajectories
    for i in range(n_traj):
        ax.plot(traj_np[:, i, 0], traj_np[:, i, 1], '-', color='blue', alpha=0.5, linewidth=0.8)
    
    ax.set_title('Trajectories Over Real Data')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_aspect('equal', adjustable='datalim')
    
    # Right: Time evolution
    ax = axes[2]
    time_colors = cm.plasma(np.linspace(0, 1, n_interp))
    for t_i in range(0, n_interp, 5):  # Every 5th time point
        ax.scatter(traj_np[t_i, :, 0], traj_np[t_i, :, 1], s=5, c=[time_colors[t_i]], alpha=0.7)
    
    # Add colorbar
    sm = cm.ScalarMappable(cmap='plasma', norm=Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Normalized Time')
    
    ax.set_title('Trajectory Evolution Over Time')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_aspect('equal', adjustable='datalim')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/eval_03_trajectories.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: eval_03_trajectories.png")

except Exception as e:
    print(f"  ✗ Trajectory error: {e}")
    import traceback
    traceback.print_exc()

# ============================================================
# Plot 4: Growth Rate Spatial Maps
# ============================================================
print("\n[5] Generating growth rate spatial maps...")

try:
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, t_val in enumerate(unique_times[:6]):
        subset = df_aligned[df_aligned['samples'] == t_val]
        feature_cols = [c for c in subset.columns if c != 'samples']
        data = subset[feature_cols].values.astype(np.float32)
        
        if len(data) > 2500:
            sample_idx = np.random.choice(len(data), 2500, replace=False)
            data = data[sample_idx]
        
        t_norm = t_val / (len(unique_times) - 1)
        with torch.no_grad():
            x_t = torch.tensor(data, device=device, dtype=torch.float32)
            t_expand = torch.ones(len(data), 1, device=device) * t_norm
            vel_in = torch.cat([x_t, t_expand], dim=1)
            g = model.growth_net(vel_in).cpu().numpy().flatten()
        
        ax = axes[idx]
        g_clip = np.clip(g, np.percentile(g, 2), np.percentile(g, 98))
        
        scatter = ax.scatter(data[:, 0], data[:, 1], s=1, c=g_clip, cmap='RdBu_r', alpha=0.8)
        ax.set_title(f'{time_points[int(t_val)]}\nmean g = {np.mean(g):.3f}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_aspect('equal', adjustable='datalim')
        plt.colorbar(scatter, ax=ax, label='Growth', shrink=0.8)
    
    plt.suptitle('Growth Rate Spatial Distribution', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/eval_04_growth_maps.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: eval_04_growth_maps.png")

except Exception as e:
    print(f"  ✗ Growth rate error: {e}")
    import traceback
    traceback.print_exc()

# ============================================================
# Plot 5: Combined Summary Figure
# ============================================================
print("\n[6] Generating combined summary figure...")

try:
    fig = plt.figure(figsize=(20, 12))
    
    # Get sample data for plots
    t0_data = df_aligned[df_aligned['samples'] == 0]
    t_last = max(unique_times)
    tf_data = df_aligned[df_aligned['samples'] == t_last]
    feature_cols = [c for c in t0_data.columns if c != 'samples']
    
    # 1. Initial distribution
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.scatter(t0_data['x1'], t0_data['x2'], s=0.5, alpha=0.5, c='blue')
    ax1.set_title(f'Initial: {time_points[0]}')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_aspect('equal', adjustable='datalim')
    
    # 2. Final real distribution
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.scatter(tf_data['x1'], tf_data['x2'], s=0.5, alpha=0.5, c='green')
    ax2.set_title(f'Final Real: {time_points[-1]}')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_aspect('equal', adjustable='datalim')
    
    # 3. Final SDE generated
    ax3 = fig.add_subplot(2, 3, 3)
    final_gen = sde_points_np[-1] if 'sde_points_np' in dir() else None
    if final_gen is not None:
        ax3.scatter(final_gen[:, 0], final_gen[:, 1], s=1, alpha=0.7, c='red')
        ax3.set_title(f'Final SDE Generated (n={len(final_gen)})')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_aspect('equal', adjustable='datalim')
    
    # 4. Velocity field at middle time
    mid_t = len(unique_times) // 2
    mid_data = df_aligned[df_aligned['samples'] == mid_t]
    data = mid_data[feature_cols].values[:1000].astype(np.float32)
    
    t_norm = mid_t / (len(unique_times) - 1)
    with torch.no_grad():
        x_t = torch.tensor(data, device=device, dtype=torch.float32)
        t_expand = torch.ones(len(data), 1, device=device) * t_norm
        vel_in = torch.cat([x_t, t_expand], dim=1)
        velocity = model.velocity_net(vel_in).cpu().numpy()
    
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.scatter(data[:, 0], data[:, 1], s=0.5, alpha=0.3, c='gray')
    q_idx = np.random.choice(len(data), min(300, len(data)), replace=False)
    ax4.quiver(data[q_idx, 0], data[q_idx, 1], 
              velocity[q_idx, 0], velocity[q_idx, 1],
              scale=np.percentile(np.abs(velocity[:, :2]), 95)*15, 
              scale_units='xy', alpha=0.8, color='blue', width=0.004)
    ax4.set_title(f'Velocity Field: {time_points[mid_t]}')
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.set_aspect('equal', adjustable='datalim')
    
    # 5. Growth rate at middle time  
    with torch.no_grad():
        g = model.growth_net(vel_in).cpu().numpy().flatten()
    
    ax5 = fig.add_subplot(2, 3, 5)
    g_clip = np.clip(g, np.percentile(g, 5), np.percentile(g, 95))
    scatter = ax5.scatter(data[:, 0], data[:, 1], s=1, c=g_clip, cmap='RdBu_r', alpha=0.8)
    ax5.set_title(f'Growth Rate: {time_points[mid_t]}')
    ax5.set_xlabel('X')
    ax5.set_ylabel('Y')
    ax5.set_aspect('equal', adjustable='datalim')
    plt.colorbar(scatter, ax=ax5, label='g', shrink=0.8)
    
    # 6. Sample trajectories
    ax6 = fig.add_subplot(2, 3, 6)
    if 'traj_np' in dir():
        for i in range(min(30, traj_np.shape[1])):
            ax6.plot(traj_np[:, i, 0], traj_np[:, i, 1], '-', alpha=0.5, linewidth=0.8)
        ax6.scatter(traj_np[0, :30, 0], traj_np[0, :30, 1], s=20, c='green', marker='o', label='Start')
        ax6.scatter(traj_np[-1, :30, 0], traj_np[-1, :30, 1], s=20, c='red', marker='x', label='End')
        ax6.legend()
    ax6.set_title('Sample Trajectories')
    ax6.set_xlabel('X')
    ax6.set_ylabel('Y')
    ax6.set_aspect('equal', adjustable='datalim')
    
    plt.suptitle('CytoBridge Model Evaluation Summary - Zebrafish', fontsize=18, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/eval_00_summary.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: eval_00_summary.png")

except Exception as e:
    print(f"  ✗ Summary error: {e}")
    import traceback
    traceback.print_exc()

# ============================================================
# Summary
# ============================================================
print("\n" + "="*70)
print("Model Evaluation Plots Complete!")
print("="*70)

print(f"\nOutput directory: {OUTPUT_DIR}")
print(f"\nGenerated evaluation plots:")
for f in sorted(os.listdir(OUTPUT_DIR)):
    if f.startswith('eval_'):
        fpath = os.path.join(OUTPUT_DIR, f)
        size = os.path.getsize(fpath) / 1024
        print(f"  - {f} ({size:.1f} KB)")

print("\n" + "="*70)
