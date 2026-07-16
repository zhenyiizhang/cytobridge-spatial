#!/usr/bin/env python3
"""
Downstream Analysis and Visualization with Trained Model
=========================================================
Generates all visualizations using the trained CytoBridge model on Zebrafish data.
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '/lustre/home/2501111653/CytoBridge-ST-package/CytoBridge')

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
from anndata import AnnData

# Paths
MODEL_DIR = '/lustre/home/2501111653/CytoBridge-ST-package/results/zebrafish_training'
OUTPUT_DIR = f'{MODEL_DIR}/plots'
H5AD_PATH = '/lustre/home/2501111653/CytoBridge-ST-1104/spatial_data/spatial_sixtime_slice_stereoseq.h5ad'
ALIGNED_CSV = f'{MODEL_DIR}/zebrafish_aligned.csv'
TRAINED_ADATA = f'{MODEL_DIR}/adata.h5ad'

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*60)
print("Downstream Analysis with Trained CytoBridge Model")
print("="*60)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

# ============================================================
# Load Data and Model
# ============================================================
print("\n[1] Loading data and model...")

# Load aligned data
df_aligned = pd.read_csv(ALIGNED_CSV)
print(f"  - Aligned CSV: {df_aligned.shape}")

# Load trained adata
adata_trained = sc.read_h5ad(TRAINED_ADATA)
print(f"  - Trained adata: {adata_trained.shape}")

# Load original data for annotations
adata_orig = sc.read_h5ad(H5AD_PATH)
print(f"  - Original adata: {adata_orig.shape}")

# Parse time points
def parse_time(t):
    if isinstance(t, str) and 'hpf' in t:
        return float(t.replace('hpf', ''))
    return float(t)

time_points = sorted(adata_orig.obs['time'].unique(), key=parse_time)
print(f"  - Time points: {time_points}")

# Get unique times from aligned data
unique_times = sorted(df_aligned['samples'].unique())
print(f"  - Aligned time indices: {unique_times}")

# ============================================================
# Plot 1: Training Summary
# ============================================================
print("\n[2] Generating training summary plot...")

# Load losses from checkpoints if available
stages = ['Pretrain', 'Refine', 'Init_interaction', 'Finetune']
stage_losses = {}

for stage in stages:
    ckpt_path = f'{MODEL_DIR}/{stage}/last_model.pth'
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu')
        if 'loss' in ckpt:
            stage_losses[stage] = ckpt['loss']

if stage_losses:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(stage_losses.keys(), stage_losses.values(), color='steelblue')
    ax.set_xlabel('Training Stage')
    ax.set_ylabel('Final Loss')
    ax.set_title('Training Loss by Stage')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/01_training_stages.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: 01_training_stages.png")

# ============================================================
# Plot 2: Spatial Distribution per Time (from aligned data)
# ============================================================
print("\n[3] Generating spatial distribution plots...")

n_times = len(unique_times)
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, t_val in enumerate(unique_times[:6]):
    subset = df_aligned[df_aligned['samples'] == t_val]
    ax = axes[idx]
    ax.scatter(subset['x1'], subset['x2'], s=0.5, alpha=0.5, c='steelblue')
    ax.set_title(f'{time_points[int(t_val)]} (n={len(subset)})')
    ax.set_xlabel('Spatial X')
    ax.set_ylabel('Spatial Y')
    ax.set_aspect('equal', adjustable='datalim')

for idx in range(len(unique_times), 6):
    axes[idx].axis('off')

plt.suptitle('Aligned Spatial Coordinates', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/02_spatial_aligned.png', dpi=200, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved: 02_spatial_aligned.png")

# ============================================================
# Plot 3: Cell Type Distribution Over Time
# ============================================================
print("\n[4] Generating cell type analysis...")

annot_col = 'bin_annotation'
time_col = 'time'

# Annotation proportions
fig, ax = plt.subplots(figsize=(12, 6))
annot_props = adata_orig.obs.groupby([time_col, annot_col]).size().unstack(fill_value=0)
annot_props = annot_props.loc[time_points]
annot_props = annot_props.div(annot_props.sum(axis=1), axis=0)

annot_props.plot(kind='bar', stacked=True, ax=ax, colormap='tab20')
ax.set_xlabel('Time Point')
ax.set_ylabel('Proportion')
ax.set_title('Cell Type Proportions Over Time')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/03_celltype_proportions.png', dpi=200, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved: 03_celltype_proportions.png")

# ============================================================
# Plot 4: Load Model and Compute Velocities
# ============================================================
print("\n[5] Computing velocities with trained model...")

try:
    import CytoBridge as cb
    from CytoBridge.tl.core.models import DynamicalModel
    from CytoBridge.utils.config import load_config
    
    # Load config
    config = load_config(f'{MODEL_DIR}/config.yaml')
    
    # Get dim
    dim = df_aligned.shape[1] - 1  # Exclude 'samples'
    
    # Create model
    model = DynamicalModel(dim, config['model'])
    
    # Load trained weights from Finetune
    ckpt_path = f'{MODEL_DIR}/Finetune/last_model.pth'
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    model.eval()
    print(f"  - Model loaded from {ckpt_path}")
    
    # Compute velocities for each time point
    from CytoBridge.tl import compute_velocity_components
    
    velocity_data = {}
    for t_idx, t_val in enumerate(unique_times[:5]):  # First 5 time points
        subset = df_aligned[df_aligned['samples'] == t_val]
        feature_cols = [c for c in subset.columns if c != 'samples']
        data = subset[feature_cols].values.astype(np.float32)
        
        # Sample if too large
        if len(data) > 3000:
            sample_idx = np.random.choice(len(data), 3000, replace=False)
            data = data[sample_idx]
        
        components = compute_velocity_components(
            data=data,
            time_value=float(t_val) / (len(unique_times) - 1),  # Normalize to [0,1]
            model=model,
            device=device,
            spatial_dim=2,
        )
        velocity_data[t_val] = {
            'data': data,
            'velocity': components.get('velocity', components.get('full', np.zeros_like(data))),
            'drift': components.get('drift', components.get('full', np.zeros_like(data))),
        }
        print(f"    - t={t_val}: computed velocities for {len(data)} cells")
    
    # Plot velocity fields
    n_plots = min(4, len(velocity_data))
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    axes = axes.flatten()
    
    for idx, (t_val, vdata) in enumerate(list(velocity_data.items())[:4]):
        ax = axes[idx]
        data = vdata['data']
        vel = vdata['drift'][:, :2]  # Spatial dims
        
        # Subsample for visualization
        n_show = min(500, len(data))
        show_idx = np.random.choice(len(data), n_show, replace=False)
        
        # Scatter
        ax.scatter(data[show_idx, 0], data[show_idx, 1], s=1, alpha=0.3, c='gray')
        
        # Quiver
        scale = np.percentile(np.linalg.norm(vel[show_idx], axis=1), 95)
        if scale > 0:
            ax.quiver(data[show_idx, 0], data[show_idx, 1], 
                     vel[show_idx, 0], vel[show_idx, 1],
                     scale=scale*10, scale_units='xy', alpha=0.7, color='blue', width=0.003)
        
        ax.set_title(f'{time_points[int(t_val)]}')
        ax.set_xlabel('Spatial X')
        ax.set_ylabel('Spatial Y')
        ax.set_aspect('equal', adjustable='datalim')
    
    plt.suptitle('Velocity Fields (Spatial)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/04_velocity_fields.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: 04_velocity_fields.png")

except Exception as e:
    print(f"  - Velocity computation error: {e}")
    import traceback
    traceback.print_exc()

# ============================================================
# Plot 5: Growth Rates
# ============================================================
print("\n[6] Computing and plotting growth rates...")

try:
    from CytoBridge.pl import plot_g_values
    
    # Get growth rates from model
    growth_data = {}
    for t_idx, t_val in enumerate(unique_times[:5]):
        subset = df_aligned[df_aligned['samples'] == t_val]
        feature_cols = [c for c in subset.columns if c != 'samples']
        data = subset[feature_cols].values.astype(np.float32)
        
        # Sample if needed
        if len(data) > 2000:
            sample_idx = np.random.choice(len(data), 2000, replace=False)
            data = data[sample_idx]
        
        with torch.no_grad():
            x_tensor = torch.tensor(data, device=device)
            t_tensor = torch.ones(len(data), 1, device=device) * (t_val / (len(unique_times) - 1))
            g = model.growth(x_tensor, t_tensor).cpu().numpy().flatten()
        
        growth_data[t_val] = {'data': data, 'g': g}
        print(f"    - t={t_val}: mean g = {np.mean(g):.4f}, std = {np.std(g):.4f}")
    
    # Plot growth rates
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, (t_val, gdata) in enumerate(growth_data.items()):
        ax = axes[idx]
        data = gdata['data']
        g = gdata['g']
        
        # Clip extreme values for visualization
        g_clipped = np.clip(g, np.percentile(g, 5), np.percentile(g, 95))
        
        scatter = ax.scatter(data[:, 0], data[:, 1], s=1, c=g_clipped, cmap='RdBu_r', alpha=0.7)
        ax.set_title(f'{time_points[int(t_val)]}\nmean g = {np.mean(g):.3f}')
        ax.set_xlabel('Spatial X')
        ax.set_ylabel('Spatial Y')
        ax.set_aspect('equal', adjustable='datalim')
        plt.colorbar(scatter, ax=ax, label='Growth Rate')
    
    axes[-1].axis('off')
    plt.suptitle('Growth Rates per Time Point', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/05_growth_rates.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: 05_growth_rates.png")

except Exception as e:
    print(f"  - Growth rate error: {e}")
    import traceback
    traceback.print_exc()

# ============================================================
# Plot 6: Sankey Diagram
# ============================================================
print("\n[7] Creating Sankey diagram...")

try:
    from CytoBridge.pl import plot_sankey
    
    # Get cell type labels per time point
    predicted_labels_list = []
    for t in time_points:
        mask = adata_orig.obs['time'] == t
        labels = adata_orig.obs.loc[mask, 'bin_annotation'].values.tolist()
        predicted_labels_list.append(labels)
    
    # Create Sankey
    fig = plot_sankey(
        predicted_labels_list,
        out_html=f'{OUTPUT_DIR}/06_sankey.html',
        time_keys=time_points,
        show_time_axis=True,
        title='Cell Type Lineage Flow',
    )
    print(f"  ✓ Saved: 06_sankey.html")

except Exception as e:
    print(f"  - Sankey error: {e}")

# ============================================================
# Plot 7: UMAP Visualization
# ============================================================
print("\n[8] Creating UMAP visualization...")

try:
    # Create adata from aligned data
    feature_cols = [c for c in df_aligned.columns if c != 'samples']
    X = df_aligned[feature_cols].values
    
    adata_aligned = AnnData(X)
    adata_aligned.obs['time_idx'] = df_aligned['samples'].values
    adata_aligned.obs['time_label'] = [time_points[int(t)] for t in df_aligned['samples'].values]
    
    # Add spatial
    adata_aligned.obsm['spatial'] = X[:, :2]
    
    # Run PCA and UMAP
    sc.pp.pca(adata_aligned, n_comps=30)
    sc.pp.neighbors(adata_aligned, n_neighbors=15)
    sc.tl.umap(adata_aligned)
    
    # Plot UMAP
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Time coloring
    ax = axes[0]
    time_colors = plt.cm.viridis(np.linspace(0, 1, len(time_points)))
    time_color_map = {t: time_colors[i] for i, t in enumerate(time_points)}
    colors = [time_color_map[t] for t in adata_aligned.obs['time_label']]
    ax.scatter(adata_aligned.obsm['X_umap'][:, 0], adata_aligned.obsm['X_umap'][:, 1], 
               c=colors, s=0.5, alpha=0.5)
    ax.set_xlabel('UMAP1')
    ax.set_ylabel('UMAP2')
    ax.set_title('UMAP - Colored by Time')
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=time_color_map[t], 
                          markersize=8, label=str(t)) for t in time_points]
    ax.legend(handles=handles, title='Time', fontsize=8)
    
    # PCA plot
    ax = axes[1]
    ax.scatter(adata_aligned.obsm['X_pca'][:, 0], adata_aligned.obsm['X_pca'][:, 1], 
               c=colors, s=0.5, alpha=0.5)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('PCA - Colored by Time')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/07_umap_pca.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: 07_umap_pca.png")

except Exception as e:
    print(f"  - UMAP error: {e}")
    import traceback
    traceback.print_exc()

# ============================================================
# Summary
# ============================================================
print("\n" + "="*60)
print("Downstream Analysis Complete!")
print("="*60)

print(f"\nOutput directory: {OUTPUT_DIR}")
print(f"\nGenerated plots:")
for f in sorted(os.listdir(OUTPUT_DIR)):
    fpath = os.path.join(OUTPUT_DIR, f)
    size = os.path.getsize(fpath) / 1024
    print(f"  - {f} ({size:.1f} KB)")

print("\n" + "="*60)
