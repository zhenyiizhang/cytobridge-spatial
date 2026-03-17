#!/usr/bin/env python3
"""
Complete Downstream Analysis - CytoBridge Zebrafish
====================================================
Runs ALL downstream analysis and generates ALL visualizations.
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
import traceback

# Paths
MODEL_DIR = '/lustre/home/2501111653/CytoBridge-ST-package/results/zebrafish_training'
OUTPUT_DIR = f'{MODEL_DIR}/plots'
H5AD_PATH = '/lustre/home/2501111653/CytoBridge-ST-1104/spatial_data/spatial_sixtime_slice_stereoseq.h5ad'
ALIGNED_CSV = f'{MODEL_DIR}/zebrafish_aligned.csv'
TRAINED_ADATA = f'{MODEL_DIR}/adata.h5ad'

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*70)
print("Complete Downstream Analysis - CytoBridge Zebrafish")
print("="*70)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

# ============================================================
# Load Data and Model
# ============================================================
print("\n[1/10] Loading data and model...")

df_aligned = pd.read_csv(ALIGNED_CSV)
print(f"  - Aligned CSV: {df_aligned.shape}")

adata_trained = sc.read_h5ad(TRAINED_ADATA)
print(f"  - Trained adata: {adata_trained.shape}")

adata_orig = sc.read_h5ad(H5AD_PATH)
print(f"  - Original adata: {adata_orig.shape}")

def parse_time(t):
    if isinstance(t, str) and 'hpf' in t:
        return float(t.replace('hpf', ''))
    return float(t)

time_points = sorted(adata_orig.obs['time'].unique(), key=parse_time)
unique_times = sorted(df_aligned['samples'].unique())
print(f"  - Time points: {time_points}")

# Load model
import CytoBridge as cb
from CytoBridge.tl.core.models import DynamicalModel
from CytoBridge.utils.config import load_config

config = load_config(f'{MODEL_DIR}/config.yaml')
dim = df_aligned.shape[1] - 1

model = DynamicalModel(dim, config['model'])
ckpt = torch.load(f'{MODEL_DIR}/Finetune/last_model.pth', map_location=device)
# Handle both wrapped and direct state dict
if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
    model.load_state_dict(ckpt['model_state_dict'])
else:
    model.load_state_dict(ckpt)
model = model.to(device)
model.eval()
print(f"  ✓ Model loaded")

# ============================================================
# Plot 1: Spatial Distribution per Time Point
# ============================================================
print("\n[2/10] Plotting spatial distribution...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

# Get annotations for coloring
annot_col = 'bin_annotation'
unique_annots = sorted(adata_orig.obs[annot_col].unique())
cmap = plt.cm.tab20(np.linspace(0, 1, len(unique_annots)))
annot_color = {a: cmap[i] for i, a in enumerate(unique_annots)}

for idx, t_val in enumerate(unique_times[:6]):
    subset = df_aligned[df_aligned['samples'] == t_val]
    t_label = time_points[int(t_val)]
    
    # Get annotations
    orig_mask = adata_orig.obs['time'] == t_label
    annots = adata_orig.obs.loc[orig_mask, annot_col].values[:len(subset)]
    if len(annots) < len(subset):
        annots = np.concatenate([annots, [annots[0]] * (len(subset) - len(annots))])
    
    colors = [annot_color.get(a, 'gray') for a in annots]
    
    ax = axes[idx]
    ax.scatter(subset['x1'].values, subset['x2'].values, s=0.5, c=colors, alpha=0.7)
    ax.set_title(f'{t_label} (n={len(subset)})')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_aspect('equal', adjustable='datalim')

plt.suptitle('Spatial Distribution by Cell Type', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/01_spatial_celltype.png', dpi=200, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved: 01_spatial_celltype.png")

# ============================================================
# Plot 2: Cell Type Proportions Over Time
# ============================================================
print("\n[3/10] Plotting cell type proportions...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Stacked bar
ax = axes[0]
annot_counts = adata_orig.obs.groupby(['time', annot_col]).size().unstack(fill_value=0)
annot_counts = annot_counts.loc[time_points]
annot_counts.plot(kind='bar', stacked=True, ax=ax, colormap='tab20')
ax.set_xlabel('Time Point')
ax.set_ylabel('Cell Count')
ax.set_title('Cell Type Distribution')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)
ax.tick_params(axis='x', rotation=45)

# Heatmap
ax = axes[1]
annot_props = annot_counts.div(annot_counts.sum(axis=1), axis=0)
sns.heatmap(annot_props.T, cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Proportion'})
ax.set_xlabel('Time Point')
ax.set_ylabel('Cell Type')
ax.set_title('Cell Type Proportions')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/02_celltype_analysis.png', dpi=200, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved: 02_celltype_analysis.png")

# ============================================================
# Plot 3: Velocity Fields
# ============================================================
print("\n[4/10] Computing and plotting velocity fields...")

try:
    from CytoBridge.tl import compute_velocity_components
    
    velocity_data = {}
    for t_idx, t_val in enumerate(unique_times[:5]):
        subset = df_aligned[df_aligned['samples'] == t_val]
        feature_cols = [c for c in subset.columns if c != 'samples']
        data = subset[feature_cols].values.astype(np.float32)
        
        if len(data) > 2000:
            sample_idx = np.random.choice(len(data), 2000, replace=False)
            data = data[sample_idx]
        
        t_norm = t_val / (len(unique_times) - 1)
        components = compute_velocity_components(
            data=data, time_value=t_norm, model=model,
            device=device, spatial_dim=2,
        )
        velocity_data[t_val] = {'data': data, 'drift': components.get('full', np.zeros_like(data))}
        print(f"    - t={time_points[int(t_val)]}: {len(data)} cells processed")
    
    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, (t_val, vdata) in enumerate(velocity_data.items()):
        ax = axes[idx]
        data = vdata['data']
        vel = vdata['drift'][:, :2]
        
        n_show = min(800, len(data))
        show_idx = np.random.choice(len(data), n_show, replace=False)
        
        ax.scatter(data[show_idx, 0], data[show_idx, 1], s=1, alpha=0.3, c='gray')
        
        scale = np.percentile(np.abs(vel[show_idx]), 95)
        if scale > 0:
            ax.quiver(data[show_idx, 0], data[show_idx, 1],
                     vel[show_idx, 0], vel[show_idx, 1],
                     scale=scale*15, scale_units='xy', alpha=0.8, color='blue', width=0.004)
        
        ax.set_title(f'{time_points[int(t_val)]}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_aspect('equal', adjustable='datalim')
    
    axes[-1].axis('off')
    plt.suptitle('Velocity Fields (Spatial)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/03_velocity_fields.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: 03_velocity_fields.png")
except Exception as e:
    print(f"  ✗ Velocity error: {e}")

# ============================================================
# Plot 4: Growth Rates
# ============================================================
print("\n[5/10] Computing and plotting growth rates...")

try:
    growth_data = {}
    for t_idx, t_val in enumerate(unique_times[:5]):
        subset = df_aligned[df_aligned['samples'] == t_val]
        feature_cols = [c for c in subset.columns if c != 'samples']
        data = subset[feature_cols].values.astype(np.float32)
        
        if len(data) > 2000:
            sample_idx = np.random.choice(len(data), 2000, replace=False)
            data = data[sample_idx]
        
        with torch.no_grad():
            x_t = torch.tensor(data, device=device)
            t_t = torch.ones(len(data), 1, device=device) * (t_val / (len(unique_times) - 1))
            g = model.growth(x_t, t_t).cpu().numpy().flatten()
        
        growth_data[t_val] = {'data': data, 'g': g}
        print(f"    - t={time_points[int(t_val)]}: mean g={np.mean(g):.4f}")
    
    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, (t_val, gdata) in enumerate(growth_data.items()):
        ax = axes[idx]
        data = gdata['data']
        g = gdata['g']
        g_clip = np.clip(g, np.percentile(g, 5), np.percentile(g, 95))
        
        scatter = ax.scatter(data[:, 0], data[:, 1], s=1, c=g_clip, cmap='RdBu_r', alpha=0.8)
        ax.set_title(f'{time_points[int(t_val)]}\nmean g={np.mean(g):.3f}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_aspect('equal', adjustable='datalim')
        plt.colorbar(scatter, ax=ax, label='g', shrink=0.7)
    
    axes[-1].axis('off')
    plt.suptitle('Growth Rates per Time Point', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/04_growth_rates.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: 04_growth_rates.png")
except Exception as e:
    print(f"  ✗ Growth rate error: {e}")

# ============================================================
# Plot 5: SDE Trajectory Simulation
# ============================================================
print("\n[6/10] Running SDE trajectory simulation...")

try:
    from CytoBridge.tl import simulate_sde_points
    
    ts_points = np.linspace(0, 1, len(unique_times))
    t0 = sorted(adata.obs['time_point_processed'].unique())[0]
    n_t0 = int((adata.obs['time_point_processed'] == t0).sum())
    
    sde_points, weights = simulate_sde_points(
        adata=adata,
        model=model,
        dim=dim,
        time_index=0,
        n_samples=min(500, n_t0),
        ts_points=ts_points,
        dt=0.02,
        sigma=0.03,
        device=device,
        time_key='time_point_processed',
        obsm_key='X_latent',
        spatial_key='spatial_aligned',
        concat_spatial=True,
    )
    
    print(f"  - SDE simulation complete: {sde_points.shape}")
    
    # Plot SDE vs Real
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, t_val in enumerate(unique_times[:6]):
        ax = axes[idx]
        
        # Real data
        real = df_aligned[df_aligned['samples'] == t_val]
        ax.scatter(real['x1'], real['x2'], s=0.5, alpha=0.3, c='blue', label='Real')
        
        # SDE prediction
        if idx < len(sde_points):
            sde = sde_points[idx]
            ax.scatter(sde[:, 0], sde[:, 1], s=1, alpha=0.5, c='red', label='SDE')
        
        ax.set_title(f'{time_points[int(t_val)]}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_aspect('equal', adjustable='datalim')
        if idx == 0:
            ax.legend(markerscale=5)
    
    plt.suptitle('SDE Prediction vs Real Data', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/05_sde_simulation.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: 05_sde_simulation.png")
except Exception as e:
    print(f"  ✗ SDE simulation error: {e}")
    traceback.print_exc()

# ============================================================
# Plot 6: Sankey Diagram
# ============================================================
print("\n[7/10] Creating Sankey diagram...")

try:
    from CytoBridge.pl import plot_sankey
    
    predicted_labels_list = []
    for t in time_points:
        mask = adata_orig.obs['time'] == t
        labels = adata_orig.obs.loc[mask, 'bin_annotation'].values.tolist()
        predicted_labels_list.append(labels)
    
    fig = plot_sankey(
        predicted_labels_list,
        out_html=f'{OUTPUT_DIR}/06_sankey.html',
        time_keys=time_points,
        show_time_axis=True,
        title='Cell Type Lineage Flow',
    )
    print(f"  ✓ Saved: 06_sankey.html")
except Exception as e:
    print(f"  ✗ Sankey error: {e}")

# ============================================================
# Plot 7: UMAP and PCA
# ============================================================
print("\n[8/10] Creating UMAP/PCA visualizations...")

try:
    feature_cols = [c for c in df_aligned.columns if c != 'samples']
    X = df_aligned[feature_cols].values
    
    adata_aligned = AnnData(X)
    adata_aligned.obs['time_idx'] = df_aligned['samples'].values.astype(int)
    adata_aligned.obs['time_label'] = [time_points[int(t)] for t in df_aligned['samples'].values]
    
    # Subsample for UMAP
    if len(adata_aligned) > 10000:
        idx = np.random.choice(len(adata_aligned), 10000, replace=False)
        adata_sub = adata_aligned[idx].copy()
    else:
        adata_sub = adata_aligned.copy()
    
    sc.pp.pca(adata_sub, n_comps=30)
    sc.pp.neighbors(adata_sub, n_neighbors=15)
    sc.tl.umap(adata_sub)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    time_colors = plt.cm.viridis(np.linspace(0, 1, len(time_points)))
    time_color_map = {t: time_colors[i] for i, t in enumerate(time_points)}
    colors = [time_color_map[t] for t in adata_sub.obs['time_label']]
    
    ax = axes[0]
    ax.scatter(adata_sub.obsm['X_umap'][:, 0], adata_sub.obsm['X_umap'][:, 1], c=colors, s=0.5, alpha=0.6)
    ax.set_xlabel('UMAP1')
    ax.set_ylabel('UMAP2')
    ax.set_title('UMAP - Colored by Time')
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=time_color_map[t], 
                          markersize=8, label=str(t)) for t in time_points]
    ax.legend(handles=handles, fontsize=8)
    
    ax = axes[1]
    ax.scatter(adata_sub.obsm['X_pca'][:, 0], adata_sub.obsm['X_pca'][:, 1], c=colors, s=0.5, alpha=0.6)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('PCA - Colored by Time')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/07_umap_pca.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: 07_umap_pca.png")
except Exception as e:
    print(f"  ✗ UMAP/PCA error: {e}")

# ============================================================
# Plot 8: Gene Expression Trends
# ============================================================
print("\n[9/10] Plotting gene expression trends...")

try:
    # Get top variable genes
    gene_var = np.var(adata_orig.X.toarray() if hasattr(adata_orig.X, 'toarray') else adata_orig.X, axis=0)
    top_idx = np.argsort(gene_var)[-12:]
    top_genes = adata_orig.var_names[top_idx]
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, gene in enumerate(top_genes):
        ax = axes[idx]
        gene_means = []
        gene_stds = []
        for t in time_points:
            mask = adata_orig.obs['time'] == t
            expr = adata_orig[:, gene].X[mask]
            if hasattr(expr, 'toarray'):
                expr = expr.toarray()
            gene_means.append(np.mean(expr))
            gene_stds.append(np.std(expr))
        
        ax.errorbar(range(len(time_points)), gene_means, yerr=gene_stds, 
                   marker='o', capsize=3, linewidth=2, markersize=6)
        ax.set_xticks(range(len(time_points)))
        ax.set_xticklabels(time_points, rotation=45, ha='right')
        ax.set_xlabel('Time')
        ax.set_ylabel('Expression')
        ax.set_title(gene)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Gene Expression Trends (Top Variable Genes)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/08_gene_trends.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: 08_gene_trends.png")
except Exception as e:
    print(f"  ✗ Gene trends error: {e}")

# ============================================================
# Plot 9: Cell Count Over Time
# ============================================================
print("\n[10/10] Plotting cell counts...")

try:
    fig, ax = plt.subplots(figsize=(10, 6))
    
    counts = [adata_orig.obs['time'].value_counts()[t] for t in time_points]
    bars = ax.bar(range(len(time_points)), counts, color='steelblue', edgecolor='black')
    
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50, 
                str(count), ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xticks(range(len(time_points)))
    ax.set_xticklabels(time_points)
    ax.set_xlabel('Time Point', fontsize=12)
    ax.set_ylabel('Cell Count', fontsize=12)
    ax.set_title('Cell Count Distribution Over Time', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/09_cell_counts.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: 09_cell_counts.png")
except Exception as e:
    print(f"  ✗ Cell count error: {e}")

# ============================================================
# Summary
# ============================================================
print("\n" + "="*70)
print("Downstream Analysis Complete!")
print("="*70)

print(f"\nOutput directory: {OUTPUT_DIR}")
print(f"\nGenerated files:")
total_size = 0
for f in sorted(os.listdir(OUTPUT_DIR)):
    fpath = os.path.join(OUTPUT_DIR, f)
    size = os.path.getsize(fpath) / 1024
    total_size += size
    print(f"  - {f} ({size:.1f} KB)")

print(f"\nTotal: {total_size/1024:.1f} MB")
print("\n" + "="*70)
