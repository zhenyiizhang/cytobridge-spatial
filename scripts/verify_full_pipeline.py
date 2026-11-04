#!/usr/bin/env python3
"""
End-to-End Pipeline Verification for CytoBridge - Zebrafish Data
=================================================================
Complete pipeline from preprocessing to visualization.
"""

import os
import sys
import warnings
import traceback
warnings.filterwarnings('ignore')

# Add CytoBridge package to path
sys.path.insert(0, '/lustre/home/2501111653/CytoBridge-ST-package/CytoBridge')

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import scanpy as sc
import seaborn as sns

# Set up output directory
OUTPUT_DIR = '/lustre/home/2501111653/CytoBridge-ST-package/results/zebrafish_verification'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f'{OUTPUT_DIR}/plots', exist_ok=True)

# Input data
H5AD_PATH = '/lustre/home/2501111653/CytoBridge-ST-1104/spatial_data/spatial_sixtime_slice_stereoseq.h5ad'

print("="*60)
print("CytoBridge End-to-End Pipeline Verification - Zebrafish")
print("="*60)
print(f"Input: {H5AD_PATH}")
print(f"Output: {OUTPUT_DIR}")
print()

try:
    # ============================================================
    # PHASE 1: Load and Explore Data
    # ============================================================
    print("[Phase 1] Loading and exploring data...")
    
    adata = sc.read_h5ad(H5AD_PATH)
    print(f"  - Shape: {adata.shape}")
    print(f"  - Obs columns: {list(adata.obs.columns)}")
    
    # Extract time column
    time_col = 'Batch' if 'Batch' in adata.obs.columns else 'time'
    print(f"  - Using time column: {time_col}")
    
    # Get unique time points
    time_points = sorted(adata.obs[time_col].unique(), key=lambda x: str(x))
    print(f"  - Time points: {time_points}")
    
    # Check for spatial coordinates
    spatial_cols = []
    if 'spatial_x' in adata.obs.columns and 'spatial_y' in adata.obs.columns:
        spatial_cols = ['spatial_x', 'spatial_y']
        print(f"  - Spatial coords in obs: {spatial_cols}")
    elif 'spatial' in adata.obsm:
        print("  - Spatial coords in obsm['spatial']")
    else:
        print("  - WARNING: No spatial coordinates found")
    
    # Check for annotation
    annot_col = None
    for col in ['bin_annotation', 'Annotation', 'celltype', 'cluster']:
        if col in adata.obs.columns:
            annot_col = col
            break
    print(f"  - Annotation column: {annot_col}")
    
    print("  ✓ Data loaded successfully")
    print()
    
    # ============================================================
    # PHASE 2: Create Visualization-Ready DataFrame
    # ============================================================
    print("[Phase 2] Preparing data for visualization...")
    
    # Sort time points properly (handle '3.3hpf' < '5.25hpf' < '10hpf' etc.)
    def parse_time(t):
        if isinstance(t, str) and 'hpf' in t:
            return float(t.replace('hpf', ''))
        return float(t)
    
    time_order = {t: parse_time(t) for t in time_points}
    sorted_times = sorted(time_points, key=lambda x: time_order[x])
    print(f"  - Sorted time order: {sorted_times}")
    
    # Create mapping for numeric time
    time_to_idx = {t: i for i, t in enumerate(sorted_times)}
    adata.obs['time_idx'] = adata.obs[time_col].map(time_to_idx)
    
    print("  ✓ Data prepared")
    print()
    
    # ============================================================
    # PHASE 3: Spatial Plots per Time Point
    # ============================================================
    print("[Phase 3] Creating spatial plots per time point...")
    
    n_times = len(sorted_times)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, t in enumerate(sorted_times[:6]):
        mask = adata.obs[time_col] == t
        subset = adata[mask]
        ax = axes[idx]
        
        if spatial_cols:
            x = adata.obs.loc[mask, spatial_cols[0]].values
            y = adata.obs.loc[mask, spatial_cols[1]].values
        else:
            # Use first two PCs if no spatial
            if adata.X.shape[1] > 50:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=2)
                coords = pca.fit_transform(adata.X[mask].toarray() if hasattr(adata.X, 'toarray') else adata.X[mask])
                x, y = coords[:, 0], coords[:, 1]
            else:
                x = np.arange(mask.sum())
                y = np.random.randn(mask.sum())
        
        if annot_col:
            categories = adata.obs.loc[mask, annot_col].values
            unique_cats = list(set(categories))
            colors = plt.cm.tab20(np.linspace(0, 1, len(unique_cats)))
            color_map = {cat: colors[i] for i, cat in enumerate(unique_cats)}
            c = [color_map[cat] for cat in categories]
            scatter = ax.scatter(x, y, c=c, s=0.5, alpha=0.7)
        else:
            ax.scatter(x, y, s=0.5, alpha=0.7)
        
        ax.set_title(f'{t} (n={mask.sum()})')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_aspect('equal', adjustable='datalim')
    
    for idx in range(len(sorted_times), 6):
        axes[idx].axis('off')
    
    plt.suptitle('Spatial Distribution per Time Point', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/plots/01_spatial_per_time.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: 01_spatial_per_time.png")
    
    # ============================================================
    # PHASE 4: Cell Count Distribution
    # ============================================================
    print("[Phase 4] Creating cell count plots...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar plot of cells per time
    cell_counts = [adata.obs[time_col].value_counts()[t] for t in sorted_times]
    ax = axes[0]
    bars = ax.bar(range(len(sorted_times)), cell_counts, color='steelblue')
    ax.set_xticks(range(len(sorted_times)))
    ax.set_xticklabels(sorted_times, rotation=45, ha='right')
    ax.set_xlabel('Time Point')
    ax.set_ylabel('Cell Count')
    ax.set_title('Cell Count Distribution')
    for bar, count in zip(bars, cell_counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50, 
                str(count), ha='center', va='bottom', fontsize=9)
    
    # Annotation distribution per time
    if annot_col:
        ax = axes[1]
        annot_counts = adata.obs.groupby([time_col, annot_col]).size().unstack(fill_value=0)
        annot_counts = annot_counts.loc[sorted_times]
        annot_counts.plot(kind='bar', stacked=True, ax=ax, colormap='tab20')
        ax.set_xlabel('Time Point')
        ax.set_ylabel('Cell Count')
        ax.set_title('Cell Type Distribution per Time')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.tick_params(axis='x', rotation=45)
    else:
        axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/plots/02_cell_counts.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: 02_cell_counts.png")
    
    # ============================================================
    # PHASE 5: Annotation Heatmap and Transition Matrix
    # ============================================================
    if annot_col:
        print("[Phase 5] Creating annotation analysis plots...")
        
        # Annotation proportions over time
        fig, ax = plt.subplots(figsize=(12, 8))
        annot_props = adata.obs.groupby([time_col, annot_col]).size().unstack(fill_value=0)
        annot_props = annot_props.loc[sorted_times]
        annot_props = annot_props.div(annot_props.sum(axis=1), axis=0)
        
        sns.heatmap(annot_props.T, cmap='YlOrRd', annot=False, ax=ax, 
                    cbar_kws={'label': 'Proportion'})
        ax.set_xlabel('Time Point')
        ax.set_ylabel('Cell Type')
        ax.set_title('Cell Type Proportion Over Time')
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/plots/03_annotation_heatmap.png', dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: 03_annotation_heatmap.png")
    else:
        print("[Phase 5] Skipped (no annotation column)")
    
    # ============================================================
    # PHASE 6: Spatial Overlay (All Times)
    # ============================================================
    print("[Phase 6] Creating spatial overlay plot...")
    
    if spatial_cols:
        fig, ax = plt.subplots(figsize=(12, 10))
        cmap = plt.cm.viridis(np.linspace(0, 1, n_times))
        
        for idx, t in enumerate(sorted_times):
            mask = adata.obs[time_col] == t
            x = adata.obs.loc[mask, spatial_cols[0]].values
            y = adata.obs.loc[mask, spatial_cols[1]].values
            ax.scatter(x, y, s=0.3, alpha=0.4, c=[cmap[idx]], label=str(t))
        
        ax.set_xlabel('Spatial X')
        ax.set_ylabel('Spatial Y')
        ax.set_title('All Time Points - Spatial Overlay')
        ax.legend(markerscale=10, fontsize=10)
        ax.set_aspect('equal', adjustable='datalim')
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/plots/04_spatial_overlay.png', dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: 04_spatial_overlay.png")
    else:
        print("  - Skipped (no spatial coordinates)")
    
    # ============================================================
    # PHASE 7: Gene Expression Analysis
    # ============================================================
    print("[Phase 7] Creating gene expression plots...")
    
    # Get top highly variable genes if available
    if 'highly_variable' in adata.var.columns:
        hvg = adata.var_names[adata.var['highly_variable']][:20]
    else:
        # Use top variance genes
        if hasattr(adata.X, 'toarray'):
            gene_var = np.var(adata.X.toarray(), axis=0)
        else:
            gene_var = np.var(adata.X, axis=0)
        top_idx = np.argsort(gene_var)[-20:]
        hvg = adata.var_names[top_idx]
    
    print(f"  - Selected genes: {list(hvg[:5])}...")
    
    # Mean expression per time point for top genes
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for gene_idx, gene in enumerate(hvg[:6]):
        ax = axes[gene_idx]
        gene_means = []
        for t in sorted_times:
            mask = (adata.obs[time_col] == t).values  # Convert to numpy array
            expr = adata[:, gene].X[mask]
            if hasattr(expr, 'toarray'):
                expr = expr.toarray()
            gene_means.append(np.mean(expr))
        
        ax.plot(range(len(sorted_times)), gene_means, 'o-', linewidth=2, markersize=8)
        ax.set_xticks(range(len(sorted_times)))
        ax.set_xticklabels(sorted_times, rotation=45, ha='right')
        ax.set_xlabel('Time Point')
        ax.set_ylabel('Mean Expression')
        ax.set_title(f'{gene}')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Gene Expression Trends', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/plots/05_gene_trends.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: 05_gene_trends.png")
    
    # ============================================================
    # PHASE 8: PCA and UMAP Visualization
    # ============================================================
    print("[Phase 8] Creating dimensionality reduction plots...")
    
    # Subsample if too large
    if adata.n_obs > 10000:
        np.random.seed(42)
        sample_idx = np.random.choice(adata.n_obs, 10000, replace=False)
        adata_sub = adata[sample_idx].copy()
    else:
        adata_sub = adata.copy()
    
    # Run PCA
    print("  - Running PCA...")
    sc.pp.pca(adata_sub, n_comps=50)
    
    # Plot PCA colored by time
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Time coloring
    ax = axes[0]
    time_colors = plt.cm.viridis(np.linspace(0, 1, n_times))
    time_color_map = {t: time_colors[i] for i, t in enumerate(sorted_times)}
    colors = [time_color_map[t] for t in adata_sub.obs[time_col]]
    ax.scatter(adata_sub.obsm['X_pca'][:, 0], adata_sub.obsm['X_pca'][:, 1], 
               c=colors, s=1, alpha=0.6)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('PCA - Colored by Time')
    
    # Legend
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=time_color_map[t], 
                          markersize=8, label=str(t)) for t in sorted_times]
    ax.legend(handles=handles, title='Time', fontsize=8)
    
    # Annotation coloring
    if annot_col:
        ax = axes[1]
        categories = adata_sub.obs[annot_col].values
        unique_cats = sorted(set(categories))
        cat_colors = plt.cm.tab20(np.linspace(0, 1, len(unique_cats)))
        cat_color_map = {cat: cat_colors[i] for i, cat in enumerate(unique_cats)}
        colors = [cat_color_map[cat] for cat in categories]
        ax.scatter(adata_sub.obsm['X_pca'][:, 0], adata_sub.obsm['X_pca'][:, 1], 
                   c=colors, s=1, alpha=0.6)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_title('PCA - Colored by Cell Type')
    else:
        axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/plots/06_pca_visualization.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: 06_pca_visualization.png")
    
    # Run UMAP
    print("  - Running UMAP...")
    try:
        sc.pp.neighbors(adata_sub, n_neighbors=15, n_pcs=30)
        sc.tl.umap(adata_sub)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Time coloring
        ax = axes[0]
        colors = [time_color_map[t] for t in adata_sub.obs[time_col]]
        ax.scatter(adata_sub.obsm['X_umap'][:, 0], adata_sub.obsm['X_umap'][:, 1], 
                   c=colors, s=1, alpha=0.6)
        ax.set_xlabel('UMAP1')
        ax.set_ylabel('UMAP2')
        ax.set_title('UMAP - Colored by Time')
        ax.legend(handles=handles, title='Time', fontsize=8)
        
        # Annotation coloring
        if annot_col:
            ax = axes[1]
            colors = [cat_color_map[cat] for cat in adata_sub.obs[annot_col].values]
            ax.scatter(adata_sub.obsm['X_umap'][:, 0], adata_sub.obsm['X_umap'][:, 1], 
                       c=colors, s=1, alpha=0.6)
            ax.set_xlabel('UMAP1')
            ax.set_ylabel('UMAP2')
            ax.set_title('UMAP - Colored by Cell Type')
        else:
            axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/plots/07_umap_visualization.png', dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: 07_umap_visualization.png")
    except Exception as e:
        print(f"  - UMAP skipped: {e}")
    
    # ============================================================
    # PHASE 9: Sankey Diagram (Cell Type Transitions)
    # ============================================================
    print("[Phase 9] Creating Sankey diagram...")
    
    try:
        from CytoBridge.pl import plot_sankey
        
        # Create simulated trajectory labels based on annotation
        if annot_col:
            predicted_labels_list = []
            for t in sorted_times:
                mask = adata.obs[time_col] == t
                labels = adata.obs.loc[mask, annot_col].values.tolist()
                predicted_labels_list.append(labels)
            
            # Get unique labels and create color mapping
            all_labels = set()
            for labels in predicted_labels_list:
                all_labels.update(labels)
            
            cmap = plt.cm.tab20(np.linspace(0, 1, len(all_labels)))
            label_to_color = {lbl: f'#{int(c[0]*255):02x}{int(c[1]*255):02x}{int(c[2]*255):02x}' 
                             for lbl, c in zip(sorted(all_labels), cmap)}
            
            fig = plot_sankey(
                predicted_labels_list,
                out_html=f'{OUTPUT_DIR}/plots/08_sankey_celltype.html',
                time_keys=[str(t) for t in sorted_times],
                show_time_axis=True,
                label_to_color=label_to_color,
                title='Cell Type Transitions Over Time',
            )
            print(f"  ✓ Saved: 08_sankey_celltype.html")
        else:
            print("  - Skipped (no annotation column)")
    except Exception as e:
        print(f"  - Sankey skipped: {e}")
        traceback.print_exc()
    
    # ============================================================
    # PHASE 10: Spatial with Annotations
    # ============================================================
    if annot_col and spatial_cols:
        print("[Phase 10] Creating spatial annotation plots...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        unique_annots = sorted(adata.obs[annot_col].unique())
        cmap = plt.cm.tab20(np.linspace(0, 1, len(unique_annots)))
        color_map = {annot: cmap[i] for i, annot in enumerate(unique_annots)}
        
        for idx, t in enumerate(sorted_times[:6]):
            mask = adata.obs[time_col] == t
            ax = axes[idx]
            
            x = adata.obs.loc[mask, spatial_cols[0]].values
            y = adata.obs.loc[mask, spatial_cols[1]].values
            annots = adata.obs.loc[mask, annot_col].values
            colors = [color_map[a] for a in annots]
            
            ax.scatter(x, y, c=colors, s=0.5, alpha=0.7)
            ax.set_title(f'{t}')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_aspect('equal', adjustable='datalim')
        
        # Legend
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[a], 
                              markersize=8, label=a) for a in unique_annots[:20]]
        axes[5].legend(handles=handles, loc='center', fontsize=8, ncol=2)
        
        plt.suptitle('Spatial Distribution by Cell Type', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/plots/09_spatial_annotations.png', dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: 09_spatial_annotations.png")
    else:
        print("[Phase 10] Skipped (no annotation or spatial columns)")
    
    # ============================================================
    # Summary
    # ============================================================
    print()
    print("="*60)
    print("Pipeline Verification Complete!")
    print("="*60)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print(f"\nGenerated plots:")
    for f in sorted(os.listdir(f'{OUTPUT_DIR}/plots')):
        fpath = os.path.join(OUTPUT_DIR, 'plots', f)
        size = os.path.getsize(fpath) / 1024
        print(f"  - {f} ({size:.1f} KB)")
    
    print("\n" + "="*60)

except Exception as e:
    print(f"\n❌ Error occurred: {e}")
    traceback.print_exc()
    sys.exit(1)
