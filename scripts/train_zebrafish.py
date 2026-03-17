#!/usr/bin/env python3
"""
Complete Zebrafish Training Pipeline
=====================================
Runs the full CytoBridge training pipeline on Zebrafish spatial transcriptomics data.

Stages:
1. Data Loading & Preprocessing (spatial alignment)
2. Model Training (6 stages: Pretrain, Refine, Init_interaction, Score, Finetune, Score_Final)
3. Model Evaluation
4. Downstream Analysis with Trained Model
5. Generate All Visualizations
"""

import os
import sys
import warnings
import time
warnings.filterwarnings('ignore')

# Add CytoBridge package to path
sys.path.insert(0, '/lustre/home/2501111653/CytoBridge-ST-package/CytoBridge')

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scanpy as sc

# Input/Output paths
H5AD_PATH = '/lustre/home/2501111653/CytoBridge-ST-1104/spatial_data/spatial_sixtime_slice_stereoseq.h5ad'
OUTPUT_DIR = '/lustre/home/2501111653/CytoBridge-ST-package/results/zebrafish_training'
CONFIG_PATH = '/lustre/home/2501111653/CytoBridge-ST-package/CytoBridge/CytoBridge/configs/zebrafish_training.yaml'

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f'{OUTPUT_DIR}/plots', exist_ok=True)
os.makedirs(f'{OUTPUT_DIR}/checkpoints', exist_ok=True)

print("="*70)
print("CytoBridge Complete Training Pipeline - Zebrafish Data")
print("="*70)
print(f"Input: {H5AD_PATH}")
print(f"Output: {OUTPUT_DIR}")
print(f"Config: {CONFIG_PATH}")
print(f"Device: cuda" if torch.cuda.is_available() else "Device: cpu")
print()

start_time = time.time()

try:
    import CytoBridge as cb
    from CytoBridge.utils.config import load_config
    from CytoBridge.pp.spatial_align import AlignConfig, preprocess_align_to_files
    
    # ============================================================
    # PHASE 1: Load and Preprocess Data
    # ============================================================
    print("[Phase 1] Loading and preprocessing data...")
    print("-" * 50)
    
    # Load config
    config = load_config(CONFIG_PATH)
    print(f"  - Loaded training config with {len(config.get('training', {}).get('plan', []))} stages")
    
    # Set up alignment configuration
    align_cfg = AlignConfig(
        center_x=True,
        center_y=False,
        scale_x=1.0,
        scale_y=1.0,
        flip_y=False,
        n_pcs=50,
        phase1_epochs=1000,
        phase2_epochs=50,
        batch_size=512,
    )
    
    # Time key and batch indices for Zebrafish
    time_key = 'time'
    
    # Load data to get time points
    adata = sc.read_h5ad(H5AD_PATH)
    print(f"  - Loaded h5ad: {adata.shape}")
    
    # Parse time points
    def parse_time(t):
        if isinstance(t, str) and 'hpf' in t:
            return float(t.replace('hpf', ''))
        return float(t)
    
    time_points = sorted(adata.obs[time_key].unique(), key=parse_time)
    print(f"  - Time points: {time_points}")
    
    # Create batch indices (0-indexed for the sorted time order)
    batch_indices = list(range(len(time_points)))
    print(f"  - Batch indices: {batch_indices}")
    
    # Output CSV path for aligned data
    aligned_csv = f'{OUTPUT_DIR}/zebrafish_aligned.csv'
    aligned_h5ad = f'{OUTPUT_DIR}/zebrafish_aligned.h5ad'
    
    # Run preprocessing
    if os.path.exists(aligned_csv):
        print(f"  - Found existing aligned data: {aligned_csv}")
        df_aligned = pd.read_csv(aligned_csv)
    else:
        print("  - Running spatial alignment (this may take a while)...")
        df_aligned = preprocess_align_to_files(
            h5ad_path=H5AD_PATH,
            time_key=time_key,
            output_csv=aligned_csv,
            output_h5ad=aligned_h5ad,
            cfg=align_cfg,
            batch_indices=batch_indices,
            device='cuda' if torch.cuda.is_available() else 'cpu',
        )
        print(f"  - Saved aligned data to: {aligned_csv}")
    
    print(f"  - Aligned data shape: {df_aligned.shape}")
    print(f"  ✓ Phase 1 complete ({time.time() - start_time:.1f}s)")
    print()
    
    # ============================================================
    # PHASE 2: Model Training
    # ============================================================
    print("[Phase 2] Training model...")
    print("-" * 50)
    
    # Update config with spatial dimension
    config['model']['spatial_dim'] = 2
    
    # Get feature dimension from data
    dim = df_aligned.shape[1] - 1  # Subtract 'samples' column
    print(f"  - Feature dimension: {dim}")
    
    # Run training
    phase2_start = time.time()
    
    model, training_history = cb.tl.fit_spatial_csv(
        aligned_csv,
        config=config,
        device='cuda' if torch.cuda.is_available() else 'cpu',
    )
    
    # Save model
    model_path = f'{OUTPUT_DIR}/model_final.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'dim': dim,
    }, model_path)
    print(f"  - Saved model to: {model_path}")
    
    print(f"  ✓ Phase 2 complete ({time.time() - phase2_start:.1f}s)")
    print()
    
    # ============================================================
    # PHASE 3: Downstream Analysis with Trained Model
    # ============================================================
    print("[Phase 3] Running downstream analysis...")
    print("-" * 50)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()
    
    # 3.1 Compute velocity components
    print("  - Computing velocity components...")
    try:
        unique_times = sorted(df_aligned['samples'].unique())
        velocity_results = {}
        
        for t_idx, t_val in enumerate(unique_times):
            subset = df_aligned[df_aligned['samples'] == t_val]
            feature_cols = [c for c in subset.columns if c != 'samples']
            data = subset[feature_cols].values
            
            from CytoBridge.tl import compute_velocity_components
            components = compute_velocity_components(
                data=data,
                time_value=float(t_val),
                model=model,
                device=device,
            )
            velocity_results[t_val] = components
            print(f"    - t={t_val}: computed velocity components")
        
        # Save velocity results
        np.save(f'{OUTPUT_DIR}/velocity_components.npy', velocity_results, allow_pickle=True)
        print(f"    - Saved velocity components to velocity_components.npy")
    except Exception as e:
        print(f"    - Velocity computation skipped: {e}")
    
    # 3.2 Train cell-type classifier and predict trajectories
    print("  - Training cell-type classifier...")
    try:
        # Get annotations from original adata
        adata = sc.read_h5ad(H5AD_PATH)
        annot_col = 'bin_annotation'
        
        # Create annotation mapping
        train_df = df_aligned.copy()
        
        # Add annotations by matching cell order
        annotations = []
        for t_val in sorted(df_aligned['samples'].unique()):
            mask = df_aligned['samples'] == t_val
            n_cells = mask.sum()
            
            # Get corresponding cells from adata
            t_idx = sorted(time_points).index(time_points[int(t_val)])
            t_label = time_points[int(t_val)]
            adata_mask = adata.obs[time_key] == t_label
            adata_annot = adata.obs.loc[adata_mask, annot_col].values
            
            if len(adata_annot) >= n_cells:
                annotations.extend(adata_annot[:n_cells])
            else:
                # Pad with most common
                annotations.extend(adata_annot.tolist())
                most_common = pd.Series(adata_annot).mode()[0]
                annotations.extend([most_common] * (n_cells - len(adata_annot)))
        
        train_df['annotation'] = annotations[:len(train_df)]
        
        from CytoBridge.tl import train_mlp_classifier
        
        classifier, label_encoder, accuracy = train_mlp_classifier(
            adata,
            label_col=annot_col,
            time_key=time_key,
            obsm_key='X_latent',
            spatial_key='spatial_aligned',
            concat_spatial=True,
            hidden_size=128,
            epochs=50,
            device=device,
        )
        print(f"    - Classifier accuracy: {accuracy:.2%}")
        
        # Save classifier
        torch.save({
            'model_state_dict': classifier.state_dict(),
            'label_encoder_classes': label_encoder.classes_,
        }, f'{OUTPUT_DIR}/classifier.pt')
        print(f"    - Saved classifier to classifier.pt")
        
    except Exception as e:
        print(f"    - Classifier training skipped: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"  ✓ Phase 3 complete ({time.time() - start_time:.1f}s total)")
    print()
    
    # ============================================================
    # PHASE 4: Generate All Visualizations
    # ============================================================
    print("[Phase 4] Generating visualizations...")
    print("-" * 50)
    
    # 4.1 Training loss curves
    if training_history is not None and 'losses' in training_history:
        print("  - Plotting training curves...")
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            losses = training_history['losses']
            ax.plot(losses)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('Training Loss')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'{OUTPUT_DIR}/plots/10_training_loss.png', dpi=200, bbox_inches='tight')
            plt.close()
            print("    - Saved: 10_training_loss.png")
        except Exception as e:
            print(f"    - Training curves skipped: {e}")
    
    # 4.2 Growth rate visualization
    print("  - Plotting growth rates...")
    try:
        from CytoBridge.pl import plot_g_values, plot_growth_per_time
        
        plot_growth_per_time(
            df=df_aligned,
            dim=dim,
            model=model,
            out_dir=f'{OUTPUT_DIR}/plots/growth',
            device=device,
            samples_column='samples',
        )
        print("    - Saved: growth plots")
    except Exception as e:
        print(f"    - Growth plots skipped: {e}")
    
    # 4.3 Velocity field visualization
    print("  - Plotting velocity fields...")
    try:
        # Simple velocity arrow plot
        for t_idx, t_val in enumerate(unique_times[:3]):  # First 3 time points
            if t_val in velocity_results:
                subset = df_aligned[df_aligned['samples'] == t_val]
                x = subset['x1'].values
                y = subset['x2'].values
                vel = velocity_results[t_val]
                
                fig, ax = plt.subplots(figsize=(8, 8))
                
                # Subsample for quiver plot
                n_show = min(500, len(x))
                idx = np.random.choice(len(x), n_show, replace=False)
                
                ax.scatter(x[idx], y[idx], s=1, alpha=0.3, c='gray')
                
                # Get velocity in spatial dims
                drift = vel['full'][:, :2]  # First 2 dims are spatial
                ax.quiver(x[idx], y[idx], drift[idx, 0], drift[idx, 1],
                         scale=1, scale_units='xy', alpha=0.7, color='blue')
                
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_title(f'Velocity Field (t={t_val})')
                ax.set_aspect('equal')
                plt.tight_layout()
                plt.savefig(f'{OUTPUT_DIR}/plots/11_velocity_t{int(t_val)}.png', dpi=200, bbox_inches='tight')
                plt.close()
                print(f"    - Saved: 11_velocity_t{int(t_val)}.png")
    except Exception as e:
        print(f"    - Velocity plots skipped: {e}")
    
    # 4.4 Sankey diagram with predictions
    print("  - Creating Sankey diagram...")
    try:
        from CytoBridge.pl import plot_sankey
        
        # Use annotations if available
        predicted_labels_list = []
        for t_val in unique_times:
            mask = train_df['samples'] == t_val if 'train_df' in dir() else df_aligned['samples'] == t_val
            if 'train_df' in dir() and 'annotation' in train_df.columns:
                labels = train_df.loc[mask, 'annotation'].values.tolist()
            else:
                # Use spatial regions as proxy
                subset = df_aligned[df_aligned['samples'] == t_val]
                labels = []
                for _, row in subset.iterrows():
                    if row['x1'] < subset['x1'].median():
                        labels.append('Region_A' if row['x2'] < subset['x2'].median() else 'Region_B')
                    else:
                        labels.append('Region_C' if row['x2'] < subset['x2'].median() else 'Region_D')
            predicted_labels_list.append(labels)
        
        fig = plot_sankey(
            predicted_labels_list,
            out_html=f'{OUTPUT_DIR}/plots/12_sankey.html',
            time_keys=[str(t) for t in unique_times],
            show_time_axis=True,
            title='Cell Type Transitions',
        )
        print("    - Saved: 12_sankey.html")
    except Exception as e:
        print(f"    - Sankey skipped: {e}")
    
    print(f"  ✓ Phase 4 complete")
    print()
    
    # ============================================================
    # Summary
    # ============================================================
    elapsed = time.time() - start_time
    print("="*70)
    print(f"Training Pipeline Complete! (Total time: {elapsed/60:.1f} minutes)")
    print("="*70)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print(f"\nGenerated files:")
    for root, dirs, files in os.walk(OUTPUT_DIR):
        for f in sorted(files):
            fpath = os.path.join(root, f)
            size = os.path.getsize(fpath) / 1024
            rel_path = os.path.relpath(fpath, OUTPUT_DIR)
            print(f"  - {rel_path} ({size:.1f} KB)")
    
    print("\n" + "="*70)

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
