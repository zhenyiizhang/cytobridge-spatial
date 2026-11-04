import sys
import os
import torch
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import pearsonr

pkg_path = "/lustre/home/2501111653/CytoBridge-ST-package/CytoBridge"
if pkg_path not in sys.path:
    sys.path.insert(0, pkg_path)

import CytoBridge as cb
from CytoBridge.tl.core.models import DynamicalModel

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Paths
results_dir = "/lustre/home/2501111653/CytoBridge-ST-package/CytoBridge/results/simulation_test"
adata_path = os.path.join(results_dir, "adata.h5ad")
gt_dir = "/lustre/home/2501111653/CytoBridge-ST-1104/results/mosta_interaction_1017_tiaocan"

# Load trained adata
print(f"Loading adata from {adata_path}")
if not os.path.exists(adata_path):
    print("Error: adata.h5ad not found. Training might have failed or is still running.")
    sys.exit(1)
adata = sc.read_h5ad(adata_path)

# Extract model config and state
model_info = adata.uns['all_model']
model_config = model_info['model_config']
model_state_dict = model_info['model_state_dict']

# Reconstruct model
print("Reconstructing model...")
dim = adata.shape[1]
model = DynamicalModel(dim, model_config)
# Load state dict (convert numpy arrays back to tensors)
state_dict_tensor = {k: torch.tensor(v) for k, v in model_state_dict.items()}
model.load_state_dict(state_dict_tensor)
model.to(device)
model.eval()

# Helper to get interaction net
interaction_net = model.interaction_net

# Verify per time point
time_key = "time_point_processed"
times = sorted(adata.obs[time_key].unique())
print(f"Time points: {times}")

correlations = []

for t_val in times:
    t_int = int(t_val)
    print(f"\nProcessing time point {t_int}")
    
    # Get data for this time point
    mask = adata.obs[time_key] == t_val
    sub_adata = adata[mask]
    
    # Prepare inputs
    # Use X_latent if available, otherwise X
    if "X_latent" in sub_adata.obsm:
        X = sub_adata.obsm["X_latent"]
    else:
        X = sub_adata.X
    
    # Convert to tensor
    x_tensor = torch.tensor(X, dtype=torch.float32, device=device)
    t_tensor = torch.tensor(t_val, dtype=torch.float32, device=device).unsqueeze(0) # scalar
    
    # We need lnw (log mass). 
    # In training, w is uniform or learned?
    # Usually assumed uniform initially or specific column.
    # DynamicalModel.forward uses lnw passed to it.
    # In fit.py, it doesn't seem to pass lnw to model.train explicitly from adata?
    # Wait, TrainingPipeline.train handles it. 
    # Let's assume uniform mass for verification as we don't have stored mass in adata.obsm easily unless we re-run growth/etc.
    # But interaction depends on lnw.
    # If growth is used, lnw evolves.
    # For 'Init_interaction', it likely uses lnw from growth?
    # The notebook just calls f_net(qt). It doesn't seem to pass lnw?
    # Old GNN forward: forward(x, lnw, t, return_attn).
    # New GNN forward: forward(x, lnw, t, return_attn).
    # If we pass zeros for lnw (mass=1), let's see.
    # Note: lnw should be shape (N, 1) or (N,).
    
    lnw_tensor = torch.zeros(x_tensor.size(0), 1, device=device) 
    
    # Run interaction net
    with torch.no_grad():
        # interaction_net.forward(x, lnw, t, return_attn=True)
        # Note: t should be tensor
        _ = interaction_net(x_tensor, lnw_tensor, t_tensor, return_attn=True)
        
        # Get attention
        # Assuming single layer GNN
        attn = interaction_net.gnn_layers[0].attn # Shape (E, H)
        
        # Calculate mean over heads
        pred_mean = attn.mean(dim=1).cpu().numpy() # Shape (E,)
        
    print(f"Pred mean shape: {pred_mean.shape}")
    
    # Load GT
    gt_path = os.path.join(gt_dir, f"attn_mean_time{t_int}.npy")
    if not os.path.exists(gt_path):
        print(f"Warning: GT file {gt_path} not found.")
        continue
        
    gt_mean = np.load(gt_path)
    print(f"GT mean shape: {gt_mean.shape}")
    
    if pred_mean.shape != gt_mean.shape:
        print(f"Mismatch in shapes! Cannot calculate correlation.")
        # Try to diagnose
        print(f"Pred edges: {len(pred_mean)}, GT edges: {len(gt_mean)}")
    else:
        corr, _ = pearsonr(pred_mean, gt_mean)
        print(f"Correlation: {corr:.4f}")
        correlations.append(corr)

print("\nSummary:")
if correlations:
    print(f"Average Correlation: {np.mean(correlations):.4f}")
else:
    print("No correlations calculated.")
