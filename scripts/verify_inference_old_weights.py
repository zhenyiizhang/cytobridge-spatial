import torch
import numpy as np
import pandas as pd
import yaml
import sys
import os

# Add package source to path
sys.path.append('/lustre/home/2501111653/CytoBridge-ST-package/CytoBridge')

from CytoBridge.tl.core.models import DynamicalModel

def load_and_map_weights(model, old_ckpt_path, old_score_ckpt_path):
    # Load old weights
    print(f"Loading weights from {old_ckpt_path}...")
    old_state = torch.load(old_ckpt_path, map_location='cpu')
    print(f"Loading score weights from {old_score_ckpt_path}...")
    old_score_state = torch.load(old_score_ckpt_path, map_location='cpu')
    
    new_state = model.state_dict()
    key_map = {}
    
    # Map Velocity (v_net -> velocity_net)
    # The structure seems identical (ModuleList of Sequentials), just prefix change
    for key in old_state:
        if key.startswith('v_net.'):
            new_key = key.replace('v_net.', 'velocity_net.')
            if new_key in new_state:
                key_map[new_key] = old_state[key]
                
    # Map Growth (g_net -> growth_net)
    # Old: g_net.net.0.weight -> New: growth_net.input_layer.0.weight or similar?
    # Actually inspection showed: 'g_net.net.0.bias' in old
    # New showed: 'growth_net.hidden_layers.1.0.bias'
    # Wait, old growthNet was a Sequential(Linear, Tanh, Linear, Tanh...)
    # New HyperNetwork uses input_layer, hidden_layers (ModuleList), output_layer
    # We need to map manually based on layer index
    
    # Growth Mapping Strategy:
    # Old: net.0 (Linear), net.2 (Linear), net.4 (Linear), net.6 (Linear)
    # New: input_layer (Sequential or Linear), hidden_layers (List of Sequential), output_layer (Linear)
    
    # Let's inspect weights by shape if possible, or just strict index mapping
    # New HyperNetwork:
    # input_layer: Linear(in, hidden) -> weight, bias
    # hidden_layers[i]: Sequential(Linear(hidden, hidden), Act) -> 0.weight, 0.bias
    # output_layer: Linear(hidden, 1) -> weight, bias
    
    growth_mapping = {
        'g_net.net.0.weight': 'growth_net.input_layer.0.weight',
        'g_net.net.0.bias': 'growth_net.input_layer.0.bias',
        'g_net.net.2.weight': 'growth_net.hidden_layers.0.0.weight',
        'g_net.net.2.bias': 'growth_net.hidden_layers.0.0.bias',
        'g_net.net.4.weight': 'growth_net.hidden_layers.1.0.weight',
        'g_net.net.4.bias': 'growth_net.hidden_layers.1.0.bias',
        'g_net.net.6.weight': 'growth_net.output_layer.weight',
        'g_net.net.6.bias': 'growth_net.output_layer.bias',
    }
    for old_k, new_k in growth_mapping.items():
        if old_k in old_state and new_k in new_state:
             key_map[new_k] = old_state[old_k]
             
    # Score Mapping Strategy (Similar to Growth, both are MLPs)
    # Old scoreNet: net.0, net.2, net.4, net.6
    score_mapping = {
        'net.0.weight': 'score_net.input_layer.0.weight',
        'net.0.bias': 'score_net.input_layer.0.bias',
        'net.2.weight': 'score_net.hidden_layers.0.0.weight',
        'net.2.bias': 'score_net.hidden_layers.0.0.bias',
        'net.4.weight': 'score_net.hidden_layers.1.0.weight',
        'net.4.bias': 'score_net.hidden_layers.1.0.bias',
        'net.6.weight': 'score_net.output_layer.weight',
        'net.6.bias': 'score_net.output_layer.bias',
    }
    for old_k, new_k in score_mapping.items():
        if old_k in old_score_state and new_k in new_state:
             key_map[new_k] = old_score_state[old_k]

    # Interaction Mapping (interaction_net -> interaction_net)
    # If using GNN, structure should be preserved if classes match
    # Old: interaction_net...
    # New: interaction_net...
    for key in old_state:
        if key.startswith('interaction_net.'):
            # Direct mapping should work if GNN implementations align
            if key in new_state:
                key_map[key] = old_state[key]
                
    # Load mapped weights
    print(f"Mapped {len(key_map)} keys.")
    missing = set(new_state.keys()) - set(key_map.keys())
    if missing:
        print(f"Missing keys in new model: {len(missing)}")
        # print(list(missing)[:5])
        
    model.load_state_dict(key_map, strict=False)
    return model

def main():
    device = torch.device('cpu') # Use CPU for inference to avoid potential CUDA issues on head node if any
    
    # 1. Config & Model
    config_path = '/lustre/home/2501111653/CytoBridge-ST-package/CytoBridge/CytoBridge/configs/simulation_config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    model = DynamicalModel(latent_dim=52, config=config['model'], use_growth_in_ode_inter=True)
    
    # 2. Load Weights
    old_ckpt = '/lustre/home/2501111653/CytoBridge-ST-1104/results/mosta_interaction_1017_tiaocan/model_final'
    old_score_ckpt = '/lustre/home/2501111653/CytoBridge-ST-1104/results/mosta_interaction_1017_tiaocan/score_model'
    
    model = load_and_map_weights(model, old_ckpt, old_score_ckpt)
    model.to(device)
    model.eval()
    
    # 3. Load Data
    data_path = '/lustre/home/2501111653/CytoBridge-ST-1104/data/mouse_brain_simulation.csv'
    df = pd.read_csv(data_path)
    # df columns: samples, x1...x52 (assuming)
    # Actually from reading file it has 'samples' column
    
    # 4. Inference Loop
    time_points = df['samples'].unique()
    dim = 52
    
    all_g_pred = []
    all_v_pred = []
    
    print("\nStarting Inference & Correlation Calculation...")
    
    for t in time_points:
        subset = df[df['samples'] == t]
        # Assuming data columns start at index 1 (0 is samples?) - NO, verify csv read
        # In extracted code: df = df.iloc[:, :config['data']['dim'] + 1] -> cols 0 to 52
        # samples is col 0, x1..x52 are cols 1..52
        
        data_np = subset.iloc[:, 1:dim+1].values
        data_tensor = torch.tensor(data_np, dtype=torch.float32).to(device)
        lnw0 = torch.log(torch.ones(subset.shape[0], 1) / subset.shape[0]).to(device)
        t_tensor = torch.tensor([t], dtype=torch.float32).to(device) # expand happens in model
        
        # Determine edges (needed for GNN) - Model's interaction_net handles this internally if configured?
        # In fit.py: cal_interaction(x, lnw, model.interaction_net ...)
        # In models.py: forward calls cal_interaction
        # Note: DynamicalModel.forward returns 'interaction' (force), 'growth', 'velocity' etc.
        # But for 'attn', we need to access interaction_net directly or modify forward to return it.
        # The easiest way is to call interaction_net manually as done in extracted code.
        
        # Get V, G
        outputs = model(t_tensor, data_tensor, lnw0)
        v = outputs['velocity'].detach().cpu().numpy()
        g = outputs['growth'].detach().cpu().numpy()
        
        all_g_pred.append(g)
        all_v_pred.append(v)
        
        # Get Attention
        # model.interaction_net is GNNInteraction
        # Forward: x, lnw, time=None, return_attn=False
        # Wait, GNNInteraction signature in new package:
        # forward(self, x, lnw, time=None, return_attn=False)
        
        # Need to expand Time for interaction Net? 
        # In extracted code: f_net.interaction_net(data, lnw0, time_tensor, return_attn=True)
        t_expanded = t_tensor.expand(data_tensor.size(0), 1)
        
        with torch.no_grad():
            model.interaction_net(data_tensor, lnw0, t=t_expanded, return_attn=True)
            attn = model.interaction_net.gnn_layers[0].attn
            attn = torch.abs(attn)
            attn_mean = attn.mean(dim=1).cpu().numpy()
            
            # Load GT Attention
            gt_attn_path = f'/lustre/home/2501111653/CytoBridge-ST-1104/results/mosta_interaction_1017_tiaocan/attn_mean_time{int(t)}.npy'
            if os.path.exists(gt_attn_path):
                gt_attn_mean = np.load(gt_attn_path)
                print(f"\n=== Time {int(t)} Attention Debug ===")
                print(f"  Pred shape: {attn_mean.shape}, GT shape: {gt_attn_mean.shape}")
                print(f"  Pred[:5]: {attn_mean[:5]}")
                print(f"  GT[:5]: {gt_attn_mean[:5]}")
                print(f"  Are identical? {np.allclose(attn_mean, gt_attn_mean)}")
                corr = np.corrcoef(attn_mean.flatten(), gt_attn_mean.flatten())[0, 1]
                print(f"  Correlation: {corr:.4f}")
            else:
                print(f"Time {int(t)} GT Attention not found.")

    # 5. Global Correlations
    all_g_pred = np.concatenate(all_g_pred)
    all_v_pred = np.concatenate(all_v_pred) # v is (N, 54) ? Spatial (2) + Gene (52) -> 54 dim?
    # Old V output: use_spatial=True -> spatial_out(2) + gene_out(dim-2) -> 2 + 50 = 52. 
    # Check if latent_dim=52 means 52 total.
    # In models.py: gene_out is in_out_dim - 2. So total output is 2 + (52-2) = 52. Correct.
    
    # GT Growth
    gt_g_path = '/lustre/home/2501111653/CytoBridge-ST-1104/results/mosta_interaction_1017_tiaocan/g_values.npy'
    if os.path.exists(gt_g_path):
        gt_g = np.load(gt_g_path)
        print(f"\n=== Growth Debug ===")
        print(f"  Pred shape: {all_g_pred.shape}, GT shape: {gt_g.shape}")
        print(f"  Pred[:5]: {all_g_pred[:5].flatten()}")
        print(f"  GT[:5]: {gt_g[:5].flatten()}")
        print(f"  Pred mean: {all_g_pred.mean():.4f}, GT mean: {gt_g.mean():.4f}")
        print(f"  Pred std: {all_g_pred.std():.4f}, GT std: {gt_g.std():.4f}")
        print(f"  Are identical? {np.allclose(all_g_pred.flatten(), gt_g.flatten())}")
        g_corr = np.corrcoef(gt_g.flatten(), all_g_pred.flatten())[0, 1]
        print(f"  Correlation: {g_corr:.4f}")
        
    # GT Velocity
    # The GT file 'simulation_gradients_np_gt.npy' likely contains the true gradients (velocity)
    gt_v_path = '/lustre/home/2501111653/CytoBridge-ST-1104/results/mosta_interaction_1017_tiaocan/simulation_gradients_np_gt.npy'
    if os.path.exists(gt_v_path):
        gt_v = np.load(gt_v_path)
        print(f"\n=== Velocity Debug ===")
        print(f"  Pred shape: {all_v_pred.shape}, GT shape: {gt_v.shape}")
        print(f"  Pred[:3, :3]: {all_v_pred[:3, :3]}")
        print(f"  GT[:3, :3]: {gt_v[:3, :3]}")
        print(f"  Pred mean: {all_v_pred.mean():.4f}, GT mean: {gt_v.mean():.4f}")
        print(f"  Pred std: {all_v_pred.std():.4f}, GT std: {gt_v.std():.4f}")
        print(f"  Are identical? {np.allclose(all_v_pred, gt_v)}")
        # Ensure predictions are flattened same way
        # Note: gt_v shape is likely (Total_Samples, 52)
        v_corr = np.corrcoef(gt_v.flatten(), all_v_pred.flatten())[0, 1]
        print(f"  Correlation: {v_corr:.4f}")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
