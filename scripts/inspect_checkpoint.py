import torch
print("Starting inspect_checkpoint...", flush=True)
import sys
import os
import yaml

# Add package source to path
sys.path.append('/lustre/home/2501111653/CytoBridge-ST-package/CytoBridge')

from CytoBridge.tl.core.models import DynamicalModel

def inspect_keys():
    # 1. Load old checkpoint
    old_ckpt_path = '/lustre/home/2501111653/CytoBridge-ST-1104/results/mosta_interaction_1017_tiaocan/model_final'
    try:
        old_state_dict = torch.load(old_ckpt_path, map_location='cpu')
        print(f"Old Checkpoint Keys ({len(old_state_dict)}):")
        for k in list(old_state_dict.keys())[:10]: # Print first 10
            print(f"  {k}")
    except Exception as e:
        print(f"Error loading old checkpoint: {e}")
        return

    # 2. Initialize new model
    config_path = '/lustre/home/2501111653/CytoBridge-ST-package/CytoBridge/CytoBridge/configs/simulation_config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Mock data dimensions (from simulation data)
    latent_dim = 52 # spatial + gene
    
    try:
        model = DynamicalModel(latent_dim=latent_dim, config=config['model'], use_growth_in_ode_inter=True)
        new_state_dict = model.state_dict()
        print(f"\nNew Model Keys ({len(new_state_dict)}):")
        for k in list(new_state_dict.keys())[:10]:
            print(f"  {k}")
            
        # Check for key mismatches
        common = set(old_state_dict.keys()) & set(new_state_dict.keys())
        only_old = set(old_state_dict.keys()) - set(new_state_dict.keys())
        only_new = set(new_state_dict.keys()) - set(old_state_dict.keys())
        
        print(f"\nCommon keys: {len(common)}")
        print(f"Only in old: {len(only_old)}")
        if len(only_old) > 0:
            print(f"Example only in old: {list(only_old)[:5]}")
            
        print(f"Only in new: {len(only_new)}")
        if len(only_new) > 0:
            print(f"Example only in new: {list(only_new)[:5]}")

    except Exception:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    inspect_keys()
