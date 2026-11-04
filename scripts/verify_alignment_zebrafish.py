
import os
import sys
import pandas as pd
import numpy as np
import scipy.stats

# Add package to path
sys.path.insert(0, "/lustre/home/2501111653/CytoBridge-ST-package/CytoBridge")

from CytoBridge.pp.spatial_align import AlignConfig, preprocess_align_to_files

def main():
    print("=== Verification: Zebrafish Alignment ===")
    
    # Paths
    h5ad_path = "/lustre/home/2501111653/CytoBridge-ST-1104/spatial_data/spatial_sixtime_slice_stereoseq.h5ad"
    ref_csv_path = "/lustre/home/2501111653/CytoBridge-ST-1104/data/zebrafish_1108.csv"
    output_dir = "results/verification/"
    output_csv = os.path.join(output_dir, "zebrafish_verified.csv")
    output_h5ad = os.path.join(output_dir, "zebrafish_verified.h5ad")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Zebrafish Preset Config
    print("Configuring Alignment (Zebrafish preset)...")
    cfg = AlignConfig(
        center_x=True,
        center_y=False,
        scale_x=1.0,
        scale_y=1.0,
        flip_y=False,
        n_pcs=50,
        # Full production epochs
        phase1_epochs=10000, 
        phase2_epochs=500,
        spatial_dim=2 
    )
    time_key = "time"
    batch_indices = [1, 2, 3, 4, 5] # Note: 0-indexed in list?
    # Original preset logic: "time", [1, 2, 3, 4, 5]. 
    # Let's check if batch_indices means indices in the list of batches or values.
    # spatial_align.py: batch_names_selected = [batch_names[i] for i in batch_indices]
    # So these are indices into the sorted batch names.
    
    print(f"Running Alignment -> {output_csv}")
    preprocess_align_to_files(
        h5ad_path=h5ad_path,
        time_key=time_key,
        output_csv=output_csv,
        output_h5ad=output_h5ad,
        cfg=cfg,
        batch_indices=batch_indices,
        device="cuda"
    )
    
    print("\n=== Validation and Comparison ===")
    if not os.path.exists(output_csv):
        print(f"ERROR: Output CSV not found at {output_csv}")
        return

    # Load Data
    print(f"Loading generated CSV: {output_csv}")
    df_new = pd.read_csv(output_csv)
    print(f"Loading reference CSV: {ref_csv_path}")
    df_ref = pd.read_csv(ref_csv_path)
    
    print(f"\nShape New: {df_new.shape}")
    print(f"Shape Ref: {df_ref.shape}")
    
    # Column Check
    print("\nChecking Columns...")
    new_cols = list(df_new.columns)
    ref_cols = list(df_ref.columns)
    
    # Ref might have x1, x2 or spatial_1, spatial_2?
    # New code generates: samples, x1, x2 ... (from spatial_align.py: "x{i}")
    # Let's check what Ref has.
    print(f"New Columns (first 5): {new_cols[:5]}")
    print(f"Ref Columns (first 5): {ref_cols[:5]}")
    
    common_cols = [c for c in new_cols if c in ref_cols and c != 'samples']
    print(f"Common feature columns (PCA/genes): {len(common_cols)}")
    
    # Feature Correlation Check
    # We expect PCA components (e.g. '0', '1', '2' ... or however named in CSV) to be highly correlated 
    # if preprocessing is identical.
    # spatial_align outputs Features as remaining columns.
    # spatial_align.py line 210: output_i = np.hstack((x_prime_full, features_full))
    # line 217: feature columns are NOT named?
    # line 217: column_names = ["samples"] + [f"x{i}" for i in range(1, combined_data.shape[1])]
    # Wait, the column naming in `spatial_align.py` was:
    # ["samples"] + [f"x{i}" for i in range(1, ...)]
    # This names EVERYTHING x1, x2, x3... 
    # So x1, x2 are spatial. x3... are features (PCA 0, PCA 1...).
    
    # Let's verify reference format.
    # If reference uses same naming, we compare x3_new vs x3_ref.
    
    if 'x3' in df_new.columns and 'x3' in df_ref.columns:
        corr, _ = scipy.stats.pearsonr(df_new['x3'], df_ref['x3'])
        print(f"Correlation of feature 'x3' (likely PC1): {corr:.4f}")
    else:
        print("Feature columns 'x3' not found in both. Cannot compare features directly by name.")
        
    # Check sample counts per time point
    print("\nSample Counts per Time Point:")
    print("NEW:")
    print(df_new['samples'].value_counts().sort_index())
    print("REF:")
    print(df_ref['samples'].value_counts().sort_index())

if __name__ == "__main__":
    main()
