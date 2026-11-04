
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

def visualize_comparison():
    new_csv = "results/verification/zebrafish_verified.csv"
    ref_csv = "/lustre/home/2501111653/CytoBridge-ST-1104/data/zebrafish_1108.csv"
    output_img = "results/verification/spatial_comparison.png"
    
    if not os.path.exists(new_csv):
        print(f"Error: {new_csv} not found.")
        return

    print("Loading data...")
    df_new = pd.read_csv(new_csv)
    df_ref = pd.read_csv(ref_csv)
    
    time_points = sorted(df_new['samples'].unique())
    n_times = len(time_points)
    
    fig, axes = plt.subplots(n_times, 2, figsize=(12, 4 * n_times))
    
    print("Plotting...")
    for i, t in enumerate(time_points):
        # Filter data
        d_new = df_new[df_new['samples'] == t]
        d_ref = df_ref[df_ref['samples'] == t]
        
        # Plot New
        ax_new = axes[i, 0]
        ax_new.scatter(d_new['x1'], d_new['x2'], s=1, alpha=0.5, c='blue')
        ax_new.set_title(f"Time {t} - New (10000 epochs)")
        ax_new.set_aspect('equal')
        
        # Plot Ref
        ax_ref = axes[i, 1]
        ax_ref.scatter(d_ref['x1'], d_ref['x2'], s=1, alpha=0.5, c='red')
        ax_ref.set_title(f"Time {t} - Reference (Full)")
        ax_ref.set_aspect('equal')
        
        # Add stats
        new_stats = f"Range X: [{d_new['x1'].min():.2f}, {d_new['x1'].max():.2f}]\nRange Y: [{d_new['x2'].min():.2f}, {d_new['x2'].max():.2f}]"
        ref_stats = f"Range X: [{d_ref['x1'].min():.2f}, {d_ref['x1'].max():.2f}]\nRange Y: [{d_ref['x2'].min():.2f}, {d_ref['x2'].max():.2f}]"
        
        ax_new.text(0.05, 0.95, new_stats, transform=ax_new.transAxes, verticalalignment='top', fontsize=8, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax_ref.text(0.05, 0.95, ref_stats, transform=ax_ref.transAxes, verticalalignment='top', fontsize=8, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Print stats to console for verification
        print(f"--- Time {t} ---")
        print(f"New: X [{d_new['x1'].min():.4f}, {d_new['x1'].max():.4f}], Y [{d_new['x2'].min():.4f}, {d_new['x2'].max():.4f}]")
        print(f"Ref: X [{d_ref['x1'].min():.4f}, {d_ref['x1'].max():.4f}], Y [{d_ref['x2'].min():.4f}, {d_ref['x2'].max():.4f}]")

    plt.tight_layout()
    plt.savefig(output_img)
    print(f"Comparison plot saved to {output_img}")

if __name__ == "__main__":
    visualize_comparison()
