"""Spatial projection and plotting used in the chicken-heart velocity panels."""
from pathlib import Path
import anndata as ad
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import scanpy as sc
import scvelo as scv

def canonical_label(value):
    return " ".join(str(value).split())

def plot_daily_velocity_stream_grid(
    adata_dict: dict[str, ad.AnnData],
    velocity_by_time: dict[str, dict[str, np.ndarray]],
    palette: dict[str, str],
    color_key: str,
    component_key: str,
    save_path: Path,
    n_cols: int = 4,
):
    ordered = list(adata_dict.keys())
    n_panels = len(ordered)
    n_rows = int(np.ceil(n_panels / n_cols))

    all_coords = np.concatenate([np.asarray(adata_dict[key].obsm["spatial"], dtype=float) for key in ordered], axis=0)
    x_min, y_min = all_coords.min(axis=0)
    x_max, y_max = all_coords.max(axis=0)
    pad_x = (x_max - x_min) * 0.05
    pad_y = (y_max - y_min) * 0.05
    global_xlim = [x_min - pad_x, x_max + pad_x]
    global_ylim = [y_min - pad_y, y_max + pad_y]

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 6 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    present_labels: list[str] = []
    for ax, time_label in zip(axes, ordered):
        adata_t = adata_dict[time_label]
        coords = np.asarray(adata_t.obsm["spatial"], dtype=np.float32)
        if color_key not in adata_t.obs.columns:
            raise KeyError(f"Velocity coloring requires obs['{color_key}'] for {time_label}.")
        labels = pd.Series(adata_t.obs[color_key].astype(str).map(canonical_label), index=adata_t.obs_names)
        for label in labels.unique().tolist():
            if label not in present_labels:
                present_labels.append(label)

        velocity_high_dim = np.asarray(velocity_by_time[time_label][component_key], dtype=np.float32)
        gene_data = np.asarray(adata_t.X[:, 2:], dtype=np.float32)
        gene_velocity = np.asarray(velocity_high_dim[:, 2:], dtype=np.float32)
        ad_plot = ad.AnnData(X=gene_data)
        ad_plot.obsm["X_spatial"] = coords.copy()
        ad_plot.layers["Ms"] = gene_data.copy()
        ad_plot.layers["velocity"] = gene_velocity.copy()
        ad_plot.obs[color_key] = labels.values
        ad_plot.obs[color_key] = ad_plot.obs[color_key].astype("category")
        cats = list(ad_plot.obs[color_key].cat.categories)
        ad_plot.uns[f"{color_key}_colors"] = [palette.get(cat, "#888888") for cat in cats]
        # Formal native gene-space rendering builds the transition graph in
        # model state space (X), then projects that velocity field back to
        # the observed spatial coordinates for stream plotting.
        sc.pp.neighbors(ad_plot, n_neighbors=30, use_rep="X")
        scv.tl.velocity_graph(ad_plot, vkey="velocity", xkey="Ms", n_jobs=1)
        scv.tl.velocity_embedding(ad_plot, basis="spatial", vkey="velocity")

        scv.pl.velocity_embedding_stream(
            ad_plot,
            basis="spatial",
            color=color_key,
            ax=ax,
            show=False,
            legend_loc="none",
            size=160,
            density=4.8,
            alpha=0.8,
            linewidth=0.45,
            arrowsize=0.6,
            min_mass=0.01,
            smooth=0.2,
            title=time_label,
        )
        ax.set_xlim(global_xlim)
        ax.set_ylim(global_ylim)
        ax.set_aspect("equal", adjustable="box")

    for ax in axes[n_panels:]:
        ax.axis("off")

    legend_handles = [mpatches.Patch(color=palette[label], label=label) for label in present_labels if label in palette]
    fig.legend(handles=legend_handles, title=color_key, loc="center left", bbox_to_anchor=(0.92, 0.5), frameon=False, labelspacing=1.25)
    fig.suptitle({"full": "Total velocity", "drift": "Intrinsic-context velocity", "interaction": "Interaction velocity"}[component_key], fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.9, 0.96])
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    fig.savefig(save_path.with_suffix(".png"), dpi=160, bbox_inches="tight")
    plt.show()
    return fig
