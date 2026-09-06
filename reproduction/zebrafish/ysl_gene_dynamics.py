"""Simulate the initial YSL lineage and draw its gene dynamics in S35."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib as mpl
import numpy as np
import pandas as pd

import CytoBridge as cb
from reproduction.zebrafish.daughter_noise import ObservedSupportLinkPredictor, TIMES, DISPLAY, object_array
from reproduction.zebrafish.plot_gene_dynamics import temporal_zscores
from CytoBridge.results._zebrafish_si_plot import _RC, _render_gene_dynamics


def lineage_expression(adata, points, lineage_ids, weights=None):
    """Inverse-transform each descendant, then average within the initial YSL lineage."""
    loadings = np.asarray(adata.varm["PCs"], dtype=np.float32)
    center = adata.var["pca_center"].to_numpy(dtype=np.float32)
    magnitude = np.max(abs(loadings), axis=1)
    active = magnitude > max(1e-12, 1e-7 * float(magnitude.max()))
    loadings, center = loadings[active], center[active]
    names = adata.var_names.astype(str).to_numpy()[active]
    t0 = np.isclose(adata.obs.time_point_processed.to_numpy(float), 0)
    initial_ysl = adata.obs.loc[t0, "Annotation"].astype(str).eq("Yolk Syncytial Layer").to_numpy()
    rows, counts = [], []
    for index in DISPLAY:
        frame, ids = np.asarray(points[index], dtype=np.float32), np.asarray(lineage_ids[index], dtype=int)
        if len(frame) != len(ids) or np.any(ids < 0) or np.any(ids >= len(initial_ysl)):
            raise ValueError("Lineage IDs do not match the initial population")
        keep = initial_ysl[ids]
        if not keep.any():
            raise ValueError(f"No initial-YSL descendants remain at t={TIMES[index]}")
        expression = np.maximum(frame[keep, 2:] @ loadings.T + center[None, :], 0)
        mass = np.ones(keep.sum()) if weights is None else np.asarray(weights[index], dtype=float).reshape(-1)[keep]
        if not np.isfinite(expression).all() or not np.isfinite(mass).all() or (mass < 0).any() or mass.sum() <= 0:
            raise ValueError("Non-finite expression or invalid population weights")
        mean = np.sum(expression * (mass / mass.sum())[:, None], axis=0, dtype=np.float64)
        rows.append(pd.DataFrame(dict(gene=names, time=TIMES[index], mean_clipped_log1p=mean)))
        counts.append(dict(time=TIMES[index], n_lineage_particles=int(keep.sum())))
    return pd.concat(rows, ignore_index=True), pd.DataFrame(counts)


def run(data_dir, output_dir, seed=42, device="cuda:0", model_dir=None):
    import anndata as ad

    data, output = Path(data_dir), Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    adata = ad.read_h5ad(data / "aligned.h5ad")
    latent = np.asarray(adata.obsm["X_latent"], dtype=np.float32)
    spatial = np.asarray(adata.obsm["spatial_aligned"], dtype=np.float32)
    mask = np.isclose(adata.obs.time_point_processed.to_numpy(float), 0)
    x0 = np.column_stack((spatial[mask], latent[mask])).astype(np.float32)
    loaded = cb.tl.load_dynamical_model_from_dir(
        Path(model_dir) if model_dir is not None else data / "model",
        dim=x0.shape[1], device=device)
    loaded.model.eval()
    runtime = cb.tl.build_dynamical_runtime(loaded)
    interaction = runtime.f_net.interaction_net
    interaction.link_predictor = ObservedSupportLinkPredictor(interaction.link_predictor, latent).to(device)
    cb.tl.set_global_random_seed(seed)
    print("Simulating continuous population weights", flush=True)
    weighted, weights = cb.tl.simulate_sde_points(
        adata=adata, model=loaded.model, dim=x0.shape[1], time_index=0, n_samples=len(x0),
        ts_points=TIMES, dt=.005, sigma=.03, include_score=True, interaction_m=1024,
        device=device, time_key="time_point_processed", obsm_key="X_latent",
        spatial_key="spatial_aligned", concat_spatial=True, interaction_seed=seed+10001, verbose=False)
    cb.tl.set_global_random_seed(seed)
    print("Simulating population resampling", flush=True)
    split, ids = cb.tl.simulate_sde_points_split_from_x0(
        x0=x0, f_net=runtime.f_net, score_net=runtime.score_net, ts_points=TIMES,
        dt=.005, sigma=.03, sigma_by_dim=None, growth_alpha=1., interaction_m=1024,
        device=device, verbose=False, resample_dt=.05, max_particles=100000,
        daughter_noise_std=0., initial_lineage_ids=np.arange(len(x0)), return_lineage_ids=True,
        interaction_seed=seed+10001)
    if not np.array_equal(np.asarray(weighted[0]), np.asarray(split[0])):
        raise ValueError("The two simulations must start with the same ordered cells")
    tables = []
    for arm, points, lineages, mass in (
        ("continuous_weight", weighted, [np.arange(len(x0)) for _ in TIMES], weights),
        ("split_growth_resampling", split, ids, None),
    ):
        table, counts = lineage_expression(adata, points, lineages, mass)
        tables.append(table.assign(arm=arm))
        counts.to_csv(output / f"{arm}_lineage_counts.csv", index=False)
    genes = pd.concat(tables, ignore_index=True)
    genes.to_csv(output / "lineage_gene_expression.csv", index=False)
    matrix = genes.pivot(index="gene", columns=["arm", "time"], values="mean_clipped_log1p")
    # The original comparison selected genes by their larger variance across
    # the two growth representations, then displayed the weighted trajectory.
    variance = pd.concat([matrix[arm].var(axis=1, ddof=0) for arm in matrix.columns.levels[0]], axis=1).max(axis=1)
    top = variance.sort_values(ascending=False).head(250).index
    selected = matrix["continuous_weight"].loc[sorted(top)]
    selected.to_csv(output / "temporal_expression.csv")
    zscores = temporal_zscores(selected)
    zscores.to_csv(output / "temporal_zscores.csv")
    np.savez_compressed(output / "continuous_trajectory.npz", times=TIMES, points=object_array(weighted), weights=weights)
    np.savez_compressed(output / "resampled_trajectory.npz", times=TIMES, points=object_array(split), lineage_ids=object_array(ids))
    (output / "run_summary.json").write_text(json.dumps(dict(
        seed=seed, initial_cells=len(x0), initial_ysl_cells=int(adata.obs.loc[mask, "Annotation"].eq("Yolk Syncytial Layer").sum()),
        dt=.005, sigma=.03, resample_dt=.05, interaction_m=1024, interaction_seed=seed+10001,
        model=str(loaded.weight_path), data=str((data / "aligned.h5ad").resolve()),
    ), indent=2) + "\n")
    with mpl.rc_context(_RC):
        return _render_gene_dynamics(SimpleNamespace(gene_zscores=zscores), output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/zebrafish"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-dir", type=Path)
    args = parser.parse_args()
    print(run(args.data_dir, args.output_dir, args.seed, args.device, args.model_dir))
