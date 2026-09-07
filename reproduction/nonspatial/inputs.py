"""Convert non-spatial numerical analysis outputs to the S4/S5 plot contract.

All quantities are derived from explicitly selected source files. No finished
figure or included numerical result is used to fill an omitted calculation.
The original per-panel algorithms remain in CytoBridge.nonspatial.*_figure.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd


def _frame(path):
    return pd.read_csv(path)


def downloaded_sources(project):
    """Select the published analysis files, explicitly and without plot defaults."""
    data = Path(project) / "data"
    w, s, paper = data / "weinreb", data / "scnt_cortex", data / "nonspatial/paper"
    weinreb = {
        "prepared_h5ad": w / "original/Weinreb_prepared_50pc.h5ad",
        "distribution": w / "frozen_full_vs_corrected_no_interaction/distribution/paired_weighted_distribution_metrics.csv",
        "clone_fate": w / "frozen_full_vs_corrected_no_interaction/clone_fate/frozen_full_vs_corrected_no_interaction_summary.csv",
        "cellchat_joined": paper / "weinreb/cellchat_joined_directed_edges.csv",
        "pathways": paper / "weinreb/cytobridge_pathway_scores.csv",
    }
    scnt = {
        "prepared_h5ad": s / "original/hvg2000/scnt_cortical_full_hvg2000_latent.h5ad",
        "full_trajectory": paper / "scnt/full_paired_dense_trajectory.npz",
        "full_distribution": paper / "scnt/full_distribution_metrics.csv",
        "no_interaction_distribution": s / "frozen_full_vs_corrected_no_interaction/distribution/no_interaction_noise/distribution_metrics.csv",
        "direction": s / "frozen_full_vs_corrected_no_interaction/scnt_direction/timewise_scnt_direction_alignment.csv",
        "exact_message_summary": paper / "scnt/exact_message_summary.csv",
        "network": paper / "scnt/cell_type_interaction_network.csv",
        "cellchat": paper / "scnt/cellchat_edge_summary.csv",
        "pathways": paper / "scnt/cell_type_pathway_scores.csv.gz",
    }
    models = [paper / f"weinreb/lr_seed{seed}/model" for seed in (42, 43, 44)]
    return weinreb, scnt, models


def _pack_cells(cells, labels, *, spring):
    labels = tuple(labels)
    values = cells["cell_type"].astype(str).map({name: i for i, name in enumerate(labels)})
    if values.isna().any():
        raise ValueError("Observed cells contain labels outside the declared paper label set")
    result = {"times": cells["time"].to_numpy(float), "label_id": values.to_numpy(np.int16),
              "label_names": np.asarray(labels), "pc_xy": cells[["pc1", "pc2"]].to_numpy(np.float32)}
    if spring:
        result["spring_xy"] = cells[["spring_x", "spring_y"]].to_numpy(np.float32)
    return result


def observed_cells(prepared_h5ad, *, dataset):
    """Read the measured latent/display coordinates and labels; no resampling."""
    import anndata as ad
    data = ad.read_h5ad(prepared_h5ad)
    pc = np.asarray(data.obsm["X_latent"][:, :2])
    result = pd.DataFrame({"cell_id": data.obs_names.astype(str), "pc1": pc[:, 0], "pc2": pc[:, 1]})
    if dataset == "weinreb":
        result["time"] = data.obs["Time point"].to_numpy(float)
        result["cell_type"] = data.obs["Cell type annotation"].astype(str).to_numpy()
        result["spring_x"] = data.obs["SPRING-x"].to_numpy(float)
        result["spring_y"] = data.obs["SPRING-y"].to_numpy(float)
    elif dataset == "scnt":
        result["time"] = data.obs["time_point_processed"].to_numpy(float)
        result["cell_type"] = data.obs["cell_type"].astype(str).to_numpy()
    else:
        raise ValueError("dataset must be weinreb or scnt")
    return result


def collect_nonspatial_inputs(weinreb, scnt, output_dir):
    """Build S4/S5 from analysis outputs selected by the two source dictionaries.

    Source keys are enumerated in ``reproduction/nonspatial/source_keys.json``.
    Paths may point directly to fresh command outputs; callers need not rename
    or copy the large H5AD, trajectory, or full cell-pair tables.
    """
    from CytoBridge.nonspatial import weinreb_figure as w, scnt_figure as s
    from CytoBridge.results.nonspatial_figures import load_nonspatial_figures, _FILES

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    wc = observed_cells(weinreb["prepared_h5ad"], dataset="weinreb")
    sc = observed_cells(scnt["prepared_h5ad"], dataset="scnt")
    np.savez_compressed(output / "weinreb_observed_cells.npz", **_pack_cells(wc, w.CELL_ORDER, spring=True))
    np.savez_compressed(output / "scnt_observed_cells.npz", **_pack_cells(sc, s.CELL_ORDER, spring=False))
    for dataset, colors in (("weinreb", w.CELL_COLORS), ("scnt", s.CELL_COLORS)):
        (output / f"{dataset}_cell_colors.json").write_text(json.dumps(colors, indent=2) + "\n")

    # S4b: select only the fields actually displayed from the newly evaluated
    # three-checkpoint ensemble. No placeholder grids are supplied.
    with np.load(weinreb["model_grids"], allow_pickle=False) as all_grids:
        grids = {}
        for day in (2, 4, 6):
            for suffix in ("x_axis", "y_axis", "support_mask", "lr_full_drift_u", "lr_full_drift_v", "lr_full_drift_speed"):
                grids[f"day{day}_{suffix}"] = all_grids[f"day{day}_{suffix}"]
        for suffix in ("u", "v", "speed"):
            grids[f"day6_lr_interaction_{suffix}"] = all_grids[f"day6_lr_interaction_{suffix}"]
    grids["spring_xlim"], grids["spring_ylim"] = w.spring_limits(wc)
    np.savez_compressed(output / "weinreb_model_fields.npz", **grids)
    wd = _frame(weinreb["distribution"])
    if not {"w1_relative_change", "w2_relative_change", "tmv_relative_change"}.issubset(wd):
        for metric in ("w1", "w2", "tmv"):
            column = "tmv_absolute" if metric == "tmv" else metric
            wd[f"{metric}_relative_change"] = (wd[f"{column}_no_interaction"] - wd[f"{column}_full"]) / wd[f"{column}_full"]
    wd.to_csv(output / "weinreb_distribution.csv", index=False)
    _frame(weinreb["clone_fate"]).to_csv(output / "weinreb_clone_fate.csv", index=False)
    joined = _frame(weinreb["cellchat_joined"])
    wedges = w.day6_network_edges(joined)
    wedges.to_csv(output / "weinreb_network_edges.csv", index=False)
    wnodes = wc[np.isclose(wc.time, 6)].groupby("cell_type").size().reindex(w.CELL_ORDER).rename("n_cells").reset_index()
    wnodes.to_csv(output / "weinreb_network_nodes.csv", index=False)
    # Correlations are recomputed from all directed type-pair scores, not read
    # from an already summarized association table.
    from scipy.stats import kendalltau, spearmanr
    concordance = []
    for day, block in joined.groupby("time", sort=True):
        block = block[block["min10_eligible"].astype(bool)]
        a, b = block["message_D_AB_raw"], block["cellchat_native_raw"]
        concordance.append({"day": day, "kendall_tau_b": kendalltau(a, b).statistic,
                            "n_contexts": len(block), "spearman_rho": spearmanr(a, b).statistic})
    wcorr = pd.DataFrame(concordance)
    wcorr.to_csv(output / "weinreb_concordance.csv", index=False)
    w.select_day6_cytobridge_pathways(_frame(weinreb["pathways"])).to_csv(output / "weinreb_pathways.csv", index=False)
    wm = {"distribution": {"equal_weight_W1_W2_error_increase_after_removal_pct": float(100 * np.r_[wd.w1_relative_change, wd.w2_relative_change].mean())},
          "cellchat": {"displayed_spearman_rho": float(wcorr.loc[np.isclose(wcorr.day, 6), "spearman_rho"].iloc[0]),
                       "mean_spearman_rho_equal_time": float(wcorr.spearman_rho.mean())}}
    (output / "weinreb_metrics.json").write_text(json.dumps(wm, indent=2) + "\n")

    # S5b uses the original finite-difference trajectory field and the exact
    # receiver-type message sums, not a new alternative projection.
    xlim, ylim = s.pc_limits(sc)
    summary = _frame(scnt["exact_message_summary"])
    grids = {"pc_xlim": xlim, "pc_ylim": ylim, "full_times": np.array([0., .5, 1.])}
    with np.load(scnt["full_trajectory"], allow_pickle=False) as trajectory:
        for index, time in enumerate(grids["full_times"]):
            values = s.trajectory_field(trajectory, time, xlim, ylim)
            grids.update({f"full{index}_{name}": value for name, value in zip(("x", "y", "u", "v", "mask"), values)})
    values = s.interaction_field(sc, summary, xlim, ylim)
    grids.update({f"interaction_{name}": value for name, value in zip(("x", "y", "u", "v", "mask"), values[:5])})
    values[5].to_csv(output / "scnt_interaction_vectors.csv", index=False)
    np.savez_compressed(output / "scnt_model_fields.npz", **grids)
    full = _frame(scnt["full_distribution"])
    no = _frame(scnt["no_interaction_distribution"])
    cols = ["time", "w1", "w2", "tmv_absolute"]
    full[cols].merge(no[cols], on="time", validate="one_to_one", suffixes=("_full", "_no_interaction")).to_csv(output / "scnt_distribution.csv", index=False)
    direction = _frame(scnt["direction"])
    direction.to_csv(output / "scnt_direction.csv", index=False)
    network, cellchat = _frame(scnt["network"]), _frame(scnt["cellchat"])
    sedges = s.select_network_edges(network, cellchat)
    sedges.to_csv(output / "scnt_network_edges.csv", index=False)
    rho, _, pairs = s.network_concordance(network, cellchat)
    pairs.to_csv(output / "scnt_network_pairs.csv", index=False)
    sc[np.isclose(sc.time, 2)].groupby("cell_type").size().reindex(s.CELL_ORDER).rename("n_cells").reset_index().to_csv(output / "scnt_network_nodes.csv", index=False)
    s.select_pathways(_frame(scnt["pathways"])).to_csv(output / "scnt_pathways.csv", index=False)
    primary = direction[direction.condition.isin(["full_interaction_noise", "no_interaction_noise"])]
    pivot = primary.pivot(index="time_hours", columns="condition", values=["cell_cosine_mean", "cell_cosine_median"])
    mean = primary.groupby("condition")["cell_cosine_mean"].mean()
    wins = {name: int((pivot[(f"cell_cosine_{name}", "full_interaction_noise")] > pivot[(f"cell_cosine_{name}", "no_interaction_noise")]).sum()) for name in ("mean", "median")}
    sm = {"network": {"spearman_rho_all_81_pairs": rho}, "direction_evaluation": {
        "equal_endpoint_average": {"mean_cellwise_cosine_full": float(mean["full_interaction_noise"]), "mean_cellwise_cosine_no_interaction": float(mean["no_interaction_noise"])}, "endpoint_wins_full": wins}}
    (output / "scnt_metrics.json").write_text(json.dumps(sm, indent=2) + "\n")

    manifest = {"schema_version": 1, "analysis": "grouped_nonspatial_s4_s5", "figures": {"s4": "Weinreb", "s5": "scNT"},
                "files": {name: {} for name in _FILES if name != "manifest.json"},
                "source_files": {"weinreb": {k: str(v) for k, v in weinreb.items()}, "scnt": {k: str(v) for k, v in scnt.items()}}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    external = {"schema_version": 1, "path_base": "project root", "datasets": {
        "weinreb": {"renderer": "CytoBridge/nonspatial/weinreb_figure.py", "full_rerun_inputs": []},
        "scnt": {"renderer": "CytoBridge/nonspatial/scnt_figure.py", "full_rerun_inputs": []}}}
    (output / "external_inputs.json").write_text(json.dumps(external, indent=2) + "\n")
    return load_nonspatial_figures(output)
