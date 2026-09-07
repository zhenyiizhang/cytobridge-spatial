"""Model evaluation and explicit result adapters for AGIST paper panels.

Velocity evaluation follows ``identify_agist_archives.py`` (11 August 2026):
the legacy base, score-gradient and randomly grouped interaction fields are
summed on the observed rows. Generator fields are reference data, not fitted
predictions. S3 exports the exact display selection in its original renderer.
"""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd


def evaluate_velocity(data_csv, config, model_dir, edge_predictor, *, device="cuda", seed=42):
    """Return fresh model fields in the original CSV row order (cells × 52)."""
    import torch
    from scripts.run_agist_split_sde_replicates import load_models, set_seed
    from reproduction.agist.integration import cal_interaction

    frame = pd.read_csv(data_csv)
    features = frame[[f"x{i}" for i in range(1, 53)]].to_numpy(np.float32)
    times = frame["samples"].to_numpy(np.float32)
    set_seed(seed)
    _, model, score, _, _ = load_models(
        Path(__file__).resolve().parents[2], Path(config), Path(model_dir), device,
        edge_predictor=Path(edge_predictor),
    )
    values = {name: np.empty_like(features) for name in ("base", "score", "interaction")}
    # The deterministic fields may be evaluated in bounded batches without
    # changing the original time-specific interaction populations or RNG order.
    for start in range(0, len(frame), 4096):
        stop = min(start + 4096, len(frame))
        x = torch.tensor(features[start:stop], device=device)
        t = torch.tensor(times[start:stop, None], device=device)
        with torch.no_grad():
            values["base"][start:stop] = model.v_net(t, x).cpu().numpy()
        values["score"][start:stop] = score.compute_gradient(t, x.requires_grad_(True)).detach().cpu().numpy()
    for time in np.unique(times):
        rows = np.flatnonzero(times == time)
        x = torch.tensor(features[rows], device=device)
        log_weight = torch.full((len(rows), 1), -np.log(len(rows)), device=device)
        with torch.no_grad():
            values["interaction"][rows] = cal_interaction(
                x, log_weight, model.interaction_net,
                torch.tensor([float(time)], device=device), m=1024, threshold=1000,
            ).cpu().numpy()
    values["full"] = values["base"] + values["interaction"] + values["score"]
    return frame, values


def velocity_inputs(frame, predicted, reference_fields, clusters, diagnostics):
    """Calculate the S2 cell table from fresh fields and recorded generator fields."""
    from CytoBridge.results.agist_figures import summarize_agist_velocity
    from scripts.build_agist_velocity_time_cluster_breakdown import row_cosine

    with np.load(reference_fields, allow_pickle=False) as reference:
        truth = sum(reference[f"truth_{name}"] for name in ("base", "interaction", "score"))
        if "time" in reference and not np.array_equal(reference["time"], frame["samples"].to_numpy(np.float32)):
            raise ValueError("Generator reference times differ from the simulation CSV row order")
    if truth.shape != predicted["full"].shape:
        raise ValueError("Generator and model velocity arrays must have the same cell × feature shape")
    assignments = pd.read_csv(clusters)
    if not np.array_equal(assignments["row_index"].to_numpy(), np.arange(len(frame))) or not np.array_equal(assignments["time"].to_numpy(), frame["samples"].to_numpy()):
        raise ValueError("State partitions must match the input CSV row order and times")
    cells = assignments[["row_index", "time", "state_cluster"]].copy()
    cells["physical_cosine"] = row_cosine(predicted["full"], truth, slice(0, 2))
    cells["gene_cosine"] = row_cosine(predicted["full"], truth, slice(2, 52))
    if not np.isfinite(cells[["physical_cosine", "gene_cosine"]]).all().all():
        raise ValueError("Undefined velocity cosine: inspect zero-length model/reference vectors")
    return {
        "velocity_per_cell": cells,
        "source_velocity_by_time": summarize_agist_velocity(cells, ("time",)),
        "source_velocity_by_cluster": summarize_agist_velocity(cells, ("state_cluster",)),
        "source_velocity_overall": summarize_agist_velocity(cells),
        "source_velocity_by_time_cluster": summarize_agist_velocity(cells, ("time", "state_cluster")),
        "cluster_diagnostics": pd.read_csv(diagnostics),
    }


def attraction_inputs(observed_h5ad, evaluation_dir):
    """Export all S3 numerical inputs from the preceding benchmark evaluation."""
    import anndata as ad
    observed = ad.read_h5ad(observed_h5ad)
    root = Path(evaluation_dir)
    with np.load(root / "dense_rollout_seed_1.npz", allow_pickle=False) as dense:
        selected = np.sort(np.random.default_rng(11).choice(dense["ground_truth"].shape[1], size=60, replace=False))
        trajectories = {
            "trajectory_time": np.asarray(dense["time_points"]),
            "trajectory_ground_truth": np.asarray(dense["ground_truth"][:, selected]),
            "trajectory_predicted": np.asarray(dense["predicted"][:, selected]),
            "trajectory_indices": selected,
        }
    gene = observed.X.toarray() if hasattr(observed.X, "toarray") else np.asarray(observed.X)
    return {
        "observed_time": observed.obs["samples"].to_numpy(float),
        "observed_spatial": np.asarray(observed.obsm["spatial_aligned"]),
        "observed_gene": gene,
        **trajectories,
        "growth_metrics": pd.read_csv(root / "growth_mass_metrics.csv"),
        "radial_curve": pd.read_csv(root / "interaction_radial_curve.csv"),
        "ablation_metrics": pd.read_csv(root / "interaction_ablation_metrics.csv"),
    }


def collect_agist_inputs(output_dir, *, velocity, attraction):
    """Write a complete S2/S3 input directory; never fill missing numbers from defaults."""
    from CytoBridge.results.agist_figures import load_agist_figures
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    columns = {
        "velocity_per_cell": "s2_velocity_cosine_per_cell.csv.gz",
        "source_velocity_by_time": "s2_velocity_cosine_by_time.csv",
        "source_velocity_by_cluster": "s2_velocity_cosine_by_state_cluster.csv",
        "source_velocity_overall": "s2_velocity_cosine_overall.csv",
        "source_velocity_by_time_cluster": "s2_velocity_cosine_by_time_and_state_cluster.csv",
        "cluster_diagnostics": "s2_cluster_selection_diagnostics.csv",
        "growth_metrics": "s3_growth_mass_metrics.csv",
        "radial_curve": "s3_interaction_radial_curve.csv",
        "ablation_metrics": "s3_interaction_ablation_metrics.csv",
    }
    for key, filename in columns.items():
        (velocity if key in velocity else attraction)[key].to_csv(output / filename, index=False)
    np.savez_compressed(output / "s3_observed_snapshots.npz", time=attraction["observed_time"], spatial=attraction["observed_spatial"], gene=attraction["observed_gene"])
    np.savez_compressed(output / "s3_display_trajectories.npz", time_points=attraction["trajectory_time"], selected_indices=attraction["trajectory_indices"], ground_truth=attraction["trajectory_ground_truth"], predicted=attraction["trajectory_predicted"])
    registry = pd.DataFrame(columns=["figure", "role", "relative_identifier", "availability"])
    registry.to_csv(output / "full_recompute_inputs.csv", index=False)
    # Only non-numerical schema metadata is inherited; every plotted input above
    # is required explicitly and is written from the supplied calculations.
    metadata = Path(__file__).resolve().parents[2] / "CytoBridge/results/data/agist_figures/manifest.json"
    manifest = json.loads(metadata.read_text())
    manifest["generation"] = "fresh model velocity and attraction evaluation, explicit adapter"
    manifest["figures"]["S2"]["calculation"] = "cosine agreement from newly evaluated model fields and recorded generator fields"
    manifest["figures"]["S3"]["calculation"] = "observations and explicit five-seed evaluation outputs converted to plot inputs"
    manifest["full_rerun"]["statement"] = "Model evaluation and input conversion completed; training is not rerun."
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return load_agist_figures(output)


def main_figure_2_from_evaluation(evaluation_dir, *, template=None):
    """Use newly evaluated W2 tables with the selected fixed external-method references."""
    from CytoBridge.results.main_figure_2 import (
        MainFigure2Data, _validate_baselines, _validate_summary, _validate_replicates,
        _validate_summary_matches_replicates,
    )
    root = Path(evaluation_dir)
    summary = _validate_summary(pd.read_csv(root / "w2_mean_sd_ci.csv"), root / "w2_mean_sd_ci.csv")
    replicates = _validate_replicates(pd.read_csv(root / "w2_replicates_long.csv"), root / "w2_replicates_long.csv")
    _validate_summary_matches_replicates(summary, replicates, source=root)
    if template is not None:
        return replace(template, source_dir=root, summary=summary, replicates=replicates)
    source = Path(__file__).resolve().parents[2] / "CytoBridge/results/data/main_figure_2"
    baseline_path = source / "baseline_w2.csv"
    return MainFigure2Data(
        source_dir=root, manifest={"analysis": "main_figure_2", "source": "explicit W2 evaluation"},
        summary=summary, replicates=replicates,
        baselines=_validate_baselines(pd.read_csv(baseline_path), baseline_path),
        # Compatibility field only; draw_distance_panels never opens this file.
        frozen_panels_pdf=source / "frozen_panels_a_to_d.pdf",
    )
