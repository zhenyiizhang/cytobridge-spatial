#!/usr/bin/env python3
"""Compute the shared corrected MOSTA numerical state for SI Figures S4--S10.

The script is deliberately calculation-only.  It uses the current CytoBridge
package API, starts once at t0, saves fully generated native-coordinate states
at all 13 quarter-step times, and never applies a display warp.  Plotting is a
separate local step that reuses the submitted SI/notebook visual grammar.

The output directory is immutable by construction: it must not exist at start,
and is sealed read-only only after every numerical gate and SHA-256 manifest
has been written successfully.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import torch

import CytoBridge as cb


TIMES = tuple(float(value) for value in np.arange(0.0, 3.0001, 0.25))
OBSERVED_TIMES = (0.0, 1.0, 2.0, 3.0)
INTERMEDIATE_TIMES = tuple(value for value in TIMES if value not in OBSERVED_TIMES)
S4_TIMES = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
S5_DISPLAY_TIMES = tuple(value for value in TIMES if not np.isclose(value, 1.5))
S7_TIMES = S4_TIMES
EXPECTED_PACKAGE_COMMIT = "2b3c79eff3face7c4dd33de24d45384b9dbd8a84"
EXPECTED_FINETUNE_SHA256 = "d08d21b86fdcd92c748bb54ca81f998fbb157ca5a7acd9548bbfe16c573bfaa5"
EXPECTED_SCORE_SHA256 = "d7d06657f8548618db1bc85409e73305fbf59feb9446793550b2c9761639e52a"
EXPECTED_CLASSIFIER_SHA256 = "f938575c145baa9002de695c39a8637f57d6cb3a06ccfaf4d18b707ca962a7e0"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": sha256(path),
        "size_bytes": int(path.stat().st_size),
    }


def package_release_identity(expected_commit: str) -> dict[str, Any]:
    """Read immutable release markers adjacent to the imported package."""
    package_root = Path(cb.__file__).resolve().parent.parent
    commit_path = package_root / "RELEASE_COMMIT"
    archive_path = package_root / "ARCHIVE_SHA256"
    if not commit_path.is_file():
        raise FileNotFoundError(f"Missing package release marker: {commit_path}")
    detected_commit = commit_path.read_text(encoding="utf-8").strip()
    if detected_commit != expected_commit:
        raise RuntimeError(
            "Imported package release mismatch: "
            f"expected={expected_commit}, detected={detected_commit}, root={package_root}"
        )
    archive_sha = archive_path.read_text(encoding="utf-8").strip() if archive_path.is_file() else None
    return {
        "package_root": str(package_root),
        "module": str(Path(cb.__file__).resolve()),
        "release_commit_marker": identity(commit_path),
        "detected_commit": detected_commit,
        "archive_sha256_marker": identity(archive_path) if archive_path.is_file() else None,
        "archive_sha256": archive_sha,
    }


def safe_time(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aligned-h5ad", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--classifier-cache-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--package-commit", default=EXPECTED_PACKAGE_COMMIT)
    parser.add_argument("--expected-input-sha256", required=True)
    parser.add_argument("--n-samples", type=int, default=50000, choices=(50000,))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def original_hvg_candidates(reference: ad.AnnData) -> tuple[list[str], dict[str, Any]]:
    if "highly_variable" not in reference.var:
        raise KeyError("Corrected MOSTA H5AD lacks var['highly_variable']")
    preprocess = dict(reference.uns.get("preprocess_info", {}))
    additions = set(map(str, preprocess.get("required_latent_features_added", [])))
    union = [
        str(name)
        for name in reference.var_names[reference.var["highly_variable"].astype(bool).to_numpy()]
    ]
    candidates = [name for name in union if name not in additions]
    expected = int(preprocess.get("n_top_genes", 2000))
    if expected != 2000 or len(candidates) != 2000:
        raise RuntimeError(
            "S8 original-HVG contract failed: "
            f"preprocess n_top_genes={expected}, recovered={len(candidates)}"
        )
    candidate_sha = hashlib.sha256(
        json.dumps(sorted(candidates), separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return candidates, {
        "source": "var['highly_variable'] minus required_latent_features_added",
        "n_hvg_union": len(union),
        "n_required_additions": len(additions),
        "n_original_hvgs": len(candidates),
        "sha256_sorted_json_utf8": candidate_sha,
    }


def representative_program_genes(
    expression: pd.DataFrame,
    normalized_profiles: pd.DataFrame,
    assignments: pd.DataFrame,
    *,
    n_per_program: int = 5,
) -> pd.DataFrame:
    """Historical S8 ranking: temporal variance plus prototype correlation."""
    assignment = assignments.rename(columns={"profile": "gene"}).copy()
    rows = []
    for cluster, subset in assignment.groupby("cluster", sort=True):
        genes = [
            gene
            for gene in subset["gene"].astype(str)
            if gene in normalized_profiles.index
        ]
        profiles = normalized_profiles.loc[genes]
        prototype = profiles.mean(axis=0).to_numpy(dtype=float)
        for gene in genes:
            vector = profiles.loc[gene].to_numpy(dtype=float)
            corr = float(np.corrcoef(vector, prototype)[0, 1]) if np.std(vector) > 0 else 0.0
            rows.append(
                {
                    "gene": gene,
                    "program": int(cluster),
                    "temporal_variance": float(expression.loc[gene].var(ddof=0)),
                    "prototype_correlation": corr,
                }
            )
    table = pd.DataFrame(rows)
    table["variance_rank"] = table.groupby("program")["temporal_variance"].rank(
        ascending=False, method="min"
    )
    table["correlation_rank"] = table.groupby("program")["prototype_correlation"].rank(
        ascending=False, method="min"
    )
    table["combined_rank"] = table["variance_rank"] + table["correlation_rank"]
    table = table.sort_values(
        ["program", "combined_rank", "prototype_correlation", "temporal_variance"],
        ascending=[True, True, False, False],
        kind="stable",
    )
    return table.groupby("program", sort=True).head(int(n_per_program)).reset_index(drop=True)


def build_lineage_tables(
    labels_by_time: list[np.ndarray],
    *,
    time_points: tuple[float, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if len(labels_by_time) != len(TIMES):
        raise RuntimeError("Non-split label count does not match dense time grid")
    index_by_time = {float(value): index for index, value in enumerate(TIMES)}
    selected = [np.asarray(labels_by_time[index_by_time[t]]).astype(str) for t in time_points]
    sizes = {int(values.shape[0]) for values in selected}
    if sizes != {50000}:
        raise RuntimeError(f"Fixed-particle lineage must retain 50,000 rows; found {sizes}")
    long_frames = []
    node_rows = []
    for time_value, labels in zip(time_points, selected):
        particle_ids = np.arange(labels.shape[0], dtype=np.int64)
        long_frames.append(
            pd.DataFrame({"particle_id": particle_ids, "time": float(time_value), "celltype": labels})
        )
        values, counts = np.unique(labels, return_counts=True)
        for value, count in zip(values, counts):
            node_rows.append({"time": float(time_value), "celltype": str(value), "count": int(count)})
    edge_rows = []
    for source_time, target_time, source_labels, target_labels in zip(
        time_points[:-1], time_points[1:], selected[:-1], selected[1:]
    ):
        pairs = pd.DataFrame({"source": source_labels, "target": target_labels})
        grouped = pairs.groupby(["source", "target"], sort=True).size().rename("count").reset_index()
        source_totals = grouped.groupby("source")["count"].transform("sum")
        grouped["source_fraction"] = grouped["count"] / source_totals
        grouped.insert(0, "target_time", float(target_time))
        grouped.insert(0, "source_time", float(source_time))
        edge_rows.append(grouped)
    return (
        pd.concat(long_frames, ignore_index=True),
        pd.DataFrame(node_rows),
        pd.concat(edge_rows, ignore_index=True),
    )


def seal_output(output: Path) -> None:
    checksum_paths = sorted(
        path
        for path in output.rglob("*")
        if path.is_file() and path.name not in {"SHA256SUMS.txt", "COMPLETE"}
    )
    (output / "SHA256SUMS.txt").write_text(
        "".join(f"{sha256(path)}  {path.relative_to(output)}\n" for path in checksum_paths),
        encoding="utf-8",
    )
    (output / "COMPLETE").write_text("complete\n", encoding="utf-8")
    for path in output.rglob("*"):
        if path.is_file():
            os.chmod(path, 0o444)
    for path in sorted((item for item in output.rglob("*") if item.is_dir()), reverse=True):
        os.chmod(path, 0o555)
    os.chmod(output, 0o555)


def main() -> int:
    args = parse_args()
    started = time.time()
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    for name in [
        "generated_states",
        "s4",
        "s5_growth",
        "s6_composition",
        "s7_lineage",
        "s8_gene_programs",
        "s10_developmental_wave",
        "provenance",
    ]:
        (output / name).mkdir()
    shutil.copy2(Path(__file__).resolve(), output / "provenance" / Path(__file__).name)

    aligned = Path(args.aligned_h5ad).expanduser().resolve()
    model_dir = Path(args.model_dir).expanduser().resolve()
    classifier_cache = Path(args.classifier_cache_path).expanduser().resolve()
    for path in [aligned, model_dir, classifier_cache]:
        if not path.exists():
            raise FileNotFoundError(path)
    if args.package_commit != EXPECTED_PACKAGE_COMMIT:
        raise RuntimeError(f"Package commit mismatch: {args.package_commit}")
    package_identity = package_release_identity(args.package_commit)
    aligned_sha = sha256(aligned)
    if aligned_sha != args.expected_input_sha256:
        raise RuntimeError(f"Aligned H5AD hash mismatch: {aligned_sha}")
    finetune = model_dir / "Finetune" / "best_model.pth"
    score = model_dir / "Score_Refine" / "score_model.pth"
    if sha256(finetune) != EXPECTED_FINETUNE_SHA256:
        raise RuntimeError("Finetune checkpoint hash mismatch")
    if sha256(score) != EXPECTED_SCORE_SHA256:
        raise RuntimeError("Score checkpoint hash mismatch")
    if sha256(classifier_cache) != EXPECTED_CLASSIFIER_SHA256:
        raise RuntimeError("Classifier cache hash mismatch")

    reference = ad.read_h5ad(aligned)
    frame, resolved_time_key = cb.tl.adata_to_aligned_dataframe(
        reference,
        time_key="time_point_processed",
        obsm_key="X_latent",
        spatial_key="spatial_aligned",
        concat_spatial=True,
        annotation_key="Annotation",
    )
    feature_columns = cb.tl.infer_feature_columns(frame, annotation_column="Annotation")
    if len(feature_columns) != 52:
        raise RuntimeError(f"Expected 52 model features, found {len(feature_columns)}")
    observed_from_data = tuple(sorted(map(float, frame["samples"].unique())))
    if observed_from_data != OBSERVED_TIMES:
        raise RuntimeError(f"Observed time mismatch: {observed_from_data}")

    loaded = cb.tl.load_dynamical_model_from_dir(model_dir, dim=52, device=args.device)
    runtime = cb.tl.build_dynamical_runtime(loaded)
    if getattr(loaded, "weight_stage", None) != "Finetune":
        raise RuntimeError("Loaded model did not select Finetune weights")
    if getattr(loaded, "score_stage", None) != "Score_Refine":
        raise RuntimeError("Loaded model did not select Score_Refine weights")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    result = cb.tl.run_interpolation_workflow(
        df=frame,
        dim=52,
        annotation_key="Annotation",
        runtime=runtime,
        device=args.device,
        output_dir=str(output / "workflow"),
        requested_plot_points=TIMES,
        interp_time_points=INTERMEDIATE_TIMES,
        max_observed_timepoints=4,
        use_real_for_observed=True,
        classifier_cache_path=str(classifier_cache),
        classifier_cache_dir=str(output / "classifier_cache_unused"),
        classifier_adata=reference,
        classifier_time_key=resolved_time_key,
        classifier_obsm_key="X_latent",
        classifier_spatial_key="spatial_aligned",
        classifier_concat_spatial=True,
        classifier_epochs=500,
        classifier_hidden_size=128,
        classifier_lr=0.001,
        classifier_best_metric="bacc",
        classifier_strict_stratification=True,
        classifier_knn_neighbors=10,
        sde_n_samples=int(args.n_samples),
        skip_nonsplit_sde=False,
        sde_dt=0.05,
        split_sde_dt=0.05,
        split_sigma_scalar=0.03,
        split_daughter_noise_std=0.0,
        split_growth_alpha=1.0,
        split_interaction_m=1024,
        split_resample_dt=None,
        split_max_particles=None,
        split_sde_piecewise=False,
        split_sde_piecewise_include_end=False,
        piecewise_observed_sample_mode="t0_fixed",
        spatial_warp_to_observed=False,
        spatial_warp_to_observed_piecewise=False,
        spatial_warp_visualization_only=False,
        slice_max_cells_per_timepoint=None,
        random_seed=int(args.seed),
    )
    if tuple(map(float, result.ts_points)) != TIMES:
        raise RuntimeError(f"Dense time grid mismatch: {result.ts_points}")
    points = result.sde_points_split_prewarp
    if points is None:
        points = result.sde_points_split
    labels = result.predicted_labels_split_prewarp
    if labels is None:
        labels = result.slice_labels_split
    if points is None or labels is None:
        raise RuntimeError("Fully generated split-SDE state or labels are missing")
    if len(points) != len(TIMES) or len(labels) != len(TIMES):
        raise RuntimeError("Generated state length does not match dense time grid")

    generated: dict[str, ad.AnnData] = {}
    state_rows = []
    for index, time_value in enumerate(TIMES):
        state = np.asarray(points[index], dtype=np.float32)
        label = np.asarray(labels[index]).astype(str)
        if state.ndim != 2 or state.shape[1] != 52 or state.shape[0] != label.shape[0]:
            raise RuntimeError(f"Generated state/label mismatch at t={time_value}")
        if not np.isfinite(state).all():
            raise RuntimeError(f"Non-finite generated state at t={time_value}")
        data = ad.AnnData(X=state)
        data.obs["Annotation"] = label
        data.obsm["spatial"] = state[:, :2].copy()
        data.uns["slice_origin"] = "generated_global_t0"
        data.uns["source_anchor_time"] = 0.0
        data.uns["spatial_warp"] = False
        key = str(float(time_value))
        generated[key] = data
        path = output / "generated_states" / f"time_{safe_time(time_value)}.h5ad"
        data.write_h5ad(path)
        state_rows.append(
            {
                "time": float(time_value),
                "origin": "generated_global_t0",
                "source_anchor_time": 0.0,
                "spatial_warp": False,
                "n_cells": int(data.n_obs),
                "n_labels": int(pd.Series(label).nunique()),
                "path": str(path),
                "sha256": sha256(path),
            }
        )
    pd.DataFrame(state_rows).to_csv(output / "generated_states" / "state_inventory.csv", index=False)

    t0_rows = frame.loc[np.isclose(frame["samples"].to_numpy(dtype=float), 0.0)].copy()
    observed_t0 = ad.AnnData(X=t0_rows[feature_columns].to_numpy(dtype=np.float32))
    observed_t0.obs["Annotation"] = t0_rows["Annotation"].astype(str).to_numpy()
    observed_t0.obsm["spatial"] = observed_t0.X[:, :2].copy()
    observed_t0.uns["slice_origin"] = "observed_real"
    observed_t0.uns["source_anchor_time"] = 0.0
    observed_t0.uns["spatial_warp"] = False
    observed_t0_path = output / "s4" / "observed_t0.h5ad"
    observed_t0.write_h5ad(observed_t0_path)

    composition = cb.tl.summarize_label_composition(labels, TIMES)
    composition.to_csv(output / "s6_composition" / "celltype_composition_fully_generated.csv", index=False)
    total_by_time = composition.groupby("time", sort=True)["total"].first()
    if (
        len(total_by_time) != 13
        or int(total_by_time.iloc[0]) != 50000
        or bool((total_by_time <= 0).any())
    ):
        raise RuntimeError("Fully generated composition total contract failed")

    growth = cb.tl.evaluate_growth_by_timepoint(
        generated,
        loaded.model,
        time_points=TIMES,
        time_keys=[str(float(value)) for value in TIMES],
        annotation_key="Annotation",
        spatial_key="spatial",
        device=args.device,
    )
    growth.to_csv(output / "s5_growth" / "growth_by_cell_fully_generated.csv", index=False)
    brain_growth = growth.loc[growth["celltype"].astype(str) == "Brain"].copy()
    brain_growth_stats = brain_growth.groupby("time", sort=True)["growth"].agg(
        ["count", "mean", "median", "min", "max"]
    )
    if (
        len(brain_growth_stats) != len(TIMES)
        or tuple(map(float, brain_growth_stats.index)) != TIMES
        or bool((brain_growth_stats["count"] <= 0).any())
    ):
        raise RuntimeError("S5 Brain growth must cover all 13 generated states")
    brain_growth_stats.to_csv(output / "s5_growth" / "brain_growth_summary.csv")
    growth_vmin = float(np.percentile(brain_growth["growth"], 5))
    growth_vmax = float(np.percentile(brain_growth["growth"], 95))
    if not np.isfinite([growth_vmin, growth_vmax]).all() or growth_vmin >= growth_vmax:
        raise RuntimeError(f"Invalid S5 common growth scale: {growth_vmin}, {growth_vmax}")
    growth_contract = {
        "time_grid": list(TIMES),
        "display_times": list(S5_DISPLAY_TIMES),
        "dropped_from_display_only": 1.5,
        "common_scale_source": "Brain growth values over all 13 generated states",
        "common_scale_percentiles": [5.0, 95.0],
        "vmin": growth_vmin,
        "vmax": growth_vmax,
        "state_source": "generated_global_t0",
        "spatial_warp": False,
    }
    (output / "s5_growth" / "growth_contract.json").write_text(
        json.dumps(growth_contract, indent=2) + "\n", encoding="utf-8"
    )

    if result.predicted_labels_list is None:
        raise RuntimeError("Non-split fixed-particle labels are missing")
    lineage_long, lineage_nodes, lineage_edges = build_lineage_tables(
        list(result.predicted_labels_list), time_points=S7_TIMES
    )
    edge_totals = lineage_edges.groupby(["source_time", "target_time"], sort=True)["count"].sum()
    if len(edge_totals) != len(S7_TIMES) - 1 or set(map(int, edge_totals)) != {50000}:
        raise RuntimeError(f"S7 fixed-particle flow conservation failed: {edge_totals.to_dict()}")
    lineage_long.to_csv(output / "s7_lineage" / "fixed_particle_labels.csv.gz", index=False, compression="gzip")
    lineage_nodes.to_csv(output / "s7_lineage" / "lineage_nodes.csv", index=False)
    lineage_edges.to_csv(output / "s7_lineage" / "lineage_edges.csv", index=False)
    (output / "s7_lineage" / "lineage_contract.json").write_text(
        json.dumps(
            {
                "trajectory_mode": "global_t0_non_split_fixed_particle",
                "n_particles": 50000,
                "time_points": list(S7_TIMES),
                "particle_id_persistent": True,
                "restart_from_observed_anchor": False,
                "classifier_k": 10,
                "spatial_warp": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    candidates, candidate_contract = original_hvg_candidates(reference)
    brain_slices: dict[str, ad.AnnData] = {}
    for time_value in TIMES:
        key = str(float(time_value))
        mask = generated[key].obs["Annotation"].astype(str).to_numpy() == "Brain"
        if not np.any(mask):
            raise RuntimeError(f"No Brain cells at generated t={time_value}")
        brain_slices[key] = generated[key][mask].copy()
    gene = cb.tl.summarize_temporal_gene_patterns(
        brain_slices,
        reference,
        time_points=TIMES,
        spatial_dim=2,
        reference_layer=None,
        n_top_genes=len(candidates),
        n_cluster_genes=len(candidates),
        n_clusters=2,
        candidate_features=candidates,
        preferred_species_tag=None,
        profile_normalization="zscore",
        profile_linkage_method="ward",
        profile_cluster_order="peak_time",
        active_features_only=True,
        clip_min=0.0,
        reconstruction_batch_size=4096,
    )
    clusters = sorted(gene.clustering.assignments["cluster"].astype(int).unique().tolist())
    if clusters != [1, 2] or int(gene.expression.shape[0]) != 2000:
        raise RuntimeError(
            f"S8 Brain Ward-k2 contract failed: clusters={clusters}, expression={gene.expression.shape}"
        )
    representatives = representative_program_genes(
        gene.expression,
        gene.clustering.normalized_profiles,
        gene.clustering.assignments,
        n_per_program=5,
    )
    if representatives.groupby("program").size().to_dict() != {1: 5, 2: 5}:
        raise RuntimeError("S8 representative-gene contract failed")
    gene.expression.to_csv(output / "s8_gene_programs" / "brain_hvg_mean_log1p_by_time.csv")
    gene.signed_expression.to_csv(output / "s8_gene_programs" / "brain_hvg_signed_mean_by_time.csv")
    gene.top_variable_genes.to_csv(output / "s8_gene_programs" / "brain_hvg_temporal_variance_rank.csv", index=False)
    gene.top_variable_genes.head(20).to_csv(
        output / "s8_gene_programs" / "brain_top20_temporal_variable_genes.csv", index=False
    )
    gene.gene_name_map.to_csv(output / "s8_gene_programs" / "brain_hvg_gene_name_map.csv", index=False)
    gene.reconstruction_diagnostics.to_csv(output / "s8_gene_programs" / "reconstruction_diagnostics.csv", index=False)
    gene.clustering.normalized_profiles.to_csv(output / "s8_gene_programs" / "brain_hvg_gene_wise_zscore.csv")
    gene.clustering.assignments.to_csv(output / "s8_gene_programs" / "brain_hvg_ward_k2_assignments.csv", index=False)
    gene.clustering.prototypes.to_csv(output / "s8_gene_programs" / "brain_hvg_ward_k2_prototypes.csv", index=False)
    gene.clustering.diagnostics.to_csv(output / "s8_gene_programs" / "brain_hvg_ward_k2_diagnostics.csv", index=False)
    representatives.to_csv(output / "s8_gene_programs" / "brain_program_representative_genes_top5.csv", index=False)
    (output / "s8_gene_programs" / "s8_gene_program_settings.json").write_text(
        json.dumps(
            {
                "package_settings": jsonable(dict(gene.settings)),
                "candidate_contract": candidate_contract,
                "roi": "Brain",
                "state_source": "fully_generated_global_t0_all_13_times",
                "legacy_published_program_sizes_reference_only": [654, 1346],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    wave = cb.tl.analyze_developmental_wave(
        gene.expression,
        n_top_profiles=1000,
        n_phases=3,
        min_phase_size=5,
        standardization="zscore",
    )
    phase_sizes = wave.assignments.groupby("phase", sort=True).size().to_dict()
    if sorted(map(int, phase_sizes)) != [1, 2, 3]:
        raise RuntimeError(f"S10 DP3 contract failed: {phase_sizes}")
    wave.selected_profiles.to_csv(output / "s10_developmental_wave" / "s10_top1000_temporal_variable_genes.csv")
    wave.standardized_profiles.to_csv(output / "s10_developmental_wave" / "s10_top1000_gene_wise_zscore.csv")
    wave.ordered_profiles.to_csv(output / "s10_developmental_wave" / "s10_top1000_peak_ordered_profiles.csv")
    wave.assignments.to_csv(output / "s10_developmental_wave" / "s10_top1000_dp3_assignments.csv", index=False)
    wave.prototypes.to_csv(output / "s10_developmental_wave" / "s10_top1000_dp3_prototypes.csv", index=False)
    wave.diagnostics.to_csv(output / "s10_developmental_wave" / "s10_top1000_dp3_diagnostics.csv", index=False)
    (output / "s10_developmental_wave" / "s10_developmental_wave_settings.json").write_text(
        json.dumps(
            {
                "package_settings": jsonable(dict(wave.settings)),
                "phase_sizes": {str(key): int(value) for key, value in phase_sizes.items()},
                "legacy_published_phase_sizes_reference_only": [365, 417, 218],
                "hardcoded_legacy_phase_sizes": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    summary = {
        "schema_version": 1,
        "status": "complete",
        "purpose": "Shared corrected numerical truth for MOSTA SI Figures S4-S10",
        "dataset": "mosta",
        "package_commit": args.package_commit,
        "package_release": package_identity,
        "model": {
            "directory": str(model_dir),
            "weight_stage": "Finetune",
            "score_stage": "Score_Refine",
            "finetune": identity(finetune),
            "score": identity(score),
        },
        "aligned_h5ad": {**identity(aligned), "shape": list(reference.shape)},
        "classifier": {
            **identity(classifier_cache),
            "k": 10,
            "accuracy": result.classifier_accuracy,
            "balanced_accuracy": result.classifier_balanced_accuracy,
            "metadata": jsonable(result.classifier_metadata),
            "evaluation": jsonable(result.classifier_evaluation),
        },
        "trajectory": {
            "mode": "global_t0_extrapolation",
            "restart_from_preceding_observed_stage": False,
            "time_points": list(TIMES),
            "n_initial": int(args.n_samples),
            "split_state_source": "native sde_points_split (no warp requested or applied)",
            "split_sde": {"dt": 0.05, "sigma": 0.03, "growth_alpha": 1.0, "daughter_noise_std": 0.0},
            "lineage_state": "non_split_fixed_particle",
            "spatial_warp": False,
            "simulation_seeds": jsonable(result.simulation_seeds),
        },
        "workflow_parameters": {
            "requested_plot_points": list(TIMES),
            "interp_time_points": list(INTERMEDIATE_TIMES),
            "use_real_for_observed": True,
            "final_state_extraction": (
                "sde_points_split_prewarp when present, otherwise sde_points_split; "
                "never adata_dict observed substitution"
            ),
            "classifier_knn_neighbors": 10,
            "sde_n_samples": int(args.n_samples),
            "skip_nonsplit_sde": False,
            "sde_dt": 0.05,
            "split_sde_dt": 0.05,
            "split_sigma_scalar": 0.03,
            "split_daughter_noise_std": 0.0,
            "split_growth_alpha": 1.0,
            "split_interaction_m": 1024,
            "split_resample_dt": None,
            "split_max_particles": None,
            "split_sde_piecewise": False,
            "piecewise_observed_sample_mode": "t0_fixed",
            "spatial_warp_to_observed": False,
            "spatial_warp_to_observed_piecewise": False,
            "spatial_warp_visualization_only": False,
            "random_seed": int(args.seed),
        },
        "panels": {
            "S4": {"times": list(S4_TIMES), "all_after_leftmost_observed_are_generated": True},
            "S5": growth_contract,
            "S6": {"state_source": "fully_generated_global_t0", "time_points": list(TIMES)},
            "S7": {"state_source": "global_t0_non_split_fixed_particle", "time_points": list(S7_TIMES)},
            "S8_S9_S10": {"roi": "Brain", "original_hvgs": candidate_contract, "ward_clusters": 2, "dp_phases": 3},
        },
        "generated_states": state_rows,
        "generated_totals": {str(float(key)): int(value) for key, value in total_by_time.items()},
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": args.device,
            "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "cuda_peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None,
            "cuda_peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else None,
            "cytobridge_module": str(Path(cb.__file__).resolve()),
        },
        "wall_seconds": float(time.time() - started),
    }
    (output / "summary.json").write_text(
        json.dumps(jsonable(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    seal_output(output)
    print(json.dumps(jsonable(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
