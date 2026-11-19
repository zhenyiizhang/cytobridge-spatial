#!/usr/bin/env python3
"""Independently audit corrected MOSTA S8 Brain gene-program tables."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import cut_tree, linkage
from sklearn.metrics import silhouette_score


EXPECTED_TIMES = tuple(float(value) for value in np.arange(0.0, 3.0001, 0.25))
EXPECTED_PACKAGE_COMMIT = "2b3c79eff3face7c4dd33de24d45384b9dbd8a84"
EXPECTED_ALIGNED_SHA256 = "8b9ca0ad3475040235036548d54b96272bf6c49f057f6c2a643152c11350ce25"
EXPECTED_CLASSIFIER_SHA256 = "f938575c145baa9002de695c39a8637f57d6cb3a06ccfaf4d18b707ca962a7e0"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "size_bytes": int(path.stat().st_size),
    }


def safe_time(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def load_matrix(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0)
    frame.index = frame.index.astype(str)
    frame.columns = [float(value) for value in frame.columns]
    return frame.astype(float)


def require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def representative_program_genes(
    expression: pd.DataFrame,
    normalized: pd.DataFrame,
    assignments: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    assignment = assignments.rename(columns={"profile": "gene"})
    for cluster, subset in assignment.groupby("cluster", sort=True):
        genes = subset["gene"].astype(str).tolist()
        profiles = normalized.loc[genes]
        prototype = profiles.mean(axis=0).to_numpy(float)
        for gene in genes:
            vector = profiles.loc[gene].to_numpy(float)
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
    return table.groupby("program", sort=True).head(5).reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shared-run", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    shared = Path(args.shared_run).expanduser().resolve()
    s8 = shared / "s8_gene_programs"
    states = shared / "generated_states"
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []

    paths = {
        "mean_log1p": s8 / "brain_hvg_mean_log1p_by_time.csv",
        "signed_mean": s8 / "brain_hvg_signed_mean_by_time.csv",
        "zscore": s8 / "brain_hvg_gene_wise_zscore.csv",
        "variance_rank": s8 / "brain_hvg_temporal_variance_rank.csv",
        "top20": s8 / "brain_top20_temporal_variable_genes.csv",
        "assignments": s8 / "brain_hvg_ward_k2_assignments.csv",
        "prototypes": s8 / "brain_hvg_ward_k2_prototypes.csv",
        "diagnostics": s8 / "brain_hvg_ward_k2_diagnostics.csv",
        "representatives": s8 / "brain_program_representative_genes_top5.csv",
        "gene_name_map": s8 / "brain_hvg_gene_name_map.csv",
        "reconstruction": s8 / "reconstruction_diagnostics.csv",
        "settings": s8 / "s8_gene_program_settings.json",
        "summary": shared / "summary.json",
        "state_inventory": states / "state_inventory.csv",
    }
    for name, path in paths.items():
        require(path.is_file(), f"missing input {name}: {path}", errors)
    if errors:
        raise FileNotFoundError("; ".join(errors))

    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    settings = json.loads(paths["settings"].read_text(encoding="utf-8"))
    package = settings["package_settings"]
    mean = load_matrix(paths["mean_log1p"])
    signed = load_matrix(paths["signed_mean"])
    zscore = load_matrix(paths["zscore"])
    variance_rank = pd.read_csv(paths["variance_rank"])
    top20 = pd.read_csv(paths["top20"])
    assignments = pd.read_csv(paths["assignments"])
    assignments["profile"] = assignments["profile"].astype(str)
    prototypes = pd.read_csv(paths["prototypes"])
    diagnostics = pd.read_csv(paths["diagnostics"])
    representatives = pd.read_csv(paths["representatives"])
    name_map = pd.read_csv(paths["gene_name_map"])
    reconstruction = pd.read_csv(paths["reconstruction"])
    state_inventory = pd.read_csv(paths["state_inventory"])

    require(summary["package_commit"] == EXPECTED_PACKAGE_COMMIT, "package commit mismatch", errors)
    require(summary["aligned_h5ad"]["sha256"] == EXPECTED_ALIGNED_SHA256, "aligned H5AD mismatch", errors)
    require(summary["classifier"]["sha256"] == EXPECTED_CLASSIFIER_SHA256, "classifier SHA mismatch", errors)
    require(int(summary["classifier"]["k"]) == 10, "classifier k is not 10", errors)
    require(summary["trajectory"]["mode"] == "global_t0_extrapolation", "trajectory is not global-t0", errors)
    require(not bool(summary["trajectory"]["restart_from_preceding_observed_stage"]), "observed restart detected", errors)
    require(not bool(summary["trajectory"]["spatial_warp"]), "spatial warp detected", errors)

    same_axes = (
        mean.index.equals(signed.index)
        and set(mean.index) == set(zscore.index)
        and tuple(mean.columns) == EXPECTED_TIMES
        and tuple(signed.columns) == EXPECTED_TIMES
        and tuple(zscore.columns) == EXPECTED_TIMES
    )
    require(mean.shape == (2000, 13), f"mean matrix shape is {mean.shape}", errors)
    require(signed.shape == (2000, 13), f"signed matrix shape is {signed.shape}", errors)
    require(zscore.shape == (2000, 13), f"z-score matrix shape is {zscore.shape}", errors)
    require(same_axes, "mean/signed/zscore axes differ", errors)
    require(not mean.index.has_duplicates, "duplicate gene names", errors)
    require(np.isfinite(mean.to_numpy()).all(), "non-finite mean values", errors)
    require(np.isfinite(signed.to_numpy()).all(), "non-finite signed values", errors)
    require(np.isfinite(zscore.to_numpy()).all(), "non-finite z-score values", errors)
    require(float(mean.to_numpy().min()) >= -1e-12, "clipped mean log1p contains negative values", errors)
    require(float(signed.to_numpy().min()) < 0.0, "signed diagnostic unexpectedly has no negatives", errors)
    require(
        bool(np.all(signed.to_numpy() <= mean.to_numpy() + 2e-6)),
        "signed mean exceeds per-cell-clipped mean",
        errors,
    )

    calc_z = mean.sub(mean.mean(axis=1), axis=0).div(
        mean.std(axis=1, ddof=0).clip(lower=1e-12), axis=0
    )
    # The z-score table is in temporal-variance order, whereas the mean table
    # retains original-HVG order.  CSV round-tripping the float32 means can
    # amplify error for nearly constant rows, so the numerical gate is set
    # above the measured round-trip maximum while the Ward labels must still
    # reproduce exactly below.
    calc_z_in_saved_order = calc_z.loc[zscore.index]
    zscore_max_abs_error = float(
        np.max(np.abs(calc_z_in_saved_order.to_numpy() - zscore.to_numpy()))
    )
    require(zscore_max_abs_error <= 2e-5, f"z-score mismatch: {zscore_max_abs_error}", errors)
    require(float(np.max(np.abs(zscore.mean(axis=1).to_numpy()))) <= 1e-9, "z-score row means are nonzero", errors)
    require(float(np.max(np.abs(zscore.std(axis=1, ddof=0).to_numpy() - 1.0))) <= 1e-9, "z-score row SDs are not one", errors)

    calc_variance = mean.var(axis=1, ddof=0).sort_values(ascending=False)
    require(variance_rank["gene"].astype(str).tolist() == calc_variance.index.tolist(), "variance gene order mismatch", errors)
    variance_max_abs_error = float(
        np.max(np.abs(variance_rank["variance"].to_numpy(float) - calc_variance.to_numpy(float)))
    )
    require(variance_max_abs_error <= 5e-8, f"temporal variance mismatch: {variance_max_abs_error}", errors)
    require(top20["gene"].astype(str).tolist() == variance_rank["gene"].astype(str).head(20).tolist(), "top20 gene selection mismatch", errors)
    require(np.allclose(top20["variance"], variance_rank["variance"].head(20), rtol=0.0, atol=1e-12), "top20 variance mismatch", errors)

    profile_order = assignments["profile"].tolist()
    require(profile_order == variance_rank["gene"].astype(str).tolist(), "assignment order is not variance order", errors)
    normalized = calc_z.loc[profile_order].to_numpy(float)
    hierarchy = linkage(normalized, method="ward", metric="euclidean")
    raw_labels = cut_tree(hierarchy, n_clusters=[2]).reshape(-1).astype(int) + 1
    order: list[tuple[int, int]] = []
    for raw_label in sorted(np.unique(raw_labels)):
        prototype = normalized[raw_labels == raw_label].mean(axis=0)
        order.append((int(np.argmax(prototype)), int(raw_label)))
    remap = {raw: index + 1 for index, (_, raw) in enumerate(sorted(order))}
    labels = np.asarray([remap[int(raw)] for raw in raw_labels], dtype=int)
    require(np.array_equal(labels, assignments["cluster"].to_numpy(int)), "independent Ward-k2 assignments differ", errors)
    require(sorted(np.unique(labels).tolist()) == [1, 2], "Ward cut did not return two clusters", errors)

    prototype_rows: list[dict[str, object]] = []
    for cluster in (1, 2):
        subset = normalized[labels == cluster]
        for column_index, time_value in enumerate(EXPECTED_TIMES):
            prototype_rows.append(
                {
                    "cluster": cluster,
                    "time": time_value,
                    "mean": float(subset[:, column_index].mean()),
                    "std": float(subset[:, column_index].std()),
                    "n_profiles": int(subset.shape[0]),
                }
            )
    calc_prototypes = pd.DataFrame(prototype_rows)
    prototype_max_abs_error = float(
        np.max(
            np.abs(
                calc_prototypes[["mean", "std"]].to_numpy(float)
                - prototypes[["mean", "std"]].to_numpy(float)
            )
        )
    )
    require(
        calc_prototypes[["cluster", "time", "n_profiles"]].equals(
            prototypes[["cluster", "time", "n_profiles"]]
        ),
        "prototype keys/counts differ",
        errors,
    )
    require(prototype_max_abs_error <= 5e-8, f"prototype mismatch: {prototype_max_abs_error}", errors)

    calc_silhouette = float(silhouette_score(normalized, labels))
    require(int(diagnostics.loc[0, "requested_clusters"]) == 2, "requested k differs", errors)
    require(int(diagnostics.loc[0, "clusters_found"]) == 2, "reported clusters differ", errors)
    require(str(diagnostics.loc[0, "normalization"]) == "zscore", "normalization differs", errors)
    require(str(diagnostics.loc[0, "linkage_method"]) == "ward", "linkage differs", errors)
    require(str(diagnostics.loc[0, "cluster_order"]) == "peak_time", "cluster order differs", errors)
    require(str(diagnostics.loc[0, "cut_strategy"]) == "scipy_cut_tree_exact_n_clusters", "cut strategy differs", errors)
    require(abs(float(diagnostics.loc[0, "silhouette"]) - calc_silhouette) <= 5e-8, "silhouette mismatch", errors)
    require(int(diagnostics.loc[0, "n_zero_distance_merges"]) == int(np.count_nonzero(np.isclose(hierarchy[:, 2], 0.0, rtol=0.0, atol=1e-15))), "zero-distance merge count mismatch", errors)

    normalized_frame = pd.DataFrame(normalized, index=profile_order, columns=EXPECTED_TIMES)
    calc_representatives = representative_program_genes(mean, normalized_frame, assignments)
    require(calc_representatives["gene"].tolist() == representatives["gene"].astype(str).tolist(), "representative gene order mismatch", errors)
    require(np.array_equal(calc_representatives["program"].to_numpy(int), representatives["program"].to_numpy(int)), "representative programs differ", errors)
    rep_numeric_error = float(
        np.max(
            np.abs(
                calc_representatives[
                    ["temporal_variance", "prototype_correlation", "variance_rank", "correlation_rank", "combined_rank"]
                ].to_numpy(float)
                - representatives[
                    ["temporal_variance", "prototype_correlation", "variance_rank", "correlation_rank", "combined_rank"]
                ].to_numpy(float)
            )
        )
    )
    require(rep_numeric_error <= 5e-8, f"representative ranking mismatch: {rep_numeric_error}", errors)

    candidate = settings["candidate_contract"]
    require(int(candidate["n_hvg_union"]) == 2747, "HVG union is not 2747", errors)
    require(int(candidate["n_required_additions"]) == 747, "required-addition count is not 747", errors)
    require(int(candidate["n_original_hvgs"]) == 2000, "original-HVG count is not 2000", errors)
    require(str(settings["roi"]) == "Brain", "ROI is not Brain", errors)
    require(str(settings["state_source"]) == "fully_generated_global_t0_all_13_times", "S8 state source differs", errors)
    require(int(package["n_clusters"]) == 2, "package n_clusters differs", errors)
    require(int(package["n_cluster_genes"]) == 2000, "package cluster-gene count differs", errors)
    require(package["profile_normalization"] == "zscore", "package normalization differs", errors)
    require(package["profile_linkage_method"] == "ward", "package linkage differs", errors)
    require(package["profile_cluster_order"] == "peak_time", "package cluster order differs", errors)
    require(float(package["clip_min"]) == 0.0, "clip_min differs", errors)
    require(package["pca_contract"]["center_source"] == "reference_adata.var['pca_center']", "PCA center source differs", errors)
    require(not bool(package["allow_complete_reference_pca_center_fallback"]), "PCA center fallback was enabled", errors)
    require(int(package["candidate_features"]["missing_count"]) == 0, "candidate features are missing", errors)
    require(int(package["candidate_features"]["inactive_count"]) == 0, "candidate features are inactive", errors)
    require(len(name_map) == 2000, "gene-name map is not 2000 rows", errors)
    require(bool(name_map["pca_active"].astype(bool).all()), "inactive PCA feature in gene-name map", errors)
    candidate_sha = hashlib.sha256(
        json.dumps(sorted(name_map["var_name"].astype(str).tolist()), separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    require(candidate_sha == candidate["sha256_sorted_json_utf8"], "original-HVG candidate SHA mismatch", errors)

    require(len(reconstruction) == 13, "reconstruction diagnostics do not cover 13 times", errors)
    require(tuple(reconstruction["time"].astype(float)) == EXPECTED_TIMES, "reconstruction times differ", errors)
    require(bool((reconstruction["n_features"].astype(int) == 2000).all()), "reconstruction n_features differs", errors)
    require(bool((reconstruction["n_values"].astype(int) == reconstruction["n_cells"].astype(int) * 2000).all()), "reconstruction value counts differ", errors)
    require(bool((reconstruction["clip_min"].astype(float) == 0.0).all()), "diagnostic clip_min differs", errors)
    require(bool((reconstruction["postclip_min"].astype(float) >= 0.0).all()), "post-clip negatives remain", errors)

    brain_counts: list[dict[str, object]] = []
    for time_value in EXPECTED_TIMES:
        path = states / f"time_{safe_time(time_value)}.h5ad"
        state = ad.read_h5ad(path, backed="r")
        labels_at_time = state.obs["Annotation"].astype(str)
        brain_count = int(labels_at_time.eq("Brain").sum())
        expected_count = int(
            reconstruction.loc[np.isclose(reconstruction["time"], time_value), "n_cells"].iloc[0]
        )
        inventory_count = int(
            state_inventory.loc[np.isclose(state_inventory["time"], time_value), "n_cells"].iloc[0]
        )
        brain_counts.append(
            {
                "time": time_value,
                "brain_cells": brain_count,
                "reconstruction_n_cells": expected_count,
                "all_generated_cells": inventory_count,
                "state_sha256": sha256(path),
            }
        )
        require(brain_count == expected_count, f"Brain cell count mismatch at t={time_value:g}", errors)
        require(int(state.n_vars) == 52, f"generated feature count differs at t={time_value:g}", errors)
        require(state.uns["slice_origin"] == "generated_global_t0", f"state origin differs at t={time_value:g}", errors)
        require(float(state.uns["source_anchor_time"]) == 0.0, f"source anchor differs at t={time_value:g}", errors)
        require(not bool(state.uns["spatial_warp"]), f"state is warped at t={time_value:g}", errors)
        state.file.close()
    pd.DataFrame(brain_counts).to_csv(output / "s8_brain_cell_counts_by_time.csv", index=False)

    program_rows: list[dict[str, object]] = []
    for cluster, group in prototypes.groupby("cluster", sort=True):
        group = group.sort_values("time")
        program_rows.append(
            {
                "program": int(cluster),
                "n_genes": int(group["n_profiles"].iloc[0]),
                "peak_time": float(group.loc[group["mean"].idxmax(), "time"]),
                "endpoint_delta": float(group["mean"].iloc[-1] - group["mean"].iloc[0]),
                "time_correlation": float(np.corrcoef(group["time"], group["mean"])[0, 1]),
                "representative_genes": ";".join(
                    representatives.loc[
                        representatives["program"].astype(int) == int(cluster), "gene"
                    ].astype(str)
                ),
                "top20_gene_count": int((top20["cluster"].astype(int) == int(cluster)).sum()),
            }
        )
    program_metrics = pd.DataFrame(program_rows)
    program_metrics.to_csv(output / "s8_program_interpretability_metrics.csv", index=False)
    require(program_metrics.loc[program_metrics["program"] == 1, "peak_time"].iloc[0] == 0.0, "Pattern 1 is not early-peaking", errors)
    require(program_metrics.loc[program_metrics["program"] == 1, "endpoint_delta"].iloc[0] < 0.0, "Pattern 1 does not decrease", errors)
    require(program_metrics.loc[program_metrics["program"] == 2, "peak_time"].iloc[0] == 3.0, "Pattern 2 is not late-peaking", errors)
    require(program_metrics.loc[program_metrics["program"] == 2, "endpoint_delta"].iloc[0] > 0.0, "Pattern 2 does not increase", errors)

    audit = {
        "schema_version": 1,
        "status": "PASS" if not errors else "FAIL",
        "panel": "Supplementary Figure S8a-b",
        "dataset": "MOSTA",
        "errors": errors,
        "inputs": {name: identity(path) for name, path in paths.items()},
        "release": {
            "package_commit": summary["package_commit"],
            "package_archive_sha256": summary["package_release"]["archive_sha256"],
            "aligned_h5ad_sha256": summary["aligned_h5ad"]["sha256"],
            "finetune_sha256": summary["model"]["finetune"]["sha256"],
            "score_sha256": summary["model"]["score"]["sha256"],
            "classifier_sha256": summary["classifier"]["sha256"],
            "classifier_k": int(summary["classifier"]["k"]),
        },
        "calculation_contract": {
            "roi": "Brain",
            "state_source": "fully_generated_global_t0_all_13_times",
            "time_points": list(EXPECTED_TIMES),
            "original_hvgs": 2000,
            "required_lr_additions_excluded": 747,
            "pca_center_source": package["pca_contract"]["center_source"],
            "per_cell_inverse_pca_then_clip_then_mean": True,
            "clip_min": 0.0,
            "normalization": "gene-wise zscore ddof=0",
            "linkage": "Ward Euclidean",
            "cut": "scipy cut_tree exact k=2",
            "cluster_order": "prototype peak time",
            "legacy_program_sizes_forced": False,
        },
        "independent_checks": {
            "matrix_shape": list(mean.shape),
            "axes_identical": same_axes,
            "clipped_mean_min": float(mean.to_numpy().min()),
            "signed_mean_min": float(signed.to_numpy().min()),
            "signed_leq_clipped": bool(np.all(signed.to_numpy() <= mean.to_numpy() + 2e-6)),
            "zscore_max_abs_error": zscore_max_abs_error,
            "variance_max_abs_error": variance_max_abs_error,
            "assignments_exact": bool(np.array_equal(labels, assignments["cluster"].to_numpy(int))),
            "prototype_max_abs_error": prototype_max_abs_error,
            "silhouette_recomputed": calc_silhouette,
            "representative_numeric_max_abs_error": rep_numeric_error,
            "candidate_sha256_recomputed": candidate_sha,
            "brain_counts_match_generated_h5ad": not any("Brain cell count mismatch" in value for value in errors),
        },
        "programs": program_metrics.to_dict(orient="records"),
        "top20_genes": top20.to_dict(orient="records"),
        "reconstruction": {
            "brain_cells_start": int(reconstruction["n_cells"].iloc[0]),
            "brain_cells_end": int(reconstruction["n_cells"].iloc[-1]),
            "preclip_fraction_min": float(reconstruction["fraction_below_clip_min"].min()),
            "preclip_fraction_max": float(reconstruction["fraction_below_clip_min"].max()),
        },
    }
    (output / "s8_numerical_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
