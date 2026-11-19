#!/usr/bin/env python3
"""Independently audit MOSTA S9/S10 DP3 inputs and clusterProfiler tables."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import hypergeom


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def parse_ratio(value: object) -> tuple[int, int]:
    left, right = str(value).split("/", 1)
    return int(left), int(right)


def bh_adjust(pvalues: np.ndarray) -> np.ndarray:
    values = np.asarray(pvalues, dtype=float)
    order = np.argsort(values, kind="stable")
    ranked = values[order]
    adjusted_ranked = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
    adjusted_ranked = np.clip(adjusted_ranked, 0.0, 1.0)
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = adjusted_ranked
    return adjusted


def segment_costs(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    prefix = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
    prefix_sq = np.concatenate(([0.0], np.cumsum(values**2, dtype=np.float64)))
    n = len(values)
    costs = np.full((n + 1, n + 1), np.inf, dtype=np.float64)
    for start in range(n):
        ends = np.arange(start + 1, n + 1, dtype=int)
        counts = ends - start
        totals = prefix[ends] - prefix[start]
        sums_sq = prefix_sq[ends] - prefix_sq[start]
        costs[start, ends] = np.maximum(sums_sq - totals**2 / counts, 0.0)
    return costs


def exact_dp(values: np.ndarray, n_segments: int = 3, min_size: int = 5) -> tuple[list[tuple[int, int]], float]:
    values = np.asarray(values, dtype=float)
    n = len(values)
    costs = segment_costs(values)
    objective = np.full((n_segments + 1, n + 1), np.inf)
    previous = np.full((n_segments + 1, n + 1), -1, dtype=int)
    objective[0, 0] = 0.0
    for count in range(1, n_segments + 1):
        smallest_end = count * min_size
        largest_end = n - (n_segments - count) * min_size
        for end in range(smallest_end, largest_end + 1):
            candidates = np.arange((count - 1) * min_size, end - min_size + 1)
            values_here = objective[count - 1, candidates] + costs[candidates, end]
            offset = int(np.argmin(values_here))
            objective[count, end] = float(values_here[offset])
            previous[count, end] = int(candidates[offset])
    boundaries: list[tuple[int, int]] = []
    end = n
    for count in range(n_segments, 0, -1):
        start = int(previous[count, end])
        boundaries.append((start, end))
        end = start
    return boundaries[::-1], float(objective[n_segments, n])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shared-run", required=True, type=Path)
    parser.add_argument("--clusterprofiler-run", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    shared = args.shared_run.expanduser().resolve()
    go = args.clusterprofiler_run.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite audit output: {output}")
    output.mkdir(parents=True)
    errors: list[str] = []

    manifest = json.loads((go / "manifest.json").read_text())
    require(manifest["status"] == "COMPLETE", "clusterProfiler manifest is not COMPLETE", errors)
    require(manifest["calculation_contract"]["ontologies_pooled"] is True, "GO ontologies were not pooled", errors)
    require(manifest["calculation_contract"]["multiple_testing"].startswith("Benjamini-Hochberg"), "GO correction is not BH", errors)
    require(manifest["mapping"]["background_input_symbols"] == 2000, "GO background is not 2,000 symbols", errors)
    require(manifest["mapping"]["ambiguous_symbol_count"] == 0, "Ambiguous symbol mappings were present", errors)

    mean = pd.read_csv(shared / "s8_gene_programs" / "brain_hvg_mean_log1p_by_time.csv", index_col=0)
    mean.columns = [float(value) for value in mean.columns]
    values = mean.to_numpy(float)
    names = mean.index.astype(str).to_numpy()
    variances = np.var(values, axis=1, ddof=0)
    selected_positions = np.lexsort((names, -variances))[:1000]
    selected_values = values[selected_positions]
    selected_names = names[selected_positions]
    selected_variances = variances[selected_positions]
    selected_z = (selected_values - selected_values.mean(axis=1, keepdims=True)) / selected_values.std(axis=1, ddof=0, keepdims=True)
    peak_indices = np.argmax(selected_z, axis=1)
    times = mean.columns.to_numpy(float)
    peak_times = times[peak_indices]
    wave_order = np.lexsort((selected_names, -selected_variances, peak_times))
    ordered_names = selected_names[wave_order]
    ordered_variances = selected_variances[wave_order]
    ordered_peaks = peak_indices[wave_order]
    ordered_peak_times = peak_times[wave_order]
    ordered_z = selected_z[wave_order]
    boundaries, objective = exact_dp(ordered_peak_times)
    phases = np.empty(1000, dtype=int)
    for phase, (start, end) in enumerate(boundaries, start=1):
        phases[start:end] = phase

    saved = pd.read_csv(shared / "s10_developmental_wave" / "s10_top1000_dp3_assignments.csv")
    require(saved["profile"].astype(str).tolist() == ordered_names.tolist(), "S10 wave profile order did not reproduce", errors)
    require(np.array_equal(saved["wave_rank"].to_numpy(int), np.arange(1000)), "S10 wave ranks did not reproduce", errors)
    require(np.allclose(saved["temporal_variance"], ordered_variances, rtol=1e-6, atol=1e-9), "S10 temporal variances did not reproduce", errors)
    require(np.array_equal(saved["peak_index"].to_numpy(int), ordered_peaks), "S10 earliest peak indices did not reproduce", errors)
    require(np.allclose(saved["peak_time"], ordered_peak_times, rtol=0, atol=1e-12), "S10 peak times did not reproduce", errors)
    require(np.array_equal(saved["phase"].to_numpy(int), phases), "S10 exact DP3 phase labels did not reproduce", errors)
    require(boundaries == [(0, 483), (483, 782), (782, 1000)], f"Unexpected independently recomputed DP boundaries: {boundaries}", errors)
    require(abs(objective - 70.34855609421834) < 1e-9, f"Unexpected DP objective: {objective}", errors)

    saved_ordered = pd.read_csv(shared / "s10_developmental_wave" / "s10_top1000_peak_ordered_profiles.csv", index_col=0)
    saved_ordered.columns = [float(value) for value in saved_ordered.columns]
    require(saved_ordered.index.astype(str).tolist() == ordered_names.tolist(), "S10 saved ordered matrix row order differs", errors)
    require(np.allclose(saved_ordered.to_numpy(float), ordered_z, rtol=1e-5, atol=2e-5), "S10 saved ordered z-score matrix differs", errors)

    go_assignments = pd.read_csv(go / "inputs" / "brain_hvg_ward_k2_assignments.csv")
    shared_assignments = pd.read_csv(shared / "s8_gene_programs" / "brain_hvg_ward_k2_assignments.csv")
    require(go_assignments.equals(shared_assignments), "GO copy of S8 assignments differs", errors)
    go_phases = pd.read_csv(go / "inputs" / "s10_top1000_dp3_assignments.csv")
    require(go_phases.equals(saved), "GO copy of S10 phase assignments differs", errors)
    require(sha256(shared / "s8_gene_programs" / "brain_hvg_ward_k2_assignments.csv") == manifest["inputs"]["s8_assignments"]["sha256"], "S8 assignment SHA mismatch", errors)
    require(sha256(shared / "s10_developmental_wave" / "s10_top1000_dp3_assignments.csv") == manifest["inputs"]["s10_phase_assignments"]["sha256"], "S10 assignment SHA mismatch", errors)

    query_gene_sets = {
        "s9_pattern_1": set(shared_assignments.loc[shared_assignments["cluster"] == 1, "profile"].astype(str)),
        "s9_pattern_2": set(shared_assignments.loc[shared_assignments["cluster"] == 2, "profile"].astype(str)),
        "s10_phase_1": set(saved.loc[saved["phase"] == 1, "profile"].astype(str)),
        "s10_phase_2": set(saved.loc[saved["phase"] == 2, "profile"].astype(str)),
        "s10_phase_3": set(saved.loc[saved["phase"] == 3, "profile"].astype(str)),
    }
    go_metrics: list[dict[str, object]] = []
    for query_id, expected_symbols in query_gene_sets.items():
        query_input = pd.read_csv(go / "inputs" / f"{query_id}_query_symbols.csv")
        require(set(query_input["SYMBOL"].astype(str)) == expected_symbols, f"{query_id} input symbol set differs", errors)
        table = pd.read_csv(go / "tables" / f"{query_id}_enrichGO_all.csv")
        sig = pd.read_csv(go / "tables" / f"{query_id}_enrichGO_fdr_lt_0p05.csv")
        display = pd.read_csv(go / "tables" / f"{query_id}_enrichGO_display_top20.csv")
        require(table["ID"].is_unique, f"{query_id} has duplicated GO IDs", errors)
        require(set(table["ONTOLOGY"]).issubset({"BP", "MF", "CC"}), f"{query_id} has invalid ontology", errors)
        recomputed_bh = bh_adjust(table["pvalue"].to_numpy(float))
        bh_max_error = float(np.max(np.abs(recomputed_bh - table["p.adjust"].to_numpy(float))))
        require(bh_max_error < 1e-12, f"{query_id} pooled BH values did not reproduce ({bh_max_error})", errors)

        hyper_errors = []
        for row in table.itertuples(index=False):
            count, query_n = parse_ratio(row.GeneRatio)
            term_n, background_n = parse_ratio(row.BgRatio)
            calc = float(hypergeom.sf(count - 1, background_n, term_n, query_n))
            hyper_errors.append(abs(calc - float(row.pvalue)))
            if count != int(row.Count):
                errors.append(f"{query_id} Count differs from GeneRatio numerator for {row.ID}")
        hyper_max_error = float(max(hyper_errors))
        require(hyper_max_error < 1e-12, f"{query_id} hypergeometric p-values did not reproduce ({hyper_max_error})", errors)

        expected_sig = table.loc[table["p.adjust"] < 0.05].copy()
        expected_sig = expected_sig.sort_values(
            ["p.adjust", "pvalue", "Count", "Description"],
            ascending=[True, True, False, True],
            kind="stable",
        )
        require(sig["ID"].tolist() == expected_sig["ID"].tolist(), f"{query_id} significant table differs", errors)
        require(display["ID"].tolist() == expected_sig.head(20)["ID"].tolist(), f"{query_id} top display selection differs", errors)
        require(len(display) == min(20, len(expected_sig)), f"{query_id} display length differs", errors)
        go_metrics.append(
            {
                "query_id": query_id,
                "input_symbols": len(expected_symbols),
                "tested_terms": len(table),
                "significant_terms": len(expected_sig),
                "displayed_terms": len(display),
                "bh_max_abs_error": bh_max_error,
                "hypergeometric_max_abs_error": hyper_max_error,
                "display_ontologies": ";".join(f"{key}:{value}" for key, value in display["ONTOLOGY"].value_counts().sort_index().items()),
                "top_term": display.iloc[0]["Description"] if len(display) else None,
                "top_p_adjust": float(display.iloc[0]["p.adjust"]) if len(display) else None,
            }
        )

    phase3 = pd.read_csv(go / "tables" / "s10_phase_3_enrichGO_all.csv")
    phase3_lookup = phase3.set_index("Description")
    interpretability = {
        "phase3_nervous_system_process_p_adjust": float(phase3_lookup.loc["nervous system process", "p.adjust"]),
        "phase3_extracellular_matrix_p_adjust": float(phase3_lookup.loc["extracellular matrix", "p.adjust"]),
        "phase3_cell_periphery_p_adjust": float(phase3_lookup.loc["cell periphery", "p.adjust"]),
        "interpretation": "late phase is statistically enriched for cell-periphery/adhesion/ECM terms; nervous system process remains significant but is not the leading pooled GO-ALL term",
    }

    metrics = pd.DataFrame(go_metrics)
    metrics.to_csv(output / "go_query_audit_metrics.csv", index=False)
    audit = {
        "schema_version": 1,
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "dataset": "MOSTA",
        "panels": ["S9a-b", "S10a-d"],
        "s10_dp3": {
            "independent_boundaries": boundaries,
            "phase_sizes": [end - start for start, end in boundaries],
            "objective": objective,
            "profile_order_exact": saved["profile"].astype(str).tolist() == ordered_names.tolist(),
            "assignments_exact": bool(np.array_equal(saved["phase"].to_numpy(int), phases)),
            "ordered_zscore_max_abs_error": float(np.max(np.abs(saved_ordered.to_numpy(float) - ordered_z))),
        },
        "clusterprofiler": {
            "server_manifest_sha256": sha256(go / "manifest.json"),
            "orgdb_sqlite_sha256": manifest["software"]["orgdb_sqlite"]["sha256"],
            "background_input_symbols": manifest["mapping"]["background_input_symbols"],
            "background_mapped_symbols": manifest["mapping"]["background_mapped_symbols"],
            "background_unique_entrez": manifest["mapping"]["background_unique_entrez"],
            "query_metrics": go_metrics,
        },
        "interpretability": interpretability,
    }
    (output / "numerical_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
