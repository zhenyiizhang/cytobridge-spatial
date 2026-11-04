#!/usr/bin/env python3
"""Decompose MOSTA S11 temporal pulses across communication components.

This is an audit, not a final figure computation.  It reconstructs the exact
package communication object from the sealed sparse attention arrays and the
deterministic seed-42 sample.  Exact source ``M_per_source`` hashes are used as
a hard gate.  Package-native ``M_sum`` and ``M_mean`` are then compared with
three explicitly diagnostic matrices that isolate total attention magnitude,
the legacy per-time max normalization, and expression-only dynamics.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import platform
import shutil
import stat
import sys
import time
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd


EXPECTED_COMMIT = "2b3c79eff3face7c4dd33de24d45384b9dbd8a84"
EXPECTED_ARCHIVE = "06852992db0d8ebedd8f0baa19a3f539bcdd271dfc0c6fae9774dd553c8fe55e"
EXPECTED_REFERENCE = "8b9ca0ad3475040235036548d54b96272bf6c49f057f6c2a643152c11350ce25"
TIMES = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
SEED = 42
N_COMMUNICATION = 12_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", required=True)
    parser.add_argument("--shared-state-root", required=True)
    parser.add_argument("--reference-h5ad", required=True)
    parser.add_argument("--source-s11-root", required=True)
    parser.add_argument("--base-audit-script", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("utf-8"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def time_token(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value).replace(".", "p")


def load_base(path: Path):
    spec = importlib.util.spec_from_file_location("s11_mrow_audit_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import base audit helpers: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def deterministic_indices(states: dict[str, ad.AnnData], inventory: pd.DataFrame) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(SEED)
    expected = inventory.set_index("time")
    selected: dict[str, np.ndarray] = {}
    for time_value in TIMES:
        state = states[str(float(time_value))]
        if state.n_obs <= N_COMMUNICATION:
            indices = np.arange(state.n_obs, dtype=np.int64)
        else:
            indices = np.sort(
                rng.choice(state.n_obs, size=N_COMMUNICATION, replace=False)
            ).astype(np.int64, copy=False)
        if sha256_array(indices) != str(expected.loc[time_value, "sampled_indices_sha256"]):
            raise RuntimeError(f"Seed-42 sampled index hash mismatch at {time_value:g}")
        selected[str(float(time_value))] = indices
    return selected


def reconstruct_components(
    cb,
    states: dict[str, ad.AnnData],
    selected: dict[str, np.ndarray],
    source: Path,
) -> tuple[dict[str, dict[str, Any]], pd.DataFrame]:
    source_audit = pd.read_csv(
        source / "tables" / "communication_matrix_audit.csv",
        float_precision="round_trip",
    ).set_index("time")
    records: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for time_value in TIMES:
        key = str(float(time_value))
        state = states[key]
        indices = selected[key]
        edge_index = np.load(
            source / "attention_sparse" / f"edge_index_interp_t{time_value}.npy"
        )
        attention = np.load(
            source / "attention_sparse" / f"attn_mean_interp_t{time_value}.npy"
        )
        labels = state.obs.iloc[indices]["Annotation"].astype(str).to_numpy()
        spatial = np.asarray(state.obsm["spatial"])[indices]
        record = cb.tl.analyze_attention_by_celltype(
            edge_index=edge_index,
            attn=attention,
            labels=labels,
            spatial_coord=spatial,
            time_title=key,
            remove_self_loop=True,
            winsor_quantile=0.995,
            distance_bins=None,
            n_permutations=0,
            plot=False,
        )
        m_per_source = np.asarray(record["M_per_source"], dtype=np.float64)
        expected_hash = str(source_audit.loc[time_value, "matrix_sha256"])
        if sha256_array(m_per_source) != expected_hash:
            raise RuntimeError(f"Exact sparse reconstruction failed at {time_value:g}")
        m_sum = np.asarray(record["M_sum"], dtype=np.float64)
        m_mean = np.asarray(record["M_mean"], dtype=np.float64)
        total = float(m_per_source.sum())
        maximum = float(m_sum.max(initial=0.0))
        n_types = len(record["types"])
        record["M_per_source_global_fraction"] = (
            m_per_source / total if total > 0 else np.zeros_like(m_per_source)
        )
        record["M_sum_maxnorm_legacy"] = (
            m_sum / maximum if maximum > 0 else np.zeros_like(m_sum)
        )
        record["M_uniform_expression_only"] = np.full(
            (n_types, n_types), 1.0 / float(n_types * n_types), dtype=np.float64
        )
        records[key] = record
        rows.append(
            {
                "time": time_value,
                "n_types_sampled": n_types,
                "m_per_source_sha256": expected_hash,
                "m_sum_sha256": sha256_array(m_sum),
                "m_mean_sha256": sha256_array(m_mean),
                "m_row_sha256": sha256_array(np.asarray(record["M_row"], dtype=np.float64)),
                "m_per_source_sum": total,
                "m_sum_sum": float(m_sum.sum()),
                "m_sum_max": maximum,
                "m_mean_nonzero_mean": float(m_mean[m_mean > 0].mean()) if (m_mean > 0).any() else 0.0,
            }
        )
    return records, pd.DataFrame(rows)


def raw_score_summary(pair_timecourse: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for time_value, subset in pair_timecourse.groupby("time", sort=True):
        values = subset["score"].to_numpy(dtype=float)
        rows.append(
            {
                "time": float(time_value),
                "n_pairs": len(values),
                "score_sum": float(values.sum()),
                "score_median": float(np.median(values)),
                "score_q10": float(np.quantile(values, 0.10)),
                "score_q90": float(np.quantile(values, 0.90)),
                "score_max": float(values.max(initial=0.0)),
            }
        )
    return pd.DataFrame(rows)


def seal(root: Path) -> None:
    (root / "COMPLETE").write_text("complete\n", encoding="utf-8")
    files = sorted(path for path in root.rglob("*") if path.is_file() and path.name != "SHA256SUMS.txt")
    (root / "SHA256SUMS.txt").write_text(
        "\n".join(f"{sha256_file(path)}  {path.relative_to(root)}" for path in files) + "\n",
        encoding="utf-8",
    )
    for path in root.rglob("*"):
        if path.is_file():
            path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    for directory in sorted((p for p in root.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
        directory.chmod(0o555)
    root.chmod(0o555)


def main() -> None:
    args = parse_args()
    package_root = Path(args.package_root).resolve()
    shared_root = Path(args.shared_state_root).resolve()
    reference_path = Path(args.reference_h5ad).resolve()
    source_root = Path(args.source_s11_root).resolve()
    base_script = Path(args.base_audit_script).resolve()
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(output)
    if (package_root / "RELEASE_COMMIT").read_text().strip() != EXPECTED_COMMIT:
        raise RuntimeError("Package commit mismatch")
    if (package_root / "ARCHIVE_SHA256").read_text().strip() != EXPECTED_ARCHIVE:
        raise RuntimeError("Package archive mismatch")
    if sha256_file(reference_path) != EXPECTED_REFERENCE:
        raise RuntimeError("Reference hash mismatch")
    base = load_base(base_script)
    source_contract = base.verify_checksums(source_root, "SHA256SUMS.txt")
    shared_contract = base.verify_checksums(shared_root, "SHA256SUMS.txt")

    sys.path.insert(0, str(package_root))
    import CytoBridge as cb
    if not str(Path(cb.__file__).resolve()).startswith(str(package_root)):
        raise RuntimeError(f"Wrong CytoBridge import: {cb.__file__}")

    output.mkdir(parents=True, exist_ok=False)
    tables = output / "tables"
    provenance = output / "provenance"
    tables.mkdir(); provenance.mkdir()
    shutil.copy2(Path(__file__).resolve(), provenance / Path(__file__).name)
    shutil.copy2(base_script, provenance / base_script.name)
    start = time.perf_counter()

    states = {
        str(float(time_value)): ad.read_h5ad(
            shared_root / "generated_states" / f"time_{time_token(time_value)}.h5ad"
        )
        for time_value in TIMES
    }
    sample_inventory = pd.read_csv(source_root / "tables" / "communication_sampling_inventory.csv")
    selected = deterministic_indices(states, sample_inventory)
    communications, component_audit = reconstruct_components(cb, states, selected, source_root)
    component_audit.to_csv(tables / "communication_component_exact_audit.csv", index=False)

    reference = ad.read_h5ad(reference_path, backed="r")
    reconstruction = cb.tl.make_pca_reconstruction_spec(
        reference.var_names.astype(str),
        np.asarray(reference.varm["PCs"], dtype=np.float32),
        np.asarray(reference.var["pca_center"], dtype=np.float32),
        metadata={"source": str(reference_path), "source_sha256": EXPECTED_REFERENCE,
                  "expression_contract": "all_generated_inverse_pca"},
    )
    lr_database = package_root / "CytoBridge" / "workflow_databases" / "CellChatDB.ligrec.mouse.csv"
    components = {
        "M_sum": {"class": "package_native", "meaning": "total attention by type pair"},
        "M_mean": {"class": "package_native", "meaning": "mean attention per observed cell edge"},
        "M_per_source_global_fraction": {
            "class": "diagnostic_only",
            "meaning": "M_per_source divided by its per-time global sum",
        },
        "M_sum_maxnorm_legacy": {
            "class": "legacy_diagnostic_only",
            "meaning": "old notebook M_sum divided by its per-time maximum",
        },
        "M_uniform_expression_only": {
            "class": "diagnostic_only",
            "meaning": "uniform type-pair weights summing to one",
        },
    }
    summaries: dict[str, Any] = {}
    retained_universe: set[str] | None = None
    for component, metadata in components.items():
        component_dir = tables / component
        component_dir.mkdir()
        result = cb.tl.project_communication_to_lr_timecourses(
            states, reference, communications, lr_database,
            time_points=TIMES, annotation_key="Annotation", matrix_key=component,
            spatial_dim=2, loadings_key="PCs", reference_layer=None,
            expression_space="count", complex_mode="min", require_all_subunits=True,
            duplicate_policy="first", preferred_species_tag=None, n_clusters=3,
            pca_reconstruction=reconstruction, profile_linkage_method="average",
            profile_cluster_order="peak_time", observed_adata=None,
            observed_time_points=None, return_type_matrices=False,
        )
        universe = set(result.clustering.assignments["pair_id"].astype(str))
        if retained_universe is None:
            retained_universe = universe
        elif universe != retained_universe:
            raise RuntimeError(f"Retained pair universe changed for {component}")
        result.pair_timecourse.to_csv(component_dir / "lr_pair_timecourse.csv", index=False)
        result.clustering.normalized_profiles.to_csv(component_dir / "lr_normalized_profiles.csv")
        result.clustering.assignments.to_csv(component_dir / "lr_pattern_assignments.csv", index=False)
        result.clustering.prototypes.to_csv(component_dir / "lr_pattern_prototypes.csv", index=False)
        result.clustering.diagnostics.to_csv(component_dir / "lr_pattern_diagnostics.csv", index=False)
        raw = raw_score_summary(result.pair_timecourse)
        raw.to_csv(component_dir / "raw_score_summary.csv", index=False)
        pulse, pulse_metrics = base.pulse_table(
            result.clustering.normalized_profiles, result.clustering.assignments
        )
        pulse.to_csv(component_dir / "pulse_diagnostics.csv", index=False)
        submitted = base.submitted_profile_audit(
            result.clustering.normalized_profiles, result.clustering.assignments
        )
        submitted = submitted.rename(
            columns={"corrected_m_row_cluster": "corrected_component_cluster"}
        )
        submitted.to_csv(component_dir / "submitted_profile_audit.csv", index=False)
        cluster_sizes = {
            str(int(key)): int(value)
            for key, value in result.clustering.assignments["cluster"].value_counts().sort_index().items()
        }
        summaries[component] = {
            **metadata,
            "cluster_sizes": cluster_sizes,
            "diagnostics": result.clustering.diagnostics.to_dict(orient="records"),
            "pulse_metrics": pulse_metrics,
            "raw_score_summary": raw.to_dict(orient="records"),
            "submitted_profile_cluster_counts": submitted[
                "corrected_component_cluster"
            ].value_counts().sort_index().to_dict(),
        }
    reference.file.close()

    source_assignments = pd.read_csv(source_root / "tables" / "lr_pattern_assignments.csv")
    if retained_universe != set(source_assignments["pair_id"].astype(str)):
        raise RuntimeError("Component audit changed the source 1757-pair universe")
    summary = {
        "schema_version": 1,
        "status": "COMPLETE_AUDIT_NOT_FINAL",
        "dataset": "MOSTA",
        "panel": "S11",
        "question": "Which communication or expression component drives the synchronized temporal pulses?",
        "hard_gates": {
            "exact_sparse_reconstruction_of_source_M_per_source": True,
            "seed42_sample_indices_exact": True,
            "same_1757_pair_universe_every_component": True,
            "same_fully_generated_states_and_inverse_pca_expression": True,
            "no_smoothing": True,
        },
        "components": summaries,
        "source_run": source_contract,
        "shared_states": shared_contract,
        "base_helper": {"path": str(base_script), "sha256": sha256_file(base_script)},
        "package": {
            "root": str(package_root), "module": str(Path(cb.__file__).resolve()),
            "commit": EXPECTED_COMMIT, "archive_sha256": EXPECTED_ARCHIVE,
            "attention_signature": str(inspect.signature(cb.tl.analyze_attention_by_celltype)),
            "projection_signature": str(inspect.signature(cb.tl.project_communication_to_lr_timecourses)),
        },
        "runtime": {"python": sys.version, "platform": platform.platform()},
        "wall_seconds": time.perf_counter() - start,
    }
    (output / "summary.json").write_text(
        json.dumps(jsonable(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    seal(output)
    print(json.dumps(jsonable(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
