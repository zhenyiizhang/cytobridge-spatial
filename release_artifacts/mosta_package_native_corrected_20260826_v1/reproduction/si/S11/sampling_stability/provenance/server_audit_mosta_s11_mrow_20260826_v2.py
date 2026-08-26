#!/usr/bin/env python3
"""Audit MOSTA S11 with the package-native row-normalized communication component.

The accepted fully generated global-t0 states, cell sampling, sparse attention,
expression reconstruction, LR database, and strict complex rules are unchanged.
Only the communication matrix component is switched from absolute
``M_per_source`` to package-native compositional ``M_row``.  This tests whether
the rare synchronized S11 pulses are driven by global attention magnitude.
"""

from __future__ import annotations

import argparse
import hashlib
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
OLD_SUBMITTED_PROFILES = (
    (1, "Col1a1_Itga9_Itgb1"), (1, "H2-Q10_Cd8b1"),
    (1, "H2-Q4_Cd8b1"), (1, "Col6a4_Itga9_Itgb1"),
    (1, "Wnt10b_Fzd7_Lrp5"), (1, "Wnt10b_Fzd7_Lrp6"),
    (1, "Scgb3a2_Marco"), (1, "Col9a3_Itga9_Itgb1"),
    (1, "Tnn_Itga9_Itgb1"), (1, "Col2a1_Itga9_Itgb1"),
    (1, "Col1a2_Itga9_Itgb1"), (1, "Col9a2_Itga9_Itgb1"),
    (2, "Wnt3a_Fzd7_Lrp5"), (2, "Wnt3a_Fzd7_Lrp6"),
    (2, "Ptprm_Ptprm"), (2, "Sema3a_Nrp1_Plxna2"),
    (2, "Col9a1_Itga9_Itgb1"), (2, "Col6a5_Itga9_Itgb1"),
    (2, "Sema3c_Nrp1_Plxna2"), (2, "Sema3c_Nrp2_Plxna2"),
    (2, "Lama4_Itga9_Itgb1"), (2, "Sema3c_Nrp2_Plxna4"),
    (2, "Sema3c_Nrp2_Plxna3"), (2, "Sema3c_Nrp1_Nrp2_Plxnd1"),
    (2, "Sema3c_Nrp2_Plxna1"), (2, "Lamb2_Itga9_Itgb1"),
    (2, "Btc_Erbb2_Erbb4"), (2, "Btc_Erbb4"),
    (2, "Lama3_Itga9_Itgb1"), (2, "Wnt10a_Fzd7_Lrp5"),
    (2, "Wnt10a_Fzd7_Lrp6"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", required=True)
    parser.add_argument("--shared-state-root", required=True)
    parser.add_argument("--reference-h5ad", required=True)
    parser.add_argument("--source-s11-root", required=True)
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


def verify_checksums(root: Path, manifest_name: str) -> dict[str, Any]:
    manifest = root / manifest_name
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    checked = 0
    for raw in manifest.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        expected, relative = raw.split(maxsplit=1)
        path = root / relative.lstrip("*")
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"Checksum mismatch: {path}")
        checked += 1
    return {"root": str(root), "manifest": manifest_name,
            "manifest_sha256": sha256_file(manifest), "files_verified": checked}


def time_token(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value).replace(".", "p")


def reconstruct_communications(source: Path) -> tuple[dict[str, dict[str, Any]], pd.DataFrame]:
    # The source table was written with pandas' shortest exact float
    # representation.  ``round_trip`` is required here: pandas' default fast
    # parser may move the final bit of a float64 and therefore break the
    # byte-level matrix provenance hash even though the printed values and
    # numerical sums are unchanged.
    edges = pd.read_csv(
        source / "tables" / "communication_edges.csv",
        float_precision="round_trip",
    )
    source_audit = pd.read_csv(source / "tables" / "communication_matrix_audit.csv").set_index("time")
    communications: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for time_value in TIMES:
        subset = edges.loc[np.isclose(edges["time"], time_value)].copy()
        types = pd.unique(subset["sender_type"].astype(str)).tolist()
        receivers = pd.unique(subset["receiver_type"].astype(str)).tolist()
        if types != receivers or len(subset) != len(types) ** 2:
            raise RuntimeError(f"Communication edge square failed at {time_value:g}")
        matrix = (
            subset.pivot(index="sender_type", columns="receiver_type", values="M_per_source")
            .reindex(index=types, columns=types)
            .to_numpy(dtype=np.float64)
        )
        if sha256_array(matrix) != str(source_audit.loc[time_value, "matrix_sha256"]):
            raise RuntimeError(f"M_per_source reconstruction hash failed at {time_value:g}")
        row_sums = matrix.sum(axis=1, keepdims=True)
        m_row = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums > 0)
        nonzero_rows = row_sums[:, 0] > 0
        if not np.allclose(m_row[nonzero_rows].sum(axis=1), 1.0, rtol=0, atol=1e-12):
            raise RuntimeError(f"M_row normalization failed at {time_value:g}")
        key = str(float(time_value))
        communications[key] = {"types": np.asarray(types), "M_per_source": matrix, "M_row": m_row}
        rows.append({
            "time": time_value,
            "n_types": len(types),
            "m_per_source_sha256": sha256_array(matrix),
            "m_per_source_sum": float(matrix.sum()),
            "m_row_sha256": sha256_array(m_row),
            "m_row_sum": float(m_row.sum()),
            "nonzero_sender_rows": int(nonzero_rows.sum()),
            "max_nonzero_row_sum_error": float(
                np.max(np.abs(m_row[nonzero_rows].sum(axis=1) - 1.0), initial=0.0)
            ),
        })
    return communications, pd.DataFrame(rows)


def pulse_table(normalized: pd.DataFrame, assignments: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    table = normalized.copy()
    table.columns = [float(value) for value in table.columns]
    profile_to_pair = assignments.set_index("profile")["pair_id"].astype(str)
    if set(table.index.astype(str)) == set(profile_to_pair.index.astype(str)):
        table.index = [profile_to_pair.loc[str(value)] for value in table.index]
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    for center in (0.5, 1.0, 1.5, 2.0, 2.5):
        left, right = center - 0.5, center + 0.5
        prominence = table[center] - 0.5 * (table[left] + table[right])
        summary[str(center)] = {
            "median_normalized_prominence": float(np.median(prominence)),
            "fraction_prominence_gt_0p25": float(np.mean(prominence > 0.25)),
            "fraction_prominence_lt_minus_0p25": float(np.mean(prominence < -0.25)),
        }
        rows.extend({"pair_id": pair_id, "center_time": center,
                     "normalized_prominence": float(value)}
                    for pair_id, value in prominence.items())
    return pd.DataFrame(rows), summary


def submitted_profile_audit(normalized: pd.DataFrame, assignments: pd.DataFrame) -> pd.DataFrame:
    assignment = assignments.set_index("profile")
    rows: list[dict[str, Any]] = []
    for position, (submitted_pattern, profile) in enumerate(OLD_SUBMITTED_PROFILES, start=1):
        present = profile in assignment.index and profile in normalized.index
        row: dict[str, Any] = {
            "display_order": position,
            "submitted_pattern": submitted_pattern,
            "profile": profile,
            "present_in_corrected_uniform_universe": bool(present),
        }
        if present:
            record = assignment.loc[profile]
            row.update({"corrected_m_row_cluster": int(record["cluster"]),
                        **{f"normalized_t{time:g}": float(normalized.at[profile, time]) for time in TIMES}})
        rows.append(row)
    return pd.DataFrame(rows)


def seal(root: Path) -> None:
    (root / "COMPLETE").write_text("complete\n", encoding="utf-8")
    paths = sorted(path for path in root.rglob("*") if path.is_file() and path.name != "SHA256SUMS.txt")
    (root / "SHA256SUMS.txt").write_text(
        "\n".join(f"{sha256_file(path)}  {path.relative_to(root)}" for path in paths) + "\n",
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
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(output)
    if (package_root / "RELEASE_COMMIT").read_text().strip() != EXPECTED_COMMIT:
        raise RuntimeError("Package commit mismatch")
    if (package_root / "ARCHIVE_SHA256").read_text().strip() != EXPECTED_ARCHIVE:
        raise RuntimeError("Package archive mismatch")
    if sha256_file(reference_path) != EXPECTED_REFERENCE:
        raise RuntimeError("Reference hash mismatch")
    source_contract = verify_checksums(source_root, "SHA256SUMS.txt")
    shared_contract = verify_checksums(shared_root, "SHA256SUMS.txt")

    sys.path.insert(0, str(package_root))
    import CytoBridge as cb
    if not str(Path(cb.__file__).resolve()).startswith(str(package_root)):
        raise RuntimeError(f"Wrong CytoBridge import: {cb.__file__}")

    output.mkdir(parents=True, exist_ok=False)
    tables = output / "tables"
    provenance = output / "provenance"
    tables.mkdir(); provenance.mkdir()
    shutil.copy2(Path(__file__).resolve(), provenance / Path(__file__).name)
    start = time.perf_counter()

    communications, matrix_audit = reconstruct_communications(source_root)
    matrix_audit.to_csv(tables / "communication_component_audit.csv", index=False)
    states = {
        str(float(time_value)): ad.read_h5ad(
            shared_root / "generated_states" / f"time_{time_token(time_value)}.h5ad"
        )
        for time_value in TIMES
    }
    reference = ad.read_h5ad(reference_path, backed="r")
    reconstruction = cb.tl.make_pca_reconstruction_spec(
        reference.var_names.astype(str),
        np.asarray(reference.varm["PCs"], dtype=np.float32),
        np.asarray(reference.var["pca_center"], dtype=np.float32),
        metadata={"source": str(reference_path), "source_sha256": EXPECTED_REFERENCE,
                  "expression_contract": "all_generated_inverse_pca"},
    )
    lr_database = package_root / "CytoBridge" / "workflow_databases" / "CellChatDB.ligrec.mouse.csv"
    result = cb.tl.project_communication_to_lr_timecourses(
        states, reference, communications, lr_database,
        time_points=TIMES, annotation_key="Annotation", matrix_key="M_row",
        spatial_dim=2, loadings_key="PCs", reference_layer=None,
        expression_space="count", complex_mode="min", require_all_subunits=True,
        duplicate_policy="first", preferred_species_tag=None, n_clusters=3,
        pca_reconstruction=reconstruction, profile_linkage_method="average",
        profile_cluster_order="peak_time", observed_adata=None,
        observed_time_points=None, return_type_matrices=False,
    )
    reference.file.close()

    result.pair_timecourse.to_csv(tables / "lr_pair_timecourse_m_row.csv", index=False)
    result.clustering.normalized_profiles.to_csv(tables / "lr_normalized_profiles_m_row.csv")
    result.clustering.assignments.to_csv(tables / "lr_pattern_assignments_m_row.csv", index=False)
    result.clustering.prototypes.to_csv(tables / "lr_pattern_prototypes_m_row.csv", index=False)
    result.clustering.diagnostics.to_csv(tables / "lr_pattern_diagnostics_m_row.csv", index=False)
    result.coverage.to_csv(tables / "lr_projection_coverage_m_row.csv", index=False)
    (tables / "lr_projection_settings_m_row.json").write_text(
        json.dumps(jsonable(dict(result.settings)), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    pulse, pulse_summary = pulse_table(result.clustering.normalized_profiles, result.clustering.assignments)
    pulse.to_csv(tables / "m_row_pulse_diagnostics.csv", index=False)
    submitted = submitted_profile_audit(result.clustering.normalized_profiles, result.clustering.assignments)
    submitted.to_csv(tables / "submitted_profile_m_row_audit.csv", index=False)

    source_assignments = pd.read_csv(source_root / "tables" / "lr_pattern_assignments.csv")
    retained_mrow = set(result.clustering.assignments["pair_id"].astype(str))
    retained_source = set(source_assignments["pair_id"].astype(str))
    if retained_mrow != retained_source:
        raise RuntimeError("M_row changed the uniform LR pair universe")
    cluster_sizes = {
        str(int(key)): int(value)
        for key, value in result.clustering.assignments["cluster"].value_counts().sort_index().items()
    }
    summary = {
        "schema_version": 1,
        "status": "COMPLETE_AUDIT_NOT_FINAL",
        "dataset": "MOSTA",
        "panel": "S11",
        "question": "Does package-native M_row remove global M_per_source magnitude from normalized LR shapes?",
        "invariants": {
            "states": "same fully generated global-t0 seven half-step states",
            "sampled_attention": "same sealed seed42 12000-cell communication run",
            "expression": "same inverse-PCA count-space means at every time",
            "lr_database": {"path": str(lr_database), "sha256": sha256_file(lr_database)},
            "complex_mode": "min", "require_all_subunits": True,
            "retained_pair_universe_exact": True, "n_retained_pairs": len(retained_mrow),
            "clustering": "package-native minmax + average linkage k=3 + peak-time order",
        },
        "tested_change": {
            "from": "M_per_source (absolute attention per source cell)",
            "to": "M_row (each nonzero sender row sums to one; relative receiver allocation)",
            "package_supported_matrix_key": True,
            "manual_smoothing": False,
        },
        "cluster_sizes_m_row": cluster_sizes,
        "diagnostics_m_row": result.clustering.diagnostics.to_dict(orient="records"),
        "pulse_metrics_m_row": pulse_summary,
        "submitted_profiles": {
            "n": len(submitted),
            "all_present": bool(submitted["present_in_corrected_uniform_universe"].all()),
            "corrected_cluster_counts": submitted["corrected_m_row_cluster"].value_counts().sort_index().to_dict(),
        },
        "source_m_per_source_run": source_contract,
        "shared_state_run": shared_contract,
        "package": {"root": str(package_root), "commit": EXPECTED_COMMIT,
                    "archive_sha256": EXPECTED_ARCHIVE, "module": str(Path(cb.__file__).resolve()),
                    "projection_signature": str(inspect.signature(cb.tl.project_communication_to_lr_timecourses))},
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
