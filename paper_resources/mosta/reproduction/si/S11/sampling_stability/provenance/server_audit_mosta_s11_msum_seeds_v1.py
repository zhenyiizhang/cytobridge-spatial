#!/usr/bin/env python3
"""Audit MOSTA S11 package-native M_sum across communication sampling seeds.

The accepted model, fully generated global-t0 states, inverse-PCA expression,
LR database, strict complex contract, 12k sample size, and clustering settings
are invariant.  Only the deterministic uniform communication sample seed is
changed from the sealed seed-42 baseline to seeds 43 and 44.
"""

from __future__ import annotations

import argparse
import gc
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
import torch
from sklearn.metrics import adjusted_rand_score


EXPECTED_COMMIT = "2b3c79eff3face7c4dd33de24d45384b9dbd8a84"
EXPECTED_ARCHIVE = "06852992db0d8ebedd8f0baa19a3f539bcdd271dfc0c6fae9774dd553c8fe55e"
EXPECTED_REFERENCE = "8b9ca0ad3475040235036548d54b96272bf6c49f057f6c2a643152c11350ce25"
EXPECTED_FINETUNE = "d08d21b86fdcd92c748bb54ca81f998fbb157ca5a7acd9548bbfe16c573bfaa5"
EXPECTED_SCORE = "d7d06657f8548618db1bc85409e73305fbf59feb9446793550b2c9761639e52a"
TIMES = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
SEEDS = (43, 44)
N_COMMUNICATION = 12_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", required=True)
    parser.add_argument("--shared-state-root", required=True)
    parser.add_argument("--reference-h5ad", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--source-s11-root", required=True)
    parser.add_argument("--baseline-component-root", required=True)
    parser.add_argument("--base-audit-script", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
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


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def select_indices(
    states: dict[str, ad.AnnData], seed: int
) -> tuple[dict[str, np.ndarray], pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    selected: dict[str, np.ndarray] = {}
    inventory_rows: list[dict[str, Any]] = []
    count_rows: list[dict[str, Any]] = []
    for time_value in TIMES:
        key = str(float(time_value))
        state = states[key]
        indices = np.sort(
            rng.choice(state.n_obs, size=N_COMMUNICATION, replace=False)
        ).astype(np.int64, copy=False)
        selected[key] = indices
        labels_all = state.obs["Annotation"].astype(str)
        labels = labels_all.iloc[indices]
        inventory_rows.append(
            {
                "seed": seed,
                "time": time_value,
                "available_cells": state.n_obs,
                "communication_cells": len(indices),
                "n_available_labels": labels_all.nunique(),
                "n_sampled_labels": labels.nunique(),
                "sampled_indices_sha256": sha256_array(indices),
            }
        )
        available = labels_all.value_counts()
        sampled = labels.value_counts()
        for cell_type in sorted(available.index.astype(str)):
            count_rows.append(
                {
                    "seed": seed,
                    "time": time_value,
                    "cell_type": cell_type,
                    "n_available": int(available.get(cell_type, 0)),
                    "n_sampled": int(sampled.get(cell_type, 0)),
                }
            )
    return selected, pd.DataFrame(inventory_rows), pd.DataFrame(count_rows)


def profiles_by_pair(normalized: pd.DataFrame, assignments: pd.DataFrame) -> pd.DataFrame:
    table = normalized.copy()
    table.columns = [float(value) for value in table.columns]
    table.index = table.index.astype(str)
    mapping = assignments.set_index("profile")["pair_id"].astype(str)
    if set(table.index) == set(mapping.index):
        table.index = [mapping.loc[value] for value in table.index]
    elif set(table.index) != set(assignments["pair_id"].astype(str)):
        raise RuntimeError("Cannot align normalized profiles to pair_id")
    return table.loc[:, list(TIMES)].sort_index()


def compare_clusterings(
    left_name: str,
    left_assignments: pd.DataFrame,
    left_profiles: pd.DataFrame,
    right_name: str,
    right_assignments: pd.DataFrame,
    right_profiles: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    left = left_assignments[["pair_id", "cluster"]].copy()
    right = right_assignments[["pair_id", "cluster"]].copy()
    left["pair_id"] = left["pair_id"].astype(str)
    right["pair_id"] = right["pair_id"].astype(str)
    merged = left.merge(right, on="pair_id", suffixes=("_left", "_right"), validate="one_to_one")
    if len(merged) != len(left) or len(merged) != len(right):
        raise RuntimeError(f"Pair universe mismatch: {left_name} vs {right_name}")
    confusion = pd.crosstab(
        merged["cluster_left"], merged["cluster_right"], dropna=False
    ).rename_axis(index=f"cluster_{left_name}", columns=f"cluster_{right_name}").reset_index()
    left_profiles = left_profiles.loc[sorted(left_profiles.index)]
    right_profiles = right_profiles.loc[left_profiles.index]
    correlations = []
    rmsd = []
    for pair_id in left_profiles.index:
        x = left_profiles.loc[pair_id].to_numpy(dtype=float)
        y = right_profiles.loc[pair_id].to_numpy(dtype=float)
        correlations.append(float(np.corrcoef(x, y)[0, 1]))
        rmsd.append(float(np.sqrt(np.mean((x - y) ** 2))))
    profile_comparison = pd.DataFrame(
        {"pair_id": left_profiles.index, "pearson": correlations, "rmsd": rmsd}
    )
    summary = {
        "left": left_name,
        "right": right_name,
        "adjusted_rand_index": float(
            adjusted_rand_score(merged["cluster_left"], merged["cluster_right"])
        ),
        "fraction_same_peak_ordered_cluster": float(
            np.mean(merged["cluster_left"] == merged["cluster_right"])
        ),
        "median_pair_profile_pearson": float(np.median(correlations)),
        "q10_pair_profile_pearson": float(np.quantile(correlations, 0.10)),
        "median_pair_profile_rmsd": float(np.median(rmsd)),
        "q90_pair_profile_rmsd": float(np.quantile(rmsd, 0.90)),
    }
    return summary, confusion, profile_comparison


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
    model_dir = Path(args.model_dir).resolve()
    source_root = Path(args.source_s11_root).resolve()
    baseline_root = Path(args.baseline_component_root).resolve()
    helper_path = Path(args.base_audit_script).resolve()
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(output)
    if (package_root / "RELEASE_COMMIT").read_text().strip() != EXPECTED_COMMIT:
        raise RuntimeError("Package commit mismatch")
    if (package_root / "ARCHIVE_SHA256").read_text().strip() != EXPECTED_ARCHIVE:
        raise RuntimeError("Package archive mismatch")
    hashes = {
        "reference": sha256_file(reference_path),
        "finetune": sha256_file(model_dir / "Finetune" / "best_model.pth"),
        "score": sha256_file(model_dir / "Score_Refine" / "score_model.pth"),
    }
    if hashes != {
        "reference": EXPECTED_REFERENCE,
        "finetune": EXPECTED_FINETUNE,
        "score": EXPECTED_SCORE,
    }:
        raise RuntimeError(f"Accepted model/reference hash mismatch: {hashes}")
    helper = load_module(helper_path, "s11_seed_helper")
    source_contract = helper.verify_checksums(source_root, "SHA256SUMS.txt")
    shared_contract = helper.verify_checksums(shared_root, "SHA256SUMS.txt")
    baseline_contract = helper.verify_checksums(baseline_root, "SHA256SUMS.txt")

    sys.path.insert(0, str(package_root))
    import CytoBridge as cb
    if not str(Path(cb.__file__).resolve()).startswith(str(package_root)):
        raise RuntimeError(f"Wrong CytoBridge import: {cb.__file__}")

    output.mkdir(parents=True, exist_ok=False)
    tables = output / "tables"
    attention_root = output / "attention_sparse"
    provenance = output / "provenance"
    tables.mkdir(); attention_root.mkdir(); provenance.mkdir()
    shutil.copy2(Path(__file__).resolve(), provenance / Path(__file__).name)
    shutil.copy2(helper_path, provenance / helper_path.name)
    start = time.perf_counter()

    states = {
        str(float(time_value)): ad.read_h5ad(
            shared_root / "generated_states" / f"time_{time_token(time_value)}.h5ad"
        )
        for time_value in TIMES
    }
    loaded = cb.tl.load_dynamical_model_from_dir(model_dir, dim=52, device=args.device)
    if {"weight_stage": loaded.weight_stage, "score_stage": loaded.score_stage} != {
        "weight_stage": "Finetune", "score_stage": "Score_Refine"
    }:
        raise RuntimeError("Loaded checkpoint stage mismatch")
    runtime = cb.tl.build_dynamical_runtime(loaded)
    reference = ad.read_h5ad(reference_path, backed="r")
    reconstruction = cb.tl.make_pca_reconstruction_spec(
        reference.var_names.astype(str),
        np.asarray(reference.varm["PCs"], dtype=np.float32),
        np.asarray(reference.var["pca_center"], dtype=np.float32),
        metadata={"source": str(reference_path), "source_sha256": EXPECTED_REFERENCE,
                  "expression_contract": "all_generated_inverse_pca"},
    )
    lr_database = package_root / "CytoBridge" / "workflow_databases" / "CellChatDB.ligrec.mouse.csv"

    baseline_assignments = pd.read_csv(
        baseline_root / "tables" / "M_sum" / "lr_pattern_assignments.csv"
    )
    baseline_normalized = pd.read_csv(
        baseline_root / "tables" / "M_sum" / "lr_normalized_profiles.csv", index_col=0
    )
    all_assignments = {"seed42": baseline_assignments}
    all_profiles = {
        "seed42": profiles_by_pair(baseline_normalized, baseline_assignments)
    }
    seed_summaries: dict[str, Any] = {}
    inventory_tables = []
    count_tables = []
    for seed in SEEDS:
        name = f"seed{seed}"
        seed_tables = tables / name
        seed_attention = attention_root / name
        seed_tables.mkdir(); seed_attention.mkdir()
        selected, inventory, counts = select_indices(states, seed)
        inventory_tables.append(inventory); count_tables.append(counts)
        cb.tl.set_global_random_seed(seed)
        communications = cb.tl.compute_timepoint_communications(
            adata_dict=states, time_points=TIMES, annotation_key="Annotation",
            f_net=runtime.f_net, device=args.device, out_dir=str(seed_attention),
            save_dense_attention_matrix=False, remove_self_loop=True,
            winsor_quantile=0.995, save_pickle_path=None,
            max_cells_per_timepoint=None, random_seed=seed,
            cell_indices_by_time=selected,
        )
        matrix_rows = []
        for time_value in TIMES:
            record = communications[str(float(time_value))]
            matrix = np.asarray(record["M_sum"], dtype=np.float64)
            matrix_rows.append(
                {
                    "seed": seed,
                    "time": time_value,
                    "n_types": len(record["types"]),
                    "types": ";".join(map(str, record["types"])),
                    "m_sum_sha256": sha256_array(matrix),
                    "m_sum_sum": float(matrix.sum()),
                    "m_sum_max": float(matrix.max(initial=0.0)),
                }
            )
        pd.DataFrame(matrix_rows).to_csv(seed_tables / "communication_M_sum_audit.csv", index=False)
        result = cb.tl.project_communication_to_lr_timecourses(
            states, reference, communications, lr_database,
            time_points=TIMES, annotation_key="Annotation", matrix_key="M_sum",
            spatial_dim=2, loadings_key="PCs", reference_layer=None,
            expression_space="count", complex_mode="min", require_all_subunits=True,
            duplicate_policy="first", preferred_species_tag=None, n_clusters=3,
            pca_reconstruction=reconstruction, profile_linkage_method="average",
            profile_cluster_order="peak_time", observed_adata=None,
            observed_time_points=None, return_type_matrices=False,
        )
        result.pair_timecourse.to_csv(seed_tables / "lr_pair_timecourse.csv", index=False)
        result.clustering.normalized_profiles.to_csv(seed_tables / "lr_normalized_profiles.csv")
        result.clustering.assignments.to_csv(seed_tables / "lr_pattern_assignments.csv", index=False)
        result.clustering.prototypes.to_csv(seed_tables / "lr_pattern_prototypes.csv", index=False)
        result.clustering.diagnostics.to_csv(seed_tables / "lr_pattern_diagnostics.csv", index=False)
        pulse, pulse_metrics = helper.pulse_table(
            result.clustering.normalized_profiles, result.clustering.assignments
        )
        pulse.to_csv(seed_tables / "pulse_diagnostics.csv", index=False)
        assignments = result.clustering.assignments.copy()
        profiles = profiles_by_pair(result.clustering.normalized_profiles, assignments)
        all_assignments[name] = assignments
        all_profiles[name] = profiles
        seed_summaries[name] = {
            "cluster_sizes": {
                str(int(key)): int(value)
                for key, value in assignments["cluster"].value_counts().sort_index().items()
            },
            "diagnostics": result.clustering.diagnostics.to_dict(orient="records"),
            "pulse_metrics": pulse_metrics,
            "n_retained_pairs": len(assignments),
        }
        del communications, result
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    pd.concat(inventory_tables, ignore_index=True).to_csv(
        tables / "communication_sampling_inventory.csv", index=False
    )
    pd.concat(count_tables, ignore_index=True).to_csv(
        tables / "communication_sampling_celltype_counts.csv", index=False
    )
    reference.file.close()

    comparisons = []
    for left, right in (("seed42", "seed43"), ("seed42", "seed44"), ("seed43", "seed44")):
        comparison, confusion, profile_table = compare_clusterings(
            left, all_assignments[left], all_profiles[left],
            right, all_assignments[right], all_profiles[right],
        )
        comparisons.append(comparison)
        confusion.to_csv(tables / f"cluster_confusion_{left}_vs_{right}.csv", index=False)
        profile_table.to_csv(tables / f"profile_similarity_{left}_vs_{right}.csv", index=False)

    summary = {
        "schema_version": 1,
        "status": "COMPLETE_AUDIT_NOT_FINAL",
        "dataset": "MOSTA",
        "panel": "S11",
        "question": "Are package-native M_sum temporal patterns stable to the 12k communication sample seed?",
        "invariants": {
            "model_and_reference_hashes": hashes,
            "states": "same sealed 50k global-t0 fully generated states",
            "communication_sample_size": N_COMMUNICATION,
            "tested_seeds": [42, 43, 44],
            "matrix_key": "M_sum",
            "expression": "inverse PCA count space at all seven times",
            "lr_complex": "min, require all subunits",
            "clustering": "package-native minmax, average linkage k=3, peak-time order",
            "manual_smoothing": False,
        },
        "seed_results": seed_summaries,
        "comparisons": comparisons,
        "source_seed42_run": source_contract,
        "baseline_component_run": baseline_contract,
        "shared_states": shared_contract,
        "package": {
            "root": str(package_root), "module": str(Path(cb.__file__).resolve()),
            "commit": EXPECTED_COMMIT, "archive_sha256": EXPECTED_ARCHIVE,
            "communication_signature": str(inspect.signature(cb.tl.compute_timepoint_communications)),
            "projection_signature": str(inspect.signature(cb.tl.project_communication_to_lr_timecourses)),
        },
        "runtime": {
            "python": sys.version, "platform": platform.platform(),
            "torch": torch.__version__, "cuda": torch.version.cuda, "device": args.device,
            "cuda_device_name": torch.cuda.get_device_name(torch.cuda.current_device())
            if torch.cuda.is_available() else None,
        },
        "wall_seconds": time.perf_counter() - start,
    }
    (output / "summary.json").write_text(
        json.dumps(jsonable(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    seal(output)
    print(json.dumps(jsonable(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
