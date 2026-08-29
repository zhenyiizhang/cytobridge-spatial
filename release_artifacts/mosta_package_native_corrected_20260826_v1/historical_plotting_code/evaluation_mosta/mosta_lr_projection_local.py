#!/usr/bin/env python3
"""Project cell-type communication matrices to LR interaction scores.

This script replaces the fragile LR projection cells in
`mosta_LR_interaction_v0.ipynb` with a consistent implementation:

1) use communication matrices already aggregated by cell type (recommended: M_per_source),
2) compute cell-type mean ligand/receptor expression in count space,
3) compute LR score matrix as:
   LR(A->B) = mean_expr_A(ligand) * mean_expr_B(receptor) * comm(A->B).

It supports both:
- real timepoints from a reference h5ad (filtered by obs[timepoint_col]),
- interpolated timepoints from per-timepoint h5ad files (JSON mapping).
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse


DEFAULT_COMM_PKL = "evaluation/mosta/data/all_timepoint_communications_merged.pkl"
DEFAULT_REAL_H5AD = "spatial_data/Mouse_embryo_all_stage.h5ad"
DEFAULT_LR_DB = "database/CellChatDB.ligrec.mouse.csv"
DEFAULT_OUTPUT_DIR = "results/mosta_lr_projection_local"


def _split_csv(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _normalize_time_key(s: str) -> str:
    raw = str(s).strip()
    try:
        return str(float(raw))
    except Exception:
        return raw


def _sanitize_key(s: str) -> str:
    out = str(s).strip()
    out = out.replace("/", "_").replace("\\", "_")
    out = out.replace(" ", "_")
    return out


def _resolve_column(columns: Sequence[str], requested: str, fallbacks: Sequence[str]) -> str:
    cols = list(columns)
    if requested in cols:
        return requested
    low_to_col = {c.lower(): c for c in cols}
    if requested.lower() in low_to_col:
        return low_to_col[requested.lower()]
    for fb in fallbacks:
        if fb in cols:
            return fb
        if fb.lower() in low_to_col:
            return low_to_col[fb.lower()]
    raise KeyError(f"Could not resolve column '{requested}' from {cols}")


def _load_communications(path: str) -> Dict[str, dict]:
    with open(path, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"Expected dict in communications pickle: {path}")
    return {_normalize_time_key(k): v for k, v in data.items()}


def _load_lr_db(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"LR database is empty: {path}")

    candidates_ligand = ["ligand", "ligand_symbol", "source", "gene_a", "0"]
    candidates_receptor = ["receptor", "receptor_symbol", "target", "gene_b", "1"]

    cols = list(df.columns)
    low_to_col = {c.lower(): c for c in cols}

    lig_col = next((low_to_col[c] for c in candidates_ligand if c in low_to_col), None)
    rec_col = next((low_to_col[c] for c in candidates_receptor if c in low_to_col), None)

    # Fallback for CSVs with shape like: ['Unnamed: 0', '0', '1', ...]
    if lig_col is None or rec_col is None:
        non_unnamed = [c for c in cols if not c.lower().startswith("unnamed")]
        if len(non_unnamed) >= 2:
            lig_col = non_unnamed[0]
            rec_col = non_unnamed[1]

    if lig_col is None or rec_col is None:
        raise ValueError(f"Could not identify ligand/receptor columns in LR DB: {path}, cols={cols}")

    out = df[[lig_col, rec_col]].rename(columns={lig_col: "ligand", rec_col: "receptor"}).copy()
    out["ligand"] = out["ligand"].astype(str).str.strip()
    out["receptor"] = out["receptor"].astype(str).str.strip()
    out = out[(out["ligand"] != "") & (out["receptor"] != "")]
    out = out.drop_duplicates().reset_index(drop=True)
    return out


def _complex_subunits(token: str) -> List[str]:
    return [x.strip() for x in str(token).split("_") if x.strip()]


def _collect_required_genes(lr_df: pd.DataFrame) -> List[str]:
    genes = set()
    for _, row in lr_df.iterrows():
        genes.update(_complex_subunits(row["ligand"]))
        genes.update(_complex_subunits(row["receptor"]))
    return sorted(genes)


def _combine_vectors(vectors: List[np.ndarray], mode: str) -> np.ndarray:
    if len(vectors) == 1:
        return vectors[0]
    arr = np.stack(vectors, axis=0)
    if mode == "min":
        return arr.min(axis=0)
    if mode == "mean":
        return arr.mean(axis=0)
    if mode == "product":
        return arr.prod(axis=0)
    raise ValueError(f"Unsupported complex combine mode: {mode}")


def _get_complex_expr_vector(
    token: str,
    gene_to_vec: Mapping[str, np.ndarray],
    complex_mode: str,
) -> Tuple[Optional[np.ndarray], List[str]]:
    subunits = _complex_subunits(token)
    present = [g for g in subunits if g in gene_to_vec]
    missing = [g for g in subunits if g not in gene_to_vec]
    if not present:
        return None, missing
    vecs = [gene_to_vec[g] for g in present]
    return _combine_vectors(vecs, complex_mode), missing


def _compute_gene_means_by_cell_type(
    adata,
    annotation_col: str,
    cell_types: Sequence[str],
    required_genes: Sequence[str],
    x_space: str,
) -> Tuple[Dict[str, np.ndarray], Dict[str, int], List[str]]:
    var_names = pd.Index(adata.var_names.astype(str))
    present_genes = [g for g in required_genes if g in var_names]
    missing_genes = [g for g in required_genes if g not in var_names]
    if not present_genes:
        return {}, {str(ct): 0 for ct in cell_types}, missing_genes

    col_idx = [int(var_names.get_loc(g)) for g in present_genes]
    X = adata[:, col_idx].X
    if sparse.issparse(X):
        X = X.tocsr(copy=True)
        if x_space == "log1p":
            X.data = np.expm1(X.data)
    else:
        X = np.asarray(X, dtype=np.float64)
        if x_space == "log1p":
            X = np.expm1(X)

    labels = adata.obs[annotation_col].astype(str).to_numpy()
    T = len(cell_types)
    G = len(present_genes)
    means = np.zeros((T, G), dtype=np.float64)
    counts: Dict[str, int] = {}

    for i, ct in enumerate(cell_types):
        mask = labels == str(ct)
        n = int(mask.sum())
        counts[str(ct)] = n
        if n == 0:
            continue
        idx = np.flatnonzero(mask)
        if sparse.issparse(X):
            means[i, :] = np.asarray(X[idx].mean(axis=0)).ravel()
        else:
            means[i, :] = X[idx].mean(axis=0)

    gene_to_vec = {g: means[:, j] for j, g in enumerate(present_genes)}
    return gene_to_vec, counts, missing_genes


def _validate_comm_matrix(comm_data: dict, matrix_key: str, cell_types: Sequence[str]) -> np.ndarray:
    if matrix_key not in comm_data:
        raise KeyError(f"matrix_key '{matrix_key}' missing in communication record keys={list(comm_data.keys())}")
    mat = np.asarray(comm_data[matrix_key], dtype=np.float64)
    t = len(cell_types)
    if mat.shape != (t, t):
        raise ValueError(f"Communication matrix shape mismatch: {mat.shape} vs ({t}, {t})")
    return mat


def project_lr_for_timepoint(
    adata,
    time_key: str,
    comm_data: dict,
    lr_df: pd.DataFrame,
    matrix_key: str,
    annotation_col: str,
    x_space: str,
    complex_mode: str,
    duplicate_policy: str,
) -> dict:
    cell_types = [str(x) for x in np.asarray(comm_data["types"]).tolist()]
    comm_matrix = _validate_comm_matrix(comm_data, matrix_key=matrix_key, cell_types=cell_types)

    required_genes = _collect_required_genes(lr_df)
    gene_to_vec, cell_type_counts, missing_genes = _compute_gene_means_by_cell_type(
        adata=adata,
        annotation_col=annotation_col,
        cell_types=cell_types,
        required_genes=required_genes,
        x_space=x_space,
    )

    lr_scores: Dict[str, np.ndarray] = {}
    skipped_no_gene = 0
    skipped_duplicate = 0
    duplicate_counter: Dict[str, int] = {}

    for _, row in lr_df.iterrows():
        ligand = str(row["ligand"])
        receptor = str(row["receptor"])
        pair_key = f"{ligand}_{receptor}"

        lig_vec, lig_missing = _get_complex_expr_vector(ligand, gene_to_vec, complex_mode=complex_mode)
        rec_vec, rec_missing = _get_complex_expr_vector(receptor, gene_to_vec, complex_mode=complex_mode)
        if lig_vec is None or rec_vec is None:
            skipped_no_gene += 1
            continue

        score = np.outer(lig_vec, rec_vec) * comm_matrix

        if pair_key in lr_scores:
            skipped_duplicate += 1
            duplicate_counter[pair_key] = duplicate_counter.get(pair_key, 1) + 1
            if duplicate_policy == "first":
                continue
            if duplicate_policy == "last":
                lr_scores[pair_key] = score
            elif duplicate_policy == "sum":
                lr_scores[pair_key] = lr_scores[pair_key] + score
            elif duplicate_policy == "max":
                lr_scores[pair_key] = np.maximum(lr_scores[pair_key], score)
            else:
                raise ValueError(f"Unsupported duplicate policy: {duplicate_policy}")
        else:
            lr_scores[pair_key] = score

        # keep lints quiet while making intent explicit
        _ = lig_missing, rec_missing

    return {
        "time_key": str(time_key),
        "lr_scores": lr_scores,
        "comm_matrix": comm_matrix,
        "matrix_key": matrix_key,
        "cell_types": np.asarray(cell_types, dtype=object),
        "cell_type_counts": cell_type_counts,
        "annotation_col": annotation_col,
        "x_space": x_space,
        "complex_mode": complex_mode,
        "n_lr_pairs_db": int(len(lr_df)),
        "n_lr_pairs_scored": int(len(lr_scores)),
        "n_skipped_missing_genes": int(skipped_no_gene),
        "n_skipped_duplicates": int(skipped_duplicate),
        "missing_genes": sorted(missing_genes),
        "duplicate_pairs": duplicate_counter,
    }


@dataclass
class RealH5adIndex:
    adata: object
    timepoint_col: str
    annotation_col: str
    index_by_time: Dict[str, np.ndarray]


def build_real_h5ad_index(
    h5ad_path: str,
    timepoint_col: str,
    annotation_col: str,
    backed: bool,
) -> RealH5adIndex:
    adata = ad.read_h5ad(h5ad_path, backed="r" if backed else None)
    tp_col = _resolve_column(adata.obs.columns, timepoint_col, ["timepoint", "samples", "time", "batch", "Batch"])
    ann_col = _resolve_column(adata.obs.columns, annotation_col, ["annotation", "Annotation", "cell_type"])

    t_values = adata.obs[tp_col].astype(str).to_numpy()
    index_by_time: Dict[str, np.ndarray] = {}
    for t in np.unique(t_values):
        index_by_time[str(t)] = np.flatnonzero(t_values == str(t))

    return RealH5adIndex(
        adata=adata,
        timepoint_col=tp_col,
        annotation_col=ann_col,
        index_by_time=index_by_time,
    )


def load_real_time_slice(real_index: RealH5adIndex, time_key: str):
    key = str(time_key)
    if key not in real_index.index_by_time:
        available = sorted(real_index.index_by_time.keys())
        raise KeyError(f"Time '{key}' not found in real h5ad. Available: {available}")
    idx = real_index.index_by_time[key]
    sub = real_index.adata[idx, :]
    if hasattr(sub, "to_memory"):
        sub = sub.to_memory()
    else:
        sub = sub.copy()
    return sub


def load_interp_map(path: Optional[str]) -> Dict[str, str]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"Interpolation map JSON must be a dict time_key->h5ad_path: {path}")
    out = {_normalize_time_key(k): str(v) for k, v in data.items()}
    return out


def parse_time_key_map(mapping: Optional[str]) -> Dict[str, str]:
    """Parse mapping string like: '0=E12.5,1=E13.5,2=E14.5,3=E15.5'."""
    if not mapping:
        return {}
    out: Dict[str, str] = {}
    for part in _split_csv(mapping):
        if "=" not in part:
            raise ValueError(f"Invalid --time-key-map entry (need src=dst): {part}")
        src, dst = part.split("=", 1)
        src_k = _normalize_time_key(src)
        dst_k = str(dst).strip()
        if not dst_k:
            raise ValueError(f"Invalid --time-key-map target in entry: {part}")
        out[src_k] = dst_k
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Project communication matrices to LR interaction scores.")
    p.add_argument("--communications-pkl", default=DEFAULT_COMM_PKL)
    p.add_argument("--lr-db", default=DEFAULT_LR_DB)
    p.add_argument("--real-h5ad", default=DEFAULT_REAL_H5AD)
    p.add_argument("--interp-map-json", default=None, help="Optional JSON dict: time_key -> h5ad path.")
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--matrix-key", default="M_per_source", choices=["M_per_source", "M_sum", "M_row", "M_mean"])
    p.add_argument("--timepoint-col", default="timepoint")
    p.add_argument("--annotation-col", default="annotation")
    p.add_argument("--time-keys", default="all", help="Comma-separated list, or 'all'.")
    p.add_argument(
        "--time-key-map",
        default=None,
        help="Optional mapping from communication keys to real h5ad time labels, e.g. '0=E12.5,1=E13.5,2=E14.5,3=E15.5'.",
    )
    p.add_argument(
        "--save-key-mode",
        default="mapped",
        choices=["mapped", "comm"],
        help="Use mapped or original communication time key in output filename/time_key field.",
    )
    p.add_argument("--x-space", default="log1p", choices=["log1p", "count"])
    p.add_argument("--complex-mode", default="min", choices=["min", "mean", "product"])
    p.add_argument("--duplicate-policy", default="first", choices=["first", "last", "sum", "max"])
    p.add_argument("--real-h5ad-backed", action="store_true", default=True)
    p.add_argument("--no-real-h5ad-backed", dest="real_h5ad_backed", action="store_false")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    comm_all = _load_communications(args.communications_pkl)
    lr_df = _load_lr_db(args.lr_db)
    interp_map = load_interp_map(args.interp_map_json)
    time_key_map = parse_time_key_map(args.time_key_map)

    available_times = sorted(comm_all.keys())
    if args.time_keys.strip().lower() == "all":
        target_times = available_times
    else:
        target_times = [_normalize_time_key(t) for t in _split_csv(args.time_keys)]
        missing = [t for t in target_times if t not in comm_all]
        if missing:
            raise KeyError(f"Requested time_keys not in communications pickle: {missing}")

    print(f"Loaded communications: {args.communications_pkl}")
    print(f"  timepoints: {available_times}")
    print(f"Loaded LR DB: {args.lr_db}")
    print(f"  unique pairs: {len(lr_df)}")
    print(f"Target timepoints: {target_times}")
    if time_key_map:
        print(f"Time key map: {time_key_map}")

    if args.dry_run:
        print("[dry-run] no projection performed.")
        return

    real_index = build_real_h5ad_index(
        h5ad_path=args.real_h5ad,
        timepoint_col=args.timepoint_col,
        annotation_col=args.annotation_col,
        backed=bool(args.real_h5ad_backed),
    )
    print(f"Loaded real h5ad: {args.real_h5ad}")
    print(f"  resolved timepoint col: {real_index.timepoint_col}")
    print(f"  resolved annotation col: {real_index.annotation_col}")

    for t in target_times:
        comm_key = _normalize_time_key(t)
        adata_time_key = time_key_map.get(comm_key, comm_key)
        save_time_key = adata_time_key if args.save_key_mode == "mapped" else comm_key

        print(f"\n[comm_time={comm_key} -> adata_time={adata_time_key}]")
        comm_data = comm_all[comm_key]
        if comm_key in interp_map:
            adata_t = ad.read_h5ad(interp_map[comm_key])
            ann_col = _resolve_column(
                adata_t.obs.columns,
                args.annotation_col,
                [real_index.annotation_col, "annotation", "Annotation", "cell_type"],
            )
            print(f"  source: interp h5ad ({interp_map[comm_key]}) | cells={adata_t.n_obs}")
        else:
            adata_t = load_real_time_slice(real_index, adata_time_key)
            ann_col = real_index.annotation_col
            print(f"  source: real h5ad slice | cells={adata_t.n_obs}")

        result = project_lr_for_timepoint(
            adata=adata_t,
            time_key=save_time_key,
            comm_data=comm_data,
            lr_df=lr_df,
            matrix_key=args.matrix_key,
            annotation_col=ann_col,
            x_space=args.x_space,
            complex_mode=args.complex_mode,
            duplicate_policy=args.duplicate_policy,
        )
        result["comm_time_key"] = comm_key
        result["adata_time_key"] = adata_time_key

        out_name = f"lr_scores_{_sanitize_key(save_time_key)}.pkl"
        out_path = os.path.join(args.output_dir, out_name)
        with open(out_path, "wb") as f:
            pickle.dump(result, f)
        print(
            "  saved:",
            out_path,
            "| pairs_scored=",
            result["n_lr_pairs_scored"],
            "| skipped_missing_genes=",
            result["n_skipped_missing_genes"],
        )

    # close backed handle if possible
    try:
        if hasattr(real_index.adata, "file") and real_index.adata.file is not None:
            real_index.adata.file.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
