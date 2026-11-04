from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.linalg import orthogonal_procrustes
from scipy.stats import spearmanr


PROJECT_ROOT = Path("/data/cytobridge/projects/CytoBridge-ST-1104")
ACCEPTED_RUN = PROJECT_ROOT / "runs/chicken-heart-full-ot-20260823-r2"
AUDIT_ROOT = PROJECT_ROOT / "runs/chicken-heart-alignment-sensitivity-audit-20260831-r1"
RUNS_DIR = AUDIT_ROOT / "runs"
SUMMARY_DIR = AUDIT_ROOT / "summary"
TIME_ORDER = ("D4", "D7", "D10", "D14")
MODEL_TIMES = (0.0, 1.0, 2.0, 3.0)
VARIANTS = (
    "baseline_repeat",
    "translate_low",
    "translate_moderate",
    "translate_strong",
    "rotate_low",
    "rotate_moderate",
    "rotate_strong",
    "translate_rotate_low",
    "translate_rotate_moderate",
    "translate_rotate_strong",
)


def _run_root(variant: str) -> Path:
    return ACCEPTED_RUN if variant == "accepted_baseline" else RUNS_DIR / variant


def _load_run(variant: str):
    root = _run_root(variant)
    adata = sc.read_h5ad(root / "preprocess/chicken_heart_aligned.h5ad")
    velocity_path = root / "downstream/velocity/velocity_components.npz"
    velocity = np.load(velocity_path)
    if velocity["full"].shape[0] != adata.n_obs:
        raise ValueError(f"Velocity/AnnData row mismatch for {variant}")
    return root, adata, velocity


def _proper_rotation(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    rotation, _ = orthogonal_procrustes(source, target)
    if np.linalg.det(rotation) < 0:
        u, _, vt = np.linalg.svd(source.T @ target)
        u[:, -1] *= -1
        rotation = u @ vt
    return rotation


def _cosine_rows(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    nx = np.linalg.norm(x, axis=1)
    ny = np.linalg.norm(y, axis=1)
    valid = (nx > 1e-10) & (ny > 1e-10)
    values = np.full(len(x), np.nan, dtype=float)
    values[valid] = np.einsum("ij,ij->i", x[valid], y[valid]) / (nx[valid] * ny[valid])
    return values, valid


def _global_vector_cosine(x: np.ndarray, y: np.ndarray) -> float:
    """Cosine between two flattened vector fields.

    Unlike an unweighted mean of per-cell directions, this statistic is not
    dominated by cells whose interaction velocity is numerically close to zero.
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    return float(np.dot(x, y) / denominator) if denominator > 0 else np.nan


def _attention_paths(root: Path, model_time: float) -> tuple[Path, Path]:
    attention_dir = root / "downstream/communication/sparse_attention"
    suffix = f"{model_time:.1f}"
    return (
        attention_dir / f"edge_index_interp_t{suffix}.npy",
        attention_dir / f"attn_mean_interp_t{suffix}.npy",
    )


def _attention_table(root: Path, adata, timepoint: str, model_time: float) -> pd.DataFrame:
    edge_path, attention_path = _attention_paths(root, model_time)
    edge_index = np.load(edge_path).astype(int)
    attention = np.load(attention_path).astype(float)
    stage = adata[adata.obs["timepoint"].astype(str) == timepoint]
    obs_ids = stage.obs_names.astype(str).to_numpy()
    labels = stage.obs["celltype_prediction"].astype(str).to_numpy()
    if edge_index.shape[0] != 2 or edge_index.shape[1] != len(attention):
        raise ValueError(f"Malformed attention arrays in {root} at {timepoint}")
    if edge_index.size and (edge_index.min() < 0 or edge_index.max() >= len(obs_ids)):
        raise IndexError(f"Attention edge index outside {timepoint} slice in {root}")
    table = pd.DataFrame(
        {
            "src_id": obs_ids[edge_index[0]],
            "dst_id": obs_ids[edge_index[1]],
            "src_label": labels[edge_index[0]],
            "dst_label": labels[edge_index[1]],
            "attention": attention,
        }
    )
    return (
        table.groupby(["src_id", "dst_id", "src_label", "dst_label"], as_index=False)["attention"]
        .mean()
    )


def _safe_spearman(x: pd.Series, y: pd.Series) -> float:
    if len(x) < 2 or x.nunique() < 2 or y.nunique() < 2:
        return np.nan
    return float(spearmanr(x, y).statistic)


def _weighted_jaccard(x: np.ndarray, y: np.ndarray) -> float:
    x = np.clip(np.asarray(x, dtype=float), 0.0, None)
    y = np.clip(np.asarray(y, dtype=float), 0.0, None)
    if x.sum() > 0:
        x = x / x.sum()
    if y.sum() > 0:
        y = y / y.sum()
    denominator = np.maximum(x, y).sum()
    return float(np.minimum(x, y).sum() / denominator) if denominator > 0 else np.nan


def compare() -> None:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    velocity_rows: list[dict] = []
    coordinate_rows: list[dict] = []
    attention_rows: list[dict] = []
    celltype_rows: list[dict] = []
    order_rows: list[dict] = []

    for variant in VARIANTS:
        # The newly trained unperturbed run is the paired control for every
        # perturbation.  The accepted manuscript run is used only to quantify
        # repeat-run variability of that paired control.
        reference_variant = "accepted_baseline" if variant == "baseline_repeat" else "baseline_repeat"
        base_root, base, base_velocity = _load_run(reference_variant)
        root, current, current_velocity = _load_run(variant)
        base_index = pd.Index(base.obs_names.astype(str))
        current_index = pd.Index(current.obs_names.astype(str))
        current_positions = current_index.get_indexer(base_index)
        if (current_positions < 0).any() or len(current_index) != len(base_index):
            raise ValueError(f"Observation sets differ for {variant}")
        same_order = bool(base_index.equals(current_index))
        order_rows.append(
            {
                "variant": variant,
                "reference": reference_variant,
                "same_obs_order": same_order,
                "n_obs": len(base_index),
            }
        )
        current_coords_all = np.asarray(current.obsm["spatial_aligned"], dtype=float)[current_positions]
        base_coords_all = np.asarray(base.obsm["spatial_aligned"], dtype=float)

        for timepoint, model_time in zip(TIME_ORDER, MODEL_TIMES):
            mask = np.asarray(base.obs["timepoint"].astype(str) == timepoint)
            base_coords = base_coords_all[mask]
            current_coords = current_coords_all[mask]
            base_center = base_coords.mean(axis=0, keepdims=True)
            current_center = current_coords.mean(axis=0, keepdims=True)
            centered_base = base_coords - base_center
            centered_current = current_coords - current_center
            rotation = _proper_rotation(centered_current, centered_base)
            corrected_coords = centered_current @ rotation + base_center
            raw_rmsd = float(np.sqrt(np.mean((current_coords - base_coords) ** 2)))
            rigid_rmsd = float(np.sqrt(np.mean((corrected_coords - base_coords) ** 2)))
            baseline_radius = float(np.sqrt(np.mean(np.sum(centered_base**2, axis=1))))
            coordinate_rows.append(
                {
                    "variant": variant,
                    "reference": reference_variant,
                    "timepoint": timepoint,
                    "n_cells": int(mask.sum()),
                    "raw_coordinate_rmsd": raw_rmsd,
                    "rigid_adjusted_coordinate_rmsd": rigid_rmsd,
                    "rigid_adjusted_rmsd_fraction_of_baseline_radius": rigid_rmsd / baseline_radius,
                    "frame_rotation_deg": float(np.degrees(np.arctan2(rotation[0, 1], rotation[0, 0]))),
                    "frame_translation_l2": float(np.linalg.norm(base_center - current_center)),
                }
            )

            for velocity_key in ("full", "interaction"):
                base_vectors = np.asarray(base_velocity[velocity_key], dtype=float)[mask, :2]
                current_vectors = np.asarray(current_velocity[velocity_key], dtype=float)[current_positions][mask, :2]
                raw_cosine, raw_valid = _cosine_rows(base_vectors, current_vectors)
                corrected_vectors = current_vectors @ rotation
                corrected_cosine, corrected_valid = _cosine_rows(base_vectors, corrected_vectors)
                valid = raw_valid & corrected_valid
                base_norm = np.linalg.norm(base_vectors, axis=1)
                current_norm = np.linalg.norm(corrected_vectors, axis=1)
                meaningful_cutoff = float(np.quantile(base_norm[base_norm > 0], 0.10))
                meaningful = valid & (base_norm >= meaningful_cutoff) & (current_norm >= meaningful_cutoff)
                labels = base.obs.loc[mask, "celltype_prediction"].astype(str).to_numpy()
                velocity_rows.append(
                    {
                        "variant": variant,
                        "reference": reference_variant,
                        "timepoint": timepoint,
                        "velocity_key": velocity_key,
                        "n_cells": int(mask.sum()),
                        "n_nonzero_pairs": int(valid.sum()),
                        "zero_pair_fraction": float(1 - valid.mean()),
                        "mean_cosine_raw": float(np.nanmean(raw_cosine)),
                        "median_cosine_raw": float(np.nanmedian(raw_cosine)),
                        "mean_cosine_rigid_adjusted": float(np.nanmean(corrected_cosine)),
                        "median_cosine_rigid_adjusted": float(np.nanmedian(corrected_cosine)),
                        "meaningful_velocity_cutoff_baseline_p10": meaningful_cutoff,
                        "n_meaningful_pairs": int(meaningful.sum()),
                        "mean_cosine_meaningful": float(np.mean(corrected_cosine[meaningful])),
                        "median_cosine_meaningful": float(np.median(corrected_cosine[meaningful])),
                        "global_vector_cosine_rigid_adjusted": _global_vector_cosine(
                            base_vectors, corrected_vectors
                        ),
                        "magnitude_spearman": _safe_spearman(
                            pd.Series(base_norm), pd.Series(current_norm)
                        ),
                        "relative_vector_field_rmse": float(
                            np.linalg.norm(base_vectors - corrected_vectors)
                            / max(np.linalg.norm(base_vectors), 1e-12)
                        ),
                        "mean_delta_l2_rigid_adjusted": float(
                            np.mean(np.linalg.norm(base_vectors[valid] - corrected_vectors[valid], axis=1))
                        ),
                    }
                )
                per_celltype = pd.DataFrame(
                    {"celltype": labels, "cosine": corrected_cosine}
                ).dropna()
                for celltype, group in per_celltype.groupby("celltype"):
                    celltype_rows.append(
                        {
                            "variant": variant,
                            "reference": reference_variant,
                            "timepoint": timepoint,
                            "velocity_key": velocity_key,
                            "celltype": celltype,
                            "n_cells": len(group),
                            "mean_cosine_rigid_adjusted": float(group["cosine"].mean()),
                            "median_cosine_rigid_adjusted": float(group["cosine"].median()),
                        }
                    )

            base_attention = _attention_table(base_root, base, timepoint, model_time).rename(
                columns={"attention": "attention_base"}
            )
            current_attention = _attention_table(root, current, timepoint, model_time).rename(
                columns={"attention": "attention_variant"}
            )
            common = base_attention.merge(
                current_attention[["src_id", "dst_id", "attention_variant"]],
                on=["src_id", "dst_id"],
                how="inner",
            )
            union = base_attention[["src_id", "dst_id", "attention_base"]].merge(
                current_attention[["src_id", "dst_id", "attention_variant"]],
                on=["src_id", "dst_id"],
                how="outer",
            ).fillna(0.0)
            topk = min(500, len(base_attention), len(current_attention))
            top_base = set(
                base_attention.nlargest(topk, "attention_base")[["src_id", "dst_id"]]
                .itertuples(index=False, name=None)
            )
            top_current = set(
                current_attention.nlargest(topk, "attention_variant")[["src_id", "dst_id"]]
                .itertuples(index=False, name=None)
            )
            top_union = top_base | top_current

            base_sr = (
                base_attention.groupby(["src_label", "dst_label"])["attention_base"].mean().rename("base")
            )
            current_sr = (
                current_attention.groupby(["src_label", "dst_label"])["attention_variant"].mean().rename("variant")
            )
            common_sr = pd.concat([base_sr, current_sr], axis=1).dropna()
            union_sr = pd.concat([base_sr, current_sr], axis=1).fillna(0.0)
            attention_rows.append(
                {
                    "variant": variant,
                    "reference": reference_variant,
                    "timepoint": timepoint,
                    "n_edges_base": len(base_attention),
                    "n_edges_variant": len(current_attention),
                    "n_common_edges": len(common),
                    "edge_set_jaccard": len(common) / len(union) if len(union) else np.nan,
                    "attention_spearman_common_edges": _safe_spearman(
                        common["attention_base"], common["attention_variant"]
                    ),
                    "attention_weighted_jaccard_union": _weighted_jaccard(
                        union["attention_base"].to_numpy(), union["attention_variant"].to_numpy()
                    ),
                    "top500_jaccard": len(top_base & top_current) / len(top_union) if top_union else np.nan,
                    "n_sender_receiver_pairs_common": len(common_sr),
                    "n_sender_receiver_pairs_union": len(union_sr),
                    "sender_receiver_spearman_common": _safe_spearman(common_sr["base"], common_sr["variant"]),
                    "sender_receiver_spearman_union": _safe_spearman(union_sr["base"], union_sr["variant"]),
                }
            )

    tables = {
        "observation_order_check.csv": pd.DataFrame(order_rows),
        "coordinate_metrics.csv": pd.DataFrame(coordinate_rows),
        "velocity_metrics_pooled.csv": pd.DataFrame(velocity_rows),
        "velocity_metrics_by_celltype.csv": pd.DataFrame(celltype_rows),
        "interaction_metrics.csv": pd.DataFrame(attention_rows),
    }
    for filename, table in tables.items():
        table.to_csv(SUMMARY_DIR / filename, index=False)

    pooled = pd.DataFrame(velocity_rows).merge(
        pd.DataFrame(attention_rows),
        on=["variant", "reference", "timepoint"],
        how="left",
    )
    pooled.to_csv(SUMMARY_DIR / "reviewer_metrics.csv", index=False)
    summary = {
        "accepted_baseline": str(ACCEPTED_RUN),
        "accepted_commit": "c72e592d0dea70941bc4971a79c3c903d7454b08",
        "comparison_contract": {
            "cell_alignment": "obs_names",
            "edge_alignment": "source and target obs_names, not local integer positions",
            "velocity_primary": "pooled cell-level mean after per-timepoint proper rigid frame adjustment",
            "paired_reference": "all perturbations are compared with baseline_repeat from the same launch batch",
            "baseline_repeat": "compared with the accepted run; same accepted input, seed 42, PYTHONHASHSEED=0",
        },
        "files": {key: str(SUMMARY_DIR / key) for key in tables},
    }
    (SUMMARY_DIR / "comparison_manifest.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    compare()
