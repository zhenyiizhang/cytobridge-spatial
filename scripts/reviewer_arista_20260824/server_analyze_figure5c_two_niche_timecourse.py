#!/usr/bin/env python3
"""Test whether Figure 5c resolves into two injury-associated interaction niches.

The spatial segmentation is fixed before biological annotation: select the upper
quartile of the accepted 5-DPI cosine field, form connected components on the
trained model's physical-radius graph, require at least 20 ROI cells, and retain
only components containing selected model edges.  The selected-edge cell states
are then used to annotate the two retained components.

For each observed time point, the exact 5-DPI sender/receiver cell-state edge
roster is followed separately in injured and uninjured dorsal tissue.  Ligand-
receptor scores use the package-native formula

    mean_source(ligand/complex) * mean_target(receptor/complex)
    * selected_attention_per_source_cell.

The injured/uninjured trajectories are descriptive within-section controls;
cells are not biological replicates and the analysis is not a causal test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from server_analyze_figure5c_interaction_mechanism import (
    _complex_tokens,
    _load_lr_table,
    _permutation_module_attention,
    _permutation_module_lr_pathways,
    _symbol_expression_by_cell,
)


TIME_TO_DPI = {0.0: 2, 1.0: 5, 2.0: 10, 3.0: 15, 4.0: 20}
CONDITIONS = ("inj", "uninj")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aligned-h5ad", type=Path, required=True)
    parser.add_argument("--assignments", type=Path, required=True)
    parser.add_argument("--attention-dir", type=Path, required=True)
    parser.add_argument("--slice-dir", type=Path, required=True)
    parser.add_argument("--lr-database", type=Path, required=True)
    parser.add_argument("--pair-timecourse", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--upper-quantile", type=float, default=0.75)
    parser.add_argument("--minimum-size", type=int, default=20)
    parser.add_argument("--interaction-cutoff", type=float, default=0.03154105148551745)
    parser.add_argument("--n-attention-permutations", type=int, default=9999)
    parser.add_argument("--n-lr-permutations", type=int, default=1999)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _observed_archive(attention_dir: Path, time: float) -> tuple[Path, Path]:
    token = f"{time:.1f}"
    return (
        attention_dir / f"edge_index_interp_t{token}.npy",
        attention_dir / f"attn_mean_interp_t{token}.npy",
    )


def _slice_path(slice_dir: Path, time: float) -> Path:
    token = str(int(time)) if float(time).is_integer() else str(time).replace(".", "p")
    return slice_dir / f"time_{token}.h5ad"


def _segment_two_niches(
    assignments: pd.DataFrame,
    edge_index: np.ndarray,
    attention: np.ndarray,
    *,
    n_time_cells: int,
    upper_quantile: float,
    minimum_size: int,
    interaction_cutoff: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray], pd.DataFrame]:
    coordinates = assignments[["current_x", "current_y"]].to_numpy(dtype=float)
    cosine = assignments["cosine_full_vs_interaction"].to_numpy(dtype=float)
    threshold = float(np.quantile(cosine, upper_quantile))
    selected = cosine >= threshold
    radius_pairs = np.asarray(
        cKDTree(coordinates).query_pairs(interaction_cutoff, output_type="ndarray"),
        dtype=np.int64,
    )
    keep = selected[radius_pairs[:, 0]] & selected[radius_pairs[:, 1]]
    graph = coo_matrix(
        (
            np.ones(int(keep.sum()), dtype=np.uint8),
            (radius_pairs[keep, 0], radius_pairs[keep, 1]),
        ),
        shape=(len(assignments), len(assignments)),
    ).tocsr()
    graph = graph + graph.T
    _, labels = connected_components(graph, directed=False)
    sizes = (
        pd.Series(labels[np.flatnonzero(selected)])
        .value_counts(sort=False)
        .sort_values(ascending=False, kind="mergesort")
    )
    global_indices = assignments["cell_index"].to_numpy(dtype=int)
    celltypes = assignments["celltype"].astype(str).to_numpy()
    candidates: list[dict[str, Any]] = []
    edge_source, edge_target = edge_index
    for raw_label, size in sizes[sizes >= minimum_size].items():
        roi_mask = selected & (labels == int(raw_label))
        time_mask = np.zeros(n_time_cells, dtype=bool)
        time_mask[global_indices[roi_mask]] = True
        internal = time_mask[edge_source] & time_mask[edge_target]
        if not np.any(internal):
            continue
        time_labels = np.full(n_time_cells, "", dtype=object)
        time_labels[global_indices] = celltypes
        edge_table = pd.DataFrame(
            {
                "sender": time_labels[edge_source[internal]],
                "receiver": time_labels[edge_target[internal]],
                "attention": attention[internal],
            }
        )
        summarized = (
            edge_table.groupby(["sender", "receiver"], sort=True)["attention"]
            .agg(n_edges="size", attention_sum="sum")
            .reset_index()
        )
        sfrp_attention = float(
            summarized.loc[
                summarized["sender"].eq("sfrpEGC") | summarized["receiver"].eq("sfrpEGC"),
                "attention_sum",
            ].sum()
        )
        wnt_rea_attention = float(
            summarized.loc[
                summarized["sender"].isin(["wntEGC", "reaEGC"])
                & summarized["receiver"].isin(["wntEGC", "reaEGC"]),
                "attention_sum",
            ].sum()
        )
        candidates.append(
            {
                "raw_label": int(raw_label),
                "roi_mask": roi_mask,
                "time_mask": time_mask,
                "n_cells": int(size),
                "n_edges": int(internal.sum()),
                "attention_sum": float(attention[internal].sum()),
                "sfrp_attention": sfrp_attention,
                "wnt_rea_attention": wnt_rea_attention,
                "edges": summarized,
            }
        )
    if len(candidates) != 2:
        raise AssertionError(f"Expected exactly two retained upper-quartile components, got {len(candidates)}")
    first = max(candidates, key=lambda item: item["sfrp_attention"] - item["wnt_rea_attention"])
    second = max(candidates, key=lambda item: item["wnt_rea_attention"] - item["sfrp_attention"])
    if first is second:
        raise AssertionError("Could not uniquely annotate the two niches from selected-edge cell states.")
    named = {
        "N1_sfrpEGC_VLMC": first,
        "N2_reaEGC_wntEGC": second,
    }
    assignment_out = assignments.copy()
    assignment_out["two_niche_region"] = ""
    edge_rows: list[pd.DataFrame] = []
    masks: dict[str, np.ndarray] = {}
    for niche, item in named.items():
        assignment_out.loc[item["roi_mask"], "two_niche_region"] = niche
        table = item["edges"].copy()
        table.insert(0, "niche", niche)
        table["n_region_cells"] = int(item["n_cells"])
        table["attention_per_region_cell"] = table["attention_sum"] / max(int(item["n_cells"]), 1)
        edge_rows.append(table)
        masks[niche] = np.asarray(item["time_mask"], dtype=bool)
    metadata = pd.DataFrame(
        [
            {
                "niche": niche,
                "n_cells": int(item["n_cells"]),
                "n_internal_edges": int(item["n_edges"]),
                "attention_sum": float(item["attention_sum"]),
                "attention_per_cell": float(item["attention_sum"] / item["n_cells"]),
                "upper_quantile": float(upper_quantile),
                "cosine_threshold": threshold,
                "minimum_size": int(minimum_size),
                "interaction_cutoff": float(interaction_cutoff),
            }
            for niche, item in named.items()
        ]
    )
    return assignment_out, pd.concat(edge_rows, ignore_index=True), masks, metadata


def _expression_by_celltype(
    group_mask: np.ndarray,
    labels: np.ndarray,
    symbol_expression: dict[str, np.ndarray],
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for celltype in sorted(np.unique(labels[group_mask])):
        selected = group_mask & (labels == str(celltype))
        result[str(celltype)] = {
            symbol: float(np.mean(values[selected])) for symbol, values in symbol_expression.items()
        }
    return result


def _score_time_condition(
    *,
    niche: str,
    dpi: int,
    time: float,
    condition: str,
    group_mask: np.ndarray,
    labels: np.ndarray,
    edge_index: np.ndarray,
    attention: np.ndarray,
    roster: pd.DataFrame,
    lr_table: pd.DataFrame,
    symbol_expression: dict[str, np.ndarray],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    edge_source, edge_target = edge_index
    expression = _expression_by_celltype(group_mask, labels, symbol_expression)
    eligible_source_types = sorted(roster["sender"].astype(str).unique())
    n_eligible_source = int(np.count_nonzero(group_mask & np.isin(labels, eligible_source_types)))
    pair_rows: list[dict[str, Any]] = []
    edge_attention_total = 0.0
    edge_count_total = 0
    for edge in roster.itertuples(index=False):
        sender = str(edge.sender)
        receiver = str(edge.receiver)
        sender_mask = group_mask & (labels == sender)
        receiver_mask = group_mask & (labels == receiver)
        selected = sender_mask[edge_source] & receiver_mask[edge_target]
        attention_sum = float(attention[selected].sum())
        n_edges = int(selected.sum())
        n_source = int(sender_mask.sum())
        communication = attention_sum / max(n_source, 1)
        edge_attention_total += attention_sum
        edge_count_total += n_edges
        sender_expression = expression.get(sender, {})
        receiver_expression = expression.get(receiver, {})
        for pair in lr_table.itertuples(index=False):
            ligand_tokens = _complex_tokens(str(pair.ligand))
            receptor_tokens = _complex_tokens(str(pair.receptor))
            if any(token not in sender_expression for token in ligand_tokens):
                continue
            if any(token not in receiver_expression for token in receptor_tokens):
                continue
            ligand_mean = float(min(sender_expression[token] for token in ligand_tokens))
            receptor_mean = float(min(receiver_expression[token] for token in receptor_tokens))
            pair_rows.append(
                {
                    "niche": niche,
                    "time": float(time),
                    "dpi": int(dpi),
                    "condition": condition,
                    "sender": sender,
                    "receiver": receiver,
                    "ligand": str(pair.ligand),
                    "receptor": str(pair.receptor),
                    "pair": str(pair.pair),
                    "pathway": str(pair.pathway),
                    "interaction_class": str(pair.interaction_class),
                    "n_source_cells": n_source,
                    "n_target_cells": int(receiver_mask.sum()),
                    "n_selected_edges": n_edges,
                    "attention_per_source_cell": communication,
                    "ligand_mean_count": ligand_mean,
                    "receptor_mean_count": receptor_mean,
                    "lr_score": ligand_mean * receptor_mean * communication,
                }
            )
    pair_table = pd.DataFrame(pair_rows)
    pathway = (
        pair_table.groupby(
            ["niche", "time", "dpi", "condition", "pathway", "interaction_class"],
            sort=True,
        )["lr_score"]
        .sum()
        .reset_index()
    )
    attention_summary = {
        "niche": niche,
        "time": float(time),
        "dpi": int(dpi),
        "condition": condition,
        "n_group_cells": int(group_mask.sum()),
        "n_eligible_source_cells": n_eligible_source,
        "n_roster_celltype_pairs": int(len(roster)),
        "n_selected_edges": edge_count_total,
        "selected_attention_sum": edge_attention_total,
        "attention_per_eligible_source_cell": edge_attention_total / max(n_eligible_source, 1),
    }
    return attention_summary, pair_table, pathway


def main() -> None:
    args = _parser().parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite immutable output: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.stage-", dir=output_dir.parent))
    try:
        inputs = {
            "aligned_h5ad": args.aligned_h5ad.expanduser().resolve(),
            "assignments": args.assignments.expanduser().resolve(),
            "lr_database": args.lr_database.expanduser().resolve(),
            "pair_timecourse": args.pair_timecourse.expanduser().resolve(),
        }
        for time in TIME_TO_DPI:
            edge_path, attention_path = _observed_archive(args.attention_dir.expanduser().resolve(), time)
            inputs[f"edge_t{time:.1f}"] = edge_path
            inputs[f"attention_t{time:.1f}"] = attention_path
            inputs[f"slice_t{time:.1f}"] = _slice_path(args.slice_dir.expanduser().resolve(), time)
        for path in inputs.values():
            if not path.is_file():
                raise FileNotFoundError(path)

        assignments = pd.read_csv(inputs["assignments"])
        aligned = ad.read_h5ad(inputs["aligned_h5ad"], backed="r")
        t1_global = np.flatnonzero(
            np.isclose(aligned.obs["time_point_processed"].to_numpy(dtype=float), 1.0)
        )
        t1_labels = aligned.obs.iloc[t1_global]["Annotation"].astype(str).to_numpy()
        t1_edge = np.load(inputs["edge_t1.0"]).astype(np.int64)
        t1_attention = np.load(inputs["attention_t1.0"]).astype(np.float64).reshape(-1)
        if not np.array_equal(
            t1_labels[assignments["cell_index"].to_numpy(dtype=int)],
            assignments["celltype"].astype(str).to_numpy(),
        ):
            raise AssertionError("ROI assignments do not align to latest t=1 cell order.")
        segmented, network_edges, niche_masks, niche_metadata = _segment_two_niches(
            assignments,
            t1_edge,
            t1_attention,
            n_time_cells=len(t1_labels),
            upper_quantile=float(args.upper_quantile),
            minimum_size=int(args.minimum_size),
            interaction_cutoff=float(args.interaction_cutoff),
        )

        roi_mask = np.zeros(len(t1_labels), dtype=bool)
        roi_mask[assignments["cell_index"].to_numpy(dtype=int)] = True
        attention_permutation_rows: list[dict[str, Any]] = []
        lr_permutation_rows: list[pd.DataFrame] = []

        lr_table = _load_lr_table(inputs["lr_database"], inputs["pair_timecourse"])
        requested_symbols = {
            token
            for row in lr_table.itertuples(index=False)
            for token in _complex_tokens(str(row.ligand)) + _complex_tokens(str(row.receptor))
        }
        t1_expression, feature_coverage = _symbol_expression_by_cell(
            aligned, t1_global, requested_symbols
        )
        for niche_index, niche in enumerate(("N1_sfrpEGC_VLMC", "N2_reaEGC_wntEGC")):
            attention_summary, _ = _permutation_module_attention(
                module=niche,
                actual_mask=niche_masks[niche],
                roi_mask=roi_mask,
                labels=t1_labels,
                edge_source=t1_edge[0],
                edge_target=t1_edge[1],
                attention=t1_attention,
                n_permutations=int(args.n_attention_permutations),
                seed=int(args.seed) + niche_index * 1009,
            )
            attention_permutation_rows.append(attention_summary)
            lr_summary, _ = _permutation_module_lr_pathways(
                module=niche,
                actual_mask=niche_masks[niche],
                roi_mask=roi_mask,
                labels=t1_labels,
                edge_source=t1_edge[0],
                edge_target=t1_edge[1],
                attention=t1_attention,
                symbol_expression=t1_expression,
                lr_table=lr_table,
                n_permutations=int(args.n_lr_permutations),
                seed=int(args.seed) + 7919 + niche_index * 1009,
            )
            lr_permutation_rows.append(lr_summary)

        # Follow the exact selected t=1 cell-type edge roster across observed times.
        time_attention_rows: list[dict[str, Any]] = []
        time_pair_rows: list[pd.DataFrame] = []
        time_pathway_rows: list[pd.DataFrame] = []
        for time, dpi in TIME_TO_DPI.items():
            global_indices = np.flatnonzero(
                np.isclose(aligned.obs["time_point_processed"].to_numpy(dtype=float), time)
            )
            labels = aligned.obs.iloc[global_indices]["Annotation"].astype(str).to_numpy()
            slice_adata = ad.read_h5ad(inputs[f"slice_t{time:.1f}"], backed="r")
            if not np.array_equal(labels, slice_adata.obs["Annotation"].astype(str).to_numpy()):
                raise AssertionError(f"Aligned h5ad and persisted slice order differ at t={time}")
            slice_adata.file.close()
            edge_index = np.load(inputs[f"edge_t{time:.1f}"]).astype(np.int64)
            attention = np.load(inputs[f"attention_t{time:.1f}"]).astype(np.float64).reshape(-1)
            if edge_index.shape != (2, len(attention)) or int(edge_index.max()) >= len(labels):
                raise AssertionError(f"Attention archive mismatch at t={time}")
            symbol_expression, _ = _symbol_expression_by_cell(
                aligned, global_indices, requested_symbols
            )
            obs = aligned.obs.iloc[global_indices]
            for condition in CONDITIONS:
                group_mask = (
                    obs["inj_uninj"].astype(str).eq(condition).to_numpy()
                    & obs["D_V"].astype(str).eq("D").to_numpy()
                )
                for niche in ("N1_sfrpEGC_VLMC", "N2_reaEGC_wntEGC"):
                    roster = network_edges.loc[network_edges["niche"].eq(niche), ["sender", "receiver"]].drop_duplicates()
                    attention_row, pair_table, pathway_table = _score_time_condition(
                        niche=niche,
                        dpi=dpi,
                        time=time,
                        condition=condition,
                        group_mask=group_mask,
                        labels=labels,
                        edge_index=edge_index,
                        attention=attention,
                        roster=roster,
                        lr_table=lr_table,
                        symbol_expression=symbol_expression,
                    )
                    time_attention_rows.append(attention_row)
                    time_pair_rows.append(pair_table)
                    time_pathway_rows.append(pathway_table)
        aligned.file.close()

        time_attention = pd.DataFrame(time_attention_rows)
        time_pairs = pd.concat(time_pair_rows, ignore_index=True)
        time_pathways = pd.concat(time_pathway_rows, ignore_index=True)
        attention_pivot = time_attention.pivot(
            index=["niche", "time", "dpi"],
            columns="condition",
            values="attention_per_eligible_source_cell",
        ).reset_index()
        attention_pivot["log2_inj_over_uninj"] = np.log2(
            (attention_pivot["inj"] + 1e-12) / (attention_pivot["uninj"] + 1e-12)
        )
        pathway_pivot = time_pathways.pivot(
            index=["niche", "time", "dpi", "pathway", "interaction_class"],
            columns="condition",
            values="lr_score",
        ).reset_index()
        pathway_pivot[["inj", "uninj"]] = pathway_pivot[["inj", "uninj"]].fillna(0.0)
        pathway_pivot["log2_inj_over_uninj"] = np.log2(
            (pathway_pivot["inj"] + 1e-12) / (pathway_pivot["uninj"] + 1e-12)
        )

        tables = temporary / "tables"
        tables.mkdir()
        outputs = {
            "roi_two_niche_assignments.csv": segmented,
            "two_niche_metadata.csv": niche_metadata,
            "two_niche_t1_celltype_edges.csv": network_edges,
            "two_niche_attention_matched_null.csv": pd.DataFrame(attention_permutation_rows),
            "two_niche_lr_pathway_matched_null.csv": pd.concat(lr_permutation_rows, ignore_index=True),
            "two_niche_attention_timecourse.csv": time_attention,
            "two_niche_attention_injury_contrast.csv": attention_pivot,
            "two_niche_lr_pair_timecourse.csv.gz": time_pairs,
            "two_niche_lr_pathway_timecourse.csv": time_pathways,
            "two_niche_lr_pathway_injury_contrast.csv": pathway_pivot,
            "lr_feature_coverage.csv": feature_coverage,
        }
        for name, frame in outputs.items():
            frame.to_csv(tables / name, index=False)

        manifest = {
            "schema": "cytobridge.arista.figure5c-two-niche-timecourse.v1",
            "claim_tested": "The heterogeneous 5-DPI Figure 5c field contains two discrete model-edge-supported ependymoglial niches whose programs can be tracked in injured versus uninjured dorsal tissue across regeneration.",
            "segmentation_contract": {
                "roi": "frozen 1,454-cell Figure 5c ROI",
                "selection": f"cosine >= within-ROI quantile {float(args.upper_quantile):.2f}",
                "connectivity": f"physical-radius graph at the trained interaction cutoff {float(args.interaction_cutoff):.17g}",
                "minimum_component_size": int(args.minimum_size),
                "model_edge_filter": "retain only components with internal selected attention edges",
                "biological_annotation": "performed after segmentation from the selected-edge cell-state network",
            },
            "timecourse_contract": {
                "times": TIME_TO_DPI,
                "conditions": list(CONDITIONS),
                "anatomy": "D_V == D",
                "network_roster": "exact sender/receiver cell-state pairs selected inside each 5-DPI niche",
                "lr_formula": "mean_source(ligand/complex) * mean_target(receptor/complex) * selected_attention_per_source_cell",
                "complex": "strict minimum, all subunits required",
                "expression": "per-cell expm1 from persisted log1p aligned h5ad before arithmetic means",
            },
            "limitations": [
                "One section per time point; cells are not biological replicates.",
                "Injured/uninjured dorsal contrasts are descriptive within-section controls, not population-level tests.",
                "The 5-DPI niche definitions and selected edges derive from the same fitted model.",
                "Attention is relative model influence, not a biophysical signaling rate.",
                "No pixel-level wound contour is available.",
            ],
            "inputs": {key: {"path": str(path), "sha256": _sha256(path)} for key, path in inputs.items()},
            "outputs": {},
        }
        for path in sorted(tables.iterdir()):
            manifest["outputs"][path.name] = {
                "sha256": _sha256(path),
                "size_bytes": int(path.stat().st_size),
            }
        (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        temporary.rename(output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
