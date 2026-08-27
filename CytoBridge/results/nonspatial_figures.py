"""Compact reader inputs for grouped non-spatial Supplementary Figures S4--S5."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ._io import prepare_output_dir, read_json, require_files, resolve_results_dir


FIGURE_IDS = ("s4", "s5")
WEINREB_TIMES = (2.0, 4.0, 6.0)
SCNT_TIMES = (0.0, 0.25, 0.5, 1.0, 2.0)
WEINREB_CELL_TYPES = (
    "Undifferentiated",
    "Monocyte",
    "Neutrophil",
    "Baso",
    "Mast",
    "Meg",
    "Erythroid",
    "Lymphoid",
    "Eos",
    "Ccr7_DC",
    "pDC",
)
SCNT_CELL_TYPES = (
    "Ex",
    "EX-NP1",
    "EX-NP2",
    "RG",
    "Inh-NP",
    "Inh1",
    "Inh2",
    "Inh3",
    "Inh4",
)
CONDITIONS = ("with_interaction", "without_interaction")

_FILES = (
    "external_inputs.json",
    "manifest.json",
    "scnt_cell_colors.json",
    "scnt_direction.csv",
    "scnt_distribution.csv",
    "scnt_interaction_vectors.csv",
    "scnt_metrics.json",
    "scnt_model_fields.npz",
    "scnt_network_edges.csv",
    "scnt_network_nodes.csv",
    "scnt_network_pairs.csv",
    "scnt_observed_cells.npz",
    "scnt_pathways.csv",
    "weinreb_cell_colors.json",
    "weinreb_clone_fate.csv",
    "weinreb_concordance.csv",
    "weinreb_distribution.csv",
    "weinreb_metrics.json",
    "weinreb_model_fields.npz",
    "weinreb_network_edges.csv",
    "weinreb_network_nodes.csv",
    "weinreb_observed_cells.npz",
    "weinreb_pathways.csv",
)


@dataclass(frozen=True)
class PackedCells:
    """Flat, non-object arrays for one observed-cell collection."""

    times: np.ndarray
    label_id: np.ndarray
    label_names: np.ndarray
    pc_xy: np.ndarray
    spring_xy: np.ndarray | None = None

    @property
    def labels(self) -> np.ndarray:
        """Return the semantic label for every packed cell."""

        return self.label_names[self.label_id]

    def frame(self, time: float, *, space: str = "pc") -> tuple[np.ndarray, np.ndarray]:
        """Return coordinates and labels at one observed time."""

        keep = np.isclose(self.times, float(time), rtol=0.0, atol=1e-6)
        if not np.any(keep):
            raise KeyError(f"Observed time not found: {time:g}")
        if space == "pc":
            coordinates = self.pc_xy
        elif space == "spring" and self.spring_xy is not None:
            coordinates = self.spring_xy
        else:
            raise KeyError(f"Coordinate space not available: {space!r}")
        return coordinates[keep], self.labels[keep]


@dataclass(frozen=True)
class NonspatialDatasetResults:
    """Processed inputs for one grouped non-spatial figure."""

    name: str
    cells: PackedCells
    colors: Mapping[str, str]
    fields: Mapping[str, np.ndarray]
    distribution: pd.DataFrame
    network_edges: pd.DataFrame
    network_nodes: pd.DataFrame
    pathways: pd.DataFrame
    metrics: Mapping[str, Any]
    clone_fate: pd.DataFrame | None = None
    concordance: pd.DataFrame | None = None
    direction: pd.DataFrame | None = None
    network_pairs: pd.DataFrame | None = None
    interaction_vectors: pd.DataFrame | None = None


@dataclass(frozen=True)
class NonspatialFigureResults:
    """Packaged reader inputs for Supplementary Figures S4 and S5."""

    source_dir: Path
    manifest: Mapping[str, Any]
    external_inputs: Mapping[str, Any]
    weinreb: NonspatialDatasetResults
    scnt: NonspatialDatasetResults


@dataclass(frozen=True)
class NonspatialPanels:
    """Recalculated tables used by the grouped figure renderer."""

    weinreb_distribution: pd.DataFrame
    weinreb_clone_fate: pd.DataFrame
    weinreb_concordance: pd.DataFrame
    weinreb_pathways: pd.DataFrame
    scnt_distribution: pd.DataFrame
    scnt_direction: pd.DataFrame
    scnt_pathways: pd.DataFrame
    summary: pd.DataFrame


def _readonly(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value)
    result.setflags(write=False)
    return result


def _load_npz(path: Path, expected: set[str]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        if set(payload.files) != expected:
            raise ValueError(f"{path} has an unexpected array roster")
        arrays = {name: np.asarray(payload[name]) for name in expected}
    if any(value.dtype.hasobject for value in arrays.values()):
        raise ValueError(f"{path} contains object arrays")
    return {name: _readonly(value) for name, value in arrays.items()}


def _load_cells(
    path: Path,
    *,
    label_names: tuple[str, ...],
    times: tuple[float, ...],
    expected_counts: tuple[int, ...],
    spring: bool,
) -> PackedCells:
    expected = {"times", "label_id", "label_names", "pc_xy"}
    if spring:
        expected.add("spring_xy")
    arrays = _load_npz(path, expected)
    packed = PackedCells(
        times=arrays["times"],
        label_id=arrays["label_id"],
        label_names=arrays["label_names"],
        pc_xy=arrays["pc_xy"],
        spring_xy=arrays.get("spring_xy"),
    )
    n_cells = sum(expected_counts)
    if (
        packed.times.shape != (n_cells,)
        or packed.label_id.shape != (n_cells,)
        or packed.pc_xy.shape != (n_cells, 2)
        or (spring and packed.spring_xy is not None and packed.spring_xy.shape != (n_cells, 2))
        or tuple(packed.label_names.astype(str)) != label_names
        or packed.label_id.size == 0
        or int(packed.label_id.min()) < 0
        or int(packed.label_id.max()) >= len(label_names)
        or not np.isfinite(packed.pc_xy).all()
        or (packed.spring_xy is not None and not np.isfinite(packed.spring_xy).all())
    ):
        raise ValueError(f"{path} has invalid observed-cell arrays")
    observed_counts = tuple(
        int(np.isclose(packed.times, time, rtol=0.0, atol=1e-6).sum())
        for time in times
    )
    if observed_counts != expected_counts or set(np.unique(packed.times.astype(float))) != set(times):
        raise ValueError(f"{path} has an unexpected time grid")
    if set(packed.labels.astype(str)) != set(label_names):
        raise ValueError(f"{path} has an unexpected label roster")
    return packed


def _color_map(path: Path, labels: Sequence[str]) -> dict[str, str]:
    value = read_json(path)
    if set(value) != set(labels) or any(
        re.fullmatch(r"#[0-9a-fA-F]{6}", str(color)) is None
        for color in value.values()
    ):
        raise ValueError(f"{path} has an invalid semantic color map")
    return {label: str(value[label]) for label in labels}


def _table(
    path: Path,
    columns: Sequence[str],
    *,
    numeric: Sequence[str] = (),
) -> pd.DataFrame:
    result = pd.read_csv(path)
    missing = sorted(set(columns).difference(result.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    result = result.loc[:, columns].copy()
    if numeric:
        converted = result.loc[:, numeric].apply(pd.to_numeric, errors="coerce")
        if not np.isfinite(converted.to_numpy(float)).all():
            raise ValueError(f"{path} contains non-finite numeric values")
        result.loc[:, numeric] = converted
    text_columns = [name for name in columns if name not in numeric]
    if text_columns and result.loc[:, text_columns].isna().any().any():
        raise ValueError(f"{path} contains missing text values")
    return result


def _weinreb_fields(path: Path) -> dict[str, np.ndarray]:
    expected = {"spring_xlim", "spring_ylim"}
    for day in (2, 4, 6):
        expected.update(
            {
                f"day{day}_x_axis",
                f"day{day}_y_axis",
                f"day{day}_support_mask",
                f"day{day}_lr_full_drift_u",
                f"day{day}_lr_full_drift_v",
                f"day{day}_lr_full_drift_speed",
            }
        )
    expected.update(
        {
            "day6_lr_interaction_u",
            "day6_lr_interaction_v",
            "day6_lr_interaction_speed",
        }
    )
    fields = _load_npz(path, expected)
    for name, value in fields.items():
        if name.endswith(("x_axis", "y_axis")) and value.shape != (50,):
            raise ValueError(f"{path} has invalid field axes")
        if name.endswith(("mask", "_u", "_v", "speed")) and value.shape != (50, 50):
            raise ValueError(f"{path} has invalid field grids")
        if name.endswith(("x_axis", "y_axis", "xlim", "ylim")) and not np.isfinite(value).all():
            raise ValueError(f"{path} contains non-finite field axes")
    for day in (2, 4, 6):
        mask = fields[f"day{day}_support_mask"].astype(bool)
        if not np.any(mask):
            raise ValueError(f"{path} has an empty support mask")
        names = (
            f"day{day}_lr_full_drift_u",
            f"day{day}_lr_full_drift_v",
            f"day{day}_lr_full_drift_speed",
        )
        if day == 6:
            names = (*names, "day6_lr_interaction_u", "day6_lr_interaction_v", "day6_lr_interaction_speed")
        if any(not np.isfinite(fields[name][mask]).all() for name in names):
            raise ValueError(f"{path} contains non-finite supported field values")
    return fields


def _scnt_fields(path: Path) -> dict[str, np.ndarray]:
    expected = {"pc_xlim", "pc_ylim", "full_times"}
    for index in range(3):
        expected.update({f"full{index}_{name}" for name in ("x", "y", "u", "v", "mask")})
    expected.update({f"interaction_{name}" for name in ("x", "y", "u", "v", "mask")})
    fields = _load_npz(path, expected)
    if not np.array_equal(fields["full_times"], [0.0, 0.5, 1.0]):
        raise ValueError(f"{path} has unexpected full-drift times")
    for name, value in fields.items():
        if name.endswith(("_x", "_y")) and value.shape != (42,):
            raise ValueError(f"{path} has invalid field axes")
        if name.endswith(("_u", "_v", "_mask")) and value.shape != (42, 42):
            raise ValueError(f"{path} has invalid field grids")
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError(f"{path} contains non-finite field values")
    return fields


def _validate_manifest(manifest: Mapping[str, Any], external: Mapping[str, Any], path: Path) -> None:
    if manifest.get("schema_version") != 1 or manifest.get("analysis") != "grouped_nonspatial_s4_s5":
        raise ValueError(f"{path} does not describe grouped non-spatial S4--S5")
    if tuple(manifest.get("figures", {})) != FIGURE_IDS:
        raise ValueError(f"{path} has an unexpected figure roster")
    if set(manifest.get("files", {})) != set(_FILES).difference({"manifest.json"}):
        raise ValueError(f"{path} has an unexpected file roster")
    if external.get("schema_version") != 1 or external.get("path_base") != "project root":
        raise ValueError("External input registry has an unsupported schema")
    for dataset in ("weinreb", "scnt"):
        record = external.get("datasets", {}).get(dataset, {})
        paths = [record.get("renderer"), *record.get("full_rerun_inputs", [])]
        if not paths or any(
            not isinstance(value, str)
            or Path(value).is_absolute()
            or ".." in Path(value).parts
            for value in paths
        ):
            raise ValueError("External input registry paths must be project-relative")


def load_nonspatial_figures(
    results_dir: str | Path | None = None,
) -> NonspatialFigureResults:
    """Load and validate the compact grouped S4--S5 result bundle."""

    source_dir = resolve_results_dir(results_dir, slug="nonspatial_figures")
    paths = require_files(source_dir, _FILES)
    manifest = read_json(paths["manifest.json"])
    external = read_json(paths["external_inputs.json"])
    _validate_manifest(manifest, external, paths["manifest.json"])

    w_cells = _load_cells(
        paths["weinreb_observed_cells.npz"],
        label_names=WEINREB_CELL_TYPES,
        times=WEINREB_TIMES,
        expected_counts=(4_638, 14_985, 29_679),
        spring=True,
    )
    w_distribution_columns = (
        "time",
        "space",
        "w1_full",
        "w2_full",
        "tmv_absolute_full",
        "w1_no_interaction",
        "w2_no_interaction",
        "tmv_absolute_no_interaction",
        "w1_relative_change",
        "w2_relative_change",
        "tmv_relative_change",
    )
    w_distribution = _table(
        paths["weinreb_distribution.csv"],
        w_distribution_columns,
        numeric=tuple(name for name in w_distribution_columns if name != "space"),
    )
    w_clone = _table(
        paths["weinreb_clone_fate.csv"],
        ("metric", "full", "no_interaction", "delta_no_interaction_minus_full", "relative_change", "higher_is_better"),
        numeric=("full", "no_interaction", "delta_no_interaction_minus_full", "relative_change"),
    )
    w_concordance = _table(
        paths["weinreb_concordance.csv"],
        ("day", "kendall_tau_b", "n_contexts", "spearman_rho"),
        numeric=("day", "kendall_tau_b", "n_contexts", "spearman_rho"),
    )
    w_edges = _table(
        paths["weinreb_network_edges.csv"],
        ("method", "sender_type", "receiver_type", "display_score", "display_rank", "display_top_n"),
        numeric=("display_score", "display_rank", "display_top_n"),
    )
    w_nodes = _table(paths["weinreb_network_nodes.csv"], ("cell_type", "n_cells"), numeric=("n_cells",))
    w_pathways = _table(
        paths["weinreb_pathways.csv"],
        ("pathway", "summed_cytobridge_score", "n_positive_cell_type_pairs", "share_of_day6_score_pct", "display_rank"),
        numeric=("summed_cytobridge_score", "n_positive_cell_type_pairs", "share_of_day6_score_pct", "display_rank"),
    )
    weinreb = NonspatialDatasetResults(
        name="weinreb",
        cells=w_cells,
        colors=_color_map(paths["weinreb_cell_colors.json"], WEINREB_CELL_TYPES),
        fields=_weinreb_fields(paths["weinreb_model_fields.npz"]),
        distribution=w_distribution,
        clone_fate=w_clone,
        concordance=w_concordance,
        network_edges=w_edges,
        network_nodes=w_nodes,
        pathways=w_pathways,
        metrics=read_json(paths["weinreb_metrics.json"]),
    )

    s_cells = _load_cells(
        paths["scnt_observed_cells.npz"],
        label_names=SCNT_CELL_TYPES,
        times=SCNT_TIMES,
        expected_counts=(2_232, 4_031, 4_470, 6_814, 3_000),
        spring=False,
    )
    s_distribution = _table(
        paths["scnt_distribution.csv"],
        ("time", "w1_full", "w2_full", "tmv_absolute_full", "w1_no_interaction", "w2_no_interaction", "tmv_absolute_no_interaction"),
        numeric=("time", "w1_full", "w2_full", "tmv_absolute_full", "w1_no_interaction", "w2_no_interaction", "tmv_absolute_no_interaction"),
    )
    s_direction_columns = (
        "training_seed",
        "condition",
        "time_hours",
        "n_cells",
        "centroid_cosine_inference_drift_vs_scnt",
        "cell_cosine_mean",
        "cell_cosine_median",
        "cell_cosine_sd",
    )
    s_direction = _table(
        paths["scnt_direction.csv"],
        s_direction_columns,
        numeric=tuple(name for name in s_direction_columns if name != "condition"),
    )
    s_edges = _table(
        paths["scnt_network_edges.csv"],
        ("method", "sender_type", "receiver_type", "display_score", "display_rank"),
        numeric=("display_score", "display_rank"),
    )
    s_pairs = _table(
        paths["scnt_network_pairs.csv"],
        ("sender_type", "receiver_type", "D_AB_mean", "cellchat_native_significant"),
        numeric=("D_AB_mean", "cellchat_native_significant"),
    )
    s_nodes = _table(paths["scnt_network_nodes.csv"], ("cell_type", "n_cells"), numeric=("n_cells",))
    s_vectors = _table(
        paths["scnt_interaction_vectors.csv"],
        ("receiver_type", "drift_pc_1_mean", "drift_pc_2_mean"),
        numeric=("drift_pc_1_mean", "drift_pc_2_mean"),
    )
    s_pathways = _table(
        paths["scnt_pathways.csv"],
        ("pathway", "summed_S", "n_positive_pairs", "share_pct", "display_rank"),
        numeric=("summed_S", "n_positive_pairs", "share_pct", "display_rank"),
    )
    scnt = NonspatialDatasetResults(
        name="scnt",
        cells=s_cells,
        colors=_color_map(paths["scnt_cell_colors.json"], SCNT_CELL_TYPES),
        fields=_scnt_fields(paths["scnt_model_fields.npz"]),
        distribution=s_distribution,
        direction=s_direction,
        network_edges=s_edges,
        network_nodes=s_nodes,
        network_pairs=s_pairs,
        interaction_vectors=s_vectors,
        pathways=s_pathways,
        metrics=read_json(paths["scnt_metrics.json"]),
    )
    results = NonspatialFigureResults(
        source_dir=source_dir,
        manifest=manifest,
        external_inputs=external,
        weinreb=weinreb,
        scnt=scnt,
    )
    calculate_nonspatial_panels(results)
    return results


def _validate_network(dataset: NonspatialDatasetResults, *, top_n: int) -> None:
    expected_methods = {"CytoBridge D", "CellChat"}
    counts = dataset.network_edges.groupby("method").size().to_dict()
    if set(counts) != expected_methods or set(counts.values()) != {top_n}:
        raise ValueError(f"{dataset.name} has an unexpected network display roster")
    for method, block in dataset.network_edges.groupby("method"):
        if block["display_rank"].astype(int).tolist() != list(range(1, top_n + 1)):
            raise ValueError(f"{dataset.name} has invalid display ranks for {method}")
    if tuple(dataset.network_nodes["cell_type"].astype(str)) != tuple(dataset.cells.label_names.astype(str)):
        raise ValueError(f"{dataset.name} node and cell-type rosters differ")
    labels = set(dataset.cells.label_names.astype(str))
    edge_labels = set(dataset.network_edges["sender_type"].astype(str)) | set(dataset.network_edges["receiver_type"].astype(str))
    if not edge_labels.issubset(labels):
        raise ValueError(f"{dataset.name} network contains unknown cell types")


def calculate_nonspatial_panels(results: NonspatialFigureResults) -> NonspatialPanels:
    """Recalculate the displayed S4--S5 summaries from packaged values."""

    w = results.weinreb
    s = results.scnt
    if w.clone_fate is None or w.concordance is None or s.direction is None or s.network_pairs is None:
        raise ValueError("Grouped non-spatial inputs are incomplete")
    if w.distribution["time"].astype(float).tolist() != [1.0, 2.0] or not w.distribution["space"].eq("pca").all():
        raise ValueError("Weinreb distribution endpoints must be Day 4 and Day 6")
    if set(w.clone_fate["metric"].astype(str)) != {"tv_agreement", "js_similarity", "dominant_fate_match"}:
        raise ValueError("Weinreb clone-fate metric roster is incomplete")
    if w.concordance["day"].astype(float).tolist() != list(WEINREB_TIMES):
        raise ValueError("Weinreb concordance days are incomplete")
    _validate_network(w, top_n=30)
    if len(w.pathways) != 8 or w.pathways["display_rank"].astype(int).tolist() != list(range(1, 9)):
        raise ValueError("Weinreb pathway display roster is incomplete")

    w_relative = np.r_[w.distribution["w1_relative_change"], w.distribution["w2_relative_change"]]
    w_error_increase = float(100.0 * np.mean(w_relative))
    w_day6 = w.concordance.loc[np.isclose(w.concordance["day"], 6.0)].iloc[0]
    w_reported = w.metrics["cellchat"]
    if not (
        np.isclose(w_error_increase, w.metrics["distribution"]["equal_weight_W1_W2_error_increase_after_removal_pct"], rtol=0.0, atol=1e-12)
        and np.isclose(w_day6["spearman_rho"], w_reported["displayed_spearman_rho"], rtol=0.0, atol=1e-12)
        and np.isclose(w.concordance["spearman_rho"].mean(), w_reported["mean_spearman_rho_equal_time"], rtol=0.0, atol=1e-12)
    ):
        raise ValueError("Weinreb recalculated summaries disagree with packaged metrics")

    if s.distribution["time"].astype(float).tolist() != [0.25, 0.5, 1.0, 2.0]:
        raise ValueError("scNT distribution endpoints are incomplete")
    primary = s.direction[s.direction["condition"].isin(("full_interaction_noise", "no_interaction_noise"))]
    if len(primary) != 8 or set(primary["time_hours"].astype(float)) != {0.25, 0.5, 1.0, 2.0}:
        raise ValueError("scNT direction endpoints are incomplete")
    direction_summary = (
        primary.groupby("condition", as_index=False)
        .agg(
            cell_cosine_mean=("cell_cosine_mean", "mean"),
            cell_cosine_median=("cell_cosine_median", "mean"),
            n_endpoints=("time_hours", "size"),
        )
        .sort_values("condition")
        .reset_index(drop=True)
    )
    pivot = primary.pivot(index="time_hours", columns="condition", values=["cell_cosine_mean", "cell_cosine_median"])
    wins_mean = int((pivot[("cell_cosine_mean", "full_interaction_noise")] > pivot[("cell_cosine_mean", "no_interaction_noise")]).sum())
    wins_median = int((pivot[("cell_cosine_median", "full_interaction_noise")] > pivot[("cell_cosine_median", "no_interaction_noise")]).sum())
    _validate_network(s, top_n=24)
    if len(s.network_pairs) != 81 or s.network_pairs.duplicated(["sender_type", "receiver_type"]).any():
        raise ValueError("scNT all-pair network table is incomplete")
    ranks_a = s.network_pairs["D_AB_mean"].rank(method="average").to_numpy(float)
    ranks_b = s.network_pairs["cellchat_native_significant"].rank(method="average").to_numpy(float)
    scnt_rho = float(np.corrcoef(ranks_a, ranks_b)[0, 1])
    if len(s.pathways) != 8 or s.pathways["display_rank"].astype(int).tolist() != list(range(1, 9)):
        raise ValueError("scNT pathway display roster is incomplete")
    reported_direction = s.metrics["direction_evaluation"]
    reported_average = reported_direction["equal_endpoint_average"]
    by_condition = direction_summary.set_index("condition")
    if not (
        np.isclose(scnt_rho, s.metrics["network"]["spearman_rho_all_81_pairs"], rtol=0.0, atol=1e-12)
        and np.isclose(by_condition.loc["full_interaction_noise", "cell_cosine_mean"], reported_average["mean_cellwise_cosine_full"], rtol=0.0, atol=1e-14)
        and np.isclose(by_condition.loc["no_interaction_noise", "cell_cosine_mean"], reported_average["mean_cellwise_cosine_no_interaction"], rtol=0.0, atol=1e-14)
        and wins_mean == reported_direction["endpoint_wins_full"]["mean"]
        and wins_median == reported_direction["endpoint_wins_full"]["median"]
    ):
        raise ValueError("scNT recalculated summaries disagree with packaged metrics")

    summary = pd.DataFrame(
        [
            ("weinreb", "cells", float(len(w.cells.times))),
            ("weinreb", "distribution_error_increase_pct", w_error_increase),
            ("weinreb", "day6_spearman_rho", float(w_day6["spearman_rho"])),
            ("scnt", "cells", float(len(s.cells.times))),
            ("scnt", "network_spearman_rho", scnt_rho),
            ("scnt", "direction_mean_wins", float(wins_mean)),
            ("scnt", "direction_median_wins", float(wins_median)),
        ],
        columns=("dataset", "metric", "value"),
    )
    return NonspatialPanels(
        weinreb_distribution=w.distribution.copy(),
        weinreb_clone_fate=w.clone_fate.copy(),
        weinreb_concordance=w.concordance.copy(),
        weinreb_pathways=w.pathways.copy(),
        scnt_distribution=s.distribution.copy(),
        scnt_direction=direction_summary,
        scnt_pathways=s.pathways.copy(),
        summary=summary,
    )


def write_nonspatial_tables(
    panels: NonspatialPanels,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write the compact recalculated tables shown by S4--S5."""

    output = prepare_output_dir(output_dir)
    tables = {
        "weinreb_distribution": panels.weinreb_distribution,
        "weinreb_clone_fate": panels.weinreb_clone_fate,
        "weinreb_concordance": panels.weinreb_concordance,
        "weinreb_pathways": panels.weinreb_pathways,
        "scnt_distribution": panels.scnt_distribution,
        "scnt_direction": panels.scnt_direction,
        "scnt_pathways": panels.scnt_pathways,
        "summary": panels.summary,
    }
    paths: dict[str, Path] = {}
    for name, table in tables.items():
        path = output / f"{name}.csv"
        table.to_csv(path, index=False)
        paths[name] = path
    return paths


def nonspatial_statistics(
    results: NonspatialFigureResults,
    panels: NonspatialPanels | None = None,
) -> dict[str, Any]:
    """Return small JSON-compatible reproduction statistics."""

    calculated = panels or calculate_nonspatial_panels(results)
    values = calculated.summary.set_index(["dataset", "metric"])["value"]
    return {
        "figure_ids": list(FIGURE_IDS),
        "bundle_bytes": int(sum(path.stat().st_size for path in results.source_dir.iterdir() if path.is_file())),
        "weinreb_cells": int(values.loc[("weinreb", "cells")]),
        "weinreb_distribution_error_increase_pct": float(values.loc[("weinreb", "distribution_error_increase_pct")]),
        "weinreb_day6_spearman_rho": float(values.loc[("weinreb", "day6_spearman_rho")]),
        "scnt_cells": int(values.loc[("scnt", "cells")]),
        "scnt_network_spearman_rho": float(values.loc[("scnt", "network_spearman_rho")]),
        "scnt_direction_mean_wins": int(values.loc[("scnt", "direction_mean_wins")]),
        "scnt_direction_median_wins": int(values.loc[("scnt", "direction_median_wins")]),
    }


def plot_nonspatial_figures(
    results: NonspatialFigureResults,
    output_dir: str | Path,
    panels: NonspatialPanels | None = None,
    figures: Sequence[str] = FIGURE_IDS,
) -> dict[str, tuple[Path, Path]]:
    """Render selected grouped non-spatial figure pairs with lazy Matplotlib import."""

    selected = tuple(str(value).lower() for value in figures)
    if len(selected) != len(set(selected)) or any(value not in FIGURE_IDS for value in selected):
        raise ValueError(f"figures must be a unique subset of {FIGURE_IDS}")
    from ._nonspatial_figures_plot import render_nonspatial_figures

    return render_nonspatial_figures(
        results,
        panels or calculate_nonspatial_panels(results),
        prepare_output_dir(output_dir),
        selected,
    )
