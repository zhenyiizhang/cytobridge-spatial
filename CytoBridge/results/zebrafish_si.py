"""Compact inputs for zebrafish Supplementary Figures S27--S34."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Iterator, Mapping

import numpy as np
import pandas as pd

from ._io import read_json, require_files, resolve_results_dir


FIGURE_IDS = tuple(f"s{number}" for number in range(27, 35))
S27_GROUPS = ("observed", "generated")
S29_CONDITIONS = ("Baseline", "YSL removal", "EVL removal")
ABLATION_VARIANTS = ("remove_YSL", "remove_EVL")
LOSS_CONDITIONS = (
    "formal_alpha_control",
    "alpha_expr_005",
    "ot_mass_10_to_1",
    "formal",
    "ot_mass_1_to_10",
)
LOSS_SPACES = ("joint", "pca", "spatial")
DAUGHTER_NOISE_LEVELS = (0.0, 0.01, 0.03, 0.06)

_FILES = (
    "celltype_colors.json",
    "s27_celltype_colors.json",
    "s27_observed_generated.npz",
    "s28_growth_per_cell.csv.gz",
    "s29_virtual_removal.npz",
    "s30_endpoint_spatial.npz",
    "s30_spatial_w1_curve.csv",
    "s30_centroid_by_seed.csv",
    "s31_gene_dynamics.csv.gz",
    "s32_loss_weight_metrics.csv",
    "s33_composition.csv.gz",
    "s33_observed_composition.csv.gz",
    "s33_lineage_values.csv.gz",
    "s33_lineage_pairs.csv",
    "s33_sensitivity.csv.gz",
    "s33_particle_counts.csv.gz",
    "s34_observed_expression.csv.gz",
    "s34_reconstructed_expression.csv.gz",
    "s34_reported_metrics.csv",
    "s34_top_genes.csv",
    "manifest.json",
)


@dataclass(frozen=True)
class PackedSpatialFrames:
    """Flat, non-object representation of labeled spatial frames."""

    xy: np.ndarray
    label_id: np.ndarray
    offsets: np.ndarray
    groups: np.ndarray
    times: np.ndarray
    label_names: np.ndarray

    @property
    def n_frames(self) -> int:
        """Number of frames in the packed array."""

        return int(self.times.size)

    @property
    def frame_counts(self) -> np.ndarray:
        """Number of points in each frame."""

        return np.diff(self.offsets)

    def frame(self, group: str, time: float) -> tuple[np.ndarray, np.ndarray]:
        """Return coordinates and string labels for one named frame."""

        matches = np.flatnonzero(
            (self.groups.astype(str) == str(group))
            & np.isclose(self.times, float(time), rtol=0.0, atol=1e-6)
        )
        if matches.size != 1:
            raise KeyError(f"Frame not found: group={group!r}, time={time:g}")
        index = int(matches[0])
        start, stop = (int(value) for value in self.offsets[index : index + 2])
        return self.xy[start:stop], self.label_names[self.label_id[start:stop]]

    def iter_frames(
        self, group: str | None = None
    ) -> Iterator[tuple[str, float, np.ndarray, np.ndarray]]:
        """Yield frames in stored display order."""

        for name, time in zip(self.groups.astype(str), self.times, strict=True):
            if group is None or name == group:
                xy, labels = self.frame(name, float(time))
                yield name, float(time), xy, labels


@dataclass(frozen=True)
class ZebrafishSIResults:
    """Processed inputs used by Supplementary Figures S27--S34."""

    source_dir: Path
    manifest: dict[str, Any]
    celltype_colors: Mapping[str, str]
    observed_generated_colors: Mapping[str, str]
    observed_generated: PackedSpatialFrames
    growth: pd.DataFrame
    virtual_removal: PackedSpatialFrames
    endpoint_baseline_xy: np.ndarray
    endpoint_ysl_xy: np.ndarray
    endpoint_evl_xy: np.ndarray
    ablation_w1_curve: pd.DataFrame
    ablation_centroid_by_seed: pd.DataFrame
    gene_dynamics: pd.DataFrame
    loss_weight_metrics: pd.DataFrame
    daughter_composition: pd.DataFrame
    daughter_observed_composition: pd.DataFrame
    daughter_lineage_values: pd.DataFrame
    daughter_lineage_pairs: pd.DataFrame
    daughter_sensitivity: pd.DataFrame
    daughter_particle_counts: pd.DataFrame
    observed_expression: pd.DataFrame
    reconstructed_expression: pd.DataFrame
    inverse_pca_reported_metrics: pd.DataFrame
    top_variable_genes: tuple[str, ...]


@dataclass(frozen=True)
class ZebrafishSIPanels:
    """Recalculated tables drawn by the eight figure renderers."""

    growth_scaled: pd.DataFrame
    growth_quantiles: pd.DataFrame
    ablation_centroid_summary: pd.DataFrame
    gene_zscores: pd.DataFrame
    loss_weight_summary: pd.DataFrame
    daughter_composition_summary: pd.DataFrame
    daughter_lineage_summary: pd.DataFrame
    daughter_sensitivity_summary: pd.DataFrame
    daughter_particle_summary: pd.DataFrame
    daughter_top_celltypes: tuple[str, ...]
    inverse_pca_metrics: pd.DataFrame


def _readonly(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value)
    result.setflags(write=False)
    return result


def _load_packed_frames(path: Path) -> PackedSpatialFrames:
    required = {"xy", "label_id", "offsets", "groups", "times", "label_names"}
    with np.load(path, allow_pickle=False) as payload:
        if set(payload.files) != required:
            raise ValueError(f"{path} has an unexpected array roster")
        arrays = {name: np.asarray(payload[name]) for name in required}
    if any(value.dtype.hasobject for value in arrays.values()):
        raise ValueError(f"{path} contains object arrays")
    xy = arrays["xy"]
    label_id = arrays["label_id"]
    offsets = arrays["offsets"]
    groups = arrays["groups"]
    times = arrays["times"]
    names = arrays["label_names"]
    if xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all():
        raise ValueError(f"{path} has invalid spatial coordinates")
    if label_id.ndim != 1 or len(label_id) != len(xy):
        raise ValueError(f"{path} has invalid label identifiers")
    if names.ndim != 1 or len(names) < 1 or len(set(names.astype(str))) != len(names):
        raise ValueError(f"{path} has invalid label names")
    if label_id.size and (
        int(label_id.min()) < 0 or int(label_id.max()) >= len(names)
    ):
        raise ValueError(f"{path} contains out-of-range label identifiers")
    if groups.ndim != 1 or times.ndim != 1 or len(groups) != len(times):
        raise ValueError(f"{path} has invalid frame metadata")
    if offsets.shape != (len(times) + 1,):
        raise ValueError(f"{path} has invalid frame offsets")
    if (
        int(offsets[0]) != 0
        or int(offsets[-1]) != len(xy)
        or np.any(np.diff(offsets) <= 0)
    ):
        raise ValueError(f"{path} has non-monotonic frame offsets")
    keys = list(zip(groups.astype(str), times.astype(float), strict=True))
    if len(keys) != len(set(keys)):
        raise ValueError(f"{path} contains duplicate group-time frames")
    return PackedSpatialFrames(
        xy=_readonly(xy),
        label_id=_readonly(label_id.astype(np.uint8, copy=False)),
        offsets=_readonly(offsets.astype(np.int32, copy=False)),
        groups=_readonly(groups),
        times=_readonly(times.astype(np.float32, copy=False)),
        label_names=_readonly(names),
    )


def _load_endpoint(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    names = {"baseline_xy", "ysl_xy", "evl_xy"}
    with np.load(path, allow_pickle=False) as payload:
        if set(payload.files) != names:
            raise ValueError(f"{path} has an unexpected array roster")
        arrays = tuple(np.asarray(payload[name]) for name in names)
        by_name = {name: np.asarray(payload[name]) for name in names}
    del arrays
    for name, value in by_name.items():
        if (
            value.dtype.hasobject
            or value.ndim != 2
            or value.shape[1] != 2
            or not np.isfinite(value).all()
        ):
            raise ValueError(f"{path} has invalid {name} coordinates")
    return (
        _readonly(by_name["baseline_xy"]),
        _readonly(by_name["ysl_xy"]),
        _readonly(by_name["evl_xy"]),
    )


def _table(
    path: Path,
    columns: tuple[str, ...],
    numeric: tuple[str, ...],
) -> pd.DataFrame:
    result = pd.read_csv(path)
    missing = sorted(set(columns).difference(result.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    result = result.loc[:, columns].copy()
    values = result.loc[:, numeric].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(values.to_numpy(float)).all():
        raise ValueError(f"{path} contains non-finite numeric values")
    result.loc[:, numeric] = values
    text_columns = [name for name in columns if name not in numeric]
    if text_columns and result.loc[:, text_columns].isna().any().any():
        raise ValueError(f"{path} contains missing text values")
    return result


def _validate_manifest(manifest: dict[str, Any], path: Path) -> None:
    if manifest.get("schema_version") != 1:
        raise ValueError(f"{path} has an unsupported schema version")
    if manifest.get("analysis") != "zebrafish_si_s27_s34":
        raise ValueError(f"{path} does not describe zebrafish S27--S34")
    if tuple(manifest.get("figures", {})) != FIGURE_IDS:
        raise ValueError(f"{path} has an unexpected figure roster")
    if set(manifest.get("files", {})) != set(_FILES).difference({"manifest.json"}):
        raise ValueError(f"{path} has an unexpected file roster")
    grids = manifest.get("time_grids", {})
    if grids.get("observed") != [0, 1, 2, 3, 4] or grids.get(
        "half_stage"
    ) != [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]:
        raise ValueError(f"{path} has unexpected time grids")


def _validate_frame_contracts(
    observed_generated: PackedSpatialFrames,
    virtual_removal: PackedSpatialFrames,
    observed_colors: Mapping[str, str],
    virtual_removal_colors: Mapping[str, str],
) -> None:
    if observed_generated.n_frames != 14 or len(observed_generated.xy) != 17_066:
        raise ValueError("S27 must contain 14 frames and 17,066 points")
    if tuple(dict.fromkeys(observed_generated.groups.astype(str))) != S27_GROUPS:
        raise ValueError("S27 has an unexpected group order")
    expected_s27 = {
        *(('observed', float(value)) for value in range(5)),
        *(('generated', float(value)) for value in np.arange(0, 4.5, 0.5)),
    }
    if set(
        zip(
            observed_generated.groups.astype(str),
            observed_generated.times.astype(float),
            strict=True,
        )
    ) != expected_s27:
        raise ValueError("S27 has an unexpected frame grid")
    if virtual_removal.n_frames != 15 or len(virtual_removal.xy) != 29_214:
        raise ValueError("S29 must contain 15 frames and 29,214 points")
    if tuple(dict.fromkeys(virtual_removal.groups.astype(str))) != S29_CONDITIONS:
        raise ValueError("S29 has an unexpected condition order")
    expected_s29 = {
        (condition, float(time))
        for condition in S29_CONDITIONS
        for time in range(5)
    }
    if set(
        zip(
            virtual_removal.groups.astype(str),
            virtual_removal.times.astype(float),
            strict=True,
        )
    ) != expected_s29:
        raise ValueError("S29 has an unexpected frame grid")
    if set(observed_generated.label_names.astype(str)) != set(observed_colors):
        raise ValueError("The S27 frames and color map use different labels")
    if set(virtual_removal.label_names.astype(str)) != set(virtual_removal_colors):
        raise ValueError("The S29 frames and color map use different labels")


def _validate_tables(results: ZebrafishSIResults) -> None:
    growth = results.growth
    if len(growth) != 11_999 or set(growth["time"].astype(float)) != set(range(5)):
        raise ValueError("S28 has an unexpected row count or time grid")
    expected_counts = {0.0: 563, 1.0: 1036, 2.0: 2081, 3.0: 3048, 4.0: 5271}
    if growth.groupby("time").size().to_dict() != expected_counts:
        raise ValueError("S28 has unexpected cells per observed stage")

    w1 = results.ablation_w1_curve
    if (
        len(w1) != 162
        or set(w1["variant"].astype(str)) != set(ABLATION_VARIANTS)
        or not w1["metric"].astype(str).eq("w1").all()
        or not w1["n"].astype(int).eq(5).all()
        or (w1[["mean", "sem"]].to_numpy(float) < 0).any()
    ):
        raise ValueError("S30 has an unexpected W1 summary")
    for _, part in w1.groupby("variant"):
        if not np.allclose(
            np.sort(part["time"].to_numpy(float)), np.arange(0, 4.05, 0.05)
        ):
            raise ValueError("S30 has an unexpected W1 time grid")
    centroid = results.ablation_centroid_by_seed
    if (
        len(centroid) != 10
        or set(centroid["variant"].astype(str)) != set(ABLATION_VARIANTS)
        or set(centroid["seed"].astype(int)) != set(range(42, 47))
        or not np.isclose(centroid["time"].to_numpy(float), 4.0).all()
    ):
        raise ValueError("S30 has an unexpected centroid table")

    genes = results.gene_dynamics
    if (
        len(genes) != 2250
        or genes["gene"].nunique() != 250
        or genes.duplicated(["gene", "time"]).any()
        or not np.allclose(
            np.sort(genes["time"].unique().astype(float)), np.arange(0, 4.5, 0.5)
        )
    ):
        raise ValueError("S31 has an unexpected gene-time matrix")

    loss = results.loss_weight_metrics
    if (
        len(loss) != 60
        or set(loss["condition"].astype(str)) != set(LOSS_CONDITIONS)
        or set(loss["space"].astype(str)) != set(LOSS_SPACES)
        or set(loss["time"].astype(float)) != {1.0, 2.0, 3.0, 4.0}
        or loss.duplicated(["condition", "space", "time"]).any()
        or (loss["w1"].to_numpy(float) < 0).any()
    ):
        raise ValueError("S32 has an unexpected metric grid")

    composition = results.daughter_composition
    comp_keys = ["daughter_noise_std", "seed", "time", "celltype"]
    if (
        len(composition) != 2128
        or composition.duplicated(comp_keys).any()
        or set(composition["daughter_noise_std"].astype(float))
        != set(DAUGHTER_NOISE_LEVELS)
        or set(composition["seed"].astype(int)) != set(range(42, 47))
        or not np.allclose(
            np.sort(composition["time"].unique().astype(float)),
            np.arange(0, 4.5, 0.5),
        )
        or not np.allclose(
            composition.groupby(comp_keys[:-1])["fraction"].sum().to_numpy(float),
            1.0,
            rtol=0.0,
            atol=2e-9,
        )
    ):
        raise ValueError("S33 has an unexpected composition table")
    observed = results.daughter_observed_composition
    if (
        len(observed) != 58
        or set(observed["time"].astype(float)) != set(range(5))
        or not np.allclose(
            observed.groupby("time")["fraction"].sum().to_numpy(float), 1.0
        )
    ):
        raise ValueError("S33 has an unexpected observed-composition table")
    lineage = results.daughter_lineage_values
    if (
        len(lineage) != 120
        or lineage.duplicated(
            [
                "daughter_noise_std",
                "seed",
                "source_celltype",
                "target_celltype",
            ]
        ).any()
        or ((lineage["fraction"] < 0) | (lineage["fraction"] > 1)).any()
    ):
        raise ValueError("S33 has an unexpected lineage table")
    pairs = results.daughter_lineage_pairs
    pair_set = set(zip(pairs["source_celltype"], pairs["target_celltype"], strict=True))
    if len(pairs) != 6 or len(pair_set) != 6 or pair_set != set(
        zip(lineage["source_celltype"], lineage["target_celltype"], strict=True)
    ):
        raise ValueError("S33 has an unexpected lineage-pair roster")
    for table, name in (
        (results.daughter_sensitivity, "sensitivity"),
        (results.daughter_particle_counts, "particle-count"),
    ):
        if (
            len(table) != 180
            or table.duplicated(["daughter_noise_std", "seed", "time"]).any()
        ):
            raise ValueError(f"S33 has an unexpected {name} table")

    observed_expression = results.observed_expression
    reconstructed = results.reconstructed_expression
    if (
        observed_expression.shape != (2584, 5)
        or reconstructed.shape != (2584, 5)
        or not observed_expression.index.equals(reconstructed.index)
        or not observed_expression.columns.equals(reconstructed.columns)
        or not np.allclose(
            observed_expression.columns.astype(float), [0, 1, 2, 3, 4]
        )
    ):
        raise ValueError("S34 has unexpected expression matrices")
    reported = results.inverse_pca_reported_metrics
    if (
        len(reported) != 5
        or set(reported["time"].astype(float)) != set(range(5))
        or not reported["decoder"].astype(str).eq("clipped_inverse_pca").all()
        or not reported["scope"].astype(str).eq("all_active_features").all()
        or len(results.top_variable_genes) != 250
        or not set(results.top_variable_genes).issubset(observed_expression.index)
    ):
        raise ValueError("S34 has an unexpected reconstruction scope")


def load_zebrafish_si_results(
    results_dir: str | Path | None = None,
) -> ZebrafishSIResults:
    """Load and validate the compact S27--S34 result bundle."""

    source_dir = resolve_results_dir(results_dir, slug="zebrafish_si")
    paths = require_files(source_dir, _FILES)
    manifest = read_json(paths["manifest.json"])
    _validate_manifest(manifest, paths["manifest.json"])
    def color_map(path: Path) -> dict[str, Any]:
        value = read_json(path)
        if (
            len(value) != 42
            or len(set(value)) != 42
            or any(
                re.fullmatch(r"#[0-9a-fA-F]{6}", str(color)) is None
                for color in value.values()
            )
        ):
            raise ValueError(f"{path} is not a 42-label color map")
        return value

    colors = color_map(paths["celltype_colors.json"])
    observed_colors = color_map(paths["s27_celltype_colors.json"])

    observed_generated = _load_packed_frames(paths["s27_observed_generated.npz"])
    virtual_removal = _load_packed_frames(paths["s29_virtual_removal.npz"])
    _validate_frame_contracts(
        observed_generated, virtual_removal, observed_colors, colors
    )
    baseline, ysl, evl = _load_endpoint(paths["s30_endpoint_spatial.npz"])
    if (baseline.shape, ysl.shape, evl.shape) != ((5177, 2), (4546, 2), (3306, 2)):
        raise ValueError("S30 has unexpected endpoint point counts")

    growth = _table(
        paths["s28_growth_per_cell.csv.gz"],
        ("time", "x", "y", "growth"),
        ("time", "x", "y", "growth"),
    )
    w1_curve = _table(
        paths["s30_spatial_w1_curve.csv"],
        ("variant", "metric", "time", "mean", "sem", "n"),
        ("time", "mean", "sem", "n"),
    )
    centroid = _table(
        paths["s30_centroid_by_seed.csv"],
        ("seed", "variant", "time", "centroid_shift"),
        ("seed", "time", "centroid_shift"),
    )
    gene_dynamics = _table(
        paths["s31_gene_dynamics.csv.gz"],
        ("time", "gene", "mean_clipped_log1p"),
        ("time", "mean_clipped_log1p"),
    )
    loss = _table(
        paths["s32_loss_weight_metrics.csv"],
        ("time", "space", "w1", "condition"),
        ("time", "w1"),
    )
    composition = _table(
        paths["s33_composition.csv.gz"],
        (
            "daughter_noise_std",
            "seed",
            "time",
            "celltype",
            "count",
            "fraction",
            "n_particles",
        ),
        (
            "daughter_noise_std",
            "seed",
            "time",
            "count",
            "fraction",
            "n_particles",
        ),
    )
    observed_composition = _table(
        paths["s33_observed_composition.csv.gz"],
        ("time", "celltype", "count", "fraction", "n_cells"),
        ("time", "count", "fraction", "n_cells"),
    )
    lineage_values = _table(
        paths["s33_lineage_values.csv.gz"],
        (
            "daughter_noise_std",
            "seed",
            "fraction",
            "source_celltype",
            "target_celltype",
        ),
        ("daughter_noise_std", "seed", "fraction"),
    )
    lineage_pairs = _table(
        paths["s33_lineage_pairs.csv"],
        ("source_celltype", "target_celltype", "global_flow"),
        ("global_flow",),
    )
    sensitivity = _table(
        paths["s33_sensitivity.csv.gz"],
        (
            "seed",
            "time",
            "daughter_noise_std",
            "composition_tv_from_reference",
            "composition_max_abs_fraction_change",
            "lineage_weighted_tv_from_reference",
            "lineage_max_source_tv_from_reference",
        ),
        (
            "seed",
            "time",
            "daughter_noise_std",
            "composition_tv_from_reference",
            "composition_max_abs_fraction_change",
            "lineage_weighted_tv_from_reference",
            "lineage_max_source_tv_from_reference",
        ),
    )
    particles = _table(
        paths["s33_particle_counts.csv.gz"],
        ("daughter_noise_std", "seed", "time", "n_particles"),
        ("daughter_noise_std", "seed", "time", "n_particles"),
    )

    def expression(path: Path) -> pd.DataFrame:
        frame = pd.read_csv(path)
        if "gene" not in frame.columns or frame["gene"].duplicated().any():
            raise ValueError(f"{path} has invalid gene identifiers")
        result = frame.set_index("gene")
        result.columns = pd.Index([float(value) for value in result.columns])
        if not np.isfinite(result.to_numpy(float)).all():
            raise ValueError(f"{path} contains non-finite expression values")
        return result

    observed_expression = expression(paths["s34_observed_expression.csv.gz"])
    reconstructed_expression = expression(
        paths["s34_reconstructed_expression.csv.gz"]
    )
    reported = _table(
        paths["s34_reported_metrics.csv"],
        (
            "time",
            "decoder",
            "scope",
            "n_cells",
            "n_features",
            "rmse",
            "mae",
            "mean_bias",
            "pearson_r",
        ),
        (
            "time",
            "n_cells",
            "n_features",
            "rmse",
            "mae",
            "mean_bias",
            "pearson_r",
        ),
    )
    top = _table(paths["s34_top_genes.csv"], ("gene",), ())

    results = ZebrafishSIResults(
        source_dir=source_dir,
        manifest=manifest,
        celltype_colors=colors,
        observed_generated_colors=observed_colors,
        observed_generated=observed_generated,
        growth=growth,
        virtual_removal=virtual_removal,
        endpoint_baseline_xy=baseline,
        endpoint_ysl_xy=ysl,
        endpoint_evl_xy=evl,
        ablation_w1_curve=w1_curve,
        ablation_centroid_by_seed=centroid,
        gene_dynamics=gene_dynamics,
        loss_weight_metrics=loss,
        daughter_composition=composition,
        daughter_observed_composition=observed_composition,
        daughter_lineage_values=lineage_values,
        daughter_lineage_pairs=lineage_pairs,
        daughter_sensitivity=sensitivity,
        daughter_particle_counts=particles,
        observed_expression=observed_expression,
        reconstructed_expression=reconstructed_expression,
        inverse_pca_reported_metrics=reported,
        top_variable_genes=tuple(top["gene"].astype(str)),
    )
    _validate_tables(results)
    return results


def _complete_composition(composition: pd.DataFrame) -> pd.DataFrame:
    index = pd.MultiIndex.from_product(
        [
            sorted(composition["daughter_noise_std"].unique()),
            sorted(composition["seed"].unique()),
            sorted(composition["time"].unique()),
            sorted(composition["celltype"].unique()),
        ],
        names=["daughter_noise_std", "seed", "time", "celltype"],
    )
    return (
        composition.set_index(list(index.names))["fraction"]
        .reindex(index, fill_value=0.0)
        .rename("fraction")
        .reset_index()
    )


def _group_summary(
    frame: pd.DataFrame,
    by: list[str],
    values: list[str],
) -> pd.DataFrame:
    records: list[dict[str, float | int | str]] = []
    for keys, part in frame.groupby(by, sort=True):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        record: dict[str, float | int | str] = dict(zip(by, key_values, strict=True))
        record["n"] = int(len(part))
        for column in values:
            array = part[column].to_numpy(float)
            record[f"{column}_mean"] = float(array.mean())
            record[f"{column}_std"] = float(array.std(ddof=1))
            record[f"{column}_sem"] = float(array.std(ddof=1) / np.sqrt(len(array)))
        records.append(record)
    return pd.DataFrame(records)


def _inverse_pca_metrics(results: ZebrafishSIResults) -> pd.DataFrame:
    reported = results.inverse_pca_reported_metrics.set_index("time")
    rows = []
    for time in results.observed_expression.columns.astype(float):
        observed = results.observed_expression[time].to_numpy(float)
        reconstructed = results.reconstructed_expression[time].to_numpy(float)
        delta = reconstructed - observed
        rows.append(
            {
                "time": float(time),
                "n_features": int(observed.size),
                "rmse": float(np.sqrt(np.mean(delta**2))),
                "mae": float(np.mean(np.abs(delta))),
                "mean_bias": float(np.mean(delta)),
                "pearson_r": float(np.corrcoef(observed, reconstructed)[0, 1]),
                "n_cells": int(reported.loc[time, "n_cells"]),
            }
        )
    calculated = pd.DataFrame(rows)
    reference = reported.loc[calculated["time"]]
    for column in ("rmse", "mae", "mean_bias", "pearson_r"):
        if not np.allclose(
            calculated[column], reference[column], rtol=2e-6, atol=2e-8
        ):
            raise ValueError(f"Calculated inverse-PCA {column} disagrees with its table")
    if not np.array_equal(
        calculated["n_features"].to_numpy(int),
        reference["n_features"].to_numpy(int),
    ):
        raise ValueError("Calculated inverse-PCA feature counts disagree with their table")
    return calculated


def calculate_zebrafish_si_panels(
    results: ZebrafishSIResults,
) -> ZebrafishSIPanels:
    """Recalculate every transformed quantity displayed in S27--S34."""

    growth = results.growth.copy()
    quantiles = (
        growth.groupby("time", sort=True)["growth"]
        .quantile([0.05, 0.95])
        .unstack()
        .rename(columns={0.05: "q05", 0.95: "q95"})
        .reset_index()
    )
    quantiles["n_cells"] = growth.groupby("time", sort=True).size().to_numpy(int)
    growth = growth.merge(quantiles[["time", "q05", "q95"]], on="time")
    growth["growth_scaled"] = np.clip(
        (growth["growth"] - growth["q05"])
        / np.maximum(growth["q95"] - growth["q05"], 1e-12),
        0,
        1,
    )

    from scipy import stats

    centroid_rows = []
    for variant in ABLATION_VARIANTS:
        values = results.ablation_centroid_by_seed.loc[
            results.ablation_centroid_by_seed["variant"].eq(variant),
            "centroid_shift",
        ].to_numpy(float)
        mean = float(values.mean())
        sem = float(stats.sem(values))
        low, high = stats.t.interval(
            0.95, df=len(values) - 1, loc=mean, scale=sem
        )
        centroid_rows.append(
            {
                "variant": variant,
                "n": int(len(values)),
                "mean": mean,
                "sem": sem,
                "ci95_low": float(low),
                "ci95_high": float(high),
            }
        )
    centroid_summary = pd.DataFrame(centroid_rows)

    gene_matrix = results.gene_dynamics.pivot(
        index="gene", columns="time", values="mean_clipped_log1p"
    ).reindex(columns=np.arange(0, 4.5, 0.5))
    values = gene_matrix.to_numpy(float)
    std = values.std(axis=1, ddof=0)
    std[std == 0] = 1.0
    zscores = (values - values.mean(axis=1, keepdims=True)) / std[:, None]
    order = np.lexsort((-np.var(zscores, axis=1), np.argmax(zscores, axis=1)))
    gene_zscores = pd.DataFrame(
        zscores[order],
        index=pd.Index(gene_matrix.index.to_numpy()[order], name="gene"),
        columns=gene_matrix.columns,
    )

    loss_summary = (
        results.loss_weight_metrics.groupby(["condition", "space"], sort=False)[
            "w1"
        ]
        .mean()
        .rename("mean_w1")
        .reset_index()
    )

    completed = _complete_composition(results.daughter_composition)
    composition_summary = (
        completed.groupby(
            ["daughter_noise_std", "time", "celltype"], sort=True
        )["fraction"]
        .agg(mean="mean", std="std", count="count")
        .reset_index()
    )
    composition_summary["sem"] = composition_summary["std"] / np.sqrt(
        composition_summary["count"]
    )
    lineage_summary = (
        results.daughter_lineage_values.groupby(
            ["daughter_noise_std", "source_celltype", "target_celltype"],
            sort=True,
        )["fraction"]
        .agg(mean="mean", std="std", count="count")
        .reset_index()
    )
    lineage_summary["sem"] = lineage_summary["std"] / np.sqrt(
        lineage_summary["count"]
    )
    sensitivity_summary = _group_summary(
        results.daughter_sensitivity,
        ["daughter_noise_std", "time"],
        [
            "composition_tv_from_reference",
            "composition_max_abs_fraction_change",
            "lineage_weighted_tv_from_reference",
            "lineage_max_source_tv_from_reference",
        ],
    )
    particle_summary = _group_summary(
        results.daughter_particle_counts,
        ["daughter_noise_std", "time"],
        ["n_particles"],
    )
    top_celltypes = tuple(
        results.daughter_composition.loc[
            results.daughter_composition["time"] > 0
        ]
        .groupby("celltype")["fraction"]
        .mean()
        .sort_values(ascending=False)
        .head(6)
        .index.astype(str)
    )

    return ZebrafishSIPanels(
        growth_scaled=growth,
        growth_quantiles=quantiles,
        ablation_centroid_summary=centroid_summary,
        gene_zscores=gene_zscores,
        loss_weight_summary=loss_summary,
        daughter_composition_summary=composition_summary,
        daughter_lineage_summary=lineage_summary,
        daughter_sensitivity_summary=sensitivity_summary,
        daughter_particle_summary=particle_summary,
        daughter_top_celltypes=top_celltypes,
        inverse_pca_metrics=_inverse_pca_metrics(results),
    )


def write_zebrafish_si_tables(
    panels: ZebrafishSIPanels,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write the recalculated panel tables to an output directory."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    tables = {
        "growth_quantiles": panels.growth_quantiles,
        "ablation_centroid": panels.ablation_centroid_summary,
        "gene_zscores": panels.gene_zscores.reset_index(),
        "loss_weight": panels.loss_weight_summary,
        "daughter_composition": panels.daughter_composition_summary,
        "daughter_lineage": panels.daughter_lineage_summary,
        "daughter_sensitivity": panels.daughter_sensitivity_summary,
        "daughter_particles": panels.daughter_particle_summary,
        "inverse_pca": panels.inverse_pca_metrics,
    }
    paths = {}
    for name, table in tables.items():
        path = output / f"{name}.csv"
        table.to_csv(path, index=False)
        paths[name] = path
    return paths


def zebrafish_si_statistics(
    results: ZebrafishSIResults,
    panels: ZebrafishSIPanels | None = None,
) -> dict[str, Any]:
    """Return a compact JSON-ready summary of the reproduction bundle."""

    calculated = panels or calculate_zebrafish_si_panels(results)
    centroid = calculated.ablation_centroid_summary.set_index("variant")
    inverse = calculated.inverse_pca_metrics.set_index("time")
    endpoint = calculated.daughter_sensitivity_summary.loc[
        np.isclose(calculated.daughter_sensitivity_summary["time"], 4.0)
        & np.isclose(
            calculated.daughter_sensitivity_summary["daughter_noise_std"], 0.06
        )
    ].iloc[0]
    return {
        "figure_ids": list(FIGURE_IDS),
        "bundle_bytes": int(
            sum(path.stat().st_size for path in results.source_dir.iterdir())
        ),
        "s27_points": int(len(results.observed_generated.xy)),
        "s28_cells": int(len(results.growth)),
        "s29_points": int(len(results.virtual_removal.xy)),
        "s30_endpoint_centroid_mean": {
            variant: float(centroid.loc[variant, "mean"])
            for variant in ABLATION_VARIANTS
        },
        "s31_genes": int(len(calculated.gene_zscores)),
        "s32_metric_rows": int(len(results.loss_weight_metrics)),
        "s33_noise_006_t4_composition_tv_percent": float(
            100 * endpoint["composition_tv_from_reference_mean"]
        ),
        "s34_t4_pearson_r": float(inverse.loc[4.0, "pearson_r"]),
    }


def plot_zebrafish_si(
    results: ZebrafishSIResults,
    output_dir: str | Path,
    panels: ZebrafishSIPanels | None = None,
    figures: tuple[str, ...] | list[str] | None = None,
) -> dict[str, tuple[Path, Path]]:
    """Render any subset of S27--S34 while keeping Matplotlib optional."""

    from ._zebrafish_si_plot import render_zebrafish_si

    selected = FIGURE_IDS if figures is None else tuple(str(value).lower() for value in figures)
    unknown = sorted(set(selected).difference(FIGURE_IDS))
    if unknown:
        raise ValueError(f"Unknown zebrafish figure identifiers: {unknown}")
    if len(selected) != len(set(selected)):
        raise ValueError("Figure identifiers must be unique")
    calculated = panels or calculate_zebrafish_si_panels(results)
    return render_zebrafish_si(results, calculated, output_dir, selected)
