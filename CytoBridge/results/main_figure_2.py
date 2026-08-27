"""Processed panel-e data and frozen panels for Main Figure 2 assembly."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ._io import read_json, require_files, resolve_results_dir


SPACES = ("gene", "physical")
TIMES = (1, 2, 3)
SEEDS = (1, 4, 8, 32, 256, 512, 1024, 2048, 4096, 8192)
METHODS_BY_SPACE = {
    "gene": ("STORIES", "stVCR", "CytoBridge"),
    "physical": ("stVCR", "CytoBridge"),
}
SUMMARY_COLUMNS = (
    "space", "time", "mean_w2", "sd_w2", "n", "min_w2", "max_w2",
    "mean_n_predicted", "sd_n_predicted", "se_w2", "ci95_halfwidth",
)
REPLICATE_COLUMNS = (
    "method", "seed", "time", "space", "w2", "n_predicted", "n_truth",
)
BASELINE_COLUMNS = ("space", "time", "method", "w2")
_FILES = (
    "w2_mean_sd_ci.csv",
    "w2_replicates_long.csv",
    "baseline_w2.csv",
    "frozen_panels_a_to_d.pdf",
    "manifest.json",
)


@dataclass(frozen=True)
class MainFigure2Data:
    """Panel-e tables and the frozen vector page containing panels a--d."""

    source_dir: Path
    manifest: dict[str, Any]
    summary: pd.DataFrame
    replicates: pd.DataFrame
    baselines: pd.DataFrame
    frozen_panels_pdf: Path


def _require_columns(
    frame: pd.DataFrame, columns: tuple[str, ...], *, source: Path
) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing columns: {missing}")


def _numeric(
    frame: pd.DataFrame, columns: tuple[str, ...], *, source: Path
) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        result[column] = pd.to_numeric(result[column], errors="raise")
    if not np.isfinite(result.loc[:, columns].to_numpy(dtype=float)).all():
        raise ValueError(f"{source} contains non-finite values")
    return result


def _integer_column(frame: pd.DataFrame, column: str, *, source: Path) -> None:
    values = frame[column].to_numpy(dtype=float)
    if not np.equal(values, np.floor(values)).all():
        raise ValueError(f"{source} contains non-integer {column} values")
    frame[column] = frame[column].astype(int)


def _validate_manifest(manifest: dict[str, Any], source: Path) -> None:
    if manifest.get("analysis") != "main_figure_2":
        raise ValueError(f"{source} must describe main_figure_2")
    if manifest.get("reader_action") != "result-summary-redraw + external-assembly":
        raise ValueError(f"{source} contains an unexpected reader action")
    if manifest.get("reproduction_scope") != "panel_e_redraw_plus_frozen_page_assembly":
        raise ValueError(f"{source} contains an unexpected reproduction scope")
    expected_files = set(_FILES).difference({"manifest.json"})
    if not isinstance(manifest.get("files"), dict) or set(manifest["files"]) != expected_files:
        raise ValueError(f"{source} contains an unexpected file roster")
    full_rerun = manifest.get("full_rerun")
    if not isinstance(full_rerun, dict) or full_rerun.get("included") is not False:
        raise ValueError(f"{source} must distinguish compact plotting from a full rerun")
    if not full_rerun.get("external_dependencies"):
        raise ValueError(f"{source} must list the external full-rerun dependencies")


def _validate_summary(frame: pd.DataFrame, source: Path) -> pd.DataFrame:
    _require_columns(frame, SUMMARY_COLUMNS, source=source)
    result = frame.loc[:, SUMMARY_COLUMNS].copy()
    result = _numeric(
        result,
        tuple(column for column in SUMMARY_COLUMNS if column != "space"),
        source=source,
    )
    _integer_column(result, "time", source=source)
    _integer_column(result, "n", source=source)
    expected = {(space, time) for space in SPACES for time in TIMES}
    observed = set(result[["space", "time"]].itertuples(index=False, name=None))
    if observed != expected or result.duplicated(["space", "time"]).any():
        raise ValueError(f"{source} must contain one row for each space and time")
    if not result["n"].eq(len(SEEDS)).all():
        raise ValueError(f"{source} must report ten replicates per row")
    if not result["mean_w2"].gt(0).all() or not result["sd_w2"].ge(0).all():
        raise ValueError(f"{source} contains invalid W2 summaries")
    return result.sort_values(["space", "time"]).reset_index(drop=True)


def _validate_replicates(frame: pd.DataFrame, source: Path) -> pd.DataFrame:
    _require_columns(frame, REPLICATE_COLUMNS, source=source)
    result = frame.loc[:, REPLICATE_COLUMNS].copy()
    result = _numeric(
        result, ("seed", "time", "w2", "n_predicted", "n_truth"), source=source
    )
    for column in ("seed", "time", "n_predicted", "n_truth"):
        _integer_column(result, column, source=source)
    if set(result["method"]) != {"CytoBridge"}:
        raise ValueError(f"{source} must contain only CytoBridge replicates")
    expected = {
        (space, time, seed)
        for space in SPACES for time in TIMES for seed in SEEDS
    }
    observed = set(
        result[["space", "time", "seed"]].itertuples(index=False, name=None)
    )
    if observed != expected or result.duplicated(["space", "time", "seed"]).any():
        raise ValueError(f"{source} must contain the complete space-time-seed grid")
    if not result["w2"].gt(0).all():
        raise ValueError(f"{source} contains non-positive W2 values")
    if not result[["n_predicted", "n_truth"]].gt(0).all().all():
        raise ValueError(f"{source} contains non-positive support sizes")
    return result.sort_values(["space", "time", "seed"]).reset_index(drop=True)


def _validate_baselines(frame: pd.DataFrame, source: Path) -> pd.DataFrame:
    _require_columns(frame, BASELINE_COLUMNS, source=source)
    result = _numeric(frame.loc[:, BASELINE_COLUMNS], ("time", "w2"), source=source)
    _integer_column(result, "time", source=source)
    expected = {
        (space, time, method)
        for space in SPACES
        for time in TIMES
        for method in METHODS_BY_SPACE[space]
        if method != "CytoBridge"
    }
    observed = set(
        result[["space", "time", "method"]].itertuples(index=False, name=None)
    )
    if observed != expected or result.duplicated(["space", "time", "method"]).any():
        raise ValueError(f"{source} must contain the complete baseline grid")
    if not result["w2"].gt(0).all():
        raise ValueError(f"{source} contains non-positive W2 values")
    return result.sort_values(["space", "method", "time"]).reset_index(drop=True)


def summarize_main_figure_2_replicates(replicates: pd.DataFrame) -> pd.DataFrame:
    """Calculate the plotted CytoBridge mean and sample SD values."""

    return (
        replicates.groupby(["space", "time"], sort=True)["w2"]
        .agg(mean_w2="mean", sd_w2="std", n="size")
        .reset_index()
        .sort_values(["space", "time"])
        .reset_index(drop=True)
    )


def _validate_summary_matches_replicates(
    summary: pd.DataFrame, replicates: pd.DataFrame, *, source: Path
) -> None:
    calculated = summarize_main_figure_2_replicates(replicates)
    reported = summary[["space", "time", "mean_w2", "sd_w2", "n"]]
    if not reported[["space", "time", "n"]].equals(
        calculated[["space", "time", "n"]]
    ):
        raise ValueError(f"{source} does not match the replicate grid")
    if not np.allclose(
        reported[["mean_w2", "sd_w2"]],
        calculated[["mean_w2", "sd_w2"]],
        rtol=1e-12,
        atol=1e-14,
    ):
        raise ValueError(f"{source} does not match the replicate calculations")


def load_main_figure_2(results_dir: str | Path | None = None) -> MainFigure2Data:
    """Load and validate the processed inputs for Main Figure 2."""

    source_dir = resolve_results_dir(results_dir, slug="main_figure_2")
    paths = require_files(source_dir, _FILES)
    manifest = read_json(paths["manifest.json"])
    _validate_manifest(manifest, paths["manifest.json"])
    summary = _validate_summary(
        pd.read_csv(paths["w2_mean_sd_ci.csv"]), paths["w2_mean_sd_ci.csv"]
    )
    replicates = _validate_replicates(
        pd.read_csv(paths["w2_replicates_long.csv"]), paths["w2_replicates_long.csv"]
    )
    baselines = _validate_baselines(
        pd.read_csv(paths["baseline_w2.csv"]), paths["baseline_w2.csv"]
    )
    _validate_summary_matches_replicates(
        summary, replicates, source=paths["w2_mean_sd_ci.csv"]
    )
    return MainFigure2Data(
        source_dir=source_dir,
        manifest=manifest,
        summary=summary,
        replicates=replicates,
        baselines=baselines,
        frozen_panels_pdf=paths["frozen_panels_a_to_d.pdf"],
    )


def write_main_figure_2_tables(
    data: MainFigure2Data, output_dir: str | Path
) -> dict[str, Path]:
    """Write the compact panel-e tables used by the renderer."""

    output = Path(output_dir)
    paths = {
        "summary": output / "w2_mean_sd_ci.csv",
        "replicates": output / "w2_replicates_long.csv",
        "baselines": output / "baseline_w2.csv",
    }
    data.summary.to_csv(paths["summary"], index=False)
    data.replicates.to_csv(paths["replicates"], index=False)
    data.baselines.to_csv(paths["baselines"], index=False)
    return paths


def assemble_main_figure_2(
    data: MainFigure2Data, output_dir: str | Path, *, dpi: int = 300
) -> tuple[Path, Path]:
    """Assemble frozen panels a--d with a redrawn panel e."""

    from ._main_figure_2_plot import assemble_main_figure_2 as _assemble

    return _assemble(data, output_dir, dpi=dpi)


def plot_main_figure_2(
    data: MainFigure2Data, output_dir: str | Path, *, dpi: int = 300
) -> tuple[Path, Path]:
    """Compatibility alias for :func:`assemble_main_figure_2`."""

    return assemble_main_figure_2(data, output_dir, dpi=dpi)


__all__ = [
    "MainFigure2Data",
    "assemble_main_figure_2",
    "load_main_figure_2",
    "plot_main_figure_2",
    "summarize_main_figure_2_replicates",
    "write_main_figure_2_tables",
]
