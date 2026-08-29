#!/usr/bin/env python3
"""Render the corrected S22 states in the original article-style 3 x 3 layout.

This is a read-only renderer.  It accepts only a complete formal S22 stage,
verifies the exact stage and run manifests plus every artifact recorded by the
stage, and then alternates independent observed integer-time references with
half-time states from the one continuous global-t0 simulation.  It neither
reruns the model nor changes the retained all-generated S22 audit.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.font_manager import FontProperties, findfont  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402


SCHEMA_VERSION = 1
FIGURE_ID = "zebrafish_s22_article_style_mixed_sources"
GLOBAL_TIMES = tuple(float(value) for value in np.arange(0.0, 4.0 + 0.5, 0.5))
OBSERVED_TIMES = (0.0, 1.0, 2.0, 3.0, 4.0)
PANELS = (
    ("observed", 0.0),
    ("generated", 0.5),
    ("observed", 1.0),
    ("generated", 1.5),
    ("observed", 2.0),
    ("generated", 2.5),
    ("observed", 3.0),
    ("generated", 3.5),
    ("observed", 4.0),
)
EXPECTED_ACCEPTANCE = {
    "status": "PASS",
    "datasets": {"zebrafish": {"status": "PASS"}},
}
GENERATED_DIR = "global_t0_fixed_population_states"
OBSERVED_DIR = "observed_reference_states"
LEGEND_RELATIVE_PATH = Path("mosaic_snapshots/label_legend.svg")
SUPPORT_RELATIVE_PATH = Path("S22_trajectory_support_audit.json")


@dataclass(frozen=True)
class Frame:
    source: str
    time: float
    path: Path
    sha256: str
    points: np.ndarray
    labels: np.ndarray


@dataclass(frozen=True)
class VerifiedInputs:
    stage_root: Path
    stage_manifest_path: Path
    stage_manifest_sha256: str
    stage_manifest: Mapping[str, Any]
    run_manifest_path: Path
    run_manifest_sha256: str
    run_manifest: Mapping[str, Any]
    frames: tuple[Frame, ...]
    colors: Mapping[str, str]
    verified_artifacts: tuple[Mapping[str, Any], ...]
    support_audit: Mapping[str, Any]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path, description: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {description}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not read {description}: {path}") from exc
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{description} must contain a JSON object: {path}")
    return value


def _require_sha256(value: str, *, name: str) -> str:
    normalized = str(value).strip().lower()
    if re.fullmatch(r"[0-9a-f]{64}", normalized) is None:
        raise ValueError(f"{name} must be exactly 64 hexadecimal characters")
    return normalized


def _same_float_sequence(actual: Any, expected: Sequence[float]) -> bool:
    if not isinstance(actual, list) or len(actual) != len(expected):
        return False
    try:
        return bool(
            np.allclose(
                np.asarray(actual, dtype=float),
                np.asarray(expected, dtype=float),
                rtol=0.0,
                atol=1e-12,
            )
        )
    except (TypeError, ValueError):
        return False


def _relative_to_declared_root(declared: Path, declared_root: Path) -> Path:
    try:
        return declared.relative_to(declared_root)
    except ValueError as exc:
        raise RuntimeError(
            f"Stage artifact escapes the declared formal S22 root: {declared}"
        ) from exc


def _verify_stage_artifacts(
    *,
    stage_root: Path,
    stage_manifest: Mapping[str, Any],
    declared_stage_root: Path,
) -> tuple[tuple[Mapping[str, Any], ...], Mapping[str, Mapping[str, Any]]]:
    records = stage_manifest.get("output_artifacts")
    outputs = stage_manifest.get("outputs")
    if not isinstance(records, list) or not records:
        raise RuntimeError("S22 stage manifest has no output_artifacts")
    if not isinstance(outputs, list):
        raise RuntimeError("S22 stage manifest has no outputs list")
    recorded_paths = [str(record.get("path", "")) for record in records]
    if len(recorded_paths) != len(set(recorded_paths)):
        raise RuntimeError("S22 stage manifest contains duplicate artifact paths")
    if recorded_paths != [str(value) for value in outputs]:
        raise RuntimeError("S22 outputs and output_artifacts are not identical")

    verified: list[Mapping[str, Any]] = []
    by_relative: dict[str, Mapping[str, Any]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise RuntimeError("Invalid S22 output_artifacts entry")
        declared = Path(str(record.get("path", "")))
        relative = _relative_to_declared_root(declared, declared_stage_root)
        relative_key = relative.as_posix()
        if relative_key in by_relative:
            raise RuntimeError(f"Duplicate S22 relative artifact: {relative_key}")
        local_path = stage_root / relative
        if not local_path.is_file():
            raise FileNotFoundError(f"Missing recorded S22 artifact: {local_path}")
        expected_hash = _require_sha256(
            str(record.get("sha256", "")), name=f"artifact SHA-256 ({relative_key})"
        )
        observed_hash = _sha256(local_path)
        if observed_hash != expected_hash:
            raise RuntimeError(
                f"S22 artifact SHA-256 mismatch for {relative_key}: "
                f"expected {expected_hash}, observed {observed_hash}"
            )
        expected_size = int(record.get("size_bytes", -1))
        observed_size = local_path.stat().st_size
        if expected_size != observed_size:
            raise RuntimeError(
                f"S22 artifact size mismatch for {relative_key}: "
                f"expected {expected_size}, observed {observed_size}"
            )
        item = {
            "relative_path": relative_key,
            "sha256": observed_hash,
            "size_bytes": observed_size,
        }
        verified.append(item)
        by_relative[relative_key] = item
    return tuple(verified), by_relative


def _verify_stage_semantics(stage_manifest: Mapping[str, Any]) -> None:
    if stage_manifest.get("schema_version") != 1:
        raise RuntimeError("Unsupported S22 stage-manifest schema")
    if (
        stage_manifest.get("stage") != "s22"
        or stage_manifest.get("status") != "complete"
    ):
        raise RuntimeError("Input must be a completed formal S22 stage")
    signature = str(stage_manifest.get("signature", ""))
    _require_sha256(signature, name="S22 stage signature")
    settings = stage_manifest.get("settings")
    details = stage_manifest.get("details")
    if not isinstance(settings, Mapping) or not isinstance(details, Mapping):
        raise RuntimeError("S22 stage lacks settings/details provenance")
    required = {
        "trajectory_mode": "global_t0_fixed_population_state_transport",
        "population_mode": "fixed_population_state_transport",
        "growth_alpha": 0.0,
        "split_sde_piecewise": False,
        "piecewise_observed_sample_mode": None,
        "piecewise_include_end": None,
        "use_real_for_observed_trajectory_frames": False,
        "observed_integer_frames": "separate_reference_only",
        "mosaic_is_subsample_of_single_global_t0_simulation": True,
    }
    for key, expected in required.items():
        if settings.get(key) != expected:
            raise RuntimeError(
                f"Incompatible S22 setting {key!r}: "
                f"expected {expected!r}, observed {settings.get(key)!r}"
            )
    if not _same_float_sequence(settings.get("mosaic_times"), GLOBAL_TIMES):
        raise RuntimeError("S22 mosaic_times are not the complete 0.0-to-4.0 half grid")
    display_warp = settings.get("display_warp")
    if (
        not isinstance(display_warp, Mapping)
        or display_warp.get("applied") is not False
    ):
        raise RuntimeError("S22 display-warp provenance is not explicitly false")
    detail_required = {
        "single_global_t0_simulation_for_mosaic_and_video": True,
        "particle_count_constant_across_all_frames": True,
        "growth_head_applied_to_transport": False,
        "observed_integer_frames_substituted_into_trajectory": False,
        "display_warp_applied": False,
    }
    for key, expected in detail_required.items():
        if details.get(key) != expected:
            raise RuntimeError(
                f"Incompatible S22 detail {key!r}: "
                f"expected {expected!r}, observed {details.get(key)!r}"
            )
    if not _same_float_sequence(
        details.get("observed_reference_times"), OBSERVED_TIMES
    ):
        raise RuntimeError("S22 observed-reference time provenance is incomplete")


def _verify_acceptance(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"Missing {name}")
    if value.get("required_exact") != EXPECTED_ACCEPTANCE:
        raise RuntimeError(f"{name} does not bind the accepted zebrafish result")
    _require_sha256(str(value.get("sha256", "")), name=f"{name} SHA-256")
    for key in (
        "aligned_h5ad_entry",
        "model_dir_entry",
        "observed_run_root",
        "path",
    ):
        if not str(value.get(key, "")).strip():
            raise RuntimeError(f"{name} lacks {key!r}")
    return value


def _verify_run_binding(
    *,
    run_manifest: Mapping[str, Any],
    stage_manifest: Mapping[str, Any],
    declared_stage_manifest_path: Path,
) -> None:
    if run_manifest.get("schema_version") != 1:
        raise RuntimeError("Unsupported formal run-manifest schema")
    if run_manifest.get("workflow") != "zebrafish_native_paper_downstream":
        raise RuntimeError("Run manifest is not the formal zebrafish paper workflow")
    if run_manifest.get("profile") != "full":
        raise RuntimeError("S22 article renderer accepts only a full formal run")
    completed = run_manifest.get("completed_stages")
    if not isinstance(completed, list) or "s22" not in completed:
        raise RuntimeError("Formal run manifest does not mark S22 complete")
    signatures = run_manifest.get("stage_signatures")
    manifests = run_manifest.get("stage_manifests")
    if not isinstance(signatures, Mapping) or not isinstance(manifests, Mapping):
        raise RuntimeError("Formal run manifest lacks stage bindings")
    if signatures.get("s22") != stage_manifest.get("signature"):
        raise RuntimeError("Run/stage S22 signatures disagree")
    if Path(str(manifests.get("s22", ""))) != declared_stage_manifest_path:
        raise RuntimeError("Run manifest points to a different formal S22 stage")
    stage_acceptance = _verify_acceptance(
        stage_manifest.get("canonical_matched_acceptance"),
        name="stage canonical matched acceptance",
    )
    run_acceptance = _verify_acceptance(
        run_manifest.get("canonical_matched_acceptance"),
        name="run canonical matched acceptance",
    )
    if stage_acceptance != run_acceptance:
        raise RuntimeError("Run/stage canonical matched-acceptance bindings disagree")
    common = run_manifest.get("common")
    if not isinstance(common, Mapping):
        raise RuntimeError("Formal run manifest lacks common provenance")
    if common.get("canonical_matched_acceptance") != run_acceptance:
        raise RuntimeError("Common/run canonical matched-acceptance bindings disagree")
    git = common.get("git")
    if (
        not isinstance(git, Mapping)
        or re.fullmatch(r"[0-9a-f]{40}", str(git.get("commit", ""))) is None
        or git.get("dirty") is not False
    ):
        raise RuntimeError("Formal run is not bound to a clean 40-character Git commit")
    for key in (
        "aligned_h5ad_sha256",
        "weight_sha256",
        "score_sha256",
        "runner_sha256",
    ):
        _require_sha256(str(common.get(key, "")), name=f"formal {key}")


def _load_state_index(
    *,
    stage_root: Path,
    directory: str,
    expected_times: Sequence[float],
    expected_sources: Sequence[str],
    recorded: Mapping[str, Mapping[str, Any]],
) -> tuple[Mapping[str, Any], dict[float, Frame]]:
    index_relative = Path(directory) / "index.json"
    index_key = index_relative.as_posix()
    if index_key not in recorded:
        raise RuntimeError(
            f"State index is not recorded by the S22 manifest: {index_key}"
        )
    index_path = stage_root / index_relative
    index = _load_json(index_path, f"{directory} index")
    if index.get("schema_version") != 1 or index.get("annotation_key") != "Annotation":
        raise RuntimeError(f"Unsupported {directory} index contract")
    entries = index.get("frames")
    if not isinstance(entries, list) or len(entries) != len(expected_times):
        raise RuntimeError(f"{directory} index has the wrong frame count")
    actual_times = [
        entry.get("time") for entry in entries if isinstance(entry, Mapping)
    ]
    if not _same_float_sequence(actual_times, expected_times):
        raise RuntimeError(f"{directory} index has the wrong time grid")
    frames: dict[float, Frame] = {}
    feature_dim: int | None = None
    generated_count: int | None = None
    for position, (entry, expected_time, expected_source) in enumerate(
        zip(entries, expected_times, expected_sources)
    ):
        if not isinstance(entry, Mapping):
            raise RuntimeError(f"Invalid frame record in {directory}")
        expected_file = f"frame_{position:03d}.npz"
        if (
            entry.get("index") != position
            or entry.get("file") != expected_file
            or str(entry.get("key")) != str(float(expected_time))
            or entry.get("source") != expected_source
        ):
            raise RuntimeError(
                f"{directory}/{expected_file} provenance is incompatible"
            )
        relative = Path(directory) / expected_file
        relative_key = relative.as_posix()
        if relative_key not in recorded:
            raise RuntimeError(f"State frame is not recorded by S22: {relative_key}")
        path = stage_root / relative
        frame_hash = _sha256(path)
        if frame_hash != _require_sha256(
            str(entry.get("sha256", "")), name=f"index SHA-256 ({relative_key})"
        ):
            raise RuntimeError(f"State-index SHA-256 mismatch: {relative_key}")
        with np.load(path, allow_pickle=False) as payload:
            if set(payload.files) != {"points", "labels"}:
                raise RuntimeError(f"Unexpected arrays in state frame: {relative_key}")
            points = np.asarray(payload["points"])
            labels = np.asarray(payload["labels"]).astype(str)
        if points.ndim != 2 or points.shape[0] < 1 or points.shape[1] < 2:
            raise RuntimeError(f"Invalid points matrix: {relative_key}")
        if labels.ndim != 1 or labels.shape[0] != points.shape[0]:
            raise RuntimeError(f"Invalid labels vector: {relative_key}")
        if not np.isfinite(points).all():
            raise RuntimeError(f"Non-finite values in state frame: {relative_key}")
        if int(entry.get("n_cells", -1)) != points.shape[0]:
            raise RuntimeError(f"Cell count mismatch: {relative_key}")
        if int(entry.get("feature_dim", -1)) != points.shape[1]:
            raise RuntimeError(f"Feature dimension mismatch: {relative_key}")
        if feature_dim is None:
            feature_dim = points.shape[1]
        elif points.shape[1] != feature_dim:
            raise RuntimeError(f"Feature dimension changes within {directory}")
        if directory == GENERATED_DIR:
            if generated_count is None:
                generated_count = points.shape[0]
            elif points.shape[0] != generated_count:
                raise RuntimeError(
                    "Global-t0 fixed-population frames changed particle count"
                )
        frames[float(expected_time)] = Frame(
            source=("generated" if directory == GENERATED_DIR else "observed"),
            time=float(expected_time),
            path=path,
            sha256=frame_hash,
            points=points,
            labels=labels,
        )
    return index, frames


def _parse_signed_legend(path: Path) -> Mapping[str, str]:
    text = path.read_text(encoding="utf-8")
    colors = re.findall(
        r'<use\b[^>]*\bx="24\.3"[^>]*\bstyle="[^"]*\bfill:\s*(#[0-9a-fA-F]{6})[^"]*"',
        text,
    )
    labels = [
        html.unescape(value.strip())
        for value in re.findall(r"<!--\s*(.*?)\s*-->", text)
    ]
    if not colors or len(colors) != len(labels):
        raise RuntimeError(
            "Could not recover the complete cell-type palette from the signed legend SVG"
        )
    if len(labels) != len(set(labels)):
        raise RuntimeError("Signed legend SVG contains duplicate cell-type labels")
    return dict(zip(labels, [value.lower() for value in colors]))


def verify_inputs(
    *,
    stage_root: str | Path,
    expected_stage_manifest_sha256: str,
    run_manifest: str | Path | None,
    expected_run_manifest_sha256: str,
) -> VerifiedInputs:
    root = Path(stage_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Missing formal S22 stage root: {root}")
    stage_manifest_path = root / "stage_manifest.json"
    expected_stage_hash = _require_sha256(
        expected_stage_manifest_sha256, name="expected S22 stage-manifest SHA-256"
    )
    observed_stage_hash = _sha256(stage_manifest_path)
    if observed_stage_hash != expected_stage_hash:
        raise RuntimeError(
            "S22 stage-manifest SHA-256 mismatch: "
            f"expected {expected_stage_hash}, observed {observed_stage_hash}"
        )
    stage = _load_json(stage_manifest_path, "formal S22 stage manifest")
    _verify_stage_semantics(stage)

    run_path = (
        Path(run_manifest).expanduser().resolve()
        if run_manifest is not None
        else root.parent / "run_manifest.json"
    )
    expected_run_hash = _require_sha256(
        expected_run_manifest_sha256, name="expected formal run-manifest SHA-256"
    )
    observed_run_hash = _sha256(run_path)
    if observed_run_hash != expected_run_hash:
        raise RuntimeError(
            "Formal run-manifest SHA-256 mismatch: "
            f"expected {expected_run_hash}, observed {observed_run_hash}"
        )
    run = _load_json(run_path, "formal zebrafish run manifest")
    manifests = run.get("stage_manifests")
    if not isinstance(manifests, Mapping):
        raise RuntimeError("Formal run manifest lacks stage-manifest paths")
    declared_stage_manifest = Path(str(manifests.get("s22", "")))
    declared_stage_root = declared_stage_manifest.parent
    _verify_run_binding(
        run_manifest=run,
        stage_manifest=stage,
        declared_stage_manifest_path=declared_stage_manifest,
    )
    verified, recorded = _verify_stage_artifacts(
        stage_root=root,
        stage_manifest=stage,
        declared_stage_root=declared_stage_root,
    )

    global_index, generated = _load_state_index(
        stage_root=root,
        directory=GENERATED_DIR,
        expected_times=GLOBAL_TIMES,
        expected_sources=(
            "sampled_observed_t0_initial_condition",
            *("generated_global_t0_fixed_population_state_transport" for _ in range(8)),
        ),
        recorded=recorded,
    )
    observed_index, observed = _load_state_index(
        stage_root=root,
        directory=OBSERVED_DIR,
        expected_times=OBSERVED_TIMES,
        expected_sources=tuple("observed_reference_only" for _ in OBSERVED_TIMES),
        recorded=recorded,
    )
    details = stage["details"]
    if _sha256(root / GENERATED_DIR / "index.json") != details.get(
        "global_t0_fixed_population_state_index_sha256"
    ):
        raise RuntimeError("Generated-state index does not match S22 details")
    if _sha256(root / OBSERVED_DIR / "index.json") != details.get(
        "observed_reference_state_index_sha256"
    ):
        raise RuntimeError("Observed-reference index does not match S22 details")
    del global_index, observed_index
    if not np.array_equal(generated[0.0].points, observed[0.0].points):
        raise RuntimeError(
            "Generated path and observed references do not share exact t=0 points"
        )

    legend_key = LEGEND_RELATIVE_PATH.as_posix()
    support_key = SUPPORT_RELATIVE_PATH.as_posix()
    if legend_key not in recorded or support_key not in recorded:
        raise RuntimeError("S22 manifest does not record the legend/support provenance")
    colors = _parse_signed_legend(root / LEGEND_RELATIVE_PATH)
    selected = tuple(
        observed[time_value] if source == "observed" else generated[time_value]
        for source, time_value in PANELS
    )
    unknown = sorted(
        set()
        .union(*(set(frame.labels.tolist()) for frame in selected))
        .difference(colors)
    )
    if unknown:
        raise RuntimeError(f"Signed legend is missing plotted cell types: {unknown}")
    support = _load_json(root / SUPPORT_RELATIVE_PATH, "retained S22 support audit")
    if support != details.get("trajectory_support_audit"):
        raise RuntimeError("Retained S22 support audit disagrees with stage details")
    return VerifiedInputs(
        stage_root=root,
        stage_manifest_path=stage_manifest_path,
        stage_manifest_sha256=observed_stage_hash,
        stage_manifest=stage,
        run_manifest_path=run_path,
        run_manifest_sha256=observed_run_hash,
        run_manifest=run,
        frames=selected,
        colors=colors,
        verified_artifacts=verified,
        support_audit=support,
    )


def _configure_matplotlib() -> str:
    font_path = findfont(FontProperties(family="Arial"), fallback_to_default=False)
    matplotlib.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial"],
            "font.size": 9,
            "axes.titlesize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
        }
    )
    return font_path


def render_figure(inputs: VerifiedInputs, *, pdf_path: Path, png_path: Path) -> str:
    font_path = _configure_matplotlib()
    all_spatial = np.concatenate(
        [frame.points[:, :2] for frame in inputs.frames], axis=0
    )
    lower = np.min(all_spatial, axis=0)
    upper = np.max(all_spatial, axis=0)
    center = (lower + upper) / 2.0
    span = float(max(*(upper - lower), 1e-6)) * 1.06
    xlim = (center[0] - span / 2.0, center[0] + span / 2.0)
    ylim = (center[1] - span / 2.0, center[1] + span / 2.0)

    figure = plt.figure(figsize=(10.9, 8.15), constrained_layout=False)
    grid = figure.add_gridspec(
        3,
        4,
        width_ratios=(1.0, 1.0, 1.0, 1.08),
        left=0.035,
        right=0.985,
        bottom=0.045,
        top=0.975,
        wspace=0.08,
        hspace=0.12,
    )
    legend_order = list(inputs.colors)
    for panel_index, frame in enumerate(inputs.frames):
        axis = figure.add_subplot(grid[panel_index // 3, panel_index % 3])
        present = set(frame.labels.tolist())
        for label in legend_order:
            if label not in present:
                continue
            mask = frame.labels == label
            axis.scatter(
                frame.points[mask, 0],
                frame.points[mask, 1],
                s=3.0,
                c=inputs.colors[label],
                alpha=0.9,
                linewidths=0.0,
                rasterized=False,
            )
        axis.set_xlim(xlim)
        axis.set_ylim(ylim)
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(f"t = {frame.time:.1f}", pad=3.0, color="#24313A")
        source_label = (
            "Observed" if frame.source == "observed" else "Generated from t=0"
        )
        axis.text(
            0.025,
            0.975,
            source_label,
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=6.8,
            color="#24313A",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.4},
        )
        axis.set_axis_off()

    legend_axis = figure.add_subplot(grid[:, 3])
    legend_axis.set_axis_off()
    handles = [
        Line2D(
            [0],
            [0],
            linestyle="none",
            marker="o",
            markersize=4.3,
            markerfacecolor=inputs.colors[label],
            markeredgewidth=0.0,
            label=label,
        )
        for label in legend_order
    ]
    legend_axis.legend(
        handles=handles,
        labels=legend_order,
        title="Cell type",
        loc="center left",
        bbox_to_anchor=(0.0, 0.5),
        frameon=False,
        fontsize=6.0,
        title_fontsize=8.0,
        handletextpad=0.45,
        labelspacing=0.38,
        borderaxespad=0.0,
        ncol=1,
    )
    metadata = {
        "Title": "Zebrafish observed references and global-t0 generated half-time states",
        "Author": "CytoBridge",
        "Subject": "Corrected S22 article-style mixed-source display",
    }
    figure.savefig(pdf_path, format="pdf", metadata=metadata)
    figure.savefig(png_path, format="png", dpi=320, facecolor="white")
    plt.close(figure)
    return font_path


def _git_revision() -> Mapping[str, Any]:
    root = Path(__file__).resolve().parents[1]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return {"commit": commit, "dirty": dirty}
    except Exception:
        return {"commit": None, "dirty": None}


def _artifact(path: Path) -> Mapping[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def write_bundle(
    *,
    inputs: VerifiedInputs,
    output_dir: str | Path,
    argv: Sequence[str],
) -> Mapping[str, Any]:
    output = Path(output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory must be new or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    pdf_path = output / "S22_article_style_observed_generated_3x3.pdf"
    png_path = output / "S22_article_style_observed_generated_3x3.png"
    sources_path = output / "S22_article_style_panel_sources.csv"
    caption_path = output / "S22_article_style_caption.md"
    provenance_path = output / "S22_article_style_provenance.md"
    manifest_path = output / "figure_manifest.json"
    sidecar_path = output / "figure_manifest.sha256"

    font_path = render_figure(inputs, pdf_path=pdf_path, png_path=png_path)
    with sources_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "panel",
                "time",
                "display_source",
                "source_anchor_time",
                "n_cells",
                "input",
                "sha256",
            ),
        )
        writer.writeheader()
        for index, frame in enumerate(inputs.frames):
            writer.writerow(
                {
                    "panel": index + 1,
                    "time": f"{frame.time:.1f}",
                    "display_source": (
                        "observed_reference_only"
                        if frame.source == "observed"
                        else "generated_global_t0_fixed_population_state_transport"
                    ),
                    "source_anchor_time": ("" if frame.source == "observed" else "0.0"),
                    "n_cells": frame.points.shape[0],
                    "input": str(frame.path),
                    "sha256": frame.sha256,
                }
            )

    fixed_n = int(inputs.stage_manifest["details"]["fixed_particle_count"])
    caption = (
        "Observed zebrafish slices at integer times alternate with model-generated "
        "half-time states. The generated states at t = 0.5, 1.5, 2.5, and 3.5 "
        "come from one continuous simulation initialized once from the observed "
        "t = 0 cohort (fixed N = "
        f"{fixed_n}). They are not re-anchored to adjacent observed slices and no "
        "spatial display warp is applied. Integer-time panels are independent "
        "observed references, not generated reconstructions. This mixed-source "
        "display is therefore not an all-generated reconstruction or a cell-"
        "abundance forecast. The separate formal all-generated fixed-population "
        "S22 outputs and their full-horizon support audit remain unchanged.\n"
    )
    caption_path.write_text(caption, encoding="utf-8")

    rebuild = " ".join(shlex.quote(value) for value in argv)
    provenance = (
        "# S22 article-style figure provenance\n\n"
        "## Source paths\n\n"
        f"- Created (UTC): `{_utc_now()}`\n"
        f"- Formal S22 stage root: `{inputs.stage_root}`\n"
        f"- S22 stage manifest SHA-256: `{inputs.stage_manifest_sha256}`\n"
        f"- Formal run manifest SHA-256: `{inputs.run_manifest_sha256}`\n"
        f"- Formal run Git commit: `{inputs.run_manifest['common']['git']['commit']}`\n"
        f"- Canonical acceptance SHA-256: `{inputs.stage_manifest['canonical_matched_acceptance']['sha256']}`\n"
        f"- Plotting script SHA-256: `{_sha256(Path(__file__).resolve())}`\n"
        f"- Arial font file: `{font_path}`\n"
        f"- Vector PDF SHA-256: `{_sha256(pdf_path)}`\n"
        f"- PNG SHA-256: `{_sha256(png_path)}`\n"
        f"- Retained all-generated support-audit status: `{inputs.support_audit.get('status')}`\n\n"
        "## Scientific contract\n\n"
        "The four generated panels are unwarped half-time snapshots selected from "
        "the signed, single global-t0 fixed-population path. The five integer-time "
        "panels are separate observed references. No model simulation, state "
        "replacement, clipping, re-anchoring, or coordinate warp is performed by "
        "this renderer. The renderer does not modify or supersede the formal "
        "all-generated S22 support audit.\n\n"
        "## Exact rebuild command\n\n"
        f"```text\n{rebuild}\n```\n"
    )
    provenance_path.write_text(provenance, encoding="utf-8")

    created = [pdf_path, png_path, sources_path, caption_path, provenance_path]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "figure_id": FIGURE_ID,
        "created_at": _utc_now(),
        "plotter_git": _git_revision(),
        "plotter": _artifact(Path(__file__).resolve()),
        "formal_inputs": {
            "stage_root": str(inputs.stage_root),
            "stage_manifest": _artifact(inputs.stage_manifest_path),
            "stage_signature": inputs.stage_manifest["signature"],
            "run_manifest": _artifact(inputs.run_manifest_path),
            "run_signature": inputs.run_manifest["signature"],
            "canonical_matched_acceptance": inputs.stage_manifest[
                "canonical_matched_acceptance"
            ],
            "verified_stage_artifact_count": len(inputs.verified_artifacts),
            "verified_stage_artifacts": list(inputs.verified_artifacts),
        },
        "panel_contract": {
            "layout": "3x3",
            "sequence": [
                {"source": source, "time": time_value} for source, time_value in PANELS
            ],
            "generated_source_anchor_time": 0.0,
            "adjacent_observed_reanchoring": False,
            "display_warp": False,
            "all_generated_reconstruction": False,
            "renderer_ran_model": False,
            "formal_all_generated_s22_audit_modified": False,
        },
        "retained_all_generated_support_audit": inputs.support_audit,
        "font": {"family": "Arial", "path": font_path, "pdf_fonttype": 42},
        "outputs": [_artifact(path) for path in created],
        "rebuild_command": rebuild,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    sidecar_path.write_text(
        f"{_sha256(manifest_path)}  {manifest_path.name}\n", encoding="utf-8"
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage-root", required=True, help="Formal S22 stage directory"
    )
    parser.add_argument(
        "--expected-stage-manifest-sha256",
        required=True,
        help="Exact SHA-256 of <stage-root>/stage_manifest.json",
    )
    parser.add_argument(
        "--run-manifest",
        default=None,
        help="Formal run_manifest.json (default: sibling of stage root)",
    )
    parser.add_argument(
        "--expected-run-manifest-sha256",
        required=True,
        help="Exact SHA-256 of the formal run manifest",
    )
    parser.add_argument(
        "--output-dir", required=True, help="New or empty output directory"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    inputs = verify_inputs(
        stage_root=args.stage_root,
        expected_stage_manifest_sha256=args.expected_stage_manifest_sha256,
        run_manifest=args.run_manifest,
        expected_run_manifest_sha256=args.expected_run_manifest_sha256,
    )
    effective_argv = [
        sys.executable,
        str(Path(__file__).resolve()),
        *(argv if argv is not None else sys.argv[1:]),
    ]
    manifest = write_bundle(
        inputs=inputs, output_dir=args.output_dir, argv=effective_argv
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
