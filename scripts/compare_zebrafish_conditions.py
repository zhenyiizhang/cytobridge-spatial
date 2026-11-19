#!/usr/bin/env python3
"""Compare controlled CytoBridge conditions and optionally build a review bundle.

The numerical comparison is dataset-agnostic and consumes metric tables written
by ``save_distribution_evaluation``.  The optional review packaging follows the
zebrafish paper-stage contract and deliberately keeps numerical model selection
separate from biological interpretation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shlex
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_CONDITIONS = ("alpha_express_0015", "alpha_express_005")
PAPER_STAGES = (
    "classifier",
    "velocity",
    "s22",
    "growth",
    "ablation",
    "s25",
    "communication",
)
AUTO_SCORE_CRITERIA = (
    {
        "name": "mean_w1",
        "source": "w1",
        "direction": "lower",
        "description": "Mean Wasserstein-1 distance across paired time/space rows",
    },
    {
        "name": "mean_w2",
        "source": "w2",
        "direction": "lower",
        "description": "Mean Wasserstein-2 distance across paired time/space rows",
    },
    {
        "name": "mean_tmv",
        "source": "tmv",
        "direction": "lower",
        "description": "Mean total-mass-variation error across paired rows",
    },
    {
        "name": "mean_clump_fraction",
        "source": "clump_fraction_at_0_1_observed_nn",
        "direction": "lower",
        "description": "Mean fraction of generated points in very tight clumps",
    },
    {
        "name": "mean_nn_dispersion_log_error",
        "source": "nn_dispersion_ratio",
        "direction": "toward_one",
        "description": "Mean absolute log NN-dispersion ratio; zero is ideal",
    },
    {
        "name": "mean_support_recall",
        "source": "support_recall_at_observed_q95",
        "direction": "higher",
        "description": "Mean observed-support recall",
    },
    {
        "name": "mean_support_precision",
        "source": "support_precision_at_observed_q95",
        "direction": "higher",
        "description": "Mean generated-support precision",
    },
)


def _json_ready(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_json_ready(item) for item in value.tolist()]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _stable_hash(value: Mapping[str, object]) -> str:
    payload = json.dumps(
        _json_ready(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_file(path: Path, description: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing {description}: {path}")
    return path


def _parse_condition_paths(
    raw_values: Iterable[str], *, option_name: str
) -> dict[str, Path]:
    return _parse_named_paths(raw_values, option_name=option_name)


def _require_safe_component(value: str, *, description: str) -> str:
    value = str(value)
    if not value or Path(value).name != value or value in {".", ".."}:
        raise ValueError(
            f"{description} must be a single safe path component: {value!r}"
        )
    return value


def _parse_named_paths(
    raw_values: Iterable[str], *, option_name: str
) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for raw in raw_values:
        if "=" not in raw:
            raise ValueError(f"{option_name} values must use NAME=PATH: {raw!r}")
        name, raw_path = raw.split("=", 1)
        name = name.strip()
        if not name or not raw_path.strip():
            raise ValueError(f"Invalid {option_name} value: {raw!r}")
        _require_safe_component(name, description=f"{option_name} name")
        if name in result:
            raise ValueError(f"Duplicate {option_name} name: {name}")
        result[name] = Path(raw_path).expanduser().resolve()
    return result


def _default_manifest_path(run_root: Path, condition: str) -> Path:
    return run_root / "conditions" / condition / "downstream" / "run_manifest.json"


def _default_metrics_path(run_root: Path, condition: str) -> Path:
    return (
        run_root
        / "conditions"
        / condition
        / "downstream"
        / "distribution_evaluation"
        / "distribution_metrics.csv"
    )


def resolve_condition_inputs(
    run_root: Path,
    conditions: Sequence[str],
    *,
    metrics_overrides: Mapping[str, Path] | None = None,
    manifest_overrides: Mapping[str, Path] | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, dict], dict[str, dict[str, str]]]:
    """Load paired metric tables/manifests and record immutable input hashes."""
    run_root = run_root.expanduser().resolve()
    metrics_overrides = dict(metrics_overrides or {})
    manifest_overrides = dict(manifest_overrides or {})
    unknown = sorted(
        (set(metrics_overrides) | set(manifest_overrides)).difference(conditions)
    )
    if unknown:
        raise ValueError(f"Input overrides refer to unknown conditions: {unknown}")

    metrics_by_condition: dict[str, pd.DataFrame] = {}
    manifests: dict[str, dict] = {}
    input_records: dict[str, dict[str, str]] = {}
    for condition in conditions:
        manifest_path = _require_file(
            manifest_overrides.get(
                condition, _default_manifest_path(run_root, condition)
            ),
            f"run manifest for {condition}",
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_condition = manifest.get("condition")
        if manifest_condition is not None and str(manifest_condition) != condition:
            raise ValueError(
                f"Manifest condition mismatch for {condition}: {manifest_condition!r}"
            )

        metrics_path = metrics_overrides.get(
            condition, _default_metrics_path(run_root, condition)
        )
        if not metrics_path.is_file():
            recorded_path = (
                manifest.get("distribution_evaluation", {})
                .get("paths", {})
                .get("metrics")
            )
            if recorded_path:
                metrics_path = Path(recorded_path).expanduser()
                if not metrics_path.is_absolute():
                    metrics_path = manifest_path.parent / metrics_path
        metrics_path = _require_file(metrics_path, f"distribution metrics for {condition}")

        metrics_by_condition[condition] = pd.read_csv(metrics_path)
        manifests[condition] = manifest
        input_records[condition] = {
            "metrics_path": str(metrics_path),
            "metrics_sha256": _sha256(metrics_path),
            "manifest_path": str(manifest_path),
            "manifest_sha256": _sha256(manifest_path),
        }
    return metrics_by_condition, manifests, input_records


def _validate_auto_score_columns(metrics_long: pd.DataFrame) -> None:
    required = {str(item["source"]) for item in AUTO_SCORE_CRITERIA}
    missing = sorted(required.difference(metrics_long.columns))
    if missing:
        raise KeyError(
            "Automatic model selection requires distribution and local-structure "
            f"metrics; missing columns: {missing}. An explicit winner may override "
            "the score result, but does not make an incomplete comparison auditable."
        )
    numeric = metrics_long[list(required)].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Automatic-score metric columns must all be finite.")
    ratio = numeric["nn_dispersion_ratio"].to_numpy(dtype=float)
    if np.any(ratio <= 0):
        raise ValueError("nn_dispersion_ratio must be positive for log-distance scoring.")
    for column in (
        "support_recall_at_observed_q95",
        "support_precision_at_observed_q95",
        "clump_fraction_at_0_1_observed_nn",
    ):
        values = numeric[column].to_numpy(dtype=float)
        if np.any((values < -1e-9) | (values > 1.0 + 1e-9)):
            raise ValueError(f"{column} must lie in [0, 1].")


def score_conditions(
    metrics_long: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, str, bool]:
    """Build an equal-criterion, rank-based diagnostic score.

    Each of the seven declared criteria contributes equally.  Rank utilities
    prevent W1's numeric scale from dominating small-valued diagnostics.  This
    is a reproducible screening rule, not evidence of biological correctness.
    """
    _validate_auto_score_columns(metrics_long)
    models = list(dict.fromkeys(metrics_long["model"].astype(str)))
    if len(models) < 2:
        raise ValueError("At least two conditions are required for ranking.")

    rows: list[dict[str, float | str]] = []
    for model in models:
        table = metrics_long.loc[metrics_long["model"].astype(str) == model]
        row: dict[str, float | str] = {"condition": model}
        for criterion in AUTO_SCORE_CRITERIA:
            source = str(criterion["source"])
            values = pd.to_numeric(table[source], errors="raise").to_numpy(dtype=float)
            if criterion["direction"] == "toward_one":
                aggregate = float(np.mean(np.abs(np.log(values))))
            else:
                aggregate = float(np.mean(values))
            row[str(criterion["name"])] = aggregate
        rows.append(row)
    ranking = pd.DataFrame(rows)

    criterion_rows: list[dict[str, float | str]] = []
    weight = 1.0 / len(AUTO_SCORE_CRITERIA)
    total_score = np.zeros(len(ranking), dtype=float)
    n_models = len(ranking)
    for criterion in AUTO_SCORE_CRITERIA:
        name = str(criterion["name"])
        direction = str(criterion["direction"])
        ascending = direction in {"lower", "toward_one"}
        ranks = ranking[name].rank(method="average", ascending=ascending)
        utilities = (
            np.full(n_models, 0.5, dtype=float)
            if n_models == 1
            else (n_models - ranks.to_numpy(dtype=float)) / (n_models - 1)
        )
        total_score += weight * utilities
        ranking[f"{name}_utility"] = utilities
        for index, condition in enumerate(ranking["condition"].astype(str)):
            criterion_rows.append(
                {
                    "condition": condition,
                    "criterion": name,
                    "source_column": str(criterion["source"]),
                    "direction": direction,
                    "aggregate_value": float(ranking.loc[index, name]),
                    "rank_utility": float(utilities[index]),
                    "weight": weight,
                    "weighted_utility": float(weight * utilities[index]),
                    "description": str(criterion["description"]),
                }
            )
    ranking["auto_score"] = total_score
    ranking = ranking.sort_values(
        ["auto_score", "condition"], ascending=[False, True]
    ).reset_index(drop=True)
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1, dtype=int))
    criterion_scores = pd.DataFrame(criterion_rows).sort_values(
        ["criterion", "condition"]
    ).reset_index(drop=True)

    best_score = float(ranking.loc[0, "auto_score"])
    tied = bool(
        np.sum(np.isclose(ranking["auto_score"], best_score, rtol=1e-12, atol=1e-12))
        > 1
    )
    auto_winner = str(ranking.loc[0, "condition"])
    return ranking, criterion_scores, auto_winner, tied


def _comparison_warnings(manifests: Mapping[str, Mapping]) -> list[str]:
    warnings: list[str] = []
    if not manifests:
        return warnings
    fields = ("input_h5ad_sha256", "aligned_h5ad_sha256")
    for field in fields:
        missing = [name for name, manifest in manifests.items() if not manifest.get(field)]
        if missing:
            warnings.append(
                f"Conditions missing recorded {field}: {sorted(missing)}; exact input "
                "pairing cannot be verified."
            )
        values = {
            str(manifest.get(field))
            for manifest in manifests.values()
            if manifest.get(field) is not None
        }
        if len(values) > 1:
            warnings.append(
                f"Conditions do not share the same {field}; interpret paired deltas cautiously."
            )
    raw_settings = {
        name: manifest.get("distribution_evaluation", {}).get("settings")
        for name, manifest in manifests.items()
    }
    missing_settings = [name for name, value in raw_settings.items() if value is None]
    if missing_settings:
        warnings.append(
            "Conditions missing recorded distribution-evaluation settings: "
            f"{sorted(missing_settings)}."
        )
    settings = {
        json.dumps(
            _json_ready(value),
            sort_keys=True,
        )
        for value in raw_settings.values()
    }
    if len(settings) > 1:
        warnings.append(
            "Distribution-evaluation settings differ between conditions; the metric grid "
            "is paired but the evaluation protocol is not identical."
        )
    return warnings


def _prepare_output_dir(path: Path, *, overwrite: bool) -> Path:
    path = path.expanduser().resolve()
    if path.exists() and not path.is_dir():
        raise NotADirectoryError(path)
    if path.is_dir() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {path}. Choose a new directory or pass "
            "--overwrite-existing-output explicitly."
        )
    path.mkdir(parents=True, exist_ok=True)
    return path


def _replace_destination(destination: Path, *, overwrite: bool) -> None:
    if not destination.exists() and not destination.is_symlink():
        return
    if not overwrite:
        raise FileExistsError(f"Refusing to overwrite bundle entry: {destination}")
    if destination.is_symlink() or destination.is_file():
        destination.unlink()
    else:
        shutil.rmtree(destination)


def _transfer(source: Path, destination: Path, *, mode: str, overwrite: bool) -> None:
    source = source.expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _replace_destination(destination, overwrite=overwrite)
    if mode == "symlink":
        destination.symlink_to(source, target_is_directory=source.is_dir())
    elif mode == "copy":
        if source.is_dir():
            shutil.copytree(source, destination)
        else:
            shutil.copy2(source, destination)
    else:
        raise ValueError(f"Unsupported transfer mode: {mode}")


def _discover_selected_panels(run_root: Path, condition: str) -> Path | None:
    condition_dir = run_root / "conditions" / condition
    candidates = (
        condition_dir / "paper_downstream" / "selected_manuscript_panels",
        condition_dir / "paper_downstream" / "02_selected_manuscript_panels",
        condition_dir / "downstream" / "selected_manuscript_panels",
        condition_dir / "downstream" / "02_selected_manuscript_panels",
        condition_dir / "manuscript_panels",
        # The native paper adapter writes stage directories (s22, velocity,
        # growth, ablation, s25, communication) below this root.  In that case
        # the bundler selects visual artifacts recursively instead of copying
        # classifier caches or other bulky intermediates.
        condition_dir / "paper_downstream",
    )
    return next((path.resolve() for path in candidates if path.is_dir()), None)


def _is_paper_output_root(path: Path) -> bool:
    """Return whether *path* follows the native seven-stage paper contract."""
    return (path / "run_manifest.json").is_file() and all(
        (path / stage).is_dir() for stage in PAPER_STAGES
    )


def _validated_paper_visual_artifacts(source_root: Path) -> list[Path]:
    """Validate a full paper run and return its manifest-recorded visuals.

    This deliberately avoids recursive suffix discovery: only outputs recorded
    by the seven stage manifests can enter a review bundle.  Every recorded
    artifact is size/hash checked first, including non-visual state files that
    remain outside the portable panel bundle.
    """
    source_root = source_root.expanduser().resolve()
    declared_root_manifest = source_root / "run_manifest.json"
    if declared_root_manifest.is_symlink():
        raise ValueError(f"Paper root manifest must not be a symlink: {declared_root_manifest}")
    root_path = _require_file(
        declared_root_manifest, "paper downstream root manifest"
    )
    root = json.loads(root_path.read_text(encoding="utf-8"))
    if root.get("workflow") != "zebrafish_native_paper_downstream":
        raise ValueError(
            f"Unexpected paper workflow in {root_path}: {root.get('workflow')!r}"
        )
    if root.get("profile") != "full":
        raise ValueError(
            f"Review bundles require profile='full', got {root.get('profile')!r} "
            f"in {root_path}."
        )
    if not isinstance(root.get("common"), Mapping):
        raise ValueError(f"Paper root lacks common signature mapping: {root_path}")
    completed = {str(value) for value in root.get("completed_stages", [])}
    missing_stages = sorted(set(PAPER_STAGES).difference(completed))
    if missing_stages:
        raise ValueError(
            f"Paper root is missing completed stages {missing_stages}: {root_path}"
        )

    recorded_stage_manifests = root.get("stage_manifests")
    if not isinstance(recorded_stage_manifests, Mapping):
        raise ValueError(f"Paper root lacks stage_manifests mapping: {root_path}")
    recorded_stage_signatures = root.get("stage_signatures")
    if not isinstance(recorded_stage_signatures, Mapping):
        raise ValueError(f"Paper root lacks stage_signatures mapping: {root_path}")

    visual_suffixes = {".pdf", ".png", ".svg", ".gif", ".mp4", ".html"}
    visual_paths: list[Path] = []
    seen: set[Path] = set()
    for stage in PAPER_STAGES:
        stage_path = _require_file(
            source_root / stage / "stage_manifest.json",
            f"{stage} stage manifest",
        )
        if (source_root / stage / "stage_manifest.json").is_symlink():
            raise ValueError(f"Stage manifest must not be a symlink: {stage_path}")
        recorded_stage_path = Path(str(recorded_stage_manifests.get(stage, "")))
        if not recorded_stage_path.is_absolute():
            recorded_stage_path = source_root / recorded_stage_path
        if recorded_stage_path.expanduser().resolve() != stage_path:
            raise ValueError(
                f"Root manifest points {stage!r} to {recorded_stage_path}, "
                f"not {stage_path}."
            )
        manifest = json.loads(stage_path.read_text(encoding="utf-8"))
        if str(manifest.get("stage")) != stage:
            raise ValueError(
                f"Stage manifest identity mismatch for {stage}: "
                f"{manifest.get('stage')!r}."
            )
        if manifest.get("status") != "complete":
            raise ValueError(
                f"Stage {stage!r} is not complete in {stage_path}: "
                f"{manifest.get('status')!r}."
            )
        signature = str(manifest.get("signature", ""))
        if not signature or signature != str(recorded_stage_signatures.get(stage, "")):
            raise ValueError(
                f"Stage signature mismatch between {stage_path} and {root_path}."
            )
        settings = manifest.get("settings")
        if not isinstance(settings, Mapping):
            raise ValueError(f"Stage settings must be a mapping in {stage_path}.")
        expected_signature = _stable_hash(
            {"stage": stage, "common": root.get("common"), "settings": settings}
        )
        if signature != expected_signature:
            raise ValueError(
                f"Stage signature cannot be reproduced from root common/settings: "
                f"{stage_path}."
            )
        artifacts = manifest.get("output_artifacts")
        if not isinstance(artifacts, list) or not artifacts:
            raise ValueError(f"{stage_path} lacks non-empty output_artifacts list.")
        output_strings = {str(value) for value in manifest.get("outputs", [])}
        artifact_strings: set[str] = set()
        for record in artifacts:
            if not isinstance(record, Mapping):
                raise ValueError(f"Invalid artifact record in {stage_path}: {record!r}")
            artifact_strings.add(str(record.get("path")))
        if len(output_strings) != len(manifest.get("outputs", [])) or len(
            artifact_strings
        ) != len(artifacts):
            raise ValueError(f"Duplicate outputs are recorded in {stage_path}.")
        if output_strings != artifact_strings:
            raise ValueError(
                f"outputs/output_artifacts disagree in {stage_path}."
            )
        for record in artifacts:
            artifact = _require_file(
                Path(str(record.get("path", ""))),
                f"recorded {stage} output",
            )
            expected_size = int(record.get("size_bytes", -1))
            if expected_size <= 0 or artifact.stat().st_size != expected_size:
                raise ValueError(
                    f"Recorded size mismatch for {artifact}: expected "
                    f"{expected_size}, got {artifact.stat().st_size}."
                )
            expected_sha = str(record.get("sha256", ""))
            actual_sha = _sha256(artifact)
            if actual_sha != expected_sha:
                raise ValueError(
                    f"Recorded SHA-256 mismatch for {artifact}: expected "
                    f"{expected_sha}, got {actual_sha}."
                )
            if artifact.suffix.lower() in visual_suffixes:
                if not artifact.is_relative_to(source_root):
                    raise ValueError(
                        f"Manifest-recorded visual lies outside paper root: {artifact}"
                    )
                if artifact not in seen:
                    visual_paths.append(artifact)
                    seen.add(artifact)
    return sorted(visual_paths)


def _validate_paper_condition_provenance(
    paper_root: Path,
    *,
    condition: str,
    condition_manifest: Mapping[str, object],
) -> None:
    """Prevent pairing one condition's panels with another model/data manifest."""
    paper_root = paper_root.expanduser().resolve()
    root_path = _require_file(
        paper_root / "run_manifest.json", "paper downstream root manifest"
    )
    paper = json.loads(root_path.read_text(encoding="utf-8"))
    if str(condition_manifest.get("condition")) != condition:
        raise ValueError(
            f"Quantitative manifest condition mismatch for {condition}: "
            f"{condition_manifest.get('condition')!r}."
        )
    common = paper.get("common")
    model = paper.get("model")
    if not isinstance(common, Mapping) or not isinstance(model, Mapping):
        raise ValueError(f"Paper root lacks common/model provenance: {root_path}")

    hash_checks = (
        (
            "aligned_h5ad_sha256",
            condition_manifest.get("aligned_h5ad_sha256"),
            common.get("aligned_h5ad_sha256"),
        ),
        (
            "weight_checkpoint_sha256",
            condition_manifest.get("weight_checkpoint_sha256"),
            common.get("weight_sha256"),
        ),
        (
            "score_checkpoint_sha256",
            condition_manifest.get("score_checkpoint_sha256"),
            common.get("score_sha256"),
        ),
    )
    for name, quantitative_value, paper_value in hash_checks:
        if not quantitative_value or not paper_value:
            raise ValueError(
                f"Missing {name} while matching {condition} to {paper_root}."
            )
        if str(quantitative_value) != str(paper_value):
            raise ValueError(
                f"{name} mismatch for {condition}: quantitative="
                f"{quantitative_value}, paper={paper_value}."
            )

    for name in ("alpha_spatial", "alpha_express"):
        quantitative_value = condition_manifest.get(name)
        paper_value = model.get(name)
        if quantitative_value is None or paper_value is None or not math.isclose(
            float(quantitative_value),
            float(paper_value),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"{name} mismatch for {condition}: quantitative="
                f"{quantitative_value}, paper={paper_value}."
            )

    quantitative_training = condition_manifest.get("training_dir")
    paper_model_dir = common.get("model_dir")
    if not quantitative_training or not paper_model_dir:
        raise ValueError(
            f"Missing training/model directory provenance for {condition}."
        )
    if Path(str(quantitative_training)).expanduser().resolve() != Path(
        str(paper_model_dir)
    ).expanduser().resolve():
        raise ValueError(
            f"Training directory mismatch for {condition}: "
            f"{quantitative_training} != {paper_model_dir}."
        )


def _transfer_panel_artifacts(
    source_root: Path,
    destination_root: Path,
    *,
    mode: str,
    overwrite: bool,
    sources: Sequence[Path] | None = None,
) -> int:
    source_root = source_root.expanduser().resolve()
    visual_suffixes = {".pdf", ".png", ".svg", ".gif", ".mp4", ".html"}
    selected_sources = (
        [Path(path).expanduser().resolve() for path in sources]
        if sources is not None
        else [
            path
            for path in sorted(source_root.rglob("*"))
            if path.is_file() and path.suffix.lower() in visual_suffixes
        ]
    )
    if not selected_sources:
        return 0
    if destination_root.exists() and not destination_root.is_dir():
        _replace_destination(destination_root, overwrite=overwrite)
    destination_root.mkdir(parents=True, exist_ok=True)
    for source in selected_sources:
        if not source.is_relative_to(source_root):
            raise ValueError(f"Panel artifact lies outside source root: {source}")
        _transfer(
            source,
            destination_root / source.relative_to(source_root),
            mode=mode,
            overwrite=overwrite,
        )
    return len(selected_sources)


def _transfer_panel_source(
    source_root: Path,
    destination_root: Path,
    *,
    mode: str,
    overwrite: bool,
) -> int:
    """Transfer one condition's panel source while excluding bulky caches."""
    if _is_paper_output_root(source_root):
        recorded_visuals = _validated_paper_visual_artifacts(source_root)
        return _transfer_panel_artifacts(
            source_root,
            destination_root,
            mode=mode,
            overwrite=overwrite,
            sources=recorded_visuals,
        )
    return _transfer_panel_artifacts(
        source_root,
        destination_root,
        mode=mode,
        overwrite=overwrite,
    )


def build_review_bundle(
    *,
    bundle_root: Path,
    run_root: Path,
    conditions: Sequence[str],
    selected_condition: str,
    input_records: Mapping[str, Mapping[str, str]],
    mode: str,
    panel_overrides: Mapping[str, Path] | None = None,
    paper_output_overrides: Mapping[str, Path] | None = None,
    canonical_logs: Mapping[str, Path] | None = None,
    overwrite: bool = False,
    allow_missing_panels: bool = False,
) -> tuple[dict[str, str], list[str]]:
    """Transfer paired panels, manifests, configs, and explicit canonical logs."""
    panel_overrides = dict(panel_overrides or {})
    paper_output_overrides = dict(paper_output_overrides or {})
    canonical_logs = dict(canonical_logs or {})
    unknown = sorted(
        (set(panel_overrides) | set(paper_output_overrides)).difference(conditions)
    )
    if unknown:
        raise ValueError(
            f"Panel/paper-output overrides refer to unknown conditions: {unknown}"
        )

    bundle_paths: dict[str, str] = {}
    warnings: list[str] = []
    condition_manifests = {
        condition: json.loads(
            Path(str(input_records[condition]["manifest_path"])).read_text(
                encoding="utf-8"
            )
        )
        for condition in conditions
    }
    panel_sources: dict[str, Path] = {}
    panel_destinations: dict[str, Path] = {}
    validated_paper_roots: set[Path] = set()
    for condition in conditions:
        panels = panel_overrides.get(condition)
        if panels is None:
            panels = _discover_selected_panels(run_root, condition)
        if panels is None or not panels.is_dir():
            message = (
                f"No manuscript panel directory found for {condition}; pass "
                f"--selected-panels {condition}=PATH."
            )
            if not allow_missing_panels:
                raise FileNotFoundError(message)
            warnings.append(message)
            continue
        panels = panels.resolve()
        if _is_paper_output_root(panels):
            _validate_paper_condition_provenance(
                panels,
                condition=condition,
                condition_manifest=condition_manifests[condition],
            )
        destination = bundle_root / "02_condition_panels" / condition
        artifact_count = _transfer_panel_source(
            panels,
            destination,
            mode=mode,
            overwrite=overwrite,
        )
        if artifact_count == 0:
            message = f"No visual panel artifacts found below {panels}."
            if not allow_missing_panels:
                raise FileNotFoundError(message)
            warnings.append(message)
            continue
        panel_sources[condition] = panels
        panel_destinations[condition] = destination
        if _is_paper_output_root(panels):
            validated_paper_roots.add(panels)
        bundle_paths[f"{condition}_manuscript_panels"] = str(destination)
        if _is_paper_output_root(panels):
            warnings.append(
                f"Collected {artifact_count} manifest-recorded, hash-verified visual "
                f"artifacts for {condition} from {panels}."
            )
        else:
            warnings.append(
                f"Collected {artifact_count} visual artifacts recursively for "
                f"{condition} from a non-native panel directory {panels}; review "
                "this superset before manuscript use."
            )

    # Backward-compatible entry containing the selected condition only.  The
    # paired sources above remain the authoritative side-by-side comparison.
    selected_panels = panel_destinations.get(selected_condition)
    if selected_panels is not None:
        selected_destination = bundle_root / "02_selected_manuscript_panels"
        _transfer(
            selected_panels,
            selected_destination,
            mode=mode,
            overwrite=overwrite,
        )
        bundle_paths["selected_manuscript_panels"] = str(selected_destination)

    for condition in conditions:
        condition_dest = bundle_root / "03_condition_inputs" / condition
        for key, output_name in (
            ("metrics_path", "distribution_metrics.csv"),
            ("manifest_path", "run_manifest.json"),
        ):
            source = Path(str(input_records[condition][key]))
            destination = condition_dest / output_name
            _transfer(source, destination, mode=mode, overwrite=overwrite)
            bundle_paths[f"{condition}_{key}"] = str(destination)

        training_dir = run_root / "conditions" / condition / "training"
        for filename in ("config.yaml", "launch_manifest.json"):
            source = training_dir / filename
            if source.is_file():
                destination = condition_dest / filename
                _transfer(source, destination, mode=mode, overwrite=overwrite)
                bundle_paths[f"{condition}_{filename}"] = str(destination)
            else:
                warnings.append(f"Optional training audit file is missing: {source}")

        paper_root = paper_output_overrides.get(condition)
        if paper_root is None:
            panel_source = panel_sources.get(condition)
            if panel_source is not None and _is_paper_output_root(panel_source):
                paper_root = panel_source
            else:
                paper_root = (
                    run_root / "conditions" / condition / "paper_downstream"
                )
        paper_root = paper_root.expanduser().resolve()
        if not _is_paper_output_root(paper_root):
            raise FileNotFoundError(
                "Paper output must contain run_manifest.json and all seven stage "
                f"directories for {condition}: {paper_root}"
            )
        panel_source = panel_sources.get(condition)
        if (
            panel_source is not None
            and _is_paper_output_root(panel_source)
            and panel_source != paper_root
        ):
            raise ValueError(
                f"Native panel source and paper-output provenance root differ for "
                f"{condition}: {panel_source} != {paper_root}."
            )
        if paper_root not in validated_paper_roots:
            _validate_paper_condition_provenance(
                paper_root,
                condition=condition,
                condition_manifest=condition_manifests[condition],
            )
            _validated_paper_visual_artifacts(paper_root)
            validated_paper_roots.add(paper_root)
        root_manifest = _require_file(
            paper_root / "run_manifest.json",
            f"paper downstream root manifest for {condition}",
        )
        destination = condition_dest / "paper_downstream" / "run_manifest.json"
        _transfer(root_manifest, destination, mode=mode, overwrite=overwrite)
        bundle_paths[f"{condition}_paper_run_manifest"] = str(destination)
        for stage in PAPER_STAGES:
            stage_manifest = _require_file(
                paper_root / stage / "stage_manifest.json",
                f"{stage} stage manifest for {condition}",
            )
            destination = (
                condition_dest
                / "paper_downstream"
                / stage
                / "stage_manifest.json"
            )
            _transfer(stage_manifest, destination, mode=mode, overwrite=overwrite)
            bundle_paths[f"{condition}_{stage}_stage_manifest"] = str(destination)

    if canonical_logs:
        for bundle_name, raw_source in sorted(canonical_logs.items()):
            if Path(bundle_name).name != bundle_name or bundle_name in {".", ".."}:
                raise ValueError(
                    "Canonical log names must be single safe path components: "
                    f"{bundle_name!r}"
                )
            source = _require_file(raw_source, f"canonical log {bundle_name}")
            destination = bundle_root / "04_logs" / bundle_name
            _transfer(source, destination, mode=mode, overwrite=overwrite)
            bundle_paths[f"canonical_log_{bundle_name}"] = str(destination)
        bundle_paths["logs"] = str(bundle_root / "04_logs")
    else:
        warnings.append(
            "No explicit canonical logs were supplied; 04_logs was intentionally "
            "omitted instead of copying the run's mixed log directory."
        )
    return bundle_paths, warnings


def _render_reproduction_commands(
    *,
    run_root: Path,
    output_dir: Path,
    conditions: Sequence[str],
    baseline: str,
    winner: str,
    metrics_overrides: Mapping[str, Path],
    manifest_overrides: Mapping[str, Path],
    bundle_mode: str,
    panel_overrides: Mapping[str, Path],
    paper_output_overrides: Mapping[str, Path],
    canonical_logs: Mapping[str, Path],
    allow_missing_panels: bool,
) -> str:
    """Render the exact comparison inputs while enforcing a fresh destination."""
    output_placeholder = "__CYTOBRIDGE_NEW_BUNDLE_DIR__"
    reproduced_dir = output_dir.parent / f"{output_dir.name}_reproduced"
    args = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--run-root",
        str(run_root),
        "--output-dir",
        output_placeholder,
        "--conditions",
        *[str(condition) for condition in conditions],
        "--baseline",
        str(baseline),
        "--winner",
        str(winner),
        "--bundle-mode",
        str(bundle_mode),
    ]
    for condition, path in sorted(metrics_overrides.items()):
        args.extend(["--metrics-path", f"{condition}={path}"])
    for condition, path in sorted(manifest_overrides.items()):
        args.extend(["--manifest-path", f"{condition}={path}"])
    for condition, path in sorted(panel_overrides.items()):
        args.extend(["--selected-panels", f"{condition}={path}"])
    for condition, path in sorted(paper_output_overrides.items()):
        args.extend(["--paper-output", f"{condition}={path}"])
    for name, path in sorted(canonical_logs.items()):
        args.extend(["--canonical-log", f"{name}={path}"])
    if allow_missing_panels:
        args.append("--allow-missing-panels")
    command = shlex.join(args).replace(
        output_placeholder,
        '"$NEW_BUNDLE_DIR"',
    )
    return "\n".join(
        (
            "# Rebuild the comparison and portable review bundle from the same inputs.",
            "# The destination must not already contain a prior bundle.",
            f"NEW_BUNDLE_DIR={shlex.quote(str(reproduced_dir))}",
            'test ! -e "$NEW_BUNDLE_DIR"',
            command,
            "",
        )
    )


def _write_review_readme(
    path: Path,
    *,
    conditions: Sequence[str],
    baseline: str,
    selected_condition: str,
    selection_reason: str,
    requested_winner: str,
    ranking: pd.DataFrame,
    manifests: Mapping[str, Mapping],
    canonical_logs: Mapping[str, Path],
    warnings: Sequence[str],
) -> Path:
    score_by_condition = {
        str(row["condition"]): float(row["auto_score"])
        for _, row in ranking.iterrows()
    }
    lines = [
        "# Zebrafish clean-counts condition review bundle",
        "",
        "This is a portable, paired **review** bundle. It preserves both condition "
        "panel sets; the selected-panel directory is a backward-compatible convenience "
        "copy, not a replacement for side-by-side biological review. It is not a "
        "standalone recomputation archive: model checkpoints, input H5AD, environment, "
        "and the full code snapshot are delivered separately, and source manifests keep "
        "their original absolute audit paths.",
        "",
        "## Selection",
        "",
        f"- Baseline: `{baseline}`",
        f"- Requested winner policy: `{requested_winner}`",
        f"- Selected condition: `{selected_condition}`",
        f"- Selection reason: `{selection_reason}`",
        "- Automatic scores summarize numerical distribution diagnostics only; they do "
        "  not establish biological validity or manuscript readiness.",
        "",
        "| Condition | alpha_spatial | alpha_express | auto score |",
        "|---|---:|---:|---:|",
    ]
    for condition in conditions:
        manifest = manifests[condition]
        lines.append(
            "| "
            f"`{condition}` | {manifest.get('alpha_spatial')} | "
            f"{manifest.get('alpha_express')} | "
            f"{score_by_condition[condition]:.6f} |"
        )
    lines.extend(
        (
            "",
            "## Directory guide",
            "",
            "- `01_condition_comparison/`: paired W1/W2/TMV and local-structure "
            "tables, figures, ranking, criteria, and selection manifest.",
            "- `02_condition_panels/<condition>/`: visual artifacts for every compared "
            "condition, preserving paper-stage subdirectories.",
            "- `02_selected_manuscript_panels/`: compatibility copy of the selected "
            "condition's visual artifacts.",
            "- `03_condition_inputs/<condition>/`: distribution metrics, quantitative "
            "manifest, resolved training config/launch manifest, paper root manifest, "
            "and all seven paper stage manifests.",
            "- `04_logs/`: only logs explicitly declared canonical at bundle creation.",
            "- `05_provenance/reproduction_commands.txt`: exact bundle command with a "
            "fresh destination guard.",
            "- `05_provenance/artifact_inventory.csv`: relative path, byte size, and "
            "SHA-256 for every bundled file except the inventory itself.",
            "",
            "## Canonical logs",
            "",
        )
    )
    if canonical_logs:
        for name, source in sorted(canonical_logs.items()):
            lines.append(f"- `{name}` from `{source}`")
    else:
        lines.append("- None supplied; mixed historical logs were intentionally omitted.")
    lines.extend(("", "## Audit warnings", ""))
    if warnings:
        for warning in warnings:
            lines.append(f"- {warning}")
    else:
        lines.append("- None.")
    lines.extend(
        (
            "",
            "The expected paper stages are: "
            + ", ".join(f"`{stage}`" for stage in PAPER_STAGES)
            + ".",
            "",
        )
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _write_artifact_inventory(bundle_root: Path, path: Path) -> Path:
    """Hash every regular bundle file except this self-referential inventory."""
    rows = []
    for artifact in sorted(bundle_root.rglob("*")):
        if not artifact.is_file() or artifact.resolve() == path.resolve():
            continue
        rows.append(
            {
                "relative_path": artifact.relative_to(bundle_root).as_posix(),
                "size_bytes": int(artifact.stat().st_size),
                "sha256": _sha256(artifact),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["relative_path", "size_bytes", "sha256"]).to_csv(
        path,
        index=False,
    )
    return path


def run_comparison(
    *,
    run_root: Path,
    output_dir: Path,
    conditions: Sequence[str],
    baseline: str,
    winner: str = "auto",
    metrics_overrides: Mapping[str, Path] | None = None,
    manifest_overrides: Mapping[str, Path] | None = None,
    bundle_mode: str = "none",
    panel_overrides: Mapping[str, Path] | None = None,
    paper_output_overrides: Mapping[str, Path] | None = None,
    canonical_logs: Mapping[str, Path] | None = None,
    allow_missing_panels: bool = False,
    overwrite: bool = False,
) -> dict:
    """Execute comparison, transparent scoring, selection, and optional packaging."""
    # Keep package/plot imports lazy so ``--help`` remains lightweight on login
    # nodes while still using the shared dataset-agnostic evaluation API.
    from CytoBridge.tl import (
        compare_distribution_metric_tables,
        save_distribution_metric_comparison,
    )

    conditions = tuple(str(value) for value in conditions)
    if len(conditions) < 2 or len(set(conditions)) != len(conditions):
        raise ValueError("Provide at least two distinct conditions.")
    for condition in conditions:
        _require_safe_component(condition, description="Condition")
    if baseline not in conditions:
        raise ValueError(f"Baseline {baseline!r} is not one of {conditions}.")
    if winner != "auto" and winner not in conditions:
        raise ValueError(f"Winner {winner!r} is not one of {conditions}.")
    if bundle_mode not in {"none", "symlink", "copy"}:
        raise ValueError("bundle_mode must be none, symlink, or copy.")

    run_root = run_root.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    metrics_overrides = dict(metrics_overrides or {})
    manifest_overrides = dict(manifest_overrides or {})
    panel_overrides = dict(panel_overrides or {})
    paper_output_overrides = dict(paper_output_overrides or {})
    canonical_logs = dict(canonical_logs or {})
    if bundle_mode != "none" and output_dir.is_dir() and any(output_dir.iterdir()):
        raise FileExistsError(
            "Review bundles require a new empty output directory; refusing to reuse "
            f"{output_dir} even with overwrite enabled."
        )
    output_dir = _prepare_output_dir(output_dir, overwrite=overwrite)
    comparison_dir = (
        output_dir / "01_condition_comparison"
        if bundle_mode != "none"
        else output_dir
    )
    if comparison_dir != output_dir:
        comparison_dir.mkdir(parents=True, exist_ok=True)

    tables, manifests, input_records = resolve_condition_inputs(
        run_root,
        conditions,
        metrics_overrides=metrics_overrides,
        manifest_overrides=manifest_overrides,
    )
    comparison = compare_distribution_metric_tables(tables, baseline=baseline)
    comparison_paths = save_distribution_metric_comparison(
        comparison, comparison_dir
    )
    ranking, criterion_scores, auto_winner, tied = score_conditions(
        comparison.metrics
    )
    ranking_path = comparison_dir / "condition_ranking.csv"
    criterion_scores_path = comparison_dir / "selection_criteria_long.csv"
    ranking.to_csv(ranking_path, index=False)
    criterion_scores.to_csv(criterion_scores_path, index=False)

    pairing_warnings = _comparison_warnings(manifests)
    warnings = list(pairing_warnings)
    if bundle_mode == "symlink":
        warnings.append(
            "The review bundle uses symlinks and is not portable; use --bundle-mode "
            "copy for download or archival."
        )
    if winner == "auto":
        if tied:
            selected_condition = baseline
            selection_reason = (
                "auto_score_tie_conservative_baseline; explicit panel review recommended"
            )
            warnings.append(
                "Automatic scores are tied; the baseline is retained conservatively."
            )
        else:
            selected_condition = auto_winner
            selection_reason = "highest_equal_criterion_rank_score"
    else:
        selected_condition = winner
        selection_reason = "explicit_user_override"

    bundle_paths: dict[str, str] = {}
    if bundle_mode != "none":
        bundle_paths, bundle_warnings = build_review_bundle(
            bundle_root=output_dir,
            run_root=run_root,
            conditions=conditions,
            selected_condition=selected_condition,
            input_records=input_records,
            mode=bundle_mode,
            panel_overrides=panel_overrides,
            paper_output_overrides=paper_output_overrides,
            canonical_logs=canonical_logs,
            overwrite=overwrite,
            allow_missing_panels=allow_missing_panels,
        )
        warnings.extend(bundle_warnings)

    canonical_log_records = {}
    for name, source in sorted(canonical_logs.items()):
        source = _require_file(source, f"canonical log {name}")
        canonical_log_records[name] = {
            "source_path": str(source),
            "source_size_bytes": int(source.stat().st_size),
            "source_sha256": _sha256(source),
            "bundle_path": bundle_paths.get(f"canonical_log_{name}"),
        }

    selected_row = ranking.loc[
        ranking["condition"].astype(str) == selected_condition
    ].iloc[0]
    next_best_scores = ranking.loc[
        ranking["condition"].astype(str) != selected_condition, "auto_score"
    ]
    score_margin = (
        float(selected_row["auto_score"] - next_best_scores.max())
        if not next_best_scores.empty
        else math.nan
    )
    manifest = {
        "schema_version": 2,
        "workflow": "controlled_distribution_condition_comparison",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "implementation": {
            "script_path": str(Path(__file__).resolve()),
            "script_sha256": _sha256(Path(__file__).resolve()),
        },
        "run_root": str(run_root),
        "conditions": list(conditions),
        "baseline": baseline,
        "bundle_mode": bundle_mode,
        "input_records": input_records,
        "canonical_log_records": canonical_log_records,
        "bundle_source_overrides": {
            "selected_panels": {
                condition: str(path)
                for condition, path in sorted(panel_overrides.items())
            },
            "paper_outputs": {
                condition: str(path)
                for condition, path in sorted(paper_output_overrides.items())
            },
        },
        "condition_metadata": {
            condition: {
                "alpha_spatial": manifests[condition].get("alpha_spatial"),
                "alpha_express": manifests[condition].get("alpha_express"),
                "weight_checkpoint_sha256": manifests[condition].get(
                    "weight_checkpoint_sha256"
                ),
                "score_checkpoint_sha256": manifests[condition].get(
                    "score_checkpoint_sha256"
                ),
            }
            for condition in conditions
        },
        "pairing": {
            "keys": ["time", "space"],
            "delta_definition": "candidate_minus_baseline",
            "strictly_same_recorded_inputs_and_settings": not pairing_warnings,
        },
        "automatic_scoring": {
            "policy": "equal_criterion_rank_v1",
            "aggregation": "arithmetic mean over paired time/space rows",
            "criteria": [
                {
                    **dict(item),
                    "weight": 1.0 / len(AUTO_SCORE_CRITERIA),
                }
                for item in AUTO_SCORE_CRITERIA
            ],
            "rank_utility": (
                "For each criterion: best=1, worst=0, intermediate ranks linearly "
                "spaced, ties averaged. Weighted utilities are summed."
            ),
            "auto_winner": auto_winner,
            "tie_at_best": tied,
            "scores": {
                str(row["condition"]): float(row["auto_score"])
                for _, row in ranking.iterrows()
            },
            "limitation": (
                "This score ranks numerical distribution diagnostics only. It does "
                "not establish biological validity, mechanism, or manuscript readiness."
            ),
        },
        "selection": {
            "requested": winner,
            "selected_condition": selected_condition,
            "reason": selection_reason,
            "selected_auto_score": float(selected_row["auto_score"]),
            "score_margin_over_best_other": score_margin,
        },
        "outputs": {
            "comparison": comparison_paths,
            "ranking": str(ranking_path),
            "selection_criteria": str(criterion_scores_path),
            "bundle": bundle_paths,
        },
        "warnings": warnings,
    }
    manifest_path = comparison_dir / "selection_manifest.json"
    manifest["outputs"]["selection_manifest"] = str(manifest_path)

    inventory_path = None
    if bundle_mode != "none":
        provenance_dir = output_dir / "05_provenance"
        provenance_dir.mkdir(parents=True, exist_ok=True)
        commands_path = provenance_dir / "reproduction_commands.txt"
        commands_path.write_text(
            _render_reproduction_commands(
                run_root=run_root,
                output_dir=output_dir,
                conditions=conditions,
                baseline=baseline,
                winner=winner,
                metrics_overrides=metrics_overrides,
                manifest_overrides=manifest_overrides,
                bundle_mode=bundle_mode,
                panel_overrides=panel_overrides,
                paper_output_overrides=paper_output_overrides,
                canonical_logs=canonical_logs,
                allow_missing_panels=allow_missing_panels,
            ),
            encoding="utf-8",
        )
        readme_path = _write_review_readme(
            output_dir / "README.md",
            conditions=conditions,
            baseline=baseline,
            selected_condition=selected_condition,
            selection_reason=selection_reason,
            requested_winner=winner,
            ranking=ranking,
            manifests=manifests,
            canonical_logs=canonical_logs,
            warnings=warnings,
        )
        inventory_path = provenance_dir / "artifact_inventory.csv"
        bundle_paths.update(
            {
                "readme": str(readme_path),
                "reproduction_commands": str(commands_path),
                "artifact_inventory": str(inventory_path),
            }
        )
    manifest_path.write_text(
        json.dumps(_json_ready(manifest), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if inventory_path is not None:
        _write_artifact_inventory(output_dir, inventory_path)
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=list(DEFAULT_CONDITIONS),
        help="Condition directory names (default: the two alpha_express runs).",
    )
    parser.add_argument(
        "--baseline",
        default=None,
        help="Paired-delta baseline (default: first --conditions value).",
    )
    parser.add_argument(
        "--winner",
        default="auto",
        help="auto or an explicit condition name.",
    )
    parser.add_argument(
        "--metrics-path",
        action="append",
        default=[],
        metavar="CONDITION=PATH",
        help="Override a condition's distribution_metrics.csv path.",
    )
    parser.add_argument(
        "--manifest-path",
        action="append",
        default=[],
        metavar="CONDITION=PATH",
        help="Override a condition's run_manifest.json path.",
    )
    parser.add_argument(
        "--bundle-mode",
        choices=["none", "symlink", "copy"],
        default="none",
        help=(
            "Optionally assemble a review bundle. Use copy for a portable bundle; "
            "the output directory must be new or empty."
        ),
    )
    parser.add_argument(
        "--selected-panels",
        action="append",
        default=[],
        metavar="CONDITION=PATH",
        help="Selected manuscript panel directory for a condition.",
    )
    parser.add_argument(
        "--paper-output",
        action="append",
        default=[],
        metavar="CONDITION=PATH",
        help=(
            "Native paper-downstream output root for a condition. It must contain "
            "run_manifest.json plus all seven stage directories. If a selected-panel "
            "override itself has that contract, it is used automatically."
        ),
    )
    parser.add_argument(
        "--canonical-log",
        action="append",
        default=[],
        metavar="BUNDLE_NAME=PATH",
        help=(
            "Explicit canonical log to copy into 04_logs. Repeat with unique bundle "
            "filenames; the mixed run logs directory is never copied implicitly."
        ),
    )
    parser.add_argument(
        "--allow-missing-panels",
        action="store_true",
        help="Permit audit bundle creation before manuscript panels exist.",
    )
    parser.add_argument(
        "--overwrite-existing-output",
        action="store_true",
        help=(
            "Allow replacement for comparison-only output. Portable review bundles "
            "always require a new or empty directory."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    conditions = tuple(args.conditions)
    baseline = args.baseline or conditions[0]
    manifest = run_comparison(
        run_root=args.run_root,
        output_dir=args.output_dir,
        conditions=conditions,
        baseline=baseline,
        winner=args.winner,
        metrics_overrides=_parse_condition_paths(
            args.metrics_path, option_name="--metrics-path"
        ),
        manifest_overrides=_parse_condition_paths(
            args.manifest_path, option_name="--manifest-path"
        ),
        bundle_mode=args.bundle_mode,
        panel_overrides=_parse_condition_paths(
            args.selected_panels, option_name="--selected-panels"
        ),
        paper_output_overrides=_parse_condition_paths(
            args.paper_output, option_name="--paper-output"
        ),
        canonical_logs=_parse_named_paths(
            args.canonical_log, option_name="--canonical-log"
        ),
        allow_missing_panels=bool(args.allow_missing_panels),
        overwrite=bool(args.overwrite_existing_output),
    )
    print(json.dumps(_json_ready(manifest), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
