"""Read saved S4–S5 panel data and rebuild the published figures."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_builder(dataset: str):
    normalized = str(dataset).strip().lower().replace("-", "_")
    if normalized == "scnt":
        normalized = "scnt_cortex"
    if normalized == "weinreb":
        from . import weinreb_figure as builder
    elif normalized == "scnt_cortex":
        from . import scnt_figure as builder
    else:
        raise KeyError("dataset must be 'weinreb' or 'scnt_cortex'.")
    return normalized, builder


def validate_historical_figure_bundle(
    dataset: str, bundle_dir: str | Path
) -> dict[str, Any]:
    """Verify every compact panel-data file against the archive manifest."""

    normalized, builder = _load_builder(dataset)
    bundle = Path(bundle_dir).expanduser().resolve()
    panel_data = bundle / "panel_data"
    source_manifest_path = panel_data / "source_manifest.json"
    if not source_manifest_path.is_file():
        raise FileNotFoundError(source_manifest_path)
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    expected_figure = {
        "weinreb": "weinreb_nonspatial_interaction_a4",
        "scnt_cortex": "scnt_nonspatial_interaction_a4",
    }[normalized]
    if source_manifest.get("figure") != expected_figure:
        raise ValueError(
            f"Figure bundle declares {source_manifest.get('figure')!r}, "
            f"expected {expected_figure!r}."
        )
    sources = source_manifest.get("sources")
    if not isinstance(sources, dict):
        raise ValueError("panel_data/source_manifest.json lacks sources.")
    expected = {"derived_observed_cells": "observed_cells.csv.gz"}
    expected.update(dict(builder.COPIED_NAMES))
    records: dict[str, Any] = {}
    for source_key, filename in expected.items():
        path = panel_data / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        record = sources.get(source_key)
        if not isinstance(record, dict) or not isinstance(record.get("sha256"), str):
            raise ValueError(f"Missing SHA-256 record for panel source {source_key!r}.")
        observed = _sha256(path)
        if observed != record["sha256"]:
            raise ValueError(
                f"Panel data {filename} changed: {observed} != {record['sha256']}."
            )
        records[filename] = {
            "path": str(path),
            "sha256": observed,
            "size_bytes": path.stat().st_size,
        }
    return {
        "dataset": normalized,
        "figure": expected_figure,
        "bundle": str(bundle),
        "source_manifest": str(source_manifest_path),
        "source_manifest_sha256": _sha256(source_manifest_path),
        "panel_data": records,
    }


def replay_nonspatial_figure(
    dataset: str,
    bundle_dir: str | Path,
    output_dir: str | Path,
    *,
    dpi: int = 320,
) -> dict[str, Any]:
    """Rebuild PDF/PNG and tables from compact archived panel data only.

    This reproduces the accepted historical figure.  It does not relabel the
    historical models as corrected matched-ablation training; new formal runs
    use :func:`CytoBridge.nonspatial.train_nonspatial_condition`.
    """

    validation = validate_historical_figure_bundle(dataset, bundle_dir)
    normalized, builder = _load_builder(dataset)
    output = Path(output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing non-empty figure output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    bundle = Path(bundle_dir).expanduser().resolve()

    builder.BUNDLE = output
    builder.PANEL_DATA = bundle / "panel_data"
    builder.METRICS = output / "metrics"
    builder.require_panel_data()
    builder.build_figure(int(dpi))

    stem = builder.FIGURE_STEM
    artifacts = {}
    for path in sorted(
        [output / f"{stem}.pdf", output / f"{stem}.png"]
        + list((output / "metrics").glob("*"))
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
        artifacts[str(path.relative_to(output))] = {
            "path": str(path),
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        }

    accepted_metrics = bundle / "metrics"
    metric_replay = {}
    for generated in sorted((output / "metrics").glob("*")):
        accepted = accepted_metrics / generated.name
        if not accepted.is_file():
            raise FileNotFoundError(
                f"Accepted bundle lacks generated metric {generated.name!r}."
            )
        metric_replay[generated.name] = {
            "accepted_sha256": _sha256(accepted),
            "generated_sha256": _sha256(generated),
            "byte_identical": _sha256(accepted) == _sha256(generated),
        }
        if not metric_replay[generated.name]["byte_identical"]:
            raise ValueError(
                f"Replayed metric {generated.name!r} differs from accepted bytes."
            )

    provenance_path = output / "figure_replay_provenance.md"
    source_lines = [
        f"- `{record['path']}` — `{record['sha256']}`"
        for record in validation["panel_data"].values()
    ]
    artifact_lines = [
        f"- `{relative}` — `{record['sha256']}`"
        for relative, record in artifacts.items()
    ]
    provenance_path.write_text(
        "\n".join(
            [
                f"# {validation['figure']} replay provenance",
                "",
                "This is an exact historical panel-data replay. It does not "
                "relabel the archived models as the corrected matched ablation.",
                "",
                "## Source paths",
                "",
                *source_lines,
                "",
                "## Rebuild",
                "",
                "```bash",
                "cytobridge nonspatial figure \\",
                f"  --dataset {normalized} \\",
                f"  --bundle-dir {Path(bundle_dir).expanduser().resolve()} \\",
                f"  --output-dir {output}",
                "```",
                "",
                "The command verifies `panel_data/source_manifest.json` before "
                "rendering and requires all regenerated metric files to be "
                "byte-identical to the accepted bundle.",
                "",
                "## SHA-256",
                "",
                *artifact_lines,
                "",
            ]
        ),
        encoding="utf-8",
    )
    artifacts[provenance_path.name] = {
        "path": str(provenance_path),
        "sha256": _sha256(provenance_path),
        "size_bytes": provenance_path.stat().st_size,
    }

    manifest = {
        "schema_version": 1,
        "operation": "replay_nonspatial_figure",
        "dataset": normalized,
        "historical_replay": True,
        "historical_training_contract": "archived-2026-run",
        "new_formal_training_contract": (
            "isolated-interaction-crn-v1 with velocity_score_cross_term"
        ),
        "input_bundle": validation,
        "metric_replay": metric_replay,
        "artifacts": artifacts,
    }
    manifest_path = output / "figure_replay_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {**manifest, "manifest": str(manifest_path)}


__all__ = ["replay_nonspatial_figure", "validate_historical_figure_bundle"]
