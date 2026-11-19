#!/usr/bin/env python3
"""Report an audited matched LOTO-versus-full-data comparison.

The input is the formal output of ``evaluate_matched_tracks.py``.  This
reporter intentionally keeps state, spatial, and joint spaces separate.  It
does not rank methods, construct an overall score, or treat random-projection
repeats as biological or model-training confidence intervals.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
import sys
import uuid
from pathlib import Path
from typing import Any, Iterable, NamedTuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


SPACES = ("state", "spatial", "joint")
METRICS = {
    "sliced_w2": {
        "label": "Sliced Wasserstein-2",
        "role": "primary",
        "loto": "sliced_w2_loto",
        "full_data": "sliced_w2_full_data",
    },
    "exact_w1": {
        "label": "Bounded exact Wasserstein-1",
        "role": "supplement",
        "loto": "exact_w1_loto",
        "full_data": "exact_w1_full_data",
    },
    "exact_w2": {
        "label": "Bounded exact Wasserstein-2",
        "role": "supplement",
        "loto": "exact_w2_loto",
        "full_data": "exact_w2_full_data",
    },
}


class ReportError(ValueError):
    """Raised when formal matched-report inputs violate their contract."""


class _InputSnapshot(NamedTuple):
    """One immutable in-memory view used for both parsing and hashing."""

    path: Path
    data: bytes
    sha256: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _snapshot_file(path: Path, *, label: str) -> _InputSnapshot:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise ReportError(f"cannot read {label} {path}: {exc}") from exc
    return _InputSnapshot(
        path=path,
        data=data,
        sha256=hashlib.sha256(data).hexdigest(),
    )


def _load_json(snapshot: _InputSnapshot, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(snapshot.data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReportError(f"cannot parse {label} {snapshot.path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReportError(f"{label} must contain a JSON object")
    return payload


def _strict_json_bytes(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _temporary_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.partial.{os.getpid()}.{uuid.uuid4().hex}")


def _link_no_overwrite(temporary: Path, destination: Path, *, label: str) -> None:
    try:
        os.link(temporary, destination)
    except FileExistsError as exc:
        raise ReportError(f"refusing to overwrite {label}: {destination}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def _immutable_bytes(path: Path, payload: bytes, *, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path)
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        _link_no_overwrite(temporary, path, label=label)
    finally:
        temporary.unlink(missing_ok=True)


def _immutable_csv(path: Path, frame: pd.DataFrame) -> None:
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False)
    _immutable_bytes(
        path,
        buffer.getvalue().encode("utf-8"),
        label="report table",
    )


def _write_final_manifest(path: Path, payload: dict[str, Any]) -> None:
    """Publish the completion marker with a hard-link and no overwrite path."""

    _immutable_bytes(
        path,
        _strict_json_bytes(payload),
        label="final report manifest",
    )


def _save_figure(figure: plt.Figure, png_path: Path) -> list[Path]:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = png_path.with_suffix(".pdf")
    for path, image_format, options in (
        (png_path, "png", {"dpi": 260}),
        (pdf_path, "pdf", {"metadata": {"Creator": "CytoBridge"}}),
    ):
        temporary = _temporary_path(path)
        try:
            figure.savefig(
                temporary,
                format=image_format,
                bbox_inches="tight",
                **options,
            )
            _link_no_overwrite(temporary, path, label="report plot")
        finally:
            temporary.unlink(missing_ok=True)
    plt.close(figure)
    return [png_path, pdf_path]


_OUTPUT_CLAIM = ".matched_report.in_progress"


def _absolute_output_path(path: Path) -> Path:
    """Return an absolute path without following a possibly foreign symlink."""

    return Path(os.path.abspath(path.expanduser()))


def _output_entries(output_dir: Path) -> list[Path]:
    try:
        return list(output_dir.iterdir())
    except OSError as exc:
        raise ReportError(
            f"cannot inspect report output directory {output_dir}: {exc}"
        ) from exc


def _assert_output_available(output_dir: Path) -> None:
    if output_dir.is_symlink():
        raise ReportError(
            f"report output directory must not be a symlink: {output_dir}"
        )
    if not output_dir.exists():
        return
    if not output_dir.is_dir():
        raise ReportError(f"report output path is not a directory: {output_dir}")
    entries = _output_entries(output_dir)
    if entries:
        names = sorted(path.name for path in entries)
        raise ReportError(
            "report output directory must be empty; refusing foreign, partial, or "
            f"completed contents in {output_dir}: {names}"
        )


def _claim_output(output_dir: Path) -> Path:
    """Exclusively claim an empty directory before creating any artifacts."""

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        output_dir.mkdir()
    except FileExistsError:
        pass
    _assert_output_available(output_dir)
    claim_path = output_dir / _OUTPUT_CLAIM
    try:
        descriptor = os.open(
            claim_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
    except FileExistsError as exc:
        raise ReportError(
            f"report output directory is already claimed or partial: {output_dir}"
        ) from exc
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(f"pid={os.getpid()} uuid={uuid.uuid4().hex}\n")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        claim_path.unlink(missing_ok=True)
        raise
    entries = _output_entries(output_dir)
    if set(entries) != {claim_path}:
        names = sorted(path.name for path in entries if path != claim_path)
        raise ReportError(
            "report output directory acquired foreign contents while being claimed: "
            f"{names}"
        )
    return claim_path


def _assert_owned_output(
    output_dir: Path,
    claim_path: Path,
    artifacts: Iterable[Path],
) -> None:
    expected = {claim_path, *(Path(path) for path in artifacts)}
    observed: set[Path] = set()
    for path in output_dir.rglob("*"):
        if path.is_symlink():
            raise ReportError(f"report output contains a foreign symlink: {path}")
        if path.is_file():
            observed.add(path)
    if observed != expected:
        missing = sorted(
            str(path.relative_to(output_dir)) for path in expected - observed
        )
        extra = sorted(
            str(path.relative_to(output_dir)) for path in observed - expected
        )
        raise ReportError(
            "report output ownership check failed before publication; "
            f"missing={missing}, foreign={extra}"
        )


def _as_bool(series: pd.Series, *, column: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    normalized = series.astype(str).str.strip().str.casefold()
    if not normalized.isin({"true", "false"}).all():
        invalid = sorted(set(series[~normalized.isin({"true", "false"})].astype(str)))
        raise ReportError(f"{column} must contain only true/false; found {invalid}")
    return normalized.eq("true")


def _finite_numeric(
    frame: pd.DataFrame,
    columns: Iterable[str],
    *,
    nonnegative: bool,
) -> None:
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any() or not np.isfinite(values.to_numpy(float)).all():
            raise ReportError(f"{column} must contain only finite numeric values")
        if nonnegative and (values < 0).any():
            raise ReportError(f"{column} must be nonnegative")
        frame[column] = values.astype(float)


def _series_is_exactly_constant(series: pd.Series) -> bool:
    values = series.tolist()
    if not values:
        return True
    first = values[0]
    first_missing = bool(pd.isna(first))
    for value in values[1:]:
        value_missing = bool(pd.isna(value))
        if first_missing or value_missing:
            if first_missing and value_missing:
                continue
            return False
        if value != first:
            return False
    return True


def _load_registry(snapshot: _InputSnapshot) -> list[dict[str, Any]]:
    payload = _load_json(snapshot, label="method registry")
    records = payload.get("methods")
    if not isinstance(records, list) or not records:
        raise ReportError("method registry must contain a non-empty methods list")
    parsed: list[dict[str, Any]] = []
    seen: set[str] = set()
    alias_owners: dict[str, str] = {}
    for order, raw in enumerate(records):
        if not isinstance(raw, dict):
            raise ReportError("each method registry entry must be an object")
        method = str(raw.get("method", "")).strip()
        display_name = str(raw.get("display_name", method)).strip()
        status = str(raw.get("status", "evaluated")).strip().casefold()
        scope = str(raw.get("scope", "")).strip().casefold()
        spaces = [str(value).strip().casefold() for value in raw.get("spaces", [])]
        if not method or not display_name or method in seen:
            raise ReportError(f"invalid or duplicate registry method {method!r}")
        if not status or not scope:
            raise ReportError(f"{method}: registry status/scope must be non-empty")
        if len(set(spaces)) != len(spaces) or any(
            space not in SPACES for space in spaces
        ):
            raise ReportError(f"{method}: invalid or duplicate spaces {spaces}")
        if status == "evaluated" and not spaces:
            raise ReportError(
                f"{method}: evaluated methods must declare a feature space"
            )
        seen.add(method)
        aliases = raw.get("aliases", [])
        if not isinstance(aliases, list) or any(
            not isinstance(alias, str) or not alias.strip() for alias in aliases
        ):
            raise ReportError(f"{method}: registry aliases must be non-empty strings")
        resolver_values = [method, display_name, *(alias.strip() for alias in aliases)]
        for alias in resolver_values:
            key = alias.casefold()
            previous = alias_owners.get(key)
            if previous is not None and previous != method:
                raise ReportError(
                    f"registry alias {alias!r} is ambiguous between "
                    f"{previous!r} and {method!r}"
                )
            alias_owners[key] = method
        record = {
            **raw,
            "method": method,
            "display_name": display_name,
            "status": status,
            "scope": scope,
            "spaces": spaces,
            "aliases": [alias.strip() for alias in aliases],
            "resolver_aliases": resolver_values,
            "method_order": order,
        }
        parsed.append(record)
    return parsed


def _registry_alias_lookup(registry: list[dict[str, Any]]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for record in registry:
        for alias in record["resolver_aliases"]:
            key = str(alias).casefold()
            previous = lookup.get(key)
            if previous is not None and previous != record["method"]:
                # _load_registry already rejects this.  Keep the guard local to
                # the trust boundary in case a caller constructs records itself.
                raise ReportError(
                    f"registry alias {alias!r} resolves ambiguously to "
                    f"{previous!r} and {record['method']!r}"
                )
            lookup[key] = str(record["method"])
    return lookup


def _resolve_declared_path(value: Any, *, relative_to: Path, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ReportError(f"{label} must be a non-empty path string")
    declared = Path(value).expanduser()
    if not declared.is_absolute():
        declared = relative_to / declared
    return declared.resolve()


def _require_sha256(value: Any, *, label: str) -> str:
    digest = value if isinstance(value, str) else ""
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ReportError(f"{label} must be a lowercase SHA-256 digest")
    return digest


def _bound_method_mapping(
    manifest: dict[str, Any],
    *,
    manifest_path: Path,
    registry_snapshot: _InputSnapshot,
    registry: list[dict[str, Any]],
) -> dict[str, str]:
    binding = manifest.get("method_registry")
    if not isinstance(binding, dict):
        raise ReportError(
            "matched evaluation manifest is missing the bound method_registry"
        )
    declared_path = _resolve_declared_path(
        binding.get("path"),
        relative_to=manifest_path.parent,
        label="method_registry.path",
    )
    if declared_path != registry_snapshot.path:
        raise ReportError(
            "--method-registry differs from the path bound by the matched manifest"
        )
    expected_sha = _require_sha256(
        binding.get("sha256"), label="method_registry.sha256"
    )
    if registry_snapshot.sha256 != expected_sha:
        raise ReportError(
            "method registry SHA-256 differs from the evaluator-bound registry"
        )

    raw_mapping = binding.get("raw_to_canonical")
    if not isinstance(raw_mapping, dict) or not raw_mapping:
        raise ReportError("method_registry.raw_to_canonical must be a non-empty object")
    registry_lookup = {record["method"]: record for record in registry}
    alias_lookup = _registry_alias_lookup(registry)
    mapping: dict[str, str] = {}
    required_fields = {
        "raw_method",
        "canonical_method",
        "display_name",
        "status",
        "scope",
        "declared_spaces",
    }
    for raw_method, metadata in raw_mapping.items():
        if (
            not isinstance(raw_method, str)
            or not raw_method
            or raw_method != raw_method.strip()
        ):
            raise ReportError(
                "method_registry.raw_to_canonical keys must be exact non-empty raw strings"
            )
        if not isinstance(metadata, dict) or not required_fields.issubset(metadata):
            raise ReportError(
                f"method_registry.raw_to_canonical[{raw_method!r}] must bind "
                f"{sorted(required_fields)}"
            )
        canonical = metadata.get("canonical_method")
        if (
            not isinstance(canonical, str)
            or not canonical
            or canonical != canonical.strip()
        ):
            raise ReportError(
                f"{raw_method!r}: canonical_method must be an exact string"
            )
        if canonical not in registry_lookup:
            raise ReportError(
                f"{raw_method!r}: bound canonical method {canonical!r} is absent "
                "from the hashed registry"
            )
        record = registry_lookup[canonical]
        resolved_canonical = alias_lookup.get(raw_method.casefold())
        if resolved_canonical != canonical:
            raise ReportError(
                f"{raw_method!r}: evaluator-bound alias maps to {canonical!r}, but "
                f"the hashed registry resolves it to {resolved_canonical!r}"
            )
        expected_metadata = {
            "raw_method": raw_method,
            "canonical_method": canonical,
            "display_name": record["display_name"],
            "status": record["status"],
            "scope": record["scope"],
            "declared_spaces": record["spaces"],
        }
        observed_metadata = {key: metadata.get(key) for key in required_fields}
        if observed_metadata != expected_metadata:
            raise ReportError(
                f"{raw_method!r}: evaluator-bound canonical metadata differs from "
                "the hashed registry"
            )
        mapping[raw_method] = canonical
    if len(set(mapping.values())) != len(mapping):
        raise ReportError(
            "method_registry.raw_to_canonical must be one-to-one; two raw methods "
            "cannot bind the same canonical method"
        )

    evaluated = {
        record["method"] for record in registry if record["status"] == "evaluated"
    }
    if set(mapping.values()) != evaluated:
        raise ReportError(
            "bound canonical methods must equal evaluated methods in the hashed "
            f"registry; bound={sorted(mapping.values())}, registry={sorted(evaluated)}"
        )

    canonical_methods = binding.get("canonical_methods")
    if (
        not isinstance(canonical_methods, list)
        or any(not isinstance(value, str) for value in canonical_methods)
        or len(canonical_methods) != len(set(canonical_methods))
        or set(canonical_methods) != set(mapping.values())
    ):
        raise ReportError(
            "method_registry.canonical_methods must contain each bound canonical "
            "name exactly once"
        )
    canonical_to_raw = {canonical: raw for raw, canonical in mapping.items()}
    return {canonical_to_raw[canonical]: canonical for canonical in canonical_methods}


def _validate_manifest(
    manifest: dict[str, Any],
    manifest_snapshot: _InputSnapshot,
    paired_snapshot: _InputSnapshot,
    registry_snapshot: _InputSnapshot,
    registry: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, str], tuple[int, ...]]:
    if manifest.get("status") != "complete":
        raise ReportError("matched evaluation manifest status must be complete")
    if manifest.get("design") != "matched_loto_vs_full_data":
        raise ReportError("matched evaluation manifest has the wrong design")
    raw_targets = manifest.get("targets", [])
    if (
        not isinstance(raw_targets, list)
        or not raw_targets
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
            or int(value) != value
            for value in raw_targets
        )
    ):
        raise ReportError(
            "matched evaluation targets must be a non-empty list of finite integer "
            "stage identifiers"
        )
    targets = tuple(int(value) for value in raw_targets)
    if len(targets) != len(set(targets)):
        raise ReportError("matched evaluation targets must be unique")
    tracks = manifest.get("tracks")
    if not isinstance(tracks, dict):
        raise ReportError("matched evaluation manifest is missing track metadata")
    loto = tracks.get("loto")
    full = tracks.get("full_data")
    if not isinstance(loto, dict) or loto.get("evaluation_scope") != "held_out":
        raise ReportError("LOTO track must be explicitly marked held_out")
    if (
        not isinstance(full, dict)
        or full.get("evaluation_scope") != "in_sample"
        or full.get("is_in_sample") is not True
    ):
        raise ReportError("full_data track must be explicitly marked in_sample")
    policy = manifest.get("reporting_policy")
    if not isinstance(policy, dict):
        raise ReportError("matched evaluation manifest is missing reporting_policy")
    required_policy = {
        "full_data_is_in_sample": True,
        "cross_space_aggregation": False,
        "overall_score": False,
        "ranking": False,
        "statistical_inference": False,
    }
    for key, expected in required_policy.items():
        if policy.get(key) is not expected:
            raise ReportError(f"reporting_policy.{key} must be {expected}")
    projection_repeats = manifest.get("projection_repeats", 0)
    if (
        isinstance(projection_repeats, bool)
        or not isinstance(projection_repeats, (int, float))
        or int(projection_repeats) != projection_repeats
        or int(projection_repeats) <= 0
    ):
        raise ReportError("projection_repeats must be a positive integer")
    declared_paired_path = _resolve_declared_path(
        manifest.get("paired_summary_csv"),
        relative_to=manifest_snapshot.path.parent,
        label="paired_summary_csv",
    )
    if declared_paired_path != paired_snapshot.path:
        raise ReportError(
            "--paired-summary differs from the path bound by the matched manifest"
        )
    expected_sha = _require_sha256(
        manifest.get("paired_summary_csv_sha256"),
        label="paired_summary_csv_sha256",
    )
    if paired_snapshot.sha256 != expected_sha:
        raise ReportError("paired summary CSV SHA-256 differs from matched manifest")
    mapping = _bound_method_mapping(
        manifest,
        manifest_path=manifest_snapshot.path,
        registry_snapshot=registry_snapshot,
        registry=registry,
    )
    evaluated = [
        record["method"] for record in registry if record["status"] == "evaluated"
    ]
    methods = manifest.get("methods")
    if (
        not isinstance(methods, list)
        or not methods
        or any(not isinstance(method, str) for method in methods)
        or len(methods) != len(set(methods))
    ):
        raise ReportError("matched evaluation manifest has no methods")
    if methods != list(mapping.values()) or set(methods) != set(evaluated):
        raise ReportError(
            "matched methods must equal the evaluator-bound canonical CLI order and "
            "evaluated methods in the top-level registry; "
            f"manifest={sorted(methods)}, registry={sorted(evaluated)}"
        )
    declared_spaces = {str(value) for value in manifest.get("spaces", [])}
    expected_spaces = {
        space
        for record in registry
        if record["status"] == "evaluated"
        for space in record["spaces"]
    }
    if declared_spaces != expected_spaces:
        raise ReportError(
            f"matched manifest spaces {sorted(declared_spaces)} differ from registry "
            f"{sorted(expected_spaces)}"
        )
    row_count = manifest.get("n_paired_rows")
    if (
        isinstance(row_count, bool)
        or not isinstance(row_count, (int, float))
        or int(row_count) != row_count
        or int(row_count) <= 0
    ):
        raise ReportError("matched evaluation manifest has invalid n_paired_rows")
    return manifest, mapping, targets


def _validate_paired(
    paired_snapshot: _InputSnapshot,
    manifest: dict[str, Any],
    registry: list[dict[str, Any]],
    raw_to_canonical: dict[str, str],
    targets: tuple[int, ...],
) -> pd.DataFrame:
    try:
        paired = pd.read_csv(io.BytesIO(paired_snapshot.data))
    except (OSError, UnicodeDecodeError, pd.errors.ParserError) as exc:
        raise ReportError(
            f"cannot parse paired summary CSV {paired_snapshot.path}: {exc}"
        ) from exc
    required = {
        "raw_method",
        "canonical_method",
        "method",
        "method_display_name",
        "target",
        "space",
        "evaluation_scope_loto",
        "evaluation_scope_full_data",
        "is_in_sample_loto",
        "is_in_sample_full_data",
        "projection_repeats_loto",
        "projection_repeats_full_data",
        "sliced_w2_mean_loto",
        "sliced_w2_std_loto",
        "sliced_w2_mean_full_data",
        "sliced_w2_std_full_data",
        "sliced_w2_mean_loto_minus_full_data",
        "exact_w1_loto",
        "exact_w1_full_data",
        "exact_w1_loto_minus_full_data",
        "exact_w2_loto",
        "exact_w2_full_data",
        "exact_w2_loto_minus_full_data",
        "source_time_loto",
        "source_time_full_data",
        "tmv_available_loto",
        "tmv_available_full_data",
        "tmv_loto",
        "tmv_full_data",
        "observed_mass_relative_loto",
        "observed_mass_relative_full_data",
        "tmv_directly_comparable",
        "tmv_loto_minus_full_data",
        "full_data_is_in_sample",
        "comparison_type",
        "comparison",
        "exact_comparison_type",
    }
    missing = sorted(required.difference(paired.columns))
    if missing:
        raise ReportError(f"paired summary CSV is missing columns: {missing}")
    forbidden = [
        column
        for column in paired.columns
        if "rank" in column.casefold() or "overall" in column.casefold()
    ]
    if forbidden:
        raise ReportError(
            f"paired summary contains forbidden rank/overall columns: {forbidden}"
        )
    if len(paired) != int(manifest["n_paired_rows"]):
        raise ReportError("paired summary row count differs from matched manifest")

    paired = paired.copy()
    raw_methods = paired["raw_method"]
    canonical_methods = paired["canonical_method"]
    compatibility_methods = paired["method"]
    if (
        raw_methods.isna().any()
        or canonical_methods.isna().any()
        or compatibility_methods.isna().any()
    ):
        raise ReportError("paired method identity columns must not contain null values")
    raw_methods = raw_methods.astype(str)
    canonical_methods = canonical_methods.astype(str)
    compatibility_methods = compatibility_methods.astype(str)
    mapped = raw_methods.map(raw_to_canonical)
    if mapped.isna().any():
        unknown = sorted(set(raw_methods[mapped.isna()]))
        raise ReportError(
            "paired summary contains raw methods absent from the evaluator-bound exact "
            f"mapping: {unknown}"
        )
    if not mapped.equals(canonical_methods):
        mismatches = paired.loc[
            mapped.ne(canonical_methods), ["raw_method", "canonical_method"]
        ].drop_duplicates()
        raise ReportError(
            "paired raw_method to canonical_method identities differ from the exact "
            f"evaluator-bound mapping: {mismatches.to_dict('records')}"
        )
    if not compatibility_methods.equals(canonical_methods):
        raise ReportError("paired method must exactly equal canonical_method")
    display_lookup = {
        str(record["method"]): str(record["display_name"]) for record in registry
    }
    expected_display = canonical_methods.map(display_lookup)
    observed_display = paired["method_display_name"].astype(str)
    if paired["method_display_name"].isna().any() or not observed_display.equals(
        expected_display
    ):
        raise ReportError(
            "paired method_display_name differs from the evaluator-bound hashed registry"
        )
    paired["raw_method"] = raw_methods
    paired["canonical_method"] = canonical_methods
    paired["method"] = canonical_methods

    numeric_nonnegative = (
        "sliced_w2_mean_loto",
        "sliced_w2_std_loto",
        "sliced_w2_mean_full_data",
        "sliced_w2_std_full_data",
        "exact_w1_loto",
        "exact_w1_full_data",
        "exact_w2_loto",
        "exact_w2_full_data",
    )
    numeric_signed = (
        "sliced_w2_mean_loto_minus_full_data",
        "exact_w1_loto_minus_full_data",
        "exact_w2_loto_minus_full_data",
        "source_time_loto",
        "source_time_full_data",
    )
    _finite_numeric(paired, numeric_nonnegative, nonnegative=True)
    _finite_numeric(paired, numeric_signed, nonnegative=False)

    for column in (
        "is_in_sample_loto",
        "is_in_sample_full_data",
        "tmv_available_loto",
        "tmv_available_full_data",
        "tmv_directly_comparable",
        "full_data_is_in_sample",
    ):
        paired[column] = _as_bool(paired[column], column=column)
    if paired["is_in_sample_loto"].any():
        raise ReportError("LOTO paired rows must not be marked in-sample")
    if (
        not paired["is_in_sample_full_data"].all()
        or not paired["full_data_is_in_sample"].all()
    ):
        raise ReportError("all full_data paired rows must be explicitly in-sample")
    if set(paired["evaluation_scope_loto"].astype(str)) != {"held_out"}:
        raise ReportError("paired LOTO scope must be held_out")
    if set(paired["evaluation_scope_full_data"].astype(str)) != {"in_sample"}:
        raise ReportError("paired full_data scope must be in_sample")
    if set(paired["comparison_type"].astype(str)) != {"descriptive_paired_gap"}:
        raise ReportError("paired comparison_type must be descriptive_paired_gap")
    if set(paired["comparison"].astype(str)) != {
        "loto_held_out_minus_full_data_in_sample"
    }:
        raise ReportError("paired comparison label is invalid")
    if set(paired["exact_comparison_type"].astype(str)) != {
        "matched_shared_observed_indices_separate_rng"
    }:
        raise ReportError("exact comparison is not the formal matched design")

    paired["target"] = pd.to_numeric(paired["target"], errors="coerce")
    if (
        paired["target"].isna().any()
        or not np.equal(paired["target"], np.floor(paired["target"])).all()
    ):
        raise ReportError("target must contain integer stage identifiers")
    paired["target"] = paired["target"].astype(int)
    if set(paired["target"]) != set(targets):
        raise ReportError(f"paired summary must contain targets {list(targets)}")
    if paired["space"].isna().any():
        raise ReportError("paired summary space must not contain null values")
    paired["space"] = paired["space"].astype(str)
    if not set(paired["space"]).issubset(SPACES):
        raise ReportError(
            f"paired summary contains invalid spaces: {sorted(set(paired['space']))}"
        )

    repeats = int(manifest["projection_repeats"])
    for column in ("projection_repeats_loto", "projection_repeats_full_data"):
        values = pd.to_numeric(paired[column], errors="coerce")
        if values.isna().any() or not (values == repeats).all():
            raise ReportError(
                f"{column} differs from formal projection_repeats={repeats}"
            )
        paired[column] = values.astype(int)

    for stem in ("sliced_w2_mean", "exact_w1", "exact_w2"):
        expected = paired[f"{stem}_loto"] - paired[f"{stem}_full_data"]
        observed = paired[f"{stem}_loto_minus_full_data"]
        if not np.allclose(expected, observed, rtol=1e-10, atol=1e-12):
            raise ReportError(f"{stem} paired gaps do not equal LOTO minus full_data")

    evaluated = {
        record["method"]: record
        for record in registry
        if record["status"] == "evaluated"
    }
    if set(paired["canonical_method"]) != set(evaluated):
        raise ReportError(
            "paired methods differ from evaluated top-level registry methods"
        )
    observed_grid = set(
        paired[["raw_method", "canonical_method", "target", "space"]].itertuples(
            index=False, name=None
        )
    )
    if len(observed_grid) != len(paired):
        raise ReportError("paired summary contains duplicate method/target/space rows")
    expected_grid = {
        (raw_method, canonical_method, target, space)
        for raw_method, canonical_method in raw_to_canonical.items()
        for record in [evaluated[canonical_method]]
        for target in targets
        for space in record["spaces"]
    }
    if observed_grid != expected_grid:
        missing_grid = sorted(expected_grid.difference(observed_grid))
        extra_grid = sorted(observed_grid.difference(expected_grid))
        raise ReportError(
            "paired summary does not match registry applicability; "
            f"missing={missing_grid}, extra={extra_grid}"
        )

    denominator_columns = (
        "observed_mass_relative_loto",
        "observed_mass_relative_full_data",
    )
    for column in denominator_columns:
        paired[column] = pd.to_numeric(paired[column], errors="coerce")
    denominator_loto = paired["observed_mass_relative_loto"]
    denominator_full = paired["observed_mass_relative_full_data"]
    finite_denominators = pd.Series(
        np.isfinite(denominator_loto.to_numpy(float))
        & np.isfinite(denominator_full.to_numpy(float)),
        index=paired.index,
    )
    equal_denominators = pd.Series(
        np.isclose(
            denominator_loto.to_numpy(float),
            denominator_full.to_numpy(float),
            equal_nan=False,
        ),
        index=paired.index,
    )
    expected_direct = (
        paired["tmv_available_loto"]
        & paired["tmv_available_full_data"]
        & paired["source_time_loto"].eq(paired["source_time_full_data"])
        & finite_denominators
        & equal_denominators
    )
    direct = paired["tmv_directly_comparable"]
    if not direct.equals(expected_direct):
        mismatches = paired.loc[
            direct.ne(expected_direct),
            [
                "raw_method",
                "canonical_method",
                "target",
                "space",
                "tmv_directly_comparable",
            ],
        ].copy()
        mismatches["expected_tmv_directly_comparable"] = expected_direct[
            direct.ne(expected_direct)
        ]
        raise ReportError(
            "tmv_directly_comparable must exactly equal the recomputed availability, "
            "source-time, and finite/equal-denominator predicate; "
            f"mismatches={mismatches.to_dict('records')}"
        )
    for column in ("tmv_loto", "tmv_full_data", "tmv_loto_minus_full_data"):
        values = pd.to_numeric(paired[column], errors="coerce")
        if values[direct].isna().any() or not np.isfinite(values[direct]).all():
            raise ReportError(
                f"{column} must be finite for directly comparable TMV rows"
            )
        if values[~direct].notna().any() and column == "tmv_loto_minus_full_data":
            raise ReportError("non-comparable TMV rows must not contain a paired gap")
        paired[column] = values
    if (paired.loc[direct, ["tmv_loto", "tmv_full_data"]] < 0).any().any():
        raise ReportError("TMV values must be nonnegative")
    if direct.any():
        expected_tmv = (
            paired.loc[direct, "tmv_loto"] - paired.loc[direct, "tmv_full_data"]
        )
        if not np.allclose(
            expected_tmv,
            paired.loc[direct, "tmv_loto_minus_full_data"],
            rtol=1e-10,
            atol=1e-12,
        ):
            raise ReportError("TMV paired gaps do not equal LOTO minus full_data")
    consistency_columns = (
        "source_time_loto",
        "source_time_full_data",
        "tmv_available_loto",
        "tmv_available_full_data",
        "tmv_loto",
        "tmv_full_data",
        "observed_mass_relative_loto",
        "observed_mass_relative_full_data",
        "tmv_directly_comparable",
        "tmv_loto_minus_full_data",
    )
    for group_key, frame in paired.groupby(
        ["raw_method", "canonical_method", "target"],
        sort=False,
        dropna=False,
    ):
        for column in consistency_columns:
            if not _series_is_exactly_constant(frame[column]):
                raise ReportError(
                    "TMV method-target metadata varies across spaces for "
                    f"{group_key}: {column}"
                )

    return paired


def _target_summary(
    paired: pd.DataFrame, registry: list[dict[str, Any]], projection_repeats: int
) -> pd.DataFrame:
    lookup = {record["method"]: record for record in registry}
    rows: list[dict[str, Any]] = []
    for row in paired.itertuples(index=False):
        record = lookup[str(row.method)]
        rows.append(
            {
                "raw_method": row.raw_method,
                "canonical_method": row.canonical_method,
                "method": row.method,
                "display_name": record["display_name"],
                "method_order": int(record["method_order"]),
                "scope": record.get("scope"),
                "space": row.space,
                "target": int(row.target),
                "loto_evaluation_scope": "held_out",
                "full_data_evaluation_scope": "in_sample",
                "full_data_is_in_sample": True,
                "comparison_type": "descriptive_paired_gap",
                "n_projection_repeats": projection_repeats,
                "projection_repeat_interpretation": "numerical integration variation; not a confidence interval",
                "sliced_w2_loto": float(row.sliced_w2_mean_loto),
                "sliced_w2_full_data": float(row.sliced_w2_mean_full_data),
                "sliced_w2_loto_minus_full_data": float(
                    row.sliced_w2_mean_loto_minus_full_data
                ),
                "sliced_w2_projection_sd_loto": float(row.sliced_w2_std_loto),
                "sliced_w2_projection_sd_full_data": float(row.sliced_w2_std_full_data),
                "exact_w1_loto": float(row.exact_w1_loto),
                "exact_w1_full_data": float(row.exact_w1_full_data),
                "exact_w1_loto_minus_full_data": float(
                    row.exact_w1_loto_minus_full_data
                ),
                "exact_w2_loto": float(row.exact_w2_loto),
                "exact_w2_full_data": float(row.exact_w2_full_data),
                "exact_w2_loto_minus_full_data": float(
                    row.exact_w2_loto_minus_full_data
                ),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["space", "method_order", "target"], kind="stable")
        .reset_index(drop=True)
    )


def _method_space_summary(
    target: pd.DataFrame, targets_expected: tuple[int, ...]
) -> pd.DataFrame:
    keys = [
        "raw_method",
        "canonical_method",
        "method",
        "display_name",
        "method_order",
        "scope",
        "space",
    ]
    value_columns = (
        "sliced_w2_loto",
        "sliced_w2_full_data",
        "sliced_w2_loto_minus_full_data",
        "exact_w1_loto",
        "exact_w1_full_data",
        "exact_w1_loto_minus_full_data",
        "exact_w2_loto",
        "exact_w2_full_data",
        "exact_w2_loto_minus_full_data",
    )
    rows: list[dict[str, Any]] = []
    for group_key, frame in target.groupby(keys, sort=False, dropna=False):
        observed_targets = frame["target"].astype(int).tolist()
        if len(observed_targets) != len(set(observed_targets)) or set(
            observed_targets
        ) != set(targets_expected):
            raise ReportError(
                "method-space summary does not contain the validated target set "
                f"{list(targets_expected)}: {group_key}"
            )
        targets = list(targets_expected)
        row = dict(zip(keys, group_key))
        row.update(
            {
                "targets": ",".join(f"t{value}" for value in targets),
                "n_targets": len(targets),
                "comparison_type": "descriptive_paired_gap",
                "full_data_is_in_sample": True,
                "target_sd_interpretation": "descriptive stage-to-stage variation; not a confidence interval",
            }
        )
        for column in value_columns:
            values = frame[column].to_numpy(float)
            row[f"{column}_mean_across_targets"] = float(values.mean())
            row[f"{column}_target_sd"] = float(values.std(ddof=1))
        rows.append(row)
    return (
        pd.DataFrame(rows)
        .sort_values(["space", "method_order"], kind="stable")
        .reset_index(drop=True)
    )


def _global_limits(frame: pd.DataFrame, left: str, right: str) -> tuple[float, float]:
    values = np.concatenate((frame[left].to_numpy(float), frame[right].to_numpy(float)))
    low = float(values.min())
    high = float(values.max())
    width = high - low
    pad = 0.08 * width if width > 0 else max(0.05 * max(abs(high), 1.0), 1e-6)
    return max(0.0, low - pad), high + pad


def _dumbbell_figure(
    target: pd.DataFrame,
    registry: list[dict[str, Any]],
    *,
    space: str,
    metric: str,
    targets: tuple[int, ...],
) -> plt.Figure:
    config = METRICS[metric]
    subset = target[target["space"] == space].copy()
    methods = [
        record["method"]
        for record in registry
        if record["status"] == "evaluated" and space in record["spaces"]
    ]
    display = {record["method"]: record["display_name"] for record in registry}
    if not methods or subset.empty:
        raise ReportError(f"no applicable rows for {space}/{metric}")
    left = str(config["full_data"])
    right = str(config["loto"])
    x_limits = _global_limits(subset, left, right)
    y = np.arange(len(methods), dtype=float)
    figure, axes = plt.subplots(
        1,
        len(targets),
        figsize=(max(5.2, 5.05 * len(targets)), max(5.2, 0.46 * len(methods) + 2.2)),
        sharey=True,
        squeeze=False,
    )
    for index, (axis, stage) in enumerate(zip(axes[0], targets)):
        stage_frame = subset[subset["target"] == stage].set_index("method")
        if set(stage_frame.index) != set(methods):
            raise ReportError(f"incomplete applicable method grid for {space}/t{stage}")
        full_values = np.asarray(
            [stage_frame.loc[method, left] for method in methods], float
        )
        loto_values = np.asarray(
            [stage_frame.loc[method, right] for method in methods], float
        )
        for position, full_value, loto_value in zip(y, full_values, loto_values):
            axis.plot(
                [full_value, loto_value],
                [position, position],
                color="#8c8c8c",
                linewidth=1.25,
                zorder=1,
            )
        axis.scatter(
            full_values,
            y,
            s=50,
            marker="o",
            facecolor="white",
            edgecolor="#2166ac",
            linewidth=1.45,
            label="Full-data (in-sample)",
            zorder=3,
        )
        axis.scatter(
            loto_values,
            y,
            s=50,
            marker="o",
            facecolor="#d95f02",
            edgecolor="#d95f02",
            linewidth=1.0,
            label="LOTO (held-out)",
            zorder=4,
        )
        axis.set_xlim(*x_limits)
        axis.set_title(f"Target t{stage}")
        axis.set_xlabel(str(config["label"]))
        axis.set_yticks(y, [display[method] for method in methods])
        axis.invert_yaxis()
        axis.grid(axis="x", alpha=0.22)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.tick_params(axis="y", length=0, labelleft=index == 0)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.935),
        ncol=2,
        frameon=False,
        fontsize=9,
    )
    role = (
        "Primary matched metric"
        if config["role"] == "primary"
        else "Supplemental matched metric"
    )
    figure.suptitle(
        f"{role}: {space.capitalize()} space — {config['label']} (lower is better)",
        fontsize=14,
        y=0.985,
    )
    if metric == "sliced_w2":
        note = (
            "Each segment compares the same method, feature space, and target. "
            "Values are means over shared random projections; projection repeats are numerical "
            "integration checks, not confidence intervals."
        )
    else:
        note = (
            "Bounded exact OT uses matched shared observed indices and separate predicted-sampling RNG. "
            "Full-data values are in-sample; all LOTO-minus-full-data gaps are descriptive."
        )
    figure.text(0.5, 0.012, note, ha="center", va="bottom", fontsize=8.5)
    figure.tight_layout(rect=(0.0, 0.06, 1.0, 0.88))
    return figure


def _tmv_table(paired: pd.DataFrame, registry: list[dict[str, Any]]) -> pd.DataFrame:
    lookup = {record["method"]: record for record in registry}
    direct = paired[paired["tmv_directly_comparable"]].copy()
    columns = [
        "raw_method",
        "canonical_method",
        "method",
        "display_name",
        "method_order",
        "target",
        "source_time",
        "tmv_loto",
        "tmv_full_data",
        "tmv_loto_minus_full_data",
        "observed_mass_relative",
        "full_data_is_in_sample",
        "comparison_type",
    ]
    if direct.empty:
        return pd.DataFrame(columns=columns)
    consistency = (
        "source_time_loto",
        "source_time_full_data",
        "tmv_loto",
        "tmv_full_data",
        "tmv_loto_minus_full_data",
        "observed_mass_relative_loto",
        "observed_mass_relative_full_data",
    )
    rows: list[dict[str, Any]] = []
    for (raw_method, canonical_method, method, target), frame in direct.groupby(
        ["raw_method", "canonical_method", "method", "target"], sort=False
    ):
        for column in consistency:
            if frame[column].nunique(dropna=False) != 1:
                raise ReportError(
                    f"TMV varies across spaces for {method}/t{target}: {column}"
                )
        record = lookup[str(method)]
        first = frame.iloc[0]
        rows.append(
            {
                "raw_method": raw_method,
                "canonical_method": canonical_method,
                "method": method,
                "display_name": record["display_name"],
                "method_order": int(record["method_order"]),
                "target": int(target),
                "source_time": float(first["source_time_loto"]),
                "tmv_loto": float(first["tmv_loto"]),
                "tmv_full_data": float(first["tmv_full_data"]),
                "tmv_loto_minus_full_data": float(first["tmv_loto_minus_full_data"]),
                "observed_mass_relative": float(first["observed_mass_relative_loto"]),
                "full_data_is_in_sample": True,
                "comparison_type": "descriptive_paired_gap",
            }
        )
    return (
        pd.DataFrame(rows, columns=columns)
        .sort_values(["method_order", "target"], kind="stable")
        .reset_index(drop=True)
    )


def _tmv_figure(tmv: pd.DataFrame, targets: tuple[int, ...]) -> plt.Figure:
    if tmv.empty:
        figure, axis = plt.subplots(figsize=(9.2, 3.8))
        axis.axis("off")
        axis.text(
            0.5,
            0.56,
            "No methods have directly comparable native mass/growth TMV",
            transform=axis.transAxes,
            ha="center",
            va="center",
            fontsize=13,
        )
        axis.text(
            0.5,
            0.40,
            "No values were imputed or inferred for methods without applicable TMV.",
            transform=axis.transAxes,
            ha="center",
            va="center",
            fontsize=9,
        )
        figure.tight_layout()
        return figure
    methods = (
        tmv[["method", "display_name", "method_order"]]
        .drop_duplicates()
        .sort_values("method_order")
    )
    method_ids = methods["method"].tolist()
    display = methods.set_index("method")["display_name"].to_dict()
    y_lookup = {method: index for index, method in enumerate(method_ids)}
    x_limits = _global_limits(tmv, "tmv_full_data", "tmv_loto")
    figure, axes = plt.subplots(
        1,
        len(targets),
        figsize=(max(5.2, 5.05 * len(targets)), max(4.5, 0.48 * len(method_ids) + 2.2)),
        sharey=True,
        squeeze=False,
    )
    for index, (axis, stage) in enumerate(zip(axes[0], targets)):
        stage_frame = tmv[tmv["target"] == stage]
        for row in stage_frame.itertuples(index=False):
            position = y_lookup[str(row.method)]
            axis.plot(
                [row.tmv_full_data, row.tmv_loto],
                [position, position],
                color="#8c8c8c",
                linewidth=1.25,
                zorder=1,
            )
            axis.scatter(
                [row.tmv_full_data],
                [position],
                s=50,
                facecolor="white",
                edgecolor="#2166ac",
                linewidth=1.45,
                zorder=3,
            )
            axis.scatter(
                [row.tmv_loto],
                [position],
                s=50,
                facecolor="#d95f02",
                edgecolor="#d95f02",
                linewidth=1.0,
                zorder=4,
            )
        axis.set_xlim(*x_limits)
        axis.set_title(f"Target t{stage}")
        axis.set_xlabel("Relative total-mass-variation error")
        axis.set_yticks(
            range(len(method_ids)), [display[method] for method in method_ids]
        )
        axis.invert_yaxis()
        axis.grid(axis="x", alpha=0.22)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.tick_params(axis="y", length=0, labelleft=index == 0)
    full_handle = axes[0, 0].scatter(
        [], [], facecolor="white", edgecolor="#2166ac", label="Full-data (in-sample)"
    )
    loto_handle = axes[0, 0].scatter(
        [], [], facecolor="#d95f02", edgecolor="#d95f02", label="LOTO (held-out)"
    )
    figure.legend(
        [full_handle, loto_handle],
        ["Full-data (in-sample)", "LOTO (held-out)"],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.935),
        ncol=2,
        frameon=False,
        fontsize=9,
    )
    figure.suptitle(
        "Native mass/growth TMV — directly comparable methods only (lower is better)",
        fontsize=14,
        y=0.985,
    )
    figure.text(
        0.5,
        0.012,
        "Methods without native unnormalised mass, or without equal paired denominators, are omitted rather than imputed.",
        ha="center",
        va="bottom",
        fontsize=8.5,
    )
    figure.tight_layout(rect=(0.0, 0.06, 1.0, 0.88))
    return figure


def _artifact(path: Path, *, role: str) -> dict[str, str]:
    return {"role": role, "path": str(path), "sha256": sha256_file(path)}


def _report_run_contract(
    *,
    output_dir: Path,
    manifest_snapshot: _InputSnapshot,
    paired_snapshot: _InputSnapshot,
    registry_snapshot: _InputSnapshot,
    reporter_snapshot: _InputSnapshot,
    evaluation: dict[str, Any],
    targets: tuple[int, ...],
) -> dict[str, Any]:
    """Return the immutable inputs/options contract bound before any report output."""

    return {
        "schema_version": "1.0.0",
        "status": "bound",
        "design": "matched_loto_vs_full_data_descriptive_report",
        "output_dir": str(output_dir),
        "inputs": {
            "matched_evaluation_manifest": str(manifest_snapshot.path),
            "matched_evaluation_manifest_sha256": manifest_snapshot.sha256,
            "paired_summary_csv": str(paired_snapshot.path),
            "paired_summary_csv_sha256": paired_snapshot.sha256,
            "method_registry": str(registry_snapshot.path),
            "method_registry_sha256": registry_snapshot.sha256,
            "reporter": str(reporter_snapshot.path),
            "reporter_sha256": reporter_snapshot.sha256,
        },
        "evaluator_bound_method_registry": evaluation["method_registry"],
        "targets": list(targets),
        "spaces": list(SPACES),
        "metrics": list(METRICS),
        "reporting_policy": {
            "full_data_is_in_sample": True,
            "comparison_type": "descriptive_paired_gap",
            "cross_space_aggregation": False,
            "overall_score": False,
            "ranking": False,
            "statistical_inference": False,
        },
    }


def report_matched(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = args.matched_manifest.expanduser().resolve()
    paired_path = args.paired_summary.expanduser().resolve()
    registry_path = args.method_registry.expanduser().resolve()
    reporter_path = Path(__file__).resolve()
    output_dir = _absolute_output_path(args.output_dir)
    final_manifest_path = output_dir / "matched_report_manifest.json"
    _assert_output_available(output_dir)

    manifest_snapshot = _snapshot_file(
        manifest_path, label="matched evaluation manifest"
    )
    paired_snapshot = _snapshot_file(paired_path, label="paired summary CSV")
    registry_snapshot = _snapshot_file(registry_path, label="method registry")
    reporter_snapshot = _snapshot_file(reporter_path, label="matched reporter")
    evaluation_payload = _load_json(
        manifest_snapshot, label="matched evaluation manifest"
    )
    registry = _load_registry(registry_snapshot)
    evaluation, raw_to_canonical, targets = _validate_manifest(
        evaluation_payload,
        manifest_snapshot,
        paired_snapshot,
        registry_snapshot,
        registry,
    )
    paired = _validate_paired(
        paired_snapshot,
        evaluation,
        registry,
        raw_to_canonical,
        targets,
    )
    projection_repeats = int(evaluation["projection_repeats"])
    target = _target_summary(paired, registry, projection_repeats)
    method_space = _method_space_summary(target, targets)
    tmv = _tmv_table(paired, registry)
    for frame, label in (
        (target, "target summary"),
        (method_space, "method-space summary"),
        (tmv, "TMV table"),
    ):
        forbidden = [
            column
            for column in frame.columns
            if "rank" in column.casefold() or "overall" in column.casefold()
        ]
        if forbidden:
            raise ReportError(f"{label} contains forbidden columns: {forbidden}")

    tables_dir = output_dir / "tables"
    plots_dir = output_dir / "plots"
    target_path = tables_dir / "matched_method_space_target_summary.csv"
    method_space_path = tables_dir / "matched_method_space_across_targets_summary.csv"
    tmv_path = tables_dir / "matched_tmv_applicable_only.csv"
    claim_path = _claim_output(output_dir)
    run_contract_path = output_dir / "report_run_contract.json"
    run_contract_payload = _report_run_contract(
        output_dir=output_dir,
        manifest_snapshot=manifest_snapshot,
        paired_snapshot=paired_snapshot,
        registry_snapshot=registry_snapshot,
        reporter_snapshot=reporter_snapshot,
        evaluation=evaluation,
        targets=targets,
    )
    _immutable_bytes(
        run_contract_path,
        _strict_json_bytes(run_contract_payload),
        label="report run contract",
    )
    run_contract_sha = sha256_file(run_contract_path)
    _immutable_csv(target_path, target)
    _immutable_csv(method_space_path, method_space)
    _immutable_csv(tmv_path, tmv)

    primary_plots: list[Path] = []
    supplemental_plots: list[Path] = []
    for metric in METRICS:
        for space in SPACES:
            figure = _dumbbell_figure(
                target,
                registry,
                space=space,
                metric=metric,
                targets=targets,
            )
            paths = _save_figure(
                figure,
                plots_dir / f"matched_{metric}_{space}_dumbbell.png",
            )
            if METRICS[metric]["role"] == "primary":
                primary_plots.extend(paths)
            else:
                supplemental_plots.extend(paths)
    tmv_plots = _save_figure(
        _tmv_figure(tmv, targets),
        plots_dir / "matched_tmv_applicable_only_dumbbell.png",
    )

    table_artifacts = [
        _artifact(target_path, role="method_space_target_descriptive_table"),
        _artifact(
            method_space_path, role="method_space_across_targets_descriptive_table"
        ),
        _artifact(tmv_path, role="tmv_applicable_only_table"),
    ]
    primary_artifacts = [
        _artifact(path, role="primary_sliced_w2_dumbbell") for path in primary_plots
    ]
    supplemental_artifacts = [
        _artifact(path, role="supplemental_exact_ot_dumbbell")
        for path in supplemental_plots
    ]
    tmv_artifacts = [
        _artifact(path, role="tmv_applicable_only_dumbbell") for path in tmv_plots
    ]
    manifest = {
        "schema_version": "1.0.0",
        "status": "complete",
        "design": "matched_loto_vs_full_data_descriptive_report",
        "dataset": evaluation.get("dataset"),
        "targets": list(targets),
        "spaces_reported_separately": list(SPACES),
        "methods": [
            {
                "raw_method": raw_method,
                "canonical_method": canonical_method,
                "method": canonical_method,
                "display_name": record["display_name"],
                "spaces": record["spaces"],
                "status": record["status"],
            }
            for raw_method, canonical_method in raw_to_canonical.items()
            for record in [
                next(
                    candidate
                    for candidate in registry
                    if candidate["method"] == canonical_method
                )
            ]
        ],
        "inputs": {
            "report_run_contract": str(run_contract_path),
            "report_run_contract_sha256": run_contract_sha,
            "matched_evaluation_manifest": str(manifest_path),
            "matched_evaluation_manifest_sha256": manifest_snapshot.sha256,
            "paired_summary_csv": str(paired_path),
            "paired_summary_csv_sha256": paired_snapshot.sha256,
            "method_registry": str(registry_path),
            "method_registry_sha256": registry_snapshot.sha256,
            "evaluator_bound_raw_to_canonical": evaluation["method_registry"][
                "raw_to_canonical"
            ],
        },
        "validated_contract": {
            "matched_status_complete": True,
            "targets_validated_from_evaluator_manifest": True,
            "full_data_evaluation_scope": "in_sample",
            "loto_evaluation_scope": "held_out",
            "paired_csv_sha256_verified": True,
            "method_registry_sha256_verified": True,
            "exact_raw_to_canonical_mapping_verified": True,
            "inputs_parsed_from_hashed_byte_snapshots": True,
            "method_space_target_grid_matches_registry": True,
            "state_only_spaces_not_fabricated": True,
            "tmv_direct_comparability_exactly_recomputed": True,
            "tmv_cross_space_method_target_consistency_verified": True,
        },
        "reporting_policy": {
            "primary_metric": "sliced_w2",
            "supplemental_metrics": ["exact_w1", "exact_w2"],
            "pairing_unit": "method + feature space + target",
            "comparison_type": "descriptive_paired_gap",
            "full_data_is_in_sample": True,
            "cross_space_aggregation": False,
            "overall_score": False,
            "ranking": False,
            "statistical_inference": False,
            "projection_repeats": projection_repeats,
            "projection_repeat_interpretation": (
                "random-projection numerical integration variation only; not a "
                "training-seed, biological, or confidence interval"
            ),
            "tmv": "directly comparable native unnormalised mass/growth only",
        },
        "interpretation_limit": (
            "Full-data values are in-sample reconstruction references. LOTO-minus-"
            "full-data differences are descriptive paired gaps, not unbiased "
            "generalization estimates."
        ),
        "n_target_rows": int(len(target)),
        "n_method_space_rows": int(len(method_space)),
        "n_tmv_applicable_rows": int(len(tmv)),
        "tables": table_artifacts,
        "primary_plots": primary_artifacts,
        "supplemental_plots": supplemental_artifacts,
        "tmv_plots": tmv_artifacts,
        "reporter": str(reporter_path),
        "reporter_sha256": reporter_snapshot.sha256,
        "software_versions": {
            "python": platform.python_version(),
            "numpy": str(np.__version__),
            "pandas": str(pd.__version__),
            "matplotlib": str(matplotlib.__version__),
        },
    }
    artifact_records = [
        *table_artifacts,
        *primary_artifacts,
        *supplemental_artifacts,
        *tmv_artifacts,
    ]
    artifact_paths = [
        run_contract_path,
        *(Path(record["path"]) for record in artifact_records),
    ]
    _assert_owned_output(output_dir, claim_path, artifact_paths)
    if sha256_file(run_contract_path) != run_contract_sha:
        raise ReportError("report run contract changed before publication")
    for record in artifact_records:
        if sha256_file(Path(record["path"])) != record["sha256"]:
            raise ReportError(
                f"report artifact changed before publication: {record['path']}"
            )
    for label, snapshot in (
        ("matched evaluation manifest", manifest_snapshot),
        ("paired summary CSV", paired_snapshot),
        ("method registry", registry_snapshot),
        ("matched reporter", reporter_snapshot),
    ):
        if sha256_file(snapshot.path) != snapshot.sha256:
            raise ReportError(
                f"{label} changed after its hashed byte snapshot and before publication"
            )
    manifest["validated_contract"]["input_snapshots_rehashed_before_publication"] = True
    _write_final_manifest(final_manifest_path, manifest)
    claim_path.unlink()
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matched-manifest", type=Path, required=True)
    parser.add_argument("--paired-summary", type=Path, required=True)
    parser.add_argument("--method-registry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = report_matched(args)
    except (ReportError, FileNotFoundError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
