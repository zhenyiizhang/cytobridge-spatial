"""Resolve the ligand-receptor databases used by formal workflow presets."""

from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Iterable
from importlib import resources
from pathlib import Path
import re
from typing import Any


FORMAL_GRAPH_DATABASES = {
    "zebrafish": "CellChatDB.ligrec.zebrafish.csv",
    "mosta": "CellChatDB.ligrec.mouse.csv",
    "arista": "CellChatDB.ligrec.human.csv",
    "admouse": "CellChatDB.ligrec.mouse.csv",
    # Gallus gallus has no CellChatDB release.  The chicken-heart workflow uses
    # the human collection only as an explicitly labelled conserved-symbol
    # prior; exact feature matching and coverage are recorded by preprocessing.
    "chicken_heart": "CellChatDB.ligrec.human.csv",
}

_LIGAND_COLUMN_NAMES = ("ligand", "ligand_symbol", "source", "gene_a", "0")
_RECEPTOR_COLUMN_NAMES = (
    "receptor",
    "receptor_symbol",
    "target",
    "gene_b",
    "1",
)


def bundled_graph_database_path(
    dataset_name: str,
    *,
    filename: str | None = None,
) -> Path:
    """Return the wheel-bundled formal graph database for a dataset."""

    selected = filename or FORMAL_GRAPH_DATABASES.get(str(dataset_name))
    if selected is None:
        known = ", ".join(sorted(FORMAL_GRAPH_DATABASES))
        raise KeyError(
            f"No bundled graph database is declared for {dataset_name!r}. "
            f"Known formal datasets: {known}."
        )
    resource = resources.files("CytoBridge").joinpath("workflow_databases", selected)
    if not resource.is_file():
        raise FileNotFoundError(
            f"Bundled graph database {selected!r} is missing from CytoBridge."
        )
    return Path(str(resource))


def resolve_graph_database(
    dataset_name: str,
    database_path: str | Path | None = None,
    *,
    bundled_filename: str | None = None,
) -> Path:
    """Return a custom database override or the dataset's bundled formal one."""

    if database_path is not None:
        resolved = Path(database_path).expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"Graph database not found: {resolved}")
        print(f"Using custom interaction-graph database: {resolved}")
        return resolved

    bundled = bundled_graph_database_path(
        dataset_name,
        filename=bundled_filename,
    )
    print(f"Using bundled formal interaction-graph database: {bundled}")
    return bundled


def _ligand_receptor_column_indices(header: list[str]) -> tuple[int, int]:
    """Resolve ligand/receptor columns without depending on graph construction."""

    normalized = [str(value).strip().casefold() for value in header]
    ligand_index = next(
        (normalized.index(name) for name in _LIGAND_COLUMN_NAMES if name in normalized),
        None,
    )
    receptor_index = next(
        (
            normalized.index(name)
            for name in _RECEPTOR_COLUMN_NAMES
            if name in normalized
        ),
        None,
    )
    if ligand_index is not None and receptor_index is not None:
        return ligand_index, receptor_index

    usable = [
        index
        for index, name in enumerate(normalized)
        if name and not name.startswith("unnamed")
    ]
    if len(usable) >= 2:
        return usable[0], usable[1]
    raise ValueError(
        "Could not identify ligand/receptor columns from graph database header "
        f"{header}."
    )


def _complex_subunits(value: str) -> tuple[str, ...]:
    """Parse an underscore-delimited CellChat ligand/receptor complex."""

    return tuple(part.strip() for part in str(value).split("_") if part.strip())


def feature_symbol_candidates(value: object) -> tuple[tuple[str, str | None], ...]:
    """Parse plain or ``symbol[tag]|symbol[tag]|ID`` feature names."""

    candidates: list[tuple[str, str | None]] = []
    for raw in str(value).split("|"):
        token = raw.strip()
        if not token or token.casefold() == "nan" or token.upper().startswith("AMEX"):
            continue
        match = re.match(r"^(.*?)(?:\[([^\]]+)\])?$", token)
        if match is None:
            continue
        symbol = match.group(1).strip()
        species = match.group(2)
        if symbol:
            candidates.append(
                (symbol, species.casefold() if species is not None else None)
            )
    return tuple(candidates)


def selected_feature_symbol(
    value: object,
    *,
    preferred_species_tag: str | None = None,
) -> str | None:
    """Select one exact symbol from a plain or species-tagged feature name.

    When a preferred species is requested, a compound feature that contains
    species tags is usable only when that species is present.  Falling back to
    a different tagged species would silently map (for example) a newt symbol
    into the human interaction database.  Plain, untagged feature names remain
    valid for every preset.
    """

    candidates = feature_symbol_candidates(value)
    preferred = (
        str(preferred_species_tag).strip().casefold() if preferred_species_tag else None
    )
    if preferred is not None:
        selected = next(
            (symbol for symbol, species in candidates if species == preferred),
            None,
        )
        if selected is not None:
            return selected
        if any(species is not None for _, species in candidates):
            return None
    if candidates:
        return candidates[0][0]
    fallback = str(value).strip()
    return fallback or None


def match_graph_database_features(
    database_path: str | Path,
    var_names: Iterable[object],
    *,
    preferred_species_tag: str | None = None,
) -> tuple[tuple[str, ...], dict[str, Any]]:
    """Match database LR subunits to an input feature universe.

    Matching is exact after whitespace stripping and case folding; aliases and
    substring matches are intentionally unsupported.  A database subunit is
    returned only when it maps to exactly one input feature position.  Missing
    and case-insensitively ambiguous symbols are reported as coverage rather
    than raising, so a partially covered species database can still be used.

    Returns
    -------
    matched_features
        Input ``var_names`` spellings, suitable for
        ``AlignConfig.required_latent_features``.
    coverage
        H5AD-safe counts and symbol lists describing the exact match contract.
    """

    resolved_database = Path(database_path).expanduser().resolve()
    if not resolved_database.is_file():
        raise FileNotFoundError(f"Graph database not found: {resolved_database}")

    database_subunits: dict[str, str] = {}
    row_count = 0
    with resolved_database.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError("Ligand-receptor database is empty.") from exc
        ligand_index, receptor_index = _ligand_receptor_column_indices(header)
        required_width = max(ligand_index, receptor_index) + 1
        for row in reader:
            row_count += 1
            if len(row) < required_width:
                continue
            for value in (row[ligand_index], row[receptor_index]):
                stripped = str(value).strip()
                if not stripped or stripped.casefold() == "nan":
                    continue
                for subunit in _complex_subunits(stripped):
                    database_subunits.setdefault(subunit.casefold(), subunit)

    feature_lookup: dict[str, list[str]] = defaultdict(list)
    for value in var_names:
        feature = str(value)
        symbol = selected_feature_symbol(
            feature,
            preferred_species_tag=preferred_species_tag,
        )
        if symbol is None:
            continue
        feature_lookup[symbol.casefold()].append(feature)

    matched: list[str] = []
    missing: list[str] = []
    ambiguous: list[str] = []
    for normalized, database_symbol in database_subunits.items():
        candidates = feature_lookup.get(normalized, [])
        if len(candidates) == 1:
            matched.append(candidates[0])
        elif not candidates:
            missing.append(database_symbol)
        else:
            ambiguous.append(database_symbol)

    total = len(database_subunits)
    coverage: dict[str, Any] = {
        "matching_policy": "selected_symbol_exact_case_insensitive_unique",
        "preferred_species_tag": preferred_species_tag,
        "database_path": str(resolved_database),
        "n_database_rows": int(row_count),
        "n_unique_database_subunits": int(total),
        "n_matched_features": int(len(matched)),
        "n_missing_database_subunits": int(len(missing)),
        "n_ambiguous_database_subunits": int(len(ambiguous)),
        "coverage_fraction": float(len(matched) / total) if total else 0.0,
        "matched_features": matched,
        "missing_database_subunits": missing,
        "ambiguous_database_subunits": ambiguous,
    }
    return tuple(matched), coverage
