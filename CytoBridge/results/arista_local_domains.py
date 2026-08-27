"""Processed results for ARISTA local interaction domains."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ._io import read_json, require_files, resolve_results_dir


DOMAIN_ORDER = ("N1_sfrpEGC_VLMC", "N2_reaEGC_wntEGC")
DOMAIN_LABELS = {
    "N1_sfrpEGC_VLMC": "sfrpEGC-VLMC",
    "N2_reaEGC_wntEGC": "reaEGC-wntEGC",
}
DOMAIN_COUNTS = {"N1_sfrpEGC_VLMC": 203, "N2_reaEGC_wntEGC": 77}
DISPLAY_PROGRAMS = {
    "N1_sfrpEGC_VLMC": ("AGRN", "LAMININ", "TENASCIN", "FGF", "THBS"),
    "N2_reaEGC_wntEGC": ("GRN", "L1CAM", "NRXN", "SEMA3", "FN1"),
}

_FILES = (
    "roi_assignments.csv",
    "domain_metadata.csv",
    "celltype_edges.csv",
    "attention_null.csv",
    "pathway_null.csv",
    "lr_pair_null.csv.gz",
    "manifest.json",
)
_ROI_COLUMNS = (
    "time",
    "cell_index",
    "celltype",
    "paper_x",
    "paper_y",
    "cosine_full_vs_interaction",
    "in_roi",
    "two_niche_region",
)
_METADATA_COLUMNS = (
    "niche",
    "n_cells",
    "n_internal_edges",
    "attention_sum",
    "attention_per_cell",
    "upper_quantile",
    "cosine_threshold",
    "minimum_size",
    "interaction_cutoff",
)
_EDGE_COLUMNS = (
    "niche",
    "sender",
    "receiver",
    "n_edges",
    "attention_sum",
    "n_region_cells",
    "attention_per_region_cell",
)
_ATTENTION_COLUMNS = (
    "module",
    "n_cells",
    "n_celltypes",
    "celltype_counts",
    "observed_internal_edges",
    "null_internal_edges_mean",
    "null_internal_edges_sd",
    "internal_edges_empirical_p_greater",
    "observed_attention_sum",
    "null_attention_sum_mean",
    "null_attention_sum_sd",
    "attention_sum_empirical_p_greater",
    "observed_attention_per_cell",
    "null_attention_per_cell_mean",
    "null_attention_per_cell_sd",
    "attention_per_cell_empirical_p_greater",
    "n_permutations",
    "null_sampling",
)
_PATHWAY_COLUMNS = (
    "module",
    "pathway",
    "observed_lr_score",
    "null_mean",
    "null_sd",
    "empirical_p_greater",
    "n_permutations",
    "fold_over_null_mean",
    "adjusted_p_value",
)
_PAIR_COLUMNS = (
    "niche",
    "ligand",
    "receptor",
    "pair",
    "pathway",
    "interaction_class",
    "observed_pair_score",
    "null_mean",
    "null_sd",
    "fold_over_null_mean",
    "log2_fold_over_null",
    "empirical_p_greater",
    "adjusted_p_value",
    "dominant_sender",
    "dominant_receiver",
    "dominant_pair_score",
    "dominant_contribution_fraction",
    "n_permutations",
)


@dataclass(frozen=True)
class AristaLocalDomainData:
    """Compact inputs for the ARISTA local-domain figure."""

    source_dir: Path
    manifest: dict[str, Any]
    roi_assignments: pd.DataFrame
    domain_metadata: pd.DataFrame
    celltype_edges: pd.DataFrame
    attention_null: pd.DataFrame
    pathway_null: pd.DataFrame
    lr_pair_null: pd.DataFrame


@dataclass(frozen=True)
class AristaLocalDomainPanels:
    """Calculated tables drawn in the four figure panels."""

    attention: pd.DataFrame
    edge_structure: pd.DataFrame
    pathways: pd.DataFrame
    lr_axes: pd.DataFrame


def _require_columns(
    table: pd.DataFrame,
    columns: tuple[str, ...],
    source: Path,
) -> pd.DataFrame:
    missing = sorted(set(columns).difference(table.columns))
    if missing:
        raise ValueError(f"{source} is missing columns: {missing}")
    return table.copy()


def _numeric(
    table: pd.DataFrame,
    columns: tuple[str, ...],
    source: Path,
) -> np.ndarray:
    values = table.loc[:, columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError(f"{source} contains non-finite values in {list(columns)}")
    return values


def _bh_adjust(p_values: pd.Series | np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranked = values[order] * values.size / np.arange(1, values.size + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted = np.empty(values.size, dtype=float)
    adjusted[order] = np.minimum(ranked, 1.0)
    return adjusted


def _validate_manifest(manifest: dict[str, Any], source: Path) -> None:
    if manifest.get("schema_version") != 1:
        raise ValueError(f"{source} has an unsupported schema version")
    if manifest.get("analysis") != "arista_local_domains":
        raise ValueError(f"{source} does not describe ARISTA local domains")
    domains = manifest.get("domains", {})
    if tuple(domains.get("order", ())) != DOMAIN_ORDER:
        raise ValueError(f"{source} has an unexpected domain order")
    if domains.get("labels") != DOMAIN_LABELS:
        raise ValueError(f"{source} has unexpected domain labels")
    if domains.get("cell_counts") != DOMAIN_COUNTS:
        raise ValueError(f"{source} has unexpected domain cell counts")
    programs = (
        manifest.get("calculation", {}).get("pathways", {}).get("display_programs", {})
    )
    if {name: tuple(value) for name, value in programs.items()} != DISPLAY_PROGRAMS:
        raise ValueError(f"{source} has an unexpected pathway display roster")
    calculation = manifest.get("calculation", {})
    if calculation.get("attention", {}).get("permutations") != 9999:
        raise ValueError(f"{source} has an unexpected attention-null size")
    if calculation.get("pathways", {}).get("permutations") != 1999:
        raise ValueError(f"{source} has an unexpected pathway-null size")
    pairs = calculation.get("lr_pairs", {})
    if pairs.get("permutations") != 1999 or pairs.get("tests_per_domain") != 531:
        raise ValueError(f"{source} has an unexpected ligand-receptor test count")


def _validate_roi(table: pd.DataFrame, source: Path) -> pd.DataFrame:
    result = _require_columns(table, _ROI_COLUMNS, source)
    if len(result) != 1454:
        raise ValueError(f"{source} must contain 1,454 ROI cells")
    if result["cell_index"].duplicated().any():
        raise ValueError(f"{source} contains duplicate cell indices")
    values = _numeric(
        result,
        ("time", "cell_index", "paper_x", "paper_y", "cosine_full_vs_interaction"),
        source,
    )
    cosine = values[:, -1]
    if (cosine < -1.000001).any() or (cosine > 1.000001).any():
        raise ValueError(f"{source} contains cosine values outside [-1, 1]")
    in_roi = result["in_roi"]
    if not pd.api.types.is_bool_dtype(in_roi):
        in_roi = in_roi.astype(str).str.lower().map({"true": True, "false": False})
    if in_roi.isna().any() or not in_roi.all():
        raise ValueError(f"{source} contains cells outside the ROI")
    result["in_roi"] = in_roi.astype(bool)
    if result["time"].nunique(dropna=False) != 1:
        raise ValueError(f"{source} must contain one time point")
    unknown_domains = sorted(
        set(result["two_niche_region"].dropna()).difference(DOMAIN_ORDER)
    )
    if unknown_domains:
        raise ValueError(f"{source} contains unexpected domain assignments")
    observed_counts = result["two_niche_region"].value_counts().to_dict()
    if {
        name: int(observed_counts.get(name, 0)) for name in DOMAIN_ORDER
    } != DOMAIN_COUNTS:
        raise ValueError(f"{source} has unexpected domain assignments")
    return result


def _validate_metadata(table: pd.DataFrame, source: Path) -> pd.DataFrame:
    result = _require_columns(table, _METADATA_COLUMNS, source)
    if len(result) != 2 or result["niche"].duplicated().any():
        raise ValueError(f"{source} must contain one row per domain")
    if set(result["niche"]) != set(DOMAIN_ORDER):
        raise ValueError(f"{source} has unexpected domains")
    _numeric(result, _METADATA_COLUMNS[1:], source)
    counts = dict(zip(result["niche"], result["n_cells"].astype(int)))
    if counts != DOMAIN_COUNTS:
        raise ValueError(f"{source} has unexpected domain cell counts")
    return result


def _validate_edges(
    table: pd.DataFrame,
    metadata: pd.DataFrame,
    source: Path,
) -> pd.DataFrame:
    result = _require_columns(table, _EDGE_COLUMNS, source)
    if len(result) != 19:
        raise ValueError(f"{source} must contain 19 sender-receiver rows")
    if result.duplicated(["niche", "sender", "receiver"]).any():
        raise ValueError(f"{source} contains duplicate sender-receiver rows")
    if set(result["niche"]) != set(DOMAIN_ORDER):
        raise ValueError(f"{source} has unexpected domains")
    _numeric(result, _EDGE_COLUMNS[3:], source)
    totals = result.groupby("niche", sort=False)[["n_edges", "attention_sum"]].sum()
    expected = metadata.set_index("niche")[["n_internal_edges", "attention_sum"]]
    totals = totals.loc[list(DOMAIN_ORDER)]
    expected = expected.loc[list(DOMAIN_ORDER)]
    if not np.array_equal(
        totals["n_edges"].to_numpy(int), expected["n_internal_edges"].to_numpy(int)
    ) or not np.allclose(
        totals["attention_sum"], expected["attention_sum"], rtol=1e-12, atol=1e-12
    ):
        raise ValueError(f"{source} does not match the domain metadata totals")
    return result


def _validate_attention(
    table: pd.DataFrame,
    metadata: pd.DataFrame,
    source: Path,
) -> pd.DataFrame:
    result = _require_columns(table, _ATTENTION_COLUMNS, source)
    if len(result) != 2 or result["module"].duplicated().any():
        raise ValueError(f"{source} must contain one row per domain")
    if set(result["module"]) != set(DOMAIN_ORDER):
        raise ValueError(f"{source} has unexpected domains")
    numeric_columns = tuple(
        name
        for name in _ATTENTION_COLUMNS
        if name not in {"module", "celltype_counts", "null_sampling"}
    )
    _numeric(result, numeric_columns, source)
    if not result["n_permutations"].eq(9999).all():
        raise ValueError(f"{source} must use 9,999 permutations")
    observed = result.set_index("module").loc[list(DOMAIN_ORDER)]
    expected = metadata.set_index("niche").loc[list(DOMAIN_ORDER)]
    comparisons = (
        ("n_cells", "n_cells"),
        ("observed_internal_edges", "n_internal_edges"),
        ("observed_attention_sum", "attention_sum"),
        ("observed_attention_per_cell", "attention_per_cell"),
    )
    for observed_name, expected_name in comparisons:
        if not np.allclose(
            observed[observed_name], expected[expected_name], rtol=1e-12, atol=1e-12
        ):
            raise ValueError(
                f"{source} does not match domain metadata: {observed_name}"
            )
    return result


def _validate_null_table(
    table: pd.DataFrame,
    *,
    group_column: str,
    columns: tuple[str, ...],
    rows_per_domain: int,
    source: Path,
) -> pd.DataFrame:
    result = _require_columns(table, columns, source)
    counts = result.groupby(group_column, sort=False).size().to_dict()
    if counts != {name: rows_per_domain for name in DOMAIN_ORDER}:
        raise ValueError(f"{source} has an unexpected row count by domain")
    if result.duplicated(
        [group_column, "pathway"] + (["pair"] if "pair" in result else [])
    ).any():
        raise ValueError(f"{source} contains duplicate result rows")
    required_numeric = [
        "observed_lr_score" if group_column == "module" else "observed_pair_score",
        "null_mean",
        "null_sd",
        "empirical_p_greater",
        "adjusted_p_value",
        "n_permutations",
    ]
    if group_column == "niche":
        required_numeric.extend(
            ["dominant_pair_score", "dominant_contribution_fraction"]
        )
    _numeric(result, tuple(required_numeric), source)
    if not result["n_permutations"].eq(1999).all():
        raise ValueError(f"{source} must use 1,999 permutations")
    probabilities = _numeric(
        result,
        ("empirical_p_greater", "adjusted_p_value"),
        source,
    )
    if (probabilities < 0).any() or (probabilities > 1).any():
        raise ValueError(f"{source} contains probabilities outside [0, 1]")
    for domain, subset in result.groupby(group_column, sort=False):
        expected = _bh_adjust(subset["empirical_p_greater"])
        if not np.allclose(
            expected, subset["adjusted_p_value"], rtol=1e-12, atol=1e-15
        ):
            raise ValueError(
                f"{source} has inconsistent adjusted p-values for {domain}"
            )
    return result


def load_arista_local_domains(
    results_dir: str | Path | None = None,
) -> AristaLocalDomainData:
    """Load the compact tables used for the ARISTA local-domain figure."""

    source_dir = resolve_results_dir(results_dir, slug="arista_local_domains")
    paths = require_files(source_dir, _FILES)
    manifest = read_json(paths["manifest.json"])
    _validate_manifest(manifest, paths["manifest.json"])
    roi = _validate_roi(
        pd.read_csv(paths["roi_assignments.csv"]), paths["roi_assignments.csv"]
    )
    metadata = _validate_metadata(
        pd.read_csv(paths["domain_metadata.csv"]), paths["domain_metadata.csv"]
    )
    edges = _validate_edges(
        pd.read_csv(paths["celltype_edges.csv"]), metadata, paths["celltype_edges.csv"]
    )
    attention = _validate_attention(
        pd.read_csv(paths["attention_null.csv"]), metadata, paths["attention_null.csv"]
    )
    pathways = _validate_null_table(
        pd.read_csv(paths["pathway_null.csv"], float_precision="round_trip"),
        group_column="module",
        columns=_PATHWAY_COLUMNS,
        rows_per_domain=80,
        source=paths["pathway_null.csv"],
    )
    pairs = _validate_null_table(
        pd.read_csv(paths["lr_pair_null.csv.gz"], float_precision="round_trip"),
        group_column="niche",
        columns=_PAIR_COLUMNS,
        rows_per_domain=531,
        source=paths["lr_pair_null.csv.gz"],
    )
    return AristaLocalDomainData(
        source_dir=source_dir,
        manifest=manifest,
        roi_assignments=roi,
        domain_metadata=metadata,
        celltype_edges=edges,
        attention_null=attention,
        pathway_null=pathways,
        lr_pair_null=pairs,
    )


def _edge_structure(edges: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for domain in DOMAIN_ORDER:
        table = edges.loc[edges["niche"].eq(domain)].copy()
        total = float(table["attention_sum"].sum())
        if domain == DOMAIN_ORDER[0]:
            definitions = (
                (
                    "sfrpEGC→sfrpEGC",
                    table["sender"].eq("sfrpEGC") & table["receiver"].eq("sfrpEGC"),
                ),
                (
                    "VLMC ↔ sfrpEGC",
                    (table["sender"].eq("VLMC") & table["receiver"].eq("sfrpEGC"))
                    | (table["sender"].eq("sfrpEGC") & table["receiver"].eq("VLMC")),
                ),
            )
        else:
            definitions = (
                (
                    "wntEGC→wntEGC",
                    table["sender"].eq("wntEGC") & table["receiver"].eq("wntEGC"),
                ),
                (
                    "reaEGC→reaEGC",
                    table["sender"].eq("reaEGC") & table["receiver"].eq("reaEGC"),
                ),
                (
                    "reaEGC ↔ wntEGC",
                    (table["sender"].eq("reaEGC") & table["receiver"].eq("wntEGC"))
                    | (table["sender"].eq("wntEGC") & table["receiver"].eq("reaEGC")),
                ),
            )
        used = np.zeros(len(table), dtype=bool)
        for label, mask in definitions:
            selected = mask.to_numpy(bool)
            used |= selected
            attention = float(table.loc[selected, "attention_sum"].sum())
            rows.append(
                {
                    "niche": domain,
                    "edge_class": label,
                    "attention_sum": attention,
                    "attention_percent": 100.0 * attention / total,
                }
            )
        if domain == DOMAIN_ORDER[0]:
            attention = float(table.loc[~used, "attention_sum"].sum())
            rows.append(
                {
                    "niche": domain,
                    "edge_class": "Other selected edges",
                    "attention_sum": attention,
                    "attention_percent": 100.0 * attention / total,
                }
            )
    return pd.DataFrame(rows)


def _displayed_pathways(pathways: pd.DataFrame) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for domain in DOMAIN_ORDER:
        programs = DISPLAY_PROGRAMS[domain]
        table = pathways.loc[
            pathways["module"].eq(domain) & pathways["pathway"].isin(programs)
        ].copy()
        table["pathway"] = pd.Categorical(table["pathway"], programs, ordered=True)
        table = table.sort_values("pathway", kind="mergesort")
        if len(table) != len(programs) or not table["adjusted_p_value"].lt(0.05).all():
            raise ValueError(f"Displayed pathway rows are incomplete for {domain}")
        table["log2_fold_over_null"] = np.log2(table["fold_over_null_mean"])
        parts.append(table)
    return pd.concat(parts, ignore_index=True)


def _displayed_lr_axes(pairs: pd.DataFrame) -> pd.DataFrame:
    selected: list[pd.DataFrame] = []
    for domain in DOMAIN_ORDER:
        for pathway in DISPLAY_PROGRAMS[domain]:
            candidates = pairs.loc[
                pairs["niche"].eq(domain)
                & pairs["pathway"].eq(pathway)
                & pairs["adjusted_p_value"].lt(0.05)
                & pairs["observed_pair_score"].gt(0)
            ].copy()
            candidates = candidates.sort_values(
                ["adjusted_p_value", "observed_pair_score", "fold_over_null_mean"],
                ascending=[True, False, False],
                kind="mergesort",
            )
            if candidates.empty:
                raise ValueError(
                    f"No displayed ligand-receptor pair for {domain}: {pathway}"
                )
            selected.append(candidates.iloc[[0]])
    result = pd.concat(selected, ignore_index=True)
    n1 = result.loc[result["niche"].eq(DOMAIN_ORDER[0])]
    if set(n1["pathway"]) != set(DISPLAY_PROGRAMS[DOMAIN_ORDER[0]]):
        raise ValueError("The N1 ligand-receptor axes do not match the pathway roster")
    return result


def calculate_arista_local_domain_panels(
    data: AristaLocalDomainData,
) -> AristaLocalDomainPanels:
    """Calculate the compact tables drawn in panels b, c, and d."""

    attention = data.attention_null.loc[
        :,
        [
            "module",
            "observed_attention_per_cell",
            "null_attention_per_cell_mean",
            "null_attention_per_cell_sd",
            "attention_per_cell_empirical_p_greater",
        ],
    ].rename(
        columns={
            "module": "niche",
            "null_attention_per_cell_mean": "null_mean",
            "null_attention_per_cell_sd": "null_sd",
            "attention_per_cell_empirical_p_greater": "empirical_p",
        }
    )
    attention["fold_over_null"] = (
        attention["observed_attention_per_cell"] / attention["null_mean"]
    )
    attention = attention.loc[
        :,
        [
            "niche",
            "observed_attention_per_cell",
            "null_mean",
            "null_sd",
            "fold_over_null",
            "empirical_p",
        ],
    ]
    return AristaLocalDomainPanels(
        attention=attention.reset_index(drop=True),
        edge_structure=_edge_structure(data.celltype_edges),
        pathways=_displayed_pathways(data.pathway_null),
        lr_axes=_displayed_lr_axes(data.lr_pair_null),
    )


def write_arista_local_domain_tables(
    panels: AristaLocalDomainPanels,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write the four calculated panel tables."""

    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    tables = {
        "attention": ("panel_b_attention_matched_null.csv", panels.attention),
        "edge_structure": (
            "panel_b_selected_edge_structure.csv",
            panels.edge_structure,
        ),
        "pathways": ("panel_c_lr_pathway_enrichment.csv", panels.pathways),
        "lr_axes": ("panel_d_candidate_lr_axes.csv", panels.lr_axes),
    }
    paths: dict[str, Path] = {}
    for name, (filename, table) in tables.items():
        path = output / filename
        table.to_csv(path, index=False)
        paths[name] = path
    return paths


def plot_arista_local_domains(
    data: AristaLocalDomainData,
    output_dir: str | Path,
    panels: AristaLocalDomainPanels | None = None,
) -> tuple[Path, Path]:
    """Render the ARISTA local-domain figure as PDF and PNG."""

    from ._arista_local_domains_plot import render_arista_local_domains

    calculated = panels or calculate_arista_local_domain_panels(data)
    return render_arista_local_domains(data, calculated, output_dir)
