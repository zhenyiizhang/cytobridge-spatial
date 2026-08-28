#!/usr/bin/env python3
"""Build the AGIST velocity-cosine breakdown by time and state partition.

The two input archives contain row-matched velocity components.  The script
calculates cosine agreement for spatial and gene-state velocity, summarizes it
by time and state cluster, and writes the tables and figure used for S2.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import zipfile

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT / "results/agist_velocity_time_cluster_breakdown_20260811"
DEFAULT_ARCHIVE_A = DEFAULT_OUTPUT_DIR / "inputs/ebfb921b-884a-4aad-b4e3-99ae9b8a512d.zip"
DEFAULT_ARCHIVE_B = DEFAULT_OUTPUT_DIR / "inputs/ed999aeb-bfe3-4426-8c04-3ee56f094b81.zip"
DEFAULT_DATA = ROOT / "data/mouse_brain_simulation.csv"
DEFAULT_CLUSTERS = DEFAULT_OUTPUT_DIR / "inputs/agist_state_cluster_assignments.csv"
DEFAULT_CLUSTER_DIAGNOSTICS = DEFAULT_OUTPUT_DIR / "inputs/cluster_selection_diagnostics.csv"
DEFAULT_STYLE = DEFAULT_OUTPUT_DIR / "inputs/cytobridge-paper.mplstyle"

COMPONENT_FILES = (
    "simulation_gradients_np_gt.npy",
    "simulation_gradients_np_retain.npy",
    "simulation_gradients_score.npy",
)

TEAL = "#07838B"
REFERENCE = "#59616A"
IQR_COLOR = "#C7CDD1"
GRID = "#D7DDE2"
TEXT = "#24313A"
HEADING = "#102A43"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_velocity_archive(path: Path) -> tuple[np.ndarray, dict[str, object]]:
    components: list[np.ndarray] = []
    component_manifest: dict[str, object] = {}
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        missing = set(COMPONENT_FILES).difference(names)
        if missing:
            raise FileNotFoundError(f"{path} is missing {sorted(missing)}")
        for name in COMPONENT_FILES:
            raw = archive.read(name)
            value = np.load(io.BytesIO(raw))
            if value.ndim != 2 or value.shape[1] < 52:
                raise ValueError(f"Unexpected velocity shape for {path}:{name}: {value.shape}")
            components.append(value)
            component_manifest[name] = {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
    shapes = {value.shape for value in components}
    if len(shapes) != 1:
        raise ValueError(f"Component shapes disagree in {path}: {sorted(shapes)}")
    return sum(components), component_manifest


def row_cosine(a: np.ndarray, b: np.ndarray, columns: slice, eps: float = 1e-12) -> np.ndarray:
    a = a[:, columns]
    b = b[:, columns]
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    values = np.full(a.shape[0], np.nan, dtype=float)
    valid = denom > eps
    values[valid] = np.sum(a[valid] * b[valid], axis=1) / denom[valid]
    return values


def summarize(values: pd.Series) -> pd.Series:
    finite = values[np.isfinite(values.to_numpy(float))].to_numpy(float)
    n = int(len(finite))
    if n == 0:
        return pd.Series(
            {
                "n": 0,
                "mean": np.nan,
                "std": np.nan,
                "se": np.nan,
                "ci95_low": np.nan,
                "ci95_high": np.nan,
                "median": np.nan,
                "q25": np.nan,
                "q75": np.nan,
            }
        )
    std = float(np.std(finite, ddof=1)) if n > 1 else 0.0
    se = std / np.sqrt(n)
    mean = float(np.mean(finite))
    return pd.Series(
        {
            "n": n,
            "mean": mean,
            "std": std,
            "se": se,
            "ci95_low": mean - 1.96 * se,
            "ci95_high": mean + 1.96 * se,
            "median": float(np.median(finite)),
            "q25": float(np.quantile(finite, 0.25)),
            "q75": float(np.quantile(finite, 0.75)),
        }
    )


def grouped_summary(per_cell: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    long = per_cell.melt(
        id_vars=["row_index", "time", "state_cluster"],
        value_vars=["physical_cosine", "gene_cosine"],
        var_name="velocity_space",
        value_name="cosine",
    )
    long["velocity_space"] = long["velocity_space"].map(
        {"physical_cosine": "physical", "gene_cosine": "gene"}
    )
    result = (
        long.groupby([*group_columns, "velocity_space"], observed=True)["cosine"]
        .apply(summarize)
        .unstack()
        .reset_index()
    )
    return result


def configure_style(style_path: Path) -> None:
    if style_path.is_file():
        plt.style.use(style_path)
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "text.color": TEXT,
            "axes.labelcolor": TEXT,
            "axes.titlecolor": TEXT,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
        }
    )


def style_axis(ax: mpl.axes.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.45, alpha=0.65, zorder=0)
    ax.set_axisbelow(True)
    ax.axhline(1.0, color="#A7ADB2", linewidth=0.75, linestyle=(0, (3, 2)), zorder=1)
    ax.tick_params(length=2.5, width=0.6)


def draw_summary_panel(
    ax: mpl.axes.Axes,
    table: pd.DataFrame,
    category_column: str,
    categories: list[object],
    labels: list[str],
) -> None:
    sub = table.set_index(category_column).loc[categories].reset_index()
    x = np.arange(len(sub), dtype=float)
    mean = sub["mean"].to_numpy(float)
    median = sub["median"].to_numpy(float)
    ci_low = sub["ci95_low"].to_numpy(float)
    ci_high = sub["ci95_high"].to_numpy(float)
    q25 = sub["q25"].to_numpy(float)
    q75 = sub["q75"].to_numpy(float)

    ax.vlines(x, q25, q75, color=IQR_COLOR, linewidth=6.5, capstyle="round", zorder=2)
    ax.scatter(
        x,
        median,
        s=22,
        marker="D",
        facecolor="white",
        edgecolor=REFERENCE,
        linewidth=1.0,
        zorder=4,
    )
    ax.plot(x, mean, color=TEAL, linewidth=1.4, zorder=3)
    ax.errorbar(
        x,
        mean,
        yerr=np.vstack([mean - ci_low, ci_high - mean]),
        fmt="o",
        color=TEAL,
        markerfacecolor=TEAL,
        markeredgecolor=TEAL,
        markersize=4.5,
        capsize=2.6,
        elinewidth=1.0,
        capthick=1.0,
        zorder=5,
    )
    ax.set_xticks(x, labels)
    ax.set_xlim(-0.38, len(sub) - 0.62)
    ax.set_ylim(0.75, 1.012)
    ax.set_yticks(np.arange(0.75, 1.001, 0.05))
    style_axis(ax)


def add_panel_heading(fig: plt.Figure, ax: mpl.axes.Axes, label: str, title: str) -> None:
    bbox = ax.get_position()
    y = bbox.y1 + 0.035
    fig.text(
        bbox.x0,
        y,
        label,
        fontsize=14,
        fontweight="bold",
        color=HEADING,
        ha="left",
        va="center",
    )
    fig.text(
        bbox.x0 + 0.038,
        y,
        title,
        fontsize=12,
        fontweight="bold",
        color=HEADING,
        ha="left",
        va="center",
    )


def build_figure(
    by_time: pd.DataFrame,
    by_cluster: pd.DataFrame,
    pdf_path: Path,
    png_path: Path,
    style_path: Path,
) -> None:
    configure_style(style_path)
    fig, axes = plt.subplots(2, 2, figsize=(8.27, 7.25), sharey=True)
    fig.subplots_adjust(left=0.105, right=0.975, bottom=0.105, top=0.835, hspace=0.56, wspace=0.25)

    time_categories = sorted(by_time["time"].unique().tolist())
    time_counts = (
        by_time[by_time["velocity_space"] == "physical"]
        .set_index("time")["n"]
        .astype(int)
    )
    time_labels = [f"t = {time:g}\nn = {time_counts.loc[time]:,}" for time in time_categories]

    cluster_categories = sorted(by_cluster["state_cluster"].unique().tolist())
    cluster_counts = (
        by_cluster[by_cluster["velocity_space"] == "physical"]
        .set_index("state_cluster")["n"]
        .astype(int)
    )
    cluster_labels = [f"{cluster}\nn = {cluster_counts.loc[cluster]:,}" for cluster in cluster_categories]

    panel_specs = [
        (axes[0, 0], by_time[by_time["velocity_space"] == "physical"], "time", time_categories, time_labels),
        (axes[0, 1], by_time[by_time["velocity_space"] == "gene"], "time", time_categories, time_labels),
        (
            axes[1, 0],
            by_cluster[by_cluster["velocity_space"] == "physical"],
            "state_cluster",
            cluster_categories,
            cluster_labels,
        ),
        (
            axes[1, 1],
            by_cluster[by_cluster["velocity_space"] == "gene"],
            "state_cluster",
            cluster_categories,
            cluster_labels,
        ),
    ]
    for ax, table, category_column, categories, labels in panel_specs:
        draw_summary_panel(ax, table, category_column, categories, labels)

    axes[0, 0].set_ylabel("Velocity cosine similarity")
    axes[1, 0].set_ylabel("Velocity cosine similarity")
    axes[0, 0].set_xlabel("Simulated time point", labelpad=5)
    axes[0, 1].set_xlabel("Simulated time point", labelpad=5)
    axes[1, 0].set_xlabel("State-space cluster", labelpad=5)
    axes[1, 1].set_xlabel("State-space cluster", labelpad=5)

    headings = [
        ("a", "Physical dynamics across time"),
        ("b", "Gene dynamics across time"),
        ("c", "Physical dynamics across state partitions"),
        ("d", "Gene dynamics across state partitions"),
    ]
    for ax, (label, title) in zip(axes.flat, headings):
        add_panel_heading(fig, ax, label, title)

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=TEAL,
            marker="o",
            markersize=4.5,
            linewidth=1.4,
            label="Mean and 95% CI",
        ),
        Line2D(
            [0],
            [0],
            color=IQR_COLOR,
            marker="D",
            markerfacecolor="white",
            markeredgecolor=REFERENCE,
            markersize=4.5,
            linewidth=5.5,
            label="Median and IQR",
        ),
        Line2D(
            [0],
            [0],
            color="#A7ADB2",
            linewidth=0.8,
            linestyle=(0, (3, 2)),
            label="Perfect directional agreement",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.54, 0.962),
        ncol=3,
        frameon=False,
        handlelength=2.3,
        columnspacing=1.8,
        handletextpad=0.6,
    )

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_path, facecolor="white")
    fig.savefig(png_path, dpi=320, facecolor="white")
    plt.close(fig)


def write_caption(path: Path) -> None:
    path.write_text(
        "**Velocity-direction agreement across time and state-space partitions.** "
        "(a,b) Cell-wise cosine similarity between inferred and generator velocity "
        "vectors in physical and gene-expression space at each simulated time point. "
        "(c,d) The same measurements across three unsupervised state-space partitions "
        "defined from the retained 50-dimensional gene state. Teal circles and error "
        "bars show the mean and 95% confidence interval. Diamonds and gray bars show "
        "the median and interquartile range. The number of evaluated cells is shown "
        "below each category. The partitions were obtained by MiniBatch k-means. The "
        "number of partitions was selected by the highest silhouette score among "
        "three to eight candidates. Original cell-type labels were not retained in "
        "the simulation table, so these partitions describe broad gene-state regions "
        "and are not interpreted as cell types.\n",
        encoding="utf-8",
    )


def write_reviewer_response(path: Path) -> None:
    path.write_text(
        "We thank the reviewer for this suggestion. We now report velocity cosine "
        "similarity separately at each simulated time point and across broad "
        "gene-state partitions. The archived simulation table does not contain "
        "cell-type annotations, so we did not infer biological cell types after the "
        "fact. Instead, we defined three reproducible unsupervised partitions from "
        "the 50-dimensional observable gene state and explicitly label them as "
        "state-space clusters. The added figure reports the mean with a 95% confidence "
        "interval and the median with an interquartile range for physical and "
        "gene-expression velocity directions. This analysis shows where agreement "
        "varies across simulated time and across the sampled state space without "
        "assigning unsupported biological identities.\n",
        encoding="utf-8",
    )


def write_readme(path: Path, overall: pd.DataFrame, selected_k: int) -> None:
    physical = overall.loc[overall["velocity_space"] == "physical", "mean"].iloc[0]
    gene = overall.loc[overall["velocity_space"] == "gene", "mean"].iloc[0]
    path.write_text(
        "# AGIST velocity agreement by time and state-space partition\n\n"
        "This folder contains the reviewer-facing combined figure and the row-level "
        "statistics recovered from two archived velocity bundles.\n\n"
        "## Traceable aggregate\n\n"
        f"- Physical velocity cosine: `{physical:.6f}`\n"
        f"- Gene-expression velocity cosine: `{gene:.6f}`\n"
        "- Velocity definition: `base + interaction + score`\n"
        "- Rows: `31,816`\n\n"
        "## Population definition\n\n"
        f"The figure uses `{selected_k}` broad state-space clusters. They were obtained "
        "from `x3` through `x52` and are not cell types. The archived table does not "
        "contain the original cell-type labels.\n\n"
        "## Main files\n\n"
        "- `agist_velocity_cosine_time_and_state_partitions.pdf`: vector figure\n"
        "- `agist_velocity_cosine_time_and_state_partitions.png`: preview\n"
        "- `velocity_cosine_per_cell_full.csv`: row-level physical and gene cosine values\n"
        "- `velocity_cosine_by_time.csv`: time-point summary\n"
        "- `velocity_cosine_by_state_cluster.csv`: state-cluster summary\n"
        "- `velocity_cosine_by_time_and_state_cluster.csv`: joint audit table\n"
        "- `figure_caption.md`: proposed caption\n"
        "- `reviewer_response_text.md`: proposed response text\n"
        "- `PROVENANCE.md`: source paths, calculations and hashes\n",
        encoding="utf-8",
    )


def write_provenance(
    path: Path,
    args: argparse.Namespace,
    manifest_path: Path,
    pdf_path: Path,
    png_path: Path,
    script_snapshot: Path,
    selected_k: int,
) -> None:
    path.write_text(
        "# Figure provenance\n\n"
        "Archived on: `2026-08-11`\n\n"
        "Manuscript figure: `Reviewer follow-up, velocity agreement by time and state-space partition`\n\n"
        "Scientific claim: Velocity-direction agreement can be examined across simulated time and broad regions of the retained gene-state space.\n\n"
        "## Source paths\n\n"
        f"- Velocity archive A: `{args.archive_a.resolve()}`\n"
        f"- Velocity archive B: `{args.archive_b.resolve()}`\n"
        f"- Simulation table: `{args.data_csv.resolve()}`\n"
        f"- State-cluster assignments: `{args.cluster_assignments.resolve()}`\n"
        f"- Cluster-selection diagnostics: `{args.cluster_diagnostics.resolve()}`\n"
        f"- Input manifest: `{manifest_path.resolve()}`\n\n"
        "The UUID archive names do not retain the original prediction-versus-generator folder names. Cosine similarity is symmetric, so exchanging the two archives does not change any reported value.\n\n"
        "## Panel sources\n\n"
        "| Panel | Content | Source file | Calculation |\n"
        "|---|---|---|---|\n"
        "| a | Physical velocity by time | `velocity_cosine_by_time.csv` | Mean, 95% CI, median and IQR of row cosine over `x1:x2` |\n"
        "| b | Gene velocity by time | `velocity_cosine_by_time.csv` | Mean, 95% CI, median and IQR of row cosine over `x3:x52` |\n"
        "| c | Physical velocity by state partition | `velocity_cosine_by_state_cluster.csv` | Same physical metric grouped by state cluster |\n"
        "| d | Gene velocity by state partition | `velocity_cosine_by_state_cluster.csv` | Same gene metric grouped by state cluster |\n\n"
        "## Evaluation protocol\n\n"
        "- Velocity: sum of base, interaction and score components\n"
        "- Spatial dimensions: `x1:x2`\n"
        "- Gene dimensions: `x3:x52`\n"
        "- Zero-norm handling: vector pairs with denominator at most `1e-12` are excluded\n"
        "- State partition: MiniBatch k-means on `x3:x52` with seed 42\n"
        f"- Selected number of clusters: `{selected_k}`\n"
        "- Uncertainty: normal-approximation 95% confidence interval of the cell-level mean\n\n"
        "## Rebuild\n\n"
        "```bash\n"
        "MPLCONFIGDIR=/tmp/mplconfig /opt/anaconda3/envs/cb_pipeline/bin/python \\\n"
        "/Users/zhenyizhang/Desktop/CytoBridge-ST-1104/scripts/reporting/build_agist_velocity_time_cluster_breakdown.py\n"
        "```\n\n"
        "## Interpretation\n\n"
        "The figure supports time-resolved and state-resolved assessment of the observable velocity subsystem. The state-space clusters are broad computational partitions and must not be described as biological cell types.\n\n"
        "## SHA-256\n\n"
        f"- Figure PDF: `{sha256(pdf_path)}`\n"
        f"- Figure PNG: `{sha256(png_path)}`\n"
        f"- Plotting script: `{sha256(script_snapshot)}`\n"
        f"- Velocity archive A: `{sha256(args.archive_a)}`\n"
        f"- Velocity archive B: `{sha256(args.archive_b)}`\n"
        f"- Simulation table: `{sha256(args.data_csv)}`\n"
        f"- State-cluster assignments: `{sha256(args.cluster_assignments)}`\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-a", type=Path, required=True, help="First velocity archive")
    parser.add_argument("--archive-b", type=Path, required=True, help="Second velocity archive")
    parser.add_argument("--data-csv", type=Path, required=True, help="Row-matched AGIST cell table")
    parser.add_argument("--cluster-assignments", type=Path, required=True, help="State cluster for each cell row")
    parser.add_argument("--cluster-diagnostics", type=Path, required=True, help="Table used to select the cluster count")
    parser.add_argument("--style", type=Path, required=True, help="Matplotlib style file")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    full_a, manifest_a = load_velocity_archive(args.archive_a)
    full_b, manifest_b = load_velocity_archive(args.archive_b)
    if full_a.shape != full_b.shape:
        raise ValueError(f"Velocity shapes differ: {full_a.shape} versus {full_b.shape}")

    data = pd.read_csv(args.data_csv, usecols=["samples"])
    clusters = pd.read_csv(args.cluster_assignments)
    expected_rows = np.arange(len(data), dtype=int)
    if full_a.shape[0] != len(data):
        raise ValueError(f"Velocity rows {full_a.shape[0]} do not match CSV rows {len(data)}")
    if len(clusters) != len(data) or not np.array_equal(clusters["row_index"].to_numpy(int), expected_rows):
        raise ValueError("Cluster assignments are not row-matched to the simulation table")
    if not np.allclose(clusters["time"].to_numpy(float), data["samples"].to_numpy(float)):
        raise ValueError("Cluster times do not match the simulation table")

    per_cell = pd.DataFrame(
        {
            "row_index": expected_rows,
            "time": data["samples"].to_numpy(float),
            "state_cluster": clusters["state_cluster"].astype(str).to_numpy(),
            "physical_cosine": row_cosine(full_a, full_b, slice(0, 2)),
            "gene_cosine": row_cosine(full_a, full_b, slice(2, 52)),
        }
    )
    overall = grouped_summary(per_cell, [])
    by_time = grouped_summary(per_cell, ["time"]).sort_values(["velocity_space", "time"])
    by_cluster = grouped_summary(per_cell, ["state_cluster"]).sort_values(
        ["velocity_space", "state_cluster"]
    )
    by_time_cluster = grouped_summary(per_cell, ["time", "state_cluster"]).sort_values(
        ["velocity_space", "time", "state_cluster"]
    )

    per_cell_path = args.output_dir / "velocity_cosine_per_cell_full.csv"
    overall_path = args.output_dir / "velocity_cosine_overall.csv"
    by_time_path = args.output_dir / "velocity_cosine_by_time.csv"
    by_cluster_path = args.output_dir / "velocity_cosine_by_state_cluster.csv"
    by_time_cluster_path = args.output_dir / "velocity_cosine_by_time_and_state_cluster.csv"
    per_cell.to_csv(per_cell_path, index=False)
    overall.to_csv(overall_path, index=False)
    by_time.to_csv(by_time_path, index=False)
    by_cluster.to_csv(by_cluster_path, index=False)
    by_time_cluster.to_csv(by_time_cluster_path, index=False)

    pdf_path = args.output_dir / "agist_velocity_cosine_time_and_state_partitions.pdf"
    png_path = args.output_dir / "agist_velocity_cosine_time_and_state_partitions.png"
    build_figure(by_time, by_cluster, pdf_path, png_path, args.style)
    write_caption(args.output_dir / "figure_caption.md")
    write_reviewer_response(args.output_dir / "reviewer_response_text.md")

    diagnostics = pd.read_csv(args.cluster_diagnostics)
    selected_k = int(diagnostics.sort_values(["silhouette", "n_clusters"], ascending=[False, True]).iloc[0]["n_clusters"])
    manifest = {
        "archive_a": {
            "path": str(args.archive_a.resolve()),
            "sha256": sha256(args.archive_a),
            "components": manifest_a,
        },
        "archive_b": {
            "path": str(args.archive_b.resolve()),
            "sha256": sha256(args.archive_b),
            "components": manifest_b,
        },
        "data_csv": {
            "path": str(args.data_csv.resolve()),
            "sha256": sha256(args.data_csv),
            "rows": len(data),
        },
        "cluster_assignments": {
            "path": str(args.cluster_assignments.resolve()),
            "sha256": sha256(args.cluster_assignments),
            "selected_k": selected_k,
        },
        "velocity_definition": "base + interaction + score",
        "spatial_dimensions": [0, 1],
        "gene_dimensions": [2, 51],
        "cosine_zero_norm_threshold": 1e-12,
    }
    manifest_path = args.output_dir / "input_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    script_snapshot = args.output_dir / Path(__file__).name
    shutil.copy2(Path(__file__), script_snapshot)
    write_readme(args.output_dir / "README.md", overall, selected_k)
    write_provenance(
        args.output_dir / "PROVENANCE.md",
        args,
        manifest_path,
        pdf_path,
        png_path,
        script_snapshot,
        selected_k,
    )

    print(overall.to_string(index=False))
    print(by_time.to_string(index=False))
    print(by_cluster.to_string(index=False))
    print(pdf_path)
    print(png_path)


if __name__ == "__main__":
    main()
