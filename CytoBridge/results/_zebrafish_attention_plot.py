"""Plot the zebrafish attention figure from processed panel data."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .zebrafish_attention import ZebrafishAttentionResults


_RC = {
    "font.family": "Arial",
    "font.size": 9.0,
    "axes.titlesize": 9.0,
    "axes.labelsize": 9.0,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "text.color": "black",
    "axes.labelcolor": "black",
    "axes.titlecolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "axes.spines.top": False,
    "axes.spines.right": False,
}


def plot_zebrafish_attention(
    results: "ZebrafishAttentionResults",
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """Draw and save the A4 portrait figure."""

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyArrowPatch

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    navy = "#214E78"
    teal = "#168A83"
    gold = "#D9A441"
    coral = "#C65F4A"
    dark_grey = "#6F777D"
    middle_grey = "#AEB6BC"
    pale_grey = "#D8DDE1"
    background_grey = "#ECEFF1"
    condition_order = ["trained", "pre_interaction", "random"]
    condition_labels = {
        "trained": "After interaction\ntraining",
        "pre_interaction": "Before interaction\nlearning",
        "random": "Randomized\nweights",
    }

    pair = results.panels.external_agreement.set_index("external_method").loc[
        ["COMMOT", "CellAgentChat"]
    ]
    quartile = results.panels.jam_quartiles.set_index("condition").loc[condition_order]
    top_rates = quartile["top_compatibility_percent"]
    bottom_rates = quartile["bottom_compatibility_percent"]
    cells = results.spatial_cells
    display_edges = results.display_edges
    association = results.panels.myog_detection.set_index("gene").loc[
        ["jam3b", "jam2a"]
    ]
    spatial = results.panels.spatial_null.iloc[0]
    null_counts = results.spatial_null_iterations[
        "orientation_compatible_pair_count"
    ].to_numpy(float)
    n_somite = int(cells["is_somite"].sum())
    n_somite_edges = int(
        results.compatibility_summary.groupby("condition", sort=False)[
            "n_directed_edges"
        ]
        .sum()
        .iloc[0]
    )

    with mpl.rc_context(_RC):
        fig = plt.figure(figsize=(8.27, 11.69), constrained_layout=False)
        outer = fig.add_gridspec(
            3,
            1,
            height_ratios=(0.20, 0.31, 0.49),
            left=0.145,
            right=0.970,
            top=0.982,
            bottom=0.055,
            hspace=0.17,
        )

        def _headed_panel(parent, label: str, title: str, *, heading_ratio: float):
            nested = parent.subgridspec(
                2,
                1,
                height_ratios=(heading_ratio, 1.0),
                hspace=0.08,
            )
            heading = fig.add_subplot(nested[0])
            heading.axis("off")
            heading.text(
                0,
                0.52,
                label,
                fontsize=14,
                fontweight="bold",
                va="center",
            )
            heading.text(
                0.065,
                0.52,
                title,
                fontsize=12,
                fontweight="bold",
                va="center",
            )
            return nested[1]

        a_body = _headed_panel(
            outer[0],
            "a",
            "External CCI agreement",
            heading_ratio=0.24,
        )
        ax_a = fig.add_subplot(a_body)
        y_a = np.arange(2)[::-1]
        labels_a = ["COMMOT", "CellAgentChat proxy"]
        colors_a = [teal, dark_grey]
        null_min = float(pair["null_adjusted_spearman_q025"].min())
        observed_max = float(pair["adjusted_spearman_rho"].max())
        for y_value, (_, row), color in zip(
            y_a,
            pair.iterrows(),
            colors_a,
            strict=True,
        ):
            ax_a.plot(
                [
                    row.null_adjusted_spearman_q025,
                    row.null_adjusted_spearman_q975,
                ],
                [y_value, y_value],
                color=middle_grey,
                linewidth=5.0,
                solid_capstyle="round",
                zorder=1,
            )
            ax_a.scatter(
                row.null_adjusted_spearman_mean,
                y_value,
                s=20,
                facecolor="white",
                edgecolor=dark_grey,
                linewidth=0.7,
                zorder=2,
            )
            ax_a.scatter(
                row.adjusted_spearman_rho,
                y_value,
                s=48,
                color=color,
                edgecolor="white",
                linewidth=0.6,
                zorder=3,
            )
            p_value = float(row.adjusted_spearman_empirical_p_upper)
            p_text = "P<0.001" if p_value < 0.001 else f"P={p_value:.3g}"
            ax_a.text(
                float(row.adjusted_spearman_rho) + 0.018,
                y_value,
                f"ρ={row.adjusted_spearman_rho:.2f}\n{p_text}",
                va="center",
                fontsize=8.3,
                linespacing=0.90,
            )
        n_pairs = int(pair["n_pairs"].iloc[0])
        ax_a.set_yticks(y_a, labels_a)
        ax_a.set_xlim(max(0, null_min - 0.07), min(1.04, observed_max + 0.13))
        ax_a.set_ylim(-0.55, 1.55)
        ax_a.set_xlabel(
            "Rank agreement with CytoBridge attention (adjusted Spearman ρ)"
        )
        ax_a.set_title(
            f"Interaction patterns across all {n_pairs} sender → receiver cell-type pairs",
            pad=5,
        )
        ax_a.grid(axis="x", color=pale_grey, linewidth=0.5)
        ax_a.legend(
            handles=[
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="none",
                    markerfacecolor=teal,
                    markeredgecolor="white",
                    label="Observed agreement",
                ),
                Line2D(
                    [0],
                    [0],
                    color=middle_grey,
                    linewidth=5,
                    label="Structured-null 95% range",
                ),
            ],
            frameon=False,
            loc="lower right",
            ncol=2,
            handlelength=1.8,
            columnspacing=1.4,
        )

        b_body = _headed_panel(
            outer[1],
            "b",
            "JAM-compatible edges receive high attention",
            heading_ratio=0.18,
        )
        b_container = b_body.subgridspec(
            2,
            1,
            height_ratios=(0.10, 1.0),
            hspace=0.11,
        )
        ax_b = fig.add_subplot(b_container[1])
        y_b = np.arange(3)[::-1]
        bar_offset = 0.16
        ax_b.barh(
            y_b + bar_offset,
            top_rates.to_numpy(float),
            height=0.25,
            color=navy,
            edgecolor="none",
            label="Top attention quartile",
            zorder=3,
        )
        ax_b.barh(
            y_b - bar_offset,
            bottom_rates.to_numpy(float),
            height=0.25,
            facecolor="white",
            edgecolor=dark_grey,
            linewidth=1.0,
            label="Bottom attention quartile",
            zorder=3,
        )
        for y_value, condition in zip(y_b, condition_order, strict=True):
            ax_b.text(
                float(top_rates.loc[condition]) + 0.45,
                y_value + bar_offset,
                f"{float(top_rates.loc[condition]):.1f}%",
                va="center",
                fontsize=8.4,
            )
            ax_b.text(
                float(bottom_rates.loc[condition]) + 0.45,
                y_value - bar_offset,
                f"{float(bottom_rates.loc[condition]):.1f}%",
                va="center",
                fontsize=8.4,
                color=dark_grey,
            )
        ax_b.set_yticks(
            y_b,
            [condition_labels[item] for item in condition_order],
        )
        ax_b.set_xlim(
            0,
            max(30.0, float(max(top_rates.max(), bottom_rates.max())) + 4.0),
        )
        ax_b.set_xlabel("JAM-compatible edges (%)")
        ax_b.set_title(
            "High-attention edges are selectively JAM-compatible only after interaction training",
            pad=5,
        )
        ax_b.grid(axis="x", color=pale_grey, linewidth=0.5, zorder=0)
        ax_b.legend(frameon=False, loc="lower right", ncol=2)

        ax_b_note = fig.add_subplot(b_container[0])
        ax_b_note.axis("off")
        ax_b_note.text(
            0.5,
            0.60,
            f"Same {n_somite:,} 18 hpf Somite cells and {n_somite_edges:,} model edges in every condition.  "
            "JAM-compatible = jam2a at one endpoint and jam3b at the other.",
            ha="center",
            va="center",
            fontsize=8.6,
            color="black",
        )

        c_body = _headed_panel(
            outer[2],
            "c",
            "Spatial and myogenic context of the JAM program",
            heading_ratio=0.11,
        )
        c_grid = c_body.subgridspec(
            1,
            2,
            width_ratios=(0.57, 0.43),
            wspace=0.20,
        )
        ax_c1 = fig.add_subplot(c_grid[0])
        c_metrics = c_grid[1].subgridspec(
            2,
            1,
            height_ratios=(0.54, 0.46),
            hspace=0.44,
        )
        ax_c2 = fig.add_subplot(c_metrics[0])
        ax_c3 = fig.add_subplot(c_metrics[1])

        background = cells.loc[~cells["is_somite"]]
        somite = cells.loc[cells["is_somite"]]
        ax_c1.scatter(
            background["x"],
            background["y"],
            s=1.4,
            color=background_grey,
            linewidth=0,
            alpha=0.70,
        )
        ax_c1.scatter(
            somite["x"],
            somite["y"],
            s=3.0,
            color="#C9DAD8",
            linewidth=0,
            alpha=0.78,
        )
        incompatible = display_edges.loc[~display_edges["jam_compatible"]]
        compatible_edges = display_edges.loc[display_edges["jam_compatible"]]
        if not incompatible.empty:
            segments = (
                incompatible[["source_x", "source_y", "target_x", "target_y"]]
                .to_numpy(float)
                .reshape(-1, 2, 2)
            )
            ax_c1.add_collection(
                LineCollection(
                    segments,
                    colors=middle_grey,
                    linewidths=0.55,
                    alpha=0.45,
                )
            )
        for row in compatible_edges.itertuples(index=False):
            ax_c1.add_patch(
                FancyArrowPatch(
                    (float(row.source_x), float(row.source_y)),
                    (float(row.target_x), float(row.target_y)),
                    arrowstyle="-|>",
                    mutation_scale=4.8,
                    linewidth=0.9,
                    color=gold,
                    alpha=0.90,
                    shrinkA=0.8,
                    shrinkB=0.8,
                    zorder=2,
                )
            )
        jam2_only = somite["jam2a_positive"] & ~somite["jam3b_positive"]
        jam3_only = somite["jam3b_positive"] & ~somite["jam2a_positive"]
        both = somite["jam2a_positive"] & somite["jam3b_positive"]
        for mask, color, marker, size in (
            (jam2_only, coral, "o", 11.0),
            (jam3_only, navy, "s", 10.5),
            (both, "#202020", "D", 12.0),
        ):
            ax_c1.scatter(
                somite.loc[mask, "x"],
                somite.loc[mask, "y"],
                s=size,
                color=color,
                marker=marker,
                edgecolor="white",
                linewidth=0.30,
                alpha=0.92,
            )
        ax_c1.set_aspect("equal", adjustable="box")
        ax_c1.set_xticks([])
        ax_c1.set_yticks([])
        for spine in ax_c1.spines.values():
            spine.set_visible(False)
        ax_c1.set_title(
            f"18 hpf tissue map ({len(cells):,} cells shown; {n_somite:,} Somite cells analyzed)",
            pad=5,
        )
        ax_c1.legend(
            handles=[
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="none",
                    markerfacecolor=background_grey,
                    markeredgecolor="none",
                    label="Other tissue cell",
                ),
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="none",
                    markerfacecolor="#C9DAD8",
                    markeredgecolor="none",
                    label="Somite cell analyzed",
                ),
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="none",
                    markerfacecolor=coral,
                    markeredgecolor="white",
                    label="jam2a+ only",
                ),
                Line2D(
                    [0],
                    [0],
                    marker="s",
                    color="none",
                    markerfacecolor=navy,
                    markeredgecolor="white",
                    label="jam3b+ only",
                ),
                Line2D(
                    [0],
                    [0],
                    marker="D",
                    color="none",
                    markerfacecolor="#202020",
                    markeredgecolor="white",
                    label="Both JAM genes",
                ),
                Line2D(
                    [0],
                    [0],
                    color=gold,
                    linewidth=1.5,
                    label=f"High-attention JAM edge (n={len(compatible_edges)})",
                ),
            ],
            frameon=False,
            fontsize=8.0,
            loc="lower left",
            ncol=2,
            handlelength=1.6,
            columnspacing=1.1,
        )

        y_c3 = np.array([1.0, 0.0])
        c3_offset = 0.16
        positive = association["myog_percent_when_gene_positive"]
        negative = association["myog_percent_when_gene_negative"]
        ax_c3.barh(
            y_c3 + c3_offset,
            positive.to_numpy(float),
            height=0.25,
            color=teal,
            edgecolor="none",
            label="JAM gene+",
            zorder=3,
        )
        ax_c3.barh(
            y_c3 - c3_offset,
            negative.to_numpy(float),
            height=0.25,
            facecolor="white",
            edgecolor=dark_grey,
            linewidth=1.0,
            label="JAM gene−",
            zorder=3,
        )
        for y_value, gene in zip(y_c3, ["jam3b", "jam2a"], strict=True):
            ax_c3.text(
                float(positive.loc[gene]) + 0.8,
                y_value + c3_offset,
                f"{float(positive.loc[gene]):.1f}%",
                va="center",
                fontsize=8.0,
            )
            ax_c3.text(
                float(negative.loc[gene]) + 0.8,
                y_value - c3_offset,
                f"{float(negative.loc[gene]):.1f}%",
                va="center",
                fontsize=8.0,
                color=dark_grey,
            )
            ax_c3.text(
                max(float(positive.loc[gene]), float(negative.loc[gene])) + 7.0,
                y_value,
                f"P={float(association.loc[gene, 'fisher_two_sided_p']):.2g}",
                va="center",
                fontsize=8.0,
            )
        ax_c3.set_xlim(
            0,
            max(70.0, float(max(positive.max(), negative.max())) + 14.0),
        )
        ax_c3.set_ylim(-0.55, 1.55)
        ax_c3.set_yticks(y_c3, ["jam3b", "jam2a"])
        ax_c3.set_xlabel("myog+ Somite cells (%)")
        ax_c3.set_title("myog detection by JAM-gene status", pad=5)
        ax_c3.grid(axis="x", color=pale_grey, linewidth=0.5)
        ax_c3.legend(frameon=False, loc="lower right", ncol=2, fontsize=8.0)

        observed_neighbors = int(spatial["observed_neighbor_pairs"])
        null_mean = float(spatial["null_mean"])
        fold = float(spatial["observed_over_null_mean"])
        p_value = float(spatial["plus_one_upper_tail_p"])
        ax_c2.hist(
            null_counts,
            bins=28,
            color=background_grey,
            edgecolor="white",
            linewidth=0.45,
        )
        ax_c2.axvline(
            observed_neighbors,
            color=teal,
            linewidth=2.2,
            label=f"Observed: {observed_neighbors}",
        )
        ax_c2.axvspan(
            float(spatial["null_q025"]),
            float(spatial["null_q975"]),
            color=middle_grey,
            alpha=0.16,
            linewidth=0,
            label=(
                f"Random labels: mean {null_mean:.0f}\n"
                f"95% range {float(spatial['null_q025']):.0f}–{float(spatial['null_q975']):.0f}"
            ),
        )
        ax_c2.set_xlim(
            min(float(null_counts.min()), float(spatial["null_q025"])) - 12,
            max(observed_neighbors, float(null_counts.max())) + 12,
        )
        ax_c2.set_xlabel("Complementary jam2a+/jam3b+ spatial-neighbor pairs")
        ax_c2.set_ylabel("Label permutations")
        ax_c2.set_title(
            "Complementary jam2a+/jam3b+ cells\nare spatial neighbors",
            pad=5,
        )
        ax_c2.legend(frameon=False, loc="upper left", fontsize=8.0)
        ax_c2.text(
            0.98,
            0.94,
            f"{fold:.2f}× enrichment\nP={p_value:.2g}",
            transform=ax_c2.transAxes,
            ha="right",
            va="top",
            fontsize=8.3,
        )

        pdf_path = output / "zebrafish_attention_validation_a4.pdf"
        png_path = output / "zebrafish_attention_validation_a4.png"
        fig.savefig(pdf_path, facecolor="white")
        fig.savefig(png_path, dpi=320, facecolor="white")
        plt.close(fig)

    return pdf_path, png_path
