#!/usr/bin/env python3
"""Evaluate AGIST stochastic replicates, test fixed baselines, and draw Fig. 2e."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


SPACES = {
    "gene": np.arange(2, 52),
    "physical": np.arange(0, 2),
}

# Values used in the archived Figure 2e plotting notebook.  The deterministic
# fitted maps are retained as fixed reference estimates; stochastic error bars
# are estimated from independently seeded CytoBridge simulations.
BASELINES = {
    "gene": {
        "STORIES": {1.0: 1.300, 2.0: 2.949, 3.0: 3.718},
        "stVCR": {1.0: 1.269, 2.0: 1.440, 3.0: 1.476},
    },
    "physical": {
        "stVCR": {1.0: 0.030, 2.0: 0.049, 3.0: 0.040},
    },
}


def exact_w2(pred: np.ndarray, truth: np.ndarray, pred_weight: np.ndarray) -> float:
    import ot

    pred = np.asarray(pred, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    pred_weight = np.asarray(pred_weight, dtype=np.float64).reshape(-1)
    pred_weight /= pred_weight.sum()
    truth_weight = np.full(truth.shape[0], 1.0 / truth.shape[0], dtype=np.float64)
    cost = ot.dist(truth, pred, metric="sqeuclidean")
    value = ot.emd2(truth_weight, pred_weight, cost, numItermax=10_000_000)
    del cost
    return float(math.sqrt(max(0.0, value)))


def sign_flip_pvalue(differences: np.ndarray) -> float:
    """Exact two-sided one-sample randomization test around zero."""
    values = np.asarray(differences, dtype=float)
    observed = abs(float(values.mean()))
    null = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        null.append(abs(float(np.mean(values * np.asarray(signs)))))
    null_array = np.asarray(null)
    return float(np.mean(null_array >= observed - 1e-15))


def holm_adjust(pvalues: list[float]) -> list[float]:
    count = len(pvalues)
    order = np.argsort(pvalues)
    adjusted = np.empty(count, dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * float(pvalues[index]))
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted.tolist()


def load_truth(path: Path) -> dict[float, np.ndarray]:
    frame = pd.read_csv(path)
    columns = [f"x{i}" for i in range(1, 53)]
    return {
        float(time): subset[columns].to_numpy(np.float64)
        for time, subset in frame.groupby("samples", sort=True)
    }


def compute_metrics(trajectory_dir: Path, truth_csv: Path, output_csv: Path) -> pd.DataFrame:
    if output_csv.exists():
        existing = pd.read_csv(output_csv)
        expected_pairs = {
            (float(time), space)
            for time in (1.0, 2.0, 3.0)
            for space in SPACES
        }
        complete_seeds = {
            int(seed)
            for seed, subset in existing.groupby("seed")
            if {
                (float(row.time), str(row.space))
                for row in subset.itertuples(index=False)
            }
            == expected_pairs
        }
        if len(complete_seeds) >= 2 and set(existing["seed"].astype(int)) == complete_seeds:
            return existing.sort_values(["space", "time", "seed"])
        rows = existing.to_dict("records")
    else:
        rows = []
    truth = load_truth(truth_csv)
    completed = {
        (int(row["seed"]), float(row["time"]), str(row["space"])) for row in rows
    }

    files = sorted(
        trajectory_dir.glob("split_sde_seed_*.npz"),
        key=lambda path: int(path.stem.rsplit("_", 1)[-1]),
    )
    if not files:
        raise FileNotFoundError(f"No trajectory files in {trajectory_dir}")
    for path in files:
        seed = int(path.stem.rsplit("_", 1)[-1])
        archive = np.load(path, allow_pickle=True)
        points = list(archive["points"])
        weights = list(archive["weights"])
        for time_index, time in enumerate((1.0, 2.0, 3.0), start=1):
            for space, indices in SPACES.items():
                key = (seed, time, space)
                if key in completed:
                    continue
                value = exact_w2(
                    np.asarray(points[time_index])[:, indices],
                    truth[time][:, indices],
                    np.asarray(weights[time_index]),
                )
                rows.append(
                    {
                        "method": "CytoBridge",
                        "seed": seed,
                        "time": time,
                        "space": space,
                        "w2": value,
                        "n_predicted": int(np.asarray(points[time_index]).shape[0]),
                        "n_truth": int(truth[time].shape[0]),
                    }
                )
                pd.DataFrame(rows).sort_values(["space", "time", "seed"]).to_csv(
                    output_csv, index=False
                )
    return pd.DataFrame(rows).sort_values(["space", "time", "seed"])


def summarize(long_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = (
        long_frame.groupby(["space", "time"], as_index=False)
        .agg(
            mean_w2=("w2", "mean"),
            sd_w2=("w2", "std"),
            n=("w2", "size"),
            min_w2=("w2", "min"),
            max_w2=("w2", "max"),
            mean_n_predicted=("n_predicted", "mean"),
            sd_n_predicted=("n_predicted", "std"),
        )
        .sort_values(["space", "time"])
    )
    summary["se_w2"] = summary["sd_w2"] / np.sqrt(summary["n"])
    summary["ci95_halfwidth"] = summary.apply(
        lambda row: stats.t.ppf(0.975, int(row["n"]) - 1) * row["se_w2"], axis=1
    )

    comparisons: list[dict[str, object]] = []
    for (space, time), subset in long_frame.groupby(["space", "time"], sort=True):
        for method, values_by_time in BASELINES[str(space)].items():
            reference = float(values_by_time[float(time)])
            differences = reference - subset["w2"].to_numpy(float)
            comparisons.append(
                {
                    "space": space,
                    "time": float(time),
                    "comparison": f"CytoBridge vs {method}",
                    "baseline_method": method,
                    "baseline_w2": reference,
                    "mean_cytobridge_w2": float(subset["w2"].mean()),
                    "baseline_minus_cytobridge": float(differences.mean()),
                    "cytobridge_lower_fraction": float(np.mean(differences > 0)),
                    "p_exact_two_sided": sign_flip_pvalue(differences),
                    "test": "exact two-sided one-sample sign-flip randomization test",
                    "n_cytobridge_runs": int(len(differences)),
                }
            )
    comparison_frame = pd.DataFrame(comparisons)
    comparison_frame["p_holm"] = holm_adjust(comparison_frame["p_exact_two_sided"].tolist())
    return summary, comparison_frame


def p_label(value: float) -> str:
    if value < 0.001:
        return "***"
    if value < 0.01:
        return "**"
    if value < 0.05:
        return "*"
    return "n.s."


def draw_figure(
    long_frame: pd.DataFrame,
    summary: pd.DataFrame,
    comparisons: pd.DataFrame,
    output_dir: Path,
) -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 9,
            "axes.titlesize": 12,
            "axes.labelsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    colors = {"STORIES": "#D9C8B7", "stVCR": "#C98572", "CytoBridge": "#913A45"}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.25), gridspec_kw={"wspace": 0.34})
    rng = np.random.default_rng(20260811)
    panel_label_artists = []

    for panel, (axis, space, title) in enumerate(
        zip(axes, ("gene", "physical"), ("Gene expression space", "Physical space"))
    ):
        methods = list(BASELINES[space]) + ["CytoBridge"]
        times = [1.0, 2.0, 3.0]
        width = 0.22 if len(methods) == 3 else 0.28
        centers = np.arange(len(times), dtype=float)
        offsets = (np.arange(len(methods)) - (len(methods) - 1) / 2.0) * width
        for method_index, method in enumerate(methods):
            values = []
            errors = []
            for time in times:
                if method == "CytoBridge":
                    row = summary[(summary["space"] == space) & (summary["time"] == time)].iloc[0]
                    values.append(float(row["mean_w2"]))
                    errors.append(float(row["sd_w2"]))
                else:
                    values.append(float(BASELINES[space][method][time]))
                    errors.append(0.0)
            x = centers + offsets[method_index]
            axis.bar(
                x,
                values,
                width=width * 0.88,
                color=colors[method],
                edgecolor="none",
                label=method,
                zorder=2,
            )
            if method == "CytoBridge":
                axis.errorbar(
                    x,
                    values,
                    yerr=errors,
                    fmt="none",
                    ecolor="#202020",
                    elinewidth=0.9,
                    capsize=2.5,
                    capthick=0.9,
                    zorder=4,
                )
                for xpos, time in zip(x, times):
                    run_values = long_frame[
                        (long_frame["space"] == space) & (long_frame["time"] == time)
                    ]["w2"].to_numpy(float)
                    jitter = rng.uniform(-width * 0.18, width * 0.18, len(run_values))
                    axis.scatter(
                        xpos + jitter,
                        run_values,
                        s=8,
                        facecolor="white",
                        edgecolor=colors[method],
                        linewidth=0.55,
                        zorder=5,
                    )

        axis.set_xticks(centers, ["t = 1", "t = 2", "t = 3"])
        axis.set_ylabel("Wasserstein-2 distance")
        axis.set_title(title, pad=7)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(direction="out", length=3, width=0.8)
        axis.set_axisbelow(True)
        axis.yaxis.grid(True, color="#E8E8E8", linewidth=0.55)

        stvcr_rows = comparisons[
            (comparisons["space"] == space) & (comparisons["baseline_method"] == "stVCR")
        ].sort_values("time")
        ymax = max(
            max(max(values.values()) for values in BASELINES[space].values()),
            float((summary.loc[summary["space"] == space, "mean_w2"] + summary.loc[summary["space"] == space, "sd_w2"]).max()),
        )
        axis.set_ylim(0, ymax * 1.17)
        stvcr_offset = offsets[methods.index("stVCR")]
        cytobridge_offset = offsets[methods.index("CytoBridge")]
        for center, time, (_, row) in zip(centers, times, stvcr_rows.iterrows()):
            cyto_row = summary[
                (summary["space"] == space) & (summary["time"] == time)
            ].iloc[0]
            pair_top = max(
                float(BASELINES[space]["stVCR"][time]),
                float(cyto_row["mean_w2"] + cyto_row["sd_w2"]),
            )
            y = pair_top + ymax * 0.035
            h = ymax * 0.012
            x1 = center + stvcr_offset
            x2 = center + cytobridge_offset
            axis.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color="#202020", lw=0.65)
            axis.text(
                (x1 + x2) / 2,
                y + h + ymax * 0.005,
                p_label(float(row["p_holm"])),
                ha="center",
                va="bottom",
                fontsize=9,
            )
        panel_label_artists.append(axis.text(
            -0.15,
            1.04,
            chr(ord("a") + panel),
            transform=axis.transAxes,
            fontsize=14,
            fontweight="bold",
            ha="left",
            va="bottom",
        ))

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", ncol=3, bbox_to_anchor=(0.53, 1.02))
    fig.subplots_adjust(top=0.80, bottom=0.18, left=0.10, right=0.98)
    for suffix in ("pdf", "svg", "png"):
        kwargs = {"dpi": 600} if suffix == "png" else {}
        fig.savefig(output_dir / f"agist_w2_stochastic_replicates.{suffix}", bbox_inches="tight", **kwargs)
    for artist in panel_label_artists:
        artist.set_visible(False)
    for suffix in ("pdf", "svg", "png"):
        kwargs = {"dpi": 600} if suffix == "png" else {}
        fig.savefig(output_dir / f"figure2e_agist_w2_mean_sd.{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory-dir", type=Path, required=True)
    parser.add_argument("--truth-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--metrics-only", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    long_path = args.output_dir / "w2_replicates_long.csv"
    long_frame = compute_metrics(args.trajectory_dir, args.truth_csv, long_path)
    summary, comparisons = summarize(long_frame)
    summary.to_csv(args.output_dir / "w2_mean_sd_ci.csv", index=False)
    comparisons.to_csv(args.output_dir / "w2_baseline_comparisons.csv", index=False)
    if not args.metrics_only:
        draw_figure(long_frame, summary, comparisons, args.output_dir)

    result = {
        "n_independent_split_sde_runs": int(long_frame["seed"].nunique()),
        "seeds": sorted(int(value) for value in long_frame["seed"].unique()),
        "error_bar": "mean +/- standard deviation across independently seeded split-SDE simulations",
        "test": "exact two-sided one-sample sign-flip randomization test against fixed fitted-map reference W2",
        "multiplicity": "Holm adjustment across all Figure 2e method/time/space comparisons",
        "baseline_source": "archived evaluation/mosta_simulation_evaluation_viz_notebook.ipynb Figure 2e values",
    }
    (args.output_dir / "analysis_manifest.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
