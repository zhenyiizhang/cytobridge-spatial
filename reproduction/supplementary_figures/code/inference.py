"""Draw the interaction-on/off comparison from the paired reconstruction errors."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

from cytobridge_figure_style import apply_style


DATASETS = ["zebrafish", "mosta", "arista", "admouse", "chicken_heart"]
LABELS = ["Zebrafish", "MOSTA", "ARISTA", "AD mouse", "Chicken\nheart"]
SPACES = ["joint", "spatial", "state"]
SPACE_LABELS = {"joint": "Joint", "spatial": "Spatial", "state": "Gene state"}
COLORS = {"joint": "#4C78A8", "spatial": "#E39D2D", "state": "#8F63A8"}
TARGET_COUNTS = {"zebrafish": 4, "mosta": 3, "arista": 4, "admouse": 2, "chicken_heart": 3}


def read_results(root):
    frames = []
    manifests = []
    for dataset in DATASETS:
        manifest = json.loads((root / dataset / "manifest.json").read_text())
        assert manifest["status"] == "complete" and manifest["training_performed"] is False
        assert manifest["dataset"] == dataset
        assert manifest["aligned_input_matches_training"]
        assert manifest["protocol"]["inference_seeds"] == [42, 43, 44]
        frame = pd.read_csv(root / dataset / "metrics.csv")
        keys = ["dataset", "inference_seed", "target", "space", "arm", "projection_repeat"]
        assert not frame.duplicated(keys).any()
        assert len(frame) == 3 * TARGET_COUNTS[dataset] * 3 * 2 * 5
        assert set(frame["dataset"]) == {dataset}
        assert np.isfinite(frame["sliced_w2"]).all() and (frame["sliced_w2"] > 0).all()
        basis_counts = frame.groupby(["dataset", "target", "space", "projection_repeat"])["projection_sha256"].nunique()
        assert basis_counts.eq(1).all()
        frames.append(frame)
        manifests.append(manifest)
    raw = pd.concat(frames, ignore_index=True)
    means = raw.groupby(["dataset", "inference_seed", "target", "space", "arm"], as_index=False)["sliced_w2"].mean()
    paired = means.pivot(index=["dataset", "inference_seed", "target", "space"], columns="arm", values="sliced_w2").reset_index()
    paired.columns.name = None
    paired["off_relative_to_on"] = paired["interaction_off"] / paired["interaction_on"] - 1
    targets = paired.groupby(["dataset", "target", "space"], as_index=False).agg(
        off_relative_to_on=("off_relative_to_on", "mean"),
        interaction_on=("interaction_on", "mean"), interaction_off=("interaction_off", "mean"))
    return raw, paired, targets, manifests


def sem(values):
    return float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0


def make_figure(targets):
    for name in ["Arial.ttf", "Arial Bold.ttf"]:
        path = Path("/System/Library/Fonts/Supplemental") / name
        if path.exists():
            font_manager.fontManager.addfont(str(path))
    apply_style()
    mpl.rcParams.update({"text.color": "black", "axes.labelcolor": "black",
                         "axes.titlecolor": "black", "xtick.color": "black", "ytick.color": "black"})
    fig, axes = plt.subplots(1, 2, figsize=(8.27, 5.85))
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.14, top=0.77, wspace=0.34)
    for label, title, ax in zip(["a", "b"], ["Interaction ablation", "Effect by evaluation space"], axes):
        box = ax.get_position()
        fig.text(box.x0, 0.9, label, fontsize=14, weight="bold", color="black")
        fig.text(box.x0 + 0.031, 0.9, title, fontsize=12, weight="bold", color="black")
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#D7DDE2", linewidth=0.5, alpha=0.75)
        ax.set_xticks(np.arange(5), LABELS)
        ax.margins(x=0.05)

    overall = targets.groupby("dataset")["off_relative_to_on"].mean().reindex(DATASETS)
    x = np.arange(5)
    width = 0.34
    axes[0].bar(x - width/2, np.ones(5), width, color="#07838B", label="With interaction", zorder=2)
    axes[0].bar(x + width/2, 1 + overall.to_numpy(), width, color="#CC6677", label="Without interaction", zorder=2)
    axes[0].set_ylabel("Relative sliced-W2\n(with interaction = 1)")
    axes[0].set_ylim(0, max(1., float((1 + overall).max())) * 1.12)
    axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.025), ncol=2, frameon=False,
                   handlelength=1, handletextpad=0.5, columnspacing=0.8, borderaxespad=0)

    limits = [0.]
    for space_index, space in enumerate(SPACES):
        values = [targets.loc[(targets.dataset == d) & (targets.space == space), "off_relative_to_on"].to_numpy() * 100 for d in DATASETS]
        averages = np.array([a.mean() for a in values])
        errors = np.array([sem(a) for a in values])
        positions = x + (space_index - 1) * 0.23
        axes[1].bar(positions, averages, 0.19, color=COLORS[space], alpha=0.88,
                    edgecolor="white", linewidth=0.5, label=SPACE_LABELS[space], zorder=2)
        axes[1].errorbar(positions, averages, yerr=errors, fmt="none", ecolor="#24313A",
                         elinewidth=0.7, capsize=2, capthick=0.7, zorder=4)
        for index, vals in enumerate(values):
            axes[1].scatter(positions[index] + np.linspace(-.035, .035, len(vals)), vals,
                            s=13, facecolor="white", edgecolor="#24313A", linewidth=.55, zorder=5)
            limits.extend(vals)
        limits.extend(averages - errors)
        limits.extend(averages + errors)
    padding = max(max(limits) - min(limits), 1.) * .12
    axes[1].set_ylim(min(limits) - padding, max(limits) + padding)
    axes[1].axhline(0, color="#24313A", linewidth=.7, zorder=1)
    axes[1].set_ylabel("Change in sliced-W2 (%)")
    axes[1].legend(loc="lower left", bbox_to_anchor=(0, 1.025), ncol=3, frameon=False,
                   handlelength=1, handletextpad=.5, columnspacing=.9, borderaxespad=0)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw, paired, targets, manifests = read_results(args.results)
    args.output.mkdir(parents=True, exist_ok=True)
    data = args.output / "source_data"
    data.mkdir(exist_ok=True)
    for dataset in DATASETS:
        destination = data / "datasets" / dataset
        destination.mkdir(parents=True, exist_ok=True)
        for name in ["manifest.json", "metrics.csv"]:
            source = args.results / dataset / name
            if source.resolve() != (destination / name).resolve():
                shutil.copy2(source, destination / name)
    for name, table in [("metrics", raw), ("paired_seed_errors", paired), ("paired_target_errors", targets)]:
        table.to_csv(data / f"{name}.csv", index=False)
    summary = targets.groupby(["dataset", "space"], as_index=False).agg(
        mean_relative_change=("off_relative_to_on", "mean"), sem_across_targets=("off_relative_to_on", sem))
    summary.to_csv(data / "space_summary.csv", index=False)
    overall = targets.groupby("dataset", as_index=False).agg(mean_relative_change=("off_relative_to_on", "mean"))
    overall.to_csv(data / "dataset_summary.csv", index=False)
    (data / "run_manifests.json").write_text(json.dumps(manifests, indent=2) + "\n")
    statistics = {"n_target_space_pairs": len(targets),
                  "interaction_off_worse": int((targets.off_relative_to_on > 0).sum()),
                  "interaction_off_better": int((targets.off_relative_to_on < 0).sum()),
                  "dataset_mean_relative_change": dict(zip(overall.dataset, overall.mean_relative_change))}
    (data / "caption_statistics.json").write_text(json.dumps(statistics, indent=2) + "\n")
    fig = make_figure(targets)
    for extension in ["pdf", "png"]:
        fig.savefig(args.output / f"interaction_inference_ablation.{extension}", dpi=320, facecolor="white")
    plt.close(fig)
    caption = (
        "Interaction ablation during prediction. (a) Relative reconstruction error with and without the learned interaction term, summarized across target stages and evaluation spaces within each dataset. "
        "The error with interaction is normalized to 1 in each matched comparison. (b) Percentage change in error after disabling interaction, shown separately in joint, spatial, and gene-state space. "
        "Positive values indicate higher error without interaction. White points show target stages. Bars and error bars show the mean and s.e.m. across target stages. "
        "Both conditions use the same fitted checkpoint, 5,000 initial particles, and paired random seeds (42, 43, and 44). Only the interaction-network output is set to zero. "
        "The intrinsic-drift, growth, and score networks are unchanged. Growth is retained through continuous particle weights without resampling. "
        "Predictions use a time step of 0.01 and diffusion scale of 0.03. Weighted sliced-W2 is evaluated against the same observed snapshots using five shared sets of 1,024 projection directions. "
        "Projection repeats are averaged before calculating the within-seed error ratio. The three paired seed ratios are then averaged for each target and space. "
        "Models were fitted using all measured stages.\n"
    )
    (args.output / "caption.md").write_text(caption)
    code = args.output / "code"
    code.mkdir(exist_ok=True)
    for name in ["plot_comparison.py", "run_comparison.py", "run_all.py", "cytobridge_figure_style.py", "upstream_weighted_interaction.py", "accelerated_metrics.py"]:
        source = Path(__file__).parent / name
        if source.resolve() != (code / name).resolve():
            shutil.copy2(source, code / name)
    hashes = {}
    for path in [args.output / "interaction_inference_ablation.pdf", args.output / "interaction_inference_ablation.png", code / "plot_comparison.py"]:
        hashes[str(path.relative_to(args.output))] = hashlib.sha256(path.read_bytes()).hexdigest()
    (args.output / "figure_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")
    print(json.dumps(statistics, indent=2))


if __name__ == "__main__":
    main()
