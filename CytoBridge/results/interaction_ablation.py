"""Matched LR-prior and inference-time interaction ablations for Figure S42."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ._io import prepare_output_dir, read_manifest, resolve_results_dir
from .interaction_evidence import (
    DATASET_ORDER, NO_LR_TARGETS, SPACE_ORDER, _sem, _validate_no_lr,
)

NO_LR = "LR-prior ablation"
INTERACTION = "Interaction disabled during inference"


@dataclass(frozen=True)
class InteractionAblationResults:
    source_dir: Path
    manifest: dict
    no_lr: pd.DataFrame
    inference_metrics: pd.DataFrame
    paired_seeds: pd.DataFrame
    interaction: pd.DataFrame
    panel_summary: pd.DataFrame


def pair_inference_errors(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Average projections, form paired seed ratios, then average the seeds."""
    keys = ["dataset", "inference_seed", "target", "space", "arm", "projection_repeat"]
    required = set(keys + ["sliced_w2", "projection_sha256", "n_projections"])
    if not required.issubset(raw.columns):
        raise ValueError(f"Missing metric columns: {sorted(required - set(raw.columns))}")
    expected = {
        (dataset, seed, target, space, arm, repeat)
        for dataset in DATASET_ORDER for target in NO_LR_TARGETS[dataset]
        for seed in (42, 43, 44) for space in SPACE_ORDER
        for arm in ("interaction_on", "interaction_off") for repeat in range(5)
    }
    observed = set(raw[keys].itertuples(index=False, name=None))
    if raw.duplicated(keys).any() or observed != expected:
        raise ValueError("Expected both arms, three paired seeds, and five projection repeats for every target and space")
    values = raw["sliced_w2"].to_numpy(float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Sliced-W2 values must be finite and positive")
    if not raw["n_projections"].eq(1024).all():
        raise ValueError("Expected 1,024 projection directions per repeat")
    basis = raw.groupby(["dataset", "target", "space", "projection_repeat"])["projection_sha256"]
    if raw["projection_sha256"].isna().any() or not basis.nunique().eq(1).all():
        raise ValueError("Projection directions differ between paired arms or seeds")
    means = raw.groupby(keys[:5], as_index=False)["sliced_w2"].mean()
    paired = means.pivot(index=keys[:4], columns="arm", values="sliced_w2").reset_index()
    paired.columns.name = None
    paired["off_relative_to_on"] = paired["interaction_off"] / paired["interaction_on"] - 1
    targets = paired.groupby(["dataset", "target", "space"], as_index=False).agg(
        off_relative_to_on=("off_relative_to_on", "mean"),
        interaction_on=("interaction_on", "mean"),
        interaction_off=("interaction_off", "mean"),
    )
    return paired, targets


def _summary(table: pd.DataFrame, column: str, comparison: str) -> pd.DataFrame:
    result = table.groupby(["dataset", "space"])[column].agg(
        n_targets="size", mean_relative_difference="mean", sem=_sem,
    ).reset_index()
    result["dataset_mean_relative_difference"] = result.dataset.map(table.groupby("dataset")[column].mean())
    result.insert(0, "comparison", comparison)
    return result


def load_interaction_ablation_results(results_dir: str | Path | None = None) -> InteractionAblationResults:
    """Read numerical results for S42. Omit the path to use the included tables."""
    root = resolve_results_dir(results_dir, slug="interaction_ablation")
    no_lr_path = root / "no_lr_paired_target_deltas.csv"
    no_lr = _validate_no_lr(pd.read_csv(no_lr_path), no_lr_path)
    raw = pd.read_csv(root / "inference_metrics.csv")
    paired, targets = pair_inference_errors(raw)
    summary = pd.concat([
        _summary(no_lr, "no_lr_prior_relative_to_full", NO_LR),
        _summary(targets, "off_relative_to_on", INTERACTION),
    ], ignore_index=True)
    return InteractionAblationResults(root, read_manifest(root), no_lr, raw, paired, targets, summary)


def interaction_ablation_statistics(results: InteractionAblationResults) -> dict:
    values = results.interaction.off_relative_to_on
    return {
        "no_lr_dataset_percent_increase": (100 * results.no_lr.groupby("dataset").no_lr_prior_relative_to_full.mean()).to_dict(),
        "interaction_off_dataset_percent_increase": (100 * results.interaction.groupby("dataset").off_relative_to_on.mean()).to_dict(),
        "interaction_target_space_pairs": len(values),
        "interaction_off_worse": int((values > 0).sum()),
        "interaction_off_better": int((values < 0).sum()),
        "inference_seeds": [42, 43, 44],
    }


def write_interaction_ablation_tables(results: InteractionAblationResults, output_dir: str | Path) -> dict[str, Path]:
    output = prepare_output_dir(output_dir)
    tables = {
        "no_lr_paired_target_deltas": results.no_lr,
        "inference_metrics": results.inference_metrics,
        "paired_seed_errors": results.paired_seeds,
        "paired_target_errors": results.interaction,
        "panel_summary": results.panel_summary,
    }
    paths = {}
    for name, table in tables.items():
        paths[name] = output / f"{name}.csv"
        table.to_csv(paths[name], index=False)
    paths["caption_statistics"] = output / "caption_statistics.json"
    paths["caption_statistics"].write_text(json.dumps(interaction_ablation_statistics(results), indent=2) + "\n")
    return paths


def plot_interaction_ablation(results: InteractionAblationResults, output_dir: str | Path) -> tuple[Path, Path]:
    """Draw all four S42 panels from numerical tables, without reading images."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from ._interaction_evidence_plot import _plot_normalized_aggregate, _plot_relative_effects
    from ._style import INTERACTION_RC, interaction_panel_heading, save_figure

    output = prepare_output_dir(output_dir)
    pdf, png = output / "interaction_ablation.pdf", output / "interaction_ablation.png"
    with mpl.rc_context(INTERACTION_RC):
        fig = plt.figure(figsize=(8.27, 11.69))
        outer = fig.add_gridspec(2, 2, left=.09, right=.97, bottom=.07, top=.975, wspace=.24, hspace=.25)
        panels = []
        for spec in outer:
            inner = spec.subgridspec(2, 1, height_ratios=[.12, .88], hspace=.03)
            panels.append((fig.add_subplot(inner[0]), fig.add_subplot(inner[1])))
        for row, table, column, comparison, baseline, other in (
            (0, results.no_lr, "no_lr_prior_relative_to_full", NO_LR, "Full model", "No LR prior"),
            (1, results.interaction, "off_relative_to_on", INTERACTION, "With interaction", "Without interaction"),
        ):
            heading, ax = panels[2 * row]
            interaction_panel_heading(heading, "ac"[row], "LR-prior ablation" if row == 0 else "Interaction ablation")
            _plot_normalized_aggregate(ax, results.panel_summary, comparison=comparison,
                baseline_label=baseline, comparison_label=other,
                baseline_color="#07838B", comparison_color="#CC6677")
            ax.set_ylabel(f"Relative sliced-W2\n({baseline.lower()} = 1)")
            ax.legend(loc="upper left", bbox_to_anchor=(0, 1.015), frameon=False, ncol=2,
                      handlelength=1, handletextpad=.5, columnspacing=.8, borderaxespad=0)
            heading, ax = panels[2 * row + 1]
            interaction_panel_heading(heading, "bd"[row], "Effect by evaluation space")
            _plot_relative_effects(ax, table, results.panel_summary, comparison=comparison,
                                  value_column=column, reference_label=baseline)
        save_figure(fig, pdf, png, dpi=320, pdf_metadata={"Creator": "CytoBridge"}, png_metadata={"Software": "CytoBridge"})
        plt.close(fig)
    return pdf, png
