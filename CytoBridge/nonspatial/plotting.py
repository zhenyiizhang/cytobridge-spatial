"""Draw S4/S5 comparison panels from completed non-spatial evaluations."""

from pathlib import Path
import json

import numpy as np
import pandas as pd


def distribution_comparison(table: pd.DataFrame) -> pd.DataFrame:
    """Pair Full and No-interaction distances, averaging inference repeats.

    Accepts the ``paired_distribution_metrics.csv`` written by
    :func:`evaluate_nonspatial_pair` or the paper's paired wide table.
    """
    table = table.copy()
    if "space" in table:
        table = table.loc[table["space"].eq("pca")].copy()
    metrics = ("w1", "w2")
    if "condition" in table:
        required = {"time", "condition", *metrics}
        if not required.issubset(table):
            raise ValueError(f"Missing distribution columns: {sorted(required - set(table))}")
        if set(table["condition"]) != {"full", "no_interaction"}:
            raise ValueError("Both Full and No-interaction evaluations are required")
        key = ["time"]
        if "inference_seed" in table:
            key.append("inference_seed")
        elif "inference_repeat" in table:
            key.append("inference_repeat")
        if table.duplicated([*key, "condition"]).any():
            raise ValueError("Duplicate rows for a time, seed and condition")
        pairs = table.groupby(key)["condition"].nunique()
        if not pairs.eq(2).all():
            raise ValueError("Full and No-interaction must have matching times and seeds")
        table = table.groupby(["time", "condition"])[list(metrics)].mean().unstack("condition")
        table.columns = [f"{metric}_{condition}" for metric, condition in table.columns]
        table = table.reset_index()
    columns = ["time", *(f"{metric}_{condition}" for metric in metrics for condition in ("full", "no_interaction"))]
    if not set(columns).issubset(table):
        raise ValueError(f"Missing paired distribution columns: {sorted(set(columns) - set(table))}")
    table = table[columns].apply(pd.to_numeric, errors="raise").sort_values("time")
    if table.empty or table["time"].duplicated().any() or not np.isfinite(table.to_numpy()).all():
        raise ValueError("Distribution values must be finite, with one row per time")
    if (table.drop(columns="time") < 0).any().any():
        raise ValueError("Distribution distances cannot be negative")
    return table.reset_index(drop=True)


def plot_nonspatial_evaluation(
    dataset: str,
    distribution_csv: str | Path,
    output_dir: str | Path,
    *,
    clone_fate_dir: str | Path | None = None,
    direction_csv: str | Path | None = None,
) -> dict[str, Path]:
    """Draw the distribution and optional fate/direction panels of S4 or S5.

    Parameters
    ----------
    dataset
        ``weinreb`` or ``scnt_cortex``.
    distribution_csv
        CSV written by ``evaluate_nonspatial_pair``. The paired distribution
        table included in each paper-data archive is also supported.
    output_dir
        A new or empty directory for PDF, PNG and the plotted numerical tables.
    clone_fate_dir
        Optional directory written by ``evaluate_weinreb_clone_fate``.
    direction_csv
        Optional ``timewise_scnt_direction_alignment.csv`` written by the
        scNT direction analysis.

    Returns
    -------
    dict
        Output file paths. The plotting functions are the same ones used by
        the S4/S5 notebook. No model is trained and no existing image is read.
    """
    if dataset not in {"weinreb", "scnt_cortex"}:
        raise ValueError("dataset must be 'weinreb' or 'scnt_cortex'")
    if clone_fate_dir is not None and dataset != "weinreb":
        raise ValueError("Clone-fate panels are available for Weinreb")
    if direction_csv is not None and dataset != "scnt_cortex":
        raise ValueError("New-RNA direction panels are available for scNT")
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Choose a new output directory: {output}")
    table = distribution_comparison(pd.read_csv(distribution_csv))
    expected = [1., 2.] if dataset == "weinreb" else [0.25, 0.5, 1., 2.]
    if table["time"].tolist() != expected:
        raise ValueError(f"The {dataset} paper panel uses model times {expected}")
    output.mkdir(parents=True, exist_ok=True)
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from CytoBridge.results import _nonspatial_figures_plot as draw

    paths = {}

    def save(fig, name, values):
        for suffix in ("pdf", "png"):
            path = output / f"{name}.{suffix}"
            fig.savefig(path, dpi=320, bbox_inches="tight", facecolor="white")
            paths[f"{name}_{suffix}"] = path
        plt.close(fig)
        path = output / f"{name}.csv"
        values.to_csv(path, index=False)
        paths[f"{name}_csv"] = path

    with mpl.rc_context(draw._rc("#000000")):
        if dataset == "weinreb":
            fig, axis = plt.subplots(figsize=(5.8, 3.0))
            draw._weinreb_distribution(axis, table)
            axis.legend(handles=draw._condition_handles(), loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=2, frameon=False)
        else:
            fig = plt.figure(figsize=(5.8, 3.2))
            draw._scnt_distribution(fig, fig.add_gridspec(1, 1)[0], table)
        save(fig, "distribution_comparison", table)

        if clone_fate_dir is not None:
            clone_dir = Path(clone_fate_dir)
            summaries = {condition: json.loads((clone_dir / condition / "summary.json").read_text())
                         for condition in ("full", "no_interaction")}
            clone = pd.DataFrame([
                {"metric": metric, **{condition: summaries[condition][f"clone_macro_{metric}"]
                                     for condition in summaries}}
                for metric in ("tv_agreement", "js_similarity", "dominant_fate_match")
            ])
            if not np.isfinite(clone[["full", "no_interaction"]].to_numpy(float)).all():
                raise ValueError("Clone-fate summaries contain non-finite values")
            fig = plt.figure(figsize=(5.8, 2.5))
            draw._weinreb_clone(fig, fig.add_gridspec(1, 1)[0], clone)
            fig.axes[1].set_yticklabels(["Clone average"])
            save(fig, "clone_fate_comparison", clone)

        if direction_csv is not None:
            direction = pd.read_csv(direction_csv)
            conditions = {"full_interaction_noise", "no_interaction_noise"}
            direction = direction[direction["condition"].isin(conditions)]
            pair_counts = direction.groupby("time_hours")["condition"].nunique()
            if set(pair_counts.index) != set(expected) or not pair_counts.eq(2).all():
                raise ValueError("Direction evaluations must cover both conditions at every endpoint")
            values = direction.groupby("condition", as_index=False)[["cell_cosine_mean", "cell_cosine_median"]].mean()
            if not np.isfinite(values.iloc[:, 1:].to_numpy(float)).all():
                raise ValueError("Direction summaries contain non-finite values")
            fig = plt.figure(figsize=(5.8, 3.0))
            draw._scnt_direction(fig, fig.add_gridspec(1, 1)[0], values)
            # Preserve the paper's range when possible, but do not hide values
            # from a new model that fall outside it.
            low, high = float(values.iloc[:, 1:].min().min()), float(values.iloc[:, 1:].max().max())
            if low < 0 or high > 0.014:
                pad = max(0.002, (high - low) * 0.12)
                fig.axes[-1].set_xlim(min(0, low - pad), max(0.014, high + pad))
            save(fig, "direction_comparison", values)
    return paths
