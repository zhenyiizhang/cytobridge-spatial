"""Draw S38 from newly calculated observed and inverse-PCA expression tables."""
from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import matplotlib as mpl
import numpy as np
import pandas as pd

from CytoBridge.results._zebrafish_si_plot import _RC, _render_inverse_pca


def read_expression(path, times):
    table = pd.read_csv(path, index_col=0)
    table.index = table.index.astype(str)
    table.columns = table.columns.astype(float)
    if table.index.duplicated().any() or table.columns.tolist() != list(times):
        raise ValueError(f"Unexpected genes or time points in {path}")
    if not np.isfinite(table.to_numpy(float)).all():
        raise ValueError(f"Non-finite expression values in {path}")
    return table


def temporal_zscores(table):
    values = table.to_numpy(float)
    std = values.std(axis=1, ddof=0)
    std[std == 0] = 1.
    zscores = (values - values.mean(axis=1, keepdims=True)) / std[:, None]
    order = np.lexsort((-np.var(zscores, axis=1), np.argmax(zscores, axis=1)))
    return pd.DataFrame(zscores[order], index=table.index[order], columns=table.columns)


def reconstruction_metrics(observed, reconstructed):
    if not observed.index.equals(reconstructed.index) or not observed.columns.equals(reconstructed.columns):
        raise ValueError("Observed and reconstructed tables must use the same genes and times")
    rows = []
    for time in observed.columns:
        x, y = observed[time].to_numpy(float), reconstructed[time].to_numpy(float)
        if np.std(x) == 0 or np.std(y) == 0:
            raise ValueError(f"Correlation is undefined for constant expression at t={time}")
        delta = y - x
        rows.append(dict(time=time, n_features=len(x), rmse=np.sqrt(np.mean(delta**2)),
                         mae=np.mean(abs(delta)), mean_bias=np.mean(delta), pearson_r=np.corrcoef(x, y)[0, 1]))
    return pd.DataFrame(rows)


def draw(run_dir, output_dir):
    stage, output = Path(run_dir) / "s25", Path(output_dir)
    expression = read_expression(stage / "mean_inverse_pca_log1p_clipped.csv", np.arange(0., 4.5, .5))
    observed = read_expression(stage / "observed_exact_log1p_anchors.csv", np.arange(5.))
    top = pd.read_csv(stage / "top_variable_genes.csv").gene.astype(str)
    if len(top) != 250 or top.duplicated().any() or not set(top).issubset(expression.index):
        raise ValueError("The S38 display requires the selected 250 temporal genes")
    reconstructed = expression.loc[observed.index, observed.columns]
    metrics = reconstruction_metrics(observed, reconstructed)
    output.mkdir(parents=True, exist_ok=False)
    observed.to_csv(output / "observed_expression.csv")
    reconstructed.to_csv(output / "reconstructed_expression.csv")
    metrics.to_csv(output / "reconstruction_metrics.csv", index=False)
    top.to_csv(output / "top_genes.csv", index=False)
    results = SimpleNamespace(observed_expression=observed, reconstructed_expression=reconstructed,
                              top_variable_genes=tuple(top))
    panels = SimpleNamespace(inverse_pca_metrics=metrics)
    with mpl.rc_context(_RC):
        return list(_render_inverse_pca(results, panels, output))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(draw(args.run_dir, args.output_dir))
