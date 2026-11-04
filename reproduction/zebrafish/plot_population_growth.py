"""Draw S31 and S32 from newly calculated zebrafish states and growth rates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib as mpl
import numpy as np
import pandas as pd

from CytoBridge.results.zebrafish_si import PackedSpatialFrames
from CytoBridge.results._zebrafish_si_plot import _RC, _render_observed_generated, _render_growth


def pack_states(run_dir):
    """Read observed and simulated state arrays written by the analysis command."""
    root = Path(run_dir) / "s22"
    frames = []
    for group, name, expected in (
        ("observed", "observed_reference_states", np.arange(5.)),
        ("generated", "global_t0_fixed_population_states", np.arange(0., 4.5, .5)),
    ):
        folder = root / name
        index = json.loads((folder / "index.json").read_text())
        records = sorted(index["frames"], key=lambda entry: entry["time"])
        if [entry["time"] for entry in records] != expected.tolist():
            raise ValueError(f"Unexpected time points in {folder}")
        for entry in records:
            with np.load(folder / entry["file"], allow_pickle=False) as data:
                xy, labels = np.asarray(data["points"][:, :2]), data["labels"].astype(str)
            if len(xy) == 0 or len(xy) != len(labels) or not np.isfinite(xy).all():
                raise ValueError(f"Invalid states in {folder / entry['file']}")
            frames.append((group, float(entry["time"]), xy, labels))
    generated_counts = [len(xy) for group, _, xy, _ in frames if group == "generated"]
    if len(set(generated_counts)) != 1:
        raise ValueError("S31 follows a fixed-size initial population")
    names = np.unique(np.concatenate([labels for _, _, _, labels in frames]))
    return PackedSpatialFrames(
        xy=np.concatenate([xy for _, _, xy, _ in frames]),
        label_id=np.concatenate([np.searchsorted(names, labels) for _, _, _, labels in frames]),
        offsets=np.cumsum([0, *(len(xy) for _, _, xy, _ in frames)]),
        groups=np.asarray([group for group, _, _, _ in frames]),
        times=np.asarray([time for _, time, _, _ in frames]),
        label_names=names,
    )


def scale_growth(table):
    """Scale growth by the within-time 5th and 95th percentiles used in S32."""
    table = table.rename(columns={"growth_rate": "growth", "spatial_x": "x", "spatial_y": "y"}).copy()
    required = ["time", "x", "y", "growth"]
    table = table[required].apply(pd.to_numeric, errors="raise")
    if set(table.time) != set(range(5)) or not np.isfinite(table.to_numpy()).all():
        raise ValueError("Growth values must be finite at each observed time 0–4")
    quantiles = table.groupby("time").growth.quantile([.05, .95]).unstack()
    quantiles.columns = ["q05", "q95"]
    table = table.merge(quantiles, on="time", validate="many_to_one")
    table["growth_scaled"] = np.clip(
        (table.growth - table.q05) / np.maximum(table.q95 - table.q05, 1e-12), 0, 1)
    return table, quantiles.reset_index()


def draw(run_dir, aligned_h5ad, output_dir):
    """Collect the preceding analysis outputs and draw the two paper figures."""
    import anndata as ad

    run, output = Path(run_dir), Path(output_dir)
    packed = pack_states(run)
    scaled, quantiles = scale_growth(pd.read_csv(run / "growth/growth_per_cell.csv"))
    observed = ad.read_h5ad(aligned_h5ad, backed="r")
    try:
        if "Color" in observed.obs:
            palette = observed.obs[["Annotation", "Color"]].astype(str)
            colors = palette.groupby("Annotation", observed=True).Color.agg(lambda values: values.mode().iloc[0]).to_dict()
        else:
            # The publication palette is a style input, not a saved result.
            palette_path = Path(__file__).resolve().parents[2] / "CytoBridge/results/data/zebrafish_si/s27_celltype_colors.json"
            colors = json.loads(palette_path.read_text())
    finally:
        observed.file.close()
    if not set(packed.label_names).issubset(colors):
        raise ValueError("The aligned data lacks colors for one or more predicted cell types")
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "observed_generated.npz", **vars(packed))
    scaled.to_csv(output / "growth_scaled.csv", index=False)
    quantiles.to_csv(output / "growth_quantiles.csv", index=False)
    (output / "celltype_colors.json").write_text(json.dumps(colors, indent=2) + "\n")
    results = SimpleNamespace(observed_generated=packed, observed_generated_colors=colors)
    with mpl.rc_context(_RC):
        paths = [*_render_observed_generated(results, output),
                 *_render_growth(results, SimpleNamespace(growth_scaled=scaled), output)]
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--aligned-h5ad", type=Path, default=Path("data/zebrafish/aligned.h5ad"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(draw(args.run_dir, args.aligned_h5ad, args.output_dir))
