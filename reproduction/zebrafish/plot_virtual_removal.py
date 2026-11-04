"""Draw S33 and S34 from completed virtual-removal simulations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib as mpl
import numpy as np
import pandas as pd
from scipy import stats

import CytoBridge as cb
from reproduction.zebrafish.classifier import CLASSIFIER_FILE, load_classifier, assign_cell_types
from CytoBridge.results.zebrafish_si import PackedSpatialFrames
from CytoBridge.results._zebrafish_si_plot import _RC, _render_virtual_removal_morphology, _render_virtual_removal_quantitative


def collect_metrics(run_dirs):
    seeds, tables = {}, []
    for directory in map(Path, run_dirs):
        seed = int(json.loads((directory / "run_summary.json").read_text())["seed"])
        if seed in seeds:
            raise ValueError(f"Seed {seed} was supplied more than once")
        seeds[seed] = directory
        table = pd.read_csv(directory / "experiment/ablation_metrics.csv")
        spatial = table[table.space.eq("spatial")].copy()
        for variant in ("remove_YSL", "remove_EVL"):
            times = spatial.loc[spatial.variant.eq(variant), "time"].to_numpy(float)
            if times.shape != (81,) or not np.allclose(times, np.linspace(0, 4, 81)):
                raise ValueError(f"Seed {seed}, {variant} lacks the complete time grid")
        if not np.isfinite(spatial[["w1", "centroid_shift"]].to_numpy()).all():
            raise ValueError(f"Seed {seed} contains undefined comparison metrics")
        tables.append(spatial.assign(seed=seed))
    if set(seeds) != set(range(42, 47)):
        raise ValueError("The paper comparison uses seeds 42–46")
    data = pd.concat(tables, ignore_index=True)
    curve = data.groupby(["variant", "time"]).w1.agg(mean="mean", sem="sem").reset_index()
    endpoint = data[np.isclose(data.time, 4)][["variant", "seed", "centroid_shift"]]
    rows = []
    for variant, group in endpoint.groupby("variant"):
        mean, sem = group.centroid_shift.mean(), group.centroid_shift.sem()
        low, high = stats.t.interval(.95, len(group)-1, loc=mean, scale=sem) if sem else (mean, mean)
        rows.append(dict(variant=variant, mean=mean, sem=sem, ci95_low=low, ci95_high=high))
    return seeds, curve, endpoint, pd.DataFrame(rows)


def draw(data_dir, run_dirs, output_dir, device="cpu", classifier_path=None):
    data, output = Path(data_dir), Path(output_dir)
    seeds, curve, centroids, summary = collect_metrics(run_dirs)
    cached = load_classifier(data, device, classifier_path=classifier_path)
    frames, endpoints = [], {}
    for condition, name in (("baseline", "Baseline"), ("remove_YSL", "YSL removal"), ("remove_EVL", "EVL removal")):
        # This file is an array produced by the preceding simulation command.
        points = np.load(seeds[42] / f"experiment/trajectories/{condition}_points.npy", allow_pickle=True)
        endpoints[condition] = np.asarray(points[-1][:, :2])
        for time in range(5):
            frame = np.asarray(points[time*20], dtype=np.float32)
            labels = assign_cell_types(frame, time, cached, device)
            frames.append((name, time, frame[:, :2], labels))
    names = np.unique(np.concatenate([labels for _, _, _, labels in frames]))
    packed = PackedSpatialFrames(
        xy=np.concatenate([xy for _, _, xy, _ in frames]),
        label_id=np.concatenate([np.searchsorted(names, labels) for _, _, _, labels in frames]),
        offsets=np.cumsum([0, *(len(xy) for _, _, xy, _ in frames)]),
        groups=np.asarray([name for name, _, _, _ in frames]), times=np.asarray([time for _, time, _, _ in frames]),
        label_names=names)
    colors = json.loads((Path(__file__).resolve().parents[2] / "CytoBridge/results/data/zebrafish_si/celltype_colors.json").read_text())
    if not set(names).issubset(colors):
        raise ValueError("The paper palette lacks one or more predicted cell types")
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "virtual_removal.npz", **vars(packed))
    np.savez_compressed(output / "endpoint_spatial.npz", **endpoints)
    curve.to_csv(output / "spatial_w1_curve.csv", index=False)
    centroids.to_csv(output / "centroid_by_seed.csv", index=False)
    summary.to_csv(output / "centroid_summary.csv", index=False)
    (output / "classification.json").write_text(json.dumps({
        "classifier": str(Path(classifier_path).resolve()) if classifier_path is not None else str((data / CLASSIFIER_FILE).resolve()),
        "features": "time, two aligned spatial coordinates, 50 original expression PCs",
        "spatial_neighbors": 10,
        "display_seed": 42,
        "simulation_directories": [str(seeds[seed]) for seed in sorted(seeds)],
    }, indent=2) + "\n")
    results = SimpleNamespace(virtual_removal=packed, celltype_colors=colors,
                              endpoint_baseline_xy=endpoints["baseline"], endpoint_ysl_xy=endpoints["remove_YSL"],
                              endpoint_evl_xy=endpoints["remove_EVL"], ablation_w1_curve=curve,
                              ablation_centroid_by_seed=centroids)
    with mpl.rc_context(_RC):
        return [*_render_virtual_removal_morphology(results, output),
                *_render_virtual_removal_quantitative(results, SimpleNamespace(ablation_centroid_summary=summary), output)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/zebrafish"))
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--classifier-cache", type=Path)
    args = parser.parse_args()
    print(draw(args.data_dir, args.run_dir, args.output_dir, args.device, args.classifier_cache))
