#!/usr/bin/env python3
"""Render latest zebrafish YSL and EVL ablation videos in submission style."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import shutil
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image


TIMES = np.linspace(0.0, 4.0, 81, dtype=np.float64)
MODEL_TIMES = np.asarray([0, 1, 2, 3, 4], dtype=float)
REAL_TIMES_HPF = np.asarray([5.25, 10, 12, 18, 24], dtype=float)
EXPECTED = {
    "baseline_points.npy": "1ed598e88b3c3e98fb1fd1107a289e8a4598406d70bdcd1c1067f61c11934020",
    "remove_YSL_points.npy": "b55f6ed6237148b6ec86173a17ebe1a3d35d79f7031a08597b4a839162015f56",
    "remove_EVL_points.npy": "7d18d1842716e6084e32fec729eeb1fed696b55f23104bf7ac751c8914571829",
    "classifier": "98c79a7f3c2c275de67e136c316021799152fb2f29c85ee6adb84d5e0354b80c",
    "pca": "b56b7900078d3c503d93fc43b96544306e9b541494fb56cb70871dc42397a6b6",
    "colors": "d43a9947d492d457c7a54296451253d13988be5ee6d3ad8c8a070c7e77ee9ec6",
}
CONDITIONS = {
    "baseline": "baseline_points.npy",
    "YSL": "remove_YSL_points.npy",
    "EVL": "remove_EVL_points.npy",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path: Path) -> dict[str, object]:
    path = path.resolve()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def require_sha(path: Path, expected: str) -> None:
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f"SHA mismatch for {path}: {actual}")


def object_array(values) -> np.ndarray:
    out = np.empty(len(values), dtype=object)
    for index, value in enumerate(values):
        out[index] = value
    return out


def common_limits(*frame_sets) -> tuple[tuple[float, float], tuple[float, float]]:
    lower = np.full(2, np.inf)
    upper = np.full(2, -np.inf)
    for frames in frame_sets:
        for frame in frames:
            xy = np.asarray(frame, dtype=np.float32)[:, :2]
            lower = np.minimum(lower, xy.min(axis=0))
            upper = np.maximum(upper, xy.max(axis=0))
    span = upper - lower
    pad = np.maximum(span * 0.03, 1e-6)
    return (float(lower[0] - pad[0]), float(upper[0] + pad[0])), (
        float(lower[1] - pad[1]),
        float(upper[1] + pad[1]),
    )


def stable_diff_boxes(
    baseline_frames,
    ablation_frames,
    xlim,
    ylim,
    *,
    top_k=1,
    grid_size=96,
    box_size_frac=0.18,
    min_center_distance_frac=0.16,
    corner=None,
    corner_bias=0.0,
    corner_constraint=False,
    corner_region_frac=0.58,
) -> list[dict[str, float]]:
    g = max(16, int(grid_size))
    x_edges = np.linspace(xlim[0], xlim[1], g + 1)
    y_edges = np.linspace(ylim[0], ylim[1], g + 1)
    difference = np.zeros((g, g), dtype=float)
    for baseline, ablation in zip(baseline_frames, ablation_frames):
        b = np.asarray(baseline)[:, :2]
        a = np.asarray(ablation)[:, :2]
        hb, _, _ = np.histogram2d(b[:, 0], b[:, 1], bins=[x_edges, y_edges])
        ha, _, _ = np.histogram2d(a[:, 0], a[:, 1], bins=[x_edges, y_edges])
        difference += np.abs(hb - ha)
    padded = np.pad(difference, 1, mode="edge")
    smooth = np.zeros_like(difference)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            smooth += padded[1 + dx : 1 + dx + g, 1 + dy : 1 + dy + g]
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    score = smooth.copy()
    x_norm = (x_centers - xlim[0]) / (xlim[1] - xlim[0])
    y_norm = (y_centers - ylim[0]) / (ylim[1] - ylim[0])
    if corner in {"lower_left", "upper_left"}:
        wx = 1 - x_norm
        wy = 1 - y_norm if corner == "lower_left" else y_norm
        score *= 1 + float(corner_bias) * np.outer(wx, wy)
        if corner_constraint:
            mask_x = x_norm <= corner_region_frac
            mask_y = (
                y_norm <= corner_region_frac
                if corner == "lower_left"
                else y_norm >= 1 - corner_region_frac
            )
            score = np.where(np.outer(mask_x, mask_y), score, 0.0)
    x_span = xlim[1] - xlim[0]
    y_span = ylim[1] - ylim[0]
    diagonal = np.hypot(x_span, y_span)
    width = box_size_frac * x_span
    height = box_size_frac * y_span
    selected = []
    centers = []
    for flat_index in np.argsort(score.ravel())[::-1]:
        if len(selected) >= int(top_k):
            break
        ix, iy = np.unravel_index(int(flat_index), score.shape)
        cx, cy = float(x_centers[ix]), float(y_centers[iy])
        if any(
            np.hypot(cx - old_x, cy - old_y)
            < min_center_distance_frac * diagonal
            for old_x, old_y in centers
        ):
            continue
        x0 = float(np.clip(cx - width / 2, xlim[0], xlim[1] - width))
        y0 = float(np.clip(cy - height / 2, ylim[0], ylim[1] - height))
        selected.append(
            {
                "x0": x0,
                "y0": y0,
                "w": float(width),
                "h": float(height),
                "score": float(score[ix, iy]),
            }
        )
        centers.append((cx, cy))
    return selected


def shift_and_scale_box(box, xlim, ylim, *, width_scale, height_scale, dx_frac, dy_frac):
    x_span, y_span = xlim[1] - xlim[0], ylim[1] - ylim[0]
    width = float(np.clip(box["w"] * width_scale, 0.08 * x_span, 0.52 * x_span))
    height = float(np.clip(box["h"] * height_scale, 0.06 * y_span, 0.40 * y_span))
    center_x = box["x0"] + box["w"] / 2 + dx_frac * x_span
    center_y = box["y0"] + box["h"] / 2 + dy_frac * y_span
    out = dict(box)
    out.update(
        {
            "x0": float(np.clip(center_x - width / 2, xlim[0], xlim[1] - width)),
            "y0": float(np.clip(center_y - height / 2, ylim[0], ylim[1] - height)),
            "w": width,
            "h": height,
        }
    )
    return out


def roi_schedule(target, baseline, ablation, xlim, ylim):
    if target == "YSL":
        boxes = stable_diff_boxes(
            baseline,
            ablation,
            xlim,
            ylim,
            top_k=1,
            corner="lower_left",
            corner_bias=1.2,
            corner_constraint=True,
        )
        # Center R1 on the initial YSL-removal footprint instead of leaving the
        # cumulative-density box slightly below it.
        boxes = [
            shift_and_scale_box(
                boxes[0],
                xlim,
                ylim,
                width_scale=1.0,
                height_scale=1.0,
                dx_frac=0.02,
                dy_frac=0.035,
            )
        ] if boxes else []
        return [{"t_min": 0.0, "t_max": 4.0, "boxes": boxes}]
    late_start = int(round(0.58 * (len(TIMES) - 1)))
    primary = stable_diff_boxes(
        baseline[late_start:], ablation[late_start:], xlim, ylim, top_k=1
    )
    primary = [
        shift_and_scale_box(
            primary[0],
            xlim,
            ylim,
            width_scale=2.25,
            height_scale=0.72,
            dx_frac=-0.30,
            dy_frac=0.20,
        )
    ] if primary else []
    secondary = stable_diff_boxes(
        baseline[late_start:],
        ablation[late_start:],
        xlim,
        ylim,
        top_k=1,
        box_size_frac=0.18 * 0.72,
        min_center_distance_frac=0.16 * 0.75,
        corner="lower_left",
        corner_bias=1.2,
        corner_constraint=True,
        corner_region_frac=0.62,
    )
    appear = 0.84 * 4.0
    return [
        {"t_min": 0.0, "t_max": appear, "boxes": []},
        {"t_min": appear, "t_max": 4.0, "boxes": primary + secondary},
    ]


def boxes_at(schedule, time_value):
    for segment in schedule:
        if segment["t_min"] - 1e-9 <= time_value <= segment["t_max"] + 1e-9:
            return segment["boxes"]
    return schedule[-1]["boxes"]


def hpf_label(model_time: float) -> str:
    hpf = float(np.interp(model_time, MODEL_TIMES, REAL_TIMES_HPF))
    return f"{int(round(hpf))} hpf" if abs(hpf - round(hpf)) <= 1e-6 else f"{hpf:.2f} hpf"


def paper_rc():
    return {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial"],
        "font.size": 9,
        "axes.titlesize": 10,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }


def render_frame(
    path,
    baseline,
    baseline_labels,
    ablation,
    ablation_labels,
    target,
    time_value,
    colors,
    xlim,
    ylim,
    boxes,
):
    with mpl.rc_context(paper_rc()):
        fig = plt.figure(figsize=(12.6, 6.3), dpi=200, facecolor="white")
        axes = [
            fig.add_axes((0.035, 0.045, 0.45, 0.84)),
            fig.add_axes((0.515, 0.045, 0.45, 0.84)),
        ]
        titles = ["Baseline", f"Ablation: -{'Yolk Syncytial Layer' if target == 'YSL' else 'EVL'}"]
        for axis, points, labels, title in zip(
            axes,
            (baseline, ablation),
            (baseline_labels, ablation_labels),
            titles,
        ):
            point_colors = [colors.get(str(label), "#888888") for label in labels]
            axis.scatter(
                points[:, 0],
                points[:, 1],
                s=2.2,
                c=point_colors,
                linewidths=0,
                alpha=0.82,
                rasterized=False,
            )
            axis.set(xlim=xlim, ylim=ylim)
            axis.set_aspect("equal", adjustable="box")
            axis.axis("off")
            axis.set_title(f"{title}\nt={time_value:.2f} ({hpf_label(time_value)})", pad=2)
        palette = ["#111111", "#D62728"]
        for index, box in enumerate(boxes):
            color = palette[index % len(palette)]
            for axis in axes:
                axis.add_patch(
                    Rectangle(
                        (box["x0"], box["y0"]),
                        box["w"],
                        box["h"],
                        fill=False,
                        edgecolor=color,
                        linewidth=1.9,
                        linestyle=(0, (4, 2)),
                        zorder=6,
                    )
                )
            axes[0].text(
                box["x0"],
                box["y0"] + box["h"],
                f"R{index + 1}",
                color=color,
                fontsize=8.5,
                ha="left",
                va="bottom",
                zorder=7,
                bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "edgecolor": "none", "alpha": 0.75},
            )
        fig.suptitle(
            "Zebrafish Virtual Tissue Ablation: Baseline vs Counterfactual",
            fontsize=12,
            y=0.985,
        )
        fig.savefig(path, dpi=200, facecolor="white")
        plt.close(fig)


def encode_mp4(frame_dir: Path, output: Path) -> None:
    subprocess.run(
        [
            shutil.which("ffmpeg") or __import__("imageio_ffmpeg").get_ffmpeg_exe(),
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            "10",
            "-i",
            str(frame_dir / "frame_%03d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            "-movflags",
            "+faststart",
            str(output),
        ],
        check=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", required=True, type=Path)
    parser.add_argument("--trajectory-root", required=True, type=Path)
    parser.add_argument("--classifier", required=True, type=Path)
    parser.add_argument("--pca", required=True, type=Path)
    parser.add_argument("--colors", required=True, type=Path)
    parser.add_argument("--reference-labels", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    inputs = {}
    for condition, filename in CONDITIONS.items():
        path = args.trajectory_root.resolve() / filename
        require_sha(path, EXPECTED[filename])
        inputs[condition] = path
    require_sha(args.classifier.resolve(), EXPECTED["classifier"])
    require_sha(args.pca.resolve(), EXPECTED["pca"])
    require_sha(args.colors.resolve(), EXPECTED["colors"])

    sys.path.insert(0, str(args.package_root.resolve()))
    import CytoBridge as cb

    trajectories = {
        condition: np.load(path, allow_pickle=True)
        for condition, path in inputs.items()
    }
    for condition, frames in trajectories.items():
        if frames.ndim != 1 or len(frames) != 81:
            raise RuntimeError(f"Unexpected trajectory for {condition}")
    cached = cb.tl.load_cached_mlp_classifier(str(args.classifier.resolve()), device="cpu")
    if (
        cached.feature_dim != 12
        or not cached.include_time_feature
        or cached.label_col != "Annotation"
        or cached.metadata.get("cache_tag") != "zebrafish-paper-ablation-spatial2-pca10"
    ):
        raise RuntimeError("Ablation classifier contract mismatch")
    with np.load(args.pca.resolve(), allow_pickle=False) as archive:
        components = np.asarray(archive["components"], dtype=np.float32)
        mean = np.asarray(archive["mean"], dtype=np.float32)
    labels = {}
    for condition, frames in trajectories.items():
        features = []
        for frame in frames:
            points = np.asarray(frame, dtype=np.float32)
            pca10 = (points[:, 2:] - mean) @ components.T
            features.append(np.hstack((points[:, :2], pca10)).astype(np.float32))
        labels[condition] = object_array(
            cb.tl.predict_labels_for_trajectories(
                sde_points=object_array(features),
                ts_points=TIMES,
                model=cached.model,
                label_encoder=cached.label_encoder,
                feature_dim=12,
                device="cpu",
                knn_neighbors=10,
                include_time_feature=True,
            )
        )
    reference = np.load(args.reference_labels.resolve(), allow_pickle=True)
    reference_names = {"baseline": "Baseline", "YSL": "YSL removal", "EVL": "EVL removal"}
    for condition, display_name in reference_names.items():
        for position, frame_index in enumerate(range(0, 81, 10)):
            expected = np.asarray(reference[f"{display_name}_labels"][position]).astype(str)
            actual = np.asarray(labels[condition][frame_index]).astype(str)
            if not np.array_equal(actual, expected):
                raise RuntimeError(f"Classifier reference mismatch: {condition}, frame {frame_index}")
    label_path = output / "classifier_assigned_labels.npz"
    np.savez_compressed(
        label_path,
        times=TIMES,
        baseline=labels["baseline"],
        remove_YSL=labels["YSL"],
        remove_EVL=labels["EVL"],
    )
    color_map = json.loads(args.colors.resolve().read_text())
    roi_payload = {}
    video_paths = []
    preview_paths = []
    for target, filename in (
        ("YSL", "Supplementary_Video_4_Zebrafish_YSL_Ablation.mp4"),
        ("EVL", "Supplementary_Video_5_Zebrafish_EVL_Ablation.mp4"),
    ):
        baseline = trajectories["baseline"]
        ablation = trajectories[target]
        xlim, ylim = common_limits(baseline, ablation)
        schedule = roi_schedule(target, baseline, ablation, xlim, ylim)
        roi_payload[target] = {"xlim": xlim, "ylim": ylim, "schedule": schedule}
        frame_dir = output / f"_{target.lower()}_frames"
        frame_dir.mkdir()
        for frame_index, time_value in enumerate(TIMES):
            render_frame(
                frame_dir / f"frame_{frame_index:03d}.png",
                np.asarray(baseline[frame_index]),
                np.asarray(labels["baseline"][frame_index]).astype(str),
                np.asarray(ablation[frame_index]),
                np.asarray(labels[target][frame_index]).astype(str),
                target,
                float(time_value),
                color_map,
                xlim,
                ylim,
                boxes_at(schedule, float(time_value)),
            )
        video_path = output / filename
        encode_mp4(frame_dir, video_path)
        video_paths.append(video_path)
        preview = Image.new("RGB", (2520 * 3, 1260), "white")
        for column, frame_index in enumerate((0, 40, 80)):
            preview.paste(Image.open(frame_dir / f"frame_{frame_index:03d}.png").convert("RGB"), (2520 * column, 0))
        preview_path = output / f"{target}_start_mid_end_preview.png"
        preview.save(preview_path)
        preview_paths.append(preview_path)
        for path in frame_dir.iterdir():
            path.unlink()
        frame_dir.rmdir()
    roi_path = output / "roi_schedule.json"
    roi_path.write_text(json.dumps(roi_payload, indent=2, sort_keys=True) + "\n")
    caption_path = output / "CAPTIONS.md"
    caption_path.write_text(
        "**Supplementary Video 4.** Baseline and YSL-removal trajectories generated continuously from t=0 to t=4.\n\n"
        "**Supplementary Video 5.** Baseline and EVL-removal trajectories generated continuously from t=0 to t=4.\n\n"
        "Colors denote classifier-assigned cell states. Dashed boxes mark regions with the largest spatial differences between the paired trajectories.\n"
    )
    provenance = output / "PROVENANCE.md"
    provenance.write_text(
        "# Provenance\n\n"
        "- Simulation: seed-42 growth-on split SDE with the frozen learned-interaction support gate.\n"
        "- Time grid: 0 to 4 in 0.05 increments, 81 frames.\n"
        "- Video: 2520 x 1260, H.264, 10 frames per second.\n"
        "- Cell colors: frozen spatial2+PCA10 time-aware ablation classifier with k=10 spatial smoothing.\n"
        "- Layout and region boxes follow the submitted Supplementary Videos 4 and 5.\n"
    )
    outputs = video_paths + preview_paths + [label_path, roi_path, caption_path, provenance]
    manifest_path = output / "video_manifest.json"
    manifest = {
        "status": "complete",
        "analysis": "latest_zebrafish_ablation_submission_videos",
        "simulation_commit": "545562ff7d119d46c5d9375edebec7a67efe3a92",
        "inputs": [
            *(record(path) for path in inputs.values()),
            record(args.classifier.resolve()),
            record(args.pca.resolve()),
            record(args.colors.resolve()),
            record(args.reference_labels.resolve()),
        ],
        "outputs": [record(path) for path in outputs],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (output / "video_manifest.json.sha256").write_text(
        f"{sha256(manifest_path)}  {manifest_path.name}\n"
    )
    print(json.dumps({"status": "complete", "output": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
