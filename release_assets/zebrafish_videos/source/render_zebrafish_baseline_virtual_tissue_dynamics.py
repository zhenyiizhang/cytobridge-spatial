#!/usr/bin/env python3
"""Render the growth-on zebrafish baseline with learned interaction edges."""

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
from matplotlib import cm, colors
import numpy as np
import pandas as pd
from PIL import Image
import torch


EXPECTED_RELEASE_COMMIT = "545562ff7d119d46c5d9375edebec7a67efe3a92"
EXPECTED = {
    "trajectory": "1ed598e88b3c3e98fb1fd1107a289e8a4598406d70bdcd1c1067f61c11934020",
    "h5ad": "14753bbfdd05c9971b4ed5db4a7e70693479c7b7074ed1ef1d6f3187e1119811",
    "model": "c651666c65570357e7f63f81fe63981a43ae2c06736e22420dabfb77b98bb824",
}
TIMES = np.linspace(0.0, 4.0, 81, dtype=np.float64)
SAMPLED = np.arange(0, 81, 2, dtype=np.int64)
TOP_EDGES = 260


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path: Path) -> dict[str, object]:
    path = path.resolve()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


class ObservedSupportLinkPredictor(torch.nn.Module):
    """Keep learned logits only when both latent endpoints remain in support."""

    def __init__(self, base, lower: np.ndarray, upper: np.ndarray, max_norm: float):
        super().__init__()
        self.base = base
        self.register_buffer("lower", torch.as_tensor(lower, dtype=torch.float32))
        self.register_buffer("upper", torch.as_tensor(upper, dtype=torch.float32))
        self.max_norm = float(max_norm)

    def endpoint_inside(self, state: torch.Tensor) -> torch.Tensor:
        lower = self.lower.to(device=state.device, dtype=state.dtype)
        upper = self.upper.to(device=state.device, dtype=state.dtype)
        return torch.all((state >= lower) & (state <= upper), dim=1) & (
            torch.linalg.vector_norm(state, dim=1) <= self.max_norm
        )

    def forward(self, pair_features: torch.Tensor) -> torch.Tensor:
        logits = self.base(pair_features)
        state_dim = pair_features.shape[1] // 2
        source = pair_features[:, :state_dim][:, 2:]
        target = pair_features[:, state_dim:][:, 2:]
        accepted = self.endpoint_inside(source) & self.endpoint_inside(target)
        return torch.where(
            accepted.reshape_as(logits), logits, torch.full_like(logits, -1e9)
        )


def require_sha(path: Path, expected: str, label: str) -> None:
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f"{label} SHA mismatch: {actual}")


def release_commit(release: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(release), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def extract_edges(args: argparse.Namespace) -> int:
    release = args.release.resolve()
    trajectory_path = args.trajectory.resolve()
    h5ad_path = args.h5ad.resolve()
    model_dir = args.model_dir.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    if args.verify_paper_inputs:
        require_sha(trajectory_path, EXPECTED["trajectory"], "trajectory")
        require_sha(h5ad_path, EXPECTED["h5ad"], "h5ad")
        require_sha(model_dir / "Finetune" / "best_model.pth", EXPECTED["model"], "model")

    sys.path.insert(0, str(release))
    import anndata as ad
    import CytoBridge as cb
    from CytoBridge.tl.downstream.attention import (
        _first_layer_attention,
        _radius_neighbor_candidates,
    )

    trajectories = np.load(trajectory_path, allow_pickle=True)
    if trajectories.ndim != 1 or len(trajectories) != len(TIMES):
        raise RuntimeError("Unexpected baseline trajectory structure")
    adata = ad.read_h5ad(h5ad_path, backed="r")
    observed_latent = np.asarray(adata.obsm["X_latent"], dtype=np.float32)
    adata.file.close()
    loaded = cb.tl.load_dynamical_model_from_dir(
        model_dir, dim=52, device=args.device
    )
    runtime = cb.tl.build_dynamical_runtime(loaded)
    interaction = runtime.f_net.interaction_net
    interaction.link_predictor = ObservedSupportLinkPredictor(
        interaction.link_predictor,
        observed_latent.min(axis=0),
        observed_latent.max(axis=0),
        float(np.linalg.norm(observed_latent, axis=1).max()),
    ).to(args.device)
    interaction.eval()

    sources: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    strengths: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    rows: list[dict[str, object]] = []
    threshold = float(interaction.edge_predictor_thre)
    batch_size = int(args.edge_batch_size)

    for frame_index in SAMPLED:
        frame = np.asarray(trajectories[int(frame_index)], dtype=np.float32)
        data = torch.as_tensor(frame, dtype=torch.float32, device=args.device)
        candidate_source, candidate_target = _radius_neighbor_candidates(
            frame,
            cutoff=float(interaction.cutoff),
            use_spatial=bool(interaction.use_spatial),
        )
        kept_source = []
        kept_target = []
        kept_probability = []
        raw_learned_count = 0
        distance_count = 0
        with torch.no_grad():
            for start in range(0, candidate_source.size, batch_size):
                stop = min(start + batch_size, candidate_source.size)
                source = torch.as_tensor(
                    candidate_source[start:stop], dtype=torch.long, device=args.device
                )
                target = torch.as_tensor(
                    candidate_target[start:stop], dtype=torch.long, device=args.device
                )
                distance = torch.linalg.vector_norm(
                    data[source, :2] - data[target, :2], dim=1
                )
                within = (distance < float(interaction.cutoff)) & (distance > 1e-6)
                distance_count += int(within.sum().item())
                pair = torch.cat((data[source], data[target]), dim=1)
                base_probability = torch.sigmoid(
                    interaction.link_predictor.base(pair)
                ).reshape(-1)
                gated_probability = torch.sigmoid(
                    interaction.link_predictor(pair)
                ).reshape(-1)
                raw_learned_count += int((within & (base_probability >= threshold)).sum().item())
                keep = within & (gated_probability >= threshold)
                kept_source.append(source[keep])
                kept_target.append(target[keep])
                kept_probability.append(gated_probability[keep])
            if kept_source:
                source = torch.cat(kept_source)
                target = torch.cat(kept_target)
                probability = torch.cat(kept_probability)
            else:
                source = torch.empty(0, dtype=torch.long, device=args.device)
                target = torch.empty(0, dtype=torch.long, device=args.device)
                probability = torch.empty(0, dtype=data.dtype, device=args.device)
            attention = _first_layer_attention(
                interaction,
                data,
                source,
                target,
                edge_batch_size=batch_size,
            )

        source_np = source.cpu().numpy().astype(np.int64, copy=False)
        target_np = target.cpu().numpy().astype(np.int64, copy=False)
        probability_np = probability.cpu().numpy().astype(np.float32, copy=False)
        attention_np = attention.cpu().numpy().astype(np.float32, copy=False)
        if attention_np.size:
            order = np.argsort(attention_np, kind="stable")[-TOP_EDGES:]
            order = order[np.argsort(attention_np[order], kind="stable")]
        else:
            order = np.empty(0, dtype=np.int64)
        sources.append(source_np[order])
        targets.append(target_np[order])
        strengths.append(attention_np[order])
        probabilities.append(probability_np[order])
        rows.append(
            {
                "frame_index": int(frame_index),
                "time": float(TIMES[frame_index]),
                "n_cells": int(len(frame)),
                "radius_candidate_edges": int(distance_count),
                "learned_edges_before_support_gate": int(raw_learned_count),
                "learned_edges_after_support_gate": int(len(source_np)),
                "displayed_edges": int(len(order)),
                "attention_median": float(np.median(attention_np)) if attention_np.size else 0.0,
                "attention_p99": float(np.quantile(attention_np, 0.99)) if attention_np.size else 0.0,
                "attention_max": float(attention_np.max()) if attention_np.size else 0.0,
            }
        )
        print(json.dumps(rows[-1]), flush=True)

    edge_path = output / "learned_interactions.npz"
    np.savez_compressed(
        edge_path,
        sampled_indices=SAMPLED,
        source=np.asarray(sources, dtype=object),
        target=np.asarray(targets, dtype=object),
        attention=np.asarray(strengths, dtype=object),
        probability=np.asarray(probabilities, dtype=object),
    )
    summary_path = output / "interaction_frame_summary.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False, float_format="%.12g")
    manifest_path = output / "extraction_manifest.json"
    write_json(
        manifest_path,
        {
            "status": "complete",
            "release_commit": release_commit(release),
            "semantics": (
                "Frozen learned edge predictor and first-layer attention evaluated "
                "on each stored growth-on baseline state; the strongest 260 directed "
                "accepted edges are retained for display."
            ),
            "inputs": {
                "trajectory": record(trajectory_path),
                "h5ad": record(h5ad_path),
                "model": record(model_dir / "Finetune" / "best_model.pth"),
            },
            "outputs": [record(edge_path), record(summary_path)],
            "edge_predictor_threshold": threshold,
            "interaction_cutoff": float(interaction.cutoff),
            "displayed_edges_per_frame_max": TOP_EDGES,
        },
    )
    (output / "extraction_manifest.json.sha256").write_text(
        f"{sha256(manifest_path)}  {manifest_path.name}\n"
    )
    return 0


def paper_rc() -> dict[str, object]:
    return {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial"],
        "font.size": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }


def spatial_limits(trajectories) -> tuple[float, float, float, float]:
    xy = np.vstack([np.asarray(trajectories[index])[:, :2] for index in SAMPLED])
    lower = xy.min(axis=0)
    upper = xy.max(axis=0)
    pad = np.maximum((upper - lower) * 0.04, 1e-6)
    return lower[0] - pad[0], upper[0] + pad[0], lower[1] - pad[1], upper[1] + pad[1]


def draw_frame(
    frame: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
    attention: np.ndarray,
    time_value: float,
    limits: tuple[float, float, float, float],
    norm: colors.Normalize,
) -> Image.Image:
    with mpl.rc_context({**paper_rc(), "figure.dpi": 130}):
        fig = plt.figure(figsize=(7.1, 6.0))
        axis = fig.add_axes((0.045, 0.055, 0.79, 0.82))
        color_axis = fig.add_axes((0.875, 0.145, 0.025, 0.64))
        cmap = mpl.colormaps["magma"]
        scalar = cm.ScalarMappable(norm=norm, cmap=cmap)
        colorbar = fig.colorbar(scalar, cax=color_axis)
        colorbar.set_label("Mean |attention|")
        colorbar.outline.set_linewidth(0.6)
        point_size = float(np.clip(2100.0 / len(frame), 0.75, 3.2))
        axis.scatter(
            frame[:, 0],
            frame[:, 1],
            s=point_size,
            c="#B7C1C8",
            alpha=0.82,
            linewidths=0,
            rasterized=False,
            zorder=1,
        )
        if len(source):
            values = np.clip(norm(attention), 0.0, 1.0)
            xy = frame[:, :2]
            delta = xy[target] - xy[source]
            axis.quiver(
                xy[source, 0],
                xy[source, 1],
                delta[:, 0],
                delta[:, 1],
                values,
                cmap=cmap,
                norm=colors.Normalize(0.0, 1.0),
                angles="xy",
                scale_units="xy",
                scale=1.0,
                width=0.0022,
                headwidth=4.0,
                headlength=5.0,
                headaxislength=4.2,
                alpha=0.82,
                zorder=2,
            )
        axis.set(xlim=limits[:2], ylim=limits[2:])
        axis.set_aspect("equal", adjustable="box")
        axis.axis("off")
        fig.text(0.055, 0.965, "Virtual tissue dynamics", fontsize=14, fontweight="bold", va="top")
        fig.text(0.945, 0.965, f"t = {time_value:g}", fontsize=12, fontweight="bold", ha="right", va="top")
        fig.text(0.055, 0.925, f"n = {len(frame):,}", fontsize=9, color="#3C4850", va="top")
        fig.canvas.draw()
        image = Image.fromarray(np.asarray(fig.canvas.buffer_rgba()).copy(), mode="RGBA")
        plt.close(fig)
        return image


def render(args: argparse.Namespace) -> int:
    trajectory_path = args.trajectory.resolve()
    extraction = args.extraction.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    if args.verify_paper_inputs:
        require_sha(trajectory_path, EXPECTED["trajectory"], "trajectory")
    manifest_path = extraction / "extraction_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("status") != "complete":
            raise RuntimeError("Interaction extraction is incomplete")
        for item in manifest["outputs"]:
            path = extraction / Path(item["path"]).name
            if not path.is_file() or sha256(path) != item["sha256"]:
                raise RuntimeError(f"Changed interaction output: {path}")

    trajectories = np.load(trajectory_path, allow_pickle=True)
    bundle = np.load(extraction / "learned_interactions.npz", allow_pickle=True)
    if len(trajectories) != len(TIMES):
        raise ValueError(f"Expected {len(TIMES)} trajectory frames on the 0–4 time grid")
    if not np.array_equal(bundle["sampled_indices"], SAMPLED):
        raise RuntimeError("Unexpected sampled frame indices")
    for position, frame_index in enumerate(SAMPLED):
        n_cells = len(trajectories[frame_index])
        src, dst, att = (np.asarray(bundle[key][position])
                         for key in ("source", "target", "attention"))
        if not (src.shape == dst.shape == att.shape) or not np.isfinite(att).all():
            raise ValueError(f"Invalid edge arrays in frame {frame_index}")
        if any(np.any((idx < 0) | (idx >= n_cells)) for idx in (src, dst)):
            raise ValueError(f"Edge endpoints are outside frame {frame_index}")
    all_attention = np.concatenate([np.asarray(values, dtype=float) for values in bundle["attention"]])
    positive = all_attention[np.isfinite(all_attention) & (all_attention > 0)]
    if not len(positive):
        raise RuntimeError("No learned interaction attention values")
    lower = float(np.quantile(positive, 0.50))
    upper = float(np.quantile(positive, 0.995))
    if upper <= lower:
        upper = float(positive.max())
    norm = colors.LogNorm(vmin=max(lower, 1e-8), vmax=max(upper, lower * 1.001), clip=True)
    limits = spatial_limits(trajectories)
    frames = []
    representative = {}
    for position, frame_index in enumerate(SAMPLED):
        image = draw_frame(
            np.asarray(trajectories[int(frame_index)], dtype=np.float32),
            np.asarray(bundle["source"][position], dtype=np.int64),
            np.asarray(bundle["target"][position], dtype=np.int64),
            np.asarray(bundle["attention"][position], dtype=np.float32),
            float(TIMES[frame_index]),
            limits,
            norm,
        )
        frames.append(image)
        if int(frame_index) in (0, 40, 80):
            representative[int(frame_index)] = image.copy()

    gif_path = output / "zebrafish_baseline_virtual_tissue_dynamics.gif"
    palette_frames = [image.convert("P", palette=Image.Palette.ADAPTIVE) for image in frames]
    palette_frames[0].save(
        gif_path,
        save_all=True,
        append_images=palette_frames[1:],
        duration=int(round(1000 / float(args.fps))),
        loop=0,
        optimize=False,
        disposal=2,
    )
    preview_path = output / "zebrafish_baseline_virtual_tissue_dynamics_t0_t2_t4.png"
    preview = Image.new("RGB", (frames[0].width * 3, frames[0].height), "white")
    for column, frame_index in enumerate((0, 40, 80)):
        preview.paste(representative[frame_index].convert("RGB"), (column * frames[0].width, 0))
    preview.save(preview_path, dpi=(240, 240))

    mp4_path = output / "zebrafish_baseline_virtual_tissue_dynamics.mp4"
    frame_dir = output / "_frames"
    frame_dir.mkdir()
    for index, image in enumerate(frames):
        image.convert("RGB").save(frame_dir / f"frame_{index:03d}.png")
    command = [
        shutil.which("ffmpeg") or __import__("imageio_ffmpeg").get_ffmpeg_exe(),
        "-y",
        "-loglevel",
        "error",
        "-framerate",
        str(float(args.fps)),
        "-i",
        str(frame_dir / "frame_%03d.png"),
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "18",
        str(mp4_path),
    ]
    subprocess.run(command, check=True)
    for frame_path in frame_dir.iterdir():
        frame_path.unlink()
    frame_dir.rmdir()

    caption_path = output / "CAPTION.md"
    caption_path.write_text(
        "**Virtual tissue dynamics.** The growth-on baseline is propagated continuously "
        "from t=0 to t=4. Arrows show the strongest directed cell-cell interactions "
        "selected by the frozen learned edge predictor at each generated state; color "
        "indicates first-layer attention strength.\n"
    )
    provenance_path = output / "PROVENANCE.md"
    provenance_path.write_text(
        "# Provenance\n\n"
        f"- Baseline trajectory: `{trajectory_path}`\n"
        f"- Interaction extraction: `{extraction}`\n"
        f"- Original paper simulation commit: `{EXPECTED_RELEASE_COMMIT}`\n"
        "- Growth-on split SDE: sigma=0.03, dt=0.005, split interval=0.05.\n"
        "- Interaction overlay: strongest 260 directed accepted edges per frame, "
        "colored by absolute mean first-layer attention.\n"
        "- The learned interaction network is evaluated on each stored generated state; "
        "the overlay is not a schematic drawing.\n"
    )
    final_manifest = output / "figure_manifest.json"
    outputs = [gif_path, mp4_path, preview_path, caption_path, provenance_path]
    write_json(
        final_manifest,
        {
            "status": "complete",
            "analysis": "zebrafish_baseline_virtual_tissue_dynamics",
            "original_simulation_commit": EXPECTED_RELEASE_COMMIT,
            "trajectory": record(trajectory_path),
            "extraction_manifest": record(manifest_path) if manifest_path.is_file() else None,
            "interaction_arrays": record(extraction / "learned_interactions.npz"),
            "outputs": [record(path) for path in outputs],
        },
    )
    (output / "figure_manifest.json.sha256").write_text(
        f"{sha256(final_manifest)}  {final_manifest.name}\n"
    )
    print(json.dumps({"status": "complete", "output": str(output)}))
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    sub = root.add_subparsers(dest="command", required=True)
    extract = sub.add_parser("extract")
    extract.add_argument("--release", required=True, type=Path)
    extract.add_argument("--trajectory", required=True, type=Path)
    extract.add_argument("--h5ad", required=True, type=Path)
    extract.add_argument("--model-dir", required=True, type=Path)
    extract.add_argument("--device", default="cuda:0")
    extract.add_argument("--edge-batch-size", type=int, default=131_072)
    extract.add_argument("--output", required=True, type=Path)
    extract.add_argument("--verify-paper-inputs", action="store_true",
                         help="Check exact file identity against the original paper run")
    extract.set_defaults(func=extract_edges)
    render_parser = sub.add_parser("render")
    render_parser.add_argument("--trajectory", required=True, type=Path)
    render_parser.add_argument("--extraction", required=True, type=Path)
    render_parser.add_argument("--output", required=True, type=Path)
    render_parser.add_argument("--fps", type=float, default=6.0)
    render_parser.add_argument("--verify-paper-inputs", action="store_true")
    render_parser.set_defaults(func=render)
    return root


def main() -> int:
    args = parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
