from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from PIL import Image, ImageChops


@dataclass(frozen=True)
class AblationPanelSeriesResult:
    output_dir: Path
    files: list[Path]


def trim_white(img: Image.Image) -> Image.Image:
    bg = Image.new("RGB", img.size, "white")
    diff = ImageChops.difference(img.convert("RGB"), bg)
    bbox = diff.getbbox()
    if bbox is None:
        return img
    x0, y0, x1, y1 = bbox
    return img.crop((max(0, x0 - 2), max(0, y0 - 2), min(img.width, x1 + 2), min(img.height, y1 + 2)))


def crop_ablation_panel(frame_path: str | Path, side: str) -> Image.Image:
    frame_path = Path(frame_path)
    img = Image.open(frame_path).convert("RGB")
    w, h = img.size
    if side == "baseline":
        box = (int(0.065 * w), int(0.19 * h), int(0.455 * w), int(0.945 * h))
    elif side == "ablation":
        box = (int(0.545 * w), int(0.19 * h), int(0.935 * w), int(0.945 * h))
    else:
        raise ValueError(f"Unsupported side '{side}'")
    return trim_white(img.crop(box))


def export_ablation_panel_series(
    *,
    baseline_frames_dir: str | Path,
    comparison_frame_dirs: Mapping[str, str | Path],
    frame_ids: Mapping[str, str],
    out_dir: str | Path,
) -> AblationPanelSeriesResult:
    baseline_frames_dir = Path(baseline_frames_dir)
    comparison_frame_dirs = {k: Path(v) for k, v in comparison_frame_dirs.items()}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files: list[Path] = []
    for label, fid in frame_ids.items():
        baseline_png = out_dir / f"baseline__{label}.png"
        crop_ablation_panel(baseline_frames_dir / f"frame_{fid}.png", "baseline").save(baseline_png)
        files.append(baseline_png)
        for variant_name, variant_dir in comparison_frame_dirs.items():
            variant_png = out_dir / f"{variant_name}__{label}.png"
            crop_ablation_panel(variant_dir / f"frame_{fid}.png", "ablation").save(variant_png)
            files.append(variant_png)
    return AblationPanelSeriesResult(output_dir=out_dir, files=files)
