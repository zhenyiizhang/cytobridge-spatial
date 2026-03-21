from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GeneProgramPanelResult:
    mean_pdf: Path
    zscore_pdf: Path


def collect_top_variable_heatmaps(
    *,
    figures_dir: str | Path,
    out_dir: str | Path,
    mean_name: str = "yolk_syncytial_layer_top250_variable_heatmap_mean.pdf",
    zscore_name: str = "yolk_syncytial_layer_top250_variable_heatmap_zscore.pdf",
    overwrite: bool = True,
) -> GeneProgramPanelResult:
    figures_dir = Path(figures_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mean_src = figures_dir / mean_name
    z_src = figures_dir / zscore_name
    if not mean_src.exists():
        raise FileNotFoundError(mean_src)
    if not z_src.exists():
        raise FileNotFoundError(z_src)
    mean_out = out_dir / mean_name
    z_out = out_dir / zscore_name
    if overwrite or not mean_out.exists():
        shutil.copy2(mean_src, mean_out)
    if overwrite or not z_out.exists():
        shutil.copy2(z_src, z_out)
    return GeneProgramPanelResult(mean_pdf=mean_out, zscore_pdf=z_out)
