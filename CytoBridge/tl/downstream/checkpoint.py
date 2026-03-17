"""Checkpoint loading utilities for downstream analysis.

This module is intentionally lightweight and **does not** change any training logic.
It provides helpers to reconstruct a :class:`~CytoBridge.tl.core.models.DynamicalModel`
from a training output directory (e.g. ``results/...``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import torch
import yaml

from CytoBridge.tl.core.models import DynamicalModel


@dataclass(frozen=True)
class LoadedModel:
    model: DynamicalModel
    config: dict
    weight_stage: str
    score_stage: Optional[str]


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _coerce_device(device: str | torch.device) -> torch.device:
    if isinstance(device, torch.device):
        return device
    return torch.device(device)


def _load_state_dict(path: Path, device: torch.device) -> dict:
    obj = torch.load(str(path), map_location=device)
    if isinstance(obj, dict) and "model_state_dict" in obj:
        return obj["model_state_dict"]
    if isinstance(obj, dict) and "state_dict" in obj:
        return obj["state_dict"]
    if isinstance(obj, dict):
        return obj
    raise TypeError(f"Unsupported checkpoint object type at {path}: {type(obj)}")


def _iter_stage_names_from_config(cfg: dict) -> Iterable[str]:
    plan = cfg.get("training", {}).get("plan", [])
    for stage in plan:
        name = stage.get("name")
        if name:
            yield str(name)


def load_dynamical_model_from_dir(
    model_dir: str | Path,
    dim: int,
    device: str | torch.device = "cpu",
    stage: str = "Finetune",
    score_stage_prefer: Sequence[str] = ("Train_Score_Final", "Train_Score"),
) -> LoadedModel:
    """Load a trained DynamicalModel (+ optional score net) from a results directory.

    Parameters
    ----------
    model_dir
        Training output directory containing ``config.yaml`` and stage subfolders.
    dim
        Feature dimension of the aligned latent space (x1..x_dim).
    device
        Torch device.
    stage
        Preferred stage folder to load ``last_model.pth`` from (default: ``Finetune``).
    score_stage_prefer
        Ordered list of stage folders to look for ``score_model.pth``.

    Returns
    -------
    LoadedModel
        Loaded model, config, and which stages were used.
    """
    model_dir = Path(model_dir)
    device = _coerce_device(device)

    cfg_path = model_dir / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"config.yaml not found under model_dir: {model_dir}")
    cfg = _load_yaml(cfg_path)
    if "model" not in cfg:
        raise KeyError("config.yaml missing required top-level key: 'model'")

    model = DynamicalModel(int(dim), cfg["model"])
    model = model.to(device)

    # ---- choose weight stage ----
    candidate_stages: list[str] = []
    preferred = str(stage)
    if (model_dir / preferred / "last_model.pth").exists():
        weight_stage = preferred
    else:
        candidate_stages = list(_iter_stage_names_from_config(cfg))
        if not candidate_stages:
            # Fallback: scan subfolders
            candidate_stages = sorted([p.name for p in model_dir.iterdir() if p.is_dir()])
        weight_stage = None
        for st in reversed(candidate_stages):
            if (model_dir / st / "last_model.pth").exists():
                weight_stage = st
                break
        if weight_stage is None:
            raise FileNotFoundError(
                f"Could not find any '*/*last_model.pth' under model_dir: {model_dir}"
            )

    weight_path = model_dir / weight_stage / "last_model.pth"
    state_dict = _load_state_dict(weight_path, device=device)
    model.load_state_dict(state_dict, strict=True)

    # ---- optional score stage ----
    score_stage_used: Optional[str] = None
    if hasattr(model, "score_net") and model.score_net is not None and "score" in getattr(model, "components", []):
        for st in score_stage_prefer:
            score_path = model_dir / str(st) / "score_model.pth"
            if score_path.exists():
                score_state = _load_state_dict(score_path, device=device)
                model.score_net.load_state_dict(score_state, strict=True)
                score_stage_used = str(st)
                break

    model.eval()
    return LoadedModel(model=model, config=cfg, weight_stage=weight_stage, score_stage=score_stage_used)

