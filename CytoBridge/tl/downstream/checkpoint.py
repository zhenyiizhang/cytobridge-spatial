"""Checkpoint loading utilities for downstream analysis.

This module exposes two stable entrypoints:
- ``load_dynamical_model_from_dir`` for current CytoBridge checkpoints
- ``load_legacy_dynamical_model_from_dir`` for legacy ST-1104 checkpoints
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import torch
import torch.nn as nn
import yaml

from CytoBridge.tl.core.models import DynamicalModel
from .legacy_models import LegacyDynamicalModel


@dataclass(frozen=True)
class LoadedModel:
    model: nn.Module
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


def _build_legacy_model_config(
    legacy_cfg: dict,
    *,
    model_dir: Path,
    edge_predictor_root: Optional[str | Path] = None,
) -> dict:
    model_cfg = legacy_cfg.get("model", {})
    if not model_cfg:
        raise KeyError("Legacy params.yml is missing the required 'model' section.")
    latent_dim = int(legacy_cfg.get("data", {}).get("dim", model_cfg.get("in_out_dim", 0)))
    if latent_dim <= 0:
        raise ValueError("Could not infer latent dimension from legacy params.yml.")

    edge_predictor_path = model_cfg.get("edge_predictor_path")
    if edge_predictor_path:
        edge_predictor_path = Path(str(edge_predictor_path))
        if not edge_predictor_path.is_absolute():
            root = Path(edge_predictor_root) if edge_predictor_root is not None else model_dir.parent.parent / "edge_classifier"
            edge_predictor_path = root / edge_predictor_path.name

    return {
        "components": ["velocity", "growth", "score", "interaction"],
        "interaction_type": "gnn",
        "interaction_group_size": 1024,
        "velocity_net": {
            "hidden_dim": int(model_cfg["hidden_dim"]),
            "n_layers": int(model_cfg["n_hiddens"]),
            "residual": False,
            "activation": str(model_cfg["activation"]),
            "use_spatial": bool(model_cfg.get("use_spatial", True)),
        },
        "growth_net": {
            "hidden_dim": int(model_cfg["hidden_dim"]),
            "n_layers": 3,
            "residual": False,
            "activation": str(model_cfg["activation"]),
        },
        "score_net": {
            "hidden_dim": int(model_cfg["score_hidden_dim"]),
            "n_layers": 3,
            "activation": str(model_cfg["activation"]),
        },
        "interaction_net": {
            "hidden_dim": int(model_cfg["hidden_dim"]),
            "num_heads": 8,
            "num_layers": 1,
            "activation": str(model_cfg["activation"]),
            "num_rbf": 8,
            "cutoff": float(model_cfg["thre"]),
            "use_spatial": bool(model_cfg.get("use_spatial", True)),
            "edge_predictor_path": str(edge_predictor_path) if edge_predictor_path is not None else None,
            "edge_predictor_thre": float(model_cfg.get("edge_predictor_thre", 0.45)),
        },
    }


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


def load_legacy_dynamical_model_from_dir(
    model_dir: str | Path,
    *,
    device: str | torch.device = "cpu",
    edge_predictor_root: Optional[str | Path] = None,
    params_name: str = "params.yml",
    model_name: str = "model_final",
    score_name: str = "score_model",
) -> LoadedModel:
    """Load a legacy ST-1104 result directory with the canonical legacy architecture.

    Expected legacy layout:
    - ``params.yml``
    - ``model_final``
    - ``score_model``
    """
    model_dir = Path(model_dir)
    device = _coerce_device(device)

    params_path = model_dir / params_name
    model_path = model_dir / model_name
    score_path = model_dir / score_name
    if not params_path.exists():
        raise FileNotFoundError(f"Legacy params file not found: {params_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Legacy model checkpoint not found: {model_path}")
    if not score_path.exists():
        raise FileNotFoundError(f"Legacy score checkpoint not found: {score_path}")

    legacy_cfg = _load_yaml(params_path)
    latent_dim = int(legacy_cfg.get("data", {}).get("dim", 0))
    if latent_dim <= 0:
        raise ValueError("Legacy params.yml is missing a valid data.dim.")

    edge_root = Path(edge_predictor_root) if edge_predictor_root is not None else model_dir.parent.parent / "edge_classifier"
    model_cfg = _build_legacy_model_config(
        legacy_cfg,
        model_dir=model_dir,
        edge_predictor_root=edge_root,
    )
    model = LegacyDynamicalModel(
        legacy_cfg,
        edge_predictor_root=str(edge_root),
    ).to(device)

    old_state = torch.load(str(model_path), map_location=device)
    old_score_state = torch.load(str(score_path), map_location=device)

    model.f_net.load_state_dict(old_state, strict=True)
    model.score_model.load_state_dict(old_score_state, strict=True)
    model.eval()
    return LoadedModel(
        model=model,
        config={"legacy": legacy_cfg, "model": model_cfg},
        weight_stage=f"legacy:{model_name}",
        score_stage=f"legacy:{score_name}",
    )
