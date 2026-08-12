"""Checkpoint loading utilities for downstream analysis.

This module exposes two stable entrypoints:
- ``load_dynamical_model_from_dir`` for current CytoBridge checkpoints
- ``load_legacy_dynamical_model_from_dir`` for legacy ST-1104 checkpoints
"""

from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
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
    weight_path: Path
    score_path: Optional[Path]


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


def _resolve_stage_checkpoint(
    model_dir: Path,
    stage: str,
    cfg: Optional[dict] = None,
) -> Optional[Path]:
    """Resolve the configured checkpoint, with legacy filename fallback."""
    stage_dir = model_dir / str(stage)
    configured_strategy = None
    if cfg is not None:
        for stage_config in cfg.get("training", {}).get("plan", []):
            if str(stage_config.get("name")) == str(stage):
                configured_strategy = str(
                    stage_config.get("save_strategy", "best")
                ).lower()
                break
    preferred_filename = {
        "best": "best_model.pth",
        "last": "last_model.pth",
    }.get(configured_strategy)
    filenames = (
        (preferred_filename,) if preferred_filename is not None else ()
    ) + tuple(
        name
        for name in ("last_model.pth", "best_model.pth")
        if name != preferred_filename
    )
    for filename in filenames:
        candidate = stage_dir / filename
        if candidate.exists():
            return candidate
    return None


def _score_stage_candidates(
    cfg: dict,
    explicit_preference: Optional[Sequence[str]],
) -> list[str]:
    """Return score-checkpoint stages in downstream loading priority order."""
    if explicit_preference is not None:
        return list(dict.fromkeys(str(stage) for stage in explicit_preference))

    configured: list[str] = []
    for stage in cfg.get("training", {}).get("plan", []):
        name = stage.get("name")
        mode = str(stage.get("mode", "")).lower()
        strategy = str(stage.get("train_strategy", "")).lower()
        if name and (mode == "score_matching" or strategy == "s"):
            configured.append(str(name))

    # Later configured score stages supersede earlier ones. Conventional names
    # remain fallbacks for older configs that did not record stage modes.
    candidates = [
        *reversed(configured),
        "Train_Score_Final",
        "Score_Refine",
        "Train_Score",
    ]
    return list(dict.fromkeys(candidates))


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
    score_stage_prefer: Optional[Sequence[str]] = None,
    edge_predictor_path: Optional[str | Path] = None,
) -> LoadedModel:
    """Load a trained DynamicalModel (+ optional score net) from a results directory.

    Parameters
    ----------
    model_dir
        Training output directory containing ``config.yaml`` and stage subfolders.
        Both ``last_model.pth`` and ``best_model.pth`` stage checkpoints are
        supported.
    dim
        Feature dimension of the aligned latent space (x1..x_dim).
    device
        Torch device.
    stage
        Preferred stage folder to load (default: ``Finetune``). The stage's
        configured ``save_strategy`` selects ``best_model.pth`` or
        ``last_model.pth`` when both are present.
    score_stage_prefer
        Optional ordered list of stage folders to look for ``score_model.pth``.
        By default, score-matching stages are inferred from the training plan
        and searched in reverse execution order, so a final score-refinement
        stage supersedes the initial score fit.
    edge_predictor_path
        Optional replacement for a recorded edge-predictor path. Current
        CytoBridge checkpoints normally embed the predictor weights, so copied
        model directories remain loadable even when the original absolute path
        no longer exists.

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

    # ---- choose weight stage ----
    candidate_stages: list[str] = []
    preferred = str(stage)
    preferred_path = _resolve_stage_checkpoint(model_dir, preferred, cfg)
    if preferred_path is not None:
        weight_stage = preferred
        weight_path = preferred_path
    else:
        candidate_stages = list(_iter_stage_names_from_config(cfg))
        if not candidate_stages:
            # Fallback: scan subfolders
            candidate_stages = sorted([p.name for p in model_dir.iterdir() if p.is_dir()])
        weight_stage = None
        for st in reversed(candidate_stages):
            candidate_path = _resolve_stage_checkpoint(model_dir, st, cfg)
            if candidate_path is not None:
                weight_stage = st
                weight_path = candidate_path
                break
        if weight_stage is None:
            raise FileNotFoundError(
                "Could not find any stage checkpoint named 'last_model.pth' or "
                f"'best_model.pth' under model_dir: {model_dir}"
            )

    state_dict = _load_state_dict(weight_path, device=device)
    model_config = deepcopy(cfg["model"])
    interaction_config = model_config.get("interaction_net", {})
    model_config["interaction_net"] = interaction_config
    if edge_predictor_path is not None:
        interaction_config["edge_predictor_path"] = str(
            Path(edge_predictor_path).expanduser().resolve()
        )
    embedded_predictor = any(
        ".link_predictor." in str(key) for key in state_dict
    )
    if embedded_predictor:
        interaction_config["load_edge_predictor_from_path"] = False

    model = DynamicalModel(int(dim), model_config)
    model = model.to(device)
    if not embedded_predictor and edge_predictor_path is not None:
        # Older current-format checkpoints stored predictor weights only in the
        # separate predictor file. The constructor has just loaded that file;
        # merge only those parameters so every other checkpoint key remains
        # subject to strict validation.
        state_dict = dict(state_dict)
        state_dict.update(
            {
                key: value
                for key, value in model.state_dict().items()
                if ".link_predictor." in str(key)
            }
        )
    model.load_state_dict(state_dict, strict=True)

    # ---- optional score stage ----
    score_stage_used: Optional[str] = None
    score_path_used: Optional[Path] = None
    if hasattr(model, "score_net") and model.score_net is not None and "score" in getattr(model, "components", []):
        for st in _score_stage_candidates(cfg, score_stage_prefer):
            score_path = model_dir / str(st) / "score_model.pth"
            if score_path.exists():
                score_state = _load_state_dict(score_path, device=device)
                model.score_net.load_state_dict(score_state, strict=True)
                score_stage_used = str(st)
                score_path_used = score_path
                break

    model.eval()
    return LoadedModel(
        model=model,
        config=cfg,
        weight_stage=weight_stage,
        score_stage=score_stage_used,
        weight_path=weight_path,
        score_path=score_path_used,
    )


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
        weight_path=model_path,
        score_path=score_path,
    )
