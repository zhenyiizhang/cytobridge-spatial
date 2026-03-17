import json
import pathlib
import re
import warnings
from typing import Dict, Any, Optional
import scanpy as sc
import pandas as pd
import numpy as np
import torch
import yaml
from CytoBridge.utils.config import load_config
from CytoBridge.tl.core.models import DynamicalModel
from CytoBridge.tl.train.trainer import TrainingPipeline

_TIME_PATTERN = re.compile(r"-?\d+(?:\.\d+)?")


def _auto_time_mapping(raw_time) -> dict:
    unique_times = list(pd.unique(raw_time))
    series = pd.Series(unique_times)
    if pd.api.types.is_numeric_dtype(series):
        sorted_times = sorted(unique_times, key=float)
        return {time_point: i for i, time_point in enumerate(sorted_times)}

    parsed = []
    for idx, value in enumerate(unique_times):
        match = _TIME_PATTERN.search(str(value))
        parsed.append((float(match.group()) if match else None, idx, value))

    if all(item[0] is not None for item in parsed):
        parsed.sort(key=lambda x: (x[0], x[1]))
        sorted_times = [item[2] for item in parsed]
        return {time_point: i for i, time_point in enumerate(sorted_times)}

    # Fallback to observed order, avoids lexicographic artifacts such as 10hpf < 2hpf.
    return {time_point: i for i, time_point in enumerate(unique_times)}


def _to_dense_float32(x):
    if hasattr(x, "toarray"):
        return x.toarray().astype(np.float32)
    return np.asarray(x, dtype=np.float32)


def _freeze_model_params(model: torch.nn.Module) -> list[tuple[torch.nn.Parameter, bool]]:
    state: list[tuple[torch.nn.Parameter, bool]] = []
    for p in model.parameters():
        state.append((p, bool(p.requires_grad)))
        if p.requires_grad:
            p.requires_grad_(False)
    return state


def _restore_model_params(state: list[tuple[torch.nn.Parameter, bool]]) -> None:
    for p, requires_grad in state:
        p.requires_grad_(requires_grad)


def _store_vector_component(
    adata: sc.AnnData,
    *,
    name: str,
    values: np.ndarray,
    spatial_dim: int,
) -> None:
    adata.obsm[f"{name}_model"] = values
    if spatial_dim > 0:
        adata.obsm[f"{name}_spatial"] = values[:, :spatial_dim]
        adata.obsm[f"{name}_latent"] = values[:, spatial_dim:]
    else:
        adata.obsm[f"{name}_latent"] = values


def _collect_adata_fit_overrides(adata: sc.AnnData) -> Dict[str, Any]:
    """Collect optional dataset-specific runtime parameters from adata.uns."""
    merged: Dict[str, Any] = {}

    # Preferred namespaces for dataset-specific runtime settings.
    for key in ("fit_params", "cytobridge_fit", "training_params"):
        raw = adata.uns.get(key, None)
        if isinstance(raw, dict):
            for k, v in raw.items():
                merged.setdefault(k, v)

    # Backward-compatible fallback from interaction graph metadata.
    ig = adata.uns.get("interaction_graph", {})
    if isinstance(ig, dict):
        if ig.get("neighborhood_threshold", None) is not None:
            merged.setdefault("interaction_cutoff", ig["neighborhood_threshold"])
            merged.setdefault("cutoff", ig["neighborhood_threshold"])
        if ig.get("edge_predictor_threshold", None) is not None:
            merged.setdefault("edge_predictor_threshold", ig["edge_predictor_threshold"])
            merged.setdefault("edge_predictor_thre", ig["edge_predictor_threshold"])
        if ig.get("edge_predictor_path", None) is not None:
            merged.setdefault("edge_predictor_path", ig["edge_predictor_path"])
        # Backward-compatible aliases written by edge predictor preprocessing.
        if ig.get("edge_predictor_model_path", None) is not None:
            merged.setdefault("edge_predictor_path", ig["edge_predictor_model_path"])

    return merged


def _resolve_override(
    explicit_value: Any,
    adata_overrides: Dict[str, Any],
    keys: tuple[str, ...],
) -> tuple[Any, Optional[str]]:
    if explicit_value is not None:
        return explicit_value, "fit argument"
    for key in keys:
        if key in adata_overrides and adata_overrides[key] is not None:
            return adata_overrides[key], f"adata.uns override ({key})"
    return None, None


def _coerce_float(
    value: Any,
    *,
    name: str,
    lower: float | None = None,
    upper: float | None = None,
    lower_inclusive: bool = False,
    upper_inclusive: bool = False,
) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a float, got {value!r}.") from exc

    if not np.isfinite(out):
        raise ValueError(f"{name} must be finite, got {out!r}.")

    if lower is not None:
        ok = out >= lower if lower_inclusive else out > lower
        if not ok:
            bound = ">=" if lower_inclusive else ">"
            raise ValueError(f"{name} must be {bound} {lower}, got {out}.")
    if upper is not None:
        ok = out <= upper if upper_inclusive else out < upper
        if not ok:
            bound = "<=" if upper_inclusive else "<"
            raise ValueError(f"{name} must be {bound} {upper}, got {out}.")
    return out


def _first_present(
    values: Dict[str, Any],
    keys: tuple[str, ...],
) -> tuple[Any, Optional[str]]:
    for key in keys:
        if key in values and values[key] is not None:
            return values[key], key
    return None, None


def _normalize_path_text(value: Any) -> str:
    return str(pathlib.Path(str(value)).expanduser())


def _values_match(lhs: Any, rhs: Any, *, mode: str) -> bool:
    if mode == "float":
        try:
            return np.isclose(float(lhs), float(rhs), rtol=1e-6, atol=1e-12)
        except (TypeError, ValueError):
            return False
    if mode == "path":
        return _normalize_path_text(lhs) == _normalize_path_text(rhs)
    return str(lhs) == str(rhs)


def _warn_if_explicit_conflicts_with_adata(
    *,
    explicit_value: Any,
    adata_overrides: Dict[str, Any],
    adata_keys: tuple[str, ...],
    label: str,
    mode: str,
) -> None:
    if explicit_value is None:
        return
    adata_value, adata_key = _first_present(adata_overrides, adata_keys)
    if adata_key is None:
        return
    if _values_match(explicit_value, adata_value, mode=mode):
        return
    msg = (
        f"[fit][warning] {label} from fit argument ({explicit_value!r}) conflicts with "
        f"adata.uns override `{adata_key}` ({adata_value!r}). "
        "Using fit argument (priority: fit > adata.uns > config)."
    )
    print(msg)
    warnings.warn(msg, UserWarning, stacklevel=2)


def _apply_runtime_overrides(
    resolved_config: Dict[str, Any],
    adata: sc.AnnData,
    *,
    interaction_cutoff: float | None = None,
    edge_predictor_path: str | None = None,
    edge_predictor_threshold: float | None = None,
    ckpt_dir: str | pathlib.Path | None = None,
    sigma: float | None = None,
) -> Dict[str, Any]:
    """Apply runtime overrides with precedence: fit args > adata.uns > config."""
    used: Dict[str, Any] = {
        "interaction_cutoff": None,
        "edge_predictor_path": None,
        "edge_predictor_threshold": None,
        "ckpt_dir": None,
        "sigma": None,
    }

    adata_overrides = _collect_adata_fit_overrides(adata)
    model_cfg = resolved_config.get("model", {})
    if not isinstance(model_cfg, dict):
        return used

    components = model_cfg.get("components", [])
    has_interaction = "interaction" in components
    interaction_cfg = model_cfg.get("interaction_net")
    interaction_type = str(model_cfg.get("interaction_type", "potential")).lower()
    is_gnn_interaction = has_interaction and interaction_type == "gnn"

    # Explicit fit args always win, but warn if they conflict with adata.uns values.
    _warn_if_explicit_conflicts_with_adata(
        explicit_value=interaction_cutoff,
        adata_overrides=adata_overrides,
        adata_keys=("interaction_cutoff", "cutoff"),
        label="interaction_cutoff",
        mode="float",
    )
    _warn_if_explicit_conflicts_with_adata(
        explicit_value=edge_predictor_path,
        adata_overrides=adata_overrides,
        adata_keys=("edge_predictor_path",),
        label="edge_predictor_path",
        mode="path",
    )
    _warn_if_explicit_conflicts_with_adata(
        explicit_value=edge_predictor_threshold,
        adata_overrides=adata_overrides,
        adata_keys=("edge_predictor_threshold", "edge_predictor_thre"),
        label="edge_predictor_threshold",
        mode="float",
    )
    _warn_if_explicit_conflicts_with_adata(
        explicit_value=ckpt_dir,
        adata_overrides=adata_overrides,
        adata_keys=("ckpt_dir",),
        label="ckpt_dir",
        mode="path",
    )
    _warn_if_explicit_conflicts_with_adata(
        explicit_value=sigma,
        adata_overrides=adata_overrides,
        adata_keys=("sigma",),
        label="sigma",
        mode="float",
    )

    if (edge_predictor_path is not None or edge_predictor_threshold is not None) and not is_gnn_interaction:
        if not has_interaction:
            msg = (
                "[fit][warning] edge predictor arguments were provided, but model has no "
                "`interaction` component; edge predictor settings will be ignored."
            )
        else:
            msg = (
                "[fit][warning] edge predictor arguments were provided, but "
                f"interaction_type='{interaction_type}' (not 'gnn'); edge predictor settings "
                "will be ignored."
            )
        print(msg)
        warnings.warn(msg, UserWarning, stacklevel=2)

    # cutoff
    raw_cutoff, cutoff_source = _resolve_override(
        interaction_cutoff,
        adata_overrides,
        ("interaction_cutoff", "cutoff"),
    )
    if raw_cutoff is not None and has_interaction and isinstance(interaction_cfg, dict):
        cutoff = _coerce_float(raw_cutoff, name="interaction_cutoff", lower=0.0)
        old_cutoff = interaction_cfg.get("cutoff")
        interaction_cfg["cutoff"] = cutoff
        used["interaction_cutoff"] = cutoff
        print(f"[fit] interaction cutoff override ({cutoff_source}): {old_cutoff} -> {cutoff}")

    # edge predictor path
    raw_edge_path, edge_path_source = _resolve_override(
        edge_predictor_path,
        adata_overrides,
        ("edge_predictor_path",),
    )
    if raw_edge_path is not None and is_gnn_interaction and isinstance(interaction_cfg, dict):
        edge_path = str(pathlib.Path(str(raw_edge_path)).expanduser())
        old_edge_path = interaction_cfg.get("edge_predictor_path")
        interaction_cfg["edge_predictor_path"] = edge_path
        used["edge_predictor_path"] = edge_path
        print(f"[fit] edge predictor path override ({edge_path_source}): {old_edge_path} -> {edge_path}")

    # edge predictor threshold
    raw_edge_thre, edge_thre_source = _resolve_override(
        edge_predictor_threshold,
        adata_overrides,
        ("edge_predictor_threshold", "edge_predictor_thre"),
    )
    if raw_edge_thre is not None and is_gnn_interaction and isinstance(interaction_cfg, dict):
        edge_thre = _coerce_float(
            raw_edge_thre,
            name="edge_predictor_threshold",
            lower=0.0,
            upper=1.0,
        )
        old_edge_thre = interaction_cfg.get("edge_predictor_thre")
        interaction_cfg["edge_predictor_thre"] = edge_thre
        used["edge_predictor_threshold"] = edge_thre
        print(
            f"[fit] edge predictor threshold override ({edge_thre_source}): "
            f"{old_edge_thre} -> {edge_thre}"
        )

    # ckpt_dir
    raw_ckpt_dir, ckpt_source = _resolve_override(ckpt_dir, adata_overrides, ("ckpt_dir",))
    if raw_ckpt_dir is not None:
        out_ckpt = str(pathlib.Path(str(raw_ckpt_dir)).expanduser())
        old_ckpt = resolved_config.get("ckpt_dir")
        resolved_config["ckpt_dir"] = out_ckpt
        used["ckpt_dir"] = out_ckpt
        print(f"[fit] ckpt_dir override ({ckpt_source}): {old_ckpt} -> {out_ckpt}")

    # sigma: apply to defaults and every stage for a single-run canonical value.
    raw_sigma, sigma_source = _resolve_override(sigma, adata_overrides, ("sigma",))
    if raw_sigma is not None:
        sigma_val = _coerce_float(raw_sigma, name="sigma", lower=0.0)
        training_cfg = resolved_config.setdefault("training", {})
        if not isinstance(training_cfg, dict):
            raise ValueError("config['training'] must be a dict.")
        defaults_cfg = training_cfg.setdefault("defaults", {})
        if not isinstance(defaults_cfg, dict):
            raise ValueError("config['training']['defaults'] must be a dict.")
        old_sigma = defaults_cfg.get("sigma")
        defaults_cfg["sigma"] = sigma_val
        plan = training_cfg.get("plan", [])
        if isinstance(plan, list):
            for stage_cfg in plan:
                if isinstance(stage_cfg, dict):
                    stage_cfg["sigma"] = sigma_val
        used["sigma"] = sigma_val
        print(f"[fit] sigma override ({sigma_source}): {old_sigma} -> {sigma_val} (applied to defaults + plan)")

    return used


def _ensure_time_point_processed(adata: sc.AnnData, *, time_key: str) -> sc.AnnData:
    if "time_point_processed" in adata.obs:
        return adata
    if time_key not in adata.obs:
        raise KeyError(
            f"Missing time column in adata.obs: expected 'time_point_processed' or '{time_key}'."
        )
    raw_time = adata.obs[time_key].to_numpy()
    if pd.api.types.is_numeric_dtype(raw_time):
        adata.obs["time_point_processed"] = raw_time.astype(float)
    else:
        mapping = _auto_time_mapping(raw_time)
        adata.obs["time_point_processed"] = pd.Series(raw_time).map(mapping).to_numpy().astype(float)
    return adata


def _ensure_x_latent(adata: sc.AnnData, *, obsm_key: str = "X_latent") -> sc.AnnData:
    if "X_latent" in adata.obsm:
        adata.obsm["X_latent"] = _to_dense_float32(adata.obsm["X_latent"])
        return adata
    if obsm_key in adata.obsm:
        adata.obsm["X_latent"] = _to_dense_float32(adata.obsm[obsm_key])
    else:
        adata.obsm["X_latent"] = _to_dense_float32(adata.X)
    return adata


def _build_model_input(
    adata: sc.AnnData,
    *,
    is_spatial: bool,
    spatial_key: str = "spatial_aligned",
    latent_key: str = "X_latent",
    warn_on_missing_spatial: bool = True,
) -> tuple[np.ndarray, int]:
    """Build model input matrix according to data modality."""
    latent = _to_dense_float32(adata.obsm[latent_key])
    if not is_spatial:
        return latent, 0

    if spatial_key not in adata.obsm:
        if warn_on_missing_spatial:
            warnings.warn(
                f"is_spatial=True but `adata.obsm['{spatial_key}']` is missing; "
                f"falling back to `{latent_key}` only.",
                UserWarning,
                stacklevel=2,
            )
        return latent, 0

    spatial = _to_dense_float32(adata.obsm[spatial_key])
    if spatial.shape[0] != latent.shape[0]:
        raise ValueError(
            f"Row mismatch between `{spatial_key}` ({spatial.shape[0]}) and "
            f"`{latent_key}` ({latent.shape[0]})."
        )
    spatial_dim = int(spatial.shape[1])
    return np.hstack((spatial, latent)).astype(np.float32), spatial_dim


def _fit_adata(
    adata: sc.AnnData,
    config: Dict[str, Any] | str,
    batch_size: int | None = None,
    device: str = "cuda",
    *,
    model_input: np.ndarray,
    spatial_dim: int = 0,
    interaction_cutoff: float | None = None,
    edge_predictor_path: str | None = None,
    edge_predictor_threshold: float | None = None,
    ckpt_dir: str | pathlib.Path | None = None,
    sigma: float | None = None,
) -> sc.AnnData:
    """
    Internal training entrypoint: requires `adata.obs['time_point_processed']`
    and prepared model input matrix.
    """
    # ---------- 1. load & resolve config ----------
    resolved_config = load_config(config)  # allow str-path or dict
    used_overrides = _apply_runtime_overrides(
        resolved_config,
        adata,
        interaction_cutoff=interaction_cutoff,
        edge_predictor_path=edge_predictor_path,
        edge_predictor_threshold=edge_predictor_threshold,
        ckpt_dir=ckpt_dir,
        sigma=sigma,
    )
    used_interaction_cutoff = used_overrides.get("interaction_cutoff")
    used_edge_predictor_threshold = used_overrides.get("edge_predictor_threshold")
    used_edge_predictor_path = used_overrides.get("edge_predictor_path")
    used_ckpt_dir = used_overrides.get("ckpt_dir")
    used_sigma = used_overrides.get("sigma")
    device = torch.device(device)
    model_input = np.asarray(model_input, dtype=np.float32)
    if model_input.shape[0] != adata.n_obs:
        raise ValueError(
            f"model_input rows ({model_input.shape[0]}) must equal adata.n_obs ({adata.n_obs})."
        )

    # ---------- 2. data preparation ----------
    time_key = "time_point_processed"
    time_points = sorted(adata.obs[time_key].unique())
    time_values = adata.obs[time_key].to_numpy()
    data_torch = []
    for t in time_points:
        mask = (time_values == t)
        tens = torch.tensor(model_input[mask], dtype=torch.float32, device=device)
        data_torch.append(tens)

    # Keep forward time ordering; reverse pass is handled inside training if enabled.

    # ---------- 3. auto batch-size ----------
    if batch_size is None:
        config_batch = resolved_config.get("training", {}).get("defaults", {}).get("batch_size")
        if config_batch is not None:
            batch_size = int(config_batch)
        else:
            batch_size = min(min(x.shape[0] for x in data_torch), 256)

    # ---------- 4. build & train model ----------
    dim = data_torch[0].shape[1]
    model = DynamicalModel(dim, resolved_config["model"])
    trainer = TrainingPipeline(model, resolved_config, batch_size, device, data=data_torch)
    model = trainer.train(data_torch, time_points)

    # ---------- 5. compute model outputs (component-aware) ----------
    from CytoBridge.tl.core.interaction import cal_interaction

    all_times = torch.tensor(adata.obs[time_key].values, dtype=torch.float32, device=device).unsqueeze(1)
    all_data = torch.tensor(model_input, dtype=torch.float32, device=device)
    components = set(getattr(model, "components", []))
    n_obs = all_data.shape[0]
    output_dim = all_data.shape[1]
    use_mass = bool(getattr(model, "use_growth_in_ode_inter", True))
    interaction_group_size = int(getattr(model, "interaction_group_size", 1024))
    interaction_net = getattr(model, "interaction_net", None)
    interaction_cutoff_model = float(getattr(interaction_net, "cutoff", 1000.0))

    velocity_np = np.zeros((n_obs, output_dim), dtype=np.float32)
    interaction_np = np.zeros((n_obs, output_dim), dtype=np.float32)
    score_grad_np = np.zeros((n_obs, output_dim), dtype=np.float32)

    freeze_state = _freeze_model_params(model)
    try:
        if "velocity" in components:
            with torch.no_grad():
                velocity = model.predict_velocity(t=all_times, x=all_data)
            velocity_np = velocity.detach().cpu().numpy()
            _store_vector_component(
                adata,
                name="velocity",
                values=velocity_np,
                spatial_dim=spatial_dim,
            )

        if "growth" in components:
            with torch.no_grad():
                growth = model.predict_growth(t=all_times, x=all_data)
            adata.obsm["growth_rate"] = growth.detach().cpu().numpy()

        if "score" in components:
            with torch.no_grad():
                score_potential = model.predict_score(t=all_times, x=all_data)
            score_potential_np = score_potential.detach().cpu().numpy()
            # Keep compatibility: score_model historically stores score net output (scalar potential).
            adata.obsm["score_model"] = score_potential_np
            adata.obsm["score_potential"] = score_potential_np

            _, score_grad = model.compute_score(
                t=all_times,
                x=all_data.detach().requires_grad_(True),
                create_graph=False,
            )
            score_grad_np = score_grad.detach().cpu().numpy()
            _store_vector_component(
                adata,
                name="score_gradient",
                values=score_grad_np,
                spatial_dim=spatial_dim,
            )

        if "interaction" in components and interaction_net is not None:
            lnw = torch.log(torch.ones(n_obs, 1, device=device, dtype=torch.float32) / float(n_obs))
            if getattr(interaction_net, "requires_time", False):
                with torch.no_grad():
                    interaction = cal_interaction(
                        z=all_data.detach(),
                        lnw=lnw,
                        interaction_potential=interaction_net,
                        m=interaction_group_size,
                        cutoff=interaction_cutoff_model,
                        use_mass=use_mass,
                        t=all_times,
                    ).float()
            else:
                interaction = cal_interaction(
                    z=all_data.detach(),
                    lnw=lnw,
                    interaction_potential=interaction_net,
                    m=interaction_group_size,
                    cutoff=interaction_cutoff_model,
                    use_mass=use_mass,
                    t=all_times,
                ).float()
            interaction_np = interaction.detach().cpu().numpy()
            _store_vector_component(
                adata,
                name="interaction",
                values=interaction_np,
                spatial_dim=spatial_dim,
            )

        if any(name in components for name in ("velocity", "interaction", "score")):
            full_drift_np = velocity_np + interaction_np + score_grad_np
            _store_vector_component(
                adata,
                name="full_drift",
                values=full_drift_np,
                spatial_dim=spatial_dim,
            )
    finally:
        _restore_model_params(freeze_state)

    # ---------- 6. store model internals ----------
    adata.uns["all_model"] = {
        "model_config": resolved_config["model"],
        "model_input_dim": int(model_input.shape[1]),
        "spatial_dim": int(spatial_dim),
        "interaction_cutoff": (
            float(used_interaction_cutoff)
            if used_interaction_cutoff is not None
            else float(getattr(getattr(model, "interaction_net", None), "cutoff", np.nan))
        ),
        "edge_predictor_threshold": (
            float(used_edge_predictor_threshold)
            if used_edge_predictor_threshold is not None
            else float(getattr(getattr(model, "interaction_net", None), "edge_predictor_thre", np.nan))
        ),
        "edge_predictor_path": (
            str(used_edge_predictor_path)
            if used_edge_predictor_path is not None
            else (
                resolved_config.get("model", {})
                .get("interaction_net", {})
                .get("edge_predictor_path")
            )
        ),
        "ckpt_dir": (
            str(used_ckpt_dir)
            if used_ckpt_dir is not None
            else str(resolved_config.get("ckpt_dir", ""))
        ),
        "sigma": (
            float(used_sigma)
            if used_sigma is not None
            else float(resolved_config["training"]["defaults"].get("sigma", np.nan))
        ),
        "training_config": {
            "defaults": resolved_config["training"]["defaults"],
            "plan": json.dumps(resolved_config["training"]["plan"]),
        },
        "model_state_dict": {k: v.cpu().numpy() for k, v in model.state_dict().items()},
    }

    # ---------- 7. save ----------
    if not resolved_config.get("ckpt_dir"):
        raise KeyError(
            "Missing `ckpt_dir` in training config. "
            "Set `ckpt_dir` in config, pass `ckpt_dir=` to `fit(...)`, "
            "or put it in `adata.uns['fit_params']['ckpt_dir']`."
        )
    ckpt_dir = pathlib.Path(resolved_config["ckpt_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    h5ad_path = ckpt_dir / "adata.h5ad"
    yaml_path = ckpt_dir / "config.yaml"

    adata.write_h5ad(h5ad_path)
    with yaml_path.open("w", encoding="utf-8") as yf:
        yaml.dump(resolved_config, yf, default_flow_style=False, allow_unicode=True)

    print(f"Model & data saved -> {ckpt_dir}")

    trainer.evaluate(adata, data_torch, time_points)  # TODO handle hold-out

    return adata

def fit(
    adata: sc.AnnData | str,
    config: Dict[str, Any] | str,
    batch_size: int | None = None,
    device: str = "cuda",
    *,
    time_key: str = "time_point_processed",
    obsm_key: str = "X_latent",
    samples_key: str = "samples",
    is_spatial: bool = True,
    spatial_key: str = "spatial_aligned",
    interaction_cutoff: float | None = None,
    edge_predictor_path: str | None = None,
    edge_predictor_threshold: float | None = None,
    ckpt_dir: str | pathlib.Path | None = None,
    sigma: float | None = None,
) -> sc.AnnData:
    """
    Main training entrypoint (prefer this single API).

    - If `adata` is an AnnData: requires/creates `obs['time_point_processed']` and `obsm['X_latent']`.
      When `is_spatial=True`, model input is built as `[obsm[spatial_key], obsm['X_latent']]`.
      When `is_spatial=False`, model input is `obsm['X_latent']` only.
    - If `adata` is a path:
      - `.h5ad`: loads AnnData then ensures keys as above (using `time_key` / `obsm_key`).
      - `.csv`: loads a table with `samples_key` + features and wraps into AnnData.

    Runtime override priority for selected training parameters:
    `fit(...) arguments` > `adata.uns['fit_params'/'cytobridge_fit'/'training_params']` > `config`.
    Supported overrides: `interaction_cutoff`, `edge_predictor_path`,
    `edge_predictor_threshold`, `ckpt_dir`, `sigma`.
    """
    if isinstance(adata, str):
        path = pathlib.Path(adata)
        if not path.exists():
            raise FileNotFoundError(path)
        if path.suffix.lower() == ".h5ad":
            adata_obj = sc.read_h5ad(str(path))
            adata_obj = _ensure_time_point_processed(adata_obj, time_key=time_key)
            adata_obj = _ensure_x_latent(adata_obj, obsm_key=obsm_key)
            model_input, spatial_dim = _build_model_input(
                adata_obj,
                is_spatial=is_spatial,
                spatial_key=spatial_key,
            )
            return _fit_adata(
                adata_obj,
                config=config,
                batch_size=batch_size,
                device=device,
                model_input=model_input,
                spatial_dim=spatial_dim,
                interaction_cutoff=interaction_cutoff,
                edge_predictor_path=edge_predictor_path,
                edge_predictor_threshold=edge_predictor_threshold,
                ckpt_dir=ckpt_dir,
                sigma=sigma,
            )
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(str(path))
            if samples_key not in df.columns:
                raise KeyError(f"{samples_key} not found in {path}")
            samples = df[samples_key].to_numpy()
            X = df.drop(columns=[samples_key]).values
            obs = pd.DataFrame({samples_key: samples})
            adata_obj = sc.AnnData(X=X, obs=obs)
            adata_obj = _ensure_time_point_processed(adata_obj, time_key=samples_key)
            adata_obj.obsm["X_latent"] = _to_dense_float32(X)
            model_input, spatial_dim = _build_model_input(
                adata_obj,
                is_spatial=is_spatial,
                spatial_key=spatial_key,
                warn_on_missing_spatial=False,
            )
            return _fit_adata(
                adata_obj,
                config=config,
                batch_size=batch_size,
                device=device,
                model_input=model_input,
                spatial_dim=spatial_dim,
                interaction_cutoff=interaction_cutoff,
                edge_predictor_path=edge_predictor_path,
                edge_predictor_threshold=edge_predictor_threshold,
                ckpt_dir=ckpt_dir,
                sigma=sigma,
            )
        raise ValueError(f"Unsupported input path: {path} (expected .h5ad or .csv)")

    adata_obj = _ensure_time_point_processed(adata, time_key=time_key)
    adata_obj = _ensure_x_latent(adata_obj, obsm_key=obsm_key)
    model_input, spatial_dim = _build_model_input(
        adata_obj,
        is_spatial=is_spatial,
        spatial_key=spatial_key,
    )
    return _fit_adata(
        adata_obj,
        config=config,
        batch_size=batch_size,
        device=device,
        model_input=model_input,
        spatial_dim=spatial_dim,
        interaction_cutoff=interaction_cutoff,
        edge_predictor_path=edge_predictor_path,
        edge_predictor_threshold=edge_predictor_threshold,
        ckpt_dir=ckpt_dir,
        sigma=sigma,
    )


def fit_spatial_csv(
    data_csv: str,
    config: Dict[str, Any] | str,
    samples_key: str = "samples",
    device: str = "cuda",
    batch_size: int | None = None,
    interaction_cutoff: float | None = None,
    edge_predictor_path: str | None = None,
    edge_predictor_threshold: float | None = None,
    ckpt_dir: str | pathlib.Path | None = None,
    sigma: float | None = None,
) -> sc.AnnData:
    warnings.warn(
        "`fit_spatial_csv` is deprecated and will be removed in a future release; "
        "use `fit(data_csv_or_adata, ...)` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return fit(
        data_csv,
        config=config,
        batch_size=batch_size,
        device=device,
        time_key=samples_key,
        samples_key=samples_key,
        is_spatial=True,
        interaction_cutoff=interaction_cutoff,
        edge_predictor_path=edge_predictor_path,
        edge_predictor_threshold=edge_predictor_threshold,
        ckpt_dir=ckpt_dir,
        sigma=sigma,
    )


def fit_spatial_h5ad(
    aligned_h5ad: str,
    config: Dict[str, Any] | str,
    time_key: str = "time_point_processed",
    obsm_key: str = "X_latent",
    device: str = "cuda",
    batch_size: int | None = None,
    interaction_cutoff: float | None = None,
    edge_predictor_path: str | None = None,
    edge_predictor_threshold: float | None = None,
    ckpt_dir: str | pathlib.Path | None = None,
    sigma: float | None = None,
) -> sc.AnnData:
    warnings.warn(
        "`fit_spatial_h5ad` is deprecated and will be removed in a future release; "
        "use `fit(h5ad_path_or_adata, ...)` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return fit(
        aligned_h5ad,
        config=config,
        batch_size=batch_size,
        device=device,
        time_key=time_key,
        obsm_key=obsm_key,
        is_spatial=True,
        interaction_cutoff=interaction_cutoff,
        edge_predictor_path=edge_predictor_path,
        edge_predictor_threshold=edge_predictor_threshold,
        ckpt_dir=ckpt_dir,
        sigma=sigma,
    )
