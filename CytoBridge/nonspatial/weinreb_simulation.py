"""Identity-preserving non-spatial simulation for clone-fate evaluation."""

from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

__all__ = ["simulate_sde_from_x0"]


def _freeze_model_for_inference(
    model,
) -> Optional[list[tuple[torch.nn.Parameter, bool]]]:
    if not isinstance(model, nn.Module):
        return None
    state = []
    for parameter in model.parameters():
        state.append((parameter, bool(parameter.requires_grad)))
        parameter.requires_grad_(False)
    return state


def _restore_model_after_inference(
    state: Optional[list[tuple[torch.nn.Parameter, bool]]],
) -> None:
    if state is None:
        return
    for parameter, requires_grad in state:
        parameter.requires_grad_(requires_grad)


def _generator(
    reference: torch.Tensor,
    seed: int | None,
    *,
    label: str,
) -> torch.Generator | None:
    if seed is None:
        return None
    if isinstance(seed, (bool, np.bool_)):
        raise ValueError(f"{label} must be an integer or None.")
    try:
        value = int(seed)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer or None.") from exc
    if value != seed or value < 0:
        raise ValueError(f"{label} must be a non-negative integer or None.")
    generator = torch.Generator(device=reference.device)
    generator.manual_seed(value)
    return generator


def _euler_sdeint_fixed_population(
    sde: nn.Module,
    initial_state: tuple[torch.Tensor, torch.Tensor],
    *,
    dt: float,
    times: Sequence[float],
    noise_generator: torch.Generator | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Euler–Maruyama integration that lands exactly on every output time."""

    z, lnw = initial_state
    current_state = (z.detach(), lnw.detach())
    current_time = float(times[0])
    outputs = [current_state]
    for target_time in times[1:]:
        target_time = float(target_time)
        while current_time < target_time - 1e-8:
            step_dt = min(float(dt), target_time - current_time)
            time_tensor = torch.tensor([current_time], device=z.device, dtype=z.dtype)
            dz, dlnw = sde.f(time_tensor, current_state)
            diffusion = sde.g(time_tensor, current_state[0])
            noise = torch.randn(
                current_state[0].shape,
                dtype=current_state[0].dtype,
                device=current_state[0].device,
                generator=noise_generator,
            ) * math.sqrt(step_dt)
            current_state = (
                (current_state[0] + dz * step_dt + diffusion * noise).detach(),
                (current_state[1] + dlnw * step_dt).detach(),
            )
            current_time += step_dt
        outputs.append(current_state)
    return (
        torch.stack([state[0] for state in outputs], dim=0),
        torch.stack([state[1] for state in outputs], dim=0),
    )


def simulate_sde_from_x0(
    *,
    x0,
    model,
    ts_points: Sequence[float],
    dt: float = 0.1,
    sigma: float = 0.0,
    include_score: bool = False,
    include_interaction: bool = True,
    interaction_m: int = 512,
    device: str = "cuda",
    noise_seed: int | None = None,
    interaction_seed: int | None = None,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate explicit initial states without sampling, splitting, or restarts.

    Row ``i`` at every returned time remains row ``i`` of ``x0``. Diffusion and
    interaction grouping use separate optional device-local random streams, so
    paired Full/No-interaction evaluations can share Brownian noise without a
    Full-only interaction call shifting that stream.
    """

    from CytoBridge.tl.core.interaction import cal_interaction

    if model is None or not hasattr(model, "predict_velocity"):
        raise TypeError("model must provide the current DynamicalModel API.")
    x0_array = (
        x0.detach().to(device="cpu", dtype=torch.float32).numpy()
        if isinstance(x0, torch.Tensor)
        else np.asarray(x0)
    )
    if x0_array.ndim != 2 or min(x0_array.shape) < 1:
        raise ValueError("x0 must be a non-empty (particles, features) matrix.")
    if not np.issubdtype(x0_array.dtype, np.number) or np.iscomplexobj(x0_array):
        raise TypeError("x0 must contain real numeric values.")
    if not np.isfinite(x0_array).all():
        raise ValueError("x0 must contain only finite values.")
    if not np.isfinite(float(dt)) or float(dt) <= 0:
        raise ValueError("dt must be finite and > 0.")
    if not np.isfinite(float(sigma)) or float(sigma) < 0:
        raise ValueError("sigma must be finite and >= 0.")
    if int(interaction_m) < 2:
        raise ValueError("interaction_m must be >= 2.")
    try:
        times = [float(value) for value in ts_points]
    except TypeError as exc:
        raise TypeError("ts_points must be an iterable of numeric times.") from exc
    if not times or not np.isfinite(times).all():
        raise ValueError("ts_points must contain finite values.")
    if any(right <= left for left, right in zip(times[:-1], times[1:])):
        raise ValueError("ts_points must be strictly increasing.")

    initial_points = torch.as_tensor(
        x0_array, dtype=torch.float32, device=device
    ).detach()
    n_particles = int(initial_points.shape[0])
    initial_log_weights = torch.full(
        (n_particles, 1),
        -float(np.log(n_particles)),
        dtype=initial_points.dtype,
        device=initial_points.device,
    )
    noise_generator = _generator(initial_points, noise_seed, label="noise_seed")
    interaction_generator = _generator(
        initial_points, interaction_seed, label="interaction_seed"
    )
    components = set(getattr(model, "components", ()))
    if "velocity" not in components:
        raise ValueError("Model is missing the velocity component.")
    interaction_net = getattr(model, "interaction_net", None)
    use_mass = bool(getattr(model, "use_growth_in_ode_inter", True))
    grouping_size = min(int(interaction_m), n_particles)

    class IdentityPreservingSDE(nn.Module):
        noise_type = "diagonal"
        sde_type = "ito"

        def f(self, time, state):
            points, log_weights = state
            expanded_time = time.expand(points.shape[0], 1).to(
                device=points.device, dtype=points.dtype
            )
            with torch.no_grad():
                drift = model.predict_velocity(t=expanded_time, x=points).detach()
                growth = (
                    model.predict_growth(t=expanded_time, x=points).detach()
                    if "growth" in components
                    else torch.zeros_like(log_weights)
                )

            interaction = torch.zeros_like(points)
            if (
                include_interaction
                and "interaction" in components
                and interaction_net is not None
                and n_particles > 1
            ):
                with torch.enable_grad():
                    interaction = cal_interaction(
                        z=points.detach(),
                        lnw=log_weights.detach(),
                        interaction_potential=interaction_net,
                        m=grouping_size,
                        cutoff=float(getattr(interaction_net, "cutoff", 1000.0)),
                        use_mass=use_mass,
                        t=time.detach(),
                        generator=interaction_generator,
                    ).detach()

            score = torch.zeros_like(points)
            if include_score and "score" in components:
                with torch.enable_grad():
                    score_points = points.detach().requires_grad_(True)
                    _, score = model.compute_score(
                        t=expanded_time.detach(),
                        x=score_points,
                        create_graph=False,
                    )
                    score = score.detach()
            return (drift + interaction + score).detach(), growth.detach()

        def g(self, _time, points):
            return torch.ones_like(points) * float(sigma)

    if verbose:
        print(
            "[simulate_sde_from_x0] start | "
            f"n={n_particles}, dim={initial_points.shape[1]}, "
            f"times={times}, dt={dt}, sigma={sigma}"
        )
    parameter_state = _freeze_model_for_inference(model)
    module_modes = (
        [(module, bool(module.training)) for module in model.modules()]
        if isinstance(model, nn.Module)
        else None
    )
    model_mode = bool(getattr(model, "training", False))
    try:
        if hasattr(model, "eval"):
            model.eval()
        points_t, log_weights_t = _euler_sdeint_fixed_population(
            IdentityPreservingSDE(),
            (initial_points, initial_log_weights),
            dt=float(dt),
            times=times,
            noise_generator=noise_generator,
        )
        weights_t = torch.exp(log_weights_t)
        normalized_t = torch.exp(
            log_weights_t - torch.logsumexp(log_weights_t, dim=1, keepdim=True)
        )
        result = tuple(
            tensor.detach().cpu().numpy()
            for tensor in (points_t, weights_t, normalized_t)
        )
    finally:
        _restore_model_after_inference(parameter_state)
        if module_modes is not None:
            for module, was_training in module_modes:
                module.training = was_training
        elif hasattr(model, "train"):
            model.train(model_mode)
    if verbose:
        print(
            f"[simulate_sde_from_x0] done | points_shape={result[0].shape}, "
            f"weights_shape={result[1].shape}"
        )
    return result
