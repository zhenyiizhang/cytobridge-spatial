"""Fixed-population S3 integrator retained from the accepted attraction evaluation.

Source: cytobridge-spatial-spatial-attraction-benchmark-20260719,
CytoBridge/tl/downstream/simulation.py. Mass mode and RNG semantics retained.
"""
from __future__ import annotations
from typing import Optional, Sequence
import numpy as np
import torch
from torch import nn



def _freeze_model_for_inference(model) -> Optional[list[tuple[torch.nn.Parameter, bool]]]:
    if not isinstance(model, nn.Module):
        return None
    state: list[tuple[torch.nn.Parameter, bool]] = []
    for param in model.parameters():
        state.append((param, bool(param.requires_grad)))
        if param.requires_grad:
            param.requires_grad_(False)
    return state


def _restore_model_after_inference(state: Optional[list[tuple[torch.nn.Parameter, bool]]]) -> None:
    if not state:
        return
    for param, requires_grad in state:
        param.requires_grad_(requires_grad)

def _euler_sdeint_fixed_population(
    sde: nn.Module,
    initial_state: tuple[torch.Tensor, torch.Tensor],
    *,
    dt: float,
    ts: torch.Tensor,
    noise_generator: Optional[torch.Generator] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Euler-Maruyama integration without particle splitting or graph carry-over.

    Unlike the historical downstream integrator, this helper takes a shortened
    final step before every requested output time.  It also detaches the state
    after each step.  The latter is important for inference with score or
    potential-based interaction forces: those forces need a small, local
    autograd calculation, but the simulation must not retain a graph spanning
    all integration steps.
    """
    import math

    if float(dt) <= 0 or not np.isfinite(float(dt)):
        raise ValueError("dt must be a finite value > 0.")
    if ts.ndim != 1 or ts.numel() == 0:
        raise ValueError("ts must be a non-empty one-dimensional tensor.")

    ts_list = [float(value) for value in ts.detach().cpu().tolist()]
    if not all(np.isfinite(ts_list)):
        raise ValueError("ts must contain only finite values.")
    if any(right <= left for left, right in zip(ts_list[:-1], ts_list[1:])):
        raise ValueError("ts must be strictly increasing.")

    z, lnw = initial_state
    current_state = (z.detach(), lnw.detach())
    current_time = ts_list[0]
    output_states = [current_state]

    for target_time in ts_list[1:]:
        while current_time < target_time - 1e-8:
            step_dt = min(float(dt), target_time - current_time)
            t_tensor = torch.tensor(
                [current_time], device=z.device, dtype=z.dtype
            )
            f_z, f_lnw = sde.f(t_tensor, current_state)
            g_z = sde.g(t_tensor, current_state[0])
            noise_z = torch.randn(
                current_state[0].shape,
                dtype=current_state[0].dtype,
                device=current_state[0].device,
                generator=noise_generator,
            ) * math.sqrt(step_dt)
            new_z = current_state[0] + f_z * step_dt + g_z * noise_z
            new_lnw = current_state[1] + f_lnw * step_dt
            current_state = (new_z.detach(), new_lnw.detach())
            current_time += step_dt
        output_states.append(current_state)

    traj_z = torch.stack([state[0] for state in output_states], dim=0)
    traj_lnw = torch.stack([state[1] for state in output_states], dim=0)
    return traj_z, traj_lnw


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
    noise_seed: Optional[int] = None,
    growth_mode: str = "learned",
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate a fixed population from explicit initial states.

    This is the identity-preserving, non-split counterpart of
    :func:`simulate_sde_points`.  It never samples, duplicates, removes, or
    reorders rows, so row ``i`` at every returned time belongs to row ``i`` of
    ``x0``.  That makes the function suitable for lineage-, clone-, or
    perturbation-aware downstream evaluation.

    Parameters
    ----------
    x0
        Initial state matrix with shape ``(n_particles, n_features)``.  NumPy
        arrays, array-like objects, and torch tensors are accepted.  Initial
        particle mass is uniform and sums to one, matching the current
        non-split downstream simulator.
    model
        A current :class:`~CytoBridge.tl.core.models.DynamicalModel`, already
        placed on ``device``.
    ts_points
        Strictly increasing output times.  The first returned state is exactly
        ``x0`` at ``ts_points[0]``.
    dt
        Maximum Euler-Maruyama step size.  A shorter step is used to land
        exactly on every requested output time.
    sigma
        Scalar Brownian diffusion amplitude.  Use zero for a deterministic
        noise ablation.
    include_score
        Add the learned score gradient when the model contains a score
        component.
    include_interaction
        Add the learned interaction force when the model contains an
        interaction component.
    interaction_m
        Interaction grouping size.  It is capped at the number of particles
        so small explicit populations remain valid.
    device
        Torch device used for simulation.  The caller is responsible for
        placing ``model`` on the same device.
    noise_seed
        Optional seed for a dedicated Brownian-noise generator.  Interaction
        grouping uses the ordinary torch RNG, so a dedicated generator makes
        the diffusion increments exactly paired across model conditions even
        when only one condition evaluates a GNN interaction.
    growth_mode
        ``"learned"`` evolves particle masses with the fitted growth network.
        ``"frozen_uniform"`` sets ``d log w / dt = 0`` throughout the rollout,
        so every initial particle keeps mass ``1 / N``.  The frozen masses are
        also supplied to mass-aware interaction models, making this mode the
        appropriate model-side control for proliferation-neutral dynamics
        evaluation.
    verbose
        Print a compact start/end summary.

    Returns
    -------
    points, weights, normalized_weights
        Dense NumPy arrays with shapes ``(T, N, D)``, ``(T, N, 1)``, and
        ``(T, N, 1)``.  ``weights`` are the unnormalized growth masses used by
        the current distribution evaluator; ``normalized_weights`` sum to one
        independently at each time point.

    Notes
    -----
    Score gradients and potential-force gradients are evaluated only inside
    each drift call.  Model parameters are frozen temporarily and the state is
    detached after every Euler step, so inference does not build a trajectory-
    length autograd graph.  Parameter ``requires_grad`` flags and train/eval
    mode are restored before returning, including when simulation raises.
    """
    from CytoBridge.tl.core.interaction import cal_interaction

    if model is None:
        raise ValueError("model must be provided.")
    if not hasattr(model, "predict_velocity"):
        raise TypeError(
            "model must provide the current DynamicalModel prediction API "
            "(missing predict_velocity)."
        )
    if isinstance(x0, torch.Tensor):
        x0_array = x0.detach().to(device="cpu", dtype=torch.float32).numpy()
    else:
        x0_array = np.asarray(x0)
    if x0_array.ndim != 2:
        raise ValueError(
            f"x0 must have shape (n_particles, n_features), got {x0_array.shape}."
        )
    if x0_array.shape[0] == 0 or x0_array.shape[1] == 0:
        raise ValueError("x0 must contain at least one particle and one feature.")
    if not np.issubdtype(x0_array.dtype, np.number) or np.iscomplexobj(x0_array):
        raise TypeError("x0 must contain real numeric values.")
    if not np.isfinite(x0_array).all():
        raise ValueError("x0 must contain only finite values.")
    if float(dt) <= 0 or not np.isfinite(float(dt)):
        raise ValueError("dt must be a finite value > 0.")
    if float(sigma) < 0 or not np.isfinite(float(sigma)):
        raise ValueError("sigma must be a finite value >= 0.")
    if int(interaction_m) < 2:
        raise ValueError("interaction_m must be >= 2.")
    growth_mode = str(growth_mode)
    if growth_mode not in {"learned", "frozen_uniform"}:
        raise ValueError(
            "growth_mode must be one of {'learned', 'frozen_uniform'}."
        )

    try:
        if isinstance(ts_points, torch.Tensor):
            ts_values = [
                float(value) for value in ts_points.detach().cpu().reshape(-1).tolist()
            ]
        else:
            ts_values = [float(value) for value in ts_points]
    except TypeError as exc:
        raise TypeError("ts_points must be an iterable of numeric times.") from exc
    if not ts_values:
        raise ValueError("ts_points must be non-empty.")
    if not all(np.isfinite(ts_values)):
        raise ValueError("ts_points must contain only finite values.")
    if any(right <= left for left, right in zip(ts_values[:-1], ts_values[1:])):
        raise ValueError("ts_points must be strictly increasing.")

    x0_t = torch.as_tensor(x0_array, dtype=torch.float32, device=device).detach()
    n_particles = int(x0_t.shape[0])
    lnw0 = torch.full(
        (n_particles, 1),
        fill_value=-float(np.log(n_particles)),
        dtype=x0_t.dtype,
        device=x0_t.device,
    )
    initial_state = (x0_t, lnw0)
    noise_generator = None
    if noise_seed is not None:
        if isinstance(noise_seed, (bool, np.bool_)):
            raise ValueError("noise_seed must be an integer or None.")
        try:
            resolved_noise_seed = int(noise_seed)
        except (TypeError, ValueError) as exc:
            raise ValueError("noise_seed must be an integer or None.") from exc
        noise_generator = torch.Generator(device=x0_t.device)
        noise_generator.manual_seed(resolved_noise_seed)

    components = set(getattr(model, "components", []))
    if "velocity" not in components:
        raise ValueError("Model missing required 'velocity' component.")
    interaction_net = getattr(model, "interaction_net", None)
    use_mass = bool(getattr(model, "use_growth_in_ode_inter", True))
    interaction_group_size = min(int(interaction_m), n_particles)

    class IdentityPreservingSDE(nn.Module):
        noise_type = "diagonal"
        sde_type = "ito"

        def f(self, t, state):
            z, lnw = state
            t_expand = t.expand(z.shape[0], 1).to(device=z.device, dtype=z.dtype)

            # Velocity and growth need no input gradients during inference.
            with torch.no_grad():
                drift = model.predict_velocity(t=t_expand, x=z).detach()
                if growth_mode == "frozen_uniform":
                    dlnw = torch.zeros_like(lnw)
                elif "growth" in components:
                    dlnw = model.predict_growth(t=t_expand, x=z).detach()
                else:
                    dlnw = torch.zeros_like(lnw)

            net_forces = torch.zeros_like(z)
            if (
                include_interaction
                and "interaction" in components
                and interaction_net is not None
                and n_particles > 1
            ):
                # Potential models require local autograd with respect to pair
                # displacements.  GNN models do not, but are also safe in this
                # bounded context.  Detaching the result prevents graph carry-
                # over to the next integration step.
                with torch.enable_grad():
                    cutoff = float(getattr(interaction_net, "cutoff", 1000.0))
                    net_forces = cal_interaction(
                        z=z.detach(),
                        lnw=lnw.detach(),
                        interaction_potential=interaction_net,
                        m=interaction_group_size,
                        cutoff=cutoff,
                        use_mass=use_mass,
                        t=t.detach(),
                    ).float().detach()

            score_grad = torch.zeros_like(z)
            if include_score and "score" in components:
                with torch.enable_grad():
                    z_for_score = z.detach().requires_grad_(True)
                    _, score_grad = model.compute_score(
                        t=t_expand.detach(),
                        x=z_for_score,
                        create_graph=False,
                    )
                    score_grad = score_grad.detach()

            return (drift + net_forces + score_grad).detach(), dlnw.detach()

        def g(self, _t, z):
            return torch.ones_like(z) * float(sigma)

    if verbose:
        print(
            "[simulate_sde_from_x0] start | "
            f"n_init={n_particles}, dim={x0_t.shape[1]}, "
            f"ts_points={len(ts_values)}, dt={dt}, sigma={sigma}, "
            f"include_score={bool(include_score)}, "
            f"include_interaction={bool(include_interaction)}, "
            f"growth_mode={growth_mode}"
        )

    freeze_state = _freeze_model_for_inference(model)
    module_training_states = (
        [(module, bool(module.training)) for module in model.modules()]
        if isinstance(model, nn.Module)
        else None
    )
    model_was_training = bool(getattr(model, "training", False))
    try:
        if hasattr(model, "eval"):
            model.eval()
        # Keep the integration clock in float64 so distinct user-requested
        # times do not collapse when the model state itself is float32.
        ts_tensor = torch.tensor(ts_values, dtype=torch.float64, device=x0_t.device)
        points_t, log_weights_t = _euler_sdeint_fixed_population(
            IdentityPreservingSDE(),
            initial_state,
            dt=float(dt),
            ts=ts_tensor,
            noise_generator=noise_generator,
        )
        weights_t = torch.exp(log_weights_t)
        normalized_weights_t = torch.exp(
            log_weights_t - torch.logsumexp(log_weights_t, dim=1, keepdim=True)
        )
        points = points_t.detach().cpu().numpy()
        weights = weights_t.detach().cpu().numpy()
        normalized_weights = normalized_weights_t.detach().cpu().numpy()
    finally:
        _restore_model_after_inference(freeze_state)
        if module_training_states is not None:
            # Restore heterogeneous submodule modes exactly (for example, an
            # edge predictor intentionally kept in eval mode inside a model
            # that was otherwise training).
            for module, was_training in module_training_states:
                module.training = was_training
        elif hasattr(model, "train"):
            model.train(model_was_training)

    if verbose:
        print(
            "[simulate_sde_from_x0] done | "
            f"points_shape={points.shape}, weights_shape={weights.shape}"
        )
    return points, weights, normalized_weights



