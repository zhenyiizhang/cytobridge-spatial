"""Split-SDE integration used for the AD population and perturbation figures.

Adapted from run_trem2_whole_tissue_scale1.py (20 August 2026).
The drift, score, interaction and mass-weight updates are unchanged.
"""
import numpy as np
import torch
import torch.nn as nn
from CytoBridge.tl.core.interaction import cal_interaction
from CytoBridge.tl.downstream.simulation import (
    _euler_sdeint_split, _freeze_model_for_inference, _restore_model_after_inference,
)

def simulate_from_x0(x0: np.ndarray, model, *, device, times, dt=0.01, sigma=0.03, interaction_m=1024) -> list[np.ndarray]:
    x0_t = torch.tensor(np.asarray(x0, dtype=np.float32), device=device)
    lnw0 = torch.log(torch.ones(x0_t.shape[0], 1, device=device) / x0_t.shape[0])
    interaction_net = getattr(model, "interaction_net", None)
    components = set(getattr(model, "components", []))
    use_mass = bool(getattr(model, "use_growth_in_ode_inter", True))

    class SDE(nn.Module):
        noise_type = "diagonal"
        sde_type = "ito"

        def f(self, t, y):
            z, lnw = y
            t_expand = t.expand(z.shape[0], 1).to(dtype=z.dtype)
            with torch.no_grad():
                drift = model.predict_velocity(t=t_expand, x=z)
                dlnw = (
                    model.predict_growth(t=t_expand, x=z)
                    if "growth" in components
                    else torch.zeros_like(lnw)
                )
            force = torch.zeros_like(z)
            if "interaction" in components and interaction_net is not None:
                with torch.no_grad():
                    force = cal_interaction(
                        z=z,
                        lnw=lnw,
                        interaction_potential=interaction_net,
                        m=interaction_m,
                        cutoff=1000,
                        use_mass=use_mass,
                        t=t,
                    ).float()
            if "score" in components:
                z_req = z.detach().requires_grad_(True)
                _, score_grad = model.compute_score(
                    t=t_expand.detach(), x=z_req, create_graph=False
                )
                drift = drift + score_grad
            return drift + force, dlnw

        def g(self, t, z):
            return torch.ones_like(z) * sigma

    frozen = _freeze_model_for_inference(model)
    try:
        points, _ = _euler_sdeint_split(
            SDE(),
            (x0_t, lnw0),
            dt=dt,
            ts=torch.tensor(times, dtype=torch.float32, device=device),
            noise_std=0.0,
        )
        return [p.detach().cpu().numpy().astype(np.float32, copy=False) for p in points]
    finally:
        _restore_model_after_inference(frozen)
