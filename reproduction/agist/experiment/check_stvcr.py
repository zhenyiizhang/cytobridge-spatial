"""Check official stochastic population counts against the fitted continuous ODE.

This is an integration check, not a replacement benchmark or another fit.
"""
import json
import numpy as np
import torch
from torchdiffeq import odeint
from run_baseline import RUN, STVCR
from scripts.spatiotemporal_benchmark.dynamic.run_dynamic import _import_stvcr

_import_stvcr(STVCR)
torch.set_num_threads(8)
model = torch.load(RUN/'stvcr/model.pt', map_location='cuda:1', weights_only=False).eval()
x = torch.as_tensor(np.load(RUN/'data/initial_states.npy'), device='cuda:1')
with torch.no_grad():
    initial_g = model.growth_rate_net(torch.tensor(0., device='cuda:1'), x)
    states, log_weights = odeint(model, (x, torch.zeros((len(x), 1), device='cuda:1')),
        torch.tensor([0., 1., 2., 3.], device='cuda:1'), method='dopri5', rtol=1e-5, atol=1e-5)
    mass = log_weights.exp().mean(dim=1).ravel().cpu().tolist()
result = dict(initial_growth_quantiles=np.quantile(initial_g.cpu().numpy(), [0,.25,.5,.75,1]).tolist(),
    continuous_ode_relative_mass=mass, times=[0,1,2,3],
    purpose='Independent integration check of the same fitted stVCR model')
(RUN/'stvcr/continuous_integration_check.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result), flush=True)
