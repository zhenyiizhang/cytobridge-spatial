# Zebrafish embryogenesis

**Notebook:** {download}`notebooks/01_zebrafish.ipynb <../../notebooks/01_zebrafish.ipynb>`

The notebook is an inspectable walkthrough of interpolation, classification,
composition, velocity, growth, sparse communication, strict ligand-receptor
projection, and unwarped distribution evaluation. It expects an aligned H5AD,
a trained checkpoint unless training is explicitly enabled, and an LR table.

## Compact and formal scope

The default compact run uses at most 5,000 generated particles and inserts one midpoint
per observed interval. `RUN_FORMAL_SCOPE=True` reads the formal scope directly
from the packaged preset:

- observed model times `0, 1, 2, 3, 4` and interpolated times
  `0.5, 1.5, 2.5, 3.5`;
- all available cells from the initial observed slice;
- classifier `k=10`, seed 42, `alpha_express=0.015`, and
  `alpha_spatial=10`;
- `sde_dt=0.05`, split `dt=0.05`, split `sigma=0.03`, and growth
  exponent 1.0;
- observed-anchored piecewise split-SDE: every interval starts from all real
  cells at its left observed time point and generates only the requested
  interior midpoint (`per_timepoint`, `include_end=False`).

Accordingly, each half-time state is a one-sided, interval-local forward
simulation from the immediately preceding observed slice. It is not
conditioned on the following observed endpoint, is not a two-endpoint bridge,
and is not one lineage-continuous population rollout from t0. It must not be
reported as global extrapolation.

```bash
cytobridge workflow --config zebrafish --dry-run
jupyter lab notebooks/01_zebrafish.ipynb
```

The k1 classifier is the pointwise-accuracy optimum in the retained sensitivity;
k10 defines the formal spatial-domain output. The notebook does not itself run
the paper's selected gene programs, targeted ablations, or final multi-panel
figure assembly. Those analyses should consume the saved unwarped trajectory
through the public downstream and plotting APIs.
