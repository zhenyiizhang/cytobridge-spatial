# MOSTA mouse organogenesis

**Notebook:** {download}`notebooks/02_mosta.ipynb <../../notebooks/02_mosta.ipynb>`

The notebook is an inspectable walkthrough of interpolation, classification,
composition, velocity, growth, sparse communication, strict ligand-receptor
projection, and unwarped distribution evaluation. It expects the aligned H5AD,
a trained checkpoint unless training is explicitly enabled, and an LR table.

## Compact and formal scope

The default compact run uses at most 5,000 generated particles and inserts one midpoint
per observed interval. `RUN_FORMAL_SCOPE=True` reads the formal scope directly
from the packaged preset:

- observed model times `0, 1, 2, 3` and nine quarter-step interpolation times;
- 12,000 particles;
- classifier `k=10`, seed 42, `alpha_express=0.015`, and
  `alpha_spatial=10`;
- `sde_dt=0.05`, split `dt=0.05`, split `sigma=0.03`, and growth
  exponent 1.0.

```bash
cytobridge workflow --config mosta --dry-run
jupyter lab notebooks/02_mosta.ipynb
```

The notebook does not itself select Brain gene/type programs or assemble the
paper's Figure 4 supplementary panels. The accepted training run retains exact
sparse loss records rather than a complete per-epoch curve; unlogged epochs are
not interpolated.
