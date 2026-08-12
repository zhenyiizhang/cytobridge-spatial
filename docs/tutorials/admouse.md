# AD mouse disease progression

**Notebook:** {download}`notebooks/04_admouse.ipynb <../../notebooks/04_admouse.ipynb>`

The notebook is an inspectable walkthrough of interpolation, classification,
composition, velocity, growth, sparse communication, strict ligand-receptor
projection, and unwarped distribution evaluation. The package workflow can now
start from the raw H5AD; the notebook's compact downstream walkthrough can also
use an existing aligned H5AD and trained checkpoint. LR analysis uses the
bundled mouse CellChatDB unless `--lr-database` explicitly overrides it.

## Compact and formal scope

The default compact run uses at most 5,000 generated particles and inserts one midpoint
per observed interval. `RUN_FORMAL_SCOPE=True` reads the formal scope directly
from the packaged preset:

- observed model times `0, 1, 2` and a 0.1-spaced interpolation grid through
  model time 2.5, excluding observed anchors;
- all available cells from the initial observed slice;
- classifier `k=1`, seed 42, `alpha_express=0.015`, and
  `alpha_spatial=10`;
- `sde_dt=0.05`, split `dt=0.01`, split `sigma=0.03`, and growth
  exponent 1.0.

```bash
cytobridge workflow --config admouse --dry-run
jupyter lab notebooks/04_admouse.ipynb
```

The preset now provides a corrected de novo raw-H5AD recipe: raw Timepoint
values 1/2/3 map to model times 0/1/2, all three batches are aligned, and a new
edge predictor uses the bundled mouse CellChatDB. This is distinct from the
historical matched run that reused released aligned and edge-model artifacts.
The notebook records the t1
holdout protocol in prose but does not fit the holdout model, run cross-method
baselines, or execute whole-tissue and Microglia-only perturbations. Those are
separate analyses, and the two perturbation scopes are different estimands that
must be reported separately.
