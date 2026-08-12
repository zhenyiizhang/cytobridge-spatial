# ARISTA salamander brain regeneration

**Notebook:** {download}`notebooks/03_arista.ipynb <../../notebooks/03_arista.ipynb>`

The notebook is an inspectable walkthrough of interpolation, classification,
composition, velocity, growth, sparse communication, strict ligand-receptor
projection, and unwarped distribution evaluation. It expects the aligned H5AD,
a trained checkpoint unless training is explicitly enabled, and an LR table.

## Compact and formal scope

The default compact run uses at most 5,000 generated particles and inserts one midpoint
per observed interval. `RUN_FORMAL_SCOPE=True` reads the formal scope directly
from the packaged preset:

- observed model times `0, 1, 2, 3, 4` and interpolated times
  `0.5, 1.5, 2.5, 3.5`;
- 7,668 particles;
- classifier `k=10`, seed 42, `alpha_express=0.015`, and
  `alpha_spatial=10`;
- `sde_dt=0.05`, split `dt=0.01`, split `sigma=0.03`, and growth
  exponent 1.0.

```bash
cytobridge workflow --config arista --dry-run
jupyter lab notebooks/03_arista.ipynb
```

The notebook does not itself reproduce the selected LR count or the paper's
Figure 5e panel. The split frames do not retain persistent particle identifiers,
so formal lineage is omitted and row order must not be interpreted as ancestry.

For a corrected de novo run, pass the complete 16,379-gene
`Regeneration.h5ad` to `cytobridge workflow --config arista --train`. The
preset selects the five named 2/5/10/15/20-DPI batches by label, not category
position; uses all eight batches for batch-aware HVG selection followed by
pooled PCA fitting; retains
species-matched LR subunits; and constructs a stable `Batch` + `CellID`
observation identity. It retains 46,209 cells because the undocumented extra
20-cell crop in the historical 46,189-cell prepared file cannot be reproduced
without guessing.
