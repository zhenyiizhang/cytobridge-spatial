# AD mouse disease progression

**Notebook:** {download}`notebooks/04_admouse.ipynb <../../notebooks/04_admouse.ipynb>`

The notebook is an inspectable walkthrough of interpolation, classification,
composition, velocity, growth, sparse spatial attention, panel-limited strict
ligand-receptor projection, and unwarped distribution evaluation. The package
workflow can start from the raw H5AD; the notebook's compact downstream
walkthrough can also use an existing aligned H5AD and trained checkpoint.

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

The preset provides a corrected de novo raw-H5AD recipe: raw Timepoint values
1/2/3 map to model times 0/1/2 and all three batches are aligned. The AD main
model uses a learned edge predictor at cutoff `0.012106042891492197`. Its
validation-selected threshold is `0.9956824779510498`.

Downstream attention reconstruction is separate. The formal run builds the
radius candidate graph over the complete analyzed time-slice cohort, whereas
the compact notebook uses its explicit seeded particle cap. It evaluates that
sparse graph in edge batches with the learned gate. These are spatial-attention summaries,
not the exact stochastic training graph and not a global cell-cell
communication screen.

The AD assay represents only seven complete ligand-receptor pairs from the
bundled mouse CellChatDB under strict all-subunit matching. Their graph labels
fit the main learned predictor and they also define the downstream projection.
Both outputs are panel-limited, not global CCI inference.

Run the matched no-LR-prior ablation only with its packaged contract:

```bash
cytobridge workflow --config admouse --step downstream \
  --training-config admouse_spatial_full_alpha_express_0015_no_lr_prior.yaml \
  --aligned-h5ad /runs/admouse-no-lr/preprocess/admouse_aligned.h5ad \
  --model-dir /runs/admouse-no-lr/training \
  --output-dir /results/admouse-no-lr-reuse \
  --device cuda
```

The notebook records the t1
holdout protocol in prose but does not fit the holdout model, run cross-method
baselines, or execute whole-tissue and Microglia-only perturbations. Those are
separate analyses, and the two perturbation scopes are different estimands that
must be reported separately.
