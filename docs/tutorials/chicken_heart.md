# Developing chicken heart

**Notebook:** {download}`notebooks/05_chicken_heart.ipynb <../../notebooks/05_chicken_heart.ipynb>`

The GSE149457 D4/D7/D10/D14 series uses a dataset-specific preparation step
before the ordinary CytoBridge graph, training, and downstream workflow. The
adapter recovers raw integer counts from the four public 10x matrices, selects
the 3,550 reviewed spots in fixed order, preserves the reviewed anatomy, and
runs the package's strict normalization, batch-aware HVG selection, and 50-PC
expression preprocessing.

## Alignment contract

The generic spatial registration routine is not fitted on these four Visium
slides. The prepared input must satisfy all of the following checks:

- D4 Atria are above Valves, and Valves are above the unsplit Ventricle;
- D7, D10, and D14 Atria are above Valves;
- D7, D10, and D14 Right ventricle lies on the same positive-x side of Compact
  LV/inter-ventricular septum; and
- all coordinates, time labels, spot order, raw sources, and any explicit
  repair are recorded by SHA-256 in the preparation manifest.

The retained legacy local alignment has a single D7 horizontal mirror. The
adapter rejects it by default. `--repair-legacy-d7-left-right` reflects only D7
around its stage mean x, preserves every within-D7 pairwise distance, leaves
D4/D10/D14 byte-identical, and records before/after coordinate hashes. It may
not repair any other orientation failure.

```bash
python scripts/prepare_chicken_heart_input.py \
  --raw-dir /data/GSE149457_RAW \
  --metadata-h5ad /data/chicken_heart_spatial_merged_with_meta.h5ad \
  --aligned-reference-h5ad /data/heart_aligned_all_timepoints.h5ad \
  --graph-database CytoBridge/workflow_databases/CellChatDB.ligrec.human.csv \
  --repair-legacy-d7-left-right \
  --output-h5ad /runs/chicken-input/chicken_heart_aligned_package.h5ad \
  --output-table /runs/chicken-input/model_input.csv \
  --manifest /runs/chicken-input/manifest.json
```

Prefer a corrected reviewed reference when one is available; it should pass
without the compatibility flag.

## Train and analyze with the package

The `chicken_heart` preset validates and copies the prepared H5AD without
refitting coordinates. It then builds four interaction graphs, fits a new edge
predictor with a validation-selected threshold, executes the packaged six-stage
model plan, and runs standard downstream analysis.

```bash
cytobridge workflow --config chicken_heart --train \
  --input-h5ad /runs/chicken-input/chicken_heart_aligned_package.h5ad \
  --output-dir /runs/chicken-heart-full \
  --device cuda:0
```

The human CellChatDB table is an explicitly scoped conserved-symbol proxy;
there is no bundled Gallus gallus CellChatDB release. Do not describe its output
as a chicken-specific interaction database.

## Velocity and figure semantics

Expression/gene velocity is evaluated in the 50-PC state. Its scVelo transition
graph is built from the 50-dimensional expression state and derivative, then
projected onto the observed `spatial_aligned[:, :2]` coordinates. Spatial
velocity already occupies the first two model dimensions and is plotted
directly on those coordinates; it is never projected through scVelo a second
time.

The standard downstream bank contains generated slices, the time mosaic,
growth, composition, paired intrinsic/interaction/full spatial and gene
velocity, sparse cell-type attention, strict LR tables, temporal gene programs,
and 3D communication. Formal manuscript panels must be assembled from a signed
fresh run, not from notebook previews or the historical D7-mirrored artifact.
