# Developing chicken heart

**Notebook:** {download}`notebooks/05_chicken_heart.ipynb <../../notebooks/05_chicken_heart.ipynb>`

The GSE149457 D4/D7/D10/D14 series uses a dataset-specific preparation step
before the ordinary CytoBridge alignment, graph, training, and downstream
workflow. The adapter recovers raw integer counts from the four public 10x
matrices and selects the 3,550 reviewed spots in fixed order. A second
deterministic step constructs `obsm['spatial_ot_input']` from the raw
coordinates. The package then fits expression-guided spatial alignment,
batch-aware HVG selection, and a 50-PC expression state.

## Alignment contract

The known reversed D7 raw section is rotated by 180 degrees around its own
stage centroid before OT. D4, D10, and D14 are unchanged. This rigid
pre-orientation preserves every within-D7 pairwise distance and is recorded in
the input manifest. The package then jointly aligns D4/D7/D10/D14 using the
same expression-guided OT implementation exposed by `align_spatial`.

The aligned output must satisfy all of the following checks:

- D4 Atria are above Valves, and Valves are above the unsplit Ventricle;
- D7, D10, and D14 Atria are above Valves;
- D7, D10, and D14 Right ventricle lies on the same positive-x side of Compact
  LV/inter-ventricular septum; and
- all coordinates, time labels, spot order, raw sources, and any explicit
  repair are recorded by SHA-256 in the preparation manifest.

`region` is retained only for these anatomical orientation checks. The
downstream MLP, generated-slice labels, composition, lineage, and grouped
communication summaries use the unsmoothed `celltype_prediction` column. A
legacy schema-2 prepared H5AD may still be paired with its original checkpoint:
the workflow ignores its historical `Annotation = region` alias and reads
`celltype_prediction` explicitly, leaving the 52-dimensional model state
unchanged.

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

This first adapter uses the reviewed file only to fix the public spot roster,
row order, and annotations and to recover the matching raw counts. Convert its
output to the OT input without fitting coordinates:

```bash
python scripts/prepare_chicken_heart_ot_input.py \
  --input-h5ad /runs/chicken-input/chicken_heart_aligned_package.h5ad \
  --output-h5ad /runs/chicken-input/chicken_heart_ot_input.h5ad \
  --output-table /runs/chicken-input/chicken_heart_ot_input.csv \
  --manifest /runs/chicken-input/chicken_heart_ot_input_manifest.json
```

The old reviewed coordinates are retained only in
`obsm['spatial_reviewed_reference']` for post-hoc visual comparison. They are
not passed to the OT objective.

## Train and analyze with the package

The `chicken_heart` preset fits spatial alignment from
`obsm['spatial_ot_input']`, builds four interaction graphs, fits a new edge
predictor with a validation-selected threshold, executes the packaged six-stage
model plan, and runs standard downstream analysis.

```bash
cytobridge workflow --config chicken_heart --train \
  --input-h5ad /runs/chicken-input/chicken_heart_ot_input.h5ad \
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
