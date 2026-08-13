# Data and checkpoints

CytoBridge does not bundle multi-gigabyte datasets or trained checkpoints in
the wheel. It does bundle the small species-matched CellChatDB tables used by
the packaged default strict LR projection and by the predictor-gated graph
workflows. In AD main, seven strict complete panel-covered pairs from this table
label the learned edge predictor and define the downstream projection. This is
panel-limited evidence, not a global CCI screen. `--lr-database` remains an
explicit downstream override. Raw study data are public:

| Application | Raw-data source | Raw time / label keys | Raw expression / coordinates |
| --- | --- | --- | --- |
| Zebrafish | [CNGB STDS0000057](https://db.cngb.org/stomics/datasets/STDS0000057/data) | `time` / `bin_annotation` | `layers['counts']` / `obs[['spatial_x', 'spatial_y']]` |
| MOSTA | [MOSTA download portal](https://db.cngb.org/stomics/mosta/download/) | `timepoint` / `annotation` | `layers['count']` / `obsm['spatial']` |
| ARISTA axolotl | [CNGB STDS0000056](https://db.cngb.org/stomics/datasets/STDS0000056/data) | `Batch` / `Annotation` | `layers['counts']` / `obsm['spatial']` |
| AD mouse | [10x Genomics TgCRND8 Xenium time course](https://www.10xgenomics.com/datasets/xenium-in-situ-analysis-of-alzheimers-disease-mouse-model-brain-coronal-sections-from-one-hemisphere-over-a-time-course-1-standard) | `Timepoint` / `major_annotation` | `layers['counts']` / `obsm['spatial']` |

These are the raw-input keys consumed by the four packaged presets. The
preprocessing output standardizes them to `time_point_processed`, the preset's
public annotation key, `obsm['X_latent']`, and `obsm['spatial_aligned']` for the
shared training and downstream APIs. Raw observation names must be unique unless
the preset declares stable identity columns. ARISTA reuses local `CellID` values
across batches, so its preset constructs a reversible `Batch` + `CellID`
identity; it does not assign identities from row order.
AD mouse similarly uses `sample` plus `cell_id`, so the original and the
notebook-deduplicated handoff H5ADs resolve to the same stable identity scheme.

The ARISTA preset accepts the complete 16,379-gene `Regeneration.h5ad`. It maps
the five named injury batches at 2/5/10/15/20 DPI to model times 0–4, uses all
eight study batches for batch-aware HVG selection and pooled PCA fitting, and then restricts
alignment/training to those five named batches (46,209 cells). The historical
fixed-2,000-gene file contains 46,189 of those cells, but no complete executable
rule survives for its additional 20 spatial exclusions. The corrected workflow
therefore retains all 46,209 provable cells and records this 0.043% historical
scope difference; it does not hard-code cell IDs or infer a crop from the
desired count. The raw H5AD also contains eight batch-keyed dense segmentation
rasters in `uns` (about 1.9 GiB total). They are imaging attachments, not model
inputs; the preset removes only those named keys before preprocessing and
records the removal while retaining annotations, colors, coordinates, and
expression data.

The processed aligned H5ADs and completed 0.015 checkpoints are currently
project artifacts and do not yet have a public archive DOI. The corrected AD
learned-predictor main and historical matched checkpoint are complete; the
matched no-LR-prior ablation remains pending. Completed artifacts must be
deposited and linked here before the 1.5 stable release. Until then, the release
candidate is fully installable and supports user-provided inputs, but a new
reader cannot download the exact manuscript artifacts from the package
documentation alone.

## Aligned AnnData contract

Every downstream workflow requires:

- `.obs[time_point_processed]`: numeric model time with every preset-observed
  anchor present;
- the dataset label column from the table above;
- `.obsm['X_latent']`: fitted latent state, in the checkpoint's feature order;
- `.obsm['spatial_aligned']`: aligned two-dimensional coordinates;
- finite arrays and the same row order across `.obs` and both `.obsm` arrays.

Gene dynamics and LR projection additionally require the fitted gene-space
reference: `.varm['PCs']` and matching `.var_names`. Current preprocessing
persists `.var['pca_center']`; complete historical aligned H5AD objects that
predate it fail closed by default. Only when the file is known to be the
complete original PCA-fit population may the workflow use its `.X` column
mean via `--allow-complete-reference-pca-center-fallback`; the inferred center
must still reproduce the saved PCA coordinates.
Formal expression summaries clip inverse-PCA log1p values to non-negative
values per cell.

## Checkpoint directory contract

Current checkpoints use:

```text
model_dir/
├── config.yaml
├── Finetune/
│   └── last_model.pth        # or best_model.pth, according to config
└── Score_Refine/
    └── score_model.pth       # optional final score stage
```

The workflow loader follows the training plan in `config.yaml`. Current
predictor-gated checkpoints, including AD main, embed learned edge-predictor
weights and are portable between machines. Older predictor-gated current-format checkpoints without embedded
weights need an explicit `--edge-predictor-path`. Legacy ST-1104 checkpoints use `params.yml`,
`model_final`, and `score_model`; load those through
`cb.tl.load_legacy_dynamical_model_from_dir` for an explicitly labelled
historical analysis. The formal package workflow does not relabel them as the
current resolved six-stage run.

## Ligand–receptor input

Supply a species-appropriate CSV with `ligand` and `receptor` columns. Common
aliases such as `ligand_symbol`/`receptor_symbol` and two-column tables are
accepted. Join complex subunits with underscores. Formal scoring requires all
subunits and uses their minimum expression; geometric mean is an explicit
sensitivity setting. The graph-building CellChatDB resources bundled with
CytoBridge are GPL-3.0 and documented under `CytoBridge/workflow_databases/`;
licensing of a custom downstream database remains the user's responsibility.
The AD expression panel fully represents seven pairs under this strict rule.
They label the main learned edge predictor and its downstream projection;
report both as panel-limited, not as a global CCI screen.

## Main downstream output tree

```text
downstream/
├── summary.json
├── slice_data/time_*.h5ad
├── classifier_cache/
├── velocity/velocity_components.npz
├── growth/growth_by_cell.csv
├── composition/celltype_composition.csv
├── communication/communication_by_celltype.csv
├── communication/sparse_attention/
├── figures/
├── gene_dynamics/             # default for the four packaged presets
├── ligand_receptor/           # default, using the preset species database
└── reconstruction_diagnostic/ # only when requested; not a holdout benchmark
```

`summary.json` exposes per-time communication selection provenance at
`analyses.communication.edge_selection_by_time`. Each normalized time key
contains `candidate_count`, `selected_count`, `selected_fraction`, and
`status`, with the scientific boundary recorded in
`structural_zero_interpretation`. A valid empty time point has
`edge_index.shape == (2, 0)` and `attn_mean.shape == (0,)`; it is not a missing
artifact. Acceptance still requires selected edges and nonzero cell-type
communication somewhere across the full trajectory.

Manuscript-specific perturbations, matched cross-method benchmarks, and final
panel assembly are separate analyses built on these outputs; the workflow does
not run them implicitly.
