# Data and checkpoints

## What to download

Choose a starting point before downloading large files.

| What you want to do | Files needed |
| --- | --- |
| Draw a figure from saved numerical results | Source repository or installed package, as stated in the figure tutorial |
| Repeat downstream analysis without training | Matching aligned H5AD, final dynamical and score checkpoints, configuration, and edge predictor |
| Train again from the paper input | Counts H5AD with the paper's observations, annotations, coordinates, and training settings |
| Repeat a sensitivity analysis or benchmark | Its own models, evaluation inputs, and settings, not just the main dataset model |

The full study-data downloads are not linked here yet. The original data
portals below provide the study data, but not necessarily the assembled H5AD
used in this paper. Downloading the repository alone is therefore sufficient
for many numerical figure redraws, **not for a complete retraining of the
paper**.

## Model files included with the source repository

These files are available in a
[GitHub source checkout](https://github.com/zhenyiizhang/cytobridge-spatial).
They are not part of the pip wheel.

| Dataset | Directory under `release_artifacts/` | Model files |
| --- | --- | --- |
| MOSTA | `mosta_package_native_corrected_20260826_v1/model/` | Final dynamical and score models, classifier, and configuration |
| ARISTA | `arista_package_native_spatialqc_z50_retrain_20260824_r1/main_run/` | Training stages, classifiers, configuration, and run record |
| Chicken heart | `chicken_heart_model_c72e592/` | Final dynamical model (8.6 MB), score model (1.4 MB), edge predictor (0.24 MB), configuration, and run record |
| Zebrafish and AD mouse | Not yet included as complete model downloads | Use the figure tutorials for the numerical results already included |

The repository's `edge_classifier/` directory contains **edge predictors**.
Those files alone are not trained CytoBridge dynamical models.

For chicken heart, see the
[model instructions](https://github.com/zhenyiizhang/cytobridge-spatial/tree/main/release_artifacts/chicken_heart_model_c72e592).
Its aligned H5AD is about 145 MB and will be provided separately. The current
figure-code archive does not contain the complete daily-interpolation inputs.

## Files needed for the full paper downloads

| Dataset | Required data download | Additional files for paper-specific analyses |
| --- | --- | --- |
| Zebrafish | Raw counts H5AD and matching `zebrafish_aligned.h5ad` | Evaluation-cell selections, classifier, model-comparison records, and sensitivity-run models |
| MOSTA | Raw counts H5AD and matching `mosta_aligned.h5ad` | Panel-specific trajectory, lineage, gene, and LR inputs listed in the figure index |
| ARISTA | Raw counts H5AD and matching `arista_aligned.h5ad` | Panel inputs for S19–S22 and Figure 5, including spatial and lineage data |
| AD mouse | Counts H5AD with sample and cell labels, and matching `admouse_aligned.h5ad` | Figure 6 and S26–S30 panel inputs and fitted classifiers |
| Chicken heart | `chicken_heart_aligned.h5ad`, raw 10x matrices, `chicken_heart_spatial_merged_with_meta.h5ad`, and `heart_aligned_all_timepoints.h5ad` | Daily interpolated slices, the corrected classifier, anatomical annotations, and the plotting helpers |
| Weinreb and scNT cortex | Dataset-specific preprocessing inputs and Full/No-interaction run directories | Clone/fate or RNA measurements and attribution inputs |
| AGIST | Simulation configuration and generated training arrays | Fitted model and simulated trajectories |
| Cross-dataset comparisons | Input selections, transforms, and predictions for each method and target time | Repeated-seed and altered-setting runs |

These are the files still needed to test reproduction from beginning to end.
The [figure guide](paper_reproduction.md) distinguishes calculations using
included numerical data from pages assembled from existing panels.

CytoBridge does not include large study datasets or trained model directories
in the wheel. Provide those files explicitly when running a workflow. The
package does include small species-specific CellChatDB tables under
`CytoBridge/workflow_databases/`; `--lr-database` can select another compatible
table for downstream analysis.

## Raw inputs

| Application | Source | Time and label keys | Expression and coordinates |
| --- | --- | --- | --- |
| Zebrafish | [CNGB STDS0000057](https://db.cngb.org/stomics/datasets/STDS0000057/data) | `time`, `bin_annotation` | `layers['counts']`, `obs[['spatial_x', 'spatial_y']]` |
| MOSTA | [MOSTA portal](https://db.cngb.org/stomics/mosta/download/) | `timepoint`, `annotation` | `layers['count']`, `obsm['spatial']` |
| ARISTA axolotl | [CNGB STDS0000056](https://db.cngb.org/stomics/datasets/STDS0000056/data) | `Batch`, `Annotation` | `layers['counts']`, `obsm['spatial']` |
| AD mouse | [10x Genomics TgCRND8 Xenium time course](https://www.10xgenomics.com/datasets/xenium-in-situ-analysis-of-alzheimers-disease-mouse-model-brain-coronal-sections-from-one-hemisphere-over-a-time-course-1-standard) | `Timepoint`, `major_annotation` | `layers['counts']`, `obsm['spatial']` |
| Chicken heart | [GEO GSE149457](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE149457) | `timepoint`, `celltype_prediction` | four raw 10x matrices, `obsm['spatial_original']` |

The dataset configuration maps these source keys to a common representation during
preprocessing.

## Repository figure releases

The source checkout contains two larger, versioned figure releases under
`release_artifacts/`:

- `mosta_package_native_corrected_20260826_v1/` contains the model files,
  numerical inputs used by the figures, calculation scripts, plotting code,
  and vector pages
  used by the MOSTA figure notebooks.
- `arista_package_native_spatialqc_z50_retrain_20260824_r1/` contains the
  ARISTA training record, downstream outputs, figure-building scripts, and
  figure pages.

These directories are available from a source checkout but are not installed
with the wheel. The full aligned MOSTA H5AD remains an external input because
of its size.

## Prepared AnnData

Downstream workflows require:

- `.obs['time_point_processed']`, with every observed time used by the configuration;
- the annotation column named by the configuration;
- `.obsm['X_latent']`, in the feature order used to fit the checkpoint;
- `.obsm['spatial_aligned']`, with two aligned spatial dimensions; and
- finite arrays with the same observation order across `.obs` and `.obsm`.

Gene reconstruction and ligand--receptor projection also use `.varm['PCs']`,
matching `.var_names`, and `.var['pca_center']`. The
`--allow-complete-reference-pca-center-fallback` option is available only for a
complete PCA-fit reference object whose stored latent coordinates can be
reconstructed from its inferred center.

Observation names must be unique unless a configuration defines identity columns.
The ARISTA configuration combines `Batch` and `CellID`; the AD configuration combines
`sample` and `cell_id`.

Chicken-heart preparation uses:

```text
scripts/prepare_chicken_heart_input.py
scripts/prepare_chicken_heart_ot_input.py
```

The first script matches raw counts to the reference spot roster and
annotations. The second writes `obsm['spatial_ot_input']` and applies the
configuration's D7 pre-orientation before spatial alignment.

## Checkpoint directory

Current checkpoints use this layout:

```text
model_dir/
├── config.yaml
├── Finetune/
│   └── last_model.pth
└── Score_Refine/
    └── score_model.pth
```

The selected dynamical checkpoint may be `last_model.pth` or
`best_model.pth`, according to `config.yaml`. Predictor-gated checkpoints can
store the edge predictor with the model. If a checkpoint does not include it,
pass `--edge-predictor-path`.

For the normal workflow, pass the complete `training/` directory rather than a
checkpoint file. The loader reads `config.yaml` to select the recorded final
dynamical and score stages. Current training runs also record the aligned H5AD
identity in `training_run_summary.json`; downstream analysis checks that record
before opening the H5AD. Edge-predictor metadata records
the matching aligned H5AD and selected threshold for the same reason.

Older ST-1104 directories use `params.yml`, `model_final`, and `score_model`.
Load them with `cb.tl.load_legacy_dynamical_model_from_dir`.

## Ligand--receptor table

Provide a CSV with `ligand` and `receptor` columns. The aliases
`ligand_symbol` and `receptor_symbol`, and a two-column table, are also
supported. Join complex subunits with underscores.

All complex subunits must be present. `min` uses the least-expressed subunit;
`geometric_mean` uses a zero-preserving geometric mean while keeping the
complete-subunit requirement. The licensing terms of a custom database remain
the user's responsibility.

## Downstream files

The workflow writes a directory with this structure:

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
├── gene_dynamics/
├── ligand_receptor/
└── reconstruction_diagnostic/
```

`reconstruction_diagnostic/` is created only when requested. Communication
selection counts are recorded in
`summary.json` at `analyses.communication.edge_selection_by_time`. Each time
entry includes `candidate_count`, `selected_count`, `selected_fraction`, and
`status`.

When a time point contains no retained communication edges, its sparse arrays
have shapes `(2, 0)` for `edge_index` and `(0,)` for `attn_mean`. The
No-interaction model omits communication and ligand--receptor directories. The
No-LR model retains interaction and uses
`all_spatial` in place of a learned edge gate.
