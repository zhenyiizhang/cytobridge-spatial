# Data and checkpoints

CytoBridge does not bundle multi-gigabyte datasets or trained checkpoints in
the wheel. It does bundle the small species-matched CellChatDB tables used by
both formal interaction-graph construction and the packaged default strict LR
projection; `--lr-database` remains an explicit downstream override. Raw study
data are public:

| Application | Raw-data source | Required label key | Model time key |
| --- | --- | --- | --- |
| Zebrafish | [CNGB STDS0000057](https://db.cngb.org/stomics/datasets/STDS0000057/data) | `Annotation` | `time_point_processed` |
| MOSTA | [MOSTA download portal](https://db.cngb.org/stomics/mosta/download/) | `Annotation` | `time_point_processed` |
| ARISTA axolotl | [CNGB STDS0000056](https://db.cngb.org/stomics/datasets/STDS0000056/data) | `Annotation` | `time_point_processed` |
| AD mouse | [10x Genomics TgCRND8 Xenium time course](https://www.10xgenomics.com/datasets/xenium-in-situ-analysis-of-alzheimers-disease-mouse-model-brain-coronal-sections-from-one-hemisphere-over-a-time-course-1-standard) | `major_annotation` | `time_point_processed` |

The processed aligned H5ADs and formal 0.015 checkpoints are currently project
artifacts and do not yet have a public archive DOI. They must be deposited and
linked here before the 1.5 stable release. Until then, the release candidate is
fully installable and supports user-provided inputs, but a new reader cannot
download the exact manuscript artifacts from the package documentation alone.

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

The workflow loader follows the training plan in `config.yaml`. Current checkpoints
embed learned edge-predictor weights and are portable between machines. Older
current-format checkpoints without those embedded weights need an explicit
`--edge-predictor-path`. Legacy ST-1104 checkpoints use `params.yml`,
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

Manuscript-specific perturbations, matched cross-method benchmarks, and final
panel assembly are separate analyses built on these outputs; the workflow does
not run them implicitly.
