# Input and output files

Use the [dataset downloads](data_checkpoints.md) with the [dataset tutorials](tutorials/dataset_workflows/index.md). This page describes the fields and file formats used by the package.

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

For graph construction, provide `Ligand`, `Receptor`, `Pathway`, and
`Annotation` columns, as in the included CellChatDB tables. Preserve the
interaction-type annotations used for contact-dependent pairs.

For downstream LR projection, a two-column `ligand,receptor` CSV is sufficient.
The aliases `ligand_symbol` and `receptor_symbol` are also supported there.
Join complex subunits with underscores.

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
