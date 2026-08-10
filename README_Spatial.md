# CytoBridge Spatial Pipeline

This guide separates the released zebrafish workflow from the reusable package
APIs. Dataset adapters define biological labels, time mappings, coordinate
conventions, and training profiles; `CytoBridge.pp` and `CytoBridge.tl` provide
the shared algorithms.

## Canonical zebrafish workflow

Use `scripts/run_zebrafish_end_to_end.py`, not the historical
`zebrafish_training.yaml` preset. The manuscript-selected configuration is
`CytoBridge/configs/zebrafish_spatial_full_alpha_express_0015.yaml`:

- `alpha_spatial=10`, `alpha_express=0.015`, `sigma=0.03`, seed 42;
- stages `Pretrain/Refine/Init_interaction/Train_Score/Finetune/Score_Refine`;
- epochs `100/100/50/2001/1000/2001`;
- raw integer-like expression from `layers['counts']`, median-library-size
  normalization, and exactly one `log1p` transform.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_zebrafish_end_to_end.py \
  --h5ad-path /path/to/spatial_sixtime_slice_stereoseq.h5ad \
  --database-path /path/to/CellChatDB.ligrec.zebrafish.csv \
  --output-dir /path/to/runs/zebrafish-clean-counts \
  --profile full \
  --stage all \
  --condition alpha_express_0015 \
  --device cuda \
  --random-seed 42
```

Use `--profile smoke` only to verify wiring. It reduces cells and epochs and is
not a scientific result. The complete staged workflow and output contract are
documented in `docs/zebrafish_clean_counts_workflow.md`.

For leave-one-timepoint-out evaluation, follow
`scripts/spatiotemporal_benchmark/cytobridge/README.md`. Every fold physically
removes the target rows, rebuilds its training-only graph and edge classifier,
and fits all six stages from scratch. Full-data and held-out metrics must remain
separate.

The matched retraining controls are:

- `CytoBridge/configs/zebrafish_spatial_full_alpha_express_0015.yaml`;
- `CytoBridge/configs/zebrafish_spatial_full_alpha_express_0015_no_interaction.yaml`;
- `CytoBridge/configs/zebrafish_spatial_full_alpha_express_0015_no_lr_prior.yaml`.

Train all three from scratch with separate output directories after running
`scripts/reviewer_zebrafish_response/audit_matched_training_configs.py`. The
no-LR-prior condition keeps the trainable interaction GNN and replaces only the
learned LR-informed edge gate with all within-cutoff spatial neighbors; it is
not the same as deleting the interaction module.

`CytoBridge/configs/zebrafish_spatial_full.yaml` is the paired
`alpha_express=0.05` sensitivity condition. The shorter
`CytoBridge/configs/zebrafish_training.yaml` is explicitly legacy verification
only and must not be used for formal full-data, LOTO, or matched-ablation runs.
`scripts/train_zebrafish.py` is retained only as a compatibility alias for the
canonical runner.

## Reusable preprocessing and training APIs

For a new dataset, preprocess expression before spatial alignment. An explicit
raw-count layer activates the strict count contract and avoids re-transforming
an already normalized `adata.X`.

```python
import scanpy as sc
import CytoBridge as cb
from CytoBridge.pp import AlignConfig

adata = sc.read_h5ad("/path/to/data.h5ad")
adata_pre = cb.pp.preprocess(
    adata,
    time_key="time",
    expression_layer="counts",
    raw_count_validation="strict",
    normalization_target_sum=None,
    n_top_genes=2000,
    n_pcs=50,
)

adata_aligned = cb.pp.align_spatial(
    adata_pre,
    time_key="time",
    cfg=AlignConfig(n_pcs=50, spatial_dim=2),
    output_h5ad="./results/aligned.h5ad",
    device="cuda",
)
```

Full interaction-model training additionally needs the interaction graph and
edge predictor. The repository-level `scripts/preprocess_pipeline.py` composes
preprocessing, alignment, graph generation, and edge-predictor training and
writes their cutoff/path/threshold provenance into the aligned H5AD:

```bash
python scripts/preprocess_pipeline.py \
  --data-name my_dataset \
  --h5ad-path /path/to/data.h5ad \
  --time-key time \
  --expression-layer counts \
  --raw-count-validation strict \
  --database-path /path/to/species_ligand_receptor.csv \
  --output-dir ./results/my_dataset_preprocess \
  --device cuda
```

Train directly from that completed artifact:

```python
import CytoBridge as cb

cb.tl.fit(
    "./results/my_dataset_preprocess/my_dataset_aligned.h5ad",
    config="/path/to/dataset_training.yaml",
    ckpt_dir="./results/my_dataset_model",
    device="cuda",
)
```

`cb.tl.fit_spatial_csv(...)` and `cb.tl.fit_spatial_h5ad(...)` remain deprecated
wrappers for backward compatibility. The canonical training entrypoint is
`cb.tl.fit(...)`.
