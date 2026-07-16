# CytoBridge Spatial

CytoBridge Spatial is the core codebase for preprocessing, spatial alignment, interaction graph construction, and dynamical model training on spatial transcriptomics time-series data.

This repository is intended to be the main methods and package repository for manuscript submission and code release. Downstream figure notebooks can live in a separate companion repository. Large raw datasets and ligand-receptor databases are not bundled here.

## Scope

This repository contains:

- the installable Python package `CytoBridge/`
- preprocessing and training scripts in `scripts/`
- training configuration files in `CytoBridge/configs/`
- bundled edge-predictor checkpoints in `edge_classifier/`

This repository does not aim to store:

- large raw or processed datasets
- manuscript-specific downstream notebooks
- external ligand-receptor databases beyond user-supplied resources

## Repository Layout

```text
cytobridge-spatial/
├── CytoBridge/
│   ├── pp/        # preprocessing, spatial alignment, interaction graph construction
│   ├── tl/        # training, model utilities, downstream analysis helpers
│   ├── pl/        # visualization helpers
│   └── configs/   # YAML training configs
├── scripts/       # end-to-end preprocessing / training / evaluation scripts
├── edge_classifier/
├── environment.yml
├── requirements.txt
└── README.md
```

## Tested Environment

The current codebase has been validated in the `cb_pipeline` Conda environment with Python 3.10. The provided `environment.yml` recreates that environment. Existing environments such as `DeepRUOTv2` can also be used after installing every package in `requirements.txt`, including `qnorm`.

## Installation

### Option 1: Reproducible Conda environment

```bash
git clone <your-repo-url>
cd cytobridge-spatial

conda env create -f environment.yml
conda activate cb_pipeline
pip install -e .
```

### Option 2: Install into an existing working environment

If you already have a working environment such as `DeepRUOTv2`:

```bash
conda activate DeepRUOTv2
cd /path/to/cytobridge-spatial
pip install -e .
```

### Optional dependencies

Some plotting utilities rely on optional packages that are not required for the core preprocessing and training workflow.

- `cellrank`: only needed for terminal-state analysis utilities

If needed:

```bash
pip install cellrank
```

## Input Requirements

The main preprocessing pipeline expects an input `.h5ad` file with:

- spatial coordinates in `adata.obsm["spatial"]` or `adata.obs["spatial_x"]`, `adata.obs["spatial_y"]`
- a time annotation column in `adata.obs[time_key]`

For interaction graph construction, the pipeline also expects a ligand-receptor database CSV, by default:

```text
database/CellNEST_database.csv
```

If your database lives elsewhere, pass it explicitly with `--database-path`.

## Quick Start

### 1. Run preprocessing, spatial alignment, interaction graph construction, and edge predictor training

```bash
python scripts/preprocess_pipeline.py \
  --data-name mosta \
  --h5ad-path /path/to/input.h5ad \
  --time-key timepoint \
  --output-dir ./results/mosta_preprocess \
  --database-path /path/to/database/CellNEST_database.csv \
  --device cuda
```

This step produces:

- aligned outputs (`*_aligned.csv`, `*_aligned.h5ad`)
- per-timepoint graph inputs in `input_graph/`
- metadata files in `metadata/`
- a trained edge predictor checkpoint in `edge_classifier/`

### 2. Train the CytoBridge dynamical model

```bash
python scripts/run_spatial_training.py \
  --preset mosta \
  --h5ad_path /path/to/input.h5ad \
  --output_csv ./results/mosta/aligned.csv \
  --output_h5ad ./results/mosta/aligned.h5ad \
  --train_config CytoBridge/configs/st_spatial.yaml \
  --device cuda
```

Alternatively, if preprocessing has already been completed, you can train directly with the package API:

```python
import CytoBridge as cb

cb.tl.fit(
    "/path/to/aligned.h5ad",
    config="CytoBridge/configs/st_spatial.yaml",
    device="cuda",
)
```

## Package API

The main package-level entry points are:

- `CytoBridge.pp.preprocess`
- `CytoBridge.pp.align_spatial`
- `CytoBridge.pp.generate_interaction_graph`
- `CytoBridge.pp.train_edge_predictor`
- `CytoBridge.tl.fit`

A typical AnnData-first workflow is:

```python
import CytoBridge as cb
from CytoBridge.pp import AlignConfig

adata_pre = cb.pp.preprocess(
    adata,
    time_key="timepoint",
    n_top_genes=2000,
    dim_reduction="pca",
    n_pcs=50,
)

adata_aligned = cb.pp.align_spatial(
    adata_pre,
    time_key="timepoint",
    cfg=AlignConfig(n_pcs=50, spatial_dim=2),
    device="cuda",
    output_h5ad="./results/aligned.h5ad",
)

cb.tl.fit(
    adata_aligned,
    config="CytoBridge/configs/st_spatial.yaml",
    device="cuda",
)
```

### Load an existing ST-1104 checkpoint

The compatibility loader is part of the public package API, so downstream
notebooks do not need a copied `DeepRUOT` source tree:

```python
import pandas as pd
from CytoBridge.tl import (
    build_dynamical_runtime,
    compute_velocity_components,
    infer_feature_columns,
    load_legacy_dynamical_model_from_dir,
)

df = pd.read_csv("arista_1108_with_annotation.csv")
features = list(infer_feature_columns(df))
loaded = load_legacy_dynamical_model_from_dir(
    "results/arista_1110",
    edge_predictor_root="edge_classifier",
    device="cuda",
)
runtime = build_dynamical_runtime(loaded)
data_t1 = df.loc[df["samples"].eq(1), features].to_numpy()
components = compute_velocity_components(
    data=data_t1,
    time_value=1.0,
    model=loaded.model,
    device="cuda",
)
```

The feature table must match the checkpoint contract recorded in
`params.yml`; the published ARISTA checkpoint expects 52 model dimensions
(two aligned spatial coordinates followed by 50 expression PCs). The companion
`cb_reproducibility` repository contains the canonical executable notebook.

## Key Scripts

- `scripts/preprocess_pipeline.py`: end-to-end preprocessing, alignment, graph generation, and edge predictor training
- `scripts/run_spatial_training.py`: preset-based spatial training entry point
- `scripts/convert_legacy_weights_to_ckpt.py`: convert legacy checkpoints into the current checkpoint format
- `scripts/run_downstream_workflow_example.py`: downstream example workflow based on trained results

## Outputs

Typical training and preprocessing outputs include:

- aligned `.csv` and `.h5ad` files
- graph inputs and metadata
- trained edge predictor checkpoints
- staged training checkpoints under a results directory
- figures and downstream artifacts generated by analysis scripts

## Companion Resources

For a clean code release, we recommend keeping the following resources separate from this repository:

- a companion repository containing downstream notebooks for figure generation
- dataset download instructions or accession numbers
- the ligand-receptor database files used for preprocessing, if redistribution is restricted

## Citation

If you use this repository in academic work, please cite the associated manuscript once available.
