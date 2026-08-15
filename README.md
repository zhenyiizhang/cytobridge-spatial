# CytoBridge Spatial

CytoBridge Spatial is the complete Python codebase for preprocessing, spatial
alignment, interaction graph construction, dynamical model fitting, downstream
analysis, benchmarking, and visualization of spatial transcriptomics time
series.

This is the release-candidate package and methods repository. New analyses and datasets should
extend these public APIs rather than copy training or downstream pipelines into
separate code trees. Large raw datasets and trained checkpoints are not bundled
here. The wheel does include the small species-matched CellChatDB tables used
by its five supported workflow presets; users can override them with another
compatible database.

Raw-data accessions, aligned-AnnData keys, checkpoint layout, LR-table format,
and the downstream output tree are documented in
[`docs/data_checkpoints.md`](docs/data_checkpoints.md). The processed formal
H5ADs/checkpoints still need a public archive DOI before the 1.5 stable release.

## Scope

This repository contains:

- the installable Python package `CytoBridge/`
- preprocessing and training scripts in `scripts/`
- five dataset tutorials plus one small synthetic preprocessing tutorial in
  `notebooks/`
- package-owned non-spatial Weinreb and scNT workflows, including audited
  preprocessing, matched training, evaluation, attribution, and figure replay
- ReadTheDocs source in `docs/`
- training configuration files in `CytoBridge/configs/`
- legacy repository-scoped edge-predictor checkpoints in `edge_classifier/`
  (these files are not installed into the wheel)

This repository does not aim to store:

- large raw or processed datasets
- large generated manuscript artifacts and raw result bundles
- large or custom ligand-receptor databases beyond the bundled preset
  tables

## Repository Layout

```text
cytobridge-spatial/
├── CytoBridge/
│   ├── pp/        # preprocessing, spatial alignment, interaction graph construction
│   ├── tl/        # training, model utilities, downstream analysis helpers
│   ├── pl/        # visualization helpers
│   └── configs/   # YAML training configs
├── scripts/       # end-to-end preprocessing / training / evaluation scripts
├── notebooks/     # five dataset workflows + one synthetic tutorial
├── docs/          # ReadTheDocs user guide, tutorials, and API reference
├── edge_classifier/
├── environment.yml
├── requirements.txt
└── README.md
```

## Supported Python versions

CytoBridge 1.5 supports Python 3.10 and 3.11. The release checks exercise both
versions. The provided `environment.yml` remains a convenient development
environment, while the wheel extras below are the supported installation
interface.

## Installation

### Option 1: Reproducible Conda environment

```bash
git clone https://github.com/zhenyiizhang/cytobridge-spatial.git
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

### Wheel dependency profiles

The base wheel installs the stable array, table, AnnData, configuration, and
metric dependencies.  Heavy scientific stacks are explicit extras:

```bash
# Raw-count preprocessing and interaction-graph construction
pip install 'CytoBridge[preprocess]'

# Dynamical-model training
pip install 'CytoBridge[train]'

# Complete spatial preprocessing/training/graph workflow
pip install 'CytoBridge[spatial]'

# Plotting, velocity/terminal-state, or the executable dataset notebooks
pip install 'CytoBridge[plot]'
pip install 'CytoBridge[velocity]'
pip install 'CytoBridge[notebook]'

# Every supported optional feature
pip install 'CytoBridge[all]'
```

The `graph` extra installs the optional PyTorch Geometric interaction models.
Kaleido is confined to the `plot`, `velocity`, and `all` profiles for the
repository's static Plotly export path. TorchVision, TorchAudio, TorchSDE, and
ImageIO are no longer installed because neither the package nor its current
workflow scripts import them. GPU-specific Torch wheels remain an environment
decision; choose the CPU or CUDA index appropriate for the target system before
installing a Torch-dependent extra.

`requirements.txt` remains a compatibility entry point for development
checkouts and installs the same base plus `all` union.  The individual
dependency contracts live in `requirements/*.txt` and use bounded major/minor
ranges; they are not CPU/CUDA environment locks.

### Additional optional dependency

Some plotting utilities rely on packages that are not required for the core
preprocessing and training workflow.

- `cellrank` is included by the `velocity` and `all` extras and is only needed
  for terminal-state analysis utilities.

If needed:

```bash
pip install 'CytoBridge[velocity]'
```

### Verify an installed package

The installed command reports the package version without importing the
scientific or plotting stacks:

```bash
cytobridge --version
cytobridge doctor
cytobridge doctor --json
```

`doctor` is read-only. It reports Python, package metadata, and whether common
dependency modules are available; it does not import those dependencies or
modify caches, data, or configuration.

### Package-native workflow command

The installed wheel includes readable presets for Zebrafish, MOSTA, ARISTA,
AD mouse, and developing chicken heart. Install `CytoBridge[all]` for the complete model, graph, and
figure stack, then inspect a complete plan before supplying large inputs:

```bash
cytobridge workflow --list-configs
cytobridge workflow --config zebrafish --dry-run
cytobridge workflow --config admouse --dry-run --json
```

The plan prints the dataset policy, scientific parameters, steps, compute
requirements, and any missing input. The primary settings are seed 42,
`alpha_spatial=10`, and `alpha_express=0.015`; generated-cell annotations use
`k=10` for Zebrafish, MOSTA, and ARISTA, and `k=1` for AD mouse and chicken
heart. The packaged
training profiles retain the fixed interaction cutoffs. All five main models
use learned edge priors. AD's targeted panel represents only seven complete
ligand-receptor pairs under strict all-subunit matching; the corrected main
model deliberately uses those labels and its validation-selected predictor
threshold rather than silently changing the model class.

| preset | interaction cutoff | main edge prior | validation-selected full predictor threshold |
|---|---:|---|---:|
| MOSTA | 0.02400244047956264 | learned predictor | 0.1192110925912857 |
| ARISTA | 0.03154105148551745 | learned predictor | 0.5884028673171997 |
| Zebrafish | 0.09606367405591873 | learned predictor | 0.6063615679740906 |
| AD mouse | 0.012106042891492197 | learned predictor | 0.9956824779510498 |
| Chicken heart | 0.21681429373719752 | learned predictor | 0.14988678693771362 |

The formal matched matrix now contains 12 completed training and package
downstream runs: full learned prior, no-LR-prior (`all_spatial`), and
no-interaction for each of those four datasets. All 12 profiles and all four
three-arm families pass acceptance SHA-256
`c4f8e203e2da73fe78e28525516bbec192d3cbbd35d423dcd64080a0f83a10df`.
No-interaction retains velocity, growth, and score; communication and LR are
`NA` by construction. The formal matched reconstruction comparison is complete:
mean paired relative sliced-W2 changes for no-LR versus full are +25.46% (AD),
+59.44% (ARISTA), +26.13% (MOSTA), and +13.53% (Zebrafish); no-interaction
changes are -0.04%, -6.02%, -28.35%, and -10.16%, respectively. These are
full-data in-sample reconstruction comparisons, not LOTO or significance
tests; the interaction effect is dataset-dependent, and no uniform full-model
superiority is claimed. The five-application cross-method benchmark is also
complete: all 110 LOTO executions completed, CytoBridge has the lowest spatial
sliced-W2 for 8/11 held-out targets, and a linear control wins 17/22
joint/state comparisons. Seven stVCR full-data targets remain explicit `NA`
after method-native numerical failures (four ARISTA and three chicken heart).
The formal Zebrafish paper
downstream completed all seven signed stages. The paper-specific S22 panel is a
single generated global-t0 rollout from `t=0` through `t=4`; observed integer
slices are exported separately as references and never replace generated
frames. S25 and communication retain their explicitly separate interval-local,
observed-anchored state contract.

Developing chicken heart is the fifth package-native application. Its reviewed
D4/D7/D10/D14 alignment, full learned-prior fit, standard downstream,
continuous D4-to-D14 perturbation/LR analysis, and corrected velocity figures
were rerun through the current package. It is a completed single full-model
application, not a fifth family in the accepted four-dataset three-arm
ablation matrix. Its separate 10-method LOTO/full-data benchmark is reported
under signed evaluation manifests. All 20 chicken-heart LOTO executions
completed; CytoBridge wins D7 joint/spatial, and the spatial winners across
D7/D10 are CytoBridge and MOSCOT. Its full-data diagnostic is explicitly
in-sample: CytoBridge wins aggregate spatial sliced-W2, while random
interpolation wins aggregate joint/state sliced-W2.

Preprocessing and downstream inference can be selected independently. These
commands call the public package APIs and do not depend on repository scripts:

```bash
# Expression preprocessing plus spatial alignment
cytobridge workflow --config zebrafish --step preprocess \
  --input-h5ad /data/zebrafish.h5ad \
  --output-dir /runs/zebrafish \
  --device cuda

# Complete shared downstream chain from an existing model
cytobridge workflow --config zebrafish --step downstream \
  --aligned-h5ad /runs/zebrafish/preprocess/zebrafish_aligned.h5ad \
  --model-dir /runs/zebrafish/training \
  --output-dir /runs/zebrafish \
  --device cuda
```

Current predictor-gated CytoBridge checkpoints contain their learned
edge-predictor weights, so copied model directories do not depend on the
original training-machine path. For an older predictor-gated current-format
checkpoint that lacks those embedded weights, add
`--edge-predictor-path /path/to/edge_model.pt` when running downstream.

The downstream step now executes the common quantitative chain, rather than
stopping after interpolation. It writes generated H5AD slices, classifier
scores, observed-slice velocity components, per-cell growth, cell-type
composition, model-attention summaries evaluated on the downstream reporting
radius graph, and readable CSV tables. It also
renders the shared white-background snapshot/mosaic, velocity, growth,
composition, and 3D communication figures. Lineage is emitted only when the
dataset config explicitly declares a persistent fixed-particle identity
contract; the current generic presets conservatively omit it. The package
workflow does not apply spatial warping, and attention, growth, gene,
LR, and reconstruction diagnostics all use the unwarped model states.
For AD, downstream attention is recomputed on a full time-slice radius graph,
or on the explicitly configured seeded time-slice subsample. That reporting
graph is distinct from the stochastic interaction groups used by training and
dynamics. The main AD checkpoint applies its learned edge gate; because that
gate is supported by only seven strict panel-covered pairs, its attention
summaries must not be described as a global cell-cell communication screen.
For every dataset, `summary.json` records downstream radius candidates and
retained edges under `analyses.communication.edge_selection_by_time`. A valid
empty time point is emitted as canonical empty sparse arrays and an explicit
structural-zero status, rather than being hidden or filled with fallback edges;
the accompanying interpretation states that this is not evidence for absence
of all biological communication.

The packaged downstream profiles also keep the formal simulation scope:
Zebrafish uses every observed t0 cell on its nine analysis slices; MOSTA uses
12,000 starting particles on 13 slices; ARISTA uses 7,668 particles on nine
slices; and AD mouse uses all 53,615 observed t0 cells on the 26-point grid
from model time 0 to 2.5. These are production analyses, not compact examples.

Temporal gene reconstruction and strict ligand-receptor projection run by
default in all four packaged downstream presets. Package preprocessing retains
the fitted PCA loadings in `varm['PCs']`, the fitted center in
`var['pca_center']`, and the matching gene order, so the aligned H5AD produced
earlier in the same command is the default reference. For Zebrafish, MOSTA, and
ARISTA, LR projection reuses the bundled species-matched CellChatDB used for
the learned edge prior. AD follows the same code path: its 347-gene panel
retains seven complete pairs under strict all-subunit matching, and those
database-derived labels train the main edge predictor. Both the model prior and
downstream projection are therefore panel-limited rather than global CCI.
Complexes
require every subunit and use minimum subunit expression. The
CLI options remain available for an explicit reference, database, species tag,
or geometric-mean sensitivity override. Historical references missing
`var['pca_center']` fail closed unless
`--allow-complete-reference-pca-center-fallback` explicitly declares that the
file is the complete original PCA-fit population; its mean must still
reproduce saved PCA coordinates:

```bash
cytobridge workflow --config zebrafish --step downstream \
  --aligned-h5ad /runs/zebrafish/preprocess/zebrafish_aligned.h5ad \
  --model-dir /runs/zebrafish/training \
  --output-dir /runs/zebrafish \
  --lr-database /data/ligand_receptor.csv \
  --lr-complex-mode min \
  --device cuda
```

`--reconstruction-diagnostic` adds W2 comparisons between the fitted model's
generated population and later observed slices. The output is deliberately
named a fitted-model reconstruction diagnostic: it is not a training holdout
and not a cross-method benchmark. Use the matched benchmark pipeline for those
claims.

Training never runs implicitly. For the four generic-alignment datasets, adding
`--train` to the raw-data workflow builds per-timepoint LR graphs, trains an
edge predictor, and passes its validation-selected threshold into model
training. `--edge-predictor-threshold` remains an explicit override for
predictor-gated runs:

```bash
cytobridge workflow --config mosta --train \
  --input-h5ad /data/mosta.h5ad \
  --output-dir /runs/mosta \
  --device cuda
```

Each preset uses its species-matched formal CellChatDB resource bundled in the
wheel for downstream strict LR projection: zebrafish for Zebrafish, mouse for
MOSTA and AD mouse, and human for ARISTA. Chicken heart uses the human table as
an explicitly labeled conserved-symbol proxy because no Gallus gallus CellChatDB
release is bundled. All five use the declared resource to fit
their learned edge prior. Pass `--graph-database /path/to/database.csv` to override graph construction
in a predictor-gated workflow, or
`--lr-database /path/to/database.csv` to override only the downstream LR
projection. An existing edge predictor may be supplied only with the aligned
H5AD and main model that were fitted in the same feature space. A raw-H5AD
predictor-gated `--train` run always fits a new edge predictor; it rejects
`--edge-predictor-path` rather than mixing an old predictor with a newly fitted
PCA basis.
The bundled tables come from the GPL-3.0
[CellChat project](https://github.com/jinworks/CellChat); cite CellChatDB when
reporting results derived from them.

Use `--training-config /path/to/config.yaml` to replace a packaged training
preset. The AD mouse preset now includes the resolved six-stage
`100/100/50/3001/1000/3001` profile. Its default is now a corrected de novo
raw-H5AD workflow: `obs['Timepoint']` values 1/2/3 map to model times 0/1/2,
`layers['counts']` is normalized to 10,000 and log1p transformed, all three
batches are aligned, and a learned edge predictor is fitted from the seven
strict complete panel-supported pairs. The corrected main run selected
`0.9956824779510498`, recorded identically in its resolved training config and
edge metadata. Legacy 0.015 artifacts that used threshold
`0.32999998331069946` remain a compatibility path and are not evidence for the
accepted matched family. The accepted `all_spatial` alternative is the
explicit no-LR-prior ablation:

```bash
cytobridge workflow --config admouse --step downstream \
  --training-config admouse_spatial_full_alpha_express_0015_no_lr_prior.yaml \
  --aligned-h5ad /runs/admouse-no-lr/preprocess/admouse_aligned.h5ad \
  --model-dir /runs/admouse-no-lr/training \
  --output-dir /results/admouse-no-lr-reuse \
  --device cuda
```

Current-format checkpoints embed the predictor weights. An older
predictor-gated checkpoint without embedded weights additionally needs its
matched `--edge-predictor-path`. Never load a radius-only ablation with the
main learned-predictor training contract, or vice versa.

### Run the tutorials

The five dataset notebooks are package-facing walkthroughs for Zebrafish,
MOSTA, ARISTA, AD mouse, and developing chicken heart. They read the wheel-bundled presets, keep training
explicit, and cover interpolation/classification, time-slice velocity, growth,
sparse spatial attention, strict ligand-receptor analysis, and pre-warp evaluation:

```bash
python -m pip install -e '.[all]'
jupyter lab notebooks/01_zebrafish.ipynb
```

That path is a source-checkout command. Wheel users can download each notebook
from the ReadTheDocs tutorial page or GitHub, then open the downloaded file;
the `notebook` and `all` extras install its runtime but do not copy notebooks
into the current directory.

Zebrafish, MOSTA, and ARISTA use the formal spatial-domain label setting
`k=10`; AD and chicken heart use `k=1`. The notebooks are committed without outputs. Their
release smoke uses synthetic data and verifies package wiring; the formal
scientific results come from the corresponding full-data runs.

For a focused source-checkout preprocessing tutorial, install only the
preprocessing dependencies in the environment that provides your Jupyter
frontend, then open the checked-in tutorial:

```bash
python -m pip install -e '.[preprocess]'
jupyter lab notebooks/01_synthetic_preprocessing_contract.ipynb
```

The notebook creates its own deterministic count matrix and records the
normalization, log transformation, time mapping, latent-feature, and PCA
contracts. It intentionally contains no manuscript dataset, result, benchmark,
or machine-specific path.

Release maintainers can build and test an installed wheel in a new private
workspace with:

```bash
python scripts/smoke_installed_wheel.py --work-dir /path/to/new-empty-workspace
```

The workspace path must not already exist. The script builds the current source
tree, installs the wheel without dependencies into a clean virtual environment,
and runs the installed-package contract tests. Pip index access is disabled, so
missing local build tools fail clearly instead of being downloaded implicitly.

### Non-spatial Weinreb and scNT workflows

The installed CLI also owns the two expression-state workflows that support the
accepted Weinreb lineage and scNT cortical figures:

```bash
cytobridge nonspatial list-presets
cytobridge nonspatial plan --dataset weinreb
cytobridge nonspatial plan --dataset scnt_cortex
```

Their explicit sequence is preprocessing, LR edge-prior construction, matched
Full/No-interaction training, weighted W1/W2/TMV, clone-fate or new-RNA
direction evaluation, exact interaction attribution, and A4 figure generation.
They use expression PCs only—no physical spatial coordinates—and simulate
continuously from the earliest observed time rather than restarting from
intermediate slices. See
[`docs/nonspatial_workflows.md`](docs/nonspatial_workflows.md) for complete
commands and the distinction between exact historical figure replay and new
corrected matched training.

## Input Requirements

The main preprocessing pipeline expects an input `.h5ad` file with:

- spatial coordinates in `adata.obsm["spatial"]` or `adata.obs["spatial_x"]`, `adata.obs["spatial_y"]`
- a time annotation column in `adata.obs[time_key]`
- unique `adata.obs_names`, or preset-declared observation columns that form a
  stable composite identity. The ARISTA preset uses `Batch` plus `CellID`, and
  AD mouse uses `sample` plus `cell_id`; neither invents row-order suffixes for
  repeated local IDs.

For interaction graph construction, the repository script also expects a
ligand-receptor database CSV, by default:

```text
database/CellNEST_database.csv
```

If your database lives elsewhere, pass it explicitly with `--database-path`.
The installed `cytobridge workflow` command instead selects the preset's
species-matched bundled formal database automatically as described above.

## Quick Start

### 1. Run preprocessing, spatial alignment, interaction graph construction, and edge predictor training

```bash
python scripts/preprocess_pipeline.py \
  --data-name mosta \
  --h5ad-path /path/to/input.h5ad \
  --time-key timepoint \
  --output-dir ./results/mosta_preprocess \
  --database-path /path/to/database/CellChatDB.ligrec.mouse.csv \
  --batch-indices 3,4,5,6 \
  --time-mapping '{"E9.5":-3,"E10.5":-2,"E11.5":-1,"E12.5":0,"E13.5":1,"E14.5":2,"E15.5":3,"E16.5":4}' \
  --expression-layer count \
  --counts-layer count \
  --raw-count-validation strict \
  --normalization-target-sum 10000 \
  --auto-scale-from-centered-x-max 0 \
  --center-x 1 --center-y 0 \
  --scale-x 0.01 --scale-y 0.01 --flip-y 1 \
  --device cuda
```

For the published MOSTA schema, prefer the audited dataset adapter. It also
forces mouse CellChat ligand/receptor subunits that are present in the source
data into the PCA feature mask, writes PCA/loadings/center contracts, trains a
fresh edge predictor, and records the canonical E12.5--E15.5 model-time axis:

```bash
CUDA_VISIBLE_DEVICES=7 python scripts/run_mosta_end_to_end.py \
  --h5ad-path /path/to/Mouse_embryo_all_stage.h5ad \
  --database-path /path/to/CellChatDB.ligrec.mouse.csv \
  --output-dir ./results/mosta_corrected_counts_alpha0015 \
  --profile full \
  --stage all \
  --alpha-spatial 10 \
  --alpha-express 0.015 \
  --device cuda
```

`Mouse_embryo_all_stage.h5ad.X` is already transformed. Its authoritative raw
UMI matrix is the singular `layers['count']`; omitting the explicit layer would
reproduce the historical double transformation. The corrected adapter uses
`raw count -> normalize_total(target_sum=10000) -> log1p` exactly once.
`--alpha-spatial` and `--alpha-express` are explicit run parameters and are
recorded in both the training and evaluation manifests. For a paired
expression-weight sensitivity run, keep every other argument and the random
seed fixed, change only `--alpha-express 0.05`, and use a fresh output
directory. Reuse the audited preprocessing without copying or recomputing it:

```bash
CUDA_VISIBLE_DEVICES=7 python scripts/run_mosta_end_to_end.py \
  --h5ad-path /path/to/Mouse_embryo_all_stage.h5ad \
  --database-path /path/to/CellChatDB.ligrec.mouse.csv \
  --output-dir ./results/mosta_corrected_counts_alpha005 \
  --reuse-preprocess-dir ./results/mosta_corrected_counts_alpha0015/preprocess \
  --profile full \
  --stage train-evaluate \
  --alpha-spatial 10 \
  --alpha-express 0.05 \
  --random-seed 42 \
  --device cuda
```

This mode treats the shared aligned H5AD, PCA contract, graph, and edge
predictor as read-only inputs and records their path/identity in the new run
manifest.

For a paired expression-weight sensitivity run, the repository also provides a
small launcher that keeps the baseline and candidate outputs in separate
directories:

```bash
scripts/launch_mosta_paired_alpha.sh \
  7 0.05 \
  ./results/mosta_corrected_counts_alpha0015 \
  ./results/mosta_corrected_counts_alpha005
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
  --train_config CytoBridge/configs/mosta_spatial_full_alpha_express_0015.yaml \
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

Each new training run writes `training_history.csv` and
`training_run_summary.json`. The history distinguishes the lowest-loss epoch
from the checkpoint actually selected by each stage and records learning-rate
endpoints, optimizer steps, and synchronized elapsed time. The summary records
the device/software contract, parameter and input dimensions, wall time,
process peak RSS, and native PyTorch CUDA peak allocated/reserved memory; an
unavailable measurement remains null instead of being inferred. Run
`scripts/summarize_training_history.py` to generate reviewer-facing curves and
`training_resource_summary.json`. Older schema-v1 histories can still be read,
but values absent from those files are explicitly marked as inferred or
unavailable.

## Package API

The main package-level entry points are:

- `CytoBridge.pp.preprocess`
- `CytoBridge.pp.align_spatial`
- `CytoBridge.pp.generate_interaction_graph`
- `CytoBridge.pp.train_edge_predictor`
- `CytoBridge.tl.fit`
- `CytoBridge.tl.evaluate_model_distributions`
- `CytoBridge.tl.train_cached_mlp_classifier_from_adata`
- `CytoBridge.tl.smooth_spatial_labels`
- `CytoBridge.tl.analyze_spatial_label_sensitivity`
- `CytoBridge.tl.run_interpolation_workflow`
- `CytoBridge.tl.compute_timepoint_communications`
- `CytoBridge.tl.load_gmt_gene_sets`
- `CytoBridge.tl.overrepresentation_analysis`
- `CytoBridge.tl.plot_lineage_sankey`
- `CytoBridge.tl.plot_spatiotemporal_3d`
- `CytoBridge.pl.plot_enrichment_bar`
- `CytoBridge.pl.plot_enrichment_dot`

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
(two aligned spatial coordinates followed by 50 expression PCs). The checked-in
`notebooks/03_arista.ipynb` provides the package-facing executable tutorial.

## Canonical ARISTA reproduction

ARISTA uses the same packaged workflow as the other datasets. The preset
contains the dataset schema and formal scientific settings; the user supplies
the full H5AD and an output directory.

### Full H5AD-to-model run

```bash
CUDA_VISIBLE_DEVICES=0 cytobridge workflow \
  --config arista \
  --train \
  --input-h5ad /path/to/Regeneration.h5ad \
  --output-dir /path/to/runs/arista-corrected \
  --device cuda
```

The preset selects the species-matched LR database bundled with the package.
Use `--graph-database` only to supply an intentional replacement database.
Preprocessing trains a new edge predictor from this dataset's aligned cells and
LR graph; it is not a model that the user must provide.

The full profile uses the recovered CytoBridge six-stage spatial schedule in
`CytoBridge/configs/arista_spatial_full.yaml`: 100 pretraining epochs, 100
refinement epochs, 50 interaction-initialization epochs, 2,001 score-matching
epochs, 1,000 finetuning epochs, and a final 2,001-epoch score refinement. It
uses the unified production weighted-OT defaults (`alpha_spatial=10`,
`alpha_express=0.015`) while preserving the recovered stage-specific mass
conventions, forward-OT checkpoint selection for intermediate stages, and the
finetuning plateau scheduler.
Finetuning also restores the historical checkpoint rule: the best forward
last-interval OT state is captured before reverse-time updates, rather than
unconditionally using epoch 1,000's final weights.
`CytoBridge/configs/arista_spatial_compact_full.yaml` retains the earlier
three-stage `500/2001/1000` profile for explicitly labeled method/threshold
ablations; it is not the six-stage package profile. The released ARISTA
checkpoint's `params.yml` records the older top-level `500/2001/1000` fields,
but the original interaction-stage training driver and execution log are not
part of the released assets. Therefore exact historical retraining should not
be claimed from metadata alone; the formal comparison treats the saved model as
the reference and the recovered package schedule as a separately labeled run.
The complete ARISTA source H5AD stores an already log-transformed matrix in `X`
and integer raw values in `layers['counts']`. The canonical runner therefore
copies `layers['counts']` into `X` before median-library-size normalization and
`log1p`, preventing the historical double transformation. This source choice
is recorded in `uns['preprocess_info']` and the run summary. A custom workflow
may explicitly set `expression_layer: null` and
`allow_retransform_preprocessed_x: true` for a labelled legacy replay; the
packaged preset never does so, and an already transformed `X` otherwise fails
fast instead of being silently transformed again. The package preset uses the
complete 16,379-gene `Regeneration.h5ad`, selects HVGs across all eight batches,
retains matched LR subunits in the PCA mask, and aligns the five named
2/5/10/15/20-DPI batches. This yields 46,209 cells. The historical prepared
2,000-gene file contains 46,189 of them, but its extra 20-cell spatial crop is
not recoverable as a complete executable rule; the corrected pipeline does not
guess it. The alignment recipe uses median-library-size normalization
(`target_sum=None` in Scanpy), 2,000 base HVGs plus required LR features, 50
PCs, two aligned spatial coordinates, and seed 42.
The packaged workflow sets `evaluate_after_training=False` because the trainer's
historical in-memory evaluation is redundant and can retain a large autograd
graph. Standard downstream analyses run from the saved checkpoint. Add
`--reconstruction-diagnostic` only for the fitted-model reconstruction table;
the matched held-out/cross-method benchmark is a separate evaluation pipeline.

The seed is applied before model construction as well as before sampling.
For stages that explicitly use `save_strategy: last`, the saved state follows
exactly the declared epoch count and does not run an undocumented extra
optimizer epoch. Non-finite ODE states,
losses, and gradients fail loudly. The ARISTA finetuning profile uses gradient
clipping (`max_grad_norm=10`) as an explicit numerical safeguard; this is the
only intentional stabilization relative to the released script and is recorded
in the resolved configuration.

For parity with the released ARISTA GNN, radial-basis centers and widths are
fixed (`model.interaction_net.rbf_trainable: false`). Set this option to `true`
only for an explicit trainable-RBF ablation; doing so changes the interaction
model and is not a reproduction of the released checkpoint semantics.

For downstream loading, score-matching stages are inferred from the resolved
training plan and searched in reverse execution order. Thus the six-stage
ARISTA profile loads `Score_Refine/score_model.pth`, while configurations ending
in `Train_Score_Final` load that final stage. The selected `score_stage` is
recorded in every downstream manifest.

The run writes:

- `preprocess/arista_aligned.h5ad` and `arista_aligned.csv`
- per-timepoint graph inputs and edge-predictor metadata
- staged checkpoints under `training/`
- generated and observed slice H5ADs under `downstream/slice_data/`
- fresh per-time velocity, growth, composition, and sparse communication tables
- temporal gene and strict complete-subunit LR tables using the same PCA reference
- standard snapshot, velocity, growth, composition, and communication figures
- `downstream/summary.json`, including the loaded weight/score stages and the
  scientific checkpoint-contract comparison

The tutorial notebooks use a clearly labelled compact scope for wiring checks.
Compact outputs are not substitutes for this full-data command because
subsampling changes spatial neighborhoods and edge-predictor training.

### Auditing a released model-input CSV

Some historical runs begin from an already aligned/reduced table such as
`arista_1108_with_annotation.csv` rather than gene-level AnnData. Import that
table explicitly as a legacy model state:

```bash
python scripts/convert_legacy_model_input_csv.py \
  --input-csv /path/to/arista_1108_with_annotation.csv \
  --output-h5ad /path/to/arista_legacy_model_input.h5ad \
  --interaction-cutoff 0.05 \
  --edge-predictor-threshold 0.45 \
  --edge-predictor-path /path/to/edge_classifier/arista.pt
```

The generic `legacy_model_input_csv_to_adata` API maps `x1,x2` to
`obsm['spatial_aligned']`, the remaining numeric columns to
`obsm['X_latent']`, and preserves numeric time values and optional annotation.
The source path, column order, and interaction settings are written into
AnnData provenance. Because the CSV contains no gene-level expression, this is
an audit/migration input—not a substitute for the full H5AD preprocessing
workflow. It is useful for separating training-code changes from preprocessing
and coordinate changes in a controlled comparison.

### Threshold provenance

The packaged ARISTA preset fixes the formal spatial cutoff at
`0.03154105148551745`. The newly trained edge predictor selects its decision
threshold on validation data and passes that value directly to the six-stage
model. Both effective values and their sources are stored in the aligned H5AD,
edge metadata, and resolved training configuration, so the three steps cannot
silently diverge. `--interaction-cutoff` and
`--edge-predictor-threshold` are explicit sensitivity overrides; they are not
required for a standard run.

Because the neighborhood policy also changes the graph used to train the edge
classifier, two independently preprocessed policy runs may have different edge
weights even when their aligned coordinates and PCA values match. To isolate
fit-time thresholds, use a matched aligned H5AD and edge predictor together:

```bash
cytobridge workflow --config arista --train --step downstream \
  --aligned-h5ad /path/to/frozen/preprocess/arista_aligned.h5ad \
  --output-dir /path/to/frozen-control \
  --interaction-cutoff 0.05 \
  --edge-predictor-threshold 0.45 \
  --edge-predictor-path /path/to/frozen/preprocess/edge_classifier/arista_edge_model.pt
```

The resolved `training/config.yaml` records the effective values and edge-model
path. A paired control should reuse the same aligned H5AD and edge-model files
before changing the thresholds.

Raw cutoff values should be interpreted together with coordinate scale. For
cross-run comparisons, report the coordinate range/standard deviation, median
nearest-neighbor distance, and `cutoff / median_nn`; do not attribute a cutoff
difference to biology until the aligned-coordinate scales have been checked.

In the full ARISTA audit, the historical and current aligned coordinates have
nearly identical scale (median nearest-neighbor distance `0.003160` versus
`0.003142`; coordinate correlations `0.9977` and `0.9900`). The historical
`0.05` cutoff is therefore not explained by a coordinate rescaling: it spans
about `15.8` historical median-neighbor distances, whereas the full-data
preprocessing cutoff `0.031543` spans about `10.0`. The intermediate
clean-counts audit selected an edge threshold of `0.35`, compared with the
historical `0.45`. The corrected de novo alpha-0.015 package run selected
`0.5884028673171997`; that validation-selected value is the current shipped
default. The former `0.23999999463558197` package value is retained only as
historical matched-run provenance.

The formal full-data comparison uses mean metrics across the five observed
times. The strict threshold control reuses the exact same aligned H5AD and edge
weights as the current-auto run, changing only fit-time thresholds:

| condition | spatial W1 | spatial W2 | TMV | NN dispersion ratio | clump fraction |
|---|---:|---:|---:|---:|---:|
| published saved model | 0.07377 | 0.08815 | 0.02321 | 0.9449 | 0.0110 |
| recovered six-stage, legacy input | 0.06308 | 0.07515 | 0.01366 | 0.7981 | 0.0261 |
| recovered six-stage, current preprocess + auto thresholds | 0.06648 | 0.08182 | 0.00624 | 0.8966 | 0.0127 |
| same current input/edge model, fit with `0.05/0.45` | 0.06949 | 0.08095 | 0.00652 | 0.3128 | 0.0976 |

The strict control illustrates why W1/W2/TMV are insufficient on their own:
its aggregate distances look competitive while the generated spatial clouds
collapse into repeated local clumps. The current-preprocess/auto-threshold run
is therefore the selected retrained model; the published checkpoint remains a
separate saved-model reference.

`compare_distribution_metric_tables` and
`save_distribution_metric_comparison` provide a dataset-agnostic paired
comparison of W1, W2, TMV, and local-structure tables and figures. Candidate
deltas are defined as `candidate - baseline`. Lower W1/W2/TMV and clump
fraction are better; the nearest-neighbor dispersion ratio should be near one,
and support recall/precision should be high.

Additional shared ARISTA panel APIs include
`summarize_growth_interaction_by_celltype`,
`plot_growth_interaction_bubble`, and
`plot_spatial_component_direction_correlation_roi_from_adata`. The shared
`summarize_label_composition` and `plot_celltype_composition` APIs export the
interpolated cell-type composition table and stacked fraction panel used for
S14b; `evaluate_growth_by_timepoint` and `plot_growth_timepoint_grid` produce
the S13 per-cell table and dense observed/generated spatial grid. Their inputs are
generic AnnData keys and model components; the ARISTA dataset tutorial contains
only the timepoint, annotation, and reaEGC ROI choices.

For manuscript-style layouts, `plot_trajectory_grid` can wrap time points into
multiple rows with one shared legend, while `plot_trajectory_comparison_grid`
renders any matched control/perturbation trajectories as time-by-condition
panels. `plot_growth_timepoint_grid` supports per-time robust 0--1 display
scaling or one global raw-value scale, and `plot_temporal_gene_heatmap` can split
one globally ordered gene list into contiguous columns with a shared colorbar.
All layout options are dataset-agnostic and leave the underlying model states
and numerical tables unchanged.

Temporal gene and ligand-receptor panels use the same separation of concerns:

- `summarize_temporal_gene_patterns` uses either the PCA contract retained in
  processed AnnData or an explicit `PCAReconstructionSpec`, inverse-projects
  simulated PCA states, and clusters the resulting gene trajectories. Its
  optional `candidate_features` argument freezes the exact original PCA-feature
  universe eligible for temporal-variance ranking; missing, duplicate, or
  center-only candidates are strict errors, and the requested/used sets are
  recorded in the result settings. Its default `clip_min=0.0` clips every
  reconstructed cell before taking the time-point mean, matching a
  non-negative log1p expression scale. `clip_min=None` remains available only
  for a separately labelled legacy signed diagnostic;
- `evaluate_pca_anchor_reconstruction` checks observed log1p anchors against
  their exact-center inverse-PCA reconstruction in bounded cell chunks. It
  expects a caller-supplied view restricted to the intended biological cells
  and active PCA candidates (for MOSTA, Brain cells and the original 2,000
  HVGs), rejects center-only features by default, and strictly validates
  feature order. It returns per-time aggregate errors plus per-feature
  RMSE/MAE, mean, population standard deviation, bias, correlation, and scale
  ratio together with the effective observation, feature, and PCA contracts,
  without constructing the full cells-by-features reconstruction;
- `project_communication_to_lr_timecourses` combines those reconstructed
  expression values with per-timepoint `M_per_source` communication matrices.
  Hybrid observed/generated runs use one all-times universe of LR subunits with
  active retained-PC loadings, require every retained pair and pair-by-cell-type
  trajectory to cover the full requested time grid, and return explicit
  `trajectory_coverage` and `dropped_trajectories` audits instead of silently
  zero-filling missing times. Pair identity is the structured
  `(ligand, receptor)` tuple and a reversible JSON `pair_id`; the historical
  underscore-joined `pair` is display-only because complex names can collide.
  The formal rule is `complex_mode="min"` with every subunit required;
  `complex_mode="geometric_mean"` is a separately labelled sensitivity.
  Unsupported cell types stay unavailable/NaN,
  never an invented zero;
- `compute_focal_lr_type_hotspots` implements the article-style focal-panel
  estimand `mean_sender(ligand) * mean_receiver(receptor_complex) *
  M_per_source(sender, receiver)`. It exports the sender-by-receiver type
  matrix, type-level incoming/outgoing/total scores, a cell mapping with one
  identical value per time/type, and a formula/subunit/cohort audit. A capped
  compute cohort and full display cohort may be supplied separately, with both
  cohort definitions and sizes recorded;
- `load_pca_reconstruction_spec` loads historical loading/center tables through
  a generic, validated feature-alignment contract rather than dataset-local
  parsing;
- `plot_temporal_gene_heatmap`, `plot_temporal_pattern_prototypes`, and
  `plot_temporal_profile_small_multiples` render dataset-agnostic S15-S17-style
  panels;
- `load_gmt_gene_sets` and `overrepresentation_analysis` provide an offline,
  dataset-independent gene-set contract with an explicit expression or
  library-wide background, while `plot_enrichment_bar` and
  `plot_enrichment_dot` render the standardized result table.

Two additional public APIs cover the numerical steps that were previously
embedded in figure notebooks. `analyze_developmental_wave` takes any
feature-by-time table, selects rows by temporal variance, performs row-wise
standardization and deterministic peak-time ordering, and uses exact dynamic
programming to divide the ordered cascade into contiguous phases subject to a
hard minimum phase size. `plot_developmental_wave_heatmap` displays that stored
ordering and those phase boundaries without recomputing them.

`cluster_temporal_profiles` uses an exact merge-count tree cut, so duplicate or
zero-distance profiles still yield the requested number of clusters (bounded by
the number of input profiles). Diagnostics record the chosen cluster count,
cut strategy, and number of zero-distance merges.

```python
from CytoBridge.tl import analyze_developmental_wave
from CytoBridge.pl import plot_developmental_wave_heatmap

wave = analyze_developmental_wave(
    gene_by_time,
    n_top_profiles=250,
    n_phases=3,
    min_phase_size=5,
)
wave.assignments.to_csv("wave_assignments.csv", index=False)
wave.prototypes.to_csv("wave_phase_prototypes.csv", index=False)
plot_developmental_wave_heatmap(wave, out_path="developmental_wave.pdf")
```

`project_velocity_to_embedding` provides a dependency-light, auditable
alternative to constructing a temporary scVelo object. It builds (or accepts)
a k-nearest-neighbor graph in the complete latent feature space, converts the
cosines between latent velocities and latent neighbor displacements to
softmax transition probabilities, and applies the centered probabilities to
2D target-embedding displacements. The embedding is only the projection
target; it is never substituted for the high-dimensional neighborhood state.

```python
from CytoBridge.tl import project_velocity_to_embedding

projection = project_velocity_to_embedding(
    latent_coordinates,
    latent_velocity,
    embedding_2d,
    n_neighbors=30,
    temperature=1.0,
)
velocity_2d = projection.projected_velocity
```

Virtual perturbations are also exposed as dataset-independent APIs.
`run_virtual_cell_type_ablation` takes caller-defined annotation keys and label
sets; it never embeds MOSTA label names.  Its default uses one shared initial
cohort, while `mass_control=True` independently samples the baseline and one
post-removal pool without replacement to the same initial size.  The latter is
the audited contract used for the historical MOSTA equal-mass experiments.
`run_virtual_interaction_ablation` instead accepts one explicit `x0` array and
runs an exact-input, common-seed interaction-on versus interaction-off pair.
Both workflows save numerical trajectories, comparison metrics, cohort or
input provenance, and a machine-readable manifest; plotting remains a separate
package call.

```python
from CytoBridge.tl import (
    run_virtual_cell_type_ablation,
    run_virtual_interaction_ablation,
)

celltype_result = run_virtual_cell_type_ablation(
    adata,
    runtime,
    ablations={"remove_target": ["Target cell type"]},
    time_points=time_grid,
    annotation_key="cell_type",
    time_key="model_time",
    mass_control=True,
    n_samples=40_000,
    output_dir="results/remove_target",
)

interaction_result = run_virtual_interaction_ablation(
    initial_states,
    runtime,
    time_points=time_grid,
    output_dir="results/interaction_off",
)
```

The enrichment API does not download or silently select a database. Callers
provide a versioned GMT file and record its name and version in the run manifest:

```python
from CytoBridge.tl import load_gmt_gene_sets, overrepresentation_analysis
from CytoBridge.pl import plot_enrichment_dot

library = load_gmt_gene_sets("GO_Biological_Process_2023.gmt")
result = overrepresentation_analysis(
    pattern_genes,
    library,
    background_genes=all_measured_genes,  # omit for a library-wide background
)
plot_enrichment_dot(
    result.loc[result["significant"]],
    out_path="pattern_go.svg",
)
```

The backward-compatible multiple-testing scope is `"reported"`, where terms
below `min_overlap` are removed before BH correction. For a preregistered
library-wide family, use `multiple_testing_scope="all_eligible"`: every term
passing the explicit set-size gates is retained in one BH family, including
zero-overlap terms with p-value 1. The result records both the eligible and
actually corrected test counts in columns and `DataFrame.attrs`.

The default, prospective workflow requires a package-processed reference H5AD
that retains PCA loadings. An older H5AD containing only `X_pca` coordinates is
not sufficient by itself. For a declared historical reproduction, callers may
instead load the archived PCA components and center with
`load_pca_reconstruction_spec`; record both source paths and the ordered feature
contract in the run manifest. `preferred_species_tag` makes cross-species
symbol choice explicit, while `profile_linkage_method` and
`profile_cluster_order` expose the clustering contract. For ARISTA, the paper's
68 LR pairs arise from the
historical first-symbol mapping (`preferred_species_tag=None`); preferring
`[hs]` yields 89 pairs. Ward linkage with dendrogram cluster ordering reproduces
the paper's clustering rule.

### Shared spatiotemporal downstream workflow

The following command replaces the historical ARISTA-local copies of
`DeepRUOT`, classifier, piecewise warp, interaction, Sankey, and 3D plotting
code. The YAML contains only dataset keys, timepoints, and figure styling.

```bash
CUDA_VISIBLE_DEVICES=7 python scripts/run_spatiotemporal_downstream.py \
  --config CytoBridge/configs/arista_downstream.yaml \
  --aligned-h5ad /path/to/runs/arista-full/preprocess/arista_aligned.h5ad \
  --model-dir /path/to/runs/arista-full/training \
  --model-format current \
  --output-dir /path/to/runs/arista-full/spatiotemporal \
  --classifier-knn-neighbors 10 \
  --classifier-cache-dir /path/to/runs/arista-full/classifier_cache \
  --device cuda
```

The classifier cache is now created through
`train_cached_mlp_classifier_from_adata`. It remains compatible with historical
`classifier_resmlp_*.pt` files and is reused only when its feature/data/training
metadata match. The downstream command produces observed/generated snapshots,
lineage Sankey, per-timepoint attention-based interactions, and the focus-anchor
3D plot as HTML plus static SVG/PDF/PNG when Plotly image export is available.
New production classifiers use hidden size 128, 500 selection epochs, Adam
initialized at `1e-3`, cosine annealing over the 500-epoch selection horizon to
`1e-5`, seed 42, and held-out balanced accuracy for checkpoint selection.

For new classifier runs, `--classifier-refit-on-full-data-after-selection`
keeps model selection and final fitting distinct: it chooses the best epoch on
the held-out validation split, initializes a fresh model, and refits on all
rows for exactly that epoch count. Selection and refit metrics, split
definitions, scheduler horizons, seeds, and row scopes remain separate in the
cache evaluation record. `--classifier-strict-stratification` makes an impossible
stratified split a hard error instead of using the documented fallback. The
legacy `--classifier-train-on-full-data` behavior remains available for replay
but cannot be combined with the new refit mode.

Classifier prediction and spatial smoothing are separate operations.
`smooth_spatial_labels` accepts the raw MLP labels and explicit spatial
coordinates, records the requested/effective k, forces or excludes the query
cell according to `include_self`, and resolves exact-distance boundaries
deterministically. The held-out sweep remains reported as sensitivity evidence.
For the formal dataset workflows, AD mouse explicitly passes `k=1`, while
Zebrafish, MOSTA, and ARISTA explicitly pass `k=10` for all generated-cell
annotations and label-dependent analyses.
`analyze_spatial_label_sensitivity` reuses one raw prediction to compare
k=1/5/10/20/50 sensitivity settings and reports label
flips, composition total variation,
maximum percentage-point change, per-type abundance/retention, and
boundary/interior flip rates. This avoids retraining the MLP while studying k.
Historical full-training-scope classifier variants remain available only as
explicitly requested replay/sensitivity settings.

The ARISTA formal-analysis config is no-warp: split-SDE coordinates are used
directly for communication and the 3D rendering. Piecewise warp has two
explicit compatibility contracts. When explicitly enabled, the default
`--spatial-warp-visualization-only` keeps classifier labels, communication, and
the next SDE segment on the prewarp state while using a boundary-continuous
warped copy for display. The compatibility flag
`--no-spatial-warp-visualization-only` reproduces the later legacy behavior in
which each warped endpoint is carried into the next segment; use that mode for
historical display variants, not for formal downstream measurements.

For a dense display-only legacy-style mosaic/video, the same dataset-agnostic
entry point can generate the interpolation grid without enumerating hundreds of
times:

```bash
CUDA_VISIBLE_DEVICES=7 python scripts/run_spatiotemporal_downstream.py \
  --config CytoBridge/configs/arista_downstream.yaml \
  --aligned-h5ad /path/to/arista_aligned.h5ad \
  --model-dir /path/to/training \
  --output-dir /path/to/runs/arista/dense-piecewise-k10 \
  --dense-time-min 0 --dense-time-max 4 --dense-time-step 0.01 \
  --snapshot-time-points 0,0.5,1,1.5,2,2.5,3,3.5,4 \
  --spatial-warp-to-observed-piecewise \
  --no-spatial-warp-visualization-only \
  --classifier-knn-neighbors 10 \
  --render-video --video-formats gif --video-fps 8 \
  --skip-nonsplit-sde --skip-lineage --skip-communication --skip-3d \
  --device cuda
```

This dense mode renders a split-SDE birth/death population and explicitly opts
into the legacy display warp. It is therefore not
a lineage input: Sankey and 3D lineage ribbons require the non-split
fixed-particle trajectory. The GIF format quantizes frame delays; transcode to
MP4 with an explicit input rate when exact playback timing matters.

To use the workflow with another dataset, create a small YAML analogous to
`arista_downstream.yaml` and specify its AnnData time, annotation, latent, and
spatial keys. No dataset-specific simulation or interaction implementation is
required.

### Running on a shared server

Keep code, immutable source data, environments, scratch files, and run outputs
separate so multiple users do not overwrite each other:

```text
/data/cytobridge/projects/<project>/
├── repos/cytobridge-spatial/       # Git checkout; no large outputs
├── envs/<environment>/             # project-owned Python environment
├── workspace/                      # read-mostly source data/databases
├── runs/<dataset>/<run-id>/        # one directory per immutable run
└── scratch/                         # temporary archives and smoke tests
```

Use a descriptive run ID, record the Git commit in the run manifest, and select
GPUs per command with `CUDA_VISIBLE_DEVICES`; do not modify another user's
environment or results directory.

## Key Scripts

- `scripts/preprocess_pipeline.py`: end-to-end preprocessing, alignment, graph generation, and edge predictor training
- `scripts/run_spatial_training.py`: preset-based spatial training entry point
- `scripts/summarize_training_history.py`: training-curve and measured-resource summary entry point
- `scripts/run_arista_end_to_end.py`: historical ARISTA compatibility runner; use `cytobridge workflow --config arista` for new runs
- `scripts/run_spatiotemporal_downstream.py`: dataset-configured interpolation/lineage/communication/3D entry point
- `scripts/convert_legacy_weights_to_ckpt.py`: convert legacy checkpoints into the current checkpoint format
- `scripts/run_downstream_workflow_example.py`: downstream example workflow based on trained results

## Outputs

Typical training and preprocessing outputs include:

- aligned `.csv` and `.h5ad` files
- graph inputs and metadata
- trained edge predictor checkpoints
- staged training checkpoints under a results directory
- figures and downstream artifacts generated by analysis scripts

## External Resources

The following large or redistribution-restricted resources stay outside the
Git repository and are supplied explicitly by users:

- dataset download instructions or accession numbers
- the ligand-receptor database files used for preprocessing, if redistribution is restricted

## Documentation and citation

The complete user guide, five dataset tutorials, and API reference are built
from `docs/` and published through Read the Docs. To preview only the
documentation pages locally, install the base package plus the documentation
tools; install `.[all]` when you also want to execute the scientific examples:

```bash
python -m pip install -e '.[docs]'
make docs
```

If you use CytoBridge in academic work, cite the software metadata in
`CITATION.cff` and the associated manuscript once its final citation is
available.
