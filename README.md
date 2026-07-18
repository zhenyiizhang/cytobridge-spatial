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
- `CytoBridge.tl.evaluate_model_distributions`
- `CytoBridge.tl.train_cached_mlp_classifier_from_adata`
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
(two aligned spatial coordinates followed by 50 expression PCs). The companion
`cb_reproducibility` repository contains the canonical executable notebook.

## Canonical ARISTA reproduction

ARISTA has a dataset-specific entry point, but all computation is delegated to
the same public preprocessing, training, evaluation, simulation, communication,
lineage, and plotting APIs used by other datasets.

### Full H5AD-to-model run

```bash
CUDA_VISIBLE_DEVICES=7 python scripts/run_arista_end_to_end.py \
  --profile full \
  --stage all \
  --threshold-policy preprocess \
  --expression-layer counts \
  --h5ad-path /path/to/ARTISTA_after_pp_with_ae_and_center.h5ad \
  --database-path /path/to/CellChatDB.ligrec.human.csv \
  --output-dir /path/to/runs/arista-full \
  --device cuda
```

The full profile uses the recovered CytoBridge six-stage spatial schedule in
`CytoBridge/configs/arista_spatial_full.yaml`: 100 pretraining epochs, 100
refinement epochs, 50 interaction-initialization epochs, 2,001 score-matching
epochs, 1,000 finetuning epochs, and a final 2,001-epoch score refinement. It
preserves the recovered weighted-OT objective (`alpha_spatial=10`,
`alpha_express=0.05`), stage-specific mass conventions, forward-OT checkpoint
selection for intermediate stages, and the finetuning plateau scheduler.
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
The ARISTA source H5AD stores an already log-transformed matrix in `X` and
integer raw values in `layers['counts']`. The canonical runner therefore
copies `layers['counts']` into `X` before median-library-size normalization and
`log1p`, preventing the historical double transformation. This source choice
is recorded in `uns['preprocess_info']` and the run manifest. Passing
`--expression-layer X --allow-retransform-preprocessed-x` is reserved for an
explicitly labelled legacy replay; an already transformed `X` otherwise fails
fast instead of being silently transformed again. Because the available
ARISTA H5AD is already restricted to 2,000 genes, this run is described as a
counts-source, fixed-2,000-gene clean rerun rather than full-gene raw-data
preprocessing. The alignment recipe uses median-library-size normalization
(`target_sum=None` in Scanpy), 2,000 HVGs, 50 PCs, two aligned spatial
coordinates, and seed 42.
The canonical runner sets `evaluate_after_training=False` because the trainer's
historical in-memory evaluation is redundant and can retain a large autograd
graph. It evaluates the saved checkpoint immediately afterward through the
bounded, reproducible distribution-evaluation API instead.

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
- PCA/spatial generated-versus-observed figures
- joint, spatial, and PCA W1/W2 metrics plus TMV and local-structure diagnostics
- compressed generated/observed sample arrays used to calculate the figures
- a machine-readable `downstream/run_manifest.json`, including the exact
  weight/score checkpoint paths and SHA-256 hashes

Use `--profile smoke` only to test wiring. Its per-timepoint subsampling changes
nearest-neighbor distances, so its automatically estimated spatial cutoff is not
scientifically comparable with the full-data or published cutoff.

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
The source path/hash, column order, and interaction settings are written into
AnnData provenance. Because the CSV contains no gene-level expression, this is
an audit/migration input—not a substitute for the full H5AD preprocessing
workflow. It is useful for separating training-code changes from preprocessing
and coordinate changes in a controlled comparison.

### Threshold provenance

With `--threshold-policy preprocess` (recommended), preprocessing estimates the
spatial neighborhood cutoff from the aligned coordinates and selects the edge
classifier threshold on validation data. Both effective values and their sources
are stored in `adata.uns` and edge metadata. `CytoBridge.tl.fit` reads those values
directly from the aligned H5AD, so preprocessing and training cannot silently use
different thresholds.

For an explicit historical control, `--threshold-policy published` stores and
uses the ARISTA manuscript values:

- spatial neighborhood cutoff: `0.05`
- edge-predictor decision threshold: `0.45`

This control is useful for sensitivity analysis; it is not an instruction to
reuse a smoke-run threshold on the full dataset.

Because the neighborhood policy also changes the graph used to train the edge
classifier, two independently preprocessed policy runs may have different edge
weights even when their aligned coordinates and PCA values match. To isolate
fit-time thresholds, copy one prepared H5AD and edge model byte-for-byte and use
the runner's explicit training overrides:

```bash
python scripts/run_arista_end_to_end.py \
  --profile full --stage train \
  --h5ad-path /path/to/frozen/preprocess/arista_aligned.h5ad \
  --database-path /path/to/CellChatDB.ligrec.human.csv \
  --output-dir /path/to/frozen-control \
  --training-interaction-cutoff 0.05 \
  --training-edge-predictor-threshold 0.45 \
  --training-edge-predictor-path /path/to/frozen/preprocess/edge_classifier/arista_edge_model.pt
```

The resolved `training/config.yaml` records the effective values and edge-model
path. A defensible paired control must also verify identical aligned-H5AD and
edge-model hashes before training.

Raw cutoff values should be interpreted together with coordinate scale. For
cross-run comparisons, report the coordinate range/standard deviation, median
nearest-neighbor distance, and `cutoff / median_nn`; do not attribute a cutoff
difference to biology until the aligned-coordinate scales have been checked.

In the full ARISTA audit, the historical and current aligned coordinates have
nearly identical scale (median nearest-neighbor distance `0.003160` versus
`0.003142`; coordinate correlations `0.9977` and `0.9900`). The historical
`0.05` cutoff is therefore not explained by a coordinate rescaling: it spans
about `15.8` historical median-neighbor distances, whereas the full-data
preprocessing cutoff `0.031543` spans about `10.0`. The current edge threshold
is `0.35` (validation-selected), compared with the historical `0.45`.

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
generic AnnData keys and model components; the companion notebooks contain only
the ARISTA timepoint, annotation, and reaEGC ROI choices.

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
  simulated PCA states, and clusters the resulting gene trajectories;
- `project_communication_to_lr_timecourses` combines those reconstructed
  expression values with per-timepoint `M_per_source` communication matrices;
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

The enrichment API does not download or silently select a database. Callers
provide a versioned GMT file and record its hash in the run manifest:

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

The default, prospective workflow requires a package-processed reference H5AD
that retains PCA loadings. An older H5AD containing only `X_pca` coordinates is
not sufficient by itself. For a declared historical reproduction, callers may
instead load the archived PCA components and center with
`load_pca_reconstruction_spec`; both source paths and hashes should be recorded
in the run manifest. `preferred_species_tag` makes cross-species symbol choice
explicit, while `profile_linkage_method` and `profile_cluster_order` expose the
clustering contract. For ARISTA, the paper's 68 LR pairs arise from the
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
  --classifier-knn-neighbors 1 \
  --classifier-cache-dir /path/to/runs/arista-full/classifier_cache \
  --device cuda
```

The classifier cache is now created through
`train_cached_mlp_classifier_from_adata`. It remains compatible with historical
`classifier_resmlp_*.pt` files and is reused only when its feature/data/training
metadata match. The downstream command produces observed/generated snapshots,
lineage Sankey, per-timepoint attention-based interactions, and the focus-anchor
3D plot as HTML plus static SVG/PDF/PNG when Plotly image export is available.

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
- `scripts/run_arista_end_to_end.py`: canonical ARISTA preprocess/train/evaluate entry point
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

## Companion Resources

For a clean code release, we recommend keeping the following resources separate from this repository:

- a companion repository containing downstream notebooks for figure generation
- dataset download instructions or accession numbers
- the ligand-receptor database files used for preprocessing, if redistribution is restricted

## Citation

If you use this repository in academic work, please cite the associated manuscript once available.
