# Clean-counts spatial-temporal workflow

This guide documents the native CytoBridge workflow represented by
`scripts/run_zebrafish_end_to_end.py`. It has two purposes:

1. reproduce the clean-counts zebrafish preprocessing, paired training, and
   quantitative evaluation; and
2. show which parts can be reused for another spatial-temporal dataset and
   which parts must remain in a small dataset adapter.

Do not run the zebrafish script unchanged on a new dataset. It deliberately
freezes zebrafish time labels, annotations, coordinate transforms, and the two
training conditions.

## Shared package APIs versus dataset definitions

The package should own algorithms and data contracts. A dataset adapter should
own biological names and coordinate/time conventions.

| Shared CytoBridge functionality | Public API | Dataset-specific decision |
| --- | --- | --- |
| Clean expression selection and double-transform guard | `cb.pp.preprocess` | Raw-count layer name and normalization convention |
| PCA and preservation of its exact fit-time center | `cb.pp.preprocess`, `cb.tl.infer_pca_center` | Number of PCs and cells/time points included in the PCA fit |
| Spatial alignment | `cb.pp.AlignConfig`, `cb.pp.align_spatial`, `cb.pp.preprocess_and_align` | Coordinate scaling, centering, orientation, time points retained, and alignment weights |
| Neighborhood estimation and interaction graphs | `cb.pp.estimate_neighborhood_threshold_from_aligned_spatial`, `cb.pp.generate_interaction_graph` | Species-specific ligand-receptor database and any justified fixed threshold |
| Edge-predictor training | `cb.pp.train_edge_predictor` | Dataset name, validation split, and optional fixed decision threshold |
| Six-stage dynamical training | `cb.tl.fit` | Training YAML, spatial/expression OT weights, noise, and seed |
| Native checkpoint loading | `cb.tl.load_dynamical_model_from_dir`, `cb.tl.build_dynamical_runtime` | Which completed condition/model directory to use |
| W1/W2/TMV and support diagnostics | `cb.tl.evaluate_model_distributions`, `cb.tl.save_distribution_evaluation` | Sample count and comparison policy |
| Velocity, interpolation, classification, growth, and communication | APIs under `cb.tl`, including `compute_velocity_components_from_adata`, `run_interpolation_workflow`, `train_cached_mlp_classifier_from_adata`, and `compute_timepoint_communications` | Biological labels, displayed time points, and visualization-only choices |
| Ligand-receptor time courses | `cb.tl.project_communication_to_lr_timecourses` | Database, symbol policy, target pair, and observed/generated expression contract |
| Virtual cell-type ablation | `cb.tl.run_virtual_cell_type_ablation` | Labels to remove and the biological interpretation of the perturbation |

`scripts/preprocess_pipeline.py` is repository-level orchestration of the
public preprocessing, graph, and edge-predictor APIs. It is useful for a new
dataset, but it is not itself the package API.

## Input contract

The zebrafish clean-counts runner requires one source H5AD and one
species-appropriate ligand-receptor CSV.

The source H5AD must contain:

- `adata.layers["counts"]`: non-negative, finite, raw count-like values;
- `adata.obs["time"]`: the original zebrafish stage labels;
- `adata.obs["bin_annotation"]`: cell-type labels;
- `adata.obs["colors"]`: label colors;
- spatial coordinates in `adata.obsm["spatial"]`, or in both
  `adata.obs["spatial_x"]` and `adata.obs["spatial_y"]`;
- stable and unique `adata.obs_names`.

`adata.X` may already be normalized/log-transformed. The clean runner does not
normalize or log-transform that matrix again: it explicitly promotes
`layers["counts"]` into `X`, normalizes to the median library size, and applies
one `log1p` transform.

The ligand-receptor CSV is used to construct interaction graphs during
preprocessing. For manuscript communication analysis, it is also the database
used to score LR pairs. It must therefore match the species and gene symbols of
the input data.

## Frozen zebrafish definitions

These settings reproduce this dataset; they are not general defaults.

### Time axis

The source data contain six stages:

```python
{
    "3.3hpf": -1.0,
    "5.25hpf": 0.0,
    "10hpf": 1.0,
    "12hpf": 2.0,
    "18hpf": 3.0,
    "24hpf": 4.0,
}
```

HVG selection, normalization, and PCA fitting occur before alignment-time
subsetting, so the PCA fit sees all six stages. Alignment and model training
retain original batch indices `[1, 2, 3, 4, 5]`, yielding model times
`0, 1, 2, 3, 4`. The saved `adata.var["pca_center"]` is the center from the
full PCA fit, not a center recomputed from the retained five stages.

For another dataset, inspect the actual categorical/observed time order before
choosing `batch_indices`; a numeric `time_mapping` does not by itself redefine
the category order used by that selection.

### Annotation and coordinates

The adapter maps:

- `bin_annotation` to the downstream key `Annotation`;
- `colors` to `Color`;
- the original `time` label to `time_label`.

Zebrafish coordinates are two-dimensional, divided by a shared scale of 500,
centered on the x axis, and not centered on the y axis. The alignment settings
are `alpha=5`, `beta=0.01`, and `lambda_local=100`.

These alignment weights are distinct from the dynamical training setting
`alpha_spatial=10`. Coordinate scale changes the numerical meaning of both
neighborhood and interaction cutoffs, so `/500`, x-only centering, and any
fixed spatial threshold must not be copied to another platform without a
nearest-neighbor and coordinate-range audit.

### Biological panel choices

The zebrafish manuscript adapter defines:

- target LR pair: `cxcl12a_cxcr4a`;
- virtual-ablation labels: `Yolk Syncytial Layer` and `EVL`;
- a zebrafish ligand-receptor database.

These are biological questions, not model defaults. A new dataset may omit
these panels or replace them with labels and LR pairs that exist in its own
annotation/database.

## Validate clean counts before a full run

Run a bounded check rather than densifying the full matrix:

```python
import numpy as np
import scanpy as sc
from scipy import sparse

adata = sc.read_h5ad("/path/to/source.h5ad", backed=None)
assert "counts" in adata.layers
assert "time" in adata.obs
assert "bin_annotation" in adata.obs

counts_sample = adata.layers["counts"][: min(256, adata.n_obs)]
values = counts_sample.data if sparse.issparse(counts_sample) else np.asarray(counts_sample).ravel()
assert np.isfinite(values).all()
assert (values >= 0).all()
assert np.mean(np.isclose(values, np.rint(values), atol=1e-7)) > 0.999

spatial = (
    np.asarray(adata.obsm["spatial"])
    if "spatial" in adata.obsm
    else adata.obs[["spatial_x", "spatial_y"]].to_numpy()
)
assert spatial.shape == (adata.n_obs, 2)
assert np.isfinite(spatial).all()
print(adata.obs["time"].value_counts(dropna=False))
```

The clean path intentionally uses:

```python
expression_layer = "counts"
allow_retransform_preprocessed_x = False
raw_count_validation = "auto"  # strict because an explicit layer is selected
```

Do not enable `allow_retransform_preprocessed_x` to make an unexplained input
pass. That option exists only for an explicitly labelled legacy replay. If
`X` is transformed and no raw-count layer exists, recover the raw counts or
define a no-normalization/no-log contract; do not silently log the data again.

After preprocessing, verify the recorded provenance:

```python
import scanpy as sc

adata = sc.read_h5ad("RUN/preprocess/zebrafish_aligned.h5ad")
info = adata.uns["preprocess_info"]
assert info["expression_source"] == "layers['counts']"
assert info["expression_layer"] == "counts"
assert info["raw_counts_layer"] == "counts"
assert info["raw_count_validation_effective"] == "strict"
assert info["raw_count_validation_stats"]["n_noninteger_like"] == 0
assert info["allow_retransform_preprocessed_x"] is False
assert info["transformation_sequence"] == ["normalize_total", "log1p"]
assert "pca_center" in adata.var
assert adata.uns["pca_center_info"]["n_vars_fit"] == adata.n_vars
assert sorted(adata.obs["time_point_processed"].astype(float).unique()) == [0, 1, 2, 3, 4]
```

The strict check visits all explicitly stored sparse values, or every dense
value in bounded row chunks; it is not the 256-row diagnostic sample shown
above. `raw_count_validation="off"` is a compatibility escape hatch for a
deliberately non-count preprocessing contract. It must not be used merely to
make an unidentified transformed layer pass.

If the raw layer has a different name, select that exact layer. It remains the
canonical source for both normalization/log1p and interaction-graph
construction, even if a stale `layers["counts"]` also exists:

```python
cfg = AlignConfig(
    expression_layer="raw_umi",
    raw_count_validation="strict",
)
```

The resolved layer is recorded in
`adata.uns["preprocess_info"]["raw_counts_layer"]`.

## Generic preprocessing CLI schema

The repository-level preprocessing runner accepts dataset schema choices
without editing Python. For example:

```bash
python scripts/preprocess_pipeline.py \
  --data-name my_dataset \
  --h5ad-path /path/to/input.h5ad \
  --database-path /path/to/species_lr.csv \
  --time-key stage \
  --time-mapping '{"E8.5": 0, "E9.5": 1, "E10.5": 2}' \
  --expression-layer raw_umi \
  --raw-count-validation strict \
  --input-spatial-key tissue_xy \
  --spatial-dim 2 \
  --device cuda
```

`--time-mapping` accepts inline JSON, a JSON file path, or `@path`. JSON object
keys are strings; for numeric source labels, a pair list preserves their types,
for example `--time-mapping '[[1, 0], [2, 1], [3, 2]]'`. The preprocessing API
also accepts numeric-key Python dictionaries. Saved preprocessing provenance
uses a type-explicit JSON record, while alignment configuration stringifies
nested mapping keys so that `AnnData.write_h5ad` remains safe.

If coordinates are stored in `obs` rather than `obsm`, name them explicitly:

```bash
--input-spatial-key absent --spatial-obs-keys x_coord,y_coord
```

The number of coordinate columns, coordinate finiteness, and
`--spatial-dim` are validated before alignment. `--spatial-key` is different:
it selects the aligned coordinate representation used later for neighborhood
and interaction-graph construction.

## Reproduce the paired zebrafish run

Use a new run root rather than writing into a repository checkout. Both alpha
conditions must use the same completed preprocessing directory and seed.

```bash
cd /path/to/cytobridge-spatial

RUN=/path/to/runs/zebrafish-clean-counts-comparison
INPUT=/path/to/spatial_sixtime_slice_stereoseq.h5ad
LR_DB=/path/to/CellChatDB.ligrec.zebrafish.csv
```

### 1. Preprocess once

```bash
CUDA_VISIBLE_DEVICES=0 CYTOBRIDGE_ASSIGNED_GPU=0 \
python scripts/run_zebrafish_end_to_end.py \
  --h5ad-path "$INPUT" \
  --database-path "$LR_DB" \
  --output-dir "$RUN" \
  --profile full \
  --stage preprocess \
  --device cuda \
  --random-seed 42
```

Before training, inspect the aligned H5AD, the selected neighborhood threshold,
the edge-predictor metadata, and the five retained time counts.

### 2. Train both alpha-expression conditions

The two provided full configs have the same six-stage schedule and
`alpha_spatial=10`; only `alpha_express` differs. Run these in separate shells
and on separate free GPUs if training them concurrently.

```bash
CUDA_VISIBLE_DEVICES=0 CYTOBRIDGE_ASSIGNED_GPU=0 \
python scripts/run_zebrafish_end_to_end.py \
  --h5ad-path "$INPUT" \
  --database-path "$LR_DB" \
  --output-dir "$RUN" \
  --profile full \
  --stage train \
  --condition alpha_express_0015 \
  --device cuda \
  --random-seed 42
```

```bash
CUDA_VISIBLE_DEVICES=1 CYTOBRIDGE_ASSIGNED_GPU=1 \
python scripts/run_zebrafish_end_to_end.py \
  --h5ad-path "$INPUT" \
  --database-path "$LR_DB" \
  --output-dir "$RUN" \
  --profile full \
  --stage train \
  --condition alpha_express_005 \
  --device cuda \
  --random-seed 42
```

The runner resolves the default YAML from the condition name. If
`--training-config` is supplied, the runner still overwrites the seed,
checkpoint directory, spatial dimension, `alpha_spatial`, and
condition-specific `alpha_express`; the resolved `training/config.yaml` is the
authoritative record of what ran.

### 3. Run native quantitative downstream evaluation for both models

```bash
CUDA_VISIBLE_DEVICES=0 CYTOBRIDGE_ASSIGNED_GPU=0 \
python scripts/run_zebrafish_end_to_end.py \
  --h5ad-path "$INPUT" \
  --database-path "$LR_DB" \
  --output-dir "$RUN" \
  --profile full \
  --stage downstream \
  --condition alpha_express_0015 \
  --device cuda \
  --random-seed 42
```

```bash
CUDA_VISIBLE_DEVICES=1 CYTOBRIDGE_ASSIGNED_GPU=1 \
python scripts/run_zebrafish_end_to_end.py \
  --h5ad-path "$INPUT" \
  --database-path "$LR_DB" \
  --output-dir "$RUN" \
  --profile full \
  --stage downstream \
  --condition alpha_express_005 \
  --device cuda \
  --random-seed 42
```

Compare both condition tables on the same `(time, space)` grid. Lower W1, W2,
and TMV are better, but model selection should also inspect spatial/PCA support
and clumping diagnostics rather than reducing the decision to one mean value.

Use `--profile smoke` first when validating a new environment or adapter. A
smoke run caps cells and epochs and is not a scientific model comparison.

### 4. Build a portable paired review bundle

Run both manuscript-downstream conditions to completion before packaging. The
paper root manifest must report all seven stages: `classifier`, `velocity`,
`s22`, `growth`, `ablation`, `s25`, and `communication`.

Always use a new output directory and `--bundle-mode copy` for an artifact that
will be downloaded or archived. Logs are opt-in: name only the canonical logs
from the formal run. The bundler deliberately does not copy the whole mixed
`RUN/logs/` directory, because it may also contain smoke runs, failed attempts,
or superseded outputs.

```bash
BUNDLE="$RUN/reviews/DATE/zebrafish_clean_counts_full_review_bundle_DATE"

python scripts/compare_zebrafish_conditions.py \
  --run-root "$RUN" \
  --output-dir "$BUNDLE" \
  --conditions alpha_express_0015 alpha_express_005 \
  --baseline alpha_express_0015 \
  --winner auto \
  --bundle-mode copy \
  --selected-panels "alpha_express_0015=$RUN/conditions/alpha_express_0015/paper_downstream_final_DATE" \
  --selected-panels "alpha_express_005=$RUN/conditions/alpha_express_005/paper_downstream_final_DATE" \
  --paper-output "alpha_express_0015=$RUN/conditions/alpha_express_0015/paper_downstream_final_DATE" \
  --paper-output "alpha_express_005=$RUN/conditions/alpha_express_005/paper_downstream_final_DATE" \
  --canonical-log "alpha_express_0015_train.log=/path/to/formal-0015-train.log" \
  --canonical-log "alpha_express_0015_paper.log=/path/to/formal-0015-paper.log" \
  --canonical-log "alpha_express_005_train.log=/path/to/formal-005-train.log" \
  --canonical-log "alpha_express_005_paper.log=/path/to/formal-005-paper.log"
```

`--selected-panels` chooses the visual-artifact source, while `--paper-output`
chooses the matching root and seven stage manifests for provenance. Passing
both explicitly is recommended whenever a corrected workflow writes to a fresh
sibling directory instead of the default `paper_downstream` path. Paper roots
are filtered to visual file types, so canonical state arrays and classifier
caches are not copied into the portable bundle.

Use `--winner alpha_express_0015` or `--winner alpha_express_005` only after an
explicit biological and visual review. `auto` applies the documented numerical
screening score; it is not a biological conclusion. Both condition panel sets
are always retained under `02_condition_panels/`. For compatibility with older
review tooling, the selected condition is also copied to
`02_selected_manuscript_panels/`.

The bundle includes:

```text
README.md
01_condition_comparison/
02_condition_panels/
├── alpha_express_0015/
└── alpha_express_005/
02_selected_manuscript_panels/
03_condition_inputs/
├── alpha_express_0015/
└── alpha_express_005/
04_logs/
05_provenance/
├── reproduction_commands.txt
└── artifact_inventory.csv
```

Each condition input directory contains its distribution metrics, quantitative
downstream manifest, resolved training configuration, launch manifest, paper
root manifest, and all seven paper stage manifests. The artifact inventory
records each bundled file's relative path, size in bytes, and SHA-256; it omits
itself to avoid a self-referential checksum. A non-empty review output directory
is rejected even when overwrite is requested, preventing stale panels from a
previous bundle from surviving unnoticed.

## Output contract

The stable core runner writes the following structure:

```text
RUN/
├── preprocess/
│   ├── zebrafish_aligned.h5ad
│   ├── zebrafish_aligned.csv
│   ├── zebrafish_aligned_with_annotation.csv
│   ├── zebrafish_nn1_stats.csv
│   ├── input_graph/
│   └── edge_classifier/
│       ├── zebrafish_edge_model.pt
│       └── zebrafish_edge_model.pt.meta.json
└── conditions/
    ├── alpha_express_0015/
    │   ├── training/
    │   │   ├── config.yaml
    │   │   ├── adata.h5ad
    │   │   ├── launch_manifest.json
    │   │   └── <six stage checkpoint directories>/
    │   └── downstream/
    │       ├── velocity_components.npz
    │       ├── distribution_evaluation/
    │       │   ├── distribution_metrics.csv
    │       │   ├── distribution_samples.npz
    │       │   ├── generated_vs_observed_spatial.svg
    │       │   └── generated_vs_observed_pca.svg
    │       └── run_manifest.json
    └── alpha_express_005/
        └── <same contract>
```

The aligned H5AD is the primary hand-off artifact. Its important contract is:

- `X`: processed gene expression;
- `layers["counts"]`: count-space expression used by interaction graphs;
- `obsm["X_latent"]`: PCA representation;
- `obsm["spatial_aligned"]`: aligned two-dimensional coordinates;
- `obs["time_point_processed"]`: model time;
- `obs["Annotation"]`: downstream label;
- `varm["PCs"]` and `var["pca_center"]`: one consistent inverse-PCA contract;
- `uns["preprocess_info"]`, `uns["pca_center_info"]`, and interaction-graph
  metadata: provenance and learned thresholds.

The training loader normally resolves the `Finetune` weight checkpoint and the
last configured score-matching checkpoint (`Score_Refine` for these configs).
Do not choose a checkpoint only by filename; read `config.yaml` and the
downstream `run_manifest.json`, which record the actual stages and SHA-256
hashes used.

## GPU isolation on a shared server

1. Check live utilization and memory with `nvidia-smi` immediately before
   launching each job.
2. Give every user/project a unique run root. Never share a training directory
   between processes.
3. Set both `CUDA_VISIBLE_DEVICES` and `CYTOBRIDGE_ASSIGNED_GPU`. The latter is
   provenance; the former performs the isolation.
4. With `CUDA_VISIBLE_DEVICES=7`, use `--device cuda` (or logical `cuda:0`), not
   physical `cuda:7`: the visible physical card is remapped to logical card 0.
5. Only the completed `RUN/preprocess/` artifacts should be shared between the
   two conditions. Each condition already has its own training/downstream
   directory.
6. Do not run `--stage all` concurrently for two conditions under one run root,
   because both processes would rewrite the shared preprocessing outputs.

The launch and downstream manifests record the requested device,
`CUDA_VISIBLE_DEVICES`, `CYTOBRIDGE_ASSIGNED_GPU`, hostname, PID, Git revision,
and relevant input/checkpoint hashes.

## Resume and restart behavior

The core runner supports safe stage-level continuation:

- after preprocessing succeeds and is validated, continue with
  `--stage train` without rerunning preprocessing;
- after training succeeds, continue or repeat `--stage downstream`;
- the two alpha conditions can proceed independently from the same immutable
  preprocessing result.

There is currently no `--resume` option for a partially completed six-stage
training invocation. Although individual stage checkpoints are written, a new
`--stage train` command starts the configured training plan again and may
overwrite files in that condition directory. After an interrupted formal
training job, preserve the incomplete directory for diagnosis and restart in a
new run/condition directory unless an explicitly documented checkpoint-resume
API is added.

The manuscript-specific runner is implemented at
`scripts/run_zebrafish_paper_downstream.py`. Use the script's live help as the
authoritative command contract for the checked-out revision:

```bash
python scripts/run_zebrafish_paper_downstream.py --help
```

Classifier reuse is part of the recorded contract. The runner trains the main
trajectory classifier once and passes the same `classifier_cache_tag` through
`cb.tl.run_interpolation_workflow`, so S22 and S25 resolve the identical cache
file rather than silently training a second classifier. For a new dataset,
choose a versioned cache tag that describes its feature contract and keep it
identical between the explicit classifier-training stage and every workflow
that consumes that classifier. Cache writers are serialized and checkpoints
are published by atomic rename, so two workers cannot leave a partially written
classifier file.

Split-SDE population events use a fixed `resample_dt=0.05`, independent of the
frames requested for plotting. This matters because birth/extinction sampling
is part of the dynamics: changing a video from 9 to 41 frames must not change
the trajectory itself. S22 performs one global simulation on its canonical
0.1 grid, keeps the pre-warp model state, applies the historical piecewise warp
only to a display copy, and selects the 0.5-step mosaic from that same run.
The runner also applies a fail-fast particle ceiling before split allocation;
this guard never downsamples a valid run.

The manuscript redraws are assembled by reusable plotting APIs rather than by
copying historical PDFs. S22 uses a 3-by-3 wrapped trajectory grid with one
figure-level cell-type legend. S23 exports the raw per-cell growth table and a
3-by-2 observed-time grid; each time point is robust-scaled from its own 5th to
95th percentiles for display, while the unscaled predictions remain in
`growth_per_cell.csv`. S24 uses the generic time-by-condition trajectory grid
for baseline, virtual YSL removal, and virtual EVL removal. These panels are a
single-seed virtual sensitivity analysis, not causal knockout estimates.

S25 follows the manuscript wording "observed and interpolated time points":
integer stages use the actual observed cells and annotations, while half-time
stages reuse S22's canonical generated pre-warp states. S22 keeps its historical
`k=10` spatial label smoothing for the display trajectory. Rare YSL selection
at generated S25 half-times defaults to the direct classifier (`k=1`), because
majority smoothing can erase a biologically present rare class. The two label
policies and every per-timepoint YSL count are recorded separately; a full run
still fails rather than fabricating cells if the direct classifier predicts no
target cells.

Communication/LR does not inherit the labels stored in the S22/S25 trajectory
bundle. It explicitly applies the same cached classifier to every generated
pre-warp frame and defaults to direct prediction (`k=1`); the corresponding CLI
setting is `--communication-classifier-knn-neighbors 1`. Observed times keep
their actual annotations. Pass `10` only to create a separate legacy
manuscript-parity sensitivity result. The stage records the classifier cache
hash and fingerprint, per-cell inherited versus analysis labels, per-time label
counts, and the fraction changed, so display smoothing cannot silently alter
biological communication scores.

For a downstream-only audit that must reuse an already published S22
trajectory byte-for-byte, pass its `canonical_prewarp_states` directory with
`--s25-canonical-state-bundle`. The runner validates the bundle index, every
frame SHA-256, and the adjacent S22 stage manifest before S25 or communication
opens it. This option is deliberately an adapter-level provenance hook; it does
not change the generic simulation or temporal-analysis APIs.

Gene dynamics at observed and generated times use the same retained-PCA inverse
map and its persisted fit-time center. Only features with a nonzero retained
loading are eligible: a zero-loading non-HVG would otherwise be the same global
center value at every generated state and is not evidence for gene dynamics.
The exported settings record the active/inactive feature counts and loading
tolerance. Rank-truncated inverse-PCA values are signed reconstruction estimates;
the temporal heatmap compares their gene-wise profiles and does not treat them
as raw counts.

The default LR dynamics uses one reconstructable feature universe and one
measurement operator across the complete trajectory: every time point,
including observed integer-state PCA coordinates, is decoded with the same
retained-PCA inverse map. Per-cell log1p estimates are clipped at zero, converted
with `expm1`, and only then averaged by cell type. These are library-size
normalized count-like abundances, not raw counts. LR complexes require every
subunit, and any center-only subunit is excluded globally. All communication
inputs remain unwarped model states.

The runner also exports a hybrid exact-observed projection as an anchor
validation and legacy manuscript-parity result. That projection uses real
observed log1p expression at integer times and inverse-PCA expression at
half-times; it is not the default continuous LR dynamics because switching
measurement operators can create artificial temporal kinks. Pair- and
cell-type-level comparisons, per-time scale/rank metrics, and an anchor
source-switch diagnostic quantify that effect. Half-time scores must agree
exactly between the two computations. Pass
`--lr-expression-time-policy hybrid_exact_observed` only when explicitly
reproducing the legacy figure. The neighbor-continuity table remains a
measurement-source diagnostic, not proof that the biological dynamics or
communication matrix should be linear in time.

Virtual-ablation branches reset to the same branch-level seed for manuscript
parity, but removing cells changes tensor shape and row order. The random
increments are therefore not matched by cell ID; use multiple seeds and report
uncertainty for inferential use beyond the single-seed manuscript reproduction.

S25 globally orders the top temporal-variance genes once, then lays that order
out as two contiguous 125-gene blocks with one shared z-score scale. The split
is page layout only; it does not refit clustering or choose separate gene sets.

Do not copy historical interpolated H5AD files, communication pickles, or
figure PDFs into a new run and treat them as recomputed results.

The review bundler accepts only `profile=full` native paper roots. It validates
the root and all seven stage manifests, checks every recorded artifact's byte
size and SHA-256, and copies only manifest-recorded visual outputs. Unrecorded
stale figures and state arrays therefore cannot enter the portable panel bundle.

## Adaptation checklist for a new dataset

Create a small `run_<dataset>_end_to_end.py` adapter or equivalent config layer;
do not fork generic preprocessing, training, or downstream implementations.

1. **Expression:** identify an actual raw-count layer; choose one normalization
   target and one log transform; retain the double-transform guard.
2. **Time:** define an explicit ordered mapping and decide whether any stage is
   PCA-only, alignment-only, or excluded. Verify the final model times.
3. **Coordinates:** define dimensionality, units, scaling, centering, and axis
   orientation from the data. Re-estimate nearest-neighbor scale after
   alignment.
4. **Annotations:** define the canonical downstream label and optional color
   columns. Preserve stable cell IDs through every conversion.
5. **Graph:** provide a species-appropriate interaction database; estimate the
   spatial neighborhood cutoff and train/validate a new edge predictor.
6. **Training:** start from a versioned YAML. Treat both alignment weights and
   dynamical `alpha_spatial`/`alpha_express` as tunable data-scale-dependent
   parameters, not zebrafish constants.
7. **Evaluation:** run the same W1/W2/TMV and support diagnostics at observed
   times with fixed seeds and sample caps across candidate models.
8. **Downstream biology:** define classifiers, target LR pairs, ablation labels,
   and any display-only warp in the adapter. Keep warping out of dynamical,
   communication, and ablation state unless the scientific method explicitly
   requires it.
9. **Provenance:** save resolved configs, data/checkpoint hashes, thresholds,
   time counts, software revision, GPU assignment, and per-stage manifests.

This separation lets another dataset reuse the same CytoBridge APIs while
making every non-transferable biological or coordinate choice visible and
reviewable.
