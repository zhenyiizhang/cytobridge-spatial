---
orphan: true
---

# Non-spatial analyses: from data and models to S5–S6

This workflow calculates numerical results from fitted models, converts them
to panel tables, and draws S5–S6. Run the Python blocks in order with the
CytoBridge code folder available. The
[S5–S6 notebook](../../tutorials/paper_figures/nonspatial_figures.ipynb) runs
the same model-analysis and plotting sequence. The
[S5 walkthrough](../../tutorials/paper_figures/weinreb_s4.ipynb) recalculates
the three-model comparison from individual simulation results on a CPU,
with an optional GPU step to repeat inference from all six checkpoints.

| Panels | Models and calculation |
| --- | --- |
| S5a, S6a | Measured coordinates, times and labels from the prepared H5AD. |
| S5b/e/f | Weinreb LR-informed models, training seeds 42–44: fields averaged over five groupings per model, exact sender-specific GNN messages and LR-weighted pathways. |
| S5c/d | The same three LR Full models as b/e/f and three separately trained No-interaction models. |
| S6b/e/f | Cortical LR-informed Full seed-42 model: simulated trajectories, exact messages and LR-weighted pathways. |
| S6c/d | Three cortical LR Full and three No-interaction models: distribution errors and the proportion of cells aligned with the measured new-RNA direction. |

## 1. Download observations and fitted models

Choose a project directory and available device. Keep the original prepared
H5AD with each checkpoint: its saved 50-PC coordinates were used to fit that
model. The loader opens the non-spatial attention model used in these analyses.

```python
from pathlib import Path
import CytoBridge as cb
from reproduction.nonspatial.inputs import downloaded_sources
from reproduction.nonspatial import analyze

project = Path("nonspatial_project").resolve()
device = "cuda:0"  # Use an available GPU, or "cpu" for a slower run.
for dataset in ("weinreb", "scnt_cortex"):
    cb.datasets.download(dataset, destination=project)
cb.datasets.download("nonspatial", kind="nonspatial_paper_inputs.zip",
                     destination=project)
weinreb, scnt, lr_models = downloaded_sources(project)
wroot, sroot = project / "data/weinreb", project / "data/scnt_cortex"
output = project / "outputs/model_recalculation"
output.mkdir(parents=True)
sfull, sno = sroot / "full/model", sroot / "no_interaction/model"
wexpression, wprior = wroot / "original/Weinreb.h5ad", wroot / "edge_prior/manifest.json"
sexpression = sroot / "original/hvg2000/scnt_cortical_full_total_expression.h5ad"
sprior = sroot / "edge_prior/manifest.json"
sraw = sroot / "original/scnt_cortical_full_20547_raw_counts.h5ad"
spca = sroot / "original/hvg2000/scnt_cortical_full_hvg2000_pca_artifacts.npz"
```

## 2. Prepare data, construct a prior and train, if starting a new run

Skip this section to use the distributed weights. Preparation writes
`model_input_50pc.h5ad`, `lr_expression.h5ad`, `pca_artifacts.npz` and
`preprocess_manifest.json`. Prior construction trains a directed LR edge
predictor from expression and writes its weights, graph samples, LR metadata
and `edge_prior/manifest.json`. Clone, cell-type and new-RNA labels are not
used to fit these components.

```bash
PROJECT=/absolute/path/nonspatial_project
WRUN="$PROJECT/outputs/new_weinreb"
SRUN="$PROJECT/outputs/new_scnt"
DEVICE=cuda:0

cytobridge nonspatial prepare --dataset weinreb \
  --input-h5ad "$PROJECT/data/weinreb/original/Weinreb.h5ad" \
  --output-dir "$WRUN/preprocess"
cytobridge nonspatial build-prior --dataset weinreb \
  --preprocess-manifest "$WRUN/preprocess/preprocess_manifest.json" \
  --output-dir "$WRUN/edge_prior" --device "$DEVICE"
cytobridge nonspatial prepare --dataset scnt_cortex \
  --input-h5ad "$PROJECT/data/scnt_cortex/original/scnt_cortical_full_20547_raw_counts.h5ad" \
  --output-dir "$SRUN/preprocess"
cytobridge nonspatial build-prior --dataset scnt_cortex \
  --preprocess-manifest "$SRUN/preprocess/preprocess_manifest.json" \
  --output-dir "$SRUN/edge_prior" --device "$DEVICE"
```

These commands make a new preprocessing/prior pair for each dataset. To train
in the paper's original PCA space, use the downloaded prepared data and its
matching prior instead.

The original state-space training implementation is distributed separately
from the current spatial GNN. Download its small source archive:

```python
cb.datasets.download("nonspatial", kind="nonspatial_training_code.zip",
                     destination=project)
```

From the CytoBridge code folder, run the selected original training entry:

```bash
TRAINING="$PROJECT/data/nonspatial/training_code"
python reproduction/nonspatial/train.py --training-code "$TRAINING" --model scnt_lr \
  --input-h5ad "$SRUN/preprocess/model_input_50pc.h5ad" \
  --edge-prior "$SRUN/edge_prior/link_predictor.pt" \
  --output-dir "$SRUN/lr_seed42" --seed 42 --device "$DEVICE"
python reproduction/nonspatial/train.py --training-code "$TRAINING" --model scnt_no_interaction \
  --input-h5ad "$SRUN/preprocess/model_input_50pc.h5ad" \
  --output-dir "$SRUN/no_interaction" --seed 42 --device "$DEVICE"
```

The S5 training code and its matching runtime are in the
[three-model archive](https://github.com/zhenyiizhang/cytobridge-spatial/releases/tag/weinreb-s4-20261029).
Use that archive for the S5 No-interaction comparison, not the older
single-model training entry. Its manifests record the six-stage configuration
for every seed and the energy-value and gradient check.

After cortical training, select the new models and their matching inputs in Python.
Each `--output-dir` above is itself the model directory. This assignment
replaces the downloaded-model selection for every calculation below.

```python
wrun, srun = project / "outputs/new_weinreb", project / "outputs/new_scnt"
sfull, sno = srun / "lr_seed42", srun / "no_interaction"
scnt["prepared_h5ad"] = srun / "preprocess/model_input_50pc.h5ad"
sexpression, sprior = srun / "preprocess/lr_expression.h5ad", srun / "edge_prior/manifest.json"
spca = srun / "preprocess/pca_artifacts.npz"
```

The raw cortical counts and expression-only CellChat reference still describe
the same measured cells. Steps 3–6 recalculate the model-dependent quantities.

## 3. Simulate the models and calculate distribution errors

These calls simulate 2,048 cells per model with score and growth, noise
`sigma=0.1`, integration step `0.05`, and 16-cell interaction groups. They
calculate weighted W1/W2 and absolute total-mass error at observed times,
using up to 1,024 points for OT. A new stochastic evaluation can differ from
the paper's saved values.

```python
from reproduction.nonspatial.weinreb_matched import summarize
repo = Path(cb.__file__).resolve().parents[1]
weinreb.update(summarize(
    repo / "reproduction/nonspatial/larry_evaluation/raw",
    output / "weinreb_three_model_summary"))
sdistribution = analyze.distribution(
    scnt["prepared_h5ad"], sfull, sno, output / "scnt_distribution",
    device=device, seed=42)
```

For S5, `summarize` averages 50 simulations within each trained model, then
the three models in each condition. The cortical call writes
`paired_distribution_metrics.csv` for S6c.

## 4. Calculate clone fate, RNA direction and dense trajectories

For S5d, each model simulates every day-2 cell to day 6. The terminal classifier
is uniform-weighted k-nearest neighbors (`k=20`) fitted on all measured day-6
cells. Clone-positive source cells are scored when their lineage also occurs
at day 6. The ten rollouts use seeds 0–9, step `0.1` and `sigma=0.1`.

The `summarize` call above averages fate probabilities across the three models
before calculating agreement with the 1,237 observed lineages. To repeat both
S5 evaluations from the six checkpoints, extract the S5 archive and run:

```python
from reproduction.nonspatial.weinreb_matched import evaluate
archive = Path("/path/to/larry_models")
weinreb.update(evaluate(archive, output / "weinreb_repeated", device=device))
```

For S6d, reconstruct the new-RNA direction from measured total/new counts,
the saved PCA transformation and the two-hour labeling interval. Evaluate
model fields on every measured cell with grouping seeds 101, 202, 303, 404
and 505. This reference is used for evaluation, not model fitting.

```python
scnt["direction"] = analyze.direction(
    sraw, scnt["prepared_h5ad"], spca,
    sfull, sno, output / "scnt_direction", device=device)
scnt["full_trajectory"] = analyze.trajectory(
    scnt["prepared_h5ad"], sfull, output / "scnt_trajectory", device=device)
```

The direction call writes `timewise_scnt_direction_alignment.csv` and cellwise
cosines. The trajectory NPZ contains 41 snapshots of the same 2,048 cells in
50 PCs, at intervals of `0.05` from time 0 to 2. S6b takes finite differences
and smooths PC1–PC2 vectors on its supported grid. The sparse distribution
sample file is a different output, not a replacement for this dense trajectory.

## 5. Calculate exact messages and LR attribution

Decompose each receiver's learned interaction into sender-type contributions.
Rebuild expression compatibility from the fitted prior's normalization and LR
database, then calculate `S_AB = D_AB × Q_AB`. Grouping replicates are averaged
within a model before averaging across training seeds.

```python
wattribution = analyze.attribution(
    wexpression, weinreb["prepared_h5ad"], wprior, lr_models,
    output / "weinreb_attribution", cell_type_key="Cell type annotation",
    training_seeds=(42, 43, 44), device=device)
sattribution = analyze.attribution(
    sexpression, scnt["prepared_h5ad"], sprior, [sfull],
    output / "scnt_attribution", cell_type_key="cell_type",
    training_seeds=(42,), device=device)
```

Outputs are `exact_message_summary.csv`, `cell_type_interaction_network.csv`
and `cell_type_pathway_scores.csv.gz`, plus reconstruction diagnostics and
per-training-seed summaries.

CellChat is a separate expression-only analysis. Its distributed
`cellchat_joined_directed_edges.csv` (Weinreb) and `cellchat_edge_summary.csv`
(cortex) remain the fixed observed reference here. GNN attribution does not
generate CellChat scores. The next step replaces the Weinreb table's old model
scores with new messages before recalculating concordance over all eligible
directed pairs.

## 6. Convert these outputs and draw the panels

The adapters handle the actual analysis schemas: long distribution tables
become paired PCA tables, and clone-fate JSON summaries become the three
displayed metric rows. If several inference repeats are present, distribution
metrics are averaged within each fitted arm and endpoint.

```python
from reproduction.nonspatial.adapters import (
    distribution_inputs, clone_fate_input, weinreb_cellchat_input,
)
from reproduction.nonspatial.fields import calculate_weinreb_fields
from reproduction.nonspatial.inputs import collect_nonspatial_inputs
from CytoBridge.results import calculate_nonspatial_panels, write_nonspatial_tables
from reproduction.paper_figures import draw_supplementary

scnt_metrics = distribution_inputs(sdistribution, output / "scnt_distribution_tables")
scnt["full_distribution"] = scnt_metrics["full_distribution"]
scnt["no_interaction_distribution"] = scnt_metrics["no_interaction_distribution"]
weinreb["cellchat_joined"] = weinreb_cellchat_input(
    wattribution / "exact_message_summary.csv", weinreb["cellchat_joined"],
    output / "weinreb_cellchat_comparison.csv")
weinreb["pathways"] = wattribution / "cell_type_pathway_scores.csv.gz"
scnt["exact_message_summary"] = sattribution / "exact_message_summary.csv"
scnt["network"] = sattribution / "cell_type_interaction_network.csv"
scnt["pathways"] = sattribution / "cell_type_pathway_scores.csv.gz"
weinreb["model_grids"] = calculate_weinreb_fields(
    weinreb["prepared_h5ad"], lr_models, output / "weinreb_fields", device=device)

data = collect_nonspatial_inputs(weinreb, scnt, output / "plot_inputs")
panels = calculate_nonspatial_panels(data)
write_nonspatial_tables(panels, output / "figures")
figures = draw_supplementary(
    [4, 5], output / "figures", results_dir=data.source_dir,
    data=data, panels=panels)
figures
```

The renderer receives the newly built data object and directory. All S5
Full-model panels now refer to the same LR ensemble. S6 is unchanged.
