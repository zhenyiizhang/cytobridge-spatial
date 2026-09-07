---
orphan: true
---

# Analysis inputs: Supplementary Figure S6: classifier smoothing

S6 uses three calculations: observed-cell model-selection accuracy, composition
across nine generated population frames, and label transitions of 563 fixed
particles. The latter two populations are not interchangeable.

## 1. Train observed-cell classifiers and evaluate k

Run in the CytoBridge code folder after preparing each aligned H5AD. These fits train
only the downstream classifier, not the dynamics model. Use new output directories.

```bash
python scripts/select_classifier_spatial_k.py --dataset zebrafish \
  --h5ad data/zebrafish/aligned.h5ad --output outputs/s6/heldout/zebrafish --device cuda:2
python scripts/select_classifier_spatial_k.py --dataset mosta \
  --h5ad data/mosta/aligned.h5ad --output outputs/s6/heldout/mosta --device cuda:2
python scripts/select_classifier_spatial_k.py --dataset arista \
  --h5ad data/arista/aligned.h5ad --output outputs/s6/heldout/arista --device cuda:2
python scripts/select_classifier_spatial_k.py --dataset admouse \
  --h5ad data/admouse/aligned.h5ad --annotation-key major_annotation \
  --output outputs/s6/heldout/admouse --device cuda:2
python scripts/select_classifier_spatial_k.py --dataset chicken_heart \
  --h5ad data/chicken_heart/aligned.h5ad --annotation-key celltype_prediction \
  --output outputs/s6/heldout/chicken_heart --device cuda:2
```

Each command fixes the stratified 90/10 split at seed 42, keeps singleton labels
in training, fits a 128-hidden-unit ResidualMLP for 500 epochs, and selects the
best epoch by held-out balanced accuracy. Features are time, two aligned spatial
coordinates, and all 50 latent PCs. Voting uses predictions within each observed
time slice, for k = 1, 5, 10, 20, 50. The held-out rows also select the epoch:
these are model-selection scores, not an untouched test set.

Each output contains `classifier.pt`, `selection.json`, `k_metrics.csv`,
`per_time_metrics.csv`, and `heldout_inputs.npz`. The NPZ records exact
train/held-out row indices, truth, raw predictions, times, coordinates, and class
encoding. This supplies the concrete arrays that the former
`select_spatial_smoothing_k(...)` example left undefined.

## 2. Fit the separate trajectory classifier

S6b/c use time + spatial(2) + leading latent PCs(10), selected by accuracy.
Do not substitute an all-PC classifier from S6a or a full-data ablation classifier.

```bash
python scripts/run_classifier_smoothing_inputs.py trajectory-classifier \
  --h5ad data/zebrafish/aligned.h5ad \
  --output-dir outputs/s6/trajectory_classifier --device cuda:2
```

This writes `outputs/s6/trajectory_classifier/classifier.pt` with its complete
feature/training metadata and `training_manifest.json`. It uses 500 epochs,
hidden size 128, learning rate 0.001, seed 42, stratified 90/10 model selection,
and the accuracy-selected classifier without full-data refitting. To replay
the original figure's existing trajectories, use their original classifier
checkpoint instead; its labels are verified in step 4.

## 3. Generate both trajectory inputs

```bash
python scripts/run_classifier_smoothing_inputs.py generate \
  --h5ad data/zebrafish/aligned.h5ad --model-dir data/zebrafish/model \
  --classifier-cache outputs/s6/trajectory_classifier/classifier.pt \
  --output-dir outputs/s6/states --device cuda:2
```

The command reads the actual ordered t0 cells and requires 563 cells with 52
joint features. It preserves the S6 population definitions: a global-t0 growing
split simulation on a 0.1 output grid, dt/resampling interval 0.05, growth scale
1, sigma 0.03, no daughter noise, and a 100,000-particle ceiling; independently,
a non-split row-preserving simulation at t = 0, 1, 2, 3, 4 uses dt 0.01. Both use
the selected full model, seed 42, and interaction groups of 1024. There is no
display warp or intermediate observed restart.

Outputs are `outputs/s6/states/generated_frames/index.json` and nine NPZ frames,
`outputs/s6/states/fixed_cohort.npz`, and `generation_manifest.json`. The manifest
records the H5AD, dynamics/score checkpoints, classifier, and settings. These are
new simulations; changing the fitted model or classifier can change the curves.
The included paper tables are not replaced.

## 4. Calculate composition and fixed-cohort transition sensitivity

```bash
python scripts/run_classifier_smoothing_inputs.py evaluate \
  --state-index outputs/s6/states/generated_frames/index.json \
  --trajectory outputs/s6/states/fixed_cohort.npz \
  --classifier-cache outputs/s6/trajectory_classifier/classifier.pt \
  --output-dir outputs/s6/generated_evaluation
```

This recomputes all five votes and verifies each saved k=10 label. It requires
exact ordered t0 agreement between the population frames and fixed cohort.
It writes `frame_sensitivity.csv`, `transition_by_interval.csv`, and
`evaluation_manifest.json`. Passing existing trajectory paths reruns
classification/statistics but does not rerun simulation.

## 5. Collect and draw those results

```bash
python scripts/run_classifier_smoothing_inputs.py collect \
  --heldout-root outputs/s6/heldout \
  --generated-results outputs/s6/generated_evaluation \
  --output-dir outputs/s6/panel_data
CYTOBRIDGE_CLASSIFIER_SMOOTHING_RESULTS="$PWD/outputs/s6/panel_data" \
  python scripts/execute_paper_notebooks.py --notebook classifier_smoothing \
  --output-dir outputs/s6/notebook
```

The collector produces all six required S6 files from the supplied calculations,
records source hashes, and validates the figure contract. Formal downstream k
remains 10 for zebrafish/MOSTA/ARISTA and 1 for AD mouse/chicken heart. If a new
experiment selects a different accuracy-optimal k, validation stops instead of
mislabeling it with the paper's fixed selection annotations. The notebook keeps
the environment-selected directory through its final S6 plotting call.
Use an absolute results path, as above: the notebook runner changes the kernel's
working directory to its separate run folder.
