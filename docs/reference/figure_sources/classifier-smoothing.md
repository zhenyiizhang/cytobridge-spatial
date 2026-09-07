---
orphan: true
---

# Classifier smoothing for Supplementary Figure S6

S6 compares classifier smoothing through observed-cell model-selection accuracy,
composition across nine generated population frames, and label transitions of
563 fixed particles. Population composition uses a growing simulation; label
transitions follow a separate fixed cohort.

## 1. Train observed-cell classifiers and evaluate k

Run the commands in the CytoBridge code folder after preparing each dataset's
`aligned.h5ad`. This step fits the downstream classifiers. The dynamics model is
loaded separately in step 3. Use new output directories for these calculations.

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
encoding. Step 5 collects `k_metrics.csv` and `selection.json` from each dataset.

## 2. Fit the separate trajectory classifier

S6b/c use a separate classifier with time, two spatial coordinates, and the
leading 10 latent PCs. Its checkpoint is selected by accuracy, whereas S6a uses
all 50 PCs and balanced accuracy.

```bash
python scripts/run_classifier_smoothing_inputs.py trajectory-classifier \
  --h5ad data/zebrafish/aligned.h5ad \
  --output-dir outputs/s6/trajectory_classifier --device cuda:2
```

This writes `outputs/s6/trajectory_classifier/classifier.pt` with its complete
feature/training metadata and `training_manifest.json`. It uses 500 epochs,
hidden size 128, learning rate 0.001, seed 42, stratified 90/10 model selection,
and the accuracy-selected checkpoint without full-data refitting. The following
simulation and evaluation commands both read this `classifier.pt`. If evaluating
previously generated trajectories, use the classifier that generated their labels.

## 3. Generate both trajectory inputs

The aligned zebrafish data and fitted model in `data/zebrafish/model` provide
the starting cells and dynamics. The classifier from step 2 labels the generated
states.

```bash
python scripts/run_classifier_smoothing_inputs.py generate \
  --h5ad data/zebrafish/aligned.h5ad --model-dir data/zebrafish/model \
  --classifier-cache outputs/s6/trajectory_classifier/classifier.pt \
  --output-dir outputs/s6/states --device cuda:2
```

The command reads the ordered t0 cells and requires 563 cells with 52
joint features. It preserves the S6 population definitions: a global-t0 growing
split simulation on a 0.1 output grid, dt/resampling interval 0.05, growth scale
1, sigma 0.03, no daughter noise, and a 100,000-particle ceiling; independently,
a non-split row-preserving simulation at t = 0, 1, 2, 3, 4 uses dt 0.01. Both use
the selected full model, seed 42, and interaction groups of 1024. There is no
display warp or intermediate observed restart.

Outputs are `outputs/s6/states/generated_frames/index.json` and nine NPZ frames,
`outputs/s6/states/fixed_cohort.npz`, and `generation_manifest.json`. The manifest
records the H5AD, dynamics/score checkpoints, classifier, and settings. Step 4
reads both the frame index and fixed cohort from this directory. Changing the
fitted model or classifier can change the resulting curves.

## 4. Calculate composition and fixed-cohort transition sensitivity

```bash
python scripts/run_classifier_smoothing_inputs.py evaluate \
  --state-index outputs/s6/states/generated_frames/index.json \
  --trajectory outputs/s6/states/fixed_cohort.npz \
  --classifier-cache outputs/s6/trajectory_classifier/classifier.pt \
  --output-dir outputs/s6/generated_evaluation
```

This applies all five voting settings to the saved states, checks the saved
k=10 labels, and requires the same ordered t0 cells in both simulations.
Population composition and fixed-cohort transitions are saved as
`outputs/s6/generated_evaluation/frame_sensitivity.csv` and
`transition_by_interval.csv`, alongside `evaluation_manifest.json`. With
previously generated trajectory paths, this step recalculates classification
and statistics only.

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

The collector combines the observed-cell results into `five_dataset_k_metrics.csv`
and `formal_k_policy.csv`, saves `arista_selection.json` and `heart_selection.json`,
and copies the two trajectory-evaluation CSVs into `outputs/s6/panel_data/`.
It also writes `manifest.json`.

S6's downstream settings are k=10 for zebrafish, MOSTA, and ARISTA, and k=1 for
AD mouse and chicken heart. The plotting input check stops if the selected k
differs from the figure's fixed selection annotations.

The notebook reads `outputs/s6/panel_data/` for both its calculations and final
S6 plot. Use the absolute results path shown above because the notebook runs
in a separate working directory. Its executed notebook and figure outputs are
saved under `outputs/s6/notebook/`.
