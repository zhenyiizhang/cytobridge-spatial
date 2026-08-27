# Non-spatial workflows

CytoBridge includes two presets for expression-state dynamics without physical
spatial coordinates.

| Preset | Observed times | Model state | Additional evaluation data |
| --- | --- | --- | --- |
| `weinreb` | Day 2, 4, and 6 | 2,000 HVGs and 50 PCs | clone labels, cell types, and SPRING coordinates |
| `scnt_cortex` | 0, 15, 30, 60, and 120 minutes | total RNA, 2,000 HVGs, and 50 PCs | new-RNA and old-RNA layers plus cell type |

List the presets or inspect a plan without starting a run:

```bash
cytobridge nonspatial list-presets
cytobridge nonspatial plan --dataset weinreb --json
cytobridge nonspatial plan --dataset scnt_cortex --json
```

## Model settings

Interaction is calculated in the 50-dimensional expression state rather than
from physical neighborhoods. Cell type, clone identity, SPRING coordinates,
and the scNT new/old RNA layers are not included in PCA, prior fitting, or
dynamical-model training.

The `full` and `no_interaction` arms are trained separately with seed 42,
`sigma=0.1`, and the `velocity_score_cross_term` score-energy objective. The
`no_interaction` arm removes interaction construction and interaction force but
keeps the six-stage schedule and score/noise component.

Distribution evaluation propagates one source population from the earliest
observed time. Predicted growth masses are normalized for optimal-transport
metrics and remain unnormalized for total-mass variation.

## Prepare inputs and build the LR prior

```bash
dataset=weinreb
run_root=outputs/nonspatial/${dataset}
source_h5ad=inputs/${dataset}.h5ad

cytobridge nonspatial prepare \
  --dataset "$dataset" \
  --input-h5ad "$source_h5ad" \
  --output-dir "$run_root/preprocess"

cytobridge nonspatial build-prior \
  --dataset "$dataset" \
  --preprocess-manifest "$run_root/preprocess/preprocess_manifest.json" \
  --output-dir "$run_root/edge_prior" \
  --device cuda:0
```

For Weinreb, `X` is treated as library-normalized linear expression and is
transformed with `normalize_total(1e4)` followed by `log1p` before PCA. For
scNT, preparation checks `total == new + old` and builds the model state from
total RNA only.

Preparation writes:

- `model_input_50pc.h5ad` for training and simulation;
- `lr_expression.h5ad` for LR-supported prior construction;
- `pca_artifacts.npz`; and
- `preprocess_manifest.json`.

Prior construction uses the bundled mouse CellChatDB and writes its predictor
and `edge_prior/manifest.json`.

## Train the two arms

```bash
cytobridge nonspatial train \
  --dataset "$dataset" --arm full \
  --preprocess-manifest "$run_root/preprocess/preprocess_manifest.json" \
  --edge-prior-manifest "$run_root/edge_prior/manifest.json" \
  --output-dir "$run_root/full" --device cuda:0

cytobridge nonspatial train \
  --dataset "$dataset" --arm no_interaction \
  --preprocess-manifest "$run_root/preprocess/preprocess_manifest.json" \
  --output-dir "$run_root/no_interaction" --device cuda:0
```

The full arm requires an edge prior. The no-interaction arm does not take an
edge-prior argument. Each directory contains the model checkpoints, resolved
configuration, and training summary.

## Evaluate distributions

```bash
cytobridge nonspatial evaluate \
  --dataset "$dataset" \
  --prepared-h5ad "$run_root/preprocess/model_input_50pc.h5ad" \
  --full-run-dir "$run_root/full" \
  --no-interaction-run-dir "$run_root/no_interaction" \
  --output-dir "$run_root/evaluation" \
  --inference-seed 10000 --inference-seed 10001 \
  --device cuda:0
```

The evaluation directory contains weighted W1, weighted W2, and total-mass
variation tables for both arms.

For Weinreb, calculate clone-fate summaries from Day 2 to Day 6:

```bash
cytobridge nonspatial weinreb-clone-fate \
  --prepared-h5ad "$run_root/preprocess/model_input_50pc.h5ad" \
  --full-run-dir "$run_root/full" \
  --no-interaction-run-dir "$run_root/no_interaction" \
  --output-dir "$run_root/clone_fate" --device cuda:0
```

For scNT, calculate new-RNA direction summaries after both models are fit:

```bash
cytobridge nonspatial scnt-direction \
  --source-h5ad "$source_h5ad" \
  --prepared-h5ad "$run_root/preprocess/model_input_50pc.h5ad" \
  --pca-artifacts-npz "$run_root/preprocess/pca_artifacts.npz" \
  --full-run-dir "$run_root/full" \
  --no-interaction-run-dir "$run_root/no_interaction" \
  --output-dir "$run_root/scnt_direction" --device cuda:0
```

## Calculate interaction attribution

```bash
cytobridge nonspatial attribution \
  --dataset "$dataset" \
  --expression-h5ad "$run_root/preprocess/lr_expression.h5ad" \
  --latent-h5ad "$run_root/preprocess/model_input_50pc.h5ad" \
  --edge-prior-manifest "$run_root/edge_prior/manifest.json" \
  --training-run-dir "$run_root/full" \
  --output-dir "$run_root/attribution" --device cuda:0
```

Attribution tables contain model-message magnitudes grouped by time,
sender/receiver cell type, and LR-supported pathway. They describe the fitted
model and do not represent a perturbation assay or a direct ligand-flux
measurement.

## External communication tools

The optional shared-input comparison is implemented by:

```bash
python scripts/run_nonspatial_communication_consistency.py --help
Rscript scripts/run_nonspatial_nichenet.R --help
```

The workflow prepares a common LR universe, runs each supported external tool,
and writes method-specific tables before calculating rank and overlap
summaries. Raw score units remain separate by method.
