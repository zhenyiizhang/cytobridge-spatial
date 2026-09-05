---
orphan: true
---

# Analysis inputs: Supplementary Figures S4–S5: grouped non-spatial analyses

The [figure notebook](../../tutorials/paper_figures/nonspatial_figures.ipynb) starts from saved numerical results or completed panels. This page records the calculations that precede it.

## Calculation programs

Each command lists the input it reads and the output passed to the next calculation. Replace a path in angle brackets with the location of that file on your computer.


### 1. prepare model input (S4 Weinreb; S5 scNT)

```text
cytobridge nonspatial prepare --dataset <dataset> --input-h5ad <raw.h5ad> --output-dir <run>/preprocess
```

Start with: `raw expression H5AD and dataset configuration`

Writes: `<run>/preprocess/model_input_50pc.h5ad; lr_expression.h5ad; pca_artifacts.npz; preprocess_manifest.json`

Next: `build prior`


Run this command twice to reproduce both figures: use --dataset weinreb for S4 and --dataset scnt_cortex for S5. Each code block is one command.



### 2. build the LR edge prior (S4 Weinreb; S5 scNT)

```text
cytobridge nonspatial build-prior --dataset <dataset> --preprocess-manifest <run>/preprocess/preprocess_manifest.json --output-dir <run>/edge_prior --device cuda:0
```

Start with: `preprocess_manifest.json; lr_expression.h5ad; bundled mouse LR database`

Writes: `<run>/edge_prior predictor, graph inputs, and manifest.json`

Next: `train the Full model`




### 3. train the Full model (S4 Weinreb; S5 scNT)

```text
cytobridge nonspatial train --dataset <dataset> --arm full --preprocess-manifest <run>/preprocess/preprocess_manifest.json --edge-prior-manifest <run>/edge_prior/manifest.json --output-dir <run>/full --device cuda:0
```

Start with: `preprocess_manifest.json; edge-prior manifest; Full-arm configuration`

Writes: `<run>/full/model checkpoints; resolved configuration; training summary`

Next: `train the No-interaction model with the same settings`




### 4. train the No-interaction model with the same settings (S4 Weinreb; S5 scNT)

```text
cytobridge nonspatial train --dataset <dataset> --arm no_interaction --preprocess-manifest <run>/preprocess/preprocess_manifest.json --output-dir <run>/no_interaction --device cuda:0
```

Start with: `preprocess_manifest.json; No-interaction-arm configuration`

Writes: `<run>/no_interaction/model checkpoints; resolved configuration; training summary`

Next: `compare the two models`




### 5. compare the two trained models (S4c/S5c)

```text
cytobridge nonspatial evaluate --dataset <dataset> --prepared-h5ad <run>/preprocess/model_input_50pc.h5ad --full-run-dir <run>/full --no-interaction-run-dir <run>/no_interaction --output-dir <run>/evaluation --inference-seed 10000 --inference-seed 10001 --device cuda:0
```

Start with: `model_input_50pc.h5ad; Full and No-interaction model directories`

Writes: `distribution metrics and paired trajectory summaries`

Next: `dataset-specific evaluation`




### 6. evaluate Weinreb clone fate (S4d)

```text
cytobridge nonspatial weinreb-clone-fate --prepared-h5ad <run>/preprocess/model_input_50pc.h5ad --full-run-dir <run>/full --no-interaction-run-dir <run>/no_interaction --output-dir <run>/clone_fate --device cuda:0
```

Start with: `prepared lineage labels and both fitted arms`

Writes: `frozen_baseline_clone_fate_summary.csv and its run record`

Next: `assemble S4 panel data`




### 7. evaluate scNT new-RNA direction (S5d)

```text
cytobridge nonspatial scnt-direction --source-h5ad <raw.h5ad> --prepared-h5ad <run>/preprocess/model_input_50pc.h5ad --pca-artifacts-npz <run>/preprocess/pca_artifacts.npz --full-run-dir <run>/full --no-interaction-run-dir <run>/no_interaction --output-dir <run>/scnt_direction --device cuda:0
```

Start with: `reference scNT new-RNA direction and both fitted models`

Writes: `timewise_scnt_direction_alignment.csv and its run record`

Next: `assemble S5 panel data`




### 8. calculate interaction attribution (S4e-f; S5e-f)

```text
cytobridge nonspatial attribution --dataset <dataset> --expression-h5ad <run>/preprocess/lr_expression.h5ad --latent-h5ad <run>/preprocess/model_input_50pc.h5ad --edge-prior-manifest <run>/edge_prior/manifest.json --training-run-dir <run>/full --output-dir <run>/attribution --device cuda:0
```

Start with: `lr_expression.h5ad; model_input_50pc.h5ad; edge-prior record; Full-model directory`

Writes: `GNN message, network, CellChat-comparison, and pathway tables`

Next: `assemble panel data`




### 9. recalculate and draw from the included numerical files (S4/S5)

```text
python scripts/execute_paper_notebooks.py --notebook nonspatial_figures --output-dir <notebook-run>
```

Start with: `included numerical files under nonspatial_figures/*`

Writes: `<notebook-run>/nonspatial_figures/outputs/nonspatial_figures/supplementary_figure_s4_weinreb_nonspatial.pdf and supplementary_figure_s5_scnt_nonspatial.pdf, matching PNGs, and derived CSV tables`


The included numerical files reproduce the paper figure. They use the paper's saved Full checkpoint and the corrected No-interaction run; they are not the output of a new matched two-arm run. Steps 1–8 show the public route for producing both arms in a new run. The notebook recalculates the displayed values and draws new PDF and PNG files rather than loading finished figure pages.



## Preparing inputs for the figure notebook

The model-analysis commands and the archived panel scripts are separate programs. The package does not yet provide one command that converts a new model run into all the inputs of this figure notebook. Use the saved paper inputs for its existing plotting cells.
