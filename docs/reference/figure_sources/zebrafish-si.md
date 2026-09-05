---
orphan: true
---

# Analysis inputs: Supplementary Figures S31–S38: zebrafish downstream panels

The [figure notebook](../../tutorials/paper_figures/zebrafish_si_s31_s38.ipynb) starts from saved numerical results or completed panels. This page records the calculations that precede it.

## Calculation programs

Each command lists the input it reads and the output passed to the next calculation. Replace a path in angle brackets with the location of that file on your computer.


### 1. calculate the zebrafish downstream results (S31-S35; S38)

```text
python -m scripts.run_zebrafish_paper_downstream --aligned-h5ad <saved-paper-root>/zebrafish/preprocess/zebrafish_aligned.h5ad --model-dir <saved-paper-root>/zebrafish/training --acceptance-report <saved-paper-root>/matched_ablation_acceptance.json --lr-database <zebrafish-lr.csv> --output-dir <paper-output> --stage all --device cuda
```

Start with: `aligned zebrafish H5AD; trained model; zebrafish LR database; matched_ablation_acceptance.json from the same run`

Writes: `observed and generated states; growth; virtual-removal arrays; gene-dynamics and inverse-PCA tables; one record for each completed analysis`

Next: `prepare the tables used by S31-S38`


Use matched_ablation_acceptance.json from the same model run.



### 2. prepare the loss-weight training files (S36)

```text
python scripts/paper_figures/zebrafish_loss_weight/prepare_configs.py --base-config <zebrafish-training.yaml> --output-dir <loss-config-dir>
```

Start with: `base zebrafish training YAML`

Writes: `one training YAML for each loss setting`

Next: `train one model from each YAML with the next command`




### 3. train one loss setting (S36)

```text
cytobridge workflow --config zebrafish --step train --train --aligned-h5ad <run>/preprocess/zebrafish_aligned.h5ad --training-config <loss-setting.yaml> --edge-predictor-path <run>/preprocess/edge_classifier/zebrafish_edge_model.pt --output-dir <loss-setting-run> --device cuda:0
```

Start with: `aligned zebrafish H5AD, one loss-setting YAML, and its matched edge model`

Writes: `<loss-setting-run>/training with model checkpoints and training history`

Next: `evaluate this trained model`


Run this command once for each YAML written in the preceding step. CytoBridge reads the fitted threshold from the edge model's matching .meta.json file; provide --edge-predictor-threshold only when that metadata file is unavailable.



### 4. evaluate each trained loss setting (S36)

```text
python scripts/paper_figures/zebrafish_loss_weight/evaluate_model.py --aligned-h5ad <run>/preprocess/zebrafish_aligned.h5ad --training-dir <loss-setting-run>/training --condition <condition-name> --output-dir <loss-evaluation-root>/<condition-name> --device cuda:0
```

Start with: `aligned H5AD and the trained model for each loss setting`

Writes: `evaluation tables for each setting`

Next: `collect the two expression-weight evaluations, then draw S36`




### 5. combine the expression-weight evaluations (S36)

```text
python scripts/paper_figures/zebrafish_loss_weight/collect_alpha_metrics.py --reference <loss-evaluation-root>/reference/distribution_metrics.csv --alternative <loss-evaluation-root>/alpha_expr_005/distribution_metrics.csv --output <loss-evaluation-root>/alpha_metrics.csv
```

Start with: `distribution_metrics.csv for expression weights 0.015 and 0.05`

Writes: `alpha_metrics.csv with model, time, space, and w1 columns`

Next: `draw S36`


Use reference, alpha_expr_005, ot_mass_10_to_1, and ot_mass_1_to_10 as the evaluation folder names. The alpha_expr_005 model uses alpha_express=0.05; the other settings are specified by prepare_configs.py.



### 6. draw loss-weight sensitivity (S36)

```text
python scripts/paper_figures/zebrafish_loss_weight/plot_figure.py --alpha-metrics <loss-evaluation-root>/alpha_metrics.csv --evaluation-root <loss-evaluation-root> --output-dir <loss-figure-dir>
```

Start with: `evaluation tables for all loss settings`

Writes: `s32_loss_weight_metrics.csv and the S36 PDF/PNG`




### 7. recalculate daughter-noise sensitivity (S37)

Follow [Daughter-cell perturbations: S37](../../tutorials/paper_figures/zebrafish_daughter_noise.md).
It gives the download, five simulation commands, and the plotting command.
Each simulation starts from the same observed time-zero population and
continues to time four. The figure uses the resulting composition and lineage
tables, not interval-by-interval simulations.




### 8. recalculate and draw all eight figures (S31-S38)

```text
python scripts/execute_paper_notebooks.py --notebook zebrafish_si_s31_s38 --output-dir <notebook-run>
```

Start with: `included NPZ arrays and CSV tables`

Writes: `Supplementary_Figure_S31.pdf/.png through Supplementary_Figure_S38.pdf/.png plus derived tables`




## Preparing inputs for the figure notebook

The model-analysis commands and the archived panel scripts are separate programs. The package does not yet provide one command that converts a new model run into all the inputs of this figure notebook. Use the saved paper inputs for its existing plotting cells.
