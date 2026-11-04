# Zebrafish loss-weight sensitivity

1. Write the four training files:

   ```bash
   python scripts/paper_figures/zebrafish_loss_weight/prepare_configs.py \
     --base-config <zebrafish-training.yaml> \
     --output-dir <loss-weight-run>/configs
   ```

2. Train each YAML with the same aligned H5AD and edge model used by the
   zebrafish dataset notebook. Run this command once for each setting:

   ```bash
   cytobridge workflow --config zebrafish --step train --train \
     --aligned-h5ad <run>/preprocess/zebrafish_aligned.h5ad \
     --training-config <loss-setting.yaml> \
     --edge-predictor-path <run>/preprocess/edge_classifier/zebrafish_edge_model.pt \
     --edge-predictor-threshold <value-written-by-preprocessing> \
     --output-dir <loss-setting-run> \
     --device cuda:0
   ```

3. Evaluate each trained model:

   ```bash
   python scripts/paper_figures/zebrafish_loss_weight/evaluate_model.py \
     --aligned-h5ad <run>/preprocess/zebrafish_aligned.h5ad \
     --training-dir <loss-weight-run>/<setting>/training \
     --condition <setting> \
     --output-dir <loss-weight-run>/evaluation/<setting> \
     --device cuda:0
   ```

4. Draw the comparison:

   ```bash
   python scripts/paper_figures/zebrafish_loss_weight/plot_figure.py \
     --alpha-metrics <alpha-expression-evaluation.csv> \
     --evaluation-root <loss-weight-run>/evaluation \
     --output-dir <loss-weight-run>/figure
   ```
